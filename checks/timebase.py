"""Sped-up recordings in MolmoAct2: does an episode play back faster than real time?

    python -m checks.timebase scan --raw RAW --out timebase.csv [--jobs 4]
    python -m checks.timebase apply --timebase timebase.csv [--labels RUN/out] EPISODES [EPISODES ...]
    python -m checks.timebase folder EPISODES [EPISODES ...]

MolmoAct2's files carry no real clock: timestamps are frame_index/30 and video pts are exactly k/30,
written by the recorder as if its loop always ran at 30 Hz. When a rig's loop ran slower, every sample
is still stamped 1/30 s apart, so both video and state play back sped up, and nothing in the metadata
shows it. The fixed physical delay of the follower arm behind the operator's leader arm is a real
clock: it is a constant number of real milliseconds, so it spans fewer frames in a compressed episode.
`action` is the leader (not a shifted copy of state: the best shift still leaves about 3x a typical
step of residual), `observation.state` the follower.

A slow recorder leaves two independent marks: it skips or repeats samples (sample_jitter) and it
shortens the follower lag in frames. Neither alone is clean (quick jerky motions also make
double-length steps; the lag is noisy per episode and varies by rig), so an episode is flagged only
when both hold. That trades a little recall for no false alarms: a flag should mean sped up.

The episode's own lag is noisy: an occasional real-time episode measures well under its neighbours
(episode 241, 3.20 frames, in a run of 3.5 to 4.0). A rig's speed holds for the episodes recorded next
to it, so the flag also needs the median lag of the SPEDUP_NEIGHBOURS episodes on either side (same
task, consecutive episode indices) to be at most SPEDUP_LAG_FRAMES. That can only remove flags.

Calibration on 41 episodes judged by watching the video (19 sped up, 22 real time, including episode
241): skipped+repeated >= 2.0% and lag <= 3.4 frames and neighbour median lag <= 3.4 flags 15 of the 19
and none of the 22. Without the neighbour term it flagged episode 241. Lag alone scored AUC 0.83 and
over-flagged; skipped+repeated alone AUC 0.93 with false alarms on jerky tasks.

The rule needs every episode's neighbours, so it runs in two steps over the whole dataset:

`scan` reads every data parquet of the dataset in RAW (state and action only, no video; it downloads the metadata
and the parquets it does not have, as `python -m prepare molmo` does), writes one row per episode to the CSV
(resuming a partial one) and prints the flagged share overall, by task and by session.

`apply` writes each prepared episode's neighbour lag into its context.json as "timebase_neighbour_lag_frames".
Labelling then measures the episode and reports the flag in dataset_checks["timebase"] (label/episode.py calls
timebase_check). With --labels it also re-applies the rule to label files already written (RUN/out/<episode>.json),
without a model call. The flag is a report field, never part of the prompt.

`folder` is for data with no dataset-wide scan (your own data, and Data Review's uploads): it measures the
neighbours inside each folder of prepared episodes (measure_folder). It is not part of `python -m checks`, so a
dataset that was scanned keeps its scan's neighbour lags.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from checks.results import stored_number
from label.atomic import write_atomic
from prepare.signal_alignment import recorder_arrival, RECORDER_ARRIVAL_LIMITATION
from prepare.frame_rate import episode_fps

JOINTS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]    # the 12 arm joints of the 14 state values (6 are grippers)

SPEDUP_JITTER_FRAC = 0.020
SPEDUP_LAG_FRAMES = 3.4
# the follower's delay is a time: 3.4 frames at the 30 Hz the rule was calibrated at, compared at each recording's rate
SPEDUP_LAG_S = SPEDUP_LAG_FRAMES / 30
SPEDUP_NEIGHBOURS = 3
# the longest follower delay searched for: 12 frames at the 30 Hz the rule was calibrated at, the same time in each
# recording's own frames (a real 133 ms delay is 16 frames at 120 fps)
MAX_LAG_S = 12 / 30
# the arm moves when its joints step at least this fast (0.3 degrees per frame at the 30 Hz the rule was calibrated
# at, in radians per second)
MOVING_RAD_S = np.radians(0.3) * 30
SPEDUP_RULE = (f"skipped+repeated >= {SPEDUP_JITTER_FRAC:.3f} and follower lag <= {SPEDUP_LAG_FRAMES} frames "
               f"and median lag of the {SPEDUP_NEIGHBOURS} episodes either side <= {SPEDUP_LAG_FRAMES} frames")


def rule_text(fps: float = 30.0) -> str:
    """The rule as it applies at fps: SPEDUP_RULE at 30, the same delay in this recording's frames otherwise."""
    if fps == 30:
        return SPEDUP_RULE
    n = round(SPEDUP_LAG_S * fps, 2)
    return (f"skipped+repeated >= {SPEDUP_JITTER_FRAC:.3f} and follower lag <= {n:g} frames at {fps:g} fps "
            f"({SPEDUP_LAG_S * 1000:.0f} ms, {SPEDUP_LAG_FRAMES} frames at 30 fps) and median lag of the "
            f"{SPEDUP_NEIGHBOURS} episodes either side <= {n:g} frames")


def is_sped_up(lag, skipped_frac, repeated_frac, neighbour_lag, fps: float = 30.0) -> bool:
    """The sped-up rule, the lags in frames at fps. False when any input is missing: no neighbour lag means the
    episode was not scanned with its neighbours (`scan`, then `apply`), and an unscanned episode is never flagged."""
    lag, sk, rp, nb = (stored_number(v) for v in (lag, skipped_frac, repeated_frac, neighbour_lag))
    if None in (lag, sk, rp, nb):
        return False
    if fps == 30:
        return (sk + rp) >= SPEDUP_JITTER_FRAC and lag <= SPEDUP_LAG_FRAMES and nb <= SPEDUP_LAG_FRAMES
    return (sk + rp) >= SPEDUP_JITTER_FRAC and lag / fps <= SPEDUP_LAG_S and nb / fps <= SPEDUP_LAG_S


def neighbour_lags(episode_index, task, lag):
    """Median follower lag of the SPEDUP_NEIGHBOURS episodes on either side of each episode (itself
    included), within runs of consecutive episode indices of the same task. Inputs are sequences of
    equal length; returns a list of floats (NaN where no neighbour has a lag)."""
    import pandas as pd
    d = pd.DataFrame({"e": list(episode_index), "t": list(task), "l": pd.to_numeric(list(lag), errors="coerce")})
    # runs are consecutive indices of one task, so each task's episodes are ordered by index on their own: two
    # datasets or collections in one folder repeat the same indices, and sorted by index alone they interleave
    order = d.sort_values(["t", "e"], kind="stable").index
    s = d.loc[order]
    run = ((s.e.diff() != 1) | (s.t != s.t.shift())).cumsum()
    w = 2 * SPEDUP_NEIGHBOURS + 1
    med = s.groupby(run).l.transform(lambda x: x.rolling(w, center=True, min_periods=1).median())
    return d.index.map(med.to_dict()).tolist()


def follower_lag_frames(state: np.ndarray, action: np.ndarray, fps: float = 30.0,
                        max_lag_s: float = MAX_LAG_S) -> float | None:
    """Median over the 12 arm joints of how many frames the follower (state) trails the leader
    (action), from the peak of the velocity cross-correlation with a parabolic sub-frame fit. Lags up to
    max_lag_s are searched, in this recording's frames at fps: the delay is a time, so at 120 fps it spans four
    times the frames it does at 30. A step with a missing or infinite reading on either side is left out of the
    correlation rather than spreading through it, and a joint that never moves (or never reads) is skipped. So are the
    recorder's timing faults (sample_jitter): a skipped or repeated sample is one step in both streams that is not
    one frame interval of motion, so both streams jump together there, and faults that come at a regular period
    would otherwise add a peak at that period."""
    s = np.asarray(state, dtype=np.float64)
    a = np.asarray(action, dtype=np.float64)
    fps = stored_number(fps)
    if fps is None or not fps > 0:
        return None
    max_lag = max(2, int(np.ceil(max_lag_s * fps)))
    if s.ndim != 2 or s.shape != a.shape or s.shape[1] <= max(JOINTS) or len(s) < 4 * max_lag:
        return None
    fault = np.zeros(len(s) - 1, dtype=bool)
    r = _step_ratios(s, fps)
    if r is not None:
        fault[1:-1] = np.logical_or(*_faults(r))
    lags = []
    for j in JOINTS:
        x = np.diff(a[:, j])
        y = np.diff(s[:, j])
        okx, oky = np.isfinite(x) & ~fault, np.isfinite(y) & ~fault
        # a NaN standard deviation never passes a "below" test, so the guard is written as what must hold
        if not (okx.sum() > 2 and oky.sum() > 2 and x[okx].std() >= 1e-5 and y[oky].std() >= 1e-5):
            continue
        x = np.where(okx, (x - x[okx].mean()) / x[okx].std(), 0.0)
        y = np.where(oky, (y - y[oky].mean()) / y[oky].std(), 0.0)
        n = len(x)
        # the mean product over the pairs with a reading on both sides, at each lag
        pairs = np.array([np.count_nonzero(okx[:n - k] & oky[k:]) for k in range(max_lag + 1)])
        if (pairs < n // 2).any():
            continue
        cc = np.array([np.dot(x[:n - k], y[k:]) for k in range(max_lag + 1)]) / pairs
        k = int(np.argmax(cc))
        if 0 < k < max_lag:
            d = cc[k - 1] - 2 * cc[k] + cc[k + 1]
            if d != 0:
                k = k + 0.5 * (cc[k - 1] - cc[k + 1]) / d
        lags.append(float(k))
    return round(float(np.median(lags)), 3) if lags else None


def _step_ratios(state: np.ndarray, fps: float = 30.0) -> np.ndarray | None:
    """Each state step over the mean of its two neighbours (the 12 arm joints), NaN where those neighbours move less
    than MOVING_RAD_S or a reading is missing; None when the state has too few steps. Entry i is the step between
    frames i + 1 and i + 2."""
    s = np.asarray(state, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] <= max(JOINTS):
        return None
    v = np.linalg.norm(np.diff(s[:, JOINTS], axis=0), axis=1)
    if len(v) < 3:
        return None
    nb = (v[:-2] + v[2:]) / 2
    mid = v[1:-1]
    # moving is a speed, the same per second at any frame rate, so the step it takes per frame shrinks as the rate
    # rises. A step next to a missing or infinite reading has no length, so it is neither a skip nor a repeat
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = np.isfinite(nb) & np.isfinite(mid) & (nb > MOVING_RAD_S / fps)
        return np.where(ok, mid / np.where(ok, nb, 1.0), np.nan)


def _faults(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(skipped, repeated) steps among step ratios r (_step_ratios)."""
    with np.errstate(invalid="ignore"):
        return (r > 1.8) & (r < 2.2), r < 0.15


def sample_jitter(state: np.ndarray, fps: float = 30.0) -> dict:
    """Recorder timing faults in the state stream, during motion: single steps ~2x both neighbours
    (a skipped sample stamped as one 1/30 s step) and near-zero steps between moving neighbours (a
    repeated sample). Fractions of moving frames."""
    fps = stored_number(fps)
    r = _step_ratios(state, fps) if fps and fps > 0 else None
    if r is None:
        return {"skipped_frac": None, "repeated_frac": None}
    r = r[np.isfinite(r)]
    if not len(r):
        return {"skipped_frac": 0.0, "repeated_frac": 0.0}
    skipped, repeated = _faults(r)
    return {"skipped_frac": round(float(skipped.mean()), 4), "repeated_frac": round(float(repeated.mean()), 4)}


def arrival_order_only(measurements: dict) -> dict:
    """Retain recorded sequence measurements without interpreting transport as a physical clock."""
    return {**measurements,
            'recorded_order_lag_frames': measurements.get('recorded_order_lag_frames',
                                                          measurements.get('follower_lag_frames')),
            'follower_lag_frames': None, 'neighbour_lag_frames': None, 'sped_up_recording': False,
            'timing_basis': 'mcap_log_time', 'status': 'not evaluated',
            'reason': RECORDER_ARRIVAL_LIMITATION,
            'not_assessed': RECORDER_ARRIVAL_LIMITATION,
            'rule': 'Physical follower-delay rule not evaluated on recorder arrival timestamps.',
            'measurement_note': 'Lag and sample-step ratios describe recorded row order, not physical delay or acquisition timing.'}


def timebase_check(state: np.ndarray, action: np.ndarray, neighbour_lag: float | None = None,
                   fps: float = 30.0, *, context: dict | None = None) -> dict:
    """One episode's measurements and flag, at the recording's frame rate. neighbour_lag comes from the dataset scan
    (via context.json, written by `apply`); without it the episode is measured but not flagged."""
    lag = follower_lag_frames(state, action, fps)
    j = sample_jitter(state, fps)
    result = {"follower_lag_frames": lag, **j, "neighbour_lag_frames": stored_number(neighbour_lag),
            "sped_up_recording": is_sped_up(lag, j["skipped_frac"], j["repeated_frac"], neighbour_lag, fps),
            "rule": rule_text(fps)}
    return arrival_order_only(result) if recorder_arrival(context or {}) else result


def scan_file(raw: Path, chunk: int, file: int) -> list[dict]:
    """timebase_check of every episode in one data parquet, downloaded into RAW if it is not there."""
    import pandas as pd
    from prepare import molmo
    p = molmo.ensure_data_parquet(raw, chunk, file)
    d = pd.read_parquet(p, columns=["observation.state", "action", "episode_index", "frame_index"])
    rows = []
    for e, g in d.groupby("episode_index"):
        g = g.sort_values("frame_index")
        s = np.stack(g["observation.state"].to_numpy()).astype(np.float64)
        a = np.stack(g["action"].to_numpy()).astype(np.float64)
        rows.append({"episode_index": int(e), "frames": len(g), "data_chunk": chunk, "data_file": file,
                     **timebase_check(s, a)})
    return rows


def cmd_scan(args) -> int:
    import pandas as pd
    from prepare import molmo
    meta = molmo.ensure_meta(args.raw)
    ep = pd.concat([pd.read_parquet(p) for p in sorted((meta / "episodes").glob("chunk-*/*.parquet"))],
                   ignore_index=True)
    ep["task"] = ep["tasks"].map(lambda v: str(v[0]))
    files = sorted({(int(c), int(f)) for c, f in zip(ep["data/chunk_index"], ep["data/file_index"])})
    done = pd.read_csv(args.out) if args.out.exists() else pd.DataFrame(columns=["data_chunk", "data_file"])
    have = {(int(c), int(f)) for c, f in zip(done["data_chunk"], done["data_file"])}
    todo = [cf for cf in files if cf not in have]
    print(f"{len(files)} data files, {len(have)} already scanned, {len(todo)} to go", flush=True)
    rows = done.to_dict("records")
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(scan_file, args.raw, c, f): (c, f) for c, f in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                rows += fu.result()
            except Exception as e:  # noqa: BLE001 - a file that fails is printed and left out of the CSV, so a rerun scans it again
                print(f"FAIL {futs[fu]}: {e}", file=sys.stderr, flush=True)
            if i % 25 == 0 or i == len(todo):
                pd.DataFrame(rows).to_csv(args.out, index=False)
                print(f"{i}/{len(todo)} files", flush=True)

    x = pd.DataFrame(rows)
    # a resumed CSV already has the columns added below; they are recomputed over every episode
    x = x.drop(columns=[c for c in ("task", "neighbour_lag_frames", "rule") if c in x.columns])
    x = x.merge(ep[["episode_index", "task"]], on="episode_index", how="left")
    # the rule needs each episode's neighbours, so it is applied once every measurement is in
    x["neighbour_lag_frames"] = neighbour_lags(x.episode_index, x.task, x.follower_lag_frames)
    x["sped_up_recording"] = [is_sped_up(r.follower_lag_frames, r.skipped_frac, r.repeated_frac,
                                         r.neighbour_lag_frames) for r in x.itertuples()]
    x.to_csv(args.out, index=False)
    sp = x["sped_up_recording"].astype(bool)
    hours = x["frames"] / 30 / 3600
    print(f"\n{len(x)} episodes, {hours.sum():.1f} h: sped-up recording {int(sp.sum())} episodes "
          f"({sp.mean():.1%}), {hours[sp].sum():.1f} h ({hours[sp].sum() / hours.sum():.1%} of hours) "
          f"({SPEDUP_RULE})")
    g = x.groupby("task").agg(eps=("episode_index", "size"), sped_up=("sped_up_recording", "mean"),
                              lag=("follower_lag_frames", "median")).sort_values("sped_up", ascending=False)
    print(g.round(3).to_string())
    x = x.sort_values("episode_index")
    x["sess"] = ((x["episode_index"].diff() != 1) | (x["task"] != x["task"].shift())).cumsum()
    s = x.groupby("sess").agg(task=("task", "first"), ep0=("episode_index", "min"), ep1=("episode_index", "max"),
                              n=("episode_index", "size"), lag=("follower_lag_frames", "median"),
                              sped_up=("sped_up_recording", "mean"))
    s = s[s.n >= 5].sort_values("lag")
    print("\nmost compressed sessions (>=5 consecutive episodes):")
    print(s.head(20).round(3).to_string())
    return 0


def measure_folder(eps: Path) -> int:
    """The neighbour lag measured inside one folder of prepared episodes, for data with no dataset-wide scan (your
    own data, Data Review's uploads). The folder holds its own neighbours: the lag is measured on every episode with
    joint state and leader actions, and an episode gets a neighbour lag only when at least 3 episodes of its run fall
    in the window (itself and two more), so a lone episode is measured, never flagged. Returns how many episodes
    got one; every episode also records how many neighbours it had (timebase_neighbours_in_upload)."""
    rows = []
    from prepare.state_contract import action_semantics
    for d in sorted(Path(eps).glob("episode_*")):
        ctx = json.loads((d / "context.json").read_text())
        if recorder_arrival(ctx):
            before = dict(ctx)
            ctx.pop('timebase_neighbour_lag_frames', None)
            ctx['timebase_neighbours_in_upload'] = 0
            if ctx != before:
                write_atomic(d / 'context.json', ctx, indent=1, default=str)
            continue
        if ctx.get("profile") != "teleop_arms" or ctx.get("state_kind") != "joints" or not (d / "state.npz").exists():
            continue
        if action_semantics(ctx)['role'] != 'command':
            continue
        z = np.load(d / "state.npz")
        if "action" not in z.files or z["state"].shape[1] != 14:
            continue
        lag = follower_lag_frames(z["state"], z["action"], float(episode_fps(ctx, ep_dir=d)[0]))
        # the generic LeRobot reader keeps the episode's index at the top of its context, a dataset's own adapter
        # (Galaxea, HABIT) in its source; either is the same number from the dataset's metadata
        src = ctx.get("source") or {}
        idx = ctx.get("episode_index", src.get("episode_index"))
        if idx is None:
            continue
        rows.append((d, int(idx), "; ".join(ctx.get("task_label") or []) + "|" +
                     str(src.get("dataset_folder") or src.get("folder") or ""), lag))
    if not rows:
        return 0
    med = neighbour_lags([r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows])
    n = 0
    for (d, e, task, _), m in zip(rows, med):
        window = [r for r in rows if r[2] == task and abs(r[1] - e) <= SPEDUP_NEIGHBOURS and r[3] is not None]
        # only consecutive indices count as neighbours: the run must be unbroken between them
        run = {r[1] for r in window}
        near = [x for x in run if all(y in run for y in range(min(x, e), max(x, e) + 1))]
        p = d / "context.json"
        ctx = json.loads(p.read_text())
        before = dict(ctx)
        if len(near) >= 3 and m == m:
            ctx["timebase_neighbour_lag_frames"] = round(float(m), 3)
            n += 1
        else:
            ctx.pop("timebase_neighbour_lag_frames", None)
        ctx["timebase_neighbours_in_upload"] = len(near)
        if ctx != before:
            write_atomic(p, ctx, indent=1, default=str)
    return n


def cmd_folder(args) -> int:
    for root in args.roots:
        print(f"{root}: neighbour lag measured on {measure_folder(root)} episodes", flush=True)
    return 0


def cmd_apply(args) -> int:
    import pandas as pd
    tb = pd.read_csv(args.timebase).set_index("episode_index")
    n_ctx = n_lab = changed = 0
    folders = {}
    for root in args.roots:
        for d in sorted(root.glob("episode_*")):
            folders[d.name] = d
            e = int(d.name.split("_")[1])
            ctx = json.loads((d / "context.json").read_text())
            if recorder_arrival(ctx):
                ctx.pop('timebase_neighbour_lag_frames', None)
                ctx['timebase_neighbours_in_upload'] = 0
            else:
                ctx["timebase_neighbour_lag_frames"] = stored_number(tb.loc[e, "neighbour_lag_frames"])
            write_atomic(d / "context.json", ctx)
            n_ctx += 1
    for p in sorted(args.labels.glob("episode_*.json")) if args.labels else []:
        r = json.loads(p.read_text())
        t = (r.get("dataset_checks") or {}).get("timebase")
        if not t:
            continue
        # the label's own episode folder, else the one of the same name among EPISODES; never a path made from
        # nothing, which would read a context.json in whatever folder the command runs from
        ep_dir = Path(r["episode_dir"]) if r.get("episode_dir") else folders.get(p.stem)
        if ep_dir is None:
            print(f"{p.name}: no episode folder (no episode_dir, and none of that name in EPISODES), left as it is",
                  file=sys.stderr)
            continue
        context_path = ep_dir / 'context.json'
        context = json.loads(context_path.read_text()) if context_path.is_file() else {}
        if recorder_arrival(context) or t.get('timing_basis') == 'mcap_log_time':
            changed += bool(t.get('sped_up_recording'))
            r['dataset_checks']['timebase'] = arrival_order_only(t)
            write_atomic(p, r)
            n_lab += 1
            continue
        e = int(ep_dir.name.split("_")[1])
        nb = stored_number(tb.loc[e, "neighbour_lag_frames"])
        flag = is_sped_up(t.get("follower_lag_frames"), t.get("skipped_frac"), t.get("repeated_frac"), nb)
        if flag != bool(tb.loc[e, "sped_up_recording"]):
            raise SystemExit(f"{p.name}: label measurements disagree with the scan "
                             f"({flag} vs {tb.loc[e, 'sped_up_recording']})")
        changed += flag != bool(t.get("sped_up_recording"))
        t.update({"neighbour_lag_frames": nb, "sped_up_recording": flag, "rule": SPEDUP_RULE})
        write_atomic(p, r)
        n_lab += 1
    print(f"context.json updated {n_ctx}; label files updated {n_lab}, flag changed on {changed}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m checks.timebase", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="measure every episode of the dataset from its data parquets")
    s.add_argument("--raw", type=Path, default=Path("data/raw/molmo"),
                   help="the dataset's download folder (meta/ and data/), as for python -m prepare molmo")
    s.add_argument("--out", type=Path, required=True, help="the CSV to write, one row per episode")
    s.add_argument("--jobs", type=int, default=4, help="data parquets read in parallel")
    s.set_defaults(func=cmd_scan)
    a = sub.add_parser("apply", help="carry the scan into prepared episodes' context.json (and label files)")
    a.add_argument("roots", nargs="+", type=Path, metavar="EPISODES", help="folders of prepared episode_* folders")
    a.add_argument("--timebase", type=Path, required=True, help="the CSV `scan` wrote")
    a.add_argument("--labels", type=Path, default=None,
                   help="a run's out/ folder: re-apply the rule to the label files already written there")
    a.set_defaults(func=cmd_apply)
    f = sub.add_parser("folder", help="measure the neighbour lag inside each folder of prepared episodes")
    f.add_argument("roots", nargs="+", type=Path, metavar="EPISODES", help="folders of prepared episode_* folders")
    f.set_defaults(func=cmd_folder)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
