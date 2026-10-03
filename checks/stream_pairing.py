"""Three deterministic checks on the recorded state and the mounted cameras, one per mode:

    python -m checks.stream_pairing [--jobs N] [--force] EPISODES [EPISODES ...]              stream_pairing
    python -m checks.stream_pairing --jumps [--jobs N] [--force] EPISODES [EPISODES ...]      recorded_jumps
    python -m checks.stream_pairing --grippers [--jobs N] [--force] EPISODES [EPISODES ...]   gripper_channels

EPISODES are folders of prepared episode_* folders. Each mode reads the episode's context.json, sources.json,
state.npz, its real-times file and kmap_*.npy files if the sources name them, and (stream_pairing, recorded_jumps)
the mounted cameras' videos, and writes its result into context.json under the key named above; an episode that
already has that key is skipped unless --force. `python -m board build` copies all three into the episode's
dataset_checks on the board (the labelling harness also reports stream_pairing). None of it goes into the prompt.
`python -m checks EPISODES` runs all three and the capture checks.

stream_pairing: a camera mounted on a gripper moves with it, so each mounted stream's frame-to-frame image
change should follow its own gripper's recorded motion and not the other one's. For every episode with a left
and a right mounted stream and two recorded actors this correlates each stream's image change with each actor's
recorded speed (joint speed for teleop arms, translation plus rotation for handheld grippers) and reports all
four correlations. `crossed` is true when both streams follow the opposite actor better than their own. That
says the stream names and the state channels disagree; it does not say which one is mislabelled (a person
checks that from the pixels, for example where the other gripper appears).

recorded_jumps: single frames where an actor's recorded motion leaps far beyond its own typical step while that
actor's own mounted camera shows no matching jump (a real fast move shifts the camera rigidly mounted on it).
Candidates come from the state alone; only a window of the actor's camera around each one is decoded (3 frames
either side). Running it on every episode means finding such a leap never depends on a model noticing a number
in a table.

gripper_channels: a gripper whose recorded value is exactly the same at every frame of the episode. That is
either a gripper that was never used or a gripper whose sensor did not record; which one is decided on the
frames. It needs the state only.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from label import episode as me
from label import frames as mf
from label import state as ms

SMALL_W = 64          # image change is measured on a 64-px-wide grey copy of each frame


def stream_motion(ep: dict, v: str) -> np.ndarray:
    """Mean absolute grey-level change between consecutive frames of this episode's own window of
    stream v, one value per anchor interval (len = anchor frames - 1). Frames are matched by exact
    pts like frames.extract_frames; a stream paired to the anchor by real time (kmap) is
    summed over its own frames between consecutive anchor frames."""
    import av
    s = ep["sources"][v]
    n = int(s["n_frames"])
    pts = ep["times"].get(f"{v}_pts") if ep.get("times") is not None else None
    fps = me.ep_fps(ep)
    own = np.zeros(max(n - 1, 0))
    with av.open(str(s["packed"])) as c:
        st = c.streams.video[0]
        st.codec_context.thread_count = 1
        if pts is not None:
            targets = [int(p) for p in pts]
        else:
            b0, step = mf.base_frame(s["base_s"], fps), mf.frame_pts_step(st.time_base, fps)
            targets = [(b0 + k) * step for k in range(n)]
        index = {p: k for k, p in enumerate(targets)}
        c.seek(targets[0], stream=st, backward=True, any_frame=False)
        prev, prev_k = None, None
        for fr in c.decode(st):
            if fr.pts is None or fr.pts < targets[0]:
                continue
            k = index.get(fr.pts)
            if k is None:
                if fr.pts > targets[-1]:
                    break
                continue
            h = max(2, int(round(fr.height * SMALL_W / fr.width)))
            g = fr.reformat(width=SMALL_W, height=h, format="gray").to_ndarray().astype(np.int16)
            if prev is not None and k == prev_k + 1:
                own[prev_k] = np.abs(g - prev).mean()
            prev, prev_k = g, k
            if k == n - 1:
                break
    km = ep["kmap"].get(v)
    if km is None:
        return own
    cum = np.concatenate([[0.0], np.cumsum(own)])
    km = np.clip(np.asarray(km, dtype=int), 0, n - 1)
    return cum[km[1:]] - cum[km[:-1]]


def actor_speed(ep: dict, g: int) -> np.ndarray:
    """Recorded speed of actor g per anchor interval: summed joint change (deg) for joints, or
    translation (cm) plus rotation (deg) for an end-effector pose."""
    s = np.asarray(ep["state"], dtype=np.float64)
    o = 7 * g
    if me.state_kind(ep) == "joints":
        return np.degrees(np.abs(np.diff(s[:, o:o + 6], axis=0)).sum(axis=1))
    tr = np.linalg.norm(np.diff(s[:, o:o + 3], axis=0), axis=1) * 100
    R = ms._rot_zyx(s[:, o + 3:o + 6])
    c = (np.einsum("tij,tij->t", R[:-1], R[1:]) - 1) / 2
    return tr + np.degrees(np.arccos(np.clip(c, -1, 1)))


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if n < 3 or a.std() < 1e-9 or b.std() < 1e-9:
        return None
    return round(float(np.corrcoef(a, b)[0, 1]), 3)


def unaligned(ep: dict) -> dict | None:
    """{"not_assessed": why} for an episode whose recorded state is not on its cameras' frames (context.json
    state_unaligned: the camera it was recorded on was taken out, board/clips.py drop_cameras), so a check that compares
    the state with the video is not run on it; None otherwise."""
    why = ep["context"].get("state_unaligned")
    return {"not_assessed": f"The recorded state is not on these cameras' frames. {why}"} if why else None


def pairing(ep_dir: Path) -> dict | None:
    """context["stream_pairing"]: the four stream-vs-actor correlations and crossed (None when a correlation is
    undefined), or None for an episode without left and right mounted streams and two recorded actors. Measured over
    the frames the state covers (label/episode.py state_span), and not assessed when the state is not on the cameras'
    frames (unaligned)."""
    ep = me.load(ep_dir)
    if not {"left", "right"} <= set(me.views(ep)) or ep["state"].shape[1] < 14:
        return None
    if unaligned(ep):
        return unaligned(ep)
    a, b = me.state_span(ep)
    mL, mR = stream_motion(ep, "left"), stream_motion(ep, "right")
    vL, vR = actor_speed(ep, 0), actor_speed(ep, 1)
    n = min(len(mL), len(mR), b - 1)
    span = slice(a, n)
    r = {"left_vs_left": _corr(mL[span], vL[span]), "right_vs_right": _corr(mR[span], vR[span]),
         "left_vs_right": _corr(mL[span], vR[span]), "right_vs_left": _corr(mR[span], vL[span])}
    if any(x is None for x in r.values()):
        r["crossed"] = None
    else:
        r["crossed"] = min(r["left_vs_right"], r["right_vs_left"]) > max(r["left_vs_left"], r["right_vs_right"])
    r["rule"] = "crossed: both streams follow the opposite actor's recorded speed better than either follows its own"
    return r


JUMP_FACTOR = 5.0          # a candidate step is this many times the actor's own 95th-percentile step
# and at least this big (cm, or deg of the largest joint), per typical frame interval
JUMP_FLOOR = {"ee_pose": 3.0, "joints": 5.0}
VISUAL_JUMP = 2.0          # the camera's change at the step must exceed this many times its window median
ISOLATION = 3.0            # a leap, not fast motion: the step is at least this many times the steps on either side


def _steps(ep: dict, g: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame steps rescaled to one typical frame interval, and the raw intervals.

    A step across a recording gap (frames missing, e.g. 0.15 s between anchors at 30 fps) is the
    motion of several frames, not a leap, so each step is scaled by typical_dt / max(dt, typical_dt).
    """
    s = np.asarray(ep["state"], dtype=np.float64)
    o = 7 * g
    if me.state_kind(ep) == "joints":
        raw = np.degrees(np.abs(np.diff(s[:, o:o + 6], axis=0)).max(axis=1))
    else:
        raw = np.linalg.norm(np.diff(s[:, o:o + 3], axis=0), axis=1) * 100
    ts = np.array([me.frame_time(ep, k) for k in range(len(s))], dtype=np.float64)
    dt = np.diff(ts)
    pos = dt[dt > 1e-4]
    ref = float(np.median(pos)) if len(pos) else 1.0
    return raw * ref / np.maximum(dt, ref), dt


def jumps(ep_dir: Path) -> dict | None:
    """context["recorded_jumps"]: up to 3 candidate leaps per actor, each with its camera's image change when the
    actor has a mounted camera, and flagged when a leap has no matching image change; None on video-only rigs. Only
    the steps inside the frames the state covers count (label/episode.py state_span), and it is not assessed when the
    state is not on the cameras' frames (unaligned)."""
    ep = me.load(ep_dir)
    kind = me.state_kind(ep)
    if kind == "none":
        return None                         # nothing recorded to jump
    if unaligned(ep):
        return unaligned(ep)
    sa, sb = me.state_span(ep)
    unit = "cm" if kind == "ee_pose" else "deg"
    vs = set(me.views(ep))
    # an actor is named by its camera's view key (left, right) on two-gripper rigs, but by the camera's display
    # name on single-gripper ones ("gripper"), so map display names back to view keys or the camera is never checked
    by_name = {me.cam_name(ep, v): v for v in vs}
    events = []
    for g, name in enumerate(me.actors(ep)):
        cam = name if name in vs else by_name.get(name)
        st, dts = _steps(ep, g)
        inside = np.zeros(len(st), dtype=bool)
        inside[sa:max(sa, sb - 1)] = True
        st = np.where(inside & np.isfinite(st), st, 0.0)
        if inside.sum() < 10:
            continue
        p95 = float(np.percentile(st[inside], 95))
        thr = max(JUMP_FLOOR[kind], JUMP_FACTOR * p95)

        def isolated(i: int) -> bool:
            nb = [st[j] for j in (i - 1, i + 1) if 0 <= j < len(st)]
            return not nb or st[i] >= ISOLATION * max(nb)
        cand = [int(i) for i in np.argsort(st)[::-1][:6] if st[i] >= thr and isolated(int(i))][:3]
        for i in cand:                        # step between anchor frames i and i+1
            ev = {"actor": name, "t_s": round(me.frame_time(ep, i + 1), 3), "step": round(float(st[i]), 2),
                  "unit": unit, "typical_p95": round(p95, 3), "dt_s": round(float(dts[i]), 3), "camera": cam}
            if cam is not None:
                ev.update(_camera_at(ep, cam, i))
            events.append(ev)
    flagged = [e for e in events if e.get("visual_jump") is False]
    return {"events": events, "flagged": len(flagged) > 0,
            "rule": f"a single-frame recorded step >= max({JUMP_FLOOR[kind]} {unit}, {JUMP_FACTOR:g} x the actor's "
                    f"95th-percentile step, every step rescaled to one typical frame interval so a recording gap is "
                    f"not a leap) and >= {ISOLATION:g} x the steps on either side (a leap, not fast motion), while "
                    f"the actor's own camera changes no more than {VISUAL_JUMP:g} x its median over the surrounding "
                    f"7 frames"}


def _camera_at(ep: dict, cam: str, i: int) -> dict:
    """Camera cam's image change at the step between anchor frames i and i+1, against its median change over
    3 frames either side (decoded at their exact pts)."""
    s = ep["sources"][cam]
    n = int(s["n_frames"])
    km = ep["kmap"].get(cam)

    def own(k: int) -> int:
        return int(km[k]) if km is not None else k
    lo, hi = max(0, own(i) - 3), min(n - 1, own(i + 1) + 3)
    ks = list(range(lo, hi + 1))
    pts = ep["times"].get(f"{cam}_pts") if ep.get("times") is not None else None
    ims = mf.extract_frames(s["packed"], s["base_s"], n, ks, pts=pts, fps=me.ep_fps(ep))
    grey = [np.asarray(ims[k].convert("L").resize((SMALL_W, max(2, int(SMALL_W * ims[k].height / ims[k].width)))),
                       dtype=np.int16) for k in ks]
    ch = [float(np.abs(b - a).mean()) for a, b in zip(grey, grey[1:])]
    at = ch[ks.index(own(i + 1)) - 1] if own(i + 1) in ks[1:] else max(ch)
    med = float(np.median(ch))
    return {"image_change_at": round(at, 2), "image_change_window_median": round(med, 2),
            "visual_jump": bool(at > VISUAL_JUMP * max(med, 0.5))}


def grippers(ep_dir: Path) -> dict | None:
    """context["gripper_channels"]: each actor's gripper range and whether it is flat, and flagged when one is;
    None on video-only rigs."""
    ep = me.load(ep_dir)
    if me.state_kind(ep) == "none":
        return None
    a, b = me.state_span(ep)                # the frames the state covers
    s = np.asarray(ep["state"], dtype=np.float64)[a:b]
    out = {}
    for g, name in enumerate(me.actors(ep)):
        v = s[:, 7 * g + 6]
        out[name] = {"min": round(float(v.min()), 5), "max": round(float(v.max()), 5),
                     "flat": bool(np.ptp(v) <= 1e-9 * max(1.0, float(np.abs(v).max())))}
    return {"actors": out, "flagged": any(a["flat"] for a in out.values()),
            "rule": "a gripper whose recorded value is exactly the same at every frame of the episode"}


# mode -> (context.json key, check, the result field that counts as a finding)
MODES = {"pairing": ("stream_pairing", pairing, "crossed"),
         "jumps": ("recorded_jumps", jumps, "flagged"),
         "grippers": ("gripper_channels", grippers, "flagged")}


def _safe(mode: str, d: str) -> tuple[str, dict | None, str | None]:
    """One check's result on one episode. A check that crashes is that check's result, {"error": why, its finding's
    field false}, which the board shows as an error, never a missing result; the other checks run on as usual."""
    try:
        return d, MODES[mode][1](Path(d)), None
    except Exception as e:  # recorded as the check's result, never silently skipped
        return d, {"error": f"{type(e).__name__}: {e}"[:300], MODES[mode][2]: False}, None


def main():
    ap = argparse.ArgumentParser(prog="python -m checks.stream_pairing", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", type=Path, metavar="EPISODES", help="folders of prepared episode_* folders")
    ap.add_argument("--jobs", type=int, default=8, help="episodes checked in parallel")
    ap.add_argument("--force", action="store_true", help="recompute episodes that already have a result")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--jumps", action="store_true", help="find recorded jumps instead of checking stream pairing")
    mode.add_argument("--grippers", action="store_true", help="find recorded gripper openings that never change instead")
    args = ap.parse_args()
    name = "grippers" if args.grippers else "jumps" if args.jumps else "pairing"
    key, _, field = MODES[name]
    eps = []
    for root in args.roots:
        for d in sorted(root.glob("episode_*")):
            if not (d / "context.json").exists():
                continue
            ctx = json.loads((d / "context.json").read_text())
            if args.force or key not in ctx:
                eps.append(str(d))
    print(f"episodes to scan: {len(eps)}", flush=True)
    done = found = failed = 0
    with ProcessPoolExecutor(args.jobs) as ex:
        for d, r, err in ex.map(_safe, [name] * len(eps), eps, chunksize=2):
            done += 1
            if err:
                failed += 1
                print(f"FAILED {Path(d).name}: {err}", flush=True)
                continue
            if r and r.get("error"):
                failed += 1
                print(f"ERRORED {Path(d).name}: {r['error']}", flush=True)
            p = Path(d) / "context.json"
            ctx = json.loads(p.read_text())
            ctx[key] = r
            p.write_text(json.dumps(ctx, indent=1))
            if r and r.get(field):
                found += 1
                print(f"{key.upper()} {Path(d).name} {r}", flush=True)
            if done % 50 == 0:
                print(f"progress {done}/{len(eps)} {field}={found} failed={failed}", flush=True)
    print(f"done {done} {field}={found} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
