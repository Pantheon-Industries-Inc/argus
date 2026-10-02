"""Deterministic checks on an episode's other signals and depth streams, before labelling.

    python -m checks.sensors [--jobs N] [--force] EPISODES [EPISODES ...]

EPISODES are folders of prepared episode_* folders. For each episode this reads signals.npz and depth.json (written by
prepare/formats.py) and writes context["sensor_checks"]; `python -m board build` copies it into the episode's
dataset_checks, and the board lists it under Checks. None of it goes into the prompt.

Like the signals the model is shown (label/signals.py), nothing here knows what a sensor is called. Each check reads
how the numbers behave:

  no_reading       a signal with no reading at more than NO_READING_SHARE of the frames (a tracker that lost the
                   hand, a glove that stopped sending)
  constant         a signal whose every value is the same at every frame (a sensor that sent nothing new)
  dead_values      values of an array that never change in any episode of the upload while the rest of it does (cells a
                   pressure map does not have, or dead ones), with where they are. Within one episode a value that
                   never changes says nothing: a saturating cell reads exactly its untouched level until something
                   presses it, so a cell the hand never touched in a two-second clip is constant. Only a value that
                   stays constant across every episode is worth naming, and it is named once per upload
  pinned           values of an array that sit exactly at the far end of their range, in the direction the array moves
                   when active, for more than PINNED_SHARE of the frames (cells pressed past what they can measure)
  slow_sensor      a signal recorded more slowly than half the camera's frame rate, so its values are held between
                   samples (ActionSense's glove at 6 Hz under a 30 fps camera)
  clock_offset     the recording keeps a clock per sensor (the time each reading was received) and one runs more than
                   CLOCK_OFFSET_FRAMES camera frames from the first, or wanders by more than that within the episode
  depth_invalid    depth pictures in which more than DEPTH_INVALID_SHARE of the pixels hold no reading, on average
  depth_frozen     consecutive sampled depth pictures that are identical while the colour camera's frames change
  depth_offset     depth frames whose capture time is more than DEPTH_OFFSET_FRAMES camera frames from the colour frame
                   they are shown with

They are reported as notes, never counted as issues: a check counts only once every episode it fires on has been
confirmed on the frames (checks/capture_qc.py states the same rule), and no dataset has been checked that way yet.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from label import depth as dp
from label import episode as me
from label import signals as sg

NO_READING_SHARE = 0.2
PINNED_SHARE = 0.1
DEPTH_INVALID_SHARE = 0.5
DEPTH_OFFSET_FRAMES = 2.0
DEPTH_SAMPLES = 24
CLOCK_OFFSET_FRAMES = 1.0

NAMES = {"no_reading": "Signal has no reading at many frames", "constant": "Signal never changes",
         "dead_values": "Values of an array never change", "pinned": "Values pinned at the end of their range",
         "slow_sensor": "Sensor slower than the camera", "clock_offset": "Sensor clocks apart", "depth_invalid": "Depth pictures mostly without readings",
         "depth_frozen": "Depth picture frozen while colour changes",
         "depth_offset": "Depth frames far in time from their colour frames"}


def _cell(i: int, shape) -> str:
    if shape and len(shape) == 2:
        r, c = divmod(int(i), int(shape[1]))
        return f"r{r}c{c}"
    return f"[{int(i)}]"


def signal_findings(ep: dict) -> list[dict]:
    sig = ep.get("signals") or {}
    meta = ep.get("signal_meta") or {}
    fps = me.ep_fps(ep)
    out = []
    for name, a in sig.items():
        a = np.asarray(a, dtype=np.float64)
        if not len(a):
            continue
        m = meta.get(name) or {}
        gone = np.isnan(a).all(axis=1)
        if gone.mean() > NO_READING_SHARE:
            out.append({"check": "no_reading", "signal": name,
                        "evidence": f"{name} has no reading at {int(gone.sum())} of {len(a)} frames"})
        ok = a[~gone]
        if not len(ok):
            continue
        with np.errstate(all="ignore"):
            span = np.nanmax(ok, axis=0) - np.nanmin(ok, axis=0)
        if (span == 0).all() and len(ok) > 1:
            out.append({"check": "constant", "signal": name,
                        "evidence": f"every value of {name} is the same at all {len(ok)} frames"})
            continue
        if a.shape[1] > sg.SMALL and (span > 0).any():
            if sg.has_rest(ok):
                way = sg.direction(ok)
                if way:
                    edge = np.nanmax(ok) if way == "up" else np.nanmin(ok)
                    pinned = (ok == edge).mean(axis=0)
                    many = np.flatnonzero(pinned > PINNED_SHARE)
                    if len(many):
                        out.append({"check": "pinned", "signal": name, "count": int(len(many)),
                                    "evidence": f"{len(many)} values of {name} read exactly {sg._num(edge)}, the "
                                                f"{'top' if way == 'up' else 'bottom'} of its range, at more than "
                                                f"{PINNED_SHARE:.0%} of the frames ("
                                                + ", ".join(_cell(i, m.get("shape")) for i in many[:8])
                                                + (" and more" if len(many) > 8 else "") + ")"})
        rate = m.get("rate_hz")
        if rate and rate < fps / 2:
            out.append({"check": "slow_sensor", "signal": name,
                        "evidence": f"{name} is recorded at {rate:g} per second under a {fps:.0f} fps camera, so each "
                                    f"reading is held for about {fps / rate:.0f} frames"})
    return out


def clock_findings(ep: dict) -> list[dict]:
    clocks = ep["context"].get("clocks") or []
    frame_ms = 1000.0 / me.ep_fps(ep)
    out = []
    for c in clocks[1:]:
        if abs(c["offset_ms"]) > CLOCK_OFFSET_FRAMES * frame_ms or c["spread_ms"] > CLOCK_OFFSET_FRAMES * frame_ms:
            out.append({"check": "clock_offset", "signal": c["name"],
                        "evidence": f"{c['name']} runs {c['offset_ms']:+.0f} ms from {clocks[0]['name']} (spread "
                                    f"{c['spread_ms']:.0f} ms), more than a camera frame of {frame_ms:.0f} ms"})
    return out


def depth_findings(ep: dict) -> list[dict]:
    d = ep.get("depth") or {}
    out = []
    if not d:
        return out
    pl = me.plan(ep)
    ks = pl["ks"]
    pick = [ks[int(i)] for i in np.linspace(0, len(ks) - 1, min(DEPTH_SAMPLES, len(ks)))]
    imgs = None
    for v, e in d.items():
        name = me.cam_name(ep, v)
        got = dp.at_anchor(ep, d, v, pick)
        if not got:
            continue
        inv = float(np.mean([(x == 0).mean() for x in got.values()]))
        if inv > DEPTH_INVALID_SHARE:
            out.append({"check": "depth_invalid", "camera": name,
                        "evidence": f"on average {inv:.0%} of the pixels of {name}'s depth pictures hold no reading"})
        same = [(a, b) for a, b in zip(pick, pick[1:]) if a in got and b in got and np.array_equal(got[a], got[b])
                and int(e["km"][a]) != int(e["km"][b])]
        if same:
            if imgs is None:
                imgs = me.frames(ep, {"ks": pick})
            moved = [(a, b) for a, b in same
                     if float(np.abs(np.asarray(imgs[v][a].convert("L"), dtype=np.float32)
                                     - np.asarray(imgs[v][b].convert("L"), dtype=np.float32)).mean()) > 2.0]
            if moved:
                out.append({"check": "depth_frozen", "camera": name, "t_s": round(me.frame_time(ep, moved[0][0]), 2),
                            "evidence": f"{name}'s depth picture is identical between {len(moved)} pairs of sampled "
                                        f"instants while its colour frames change (first at "
                                        f"{me.frame_time(ep, moved[0][0]):.2f} s)"})
        td = e.get("t") if e.get("t") is not None else (ep.get("times") or {}).get(f"depth_{v}")
        if td is not None:
            n = int(pl["n"])
            ta = np.array([me.frame_time(ep, k) for k in range(n)])
            step = float(np.median(np.diff(ta))) if n > 1 else 1 / me.ep_fps(ep)
            off = np.abs(np.asarray(td)[np.asarray(e["km"][:n], dtype=int)] - ta)
            far = off > DEPTH_OFFSET_FRAMES * step
            if far.mean() > 0.01:
                out.append({"check": "depth_offset", "camera": name,
                            "evidence": f"{int(far.sum())} of {n} of {name}'s colour frames have no depth frame within "
                                        f"{DEPTH_OFFSET_FRAMES:g} frames of them (median gap "
                                        f"{float(np.median(off)) * 1000:.0f} ms)"})
    return out


def constant_values(ep_dir: Path) -> dict:
    """{signal: [indices of its values that never change in this episode]} for every array of more than SMALL values."""
    ep = me.load(ep_dir)
    out = {}
    for name, a in (ep.get("signals") or {}).items():
        a = np.asarray(a, dtype=np.float64)
        if a.shape[1] <= sg.SMALL or not np.isfinite(a).any():
            continue
        with np.errstate(all="ignore"):
            span = np.nanmax(a, axis=0) - np.nanmin(a, axis=0)
        out[name] = {"never": [int(i) for i in np.flatnonzero(span == 0)], "n": int(a.shape[1]),
                     "shape": (ep.get("signal_meta") or {}).get(name, {}).get("shape")}
    return out


def dead_across(per_episode: dict) -> dict:
    """{signal: finding} for values that never change in any episode of the upload (at least two episodes record the
    signal) while some of its other values do change."""
    by_sig = {}
    for cv in per_episode.values():
        for name, x in (cv or {}).items():
            by_sig.setdefault(name, []).append(x)
    out = {}
    for name, xs in by_sig.items():
        if len(xs) < 2:
            continue
        never = set(xs[0]["never"]).intersection(*[set(x["never"]) for x in xs[1:]])
        if never and len(never) < xs[0]["n"]:
            idx = sorted(never)
            where = ", ".join(_cell(i, xs[0]["shape"]) for i in idx[:12]) + (" and more" if len(idx) > 12 else "")
            out[name] = {"check": "dead_values", "signal": name, "count": len(idx),
                         "evidence": f"{len(idx)} of the {xs[0]['n']} values of {name} never change in any of the "
                                     f"{len(xs)} episodes of this upload ({where}): values the sensor may not have, or "
                                     "dead ones"}
    return out


def run_episode(ep_dir: Path) -> dict | None:
    """context["sensor_checks"], or None when the episode has neither other signals nor depth."""
    ep = me.load(ep_dir)
    if not ep.get("signals") and not ep.get("depth") and not ep["context"].get("clocks"):
        return None
    found = signal_findings(ep) + clock_findings(ep) + depth_findings(ep)
    applies = set()
    if ep.get("signals"):
        applies |= {"no_reading", "constant", "dead_values", "pinned", "slow_sensor"}
    if len(ep["context"].get("clocks") or []) >= 2:
        applies.add("clock_offset")
    if ep.get("depth"):
        applies |= {"depth_invalid", "depth_frozen", "depth_offset"}
    fired = {f["check"] for f in found}
    return {"notes": found, "flagged": False,
            "checks": [{"check": c, "name": NAMES[c], "status": "fired" if c in fired else "clear" if c in applies else "na"}
                       for c in NAMES],
            "rule": "notes only: a sensor check counts as an issue once every episode it fires on is confirmed on the frames"}


def _constants(d: str) -> dict:
    try:
        return constant_values(Path(d))
    except Exception:
        return {}


def _safe(d: str):
    try:
        return d, run_episode(Path(d)), None
    except Exception as e:  # reported per episode, never silently skipped
        return d, None, f"{type(e).__name__}: {e}"[:300]


def main():
    ap = argparse.ArgumentParser(prog="python -m checks.sensors", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", type=Path, metavar="EPISODES", help="folders of prepared episode_* folders")
    ap.add_argument("--jobs", type=int, default=8, help="episodes checked in parallel")
    ap.add_argument("--force", action="store_true", help="recompute episodes that already have a result")
    a = ap.parse_args()
    eps = []
    for root in a.roots:
        for d in sorted(root.glob("episode_*")):
            if (d / "context.json").exists() and (a.force or "sensor_checks" not in json.loads((d / "context.json").read_text())):
                eps.append(str(d))
    print(f"episodes to check: {len(eps)}", flush=True)
    done = fired = failed = 0
    # dead values are judged across every episode of each folder, never within one (module docstring)
    every = {}
    for root in a.roots:
        ds = [str(d) for d in sorted(root.glob("episode_*")) if (d / "context.json").exists()]
        with ProcessPoolExecutor(a.jobs) as ex:
            consts = dict(zip(ds, ex.map(_constants, ds, chunksize=4)))
        dead = dead_across(consts)
        for d in ds:
            every[d] = [f for name, f in dead.items() if name in (consts.get(d) or {})]
    with ProcessPoolExecutor(a.jobs) as ex:
        for d, r, err in ex.map(_safe, eps, chunksize=2):
            if r is not None and every.get(d):
                r["notes"] += every[d]
                for c in r["checks"]:
                    if c["check"] == "dead_values":
                        c["status"] = "fired"
            done += 1
            if err:
                failed += 1
                print(f"FAILED {Path(d).name}: {err}", flush=True)
                continue
            p = Path(d) / "context.json"
            ctx = json.loads(p.read_text())
            ctx["sensor_checks"] = r
            p.write_text(json.dumps(ctx, indent=1))
            if r and r["notes"]:
                fired += 1
                print(f"SENSORS {Path(d).name}: " + "; ".join(n["evidence"] for n in r["notes"][:4]), flush=True)
    print(f"done {done} with_notes={fired} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
