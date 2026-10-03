"""A recording's other signals and depth streams, as the board's episode page shows them (its "Sensors" panel and the
depth view of each camera).

A prepared episode can carry per-frame signals (signals.npz, described in context.json "signals": a force, a joint's
velocity, a 16 x 16 pressure map, hand landmarks) and a depth stream per camera (depth.json). This writes one compact
file per board episode that has either, named after its label file, beside the board's label files (like hands/),
never in qa/, so nothing the board counts or exports as labels includes it:

  BOARD/sensors/<label file>   one episode (below)
  BOARD/sensors/index.json     {"format", "files": {label file: {"signals": n, "depth": [camera view, ...]}}}, so
                               the page knows before it fetches which episodes have sensors and which cameras depth

What a signal does is read by label/signals.py (its resting level, whether it rests and rises like a touch, which
way it moves when active, the spans it is away from rest), never here. When context.json gives a signal's resting
level and typical swing measured across the whole upload ("rest", "swing"), they are used, so a colour on the page
means the same distance from rest in every episode of a dataset; a signal prepared before those were measured falls
back to its own episode's (swing_from says which).

One file:

  {"format": "board-sensors/1",
   "frames": N,                            the anchor camera's frames, one signal row each
   "times": {"ms0", "dur", "d"},           each anchor frame's time on the board clip's clock (the main camera's first
                                           frame is 0), in whole milliseconds: the first, then per frame its gap to
                                           the previous minus dur, as Base64 VLQ integers (board/hands.py's encoding)
   "stride": S, "n": M,                    the page's samples are anchor frames 0, S, 2S, ... (M of them), at most
                                           RATE_HZ a second
   "signals": [one per signal, in the dataset's order],
   "depth": {view: {"units", "scale_m", "kind", "ticks", "bar"}}}

A signal: "name", "dims", and when the dataset gives them "shape" ([16, 16]), "names" (one per value), "rate_hz" and
"source", and "aligned_by" when the reader placed it on the video from both starts because no clock was shared
(prepare/formats.py mark_assumed), which the page says in its lane. Each signal keeps its own length: one that ends
before the others has no reading after its last row, and a gap or a stretch before a signal starts is no reading, which
the page draws as a gap. A signal that cannot be drawn is left out and named in the file's "errors" ([{"name",
"error"}]), and the page names it under the lanes it drew. "constant": true when no value ever changes (its "value" is
the first row, or null with no reading); the page lists those by name. Otherwise "rests_and_rises" and "touch"
(label/signals.py; touch is is_touch, by the signal's name and its numbers), "direction" ("up", "down" or null), "spans"
([[start s, end s], ...] on the clip clock, from every frame, for a signal that rests and rises or is touch), for a
signal that times one of the episode's contacts (context.json "contacts") its "strength" ({"lo", "step", "data"}, one
value per sample: its activity over its swing, label/contacts.py, so the page sums a contact's signals into the curve
drawn inside its bar), and its samples:

  a vector of SMALL values or fewer   "values": {"lo": [...], "step": [...], "data"}: per value v = lo + q * step
  an array of more values             "activity": {"lo": [x], "step": [x], "data"}: its activity (label/signals.py)
                                      per sample; a 2-D array that is touch also has "map": {"lo", "step",
                                      "data"} (every value, one byte each), "rest" (its resting level per value) and
                                      "swing", so the page draws it as a heatmap: distance from rest over swing,
                                      clipped to 0..1, in the signal's direction

"data" is Base64 of little-endian unsigned integers, M rows of the stated values each: uint16 for "values",
"activity" and "strength" (65535 is no reading), uint8 for "map" (255 is no reading). Every decoded value is within
step / 2 of the recorded one.

A depth camera: "units" ("metres" or "relative"), "scale_m" (metres per stored unit, or null) and "kind", and the colour
bar of its pictures from label/depth.py: "ticks" (legend_ticks) and "bar" (colours along the bar, near first). The
board's depth clip of the camera (board/clips.py, CLIPS/depth_<view>/) is drawn by the same label/depth.py picture().

  python -m board sensors --episodes EPISODES --qa BOARD/qa --out OUT    write the files for a board's label files
board/build.py writes them on every build for the episodes that have signals or depth ("sensors": false in the
manifest turns it off).
"""
from __future__ import annotations

import argparse
import base64
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np

from board.hands import vlq_append, vlq_decode

FORMAT = "board-sensors/1"
RATE_HZ = 15.0           # the page's samples a second, at most
NONE16, NONE8 = 65535, 255
BAR_STOPS = 24           # colours along a depth colour bar


# ---------------------------------------------------------------- label/signals.py, with the upload's rest and swing

def _call(fn, a, rest=None, swing=None):
    """fn(a) from label/signals.py, given the upload-wide rest and swing when the function takes them."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = {}
    kw = {}
    if rest is not None and "rest" in params:
        kw["rest"] = rest
    if swing is not None and "swing" in params:
        kw["swing"] = swing
    return fn(a, **kw)


def _spans(S, a, t, rest, swing):
    try:
        params = inspect.signature(S.active_spans).parameters
    except (TypeError, ValueError):
        params = {}
    kw = {k: v for k, v in (("rest", rest), ("swing", swing)) if v is not None and k in params}
    return S.active_spans(a, t, **kw)


# ---------------------------------------------------------------- quantization

def quantize(a: np.ndarray, bits: int = 16, per_value: bool = True) -> dict:
    """{"lo", "step", "data"} for a (rows x values): per value (or one for the whole array) lo and step, every finite
    value stored as round((v - lo) / step) in `bits` bits, the top code meaning no reading."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    none = NONE16 if bits == 16 else NONE8
    fin = np.isfinite(a)
    with np.errstate(all="ignore"):
        if per_value:
            lo = np.where(fin.any(axis=0), np.nanmin(np.where(fin, a, np.nan), axis=0), 0.0)
            hi = np.where(fin.any(axis=0), np.nanmax(np.where(fin, a, np.nan), axis=0), 0.0)
        else:
            lo = np.full(a.shape[1], np.nanmin(a[fin]) if fin.any() else 0.0)
            hi = np.full(a.shape[1], np.nanmax(a[fin]) if fin.any() else 0.0)
    step = (hi - lo) / (none - 1)
    q = np.where(step > 0, np.round((np.where(fin, a, 0.0) - lo) / np.where(step > 0, step, 1.0)), 0.0)
    q = np.clip(q, 0, none - 1).astype(np.uint16 if bits == 16 else np.uint8)
    q[~fin] = none
    data = base64.b64encode(q.astype("<u2" if bits == 16 else "u1").tobytes()).decode()
    if per_value:
        return {"lo": [float(x) for x in lo], "step": [float(x) for x in step], "data": data}
    return {"lo": float(lo[0]), "step": float(step[0]), "data": data}


def dequantize(block: dict, rows: int, bits: int = 16) -> np.ndarray:
    """The page's decoder, in Python: rows x values floats, NaN for no reading."""
    raw = np.frombuffer(base64.b64decode(block["data"]), dtype="<u2" if bits == 16 else "u1").astype(np.float64)
    q = raw.reshape(rows, -1)
    lo, step = np.asarray(block["lo"], dtype=np.float64), np.asarray(block["step"], dtype=np.float64)
    v = lo + q * step
    v[q == (NONE16 if bits == 16 else NONE8)] = np.nan
    return v


def encode_times(t_s: np.ndarray) -> dict:
    ms = np.round(np.asarray(t_s, dtype=np.float64) * 1000).astype(np.int64)
    gaps = np.diff(ms)
    dur = int(np.bincount(gaps - gaps.min()).argmax() + gaps.min()) if len(gaps) else 0
    out: list = []
    for g in gaps:
        vlq_append(int(g) - dur, out)
    return {"ms0": int(ms[0]) if len(ms) else 0, "dur": dur, "d": "".join(out)}


def decode_times(block: dict) -> np.ndarray:
    p = block["ms0"]
    out = [p]
    for g in vlq_decode(block["d"]):
        p += block["dur"] + g
        out.append(p)
    return np.asarray(out, dtype=np.float64) / 1000.0


# ---------------------------------------------------------------- one episode

def clip_times(ep_dir: Path, ctx: dict, n: int) -> np.ndarray:
    """Seconds on the board clip's clock for each anchor frame: the anchor camera's real capture times (times.npz)
    from its first frame, which the clip shows at 0 (board/clips.py), else frame / fps."""
    from label.episode import order_views
    tp = ep_dir / "times.npz"
    src_p = ep_dir / "sources.json"
    if tp.exists() and src_p.exists():
        views = order_views(json.loads(src_p.read_text()))
        with np.load(tp) as z:
            if views and views[0] in z.files and len(z[views[0]]) >= n > 0:
                t = np.asarray(z[views[0]][:n], dtype=np.float64)
                return t - float(ctx.get("clock_zero_s") or 0.0)
    fps = float(ctx.get("fps") or 30.0)
    return np.arange(n, dtype=np.float64) / fps


def _range(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each value's smallest and largest finite reading (NaN for a value with none), a few thousand rows at a time, so
    a wide signal is never copied whole."""
    import warnings
    step = max(1, (1 << 19) // max(1, a.shape[1]))
    lo = np.full(a.shape[1], np.nan)
    hi = np.full(a.shape[1], np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # a value with no reading in a stretch: NaN, as wanted
        for i in range(0, len(a), step):
            c = np.asarray(a[i:i + step], dtype=np.float64)
            c = np.where(np.isfinite(c), c, np.nan)
            lo = np.fmin(lo, np.nanmin(c, axis=0))
            hi = np.fmax(hi, np.nanmax(c, axis=0))
    return lo, hi


def pad(a: np.ndarray, n: int) -> np.ndarray:
    """A signal's rows as floats on n anchor frames: its own rows, then no reading (NaN) after its last, so a signal
    shorter than the others is drawn for the frames it has; a longer one is cut to the frames the clip has. A floating
    signal keeps its precision (a wide float32 skin is never copied whole as float64; label/signals.py reads it in
    float64 pieces), and any other is made float64, which can hold no reading."""
    a = np.asarray(a)
    if not np.issubdtype(a.dtype, np.floating):
        a = a.astype(np.float64)
    if len(a) >= n:
        return a[:n]
    return np.concatenate([a, np.full((n - len(a),) + a.shape[1:], np.nan, dtype=a.dtype)])


def signal_doc(meta: dict, a: np.ndarray, t: np.ndarray, stride: int, in_contact: bool = False) -> dict:
    """One signal's entry (module docstring); in_contact: one of the episode's contacts is timed by it, so its strength
    is kept too."""
    from label import signals as S
    a = np.asarray(a)
    a = a if np.issubdtype(a.dtype, np.floating) else a.astype(np.float64)    # as stored, never a whole float64 copy
    if a.ndim == 1:
        a = a[:, None]
    d = a.shape[1]
    doc = {"name": meta["name"], "dims": d}
    for k in ("shape", "names", "rate_hz", "source", "aligned_by"):
        if meta.get(k) is not None:
            doc[k] = meta[k]
    fin = np.isfinite(a)
    lo, hi = _range(a)
    if not fin.any() or bool((hi[fin.any(axis=0)] == lo[fin.any(axis=0)]).all() and fin.any(axis=0).all()):
        first = next((r for r in a if np.isfinite(r).all()), None)
        doc["constant"] = True
        doc["value"] = None if first is None else [float(x) for x in first[:16]]
        return doc
    rest = np.asarray(meta["rest"], dtype=np.float64) if meta.get("rest") is not None else None
    swing = float(meta["swing"]) if meta.get("swing") else None
    rr = bool(_call(S.rests_and_rises, a, rest, swing))
    touch = bool(S.is_touch(meta["name"], a, rest, swing))
    doc.update({"constant": False, "rests_and_rises": rr, "touch": touch,
                "direction": _call(S.direction, a, rest, swing)})
    doc["spans"] = [[round(s, 3), round(e, 3)] for s, e in _spans(S, a, t, rest, swing)] if rr or touch else []
    pick = np.arange(0, len(a), stride)
    if in_contact:
        from label import contacts as lc
        doc["strength"] = quantize(lc._strength(a, {"rest": rest, "swing": swing})[pick])
    if d <= S.SMALL:
        doc["values"] = quantize(a[pick])
        return doc
    act = np.asarray(_call(S.activity, a, rest, swing), dtype=np.float64)
    doc["activity"] = quantize(act[pick])
    shape = meta.get("shape") or []
    if len(shape) == 2 and touch:
        level = rest if rest is not None else np.asarray(_call(S.resting_level, a, rest, swing), dtype=np.float64)
        if swing is None:
            swing_ep = S.swing_of(a)
            doc["swing"], doc["swing_from"] = float(swing_ep), "episode"
        else:
            doc["swing"], doc["swing_from"] = swing, "upload"
        doc["rest"] = [round(float(x), 4) for x in level]
        doc["map"] = quantize(a[pick], bits=8, per_value=False)
    return doc


def depth_doc(entry: dict, ctx_depth: dict) -> dict:
    """One camera's depth: its units and the colour bar of its pictures (label/depth.py)."""
    out = {"units": ctx_depth.get("units") or ("metres" if entry.get("scale_m") else "relative"),
           "scale_m": entry.get("scale_m"), "kind": entry.get("kind") or "depth"}
    out.update(legend_bar(entry))
    return out


def legend_bar(entry: dict) -> dict:
    """{"bar": BAR_STOPS colours along the colour bar of a camera's depth pictures, near first, "ticks": [[position
    0..1 along that bar, label], ...]}, both from label/depth.py: the colours are picture() of a ramp of readings
    across its scale (metric depth from METRIC_NEAR_M to METRIC_FAR_M on its log scale, any other from the near to
    the far end of its range), the ticks legend_ticks() turned to run near first."""
    from label import depth as dp
    if not hasattr(dp, "picture") or not hasattr(dp, "legend_ticks"):
        return {}
    disparity = entry.get("kind") == "disparity"
    if entry.get("scale_m") and not disparity:
        ramp = np.geomspace(dp.METRIC_NEAR_M, dp.METRIC_FAR_M, BAR_STOPS) / float(entry["scale_m"])
        e = entry
    else:
        # a ramp across the range of unknown-unit depth (or disparity) is the same whatever the range, so it is drawn
        # on a range of its own
        ramp = np.linspace(2.0, 1.0, BAR_STOPS) if disparity else np.linspace(1.0, 2.0, BAR_STOPS)
        e = {**entry, "range": (1.0, 2.0)}
    im = np.asarray(dp.picture(ramp[None, :].astype(np.float64), e).convert("RGB"))[0]
    ticks = [(float(p), str(label)) for p, label in dp.legend_ticks(entry)]
    # legend_ticks gives positions along its own bar; the bar here runs near first, so the ticks are turned when the
    # nearest tick ("near", or the smallest distance) sits at its far end
    near = next((p for p, label in ticks if label == "near"), ticks[0][0] if ticks else 0.0)
    if near > 0.5:
        ticks = [(1.0 - p, label) for p, label in ticks]
    return {"bar": ["#%02x%02x%02x" % tuple(int(c) for c in px) for px in im],
            "ticks": [[round(p, 4), label] for p, label in sorted(ticks)]}


def episode_doc(ep_dir: Path) -> dict | None:
    """The sensors file of one prepared episode, or None when it has neither signals nor depth."""
    ep_dir = Path(ep_dir)
    ctx = json.loads((ep_dir / "context.json").read_text())
    metas = ctx.get("signals") or []
    has_sig = bool(metas) and (ep_dir / "signals.npz").exists()
    depth_p = ep_dir / "depth.json"
    if not has_sig and not depth_p.exists():
        return None
    doc: dict = {"format": FORMAT}
    if has_sig:
        with np.load(ep_dir / "signals.npz") as z:
            arrays = {m["key"]: np.asarray(z[m["key"]]) for m in metas if m.get("key") in z.files}
        # every anchor frame any signal reaches: a signal shorter than the others keeps its own length and has no
        # reading after it (pad), never cutting the others to it
        n = max((len(v) for v in arrays.values()), default=0)
        t = clip_times(ep_dir, ctx, n)
        n = min(n, len(t))
        t = t[:n]
        dt = float(np.median(np.diff(t))) if n > 1 else 1.0 / RATE_HZ
        stride = max(1, int(np.ceil((1.0 / max(dt, 1e-6)) / RATE_HZ - 1e-6)))
        touched = {nm for c in ctx.get("contacts") or [] for nm in c.get("signals") or []}
        docs, errors = [], []
        for m in metas:
            if m.get("key") not in arrays:
                continue
            try:
                docs.append(signal_doc(m, pad(arrays[m["key"]], n), t, stride, m["name"] in touched))
            except Exception as err:  # noqa: BLE001 - this signal is named with the reason, the others are drawn
                errors.append({"name": m["name"], "error": f"{type(err).__name__}: {err}"[:300]})
        doc.update({"frames": n, "times": encode_times(t), "stride": stride, "n": len(range(0, n, stride)),
                    "signals": docs, **({"errors": errors} if errors else {})})
        if (ep_dir / "times.npz").exists() and (ep_dir / "sources.json").exists():
            from board.clips import DISPLAY_TICKS_PER_S, display_ticks, main_cam, start_offsets
            src = json.loads((ep_dir / "sources.json").read_text())
            cam = main_cam(src)
            _, skip = start_offsets(ep_dir, src, float(ctx.get("fps") or 30)).get(cam, (0, 0))
            captures = t[skip:min(n, int(src.get(cam, {}).get("n_frames") or n))]
            if len(captures):
                ticks, _ = display_ticks(captures, float(ctx.get("fps") or 30), True, t, np.arange(n) - skip)
                doc["playback"] = {"starts": (ticks / DISPLAY_TICKS_PER_S).tolist(), "captures": captures.tolist()}
    if depth_p.exists():
        from label import depth as dp
        entries = dp.load(ep_dir)
        src = json.loads((ep_dir / "sources.json").read_text()) if (ep_dir / "sources.json").exists() else {}
        cd = ctx.get("depth") or {}
        doc["depth"] = {v: depth_doc(e, cd.get(v) or {}) for v, e in entries.items() if not src or v in src}
    return doc


def summary(doc: dict) -> dict:
    """The index entry of one file: how many of its signals change, and the cameras with depth."""
    return {"signals": sum(1 for s in doc.get("signals") or [] if not s.get("constant")),
            "constant": sum(1 for s in doc.get("signals") or [] if s.get("constant")),
            "depth": sorted(doc.get("depth") or {})}


def build(episodes: dict, out_dir: Path) -> dict:
    """Write the sensors file of every board label file whose prepared episode has signals or depth into out_dir
    (which must exist), and index.json. episodes maps a label file to its prepared episode folder."""
    files, skipped, total, biggest = {}, [], 0, 0
    for f, ep in sorted(episodes.items()):
        try:
            doc = episode_doc(Path(ep))
        except Exception as err:  # noqa: BLE001 - reported per episode, the build goes on
            skipped.append({"file": f, "skip": f"{type(err).__name__}: {err}"[:300]})
            continue
        if doc is None:
            continue
        body = json.dumps(doc, separators=(",", ":"))
        tmp = out_dir / f".{f}.part"
        tmp.write_text(body)
        os.replace(tmp, out_dir / f)
        files[f] = summary(doc)
        total += len(body)
        biggest = max(biggest, len(body))
    (out_dir / "index.json").write_text(json.dumps({"format": FORMAT, "files": files}, separators=(",", ":")))
    return {"written": len(files), "skipped": skipped, "bytes": {"total": total, "max": biggest}}


def episode_folders(qa: Path, roots: list) -> dict:
    """{label file: prepared episode folder} for the board's label files, found by the run's episode name under any
    of roots."""
    out = {}
    for f in sorted(Path(qa).glob("episode_*.json")):
        d = json.loads(f.read_text())
        name = (d.get("_meta") or {}).get("run_episode") or f.stem
        hit = next((Path(r) / name for r in roots if (Path(r) / name / "context.json").exists()), None)
        if hit is not None:
            out[f.name] = hit
    return out


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m board sensors", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qa", type=Path, required=True, help="the board's label files (BOARD/qa)")
    ap.add_argument("--episodes", type=Path, nargs="+", required=True,
                    help="the folders of prepared episodes the board's runs labelled")
    ap.add_argument("--out", type=Path, required=True, help="the folder of sensors files")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    res = build(episode_folders(a.qa, a.episodes), a.out)
    for sk in res["skipped"]:
        print(f"sensors: skipped {sk['file']}: {sk['skip']}", file=sys.stderr)
    print(json.dumps({**res, "skipped": len(res["skipped"])}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
