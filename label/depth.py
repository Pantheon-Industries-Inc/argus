"""Depth frames: decoded exactly as recorded, and drawn so that a colour means the same thing everywhere it can.

An episode's depth streams are described in depth.json (prepare/formats.py write_depth): per camera, the file, its
frame for each anchor frame (depth_kmap_<view>.npy), its exact pts (times.npz depth_<view>_pts), metres per stored unit
when the recording says (scale_m), whether a larger value is farther (kind "depth") or nearer (kind "disparity"), and,
for depth whose unit is not known, the range of its readings across the whole upload (range).

The conventions, the same in the prompt, the board and the clips:
- No reading is black: a stored 0, NaN, an infinity, or 65535 in a 16-bit stream (what sensors write where they could
  not measure). Never a distance.
- Depth in metres is drawn on ONE fixed scale for every episode of every dataset: turbo on a log scale from
  METRIC_NEAR_M (red) to METRIC_FAR_M (blue). A colour is a distance: the same orange is the same 0.5 m in a head
  camera's depth and in a scene camera's, in this upload and the next. The log scale keeps a hand at 0.3 m and a wall at
  3 m both readable.
- Depth whose unit the recording does not state is drawn with the same colours, near red and far blue, scaled across
  the whole upload for that camera (range: the 0.5th to 99.5th percentile of its readings), so a colour means the same
  reading in every episode of the upload, though not a distance in metres.
- The colours stop short of turbo's near-black far end (FAR_STOP), so no reading is the only black in a picture.
- Disparity is drawn with its values reversed, so near is red in both.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

# matplotlib's turbo, 9 stops, enough for a smooth ramp by linear interpolation (index 0 = blue far, 8 = red near)
TURBO = np.array([[0.19, 0.07, 0.23], [0.27, 0.42, 0.89], [0.16, 0.73, 0.92], [0.18, 0.95, 0.60], [0.64, 0.99, 0.24],
                  [0.95, 0.80, 0.23], [0.98, 0.49, 0.13], [0.82, 0.19, 0.03], [0.48, 0.02, 0.01]])
METRIC_NEAR_M, METRIC_FAR_M = 0.1, 10.0
FAR_STOP = 0.12              # the far end of the colours, kept off turbo's near-black so only no reading is black
METRIC_TICKS_M = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)


def load(ep_dir: Path) -> dict:
    """{view: entry} from depth.json, each with its kmap loaded ("km"); {} when the episode has no depth."""
    p = Path(ep_dir) / "depth.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    tz = np.load(Path(ep_dir) / "depth_times.npz") if (Path(ep_dir) / "depth_times.npz").exists() else None
    for v, e in d.items():
        e["km"] = np.load(Path(ep_dir) / e["kmap"])
        if tz is not None and f"depth_{v}_pts" in tz.files:
            e["pts"], e["t"] = tz[f"depth_{v}_pts"], tz[f"depth_{v}"]
    return d


def decode(entry: dict, pts_all, idx: list[int]) -> dict:
    """{index: uint16 array} for the depth frames idx (indices into the stream), each found at its exact pts. A frame
    that is not found is left out, never replaced by a neighbour."""
    import av
    want = {int(pts_all[i]): int(i) for i in sorted(set(int(i) for i in idx)) if 0 <= int(i) < len(pts_all)}
    out = {}
    if not want:
        return out
    with av.open(str(entry["packed"])) as c:
        s = c.streams.video[0]
        s.codec_context.thread_count = 1
        for p in sorted(want):
            if want[p] in out:
                continue
            c.seek(p, stream=s, backward=True, any_frame=False)
            for fr in c.decode(s):
                if fr.pts is None:
                    continue
                if fr.pts in want and want[fr.pts] not in out:
                    out[want[fr.pts]] = _array(fr)
                if fr.pts >= p:
                    break
    return out


def _array(fr) -> np.ndarray:
    name = fr.format.name
    if name.startswith("gray16") or name.startswith("gray12") or name.startswith("gray10"):
        return fr.to_ndarray().astype(np.uint16)
    if name in ("gray", "gray8"):
        return fr.to_ndarray().astype(np.uint16)
    return fr.to_ndarray(format="gray16le").astype(np.uint16)


def at_anchor(ep: dict, depth: dict, view: str, ks: list[int]) -> dict:
    """{anchor frame k: depth array} for camera view at anchor frames ks, through its depth kmap."""
    e = depth[view]
    pts = e.get("pts") if e.get("pts") is not None else (ep.get("times") or {}).get(f"depth_{view}_pts")
    if pts is None:
        return {}
    own = {int(k): int(e["km"][int(k)]) for k in ks if int(k) < len(e["km"])}
    got = decode(e, pts, list(own.values()))
    return {k: got[j] for k, j in own.items() if j in got}


def valid(a: np.ndarray) -> np.ndarray:
    """Where a depth frame holds a reading: finite, above 0, and not 65535 in a 16-bit stream."""
    a = np.asarray(a)
    ok = np.isfinite(a) & (a > 0)
    if a.dtype == np.uint16:
        ok &= a != 65535
    return ok


def scale_range(frames) -> tuple[float, float] | None:
    """(low, high) stored values of a set of depth frames: the 0.5th and 99.5th percentile of their readings (valid).
    prepare/formats.py measures it across a whole upload for depth with no stated unit. None when there are none."""
    vals = [f[valid(f)].ravel()[::7] for f in frames if f is not None and valid(f).any()]
    if not vals:
        return None
    v = np.concatenate(vals).astype(np.float64)
    lo, hi = float(np.percentile(v, 0.5)), float(np.percentile(v, 99.5))
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def _ramp(x: np.ndarray) -> np.ndarray:
    """turbo at x in 0..1 (0 far blue, 1 near red), as RGB 0..1, from FAR_STOP up."""
    pos = (FAR_STOP + (1 - FAR_STOP) * np.clip(x, 0, 1)) * (len(TURBO) - 1)
    i = np.clip(pos.astype(np.int32), 0, len(TURBO) - 2)
    f = (pos - i)[..., None]
    return TURBO[i] * (1 - f) + TURBO[i + 1] * f


def metric_position(m: np.ndarray) -> np.ndarray:
    """Where a distance in metres sits on the fixed scale: 1 at METRIC_NEAR_M, 0 at METRIC_FAR_M, log in between."""
    with np.errstate(all="ignore"):
        return 1.0 - (np.log(np.clip(m, METRIC_NEAR_M, METRIC_FAR_M)) - np.log(METRIC_NEAR_M)) / \
            (np.log(METRIC_FAR_M) - np.log(METRIC_NEAR_M))


def picture(a: np.ndarray, entry: dict, rng: tuple[float, float] | None = None):
    """One depth frame as an RGB PIL image, by the module's conventions: metric on the fixed log turbo scale, any other
    in upload-scaled grey (entry["range"], else rng), disparity reversed, no reading black."""
    from PIL import Image
    a = np.asarray(a)
    ok = valid(a)
    x = a.astype(np.float64)
    disparity = entry.get("kind") == "disparity"
    if entry.get("scale_m") and not disparity:
        rgb = _ramp(metric_position(x * float(entry["scale_m"])))
    else:
        lo, hi = entry.get("range") or rng or (scale_range([a]) or (0.0, 1.0))
        t = np.clip((x - lo) / max(hi - lo, 1e-9), 0, 1)
        rgb = _ramp(t if disparity else 1.0 - t)
    rgb[~ok] = 0.0
    return Image.fromarray((rgb * 255).astype(np.uint8), "RGB")


def colorize(a: np.ndarray, rng: tuple[float, float], scale_m: float | None = None):
    """picture() for a bare range and scale (an older call site)."""
    return picture(a, {"scale_m": scale_m, "range": rng})


def legend(entry_or_rng, scale_m: float | None = None) -> str:
    """The words for a camera's depth colours."""
    entry = entry_or_rng if isinstance(entry_or_rng, dict) else {"scale_m": scale_m}
    if entry.get("scale_m"):
        return (f"metric: one fixed scale in every episode, dark red {METRIC_NEAR_M:g} m, orange 0.3 m, yellow 0.5 m, "
                f"green 1 m, light blue 3 m, dark blue {METRIC_FAR_M:g} m (a log scale), black no reading")
    return ("red near to blue far, relative to this camera's readings across the upload (the recording does not say "
            "its unit, so not in metres); black no reading")


def legend_ticks(entry: dict) -> list[tuple[float, str]]:
    """[(position 0..1 along the colour bar, far to near, label)] for a camera's depth colours."""
    if entry.get("scale_m") and entry.get("kind") != "disparity":
        return [(float(1.0 - metric_position(np.array(m))), f"{m:g} m") for m in METRIC_TICKS_M]
    return [(0.0, "far"), (1.0, "near")]


def to_jpeg(im, quality: int = 88) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
