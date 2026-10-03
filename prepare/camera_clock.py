"""An explicit display placement for camera frames whose recorded clocks cannot order them."""
from pathlib import Path

import numpy as np

# No recorded camera cadence can be recovered when every stamp is the same. This is a named display fallback.
UNKNOWN_CAMERA_FPS = 30.0


def coarse_rows(t: np.ndarray) -> tuple[np.ndarray, float | None]:
    """Place one stream's tied rows in order within its next stamp interval, evenly as an assumption. The last
    interval uses the median positive distinct step. Increasing clocks pass through unchanged. Tables and camera
    display clocks share this rule; neither placement establishes precise recorded instants."""
    t = np.asarray(t, dtype=np.float64)
    if len(t) < 2 or not np.isfinite(t).all():
        return t, None
    d = np.diff(t)
    if not (d >= 0).all() or not (d == 0).any() or not (d > 0).any():
        return t, None
    resolution = float(np.median(d[d > 0]))
    starts = np.r_[0, np.flatnonzero(d > 0) + 1]
    out = t.copy()
    for a, b in zip(starts, np.r_[starts[1:], len(t)]):
        width = float(t[b] - t[a]) if b < len(t) else resolution
        out[a:b] = t[a] + np.arange(b - a) * width / (b - a)
    return out, resolution


def presentation_clock(t: np.ndarray, nominal_fps: float | None = None) -> tuple[np.ndarray, dict | None]:
    """Camera times and an explicit assumed placement note for tied or unusable recorded clocks.
    Ties share the table interval rule. An all tied clock uses its declared nominal rate when supplied, otherwise
    the named unknown cadence fallback. The original input is never changed and encoded PTS never enter this rule."""
    t = np.asarray(t, dtype=np.float64)
    if not len(t):
        return t, None
    finite = np.isfinite(t).all()
    d = np.diff(t) if finite else np.empty(0)
    problem = "nonfinite" if not finite else "backwards" if (d < 0).any() else None
    if problem:
        declared = (nominal_fps is not None and not isinstance(nominal_fps, bool)
                    and np.isfinite(nominal_fps) and nominal_fps > 0)
        fps = float(nominal_fps) if declared else UNKNOWN_CAMERA_FPS
        source = "declared nominal cadence" if declared else "unknown cadence fallback"
        start = float(t[0]) if np.isfinite(t[0]) else 0.0
        what = "nonfinite values" if problem == "nonfinite" else "backwards steps"
        return start + np.arange(len(t)) / fps, {"method": "nominal cadence", "nominal_fps": fps,
            "clock_problem": problem, "cadence_source": source,
            "what": f"the recorded camera clock contains {what} and cannot order all frames; their assumed "
                    f"presentation times place every frame in row order at {fps:g} Hz from {source}, never a "
                    "measured capture rate. A nonfinite first stamp uses display zero. Original capture times "
                    "and encoded PTS are kept."}
    if len(t) < 2 or not (d == 0).any():
        return t, None
    placed, resolution = coarse_rows(t)
    if resolution is not None:
        return placed, {"method": "stamp intervals", "resolution_s": resolution,
                        "what": "distinct frames share recorded timestamps; their assumed presentation times "
                                "spread tied frames in order within each stamp interval, with the final interval "
                                "using the median distinct step. Original capture times and encoded PTS are kept."}
    declared = (nominal_fps is not None and not isinstance(nominal_fps, bool)
                and np.isfinite(nominal_fps) and nominal_fps > 0)
    fps = float(nominal_fps) if declared else UNKNOWN_CAMERA_FPS
    source = "declared nominal cadence" if declared else "unknown cadence fallback"
    return t[0] + np.arange(len(t)) / fps, {"method": "nominal cadence", "nominal_fps": fps,
            "cadence_source": source, "what": f"all frames share one recorded timestamp; their assumed presentation "
            f"times use {fps:g} Hz from {source}, never a measured capture rate. Original capture times and encoded "
            "PTS are kept."}


def load_times(ep_dir: Path, ctx: dict, recorded: bool = False) -> dict:
    """Stored camera times and exact encoded PTS. Only an explicitly opted in episode reads separate presentation
    times. Unmarked saved episodes keep their old label clock. recorded=True always reads the untouched originals."""
    with np.load(Path(ep_dir) / (ctx.get("real_times") or "times.npz")) as source:
        times = {key: source[key] for key in source.files}
    if not recorded and ctx.get("presentation_times"):
        with np.load(Path(ep_dir) / ctx["presentation_times"]) as presentation:
            for key in presentation.files:
                if key in times and len(presentation[key]) == len(times[key]):
                    times[key] = presentation[key]
                else:
                    raise ValueError(f"presentation times do not match the recorded frames of {key}")
    return times


def pairing_times(ep_dir: Path, ctx: dict) -> dict:
    """Select both camera and depth clocks for frame pairing, preserving their exact encoded PTS.
    A camera's presentation opt in applies independently of whether depth needs its own presentation array.
    Unmarked episodes retain their recorded pairing clocks."""
    times = load_times(ep_dir, ctx)
    path = Path(ep_dir) / "depth_times.npz"
    if path.exists():
        with np.load(path) as source:
            times.update({key: source[key] for key in source.files})
    if ctx.get("depth_presentation_times"):
        with np.load(Path(ep_dir) / ctx["depth_presentation_times"]) as presentation:
            for key in presentation.files:
                if key not in times or len(presentation[key]) != len(times[key]):
                    raise ValueError(f"depth presentation times do not match the recorded frames of {key}")
                times[key] = presentation[key]
    return times


def unshown_times(ep_dir: Path, entry: dict, zero_s: float = 0.0) -> tuple[np.ndarray, np.ndarray] | None:
    """An opted in side camera's presentation clock and original packets for its explicit frame span.
    Offsets move display placement onto a part or replacement anchor clock without changing the source sidecar."""
    if not entry.get("camera_times"):
        return None
    with np.load(Path(ep_dir) / entry["camera_times"]) as z:
        a, b = entry.get("frame_span") or (0, len(z["presentation"]))
        offset = float(entry.get("camera_clock_offset_s") or 0.0) + zero_s
        return z["presentation"][a:b] - offset, z["pts"][a:b]


def unshown_span(ep_dir: Path, entries: list, t0: float, t1: float, zero_s: float = 0.0,
                 shift: bool = False) -> list:
    """Select an opted in side camera's original packets within a display interval, preserving every raw source
    timestamp. A part shifts that interval to its own zero; a trim keeps the current clock."""
    out = []
    for entry in entries:
        clocks = unshown_times(ep_dir, entry, zero_s)
        if clocks is None:
            out.append(entry)
            continue
        times, _ = clocks
        a = (entry.get("frame_span") or [0])[0]
        lo, hi = (int(np.searchsorted(times, edge, side="left")) for edge in (t0, t1))
        e = {**entry, "camera_times": str((Path(ep_dir) / entry["camera_times"]).resolve()),
             "frame_span": [a + lo, a + hi], "n_frames": hi - lo}
        if shift:
            e["camera_clock_offset_s"] = float(entry.get("camera_clock_offset_s") or 0.0) + zero_s + t0
            e["start_s"] = float(times[lo] - t0) if hi > lo else 0.0
        out.append(e)
    return out


def camera_issue_kind(note: dict) -> str:
    """The recorded problem, distinct from the assumption used to display its frames."""
    return "camera_timestamp_invalid" if note.get("clock_problem") else "camera_timestamp_repeated"
