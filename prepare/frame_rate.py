"""The one way any code reads a frame rate. tests/test_frame_rate_windows.py fails on a read of an "fps" key or a frame
count over a rate anywhere else in label, board, checks, compare, gate, review or prepare.

- episode_fps: a prepared episode's frame rate, the stated fps unless the anchor camera's own capture times contradict
  it (supported_rate), with a note when they did. Every reader of an episode goes through it, so an episode prepared
  with a wrong stated rate is read at its frames' rate with no re-prepare.
- episode_seconds: an episode's length. Its stated duration_s, else the span of the anchor camera's capture times,
  and only without either its frame count at episode_fps (frames_seconds). Every reader of a length goes through it,
  so no reader divides a frame count by a rate when the capture times say how long the footage is.
- measured_rate: a camera's frame rate measured from its capture times, as an adapter writes it into a context. The
  median step, unless the frames over the time they were recorded say otherwise.
- capture_seconds: an episode's length from its cameras' capture times, as an adapter writes duration_s.
- declared_fps: the rate a source declares for its own frames before any episode exists (a dataset's info.json, a
  video probe), or that a camera entry stores, read as written. Never an episode context's: that is episode_fps.
- stored_fps: the episode context's stored value as written, only to fingerprint or report the stored inputs, never to
  count frames with.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

KEY = "fps"

# A stated frame rate gives way to the rate its frames' own capture times show only past this factor either way. That
# is a clock or units fault, never dropped frames or a pause: ABC-130k's top camera stamps about half its frames 1 us
# after the frame before, so the median step reads 976562.5 fps for a recording of 3077 frames over 103 s (29.8 fps).
FRAME_RATE_MISMATCH = 10.0

# A rate measured from capture times is the median step's only while the frames over the time they were recorded agree
# with it to within this fraction. Past it the median step is no frame interval: ABC-130k stations stamp frames 1, 2 or
# 3 frame intervals apart, so a median step of 1/60 s sits on a camera whose frames come at 30 fps over their span, and
# 249 of 3,417 prepared episodes counted their length from such a rate.
SPAN_AGREEMENT = 0.02
# A step of at least PAUSE_S seconds and PAUSE_STEPS median steps is a pause in the recording, not a frame interval, so
# a camera that paused is measured at the rate it recorded at, not over the pause.
PAUSE_S = 1.0
PAUSE_STEPS = 10.0


def span_rate(t) -> float | None:
    """Frames per second over a clock's own span, (n - 1) / (last - first) of its finite times in seconds, or None
    without two of them a positive time apart."""
    t = np.asarray(t, dtype=np.float64).ravel()
    t = t[np.isfinite(t)]
    span = float(t.max() - t.min()) if len(t) >= 2 else 0.0
    return (len(t) - 1) / span if span > 0 else None


def measured_rate(t, *, per_second: float = 1.0, min_step: float = 0.0, digits: int = 3) -> float | None:
    """A camera's frame rate from its capture times t (per_second units to a second: 1e9 for nanoseconds), or None
    without two times more than min_step apart. It is the median step's rate, rounded to digits, while the frames'
    rate over the time they were recorded (their steps over the steps' total, pauses left out) agrees with it to within
    SPAN_AGREEMENT. Otherwise the median step is no frame interval and that recorded rate is used: a 60 Hz median
    step on frames that come at 30 fps, or a 1 us median step on frames that come at 30 fps."""
    t = np.asarray(t, dtype=np.float64).ravel()
    t = t[np.isfinite(t)]
    d = np.diff(t)
    stepped = d[d > min_step]
    if not len(stepped):
        return None
    median = float(np.median(stepped))
    rate = round(per_second / median, digits)
    recorded = d[(d >= 0) & (d <= max(PAUSE_S * per_second, PAUSE_STEPS * median))]
    total = float(recorded.sum())
    if not len(recorded) or total <= 0:
        return rate
    own = len(recorded) * per_second / total
    return rate if abs(rate / own - 1.0) <= SPAN_AGREEMENT else round(own, 3)


def capture_seconds(clocks, rate, zero: float | None = None) -> float | None:
    """An episode's length from its cameras' capture times (seconds, one clock): from zero, the episode clock's origin
    (by default the first frame of any camera), to one frame step (1 / rate) after the last frame of any camera, or None
    without a finite time or a positive rate. This is what an adapter writes as duration_s, so no reader counts the
    length from a rate."""
    finite = [t[np.isfinite(t)] for t in (np.asarray(c, dtype=np.float64).ravel() for c in clocks)]
    finite = [t for t in finite if len(t)]
    rate = _number(rate)
    if not finite or rate is None:
        return None
    start = min(float(t.min()) for t in finite) if zero is None else float(zero)
    return max(float(t.max()) for t in finite) - start + 1.0 / rate


def supported_rate(stated, t) -> tuple:
    """(the frame rate seconds are turned into frames with, a note or None). The stated rate is kept unless the frames'
    capture times t (seconds) show a rate more than FRAME_RATE_MISMATCH times higher or lower. Then the frames' own
    rate is used, because a window or a lag of so many seconds counted at the stated rate would not be that long in
    the recording (a 1 s window at 976562.5 fps is longer than the whole 3077-frame episode). The note names both."""
    if isinstance(stated, bool) or not isinstance(stated, (int, float)) or not np.isfinite(stated) or stated <= 0:
        return stated, None
    own = span_rate(t) if t is not None else None
    if own is None or own / FRAME_RATE_MISMATCH <= float(stated) <= own * FRAME_RATE_MISMATCH:
        return stated, None
    used = round(own, 3)
    return used, {"stated_fps": float(stated), "used_fps": used, "frames": int(np.isfinite(np.asarray(t)).sum()),
                  "what": f"The stated frame rate of {float(stated):.10g} fps is not used, because the frames' own "
                          f"capture times put them at {used:g} fps, more than {FRAME_RATE_MISMATCH:g} times apart; "
                          f"every window and lag in seconds is counted in frames at {used:g} fps."}


def stored_fps(ctx: dict | None):
    """The context's stored fps as written, or None. Only for a fingerprint or a report of the stored inputs."""
    return (ctx or {}).get(KEY)


def declared_fps(entry: dict | None, default=None, *, required: bool = False):
    """The rate a source or a camera entry declares for its own frames, as written, or default when it has none.
    required=True reads it as entry["fps"] does, raising KeyError when it is missing."""
    if required:
        return entry[KEY]
    value = (entry or {}).get(KEY)
    return default if value is None else value


def _number(x) -> float | None:
    """A stored rate as a positive finite float, else None (no value, text, true or false, zero, NaN or infinity)."""
    if isinstance(x, bool):
        return None
    try:
        value = float(x)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def anchor_times(ep_dir, ctx: dict) -> np.ndarray | None:
    """The anchor camera's capture times (seconds) from the prepared episode's real_times, the untouched recorded ones
    when the episode keeps separate presentation times, or None when it has none or they cannot be read."""
    if ep_dir is None or not ctx.get("real_times"):
        return None
    ep_dir = Path(ep_dir)
    try:
        import json
        from label.episode import order_views
        from prepare.camera_clock import load_times
        sources = json.loads((ep_dir / "sources.json").read_text()) if (ep_dir / "sources.json").exists() else {}
        times = load_times(ep_dir, ctx, recorded=True)
        views = order_views(sources or [k for k in times if not k.endswith("_pts")])
        return np.asarray(times[views[0]], dtype=np.float64) if views and views[0] in times else None
    except (OSError, ValueError, KeyError):
        return None


def episode_fps(ctx: dict | None, *, ep_dir=None, times=None, default=30.0) -> tuple:
    """(an episode's frame rate, a note or None). The stated fps (default when it has none) unless the anchor's capture
    times contradict it (supported_rate); times are read from ep_dir when not given. A stated rate is returned as
    written when it is kept, so a reader that shows or hashes it sees the same value as before."""
    stated = stored_fps(ctx)
    if _number(stated) is None:
        return default, None
    if times is None:
        times = anchor_times(ep_dir, ctx or {})
    if not isinstance(stated, (int, float)):
        stated = _number(stated)
    return supported_rate(stated, times)


def frames_seconds(ctx: dict | None, *, ep_dir=None, times=None, default=30.0) -> float:
    """An episode's length as its frame count (n_state_frames) at its frame rate (episode_fps), 0 without frames. Only
    episode_seconds calls it, for an episode with neither a duration_s nor capture times."""
    rate = episode_fps(ctx, ep_dir=ep_dir, times=times, default=default)[0]
    n = (ctx or {}).get("n_state_frames") or 0
    return n / float(rate) if rate else 0.0


def episode_seconds(ctx: dict | None, *, ep_dir=None, times=None, default=30.0) -> float:
    """An episode's length in seconds. Its stated duration_s when that is a positive number. Else the span of the
    anchor camera's capture times (times, or read from ep_dir) plus one step at the frames' own rate, (last - first)
    * n / (n - 1), the length an adapter writes as duration_s (capture_seconds). Only an episode with neither is its
    frame count at its frame rate (frames_seconds). A context written before adapters wrote duration_s, with a rate
    measured from a misleading median step (ABC-130k: 60 fps stated, 30 fps over the span), so reads its true length
    with no re-prepare."""
    stated = _number((ctx or {}).get("duration_s"))
    if stated is not None:
        return stated
    if times is None:
        times = anchor_times(ep_dir, ctx or {})
    if times is not None:
        t = np.asarray(times, dtype=np.float64).ravel()
        t = t[np.isfinite(t)]
        own = span_rate(t)
        if own is not None:
            return len(t) / own
    return frames_seconds(ctx, ep_dir=ep_dir, times=times, default=default)
