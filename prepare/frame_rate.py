"""The one way any code reads a frame rate. tests/test_frame_rate_windows.py fails on a read of an "fps" key or a frame
count over a rate anywhere else in label, board, checks, compare, gate, review or prepare.

- episode_fps: a prepared episode's frame rate, the stated fps unless the anchor camera's own capture times contradict
  it (supported_rate), with a note when they did. Every reader of an episode goes through it, so an episode prepared
  with a wrong stated rate is read at its frames' rate with no re-prepare.
- frames_seconds: an episode's length from its frame count at that rate, for a context with no duration_s.
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


def span_rate(t) -> float | None:
    """Frames per second over a clock's own span, (n - 1) / (last - first) of its finite times in seconds, or None
    without two of them a positive time apart."""
    t = np.asarray(t, dtype=np.float64).ravel()
    t = t[np.isfinite(t)]
    span = float(t.max() - t.min()) if len(t) >= 2 else 0.0
    return (len(t) - 1) / span if span > 0 else None


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
    """An episode's length as its frame count (n_state_frames) at its frame rate (episode_fps), 0 without frames."""
    rate = episode_fps(ctx, ep_dir=ep_dir, times=times, default=default)[0]
    n = (ctx or {}).get("n_state_frames") or 0
    return n / float(rate) if rate else 0.0
