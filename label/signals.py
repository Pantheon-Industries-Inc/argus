"""Reading a recording's other signals (prepare/formats.py Signals): what each one does over an episode, in numbers a
person or a model can check against the frames.

A signal is read by how its numbers behave, and only whether it measures touch also asks its name (is_touch):

- Its resting level, per value. A vector of SMALL values or fewer (a force, a flag, a position) rests at the end of
  each value's range it sits near most of the time (its 5th or 95th percentile, when its median is within a quarter
  of the range of that end), else at its median: a force sensor at its offset, an IMU or a position at its middle. An
  array (a pressure map) is read as one sensor: its values pooled say which end it rests at (OpenTouch's raw glove
  reads 3072 untouched and less when pressed, so most of its readings sit at the top), and each value rests at its own
  extreme on that end (its 1st or 99th percentile), so a cell pressed through most of a clip still rests where it
  reads untouched. An array whose pooled values sit in the middle (hand landmarks) rests at each value's median.
- A value is active when it is away from rest by more than REST_FRACTION of the array's typical swing (the 99th
  percentile of every value's distance from rest), so the noise of many idle cells never adds up to a contact. The
  signal's activity at a frame is the summed distance of its active values, and its peak the largest in the episode.
- It has a rest only when it really holds one level: at least HOLD_SHARE of its readings lie within HOLD_BAND of its
  typical swing from their resting level (an untouched cell reads 3072 frame after frame). A position or an IMU,
  whose values drift through their range, has no rest, whatever its percentiles say.
- Touch is bounded at rest: a pressure moves away from its unloaded reading one way and never reads past it except
  by noise. A hand's position can hold one level for a while and then leave it one way, but it also moves past that
  level, so it is not touch (direction). A pad read in whole numbers that only flickers by one step has measured
  nothing but the rounding of its readout (rounding_only).
- It rests and rises when it has a rest, and its activity is under REST_FRACTION of its peak for at least MIN_REST of
  the episode and above it for at least MIN_ACTIVE: a tactile glove's pressure, a fingertip's force, a contact flag, a
  gripper's effort. For such a signal the spans it is away from rest are reported with their exact start and end, from
  every recorded frame rather than only the sampled instants, because that is when a hand or a gripper touches
  something.
- An array of more than SMALL values per frame is shown as numbers that summarise it. One that has a rest and whose
  activity is local
  (at a typical frame fewer than half its values are active: a pressure map under a grasp, whether or not it ever
  rests) gets its total activity, how many values are active and, for a 2-D array, where its strongest value is; any
  other (hand landmarks, whose values all move together) gets how much its values changed since the instant before.
"""
from __future__ import annotations

import warnings

import numpy as np

REST_FRACTION = 0.1
MIN_REST = 0.2
MIN_ACTIVE = 0.02
SMALL = 4
# a signal of this many values or fewer is described value by value (each value's range, or its constant values); the
# widest a prompt showed value by value before arrays were read as wholes, so a 14-joint action or a 9-value camera
# matrix keeps every number, while a pressure map or hand landmarks get the pooled range of all their values
PER_VALUE_MAX = 64
MERGE_GAP_S = 0.15
HOLD_BAND = 0.02
SETTING_STATES = 3        # a signal of several values with this many distinct readings or fewer is a setting
HOLD_SHARE = 0.3


def _num(x: float) -> str:
    return "-" if not np.isfinite(x) else f"{float(x):.3g}"


def _pooled_end(a: np.ndarray) -> str | None:
    """"low" or "high" when an array's values pooled sit near one end of their range (a pressure map), else None."""
    v = a[np.isfinite(a)]
    if not len(v):
        return None
    lo, med, hi = np.percentile(v, [1, 50, 99])
    if not hi > lo:
        return None
    pos = (med - lo) / (hi - lo)
    return "low" if pos <= 0.25 else "high" if pos >= 0.75 else None


def resting_level(a: np.ndarray, rest=None) -> np.ndarray:
    """Per value, where the signal rests (module docstring); rest, the upload's own (prepare/formats.py
    measure_signal_scales), is used as given when it has one value per value."""
    if rest is not None and len(rest) == a.shape[1]:
        return np.asarray(rest, dtype=np.float64)
    if not np.isfinite(a).any():
        return np.zeros(a.shape[1])
    with np.errstate(all="ignore"):
        if a.shape[1] > SMALL:
            end = _pooled_end(a)
            if end == "low":
                return np.nan_to_num(np.nanpercentile(a, 1, axis=0))
            if end == "high":
                return np.nan_to_num(np.nanpercentile(a, 99, axis=0))
            return np.nan_to_num(np.nanmedian(a, axis=0))
        lo, med, hi = np.nanpercentile(a, [5, 50, 95], axis=0)
    span = hi - lo
    pos = np.where(span > 0, (med - lo) / np.where(span > 0, span, 1), 0.5)
    return np.nan_to_num(np.where(pos <= 0.25, lo, np.where(pos >= 0.75, hi, med)))


def direction(a: np.ndarray, rest=None, swing=None) -> str | None:
    """How a signal moves away from rest: "up" when its active values rise above their resting levels, "down" when
    they fall below them, else None (both, never active, or not bounded at rest). A signal moving one way is bounded
    at rest when on the other side its readings go no further than REST_FRACTION of how far it moves: a pressure
    never reads past its unloaded value except by noise, while a position that holds one level for a while moves past
    it whenever the hand does."""
    dev = a - resting_level(a, rest)
    act = _active(a, rest, swing)
    if not act.any():
        return None
    up = float((dev[act] > 0).mean())
    way = "up" if up >= 0.8 else "down" if up <= 0.2 else None
    if way is None:
        return None
    toward = (dev if way == "up" else -dev)[np.isfinite(dev)]
    beyond = float(np.percentile(np.maximum(-toward, 0.0), 99))
    return way if beyond <= REST_FRACTION * float(toward.max()) else None


def rounding_only(a: np.ndarray, rest=None, swing=None) -> bool:
    """Whether a map of many levels has recorded nothing but the rounding of its readout: its typical swing is no
    more than the smallest step between its readings, so its cells only flip between neighbouring readings (an
    untouched pad read in whole numbers flickers between 0 and 1). A map of a few readings (taxels that are on or
    off) moves by one step when pressed, so it is never rounding only."""
    v = a[np.isfinite(a)]
    u = np.unique(v)
    if a.shape[1] <= SMALL or len(u) <= SETTING_STATES:
        return False
    _, sw = _distance(a, rest, swing)
    return sw <= float(np.diff(u).min()) * (1 + 1e-6)


def _distance(a: np.ndarray, rest=None, swing=None) -> tuple[np.ndarray, float]:
    """(every value's distance from rest, the typical swing: the upload's when given, else the 99th percentile of these
    distances)."""
    d = np.abs(a - resting_level(a, rest))
    if swing is not None and swing > 0:
        return d, float(swing)
    ok = d[np.isfinite(d)]
    return d, (float(np.percentile(ok, 99)) if len(ok) else 0.0)


def _active(a: np.ndarray, rest=None, swing=None) -> np.ndarray:
    d, sw = _distance(a, rest, swing)
    return np.nan_to_num(d) > REST_FRACTION * sw if sw > 0 else np.zeros(a.shape, dtype=bool)


def activity(a: np.ndarray, rest=None, swing=None) -> np.ndarray:
    """Per frame, the summed distance from rest of the signal's active values (NaN where it has no reading)."""
    d, sw = _distance(a, rest, swing)
    m = np.nansum(np.where(np.nan_to_num(d) > REST_FRACTION * sw, d, 0.0), axis=1) if sw > 0 else np.zeros(len(a))
    m = m.astype(np.float64)
    m[np.isnan(a).all(axis=1)] = np.nan
    return m


def has_rest(a: np.ndarray, rest=None, swing=None) -> bool:
    """Whether a signal holds one level: HOLD_SHARE of its readings within HOLD_BAND of its swing from rest."""
    d, sw = _distance(a, rest, swing)
    ok = d[np.isfinite(d)]
    return sw > 0 and len(ok) > 0 and float((ok <= HOLD_BAND * sw).mean()) >= HOLD_SHARE


def localized(a: np.ndarray, rest=None, swing=None) -> bool:
    """Whether an array has a rest and its activity is local: at a typical active frame fewer than half its values are
    active."""
    if not has_rest(a, rest, swing):
        return False
    act = _active(a, rest, swing)
    rows = act.any(axis=1)
    return bool(rows.any()) and float(np.median(act[rows].mean(axis=1))) < 0.5


def rests_and_rises(a: np.ndarray, rest=None, swing=None) -> bool:
    if not has_rest(a, rest, swing):
        return False
    m = activity(a, rest, swing)
    ok = m[np.isfinite(m)]
    if len(ok) < 5 or not ok.max() > 0:
        return False
    on = ok >= REST_FRACTION * ok.max()
    return (~on).mean() >= MIN_REST and on.mean() >= MIN_ACTIVE


def touch_like(a: np.ndarray, rest=None, swing=None) -> bool:
    """Whether a signal behaves the way touch does: it has a rest, moves away from it in one direction only and is
    bounded there (pressure rises, a raw glove reading falls, and neither reads past untouched), and either rests and
    rises or is a local array. A joint velocity rests at zero too, but swings both ways, so it is not touch; a hand's
    position can hold one level and leave it one way in an episode, but it also moves past that level, so it is not
    either; a camera's calibration that switches between two settings rests and moves one way too, but a sensor of
    several values never takes only SETTING_STATES readings; and a pad that only flickers by one step of its readout
    (rounding_only) has measured nothing."""
    a = np.asarray(a, dtype=np.float64)
    if not len(a) or not has_rest(a, rest, swing) or direction(a, rest, swing) is None:
        return False
    if a.shape[1] > 1 and len(np.unique(a[np.isfinite(a).all(axis=1)], axis=0)) <= SETTING_STATES:
        return False  # several values that only ever take a few readings together: a setting switching, not a sensor
    if rounding_only(a, rest, swing):
        return False
    return rests_and_rises(a, rest, swing) or (a.shape[1] > SMALL and localized(a, rest, swing))


def is_touch(name: str, a: np.ndarray, rest=None, swing=None) -> bool:
    """Whether a signal measures touch: its own name says so (prepare/formats.py names_touch: tactile, pressure,
    contact, force, and never a command) and its numbers behave like touch (touch_like). Numbers alone cannot
    decide it: a humanoid's torso joint that holds still and then moves one way, a mobile base's odometry, an action
    or a pose rest and rise like a pressure pad, and only the name says which one measures touch. A touch signal
    whose name says nothing (ch0) is not read as touch until a data dictionary can say it is."""
    from prepare.formats import names_touch
    return names_touch(name) and touch_like(a, rest, swing)


def active_spans(a: np.ndarray, t: np.ndarray, rest=None, swing=None) -> list[tuple[float, float]]:
    """[(start s, end s)] where the signal is away from rest (activity at least REST_FRACTION of its peak), from every
    frame; spans closer than MERGE_GAP_S are one, and a single-frame flicker is not a span."""
    m = activity(a, rest, swing)
    if not np.isfinite(m).any() or not np.nanmax(m) > 0:
        return []
    on = np.nan_to_num(m) >= REST_FRACTION * np.nanmax(m)
    spans, i, n = [], 0, len(on)
    while i < n:
        if not on[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and on[j + 1]:
            j += 1
        if j > i:
            if spans and t[i] - spans[-1][1] <= MERGE_GAP_S:
                spans[-1] = (spans[-1][0], float(t[j]))
            else:
                spans.append((float(t[i]), float(t[j])))
        i = j + 1
    return spans


MOVING_MIN = 10.0     # a value moves when its range is at least this many times its typical step between frames
QUIET_SHARE = 0.01    # a quiet stretch: every moving value stays within this share of its own range


def movements(a: np.ndarray) -> np.ndarray:
    """How much each value of a signal moves over the episode, unit free: its range over its typical (median) step
    between consecutive frames, so a joint that sweeps its range or a flag that switches scores high and a value that
    only jitters by its noise scores near 1. 0 for a value that never changes or has fewer than 3 readings."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = np.nanmax(a, axis=0) - np.nanmin(a, axis=0)
        step = np.nanmedian(np.abs(np.diff(a, axis=0)), axis=0) if len(a) > 1 else np.full(a.shape[1], np.nan)
        m = r / (step + 1e-3 * r)
    ok = (np.isfinite(a).sum(axis=0) >= 3) & (r > 0) & np.isfinite(m)
    return np.where(ok, m, 0.0)


def quiet_spans(arrs: dict, need: int) -> list[tuple[int, int]]:
    """[(a, b)] frame spans of at least need frames over which every moving value of every signal (movements at least
    MOVING_MIN) stays within QUIET_SHARE of its own range: where a recording with no arm state shows nothing moving.
    Used only to choose instants (label/episode.py plan), never told to the model as a claim. A value with a frame
    that has no reading is left out of the test."""
    from label import state as ms
    cols = []
    for a in arrs.values():
        a = np.asarray(a, dtype=np.float64)
        a = a[:, None] if a.ndim == 1 else a
        mv = movements(a)
        for j in np.flatnonzero(mv >= MOVING_MIN):
            v = a[:, j]
            if np.isfinite(v).all():
                cols.append(v / float(v.max() - v.min()))
    if not cols:
        return []
    s = np.stack(cols, axis=1)
    return ms.spans_within(s, np.full(s.shape[1], QUIET_SHARE), need)


def summary_rows(name: str, a: np.ndarray, ks: list[int], shape=None, names=None, rest=None,
                 swing=None) -> list[tuple[str, list[str]]]:
    """[(row label, one value per sampled instant)] for one signal: its values when it has SMALL or fewer, else the
    numbers that summarise the array (module docstring). "-" is no reading at that instant."""
    a = np.asarray(a, dtype=np.float64)
    d = a.shape[1]
    if d <= SMALL:
        labels = names if names and len(names) == d else ([""] if d == 1 else [f"[{i}]" for i in range(d)])
        return [(f"{name}{(' ' + lb) if lb else ''}", [_num(a[k, i]) for k in ks]) for i, lb in enumerate(labels)]
    rows = []
    if localized(a, rest, swing):
        dist, _ = _distance(a, rest, swing)
        act = _active(a, rest, swing)
        m = activity(a, rest, swing)
        rows.append((f"{name} total activity", [_num(m[k]) for k in ks]))
        rows.append((f"{name} active values", ["-" if np.isnan(a[k]).all() else str(int(act[k].sum())) for k in ks]))
        if shape and len(shape) == 2:
            h, w = int(shape[0]), int(shape[1])
            cells = []
            for k in ks:
                if np.isnan(a[k]).all() or not act[k].any():
                    cells.append("-")
                else:
                    r, c = divmod(int(np.nanargmax(np.where(act[k], dist[k], -1))), w)
                    cells.append(f"r{r}c{c}")
            rows.append((f"{name} strongest cell (row r, column c of {h} x {w})", cells))
    else:
        ch = ["-"]
        for a0, b0 in zip(ks, ks[1:]):
            x = np.abs(a[b0] - a[a0])
            ch.append("-" if not np.isfinite(x).any() else _num(np.nanmax(x)))
        rows.append((f"{name} largest change of any value since the instant before", ch))
    return rows


def describe(name: str, a: np.ndarray, shape=None, names=None, rest=None, swing=None) -> str:
    """One line: the signal's name, its shape or value names, and the range each value takes, up to PER_VALUE_MAX
    values (for a wider array, the range of all its values together)."""
    a = np.asarray(a, dtype=np.float64)
    d = a.shape[1]
    what = (f"{' x '.join(str(int(x)) for x in shape)} values" if shape and len(shape) > 1 else
            f"{d} value{'s' if d > 1 else ''}")
    if names and len(names) == d and d <= PER_VALUE_MAX:
        what += " (" + ", ".join(names) + ")"
    head = f"  {name} ({what})"
    if not np.isfinite(a).any():
        return f"{head}: no reading"
    gaps = int(np.isnan(a).all(axis=1).sum())
    tail = f"; no reading at {gaps} of {len(a)} frames" if gaps else ""
    with np.errstate(all="ignore"):
        lo, hi = np.nanmin(a, axis=0), np.nanmax(a, axis=0)
    if d > PER_VALUE_MAX:
        if (hi == lo).all():
            return f"{head}: every value constant throughout{tail}"
        return f"{head}: values from {_num(np.nanmin(lo))} to {_num(np.nanmax(hi))}{tail}"
    if (hi == lo).all():
        value = _num(lo[0]) if d == 1 else "[" + ", ".join(_num(x) for x in lo) + "]"
        return f"{head}: {value} throughout{tail}"
    return f"{head}: " + ", ".join(_num(l) if l == h else f"{_num(l)} to {_num(h)}" for l, h in zip(lo, hi)) + tail
