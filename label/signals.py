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
- At each sampled instant a signal gets one row per value when per_value says so (SMALL values or fewer, a flat vector
  whose values the dataset names, or one whose own name says joints or a state), and otherwise rows of numbers that
  summarise it (summary_rows). One that has a rest and whose activity is local (at a typical frame fewer than half
  its values are active: a pressure map under a grasp, whether or not it ever rests) gets its total activity, how
  many values are active and, for a 2-D array, where its strongest value is; any other (hand landmarks, whose values
  all move together) gets how much its values changed since the instant before.
"""
from __future__ import annotations

import re
import warnings

import numpy as np

from prepare.signal_alignment import ALIGNED_ASSUMED, ALIGNED_ROWS, ALIGNED_CAMERA, COARSE_CLOCK, placement_text

REST_FRACTION = 0.1
MIN_REST = 0.2
MIN_ACTIVE = 0.02
SMALL = 4
# a signal of this many values or fewer is described value by value (each value's range, or its constant values); the
# widest a prompt showed value by value before arrays were read as wholes, so a 14-joint action or a 9-value camera
# matrix keeps every number, while a pressure map or hand landmarks get the pooled range of all their values
PER_VALUE_MAX = 64
# The values at each instant follow describe for a flat vector of up to PER_VALUE_MAX values whose values the dataset
# names, or whose own name says it holds joints or a state: one row per value under its own name. A humanoid's 26
# joints and a base's x, y, yaw, vx, wz reached that readout as one "largest change" or "total activity" row while
# it went value by value only up to SMALL. A map (a 2 D shape, whose strongest cell says more than its cells one by
# one) and a long unnamed vector (audio samples) keep their summary rows.
JOINT_LIKE = re.compile(r"joint|qpos|state", re.I)
# A signal recorded below this share of the frame rate says its rate: a 6 Hz glove otherwise reads as if it were
# sampled every frame.
RATE_SLOWER = 0.9
# A flat vector whose values are unnamed and that only its own name calls joints or a state gets one row per value up
# to this many values: a humanoid's 26 joints do, an unnamed 21 x 3 hand array named hand_joints (63 values, rows
# labelled only by index) does not. A vector whose values the dataset names keeps the PER_VALUE_MAX limit.
JOINT_NAME_MAX = 32
MERGE_GAP_S = 0.15
HOLD_BAND = 0.02
SETTING_STATES = 3        # a signal of several values with this many distinct readings or fewer is a setting
HOLD_SHARE = 0.3
# A signal is read in pieces of about this many values (4 MB as float64), each in float64, so a large one (a tactile
# skin of 480 MB float32) is never copied whole and every number comes out as a whole float64 copy would give it. Only
# a percentile over all its values (a resting end, the typical swing) holds one array of them while it is taken: float64
# up to POOL_F64_BYTES, so every prompt of a small signal reads as before, and float32 past it, one copy the size of a
# float32 signal, whose percentile is then to float32 precision.
CHUNK_VALUES = 1 << 19
POOL_F64_BYTES = 64 << 20


def _float(a) -> np.ndarray:
    """a as a floating array without a copy when it already is one (a float32 signal stays float32)."""
    a = np.asarray(a)
    return a if np.issubdtype(a.dtype, np.floating) else a.astype(np.float64)


def _row_chunks(a: np.ndarray):
    """(start, end) row ranges of a (frames by values) of about CHUNK_VALUES values each."""
    step = max(1, CHUNK_VALUES // max(1, int(np.prod(a.shape[1:]))))
    return [(i, min(i + step, len(a))) for i in range(0, len(a), step)]


def _col_chunks(a: np.ndarray):
    """(start, end) column ranges of a (frames by values) of about CHUNK_VALUES values each."""
    step = max(1, CHUNK_VALUES // max(1, len(a)))
    return [(j, min(j + step, a.shape[1])) for j in range(0, a.shape[1], step)]


def _rows64(a: np.ndarray):
    """Each row chunk of a as float64, with where it starts."""
    for i, j in _row_chunks(a):
        yield i, np.asarray(a[i:j], dtype=np.float64)


def _pool(a: np.ndarray) -> np.ndarray:
    """An empty array to hold every value of a once (POOL_F64_BYTES says its precision)."""
    return np.empty(a.size, dtype=np.float64 if a.size * 8 <= POOL_F64_BYTES else np.float32)


def _percentile(a: np.ndarray, q, pick):
    """np.percentile(values, q) of the values pick(chunk) gives from each float64 row chunk of a, held in one array of
    them (_pool) and taken in place; None when there are none."""
    buf = _pool(a)
    k = 0
    for _, c in _rows64(a):
        v = pick(c)
        buf[k:k + len(v)] = v
        k += len(v)
    if not k:
        return None
    return np.percentile(buf[:k], q, overwrite_input=True)


def _any_finite(a: np.ndarray) -> bool:
    return any(bool(np.isfinite(a[i:j]).any()) for i, j in _row_chunks(a))


def reading_gaps(a: np.ndarray) -> tuple[int, int]:
    """Counts of wholly missing and partially read frames. A finite value is a reading, so a frame with some
    finite values remains distinct from one with none on the prompt, board and checks."""
    missing = partial = 0
    for i, j in _row_chunks(a):
        fin = np.isfinite(a[i:j])
        any_read = fin.any(axis=1)
        missing += int((~any_read).sum())
        partial += int((any_read & ~fin.all(axis=1)).sum())
    return missing, partial


def gap_words(a: np.ndarray) -> str:
    """The recording gaps, with partial values named separately from frames with no reading."""
    missing, partial = reading_gaps(a)
    return "; ".join(f"{words} at {count} of {len(a)} frames" for count, words in
                     ((missing, "no reading"), (partial, "partial reading")) if count)


def finite_range(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each value's finite range, NaN for a value with no reading. Empty columns never enter a NaN reduction, and
    row chunks bound memory when a signal has many values."""
    lo, hi = np.full(a.shape[1], np.inf), np.full(a.shape[1], -np.inf)
    for _, c in _rows64(a):
        fin = np.isfinite(c)
        lo = np.minimum(lo, np.min(c, axis=0, where=fin, initial=np.inf))
        hi = np.maximum(hi, np.max(c, axis=0, where=fin, initial=-np.inf))
    lo[~np.isfinite(lo)] = np.nan
    hi[~np.isfinite(hi)] = np.nan
    return lo, hi


def per_value(name: str, d: int, shape=None, names=None) -> bool:
    """Whether a signal of d values per frame gets one row per value at each instant (JOINT_LIKE above)."""
    if d <= SMALL:
        return True
    if d > PER_VALUE_MAX or (shape and len(shape) > 1):
        return False
    if names and len(names) == d:
        return True
    return d <= JOINT_NAME_MAX and bool(JOINT_LIKE.search(name))


def names_joints_or_state(name: str) -> bool:
    """Whether a signal's name says it holds joints or a state, as the no state line names it (label/episode.py): a word
    of it is joint or qpos (left_joint1, /left/joint_states, observations/qpos), or its last word is state or states
    (observation.state, observation.leader_state), and no word of it says a command (prepare/formats.py
    COMMAND_WORDS, so /left/joint_command and leader_cmd_state are not). Words are split as prepare/formats.py tokens
    splits them (separators and camelCase). A one value battery_state or estop_state is left out by the line's own
    rule that a named signal has more than one value."""
    from prepare.formats import COMMAND_WORDS, _names_word, tokens
    words = tokens(name)
    return ((_names_word(name, ("joint", "qpos")) or bool(words) and words[-1] in ("state", "states"))
            and not _names_word(name, COMMAND_WORDS))


def names_scalar_observed_state(name: str) -> bool:
    """A scalar can disprove absent arm state only when its name identifies a joint or observed arm state.
    Sensor readings, bookkeeping and commands do not establish that state, even under observation.*."""
    from prepare.formats import TOUCH_WORDS, _names_word, tokens
    if not names_joints_or_state(name) or _names_word(name, TOUCH_WORDS + (
            "torque", "clock", "time", "timestamp",
            "battery", "estop", "emergency", "health", "status", "mode", "power", "temperature", "voltage",
            "current")):
        return False
    words = tokens(name)
    return (_names_word(name, ("joint", "qpos")) or len(words) > 1 and words[-1] in ("state", "states")
            and _names_word(words[-2], ("observation", "arm", "robot", "gripper", "leader", "follower")))


def _num(x: float) -> str:
    return "-" if not np.isfinite(x) else f"{float(x):.3g}"


def _pooled_end(a: np.ndarray) -> str | None:
    """"low" or "high" when an array's values pooled sit near one end of their range (a pressure map), else None."""
    got = _percentile(a, [1, 50, 99], lambda c: c[np.isfinite(c)])
    if got is None:
        return None
    lo, med, hi = got
    if not hi > lo:
        return None
    pos = (med - lo) / (hi - lo)
    return "low" if pos <= 0.25 else "high" if pos >= 0.75 else None


def resting_level(a: np.ndarray, rest=None) -> np.ndarray:
    """Per value, where the signal rests (module docstring); rest, the upload's own (prepare/formats.py
    measure_signal_scales), is used as given when it has one value per value."""
    if rest is not None and len(rest) == a.shape[1]:
        return np.asarray(rest, dtype=np.float64)
    if not _any_finite(a):
        return np.zeros(a.shape[1])
    # a value with no reading at all rests at 0 (nan_to_num); the percentile of its empty column is NaN, said quietly
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if a.shape[1] > SMALL:
            end = _pooled_end(a)
            # value by value, a few thousand values at a time
            per = (lambda c: np.nanpercentile(c, 1, axis=0)) if end == "low" else \
                (lambda c: np.nanpercentile(c, 99, axis=0)) if end == "high" else (lambda c: np.nanmedian(c, axis=0))
            return np.nan_to_num(np.concatenate([per(np.asarray(a[:, i:j], dtype=np.float64))
                                                 for i, j in _col_chunks(a)]))
        a = np.asarray(a, dtype=np.float64)
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
    level = resting_level(a, rest)
    sw = _swing(a, level, swing)
    if not sw > 0:
        return None
    n_act = n_up = 0
    for _, c in _rows64(a):
        dev = c - level
        act = np.nan_to_num(np.abs(dev)) > REST_FRACTION * sw
        n_act += int(act.sum())
        n_up += int((dev[act] > 0).sum())
    if not n_act:
        return None
    up = n_up / n_act
    way = "up" if up >= 0.8 else "down" if up <= 0.2 else None
    if way is None:
        return None
    sign = 1.0 if way == "up" else -1.0

    def toward(c):
        t = sign * (c - level)
        return t[np.isfinite(t)]
    beyond = float(_percentile(a, 99, lambda c: np.maximum(-toward(c), 0.0)))
    most = max(float(t.max()) for t in (toward(c) for _, c in _rows64(a)) if len(t))
    return way if beyond <= REST_FRACTION * most else None


def rounding_only(a: np.ndarray, rest=None, swing=None) -> bool:
    """Whether a map of many levels has recorded nothing but the rounding of its readout: its typical swing is no
    more than the smallest step between its readings, so its cells only flip between neighbouring readings (an
    untouched pad read in whole numbers flickers between 0 and 1). A map of a few readings (taxels that are on or
    off) moves by one step when pressed, so it is never rounding only."""
    if a.shape[1] <= SMALL:
        return False
    # the readings sorted (one array of them, _pool): how many distinct ones, and the smallest step between two
    buf = _pool(a)
    k = 0
    for _, c in _rows64(a):
        v = c[np.isfinite(c)]
        buf[k:k + len(v)] = v
        k += len(v)
    v = buf[:k]
    v.sort()
    distinct, gap = min(k, 1), np.inf
    for i in range(0, max(k - 1, 0), CHUNK_VALUES):
        dv = np.diff(np.asarray(v[i:i + CHUNK_VALUES + 1], dtype=np.float64))
        pos = dv[dv > 0]
        distinct += len(pos)
        if len(pos):
            gap = min(gap, float(pos.min()))
    del buf, v
    if distinct <= SETTING_STATES:
        return False
    sw = _swing(a, resting_level(a, rest), swing)
    return sw <= gap * (1 + 1e-6)


def _distance(a: np.ndarray, rest=None, swing=None) -> tuple[np.ndarray, float]:
    """(every value's distance from rest, the typical swing: the upload's when given, else the 99th percentile of these
    distances)."""
    d = np.abs(a - resting_level(a, rest))
    if swing is not None and swing > 0:
        return d, float(swing)
    ok = d[np.isfinite(d)]
    return d, (float(np.percentile(ok, 99)) if len(ok) else 0.0)


def _swing(a: np.ndarray, level: np.ndarray, swing=None) -> float:
    """The typical swing (_distance) from the resting level, read in row chunks: the upload's when given."""
    if swing is not None and swing > 0:
        return float(swing)
    got = _percentile(a, 99, lambda c: (lambda d: d[np.isfinite(d)])(np.abs(c - level)))
    return float(got) if got is not None else 0.0


def swing_of(a: np.ndarray, rest=None, swing=None) -> float:
    """The typical swing of a signal (_distance's), never holding a whole float64 copy of it."""
    a = _float(a)
    return _swing(a, resting_level(a, rest), swing)


def distance_at(a: np.ndarray, k: int, rest=None, swing=None) -> tuple[np.ndarray, float]:
    """(every value's distance from rest at frame k, the typical swing): _distance's row k, never a whole copy."""
    a = _float(a)
    level = resting_level(a, rest)
    return np.abs(np.asarray(a[k], dtype=np.float64) - level), _swing(a, level, swing)


def active_at(a: np.ndarray, k: int, rest=None, swing=None) -> np.ndarray:
    """_active's row k: which values are away from rest at frame k."""
    d, sw = distance_at(a, k, rest, swing)
    return np.nan_to_num(d) > REST_FRACTION * sw if sw > 0 else np.zeros(len(d), dtype=bool)


def _active(a: np.ndarray, rest=None, swing=None) -> np.ndarray:
    a = _float(a)
    level = resting_level(a, rest)
    sw = _swing(a, level, swing)
    out = np.zeros(a.shape, dtype=bool)
    if sw > 0:
        for i, c in _rows64(a):
            out[i:i + len(c)] = np.nan_to_num(np.abs(c - level)) > REST_FRACTION * sw
    return out


def activity(a: np.ndarray, rest=None, swing=None) -> np.ndarray:
    """Per frame, the summed distance from rest of the signal's active values (NaN where it has no reading)."""
    a = _float(a)
    level = resting_level(a, rest)
    sw = _swing(a, level, swing)
    m = np.zeros(len(a))
    for i, c in _rows64(a):
        if sw > 0:
            d = np.abs(c - level)
            m[i:i + len(c)] = np.nansum(np.where(np.nan_to_num(d) > REST_FRACTION * sw, d, 0.0), axis=1)
        m[i:i + len(c)][np.isnan(c).all(axis=1)] = np.nan
    return m


def has_rest(a: np.ndarray, rest=None, swing=None) -> bool:
    """Whether a signal holds one level: HOLD_SHARE of its readings within HOLD_BAND of its swing from rest."""
    a = _float(a)
    level = resting_level(a, rest)
    sw = _swing(a, level, swing)
    if not sw > 0:
        return False
    n_ok = n_in = 0
    for _, c in _rows64(a):
        d = np.abs(c - level)
        ok = d[np.isfinite(d)]
        n_ok += len(ok)
        n_in += int((ok <= HOLD_BAND * sw).sum())
    return n_ok > 0 and n_in / n_ok >= HOLD_SHARE


def localized(a: np.ndarray, rest=None, swing=None) -> bool:
    """Whether an array has a rest and its activity is local: at a typical active frame fewer than half its values are
    active."""
    if not has_rest(a, rest, swing):
        return False
    a = _float(a)
    level = resting_level(a, rest)
    sw = _swing(a, level, swing)
    shares = []
    for _, c in _rows64(a):
        act = np.nan_to_num(np.abs(c - level)) > REST_FRACTION * sw
        shares.append(act[act.any(axis=1)].mean(axis=1))
    shares = np.concatenate(shares) if shares else np.zeros(0)
    return bool(len(shares)) and float(np.median(shares)) < 0.5


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
    a = _float(a)
    if not len(a) or not has_rest(a, rest, swing) or direction(a, rest, swing) is None:
        return False
    if a.shape[1] > 1 and _few_rows(a, SETTING_STATES):
        return False  # several values that only ever take a few readings together: a setting switching, not a sensor
    if rounding_only(a, rest, swing):
        return False
    return rests_and_rises(a, rest, swing) or (a.shape[1] > SMALL and localized(a, rest, swing))


def _few_rows(a: np.ndarray, most: int) -> bool:
    """Whether a's rows with a reading at every value take at most `most` distinct readings together (np.unique of
    them, axis 0, has at most that many), read in row chunks."""
    seen: list = []
    for _, c in _rows64(a):
        c = c[np.isfinite(c).all(axis=1)]
        while len(c):
            new = np.ones(len(c), dtype=bool)
            for r in seen:
                new &= ~(c == r).all(axis=1)
            if not new.any():
                break
            seen.append(c[int(np.flatnonzero(new)[0])].copy())
            if len(seen) > most:
                return False
            c = c[new]
    return True


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


MOVING_MIN = 10.0     # a value moves when its range is at least this many times its typical step between readings


def columns(a) -> np.ndarray:
    """a as frames by values: a vector is one value per frame, further axes (a pressure map) are flattened. The readers
    write every signal this way; label/episode.py load applies it too, so a signal stored as a bare vector is read as
    one column by the prompt and the checks rather than breaking them."""
    a = _float(a)
    return a.reshape(a.shape[0], int(np.prod(a.shape[1:])))


def pad_rows(a: np.ndarray, n: int) -> np.ndarray:
    """a (frames by values) with no reading (NaN) past its last row up to n frames, as a reader writes a signal that
    ends inside the footage: a signal whose rows stop short of the episode is read over every frame, so the prompt
    names the frames it has no reading at, and a step that lines every signal up frame by frame (quiet_spans) never
    breaks on it. A signal with no rows (named as having none) or with at least n rows is kept as it is."""
    if not 0 < len(a) < n:
        return a
    out = np.full((n, a.shape[1]), np.nan, dtype=a.dtype if np.issubdtype(a.dtype, np.floating) else np.float64)
    out[:len(a)] = a
    return out


def _range_and_step(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per value of a (frames by values, any further axes flattened): its range, and its typical step, the median of
    its nonzero absolute steps between consecutive readings (a pair with a missing reading is no step). NaN where a
    value has no reading or never changes."""
    a = columns(a)
    if len(a) == 0:
        return np.full(a.shape[1], np.nan), np.full(a.shape[1], np.nan)
    rs, steps = [], []
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i, j in _col_chunks(a):
            c = np.asarray(a[:, i:j], dtype=np.float64)
            rs.append(np.nanmax(c, axis=0) - np.nanmin(c, axis=0))
            d = np.abs(np.diff(c, axis=0))
            steps.append(np.nanmedian(np.where(d > 0, d, np.nan), axis=0) if len(c) > 1 else np.full(j - i, np.nan))
    return np.concatenate(rs), np.concatenate(steps)


def _score(r: np.ndarray, step: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"):
        m = r / step
    return np.where(np.isfinite(m) & (r > 0), m, 0.0)


def movements(a: np.ndarray) -> np.ndarray:
    """How much each value of a signal moves over the episode, unit free: its range over the size of one move (its
    median nonzero step between consecutive readings), whatever share of the frames it moves on, so it does not depend
    on how long the recording is. A joint or an odometer that sweeps far scores in the hundreds, a value that only
    jitters by its noise, or a count or flag that changes by one, scores near 1. 0 for a value that never changes, has
    no reading, or has no rows."""
    return _score(*_range_and_step(a))


def quiet_spans(arrs: dict, need: int) -> list[tuple[int, int]]:
    """[(a, b)] frame spans of at least need frames over which every moving value of every signal (movements at least
    MOVING_MIN) stays within MOVING_MIN of its own typical steps: where a recording with no arm state shows nothing
    moving, whatever the length of the recording. Used only to choose instants (label/episode.py plan), never told to
    the model as a claim. A value with some frames that have no reading takes part where it has readings, and a frame
    where none has a reading is never quiet."""
    from label import state as ms
    cols, tols = [], []
    for a in arrs.values():
        a = columns(a)
        r, step = _range_and_step(a)
        for j in np.flatnonzero(_score(r, step) >= MOVING_MIN):
            cols.append(a[:, j])
            tols.append(MOVING_MIN * step[j])
    if not cols:
        return []
    s, tol = np.stack(cols, axis=1), np.asarray(tols)
    # a frame where no value has a reading shows nothing quiet: each stretch of frames with a reading is searched alone
    read = np.isfinite(s).any(axis=1)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], read.astype(int), [0]])))
    spans = []
    for lo, hi in zip(edges[::2].tolist(), edges[1::2].tolist()):
        spans += [(lo + a, lo + b) for a, b in ms.spans_within(s[lo:hi], tol, need)]
    return spans


def summary_rows(name: str, a: np.ndarray, ks: list[int], shape=None, names=None, rest=None,
                 swing=None) -> list[tuple[str, list[str]]]:
    """[(row label, one value per sampled instant)] for one signal: its values when it gets one row per value
    (per_value), else the numbers that summarise the array (module docstring). "-" is no reading at that instant."""
    a = _float(a)
    d = a.shape[1]
    if per_value(name, d, shape, names):
        labels = names if names and len(names) == d else ([""] if d == 1 else [f"[{i}]" for i in range(d)])
        return [(f"{name}{(' ' + lb) if lb else ''}", [_num(a[k, i]) for k in ks]) for i, lb in enumerate(labels)]
    rows = []
    if localized(a, rest, swing):
        # the distances, the active values and the activity at the sampled instants alone, as the whole would give them
        level = resting_level(a, rest)
        sw = _swing(a, level, swing)
        dist = {k: np.abs(np.asarray(a[k], dtype=np.float64) - level) for k in ks}
        act = {k: (np.nan_to_num(v) > REST_FRACTION * sw if sw > 0 else np.zeros(d, dtype=bool))
               for k, v in dist.items()}
        m = {k: (np.nan if np.isnan(a[k]).all() else float(np.nansum(np.where(act[k], dist[k], 0.0)))) for k in ks}
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
            x = np.abs(np.asarray(a[b0], dtype=np.float64) - np.asarray(a[a0], dtype=np.float64))
            ch.append("-" if not np.isfinite(x).any() else _num(np.nanmax(x)))
        rows.append((f"{name} largest change of any value since the instant before", ch))
    return rows


def describe(name: str, a: np.ndarray, shape=None, names=None, rest=None, swing=None, rate_hz=None,
             fps=None, aligned_by=None, camera_aligned_by=None, clock_problem=None,
             source_rows=None, camera_frames=None) -> str:
    """One line: the signal's name, its shape or value names, its rate when it is recorded slower than the frames
    (rate_hz below RATE_SLOWER of fps), that it was placed from both starts when the reader had no clock in common
    to place it by (aligned_by, prepare/formats.py mark_assumed) or one row per frame when it has as many rows as the
    video has frames (ALIGNED_ROWS), or tied readings placed within each recorded stamp interval as an assumption
    (COARSE_CLOCK, whose rate is estimated from that placement), and the range each value takes, up to PER_VALUE_MAX
    values (for a wider array, the range of all its values together)."""
    a = _float(a)          # as stored: its smallest and largest readings are exact in any precision
    d = a.shape[1]
    what = (f"{' x '.join(str(int(x)) for x in shape)} values" if shape and len(shape) > 1 else
            f"{d} value{'s' if d > 1 else ''}")
    if names and len(names) == d and d <= PER_VALUE_MAX:
        what += " (" + ", ".join(names) + ")"
    if rate_hz and fps and float(rate_hz) < RATE_SLOWER * float(fps):
        rate = _num(float(rate_hz))
        what += (f", estimated at {rate} Hz from assumed placement" if aligned_by == COARSE_CLOCK or clock_problem else
                 f", recorded at {rate} Hz")
    if aligned_by == ALIGNED_ROWS:
        if clock_problem:
            what += ", placed one row per frame as an assumption"
            if source_rows is not None and camera_frames is not None:
                what += f" from {source_rows} original rows and {camera_frames} original camera frames"
        else:
            what += ", placed one row per frame as it has as many rows as the video has frames"
    elif aligned_by == COARSE_CLOCK:
        what += ", tied readings placed within each stamp interval as an assumption"
    elif aligned_by == ALIGNED_CAMERA:
        what += ", placed on footage through its assumed camera presentation clock"
    elif aligned_by == ALIGNED_ASSUMED:
        what += ", placed from both starts as no clock is shared"
    elif aligned_by:
        what += ", " + placement_text(aligned_by)
    if camera_aligned_by == ALIGNED_CAMERA and aligned_by != ALIGNED_CAMERA:
        what += ", shown on footage through its assumed camera presentation clock"
    head = f"  {name} ({what})"
    if not np.isfinite(a).any():
        return f"{head}: no reading"
    gaps = gap_words(a)
    tail = f"; {gaps}" if gaps else ""
    lo, hi = finite_range(a)
    if d > PER_VALUE_MAX:
        if (hi == lo).all():
            return f"{head}: every value constant throughout{tail}"
        return f"{head}: values from {_num(np.nanmin(lo))} to {_num(np.nanmax(hi))}{tail}"
    if (hi == lo).all():
        value = _num(lo[0]) if d == 1 else "[" + ", ".join(_num(x) for x in lo) + "]"
        return f"{head}: {value} throughout{tail}"
    return f"{head}: " + ", ".join(_num(l) if l == h else f"{_num(l)} to {_num(h)}" for l, h in zip(lo, hi)) + tail
