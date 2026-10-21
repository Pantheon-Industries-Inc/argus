"""Reading a recording's other signals (prepare/formats.py Signals): what each one does over an episode, in numbers a
person or a model can check against the frames.

A signal is read by how its numbers behave, and only whether it measures touch also asks its name (is_touch):

- Its resting level, per value. A vector of SMALL values or fewer (a force, a flag, a position) that holds a level at
  an end of its range rests at that end (its 1st or 99th percentile, so a rest held for only a few percent of the
  recording is still inside it), else at its median: a force sensor at its offset, an IMU or a position at its middle.
  Which end is never read from how long it sits there, since a sensor can be loaded through most of a recording: it
  is the end the recordings start and end at unless the other end reads zero, else the end that reads zero, else the
  one it sits at most (_rest_end). An array (a pressure map) is read as one sensor: its values pooled say which end it rests at the same
  way (OpenTouch's raw glove reads 3072 untouched and less when pressed), and each value rests at its own extreme on
  that end (its 1st or 99th percentile), so a cell pressed through most of a clip still rests where it reads
  untouched. An array whose pooled values sit in the middle (hand landmarks) rests at each value's median.
- A value is active when it is away from rest by more than REST_FRACTION of the array's typical swing (the 99th
  percentile of every value's distance from rest), so the noise of many idle cells never adds up to a contact. The
  signal's activity at a frame is the summed distance of its active values, and its peak the largest in the episode.
- It has a rest only when it really holds one level: at least HOLD_SHARE of its readings near rest lie within
  HOLD_BAND of its typical swing from their resting level (an untouched cell reads 3072 frame after frame). A position
  or an IMU, whose values drift through their range, has no rest, whatever its percentiles say.
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

import json
import warnings

import numpy as np

from prepare.signal_alignment import (ALIGNED_ASSUMED, ALIGNED_ROWS, ALIGNED_CAMERA, COARSE_CLOCK,
                                      RECORDER_ARRIVAL, placement_text)

REST_FRACTION = 0.1
MIN_ACTIVE = 0.02
# as short a share at rest as one active counts: a sensor loaded through 95% of a recording still rests and rises
MIN_REST = MIN_ACTIVE
SMALL = 4
# a signal of this many values or fewer is described value by value (each value's range, or its constant values); the
# widest a prompt showed value by value before arrays were read as wholes, so a 14-joint action or a 9-value camera
# matrix keeps every number, while a pressure map or hand landmarks get the pooled range of all their values
PER_VALUE_MAX = 64
# Ordinary retained descriptors are at most 585 characters. Longer rows group readings by value index and keep
# their full names in the original metadata, retaining every range without an unbounded inline name list.
DESCRIBE_ROW_MAX_CHARS = 768
# The values at each instant follow describe for a flat vector of up to PER_VALUE_MAX values whose values the dataset
# names, or whose own name says it holds joints or a state: one row per value under its own name. A humanoid's 26
# joints and a base's x, y, yaw, vx, wz reached that readout as one "largest change" or "total activity" row while
# it went value by value only up to SMALL. A map (a 2 D shape, whose strongest cell says more than its cells one by
# one) and a long unnamed vector (audio samples) keep their summary rows. The name's words are matched whole
# (prepare/formats.py _names_word, as every name word is), so jointly and statevec are not joints or a state.
JOINT_LIKE = ("joint", "qpos", "state")
# A signal recorded below this share of the frame rate says its rate: a 6 Hz glove otherwise reads as if it were
# sampled every frame.
RATE_SLOWER = 0.9
# A flat vector whose values are unnamed and that only its own name calls joints or a state gets one row per value up
# to this many values: a humanoid's 26 joints do, an unnamed 21 x 3 hand array named hand_joints (63 values, rows
# labelled only by index) does not. A vector whose values the dataset names keeps the PER_VALUE_MAX limit.
JOINT_NAME_MAX = 32
MOVEMENT_ROLES = frozenset(("joint_state", "actuator_command", "end_effector_pose", "base_motion", "pose",
                            "landmarks", "hand_pose", "imu", "odometry", "action", "state"))
NON_TOUCH_ROLES = MOVEMENT_ROLES | frozenset(("event_flag", "annotation", "calibration", "bookkeeping"))
MERGE_GAP_S = 0.15
HOLD_BAND = 0.02
SETTING_STATES = 3        # a signal of several values with this many distinct readings or fewer is a setting
HOLD_SHARE = 0.3
# the share of a recording's rows at its start and at its end read as where it starts and ends (recording_ends)
ENDS_SHARE = 0.01
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


def _readings(c: np.ndarray) -> np.ndarray:
    """c with every value that is not a finite number as NaN: an infinite value is no reading, as a NaN is."""
    return np.where(np.isfinite(c), c, np.nan)


def _away(d: np.ndarray, sw: float) -> np.ndarray:
    """Which distances from rest d are away from it (over REST_FRACTION of the swing sw). A value with no finite
    reading is never away: an infinite one would otherwise count as the largest activity of the episode."""
    with np.errstate(invalid="ignore"):
        return np.isfinite(d) & (d > REST_FRACTION * sw)


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


def per_value(name: str, d: int, shape=None, names=None, role=None) -> bool:
    """Whether a signal of d values per frame gets one row per value at each instant (JOINT_LIKE above)."""
    if d <= SMALL:
        return True
    if d > PER_VALUE_MAX or (shape and len(shape) > 1):
        return False
    if _role(role) == "force_torque":
        return True
    if names and len(names) == d:
        return True
    from prepare.formats import _names_word
    return d <= JOINT_NAME_MAX and _names_word(name, JOINT_LIKE)


def names_joints_or_state(name: str) -> bool:
    """Whether a signal's name says it holds joints or a state, as the no state line names it (label/episode.py): a word
    of it is joint or qpos (left_joint1, /left/joint_states, observations/qpos), or its last word is state or states
    (observation.state, observation.leader_state), and no word of it says a command (prepare/formats.py
    COMMAND_WORDS, so /left/joint_command and leader_cmd_state are not). Words are split as every name matcher splits
    them (prepare/formats.py word_tokens: separators, camelCase and glued known words, so observation.robotstate ends
    in state). A one value battery_state or estop_state is left out by the line's own rule that a named signal has
    more than one value."""
    from prepare.formats import COMMAND_WORDS, _names_word, word_tokens
    words = word_tokens(name)
    return ((_names_word(name, ("joint", "qpos")) or bool(words) and words[-1] in ("state", "states"))
            and not _names_word(name, COMMAND_WORDS))


def names_scalar_observed_state(name: str) -> bool:
    """A scalar can disprove absent arm state only when its name identifies a joint or observed arm state.
    Sensor readings, bookkeeping and commands do not establish that state, even under observation.*."""
    from prepare.formats import TOUCH_WORDS, _names_word, word_tokens
    if not names_joints_or_state(name) or _names_word(name, TOUCH_WORDS + (
            "torque", "clock", "time", "timestamp",
            "battery", "estop", "emergency", "health", "status", "mode", "power", "temperature", "voltage",
            "current")):
        return False
    words = [w for w in word_tokens(name) if not w.isdigit()]      # arm0_state's word before state is arm
    return (_names_word(name, ("joint", "qpos")) or len(words) > 1 and words[-1] in ("state", "states")
            and _names_word(words[-2], ("observation", "arm", "robot", "gripper", "leader", "follower")))


def reading_text(x: float) -> str:
    """A reading as the request writes it: three significant figures, and - for no reading."""
    return "-" if not np.isfinite(x) else f"{float(x):.3g}"


def recording_ends(arrs) -> np.ndarray:
    """The readings at the start and at the end of each recording in arrs (frames by values each): ENDS_SHARE of its
    rows at each end, at least one, stacked. Where a sensor reads before anything is touched and after it is let go
    (_rest_end)."""
    out = []
    for x in arrs:
        if len(x):
            k = max(1, int(len(x) * ENDS_SHARE))
            out += [np.asarray(x[:k], dtype=np.float64), np.asarray(x[-k:], dtype=np.float64)]
    width = int(np.prod(np.shape(arrs[0])[1:])) if len(arrs) else 0
    return np.concatenate(out).reshape(-1, width) if out else np.zeros((0, width))


def _rest_end(lo, med, hi, held_lo, held_hi, ends) -> np.ndarray:
    """Per value (or one entry for values pooled): -1 when it rests at its low end lo, 1 at its high end hi, 0 at its
    median med. held_lo and held_hi are the shares of its readings within REST_FRACTION of its range of each end, ends
    its readings at the starts and ends of its recordings (recording_ends, frames by values).

    Which end a sensor rests at cannot be read from how much of the time it sits there: a force sensor loaded through
    most of a recording sits mostly at its loaded level. So those shares only say whether the signal holds a level at
    an end at all (HOLD_SHARE of its readings at one end; an IMU or a drifting position holds none and rests at its
    median). Which end is rest then comes from evidence that does not depend on how long it is loaded, in this order:
    - the recordings' starts and ends: most of them read what the sensor reads with nothing touched, since a
      demonstration starts before the hand or gripper takes hold of anything and ends after it lets go;
    - zero, when those do not agree (a recording that starts mid-grasp): the one end that reads zero within
      REST_FRACTION of the range, since a calibrated force, a pressure or a contact flag reads zero unloaded;
    - else the end it sits at for most of its readings, as when nothing else says.
    The starts and ends count only when zero does not say otherwise. Ends at the far end from a zero reading are a
    recording that starts and ends mid-grasp as often as a sensor that rests away from zero, and the two read alike,
    so then neither is assumed and the end it sits at for most of its readings decides (zero when it sits at
    neither)."""
    lo, med, hi = (np.asarray(x, dtype=np.float64) for x in (lo, med, hi))
    span = hi - lo
    ok = span > 0
    pos = (med - lo) / np.where(ok, span, 1.0)
    by_share = np.where(pos <= 0.25, -1, np.where(pos >= 0.75, 1, 0))
    held = (np.asarray(held_lo) >= HOLD_SHARE) | (np.asarray(held_hi) >= HOLD_SHARE)
    e = np.asarray(ends, dtype=np.float64).reshape(-1, len(lo))
    fin = np.isfinite(e)
    with np.errstate(invalid="ignore"):
        e_lo = ((e <= lo + 0.25 * span) & fin).sum(axis=0)
        e_hi = ((e >= hi - 0.25 * span) & fin).sum(axis=0)
    half = fin.sum(axis=0) / 2
    by_ends = np.where((e_lo > e_hi) & (e_lo >= half), -1, np.where((e_hi > e_lo) & (e_hi >= half), 1, 0))
    band = REST_FRACTION * span
    z_lo, z_hi = np.abs(lo) <= band, np.abs(hi) <= band
    by_zero = np.where(z_lo & ~z_hi, -1, np.where(z_hi & ~z_lo, 1, 0))
    dwell = np.where(by_share != 0, by_share, by_zero)
    chosen = np.where(by_ends == 0, np.where(by_zero != 0, by_zero, by_share),
                      np.where((by_zero == 0) | (by_zero == by_ends), by_ends, dwell))
    return np.where(ok, np.where(held, chosen, by_share), 0)


def _pooled_end(a: np.ndarray, ends=None) -> str | None:
    """"low" or "high" when an array's values pooled rest at one end of their range (a pressure map, _rest_end), else
    None. ends, its readings at the starts and ends of its recordings, default to its own first and last rows."""
    got = _percentile(a, [1, 50, 99], lambda c: c[np.isfinite(c)])
    if got is None:
        return None
    lo, med, hi = (float(x) for x in got)
    if not hi > lo:
        return None
    band = REST_FRACTION * (hi - lo)
    n = n_lo = n_hi = 0
    for _, c in _rows64(a):
        f = c[np.isfinite(c)]
        n += len(f)
        n_lo += int((f <= lo + band).sum())
        n_hi += int((f >= hi - band).sum())
    ends = recording_ends([a]) if ends is None else np.asarray(ends, dtype=np.float64)
    end = int(_rest_end([lo], [med], [hi], [n_lo / max(n, 1)], [n_hi / max(n, 1)], ends.reshape(-1, 1))[0])
    return "low" if end < 0 else "high" if end > 0 else None


def resting_level(a: np.ndarray, rest=None, ends=None) -> np.ndarray:
    """Per value, where the signal rests (module docstring, _rest_end); rest, the upload's own (prepare/formats.py
    measure_signal_scales), is used as given when it has one value per value. ends are its readings at the starts and
    ends of its recordings (recording_ends) when a joins several recordings; by default its own first and last rows."""
    if rest is not None and len(rest) == a.shape[1]:
        return np.asarray(rest, dtype=np.float64)
    if not _any_finite(a):
        return np.zeros(a.shape[1])
    # a value with no reading at all rests at 0 (nan_to_num); the percentile of its empty column is NaN, said quietly
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if a.shape[1] > SMALL:
            end = _pooled_end(a, ends)
            # value by value, a few thousand values at a time
            per = (lambda c: np.nanpercentile(c, 1, axis=0)) if end == "low" else \
                (lambda c: np.nanpercentile(c, 99, axis=0)) if end == "high" else (lambda c: np.nanmedian(c, axis=0))
            return np.nan_to_num(np.concatenate([per(_readings(np.asarray(a[:, i:j], dtype=np.float64)))
                                                 for i, j in _col_chunks(a)]))
        a = _readings(np.asarray(a, dtype=np.float64))
        lo, med, hi = np.nanpercentile(a, [1, 50, 99], axis=0)
        fin = np.isfinite(a)
        n = np.maximum(fin.sum(axis=0), 1)
        band = REST_FRACTION * (hi - lo)
        held_lo = ((a <= lo + band) & fin).sum(axis=0) / n
        held_hi = ((a >= hi - band) & fin).sum(axis=0) / n
        end = _rest_end(lo, med, hi, held_lo, held_hi, recording_ends([a]) if ends is None else ends)
    return np.nan_to_num(np.where(end < 0, lo, np.where(end > 0, hi, med)))


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
        act = _away(np.abs(dev), sw)
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
    """Which values are away from rest at frame k."""
    d, sw = distance_at(a, k, rest, swing)
    return _away(d, sw) if sw > 0 else np.zeros(len(d), dtype=bool)


def activity(a: np.ndarray, rest=None, swing=None) -> np.ndarray:
    """Per frame, the summed distance from rest of the signal's active values (NaN where it has no reading)."""
    a = _float(a)
    level = resting_level(a, rest)
    sw = _swing(a, level, swing)
    m = np.zeros(len(a))
    for i, c in _rows64(a):
        if sw > 0:
            d = np.abs(c - level)
            m[i:i + len(c)] = np.where(_away(d, sw), d, 0.0).sum(axis=1)
        m[i:i + len(c)][~np.isfinite(c).any(axis=1)] = np.nan
    return m


def declared_zero_reference(meta: dict, width: int | None = None):
    """Only a literal finite calibration zero and explicit polarity establish this reference."""
    calibration = meta.get('calibration')
    if isinstance(calibration, str) and len(calibration) <= 65536:
        try:
            calibration = json.loads(calibration)
        except (ValueError, RecursionError):
            return None
    direction = meta.get('response_direction')
    if not isinstance(calibration, dict) or direction not in ('up', 'down'):
        return None
    zero = calibration.get('zero')
    if isinstance(zero, bool):
        return None
    try:
        if np.asarray(zero).dtype.kind not in 'iuf':
            return None
        level = np.asarray(zero, dtype=float)
    except (TypeError, ValueError):
        return None
    if width is None:
        width = level.size if level.ndim == 1 else 1
    if level.ndim == 0:
        level = np.full(width, float(level))
    if not width or level.shape != (width,) or not np.isfinite(level).all():
        return None
    return level, 1 if direction == 'up' else -1


def declared_zero_activity(a: np.ndarray, meta: dict):
    """Positive recorded response from a declared zero; missing rows stay missing.

    No episode-relative threshold or inferred resting endpoint changes the
    declared zero. This response does not by itself verify physical contact.
    """
    reference = declared_zero_reference(meta, a.shape[1])
    if reference is None:
        return None
    level, sign = reference
    result = np.zeros(len(a))
    for start, block in _rows64(a):
        valid = np.isfinite(block)
        response = np.where(valid, np.maximum(sign * (block - level), 0), 0)
        result[start:start + len(block)] = response.sum(axis=1)
        result[start:start + len(block)][~valid.any(axis=1)] = np.nan
    return result


def has_rest(a: np.ndarray, rest=None, swing=None) -> bool:
    """Whether a signal holds one level: of its readings at rest (within REST_FRACTION of its swing from rest, at least
    MIN_ACTIVE of them), HOLD_SHARE lie within HOLD_BAND of it. Counted among the readings at rest rather than all of
    them, so a sensor loaded through most of the recording still holds its rest whenever it is unloaded."""
    a = _float(a)
    level = resting_level(a, rest)
    sw = _swing(a, level, swing)
    if not sw > 0:
        return False
    n_ok = n_near = n_in = 0
    for _, c in _rows64(a):
        d = np.abs(c - level)
        ok = d[np.isfinite(d)]
        n_ok += len(ok)
        n_near += int((ok <= REST_FRACTION * sw).sum())
        n_in += int((ok <= HOLD_BAND * sw).sum())
    return n_near > 0 and n_near >= MIN_ACTIVE * n_ok and n_in / n_near >= HOLD_SHARE


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
        act = _away(np.abs(c - level), sw)
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


def _role(role):
    return role.strip().casefold() if isinstance(role, str) else ""


# The roles a numeric reader reads as a touch measurement.
TOUCH_QUANTITY_ROLES = ("touch", "tactile", "pressure", "normal_force", "shear_force", "force_torque")


def signal_role(role) -> str:
    """The role an interpretation gives a numeric signal, checked against what the signal is: numbers per frame. A
    role that names a camera image's content (label/dictionary_context.py SCENE_CONTENT_ROLES, SENSOR_CONTENT_ROLES)
    cannot be read as stated, since the signal is no image. A tactile image role still says the numbers are touch
    readings, so it reads as touch; any other image role says nothing a numeric reader can use, so it reads as no
    role and the signal's own name decides, as it does before a dictionary runs. Every other role is kept."""
    from label.dictionary_context import SCENE_CONTENT_ROLES, SENSOR_CONTENT_ROLES, TACTILE_CONTENT_ROLES
    role = _role(role)
    if role in TOUCH_QUANTITY_ROLES or role not in SCENE_CONTENT_ROLES | SENSOR_CONTENT_ROLES:
        return role
    return "touch" if role in TACTILE_CONTENT_ROLES else ""


def role_mismatch(name: str, role) -> str | None:
    """The note that says a numeric signal's role names an image, and how it is read instead (signal_role)."""
    if not isinstance(role, str) or signal_role(role) == _role(role):
        return None
    read = "touch readings" if signal_role(role) == "touch" else "a signal its name describes"
    return (f"{name}: the field dictionary called this signal {role.strip()}, an image role, but it is recorded as "
            f"numbers, so it is read as {read}.")


def touch_permission(name: str, role=None):
    """Commands and known noncontact roles veto stored verdicts before any numeric reuse."""
    from prepare.formats import COMMAND_WORDS, _names_word
    if _names_word(name, COMMAND_WORDS) or signal_role(role) in NON_TOUCH_ROLES:
        return False
    return True if signal_role(role) == "touch" else None


def touch_cache_permission(name: str, role=None, qualified_role=None):
    """A cleared role can reuse a recorded verdict only when the native name itself names touch."""
    permission = touch_permission(name, role)
    if permission is None and qualified_role == "touch":
        from prepare.formats import names_touch
        return names_touch(name)
    return permission


def is_touch(name: str, a: np.ndarray, rest=None, swing=None, role=None) -> bool:
    """Whether a signal measures touch: its own name says so (prepare/formats.py names_touch: tactile, pressure,
    contact, force, and never a command) and its numbers behave like touch (touch_like). Numbers alone cannot
    decide it: a humanoid's torso joint that holds still and then moves one way, a mobile base's odometry, an action
    or a pose rest and rise like a pressure pad, and only the name says which one measures touch. A touch signal
    whose name says nothing (ch0) is not read as touch until a data dictionary can say it is."""
    from prepare.formats import names_touch
    permission = touch_permission(name, role)
    return permission is not False and (permission is True or names_touch(name)) and touch_like(a, rest, swing)


def active_spans(a: np.ndarray, t: np.ndarray, rest=None, swing=None) -> list[tuple[float, float]]:
    """[(start s, end s)] where the signal is away from rest (activity at least REST_FRACTION of its peak), from every
    frame; spans closer than MERGE_GAP_S are one, and a single-frame flicker is not a span."""
    return response_spans(activity(a, rest, swing), t)


def response_spans(m: np.ndarray, t: np.ndarray, *, declared_zero=False) -> list[tuple[float, float]]:
    """Response intervals; a declared zero uses positive response rather than an inferred relative threshold."""
    if not np.isfinite(m).any() or not np.nanmax(m) > 0:
        return []
    on = np.isfinite(m) & (m > 0 if declared_zero else m >= REST_FRACTION * np.nanmax(m))
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


# A value's readings at the sampled instants resolve it when each reading stands for the stretch of frames its instant
# represents (the share of the sampled readings' variance their stretch means explain, at least RESOLVED_STAND), or
# when consecutive readings follow one another (lag-1 autocorrelation of the sampled readings at least RESOLVED_FOLLOW)
# and the readings are no further from their stretch means than from their own average (stand at least 0, so a
# vibration sampled near its own period, whose readings drift slowly while the frames around them swing, is not
# taken for a slow value). A joint or gripper position passes; a velocity or a noisy effort read once a second does
# not, because its reading at an instant says nothing about the second around it. On 21 bimanual teleop requests at
# 1 instant per second, velocity rows sent fell from 284 to 22 and effort rows from 276 to 179, joint position rows
# rose from 523 to 528 (the budget no longer cut them), and the only gripper rows left out were of grippers that did
# not move in that request (0.999 to 1), whose readings only jitter.
RESOLVED_FOLLOW = 0.5
RESOLVED_STAND = 0.85
# Below this many instants neither score is estimated well enough to drop a row (a lag-1 autocorrelation of n readings
# has a standard error near 1 / sqrt(n), 0.28 at 13 instants), and a row of so few readings costs little: every row is
# kept as it is, and no row is taken as a copy of another.
RESOLVED_MIN_INSTANTS = 30


def is_resolved(follow: np.ndarray, stand: np.ndarray) -> np.ndarray:
    """The rule above, on resolution's (follow, stand)."""
    return (stand >= RESOLVED_STAND) | ((follow >= RESOLVED_FOLLOW) & (stand >= 0))


def resolution(a: np.ndarray, ks: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """(follow, stand) per value of a (frames by values) at the sampled instants ks: the lag-1 autocorrelation of its
    readings at ks, and the share of their variance explained by the mean of the frames each instant stands for (from
    halfway to the instant before to halfway to the instant after). Missing readings are left out of both. A value
    whose sampled readings do not vary scores 0 on both: its readings at the instants say nothing of how it changed.
    With fewer than RESOLVED_MIN_INSTANTS instants both are 1."""
    a = columns(a)
    ks = np.asarray(ks, dtype=int)
    d = a.shape[1]
    follow, stand = np.zeros(d), np.zeros(d)
    if len(ks) < RESOLVED_MIN_INSTANTS:
        return np.ones(d), np.ones(d)
    edges = np.concatenate([[ks[0]], (ks[1:] + ks[:-1] + 1) // 2, [ks[-1] + 1]])
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for j in range(d):
            col = np.asarray(a[:, j], dtype=np.float64)
            v = col[ks]
            means = np.array([np.nanmean(col[edges[i]:edges[i + 1]]) for i in range(len(ks))])
            ok = np.isfinite(v)
            dv = v[ok] - v[ok].mean() if ok.any() else np.zeros(0)
            total = float((dv * dv).sum())
            if total <= 0:
                continue
            both = ok[1:] & ok[:-1]
            centred = np.where(ok, v - v[ok].mean(), 0.0)
            follow[j] = float((centred[1:] * centred[:-1])[both].sum() / total)
            fit = ok & np.isfinite(means)
            stand[j] = float(1 - ((v[fit] - means[fit]) ** 2).sum() / total)
    return follow, stand


# A row the instants do not resolve can still carry its trend: a joint a policy steps every half second reads as noise
# at one instant a second, yet its mean over each instant's stretch of frames follows the motion. Such a row is sent as
# those window means when they keep at least WINDOW_SIGNAL times the share of variance that means of the same windows
# keep of white noise (one over the frames in a window); means of noise are left out. On a 15 fps GelSight rollout
# sampled every 1.47 s the joints, end-effector pose and actions keep 0.19 to 0.48 against 0.045 for noise, and its
# velocities mostly 0.03 to 0.05; on the YAM spot check velocities and efforts keep 0.19 to 0.74 against 0.033, so they
# are sent as window means too: the mean velocity over a second is real motion.
WINDOW_SIGNAL = 4.0
# Window-mean rows rank after every row whose own readings resolve it, among themselves by the share of variance their
# means keep (so a gripper effort that steps during grasps comes before a joint velocity that mostly mirrors its
# position), and together use at most WINDOW_SHARE of the readout budget; those that do not fit are named with when
# they peaked. Set by cost: on the 21 spot requests, sending every window-mean row adds 0.41M characters per footage
# hour (1.61M to 2.03M); at this share the prompts come to 1.656M per hour (1.661M at 0.035, 1.682M at 0.05), under
# the 1.665M per hour of the build the 100 h run was priced on.
WINDOW_SHARE = 0.03


def window_means(a: np.ndarray, ks: list[int]) -> tuple[np.ndarray, np.ndarray, float]:
    """(means, kept, noise): per value of a, its mean over the frames each instant stands for (the stretches resolution
    uses), one per instant; the share of the value's variance over the first to last instant that those means keep; and
    the share means of the same windows keep of white noise (the mean of one over each window's length)."""
    a = columns(a)
    ks = np.asarray(ks, dtype=int)
    edges = np.concatenate([[ks[0]], (ks[1:] + ks[:-1] + 1) // 2, [ks[-1] + 1]])
    lens = np.diff(edges)
    means = np.full((len(ks), a.shape[1]), np.nan)
    kept = np.zeros(a.shape[1])
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for j in range(a.shape[1]):
            col = np.asarray(a[:, j], dtype=np.float64)
            means[:, j] = [np.nanmean(col[edges[i]:edges[i + 1]]) for i in range(len(ks))]
            seg = col[edges[0]:edges[-1]]
            total = float(np.nanvar(seg)) if np.isfinite(seg).any() else 0.0
            ok = np.isfinite(means[:, j])
            if total > 0 and ok.sum() > 1:
                mu = np.average(means[ok, j], weights=lens[ok])
                kept[j] = float(np.average((means[ok, j] - mu) ** 2, weights=lens[ok]) / total)
    return means, kept, float(np.mean(1.0 / np.maximum(lens, 1)))


def resolved(a: np.ndarray, ks: list[int]) -> np.ndarray:
    """Per value of a, whether its readings at the sampled instants resolve it (resolution)."""
    return is_resolved(*resolution(a, ks))


# One value tracks another when its readings at the sampled instants are a straight-line function of the other's
# (either sign, any scale and offset, so degrees against radians or a mirrored joint count), read up to TRACK_LAG_S
# apart, both in level (r squared at least TRACK_LEVEL) and in how they change from one instant to the next (r squared
# at least TRACK_CHANGE), and their frame-to-frame changes line up best within TRACK_LAG_S. The change test keeps apart
# two values that make one big move together and nothing else alike; the lag test keeps apart two slow values that
# make the same moves a second or more apart. Set from 16 bimanual teleop episodes (21 requests, 767 sent value
# rows): each teleoperated joint against the same joint of the arm that commands it passes in 264 of 279 pairs, and in
# all 264 the measured joint moves later; 18 of the 767 rows would merge with some other row, all in three requests
# (a joint's position with its own effort, or two joints of one arm that moved together).
TRACK_LAG_S = 0.5
TRACK_LEVEL = 0.95
TRACK_CHANGE = 0.9
# A good fit on average is not a copy everywhere: a measured joint that stalls while its command moves still fits the
# whole window closely. So a tracking row is a copy only where the pair tracks as it usually does, and its readings are
# still sent at the instants where it departs from that (apart_instants). A pair apart for much of the request fails
# the tracking fit itself and both rows are sent whole: of 264 real tracked pairs none is apart at more than 0.43 of its
# instants, and a synthetic stall of 8 s still tracks while one of 10 s does not. A follower behind its command is off by about its speed times its lag, so a pair's usual
# one second residual is its own b0 + c x speed, and the excess over that is what departs: a stall leaves the whole
# distance the command moved, a grasp the distance the commanded close goes past the gripper. A second is apart when
# its excess is above both the pair's own median excess plus TRACK_APART_TIMES spreads (median absolute deviation,
# scaled to a standard deviation, which a stall or grasp over less than half the request cannot widen) and TRACK_APART
# of the range. Set on the 21 spot check requests with stalls and grasps injected into the real commanded and measured
# pairs: 28 of 229 joint pairs have any instant apart with nothing injected, and 253 of 264 stalls of 1 s (where the
# command moves most), 260 of 264 of 2 s and 31 of 35 grasps are found; in synthetic 60 s requests every stall of 1 to
# 10 s is found except one of 1 s in which the command moves only 3.5% of its range.
TRACK_HOLD_S = 1.0
TRACK_APART_TIMES = 20.0
TRACK_APART = 0.03


def window_sum(a: np.ndarray, win: int) -> np.ndarray:
    """The sum of a over the win frames centred on each frame (frames past either end count nothing), one value per
    frame of a whatever win is. np.convolve(mode="same") returns max(len(a), win) values, so a window longer than the
    frames gave a series that no longer lined up with them, and a per-frame mask applied to it stopped the whole
    episode. Up to len(a) frames this is np.convolve(mode="same") itself, so its sums are unchanged."""
    a = np.asarray(a, dtype=np.float64)
    win = max(1, int(win))
    if win <= len(a):
        return np.convolve(a, np.ones(win), mode="same")
    c = np.concatenate([[0.0], np.cumsum(a)])
    i = np.arange(len(a))
    return c[np.minimum(i + (win - 1) // 2 + 1, len(a))] - c[np.maximum(i - win // 2, 0)]


def _r2_rows(xs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """r squared of y against each row of xs, on the instants where both have a reading (0 where either is flat)."""
    ok = np.isfinite(xs) & np.isfinite(y)[None, :]
    out = np.zeros(len(xs))
    for i in range(len(xs)):
        a, b = xs[i][ok[i]], y[ok[i]]
        if len(a) >= 3 and a.std() > 0 and b.std() > 0:
            out[i] = np.corrcoef(a, b)[0, 1] ** 2
    return out


LINE_MIN_POINTS = 3     # a line through two points always fits exactly, so it says nothing about how closely y follows x


def line_through(x, y) -> tuple[float, float] | None:
    """(slope, intercept) of the least-squares line y = intercept + slope x through the pairs where both are finite, or
    None when fewer than LINE_MIN_POINTS pairs are finite or x has no spread among them. Then no line can be stated,
    and a fit would divide by zero inside the solver and stop the whole episode (a partner that never moves outside
    the frames where the pair comes apart). Every straight-line fit on recorded values goes through here."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < LINE_MIN_POINTS or np.ptp(x[ok]) == 0:
        return None
    slope, intercept = np.polyfit(x[ok], y[ok], 1)
    return (float(slope), float(intercept)) if np.isfinite(slope) and np.isfinite(intercept) else None


def tracking(x: np.ndarray, y: np.ndarray, ks: list[int], fps: float) -> tuple[bool, int]:
    """(whether y's readings at the instants ks track x's, how many frames later x moves than y). Whether they track
    is judged on the readings at ks, x read at ks shifted by every lag up to TRACK_LAG_S. Which one moves later is read
    from every frame between the first and last instant (the lag that best lines up their frame-to-frame changes, the
    shortest of equal ones), since one reading a second cannot place a 0.1 s lag: positive means x follows y. Two
    columns of frames by one value; fewer than RESOLVED_MIN_INSTANTS instants never track."""
    ks = np.asarray(ks, dtype=int)
    if len(ks) < RESOLVED_MIN_INSTANTS:
        return False, 0
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    reach = int(round(TRACK_LAG_S * fps))
    lags = np.arange(-reach, reach + 1)
    xs = np.stack([x[np.clip(ks + lag, 0, len(x) - 1)] for lag in lags])
    yk = y[ks]
    same = _r2_rows(xs, yk).max() >= TRACK_LEVEL and _r2_rows(np.diff(xs, axis=1), np.diff(yk)).max() >= TRACK_CHANGE
    if not same:
        return False, 0
    lo, hi = int(ks[0]), int(ks[-1]) + 1
    dx = np.nan_to_num(np.diff(x[lo:hi]))
    dy = np.nan_to_num(np.diff(y[lo:hi]))
    # a jump in one value (a slip, a step) is a few frames of very large change that would pull the alignment its way,
    # so each value's changes are clipped at their own 99th percentile before they are lined up
    for d in (dx, dy):
        cap = float(np.quantile(np.abs(d), 0.99)) if len(d) else 0.0
        if cap > 0:     # a value that moves in fewer than 1% of its frames keeps its changes as they are
            np.clip(d, -cap, cap, out=d)
    m = len(dy)
    # searched to twice the allowed lag: a slow copy shifted by more than TRACK_LAG_S can still fit closely at the
    # window's edge, and lining up best outside the window is what tells it apart from a value that follows
    score = {int(L): float(np.dot(dx[max(L, 0):m + min(L, 0)], dy[max(-L, 0):m - max(L, 0)]))
             for L in range(-2 * reach, 2 * reach + 1) if abs(L) < m}
    # a mirrored copy lines up with the opposite sign, so the size of the agreement decides
    best = max(sorted(score, key=abs), key=lambda L: abs(score[L]))
    return (True, best) if abs(best) <= reach else (False, 0)


def apart_instants(x: np.ndarray, y: np.ndarray, ks: list[int], lag: int, fps: float) -> list[int]:
    """The places in ks of the instants where two values that track each other (tracking, x read lag frames later)
    come apart: every frame from the first to the last instant is read, since one reading a second can fall either
    side of a one second stall, and a frame is apart where the mean |residual| from the straight line through the
    other value, over the TRACK_HOLD_S around it, departs from the pair's own usual residual at that speed (the rule
    above TRACK_HOLD_S), either way round.
    Each run of such frames gives the instants inside it and one more on each side, so a stall between two instants
    still shows at both. Every instant when either value is flat or has too few readings."""
    ks = np.asarray(ks, dtype=int)
    lo, hi = int(ks[0]), int(ks[-1]) + 1
    idx = np.arange(lo, hi)
    xa, ya = x[np.clip(idx + lag, 0, len(x) - 1)], y[idx]
    ok = np.isfinite(xa) & np.isfinite(ya)
    win = max(1, int(round(TRACK_HOLD_S * fps)))
    flagged = np.zeros(len(idx), dtype=bool)
    count = np.maximum(window_sum(ok.astype(float), win), 1)
    for a, b in ((xa, ya), (ya, xa)):
        span = float(np.ptp(b[ok])) if ok.sum() >= 3 else 0.0
        if span == 0 or a[ok].std() == 0:
            return list(range(len(ks)))
        # fitted on the frames where they agree: a long stall would otherwise tilt the line and spread the gap
        # beyond it, so the line is fitted again without the frames the first fit found apart
        use, here = ok, np.zeros(len(idx), dtype=bool)
        for _ in range(2):
            line = line_through(a[use], b[use])
            if line is None:
                # the frames left after a first fit no longer carry a line: there is no usual tracking to depart from
                return list(range(len(ks)))
            fit = line[1] + line[0] * np.where(ok, a, 0.0)
            r = np.where(ok, np.abs(b - fit), 0.0)
            held = window_sum(r, win) / count
            # the pair's own normal: a follower behind its command is off by about its speed times the lag, and a
            # still pair by its noise, so its usual residual is b0 + c x speed, read from the pair itself
            speed = window_sum(np.where(ok, np.abs(np.diff(fit, prepend=fit[0])) * fps, 0.0), win) / count
            # moving means moving: a gripper still for most of the request has a median speed of its noise, and its
            # residual over noise speed would set the lag term from nothing
            moving = ok & (speed > max(float(np.median(speed[ok])), 0.1 * float(np.quantile(speed[ok], 0.99))))
            c = float(np.median(held[moving] / speed[moving])) if moving.any() else 0.0
            still = ok & (speed <= np.quantile(speed[ok], 0.25))
            b0 = float(np.median(held[still])) if still.any() else 0.0
            excess = held - (b0 + c * speed)
            # its spread as a median absolute deviation, which a departure over less than half the request cannot widen
            mid = float(np.median(excess[ok]))
            usual = mid + TRACK_APART_TIMES * 1.4826 * float(np.median(np.abs(excess[ok] - mid)))
            here = ok & (excess > max(usual, TRACK_APART * span))
            if (ok & ~here).sum() < max(3, ok.sum() // 2):
                break
            use = ok & ~here
        flagged |= here
    out = set()
    edges = np.flatnonzero(np.diff(np.concatenate([[0], flagged.astype(int), [0]])))
    for a0, b0 in zip(edges[::2], edges[1::2]):
        first = max(int(np.searchsorted(ks, lo + a0, side="left")) - 1, 0)
        last = min(int(np.searchsorted(ks, lo + b0 - 1, side="right")), len(ks) - 1)
        out.update(range(first, last + 1))
    return sorted(out)


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
                 swing=None, role=None) -> list[tuple[str, list[str]]]:
    """[(row label, one value per sampled instant)] for one signal: its values when it gets one row per value
    (per_value), else the numbers that summarise the array (module docstring). "-" is no reading at that instant."""
    a = _float(a)
    d = a.shape[1]
    if per_value(name, d, shape, names, role):
        labels = names if names and len(names) == d else ([""] if d == 1 else [f"[{i}]" for i in range(d)])
        return [(f"{name}{(' ' + lb) if lb else ''}", [reading_text(a[k, i]) for k in ks]) for i, lb in enumerate(labels)]
    rows = []
    if _role(role) not in MOVEMENT_ROLES and localized(a, rest, swing):
        # the distances, the active values and the activity at the sampled instants alone, as the whole would give them
        level = resting_level(a, rest)
        sw = _swing(a, level, swing)
        dist = {k: np.abs(np.asarray(a[k], dtype=np.float64) - level) for k in ks}
        act = {k: (_away(v, sw) if sw > 0 else np.zeros(d, dtype=bool))
               for k, v in dist.items()}
        unread = {k: not np.isfinite(a[k]).any() for k in ks}
        m = {k: (np.nan if unread[k] else float(np.where(act[k], dist[k], 0.0).sum())) for k in ks}
        rows.append((f"{name} total activity", [reading_text(m[k]) for k in ks]))
        rows.append((f"{name} active values", ["-" if unread[k] else str(int(act[k].sum())) for k in ks]))
        if shape and len(shape) == 2:
            h, w = int(shape[0]), int(shape[1])
            cells = []
            for k in ks:
                if unread[k] or not act[k].any():
                    cells.append("-")
                else:
                    r, c = divmod(int(np.nanargmax(np.where(act[k], dist[k], -1))), w)
                    cells.append(f"r{r}c{c}")
            rows.append((f"{name} strongest cell (row r, column c of {h} x {w})", cells))
    else:
        ch = ["-"]
        for a0, b0 in zip(ks, ks[1:]):
            # a value with no finite reading at either instant has no change (an infinite one too)
            with np.errstate(invalid="ignore"):
                x = np.abs(np.asarray(a[b0], dtype=np.float64) - np.asarray(a[a0], dtype=np.float64))
            x = x[np.isfinite(x)]
            ch.append(reading_text(x.max()) if len(x) else "-")
        rows.append((f"{name} largest change of any value since the instant before", ch))
    return rows


def describe(name: str, a: np.ndarray, shape=None, names=None, rest=None, swing=None, rate_hz=None,
             fps=None, aligned_by=None, camera_aligned_by=None, clock_problem=None,
             source_rows=None, camera_frames=None, unnamed=False, names_source=None) -> str:
    """One line: the signal's name, its shape or value names (with where they were declared, names_source, when a
    description beside the values gave them, prepare/value_names.py), or that its values are unnamed when nothing in
    the recording or its rig description names them (unnamed), its rate when it is recorded slower than the frames
    (rate_hz below RATE_SLOWER of fps), that it was placed from both starts when the reader had no clock in common
    to place it by (aligned_by, prepare/formats.py mark_assumed) or one row per frame when it has as many rows as the
    video has frames (ALIGNED_ROWS), or tied readings placed within each recorded stamp interval as an assumption
    (COARSE_CLOCK, whose rate is estimated from that placement), and the range each value takes, up to PER_VALUE_MAX
    values (for a wider array, the range of all its values together)."""
    a = _float(a)          # as stored: its smallest and largest readings are exact in any precision
    d = a.shape[1]
    unnamed = bool(unnamed) and d > 1 and not (names and len(names) == d)
    what = (f"{' x '.join(str(int(x)) for x in shape)}{' unnamed' if unnamed else ''} values" if shape and len(shape) > 1
            else f"{d}{' unnamed' if unnamed else ''} value{'s' if d > 1 else ''}")
    named = None
    if names and len(names) == d and d <= PER_VALUE_MAX:
        named = ", ".join(names)
        if len(named) <= DESCRIBE_ROW_MAX_CHARS:
            what += " (" + named + ")"
        else:
            what = (f"{d} named values in value order, value names retained in episode metadata"
                    + (f", {' x '.join(str(int(x)) for x in shape)} shape" if shape and len(shape) > 1 else ""))
        if names_source:
            what += f", names from {names_source}"
    if rate_hz and fps and float(rate_hz) < RATE_SLOWER * float(fps):
        rate = reading_text(float(rate_hz))
        what += (f", recorder arrivals at {rate} Hz" if aligned_by == RECORDER_ARRIVAL else
                 f", estimated at {rate} Hz from assumed placement" if aligned_by == COARSE_CLOCK or clock_problem else
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
        return f"{head}: values from {reading_text(np.nanmin(lo))} to {reading_text(np.nanmax(hi))}{tail}"
    if (hi == lo).all():
        value = reading_text(lo[0]) if d == 1 else "[" + ", ".join(reading_text(x) for x in lo) + "]"
        text = f"{head}: {value} throughout{tail}"
    else:
        text = f"{head}: " + ", ".join(reading_text(l) if l == h else f"{reading_text(l)} to {reading_text(h)}"
                                     for l, h in zip(lo, hi)) + tail
    if len(text) <= DESCRIBE_ROW_MAX_CHARS:
        return text
    if named and f" ({named})" in what:
        what = what.replace(f" ({named})", "", 1)
        what += ", value names retained in episode metadata"
    readings = [reading_text(l) + " throughout" if l == h else f"{reading_text(l)} to {reading_text(h)}" for l, h in zip(lo, hi)]
    # Eight ranges fit within the descriptor allowance even at the longest numeric format. The signal name and
    # clock qualifications are never truncated; the allowance bounds value detail, not original source identities.
    groups = [f"    values {i}-{min(i + 7, d - 1)}: " + ", ".join(readings[i:i + 8])
              for i in range(0, d, 8)]
    return "\n".join([f"  {name} ({what}): ranges by value index"] + groups) + tail
