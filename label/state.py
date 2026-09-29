"""Recorded state: which frames to send, still spans, and the recorded motion shown to the model.

State arrives per frame as 7 values per arm or handheld gripper: 6 joint angles (rad) + gripper for
kind "joints" (MolmoAct2, ABC-130k, Galaxea), or x y z (m), roll pitch yaw (rad) + opening for kind
"ee_pose" (FastUMI, RealOmin, HABIT). The episode's context.json declares the kind.

State is used for three things, all deterministic:

1. Frame selection: one instant every N s for the whole episode (N per rig, episode.SAMPLE_EVERY_S),
   plus the first and last frame.
2. Still spans: a span is reported only when, over the whole span, every channel stayed inside its
   tolerance. It is handed to the model as the RECORDING'S CLAIM, to check against the video: a
   recording that stopped updating looks exactly like a still span.
3. Recorded motion per sampled interval (joint or pose deltas, gripper values), also as a claim to
   check: mounted cameras are rigid on their gripper, so the video must agree.

State is never turned into claims about grasps, releases, or contact. A closed gripper on a thin
object and an empty closed gripper differ by a few hundredths, so no threshold on it can say
"holding"; the gripper cameras show the fingers in every frame we send.

The joint tolerances are measured on MolmoAct2: joint encoders step in 0.022 deg ticks and a resting
arm dithers within ~0.1 deg over 10 s; a resting gripper reading wanders within ~0.006 of its 0 to 1 range.
STILL_JOINT_DEG (0.25 deg) and STILL_GRIP_FRAC (1% of the gripper's full range, 0.01 there) sit just above that
noise. Grippers are recorded in different units (Galaxea 0 to 100, RealOmni metres, 0 to about 0.10), so the
gripper's tolerance is a share of its own range: the range the episode's context declares ("gripper_range", from
the dataset's adapter), or for an upload the range measured across all its episodes (prepare/formats.py
measure_gripper_range). An episode with neither is taken as a 0 to 1 opening. Measured on Galaxea, a resting
gripper wanders by 0.016 at the 90th percentile and 0.22 at the 99th, under its 1.0 tolerance; the old fixed 0.01
split its still spans.
"""
from __future__ import annotations

import numpy as np

FPS = 30
STILL_JOINT_DEG = 0.25
STILL_GRIP_FRAC = 0.01      # of the gripper's full range
MIN_STILL_S = 3.0          # shorter pauses are already covered by the 1 fps grid
MOVING_EVERY_S = 1.0       # sample period while an arm moves
STILL_EVERY_S = 5.0        # sample period inside a still span (scene can still change)


# State kind "ee_pose" (FastUMI, RealOmin, HABIT) records no joints: the state is each gripper's pose (x, y, z in
# m, roll, pitch, yaw in rad) plus its opening. A still span there means no gripper moved more than 3 mm or
# turned more than 1 degree, and no opening changed by more than 0.01, over the whole span.
STILL_POSE_M = 0.003
STILL_POSE_DEG = 1.0


def gripper_full_range(ctx: dict) -> float | None:
    """The gripper's full range (open minus shut) in its recorded unit, from the context's "gripper_range"
    [shut, open]; None when the context does not give one."""
    r = ctx.get("gripper_range")
    try:
        return abs(float(r[1]) - float(r[0]))
    except (TypeError, ValueError, IndexError, KeyError):
        return None


def still_tolerance(kind: str, dims: int, grip_range: float | None = None) -> np.ndarray:
    """Per-channel range allowed over a still span. kind "joints": 6 joint angles (rad) + gripper per
    arm. kind "ee_pose": x y z (m), roll pitch yaw (rad) + gripper per gripper. The gripper's is STILL_GRIP_FRAC of
    grip_range, its full range in its own unit (None: a 0 to 1 opening)."""
    grip = STILL_GRIP_FRAC * (1.0 if grip_range is None else float(grip_range))
    if kind == "joints":
        per = [np.radians(STILL_JOINT_DEG)] * 6 + [grip]
    elif kind == "ee_pose":
        per = [STILL_POSE_M] * 3 + [np.radians(STILL_POSE_DEG)] * 3 + [grip]
    else:
        raise ValueError(f"unknown state kind {kind!r}")
    if dims % 7:
        raise ValueError(f"state has {dims} values per frame, not a multiple of 7")
    return np.asarray(per * (dims // 7), dtype=np.float64)


def _span_ok(seg: np.ndarray, tol: np.ndarray | None = None) -> bool:
    rng = seg.max(axis=0) - seg.min(axis=0)
    tol = still_tolerance("joints", seg.shape[1]) if tol is None else tol
    return bool((rng <= tol + 1e-12).all())


def still_spans(state: np.ndarray, min_s: float = MIN_STILL_S, fps: float = FPS,
                kind: str = "joints", grip_range: float | None = None) -> list[tuple[int, int]]:
    """Maximal frame intervals [a, b] (inclusive) of at least min_s where every arm (or handheld
    gripper) is at rest, by the whole-span range test above (grip_range: the gripper's full range, see
    still_tolerance). Greedy left to right: grow a span while the range test holds; a span that cannot reach
    min_s advances the start by one frame."""
    s = np.asarray(state, dtype=np.float64)
    T = len(s)
    need = int(round(min_s * fps))
    if T < need:
        return []
    tol = still_tolerance(kind, s.shape[1], grip_range)
    # cheap necessary condition: frame-to-frame motion within tolerance on every channel
    d = np.abs(np.diff(s, axis=0))
    calm = np.concatenate([[True], (d <= tol + 1e-12).all(1)])
    spans, i = [], 0
    while i + need <= T:
        if not calm[i]:
            i += 1
            continue
        lo, hi = s[i].copy(), s[i].copy()
        j = i + 1
        while j < T:
            lo2, hi2 = np.minimum(lo, s[j]), np.maximum(hi, s[j])
            if ((hi2 - lo2) > tol + 1e-12).any():
                break
            lo, hi = lo2, hi2
            j += 1
        if j - i >= need:
            spans.append((i, j - 1))
            i = j
        else:
            i += 1
    for a, b in spans:  # invariant, so a caller can state it as fact
        assert _span_ok(s[a:b + 1], tol), (a, b)
    return spans


def sample_frames(T: int, spans: list[tuple[int, int]], fps: float = FPS,
                  moving_every_s: float = MOVING_EVERY_S, still_every_s: float = STILL_EVERY_S) -> list[int]:
    """Frame indices to send. Every moving stretch is sampled from its first frame at
    moving_every_s; every still span from its first frame at still_every_s; the frame right after
    a still span is the first frame of the next moving stretch, so the instant motion resumes is
    always sent; frame 0 and frame T-1 are always sent."""
    if T <= 0:
        return []
    mv, st = max(1, int(round(moving_every_s * fps))), max(1, int(round(still_every_s * fps)))
    picks = {0, T - 1}
    cur = 0
    for a, b in sorted(spans):
        picks.update(range(cur, a, mv))        # moving stretch [cur, a)
        picks.update(range(a, b + 1, st))      # still span [a, b]
        cur = b + 1                            # motion resumes here
    picks.update(range(cur, T, mv))
    return sorted(p for p in picks if 0 <= p < T)


# Mounted cameras are rigid on their gripper, so the recorded state is checkable against the video, and a
# broken recording (a jump the video does not show, a state that stops updating while the demonstrator keeps
# moving) is a main failure mode. The model gets the recorded motion between consecutive sampled instants
# as a claim to check: pose deltas for end-effector state, joint deltas for joint state.


def _rot_zyx(rpy: np.ndarray) -> np.ndarray:
    """Rotation matrices for roll, pitch, yaw (rad), R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = rpy[..., 0], rpy[..., 1], rpy[..., 2]
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    R = np.empty(rpy.shape[:-1] + (3, 3))
    R[..., 0, 0] = cy * cp; R[..., 0, 1] = cy * sp * sr - sy * cr; R[..., 0, 2] = cy * sp * cr + sy * sr
    R[..., 1, 0] = sy * cp; R[..., 1, 1] = sy * sp * sr + cy * cr; R[..., 1, 2] = sy * sp * cr - cy * sr
    R[..., 2, 0] = -sp;     R[..., 2, 1] = cp * sr;                R[..., 2, 2] = cp * cr
    return R


def _turn_deg(rpy_a: np.ndarray, rpy_b: np.ndarray) -> float:
    Ra, Rb = _rot_zyx(rpy_a), _rot_zyx(rpy_b)
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def recorded_motion(state: np.ndarray, ks: list[int], names: list[str]) -> list[dict]:
    """Per interval between consecutive sampled frames, per gripper (7 values each: x y z m, roll pitch
    yaw rad, opening): net move (cm), the largest single-frame step inside the interval (cm), net turn
    (deg), and the opening at both ends, all straight from the recorded state."""
    s = np.asarray(state, dtype=np.float64)
    out = []
    for a, b in zip(ks, ks[1:]):
        row = {"a": a, "b": b, "grippers": {}}
        for g, name in enumerate(names):
            o = 7 * g
            xyz = s[a:b + 1, o:o + 3]
            steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1) if b > a else np.zeros(1)
            row["grippers"][name] = {
                "move_cm": float(np.linalg.norm(xyz[-1] - xyz[0]) * 100),
                "max_step_cm": float(steps.max() * 100),
                "turn_deg": _turn_deg(s[a, o + 3:o + 6], s[b, o + 3:o + 6]),
                "open_a": float(s[a, o + 6]), "open_b": float(s[b, o + 6]),
            }
        out.append(row)
    return out


def recorded_joint_motion(state: np.ndarray, ks: list[int], names: list[str]) -> list[dict]:
    """Per interval between consecutive sampled frames, per arm (7 values each: 6 joints rad, gripper):
    the largest net change of any joint (deg), the largest single-frame change of any joint (deg), and
    the gripper value at both ends, all straight from the recorded state."""
    s = np.asarray(state, dtype=np.float64)
    out = []
    for a, b in zip(ks, ks[1:]):
        row = {"a": a, "b": b, "arms": {}}
        for g, name in enumerate(names):
            o = 7 * g
            j = s[a:b + 1, o:o + 6]
            steps = np.abs(np.diff(j, axis=0)).max() if b > a else 0.0
            row["arms"][name] = {"max_deg": float(np.degrees(np.abs(j[-1] - j[0]).max())),
                                 "max_step_deg": float(np.degrees(steps)),
                                 "grip_a": float(s[a, o + 6]), "grip_b": float(s[b, o + 6])}
        out.append(row)
    return out
