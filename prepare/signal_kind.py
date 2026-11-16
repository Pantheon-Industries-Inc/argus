"""What kind of stream a recorded signal is, decided at preparation with no paid call, so the annotation policy can
treat each kind its own way (label/policy.py SIGNAL_TREATMENT).

prepare/formats.py write_signals stores the kind in each context.json signal entry as "stream_kind", since "kind" is
already a reader's own key there (an audio channel's "samples"). An episode prepared before that has none, and
label/episode.py load derives it at label time from the same evidence (kind_of), so an old prepared folder labels under
the same policy as a new one.

The evidence, strongest first:
- the entry's own saved kind;
- a signal whose every value never changes over the recording (or whose measured swing across the upload is 0) is
  constant, whatever its name says;
- what the source declares about the field (its description, sensor type or sensor semantics) and the touch verdict
  preparation already made ("touch");
- a bookkeeping word of the field's name, then the kind its MCAP message type says (schema_kind), then the other words
  of its name (prepare/formats.py word_tokens, the split every name matcher uses), in the order of NAME_RULES;
- only for a stream the names leave unknown, the role the paid field dictionary gave it (label/dictionary.py). The
  dictionary refines an unknown stream and never overrides a kind the names or the source declare.

Words are matched whole (prepare/formats.py _names_word), so a covariance is never a variance and reinforcement is
never force. The order of NAME_RULES carries the precedence the kinds need: bookkeeping first (an IMU's covariance is
bookkeeping, not IMU), touch before force (a fingertip force is touch), force/torque before effort (a wrist wrench's
torque is force/torque, an arm's torque is effort), IMU before base (an IMU on a base is IMU), gripper before velocity
and effort (a gripper's effort is the gripper), base before pose (odometry is base motion), velocity and effort before
pose (an end effector's velocity is a velocity), and joints last.
"""
from __future__ import annotations

import numpy as np

KINDS = ("joints", "gripper", "pose", "touch", "force_torque", "imu", "velocity", "effort", "base", "depth",
         "constant", "bookkeeping", "unknown")

# (kind, words any of which in the name decides it), tried in order
NAME_RULES = (
    ("bookkeeping", ("covariance", "health", "status", "diagnostic", "diagnostics", "info", "intrinsic", "intrinsics",
                     "calibration", "done", "dones", "terminal", "first", "last", "discount", "reward", "rewards",
                     "receive", "stamp", "timestamp", "stats", "step", "seq", "sequence", "counter", "tokens",
                     "token", "attention", "mask", "language", "battery", "temperature", "voltage", "latency")),
    ("touch", ("tactile", "tactiles", "touch", "visuotactile", "haptic", "taxel", "skin", "pressure", "contact", "fsr",
               "piezo", "glove")),
    ("force_torque", ("wrench", "ft", "force", "forces")),
    ("depth", ("depth",)),
    ("imu", ("imu", "accel", "accelerometer", "gyro", "gyroscope", "acceleration", "magnetometer", "mag")),
    ("gripper", ("gripper", "jaw", "claw", "width")),
    ("base", ("base", "chassis", "odom", "odometry", "wheel", "wheels", "cmd_vel", "pad", "move", "joystick",
              "controller", "locomotion", "nav", "navigation", "mobile")),
    ("velocity", ("vel", "velocity", "velocities", "qvel", "speed", "speeds", "twist")),
    ("effort", ("effort", "eff", "torque", "torques", "current", "currents", "load")),
    ("pose", ("pose", "poses", "eef", "ee", "keypoint", "keypoints", "landmark", "landmarks", "skeleton", "tf",
              "transforms", "transform", "quat", "orientation", "pos", "position")),
    ("joints", ("joint", "joints", "qpos", "state", "states", "action", "actions", "leader", "follower", "arm",
                "command", "cmd", "wbc", "torso")),
)
# a force on a finger, a fingertip or a palm is a touch, not a wrist force/torque sensor
TOUCH_PARTS = ("finger", "fingers", "fingertip", "fingertips", "palm", "thumb")
# a pose word is a pose only when nothing in the name says it is a joint's or an arm's own reading: joint_pos is a
# joint position, ee_pos an end effector's
JOINT_WORDS = ("joint", "joints", "qpos")
# a dictionary role (label/signals.py MOVEMENT_ROLES, NON_TOUCH_ROLES, TOUCH_QUANTITY_ROLES) as a kind
ROLE_KINDS = {"joint_state": "joints", "actuator_command": "joints", "action": "joints", "state": "joints",
              "end_effector_pose": "pose", "pose": "pose", "landmarks": "pose", "hand_pose": "pose",
              "base_motion": "base", "odometry": "base", "imu": "imu",
              "touch": "touch", "tactile": "touch", "pressure": "touch", "normal_force": "touch",
              "shear_force": "touch", "force_torque": "force_torque",
              "event_flag": "bookkeeping", "annotation": "bookkeeping", "calibration": "bookkeeping",
              "bookkeeping": "bookkeeping"}
DECLARED_KEYS = ("sensor_type", "sensor_semantics", "description")
# ROS and MCAP message types (their last path part) that say what their fields are
SCHEMA_KINDS = {"Imu": "imu", "MagneticField": "imu", "CameraInfo": "bookkeeping", "BatteryState": "bookkeeping",
                "DiagnosticArray": "bookkeeping", "DiagnosticStatus": "bookkeeping", "Odometry": "base",
                "Twist": "base", "TwistStamped": "base", "Joy": "base", "Wrench": "force_torque",
                "WrenchStamped": "force_torque", "Pose": "pose", "PoseStamped": "pose", "PoseArray": "pose",
                "TransformStamped": "pose", "TFMessage": "pose"}


def _words(text) -> list[str]:
    from prepare.formats import word_tokens
    if isinstance(text, dict):
        return [w for v in text.values() for w in _words(v)]
    if isinstance(text, (list, tuple)):
        return [w for v in text for w in _words(v)]
    return word_tokens(text) if isinstance(text, str) else []


def _by_words(words: list[str]) -> str | None:
    found = set(words) | {w[:-1] for w in words if w.endswith("s")}
    for kind, vocabulary in NAME_RULES:
        if kind == "pose" and found & set(JOINT_WORDS):
            continue
        if kind == "force_torque" and found & set(vocabulary) and found & set(TOUCH_PARTS):
            return "touch"
        if found & set(vocabulary):
            return kind
    return None


def _constant(meta: dict, values) -> bool:
    swing = meta.get("swing")
    if type(swing) in (int, float) and swing == 0:
        return True
    if values is None:
        return False
    a = np.asarray(values)
    if not a.size or not np.issubdtype(a.dtype, np.number):
        return False
    a = a.reshape(len(a), -1)
    finite = np.isfinite(a)
    if not finite.any():
        return False
    lo = np.where(finite, a, np.inf).min(axis=0)
    hi = np.where(finite, a, -np.inf).max(axis=0)
    return bool((hi == lo)[finite.any(axis=0)].all())


def schema_kind(schema: str) -> str | None:
    """The kind an MCAP or ROS message type says its fields are (sensor_msgs/msg/Imu: imu), or None when the type says
    nothing of it (a JointState, whose fields' own names decide, or a recorder's untyped {"type": "object"})."""
    if not isinstance(schema, str) or not schema:
        return None
    return SCHEMA_KINDS.get(schema.replace("::", "/").rsplit("/", 1)[-1])


def signal_kind(name: str, meta: dict | None = None, values=None, from_schema: str | None = None) -> str:
    """The kind (KINDS) a signal named name, with its context.json entry meta, optionally its readings and the kind its
    MCAP message type says (schema_kind), is, from the recording alone: no saved kind and no dictionary are read
    (kind_of reads both). The message type decides after a bookkeeping word of the name (an Imu's
    orientation_covariance is bookkeeping) and before the other name words (an Imu's orientation is IMU, not a pose)."""
    meta = meta or {}
    if _constant(meta, values):
        return "constant"
    if meta.get("touch") is True:
        return "touch"
    declared = _by_words(_words([meta.get(k) for k in DECLARED_KEYS]))
    if declared:
        return declared
    words = _words(name)
    by_name = _by_words(words)
    if by_name == "bookkeeping":
        return by_name
    return (from_schema if from_schema in KINDS else None) or by_name or "unknown"


def kind_of(name: str, meta: dict | None = None, values=None) -> str:
    """The kind a label run uses: the entry's saved kind, else signal_kind from the same evidence preparation reads,
    and for a stream that leaves unknown, the paid field dictionary's role (meta["dictionary"], label/episode.py load)
    when it names a kind."""
    meta = meta or {}
    saved = meta.get("stream_kind")
    kind = saved if saved in KINDS else signal_kind(name, meta, values)
    if kind != "unknown":
        return kind
    from label.signals import signal_role
    role = signal_role((meta.get("dictionary") or {}).get("role"))
    return ROLE_KINDS.get(role, "unknown")
