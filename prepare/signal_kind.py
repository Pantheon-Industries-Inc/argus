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
pose (an end effector's velocity is a velocity), and joints last. Two exceptions keep a ROS name what it is: cmd_vel
(a mobile base's velocity command) is one word, which word_tokens alone splits into cmd and vel, unless the name also
names a joint; and a controller whose name gives a joint or state field and no other base word (/arm_controller/state,
/joint_trajectory_controller/state) is the arm's joints, so it skips base and pose and its field's own word decides.
Three more keep a reading what it is: an IMU word beside a joint word is the joint's rate (a numerical joint
acceleration), "info" beside a transform or pose is the frame it carries, not a setting, and an end effector's reading of
one number (ABC-130k's /left-ee-state velocity and torque) is its gripper. tests/test_signal_kind_real_names.py pins the
kind of every real stream name our prepared datasets carry.
"""
from __future__ import annotations

import re

import numpy as np

KINDS = ("joints", "gripper", "pose", "touch", "force_torque", "imu", "velocity", "effort", "base", "depth",
         "constant", "bookkeeping", "unknown")

# (kind, words any of which in the name decides it), tried in order
NAME_RULES = (
    ("bookkeeping", ("covariance", "health", "status", "diagnostic", "diagnostics", "info", "intrinsic", "intrinsics",
                     "calibration", "done", "dones", "terminal", "first", "last", "discount", "reward", "rewards",
                     "receive", "stamp", "timestamp", "stats", "step", "seq", "sequence", "counter", "tokens",
                     "token", "attention", "mask", "language", "battery", "temperature", "temp", "voltage", "latency",
                     "time", "success", "successful")),
    ("touch", ("tactile", "tactiles", "touch", "visuotactile", "haptic", "taxel", "skin", "pressure", "contact", "fsr",
               "piezo", "glove")),
    ("force_torque", ("wrench", "ft", "force", "forces")),
    ("depth", ("depth",)),
    ("imu", ("imu", "accel", "accelerometer", "gyro", "gyroscope", "acceleration", "magnetometer", "mag")),
    ("gripper", ("gripper", "jaw", "claw", "width")),
    ("base", ("base", "chassis", "odom", "odometry", "wheel", "wheels", "cmd_vel", "pad", "move", "joystick",
              "controller", "locomotion", "nav", "navigation", "mobile")),
    ("velocity", ("vel", "velocity", "velocities", "qvel", "speed", "speeds", "twist", "acceleration",
                  "accelerations")),
    ("effort", ("effort", "eff", "torque", "torques", "current", "currents")),
    ("pose", ("pose", "poses", "eef", "ee", "keypoint", "keypoints", "landmark", "landmarks", "skeleton", "tf",
              "transforms", "transform", "quat", "orientation", "pos", "position", "cartesian", "traj", "trajectory")),
    ("joints", ("joint", "joints", "qpos", "state", "states", "action", "actions", "leader", "follower", "arm",
                "command", "cmd", "wbc", "torso")),
)
# a force on a finger, a fingertip or a palm is a touch, not a wrist force/torque sensor
TOUCH_PARTS = ("finger", "fingers", "fingertip", "fingertips", "palm", "thumb")
# a pose word is a pose only when nothing in the name says it is a joint's or an arm's own reading: joint_pos is a
# joint position, ee_pos an end effector's
JOINT_WORDS = ("joint", "joints", "qpos")
# ROS's mobile base velocity command, kept whole where word_tokens would split it into cmd and vel
CMD_VEL = re.compile(r"(?<![a-z0-9])cmd_vel(?![a-z0-9])")
# a base word that names a ROS controller of any kind: alone with a joint or state word it is an arm controller's state
CONTROLLER_WORDS = ("controller",)
CONTROLLER_STATE_WORDS = ("joint", "joints", "qpos", "state", "states")
# "info" names a message's settings (camera_info, range_info), except beside a transform or pose, where it is the
# message a recorder put a moving frame in (RealOmin's /robot0/sim/robot_info base_link.transform)
INFO_WORDS = ("info",)
INFO_MOVING_WORDS = ("transform", "transforms", "tf", "pose", "poses")
# an end effector's one-number reading is its gripper's (ABC-130k's /left-ee-state velocity, torque)
EE_WORDS = ("ee", "eef")
EE_READINGS = ("velocity", "effort", "pose")
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
                "TransformStamped": "pose", "TFMessage": "pose",
                # a PointCloud2's decoded fields are its header (height, width, point_step, is_dense), not its points
                "PointCloud2": "bookkeeping",
                # RealOmin's foxglove types: its grippers' opening (metres), IMUs and VIO poses
                "MagneticEncoderMeasurement": "gripper", "IMUMeasurement": "imu", "PoseInFrame": "pose"}


def _words(text) -> list[str]:
    from prepare.formats import word_tokens
    if isinstance(text, dict):
        return [w for v in text.values() for w in _words(v)]
    if isinstance(text, (list, tuple)):
        return [w for v in text for w in _words(v)]
    if not isinstance(text, str):
        return []
    words = word_tokens(text)
    return words + ["cmd_vel"] if CMD_VEL.search(text.lower()) else words


def _by_words(words: list[str]) -> str | None:
    found = set(words) | {w[:-1] for w in words if w.endswith("s")}
    if "cmd_vel" in found and found & set(JOINT_WORDS):
        found.discard("cmd_vel")
    controller_state = False
    for kind, vocabulary in NAME_RULES:
        if kind == "bookkeeping" and found & set(vocabulary) <= set(INFO_WORDS) and found & set(INFO_MOVING_WORDS):
            continue
        if kind == "imu" and found & set(JOINT_WORDS):
            continue
        if kind == "base" and found & set(vocabulary) <= set(CONTROLLER_WORDS) and found & set(CONTROLLER_STATE_WORDS):
            controller_state = bool(found & set(CONTROLLER_WORDS))
            if controller_state:
                continue
        if kind == "pose" and (controller_state or found & set(JOINT_WORDS)):
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
    return SCHEMA_KINDS.get(re.split(r"[/.]", schema.replace("::", "/"))[-1])


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
    kind = (from_schema if from_schema in KINDS else None) or by_name or "unknown"
    if kind in EE_READINGS and set(words) & set(EE_WORDS) and _width(meta, values) == 1:
        return "gripper"
    return kind


def _width(meta: dict, values) -> int | None:
    """How many numbers a reading has: the entry's dims, else its values' width."""
    if type(meta.get("dims")) is int:
        return meta["dims"]
    if values is None:
        return None
    a = np.asarray(values)
    return int(np.prod(a.shape[1:])) if a.ndim > 1 else 1 if a.ndim == 1 else None


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
