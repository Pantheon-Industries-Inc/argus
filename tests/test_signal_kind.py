"""The stream kind every prepared signal carries (prepare/signal_kind.py), and the treatment label/policy.py gives it.

The names below are real signal names read from prepared episodes of the datasets Argus has labelled (Pantheon YAM
teleop, the Ego wrist and head IMUs, go2 rosbags and LeRobot, Galaxea, BitRobot HIW-500, ABC-130k, RealOmin, the
nodrop regression fixtures), so a classifier that only fits made-up names fails here.
"""
import numpy as np
import pytest

from prepare.signal_kind import KINDS, kind_of, signal_kind

REAL = {
    # Pantheon YAM teleop (robots_realtime MCAP, unnamed channels)
    "/yam_left/joint_state joint_pos": "joints",
    "/yam_left/joint_state joint_vel": "velocity",
    "/yam_left/joint_state joint_eff": "effort",
    "/yam_left/joint_state gripper_pos": "gripper",
    "/yam_left/joint_state gripper_vel": "gripper",
    "/yam_left/joint_state gripper_eff": "gripper",
    "/yam_leader_left/joint_pos joint_pos": "joints",
    "/yam_leader_left/health": "bookkeeping",
    # Ego pilot
    "ego.accel_mps2": "imu",
    "ego.gyro_rads": "imu",
    "left_wrist.accel_mps2_std": "imu",
    "right_wrist.gyro_rads": "imu",
    # go2 rosbag (RealSense d435i)
    "/camera/camera/accel/sample angular_velocity": "imu",
    "/camera/camera/gyro/sample linear_acceleration": "imu",
    "/camera/camera/accel/sample orientation_covariance": "bookkeeping",
    "/camera/camera/gyro/sample angular_velocity_covariance": "bookkeeping",
    "/camera/camera/infra1/camera_info k": "bookkeeping",
    "/camera/camera/infra2/camera_info": "bookkeeping",
    # go2 LeRobot controller inputs
    "observation.pad": "base",
    "observation.move": "base",
    # LeRobot and HDF5 arm datasets
    "observation.state": "joints",
    "observation.leader_state": "joints",
    "action": "joints",
    "actions": "joints",
    "observation.state.wbc": "joints",
    "observation.effort": "effort",
    "observations/effort": "effort",
    "observations/qvel": "velocity",
    "obs/robot0_joint_vel": "velocity",
    "obs/robot0_joint_pos": "joints",
    "obs/robot0_gripper_qpos": "gripper",
    "observation.gripper_width": "gripper",
    "/left/franka_gripper/joint_states effort": "gripper",
    "/left/joint_states position": "joints",
    "/left/joint_states effort": "effort",
    "/left/joint_command position": "joints",
    "/left-arm-state torque": "effort",
    # end effector, hand and body poses
    "observation.ee_pose": "pose",
    "obs/robot0_eef_pos": "pose",
    "/right_arm/ee_pose position": "pose",
    "/hands/left poses": "pose",
    "observation.hand_keypoints.left": "pose",
    "head_pose": "pose",
    "right_wrist_pos": "pose",
    # FreeTacMan's Hold_10_traj.csv (the TCP's position and rotation) and habit_two's cartesian targets
    "hold_traj": "pose",
    "robot0.action.target_cartesian_delta": "pose",
    # force/torque
    "/left/ft_sensor/wrench wrench.force": "force_torque",
    "/right_arm/wrist_ft torque": "force_torque",
    "observation.wrench.left_wrist": "force_torque",
    "observation.ft_wrist": "force_torque",
    "observations/ft_left": "force_torque",
    # touch
    "observation.tactiles.left_tactile_index_finger_pad": "touch",
    "observation.tactile_left_raw": "touch",
    "observations/tactile_left": "touch",
    "/tactile/right/pressure pressure": "touch",
    "right_glove_pressure": "touch",
    "/d360_0/pressure_topic": "touch",
    "observation.finger_force.right": "touch",
    # base motion and odometry
    "/base/odom twist": "base",
    "/odom pose.pose.position": "base",
    "/base/wheels speeds": "base",
    "observation.base.wheel_speed": "base",
    "action.chassis.velocities": "base",
    "observation.state.chassis": "base",
    # a mobile base's velocity command, with or without its Twist message type
    "/cmd_vel linear": "base",
    "/cmd_vel angular": "base",
    "/mobile_base/cmd_vel linear.x": "base",
    "/base_controller/state": "base",
    # a ROS arm controller's state is the arm's joints, and a joint's own vel or cmd stays the joint's
    "/arm_controller/state actual.positions": "joints",
    "/joint_trajectory_controller/state position": "joints",
    "/joint_trajectory_controller/state velocity": "velocity",
    "/arm_controller/state actual.effort": "effort",
    "/yam_left/joint_cmd vel": "velocity",
    "/yam_left/joint_state joint_cmd_vel": "velocity",
    "action.cmd vel": "velocity",
    # IMU on a base is still IMU
    "/base/imu angular_velocity": "imu",
    "observation.state.chassis.imu": "imu",
    "/robot0/sensor/imu linear_acceleration": "imu",
    # bookkeeping: done flags, rewards, receive times, stats, language tokens
    "next.done": "bookkeeping",
    "is_terminal": "bookkeeping",
    "discount": "bookkeeping",
    "rewards": "bookkeeping",
    "observation.tactile_left_receive_time_s": "bookkeeping",
    "/recorder/stats": "bookkeeping",
    "observation.language.tokens": "bookkeeping",
    "/odom pose.covariance": "bookkeeping",
    "/slam/health": "bookkeeping",
    # the YAM motors' temperatures, outcome flags and a sensor bundle's clock
    "/yam_left/aux_state temp_rotor": "bookkeeping",
    "/yam_right/joint_state temp_mos": "bookkeeping",
    "next.success": "bookkeeping",
    "robot0.observation.state.prev_command_successful": "bookkeeping",
    "observation.sensor_bundle_time_s": "bookkeeping",
    # a joint's acceleration is a rate of the joint, not an IMU
    "robot0.action.numerical_joint_acceleration": "velocity",
    # nothing in the name says what it is
    "/d360_0/ght_topic": "unknown",
    "gaze": "unknown",
}


@pytest.mark.parametrize("name,kind", sorted(REAL.items()))
def test_real_names_get_their_kind(name, kind):
    assert signal_kind(name, {}) == kind


def test_an_end_effectors_one_number_reading_is_its_gripper():
    """ABC-130k's /left-ee-state carries the 1-DOF gripper's velocity and torque; an end effector's pose stays a pose."""
    one = np.array([[0.0], [1.0]])
    assert signal_kind("/left-ee-state velocity", {"dims": 1}) == "gripper"
    assert signal_kind("/left-ee-state torque", {}, one) == "gripper"
    assert signal_kind("observation.ee_pose", {"dims": 7}) == "pose"
    assert signal_kind("/right_arm/ee_pose position", {}, np.hstack([one] * 3)) == "pose"


def test_a_message_type_decides_info_and_point_cloud_streams():
    """RealOmin's foxglove types name their package with a dot; a PointCloud2's decoded fields are its header; an info
    message that carries a moving transform is that transform, while camera_info and range_info stay settings."""
    from prepare.signal_kind import schema_kind
    assert schema_kind("foxglove.MagneticEncoderMeasurement") == "gripper"
    assert schema_kind("foxglove.IMUMeasurement") == "imu"
    assert schema_kind("sensor_msgs/msg/PointCloud2") == "bookkeeping"
    assert signal_kind("/utlidar/cloud_base", {}, from_schema="bookkeeping") == "bookkeeping"
    assert signal_kind("/robot0/sim/robot_info base_link.transform.position", {}) == "base"
    assert signal_kind("/camera/front/camera_info K", {}) == "bookkeeping"
    assert signal_kind("/utlidar/range_info point", {}) == "bookkeeping"


def test_every_kind_is_listed():
    assert set(REAL.values()) <= set(KINDS)


def test_a_signal_that_never_moves_is_constant_whatever_its_name():
    a = np.tile(np.array([[0.1, 0.2, 0.3]], dtype=np.float32), (50, 1))
    assert signal_kind("/yam_left/joint_state joint_pos", {}, a) == "constant"
    assert signal_kind("/yam_left/health", {"swing": 0.0}) == "constant"
    moving = a + np.linspace(0, 1, 50, dtype=np.float32)[:, None]
    assert signal_kind("/yam_left/joint_state joint_pos", {}, moving) == "joints"


def test_declared_touch_verdict_and_sensor_type_decide_before_the_name():
    assert signal_kind("/d360_0/ght_topic", {"touch": True}) == "touch"
    assert signal_kind("channel_7", {"sensor_type": "force_torque"}) == "force_torque"
    assert signal_kind("channel_7", {"description": "6-axis IMU accelerometer"}) == "imu"


def test_a_saved_kind_is_kept_and_the_dictionary_refines_only_unknown():
    meta = {"stream_kind": "imu"}
    assert kind_of("observation.state", meta) == "imu"
    # the paid dictionary's role names an unknown stream, never overrides one the names declare
    assert kind_of("/d360_0/ght_topic", {"dictionary": {"role": "base_motion"}}) == "base"
    assert kind_of("observation.state", {"dictionary": {"role": "imu"}}) == "joints"
    # a saved kind that is not one of KINDS (a later preparation's word) is derived again
    assert kind_of("ego.gyro_rads", {"stream_kind": "spaceship"}) == "imu"


def test_an_mcap_message_type_decides_below_bookkeeping_words_and_above_the_other_name_words():
    from prepare.signal_kind import schema_kind
    assert schema_kind("sensor_msgs/msg/Imu") == "imu"
    assert schema_kind("geometry_msgs/msg/WrenchStamped") == "force_torque"
    assert schema_kind("sensor_msgs/msg/JointState") is None and schema_kind('{"type": "object"}') is None
    # an IMU topic named for its mount is IMU by its message type, though its name alone reads as a pose
    assert signal_kind("/body/upper orientation", {}) == "pose"
    assert signal_kind("/body/upper orientation", {}, from_schema="imu") == "imu"
    assert signal_kind("/body/upper orientation_covariance", {}, from_schema="imu") == "bookkeeping"


def test_write_signals_turns_a_readers_message_type_into_the_saved_kind(tmp_path):
    """mcap_signals leaves its message type's kind in the reader's meta; write_signals decides the saved kind from it
    and every other reader of the entry sees only that."""
    import json
    from prepare import formats
    n = 120
    sig = formats.Signals()
    moving = np.stack([np.sin(np.arange(n) / 9.0 + j) for j in range(4)], axis=1).astype(np.float32)
    sig["/body/upper orientation"] = moving
    sig.meta["/body/upper orientation"] = {"schema_kind": "imu", "names": ["x", "y", "z", "w"]}
    sig["/body/upper pose"] = moving
    sig.meta["/body/upper pose"] = {"names": ["x", "y", "z", "w"]}
    # a reader's own "kind" (mcap_signals marks an audio channel kept as its loudness "samples") is kept as it was
    sig["/d360_0/mic_0 data"] = moving[:, :2]
    sig.meta["/d360_0/mic_0 data"] = {"kind": "samples", "sample_rate": 48000.0,
                                      "names": ["root mean square", "peak"]}
    ctx = {"fps": 30, "n_state_frames": n, "profile": "teleop_arms", "state_kind": "none"}
    formats.write_signals(tmp_path, ctx, sig)
    kinds = {s["name"]: s["stream_kind"] for s in ctx["signals"]}
    assert kinds == {"/body/upper orientation": "imu", "/body/upper pose": "pose", "/d360_0/mic_0 data": "unknown"}
    assert [s.get("kind") for s in ctx["signals"]] == [None, None, "samples"]
    assert "schema_kind" not in json.dumps(ctx["signals"])
