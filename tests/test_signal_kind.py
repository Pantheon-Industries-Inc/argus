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
    "/left-ee-state velocity": "velocity",
    # end effector, hand and body poses
    "observation.ee_pose": "pose",
    "obs/robot0_eef_pos": "pose",
    "/right_arm/ee_pose position": "pose",
    "/hands/left poses": "pose",
    "observation.hand_keypoints.left": "pose",
    "head_pose": "pose",
    "right_wrist_pos": "pose",
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
    # nothing in the name says what it is
    "/d360_0/ght_topic": "unknown",
    "hold_traj": "unknown",
    "gaze": "unknown",
}


@pytest.mark.parametrize("name,kind", sorted(REAL.items()))
def test_real_names_get_their_kind(name, kind):
    assert signal_kind(name, {}) == kind


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
    meta = {"kind": "imu"}
    assert kind_of("observation.state", meta) == "imu"
    # the paid dictionary's role names an unknown stream, never overrides one the names declare
    assert kind_of("/d360_0/ght_topic", {"dictionary": {"role": "base_motion"}}) == "base"
    assert kind_of("observation.state", {"dictionary": {"role": "imu"}}) == "joints"
    # a saved kind that is not one of KINDS (a later preparation's word) is derived again
    assert kind_of("ego.gyro_rads", {"kind": "spaceship"}) == "imu"


def test_an_mcap_message_type_decides_below_bookkeeping_words_and_above_the_other_name_words():
    from prepare.signal_kind import schema_kind
    assert schema_kind("sensor_msgs/msg/Imu") == "imu"
    assert schema_kind("geometry_msgs/msg/WrenchStamped") == "force_torque"
    assert schema_kind("sensor_msgs/msg/JointState") is None and schema_kind('{"type": "object"}') is None
    # an IMU topic named for its mount is IMU by its message type, though its name alone reads as a pose
    assert signal_kind("/body/upper orientation", {}) == "pose"
    assert signal_kind("/body/upper orientation", {}, from_schema="imu") == "imu"
    assert signal_kind("/body/upper orientation_covariance", {}, from_schema="imu") == "bookkeeping"
