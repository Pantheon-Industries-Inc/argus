"""Every distinct recorded stream our prepared datasets carry keeps the kind (prepare/signal_kind.py) and the treatment
under --signals auto (label/policy.py) it was audited at.

tests/fixtures/signal_kind_real_names.json holds one row per distinct (name, dims, declared hints, MCAP message type,
constancy) read from the context.json and signals.npz of 2026-11-16's prepared episodes: Pantheon YAM teleop, the go2
rosbags and LeRobot sets, BitRobot HIW-500, Galaxea, ABC-130k, RealOmin, MolmoAct2, ALOHA, the Ego IMUs, Digit 360,
the sensor A/B set on pluto and the nodrop regression variants. Each row was judged against what the stream is. A row
whose kind is right in treatment but not in name, or that no rule could settle, carries a note saying so.

kind is what preparation saves (the hints, the readings' constancy, the message type); name_kind is the same stream
when it moves, so a stream that happened to sit still in its episode still pins its name's rule. A rule change that
moves any row fails here, and the fixture is then updated on purpose, with the reason in the commit.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from label.policy import signal_treatment
from prepare.signal_kind import kind_of, schema_kind, signal_kind

ROWS = json.loads((Path(__file__).parent / "fixtures" / "signal_kind_real_names.json").read_text())


def _values(dims: int, constant):
    if constant is None:
        return None
    return np.zeros((2, dims)) if constant else np.array([np.zeros(dims), np.ones(dims)])


def _id(row):
    return f'{row["datasets"][0]}|{row["name"]}|{row["dims"]}|{row["schema"]}|{row["constant"]}|{sorted(row["meta"])}'


@pytest.mark.parametrize("row", ROWS, ids=[_id(r) for r in ROWS])
def test_a_real_stream_keeps_its_kind_and_treatment(row):
    meta = {**row["meta"], "dims": row["dims"]}
    from_schema = schema_kind(row["schema"] or "")
    kind = signal_kind(row["name"], dict(meta), _values(row["dims"], row["constant"]), from_schema=from_schema)
    assert kind == row["kind"]
    assert signal_treatment(kind, "auto")["prompt"] == row["treatment"]
    moving = signal_kind(row["name"], {"dims": row["dims"]}, _values(row["dims"], False), from_schema=from_schema)
    assert moving == row["name_kind"]


@pytest.mark.parametrize("row", [r for r in ROWS if not r["schema"]], ids=lambda r: _id(r))
def test_an_old_prepared_episode_labels_its_stream_as_preparation_would(row):
    """label/episode.py load derives the kind of an episode prepared before stream_kind existed with kind_of, from
    the context entry alone (no message type), so it must agree with what preparation saves now."""
    meta = {**row["meta"], "dims": row["dims"]}
    assert kind_of(row["name"], meta, _values(row["dims"], row["constant"])) == row["kind"]


def test_the_audited_table_covers_the_streams_the_policy_must_get_right():
    """The rows the audit judged by hand (arm and leader joints, grippers, base and controller inputs, humanoid
    bodies, IMU fields, health flags, camera_info, covariances, touch and force/torque) are all still in the table."""
    kinds = {(r["name"], r["dims"]): r["name_kind"] for r in ROWS}
    must = {
        ("/yam_left/joint_state joint_pos", 6): "joints",
        ("/yam_leader_left/joint_pos joint_pos", 7): "joints",
        ("/yam_left/joint_state gripper_pos", 1): "gripper",
        ("/yam_left/joint_state gripper_eff", 1): "gripper",
        ("/left/franka_gripper/joint_states effort", 2): "gripper",
        ("/yam_left/aux_state temp_rotor", 7): "bookkeeping",
        ("/left-ee-state torque", 1): "gripper",
        ("/robot0/sensor/magnetic_encoder", 1): "gripper",
        ("observation.state", 29): "joints",
        ("observation.state.wbc", 23): "joints",
        ("action", 14): "joints",
        ("observation.pad", 4): "base",
        ("observation.move", 1): "base",
        ("action.chassis.velocities", 3): "base",
        ("/utlidar/robot_odom twist.twist.linear", 3): "base",
        ("/utlidar/imu orientation", 4): "imu",
        ("/camera/camera/gyro/sample angular_velocity_covariance", 9): "bookkeeping",
        ("/camera/camera/color/camera_info k", 9): "bookkeeping",
        ("/utlidar/cloud_base", 6): "bookkeeping",
        ("/yam_left/health", 1): "bookkeeping",
        ("observation.tactiles.left_tactile_index_finger_pad", 80): "touch",
        ("/left/ft_sensor/wrench wrench.force", 3): "force_torque",
        ("robot0.action.numerical_joint_acceleration", 7): "velocity",
        ("next.success", 1): "bookkeeping",
    }
    assert {k: kinds.get(k) for k in must} == must
