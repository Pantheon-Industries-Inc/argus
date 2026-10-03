"""Where each family of the 2026-10-02 coverage audit lands after Phase 2 (tests/fixtures/audit_recordings.py builds
one synthetic recording per container): state read by value names on LeRobot and MCAP and by array name on HDF5,
joint and named signals shown value by value, the readout never dropped whole, video only claims never made, and
the reader's provenance on the board. Every case is converted and its first episode's request built once (no model
call)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("h5py")
pytest.importorskip("mcap")
pytest.importorskip("mcap_ros2")
pytest.importorskip("av")

from board.build import add_context  # noqa: E402
from label import episode as me  # noqa: E402
from prepare import formats  # noqa: E402

_spec = importlib.util.spec_from_file_location("audit_recordings",
                                               Path(__file__).parent / "fixtures" / "audit_recordings.py")
rec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rec)


@pytest.fixture(scope="module")
def landed(tmp_path_factory):
    root = tmp_path_factory.mktemp("audit")
    out = {}
    for name in rec.CASES:
        up, rig = rec.build(root / "fixtures", name)
        rep = formats.convert(up, rig, root / "eps" / name, name, 900)
        assert not rep["failed"], (name, rep["failed"])
        ep = root / "eps" / name / rep["episodes"][0]["episode_id"]
        ctx = json.loads((ep / "context.json").read_text())
        prompt = me.build_request(ep)["prompt"].split("THE EPISODE TO LABEL.", 1)[1]
        board = {}
        add_context(board, ctx, ep)
        out[name] = (ctx, prompt, board)
    return out


def _signals(ctx):
    return {s["name"]: s for s in ctx.get("signals") or []}


def test_no_family_has_its_readout_dropped_or_a_video_only_claim(landed):
    for name, (ctx, prompt, _) in landed.items():
        if ctx.get("signals"):
            assert "Each signal that changes, at every instant you receive" in prompt, name
        assert "because not even one row fits" not in prompt, name
        if ctx.get("signals") or ctx["state_kind"] != "none":
            assert "this dataset records no" not in prompt, name


def test_a_bimanual_state_of_seven_joints_and_a_gripper_per_arm_is_shown_value_by_value(landed):
    """observation.state [16] (two 7 DoF arms and their grippers) is no layout the checks read; its 16 names reach
    the model, each value at every instant, and the reader's note and left out mask camera reach the board."""
    ctx, prompt, board = landed["lerobot_v21_bimanual_7dof"]
    assert ctx["state_kind"] == "none"
    assert ("observation.state (16 values (left_joint1, left_joint2, left_joint3, left_joint4, left_joint5, "
            "left_joint6, left_joint7, left_gripper, right_joint1") in prompt
    assert "    observation.state left_gripper: " in prompt
    assert "The joint readings it records (" in prompt
    assert "16 values per frame" in board["reader_notes"]["state_note"]
    assert any("cam_high_mask" in x for x in board["reader_notes"]["left_out"]["cameras"])


def test_a_single_arm_named_six_joints_and_a_gripper_is_still_its_state(landed):
    ctx, prompt, _ = landed["lerobot_v21_single_arm_6dof"]
    assert ctx["state_kind"] == "joints" and "RECORDED MOTION" in prompt


def test_a_humanoid_and_its_base_are_shown_under_their_own_names(landed):
    ctx, prompt, _ = landed["lerobot_v30_humanoid_base_ft"]
    assert ctx["state_kind"] == "none"
    assert "observation.state (26 values (left_arm_j1," in prompt and "right_hand_pinky" in prompt
    assert "    observation.base.odom vx: " in prompt and "observation.base.odom total activity" not in prompt


def test_a_handheld_quaternion_pose_stays_a_named_signal(landed):
    ctx, prompt, _ = landed["lerobot_v21_handheld_quat"]
    assert ctx["state_kind"] == "none" and "observation.state (8 values (" in prompt


def test_alohas_qpos_is_the_state_with_its_action(landed):
    ctx, prompt, _ = landed["hdf5_aloha_multirate"]
    assert ctx["state_kind"] == "joints" and ctx["source"]["state"].endswith("qpos")
    names = set(_signals(ctx))
    assert not any(n.endswith("qpos") for n in names) and "action" not in names
    assert any(n.endswith("qvel") for n in names) and "RECORDED MOTION" in prompt


def test_robomimics_joint_positions_stay_a_signal_and_its_rate_is_kept(landed):
    ctx, _, board = landed["hdf5_robomimic"]
    assert ctx["state_kind"] == "none" and abs(ctx["fps"] - 20.0) < 0.1
    assert any(n.endswith("robot0_joint_pos") for n in _signals(ctx))
    assert "45 values per frame" in board["reader_notes"]["state_note"]


def test_an_mcap_arm_with_its_gripper_field_is_the_state_and_named_messages_go_value_by_value(landed):
    ctx, prompt, _ = landed["mcap_json_mobile_manip"]
    assert ctx["state_kind"] == "joints" and "RECORDED MOTION" in prompt
    assert "    /rl/step reward: " in prompt and "/rl/step total activity" not in prompt


def test_a_franka_of_seven_named_joints_is_no_state_and_its_gripper_is_no_longer_dropped(landed):
    ctx, prompt, board = landed["mcap_ros2_franka_base"]
    assert ctx["state_kind"] == "none" and "7 joints and no gripper" in ctx["state_note"]
    names = list(_signals(ctx))
    assert any(n.startswith("/left/joint_states") and "position" in n for n in names)
    assert any(n.startswith("/left/franka_gripper/joint_states") and "position" in n for n in names)
    assert "RECORDED MOTION" not in prompt and "RECORDED STATE: no arm state in the layout our checks read." in prompt
    assert "7 joints and no gripper" in board["reader_notes"]["state_note"]


def test_a_video_folders_imu_keeps_all_ten_values(landed):
    ctx, _, _ = landed["video_folder_sidecars"]
    assert _signals(ctx)["imu"]["dims"] == 10


def test_a_head_camera_with_tracks_is_never_told_it_has_no_tracking(landed):
    ctx, prompt, _ = landed["hdf5_ego_mocap"]
    assert ctx["state_kind"] == "none" and "no hand, head or device tracking" not in prompt
    assert "RECORDED STATE: no hand state in the layout our checks read." in prompt


def test_an_intervention_flag_is_never_a_contact_and_keeps_its_row(landed):
    """An intervention flag rests and rises like a pad, but its name says nothing of touch (label/signals.py
    is_touch), so it times no contact and its values stay in the readout at each instant."""
    for name, (ctx, _, _) in landed.items():
        assert not any("intervention" in s for c in ctx.get("contacts") or [] for s in c["signals"]), name
    _, prompt, _ = landed["mcap_json_mobile_manip"]
    assert "    /teleop/intervention active: " in prompt
