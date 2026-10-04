"""Where each family of the 2026-10-02 coverage audit lands after Phase 2 (tests/fixtures/audit_recordings.py builds
one synthetic recording per container): state read by value names on LeRobot and MCAP and by array name on HDF5,
joint and named signals shown value by value, the readout never dropped whole, video only claims never made, and
the reader's provenance on the board. Every case is converted and its first episode's request built once (no model
call)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
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


class Landed(dict):
    """Prepared audit contexts and the paths holding their original readings."""


@pytest.fixture(scope="module")
def landed(tmp_path_factory):
    root = tmp_path_factory.mktemp("audit")
    out = Landed()
    out.episodes = {}
    for name in rec.CASES:
        up, rig = rec.build(root / "fixtures", name)
        rep = formats.convert(up, rig, root / "eps" / name, name, 900)
        assert not rep["failed"], (name, rep["failed"])
        ep = root / "eps" / name / rep["episodes"][0]["episode_id"]
        ctx = json.loads((ep / "context.json").read_text())
        req = me.build_request(ep)
        prompt = req["prompt"].split("THE EPISODE TO LABEL.", 1)[1]
        board = {}
        # the request stands in for the labelling result: it holds the contacts it found when the context has none
        add_context(board, ctx, ep, req)
        out[name] = (ctx, prompt, board)
        out.episodes[name] = ep
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
    assert "The signals whose names say joints or a state (observation.state, observation.leader_state)" in prompt
    assert "16 values per frame" in board["reader_notes"]["state_note"]
    assert any("cam_high_mask" in x for x in board["reader_notes"]["left_out"]["cameras"])


def test_the_same_bimanual_recording_without_its_depth_camera_lands_the_same_with_no_depth_block(landed):
    """The bimanual case built without cam_high_depth: the same state read the same way, value by value under the
    same 16 names, and no DEPTH block, while the case with the depth camera has one."""
    ctx, prompt, _ = landed["lerobot_v21_bimanual_7dof_nodepth"]
    assert ctx["state_kind"] == "none"
    assert ("observation.state (16 values (left_joint1, left_joint2, left_joint3, left_joint4, left_joint5, "
            "left_joint6, left_joint7, left_gripper, right_joint1") in prompt
    assert "    observation.state left_gripper: " in prompt
    assert "The signals whose names say joints or a state (observation.state, observation.leader_state)" in prompt
    assert "\nDEPTH: " not in prompt and "\nDEPTH: cam_high records depth" in landed["lerobot_v21_bimanual_7dof"][1]


def test_a_single_arm_named_six_joints_and_a_gripper_is_still_its_state(landed):
    ctx, prompt, _ = landed["lerobot_v21_single_arm_6dof"]
    assert ctx["state_kind"] == "joints" and "RECORDED MOTION" in prompt


def test_a_humanoid_and_its_base_are_shown_under_their_own_names(landed):
    ctx, prompt, _ = landed["lerobot_v30_humanoid_base_ft"]
    assert ctx["state_kind"] == "none"
    assert "observation.state (26 values, value names retained in episode metadata): ranges by value index" in prompt
    names = _signals(ctx)["observation.state"]["names"]
    assert len(names) == 26 and names[0] == "left_arm_j1" and names[-1] == "right_hand_pinky"
    assert all(f"    observation.state {name}: " in prompt for name in names)
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


def test_mixed_mcap_camera_clocks_keep_arm_readings_without_inferred_alignment(landed):
    ctx, prompt, board = landed["mcap_json_mobile_manip"]
    assert ctx["state_kind"] == "none" and ctx["state_why"] == "assumed_clock"
    assert ctx["source"]["camera_clock"] == "mixed capture and arrival stamps"
    assert "RECORDED MOTION" not in prompt
    assert "Precise arm alignment was not inferred" in ctx["state_note"]
    assert board["reader_notes"]["state_note"] == ctx["state_note"]
    ep = landed.episodes["mcap_json_mobile_manip"]
    records = {r["topic"]: r for r in ctx["mcap_field_inventory"]}
    with np.load(ep / ctx["recorded_mcap_fields"]) as arrays:
        arm = records["/right_arm/joint_state"]["fields"]
        assert arrays[arm["joint_pos"]].shape == (305, 6)
        assert arrays[arm["gripper_pos"]].shape == (305, 1)
        assert arrays[arm["joint_pos"]].dtype == np.float64
        assert np.isfinite(arrays[arm["joint_pos"]]).all()
        assert arrays[records["/rl/step"]["fields"][""]].shape == (92, 6)
    with np.load(ep / ctx["recorded_camera_ns"]) as clocks:
        assert np.array_equal(clocks["exo"], np.zeros(90, dtype=np.int64))


def test_a_franka_of_seven_named_joints_is_no_state_and_its_gripper_is_no_longer_dropped(landed):
    ctx, prompt, board = landed["mcap_ros2_franka_base"]
    assert ctx["state_kind"] == "none" and "7 joints and no gripper" in ctx["state_note"]
    names = list(_signals(ctx))
    assert any(n.startswith("/left/joint_states") and "position" in n for n in names)
    assert any(n.startswith("/left/franka_gripper/joint_states") and "position" in n for n in names)
    assert "RECORDED MOTION" not in prompt and "RECORDED STATE: no arm state in the layout our checks read." in prompt
    assert "7 joints and no gripper" in board["reader_notes"]["state_note"]
    # the joint names each JointState message gives name its values in the signals too, not [0] to [6]
    assert "    /left/joint_states position fr3_left_joint1: " in prompt
    assert "    /left/joint_states velocity fr3_left_joint7: " in prompt
    assert not any(" position [0]: " in row for row in prompt.splitlines() if "/joint_states " in row)


def test_a_video_folders_imu_keeps_all_ten_values(landed):
    ctx, _, _ = landed["video_folder_sidecars"]
    assert _signals(ctx)["imu"]["dims"] == 10


def test_a_head_camera_with_tracks_is_never_told_it_has_no_tracking(landed):
    ctx, prompt, _ = landed["hdf5_ego_mocap"]
    assert ctx["state_kind"] == "none" and "no hand, head or device tracking" not in prompt
    assert ("RECORDED STATE: no tracked actor state was read; the recorded signals below retain their own "
            "shapes and names.") in prompt
    assert any(c["signals"] == ["right_glove_pressure"] for c in ctx["contacts"])
    contacts = prompt.split("\nCONTACTS: ", 1)[1].split("\nAfter the detail views", 1)[0]
    assert "from right_glove_pressure" in contacts


def test_an_intervention_flag_is_never_a_contact_and_keeps_its_original_rows(landed):
    """An intervention flag rests and rises like a pad, but its name says nothing of touch (label/signals.py
    is_touch), so it times no contact and its values stay in the readout at each instant."""
    for name, (ctx, _, _) in landed.items():
        assert not any("intervention" in s for c in ctx.get("contacts") or [] for s in c["signals"]), name
    ctx, prompt, _ = landed["mcap_json_mobile_manip"]
    ep = landed.episodes["mcap_json_mobile_manip"]
    record = next(r for r in ctx["mcap_field_inventory"] if r["topic"] == "/teleop/intervention")
    with np.load(ep / ctx["recorded_mcap_fields"]) as arrays:
        original = arrays[record["fields"][""]]
        assert original.shape == (92, 1) and set(np.unique(original)) == {0.0, 1.0}
    assert "    /teleop/intervention active: " not in prompt
    assert ctx["state_why"] == "assumed_clock"
