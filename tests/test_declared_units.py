"""Every signal carries the unit its recording declares for it, with where it was declared, and a signal nothing gives a
unit says so (prepare/value_names.py resolve, through prepare/formats.py write_signals). A unit the readings contradict
is named as contradicted, as an arm state's is."""
import json

import numpy as np

from prepare import formats as f
from prepare import value_names as vn
from test_value_names import _declared_yam_episode, _read

JOINTS = "joint1, joint2, joint3, joint4, joint5, joint6"


def test_units_an_mcap_channel_declares_reach_every_signal_they_describe(tmp_path):
    """robots_realtime declares joint_vel_unit, joint_eff_unit and gripper_vel_unit on each channel. They reached only
    the arm state; the velocity and effort signals reached the model with no unit at all."""
    root = tmp_path / "upload"
    _declared_yam_episode(root)
    _, signals, prompt, ctx = _read(tmp_path, root)
    vel = signals["/yam_left/joint_state joint_vel"]
    assert vel["unit"] == "rad/s" and vel["units_source"] == "yam_left.mcap channel metadata"
    assert signals["/yam_left/joint_state joint_eff"]["unit"] == "N·m"
    assert signals["/yam_right/joint_state gripper_vel"]["unit"] == "normalized/s"
    assert "units_undeclared" not in vel
    assert (f"/yam_left/joint_state joint_eff (6 values ({JOINTS}), names from yam_left.mcap channel metadata, in "
            "N·m from yam_left.mcap channel metadata):") in prompt
    assert ctx["state_kind"] == "joints"


def test_a_signal_nothing_gives_a_unit_says_so_on_the_board_and_not_in_the_prompt(tmp_path):
    root = tmp_path / "upload"
    _declared_yam_episode(root, declare=False)
    _, signals, prompt, _ = _read(tmp_path, root)
    assert signals["/yam_left/joint_state joint_vel"]["units_undeclared"] is True
    assert "/yam_left/joint_state joint_vel (6 unnamed values):" in prompt


def _write(tmp_path, sig, n):
    ep = tmp_path / "ep"
    ep.mkdir()
    ctx = {"n_state_frames": n, "fps": 30.0}
    f.write_signals(ep, ctx, sig)
    return ctx, {s["name"]: s for s in ctx["signals"]}


def test_hdf5_units_from_the_dataset_or_a_group_attribute(tmp_path):
    """A dataset's own unit attribute is the reader's (kept as recorded); a group's {dataset}_unit or
    {dataset}_units attribute is a declaration the same step attaches, with where it came from."""
    import h5py
    n = 10
    path = tmp_path / "rec.h5"
    with h5py.File(path, "w") as h:
        h["obs/force"] = np.zeros((n, 3))
        h["obs/pose"] = np.zeros((n, 2))
        h["obs"].attrs["force_unit"] = "N"
        h["obs"].attrs["pose_units"] = '["m", "rad"]'
    with h5py.File(path, "r") as h:
        claims = vn.h5_claims(h, ["/obs/force", "/obs/pose"])
    sig = f.Signals()
    sig.add("obs/force", np.random.rand(n, 3), source="HDF5 dataset /obs/force")
    sig.add("obs/pose", np.random.rand(n, 2), source="HDF5 dataset /obs/pose")
    sig.add("obs/temp", np.random.rand(n, 1), metadata={"unit": "degC"})
    sig.claims += claims
    _, got = _write(tmp_path, sig, n)
    assert got["obs/force"]["unit"] == "N" and got["obs/force"]["units_source"] == "HDF5 attributes of /obs"
    assert got["obs/pose"]["units"] == ["m", "rad"]
    assert got["obs/temp"]["unit"] == "degC" and "units_source" not in got["obs/temp"]
    assert "units_undeclared" not in got["obs/temp"]


def test_lerobot_feature_and_json_schema_units_are_declarations():
    info = {"features": {"observation.effort": {"dtype": "float32", "shape": [2], "unit": "N·m"}}}
    schema = {"type": "object", "properties": {
        "wrench": {"type": "array", "prefixItems": [{"title": "fx", "unit": "N"}, {"title": "tz", "unit": "N·m"}]},
        "speed": {"type": "number", "unit": "m/s"}}}
    claims = (vn.claims_in(info, "meta/info.json", kinds=("names", "units"))
              + vn.claims_in(schema, "schema /base", ("/base",), ("names", "units")))
    metas = {"observation.effort": {}, "/base wrench": {}, "/base speed": {}, "/base other": {}}
    issues = vn.resolve(metas, {"observation.effort": 2, "/base wrench": 2, "/base speed": 1, "/base other": 1}, claims)
    assert not issues
    assert metas["observation.effort"] == {"unit": "N·m", "units_source": "meta/info.json"}
    assert metas["/base wrench"]["units"] == ["N", "N·m"] and metas["/base wrench"]["names"] == ["fx", "tz"]
    assert metas["/base speed"]["unit"] == "m/s"
    assert metas["/base other"] == {"units_undeclared": True}


def test_units_that_disagree_give_none():
    a = vn.claims_in({"joint_vel_unit": "rad/s"}, "one.yaml", kinds=("units",))
    b = vn.claims_in({"joint_vel_unit": "deg/s"}, "two.yaml", kinds=("units",))
    metas = {"/arm joint_vel": {}}
    issues = vn.resolve(metas, {"/arm joint_vel": 6}, a + b)
    assert issues[0]["kind"] == "value_units_conflict" and metas["/arm joint_vel"] == {"units_undeclared": True}


def test_radians_the_readings_contradict_are_named_and_never_trusted(tmp_path):
    """A signal declared in radians whose readings only degrees explain keeps its declared unit, named as
    contradicted, and raises the same reader issue an arm state does; one whose readings fit radians does not."""
    n = 30
    t = np.linspace(0, 1, n)
    sig = f.Signals()
    sig.add("wrist angle", np.stack([90 * np.sin(3 * t), 120 * np.cos(2 * t)], axis=1), metadata={"unit": "rad"})
    sig.add("elbow angle", np.stack([0.9 * np.sin(3 * t), 1.2 * np.cos(2 * t)], axis=1), metadata={"unit": "rad"})
    ctx, got = _write(tmp_path, sig, n)
    assert got["wrist angle"]["units_contradicted"] is True and "units_contradicted" not in got["elbow angle"]
    issue = next(i for i in ctx["reader_issues"] if i.get("signal") == "wrist angle")
    assert issue["kind"] == f.STATE_UNIT_CONTRADICTED
    from label import signals as sg
    a = np.stack([90 * np.sin(3 * t), 120 * np.cos(2 * t)], axis=1)
    assert ", declared in rad, which its readings contradict" in sg.describe(
        "wrist angle", a, units=["rad", "rad"], units_contradicted=True)


def test_a_value_that_winds_past_one_turn_in_radians_is_not_read_as_degrees(tmp_path):
    """A wheel or a continuous joint recorded in radians reads past one turn by turning, not by being in degrees: it is
    never named as contradicted, while a degree-valued arm beside it still is."""
    n = 300
    t = np.linspace(0, 1, n)
    sig = f.Signals()
    sig.add("wheel angle", (40 * t)[:, None], metadata={"unit": "rad"})
    sig.add("wrist angle", np.stack([90 * np.sin(3 * t), 120 * np.cos(2 * t)], axis=1), metadata={"unit": "rad"})
    ctx, got = _write(tmp_path, sig, n)
    assert "units_contradicted" not in got["wheel angle"] and got["wrist angle"]["units_contradicted"] is True
    assert not any(i.get("signal") == "wheel angle" for i in ctx["reader_issues"])
    assert f.winds_past_one_turn(-40 * t) and not f.winds_past_one_turn(90 * np.sin(3 * t))
    # one reach out in degrees moves in steps of half a turn or more at this rate, so it does not wind
    assert not f.winds_past_one_turn(np.linspace(0, 90, 20))


def test_the_readout_and_the_board_say_each_signals_unit():
    from board.sensors import signal_doc
    from label import signals as sg
    a = np.stack([np.linspace(0, 1, 20), np.linspace(1, 0, 20)], axis=1)
    assert sg.describe("q", a, units=["rad", "normalized"]).startswith("  q (2 values, in rad, normalized):")
    assert sg.describe("q", a, units=["m/s", "m/s"], units_source="rig.yaml").startswith(
        "  q (2 values, in m/s from rig.yaml):")
    # nothing declares a unit: the board and the export say so, the prompt line says nothing of it
    assert sg.describe("q", a).startswith("  q (2 values):")
    t = np.arange(20) / 30
    doc = signal_doc({"name": "q", "unit": "rad/s", "units_source": "rig.yaml"}, a, t, 1)
    assert doc["unit"] == "rad/s" and doc["units_source"] == "rig.yaml"
    assert signal_doc({"name": "q", "units_undeclared": True}, a, t, 1)["units_undeclared"] is True
    assert json.dumps(doc)
