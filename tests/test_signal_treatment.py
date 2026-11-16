"""The signals setting on every path to the model (label/episode.py, label/sensor_evidence.py,
label/evidence_access.py), and the paths that must not see it (checks/sensors.py, board/sensors.py)."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from label import episode as me
from label import policy
from label import state as ms

pytestmark = pytest.mark.signals_policy

N = 900
JOINT = "/yam_left/joint_state joint_pos"
GRIPPER = "/yam_left/joint_state gripper_eff"
IMU = "ego.gyro_rads"
INFO = "/camera/camera/infra1/camera_info k"
HEALTH = "/yam_left/health"
GLOVE = "left_glove_pressure"


def _signals():
    t = np.arange(N) / 30.0
    rng = np.random.default_rng(0)
    joints = np.stack([0.4 * np.sin(0.3 * t + j) for j in range(6)], axis=1)
    joints[300:450] = joints[300]
    gripper = (0.5 + 0.5 * ((t > 5) & (t < 12))).astype(float)[:, None]
    gripper[300:450] = gripper[300]
    gyro = rng.normal(0, 0.02, (N, 3))
    gyro[600:603, 1] += 3.0            # one spike at 20 s
    gyro[330:340] += 0.5               # a move inside the still span 10 to 15 s
    info = np.tile(np.arange(9, dtype=float), (N, 1))
    health = ((t > 2) & (t < 3)).astype(float)[:, None]
    glove = 3072.0 + rng.normal(0, 2, (N, 16))
    glove[180:240, 5] -= 1000.0
    return {JOINT: joints, GRIPPER: gripper, IMU: gyro, INFO: info, HEALTH: health, GLOVE: glove}


META = {JOINT: {"names": [f"j{i}" for i in range(6)]}, GRIPPER: {}, IMU: {"names": ["x", "y", "z"]},
        INFO: {"shape": [3, 3]}, HEALTH: {}, GLOVE: {"shape": [4, 4], "swing": 1000.0}}


def _episode(mode=None):
    """A video-only teleop episode (state named nowhere, as the Pantheon YAM recorder writes it) with a joint state, a
    gripper effort, an IMU, a constant camera_info, a health flag and a touch glove. mode None is a policy saved before
    the signals setting."""
    ctx = {"dataset": "you/rig", "fps": 30, "profile": "teleop_arms", "state_kind": "none",
           "cameras": {"exo": {"width": 640, "height": 480}}}
    selected = policy.resolve(ctx, **({"signals": mode} if mode else {}))
    if mode is None:
        selected.pop("signals", None)
    ctx["annotation_policy"] = selected
    ep = {"context": ctx, "state": np.zeros((N, 0)), "sources": {"exo": {}},
          "signals": _signals(), "signal_meta": copy.deepcopy(META)}
    spans = [(300, 449)]
    ks = ms.sample_frames(N, spans, fps=30, moving_every_s=1.5, still_every_s=1.5)
    return ep, {"n": N, "ks": ks, "spans": spans, "state_usable": True, "contact": []}


def _readout_rows(text):
    """The signal names that have rows in the values at each instant."""
    rows = text.split("at every instant you receive")[-1].split("\n") if "at every instant you receive" in text else []
    return {name for name in _signals() for line in rows if line.startswith(f"    {name}")}


def test_all_is_the_policy_before_the_setting():
    legacy, pl = _episode(None)
    every, _ = _episode("all")
    assert me._signals_table(legacy, pl) == me._signals_table(every, pl)
    assert me.build_prompt(legacy, pl, cell_w=448, cell_h=336) == me.build_prompt(every, pl, cell_w=448, cell_h=336)
    # every stream that moves has rows, the IMU among them
    assert {JOINT, GRIPPER, IMU} <= _readout_rows(me._signals_table(every, pl))


def test_auto_keeps_joint_gripper_and_touch_and_gives_imu_one_line():
    ep, pl = _episode("auto")
    text = me._signals_table(ep, pl)
    rows = _readout_rows(text)
    assert {JOINT, GRIPPER} <= rows and IMU not in rows
    # the touch glove keeps its treatment: range listed, timing as the contacts, no rows
    assert GLOVE in text and GLOVE not in rows
    imu = [line for line in text.split("\n") if IMU in line]
    assert len(imu) == 1, imu
    line = imu[0]
    assert "moved during the still span 10.00-14.97s" in line and "spike" in line and "20.0" in line, line


def test_auto_names_constant_and_bookkeeping_streams_once_and_gives_nothing_else():
    ep, pl = _episode("auto")
    text = me._signals_table(ep, pl)
    for name in (INFO, HEALTH):
        assert text.count(name) == 1, name
    listed = [line for line in text.split("\n") if INFO in line]
    assert HEALTH in listed[0] and "not given" in listed[0]


def test_off_gives_no_rows_and_names_every_stream_once():
    ep, pl = _episode("off")
    text = me._signals_table(ep, pl)
    assert not _readout_rows(text) and "at every instant" not in text
    assert all(text.count(name) == 1 for name in _signals() if name != GLOVE)
    # the block stays, so the shared instructions still treat it as a recording
    assert me.is_recorded(ep, pl)


def test_auto_is_smaller_and_says_every_stream_exists():
    auto, pl = _episode("auto")
    every, _ = _episode("all")
    a, b = me._signals_table(auto, pl), me._signals_table(every, pl)
    assert len(a) < len(b)
    assert all(name in a for name in _signals())


def test_a_kind_saved_at_prepare_decides_over_the_name():
    ep, pl = _episode("auto")
    ep["signal_kind"] = {**me.signal_kinds(ep), IMU: "joints"}
    assert IMU in _readout_rows(me._signals_table(ep, pl))


def _write_episode(tmp_path, *, kind=False, mode=None):
    """A prepared folder as prepare/formats.py write_signals leaves it (kind=True) or as an older preparation did."""
    from prepare import formats
    d = tmp_path / ("new" if kind else "old")
    d.mkdir()
    ctx = {"dataset": "you/rig", "fps": 30, "profile": "teleop_arms", "state_kind": "none", "n_state_frames": N,
           "cameras": {"exo": {"width": 640, "height": 480}}}
    sig = _signals()
    formats.write_signals(d, ctx, {k: v.astype(np.float32) for k, v in sig.items()})
    if not kind:
        for s in ctx["signals"]:
            s.pop("stream_kind")
    if mode:
        ctx["annotation_policy"] = policy.resolve(ctx, signals=mode)
    (d / "context.json").write_text(json.dumps(ctx))
    (d / "sources.json").write_text(json.dumps({"exo": {"n_frames": N}}))
    return d


def test_prepare_writes_each_signals_kind(tmp_path):
    d = _write_episode(tmp_path, kind=True)
    kinds = {s["name"]: s["stream_kind"] for s in json.loads((d / "context.json").read_text())["signals"]}
    assert kinds == {JOINT: "joints", GRIPPER: "gripper", IMU: "imu", INFO: "constant", HEALTH: "bookkeeping",
                     GLOVE: "touch"}


def test_an_episode_prepared_before_kinds_labels_under_the_same_policy(tmp_path):
    new, old = _write_episode(tmp_path, kind=True), _write_episode(tmp_path)
    a, b = me.load(new), me.load(old)
    assert a["signal_kind"] == b["signal_kind"]
    # the kind stays out of the descriptors every other reader of the signals takes
    assert a["signal_meta"] == b["signal_meta"]


def test_qc_and_the_board_read_every_stream_whatever_the_setting(tmp_path):
    from board.sensors import episode_doc
    from checks.sensors import run_episode
    docs, checks = [], []
    for mode in ("all", "auto", "off"):
        (tmp_path / mode).mkdir()
        d = _write_episode(tmp_path / mode, kind=True, mode=mode)
        docs.append(episode_doc(d))
        checks.append(run_episode(d))
    assert docs[0] == docs[1] == docs[2]
    assert checks[0] == checks[1] == checks[2]
    assert {s["name"] for s in docs[1]["signals"]} == set(_signals())
    assert me.load(d)["signals"].keys() == _signals().keys()


def test_sensor_evidence_gives_force_torque_once_under_auto():
    from label import sensor_evidence as se
    for mode, expected in (("all", True), ("auto", False)):
        ep, pl = _episode(mode)
        t = np.arange(N) / 30.0
        ep["signals"] = {"/left/ft_sensor/wrench wrench.force": np.stack([np.sin(t), 0 * t, np.cos(t)], axis=1)}
        ep["signal_meta"] = {"/left/ft_sensor/wrench wrench.force": {"names": ["fx", "fy", "fz"], "unit": "N"}}
        ep["context"]["signals"] = [{"name": "/left/ft_sensor/wrench wrench.force", "key": "s0", "dims": 3,
                                     "names": ["fx", "fy", "fz"], "unit": "N"}]
        doc = se.build(ep, pl)
        assert bool(doc["sensors"]) is expected, (mode, doc.get("coverage"))
        # its rows stay in the signals block under both
        assert "wrench.force fx:" in me._signals_table(ep, pl)


def test_inspection_lists_what_left_the_rows_but_not_settings_or_bookkeeping(tmp_path):
    from label.evidence_access import Access
    for mode in ("all", "auto"):
        (tmp_path / mode).mkdir()
        ep = me.load(_write_episode(tmp_path / mode, kind=True, mode=mode))
        names = {f["public"]["name"] for f in Access(ep).fields.values() if f["public"]["kind"] == "numeric"}
        assert IMU in names and JOINT in names
        if mode == "all":
            assert {INFO, HEALTH} <= names
        else:
            assert not {INFO, HEALTH} & names


GO2 = Path(__file__).parent / "fixtures" / "signal_policy_go2"


def _go2(tmp_path, mode):
    """A real prepared episode (realworld_go2_task0 episode 0, LeRobot, prepared before signal kinds existed): arm
    state as observation.state and action, the controller pad and move inputs, and front depth."""
    d = tmp_path / mode / "episode_000000"
    shutil.copytree(GO2 / "episode_000000", d)
    ctx = json.loads((d / "context.json").read_text())
    ctx["annotation_policy"] = policy.resolve(ctx, model="openai/gpt-6.1-sol", reasoning="high",
                                              **({} if mode == "legacy" else {"signals": mode}))
    if mode == "legacy":
        # a policy saved before the signals setting existed
        ctx["annotation_policy"].pop("signals")
    (d / "context.json").write_text(json.dumps(ctx))
    ep = me.load(d)
    return ep, me.plan(ep)


@pytest.mark.parametrize("mode", ["legacy", "all"])
def test_all_gives_a_real_episode_the_text_origin_main_gave_it(tmp_path, mode):
    """tests/fixtures/signal_policy_go2/main_869c968.json is what origin/main at 869c968 wrote for this episode."""
    want = json.loads((GO2 / "main_869c968.json").read_text())
    ep, pl = _go2(tmp_path, mode)
    assert len(pl["ks"]) == want["ks"]
    assert me._signals_table(ep, pl) == want["signals"]
    assert me._depth_note(ep) == want["depth"] and me.depth_pictures(ep)


def test_auto_on_the_real_episode_keeps_the_arm_and_one_lines_the_controller(tmp_path):
    ep, pl = _go2(tmp_path, "auto")
    assert me.signal_kinds(ep) == {"observation.state": "joints", "observation.pad": "constant",
                                   "observation.move": "base", "action": "joints"}
    text = me._signals_table(ep, pl)
    rows = text.split("at every instant you receive")[-1]
    assert "    observation.state" in rows and "    observation.move" not in rows
    assert "  observation.move (base motion or controller input, 1 value): 0 to 1" in text
    assert "Also recorded but not given here: observation.pad." in text
    assert not me.depth_pictures(ep) and "Its pictures are not in this request." in me._depth_note(ep)


def test_depth_pictures_ride_only_under_all_and_the_note_says_so():
    every, pl = _episode("all")
    auto, _ = _episode("auto")
    for ep in (every, auto):
        ep["depth"] = {"exo": {"paired_camera": "exo"}}
    assert me.depth_pictures(every) and not me.depth_pictures(auto)
    assert "Detail views are followed by available recorded depth" in me._depth_note(every)
    note = me._depth_note(auto)
    assert note.startswith("\nDEPTH: ") and "Detail views are followed" not in note and "not in this request" in note
