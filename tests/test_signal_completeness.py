"""Missing rows and missing values remain distinct on every signal surface."""
import warnings
import json

import numpy as np
import pytest

from board import sensors as bs
from checks import sensors as cs
from label import episode as me


def _episode(a):
    return {"context": {"fps": 30, "profile": "teleop_arms", "state_kind": "none"},
            "state": np.zeros((len(a), 0)), "signals": {"qpos": a}, "sources": {"exo": {}}}


@pytest.mark.parametrize("rows,missing,partial", [
    ([[1, np.nan], [1, 2], [1, 2]], 0, 1),
    ([[1, np.nan], [np.nan, 2], [1, 2]], 0, 2),
    ([[1, np.nan], [np.nan, 2]], 0, 2),
    ([[1, np.nan], [np.nan, np.nan], [1, 2]], 1, 1),
])
def test_partial_values_are_named_without_calling_their_rows_unread(rows, missing, partial):
    a = np.array(rows)
    ep = _episode(a)
    pl = {"n": len(a), "ks": [0], "spans": [], "touch": frozenset()}
    text = me._signals_table(ep, pl)
    assert "The same at every frame" not in text
    assert f"partial reading at {partial} of {len(a)} frames" in text
    if not missing:
        assert "no reading at" not in text
    doc = bs.signal_doc({"name": "qpos"}, a, np.arange(len(a)) / 30, 1)
    assert doc["constant"] and doc["partial_reading_frames"] == partial and doc["frames"] == len(a)
    assert doc.get("no_reading_frames", 0) == missing
    notes = cs.signal_findings(ep)
    assert any(x["check"] == "partial_reading" and str(partial) in x["evidence"] for x in notes)
    constant = next(x for x in notes if x["check"] == "constant")
    assert "wherever it reads" in constant["evidence"]


def test_complete_constant_signals_keep_their_existing_words():
    a = np.array([[1., 2.], [1., 2.], [1., 2.]])
    ep = _episode(a)
    text = me._signals_table(ep, {"n": 3, "ks": [0], "spans": [], "touch": frozenset()})
    assert "  The same at every frame: qpos [1, 2]" in text
    assert cs.signal_findings(ep) == [{"check": "constant", "signal": "qpos",
                                      "evidence": "every value of qpos is the same at all 3 frames"}]
    assert bs.signal_doc({"name": "qpos"}, a, np.arange(3) / 30, 1) == {
        "name": "qpos", "dims": 2, "constant": True, "value": [1., 2.]}


@pytest.mark.parametrize("gap", [np.inf, -np.inf, np.nan])
def test_partial_constant_keeps_every_finite_value_without_a_complete_row(gap):
    a = np.array([[1., gap], [np.nan, 2.]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        text = me._signals_table(_episode(a), {"n": 2, "ks": [0, 1], "spans": [], "touch": frozenset()})
    assert "qpos [1, 2] (partial reading at 2 of 2 frames)" in text
    assert "inf" not in text


def test_partial_signal_keeps_an_empty_column_without_a_warning():
    a = np.array([[1., np.nan], [1., np.nan]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        text = me._signals_table(_episode(a), {"n": 2, "ks": [0, 1], "spans": [], "touch": frozenset()})
    assert "partial reading at 2 of 2 frames" in text
    assert "qpos [0]: 1 1" in text and "qpos [1]: - -" in text


def test_a_value_with_no_reading_is_preserved_without_a_warning():
    a = np.array([[1., np.nan], [2., np.nan], [3., np.nan]])
    ep = _episode(a)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        text = me._signals_table(ep, {"n": 3, "ks": [0, 1, 2], "spans": [(0, 2)], "touch": frozenset()})
        doc = bs.signal_doc({"name": "qpos"}, a, np.arange(3) / 30, 1)
        notes = cs.signal_findings(ep)
    assert "partial reading at 3 of 3 frames" in text
    back = bs.dequantize(doc["values"], 3)
    assert np.array_equal(back[:, 0], [1., 2., 3.]) and np.isnan(back[:, 1]).all()
    assert any(x["check"] == "partial_reading" for x in notes)


@pytest.mark.parametrize("real_clock", [False, True])
@pytest.mark.parametrize("rows", [2, 5])
def test_board_signals_use_anchor_frames_even_when_the_only_signal_has_another_length(tmp_path, rows, real_clock):
    (tmp_path / "context.json").write_text(json.dumps({"fps": 10, "n_state_frames": 99,
        "signals": [{"key": "s0", "name": "health"}]}))
    (tmp_path / "sources.json").write_text(json.dumps({"exo": {"n_frames": 3}}))
    a = np.ones((rows, 1))
    if rows > 3:
        a[3:] = [8]
    np.savez(tmp_path / "signals.npz", s0=a)
    if real_clock:
        np.savez(tmp_path / "times.npz", exo=np.array([12., 12.2, 12.5]))
    doc = bs.episode_doc(tmp_path)
    assert doc["frames"] == 3
    assert np.allclose(bs.decode_times(doc["times"]), [0, .2, .5] if real_clock else [0, .1, .2])
    sig = doc["signals"][0]
    assert sig["constant"] and sig["value"] == [1.]
    assert sig.get("no_reading_frames", 0) == (1 if rows == 2 else 0)


@pytest.mark.parametrize("rows", [np.ones((3, 6)), np.arange(18).reshape(3, 6)])
def test_missing_designated_state_does_not_deny_preserved_joint_observations(rows):
    ep = _episode(rows)
    ep["signals"] = {"observation.joint_positions": rows}
    ep["context"].update(state_why="not_recorded", state_note="no observation.state field was recorded")
    pl = {"n": 3, "ks": [0, 1, 2], "spans": [], "touch": frozenset()}
    text = me._no_state_text(ep, pl)
    assert "the recording holds none" not in text
    assert "no arm state in the layout our checks read" in text


def test_a_command_channel_does_not_disprove_absent_observed_state():
    ep = _episode(np.ones((3, 6)))
    ep["signals"] = {"observation.joint_command": ep["signals"]["qpos"]}
    ep["context"].update(state_why="not_recorded")
    assert "as the recording holds none" in me._no_state_text(ep,
        {"n": 3, "ks": [0], "spans": [], "touch": frozenset()})


@pytest.mark.parametrize("name", ["observation.joint_positions", "qpos", "left_joint1", "observation.state",
                                 "observation.arm_state", "robot_state", "gripper_state"])
def test_finite_scalar_observed_joint_or_state_disproves_old_absence(name):
    ep = _episode(np.array([[1.], [3.], [2.]]))
    ep["signals"] = {name: ep["signals"]["qpos"]}
    ep["context"].update(state_why="not_recorded", state_note="no observation.state field was recorded")
    text = me._no_state_text(ep, {"n": 3, "ks": [0, 1, 2], "spans": [], "touch": frozenset()})
    assert "the recording holds none" not in text
    assert "no arm state in the layout our checks read" in text


@pytest.mark.parametrize("name", ["observation.joint_command", "action.state", "observation.joint_force",
    "observation.joint_torque", "observation.touch_state", "observation.tactile_state", "observation.clock_state",
    "observation.timestamp_state", "observation.battery_state", "observation.estop_state", "observation.health_state",
    "observation.haptic_state", "observation.joint_fsr", "observation.joint_piezo", "observation.unrelated_state",
    "temperature_state", "unrelated", "force", "touch", "clock", "battery"])
def test_unrelated_or_command_scalars_do_not_disprove_absent_state(name):
    ep = _episode(np.array([[1.], [3.], [2.]]))
    ep["signals"] = {name: ep["signals"]["qpos"]}
    ep["context"]["state_why"] = "not_recorded"
    assert "as the recording holds none" in me._no_state_text(ep,
        {"n": 3, "ks": [0, 1, 2], "spans": [], "touch": frozenset()})


@pytest.mark.parametrize("rows,n", [(np.array([[np.nan], [np.inf], [-np.inf]]), 3),
                                   (np.array([[np.nan], [np.nan], [2.]]), 2)])
def test_scalar_joint_must_have_a_finite_value_within_the_episode(rows, n):
    ep = _episode(rows)
    ep["signals"] = {"observation.joint_positions": rows}
    ep["context"]["state_why"] = "not_recorded"
    assert "as the recording holds none" in me._no_state_text(ep,
        {"n": n, "ks": [0], "spans": [], "touch": frozenset()})


def test_absent_state_reason_keeps_scalar_observation_wording():
    ep = _episode(np.array([[1.], [3.], [2.]]))
    ep["signals"] = {"observation.joint_positions": ep["signals"]["qpos"]}
    assert me._no_state_text(ep, {"n": 3, "ks": [0, 1, 2], "spans": [], "touch": frozenset()}) == (
        "\nRECORDED STATE: no arm state in the layout our checks read.")


def test_an_assumed_contributor_clock_does_not_claim_all_state_has_that_clock():
    ep = _episode(np.arange(18).reshape(3, 6))
    ep["context"].update(state_why="assumed_clock", state_note="the left follower uses an assumed start")
    text = me._no_state_text(ep, {"n": 3, "ks": [0], "spans": [], "touch": frozenset()})
    assert "recorded only on a clock" not in text
    assert "a contributing state channel" in text and "left follower uses an assumed start" in text


def test_every_shared_state_reason_has_request_wording():
    from prepare.state_notes import STATE_WHY
    assert set(me.STATE_WHY) == set(STATE_WHY)
