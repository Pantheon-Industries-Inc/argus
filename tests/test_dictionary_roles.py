import copy
import json

import numpy as np
import pytest

from label import contacts as lc
from label import episode as me
from label import signals as sg

RISE = np.r_[np.zeros(30), np.linspace(0, .3, 10), np.full(20, .3)][:, None]


def fixture(name="left_pressure", array=RISE, role="touch", *, shape=None, stored=None):
    context = {"episode_id": "episode_one", "fps": 30, "profile": "teleop_arms", "state_kind": "none",
               "signals": [{"name": name, "key": "s0", "shape": shape or [array.shape[1]], "source": "sensor.h5/pad"}]}
    field = {"id": "field0", "name": name, "kind": "signal", "shape": shape or [array.shape[1]],
             "dtype": str(array.dtype), "names": [], "source": "sensor.h5/pad", "episodes": ["episode_one"],
             "bindings": [{"episode": "episode_one", "context_path": "signals/0", "file": "signals.npz", "key": "s0"}]}
    record = {"schema": 1, "inventory_digest": "controlled", "inventory": {"fields": [field]}, "status": "success",
              "entries": {"field0": {"meaning": "Reviewed measured values", "role": role, "provenance": "machine"}},
              "limitations": []}
    meta = {"shape": shape or [array.shape[1]], "source": "sensor.h5/pad"}
    if stored is not None:
        meta["touch"] = stored
    ep = {"context": context, "signals": {name: array}, "signal_meta": {name: meta},
          "state": np.zeros((len(array), 0)), "sources": {"exo": {}}, "times": None}
    return ep, record


def bind(ep, record, overrides=None):
    from label.dictionary_context import apply_context
    ep = dict(ep)
    ep["context"] = apply_context(ep["context"], record, overrides)
    return ep


def plan(ep):
    return {"n": len(ep["state"]), "ks": [0, 20, 40, len(ep["state"]) - 1], "spans": [], "state_usable": True}


@pytest.mark.parametrize("role", ["event_flag", "actuator_command"])
def test_veto_precedes_stored_touch_and_contacts_but_retains_signal_rows(role):
    ep, record = fixture(role=role, stored=True)
    ep = bind(ep, record)
    assert me.touch_verdicts(ep, 60) == frozenset()
    meta = {"left_pressure": {**ep["signal_meta"]["left_pressure"], "dictionary": record["entries"]["field0"]}}
    assert lc.find(ep["signals"], meta, np.arange(60) / 30, verdicts={"left_pressure"}) == []
    ep["context"]["contacts"] = [{"id": "c1", "signals": ["left_pressure"]}]
    assert lc.of_episode(ep, {"left_pressure"}) == []
    text = me._signals_table(ep, plan(ep))
    assert "left_pressure:" in text.split("    at: ")[1]
    assert "Reviewed measured values" in text and "machine interpretation" in text


def test_positive_touch_role_needs_numbers_and_cannot_override_command_names():
    assert sg.is_touch("channel0", RISE, role="touch")
    assert not sg.is_touch("channel0", np.zeros((60, 1)), role="touch")
    assert not sg.is_touch("actuator_command", RISE, role="touch")
    assert not sg.is_touch("action.pressure", RISE, role="touch")
    ep, record = fixture(name="channel0", role="touch")
    assert me.touch_verdicts(bind(ep, record), 60) == frozenset({"channel0"})
    ep, record = fixture(name="actuator_command", role="touch", stored=True)
    assert me.touch_verdicts(bind(ep, record), 60) == frozenset()


@pytest.mark.parametrize("role", ["", "unknown_open_role"])
def test_blank_or_unknown_role_preserves_legacy_reading_and_human_clearing(role):
    ep, record = fixture(role="event_flag")
    ep = bind(ep, record, {"field0": {"role": role}})
    assert me.touch_verdicts(ep, 60) == frozenset({"left_pressure"})
    from label.dictionary_context import field_interpretation
    entry = field_interpretation(ep["context"], "left_pressure")
    assert entry["role"] == role and entry["provenance"] == "human"
    assert record["entries"]["field0"]["role"] == "event_flag"


@pytest.mark.parametrize("role", ["end_effector_pose", "landmarks", "actuator_command", "imu", "odometry", "base_motion"])
def test_movement_roles_do_not_give_pose_grids_pressure_wording(role):
    values = np.zeros((60, 12))
    values[:, 4] = RISE[:, 0]
    assert sg.localized(values)
    rows = sg.summary_rows("original.grid", values, [0, 40, 59], shape=[3, 4], role=role)
    labels = [row[0] for row in rows]
    assert labels == ["original.grid largest change of any value since the instant before"]
    assert rows[0][1] == ["-", "0.3", "0"]
    assert np.array_equal(values[:, 4], RISE[:, 0])


def test_unnamed_force_torque_vector_gets_all_values_within_existing_budget():
    values = np.arange(60)[:, None] * np.arange(1, 7)[None, :]
    rows = sg.summary_rows("observations/ft_left", values, [0, 59], shape=[6], role="force_torque")
    assert [r[0] for r in rows] == [f"observations/ft_left [{i}]" for i in range(6)]
    assert rows[-1][1] == ["0", "354"]
    ep, record = fixture("observations/ft_left", values, "force_torque", shape=[6])
    lines, _ = me._signal_readout(bind(ep, record), plan(ep))
    assert sum("observations/ft_left [" in line for line in lines) == 6


def test_exact_binding_keeps_variants_and_original_context_unchanged():
    from label.dictionary_context import apply_context, field_interpretation
    ep, record = fixture()
    other = copy.deepcopy(record["inventory"]["fields"][0])
    other.update(id="field1", shape=[2], episodes=["episode_two"])
    other["bindings"][0]["episode"] = "episode_two"
    record["inventory"]["fields"].append(other)
    record["entries"]["field1"] = {"meaning": "Other vector", "role": "event_flag", "provenance": "machine"}
    original = copy.deepcopy(ep["context"])
    context = apply_context(ep["context"], record)
    assert ep["context"] == original
    assert set(context["data_dictionary"]["entries"]) == {"field0"}
    assert field_interpretation(context, "left_pressure")["role"] == "touch"
    assert field_interpretation(context, "pressure") == {}
    context["signals"][0]["key"] = "changed_key"
    assert field_interpretation(context, "left_pressure") == {}
    wrong = apply_context(context, record)
    assert wrong["data_dictionary"]["entries"] == {} and wrong["data_dictionary"]["limitations"]


def test_explicit_upload_owner_binds_a_native_context_without_episode_id():
    from label.dictionary_context import apply_context, field_interpretation
    ep, record = fixture()
    ep["context"].pop("episode_id")
    context = apply_context(ep["context"], record, episode_id="episode_one")
    assert "episode_id" not in context
    assert field_interpretation(context, "left_pressure")["role"] == "touch"
    context["episode_id"] = "other_owner"
    assert field_interpretation(context, "left_pressure") == {}


def test_layout_is_display_interpretation_and_never_enables_arm_checks():
    from label.dictionary_context import apply_context
    ep, record = fixture()
    context = {**ep["context"], "source": {"state": "observation.state"},
               "state_note": "26 columns outside the supported arm layout", "state_why": "layout"}
    field = record["inventory"]["fields"][0]
    field.update(name="observation.state", kind="state", shape=[26], bindings=[
        {"episode": "episode_one", "context_path": "state", "file": "state.npz", "key": "state"}])
    layout = [{"start": 0, "count": 7, "name": "left arm"}, {"start": 7, "count": 7, "name": "right arm"},
              {"start": 14, "count": 6, "name": "left hand"}, {"start": 20, "count": 6, "name": "right hand"}]
    record["entries"]["field0"].update(role="joint_state", layout=layout)
    ep.update(context=apply_context(context, record), state=np.zeros((60, 26)))
    assert me.state_kind(ep) == "none" and not me._has_state(ep, plan(ep))
    text = me._dictionary_text(ep, plan(ep))
    assert "observation.state" in text and "left hand" in text and "columns 14 to 19" in text
    assert "machine interpretation" in text and "supported arm" not in text
    assert ep["state"].shape == (60, 26) and ep["context"]["state_note"] == context["state_note"]


@pytest.mark.parametrize("role", ["touch", "TOUCH"])
def test_long_piece_inherits_original_owner_and_qualified_touch_role(tmp_path, monkeypatch, role):
    from label import pieces
    ep, record = fixture("channel0", role=role)
    ep = bind(ep, record)
    path = tmp_path / "episode_one"
    path.mkdir()
    context = {**ep["context"], "duration_s": 2, "n_state_frames": 60, "cameras": {"exo": {"name": "top"}}}
    (path / "context.json").write_text(json.dumps(context))
    (path / "sources.json").write_text(json.dumps({"exo": {"base_s": 0, "n_frames": 60, "packed": "unused.mp4"}}))
    np.savez(path / "signals.npz", s0=RISE)
    monkeypatch.setattr(pieces, "motion", lambda loaded: (np.arange(60) / 30, np.ones(60)))
    monkeypatch.setitem(pieces.PIECE_MAX_S, "teleop_arms", .5)
    parts = pieces.write_pieces(path, tmp_path / "parts")
    from label.dictionary_context import field_interpretation
    assert len(parts) > 1
    for part in parts:
        loaded = me.load(part)
        assert field_interpretation(loaded["context"], "channel0")["role"] == role
        assert loaded["context"]["data_dictionary"]["episode_id"] == "episode_one"
        assert loaded["signal_meta"]["channel0"]["touch_role"] == "touch"
        assert me.touch_verdicts(loaded, len(loaded["state"])) == frozenset({"channel0"})


@pytest.mark.parametrize('role', ['', 'unknown_open_role'])
@pytest.mark.parametrize('name,expected', [('channel0', False), ('left_pressure', True)])
def test_human_clear_or_unknown_role_cannot_reuse_machine_piece_qualification(role, name, expected):
    ep, record = fixture(name=name, role='touch', array=RISE[-10:], stored=True)
    ep['signal_meta'][name]['touch_role'] = 'touch'
    ep['context']['contacts'] = [{'id': 'c1', 'signals': [name]}]
    reviewed = bind(ep, record, {'field0': {'role': role}})
    touch = frozenset({name}) if expected else frozenset()
    assert me.touch_verdicts(reviewed, 10) == touch
    contacts = lc.of_episode(reviewed)
    assert [c['id'] for c in contacts] == (['c1'] if expected else [])
    assert me._touch(reviewed, {'n': 10, 'touch': frozenset({name})}) == touch
    assert record['entries']['field0']['role'] == 'touch'


def test_original_piece_without_dictionary_qualification_keeps_stored_verdict():
    ep, _ = fixture(name='channel0', array=RISE[-10:], stored=True)
    assert me.touch_verdicts(ep, 10) == frozenset({'channel0'})


@pytest.mark.parametrize('signals', [[], None, 'absent'])
def test_saved_contact_without_named_signals_keeps_timing_and_unknown_fields(tmp_path, signals):
    path = tmp_path / 'episode_one'
    path.mkdir()
    contact = {'id': 'c1', 'start_s': .6, 'end_s': 2.7, 'peak_s': 1.5, 'dips_s': [1.6],
               'from_start': False, 'to_end': False, 'unknown': {'original': [None, 37]}}
    if signals != 'absent':
        contact['signals'] = signals
    context = {'episode_id': 'episode_one', 'fps': 30, 'profile': 'teleop_arms',
               'state_kind': 'none', 'n_state_frames': 90, 'contacts': [contact]}
    (path / 'context.json').write_text(json.dumps(context))
    (path / 'sources.json').write_text(json.dumps({'exo': {'n_frames': 90, 'packed': 'unused.mp4'}}))
    before = (path / 'context.json').read_bytes()
    loaded = me.load(path)
    original = copy.deepcopy(loaded['context'])
    assert lc.of_episode(loaded) == [contact]
    assert loaded['context'] == original
    assert (path / 'context.json').read_bytes() == before


def test_saved_contact_veto_filters_named_signals_without_removing_unattributed_contact():
    ep, record = fixture(name='channel0', role='touch', stored=True)
    ep['signal_meta']['channel0']['touch_role'] = 'touch'
    unbound = {'id': 'unbound', 'signals': [], 'start_s': .6, 'unknown': {'saved': True}}
    mixed = {'id': 'mixed', 'signals': ['channel0', 'left_pressure', 'actuator_command'], 'end_s': 2.7}
    ep['context']['contacts'] = [unbound, mixed]
    reviewed = bind(ep, record, {'field0': {'role': ''}})
    before = copy.deepcopy(reviewed['context'])
    assert lc.of_episode(reviewed) == [unbound, {**mixed, 'signals': ['left_pressure']}]
    assert reviewed['context'] == before
