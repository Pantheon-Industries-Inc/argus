"""Sampling and sensor text describe the actual selected evidence."""
import copy

import numpy as np
import pytest

from label import episode as me
from label import signals as sg
from label import state as ms


def episode(n=600, profile="teleop_arms"):
    return {"context": {"dataset": "sample", "fps": 30, "profile": profile, "state_kind": "none",
                        "cameras": {"exo": {"width": 640, "height": 480}}},
            "state": np.zeros((n, 0)), "sources": {"exo": {"n_frames": n}}, "kmap": {}}


@pytest.mark.parametrize("signal_only", [False, True])
def test_stop_and_restart_instants_are_named_without_changing_selection(signal_only):
    ep = episode()
    t = np.arange(600) / 30
    a = np.stack([np.sin(t + j) for j in range(7)], axis=1)
    a[107:251] = a[107]
    if signal_only:
        ep["signals"] = {"observed.joints": a}
    else:
        ep["state"] = a
        ep["context"]["state_kind"] = "joints"
    pl = me.plan(ep)
    spans = pl.get("quiet_spans", pl["spans"])
    assert spans and any(k % 45 for k in pl["ks"][:-1])
    selected = ms.sample_frames(600, spans, fps=30, moving_every_s=1.5, still_every_s=1.5)
    assert pl["ks"] == selected
    text = me.episode_text(ep, pl, 448, 252, (640, 480))
    assert "boundary instants" in text
    assert ("quiet signal spans" if signal_only else "recorded still spans") in text
    if signal_only:
        assert "quiet does not establish that an arm was still" in text
    assert me.plan(ep)["ks"] == selected


def test_ego_signals_do_not_imply_a_missing_hand_layout():
    ep = episode(profile="ego_head")
    ep["signals"] = {"imu.acceleration": np.arange(1800).reshape(600, 3)}
    text = me._no_state_text(ep, me.plan(ep))
    assert "no hand state in the layout" not in text
    assert "no tracked actor state was read" in text
    assert "imu.acceleration" in me._signals_table(ep, me.plan(ep))


def test_terminal_span_has_no_invented_restart_after_the_last_frame():
    ep = episode()
    ep["context"]["state_kind"] = "joints"
    a = np.stack([np.sin(np.arange(600) / 30 + j) for j in range(7)], axis=1)
    a[107:] = a[107]
    ep["state"] = a
    ep["unavailable_instants"] = [0]
    pl = me.plan(ep)
    assert pl["spans"][-1][1] == 599
    assert pl["ks"] == ms.sample_frames(600, pl["spans"], fps=30, moving_every_s=1.5, still_every_s=1.5)
    text = me._instants_line(ep, pl)
    assert "within each stretch between boundaries" in text
    assert "where available" in text
    assert "20.00" not in text


def test_touch_alone_keeps_sensor_and_contact_fields_without_claiming_recorded_motion():
    ep = episode()
    a = 3072 + np.random.default_rng(0).normal(0, 2, (600, 16))
    a[180:240, 5] -= 1000
    ep["signals"] = {"left_glove_pressure": a}
    ep["signal_meta"] = {"left_glove_pressure": {"shape": [4, 4], "swing": 1000}}
    ep["contacts_shown"] = [{"id": "c1", "hand": "left", "signals": ["left_glove_pressure"],
                             "start_s": 6, "end_s": 8, "peak_s": 7, "regions": {}}]
    pl = me.plan(ep)
    assert pl["touch"] == {"left_glove_pressure"}
    fixed, text = me.build_prompt(ep, pl, cell_w=448, cell_h=252)
    assert "recorded motion" not in fixed
    assert "OTHER RECORDED SIGNALS" in text and "CONTACTS:" in text
    assert "contacts" in me.requested_schema(ep, pl)
    assert me.implied_checks(ep, pl) == ("sensor_checks", "contact_checks")
    assert not me.is_recorded(ep, pl)
    ep["signals"]["base.odom"] = np.arange(600)[:, None]
    assert me.is_recorded(ep, me.plan(ep))
    ep["signals"].pop("base.odom")
    ep["context"]["state_kind"] = "joints"
    ep["state"] = np.tile(np.arange(600)[:, None], (1, 7))
    assert me.is_recorded(ep, me.plan(ep))


def test_long_named_vector_keeps_ranges_and_names_in_metadata_with_bounded_description():
    a = np.vstack([np.arange(48), np.arange(48) + 100]).astype(float)
    names = [f"joint_{j}_" + "sensor_coordinate_" * 12 for j in range(48)]
    before = copy.deepcopy(names)
    text = sg.describe("observed.joints", a, names=names)
    assert len(text) < 1000
    assert "48 named values" in text
    assert "0 to 100" in text and "47 to 147" in text
    assert "names retained in episode metadata" in text
    assert names == before


def test_long_descriptor_groups_rows_without_losing_individual_ranges():
    a = np.vstack([np.arange(64) * 1234.56789, np.arange(64) * 1234.56789 + 76543.21])
    names = [f"coordinate_{j}" for j in range(64)]
    text = sg.describe("observed.joints", a, names=names)
    assert max(map(len, text.splitlines())) <= 768
    assert "value names retained in episode metadata" in text
    for lo, hi in zip(*sg.finite_range(a)):
        assert f"{sg._num(lo)} to {sg._num(hi)}" in text


def test_readout_grows_with_selected_instants_and_has_an_explicit_maximum():
    ep = episode(27000)
    ep["signals"] = {"observed.joints": np.stack([np.sin(np.arange(27000) / 200 + j)
                                                for j in range(4)], axis=1)}
    pl = {"n": 27000, "ks": list(range(0, 27000, 30)), "spans": [], "touch": frozenset()}
    lines, whole = me._signal_readout(ep, pl)
    assert whole == {"observed.joints"}
    assert "not even one row fits" not in "\n".join(lines)
    assert me.signal_readout_budget(20) == 12000
    assert me.signal_readout_budget(134) > me.signal_readout_budget(40)
    assert me.signal_readout_budget(100000) == me.SIGNAL_TABLE_MAX_CHARS


def test_oversized_row_does_not_starve_a_smaller_lower_ranked_row():
    ep = episode()
    t = np.arange(600)
    huge = "observed.joints." + "coordinate" * 2000
    ep["signals"] = {huge: np.sin(t / 500)[:, None], "flag": (t % 2)[:, None]}
    pl = {"n": 600, "ks": list(range(0, 600, 30)), "spans": [], "touch": frozenset()}
    lines, whole = me._signal_readout(ep, pl)
    assert whole == {"flag"}
    text = "\n".join(lines)
    assert huge in text and "move least" not in text
    assert "ranked by movement and kept when they fit" in text
    assert "character budget" in text


def test_small_signal_descriptors_and_readouts_remain_exact():
    a = np.array([[1., 2.], [3., 4.]])
    assert sg.describe("sensor", a, names=["x", "y"]) == "  sensor (2 values (x, y)): 1 to 3, 2 to 4"
    ep = episode(2)
    ep["signals"] = {"sensor": a}
    ep["signal_meta"] = {"sensor": {"names": ["x", "y"]}}
    pl = {"n": 2, "ks": [0, 1], "spans": [], "touch": frozenset()}
    assert me._signal_readout(ep, pl) == ([
        '  Each signal that changes, at every instant you receive (seconds in the first row; "-" is no reading):',
        "    at: 0.00 0.03", "    sensor x: 1 3", "    sensor y: 2 4"], frozenset({"sensor"}))
