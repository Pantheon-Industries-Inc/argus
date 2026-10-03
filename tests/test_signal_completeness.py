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
