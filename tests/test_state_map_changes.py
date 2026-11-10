"""DiverseUMI-10k records each gripper's pose in the map of the SLAM chunk named by <side>_chunk, and its conventions say
positions in two chunks cannot be compared. Argus read the switch as a single-frame step of 8 to 58 cm, filled short
gaps across it with a mix of two maps, and told the model such a step with no jump in the video is a data issue, so
runs filed false state_video_mismatch issues there. A change of map now blanks that gripper's position and turn across
the switch (its opening is kept), the motion rows say map change, and capture_qc does not count those rows as broken."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from checks import capture_qc as cq
from label import episode as me
from label import state as ms
from prepare import formats as f

FPS = 30.0
N = 300
NAMES = [f"{s}_{v}" for s in ("left", "right") for v in ("x", "y", "z", "rx", "ry", "rz", "gripper")]


def _table(change=150, jump=0.3, lost=None):
    """Two grippers moving smoothly; left_chunk switches at frame change, where the left position jumps by jump m (the
    new map's origin). lost (first, end) marks left frames invalid, as DiverseUMI's state_valid does."""
    t = np.arange(N) / FPS
    one = np.stack([0.1 * np.sin(t), 0.1 * np.cos(t), 0.05 * t, 0.1 * t, 0.05 * t, -0.1 * t, 0.5 + 0.4 * np.sin(t)], 1)
    st = np.concatenate([one, one + 0.2], axis=1)
    st[change:, 0] += jump
    valid = np.ones_like(st, dtype=bool)
    if lost:
        valid[lost[0]:lost[1], :7] = False
    st[~valid] = np.nan
    chunk = np.where(np.arange(N) < change, 3, 4).astype(np.int8)
    df = pd.DataFrame({"frame_index": np.arange(N), "observation.state": list(st),
                       "observation.state_valid": list(valid), "left_chunk": chunk,
                       "right_chunk": np.full(N, 7, dtype=np.int8)})
    return df, st


def _read(df):
    state = f.recorded_validity(df, "observation.state", f._cells(df["observation.state"]))
    return f.state_on_frames(df, state, None, N, FPS, "ee_pose", None, names=NAMES)


def _ep(state, df):
    ctx = {"profile": "handheld_gripper", "state_kind": "ee_pose", "fps": FPS,
           "state_maps": f.state_maps(df, NAMES, state, "ee_pose")}
    src = {v: {"packed": "/nonexistent.mp4", "base_s": 0.0, "n_frames": N} for v in ("left", "right")}
    return {"dir": Path("/tmp/x"), "context": ctx, "sources": src, "state": np.asarray(state, dtype=np.float64),
            "action": None, "times": None, "kmap": {},
            "signals": {c: df[c].to_numpy().astype(np.float64)[:, None] for c in ("left_chunk", "right_chunk")}}


def _largest_step_cm(state, g):
    return float(np.nanmax(np.linalg.norm(np.diff(state[:, 7 * g:7 * g + 3], axis=0), axis=1)) * 100)


def test_no_position_step_is_read_across_a_change_of_map():
    df, _ = _table()
    state, _, kind, _, _ = _read(df)
    assert kind == "ee_pose"
    # main: the 30 cm switch is a recorded single-frame step
    assert _largest_step_cm(state, 0) < 1.0
    assert np.isnan(state[150, :6]).all() and np.isfinite(state[150, 6])          # the opening is kept
    assert np.isfinite(state[:150, :7]).all() and np.isfinite(state[151:, :7]).all()
    assert np.isfinite(state[:, 7:]).all()                                         # the other gripper is untouched
    assert f.state_maps(df, NAMES, state, "ee_pose") == ["left_chunk", "right_chunk"]


def test_a_short_gap_is_never_filled_across_a_change_of_map():
    df, _ = _table(lost=(144, 153))
    state, _, _, _, issues = _read(df)
    # main filled 144-152 between a reading in map 3 and one in map 4, a mix of two maps
    assert np.isnan(state[144:154, :6]).all() and np.isfinite(state[144:154, 6]).all()
    assert np.isfinite(state[:144, :7]).all() and np.isfinite(state[154:, :7]).all()
    assert _largest_step_cm(state, 0) < 1.0
    assert not [i for i in issues if i["kind"] == "state_untracked"]


def test_a_long_untracked_stretch_across_a_change_stays_untracked():
    df, _ = _table(lost=(100, 200))
    state, _, _, _, issues = _read(df)
    assert np.isnan(state[100:200, :7]).all() and np.isfinite(state[200, :7]).all()
    assert [i for i in issues if i["kind"] == "state_untracked"]
    ep = _ep(state, df)
    assert me.map_change_runs(ep) == []
    assert me.untracked_spans(ep, {"n": N}) == [(0, 100, 199)]


def test_without_a_map_column_nothing_changes():
    df, st = _table()
    df = df.drop(columns=["left_chunk", "right_chunk"])
    state = _read(df)[0]
    assert np.isfinite(state).all() and np.allclose(state, st)
    assert f.state_maps(df, NAMES, state, "ee_pose") is None


def test_motion_rows_say_map_change_and_list_no_untracked_stretch():
    df, _ = _table(lost=(144, 153))
    state = _read(df)[0]
    ep = _ep(state, df)
    assert me.map_change_runs(ep) == [(0, 144, 153)]
    assert me.untracked_spans(ep, {"n": N}) == []
    rows = ms.recorded_motion(state, [0, 120, 180, 270], ["left", "right"], moved=me.map_change_frames(ep))
    assert rows[1]["grippers"]["left"]["map_change"] and "move_cm" not in rows[1]["grippers"]["left"]
    assert np.isfinite(rows[1]["grippers"]["left"]["open_a"])
    assert "move_cm" in rows[0]["grippers"]["left"] and "move_cm" in rows[2]["grippers"]["left"]
    assert "move_cm" in rows[1]["grippers"]["right"]
    text = me._motion_table(ep, {"n": N, "ks": [0, 120, 180, 270], "state_span": (0, N)})
    assert "left map change" in text and "neither motion nor a disagreement with the video" in text
    assert "untracked" not in text


def _blip_then_lost():
    """A real 10K pattern (k10 fob_319d48-4b1a68e1-408-8, right frame 30636): the map switches for one frame, back
    again, and the gripper then loses tracking for over a second."""
    df, _ = _table(lost=(152, 200), jump=0.0)
    chunk = np.full(N, 3, dtype=np.int8)
    chunk[151] = 4
    df["left_chunk"] = chunk
    return df


def test_a_one_frame_map_switch_before_an_untracked_stretch_is_a_map_change_not_a_broken_value():
    df = _blip_then_lost()
    state, _, _, _, issues = _read(df)
    assert np.isnan(state[151, :6]).all() and np.isfinite(state[151, 6])
    ep = _ep(state, df)
    ep["context"]["reader_issues"] = issues          # the untracked stretch is its own data issue
    assert me.map_change_runs(ep) == [(0, 151, 151)]
    assert me.untracked_spans(ep, {"n": N}) == [(0, 152, 199)]
    assert cq.canonical_states(ep)["nonfinite"] == []
    rows = ms.recorded_motion(state, [120, 180, 240], ["left", "right"], moved=me.map_change_frames(ep))
    assert rows[0]["grippers"]["left"] == {"untracked": True, "map_change": True}
    text = me._motion_table(ep, {"n": N, "ks": [120, 180, 240], "state_span": (0, N)})
    assert "left untracked, map change" in text


def test_a_map_change_on_the_last_frame_is_not_a_broken_value():
    df, _ = _table(change=N - 1)
    state = _read(df)[0]
    assert np.isnan(state[N - 1, :6]).all()
    ep = _ep(state, df)
    assert me.map_change_runs(ep) == [(0, N - 1, N - 1)]
    assert cq.canonical_states(ep)["nonfinite"] == []


def test_capture_qc_does_not_count_map_change_rows_as_broken_values():
    df, _ = _table(lost=(144, 153))
    state = _read(df)[0]
    cs = cq.canonical_states(_ep(state, df))
    assert cs["nonfinite"] == []
    assert not cs["valid"][144:154, :6].any()            # no step is read across the switch
    broken = state.copy()
    broken[40, 2] = np.nan                               # a real broken value is still reported
    cs = cq.canonical_states(_ep(broken, df))
    assert [e["rows"] for e in cs["nonfinite"]] == [1]
