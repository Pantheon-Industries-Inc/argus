"""DiverseUMI marks each gripper's untracked stretches in observation.state_valid (NaN in observation.state). The reader
dropped the whole state for any gap over 0.5 s, so no kinematic check ran on any of its episodes and the model got the
state only as a raw signal. Each gripper now keeps its tracked stretches, an untracked one stays NaN with a
state_untracked issue, and the motion rows say untracked there."""
from __future__ import annotations

import numpy as np
import pandas as pd

from prepare import formats as f
from label import state as ms

FPS = 30.0


def _table(n=300, right_lost=(0, 120), left_lost=None, fill=np.nan):
    """A two-gripper ee_pose table: each value moves smoothly; the lost frames hold fill and are flagged invalid."""
    t = np.arange(n) / FPS
    one = np.stack([0.1 * np.sin(t), 0.1 * np.cos(t), 0.05 * t, 0.1 * t, 0.05 * t, -0.1 * t, 0.5 + 0.4 * np.sin(t)], 1)
    st = np.concatenate([one, one + 0.2], axis=1)
    valid = np.ones_like(st, dtype=bool)
    for g, lost in ((1, right_lost), (0, left_lost)):
        if lost:
            valid[lost[0]:lost[1], 7 * g:7 * g + 7] = False
    st[~valid] = fill
    df = pd.DataFrame({"frame_index": np.arange(n), "observation.state": list(st),
                       "observation.state_valid": list(valid)})
    return df, st, valid


def _read(df, n=300):
    state = f.recorded_validity(df, "observation.state", f._cells(df["observation.state"]))
    return f.state_on_frames(df, state, None, n, FPS, "ee_pose", None)


def test_a_gripper_untracked_for_4s_keeps_the_other_and_its_own_tracked_frames():
    df, st, valid = _table()
    # the table as DiverseUMI ships it (NaN where invalid), read without the validity helper, as main read it
    state, _, kind, note, issues = f.state_on_frames(df, f._cells(df["observation.state"]), None, 300, FPS, "ee_pose",
                                                     None)
    assert kind == "ee_pose" and note is None
    assert np.isnan(state[:120, 7:]).all() and np.isfinite(state[120:, 7:]).all()
    assert np.allclose(state[:, :7], st[:, :7])
    untracked = [i for i in issues if i["kind"] == "state_untracked"]
    assert len(untracked) == 1 and untracked[0]["t0_s"] == 0.0 and abs(untracked[0]["t1_s"] - 119 / FPS) < 1e-9


def test_values_the_validity_column_marks_invalid_are_no_reading_even_when_they_hold_numbers():
    # zeros where the dataset says the gripper was not tracked had been read as a gripper parked at the origin
    df, st, valid = _table(fill=0.0)
    state, _, kind, _, _ = _read(df)
    assert kind == "ee_pose" and np.isnan(state[:120, 7:]).all()
    assert "observation.state_valid" in f._used_columns(kind, state, None, df=df)


def test_both_grippers_lost_at_the_end_is_declared_untracked_not_a_whole_state_gap():
    df, _, _ = _table(right_lost=(0, 120), left_lost=(220, 300))
    valid = np.stack(df["observation.state_valid"].to_numpy())
    valid[220:, 7:] = False
    df["observation.state_valid"] = list(valid)
    st = np.stack(df["observation.state"].to_numpy()).copy()
    st[~valid] = np.nan
    df["observation.state"] = list(st)
    state, _, kind, _, issues = _read(df)
    assert kind == "ee_pose" and np.isnan(state[220:]).all() and np.isfinite(state[120:220]).all()
    # one issue per gripper, however many stretches it lost
    assert len([i for i in issues if i["kind"] == "state_untracked"]) == 2


def test_a_short_gap_is_still_filled_and_a_table_short_of_its_video_is_still_not_read():
    df, st, _ = _table(right_lost=(100, 106))            # 0.2 s: filled, as fill_rows fills it
    state, _, kind, _, issues = _read(df)
    assert kind == "ee_pose" and np.isfinite(state).all()
    assert [i["kind"] for i in issues] == ["state_filled"]
    # no validity column and no row for the last half of the frames: the whole state stops, read as before
    short = pd.DataFrame({"frame_index": np.arange(150), "observation.state": list(st[:150])})
    _, _, kind, note, _ = f.state_on_frames(short, f._cells(short["observation.state"]), None, 300, FPS, "ee_pose", None)
    assert kind == "none" and "no reading" in note


def test_motion_rows_touching_an_untracked_stretch_claim_no_motion():
    df, _, _ = _table()
    state = _read(df)[0]
    rows = ms.recorded_motion(state, [0, 60, 150, 240], ["left", "right"])
    assert rows[0]["grippers"]["right"] == {"untracked": True}
    assert rows[1]["grippers"]["right"] == {"untracked": True}       # 60-150 crosses the end of the stretch
    assert "move_cm" in rows[2]["grippers"]["right"] and "move_cm" in rows[0]["grippers"]["left"]
