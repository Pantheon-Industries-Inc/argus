"""The capture-check adapter on synthetic arrays (no video, no sidecars on disk). tests/test_checks.py runs the
capture checks on a decoded video."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from checks import capture_qc as cq
from checks.vendor import public_dataset_adapter_qc as up
from label import state as ms


def test_rotation_convention():
    """Our rpy -> rotation vector must be the rotation label.state._rot_zyx builds (Rz Ry Rx)."""
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(0)
    rpy = rng.uniform(-3.0, 3.0, size=(200, 3))
    ours = ms._rot_zyx(rpy)
    theirs = Rotation.from_euler("xyz", rpy).as_matrix()
    assert np.abs(ours - theirs).max() < 1e-12


def _ep(state, kind="ee_pose", profile="handheld_gripper", views=("left", "right"), n_frames=None, kmap=None,
        gripper_value=None, times=None):
    T = len(state)
    ctx = {"profile": profile, "state_kind": kind, "fps": 30}
    if gripper_value:
        ctx["gripper_value"] = gripper_value
    src = {v: {"packed": "/nonexistent.mp4", "base_s": 0.0, "n_frames": (n_frames or {}).get(v, T)} for v in views}
    return {"dir": Path("/tmp/x"), "context": ctx, "sources": src, "state": np.asarray(state, dtype=np.float64),
            "action": None, "times": times, "kmap": kmap or {}}


def test_canonical_states_ee_pose():
    T = 5
    s = np.zeros((T, 14))
    s[:, 0] = np.linspace(0, 0.04, T)            # left x moves 1 cm per frame
    s[:, 5] = np.linspace(0, 0.4, T)             # left yaw turns 0.1 rad per frame
    s[:, 6] = 0.5
    s[:, 13] = 0.2
    ep = _ep(s)
    cs = cq.canonical_states(ep)
    assert cs["valid"][:, :6].all() and cs["valid"][:, 6].all() and cs["valid"][:, 7:13].all()
    assert np.allclose(cs["states"][:, 5], np.linspace(0, 0.4, T))   # pure yaw -> rotvec z
    ts = np.arange(T) * int(1e9 / 30)
    local, glob, valid, _ = up.delta_actions(cs["states"], cs["valid"], ts)
    assert np.allclose(glob[:, 0], 0.01, atol=1e-6)
    assert np.allclose(glob[:, 5], 0.1, atol=1e-6)


def test_canonical_states_joints_only_gripper():
    s = np.zeros((4, 14))
    s[:, 6] = [0.1, 0.2, 0.3, 0.4]
    cs = cq.canonical_states(_ep(s, kind="joints", profile="teleop_arms", views=("exo", "left", "right")))
    assert not cs["valid"][:, :6].any() and cs["valid"][:, 6].all()


def test_nonfinite_reported():
    s = np.zeros((4, 14))
    s[2, 3] = np.nan
    cs = cq.canonical_states(_ep(s))
    assert cs["nonfinite"] and cs["nonfinite"][0]["actor"] == "left" and cs["nonfinite"][0]["first_row"] == 2


def test_gripper_unit():
    verified = "0 = jaws shut, 1 = fully open (checked against the wrist frames)"
    assert cq.gripper_unit({"gripper_value": verified}) == "normalized_open_fraction"
    assert cq.gripper_unit({"gripper_value": "the measured opening width in metres, about 0 = jaws shut"}) == "metres"
    assert cq.gripper_unit({"gripper_value": "about 0 = jaws shut and about 100 = fully open"}) == "source_units"
    assert cq.gripper_unit({}) == "unknown"
    assert cq.gripper_unit({"gripper_unit": {"unit": "normalized_open_fraction"}}) == "normalized_open_fraction"


def test_to_anchor_kmap():
    """Camera paired by kmap: interval sums over its own frames; a repeated camera frame is NaN, not 0."""
    own = np.array([1.0, 2.0, 3.0, 4.0, 5.0])          # 6 own frames -> 5 own pairs
    km = np.array([0, 1, 1, 3, 5])                      # anchor frames 0..4
    ep = _ep(np.zeros((5, 14)), kmap={"right": km}, n_frames={"right": 6})
    out = cq.to_anchor(ep, "right", own, 5)
    assert np.isclose(out[0], 1.0)                      # own 0 -> 1
    assert np.isnan(out[1])                             # own 1 -> 1: no new camera frame
    assert np.isclose(out[2], 2.0 + 3.0)                # own 1 -> 3
    assert np.isclose(out[3], 4.0 + 5.0)                # own 3 -> 5
    own[3] = np.nan                                     # a pair the decoder could not form
    out = cq.to_anchor(ep, "right", own, 5)
    assert np.isnan(out[3]) and np.isclose(out[2], 5.0)


def test_to_anchor_unpaired():
    ep = _ep(np.zeros((4, 14)))
    assert np.allclose(cq.to_anchor(ep, "left", np.array([1.0, 2.0, 3.0]), 4), [1, 2, 3])
    assert cq.to_anchor(ep, "left", np.array([1.0, 2.0]), 4) is None          # frame counts differ, no kmap


def test_largest_action_skips_unobserved_interval():
    """Vendored modification 4: a NaN camera interval (no new frame) must not count as 'no visual motion'."""
    T = 60
    acts = np.zeros((T - 1, 14))
    acts[:, 0] = 0.001
    acts[30, 0] = 0.08                                    # one large step
    valid = np.ones((T - 1, 14), dtype=bool)
    pix = np.full(T - 1, 5.0)
    pix[30] = np.nan
    checks, ev = up._largest_action_video_checks(acts, valid, {"right": pix}, up.FilterPolicy())
    assert checks["left_translation"]["eligible"] and not ev
    pix[30] = 0.1                                         # observed and still: unsupported
    checks, ev = up._largest_action_video_checks(acts, valid, {"right": pix}, up.FilterPolicy())
    assert ev and checks["left_translation"]["status"] == "unsupported"


def test_repeat_counts_only_inside_motion():
    """A repeated picture counts only between two clearly moving steps; a still scene's identical frames never do."""
    T = 200
    s = np.zeros((T, 14))
    s[:, 6] = 0.5
    s[:, 13] = 0.5
    ep = _ep(s, views=("left",))
    still = np.zeros(T - 1, dtype=np.float32)                   # identical frames throughout: a still scene
    still[::30] = 2.2                                            # keyframe steps
    moving = np.full(T - 1, 6.0, dtype=np.float32)
    moving[1::2] = 0.02                                          # every other frame repeats while moving
    for pair, expect in ((still, False), (moving, True)):
        cams = {"left": {"n": T, "decoded": T, "error": None, "fps": 30.0,
                         "means": np.full(T, 100.0, np.float32), "stds": np.full(T, 20.0, np.float32),
                         "pair": pair, "pchange": pair}}
        a = cq.assess({"ep": ep, "T": T, "cams": cams})
        assert (a["checks"]["video_duplicate_frames"]["status"] == "fired") == expect
        assert a["checks"]["video_frozen_run"]["status"] != "fired"   # nothing recorded moves, so no frozen claim


def test_a_frozen_picture_is_reported_whole_across_a_lone_keyframe_step():
    """A frozen camera that a recorder re-encodes changes a little at each keyframe; that one step does not end the
    frozen run, so the run is reported from where it starts. Two short still runs never add up to a new firing."""
    T = 300
    s = np.zeros((T, 14))
    s[:, 0] = np.linspace(0.0, 0.6, T)                          # the left gripper moves 60 cm throughout
    s[:, 6] = 0.5
    s[:, 13] = 0.5
    ep = _ep(s, views=("left",))

    def run(pair):
        cams = {"left": {"n": T, "decoded": T, "error": None, "fps": 30.0,
                         "means": np.full(T, 100.0, np.float32), "stds": np.full(T, 20.0, np.float32),
                         "pair": pair, "pchange": pair}}
        return cq.assess({"ep": ep, "T": T, "cams": cams})["checks"]["video_frozen_run"]

    pair = np.full(T - 1, 6.0, dtype=np.float32)
    pair[60:180] = 0.0                                           # frozen from 2.0 s to 6.0 s
    pair[120] = 0.4                                              # one keyframe step inside the freeze
    c = run(pair)
    assert c["status"] == "fired"
    ev = c["events"][0]
    assert abs(ev["t_s"] - 2.0) < 0.05 and "for 4.0 s from 2.0 s" in ev["evidence"]
    assert "apart from 1 single keyframe step under 3" in ev["evidence"]
    short = np.full(T - 1, 6.0, dtype=np.float32)
    short[60:84] = 0.0                                           # 0.8 s still, one keyframe step, 0.8 s still
    short[84] = 0.4
    short[85:109] = 0.0
    assert run(short)["status"] != "fired"


def test_disposition_covers_every_check():
    for c in cq.CHECKS:
        d = cq.DISPOSITION[c]
        assert d["disposition"] in ("flag", "note", "excluded"), c
        assert d["reason"], c
        if d["disposition"] == "flag":
            assert d["title"], c



def test_a_note_reads_as_sentences_and_an_older_record_is_reworded():
    """A note is its check's lead, its evidence and why it is a note, each a sentence; a stored note in the earlier
    wording ("... holds still Not counted as an issue: ...") is reworded on the next build, and rewording is stable."""
    ev = "camera left changes by at least 8 grey levels on 22 frame steps while every gripper holds still"
    n = cq.note_record("visual_change_unexplained_by_action", ev, "teleop_arms")
    assert n["evidence"] == "Camera left changes by at least 8 grey levels on 22 frame steps while every gripper holds still."
    assert n["text"].endswith(n["evidence"] + " " + cq.NOT_COUNTED["visual_change_unexplained_by_action"])
    assert "Not counted" not in n["text"] and ":" not in n["text"]
    lead = cq._lead("visual_change_unexplained_by_action")
    old = {"notes": [{"check": "visual_change_unexplained_by_action", "text": lead + " " + ev
                      + " Not counted as an issue: slow handheld drift reads as holding still, so it fires on sound recordings."}],
           "checks": [{"check": "visual_change_unexplained_by_action", "status": "fired", "shown_as": "note",
                       "why": "Not counted as an issue: slow handheld drift reads as holding still."}],
           "metrics": {"episode": {"rig": "teleop_arms"}}}
    new = cq.refresh_notes(old)
    assert new["notes"][0] == n and new["checks"][0]["why"] == cq.NOT_COUNTED["visual_change_unexplained_by_action"]
    assert cq.refresh_notes(new) == new
    other = cq.note_why("nonfinite_signal", "handheld_gripper")
    assert "UMI datasets" in other
    assert "rig" not in other and "Not counted" not in other


def test_a_stored_speed_note_is_reworded_with_the_right_plural():
    """The pose-speed evidence stored as "(rule: over 2 m/s or 8 rad/s in any one interval; 1 intervals)" is
    reworded on the next build, with "interval" singular for one."""
    old = {"notes": [{"check": "gross_umi_speed", "evidence": "The right pose moves at up to 5.02 m/s and 2.0 rad/s "
                      "between two frames (rule: over 2 m/s or 8 rad/s in any one interval; 1 intervals).", "text": "x"}],
           "metrics": {"episode": {"rig": "handheld_gripper"}}}
    ev = cq.refresh_notes(old)["notes"][0]["evidence"]
    assert ev == ("The right pose moves at up to 5.02 m/s and 2.0 rad/s between two frames, over the limit of 2 m/s or "
                  "8 rad/s in 1 interval.")
    assert "3 intervals." in cq.refresh_notes({**old, "notes": [{**old["notes"][0], "evidence": old["notes"][0][
        "evidence"].replace("1 intervals", "3 intervals")}]})["notes"][0]["evidence"]


VERIFIED = "0 = jaws shut, 1 = fully open (checked against the wrist frames)"


def _gripper_episode(T: int = 60):
    """Two handheld grippers whose openings move, timed at 30 fps, with no camera decoded."""
    s = np.zeros((T, 14))
    s[:, 0] = np.linspace(0, 0.3, T)
    s[:, 6] = 0.5 + 0.4 * np.sin(np.linspace(0, 6, T))
    s[:, 7] = np.linspace(0, 0.2, T)
    s[:, 13] = 0.5 + 0.4 * np.cos(np.linspace(0, 6, T))
    return s


def test_one_nan_state_row_never_turns_off_the_state_checks():
    """A state with one row that has no reading: that row is left out of the checks and reported, and every other row
    is checked as usual."""
    s = _gripper_episode()
    s[10, 2] = np.nan
    ep = _ep(s, gripper_value=VERIFIED)
    cs = cq.canonical_states(ep)
    assert not cs["valid"][10, :7].any() and cs["valid"][[9, 11], :7].all() and cs["valid"][10, 7:].all()
    assert cs["nonfinite"] == [{"actor": "left", "rows": 1, "first_row": 10}]
    R = cq.assess({"ep": ep, "T": len(s), "cams": {}})["checks"]
    assert R["nonfinite_signal"]["status"] == "fired"
    for c in ("gripper_never_acts", "normalized_gripper_out_of_range", "gripper_sensor_bug", "jump_return_event",
              "over_95_percent_static"):
        assert R[c]["status"] in ("fired", "clear"), (c, R[c])


def test_a_check_that_crashes_is_recorded_as_errored_and_the_rest_are_kept(monkeypatch):
    def boom(*a, **kw):
        raise ValueError("boom")
    monkeypatch.setattr(cq.up, "gripper_sensor_bug_checks", boom)
    s = _gripper_episode()
    a = cq.assess({"ep": _ep(s, gripper_value=VERIFIED), "T": len(s), "cams": {}})
    R = a["checks"]
    assert R["gripper_sensor_bug"]["status"] == "errored" and "ValueError: boom" in R["gripper_sensor_bug"]["error"]
    assert R["gripper_never_acts"]["status"] in ("fired", "clear")
    assert R["jump_return_event"]["status"] in ("fired", "clear") and R["episode_too_short"]["status"] == "fired"
    rec = cq.format_result(a)
    row = next(r for r in rec["checks"] if r["check"] == "gripper_sensor_bug")
    assert row["status"] == "errored" and "boom" in row["why"]
    assert not [f for f in rec["flags"] if f["check"] == "gripper_sensor_bug"]


def test_an_episode_whose_checks_cannot_run_records_every_check_as_errored(monkeypatch):
    def boom(d):
        raise RuntimeError("the state file does not read")
    monkeypatch.setattr(cq, "extract", boom)
    rec = cq.run_episode("/nowhere/episode_000000")
    assert rec["checks"] and all(r["status"] == "errored" for r in rec["checks"]) and not rec["flags"]
    assert "the state file does not read" in rec["checks"][0]["why"]
