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


def _two_cameras(right_crashes: bool, nan_row: bool = False):
    """Two handheld grippers whose left one moves 30 cm while its camera shows the same picture throughout (a frozen
    left camera), and a right camera that changes; the right camera's record is broken (no stds) when it crashes."""
    T = 120
    s = np.zeros((T, 14))
    s[:, 0] = np.linspace(0, 0.3, T)
    s[:, 6] = 0.5
    s[:, 13] = 0.5
    if nan_row:
        s[60, 1] = np.nan

    def cam(level, crash=False):
        rng = np.random.default_rng(1)
        return {"n": T, "error": None, "decoded": T, "means": np.full(T, 120.0) + rng.normal(0, 1, T),
                "stds": None if crash else np.full(T, 40.0), "pair": np.full(T - 1, level),
                "pchange": np.full(T - 1, level), "fps": 30.0}
    return {"ep": _ep(s, gripper_value=VERIFIED), "T": T,
            "cams": {"left": cam(0.02), "right": cam(30.0, crash=right_crashes)}}


def test_a_camera_that_crashes_costs_only_its_own_evidence():
    """The right camera's record breaks the low contrast check: its error is recorded and it is left out. A video check
    that fired on the left camera stays fired and names the camera that was never checked; one that did not fire is
    errored with the camera and its error, never clear for a camera nobody checked. The checks that finished on the
    right camera before the crash stand as they came out. The checks that compare every camera with the motion are
    errored with the error."""
    a = cq.assess(_two_cameras(True))
    R = a["checks"]
    assert R["video_frozen_run"]["status"] == "fired" and R["video_frozen_run"]["events"][0]["camera"] == "left"
    assert "camera right" in R["video_frozen_run"]["why"] and "TypeError" in R["video_frozen_run"]["why"]
    for c in ("video_low_contrast", "video_duplicate_frames"):
        assert R[c]["status"] == "errored" and "camera right" in R[c]["why"] and "TypeError" in R[c]["why"], (c, R[c])
    for c in ("camera_state_alignment_mismatch", "video_decode_failure", "video_decode_frame_count_mismatch",
              "video_extreme_exposure"):
        assert R[c]["status"] == "clear", (c, R[c])
    assert R["missing_camera"]["status"] == "clear"
    rec = cq.format_result(a)
    row = next(r for r in rec["checks"] if r["check"] == "video_frozen_run")
    assert row["status"] == "fired" and "camera right" in row["why"]
    assert "TypeError" in a["cameras"]["right"]["error"] and "error" not in a["cameras"]["left"]
    for c in ("largest_action_not_in_video", "visual_change_unexplained_by_action", "pixel_action_corr_mismatch"):
        assert R[c]["status"] == "errored" and "right" in R[c]["why"] and "TypeError" in R[c]["why"], (c, R[c])
    clean = cq.assess(_two_cameras(False))["checks"]
    assert clean["video_frozen_run"]["status"] == "fired"
    assert clean["visual_change_unexplained_by_action"]["status"] in ("fired", "clear")


def test_a_camera_that_crashes_after_its_checks_finished_leaves_them_standing():
    """The right camera's record breaks only after every video check finished on it with nothing found (its motion
    change is missing): no video check says it was not run on that camera. The frozen left camera still fires with no
    such reason, and the checks that compare every camera with the motion are errored, as the camera is left out."""
    f = _two_cameras(False)
    f["cams"]["right"]["pchange"] = None
    a = cq.assess(f)
    R = a["checks"]
    assert "AttributeError" in a["cameras"]["right"]["error"]
    assert R["video_frozen_run"]["status"] == "fired" and not R["video_frozen_run"].get("why"), R["video_frozen_run"]
    for c in cq.VIDEO_CHECKS:
        assert R[c]["status"] in ("fired", "clear") and "camera right" not in str(R[c].get("why")), (c, R[c])
    for c in ("largest_action_not_in_video", "visual_change_unexplained_by_action", "pixel_action_corr_mismatch"):
        assert R[c]["status"] == "errored" and "right" in R[c]["why"], (c, R[c])


def test_a_note_names_the_camera_it_was_not_run_on_after_the_board_build():
    """The left camera's picture is nearly uniform (a note on every setup) and the right camera's checks crash: the
    note's row names the camera that was never checked, also after the board build words every note's reason again."""
    from board.build import capture_names
    f = _two_cameras(True)
    f["cams"]["left"]["stds"] = np.full(f["T"], 1.0)
    rec = cq.format_result(cq.assess(f))
    board = capture_names(rec)
    for r in (rec, board):
        row = next(x for x in r["checks"] if x["check"] == "video_low_contrast")
        assert row["status"] == "fired" and row["shown_as"] == "note", row
        assert "camera right" in row["why"] and "TypeError" in row["why"], row
    assert capture_names(board) == board


def test_a_camera_that_crashes_keeps_the_defects_already_found_on_it():
    """The right camera's file decodes 60 of its 120 frames with an error, then its record breaks a later check: the
    decode failure and the short frame count found before the crash are still reported with the decoder's error, and
    the camera keeps both errors."""
    f = _two_cameras(True)
    r = f["cams"]["right"]
    r["error"], r["decoded"] = "moov atom not found", 60
    r["means"][60:] = np.nan
    a = cq.assess(f)
    R = a["checks"]
    dec, cnt = R["video_decode_failure"], R["video_decode_frame_count_mismatch"]
    assert dec["status"] == "fired" and "moov atom not found" in dec["events"][0]["evidence"], dec
    assert cnt["status"] == "fired" and "60 of its 120" in cnt["events"][0]["evidence"], cnt
    # both ran to the end on both cameras, so neither says a camera went unchecked
    assert not dec["why"] and not cnt["why"]
    assert R["video_low_contrast"]["status"] == "errored" and "camera right" in R["video_low_contrast"]["why"]
    assert "TypeError" in a["cameras"]["right"]["error"]
    assert a["cameras"]["right"]["decode_error"] == "moov atom not found"


def test_a_crash_finding_each_actor_s_camera_costs_only_the_checks_that_need_it(monkeypatch):
    """Which camera each gripper is mounted on tells a frozen picture from a still scene and pairs each gripper's motion
    with its own camera: when it cannot be worked out, those checks are errored with the error and every other check
    of the episode still runs."""
    def boom(*a, **k):
        raise RuntimeError("the mounts do not read")
    monkeypatch.setattr(cq, "actor_views", boom)
    R = cq.assess(_two_cameras(False))["checks"]
    need = ("video_frozen_run",) + cq.CAMERA_MOTION_CHECKS
    for c in need:
        assert R[c]["status"] == "errored" and "the mounts do not read" in R[c]["why"], (c, R[c])
    assert not [c for c, r in R.items() if c not in need and r["status"] == "errored"]


def test_a_missing_state_row_never_hides_a_frozen_camera():
    """The moving gripper's state has one row with no reading inside the frozen stretch: its motion is summed over the
    rows that have readings, so the frozen camera is still reported."""
    R = cq.assess(_two_cameras(False, nan_row=True))["checks"]
    assert R["video_frozen_run"]["status"] == "fired"
