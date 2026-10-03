"""The episode prompt as a base (the task and the rig's camera geometry) plus blocks, each present only when the
episode holds its data (label/episode.py BLOCKS). Today's episode prompts are pinned as text in tests/fixtures/prompts,
so moving text between blocks cannot change a word, and the prompts of episodes with no optional data (NO_DATA) are
never rewritten: an episode without a kind of data never hears of it."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from label import episode as me
from label import state as ms

GOLDEN = Path(__file__).parent / "fixtures" / "prompts"


def _pl(n, spans=(), contact=(), usable=True, every=1.5, fps=30):
    ks = ms.sample_frames(n, list(spans), fps=fps, moving_every_s=every, still_every_s=every)
    return {"n": n, "ks": ks, "spans": list(spans), "state_usable": usable, "contact": list(contact)}


def _joints(n, arms=2):
    """Joint state that sweeps smoothly, each gripper closed from 5 s to 12 s, everything still from 10 s to 15 s."""
    t = np.arange(n) / 30.0
    s = np.stack([0.3 * np.sin(0.4 * t + j) for j in range(7 * arms)], axis=1)
    for g in range(arms):
        s[:, 7 * g + 6] = ((t > 5) & (t < 12)).astype(float)
    s[300:450] = s[300]
    return s


def case_teleop_joints():
    n = 900
    ep = {"context": {"dataset": "allenai/MolmoAct2-BimanualYAM-Dataset", "robot_type": "bi_yam_follower", "fps": 30,
                      "task_label": ["Spell out Ai2"], "instruction": "Spell AI2.", "profile": "teleop_arms",
                      "state_kind": "joints", "cameras": {"exo": {"width": 640, "height": 360}}},
          "state": _joints(n), "sources": {"exo": {}, "left": {}, "right": {}}}
    return ep, _pl(n, [(300, 449)])


def case_handheld_pose():
    n = 600
    t = np.arange(n) / 30.0
    s = np.stack([0.1 * np.sin(t), 0.05 * np.cos(t), 0.2 + 0 * t, 0.1 * np.sin(0.5 * t), 0 * t, 0.2 * t,
                  (t > 8).astype(float)], axis=1)
    ep = {"context": {"dataset": "you/handheld", "fps": 30, "profile": "handheld_gripper", "state_kind": "ee_pose",
                      "cameras": {"right": {"name": "gripper", "width": 640, "height": 480}}},
          "state": s, "sources": {"right": {}}}
    return ep, _pl(n, every=1.0)


def case_teleop_video_only():
    n = 450
    ep = {"context": {"dataset": "you/rig", "fps": 30, "profile": "teleop_arms", "state_kind": "none",
                      "cameras": {"exo": {"width": 640, "height": 480}, "left": {"width": 640, "height": 480}}},
          "state": np.zeros((n, 0)), "sources": {"exo": {}, "left": {}}}
    return ep, _pl(n)


def case_ego_plain():
    n = 300
    ep = {"context": {"dataset": "you/head", "fps": 30, "profile": "ego_head", "state_kind": "none",
                      "cameras": {"exo": {"width": 1280, "height": 720}}},
          "state": np.zeros((n, 0)), "sources": {"exo": {}}}
    return ep, _pl(n, every=0.5)


def case_teleop_unaligned():
    ep, _ = case_teleop_joints()
    return ep, _pl(900, usable=False)


def case_teleop_video_only_two_wrists():
    """Video only with both mounted cameras: its camera paragraph loses one clause in Task 3 (controller decision 3)."""
    ep, pl = case_teleop_video_only()
    ep["sources"]["right"] = {}
    ep["context"]["cameras"]["right"] = {"width": 640, "height": 480}
    return ep, pl


def _glove(n, on=(180, 240)):
    """A 4 x 4 glove pressure map with a real sensor's noise that rests at its untouched reading and falls where it is
    pressed, frames on[0] to on[1]: touch by its name and by its numbers (label/signals.py is_touch)."""
    a = 3072.0 + np.random.default_rng(0).normal(0, 2, (n, 16))
    a[on[0]:on[1], 5] -= 1000.0
    return a


# the glove's upload scale as prepare writes it (the depth of a press across the upload); without it, noise hides its
# rest
GLOVE_META = {"shape": [4, 4], "swing": 1000.0}


CONTACT = {"id": "c1", "hand": "left", "signals": ["left_glove_pressure"], "start_s": 6.0, "end_s": 8.0,
           "peak_s": 7.0, "regions": {}, "peak_strength": 2.0}


def case_teleop_everything():
    """Every block at once: a collection note, gripper detail views, depth, three signals (one of them the glove whose
    touch times the contact), a contact, uploader notes."""
    ep, pl = case_teleop_joints()
    t = np.arange(900) / 30.0
    ep["context"]["collection_note"] = "consecutive 3-minute clips of a shift."
    ep["context"]["uploader_annotation"] = '{"operator": "A"}\n'
    ep["signals"] = {"base.odom": np.stack([0.01 * t, 0 * t, 0.002 * t], axis=1),
                     "teleop.intervention": ((t > 20) & (t < 22)).astype(float)[:, None],
                     "left_glove_pressure": _glove(900)}
    ep["signal_meta"] = {"base.odom": {"names": ["x", "y", "yaw"]}, "teleop.intervention": {},
                         "left_glove_pressure": dict(GLOVE_META)}
    ep["depth"] = {"exo": {}}
    ep["contacts"] = ep["contacts_shown"] = [dict(CONTACT)]
    pl["contact"] = [pl["ks"][4]]
    return ep, pl


def case_ego_annotated_tracks():
    ep, pl = case_ego_plain()
    t = np.arange(300) / 30.0
    ep["context"].update(instruction="chop the carrot",
                         annotation_subtasks=[{"t0": 0.0, "t1": 4.0, "label": "pick up knife"},
                                              {"t0": 4.0, "t1": 10.0, "label": "chop", "ok": False}])
    ep["signals"] = {"right_hand_landmarks": np.stack([np.sin(t + j) for j in range(63)], axis=1)}
    ep["signal_meta"] = {"right_hand_landmarks": {"shape": [21, 3], "rate_hz": 15.0}}
    return ep, pl


CASES = {"teleop_joints": case_teleop_joints, "handheld_pose": case_handheld_pose,
         "teleop_video_only": case_teleop_video_only, "ego_plain": case_ego_plain,
         "teleop_unaligned": case_teleop_unaligned, "teleop_everything": case_teleop_everything,
         "ego_annotated_tracks": case_ego_annotated_tracks,
         "teleop_video_only_two_wrists": case_teleop_video_only_two_wrists}
# episodes with no optional data: their episode prompts never change (an episode without a kind of data never hears of
# it). teleop_video_only_two_wrists has none either, and changes once, in Task 3, by one false clause.
NO_DATA = ("teleop_joints", "handheld_pose", "teleop_video_only", "ego_plain", "teleop_unaligned")


def _episode(name):
    ep, pl = CASES[name]()
    return me.build_prompt(ep, pl, cell_w=448, cell_h=252)[1]


@pytest.mark.parametrize("name", sorted(CASES))
def test_episode_prompts_match_their_pinned_text(name):
    """The episode part of each case's prompt is its golden file. ARGUS_WRITE_PROMPTS=<name> (or all) rewrites one
    from the current code, for a deliberate change to a case with data; a NO_DATA case is never rewritten
    (=all rewrites only the cases with data)."""
    text = _episode(name)
    path = GOLDEN / f"{name}.txt"
    if os.environ.get("ARGUS_WRITE_PROMPTS") in (name, "all") and name not in NO_DATA:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    assert path.read_text() == text, f"{name}: the episode prompt changed"


def _add(**ctx):
    def f(ep, pl):
        ep["context"].update(ctx)
    return f


def _add_contact_views(ep, pl):
    pl["contact"] = [pl["ks"][3]]


def _add_late_camera(ep, pl):
    """The left camera starts 3 s after the scene camera (paired by real time), so it has no frame at the first
    instants."""
    n = len(ep["state"])
    ep["times"] = {"exo": np.arange(n) / 30.0, "left": np.arange(90, n) / 30.0, "right": np.arange(n) / 30.0}
    ep["kmap"] = {"left": np.clip(np.arange(n) - 90, 0, None), "right": np.arange(n)}


def _add_depth(ep, pl):
    ep["depth"] = {"exo": {}}


def _add_arm_state(ep, pl):
    ep["context"]["state_kind"] = "joints"
    ep["state"] = _joints(len(ep["state"]), arms=1)


def _misalign(ep, pl):
    pl["state_usable"] = False


def _drop_state(ep, pl):
    ep["context"]["state_kind"] = "none"
    ep["state"] = np.zeros((len(ep["state"]), 0))


def _add_signals(ep, pl):
    t = np.arange(len(ep["state"])) / 30.0
    ep["signals"] = {"base.odom": np.stack([0.01 * t, 0 * t, 0.002 * t], axis=1)}
    ep["signal_meta"] = {"base.odom": {"names": ["x", "y", "yaw"]}}


def _add_contact(ep, pl):
    """The left glove's pressure (touch by name and numbers) and the contact it times."""
    n = len(ep["state"])
    ep.setdefault("signals", {})["left_glove_pressure"] = _glove(n)
    ep.setdefault("signal_meta", {})["left_glove_pressure"] = dict(GLOVE_META)
    ep["contacts"] = ep["contacts_shown"] = [dict(CONTACT)]


# (block, the case without its data, what adds the data, text only that block says)
BLOCK_CASES = [
    ("collection_note", "teleop_joints", _add(collection_note="consecutive 3-minute clips of a shift."),
     ("How the dataset cuts its recordings",)),
    ("contact_views", "teleop_joints", _add_contact_views, ("So are the instants just after",)),
    ("coverage", "teleop_joints", _add_late_camera, ("has frames only from",)),
    ("depth", "teleop_joints", _add_depth, ("DEPTH:",)),
    ("state", "teleop_video_only", _add_arm_state, ("RECORDED STILL SPANS", "RECORDED MOTION")),
    ("state_unaligned", "teleop_joints", _misalign, ("RECORDED STATE: not given",)),
    ("no_state", "teleop_joints", _drop_state, ("RECORDED STATE: none",)),
    ("signals", "teleop_joints", _add_signals, ("OTHER RECORDED SIGNALS", "base.odom")),
    ("contacts", "teleop_joints", _add_contact, ("CONTACTS:",)),
    ("uploader_notes", "teleop_joints", _add(uploader_annotation='{"operator": "A"}\n'),
     ("THE UPLOADER'S OWN NOTES",)),
]
# blocks whose presence switches the shared instructions to another variant
CHANGES_FIXED = ("state", "no_state")


@pytest.mark.parametrize("block,base,add,markers", BLOCK_CASES, ids=[c[0] for c in BLOCK_CASES])
def test_a_block_is_in_the_prompt_and_the_schema_only_when_its_data_is(block, base, add, markers):
    ep, pl = CASES[base]()
    fields = next(b.schema_fields for b in me.BLOCKS if b.name == block)
    fixed0, ep0 = me.build_prompt(ep, pl, cell_w=448, cell_h=252)
    assert block not in [b.name for b in me.present_blocks(ep, pl)]
    assert not any(m in fixed0 + ep0 for m in markers)
    assert not set(fields) & set(me.requested_schema(ep, pl))
    assert not any(f'"{f}"' in fixed0 + ep0 for f in fields)
    add(ep, pl)
    fixed1, ep1 = me.build_prompt(ep, pl, cell_w=448, cell_h=252)
    assert block in [b.name for b in me.present_blocks(ep, pl)]
    assert all(m in ep1 for m in markers)
    assert set(fields) <= set(me.requested_schema(ep, pl))
    assert all(f'"{f}"' in ep1 for f in fields)
    if block not in CHANGES_FIXED:
        assert fixed1 == fixed0                      # the cached shared instructions stay byte for byte the same


def test_every_block_is_tested_and_presence_never_reads_the_rig_or_dataset_name():
    import inspect
    assert {c[0] for c in BLOCK_CASES} == {b.name for b in me.BLOCKS}
    for b in me.BLOCKS:
        assert b.present.__name__ != "<lambda>", b.name   # a named test, so its source can be read here
        src = inspect.getsource(b.present)
        assert "rig(" not in src and "dataset" not in src, b.name
    assert {b.slot for b in me.BLOCKS} <= set(me.PROMPT_SLOTS)


def test_contacts_ask_for_their_fields_and_imply_the_contact_check():
    ep, pl = CASES["teleop_joints"]()
    assert me.requested_schema(ep, pl) == () and "contact_checks" not in me.implied_checks(ep, pl)
    _add_contact(ep, pl)
    assert me.requested_schema(ep, pl) == ("contacts", "contacts_missing")
    assert "contact_checks" in me.implied_checks(ep, pl)


def test_a_contact_timed_only_by_a_signal_that_does_not_measure_touch_is_not_shown():
    """A context.json prepared before is_touch can hold a contact found from an intervention flag, which rests and
    rises like a pad but whose name says nothing of touch: it is neither told to the model nor asked about."""
    ep, pl = CASES["teleop_joints"]()
    t = np.arange(900) / 30.0
    ep["signals"] = {"teleop.intervention": ((t > 6) & (t < 8)).astype(float)[:, None]}
    ep["signal_meta"] = {"teleop.intervention": {}}
    ep["contacts"] = ep["contacts_shown"] = [dict(CONTACT, signals=["teleop.intervention"])]
    episode = me.build_prompt(ep, pl, cell_w=448, cell_h=252)[1]
    assert "contacts" not in [b.name for b in me.present_blocks(ep, pl)] and me.requested_schema(ep, pl) == ()
    assert "CONTACTS:" not in episode and '"contacts"' not in episode
    assert me.touch_contacts(ep, ep["contacts"], pl["n"]) == []
    _add_contact(ep, pl)                             # a contact the glove's pressure times is shown
    assert [c["id"] for c in me.touch_contacts(ep, ep["contacts"], pl["n"])] == ["c1"]
    assert "contacts" in [b.name for b in me.present_blocks(ep, pl)]


def test_a_parts_context_carries_the_whole_recordings_touch_verdict_and_presence_leaves_the_episode_as_it_was():
    """A part of a long recording is told by its context whether each signal is touch (label/pieces.py write_pieces
    judges it on the whole recording), because a part inside a long press has no rest of its own to judge from."""
    ep, pl = CASES["teleop_joints"]()
    _add_contact(ep, pl)
    ep["signal_meta"]["left_glove_pressure"]["touch"] = False
    assert not me._has_contacts(ep, pl) and "CONTACTS:" not in me.build_prompt(ep, pl, cell_w=448, cell_h=252)[1]
    # pressed all through, so its own numbers show no rest; the recording's verdict says touch
    ep["signals"]["left_glove_pressure"] = 2072.0 + np.random.default_rng(1).normal(0, 2, (900, 16))
    ep["signal_meta"]["left_glove_pressure"]["touch"] = True
    assert me._has_contacts(ep, pl) and "CONTACTS:" in me.build_prompt(ep, pl, cell_w=448, cell_h=252)[1]
    # the readout agrees: a touch signal's timing is given only as the contacts, never per instant
    assert "Each signal that changes" not in me._signals_table(ep, pl)
    before = set(ep)
    me.present_blocks(ep, pl)
    assert set(ep) == before                         # presence tests leave the episode as it was


def test_an_unaligned_state_no_longer_hides_the_signals():
    ep, pl = CASES["teleop_unaligned"]()
    _add_signals(ep, pl)
    episode = me.build_prompt(ep, pl, cell_w=448, cell_h=252)[1]
    assert "RECORDED STATE: not given" in episode and "OTHER RECORDED SIGNALS" in episode


def test_a_video_only_episode_is_never_told_of_a_recorded_motion():
    ep, pl = CASES["teleop_video_only"]()
    fixed, episode = me.build_prompt(ep, pl, cell_w=448, cell_h=252)
    assert "recorded motion" not in fixed + episode
    ep2, pl2 = CASES["teleop_joints"]()
    assert "the recorded motion" in me.build_prompt(ep2, pl2, cell_w=448, cell_h=252)[0]
    ep3, pl3 = CASES["teleop_video_only"]()
    _add_signals(ep3, pl3)                           # recorded numbers, even without arm state, keep the wording
    assert "the recorded motion" in me.build_prompt(ep3, pl3, cell_w=448, cell_h=252)[0]


def test_a_video_only_rig_with_two_wrist_cameras_is_not_told_its_views_follow_a_recorded_motion():
    ep, pl = CASES["teleop_video_only_two_wrists"]()
    fixed, episode = me.build_prompt(ep, pl, cell_w=448, cell_h=252)
    assert "is turned at that instant." in episode and "recorded motion" not in fixed + episode
    ep2, pl2 = CASES["teleop_joints"]()
    assert "and which recorded motion its view follows." in me.build_prompt(ep2, pl2, cell_w=448, cell_h=252)[1]


def test_sensor_data_the_reader_left_unread_never_shows_its_note_or_a_claim_that_none_exists():
    plain = "RECORDED STATE: none was read from this episode, so the video is all there is."
    ep, pl = CASES["ego_plain"]()
    ep["context"]["state_note"] = ("Labelled from the camera. The hand, body and camera tracks the file records "
                                   "(/hand/left, /hand/right) are not read yet.")
    fixed, episode = me.build_prompt(ep, pl, cell_w=256, cell_h=144)
    assert "no hand, head or device tracking" not in episode and plain in episode
    assert "Labelled from the camera" not in fixed + episode and "/hand/left" not in fixed + episode
    ep, pl = CASES["teleop_video_only"]()
    ep["context"]["source"] = {"unused_arrays": ["observations/qpos_raw (8192 values per sample)"]}
    fixed, episode = me.build_prompt(ep, pl, cell_w=448, cell_h=252)
    assert "records no robot or gripper state" not in episode and plain in episode
    assert "qpos_raw" not in fixed + episode         # the list itself goes to the board (Task 4), not the prompt


def test_the_note_of_a_teleop_file_with_motion_channels_never_puts_a_recorded_motion_into_a_video_only_prompt():
    ep, pl = CASES["teleop_video_only"]()
    ep["context"]["state_note"] = ("Labelled from the cameras. The checks on recorded motion read six joints and a "
                                   "gripper per arm, so they did not run on this file's motion channels "
                                   "(/left/joint_states, /right/joint_states).")
    fixed, episode = me.build_prompt(ep, pl, cell_w=448, cell_h=252)
    assert "recorded motion" not in fixed and "recorded motion" not in episode
    assert "joint_states" not in fixed + episode
    assert "RECORDED STATE: none was read from this episode, so the video is all there is." in episode


def _plan_ep(signals, n=900):
    return {"dir": None, "context": {"profile": "teleop_arms", "state_kind": "none", "fps": 30},
            "sources": {"exo": {"n_frames": n}}, "state": np.zeros((n, 0)), "action": None, "times": None,
            "kmap": {}, "signals": signals, "signal_meta": {k: {} for k in signals}}


def test_with_no_arm_state_the_instants_where_the_signals_fall_quiet_and_move_again_are_sent():
    """A base drives, stops from 10 s to 20 s and drives on: the instant it stops and the one it moves again are sampled
    as an arm's still span would give them, but no still span is claimed. A signal that only jitters chooses nothing."""
    from label import signals as sg
    n = 900
    t = np.arange(n) / 30.0
    x = np.where(t < 10, t, np.where(t < 20, 10.0, 10 + (t - 20)))
    pl = me.plan(_plan_ep({"base.odom": x[:, None]}))
    assert pl["spans"] == []
    (a, b), = pl["quiet_spans"]
    assert abs(a - 300) <= 12 and abs(b - 600) <= 12
    assert a in pl["ks"] and b + 1 in pl["ks"]
    jitter = np.random.default_rng(0).normal(0, 1, (n, 3))
    assert sg.movements(jitter).max() < sg.MOVING_MIN
    assert me.plan(_plan_ep({"imu": jitter}))["quiet_spans"] == []
    ep, _ = CASES["teleop_joints"]()
    ep.update(dir=None, action=None, times=None, kmap={})
    ep["sources"] = {v: {"n_frames": 900} for v in ep["sources"]}
    assert "quiet_spans" not in me.plan(ep)           # an arm state finds its own still spans


def test_a_signal_is_quiet_by_its_own_step_so_the_length_of_the_recording_does_not_matter():
    """A base that drives at a steady speed is never quiet, whether the recording is 30 s or 20 minutes; one that stops
    for 10 s inside a 6 minute recording still gives that stop; values that only flicker (a flag on 5 percent of
    frames, a pad count off by one) never block a quiet span; a dropped reading does not hide a base that moves."""
    from label import signals as sg
    for n in (900, 9000, 36000):
        assert sg.quiet_spans({"odom": (np.arange(n) / 30.0)[:, None]}, 90) == []
    n = 10800
    t = np.arange(n) / 30.0
    x = np.where(t < 100, t, np.where(t < 110, 100.0, 100 + (t - 110)))
    (a, b), = sg.quiet_spans({"odom": x}, 90)
    assert abs(a - 3000) <= 12 and abs(b - 3300) <= 12
    flag = (np.random.default_rng(1).random(n) < 0.05).astype(float)
    pad = np.where(np.random.default_rng(2).random(n) < 0.3, 1, 0)         # integer counts at rest, off by one
    assert sg.movements(flag)[0] < sg.MOVING_MIN and sg.movements(pad)[0] < sg.MOVING_MIN
    assert sg.quiet_spans({"odom": x, "flag": flag, "pad": pad.astype(np.int16)}, 90) == [(a, b)]
    drop = x.copy()
    drop[1000] = np.nan                                                    # one dropout while the base moves
    assert sg.quiet_spans({"odom": drop}, 90) == [(a, b)]
    assert sg.movements(np.zeros((0, 2))).tolist() == [0.0, 0.0]
    assert sg.movements(np.full((50, 1), np.nan)).tolist() == [0.0]
    late = x.copy()
    late[:100] = np.nan
    assert sg.quiet_spans({"odom": late}, 90) == [(a, b)]


def test_a_state_shorter_than_a_still_span_gives_none_whatever_its_width():
    assert ms.still_spans(np.zeros((10, 6))) == []


HUMANOID = ([f"{s}_arm_j{j}" for s in ("left", "right") for j in range(1, 8)]
            + [f"{s}_hand_{f}" for s in ("left", "right") for f in ("thumb_yaw", "thumb_pitch", "index", "middle",
                                                                     "ring", "pinky")])


def test_a_humanoid_state_and_a_bases_odometry_are_shown_value_by_value_under_their_own_names():
    ep, pl = CASES["teleop_video_only"]()
    t = np.arange(450) / 30.0
    ep["signals"] = {"observation.state": np.stack([0.3 * np.sin(0.4 * t + j) for j in range(26)], axis=1),
                     "observation.base.odom": np.stack([0.01 * t, 0.001 * t, 0.02 * t, 0.2 + 0.1 * np.sin(t),
                                                        0.1 * np.cos(t)], axis=1)}
    ep["signal_meta"] = {"observation.state": {"names": HUMANOID},
                         "observation.base.odom": {"names": ["x", "y", "yaw", "vx", "wz"]}}
    episode = me.build_prompt(ep, pl, cell_w=448, cell_h=252)[1]
    # the signal's own line already names and ranges every value since 942d399; kept here as a guard
    assert "observation.state (26 values (left_arm_j1, left_arm_j2," in episode and "right_hand_pinky))" in episode
    line = next(l for l in episode.splitlines() if l.startswith("  observation.state (26 values"))
    assert "values from" not in line and line.count(" to ") == 26
    # new: one row per value at each instant, and the state line names the joint readings
    assert "    observation.state left_arm_j1: " in episode and "    observation.base.odom vx: " in episode
    assert "total activity" not in episode
    assert ("RECORDED STATE: no arm state in the layout our checks read. The joint readings it records "
            "(observation.state) are given value by value under their own names among the other recorded signals "
            "below.") in episode


def test_a_signal_slower_than_the_frames_says_its_rate_and_one_at_the_frame_rate_does_not():
    ep, pl = CASES["ego_annotated_tracks"]()
    episode = me.build_prompt(ep, pl, cell_w=256, cell_h=144)[1]
    assert "right_hand_landmarks (21 x 3 values, recorded at 15 Hz)" in episode
    ep["signal_meta"]["right_hand_landmarks"]["rate_hz"] = 30.0
    assert "recorded at" not in me.build_prompt(ep, pl, cell_w=256, cell_h=144)[1]
