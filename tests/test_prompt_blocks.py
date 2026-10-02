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
