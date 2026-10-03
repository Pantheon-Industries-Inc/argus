"""Dataset step claims on each rig, without inventing an episode instruction."""
from copy import deepcopy

import numpy as np
import pytest

from label import episode as me
from prepare import formats as f


SUBTASKS = [
    {"t0": "0.5 s", "t1": "1.5", "label": "reach"},
    {"t0": None, "t1": "2", "label": "untimed claim"},
    {"t0": "2", "t1": "2.0", "label": "release", "ok": False},
    {"t0": True, "t1": "NaN", "label": "invalid time claim"},
]


def prompt(rig, instruction=None, subtasks=None):
    ctx = {"dataset": "fixture", "fps": 30, "profile": rig, "state_kind": "none",
           "cameras": {"exo": {"width": 640, "height": 480}}}
    if instruction is not None:
        ctx["instruction"] = instruction
    if subtasks is not None:
        ctx["annotation_subtasks"] = deepcopy(subtasks)
        ctx["annotation_note"] = "Original step declarations."
    ep = {"context": ctx, "state": np.zeros((90, 0)), "sources": {"exo": {}}}
    pl = {"n": 90, "ks": [0, 45, 89], "spans": [], "state_usable": True, "contact": []}
    before = deepcopy(ep["context"]), ep["state"].tobytes(), deepcopy(pl)
    texts = me.build_prompt(ep, pl, cell_w=448, cell_h=252)
    assert before == (ep["context"], ep["state"].tobytes(), pl)
    return texts


@pytest.mark.parametrize("rig", ["ego_head", "teleop_arms", "handheld_gripper"])
@pytest.mark.parametrize("instruction", [None, "Place the cup."])
def test_each_rig_shows_typed_step_claims_without_inventing_a_goal(rig, instruction):
    fixed, text = prompt(rig, instruction, SUBTASKS)
    assert "0.5-1.5s  reach" in text
    assert "no time  untimed claim" in text
    assert "2.0s  release  (marked unsuccessful)" in text
    assert "no time  invalid time claim" in text
    assert "Original step declarations." in text
    assert "claims" in text
    assert fixed == prompt(rig, instruction)[0]
    if rig != "ego_head":
        assert "see ABOUT THE DATASET'S ANNOTATION above" not in text
        assert "goal:" not in text
        assert text.count('"Place the cup."') == (1 if instruction else 0)
        assert ('"goal_alignment": {' in fixed) == bool(instruction)


@pytest.mark.parametrize("rig", ["teleop_arms", "handheld_gripper"])
def test_absent_robot_annotations_add_no_heading(rig):
    assert prompt(rig) == prompt(rig, subtasks=[])
    assert prompt(rig, "Place the cup.") == prompt(rig, "Place the cup.", [])


def test_existing_ego_annotation_bytes_stay_exact():
    ctx = {"instruction": "Place the cup.", "annotation_subtasks": SUBTASKS,
           "annotation_note": "Original step declarations."}
    assert me.ego_annotation_block(ctx) == (
        "\nTHE DATASET'S ANNOTATION FOR THIS EPISODE (claims to check, see ABOUT THE DATASET'S ANNOTATION above):\n"
        '  goal: "Place the cup."\n'
        "  subtasks, with the times the dataset gives:\n"
        "  0.5-1.5s  reach\n"
        "  no time  untimed claim\n"
        "  2.0s  release  (marked unsuccessful)\n"
        "  no time  invalid time claim\n"
        "  about these annotations: Original step declarations.\n")


@pytest.mark.parametrize("reverse", [False, True])
def test_mcap_episode_task_wins_over_language_steps_and_health(reverse):
    base = 1_000_000_000
    topics = [
        ("/language_instruction/steps", [(base, "Reach"), (base + 500_000_000, "Reach"),
                                          (base + 2_000_000_000, "Release")]),
        ("/task", [(base, 'title: "Place the cup."\n')]),
        ("/task/health", [(base + n, "healthy") for n in range(f.TEXT_MSGS_MAX + 1)]),
    ]
    texts = dict(reversed(topics) if reverse else topics)
    counts = {t: len(xs) for t, xs in topics}
    instruction, notes = f.mcap_task_texts(texts, counts, base)
    assert instruction == "Place the cup."
    assert notes["/language_instruction/steps"] == ["0.0 s: Reach", "0.5 s: Reach", "2.0 s: Release"]
    topic, steps = f.mcap_step_subtasks(texts, base, 3.0)
    assert topic == "/language_instruction/steps"
    assert steps == [{"t0": 0.0, "t1": 0.5, "label": "Reach"},
                     {"t0": 0.5, "t1": 2.0, "label": "Reach"},
                     {"t0": 2.0, "t1": 3.0, "label": "Release"}]
    assert f.mcap_step_subtasks({"/task/health": texts["/task/health"]}, base, 3.0) == (None, [])
    assert f.mcap_task_texts({topic: texts[topic]}, counts, base)[0] is None
