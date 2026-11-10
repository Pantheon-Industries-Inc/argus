"""DiverseUMI-10k ships its labels with the summary as recording_summary and the review as recording_review. The board
read only task_summary and performance_review, so every card of a board built from those labels came out empty
(argus#239). The release's names are read when the model's own are absent, and the model's own still win."""
from __future__ import annotations

from board import to_board
from tests.test_board import _output


def _shipped(**extra) -> dict:
    out = _output("episode_000023")
    labels = {k: v for k, v in out["labels"].items() if k != "task_summary"}
    labels.update(recording_summary="The right gripper lines the keycaps up on the lid.",
                  recording_review="Steady work; a few empty pickups of one keycap.", **extra)
    return {**out, "labels": labels}


def test_a_released_label_fills_the_card_summary_and_review():
    d = to_board.convert(_shipped(), "DiverseUMI-10k")
    assert d["episode_prompt"] == "The right gripper lines the keycaps up on the lid."
    assert d["performance_review"] == "Steady work; a few empty pickups of one keycap."


def test_the_models_own_fields_win_over_the_released_names():
    d = to_board.convert(_shipped(task_summary="own summary", performance_review="own review"), "x")
    assert d["episode_prompt"] == "own summary" and d["performance_review"] == "own review"


def test_a_label_with_neither_name_stays_empty():
    out = _output("episode_000001")
    out["labels"].pop("task_summary")
    d = to_board.convert(out, "x")
    assert d["episode_prompt"] == "" and d["performance_review"] == ""
