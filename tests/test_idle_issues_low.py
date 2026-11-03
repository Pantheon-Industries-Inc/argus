"""An idle stretch is never more than a low data issue (label/harness.py idle_issues_low): idleness is trimmed from the
dataset, it is not a defect of the footage. The parser sets every data issue with an idle category to low and keeps
the model's own rating in severity_as_labelled; a recording fault, a frozen or black camera, keeps its severity, and
so does an operator mistake. The stitch of a recording labelled in parts applies the same rule to parts labelled
before it, and the prompt says it where it describes severity."""
import json

import pytest

from label import harness, pieces, prompts
from prepare import formats as f
from test_formats import recorder_folder


def _issue(category, severity="medium", **extra):
    return {"issue": "nothing happens for a while", "category": category, "severity": severity, "t_s": 3.0,
            "end_s": 9.0, "evidence": "top camera 3-9s", **extra}


def _parse(*issues, **labels) -> dict:
    parsed, ok = harness.parse_response(json.dumps({"task_summary": "x", "data_issues": list(issues), **labels}))
    assert ok
    return parsed


@pytest.mark.parametrize("category", ["idle_stretch", "idle", "Idle Stretch", "IdleStretch", "idling_operator",
                                      "pause", "operator_paused", "long_pauses", "stall", "Stalled_work",
                                      "inactivity", "inactive_period", "idle-stretch"])
@pytest.mark.parametrize("severity", ["medium", "high"])
def test_an_idle_data_issue_is_low_and_keeps_the_rating_it_was_labelled_with(category, severity):
    (row,) = _parse(_issue(category, severity))["data_issues"]
    assert row["severity"] == "low" and row["severity_as_labelled"] == severity
    assert {k: v for k, v in row.items() if k not in ("severity", "severity_as_labelled")} == {
        k: v for k, v in _issue(category).items() if k != "severity"}


def test_an_idle_issue_labelled_low_or_unrated_is_low_and_the_rule_is_idempotent():
    low, unrated = _parse(_issue("idle_stretch", "low"), _issue("idle_stretch", None))["data_issues"]
    assert low == _issue("idle_stretch", "low")
    assert unrated["severity"] == "low" and unrated["severity_as_labelled"] is None
    once = _parse(_issue("idle_stretch", "high"))
    again, ok = harness.parse_response(json.dumps(once))
    assert ok and again["data_issues"] == once["data_issues"]
    assert again["data_issues"][0]["severity_as_labelled"] == "high"


@pytest.mark.parametrize("category", ["recording_fault", "camera_stall", "video_paused", "frozen_frame_pause",
                                      "black_frames", "stream_stalled", "gripper_installed", "reinstall_camera",
                                      "phone_use", "face_visible", "instruction_mismatch", "observability_loss"])
def test_a_recording_fault_or_any_other_issue_keeps_its_severity(category):
    for severity in ("medium", "high"):
        assert _parse(_issue(category, severity))["data_issues"] == [_issue(category, severity)]


def test_an_idle_operator_mistake_is_left_as_it_is():
    mistake = {**_issue("idle_stretch", "medium"), "recovered": True, "deliberate": "no"}
    assert _parse(operator_mistakes=[mistake])["operator_mistakes"] == [mistake]


def test_the_prompt_says_idle_stretches_are_low_on_every_rig():
    for rig in ("teleop_arms", "handheld_gripper", "ego_head"):
        text = " ".join(prompts.fixed_instructions(rig).split())
        assert "An idle stretch is always a low data issue" in text, rig
        assert "a frozen or black camera is a recording fault, not idleness" in text, rig


def test_a_stitch_of_parts_labelled_before_the_rule_files_their_idle_issues_low(tmp_path, monkeypatch):
    monkeypatch.setattr(pieces, "piece_max", lambda ctx: 1.0)
    root = tmp_path / "upload"
    recorder_folder(root, n=90, duplicate_stamps=False)
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    src = tmp_path / "eps" / rep["episodes"][0]["episode_id"]
    folders = pieces.write_pieces(src, tmp_path / "job" / "pieces")
    parts = []
    for i, folder in enumerate(folders):
        issues = [{**_issue("idle_stretch", "high" if i == 0 else "medium"), "t_s": 0.2, "end_s": 0.4},
                  {**_issue("recording_fault", "high"), "t_s": 0.5, "end_s": 0.6}]
        record = {"episode_dir": str(folder), "parse_ok": True, "model": "m", "config": {"timesteps_s": [0.0, 0.5]},
                  "usage": {"est_cost_usd": 0.5}, "decode_failed": [],
                  "labels": {"task_summary": "wait", "timeline": [{"start_s": 0.1, "end_s": 0.6, "action": "wait"}],
                             "completion": {"task_completed": "success"}, "data_issues": issues}}
        parts.append((json.loads((folder / "context.json").read_text()), record))
    issues = pieces.stitch(src, parts)["labels"]["data_issues"]
    idle = [row for row in issues if row["category"] == "idle_stretch"]
    faults = [row for row in issues if row["category"] == "recording_fault"]
    assert idle and faults
    assert all(row["severity"] == "low" for row in idle)
    assert sorted(row["severity_as_labelled"] for row in idle) == sorted(
        ["high"] + ["medium"] * (len(idle) - 1))
    assert all(row["severity"] == "high" and "severity_as_labelled" not in row for row in faults)


# Idle that is really other activity keeps the model's severity and category: each of these is at least medium under
# PHONES, BARE HANDS, PARKED GRIPPERS AND FACES (label/prompts.py), and a medium item inside a take drops it.
OTHER_ACTIVITY = [
    ("idle_stretch", "The operator stops working and looks at their phone for about 9s.",
     "top camera 41-50s: lit phone screen in the operator's left hand"),
    ("idle_stretch", "Work pauses while the operator holds a smartphone.", "left gripper camera 12-20s"),
    ("pause", "Another person reaches in and moves the cup while the operator waits.",
     "top camera 30-36s: a second person's arm enters from the right"),
    ("inactivity", "No gripper motion; someone else handles the towel.", "exo view 5-11s"),
    ("idle_stretch", "The grippers do nothing while the operator moves the block with a bare hand.",
     "top camera 18-25s: bare hand slides the block"),
    ("stall", "Little progress: the operator's fingers fiddle with the lid while the gripper is on.",
     "right gripper camera 7-12s"),
    ("idle_stretch", "The left gripper is set down on the table while the right gripper keeps working.",
     "top camera 60-75s"),
    ("idle_stretch", "Idle left wrist: the operator takes off the left gripper and keeps working with the right.",
     "left gripper camera 80-95s shows the table, static"),
    ("idle_period", "The right gripper is parked on the table for a while.", "top camera 100-112s"),
]


@pytest.mark.parametrize("category,issue,evidence", OTHER_ACTIVITY)
def test_an_idle_item_that_describes_other_activity_keeps_its_severity_and_category(category, issue, evidence):
    item = {**_issue(category, "medium"), "issue": issue, "evidence": evidence}
    assert _parse(item)["data_issues"] == [item]
    assert _parse({**item, "severity": "high"})["data_issues"] == [{**item, "severity": "high"}]


@pytest.mark.parametrize("issue,evidence", [
    ("The operator stands still with both grippers raised for 14s before the next grasp.", "top camera 20-34s"),
    ("A long pause before the operator starts the task.", "all cameras 0-12s, the scene is static"),
    ("The operator presses the phone-like fob on the table to stop the recording, then nothing happens.",
     "top camera 88-96s, a gripper set down to press the fob"),
])
def test_a_plain_idle_item_is_still_low(issue, evidence):
    (row,) = _parse({**_issue("idle_stretch", "medium"), "issue": issue, "evidence": evidence})["data_issues"]
    assert row["severity"] == "low" and row["severity_as_labelled"] == "medium"
