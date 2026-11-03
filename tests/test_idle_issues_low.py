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


# The same kinds in words real UMI labels used (every idle item of the UMI labels, scanned 2026-11-03): a gripper that
# remains parked, rests on the table, has been set down or is left resting there, and work assisted by someone else.
# Each was lowered before the rule read those words.
OTHER_ACTIVITY += [
    ("idle_stretch", "Bag loading pauses while one gripper remains parked on the table.",
     "exo at 335.33-344.34 s shows the right gripper on the table"),
    ("idle_stretch", "A long non-task pause includes headphone adjustment while the right gripper remains on the table.",
     "exo and left show no task-object manipulation around 272-277 s"),
    ("idle_stretch", "A roughly ten-second pause interrupts the paper work while the right gripper rests on the table.",
     "exo shows no paper work at 140-150 s"),
    ("idle_stretch", "Resting after hardware manipulation stops.",
     "All three views at 436.44-446.45 s show no further task-object manipulation; the left gripper rests on the table."),
    ("idle_stretch", "The recording continues through a long inactive stretch after the rice transfers; the right gripper "
     "has been set down.", "exo shows no further task-object manipulation"),
    ("idle_stretch", "The demonstrator pauses and turns toward the neighboring participant while the right gripper "
     "remains set down.", "Exo shows the pause"),
    ("idle_stretch", "A long pause with no task contact.",
     "exo and both wrist views at 780.31-787.32 s show unchanged fixtures and no task contact; the right gripper is "
     "visibly parked."),
    ("idle_stretch", "A parked-gripper pause records several seconds without excavation activity.", "exo 50-58 s"),
    ("idle_stretch", "There is a long pause around the externally assisted clearing of the headphones.",
     "exo and both gripper views show no card manipulation during 169.17-179.18 s."),
    ("idle_stretch", "Both grippers are left resting on the table while the demonstrator leans away.", "exo 80-88 s"),
    # a fob named elsewhere in the text does not hide phone use or a parked gripper
    ("idle_stretch", "The operator pauses and looks at a phone.",
     "exo 40-52 s; the recording fob lies untouched on the table."),
    ("idle_stretch", "The right gripper remains parked while the left holds the bag.",
     "exo 10-20 s. The fob is pressed at 95 s."),
    ("idle_stretch", "Work stops while the operator checks an iPhone.", "exo 300-320 s"),
    ("idle_stretch", "Work stops while the operator answers a call.", "exo 300-320 s"),
    # a denial of the grippers' motion, or one after what the person did, leaves the person's action standing
    ("idle_stretch", "The grippers do not move while another person adjusts the bowl.", "exo 30-40 s"),
    ("inactivity", "No gripper motion while someone else handles the towel.", "exo view 5-11s"),
    ("pause", "Another person reaches in and moves the cup, which the operator did not expect.", "top camera 30-36s"),
    # a denial about the operator is not a denial of what the other person did (issue 185)
    ("pause", "Another person reaches in and moves the cup without the operator noticing.", "top camera 30-36s"),
    ("idle_stretch", "Someone else, not the operator, handles the towel.", "exo 12-20 s"),
    # a denial after the person's action that is about the operator or about something else leaves the action standing
    ("idle_stretch", "Another person adjusts the bowl while the operator does not move.", "exo 50-60 s"),
    ("idle_stretch", "Someone else takes the cup and the operator doesn't react.", "exo 50-60 s"),
    ("pause", "Someone else hands the cup to the operator with no gripper motion.", "top camera 10-18s"),
    ("idle_stretch", "Another person moves the bowl while no task progress is made.", "exo 70-80 s"),
    ("idle_stretch", "The operator presses the fob, then looks at the phone.", "exo 90-99 s"),
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
    # real label text: grippers that stay still, near the table or clear of the work, are plain idle
    ("Both grippers remain inactive for about ten seconds between jug placements.", "All three views show no change"),
    ("A prolonged stretch contains no puzzle manipulation while both grippers remain near the table edge.",
     "exo, left and right at 359.36-386.39 s show stationary grippers"),
    ("An extended pause occurs while the demonstrator looks toward a nearby person.",
     "exo and right at 1180.05-1196.06 s show the demonstrator oriented toward them"),
    # real label text: a person nearby who never touches the work, the gripper's own fingers, a gripper held still in
    # the air and a grasp are plain idle
    ("Candy handling pauses while another person stands nearby and gestures.", "exo and right at 205.21-214.21 s"),
    ("A long pause between handling sequences.",
     "exo and both close views show nearly stationary grippers while another person stands at the doorway."),
    ("The striped towel is left unchanged for an interval.",
     "The towel remains unchanged in exo; the other person moves beside the workspace without visibly touching task "
     "objects."),
    ("The pitcher is held with little change.",
     "left and right views show only small orientation and empty-finger movements without a new grasp."),
    ("A pause in the pouch work.", "All three views show nearly unchanged positions, apart from small posture and "
     "finger movements."),
    ("Both grippers wait above the work.", "exo shows both grippers parked above the table at 98.10-108.11 s."),
    ("A pause before the cup is lifted.", "right shows the operator places the gripper jaw around the cup and waits."),
    # the other person's own action denied, with the operator named in the same part, is still plain idle (issue 185)
    ("A pause in the cup work.", "exo 40-48 s: another person beside the operator does not touch the cup."),
    # a verb that is the operator's, denied, is no action of the other person's
    ("A pause in the cup work.", "Another person stands in the background with no operator movement."),
    ("A pause in the cup work.", "A bystander watches without the operator moving the cup."),
    ("A pause in the cup work.", "Someone else watches while the operator is not moving the cup."),
])
def test_a_plain_idle_item_is_still_low(issue, evidence):
    (row,) = _parse({**_issue("idle_stretch", "medium"), "issue": issue, "evidence": evidence})["data_issues"]
    assert row["severity"] == "low" and row["severity_as_labelled"] == "medium"


def test_the_head_camera_board_cap_never_lowers_an_idle_item_that_describes_other_activity():
    """The ego_head board rule caps idle tags at low (board/rules.py); an idle item that is really other activity keeps
    its severity there too, as it does in the parser."""
    from board import build as board_build, rules
    phone = {**_issue("idle_stretch", "high"), "issue": "The wearer stops and looks at a phone."}
    plain = _issue("idle_stretch", "high")
    d = {"data_issues": [phone, plain]}
    board_build.apply_rules(d, {}, rules.rules_for("ego_head"))
    assert d["data_issues"][0] == phone
    assert d["data_issues"][1]["severity"] == "low" and d["data_issues"][1]["model_severity"] == "high"
