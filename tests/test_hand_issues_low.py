"""A bare-hand data issue is medium or high only when it is egregious (label/hand_issues.py, label/prompts.py PHONES, BARE
HANDS, PARKED GRIPPERS AND FACES). The cases are the issue and evidence text of 19 real human_intervention items the 10K
dropped, read by the owner on 2026-11-11: something in the mouth, a helper's hand, a hand reaching below the table for
a fallen object and fingers holding a tool keep medium; fingers that appear to help, or help while the gripper stays on,
become low and keep the model's rating in severity_as_labelled."""
import json

import pytest

from label import harness
from label.hand_issues import hand_issues_low

KEEP = {
    "mouth": ("The fine thread end is brought into contact with the operator's mouth again.",
              "Right at 344.34-345.35 s shows close lip contact with the fine material at the jaw tips."),
    "mouth_work": ("The operator uses the mouth to work the fine sewing material held at the right jaw tips.",
                   "exo shows the right gripper repeatedly raised to the face."),
    "helper": ("A helper's bare hand returns the dropped white collar.",
               "exo at 31.03 s shows an additional bare arm entering from the right with the white ring."),
    "below_table": ("A bare hand retrieves the fallen cap and puts it back in the task workspace.",
                    "exo shows the operator reaching below the right edge, then returning the white cap to the table by "
                    "454.28 s; the grippers remain parked."),
    "tool_held": ("The pliers are held and operated by bare fingers while the right gripper remains strapped on.",
                  "left clearly shows right fingers around the red plier handles."),
    "gripper_off": ("The operator takes off the right gripper and slides the cup across the table by hand.",
                    "exo at 12.0 s."),
}
LOW = {
    "brief_help": ("A bare index finger helps position the wire stripper while the gripper remains worn.",
                   "right, 212.21-213.21 s: the extended bare finger contacts the tool near its red handle."),
    "appears": ("Exposed right fingers appear to press or smooth the folded task towel directly.",
                "The exposed fingers appear to touch the cloth; the precise contact boundary is partly obscured."),
    "zipper": ("Bare fingertips appear to assist the insulated bag zipper rather than only holding the gripper.",
               "Left close view at 206.21 s shows fingertips reaching beneath the bag edge at the zipper."),
    "strand": ("Bare fingers manipulate a task strand while the grippers are being used.",
               "left: bare thumb and fingers visibly pinch or pull a long paper strand near the left jaws."),
    "flange": ("Bare fingertips contact and appear to turn the white upper flange while the gripper remains worn.",
               "left close views show fingertips directly against the flange beneath the upper rim."),
    "deck": ("Exposed fingers press or adjust the mounting area while the grippers hold the deck.",
             "Exo shows the left fingers reaching past the housing onto the near deck/baseplate area."),
    "hoop": ("Bare fingers repeatedly assist with supporting and turning the hoop and fabric alongside the grippers.",
             "exo shows fingers directly on the hoop or cloth."),
    "hose": ("Bare fingers brace and manipulate the corrugated hose while the grippers continue working.",
             "exo shows fingers spread over the hose."),
    "board": ("Bare fingertips brace the raised task board in addition to the grippers.",
              "Left wrist shows bare fingers curled onto the board's left edge, above the foreground gripper tips."),
}


def _issue(text, severity="medium", category="human_intervention"):
    issue, evidence = text
    return {"issue": issue, "category": category, "severity": severity, "t_s": 3.0, "end_s": 5.0, "evidence": evidence}


@pytest.mark.parametrize("name", sorted(KEEP))
@pytest.mark.parametrize("severity", ["medium", "high"])
def test_an_egregious_bare_hand_issue_keeps_its_severity(name, severity):
    labels = {"data_issues": [_issue(KEEP[name], severity)]}
    hand_issues_low(labels)
    assert labels["data_issues"][0]["severity"] == severity and "severity_as_labelled" not in labels["data_issues"][0]


@pytest.mark.parametrize("name", sorted(LOW))
def test_a_guessed_or_gripper_on_bare_hand_issue_is_low_and_keeps_its_rating(name):
    labels = {"data_issues": [_issue(LOW[name])]}
    hand_issues_low(labels)
    row = labels["data_issues"][0]
    assert row["severity"] == "low" and row["severity_as_labelled"] == "medium"
    hand_issues_low(labels)
    assert row["severity"] == "low" and row["severity_as_labelled"] == "medium"


def test_other_categories_and_low_items_are_left_alone():
    face = _issue(LOW["appears"], category="face_visible")
    camera = _issue(LOW["appears"], category="camera_fault")
    low = _issue(LOW["appears"], severity="low")
    labels = {"data_issues": [dict(face), dict(camera), dict(low)]}
    hand_issues_low(labels)
    assert labels["data_issues"] == [face, camera, low]


def test_the_parser_applies_it():
    parsed, ok = harness.parse_response(json.dumps({"task_summary": "x",
                                                    "data_issues": [_issue(LOW["appears"]), _issue(KEEP["mouth"])]}))
    assert ok
    assert [r["severity"] for r in parsed["data_issues"]] == ["low", "medium"]
