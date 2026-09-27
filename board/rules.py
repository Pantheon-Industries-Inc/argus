"""The rules board/build.py applies after labelling (apply_rules) that hold for every dataset of a rig.

A board manifest lists each dataset's rules explicitly: rules_for(rig) and whatever is particular to the dataset
(a fixed file length, a check withheld where its flags are normal). configs/quickstart/board.json is built this way.

    from board import rules
    entry["rules"] = rules.rules_for("ego_head") + [<dataset rules>]
"""
from __future__ import annotations

# every dataset: an operator mistake that another finding already counts is capped, never counted twice
GENERAL = [
    {"kind": "cap_when", "list": "operator_mistakes", "tags": ["incomplete_task"],
     "when_tags": ["instruction_mismatch", "annotation_mismatch", "phase_annotation_mismatch"], "severity": "low",
     "reason": "judged unfinished against an instruction the footage does not match; the instruction mismatch already "
               "counts this episode"},
    {"kind": "cap_when", "list": "operator_mistakes", "tags": ["goal_undone", "task_undone", "incomplete_task"],
     "when_outcome": ["success_then_undone"], "severity": "low",
     "reason": "the goal was reached and then taken apart; the outcome already counts the episode as a data issue "
               "(success then undone), and neither the set-down after the goal nor a task left unfinished after it is "
               "a demonstration mistake"},
    # an episode that ships no task text: the model infers the task from the footage, and the missing text is not a
    # fault of the recording (Egocentric-100K and OpenAoE ship none; neither does most plain video)
    {"kind": "no_task_text",
     "pattern": r"^(missing|no|absent|lacking)_?(task_)?(instruction|annotation|task_text|task_description|label)s?$",
     "reason": "the episode ships no task text; the task is inferred from the footage, and a missing task text is not "
               "a fault in the recording"},
]

BY_RIG = {
    "teleop_arms": [],
    "handheld_gripper": [],
    "ego_head": [
        {"kind": "severity_cap", "tags": ["idle_stretch", "excessive_idle"], "severity": "low",
         "reason": "a head-camera wearer pausing between activities is the normal rhythm of the recording, not a "
                   "defect in the footage"},
    ],
}


def rules_for(rig: str | None) -> list[dict]:
    """The general rules and the rig's own (teleop_arms, handheld_gripper or ego_head), as fresh copies."""
    return [dict(r) for r in GENERAL + BY_RIG.get(rig or "", [])]
