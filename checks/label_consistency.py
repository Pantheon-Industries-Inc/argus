"""Find annotations that contradict themselves.

A label is one model call's answer, and its fields are written in one pass: the outcome, the goal
alignment, the state changes and the times. When two of them cannot both be true, at least one is wrong,
and no frame is needed to know it. This check reads only the stored label (completion, goal_alignment,
state_changes, and timeline or event_labels) and the episode's length. `python -m board build` runs it on every
episode and stores its findings as label_consistency in the board file. It never changes a label: it reports
the contradiction so the episode is looked at, and relabelled with the current harness. It has no command line.

Each finding is {"rule": <id>, "note": <plain sentence>}. The rules follow from what the fields mean in
the output schema, not from any one episode:
- outcome_vs_alignment: the outcome says the given goal was reached (success, or success then undone)
  while goal_alignment says the footage shows a different or unrelated task. A goal cannot be reached by
  doing another task.
- undone_timing: success then undone without a goal time, without an undo time, or with the undo at or
  before the goal.
- nothing_after_goal: success then undone, yet the label's own state changes record nothing after the goal
  time, so by its own account the state at the goal is the end state and nothing was undone.
- time_past_end: an outcome time later than the end of the episode.
- outcome_vs_partial: the outcome says the given goal was reached, while goal_alignment says only part of it
  happened ("broader"). A goal cannot be reached when part of it never happened.
- alignment_vs_match: goal_alignment says "aligned" (exactly the given goal) and also that the footage does not
  depict the given goal (matches_given false).
- progress_vs_outcome: the outcome is failure or partial, yet the timeline's progress reaches 1.0, which the
  schema reserves for the moment the full success predicate holds. The progress line would read as the goal
  reached on an episode the label says never reached it.
"""
from __future__ import annotations

REACHED = ("success", "success_then_undone")
NOT_REACHED = ("failure", "partial")
OTHER_TASK = ("different", "unrelated")


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def check(label: dict, duration_s: float | None = None) -> list[dict]:
    c = label.get("completion") or {}
    outcome = str(c.get("task_completed") or "").lower()
    rel = str((label.get("goal_alignment") or {}).get("relation") or "").lower()
    goal, undone = _num(c.get("goal_reached_at_s")), _num(c.get("undone_at_s"))
    out = []
    if outcome in REACHED and rel in OTHER_TASK:
        out.append({"rule": "outcome_vs_alignment",
                    "note": f"The outcome says the given goal was reached ({outcome.replace('_', ' ')}), but the goal "
                            f"alignment says the footage shows a {rel} task."})
    if outcome in REACHED and rel == "broader":
        out.append({"rule": "outcome_vs_partial",
                    "note": f"The outcome says the given goal was reached ({outcome.replace('_', ' ')}), but the goal "
                            "alignment says only part of it happened."})
    if rel == "aligned" and (label.get("goal_alignment") or {}).get("matches_given") is False:
        out.append({"rule": "alignment_vs_match",
                    "note": "The goal alignment says the footage is exactly the given goal and also that it does not "
                            "depict it."})
    if outcome == "success_then_undone":
        if goal is None or undone is None or undone <= goal:
            out.append({"rule": "undone_timing",
                        "note": "The outcome is success then undone, but the goal and undo times are missing or the "
                                "undo does not come after the goal."})
        elif not any((_num((s or {}).get("t_s")) or -1) > goal for s in label.get("state_changes") or []
                     if isinstance(s, dict)):
            out.append({"rule": "nothing_after_goal",
                        "note": "The outcome is success then undone, but the label records no state change after "
                                "the goal was reached."})
    # the harness returns timeline rows; the board stores them as event_labels
    steps = [s for key in ("timeline", "event_labels") for s in (label.get(key) or []) if isinstance(s, dict)]
    peak = max((p for p in (_num(s.get("progress")) for s in steps) if p is not None), default=None)
    if outcome in NOT_REACHED and peak is not None and peak >= 0.99:
        out.append({"rule": "progress_vs_outcome",
                    "note": f"The outcome is {outcome}, but the timeline's progress reaches {peak:.0%}, the level kept "
                            f"for the moment the goal is reached."})
    if duration_s:
        for key in ("goal_reached_at_s", "undone_at_s", "completed_at_s"):
            t = _num(c.get(key))
            if t is not None and t > duration_s + 1.0:
                out.append({"rule": "time_past_end",
                            "note": f"{key} is {t:.1f} s, after the episode ends at {duration_s:.1f} s."})
    return out
