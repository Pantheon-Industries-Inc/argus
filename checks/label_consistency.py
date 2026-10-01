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
- progress_vs_outcome: the outcome is failure, partial or unclear, yet the timeline's progress reaches 1.0, which
  the schema reserves for the moment the full success predicate holds.
- goal_times_differ: the outcome is success, but goal_reached_at_s and completed_at_s differ; for a success the
  schema makes them the same frame.
- progress_before_goal: the outcome is success, yet the timeline's progress reaches 1.0 more than 3 s before the
  goal frame, which is the first frame the success predicate holds. (Within a few seconds the two differ only by
  where a step ends.)
- progress_past_goal: the outcome is success, yet the timeline's progress reaches 1.0 only more than 3 s after the
  goal frame, while the goal alignment calls the footage exactly the given goal ("aligned") and no instruction
  mismatch is recorded. The demonstration kept working past the goal, typically handling more of a
  repeated item than the instruction names ("place the coffee filter" with two filters), and nothing says so.
"""
from __future__ import annotations

REACHED = ("success", "success_then_undone")
NOT_REACHED = ("failure", "partial", "unclear")
OTHER_TASK = ("different", "unrelated")


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _levels(steps: list[dict]) -> list[tuple[float, float]]:
    """The timeline's progress in time order, each value at its step's end; an idle step counts only when it raises
    the level (a parked arm's step would otherwise read as a drop), as the board's chart reads it."""
    def end(s):
        a, b = _num(s.get("start_s", s.get("t_s"))), _num(s.get("end_s"))
        return b if b is not None and a is not None and b >= a else a
    rows = sorted(((end(s), i, s) for i, s in enumerate(steps) if end(s) is not None and _num(s.get("progress")) is not None),
                  key=lambda r: (r[0], r[1]))
    out, cur = [], 0.0
    for t, _, s in rows:
        p = max(0.0, min(1.0, _num(s.get("progress"))))
        if str(s.get("contribution") or "").lower() == "idle" and p <= cur:
            continue
        out.append((t, p))
        cur = p
    return out


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
    done = _num(c.get("completed_at_s"))
    if outcome == "success" and done is not None and goal is not None and abs(goal - done) > 0.5:
        out.append({"rule": "goal_times_differ",
                    "note": f"The outcome is success, but the goal is reached at {goal:.1f} s and the goal frame is "
                            f"{done:.1f} s; for a success they are the same frame."})
    if outcome == "success" and done is not None and steps:
        lv = _levels(steps)
        first_full = next((t for t, p in lv if p >= 0.99), None)
        at_goal = ([p for t, p in lv if t <= done + 0.05] or [0.0])[-1]
        GAP = 3.0   # seconds; within this the timeline and the goal frame differ only by where a step ends
        mism = any(str((i or {}).get("category") or (i or {}).get("family") or "").replace("-", "_") == "instruction_mismatch"
                   for i in label.get("data_issues") or [] if isinstance(i, dict))
        if first_full is not None and first_full < done - GAP:
            out.append({"rule": "progress_before_goal",
                        "note": f"The timeline's progress reaches 100% at {first_full:.1f} s, before the goal frame at "
                                f"{done:.1f} s, which is the first frame the goal holds."})
        elif first_full is not None and first_full > done + GAP and rel == "aligned" and not mism:
            out.append({"rule": "progress_past_goal",
                        "note": f"The demonstration keeps working past the goal frame: progress is {at_goal:.0%} at "
                                f"the goal frame ({done:.1f} s) and reaches 100% only at {first_full:.1f} s, yet nothing "
                                "says it does more than the instruction. The instruction may name fewer items than "
                                "were handled."})
    if duration_s:
        for key in ("goal_reached_at_s", "undone_at_s", "completed_at_s"):
            t = _num(c.get(key))
            if t is not None and t > duration_s + 1.0:
                out.append({"rule": "time_past_end",
                            "note": f"{key} is {t:.1f} s, after the episode ends at {duration_s:.1f} s."})
    return out
