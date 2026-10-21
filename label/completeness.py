"""Whether a reply holds the work its request asked for. Every site that calls an annotation or a part done, reusable
or skippable reads this one rule through label/harness.py annotation_problems: the harness when a reply arrives and
on resume, each part of a recording labelled in parts (label/pieces.py), stored replies read again
(label/reparse.py), the review's cost floor (review/__main__.py), the run's count of labelled units (label/run.py),
the model comparison (compare/metrics.py) and the board's labelled episodes (board/to_board.py label_failed).

A request over footage asks for two things, and both are read from the record's own request receipt.

The timeline. The footage sent is the request's sampled instants (config.timesteps_s), and the model saw nothing
between them. Each instant therefore stands for the stretch of time nearer to it than to its neighbours, and the
first and last stretch reach as far out as their one neighbour does. The timeline covers the footage when every
instant's stretch meets a timed step. A step outside the sent span meets no stretch and covers nothing.

The outcome. The field the request's own instructions ask for (label/prompts.py requested_outcome). A robot request
with an instruction asks for completion.task_completed, judged against it: that is its outcome, and the list of tasks
it asks beside it is not (a missing list, or a task with no outcome, is a schema issue the record names). Any other
request asks for a tasks list with an outcome on every task: at least one task, except on a robot recording with
neither an instruction nor timed steps, where idle footage holds none. A record asked before the tasks format (no
label_schema) is held to what its mode asked: completion.task_completed for an episode, a tasks list with an outcome on
every task for a session or the head camera.

A recording labelled in parts is complete when every part was, each against its own request (label/pieces.py
stitch). A record that names no sampled instants (written by hand, or by a harness older than the receipt) is held
to a timed timeline and the outcome only.
"""
from __future__ import annotations

import math


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _answered(value) -> bool:
    """The reply gave a value: anything but null, blank text or an empty container. A value outside the field's
    allowed set is an answer the parser lists in _schema_violations, never a missing one."""
    return value is not None and not (isinstance(value, str) and not value.strip()) and value not in ([], {})


def sampled_instants(record: dict) -> list[float]:
    """The instants the request sent, in order, from its receipt (config.timesteps_s)."""
    config = record.get('config') if isinstance(record.get('config'), dict) else {}
    times = config.get('timesteps_s')
    return sorted({float(t) for t in times if _number(t)}) if isinstance(times, list) else []


def stretches(instants: list[float]) -> list[tuple[float, float, float]]:
    """(instant, from, to): the stretch of time each sampled instant stands for."""
    out = []
    for k, t in enumerate(instants):
        before = (t - instants[k - 1]) / 2 if k else None
        after = (instants[k + 1] - t) / 2 if k + 1 < len(instants) else None
        out.append((t, t - (before if before is not None else after or 0.0),
                    t + (after if after is not None else before or 0.0)))
    return out


def timeline_gaps(labels: dict, record: dict) -> list[str]:
    """Why the timeline does not cover the footage the request sent, or [] when it does."""
    rows = labels.get('timeline')
    if not isinstance(rows, list) or not rows:
        return ['the reply has no timeline']
    steps = [(float(r['start_s']), float(r['end_s'])) for r in rows if isinstance(r, dict)
             and _number(r.get('start_s')) and _number(r.get('end_s')) and r['end_s'] >= r['start_s']]
    if not steps:
        return ['no timeline step has a start and an end']
    instants = sampled_instants(record)
    eps = 1e-6
    uncovered = [k for k, (_, lo, hi) in enumerate(stretches(instants))
                 if not any(start <= hi + eps and end >= lo - eps for start, end in steps)]
    if not uncovered:
        return []
    runs = []
    for k in uncovered:
        if runs and k == runs[-1][1] + 1:
            runs[-1][1] = k
        else:
            runs.append([k, k])
    where = ', '.join(f'{instants[a]:g} s' if a == b else f'{instants[a]:g} to {instants[b]:g} s' for a, b in runs)
    return [f'the timeline leaves {len(uncovered)} of the {len(instants)} sampled instants '
            f'({instants[0]:g} to {instants[-1]:g} s) uncovered: {where}']


def _steps(record: dict) -> bool:
    """Whether the request gave the dataset's own timed steps (label/episode.py has_steps)."""
    count = record.get('task_prompts')
    return type(count) is int and count > 0


def tasks_format(record: dict) -> bool:
    """Whether the request was asked in the tasks format (label/prompts.py): its record says so (label_schema), or its
    policy names a rig and no task mode, as every policy since that format does. A policy saved before it always
    names its task mode, episode or session (label/policy.py)."""
    from board.label_projection import LABEL_SCHEMA
    policy = record.get('annotation_policy') if isinstance(record.get('annotation_policy'), dict) else {}
    return record.get('label_schema') == LABEL_SCHEMA or (isinstance(policy.get('rig'), str)
                                                          and policy.get('task_mode') is None)


def requested_outcome(record: dict) -> str | None:
    """Where the request asked for its outcome: in the tasks format, read from the text it sent (label/prompts.py
    requested_outcome); for an older record, from the rig and task mode it resolved: "tasks" for a session or the head
    camera, the one completion for an episode. None for a record that does not say how it was asked."""
    from label.prompts import requested_outcome as asked
    policy = record.get('annotation_policy') if isinstance(record.get('annotation_policy'), dict) else {}
    if tasks_format(record):
        rig = policy.get('rig') if isinstance(policy.get('rig'), str) else 'teleop_arms'
        return asked(rig, has_instruction=_text(record.get('given_prompt')), has_steps=_steps(record))
    if not isinstance(policy.get('rig'), str):
        return None
    sessions = policy.get('task_mode') == 'session' or record.get('prompt_mode') == 'session'
    return 'tasks' if policy['rig'] == 'ego_head' or sessions else 'completion.task_completed'


def outcome_gaps(labels: dict, record: dict) -> list[str]:
    """Why the reply lacks the outcome its request asked for, or [] when it has it."""
    field = requested_outcome(record)
    tasks = labels.get('tasks')
    if field is None:
        # a record that does not say how it was asked holds its outcome in either form
        either = [outcome_gaps(labels, {**record, 'annotation_policy': {'rig': 'teleop_arms', 'task_mode': mode}})
                  for mode in ('episode', 'session')]
        return [] if not all(either) else either[0]
    if field == 'tasks':
        # in the tasks format a robot recording with neither an instruction nor timed steps may hold no task (idle
        # footage); the list is still asked. A head-camera clip always holds one, as it always did.
        policy = record.get('annotation_policy') if isinstance(record.get('annotation_policy'), dict) else {}
        none_allowed = (tasks_format(record) and policy.get('rig') != 'ego_head'
                        and not _text(record.get('given_prompt')) and not _steps(record))
        if not isinstance(tasks, list) or not (tasks or none_allowed):
            return ['the request asks each activity in tasks with its outcome, and the reply lists none']
        missing = [str(i) for i, task in enumerate(tasks)
                   if not isinstance(task, dict) or not _answered(task.get('outcome'))]
        return [f'task {", ".join(missing)} of {len(tasks)} has no outcome'] if missing else []
    completion = labels.get('completion')
    if not isinstance(completion, dict) or not _answered(completion.get('task_completed')):
        return ['the request asks the outcome of its task, and completion.task_completed is missing']
    # the tasks a robot request with an instruction asks beside its completion are not its outcome: a list that is
    # missing, or a task with no outcome, is a schema issue the record names (board/label_projection.py
    # derive_completion), never a reason to call the reply unfinished
    return []


def parts_gaps(record: dict) -> list[str]:
    """Why a recording labelled in parts is not complete: each part with no labels, with its own reason."""
    stitched = record.get('stitched') if isinstance(record.get('stitched'), dict) else {}
    gaps = [f"part {g.get('part')} has no labels: {g.get('why') or 'no reason recorded'}"
            for g in stitched.get('missing') or [] if isinstance(g, dict)]
    return gaps or (['no part of the recording has labels'] if record.get('no_part') else [])


def annotation_gaps(labels: dict, record: dict) -> list[str]:
    """Everything the request asked for that the parsed reply's labels do not hold; [] means the reply is complete.
    record is the saved record, or the request fields it is built from (config, annotation_policy, given_prompt)."""
    if not isinstance(labels, dict):
        return ['the labels are not a JSON object']
    if isinstance(record.get('stitched'), dict):
        return parts_gaps(record)
    return timeline_gaps(labels, record) + outcome_gaps(labels, record)
