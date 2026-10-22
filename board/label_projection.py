"""Project partial labels into consumer-safe values without changing the saved reply.

This module is the one reader of a raw model reply for every consumer: the board (board/to_board.py), compare
(compare/metrics.py), the gate (gate/score.py) and the consistency checks (checks/label_consistency.py). project()
keeps the valid fields and lists every departure from the output format in _schema_violations, one per source
field. The values those consumers derive from a label are defined here once: the effective outcome, an issue's
severity tier (unrated when it has none of SEVERITIES) and the highest rated severity of a list.
"""
from __future__ import annotations

import copy
import json
import math
import re

OUTCOMES = ('success', 'success_then_undone', 'failure', 'partial', 'unclear')
TASK_OUTCOMES = OUTCOMES + ('continues_past_cut',)
EVENT_OUTCOMES = ('success', 'failure', 'unclear')


SEVERITIES = ('low', 'medium', 'high')
SEVERITY_RANK = {name: rank for rank, name in enumerate(SEVERITIES, start=1)}
ISSUE_LISTS = ('data_issues', 'operator_mistakes')


def effective_outcome(value):
    return 'failure' if value == 'partial' else value


def open_at_cut(task) -> bool:
    """Whether a task is still under way at a cut of the recording (continues_past_cut, label/pieces.py
    mark_cut_tasks). Its outcome is unknown, so a count of tasks done leaves it out of both the done count and the
    total, and says how many it left out."""
    return isinstance(task, dict) and str(task.get('outcome') or '').strip().lower() == 'continues_past_cut'


# The output format of a request that asks every recording for a list of tasks, each with its own outcome
# (label/prompts.py fixed_instructions). label/harness.py writes it into every request record as label_schema, so it
# names the prompt actually sent; a record without it was asked in the episode or session mode of its saved policy.
LABEL_SCHEMA = 'tasks'
# what a task is judged against: the dataset's timed step that covers it, the episode's instruction, or its own end
# state (label/prompts.py TASK_RULES)
JUDGED_AGAINST = ('subtask', 'instruction', 'own_end_state')
# what a completion in that format stands on (its basis): the reply's own completion, judged against the episode's
# instruction; a recording's one task; no verdict, for a recording whose dataset times its own steps and gives no
# instruction (task_prompts) or one of several tasks, or none (tasks); or "unclear" for a recording with an instruction
# labelled in parts, which no part sees whole (parts)
BASES = ('instruction', 'one_task', 'task_prompts', 'tasks', 'parts')
# a task's outcome fields, under the names the recording's completion gives them
COMPLETION_FROM_TASK = (('task_completed', 'outcome'), ('success_predicate', 'success_predicate'),
                        ('completed_at_s', 'completed_at_s'), ('goal_reached_at_s', 'goal_reached_at_s'),
                        ('undone_at_s', 'undone_at_s'), ('undone_by', 'undone_by'), ('reason', 'note'),
                        ('judged_against', 'judged_against'))


def recording_completion(tasks, *, task_prompts: int = 0) -> dict:
    """The completion of a recording with no instruction, derived from its tasks. A recording of exactly one task
    takes that task's outcome, times and predicate as its own. One whose dataset times its own steps, or one of
    several tasks or none, has no single verdict: every field is null, and each task keeps its own outcome under
    tasks. from_tasks says how many tasks it was derived from and basis what it stands on (BASES). A task still open
    at a cut of the recording (continues_past_cut), or one whose outcome is no outcome word, is no verdict either."""
    rows = [t for t in tasks if isinstance(t, dict)] if isinstance(tasks, list) else []
    empty = {name: None for name, _ in COMPLETION_FROM_TASK}
    n = len(rows)
    if task_prompts:
        return {**empty, 'reason': "the dataset times its own steps and gives no instruction for the recording; each "
                                   "task's outcome, against its step, is under tasks", 'from_tasks': n,
                'basis': 'task_prompts'}
    word = str(rows[0].get('outcome') or '').strip().lower() if n == 1 else None
    if word == 'continues_past_cut':
        return {**empty, 'reason': 'its one task continues past a cut of the recording, so its outcome is open',
                'from_tasks': 1, 'basis': 'tasks'}
    if n == 1 and word not in OUTCOMES:
        return {**empty, 'reason': f"its one task has no known outcome ({rows[0].get('outcome')!r})",
                'from_tasks': 1, 'basis': 'tasks'}
    if n == 1:
        out = {name: copy.deepcopy(rows[0].get(key)) for name, key in COMPLETION_FROM_TASK}
        # "Success" is success, as the parser reads a completion's outcome
        return {**out, 'task_completed': word, 'from_tasks': 1, 'basis': 'one_task'}
    reason = 'no task observed' if not rows else f"{n} tasks; each task's outcome is under tasks"
    return {**empty, 'reason': reason, 'from_tasks': n, 'basis': 'tasks'}


# A record's progress_scope says what its timeline's progress measures when that is not the recording's verdict:
# "request_parts" for a recording labelled in parts (label/pieces.py), whose steps keep each part's local progress, and
# PROGRESS_PER_TASK for a head-camera clip whose verdict is its annotated goal's outcome, whose progress restarts at
# each sub-task. A record without one measures progress toward its verdict.
PROGRESS_PER_TASK = 'tasks'


def progress_toward_verdict(record) -> bool:
    """Whether a record's timeline progress measures progress toward its own verdict (no progress_scope)."""
    return not (isinstance(record, dict) and record.get('progress_scope'))


def goal_outcome_word(goal_outcome) -> str | None:
    """The outcome a head-camera reply gives its annotated goal (goal_outcome.outcome), as one of OUTCOMES, else
    None."""
    word = goal_outcome.get('outcome') if isinstance(goal_outcome, dict) else None
    word = word.strip().lower() if isinstance(word, str) else None
    return word if word in OUTCOMES else None


def derive_completion(labels: dict, *, rig: str, instruction: bool, task_prompts: int = 0) -> dict:
    """labels with the recording's completion as the LABEL_SCHEMA format reads it, by what the request gave. A robot
    request with an instruction asks for the completion itself, judged against the instruction as the episode prompt
    always did; it is kept and marked basis "instruction". One without asks only for tasks, and its completion is
    derived from them (recording_completion); a completion the reply writes anyway is replaced and noted in
    _schema_violations. The head camera's schema has no completion: a clip with an annotated goal whose reply gives
    the goal's outcome (goal_outcome, label/prompts.py EGO_GOAL_OUTCOME) takes it as its completion, judged against
    the goal; otherwise a clip of one task with no timed steps takes that task's verdict, and any other clip is read
    task by task, as it always was. Call it only on a record whose
    label_schema is LABEL_SCHEMA: an older record keeps the completion its own request asked for."""
    if not isinstance(labels, dict):
        return labels
    out = dict(labels)
    # a task can be judged only against a claim the request gave: one the reply says it was judged against that the
    # recording does not carry (an instruction on a recording without one, a timed step on one without steps) is judged
    # against its own end state, and the correction is named
    if isinstance(out.get('tasks'), list):
        given = {'instruction': instruction, 'subtask': bool(task_prompts)}
        tasks, notes = [], []
        for i, task in enumerate(out['tasks']):
            claim = task.get('judged_against') if isinstance(task, dict) else None
            if claim in given and not given[claim]:
                task = {**task, 'judged_against': 'own_end_state', 'subtask': None}
                notes.append(f'tasks: task {i} said it was judged against the '
                             f'{"timed step" if claim == "subtask" else claim}, which the request did not give; it is '
                             'judged against its own end state')
            tasks.append(task)
        out['tasks'] = tasks
        notes = [n for n in notes if n not in (out.get('_schema_violations') or [])]
        if notes:
            out['_schema_violations'] = [*(out.get('_schema_violations') or []), *notes]
    if rig != 'ego_head' and instruction:
        if isinstance(out.get('completion'), dict):
            out['completion'] = {**out['completion'], 'basis': 'instruction'}
        # the completion is the reply's outcome; the tasks asked beside it, missing or without an outcome, are named
        # as schema issues and never make the reply unfinished (label/completeness.py)
        tasks = out.get('tasks')
        # with the dataset's timed steps, the tasks hold the judgment against each step the request asks for
        notes = (['tasks: the request asks the recording\'s tasks beside its completion, and the reply lists none']
                 if not isinstance(tasks, list) else
                 [f'tasks: the request asks each of the dataset\'s {task_prompts} timed steps judged in tasks, and the '
                  'reply lists none'] if not tasks and task_prompts else
                 [f'tasks: task {i} of {len(tasks)} has no outcome' for i, task in enumerate(tasks)
                  if not isinstance(task, dict) or not str(task.get('outcome') or '').strip()])
        notes = [n for n in notes if n not in (out.get('_schema_violations') or [])]
        if notes:
            out['_schema_violations'] = [*(out.get('_schema_violations') or []), *notes]
        return out
    derived = None
    if rig == 'ego_head':
        tasks = [t for t in out.get('tasks') or [] if isinstance(t, dict)] if isinstance(out.get('tasks'), list) else []
        if instruction and goal_outcome_word(out.get('goal_outcome')):
            # the clip has an annotated goal and the reply says whether it reached the goal's end state: that is the
            # clip's verdict, whatever its tasks
            g = out['goal_outcome']
            derived = {name: None for name, _ in COMPLETION_FROM_TASK}
            derived.update(task_completed=goal_outcome_word(g), success_predicate=g.get('success_predicate'),
                           completed_at_s=g.get('completed_at_s'), reason=g.get('note'), judged_against='instruction',
                           from_tasks=len(tasks), basis='instruction')
            # the head camera's timeline measures progress toward the current sub-task (label/prompts.py), never toward
            # this goal: no reader may take it for the goal's progress (PROGRESS_PER_TASK)
            out['progress_scope'] = PROGRESS_PER_TASK
        # otherwise one task gives the clip its verdict when the clip has an annotated goal, or has no timed steps to
        # judge it against instead; any other clip is read task by task, as on main
        elif len(tasks) != 1 or (task_prompts and not instruction):
            return out
        task_prompts = 0
    source = 'goal_outcome' if derived else 'tasks'
    derived = derived or recording_completion(out.get('tasks'), task_prompts=task_prompts)
    # labels read again (a saved record checked on resume) already hold the derived completion
    if 'completion' in out and out['completion'] != derived:
        out['_schema_violations'] = [*(out.get('_schema_violations') or []),
                                     f'completion: the output format has no completion; it is derived from {source}']
    out['completion'] = derived
    return out


def judge_head_camera_tasks(record: dict, steps: list[dict], goal: str | None) -> None:
    """What each task of a head-camera record was judged against, which its schema does not ask (label/prompts.py
    EGO_SCHEMA): the dataset's timed step covering most of it ("subtask", with the step's number in the whole
    recording, label/task_claims.py), else the annotated goal ("instruction"), else its own end state. Each is marked
    judged_against_derived; a completion taken from the one task carries the same."""
    def number(x):
        return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None
    for task in record.get('tasks') or []:
        if not isinstance(task, dict) or task.get('judged_against') in JUDGED_AGAINST:
            continue
        a, b = number(task.get('start_s')), number(task.get('end_s'))
        best, cover = None, 0.0
        for step in steps:
            s0, s1 = number(step.get('t0')), number(step.get('t1'))
            if None in (a, b, s0, s1) or b <= a:
                continue
            share = max(0.0, min(b, s1) - max(a, s0)) / (b - a)
            if share > cover:
                best, cover = step, share
        if best is not None and cover >= 0.5:
            task.update(judged_against='subtask', subtask=best.get('step'))
        else:
            task.update(judged_against='instruction' if goal else 'own_end_state', subtask=None)
        task['judged_against_derived'] = True
    completion = record.get('completion')
    tasks = [t for t in record.get('tasks') or [] if isinstance(t, dict)]
    if isinstance(completion, dict) and completion.get('basis') == 'one_task' and len(tasks) == 1:
        completion['judged_against'] = tasks[0].get('judged_against')


def recording_verdict(completion) -> str | None:
    """The recording's own outcome word, or None when it has none (several tasks, no task, a session, the head
    camera's activities, a long recording whose task is open at a cut)."""
    word = completion.get('task_completed') if isinstance(completion, dict) else None
    return word.strip().lower() if isinstance(word, str) and word.strip() else None


def verdict_or_tasks(record: dict) -> tuple[str | None, list[dict]]:
    """(the recording's verdict, the tasks that stand in for it). Every reader of an outcome uses this one rule. A
    completion in the tasks format (it names its basis, derive_completion) is read first: its verdict when it has one,
    and then no tasks, so a recording with an instruction is read by its completion against it and a recording of one
    task as one goal, the way an episode always was; otherwise no verdict and the tasks. A record labelled before the
    tasks format keeps the rule it was shown and scored by: its tasks whenever it lists any (a session, the head
    camera's activities, an older long session whose stitched completion says unclear), otherwise its completion's
    verdict."""
    record = record if isinstance(record, dict) else {}
    completion = record.get('completion')
    tasks = record.get('tasks')
    tasks = [t for t in tasks if isinstance(t, dict)] if isinstance(tasks, list) else []
    verdict = recording_verdict(completion)
    if isinstance(completion, dict) and 'basis' in completion:
        return (verdict, []) if verdict is not None else (None, tasks)
    return (None, tasks) if tasks else (verdict, [])


def tasks_format(record) -> bool:
    """Whether a record's completion is in the tasks format (it names its basis, derive_completion)."""
    completion = record.get('completion') if isinstance(record, dict) else None
    return isinstance(completion, dict) and 'basis' in completion


def shown_tasks(record: dict) -> list[dict]:
    """The tasks a reader lists beside the outcome. A record in the tasks format with a verdict of its own lists them
    when there are two or more, or when one is judged against one of the dataset's timed steps (each such judgment
    was asked for and is shown), and always when it was labelled in parts (its tasks joined across the cuts, beside
    its "unclear"); one with no verdict lists them all. A record labelled before keeps the rule it was
    shown by (verdict_or_tasks). Scoring reads verdict_or_tasks, which never lists tasks beside a verdict."""
    record = record if isinstance(record, dict) else {}
    verdict, tasks = verdict_or_tasks(record)
    if not tasks_format(record) or verdict is None:
        return tasks
    every = [t for t in record.get('tasks') or [] if isinstance(t, dict)] if isinstance(record.get('tasks'), list) else []
    # a recording labelled in parts lists every task joined across its cuts beside its "unclear"
    parted = (record.get('completion') or {}).get('basis') == 'parts'
    return every if parted or len(every) >= 2 or any(t.get('judged_against') == 'subtask' for t in every) else []


def severity(item) -> str | None:
    """An issue's severity tier, or None when the issue is unrated (no severity, or one outside SEVERITIES)."""
    value = item.get('severity') if isinstance(item, dict) else None
    value = value.strip().lower() if isinstance(value, str) else None
    return value if value in SEVERITY_RANK else None


def issues(labels: dict, key: str) -> list[dict]:
    """The issues of a list ("data_issues" or "operator_mistakes") that say something: rows with issue text."""
    rows = labels.get(key) if isinstance(labels, dict) else None
    return [item for item in rows if isinstance(item, dict) and isinstance(item.get('issue'), str)
            and item['issue'].strip()] if isinstance(rows, list) else []


def max_severity(items) -> str | None:
    """The highest severity among the rated issues, None when none is rated. An unrated issue is shown and counted as
    unrated; it has no tier, so it neither raises nor lowers this."""
    rated = [severity(item) for item in items if severity(item)]
    return max(rated, key=SEVERITY_RANK.get) if rated else None


def _arm_names(config=None):
    arms = {arm: arm for arm in ('left', 'right', 'both')}
    config = config or {}
    labels = config.get('cam_labels') or []
    if len(config.get('views') or []) == 1 and labels:
        name = str(labels[0])
        arms[name.casefold()] = name
    return arms


def allowed_arms(config=None):
    return set(_arm_names(config).values()) | {'left', 'right', 'both'}
TEXT = {
    'timeline': ('arm', 'action', 'object', 'destination', 'spatial_relation', 'contribution', 'hands_wearing', 'notes'),
    'key_events': ('label', 'kind', 'outcome', 'note'),
    'state_changes': ('object', 'from', 'to', 'seen_in'),
    'completion': ('task_completed', 'success_predicate', 'undone_by', 'reason', 'judged_against', 'basis'),
    'tasks': ('task', 'outcome', 'success_predicate', 'note', 'undone_by', 'judged_against', 'own_outcome',
              'own_success_predicate'),
    'recovery': ('failure', 'correction', 'how_to_fix'),
    'data_issues': ('issue', 'category', 'severity', 'evidence', 'deliberate'),
    'operator_mistakes': ('issue', 'category', 'severity', 'evidence', 'deliberate'),
    'goal_alignment': ('relation', 'reason', 'observed_goal', 'given_goal', 'evidence', 'note'),
    'goal_outcome': ('outcome', 'success_predicate', 'note'),
    'scene': ('setting',), 'scene.objects': ('name', 'location'),
    'contacts': ('id', 'hand', 'object', 'action', 'touch_seen', 'slip', 'note', 'evidence'),
    'contacts_missing': ('hand', 'object', 'action', 'note', 'evidence'),
}
ENUMS = {'task_completed': OUTCOMES, 'outcome': OUTCOMES, 'own_outcome': OUTCOMES,
         'severity': ('low', 'medium', 'high'), 'contribution': ('advancing', 'wasteful', 'idle'),
         'relation': ('aligned', 'narrower', 'broader', 'different', 'unrelated'),
         'deliberate': ('yes', 'no', 'unclear'), 'judged_against': JUDGED_AGAINST, 'basis': BASES}
BOOLS = {'timeline': ('hands_visible',), 'recovery': ('recovered',),
         'operator_mistakes': ('recovered',), 'goal_alignment': ('matches_given',)}
TEXT_LISTS = {'tasks': ('objects',), 'scene.objects': ('attributes',), 'scene_graph': ('relations',)}


def field_identity(field: str, row: int | None = None) -> str:
    if row is None:
        return field
    scope, dot, key = field.rpartition('.')
    return f'{scope}[{row}].{key}' if dot else f'{field}[{row}]'


def source_row_indices(labels: dict, scope: str, dropped: list | None = None) -> list[int]:
    """Derive provenance from actual omitted rows, never a reply-supplied index map."""
    rows = labels
    for key in scope.split('.'):
        rows = rows.get(key) if isinstance(rows, dict) else None
    size = len(rows) if isinstance(rows, list) else 0
    omitted = dropped if dropped is not None else labels.get('_dropped')
    omitted = {item['row'] for item in (omitted if isinstance(omitted, list) else [])
               if isinstance(item, dict) and item.get('field') == scope
               and type(item.get('row')) is int and item['row'] >= 0}
    total = size + len(omitted)
    if any(index >= total for index in omitted):
        return list(range(size))
    return [index for index in range(total) if index not in omitted]


def diagnostics(labels: dict, dropped: list[dict]) -> dict[str, str]:
    """One diagnostic per source field, independent of the layer reporting it."""
    entries = {}
    mappings = {}
    for message in labels.get('_schema_violations') or []:
        row = re.match(r'^(\w+) row (\d+): (?:its )?(\w+)', message)
        field = re.match(r'^([\w.\[\]]+):', message)
        if row:
            if row[1] not in mappings:
                mappings[row[1]] = source_row_indices(labels, row[1], dropped)
            mapping = mappings[row[1]]
            index = int(row[2])
            source_index = mapping[index] if index < len(mapping) else index
            identity = field_identity(row[1] + '.' + row[3], source_index)
            if source_index != int(row[2]):
                message = identity + ':' + message.split(':', 1)[1]
        elif field:
            identity = field[1]
        elif message.startswith('completion.task_completed '):
            identity = 'completion.task_completed'
        else:
            identity = 'diagnostic:' + message
        entries.setdefault(identity, message)
    for item in dropped:
        identity = field_identity(item['field'], item.get('row'))
        value = f" ({item['value']!r})" if 'value' in item else ''
        entries.setdefault(identity, f"{identity}: {item['why']}{value}")
    return entries


def _number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


# the fields every rig's output format asks for, and their types
REQUIRED = {"scene": dict, "timeline": list, "task_summary": str, "key_events": list, "state_changes": list,
            "scene_graph": list, "recovery": list, "instruction_variants": list, "performance_review": str,
            "data_issues": list}
TYPE_WORDS = {dict: 'an object', list: 'a list', str: 'text'}


def schema_rules(labels: dict, config: dict | None, dropped: list[dict]) -> dict[str, str]:
    """Where a projected reply departs from the schema every model was given, beyond the fields project() leaves
    out: a required field missing, a value missing where the schema asks for one (a step's time, arm or contribution,
    an issue's severity, an outcome). Head-camera replies carry tasks instead of one completion. An episode with a
    single camera, named for its actor (FastUMI's one-gripper tasks: the prompt names the camera "gripper" and still
    lists left, right or both), may name that actor: the ambiguity is the prompt's, and counting it would score our
    wording as the model's failure to follow it. One message per source field, keyed as diagnostics() keys them."""
    found = {}
    mappings = {}

    def add(scope, key, row, why):
        if row is not None:
            if scope not in mappings:
                mappings[scope] = source_row_indices(labels, scope, dropped)
            if row < len(mappings[scope]):
                row = mappings[scope][row]
        identity = field_identity(scope + ('.' + key if key else ''), row)
        found.setdefault(identity, f'{identity}: {why}')

    for key, kind in REQUIRED.items():
        if key not in labels:
            add(key, '', None, 'missing')
        elif not isinstance(labels[key], kind):
            add(key, '', None, f'not {TYPE_WORDS[kind]}')
    if 'tasks' in labels:
        if isinstance(labels['tasks'], list):
            for i, task in enumerate(labels['tasks']):
                if not isinstance(task, dict):
                    add('tasks', '', i, 'not an object')
                elif str(task.get('outcome') or '').lower() not in TASK_OUTCOMES:
                    add('tasks', 'outcome', i, f"{task.get('outcome')!r} is not an outcome")
    else:
        c = labels.get('completion')
        if not isinstance(c, dict):
            add('completion', '', None, 'missing')
        elif str(c.get('task_completed') or '').lower() not in OUTCOMES:
            add('completion', 'task_completed', None, f"{c.get('task_completed')!r} is not an outcome")
    arms = allowed_arms(config)
    timeline = labels.get('timeline') if isinstance(labels.get('timeline'), list) else []
    for i, segment in enumerate(timeline):
        if not isinstance(segment, dict):
            add('timeline', '', i, 'not a segment')
            continue
        for key in ('start_s', 'end_s'):
            if not _number(segment.get(key)):
                add('timeline', key, i, 'not a time')
        if _number(segment.get('start_s')) and _number(segment.get('end_s')) and segment['end_s'] < segment['start_s']:
            add('timeline', 'end_s', i, 'ends before it starts')
        if not isinstance(segment.get('arm'), str) or segment['arm'] not in arms:
            add('timeline', 'arm', i, f"{segment.get('arm')!r} is not one of {', '.join(sorted(arms))}")
        if segment.get('contribution') not in ENUMS['contribution']:
            add('timeline', 'contribution', i, f"{segment.get('contribution')!r} is not a contribution")
        if segment.get('progress') is not None and not (_number(segment['progress'])
                                                        and 0 <= segment['progress'] <= 1):
            add('timeline', 'progress', i, f"{segment.get('progress')!r} is not a number from 0 to 1")
    events = labels.get('key_events') if isinstance(labels.get('key_events'), list) else []
    for i, event in enumerate(events):
        if not isinstance(event, dict):
            add('key_events', '', i, 'not an object')
        elif not _number(event.get('t_s')):
            add('key_events', 't_s', i, 'not a time')
        elif str(event.get('outcome') or '').lower() not in EVENT_OUTCOMES:
            add('key_events', 'outcome', i, f"{event.get('outcome')!r} is not an outcome")
    for key in ISSUE_LISTS:
        rows = labels.get(key)
        if rows is None:
            continue
        if not isinstance(rows, list):
            add(key, '', None, 'not a list')
            continue
        for i, item in enumerate(rows):
            if not isinstance(item, dict):
                add(key, '', i, 'not an object')
            elif not str(item.get('issue') or '').strip():
                add(key, 'issue', i, 'no description')
            elif severity(item) is None:
                add(key, 'severity', i, f"{item.get('severity')!r} is not one of {', '.join(SEVERITIES)}")
    return found


def project(labels: dict, config: dict | None = None, *, _part: bool = False) -> tuple[dict, list[dict]]:
    """Keep valid fields, omit unsafe nested values, and report every omission with its source value. A whole reply
    is also checked against the schema (schema_rules); a part's local assessment carries only part of it."""
    labels = copy.deepcopy(labels)
    dropped = []
    # Replies can supply these internal keys too. Validate their shapes before the
    # structural parser appends diagnostics or consumers count them.
    for key in ('_schema_violations', '_dropped'):
        if key not in labels:
            continue
        value = labels[key]
        def valid(item):
            if key == '_schema_violations':
                return isinstance(item, str)
            return (isinstance(item, dict) and isinstance(item.get('field'), str)
                    and isinstance(item.get('why'), str)
                    and ('row' not in item or type(item['row']) is int and item['row'] >= 0))
        if not isinstance(value, list) or any(not valid(item) for item in value):
            dropped.append({'field': key, 'why': 'invalid diagnostic metadata', 'value': copy.deepcopy(value)})
            labels[key] = [item for item in value if valid(item)] if isinstance(value, list) else []
    from label.harness import typed_labels
    labels = typed_labels(labels)
    dropped.extend(labels.pop('_dropped', []))
    supplied_mapping = labels.pop('_schema_row_indices', None)
    mappings = {}
    for scope, rows in labels.items():
        if isinstance(rows, list) and any(item.get('field') == scope and 'row' in item for item in dropped):
            mappings[scope] = source_row_indices(labels, scope, dropped)
    if supplied_mapping is not None and supplied_mapping != mappings:
        dropped.append({'field': '_schema_row_indices', 'why': 'row map does not match omitted source rows',
                        'value': supplied_mapping})
    if mappings:
        labels['_schema_row_indices'] = mappings

    def invalid(row, key, scope, index, why):
        value = row.pop(key)
        dropped.append({'field': scope + '.' + key, **({'row': index} if index is not None else {}),
                        'why': why, 'value': value})

    def row_fields(row, scope, index=None):
        for key in TEXT.get(scope, ()):
            value = row.get(key)
            if value is not None and not isinstance(value, str):
                invalid(row, key, scope, index, 'not text')
            elif key == 'issue' and isinstance(value, str) and not value.strip():
                invalid(row, key, scope, index, 'empty text')
            elif value is not None and (key in ENUMS or key == 'arm'):
                normalized = value.strip().lower()
                allowed = (allowed_arms(config) if key == 'arm' else
                           TASK_OUTCOMES if scope == 'tasks' and key == 'outcome' else
                           EVENT_OUTCOMES if scope == 'key_events' and key == 'outcome' else ENUMS[key])
                if key == 'arm':
                    canonical = _arm_names(config)
                    normalized = canonical.get(value.strip().casefold(), normalized)
                if normalized in allowed:
                    row[key] = normalized
                else:
                    invalid(row, key, scope, index, 'unknown value')
        for key in BOOLS.get(scope, ()):
            if row.get(key) is not None and type(row[key]) is not bool:
                invalid(row, key, scope, index, 'not true or false')
                # Explicit unknown must not become a missing-field inference.
                row[key] = None
        for key in TEXT_LISTS.get(scope, ()):
            value = row.get(key)
            if value is not None and not isinstance(value, list):
                invalid(row, key, scope, index, 'not a list of text')
                row[key] = []
            elif isinstance(value, list):
                if any(not isinstance(item, str) for item in value):
                    dropped.append({'field': scope + '.' + key, **({'row': index} if index is not None else {}),
                                    'why': 'non-text list entries', 'value': copy.deepcopy(value)})
                row[key] = [item for item in value if isinstance(item, str)]
        if scope == 'timeline' and row.get('progress') is not None:
            value = row['progress']
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
                invalid(row, 'progress', scope, index, 'not a finite number from 0 to 1')
        if scope == 'contacts' and isinstance(row.get('seen'), dict):
            row_fields(row['seen'], 'contacts', index)

    for scope in sorted(set(TEXT) | set(TEXT_LISTS) | set(BOOLS)):
        if scope == 'scene.objects':
            rows = (labels.get('scene') or {}).get('objects') or []
        else:
            rows = labels.get(scope)
        if isinstance(rows, dict):
            row_fields(rows, scope)
        elif isinstance(rows, list):
            indices = source_row_indices(labels, scope, dropped)
            for index, row in enumerate(rows):
                if isinstance(row, dict):
                    row_fields(row, scope, indices[index])
    # Local assessments use the same annotation schema and retain their own attribution.
    assessments = labels.get('part_assessments')
    if assessments is not None and not isinstance(assessments, list):
        dropped.append({'field': 'part_assessments', 'why': 'not a list', 'value': assessments})
        labels.pop('part_assessments')
    elif isinstance(assessments, list):
        kept = []
        for index, assessment in enumerate(assessments):
            if not isinstance(assessment, dict):
                dropped.append({'field': 'part_assessments', 'row': index, 'why': 'not an object',
                                'value': assessment})
                continue
            local, omitted = project(assessment, config, _part=True)
            kept.append(local)
            dropped.extend({**item, 'field': f'part_assessments[{index}].' + item['field']} for item in omitted)
        labels['part_assessments'] = kept
    # Nested assessments return their inherited omissions too. Preserve each
    # source omission once when their projection is revisited by another consumer.
    unique = {}
    for item in dropped:
        unique.setdefault(json.dumps(item, sort_keys=True, ensure_ascii=True), item)
    dropped = list(unique.values())
    found = diagnostics(labels, dropped)
    if not _part:
        for identity, message in schema_rules(labels, config, dropped).items():
            found.setdefault(identity, message)
    if found:
        labels['_schema_violations'] = list(found.values())
    if dropped:
        labels['_dropped'] = dropped
    return labels, dropped
