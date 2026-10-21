"""A recording's completion is derived from its tasks (board/label_projection.py recording_completion), and every
reader of an outcome reads the recording's verdict first and its tasks only when there is none (verdict_or_tasks)."""
import copy

from board.label_projection import (LABEL_SCHEMA, derive_completion, project, recording_completion,
                                    recording_verdict, verdict_or_tasks)

TASK = {'start_s': 1.0, 'end_s': 9.0, 'task': 'stack the cups', 'objects': ['cup'], 'judged_against': 'instruction',
        'success_predicate': 'three cups stacked', 'outcome': 'success_then_undone', 'completed_at_s': None,
        'goal_reached_at_s': 7.5, 'undone_at_s': 8.5, 'undone_by': 'the stack is taken apart', 'note': 'reset'}


def test_one_task_is_the_recordings_verdict():
    c = recording_completion([TASK])
    assert c == {'task_completed': 'success_then_undone', 'success_predicate': 'three cups stacked',
                 'completed_at_s': None, 'goal_reached_at_s': 7.5, 'undone_at_s': 8.5,
                 'undone_by': 'the stack is taken apart', 'reason': 'reset', 'judged_against': 'instruction',
                 'from_tasks': 1}


def test_no_task_and_several_tasks_have_no_verdict():
    none = recording_completion([])
    assert none['task_completed'] is None and none['from_tasks'] == 0 and none['reason'] == 'no task observed'
    several = recording_completion([TASK, {**TASK, 'outcome': 'failure'}, {**TASK, 'outcome': 'success'}])
    assert several['from_tasks'] == 3 and several['reason'] == "3 tasks; each task's outcome is under tasks"
    assert all(several[k] is None for k in several if k not in ('from_tasks', 'reason'))
    # a reply with no tasks list, or rows that are not tasks, counts no task
    assert recording_completion(None)['from_tasks'] == 0
    assert recording_completion(['x', 3])['from_tasks'] == 0


def test_one_task_open_at_a_cut_is_no_verdict():
    c = recording_completion([{**TASK, 'outcome': 'continues_past_cut'}])
    assert c['task_completed'] is None and c['from_tasks'] == 1 and 'cut' in c['reason']


def test_a_reply_supplied_completion_is_replaced_and_noted():
    labels = {'tasks': [TASK], 'completion': {'task_completed': 'failure'}}
    before = copy.deepcopy(labels)
    out = derive_completion(labels)
    assert labels == before, 'the reply is not changed in place'
    assert out['completion']['task_completed'] == 'success_then_undone'
    assert any(v.startswith('completion:') for v in out['_schema_violations'])
    clean = derive_completion({'tasks': [TASK, TASK]})
    assert clean['completion']['from_tasks'] == 2 and '_schema_violations' not in clean
    assert LABEL_SCHEMA == 'tasks'


def test_verdict_first_then_tasks():
    # an old episode: a verdict and no tasks
    assert verdict_or_tasks({'completion': {'task_completed': 'Success'}, 'tasks': []}) == ('success', [])
    # a new recording of one task: the verdict wins over the one task
    one = {'tasks': [TASK], 'completion': recording_completion([TASK])}
    assert verdict_or_tasks(one) == ('success_then_undone', [])
    # an old session, an old head-camera clip and a new recording of several tasks: no verdict, the tasks
    session = {'completion': {'task_completed': None, 'reason': 'a session of several tasks'}, 'tasks': [TASK, TASK]}
    assert verdict_or_tasks(session) == (None, [TASK, TASK])
    assert verdict_or_tasks({'tasks': [TASK]}) == (None, [TASK])
    assert verdict_or_tasks({}) == (None, [])
    assert recording_verdict({'task_completed': '  '}) is None and recording_verdict('success') is None


def test_judged_against_is_projected_and_checked():
    labels, dropped = project({'tasks': [{**TASK, 'judged_against': 'Instruction'},
                                         {**TASK, 'judged_against': 'the vibe'}]})
    assert labels['tasks'][0]['judged_against'] == 'instruction'
    assert 'judged_against' not in labels['tasks'][1]
    assert any(d['field'] == 'tasks.judged_against' and d['row'] == 1 for d in dropped)
