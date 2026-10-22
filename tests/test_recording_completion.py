"""A recording's completion in the tasks format follows what its dataset gives (board/label_projection.py
derive_completion): the reply's own completion against an instruction, the one task's verdict, or none. Every reader of
an outcome reads the recording's verdict first and its tasks only when there is none (verdict_or_tasks)."""
import copy
from pathlib import Path

from board import scores, serve, static
from board.label_projection import (LABEL_SCHEMA, derive_completion, project, recording_completion,
                                    recording_verdict, verdict_or_tasks)
from compare import metrics

TASK = {'start_s': 1.0, 'end_s': 9.0, 'task': 'stack the cups', 'objects': ['cup'], 'judged_against': 'instruction',
        'success_predicate': 'three cups stacked', 'outcome': 'success_then_undone', 'completed_at_s': None,
        'goal_reached_at_s': 7.5, 'undone_at_s': 8.5, 'undone_by': 'the stack is taken apart', 'note': 'reset'}
# the same task in a request without an instruction, which can be judged only against its own end state
OWN = {**TASK, 'judged_against': 'own_end_state'}


def test_one_task_is_the_recordings_verdict():
    c = recording_completion([TASK])
    assert c == {'task_completed': 'success_then_undone', 'success_predicate': 'three cups stacked',
                 'completed_at_s': None, 'goal_reached_at_s': 7.5, 'undone_at_s': 8.5,
                 'undone_by': 'the stack is taken apart', 'reason': 'reset', 'judged_against': 'instruction',
                 'from_tasks': 1, 'basis': 'one_task'}
    # an outcome that is no outcome word is no verdict, so the task list is still shown
    odd = recording_completion([{**TASK, 'outcome': 'done'}])
    assert odd['task_completed'] is None and odd['basis'] == 'tasks' and "'done'" in odd['reason']
    assert verdict_or_tasks({'tasks': [TASK], 'completion': odd})[0] is None


def test_no_task_and_several_tasks_have_no_verdict():
    none = recording_completion([])
    assert none['task_completed'] is None and none['from_tasks'] == 0 and none['reason'] == 'no task observed'
    several = recording_completion([TASK, {**TASK, 'outcome': 'failure'}, {**TASK, 'outcome': 'success'}])
    assert several['from_tasks'] == 3 and several['reason'] == "3 tasks; each task's outcome is under tasks"
    assert all(several[k] is None for k in several if k not in ('from_tasks', 'reason', 'basis'))
    # the dataset's own timed steps and no instruction: each task against its step, and no recording verdict
    steps = recording_completion([TASK], task_prompts=2)
    assert steps['task_completed'] is None and steps['basis'] == 'task_prompts' and steps['from_tasks'] == 1
    # a reply with no tasks list, or rows that are not tasks, counts no task
    assert recording_completion(None)['from_tasks'] == 0
    assert recording_completion(['x', 3])['from_tasks'] == 0


def test_one_task_open_at_a_cut_is_no_verdict():
    c = recording_completion([{**TASK, 'outcome': 'continues_past_cut'}])
    assert c['task_completed'] is None and c['from_tasks'] == 1 and 'cut' in c['reason']


def test_a_reply_supplied_completion_is_replaced_and_noted_where_none_was_asked():
    robot = dict(rig='teleop_arms', instruction=False)
    labels = {'tasks': [OWN], 'completion': {'task_completed': 'failure'}}
    before = copy.deepcopy(labels)
    out = derive_completion(labels, **robot)
    assert labels == before, 'the reply is not changed in place'
    assert out['completion']['task_completed'] == 'success_then_undone'
    assert any(v.startswith('completion:') for v in out['_schema_violations'])
    clean = derive_completion({'tasks': [OWN, OWN]}, **robot)
    assert clean['completion']['from_tasks'] == 2 and '_schema_violations' not in clean
    # labels read again keep their derived completion and gain no note
    assert derive_completion(clean, **robot) == clean
    assert derive_completion(out, **robot)['_schema_violations'] == out['_schema_violations']
    assert LABEL_SCHEMA == 'tasks'


def test_with_an_instruction_the_recording_is_judged_by_its_own_completion_never_by_its_tasks():
    """A robot recording with an instruction asks for the completion itself, as the episode prompt did: an asked task
    that fails followed by an unasked task that succeeds is a failure, never "no verdict"."""
    asked = {**TASK, 'outcome': 'failure', 'judged_against': 'instruction'}
    extra = {**TASK, 'outcome': 'success', 'judged_against': 'own_end_state'}
    labels = {'tasks': [asked, extra], 'completion': {'task_completed': 'failure', 'reason': 'the row is never made'}}
    out = derive_completion(labels, rig='handheld_gripper', instruction=True, task_prompts=3)
    assert out['completion'] == {**labels['completion'], 'basis': 'instruction'} and '_schema_violations' not in out
    assert verdict_or_tasks(out) == ('failure', [])


def test_the_head_camera_is_read_as_it_always_was_except_one_task():
    """The head camera's schema has no completion: one task (and no timed steps) is the clip's verdict, and any other
    clip keeps the reply as it is, with no completion added, so it is read task by task exactly as before."""
    one = derive_completion({'tasks': [TASK]}, rig='ego_head', instruction=True)
    assert one['completion']['task_completed'] == 'success_then_undone'
    for tasks, prompts in (([OWN, OWN], 0), ([], 0), ([{**TASK, 'judged_against': 'subtask', 'subtask': 1}], 4)):
        labels = {'tasks': tasks}
        assert derive_completion(labels, rig='ego_head', instruction=False, task_prompts=prompts) == labels


def test_verdict_first_then_tasks():
    # an old episode: a verdict and no tasks
    assert verdict_or_tasks({'completion': {'task_completed': 'Success'}, 'tasks': []}) == ('success', [])
    # a new recording of one task: the verdict wins over the one task
    one = {'tasks': [TASK], 'completion': recording_completion([TASK])}
    assert verdict_or_tasks(one) == ('success_then_undone', [])
    # a new recording of several tasks, and of none: no verdict, the tasks
    several = {'tasks': [TASK, TASK], 'completion': recording_completion([TASK, TASK])}
    assert verdict_or_tasks(several) == (None, [TASK, TASK])
    assert verdict_or_tasks({'tasks': [], 'completion': recording_completion([])}) == (None, [])
    # an old session and an old head-camera clip: no verdict, the tasks
    session = {'completion': {'task_completed': None, 'reason': 'a session of several tasks'}, 'tasks': [TASK, TASK]}
    assert verdict_or_tasks(session) == (None, [TASK, TASK])
    assert verdict_or_tasks({'tasks': [TASK]}) == (None, [TASK])
    assert verdict_or_tasks({}) == (None, [])
    # a record labelled before the tasks format with a verdict and tasks (an older stitched session whose completion
    # says unclear, a hand-written head-camera fixture) keeps the tasks it was shown and scored by
    old = {'completion': {'task_completed': 'unclear', 'reason': 'Labelled in 2 requests'}, 'tasks': [TASK, TASK]}
    assert verdict_or_tasks(old) == (None, [TASK, TASK])
    assert recording_verdict({'task_completed': '  '}) is None and recording_verdict('success') is None


def test_judged_against_is_projected_and_checked():
    labels, dropped = project({'tasks': [{**TASK, 'judged_against': 'Instruction'},
                                         {**TASK, 'judged_against': 'the vibe'}]})
    assert labels['tasks'][0]['judged_against'] == 'instruction'
    assert 'judged_against' not in labels['tasks'][1]
    assert any(d['field'] == 'tasks.judged_against' and d['row'] == 1 for d in dropped)


# every reader of an outcome: a recording of one task reads as the episode it is, one of several task by task

STEPS = [(0, 5, 'advancing', 0.5), (5, 10, 'advancing', 1.0)]


def _episode(goal_s=10.0):
    """An episode as the board stores it: one verdict, no tasks."""
    return {'dataset': 'ds', 'episode_prompt': 'stack the cups', 'duration_s': 10,
            'event_labels': [{'t_s': a, 'end_s': b, 'contribution': c, 'progress': p} for a, b, c, p in STEPS],
            'key_events': [{'t_s': 5, 'kind': 'subgoal_complete', 'outcome': 'success', 'label': 'first cup on'}],
            'completion': {'task_completed': 'success', 'completed_at_s': goal_s, 'success_predicate': 'done',
                           'goal_reached_at_s': goal_s},
            'tasks': []}


def _one_task(goal_s=10.0):
    """The same recording labelled in the tasks format: one task, whose verdict is the recording's."""
    task = {'start_s': 0, 'end_s': 10, 'task': 'stack the cups', 'outcome': 'success', 'completed_at_s': goal_s,
            'goal_reached_at_s': goal_s, 'success_predicate': 'done', 'judged_against': 'instruction'}
    return {**_episode(goal_s), 'completion': recording_completion([task]), 'tasks': [task]}


def test_one_task_scores_exactly_as_the_episode_it_is():
    peers = {f'p{i}.json': _episode(10.0 + i) for i in range(6)}
    old = scores.score_board({**peers, 'r.json': _episode()})['r.json']
    new = scores.score_board({**peers, 'r.json': _one_task()})['r.json']
    assert new == old
    assert new['efficiency'] is not None and new['subgoals'][-1]['goal'] and new['subgoals'][-1]['label'] == 'done'


def test_several_tasks_score_task_by_task():
    rec = _one_task()
    tasks = [{**rec['tasks'][0], 'end_s': 5, 'completed_at_s': 5},
             {**rec['tasks'][0], 'start_s': 5, 'outcome': 'failure', 'completed_at_s': None}]
    rec.update(tasks=tasks, completion=recording_completion(tasks))
    s = scores.score_board({'r.json': rec})['r.json']
    assert s['inputs']['tasks_total'] == 2 and s['inputs']['tasks_done'] == 1 and s['efficiency'] is None


def test_goal_frames_rail_and_compare_read_the_verdict_first():
    one = _one_task()
    assert static.goal_times(one) == [10.0]
    rail = serve._rail_record(Path('episode_x.json'), one)
    assert rail['task_completed'] == 'success' and rail['n_tasks'] == 0
    assert metrics.measure(one, 'ds')['outcome'] == 'success' and 'tasks' not in metrics.measure(one, 'ds')
    # several tasks: the frame each task is done at, the only task goal frame the page shows (none for a task undone
    # later), and the share done
    tasks = [{**one['tasks'][0], 'end_s': 5, 'completed_at_s': 4.0},
             {**one['tasks'][0], 'start_s': 5, 'outcome': 'success_then_undone', 'completed_at_s': None,
              'goal_reached_at_s': 8.0, 'undone_at_s': 9.0}]
    several = {**one, 'tasks': tasks, 'completion': recording_completion(tasks)}
    assert static.goal_times(several) == [4.0]
    rail = serve._rail_record(Path('episode_x.json'), several)
    assert rail['task_completed'] is None and (rail['n_tasks'], rail['n_task_success']) == (2, 1)
    m = metrics.measure(several, 'ds')
    assert m['outcome'] is None and m['tasks'] == [2, 1]


# A real stitched UMI 900h take as main writes it (fob_f4e9d4-ffd66b8e-2845-2, cut at 403.603s,
# audit_2026-10-21/long_recordings.md F1): one tower built across the cut, whose first half is still under way at it.
TOWER_OPEN = {'start_s': 6.01, 'end_s': 403.57, 'task': 'Build a freestanding tower from the loose wooden blocks using '
              'alternating three-block layers.', 'objects': ['wooden blocks'], 'judged_against': 'own_end_state',
              'outcome': 'continues_past_cut', 'outcome_in_part': 'unclear',
              'note': 'A substantial, standing tower is built, but the demonstrator is still adding and aligning blocks '
                      'at the internal preparation cut.'}
TOWER_DONE = {'start_s': 403.603, 'end_s': 647.843, 'task': 'Finish stacking the remaining wooden blocks into the '
              'alternating-layer tower.', 'objects': ['wooden blocks', 'wooden block tower'],
              'judged_against': 'own_end_state', 'outcome': 'success', 'completed_at_s': 647.0,
              'note': 'This is the continuation of a tower-building activity begun before the supplied window.'}


def test_a_task_open_at_a_cut_is_neither_done_nor_not_done_on_the_rail_in_compare_and_in_the_gate():
    """F4: the rail, compare and the gate counted a continues_past_cut task in the total and never as done, so 595 of
    1,279 stitched UMI takes (46.5%) read amber "1/2 tasks" only because of the open first half of a task that succeeded
    in the next part. Its outcome is unknown: it is in neither count, and each reader says how many it left out."""
    from gate.score import judge
    tasks = [TOWER_OPEN, TOWER_DONE]
    take = {**_episode(), 'episode_prompt': '', 'duration_s': 647.843, 'tasks': tasks,
            'completion': {**recording_completion(tasks), 'basis': 'tasks'}}
    rail = serve._rail_record(Path('fob_f4e9d4-ffd66b8e-2845-2.json'), take)
    assert (rail['n_tasks'], rail['n_task_success'], rail['n_tasks_open']) == (1, 1, 1)
    m = metrics.measure(take, 'umi')
    assert m['tasks'] == [1, 1] and m['tasks_open'] == 1
    assert judge({'outcome': ['tasks:success']}, take)[0]
    # an open half beside a real failure still reads as the failure, and a take of open halves alone has no known
    # outcome, so a fact about what it must not be stays unmet
    failed = [TOWER_OPEN, {**TOWER_DONE, 'outcome': 'failure', 'completed_at_s': None}]
    assert judge({'outcome': ['tasks:failure']}, {**take, 'tasks': failed})[0]
    only_open = {**take, 'tasks': [TOWER_OPEN, {**TOWER_OPEN, 'start_s': 403.603, 'end_s': 647.843}]}
    rail = serve._rail_record(Path('x.json'), only_open)
    assert (rail['n_tasks'], rail['n_task_success'], rail['n_tasks_open']) == (0, 0, 2)
    assert 'tasks' not in metrics.measure(only_open, 'umi') and metrics.measure(only_open, 'umi')['tasks_open'] == 2
    assert not judge({'not_outcome': ['failure']}, only_open)[0]
    # with no task open, nothing changes and no count of open tasks is written
    done = {**take, 'tasks': [TOWER_DONE]}
    assert 'n_tasks_open' not in serve._rail_record(Path('x.json'), {**done, 'tasks': [TOWER_DONE, TOWER_DONE]})
    assert 'tasks_open' not in metrics.measure({**done, 'tasks': [TOWER_DONE, TOWER_DONE]}, 'umi')
