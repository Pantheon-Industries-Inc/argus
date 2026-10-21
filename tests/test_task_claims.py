"""The second review of the tasks format (one_schema/review2/REVIEW2.md), one test per decision.

- One law for the dataset's task texts (label/task_claims.py): a text held over the whole recording is a claim about
  the whole recording, never a timed step, whatever its source, read the same way from a context prepared before.
- With an instruction the completion is a robot reply's outcome; its tasks list is not.
- A recording with a verdict still lists its tasks where they carry judgments of their own, and one labelled in parts
  with an instruction is read by its tasks; scores keep the verdict.
- The undone family and the mistake cap read an unasked side task never, and old records as they always did.
- A step's number is its number in the whole recording, in every part of it.
- The head camera records what each task was judged against, and one task with a goal gives the clip its verdict.
"""
import copy
import json
from pathlib import Path

import pytest

from board import build, rules, scores, serve, to_board
from board.families import Families
from board.label_projection import derive_completion, verdict_or_tasks
from compare import metrics
from gate.score import judge
from label import episode as me, harness, pieces
from prepare import formats as f


def claims(ctx):
    from label.task_claims import claims as read      # imported per test, so each one fails on its own before it
    return read(ctx)


def shown_tasks(record):
    from board.label_projection import shown_tasks as shown
    return shown(record)


INSTANTS = [round(1.5 * k, 3) for k in range(7)]


# ---- one law for task texts -----------------------------------------------------------------------------------------

MOLMO = {'instruction': 'Pick up the C, O, R, and L blocks and arrange them in order to spell CORL.',
         'n_state_frames': 120, 'duration_s': 4.0,
         'annotation_subtasks': [{'column': 'task_index', 'source': '/x/meta/tasks.jsonl', 'value': 0,
                                  'row_start': 0, 'row_end': 119, 't0': 0.0, 't1': 3.967, 'label': 'spell out CORL'}]}


def test_a_lerobot_task_held_over_the_whole_episode_is_never_a_timed_step_even_in_an_old_context():
    """Molmo's task_index template ("spell out CORL") differs from its per-episode instruction. Main and the earlier
    branch listed it as a timed step each task was judged against (STEP_RULES); read by the law it is the dataset's
    task for the whole episode, from a context prepared before the law with no new preparation."""
    read = claims(MOLMO)
    assert read['steps'] == [] and [c['text'] for c in read['whole']] == ['spell out CORL']
    assert read['instruction'] == MOLMO['instruction'] and read['instruction_from'] == 'context'
    gelsight = {**MOLMO, 'annotation_subtasks': MOLMO['annotation_subtasks'] + [
        {**MOLMO['annotation_subtasks'][0], 'column': 'language_instruction', 'source': '/x/data.parquet'}]}
    assert claims(gelsight)['steps'] == [] and len(claims(gelsight)['whole']) == 1
    # with no instruction the held text is the instruction
    bare = {**MOLMO, 'instruction': ''}
    assert claims(bare)['instruction'] == 'spell out CORL' and claims(bare)['whole'] == []


def test_a_task_text_that_changes_within_the_recording_gives_numbered_steps():
    ctx = {'duration_s': 10.0, 'annotation_subtasks': [
        {'t0': 0.0, 't1': 4.0, 'label': 'reach'}, {'t0': 4.0, 't1': 4.02, 'label': 'twitch'},
        {'t0': 4.02, 't1': 7.0, 'label': 'carry'}, {'t0': 7.0, 't1': 8.0, 'label': 'carry'},
        {'t0': 8.0, 't1': 10.0, 'label': '  '}]}
    steps = claims(ctx)['steps']
    # a message replaced at once and an empty text claim nothing; repeats of one text are one step
    assert [(s['label'], s['t0'], s['t1'], s['step']) for s in steps] == [('reach', 0.0, 4.0, 1), ('carry', 4.02, 8.0, 2)]
    # a claim that ends in a recording whose length is not stated is never read as held over the whole recording
    assert claims({'annotation_subtasks': [{'t0': 0.5, 't1': 1.5, 'label': 'move'}]})['steps'][0]['label'] == 'move'


@pytest.mark.parametrize('case,instruction,prompts', [
    # a placeholder and the title sent at once: one text over the whole clip
    ('placeholder_then_title_at_once', 'fold the cloth', []),
    # both sent before the footage starts: within the footage only the title holds
    ('both_before_footage', 'fold the cloth', []),
    # the same title sent twice is a repeat, not a change
    ('same_text_twice', 'fold the cloth', []),
    # a new title mid-clip: the task of each stretch
    ('title_then_new_title', None, ['fold the cloth', 'stack the cups']),
])
def test_an_mcap_task_topic_is_read_by_the_law(case, instruction, prompts):
    """The reviewer's probe (review2/probe_mcap.py): only a /task whose text changes within the recording gives timed
    task prompts; one text over the whole clip is its instruction."""
    S, base = 10 ** 9, 100 * 10 ** 9
    texts = {
        'placeholder_then_title_at_once': [(base, 'The agent is idle'), (base + 1000, 'title: "fold the cloth"')],
        'both_before_footage': [(base - 5 * S, 'The agent is idle'), (base - 1 * S, 'title: "fold the cloth"')],
        'same_text_twice': [(base, 'title: "fold the cloth"'), (base + 3 * S, 'title: "fold the cloth"')],
        'title_then_new_title': [(base - 1 * S, 'title: "fold the cloth"'), (base + 6 * S, 'title: "stack the cups"')],
    }[case]
    got, notes, timed = f.mcap_task_texts({'/task': texts}, {'/task': len(texts)}, base, 10.0)
    assert got == instruction and [p['label'] for p in timed] == prompts
    assert '/task' in notes or timed       # every text stays with the notes or the prompts


def test_the_request_and_the_board_read_one_held_text_as_the_task_for_the_whole_episode(tmp_path):
    """A Molmo-shaped episode asks for no step rules; its request names the template as the dataset's task for the whole
    episode, and the board shows it beside the instruction, never among the timed steps."""
    from test_label import _packed_episode
    ep, _ = _packed_episode(tmp_path)
    ctx = json.loads((ep / 'context.json').read_text())
    ctx.update(instruction=MOLMO['instruction'], annotation_subtasks=MOLMO['annotation_subtasks'],
               n_state_frames=MOLMO['n_state_frames'])
    ctx.pop('instruction_note', None)
    (ep / 'context.json').write_text(json.dumps(ctx))
    request = me.build_request(ep, cell_w=128)
    assert "ABOUT THE DATASET'S TIMED STEPS" not in request['content'][0]['text']
    assert request['task_prompts'] == 0 and request['given_prompt'] == MOLMO['instruction']
    assert 'the dataset\'s task for the whole episode (task_index): "spell out CORL"' in request['prompt']
    d = {}
    build.add_context(d, ctx, ep)
    assert d['dataset_task_claims'] == ['spell out CORL'] and 'dataset_labels' not in d


# ---- H1: the completion is the outcome ----------------------------------------------------------------------------------

def _request(instruction='Stack the cups.', steps=0, rig='teleop_arms'):
    return {'annotation_policy': {'version': 1, 'rig': rig}, 'given_prompt': instruction, 'label_schema': 'tasks',
            'config': {'timesteps_s': INSTANTS}, 'task_prompts': steps}


TIMELINE = [{'start_s': 0.0, 'end_s': 9.0, 'arm': 'right', 'action': 'stack', 'contribution': 'advancing',
             'progress': 1.0}]


@pytest.mark.parametrize('tasks', [None, [{'start_s': 0, 'end_s': 9, 'task': 'stack', 'judged_against': 'instruction'}]])
def test_a_valid_completion_is_a_finished_reply_whatever_its_tasks_list(tasks):
    """Main read these replies as complete; the earlier branch threw the verdict away over the tasks list."""
    reply = {'task_summary': 'Stack the cups.', 'timeline': TIMELINE, 'completion': {'task_completed': 'success'},
             **({'tasks': tasks} if tasks is not None else {})}
    labels, ok = harness.read_reply(json.dumps(reply), _request())
    assert ok and '_incomplete' not in labels and labels['completion']['task_completed'] == 'success'
    assert [v for v in labels['_schema_violations'] if v.startswith('tasks:')]
    record = {**_request(), 'finish_reason': 'stop', 'parse_ok': True, 'labels': labels,
              'response_text': json.dumps(reply)}
    assert harness.annotation_problems(record) == []
    # the completion itself stays required
    labels, ok = harness.read_reply(json.dumps({k: v for k, v in reply.items() if k != 'completion'}), _request())
    assert not ok


# ---- M4: verdicts and task lists ------------------------------------------------------------------------------------------

def _board(labels, instruction='Stack the cups.'):
    """The board's record of a reply; one whose completion says it was labelled in parts is stitched, as
    label/pieces.py stitch writes it, and complete when every part is."""
    parted = (labels.get('completion') or {}).get('basis') == 'parts'
    return to_board.convert({'episode_dir': 'episode_x', 'parse_ok': True, 'labels': labels,
                             'given_prompt': instruction, 'label_schema': 'tasks',
                             **({'stitched': {'parts': 3, 'cuts_s': [3.0, 6.0]}} if parted else {})}, 'demo')


def _task(name, outcome, against, a, b, **kw):
    return {'start_s': a, 'end_s': b, 'task': name, 'objects': ['cup'], 'judged_against': against,
            'outcome': outcome, **kw}


def test_a_verdict_keeps_the_tasks_that_carry_their_own_judgments_on_the_rail_and_in_compare():
    """Galaxea, HABIT and robot MCAP steps: each task judged against its step is shown beside the recording's verdict.
    One asked task alone adds nothing; scores keep the verdict's numbers."""
    both = derive_completion({'timeline': TIMELINE, 'completion': {'task_completed': 'failure'}, 'tasks': [
        _task('pick up the cup', 'success', 'subtask', 0, 4, subtask=1),
        _task('place it on the stack', 'failure', 'subtask', 4, 9, subtask=2)]}, rig='teleop_arms', instruction=True,
        task_prompts=2)
    d = _board(both)
    assert verdict_or_tasks(d) == ('failure', []) and len(shown_tasks(d)) == 2
    rail = serve._rail_record(Path('e.json'), d)
    assert (rail['task_completed'], rail['n_tasks'], rail['n_task_success'], rail.get('verdict_first')) == (
        'failure', 2, 1, True)
    m = metrics.measure(both, 'demo')
    assert (m['outcome'], m['tasks'], m.get('verdict_first')) == ('failure', [2, 1], True)
    one = derive_completion({'timeline': TIMELINE, 'completion': {'task_completed': 'success'},
                             'tasks': [_task('stack the cups', 'success', 'instruction', 0, 9)]},
                            rig='teleop_arms', instruction=True)
    assert shown_tasks(_board(one)) == [] and serve._rail_record(Path('e.json'), _board(one))['n_tasks'] == 0
    f = scores.facts(d)
    assert f['tasks'] == [] and f['outcome'] == 'failure'


def test_a_long_recording_with_an_instruction_is_read_by_its_tasks_as_main_read_a_session():
    """The review's probe_cycles: four cycles of a teleop session labelled in parts, three succeeded. Main's session
    record showed "3 of 4 tasks"; a verdict of "unclear" only because no part sees the whole recording is no
    judgment, so the stitched record has none and shows its tasks, in scores, rail and compare."""
    labels = {'timeline': TIMELINE, 'tasks': [_task(f'cycle {i}', o, 'instruction', 2 * i, 2 * i + 2)
                                             for i, o in enumerate(['success', 'success', 'failure', 'success'])],
              'completion': {'task_completed': None, 'basis': 'parts', 'reason': 'Labelled in 3 requests'}}
    d = _board(labels)
    assert verdict_or_tasks(d)[0] is None and len(shown_tasks(d)) == 4
    rail = serve._rail_record(Path('e.json'), d)
    assert (rail['n_tasks'], rail['n_task_success']) == (4, 3) and 'verdict_first' not in rail
    assert metrics.measure(labels, 'demo')['tasks'] == [4, 3]
    _, why = scores._outcome(scores.facts(d))
    assert (why['tasks_total'], why['tasks_done']) == (4, 3)


# ---- M5 and L11: families, caps and the gate ------------------------------------------------------------------------------

def test_an_unasked_undone_side_task_never_fires_the_undone_family():
    side = derive_completion({'timeline': TIMELINE, 'completion': {'task_completed': 'success'}, 'tasks': [
        _task('stack the cups', 'success', 'instruction', 0, 6),
        _task('stack the blocks', 'success_then_undone', 'own_end_state', 7, 9)]}, rig='teleop_arms', instruction=True)
    assert 'undone' not in Families().classify(_board(side))['counted']
    asked = {'timeline': TIMELINE, 'completion': {'task_completed': None, 'basis': 'parts'}, 'tasks': [
        _task('stack the cups', 'success_then_undone', 'instruction', 0, 6)]}
    assert 'undone' in Families().classify(_board(asked))['counted']


def test_old_records_keep_mains_gate_and_mistake_cap():
    """The review's probe_gate_old: an old session label (no basis) passes a not_outcome case and an old head-camera
    record's undone task caps no mistake, as on main."""
    old = {'completion': {'task_completed': None},
           'tasks': [{'task': 'sort', 'outcome': 'failure'}, {'task': 'stack', 'outcome': 'success'}]}
    assert judge({'not_outcome': ['success', 'success_then_undone']}, old) == (True, [])
    ego = {'tasks': [{'task': 'fold', 'outcome': 'success_then_undone'}, {'task': 'wipe', 'outcome': 'success'}],
           'operator_mistakes': [{'issue': 'left unfinished', 'category': 'incomplete_task', 'severity': 'medium'}]}
    d = copy.deepcopy(ego)
    build.apply_rules(d, {}, rules.GENERAL)
    assert [m['severity'] for m in d['operator_mistakes']] == ['medium']
    # the same shapes in the tasks format follow the new rules
    new = {**old, 'completion': {'task_completed': None, 'basis': 'tasks'}}
    assert judge({'not_outcome': ['success']}, new)[0] is False
    d = {**copy.deepcopy(ego), 'completion': {'task_completed': None, 'basis': 'tasks'}}
    build.apply_rules(d, {}, rules.GENERAL)
    assert [m['severity'] for m in d['operator_mistakes']] == ['low']


# ---- M6: a step's number in the whole recording ---------------------------------------------------------------------------

def test_a_part_numbers_its_steps_as_the_whole_recording_does_and_the_stitch_keeps_them(tmp_path, monkeypatch):
    from test_formats import recorder_folder
    from test_pieces import _part_result
    from board.label_projection import recording_completion
    monkeypatch.setattr(pieces, 'piece_max', lambda ctx: 1.0)
    recorder_folder(tmp_path / 'upload', n=90, duplicate_stamps=False)
    report = f.convert(tmp_path / 'upload', 'teleop_arms', tmp_path / 'eps', 'test', 900)
    src = tmp_path / 'eps' / report['episodes'][0]['episode_id']
    ctx = json.loads((src / 'context.json').read_text())
    ctx['instruction'] = ''
    ctx['annotation_subtasks'] = [{'t0': 0.0, 't1': 0.9, 'label': 'reach'}, {'t0': 0.9, 't1': 2.1, 'label': 'carry'},
                                  {'t0': 2.1, 't1': 3.0, 'label': 'place'}]
    (src / 'context.json').write_text(json.dumps(ctx))
    parts = pieces.write_pieces(src, tmp_path / 'job' / 'pieces')
    assert len(parts) == 3
    contexts = [json.loads((p / 'context.json').read_text()) for p in parts]
    block = me.robot_annotation_block(contexts[2])
    # the last part shows "carry" and "place" under their numbers in the whole recording, never as 1 and 2
    assert '  2. ' in block and '  3. ' in block and '  1. ' not in block
    records = []
    for p, c in zip(parts, contexts):
        own = {row['label']: row['step'] for row in claims(c)['steps']}
        tasks = [{'start_s': 0.0, 'end_s': round(c['piece']['t1_s'] - c['piece']['t0_s'], 3), 'task': 'move it',
                  'objects': ['block'], 'judged_against': 'subtask', 'subtask': own.get('place', own.get('carry')),
                  'outcome': 'success', 'note': ''}]
        record = _part_result(p, 'move it')
        record['label_schema'] = 'tasks'
        record['labels'].update(tasks=tasks, completion=recording_completion(tasks, task_prompts=len(own)))
        records.append(record)
    stitched = pieces.stitch(src, list(zip(contexts, records)))
    assert [t['subtask'] for t in stitched['labels']['tasks']] == [2, 2, 3]
    d = {}
    build.add_context(d, ctx, src)
    assert [(x['label'], x['step']) for x in d['dataset_labels']] == [('reach', 1), ('carry', 2), ('place', 3)]


# ---- M9: the head camera ---------------------------------------------------------------------------------------------------

def test_head_camera_tasks_record_what_they_were_judged_against_and_one_task_with_a_goal_gives_the_verdict():
    from board.label_projection import judge_head_camera_tasks
    one = derive_completion({'tasks': [{'start_s': 0, 'end_s': 9, 'task': 'dry the mug', 'outcome': 'success'}]},
                            rig='ego_head', instruction=True, task_prompts=2)
    assert one['completion']['task_completed'] == 'success' and one['completion']['basis'] == 'one_task'
    several = derive_completion({'tasks': [{'task': 'a', 'outcome': 'success'}, {'task': 'b', 'outcome': 'success'}]},
                                rig='ego_head', instruction=True)
    assert 'completion' not in several
    record = {'tasks': [{'start_s': 0.0, 'end_s': 4.0, 'task': 'pick up the towel'},
                        {'start_s': 4.0, 'end_s': 9.0, 'task': 'wipe the counter'}]}
    judge_head_camera_tasks(record, [{'t0': 0.0, 't1': 4.5, 'label': 'pick up the towel', 'step': 1}], 'dry the mug')
    assert [(t['judged_against'], t['subtask']) for t in record['tasks']] == [('subtask', 1), ('instruction', None)]
    bare = {'tasks': [{'start_s': 0.0, 'end_s': 4.0, 'task': 'wipe'}]}
    judge_head_camera_tasks(bare, [], None)
    assert bare['tasks'][0]['judged_against'] == 'own_end_state' and bare['tasks'][0]['judged_against_derived']
