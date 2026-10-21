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
    rows = [{'t0': 0.0, 't1': 4.0, 'label': 'reach'}, {'t0': 4.0, 't1': 4.02, 'label': 'twitch'},
            {'t0': 4.02, 't1': 7.0, 'label': 'carry'}, {'t0': 7.0, 't1': 8.0, 'label': 'carry'},
            {'t0': 8.0, 't1': 10.0, 'label': '  '}]
    # a message source (an MCAP topic): the text replaced at once claims nothing, the touching repeats join, and the
    # empty text claims nothing
    topic = claims({'duration_s': 10.0, 'annotation_subtasks': [{**r, 'column': '/task/subtask'} for r in rows]})
    assert [(s['label'], s['t0'], s['t1'], s['step']) for s in topic['steps']] == [
        ('reach', 0.0, 4.0, 1), ('carry', 4.02, 8.0, 2)]
    # an annotation of spans: every span is kept however short, and touching repeats join
    spans = claims({'duration_s': 10.0, 'annotation_subtasks': rows})
    assert [(s['label'], s['t0'], s['t1']) for s in spans['steps']] == [
        ('reach', 0.0, 4.0), ('twitch', 4.0, 4.02), ('carry', 4.02, 8.0)]
    # a claim that ends in a recording whose length is not stated is never read as held over the whole recording
    assert claims({'annotation_subtasks': [{'t0': 0.5, 't1': 1.5, 'label': 'move'}]})['steps'][0]['label'] == 'move'


D = {'duration_s': 100.0}


def _read(rows, **ctx):
    got = claims({**D, **ctx, 'annotation_subtasks': rows})
    return got['instruction'], [(s.get('t0'), s.get('t1'), s['label']) for s in got['steps']]


def test_moments_and_short_spans_of_an_annotation_are_kept_as_given():
    """N2, the reviewer's probe_claims: three point annotations (t0 == t1) and two 0.05s taps are sent as main sent
    them; the 0.1s rule reads only a message source's text replaced by the next message."""
    moments = [{'t0': 10, 't1': 10, 'label': 'grasp'}, {'t0': 40, 't1': 40, 'label': 'fold'},
               {'t0': 80, 't1': 80, 'label': 'release'}]
    assert _read(moments, instruction='fold the towel') == ('fold the towel', [
        (10, 10, 'grasp'), (40, 40, 'fold'), (80, 80, 'release')])
    taps = [{'t0': 10, 't1': 10.05, 'label': 'tap'}, {'t0': 20, 't1': 20.05, 'label': 'tap again'},
            {'t0': 30, 't1': 40, 'label': 'press'}]
    assert [label for _, _, label in _read(taps, instruction='x')[1]] == ['tap', 'tap again', 'press']
    # identical moments never join
    assert len(_read([{'t0': 5, 't1': 5, 'label': 'tap'}, {'t0': 5, 't1': 5, 'label': 'tap'}])[1]) == 2


@pytest.mark.parametrize('row', [{'t0': 0.5, 'label': 'pick up the cup'},
                                 {'t0': 0.5, 't1': None, 'label': 'pick up the cup'}])
def test_a_single_moment_is_never_the_recordings_instruction(row):
    """N3: a moment with no end is a moment, never a claim held over the whole recording."""
    assert claims({'annotation_subtasks': [row]})['instruction'] is None
    assert claims({**D, 'annotation_subtasks': [row]})['steps'][0]['label'] == 'pick up the cup'


@pytest.mark.parametrize('column,rows', [
    ('subtask', [{'t0': 0, 't1': 50, 'label': '', 'row_start': 0, 'row_end': 49},
                 {'t0': 50, 't1': 100, 'label': 'place the cup', 'row_start': 50, 'row_end': 99}]),
    ('/task', [{'t0': 0, 't1': 50, 'label': ''}, {'t0': 50, 't1': 100, 'label': 'place the cup'}]),
])
def test_a_text_after_an_empty_first_half_covers_only_its_half(column, rows):
    """N4: an empty text claims nothing and is covered by nothing; the text after it is a step."""
    instruction, steps = _read([{**r, 'column': column} for r in rows], n_state_frames=100)
    assert instruction is None and [label for _, _, label in steps] == ['place the cup']


def test_a_text_that_returns_after_a_gap_is_two_steps():
    """N5: identical texts join only where their spans touch."""
    gap = [{'t0': 0, 't1': 30, 'label': 'A'}, {'t0': 70, 't1': 100, 'label': 'A'}]
    assert _read(gap) == (None, [(0, 30, 'A'), (70, 100, 'A')])
    assert _read([{**gap[0], 'column': '/task'}, {**gap[1], 'column': '/task'}]) == (None, [(0, 30, 'A'), (70, 100, 'A')])
    touching = [{'t0': 0, 't1': 30, 'label': 'A'}, {'t0': 30, 't1': 100, 'label': 'A'}]
    assert _read(touching) == ('A', [])


@pytest.mark.parametrize('t0,t1,whole', [(0, 100, True), (0.9, 99.1, True), (0, 95, False), (4.9, 100, False),
                                         (0, 98.9, False)])
def test_a_whole_recording_claim_reaches_within_a_second_of_both_ends(t0, t1, whole):
    """N7: no share of the length is forgiven; a claim leaving more than 1s at either end is a step."""
    assert (_read([{'t0': t0, 't1': t1, 'label': 'wipe the table'}])[0] == 'wipe the table') is whole


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


@pytest.mark.parametrize('tasks', [None, []])
def test_with_timed_steps_a_reply_judging_no_step_names_what_it_lacks(tasks):
    """N6: with the dataset's timed steps the tasks hold the judgment against each step. A valid completion keeps the
    reply complete (H1), and a missing or empty tasks list is named in _schema_violations."""
    reply = {'task_summary': 'Stack the cups.', 'timeline': TIMELINE, 'completion': {'task_completed': 'success'},
             **({'tasks': tasks} if tasks is not None else {})}
    labels, ok = harness.read_reply(json.dumps(reply), _request(steps=2))
    assert ok and [v for v in labels['_schema_violations'] if v.startswith('tasks:')]


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


PARTS_SHAPES = {
    # the review's probe_parts: an asked task fails, then an unasked tower succeeds; an asked task undone by a reset;
    # one asked task that succeeds; and four cycles, three of them done
    'asked_fails_then_unasked_tower': [_task('stack the cups', 'failure', 'instruction', 0, 4),
                                       _task('build a tower', 'success', 'own_end_state', 4, 9, completed_at_s=8.5)],
    'asked_success_then_undone_by_reset': [_task('stack the cups', 'success_then_undone', 'instruction', 0, 8,
                                                 goal_reached_at_s=6.0, undone_at_s=7.5, undone_by='reset')],
    'one_asked_success': [_task('stack the cups', 'success', 'instruction', 0, 9, completed_at_s=8.0)],
    'four_cycles': [_task(f'cycle {i}', o, 'instruction', 2 * i, 2 * i + 2)
                    for i, o in enumerate(['success', 'success', 'failure', 'success'])],
}
MISTAKE = {'t_s': 5.0, 'issue': 'left the third cup unplaced', 'category': 'incomplete_task', 'severity': 'medium'}


@pytest.mark.parametrize('shape', sorted(PARTS_SHAPES))
def test_a_long_recording_with_an_instruction_keeps_mains_unclear_and_lists_its_tasks_beside_it(shape):
    """N1: main stitches an instructed recording labelled in parts to "unclear" with no tasks. The branch keeps that
    verdict, so scores, gate, caps, families, rail and compare read it as main does, and lists the joined tasks beside
    it."""
    base = {'timeline': TIMELINE, 'operator_mistakes': [MISTAKE], 'goal_alignment': {'relation': 'aligned'}}
    main = {**base, 'completion': {'task_completed': 'unclear', 'reason': 'Labelled in 2 requests'}, 'tasks': []}
    ours = {**base, 'tasks': PARTS_SHAPES[shape],
            'completion': {'task_completed': 'unclear', 'reason': 'Labelled in 2 requests', 'basis': 'parts'}}
    d_main = to_board.convert({'episode_dir': 'x', 'parse_ok': True, 'labels': main, 'given_prompt': 'Stack the cups.',
                               'stitched': {'parts': 2}}, 'demo')
    d_ours = _board(ours)
    for d in (d_main, d_ours):
        d['duration_s'] = 9.0
    score = lambda d: scores.score_board({'e.json': {k: d.get(k) for k in scores.FIELDS}})['e.json']
    assert score(d_main)['outcome'] == score(d_ours)['outcome']
    assert score(d_main)['inputs'].get('outcome') == score(d_ours)['inputs'].get('outcome') == 'unclear'
    for case in ({'outcome': ['unclear']}, {'not_outcome': ['success', 'success_then_undone']}):
        assert judge(case, main) == judge(case, ours)
    caps = []
    for d in (d_main, d_ours):
        capped = copy.deepcopy(d)
        build.apply_rules(capped, {}, rules.GENERAL)
        caps.append([(m['category'], m['severity']) for m in capped['operator_mistakes']])
    assert caps[0] == caps[1]
    assert sorted(Families().classify(d_main)['counted']) == sorted(Families().classify(d_ours)['counted'])
    assert metrics.measure(main, 'demo')['outcome'] == metrics.measure(ours, 'demo')['outcome'] == 'unclear'
    rail = serve._rail_record(Path('e.json'), d_ours)
    assert rail['task_completed'] == serve._rail_record(Path('e.json'), d_main)['task_completed'] == 'unclear'
    # and beside the verdict, every joined task
    assert len(shown_tasks(d_ours)) == len(PARTS_SHAPES[shape]) == rail['n_tasks']
    assert metrics.measure(ours, 'demo')['tasks'][0] == len(PARTS_SHAPES[shape])


# ---- M5 and L11: families, caps and the gate ------------------------------------------------------------------------------

def test_an_unasked_undone_side_task_never_fires_the_undone_family():
    side = derive_completion({'timeline': TIMELINE, 'completion': {'task_completed': 'success'}, 'tasks': [
        _task('stack the cups', 'success', 'instruction', 0, 6),
        _task('stack the blocks', 'success_then_undone', 'own_end_state', 7, 9)]}, rig='teleop_arms', instruction=True)
    assert 'undone' not in Families().classify(_board(side))['counted']
    # a record in the tasks format with no verdict: the task judged against the instruction stands for it
    asked = {'completion': {'task_completed': None, 'basis': 'tasks'}, 'tasks': [
        _task('stack the cups', 'success_then_undone', 'instruction', 0, 6)]}
    assert 'undone' in Families().classify(asked)['counted']
    side_only = {**asked, 'tasks': [{**asked['tasks'][0], 'judged_against': 'own_end_state'}]}
    assert 'undone' not in Families().classify(side_only)['counted']


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
