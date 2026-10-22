"""The recording verdict and each task's claim, by what the dataset gives (its instruction, its own timed per-task
prompts, both or neither), and the readers that put each claim where it belongs.

Each test holds a fault the earlier tasks-format build had (one_schema/REVIEW.md, findings 1 to 3) or a reader that
put a claim in the wrong place on main: an instruction judged through the tasks instead of directly, a step shown
twice or as a fake step, a changing task text flattened into one instruction, a robot's step topic left in the notes.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from board import build, rules, scores, to_board
from gate.score import judge
from label import episode, harness, prompts

INSTANTS = [round(1.5 * k, 3) for k in range(7)]          # a 9s request sampled every 1.5s
INSTRUCTION = 'Stack the cups in a row.'


def _request(instruction=INSTRUCTION, *, schema=True, steps=0):
    """The record of a robot request: in the tasks format (label_schema), or asked in the episode mode before it."""
    policy = {'version': 1, 'rig': 'teleop_arms', **({} if schema else {'task_mode': 'episode'})}
    return {'annotation_policy': policy, 'given_prompt': instruction, 'config': {'timesteps_s': INSTANTS},
            'task_prompts': steps, **({'label_schema': 'tasks'} if schema else {})}


def _timeline(progress=0.5):
    return [{'start_s': 0.0, 'end_s': 4.5, 'arm': 'right', 'action': 'place', 'object': 'cup',
             'contribution': 'advancing', 'progress': progress},
            {'start_s': 4.5, 'end_s': 9.0, 'arm': 'right', 'action': 'stack', 'object': 'block',
             'contribution': 'advancing', 'progress': progress}]


def _task(task, outcome, against, start, end, **fields):
    return {'start_s': start, 'end_s': end, 'task': task, 'objects': ['cup'], 'judged_against': against,
            'success_predicate': f'{task} done', 'outcome': outcome, **fields}


def _read(reply, request):
    labels, ok = harness.read_reply(json.dumps(reply), request)
    return labels, ok


def _board(labels, request):
    result = {'episode_dir': 'episode_x', 'parse_ok': True, 'labels': labels, 'given_prompt': request['given_prompt'],
              **({'label_schema': request['label_schema']} if 'label_schema' in request else {})}
    return to_board.convert(result, 'demo')


def _outcome_score(d):
    return scores._outcome(scores.facts(d))[0]


# ---- finding 1: a single-instruction robot recording with an extra, unasked task --------------------------------------

def test_an_unasked_task_never_raises_the_recordings_outcome_against_its_instruction():
    """The asked row fails (progress 0.5), then the demonstrator builds a tower nobody asked for and succeeds. The
    episode prompt judged the recording a failure scoring 4.0; the reply now judges it the same way in its completion,
    and lists both tasks beside it. Derived from the tasks (the earlier build) it scored 7.0, 75% too high."""
    old = {'task_summary': 'Stack the cups in a row.', 'timeline': _timeline(),
           'completion': {'task_completed': 'failure', 'reason': 'the row is never finished'}}
    new = {**old, 'tasks': [_task('stack the cups in a row', 'failure', 'instruction', 0.0, 4.5),
                            _task('build a tower', 'success', 'own_end_state', 4.5, 9.0, completed_at_s=9.0)]}
    old_labels, ok_old = _read(old, _request(schema=False))
    new_labels, ok_new = _read(new, _request())
    assert ok_old and ok_new
    assert new_labels['completion'] == {**old['completion'], 'basis': 'instruction'}
    old_d, new_d = _board(old_labels, _request(schema=False)), _board(new_labels, _request())
    assert _outcome_score(old_d) == _outcome_score(new_d) == 4.0
    old_f, new_f = scores.facts(old_d), scores.facts(new_d)
    assert new_f['tasks'] == [] and {k: v for k, v in new_f.items() if k != 'tasks'} == {
        k: v for k, v in old_f.items() if k != 'tasks'}


def test_a_goal_undone_mistake_is_capped_when_the_recording_against_its_instruction_was_undone():
    """GENERAL's cap_when caps a goal_undone mistake on a recording whose outcome already counts the undo. With an
    instruction the outcome is the completion against it, also when the reply lists an unasked task after it."""
    mistake = {'issue': 'the row is swept aside', 'category': 'goal_undone', 'severity': 'medium', 't_s': 6.0}
    reply = {'task_summary': 'Stack the cups in a row.', 'timeline': _timeline(1.0), 'operator_mistakes': [mistake],
             'completion': {'task_completed': 'success_then_undone', 'goal_reached_at_s': 4.0, 'undone_at_s': 6.0,
                            'undone_by': 'the tower build sweeps the row aside'},
             'tasks': [_task('stack the cups in a row', 'success_then_undone', 'instruction', 0.0, 4.5,
                             goal_reached_at_s=4.0, undone_at_s=6.0, undone_by='swept aside'),
                       _task('build a tower', 'success', 'own_end_state', 4.5, 9.0, completed_at_s=9.0)]}
    labels, ok = _read(reply, _request())
    assert ok
    d = _board(labels, _request())
    build.apply_rules(d, {}, rules.GENERAL)
    assert [m['severity'] for m in d['operator_mistakes']] == ['low']
    # with no instruction and two tasks there is no verdict; the undone task caps it the same way
    bare = {**{k: v for k, v in reply.items() if k != 'completion'},
            'tasks': [{**t, 'judged_against': 'own_end_state'} for t in reply['tasks']]}
    labels, ok = _read(bare, _request(None))
    assert ok and labels['completion']['task_completed'] is None
    d = _board(labels, _request(None))
    build.apply_rules(d, {}, rules.GENERAL)
    assert [m['severity'] for m in d['operator_mistakes']] == ['low']


# ---- finding 2: a scene reset for the next attempt is an undo for single-task robot data --------------------------------

def test_a_reset_for_the_next_attempt_still_undoes_the_recording_against_its_instruction():
    """The goal is reached, the scene is reset and a second attempt is cut off by the end of the recording. The
    episode prompt's rule holds the completion to the end of the episode, so the recording is success_then_undone
    whatever the tasks say about each attempt, and the gate case of that shape is met."""
    text = ' '.join(prompts.fixed_instructions('teleop_arms', has_instruction=True).split())
    assert 'the recording kept running while the scene was reset for the next attempt' in text
    assert 'a scene reset for the next attempt still makes completion success_then_undone.' in text
    reply = {'task_summary': 'Stack the cups in a row.', 'timeline': _timeline(1.0),
             'completion': {'task_completed': 'success_then_undone', 'goal_reached_at_s': 4.0, 'undone_at_s': 5.0,
                            'undone_by': 'the scene is reset for the next attempt'},
             'tasks': [_task('stack the cups in a row', 'success', 'instruction', 0.0, 4.5, completed_at_s=4.0),
                       _task('stack the cups in a row', 'unclear', 'instruction', 5.0, 9.0)]}
    labels, ok = _read(reply, _request())
    assert ok and labels['completion']['task_completed'] == 'success_then_undone'
    assert judge({'outcome': ['success_then_undone']}, labels) == (True, [])
    assert _outcome_score(_board(labels, _request())) == 7.0


# ---- finding 3: a recording with an instruction is judged against it ------------------------------------------------------

def test_a_reply_that_judges_no_task_against_the_instruction_is_incomplete_and_never_passes_the_gate():
    """008276: the instruction asks for black pants, the garment is a shirt. A reply that only names its own task
    ("fold the shirt", success) gives no verdict against the instruction: it is an incomplete annotation, and a gate
    case saying what the outcome must not be is unmet rather than passed by the set of task outcomes."""
    instruction = 'Fold the black pants.'
    shirt = _task('fold the shirt', 'success', 'own_end_state', 0.0, 9.0, completed_at_s=8.0)
    reply = {'task_summary': 'Fold the shirt.', 'timeline': _timeline(1.0), 'tasks': [shirt],
             'goal_alignment': {'matches_given': False, 'relation': 'different', 'note': 'a shirt, not pants'}}
    labels, ok = _read(reply, _request(instruction))
    assert not ok and labels['_incomplete'] == [
        'the request asks the outcome of its task, and completion.task_completed is missing']
    case = {'not_outcome': ['success', 'success_then_undone']}
    met, why = judge(case, {**labels, 'tasks': [shirt, {**shirt, 'start_s': 9.0}],
                            'completion': {'task_completed': None, 'basis': 'tasks'}})
    assert not met and 'no task judged against the instruction' in why[0]
    # judged against the instruction, the recording fails it and the case is met
    labels, ok = _read({**reply, 'completion': {'task_completed': 'failure', 'reason': 'a shirt is folded'}},
                       _request(instruction))
    assert ok and judge(case, labels)[0]


# ---- what each structure asks, on a real request ----------------------------------------------------------------------

def _episode(tmp_path, *, instruction='', steps=None):
    from test_label import _packed_episode
    ep, _ = _packed_episode(tmp_path)
    ctx = json.loads((ep / 'context.json').read_text())
    ctx['instruction'] = instruction
    ctx.pop('instruction_note', None)
    if steps:
        ctx['annotation_subtasks'] = steps
    (ep / 'context.json').write_text(json.dumps(ctx))
    return ep


STEPS = [{'t0': 0.0, 't1': 1.5, 'label': 'pick up the block'}, {'t0': 1.5, 't1': 3.9, 'label': 'place the block'}]


@pytest.mark.parametrize('instruction,steps', [('Spell AI2.', None), ('', STEPS), ('Spell AI2.', STEPS), ('', None)])
def test_each_structure_asks_for_what_it_has_and_numbers_its_steps_once(tmp_path, instruction, steps):
    """Global only: the completion against the instruction and the tasks. Per-task only: the tasks, each against
    its step, and goal_alignment comparing the steps with the tasks. Both: the completion, and the tasks against
    their steps. Neither: the tasks alone. Every step reaches the request once, numbered as a task's subtask names it."""
    request = episode.build_request(_episode(tmp_path, instruction=instruction, steps=steps), cell_w=128)
    fixed, text = request['content'][0]['text'], request['prompt']
    assert ('"task_completed"' in fixed) is bool(instruction) and '"tasks": [' in fixed
    assert ("ABOUT THE DATASET'S TIMED STEPS" in fixed) is bool(steps)
    assert ('"goal_alignment"' in fixed) is bool(instruction or steps)
    assert ("comparing the dataset's steps, as one sequence" in ' '.join(fixed.split())) is (bool(steps)
                                                                                             and not instruction)
    assert request['task_prompts'] == len(steps or [])
    for i, step in enumerate(steps or [], 1):
        assert text.count(step['label']) == 1 and f'  {i}. ' in text


# ---- the readers --------------------------------------------------------------------------------------------------------

def test_openaoe_notes_reach_an_upload_only_where_its_own_metadata_names_openaoe(tmp_path):
    """An OpenAoE clip folder whose upload names OpenAoE (its README) gets what the dataset's adapter says of its
    action segments and its clips. The same rows (start_ts, end_ts) and folder name (raw_*_seg_N) without it state
    neither fact: another upload can share both, and a name match establishes nothing."""
    from prepare import formats
    from test_openaoe_generic import clip, convert
    clip(tmp_path / 'named' / 'raw_kitchen_0412_seg_3')
    (tmp_path / 'named' / 'raw_kitchen_0412_seg_3' / 'README.md').write_text('# OpenAoE-2000h clip\n')
    _, ctx, _ = convert(tmp_path / 'named' / 'raw_kitchen_0412_seg_3', tmp_path / 'out_named')
    assert formats.ATOMIC_ACTION_NOTE in ctx['annotation_note']
    assert formats.SEGMENT_CLIP_NOTE in ctx['collection_note']
    clip(tmp_path / 'bare' / 'raw_kitchen_0412_seg_3')
    _, ctx, _ = convert(tmp_path / 'bare' / 'raw_kitchen_0412_seg_3', tmp_path / 'out_bare')
    assert 'whole-second grid' not in ctx['annotation_note']
    assert 'numbered segment' not in str(ctx.get('collection_note') or '')


def test_a_lerobot_task_index_over_the_whole_episode_is_its_task_never_a_step(tmp_path):
    """A task_index column that holds one value over every row and names the episode's task is that task, given once
    as the instruction; it was also listed as a step repeating the instruction."""
    import numpy as np
    from prepare import formats
    from test_formats import _lerobot
    root = tmp_path / 'upload'
    _lerobot(root, {0: {'observation.state': [np.zeros(14)] * 10, 'task_index': [0] * 10}})
    (root / 'meta' / 'tasks.jsonl').write_text(json.dumps({'task_index': 0, 'task': 'fold the towel'}) + '\n')
    report = formats.convert(root, 'teleop_arms', tmp_path / 'out', 'task index', 900)
    assert not report['failed'], report
    ctx = json.loads((tmp_path / 'out' / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    assert ctx['instruction'] == 'fold the towel'
    assert not [s for s in ctx.get('annotation_subtasks') or [] if s.get('label') == 'fold the towel']
    assert [s['label'] for s in ctx['annotation_task_claims']] == ['fold the towel']


def test_galaxea_gives_its_timed_steps_once(tmp_path):
    """Galaxea's sub-steps reached the model twice, inside the instruction note and as the dataset's steps."""
    from prepare import galaxea
    from test_prepare import _galaxea_folder
    root = tmp_path / 'galaxea_one'
    _galaxea_folder(root)
    meta = galaxea.meta_from(lambda rel: root / rel, 'galaxea_one')
    ctx = galaxea.write_episode(meta, meta['episodes'][0], lambda rel: root / rel, tmp_path / 'ep', 'galaxea_test')
    assert ctx['annotation_subtasks']
    assert ctx['instruction_note'] == ("This is the dataset's task for the whole episode. Its quality tag for the "
                                       "episode: qualified.")


def test_habit_gives_its_timed_steps_once(tmp_path):
    """HABIT's person and robot parts reached the model twice, listed in the instruction note and as steps."""
    from test_habit_generic import native, published
    root = tmp_path / 'recording'
    native(root)
    ep, ctx = published(root, tmp_path / 'old')
    for label in ('robot reach', 'robot fold', 'person hold', 'person release'):
        assert label not in ctx['instruction_note'], label
    text = episode.build_request(ep, cell_w=128)['prompt']
    block = text[text.index("THE DATASET'S TIMED STEPS FOR THIS EPISODE"):]
    block = block[:block.index('\n\n')] if '\n\n' in block else block
    for label in ('robot reach', 'robot fold', 'person hold', 'person release'):
        assert block.count(label) == sum(s['label'] == label for s in ctx['annotation_subtasks']) > 0, label


def _text_mcap(tmp_path, rig):
    from prepare import formats
    from test_ingestion_v5 import text_recording
    source = tmp_path / 'source.mcap'
    text_recording(source, clock_id='fixture')
    report = formats.convert(source, rig, tmp_path / 'out', 'task topics', 900)
    assert not report['failed'], report
    return json.loads((tmp_path / 'out' / report['episodes'][0]['episode_id'] / 'context.json').read_text())


@pytest.mark.parametrize('rig', ['ego_head', 'teleop_arms'])
def test_a_task_topic_whose_text_changes_gives_timed_task_prompts_not_one_instruction(tmp_path, rig):
    """MicroAGI's /task names the task of each stretch ("prepare", then "fold the cloth" at 1s). It was flattened into
    one instruction ("0.0 s: prepare; 1.0 s: fold the cloth"); each text is now a task prompt with its own span, named
    with its topic, beside the /task/subtask steps."""
    ctx = _text_mcap(tmp_path, rig)
    assert not ctx.get('instruction')
    rows = [(s.get('column'), s['label'], s['t0']) for s in ctx['annotation_subtasks']]
    assert ('/task', 'prepare', 0.0) in rows and ('/task', 'fold the cloth', 1.0) in rows
    assert ('/task/subtask', 'pick up', 0.0) in rows and ('/task/subtask', 'fold', 2.5) in rows


def test_a_robot_mcap_step_topic_becomes_the_datasets_timed_steps(tmp_path):
    """On a robot rig the /task/subtask steps stayed in the uploader's notes and never became steps a task is judged
    against; the request now numbers them under the dataset's timed steps."""
    ctx = _text_mcap(tmp_path, 'teleop_arms')
    assert [s['label'] for s in ctx['annotation_subtasks'] if s.get('column') == '/task/subtask'] == ['pick up', 'fold']
    # main kept them as a timeline of notes ("0.0 s: pick up"); the topic's clock record stays in the notes
    assert 's: pick up' not in json.dumps(ctx.get('uploader_annotation') or {})
    request = episode.build_request(tmp_path / 'out' / ctx['episode_id'], cell_w=128)
    assert "THE DATASET'S TIMED STEPS FOR THIS EPISODE" in request['prompt']
    assert '[/task/subtask] pick up' in request['prompt'] and '[/task] fold the cloth' in request['prompt']


# ---- the board -----------------------------------------------------------------------------------------------------------

PAGE = Path(__file__).resolve().parents[1] / 'board' / 'index.html'
VERDICT_SCRIPT = r'''
const fs = require('fs');
const s = fs.readFileSync(process.argv[1], 'utf8');
const cut = (a, b) => { const i = s.indexOf(a), j = s.indexOf(b, i); if (i < 0 || j < i) throw new Error(a); return s.slice(i, j); };
const src = cut("  const verdict = (comp.task_completed || '').toLowerCase();", '  const taskRows = [];')
  + cut('  const tAll = tasks.length, tOk = tcount.success;', '  const outcomeKv =');
const line = new Function('d', 'comp', 'tasks', 'hasTasks', 'partProgress', 'esc', 'fmtT', 'cap', 'outcomeWords',
  'partsGapHtml', src + 'return verdictHtml;');
const tag = x => String(x).replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
console.log(JSON.stringify(JSON.parse(process.argv[2]).map(c => tag(line(c.d, c.d.completion || {}, c.tasks,
  c.tasks.length > 0, c.partProgress, String, t => `${t}s`, x => x, x => x.replace(/_/g, ' '), () => '')))));
'''


@pytest.mark.skipif(not shutil.which('node'), reason='no node')
def test_the_verdict_line_counts_an_undone_task_and_a_stitched_task_verdict_has_its_goal_frame():
    """A task reached and then undone is counted as undone, never as unclear. A recording labelled in parts whose one
    task was joined across the cuts has that task's verdict and goal frame, never "goal frame unassessed". One with an
    instruction, which no part sees whole, stays unclear with its goal frame unassessed, as on main."""
    task = lambda outcome: {'task': 't', 'outcome': outcome}
    cases = [
        {'d': {}, 'tasks': [task('success'), task('success_then_undone'), task('failure'), task('unclear')],
         'partProgress': False},
        {'d': {'completion': {'task_completed': 'success', 'completed_at_s': 12.5, 'basis': 'one_task'}}, 'tasks': [],
         'partProgress': True},
        {'d': {'completion': {'task_completed': 'unclear', 'basis': 'parts'}}, 'tasks': [], 'partProgress': True},
        # F4: the open first half of a task that succeeded in the next part is neither done nor unclear
        {'d': {}, 'tasks': [task('continues_past_cut'), task('success')], 'partProgress': True},
    ]
    p = subprocess.run(['node', '-e', VERDICT_SCRIPT, str(PAGE), json.dumps(cases)], text=True, capture_output=True)
    assert p.returncode == 0, p.stderr
    several, joined, parted, open_half = json.loads(p.stdout)
    assert open_half == '1 of 1 task succeeded, 1 open at a cut'
    assert several == '1 of 4 tasks succeeded, 1 undone, 1 failed, 1 unclear'
    assert joined == 'success goal frame 12.5s'
    assert parted == 'unclear Whole-recording goal frame unassessed'
