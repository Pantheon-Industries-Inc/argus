"""The banner over the dense timeline, run from the page's own source in node: a recording that carries an instruction
shows it above the model's reading in either mode (a session's summary or an episode's assessment), and one with none
shows the model's reading alone."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[1] / 'board' / 'index.html'

SCRIPT = r'''
const fs = require('fs');
const s = fs.readFileSync(process.argv[1], 'utf8');
const cut = (a, b) => { const i = s.indexOf(a), j = s.indexOf(b, i); if (i < 0 || j < i) throw new Error(a); return s.slice(i, j); };
const fn = name => cut(`function ${name}(`, '\n}\n') + '\n}\n';
const esc = x => String(x ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;');
const src = fn('nameParts') + fn('whoOwn') + cut('  // Given or inferred. When the dataset ships', '  // Key events (milestone layer), sorted.');
const banner = new Function('d', 'comp', 'meta', 'hasTasks', 'who', 'esc', src + 'return {bannerLabel, bannerBody};');
const cases = JSON.parse(process.argv[2]);
console.log(JSON.stringify(cases.map(c => banner(c.d, c.d.completion || {}, c.meta, c.hasTasks, 'Astra', esc))));
'''

CASES = [
    # a session labelled from a recording with an instruction: the instruction, then the summary
    {'d': {'episode_prompt': 'packs four toys', 'goal_alignment': {'relation': 'aligned', 'matches_given': True}},
     'meta': {'prompt_mode': 'session', 'given_prompt': 'pack the toys into the box'}, 'hasTasks': True},
    # an episode labelled against its instruction: the instruction, then the model's own assessment
    {'d': {'episode_prompt': 'stacks the cups'}, 'meta': {'prompt_mode': 'given', 'given_prompt': 'stack the cups'},
     'hasTasks': False},
    # a session with no instruction: the summary alone
    {'d': {'episode_prompt': 'wipes the table'}, 'meta': {'prompt_mode': 'session'}, 'hasTasks': True},
]


@pytest.mark.skipif(not shutil.which('node'), reason='no node')
def test_a_recorded_instruction_shows_above_the_models_reading_in_either_mode():
    p = subprocess.run(['node', '-e', SCRIPT, str(PAGE), json.dumps(CASES)], text=True, capture_output=True)
    assert p.returncode == 0, p.stderr
    session, episode, bare = json.loads(p.stdout)
    assert session['bannerLabel'] == 'Given task and session summary'
    assert session['bannerBody'].index('pack the toys into the box') < session['bannerBody'].index('packs four toys')
    assert 'matches given goal' in session['bannerBody']
    assert episode['bannerBody'].index('stack the cups') < episode['bannerBody'].index('stacks the cups')
    assert 'given' not in bare['bannerBody'] and 'wipes the table' in bare['bannerBody']
    assert bare['bannerLabel'] == 'Session summary'


# which tasks the page shows one by one, by the rule every outcome reader shares (board/label_projection.py
# verdict_or_tasks), then the banner those tasks give
TASKS_SCRIPT = r'''
const fs = require('fs');
const s = fs.readFileSync(process.argv[1], 'utf8');
const cut = (a, b) => { const i = s.indexOf(a), j = s.indexOf(b, i); if (i < 0 || j < i) throw new Error(a); return s.slice(i, j); };
const fn = name => cut(`function ${name}(`, '\n}\n') + '\n}\n';
const esc = x => String(x ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;');
const src = fn('nameParts') + fn('whoOwn') + cut("  // the rule every outcome reader shares", '  // Key events (milestone layer), sorted.');
const page = new Function('d', 'comp', 'meta', 'who', 'esc', src + 'return {shown: tasks.length, bannerLabel, bannerBody};');
console.log(JSON.stringify(JSON.parse(process.argv[2]).map(c => page(c.d, c.d.completion || {}, c.meta, 'Astra', esc))));
'''
TASK = {'start_s': 0, 'end_s': 9, 'task': 'stack the cups', 'outcome': 'success'}
ALIGNED = {'relation': 'aligned', 'matches_given': True}
TASK_CASES = [
    # one task: its verdict is the recording's, shown as an episode against its instruction, with the alignment chip
    {'d': {'episode_prompt': 'stacks the cups', 'goal_alignment': ALIGNED, 'tasks': [TASK],
           'completion': {'task_completed': 'success', 'from_tasks': 1, 'basis': 'one_task'}},
     'meta': {'prompt_mode': 'given', 'given_prompt': 'stack the cups'}},
    # several tasks: no verdict, each task shown, the instruction above the summary and the chip beside them
    {'d': {'episode_prompt': 'stacks the cups twice', 'goal_alignment': ALIGNED, 'tasks': [TASK, TASK],
           'completion': {'task_completed': None, 'from_tasks': 2, 'basis': 'tasks'}},
     'meta': {'prompt_mode': 'given', 'given_prompt': '(Cyclical) stack the cups, then reset'}},
    # a record labelled before the tasks format with a verdict and tasks keeps its tasks
    {'d': {'episode_prompt': 'stacks the cups', 'tasks': [TASK, TASK], 'completion': {'task_completed': 'unclear'}},
     'meta': {'prompt_mode': 'session'}},
    # an old episode: its verdict, no tasks
    {'d': {'episode_prompt': 'stacks the cups', 'completion': {'task_completed': 'success'}},
     'meta': {'prompt_mode': 'inferred'}},
]


@pytest.mark.skipif(not shutil.which('node'), reason='no node')
def test_a_recording_of_one_task_reads_as_an_episode_and_one_of_several_task_by_task():
    p = subprocess.run(['node', '-e', TASKS_SCRIPT, str(PAGE), json.dumps(TASK_CASES)], text=True, capture_output=True)
    assert p.returncode == 0, p.stderr
    one, several, legacy, episode = json.loads(p.stdout)
    assert one['shown'] == 0 and one['bannerLabel'].startswith('Given goal and'), one['bannerLabel']
    assert 'matches given goal' in one['bannerBody']
    assert several['shown'] == 2 and several['bannerLabel'] == 'Given task and session summary'
    assert 'matches given goal' in several['bannerBody']
    assert legacy['shown'] == 2 and episode['shown'] == 0


STRUCTURE_CASES = [
    # the dataset's timed steps and no instruction: the steps beside the tasks the model found, and their alignment
    {'d': {'episode_prompt': 'stacks the cups', 'tasks': [TASK, {**TASK, 'task': 'wipe the table'}],
           'goal_alignment': {'relation': 'broader', 'matches_given': True},
           'dataset_labels': [{'t0': 0, 't1': 4.5, 'label': 'stack the cups'}, {'t0': 4.5, 't1': 9, 'label': 'wipe up'}],
           'completion': {'task_completed': None, 'basis': 'task_prompts', 'from_tasks': 2}},
     'meta': {'prompt_mode': 'inferred'}},
    # neither: the goal the model inferred, with its success criterion
    {'d': {'episode_prompt': 'stacks the cups', 'tasks': [TASK],
           'completion': {'task_completed': 'success', 'success_predicate': 'three cups stacked', 'basis': 'one_task'}},
     'meta': {'prompt_mode': 'inferred'}},
    # an instruction and timed steps: the instruction and assessment, the chip, then the steps
    {'d': {'episode_prompt': 'stacks the cups', 'goal_alignment': ALIGNED, 'tasks': [TASK],
           'dataset_labels': [{'t0': 0, 't1': 9, 'label': 'stack the cups'}],
           'completion': {'task_completed': 'success', 'basis': 'instruction'}},
     'meta': {'prompt_mode': 'given', 'given_prompt': 'stack the cups'}},
]


@pytest.mark.skipif(not shutil.which('node'), reason='no node')
def test_the_banner_shows_every_claim_the_dataset_makes_beside_the_models_reading():
    p = subprocess.run(['node', '-e', TASKS_SCRIPT, str(PAGE), json.dumps(STRUCTURE_CASES)], text=True,
                       capture_output=True)
    assert p.returncode == 0, p.stderr
    steps, neither, both = json.loads(p.stdout)
    assert steps['bannerLabel'].startswith("The dataset's timed steps and")
    assert 'wipe up' in steps['bannerBody'] and 'wipe the table' in steps['bannerBody']
    assert "only part of the dataset's steps" in steps['bannerBody']
    assert 'success criterion' in neither['bannerBody'] and 'three cups stacked' in neither['bannerBody']
    body = both['bannerBody']
    assert body.index('given goal') < body.index('matches given goal') < body.index('timed steps')


REVIEW_CASES = [
    # a verdict against the instruction beside two tasks each judged against its step: the verdict, and both tasks
    {'d': {'episode_prompt': 'stacks the cups', 'goal_alignment': ALIGNED,
           'tasks': [{**TASK, 'judged_against': 'subtask', 'subtask': 1},
                     {**TASK, 'task': 'wipe up', 'judged_against': 'subtask', 'subtask': 2}],
           'dataset_labels': [{'t0': 0, 't1': 4.5, 'label': 'stack the cups', 'step': 1},
                              {'t0': 4.5, 't1': 9, 'label': 'wipe up', 'step': 2}],
           'completion': {'task_completed': 'failure', 'basis': 'instruction'}},
     'meta': {'prompt_mode': 'given', 'given_prompt': 'stack the cups and wipe up', 'label_schema': 'tasks'}},
    # a head-camera clip with timed steps and no goal, in the tasks format (no completion): its steps and alignment
    {'d': {'episode_prompt': 'wipes the table', 'tasks': [TASK, {**TASK, 'task': 'wipe the table'}],
           'goal_alignment': {'relation': 'aligned', 'matches_given': True},
           'dataset_labels': [{'t0': 0, 't1': 9, 'label': 'wipe up', 'step': 1}]},
     'meta': {'prompt_mode': 'inferred', 'label_schema': 'tasks'}},
    # a second text the dataset holds over the whole recording, beside the instruction
    {'d': {'episode_prompt': 'spells CORL', 'tasks': [TASK], 'dataset_task_claims': ['spell out CORL'],
           'completion': {'task_completed': 'success', 'basis': 'instruction'}},
     'meta': {'prompt_mode': 'given', 'given_prompt': 'arrange the blocks to spell CORL', 'label_schema': 'tasks'}},
]


@pytest.mark.skipif(not shutil.which('node'), reason='no node')
def test_a_verdict_keeps_its_judged_tasks_and_every_claim_reaches_the_banner():
    p = subprocess.run(['node', '-e', TASKS_SCRIPT, str(PAGE), json.dumps(REVIEW_CASES)], text=True,
                       capture_output=True)
    assert p.returncode == 0, p.stderr
    both, ego, whole = json.loads(p.stdout)
    assert both['shown'] == 2 and both['bannerLabel'].startswith('Given goal and')
    assert ego['bannerLabel'].startswith("The dataset's timed steps and") and 'matches the dataset' in ego['bannerBody']
    assert "dataset's task for the whole episode" in whole['bannerBody'] and 'spell out CORL' in whole['bannerBody']
    assert whole['shown'] == 0
