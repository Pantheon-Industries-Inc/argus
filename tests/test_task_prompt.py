"""The shared instructions (label/prompts.py fixed_instructions) by what the dataset gives: a robot episode with an
instruction keeps main's episode prompt word for word and adds its list of tasks; one without asks only for tasks;
the dataset's own timed steps add the rules for judging a task against its step; and the head camera keeps main's
prompt except for two approved edits. tests/fixtures/main_prompts.json holds main's (58c452e) text."""
import json
import re
from pathlib import Path

import pytest

from label import prompts

ROOT = Path(__file__).resolve().parents[1]
MAIN = json.loads((ROOT / 'tests' / 'fixtures' / 'main_prompts.json').read_text())
ROBOTS = ('teleop_arms', 'handheld_gripper')
RIGS = ROBOTS + ('ego_head',)
STRUCTURES = [(given, steps) for given in (True, False) for steps in (True, False)]


def flat(text: str) -> str:
    return ' '.join(text.split())


def text(rig, given, recorded=True, steps=False):
    return prompts.fixed_instructions(rig, has_instruction=given, recorded=recorded, has_steps=steps)


def main(rig, recorded=True):
    return MAIN[f'{rig}/{"recorded" if recorded else "video"}']


# The owner's two approved edits to the head camera's prompt: its absolutes softened, so a short clip of one task is
# an ordinary case, and (with no annotated goal) the rule for naming each task's end state.
EGO_SOFTENED = [
    ('''single overall goal, especially in a longer clip; the person does a SEQUENCE of
  distinct activities. "arm"/left/right means the person's LEFT or RIGHT HAND. Do NOT
  force one global success/failure or one goal frame. Segment the clip into the
  distinct activities in `tasks`. (A short atomic clip that is genuinely one task
  simply yields ONE task, with its own goal frame.)''',
     '''single overall goal, especially in a longer clip; the person may do a sequence of
  distinct activities. "arm"/left/right means the person's LEFT or RIGHT HAND. Do not
  force one global success/failure or one goal frame when there are several activities.
  Segment the clip into the distinct activities in `tasks`. A short clip that is genuinely
  one task yields one task, with its own goal frame.'''),
]


@pytest.mark.parametrize('given', [True, False])
@pytest.mark.parametrize('recorded', [True, False])
@pytest.mark.parametrize('steps', [True, False])
def test_the_head_camera_prompt_is_mains_but_for_the_two_approved_edits(given, recorded, steps):
    expected = main('ego_head', recorded)
    for old, new in EGO_SOFTENED:
        assert expected.count(old) == 1, old[:60]
        expected = expected.replace(old, new)
    if not given:
        lean = prompts.lean('ego_head')
        expected = expected[:expected.index(lean)] + prompts.no_instruction_rules('ego_head') + lean
    assert text('ego_head', given, recorded, steps) == expected


def test_the_head_camera_keeps_its_own_segmentation_outcomes_and_no_undo():
    """Its tasks keep the schema's own fields and outcomes, its own rule for where a task begins, and no undo: the
    shared robot task rules never reach it."""
    for given in (True, False):
        s = text('ego_head', given)
        assert 'Segment by intent\n  shift: a new task begins when the person turns to a different objective.' in s
        assert 'success_then_undone' not in s and '"judged_against"' not in s and '\nTASKS. ' not in s
        assert 'belong to no task' not in s and 'recording context goal' not in s


# S1 (audit_2026-10-21): with an instruction, a task that stands in for it by doing other work also gets the outcome of
# the work it did, beside its outcome against the instruction
OWN_FIELDS = '''     "own_success_predicate": "<for a task judged against the instruction whose work is not the work it asks for: the end state of the work it evidently set out to do; otherwise null>",
     "own_outcome": "success" | "success_then_undone" | "failure" | "partial" | "unclear" | null,
'''
OWN_RULE = '''
- A task judged against the instruction whose work is not the work it asks for (it stands in for it) is also judged
  against the end state of the work it evidently set out to do, as a task with no instruction would be:
  own_success_predicate states that end state and own_outcome says whether the footage shows it reached, by the same
  rules. They never change the task's outcome against the instruction, or completion. Every other task leaves both
  null.'''


def _robot_expected(rig, recorded, steps):
    """Main's episode prompt with the list of tasks after its completion, the task rules (and the step rules) before
    the granularity note, and that note naming the tasks."""
    expected = main(rig, recorded)
    end = '    "reason": "<if not a clean success, why>"\n  },\n'
    assert expected.count(end) == 1
    note = '     "note": "<if not a clean success, why>"}'
    assert prompts.TASK_FIELDS.count(note) == 1
    expected = expected.replace(end, end + prompts.TASK_FIELDS.replace(note, OWN_FIELDS + note) + '\n')
    lean = prompts.lean(rig, has_instruction=True)
    old_lean = lean.replace('(completion, tasks, goal_alignment,', '(completion, goal_alignment,')
    assert expected.endswith(old_lean)
    return (expected[:-len(old_lean)] + prompts.task_rules(rig, has_instruction=True, has_steps=steps)
            + (prompts.step_rules(True) if steps else '') + lean)


@pytest.mark.parametrize('rig', ROBOTS)
@pytest.mark.parametrize('recorded', [True, False])
@pytest.mark.parametrize('steps', [True, False])
def test_a_robot_episode_with_an_instruction_keeps_mains_prompt_word_for_word_and_adds_its_tasks(rig, recorded,
                                                                                                 steps):
    """The completion judges the recording against the instruction exactly as the episode prompt always did: held
    through the end, a scene reset for the next attempt undoes it, unasked work never raises it."""
    got = text(rig, True, recorded, steps)
    assert got == _robot_expected(rig, recorded, steps)
    s = flat(got)
    for sentence in ('the recording kept running while the scene was reset for the next attempt',
                     'success_predicate is satisfied AND then stays satisfied through the end of the episode.',
                     'so a demo of a different task is a completion "failure" against the given goal',
                     'a scene reset for the next attempt still makes completion success_then_undone.'):
        assert sentence in s, sentence
    assert prompts.INSTRUCTION_RULES in got
    assert prompts.requested_outcome(rig, has_instruction=True, has_steps=steps) == 'completion.task_completed'


@pytest.mark.parametrize('rig', ROBOTS)
@pytest.mark.parametrize('given,steps', STRUCTURES)
def test_a_task_is_judged_only_against_claims_the_episode_has(rig, given, steps):
    s = flat(prompts.task_rules(rig, has_instruction=given, has_steps=steps))
    assert ('"subtask" when one of the dataset\'s timed steps covers it' in s) is steps
    assert ('"instruction" ' in s) is given
    assert '"own_end_state"' in s
    full = flat(text(rig, given, True, steps))
    assert ('ABOUT THE DATASET\'S TIMED STEPS' in full) is steps
    # goal_alignment: against the instruction when there is one, else the steps as a sequence, else none
    assert ('"goal_alignment"' in full) is (given or steps)
    assert ("comparing the dataset's steps, as one sequence, with the tasks you observed" in full) is (steps and not given)


@pytest.mark.parametrize('rig', ROBOTS)
@pytest.mark.parametrize('recorded', [True, False])
@pytest.mark.parametrize('steps', [True, False])
def test_a_robot_episode_without_an_instruction_asks_only_for_tasks(rig, recorded, steps):
    got = text(rig, False, recorded, steps)
    assert '"tasks": [' in got and '"completion"' not in got
    assert not re.findall(r'completion[.:]|\(completion,|completion (judges|above)|(before|change) completion', flat(got))
    s = flat(got)
    for sentence in ('a recording of one demonstration is one task.',
                     'in a cycle, where the objects are reset and the same work repeats, each repeat is its own task, '
                     'judged on its own.',
                     'never as a description that any motion would satisfy (moving, repositioning or rearranging the '
                     'objects).',
                     'Resetting or reusing its objects to start another task, or the next repeat of a cycle, is the start '
                     'of that work, not an undo.'):
        assert s.count(sentence) == 1, sentence
    assert prompts.requested_outcome(rig, has_instruction=False, has_steps=steps) == 'tasks'


@pytest.mark.parametrize('rig', RIGS)
@pytest.mark.parametrize('given,steps', STRUCTURES)
def test_every_undo_rule_says_when_a_result_must_hold(rig, given, steps):
    """No prompt says that a result taken apart is undone without saying how long it had to hold: the old no-instruction
    sentence did, and it contradicted the rule that resetting objects to start other work is no undo."""
    s = flat(text(rig, given, True, steps))
    assert 'When it was reached and later taken apart within the episode, the outcome is success_then_undone.' not in s
    assert 'taken apart within the clip' not in s
    if not given and rig != 'ego_head':
        assert ('When that end state was reached and then lost before the recording moved on to other work, the '
                'outcome is success_then_undone.') in s


@pytest.mark.parametrize('rig', RIGS)
def test_the_rigs_words_survive(rig):
    for given in (True, False):
        got = text(rig, given)
        if rig == 'handheld_gripper':
            assert 'It is NOT when the grippers finish moving.' in got and 'by ONE gripper' in got
        if rig == 'teleop_arms':
            assert 'It is NOT when the arms finish moving.' in got and 'by ONE arm' in got
        if rig == 'ego_head':
            assert 'by ONE hand' in got and 'hands_visible' in got and 'demonstrator' not in got


@pytest.mark.parametrize('recorded', [True, False])
def test_a_head_camera_clip_with_no_annotated_goal_names_each_task_by_a_specific_end_state(recorded):
    """The robots' rule for naming a task with no instruction holds for every task a head-camera clip with no
    annotated goal infers; it changes no segmentation and no outcome."""
    rule = flat(prompts.no_instruction_rules('ego_head'))
    assert ('Name each as the most specific end state the person evidently worked toward' in rule
            and "Each task's success_predicate states that end state" in rule
            and 'never as a description that any motion would satisfy' in rule and 'the work coming to a stop' in rule)
    assert 'demonstrator' not in rule and 'episode' not in rule and 'success_then_undone' not in rule
    assert flat(text('ego_head', False, recorded)).count(rule) == 1 and rule not in flat(text('ego_head', True, recorded))


@pytest.mark.parametrize('rig', RIGS)
def test_the_in_context_example_is_in_the_format_it_closes(rig):
    """A --with-example run shows one complete annotation after the schema (label/prompts.py example_block); it holds
    every field the schema it closes asks a task for, and the completion when that schema asks for one."""
    example = json.loads((ROOT / 'configs' / 'examples' / f'example_{rig}.json').read_text())
    labels = example['labels']
    given = not str(example['context'].get('instruction') or '').startswith('(none')
    got = text(rig, given)
    task_fields = re.findall(r'"(\w+)":', got[got.index('"tasks": [') + 10:got.index('}', got.index('"tasks": ['))])
    assert labels['tasks'] and ('completion' in labels) is ('"task_completed"' in got)
    for task in labels['tasks']:
        assert set(task_fields) <= set(task), set(task_fields) - set(task)


@pytest.mark.parametrize('rig', ROBOTS)
@pytest.mark.parametrize('recorded', [True, False])
@pytest.mark.parametrize('steps', [True, False])
def test_a_task_standing_in_for_the_instruction_is_also_judged_on_its_own_work(rig, recorded, steps):
    """S1: a task that stands in for the instruction by doing other work had only its outcome against the instruction,
    so a competent demonstration of other work kept no outcome of its own. With an instruction the tasks ask for both;
    with none every task is already judged on its own end state, and that prompt is unchanged."""
    given = text(rig, True, recorded, steps)
    assert given.count(OWN_RULE) == 1 and given.count(OWN_FIELDS) == 1
    bare = text(rig, False, recorded, steps)
    assert 'own_outcome' not in bare and 'own_success_predicate' not in bare
    assert 'own_outcome' not in text('ego_head', True, recorded, steps)
