"""The shared instructions (label/prompts.py fixed_instructions): every recording is a list of tasks with an outcome
each, and nothing the episode mode asked of a recording with one task is lost."""
import re

import pytest

from label import prompts

RIGS = ('teleop_arms', 'handheld_gripper', 'ego_head')
MATRIX = [(rig, given, recorded) for rig in RIGS for given in (True, False) for recorded in (True, False)]


def flat(text: str) -> str:
    return ' '.join(text.split())


def tasks_text(rig, given, recorded):
    return prompts.fixed_instructions(rig, has_instruction=given, recorded=recorded)


# The sentences of the episode mode's instruction rules that the tasks format keeps word for word, so a recording of
# one task is still graded against its instruction as an episode was.
PINNED_INSTRUCTION = [
    'This instruction is metadata whose correctness is NOT guaranteed; treat it as a claim to check, not as the truth '
    'about the video.',
    'The timeline describes the trajectory, so it follows what the operator is ACTUALLY, demonstrably doing (your '
    'independently-inferred task_summary), not the given goal.',
    'contribution follows the shared rule in the schema, against the operator\'s evident activity, never simply against '
    'whether the step serves the given instruction.',
    'so a demo that never reaches the given goal never reaches 1.0, however well it does its own task.',
    'A competent demonstration is good data even when it is mislabeled; the dense timeline must stay a faithful '
    'description of the real behavior.',
    'Calling it success would teach a policy that the instruction means that other task.',
    'Still fill task_summary with what you INDEPENDENTLY observe.',
    'a genuinely unmet requested goal remains partial or failed when the available recording establishes that '
    'outcome; unseen continuation is unknown.',
    'A demonstration that reaches the given goal and then does more is "narrower".',
    'cannot sit beside a "different" or "unrelated" relation (the footage never shows the given goal).',
]
PINNED_NO_INSTRUCTION = [
    'never as a description that any motion would satisfy (moving, repositioning or rearranging the objects).',
    'When it was reached and later taken apart within the episode, the outcome is success_then_undone.',
]
PINNED_TASKS = [
    'a recording of one demonstration is one task.',
    'in a cycle, where the objects are reset and the same work repeats, each repeat is its own task, judged on its own.',
    'failed attempts and their retries, recovery from a fumble, regrasps, setup before the work, and parking or retreat '
    'after it.',
    'Resetting or reusing its objects to start another task, or the next repeat of a cycle, is the start of that work, '
    'not an undo.',
    'A task judged against the instruction must hold to the end of the recording',
]


@pytest.mark.parametrize('rig,given,recorded', MATRIX)
def test_every_variant_asks_for_tasks_with_their_own_outcome_and_no_completion(rig, given, recorded):
    text = tasks_text(rig, given, recorded)
    assert '"tasks": [' in text and '"completion"' not in text
    # no rule may still speak of the one completion ("partial completion" and "quantity and completion" are words, not
    # the field)
    assert not re.findall(r'completion[.:]|\(completion,|completion (judges|above)|(before|change) completion',
                          flat(text))
    schema = text[text.index('"tasks": ['):text.index('"scene_graph"')]
    for field in ('"judged_against": "instruction" | "own_end_state"', 'success_then_undone', '"goal_reached_at_s"',
                  '"undone_at_s"', '"undone_by"', '"completed_at_s"', '"success_predicate"'):
        assert field in schema, field
    s = flat(text)
    for sentence in PINNED_TASKS:
        assert s.count(sentence) == 1, sentence
    assert 'session' not in s.lower(), 'the task mode no longer shapes the text'
    assert prompts.requested_outcome(rig, has_instruction=given) == 'tasks'


@pytest.mark.parametrize('rig,recorded', [(r, rec) for r in RIGS[:2] for rec in (True, False)])
def test_a_robot_recording_with_an_instruction_keeps_every_episode_mode_clause(rig, recorded):
    s = flat(tasks_text(rig, True, recorded))
    for sentence in PINNED_INSTRUCTION:
        assert sentence in s, sentence
    assert prompts._INSTRUCTION_CLAUSES in tasks_text(rig, True, recorded)
    # the instruction is checked against the whole list of tasks, and a cycle is graded repeat by repeat
    assert '"goal_alignment"' in s and 'Every recording with an instruction has at least one task judged against it.' in s
    assert ('The reset the instruction asks for is part of each repeat\'s end state: it is never a scene_reset issue and '
            'never an undo, and starting the next repeat is not an undo either.') in s
    assert 'matches_given is false only when the relation is different or unrelated.' in s
    for sentence in PINNED_NO_INSTRUCTION:
        assert sentence not in s


@pytest.mark.parametrize('rig,recorded', [(r, rec) for r in RIGS[:2] for rec in (True, False)])
def test_a_robot_recording_without_an_instruction_names_each_task_by_its_end_state(rig, recorded):
    s = flat(tasks_text(rig, False, recorded))
    for sentence in PINNED_NO_INSTRUCTION:
        assert sentence in s, sentence
    assert '"goal_alignment"' not in s and 'ABOUT THE EPISODE\'S INSTRUCTION' not in s


@pytest.mark.parametrize('recorded', [True, False])
def test_the_head_camera_has_one_annotation_rule_whatever_the_source_declares(recorded):
    s = flat(tasks_text('ego_head', True, recorded))
    assert flat(prompts.EGO_ANNOTATION_RULES) in s
    assert 'its tasks are the steps of that goal' not in s and 'independent activities' not in s
    assert 'is not a missing or invented task' in s
    assert 'demonstrator' not in s and 'object between the jaws' not in s
    # with an annotated goal the tasks carrying it are judged against it; the end-state rule is for a clip without one
    assert 'ABOUT THE TASKS.' not in s


@pytest.mark.parametrize('rig', RIGS)
def test_the_rigs_words_survive_the_tasks_format(rig):
    text = tasks_text(rig, True, True)
    if rig == 'handheld_gripper':
        assert 'It is NOT when the grippers finish moving.' in text and 'by ONE gripper' in text
    if rig == 'teleop_arms':
        assert 'It is NOT when the arms finish moving.' in text and 'by ONE arm' in text
    if rig == 'ego_head':
        assert 'by ONE hand' in text and 'hands_visible' in text


@pytest.mark.parametrize('rig', RIGS)
def test_the_in_context_example_is_in_the_format_it_closes(rig):
    """A --with-example run shows one complete annotation after the schema (label/prompts.py example_block); it holds
    the tasks the schema asks for, each with the schema's fields, and no completion."""
    import json
    from pathlib import Path
    example = json.loads((Path(__file__).resolve().parents[1] / 'configs' / 'examples' / f'example_{rig}.json').read_text())
    labels = example['labels']
    text = tasks_text(rig, True, True)
    fields = re.findall(r'"(\w+)":', text[text.index('"tasks": ['):text.index('"scene_graph"')])[1:]
    assert 'completion' not in labels and labels['tasks']
    for task in labels['tasks']:
        assert set(fields) <= set(task), set(fields) - set(task)
        assert task['judged_against'] in ('instruction', 'own_end_state')


@pytest.mark.parametrize('recorded', [True, False])
def test_the_head_camera_keeps_its_framing_of_several_activities_and_reads_one_task_as_normal(recorded):
    """Most head-camera recordings are several activities with no single goal, so that framing stays; only its
    absolutes are softened, so a short clip of one task is an ordinary case."""
    s = flat(tasks_text('ego_head', False, recorded))
    for kept in ('There is NO robot and usually NO single overall goal, especially in a longer clip',
                 'Segment the clip into the distinct activities in `tasks`.',
                 'tasks is the list of distinct activities, in time order.',
                 'NOT the whole recording: 0 when they start a sub-task'):
        assert kept in s, kept
    assert 'the person may do a sequence of distinct activities.' in s
    assert 'Do not force one global success/failure or one goal frame when there are several activities.' in s
    assert 'A short clip that is genuinely one task yields one task, with its own goal frame.' in s
    assert 'the person does a SEQUENCE' not in s and '(A short atomic clip' not in s
    assert 'A clip of several tasks has no single episode-level goal frame' in s


@pytest.mark.parametrize('recorded', [True, False])
def test_a_head_camera_clip_with_no_annotated_goal_names_each_task_by_a_specific_end_state(recorded):
    """The robot rigs' rule for a recording with no instruction holds for every task a head-camera clip infers: its
    success_predicate is a concrete end state, never a description any motion satisfies or the work stopping. It
    changes no segmentation: the clip is still cut into its distinct activities."""
    s = flat(tasks_text('ego_head', False, recorded))
    rule = flat(prompts.no_instruction_rules('ego_head'))
    assert s.count(rule) == 1 and rule not in flat(tasks_text('ego_head', True, recorded))
    assert ('Name each as the most specific end state the person evidently worked toward' in rule
            and "Each task's success_predicate states that end state" in rule
            and 'never as a description that any motion would satisfy' in rule and 'the work coming to a stop' in rule)
    assert 'demonstrator' not in rule and 'episode' not in rule
    assert 'Segment the clip into the distinct activities in `tasks`.' in s
