"""The shared instructions (label/prompts.py fixed_instructions) by what the dataset gives: a robot episode with an
instruction keeps main's episode prompt word for word and adds its list of tasks; one without asks only for tasks;
the dataset's own timed steps add the rules for judging a task against its step; and the head camera keeps main's
prompt except for three approved edits. tests/fixtures/main_prompts.json holds main's (58c452e) text."""
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
    """main's text with the approved edits every rig shares (MISSED_GRASPS), and on the robot rigs the edits for
    contact without holding (UNHELD_CONTACT)."""
    text = MAIN[f'{rig}/{"recorded" if recorded else "video"}']
    assert text.count(REPETITION) == 1
    actor = {'teleop_arms': 'arm', 'handheld_gripper': 'gripper', 'ego_head': 'hand'}[rig]
    text = text.replace(REPETITION, MISSED_GRASPS.format(actor=actor))
    if rig != 'ego_head':
        for old, new in UNHELD_CONTACT + (UNHELD_CONTACT_HANDHELD if rig == 'handheld_gripper' else []):
            assert text.count(old) == 1, old[:60]
            text = text.replace(old, new)
    for old, new in footage_tiers(rig) + ([MANNER_SEVERITY] if rig != 'ego_head' else []):
        assert text.count(old) == 1, old[:60]
        text = text.replace(old, new)
    return text


# Part of the same edit: the instruction rules graded a difference in manner by "its training impact", the scale the
# consequence tiers replaced.
MANNER_SEVERITY = ('instruction_mismatch whose severity is its training impact. A detail',
                   'instruction_mismatch whose severity is its consequence for the footage. A detail')


# An approved edit to every rig (2026-10-31): severity is the consequence for the footage, and every data issue says
# what its span shows (footage). An operator holding a lit phone for 9s was filed as a low idle_stretch and shipped,
# since low read as "a model would barely be affected" rather than "the span is still task data". The test holds its
# own copy of the wording, as for the edits above.
FOOTAGE_WORDS = {
    'teleop_arms': 'task | idle | off_task | operator_absent | operator_phone | rig_handling | bare_hand',
    'handheld_gripper': ('task | idle | off_task | operator_absent | operator_phone | rig_handling | bare_hand | '
                         'grippers_off'),
    'ego_head': 'task | idle | off_task | operator_absent | operator_phone | rig_handling',
}
FOOTAGE_RIG = {
    'teleop_arms': """  bare_hand: a person's bare hand moves or works a task object in the workspace, unless the episode's own
  description makes a person part of the task.
""",
    'handheld_gripper': """  bare_hand: the operator moves or works a task object with a bare hand instead of a gripper; fingers on or
  beside a gripper that hand is holding are how the rig is worn, and are task. grippers_off: both grippers
  are set down with no hand on them while the recording runs; one gripper set down while the other works is
  task.
""",
    'ego_head': '',
}


def footage_tiers(rig):
    return [
        ('     "severity": "low" | "medium" | "high",\n     "t_s": <float when it starts, or null if it spans the episode>,',
         '     "severity": "low" | "medium" | "high",\n     "footage": "<what the span shows: ' + FOOTAGE_WORDS[rig]
         + ', or your own snake_case word when none fits>",\n'
         '     "t_s": <float when it starts, or null if it spans the episode>,'),
        ("""- SEVERITY IS TRAINING IMPACT, not how visible or dramatic something looks, in both lists. It comes from
  two things together:""",
         """- SEVERITY IS THE CONSEQUENCE FOR THE FOOTAGE, not how visible or dramatic something looks, in both lists. It
  comes from two things together:"""),
        ("""high: the episode cannot be used as it is; blindly training on it would
  teach something wrong at its core.""",
         """high: the episode is unusable, whatever is trimmed; blindly training on it
  would teach something wrong at its core."""),
        ("""every frame inherits it. medium: a bounded part of the episode is wrong or wasted, and the rest is good
  once that part is trimmed, masked or corrected. low: worth knowing, but a model trained on the episode as
  it is would barely be affected.""",
         """every frame inherits it. medium: the span from t_s to end_s is not usable task data and is cut out, and the
  rest of the episode is good once it is; a medium issue with no time loses the whole episode, so time every
  bounded problem. low: cosmetic; the footage in the span is still usable task data as it is, and a model
  trained on it would barely be affected."""),
        ("""  a real problem away because some kind of training might tolerate it.
""",
         """  a real problem away because some kind of training might tolerate it.
- FOOTAGE SAYS WHAT EACH DATA ISSUE'S SPAN SHOWS. task: the operator is doing the task, however imperfectly,
  and the issue is about how it was recorded, framed or labelled. idle: work on every task has stopped for longer
  than a brief pause to look, think or shift a grip, with nothing in progress. off_task: the operator does
  something that is not the task, such as eating, drinking or tidying something unrelated. operator_absent: the
  operator is out of the scene while the recording runs. operator_phone: the operator uses a phone or another
  personal device. rig_handling: the operator handles the recording equipment instead of the task, such as
  tapping the recording controller or its screen, starting or stopping the recording, or fixing a camera or
  strap; a phone-like recording controller in the operator's hand is rig_handling, not operator_phone.
""" + FOOTAGE_RIG[rig] + """  Another person reaching in while the operator keeps working is task footage, graded like any other
  issue. When none of these fits, write your own snake_case word. Footage other than task is never usable task
  data, so its severity is at least medium, and its span is cut from the episode.
"""),
    ]


# An approved edit to the shared contribution rule (Eric, 2026-10-23): runs of missed grasps on a fine object were
# labelled as adjusting or repositioning. The sentence on repetition goes on to say a real correction advances, that
# returns to an object that never get their result are failed attempts, and gives missed grasps as one example among
# many.
REPETITION = '  Repeated failed attempts at the same aim are wasteful, but repetition alone is not evidence of failure.\n'
MISSED_GRASPS = '''  Repeated failed attempts at the same aim are wasteful, but repetition alone is not evidence of failure: a few
  seconds spent straightening a piece already set in a bin, or turning a part into line, gets what it aims at and
  advances, however small the motions are. Equally, do not be overly lenient. When the {actor} keeps returning to
  the same object without ever getting the result those returns evidently aim at, each return is a failed attempt.
  One example among many, not a complete list: repeated missed grasps of a particularly fine object can look like
  small adjustments, but naming them "adjust", "reposition", "align" or "nudge" does not make them succeed; if the
  frames never show the object held, each try aimed at holding it failed. Those words fit whenever the object does
  end up where it was evidently meant to be.
'''


# An edit to the robot rigs' prompts (2026-10-24): every example action was a grasp-based phase, and the handheld
# gripper's contract named the gripper's housing as outside the acting part, so a gripper that pushed, poked or
# knocked an object it did not hold had no action to label it with. On a DiverseUMI take where the operator knocked a
# sponge off a cup again and again, Sol 6.1 labelled the span as rubbing the cup and Qwen3.8 Max as reseating the
# sponge, both as progress. The test holds its own copy, as for the edits above.
UNHELD_CONTACT = [
    ('release, a retreat). Do not collapse a multi-step activity into one',
     '''release, a retreat). Grasping is one way to act on an object, not the
only one: the end effector also acts through contact without holding,
with any part of itself, not only the part that holds. Such a contact
is an action phase like any other, named for what it does to
the object and judged by where the object ends up. Do not collapse a multi-step activity into one'''),
    ('  frames that are the same ongoing action.\n',
     '''  frames that are the same ongoing action. A contact that moves an object without holding it is
  such an event: each one is its own segment, and one that leaves the object where it was evidently
  not meant to be is an error, however brief.
'''),
]
UNHELD_CONTACT_HANDHELD = [
    ("  a demonstration collected to train robots. Gripper bodies and fingers, and people's hands, arms\n",
     "  a demonstration collected to train robots. Gripper bodies and jaws, and people's hands, arms\n"),
    ("  its own fingers and handle or strap, independent motion and an object held in its fingers are\n",
     "  its own jaws and handle or strap, independent motion and an object held in its jaws are\n"),
    ('''  gripper-only execution contract treats the fingers as the acting part, subject to explicit source task or actor rules.
  A task object moved by a bare hand, arm or the gripper's housing outside that contract is worth
  reporting: its change was not made by the declared acting part. This is an annotation contract,
  not an inferred limit of a physical robot. Anyone else touching the scene is reported as usual.
''',
     '''  gripper-only execution contract treats the whole gripper as the acting part, subject to explicit source task or actor rules:
  any part of it a robot carrying the same gripper would also have, such as its jaws, their tips and
  outer faces, or its housing, acts on objects whether or not it holds them. A task object moved by a
  person's bare hand, arm or body is outside that contract and worth reporting: its change was not made
  by the declared acting part. This is an annotation contract, not an inferred limit of a physical
  robot. Anyone else touching the scene is reported as usual.
'''),
]


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


# The second approved edit, word for word as PR 107 left it: a clip with no annotated goal names each task's end state.
# The test holds its own copy, so a change to it fails here until the copy is changed with an approved edit.
EGO_NO_GOAL = """
ABOUT THE TASKS. This clip comes with no annotated goal, so the end state of each of its tasks is yours to infer. Name
each as the most specific end state the person evidently worked toward (the blocks in a row, a tower, the cloth folded
in half, the cup in the bin), read from what their actions were evidently trying to achieve and not from how far they
got, never as a description that any motion would satisfy (moving, repositioning or rearranging the objects, or the
work coming to a stop). Each task's success_predicate states that end state, and whatever the work was evidently meant
to cover but did not is a shortfall in the outcome, not something outside the task. A task still under way when the
clip ends is named by the end state of the work the clip shows under way, and is unclear when that end is not shown; a
further step that would follow that work (setting the result down, packing, boxing or storing it, carrying it away) is
part of the task only when the clip shows it begin. When the actions fit no single end state, name the one that fits
best and say in the task's note what leaves it uncertain.
"""
# The third approved edit (Eric, 2026-10-21, audit_2026-10-21 S3): a clip with an annotated goal is asked one optional
# field, whether it reached the goal's end state, and nothing else changes.
EGO_GOAL_OUTCOME = """
THE ANNOTATED GOAL'S OUTCOME. Because the annotation gives a goal, also emit a top-level "goal_outcome" object saying
whether the clip reached that goal's end-state:
  {"outcome": "success" | "partial" | "failure" | "unclear",
   "success_predicate": "<the end-state that means the annotated goal is done>",
   "completed_at_s": <float: when that end-state is first reached, or null>,
   "note": "<why, if partial, failure or unclear>"}
Judge it against the annotated goal alone, from what the clip shows, whatever tasks you segment: success when the frames
show the goal's end-state; partial when only part of it is reached; failure when the frames show it is not reached,
including when the person does different work; unclear when the clip ends before it can be told or the view hides it.
The success_predicate is the whole end-state the goal names, never narrowed to the part the clip shows: when the goal's
work is still under way at the clip's end (more of its items left to do), the outcome is unclear, never a success at
the first finished item.
Keep it consistent with goal_alignment: success never sits beside a "different" or "unrelated" relation.
"""


@pytest.mark.parametrize('given', [True, False])
@pytest.mark.parametrize('recorded', [True, False])
@pytest.mark.parametrize('steps', [True, False])
def test_the_head_camera_prompt_is_mains_but_for_the_three_approved_edits(given, recorded, steps):
    """main's text, the softened framing, and in the slot before the timeline note either the goal's outcome (a clip
    with an annotated goal) or the end-state rule (one without), each pinned here as a literal: B6, the earlier test
    took the end-state block from the function under test, so PR 107 rewrote it and the test still passed."""
    expected = main('ego_head', recorded)
    for old, new in EGO_SOFTENED:
        assert expected.count(old) == 1, old[:60]
        expected = expected.replace(old, new)
    lean = prompts.lean('ego_head')
    assert expected.count(lean) == 1
    expected = expected[:expected.index(lean)] + (EGO_GOAL_OUTCOME if given else EGO_NO_GOAL) + lean
    assert text('ego_head', given, recorded, steps) == expected
    # the goal's outcome is asked only with a goal, the end-state rule only without one
    assert ('"goal_outcome"' in text('ego_head', given, recorded, steps)) is given
    assert ('This clip comes with no annotated goal' in text('ego_head', given, recorded, steps)) is not given


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


# An approved edit (Eric, 2026-10-23) to the handheld gripper's prompt only: after what keeps the task observable, a
# check of which hand each gripper camera is on. 317 of the 4,609 DiverseUMI 1K recordings have their two wrist
# streams exchanged at capture, and the labels named it on 272 of them, unasked; the prompt only listed "swapped"
# among a camera's faults. The test holds its own copy, as for the head camera's edits.
HANDHELD_SIDES_AFTER = '  the task state at such a moment cannot be established from any view, and the task did not intend it.\n'
HANDHELD_SIDES = """- WHICH HAND EACH GRIPPER CAMERA IS ON. Where the dataset names the hand that carries each of two gripper
  cameras, check that naming. The camera on the left hand sees the other gripper toward the right of its
  image, and the camera on the right hand toward the left. A top or side camera can also help, since when
  one hand makes a clear move there, the view that moves with it is that hand's camera. If both gripper
  views clearly show the opposite of their names across most of the recording, report a data issue tagged
  camera_side_mismatch, with two or three moments as evidence. When unsure, report nothing.
"""


def _robot_expected(rig, recorded, steps):
    """Main's episode prompt with the list of tasks after its completion, the task rules (and the step rules) before
    the granularity note, and that note naming the tasks; the handheld gripper's also has its camera side check."""
    expected = main(rig, recorded)
    if rig == 'handheld_gripper':
        assert expected.count(HANDHELD_SIDES_AFTER) == 1
        expected = expected.replace(HANDHELD_SIDES_AFTER, HANDHELD_SIDES_AFTER + HANDHELD_SIDES)
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


@pytest.mark.parametrize('rig', RIGS)
@pytest.mark.parametrize('given,steps', STRUCTURES)
@pytest.mark.parametrize('recorded', [True, False])
def test_only_the_handheld_gripper_is_asked_which_hand_each_camera_is_on(rig, given, steps, recorded):
    """Once, in the handheld gripper's prompt with or without an instruction, steps or recorded motion; never in a
    teleoperated arm's or a head camera's."""
    assert text(rig, given, recorded, steps).count('WHICH HAND EACH GRIPPER CAMERA IS ON') == (rig == 'handheld_gripper')


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
