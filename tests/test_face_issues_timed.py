"""Every face_visible data issue carries its times (label/completeness.py face_gaps).

The face blur that reads these labels runs only over the stretches a face issue names. An issue with t_s and end_s
null means the whole recording, so the blur has to run a face detector over every frame of it. In 44 in-the-wild UMI
recordings labelled in parts, 17 of 1,241 face issues came back that way although their own evidence listed the
moments ("Left view at 0, 14, 84, 148, 256 ... s"): pictures on a wall, printed faces on pillows, a cartoon on a
bedspread, each seen on and off. The prompt now asks for each stretch a face is seen as its own timed issue, and a
reply that still reports a face without its times is not a finished annotation and is sent again automatically
(label/harness.py auto_resend), within the same bound as a provider failure.
"""
import json

import pytest

from label import completeness, harness, prompts
from test_completion_gate import TASKS, provider, reply, run_options

INSTANTS = [round(1.5 * k, 3) for k in range(41)]


def face(t_s=12.0, end_s=14.5, category='face_visible', **extra):
    return {'issue': 'A framed portrait with a face is on the wall', 'category': category, 'severity': 'medium',
            't_s': t_s, 'end_s': end_s, 'evidence': 'left view at 12-14.5 s', **extra}


def request(asked=True):
    """A request over 60 s of footage; asked says whether it was sent under the rule (config.face_issues_timed)."""
    return {'annotation_policy': {'version': 1, 'rig': 'handheld_gripper'}, 'given_prompt': None,
            'config': {'timesteps_s': INSTANTS, **({'face_issues_timed': True} if asked else {})}}


def labels(*issues):
    parsed, ok = harness.parse_response(reply(timeline=((0.0, 60.0),), **TASKS, data_issues=list(issues)))
    assert ok
    return parsed


@pytest.mark.parametrize('rig', ['teleop_arms', 'handheld_gripper', 'ego_head'])
def test_every_rig_asks_a_face_issue_for_its_times(rig):
    text = prompts.fixed_instructions(rig, has_instruction=False)
    assert 'never for face_visible' in text


def test_the_handheld_face_rule_asks_one_timed_issue_per_stretch():
    text = prompts.fixed_instructions('handheld_gripper', has_instruction=False)
    assert 'a face that comes and goes is several timed issues, never one issue without times' in ' '.join(text.split())


@pytest.mark.parametrize('t_s,end_s', [(None, None), (12.0, None), (None, 14.5), (14.5, 12.0)])
def test_an_untimed_face_issue_is_a_gap(t_s, end_s):
    gaps = completeness.annotation_gaps(labels(face(), face(t_s=t_s, end_s=end_s)), request())
    assert gaps == ['data issue 1 (face_visible) has no t_s and end_s: each stretch a face is seen is its own '
                    'timed issue']


def test_timed_faces_and_untimed_other_issues_are_no_gap():
    other = {**face(None, None, category='camera_fault'), 'issue': 'the left camera is out of focus'}
    surface = face(None, None, category='surface_glare')
    assert completeness.annotation_gaps(labels(face(), face(30.0, 31.0), other, surface), request()) == []


@pytest.mark.parametrize('category', ['face_visible', 'Face_Visible', 'faces_in_view', 'bystander_face',
                                      'face-visible', 'face visible', 'FaceVisible', 'bystanderFace'])
def test_any_face_category_is_held_to_the_rule(category):
    assert completeness.annotation_gaps(labels(face(None, None, category=category)), request()) != []


@pytest.mark.parametrize('category', ['surface_glare', 'interface_glitch', 'typeface_text', 'facing_away'])
def test_a_word_that_only_contains_face_is_not_a_face_category(category):
    assert completeness.annotation_gaps(labels(face(None, None, category=category)), request()) == []


def test_a_grouped_face_issue_needs_every_occurrence_timed():
    grouped = {**face(10.0, 40.0), 'grouped': {'why': 'one condition'},
               'occurrences': [face(10.0, 12.0), face(None, None)]}
    assert completeness.face_gaps({'data_issues': [grouped]}) != []
    grouped['occurrences'][1] = face(38.0, 40.0)
    assert completeness.face_gaps({'data_issues': [grouped]}) == []


def test_a_record_asked_before_the_rule_is_judged_as_it_was():
    """A delivered label is never turned into a failure by this rule: only requests sent under it are held to it."""
    assert completeness.annotation_gaps(labels(face(None, None)), request(asked=False)) == []


def test_an_untimed_face_reply_is_sent_again_and_the_timed_reply_kept(tmp_path, monkeypatch):
    from test_label import _packed_episode
    ep, _ = _packed_episode(tmp_path)
    out = tmp_path / 'out'
    untimed = reply(timeline=((0.0, 600.0),), **TASKS, data_issues=[face(None, None)])
    timed = reply(timeline=((0.0, 600.0),), **TASKS, data_issues=[face(1.0, 2.0), face(3.0, 3.5)])
    sent = provider(monkeypatch, [untimed, timed])
    assert harness.run_batch([ep], out, **run_options()) == 0
    assert sent.count('final') == 2
    record = json.loads((out / f'{ep.name}.json').read_text())
    assert record['config']['face_issues_timed'] is True
    assert harness.annotation_complete(record)
    assert [(i['t_s'], i['end_s']) for i in record['labels']['data_issues']] == [(1.0, 2.0), (3.0, 3.5)]


@pytest.mark.usefixtures('resends_used_up')
def test_past_its_resends_an_untimed_face_reply_stays_a_named_failure(tmp_path, monkeypatch):
    from test_label import _packed_episode
    ep, _ = _packed_episode(tmp_path)
    out = tmp_path / 'out'
    sent = provider(monkeypatch, [reply(timeline=((0.0, 600.0),), **TASKS, data_issues=[face(None, None)])])
    assert harness.run_batch([ep], out, **run_options()) == 1
    assert sent.count('final') == 1
    record = json.loads((out / f'{ep.name}.json').read_text())
    assert record['labels']['_incomplete'] == [
        'data issue 0 (face_visible) has no t_s and end_s: each stretch a face is seen is its own timed issue']
