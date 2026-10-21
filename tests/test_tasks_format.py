"""Every request asks for a list of tasks (label/prompts.py), whatever task mode a source declares or a saved policy
names, and the recording's completion is derived from its tasks when the reply arrives (label/harness.py read_reply,
board/label_projection.py recording_completion). A record labelled before keeps what it was asked and given."""
import json

import pytest

import complete_reply
from label import episode as me, harness, policy, prompts
from test_label import _packed_episode

TASK = {'task': 'stack the cups', 'objects': ['cup'], 'judged_against': 'own_end_state', 'outcome': 'success',
        'success_predicate': 'the cups stacked', 'completed_at_s': 1.0, 'goal_reached_at_s': 1.0}


def _reply(tasks, **fields):
    labels = complete_reply.labels('Stack the cups.')
    assert 'completion' not in labels
    lo, hi = labels['timeline'][0]['start_s'], labels['timeline'][0]['end_s']
    labels['tasks'] = [{'start_s': lo, 'end_s': hi, **t} for t in tasks]
    labels.update(fields)
    return labels


def _label(tmp_path, monkeypatch, reply, *, instruction='', saved_mode=None, name='label.json'):
    ep, _ = _packed_episode(tmp_path)
    ctx = json.loads((ep / 'context.json').read_text())
    ctx['instruction'] = instruction
    ctx['task_mode'] = 'session'            # a source declaration: provenance only
    if saved_mode:
        ctx['annotation_policy'] = {**policy.resolve(ctx), 'task_mode': saved_mode}
    (ep / 'context.json').write_text(json.dumps(ctx))
    sent = []

    def provider(content, *_args, **_options):
        sent.append(content)
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(reply)}}],
                'usage': {'cost': 0.0}}
    monkeypatch.setattr(harness, 'call_model', provider)
    out = tmp_path / name
    harness.label_episode(ep, out, model='openai/gpt-6-astra', reasoning='medium', api_key='sk-or-fixture',
                          max_tokens=2000, timeout=60, cell_w=224)
    return json.loads(out.read_text()), sent[-1][0]['text']


@pytest.mark.parametrize('saved_mode', [None, 'episode', 'session'])
def test_every_request_asks_for_tasks_and_one_task_is_the_recordings_verdict(tmp_path, monkeypatch, saved_mode):
    record, fixed = _label(tmp_path, monkeypatch, _reply([TASK]), saved_mode=saved_mode)
    assert fixed.endswith(prompts.lean('teleop_arms')) and prompts.TASK_RULES in fixed
    assert '"completion"' not in fixed and 'ABOUT THE RECORDING\'S INSTRUCTION' not in fixed
    assert record['label_schema'] == 'tasks' and record['prompt_mode'] == 'inferred' and record['parse_ok']
    assert record['labels']['completion'] == {
        'task_completed': 'success', 'success_predicate': 'the cups stacked', 'completed_at_s': 1.0,
        'goal_reached_at_s': 1.0, 'undone_at_s': None, 'undone_by': None, 'reason': None,
        'judged_against': 'own_end_state', 'from_tasks': 1}
    # a saved legacy mode is kept for the job's identity; it never shapes the request
    assert record['annotation_policy'].get('task_mode') == saved_mode
    assert '_schema_violations' not in record['labels']


def test_several_tasks_give_no_recording_verdict_and_a_reply_completion_is_replaced(tmp_path, monkeypatch):
    reply = _reply([TASK, {**TASK, 'outcome': 'failure', 'completed_at_s': None, 'goal_reached_at_s': None}],
                   completion={'task_completed': 'success'},
                   goal_alignment={'relation': 'narrower', 'matches_given': True})
    record, fixed = _label(tmp_path, monkeypatch, reply, instruction='Stack the cups.')
    assert record['prompt_mode'] == 'given' and 'ABOUT THE EPISODE\'S INSTRUCTION' in fixed
    completion = record['labels']['completion']
    assert completion['task_completed'] is None and completion['from_tasks'] == 2
    assert any(v.startswith('completion:') for v in record['labels']['_schema_violations'])
    # a saved record read again on resume is still complete and gains nothing
    assert harness.annotation_problems(record) == []


def test_a_recording_with_an_instruction_and_no_task_is_incomplete_and_one_without_is_not(tmp_path, monkeypatch):
    record, _ = _label(tmp_path, monkeypatch, _reply([]), instruction='Stack the cups.', name='given.json')
    assert record['parse_ok'] is False and 'lists none' in ' '.join(record['labels']['_incomplete'])
    (tmp_path / 'b').mkdir()
    record, _ = _label(tmp_path / 'b', monkeypatch, _reply([]), name='bare.json')
    assert record['parse_ok'] is True
    assert record['labels']['completion'] == {**{k: None for k in record['labels']['completion']
                                                 if k not in ('reason', 'from_tasks')},
                                              'reason': 'no task observed', 'from_tasks': 0}


def test_a_record_labelled_before_the_tasks_format_is_read_as_it_was_asked():
    """No label_schema: an episode's reply keeps its own completion, no completion is derived, and a session's
    reply is held to its tasks."""
    labels = complete_reply.labels('Stack the cups.', completion={'task_completed': 'success'}, tasks=[])
    raw = json.dumps(labels)
    for mode, field in (('episode', 'completion.task_completed'), ('session', 'tasks')):
        record = {'annotation_policy': {'rig': 'teleop_arms', 'task_mode': mode}, 'given_prompt': 'Stack the cups.',
                  'prompt_mode': 'session' if mode == 'session' else 'given'}
        read, ok = harness.read_reply(raw, record)
        assert read['completion'] == labels['completion'] and 'from_tasks' not in read['completion']
        assert ok is (mode == 'episode'), field


@pytest.mark.parametrize('rig', ['teleop_arms', 'handheld_gripper', 'ego_head'])
def test_the_shared_prefix_is_the_same_for_every_model_and_task_mode(tmp_path, rig):
    ep, _ = _packed_episode(tmp_path)
    ctx = json.loads((ep / 'context.json').read_text())
    ctx['profile'] = rig
    prefixes = set()
    for model in (policy.ASTRA, policy.SOL61):
        for mode in (None, 'episode', 'session'):
            ctx['annotation_policy'] = policy.resolve(ctx, model=model, task_mode=mode)
            (ep / 'context.json').write_text(json.dumps(ctx))
            request = me.build_request(ep, cell_w=128)
            prefixes.add(request['content'][0]['text'])
    assert len(prefixes) == 1
