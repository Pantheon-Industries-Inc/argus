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
    labels = complete_reply.uninstructed('Stack the cups.')
    assert 'completion' not in labels
    lo, hi = labels['timeline'][0]['start_s'], labels['timeline'][0]['end_s']
    labels['tasks'] = [{'start_s': lo, 'end_s': hi, **t} for t in tasks]
    labels.update(fields)
    return labels


def _label(tmp_path, monkeypatch, reply, *, instruction='', saved_mode=None, name='label.json', steps=None):
    ep, _ = _packed_episode(tmp_path)
    ctx = json.loads((ep / 'context.json').read_text())
    ctx['instruction'] = instruction
    if steps:
        ctx['annotation_subtasks'] = steps
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
def test_with_neither_an_instruction_nor_timed_steps_one_task_is_the_recordings_verdict(tmp_path, monkeypatch,
                                                                                         saved_mode):
    record, fixed = _label(tmp_path, monkeypatch, _reply([TASK]), saved_mode=saved_mode)
    assert fixed.endswith(prompts.lean('teleop_arms', has_instruction=False))
    assert '\nTASKS. Segment the recording into its tasks' in fixed and '"completion"' not in fixed
    assert record['label_schema'] == 'tasks' and record['prompt_mode'] == 'inferred' and record['parse_ok']
    assert record['task_prompts'] == 0
    assert record['labels']['completion'] == {
        'task_completed': 'success', 'success_predicate': 'the cups stacked', 'completed_at_s': 1.0,
        'goal_reached_at_s': 1.0, 'undone_at_s': None, 'undone_by': None, 'reason': None,
        'judged_against': 'own_end_state', 'from_tasks': 1, 'basis': 'one_task'}
    # a saved legacy mode is kept for the job's identity; it never shapes the request
    assert record['annotation_policy'].get('task_mode') == saved_mode
    assert '_schema_violations' not in record['labels']


def test_with_an_instruction_the_reply_judges_the_recording_itself_and_lists_its_tasks(tmp_path, monkeypatch):
    """An asked task that fails, then unasked work that succeeds: the recording is a failure, as the episode was."""
    asked = {**TASK, 'judged_against': 'instruction', 'outcome': 'failure', 'completed_at_s': None,
             'goal_reached_at_s': None}
    reply = _reply([asked, {**TASK, 'task': 'build a tower'}],
                   completion={'task_completed': 'failure', 'reason': 'the cups never stack'},
                   goal_alignment={'relation': 'narrower', 'matches_given': True})
    record, fixed = _label(tmp_path, monkeypatch, reply, instruction='Stack the cups.')
    assert record['prompt_mode'] == 'given' and 'ABOUT THE EPISODE\'S INSTRUCTION' in fixed
    assert prompts.INSTRUCTION_RULES in fixed and '"task_completed"' in fixed and '"tasks": [' in fixed
    assert record['labels']['completion'] == {'task_completed': 'failure', 'reason': 'the cups never stack',
                                              'basis': 'instruction'}
    assert '_schema_violations' not in record['labels'] and len(record['labels']['tasks']) == 2
    # a saved record read again on resume is still complete and gains nothing
    assert harness.annotation_problems(record) == []
    # the completion is required beside the tasks, as the episode prompt required it
    (tmp_path / 'b').mkdir()
    bare, _ = _label(tmp_path / 'b', monkeypatch, _reply([asked]), instruction='Stack the cups.', name='bare.json')
    assert bare['parse_ok'] is False
    assert bare['labels']['_incomplete'] == ['the request asks the outcome of its task, and completion.task_completed '
                                             'is missing']


def test_with_timed_steps_and_no_instruction_each_task_is_judged_against_its_step_and_there_is_no_verdict(
        tmp_path, monkeypatch):
    steps = [{'t0': 0.0, 't1': 1.5, 'label': 'pick up the cup'}, {'t0': 1.5, 't1': 3.9, 'label': 'stack the cup'}]
    task = {**TASK, 'judged_against': 'subtask', 'subtask': 2}
    reply = _reply([task], goal_alignment={'relation': 'aligned', 'matches_given': True})
    record, fixed = _label(tmp_path, monkeypatch, reply, steps=steps)
    assert record['task_prompts'] == 2 and record['parse_ok']
    assert "ABOUT THE DATASET'S TIMED STEPS" in fixed
    assert "comparing the dataset's steps, as one sequence, with the tasks you observed" in ' '.join(fixed.split())
    completion = record['labels']['completion']
    assert completion['task_completed'] is None and completion['basis'] == 'task_prompts'
    assert record['labels']['tasks'][0]['judged_against'] == 'subtask'
    # with timed steps a recording holds at least one task
    (tmp_path / 'b').mkdir()
    empty, _ = _label(tmp_path / 'b', monkeypatch, _reply([]), steps=steps, name='empty.json')
    assert empty['parse_ok'] is False


def test_with_neither_a_robot_recording_may_hold_no_task(tmp_path, monkeypatch):
    record, _ = _label(tmp_path, monkeypatch, _reply([]), name='bare.json')
    assert record['parse_ok'] is True
    assert record['labels']['completion'] == {**{k: None for k in record['labels']['completion']
                                                 if k not in ('reason', 'from_tasks', 'basis')},
                                              'reason': 'no task observed', 'from_tasks': 0, 'basis': 'tasks'}


def test_a_record_labelled_before_the_tasks_format_is_read_as_it_was_asked():
    """No label_schema: an episode's reply keeps its own completion, no completion is derived, and a session's
    reply is held to its tasks."""
    labels = complete_reply.labels('Stack the cups.', completion={'task_completed': 'success'}, tasks=[])
    raw = json.dumps(labels)
    for mode, field in (('episode', 'completion.task_completed'), ('session', 'tasks')):
        record = {'annotation_policy': {'rig': 'teleop_arms', 'task_mode': mode}, 'given_prompt': 'Stack the cups.',
                  'prompt_mode': 'session' if mode == 'session' else 'given'}
        read, ok = harness.read_reply(raw, record)
        assert read['completion'] == labels['completion'] and 'basis' not in read['completion']
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


@pytest.mark.parametrize('claim, instruction, task_prompts', [('instruction', False, 0), ('subtask', True, 0)])
def test_a_task_is_never_judged_against_a_claim_the_recording_does_not_carry(claim, instruction, task_prompts):
    # a 550s freeform teleop recording (no instruction, no steps) came back with both tasks "judged against the
    # instruction": the claim it names does not exist, so the task is judged against its own end state, and said so
    from board.label_projection import derive_completion
    labels = {'tasks': [{'task': 'move the cloths', 'outcome': 'success', 'judged_against': claim, 'subtask': 2},
                        {'task': 'stack the lids', 'outcome': 'success', 'judged_against': 'own_end_state'}]}
    out = derive_completion(labels, rig='teleop_arms', instruction=instruction, task_prompts=task_prompts)
    assert [t['judged_against'] for t in out['tasks']] == ['own_end_state', 'own_end_state']
    assert out['tasks'][0]['subtask'] is None
    assert any('task 0 said it was judged against' in n for n in out['_schema_violations'])
    # read again (a saved record checked on resume), nothing changes and nothing is named twice
    again = derive_completion(out, rig='teleop_arms', instruction=instruction, task_prompts=task_prompts)
    assert again['tasks'] == out['tasks'] and again['_schema_violations'] == out['_schema_violations']


def test_a_task_judged_against_a_claim_the_recording_carries_is_kept():
    from board.label_projection import derive_completion
    labels = {'completion': {'task_completed': 'success'},
              'tasks': [{'task': 'fold', 'outcome': 'success', 'judged_against': 'instruction'},
                        {'task': 'step', 'outcome': 'success', 'judged_against': 'subtask', 'subtask': 1}]}
    out = derive_completion(labels, rig='teleop_arms', instruction=True, task_prompts=3)
    assert [t['judged_against'] for t in out['tasks']] == ['instruction', 'subtask']
    assert not any('said it was judged against' in n for n in out.get('_schema_violations') or [])
