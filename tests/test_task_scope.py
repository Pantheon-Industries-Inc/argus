"""Recording controls are reviewed by purpose, without deleting camera work by name."""
import copy

import pytest

from label.task_scope import apply_review, digest


WORK = {'task': 'Arrange sofa cushions', 'start_s': 0, 'end_s': 120,
        'outcome': 'success', 'completed_at_s': 119, 'success_predicate': 'Cushions arranged'}
CONTROL = {'task': 'Adjust the left wrist-mounted camera', 'start_s': 120, 'end_s': 121,
           'outcome': 'unclear', 'completed_at_s': None}


def record(tasks=None):
    import complete_reply
    return {'labels': complete_reply.uninstructed(tasks=copy.deepcopy([WORK, CONTROL] if tasks is None else tasks)),
            'label_schema': 'tasks', 'annotation_policy': {'rig': 'ego_head'}, 'parse_ok': True}


def review(result, *, kind='recording_control', target='capture_rig', human=None):
    tasks = result['labels']['tasks']
    item = {'task_index': len(tasks) - 1, 'task_sha256': digest(tasks[-1]), 'kind': kind,
            'target': target, 'reason': 'The demonstration work has finished; this operates its worn recorder.',
            'evidence': 'At 120s the right hand operates the same wrist-mounted capture unit visible earlier.'}
    return {'version': 1, 'result_sha256': digest(result), 'labels_sha256': digest(result['labels']), 'machine': [item],
            'human': [] if human is None else [{**item, 'kind': human, 'target': 'task_object',
                'reason': 'The assigned task is operating this camera.', 'evidence': 'Reviewer confirms the goal.'}]}


def test_review_separates_capture_action_without_changing_paid_reply():
    source = record(); before = copy.deepcopy(source)
    out = apply_review(source, review(source), rig='ego_head')
    assert source == before
    assert out['labels']['tasks'] == [WORK]
    activity = out['labels']['recording_activities'][0]
    assert activity['source_task'] == CONTROL and activity['activity'] == CONTROL['task']
    assert out['labels']['completion']['task_completed'] == 'success'
    assert out['task_scope_review']['source_labels_sha256'] == digest(before['labels'])


@pytest.mark.parametrize('title', ['Operate a camera to film a product', 'Adjust the milling machine',
    'Repair the left wrist-mounted camera', 'Press the shutter to take the requested photograph'])
def test_names_and_end_position_never_exclude_a_task(title):
    source = record([{**WORK, 'task': title}])
    empty = {'version': 1, 'result_sha256': digest(source), 'labels_sha256': digest(source['labels']), 'machine': [], 'human': []}
    assert apply_review(source, empty, rig='ego_head')['labels']['tasks'] == source['labels']['tasks']
    keep = review(source, kind='work', target='task_object')
    assert apply_review(source, keep, rig='ego_head')['labels']['tasks'] == source['labels']['tasks']


def test_human_work_decision_overrides_machine_capture_decision():
    source = record()
    out = apply_review(source, review(source, human='work'), rig='ego_head')
    assert out['labels']['tasks'] == source['labels']['tasks']
    assert not out['labels'].get('recording_activities')
    assert out['task_scope_review']['decisions'][0]['source'] == 'human'


def test_uncertain_decision_and_unclear_work_outcome_are_preserved():
    source = record([{**WORK, 'outcome': 'unclear', 'completed_at_s': None}, CONTROL])
    out = apply_review(source, review(source, kind='uncertain', target='unknown'), rig='ego_head')
    assert out['labels']['tasks'] == source['labels']['tasks']
    out = apply_review(source, review(source), rig='ego_head')
    assert out['labels']['completion']['task_completed'] == 'unclear'


def test_only_capture_actions_never_turn_into_a_successful_demonstration():
    source = record([CONTROL])
    out = apply_review(source, review(source), rig='ego_head')
    assert out['labels']['tasks'] == []
    assert out['labels']['completion']['task_completed'] is None


@pytest.mark.parametrize('mutation', ['labels', 'hash', 'target', 'reason', 'duplicate', 'index', 'kind'])
def test_invalid_or_stale_review_is_refused(mutation):
    source = record(); checked = review(source); decision = checked['machine'][0]
    if mutation == 'labels': source['labels']['tasks'][0]['task'] = 'Different work'
    if mutation == 'hash': decision['task_sha256'] = 'stale'
    if mutation == 'target': decision['target'] = 'task_object'
    if mutation == 'reason': decision['reason'] = ''
    if mutation == 'duplicate': checked['machine'].append(copy.deepcopy(decision))
    if mutation == 'index': decision['task_index'] = 99
    if mutation == 'kind': decision['kind'] = 'delete'
    with pytest.raises(ValueError): apply_review(source, checked, rig='ego_head')


def test_board_projection_keeps_recording_evidence_out_of_task_counts():
    from board.to_board import convert
    from board.label_projection import shown_tasks
    source = record([WORK, {**WORK, 'task': 'Fold blanket'}, CONTROL])
    out = apply_review(source, review(source), rig='ego_head')
    board = convert(out, dataset='fixture')
    assert len(shown_tasks(board)) == 2
    assert len(board['recording_activities']) == 1
    assert board['recording_activities'][0]['source_task'] == CONTROL


def test_review_can_be_reapplied_but_not_after_a_derived_label_edit():
    source = record(); checked = review(source)
    out = apply_review(source, checked, rig='ego_head')
    assert apply_review(out, checked, rig='ego_head') == out
    out['labels']['tasks'][0]['task'] = 'Changed'
    with pytest.raises(ValueError): apply_review(out, checked, rig='ego_head')


def test_explicit_goal_verdict_is_never_replaced_by_a_work_task_success():
    source = record()
    source['labels']['completion'] = {'basis': 'instruction', 'task_completed': 'failure'}
    out = apply_review(source, review(source), rig='ego_head')
    assert out['labels']['completion'] == source['labels']['completion']


def test_only_reviewed_related_events_are_removed_from_work_penalties():
    from board.to_board import convert
    from board.scores import facts
    source = record()
    source['labels']['key_events'] = [
        {'t_s': 120.5, 'kind': 'attempt', 'label': 'Recorder button did not respond', 'outcome': 'failure'},
        {'t_s': 120.5, 'kind': 'attempt', 'label': 'Dropped the work object', 'outcome': 'failure'}]
    source['labels']['timeline'] = [
        {'start_s': 0, 'end_s': 120, 'action': 'arrange cushions', 'contribution': 'advancing'},
        {'start_s': 120, 'end_s': 121, 'action': 'operate recorder', 'contribution': 'advancing'}]
    checked = review(source)
    checked['machine'][0]['related'] = {'key_events': [0], 'timeline': [1]}
    out = apply_review(source, checked, rig='ego_head')
    assert out['labels']['key_events'] == source['labels']['key_events'][1:]
    assert out['labels']['recording_activities'][0]['source_annotations']['key_events'][0]['row'] == source['labels']['key_events'][0]
    assert out['labels']['timeline'][1]['activity_scope'] == 'recording_control'
    assert facts(convert(out, dataset='fixture'))['failed_attempts'] == 1


def test_terminal_position_is_a_prior_not_a_camera_keyword_classifier():
    from label.task_scope import terminal_task
    assert terminal_task(record()['labels']['tasks'], 1, 121, sample_period_s=0.5)
    assert not terminal_task(record()['labels']['tasks'], 0, 121, sample_period_s=0.5)
    assert not terminal_task(record()['labels']['tasks'], 1, 150, sample_period_s=0.5)
    source = record(); source['config'] = {'timesteps_s': [0.0, 0.5, 120.5, 121.0]}
    out = apply_review(source, review(source, kind='work', target='task_object'), rig='ego_head')
    assert out['task_scope_review']['decisions'][0]['terminal_capture_prior'] is True
    assert out['labels']['tasks'] == source['labels']['tasks']


def test_related_review_cannot_relabel_a_mixed_work_and_capture_step():
    source = record(); checked = review(source)
    checked['machine'][0]['related'] = {'timeline': [0]}
    with pytest.raises(ValueError): apply_review(source, checked, rig='ego_head')


def test_capture_and_work_rows_are_not_joined_across_a_request_cut():
    from label.pieces import _step_match
    left = {'start_s': 0, 'end_s': 10, 'activity_scope': 'work'}
    right = {**left, 'activity_scope': 'recording_control'}
    assert _step_match(left, right, {'t0_s': 0, 't1_s': 10}) is None


def test_cli_refuses_robot_source_and_keeps_source_files_unchanged(tmp_path, monkeypatch):
    import json
    from label.task_scope import main
    source = record(); source['annotation_policy']['rig'] = 'teleop_arms'
    src = tmp_path / 'source'; src.mkdir()
    path = src / 'episode_a.json'; path.write_text(json.dumps(source))
    checked = tmp_path / 'review.json'; checked.write_text(json.dumps({'episodes': {'episode_a': review(source)}}))
    out = tmp_path / 'derived'
    monkeypatch.setattr('sys.argv', ['task_scope', '--input', str(src), '--reviews', str(checked), '--out', str(out)])
    with pytest.raises(ValueError): main()
    assert json.loads(path.read_text()) == source and not out.exists()


def test_cli_is_resumable_and_preserves_paid_source(tmp_path, monkeypatch):
    import json
    from label.task_scope import main
    source = record(); src = tmp_path / 'source'; src.mkdir()
    path = src / 'episode_a.json'; path.write_text(json.dumps(source))
    checked = tmp_path / 'review.json'; checked.write_text(json.dumps({'episodes': {'episode_a': review(source)}}))
    out = tmp_path / 'derived'
    monkeypatch.setattr('sys.argv', ['task_scope', '--input', str(src), '--reviews', str(checked), '--out', str(out)])
    main(); before = (out / path.name).read_bytes(); main()
    assert (out / path.name).read_bytes() == before and json.loads(path.read_text()) == source
