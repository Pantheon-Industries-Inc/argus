import json

import pytest

from gate import score


def test_issue_requires_severity_only_when_case_states_threshold():
    labels = {'data_issues': [{'category': 'camera_fault', 'issue': 'The camera image is frozen.'}]}
    assert not score.judge({'issue': ['camera_fault']}, {'data_issues': [{'category': 'camera_fault'}]})[0]
    assert score.judge({'issue': ['camera_fault']}, labels)[0]
    assert not score.judge({'issue': ['camera_fault'], 'sev': 'low'}, labels)[0]


def test_named_matches_literal_words_and_phrases_only_where_asserted():
    assert not score.judge({'named': ['card']}, {'task_summary': 'cardboard'})[0]
    assert not score.judge({'named': ['card']}, {'task_summary': 'does not handle a card'})[0]
    assert not score.judge({'named': ['sew']}, {'task_summary': 'sewing fabric'})[0]
    assert score.judge({'named': ['lotus root']}, {'task_summary': 'Peels LOTUS ROOT.'})[0]
    # a denial in one field does not hide the word asserted in another, and an unrelated denial does not cancel it
    assert score.judge({'named': ['sew']}, {'task_summary': 'The operator does not sew.',
                                            'timeline': [{'action': 'sew', 'object': 'cloth'}]})[0]
    assert score.judge({'named': ['card']}, {'task_summary': 'Picks up the card but does not drop it.'})[0]
    for denial in ('does not sew', 'The operator never sews the cloth', 'no sewing happens'):
        word = 'sews' if 'sews' in denial else 'sewing' if 'sewing' in denial else 'sew'
        assert not score.judge({'named': [word]}, {'timeline': [{'action': denial, 'object': 'cloth'}]})[0]


@pytest.mark.parametrize('action', [
    'picks up chopsticks with no slip',
    'given a fork, picks up chopsticks',
    'Pick up chopsticks instead of fork',
    'the fork, not chopsticks',
])
def test_free_prose_prohibited_object_mentions_need_review(action):
    passed, why = score.judge({'held': {'named': 'fork', 'not_named': 'chopstick'}},
                             {'timeline': [{'object': 'fork'}, {'action': action}]})
    assert not passed and any(reason.startswith('needs review:') for reason in why)
    assert not any(reason.startswith('claims ') for reason in why)


def test_held_needs_positive_structured_claim_and_reports_explicit_prohibited_object():
    case = {'held': {'named': 'fork', 'not_named': 'chopstick'}}
    assert score.judge(case, {'timeline': [{'object': 'fork'}]})[0]
    assert not score.judge(case, {'task_summary': 'handles a fork'})[0]
    passed, why = score.judge(case, {'timeline': [{'object': 'chopsticks'}]})
    assert not passed and any(reason.startswith('claims chopstick') for reason in why)


def test_duplicate_episode_across_runs_is_rejected_with_both_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(score, 'load_selection', lambda: {'ego': [{'name': 'episode_x', 'role': 'case'}]})
    monkeypatch.setattr(score, 'load_cases', lambda: {})
    runs = [tmp_path / name for name in ('first', 'second')]
    for run in runs:
        (run / 'out').mkdir(parents=True)
        (run / 'out/episode_x.json').write_text('{}')
    with pytest.raises(ValueError) as error:
        score.score(runs)
    message = str(error.value)
    assert 'episode_x' in message and all(str(run) in message for run in runs)


def test_unresolved_held_fact_is_counted_separately_and_does_not_pass(tmp_path, monkeypatch):
    ep = tmp_path / 'episode_x'
    ep.mkdir()
    (ep / 'context.json').write_text('{"duration_s": 1}')
    run = tmp_path / 'run'
    (run / 'out').mkdir(parents=True)
    (run / 'out/episode_x.json').write_text(json.dumps({
        'parse_ok': True, 'episode_dir': str(ep),
        'labels': {'timeline': [{'start_s': 0, 'end_s': .5, 'object': 'fork'},
                                {'start_s': .5, 'end_s': 1, 'action': 'the fork, not chopsticks'}],
                   'tasks': [{'task': 'set the table', 'outcome': 'success'}]}}))
    monkeypatch.setattr(score, 'load_selection', lambda: {'ego': [{'name': 'episode_x', 'role': 'case'}]})
    monkeypatch.setattr(score, 'load_cases', lambda: {'episode_x': {
        'held': {'named': 'fork', 'not_named': 'chopstick'}}})
    result = score.score([run])['ego']
    assert (result['cases'], result['met'], result['unresolved']) == (1, 0, 1)
    assert result['failures'][0].startswith('episode_x: needs review:')


def test_published_cases_accept_explicit_word_variants_without_substring_matches():
    cases = score.load_cases()
    cards = score.case_for('episode_8e27b71252974297b559c45bfd18f8fa', cases)
    assert score.judge(cards, {'task_summary': 'Collect playing cards and put them in their box.'})[0]
    assert not score.judge(cards, {'task_summary': 'Move a cardboard box.'})[0]
    sewing = score.case_for('episode_factory_042_worker_061_0059', cases)
    assert score.judge(sewing, {'tasks': [{'task': 'Sewing the fabric panels.'}]})[0]


@pytest.mark.parametrize('outcome,cost,hold,measured,complete', [
    ('received', .02, 0, 1, True),
    ('unverified', None, .8, 0, False),
    ('not dispatched', None, 0, 0, True),
])
def test_cost_sample_excludes_unreceived_and_unverified_calls(tmp_path, monkeypatch,
                                                            outcome, cost, hold, measured, complete):
    ep = tmp_path / 'episode_x'; ep.mkdir()
    (ep / 'context.json').write_text('{"duration_s": 60}')
    run = tmp_path / 'run'; (run / 'out').mkdir(parents=True)
    record = {'episode_dir': str(ep), 'final_dispatch_outcome': outcome,
              'parse_ok': outcome == 'received', 'labels': {},
              'final_reserved_usd': hold, 'usage': {'est_cost_usd': cost},
              'evidence_inspection': {'cost_usd': .003} if outcome != 'not dispatched' else {}}
    (run / 'out' / ('episode_x.json' if measured else 'noreply_episode_x.json')).write_text(json.dumps(record))
    monkeypatch.setattr(score, 'load_cases', lambda: {})
    monkeypatch.setattr(score, 'load_selection', lambda: {'ego': [{'name': 'episode_x', 'role': 'cost:fixture'}]})
    result = score.score([run])['ego']['cost_sample']
    assert result['episodes'] == measured
    assert result['cost_complete'] is complete
    assert result['reserved_usd'] == pytest.approx(hold)
    assert result['billed_usd'] == pytest.approx((cost or 0) + (.003 if outcome != 'not dispatched' else 0))
    if not measured:
        assert result['billed_per_hour'] is None and result['cold_per_hour'] is None


def test_scoring_unrelated_run_does_not_report_empty_success(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(sys, 'argv', ['gate score', str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        score.main()
    assert error.value.code == 2
