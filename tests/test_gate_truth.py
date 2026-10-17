import json

import pytest

from gate import score


def test_issue_requires_severity_only_when_case_states_threshold():
    labels = {'data_issues': [{'category': 'camera_fault'}]}
    assert score.judge({'issue': ['camera_fault']}, labels)[0]
    assert not score.judge({'issue': ['camera_fault'], 'sev': 'low'}, labels)[0]


def test_named_matches_literal_words_and_phrases_even_when_negated():
    assert not score.judge({'named': ['card']}, {'task_summary': 'cardboard'})[0]
    assert score.judge({'named': ['card']}, {'task_summary': 'does not handle a card'})[0]
    assert not score.judge({'named': ['sew']}, {'task_summary': 'sewing fabric'})[0]
    assert score.judge({'named': ['lotus root']}, {'task_summary': 'Peels LOTUS ROOT.'})[0]


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
        'labels': {'timeline': [{'object': 'fork'}, {'action': 'the fork, not chopsticks'}]}}))
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
