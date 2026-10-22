"""The seam check (label/seam.py): which cuts it asks about, what one request holds and costs, how strictly its answer
is read, and how its verdicts join a stitched recording's tasks. No test makes a model call."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from label import episode as me
from label import harness, pieces, seam
from prepare import formats as f
from test_formats import recorder_folder

CUT = 304.833


def _source(index, lo, hi):
    return {'index': index, 't0_s': lo, 't1_s': hi}


def _halves(cut=CUT, end=600.):
    left = {'start_s': 190., 'end_s': cut - .03, 'task': 'Unfold the blanket over the bed', 'objects': ['blanket'],
            'outcome': 'unclear', 'success_predicate': 'The blanket lies flat.', 'completed_at_s': None,
            'note': 'still being spread at the preparation cut', 'source_piece': _source(1, 0., cut)}
    right = {'start_s': cut, 'end_s': cut + 9., 'task': 'Settle the floral blanket', 'objects': ['floral blanket'],
             'outcome': 'success', 'success_predicate': 'The corner is tucked.', 'completed_at_s': cut + 8.6,
             'note': 'begun before this supplied window', 'source_piece': _source(2, cut, end)}
    return left, right


def _result(tasks, cuts, ends, *, views=('exo', 'left', 'right'), cell=(320, 320), **fields):
    """A stitched result as main writes it: its tasks joined and marked by main's stitch (nothing joins)."""
    starts = [0.] + list(cuts)
    marked = pieces.mark_cut_tasks(pieces.join_across_cuts(tasks, cuts), cuts)
    return {'model': 'openai/gpt-6.1-sol', 'reasoning_effort': 'high',
            'annotation_policy': {'rig': 'handheld_gripper', 'model': 'openai/gpt-6.1-sol', 'reasoning': 'high'},
            'config': {'views': list(views), 'cam_labels': list(views), 'cell': list(cell),
                       'pieces': [{'part': i + 1, 't0_s': a, 't1_s': b} for i, (a, b) in enumerate(zip(starts, ends))]},
            'stitched': {'parts': len(ends), 'cuts_s': list(cuts)}, 'labels': {'tasks': marked}, **fields}


def test_a_stitched_label_gives_back_its_parts_tasks_and_the_cuts_to_ask_about():
    left, right = _halves()
    result = _result([left, right], [CUT], [CUT, 600.])
    assert result['labels']['tasks'][0]['outcome'] == 'continues_past_cut'
    assert seam.part_tasks(result['labels']) == [left, right]
    (candidate,) = seam.candidates(result)
    assert candidate['cut_s'] == CUT and candidate['left'] == left and candidate['right'] == right
    # a left task its part saw complete is no candidate; neither is a cut with two open tasks before it
    done = {**left, 'completed_at_s': 300.}
    assert seam.candidates(_result([done, right], [CUT], [CUT, 600.])) == []
    two = {**left, 'task': 'Hold the pillow'}
    assert seam.candidates(_result([left, two, right], [CUT], [CUT, 600.])) == []
    # parts that share a window at their cut are joined by time, never asked about
    assert seam.candidates({**result, 'stitched': {**result['stitched'], 'windows': [{'cut_s': CUT}]}}) == []


def test_a_dry_run_counts_and_prices_the_checks_and_calls_nothing(tmp_path, monkeypatch, capsys):
    def no_call(*args, **kwargs):
        raise AssertionError('a dry run makes no call')
    monkeypatch.setattr(harness, 'call_model', no_call)
    monkeypatch.setattr(harness, 'call_model_once', no_call)
    left, right = _halves()
    rows = [{'take': 'take_a', 'label': _result([left, right], [CUT], [CUT, 600.])},
            {'take': 'take_b', 'label': {'stitched': None, 'labels': {}}}]
    path = tmp_path / 'labels.jsonl.gz'
    with gzip.open(path, 'wt') as stream:
        stream.write('\n'.join(json.dumps(row) for row in rows) + '\n')
    out = tmp_path / 'seams.json'
    assert seam.main(['--label', str(path), '--out', str(out), '--dry-run']) == 0
    printed = capsys.readouterr().out
    assert '1 recordings cut into parts; seam checks to ask: 1 (openai/gpt-6.1-sol at high reasoning)' in printed
    assert not out.exists()
    # 15s each side at 2 frames a second is 61 instants, four to a grid image
    guess = seam.estimate(rows[0]['label'], seam.candidates(rows[0]['label'])[0], 'openai/gpt-6.1-sol',
                          seam.settings(rows[0]['label'])['max_tokens'])
    assert (guess['instants'], guess['images'], guess['views']) == (61, 16, ['exo', 'left', 'right'])
    assert 0 < guess['low_usd'] < guess['high_usd'] < 1
    assert seam.settings(rows[0]['label']) == {'model': 'openai/gpt-6.1-sol', 'reasoning': 'high',
                                               'max_tokens': seam.ANSWER_TOKENS + 8000}


@pytest.mark.parametrize('text, same, read', [
    ('{"same_task": true, "evidence": "The same cups."}', True, True),
    ('```json\n{"same_task": false, "evidence": "A new task."}\n```', False, True),
    ('{"same_task": "unclear", "evidence": "Blocked view."}', 'unclear', True),
    ('{"same_task": "true", "evidence": "x"}', 'unclear', False),
    ('{"same_task": 1, "evidence": "x"}', 'unclear', False),
    ('{"same_task": true}', 'unclear', False),
    ('yes, the same task', 'unclear', False)])
def test_only_a_strict_answer_is_read_as_a_verdict(text, same, read):
    answer = seam.read_answer(text)
    assert type(answer['same_task']) is type(same) and answer['same_task'] == same
    assert ('reply_problem' not in answer) is read


def _episode(tmp_path):
    recorder_folder(tmp_path / 'upload', n=90, duplicate_stamps=False)
    rep = f.convert(tmp_path / 'upload', 'handheld_gripper', tmp_path / 'eps', 'test', 900)
    return tmp_path / 'eps' / rep['episodes'][0]['episode_id']


def test_a_seam_check_asks_once_per_candidate_and_its_verdict_joins_the_halves(tmp_path, monkeypatch, capsys):
    src = _episode(tmp_path)
    views = me.views(me.load(src))
    left, right = _halves(cut=1.5, end=3.0)
    left.update(start_s=0.1, end_s=1.5)
    right.update(end_s=2.8, completed_at_s=2.7)
    result = _result([left, right], [1.5], [1.5, 3.0], views=views, label_schema='tasks',
                     **{'given_prompt': None})
    result['labels']['completion'] = {'basis': 'tasks', 'from_tasks': 2}
    label = tmp_path / 'labels' / f'{src.name}.json'
    label.parent.mkdir()
    label.write_text(json.dumps(result))
    content, holds = seam.build_request(result, seam.candidates(result)[0], src)
    # the toy recording is 3s long, so the window is cut to it: every half second from 0s to 3s
    assert holds['instants'] == 7 and holds['images'] == 2 and holds['views'] == views
    assert sum(part['type'] == 'image_url' for part in content) == 2
    assert left['task'] in content[0]['text'] and right['task'] in content[0]['text']
    assert '"same_task": true | false | "unclear"' in content[0]['text']
    calls = []

    def reply(content, model, reasoning, key, max_tokens, timeout, **kwargs):
        calls.append((model, reasoning, max_tokens))
        return {'id': 'gen-1', 'provider': 'test',
                'usage': {'cost': 0.004, 'prompt_tokens': 900, 'completion_tokens': 40},
                'choices': [{'finish_reason': 'stop', 'message': {
                    'content': '{"same_task": true, "evidence": "The same blanket is spread across the cut."}'}}]}
    monkeypatch.setattr(harness, 'get_keys', lambda: ['sk-or-test'])
    monkeypatch.setattr(harness, 'call_model_once', reply)
    out, joined = tmp_path / 'seams.json', tmp_path / 'joined'
    argv = ['--label', str(label.parent), '--footage', str(src), '--out', str(out), '--apply', str(joined)]
    assert seam.main(argv) == 0
    saved = json.loads(out.read_text())
    verdict = saved['seams'][src.name]['1.5']
    assert verdict['same_task'] is True and verdict['request']['cost_usd'] == 0.004
    assert calls == [('openai/gpt-6.1-sol', 'high', seam.ANSWER_TOKENS + 8000)]
    assert saved['cost_usd'] == pytest.approx(0.004)
    (task,) = json.loads((joined / f'{src.name}.json').read_text())['labels']['tasks']
    assert task['task'] == left['task'] and task['outcome'] == 'success' and len(task['joined_from']) == 2
    assert task['cut_joins'][0]['by'] == 'seam_check'
    # a verdict once paid for is never asked again
    assert seam.main(argv) == 0 and len(calls) == 1


def test_rejoining_a_stitched_label_follows_each_verdict():
    left, right = _halves()
    result = _result([left, right], [CUT], [CUT, 600.], label_schema='tasks')
    result['labels']['completion'] = {'basis': 'tasks', 'from_tasks': 2, 'task_completed': None}
    same = seam.rejoin(result, {str(CUT): {'same_task': True, 'evidence': 'one blanket'}})
    (task,) = same['labels']['tasks']
    assert task['outcome'] == 'success' and task['success_predicate'] == left['success_predicate']
    assert (same['labels']['completion']['basis'], same['labels']['completion']['task_completed']) == (
        'one_task', 'success')
    different = seam.rejoin(result, {str(CUT): {'same_task': False, 'evidence': 'a new task'}})
    assert [t['outcome'] for t in different['labels']['tasks']] == ['continues_past_cut', 'success']
    assert different['labels']['tasks'][0]['cut_seams'][0]['same_task'] is False
    # without a verdict the tasks are the stitched ones, unchanged
    assert seam.rejoin(result, {})['labels']['tasks'] == result['labels']['tasks']


def test_a_task_its_part_judged_a_success_is_never_asked_about_even_without_a_time():
    # a success the part judged ended inside that part; asking whether it continues past the cut pays for nothing
    left, right = _halves()
    done = {**left, 'outcome': 'success', 'completed_at_s': None}
    assert pieces.seam_candidates([left, right], [CUT]) != []
    assert pieces.seam_candidates([done, right], [CUT]) == []
