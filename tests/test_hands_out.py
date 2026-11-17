"""Every Ego label carries three hands-out-of-frame figures (both hands out, only the left out, only the right out),
and a run sums them in run.json, never one blended number (label/hand_visibility.py)."""
import json

import pytest

from label import hand_visibility as hv


def _rows(*steps):
    return {'timeline': [{'start_s': a, 'end_s': b, 'arm': 'both', 'action': 'work',
                          'left_hand_visible': left, 'right_hand_visible': right} for a, b, left, right in steps]}


def test_a_step_with_only_the_left_hand_out_counts_in_left_only_and_not_in_both():
    got = hv.hands_out(_rows((0, 2, False, True)), 10)
    assert got == {'episode_s': 10.0, 'both_s': 0.0, 'both_pct': 0.0, 'left_only_s': 2.0, 'left_only_pct': 20.0,
                   'right_only_s': 0.0, 'right_only_pct': 0.0}


def test_each_figure_counts_its_own_moments():
    got = hv.hands_out(_rows((0, 2, False, True), (2, 3, True, False), (3, 7, False, False), (7, 10, True, True)), 20)
    assert (got['both_s'], got['left_only_s'], got['right_only_s']) == (4.0, 2.0, 1.0)
    assert (got['both_pct'], got['left_only_pct'], got['right_only_pct']) == (20.0, 10.0, 5.0)


def test_overlapping_rows_count_a_moment_once_and_an_unknown_hand_counts_nowhere():
    labels = _rows((0, 4, False, True), (2, 4, False, False), (4, 6, False, None), (6, 7, None, None))
    labels['timeline'][1]['arm'] = 'right'
    got = hv.hands_out(labels, 10)
    # 2 to 4 s: one row says both are out, so the moment is both-out, not also left-only
    assert (got['both_s'], got['left_only_s'], got['right_only_s']) == (2.0, 2.0, 0.0)


def test_spans_are_clipped_to_the_episode_and_an_unknown_length_gives_no_percent():
    assert hv.hands_out(_rows((8, 14, True, False)), 10)['right_only_s'] == 2.0
    got = hv.hands_out(_rows((0, 3, False, False)), None)
    assert got['both_s'] == 3.0 and got['both_pct'] is None and got['episode_s'] is None


def test_labels_without_per_hand_fields_get_no_figures():
    old = {'timeline': [{'start_s': 0, 'end_s': 4, 'hands_visible': False}]}
    assert hv.hands_out(old, 10) is None
    result = {'labels': old, 'hands_out': {'both_s': 1}}
    assert 'hands_out' not in hv.attach(result, 10)


def test_run_totals_sum_hours_over_the_episodes_that_carry_figures():
    a = {'hands_out': hv.hands_out(_rows((0, 360, False, True), (360, 720, False, False)), 3600)}
    b = {'hands_out': hv.hands_out(_rows((0, 720, True, False)), 7200)}
    got = hv.run_totals([a, b, {'labels': {}}])
    assert got['episodes'] == 2 and got['hours'] == 3.0
    assert (got['both_hours'], got['left_only_hours'], got['right_only_hours']) == (0.1, 0.1, 0.2)
    assert got['both_pct'] == pytest.approx(100 * 360 / 10800, abs=1e-3)
    assert got['right_only_pct'] == pytest.approx(100 * 720 / 10800, abs=1e-3)
    assert hv.run_totals([{'labels': {}}]) is None


def test_the_harness_writes_the_figures_into_each_saved_result(tmp_path, monkeypatch):
    from label import harness
    (tmp_path / 'context.json').write_text(json.dumps({'duration_s': 10.0}))
    text = json.dumps({'task_summary': 'Wipe the table.', 'completion': {'task_completed': 'success'},
                       'timeline_columns': ['start_s', 'end_s', 'arm', 'action', 'hands_visible',
                                            'left_hand_visible', 'right_hand_visible'],
                       'timeline': [[0, 3, 'right', 'wipe', True, False, True],
                                    [3, 10, 'right', 'wipe', False, False, False]]})
    monkeypatch.setattr(harness, 'call_model', lambda *a, **kw: {
        'usage': {'prompt_tokens': 100, 'completion_tokens': 10},
        'choices': [{'finish_reason': 'stop', 'message': {'content': text}}]})
    out = tmp_path / 'episode_a.json'
    harness._call_and_record(tmp_path, out, [{'type': 'text', 'text': 'test'}], 0, {}, model='openai/gpt-6-astra',
                             reasoning='medium', api_key='fixture', max_tokens=100, timeout=1)
    saved = json.loads(out.read_text())['hands_out']
    assert (saved['both_s'], saved['left_only_s'], saved['right_only_s']) == (7.0, 3.0, 0.0)
    assert (saved['both_pct'], saved['left_only_pct']) == (70.0, 30.0)


def test_a_stitched_recording_gets_figures_over_its_whole_length(tmp_path, monkeypatch):
    from label import pieces
    from test_pieces import _part_result, _three_parts
    job, src, parts = _three_parts(tmp_path, monkeypatch)
    for part in parts:
        record = _part_result(part, 'carry')
        record['labels']['timeline'][0].update(left_hand_visible=False, right_hand_visible=True)
        record['hands_out'] = {'episode_s': 1.0, 'left_only_s': 0.5}      # a part's own, which the stitch replaces
        (job / 'run' / 'out' / f'{part.name}.json').write_text(json.dumps(record))
    pieces.stitch_run(job, src.parent, {src.name: [p.name for p in parts]}, tmp_path / 'final')
    stitched = json.loads((tmp_path / 'final' / f'{src.name}.json').read_text())
    want = hv.hands_out(stitched['labels'], pieces.duration(json.loads((src / 'context.json').read_text()), src))
    assert stitched['hands_out'] == want
    assert want['left_only_s'] == pytest.approx(1.5) and want['both_s'] == 0.0


def test_run_json_sums_the_three_figures(tmp_path):
    from label.finalize import finalize
    run, slices = tmp_path / 'run', tmp_path / 'slice'
    (run / 'out').mkdir(parents=True)
    (run / 'log.txt').write_text('done=2\n')
    for name, length, steps in (('episode_a', 3600, ((0, 360, False, True),)),
                                ('episode_b', 3600, ((0, 720, False, False),))):
        (slices / name).mkdir(parents=True)
        (slices / name / 'context.json').write_text(json.dumps({'duration_s': length}))
        (run / 'out' / f'{name}.json').write_text(json.dumps({'hands_out': hv.hands_out(_rows(*steps), length)}))
    finalize(run, {'kind': 'full'}, slices, 0)
    got = json.loads((run / 'run.json').read_text())['hands_out']
    assert got['episodes'] == 2 and got['hours'] == 2.0
    assert (got['both_hours'], got['left_only_hours'], got['right_only_hours']) == (0.2, 0.1, 0.0)
    assert (got['both_pct'], got['left_only_pct'], got['right_only_pct']) == (10.0, 5.0, 0.0)


def test_the_board_carries_the_labels_figures_and_computes_none_of_its_own(tmp_path):
    import complete_reply
    from board import serve, to_board
    figures = hv.hands_out(_rows((0, 2, False, True), (2, 3, False, False)), 10)
    result = {'episode_dir': '/x/episode_a', 'parse_ok': True, 'hands_out': figures,
              'labels': complete_reply.labels('wipe', **_rows((0, 2, False, True), (2, 3, False, False)))}
    d = to_board.convert(result, 'ds')
    assert d['hands_out'] == figures
    d.update(_rig='ego_head', _meta={'episode_id': 'episode_a'}, duration_s=10)
    (tmp_path / 'episode_a.json').write_text(json.dumps(d))
    old = {**d, '_meta': {'episode_id': 'episode_b'}}
    old.pop('hands_out')
    old['event_labels'] = [{'t_s': 0, 'end_s': 5, 'hands_visible': False}]
    (tmp_path / 'episode_b.json').write_text(json.dumps(old))
    recs = {r['episode_id']: r for r in serve.rail_records(tmp_path)}
    assert recs['episode_a']['hands_out'] == figures
    # a label without per-hand fields gets no figures on the board either, and no blended one
    assert 'hands_out' not in recs['episode_b'] and 'hands_hidden_s' not in recs['episode_b']


BOARD_SHARES = r'''
const assert = require('node:assert/strict'), {piece, src} = require('./tests/page_functions')();
eval(piece('handsOutShares'));
const eps = [{hands_out: {episode_s: 100, both_s: 4, left_only_s: 2, right_only_s: 1}},
             {hands_out: {episode_s: 300, both_s: 0, left_only_s: 6, right_only_s: 0}},
             {duration_s: 600}];
const got = handsOutShares(eps);
const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-12, a + ' != ' + b);
close(got.both, 4 / 400); close(got.left_only, 8 / 400); close(got.right_only, 1 / 400);
assert.equal(got.seconds, 400);
assert.equal(handsOutShares([{duration_s: 50}]), null);
assert.ok(!src.includes('hands_hidden_s'));
'''


def test_the_board_shares_cover_only_the_episodes_whose_labels_record_each_hand():
    import subprocess
    subprocess.run(['node', '-e', BOARD_SHARES], check=True)
