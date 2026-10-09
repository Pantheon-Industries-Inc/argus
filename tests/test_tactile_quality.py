import copy
import subprocess
from pathlib import Path

import numpy as np
import pytest

from checks import tactile_quality as qc
from board import to_board


def fixture(offset=0):
    contacts, strips = [], {}
    for i, start in enumerate([1., 3., 5.]):
        cid = f'c{i}'
        contacts.append({'id': cid, 'hand': 'right', 'signals': ['right_pressure'],
                         'start_s': start, 'end_s': start + .8, 'shown': True,
                         'seen': {'touch_seen': 'yes', 'hand': 'right',
                                  'first_touch_frame': 2, 'last_touch_frame': 1}})
        strips[cid] = {'begin': [start + offset - .05, start + offset + .05],
                       'end': [start + .8 + offset - .05, start + .8 + offset + .05]}
    t = np.arange(210) / 30
    a = np.random.default_rng(4).normal(3000, 3, (210, 8))
    return contacts, strips, {'right_pressure': a}, t


def run(parts, meta=None):
    return qc.check(*parts, meta=meta or {}, fps=30)


def kinds(result):
    return {w['kind'] for w in result['warnings']}


def test_clean_and_small_boundary_uncertainty_are_quiet():
    for offset in [0, .12, -.12]:
        assert run(fixture(offset))['warnings'] == []


@pytest.mark.parametrize('offset', [.5, -.5])
def test_repeated_consistent_offset_flags_with_interval_evidence(offset):
    parts = fixture(offset)
    before = copy.deepcopy(parts[0])
    result = run(parts)
    warning = result['warnings'][0]
    assert warning['kind'] == 'timing_mismatch'
    assert warning['contact_ids'] == ['c0', 'c1', 'c2']
    assert abs(warning['offset_s']) == pytest.approx(.5)
    assert warning['offset_bounds_s'][1] - warning['offset_bounds_s'][0] == pytest.approx(.1)
    assert parts[0] == before


@pytest.mark.parametrize('change', ['unshown', 'rejected', 'wrong_hand', 'assumed', 'censored', 'bool', 'nonfinite', 'other_source', 'opposite'])
def test_ambiguous_or_conflicting_evidence_does_not_flag(change):
    parts = fixture(.5)
    contact = parts[0][0]
    if change == 'unshown': contact['shown'] = False
    elif change == 'rejected': contact['review_status'] = 'rejected'
    elif change == 'wrong_hand': contact['seen']['hand'] = 'left'
    elif change == 'assumed': contact['aligned_by'] = 'starts_together'
    elif change == 'censored': contact.update(from_start=True, to_end=True)
    elif change == 'bool': contact['seen'].update(first_touch_frame=True, last_touch_frame=True)
    elif change == 'nonfinite': parts[1]['c0'] = {'begin': [float('nan'), 2], 'end': [3, float('inf')]}
    elif change == 'other_source': contact['signals'] = ['left_pressure']
    elif change == 'opposite':
        parts[1]['c0'] = {k: [x - 1 for x in v] for k, v in parts[1]['c0'].items()}
    assert 'timing_mismatch' not in kinds(run(parts))


def test_sustained_missing_rows_but_not_dead_cells_or_single_drop():
    parts = fixture()
    parts[2]['right_pressure'][:, 0] = 0
    parts[2]['right_pressure'][30] = np.nan
    assert run(parts)['warnings'] == []
    parts[2]['right_pressure'][60:100] = np.nan
    result = run(parts)
    assert kinds(result) == {'missing_readings'}
    assert result['warnings'][0]['start_s'] == 2
    assert result['warnings'][0]['end_s'] == pytest.approx(100 / 30)


def test_frozen_requires_multiple_visible_changes_not_steady_hold():
    parts = fixture()
    parts[2]['right_pressure'][:] = 3000
    assert kinds(run(parts)) == {'frozen_readings'}
    parts[1].clear()
    assert run(parts)['warnings'] == []
    assert run(parts)['checks']['timing'] == 'insufficient_evidence'


def test_native_repeats_and_non_touch_role_do_not_flag():
    parts = fixture()
    parts[2]['right_pressure'][:] = np.repeat(np.arange(42), 5)[:, None]
    assert run(parts, {'right_pressure': {'rate_hz': 6}})['warnings'] == []
    parts[2]['right_pressure'][:] = np.nan
    assert run(parts, {'right_pressure': {'dictionary': {'role': 'landmarks'}}})['warnings'] == []


def test_assumed_signal_clock_and_truncated_tail():
    parts = fixture(.5)
    assert 'timing_mismatch' not in kinds(run(parts, {'right_pressure': {'aligned_by': 'starts_together'}}))
    parts = fixture()
    parts[2]['right_pressure'] = parts[2]['right_pressure'][:150]
    result = run(parts)
    assert kinds(result) == {'missing_readings'}
    assert result['warnings'][0]['start_s'] == 5


def test_board_build_computes_quality_and_preserves_reviewed_warning(tmp_path):
    import json
    from board.build import add_context
    from test_board_touch import _episode
    ep = _episode(tmp_path)
    ctx = json.loads((ep / 'context.json').read_text())
    ctx['state_kind'] = 'none'
    (ep / 'context.json').write_text(json.dumps(ctx))
    d = {}
    add_context(d, ctx, ep, {})
    assert d['tactile_qc']['warnings'] == []
    assert d['tactile_qc']['checks']['timing'] == 'insufficient_evidence'
    reviewed = {'version': 1, 'warnings': [{'headline': 'Old warning', 'review_status': 'rejected'}]}
    d = {'tactile_qc': copy.deepcopy(reviewed)}
    add_context(d, ctx, ep, {})
    assert d['tactile_qc'] == reviewed


def test_current_human_dictionary_role_overrides_sensor_name(tmp_path, monkeypatch):
    import json
    from label import dictionary_editor
    from test_board_touch import _episode
    ep = _episode(tmp_path)
    ctx = json.loads((ep / 'context.json').read_text())
    ctx['signals'] = ctx['signals'][:1]
    ctx['state_kind'] = 'none'
    (ep / 'context.json').write_text(json.dumps(ctx))
    np.savez(ep / 'signals.npz', s0=np.full((300, 16), np.nan))
    human = {'episode_id': ep.name, 'fields': [{'id': 'f', 'name': 'right_pressure', 'kind': 'signal',
        'bindings': [{'episode': ep.name, 'context_path': 'signals/0', 'file': 'signals.npz', 'key': 's0'}]}],
        'entries': {'f': {'role': 'event_flag', 'provenance': 'human'}}}
    monkeypatch.setattr(dictionary_editor, 'for_episode', lambda *a: human)
    assert qc.for_episode(ep, [], {})['warnings'] == []


def test_saved_quality_survives_conversion_without_mutating_result():
    result = {'episode_dir': '/x/episode_test', 'labels': {}, 'tactile_qc': run(fixture(.5))}
    before = copy.deepcopy(result)
    assert to_board.convert(result, 'test')['tactile_qc'] == result['tactile_qc']
    assert result == before


def test_warning_ui_is_quiet_when_empty_and_seeks_to_evidence():
    root = Path(__file__).resolve().parents[1]
    script = r'''
const fs=require('fs'),assert=require('assert');
const s=fs.readFileSync(process.argv[1],'utf8'),begin=s.indexOf('// ================= sensor evidence:'),end=s.indexOf('// ================= touch:',begin);
const api=new Function('esc','fmtT',s.slice(begin,end)+'return {sensorEvidence,sensorEvidenceHtml};')(s=>String(s).replace(/</g,'&lt;').replace(/"/g,'&quot;'),t=>t+'s');
assert.equal(api.sensorEvidenceHtml(api.sensorEvidence({tactile_qc:{version:1,warnings:[]}}),[]),'');
const E=api.sensorEvidence({tactile_qc:{version:1,warnings:[{kind:'timing_mismatch',start_s:2,end_s:3,headline:'Possible tactile/video timing mismatch',detail:'<unsafe>',contact_ids:['c1']},{start_s:2,end_s:3,headline:'Rejected',review_status:'rejected'},{start_s:'bad',end_s:3,headline:'Bad'}]}});
assert.equal(E.warnings.length,1);
const html=api.sensorEvidenceHtml(E,[]);
assert(html.includes('data-evidence-t="2"'));
assert(html.includes('data-evidence-pause="true"'));
assert(html.includes('&lt;unsafe>') && !html.includes('<unsafe>'));
assert(!html.includes('Astra annotation'));
assert(!html.includes('Rejected'));
'''
    subprocess.run(['node', '-e', script, str(root / 'board/serve.py')], check=True)
