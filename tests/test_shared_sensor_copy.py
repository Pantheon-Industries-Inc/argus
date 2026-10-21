"""Only requested shared-copy findings may use list detail on video."""
import copy
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from label import sensor_copy

from label import evidence_access as ea, harness, reparse, sensor_evidence as se
from tests.test_tactile_contract import episode
import complete_reply


def source(tmp_path):
    ep = episode({'pad': np.arange(20.).reshape(5, 4)},
                 [{'name': 'pad', 'shape': [4]}], {'pad': {'role': 'pressure'}})
    ep.update(dir=tmp_path, sources={}, action=None, depth={})
    (tmp_path / 'context.json').write_text(json.dumps(ep['context']))
    np.savez(tmp_path / 'signals.npz', pad=ep['signals']['pad'])
    np.savez(tmp_path / 'state.npz', state=ep['state'])
    return ep


def native_finding(doc):
    return {'start_s': .3, 'end_s': .3, 'headline': 'Pad response changes',
            'detail': 'The response grows while the recorded configuration stays similar.',
            'overlay_detail': 'Unrequested model caveat.', 'adds_beyond_video': True,
            'claim_type': 'pressure_response_change', 'copy_contract_version': 2,
            'evidence': [{'sensor_id': 'signal:pad', 'series_id': doc['series'][0]['id'], 'time_s': [.3]}]}


def inspected(access):
    field = next(f for f in access.inventory() if f['name'] == 'pad')
    receipt = access.inspect({'field_id': field['id'], 'mode': 'samples', 'columns': [0]})
    finding = {'start_s': .3, 'end_s': .3, 'headline': 'Stored response grows',
               'detail': 'A compact 4-unit increase accompanies the held configuration.',
               'overlay_detail': 'Unrequested model caveat.', 'adds_beyond_video': True,
               'copy_contract_version': 2,
               'evidence': [{'evidence_id': receipt['id'], 'time_s': [.3], 'columns': [0]}]}
    return finding


@pytest.mark.parametrize('version', [None, 2, 999, True])
def test_native_binding_uses_requested_copy_contract_without_mutating_raw(tmp_path, version):
    doc = se.build(source(tmp_path), {'n': 5})
    raw = native_finding(doc)
    original = copy.deepcopy(raw)
    bound = se.bind(doc, [raw], copy_contract_version=version)['findings'][0]
    if type(version) is int and version == 2:
        assert bound['copy_contract_version'] == 2 and 'overlay_detail' not in bound
    else:
        assert 'copy_contract_version' not in bound
        assert bound['overlay_detail'] == 'Unrequested model caveat.'
    assert bound['detail'] == raw['detail'] and raw == original


@pytest.mark.parametrize('version', [None, 2, 999])
def test_inspected_copy_contract_preserves_receipts_and_raw_labels(tmp_path, version):
    ep = source(tmp_path)
    access = ea.Access(ep, plan={'n': 5}, copy_contract_version=version)
    raw = inspected(access)
    original, receipts = copy.deepcopy(raw), copy.deepcopy(access.receipts)
    bound = access.bind([raw])['findings'][0]
    if version == 2:
        assert bound['copy_contract_version'] == 2 and 'overlay_detail' not in bound
    else:
        assert 'copy_contract_version' not in bound
        assert bound['overlay_detail'] == 'Unrequested model caveat.'
    assert bound['detail'] == raw['detail'] and raw == original and access.receipts == receipts


def test_shared_inspection_does_not_promote_reasoning_into_missing_detail(tmp_path):
    access = ea.Access(source(tmp_path), plan={'n': 5}, copy_contract_version=2)
    raw = inspected(access)
    raw.pop('detail')
    raw['claim'] = 'A long physical interpretation belongs in reasoning.'
    bound = access.bind([raw])['findings'][0]
    assert not bound.get('detail')
    assert bound['claim'] == raw['claim']


def test_mixed_piece_merge_keeps_only_each_findings_trusted_copy_version(tmp_path):
    ep = source(tmp_path)
    doc = se.build(ep, {'n': 5})
    raw = native_finding(doc)
    fresh = se.bind(doc, [raw], copy_contract_version=2)
    legacy = se.bind(doc, [raw])
    merged = se.merge([({'piece': {'index': 0, 't0_s': 0}}, {'sensor_evidence': fresh}),
                       ({'piece': {'index': 1, 't0_s': 1}}, {'sensor_evidence': legacy})], ep['context'])
    assert merged['findings'][0]['copy_contract_version'] == 2
    assert 'overlay_detail' not in merged['findings'][0]
    assert 'copy_contract_version' not in merged['findings'][1]
    assert merged['findings'][1]['overlay_detail'] == raw['overlay_detail']
    assert 'copy_contract_version' not in se.finding_fields(raw)


@pytest.mark.parametrize('override', ['', 'Human-reviewed comparison.'])
@pytest.mark.parametrize('version', [None, 2, 999])
def test_reparse_uses_saved_version_and_preserves_marked_human_override(tmp_path, monkeypatch, override, version):
    ep = source(tmp_path)
    doc = se.build(ep, {'n': 5})
    raw = native_finding(doc)
    access = ea.Access(ep, plan={'n': 5}, copy_contract_version=version)
    extra_raw = inspected(access)
    evidence = se.bind(doc, [raw], copy_contract_version=version)
    additional = access.bind([extra_raw])
    for key in ('sensors', 'series', 'findings', 'unbound_findings', 'coverage', 'limitations'):
        evidence.setdefault(key, []).extend(additional[key])
    for finding in evidence['findings']:
        finding['review_status'] = 'accepted'
        if version == 2:
            finding['overlay_detail'] = override
    record = {'episode_dir': str(tmp_path), 'config': {'sensor_copy_contract_version': version},
              'labels': {'sensor_findings': [raw], 'evidence_findings': [extra_raw]},
              'sensor_evidence': evidence, 'evidence_inspection': access.record()}
    original_labels = copy.deepcopy(record['labels'])
    monkeypatch.setattr(harness.me, 'load', lambda path: ep)
    for _ in range(2):
        reparse.rebind(record)
        findings = record['sensor_evidence']['findings']
        assert len(findings) == 2
        if version == 2:
            assert all(f['copy_contract_version'] == 2 for f in findings)
            assert all(f['overlay_detail'] == override and f['review_status'] == 'accepted' for f in findings)
        else:
            assert all('copy_contract_version' not in f for f in findings)
            assert all(f['overlay_detail'] == 'Unrequested model caveat.' for f in findings)
        assert record['labels'] == original_labels


def test_recorded_request_version_controls_binding_not_model_marker(tmp_path, monkeypatch):
    ep = source(tmp_path)
    doc = se.build(ep, {'n': 5})
    raw = {**complete_reply.labels(), 'sensor_findings': [native_finding(doc)]}
    response_text = json.dumps(raw)
    monkeypatch.setattr(harness, 'call_model_once', lambda *a, **k: {
        'choices': [{'finish_reason': 'stop', 'message': {'content': response_text}}],
        'usage': {'cost': 0.0}})
    output = tmp_path / 'reply.json'
    result = harness._call_and_record(tmp_path, output, [{'type': 'text', 'text': 'Offline request'}], 0,
        {'sensor_evidence': doc, 'config': {'sensor_copy_contract_version': 2}},
        model='openai/gpt-6.1-sol', reasoning='high', api_key='offline', max_tokens=128, timeout=1)
    assert result['config']['sensor_copy_contract_version'] == 2
    assert result['response_text'] == response_text and result['labels'] == raw
    assert result['sensor_evidence']['findings'][0]['copy_contract_version'] == 2
    assert 'overlay_detail' not in result['sensor_evidence']['findings'][0]
    assert json.loads(output.read_text())['labels'] == raw


def test_unbound_model_marker_cannot_become_a_human_overlay_override(tmp_path):
    ep = source(tmp_path)
    doc = se.build(ep, {'n': 5})
    raw = native_finding(doc)
    doc['unbound_findings'] = [{'finding': copy.deepcopy(raw), 'reason': 'previous binder rejected it'}]
    record = {'episode_dir': str(tmp_path), 'config': {'sensor_copy_contract_version': 2},
              'labels': {'sensor_findings': [raw]}, 'sensor_evidence': doc}
    reparse.rebind(record)
    assert 'overlay_detail' not in record['sensor_evidence']['findings'][0]


def test_requested_prompt_contract_reaches_both_outputs_and_legacy_stays_legacy(tmp_path):
    ep = source(tmp_path)
    doc = se.build(ep, {'n': 5})
    assert 'overlay_detail' not in se.prompt(doc)
    legacy = copy.deepcopy(doc)
    legacy.pop('copy_contract_version', None)
    assert 'overlay_detail' in se.prompt(legacy)
    for version in (None, 2, 3, 4, 5, 999):
        access = ea.Access(ep, plan={'n': 5}, copy_contract_version=version)
        parts, _, _ = ea.final_content([], 0, access, access.record(), [],
            image_limit=100, image_bytes_limit=100000, image_inflation=1)
        text = '\n'.join(p['text'] for p in parts)
        assert ('overlay_detail' not in text) is (version in (2, 3, 4, 5))


@pytest.mark.parametrize('version', [None, 999])
def test_legacy_prompt_assembly_remains_byte_exact(tmp_path, version):
    doc = se.build(source(tmp_path), {'n': 5})
    doc['copy_contract_version'] = version
    assert hashlib.sha256(se.prompt(doc).encode()).hexdigest() == 'bdcf39d931f15892247af03e856543e656da895ecec6c8a5d89b959d31d594f0'
    assert hashlib.sha256(ea.final_prompt(version).encode()).hexdigest() == 'f5d6e5f73362a487085e3dad574d22a8183b96271a8d7652f8ae881710797d20'


def test_version_3_prompts_stay_byte_exact_so_saved_requests_remain_reproducible(tmp_path):
    doc = se.build(source(tmp_path), {'n': 5})
    doc['copy_contract_version'] = 3
    assert hashlib.sha256(se.prompt(doc).encode()).hexdigest() == 'a5372068cf1259088c0e052a75b728a4899d784bd7e8d93781a30cdc23449e1b'
    assert hashlib.sha256(ea.final_prompt(3).encode()).hexdigest() == '611b01fcae96c133ad4679255832381f4e1becc38c70b51b5e3eee1a87b10143'


def test_version_4_keeps_limits_off_the_card_and_words_an_undeclared_direction_as_rose_or_fell(tmp_path):
    doc = se.build(source(tmp_path), {'n': 5})
    doc['copy_contract_version'] = 4
    text = se.prompt(doc)
    assert sensor_copy.SHARED_RULE_V4 in text and 'say what rose or fell' in text
    assert sensor_copy.V4_LIMITS in text and sensor_copy.LEGACY_LIMITS_LEAD not in text
    assert sensor_copy.LEGACY_LIMITS_MAPPING not in text
    inspection = ea.final_prompt(4)
    assert sensor_copy.V4_LIMITS in inspection and sensor_copy.LEGACY_LIMITS_MAPPING not in inspection


def test_new_request_saves_the_copy_contract_it_actually_supplies(tmp_path):
    from tests.test_label import _packed_episode
    folder, n = _packed_episode(tmp_path)
    path = folder / 'context.json'
    ctx = json.loads(path.read_text())
    ctx['signals'] = [{'name': 'right_pressure', 'key': 'pressure', 'shape': [4]}]
    path.write_text(json.dumps(ctx))
    np.savez(folder / 'signals.npz', pressure=np.arange(n * 4).reshape(n, 4))
    result = harness.label_episode(folder, tmp_path / 'dry.json', model='openai/gpt-6.1-sol',
        reasoning='high', api_key='offline', max_tokens=128, timeout=1, cell_w=128, dry_run=True)
    assert result['config']['sensor_copy_contract_version'] == sensor_copy.VERSION
    assert result['sensor_evidence']['copy_contract_version'] == sensor_copy.VERSION
    assert sensor_copy.rule('sensor', sensor_copy.VERSION) in result['prompt_text']
    assert '"sensor_findings"' in result['prompt_text']
    assert 'overlay_detail' not in result['prompt_text']


@pytest.mark.skipif(not shutil.which('node'), reason='node is required')
def test_shared_detail_renders_on_video_only_when_bound_marker_authorizes_it():
    html = Path(se.__file__).parents[1] / 'board' / 'index.html'
    script = r'''
const fs=require('fs'),assert=require('assert');const src=fs.readFileSync(process.argv[1],'utf8');
const begin=src.indexOf('// ================= sensor evidence:'),end=src.indexOf('// ================= touch:',begin);
const api=new Function('esc','fmtT','document',src.slice(begin,end)+'return {sensorEvidence,gripFindingStripHtml};')(String,String,{});
const raw={start_s:0,end_s:1,headline:'Pad response',detail:'A useful supported contrast.',adds_beyond_video:true};
function render(f){const row=api.sensorEvidence({sensor_evidence:{version:1,findings:[f]}}).insights[0];return api.gripFindingStripHtml(row);}
assert(render({...raw,copy_contract_version:2}).includes(raw.detail));
for(const version of [undefined,999,true])assert(!render({...raw,copy_contract_version:version}).includes(raw.detail));
assert(!render({...raw,copy_contract_version:2,overlay_detail:''}).includes('gf-detail'));
assert(render({...raw,copy_contract_version:2,overlay_detail:'Reviewed caption.'}).includes('Reviewed caption.'));
'''
    subprocess.run(['node', '-e', script, str(html)], check=True)


def test_version_4_prompts_stay_byte_exact_so_saved_requests_remain_reproducible(tmp_path):
    doc = se.build(source(tmp_path), {'n': 5})
    doc['copy_contract_version'] = 4
    assert hashlib.sha256(se.prompt(doc).encode()).hexdigest() == 'bc285db4cc5197bb22e210ad4a9c9c8df3167129a0199d0583528168f9a24a88'
    assert hashlib.sha256(ea.final_prompt(4).encode()).hexdigest() == 'd2e5b4eec2215d39cd09ded11f369e5cdf30871d3b67a716a878a1364774320b'


def test_version_5_asks_for_plain_english_that_keeps_every_specific(tmp_path):
    doc = se.build(source(tmp_path), {'n': 5})
    doc['copy_contract_version'] = 5
    for text in (se.prompt(doc), ea.final_prompt(5)):
        assert sensor_copy.SHARED_RULE_V5 in text and sensor_copy.PLAIN_ENGLISH in text
        assert 'a few plain words' not in text and 'one part of that region' not in text
        assert sensor_copy.V4_LIMITS in text and sensor_copy.LEGACY_LIMITS_MAPPING not in text
    assert sensor_copy.VERSION == 5 and sensor_copy.shared(5)


def test_the_board_treats_the_same_copy_versions_as_shared():
    html = (Path(se.__file__).parents[1] / 'board' / 'index.html').read_text()
    listed = re.search(r'const SHARED_COPY_VERSIONS = \[([0-9, ]*)\];', html).group(1)
    assert [int(v) for v in listed.split(',')] == list(sensor_copy.SHARED_VERSIONS)
