"""Cross-format sensor evidence, without network calls or large recordings."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from label import sensor_evidence as se


def test_contact_checks_follow_recorded_hands_even_when_the_pressure_stays_quiet(tmp_path):
    from test_touch_depth import _hdf5
    from prepare import formats
    from label import contacts as lc
    from checks import contacts as cc

    source = tmp_path / 'recording'
    source.mkdir()
    _hdf5(source / 'glove.hdf5', demos=1, press=None)
    prepared = tmp_path / 'prepared'
    report = formats.convert(source, 'ego_head', prepared, 'quiet_glove', 10)
    assert not report['failed']
    ctx = json.loads((prepared / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    meta = {entry['name']: entry for entry in ctx['signals']}
    assert lc.coverage(meta) == {'right': ['right_pressure']}
    assert not ctx['contacts'], 'quiet measurements must not invent positive contacts to establish coverage'
    labels = {'contacts_missing': [
        {'t_s': .5, 'hand': 'right', 'object': 'cup'},
        {'t_s': .6, 'hand': 'left', 'object': 'plate'},
        {'t_s': .7, 'hand': None, 'object': 'bowl'},
        {'t_s': .8, 'hand': 'both', 'object': 'tray'},
        {'t_s': .9, 'hand': 'right', 'object': 'spoon', 'review_status': 'rejected'}]}
    original = copy.deepcopy(labels)
    checked = cc.check(labels, ctx['contacts'], {}, ctx['fps'], instrumented_hands=lc.coverage(meta))
    assert checked['contacts'] == checked['checked'] == 0
    assert [note['check'] for note in checked['notes']] == ['contact_missing']
    assert checked['notes'][0]['evidence'].endswith('0.5 s, right, cup')
    assert labels == original
    # A reviewed interpretation can withdraw touch meaning without altering the recording or its raw labels.
    meta['right_pressure']['dictionary'] = {'role': 'annotation', 'provenance': 'human'}
    assert lc.coverage(meta) == {}
    assert cc.check(labels, [], {}, ctx['fps'], instrumented_hands=lc.coverage(meta))['notes'] == []
    assert labels == original


def test_contact_coverage_respects_effective_roles_companions_and_legacy_check_scope():
    from label import contacts as lc
    from checks import contacts as cc

    meta = {'left_channel0': {'dictionary': {'role': 'touch'}},
            'right_pressure': {'touch': False, 'touch_role': 'touch'},
            'left_pressure': {'dictionary': {'role': 'joint_state'}, 'touch_role': 'touch'},
            'right_pressure_cmd': {'dictionary': {'role': 'touch'}},
            'left_channel1': {'touch_role': 'touch', 'dictionary': {'role': None}},
            'left_pressure variation within each frame': {'variation_of': 'left_pressure'},
            'left_right_pressure': {'dictionary': {'role': 'touch'}},
            'pad': {'dictionary': {'role': 'touch'}}}
    assert lc.coverage(meta) == {'left': ['left_channel0'], 'right': ['right_pressure']}
    recorded = [{'id': 'c1', 'hand': 'right', 'start_s': 0, 'end_s': 1}]
    labels = {'contacts_missing': [{'t_s': 2, 'hand': 'left', 'object': 'cup'},
                                   {'t_s': 3, 'hand': 'right', 'object': 'bottle'}]}
    original = copy.deepcopy(labels)
    legacy = cc.check(labels, recorded, {}, 30)
    assert legacy['notes'][0]['evidence'].endswith('3.0 s, right, bottle')
    assert cc.check(labels, [], {}, 30)['notes'] == []
    assert cc.check({}, [], {}, 30, instrumented_hands={}) is None
    assert cc.check({}, [], {}, 30, instrumented_hands={'right'}) == {'contacts': 0, 'checked': 0, 'notes': []}
    assert labels == original


def episode(signals, metadata, entries=None, cameras=None):
    n = len(next(iter(signals.values()))) if signals else 5
    ctx = {'episode_id': 'episode_test', 'fps': 10, 'signals': metadata,
           'cameras': cameras or {'exo': {'name': 'scene'}}}
    fields, interpretations = [], {}
    for i, meta in enumerate(metadata):
        ident = 'field_' + str(i)
        fields.append({'id': ident, 'kind': 'signal', 'name': meta['name'],
                       'bindings': [{'episode': 'episode_test', 'context_path': f'signals/{i}',
                                     'file': 'signals.npz', 'key': meta.get('key')}]})
        if entries and meta['name'] in entries:
            interpretations[ident] = entries[meta['name']]
    ctx['data_dictionary'] = {'episode_id': 'episode_test', 'fields': fields, 'entries': interpretations}
    return {'context': ctx, 'signals': signals, 'signal_meta': {m['name']: m for m in metadata},
            'times': {view: np.arange(n) / 10 for view in ctx['cameras']}, 'state': np.zeros((n, 1))}


def test_force_axes_and_contact_flags_keep_their_meaning_without_fingers():
    ep = episode({'load': np.arange(30).reshape(5, 6), 'switch': np.array([[0], [1], [1], [0], [0]]),
                  'force_cmd': np.ones((5, 1)), 'pressure_notes': np.ones((5, 1))},
                 [{'name': 'load', 'shape': [6], 'names': ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz'],
                   'units': ['N', 'N', 'N', 'Nm', 'Nm', 'Nm']},
                  {'name': 'switch', 'shape': [1]}, {'name': 'force_cmd', 'shape': [1]},
                  {'name': 'pressure_notes', 'shape': [1]}],
                 {'load': {'role': 'force_torque'}, 'switch': {'role': 'contact_state'},
                  'force_cmd': {'role': 'touch'}, 'pressure_notes': {'role': 'annotation', 'provenance': 'human'}})
    doc = se.build(ep, {'n': 5, 'ks': [0, 2, 4]})
    assert [s['name'] for s in doc['sensors']] == ['load', 'switch']
    assert [r['label'] for r in doc['series'][:6]] == ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
    assert [r['unit'] for r in doc['series'][:6]] == ['N', 'N', 'N', 'Nm', 'Nm', 'Nm']
    assert doc['series'][-1]['values'] == [0, 1, 1, 0, 0]
    assert 'thumb' not in json.dumps(doc)


def test_unknown_grid_keeps_signed_raw_readings_gaps_and_declared_groups():
    raw = np.array([[0, 10, 20, 30], [5, 8, np.nan, 32], [10, 6, 20, 34], [5, 4, 20, 36], [0, 2, 20, 38]])
    meta = {'name': 'pad', 'shape': [2, 2]}
    entry = {'role': 'touch', 'layout': [{'start': 0, 'count': 2, 'name': 'Upper pad'},
                                        {'start': 2, 'count': 2, 'name': 'Lower pad'}]}
    ep = episode({'pad': raw}, [meta], {'pad': entry})
    before = copy.deepcopy(ep['context'])
    doc = se.build(ep, {'n': 5, 'ks': [0, 2, 4]})
    assert ep['context'] == before
    assert doc['sensors'][0]['quantity'] == 'sensor_response'
    assert [r['label'] for r in doc['series']] == ['Cell 0,0', 'Cell 0,1', 'Cell 1,0', 'Cell 1,1']
    assert doc['series'][0]['values'][0] == 0
    assert doc['series'][2]['values'][1] is None
    assert all(r['unit'] is None for r in doc['series'])
    assert 'ADC' not in se.prompt(doc)
    assert 'not a verified unloaded baseline' in se.prompt(doc)


def test_optical_evidence_uses_only_frames_actually_sent():
    ep = episode({}, [], cameras={'exo': {'name': 'scene'}, 'extra1': {'name': 'gelsight_left'},
                                  'extra2': {'name': 'digit_right'}, 'extra3': {'name': 'tactile_left_heatmap'}})
    from PIL import Image
    gel = {k: Image.new('RGB', (64, 48), (120, 160, 140)) for k in (0, 4)}
    for im in gel.values():
        im.paste((60, 90, 80), (20, 10, 40, 30))
    doc = se.build(ep, {'n': 5, 'ks': [0, 2, 4]}, shown={'exo': [0, 2, 4], 'extra1': [0, 4], 'extra3': [0, 4]},
                   images={'extra1': gel})
    assert [s['view'] for s in doc['sensors']] == ['extra1', 'extra3']
    assert doc['sensors'][0]['times'] == [0, .4]
    # optical from its frames' fixed view, never from the brand in its name (label/camera_class.py)
    assert 'image_deformation' in doc['sensors'][0]['capabilities']
    unseen = se.build(ep, {'n': 5, 'ks': [0, 2, 4]}, shown={'extra1': [0, 4]})
    assert unseen['sensors'][0]['quantity'] == 'tactile_image'
    assert doc['sensors'][1]['capabilities'] == ['image_response_change']
    assert any('digit_right' in x for x in doc['limitations'])


def test_findings_require_real_evidence_and_supported_claims():
    ep = episode({'right_pressure': np.arange(5).reshape(5, 1)}, [{'name': 'right_pressure', 'shape': [1]}])
    doc = se.build(ep, {'n': 5, 'ks': [0, 2, 4]})
    sensor, trace = doc['sensors'][0], doc['series'][0]
    finding = {'start_s': .1, 'end_s': .3, 'headline': 'Response rises while the tool stays held',
               'claim_type': 'response_change', 'adds_beyond_video': True, 'confidence': 'medium',
               'evidence': [{'sensor_id': sensor['id'], 'series_id': trace['id'], 'time_s': [.1, .3]}]}
    bad_ref = {**finding, 'evidence': [{'sensor_id': sensor['id'], 'series_id': 'invented', 'time_s': [.1]}]}
    bad_claim = {**finding, 'claim_type': 'slip_candidate'}
    rejected = {**finding, 'review_status': 'rejected'}
    result = se.bind(doc, [finding, bad_ref, bad_claim, rejected])
    assert result['findings'] == [finding, rejected]
    assert len(result['unbound_findings']) == 2
    assert doc['findings'] == []
    edited = copy.deepcopy(ep['context'])
    edited['data_dictionary']['entries']['field_0'] = {'role': 'joint_state', 'provenance': 'human'}
    assert not se.compatible(result, edited)
    # Preserve browsing and recorded values while preventing timed claims on display placements.
    ep['context']['camera_clock'] = {'left': {'clock_problem': 'repeated timestamps'}}
    assert se.bind(se.build(ep, {'n': 5, 'ks': [0, 2, 4]}), [finding])['findings'] == [finding]
    ep['context']['camera_clock']['exo'] = {'clock_problem': 'repeated timestamps'}
    assumed = se.build(ep, {'n': 5, 'ks': [0, 2, 4]})
    assert assumed['series'][0]['values'] == trace['values']
    assert not se.bind(assumed, [finding])['findings']
    assert 'alignment' in se.bind(assumed, [finding])['unbound_findings'][0]['reason']
    legacy = copy.deepcopy(assumed)
    legacy['sensors'][0].pop('episode_alignment_verified')
    legacy['findings'] = [finding]
    assert not se.compatible(legacy, ep['context'])
    merged = se.merge([({'piece': {'index': 1, 't0_s': 10}}, {'sensor_evidence': legacy})], ep['context'])
    assert not merged['findings'] and merged['unbound_findings'][0]['finding']['start_s'] == 10.1
    ep['context'].pop('camera_clock')
    ep['signal_meta']['right_pressure']['aligned_by'] = 'assumed start'
    assert not se.bind(se.build(ep, {'n': 5, 'ks': [0, 2, 4]}), [finding])['findings']


def test_large_arrays_and_long_recordings_have_explicit_bounded_coverage():
    n = 10000
    a = np.zeros((n, 256), dtype=np.float32)
    a[5123, 17] = 1000
    ep = episode({'touch_pad': a}, [{'name': 'touch_pad', 'shape': [16, 16]}])
    doc = se.build(ep, {'n': n, 'ks': [0, n - 1]})
    assert len(doc['series']) <= se.MAX_SERIES
    assert all(len(r['times']) <= se.MAX_SAMPLES for r in doc['series'])
    assert any(1000 in r['values'] for r in doc['series'])
    assert doc['limitations']
    assert len(se.prompt(doc).encode()) <= se.MAX_PROMPT_BYTES


def test_piece_evidence_keeps_clock_offsets_and_independent_sensor_ids():
    ep = episode({'contact': np.array([[0], [1], [1], [0], [0]])}, [{'name': 'contact', 'shape': [1]}],
                 {'contact': {'role': 'contact_state'}})
    doc = se.build(ep, {'n': 5, 'ks': [0, 2, 4]})
    finding = {'start_s': .1, 'end_s': .3, 'inspect_s': .2, 'headline': 'Contact ends',
               'claim_type': 'contact_change', 'evidence': [{'sensor_id': doc['sensors'][0]['id'],
                   'series_id': doc['series'][0]['id'], 'time_s': [.2, .3]}]}
    bound = se.bind(doc, [finding])
    bound['findings'][0]['review_status'] = 'rejected'
    merged = se.merge([({'piece': {'index': 1, 't0_s': 10}}, {'sensor_evidence': bound}),
                       ({'piece': {'index': 3, 't0_s': 20}}, {'sensor_evidence': bound})], ep['context'])
    assert all(finding['review_status'] == 'rejected' for finding in merged['findings'])
    assert merged['findings'][1]['inspect_s'] == 20.2
    assert merged['findings'][1]['evidence'][0]['time_s'] == [20.2, 20.3]
    assert merged['findings'][0]['evidence'][0]['series_id'] == merged['series'][0]['id']
    assert merged['series'][0]['id'] != merged['series'][1]['id']
    assert se.compatible(merged, ep['context'])
    edited = copy.deepcopy(ep['context'])
    edited['data_dictionary']['entries']['field_0'] = {'role': 'annotation', 'provenance': 'human'}
    stale = se.merge([({'piece': {'index': 1, 't0_s': 10}}, {'sensor_evidence': bound})], edited)
    assert not stale['findings']
    assert stale['unbound_findings'][0]['finding']['start_s'] == 10.1


@pytest.mark.parametrize('parts_folder', ['pieces', 'custom_sidecars'])
def test_merged_piece_finding_follows_its_piece_source(tmp_path, parts_folder):
    from board import build
    from label.evidence_access import source_proof
    parent = tmp_path / 'episodes' / 'episode_test'
    piece = tmp_path / parts_folder / 'episode_test__p01'
    for directory in (parent, piece):
        directory.mkdir(parents=True)
        (directory / 'context.json').write_text(json.dumps({'state_kind': 'none',
            **({'piece': {'of': parent.name, 'index': 1}} if directory == piece else {})}))
        (directory / 'sources.json').write_text(json.dumps({'exo': {'packed': 'camera.bin'}}))
        (directory / 'camera.bin').write_bytes(b'encoded first frames')
        np.savez(directory / 'signals.npz', contact=np.array([[0], [1], [1], [0], [0]]))
    ep = episode({'contact': np.array([[0], [1], [1], [0], [0]])},
                 [{'name': 'contact', 'shape': [1]}], {'contact': {'role': 'contact_state'}})
    ctx = ep['context']
    ctx['state_kind'] = 'none'
    doc = se.build(ep, {'n': 5, 'ks': [0, 2, 4]})
    finding = {'start_s': .1, 'end_s': .3, 'headline': 'Contact changes',
               'claim_type': 'contact_change', 'evidence': [{'sensor_id': doc['sensors'][0]['id'],
                   'series_id': doc['series'][0]['id'], 'time_s': [.1, .3]}]}
    bound = se.bind(doc, [finding])
    bound['source_proof'] = source_proof(ctx, piece)
    merged = se.merge([({'piece': {'index': 1, 't0_s': 10}, **ctx},
                        {'episode_dir': str(piece), 'sensor_evidence': bound})], ctx, parent)
    assert merged['findings']
    assert se.compatible(merged, ctx, parent)
    (piece / 'camera.bin').write_bytes(b'encoded changed frames')
    assert not se.compatible(merged, ctx, parent)
    saved = {'sensor_evidence': copy.deepcopy(merged)}
    build.add_context(saved, ctx, parent)
    assert not saved['sensor_evidence']['findings']
    assert saved['sensor_evidence']['withheld_findings'] == merged['findings']


def test_normal_request_and_reply_use_shared_contract(tmp_path, monkeypatch):
    from tests.test_label import _packed_mp4
    from label import episode as me, harness
    from board import to_board

    ep_dir = tmp_path / 'episode_other_sensor'
    ep_dir.mkdir()
    movie = ep_dir / 'scene.mp4'
    _packed_mp4(movie, 15)
    ctx = {'dataset': 'unfamiliar_sensor', 'profile': 'ego_head', 'state_kind': 'none', 'fps': 30,
           'cameras': {'exo': {'name': 'scene', 'width': 64, 'height': 36}},
           'signals': [{'name': 'contact', 'key': 's0', 'dims': 1, 'shape': [1]}]}
    (ep_dir / 'context.json').write_text(json.dumps(ctx))
    (ep_dir / 'sources.json').write_text(json.dumps({'exo': {'packed': str(movie), 'base_s': 0, 'n_frames': 15}}))
    np.savez(ep_dir / 'signals.npz', s0=np.array([[0]] * 4 + [[1]] * 6 + [[0]] * 5))
    request = me.build_request(ep_dir, cell_w=192)
    assert 'sensor_findings' in request['schema_fields']
    assert 'SENSOR EVIDENCE FOR ADDITIONAL ANNOTATION' in request['prompt']
    evidence = request['sensor_evidence']
    sensor, row = evidence['sensors'][0], evidence['series'][0]
    finding = {'start_s': .1, 'end_s': .3, 'headline': 'Contact remains active during the hidden movement',
               'claim_type': 'response_change', 'evidence': [{'sensor_id': sensor['id'], 'series_id': row['id'],
                                                           'time_s': [row['times'][4], row['times'][8]]}]}
    def provider_reply(*args, **kwargs):
        return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({'sensor_findings': [finding]})}}]}
    monkeypatch.setattr(harness, 'call_model', provider_reply)
    out = harness.label_episode(ep_dir, tmp_path / 'result.json', model='test-provider', reasoning='low',
                                api_key='sk-or-fixture', max_tokens=1000, timeout=1, cell_w=192)
    from label import sensor_copy
    assert out['sensor_evidence']['findings'] == [{**finding, 'copy_contract_version': sensor_copy.VERSION}]
    assert out['labels']['sensor_findings'] == [finding]
    board = to_board.convert(out, 'unfamiliar_sensor')['sensor_evidence']['findings']
    # A cited channel that is not part of a declared array is drawn by the board as it always is, so the finding
    # reaches the board unchanged.
    assert board == [{**finding, 'copy_contract_version': sensor_copy.VERSION}]


def test_shared_viewer_uses_cited_channels_units_and_optical_evidence():
    import shutil
    import subprocess

    script = r'''
const fs=require('fs'),assert=require('assert');
const src=fs.readFileSync(process.argv[1],'utf8');
const start=src.indexOf('// ================= sensor evidence:');
const stop=src.indexOf('// ================= touch:',start);
const api=new Function('esc','fmtT','videoSrc',src.slice(start,stop)+'return {sensorEvidence,gripFindingStripHtml,gripEvidencePanelHtml,sensorEvidenceHtml};')(
  s=>String(s).replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[c])),String,(id,v)=>'/clip/'+id+'/'+v);
const f={start_s:0,end_s:.3,headline:'Sideways force changes',detail:'The object pulls against the held grip',confidence:'medium',adds_beyond_video:true,
  evidence:[{sensor_id:'pad',series_id:'axis-x',time_s:[0,.2]}]};
const doc={version:1,sensors:[{id:'pad',name:'Left pad',kind:'numeric'}],
  series:[{id:'axis-x',sensor_id:'pad',label:'Fx',unit:'N',times:[0,.1,.2,.3],values:[0,-2,null,3]},
          {id:'unused',sensor_id:'pad',label:'Unrelated',unit:'Nm',times:[0,.1],values:[0,5]}],findings:[f],limitations:[]};
const E=api.sensorEvidence({sensor_evidence:doc});
const html=api.gripFindingStripHtml({...E.insights[0],phase:'Now'},E.grip);
assert(html.includes('Fx (N)')&&!html.includes('Unrelated'));
assert(!html.includes('finger')&&!html.includes('pressure change'));
const panel=api.sensorEvidenceHtml(E,[]);
assert(panel.includes('Recorded sensor evidence')&&!panel.includes('OpenTouch')&&!panel.includes('matched video-and-pose control produced'));
const optical={version:1,sensors:[{id:'camera:extra1',name:'Tactile camera',kind:'image',view:'extra1'}],series:[],
  findings:[{...f,headline:'Contact shifts across the fingertip',evidence:[{sensor_id:'camera:extra1',time_s:[0,.3]}]}]};
const O=api.sensorEvidence({sensor_evidence:optical});O.grip.media={eid:'episode'};
const opticalOverlay=api.gripFindingStripHtml({...O.insights[0],phase:'Now'},O.grip);
assert(opticalOverlay.includes('Contact shifts across the fingertip')&&!opticalOverlay.includes('<video'));
const opticalPanel=api.gripEvidencePanelHtml(O.grip,O.insights[0]);
assert(opticalPanel.includes('data-sensor-evidence-video')&&opticalPanel.includes('/clip/episode/extra1'));
const rejected=api.sensorEvidence({sensor_evidence:{...doc,findings:[{...f,review_status:'rejected'}]}});
assert.equal(rejected.insights.length,0);
'''
    root = Path(__file__).resolve().parents[1]
    run = subprocess.run([shutil.which('node'), '-e', script, str(root / 'board/index.html')], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
