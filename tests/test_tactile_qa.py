"""Regression checks for evidence boundaries found during the second QA pass."""
import copy
import json

import numpy as np
import pytest

from label import sensor_evidence as se
from tests.test_tactile_contract import episode


def image_and_numeric():
    ep = episode({'pressure': np.arange(5).reshape(5, 1)}, [{'name': 'pressure', 'shape': [1]}],
                 cameras={'exo': {'name': 'scene'}, 'extra1': {'name': 'gelsight'}})
    doc = se.build(ep, {'n': 5}, shown={'extra1': [0, 4]})
    finding = {'start_s': 0, 'end_s': .4, 'headline': 'Contact shape changes', 'claim_type': 'image_deformation',
               'evidence': [{'sensor_id': 'camera:extra1', 'time_s': [0, .4]}]}
    return doc, finding


def test_image_citation_cannot_borrow_numeric_samples():
    doc, finding = image_and_numeric()
    borrowed = copy.deepcopy(finding)
    borrowed['evidence'][0].update(series_id=doc['series'][0]['id'], time_s=[.1, .3])
    bound = se.bind(doc, [finding, borrowed])
    assert bound['findings'] == [finding]
    assert bound['unbound_findings'][0]['finding'] == borrowed


@pytest.mark.parametrize('key,value', [('sensor_id', []), ('series_id', {}), ('sensor_id', 3)])
def test_malformed_model_reference_is_saved_without_crashing(key, value):
    doc, finding = image_and_numeric()
    finding['evidence'][0][key] = value
    bound = se.bind(doc, [finding])
    assert not bound['findings']
    assert bound['unbound_findings'][0]['finding'] == finding


def test_storage_summary_cannot_be_presented_as_individual_fingers_or_force_axes():
    ep = episode({'pressure': np.array([[0, 2, 4], [0, 5, 10], [0, 2, 4], [0, 5, 10], [0, 2, 4]])},
                 [{'name': 'pressure', 'shape': [3], 'names': ['lowest', 'mean', 'highest'], 'summary_of': 384}],
                 {'pressure': {'role': 'force_torque', 'layout': [
                     {'name': 'Thumb', 'start': 0, 'count': 1}, {'name': 'Index finger', 'start': 1, 'count': 2}]}})
    doc = se.build(ep, {'n': 5})
    assert doc['sensors'][0]['capabilities'] == ['response_change']
    assert [r['label'] for r in doc['series']] == ['lowest', 'mean', 'highest']
    assert all('summary' in r['operation'] for r in doc['series'])
    assert any('384' in note and 'summary' in note for note in doc['limitations'])


def test_clock_changes_invalidate_saved_evidence():
    ep = episode({'contact': np.zeros((5, 1))}, [{'name': 'contact', 'shape': [1]}])
    doc = se.build(ep, {'n': 5})
    for key, value in [('fps', 20), ('camera_clock', {'exo': {'clock_problem': 'invalid'}})]:
        changed = {**ep['context'], key: value}
        assert not se.compatible(doc, changed)
    changed = copy.deepcopy(ep['context'])
    changed['signals'][0]['clock_problem'] = 'invalid'
    assert not se.compatible(doc, changed)


def test_stitch_does_not_reinterpret_saved_force_after_units_or_clock_change():
    ep = episode({'load': np.arange(5).reshape(5, 1)}, [{'name': 'load', 'shape': [1], 'unit': 'N'}],
                 {'load': {'role': 'normal_force'}})
    doc = se.build(ep, {'n': 5})
    finding = {'start_s': .1, 'end_s': .3, 'headline': 'Recorded force increases', 'claim_type': 'force_change',
               'evidence': [{'sensor_id': 'signal:load', 'series_id': doc['series'][0]['id'], 'time_s': [.1, .3]}]}
    bound = se.bind(doc, [finding])
    unit_changed = copy.deepcopy(ep['context'])
    unit_changed['signals'][0]['unit'] = 'Nm'
    for ctx in [unit_changed, {**ep['context'], 'fps': 20}]:
        result = se.merge([({'piece': {'index': 1, 't0_s': 10}}, {'sensor_evidence': bound})], ctx)
        assert not result['findings']
        assert result['unbound_findings'][0]['finding']['start_s'] == 10.1


def test_request_cites_only_camera_frames_that_reach_the_model(tmp_path, monkeypatch):
    from label import episode as me
    from tests.test_label import _packed_mp4
    ep_dir = tmp_path / 'episode_camera_boundary'
    ep_dir.mkdir()
    movie = ep_dir / 'scene.mp4'
    _packed_mp4(movie, 15)
    ctx = {'profile': 'ego_head', 'state_kind': 'none', 'fps': 30,
           'cameras': {'exo': {'name': 'scene', 'width': 64, 'height': 36},
                       'extra1': {'name': 'gelsight', 'width': 64, 'height': 36}}}
    (ep_dir / 'context.json').write_text(json.dumps(ctx))
    (ep_dir / 'sources.json').write_text(json.dumps({v: {'packed': str(movie), 'base_s': 0, 'n_frames': 15}
                                                    for v in ctx['cameras']}))
    frames = me.frames
    excluded = []
    def decoded_with_unused_frames(ep, pl, *args, **kwargs):
        got = frames(ep, pl, *args, **kwargs)
        blocked = pl['ks'][0]
        unused = next(k for k in range(15) if k not in pl['ks'])
        ep.setdefault('no_frame', {}).setdefault('extra1', set()).add(blocked)
        got['extra1'][unused] = next(iter(got['extra1'].values()))
        excluded.extend([blocked / 30, unused / 30])
        return got
    monkeypatch.setattr(me, 'frames', decoded_with_unused_frames)
    request = me.build_request(ep_dir, cell_w=192)
    camera = next(s for s in request['sensor_evidence']['sensors'] if s['id'] == 'camera:extra1')
    assert not set(camera['times']).intersection(excluded)


def test_repeated_context_refresh_preserves_withheld_findings(tmp_path):
    from board.build import add_context
    ep = episode({'contact': np.zeros((5, 1))}, [{'name': 'contact', 'shape': [1]}])
    doc = se.build(ep, {'n': 5})
    doc['findings'] = [{'headline': 'Saved human-reviewed finding', 'review_status': 'accepted'}]
    ep['context']['data_dictionary']['entries']['field_0'] = {'role': 'annotation', 'provenance': 'human'}
    board = {'sensor_evidence': doc}
    add_context(board, ep['context'], tmp_path)
    add_context(board, ep['context'], tmp_path)
    assert board['sensor_evidence']['withheld_findings'][0]['review_status'] == 'accepted'


def test_all_force_axes_follow_the_video_in_the_detail_panel():
    import shutil
    import subprocess
    from pathlib import Path
    script = r'''
const fs=require('fs'),assert=require('assert'),s=fs.readFileSync(process.argv[1],'utf8');
const nodes=n=>Array.from({length:n},(_,i)=>({dataset:{gripMiniDot:String(i)},attrs:{visibility:'hidden'},setAttribute(k,v){this.attrs[k]=v}}));
const strip=nodes(3),panel=nodes(6);
const scope=dots=>({querySelectorAll(q){return q==='[data-grip-mini-dot]'?dots:[]}});
const document=scope([...strip,...panel]);
const start=s.indexOf('// ================= sensor evidence:'),stop=s.indexOf('// ================= touch:',start);
const api=new Function('esc','fmtT','videoSrc','document',s.slice(start,stop)+'return {sensorEvidence,syncGripOverlayPressure};')(String,String,()=>'',document);
const evidence=Array.from({length:6},(_,i)=>({sensor_id:'force',series_id:'axis'+i,time_s:[0,.5,1]}));
const G={version:1,sensors:[{id:'force',kind:'numeric',name:'Measured force'}],
 series:evidence.map((e,i)=>({id:e.series_id,sensor_id:'force',label:'Axis '+i,times:[0,.5,1],values:[0,i+1,0],sample_period_s:.5})),
 findings:[{start_s:0,end_s:1,headline:'Force changes',adds_beyond_video:true,evidence}]};
const E=api.sensorEvidence({sensor_evidence:G});
api.syncGripOverlayPressure(E.grip,E.insights[0],.5,scope(strip));
api.syncGripOverlayPressure(E.grip,E.insights[0],.5,scope(panel),Infinity);
assert(panel.every(n=>n.attrs.visibility==='visible'&&n.attrs.cx===90),'all six force axes must show the current reading');
assert(strip.every(n=>n.attrs.visibility==='visible'&&n.attrs.cx===90),'the compact overlay stays independent');
'''
    source = Path(__file__).resolve().parents[1] / 'board/serve.py'
    run = subprocess.run([shutil.which('node'), '-e', script, str(source)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
