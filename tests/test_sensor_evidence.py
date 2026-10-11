"""Saved sensor observations stay distinct from model interpretation in the viewer."""
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(not shutil.which('node'), reason='no node')
def test_sensor_evidence_presentation():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([shutil.which('node'), str(root / 'tests/sensor_evidence.js'),
                             str(root / 'board/serve.py')], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not shutil.which('node'), reason='no node')
def test_sparse_supplied_evidence_inspection():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([shutil.which('node'), str(root / 'tests/sparse_evidence.js'),
                             str(root / 'board/serve.py'),
                             str(root / 'tests/fixtures/depth_release_cited_samples.json')],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not shutil.which('node'), reason='no node')
def test_recorded_camera_hierarchy_and_visible_sensor_interpretation(tmp_path):
    import copy
    from board.dictionary_projection import apply

    # The same exact prepared camera must change presentation with the current
    # interpretation, without changing source assignment or stale saved labels.
    ctx = {'episode_id': 'ep', 'cameras': {'extra1': {'name': 'cam7'}},
           'data_dictionary': {'schema': 1, 'episode_id': 'ep', 'status': 'success',
               'fields': [{'id': 'image', 'kind': 'camera', 'name': 'cam7',
                           'bindings': [{'episode': 'ep', 'context_path': 'cameras/extra1'}]}],
               'entries': {'image': {'role': 'tactile_image', 'provenance': 'machine'}}}}
    saved = {'camera_views': ['exo', 'extra1'], 'camera_labels': ['head', 'cam7']}
    projected = apply(copy.deepcopy(saved), ctx, tmp_path)
    assert projected['camera_presentation']['extra1']['role'] == 'tactile'
    ctx['data_dictionary']['entries']['image'] = {'role': 'rgb_image', 'provenance': 'human'}
    revised = apply(copy.deepcopy(projected), ctx, tmp_path)
    assert revised['camera_presentation']['extra1']['role'] == 'other'
    assert revised['camera_views'] == saved['camera_views']
    assert 'camera_presentation' not in saved

    root = Path(__file__).resolve().parents[1]
    script = r'''
const assert=require('assert'), fs=require('fs');
const {src,piece}=require(process.argv[1])();
const rows=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const begin=src.indexOf('// ================= sensor evidence:');
const end=src.indexOf('// ================= touch:',begin);
const api=new Function('esc','fmtT',piece('episodeCams')+piece('cameraPresentation')
  +piece('supportingCameras')+piece('unshownCams')+src.slice(begin,end)
  +'return {episodeCams,supportingCameras,sensorEvidence,gripFindingStripHtml};')(String,String);
const optical=rows['audit_optical_handoff.json'];
const original=JSON.stringify(optical);
const support=api.supportingCameras(optical);
assert.deepEqual(support.map(c=>c.view),['left','right','unshown1','extra1','extra2','extra3','extra4'],
  'the second scene view precedes tactile pictures, even though the model did not see it');
assert(support.slice(3).every(c=>c.role==='tactile'));
assert.equal(api.episodeCams(optical).main,'exo','keep the labelled scene clock and main player');
const f=api.sensorEvidence(optical).insights[0];
const card=api.gripFindingStripHtml({...f,phase:'Now'});
assert(card.includes(f.claim),'the physical interpretation must be visible, not only a changes headline');
assert(card.includes('do not establish calibrated force or confirmed slip'), 'retain uncertainty in legacy interpretation');
const short=api.gripFindingStripHtml({...f,phase:'Now',overlayDetail:'Contact may shift across the pads; slip is not established.'});
assert(short.includes('Contact may shift across the pads; slip is not established.'));
assert(!short.includes(f.claim),'a model-authored compact explanation replaces the longer fallback');
assert.equal(JSON.stringify(optical),original,'presentation does not alter annotations or camera identity');
for(const name of ['audit_umi_rgb.json','audit_franka.json','audit_ego_rgb.json']) {
  const d=rows[name], base=api.episodeCams(d);
  assert.deepEqual(api.supportingCameras(d).map(c=>c.view),base.side,
    'ordinary RGB layouts retain their camera order');
}
// A reviewed role takes precedence over a camera name. Unknown views retain stable ordering.
const roleData={camera_views:['exo','extra1','extra2'],camera_labels:['head','touch_panel','cam7'],
  camera_presentation:{extra1:{role:'scene'},extra2:{role:'tactile'}}};
assert.deepEqual(api.supportingCameras(roleData).map(c=>[c.view,c.role]),[['extra1','scene'],['extra2','tactile']]);
const opaque=rows['episode_waveform_sound_probe.json'];
opaque.camera_presentation={right:{role:'tactile'},extra1:{role:'tactile'},extra2:{role:'scene'},extra3:{role:'scene'}};
assert.equal(api.episodeCams(opaque).main,'extra2','a known scene camera leads over a misassigned sensor view');
assert.deepEqual(new Set([api.episodeCams(opaque).main,...api.supportingCameras(opaque).map(c=>c.view)]),
  new Set(opaque.camera_views),'all original camera identities remain available');
const denied={...f,review_status:'rejected'};
assert.equal(api.sensorEvidence({sensor_evidence:{version:1,findings:[denied]}}).insights.length,0);
'''
    result = subprocess.run([shutil.which('node'), '-e', script,
        str(root / 'tests/page_functions.js'),
        str(root / 'tests/fixtures/camera_and_finding_presentation.json')], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
