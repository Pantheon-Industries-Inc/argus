"""Saved sensor observations stay distinct from model interpretation in the viewer."""
import shutil
import subprocess
from pathlib import Path

import pytest
import complete_reply


def test_observed_camera_content_uses_native_ids_and_current_roles_for_exports(tmp_path, monkeypatch):
    import copy
    import hashlib
    import json
    from test_label import _packed_episode
    from label import evidence_access, harness
    from board import dictionary_projection, public_export, static, to_board

    folder, _ = _packed_episode(tmp_path)
    path = folder / 'context.json'
    ctx = json.loads(path.read_text())
    ctx['cameras'] = {v: {'name': 'opaque camera', 'width': 64, 'height': 36} for v in ('exo', 'left', 'right')}
    ctx['data_dictionary'] = {'schema': 1, 'episode_id': folder.name, 'status': 'success',
        'fields': [{'id': v, 'kind': 'camera', 'name': 'opaque camera',
                    'bindings': [{'episode': folder.name, 'context_path': 'cameras/' + v}]} for v in ctx['cameras']],
        'entries': {v: {'role': 'rgb_image', 'provenance': 'machine'} for v in ctx['cameras']}}
    path.write_text(json.dumps(ctx))
    media = [Path(s['packed']) for s in json.loads((folder / 'sources.json').read_text()).values()]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in media}
    fields = {'config': {'views': ['exo', 'left', 'right'], 'cam_labels': ['opaque camera'] * 3,
                         'camera_observation_views': ['exo', 'left', 'right'],
                         'schema_fields': ['camera_observations']},
              'input_identity': {'source_proof': evidence_access.source_proof(ctx, folder)}}
    rows = [{'view': 'exo', 'content': 'sensor', 'evidence': 'A deforming marked surface fills this view.'},
            {'view': 'left', 'content': 'sensor', 'evidence': 'Only another marked contact surface is visible.'},
            {'view': 'right', 'content': 'scene', 'evidence': 'The device and work surface are visible.'}]
    # Exercise the real annotation writer with an offline reply, without invoking a provider.
    monkeypatch.setattr(harness, 'call_model_once', lambda *a, **k: {
        'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({
            **complete_reply.labels(), 'camera_observations': rows})}}], 'usage': {'cost': 0.0}})
    recorded = harness.label_episode(folder, tmp_path / 'reply.json', model='openai/gpt-6.1-sol',
        reasoning='high', api_key='offline', max_tokens=2000, timeout=1, cell_w=128)
    assert recorded['labels']['camera_observations'] == rows
    assert recorded['camera_observations']['observations'] == rows
    assert recorded['camera_observations']['source_proof'] == recorded['input_identity']['source_proof']
    # The native YAM reply used a displayed alias instead of the supplied exo ID.
    # Neither that reply nor an older saved partial envelope may promote its wrists.
    for incomplete in ([{**rows[0], 'view': 'exo_cam'}, *rows[1:]], rows[1:]):
        findings = [{'signal': 'recorded_force', 'observation': 'Recorded values change during the work.'}]
        monkeypatch.setattr(harness, 'call_model_once', lambda *a, **k: {
            'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({
                **complete_reply.labels(), 'camera_observations': incomplete,
                'sensor_findings': findings})}}], 'usage': {'cost': 0.0}})
        partial = harness.label_episode(folder, tmp_path / 'partial.json', model='openai/gpt-6.1-sol',
            reasoning='high', api_key='offline', max_tokens=2000, timeout=1, cell_w=128)
        assert partial['labels']['camera_observations'] == incomplete
        assert partial['labels']['sensor_findings'] == findings
        assert not partial['camera_observations']['observations']
        assert partial['camera_observations']['withheld_observations']
        projected = dictionary_projection.apply(to_board.convert(partial, 'native partial'), ctx, folder)
        assert static.shown_cams(projected) == ['exo', 'left', 'right']
        assert all(r.get('provenance') != 'observed' for r in projected['camera_presentation'].values())
        legacy = to_board.convert(recorded, 'native legacy')
        legacy['camera_observations']['observations'] = rows[1:]
        legacy['camera_observations']['withheld_observations'] = []
        projected = dictionary_projection.apply(legacy, ctx, folder)
        assert projected['camera_observations']['source_current'] is False
        assert not projected['camera_observations']['observations']
        assert static.shown_cams(projected) == ['exo', 'left', 'right']
        authoritative = copy.deepcopy(ctx)
        authoritative['data_dictionary']['entries']['right'] = {'role': 'scene', 'provenance': 'human'}
        authoritative['cameras']['exo']['role'] = 'sensor_view'
        projected = dictionary_projection.apply(to_board.convert(partial, 'native partial'), authoritative, folder)
        assert projected['camera_presentation']['right'] == {'role': 'scene'}
        assert projected['camera_presentation']['exo'] == {'role': 'sensor'}
    unclear = harness.bind_camera_observations([{**rows[0], 'content': 'unclear'}, *rows[1:]], fields)
    assert len(unclear['observations']) == 3
    bound = harness.bind_camera_observations(rows, fields)
    result = {**fields, 'episode_dir': str(folder), 'parse_ok': True, 'labels': complete_reply.labels(),
              'camera_observations': bound}
    saved = to_board.convert(result, 'native')
    shown = dictionary_projection.apply(copy.deepcopy(saved), ctx, folder)
    assert shown['camera_presentation']['right'] == {'role': 'scene', 'provenance': 'observed'}
    assert shown['camera_presentation']['exo'] == {'role': 'sensor', 'provenance': 'observed'}
    assert static.shown_cams(shown) == ['right', 'exo', 'left']
    assert static.shown_cams({'camera_views': ['right', 'extra1', 'extra2', 'extra3', 'extra4'],
        'camera_presentation': {'right': {'role': 'tactile'}, 'extra1': {'role': 'sensor'},
                               'extra2': {'role': 'scene'}, 'extra3': {'role': 'tactile'},
                               'extra4': {'role': 'sensor'}}}) == ['extra2', 'extra1', 'extra4', 'right', 'extra3']
    assert public_export.project(shown)['camera_observations']['observations'] == rows
    assert saved['camera_observations'] == bound and 'camera_presentation' not in saved
    assert 'camera_observations' not in to_board.convert({**result, 'parse_ok': False}, 'native')
    inherited = copy.deepcopy(ctx)
    inherited['data_dictionary']['machine_entries'] = copy.deepcopy(ctx['data_dictionary']['entries'])
    inherited['data_dictionary']['entries']['right'].update(meaning='A reviewed description', provenance='human')
    assert dictionary_projection.apply(copy.deepcopy(saved), inherited, folder)['camera_presentation']['right']['role'] == 'scene'
    inherited['data_dictionary']['override_history'] = [{'field_id': 'right', 'after': {'role': 'rgb_image'}}]
    assert dictionary_projection.apply(copy.deepcopy(saved), inherited, folder)['camera_presentation']['right']['role'] == 'other'

    # The current exact human interpretation wins without rewriting the saved model receipt.
    revised = copy.deepcopy(ctx)
    revised['data_dictionary']['entries']['right'] = {'role': 'tactile_image', 'provenance': 'human'}
    revised['data_dictionary']['entries']['left'] = {'role': 'scene', 'provenance': 'human'}
    shown = dictionary_projection.apply(copy.deepcopy(saved), revised, folder)
    assert shown['camera_presentation']['right']['role'] == 'tactile'
    assert static.shown_cams(shown)[0] == 'left'
    revised['data_dictionary']['entries']['right'] = {'role': 'unknown', 'provenance': 'human'}
    assert dictionary_projection.apply(copy.deepcopy(saved), revised, folder)['camera_presentation']['right']['role'] == 'other'
    # Explicit recorded content also takes precedence over a model observation.
    revised = copy.deepcopy(ctx)
    revised['cameras']['right']['role'] = 'tactile_image'
    assert dictionary_projection.apply(copy.deepcopy(saved), revised, folder)['camera_presentation']['right']['role'] == 'tactile'

    malformed = rows + [{'view': 'missing', 'content': 'scene', 'evidence': 'An invented camera.'}, rows[-1]]
    checked = harness.bind_camera_observations(malformed, fields)
    assert not checked['observations']
    assert len(checked['withheld_observations']) >= 3
    unsolicited = copy.deepcopy(fields)
    unsolicited['config']['schema_fields'] = []
    assert not harness.bind_camera_observations(rows, unsolicited)['observations']
    unseen = copy.deepcopy(fields)
    unseen['config']['camera_observation_views'] = ['exo', 'left']
    assert not harness.bind_camera_observations(rows, unseen)['observations']
    assert [r['view'] for r in harness.bind_camera_observations(rows[:2], unseen)['observations']] == ['exo', 'left']
    unsupported = copy.deepcopy(fields)
    unsupported['input_identity']['source_proof'] = []
    assert not harness.bind_camera_observations(rows, unsupported)['observations']

    # A saved observation never promotes a view after its native source changes.
    media[-1].write_bytes(media[-1].read_bytes() + b'changed source')
    stale = dictionary_projection.apply(copy.deepcopy(saved), ctx, folder)
    assert stale['camera_observations']['source_current'] is False
    assert not stale['camera_observations']['observations']
    assert all(r['role'] != 'scene' for r in stale['camera_presentation'].values())
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in media[:-1]} == {p: before[p] for p in media[:-1]}


def test_duplicate_camera_names_keep_exact_sensor_owner_across_request_and_stitch(tmp_path):
    import json
    import hashlib
    from test_label import _packed_episode
    from label import episode, sensor_evidence
    ep_dir, _ = _packed_episode(tmp_path)
    path = ep_dir / 'context.json'
    ctx = json.loads(path.read_text())
    views = ['exo', 'left']
    for view in views:
        ctx['cameras'].setdefault(view, dict(ctx['cameras']['exo']))
        ctx['cameras'][view]['name'] = 'recorded camera'
    ctx['data_dictionary'] = {'schema': 1, 'episode_id': ep_dir.name, 'status': 'success',
        'fields': [{'id': view, 'kind': 'camera', 'name': 'recorded camera',
                    'bindings': [{'episode': ep_dir.name, 'context_path': 'cameras/' + view}]} for view in views],
        'entries': {'exo': {'role': 'scene', 'provenance': 'human'},
                    'left': {'role': 'tactile_image', 'provenance': 'human'}}}
    path.write_text(json.dumps(ctx))
    media = {Path(source['packed']) for source in json.loads((ep_dir / 'sources.json').read_text()).values()}
    assert media and all(p.is_file() for p in media)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in media}
    ep = episode.load(ep_dir)
    plan = episode.plan(ep)
    evidence = sensor_evidence.build(ep, plan, shown={v: plan['ks'] for v in views})
    assert [(s['id'], s['view'], s['interpretation']['role']) for s in evidence['sensors']] == [
        ('camera:left', 'left', 'tactile_image')]
    merged = sensor_evidence.merge([({**ctx, 'piece': {'index': 1, 't0_s': 0, 't1_s': 4}},
                                      {'sensor_evidence': evidence})], ctx)
    assert [(s['view'], s['interpretation']['role']) for s in merged['sensors']] == [('left', 'tactile_image')]
    assert {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in media} == before


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
  +piece('supportingCameras')+piece('cameraSectionLabel')+piece('sceneFirstLayout')+piece('cameraCaption')+piece('unshownCams')+src.slice(begin,end)
  +'return {episodeCams,supportingCameras,cameraSectionLabel,sceneFirstLayout,cameraCaption,sensorEvidence,gripFindingStripHtml,gripFindingDetailHtml};')(String,String);
const optical=rows['audit_optical_handoff.json'];
const original=JSON.stringify(optical);
const support=api.supportingCameras(optical);
assert.deepEqual(support.map(c=>c.view),['left','right','extra1','extra2','extra3','extra4','unshown1'],
  'unverified camera names do not manufacture scene or tactile roles');
assert(support.every(c=>c.role==='other'));
assert.equal(api.episodeCams(optical).main,'exo','keep the labelled scene clock and main player');
assert.equal(api.sensorEvidence(optical).insights.length,0,'a cropped audit row without backing sensors is not eligible');
// Exercise copy rendering separately from evidence eligibility.
const row=optical.sensor_evidence.findings[0];
const f={...row,start:row.start_s,end:row.end_s,generic:true,kind:'insight'};
const card=api.gripFindingStripHtml({...f,phase:'Now'});
assert(!card.includes(f.claim),'the full interpretation stays out of the compact video overlay');
assert(api.gripFindingDetailHtml(f).includes(f.claim),'the interpretation remains available in the evidence panel');
const short=api.gripFindingStripHtml({...f,phase:'Now',overlayDetail:'The right pad changes more than the left while the object remains held.'});
assert(short.includes('The right pad changes more than the left while the object remains held.'));
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
assert.deepEqual(api.supportingCameras(roleData).map(c=>[c.view,c.role]),[['exo','other'],['extra2','tactile']]);
assert.equal(api.episodeCams(roleData).main,'extra1');
const separateClock={camera_views:['exo'], unshown_cameras:[{view:'unshown1',clock_alignment:'unverified'}],
  camera_presentation:{unshown1:{role:'scene'}}};
assert.equal(api.episodeCams(separateClock).main,'exo','an unrelated clock must not drive the labelled timeline');
assert.equal(api.supportingCameras(separateClock)[0].view,'unshown1','retain its independently playable camera');
const generic={camera_views:['exo','left','right'],camera_presentation:{exo:{role:'sensor'},left:{role:'scene'},right:{role:'tactile'}}};
assert.equal(api.episodeCams(generic).main,'left');
assert.deepEqual(api.supportingCameras(generic).map(c=>c.role),['sensor','tactile']);
assert.equal(api.cameraSectionLabel('sensor'),'Sensor views');
assert.equal(api.cameraSectionLabel('tactile'),'Tactile cameras');
const mixed={camera_views:['right','extra1','extra2'],camera_presentation:{right:{role:'sensor'},extra1:{role:'tactile'},extra2:{role:'scene'}}};
assert.equal(api.sceneFirstLayout(mixed),true,'a verified scene leads sensor views without an exo slot');
const interleaved={camera_views:['right','extra1','extra2','extra3','extra4'],camera_presentation:{right:{role:'tactile'},extra1:{role:'sensor'},extra2:{role:'scene'},extra3:{role:'tactile'},extra4:{role:'sensor'}}};
assert.deepEqual(api.supportingCameras(interleaved).map(c=>c.view),['extra1','extra4','right','extra3'],
  'sensor and tactile sections remain contiguous with stable order within each');
const grippers={camera_views:['left','right'],camera_presentation:{left:{role:'scene'},right:{role:'scene'}}};
assert.equal(api.sceneFirstLayout(grippers),false,'ordinary RGB-only gripper layout is preserved');
assert(src.includes('<p class="grip-note">One camera view is shown.</p>'));
assert(!src.includes('A single-arm task: the dataset records one gripper camera.'));
assert.equal(api.cameraCaption({...mixed,camera_labels:['opaque sensor','pad','workbench']},'right'),'opaque sensor');
assert.equal(api.cameraCaption({...grippers,camera_mounting:{right:'fixed'}},'right'),'right camera');
assert.equal(api.cameraCaption({...grippers,camera_mounting:{right:'wrist'}},'right'),'right wrist');
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
