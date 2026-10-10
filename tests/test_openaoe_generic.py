"""Structured video notes retain their source claims through generic upload conversion."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from label import episode
from prepare import formats, openaoe
from test_adapter_output_identity import original_context
from request_identity import assert_same_model_inputs
from test_formats import _clip


def clip(root, *, annotation_name='ego_annotation/ego_action_annotation.json'):
    root.mkdir(parents=True)
    _clip(root / 'raw_video.mp4', 20)
    annotation = [
        {'start_ts': '0.125', 'end_ts': 0.275,
         'atomic_action': [{'verb': 'align', 'object': 'fabric', 'hand': 'both', 'confidence': 0.37}],
         'description': 'retain this complete claim', 'unknown': {'calibration': [1, 2, 3]}},
        {'start_ts': 0.275, 'end_ts': 0.625, 'atomic_action': [], 'scene': 'at the table'}]
    path = root / annotation_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(annotation))
    info = {'deviceInfo': {'brand': 'fixture', 'model': 'phone', 'unknown': False},
            'cameraParams': {'resolution': [64, 36], 'calibration': {'fx': 53}},
            'unmeasured': {'rate': 99}}
    (root / 'video_info.json').write_text(json.dumps(info))
    return annotation, info


def convert(root, out, monkeypatch, *, legacy=False, rig='ego_head', ownership=None):
    discover = formats.upload_adapters
    with monkeypatch.context() as patch:
        patch.setattr(formats, 'upload_adapters', lambda kind: [openaoe] if legacy and kind == 'video'
                      else [] if kind == 'video' else discover(kind))
        report = formats.convert(root, rig, out, 'native parity', 900, ownership_context=ownership)
    assert not report['failed'] and len(report['episodes']) == 1, report
    ep = out / report['episodes'][0]['episode_id']
    return ep, json.loads((ep / 'context.json').read_text()), report


def hashes(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def test_native_public_upload_matches_forced_generic_frames_clocks_and_claims(tmp_path, monkeypatch):
    root = tmp_path / 'original_clip'
    annotation, info = clip(root)
    before = hashes(root)
    old_ep, old, _ = convert(root, tmp_path / 'legacy', monkeypatch, legacy=True)
    new_ep, new, report = convert(root, tmp_path / 'generic', monkeypatch)
    assert old_ep.name == new_ep.name == 'episode_original_clip'
    for key in ['profile', 'state_kind', 'fps', 'n_state_frames', 'duration_s', 'cameras']:
        assert new[key] == old[key], key
    for step, prior in zip(new['annotation_subtasks'], old['annotation_subtasks']):
        assert {k: step[k] for k in ['t0', 't1', 'label']} == {k: prior[k] for k in ['t0', 't1', 'label']}
    assert len(new['annotation_subtasks']) == 2
    assert new['annotation_subtasks'][0]['raw_times'] == ['0.125', 0.275]
    assert new['annotation_subtasks'][0]['source'] == 'ego_annotation/ego_action_annotation.json'
    assert new['annotation_subtasks'][0]['notes'] == annotation[0]
    assert 'ok' not in new['annotation_subtasks'][0]
    assert new['uploader_notes']['ego_annotation/ego_action_annotation.json'] == annotation
    assert new['uploader_notes']['video_info.json'] == info
    assert new['source']['device'] == old['source']['device'] == 'fixture phone'
    assert new['source']['resolution'] == old['source']['resolution'] == [64, 36]
    assert new['source']['annotation_segments'] == old['source']['annotation_segments'] == 2
    assert not new.get('instruction')
    assert not (new_ep / 'state.npz').exists() and not (new_ep / 'signals.npz').exists()
    clocks = {}
    with np.load(old_ep / 'times.npz') as a, np.load(new_ep / 'times.npz') as b:
        assert a.files == b.files
        for key in a.files:
            assert a[key].dtype == b[key].dtype and a[key].tobytes() == b[key].tobytes()
            clocks[key] = {'dtype': str(b[key].dtype), 'shape': list(b[key].shape),
                           'sha256': hashlib.sha256(b[key].tobytes()).hexdigest(), 'values': b[key].tolist()}
    assert json.loads((old_ep / 'sources.json').read_text()) == json.loads((new_ep / 'sources.json').read_text())
    requests = [episode.build_request(p) for p in [old_ep, new_ep]]
    images = [[part for part in req['content'] if part['type'] == 'image_url'] for req in requests]
    assert images[0] == images[1] and images[0]
    assert requests[0]['plan']['ks'] == requests[1]['plan']['ks']
    assert hashes(root) == before
    print(json.dumps({'source_sha256': before, 'frames': new['n_state_frames'],
                      'selected_indices': requests[1]['plan']['ks'], 'request_images': len(images[1]),
                      'clock_arrays': clocks,
                      'image_payload_sha256': [hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
                                               for p in images[1]],
                      'boundaries': [[s['t0'], s['t1']] for s in new['annotation_subtasks']],
                      'differences': ['qualified annotation clock', 'complete attributed JSON',
                                      'no invented success', 'no undeclared physical collection claims']}))


def test_selected_original_root_keeps_generic_identity_and_request(tmp_path, monkeypatch):
    root = tmp_path / 'original_clip'
    clip(root)
    ownership = {**original_context(root), 'root_name': root.name}
    selected = tmp_path / 'selected_alias'
    shutil.copytree(root, selected)
    full, _, _ = convert(root, tmp_path / 'full', monkeypatch)
    subset, _, _ = convert(selected, tmp_path / 'subset', monkeypatch, ownership=ownership)
    assert full.name == subset.name == 'episode_original_clip'
    assert_same_model_inputs(full, subset)


def test_generic_structural_notes_use_content_and_keep_caller_rig(tmp_path, monkeypatch):
    root = tmp_path / 'neutral_clip'
    annotation, _ = clip(root, annotation_name='annotations/motions.json')
    (root / 'raw_video.mp4').rename(root / 'camera.mp4')
    (root / 'instruction.txt').write_text('fold the fabric')
    ep, ctx, _ = convert(root, tmp_path / 'output', monkeypatch, rig='handheld_gripper')
    assert ctx['profile'] == 'handheld_gripper'
    assert ctx['instruction'] == 'fold the fabric'
    assert ctx['annotation_subtasks'][0]['notes'] == annotation[0]
    assert ctx['uploader_notes']['annotations/motions.json'] == annotation
    assert 'collection_note' not in ctx
    assert ctx['source']['note_files'] == ['instruction.txt', 'annotations/motions.json', 'video_info.json']


def test_invalid_structural_boundaries_stay_unresolved_with_complete_notes(tmp_path, monkeypatch):
    root = tmp_path / 'invalid'
    clip(root)
    data = [{'start_ts': True, 'end_ts': 1, 'atomic_action': [{'verb': 'lift'}]},
            {'start_ts': 1, 'end_ts': 0, 'atomic_action': [{'object': 'cup'}]},
            {'start_ts': 'NaN', 'end_ts': 2, 'atomic_action': 'damaged'},
            {'scene': 'unmeasured', 'atomic_action': []}]
    (root / 'ego_annotation/ego_action_annotation.json').write_text(json.dumps(data))
    _, ctx, _ = convert(root, tmp_path / 'output', monkeypatch)
    assert not ctx.get('annotation_subtasks')
    assert len(ctx['annotation_unresolved']) == 4
    assert ctx['uploader_notes']['ego_annotation/ego_action_annotation.json'] == data
    assert all(row['timing_reason'] for row in ctx['annotation_unresolved'])


def test_structural_note_scope_never_consumes_an_unrelated_nested_task(tmp_path, monkeypatch):
    root = tmp_path / 'scope'
    clip(root)
    note = root / 'other_episode/task.json'
    note.parent.mkdir()
    note.write_text('{"task":"a different task"}')
    _, ctx, _ = convert(root, tmp_path / 'output', monkeypatch)
    assert not ctx.get('instruction')
    assert 'other_episode/task.json' not in ctx['source']['note_files']
    assert note not in formats.opened_notes(formats.plan(root)[1])


def test_unreadable_annotation_stays_attributed_without_losing_the_clip(tmp_path, monkeypatch):
    root = tmp_path / 'broken'
    clip(root)
    (root / 'ego_annotation/ego_action_annotation.json').write_text('[{"start_ts":0')
    _, ctx, _ = convert(root, tmp_path / 'output', monkeypatch)
    assert not ctx.get('annotation_subtasks')
    assert ctx['uploader_notes']['ego_annotation/ego_action_annotation.json'] == '[{"start_ts":0'
    assert any(issue['kind'] == 'metadata_unreadable' for issue in ctx['reader_issues'])


def test_nested_annotations_of_another_recording_do_not_bind_to_the_parent_video(tmp_path, monkeypatch):
    root = tmp_path / 'scope'
    annotation, _ = clip(root)
    other = root / 'other_episode/annotations/task.json'
    other.parent.mkdir(parents=True)
    other.write_text(json.dumps([{'start_ts': 0, 'end_ts': 1,
                                  'atomic_action': [{'verb': 'unrelated'}]}]))
    _, ctx, _ = convert(root, tmp_path / 'output', monkeypatch)
    assert len(ctx['annotation_subtasks']) == 2
    assert 'other_episode/annotations/task.json' not in ctx['source']['note_files']


def test_conflicting_device_claims_keep_both_sources_without_choosing_a_device(tmp_path, monkeypatch):
    root = tmp_path / 'conflict'
    _, info = clip(root)
    (root / 'other_device.json').write_text(json.dumps({'deviceInfo': {'brand': 'other', 'model': 'camera'},
                                                      'cameraParams': {'resolution': [640, 480]}}))
    _, ctx, _ = convert(root, tmp_path / 'output', monkeypatch)
    assert not ctx['source'].get('device') and not ctx['source'].get('resolution')
    assert {row['source'] for row in ctx['source']['device_claims']} == {'video_info.json', 'other_device.json'}
    assert ctx['uploader_notes']['video_info.json'] == info
    assert any(issue['kind'] == 'device_metadata_disagree' for issue in ctx['reader_issues'])


def test_an_atomic_action_file_named_for_an_absent_take_keeps_its_owner(tmp_path, monkeypatch):
    root = tmp_path / 'ownership'
    clip(root)
    (root / 'ep2_annotations.json').write_text(json.dumps([
        {'start_ts': 0, 'end_ts': 1, 'atomic_action': [{'verb': 'unrelated'}]}]))
    _, ctx, _ = convert(root, tmp_path / 'output', monkeypatch)
    assert len(ctx['annotation_subtasks']) == 2
    assert 'ep2_annotations.json' not in ctx['source']['note_files']
