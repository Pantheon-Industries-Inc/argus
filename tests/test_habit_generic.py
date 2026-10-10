"""Tiny native LeRobot records retain scoped tasks and publisher claims without judgment leakage."""
import hashlib
import io
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from label import episode
from prepare import formats, habit
from test_prepare import _lerobot_v21


def native(root, *, images=False, indexed=False):
    n = 8
    state = _lerobot_v21(root, n=n, task='coarse recorded task', extra={
        'task_index': [0] * n, 'low_level_task_index': [0, 0, 1, 1, 0, 0, 1, 1],
        'human_role_subtask_index': [0, 0, 0, 0, 1, 1, 1, 1],
        'is_error_segment': [False, True, True, False, False, False, False, False],
        'is_intervention_segment': [False, False, False, True, True, False, False, False],
        'is_high_jerk_segment': [False, False, False, False, False, True, True, False]})
    meta = root / 'meta'
    info = json.loads((meta / 'info.json').read_text())
    names = [side + '_' + value for side in ['left', 'right']
             for value in ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']]
    info['robot_type'] = 'recorded human interface'
    info['features']['observation.state']['names'] = [f'position_{i}' for i in range(14)] if indexed else names
    for column in ['task_index', 'low_level_task_index', 'human_role_subtask_index']:
        info['features'][column] = {'dtype': 'int64', 'shape': [1]}
    for column in habit.PUBLISHER_COLUMNS:
        info['features'][column] = {'dtype': 'bool', 'shape': [1]}
    row = {'episode_index': 0, 'length': n, 'tasks': ['coarse recorded task'],
           'high_level_instruction': 'fold the recorded cloth', 'task_status': 'publisher verdict sentinel',
           'sid': 'recorded session', 'unit_name': 'recorded unit'}
    (meta / 'episodes.jsonl').write_text(json.dumps(row) + '\n')
    for filename, texts in [('tasks.jsonl', ['coarse recorded task']),
                            ('subtasks.jsonl', ['robot reach', 'robot fold']),
                            ('human_subtasks.jsonl', ['person hold', 'person release'])]:
        (meta / filename).write_text(''.join(json.dumps({'task_index': i, 'task': text}) + '\n'
                                           for i, text in enumerate(texts)))
    if images:
        path = root / 'data/chunk-000/episode_000000.parquet'
        df = pd.read_parquet(path)
        for key, feature in info['features'].items():
            if feature.get('dtype') != 'video':
                continue
            frames = []
            for i in range(n):
                buffer = io.BytesIO()
                Image.fromarray(np.full((96, 128, 3), 20 + i * 10, dtype=np.uint8)).save(buffer, format='PNG')
                frames.append({'bytes': buffer.getvalue(), 'path': None})
            df[key] = frames
            feature.update(dtype='image', shape=[96, 128, 3])
        df.to_parquet(path)
        import shutil
        shutil.rmtree(root / 'videos')
    (meta / 'info.json').write_text(json.dumps(info))
    return state, row


def converted(root, out, monkeypatch, *, legacy):
    original = formats.upload_adapters
    with monkeypatch.context() as patch:
        patch.setattr(formats, 'upload_adapters', lambda kind: [habit] if kind == 'lerobot' and legacy
                      else [] if kind == 'lerobot' else original(kind))
        report = formats.convert(root, 'teleop_arms', out, 'native parity', 900)
    assert not report['failed'] and len(report['episodes']) == 1, report
    path = out / report['episodes'][0]['episode_id']
    return path, json.loads((path / 'context.json').read_text())


def source_hashes(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def prompt(path):
    request = episode.build_request(path)
    return request, '\n'.join(p.get('text', '') for p in request['content'])


@pytest.mark.parametrize('images', [False, True])
def test_native_pose_arrays_camera_images_and_high_level_task_survive_generic(tmp_path, monkeypatch, images):
    root = tmp_path / 'recording'
    expected, _ = native(root, images=images)
    before = source_hashes(root)
    old_ep, old = converted(root, tmp_path / 'old', monkeypatch, legacy=True)
    new_ep, new = converted(root, tmp_path / 'generic', monkeypatch, legacy=False)
    assert new['instruction'] == old['instruction'] == 'fold the recorded cloth'
    assert new['state_kind'] == old['state_kind'] == 'ee_pose'
    with np.load(old_ep / 'state.npz') as a, np.load(new_ep / 'state.npz') as b:
        for key, want in [('state', expected), ('action', expected + np.float32(.01))]:
            assert a[key].dtype == b[key].dtype == want.dtype
            assert a[key].tobytes() == b[key].tobytes() == want.tobytes()
    assert new['fps'] == old['fps'] == 30
    assert new['cameras'] == old['cameras']
    old_request, _ = prompt(old_ep)
    new_request, _ = prompt(new_ep)
    images_of = lambda req: [p for p in req['content'] if p['type'] == 'image_url']
    assert images_of(old_request) == images_of(new_request) and images_of(new_request)
    assert old_request['plan']['ks'] == new_request['plan']['ks']
    clocks = [[episode.frame_time(episode.load(ep), i) for i in range(8)] for ep in [old_ep, new_ep]]
    assert clocks[0] == clocks[1]
    if images:
        assert np.rint(np.array(clocks[1]) * 1_000_000).astype(np.int64).tolist() == [
            0, 33333, 66667, 100000, 133333, 166667, 200000, 233333]
    else:
        assert clocks[1] == [i / 30 for i in range(8)]
    for name in ['times.npz', 'sources.json']:
        if name == 'times.npz' and not (old_ep / name).exists():
            assert not (new_ep / name).exists()
        elif name == 'times.npz':
            with np.load(old_ep / name) as a, np.load(new_ep / name) as b:
                assert a.files == b.files
                for key in a.files:
                    assert a[key].dtype == b[key].dtype and a[key].tobytes() == b[key].tobytes()
        else:
            a, b = [json.loads((ep / name).read_text()) for ep in [old_ep, new_ep]]
            assert {view: {k: v for k, v in desc.items() if k != 'packed'} for view, desc in a.items()} == {
                view: {k: v for k, v in desc.items() if k != 'packed'} for view, desc in b.items()}
    assert source_hashes(root) == before


def test_repeated_task_codes_bind_robot_and_human_to_their_original_tables(tmp_path, monkeypatch):
    root = tmp_path / 'recording'
    native(root)
    _, old = converted(root, tmp_path / 'old', monkeypatch, legacy=True)
    _, new = converted(root, tmp_path / 'generic', monkeypatch, legacy=False)
    assert 'robot reach' in old['instruction_note'] and 'person hold' in old['instruction_note']
    steps = new['annotation_subtasks']
    for column, labels, filename in [
        ('low_level_task_index', ['robot reach', 'robot fold', 'robot reach', 'robot fold'], 'subtasks.jsonl'),
        ('human_role_subtask_index', ['person hold', 'person release'], 'human_subtasks.jsonl')]:
        selected = [s for s in steps if s['column'] == column]
        assert [s['label'] for s in selected] == labels
        assert all(s['source'] == str(root / 'meta' / filename) for s in selected)
        assert all(s['claims'][0]['value'] == s['value'] for s in selected)
    robot = [s for s in steps if s['column'] == 'low_level_task_index']
    assert [(s['row_start'], s['row_end']) for s in robot] == [(0, 1), (2, 3), (4, 5), (6, 7)]
    assert robot[0]['t0'] == 0 and robot[0]['t1'] == 2 / 30
    assert 't1' not in robot[-1] and robot[-1]['timing_reason']


@pytest.mark.parametrize('images', [False, True])
def test_publisher_flags_and_status_are_retained_and_held_out_of_judgment(tmp_path, monkeypatch, images):
    root = tmp_path / 'recording'
    _, row = native(root, images=images)
    before = source_hashes(root)
    _, old = converted(root, tmp_path / 'old', monkeypatch, legacy=True)
    new_ep, new = converted(root, tmp_path / 'generic', monkeypatch, legacy=False)
    assert old['publisher_labels']['task_status'] == row['task_status']
    assert new['publisher_labels']['task_status'] == row['task_status']
    assert all(c not in [s['name'] for s in new.get('signals', [])] for c in habit.PUBLISHER_COLUMNS)
    _, text = prompt(new_ep)
    assert row['task_status'] not in text
    assert all(c not in text for c in habit.PUBLISHER_COLUMNS)
    retained = new['recorded_metadata'][str(root / 'meta/episodes.jsonl')][0]['fields']
    assert retained == row
    for column in habit.PUBLISHER_COLUMNS:
        claim = new['publisher_labels']['columns'][column]
        assert claim['source'] == str(root / 'data/chunk-000/episode_000000.parquet')
        assert claim['dtype'] == 'bool'
        assert claim['values'] == pd.read_parquet(root / 'data/chunk-000/episode_000000.parquet')[column].tolist()
    assert source_hashes(root) == before


def test_recorded_pose_has_no_undeclared_physical_robot_or_opening_range(tmp_path, monkeypatch):
    root = tmp_path / 'recording'
    native(root)
    _, old = converted(root, tmp_path / 'old', monkeypatch, legacy=True)
    _, new = converted(root, tmp_path / 'generic', monkeypatch, legacy=False)
    assert 'Franka FR3' in old['robot_type'] and old['gripper_range'] == [0., 1.]
    assert new['robot_type'] == 'recorded human interface'
    assert new['state_kind'] == 'ee_pose'
    assert 'Franka' not in json.dumps(new) and 'Quest' not in json.dumps(new)
    assert new.get('gripper_range') != [0., 1.]


def test_bare_indexed_positions_never_invent_joint_or_cartesian_axes(tmp_path, monkeypatch):
    root = tmp_path / 'recording'
    expected, _ = native(root, indexed=True)
    original_sources = source_hashes(root)
    old_ep, old = converted(root, tmp_path / 'old', monkeypatch, legacy=True)
    new_ep, new = converted(root, tmp_path / 'generic', monkeypatch, legacy=False)
    assert old['state_kind'] == 'ee_pose'
    assert new['state_kind'] == 'none'
    assert 'indexed position' in new['state_note'].lower()
    descriptor = next(s for s in new['signals'] if s['name'] == 'observation.state')
    with np.load(old_ep / 'state.npz') as a, np.load(new_ep / 'signals.npz') as b, np.load(new_ep / 'state.npz') as raw:
        assert a['state'].shape == raw['state'].shape == expected.shape
        assert a['state'].dtype == raw['state'].dtype == expected.dtype
        assert a['state'].tobytes() == raw['state'].tobytes() == expected.tobytes()
        np.testing.assert_array_equal(b[descriptor['key']], expected)
    assert descriptor['names'] == [f'position_{i}' for i in range(14)]
    before, _ = prompt(old_ep)
    after, text = prompt(new_ep)
    assert episode._has_state(episode.load(old_ep), before['plan'])
    assert not episode._has_state(episode.load(new_ep), after['plan'])
    assert 'timebase' not in after['plan']['checks']
    assert 'RECORDED MOTION' not in text
    assert source_hashes(root) == original_sources


def test_publisher_only_notes_cannot_leave_previously_rendered_judgment_text(tmp_path):
    from prepare.lerobot_labels import retain_metadata
    root = {'dir': str(tmp_path), 'episode_metadata': {}, 'metadata_read': set()}
    ctx = {'episode_index': 0, 'uploader_notes': {'task_status': 'publisher verdict sentinel'},
           'uploader_annotation': 'publisher verdict sentinel'}
    retain_metadata(ctx, root, None, None)
    assert not ctx.get('uploader_annotation')
    assert ctx['recorded_uploader_notes'] == {'task_status': 'publisher verdict sentinel'}


def test_explicit_subtask_column_is_not_replaced_by_a_same_table_task_code(tmp_path):
    from prepare.lerobot_labels import scoped_index_tables
    path = tmp_path / 'subtasks.jsonl'
    path.write_text('{"task_index":0,"low_level_task_index":2,"task":"robot fold"}\n')
    tables = scoped_index_tables(tmp_path, ['low_level_task_index', 'task_index'], formats.read_jsonl)
    labels = tables[('low_level_task_index', str(path))]['labels']
    assert set(labels) == {2}
    assert labels[2]['label'] == 'robot fold'
    assert labels[2]['claims'][0]['fields']['task_index'] == 0


def test_default_upload_reaches_generic_without_changing_published_wrapper(tmp_path):
    root = tmp_path / 'recording'
    native(root)
    report = formats.convert(root, 'teleop_arms', tmp_path / 'output', 'native parity', 900)
    assert not report['failed'] and len(report['episodes']) == 1
    ep = tmp_path / 'output' / report['episodes'][0]['episode_id']
    ctx = json.loads((ep / 'context.json').read_text())
    assert 'adapter' not in ctx['source']
    assert ctx['instruction'] == 'fold the recorded cloth'
    assert ctx['robot_type'] == 'recorded human interface'
