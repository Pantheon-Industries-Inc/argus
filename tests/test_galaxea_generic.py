"""Native split state fields retain their sources through generic LeRobot conversion."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from label import episode
from prepare import formats, galaxea
from test_annotations import _galaxea_case


def native(root, *, reverse=False, width=6):
    _galaxea_case(root, 2)
    path = root / 'data/chunk-000/episode_000000.parquet'
    df = pd.read_parquet(path)
    expected = {}
    for prefix, offset in [('observation.state', 0), ('action', 1000)]:
        groups = []
        for side, base in [('left', 10), ('right', 100)]:
            arms = np.arange(4 * width, dtype=np.float64).reshape(4, width) / 100 + base + offset
            gripper = np.array([0, 25, 50, 100], dtype=np.float64) + (1 if side == 'right' else 0) + offset
            df[f'{prefix}.{side}_arm'] = list(arms)
            df[f'{prefix}.{side}_gripper'] = gripper
            groups.extend([arms, gripper[:, None]])
        expected[prefix] = np.concatenate(groups, axis=1).astype(np.float32)
    df['observation.state.chassis.imu'] = [np.array([k, k + 1, k + 2], dtype=np.float64) for k in range(4)]
    if reverse:
        df = df[list(reversed(df.columns))]
    df.to_parquet(path)
    info = json.loads((root / 'meta/info.json').read_text())
    for prefix in ['observation.state', 'action']:
        for side in ['left', 'right']:
            info['features'][f'{prefix}.{side}_arm']['shape'] = [width]
    info['features']['observation.state.chassis.imu'] = {'dtype': 'float64', 'shape': [3], 'names': ['ax', 'ay', 'az']}
    (root / 'meta/info.json').write_text(json.dumps(info))
    (root / 'meta/tasks.jsonl').write_text('\n'.join(json.dumps(row) for row in [
        {'task_index': 0, 'task': 'pick'}, {'task_index': 1, 'task': 'place'},
        {'task_index': 2, 'task': 'fold the cloth'}, {'task_index': 7, 'task': 'qualified'}]) + '\n')
    return path, expected


def convert(root, out, monkeypatch, *, legacy=False):
    discover = formats.upload_adapters
    with monkeypatch.context() as patch:
        patch.setattr(formats, 'upload_adapters', lambda kind: [galaxea] if legacy and kind == 'lerobot'
                      else [] if kind == 'lerobot' else discover(kind))
        report = formats.convert(root, 'teleop_arms', out, 'native parity', 900)
    assert not report['failed'] and len(report['episodes']) == 1, report
    path = out / report['episodes'][0]['episode_id']
    return path, json.loads((path / 'context.json').read_text())


def hashes(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('reverse', [False, True])
def test_native_wrapper_matches_generic_split_arrays_goals_annotations_and_images(tmp_path, monkeypatch, reverse):
    root = tmp_path / 'recording'
    _, expected = native(root, reverse=reverse)
    originals = hashes(root)
    old_ep, old = convert(root, tmp_path / 'legacy', monkeypatch, legacy=True)
    new_ep, new = convert(root, tmp_path / 'generic', monkeypatch)
    assert new['state_kind'] == old['state_kind'] == 'joints'
    assert new['instruction'] == old['instruction'] == 'fold the cloth'
    assert new['fps'] == old['fps'] == 15
    assert new['n_state_frames'] == old['n_state_frames'] == 4
    assert [s['side'] for s in new['state_identities']] == ['left', 'right']
    arrays = {}
    with np.load(old_ep / 'state.npz') as a, np.load(new_ep / 'state.npz') as b:
        for key, prefix in [('state', 'observation.state'), ('action', 'action')]:
            assert a[key].dtype == b[key].dtype == expected[prefix].dtype
            assert a[key].tobytes() == b[key].tobytes() == expected[prefix].tobytes()
            arrays[key] = {'shape': list(b[key].shape), 'dtype': str(b[key].dtype),
                           'sha256': hashlib.sha256(b[key].tobytes()).hexdigest()}
    assert new['signals'] == old['signals']
    with np.load(old_ep / 'signals.npz') as a, np.load(new_ep / 'signals.npz') as b:
        assert a.files == b.files
        for key in a.files:
            assert a[key].dtype == b[key].dtype and a[key].tobytes() == b[key].tobytes()
    assert new['signals'][0]['names'] == ['ax', 'ay', 'az']
    assert new['annotation_subtasks'] == old['annotation_subtasks']
    assert new['annotation_unresolved'] == old['annotation_unresolved']
    assert any(row['column'] == 'quality_index' and row['value'] == 7 for row in new['annotation_unresolved'])
    assert all(new['recorded_metadata'][key] == value for key, value in old['recorded_metadata'].items())
    assert new['recorded_metadata'][str(root / 'meta/episodes.jsonl')][0]['fields'] == {
        'episode_index': 0, 'length': 4, 'tasks': []}
    requests = [episode.build_request(path) for path in [old_ep, new_ep]]
    images = [[part for part in req['content'] if part['type'] == 'image_url'] for req in requests]
    assert images[0] == images[1] and images[0]
    assert requests[0]['plan']['ks'] == requests[1]['plan']['ks']
    clocks = [[episode.frame_time(episode.load(path), index) for index in range(4)]
              for path in [old_ep, new_ep]]
    assert clocks[0] == clocks[1] == [0, 1 / 15, 2 / 15, 3 / 15]
    assert set(new['cameras']) == set(old['cameras']) == {'exo', 'left', 'right'}
    sources = [json.loads((path / 'sources.json').read_text()) for path in [old_ep, new_ep]]
    for view in sources[0]:
        assert {k: sources[0][view][k] for k in ['packed', 'base_s', 'n_frames', 'camera_key']} == {
            k: sources[1][view][k] for k in ['packed', 'base_s', 'n_frames', 'camera_key']}
    assert hashes(root) == originals
    print(json.dumps({'source_sha256': originals, 'arrays': arrays, 'fps': new['fps'], 'frame_times': clocks[1],
                      'gripper_range_legacy': old.get('gripper_range'), 'gripper_range_generic': new.get('gripper_range'),
                      'gripper_range_qualification': new.get('gripper_range_note'),
                      'robot_description_legacy': old.get('robot_type'), 'robot_metadata_generic': new.get('robot_type'),
                      'quality_tags_legacy': old.get('quality_tag'),
                      'selected_indices': requests[1]['plan']['ks'], 'request_images': len(images[1]),
                      'image_payload_sha256': [hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
                                               for p in images[1]],
                      'differences': ['native robot metadata retained without adapter physical description',
                                      'measured gripper range instead of undeclared full scale',
                                      'quality codes retained without inferred publisher verdict',
                                      'generic camera descriptions', 'legacy approximations removed',
                                      'owned episode row metadata retained']}))


def test_partial_split_group_never_drops_the_remaining_native_arrays(tmp_path, monkeypatch):
    root = tmp_path / 'partial'
    path, _ = native(root)
    df = pd.read_parquet(path).drop(columns=['observation.state.right_gripper'])
    df.to_parquet(path)
    _, ctx = convert(root, tmp_path / 'output', monkeypatch)
    assert ctx['state_kind'] == 'none'
    assert 'incomplete' in ctx['state_note'].lower()
    assert {'observation.state.left_arm', 'observation.state.left_gripper',
            'observation.state.right_arm'} <= {s['name'] for s in ctx['signals']}


def test_unknown_split_actor_never_becomes_a_named_arm(tmp_path, monkeypatch):
    root = tmp_path / 'unknown'
    path, _ = native(root)
    df = pd.read_parquet(path).rename(columns={'observation.state.right_arm': 'observation.state.actor2_arm',
                                             'observation.state.right_gripper': 'observation.state.actor2_gripper'})
    df.to_parquet(path)
    _, ctx = convert(root, tmp_path / 'output', monkeypatch)
    assert ctx['state_kind'] == 'none'
    assert 'actor' in ctx['state_note'].lower()
    assert 'observation.state.actor2_arm' in {s['name'] for s in ctx['signals']}


def test_seven_joint_groups_stay_video_labelled_with_every_split_signal(tmp_path, monkeypatch):
    root = tmp_path / 'unsupported'
    native(root, width=7)
    ep, ctx = convert(root, tmp_path / 'output', monkeypatch)
    assert ctx['state_kind'] == 'none'
    assert not (ep / 'state.npz').exists()
    assert {'observation.state.left_arm', 'observation.state.right_arm', 'action.left_arm',
            'action.right_arm'} <= {s['name'] for s in ctx['signals']}
    episode.build_request(ep)


@pytest.mark.parametrize('coarse', [True, 0.5])
def test_invalid_coarse_code_never_borrows_a_constant_fine_task(tmp_path, monkeypatch, coarse):
    root = tmp_path / 'invalid'
    path, _ = native(root)
    df = pd.read_parquet(path)
    df['coarse_task_index'] = coarse
    df['task_index'] = 0
    df.to_parquet(path)
    _, ctx = convert(root, tmp_path / 'output', monkeypatch)
    assert not ctx.get('instruction')


def test_explicit_episode_goal_keeps_priority_over_coarse_and_quality_codes(tmp_path, monkeypatch):
    root = tmp_path / 'declared'
    native(root)
    (root / 'meta/episodes.jsonl').write_text('{"episode_index":0,"length":4,"tasks":["recorded episode goal"]}\n')
    _, ctx = convert(root, tmp_path / 'output', monkeypatch)
    assert ctx['instruction'] == 'recorded episode goal'
    assert any(row['column'] == 'quality_index' for row in ctx['annotation_unresolved'])


def test_declared_rate_disagreement_retains_native_stamps_and_qualifies_mapping(tmp_path, monkeypatch):
    root = tmp_path / 'rate'
    native(root)
    info = json.loads((root / 'meta/info.json').read_text())
    info['fps'] = 30
    (root / 'meta/info.json').write_text(json.dumps(info))
    ep, ctx = convert(root, tmp_path / 'output', monkeypatch)
    assert ctx['fps'] == 30
    assert any(issue['kind'] == 'recorded_rate_mismatch' for issue in ctx['reader_issues'])
    with np.load(ep / 'times.npz') as clocks:
        assert clocks['exo_pts'].tolist() == [0, 1024, 2048, 3072]


def test_competing_coarse_table_claims_never_supply_an_episode_goal(tmp_path, monkeypatch):
    root = tmp_path / 'conflict'
    native(root)
    with (root / 'meta/tasks.jsonl').open('a') as stream:
        stream.write('{"task_index":2,"task":"a competing coarse task"}\n')
    _, ctx = convert(root, tmp_path / 'output', monkeypatch)
    assert not ctx.get('instruction')
    assert any(issue['kind'] == 'metadata_unreadable' for issue in ctx['reader_issues'])
    assert ctx['recorded_metadata'][str(root / 'meta/info.json')]['fps'] == 15


def test_fourth_stereo_camera_is_preserved_and_its_selection_is_explicit(tmp_path, monkeypatch):
    import shutil
    root = tmp_path / 'stereo'
    native(root)
    info = json.loads((root / 'meta/info.json').read_text())
    key = 'observation.images.head_right_rgb'
    info['features'][key] = {'dtype': 'video', 'shape': [36, 64, 3]}
    (root / 'meta/info.json').write_text(json.dumps(info))
    path = root / 'videos/chunk-000' / key / 'episode_000000.mp4'
    path.parent.mkdir()
    shutil.copyfile(root / 'videos/chunk-000/observation.images.head_rgb/episode_000000.mp4', path)
    ep, ctx = convert(root, tmp_path / 'output', monkeypatch)
    sources = json.loads((ep / 'sources.json').read_text())
    shown = [row['camera_key'] for row in sources.values()]
    unshown = [row['name'] for row in ctx.get('unshown_cameras', [])]
    assert key in shown + unshown
    assert len(shown + unshown) == 4


def test_generic_upload_discovery_does_not_import_a_retired_adapter(monkeypatch):
    import importlib
    load = importlib.import_module
    def declared_only(name, *args, **kwargs):
        if name == 'prepare.galaxea':
            pytest.fail('retired adapter imported on the generic path')
        return load(name, *args, **kwargs)
    monkeypatch.setattr(importlib, 'import_module', declared_only)
    assert 'prepare.galaxea' not in [module.__name__ for module in formats.upload_adapters('lerobot')]


@pytest.mark.parametrize('mode', ['image', 'packed'])
def test_structural_split_columns_work_on_other_lerobot_camera_storage(tmp_path, monkeypatch, mode):
    from test_annotations import _native_case
    root = tmp_path / 'structural'
    expected = _native_case(root, mode)
    path = next(root.glob('data/chunk-*/*.parquet'))
    df = pd.read_parquet(path)
    for side, start in [('left', 0), ('right', 7)]:
        df[f'observation.state.{side}_arm'] = list(expected[:, start:start + 6])
        df[f'observation.state.{side}_gripper'] = expected[:, start + 6]
    df.drop(columns=['observation.state']).to_parquet(path)
    ep, ctx = convert(root, tmp_path / 'output', monkeypatch)
    assert ctx['state_kind'] == 'joints'
    with np.load(ep / 'state.npz') as arrays:
        assert arrays['state'].dtype == expected.dtype and arrays['state'].tobytes() == expected.tobytes()
    assert not any(row['name'].endswith(('_arm', '_gripper')) for row in ctx['signals'])


@pytest.mark.parametrize('mode', ['video', 'image', 'packed'])
@pytest.mark.parametrize('quantity, want', [('pose', 'ee_pose'), ('effort', 'none')])
def test_split_value_names_set_the_recorded_layout_on_every_storage(tmp_path, monkeypatch, mode, quantity, want):
    from test_annotations import _native_case
    root = tmp_path / 'names'
    if mode == 'video':
        native(root)
    else:
        expected = _native_case(root, mode)
        path = next(root.glob('data/chunk-*/*.parquet'))
        df = pd.read_parquet(path)
        for side, start in [('left', 0), ('right', 7)]:
            df[f'observation.state.{side}_arm'] = list(expected[:, start:start + 6])
            df[f'observation.state.{side}_gripper'] = expected[:, start + 6]
        df.drop(columns=['observation.state']).to_parquet(path)
    path = next(root.glob('data/chunk-*/*.parquet'))
    original = pd.read_parquet(path)
    info = json.loads((root / 'meta/info.json').read_text())
    for side in ['left', 'right']:
        names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw'] if quantity == 'pose' else [f'joint{i}_effort' for i in range(6)]
        info['features'][f'observation.state.{side}_arm'] = {'dtype': 'float32', 'shape': [6], 'names': names}
        info['features'][f'observation.state.{side}_gripper'] = {'dtype': 'float32', 'shape': [1]}
    (root / 'meta/info.json').write_text(json.dumps(info))
    ep, ctx = convert(root, tmp_path / 'out', monkeypatch)
    assert ctx['state_kind'] == want
    assert ctx['recorded_metadata'][str(root / 'meta/info.json')] == info
    if want == 'none':
        assert not (ep / 'state.npz').exists()
        with np.load(ep / 'signals.npz') as arrays:
            for signal in ctx['signals']:
                if signal['name'].startswith('observation.state.'):
                    values = np.asarray(list(original[signal['name']]))
                    assert arrays[signal['key']].tobytes() == values.astype(np.float32).reshape(len(values), -1).tobytes()
    else:
        assert ctx['state_identities'][0]['names'][-1] == 'observation.state.left_gripper'


@pytest.mark.parametrize('names', [['x', 'y'], ['x', 'y', 'z', 'joint3', 'joint4', 'joint5']])
def test_partial_or_contradictory_split_names_never_promote_width_to_joints(tmp_path, monkeypatch, names):
    root = tmp_path / 'uncertain'
    native(root)
    info = json.loads((root / 'meta/info.json').read_text())
    for side in ['left', 'right']:
        info['features'][f'observation.state.{side}_arm']['names'] = names
    (root / 'meta/info.json').write_text(json.dumps(info))
    ep, ctx = convert(root, tmp_path / 'out', monkeypatch)
    assert ctx['state_kind'] == 'none'
    assert not (ep / 'state.npz').exists()
    assert {'observation.state.left_arm', 'observation.state.right_arm'} <= {s['name'] for s in ctx['signals']}
