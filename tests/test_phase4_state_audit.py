"""Controlled native observations from MASTER item 21, not borrowed original uploads."""
import hashlib
import json

import h5py
import numpy as np
import pandas as pd
import pytest

from prepare import formats
from test_prepare import _lerobot_v21


JOINTS = [f'joint{i}' for i in range(1, 7)]


def native_state(root, names, *, joint_names=None, split=False):
    _lerobot_v21(root, n=8)
    info_path = root / 'meta/info.json'
    info = json.loads(info_path.read_text())
    path = root / 'data/chunk-000/episode_000000.parquet'
    df = pd.read_parquet(path)
    values = np.arange(56, dtype=np.float32).reshape(8, 7) / 100
    df['action'] = list(values + np.float32(.01))
    info['features']['action'].update(shape=[7])
    if split:
        df = df.drop(columns=['observation.state'])
        info['features'].pop('observation.state')
        for column, array, labels in [('observation.state.left_arm', values[:, :6], names[:6]),
                                      ('observation.state.left_gripper', values[:, 6:], names[6:])]:
            df[column] = list(array)
            info['features'][column] = {'dtype': 'float32', 'shape': [array.shape[1]], 'names': labels}
    else:
        df['observation.state'] = list(values)
        info['features']['observation.state'].update(shape=[7], names=names)
        if joint_names is not None:
            info['features']['observation.state']['joint_names'] = joint_names
    df.to_parquet(path)
    info_path.write_text(json.dumps(info, ensure_ascii=False))
    return values


def converted(root, out):
    before = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in root.rglob('*') if p.is_file()}
    report = formats.convert(root, 'teleop_arms', out, 'controlled named state', 900)
    assert not report['failed'] and len(report['episodes']) == 1, report
    path = out / report['episodes'][0]['episode_id']
    assert before == {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in root.rglob('*') if p.is_file()}
    return path, json.loads((path / 'context.json').read_text())


@pytest.mark.parametrize('names', [
    ['末端位置x', '末端位置y', '末端位置z', '末端姿态roll', '末端姿态pitch', '末端姿态yaw', '夹爪'],
    [f'未解释值{i}' for i in range(7)]])
def test_chinese_pose_and_unknown_names_never_become_joints_by_width(tmp_path, names):
    root = tmp_path / 'native'
    original = native_state(root, names)
    path, ctx = converted(root, tmp_path / 'prepared')
    assert ctx['state_kind'] == 'none'
    assert ctx.get('state_note') and ctx['state_why'] == 'layout'
    field = next(s for s in ctx['signals'] if s['name'] == 'observation.state')
    assert field['names'] == names
    with np.load(path / 'signals.npz') as signals, np.load(path / 'state.npz') as archive:
        assert signals[field['key']].tobytes() == archive['state'].tobytes() == original.tobytes()


@pytest.mark.parametrize('split', [False, True])
def test_drive_joint_requires_recorded_scalar_gripper_column(tmp_path, split):
    root = tmp_path / 'native'
    names = JOINTS + ['drive_joint']
    original = native_state(root, names, split=split)
    path, ctx = converted(root, tmp_path / 'prepared')
    assert ctx['state_kind'] == ('joints' if split else 'none')
    if split:
        assert ctx['source']['state_columns'] == ['observation.state.left_arm', 'observation.state.left_gripper']
        assert ctx['state_value_roles'] == [None] * 6 + ['gripper']
        assert ctx['state_value_role_sources'][-1] == 'observation.state.left_gripper'
        assert ctx['state_identity']['names'] == names
        with np.load(path / 'state.npz') as z:
            assert z['state'].tobytes() == original.tobytes()
    else:
        assert next(s for s in ctx['signals'] if s['name'] == 'observation.state')['names'] == names


@pytest.mark.parametrize('declared_joints', [False, True])
def test_sliding_base_axes_do_not_establish_an_end_effector_pose(tmp_path, declared_joints):
    root = tmp_path / 'native'
    names = ['base_' + axis for axis in ['x', 'y', 'z', 'roll', 'pitch', 'yaw']] + ['gripper']
    original = native_state(root, names, joint_names=names[:6] if declared_joints else None)
    path, ctx = converted(root, tmp_path / 'prepared')
    assert ctx['state_kind'] == 'none' and ctx.get('state_note')
    field = next(s for s in ctx['signals'] if s['name'] == 'observation.state')
    assert field['names'] == names
    with np.load(path / 'state.npz') as z:
        assert z['state'].tobytes() == original.tobytes()
    assert formats.state_layout(7, 'teleop_arms', ['ee_' + a for a in ['x', 'y', 'z', 'roll', 'pitch', 'yaw']]
                                + ['gripper'])[0] == 'ee_pose'


@pytest.mark.parametrize('key,names,expected', [
    ('joint_action', JOINTS + ['gripper'], True),
    ('joint_action/vector', JOINTS + ['gripper'], True),
    ('joint_action', None, False),
    ('commands/joint_action', JOINTS + ['gripper'], False),
    ('actions/joint_action', JOINTS + ['gripper'], False)])
def test_joint_action_native_state_requires_named_joint_and_gripper_authority(tmp_path, key, names, expected):
    path = tmp_path / 'controlled.h5'
    times = 1_790_000_000 + np.arange(8) / 30
    original = np.arange(56, dtype=np.float64).reshape(8, 7) / 100
    with h5py.File(path, 'w') as f:
        f['timestamps'] = times
        ds = f.create_dataset(key, data=original)
        if names:
            ds.attrs['names'] = names
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with h5py.File(path, 'r') as f:
        streams = formats.h5_streams(f, '')
        signals = formats.h5_signals(f, streams, times, 30, 8)
    native_name = next(s['name'] for s in streams['signal'] if s['path'] == key)
    assert signals.meta[native_name]['source'] == 'HDF5 dataset ' + key
    identity = {}
    state, action, got_names, source, note = formats.h5_state(signals, 'teleop_arms', times, identity_ctx=identity)
    if expected:
        assert state is not None and state.tobytes() == original.tobytes()
        assert source == key and got_names == names
        assert identity['state_identity']['source'] == key and identity['state_identity']['names'] == names
    else:
        assert state is None and native_name in signals
        if key == 'joint_action':
            assert note and note.why == 'layout'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
