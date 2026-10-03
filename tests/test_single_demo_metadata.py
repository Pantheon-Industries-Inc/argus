"""A single recorded HDF5 demo keeps its streams and local metadata."""
import hashlib
import json

import av
import h5py
import numpy as np
import pytest

from prepare import formats


def _single_demo(root):
    root.mkdir()
    clock = np.int64(10**12) + np.arange(6, dtype=np.int64) * 40_000_000
    state = np.arange(84, dtype=np.float32).reshape(6, 14) / 16
    sensor = np.arange(6, dtype=np.float32).reshape(6, 1) + 100
    frames = np.stack([np.full((32, 48, 3), 10 + 20 * k, np.uint8) for k in range(6)])
    path = root / 'recording.h5'
    with h5py.File(path, 'w') as h:
        h.attrs['serial'] = 'test recorder'
        h.create_dataset('calibration/extrinsics', data=np.eye(4, dtype=np.float64))
        demo = h.create_group('data/demo_0')
        demo.attrs['fps'] = 25
        demo.attrs['task'] = 'pick up the test block'
        demo.attrs['operator_note'] = 'camera stayed fixed'
        demo.create_dataset('timestamps', data=clock).attrs['units'] = 'ns'
        demo.create_dataset('images/cam_high', data=frames)
        qpos = demo.create_dataset('qpos', data=state)
        qpos.attrs['names'] = [*[f'left_joint_{j}' for j in range(6)], 'left_gripper',
                               *[f'right_joint_{j}' for j in range(6)], 'right_gripper']
        demo.create_dataset('aux_sensor', data=sensor)
    return path, clock, state, sensor, frames


def _convert(root, output):
    report = formats.convert(root, 'teleop_arms', output, 'test', 900)
    assert not report['failed'] and len(report['episodes']) == 1, report
    ep = output / report['episodes'][0]['episode_id']
    return ep, json.loads((ep / 'context.json').read_text())


def test_a_single_demo_keeps_camera_clock_state_and_original_bytes(tmp_path):
    root = tmp_path / 'upload'
    source, clock, state, sensor, frames = _single_demo(root)
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    ep, ctx = _convert(root, tmp_path / 'prepared')
    assert ctx['fps'] == 25 and ctx['state_kind'] == 'joints'
    assert ctx['cameras']['exo']['name'] == 'cam_high'
    assert ctx['cameras']['exo']['key'] == 'images/cam_high'
    assert 'clock_note' not in ctx['source']
    meta = ctx['recorded_container_times']
    key = next(k for k, v in meta['clocks'].items() if v['source'] == '/data/demo_0/timestamps')
    with np.load(ep / meta['file']) as saved:
        assert saved[key].dtype == clock.dtype and saved[key].shape == clock.shape
        assert saved[key].tobytes() == clock.tobytes()
    with np.load(ep / 'state.npz') as saved:
        assert saved['state'].dtype == state.dtype and saved['state'].shape == state.shape
        assert saved['state'].tobytes() == state.tobytes()
    aux = next(s for s in ctx['signals'] if s['name'] == 'aux_sensor')
    with np.load(ep / 'signals.npz') as saved:
        assert saved[aux['key']].dtype == sensor.dtype and saved[aux['key']].shape == sensor.shape
        assert saved[aux['key']].tobytes() == sensor.tobytes()
    with av.open(str(ep / 'exo.mp4')) as video:
        decoded = [f.to_ndarray(format='rgb24') for f in video.decode(video=0)]
    assert len(decoded) == len(frames)
    assert all(f.shape == (32, 48, 3) for f in decoded)
    assert all(abs(float(actual.mean()) - float(want.mean())) < 3 for actual, want in zip(decoded, frames))
    assert 'calibration/extrinsics' in ctx['uploader_annotation']
    assert ctx['uploader_notes']['file attribute serial'] == 'test recorder'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original


def test_a_single_recorded_demo_keeps_its_local_rate_task_and_notes(tmp_path):
    root = tmp_path / 'upload'
    source, _, _, _, _ = _single_demo(root)
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    ep, ctx = _convert(root, tmp_path / 'prepared')
    assert ctx.get('instruction') == 'pick up the test block'
    assert ctx['source']['group'] == 'data/demo_0'
    assert ctx['uploader_notes']['episode attribute fps'] == 25
    assert ctx['uploader_notes']['episode attribute operator_note'] == 'camera stayed fixed'
    assert ctx['uploader_notes']['file calibration/extrinsics'] == np.eye(4).ravel().tolist()
    with h5py.File(source, 'r') as h:
        assert formats.h5_fps(h, ctx['source']['group']) == 25
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original


@pytest.mark.parametrize('name', ['sensor', 'camera', 'observations'])
def test_an_ordinary_single_stream_group_remains_a_whole_file(name, tmp_path):
    path = tmp_path / 'ordinary.h5'
    with h5py.File(path, 'w') as h:
        group = h.create_group(name)
        group.attrs['fps'] = 25
        group.attrs['task'] = 'a local note is not an episode declaration'
        group.create_dataset('timestamps', data=np.arange(6, dtype=np.int64) * 40_000_000)
        group.create_dataset('values', data=np.arange(84, dtype=np.float32).reshape(6, 14))
        if name != 'sensor':
            group.create_dataset('images', data=np.zeros((6, 32, 48, 3), np.uint8))
        assert formats.h5_episodes(h) == ['']


@pytest.mark.parametrize('name', ['trial7', 'episode_3'])
def test_a_numbered_recording_group_keeps_its_declared_metadata_scope(name, tmp_path):
    root = tmp_path / 'upload'
    source, _, _, _, _ = _single_demo(root)
    with h5py.File(source, 'a') as h:
        h.move('data/demo_0', 'data/' + name)
        assert formats.h5_episodes(h) == ['data/' + name]


@pytest.mark.parametrize('outside', ['sensor', 'camera', 'clock', 'notes'])
def test_a_single_demo_does_not_hide_an_outside_recorded_stream(outside, tmp_path):
    root = tmp_path / 'upload'
    source, _, _, _, _ = _single_demo(root)
    with h5py.File(source, 'a') as h:
        if outside == 'camera':
            h.create_dataset('outside/images', data=np.zeros((6, 32, 48, 3), np.uint8))
        elif outside == 'clock':
            h.create_dataset('outside/timestamps', data=np.arange(6, dtype=np.int64) * 40_000_000)
        elif outside == 'notes':
            h.create_dataset('outside/notes', data=np.array([b'claim'] * 80))
        else:
            h.create_dataset('outside/values', data=np.arange(6, dtype=np.float32))
        assert formats.h5_episodes(h) == ['']


def test_a_numbered_sensor_group_with_no_camera_is_not_a_demo(tmp_path):
    with h5py.File(tmp_path / 'sensor.h5', 'w') as h:
        group = h.create_group('data/demo_8')
        group.create_dataset('timestamps', data=np.arange(6, dtype=np.int64) * 40_000_000)
        group.create_dataset('values', data=np.arange(84, dtype=np.float32).reshape(6, 14))
        assert formats.h5_episodes(h) == ['']
