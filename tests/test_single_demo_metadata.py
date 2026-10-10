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
        h.create_dataset('calibration/covariance', data=np.arange(64, dtype=np.float64).reshape(8, 8))
        demo = h.create_group('data/demo_0')
        demo.attrs['fps'] = 25
        demo.attrs['task'] = np.array([b'pick up the test block'])
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
        np.testing.assert_array_equal(saved[aux['key']], sensor)
    with h5py.File(source, 'r') as native:
        retained = native['data/demo_0/aux_sensor'][()]
        assert retained.dtype == sensor.dtype and retained.shape == sensor.shape
        assert retained.tobytes() == sensor.tobytes()
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
    assert ctx['fps'] == 25
    from label import episode, evidence_access
    access = evidence_access.Access(episode.load(ep))
    field = next(f for f in access.inventory()
                 if f['name'] == 'HDF5 recording data/demo_0 attribute fps')
    assert access.inspect({'field_id': field['id'], 'mode': 'metadata'})['value'] == 25
    assert ctx['uploader_notes']['episode attribute operator_note'] == 'camera stayed fixed'
    assert ctx['uploader_notes']['file calibration/extrinsics'] == np.eye(4).tolist()
    assert ctx['uploader_notes']['file calibration/covariance'] == np.arange(64).reshape(8, 8).tolist()
    with h5py.File(source, 'r') as h:
        assert formats.h5_fps(h, ctx['source']['group']) == 25
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original


def test_original_metadata_is_selectable_without_copying_other_episode_data(tmp_path):
    from label import episode, evidence_access
    root = tmp_path / 'upload'
    source, _, _, _, _ = _single_demo(root)
    with h5py.File(source, 'a') as h:
        h.create_dataset('calibration/response_table', data=np.arange(8192, dtype=np.uint64).reshape(128, 64) + 2**63)
        h.create_dataset('calibration/no_readings', dtype='f8')
        h.attrs['no_calibration'] = h5py.Empty('f8')
        h.create_dataset('data/demo_0/sensor_setup/response_table', data=np.arange(8192, dtype=np.float64).reshape(128, 64))
        h['data/demo_0'].attrs['reference_table'] = np.arange(5120, dtype=np.int64).reshape(80, 64)
        h['data/demo_0'].attrs['calibration_record'] = np.array([(2.5, 7)], dtype=[('scale', 'f8'), ('revision', 'i8')])
        h.copy('data/demo_0', 'data/demo_1')
        h['data/demo_1'].attrs['operator_note'] = 'another recording'
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    prepared = tmp_path / 'prepared'
    report = formats.convert(root, 'teleop_arms', prepared, 'test', 900)
    assert not report['failed'] and len(report['episodes']) == 2
    epdir = prepared / report['episodes'][0]['episode_id']
    access = evidence_access.Access(episode.load(epdir))
    fields = access.inventory()
    absent = next(f for f in fields if f['name'] == 'HDF5 file calibration/no_readings')
    assert absent['kind'] == 'unreadable' and not absent.get('available_modes')
    absent_attribute = next(f for f in fields if f['name'] == 'HDF5 file / attribute no_calibration')
    assert absent_attribute['kind'] == 'unreadable'
    response = next(f for f in fields if f['name'] == 'HDF5 file calibration/response_table')
    attribute = next(f for f in fields if f['name'] == 'HDF5 recording data/demo_0 attribute reference_table')
    setup = next(f for f in fields if f['name'] == 'HDF5 recording data/demo_0/sensor_setup/response_table')
    assert access.inspect({'field_id': setup['id'], 'mode': 'metadata', 'pointer': '/127/63'})['value'] == 8191
    assert not any('demo_1' in f['name'] for f in fields)
    with pytest.raises(ValueError, match='pointer'):
        access.inspect({'field_id': response['id'], 'mode': 'metadata'})
    result = access.inspect({'field_id': response['id'], 'mode': 'metadata', 'pointer': '/127/63'})
    assert result['value'] == 2**63 + 8191 and 'times_s' not in result
    assert result['exact_metadata_values'] == [{'pointer': '', 'decimal': str(2**63 + 8191)}]
    assert access.inspect({'field_id': attribute['id'], 'mode': 'metadata', 'pointer': '/70/5'})['value'] == 4485
    calibration = next(f for f in fields if f['name'].endswith('demo_0 attribute calibration_record'))
    assert access.inspect({'field_id': calibration['id'], 'mode': 'metadata', 'pointer': '/0/scale'})['value'] == 2.5
    with pytest.raises(ValueError, match='pointer'):
        access.inspect({'field_id': attribute['id'], 'mode': 'metadata', 'pointer': '/-1/5'})
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original
    assert not (epdir / source.name).exists()
    with h5py.File(source, 'a') as h:
        h.attrs['changed'] = True
    with pytest.raises(ValueError, match='changed'):
        access.inspect({'field_id': response['id'], 'mode': 'metadata', 'pointer': '/127/63'})


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
