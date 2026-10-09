"""Publisher clock, measurement and recording declarations survive generic HDF preparation."""
import h5py
import numpy as np
import pytest

from prepare import formats


def rgb(group, rows=6):
    group['rgb'] = np.zeros((rows, 32, 48, 3), dtype=np.uint8)


def convert(source, tmp_path):
    item = {'file': source, 'group': '', 'name': source.stem}
    ctx = formats.convert_hdf5(item, 'ego_head', tmp_path / 'episodes', 'test')
    return next((tmp_path / 'episodes').iterdir()), ctx


@pytest.mark.parametrize('unit_key', ['unit', 'units'])
def test_declared_seconds_survive_slow_camera_conversion(tmp_path, unit_key):
    source = tmp_path / 'slow.h5'
    raw = np.array([0., 2., 4.], dtype=np.float64)
    with h5py.File(source, 'w') as h:
        rgb(h, 3)
        h.create_dataset('timestamp', data=raw).attrs[unit_key] = 'seconds'
    ep, ctx = convert(source, tmp_path)
    with np.load(ep / 'times.npz') as times:
        np.testing.assert_allclose(times['exo'], [0., 2., 4.], atol=1e-6)
    with np.load(ep / ctx['recorded_container_times']['file']) as retained:
        np.testing.assert_array_equal(retained['timestamp'], raw)
        assert retained['timestamp'].dtype == raw.dtype
    assert ctx['recorded_container_times']['clocks']['timestamp']['units'] == 'seconds'


def test_declared_sensor_unit_cannot_be_changed_to_match_camera_span(tmp_path):
    with h5py.File(tmp_path / 'units.h5', 'w') as h:
        rgb(h)
        h.create_dataset('timestamp', data=[0., .1, .2, .3, .4, .5]).attrs['unit'] = 's'
        pad = h.create_group('pad')
        pad.create_dataset('timestamp', data=np.arange(600)).attrs['units'] = 'seconds'
        pad['pressure'] = np.arange(600, dtype=np.float32)
        streams = formats.h5_streams(h, '')
        np.testing.assert_array_equal(streams['clock']['pad/timestamp'], np.arange(600))
        np.testing.assert_allclose(streams['clock']['timestamp'], [0., .1, .2, .3, .4, .5])


@pytest.mark.parametrize('unit,raw', [('ms', [0, 100, 200]), ('microseconds', [0, 100000, 200000]),
                                   ('ns', [0, 100000000, 200000000])])
def test_declared_clock_units_set_camera_seconds(tmp_path, unit, raw):
    with h5py.File(tmp_path / 'clock.h5', 'w') as h:
        rgb(h, 3)
        h.create_dataset('timestamp', data=np.asarray(raw, dtype=np.int64)).attrs['units'] = unit
        np.testing.assert_allclose(formats.h5_streams(h, '')['clock']['timestamp'], [0., .1, .2])


def test_conflicting_clock_units_keep_source_but_only_assume_frame_placement(tmp_path):
    source = tmp_path / 'conflict.h5'
    raw = np.array([0, 100, 200, 300, 400, 500], dtype=np.int64)
    with h5py.File(source, 'w') as h:
        h.attrs['fps'] = 10
        rgb(h)
        clock = h.create_dataset('timestamp', data=raw)
        clock.attrs.update(unit='seconds', units='milliseconds')
        h['pressure'] = np.array([0., 0., 10., 10., 0., 0.])
    ep, ctx = convert(source, tmp_path)
    assert ctx.get('presentation_times')
    assert any(i['kind'] == 'clock_units_unresolved' for i in ctx['reader_issues'])
    assert 'declarations conflict' in ctx['camera_clock']['exo']['what']
    assert 'nonfinite' not in ctx['camera_clock']['exo']['what']
    pressure = next(s for s in ctx['signals'] if s['name'] == 'pressure')
    assert pressure.get('camera_aligned_by') == 'assumed camera clock'
    with np.load(ep / ctx['recorded_container_times']['file']) as retained:
        np.testing.assert_array_equal(retained['timestamp'], raw)
    with h5py.File(source) as h:
        assert h['timestamp'].attrs['unit'] == 'seconds'
        assert h['timestamp'].attrs['units'] == 'milliseconds'


@pytest.mark.parametrize('name,shape,dtype', [('tactile_pressure', (128, 128), 'uint16'),
                                           ('taxels', (32, 48, 3), 'float32')])
def test_declared_tactile_maps_are_signals_with_original_units(tmp_path, name, shape, dtype):
    source = tmp_path / 'skin.h5'
    with h5py.File(source, 'w') as h:
        rgb(h)
        h['timestamp'] = np.arange(6) / 10
        ds = h.create_dataset(name, data=np.ones((6, *shape), dtype=dtype))
        ds.attrs.update(units='kPa', sensor_type='tactile pressure')
    ep, ctx = convert(source, tmp_path)
    assert not ctx.get('depth')
    measurement = next(s for s in ctx['signals'] if s['name'] == name)
    assert measurement['shape'] == list(shape) and measurement['units'] == 'kPa'
    with np.load(ep / 'signals.npz') as signals:
        np.testing.assert_array_equal(signals[measurement['key']], np.ones((6, np.prod(shape))))


def test_unnamed_numeric_map_does_not_acquire_depth_semantics(tmp_path):
    with h5py.File(tmp_path / 'array.h5', 'w') as h:
        field = h.create_dataset('readings', data=np.ones((6, 128, 128), dtype=np.float32))
        assert formats.h5_kind('readings', field) == 'signal'
        depth = h.create_dataset('depth', data=np.ones((6, 128, 128), dtype=np.float32))
        assert formats.h5_kind('depth', depth) == 'depth'


@pytest.mark.parametrize('name,sensor_type', [('tactile_image', 'optical tactile sensor'),
                                             ('rgb', 'hardware model 7'),
                                             ('rgb', ['vendor_model_7', 'revision_B']),
                                             ('rgb', ['pressure', 'camera'])])
def test_optical_touch_images_and_unknown_hardware_types_remain_cameras(tmp_path, name, sensor_type):
    with h5py.File(tmp_path / 'optical.h5', 'w') as h:
        image = h.create_dataset(name, data=np.zeros((6, 32, 48, 3), dtype=np.uint8))
        image.attrs['sensor_type'] = sensor_type
        assert formats.h5_kind(name, image) == 'camera'
        np.testing.assert_array_equal(image.attrs['sensor_type'], sensor_type)


@pytest.mark.parametrize('cameras_in_groups', [False, True])
def test_sensor_siblings_remain_one_recording(tmp_path, cameras_in_groups):
    with h5py.File(tmp_path / 'actors.h5', 'w') as h:
        if not cameras_in_groups:
            rgb(h)
        for side in ('left', 'right'):
            group = h.create_group(side)
            group['qpos'] = np.zeros((6, 7), dtype=np.float32)
            group['timestamp'] = np.arange(6) / 10
            if cameras_in_groups:
                rgb(group)
        assert formats.h5_episodes(h) == ['']


@pytest.mark.parametrize('outside', ['camera', 'sensor', None])
def test_numbered_demo_siblings_split_only_when_all_recorded_streams_are_owned(tmp_path, outside):
    with h5py.File(tmp_path / 'demos.h5', 'w') as h:
        h['calibration'] = np.eye(4)
        for name in ('demo_0', 'demo_1'):
            group = h.create_group('data/' + name)
            rgb(group)
            group['timestamp'] = np.arange(6) / 10
        if outside == 'camera':
            rgb(h)
        elif outside == 'sensor':
            h['force'] = np.arange(6, dtype=np.float32)
            h['timestamp'] = np.arange(6) / 10
        assert formats.h5_episodes(h) == (['data/demo_0', 'data/demo_1'] if outside is None else [''])
