"""Real reader outputs keep source inventory without paying for represented RGB metadata."""
import json

import h5py
import numpy as np
import pytest

from label import dictionary, episode, evidence_access
from prepare import formats
from test_reader_cameras import _lerobot_images


def _prepared(tmp_path, kind, extra, *, json_rate=False, json_attr_kind='scalar',
              numeric_unknown=False, sidecar=False):
    root = tmp_path / 'upload'
    if kind == 'lerobot':
        _lerobot_images(root, n=12)
        info_path = root / 'meta/info.json'
        info = json.loads(info_path.read_text())
        if extra:
            info['features']['observation.images.cam_high']['calibration'] = {'gain': 1.25}
        info_path.write_text(json.dumps(info))
    else:
        root.mkdir()
        with h5py.File(root / 'recording.h5', 'w') as source:
            if json_rate is False:
                source.attrs['fps'] = 10
            else:
                config = {'control_freq': 10}
                if json_rate == 'sibling':
                    config['sensor_config'] = {'gain': 1.25}
                value = json.dumps(config)
                if json_attr_kind == 'vlen':
                    source.attrs['env_args'] = np.array([value], dtype=h5py.string_dtype('utf-8'))
                elif json_attr_kind == 'fixed':
                    source.attrs['env_args'] = np.array([value.encode()], dtype=f'S{len(value)}')
                else:
                    source.attrs['env_args'] = value
            ticks = np.int64(1790000000000000000) + np.arange(12, dtype=np.int64) * 100000000
            source.create_dataset('timestamps', data=ticks).attrs['units'] = 'ns'
            camera = source.create_dataset('images/top', data=np.full((12, 36, 64, 3), 60, np.uint8))
            if extra:
                camera.attrs['calibration'] = 'fixture calibration'
            source.create_dataset('observations/qpos', data=np.zeros((12, 14)))
            if numeric_unknown:
                source.create_dataset('calibration/response_table', data=np.arange(4).reshape(2, 2))
        if sidecar:
            with h5py.File(root / 'sidecar.h5', 'w') as source:
                source.create_dataset('timestamps', data=ticks).attrs['units'] = 'ns'
    report = formats.convert(root, 'teleop_arms', tmp_path / 'prepared', 'fixture', 900)
    assert not report['failed'], report
    return tmp_path / 'prepared' / report['episodes'][0]['episode_id']


@pytest.mark.parametrize('kind', ['hdf5', 'lerobot'])
@pytest.mark.parametrize('extra', [False, True])
def test_reader_coverage_controls_discovery_and_paid_dictionary(tmp_path, kind, extra):
    path = _prepared(tmp_path, kind, extra)
    loaded = episode.load(path)
    assert loaded['context']['state_kind'] == 'joints'
    request = episode.build_request(path, inspect_evidence=True)
    assert bool(request['evidence_access']) == extra
    assert bool(evidence_access.needs_inspection(loaded, request['sensor_evidence'])) == extra
    assert not loaded['signals']
    if kind == 'lerobot' and not extra:
        assert 'info.json' not in request['prompt']
    calls = []
    def fake(*args, **kwargs):
        calls.append(1)
        return {'choices': [{'message': {'content': '{"entries": []}'}}], 'usage': {'cost': 0.01}}
    result = dictionary.prepare_upload(tmp_path / 'job', [path], 'fake', fake)
    assert result['status'] == ('success' if extra else 'skipped')
    assert len(calls) == int(extra)
    assert result['inventory']['fields']
    if extra:
        names = [field['name'] for field in request['evidence_access'].inventory()]
        assert 'represented_source_members' not in names
        assert any('calibration' in name for name in names) if kind == 'hdf5' else 'recorded_metadata' in names


@pytest.mark.parametrize('json_rate,expected,attr_kind', [
    ('rate_only', False, 'scalar'), ('rate_only', False, 'vlen'), ('rate_only', False, 'fixed'),
    ('sibling', True, 'vlen'), ('sibling', True, 'fixed')])
def test_json_rate_covers_only_selected_hdf_attribute_member(tmp_path, json_rate, expected, attr_kind):
    path = _prepared(tmp_path, 'hdf5', False, json_rate=json_rate, json_attr_kind=attr_kind)
    request = episode.build_request(path, inspect_evidence=True)
    assert bool(request['evidence_access']) == expected
    assert dictionary.needs_interpretation([path]) == expected
    assert 'control_freq' not in request['prompt']
    if expected:
        assert 'sensor_config' in request['prompt']


def test_unrecognized_hdf_numeric_array_opts_in(tmp_path):
    path = _prepared(tmp_path, 'hdf5', False, numeric_unknown=True)
    request = episode.build_request(path, inspect_evidence=True)
    assert request['evidence_access'] is not None
    assert dictionary.needs_interpretation([path])
    fields = request['evidence_access'].inventory()
    assert any(field['name'] == 'HDF5 recording calibration/response_table' for field in fields)


def test_same_named_sidecar_clock_is_still_unknown(tmp_path):
    path = _prepared(tmp_path, 'hdf5', False, sidecar=True)
    loaded = episode.load(path)
    assert loaded['context'].get('recorded_sensor_fields')
    request = episode.build_request(path, inspect_evidence=True)
    assert request['evidence_access'] is not None
    assert dictionary.needs_interpretation([path])


def test_legacy_hdf_coverage_without_source_identity_is_unknown(tmp_path):
    path = _prepared(tmp_path, 'hdf5', False)
    context_path = path / 'context.json'
    context = json.loads(context_path.read_text())
    source_members = next(iter(context['represented_source_members']['hdf5'].values()))
    context['represented_source_members']['hdf5'] = source_members
    context_path.write_text(json.dumps(context))
    assert episode.build_request(path, inspect_evidence=True)['evidence_access'] is not None
    assert dictionary.needs_interpretation([path])
