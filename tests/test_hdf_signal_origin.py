"""A damaged camera origin must not exclude valid sensor readings on its epoch clock."""
import json

import h5py
import numpy as np
import pytest

from label import episode
from prepare import formats


@pytest.mark.parametrize('origin,bad,side', [
    (1790000000., 'first', False),
    (1790000000., 'all', True),
    (0., 'first', False),
    (1790000000., 'none', False),
])
@pytest.mark.parametrize('external', [False, True])
def test_valid_force_rows_survive_nonfinite_camera_origin(tmp_path, origin, bad, side, external):
    upload = tmp_path / 'upload'
    upload.mkdir()
    t = origin + np.arange(40) / 10
    raw = t.copy()
    if bad == 'first':
        raw[0] = np.nan
    elif bad == 'all':
        raw[:] = np.nan
    values = np.arange(40, dtype=np.float32)
    with h5py.File(upload / 'recording.h5', 'w') as f:
        f.attrs['fps'] = 10
        f.create_dataset('timestamps', data=raw).attrs['units'] = 's'
        f.create_dataset('images/top', data=np.stack([
            np.full((36, 64, 3), 5 * k, np.uint8) for k in range(40)
        ]))
        if side:
            f.create_dataset('wrist/frame_timestamps', data=t).attrs['units'] = 's'
            f.create_dataset('wrist/wrist_left_images', data=np.stack([
                np.full((36, 64, 3), 11 + 5 * k, np.uint8) for k in range(40)
            ]))
        if not external:
            f.create_dataset('sensors/timestamps', data=t).attrs['units'] = 's'
            force = f.create_dataset('sensors/left_force', data=values)
            force.attrs['units'] = np.bytes_(b'N')
            force.attrs['coordinate_frame'] = 'sensor'
    if external:
        with h5py.File(upload / 'force.h5', 'w') as f:
            f.attrs['operator_note'] = 'zeroed before recording'
            f.create_dataset('calibration/response_table', data=np.arange(1024).reshape(128, 8))
            f.create_dataset('timestamps', data=t).attrs['units'] = 's'
            force = f.create_dataset('left_force', data=values)
            force.attrs['units'] = np.bytes_(b'N')
            force.attrs['coordinate_frame'] = 'sensor'
    report = formats.convert(upload, 'teleop_arms', tmp_path / 'prepared', 'test', 900)
    assert not report['failed'] and len(report['episodes']) == 1
    ep = tmp_path / 'prepared' / report['episodes'][0]['episode_id']
    ctx = json.loads((ep / 'context.json').read_text())
    force = next(s for s in ctx.get('signals', []) if s['name'].endswith('left_force'))
    assert force['units'] == 'N' and force['coordinate_frame'] == 'sensor'
    with np.load(ep / 'signals.npz') as arrays:
        np.testing.assert_array_equal(arrays[force['key']].ravel(), values)
    if bad != 'none':
        assert force['camera_aligned_by'] == 'assumed camera clock'
        assert 'assumed presentation' in episode.build_request(ep)['prompt']
    if bad == 'first':
        assert ctx['clock_start_assumed'] is True
        assert 'source clock origin is estimated' in episode.build_request(ep)['prompt']
    if external and bad == 'none':
        from label import evidence_access
        access = evidence_access.Access(episode.load(ep))
        note = next(v for v in access.inventory() if v['name'] == 'HDF5 recording / attribute operator_note'
                    and v['descriptor']['source_file'] == 'force.h5')
        result = access.inspect({'field_id': note['id'], 'mode': 'metadata'})
        assert result['value'] == 'zeroed before recording' and 'times_s' not in result
        calibration = next(v for v in access.inventory() if v['name'] == 'HDF5 recording calibration/response_table'
                           and v['descriptor']['source_file'] == 'force.h5')
        assert access.inspect({'field_id': calibration['id'], 'mode': 'metadata', 'pointer': '/127/7'})['value'] == 1023
