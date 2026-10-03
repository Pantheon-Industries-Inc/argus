"""Named rejected clocks remain provenance while display placement stays qualified."""
import json
from pathlib import Path

import av
import h5py
import numpy as np
import pytest

from board import families
from checks import capture_qc
from label import episode
from label import signals as signal_words
from prepare import camera_clock, formats
from test_container_camera_clocks import convert, hdf_upload
from test_timing_truth import native_recording


@pytest.mark.parametrize('bad', ['backwards', 'nan', 'infinity', 'first_nan', 'all_nan'])
def test_named_hdf_rejected_clock_keeps_originals_frames_and_qualified_state(tmp_path, bad):
    raw = np.arange(40, dtype=float) / 10
    if bad == 'backwards':
        raw = 1790000000000000003 + np.arange(40, dtype=np.int64) * 100000000
        raw[20] = raw[18]
    elif bad == 'all_nan':
        raw[:] = np.nan
    else:
        raw[0 if bad == 'first_nan' else 20] = np.nan if 'nan' in bad else np.inf
    upload = hdf_upload(tmp_path, raw, signal=True)
    with h5py.File(upload / 'recording.h5', 'a') as f:
        ds = f.create_dataset('observations/qpos', data=np.ones((40, 14), np.float32))
        ds.attrs['names'] = [f'{side}_{v}' for side in ('left', 'right')
                             for v in ('joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'gripper')]
    ep, ctx = convert(upload, tmp_path)
    assert 'no frame times' not in ctx.get('source', {}).get('clock_note', '')
    with np.load(ep / ctx['recorded_container_times']['file']) as clocks:
        assert clocks['timestamps'].dtype == raw.dtype
        np.testing.assert_array_equal(clocks['timestamps'], raw)
    assert ctx['recorded_container_times']['cameras']['exo'] == 'timestamps'
    assert ctx['state_kind'] == 'none' and ctx['state_why'] == 'assumed_clock'
    assert 'tied' not in ctx['state_note']
    assert ctx['camera_clock']['exo']['clock_problem'] == ('backwards' if bad == 'backwards' else 'nonfinite')
    assert ctx['camera_clock']['exo']['cadence_source'] == 'declared nominal cadence'
    with np.load(ep / ctx['presentation_times']) as clocks:
        np.testing.assert_allclose(clocks['exo'], np.arange(40) / 10)
    with np.load(ep / 'signals.npz') as z:
        force = next(s for s in ctx['signals'] if s['name'] == 'force')
        np.testing.assert_array_equal(z[force['key']].ravel(), np.arange(40))
    assert all(s['camera_aligned_by'] == 'assumed camera clock' for s in ctx['signals'])
    for path in ep.glob('*.mp4'):
        with av.open(str(path)) as c:
            assert sum(1 for _ in c.decode(video=0)) == 40
    assert ctx['unshown_cameras'][0]['n_frames'] == 40
    assert any(i['kind'] == 'camera_timestamp_invalid' for i in ctx['reader_issues'])
    request = episode.build_request(ep)
    assert 'assumed presentation' in request['prompt']
    assert 'The times are exact: use them, do not invent your own.' not in request['prompt']


@pytest.mark.parametrize('raw', [np.array([0., 1., .5]), np.array([np.nan, 1., 2.]), np.full(3, np.nan)])
def test_invalid_display_clock_never_claims_measured_cadence(raw):
    shown, note = camera_clock.presentation_clock(raw)
    assert note and note['cadence_source'] == 'unknown cadence fallback'
    assert 'never a measured capture rate' in note['what']
    assert np.isfinite(shown).all() and (np.diff(shown) > 0).all()


@pytest.mark.parametrize('invalid', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('bad_view', ['exo', 'left'])
def test_nonfinite_capture_is_not_assessed_per_clock_and_preserves_other_camera(tmp_path, invalid, bad_view):
    feats = capture_qc.extract(native_recording(tmp_path))
    ep = feats['ep']
    recorded = {v: np.asarray(a).copy() for v, a in ep['times'].items()}
    recorded[bad_view][20] = invalid
    other = 'left' if bad_view == 'exo' else 'exo'
    recorded[other][20:] += 1
    ep['recorded_times'] = recorded
    with np.errstate(invalid='raise'):
        checks = capture_qc.assess(feats)['checks']
    if bad_view == 'exo':
        assert checks['state_timestamp_gap']['status'] == 'not_assessed'
        assert 'nonfinite' in checks['state_timestamp_gap']['why']
        assert checks['native_camera_timestamp_gap']['status'] == 'fired'
        assert checks['native_camera_timestamp_gap']['events'][0]['camera'] == 'left'
    else:
        assert checks['state_timestamp_gap']['status'] == 'fired'
        assert checks['native_camera_timestamp_gap']['status'] == 'not_assessed'
        assert 'nonfinite' in checks['native_camera_timestamp_gap']['why']
    duplicate = checks['state_time_non_monotonic_or_duplicate']
    assert duplicate['status'] == 'not_assessed' and 'nonfinite' in duplicate['why']
    assert not duplicate['events']


def test_invalid_camera_clock_has_a_closed_family():
    catalog = json.loads((Path(__file__).parents[1] / 'board' / 'families.json').read_text())
    assert any('camera_timestamp_invalid' in f.get('reader_issues', []) for f in catalog['families'])


def test_unusable_signal_clock_on_valid_camera_never_becomes_measured_state(tmp_path):
    upload = hdf_upload(tmp_path, np.arange(40) / 10)
    with h5py.File(upload / 'recording.h5', 'a') as f:
        bad = np.arange(40) / 10
        bad[20] = np.nan
        f.create_dataset('observations/timestamps', data=bad)
        ds = f.create_dataset('observations/qpos', data=np.ones((40, 14), np.float32))
        ds.attrs['names'] = [f'{side}_{v}' for side in ('left', 'right')
                             for v in ('joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'gripper')]
    ep, ctx = convert(upload, tmp_path)
    assert ctx['state_kind'] == 'none' and ctx['state_why'] == 'assumed_clock'
    qpos = next(s for s in ctx['signals'] if s['name'] == 'observations/qpos')
    assert qpos['aligned_by'] == 'row per frame'
    assert qpos['clock_problem'] == 'nonfinite'
    with np.load(ep / 'signals.npz') as z:
        np.testing.assert_array_equal(z[qpos['key']], np.ones((40, 14)))


def test_adjacent_nonfinite_display_clock_has_no_invalid_arithmetic():
    with np.errstate(invalid='raise'):
        shown, note = camera_clock.presentation_clock(np.array([np.inf, np.inf, np.nan]))
    assert note['clock_problem'] == 'nonfinite' and np.isfinite(shown).all()


def test_all_nonfinite_main_clock_does_not_move_valid_side_to_a_false_epoch_offset(tmp_path):
    upload = tmp_path / 'upload'
    upload.mkdir()
    with h5py.File(upload / 'recording.h5', 'w') as f:
        f.attrs['fps'] = 10
        for name, t in [('top', np.full(40, np.nan)), ('wrist_left', 1790000000 + np.arange(40) / 10)]:
            f.create_dataset(name + ('/timestamps' if name == 'top' else '/frame_timestamps'), data=t)
            f.create_dataset(name + '/' + name + '_images', data=np.stack([
                np.full((36, 64, 3), k * 5, np.uint8) for k in range(40)]))
    ep, ctx = convert(upload, tmp_path)
    unit = episode.load(ep)
    np.testing.assert_allclose(unit['times']['left'], np.arange(40) / 10, atol=3e-7)
    assert ctx['cameras']['left'] and 'left' not in ctx['camera_clock']
    with np.load(ep / 'recorded_container_times.npz') as z:
        np.testing.assert_array_equal(z['wrist_left/frame_timestamps'], 1790000000 + np.arange(40) / 10)

def test_invalid_depth_clock_does_not_claim_shared_timestamps(tmp_path):
    upload = hdf_upload(tmp_path, np.arange(40) / 10)
    with h5py.File(upload / 'recording.h5', 'a') as f:
        raw = np.arange(40) / 10
        raw[20] = np.inf
        f.create_dataset('depth/timestamps', data=raw)
        f.create_dataset('depth/top_depth', data=np.stack([np.full((64, 64), 500 + k, np.uint16)
                                                         for k in range(40)]))
    ep, ctx = convert(upload, tmp_path)
    assert ctx['depth_camera_clock']['exo']['clock_problem'] == 'nonfinite'
    with np.load(ep / 'recorded_container_times.npz') as z:
        np.testing.assert_array_equal(z['depth/timestamps'], raw)
    req = episode.build_request(ep)
    assert 'Shared timestamps' not in req['prompt']
    assert 'unusable recorded clock' in req['prompt']
    assert req['depth_views'] == ['exo']
    with av.open(str(ep / 'depth_exo.mkv')) as c:
        decoded = [fr.to_ndarray() for fr in c.decode(video=0)]
    np.testing.assert_array_equal(decoded, np.stack([np.full((64, 64), 500 + k, np.uint16)
                                                    for k in range(40)]))

def test_valid_recorded_state_on_invalid_camera_does_not_invent_an_action(tmp_path):
    upload = hdf_upload(tmp_path, np.r_[np.arange(20) / 10, np.nan, np.arange(21, 40) / 10])
    with h5py.File(upload / 'recording.h5', 'a') as f:
        f.create_dataset('observations/timestamps', data=np.arange(40) / 10)
        ds = f.create_dataset('observations/qpos', data=np.ones((40, 14), np.float32))
        ds.attrs['names'] = [f'{side}_{v}' for side in ('left', 'right')
                             for v in ('joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'gripper')]
    ep, ctx = convert(upload, tmp_path)
    assert ctx['state_kind'] == 'none' and ctx['state_why'] == 'assumed_clock'
    assert 'recorded state' in {s['name'] for s in ctx['signals']}
    assert 'recorded action' not in {s['name'] for s in ctx['signals']}
    assert 'state and action remain signals' not in ctx['state_note']


def test_invalid_side_clock_keeps_valid_anchor_depth_measured(tmp_path):
    upload = tmp_path / 'upload'
    upload.mkdir()
    with h5py.File(upload / 'recording.h5', 'w') as f:
        f.attrs['fps'] = 10
        for name, leaf in [('top', 'timestamps'), ('wrist_left', 'frame_timestamps')]:
            t = np.arange(40) / 10
            if name == 'wrist_left':
                t[20] = np.nan
            f.create_dataset(name + '/' + leaf, data=t)
            f.create_dataset(name + '/' + name + '_images', data=np.stack([
                np.full((36, 64, 3), k * 5, np.uint8) for k in range(40)]))
        f.create_dataset('depth/capture_times', data=np.arange(40) / 10)
        f.create_dataset('depth/top_depth', data=np.stack([np.full((64, 64), 500 + k, np.uint16)
                                                         for k in range(40)]))
    ep, ctx = convert(upload, tmp_path)
    assert ctx['camera_clock']['left']['clock_problem'] == 'nonfinite'
    assert 'exo' not in ctx['camera_clock'] and not ctx.get('depth_camera_clock')
    request = episode.build_request(ep)
    line = next(x for x in request['prompt'].splitlines() if x.startswith('DEPTH:'))
    assert ' at that instant.' in line and 'unusable recorded clock' not in line
    result = capture_qc.run_episode(ep)
    clocks = result['metrics']['clocks']
    assert clocks['anchor']['status'] == 'assessed'
    assert clocks['cameras']['left']['status'] == 'not_assessed'
    assert 'nonfinite' in clocks['cameras']['left']['why']
    checks = {c['check']: c for c in result['checks']}
    assert checks['state_timestamp_gap']['status'] == 'clear'
    assert checks['native_camera_timestamp_gap']['status'] == 'not_applicable'
    assert 'nonfinite' in checks['native_camera_timestamp_gap']['why']


def test_native_nonfinite_clock_without_declared_rate_names_only_display_fallback(tmp_path):
    upload = hdf_upload(tmp_path, np.full(40, np.nan))
    with h5py.File(upload / 'recording.h5', 'a') as f:
        del f.attrs['fps']
    ep, ctx = convert(upload, tmp_path)
    note = ctx['camera_clock']['exo']
    assert note['cadence_source'] == 'unknown cadence fallback'
    assert note['nominal_fps'] == 30 and 'never a measured capture rate' in note['what']
    req = episode.build_request(ep)
    assert note['what'] in req['prompt'] and 'assumed presentation' in req['prompt']
    checks = {c['check']: c for c in capture_qc.run_episode(ep)['checks']}
    assert checks['state_timestamp_gap']['status'] == 'not_applicable'
    assert 'nonfinite' in checks['state_timestamp_gap']['why']
    with av.open(str(ep / 'exo.mp4')) as c:
        assert sum(1 for _ in c.decode(video=0)) == 40


def test_invalid_signal_clock_uses_row_assumption_with_valid_camera(tmp_path):
    upload = hdf_upload(tmp_path, np.arange(40) / 10)
    with h5py.File(upload / 'recording.h5', 'a') as f:
        raw = np.arange(40) / 10
        raw[20] = np.nan
        f.create_dataset('sensors/timestamps', data=raw)
        f.create_dataset('sensors/left_force', data=np.arange(40, dtype=np.float32))
    ep, ctx = convert(upload, tmp_path)
    signal = next(s for s in ctx['signals'] if s['name'] == 'sensors/left_force')
    assert signal['aligned_by'] == 'row per frame'
    assert signal['clock_problem'] == 'nonfinite'
    assert not ctx.get('camera_clock')
    issue = next(i for i in ctx['reader_issues'] if i['kind'] == 'signal_timestamp_invalid')
    assert issue['signal'] == 'sensors/left_force' and 'row per frame' in issue['what']
    req = episode.build_request(ep)
    assert 'assumed camera presentation clock' not in req['prompt']
    assert 'placed one row per frame' in req['prompt']


@pytest.mark.parametrize('rows,bad', [(20, 'nan'), (60, 'backwards')])
def test_invalid_signal_clock_with_unequal_original_rows_is_not_padded_or_trimmed(tmp_path, rows, bad):
    upload = hdf_upload(tmp_path, np.arange(40) / 10)
    raw = np.arange(rows, dtype=np.float64) / 10
    raw[10] = np.nan if bad == 'nan' else raw[8]
    values = np.arange(rows * 14, dtype=np.float32).reshape(rows, 14)
    with h5py.File(upload / 'recording.h5', 'a') as f:
        f.create_dataset('observations/timestamps', data=raw)
        f.create_dataset('observations/qpos', data=values)
    ep, ctx = convert(upload, tmp_path)
    assert ctx['state_kind'] == 'none'
    assert not ctx.get('camera_clock')
    assert not any(s['name'] == 'observations/qpos' for s in ctx.get('signals', []))
    issue = next(i for i in ctx['reader_issues'] if i['kind'] == 'signal_timestamp_invalid')
    assert issue['source_rows'] == rows and issue['camera_frames'] == 40
    assert 'not placed' in issue['what'] and 'one row per frame' not in issue['what']
    assert issue['clock_problem'] == ('nonfinite' if bad == 'nan' else 'backwards')
    with np.load(ep / ctx['recorded_container_times']['file']) as z:
        np.testing.assert_array_equal(z['observations/timestamps'], raw)
        assert z['observations/timestamps'].dtype == raw.dtype
    with h5py.File(upload / 'recording.h5') as f:
        np.testing.assert_array_equal(f['observations/qpos'][()], values)
    req = episode.build_request(ep)
    assert 'assumed camera presentation clock' not in req['prompt']
    assert issue['what'] in req['prompt']


def test_invalid_signal_rate_and_original_row_counts_stay_qualified_in_words():
    values = np.arange(20, dtype=np.float32)[:, None]
    text = signal_words.describe('force', values, rate_hz=5, fps=10, aligned_by='row per frame',
                                 clock_problem='nonfinite', source_rows=40, camera_frames=40)
    assert 'recorded at' not in text
    assert 'estimated at 5 Hz from assumed placement' in text
    assert '40 original rows' in text and '40 original camera frames' in text
    assert 'assumption' in text
    exact = signal_words.describe('force', values, rate_hz=5, fps=10)
    assert 'recorded at 5 Hz' in exact and 'assumption' not in exact


def test_invalid_signal_clock_does_not_hide_valid_recorded_state_or_promote_action(tmp_path):
    upload = hdf_upload(tmp_path, np.arange(40) / 10)
    with h5py.File(upload / 'recording.h5', 'a') as f:
        f.create_dataset('observations/timestamps', data=np.arange(40) / 10)
        ds = f.create_dataset('observations/qpos', data=np.ones((40, 14), np.float32))
        names = [f'{side}_{v}' for side in ('left', 'right')
                 for v in ('joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'gripper')]
        ds.attrs['names'] = names
        raw = np.arange(40) / 10
        raw[20] = np.inf
        f.create_dataset('commands/capture_times', data=raw)
        ds = f.create_dataset('commands/action', data=np.arange(560, dtype=np.float32).reshape(40, 14))
        ds.attrs['names'] = names
    ep, ctx = convert(upload, tmp_path)
    assert ctx['state_kind'] != 'none' and (ep / 'state.npz').exists()
    with np.load(ep / 'state.npz') as z:
        assert 'action' not in z.files
    action = next(s for s in ctx['signals'] if s['name'] == 'commands/action')
    assert action['clock_problem'] == 'nonfinite' and action['aligned_by'] == 'row per frame'
    assert action['source_rows'] == 40 and action['camera_frames'] == 40
    assert not ctx.get('camera_clock')
