"""Retained capture rows and particular placement assumptions stay truthful downstream."""
import hashlib
import json
import subprocess

import h5py
import numpy as np
import pytest

from prepare import formats, signal_alignment, state_notes
from label import episode, contacts
from checks import capture_qc
from checks import contacts as contact_checks
from board import serve


def native_recording(tmp_path, camera='unique', force='unique', state=False):
    raw = tmp_path / 'upload'
    raw.mkdir(parents=True)
    origin = 1790000000000000003
    unique = origin + np.arange(40, dtype=np.int64) * 100000000
    tied = origin + np.repeat(np.arange(4), 10).astype(np.int64) * 1000000000
    gap = unique.copy()
    gap[20:] += 1000000000
    coarse_gap = tied.copy()
    coarse_gap[30:] += 3000000000
    backwards = unique.copy()
    backwards[20] = unique[19] - 100000000
    times = {'coarse_gap': coarse_gap, 'backwards': backwards, 'unique': unique, 'coarse': tied, 'constant': np.full(40, origin, np.int64), 'gap': gap}
    path = raw / 'recording.h5'
    with h5py.File(path, 'w') as f:
        f.attrs['fps'] = 10
        f.create_dataset('timestamps', data=times[camera]).attrs['units'] = 'ns'
        f.create_dataset('sensors/frame_timestamps' if state else 'sensors/timestamps', data=times[force]).attrs['units'] = 'ns'
        f['sensors/left_force'] = np.r_[np.zeros(10), np.ones(15), np.zeros(15)].astype(np.float32)
        if state:
            names = [f'{side}_joint{k}' for side in ('left', 'right')
                     for k in range(1, 7)]
            names = names[:6] + ['left_gripper'] + names[6:] + ['right_gripper']
            qpos = f.create_dataset('observations/qpos', data=np.ones((40, 14), np.float32))
            qpos.attrs['names'] = names
            f.create_dataset('observations/timestamps', data=unique).attrs['units'] = 'ns'
        for name in ('top', 'wrist_left'):
            f['images/' + name] = np.stack([np.full((36, 64, 3), k * 5, np.uint8) for k in range(40)])
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    report = formats.convert(raw, 'teleop_arms', tmp_path / 'prepared', 'test', 900)
    assert not report['failed'] and len(report['episodes']) == 1
    unit = tmp_path / 'prepared' / report['episodes'][0]['episode_id']
    request = episode.build_request(unit)
    assert request['content']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    return unit


@pytest.mark.parametrize('clock', ['coarse', 'constant'])
def test_repeated_native_camera_times_do_not_claim_a_gap_or_frame_loss(tmp_path, clock):
    unit = native_recording(tmp_path, camera=clock)
    result = capture_qc.run_episode(unit)
    flags = result['flags']
    assert any('repeat' in f['evidence'] for f in flags)
    assert not any(f['check'] in ('state_timestamp_gap', 'native_camera_timestamp_gap') for f in flags)
    assert not any('frames lost' in f['title'] for f in flags)


def test_measured_native_gap_and_clean_capture_controls(tmp_path):
    clean = native_recording(tmp_path / 'clean')
    assert capture_qc.run_episode(clean)['flags'] == []
    gap = native_recording(tmp_path / 'gap', camera='gap', force='gap')
    flags = capture_qc.run_episode(gap)['flags']
    assert {f['check'] for f in flags} == {'state_timestamp_gap', 'native_camera_timestamp_gap'}
    assert all('1100 ms' in f['evidence'] for f in flags)


def test_coarse_native_force_keeps_its_stamp_interval_assumption(tmp_path):
    unit = native_recording(tmp_path, force='coarse')
    cs = contacts.of_episode(episode.load(unit))
    assert cs and cs[0]['aligned_by'] == 'coarse clock'
    text = episode._contact_line(cs[0])
    assert 'stamp interval' in text
    assert 'shares no clock' not in text and 'both starts' not in text
    assert 'stamp interval' in episode.contact_placement(cs[0])


@pytest.mark.parametrize('by,words', [('coarse clock', 'stamp interval'),
    ('row per frame', 'row per frame'), ('assumed camera clock', 'camera'), ('future alignment', 'assum')])
def test_contact_checks_do_not_substitute_a_common_start(by, words):
    c = dict(id='c1', hand='left', start_s=1, end_s=2, peak_s=1.5, signals=['force'], aligned_by=by)
    result = contact_checks.check({'contacts_missing': [{'t_s': 2, 'hand': 'left', 'object': 'block'}]},
                                  [c], {}, 10)
    assert 'placed_from_both_starts' not in result
    evidence = result['notes'][0]['evidence']
    assert 'both starts' not in evidence and words in evidence


def test_common_start_contact_compatibility_and_measured_controls():
    c = dict(id='c1', hand='left', start_s=1, end_s=2, peak_s=1.5, signals=['force'], aligned_by='assumed start')
    assert episode.CONTACT_ASSUMED in episode._contact_line(c)
    result = contact_checks.check({'contacts_missing': [{'t_s': 2}]}, [c], {}, 10)
    assert result['placed_from_both_starts'] == ['c1']
    assert 'both starts' in result['notes'][0]['evidence']
    c.pop('aligned_by')
    assert 'placed' not in episode._contact_line(c)


def test_coarse_and_unknown_viewer_signal_timing_do_not_claim_no_shared_clock():
    src = serve.INDEX_HTML
    start = src.index('function snWhat(')
    end = src.index('function snErrorsHtml(', start)
    code = src[start:end]
    values = [{'dims': 1, 'aligned_by': 'coarse clock'}, {'dims': 1, 'aligned_by': 'future alignment'}]
    run = subprocess.run(['node', '-e', code + '\nconsole.log(JSON.stringify(' + json.dumps(values)
                          + '.map(snWhat)));'], capture_output=True, text=True, check=True)
    coarse, unknown = json.loads(run.stdout)
    assert 'stamp interval' in coarse
    assert 'both starts' not in coarse + unknown and 'no clock is shared' not in coarse + unknown


def test_retained_native_state_is_not_described_as_absent(tmp_path):
    unit = native_recording(tmp_path, camera='coarse', state=True)
    ep = episode.load(unit)
    assert ep['context']['state_kind'] == 'none'
    assert ep['context']['state_why'] == 'assumed_clock'
    assert 'recorded state' in ep['signals']
    result = capture_qc.run_episode(unit)
    checks = {c['check']: c for c in result['checks']}
    for name in capture_qc.STRUCTURE_CHECKS + capture_qc.GRIPPER_CHECKS + ('invalid_rotation_matrix',):
        assert checks[name]['status'] == 'not_applicable'
        assert 'no robot state, only video' not in checks[name]['why']
        assert ep['context']['state_note'] in checks[name]['why']


def test_absent_state_and_recorded_joint_reasons_stay_exact(tmp_path):
    absent = capture_qc.run_episode(native_recording(tmp_path / 'absent'))
    joint = capture_qc.run_episode(native_recording(tmp_path / 'joints', state=True))
    a = {c['check']: c for c in absent['checks']}
    j = {c['check']: c for c in joint['checks']}
    assert a['invalid_state_shape']['why'] == 'the recording has no robot state, only video'
    assert j['invalid_rotation_matrix']['why'] == ('the recording has joint angles but no gripper pose, '
                                                  'which this check needs')


def test_real_coarse_gap_remains_visible(tmp_path):
    result = capture_qc.run_episode(native_recording(tmp_path, camera='coarse_gap'))
    flags = result['flags']
    assert any(f['check'] == 'state_time_non_monotonic_or_duplicate' for f in flags)
    gaps = [f for f in flags if f['check'] in ('state_timestamp_gap', 'native_camera_timestamp_gap')]
    assert len(gaps) == 2 and all('4000 ms' in f['evidence'] for f in gaps)


def test_capture_consumer_keeps_backwards_clock_events(tmp_path):
    # A consumer probe supplies a backwards clock directly. Native malformed clock discovery is a separate rule.
    feats = capture_qc.extract(native_recording(tmp_path))
    ep = feats['ep']
    ep['recorded_times'] = {v: np.asarray(a).copy() for v, a in ep['times'].items()}
    for a in ep['recorded_times'].values():
        a[20] = a[19] - 0.1
    result = capture_qc.assess(feats)
    check = result['checks']['state_time_non_monotonic_or_duplicate']
    assert check['status'] == 'fired'
    assert len(check['events']) == 2 and all('backwards' in e['evidence'] for e in check['events'])


@pytest.mark.parametrize('why', ['layout', 'unreadable', 'short', 'assumed_clock'])
def test_check_state_reasons_use_shared_reader_authority(why):
    note = 'The particular recorded channel cannot supply measured state.'
    ep = {'context': {'state_why': why, 'state_note': note}, 'signals': {}}
    text = capture_qc._no_state_reason(ep)
    assert state_notes.STATE_WHY[why] in text and note in text
    assert 'no robot state, only video' not in text


def test_unknown_state_reason_and_retained_legacy_state_remain_qualified():
    for ctx in ({'state_why': 'future reason'}, {'state_why': 'not_recorded'}, {}):
        ep = {'context': ctx, 'signals': {'observations/qpos': np.ones((2, 14))}}
        text = capture_qc._no_state_reason(ep)
        if ctx.get('state_why') in (None, 'not_recorded'):
            assert state_notes.STATE_WHY['layout'] in text
        else:
            assert 'no usable robot state' in text
        assert 'no robot state, only video' not in capture_qc._no_state_reason(ep)
    command = {'context': {}, 'signals': {'leader_cmd_state': np.ones((2, 14))}}
    assert capture_qc._no_state_reason(command) == 'the recording has no robot state, only video'


def test_browser_placement_vocabulary_matches_native_for_all_kinds():
    src = serve.INDEX_HTML
    start = src.index('function placementText(')
    end = src.index('\n}\n', start) + 3
    values = [None, '', *signal_alignment.PLACEMENT_TEXT, 'future alignment', 'toString', '__proto__']
    run = subprocess.run(['node', '-e', src[start:end] + '\nconsole.log(JSON.stringify(' + json.dumps(values)
                          + '.map(placementText)));'], capture_output=True, text=True, check=True)
    assert json.loads(run.stdout) == [signal_alignment.placement_text(v) for v in values]
