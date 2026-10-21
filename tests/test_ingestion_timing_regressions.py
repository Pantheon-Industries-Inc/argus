"""Clock placement regressions using native MCAP messages and declared HDF state."""
import json

import numpy as np
import pytest
from mcap.writer import Writer

from prepare import formats


def sensor_file(path, messages, topic='/sensor'):
    with path.open('wb') as stream:
        writer = Writer(stream)
        writer.start()
        schema = writer.register_schema('sensor_msgs/msg/Imu', 'jsonschema', b'{}')
        channel = writer.register_channel(topic, 'json', schema)
        for stamp, body in messages:
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            writer.add_message(channel, int(stamp * 1e9), data, int(stamp * 1e9))
        writer.finish()


def test_signals_from_overlapping_files_keep_values_on_their_capture_times(tmp_path):
    early, late = tmp_path / 'early.mcap', tmp_path / 'late.mcap'
    sensor_file(early, [(t, {'pressure': v}) for t, v in [(0, 2), (.2, 9), (.4, 3)]])
    sensor_file(late, [(t, {'pressure': v}) for t, v in [(.1, 7), (.3, 1), (.5, 8)]])
    q = np.arange(6) / 10
    out = formats.mcap_signals([late, early], q)
    # The two files' readings overlap in time, so nothing recorded says they are one recording: each is kept apart,
    # every value on its own capture time, and a note says why (Signals.keep, relate).
    np.testing.assert_array_equal(out['early /sensor'][[0, 2, 4], 0], [2, 9, 3])
    np.testing.assert_array_equal(out['late /sensor'][[1, 3, 5], 0], [7, 1, 8])
    assert any(i['kind'] == 'signal_name_collision' and 'overlap' in i['what'] for i in out.issues)


def test_one_bad_message_keeps_later_signals_and_records_decode_failure(tmp_path):
    source = tmp_path / 'malformed.mcap'
    sensor_file(source, [(0, {'pressure': 2}), (.1, b'{broken'), (.2, {'pressure': 9}),
                         (.3, {'pressure': 3})])
    out = formats.mcap_signals([source], np.array([0, .2, .3]))
    np.testing.assert_array_equal(out['/sensor'][:, 0], [2, 9, 3])
    context = {'n_state_frames': 3, 'fps': 10}
    formats.write_signals(tmp_path, context, out)
    issue = next(i for i in context['reader_issues'] if i['kind'] == 'mcap_message_unreadable')
    assert issue['signal'] == '/sensor' and '1 message' in issue['what']
    assert source.name in issue['what']


def test_signal_programming_errors_are_not_reported_as_successful_partial_reads(tmp_path, monkeypatch):
    source = tmp_path / 'valid.mcap'
    sensor_file(source, [(0, {'pressure': 2}), (.1, {'pressure': 9})])
    def broken_numbers(decoded):
        raise RuntimeError('regression in numeric extraction')
    monkeypatch.setattr(formats, '_numbers', broken_numbers)
    with pytest.raises(RuntimeError, match='regression in numeric extraction'):
        formats.mcap_signals([source], np.array([0, .1]))


@pytest.mark.parametrize('columns', [7, 14])
def test_hdf_pose_gap_follows_angle_wrap_and_keeps_recorded_values(columns):
    names = [f'{side}{name}' for side in ([''] if columns == 7 else ['left_', 'right_'])
             for name in ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']]
    values = np.zeros((3, columns))
    for start in range(0, columns, 7):
        values[:, start] = [0, np.nan, 2]
        values[:, start + 5] = np.deg2rad([179, np.nan, -179])
    signals = formats.Signals()
    metadata = {'units': (['m'] * 3 + ['rad'] * 3 + ['m']) * (columns // 7)}
    signals.add('state', values, names=names, metadata=metadata)
    signals.add('action', values.copy(), names=names, metadata=metadata)
    state, action, _, _, note = formats.h5_state(signals, 'handheld_gripper', np.arange(3) / 30)
    assert note is None
    np.testing.assert_array_equal(action, state)
    np.testing.assert_array_equal(state[[0, 2]], values[[0, 2]])
    for start in range(0, columns, 7):
        assert state[1, start] == pytest.approx(1)
        assert abs(state[1, start + 5]) == pytest.approx(np.pi)


def test_packed_lerobot_v3_counts_its_window_packets_and_keeps_publisher_clock(tmp_path):
    import pandas as pd
    from PIL import Image
    from label import episode
    from test_formats import _lerobot_v3

    root = tmp_path / 'upload'
    camera, rear = 'observation.images.front', 'observation.images.rear'
    _lerobot_v3(root, {camera: 7, rear: 7}, None)
    video = root / 'videos' / camera / 'chunk-000/file-000.mp4'
    writer = formats.FrameWriter(video, '')
    for stamp, value in [(0, 20), (.1, 40), (.2, 60), (.31, 80), (.45, 100), (.51, 120), (.6, 140)]:
        writer.add_image(stamp, Image.new('RGB', (64, 36), (value,) * 3))
    assert writer.close() == 7
    rear_video = root / 'videos' / rear / 'chunk-000/file-000.mp4'
    rear_video.write_bytes(video.read_bytes())
    metadata = root / 'meta/episodes/chunk-000/file-000.parquet'
    metadata.parent.mkdir(parents=True)
    pd.DataFrame([{'episode_index': 0, 'length': 3,
                   f'videos/{camera}/chunk_index': 0, f'videos/{camera}/file_index': 0,
                   f'videos/{camera}/from_timestamp': .19, f'videos/{camera}/to_timestamp': .5,
                   f'videos/{rear}/chunk_index': 0, f'videos/{rear}/file_index': 0,
                   f'videos/{rear}/from_timestamp': .19, f'videos/{rear}/to_timestamp': .5}]).to_parquet(metadata)
    report = formats.convert(root, 'ego_head', tmp_path / 'out', 'packed clock', 900)
    assert not report['failed'], report
    ep = tmp_path / 'out' / report['episodes'][0]['episode_id']
    loaded = episode.load(ep)
    assert loaded['sources']['exo']['n_frames'] == 3
    unshown = loaded['context']['unshown_cameras'][0]
    assert unshown['name'] == rear and unshown['n_frames'] == 3
    with np.load(ep / unshown['camera_times']) as retained:
        np.testing.assert_array_equal(retained['pts'], [200000, 310000, 450000])
    # the frames are shown at their own encoded times within the episode's window, not at a declared 30 per second
    np.testing.assert_allclose(loaded['times']['exo'], [.01, .12, .26], atol=1e-9)
    np.testing.assert_array_equal(loaded['times']['exo_pts'], [200000, 310000, 450000])
    assert any(i['kind'] == 'declared_rate_wrong' for i in loaded['context']['reader_issues'])
    pictures = episode._decode_view(loaded, 'exo', [0, 1, 2])
    assert [np.asarray(pictures[i]).mean() for i in range(3)] == pytest.approx([60, 80, 100], abs=3)


def h264_packets(count):
    import av
    from fractions import Fraction
    encoder = av.CodecContext.create('libx264', 'w')
    encoder.width, encoder.height, encoder.pix_fmt = 64, 48, 'yuv420p'
    encoder.time_base = Fraction(1, 30)
    encoder.options = {'preset': 'medium', 'x264-params': 'bframes=2:b-adapt=0:keyint=30:scenecut=0'}
    packets = []
    for i in range(count):
        frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i * 20, np.uint8), format='rgb24')
        frame.pts = i
        packets.extend(encoder.encode(frame))
    packets.extend(encoder.encode())
    return packets


@pytest.fixture
def reordered_h264():
    packets = h264_packets(10)
    assert (np.diff([packet.pts for packet in packets]) < 0).any()
    return [bytes(packet) for packet in packets]


@pytest.mark.parametrize('writer_path', ['remux', 'frame_writer'])
def test_annexb_copy_rejects_reordered_frames_instead_of_fabricating_capture_pairs(
        tmp_path, reordered_h264, writer_path):
    from prepare.remux import remux
    output = tmp_path / 'camera.mp4'
    with pytest.raises(ValueError, match='reordered.*B.frames'):
        if writer_path == 'remux':
            remux(reordered_h264, np.arange(10) / 30, 'h264', output)
        else:
            writer = formats.FrameWriter(output, 'h264')
            for i, packet in enumerate(reordered_h264):
                writer.add(i / 30, packet)
            writer.close()


def test_uninitialized_ros_header_uses_arrival_for_display_without_losing_native_zero(tmp_path):
    import base64
    import io
    from PIL import Image
    from label import episode
    image = io.BytesIO()
    Image.new('RGB', (64, 48), (80,) * 3).save(image, format='JPEG')
    payload = base64.b64encode(image.getvalue()).decode()
    source = tmp_path / 'zero_header.mcap'
    topics = ['/camera/front/image', '/camera/wrist/image']
    arrival = 1_700_000_000_000_000_000 + np.arange(6, dtype=np.int64) * 33_333_333
    with source.open('wb') as stream:
        writer = Writer(stream)
        writer.start()
        schema = writer.register_schema('sensor_msgs/msg/CompressedImage', 'jsonschema', b'{}')
        channels = [writer.register_channel(topic, 'json', schema) for topic in topics]
        for stamp in arrival:
            for index, channel in enumerate(channels):
                sec, nanos = divmod(int(stamp) if index else 0, 1_000_000_000)
                body = {'format': 'jpeg', 'data': payload,
                        'header': {'stamp': {'sec': sec, 'nanosec': nanos}}}
                writer.add_message(channel, int(stamp), json.dumps(body).encode(), int(stamp))
        writer.finish()
    ep = tmp_path / 'prepared'
    ctx = formats.convert_mcap_generic({'file': source, 'name': 'zero', 'seconds': .2, 'topics': topics},
                                       'teleop_arms', ep, 'ROS header clocks')
    loaded = episode.load(ep)
    assert ctx['source']['camera_clock'] == 'arrival'
    assert ctx['source']['presentation_clock']['basis'] == 'mcap_log_time'
    assert any(i['kind'] == 'capture_timestamp_invalid' and 'uninitialized' in i['what']
               for i in ctx['reader_issues'])
    assert not any(i['kind'] in ('camera_coverage_partial', 'main_camera_short') for i in ctx['reader_issues'])
    for view in loaded['sources']:
        np.testing.assert_allclose(loaded['times'][view], np.arange(6) / 30, atol=1e-6)
    np.testing.assert_array_equal(loaded['kmap'].get('extra1', np.arange(6)), np.arange(6))
    inventory = next(row for row in ctx['mcap_field_inventory'] if row['topic'] == topics[0])
    with np.load(ep / ctx['recorded_mcap_fields']) as native:
        np.testing.assert_array_equal(native[inventory['capture_ns']], np.zeros(6, dtype=np.int64))
        np.testing.assert_array_equal(native[inventory['log_ns']], arrival)
    assert inventory['capture_fields'] == [{'sec': 0, 'nanosec': 0}] * 6


def test_portrait_probe_and_decoded_picture_agree_without_ffprobe(tmp_path, monkeypatch):
    import av
    import shutil
    import subprocess
    from label import frames
    from prepare import display
    from test_formats import _clip

    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        pytest.skip('ffmpeg is required to author display rotation metadata')
    source, portrait = tmp_path / 'landscape.mp4', tmp_path / 'portrait.mp4'
    _clip(source, 3)
    subprocess.run([ffmpeg, '-v', 'error', '-y', '-display_rotation', '90', '-i', str(source),
                    '-c', 'copy', str(portrait)], check=True)
    # no ffprobe anywhere: none on PATH, none in the install places prepare/tools.py also looks in
    from prepare import tools
    monkeypatch.setenv('PATH', '')
    monkeypatch.delenv('FFPROBE', raising=False)
    monkeypatch.delenv('FFMPEG', raising=False)
    monkeypatch.setattr(tools, 'INSTALL_DIRS', ())
    display._geometry.cache_clear()
    geometry = display.geometry(str(portrait))
    assert geometry['stored'] == (0, 0)
    facts = formats.probe(portrait)
    with av.open(str(portrait)) as container:
        picture = frames.upright(next(container.decode(video=0)), geometry)
    assert facts['rotation'] == 90
    assert picture.size == (facts['width'], facts['height']) == (36, 64)


def test_reordered_mcap_camera_reports_why_preparation_failed(tmp_path, reordered_h264):
    import base64
    source = tmp_path / 'reordered.mcap'
    with source.open('wb') as stream:
        writer = Writer(stream)
        writer.start()
        schema = writer.register_schema('foxglove.CompressedVideo', 'jsonschema', b'{}')
        channel = writer.register_channel('/camera/front/image', 'json', schema)
        for index, packet in enumerate(reordered_h264):
            stamp = 1_000_000_000 + index * 33_333_333
            body = {'format': 'h264', 'data': base64.b64encode(packet).decode(),
                    'timestamp': {'sec': 1, 'nsec': index * 33_333_333}}
            writer.add_message(channel, stamp, json.dumps(body).encode(), stamp)
        writer.finish()
    report = formats.convert(source, 'ego_head', tmp_path / 'prepared', 'reordered camera', 900)
    assert not report['episodes']
    assert len(report['failed']) == 1
    assert 'B-frames' in report['failed'][0]['why']
    assert 'decode timestamps' in report['failed'][0]['why']


def test_hdf_pose_gap_uses_short_physical_rotation_near_gimbal_lock():
    from scipy.spatial.transform import Rotation
    values = np.zeros((3, 7))
    values[0, 3:6] = np.deg2rad([-100, -80, 85])
    values[1] = np.nan
    values[2, 3:6] = np.deg2rad([80, -83, -99])
    signals = formats.Signals()
    signals.add('state', values, names=['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper'],
                metadata={'units': ['m'] * 3 + ['rad'] * 3 + ['m']})
    state, _, _, _, note = formats.h5_state(signals, 'handheld_gripper', np.arange(3) / 30)
    assert note is None
    np.testing.assert_array_equal(state[[0, 2]], values[[0, 2]])
    rotations = Rotation.from_euler('xyz', state[:, 3:6])
    steps = (rotations[:-1].inv() * rotations[1:]).magnitude()
    # The recorded orientations are 17.46 degrees apart, not two rotations near 180 degrees.
    assert np.rad2deg(steps) == pytest.approx([8.73044, 8.73044], abs=.00001)


@pytest.mark.parametrize('count', [1, 2])
@pytest.mark.parametrize('writer_path', ['remux', 'frame_writer'])
def test_bframe_capability_without_actual_reordering_keeps_short_camera(tmp_path, count, writer_path):
    import av
    from prepare.remux import remux
    output = tmp_path / 'short.mp4'
    packets = h264_packets(count)
    assert all(p.pts == p.dts for p in packets)
    encoded = [bytes(p) for p in packets]
    if writer_path == 'remux':
        remux(encoded, np.arange(count) / 30, 'h264', output)
    else:
        writer = formats.FrameWriter(output, 'h264')
        for i, packet in enumerate(encoded):
            writer.add(i / 30, packet)
        assert writer.close() == count
    with av.open(str(output)) as container:
        frames = list(container.decode(video=0))
    assert len(frames) == count
    assert [float(f.pts * f.time_base) for f in frames] == pytest.approx(np.arange(count) / 30, abs=2e-6)


def test_joint_stream_extraction_failure_is_not_returned_as_partial_success(tmp_path, monkeypatch):
    source = tmp_path / 'joints.mcap'
    sensor_file(source, [(i / 10, {'name': [f'joint{j}_rad' for j in range(6)] + ['gripper_m'],
                                 'position': [float(i)] * 7}) for i in range(6)])
    original, calls = formats._joint_names, []
    def broken_names(*args):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError('joint extraction regression')
        return original(*args)
    monkeypatch.setattr(formats, '_joint_names', broken_names)
    with pytest.raises(RuntimeError, match='joint extraction regression'):
        formats.mcap_joint_streams([source], np.arange(6) / 10, [])


def test_bad_first_joint_message_does_not_hide_later_readings(tmp_path):
    source = tmp_path / 'joints.mcap'
    names = [f'joint{j}_rad' for j in range(6)] + ['gripper_m']
    sensor_file(source, [(0, b'{broken')] + [(i / 10, {'name': names, 'position': [float(i)] * 7})
                                          for i in range(1, 4)])
    streams = formats.mcap_joint_streams([source])
    np.testing.assert_allclose(streams['/sensor']['t'], [.1, .2, .3])
    np.testing.assert_array_equal(streams['/sensor']['pos'][:, 0], [1, 2, 3])


def test_zero_ros_signal_reports_arrival_fallback_and_withholds_precise_alignment(tmp_path):
    source = tmp_path / 'uninitialized_sensor.mcap'
    sensor_file(source, [(1 + i / 10, {'header': {'stamp': {'sec': 0, 'nanosec': 0}}, 'pressure': value})
                         for i, value in enumerate([2, 9, 3])], topic='/touch_pressure')
    signals = formats.mcap_signals([source], np.array([1, 1.1, 1.2]), capture_clock=True)
    np.testing.assert_array_equal(signals['/touch_pressure'][:, 0], [2, 9, 3])
    assert signals.meta['/touch_pressure']['clock_source'] == 'arrival'
    assert 'uninitialized' in signals.meta['/touch_pressure']['clock_problem']
    assert any(i['kind'] == 'capture_timestamp_invalid' and i['signal'] == '/touch_pressure' for i in signals.issues)
    from label import sensor_evidence
    context = {'fps': 10, 'state_kind': 'none', 'n_state_frames': 3}
    formats.write_signals(tmp_path, context, signals)
    evidence = sensor_evidence.build({'context': context, 'sources': {'exo': {}}, 'signals': dict(signals),
        'signal_meta': signals.meta, 'times': {'exo': np.arange(3) / 10}, 'state': np.zeros((3, 0))}, {'n': 3})
    assert evidence['sensors'][0]['episode_alignment_verified'] is False
    bound = sensor_evidence.bind(evidence, [{'start_s': 0., 'end_s': .2, 'headline': 'Pressure changes',
        'claim_type': 'response_change', 'evidence': [{'sensor_id': evidence['sensors'][0]['id'],
        'series_id': evidence['series'][0]['id'], 'time_s': [0., .2]}]}])
    assert not bound['findings'] and bound['unbound_findings']
