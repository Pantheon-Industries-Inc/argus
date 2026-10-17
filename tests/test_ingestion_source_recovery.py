"""Source boundary ownership and channel failures through real conversion."""
import base64
import io
import json

import av
import numpy as np
import pandas as pd
import pytest
from mcap.writer import Writer
from PIL import Image

from label import episode
from prepare import formats
from test_formats import _lerobot_v3
from test_ingestion_timing_regressions import reordered_h264
from test_prepare import _mp4, joint_names


def test_packed_probe_keeps_the_existing_large_rounded_offset_contract(tmp_path):
    source = tmp_path / 'packed.mp4'
    pts = _mp4(source, 6573)
    # This actual publisher offset is documented by 53f4fd4 and accepted by label.frames.base_frame.
    facts = formats.packed_video_window(source, 218.933334, round(6571 / 30, 6))
    np.testing.assert_array_equal(facts['pts'], pts[6568:6571])


def test_cfr_ownership_survives_ratio_and_accumulated_publisher_offsets(tmp_path):
    source = tmp_path / 'packed.mp4'
    pts = _mp4(source, 6600)
    windows = [(index / 30, (index + 1) / 30, index, index + 1) for index in range(1, 600)]
    offset, first = 0.0, 0
    for length in np.random.default_rng(19).integers(1, 11, size=1000):
        stop = offset + round(int(length) / 30, 6)
        windows.append((offset, stop, first, first + int(length)))
        offset, first = stop, first + int(length)
    errors = []
    for start, stop, first, end in windows:
        owned = formats.packed_video_window(source, start, stop, fps=30)['pts']
        if not np.array_equal(owned, pts[first:end]):
            errors.append((start, stop, first, end, owned[:2].tolist()))
    assert not errors, errors[:5]


def test_off_grid_cfr_boundary_keeps_its_recorded_precision(tmp_path):
    source = tmp_path / 'precise_boundary.mp4'
    _mp4(source, 10)
    facts = formats.packed_video_window(source, .068, .168, fps=30)
    np.testing.assert_array_equal(facts['pts'], np.arange(3, 6) * 512)


@pytest.mark.parametrize('start,stop,expected', [(.100001, .100002, [100001]),
                                                (.1000014, .1000023, [100002])])
def test_packed_vfr_precision_does_not_claim_an_earlier_frame(tmp_path, start, stop, expected):
    source = tmp_path / 'precise.mp4'
    writer = formats.FrameWriter(source, '', fine_clock=True)
    for index, stamp in enumerate([0, .1, .100001, .100002, .2]):
        writer.add_image(stamp, Image.new('RGB', (64, 48), (index * 40,) * 3))
    assert writer.close() == 5
    facts = formats.packed_video_window(source, start, stop, fps=30)
    np.testing.assert_array_equal(facts['pts'], expected)


@pytest.mark.parametrize('first,clock', [(1, 'microseconds'), (2, 'microseconds'), (586, 'ratio')])
def test_rounded_v3_windows_keep_exact_pictures_with_their_state_rows(tmp_path, first, clock):
    root = tmp_path / 'upload'
    front, rear, depth = [f'observation.images.{key}' for key in ('front', 'rear_mask', 'front_depth')]
    owned = np.arange(first, first + 6)
    total = max(10, first + 9)
    boundary = lambda frame: round(frame / 30, 6) if clock == 'microseconds' else frame / 30
    state = np.zeros((6, 7))
    state[:, 0] = owned
    _lerobot_v3(root, {front: total, rear: total}, {
        'episode_index': np.repeat([0, 1], 3), 'frame_index': np.tile(np.arange(3), 2),
        'timestamp': np.tile(np.arange(3) / 30, 2), 'observation.state': list(state)})
    info_path = root / 'meta/info.json'
    info = json.loads(info_path.read_text())
    info['features']['observation.state'].update(shape=[7], names=joint_names((None,)))
    info['features'][depth] = {'dtype': 'video', 'shape': [36, 64, 1],
                               'info': {'video.is_depth_map': True}}
    info_path.write_text(json.dumps(info))
    for key in (front, rear):
        _mp4(root / 'videos' / key / 'chunk-000/file-000.mp4', total)
    # PNG in QuickTime retains grayscale pixels and the exact CFR clock.
    depth_path = root / 'videos' / depth / 'chunk-000/file-000.mp4'
    depth_path.parent.mkdir(parents=True)
    with av.open(str(depth_path), 'w', format='mov') as container:
        stream = container.add_stream('png', rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 36, 'gray16be'
        for index in range(total):
            frame = av.VideoFrame.from_ndarray(np.full((36, 64), index, np.uint16), format='gray16be')
            frame.pts = index
            from fractions import Fraction
            frame.time_base = Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    rows = []
    for index in (0, 1):
        row = {'episode_index': index, 'length': 3, 'data/chunk_index': 0, 'data/file_index': 0,
               'dataset_from_index': index * 3, 'dataset_to_index': (index + 1) * 3}
        for key in (front, rear, depth):
            row.update({f'videos/{key}/chunk_index': 0, f'videos/{key}/file_index': 0,
                        f'videos/{key}/from_timestamp': boundary(first + index * 3),
                        f'videos/{key}/to_timestamp': boundary(first + (index + 1) * 3)})
        rows.append(row)
    metadata = root / 'meta/episodes/chunk-000/file-000.parquet'
    metadata.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(metadata)
    report = formats.convert(root, 'teleop_arms', tmp_path / 'out', 'rounded ownership', 900)
    assert not report['failed'], report
    assert len(report['episodes']) == 2
    with av.open(str(root / 'videos' / front / 'chunk-000/file-000.mp4')) as container:
        reference = list(container.decode(video=0))
    for index, result in enumerate(report['episodes']):
        ep = tmp_path / 'out' / result['episode_id']
        loaded = episode.load(ep)
        own = owned[index * 3:(index + 1) * 3]
        assert loaded['context']['state_kind'] == 'joints'
        np.testing.assert_array_equal(loaded['state'][:, 0], own)
        np.testing.assert_allclose([episode.frame_time(loaded, k) for k in range(3)], np.arange(3) / 30)
        pictures = episode._decode_view(loaded, 'exo', [0, 1, 2])
        for k, source_index in enumerate(own):
            np.testing.assert_array_equal(np.asarray(pictures[k]), reference[source_index].to_ndarray(format='rgb24'))
        assert not any(i['kind'] == 'recorded_rate_mismatch' for i in loaded['context'].get('reader_issues', []))
        unshown = next(c for c in loaded['context']['unshown_cameras'] if c['name'] == rear)
        with np.load(ep / unshown['camera_times']) as clock:
            np.testing.assert_array_equal(clock['pts'], own * 512)
            np.testing.assert_allclose(clock['capture'], np.arange(3) / 30)
        with np.load(ep / 'depth_times.npz') as clock:
            selected = clock['depth_exo_pts']
        with av.open(str(depth_path)) as container:
            depth_frames = {f.pts: f.to_ndarray() for f in container.decode(video=0)}
        assert [depth_frames[int(pts)][0, 0] for pts in selected] == list(own)


def mixed_mcap(path, *, bad_schema=False, bad_video=None, bad_primary=False):
    picture = io.BytesIO()
    Image.new('RGB', (64, 48), (120,) * 3).save(picture, format='JPEG')
    with path.open('wb') as stream:
        writer = Writer(stream)
        writer.start()
        camera_schema = writer.register_schema('foxglove.CompressedVideo', 'jsonschema', b'{}')
        good = writer.register_channel('/camera/wrist/image' if bad_primary else '/camera/front/image', 'json', camera_schema)
        signals = writer.register_channel('/pressure', 'json', writer.register_schema('Pressure', 'jsonschema', b'{}'))
        joints = writer.register_channel('/joint_states', 'json', writer.register_schema('Joints', 'jsonschema', b'{}'))
        if bad_schema:
            encoding, name, schema_encoding, schema_data = {
                'ros2': ('cdr', 'sensor_msgs/msg/Imu', 'ros2msg', b'not a valid schema'),
                'ros1': ('ros1', 'sensor_msgs/Imu', 'ros1msg', b'not a valid schema'),
                'protobuf': ('protobuf', 'Pressure', 'protobuf', b'not a descriptor'),
            }[bad_schema]
            bad = writer.register_channel('/unreadable', encoding,
                                          writer.register_schema(name, schema_encoding, schema_data))
        if bad_video:
            bad = writer.register_channel('/camera/front/image' if bad_primary else '/camera/wrist/image', 'json', camera_schema)
        for index in range(10):
            stamp = 1_000_000_000 + index * 33_333_333
            body = {'format': 'jpeg', 'data': base64.b64encode(picture.getvalue()).decode()}
            writer.add_message(good, stamp, json.dumps(body).encode(), stamp)
            writer.add_message(signals, stamp, json.dumps({'pressure': index}).encode(), stamp)
            writer.add_message(joints, stamp, json.dumps({'name': joint_names((None,)),
                                                        'position': [float(index)] * 7}).encode(), stamp)
            if bad_schema:
                writer.add_message(bad, stamp, b'bad cdr', stamp)
            if bad_video:
                body = {'format': 'h264', 'data': base64.b64encode(bad_video[index]).decode()}
                writer.add_message(bad, stamp, json.dumps(body).encode(), stamp)
        writer.finish()


@pytest.mark.parametrize('schema', ['ros1', 'ros2', 'protobuf'])
def test_unreadable_schema_keeps_other_channels_through_conversion(tmp_path, schema):
    source = tmp_path / 'mixed.mcap'
    mixed_mcap(source, bad_schema=schema)
    report = formats.convert(source, 'ego_head', tmp_path / 'out', 'channel recovery', 900)
    assert not report['failed'], report
    ep = tmp_path / 'out' / report['episodes'][0]['episode_id']
    loaded = episode.load(ep)
    assert loaded['sources']['exo']['n_frames'] == 10
    np.testing.assert_array_equal(loaded['signals']['/pressure'][:, 0], np.arange(10))
    assert any(i['kind'] == 'mcap_schema_unreadable' and i['signal'] == '/unreadable'
               for i in loaded['context']['reader_issues'])
    streams = formats.mcap_joint_streams([source])
    np.testing.assert_array_equal(streams['/joint_states']['pos'][:, 0], np.arange(10))


def test_schema_factory_programming_error_propagates(tmp_path, monkeypatch):
    from mcap_ros2.decoder import DecoderFactory
    source = tmp_path / 'mixed.mcap'
    mixed_mcap(source, bad_schema='ros2')
    def broken_factory(self, encoding, schema):
        raise RuntimeError('schema factory regression')
    monkeypatch.setattr(DecoderFactory, 'decoder_for', broken_factory)
    with pytest.raises(RuntimeError, match='schema factory regression'):
        formats.mcap_signals([source], np.arange(10) / 30 + 1)


def test_unknown_decoder_factory_is_not_classified_as_malformed_source():
    class OtherFactory:
        def decoder_for(self, encoding, schema):
            raise ValueError('decoder integration regression')
    with pytest.raises(ValueError, match='decoder integration regression'):
        formats._decoder_for('cdr', None, [OtherFactory()])


def test_camera_finalization_programming_error_propagates(tmp_path, monkeypatch):
    source = tmp_path / 'good.mcap'
    mixed_mcap(source)
    def broken_close(self):
        raise RuntimeError('camera finalization regression')
    monkeypatch.setattr(formats.FrameWriter, 'close', broken_close)
    with pytest.raises(RuntimeError, match='camera finalization regression'):
        formats.convert_mcap_generic({'file': source, 'name': 'good', 'seconds': 1, 'topics': []},
                                     'ego_head', tmp_path / 'out', 'programming error')


@pytest.mark.parametrize('rig', ['ego_head', 'teleop_arms'])
@pytest.mark.parametrize('bad_primary', [False, True])
def test_reordered_camera_keeps_readable_camera_and_reports_rejected_stream(tmp_path, reordered_h264, rig, bad_primary):
    source = tmp_path / 'mixed.mcap'
    mixed_mcap(source, bad_video=reordered_h264, bad_primary=bad_primary)
    report = formats.convert(source, rig, tmp_path / 'out', 'camera recovery', 900)
    assert not report['failed'], report
    ep = tmp_path / 'out' / report['episodes'][0]['episode_id']
    loaded = episode.load(ep)
    assert len(loaded['sources']) == 1
    assert next(iter(loaded['sources'].values()))['n_frames'] == 10
    bad_topic = '/camera/front/image' if bad_primary else '/camera/wrist/image'
    assert any(i['kind'] == 'camera_not_decodable' and i['camera'] == bad_topic
               and 'decode timestamps' in i['what'] for i in loaded['context']['reader_issues'])
    pictures = episode._decode_view(loaded, next(iter(loaded['sources'])), [0, 9])
    assert [np.asarray(pictures[k]).mean() for k in (0, 9)] == pytest.approx([120, 120], abs=3)


@pytest.mark.parametrize('reject_front', [False, True])
def test_camera_fallback_keeps_depth_source_ownership(tmp_path, reordered_h264, reject_front):
    from checks import sensors
    from label import depth as dp
    source = tmp_path / 'depth.mcap'
    def jpeg(value):
        data = io.BytesIO()
        Image.new('RGB', (64, 48), (value,) * 3).save(data, format='JPEG')
        return {'format': 'jpeg', 'data': base64.b64encode(data.getvalue()).decode()}
    depth_pixels = np.full((48, 64), 500, dtype='<u2')
    with source.open('wb') as stream:
        writer = Writer(stream)
        writer.start()
        schema = writer.register_schema('foxglove.CompressedVideo', 'jsonschema', b'{}')
        front = writer.register_channel('/camera/front/image', 'json', schema)
        wrist = writer.register_channel('/camera/wrist/image', 'json', schema)
        depth = writer.register_channel('/camera/front/depth', 'json',
            writer.register_schema('sensor_msgs/msg/Image', 'jsonschema', b'{}'))
        for index in range(10):
            stamp = 1_000_000_000 + index * 33_333_333
            front_image = ({'format': 'h264', 'data': base64.b64encode(reordered_h264[index]).decode()}
                           if reject_front else jpeg(80))
            for channel, value in [(front, front_image), (wrist, jpeg(20 + index * 20)),
                    (depth, {'encoding': '16UC1', 'width': 64, 'height': 48,
                             'data': base64.b64encode(depth_pixels.tobytes()).decode()})]:
                writer.add_message(channel, stamp, json.dumps(value).encode(), stamp)
        writer.finish()
    out = tmp_path / 'out'
    report = formats.convert(source, 'ego_head', out, 'depth ownership', 900)
    assert not report['failed'], report
    prepared = out / report['episodes'][0]['episode_id']
    loaded = episode.load(prepared)
    assert loaded['context']['cameras']['exo']['key'] == ('/camera/wrist/image' if reject_front else '/camera/front/image')
    view, descriptor = next(iter(loaded['depth'].items()))
    assert descriptor['source'] == '/camera/front/depth'
    assert descriptor['paired_camera'] == (None if reject_front else 'exo')
    got = dp.decode(descriptor, descriptor['pts'], [0, 9])
    for frame in got.values():
        np.testing.assert_array_equal(frame, depth_pixels)
    request = episode.build_request(prepared)
    assert ('head records depth as well as colour' in request['prompt']) is not reject_front
    assert not any(row['check'] == 'depth_frozen' for row in sensors.depth_findings(loaded))
    assessed = sensors.run_episode(prepared)
    assert all(row['status'] != 'errored' for row in assessed['checks'])
    statuses = {row['check']: row for row in assessed['checks']}
    assert statuses['depth_invalid']['status'] == 'clear'
    for check in ('depth_frozen', 'depth_offset'):
        assert statuses[check]['status'] == ('na' if reject_front else 'clear')
        if reject_front:
            assert 'paired colour camera' in statuses[check]['why']


def test_unreadable_camera_schema_reports_source_schema_failure(tmp_path):
    source = tmp_path / 'bad_schema.mcap'
    with source.open('wb') as stream:
        writer = Writer(stream)
        writer.start()
        schema = writer.register_schema('sensor_msgs/msg/Image', 'ros2msg', b'not a valid schema')
        camera = writer.register_channel('/camera/front/image', 'cdr', schema)
        writer.add_message(camera, 1_000_000_000, b'bad cdr', 1_000_000_000)
        writer.finish()
    report = formats.convert(source, 'ego_head', tmp_path / 'out', 'schema failure', 900)
    assert not report['episodes'] and len(report['failed']) == 1
    reason = report['failed'][0]['why']
    assert 'schema could not be read' in reason and '/camera/front/image' in reason
    assert 'which we do not decode' not in reason
