"""Recovery retains source-row ownership through the evidence consumers."""
import io
import json
from fractions import Fraction

import av
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from label import depth, episode
from prepare import formats
from test_ingestion_v5 import lerobot_source


@pytest.mark.parametrize('representation', ['packed', 'embedded'])
def test_recovered_mixed_camera_depth_uses_the_surviving_publisher_rows(tmp_path, representation):
    root = tmp_path / 'source'
    paths, _ = lerobot_source(root, length=6, offset=0, mixed=True)
    table = root / 'data/chunk-000/file-000.parquet'
    df = pd.read_parquet(table)
    df['observation.images.left_wrist'] = [None] * 6
    key = 'observation.images.front_depth'
    info_path = root / 'meta/info.json'
    info = json.loads(info_path.read_text())
    info['features'][key] = {'dtype': 'video' if representation == 'packed' else 'image',
                             'shape': [72, 96, 1], 'sensor_type': 'depth', 'info': {'video.is_depth_map': True}}
    info_path.write_text(json.dumps(info))
    if representation == 'embedded':
        cells = []
        for row in range(6):
            image = io.BytesIO()
            Image.fromarray(np.full((72, 96), 1000 + row, np.uint16)).save(image, 'PNG')
            cells.append({'bytes': image.getvalue(), 'path': None})
        df[key] = cells
    else:
        meta = root / 'meta/episodes/chunk-000/file-000.parquet'
        metadata = pd.read_parquet(meta)
        for suffix, value in [('chunk_index', 0), ('file_index', 0), ('from_timestamp', 0.), ('to_timestamp', .2)]:
            metadata[f'videos/{key}/{suffix}'] = value
        metadata.to_parquet(meta)
        path = root / 'videos' / key / 'chunk-000/file-000.mp4'
        path.parent.mkdir(parents=True)
        with av.open(str(path), 'w', format='mov') as container:
            stream = container.add_stream('png', rate=30)
            stream.width, stream.height, stream.pix_fmt = 96, 72, 'gray16be'
            for row in range(3, 6):
                frame = av.VideoFrame.from_ndarray(np.full((72, 96), 1000 + row, np.uint16), format='gray16be')
                frame.pts, frame.time_base = row, Fraction(1, 30)
                for packet in stream.encode(frame): container.mux(packet)
            for packet in stream.encode(): container.mux(packet)
    df.to_parquet(table)
    writer = formats.FrameWriter(paths['observation.images.front'], '', fine_clock=True)
    for row in range(3, 6):
        writer.add_image(row / 30, Image.new('RGB', (96, 72), (row * 30,) * 3))
    assert writer.close() == 3
    report = formats.convert(root, 'teleop_arms', tmp_path / 'out', 'recovered depth', 900)
    assert not report['failed'] and len(report['episodes']) == 1, report
    loaded = episode.load(tmp_path / 'out' / report['episodes'][0]['episode_id'])
    np.testing.assert_array_equal(loaded['state'][:, 0], [3, 4, 5])
    np.testing.assert_allclose(loaded['times']['exo'], [0., 1 / 30, 2 / 30], atol=1e-6)
    retained = loaded['depth'] | loaded['additional_depth']
    view = next(iter(retained))
    values = depth.at_anchor(loaded, retained, view, [0, 1, 2])
    assert [int(values[k][0, 0]) if k in values else None for k in range(3)] == [1003, 1004, 1005]
    np.testing.assert_allclose(retained[view]['t'][retained[view]['km']],
                               [0., 1 / 30, 2 / 30], atol=1e-6)


@pytest.mark.parametrize('packed', [False, True])
def test_video_only_camera_policy_retries_healthy_unused_candidates(tmp_path, packed):
    root = tmp_path / 'source'
    paths, _ = lerobot_source(root, length=6, offset=0)
    paths['observation.images.front'].write_bytes(b'not a video')
    if not packed:
        info = json.loads((root / 'meta/info.json').read_text())
        info['codebase_version'] = 'v2.1'
        (root / 'meta/info.json').write_text(json.dumps(info))
        for key, path in paths.items():
            target = root / 'videos/chunk-000' / key / 'episode_000000.mp4'
            target.parent.mkdir(parents=True)
            path.rename(target)
        (root / 'meta/episodes.jsonl').write_text(json.dumps({'episode_index': 0, 'length': 6}) + '\n')
    report = formats.convert(root, 'ego_head', tmp_path / 'out', 'camera fallback', 900)
    assert not report['failed'] and len(report['episodes']) == 1, report
    loaded = episode.load(tmp_path / 'out' / report['episodes'][0]['episode_id'])
    assert loaded['sources']['exo']['camera_key'] == 'observation.images.left_wrist'
    assert len(episode._decode_view(loaded, 'exo', list(range(6)))) == 6
    assert any(issue['kind'] == 'camera_not_decodable' and issue['camera'] == 'observation.images.front'
               for issue in loaded['context']['reader_issues'])


@pytest.mark.parametrize('clock_error', [0., .005])
def test_partial_packed_float32_clock_keeps_precision_without_accepting_a_conflict(tmp_path, clock_error):
    root = tmp_path / 'source'
    paths, _ = lerobot_source(root, length=1200, offset=0)
    for path in paths.values():
        writer = formats.FrameWriter(path, '', fine_clock=True)
        for row in range(1197, 1200):
            writer.add_image(row / 30, Image.new('RGB', (96, 72), (row % 240,) * 3))
        assert writer.close() == 3
    table = root / 'data/chunk-000/file-000.parquet'
    df = pd.read_parquet(table)
    df['timestamp'] = df['timestamp'].astype(np.float32)
    df.loc[1190, 'timestamp'] += clock_error
    df.to_parquet(table)
    report = formats.convert(root, 'teleop_arms', tmp_path / 'out', 'float32 row clock', 900)
    assert not report['failed'] and len(report['episodes']) == 1, report
    loaded = episode.load(tmp_path / 'out' / report['episodes'][0]['episode_id'])
    if clock_error:
        assert loaded['context']['state_kind'] == 'none'
    else:
        assert loaded['context']['state_kind'] == 'joints'
        np.testing.assert_array_equal(loaded['state'][:, 0], [1197, 1198, 1199])


@pytest.mark.parametrize('damage', ['crc', 'directory_cut'])
@pytest.mark.parametrize('compression', [0, 8])
def test_damaged_zip_retains_decodable_member_bytes_and_records_damage(tmp_path, damage, compression):
    import struct
    import zipfile
    from test_prepare import _mp4
    video = tmp_path / 'source.mp4'
    _mp4(video, 6)
    archive = tmp_path / 'recording.zip'
    with zipfile.ZipFile(archive, 'w', compression) as writer:
        writer.write(video, 'front.mp4')
    data = bytearray(archive.read_bytes())
    central = data.index(b'PK\x01\x02')
    if damage == 'crc':
        struct.pack_into('<I', data, central + 16, 123)
    else:
        data = data[:central]
    archive.write_bytes(data)
    report = formats.convert(archive, 'ego_head', tmp_path / 'out', 'damaged archive', 900)
    assert not report['failed'] and len(report['episodes']) == 1, report
    loaded = episode.load(tmp_path / 'out' / report['episodes'][0]['episode_id'])
    assert len(episode._decode_view(loaded, 'exo', list(range(6)))) == 6
    assert any(issue['kind'] == 'archive_damaged' for issue in loaded['context']['reader_issues'])
    assert report['missing']


def test_hdf5_damaged_camera_chunk_preserves_other_rows_cameras_and_state(tmp_path):
    import h5py
    from test_prepare import joint_names
    source = tmp_path / 'recording.hdf5'
    with h5py.File(source, 'w') as container:
        for name, value in [('front', 40), ('left_wrist', 160)]:
            ds = container.create_dataset('observations/images/' + name,
                data=np.full((6, 72, 96, 3), value, np.uint8), chunks=(1, 72, 96, 3), fletcher32=True)
            if name == 'front':
                offset = ds.id.get_chunk_info_by_coord((2, 0, 0, 0)).byte_offset
        state = container.create_dataset('observations/qpos', data=np.arange(6)[:, None] * np.ones((6, 7)))
        state.attrs['names'] = joint_names((None,))
        container.create_dataset('observations/timestamp', data=np.arange(6) / 30)
    data = bytearray(source.read_bytes())
    data[offset + 50] ^= 128
    source.write_bytes(data)
    with h5py.File(source, 'r') as container:
        with pytest.raises(OSError): container['observations/images/front'][2]
        assert container['observations/images/left_wrist'][2, 0, 0, 0] == 160
    report = formats.convert(source, 'teleop_arms', tmp_path / 'out', 'damaged HDF camera', 900)
    assert not report['failed'] and len(report['episodes']) == 1, report
    loaded = episode.load(tmp_path / 'out' / report['episodes'][0]['episode_id'])
    np.testing.assert_array_equal(loaded['state'][:, 0], np.arange(6))
    assert len(episode._decode_view(loaded, 'left', list(range(6)))) == 6
    assert 2 not in episode._decode_view(loaded, 'exo', list(range(6)))
    assert any(issue['kind'] == 'frames_not_decodable' and 'front' in issue['camera']
               for issue in loaded['context']['reader_issues'])


def rewrite_owned_camera(source, destination, mode):
    import base64
    from mcap.writer import Writer
    from test_prepare import _h264_frames
    pictures = _h264_frames(6)
    with destination.open('wb') as target:
        writer = Writer(target); writer.start()
        schema = writer.register_schema('foxglove.CompressedVideo', 'jsonschema', b'{}')
        channel = writer.register_channel('/camera/front/video', 'json', schema)
        for row, picture in enumerate(pictures):
            stamp = 1_790_000_000_000_000_000 + round(row / 30 * 1e9)
            if row == 0 and mode == 'parameters':
                import re
                starts = [match.start() - int(match.start() > 0 and picture[match.start() - 1] == 0)
                          for match in re.finditer(b'\x00\x00\x01', picture)]
                units = [picture[a:b] for a, b in zip(starts, starts[1:] + [len(picture)])]
                parameters = [unit for unit in units if next(formats.nal_units(unit)) & 31 in (7, 8)]
                picture = b''.join(unit for unit in units if next(formats.nal_units(unit)) & 31 not in (7, 8))
                assert len(parameters) == 2
                for index, parameter in enumerate(parameters):
                    data = json.dumps({'format': 'h264', 'data': base64.b64encode(parameter).decode()}).encode()
                    writer.add_message(channel, stamp - 2 + index, data, stamp - 2 + index)
            body = json.dumps({'format': 'h264', 'data': base64.b64encode(picture).decode()}).encode()
            writer.add_message(channel, stamp, body, stamp)
            if mode == 'duplicate':
                writer.add_message(channel, stamp, body, stamp)
            if row == 2 and mode in ('stray', 'sps'):
                bad = b'hello world' if mode == 'stray' else b'\x00\x00\x00\x01\x09\x10\x00\x00\x00\x01\x67\x64\x00\x1e\xac'
                extra = json.dumps({'format': 'h264', 'data': base64.b64encode(bad).decode()}).encode()
                writer.add_message(channel, stamp + 1, extra, stamp + 1)
        writer.finish()


@pytest.mark.parametrize('mode', ['stray', 'sps', 'duplicate', 'parameters'])
def test_mcap_non_picture_or_duplicate_messages_do_not_retime_or_drop_real_pictures(tmp_path, mode):
    source = tmp_path / 'recording.mcap'
    rewrite_owned_camera(None, source, mode)
    report = formats.convert(source, 'ego_head', tmp_path / 'out', 'camera packet recovery', 900)
    assert not report['failed'] and len(report['episodes']) == 1, report
    loaded = episode.load(tmp_path / 'out' / report['episodes'][0]['episode_id'])
    assert loaded['sources']['exo']['n_frames'] == 6
    np.testing.assert_allclose([episode.frame_time(loaded, k) for k in range(6)], np.arange(6) / 30, atol=1e-6)
    assert len(episode._decode_view(loaded, 'exo', list(range(6)))) == 6
    assert loaded['context'].get('reader_issues')


def test_declared_mcap_capture_text_stays_explicit_about_unverified_physical_timing(tmp_path):
    from test_ingestion_v5 import text_recording
    source = tmp_path / 'recording.mcap'
    text_recording(source, early=1., late=3.5, clock_id='fixture', text_offset=-.05)
    report = formats.convert(source, 'ego_head', tmp_path / 'out', 'declared text clock', 900)
    assert not report['failed'], report
    context = json.loads((tmp_path / 'out' / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    assert any(issue['kind'] == 'publisher_text_timing_unverified' for issue in context['reader_issues'])
    assert context['source']['text_clock_alignment']['physical_synchronization_verified'] is False


def test_one_bad_parquet_row_group_preserves_other_episode_state_and_reports_damage(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = tmp_path / 'source'
    lerobot_source(root, length=6, episodes=2, offset=0)
    path = root / 'data/chunk-000/file-000.parquet'
    table = pq.read_table(path)
    pq.write_table(table, path, row_group_size=6)
    metadata = pq.ParquetFile(path).metadata
    column = next(metadata.row_group(0).column(k) for k in range(metadata.row_group(0).num_columns)
                  if metadata.row_group(0).column(k).path_in_schema.startswith('observation.state'))
    start = column.dictionary_page_offset or column.data_page_offset
    data = bytearray(path.read_bytes())
    data[start + 8:start + column.total_compressed_size] = b'\xff' * (column.total_compressed_size - 8)
    path.write_bytes(data)
    with pytest.raises((pa.ArrowException, OSError)):
        pq.ParquetFile(path).read_row_group(0)
    assert pq.ParquetFile(path).read_row_group(1).num_rows == 6
    report = formats.convert(root, 'teleop_arms', tmp_path / 'out', 'row group recovery', 900)
    assert not report['failed'] and len(report['episodes']) == 2, report
    loaded = episode.load(tmp_path / 'out' / report['episodes'][1]['episode_id'])
    np.testing.assert_array_equal(loaded['state'][:, 0], [6, 7, 8, 9, 10, 11])
    assert any(issue['kind'] == 'table_row_group_unreadable' for issue in loaded['context']['reader_issues'])


def test_invalid_optional_pandas_metadata_keeps_arrow_rows_and_source_damage(tmp_path):
    import pyarrow.parquet as pq
    root = tmp_path / 'source'
    lerobot_source(root, length=6, offset=0)
    path = root / 'data/chunk-000/file-000.parquet'
    table = pq.read_table(path)
    metadata = dict(table.schema.metadata or {})
    metadata[b'pandas'] = b'{"columns": ['
    pq.write_table(table.replace_schema_metadata(metadata), path)
    with pytest.raises(json.JSONDecodeError):
        pq.ParquetFile(path).read().to_pandas()
    report = formats.convert(root, 'teleop_arms', tmp_path / 'out', 'optional metadata damage', 900)
    assert not report['failed'] and len(report['episodes']) == 1, report
    loaded = episode.load(tmp_path / 'out' / report['episodes'][0]['episode_id'])
    np.testing.assert_array_equal(loaded['state'][:, 0], np.arange(6))
    assert len(episode._decode_view(loaded, 'exo', list(range(6)))) == 6
    assert any(issue['kind'] == 'table_metadata_unreadable' for issue in loaded['context']['reader_issues'])


@pytest.mark.parametrize('fps', [float('nan'), -30, 'nonsense'])
def test_unusable_declared_frame_rate_does_not_discard_readable_cameras(tmp_path, fps):
    root = tmp_path / 'source'
    lerobot_source(root, length=6, offset=0)
    path = root / 'meta/info.json'
    info = json.loads(path.read_text()); info['fps'] = fps; path.write_text(json.dumps(info))
    report = formats.convert(root, 'teleop_arms', tmp_path / 'out', 'invalid nominal rate', 900)
    assert not report['failed'] and len(report['episodes']) == 1, report
    loaded = episode.load(tmp_path / 'out' / report['episodes'][0]['episode_id'])
    assert len(episode._decode_view(loaded, 'exo', list(range(6)))) == 6
    assert any('fps' in issue['what'] for issue in loaded['context']['reader_issues'])


def test_mcap_corrupt_picture_quarantines_reference_frames_until_next_keyframe(tmp_path):
    # Generated by the supplied timing harness, TRUNC_IDX=50 build_ego4.py, GOP=15 with no B-frames.
    from pathlib import Path
    source = Path(__file__).parent / 'fixtures/corrupt_h264_reference.mcap'
    report = formats.convert(source, 'ego_head', tmp_path / 'out', 'corrupt reference recovery', 900)
    assert not report['failed'] and len(report['episodes']) == 1, report
    loaded = episode.load(tmp_path / 'out' / report['episodes'][0]['episode_id'])
    times = np.array([episode.frame_time(loaded, row) for row in range(loaded['sources']['exo']['n_frames'])])
    np.testing.assert_allclose(times, np.r_[np.arange(50), np.arange(60, 120)] / 30, atol=1e-6)
    assert len(episode._decode_view(loaded, 'exo', list(range(110)))) == 110
    assert any(issue['kind'] == 'frames_not_decodable' for issue in loaded['context']['reader_issues'])


@pytest.mark.parametrize('rows,damaged', [([51], True), ([49, 51], True), ([49, 60, 61], False)])
def test_file_decoder_checks_reference_damage_before_requested_frame(rows, damaged):
    from pathlib import Path
    from label.frames import DamagedFrame, extract_frames
    source = Path(__file__).parent / 'fixtures/corrupt_h264_reference.mp4'
    pts = formats.probe(source)['pts']
    if damaged:
        with pytest.raises(DamagedFrame):
            extract_frames(source, 0., 120, rows, pts=pts)
    else:
        assert set(extract_frames(source, 0., 120, rows, pts=pts)) == set(rows)


@pytest.mark.parametrize('proven', [False, True])
def test_mcap_presentation_uses_one_recorded_axis_for_camera_and_text(tmp_path, proven):
    from test_ingestion_v5 import text_recording
    source = tmp_path / 'recording.mcap'
    text_recording(source, early=1., late=3.5, camera_delay=.1, text_delay=.03, publish_clock=proven)
    report = formats.convert(source, 'ego_head', tmp_path / 'out', 'recorded presentation axis', 900)
    assert not report['failed'], report
    loaded = episode.load(tmp_path / 'out' / report['episodes'][0]['episode_id'])
    context = loaded['context']
    # Header event stamps place the first step at 1.0; recorder arrivals place it at 1.03 - .1 = .93.
    assert [step['t0'] for step in context['annotation_subtasks'] if step.get('column') != '/task'] == (
        [1., 2.5, 3.5] if proven else [.93, 2.43, 3.43])
    assert context['source']['presentation_clock']['basis'] == ('capture' if proven else 'mcap_log_time')
    assert context['source']['presentation_clock']['physical_synchronization_verified'] is False
    if not proven:
        assert 'arrival' in context['collection_note']
        assert 'arrival' in episode.build_request(loaded['dir'])['prompt']
    with np.load(loaded['dir'] / context['recorded_camera_clocks']) as clocks:
        assert clocks['exo_capture_ns'][0] == 1_000_000_000_000
        assert clocks['exo_log_ns'][0] == round((1_790_000_000 + .1) * 1e9)


@pytest.mark.parametrize('disconnected', [False, True])
def test_mcap_arrival_axis_places_state_depth_and_text_without_shifting_native_clocks(tmp_path, disconnected):
    import base64
    from mcap.writer import Writer
    from test_prepare import joint_names
    source = tmp_path / 'recording.mcap'
    png = io.BytesIO(); Image.new('RGB', (64, 48), 'white').save(png, 'PNG')
    wall = 1_790_000_000_000_000_000
    def header(seconds):
        ns = round(seconds * 1e9)
        return {'stamp': {'sec': ns // 10**9, 'nanosec': ns % 10**9}}
    with source.open('wb') as target:
        writer = Writer(target); writer.start()
        camera = writer.register_channel('/camera/front/image', 'json', writer.register_schema('sensor_msgs/CompressedImage', 'jsonschema', b'{}'))
        wrist = writer.register_channel('/camera/wrist/image', 'json', writer.register_schema('sensor_msgs/CompressedImage', 'jsonschema', b'{}'))
        raster = writer.register_channel('/camera/front/depth', 'json', writer.register_schema('sensor_msgs/Image', 'jsonschema', b'{}'))
        state = writer.register_channel('/joint_states', 'json', writer.register_schema('sensor_msgs/JointState', 'jsonschema', b'{}'),
            {'log_clock_id': 'disconnected-state'} if disconnected else {})
        text = writer.register_channel('/task/subtask', 'json', writer.register_schema('TaskText', 'jsonschema', b'{}'))
        for row in range(30):
            log = wall + 100_000_000 + round(row / 30 * 1e9)
            records = [(camera, log, {'format': 'png', 'data': base64.b64encode(png.getvalue()).decode(), 'header': header(1000 + row / 30)}),
                (state, log, {'name': joint_names((None,)), 'position': [row, 0, 0, 0, 0, 0, 0], 'header': header(2000 + row / 30)}),
                (raster, log + 100_000_000, {'encoding': '16UC1', 'width': 64, 'height': 48,
                   'data': base64.b64encode(np.full((48, 64), 1000 + row, '<u2').tobytes()).decode(), 'header': header(3000 + row / 30)})]
            records.append((wrist, log, records[0][2]))
            for channel, stamp, body in records:
                writer.add_message(channel, stamp, json.dumps(body).encode(), stamp)
        for at, label in [(.43, 'pick'), (.8, 'place')]:
            stamp = wall + round(at * 1e9)
            writer.add_message(text, stamp, json.dumps({'data': label, 'header': header(4000 + at)}).encode(), stamp)
        writer.finish()
    report = formats.convert(source, 'teleop_arms', tmp_path / 'out', 'shared recorder presentation', 900)
    assert not report['failed'], report
    loaded = episode.load(tmp_path / 'out' / report['episodes'][0]['episode_id'])
    retained = loaded['depth'] | loaded['additional_depth']
    view = next(iter(retained))
    if disconnected:
        assert loaded['context']['source']['presentation_clock']['basis'] == 'unresolved'
        assert loaded['context']['state_kind'] == 'none'
        assert not loaded.get('signals')
        assert np.all(retained[view]['km'] == -1)
        assert all(np.all(mapping == -1) for mapping in loaded['kmap'].values())
        assert loaded['context']['mcap_field_inventory']
        return
    assert loaded['context']['source']['presentation_clock']['basis'] == 'mcap_log_time'
    np.testing.assert_array_equal(loaded['state'][:, 0], np.arange(30))
    np.testing.assert_allclose(retained[view]['t'], .1 + np.arange(30) / 30, atol=1e-6)
    pixels = depth.at_anchor(loaded, retained, view, [3, 4, 5])
    assert [int(pixels[k][0, 0]) for k in [3, 4, 5]] == [1000, 1001, 1002]
    assert 'arrival' in loaded['context']['collection_note']


@pytest.mark.parametrize('disconnect', ['zero_log', 'different_log_ids'])
def test_mcap_unknown_or_explicitly_disconnected_log_domains_do_not_place_text(tmp_path, disconnect):
    from mcap.reader import make_reader
    from mcap.writer import Writer
    from test_ingestion_v5 import text_recording
    original, source = tmp_path / 'original.mcap', tmp_path / 'recording.mcap'
    text_recording(original, early=1., late=3.5)
    channels = {}
    with original.open('rb') as incoming, source.open('wb') as target:
        writer = Writer(target); writer.start()
        for schema, channel, message in make_reader(incoming).iter_messages():
            if channel.id not in channels:
                sid = writer.register_schema(schema.name, schema.encoding, schema.data)
                metadata = {'log_clock_id': 'camera' if '/camera/' in channel.topic else 'other'} if disconnect == 'different_log_ids' else {}
                channels[channel.id] = writer.register_channel(channel.topic, channel.message_encoding, sid, metadata)
            writer.add_message(channels[channel.id], 0 if disconnect == 'zero_log' else message.log_time,
                               message.data, message.publish_time)
        writer.finish()
    report = formats.convert(source, 'ego_head', tmp_path / 'out', 'disconnected log domains', 900)
    assert not report['failed'], report
    context = json.loads((tmp_path / 'out' / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    assert not context.get('annotation_subtasks')
    assert context['source']['presentation_clock']['basis'] == 'unresolved'
    assert 'pick up' in context['uploader_annotation'] and 'unverified' in context['uploader_annotation']


@pytest.mark.parametrize('malformed_row', [None, 2])
def test_standard_ros2_camera_recovers_cdr_pictures_without_trusting_damaged_schema_clock(tmp_path, malformed_row):
    from pathlib import Path
    from mcap.reader import make_reader
    from mcap.writer import Writer
    fixture = Path(__file__).parent / 'fixtures/corrupt_ros2_image_schema.mcap'
    source = fixture
    if malformed_row is not None:
        import struct
        source = tmp_path / 'recording.mcap'
        channels, schemas, index = {}, {}, 0
        with fixture.open('rb') as incoming, source.open('wb') as target:
            writer = Writer(target); writer.start()
            for schema, channel, message in make_reader(incoming).iter_messages():
                if schema.id not in schemas:
                    schemas[schema.id] = writer.register_schema(schema.name, schema.encoding, schema.data)
                if channel.id not in channels:
                    channels[channel.id] = writer.register_channel(channel.topic, channel.message_encoding, schemas[schema.id])
                payload = message.data
                if channel.topic == '/cam_front/image/compressed':
                    if index == malformed_row:
                        payload = bytearray(payload)
                        struct.pack_into('<I', payload, 12, 0xffffffff)
                        payload = bytes(payload)
                    index += 1
                writer.add_message(channels[channel.id], message.log_time, payload, message.publish_time)
            writer.finish()
    report = formats.convert(source, 'teleop_arms', tmp_path / 'out', 'canonical standard camera recovery', 900)
    assert not report['failed'], report
    loaded = episode.load(tmp_path / 'out' / report['episodes'][0]['episode_id'])
    front = next(view for view, entry in loaded['sources'].items() if entry['camera_key'] == '/cam_front/image/compressed')
    assert len(loaded['sources']) == 2
    assert loaded['sources'][front]['n_frames'] == (60 if malformed_row is None else 59)
    pictures = episode._decode_view(loaded, front, list(range(loaded['sources'][front]['n_frames'])))
    expected = [row for row in range(60) if row != malformed_row]
    for row, picture in pictures.items():
        pixels = np.asarray(picture)
        barcode = sum((1 << bit) for bit in range(12) if pixels[20, bit * 16 + 8].mean() > 128)
        assert barcode == expected[row]
    context = loaded['context']
    assert context['source']['presentation_clock']['basis'] == 'mcap_log_time'
    assert any(issue['kind'] == 'mcap_standard_schema_recovered' for issue in context['reader_issues'])
    assert context['recovered_mcap_schemas']
    if malformed_row is not None:
        assert any(issue['kind'] == 'mcap_message_unreadable' for issue in context['reader_issues'])


@pytest.mark.parametrize('storage', ['mcap', 'hdf5'])
@pytest.mark.parametrize('namespace', ['camera', 'other', None])
@pytest.mark.parametrize('basis', ['arrival', 'capture'])
def test_generic_sidecar_admission_uses_the_selected_clock_namespace_for_all_numeric_fields(tmp_path, storage, namespace, basis):
    from mcap.reader import make_reader
    from mcap.writer import Writer
    from test_ingestion_v5 import text_recording
    from test_prepare import joint_names
    raw = tmp_path / 'raw'; raw.mkdir()
    original = tmp_path / 'original.mcap'
    text_recording(original, clock=basis, early=1., late=3.5, clock_id='camera' if basis == 'capture' else None)
    channels = {}
    with original.open('rb') as incoming, (raw / 'recording.mcap').open('wb') as target:
        writer = Writer(target); writer.start()
        for schema, channel, message in make_reader(incoming).iter_messages():
            if channel.id not in channels:
                sid = writer.register_schema(schema.name, schema.encoding, schema.data)
                channels[channel.id] = writer.register_channel(channel.topic, channel.message_encoding, sid, {('log_clock_id' if basis == 'arrival' else 'capture_clock_id'): 'camera'})
            writer.add_message(channels[channel.id], message.log_time, message.data, message.publish_time)
        writer.finish()
    wall = 1_790_000_000_000_000_000
    if storage == 'mcap':
        with (raw / 'sensors.mcap').open('wb') as target:
            writer = Writer(target); writer.start()
            schema = writer.register_schema('JointAndPressure', 'jsonschema', b'{}')
            channel = writer.register_channel('/joint_states', 'json', schema,
                {('log_clock_id' if basis == 'arrival' else 'capture_clock_id'): namespace} if namespace else {})
            for row in range(121):
                stamp = wall + round(row / 30 * 1e9)
                writer.add_message(channel, stamp, json.dumps({'name': joint_names((None,)),
                    'position': [row, 0, 0, 0, 0, 0, 0], 'pressure': row % 5 + 1000,
                    **({'header': {'stamp': {'sec': 1000 + row // 30, 'nanosec': round((row % 30) / 30 * 1e9)}}} if basis == 'capture' else {})}).encode(), stamp)
            writer.finish()
    else:
        import h5py
        with h5py.File(raw / 'sensors.hdf5', 'w') as container:
            if namespace: container.attrs['log_clock_id' if basis == 'arrival' else 'capture_clock_id'] = namespace
            clock = container.create_dataset('timestamp', data=(wall / 1e9 if basis == 'arrival' else 1000) + np.arange(121) / 30)
            if namespace: clock.attrs['log_clock_id' if basis == 'arrival' else 'capture_clock_id'] = namespace
            state = container.create_dataset('observations/qpos', data=np.arange(121)[:, None] * np.ones((121, 7)))
            state.attrs['names'] = joint_names((None,))
            container.create_dataset('pressure', data=np.arange(121) % 5 + 1000)
    report = formats.convert(raw, 'teleop_arms', tmp_path / 'out', 'sidecar clock namespace', 900)
    assert not report['failed'], report
    loaded = episode.load(tmp_path / 'out' / report['episodes'][0]['episode_id'])
    if namespace == 'camera':
        if storage == 'mcap':
            assert loaded['context']['state_kind'] == 'joints'
            np.testing.assert_allclose(loaded['state'][:, 0], np.arange(121), atol=1e-5)
        assert loaded['signals']
    else:
        assert loaded['context']['state_kind'] == 'none'
        assert not loaded['signals']
        assert 'not placed' in json.dumps(loaded['context']['source']['sensor_files'])
        assert loaded['context']['recorded_sensor_fields']
        from label.evidence_access import Access
        assert any(field['kind'] in ('array', 'metadata') for field in Access(loaded).inventory())


@pytest.mark.parametrize('mixed', [False, True])
def test_camera_policy_requires_an_actual_owned_picture_before_discarding_fallback(tmp_path, mixed):
    from pathlib import Path
    root = tmp_path / 'source'
    paths, _ = lerobot_source(root, length=6, offset=0, mixed=mixed)
    paths['observation.images.front'].write_bytes((Path(__file__).parent / 'fixtures/undecodable_video_packets.mp4').read_bytes())
    assert len(formats.probe(paths['observation.images.front'])['pts']) == 9
    report = formats.convert(root, 'ego_head', tmp_path / 'out', 'packet-only camera fallback', 900)
    assert not report['failed'], report
    loaded = episode.load(tmp_path / 'out' / report['episodes'][0]['episode_id'])
    assert loaded['sources']['exo']['camera_key'] == 'observation.images.left_wrist'
    assert len(episode._decode_view(loaded, 'exo', list(range(6)))) == 6
    assert any(issue['kind'] == 'camera_not_decodable' for issue in loaded['context']['reader_issues'])


def test_inferred_capture_sidecar_cannot_bypass_causality_with_unknown_namespace(tmp_path):
    from mcap.writer import Writer
    from test_ingestion_v5 import text_recording
    from test_prepare import joint_names
    raw = tmp_path / 'raw'; raw.mkdir()
    text_recording(raw / 'recording.mcap', early=1., late=3.5, publish_clock=True)
    with (raw / 'sensors.mcap').open('wb') as target:
        writer = Writer(target); writer.start()
        channel = writer.register_channel('/joint_states', 'json', writer.register_schema('JointState', 'jsonschema', b'{}'))
        for row in range(121):
            stamp = 1000_000_000_000 + round(row / 30 * 1e9)
            body = {'header': {'stamp': {'sec': stamp // 10**9, 'nanosec': stamp % 10**9}},
                    'name': joint_names((None,)), 'position': [row, 0, 0, 0, 0, 0, 0]}
            writer.add_message(channel, stamp - 100_000_000, json.dumps(body).encode(), 0)
        writer.finish()
    report = formats.convert(raw, 'teleop_arms', tmp_path / 'out', 'inferred capture sidecar causality', 900)
    assert not report['failed'], report
    loaded = episode.load(tmp_path / 'out' / report['episodes'][0]['episode_id'])
    assert loaded['context']['source']['presentation_clock']['basis'] == 'capture'
    assert loaded['context']['state_kind'] == 'none'
    assert not loaded['signals']
    assert 'not placed' in json.dumps(loaded['context']['source']['sensor_files'])


def test_log_presentation_keeps_camera_messages_before_another_cameras_capture_origin(tmp_path):
    import base64
    from mcap.writer import Writer
    source = tmp_path / 'recording.mcap'
    wall = 1_790_000_000_000_000_000
    with source.open('wb') as target:
        writer = Writer(target); writer.start()
        schema = writer.register_schema('foxglove.CompressedImage', 'jsonschema', b'{}')
        channels = [writer.register_channel(topic, 'json', schema) for topic in ['/front/image', '/wrist/image']]
        for row in range(18):
            for camera, channel in enumerate(channels):
                log = wall + round(row / 30 * 1e9) + camera * 5_000_000
                capture = log + (400_000_000 if camera == 0 else 0)
                picture = io.BytesIO()
                Image.new('RGB', (96, 72), (row * 10 + camera,) * 3).save(picture, 'PNG')
                body = {'timestamp': {'sec': capture // 10**9, 'nsec': capture % 10**9},
                        'format': 'png', 'data': base64.b64encode(picture.getvalue()).decode()}
                writer.add_message(channel, log, json.dumps(body).encode(), log)
        writer.finish()
    report = formats.convert(source, 'teleop_arms', tmp_path / 'out', 'storage clock ownership', 900)
    assert not report['failed'], report
    loaded = episode.load(tmp_path / 'out' / report['episodes'][0]['episode_id'])
    assert loaded['context']['source']['presentation_clock']['basis'] == 'mcap_log_time'
    for view, descriptor in loaded['sources'].items():
        camera = int('wrist' in descriptor['camera_key'])
        assert descriptor['n_frames'] == 18
        np.testing.assert_allclose(loaded['times'][view], np.arange(18) / 30 + camera * .005, atol=1e-9)
        pictures = episode._decode_view(loaded, view, list(range(18)))
        assert len(pictures) == 18
        for row, picture in pictures.items():
            assert abs(np.asarray(picture).mean() - (row * 10 + camera)) <= 3
    with np.load(loaded['dir'] / loaded['context']['recorded_camera_clocks']) as clocks:
        assert len(clocks['exo_log_ns']) == 18


def test_non_picture_future_stamp_cannot_own_video_storage_clock(tmp_path):
    import base64
    from mcap.writer import Writer
    from test_prepare import _h264_frames
    source = tmp_path / 'recording.mcap'
    wall = 1_790_000_000_000_000_000
    with source.open('wb') as target:
        writer = Writer(target); writer.start()
        channel = writer.register_channel('/head/video', 'json',
                                           writer.register_schema('foxglove.CompressedVideo', 'jsonschema', b'{}'))
        packets = [b'\x00\x00\x00\x01\x09\xf0'] + _h264_frames(18)
        for row, packet in enumerate(packets):
            log = wall + round(max(0, row - 1) / 30 * 1e9)
            capture = wall + 1_000_000_000 if row == 0 else log
            body = {'timestamp': {'sec': capture // 10**9, 'nsec': capture % 10**9},
                    'format': 'h264', 'data': base64.b64encode(packet).decode()}
            writer.add_message(channel, log, json.dumps(body).encode(), log)
        writer.finish()
    report = formats.convert(source, 'ego_head', tmp_path / 'out', 'message address ownership', 900)
    assert not report['failed'] and len(report['episodes']) == 1, report
    loaded = episode.load(tmp_path / 'out' / report['episodes'][0]['episode_id'])
    assert loaded['sources']['exo']['n_frames'] == 18
    np.testing.assert_allclose(loaded['times']['exo'], np.arange(18) / 30, atol=1e-9)
    assert len(episode._decode_view(loaded, 'exo', list(range(18)))) == 18
    assert any(issue['kind'] == 'camera_non_picture_messages' for issue in loaded['context']['reader_issues'])


@pytest.mark.parametrize('same_domain', [False, True])
def test_unshown_disconnected_camera_keeps_own_native_clock_without_changing_selected_axis(tmp_path, same_domain):
    import base64
    from mcap.writer import Writer
    from board.clips import episode_jobs
    source = tmp_path / 'recording.mcap'
    wall = 1_790_000_000_000_000_000
    with source.open('wb') as target:
        writer = Writer(target); writer.start()
        schema = writer.register_schema('foxglove.CompressedImage', 'jsonschema', b'{}')
        camera = writer.register_channel('/head/image', 'json', schema,
                         metadata={'capture_clock_id': 'head', 'log_clock_id': 'head-recorder'})
        side = writer.register_channel('/wrist/image', 'json', schema,
                         metadata={'capture_clock_id': 'head' if same_domain else 'other',
                                   'log_clock_id': 'head-recorder' if same_domain else 'other-recorder'})
        sensor = writer.register_channel('/pressure', 'json', writer.register_schema('Pressure', 'jsonschema', b'{}'),
                         metadata={'capture_clock_id': 'head', 'log_clock_id': 'head-recorder'})
        for row in range(10):
            for channel, origin, interval in [(camera, 1000, 1 / 30), (side, 1000 if same_domain else 2000, .05)]:
                stamp = int(origin * 1e9) + round(row * interval * 1e9)
                picture = io.BytesIO(); Image.new('RGB', (96, 72), (row * 20,) * 3).save(picture, 'PNG')
                body = {'timestamp': {'sec': stamp // 10**9, 'nsec': stamp % 10**9},
                        'format': 'png', 'data': base64.b64encode(picture.getvalue()).decode()}
                writer.add_message(channel, wall + round(row * interval * 1e9), json.dumps(body).encode(), stamp)
            stamp = 1000_000_000_000 + round(row / 30 * 1e9)
            writer.add_message(sensor, wall + round(row / 30 * 1e9), json.dumps({
                'timestamp': {'sec': stamp // 10**9, 'nsec': stamp % 10**9}, 'pressure': row}).encode(), stamp)
        writer.finish()
    report = formats.convert(source, 'ego_head', tmp_path / 'out', 'unshown source clock', 900)
    assert not report['failed'] and len(report['episodes']) == 1, report
    path = tmp_path / 'out' / report['episodes'][0]['episode_id']
    loaded = episode.load(path)
    assert loaded['context']['source']['presentation_clock']['basis'] == 'capture'
    assert loaded['signals']
    entry = loaded['context']['unshown_cameras'][0]
    assert entry['name'] == '/wrist/image'
    assert entry['clock_alignment'] == ('source-domain' if same_domain else 'unverified')
    assert entry['presentation_clock']['basis'] == ('capture' if same_domain else 'native_unpaired')
    with np.load(path / entry['camera_times']) as clocks:
        np.testing.assert_allclose(clocks['presentation'], np.arange(10) * .05, atol=1e-9)
        np.testing.assert_array_equal(clocks['capture_ns'], (1000 if same_domain else 2000) * 10**9 + np.arange(10) * 50_000_000)
        np.testing.assert_array_equal(clocks['log_ns'], wall + np.arange(10) * 50_000_000)
    job = next(job for job in episode_jobs(path, tmp_path / 'clips', True) if job[-1] == 'unshown1')
    assert (job[9] is not None and job[10] is not None) if same_domain else (job[9] is None and job[10] is None)
    if not same_domain:
        from board import clips
        assert clips.extract_one(*job[:4], clips.find_ffmpeg(), 1, *job[4:13]) is None
        with av.open(str(job[3])) as clip:
            pictures = list(clip.decode(video=0))
        assert len(pictures) == 10
        np.testing.assert_allclose([float(frame.pts * frame.time_base) for frame in pictures],
                                   np.r_[0., np.arange(9) * .05 + .025], atol=2e-6)
        for row, picture in enumerate(pictures):
            assert abs(picture.to_ndarray(format='rgb24').mean() - row * 20) <= 3
        request = episode.build_request(path)
        assert entry['presentation_clock']['reason'] in request['prompt']
        assert '/wrist/image is not shown to the model' in request['prompt']
        assert request['views'] == ['exo']
        formats.trim_episode(path, .1)
        trimmed = json.loads((path / 'context.json').read_text())['unshown_cameras'][0]
        assert trimmed['n_frames'] == 10 and trimmed['clock_alignment'] == 'unverified'
        with np.load(path / trimmed['camera_times']) as clocks:
            np.testing.assert_allclose(clocks['presentation'], np.arange(10) * .05, atol=1e-9)
            np.testing.assert_array_equal(clocks['capture_ns'], 2000_000_000_000 + np.arange(10) * 50_000_000)
        after_trim = next(job for job in episode_jobs(path, tmp_path / 'trimmed-clips', True) if job[-1] == 'unshown1')
        assert after_trim[2] == 10 and after_trim[9] is None and after_trim[10] is None
        assert clips.extract_one(*after_trim[:4], clips.find_ffmpeg(), 1, *after_trim[4:13]) is None
        with av.open(str(after_trim[3])) as clip:
            pictures = list(clip.decode(video=0))
        assert len(pictures) == 10
        np.testing.assert_allclose([float(frame.pts * frame.time_base) for frame in pictures],
                                   np.r_[0., np.arange(9) * .05 + .025], atol=2e-6)


def test_recorder_arrival_episode_keeps_a_sidecars_explicit_assumed_start_placement(tmp_path):
    from test_ingestion_v5 import text_recording
    from test_formats import _json_mcap
    from prepare.signal_alignment import ALIGNED_ASSUMED
    raw = tmp_path / 'raw'; raw.mkdir()
    text_recording(raw / 'camera.mcap', clock='arrival', early=1., late=3.5)
    _json_mcap(raw / 'force.mcap', {'/right_pressure': [(.1 + row / 30, {'force': float(row % 7)})
                                                    for row in range(121)]}, t0=0.)
    report = formats.convert(raw, 'ego_head', tmp_path / 'out', 'sidecar placement truth', 900)
    assert not report['failed'], report
    loaded = episode.load(tmp_path / 'out' / report['episodes'][0]['episode_id'])
    assert loaded['context']['source']['presentation_clock']['basis'] == 'mcap_log_time'
    assert loaded['context']['source']['sensor_files']['force.mcap'] == 'placed from both starts'
    signal = next(s for s in loaded['context']['signals'] if s['name'] == '/right_pressure')
    assert signal['aligned_by'] == ALIGNED_ASSUMED
    assert signal.get('timing_basis') != 'mcap_log_time'
