"""Published dataset converters isolate an unreadable camera and retain native clocks."""
import json
from pathlib import Path

import av
import numpy as np
import pytest
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory
from mcap_protobuf.writer import Writer

from label import episode
from prepare import abc130k, genhumanego, realomin
from test_abc_generic import capture_mcap
from test_genhumanego_generic import recording
from test_ingestion_timing_regressions import h264_packets
from test_realomin_generic import indexed_recording


def damage_camera(source, destination, topic, *, move_topic=None, move_ns=0, pad_state=False):
    packets = h264_packets(6)
    assert (np.diff([packet.pts for packet in packets]) < 0).any()
    index, last_state = 0, {}
    with source.open('rb') as stream, Writer(str(destination)) as writer:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for metadata in reader.iter_metadata():
            writer._writer.add_metadata(metadata.name, dict(metadata.metadata))
        for _, channel, message, decoded in reader.iter_decoded_messages():
            if channel.topic == topic:
                decoded.data = bytes(packets[index])
                index += 1
            stamp = message.log_time
            if channel.topic == move_topic:
                stamp += move_ns
                if hasattr(decoded, 'timestamp'):
                    seconds = decoded.timestamp.seconds * 1_000_000_000 + decoded.timestamp.nanos + move_ns
                    decoded.timestamp.seconds, decoded.timestamp.nanos = divmod(seconds, 1_000_000_000)
            writer.write_message(channel.topic, decoded, log_time=stamp, publish_time=message.publish_time)
            if pad_state and channel.topic in abc130k.ARM + abc130k.ARM_ACT:
                last_state[channel.topic] = (decoded, stamp, message.publish_time)
        for state_topic, (decoded, stamp, publish) in last_state.items():
            stamp_ns = decoded.timestamp.seconds * 1_000_000_000 + decoded.timestamp.nanos + 33_000_000
            decoded.timestamp.seconds, decoded.timestamp.nanos = divmod(stamp_ns, 1_000_000_000)
            decoded.position[:] = [value + .01 for value in decoded.position]
            writer.write_message(state_topic, decoded, log_time=stamp + 33_000_000, publish_time=publish + 33_000_000)
    assert index == 6


def decode(path):
    with av.open(str(path)) as video:
        return [frame.to_ndarray(format='rgb24') for frame in video.decode(video=0)]


def test_realomin_rejected_left_camera_uses_right_clock_for_both_actors(tmp_path):
    source, raw, out = tmp_path / 'source.mcap', tmp_path / 'episode.mcap', tmp_path / 'prepared'
    indexed_recording(source)
    damage_camera(source, raw, '/robot0/sensor/camera0/compressed',
                  move_topic='/robot1/sensor/camera0/compressed', move_ns=19_000_000)
    ctx = realomin.convert(raw, out, 'task/episode.mcap')
    sources = json.loads((out / 'sources.json').read_text())
    assert list(sources) == ['right'] and list(ctx['cameras']) == ['right']
    loaded = episode.load(out)
    assert episode.anchor(loaded) == 'right' and ctx['n_state_frames'] == 6
    with np.load(out / 'state.npz') as state:
        assert state['state'].shape == (6, 14)
        assert state['state'][0, 0] == pytest.approx(.11)
        assert state['state'][0, 7] == pytest.approx(1.11)
    with np.load(out / 'times.npz') as times:
        assert set(times.files) == {'right', 'right_pts'} and times['right'][0] == 0
    assert len(decode(sources['right']['packed'])) == 6
    assert not (out / 'left.mp4').exists()
    assert any('/robot0/sensor/camera0/compressed' in item['what'] for item in ctx['reader_issues'])


def test_abc_rejected_top_camera_interpolates_state_on_left_capture_clock(tmp_path):
    source, raw, out = tmp_path / 'source.mcap', tmp_path / 'episode.mcap', tmp_path / 'prepared'
    capture_mcap(source)
    damage_camera(source, raw, '/top-camera', move_topic='/left-wrist-camera', move_ns=20_000_000, pad_state=True)
    ctx = abc130k.convert(raw, out, 'episode_x', 'align fabric')
    sources = json.loads((out / 'sources.json').read_text())
    assert list(sources) == ['left', 'right'] and set(ctx['cameras']) == {'left', 'right'}
    loaded = episode.load(out)
    assert episode.anchor(loaded) == 'left' and ctx['n_state_frames'] == 6
    with np.load(out / 'state.npz') as state:
        assert state['state'][0, 0] == pytest.approx(.01 * 22 / 33, abs=5e-8)
        assert state['state'].shape == state['action'].shape == (6, 14)
    with np.load(out / 'times.npz') as times:
        assert 'exo' not in times and times['left'][0] == pytest.approx(.018)
        assert times['right'][0] == 0
    assert 'kmap' not in sources['left'] and sources['right']['kmap'] == 'kmap_right.npy'
    assert len(decode(sources['left']['packed'])) == 6
    assert not (out / 'exo.mp4').exists()
    assert any('/top-camera' in item['what'] for item in ctx['reader_issues'])


def test_genhumanego_rejected_forward_camera_does_not_lend_geometry_to_fallback(tmp_path):
    source, raw, extracted, out = (tmp_path / name for name in ['source.mcap', 'episode.mcap', 'raw', 'prepared'])
    recording(source, six_cameras=True, both_calibrated=True)
    damage_camera(source, raw, genhumanego.CAMERA_TOPIC,
                  move_topic='/robot0/sensor/camera0/compressed', move_ns=-10_000_000)
    genhumanego.extract(raw, extracted, 'episode.mcap')
    ctx = genhumanego.write_sidecar(extracted, out)
    sources = json.loads((out / 'sources.json').read_text())
    meta = json.loads((extracted / 'meta.json').read_text())
    assert ctx['cameras']['exo']['key'] == '/robot0/sensor/camera0/compressed'
    assert 'forward' not in ctx['cameras']['exo']['desc']
    assert meta.get('calib2') is None
    assert meta['selected_camera']['calibration']['frame'] == '/robot0/sensor/camera0'
    assert Path(sources['exo']['packed']).name == 'cam0.mp4'
    assert len(decode(sources['exo']['packed'])) == 6 and ctx['n_state_frames'] == 6
    assert ctx['instruction'] == 'fold the cloth' and ctx['annotation_subtasks'] == []
    annotation = next(row for row in ctx['recorded_metadata']['mcap_records']
                      if row['topic'] == genhumanego.ANNOTATION_TOPIC)
    assert annotation['fields']['segments_info'][0]['sub_segments_info'][0]['is_success'] is False
    assert any(genhumanego.CAMERA_TOPIC in item['what'] for item in ctx['reader_issues'])


@pytest.mark.parametrize('dataset,bad', [('realomin', '/robot1/sensor/camera0/compressed'),
    ('abc130k', '/left-wrist-camera')])
def test_rejected_secondary_camera_keeps_original_anchor_and_state(tmp_path, dataset, bad):
    source, raw, out = tmp_path / 'source.mcap', tmp_path / 'episode.mcap', tmp_path / 'prepared'
    if dataset == 'realomin':
        indexed_recording(source)
    else:
        capture_mcap(source)
    damage_camera(source, raw, bad)
    if dataset == 'realomin':
        ctx = realomin.convert(raw, out, 'task/episode.mcap')
        expected_anchor, missing = 'left', 'right'
    else:
        ctx = abc130k.convert(raw, out, 'episode_x', 'align fabric')
        expected_anchor, missing = 'exo', 'left'
    sources = json.loads((out / 'sources.json').read_text())
    assert missing not in sources and missing not in ctx['cameras']
    assert episode.anchor(episode.load(out)) == expected_anchor
    with np.load(out / 'state.npz') as state:
        assert state['state'].shape == (6, 14)
    assert any(item['kind'] == 'camera_not_decodable' and bad in item['what'] for item in ctx['reader_issues'])


def test_no_readable_genhumanego_camera_reports_each_rejected_source(tmp_path):
    source = tmp_path / 'source.mcap'
    recording(source, other_camera=True)
    damaged_primary, damaged_all = tmp_path / 'primary.mcap', tmp_path / 'all.mcap'
    damage_camera(source, damaged_primary, genhumanego.CAMERA_TOPIC)
    damage_camera(damaged_primary, damaged_all, '/robot0/sensor/camera0/compressed')
    with pytest.raises(RuntimeError, match='no readable headset camera remains') as error:
        genhumanego.extract(damaged_all, tmp_path / 'raw', 'episode.mcap')
    assert genhumanego.CAMERA_TOPIC in str(error.value) and '/robot0/sensor/camera0/compressed' in str(error.value)
    assert not (tmp_path / 'raw/meta.json').exists()


def remove_camera_packets(source, destination, topic, *, omit=False, payload=b''):
    with source.open('rb') as stream, Writer(str(destination)) as writer:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for metadata in reader.iter_metadata():
            writer._writer.add_metadata(metadata.name, dict(metadata.metadata))
        for _, channel, message, decoded in reader.iter_decoded_messages():
            if channel.topic == topic:
                if omit:
                    continue
                decoded.data = payload
            writer.write_message(channel.topic, decoded, log_time=message.log_time, publish_time=message.publish_time)


@pytest.mark.parametrize('omit', [True, False])
def test_realomin_missing_keyframe_keeps_the_other_recorded_camera(tmp_path, omit):
    source, raw, out = tmp_path / 'source.mcap', tmp_path / 'episode.mcap', tmp_path / 'prepared'
    indexed_recording(source)
    remove_camera_packets(source, raw, '/robot0/sensor/camera0/compressed', omit=omit)
    ctx = realomin.convert(raw, out, 'task/episode.mcap')
    assert list(ctx['cameras']) == ['right']
    assert episode.anchor(episode.load(out)) == 'right'
    assert len(decode(out / 'right.mp4')) == 6
    assert any('SPS, PPS and an IDR' in item['what'] for item in ctx['reader_issues'])


@pytest.mark.parametrize('payload', [b'', b'\x00\x00\x01', b'\x00\x00\x00\x01'])
@pytest.mark.parametrize('malformed_candidate', ['primary', 'fallback'])
def test_genhumanego_unknown_primary_origin_retains_claims_without_guessing_camera_time(tmp_path, payload, malformed_candidate):
    source, raw, extracted, out = (tmp_path / name for name in ['source.mcap', 'episode.mcap', 'raw', 'prepared'])
    recording(source, six_cameras=True, early_other_ns=10_000_000)
    if malformed_candidate == 'primary':
        remove_camera_packets(source, raw, genhumanego.CAMERA_TOPIC, payload=payload)
    else:
        primary_empty = tmp_path / 'primary_empty.mcap'
        remove_camera_packets(source, primary_empty, genhumanego.CAMERA_TOPIC)
        remove_camera_packets(primary_empty, raw, '/robot0/sensor/camera0/compressed', payload=payload)
    genhumanego.extract(raw, extracted, 'episode.mcap')
    ctx = genhumanego.write_sidecar(extracted, out)
    assert ctx['annotation_subtasks'] == []
    selected = 'camera0' if malformed_candidate == 'primary' else 'camera1'
    assert ctx['cameras']['exo']['key'] == f'/robot0/sensor/{selected}/compressed'
    assert 'frame_valid_false' not in ctx['source'] and 'dataset_invalid_ranges' not in ctx['source']
    assert ctx['source']['claims_scope']['camera'] == 'unverified'
    assert ctx['source']['claims_scope']['clock_origin'] == 'unverified'
    records = ctx['recorded_metadata']['mcap_records']
    annotation = next(row for row in records if row['topic'] == genhumanego.ANNOTATION_TOPIC)
    first = annotation['fields']['segments_info'][0]['sub_segments_info'][0]
    assert first['start_time_s'] == .033 and first['is_success'] is False
    validity = [row for row in records if row['topic'] == '/robot0/frame_validity']
    assert len(validity) == 6 and sum(row['fields']['is_valid'] is False for row in validity) == 2
    ranges = next(row for row in records if row['topic'] == '/robot0/time_range_validity')
    assert ranges['fields']['invalid_ranges'][0]['start_time_s'] == .04
    assert all(type(row['log_ns']) is int and type(row['publish_ns']) is int for row in records)


def rewrite_channel(source, destination, topic, failure):
    """Change one recorded source without decoding or rewriting its neighbors."""
    from mcap.writer import Writer as RawWriter
    channels, counts = {}, {}
    with source.open('rb') as stream, destination.open('wb') as target:
        writer = RawWriter(target)
        writer.start()
        reader = make_reader(stream)
        for metadata in reader.iter_metadata():
            writer.add_metadata(metadata.name, dict(metadata.metadata))
        for schema, channel, message in reader.iter_messages():
            bad = channel.topic == topic
            if bad and failure == 'absent':
                continue
            if channel.id not in channels:
                sid = 0
                if schema is not None and not (bad and failure == 'missing_schema'):
                    schema_data = schema.data
                    if bad and failure == 'schema':
                        schema_data = b'\xff'
                    elif bad and failure in ('missing_field', 'scalar_position', 'scalar_calibration', 'repeated_data', 'repeated_value', 'scalar_segments'):
                        from google.protobuf.descriptor_pb2 import FileDescriptorSet
                        descriptor = FileDescriptorSet.FromString(schema_data)
                        for file in descriptor.file:
                            for message_type in file.message_type:
                                if message_type.name == schema.name.rsplit('.', 1)[-1]:
                                    for field_index, field in reversed(list(enumerate(message_type.field))):
                                        if failure == 'missing_field' and field.name == 'data':
                                            del message_type.field[field_index]
                                        elif (failure, field.name) in {('scalar_position', 'position'),
                                                ('scalar_calibration', 'D'), ('scalar_segments', 'segments_info')}:
                                            field.label = 1
                                        elif (failure, field.name) in {('repeated_data', 'data'), ('repeated_value', 'value')}:
                                            field.label = 3
                        schema_data = descriptor.SerializeToString()
                    sid = writer.register_schema(schema.name, schema.encoding, schema_data)
                channels[channel.id] = writer.register_channel(channel.topic, channel.message_encoding, sid)
            index = counts.get(channel.id, 0)
            counts[channel.id] = index + 1
            payload = message.data
            if bad and failure == 'payload' and index == (0 if topic.endswith('/camera_info') else 2):
                payload = b'\xff'
            elif bad and failure in ('packet_count', 'packet_count_many', 'unsupported_codec') and index == (0 if failure == 'unsupported_codec' else 2):
                decoded = DecoderFactory().decoder_for(channel.message_encoding, schema)(payload)
                if failure == 'unsupported_codec':
                    decoded.format = 'unsupported_source_codec'
                else:
                    decoded.data = b'' if failure == 'packet_count' else decoded.data + decoded.data
                payload = decoded.SerializeToString()
            writer.add_message(channels[channel.id], message.log_time, payload,
                               publish_time=message.publish_time, sequence=message.sequence)
        writer.finish()


def dedicated_convert(dataset, source, out):
    if dataset == 'realomin':
        return realomin.convert(source, out, 'task/episode.mcap')
    if dataset == 'abc130k':
        return abc130k.convert(source, out, 'episode_x', 'align fabric')
    genhumanego.extract(source, out / 'source', 'episode.mcap')
    return genhumanego.write_sidecar(out / 'source', out)


@pytest.mark.parametrize('dataset,bad,retained', [
    ('realomin', '/robot0/sensor/camera0/compressed', 'right'),
    ('realomin', '/robot1/sensor/camera0/compressed', 'left'),
    ('abc130k', '/top-camera', 'left'),
    ('abc130k', '/left-wrist-camera', 'exo'),
    ('genhumanego', genhumanego.CAMERA_TOPIC, 'exo'),
])
@pytest.mark.parametrize('failure', ['payload', 'schema', 'missing_schema', 'missing_field', 'packet_count', 'packet_count_many'])
def test_source_camera_failure_preserves_readable_neighbor(tmp_path, dataset, bad, retained, failure):
    source, damaged = tmp_path / 'source.mcap', tmp_path / 'damaged.mcap'
    {'realomin': indexed_recording, 'abc130k': capture_mcap,
     'genhumanego': lambda path: recording(path, other_camera=True, both_calibrated=True)}[dataset](source)
    rewrite_channel(source, damaged, bad, failure)
    ctx = dedicated_convert(dataset, damaged, tmp_path / 'out')
    sources = json.loads((tmp_path / 'out/sources.json').read_text())
    assert retained in sources and ctx['n_state_frames'] == 6
    assert len(decode(sources[retained]['packed'])) == 6
    retained_topic = (ctx['cameras'][retained]['key'] if dataset != 'realomin' else
                      next(topic for topic, view in realomin.CAMERA_TOPICS.items() if view == retained))
    import io
    with source.open('rb') as stream:
        records = list(make_reader(stream, decoder_factories=[DecoderFactory()]).iter_decoded_messages(
            topics=[retained_topic]))
    with av.open(io.BytesIO(b''.join(bytes(row[3].data) for row in records)), format='h264') as original:
        pixels = [frame.to_ndarray(format='rgb24') for frame in original.decode(video=0)]
    assert all(np.array_equal(a, b) for a, b in zip(pixels, decode(sources[retained]['packed']), strict=True))
    stamps = np.asarray([abc130k._ts(row[3]) if dataset == 'abc130k' else row[2].log_time for row in records],
                        dtype=np.int64)
    with np.load(tmp_path / 'out/times.npz') as times:
        np.testing.assert_allclose(times[retained] - times[retained][0], (stamps - stamps[0]) / 1e9,
                                   rtol=0, atol=1e-15)
    assert any(bad in item['what'] for item in ctx['reader_issues'])
    if dataset == 'genhumanego':
        assert ctx['cameras']['exo']['key'] == '/robot0/sensor/camera0/compressed'
        assert ctx['annotation_subtasks'] == []
    else:
        assert len(sources) == (1 if dataset == 'realomin' else 2)
        assert episode.anchor(episode.load(tmp_path / 'out')) == retained


@pytest.mark.parametrize('camera', [0, 2])
@pytest.mark.parametrize('failure', ['payload', 'schema', 'missing_schema', 'absent'])
def test_genhumanego_optional_calibration_failure_keeps_forward_footage(tmp_path, camera, failure):
    source, damaged = tmp_path / 'source.mcap', tmp_path / 'damaged.mcap'
    recording(source, six_cameras=True, both_calibrated=True)
    topic = f'/robot0/sensor/camera{camera}/camera_info'
    # A calibration has one message, so corrupt its full schema or payload directly.
    rewrite_channel(source, damaged, topic, failure)
    ctx = dedicated_convert('genhumanego', damaged, tmp_path / 'out')
    assert ctx['cameras']['exo']['key'] == genhumanego.CAMERA_TOPIC
    assert ctx['instruction'] == 'fold the cloth' and len(ctx['annotation_subtasks']) == 2
    sources = json.loads((tmp_path / 'out/sources.json').read_text())
    assert len(decode(sources['exo']['packed'])) == 6
    meta = json.loads((tmp_path / 'out/source/meta.json').read_text())
    if camera == 2:
        assert meta['selected_camera']['calibration'] is None
    else:
        assert meta['selected_camera']['calibration']['frame'] == '/robot0/sensor/camera2'
    issues = [item for item in ctx.get('reader_issues', []) if topic in item['what']]
    assert bool(issues) is (failure != 'absent')


@pytest.mark.parametrize('dataset,topic', [
    ('realomin', '/robot0/vio/eef_pose'), ('abc130k', '/left-arm-state'),
    ('abc130k', '/left-arm-action')])
@pytest.mark.parametrize('failure', ['payload', 'schema', 'missing_schema', 'absent'])
def test_unreadable_state_channel_keeps_footage_without_partial_actor_state(tmp_path, dataset, topic, failure):
    source, damaged, out = tmp_path / 'source.mcap', tmp_path / 'damaged.mcap', tmp_path / 'out'
    (indexed_recording if dataset == 'realomin' else capture_mcap)(source)
    rewrite_channel(source, damaged, topic, failure)
    ctx = dedicated_convert(dataset, damaged, out)
    sources = json.loads((out / 'sources.json').read_text())
    assert len(sources) == (2 if dataset == 'realomin' else 3)
    assert all(len(decode(row['packed'])) == 6 for row in sources.values())
    if topic.endswith('action'):
        with np.load(out / 'state.npz') as state:
            assert state['state'].shape == (6, 14) and 'action' not in state
    else:
        assert ctx['state_kind'] == 'none' and not (out / 'state.npz').exists()
    if failure != 'absent':
        assert any(topic in item['what'] for item in ctx['reader_issues'])


@pytest.mark.parametrize('dataset', ['realomin', 'abc130k', 'genhumanego'])
def test_unexpected_remux_programming_error_propagates(tmp_path, monkeypatch, dataset):
    source = tmp_path / 'source.mcap'
    {'realomin': indexed_recording, 'abc130k': capture_mcap, 'genhumanego': recording}[dataset](source)
    def broken(*args, **kwargs):
        raise RuntimeError('unexpected implementation failure')
    monkeypatch.setattr({'realomin': realomin, 'abc130k': abc130k, 'genhumanego': genhumanego}[dataset], 'remux', broken)
    with pytest.raises(RuntimeError, match='unexpected implementation failure'):
        dedicated_convert(dataset, source, tmp_path / 'out')


@pytest.mark.parametrize('dataset,topics', [
    ('realomin', ['/robot0/sensor/camera0/compressed', '/robot1/sensor/camera0/compressed']),
    ('abc130k', ['/top-camera', '/left-wrist-camera', '/right-wrist-camera']),
    ('genhumanego', [genhumanego.CAMERA_TOPIC, '/robot0/sensor/camera0/compressed'])])
def test_all_source_cameras_unreadable_names_each_source(tmp_path, dataset, topics):
    source = tmp_path / 'source.mcap'
    {'realomin': indexed_recording, 'abc130k': capture_mcap,
     'genhumanego': lambda path: recording(path, other_camera=True)}[dataset](source)
    for index, topic in enumerate(topics):
        damaged = tmp_path / f'damaged{index}.mcap'
        rewrite_channel(source, damaged, topic, 'payload')
        source = damaged
    out = tmp_path / 'out'
    with pytest.raises(RuntimeError, match='no readable .*camera remains') as error:
        dedicated_convert(dataset, source, out)
    assert all(topic in str(error.value) for topic in topics)
    assert not (out / 'context.json').exists()


@pytest.mark.parametrize('failure', ['payload', 'schema', 'missing_schema', 'absent'])
def test_genhumanego_selected_fallback_calibration_failure_is_not_forward_geometry(tmp_path, failure):
    source, empty, damaged = (tmp_path / name for name in ['source.mcap', 'empty.mcap', 'damaged.mcap'])
    recording(source, other_camera=True, both_calibrated=True)
    remove_camera_packets(source, empty, genhumanego.CAMERA_TOPIC)
    topic = '/robot0/sensor/camera0/camera_info'
    rewrite_channel(empty, damaged, topic, failure)
    ctx = dedicated_convert('genhumanego', damaged, tmp_path / 'out')
    meta = json.loads((tmp_path / 'out/source/meta.json').read_text())
    assert ctx['cameras']['exo']['key'] == '/robot0/sensor/camera0/compressed'
    assert meta['selected_camera']['calibration'] is None and meta['calib2'] is None
    assert ctx['annotation_subtasks'] == []
    issues = [item for item in ctx.get('reader_issues', []) if topic in item['what']]
    assert bool(issues) is (failure != 'absent')
    assert len(decode(tmp_path / 'out/source/cam0.mp4')) == 6


@pytest.mark.parametrize('dataset', ['realomin', 'abc130k', 'genhumanego'])
def test_unexpected_decoder_programming_error_propagates(tmp_path, monkeypatch, dataset):
    from prepare import formats
    source = tmp_path / 'source.mcap'
    {'realomin': indexed_recording, 'abc130k': capture_mcap, 'genhumanego': recording}[dataset](source)
    def broken(*args, **kwargs):
        raise RuntimeError('unexpected decoder implementation failure')
    monkeypatch.setattr(formats, '_decoder_for', broken)
    with pytest.raises(RuntimeError, match='unexpected decoder implementation failure'):
        dedicated_convert(dataset, source, tmp_path / 'out')


def test_abc_failed_stereo_top_preserves_recorded_station_identity(tmp_path):
    source, damaged = tmp_path / 'source.mcap', tmp_path / 'damaged.mcap'
    capture_mcap(source, top='/top-left-camera')
    rewrite_channel(source, damaged, '/top-left-camera', 'payload')
    ctx = dedicated_convert('abc130k', damaged, tmp_path / 'out')
    assert ctx['station'] == 'ZED-X' and set(ctx['cameras']) == {'left', 'right'}
    assert all('/top-camera' not in item['what'] for item in ctx['reader_issues'])


def test_abc_unsupported_recorded_codec_keeps_other_cameras(tmp_path):
    source, damaged = tmp_path / 'source.mcap', tmp_path / 'damaged.mcap'
    capture_mcap(source)
    rewrite_channel(source, damaged, '/top-camera', 'unsupported_codec')
    ctx = dedicated_convert('abc130k', damaged, tmp_path / 'out')
    assert set(ctx['cameras']) == {'left', 'right'}
    assert any('/top-camera' in item['what'] and 'unsupported_source_codec' in item['what']
               for item in ctx['reader_issues'])
    assert len(decode(tmp_path / 'out/left.mp4')) == 6


@pytest.mark.parametrize('dataset,topic,failure', [
    ('abc130k', '/left-arm-state', 'scalar_position'),
    ('abc130k', '/top-camera', 'repeated_data'),
    ('realomin', '/robot0/sensor/camera0/compressed', 'repeated_data'),
    ('realomin', '/robot0/sensor/magnetic_encoder', 'repeated_value'),
    ('genhumanego', genhumanego.CAMERA_TOPIC, 'repeated_data'),
    ('genhumanego', genhumanego.CALIB_TOPIC, 'scalar_calibration'),
    ('genhumanego', genhumanego.ANNOTATION_TOPIC, 'scalar_segments'),
])
def test_incompatible_source_field_cardinality_keeps_other_channels(tmp_path, dataset, topic, failure):
    source, damaged, out = tmp_path / 'source.mcap', tmp_path / 'damaged.mcap', tmp_path / 'out'
    {'realomin': indexed_recording, 'abc130k': capture_mcap,
     'genhumanego': lambda path: recording(path, other_camera=True, both_calibrated=True)}[dataset](source)
    rewrite_channel(source, damaged, topic, failure)
    ctx = dedicated_convert(dataset, damaged, out)
    sources = json.loads((out / 'sources.json').read_text())
    assert all(len(decode(row['packed'])) == 6 for row in sources.values())
    assert any(topic in item['what'] for item in ctx['reader_issues'])
    if failure in ('scalar_position', 'repeated_value'):
        assert ctx['state_kind'] == 'none' and not (out / 'state.npz').exists()
    elif failure == 'scalar_calibration':
        meta = json.loads((out / 'source/meta.json').read_text())
        assert meta['selected_camera']['calibration'] is None
    elif failure == 'scalar_segments':
        assert ctx['instruction'] is None and ctx['annotation_subtasks'] == []
