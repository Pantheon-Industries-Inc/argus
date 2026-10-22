"""Real source ownership, encoded clocks and adapter recovery regressions."""
import io
import json
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
import threading
import zipfile

import av
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from label import episode
from prepare import abc130k, formats
from test_abc_generic import capture_mcap
from test_dedicated_camera_recovery import decode
from test_prepare import joint_names


def encoded_video(path, count, fps=30, missing=(), timescale=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if missing:
        writer = formats.FrameWriter(path, '', fine_clock=True)
        for index in range(count):
            if index not in missing:
                writer.add_image(index / fps, Image.new('RGB', (96, 72), ((index * 17) % 240,) * 3))
        assert writer.close() == count - len(missing)
        return
    rate = Fraction(fps).limit_denominator(1_000_000)
    with av.open(str(path), 'w', options={'video_track_timescale': str(timescale)} if timescale else {}) as output:
        stream = output.add_stream('mpeg4', rate=rate)
        stream.width, stream.height, stream.pix_fmt = 96, 72, 'yuv420p'
        stream.time_base = Fraction(1, 15360) if fps == 50 else 1 / rate / 512
        stream.codec_context.gop_size = 10
        for index in range(count):
            if index in missing:
                continue
            frame = av.VideoFrame.from_ndarray(np.full((72, 96, 3), (index * 17) % 240, np.uint8), format='rgb24')
            frame.pts, frame.time_base = index, 1 / rate
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def lerobot_source(root, *, length=3, episodes=1, offset=.068, fps=30, mixed=False, missing=()):
    front, wrist = 'observation.images.front', 'observation.images.left_wrist'
    info = {'codebase_version': 'v3.0', 'fps': fps, 'features': {
        front: {'dtype': 'video', 'shape': [72, 96, 3]},
        wrist: {'dtype': 'image' if mixed else 'video', 'shape': [72, 96, 3]},
        'observation.state': {'dtype': 'float32', 'shape': [7], 'names': joint_names((None,))}}}
    (root / 'meta').mkdir(parents=True)
    (root / 'meta/info.json').write_text(json.dumps(info))
    first = int(round(offset * fps))
    total = first + length * episodes + 3
    paths = {key: root / 'videos' / key / 'chunk-000/file-000.mp4' for key in (front, wrist)}
    encoded_video(paths[front], total, fps)
    if not mixed:
        encoded_video(paths[wrist], total, fps, missing)
    rows, values, start = [], [], offset
    for index in range(episodes):
        stop = start + round(length / fps, 6)
        row = {'episode_index': index, 'length': length, 'data/chunk_index': 0, 'data/file_index': 0,
               'dataset_from_index': index * length, 'dataset_to_index': (index + 1) * length}
        for key in (front,) if mixed else (front, wrist):
            row.update({f'videos/{key}/chunk_index': 0, f'videos/{key}/file_index': 0,
                        f'videos/{key}/from_timestamp': start, f'videos/{key}/to_timestamp': stop})
        rows.append(row)
        for local in range(length):
            frame = first + index * length + local
            value = {'episode_index': index, 'frame_index': local, 'timestamp': local / fps,
                     'observation.state': [frame, 0., 0., 0., 0., 0., 0.]}
            if mixed:
                image = io.BytesIO()
                Image.new('RGB', (96, 72), ((frame * 17) % 240,) * 3).save(image, 'PNG')
                value[wrist] = {'bytes': image.getvalue(), 'path': None}
            values.append(value)
        start = stop
    metadata = root / 'meta/episodes/chunk-000/file-000.parquet'
    metadata.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(metadata)
    data = root / 'data/chunk-000/file-000.parquet'
    data.parent.mkdir(parents=True)
    pd.DataFrame(values).to_parquet(data)
    return paths, first


def assert_rows(item, out, index, first, length):
    ctx = formats.convert_lerobot(item, 'teleop_arms', out, 'timing fixture')
    ep = out / ctx['episode_id']
    loaded = episode.load(ep)
    np.testing.assert_array_equal(loaded['state'][:, 0], np.arange(first, first + length))
    for view in loaded['sources']:
        frames = episode._decode_view(loaded, view, list(range(length)))
        assert len(frames) == length
        for row, image in frames.items():
            assert np.asarray(image).mean() == pytest.approx(((first + row) * 17) % 240, abs=3)
    return ctx, ep


@pytest.mark.parametrize('fps', [30, 50, 30000 / 1001])
def test_mixed_storage_uses_the_corrected_video_rows(tmp_path, fps):
    root = tmp_path / 'source'
    _, first = lerobot_source(root, mixed=True, fps=fps)
    _, items = formats.plan(root)
    assert_rows(items[0], tmp_path / 'out', 0, first, 3)


@pytest.mark.parametrize('fps,length,count', [(50, 60, 2), (30000 / 1001, 70, 3)])
def test_fractional_encoded_steps_keep_exact_episode_pictures(tmp_path, fps, length, count):
    root = tmp_path / 'source'
    _, first = lerobot_source(root, length=length, episodes=count, offset=47 / fps, fps=fps)
    _, items = formats.plan(root)
    for index, item in enumerate(items):
        assert_rows(item, tmp_path / 'out', index, first + index * length, length)


def test_sibling_vfr_camera_uses_the_same_publisher_query_origin(tmp_path):
    root = tmp_path / 'source'
    lerobot_source(root, length=2, episodes=80, offset=0, missing=(3,))
    _, items = formats.plan(root)
    assert_rows(items[40], tmp_path / 'out', 40, 80, 2)


def test_archive_replacement_reads_new_pixels_and_task(tmp_path):
    source = tmp_path / 'recording.zip'
    video = tmp_path / 'video.mp4'
    def archive(shade, task):
        with av.open(str(video), 'w') as output:
            stream = output.add_stream('mpeg4', rate=30)
            stream.width, stream.height, stream.pix_fmt = 96, 72, 'yuv420p'
            for index in range(3):
                frame = av.VideoFrame.from_ndarray(np.full((72, 96, 3), shade, np.uint8), format='rgb24')
                for packet in stream.encode(frame): output.mux(packet)
            for packet in stream.encode(): output.mux(packet)
        with zipfile.ZipFile(source, 'w') as output:
            output.write(video, 'recording.mp4')
            output.writestr('instruction.txt', task)
    out = tmp_path / 'out'
    for shade, task in [(40, 'put the cup down'), (180, 'lift the plate')]:
        archive(shade, task)
        report = formats.convert(source, 'ego_head', out, 'archive fixture', 900)
        assert not report['failed'] and len(report['episodes']) == 1, report
        ep = out / report['episodes'][0]['episode_id']
        ctx = json.loads((ep / 'context.json').read_text())
        assert ctx['instruction'] == task
        loaded = episode.load(ep)
        image = episode._decode_view(loaded, 'exo', [0])[0]
        assert np.asarray(image).mean() == pytest.approx(shade, abs=3)


@pytest.mark.parametrize('widths', [[0] * 6, [6, 6, 0, 6, 6, 6], [5] * 6])
@pytest.mark.parametrize('topic', ['/left-arm-state', '/left-arm-action'])
def test_abc_malformed_vector_width_keeps_all_footage(tmp_path, widths, topic):
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory
    from mcap_protobuf.writer import Writer
    source, damaged, out = tmp_path / 'source.mcap', tmp_path / 'bad.mcap', tmp_path / 'out'
    capture_mcap(source)
    index = 0
    with source.open('rb') as stream, Writer(str(damaged)) as writer:
        for _, channel, message, decoded in make_reader(stream, decoder_factories=[DecoderFactory()]).iter_decoded_messages():
            if channel.topic == topic:
                decoded.position[:] = list(decoded.position)[:widths[index]]
                index += 1
            writer.write_message(channel.topic, decoded, log_time=message.log_time, publish_time=message.publish_time)
    ctx = abc130k.convert(damaged, out, 'episode_x', 'task')
    assert set(ctx['cameras']) == {'exo', 'left', 'right'}
    assert all(len(decode(out / f'{view}.mp4')) == 6 for view in ctx['cameras'])
    assert any(topic in issue['what'] for issue in ctx['reader_issues'])
    if topic.endswith('state'):
        assert ctx['state_kind'] == 'none' and not (out / 'state.npz').exists()
    else:
        with np.load(out / 'state.npz') as state:
            assert state['state'].shape == (6, 14) and 'action' not in state


def test_malformed_json_channel_does_not_abort_dedicated_camera_decode(tmp_path):
    from mcap.reader import make_reader
    from mcap.writer import Writer
    from prepare.mcap_source import decoded_messages
    source = tmp_path / 'source.mcap'
    with source.open('wb') as stream:
        writer = Writer(stream)
        writer.start()
        bad = writer.register_channel('/camera/bad', 'json', 0)
        healthy = writer.register_channel('/camera/healthy', 'json', 0)
        writer.add_message(bad, 1, b'{broken', 1)
        writer.add_message(healthy, 2, b'{}', 2)
        writer.finish()
    failures = {}
    with source.open('rb') as stream:
        rows = list(decoded_messages(make_reader(stream), {'/camera/bad': {}, '/camera/healthy': {}}, failures))
    assert [channel.topic for _, channel, _, _ in rows] == ['/camera/healthy']
    assert '/camera/bad' in failures


@pytest.mark.parametrize('text_clock', ['capture', 'capture_late', 'arrival_only', 'different_capture'])
def test_mcap_text_uses_its_verified_camera_clock(tmp_path, text_clock):
    import base64
    from mcap.writer import Writer
    source = tmp_path / 'source.mcap'
    picture = io.BytesIO()
    Image.new('RGB', (32, 24), 'white').save(picture, 'PNG')
    camera_base, wall = 1000, 1_790_000_000
    def header(stamp):
        seconds = int(stamp)
        return {'stamp': {'sec': seconds, 'nanosec': round((stamp - seconds) * 1e9)}}
    with source.open('wb') as stream:
        writer = Writer(stream)
        writer.start()
        schema = writer.register_schema('sensor_msgs/CompressedImage', 'jsonschema', b'{}')
        camera = writer.register_channel('/camera/front/compressed', 'json', schema, metadata={'capture_clock_id': 'fixture'})
        text_schema = writer.register_schema('TaskText', 'jsonschema', b'{}')
        task = writer.register_channel('/task', 'json', text_schema, metadata={'capture_clock_id': 'fixture' if text_clock.startswith('capture') else 'other'})
        step = writer.register_channel('/task/subtask', 'json', text_schema, metadata={'capture_clock_id': 'fixture' if text_clock.startswith('capture') else 'other'})
        for index in range(120):
            data = {'header': header(camera_base + index / 30), 'format': 'png',
                    'data': base64.b64encode(picture.getvalue()).decode()}
            writer.add_message(camera, round((wall + index / 30) * 1e9), json.dumps(data).encode(), 0)
        for channel, when, text in [(task, 1., 'fold the cloth'), (step, 1., 'pick up'), (step, 2.5, 'fold')]:
            data = {'data': text}
            if text_clock != 'arrival_only':
                data['header'] = header((camera_base if text_clock.startswith('capture') else wall) + when)
            # text arrives about as fast as the frames, or a second late; on the camera's declared clock a late
            # delivery cannot disprove the declaration, so it keeps its stamps and the difference is flagged
            late = .005 if text_clock == 'capture' else 1
            writer.add_message(channel, round((wall + when + late) * 1e9), json.dumps(data).encode(), 0)
        writer.finish()
    out = tmp_path / 'out'
    report = formats.convert(source, 'ego_head', out, 'clock fixture', 900)
    assert not report['failed'], report
    ctx = json.loads((out / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    if text_clock.startswith('capture'):
        assert [row['t0'] for row in steps(ctx)] == [1., 2.5]
        assert ctx['instruction'] == 'fold the cloth'
        flagged = any(i['kind'] == 'capture_clock_offset_possible' for i in ctx.get('reader_issues') or [])
        assert flagged is (text_clock == 'capture_late')
    else:
        assert [row['t0'] for row in steps(ctx)] == [2., 3.5]
        assert ctx['source']['presentation_clock']['basis'] == 'mcap_log_time'
        assert 'arrival' in ctx['collection_note']
        assert 'pick up' in json.dumps(ctx) and 'fold' in json.dumps(ctx)


def test_parallel_conversion_never_observes_partial_episode_ownership(tmp_path, monkeypatch):
    from concurrent.futures import TimeoutError
    root = tmp_path / 'source'
    paths, _ = lerobot_source(root, length=2, episodes=1000, offset=0)
    _, items = formats.plan(root)
    path = paths['observation.images.front']
    stat = path.stat()
    facts = formats._packed_video_probe(str(path.resolve()),
        (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino))
    entered, release, second_started = threading.Event(), threading.Event(), threading.Event()
    difference = np.diff
    def schedule(value, *args, **kwargs):
        if value is facts['pts'] and not entered.is_set():
            entered.set()
            assert release.wait(5)
        return difference(value, *args, **kwargs)
    monkeypatch.setattr(formats.np, 'diff', schedule)
    def convert(index, second=False):
        if second:
            second_started.set()
        return formats.convert_lerobot(items[index], 'teleop_arms', tmp_path / 'out', 'parallel fixture')
    with ThreadPoolExecutor(2) as workers:
        first = workers.submit(convert, 999)
        assert entered.wait(5)
        second = workers.submit(convert, 998, True)
        assert second_started.wait(5)
        try:
            second.result(timeout=.25)
        except TimeoutError:
            pass
        finally:
            release.set()
        results = [(999, first.result()), (998, second.result())]
    for index, context in results:
        ep = episode.load(tmp_path / 'out' / context['episode_id'])
        np.testing.assert_array_equal(ep['state'][:, 0], [index * 2, index * 2 + 1])
        for view in ep['sources']:
            pictures = episode._decode_view(ep, view, [0, 1])
            assert len(pictures) == 2
            for row, picture in pictures.items():
                assert np.asarray(picture).mean() == pytest.approx(((index * 2 + row) * 17) % 240, abs=3)


def test_missing_vfr_query_row_is_rejected_without_shifting_the_other_camera(tmp_path):
    root = tmp_path / 'source'
    lerobot_source(root, length=2, episodes=80, offset=0, missing=(3, 81))
    _, items = formats.plan(root)
    context, ep_dir = assert_rows(items[40], tmp_path / 'out', 40, 80, 2)
    assert set(context['cameras']) == {'exo'}
    assert any('observation.images.left_wrist' in issue['what'] for issue in context['reader_issues'])
    assert episode.load(ep_dir)['state'].shape == (2, 7)


def test_vfr_sibling_retains_exact_small_encoded_clock_offsets(tmp_path):
    root = tmp_path / 'source'
    paths, _ = lerobot_source(root, length=2, episodes=80, offset=0, missing=(3,))
    writer = formats.FrameWriter(paths['observation.images.left_wrist'], '', fine_clock=True)
    for index in range(163):
        if index != 3:
            writer.add_image(index / 30 + (0.00005 if index == 81 else 0),
                             Image.new('RGB', (96, 72), ((index * 17) % 240,) * 3))
    assert writer.close() == 162
    _, items = formats.plan(root)
    _, ep_dir = assert_rows(items[40], tmp_path / 'out', 40, 80, 2)
    with np.load(ep_dir / 'times.npz') as times:
        assert times['left_pts'].tolist() == [2666667, 2700050]
        np.testing.assert_allclose(times['left'], np.array([2666667, 2700050]) / 1e6 - 80 / 30, atol=1e-12)
        assert times['left'][1] - times['left'][0] > 1 / 30


def test_incomplete_archive_extraction_is_never_reused(tmp_path, monkeypatch):
    source, video = tmp_path / 'source.zip', tmp_path / 'video.mp4'
    encoded_video(video, 3)
    with zipfile.ZipFile(source, 'w') as writer:
        writer.write(video, 'recording.mp4')
        writer.writestr('instruction.txt', 'fold the cloth')
    read = zipfile.ZipExtFile.read
    def interrupted(stream, *args, **kwargs):
        if stream.name == 'recording.mp4':
            raise OSError('interrupted source member')
        return read(stream, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(zipfile.ZipExtFile, 'read', interrupted)
        with pytest.raises(ValueError):
            formats.convert(source, 'ego_head', tmp_path / 'out', 'archive', 900)
    report = formats.convert(source, 'ego_head', tmp_path / 'out', 'archive', 900)
    assert len(report['episodes']) == 1 and not report['failed']
    ep = episode.load(tmp_path / 'out' / report['episodes'][0]['episode_id'])
    assert ep['context']['instruction'] == 'fold the cloth'
    assert len(episode._decode_view(ep, 'exo', [0, 1, 2])) == 3


def test_malformed_json_camera_keeps_real_abc_neighboring_pixels(tmp_path):
    from mcap.reader import make_reader
    from mcap.writer import Writer
    source, damaged, out = tmp_path / 'source.mcap', tmp_path / 'bad.mcap', tmp_path / 'out'
    capture_mcap(source)
    channels = {}
    with source.open('rb') as incoming, damaged.open('wb') as target:
        writer = Writer(target)
        writer.start()
        for schema, channel, message in make_reader(incoming).iter_messages():
            bad = channel.topic == '/top-camera'
            if channel.id not in channels:
                sid = writer.register_schema(schema.name, schema.encoding, schema.data) if not bad else 0
                channels[channel.id] = writer.register_channel(channel.topic, 'json' if bad else channel.message_encoding, sid)
            writer.add_message(channels[channel.id], message.log_time, b'{bad' if bad else message.data,
                               publish_time=message.publish_time)
        writer.finish()
    context = abc130k.convert(damaged, out, 'episode_x', 'task')
    assert set(context['cameras']) == {'left', 'right'}
    assert all(len(decode(out / f'{view}.mp4')) == 6 for view in context['cameras'])
    assert any('/top-camera' in issue['what'] for issue in context['reader_issues'])
    np.testing.assert_allclose(np.load(out / 'state.npz')['state'][:, 0],
        [.00060606, .01060606, .02060606, .03060606, .04060606, .05], atol=1e-7)


def test_all_vfr_cameras_keep_count_ownership_anchored_at_shared_zero(tmp_path):
    root = tmp_path / 'source'
    paths, _ = lerobot_source(root, length=2, episodes=80, offset=0, missing=(3,))
    encoded_video(paths['observation.images.front'], 163, 30, missing=(5,))
    _, items = formats.plan(root)
    context, _ = assert_rows(items[40], tmp_path / 'out', 40, 80, 2)
    assert set(context['cameras']) == {'exo', 'left'}


@pytest.mark.parametrize('cleanup', ['vanished_sidecar', 'permission_denied'])
def test_archive_failure_cleanup_preserves_source_error_and_real_permissions(tmp_path, monkeypatch, cleanup):
    import os
    source = tmp_path / 'source.zip'
    with zipfile.ZipFile(source, 'w') as writer:
        writer.writestr('instruction.txt', 'source claim')
    def extraction_stopped(snapshot, destination):
        snapshot.with_name('._' + snapshot.name).write_bytes(b'AppleDouble')
        raise ValueError('source extraction stopped')
    unlink, scandir = os.unlink, os.scandir
    class Scan:
        def __init__(self, path):
            with scandir(path) as entries:
                self.entries = iter(sorted(entries, key=lambda entry: entry.name.startswith('._')))
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __iter__(self): return self
        def __next__(self): return next(self.entries)
    def remove(path, *args, **kwargs):
        if Path(path).name == 'source.zip':
            if cleanup == 'permission_denied':
                raise PermissionError('owned archive cleanup denied')
            unlink(path, *args, **kwargs)
            companion = '._source.zip' if kwargs.get('dir_fd') is not None else Path(path).with_name('._source.zip')
            unlink(companion, *args, **kwargs)
        else:
            unlink(path, *args, **kwargs)
    monkeypatch.setattr(formats, '_unpack_archives', extraction_stopped)
    monkeypatch.setattr(os, 'scandir', Scan)
    monkeypatch.setattr(os, 'unlink', remove)
    error = PermissionError if cleanup == 'permission_denied' else ValueError
    message = 'owned archive cleanup denied' if cleanup == 'permission_denied' else 'source extraction stopped'
    with pytest.raises(error, match=message):
        formats.convert(source, 'ego_head', tmp_path / 'out', 'archive fixture', 900)
    if cleanup == 'vanished_sidecar':
        assert not list((tmp_path / 'upload_unpacked').glob('archive-*'))
    assert source.exists() and not list((tmp_path / 'out').glob('*/context.json'))


@pytest.mark.parametrize('coarse', [False, True])
def test_packed_scalar_depth_resolves_the_same_publisher_queries_as_rgb_and_state(tmp_path, coarse):
    from label import depth
    root = tmp_path / 'source'
    lerobot_source(root)
    key = 'observation.images.front_depth'
    info_path = root / 'meta/info.json'
    info = json.loads(info_path.read_text())
    info['features'][key] = {'dtype': 'video', 'shape': [72, 96, 1],
                             'info': {'video.is_depth_map': True, 'depth_scale': .001}}
    info_path.write_text(json.dumps(info))
    metadata = root / 'meta/episodes/chunk-000/file-000.parquet'
    rows = pd.read_parquet(metadata)
    for field, value in {'chunk_index': 0, 'file_index': 0, 'from_timestamp': .068, 'to_timestamp': .168}.items():
        rows[f'videos/{key}/{field}'] = value
    rows.to_parquet(metadata)
    path = root / 'videos' / key / 'chunk-000/file-000.mp4'
    path.parent.mkdir(parents=True)
    denominator = 1000 if coarse else 1_000_000
    with av.open(str(path), 'w', format='mov', options={'video_track_timescale': str(denominator)}) as output:
        stream = output.add_stream('png', rate=30)
        stream.width, stream.height, stream.pix_fmt = 96, 72, 'gray16be'
        stream.time_base = stream.codec_context.time_base = Fraction(1, denominator)
        for index in range(8):
            if index == 6 and not coarse:
                continue
            frame = av.VideoFrame.from_ndarray(np.full((72, 96), 1000 + index, np.uint16), format='gray16be')
            frame.pts = 66617 if index == 2 and not coarse else round(index / 30 * denominator)
            frame.time_base = Fraction(1, denominator)
            for packet in stream.encode(frame): output.mux(packet)
        for packet in stream.encode(): output.mux(packet)
    _, items = formats.plan(root)
    context, ep_dir = assert_rows(items[0], tmp_path / 'out', 0, 2, 3)
    loaded = episode.load(ep_dir)
    own = next(iter(depth.load(ep_dir)))
    pixels = depth.at_anchor(loaded, depth.load(ep_dir), own, [0, 1, 2])
    assert [int(pixels[row][0, 0]) for row in range(3)] == [1002, 1003, 1004]
    with np.load(ep_dir / 'depth_times.npz') as times:
        assert times['depth_' + own + '_pts'].tolist() == ([67, 100, 133] if coarse else [66617, 100000, 133333])
    assert not any(issue['kind'] == 'depth_partial' for issue in context.get('reader_issues', []))


def text_recording(path, *, clock='capture', early=-.05, late=4.2, malformed=None, neighbor=False, text_offset=0.,
                   camera_delay=0., text_delay=0., publish_clock=False, clock_id=None, text_clock_id=None, task_change_time=1., publish_offset=0.):
    import base64
    from mcap.writer import Writer
    png = io.BytesIO(); Image.new('RGB', (32, 24), 'white').save(png, 'PNG')
    def stamp(value):
        ns = round(value * 1e9)
        seconds, nanos = divmod(ns, 1_000_000_000)
        return {'stamp': {'sec': seconds, 'nanosec': nanos}}
    wall, capture = 1_790_000_000, 1000
    with path.open('wb') as target:
        writer = Writer(target); writer.start()
        image = writer.register_schema('sensor_msgs/CompressedImage', 'jsonschema', b'{}')
        text = writer.register_schema('TaskText', 'jsonschema', b'{}')
        metadata = {'capture_clock_id': clock_id} if clock_id else {}
        camera = writer.register_channel('/camera/front/compressed', 'json', image, metadata=metadata)
        wrist = writer.register_channel('/camera/wrist/compressed', 'json', image) if neighbor else None
        text_metadata = {'capture_clock_id': text_clock_id} if text_clock_id else metadata
        task = writer.register_channel('/task', 'json', text, metadata=text_metadata)
        step = writer.register_channel('/task/subtask', 'json', text, metadata=text_metadata)
        for index in range(121):
            message = {'format': 'png', 'data': base64.b64encode(png.getvalue()).decode()}
            if clock != 'arrival': message['header'] = stamp(capture + index / 30)
            body = json.dumps(message).encode()
            data = (b'{broken' if malformed == 'camera' else b'\xff') if malformed in {'camera', 'utf8'} and index == 45 else body
            writer.add_message(camera, round((wall + index / 30 + camera_delay) * 1e9), data,
                               round((capture + index / 30) * 1e9) if publish_clock else 0)
            if wrist is not None:
                writer.add_message(wrist, round((wall + index / 30) * 1e9), body, 0)
        for channel, at, label in [(task, early, 'prepare'), (task, task_change_time, 'fold the cloth'),
                                   (step, early, 'pick up'), (step, 2.5, 'fold'), (step, late, 'after recording')]:
            message = {'data': label}
            if clock != 'arrival': message['header'] = stamp((wall if clock == 'unrelated' else capture) + at + text_offset)
            if malformed == 'backwards_clock' and label == 'fold':
                message['header'] = stamp(capture - .5)
            data = b'{broken' if malformed == 'text' and label == 'fold' else json.dumps(message).encode()
            writer.add_message(channel, round((wall + at + text_delay) * 1e9), data,
                               round((capture + at + publish_offset) * 1e9) if publish_clock else 0)
        writer.finish()


def steps(context):
    """The /task/subtask steps among a recording's timed prompts: a /task whose text changes adds the task of each
    stretch beside them, named with its topic (prepare/formats.py mcap_task_texts)."""
    return [row for row in context.get('annotation_subtasks') or [] if row.get('column') != '/task']


def task_prompts(context):
    """{label: t0} of the /task texts given as timed task prompts."""
    return {row['label']: row['t0'] for row in context.get('annotation_subtasks') or [] if row.get('column') == '/task'}


@pytest.mark.parametrize('clock', ['arrival', 'capture'])
def test_mcap_text_span_boundaries_do_not_change_clock_compatibility(tmp_path, clock):
    source = tmp_path / 'source.mcap'
    text_recording(source, clock=clock, early=-.02 if clock == 'arrival' else -.05,
                   clock_id='fixture' if clock == 'capture' else None)
    report = formats.convert(source, 'ego_head', tmp_path / 'out', 'source timeline', 900)
    assert not report['failed'], report
    ctx = json.loads((tmp_path / 'out' / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    assert [row['label'] for row in steps(ctx)] == ['pick up', 'fold']
    assert [row['t0'] for row in steps(ctx)] == [0., 2.5]
    assert steps(ctx)[-1]['t1'] == pytest.approx(4.033, abs=.001)
    # the task text changes, so each text is the task of its stretch and the recording has no instruction
    assert not ctx.get('instruction') and task_prompts(ctx) == {'prepare': 0., 'fold the cloth': 1.}
    assert not any(issue['kind'] == 'text_clock_unverified' for issue in ctx.get('reader_issues', []))
    assert 'after recording' in json.dumps(ctx)
    proof = ctx['source']['text_clock_alignment']
    assert proof['physical_synchronization_verified'] is False
    assert 'Publisher clock declarations' in proof['limitation']


@pytest.mark.parametrize('malformed', ['camera', 'text', 'utf8'])
def test_generic_mcap_malformed_payload_keeps_later_frames_and_steps(tmp_path, malformed):
    source = tmp_path / 'source.mcap'
    text_recording(source, early=.1, late=3.5, malformed=malformed, neighbor=True, clock_id='fixture')
    report = formats.convert(source, 'ego_head', tmp_path / 'out', 'damaged message', 900)
    assert not report['failed'], report
    context = json.loads((tmp_path / 'out' / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    ep_dir = tmp_path / 'out' / context['episode_id']
    assert context['n_state_frames'] == (120 if malformed in {'camera', 'utf8'} else 121)
    neighbor = next(camera for camera in context['unshown_cameras'] if 'wrist' in camera['name'])
    assert neighbor['n_frames'] == 121 and len(decode(Path(neighbor['packed']))) == 121
    assert len(decode(ep_dir / 'exo.mp4')) == context['n_state_frames']
    assert any(step['label'] == 'after recording' for step in context['annotation_subtasks'])
    assert not any('ends early' in note for note in context.get('notes', []))
    assert any(issue['kind'] == 'mcap_message_unreadable' for issue in context['reader_issues'])


@pytest.mark.parametrize('reader', ['lerobot', 'egocentric'])
def test_one_file_fractional_rate_has_no_false_grid_warning(tmp_path, reader):
    from prepare import egocentric100k
    rate = 30000 / 1001
    source = tmp_path / 'video.mp4'
    encoded_video(source, 8, rate)
    if reader == 'lerobot':
        root = tmp_path / 'source'
        (root / 'meta').mkdir(parents=True)
        (root / 'meta/info.json').write_text(json.dumps({'codebase_version': 'v2.1', 'fps': rate,
            'features': {'observation.images.front': {'dtype': 'video', 'shape': [72, 96, 3]}}}))
        destination = root / 'videos/chunk-000/observation.images.front/episode_000000.mp4'
        destination.parent.mkdir(parents=True); destination.write_bytes(source.read_bytes())
        report = formats.convert(root, 'ego_head', tmp_path / 'out', 'fractional cadence', 900)
        assert not report['failed'], report
        context = json.loads((tmp_path / 'out' / report['episodes'][0]['episode_id'] / 'context.json').read_text())
        assert not any(issue['kind'] == 'recorded_rate_mismatch' for issue in context.get('reader_issues', []))
    else:
        monkey = pytest.MonkeyPatch()
        metadata = tmp_path / 'video.json'; metadata.write_text(json.dumps({'fps': rate}))
        try:
            monkey.setattr(egocentric100k, 'clip_files', lambda *args: (source, metadata))
            context = egocentric100k.prepare_clip('shard', 'video.mp4', tmp_path, tmp_path / 'out')
        finally:
            monkey.undo()
        assert 'real_times' not in context


@pytest.mark.parametrize('fps', [30000 / 1001, 24, 60, 60000 / 1001])
def test_count_owned_coarse_container_ticks_keep_every_exact_picture(tmp_path, fps):
    root = tmp_path / 'source'
    paths, first = lerobot_source(root, length=10, episodes=2, offset=47 / fps, fps=fps)
    for path in paths.values():
        encoded_video(path, 70, fps, timescale=1000)
        assert formats.probe(path)['time_base'] == Fraction(1, 1000)
    _, items = formats.plan(root)
    for index, item in enumerate(items):
        context, ep_dir = assert_rows(item, tmp_path / 'out', index, first + index * 10, 10)
        assert any(issue['kind'] == 'packed_clock_quantized' for issue in context['reader_issues'])
        with np.load(ep_dir / 'times.npz') as times:
            facts = formats.probe(paths['observation.images.front'])
            np.testing.assert_array_equal(times['exo_pts'], facts['pts'][first + index * 10:first + (index + 1) * 10])


@pytest.mark.parametrize('jitter', [.0002, .004])
def test_arbitrary_jitter_is_not_relabelled_as_container_quantization(tmp_path, jitter):
    root = tmp_path / 'source'
    paths, first = lerobot_source(root, length=10, episodes=1, offset=0)
    path = paths['observation.images.left_wrist']
    writer = formats.FrameWriter(path, '', fine_clock=True)
    for index in range(13):
        writer.add_image(index / 30 + (jitter if index % 2 else 0),
                         Image.new('RGB', (96, 72), ((index * 17) % 240,) * 3))
    assert writer.close() == 13 and len(decode(path)) == 13
    _, items = formats.plan(root)
    context = formats.convert_lerobot(items[0], 'teleop_arms', tmp_path / 'out', 'jitter countercase')
    loaded = episode.load(tmp_path / 'out' / context['episode_id'])
    np.testing.assert_array_equal(loaded['state'][:, 0], np.arange(first, first + 10))
    pictures = episode._decode_view(loaded, 'exo', list(range(10)))
    originals = decode(paths['observation.images.front'])
    assert len(pictures) == 10
    for index, picture in pictures.items():
        np.testing.assert_array_equal(np.asarray(picture), originals[index])
    assert set(context['cameras']) == {'exo'}
    assert any(issue['kind'] == 'camera_not_aligned' and 'left_wrist' in issue['camera']
               for issue in context['reader_issues'])
    assert not any(issue['kind'] == 'camera_not_decodable' and 'left_wrist' in issue['camera']
                   for issue in context['reader_issues'])


def test_retained_packed_camera_keeps_query_pts_and_exact_clock_offsets(tmp_path):
    from label.frames import extract_frames
    root = tmp_path / 'source'
    paths, _ = lerobot_source(root)
    key = 'observation.images.rear_mask'
    info_path = root / 'meta/info.json'
    info = json.loads(info_path.read_text())
    info['features'][key] = {'dtype': 'video', 'shape': [72, 96, 3]}
    info_path.write_text(json.dumps(info))
    metadata = root / 'meta/episodes/chunk-000/file-000.parquet'
    table = pd.read_parquet(metadata)
    for field in ('chunk_index', 'file_index', 'from_timestamp', 'to_timestamp'):
        table[f'videos/{key}/{field}'] = table[f'videos/observation.images.front/{field}']
    table.to_parquet(metadata)
    path = root / 'videos' / key / 'chunk-000/file-000.mp4'
    path.parent.mkdir(parents=True)
    writer = formats.FrameWriter(path, '', fine_clock=True)
    for index in range(8):
        if index != 6:
            writer.add_image(index / 30 - (.00005 if index == 2 else 0),
                             Image.new('RGB', (96, 72), ((index * 17) % 240,) * 3))
    assert writer.close() == 7
    _, items = formats.plan(root)
    context, ep_dir = assert_rows(items[0], tmp_path / 'out', 0, 2, 3)
    retained = next(camera for camera in context['unshown_cameras'] if camera['name'] == key)
    with np.load(ep_dir / retained['camera_times']) as times:
        assert times['pts'].tolist() == [66617, 100000, 133333]
        np.testing.assert_allclose(times['capture'], np.array([66617, 100000, 133333]) / 1e6 - 2 / 30,
                                   rtol=0, atol=1e-15)
        pictures = extract_frames(retained['packed'], retained['base_s'], 3, [0, 1, 2], pts=times['pts'])
    for row, picture in pictures.items():
        assert np.asarray(picture).mean() == pytest.approx((row + 2) * 17, abs=3)


def test_mixed_embedded_camera_keeps_count_owned_coarse_packed_sibling(tmp_path):
    root = tmp_path / 'source'
    paths, first = lerobot_source(root, length=10, offset=47 / 24, fps=24, mixed=True)
    encoded_video(paths['observation.images.front'], 60, 24, timescale=1000)
    _, items = formats.plan(root)
    context, _ = assert_rows(items[0], tmp_path / 'out', 0, first, 10)
    assert set(context['cameras']) == {'exo', 'left'}
    assert any(issue['kind'] == 'packed_clock_quantized' for issue in context['reader_issues'])


def test_mcap_outside_topic_bounds_do_not_join_unrelated_capture_domains(tmp_path):
    source = tmp_path / 'source.mcap'
    text_recording(source, clock='unrelated')
    report = formats.convert(source, 'ego_head', tmp_path / 'out', 'unrelated clock bounds', 900)
    assert not report['failed'], report
    context = json.loads((tmp_path / 'out' / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    assert [row['t0'] for row in steps(context)] == [0., 2.5]
    assert 'fold the cloth' in task_prompts(context) and not context.get('instruction')
    assert context['source']['presentation_clock']['basis'] == 'mcap_log_time'
    assert any(issue['kind'] == 'recorder_arrival_timing' for issue in context['reader_issues'])
    assert 'after recording' in json.dumps(context['recorded_metadata']['text_channels'])


def test_generic_mcap_unexpected_decoder_failure_propagates_after_readable_frames(tmp_path, monkeypatch):
    source = tmp_path / 'source.mcap'
    text_recording(source)
    _, items = formats.plan(tmp_path)
    original, calls = formats._decoder_for, []
    def faulty_decoder(*args, **kwargs):
        decoder = original(*args, **kwargs)
        if decoder is None:
            return None
        def decode_message(data):
            calls.append(data)
            if len(calls) == 50:
                raise RuntimeError('decoder implementation regression')
            return decoder(data)
        return decode_message
    monkeypatch.setattr(formats, '_decoder_for', faulty_decoder)
    with pytest.raises(RuntimeError, match='decoder implementation regression'):
        formats.convert_mcap_generic(items[0], 'ego_head', tmp_path / 'out', 'programming fault')
    assert len(calls) == 50


def test_mcap_capture_clock_cannot_lead_its_recorded_arrival_origin(tmp_path):
    source = tmp_path / 'source.mcap'
    text_recording(source, text_offset=2.)
    report = formats.convert(source, 'ego_head', tmp_path / 'out', 'future capture clock', 900)
    assert not report['failed'], report
    context = json.loads((tmp_path / 'out' / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    assert [row['t0'] for row in steps(context)] == [0., 2.5]
    assert 'fold the cloth' in task_prompts(context) and not context.get('instruction')
    assert context['source']['presentation_clock']['basis'] == 'mcap_log_time'
    assert any(issue['kind'] == 'recorder_arrival_timing' for issue in context['reader_issues'])
    records = context['recorded_metadata']['text_channels']['/task/subtask']
    assert records[0]['capture_ns'] == 1001950000000
    assert 'after recording' in json.dumps(records)


def owned_annexb_mcap(path, defect):
    import base64
    from mcap.writer import Writer
    encoder = av.CodecContext.create('libx264', 'w')
    encoder.width, encoder.height, encoder.pix_fmt = 64, 48, 'yuv420p'
    encoder.time_base = Fraction(1, 30)
    encoder.options = {'preset': 'ultrafast', 'tune': 'zerolatency', 'x264-params': 'keyint=1'}
    pictures = []
    for index in range(6):
        frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), index * 30, np.uint8), format='rgb24')
        frame.pts = index
        pictures.extend(bytes(packet) for packet in encoder.encode(frame))
    pictures.extend(bytes(packet) for packet in encoder.encode())
    with path.open('wb') as target:
        writer = Writer(target); writer.start()
        image = writer.register_schema('sensor_msgs/CompressedImage', 'jsonschema', b'{}')
        cameras = [writer.register_channel(topic, 'json', image)
                   for topic in ['/camera/front/compressed', '/camera/wrist/compressed']]
        joint = writer.register_channel('/joint_states', 'json',
            writer.register_schema('sensor_msgs/JointState', 'jsonschema', b'{}'))
        for index in range(6):
            at = 1_700_000_000_000_000_000 + index * 33_333_333
            for camera, channel in enumerate(cameras):
                if camera == 0 and defect == 'batch' and index == 3:
                    continue
                packet = pictures[index]
                if camera == 0 and index == 2:
                    packet = b'' if defect in {'empty', 'cancelled_count'} else pictures[2] + pictures[3] if defect == 'batch' else packet
                if camera == 0 and defect == 'cancelled_count' and index == 3:
                    packet = pictures[3] + pictures[4]
                if camera == 0 and defect == 'cancelled_count' and index == 4:
                    continue
                body = {'format': 'h264', 'data': base64.b64encode(packet).decode()}
                if camera == 0 and index == 2 and defect == 'base64': body['data'] = 'bad base64!'
                if camera == 0 and index == 2 and defect == 'null': body = None
                writer.add_message(channel, at, json.dumps(body).encode(), at)
            writer.add_message(joint, at, json.dumps({'name': joint_names((None,)),
                'position': [index, 0, 0, 0, 0, 0, 0], 'units': ['rad'] * 6 + ['fraction']}).encode(), at)
        writer.finish()


@pytest.mark.parametrize('defect', ['empty', 'batch', 'cancelled_count', 'base64', 'null'])
def test_generic_camera_message_ownership_preserves_neighbor_and_physical_state(tmp_path, defect):
    source = tmp_path / 'source.mcap'
    owned_annexb_mcap(source, defect)
    out = tmp_path / 'prepared'
    context = formats.convert_mcap_generic({'file': source, 'name': 'owned', 'seconds': .2,
        'topics': ['/camera/front/compressed', '/camera/wrist/compressed', '/joint_states']},
        'teleop_arms', out, 'message ownership')
    loaded = episode.load(out)
    assert loaded['context']['state_kind'] == 'joints'
    state = loaded['state'][:, 0]
    for view in loaded['sources']:
        pictures = episode._decode_view(loaded, view, list(range(len(state))))
        for row, picture in pictures.items():
            assert np.asarray(picture).mean() == pytest.approx(state[row] * 30, abs=3)
    if defect in {'batch', 'cancelled_count'}:
        assert len(context['cameras']) == 1 and len(state) == 6
        assert any(issue['kind'] == 'camera_not_decodable' for issue in context['reader_issues'])
    else:
        assert len(context['cameras']) == 2 and len(state) == 5
        np.testing.assert_allclose(state, [0, 1, 3, 4, 5])
        assert any(issue['kind'] in {'frames_not_decodable', 'mcap_message_unreadable'} for issue in context['reader_issues'])


def raw_camera_mcap(path, damaged=None, field=None, value=None):
    import base64
    from mcap.writer import Writer
    with path.open('wb') as target:
        writer = Writer(target); writer.start()
        image = writer.register_schema('sensor_msgs/Image', 'jsonschema', b'{}')
        topics = ['/camera/front/image_raw', '/camera/wrist/image_raw', '/camera/front/depth']
        cameras = [writer.register_channel(topic, 'json', image) for topic in topics]
        joint = writer.register_channel('/joint_states', 'json',
            writer.register_schema('sensor_msgs/JointState', 'jsonschema', b'{}'))
        for index in range(6):
            at = 1_700_000_000_000_000_000 + index * 33_333_333
            for camera, channel in enumerate(cameras):
                depth = camera == 2
                pixels = (np.full((48, 64), 1000 + index, '<u2') if depth else
                          np.full((48, 64, 3), index * 30, np.uint8))
                # Padding exercises declared row strides without changing the recorded pixels.
                row_bytes = pixels.view(np.uint8).reshape(48, -1)
                padded = np.pad(row_bytes, ((0, 0), (0, 4)), constant_values=255)
                body = {'width': 64, 'height': 48, 'step': padded.shape[1],
                        'encoding': '16UC1' if depth else 'rgb8', 'is_bigendian': 0,
                        'data': base64.b64encode(padded.tobytes()).decode()}
                if camera == damaged and index == 2:
                    body[field] = value
                writer.add_message(channel, at, json.dumps(body).encode(), at)
            writer.add_message(joint, at, json.dumps({'name': joint_names((None,)),
                'position': [index, 0, 0, 0, 0, 0, 0], 'units': ['rad'] * 6 + ['fraction']}).encode(), at)
        writer.finish()
    return topics


@pytest.mark.parametrize('damaged,field,value', [(None, None, None),
    *[(camera, field, value) for camera in (0, 2) for field, value in
      [('width', 'not-a-width'), ('width', 64.5), ('height', -1),
       ('step', 'not-a-stride'), ('step', 1)]]])
def test_raw_mcap_source_geometry_keeps_neighbor_pixels_and_physical_state(tmp_path, damaged, field, value):
    from label import depth
    source = tmp_path / 'source.mcap'
    topics = raw_camera_mcap(source, damaged, field, value)
    out = tmp_path / 'prepared'
    context = formats.convert_mcap_generic({'file': source, 'name': 'raw', 'seconds': .2,
        'topics': [*topics, '/joint_states']}, 'teleop_arms', out, 'raw source geometry')
    loaded = episode.load(out)
    assert len(context['cameras']) == 2 and context['state_kind'] == 'joints'
    state = loaded['state'][:, 0]
    np.testing.assert_array_equal(state, [0, 1, 3, 4, 5] if damaged == 0 else np.arange(6))
    for view in loaded['sources']:
        pictures = episode._decode_view(loaded, view, list(range(len(state))))
        assert len(pictures) == len(state)
        for row, picture in pictures.items():
            assert np.asarray(picture).mean() == pytest.approx(state[row] * 30, abs=3)
    streams = depth.load(out)
    assert len(streams) == 1
    own = next(iter(streams))
    depth_rows = [0, 1, 3, 4, 5] if damaged == 2 else list(range(6))
    # Exact source log nanoseconds are independent of FFV1's millisecond transport ticks.
    np.testing.assert_array_equal(streams[own]['t'], np.array(depth_rows) * 33_333_333 / 1e9)
    native = depth.decode(streams[own], streams[own]['pts'], list(range(len(depth_rows))))
    assert len(native) == len(depth_rows)
    for row, pixels in native.items():
        np.testing.assert_array_equal(pixels, np.full((48, 64), 1000 + depth_rows[row], np.uint16))
    # The generic depth kmap permits a nearest frame within one period. Verify exact
    # ownership at all anchors that have their own recorded depth frame.
    ks = [row for row in range(len(state)) if not (damaged == 2 and state[row] == 2)]
    readings = depth.at_anchor(loaded, streams, own, ks)
    assert set(readings) == set(ks)
    for row, pixels in readings.items():
        np.testing.assert_array_equal(pixels, np.full((48, 64), 1000 + state[row], np.uint16))
    issues = [issue for issue in context.get('reader_issues', []) if issue['kind'] == 'mcap_message_unreadable']
    assert bool(issues) == (damaged is not None)
    if issues:
        assert issues[0]['signal'] == topics[damaged]


@pytest.mark.parametrize('decoder_name', ['raw_image', 'depth_image'])
def test_raw_mcap_unexpected_pixel_decoder_failure_propagates(tmp_path, monkeypatch, decoder_name):
    source = tmp_path / 'source.mcap'
    topics = raw_camera_mcap(source)
    original, calls = getattr(formats, decoder_name), []
    def faulty_pixels(message):
        calls.append(message)
        if len(calls) == 4:
            raise RuntimeError('pixel decoder implementation regression')
        return original(message)
    monkeypatch.setattr(formats, decoder_name, faulty_pixels)
    with pytest.raises(RuntimeError, match='pixel decoder implementation regression'):
        formats.convert_mcap_generic({'file': source, 'name': 'raw', 'seconds': .2,
            'topics': [*topics, '/joint_states']}, 'teleop_arms', tmp_path / 'out', 'pixel decoder fault')
    assert len(calls) == 4


@pytest.mark.parametrize('ambiguous', [False, True])
def test_packed_missing_prefix_preserves_proven_table_rows_and_original_origin(tmp_path, ambiguous):
    root = tmp_path / 'source'
    paths, _ = lerobot_source(root, length=6, offset=0)
    for path in paths.values():
        writer = formats.FrameWriter(path, '', fine_clock=True)
        for index in range(3, 6):
            writer.add_image(index / 30 + (.005 if ambiguous else 0),
                Image.new('RGB', (96, 72), (index * 30,) * 3))
        assert writer.close() == 3
    _, items = formats.plan(root)
    context = formats.convert_lerobot(items[0], 'teleop_arms', tmp_path / 'out', 'missing prefix')
    loaded = episode.load(tmp_path / 'out' / context['episode_id'])
    pictures = episode._decode_view(loaded, 'exo', [0, 1, 2])
    assert [np.asarray(pictures[k]).mean() for k in range(3)] == pytest.approx([90, 120, 150], abs=3)
    if ambiguous:
        assert context['state_kind'] == 'none' and loaded['state'].shape == (3, 0)
    else:
        assert context['state_kind'] == 'joints'
        np.testing.assert_array_equal(loaded['state'][:, 0], [3, 4, 5])
        np.testing.assert_allclose(loaded['times']['exo'], [.1, 4 / 30, 5 / 30], atol=1e-6)
    assert any(issue['kind'] == 'packed_partial_coverage' for issue in context['reader_issues'])
    assert context.get('recorded_table_fields')
    descriptor = next(field for field in context['table_field_inventory'] if field['name'] == 'observation.state')
    with np.load(loaded['dir'] / context['recorded_table_fields']) as original:
        np.testing.assert_array_equal(original[descriptor['array']][:, 0], np.arange(6))


@pytest.mark.parametrize('proof', ['publish', 'clock_id', 'unproved'])
@pytest.mark.parametrize('camera_delay', [.03, .002])
def test_mcap_faster_text_retains_exact_timing_only_with_recorded_clock_evidence(tmp_path, proof, camera_delay):
    source = tmp_path / 'source.mcap'
    text_recording(source, early=1., late=3.5, camera_delay=camera_delay, text_delay=.001,
                   publish_clock=proof == 'publish', clock_id='fixture-device' if proof == 'clock_id' else None)
    report = formats.convert(source, 'ego_head', tmp_path / 'out', 'faster text', 900)
    assert not report['failed'], report
    context = json.loads((tmp_path / 'out' / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    if proof == 'unproved':
        assert [step['t0'] for step in steps(context)] == [
            round(1.001 - camera_delay, 3), round(2.501 - camera_delay, 3), round(3.501 - camera_delay, 3)]
        assert context['source']['presentation_clock']['basis'] == 'mcap_log_time'
        # "prepare" and "fold the cloth" are sent at the same moment, 1s into a 4s recording: the one text left holds
        # its last three seconds, a timed task prompt and not the recording's instruction (a claim must reach within
        # 5% of a short recording's ends, label/task_claims.py)
        assert 'instruction' not in context and task_prompts(context) == {'fold the cloth': round(1.001 - camera_delay, 3)}
        assert 'unverified' in context['collection_note']
        loaded = episode.load(tmp_path / 'out' / context['episode_id'])
        prompt = episode.build_request(loaded['dir'])['prompt']
        assert 'pick up' in prompt and 'fold the cloth' in prompt and 'arrival' in prompt
    else:
        assert [step['t0'] for step in steps(context)] == [1., 2.5, 3.5]
        assert 'instruction' not in context and task_prompts(context) == {'fold the cloth': 1.}


@pytest.mark.parametrize('bias', [-3., 2.])
def test_mcap_independent_capture_bias_is_visible_context_without_false_footage_placement(tmp_path, bias):
    source = tmp_path / 'source.mcap'
    text_recording(source, early=1., late=3.5, text_offset=bias, publish_clock=True)
    report = formats.convert(source, 'ego_head', tmp_path / 'out', 'biased text clock', 900)
    assert not report['failed'], report
    context = json.loads((tmp_path / 'out' / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    assert [row['t0'] for row in steps(context)] == [1., 2.5, 3.5]
    assert context['source']['presentation_clock']['basis'] == 'mcap_log_time'
    assert 'unverified' in context['collection_note']
    original = context['recorded_metadata']['text_channels']['/task/subtask']
    assert original[0]['capture_ns'] == round((1001 + bias) * 1e9)


def test_mcap_outside_footage_steps_and_changing_tasks_have_visible_diagnostics(tmp_path):
    source = tmp_path / 'source.mcap'
    text_recording(source, early=5., late=6., task_change_time=7., clock_id='fixture-device')
    report = formats.convert(source, 'ego_head', tmp_path / 'out', 'outside footage', 900)
    assert not report['failed'], report
    context = json.loads((tmp_path / 'out' / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    assert not context.get('instruction')
    assert 'after recording' in context['uploader_annotation'] and 'fold the cloth' in context['uploader_annotation']
    assert any(issue['kind'] == 'publisher_text_outside_footage' for issue in context['reader_issues'])


def test_mcap_conflicting_explicit_clock_ids_override_equal_numeric_publication_stamps(tmp_path):
    source = tmp_path / 'source.mcap'
    text_recording(source, clock_id='camera-device', text_clock_id='text-device', publish_clock=True)
    report = formats.convert(source, 'ego_head', tmp_path / 'out', 'conflicting clock IDs', 900)
    assert not report['failed'], report
    context = json.loads((tmp_path / 'out' / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    assert [row['t0'] for row in steps(context)] == [0., 2.5]
    assert context['source']['presentation_clock']['basis'] == 'mcap_log_time'
    assert context['recorded_metadata']['text_channels']['/task/subtask'][0]['capture_clock_id'] == 'text-device'
    assert 'unverified' in context['collection_note']


@pytest.mark.parametrize('mixed', [False, True])
def test_partial_packed_camera_never_reassigns_the_healthy_neighbor_rows(tmp_path, mixed):
    root = tmp_path / 'source'
    paths, _ = lerobot_source(root, length=6, offset=0, mixed=mixed)
    writer = formats.FrameWriter(paths['observation.images.front'], '', fine_clock=True)
    for index in range(3, 6):
        writer.add_image(index / 30, Image.new('RGB', (96, 72), (index * 30,) * 3))
    assert writer.close() == 3
    _, items = formats.plan(root)
    context = formats.convert_lerobot(items[0], 'teleop_arms', tmp_path / 'out', 'partial neighbor')
    loaded = episode.load(tmp_path / 'out' / context['episode_id'])
    assert context['state_kind'] == 'joints'
    np.testing.assert_array_equal(loaded['state'][:, 0], np.arange(6))
    assert 'observation.images.front' not in [camera['key'] for camera in context['cameras'].values()]
    pictures = episode._decode_view(loaded, next(iter(loaded['sources'])), list(range(6)))
    assert len(pictures) == 6
    for row, picture in pictures.items():
        assert np.asarray(picture).mean() == pytest.approx(row * 17, abs=3)
    if mixed:
        retained = next(camera for camera in context['unshown_cameras'] if camera['name'] == 'observation.images.front')
        with np.load(loaded['dir'] / retained['camera_times']) as clock:
            np.testing.assert_allclose(clock['capture'], [.1, 4 / 30, 5 / 30], atol=1e-6)


def test_native_remux_rejects_empty_and_batched_messages_even_when_total_counts_match(tmp_path):
    from prepare.remux import RemuxSourceError, remux
    from test_prepare import _h264_frames
    frames = _h264_frames(6)
    good = tmp_path / 'good.mp4'
    remux(frames, np.arange(6) / 30, 'h264', good)
    assert len(decode(good)) == 6
    ambiguous = [frames[0], frames[1], b'', frames[2] + frames[3], frames[4], frames[5]]
    with pytest.raises(RemuxSourceError, match='ownership'):
        remux(ambiguous, np.arange(6) / 30, 'h264', tmp_path / 'bad.mp4')


def test_generic_camera_encoder_implementation_error_propagates(tmp_path, monkeypatch):
    source = tmp_path / 'source.mcap'
    text_recording(source)
    _, items = formats.plan(tmp_path)
    def broken_encode(*args):
        raise RuntimeError('camera encoder implementation regression')
    monkeypatch.setattr(formats.FrameWriter, '_encode', broken_encode)
    with pytest.raises(RuntimeError, match='camera encoder implementation regression'):
        formats.convert_mcap_generic(items[0], 'ego_head', tmp_path / 'out', 'encoder regression')


def test_matching_clock_ids_do_not_override_backwards_channel_chronology(tmp_path):
    source = tmp_path / 'source.mcap'
    text_recording(source, early=1., late=3.5, clock_id='fixture-device', malformed='backwards_clock')
    report = formats.convert(source, 'ego_head', tmp_path / 'out', 'backwards source stamps', 900)
    assert not report['failed'], report
    context = json.loads((tmp_path / 'out' / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    assert [row['t0'] for row in steps(context)] == [1., 2.5, 3.5]
    assert context['source']['presentation_clock']['basis'] == 'mcap_log_time'
    original = context['recorded_metadata']['text_channels']['/task/subtask']
    assert original[1]['capture_ns'] < original[0]['capture_ns']
    assert any(issue['kind'] == 'recorder_arrival_timing' for issue in context['reader_issues'])


def test_partial_packed_rows_remain_useful_when_embedded_neighbor_is_unreadable(tmp_path):
    root = tmp_path / 'source'
    paths, _ = lerobot_source(root, length=6, offset=0, mixed=True)
    table = root / 'data/chunk-000/file-000.parquet'
    df = pd.read_parquet(table)
    df['observation.images.left_wrist'] = [None] * len(df)
    df.to_parquet(table)
    writer = formats.FrameWriter(paths['observation.images.front'], '', fine_clock=True)
    for index in range(3, 6):
        writer.add_image(index / 30, Image.new('RGB', (96, 72), (index * 30,) * 3))
    assert writer.close() == 3
    _, items = formats.plan(root)
    context = formats.convert_lerobot(items[0], 'teleop_arms', tmp_path / 'out', 'partial fallback')
    loaded = episode.load(tmp_path / 'out' / context['episode_id'])
    assert context['state_kind'] == 'joints'
    np.testing.assert_array_equal(loaded['state'][:, 0], [3, 4, 5])
    assert context['clock_start_s'] == pytest.approx(.1)
    pictures = episode._decode_view(loaded, 'exo', [0, 1, 2])
    assert [np.asarray(pictures[k]).mean() for k in range(3)] == pytest.approx([90, 120, 150], abs=3)
    from label.evidence_access import Access
    access = Access(loaded)
    field = next(field for field in access.inventory() if field['kind'] == 'array' and field['name'] == 'observation.state')
    receipt = access.inspect({'field_id': field['id'], 'start_row': 0, 'end_row': 6, 'columns': [0]})
    assert 'times_s' not in receipt
    assert [row[0] for row in receipt['values']] == list(range(6))


def test_high_rate_partial_packets_do_not_claim_ambiguous_publisher_rows(tmp_path):
    root = tmp_path / 'source'
    paths, _ = lerobot_source(root, length=6, offset=0, fps=10_000)
    for path in paths.values():
        writer = formats.FrameWriter(path, '', fine_clock=True)
        for index in range(3, 6):
            writer.add_image(index / 10_000, Image.new('RGB', (96, 72), (index * 30,) * 3))
        assert writer.close() == 3 and len(decode(path)) == 3
    _, items = formats.plan(root)
    context = formats.convert_lerobot(items[0], 'teleop_arms', tmp_path / 'out', 'ambiguous high-rate rows')
    loaded = episode.load(tmp_path / 'out' / context['episode_id'])
    assert context['state_kind'] == 'none' and loaded['state'].shape == (3, 0)
    pictures = episode._decode_view(loaded, 'exo', [0, 1, 2])
    assert [np.asarray(pictures[k]).mean() for k in range(3)] == pytest.approx([90, 120, 150], abs=3)
    assert context['recorded_state_archive']['row_count'] == 6


def test_a_joint_bias_the_delivery_delays_reveal_is_placed_by_arrival(tmp_path):
    # stamps 1.5 s early on both capture and publication: the text then arrives 1.5 s later after its stamps than
    # the camera's frames do, which a shared clock does not explain, so it is placed on the recorder's arrival clock
    source = tmp_path / 'source.mcap'
    text_recording(source, early=1., late=3.5, text_offset=-1.5, publish_offset=-1.5, publish_clock=True)
    report = formats.convert(source, 'ego_head', tmp_path / 'out', 'declared publication domain', 900)
    assert not report['failed'], report
    context = json.loads((tmp_path / 'out' / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    assert context['source']['presentation_clock']['basis'] == 'mcap_log_time'
    assert any(i['kind'] == 'recorder_arrival_timing' for i in context['reader_issues'])


def test_jointly_biased_capture_and_publication_times_expose_the_declared_domain_assumption(tmp_path):
    # a bias small enough to be a later delivery is kept on the declared domain, with the assumption and its size
    source = tmp_path / 'source.mcap'
    text_recording(source, early=1., late=3.5, text_offset=-.1, publish_offset=-.1, publish_clock=True)
    report = formats.convert(source, 'ego_head', tmp_path / 'out', 'declared publication domain', 900)
    assert not report['failed'], report
    directory = tmp_path / 'out' / report['episodes'][0]['episode_id']
    context = json.loads((directory / 'context.json').read_text())
    assert context['annotation_subtasks']
    assert any(i['kind'] == 'capture_clock_offset_possible' for i in context['reader_issues'])
    proof = context['uploader_notes']['publisher_text_timing']['/task/subtask']
    assert proof['physical_synchronization_verified'] is False
    assert 'publish_time' in proof['basis']
    assert 'jointly biased' in proof['limitation'].lower()
    prompt = episode.build_request(directory)['prompt']
    assert 'jointly biased' in prompt.lower() and 'pick up' in prompt and 'fold the cloth' in prompt
