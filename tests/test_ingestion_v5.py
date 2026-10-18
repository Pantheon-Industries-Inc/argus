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


def encoded_video(path, count, fps=30, missing=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    if missing:
        writer = formats.FrameWriter(path, '', fine_clock=True)
        for index in range(count):
            if index not in missing:
                writer.add_image(index / fps, Image.new('RGB', (96, 72), ((index * 17) % 240,) * 3))
        assert writer.close() == count - len(missing)
        return
    rate = Fraction(fps).limit_denominator(1_000_000)
    with av.open(str(path), 'w') as output:
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


@pytest.mark.parametrize('text_clock', ['capture', 'arrival_only', 'different_capture'])
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
        camera = writer.register_channel('/camera/front/compressed', 'json', schema)
        text_schema = writer.register_schema('TaskText', 'jsonschema', b'{}')
        task = writer.register_channel('/task', 'json', text_schema)
        step = writer.register_channel('/task/subtask', 'json', text_schema)
        for index in range(120):
            data = {'header': header(camera_base + index / 30), 'format': 'png',
                    'data': base64.b64encode(picture.getvalue()).decode()}
            writer.add_message(camera, round((wall + index / 30) * 1e9), json.dumps(data).encode(), 0)
        for channel, when, text in [(task, 1., 'fold the cloth'), (step, 1., 'pick up'), (step, 2.5, 'fold')]:
            data = {'data': text}
            if text_clock != 'arrival_only':
                data['header'] = header((camera_base if text_clock == 'capture' else wall) + when)
            writer.add_message(channel, round((wall + when + 1) * 1e9), json.dumps(data).encode(), 0)
        writer.finish()
    out = tmp_path / 'out'
    report = formats.convert(source, 'ego_head', out, 'clock fixture', 900)
    assert not report['failed'], report
    ctx = json.loads((out / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    if text_clock == 'capture':
        assert [row['t0'] for row in ctx['annotation_subtasks']] == [1., 2.5]
        assert ctx['instruction'] == 'fold the cloth'
    else:
        assert not ctx.get('annotation_subtasks')
        assert any(issue['kind'] == 'text_clock_unverified' for issue in ctx['reader_issues'])
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
