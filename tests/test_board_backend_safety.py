import io
import json
import os
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from board import build, serve, static
from review import __main__ as review


def test_replaced_media_has_new_identity_and_cached_digest(tmp_path):
    clip = tmp_path / 'x.mp4'
    clip.write_bytes(b'aaaa')
    before = clip.stat()
    first = static.src_sig(clip)
    clip.write_bytes(b'bbbb')
    os.utime(clip, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert static.src_sig(clip) != first
    with patch.object(Path, 'open', side_effect=AssertionError('unchanged media rehashed')):
        assert static.src_sig(clip) == static.src_sig(clip)


def test_build_identity_includes_comparison_episode_and_media(tmp_path):
    qa = tmp_path / 'qa'
    qa.mkdir()
    (qa / 'episode_x.json').write_text('{}')
    compare = tmp_path / 'compare'
    (compare / 'model').mkdir(parents=True)
    f = compare / 'model/episode_x.json'
    f.write_text('{}')
    first = static.build_id_for(qa, ['episode_x.json'], compare)
    f.write_text('{"changed": true}')
    assert static.build_id_for(qa, ['episode_x.json'], compare) != first
    assert static.build_id_for(qa, ['episode_x.json'], media=['v/depth/x.a.mp4']) != static.build_id_for(
        qa, ['episode_x.json'], media=['v/depth/x.b.mp4'])


def test_swap_keeps_live_directory_when_exchange_fails(tmp_path, monkeypatch):
    (tmp_path / 'qa').mkdir()
    (tmp_path / 'qa/old').write_text('old')
    (tmp_path / 'qa.new').mkdir()
    (tmp_path / 'qa.new/new').write_text('new')
    monkeypatch.setattr(build, 'exchange_directories', lambda *args: (_ for _ in ()).throw(OSError('failed')))
    with pytest.raises(OSError):
        build._swap(tmp_path, 'qa', True)
    assert (tmp_path / 'qa/old').read_text() == 'old'


def test_dictionary_bad_json_and_export_bad_shapes(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, 'HERE', tmp_path)
    (tmp_path / 'bad.json').write_text('{')
    handler = object.__new__(serve.Handler)
    replies = []
    handler._send = lambda status, value, *args, **kw: replies.append((status, value))
    handler.path = '/api/dictionary?file=bad.json'
    handler._dictionary()
    assert replies[-1][0] == 500
    for body in ([], {'files': 3}, {'files': 'bad.json'}, {'files': [{}]}):
        raw = json.dumps(body).encode()
        handler.path = '/api/export'
        handler.headers = {'Content-Length': str(len(raw))}
        handler.rfile = io.BytesIO(raw)
        handler.do_POST()
        assert replies[-1][0] == 400


def test_disk_cache_enforces_bytes_and_count(tmp_path):
    for i in range(6):
        (tmp_path / f'{i:040x}.jpg').write_bytes(b'x' * 10)
    serve._trim_disk_cache(tmp_path, '.jpg', max_bytes=25, max_files=3)
    assert len(list(tmp_path.iterdir())) <= 2


def test_frame_rejects_nonfinite_and_bad_dimensions(tmp_path, monkeypatch):
    clip = tmp_path / 'x.mp4'
    clip.write_bytes(b'bad')
    monkeypatch.setattr(serve, '_clip_times', lambda p: (_ for _ in ()).throw(AssertionError('probed')))
    for t, w in ((float('nan'), 640), (0, 0), (0, 100000)):
        with pytest.raises(ValueError):
            serve.extract_frame(clip, t, w)


def test_download_has_timeout_and_size_limit(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        headers = {'Content-Length': '100'}
    calls = []
    def urlopen(data, **kwargs):
        calls.append(kwargs)
        return Response(b'x' * 100)
    monkeypatch.setattr(review.urllib.request, 'urlopen', urlopen)
    monkeypatch.setattr(review, 'DOWNLOAD_MAX_BYTES', 10)
    with pytest.raises(ValueError, match='limit'):
        review.fetch('https://example.com/data', tmp_path)
    assert calls[0]['timeout'] > 0
    assert not list((tmp_path / 'upload').rglob('.download-*'))


def test_atomic_swap_succeeds_and_open_reader_keeps_previous_content(tmp_path):
    (tmp_path / 'qa').mkdir()
    (tmp_path / 'qa.new').mkdir()
    (tmp_path / 'qa/value').write_text('old')
    (tmp_path / 'qa.new/value').write_text('new')
    with (tmp_path / 'qa/value').open() as old_reader:
        build._swap(tmp_path, 'qa', True)
        assert old_reader.read() == 'old'
    assert (tmp_path / 'qa/value').read_text() == 'new'
    assert not (tmp_path / 'qa.new').exists()


def test_footage_lock_entries_are_released_after_errors():
    for i in range(100):
        with pytest.raises(ValueError):
            with serve._footage_lock(str(i)):
                raise ValueError('failed')
    assert not serve._FOOTAGE_LOCKS


def test_busy_work_is_rejected_without_waiting():
    slot = threading.BoundedSemaphore(1)
    @serve._bounded_work(slot)
    def work():
        return True
    slot.acquire()
    try:
        with pytest.raises(serve.CacheBusy):
            work()
    finally:
        slot.release()
    assert work()


def test_footage_labels_do_not_read_every_json_again(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, 'HERE', tmp_path)
    for i in range(20):
        (tmp_path / f'prefix_{i}.json').write_text(json.dumps({'_meta': {'episode_id': f'episode_{i}'}}))
    assert serve.footage_label('episode_15')['_meta']['episode_id'] == 'episode_15'
    original = Path.read_text
    reads = []
    def track(path, *args, **kw):
        reads.append(path)
        return original(path, *args, **kw)
    with patch.object(Path, 'read_text', track):
        assert serve.footage_label('episode_15')['_meta']['episode_id'] == 'episode_15'
    assert reads == [tmp_path / 'prefix_15.json']


def test_clip_timeout_cleans_temporary_output(tmp_path, monkeypatch):
    import subprocess
    from board import clips
    monkeypatch.setattr(clips, 'source_size', lambda *args: (16, 16, False))
    def fail(cmd, **kwargs):
        assert kwargs['timeout'] == 600
        Path(cmd[-1]).write_bytes(b'partial')
        raise subprocess.TimeoutExpired(cmd, kwargs['timeout'])
    monkeypatch.setattr(clips.subprocess, 'run', fail)
    with pytest.raises(subprocess.TimeoutExpired):
        clips.extract_one('source', 0, 30, tmp_path / 'out.mp4', 'ffmpeg', 1)
    assert not list(tmp_path.glob('*.mp4'))


def test_hand_pose_full_exits_nonzero_for_failed_episode(tmp_path):
    import ast
    import time
    from types import SimpleNamespace
    path = Path(build.__file__).parent / 'hand_pose/modal_app.py'
    tree = ast.parse(path.read_text())
    full = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'full')
    full.decorator_list = []
    specs = tmp_path / 'specs.json'
    specs.write_text('[{"ds": "d", "ep": "e"}]')
    env = {'json': json, 'Path': Path, 'time': time,
           'ovol': SimpleNamespace(iterdir=lambda *args, **kwargs: []),
           'Ace': lambda: SimpleNamespace(run=SimpleNamespace(map=lambda *args, **kwargs: [ValueError('broken')]))}
    exec(compile(ast.Module(body=[full], type_ignores=[]), str(path), 'exec'), env)
    with pytest.raises(SystemExit) as error:
        env['full'](str(specs))
    assert error.value.code == 1


def test_subsecond_footage_replacement_gets_a_new_clip(tmp_path, monkeypatch):
    import subprocess
    from board import clips
    source = tmp_path / 'clips/episode_x.mp4'
    source.parent.mkdir()
    source.write_bytes(b'old')
    qa = tmp_path / 'qa'
    qa.mkdir()
    monkeypatch.setattr(serve, 'HERE', qa)
    monkeypatch.setattr(serve, 'MP4_DIR', source.parent)
    monkeypatch.setattr(serve, 'FOOTAGE_DIR', tmp_path / 'footage')
    monkeypatch.setattr(serve, 'FFMPEG', 'ffmpeg')
    monkeypatch.setattr(serve, '_probe', lambda p: (16, 16, 1, 30))
    monkeypatch.setattr(clips, 'frame_lengths', lambda p: None)
    calls = []
    def encode(cmd, **kwargs):
        calls.append(cmd)
        Path(cmd[-1]).write_bytes(source.read_bytes())
        return subprocess.CompletedProcess(cmd, 0, '', '')
    monkeypatch.setattr(serve, '_run_footage', encode)
    first, _ = serve.footage('episode_x')
    timestamp = source.stat().st_mtime_ns
    source.write_bytes(b'new')
    os.utime(source, ns=(timestamp, timestamp + 1))
    second, _ = serve.footage('episode_x')
    assert first != second and second.read_bytes() == b'new'
    assert len(calls) == 2 and not serve._FOOTAGE_LOCKS


@pytest.mark.parametrize('failure', ['exit', 'interrupt', 'return', 'stitch', 'board', 'success'])
def test_review_records_label_failure_atomically(tmp_path, monkeypatch, failure):
    import sys
    from board import clips, rules
    from label import dictionary_stage, pieces
    data = tmp_path / 'input'
    data.mkdir()
    job = tmp_path / 'job'
    monkeypatch.setattr(review, 'verified_preparation', lambda *args: (
        {'episodes': [{'episode_id': 'episode_x', 'state_kind': 'none'}]}, {'policies': {}, 'source': {}}))
    monkeypatch.setattr(review, 'source_identity', lambda *args: {})
    monkeypatch.setattr(dictionary_stage, 'prepare', lambda *args, **kwargs: None)
    monkeypatch.setattr(dictionary_stage, 'remaining_label_budget', lambda *args: 1)
    monkeypatch.setattr(dictionary_stage, 'dictionary_spend', lambda *args: {
        'complete': True, 'cost_usd': 0, 'reserved_usd': 0})
    monkeypatch.setattr(dictionary_stage, 'label_spend', lambda *args: 0)
    monkeypatch.setattr(dictionary_stage, 'label_spend_complete', lambda *args: True)
    monkeypatch.setattr(clips, 'set_aside_failed', lambda *args: [])
    monkeypatch.setattr(clips, 'note_camera_problems', lambda *args: None)
    monkeypatch.setattr(pieces, 'write_units', lambda *args: [])
    def stitch(*args):
        assert json.loads((job / 'run/run.json').read_text())['status'] == 'running'
        if failure == 'stitch':
            raise RuntimeError('stitch failed')
        return {'incomplete': [], 'unlabelled': []}
    def board(*args):
        assert json.loads((job / 'run/run.json').read_text())['status'] == 'running'
        assert json.loads((job / 'run_final/run.json').read_text())['status'] == 'running'
        if failure == 'board':
            raise RuntimeError('board failed')
    monkeypatch.setattr(pieces, 'stitch_run', stitch)
    monkeypatch.setattr(build, 'build', board)
    def run_step(job, step, *args, **kwargs):
        if step == 'label':
            if failure == 'exit':
                raise SystemExit('label stopped')
            if failure == 'interrupt':
                raise KeyboardInterrupt()
            return 1 if failure == 'return' else 0
        return 0
    monkeypatch.setattr(review, 'run_step', run_step)
    monkeypatch.setattr(sys, 'argv', ['review', '--data', str(data), '--rig', 'ego_head', '--out', str(job)])
    if failure in ('return', 'success'):
        assert review.main() == int(failure == 'return')
    else:
        error = SystemExit if failure == 'exit' else KeyboardInterrupt if failure == 'interrupt' else RuntimeError
        with pytest.raises(error):
            review.main()
    info = json.loads((job / 'run/run.json').read_text())
    assert info['status'] == ('done' if failure == 'success' else 'interrupted' if failure == 'interrupt' else 'failed')
    assert info['finished_at']
    assert (job / 'preparation_receipt.json').is_file()
    if failure in ('board', 'success'):
        assert json.loads((job / 'run_final/run.json').read_text())['status'] == info['status']


def test_static_depth_change_gets_new_build_and_same_id_rewrite_is_rejected(tmp_path):
    from types import SimpleNamespace
    board = tmp_path / 'board'
    qa = board / 'qa'
    qa.mkdir(parents=True)
    label = {'dataset': 'test', '_rig': 'ego_head', 'duration_s': 1, 'episode_prompt': 'test',
             '_meta': {'episode_id': 'episode_x'}, 'completion': {'task_completed': 'failure'}}
    (qa / 'episode_x.json').write_text(json.dumps(label))
    clips = tmp_path / 'clips'
    (clips / 'depth_exo').mkdir(parents=True)
    (clips / 'episode_x.mp4').write_bytes(b'rgb')
    depth = clips / 'depth_exo/episode_x.mp4'
    depth.write_bytes(b'old')
    args = SimpleNamespace(qa=qa, board=board, clips=clips, compare=None, hands=None, keypoints=None,
                           sensors=None, dictionary_job=None, out=tmp_path / 'site', build_id=None,
                           force=False, title='test', public_base='https://example.com/', header=None)
    assert static.cmd_site(args) == 0
    first = (args.out / 'LATEST').read_text().strip()
    depth.write_bytes(b'new')
    assert static.cmd_site(args) == 0
    second = (args.out / 'LATEST').read_text().strip()
    assert first != second
    old = (args.out / first / 'data/ep/episode_x.json').read_bytes()
    receipt_root = args.out.parent / '.argus-build-receipts'
    receipt_path = next(receipt_root.glob(f'*/{first}.json'))
    original_receipt = receipt_path.read_bytes()
    label['episode_prompt'] = 'changed content'
    (qa / 'episode_x.json').write_text(json.dumps(label))
    args.build_id, args.force = first, True
    with pytest.raises(ValueError, match='existing build id'):
        static.cmd_site(args)
    assert (args.out / first / 'data/ep/episode_x.json').read_bytes() == old
    assert receipt_path.read_bytes() == original_receipt


@pytest.mark.parametrize('failed_stage', ['copy', 'check'])
def test_publish_failure_never_promotes_root_page(tmp_path, failed_stage):
    import subprocess
    root = tmp_path / 'site'
    build_dir = root / 'b1'
    build_dir.mkdir(parents=True)
    (root / 'LATEST').write_text('b1')
    (root / 'media').mkdir()
    (build_dir / 'index.html').write_text('relative page')
    (build_dir / 'index.public.html').write_text('public page')
    (build_dir / 'BUILD.json').write_text(json.dumps({
        'build_id': 'b1', 'episodes': 1, 'public_base': 'https://example.com/', 'missing_media': [],
        'media': {'videos_present': 1, 'videos': 1, 'videos_no_source': 0, 'frames_present': 0, 'frames': 0}}))
    fake = tmp_path / 'rclone'
    log = tmp_path / 'calls.jsonl'
    fake.write_text('#!' + os.sys.executable + '\n' +
        'import json, os, sys\n'
        'with open(os.environ["CALL_LOG"], "a") as log: log.write(json.dumps(sys.argv[1:]) + "\\n")\n'
        'args = sys.argv[1:]\n'
        'failure = os.environ["FAIL_STAGE"]\n'
        'sys.exit(1 if failure in args and (failure == "check" or "--immutable" in args) else 0)\n')
    fake.chmod(0o755)
    script = Path(build.__file__).parent / 'publish.sh'
    result = subprocess.run(['bash', str(script), str(root)], capture_output=True, text=True, timeout=10,
                            env={**os.environ, 'DEST': str(tmp_path / 'remote'), 'RCLONE': str(fake),
                                 'CALL_LOG': str(log), 'FAIL_STAGE': failed_stage})
    assert result.returncode != 0
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert not any('copyto' in args for args in calls)
    data_copy = next(args for args in calls if 'copy' in args and '--immutable' in args)
    assert '--checksum' in data_copy and '--ignore-existing' not in data_copy


def test_real_footage_encode_respects_budget_and_leaves_no_partial_files(tmp_path, monkeypatch):
    import av
    import subprocess
    from board import clips
    ffmpeg = clips.find_ffmpeg()
    if not ffmpeg:
        pytest.skip('ffmpeg is unavailable')
    source = tmp_path / 'clips/episode_smoke.mp4'
    source.parent.mkdir()
    qa = tmp_path / 'qa'
    qa.mkdir()
    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'lavfi', '-i',
                    'testsrc=size=64x48:rate=30', '-t', '2', '-pix_fmt', 'yuv420p', str(source)],
                   check=True, timeout=10)
    monkeypatch.setattr(serve, 'HERE', qa)
    monkeypatch.setattr(serve, 'MP4_DIR', source.parent)
    monkeypatch.setattr(serve, 'FOOTAGE_DIR', tmp_path / 'footage')
    monkeypatch.setattr(serve, 'FFMPEG', ffmpeg)
    out, _ = serve.footage('episode_smoke', 0.2, 1.8)
    with av.open(str(out)) as encoded:
        assert len(list(encoded.decode(video=0))) >= 45
    assert len(serve.extract_frame(source, 0.5, 64)) > 100
    monkeypatch.setattr(serve, '_FOOTAGE_DISK_BYTES', 10)
    with pytest.raises(ValueError, match='budget'):
        serve.footage('episode_smoke', 0.1, 1.1)
    assert out.exists() and not list(serve.FOOTAGE_DIR.glob('*.part.mp4'))
    assert not serve._FOOTAGE_LOCKS


def test_explicit_static_worker_pool_can_wait_for_a_media_slot():
    slot = threading.BoundedSemaphore(1)
    @serve._bounded_work(slot)
    def work():
        return True
    slot.acquire()
    release = threading.Timer(0.02, slot.release)
    release.start()
    try:
        assert work(_wait=True)
    finally:
        release.join()


@pytest.mark.parametrize('disk_only', [False, True])
def test_http_cached_frames_bypass_full_encode_admission(tmp_path, monkeypatch, disk_only):
    import concurrent.futures
    import socketserver
    import subprocess
    import urllib.error
    import urllib.request
    source = tmp_path / 'episode_x.mp4'
    source.write_bytes(b'fixture')
    monkeypatch.setattr(serve, 'MP4_DIR', tmp_path)
    monkeypatch.setattr(serve, 'FFMPEG', 'ffmpeg-fixture')
    monkeypatch.setattr(serve, '_FRAME_CACHE', serve.OrderedDict())
    monkeypatch.setattr(serve, '_FRAME_DIR', tmp_path / 'frames')
    monkeypatch.setattr(serve, '_clip_times', lambda path: [])
    release, saturated = threading.Event(), threading.Event()
    lock, active = threading.Lock(), [0]
    block = [False]
    def encode(cmd, **kwargs):
        if block[0]:
            with lock:
                active[0] += 1
                if active[0] == 4:
                    saturated.set()
            assert release.wait(5)
        return subprocess.CompletedProcess(cmd, 0, b'cached-jpeg', b'')
    monkeypatch.setattr(serve.subprocess, 'run', encode)
    assert serve.extract_frame(source, 0.5, 64) == b'cached-jpeg'
    if disk_only:
        serve._FRAME_CACHE.clear()
    scans = []
    monkeypatch.setattr(serve, '_trim_disk_cache', lambda *args, **kwargs: scans.append(True))
    server = socketserver.ThreadingTCPServer(('127.0.0.1', 0), serve.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_address[1]}/api/frame?id=episode_x&w=64&t='
    def get(t):
        try:
            with urllib.request.urlopen(base + str(t), timeout=3) as response:
                return response.status, response.read(), dict(response.headers)
        except urllib.error.HTTPError as error:
            return error.code, error.read(), dict(error.headers)
    block[0] = True
    try:
        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            workers = [pool.submit(get, t) for t in (1, 2, 3, 4)]
            try:
                assert saturated.wait(2)
                status, data, _ = get(0.5)
                assert status == 200 and data == b'cached-jpeg'
                assert not scans
                status, _, headers = get(5)
                assert status == 503 and headers['Retry-After'] == '1'
                assert 'no-store' in headers['Cache-Control']
            finally:
                release.set()
            assert all(worker.result()[0] == 200 for worker in workers)
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join()


def test_disk_trim_only_removes_owned_cache_files(tmp_path):
    unrelated = [tmp_path / 'operator_video.mp4', tmp_path / 'operator.partial-not-a-cache-file']
    for path in unrelated:
        path.write_bytes(b'operator data')
        os.utime(path, (1, 1))
    cached = tmp_path / ('a' * 20 + '.mp4')
    cached.write_bytes(b'cache')
    partial = tmp_path / ('b' * 20 + '.123.456.part.mp4')
    partial.write_bytes(b'partial')
    os.utime(partial, (1, 1))
    serve._trim_disk_cache(tmp_path, '.mp4', max_bytes=0, max_files=0)
    assert all(path.read_bytes() == b'operator data' for path in unrelated)
    assert not cached.exists() and not partial.exists()
