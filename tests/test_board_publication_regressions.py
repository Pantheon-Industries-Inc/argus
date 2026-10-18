"""Filesystem, process, and HTTP regressions from the October board reviews."""
import errno
import json
import os
from pathlib import Path
import socketserver
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

from board import build, clips, serve, static


def _site(tmp_path):
    board = tmp_path / 'board'
    qa = board / 'qa'
    qa.mkdir(parents=True)
    (qa / 'episode_x.json').write_text(json.dumps({
        'dataset': 'test', '_rig': 'ego_head', 'duration_s': 1, 'episode_prompt': 'test',
        '_meta': {'episode_id': 'episode_x'}, 'completion': {'task_completed': 'failure'}}))
    media = tmp_path / 'clips'
    media.mkdir()
    return SimpleNamespace(board=board, qa=qa, clips=media, out=tmp_path / 'site', compare=None,
        hands=None, keypoints=None, sensors=None, dictionary_job=None, title='Test', public_base=None,
        build_id=None, force=False, header=None)


def test_static_site_changes_identity_when_export_dependency_changes(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    package = source / 'board'
    package.mkdir(parents=True)
    for path in static.HERE_DIR.glob('*.py'):
        (package / path.name).write_bytes(path.read_bytes())
    for name in ('families.json', 'tag_names.json'):
        (package / name).write_bytes((static.HERE_DIR / name).read_bytes())
    (package / '__init__.py').touch()
    (source / 'prepare').mkdir()
    (source / 'prepare/__init__.py').touch()
    helper = source / 'prepare/formats.py'
    helper.write_text('def annotation_text(value): return str(value)\n')
    monkeypatch.setattr(static, 'HERE_DIR', package)
    a = _site(tmp_path)
    assert static.cmd_site(a) == 0
    first = (a.out / 'LATEST').read_text()
    helper.write_text('def annotation_text(value): return "changed:" + str(value)\n')
    assert static.cmd_site(a) == 0
    assert (a.out / 'LATEST').read_text() != first


def test_ready_footage_is_served_while_another_encode_holds_admission(tmp_path, monkeypatch):
    source = tmp_path / 'clips/episode_x.mp4'
    source.parent.mkdir()
    source.write_bytes(b'source')
    qa = tmp_path / 'qa'
    qa.mkdir()
    monkeypatch.setattr(serve, 'HERE', qa)
    monkeypatch.setattr(serve, 'MP4_DIR', source.parent)
    monkeypatch.setattr(serve, 'FOOTAGE_DIR', tmp_path / 'footage')
    monkeypatch.setattr(serve, 'FFMPEG', 'ffmpeg')
    monkeypatch.setattr(serve, '_probe', lambda p: (16, 16, 4, 30))
    monkeypatch.setattr(clips, 'frame_lengths', lambda p: None)
    entered, release, busy = threading.Event(), threading.Event(), threading.Event()
    def encode(cmd, **kwargs):
        if entered.is_set():
            busy.set()
            assert release.wait(5)
        Path(cmd[-1]).write_bytes(b'ready footage')
        return subprocess.CompletedProcess(cmd, 0, '', '')
    monkeypatch.setattr(serve.subprocess, 'run', encode)
    # Also support the progress-aware process wrapper after the repair.
    monkeypatch.setattr(serve, '_run_footage', encode, raising=False)
    ready, _ = serve.footage('episode_x', 0, 1)
    entered.set()
    errors = []
    def other():
        try:
            serve.footage('episode_x', 1, 3)
        except BaseException as error:
            errors.append(error)
    worker = threading.Thread(target=other)
    worker.start()
    assert busy.wait(3)
    with socketserver.ThreadingTCPServer(('127.0.0.1', 0), serve.Handler) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            url = f'http://127.0.0.1:{server.server_address[1]}/api/footage?id=episode_x&t0=0&t1=1'
            with urllib.request.urlopen(url, timeout=3) as response:
                assert response.status == 200
                assert response.read() == ready.read_bytes()
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(url.replace('t0=0&t1=1', 't0=3&t1=4'), timeout=3)
            assert error.value.code == 503
        finally:
            release.set()
            worker.join(5)
            server.shutdown()
    assert not errors and not worker.is_alive()


def test_whole_footage_over_an_hour_is_encoded_without_span_rejection(tmp_path, monkeypatch):
    source = tmp_path / 'episode_x.mp4'
    source.write_bytes(b'source')
    monkeypatch.setattr(serve, 'MP4_DIR', tmp_path)
    monkeypatch.setattr(serve, 'HERE', tmp_path / 'qa')
    monkeypatch.setattr(serve, 'FOOTAGE_DIR', tmp_path / 'footage')
    monkeypatch.setattr(serve, 'FFMPEG', 'ffmpeg')
    monkeypatch.setattr(serve, '_probe', lambda p: (16, 16, 3601, 30))
    monkeypatch.setattr(clips, 'frame_lengths', lambda p: None)
    def encode(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b'long footage')
        assert '-fs' in cmd and '-threads' in cmd
        return subprocess.CompletedProcess(cmd, 0, '', '')
    monkeypatch.setattr(serve.subprocess, 'run', encode)
    monkeypatch.setattr(serve, '_run_footage', encode, raising=False)
    path, name = serve.footage('episode_x')
    assert path.read_bytes() == b'long footage' and name == 'episode_x.mp4'


def test_digest_survives_process_restart_and_invalidates_restored_mtime(tmp_path):
    source = tmp_path / 'clip.mp4'
    source.write_bytes(b'aaaa')
    script = '''
import builtins, json, os
from board import static
from pathlib import Path
source = Path(os.environ['SOURCE'])
if os.environ.get('DENY_READ'):
    original = builtins.open
    def guarded(path, *args, **kwargs):
        if str(path) == str(source): raise AssertionError('source bytes reread')
        return original(path, *args, **kwargs)
    builtins.open = guarded
print(static.src_sig(source))
'''
    env = {**os.environ, 'SOURCE': str(source), 'BOARD_SOURCE_DIGEST_CACHE': str(tmp_path / 'digests.sqlite3')}
    def signature(deny=False):
        result = subprocess.run([sys.executable, '-c', script], env={**env, **({'DENY_READ': '1'} if deny else {})},
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()
    first = signature()
    assert signature(True) == first
    before = source.stat()
    source.write_bytes(b'bbbb')
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert signature() != first
    assert signature(True) != first


def test_progressing_encode_outlives_stall_timeout_and_stalled_process_is_reaped(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, '_FOOTAGE_STALL_TIMEOUT', 0.12)
    out = tmp_path / 'progress.mp4'
    code = ('import pathlib, sys, time; p=pathlib.Path(sys.argv[-1]); '
            '[(p.write_bytes(b"x"*(i+1)), print("out_time_us="+str(i+1), flush=True), time.sleep(.035)) '
            'for i in range(12)]')
    assert serve._run_footage([sys.executable, '-c', code, str(out)]).returncode == 0
    assert out.stat().st_size == 12
    pid_file = tmp_path / 'pid'
    code = 'import os, pathlib, sys, time; pathlib.Path(sys.argv[-1]).write_text(str(os.getpid())); time.sleep(10)'
    with pytest.raises(subprocess.TimeoutExpired):
        serve._run_footage([sys.executable, '-c', code, str(pid_file)])
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


@pytest.mark.parametrize('fail', [False, True])
def test_portable_swap_keeps_live_paths_and_restores_old_generation_on_error(tmp_path, monkeypatch, fail):
    old, new = tmp_path / 'qa', tmp_path / 'qa.new'
    old.mkdir()
    new.mkdir()
    for folder, value in ((old, 'old'), (new, 'new')):
        (folder / 'a.json').write_text(value + ' a')
        (folder / 'b.json').write_text(value + ' b')
    (old / 'stale.json').write_text('old stale')
    (new / 'added.json').write_text('new added')
    monkeypatch.setattr(build, 'exchange_directories', lambda *args: (_ for _ in ()).throw(
        OSError(errno.ENOTSUP, 'exchange unsupported')))
    original = os.replace
    failed = False
    observed = []
    def replace(source, destination):
        nonlocal failed
        assert old.is_dir()
        observed.append((old / 'a.json').read_text())
        if fail and Path(destination) == old / 'b.json' and not failed:
            failed = True
            raise OSError(errno.EIO, 'injected publication failure')
        return original(source, destination)
    monkeypatch.setattr(os, 'replace', replace)
    with (old / 'a.json').open() as reader:
        if fail:
            with pytest.raises(OSError, match='injected publication failure'):
                build._swap(tmp_path, 'qa', True)
        else:
            build._swap(tmp_path, 'qa', True)
        assert reader.read() == 'old a'
    assert observed
    value = 'old' if fail else 'new'
    assert (old / 'a.json').read_text() == value + ' a'
    assert (old / 'b.json').read_text() == value + ' b'
    assert (old / ('stale.json' if fail else 'added.json')).is_file()
    assert not (old / ('added.json' if fail else 'stale.json')).exists()
    assert not list(tmp_path.glob('.*rollback*'))


def test_static_same_id_force_needs_no_directory_exchange(tmp_path, monkeypatch):
    a = _site(tmp_path)
    assert static.cmd_site(a) == 0
    bid = (a.out / 'LATEST').read_text().strip()
    public = (a.out / bid / 'data/ep/episode_x.json').read_bytes()
    monkeypatch.setattr(build, 'exchange_directories', lambda *args: (_ for _ in ()).throw(
        AssertionError('same public payload must not need directory exchange')))
    a.force = True
    assert static.cmd_site(a) == 0
    assert (a.out / bid / 'data/ep/episode_x.json').read_bytes() == public
    assert not list(a.out.glob('*.staging'))


def test_corrupt_digest_cache_falls_back_to_content_and_unknown_filesystems_rehash(tmp_path, monkeypatch):
    import hashlib
    cache = tmp_path / 'corrupt.sqlite3'
    cache.write_bytes(b'not a database')
    monkeypatch.setenv('BOARD_SOURCE_DIGEST_CACHE', str(cache))
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'actual bytes')
    static._source_digest.cache_clear()
    assert static.src_sig(source) == hashlib.sha256(b'actual bytes').hexdigest()
    # Remote and coarse metadata never certifies durable byte identity.
    monkeypatch.setattr(static, '_durable_metadata', lambda *args: False)
    cache.unlink()
    static._source_digest.cache_clear()
    assert static.src_sig(source) == hashlib.sha256(b'actual bytes').hexdigest()
    assert not cache.exists()


@pytest.mark.parametrize('command', ['build', 'static media', 'static site'])
def test_free_job_board_commands_fail_clearly_without_creating_empty_site(tmp_path, command):
    job = tmp_path / 'free-job'
    (job / 'dry').mkdir(parents=True)
    (job / 'clips').mkdir()
    (job / 'report.json').write_text('{"episodes": []}')
    arguments = ([str(job)] if command == 'build' else ['--board', str(job), '--clips', str(job / 'clips')])
    result = subprocess.run([sys.executable, '-m', 'board', *command.split(), *arguments],
                            text=True, capture_output=True, timeout=10)
    assert result.returncode != 0
    assert 'Traceback' not in result.stderr
    assert 'board' in result.stderr.lower()
    assert not (job / 'static').exists()


def test_custom_static_board_without_manifest_keeps_failure_only_episode(tmp_path):
    board = tmp_path / 'custom'
    (board / 'qa').mkdir(parents=True)
    (board / 'clips').mkdir()
    (board / 'qa/episode_x.json').write_text(json.dumps({'episode_prompt': '',
        '_meta': {'episode_id': 'episode_x'}, '_label_failed': {'kind': 'unparsed', 'why': 'Reply needs reparse'}}))
    result = subprocess.run([sys.executable, '-m', 'board', 'static', 'site', '--board', str(board),
                             '--clips', str(board / 'clips')], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    latest = (board / 'static/LATEST').read_text().strip()
    output = json.loads((board / 'static' / latest / 'data/ep/episode_x.json').read_text())
    assert output['_label_failed'] and not latest.startswith('1969')
