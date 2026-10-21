"""Publication recovery, filesystem ABI, and footage admission regressions."""
import concurrent.futures
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

import pytest

from board import build, clips, serve, static


def test_portable_publication_survives_appledouble_disappearing_during_cleanup(tmp_path, monkeypatch):
    old, new = tmp_path / 'qa', tmp_path / 'qa.new'
    old.mkdir()
    new.mkdir()
    (old / 'a.json').write_text('old')
    (new / 'a.json').write_text('new')
    monkeypatch.setattr(build, 'exchange_directories', lambda *args: (_ for _ in ()).throw(
        OSError(errno.ENOTSUP, 'unsupported')))
    copytree = build.shutil.copytree
    def backup(*args, **kwargs):
        result = copytree(*args, **kwargs)
        (Path(args[1]) / '._a.json').write_bytes(b'AppleDouble')
        return result
    monkeypatch.setattr(build.shutil, 'copytree', backup)
    unlink = os.unlink
    def remove(path, *args, **kwargs):
        unlink(path, *args, **kwargs)
        if str(path).endswith('a.json') and not str(path).endswith('._a.json'):
            try:
                unlink('._a.json' if kwargs.get('dir_fd') is not None else Path(path).with_name('._a.json'),
                       *args, **kwargs)
            except FileNotFoundError:
                pass
    monkeypatch.setattr(os, 'unlink', remove)
    build._swap(tmp_path, 'qa', True)
    assert (old / 'a.json').read_text() == 'new'
    assert not new.exists() and not list(tmp_path.glob('.*rollback-*'))


def test_next_publication_recovers_a_killed_install_without_deleting_unknown_backup(tmp_path):
    old, new = tmp_path / 'qa', tmp_path / 'qa.new'
    old.mkdir()
    new.mkdir()
    for root, text in ((old, 'old'), (new, 'new')):
        (root / 'a.json').write_text(text)
        (root / 'b.json').write_text(text)
    script = '''
import errno, os, sys
from pathlib import Path
from board import build
root = Path(sys.argv[1])
build.exchange_directories = lambda *args: (_ for _ in ()).throw(OSError(errno.ENOTSUP, 'unsupported'))
original = os.replace
def replace(source, destination):
    result = original(source, destination)
    if Path(destination) == root / 'qa/a.json': os._exit(91)
    return result
os.replace = replace
build.publish_directory(root / 'qa', root / 'qa.new')
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)], timeout=5)
    assert result.returncode == 91
    assert (old / 'a.json').read_text() == 'new' and (old / 'b.json').read_text() == 'old'
    orphan = tmp_path / '.qa.rollback-operator'
    orphan.mkdir()
    (orphan / 'only-copy').write_text('precious')
    # A fresh build can no longer produce this old generation. Recovery must
    # restore it before any new publication or cleanup is attempted.
    build.recover_publications(tmp_path, 'qa')
    assert (old / 'a.json').read_text() == (old / 'b.json').read_text() == 'old'
    assert (orphan / 'only-copy').read_text() == 'precious'
    assert list(tmp_path.glob('.qa.rollback-*')) == [orphan]


@pytest.mark.skipif(sys.platform != 'darwin', reason='Darwin ABI test')
def test_durable_digest_uses_the_native_and_intel_statfs_abi(tmp_path):
    import ast
    source = tmp_path / 'source'
    source.write_bytes(b'content')
    # Execute the production function with only stdlib imports, so the system's
    # universal Python can exercise the Intel libc symbol under Rosetta too.
    from label import content_identity
    tree = ast.parse(Path(content_identity.__file__).read_text())
    functions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and
                 n.name in ('durable_metadata', '_DarwinStatFS')]
    script = 'import os, sys, ctypes\n' + ast.unparse(ast.Module(body=functions, type_ignores=[])) + '\n'
    script += "p=sys.argv[1]; s=os.stat(p); print(durable_metadata(p,(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)))\n"
    probe = tmp_path / 'probe.py'
    probe.write_text(script)
    for command in ([sys.executable], ['/usr/bin/arch', '-x86_64', '/usr/bin/python3']):
        if not Path(command[-1]).exists():
            continue
        result = subprocess.run([*command, str(probe), str(source)], capture_output=True, text=True, timeout=10)
        if 'Bad CPU type' in result.stderr:
            continue
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == 'True', result.stderr


@pytest.mark.parametrize('weak', [False, True])
def test_http_probe_admission_bounds_distinct_and_duplicate_misses_and_keeps_ready_downloads(tmp_path, monkeypatch, weak):
    held_stage = 'probe'
    source = tmp_path / 'clips'
    source.mkdir()
    for name in ('ready', 'cold', 'other'):
        (source / f'episode_{name}.mp4').write_bytes(b'fixture')
    monkeypatch.setattr(serve, 'HERE', tmp_path / 'qa')
    monkeypatch.setattr(serve, 'MP4_DIR', source)
    monkeypatch.setattr(serve, 'FOOTAGE_DIR', tmp_path / 'footage')
    monkeypatch.setattr(serve, 'FFMPEG', 'ffmpeg-fixture')
    monkeypatch.setattr(clips, 'frame_lengths', lambda p: None)
    serve._probe_versioned.cache_clear()
    entered, release = threading.Event(), threading.Event()
    lock, probes = threading.Lock(), []
    def run(cmd, **kwargs):
        if '-show_entries' in cmd:
            path = next((Path(reader.name) for reader, _ in getattr(serve._MEDIA_READERS, 'active', [])
                         if Path(cmd[-1]).name == str(reader.fileno())), Path(cmd[-1]))
            with lock:
                probes.append(path.name)
            if held_stage == 'probe' and path.name == 'episode_cold.mp4':
                entered.set()
                assert release.wait(5)
            return subprocess.CompletedProcess(cmd, 0,
                '{"streams":[{"width":16,"height":16,"avg_frame_rate":"30/1"}],"format":{"duration":"4"}}', '')
        Path(cmd[-1]).write_bytes(b'prepared download')
        return subprocess.CompletedProcess(cmd, 0, '', '')
    # Packet demux is also part of the bounded production probe.
    class Container:
        streams = type('Streams', (), {'video': [None]})()
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def demux(self, stream): return iter(())
    import av
    monkeypatch.setattr(av, 'open', lambda *args, **kwargs: Container())
    monkeypatch.setattr(serve.subprocess, 'run', run)
    monkeypatch.setattr(serve, '_run_footage', run)
    if weak:
        # a weak filesystem takes the same path: identity from stat, nothing hashed or queued
        durable = static._durable_metadata
        monkeypatch.setattr(static, '_durable_metadata', lambda path, version:
            False if Path(path).name in ('episode_cold.mp4', 'episode_other.mp4') else durable(path, version))
    ready, _ = serve.footage('episode_ready', 0, 1)
    with socketserver.ThreadingTCPServer(('127.0.0.1', 0), serve.Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_address[1]}/api/footage?id='
        def get(name):
            try:
                with urllib.request.urlopen(base + 'episode_' + name + '&t0=0&t1=1', timeout=3) as response:
                    return response.status, response.read()
            except urllib.error.HTTPError as error:
                return error.code, error.read()
        try:
            with concurrent.futures.ThreadPoolExecutor(3) as pool:
                worker = pool.submit(get, 'cold')
                assert entered.wait(2)
                duplicate = pool.submit(get, 'cold')
                other = pool.submit(get, 'other')
                try:
                    assert duplicate.result()[0] == 503
                    assert other.result()[0] == 503
                    assert get('ready') == (200, ready.read_bytes())
                    assert probes == ['episode_ready.mp4', 'episode_cold.mp4']
                finally:
                    release.set()
                assert worker.result()[0] == 200
        finally:
            release.set()
            server.shutdown()
            thread.join()
    serve._probe_versioned.cache_clear()


def test_published_cleanup_interruption_does_not_leave_a_recovery_only_orphan(tmp_path):
    old, new = tmp_path / 'qa', tmp_path / 'qa.new'
    old.mkdir()
    new.mkdir()
    (old / 'a.json').write_text('old')
    (new / 'a.json').write_text('new')
    script = '''
import errno, os, sys
from pathlib import Path
from board import build
root = Path(sys.argv[1])
build.exchange_directories = lambda *args: (_ for _ in ()).throw(OSError(errno.ENOTSUP, 'unsupported'))
original_scan = os.scandir
class Scan:
    def __init__(self, path):
        with original_scan(path) as entries:
            self.entries = iter(sorted(entries, key=lambda entry: entry.name != 'publication.json'))
    def __iter__(self): return self
    def __next__(self): return next(self.entries)
    def __enter__(self): return self
    def __exit__(self, *args): pass
os.scandir = Scan
original = os.unlink
def unlink(path, *args, **kwargs):
    result = original(path, *args, **kwargs)
    if str(path).endswith('a.json'): os._exit(92)
    return result
os.unlink = unlink
build.publish_directory(root / 'qa', root / 'qa.new')
'''
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)], timeout=5)
    assert result.returncode == 92
    build.recover_publications(tmp_path, 'qa')
    assert (old / 'a.json').read_text() == 'new'
    assert not list(tmp_path.glob('.qa.rollback-*'))


@pytest.mark.parametrize('probe_slots', [1, 2])
def test_real_http_footage_probes_have_finite_processes_and_no_duplicate_work(tmp_path, monkeypatch, probe_slots):
    import shutil
    ffmpeg = clips.find_ffmpeg()
    ffprobe = shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        pytest.skip('ffmpeg and ffprobe are required')
    source = tmp_path / 'clips'
    source.mkdir()
    fixture = Path(__file__).parent / 'fixtures/clip_without_frame_lengths.mp4'
    for name in ('ready', 'cold', 'other'):
        shutil.copyfile(fixture, source / f'episode_{name}.mp4')
    programs = tmp_path / 'bin'
    programs.mkdir()
    (programs / 'ffmpeg').symlink_to(ffmpeg)
    calls, entered, release = tmp_path / 'calls', tmp_path / 'entered', tmp_path / 'release'
    wrapper = programs / 'ffprobe'
    wrapper.write_text('#!' + sys.executable + '\n' +
        'import os, pathlib, sys, time\n'
        f'calls=pathlib.Path({str(calls)!r}); entered=pathlib.Path({str(entered)!r}); release=pathlib.Path({str(release)!r})\n'
        'path=sys.argv[-1]\n'
        'if path.startswith("/dev/fd/"):\n'
        '    fd=int(path.rsplit("/",1)[1])\n'
        '    if sys.platform == "darwin":\n'
        '        import fcntl\n'
        '        path=fcntl.fcntl(fd,50,b"\\0"*1024).split(b"\\0")[0].decode()\n'
        '    else: path=os.readlink(path)\n'
        'with calls.open("a") as out: out.write(pathlib.Path(path).name + "\\n")\n'
        'if path.endswith("episode_cold.mp4"):\n'
        '    entered.touch()\n'
        '    deadline=time.monotonic()+8\n'
        '    while not release.exists() and time.monotonic()<deadline: time.sleep(.01)\n'
        f'os.execv({ffprobe!r}, [{ffprobe!r}, *sys.argv[1:]])\n')
    wrapper.chmod(0o755)
    # the install in use is the one $FFMPEG names, and its ffprobe the one beside it (prepare/tools.py)
    monkeypatch.setenv('FFMPEG', str(programs / 'ffmpeg'))
    monkeypatch.delenv('FFPROBE', raising=False)
    monkeypatch.setattr(serve, 'FFMPEG', str(programs / 'ffmpeg'))
    monkeypatch.setattr(serve, 'HERE', tmp_path / 'qa')
    monkeypatch.setattr(serve, 'MP4_DIR', source)
    monkeypatch.setattr(serve, 'FOOTAGE_DIR', tmp_path / 'footage')
    # Two slots exercise duplicate rejection independently of saturation.
    monkeypatch.setattr(serve, '_FOOTAGE_PROBE_SEM', threading.BoundedSemaphore(probe_slots))
    serve._probe_versioned.cache_clear()
    ready, _ = serve.footage('episode_ready', 0, .1)
    serve._probe_versioned.cache_clear()
    with socketserver.ThreadingTCPServer(('127.0.0.1', 0), serve.Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_address[1]}/api/footage?id=episode_'
        def get(name):
            try:
                with urllib.request.urlopen(base + name + '&t0=0&t1=0.1', timeout=5) as response:
                    return response.status, response.read()
            except urllib.error.HTTPError as error:
                return error.code, error.read()
        try:
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                worker = pool.submit(get, 'cold')
                import time
                deadline = time.monotonic() + 3
                while not entered.exists() and time.monotonic() < deadline:
                    threading.Event().wait(.01)
                assert entered.exists()
                try:
                    assert get('cold')[0] == 503
                    # A spare slot admits a distinct source; saturation rejects it.
                    assert get('other')[0] == (200 if probe_slots == 2 else 503)
                    assert get('ready') == (200, ready.read_bytes())
                    expected = ['episode_ready.mp4', 'episode_cold.mp4']
                    if probe_slots == 2:
                        expected.append('episode_other.mp4')
                    assert calls.read_text().splitlines() == expected
                finally:
                    release.touch()
                assert worker.result()[0] == 200
        finally:
            release.touch()
            server.shutdown()
            thread.join()
    assert not serve._PROBE_ACTIVE and not serve._FOOTAGE_LOCKS
    serve._probe_versioned.cache_clear()


@pytest.mark.skipif(not os.environ.get('ARGUS_EXFAT_TEST_ROOT'), reason='requires a scratch exFAT mount')
def test_real_exfat_multi_component_build_and_cleanup(tmp_path):
    import shutil
    import tempfile
    from test_board_serve import _built_board
    original = _built_board(tmp_path)
    with tempfile.TemporaryDirectory(dir=os.environ['ARGUS_EXFAT_TEST_ROOT'], prefix='publication-test-') as folder:
        board = Path(folder) / 'board'
        shutil.copytree(original, board)
        for path in board.rglob('*'):
            if path.is_file() and not path.name.startswith('._'):
                subprocess.run(['/usr/bin/xattr', '-w', 'com.apple.test.argus-v3', 'test', str(path)], check=True)
        manifest = json.loads((board / 'manifest.json').read_text())
        manifest['board'] = 'generation-two'
        (board / 'manifest.json').write_text(json.dumps(manifest))
        for entry in manifest['datasets']:
            for path in (Path(entry['run']) / 'out').glob('*.json'):
                output = json.loads(path.read_text())
                if output.get('labels'):
                    output['labels']['task_summary'] = 'generation-two'
                path.write_text(json.dumps(output))
        build.build(board)
        assert json.loads((board / 'BUILT.json').read_text())['manifest']['board'] == 'generation-two'
        assert json.loads((board / 'qa/episode_000000.json').read_text())['episode_prompt'] == 'generation-two'
        assert (board / 'compare/metrics.json').is_file() and (board / 'sensors/index.json').is_file()
        for name in ('qa', 'compare', 'hands', 'hand_keypoints', 'sensors'):
            old, new = board / name, board / (name + '.new')
            old.mkdir(exist_ok=True)
            new.mkdir()
            (old / 'old.json').write_text('old')
            (new / 'new.json').write_text('new')
            build._swap(board, name, True)
            assert (old / 'new.json').read_text() == 'new'
            assert not (old / 'old.json').exists() and not new.exists()
        assert not list(board.glob('.*rollback-*'))
        # TemporaryDirectory itself uses pre-3.13 rmtree, which has the same
        # AppleDouble race. Remove our derived files with the repaired helper.
        build.remove_tree(board)


def test_publication_journal_write_failure_restores_the_live_generation(tmp_path, monkeypatch):
    old, new = tmp_path / 'qa', tmp_path / 'qa.new'
    old.mkdir()
    new.mkdir()
    (old / 'a.json').write_text('old')
    (new / 'a.json').write_text('new')
    monkeypatch.setattr(build, 'exchange_directories', lambda *args: (_ for _ in ()).throw(
        OSError(errno.ENOTSUP, 'unsupported')))
    write = build.write_atomic
    def fail_commit(path, value, *args, **kwargs):
        if path.name == 'publication.json' and value['phase'] == 'published':
            raise OSError(errno.EIO, 'journal write failure')
        return write(path, value, *args, **kwargs)
    monkeypatch.setattr(build, 'write_atomic', fail_commit)
    with pytest.raises(OSError, match='journal write failure'):
        build.publish_directory(old, new)
    assert (old / 'a.json').read_text() == 'old'
    assert not list(tmp_path.glob('.*rollback-*'))


def test_recovery_preserves_the_complete_backup_when_restore_fails(tmp_path, monkeypatch):
    backup = tmp_path / '.qa.rollback-test'
    (backup / 'previous').mkdir(parents=True)
    (backup / 'previous/a.json').write_text('only old copy')
    (backup / 'publication.json').write_text(json.dumps({
        'format': 'argus-publication/1', 'target': 'qa', 'phase': 'ready'}))
    monkeypatch.setattr(build, '_install_tree', lambda *args: (_ for _ in ()).throw(OSError(errno.EIO, 'restore')))
    with pytest.raises(RuntimeError, match='previous data is preserved'):
        build.recover_publications(tmp_path)
    assert (backup / 'previous/a.json').read_text() == 'only old copy'
    assert json.loads((backup / 'publication.json').read_text())['phase'] == 'ready'


def test_tree_removal_does_not_suppress_permission_errors(tmp_path, monkeypatch):
    target = tmp_path / 'cache'
    target.mkdir()
    (target / 'data').write_text('retained')
    monkeypatch.setattr(os, 'unlink', lambda *args, **kwargs: (_ for _ in ()).throw(
        PermissionError(errno.EACCES, 'denied')))
    with pytest.raises(PermissionError):
        build.remove_tree(target)
    assert (target / 'data').read_text() == 'retained'


def test_recovery_ignores_appledouble_metadata_files(tmp_path, capsys):
    (tmp_path / '._.qa.rollback-previous').write_bytes(b'AppleDouble metadata')
    build.recover_publications(tmp_path)
    assert capsys.readouterr().err == ''


@pytest.mark.parametrize('blocked', ['backup', 'staging'])
def test_successful_publication_does_not_abort_for_deferred_cleanup(tmp_path, monkeypatch, blocked):
    old, new = tmp_path / 'qa', tmp_path / 'qa.new'
    old.mkdir()
    new.mkdir()
    (old / 'a.json').write_text('old')
    (new / 'a.json').write_text('new')
    monkeypatch.setattr(build, 'exchange_directories', lambda *args: (_ for _ in ()).throw(
        OSError(errno.ENOTSUP, 'unsupported')))
    remove = build.remove_tree
    def denied(path):
        if ((blocked == 'backup' and '.rollback-' in str(path)) or
                (blocked == 'staging' and path == new)):
            raise PermissionError(errno.EACCES, 'cleanup denied')
        return remove(path)
    monkeypatch.setattr(build, 'remove_tree', denied)
    build._swap(tmp_path, 'qa', True)
    assert (old / 'a.json').read_text() == 'new'
    if blocked == 'backup':
        backup = next(tmp_path.glob('.qa.rollback-*'))
        assert json.loads((backup / 'publication.json').read_text())['phase'] == 'published'
        assert (backup / 'previous/a.json').read_text() == 'old'
    monkeypatch.setattr(build, 'remove_tree', remove)
    build.recover_publications(tmp_path)
    assert (old / 'a.json').read_text() == 'new'
    assert not list(tmp_path.glob('.*rollback-*'))


@pytest.mark.parametrize('mutation', ['source', 'source_removed', 'output', 'output_removed'])
def test_ready_receipt_survives_restart_and_rejects_changed_media(tmp_path, monkeypatch, mutation):
    import shutil
    ffmpeg = clips.find_ffmpeg()
    if not ffmpeg:
        pytest.skip('ffmpeg is required')
    source = tmp_path / 'clips/episode_x.mp4'
    source.parent.mkdir()
    shutil.copyfile(Path(__file__).parent / 'fixtures/clip_without_frame_lengths.mp4', source)
    monkeypatch.setattr(serve, 'FFMPEG', ffmpeg)
    monkeypatch.setattr(serve, 'HERE', tmp_path / 'qa')
    monkeypatch.setattr(serve, 'MP4_DIR', source.parent)
    monkeypatch.setattr(serve, 'FOOTAGE_DIR', tmp_path / 'footage')
    out, name = serve.footage('episode_x', 0, .1)
    script = '''
import json, sys
from pathlib import Path
from board import serve
root=Path(sys.argv[1])
serve.HERE=root/'qa'; serve.MP4_DIR=root/'clips'; serve.FOOTAGE_DIR=root/'footage'
serve._FOOTAGE_PROBE_SEM.acquire()
result=serve.footage('episode_x', 0, .1)
print(json.dumps([result[0].name,result[1]]))
'''
    restart = subprocess.run([sys.executable, '-c', script, str(tmp_path)],
                             capture_output=True, text=True, timeout=5)
    assert restart.returncode == 0, restart.stderr
    assert json.loads(restart.stdout) == [out.name, name]
    if mutation in ('source', 'output'):
        path = source if mutation == 'source' else out
        previous = path.stat()
        with path.open('r+b') as stream:
            stream.write(b'corrupt!')
        # Size and mtime are insufficient. ctime must reject this same-size edit.
        os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    else:
        (source if mutation == 'source_removed' else out).unlink()
    serve._probe_versioned.cache_clear()
    serve._FOOTAGE_PROBE_SEM.acquire()
    try:
        if mutation == 'source_removed':
            assert serve.footage('episode_x', 0, .1) is None
        else:
            with pytest.raises(serve.CacheBusy):
                serve.footage('episode_x', 0, .1)
    finally:
        serve._FOOTAGE_PROBE_SEM.release()
    if mutation == 'output':
        # Validation rejects this output without deleting outside a publication transaction.
        assert out.exists()


def test_canonical_receipts_follow_media_lifetime_and_unknown_metadata_is_not_trusted(tmp_path, monkeypatch):
    source = tmp_path / 'clips/episode_x.mp4'
    source.parent.mkdir()
    source.write_bytes(b'source')
    monkeypatch.setattr(serve, 'HERE', tmp_path / 'qa')
    monkeypatch.setattr(serve, 'MP4_DIR', source.parent)
    monkeypatch.setattr(serve, 'FOOTAGE_DIR', tmp_path / 'footage')
    monkeypatch.setattr(serve, 'FFMPEG', 'ffmpeg-fixture')
    monkeypatch.setattr(serve, '_FOOTAGE_DISK_FILES', 2)
    monkeypatch.setattr(serve, '_probe', lambda p: (16, 16, 10, 30))
    # This cache-lifetime fixture stubs the container and encoder together.
    monkeypatch.setattr(serve, '_footage_seek', lambda p, t: max(0, t - 3))
    monkeypatch.setattr(clips, 'frame_lengths', lambda p: None)
    def encode(cmd):
        Path(cmd[-1]).write_bytes(b'prepared footage')
        return subprocess.CompletedProcess(cmd, 0, '', '')
    monkeypatch.setattr(serve, '_run_footage', encode)
    for start in range(5):
        serve.footage('episode_x', start, start + 1)
    assert len(list(serve.FOOTAGE_DIR.glob('*.mp4'))) == 2
    assert len([p for p in (serve.FOOTAGE_DIR / '.ready').glob('*.json') if len(p.stem) == 20]) == 2
    from board import static
    monkeypatch.setattr(static, '_durable_metadata', lambda *args: False)
    monkeypatch.setattr(serve, '_probe_uncached', lambda p: (16, 16, 10, 30))
    serve.footage('episode_x', 5, 6)
    assert len([p for p in (serve.FOOTAGE_DIR / '.ready').glob('*.json') if len(p.stem) == 20]) == 2


def test_changed_ready_output_is_never_served_when_cache_cleanup_is_denied(tmp_path, monkeypatch):
    source = tmp_path / 'clips/episode_x.mp4'
    source.parent.mkdir()
    source.write_bytes(b'source')
    monkeypatch.setattr(serve, 'HERE', tmp_path / 'qa')
    monkeypatch.setattr(serve, 'MP4_DIR', source.parent)
    monkeypatch.setattr(serve, 'FOOTAGE_DIR', tmp_path / 'footage')
    monkeypatch.setattr(serve, 'FFMPEG', 'ffmpeg-fixture')
    monkeypatch.setattr(serve, '_probe', lambda p: (16, 16, 10, 30))
    monkeypatch.setattr(clips, 'frame_lengths', lambda p: None)
    def encode(cmd):
        Path(cmd[-1]).write_bytes(b'prepared footage')
        return subprocess.CompletedProcess(cmd, 0, '', '')
    monkeypatch.setattr(serve, '_run_footage', encode)
    out, _ = serve.footage('episode_x', 0, 1)
    out.write_bytes(b'corrupt')
    unlink = Path.unlink
    def deny(path, *args, **kwargs):
        if path == out:
            raise PermissionError(errno.EACCES, 'cache cleanup denied')
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', deny)
    with pytest.raises(RuntimeError, match='changed'):
        serve.footage('episode_x', 0, 1)


def test_untrusted_metadata_never_reuses_a_same_process_digest(tmp_path, monkeypatch):
    import hashlib
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'aaaa')
    st = source.stat()
    version = (st.st_dev, st.st_ino, 4, st.st_mtime_ns // 10**9 * 10**9, st.st_ctime_ns // 10**9 * 10**9)
    monkeypatch.setattr(static, '_source_version', lambda path: version)
    monkeypatch.setattr(static, '_durable_metadata', lambda *args: False)
    static._source_digest.cache_clear()
    assert static.src_sig(source) == hashlib.sha256(b'aaaa').hexdigest()
    source.write_bytes(b'bbbb')
    assert static.src_sig(source) == hashlib.sha256(b'bbbb').hexdigest()


def test_untrusted_metadata_probe_follows_size_and_modification_time(tmp_path, monkeypatch):
    # Suggestion 9: on a weak filesystem a probe is reused while size and modification time match, whatever the
    # unreliable inode and change time say, and redone when either changes.
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'a')
    monkeypatch.setattr(static, '_durable_metadata', lambda *args: False)
    monkeypatch.setattr(serve, '_probe_uncached', lambda path: (path.read_bytes()[0], 16, 4, 30))
    serve._probe_versioned.cache_clear()
    assert serve._probe_versioned(source, (1, 2, 1, 0, 0))[0] == 97
    source.write_bytes(b'b')
    assert serve._probe_versioned(source, (1, 9, 1, 0, 7))[0] == 97
    assert serve._probe_versioned(source, (1, 2, 1, 5, 0))[0] == 98
    source.write_bytes(b'cc')
    assert serve._probe_versioned(source, (1, 2, 2, 5, 0))[0] == 99


def test_untrusted_footage_key_tracks_bytes_when_stat_identity_collides(tmp_path, monkeypatch):
    source = tmp_path / 'clips/episode_x.mp4'
    source.parent.mkdir()
    source.write_bytes(b'aaaa')
    st = source.stat()
    version = (st.st_dev, st.st_ino, 4, st.st_mtime_ns // 10**9 * 10**9, st.st_ctime_ns // 10**9 * 10**9)
    monkeypatch.setattr(static, '_source_version', lambda path: version)
    monkeypatch.setattr(static, '_durable_metadata', lambda *args: False)
    stat = Path.stat
    class CoarseStat:
        def __init__(self, original): self.original = original
        def __getattr__(self, name):
            return dict(zip(('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'), version)).get(
                name, getattr(self.original, name))
    def coarse_stat(path, *args, **kwargs):
        result = stat(path, *args, **kwargs)
        return CoarseStat(result) if path == source else result
    monkeypatch.setattr(Path, 'stat', coarse_stat)
    monkeypatch.setattr(serve, 'HERE', tmp_path / 'qa')
    monkeypatch.setattr(serve, 'MP4_DIR', source.parent)
    monkeypatch.setattr(serve, 'FOOTAGE_DIR', tmp_path / 'footage')
    monkeypatch.setattr(serve, 'FFMPEG', 'ffmpeg-fixture')
    monkeypatch.setattr(serve, '_probe_uncached', lambda p: (16, 16, 4, 30))
    monkeypatch.setattr(clips, 'frame_lengths', lambda p: None)
    def encode(cmd):
        Path(cmd[-1]).write_bytes(source.read_bytes())
        return subprocess.CompletedProcess(cmd, 0, '', '')
    monkeypatch.setattr(serve, '_run_footage', encode)
    first, _ = serve.footage('episode_x', 0, 1)
    assert first.read_bytes() == b'aaaa'
    source.write_bytes(b'bbbb')
    second, _ = serve.footage('episode_x', 0, 1)
    assert second != first and second.read_bytes() == b'bbbb'
    assert serve.footage('episode_x', 0, 1)[0].read_bytes() == b'bbbb'
