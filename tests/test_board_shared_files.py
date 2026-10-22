"""Several board servers share one footage folder: none acts on its own earlier read of a file another one changes.

Each server here is its own OS process, reached over real HTTP; only ticket lifetimes and cache budgets are shortened,
and the time the board's ticket code reads is one clock the test sets for both servers, so whether a ticket is live
never depends on how fast the machine runs the requests between two steps. The helper tests race a writer process
against a remover process on one file."""
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request

import pytest

from board import serve
from label.atomic import file_version, remove_if, small_file, sweep_removals

REPO = Path(__file__).resolve().parents[1]

SERVER = textwrap.dedent('''
    import sys, threading, types
    import time as real_time
    from pathlib import Path
    sys.path.insert(0, sys.argv[1])
    from board import serve
    root = Path(sys.argv[2])
    serve.HERE, serve.MP4_DIR, serve.FOOTAGE_DIR = root / 'qa', root / 'clips', root / 'footage'
    serve._FOOTAGE_TICKET_SECONDS = float(sys.argv[3])
    serve._FOOTAGE_DISK_FILES = int(sys.argv[4])

    class Clock(types.ModuleType):
        """serve's time module, except that time() is the moment the test last wrote to <root>/clock."""
        def __getattr__(self, name):
            return getattr(real_time, name)

        def time(self):
            return float((root / 'clock').read_text())
    serve.time = Clock('time')
    server = serve.BoardHTTPServer(('127.0.0.1', 0), serve.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(server.server_address[1], flush=True)
    sys.stdin.read()
    server.shutdown()
''')


class Clock:
    """The time both servers' ticket code reads. It stands still between steps and moves only when the test moves it."""

    def __init__(self, root):
        self.path = root / 'clock'
        self.set(time.time())

    def set(self, moment):
        staged = self.path.with_name('clock.next')
        staged.write_text(repr(moment))
        os.replace(staged, self.path)
        self.now = moment

    def advance(self, seconds):
        self.set(self.now + seconds)

    def past(self, moment):
        self.set(max(self.now, moment + .15))


class Servers:
    """Two board server processes on one board and one footage folder, reading one clock."""

    def __init__(self, root, ticket_seconds, disk_files):
        self.clock = Clock(root)
        self.processes = []
        for _ in range(2):
            process = subprocess.Popen([sys.executable, '-c', SERVER, str(REPO), str(root), str(ticket_seconds),
                                        str(disk_files)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            self.processes.append((process, int(process.stdout.readline())))

    def get(self, server, route, client='10.0.0.1'):
        request = urllib.request.Request(f'http://127.0.0.1:{self.processes[server][1]}/{route}',
                                         headers={'X-Forwarded-For': client})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    def prepare(self, server, span, client='10.0.0.1'):
        status, body = self.get(server, f'api/footage?id=episode_x&t0={span[0]}&t1={span[1]}&prepare=1', client)
        return status, (json.loads(body).get('ticket') if status == 200 else None)

    def close(self):
        for process, _ in self.processes:
            process.stdin.close()
            try:
                process.wait(5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(5)


@pytest.fixture
def board(tmp_path):
    if not serve.FFMPEG:
        pytest.skip('ffmpeg is required')
    (tmp_path / 'clips').mkdir()
    (tmp_path / 'qa').mkdir()
    subprocess.run([serve.FFMPEG, '-v', 'error', '-f', 'lavfi', '-i', 'color=c=blue:s=32x32:r=30:d=0.6', '-c:v',
                    'libx264', '-pix_fmt', 'yuv420p', str(tmp_path / 'clips' / 'episode_x.mp4')],
                   check=True, capture_output=True, timeout=10)
    (tmp_path / 'qa' / 'episode_x.json').write_text(json.dumps({
        'episode_prompt': 'test', '_rig': 'ego_head', 'camera_views': ['exo'], '_meta': {'episode_id': 'episode_x'},
        'duration_s': .6}))
    return tmp_path


def expiry(root, token):
    return json.loads((root / 'footage' / '.tickets' / f'{token}.json').read_text())['expires_at']


@pytest.mark.parametrize('later', ['footage', 'other client prepares'])
def test_a_ticket_renewed_by_one_server_survives_the_other_servers_earlier_read(board, later):
    servers = Servers(board, ticket_seconds=2, disk_files=8)
    try:
        status, token = servers.prepare(0, (0, .3))
        assert status == 200
        first = expiry(board, token)
        assert servers.get(1, 'api/footage?id=episode_x&t0=0&t1=0.3')[0] == 200    # server 1 reads the ticket
        time.sleep(.5)                                  # the renewed record gets a new modification time
        servers.clock.advance(.5)
        assert servers.prepare(0, (0, .3)) == (200, token)                          # server 0 renews the same token
        renewed = expiry(board, token)
        assert renewed > first
        servers.clock.past(first)                       # server 1's earlier read has expired; the renewal has not
        if later == 'footage':
            assert servers.get(1, 'api/footage?id=episode_x&t0=0&t1=0.3')[0] == 200
        else:
            assert servers.prepare(1, (.1, .4), client='10.0.0.2')[0] == 200
        assert servers.clock.now < renewed
        assert servers.get(0, f'api/footage?ticket={token}')[0] == 200
        assert servers.get(1, f'api/footage?ticket={token}')[0] == 200
    finally:
        servers.close()


def test_eviction_by_another_server_never_takes_the_movie_of_a_renewed_ticket(board):
    servers = Servers(board, ticket_seconds=2, disk_files=1)
    try:
        status, token = servers.prepare(0, (0, .3))
        assert status == 200
        held = servers.get(0, f'api/footage?ticket={token}')
        first = expiry(board, token)
        assert servers.get(1, 'api/footage?id=episode_x&t0=0&t1=0.3')[0] == 200
        time.sleep(.5)
        servers.clock.advance(.5)
        assert servers.prepare(0, (0, .3)) == (200, token)
        servers.clock.past(first)
        # a second client's span needs the only cache slot, which the renewed ticket still holds
        assert servers.prepare(1, (.1, .4), client='10.0.0.2')[0] == 503
        assert servers.get(1, f'api/footage?ticket={token}') == held
        assert servers.get(0, f'api/footage?ticket={token}') == held
    finally:
        servers.close()


def test_an_expired_ticket_is_still_removed_by_the_other_server(board):
    servers = Servers(board, ticket_seconds=1, disk_files=8)
    try:
        status, token = servers.prepare(0, (0, .3))
        assert status == 200
        servers.clock.past(expiry(board, token))
        assert servers.get(1, 'api/footage?id=episode_x&t0=0&t1=0.3')[0] == 200
        assert not (board / 'footage' / '.tickets' / f'{token}.json').exists()
        assert servers.get(0, f'api/footage?ticket={token}')[0] == 404
    finally:
        servers.close()


RACE = textwrap.dedent('''
    import json, os, sys, time
    sys.path.insert(0, sys.argv[1])
    from label.atomic import remove_if, small_file, write_atomic
    path, role, until, target = sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5])
    done, ready = path + '.writer-done', path + '.remover-ready'
    read = small_file(4096)
    lapsed = lambda snapshot: snapshot[2] is not None and not json.loads(snapshot[2])['live']
    if role == 'remover':
        removed = 0
        open(ready, 'w').close()
        # races the writer for as long as it writes, however fast the machine is
        while not os.path.exists(done) and time.time() < until:
            try:
                removed += remove_if(path, lapsed, read=read)
            except (FileNotFoundError, ValueError):
                pass
        print(removed)
        sys.exit()
    lost = written = 0
    while not os.path.exists(ready) and time.time() < until:      # the writes start once the remover is racing
        time.sleep(.01)
    while written < target and time.time() < until:
        written += 1
        for live in (True, False):
            write_atomic(path, {'seq': written, 'live': live}, indent=None)
            if live:
                # a live version stays at its name; one set aside by a remover that raced the write comes back at once
                for _ in range(200):
                    try:
                        with open(path) as source:
                            if json.load(source)['seq'] >= written:
                                break
                    except (FileNotFoundError, ValueError):
                        pass
                    time.sleep(.0001)
                else:
                    lost += 1
    open(done, 'w').close()
    print(lost, written)
''')


def test_remove_if_never_removes_a_version_written_after_its_read_across_processes(tmp_path):
    # a set number of writes rather than a time budget: a slow runner made 45 writes in 2.5s against a floor of 100
    # and failed with nothing lost; the deadline only stops a hung run
    path, until, target = tmp_path / 'ticket.json', time.time() + 60, 150
    run = lambda role: subprocess.Popen([sys.executable, '-c', RACE, str(REPO), str(path), role, str(until),
                                         str(target)], stdout=subprocess.PIPE, text=True)
    remover, writer = run('remover'), run('writer')
    lost, written = map(int, writer.communicate(timeout=90)[0].split())
    removed = int(remover.communicate(timeout=90)[0])
    assert written == target and removed > 0
    assert lost == 0
    assert not [name for name in os.listdir(tmp_path) if name.endswith('.removing')]


def test_remove_if_puts_back_a_version_that_replaced_the_one_it_read(tmp_path):
    path = tmp_path / 'record.json'
    path.write_text('{"live": false}')
    read = small_file()

    def replaced_while_reading(target):
        seen = read(target)
        if Path(target) == path:                 # another process replaces the file right after this read
            os.replace(Path(tmp_path / 'new'), path) if (tmp_path / 'new').exists() else None
        return seen
    (tmp_path / 'new').write_text('{"live": true, "renewed": 1}')
    lapsed = lambda snapshot: not json.loads(snapshot[2])['live']
    assert not remove_if(path, lapsed, read=replaced_while_reading)
    assert json.loads(path.read_text()) == {'live': True, 'renewed': 1}
    assert sorted(os.listdir(tmp_path)) == ['record.json']


def test_remove_if_takes_only_the_listed_version_and_reads_under_the_writers_lock(tmp_path):
    import contextlib
    path = tmp_path / 'clip.mp4'
    path.write_bytes(b'a' * 10)
    listed = file_version(path)
    os.utime(path, ns=(time.time_ns(), time.time_ns() + 10**9))       # used again after it was listed
    assert not remove_if(path, lambda now: now == listed)
    assert path.exists()
    held = []

    @contextlib.contextmanager
    def lock():
        held.append(True)
        yield
    assert remove_if(path, lambda now: now == file_version(path), lock=lock)
    assert held and not path.exists()


def test_sweep_removals_clears_only_old_files_set_aside(tmp_path):
    old, fresh = tmp_path / '.a.json.1.x.removing', tmp_path / '.b.json.1.y.removing'
    old.write_text('{}')
    fresh.write_text('{}')
    sweep_removals(tmp_path, older_than=-1)
    assert not old.exists() and not fresh.exists()
    fresh.write_text('{}')
    sweep_removals(tmp_path)
    assert fresh.exists()
