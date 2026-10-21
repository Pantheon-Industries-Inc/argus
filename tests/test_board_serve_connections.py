"""The board server holds idle and slow connections without refusing other viewers."""
import contextlib
import http.client
import json
import socket
import threading
import time

from board import serve


@contextlib.contextmanager
def board_server(tmp_path, monkeypatch, **bounds):
    monkeypatch.setattr(serve, 'HERE', tmp_path)
    (tmp_path / 'episode_x.json').write_text(json.dumps({'episode_prompt': 'p'}))
    server = serve.BoardHTTPServer(('127.0.0.1', 0), serve.Handler)
    for name, value in bounds.items():
        setattr(server, name, value)
    worker = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .05}, daemon=True)
    worker.start()
    try:
        yield server.server_address[1], server
    finally:
        server.shutdown()
        server.server_close()
        worker.join(3)


def fetch(port, path='/api/episodes', headers=None, timeout=5):
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
    try:
        c.request('GET', path, headers=headers or {})
        r = c.getresponse()
        return r.status, r.read()
    finally:
        c.close()


def test_idle_sockets_over_the_bound_never_refuse_a_new_viewer(tmp_path, monkeypatch):
    with board_server(tmp_path, monkeypatch, max_connections=8) as (port, server):
        idle = [socket.create_connection(('127.0.0.1', port)) for _ in range(30)]
        try:
            for _ in range(5):
                assert fetch(port)[0] == 200
            time.sleep(.2)
            with server._lock:
                assert server._connections() <= 8
        finally:
            for s in idle:
                s.close()


def test_trickled_heads_and_slow_bodies_hold_no_worker_another_client_needs(tmp_path, monkeypatch):
    with board_server(tmp_path, monkeypatch, max_workers=2, client_workers=1) as (port, server):
        slow = []
        for _ in range(5):
            s = socket.create_connection(('127.0.0.1', port))
            s.sendall(b'GET /api/episodes HTTP/1.1\r\nHost: localhost\r\nX-Slow: ')
            slow.append(s)
        for _ in range(3):
            s = socket.create_connection(('127.0.0.1', port))
            s.sendall(b'POST /api/export HTTP/1.1\r\nHost: localhost\r\nX-Forwarded-For: 10.0.0.66\r\n'
                      b'Content-Type: application/json\r\nContent-Length: 1000\r\n\r\n{')
            slow.append(s)
        try:
            time.sleep(.3)
            started = time.monotonic()
            assert fetch(port, headers={'X-Forwarded-For': '10.0.0.7'})[0] == 200
            assert time.monotonic() - started < 1
            with server._lock:
                assert server._busy.get('10.0.0.66', 0) <= 1 and server._workers <= 2
        finally:
            for s in slow:
                s.close()


def test_keep_alive_and_pipelined_requests_share_one_connection(tmp_path, monkeypatch):
    with board_server(tmp_path, monkeypatch) as (port, _):
        c = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
        for _ in range(3):
            c.request('GET', '/api/episode?file=episode_x.json')
            r = c.getresponse()
            assert r.status == 200 and json.loads(r.read())['episode_prompt'] == 'p'
        c.close()
        with socket.create_connection(('127.0.0.1', port), timeout=5) as s:
            s.sendall(b'GET /api/episodes HTTP/1.1\r\nHost: localhost\r\n\r\n' * 2 +
                      b'GET /api/episodes HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n')
            data = b''
            while chunk := s.recv(65536):
                data += chunk
        assert data.count(b'HTTP/1.1 200') == 3


def test_shutdown_leaves_no_connection_or_worker(tmp_path, monkeypatch):
    before = threading.active_count()
    with board_server(tmp_path, monkeypatch) as (port, server):
        held = [socket.create_connection(('127.0.0.1', port)) for _ in range(5)]
        assert fetch(port)[0] == 200
    for s in held:
        s.settimeout(1)
        assert s.recv(1) == b''
        s.close()
    deadline = time.monotonic() + 3
    while threading.active_count() > before and time.monotonic() < deadline:
        time.sleep(.05)
    assert threading.active_count() <= before and not server._by_peer


def test_client_key_groups_ipv6_hosts_and_trusts_forwarding_only_from_loopback():
    assert serve.client_key('2001:db8::1') == serve.client_key('2001:db8::ffff') == '2001:db8::/64'
    assert serve.client_key('127.0.0.1', '1.2.3.4, 10.0.0.5') == '10.0.0.5'
    assert serve.client_key('192.168.1.9', '10.0.0.5') == '192.168.1.9'
    assert serve.client_key('::ffff:10.1.2.3') == '10.1.2.3'


def test_a_port_already_taken_is_reported_as_taken_and_leaves_nothing_open():
    import errno
    import os
    import pytest
    taken = socket.socket()
    taken.bind(('127.0.0.1', 0))
    taken.listen()
    fds = len(os.listdir('/dev/fd')) if os.path.isdir('/dev/fd') else None
    try:
        with pytest.raises(OSError) as raised:
            serve.BoardHTTPServer(taken.getsockname(), serve.Handler)
        assert raised.value.errno == errno.EADDRINUSE
        if fds is not None:
            assert len(os.listdir('/dev/fd')) == fds
    finally:
        taken.close()
