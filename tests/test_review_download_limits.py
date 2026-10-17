import contextlib
import http.server
import threading
import time

import pytest

from review import __main__ as review


@contextlib.contextmanager
def download_server(*, first_delay=0, byte_delay=0, count=100):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Length', str(count))
            self.end_headers()
            time.sleep(first_delay)
            try:
                for _ in range(count):
                    self.wfile.write(b'x')
                    self.wfile.flush()
                    time.sleep(byte_delay)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/recording.bin'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def test_download_trickling_body_still_obeys_total_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(review, 'DOWNLOAD_DEADLINE', .15)
    monkeypatch.setattr(review, 'DOWNLOAD_TIMEOUT', .08)
    with download_server(byte_delay=.01) as url:
        started = time.monotonic()
        with pytest.raises(ValueError, match='time limit'):
            review.fetch(url, tmp_path)
        elapsed = time.monotonic() - started
    assert elapsed < .6
    assert not list((tmp_path / 'upload').rglob('.download-*'))
    assert not list((tmp_path / 'upload').rglob('recording.bin'))


def test_download_stalled_body_has_read_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(review, 'DOWNLOAD_DEADLINE', 1)
    monkeypatch.setattr(review, 'DOWNLOAD_TIMEOUT', .05)
    with download_server(first_delay=.3, count=1) as url:
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            review.fetch(url, tmp_path)
        elapsed = time.monotonic() - started
    assert elapsed < .2
    assert not list((tmp_path / 'upload').rglob('.download-*'))
    assert not list((tmp_path / 'upload').rglob('recording.bin'))


def test_download_completed_body_keeps_every_chunk(tmp_path):
    with download_server(count=100, byte_delay=.0001) as url:
        dest = review.fetch(url, tmp_path)
    assert dest.read_bytes() == b'x' * 100
    assert not list((tmp_path / 'upload').rglob('.download-*'))


def test_download_duration_override_allows_a_slow_healthy_transfer(tmp_path, monkeypatch):
    monkeypatch.setattr(review, 'DOWNLOAD_DEADLINE', .05)
    monkeypatch.setenv('REVIEW_DOWNLOAD_DEADLINE', '.5')
    with download_server(first_delay=.12, count=2) as url:
        dest = review.fetch(url, tmp_path)
    assert dest.read_bytes() == b'xx'
    assert not list((tmp_path / 'upload').rglob('.download-*'))
