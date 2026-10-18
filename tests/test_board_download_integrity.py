"""HTTP framing and cache transactions at the public download boundary."""
import concurrent.futures
import contextlib
import json
import socket
import socketserver
import threading
import urllib.error
import urllib.request

import pytest

from board import serve, static
from test_prepared_footage import media, serving, get


def test_get_with_body_closes_connection_without_parsing_body_as_request(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, 'HERE', tmp_path)
    with serving() as base:
        host, port = base.split('/')[2].split(':')
        with socket.create_connection((host, int(port)), timeout=3) as connection:
            body = b'GET /api/nope HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n'
            connection.sendall(b'GET /api/nope HTTP/1.1\r\nHost: localhost\r\nContent-Length: ' + str(len(body)).encode() + b'\r\n\r\n' + body)
            response = b''
            while chunk := connection.recv(65536):
                response += chunk
    assert response.count(b'HTTP/1.1 ') == 1
    assert b'Connection: close' in response


@pytest.mark.parametrize('wire_request', [
    b'POST /unknown HTTP/1.1\r\nContent-Length: 99999999\r\n',
    b'POST /api/export HTTP/1.1\r\nTransfer-Encoding: chunked\r\n',
    b'POST /api/export HTTP/1.1\r\nContent-Length: ' + b'9' * 5000 + b'\r\n',
    b'POST /api/dictionary HTTP/1.1\r\nContent-Length: 99999999\r\n',
    b'POST /api/export HTTP/1.1\r\nContent-Length: 2\r\nContent-Length: 3\r\n',
])
def test_rejected_body_framing_closes_connection(wire_request, tmp_path, monkeypatch):
    monkeypatch.setattr(serve, 'HERE', tmp_path)
    with serving() as base:
        host, port = base.split('/')[2].split(':')
        with socket.create_connection((host, int(port)), timeout=3) as connection:
            connection.sendall(wire_request + b'Host: localhost\r\n\r\n')
            response = connection.recv(65536)
            assert b'Connection: close' in response
            while connection.recv(65536):
                pass


def test_export_deduplicates_requested_episode_and_spools_bounded_response(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, 'HERE', tmp_path)
    (tmp_path / 'episode_x.json').write_text(json.dumps({'episode_prompt': 'hello', '_meta': {'episode_id': 'episode_x'}}))
    with serving() as base:
        data = json.dumps({'files': ['episode_x.json'] * 10000}).encode()
        request = urllib.request.Request(base + 'api/export', data=data, headers={'Content-Type': 'application/json'})
        output = get(request)
        assert len(output.splitlines()) == 1
        monkeypatch.setattr(serve, '_EXPORT_MAX_BYTES', 8)
        with pytest.raises(urllib.error.HTTPError) as error:
            get(request)
        assert error.value.code == 413


def test_ready_receipts_do_not_age_out_before_media_and_zero_count_keeps_ticket(tmp_path, monkeypatch):
    media(tmp_path, monkeypatch)
    monkeypatch.setattr(serve, '_FOOTAGE_DISK_FILES', 3)
    first = serve.footage('episode_x', 0, .1)
    token = serve.prepare_footage_ticket(first)
    version = serve._footage_version(first[0])
    serve.footage('episode_x', .05, .2)
    serve.footage('episode_x', .1, .3)
    assert serve.footage('episode_x', 0, .1)[0] == first[0]
    assert serve._footage_version(first[0]) == version
    assert serve.ticket_footage(token) == first
    monkeypatch.setattr(serve, '_FOOTAGE_DISK_FILES', 0)
    token = serve.prepare_footage_ticket(first)
    assert serve.ticket_footage(token) == first


def test_parallel_ticket_range_downloads_do_not_consume_probe_admission(tmp_path, monkeypatch):
    media(tmp_path, monkeypatch)
    monkeypatch.setattr(static, '_durable_metadata', lambda *args: False)
    with serving() as base:
        prepared = json.loads(get(base + 'api/footage?id=episode_x&prepare=1'))
        assert serve._FOOTAGE_PROBE_SEM.acquire(blocking=False)
        try:
            def download(_):
                request = urllib.request.Request(base + 'api/footage?ticket=' + prepared['ticket'], headers={'Range': 'bytes=0-31'})
                return get(request)
            with concurrent.futures.ThreadPoolExecutor(3) as pool:
                assert list(pool.map(download, range(3))) == [download(0)] * 3
        finally:
            serve._FOOTAGE_PROBE_SEM.release()


def test_repeat_untrusted_prepare_hashes_without_copy_and_large_source_can_make_short_span(tmp_path, monkeypatch):
    source = media(tmp_path, monkeypatch)
    monkeypatch.setattr(serve, '_FOOTAGE_DISK_BYTES', 3000)
    assert source.stat().st_size > 3000
    copies = []
    original = serve._copy_source_snapshot
    monkeypatch.setattr(serve, '_copy_source_snapshot', lambda *args: (copies.append(args), original(*args))[1])
    first = serve.footage('episode_x', 0, .1)
    assert first and first[0].stat().st_size < 3000
    assert serve.footage('episode_x', 0, .1) == first
    assert copies == []


def test_prepared_span_bypasses_another_untrusted_encode_and_sources_are_descriptor_bound(tmp_path, monkeypatch):
    import os
    source = media(tmp_path, monkeypatch)
    (source.parent / 'episode_y.mp4').write_bytes(source.read_bytes())
    initial = serve.footage('episode_y', 0, .1)
    entered, release = threading.Event(), threading.Event()
    original = serve._run_footage
    def blocked(cmd):
        # Replacement after identities and probes must not switch the open inputs.
        replacement = source.with_suffix('.new')
        replacement.write_bytes((tmp_path / 'blue.mp4').read_bytes().ljust(4096, b'\0'))
        os.replace(replacement, source)
        entered.set()
        assert release.wait(5)
        return original(cmd)
    monkeypatch.setattr(serve, '_run_footage', blocked)
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        pending = pool.submit(serve.footage, 'episode_x', .1, .3)
        try:
            assert entered.wait(3)
            # A ready span for unchanged bytes still works while an encoder is busy.
            assert serve.footage('episode_y', 0, .1) == initial
        finally:
            release.set()
        result = pending.result()
    import av
    with av.open(str(result[0])) as encoded:
        pixel = next(encoded.decode(video=0)).to_ndarray(format='rgb24')[0, 0]
    assert int(pixel[0]) > int(pixel[2]) + 100


def _server_process(connection, encoded=None, release=None):
    original = serve._run_footage
    if encoded is not None:
        def run(cmd):
            result = original(cmd)
            encoded.set()
            assert release.wait(10)
            return result
        serve._run_footage = run
    with socketserver.ThreadingTCPServer(('127.0.0.1', 0), serve.Handler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        connection.send(server.server_address[1])
        connection.recv()
        server.shutdown()
        worker.join(3)


@pytest.mark.skipif(__import__('os').name != 'posix', reason='fork and inherited source descriptors')
@pytest.mark.parametrize('durable', [True, False])
def test_two_http_processes_adopt_first_published_output_without_breaking_ticket(tmp_path, monkeypatch, durable):
    import multiprocessing
    media(tmp_path, monkeypatch)
    monkeypatch.setattr(static, '_durable_metadata', lambda *args: durable)
    context = multiprocessing.get_context('fork')
    encoded, release = context.Event(), context.Event()
    processes, pipes, bases = [], [], []
    try:
        for paused in (False, True):
            parent, child = context.Pipe()
            process = context.Process(target=_server_process, args=(child, encoded if paused else None, release))
            process.start()
            processes.append(process)
            pipes.append(parent)
            assert parent.poll(5)
            bases.append(f'http://127.0.0.1:{parent.recv()}/')
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            pending = pool.submit(get, bases[1] + 'api/footage?id=episode_x&prepare=1')
            try:
                assert encoded.wait(5)
                first = json.loads(get(bases[0] + 'api/footage?id=episode_x&prepare=1'))
                url = bases[0] + 'api/footage?ticket=' + first['ticket']
                content = get(url)
            finally:
                release.set()
            second = json.loads(pending.result())
            assert get(url) == content
            assert get(bases[1] + 'api/footage?ticket=' + second['ticket']) == content
        assert len(list(serve.FOOTAGE_DIR.glob('*.mp4'))) == 1
    finally:
        release.set()
        for pipe, process in zip(pipes, processes):
            if process.is_alive():
                pipe.send('stop')
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join()
            assert process.exitcode == 0


def test_output_receipt_transaction_keeps_same_span_requests_and_tickets_valid(tmp_path, monkeypatch):
    media(tmp_path, monkeypatch)
    encoded, release = threading.Event(), threading.Event()
    original = serve._remember_ready_footage
    def remember(key, *args):
        if len(key) == 20:
            encoded.set()
            assert release.wait(5)
        return original(key, *args)
    monkeypatch.setattr(serve, '_remember_ready_footage', remember)
    with serving() as base, concurrent.futures.ThreadPoolExecutor(2) as pool:
        request = base + 'api/footage?id=episode_x&prepare=1'
        first = pool.submit(get, request)
        try:
            assert encoded.wait(3)
            version = serve._footage_version(next(serve.FOOTAGE_DIR.glob('*.mp4')))
            second = pool.submit(get, request)
        finally:
            release.set()
        tickets = [json.loads(future.result())['ticket'] for future in (first, second)]
        assert get(base + 'api/footage?ticket=' + tickets[0]) == get(base + 'api/footage?ticket=' + tickets[1])
        assert serve._footage_version(next(serve.FOOTAGE_DIR.glob('*.mp4'))) == version


def test_source_reads_are_bounded_to_two_on_miss_and_one_on_ready_hit(tmp_path, monkeypatch):
    source = media(tmp_path, monkeypatch)
    monkeypatch.setattr(serve, '_CONTENT_PROBES', serve.OrderedDict())
    reads, probes = [], []
    digest, probe = serve._stream_digest, serve._probe_uncached
    def count(reader, size, *args):
        if str(reader.name) == str(source):
            reads.append(size)
        return digest(reader, size, *args)
    def inspect(path):
        probes.append(path)
        return probe(path)
    monkeypatch.setattr(serve, '_stream_digest', count)
    monkeypatch.setattr(serve, '_probe_uncached', inspect)
    first = serve.footage('episode_x', 0, .1)
    assert reads == [source.stat().st_size] * 2 and len(probes) == 1
    assert serve.footage('episode_x', 0, .1) == first
    assert reads == [source.stat().st_size] * 3 and len(probes) == 1
    assert not (serve.FOOTAGE_DIR / '.sources').exists()


def test_ticket_range_sends_validated_descriptor_after_path_is_replaced(tmp_path, monkeypatch):
    import os
    media(tmp_path, monkeypatch)
    with serving() as base:
        prepared = json.loads(get(base + 'api/footage?id=episode_x&prepare=1'))
        url = base + 'api/footage?ticket=' + prepared['ticket']
        original_bytes = get(url)
        send = serve.Handler._send_file
        def replace(self, path, *args, **kwargs):
            replacement = path.with_suffix('.changed')
            replacement.write_bytes(b'different bytes')
            os.replace(replacement, path)
            return send(self, path, *args, **kwargs)
        monkeypatch.setattr(serve.Handler, '_send_file', replace)
        request = urllib.request.Request(url, headers={'Range': 'bytes=0-31'})
        assert get(request) == original_bytes[:32]


@pytest.mark.parametrize('canonical', [None, '{', '[]', '{}', '{"format":"invalid"}',
    'wrong-version', 'wrong-hash'])
def test_live_ticket_recovers_an_unusable_canonical_receipt_without_replacing_movie(tmp_path, monkeypatch, canonical):
    media(tmp_path, monkeypatch)
    result = serve.footage('episode_x', 0, .1)
    token = serve.prepare_footage_ticket(result)
    version = serve._footage_version(result[0])
    for receipt in (serve.FOOTAGE_DIR / '.ready').glob('*.json'):
        if receipt.stem == result[0].stem and canonical is not None:
            if canonical in ('wrong-version', 'wrong-hash'):
                record = json.loads(receipt.read_text())
                record['version' if canonical == 'wrong-version' else 'sha256'] = []
                receipt.write_text(json.dumps(record))
            else:
                receipt.write_text(canonical)
        else:
            receipt.unlink()
    assert serve.footage('episode_x', 0, .1) == result
    assert serve._footage_version(result[0]) == version
    assert serve.ticket_footage(token) == result
    assert serve.ticket_footage(serve.prepare_footage_ticket(result)) == result
