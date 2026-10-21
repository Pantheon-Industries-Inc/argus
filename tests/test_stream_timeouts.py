"""No client-side timeout ends a reply the provider is still sending (label/harness.py SILENT_CONNECTION_LIMIT_S).

A fake provider on localhost answers the way a long reasoning reply comes back: headers at once, keepalive bytes while
the model reasons, then a long stretch with no bytes at all, then the reply. Every duration is written in real seconds
and multiplied by SCALE, so a 9 minute reply takes about 2 seconds here."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from label import harness
from label.errors import DispatchError

SCALE = 0.004                 # test seconds per real second
KEEPALIVE_EVERY_S = 15        # a keepalive byte this often while the model reasons
REASONING_S = 300             # how long keepalives come before the silent stretch
REPLY = {'id': 'gen-fake', 'model': 'openai/gpt-6.1-sol', 'provider': 'OpenAI',
         'choices': [{'finish_reason': 'stop', 'message': {'content': '{"timeline": []}'}}],
         'usage': {'cost': 0.5, 'prompt_tokens': 10, 'completion_tokens': 5}}


class _Provider:
    """A local chat-completions endpoint whose reply shape each test sets: keepalives for keepalive_s, then
    silence for silent_s (forever when None), then the reply, unless the request is an evidence selection."""

    def __init__(self, silent_s):
        self.silent_s = silent_s
        self.stop = threading.Event()
        provider = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                from tests.test_label import _accounted_evidence_selection
                selection = _accounted_evidence_selection(body['messages'][0]['content'])
                if selection is not None:
                    self.wfile.write(json.dumps(selection).encode())
                    return
                sent = 0.0
                while sent < REASONING_S * SCALE and not provider.stop.is_set():
                    self.wfile.write(b'\n')            # whitespace before a JSON body, as a keepalive
                    self.wfile.flush()
                    provider.stop.wait(KEEPALIVE_EVERY_S * SCALE)
                    sent += KEEPALIVE_EVERY_S * SCALE
                if provider.silent_s is None:
                    provider.stop.wait(60)
                    return
                provider.stop.wait(provider.silent_s * SCALE)
                self.wfile.write(json.dumps(REPLY).encode())

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.url = f'http://127.0.0.1:{self.server.server_address[1]}/api/v1/chat/completions'
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.stop.set()
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def provider(monkeypatch):
    made = []

    def make(silent_s):
        made.append(_Provider(silent_s))
        monkeypatch.setattr(harness, 'OPENROUTER_URL', made[-1].url)
        return made[-1]
    yield make
    for p in made:
        p.close()


@pytest.mark.parametrize('timeout_s, silent_s', [
    (120, 240),     # the evidence selection request (120s per wait before), a 9 minute reply silent for 4 minutes
    (600, 720),     # the final request at the old default 600s, silent for 12 minutes after 5 of keepalives
], ids=['selection-9min-4min-gap', 'final-17min-12min-gap'])
def test_a_live_reply_is_never_cut_off(provider, monkeypatch, timeout_s, silent_s):
    # the bound scales with every other duration; raising=False so the test runs, and fails, on a harness without it
    monkeypatch.setattr(harness, 'SILENT_CONNECTION_LIMIT_S', 6060 * SCALE, raising=False)
    provider(silent_s)
    resp = harness.call_model_once([{'type': 'text', 'text': 'label this'}], 'openai/gpt-6.1-sol', 'high',
                                   'sk-or-fake', 1000, timeout_s * SCALE)
    assert resp['choices'][0]['message']['content'] == '{"timeline": []}'


def test_a_silent_connection_is_closed_as_ours(provider, monkeypatch):
    # shortened so the test is quick: the fake provider stops sending anything after its keepalives
    monkeypatch.setattr(harness, 'SILENT_CONNECTION_LIMIT_S', 300 * SCALE)
    provider(None)
    with pytest.raises(DispatchError) as caught:
        harness.call_model_once([{'type': 'text', 'text': 'label this'}], 'openai/gpt-6.1-sol', 'high',
                                'sk-or-fake', 1000, 1 * SCALE)
    error = caught.value
    assert error.client_timeout and error.stage == 'unknown' and not error.verified_unpaid
    assert 'our client closed the connection' in str(error)


def test_our_timeout_is_recorded_apart_from_a_provider_failure(provider, monkeypatch, resends_used_up,
                                                               inspect_every_field, tmp_path):
    from tests.test_label import _packed_episode
    monkeypatch.setattr(harness, 'SILENT_CONNECTION_LIMIT_S', 300 * SCALE)
    provider(None)
    ep, _ = _packed_episode(tmp_path)
    out = tmp_path / 'out'
    assert harness.run_batch([ep], out, keys=['sk-or-fake'], concurrency=1, force=False, model='openai/gpt-6-astra',
                             reasoning='medium', max_tokens=2000, timeout=1, cell_w=128, max_spend=20) == 1
    record = json.loads((out / f'noreply_{ep.name}.json').read_text())
    assert record['no_reply_kind'] == 'client_timeout' and record['final_dispatch_outcome'] == 'unverified'
    assert 'our client closed the connection' in record['no_reply']
    assert harness.client_timeout(record) and harness.provider_failure(record) is None
    monkeypatch.setattr(harness, 'AUTO_RESENDS', 2)
    assert harness.auto_resend(record).startswith('our client closed a dead connection')
    from board.to_board import label_failed
    assert label_failed(record)['no_reply_kind'] == 'client_timeout'


@pytest.mark.parametrize('no_reply, ours', [
    ('DispatchError: model call failed after 1 HTTP attempts: TimeoutError: The read operation timed out', True),
    ('DispatchError: model call failed after 1 HTTP attempts: IncompleteRead: IncompleteRead(561 bytes read)', False),
    ('DispatchError: model call failed after 6 HTTP attempts: HTTP 504: {"error": "TimeoutError upstream"}', False),
    ('final model call has no verified response', False),
])
def test_records_written_before_the_kind_was_named(no_reply, ours):
    record = {'parse_ok': False, 'final_dispatch_outcome': 'unverified', 'no_reply': no_reply}
    assert bool(harness.client_timeout(record)) is ours
    assert (harness.provider_failure(record) is None) is ours
