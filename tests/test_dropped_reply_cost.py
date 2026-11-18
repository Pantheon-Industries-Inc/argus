"""A reply cut off in transit is still billed: its attempt is counted at what OpenRouter charged for the generation.

On 2026-11-18 a paired Sol 6.1 test sent 48 episodes; 9 final calls failed with IncompleteRead after OpenRouter had
sent the response headers (so the claim held the X-Generation-Id), and Argus sent each one again. OpenRouter's
generation records show all 9 finished with finish_reason stop and billed $1.41 in total, while every such episode's
run.json said cost_usd None (each dropped attempt kept its full hold), so the run totals read $6.46 against $9.34
billed. A fake provider on localhost reproduces the drop: headers with the generation id, then half of a body whose
Content-Length promises more, then a closed connection, which urllib raises as http.client.IncompleteRead."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from label import harness
from label.dictionary_stage import output_spend, output_spend_complete

# what OpenRouter's /api/v1/generation returned on 2026-11-18 for one of the 9 dropped Sol 6.1 calls, trimmed to the
# fields the harness reads
DROPPED = {'total_cost': 0.2171513, 'cancelled': False, 'finish_reason': 'stop', 'tokens_completion': 11464}
REPLY = {'id': 'gen-second', 'model': 'openai/gpt-6.1-sol', 'provider': 'OpenAI',
         'choices': [{'finish_reason': 'stop', 'message': {'content': '{"timeline": []}'}}],
         'usage': {'cost': 0.5, 'prompt_tokens': 10, 'completion_tokens': 5}}


class _DroppingProvider:
    """The first final request gets headers naming gen-dropped and a body cut off halfway; later ones get REPLY."""

    def __init__(self):
        self.finals = []
        provider = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                from tests.test_label import _accounted_evidence_selection
                selection = _accounted_evidence_selection(body['messages'][0]['content'])
                if selection is not None:
                    data = json.dumps(selection).encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                provider.finals.append(body)
                first = len(provider.finals) == 1
                data = json.dumps({**REPLY, 'id': 'gen-dropped'} if first else REPLY).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.send_header('X-Generation-Id', 'gen-dropped' if first else 'gen-second')
                self.end_headers()
                self.wfile.write(data[:len(data) // 2] if first else data)
                self.wfile.flush()
                self.close_connection = True

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.url = f'http://127.0.0.1:{self.server.server_address[1]}/api/v1/chat/completions'
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def provider(monkeypatch):
    p = _DroppingProvider()
    monkeypatch.setattr(harness, 'OPENROUTER_URL', p.url)
    yield p
    p.close()


def _run(tmp_path, monkeypatch, known):
    from tests.test_label import _packed_episode
    asked = []

    def lookup(generation_id, keys):
        asked.append(generation_id)
        return {'id': generation_id, **known[generation_id]} if generation_id in known else None
    monkeypatch.setattr(harness, 'openrouter_generation', lookup)
    monkeypatch.setattr(harness, 'GENERATION_RECORD_WAITS_S', (0.0, 0.0), raising=False)
    ep, _ = _packed_episode(tmp_path)
    out = tmp_path / 'out'
    harness.run_batch([ep], out, keys=['sk-or-fake'], concurrency=1, force=False, model='openai/gpt-6-astra',
                      reasoning='medium', max_tokens=2000, timeout=1, cell_w=128, max_spend=20)
    return out, ep, asked


def test_a_dropped_reply_is_counted_at_what_openrouter_billed(provider, tmp_path, monkeypatch):
    out, ep, asked = _run(tmp_path, monkeypatch, {'gen-dropped': DROPPED})
    assert len(provider.finals) == 2                      # dropped once, then sent again automatically
    _, record = harness.billing_record(out, f'{ep.name}.json')
    dropped = record['final_cost_history'][0]
    assert dropped['final_generation_id'] == 'gen-dropped'
    assert 'IncompleteRead' in dropped['no_reply']
    assert dropped['final_reserved_source'] == 'generation lookup'
    assert dropped['final_reserved_usd'] == pytest.approx(DROPPED['total_cost'])
    assert 'gen-dropped' in asked
    # the run's cost is complete, and holds the dropped attempt's real charge beside the reply's
    assert output_spend_complete(out)
    assert output_spend(out) == pytest.approx(harness.episode_cost(record))
    assert harness.final_history_cost(record) == pytest.approx(DROPPED['total_cost'])
    from label.receipts import billing_summary
    summary = billing_summary(record)
    assert summary['cost_complete'] and summary['reserved_usd'] == 0


def test_a_dropped_reply_openrouter_has_no_record_of_keeps_its_hold(provider, tmp_path, monkeypatch):
    out, ep, asked = _run(tmp_path, monkeypatch, {})
    _, record = harness.billing_record(out, f'{ep.name}.json')
    dropped = record['final_cost_history'][0]
    # a missing record may only be late: the attempt keeps its whole bound, never zero, and the cost stays unknown
    assert dropped['final_dispatch_outcome'] == 'unverified' and 'final_reserved_source' not in dropped
    assert dropped['final_reserved_usd'] > DROPPED['total_cost']
    assert not output_spend_complete(out)
    # looked up before the resend, then at the batch end and once per wait
    assert asked == ['gen-dropped'] * 4
