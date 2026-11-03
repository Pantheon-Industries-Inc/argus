"""A final call killed before its reply holds its full bound only until OpenRouter says what it billed for it.

The fixtures in fixtures/go2_killed_calls are three records of a Go2 labelling run on 2026-11-02, trimmed to
their billing fields (the trim keeps each file's output_spend exactly): a plain label, a label whose first attempt was
killed by `modal app stop`, and an episode killed twice. Each killed attempt holds final_cost_bound, about $0.86, the
price of its prompt and its full 64,000-token output limit, while the episode's real final calls cost $0.12 to $0.21.
That run saved no generation ids, so the tests add the ids that the fixed harness now saves at dispatch."""
import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from label import harness
from label import run as lrun
from label.dictionary_stage import output_spend

FIXTURES = Path(__file__).parent / 'fixtures' / 'go2_killed_calls'
# what OpenRouter's /api/v1/generation returned on 2026-11-02 for a gpt-6.1-sol request whose client closed it after
# 8s, trimmed to the fields the harness reads
CANCELLED = {'total_cost': 0.013424, 'cancelled': True, 'finish_reason': None, 'tokens_completion': 1339}
FINISHED = {'total_cost': 0.19, 'cancelled': False, 'finish_reason': 'stop', 'tokens_completion': 9000}


def _go2_out(out: Path) -> Path:
    """The fixtures, with the generation id each killed attempt would now carry."""
    out.mkdir(parents=True)
    ids = {'episode_000033.json': ['gen-33-attempt1'], 'noreply_episode_000034.json': ['gen-34-attempt1']}
    for path in FIXTURES.glob('*.json'):
        record = json.loads(path.read_text())
        for invoice, gid in zip(record.get('final_cost_history') or [], ids.get(path.name, [])):
            invoice['final_generation_id'] = gid
        if record.get('final_dispatch_outcome') == 'claimed':
            record['final_generation_id'] = 'gen-34-attempt3'
        (out / path.name).write_text(json.dumps(record))
    return out


def _openrouter(known):
    asked = []

    def lookup(generation_id, keys):
        asked.append(generation_id)
        return {'id': generation_id, **known[generation_id]} if generation_id in known else None
    return lookup, asked


def test_the_real_records_count_each_killed_attempt_at_its_full_bound(tmp_path):
    out = tmp_path / 'out'
    shutil.copytree(FIXTURES, out)
    # $0.21 + ($0.13 + $0.12 inspection + $0.865 hold) + ($0.861 + $0.861 holds + $0.07 inspection)
    assert output_spend(out) == pytest.approx(3.1239959, abs=1e-6)


def test_a_killed_call_is_counted_at_what_openrouter_billed(tmp_path):
    out = _go2_out(tmp_path / 'out')
    before = output_spend(out)
    lookup, asked = _openrouter({'gen-33-attempt1': FINISHED, 'gen-34-attempt1': CANCELLED})
    settled = harness.resolve_unanswered_charges(out, ['sk-or-a'], lookup=lookup)
    assert settled['resolved'] == 2
    # episode 33's first attempt held $0.865246 and was billed $0.19; episode 34's held $0.860681 and was billed $0.0134
    assert output_spend(out) == pytest.approx(before - (0.865246 - 0.19) - (0.860681 - 0.013424), abs=1e-6)
    # episode 34's current attempt has no record at OpenRouter yet: it keeps its whole hold, never zero
    current = json.loads((out / 'noreply_episode_000034.json').read_text())
    assert current['final_reserved_usd'] == 0.860681 and current['final_dispatch_outcome'] == 'claimed'
    assert current['final_cost_history'][0]['final_reserved_usd'] == pytest.approx(0.013424)
    # a settled hold is never looked up again
    asked.clear()
    harness.resolve_unanswered_charges(out, ['sk-or-a'], lookup=lookup)
    assert asked == ['gen-34-attempt3']


def test_a_generation_still_running_settles_nothing(tmp_path):
    out = _go2_out(tmp_path / 'out')
    before = output_spend(out)
    lookup, _ = _openrouter({'gen-34-attempt1': {'total_cost': 0.0, 'cancelled': False, 'finish_reason': None}})
    assert harness.resolve_unanswered_charges(out, ['sk-or-a'], lookup=lookup)['resolved'] == 0
    assert output_spend(out) == pytest.approx(before)


def test_resume_gets_the_allowance_the_killed_calls_did_not_spend(tmp_path, monkeypatch):
    from tests.test_label import _argv, _fake_harness, _slice
    sl = _slice(tmp_path)
    for name in ('episode_000032', 'episode_000033', 'episode_000034'):
        shutil.copytree(sl / 'episode_a', sl / name)
    run = tmp_path / 'runs' / 'd' / '20261102-0142_full_abc1234'
    _go2_out(run / 'out')
    # a $3 cap for three episodes, as the Go2 run had $20 for its first 49
    (run / 'run.json').write_text(json.dumps({'run_id': run.name, 'kind': 'full', 'dataset': 'd', 'slice': str(sl),
                                              'code': 'abc1234', 'cap_usd': 3.0, 'status': 'running'}))
    (run / 'log.txt').write_text('before the kill\n')
    lookup, _ = _openrouter({'gen-33-attempt1': FINISHED, 'gen-34-attempt1': CANCELLED})
    monkeypatch.setattr(harness, 'openrouter_generation', lookup, raising=False)
    monkeypatch.setattr(lrun, 'commit', lambda: ('abc1234', False))
    monkeypatch.setenv('OPENROUTER_API_KEYS', 'sk-or-a')
    calls = []
    _fake_harness(monkeypatch, calls, cost=0.0)
    _argv(monkeypatch, '--dataset', 'd', '--episodes', sl, '--kind', 'full', '--resume', run, '--why', 'app stopped')
    lrun.main()
    spent = json.loads((run / 'run.json').read_text())['resumes'][0]['spent_before']
    assert spent == pytest.approx(3.1239959 - (0.865246 - 0.19) - (0.860681 - 0.013424), abs=1e-6)
    cmd = calls[0]['cmd']
    assert float(cmd[cmd.index('--max-spend') + 1]) == pytest.approx(3.0 - spent)


class _SlowProvider:
    """Answers the way OpenRouter does: headers with X-Generation-Id at once, then nothing until the model is done."""

    def __init__(self):
        self.stop = threading.Event()
        provider = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('X-Generation-Id', 'gen-final-1')
                self.end_headers()
                self.wfile.flush()
                from tests.test_label import _accounted_evidence_selection
                selection = _accounted_evidence_selection(body['messages'][0]['content'])
                if selection is not None:
                    self.wfile.write(json.dumps(selection).encode())
                    return
                provider.stop.wait(60)          # the final call: the run is killed before the reply

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.url = f'http://127.0.0.1:{self.server.server_address[1]}/api/v1/chat/completions'
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.stop.set()
        self.server.shutdown()
        self.server.server_close()


def test_the_claim_names_its_generation_before_the_reply(tmp_path, monkeypatch, resends_used_up):
    from tests.test_label import _packed_episode
    provider = _SlowProvider()
    try:
        monkeypatch.setattr(harness, 'OPENROUTER_URL', provider.url)
        monkeypatch.setattr(harness, 'SILENT_CONNECTION_LIMIT_S', 1.0)    # stands in for the kill
        ep, _ = _packed_episode(tmp_path)
        out = tmp_path / 'out'
        assert harness.run_batch([ep], out, keys=['sk-or-fake'], concurrency=1, force=False,
                                 model='openai/gpt-6-astra', reasoning='medium', max_tokens=2000, timeout=1,
                                 cell_w=128, max_spend=20) == 1
    finally:
        provider.close()
    record = json.loads((out / f'noreply_{ep.name}.json').read_text())
    assert record['final_dispatch_outcome'] in ('claimed', 'unverified')
    assert record['final_generation_id'] == 'gen-final-1'
    hold, before = record['final_reserved_usd'], output_spend(out)
    lookup, _ = _openrouter({'gen-final-1': CANCELLED})
    assert harness.resolve_unanswered_charges(out, ['sk-or-fake'], lookup=lookup)['resolved'] == 1
    assert output_spend(out) == pytest.approx(before - hold + CANCELLED['total_cost'])
