"""A refusal OpenRouter sends inside a 200 reply costs nothing, as the same HTTP status does: it holds no charge and is retried."""
import io
import json

import numpy as np
import pytest

from label import harness, pricing, tier
import complete_reply
from test_label_review_v2 import answer
from test_service_tier import ENDPOINTS, SOL, TEXT

# What OpenRouter sent k10-0315 (2026-11-10) for 16 sends of episode_1, status 200: Argus held each at $3.105 as a possibly
# billed call ($49.68 in all), while the run's label.log said total_cost=$0.00 and $1.01 was billed for the whole run.
RATE_LIMITED = {'error': {'message': 'openai/gpt-6.1-sol is temporarily rate-limited upstream. Please retry shortly, or add '
                                     'your own provider key to use your own limits: https://openrouter.ai/workspaces/default/byok',
                          'code': 429, 'metadata': {'error_type': 'rate_limit_exceeded'}}}


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(pricing, 'search_seen', None)
    monkeypatch.setattr(harness.time, 'sleep', lambda s: None)
    tier.configure(None, False)
    yield
    tier.configure(None, False)


def provider(sent, refusals, body=RATE_LIMITED):
    left = [refusals]
    def reply(request, timeout):
        if request.get_method() == 'GET':
            return io.BytesIO(json.dumps({'data': {'id': SOL, 'endpoints': ENDPOINTS}}).encode())
        sent.append(json.loads(request.data))
        if left[0]:
            left[0] -= 1
            return io.BytesIO(json.dumps(body).encode())
        selection = 'FIELD INVENTORY' in json.dumps(sent[-1]['messages'])
        return answer({'requests': []} if selection else complete_reply.labels('The hand reaches.'))
    return reply


def test_a_rate_limited_send_is_retried_with_backoff_and_then_answered(monkeypatch):
    sent = []
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider(sent, 2))
    assert harness.call_model_once(TEXT, SOL, 'high', 'sk-or-test', 10, 5)['choices']
    assert len(sent) == 3


@pytest.mark.parametrize('code', [429, 503, 400, 403])
def test_a_lasting_refusal_is_proved_unpaid_so_its_hold_is_released(monkeypatch, code):
    body = {'error': {**RATE_LIMITED['error'], 'code': code}}
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider([], 99, body))
    with pytest.raises(harness.DispatchError) as caught:
        harness.call_model_once(TEXT, SOL, 'high', 'sk-or-test', 10, 5)
    assert caught.value.verified_unpaid and caught.value.http_status == code and caught.value.stage == 'rejected'
    assert 'HTTP 200 carrying error' in str(caught.value) and 'rate-limited upstream' in str(caught.value)


@pytest.mark.parametrize('body', [
    {'error': {'message': 'Insufficient credits', 'code': 402}},               # no proof the key is spent
    {'error': {'message': 'User not found.', 'code': 401}},
    {**RATE_LIMITED, 'usage': {'cost': .5}},                                   # it says it billed: kept as possibly billed
    {**RATE_LIMITED, 'usage': {}},                                             # usage with no cost: unpriced, kept
    {'error': {**RATE_LIMITED['error'], 'code': 502}},                         # a generation that failed after it started
    {'error': {**RATE_LIMITED['error'], 'code': 408}},
    {'error': {**RATE_LIMITED['error'], 'code': '429'}},                       # not OpenRouter's numeric code
    {'error': {'message': 'no code'}}])
def test_any_other_empty_200_stays_a_possibly_billed_call(monkeypatch, body):
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider([], 99, body))
    with pytest.raises(harness.DispatchError) as caught:
        harness.call_model_once(TEXT, SOL, 'high', 'sk-or-test', 10, 5)
    assert not caught.value.verified_unpaid and caught.value.stage == 'unknown'


def test_a_lost_connection_after_the_send_stays_a_possibly_billed_call(monkeypatch):
    def lost(request, timeout):
        raise ConnectionResetError('reset by peer')
    monkeypatch.setattr(harness.urllib.request, 'urlopen', lost)
    with pytest.raises(harness.DispatchError) as caught:
        harness.call_model_once(TEXT, SOL, 'high', 'sk-or-test', 10, 5)
    assert not caught.value.verified_unpaid and caught.value.stage == 'unknown'


def run(tmp_path, reply, monkeypatch, name):
    from test_camera_timing import recording
    made = tmp_path / 'made'
    if not made.exists():
        made.mkdir()
        recording(made, np.arange(12) / 30)
    ep = next(p for p in made.glob('*') if p.is_dir())
    monkeypatch.setattr(harness.urllib.request, 'urlopen', reply)
    out = tmp_path / name
    return ep, out, harness.run_batch([ep], out, keys=['sk-or-test'], concurrency=1, force=False, model=SOL, reasoning='high',
                                      max_tokens=64000, timeout=1, cell_w=128, max_spend=8.)


def test_a_run_whose_sends_are_all_rate_limited_holds_nothing_and_a_resume_labels_it(tmp_path, monkeypatch, capsys):
    """k10-0315's take held $12.42 for four refused sends of one unit; refused sends now hold $0, and a resume sends them."""
    sent = []
    ep, out, rc = run(tmp_path, provider(sent, 10_000), monkeypatch, 'out')
    assert rc != 0 and sent
    records = {p.name: json.loads(p.read_text()) for p in out.glob('*.json')}
    held = [n for n, r in records.items() if r.get('final_reserved_usd') or r.get('final_dispatch_outcome') in ('unverified', 'claimed')]
    assert records and not held
    said = capsys.readouterr()
    assert 'total_cost=$0.00' in said.out and 'not billed, a resume retries it' in said.err and 'carrying error 429' in said.err
    sent.clear()
    ep, out, rc = run(tmp_path, provider(sent, 0), monkeypatch, 'out')
    assert rc == 0 and json.loads((out / (ep.name + '.json')).read_text()).get('parse_ok') is True
