"""--service-tier flex: the same request at half the price, or a rejection to retry, priced by what can actually be billed."""
import io
import json
import sys
import urllib.error

import numpy as np
import pytest

from label import harness, pricing, tier
import complete_reply
from test_label_review_v2 import OPENAI_PRICING, answer, reject

# Published for openai/gpt-6.1-sol on 2026-11-05: the flex endpoint at half of each standard rate
FLEX_PRICING = {'prompt': '0.000001', 'completion': '0.000005', 'web_search': '0.01', 'input_cache_read': '0.00000005',
                'input_cache_write': '0.00000125', 'discount': 0}
ENDPOINTS = [{'context_length': 1_050_000, 'pricing': OPENAI_PRICING},
             {'context_length': 1_050_000, 'pricing': FLEX_PRICING, 'service_tier': 'flex'}]
TEXT = [{'type': 'text', 'text': 'x'}]
SOL = 'openai/gpt-6.1-sol'


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(pricing, 'search_seen', None)
    monkeypatch.setattr(harness.time, 'sleep', lambda s: None)
    tier.configure(None, False)
    yield
    tier.configure(None, False)


def provider_for(model, sent, rejections=0, status=429, reply=None):
    left = [rejections]
    def provider(request, timeout):
        if request.get_method() == 'GET':
            return io.BytesIO(json.dumps({'data': {'id': model, 'endpoints': ENDPOINTS}}).encode())
        body = json.loads(request.data); sent.append(body)
        if left[0] and body.get('service_tier') == 'flex':
            left[0] -= 1
            reject(request, status, {'error': {'message': 'flex capacity unavailable'}})
        if reply is not None:
            return io.BytesIO(json.dumps(reply).encode())
        selection = 'FIELD INVENTORY' in json.dumps(body['messages'])
        return answer({'requests': []} if selection else complete_reply.labels('The hand reaches.'))
    return provider


def test_unset_sends_what_it_sent_before(monkeypatch):
    """Public behaviour does not change: no service_tier in the body and standard rates in the bound."""
    sent = []
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider_for(SOL, sent))
    harness.call_model(TEXT, SOL, 'high', 'sk-or-test', 10, 5)
    assert 'service_tier' not in sent[0]
    standard = harness.final_cost_bound(TEXT, 1000, model=SOL)
    tier.configure('default', False)
    assert harness.final_cost_bound(TEXT, 1000, model=SOL) == standard
    harness.call_model(TEXT, SOL, 'high', 'sk-or-test', 10, 5)
    assert 'service_tier' not in sent[1]


def test_flex_is_one_field_of_the_body_and_changes_nothing_the_model_is_asked(monkeypatch):
    sent = []
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider_for(SOL, sent))
    harness.call_model(TEXT, SOL, 'high', 'sk-or-test', 10, 5)
    tier.configure('flex', False)
    harness.call_model(TEXT, SOL, 'high', 'sk-or-test', 10, 5)
    plain, flex = sent
    assert flex.pop('service_tier') == 'flex' and flex == plain


def test_the_cap_reserves_flex_rates_and_a_quote_takes_only_flex_endpoints(monkeypatch):
    sent = []
    monkeypatch.setattr(pricing.urllib.request, 'urlopen', provider_for(SOL, sent))
    standard = harness.final_cost_bound(TEXT, 100_000, model=SOL)
    tier.configure('flex', False)
    assert harness.final_cost_bound(TEXT, 100_000, model=SOL) == pytest.approx(standard / 2, rel=.01)
    quote = pricing.quote(SOL, 5, service_tier='flex')
    assert quote['output_usd_per_token'] == pytest.approx(5e-6) and quote['provider_max_price']['completion'] == 5.
    assert pricing.quote(SOL, 5)['output_usd_per_token'] == pytest.approx(1e-5)


def test_a_model_with_no_flex_endpoint_is_refused_rather_than_billed_at_standard(monkeypatch):
    model = 'openai/gpt-6-astra'
    monkeypatch.setattr(pricing.urllib.request, 'urlopen', lambda request, timeout: io.BytesIO(json.dumps(
        {'data': {'id': model, 'endpoints': [ENDPOINTS[0]]}}).encode()))
    with pytest.raises(ValueError, match='flex'):
        pricing.quote(model, 5, service_tier='flex')


def test_with_the_fallback_the_cap_reserves_the_standard_rate_it_may_be_billed(monkeypatch):
    standard = harness.final_cost_bound(TEXT, 100_000, model=SOL)
    tier.configure('flex', True)
    assert harness.final_cost_bound(TEXT, 100_000, model=SOL) == standard


def test_a_capacity_rejection_is_retried_and_logged_and_then_answered_at_flex(monkeypatch):
    sent = []
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider_for(SOL, sent, rejections=2, reply={
        'service_tier': 'flex', 'usage': {'cost': .01}, 'choices': [{'finish_reason': 'stop', 'message': {'content': '{}'}}]}))
    tier.configure('flex', False)
    response = harness.call_model_once(TEXT, SOL, 'high', 'sk-or-test', 10, 5)
    assert response['service_tier'] == 'flex' and [b.get('service_tier') for b in sent] == ['flex'] * 3
    assert tier.counts() == {'requested': 'flex', 'fallback': False, 'rejected': 2, 'fell_back': 0, 'replies': {'flex': 1}}


@pytest.mark.parametrize('status', [429, 503])
def test_without_the_fallback_a_lasting_rejection_stays_unpaid_and_goes_to_standard_never(monkeypatch, status):
    sent = []
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider_for(SOL, sent, rejections=99, status=status))
    tier.configure('flex', False)
    with pytest.raises(harness.DispatchError) as caught:
        harness.call_model_once(TEXT, SOL, 'high', 'sk-or-test', 10, 5)
    assert caught.value.verified_unpaid and all(b.get('service_tier') == 'flex' for b in sent)
    assert tier.counts()['fell_back'] == 0 and tier.counts()['rejected'] == len(sent)


def test_flex_backs_off_for_about_four_minutes_and_a_standard_request_does_not(monkeypatch):
    """Flex capacity comes back in minutes, so a flex rejection waits that long before the unit is left for a resume."""
    slept, sent = [], []
    monkeypatch.setattr(harness.time, 'sleep', slept.append)
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider_for(SOL, sent, rejections=99))
    tier.configure('flex', False)
    with pytest.raises(harness.DispatchError):
        harness.call_model_once(TEXT, SOL, 'high', 'sk-or-test', 10, 5)
    assert len(sent) == harness.FLEX_UNPAID_ATTEMPTS and 200 <= sum(slept) <= 300
    def rejected(request, timeout):
        sent.append(json.loads(request.data))
        reject(request, 429, {'error': {'message': 'rate limited'}})
    sent.clear()
    monkeypatch.setattr(harness.urllib.request, 'urlopen', rejected)
    tier.configure(None, False)
    with pytest.raises(harness.DispatchError):
        harness.call_model_once(TEXT, SOL, 'high', 'sk-or-test', 10, 5)
    assert len(sent) == harness.UNPAID_ATTEMPTS


def test_with_the_fallback_a_lasting_rejection_is_sent_once_at_standard_and_counted(monkeypatch):
    sent = []
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider_for(SOL, sent, rejections=99))
    tier.configure('flex', True)
    harness.call_model_once(TEXT, SOL, 'high', 'sk-or-test', 10, 5)
    assert 'service_tier' not in sent[-1] and all(b['service_tier'] == 'flex' for b in sent[:-1])
    assert tier.counts()['fell_back'] == 1


def test_a_reply_that_may_have_been_billed_is_never_sent_again_at_standard(monkeypatch):
    """A flex reply that timed out may still be billed, so a second request at standard could pay twice."""
    sent = []
    def provider(request, timeout):
        sent.append(json.loads(request.data))
        raise TimeoutError('timed out')
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    tier.configure('flex', True)
    with pytest.raises(harness.DispatchError) as caught:
        harness.call_model_once(TEXT, SOL, 'high', 'sk-or-test', 10, 5)
    assert not caught.value.verified_unpaid and all(b['service_tier'] == 'flex' for b in sent)
    assert tier.counts()['fell_back'] == 0


def test_a_request_straight_to_openai_carries_the_tier_and_costs_what_was_served(monkeypatch):
    usage = {'prompt_tokens': 1000, 'completion_tokens': 1000}
    sent = []
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider_for(SOL, sent, reply={
        'service_tier': 'flex', 'usage': usage, 'choices': [{'finish_reason': 'stop', 'message': {'content': '{}'}}]}))
    tier.configure('flex', False)
    response = harness.call_model(TEXT, SOL, 'high', 'sk-proj-x', 10, 5)
    assert sent[0]['service_tier'] == 'flex'
    assert response['usage']['list_cost'] == pytest.approx(1000 * 1e-6 + 1000 * 5e-6)


def test_a_run_writes_its_tier_in_each_record_and_its_counts_beside_them(tmp_path, monkeypatch):
    from test_camera_timing import recording
    ep = recording(tmp_path, np.arange(12) / 30)
    sent = []
    def provider(request, timeout):
        if request.get_method() == 'GET':
            return io.BytesIO(json.dumps({'data': {'id': SOL, 'endpoints': ENDPOINTS}}).encode())
        body = json.loads(request.data); sent.append(body)
        selection = 'FIELD INVENTORY' in json.dumps(body['messages'])
        content = {'requests': []} if selection else complete_reply.labels('The hand reaches.')
        return io.BytesIO(json.dumps({'service_tier': 'flex', 'usage': {'cost': .01}, 'choices': [{
            'finish_reason': 'stop', 'message': {'content': json.dumps(content)}}]}).encode())
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    out = tmp_path / 'out'
    assert harness.run_batch([ep], out, keys=['sk-or-test'], concurrency=1, force=False, model=SOL, reasoning='high',
                             max_tokens=64000, timeout=1, cell_w=128, max_spend=8., service_tier='flex') == 0
    record = json.loads((out / (ep.name + '.json')).read_text())
    assert record['service_tier'] == 'flex' and record['config']['pricing_quote']['output_usd_per_token'] == pytest.approx(5e-6)
    assert sorted(p.name for p in out.glob('*.json')) == [ep.name + '.json']     # readers take every top level *.json as a label
    counts = json.loads((out / tier.COUNTS).read_text())
    assert counts['requested'] == 'flex' and counts['replies']['flex'] == len(sent) and counts['rejected'] == 0


def test_the_harness_and_review_command_lines_take_both_settings(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(harness, 'run_batch', lambda episodes, out, **kw: seen.update(kw) or 0)
    monkeypatch.setattr(harness, 'discover_episodes', lambda root: [tmp_path])
    monkeypatch.setenv('OPENROUTER_API_KEYS', 'sk-or-test')
    monkeypatch.setattr(sys, 'argv', ['h', '--episodes-root', str(tmp_path), '--out-dir', str(tmp_path / 'o'),
                                      '--service-tier', 'flex', '--flex-fallback', 'standard'])
    assert harness.main() == 0
    assert seen['service_tier'] == 'flex' and seen['flex_fallback'] == 'standard'
    monkeypatch.setattr(sys, 'argv', ['h', '--episodes-root', str(tmp_path), '--out-dir', str(tmp_path / 'o')])
    seen.clear(); harness.main()
    assert seen['service_tier'] is None and seen['flex_fallback'] == 'off'


def run_one(tmp_path, provider, monkeypatch, name='out', **kw):
    from test_camera_timing import recording
    made = tmp_path / 'made'
    ep = next((p for p in made.glob('*') if p.is_dir()), None) if made.exists() else None
    if ep is None:
        made.mkdir()
        ep = recording(made, np.arange(12) / 30)
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    out = tmp_path / name
    return ep, out, harness.run_batch([ep], out, keys=['sk-or-test'], concurrency=1, force=False, model=SOL, reasoning='high',
                                      max_tokens=64000, timeout=1, cell_w=128, max_spend=8., service_tier='flex', **kw)


def test_a_lasting_rejection_costs_nothing_through_the_whole_batch_and_a_resume_sends_it(tmp_path, monkeypatch):
    """Not counted as paid: no reservation stays, nothing is recorded as possibly billed, and a plain resume sends it."""
    sent = []
    ep, out, rc = run_one(tmp_path, provider_for(SOL, sent, rejections=10_000), monkeypatch)
    assert rc != 0 and sent and all(b['service_tier'] == 'flex' for b in sent)
    records = {p.name: json.loads(p.read_text()) for p in out.glob('*.json')}
    assert records and not [n for n, r in records.items() if r.get('final_reserved_usd') or r.get('final_dispatch_outcome') in ('unverified', 'claimed')]
    assert json.loads((out / tier.COUNTS).read_text())['rejected'] == len(sent)
    sent.clear()
    ep, out, rc = run_one(tmp_path, provider_for(SOL, sent), monkeypatch)
    assert rc == 0 and sent and (out / (ep.name + '.json')).exists()
    assert json.loads((out / (ep.name + '.json')).read_text()).get('parse_ok') is True


def test_with_the_fallback_a_run_quotes_standard_rates_so_the_cap_and_max_price_hold_at_standard(tmp_path, monkeypatch):
    """The fallback request costs the standard rate, so a flex quote would undercount its reservation, and its max_price
    would refuse the very request the fallback sends."""
    sent = []
    ep, out, rc = run_one(tmp_path, provider_for(SOL, sent, rejections=10_000), monkeypatch, flex_fallback='standard')
    assert rc == 0
    record = json.loads((out / (ep.name + '.json')).read_text())
    assert record['config']['pricing_quote']['output_usd_per_token'] == pytest.approx(1e-5)
    fallen = [b for b in sent if 'service_tier' not in b]
    assert fallen and all(b['provider']['max_price']['completion'] == 10. for b in fallen)
    assert all(b['provider']['max_price']['completion'] == 10. for b in sent)


@pytest.mark.parametrize('served, cost', [('flex', 1000 * 1e-6 + 1000 * 5e-6), ('default', 1000 * 2e-6 + 1000 * 1e-5), (None, 1000 * 2e-6 + 1000 * 1e-5)])
def test_straight_to_openai_a_reply_is_priced_at_flex_only_if_it_says_flex_served_it(monkeypatch, served, cost):
    reply = {'usage': {'prompt_tokens': 1000, 'completion_tokens': 1000}, 'choices': [{'finish_reason': 'stop', 'message': {'content': '{}'}}],
             **({'service_tier': served} if served else {})}
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider_for(SOL, [], reply=reply))
    tier.configure('flex', False)
    assert harness.call_model(TEXT, SOL, 'high', 'sk-proj-x', 10, 5)['usage']['list_cost'] == pytest.approx(cost)


def test_a_rejection_after_a_timeout_is_not_proved_unpaid_so_it_never_goes_to_standard(monkeypatch):
    """The timed out attempt may be billed even though the last one was refused, so a standard request could pay twice."""
    sent = []
    def provider(request, timeout):
        sent.append(json.loads(request.data))
        if len(sent) == 1:
            raise TimeoutError('timed out')
        raise urllib.error.HTTPError(request.full_url, 429, 'rejected', {}, io.BytesIO(b'{"error": {"message": "no capacity"}}'))
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    tier.configure('flex', True)
    with pytest.raises(harness.DispatchError) as caught:
        harness.call_model(TEXT, SOL, 'high', 'sk-or-test', 10, 5, attempts=2)
    assert not caught.value.verified_unpaid and all(b['service_tier'] == 'flex' for b in sent)
    assert tier.counts()['fell_back'] == 0


def test_a_spent_key_is_not_a_capacity_rejection(monkeypatch):
    sent = []
    def provider(request, timeout):
        sent.append(json.loads(request.data))
        raise urllib.error.HTTPError(request.full_url, 429, 'quota', {}, io.BytesIO(b'{"error": {"code": "insufficient_quota"}}'))
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    tier.configure('flex', True)
    with pytest.raises(harness.KeyExhausted):
        harness.call_model(TEXT, SOL, 'high', 'sk-proj-x', 10, 5)
    assert all(b['service_tier'] == 'flex' for b in sent) and tier.counts()['fell_back'] == 0


def test_unreadable_earlier_counts_do_not_stop_a_run_that_has_finished(tmp_path, capsys):
    (tmp_path / tier.COUNTS).parent.mkdir()
    (tmp_path / tier.COUNTS).write_text('{cut off')
    tier.configure('flex', False)
    tier.write(tmp_path)
    assert 'could not record' in capsys.readouterr().out


# What OpenRouter sent on 2026-11-05 for a Sol flex request at capacity: status 200, no choices, no usage. It was held as a possibly billed call, so
# a run that spent $0.77 read $4.58, which a stop line per footage hour would read as a cost rise.
FLEX_BUSY = {'message': 'Flex processing is temporarily unavailable. Please try again later or use standard processing.', 'code': 502,
             'metadata': {'error_type': 'provider_unavailable'}}


def busy_then(sent, busy, reply=None):
    left = [busy]
    def provider(request, timeout):
        body = json.loads(request.data); sent.append(body)
        if left[0] and body.get('service_tier') == 'flex':
            left[0] -= 1
            return io.BytesIO(json.dumps(FLEX_BUSY).encode())
        return io.BytesIO(json.dumps(reply or {'service_tier': body.get('service_tier', 'default'), 'usage': {'cost': .01}, 'choices': [{'finish_reason': 'stop', 'message': {'content': '{}'}}]}).encode())
    return provider


def test_a_capacity_rejection_inside_a_200_is_retried_as_unpaid(monkeypatch):
    sent = []
    monkeypatch.setattr(harness.urllib.request, 'urlopen', busy_then(sent, 2))
    tier.configure('flex', False)
    assert harness.call_model_once(TEXT, SOL, 'high', 'sk-or-test', 10, 5)['service_tier'] == 'flex'
    assert len(sent) == 3 and tier.counts()['rejected'] == 2


def test_a_lasting_one_ends_unpaid_so_no_reservation_is_held(monkeypatch):
    sent = []
    monkeypatch.setattr(harness.urllib.request, 'urlopen', busy_then(sent, 99))
    tier.configure('flex', False)
    with pytest.raises(harness.DispatchError) as caught:
        harness.call_model_once(TEXT, SOL, 'high', 'sk-or-test', 10, 5)
    assert caught.value.verified_unpaid and caught.value.http_status == 502 and caught.value.stage == 'rejected'


def test_the_fallback_takes_it_too(monkeypatch):
    sent = []
    monkeypatch.setattr(harness.urllib.request, 'urlopen', busy_then(sent, 99))
    tier.configure('flex', True)
    harness.call_model_once(TEXT, SOL, 'high', 'sk-or-test', 10, 5)
    assert 'service_tier' not in sent[-1] and tier.counts()['fell_back'] == 1


@pytest.mark.parametrize('service_tier, body', [('flex', {'message': 'something else', 'code': 500}), (None, FLEX_BUSY)])
def test_any_other_empty_200_stays_a_possibly_billed_call(monkeypatch, service_tier, body):
    monkeypatch.setattr(harness.urllib.request, 'urlopen', lambda request, timeout: io.BytesIO(json.dumps(body).encode()))
    tier.configure(service_tier, False)
    with pytest.raises(harness.DispatchError) as caught:
        harness.call_model_once(TEXT, SOL, 'high', 'sk-or-test', 10, 5)
    assert not caught.value.verified_unpaid
