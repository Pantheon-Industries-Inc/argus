"""Inspection resumability through actual requests and durable billing receipts."""
import io
import json

import numpy as np
import pytest

from label import evidence_access as ea, harness
from label.dictionary_stage import output_spend, output_spend_complete
from test_camera_timing import recording
from test_evidence_access import access
from test_label_review_v2 import answer
import complete_reply


@pytest.mark.parametrize('legacy_pause', [False, True])
def test_partial_inspection_budget_resume_finishes_without_rebilling(tmp_path, monkeypatch, legacy_pause):
    ep = recording(tmp_path, np.arange(12) / 30)
    sent = []
    def provider(request, timeout):
        if request.get_method() == 'GET':
            return io.BytesIO(json.dumps({'data': {'id': 'custom/model', 'endpoints': [{
                'context_length': 50000, 'pricing': {'prompt': '.000004', 'completion': '.00001'}}]}}).encode())
        selection = 'FIELD INVENTORY' in json.dumps(json.loads(request.data)['messages'])
        sent.append('selection' if selection else 'final')
        return answer({'requests': [{'field_id': 'unknown-field', 'mode': 'metadata'}]}
                      if sent == ['selection'] else {'requests': []} if selection
                      else complete_reply.labels('The hand reaches.'), cost=.1)
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    out = tmp_path / 'out'
    args = dict(keys=['sk-or-test'], concurrency=1, force=False, model='custom/model',
                reasoning='medium', max_tokens=100, timeout=1, cell_w=128)
    # a selection reserves about $0.21 and is bought only with the final's $0.20 still covered; $0.44 allows one
    # selection, and after its $0.10 charge the second no longer fits, so the inspection pauses
    assert harness.run_batch([ep], out, max_spend=.44, **args) == 1
    cache = out / '.evidence' / (ep.name + '.json.inspection.json')
    paused = json.loads(cache.read_text())
    assert paused['pause_reason'] == 'spend_cap'
    assert len(paused['rounds']) == 1 and paused['cost_usd'] == .1
    assert output_spend(out) == .1 and output_spend_complete(out)
    assert sent == ['selection']
    if legacy_pause:
        paused.pop('pause_reason')
        cache.write_text(json.dumps(paused))
    assert harness.run_batch([ep], out, max_spend=.01, **args) == 1
    assert sent == ['selection']
    assert json.loads(cache.read_text())['pause_reason'] == 'spend_cap'
    assert harness.run_batch([ep], out, max_spend=1., **args) == 0
    target = out / (ep.name + '.json')
    assert target.exists(), json.loads(cache.read_text())
    result = json.loads(target.read_text())
    inspected = result['evidence_inspection']
    assert inspected['status'] == 'partial' and 'pause_reason' not in inspected
    assert any('unknown field_id' in note for note in inspected['limitations'])
    assert [row['state'] for row in inspected['rounds']] == ['received', 'received']
    assert inspected['cost_usd'] == .2 and harness.episode_cost(result) == pytest.approx(.3)
    assert output_spend(out) == pytest.approx(.3) and output_spend_complete(out)
    assert not (out / ('noreply_' + ep.name + '.json')).exists()
    assert harness.run_batch([ep], out, max_spend=1., **args) == 0
    assert sent == ['selection', 'selection', 'final']


def test_unknown_selection_cost_cannot_be_cleared_by_recoverable_request_error(tmp_path):
    calls, reservations = [], []
    def call(_):
        calls.append(1)
        return {'choices': [{'message': {'content': json.dumps({'requests': [
            {'field_id': 'unknown-field', 'mode': 'metadata'}]})}}]}
    cache = tmp_path / 'inspection.json'
    record, _ = ea.discover(access(tmp_path), [], call, cache, reserve_cost=.2,
        reserve_dispatch=lambda amount: reservations.append(amount) or True,
        settle_dispatch=lambda *args: pytest.fail('unknown charge released'), ledger_managed=True)
    assert len(calls) == 1 and reservations == [.2]
    assert record['status'] == 'incomplete' and record['cost_usd'] == .2
    assert record['cost_is_conservative_estimate'] is True
    resumed, _ = ea.discover(access(tmp_path), [], call, cache, reserve_cost=.2,
        reserve_dispatch=lambda amount: pytest.fail('unknown selection permits new reservation'),
        settle_dispatch=lambda *args: pytest.fail('unknown charge released'), ledger_managed=True)
    assert len(calls) == 1 and resumed['cost_usd'] == .2


def test_proven_unpaid_selection_remains_resumable_across_budget_refusal(tmp_path):
    cache = tmp_path / 'inspection.json'
    def rejected(_):
        raise harness.KeyExhausted('no credits', http_status=402, first_attempt=True)
    with pytest.raises(harness.KeyExhausted):
        ea.discover(access(tmp_path), [], rejected, cache, reserve_cost=.2)
    paused, _ = ea.discover(access(tmp_path), [], lambda _: pytest.fail('cap refusal dispatched'),
        cache, reserve_cost=.2, reserve_dispatch=lambda _: False)
    assert paused['pause_reason'] == 'spend_cap' and paused['cost_usd'] == 0
    settlements = []
    completed, _ = ea.discover(access(tmp_path), [], lambda _: {
        'usage': {'cost': .01}, 'choices': [{'message': {'content': '{"requests": []}'}}]},
        cache, reserve_cost=.2, reserve_dispatch=lambda _: True,
        settle_dispatch=lambda reserve, actual: settlements.append((reserve, actual)))
    assert completed['status'] == 'complete' and 'pause_reason' not in completed
    assert completed['cost_usd'] == .01 and settlements == [(.2, .01)]
    row = completed['rounds'][0]
    assert row['state'] == 'received' and row['rejected_attempts'][0]['verified_unpaid'] is True


def test_inspection_turned_off_sends_only_the_annotation_and_a_touch_sensor_is_still_inspected_by_default(tmp_path, monkeypatch):
    sent = []
    def provider(request, timeout):
        if request.get_method() == 'GET':
            return io.BytesIO(json.dumps({'data': {'id': 'custom/model', 'endpoints': [{
                'context_length': 50000, 'pricing': {'prompt': '.000004', 'completion': '.00001'}}]}}).encode())
        selection = 'FIELD INVENTORY' in json.dumps(json.loads(request.data)['messages'])
        sent.append('selection' if selection else 'final')
        return answer({'requests': []} if selection else complete_reply.labels('The hand reaches.'), cost=.1)
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    args = dict(keys=['sk-or-test'], concurrency=1, force=False, model='custom/model',
                reasoning='medium', max_tokens=100, timeout=1, cell_w=128, max_spend=1.)
    (tmp_path / 'off').mkdir()
    off = recording(tmp_path / 'off', np.arange(12) / 30)
    assert harness.run_batch([off], tmp_path / 'off_out', inspection='off', **args) == 0
    assert sent == ['final']
    result = json.loads((tmp_path / 'off_out' / (off.name + '.json')).read_text())
    assert result['annotation_policy']['inspection'] == 'off'
    assert ea.INSPECTION_OFF in result['evidence_inspection']['limitations']
    assert result['evidence_inspection']['cost_usd'] == 0 and harness.episode_cost(result) == pytest.approx(.1)
    sent.clear()
    (tmp_path / 'on').mkdir()
    on = recording(tmp_path / 'on', np.arange(12) / 30)
    assert harness.run_batch([on], tmp_path / 'on_out', **args) == 0
    assert sent[0] == 'selection' and sent[-1] == 'final'
    default = json.loads((tmp_path / 'on_out' / (on.name + '.json')).read_text())
    assert default['annotation_policy']['inspection'] == 'auto'
    assert ea.INSPECTION_OFF not in default['evidence_inspection']['limitations']


@pytest.mark.parametrize('cost', [0, .02])
def test_a_provider_error_inside_a_200_reply_is_retried_at_once_when_unbilled_and_sent_again_when_billed(tmp_path, monkeypatch,
                                                                                           capsys, cost):
    # OpenAI overloaded behind OpenRouter: HTTP 200, finish_reason error, an error on the choice and a fragment of JSON
    sent = []
    def provider(request, timeout):
        if request.get_method() == 'GET':
            return io.BytesIO(json.dumps({'data': {'id': 'custom/model', 'endpoints': [{
                'context_length': 50000, 'pricing': {'prompt': '.000004', 'completion': '.00001'}}]}}).encode())
        if 'FIELD INVENTORY' in json.dumps(json.loads(request.data)['messages']):
            return answer({'requests': []}, cost=.01)
        sent.append('final')
        if len(sent) == 1:
            return io.BytesIO(json.dumps({'usage': {'cost': cost}, 'choices': [{
                'finish_reason': 'error', 'error': {'code': 502, 'message': 'Our servers are currently overloaded'},
                'message': {'content': '{"timeline": [{"start_s": 0'}}]}).encode())
        return answer(complete_reply.labels('The hand reaches.'), cost=.1)
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    monkeypatch.setattr(harness.time, 'sleep', lambda seconds: None)
    args = dict(keys=['sk-or-test'], concurrency=1, force=False, model='custom/model',
                reasoning='medium', max_tokens=100, timeout=1, cell_w=128, max_spend=1.)
    ep = recording(tmp_path, np.arange(12) / 30)
    out = tmp_path / 'out'
    result = out / (ep.name + '.json')
    if cost == 0:
        assert harness.run_batch([ep], out, **args) == 0                   # retried at once, nothing paid twice
        assert sent == ['final', 'final'] and json.loads(result.read_text())['parse_ok'] is True
        return
    # billed: kept as received and named as the provider's error, then sent again automatically (AUTO_RESENDS), with
    # the failed attempt's charge still counted beside the one that worked
    assert harness.run_batch([ep], out, **args) == 0 and sent == ['final', 'final']
    record = json.loads(result.read_text())
    assert record['parse_ok'] is True and harness.dispatched_attempts(record) == 2
    first = record['final_cost_history'][-1]
    assert 'provider returned an error in the reply (502 Our servers are currently overloaded)' in first['labels']['_parse_error']
    assert harness.episode_cost(record) >= cost + .1 - 1e-9
    assert harness.run_batch([ep], out, **args) == 0 and sent == ['final', 'final']     # done: nothing more is sent


def test_a_run_labelled_before_inspection_could_be_set_is_told_how_to_resume_it_as_paid(tmp_path, monkeypatch, capsys):
    # a staged run whose episode context never saved a policy: its saved request inspected every recorded field (no
    # inspection key), today's default is auto, so the saved output is stale and the message names the flag that
    # resumes it without paying again
    from test_dispatch_integrity_v5 import options
    from test_request_attestation import paid_episode
    ep, out, calls = paid_episode(tmp_path, monkeypatch, inspection='all')
    capsys.readouterr()
    assert harness.run_batch([ep], out, **options()) != 0
    assert '--inspection all to resume it as it was paid for' in capsys.readouterr().err
    assert harness.run_batch([ep], out, **options(inspection='all')) == 0
    assert calls == ['selection', 'selection', 'final', 'cards']                # nothing was paid again
