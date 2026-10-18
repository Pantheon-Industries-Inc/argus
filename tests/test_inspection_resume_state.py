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
                      else {'task_summary': 'The hand reaches.'}, cost=.1)
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    out = tmp_path / 'out'
    args = dict(keys=['sk-or-test'], concurrency=1, force=False, model='custom/model',
                reasoning='medium', max_tokens=100, timeout=1, cell_w=128)
    assert harness.run_batch([ep], out, max_spend=.31, **args) == 0
    cache = out / '.evidence' / (ep.name + '.json.inspection.json')
    paused = json.loads(cache.read_text())
    assert paused['pause_reason'] == 'spend_cap'
    assert len(paused['rounds']) == 1 and paused['cost_usd'] == .1
    assert output_spend(out) == .1 and output_spend_complete(out)
    assert sent == ['selection']
    if legacy_pause:
        paused.pop('pause_reason')
        cache.write_text(json.dumps(paused))
    assert harness.run_batch([ep], out, max_spend=.01, **args) == 0
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
