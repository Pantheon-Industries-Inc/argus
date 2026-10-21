"""Auxiliary mechanics dispatch reuses the cumulative inspection invoice."""
import copy
import json

import pytest

from label import evidence_access as ea, harness
from label.dictionary_stage import output_spend, output_spend_complete
from label.errors import DispatchError, SpendCap
import complete_reply


REQUEST = {'digest': 'request-one', 'content': [{'type': 'text', 'text': 'SENSOR ANALYSIS FIXTURE'}],
           'content_sha256': 'b00323b4798f60abb091030ff823a83dcb350c28b15deb27a628261a9dec2024',
           'sensor_evidence': {}}


def ledger():
    return {'version': 1, 'digest': 'inspection-one', 'status': 'complete',
            'rounds': [], 'cost_usd': .01, 'limitations': []}


def response(text='{"sensor_findings":[]}', cost=.1, finish='stop'):
    return {'id': 'analysis-response', 'usage': {'cost': cost},
            'choices': [{'finish_reason': finish, 'message': {'content': text}}]}


def execute(request, doc, path, call, **kwargs):
    from label.sensor_analysis_run import execute as run
    return run(request, doc, path, call=call, reserve_cost=lambda parts: .4,
               model=harness.DEFAULT_MODEL, **kwargs)


@pytest.mark.parametrize('text', ['{"sensor_findings":[]}',
                                '```json\n{"sensor_findings":[]}\n```',
                                '```\n{"sensor_findings":[]}\n```'])
def test_received_response_is_durable_before_parser_and_reused_without_payment(tmp_path, text):
    path = tmp_path / 'out' / '.evidence' / 'episode_a.json.inspection.json'
    doc, calls, settlements = ledger(), [], []
    def provider(parts):
        calls.append(parts)
        saved = json.loads(path.read_text())
        assert saved['sensor_analysis']['state'] == 'dispatched'
        assert saved['cost_usd'] == pytest.approx(.41)
        assert output_spend(path.parents[1]) == pytest.approx(.41)
        assert not output_spend_complete(path.parents[1])
        return response(text)
    receipt = execute(REQUEST, doc, path, provider, reserve_dispatch=lambda bound: True,
                      settle_dispatch=lambda held, paid: settlements.append((held, paid)))
    assert receipt['response'] == response(text) and receipt['response_text'] == text
    assert receipt['state'] == 'received'
    assert settlements == [(.4, .1)] and doc['cost_usd'] == pytest.approx(.11)
    assert output_spend(path.parents[1]) == pytest.approx(.11)
    assert output_spend_complete(path.parents[1])
    saved = path.read_bytes()
    assert execute(REQUEST, ledger(), path, provider)['response'] == response(text)
    assert len(calls) == 1 and path.read_bytes() == saved
    out = path.parents[1]
    (out / 'noreply_episode_a.json').write_text(json.dumps({
        'final_dispatch_outcome': 'not dispatched', 'evidence_inspection': doc}))
    assert output_spend(out) == pytest.approx(.11)
    history = harness._billing_fields(out / 'episode_a.json')
    carried = harness.inspection_billed(history, doc['digest'])
    resumed = {**doc, 'previously_billed_usd': carried}
    assert harness.episode_cost({**history, 'evidence_inspection': resumed}) == pytest.approx(.11)


@pytest.mark.parametrize('failure', ['exception', 'process_crash', 'missing_cost'])
def test_unknown_outcome_keeps_full_hold_and_never_resends(tmp_path, failure):
    path, doc, calls = tmp_path / 'cache.json', ledger(), []
    def provider(parts):
        calls.append(parts)
        if failure == 'process_crash':
            raise KeyboardInterrupt()
        if failure == 'exception':
            raise TimeoutError('response lost')
        return {'choices': response()['choices'], 'usage': {}}
    with pytest.raises((KeyboardInterrupt, ValueError)):
        execute(REQUEST, doc, path, provider)
    saved = json.loads(path.read_text())
    assert saved['cost_usd'] == pytest.approx(.41)
    assert saved['cost_is_conservative_estimate'] is True
    assert ea.recover_selection_reservations(saved) is False
    assert saved['cost_usd'] == pytest.approx(.41)
    with pytest.raises(ValueError):
        execute(REQUEST, ledger(), path, provider)
    assert len(calls) == 1


@pytest.mark.parametrize('text,finish', [('not json', 'stop'), ('[]', 'stop'), ('{"sensor_findings":[]}', 'length')])
def test_paid_unusable_response_remains_saved_and_is_not_retried(tmp_path, text, finish):
    doc, path, calls = ledger(), tmp_path / 'cache.json', []
    def provider(parts):
        calls.append(parts)
        return response(text, finish=finish)
    for _ in range(2):
        with pytest.raises(ValueError):
            execute(REQUEST, doc, path, provider)
    saved = json.loads(path.read_text())
    assert saved['sensor_analysis']['response_text'] == text
    assert saved['sensor_analysis']['response'] == response(text, finish=finish)
    assert saved['cost_usd'] == pytest.approx(.11) and len(calls) == 1


def test_cap_refusal_is_unpaid_and_resumes_once_with_same_source(tmp_path):
    doc, path, calls = ledger(), tmp_path / 'cache.json', []
    def provider(parts):
        calls.append(parts)
        return response()
    with pytest.raises(SpendCap):
        execute(REQUEST, doc, path, provider, reserve_dispatch=lambda bound: False)
    assert calls == [] and doc['cost_usd'] == .01
    assert json.loads(path.read_text())['sensor_analysis']['state'] == 'not dispatched'
    execute(REQUEST, doc, path, provider, reserve_dispatch=lambda bound: True)
    assert len(calls) == 1 and doc['cost_usd'] == pytest.approx(.11)
    assert 'pause_reason' not in doc


def test_changed_source_or_settings_cannot_reuse_or_resend_paid_stage(tmp_path):
    doc, path, calls = ledger(), tmp_path / 'cache.json', []
    def provider(parts):
        calls.append(parts)
        return response()
    execute(REQUEST, doc, path, provider)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        execute({**REQUEST, 'digest': 'changed-source-or-settings'}, doc, path, provider)
    assert len(calls) == 1 and path.read_bytes() == before
    def changed():
        raise ValueError('source ownership changed')
    with pytest.raises(ValueError):
        execute(REQUEST, doc, path, provider, verify_inputs=changed)
    assert len(calls) == 1 and path.read_bytes() == before


def test_changed_wire_body_cannot_use_the_original_request_proof(tmp_path):
    request = copy.deepcopy(REQUEST)
    request['content'][0]['text'] = 'A different request under the original proof.'
    calls, doc = [], ledger()
    with pytest.raises(ValueError):
        execute(request, doc, tmp_path / 'cache.json', lambda parts: calls.append(parts))
    assert not calls and doc['cost_usd'] == .01


def test_verified_unpaid_rejection_releases_hold_and_allows_one_resume(tmp_path):
    doc, path, calls, settlements = ledger(), tmp_path / 'cache.json', [], []
    def provider(parts):
        calls.append(parts)
        if len(calls) == 1:
            raise DispatchError('rejected', verified_unpaid=True, http_status=429)
        return response()
    with pytest.raises(DispatchError):
        execute(REQUEST, doc, path, provider, settle_dispatch=lambda held, paid: settlements.append((held, paid)))
    assert doc['cost_usd'] == pytest.approx(.01) and settlements == [(.4, 0.)]
    execute(REQUEST, doc, path, provider)
    assert len(calls) == 2 and doc['cost_usd'] == pytest.approx(.11)
    assert len(doc['sensor_analysis']['rejected_attempts']) == 1


def test_invalid_unpaid_analysis_proof_recovers_hold_once(tmp_path):
    doc, path = ledger(), tmp_path / 'cache.json'
    def rejected(parts):
        raise DispatchError('rejected', verified_unpaid=True, http_status=429)
    with pytest.raises(DispatchError):
        execute(REQUEST, doc, path, rejected)
    doc['sensor_analysis']['verified_unpaid'] = False
    assert ea.recover_selection_reservations(doc)
    assert doc['cost_usd'] == pytest.approx(.41)
    assert doc['cost_is_conservative_estimate'] is True
    assert not ea.recover_selection_reservations(doc)
    assert doc['cost_usd'] == pytest.approx(.41)


def test_incomplete_analysis_claim_never_uses_legacy_one_dollar_fallback(tmp_path):
    doc = ledger()
    doc['sensor_analysis'] = {'state': 'dispatched', 'digest': REQUEST['digest']}
    with pytest.raises(ValueError):
        ea.recover_selection_reservations(doc)
    path = tmp_path / '.evidence' / 'episode_a.json.inspection.json'
    path.parent.mkdir()
    path.write_text(json.dumps(doc))
    assert output_spend(tmp_path) >= 1e100 and not output_spend_complete(tmp_path)


def prepared_mechanics(tmp_path):
    import numpy as np
    from tests.test_camera_timing import recording
    from tests.test_semantic_evidence import descriptor
    ep = recording(tmp_path, np.arange(12) / 30)
    path = ep / 'context.json'
    ctx = json.loads(path.read_text())
    ctx['signals'] = [{'name': 'right_pressure', 'key': 's0', 'shape': [4, 4], 'unit': 'ADC',
        'sensor_semantics': descriptor(regions=[{'name': 'region A', 'indices': list(range(8))},
                                              {'name': 'region B', 'indices': list(range(8, 16))}])}]
    path.write_text(json.dumps(ctx))
    values = np.full((12, 16), 100.)
    values[6:, :8] = 80
    np.savez(ep / 'signals.npz', s0=values)
    return ep


def request_kind(parts):
    texts = [p.get('text', '') for p in parts]
    if any('FIELD INVENTORY' in text for text in texts):
        return 'selection'
    if any(text.startswith('Analyze the supplied regional sensor responses') for text in texts):
        return 'mechanics'
    return 'final'


def test_batch_stage_cap_includes_final_floor_and_resume_does_not_rebill(tmp_path, monkeypatch):
    from tests.test_dispatch_integrity_v5 import options, response as encoded
    ep = prepared_mechanics(tmp_path)
    calls = []
    def bound(parts, max_tokens, **kwargs):
        return {'selection': .03, 'mechanics': .4, 'final': .5}[request_kind(parts)]
    def provider(request, timeout):
        parts = json.loads(request.data)['messages'][0]['content']
        kind = request_kind(parts)
        calls.append(kind)
        answer = {'selection': {'requests': []}, 'mechanics': {'sensor_findings': []},
                  'final': complete_reply.labels('A hand holds the object.')}[kind]
        return encoded(json.dumps(answer), cost={'selection': .01, 'mechanics': .1, 'final': .02}[kind])
    monkeypatch.setattr(harness, 'final_cost_bound', bound)
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    out = tmp_path / 'out'
    args = options(); args['max_spend'] = .8
    assert harness.run_batch([ep], out, **args) == 1
    assert calls == ['selection'], 'mechanics plus the final floor cannot fit under the cap'
    assert output_spend(out) == pytest.approx(.01)
    args['max_spend'] = 1.
    assert harness.run_batch([ep], out, **args) == 0
    assert calls == ['selection', 'mechanics', 'final']
    saved = json.loads((out / (ep.name + '.json')).read_text())
    assert saved['labels'] == complete_reply.labels('A hand holds the object.')
    assert saved['sensor_analysis']['response_text'] == '{"sensor_findings": []}'
    assert output_spend(out) == pytest.approx(.13) and output_spend_complete(out)
    assert harness.run_batch([ep], out, **args) == 0
    assert calls == ['selection', 'mechanics', 'final'] and output_spend(out) == pytest.approx(.13)


def test_batch_unknown_stage_charge_blocks_final_and_all_resume_calls(tmp_path, monkeypatch):
    from tests.test_dispatch_integrity_v5 import options, response as encoded
    ep = prepared_mechanics(tmp_path)
    calls = []
    monkeypatch.setattr(harness, 'final_cost_bound',
                        lambda parts, max_tokens, **kwargs: {'selection': .03, 'mechanics': .4, 'final': .5}[request_kind(parts)])
    def provider(request, timeout):
        kind = request_kind(json.loads(request.data)['messages'][0]['content'])
        calls.append(kind)
        if kind == 'mechanics':
            import io
            return io.BytesIO(json.dumps({'usage': {}, 'choices': response()['choices']}).encode())
        assert kind == 'selection'
        return encoded('{"requests":[]}', cost=.01)
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    out = tmp_path / 'out'
    for _ in range(2):
        assert harness.run_batch([ep], out, **options()) == 1
    assert calls == ['selection', 'mechanics']
    assert output_spend(out) == pytest.approx(.41) and not output_spend_complete(out)


@pytest.mark.parametrize('stage_changed', [False, True])
def test_completed_staged_request_reuses_only_when_both_actual_requests_match(tmp_path, monkeypatch, stage_changed):
    from tests.test_dispatch_integrity_v5 import options, response as encoded
    from tests.test_request_attestation import source_edit
    from label import sensor_analysis as sa
    ep = prepared_mechanics(tmp_path)
    calls = []
    def provider(request, timeout):
        kind = request_kind(json.loads(request.data)['messages'][0]['content'])
        calls.append(kind)
        answer = {'selection': {'requests': []}, 'mechanics': {'sensor_findings': []},
                  'final': complete_reply.labels('A hand holds the object.')}[kind]
        return encoded(json.dumps(answer), cost=.01)
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    out = tmp_path / 'out'
    assert harness.run_batch([ep], out, **options()) == 0
    path = out / (ep.name + '.json')
    saved = json.loads(path.read_text())
    source_edit(monkeypatch, 'label/run.py')
    if stage_changed:
        monkeypatch.setattr(sa, 'HEADER', sa.HEADER + 'A changed analysis requirement.\n')
    assert harness.run_batch([ep], out, **options()) == (1 if stage_changed else 0)
    current = json.loads(path.read_text())
    assert calls == ['selection', 'mechanics', 'final']
    assert current['response_text'] == saved['response_text']
    assert current['sensor_analysis']['response'] == saved['sensor_analysis']['response']
    assert output_spend(out) == pytest.approx(.03)
