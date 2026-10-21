"""Exercise review reports through real parsing, prepared requests, and HTTP dispatch."""
import io
import json
import urllib.error

import numpy as np
import pytest

from label import dictionary as dd, harness
import complete_reply


def answer(content, *, cost=.01):
    return io.BytesIO(json.dumps({'usage': {'cost': cost}, 'choices': [{
        'finish_reason': 'stop', 'message': {'content': json.dumps(content)}}]}).encode())


def reject(request, status, body):
    raise urllib.error.HTTPError(request.full_url, status, 'rejected', {},
                                 io.BytesIO(json.dumps(body).encode()))


def test_dictionary_quota_failover_uses_transport_unpaid_proof(tmp_path, monkeypatch):
    from test_dictionary import episode
    ep = episode(tmp_path, 'one', [('pad', np.ones((2, 1)), {})])
    sent = []
    def provider(request, timeout):
        sent.append(request.get_header('Authorization'))
        if len(sent) == 1:
            reject(request, 429, {'error': {'code': 'insufficient_quota'}})
        return answer({'entries': []})
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    result = dd.prepare_upload(tmp_path / 'job', [ep], None, harness.call_model_once,
                               api_keys=['sk-proj-empty', 'sk-proj-good'])
    assert result['status'] == 'success'
    assert [r['state'] for r in result['dispatch_attempts']] == ['rejected unpaid', 'received']
    unpaid = result['dispatch_attempts'][0]
    assert unpaid['cost_usd'] == 0 and unpaid['rejection_code'] == 'insufficient_quota'
    assert result['cost_usd'] == .01
    resumed = dd.prepare_upload(tmp_path / 'job', [ep], None, harness.call_model_once,
                                api_keys=['sk-proj-empty', 'sk-proj-good'])
    assert resumed == result and len(sent) == 2


@pytest.mark.parametrize('status,body', [(429, {}), (503, {}),
    (429, {'error': {'code': 'insufficient_quota'}})])
def test_dictionary_openrouter_error_status_is_unpaid(tmp_path, monkeypatch, status, body):
    # An error status is the provider declining to produce a completion: nothing is billed, so the request is
    # retried, then sent with the next key, and nothing stays reserved.
    from test_dictionary import episode
    ep = episode(tmp_path, 'one', [('pad', np.ones((2, 1)), {})])
    sent = []
    def provider(request, timeout):
        sent.append(request.get_header('Authorization'))
        reject(request, status, body)
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    monkeypatch.setattr(harness.time, 'sleep', lambda seconds: None)
    job = tmp_path / 'job'
    result = dd.prepare_upload(job, [ep], None, harness.call_model_once,
                               api_keys=['sk-or-first', 'sk-or-second'])
    assert result['status'] == 'failed' and result['cost_usd'] == 0
    assert [a['state'] for a in result['dispatch_attempts']] == ['rejected unpaid', 'rejected unpaid']
    # a spent quota is not retried on the same key; a busy provider is
    tries = 1 if body else harness.UNPAID_ATTEMPTS
    assert sent == ['Bearer sk-or-first'] * tries + ['Bearer sk-or-second'] * tries


def test_only_missing_trailing_notes_are_recovered_without_column_shift():
    good = [0, 1, 'left', 'reach', 'cup', None, None, 'advancing', .2]
    shifted = [0, 1, 'left', 'reach', 'cup', None, 'advancing', .2, None]
    labels, ok = harness.parse_response(json.dumps({'timeline_columns': harness.TIMELINE_COLUMNS,
                                                   'timeline': [good, shifted]}))
    assert ok and labels['timeline'] == [dict(zip(harness.TIMELINE_COLUMNS, good))]
    assert labels['_dropped'][0]['row'] == 1
    assert harness.parse_response(json.dumps({'timeline': [shifted]}))[1] is False


def test_ego_trailing_notes_recovery_checks_hand_columns():
    columns = harness.TIMELINE_COLUMNS[:-1] + ['hands_visible', 'hands_wearing', 'notes']
    good = [0, 1, 'left', 'reach', 'cup', None, None, 'advancing', .2, True, None]
    shifted = [0, 1, 'left', 'reach', 'cup', None, None, 'advancing', .2, 'bare', None]
    labels, ok = harness.parse_response(json.dumps({'timeline_columns': columns, 'timeline': [good, shifted]}))
    assert ok and len(labels['timeline']) == 1 and labels['timeline'][0]['hands_visible'] is True
    assert labels['_dropped'][0]['row'] == 1


def test_completion_case_variation_preserves_the_result_without_a_false_violation():
    labels, ok = harness.parse_response('{"completion":{"task_completed":"Success"}}')
    assert ok and labels['completion']['task_completed'] == 'success' and not labels.get('_schema_violations')


@pytest.mark.parametrize('profile,bad_field,bad_value', [
    ('teleop_arms', 'arm', True), ('teleop_arms', 'progress', 'late'),
    ('handheld_gripper', 'object', ['cup']), ('ego_head', 'hands_visible', 'true'),
    ('ego_head', 'hands_wearing', False), ('ego_head', 'progress', 'late'),
])
def test_full_width_partial_rows_keep_paid_content_and_resume(tmp_path, monkeypatch, profile, bad_field, bad_value):
    from test_camera_timing import recording
    columns = (harness.TIMELINE_COLUMNS[:-1] + ['hands_visible', 'hands_wearing', 'notes']
               if profile == 'ego_head' else harness.TIMELINE_COLUMNS)
    values = dict(start_s=0, end_s=.3, arm='left', action='reach', object='cup', destination=None,
                  spatial_relation=None, contribution='advancing', progress=.2, notes=None,
                  hands_visible=True, hands_wearing='bare')
    values[bad_field] = bad_value
    row = [values[column] for column in columns]
    reply = {'timeline_columns': columns, 'timeline': [row], 'completion': {'task_completed': 'success'},
             'tasks': [{'task': 'reach', 'outcome': 'success'}]}
    labels, accepted = harness.parse_response(json.dumps(reply))
    assert accepted and labels['timeline'][0][bad_field] == bad_value
    assert labels['timeline'][0]['action'] == 'reach' and labels['_schema_violations']
    short, accepted = harness.parse_response(json.dumps({'timeline_columns': columns, 'timeline': [row[:-1]]}))
    if accepted:
        assert short['_dropped'] and all('progress' not in step for step in short['timeline'])

    ep = recording(tmp_path, np.arange(12) / 30)
    context_path = ep / 'context.json'
    context = json.loads(context_path.read_text())
    context['profile'] = profile
    context_path.write_text(json.dumps(context))
    sent = []
    def provider(request, timeout):
        selection = 'FIELD INVENTORY' in json.dumps(json.loads(request.data)['messages'])
        sent.append('selection' if selection else 'final')
        return answer({'requests': []} if selection else reply)
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    out = tmp_path / 'out'
    args = dict(keys=['sk-or-test'], concurrency=1, force=False, model=harness.DEFAULT_MODEL,
                reasoning='medium', max_tokens=100, timeout=1, cell_w=128, max_spend=1.)
    assert harness.run_batch([ep], out, **args) == 0
    target = out / (ep.name + '.json')
    result = json.loads(target.read_text())
    assert result['parse_ok'] and result['labels']['timeline'][0][bad_field] == bad_value
    assert result['labels']['_schema_violations'] and harness.episode_cost(result) == pytest.approx(.02)
    original = target.read_bytes()
    assert harness.run_batch([ep], out, **args) == 0
    assert target.read_bytes() == original and sent == ['selection', 'final']


@pytest.mark.parametrize('row,completion', [
    ({'start_s': 0, 'end_s': 1, 'action': 'reach', 'progress': 50}, {'task_completed': 'success'}),
    ({'start_s': 0, 'end_s': 1, 'action': 'reach', 'progress': '0.5'}, {'task_completed': 'success'}),
    ({'start_s': 0, 'end_s': 1, 'action': 'reach', 'progress': .5}, {'task_completed': 'Success'}),
])
def test_partial_paid_annotation_is_preserved_and_resumed(tmp_path, monkeypatch, row, completion):
    from test_camera_timing import recording
    ep = recording(tmp_path, np.arange(12) / 30)
    sent = []
    reply = {'timeline': [row], 'completion': completion}
    def provider(request, timeout):
        payload = json.loads(request.data)
        selection = 'FIELD INVENTORY' in json.dumps(payload['messages'])
        sent.append('selection' if selection else 'final')
        return answer({'requests': []} if selection else reply)
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    out = tmp_path / 'out'
    args = dict(keys=['sk-or-test'], concurrency=1, force=False, model=harness.DEFAULT_MODEL,
                reasoning='medium', max_tokens=100, timeout=1, cell_w=128, max_spend=1.)
    assert harness.run_batch([ep], out, **args) == 0
    target = out / (ep.name + '.json')
    result = json.loads(target.read_text())
    assert result['parse_ok'] and result['labels']['timeline'][0]['action'] == 'reach'
    if row['progress'] == 50:
        assert result['labels']['_schema_violations']
    assert harness.episode_cost(result) == pytest.approx(.02)
    original = target.read_bytes()
    assert harness.run_batch([ep], out, **args) == 0
    assert target.read_bytes() == original and sent == ['selection', 'final']
    legacy = json.loads(target.read_text())
    legacy.pop('input_identity')
    target.write_text(json.dumps(legacy))
    assert harness.run_batch([ep], out, **args) == 0
    assert sent == ['selection', 'final']


@pytest.mark.parametrize('rejected_stage', ['selection', 'final'])
@pytest.mark.parametrize('error_length', [20, 1000])
def test_force_refresh_is_consumed_once_when_key_is_rejected(tmp_path, monkeypatch, rejected_stage, error_length):
    from test_camera_timing import recording
    ep = recording(tmp_path, np.arange(12) / 30)
    sent = []
    fail_final = [False]
    def provider(request, timeout):
        selection = 'FIELD INVENTORY' in json.dumps(json.loads(request.data)['messages'])
        key = request.get_header('Authorization')
        sent.append((key, 'selection' if selection else 'final'))
        if fail_final[0] and selection == (rejected_stage == 'selection') and key.endswith('first'):
            reject(request, 429, {'error': {'code': 'insufficient_quota', 'message': 'x' * error_length}})
        return answer({'requests': []} if selection else complete_reply.labels('The hand reaches for the cup.'))
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    out = tmp_path / 'out'
    args = dict(concurrency=1, model=harness.DEFAULT_MODEL, reasoning='medium', max_tokens=100,
                timeout=1, cell_w=128, max_spend=1.)
    assert harness.run_batch([ep], out, keys=['sk-proj-first'], force=False, **args) == 0
    fail_final[0] = True
    assert harness.run_batch([ep], out, keys=['sk-proj-first', 'sk-proj-second'], force=True, **args) == 0
    expected = [('Bearer sk-proj-first', 'selection'), ('Bearer sk-proj-first', 'final'),
                ('Bearer sk-proj-first', 'selection')]
    expected += ([('Bearer sk-proj-first', 'final')] if rejected_stage == 'final' else
                 [('Bearer sk-proj-second', 'selection')])
    assert sent == expected + [('Bearer sk-proj-second', 'final')]
    saved = json.loads((out / (ep.name + '.json')).read_text())
    assert harness.episode_cost(saved) == pytest.approx(.04)
    assert len(list((out / '.evidence').glob('*.previous.*'))) == 2


@pytest.mark.parametrize('has_identity', [False, True])
@pytest.mark.parametrize('labels', [
    {'timeline': [{'start_s': 0, 'end_s': 1, 'action': 'reach', 'progress': 50}]},
    {'timeline': [{'start_s': 0, 'end_s': 1, 'action': 'reach', 'progress': '0.5'}]},
    {'completion': {'task_completed': 'Success'}},
])
def test_legacy_partial_paid_output_is_not_sent_again(tmp_path, monkeypatch, has_identity, labels):
    """A saved paid reply that lacks a timeline or an outcome is not a finished annotation (label/completeness.py),
    and it is never sent again on its own: the run fails and names how to resend it."""
    from test_camera_timing import recording
    ep = recording(tmp_path, np.arange(12) / 30)
    out = tmp_path / 'out'
    out.mkdir()
    args = dict(model=harness.DEFAULT_MODEL, reasoning='medium', max_tokens=100, timeout=1,
                cell_w=128, example_dir=None)
    result = {'parse_ok': True, 'labels': labels, 'usage': {'est_cost_usd': .01}}
    if has_identity:
        result['input_identity'] = harness.input_identity(ep, **{k: v for k, v in args.items() if k != 'timeout'})
    target = out / (ep.name + '.json')
    target.write_text(json.dumps(result))
    original = target.read_bytes()
    monkeypatch.setattr(harness.urllib.request, 'urlopen', lambda *a, **kw: pytest.fail('paid output replayed'))
    assert harness.run_batch([ep], out, keys=['sk-or-test'], concurrency=1, force=False, max_spend=1., **args) == 1
    assert target.read_bytes() == original
    assert json.loads((out / f'stale_{ep.name}.json').read_text())['status'] in ('stale', 'unverified')


@pytest.mark.parametrize('status', [429, 503])
def test_final_http_error_status_is_unpaid_and_resumes(tmp_path, monkeypatch, status):
    # The provider answered with an error instead of a completion, so nothing was billed: the reservation is released
    # and a plain resume sends the final request again, with no --force.
    from test_camera_timing import recording
    ep = recording(tmp_path, np.arange(12) / 30)
    sent, fail = [], [True]
    def provider(request, timeout):
        selection = 'FIELD INVENTORY' in json.dumps(json.loads(request.data)['messages'])
        sent.append('selection' if selection else 'final')
        if not selection and fail[0]:
            reject(request, status, {'error': {'message': 'busy'}})
        return answer({'requests': []} if selection else complete_reply.labels('The hand reaches.'))
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    monkeypatch.setattr(harness.time, 'sleep', lambda seconds: None)
    out = tmp_path / 'out'
    args = dict(keys=['sk-or-first', 'sk-or-second'], concurrency=1, force=False, max_spend=1.,
                model=harness.DEFAULT_MODEL, reasoning='medium', max_tokens=100, timeout=1, cell_w=128)
    assert harness.run_batch([ep], out, **args) == 1
    receipt = json.loads((out / ('noreply_' + ep.name + '.json')).read_text())
    assert receipt['final_dispatch_outcome'] == 'not dispatched' and receipt['final_reserved_usd'] == 0
    assert receipt['final_rejection']['http_status'] == status
    assert sent.count('final') == harness.UNPAID_ATTEMPTS
    fail[0] = False
    assert harness.run_batch([ep], out, **args) == 0
    assert sent.count('final') == harness.UNPAID_ATTEMPTS + 1


@pytest.mark.parametrize('outcome', ['timeout', 504])
def test_unknown_final_outcome_is_sent_again_twice_and_keeps_every_reserve(tmp_path, monkeypatch, outcome):
    # Sent, then no answer (a read timeout, a gateway timeout): the request may have been billed, so its reservation
    # stays. Argus sends it again automatically, at most twice (AUTO_RESENDS), each attempt keeping its reservation,
    # and a plain resume past that bound never sends it again.
    from test_camera_timing import recording
    ep = recording(tmp_path, np.arange(12) / 30)
    sent = []
    def provider(request, timeout):
        selection = 'FIELD INVENTORY' in json.dumps(json.loads(request.data)['messages'])
        sent.append('selection' if selection else 'final')
        if not selection:
            if outcome == 'timeout':
                raise TimeoutError('The read operation timed out')
            reject(request, outcome, {'error': {'message': 'request outcome unavailable'}})
        return answer({'requests': []})
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    out = tmp_path / 'out'
    args = dict(keys=['sk-or-first', 'sk-or-second'], concurrency=1, force=False, max_spend=1.,
                model=harness.DEFAULT_MODEL, reasoning='medium', max_tokens=100, timeout=1, cell_w=128)
    # a cap that covers one reservation but not a second stops the resend: the first possible charge still counts
    assert harness.run_batch([ep], tmp_path / 'capped', **args) == 1 and sent.count('final') == 1
    sent.clear()
    args['max_spend'] = 5.
    assert harness.run_batch([ep], out, **args) == 1
    receipt = out / ('noreply_' + ep.name + '.json')
    result = json.loads(receipt.read_text())
    assert result['final_dispatch_outcome'] == 'unverified' and result['final_reserved_usd'] > 0
    assert sent.count('final') == 1 + harness.AUTO_RESENDS == harness.dispatched_attempts(result)
    # every attempt that may have been billed is still counted, not only the last
    assert harness.episode_cost(result) >= (1 + harness.AUTO_RESENDS) * result['final_reserved_usd'] * 0.999
    original, before = receipt.read_bytes(), list(sent)
    # Resume reports the unresolved charge while retaining the receipt and refusing a fourth send.
    assert harness.run_batch([ep], out, **args) == 1
    assert receipt.read_bytes() == original and sent == before


def endpoint_metadata(model):
    # Rates are fixture values, not published prices or production defaults.
    return {'data': {'id': model, 'endpoints': [
        {'context_length': 32768, 'pricing': {'prompt': '0.000002', 'completion': '0.000008',
                                             'image': '0.001', 'request': '0.002'}},
        {'context_length': 65536, 'pricing': {'prompt': '0.000004', 'completion': '0.000010',
                                             'image': '0.002', 'request': '0.003',
                                             'input_cache_write': '0.000005'}}]}}


def test_provider_quote_covers_time_and_long_context_price_overrides(tmp_path, monkeypatch):
    from test_camera_timing import recording
    ep = recording(tmp_path, np.arange(12) / 30)
    model = 'deepseek/deepseek-v4.1-flash'
    sent = []
    def provider(request, timeout):
        sent.append(request.get_method())
        if request.get_method() == 'GET':
            return io.BytesIO(json.dumps({'data': {'id': model, 'endpoints': [{
                'context_length': 1048576, 'pricing': {'prompt': '0.00000015', 'completion': '0.0000006',
                'input_cache_read': '0.000000003', 'discount': 0, 'overrides': [
                    {'utc_days': ['monday'], 'utc_start': 100, 'utc_end': 400,
                     'prompt': '0.0000003', 'completion': '0.0000012'},
                    {'min_prompt_tokens': 128000, 'prompt': '0.0000006', 'completion': '0.0000024'}]}}]}}).encode())
        body = json.loads(request.data)
        assert body['provider']['max_price'] == {'prompt': .6, 'completion': 2.4, 'image': 0., 'request': 0.}
        selection = 'FIELD INVENTORY' in json.dumps(body['messages'])
        return answer({'requests': []} if selection else complete_reply.labels('The hand reaches.'))
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    out = tmp_path / 'out'
    assert harness.run_batch([ep], out, keys=['sk-or-test'], concurrency=1, force=False,
        model=model, reasoning='medium', max_tokens=64000, timeout=1, cell_w=128, max_spend=2.) == 0
    assert sent == ['GET', 'POST', 'POST']
    result = json.loads((out / (ep.name + '.json')).read_text())
    assert result['config']['pricing_quote']['input_usd_per_token'] == .0000006


def test_premium_service_tier_does_not_inflate_standard_comparison_reserve(monkeypatch):
    from label import pricing
    model = 'anthropic/claude-opus-5.5'
    payload = {'data': {'id': model, 'endpoints': [
        {'context_length': 1_000_000, 'service_tier': None, 'pricing': {
            'prompt': '0.000004', 'completion': '0.00002', 'input_cache_write_1h': '0.000008'}},
        {'context_length': 1_000_000, 'service_tier': 'priority', 'pricing': {
            'prompt': '0.000008', 'completion': '0.00004', 'input_cache_write_1h': '0.000016'}}]}}
    monkeypatch.setattr(pricing.urllib.request, 'urlopen', lambda *a, **kw: io.BytesIO(json.dumps(payload).encode()))
    quote = pricing.quote(model, 1)
    assert quote['provider_max_price']['prompt'] == 4.
    # standard-tier rates only; an empty request is bounded at 1,024 prompt tokens, never the whole context window
    assert pricing.cost_bound([], 64000, quote) == pytest.approx(1024 * .000004 + 64000 * .00002, abs=2e-6)
    assert pricing.cost_bound([{'type': 'image_url'}] * 1000, 64000, quote) == pytest.approx(
        1_000_000 * .000004 + 64000 * .00002, abs=2e-6)    # never more than the published window


def test_provider_unit_conversion_does_not_lower_the_published_ceiling(monkeypatch):
    from label import pricing
    payload = endpoint_metadata('custom/model')
    payload['data']['endpoints'] = [{'context_length': 1024,
        'pricing': {'prompt': '0.00000045', 'completion': '0.0000018'}}]
    monkeypatch.setattr(pricing.urllib.request, 'urlopen', lambda *a, **kw: io.BytesIO(json.dumps(payload).encode()))
    quote = pricing.quote('custom/model', 1)
    assert quote['provider_max_price']['prompt'] == .45
    assert quote['provider_max_price']['completion'] == 1.8


def test_public_metadata_lookup_does_not_bind_run_to_the_first_key(tmp_path, monkeypatch):
    from test_camera_timing import recording
    ep = recording(tmp_path, np.arange(12) / 30)
    model = 'deepseek/deepseek-v4.1-flash'
    calls = []
    def provider(request, timeout):
        key = request.get_header('Authorization')
        calls.append((request.get_method(), key))
        if request.get_method() == 'GET':
            if key is not None:
                reject(request, 401, {'error': {'message': 'first key revoked'}})
            return io.BytesIO(json.dumps(endpoint_metadata(model)).encode())
        if key.endswith('first'):
            reject(request, 402, {'error': {'message': 'no credits'}})
        selection = 'FIELD INVENTORY' in json.dumps(json.loads(request.data)['messages'])
        return answer({'requests': []} if selection else complete_reply.labels('The hand reaches.'))
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    out = tmp_path / 'out'
    assert harness.run_batch([ep], out, keys=['sk-or-first', 'sk-or-second'], concurrency=1, force=False,
        model=model, reasoning='medium', max_tokens=100, timeout=1, cell_w=128, max_spend=2.) == 0
    assert calls == [('GET', None), ('POST', 'Bearer sk-or-first'), ('GET', None),
                     ('POST', 'Bearer sk-or-second'), ('POST', 'Bearer sk-or-second')]


@pytest.mark.parametrize('model', ['anthropic/claude-opus-5.5', 'deepseek/deepseek-v4.1-flash'])
def test_capped_non_openai_run_uses_provider_prices_and_enforces_them(tmp_path, monkeypatch, model):
    from test_camera_timing import recording
    ep = recording(tmp_path, np.arange(12) / 30)
    sent = []
    def provider(request, timeout):
        sent.append(request)
        if request.get_method() == 'GET':
            assert request.full_url == 'https://openrouter.ai/api/v1/models/' + model + '/endpoints'
            return io.BytesIO(json.dumps(endpoint_metadata(model)).encode())
        body = json.loads(request.data)
        assert body['model'] == model
        assert body['provider']['max_price'] == {'prompt': 4., 'completion': 10., 'image': .002, 'request': .003}
        selection = 'FIELD INVENTORY' in json.dumps(body['messages'])
        return answer({'requests': []} if selection else complete_reply.labels('The hand reaches for the cup.'))
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    out = tmp_path / 'out'
    args = dict(keys=['sk-or-test'], concurrency=1, model=model, reasoning='medium', max_tokens=100,
                timeout=1, cell_w=128, max_spend=2., force=False)
    assert harness.run_batch([ep], tmp_path / 'dry', dry_run=True, **args) == 0
    assert sent == []
    assert harness.run_batch([ep], out, **args) == 0
    saved = json.loads((out / (ep.name + '.json')).read_text())
    quote = saved['config']['pricing_quote']
    assert quote['model'] == model and quote['source'].endswith('/endpoints')
    assert quote['prompt_token_bound'] == 65536 and quote['input_usd_per_token'] == .000005
    assert harness.episode_cost(saved) == pytest.approx(.02)
    assert [r.get_method() for r in sent] == ['GET', 'POST', 'POST']
    assert harness.run_batch([ep], out, **args) == 0
    assert len(sent) == 3


@pytest.mark.parametrize('cap', [1., 8.])
def test_quote_handles_published_opus_fee_shape_without_inventing_surcharges(tmp_path, monkeypatch, cap):
    from test_camera_timing import recording
    ep = recording(tmp_path, np.arange(12) / 30)
    model = 'anthropic/claude-opus-5.5'
    sent = []
    def provider(request, timeout):
        sent.append(request.get_method())
        if request.get_method() == 'GET':
            # Shape and rates published by the endpoint URL on 2026-10-17.
            return io.BytesIO(json.dumps({'data': {'id': model, 'endpoints': [{
                'context_length': 1_000_000, 'pricing': {'prompt': '0.000004', 'completion': '0.00002',
                'web_search': '0.01', 'input_cache_read': '0.0000002', 'input_cache_write': '0.000005',
                'input_cache_write_1h': '0.000008', 'discount': 0}}]}}).encode())
        body = json.loads(request.data)
        assert body['provider']['max_price'] == {'prompt': 4., 'completion': 20., 'image': 0., 'request': 0.}
        selection = 'FIELD INVENTORY' in json.dumps(body['messages'])
        return answer({'requests': []} if selection else complete_reply.labels('The hand reaches.'))
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    out = tmp_path / 'out'
    assert harness.run_batch([ep], out, keys=['sk-or-test'], concurrency=1, force=False,
        model=model, reasoning='medium', max_tokens=64000, timeout=1, cell_w=128, max_spend=cap) == (1 if cap < 1.28 else 0)
    # 64,000 output tokens alone reserve $1.28; the prompt is bounded by the request, not the 1M token window
    if cap == 1.:
        assert sent == ['GET']
        result = json.loads((out / ('noreply_' + ep.name + '.json')).read_text())
        assert harness.episode_cost(result) == 0
        assert result['evidence_inspection']['status'] == 'incomplete'
        assert 'remaining batch spend cap' in result['evidence_inspection']['limitations'][0]
        return
    assert sent == ['GET', 'POST', 'POST']
    result = json.loads((out / (ep.name + '.json')).read_text())
    assert result['config']['pricing_quote']['input_usd_per_token'] == .000005
    assert result['evidence_inspection']['status'] == 'complete'
    assert result['evidence_inspection']['cost_usd'] == .01


@pytest.mark.parametrize('metadata', [None, {'data': {'id': 'wrong', 'endpoints': []}},
    {'data': {'id': 'custom/model', 'endpoints': [{'context_length': 32000, 'pricing': {'prompt': 'NaN'}}]}}])
def test_unverified_non_openai_rates_never_authorize_paid_work(tmp_path, monkeypatch, metadata):
    from test_camera_timing import recording
    ep = recording(tmp_path, np.arange(12) / 30)
    sent = []
    def provider(request, timeout):
        sent.append(request.get_method())
        if request.get_method() != 'GET':
            pytest.fail('paid request dispatched with unverified metadata')
        if metadata is None:
            raise urllib.error.URLError('metadata unavailable')
        return io.BytesIO(json.dumps(metadata).encode())
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    out = tmp_path / 'out'
    assert harness.run_batch([ep], out, keys=['sk-or-test'], concurrency=1, model='custom/model',
        reasoning='medium', max_tokens=100, timeout=1, cell_w=128, max_spend=2., force=False) == 1
    assert sent == ['GET']


def test_non_openai_provider_quote_still_obeys_dispatch_cap(tmp_path, monkeypatch):
    from test_camera_timing import recording
    ep = recording(tmp_path, np.arange(12) / 30)
    sent = []
    def provider(request, timeout):
        sent.append(request.get_method())
        assert request.get_method() == 'GET'
        return io.BytesIO(json.dumps(endpoint_metadata('custom/model')).encode())
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    out = tmp_path / 'out'
    harness.run_batch([ep], out, keys=['sk-or-test'], concurrency=1, model='custom/model',
        reasoning='medium', max_tokens=100, timeout=1, cell_w=128, max_spend=.01, force=False)
    assert sent == ['GET']
    receipt = json.loads((out / ('noreply_' + ep.name + '.json')).read_text())
    assert harness.episode_cost(receipt) == 0


@pytest.mark.parametrize('contribution', [None, 'Advancing'])
def test_short_rows_with_identifiable_progress_keep_content_and_resume(tmp_path, monkeypatch, contribution):
    from test_camera_timing import recording
    ep = recording(tmp_path, np.arange(12) / 30)
    row = [0, .3, 'left', 'reach', 'cup', None, None, contribution, .2]
    sent = []
    def provider(request, timeout):
        selection = 'FIELD INVENTORY' in json.dumps(json.loads(request.data)['messages'])
        sent.append('selection' if selection else 'final')
        return answer({'requests': []} if selection else {**complete_reply.labels(), 'timeline': [row]})
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    out = tmp_path / 'out'
    args = dict(keys=['sk-or-test'], concurrency=1, force=False, max_spend=1.,
                model=harness.DEFAULT_MODEL, reasoning='medium', max_tokens=100, timeout=1, cell_w=128)
    assert harness.run_batch([ep], out, **args) == 0
    result = json.loads((out / (ep.name + '.json')).read_text())
    assert result['parse_ok'] and result['labels']['timeline'][0]['action'] == 'reach'
    assert result['labels']['timeline'][0]['contribution'] == contribution
    assert bool(result['labels'].get('_schema_violations')) is (contribution is None)
    assert harness.run_batch([ep], out, **args) == 0
    assert sent == ['selection', 'final']


@pytest.mark.parametrize('initial_cap', [.01, .015])
def test_budget_increase_resumes_undispatched_selection(tmp_path, monkeypatch, initial_cap):
    from test_camera_timing import recording
    ep = recording(tmp_path, np.arange(12) / 30)
    sent = []
    def provider(request, timeout):
        if request.get_method() == 'GET':
            return io.BytesIO(json.dumps(endpoint_metadata('custom/model')).encode())
        selection = 'FIELD INVENTORY' in json.dumps(json.loads(request.data)['messages'])
        sent.append('selection' if selection else 'final')
        return answer({'requests': []} if selection else complete_reply.labels('The hand reaches.'))
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    out = tmp_path / 'out'
    args = dict(keys=['sk-or-test'], concurrency=1, force=False, model='custom/model',
                reasoning='medium', max_tokens=100, timeout=1, cell_w=128)
    assert harness.run_batch([ep], out, max_spend=initial_cap, **args) == 1
    cache = out / '.evidence' / (ep.name + '.json.inspection.json')
    paused = json.loads(cache.read_text())
    assert paused['rounds'] == [] and paused['cost_usd'] == 0 and sent == []
    assert harness.run_batch([ep], out, max_spend=1., **args) == 0
    result = json.loads((out / (ep.name + '.json')).read_text())
    assert sent == ['selection', 'final']
    assert result['evidence_inspection']['status'] == 'complete'
    assert result['evidence_inspection']['cost_usd'] == .01
    assert harness.run_batch([ep], out, max_spend=1., **args) == 0
    assert sent == ['selection', 'final']


@pytest.mark.parametrize('cap', [0., .1, 20.])
def test_known_rate_inspection_uses_the_funded_batch_cap(tmp_path, monkeypatch, cap):
    from test_camera_timing import recording
    from label import evidence_access as ea
    ep = recording(tmp_path, np.arange(12) / 30)
    context_path = ep / 'context.json'
    context = json.loads(context_path.read_text())
    context['instruction'] = 'Source task context. ' * 12000
    context['new_fact'] = {'value': 7}
    context_path.write_text(json.dumps(context))
    request = harness.me.build_request(ep, model=harness.DEFAULT_MODEL, reasoning='medium', cell_w=128,
                                       inspect_evidence=True)
    access = request['evidence_access']
    selection = access.selection_content(request['content']) + [{'type': 'text', 'text': ea.SELECTION +
                 '\nFIELD INVENTORY\n' + ea.packed(access.page())}]
    assert harness.final_cost_bound(selection, ea.selection_tokens('medium')) > ea.MAX_SELECTION_COST
    sent = []
    def provider(request, timeout):
        assert request.get_method() != 'GET', 'Known model rates require no external quote'
        selecting = 'FIELD INVENTORY' in json.dumps(json.loads(request.data)['messages'])
        sent.append('selection' if selecting else 'final')
        return answer({'requests': []} if selecting else complete_reply.labels('The hand reaches.'))
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    out = tmp_path / 'funded'
    rc = harness.run_batch([ep], out, keys=['sk-or-test'], concurrency=1, force=False,
        model=harness.DEFAULT_MODEL, reasoning='medium', max_tokens=100, timeout=1, cell_w=128, max_spend=cap)
    if 0 < cap < 1:
        assert rc == 1 and sent == []
    else:
        result = json.loads((out / (ep.name + '.json')).read_text())
        assert rc == 0 and sent == ['selection', 'final']
        assert result['evidence_inspection']['status'] == 'complete'
        assert harness.episode_cost(result) == pytest.approx(.02)


def test_uncapped_batch_completes_second_inspection_round_and_reuses_paid_output(tmp_path, monkeypatch):
    from test_camera_timing import recording
    from label import evidence_access as ea
    ep = recording(tmp_path, np.arange(12) / 30)
    context_path = ep / 'context.json'
    context = json.loads(context_path.read_text())
    context.update(instruction='Source task context. ' * 2000, new_fact={'value': 7})
    context_path.write_text(json.dumps(context))
    request = harness.me.build_request(ep, model=harness.DEFAULT_MODEL, reasoning='medium', cell_w=128,
                                       inspect_evidence=True)
    field = next(row for row in request['evidence_access'].inventory() if row['name'] == 'new_fact')
    sent, reservations = [], []
    def provider(request, timeout):
        assert request.get_method() != 'GET'
        content = json.loads(request.data)['messages'][0]['content']
        selecting = 'FIELD INVENTORY' in json.dumps(content)
        sent.append('selection' if selecting else 'final')
        if selecting:
            reservations.append(harness.final_cost_bound(content, ea.selection_tokens('medium')))
            if len(reservations) == 1:
                return answer({'requests': [{'field_id': field['id'], 'mode': 'metadata'}]}, cost=.6)
            assert 'INSPECTED EVIDENCE' in json.dumps(content)
            return answer({'requests': []})
        assert 'INSPECTED EVIDENCE' in json.dumps(content)
        return answer(complete_reply.labels('The hand reaches.'))
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    out = tmp_path / 'uncapped'
    args = dict(keys=['sk-or-test'], concurrency=1, force=False, max_spend=0.,
        model=harness.DEFAULT_MODEL, reasoning='medium', max_tokens=100, timeout=1, cell_w=128)
    assert harness.run_batch([ep], out, **args) == 0
    result = json.loads((out / (ep.name + '.json')).read_text())
    inspection = result['evidence_inspection']
    assert sent == ['selection', 'selection', 'final']
    assert reservations[0] < ea.MAX_SELECTION_COST < .6 + reservations[1]
    assert inspection['status'] == 'complete' and len(inspection['rounds']) == 2
    assert inspection['inspections'][0]['value'] == {'value': 7}
    assert harness.episode_cost(result) == pytest.approx(.62)
    assert harness.run_batch([ep], out, **args) == 0
    assert sent == ['selection', 'selection', 'final']


def test_native_search_fee_without_enforceable_ceiling_refuses_paid_dispatch(tmp_path, monkeypatch):
    from test_camera_timing import recording
    ep = recording(tmp_path, np.arange(12) / 30)
    def provider(request, timeout):
        assert request.get_method() == 'GET', 'native search exceeded its verified quote'
        return io.BytesIO(json.dumps({'data': {'id': 'perplexity/sonar', 'endpoints': [{
            'context_length': 127072, 'pricing': {'prompt': '.000001', 'completion': '.000001',
                                                'web_search': '.005'}}]}}).encode())
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    out = tmp_path / 'out'
    assert harness.run_batch([ep], out, keys=['sk-or-test'], concurrency=1, force=False,
        model='perplexity/sonar', reasoning='medium', max_tokens=100, timeout=1,
        cell_w=128, max_spend=.128) == 1
    receipt = json.loads((out / ('noreply_' + ep.name + '.json')).read_text())
    assert 'search' in receipt['no_reply'].lower()
    assert harness.episode_cost(receipt) == 0


@pytest.mark.parametrize('limitation', [
    'Inspection selection exceeds the remaining batch spend cap.',
    'Inspection selection stopped at the cost or dispatch boundary.',
])
def test_legacy_paid_final_after_budget_pause_requires_explicit_refresh(tmp_path, monkeypatch, capsys, limitation):
    from test_camera_timing import recording
    ep = recording(tmp_path, np.arange(12) / 30)
    out = tmp_path / 'out'
    out.mkdir()
    args = dict(model=harness.DEFAULT_MODEL, reasoning='medium', max_tokens=100, cell_w=128,
                example_dir=None)
    record = {'parse_ok': True, 'labels': complete_reply.labels('The hand reaches.'),
              'usage': {'est_cost_usd': .01}, 'input_identity': harness.input_identity(ep, **args),
              'evidence_inspection': {'status': 'incomplete', 'rounds': [], 'cost_usd': 0,
                  'limitations': [limitation]}}
    target = out / (ep.name + '.json')
    target.write_text(json.dumps(record))
    original = target.read_bytes()
    monkeypatch.setattr(harness.urllib.request, 'urlopen', lambda *a, **kw: pytest.fail('paid output replayed'))
    assert harness.run_batch([ep], out, keys=['sk-or-test'], concurrency=1, force=False,
                             max_spend=1., timeout=1, **args) == 0
    assert target.read_bytes() == original
    warning = json.loads((out / ('stale_' + ep.name + '.json')).read_text())
    assert warning['refresh'] == '--force' and 'inspection' in warning['reason']
    if 'cost or dispatch boundary' in limitation:
        assert warning['status'] == 'unverified'
        console = capsys.readouterr()
        assert 'LIMITED ' + ep.name in console.err and '--force' in console.err


@pytest.mark.parametrize('contribution,progress,kept', [
    (None, .2, True), ('Advancing', .2, True), ('mostly', .2, True),
    ('advancing', None, True), (None, None, False),
])
def test_short_row_recovery_requires_positional_evidence(contribution, progress, kept):
    row = [0, 1, 'left', 'reach', 'cup', None, None, contribution, progress]
    labels, ok = harness.parse_response(json.dumps({'task_summary': 'reach', 'timeline': [row]}))
    assert ok
    assert labels.get('timeline')
    if kept:
        assert labels['timeline'][0]['contribution'] == contribution
        assert labels['timeline'][0]['progress'] == progress
    else:
        assert labels['_dropped'][0]['row'] == 0
        assert 'contribution' not in labels['timeline'][0] and 'progress' not in labels['timeline'][0]


def test_short_row_with_a_missing_middle_column_is_not_reinterpreted():
    shifted = [0, 1, 'left', 'reach', 'cup', None, 'advancing', None, None]
    labels, ok = harness.parse_response(json.dumps({'task_summary': 'reach', 'timeline': [shifted]}))
    assert ok and labels['timeline'] == [{'start_s': 0, 'end_s': 1}]
    assert labels['_dropped'][0]['row'] == 0


def test_ego_short_row_does_not_guess_whether_text_is_handwear_or_notes():
    columns = harness.TIMELINE_COLUMNS[:-1] + ['hands_visible', 'hands_wearing', 'notes']
    ambiguous = [0, 1, 'left', 'reach', 'cup', None, None, 'advancing', .2, True, 'bare']
    labels, ok = harness.parse_response(json.dumps({'task_summary': 'reach', 'timeline_columns': columns,
                                                   'timeline': [ambiguous]}))
    assert ok and labels['timeline'][0]['action'] == 'reach'
    assert 'hands_wearing' not in labels['timeline'][0] and 'notes' not in labels['timeline'][0]
    assert labels['_dropped'][0]['value'] == ambiguous


# Endpoint pricing published for google/gemini-4-argon on 2026-10-19 (standard tier): audio prices and a search price
# beside the token rates.
GEMINI_PRICING = {'prompt': '0.000002', 'completion': '0.00001', 'image': '0.000002', 'audio': '0.000002',
                  'input_audio_cache': '0.0000001', 'web_search': '0.014', 'internal_reasoning': '0.00001',
                  'input_cache_read': '0.0000001', 'input_cache_write': '0.0000001875', 'discount': 0.5}


def test_quote_caps_a_gemini_request_whose_audio_and_search_prices_cannot_apply(tmp_path, monkeypatch):
    """Gemini lists audio prices and a search price. Requests carry text and images only and never ask for search (the
    web plugin or :online), so the quote accepts both and caps the request by its token and image rates."""
    from test_camera_timing import recording
    ep = recording(tmp_path, np.arange(12) / 30)
    model = 'google/gemini-4-argon'
    sent = []
    def provider(request, timeout):
        sent.append(request.get_method())
        if request.get_method() == 'GET':
            return io.BytesIO(json.dumps({'data': {'id': model, 'endpoints': [{
                'context_length': 1_048_576, 'pricing': GEMINI_PRICING}]}}).encode())
        body = json.loads(request.data)
        assert body['provider']['max_price'] == {'prompt': 2., 'completion': 10., 'image': .000002, 'request': 0.}
        assert 'plugins' not in body and ':online' not in body['model']
        selection = 'FIELD INVENTORY' in json.dumps(body['messages'])
        return answer({'requests': []} if selection else complete_reply.labels('The hand reaches.'))
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    out = tmp_path / 'out'
    assert harness.run_batch([ep], out, keys=['sk-or-test'], concurrency=1, force=False, model=model,
        reasoning='medium', max_tokens=64000, timeout=1, cell_w=128, max_spend=8.) == 0
    assert sent == ['GET', 'POST', 'POST']
    quoted = json.loads((out / (ep.name + '.json')).read_text())['config']['pricing_quote']
    # the output rate carries the reasoning charge; the input rate is the prompt rate, never an audio one
    assert quoted['input_usd_per_token'] == .000002 and quoted['output_usd_per_token'] == pytest.approx(.00002)


@pytest.mark.parametrize('model, pricing, words', [
    ('google/gemini-4-argon', {**GEMINI_PRICING, 'video_seconds': '0.001'}, 'unsupported endpoint charge: video_seconds'),
    ('google/gemini-4-argon:online', GEMINI_PRICING, 'native search fees'),
    ('perplexity/sonar-x', {'prompt': '0.000001', 'completion': '0.000001', 'web_search': '0.005'}, 'native search fees'),
])
def test_a_charge_that_could_apply_still_blocks_the_quote(monkeypatch, model, pricing, words):
    """Accepting Gemini's unsent-input and on-request search prices is scoped to exactly those: a charge the quote does
    not know, a variant that searches, or a model that always searches still has no verified ceiling."""
    from label import pricing as pricing_module
    def provider(request, timeout):
        return io.BytesIO(json.dumps({'data': {'id': model, 'endpoints': [{'context_length': 100_000, 'pricing': pricing}]}}).encode())
    monkeypatch.setattr(pricing_module.urllib.request, 'urlopen', provider)
    with pytest.raises(ValueError, match=words):
        pricing_module.quote(model, 5)


def test_a_request_part_with_no_verified_bound_is_refused():
    """The quote accepts audio prices only because requests carry no audio; a part that is not text or an image would
    break that, so the bound refuses it rather than leave it out."""
    from label import pricing as pricing_module
    quote = {'prompt_token_bound': 100_000, 'input_usd_per_token': 1e-6, 'output_usd_per_token': 1e-5,
             'request_usd': 0., 'image_usd': 0.}
    assert pricing_module.cost_bound([{'type': 'text', 'text': 'hello'}], 100, quote) > 0
    with pytest.raises(ValueError, match='no verified bound: input_audio'):
        pricing_module.cost_bound([{'type': 'text', 'text': 'hello'}, {'type': 'input_audio', 'input_audio': {}}], 100, quote)
