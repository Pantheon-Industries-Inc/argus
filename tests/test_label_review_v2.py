"""Exercise review reports through real parsing, prepared requests, and HTTP dispatch."""
import io
import json
import urllib.error

import numpy as np
import pytest

from label import dictionary as dd, harness


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
def test_dictionary_ambiguous_openrouter_error_keeps_claim(tmp_path, monkeypatch, status, body):
    from test_dictionary import episode
    ep = episode(tmp_path, 'one', [('pad', np.ones((2, 1)), {})])
    sent = []
    def provider(request, timeout):
        sent.append(request.full_url)
        reject(request, status, body)
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    job = tmp_path / 'job'
    result = dd.prepare_upload(job, [ep], None, harness.call_model_once,
                               api_keys=['sk-or-first', 'sk-or-second'])
    assert result['status'] == 'failed' and result['cost_usd'] is None
    assert result['dispatch_attempts'] == [{'attempt': 1, 'state': 'dispatched'}]
    assert dd.prepare_upload(job, [ep], None, harness.call_model_once,
                             api_keys=['sk-or-first', 'sk-or-second']) == result
    assert len(sent) == 1


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
    reply = {'timeline_columns': columns, 'timeline': [row]}
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
def test_force_refresh_is_consumed_once_when_key_is_rejected(tmp_path, monkeypatch, rejected_stage):
    from test_camera_timing import recording
    ep = recording(tmp_path, np.arange(12) / 30)
    sent = []
    fail_final = [False]
    def provider(request, timeout):
        selection = 'FIELD INVENTORY' in json.dumps(json.loads(request.data)['messages'])
        key = request.get_header('Authorization')
        sent.append((key, 'selection' if selection else 'final'))
        if fail_final[0] and selection == (rejected_stage == 'selection') and key.endswith('first'):
            reject(request, 429, {'error': {'code': 'insufficient_quota'}})
        return answer({'requests': []} if selection else {'task_summary': 'The hand reaches for the cup.'})
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
    assert harness.run_batch([ep], out, keys=['sk-or-test'], concurrency=1, force=False, max_spend=1., **args) == 0
    assert target.read_bytes() == original


@pytest.mark.parametrize('status', [429, 503])
def test_ambiguous_final_http_error_retains_reserve_and_never_replays(tmp_path, monkeypatch, status):
    from test_camera_timing import recording
    ep = recording(tmp_path, np.arange(12) / 30)
    sent = []
    def provider(request, timeout):
        selection = 'FIELD INVENTORY' in json.dumps(json.loads(request.data)['messages'])
        sent.append('selection' if selection else 'final')
        if not selection:
            reject(request, status, {'error': {'message': 'request outcome unavailable'}})
        return answer({'requests': []})
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    out = tmp_path / 'out'
    args = dict(keys=['sk-or-first', 'sk-or-second'], concurrency=1, force=False, max_spend=1.,
                model=harness.DEFAULT_MODEL, reasoning='medium', max_tokens=100, timeout=1, cell_w=128)
    assert harness.run_batch([ep], out, **args) == 1
    receipt = out / ('noreply_' + ep.name + '.json')
    result = json.loads(receipt.read_text())
    assert result['final_dispatch_outcome'] == 'unverified' and result['final_reserved_usd'] > 0
    assert harness.episode_cost(result) > .01
    original = receipt.read_bytes()
    assert harness.run_batch([ep], out, **args) == 0
    assert receipt.read_bytes() == original and sent == ['selection', 'final']


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
        return answer({'requests': []} if selection else {'task_summary': 'The hand reaches.'})
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
    assert pricing.cost_bound([], 64000, quote) == pytest.approx(5.28)


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
        return answer({'requests': []} if selection else {'task_summary': 'The hand reaches.'})
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
        return answer({'requests': []} if selection else {'task_summary': 'The hand reaches for the cup.'})
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


@pytest.mark.parametrize('cap', [2., 8.])
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
        return answer({'requests': []} if selection else {'task_summary': 'The hand reaches.'})
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    out = tmp_path / 'out'
    assert harness.run_batch([ep], out, keys=['sk-or-test'], concurrency=1, force=False,
        model=model, reasoning='medium', max_tokens=64000, timeout=1, cell_w=128, max_spend=cap) == 0
    if cap == 2.:
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
        return answer({'requests': []} if selection else {'timeline': [row]})
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


@pytest.mark.parametrize('initial_cap', [.01, .35])
def test_budget_increase_resumes_undispatched_selection(tmp_path, monkeypatch, initial_cap):
    from test_camera_timing import recording
    ep = recording(tmp_path, np.arange(12) / 30)
    sent = []
    def provider(request, timeout):
        if request.get_method() == 'GET':
            return io.BytesIO(json.dumps(endpoint_metadata('custom/model')).encode())
        selection = 'FIELD INVENTORY' in json.dumps(json.loads(request.data)['messages'])
        sent.append('selection' if selection else 'final')
        return answer({'requests': []} if selection else {'task_summary': 'The hand reaches.'})
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    out = tmp_path / 'out'
    args = dict(keys=['sk-or-test'], concurrency=1, force=False, model='custom/model',
                reasoning='medium', max_tokens=100, timeout=1, cell_w=128)
    assert harness.run_batch([ep], out, max_spend=initial_cap, **args) == 0
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


def test_legacy_paid_final_after_budget_pause_requires_explicit_refresh(tmp_path, monkeypatch):
    from test_camera_timing import recording
    ep = recording(tmp_path, np.arange(12) / 30)
    out = tmp_path / 'out'
    out.mkdir()
    args = dict(model=harness.DEFAULT_MODEL, reasoning='medium', max_tokens=100, cell_w=128,
                example_dir=None)
    record = {'parse_ok': True, 'labels': {'task_summary': 'The hand reaches.'},
              'usage': {'est_cost_usd': .01}, 'input_identity': harness.input_identity(ep, **args),
              'evidence_inspection': {'status': 'incomplete', 'rounds': [], 'cost_usd': 0,
                  'limitations': ['Inspection selection exceeds the remaining batch spend cap.']}}
    target = out / (ep.name + '.json')
    target.write_text(json.dumps(record))
    original = target.read_bytes()
    monkeypatch.setattr(harness.urllib.request, 'urlopen', lambda *a, **kw: pytest.fail('paid output replayed'))
    assert harness.run_batch([ep], out, keys=['sk-or-test'], concurrency=1, force=False,
                             max_spend=1., timeout=1, **args) == 0
    assert target.read_bytes() == original
    warning = json.loads((out / ('stale_' + ep.name + '.json')).read_text())
    assert warning['refresh'] == '--force' and 'inspection' in warning['reason']


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
