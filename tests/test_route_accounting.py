"""Routing dispatches retain spend through build failures and resumed runs."""
import json
import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from label import harness, route
from label.dictionary_stage import label_spend, label_spend_complete


def episode(root, name='episode_a', instruction='Place the cup.'):
    ep = root / name
    ep.mkdir()
    (ep / 'context.json').write_text(json.dumps({
        'dataset': 'tiny', 'profile': 'teleop_arms', 'state_kind': 'none',
        'fps': 30, 'instruction': instruction}))
    (ep / 'sources.json').write_text(json.dumps({'exo': {'n_frames': 2}}))
    return ep


def build_failure(*args, **kwargs):
    raise RuntimeError('fixture frame decode failure')


def args(**changes):
    return dict(model=harness.DEFAULT_MODEL, reasoning='medium', max_tokens=100,
                timeout=1, **changes)


@pytest.mark.parametrize('received_first', [False, True])
def test_http_rejected_selection_fails_over_without_replaying_received_rounds(tmp_path, monkeypatch, received_first):
    import io
    import urllib.error
    from label import evidence_access as ea
    from test_label import _packed_episode
    ep, _ = _packed_episode(tmp_path)
    job = tmp_path / 'job'
    out = job / 'run' / 'out'
    calls = []
    class Response(io.BytesIO):
        pass
    def provider(req, timeout):
        content = json.loads(req.data)['messages'][0]['content']
        key = req.get_header('Authorization').split()[-1]
        selection = next((p['text'] for p in content if p.get('text', '').startswith(ea.SELECTION)), None)
        inspected = any(p.get('text', '').startswith('INSPECTED EVIDENCE') for p in content)
        calls.append((key, selection is not None, inspected))
        if key == 'sk-or-empty' and (not received_first or inspected):
            raise urllib.error.HTTPError(req.full_url, 402, 'Payment Required', {}, io.BytesIO(b'{"error":{"code":402}}'))
        if selection is not None and not inspected:
            page = json.loads(selection.split('FIELD INVENTORY\n', 1)[1])
            field = next(f for f in page['fields'] if 'samples' in f.get('available_modes', []))
            reply = {'requests': [{'field_id': field['id'], 'mode': 'samples'}]}
        else:
            reply = {'requests': []} if selection is not None else {'timeline': [], 'task_summary': 'Wait beside the visible workspace.'}
        return Response(json.dumps({'usage': {'cost': .004 if selection is not None else .01},
            'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(reply)}}]}).encode())
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    assert harness.run_batch([ep], out, keys=['sk-or-empty', 'sk-or-funded'], concurrency=1,
        force=False, max_spend=2, **args(cell_w=128)) == 0
    result = json.loads((out / f'{ep.name}.json').read_text())
    inspection = result['evidence_inspection']
    assert result['parse_ok'] and inspection['status'] == 'complete' and inspection['inspections']
    assert len(inspection['rounds']) == 2 and all(r['state'] == 'received' for r in inspection['rounds'])
    rejected = inspection['rounds'][1 if received_first else 0]['rejected_attempts']
    assert len(rejected) == 1 and rejected[0]['http_status'] == 402
    assert rejected[0]['state'] == 'rejected unpaid' and rejected[0]['cost_usd'] == 0
    assert len(calls) == 4
    assert calls.count(('sk-or-empty', True, False)) == 1
    assert calls.count(('sk-or-funded', True, False)) == (0 if received_first else 1)
    assert label_spend(job) == pytest.approx(.018) and label_spend_complete(job)


@pytest.mark.parametrize('stage,status', [('inspection', 402), ('routing', 401)])
def test_module_cli_retires_rejected_key_without_losing_paid_inspection_or_route(tmp_path, monkeypatch, stage, status):
    import io
    import runpy
    import sys
    import urllib.error
    from label import evidence_access as ea
    from test_label import _packed_episode
    ep = _packed_episode(tmp_path)[0] if stage == 'inspection' else observed_packed_episode(tmp_path)
    job = tmp_path / 'job'
    out = job / 'run' / 'out'
    calls = []
    def provider(req, timeout):
        body = json.loads(req.data)
        content = body['messages'][0]['content']
        key = req.get_header('Authorization').split()[-1]
        selection = next((p['text'] for p in content if p.get('text', '').startswith(ea.SELECTION)), None)
        inspected = any(p.get('text', '').startswith('INSPECTED EVIDENCE') for p in content)
        routed = body['model'] == route.ROUTE_MODEL
        calls.append((key, routed, selection is not None, inspected))
        if key == 'sk-or-empty' and (stage == 'routing' or inspected):
            raise urllib.error.HTTPError(req.full_url, status, 'Rejected', {}, io.BytesIO(b'{}'))
        if routed:
            answer = {'fine_detail': False, 'why': 'whole-object view is sufficient'}
        elif selection is not None and not inspected:
            page = json.loads(selection.split('FIELD INVENTORY\n', 1)[1])
            field = next(f for f in page['fields'] if 'samples' in f.get('available_modes', []))
            answer = {'requests': [{'field_id': field['id'], 'mode': 'samples'}]}
        else:
            answer = {'requests': []} if selection is not None else {'timeline': [], 'task_summary': 'Wait beside the visible workspace.'}
        return io.BytesIO(json.dumps({'usage': {'cost': .004 if routed or selection is not None else .01},
            'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(answer)}}]}).encode())
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    monkeypatch.setattr(route, '_CACHE', {})
    monkeypatch.setenv('OPENROUTER_API_KEYS', 'sk-or-empty,sk-or-funded')
    monkeypatch.setattr(sys, 'argv', ['label.harness', '--episodes-root', str(tmp_path), '--out-dir', str(out),
        '--model', harness.DEFAULT_MODEL, '--reasoning', 'medium', '--max-tokens', '100', '--timeout', '1',
        '--concurrency', '1', '--max-spend', '2', '--cell-w', '128' if stage == 'inspection' else '0'])
    with pytest.warns(RuntimeWarning, match='label.harness.*found in sys.modules'), pytest.raises(SystemExit) as finished:
        runpy.run_module('label.harness', run_name='__main__', alter_sys=True)
    assert finished.value.code == 0
    result = json.loads((out / f'{ep.name}.json').read_text())
    assert result['parse_ok'] and result['model'] == harness.DEFAULT_MODEL
    assert calls[-1][0] == 'sk-or-funded' and len(calls) == (4 if stage == 'inspection' else 3)
    if stage == 'inspection':
        inspection = result['evidence_inspection']
        assert inspection['status'] == 'complete' and inspection['inspections']
        assert all(r['state'] == 'received' for r in inspection['rounds'])
        rejected = inspection['rounds'][1]['rejected_attempts']
        assert len(rejected) == 1 and rejected[0]['http_status'] == status and rejected[0]['cost_usd'] == 0
        assert calls.count(('sk-or-empty', False, True, False)) == 1
        assert not any(key == 'sk-or-funded' and selected and not inspected for key, _, selected, inspected in calls)
    else:
        routing = result['config']['resolution_route']
        assert routing['dispatch_outcome'] == 'received' and routing['cost_usd'] == pytest.approx(.004)
        assert calls[:2] == [('sk-or-empty', True, False, False), ('sk-or-funded', True, False, False)]
    assert label_spend(job) == pytest.approx(.018 if stage == 'inspection' else .014)
    assert label_spend_complete(job)


@pytest.mark.parametrize('response_kind', ['body_error', 'timeout', 'unproven_key_error'])
def test_unverified_final_key_error_never_releases_hold_or_fails_over(tmp_path, monkeypatch, response_kind):
    import io
    ep = observed_packed_episode(tmp_path)
    out = tmp_path / 'job' / 'run' / 'out'
    calls = []
    def provider(req, timeout):
        calls.append(req.get_header('Authorization'))
        if response_kind == 'unproven_key_error':
            raise harness.KeyExhausted('HTTP 402: no typed response proof')
        if response_kind == 'timeout':
            raise TimeoutError('response acceptance unknown')
        return io.BytesIO(b'{"error":{"code":402,"message":"insufficient credits"}}')
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    assert harness.run_batch([ep], out, keys=['sk-or-empty', 'sk-or-funded'], concurrency=1,
        force=False, max_spend=2, **args(cell_w=128)) == 1
    assert calls == ['Bearer sk-or-empty']
    failure = json.loads((out / f'noreply_{ep.name}.json').read_text())
    assert failure['final_dispatch_outcome'] == 'unverified' and failure['final_reserved_usd'] > 0
    assert label_spend(tmp_path / 'job') >= failure['final_reserved_usd']
    assert not label_spend_complete(tmp_path / 'job')


@pytest.mark.parametrize('status', [401, 402])
@pytest.mark.parametrize('prior_unknown', [False, True])
def test_http_rejection_proof_applies_only_before_any_unknown_attempt(monkeypatch, status, prior_unknown):
    import io
    import urllib.error
    calls = []
    def provider(req, timeout):
        calls.append(1)
        if prior_unknown and len(calls) == 1:
            raise TimeoutError('first acceptance unknown')
        raise urllib.error.HTTPError(req.full_url, status, 'Rejected', {}, io.BytesIO(b'{}'))
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    monkeypatch.setattr(harness.time, 'sleep', lambda _: None)
    with pytest.raises(RuntimeError) as caught:
        harness._post(harness.OPENROUTER_URL, {}, 'sk-or-fixture', 1, attempts=2)
    if prior_unknown:
        assert not isinstance(caught.value, harness.KeyExhausted) and len(calls) == 2
    else:
        assert caught.value.verified_unpaid and caught.value.http_status == status and calls == [1]


def test_native_rgb_final_http_rejection_reuses_exact_payload_on_next_key(tmp_path, monkeypatch):
    import io
    import urllib.error
    ep = observed_packed_episode(tmp_path)
    job = tmp_path / 'job'
    out = job / 'run' / 'out'
    requests = []
    def provider(req, timeout):
        requests.append((req.get_header('Authorization'), req.data))
        if len(requests) == 1:
            raise urllib.error.HTTPError(req.full_url, 402, 'Payment Required', {}, io.BytesIO(b'{}'))
        return io.BytesIO(json.dumps({'usage': {'cost': .01}, 'choices': [{'finish_reason': 'stop',
            'message': {'content': '{"timeline":[], "task_summary":"Wait beside the visible workspace."}'}}]}).encode())
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    assert harness.run_batch([ep], out, keys=['sk-or-empty', 'sk-or-funded'], concurrency=1,
        force=False, max_spend=2, **args(cell_w=128)) == 0
    assert [row[0] for row in requests] == ['Bearer sk-or-empty', 'Bearer sk-or-funded']
    assert requests[0][1] == requests[1][1]
    result = json.loads((out / f'{ep.name}.json').read_text())
    assert result['parse_ok'] and result['final_attempt_index'] == 2
    assert label_spend(job) == pytest.approx(.01) and label_spend_complete(job)


def test_unproven_routing_key_error_preserves_unknown_route_charge(tmp_path, monkeypatch):
    ep = observed_packed_episode(tmp_path)
    job = tmp_path / 'job'
    out = job / 'run' / 'out'
    calls = []
    def provider(content, model, reasoning, api_key, *a, **kw):
        calls.append((model, api_key))
        if model == route.ROUTE_MODEL:
            raise harness.KeyExhausted('legacy callable error, no server rejection proof')
        return {'usage': {'cost': .01}, 'choices': [{'finish_reason': 'stop',
                'message': {'content': '{"timeline":[], "task_summary":"Wait beside the visible workspace."}'}}]}
    monkeypatch.setattr(harness, 'call_model', provider)
    monkeypatch.setattr(route, '_CACHE', {})
    assert harness.run_batch([ep], out, keys=['sk-or-empty', 'sk-or-funded'], concurrency=1,
        force=False, max_spend=2, **args()) == 0
    result = json.loads((out / f'{ep.name}.json').read_text())
    routing = result['config']['resolution_route']
    assert routing['dispatch_outcome'] == 'unverified' and routing['cost_usd'] > 0
    assert [key for _, key in calls] == ['sk-or-empty', 'sk-or-empty']
    assert label_spend(job) == pytest.approx(.01 + routing['cost_usd'])
    assert not label_spend_complete(job)


def observed_packed_episode(root):
    """A recording with observed joints only, for isolated route/final invoice contracts."""
    import numpy as np
    from test_label import _packed_episode
    from label import episode as me, evidence_access
    ep, _ = _packed_episode(root)
    with np.load(ep / 'state.npz') as stored:
        observed = stored['state']
    np.savez(ep / 'state.npz', state=observed)
    loaded = me.load(ep)
    assert loaded['action'] is None
    assert not evidence_access.needs_inspection(loaded, {})
    return ep


@pytest.mark.parametrize('stage', ['build', 'inspection'])
def test_changed_source_during_request_assembly_cannot_dispatch_stale_annotation(tmp_path, monkeypatch, stage):
    from test_label import _packed_episode, _accounted_evidence_selection
    ep = observed_packed_episode(tmp_path) if stage == 'build' else _packed_episode(tmp_path)[0]
    out = tmp_path / 'out' / f'{ep.name}.json'
    calls = []
    def change_instruction():
        path = ep / 'context.json'
        ctx = json.loads(path.read_text())
        ctx['instruction'] = 'A new publisher instruction while the request is being assembled.'
        path.write_text(json.dumps(ctx))
    original = harness.me.build_request
    def build_request(*args, **kwargs):
        result = original(*args, **kwargs)
        if stage == 'build':
            change_instruction()
        return result
    def provider(content, *args, **kwargs):
        selection = _accounted_evidence_selection(content)
        assert selection is not None, 'A final annotation must not use changed source interpretation.'
        calls.append('selection')
        change_instruction()
        return selection
    monkeypatch.setattr(harness.me, 'build_request', build_request)
    monkeypatch.setattr(harness, 'call_model_once', provider)
    with pytest.raises(RuntimeError, match='inputs changed'):
        harness.label_episode(ep, out, model='openai/gpt-6-astra', reasoning='medium', api_key='sk-or-fixture',
                              max_tokens=2000, timeout=1, cell_w=128, dry_run=stage == 'build')
    assert not out.exists()
    receipt = json.loads(out.with_name('noreply_' + out.name).read_text())
    assert receipt['final_dispatch_outcome'] == 'not dispatched'
    assert receipt['final_reserved_usd'] == 0
    assert calls == ([] if stage == 'build' else ['selection'])
    assert harness.episode_cost(receipt) == pytest.approx(0 if stage == 'build' else .004)


def test_paid_route_survives_request_build_failure_and_resume(tmp_path, monkeypatch):
    ep = episode(tmp_path)
    job = tmp_path / 'job'
    out = job / 'run' / 'out'
    calls = []
    def provider(*a, **kw):
        calls.append(kw.get('attempts'))
        return {'usage': {'cost': .004}, 'choices': [{'message': {
            'content': '{"fine_detail": false, "why": "whole objects"}'}}]}
    monkeypatch.setattr(route, '_CACHE', {})
    monkeypatch.setattr(harness, 'call_model', provider)
    monkeypatch.setattr(harness.me, 'build_request', build_failure)
    batch = dict(keys=['sk-or-fixture'], concurrency=1, force=False, max_spend=1, **args())
    assert harness.run_batch([ep], out, **batch) == 1
    assert label_spend(job) == pytest.approx(.004)
    assert label_spend_complete(job) is True
    monkeypatch.setattr(route, '_CACHE', {})
    assert harness.run_batch([ep], out, **batch) == 1
    assert calls == [1]
    assert label_spend(job) == pytest.approx(.004)


def test_route_respects_batch_budget_before_dispatch(tmp_path, monkeypatch):
    ep = episode(tmp_path)
    calls = []
    def provider(*a, **kw):
        calls.append(1)
        return {'usage': {'cost': .004}, 'choices': [{'message': {'content': '{}'}}]}
    monkeypatch.setattr(route, '_CACHE', {})
    monkeypatch.setattr(harness, 'call_model', provider)
    monkeypatch.setattr(harness.me, 'build_request', build_failure)
    harness.run_batch([ep], tmp_path / 'out', keys=['sk-or-fixture'], concurrency=1,
                      force=False, max_spend=.00001, **args())
    assert calls == []


@pytest.mark.parametrize('choices', [[], [{'message': {'content': None}}]])
def test_malformed_route_retains_verified_invoice_and_falls_back(tmp_path, monkeypatch, choices):
    ep = episode(tmp_path)
    job = tmp_path / 'job'
    calls = []
    def provider(*a, **kw):
        calls.append(kw['attempts'])
        return {'usage': {'cost': .004}, 'choices': choices}
    monkeypatch.setattr(route, '_CACHE', {})
    monkeypatch.setattr(harness, 'call_model', provider)
    monkeypatch.setattr(harness.me, 'build_request', build_failure)
    out = job / 'run' / 'out' / 'episode_a.json'
    for _ in range(2):
        with pytest.raises(RuntimeError, match='fixture frame decode failure'):
            harness.label_episode(ep, out, api_key='sk-or-fixture', **args())
        monkeypatch.setattr(route, '_CACHE', {})
    saved = json.loads(out.with_name('noreply_episode_a.json').read_text())
    assert saved['config']['resolution_route']['fine_detail'] is None
    assert calls == [1]
    assert label_spend(job) == pytest.approx(.004)
    assert label_spend_complete(job) is True


@pytest.mark.parametrize('response', [TimeoutError('ambiguous routing dispatch'),
                                    {'choices': [{'message': {'content': '{}'}}]},
                                    {'usage': {'cost': -1}, 'choices': [{'message': {'content': '{}'}}]},
                                    {'usage': {'cost': True}, 'choices': [{'message': {'content': '{}'}}]}])
def test_unknown_route_cost_is_reserved_and_never_rerouted(tmp_path, monkeypatch, response):
    ep = episode(tmp_path)
    job = tmp_path / 'job'
    out = job / 'run' / 'out'
    calls = []
    def provider(*a, **kw):
        calls.append(kw.get('attempts'))
        if isinstance(response, Exception):
            raise response
        return response
    monkeypatch.setattr(route, '_CACHE', {})
    monkeypatch.setattr(harness, 'call_model', provider)
    monkeypatch.setattr(harness.me, 'build_request', build_failure)
    batch = dict(keys=['sk-or-fixture'], concurrency=1, force=False, max_spend=1, **args())
    assert harness.run_batch([ep], out, **batch) == 1
    reserved = label_spend(job)
    assert 0 < reserved < 1
    assert label_spend_complete(job) is False
    monkeypatch.setattr(route, '_CACHE', {})
    assert harness.run_batch([ep], out, **batch) == 1
    assert calls == [1]
    assert label_spend(job) == reserved


def _crash_after_route_dispatch(ep, out):
    route._CACHE = {}
    def provider(*a, **kw):
        assert kw['attempts'] == 1
        raise SystemExit(9)
    harness.call_model = provider
    harness.label_episode(Path(ep), Path(out), api_key='sk-or-fixture', **args())


def test_route_claim_survives_process_exit_and_rejects_changed_input(tmp_path, monkeypatch):
    ep = episode(tmp_path)
    job = tmp_path / 'job'
    out = job / 'run' / 'out' / 'episode_a.json'
    process = multiprocessing.get_context('spawn').Process(
        target=_crash_after_route_dispatch, args=(str(ep), str(out)))
    process.start()
    process.join(5)
    assert not process.is_alive() and process.exitcode == 9
    receipt = out.with_name('noreply_episode_a.json')
    saved = json.loads(receipt.read_text())
    assert saved['config']['resolution_route']['dispatch_outcome'] == 'claimed'
    reserved = label_spend(job)
    assert 0 < reserved < 1 and label_spend_complete(job) is False
    monkeypatch.setattr(route, '_CACHE', {})
    monkeypatch.setattr(harness, 'call_model', lambda *a, **kw: pytest.fail('route was redispatched'))
    monkeypatch.setattr(harness.me, 'build_request', build_failure)
    with pytest.raises(RuntimeError, match='fixture frame decode failure'):
        harness.label_episode(ep, out, api_key='sk-or-fixture', **args())
    assert label_spend(job) == reserved
    ctx = json.loads((ep / 'context.json').read_text())
    ctx['instruction'] = 'Read the cup lettering.'
    (ep / 'context.json').write_text(json.dumps(ctx))
    with pytest.raises(ValueError, match='changed or missing input'):
        harness.label_episode(ep, out, api_key='sk-or-fixture', **args())
    assert json.loads(receipt.read_text()) == saved


def test_distinct_routes_dispatch_concurrently_and_keep_build_failure_costs(tmp_path, monkeypatch):
    eps = [episode(tmp_path, 'episode_a', 'Place the cup.'),
           episode(tmp_path, 'episode_b', 'Read the label.')]
    barrier = threading.Barrier(2)
    calls = []
    def provider(*a, **kw):
        calls.append(kw['attempts'])
        barrier.wait(timeout=2)
        return {'usage': {'cost': .004}, 'choices': [{'message': {'content': '{}'}}]}
    monkeypatch.setattr(route, '_CACHE', {})
    monkeypatch.setattr(harness, 'call_model', provider)
    monkeypatch.setattr(harness.me, 'build_request', build_failure)
    job = tmp_path / 'job'
    assert harness.run_batch(eps, job / 'run' / 'out', keys=['sk-or-fixture'], concurrency=2,
                             force=False, max_spend=1, **args()) == 1
    assert calls == [1, 1]
    assert label_spend(job) == pytest.approx(.008)
    assert label_spend_complete(job) is True


def test_concurrent_routes_cannot_dispatch_beyond_reserved_budget(tmp_path, monkeypatch):
    eps = [episode(tmp_path, 'episode_a', 'Place the cup.'),
           episode(tmp_path, 'episode_b', 'Read the label.')]
    entered, release, refused = threading.Event(), threading.Event(), threading.Event()
    calls = []
    def provider(*a, **kw):
        calls.append(kw['attempts'])
        entered.set()
        assert release.wait(2)
        return {'usage': {'cost': .004}, 'choices': [{'message': {'content': '{}'}}]}
    original = route._paid_route
    def paid_route(*a, **kw):
        try:
            return original(*a, **kw)
        except harness.SpendCap:
            refused.set()
            raise
    monkeypatch.setattr(route, '_CACHE', {})
    monkeypatch.setattr(route, '_paid_route', paid_route)
    monkeypatch.setattr(harness, 'call_model', provider)
    monkeypatch.setattr(harness.me, 'build_request', build_failure)
    job = tmp_path / 'job'
    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(harness.run_batch, eps, job / 'run' / 'out',
            keys=['sk-or-fixture'], concurrency=2, force=False, max_spend=.04, **args())
        try:
            assert entered.wait(2)
        finally:
            release.set()
        assert future.result(timeout=2) == 1
    assert refused.is_set()
    assert calls == [1]
    assert label_spend(job) == pytest.approx(.004)


@pytest.mark.parametrize('first_reply', ['received', 'unknown', 'changed_input'])
def test_parallel_final_waits_for_live_reservation_but_preserves_unknown_charge(tmp_path, monkeypatch, first_reply):
    from types import SimpleNamespace
    entered, waiting, release = threading.Event(), threading.Event(), threading.Event()
    class ObservedCondition(threading.Condition):
        def wait(self, timeout=None):
            waiting.set()
            return super().wait(timeout)
    monkeypatch.setattr(harness, 'threading', SimpleNamespace(Condition=ObservedCondition, Lock=threading.Lock,
                                                           Event=threading.Event))
    eps = []
    for name in ('a', 'b'):
        root = tmp_path / name
        root.mkdir()
        ep = observed_packed_episode(root)
        ep = ep.rename(root / ('episode_' + name))
        eps.append(ep)
    job = tmp_path / 'job'
    out = job / 'run' / 'out'
    bounds, calls = [], []
    original_bound = harness.final_cost_bound
    def bound(content, tokens, **kwargs):
        value = original_bound(content, tokens, **kwargs)
        bounds.append(value)
        return value
    def provider(*a, **kw):
        calls.append(1)
        assert len(calls) <= 2
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
            if first_reply == 'unknown':
                raise TimeoutError('final charge is unknown')
            if first_reply == 'changed_input':
                claimed = {p.name.removeprefix('noreply_').removesuffix('.json') for p in out.glob('noreply_*.json')}
                pending = next(ep for ep in eps if ep.name not in claimed)
                path = pending / 'context.json'
                ctx = json.loads(path.read_text())
                ctx['instruction'] = 'A changed publisher instruction during the budget wait.'
                path.write_text(json.dumps(ctx))
        return {'usage': {'cost': .01}, 'choices': [{'finish_reason': 'stop',
                'message': {'content': '{"timeline": [], "task_summary": "Wait beside the visible workspace."}'}}]}
    monkeypatch.setattr(harness, 'final_cost_bound', bound)
    monkeypatch.setattr(harness, 'call_model', provider)
    cap = max(original_bound(harness.me.build_request(ep, cell_w=128)['content'], 100) for ep in eps) + .02
    with ThreadPoolExecutor(1) as executor:
        future = executor.submit(harness.run_batch, eps, out, keys=['sk-or-fixture'], concurrency=2,
            force=False, max_spend=cap, **dict(args(cell_w=128), model=None))
        try:
            assert entered.wait(5) and waiting.wait(5)
            assert calls == [1]
        finally:
            release.set()
        assert future.result(timeout=5) == (0 if first_reply == 'received' else 1)
    assert len(calls) == (2 if first_reply == 'received' else 1)
    assert label_spend(job) == pytest.approx(.02 if first_reply == 'received' else
        bounds[0] if first_reply == 'unknown' else .01)
    assert label_spend_complete(job) is (first_reply != 'unknown')
    assert len(list(out.glob('episode_*.json'))) == (2 if first_reply == 'received' else 1 if first_reply == 'changed_input' else 0)
    for path in out.glob('noreply_episode_*.json'):
        receipt = json.loads(path.read_text())
        assert receipt['model'] == receipt['input_identity']['settings']['model']
        if first_reply == 'unknown' and receipt['final_dispatch_outcome'] == 'not dispatched':
            assert receipt['no_reply_kind'] == 'spend_cap'


def test_exhausted_review_budget_retains_new_request_identity_and_prior_unknown_charge(tmp_path, monkeypatch):
    ep = observed_packed_episode(tmp_path)
    job = tmp_path / 'job'
    out = job / 'run' / 'out' / (ep.name + '.json')
    request = harness.label_episode(ep, tmp_path / 'dry.json', api_key='dry-run', dry_run=True,
                                    **args(cell_w=128))
    monkeypatch.setattr(harness, 'call_model', lambda *a, **kw: pytest.fail('budget refusal dispatched'))
    assert harness.record_budget_refusal(out, request)
    refusal = out.with_name('noreply_' + out.name)
    receipt = json.loads(refusal.read_text())
    assert receipt['model'] == request['model']
    assert receipt['input_identity'] == request['input_identity']
    assert receipt['final_dispatch_outcome'] == 'not dispatched' and receipt['no_reply_kind'] == 'spend_cap'
    assert label_spend(job) == 0 and label_spend_complete(job)
    # A previous dispatched request is authoritative even if a later Review has no allowance.
    receipt.update(final_dispatch_outcome='unverified', final_reserved_usd=6.4)
    refusal.write_text(json.dumps(receipt))
    prior = refusal.read_bytes()
    assert not harness.record_budget_refusal(out, request)
    assert refusal.read_bytes() == prior
    assert label_spend(job) == 6.4 and not label_spend_complete(job)


def test_same_task_route_cost_belongs_only_to_the_dispatched_episode(tmp_path, monkeypatch):
    eps = [episode(tmp_path, 'episode_a'), episode(tmp_path, 'episode_b')]
    calls = []
    def provider(*a, **kw):
        calls.append(kw['attempts'])
        return {'usage': {'cost': .004}, 'choices': [{'message': {'content': '{}'}}]}
    monkeypatch.setattr(route, '_CACHE', {})
    monkeypatch.setattr(harness, 'call_model', provider)
    monkeypatch.setattr(harness.me, 'build_request', build_failure)
    job = tmp_path / 'job'
    assert harness.run_batch(eps, job / 'run' / 'out', keys=['sk-or-fixture'], concurrency=2,
                             force=False, max_spend=1, **args()) == 1
    assert calls == [1]
    assert label_spend(job) == pytest.approx(.004)


def test_route_build_failure_settles_shared_budget_once(tmp_path, monkeypatch):
    eps = [episode(tmp_path, f'episode_{i}', f'Place the cup {i}.') for i in range(3)]
    calls = []
    def provider(*a, **kw):
        calls.append(kw['attempts'])
        return {'usage': {'cost': .004}, 'choices': [{'message': {'content': 'not JSON'}}]}
    monkeypatch.setattr(route, '_CACHE', {})
    monkeypatch.setattr(harness, 'call_model', provider)
    monkeypatch.setattr(harness.me, 'build_request', build_failure)
    job = tmp_path / 'job'
    assert harness.run_batch(eps, job / 'run' / 'out', keys=['sk-or-fixture'], concurrency=1,
                             force=False, max_spend=.043, **args()) == 1
    assert calls == [1, 1]
    assert label_spend(job) == pytest.approx(.008)
    assert label_spend_complete(job) is True


@pytest.mark.parametrize('mode', ['explicit', 'nonteleop', 'dry', 'no_task'])
def test_routing_guards_make_no_paid_call(tmp_path, monkeypatch, mode):
    ep = episode(tmp_path)
    ctx = json.loads((ep / 'context.json').read_text())
    if mode == 'nonteleop':
        ctx['profile'] = 'ego_head'
    if mode == 'no_task':
        ctx.pop('instruction')
    (ep / 'context.json').write_text(json.dumps(ctx))
    monkeypatch.setattr(harness, 'call_model', lambda *a, **kw: pytest.fail('unexpected paid call'))
    monkeypatch.setattr(harness.me, 'build_request', build_failure)
    job = tmp_path / 'job'
    with pytest.raises(RuntimeError, match='fixture frame decode failure'):
        harness.label_episode(ep, job / 'run' / 'out' / 'episode_a.json', api_key='sk-or-fixture',
            **args(cell_w=128 if mode == 'explicit' else 0, dry_run=mode == 'dry'))
    assert label_spend(job) == 0 and label_spend_complete(job) is True


@pytest.mark.parametrize('route_usage, complete', [({'cost': .004}, True), ({}, False)])
def test_successful_final_retains_routing_invoice_and_width(tmp_path, monkeypatch, route_usage, complete):
    ep = observed_packed_episode(tmp_path)
    job = tmp_path / 'job'
    out = job / 'run' / 'out' / f'{ep.name}.json'
    calls = []
    def provider(content, model, *a, **kw):
        calls.append((model, kw['attempts']))
        if model == route.ROUTE_MODEL:
            return {'usage': route_usage, 'choices': [{'message': {
                'content': '{"fine_detail": false, "why": "whole objects"}'}}]}
        return {'usage': {'cost': .01}, 'choices': [{'finish_reason': 'stop',
                'message': {'content': '{"timeline": [], "task_summary": "Wait beside the visible workspace."}'}}]}
    monkeypatch.setattr(route, '_CACHE', {})
    monkeypatch.setattr(harness, 'call_model', provider)
    result = harness.label_episode(ep, out, api_key='sk-or-fixture', **args())
    assert calls == [(route.ROUTE_MODEL, 1), (harness.DEFAULT_MODEL, 1)]
    assert result['parse_ok'] and result['config']['cell'][0] == 224
    route_cost = result['config']['resolution_route']['cost_usd']
    assert route_cost > 0
    assert label_spend(job) == pytest.approx(.01 + route_cost)
    assert label_spend_complete(job) is complete


@pytest.mark.parametrize('final_reply', ['truncated', 'unparseable', 'key_exhausted'])
def test_final_retry_reuses_paid_route_receipt(tmp_path, monkeypatch, final_reply):
    ep = observed_packed_episode(tmp_path)
    job = tmp_path / 'job'
    out = job / 'run' / 'out'
    calls = []
    finals = []
    def provider(content, model, *a, **kw):
        calls.append((model, kw['attempts']))
        if model == route.ROUTE_MODEL:
            return {'usage': {'cost': .004}, 'choices': [{'message': {
                'content': '{"fine_detail": false, "why": "whole objects"}'}}]}
        finals.append(1)
        if len(finals) == 1:
            if final_reply == 'key_exhausted':
                raise harness.KeyExhausted('fixture key empty', http_status=402, first_attempt=True)
            return {'usage': {'cost': .01}, 'choices': [{
                'finish_reason': 'length' if final_reply == 'truncated' else 'stop',
                'message': {'content': 'unparseable'}}]}
        return {'usage': {'cost': .01}, 'choices': [{'finish_reason': 'stop',
                'message': {'content': '{"timeline": [], "task_summary": "Wait beside the visible workspace."}'}}]}
    monkeypatch.setattr(route, '_CACHE', {})
    monkeypatch.setattr(harness, 'call_model', provider)
    batch = dict(keys=['sk-or-fixture'], concurrency=1, force=False, max_spend=1, **args())
    if final_reply == 'truncated':
        assert harness.run_batch([ep], out, **batch) == 1
    elif final_reply == 'key_exhausted':
        with pytest.raises(harness.KeyExhausted):
            harness.label_episode(ep, out / f'{ep.name}.json', api_key='sk-or-fixture', **args())
    else:
        assert harness.label_episode(ep, out / f'{ep.name}.json', api_key='sk-or-fixture', **args())['parse_ok'] is False
    monkeypatch.setattr(route, '_CACHE', {})
    if final_reply != 'key_exhausted':
        assert harness.run_batch([ep], out, **batch) == 1
        assert len(finals) == 1
        batch['force'] = True
    assert harness.run_batch([ep], out, **batch) == 0
    assert len(finals) == 2
    assert [model for model, _ in calls].count(route.ROUTE_MODEL) == 1
    assert all(attempts == 1 for _, attempts in calls)
    assert label_spend(job) == pytest.approx(.014 if final_reply == 'key_exhausted' else .024)
    assert label_spend_complete(job) is True


@pytest.mark.parametrize('refresh', ['explicit', 'max_tokens', 'task'])
def test_forced_refresh_preserves_prior_bills_and_remaining_cap(tmp_path, monkeypatch, refresh):
    ep = observed_packed_episode(tmp_path)
    job = tmp_path / 'job'
    out = job / 'run' / 'out'
    calls = []
    reservations = []
    final_cost_bound = harness.final_cost_bound
    def observe_bound(content, max_tokens, **kwargs):
        bound = final_cost_bound(content, max_tokens, **kwargs)
        reservations.append((max_tokens, bound))
        return bound
    def provider(content, model, *a, **kw):
        calls.append(model)
        return {'id': f'call-{len(calls)}', 'usage': {'cost': .004 if model == route.ROUTE_MODEL else .01},
                'choices': [{'finish_reason': 'stop', 'message': {'content':
                    '{"fine_detail": false}' if model == route.ROUTE_MODEL else '{"timeline": [], "task_summary": "Wait beside the visible workspace."}'}}]}
    monkeypatch.setattr(route, '_CACHE', {})
    monkeypatch.setattr(harness, 'call_model', provider)
    monkeypatch.setattr(harness, 'final_cost_bound', observe_bound)
    batch = dict(keys=['sk-or-fixture'], concurrency=1, force=False, max_spend=1, **args())
    assert harness.run_batch([ep], out, **batch) == 0
    assert label_spend(job) == pytest.approx(.014)
    if refresh == 'task':
        context = json.loads((ep / 'context.json').read_text())
        context['instruction'] = 'Move the spoon.'
        (ep / 'context.json').write_text(json.dumps(context))
    options = dict(batch, force=True, max_spend=.00001)
    options.update({'cell_w': 128} if refresh == 'explicit' else {'max_tokens': 120} if refresh == 'max_tokens' else {})
    monkeypatch.setattr(route, '_CACHE', {})
    assert harness.run_batch([ep], out, **options) == 0
    assert len(calls) == 2
    assert label_spend(job) == pytest.approx(.014)
    retry_cap = 1
    if refresh == 'max_tokens':
        tokens, bound = reservations[-1]
        assert tokens == 120 and bound > 0
        assert harness.run_batch([ep], out, **dict(options, max_spend=bound - .000001)) == 0
        assert len(calls) == 2 and label_spend(job) == pytest.approx(.014)
        refused = json.loads((out / f'noreply_{ep.name}.json').read_text())
        assert refused['final_dispatch_outcome'] == 'not dispatched'
        assert refused['final_reserved_usd'] == 0
        assert refused['final_cost_history'][0]['generation_id'] == 'call-2'
        assert label_spend_complete(job) is True
        retry_cap = bound + .000001
    assert harness.run_batch([ep], out, **dict(options, max_spend=retry_cap)) == 0
    assert calls.count(route.ROUTE_MODEL) == (2 if refresh == 'task' else 1)
    assert calls.count(harness.DEFAULT_MODEL) == 2
    assert label_spend(job) == pytest.approx(.028 if refresh == 'task' else .024)
    result = json.loads((out / f'{ep.name}.json').read_text())
    assert result['final_cost_history'][0]['generation_id'] == 'call-2'
    assert label_spend_complete(job) is True


def test_forced_legacy_unknown_final_retains_reservation_before_routing(tmp_path, monkeypatch):
    ep = observed_packed_episode(tmp_path)
    job = tmp_path / 'job'
    out = job / 'run' / 'out'
    out.mkdir(parents=True)
    old = {'final_dispatch_outcome': 'unverified', 'final_reserved_usd': 3.2,
           'input_identity': harness.input_identity(ep, cell_w=0, example_dir=None,
               **{k: v for k, v in args().items() if k != 'timeout'}),
           'config': {'resolution_route': {'cost_usd': .004, 'model': route.ROUTE_MODEL}}}
    (out / f'noreply_{ep.name}.json').write_text(json.dumps(old))
    calls = []
    def provider(content, model, *a, **kw):
        calls.append(model)
        return {'usage': {'cost': .004 if model == route.ROUTE_MODEL else .01},
                'choices': [{'finish_reason': 'stop', 'message': {'content':
                    '{"fine_detail": false}' if model == route.ROUTE_MODEL else '{"timeline": [], "task_summary": "Wait beside the visible workspace."}'}}]}
    monkeypatch.setattr(route, '_CACHE', {})
    monkeypatch.setattr(harness, 'call_model', provider)
    batch = dict(keys=['sk-or-fixture'], concurrency=1, force=True, max_spend=.00001, **args())
    assert harness.run_batch([ep], out, **batch) == 0
    assert not calls
    assert label_spend(job) == pytest.approx(3.204)
    assert label_spend_complete(job) is False
    assert harness.run_batch([ep], out, **dict(batch, max_spend=1)) == 0
    assert calls == [route.ROUTE_MODEL, harness.DEFAULT_MODEL]
    assert label_spend(job) == pytest.approx(3.218)
    assert label_spend_complete(job) is False


@pytest.mark.parametrize('cache_cost', [.02, .025])
@pytest.mark.parametrize('inspection_error', [False, True])
def test_inspection_cache_is_carried_through_new_route_claim(tmp_path, monkeypatch, cache_cost, inspection_error):
    from test_label import _packed_episode
    from label import evidence_access as ea
    ep, _ = _packed_episode(tmp_path)
    job = tmp_path / 'job'
    out = job / 'run' / 'out'
    out.mkdir(parents=True)
    cache = out / '.evidence' / f'{ep.name}.json.inspection.json'
    cache.parent.mkdir()
    cache.write_text(json.dumps({'digest': 'same-cache', 'cost_usd': cache_cost, 'rounds': [], 'limitations': []}))
    old = {'final_dispatch_outcome': 'not dispatched', 'final_reserved_usd': 0,
           'config': {'resolution_route': {'cost_usd': .004}},
           'evidence_inspection': {'digest': 'same-cache', 'cost_usd': .02}}
    (out / f'noreply_{ep.name}.json').write_text(json.dumps(old))
    at_dispatch = []
    def provider(content, model, *a, **kw):
        if model == route.ROUTE_MODEL:
            claimed = json.loads((out / f'noreply_{ep.name}.json').read_text())
            at_dispatch.append((label_spend(job), claimed['config']['resolution_route']['cost_usd']))
            return {'usage': {'cost': .004}, 'choices': [{'message': {'content': '{}'}}]}
        raise harness.KeyExhausted('fixture key empty', http_status=402, first_attempt=True)
    monkeypatch.setattr(route, '_CACHE', {})
    monkeypatch.setattr(harness, 'call_model', provider)
    build_request = harness.me.build_request
    def build_with_access(*a, **kw):
        result = build_request(*a, **kw)
        result['evidence_access'] = ea.Access(harness.me.load(ep))
        return result
    monkeypatch.setattr(harness.me, 'build_request', build_with_access)
    def discover(*a, **kw):
        if inspection_error:
            raise RuntimeError('fixture inspection error')
        return {'digest': 'same-cache', 'cost_usd': cache_cost, 'rounds': [], 'limitations': []}, []
    monkeypatch.setattr(ea, 'discover', discover)
    with pytest.raises(RuntimeError if inspection_error else harness.KeyExhausted):
        harness.label_episode(ep, out / f'{ep.name}.json', api_key='sk-or-fixture', force_refresh=True, **args())
    result = json.loads((out / f'noreply_{ep.name}.json').read_text())
    assert at_dispatch[0][0] == pytest.approx(.004 + cache_cost + at_dispatch[0][1])
    assert harness.episode_cost(result) == pytest.approx(.008 + cache_cost)
    assert label_spend(job) == pytest.approx(.008 + cache_cost)
    assert result['evidence_inspection']['previously_billed_usd'] == pytest.approx(cache_cost)


@pytest.mark.parametrize('history', [
    {'final_cost_history': [{'usage': {'est_cost_usd': None}}]},
    {'final_cost_history': [{'final_dispatch_outcome': 'unverified', 'final_reserved_usd': None}]},
    {'auxiliary_cost_history': [{'kind': 'route', 'cost_usd': None}]},
    {'auxiliary_cost_history': [{'kind': 'inspection', 'cost_usd': True}]},
])
def test_unknown_historical_invoices_retain_conservative_service_reservation(tmp_path, history):
    job = tmp_path / 'job'
    out = job / 'run' / 'out'
    out.mkdir(parents=True)
    (out / 'episode_a.json').write_text(json.dumps({'parse_ok': True, 'usage': {'est_cost_usd': .01}, **history}))
    assert label_spend(job) == 1e300
    assert label_spend_complete(job) is False


def test_cache_only_inspection_does_not_consume_remaining_cap_again(tmp_path, monkeypatch):
    from test_label import _packed_episode
    from label import evidence_access as ea
    ep, _ = _packed_episode(tmp_path)
    job = tmp_path / 'job'
    out = job / 'run' / 'out'
    cache = out / '.evidence' / f'{ep.name}.json.inspection.json'
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({'digest': 'same-cache', 'cost_usd': .02, 'rounds': [], 'limitations': []}))
    calls = []
    def provider(*a, **kw):
        calls.append(1)
        return {'usage': {'cost': .01}, 'choices': [{'finish_reason': 'stop',
                'message': {'content': '{"timeline": [], "task_summary": "Wait beside the visible workspace."}'}}]}
    build_request = harness.me.build_request
    def build_with_access(*a, **kw):
        result = build_request(*a, **kw)
        result['evidence_access'] = ea.Access(harness.me.load(ep))
        return result
    monkeypatch.setattr(harness.me, 'build_request', build_with_access)
    monkeypatch.setattr(harness, 'call_model', provider)
    monkeypatch.setattr(ea, 'discover', lambda *a, **kw: ({'digest': 'same-cache', 'cost_usd': .02,
        'rounds': [], 'limitations': []}, []))
    req = build_with_access(ep, cell_w=128)
    addition = [{'type': 'text', 'text': ea.FINAL + '\nINSPECTION COVERAGE\n' +
                 ea.packed(ea.Access.coverage_summary({'limitations': []}))}]
    cap = harness.final_cost_bound(req['content'] + addition, 100) + .000001
    assert label_spend(job) == pytest.approx(.02)
    assert harness.run_batch([ep], out, keys=['sk-or-fixture'], concurrency=1,
        force=False, max_spend=cap, **args(cell_w=128)) == 0
    assert calls == [1]
    assert label_spend(job) == pytest.approx(.03)


@pytest.mark.parametrize('unknown', [False, True])
def test_dry_run_preserves_prior_paid_invoices(tmp_path, monkeypatch, unknown):
    ep = observed_packed_episode(tmp_path)
    job = tmp_path / 'job'
    out = job / 'run' / 'out'
    calls = []
    def provider(content, model, *a, **kw):
        calls.append(model)
        if model == route.ROUTE_MODEL:
            return {'usage': {'cost': .004}, 'choices': [{'message': {'content': '{}'}}]}
        if unknown:
            raise TimeoutError('fixture final outcome unknown')
        return {'usage': {'cost': .01}, 'choices': [{'finish_reason': 'stop',
                'message': {'content': '{"timeline": [], "task_summary": "Wait beside the visible workspace."}'}}]}
    monkeypatch.setattr(route, '_CACHE', {})
    monkeypatch.setattr(harness, 'call_model', provider)
    batch = dict(keys=['sk-or-fixture'], concurrency=1, force=False, max_spend=1, **args())
    assert harness.run_batch([ep], out, **batch) == (1 if unknown else 0)
    before = label_spend(job)
    assert before > 0
    assert harness.run_batch([ep], out, **dict(batch, force=unknown, dry_run=True)) == 0
    assert calls == [route.ROUTE_MODEL, harness.DEFAULT_MODEL]
    assert label_spend(job) == pytest.approx(before)
    assert label_spend_complete(job) is (not unknown)


def test_new_route_claim_is_latest_after_dry_output(tmp_path, monkeypatch):
    ep = observed_packed_episode(tmp_path)
    job = tmp_path / 'job'
    out = job / 'run' / 'out' / f'{ep.name}.json'
    def provider(content, model, *a, **kw):
        return {'usage': {'cost': .004 if model == route.ROUTE_MODEL else .01},
                'choices': [{'finish_reason': 'stop', 'message': {'content': '{}'}}]}
    monkeypatch.setattr(route, '_CACHE', {})
    monkeypatch.setattr(harness, 'call_model', provider)
    harness.label_episode(ep, out, api_key='sk-or-fixture', **args())
    harness.label_episode(ep, out, api_key='sk-or-fixture', **args(dry_run=True))
    context = json.loads((ep / 'context.json').read_text())
    context['instruction'] = 'Move the spoon.'
    (ep / 'context.json').write_text(json.dumps(context))
    class Crash(BaseException):
        pass
    def crash(*a, **kw):
        raise Crash()
    monkeypatch.setattr(route, '_CACHE', {})
    monkeypatch.setattr(harness, 'call_model', crash)
    with pytest.raises(Crash):
        harness.label_episode(ep, out, api_key='sk-or-fixture', force_refresh=True, **args())
    claim = json.loads(out.with_name('noreply_' + out.name).read_text())
    assert label_spend(job) == pytest.approx(.014 + claim['config']['resolution_route']['cost_usd'])
    assert label_spend_complete(job) is False


def test_request_build_failure_without_routing_remains_resumable(tmp_path, monkeypatch):
    ep = observed_packed_episode(tmp_path)
    job = tmp_path / 'job'
    out = job / 'run' / 'out'
    original = harness.me.build_request
    monkeypatch.setattr(harness.me, 'build_request', build_failure)
    monkeypatch.setattr(harness, 'call_model', lambda *a, **kw: pytest.fail('build failure cannot dispatch'))
    batch = dict(keys=['sk-or-fixture'], concurrency=1, force=False, max_spend=1, **args(cell_w=128))
    assert harness.run_batch([ep], out, **batch) == 1
    failure = json.loads((out / f'noreply_{ep.name}.json').read_text())
    assert failure['final_dispatch_outcome'] == 'not dispatched'
    assert failure['input_identity'] and label_spend(job) == 0
    monkeypatch.setattr(harness.me, 'build_request', original)
    calls = []
    def provider(*a, **kw):
        calls.append(1)
        return {'usage': {'cost': .01}, 'choices': [{'finish_reason': 'stop',
                'message': {'content': '{"timeline": [], "task_summary": "Wait beside the visible workspace."}'}}]}
    monkeypatch.setattr(harness, 'call_model', provider)
    assert harness.run_batch([ep], out, **batch) == 0
    assert calls == [1] and label_spend(job) == pytest.approx(.01)
