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
            assert entered.wait(2) and refused.wait(2)
        finally:
            release.set()
        assert future.result(timeout=2) == 1
    assert calls == [1]
    assert label_spend(job) == pytest.approx(.004)


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
    from test_label import _packed_episode
    ep, _ = _packed_episode(tmp_path)
    job = tmp_path / 'job'
    out = job / 'run' / 'out' / f'{ep.name}.json'
    calls = []
    def provider(content, model, *a, **kw):
        calls.append((model, kw['attempts']))
        if model == route.ROUTE_MODEL:
            return {'usage': route_usage, 'choices': [{'message': {
                'content': '{"fine_detail": false, "why": "whole objects"}'}}]}
        return {'usage': {'cost': .01}, 'choices': [{'finish_reason': 'stop',
                'message': {'content': '{"timeline": []}'}}]}
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
    from test_label import _packed_episode
    ep, _ = _packed_episode(tmp_path)
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
                raise harness.KeyExhausted('fixture key empty')
            return {'usage': {'cost': .01}, 'choices': [{
                'finish_reason': 'length' if final_reply == 'truncated' else 'stop',
                'message': {'content': 'unparseable'}}]}
        return {'usage': {'cost': .01}, 'choices': [{'finish_reason': 'stop',
                'message': {'content': '{"timeline": []}'}}]}
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
    assert harness.run_batch([ep], out, **batch) == 0
    assert len(finals) == 2
    assert [model for model, _ in calls].count(route.ROUTE_MODEL) == 1
    assert all(attempts == 1 for _, attempts in calls)
    assert label_spend(job) == pytest.approx(.014 if final_reply == 'key_exhausted' else .024)
    assert label_spend_complete(job) is True


@pytest.mark.parametrize('refresh', ['explicit', 'max_tokens', 'task'])
def test_forced_refresh_preserves_prior_bills_and_remaining_cap(tmp_path, monkeypatch, refresh):
    from test_label import _packed_episode
    ep, _ = _packed_episode(tmp_path)
    job = tmp_path / 'job'
    out = job / 'run' / 'out'
    calls = []
    def provider(content, model, *a, **kw):
        calls.append(model)
        return {'id': f'call-{len(calls)}', 'usage': {'cost': .004 if model == route.ROUTE_MODEL else .01},
                'choices': [{'finish_reason': 'stop', 'message': {'content':
                    '{"fine_detail": false}' if model == route.ROUTE_MODEL else '{"timeline": []}'}}]}
    monkeypatch.setattr(route, '_CACHE', {})
    monkeypatch.setattr(harness, 'call_model', provider)
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
    assert harness.run_batch([ep], out, **dict(options, max_spend=.75 if refresh == 'max_tokens' else 1)) == 0
    assert calls.count(route.ROUTE_MODEL) == (2 if refresh == 'task' else 1)
    assert calls.count(harness.DEFAULT_MODEL) == 2
    assert label_spend(job) == pytest.approx(.028 if refresh == 'task' else .024)
    result = json.loads((out / f'{ep.name}.json').read_text())
    assert result['final_cost_history'][0]['generation_id'] == 'call-2'
    assert label_spend_complete(job) is True


def test_forced_legacy_unknown_final_retains_reservation_before_routing(tmp_path, monkeypatch):
    from test_label import _packed_episode
    ep, _ = _packed_episode(tmp_path)
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
                    '{"fine_detail": false}' if model == route.ROUTE_MODEL else '{"timeline": []}'}}]}
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
        raise harness.KeyExhausted('fixture key empty')
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
                'message': {'content': '{"timeline": []}'}}]}
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
    from test_label import _packed_episode
    ep, _ = _packed_episode(tmp_path)
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
                'message': {'content': '{"timeline": []}'}}]}
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
    from test_label import _packed_episode
    ep, _ = _packed_episode(tmp_path)
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
    from test_label import _packed_episode
    ep, _ = _packed_episode(tmp_path)
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
                'message': {'content': '{"timeline": []}'}}]}
    monkeypatch.setattr(harness, 'call_model', provider)
    assert harness.run_batch([ep], out, **batch) == 0
    assert calls == [1] and label_spend(job) == pytest.approx(.01)
