"""Offline regressions for reported label validity, evidence, and accounting faults."""
import io
import json
import urllib.error

import numpy as np
import pytest

from label import evidence_access as ea, harness, pieces, reparse, run, sensor_evidence


@pytest.mark.parametrize('reply', [{}, {'timeline': [], 'completion': {'task_completed': 'banana'}}])
def test_invalid_annotation_is_not_complete(reply):
    labels, ok = harness.parse_response(json.dumps(reply))
    assert not ok
    assert labels['_raw'] == json.dumps(reply)


@pytest.mark.parametrize('columns,row', [
    (harness.TIMELINE_COLUMNS, [0, 1, 'left', 'reach', 'cup', None, 'advancing', .2, None]),
    (['start_s', 'start_s', 'progress'], [0, 10, 15]),
    ('start_s', [0, 1, 2, 3, 4, 5, 6]),
])
def test_ambiguous_positional_timeline_is_withheld(columns, row):
    labels = harness.normalize_timeline({'timeline_columns': columns, 'timeline': [row]})
    assert labels['timeline'] == []
    assert labels['_dropped']


def test_valid_schema_variants_and_partial_replies_remain_usable():
    replies = [
        {'timeline': [{'start_s': 0, 'end_s': 1, 'action': 'reach'}]},
        {'tasks': [{'task': 'prepare', 'outcome': 'unclear'}], 'completion': {'task_completed': None}},
        {'task_summary': 'An empty workspace is visible.', 'timeline': []},
    ]
    for reply in replies:
        assert harness.parse_response(json.dumps(reply)) == (reply, True)


@pytest.mark.parametrize('progress', [15, -1, True, float('inf')])
def test_invalid_progress_is_not_accepted(progress):
    _, ok = harness.parse_response(json.dumps({'timeline': [{'start_s': 0, 'end_s': 1, 'progress': progress}]}))
    assert not ok


def test_openai_quota_rejection_retires_key_with_unpaid_proof(monkeypatch):
    def rejected(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 429, 'quota', {},
                                     io.BytesIO(b'{"error":{"code":"insufficient_quota"}}'))
    monkeypatch.setattr(harness.urllib.request, 'urlopen', rejected)
    with pytest.raises(harness.KeyExhausted) as caught:
        harness.call_model_once([], 'openai/gpt-6-astra', 'medium', 'sk-proj-test', 10, 1)
    assert caught.value.verified_unpaid


@pytest.mark.parametrize('status', [429, 503])
def test_ambiguous_http_failure_does_not_prove_unpaid(monkeypatch, status):
    def rejected(req, timeout):
        raise urllib.error.HTTPError(req.full_url, status, 'unknown', {}, io.BytesIO(b'{}'))
    monkeypatch.setattr(harness.urllib.request, 'urlopen', rejected)
    with pytest.raises(RuntimeError) as caught:
        harness.call_model_once([], 'openai/gpt-6-astra', 'medium', 'sk-or-test', 10, 1)
    assert not isinstance(caught.value, harness.KeyExhausted)


def test_batch_parse_failure_counts_as_failure_and_keeps_paid_reply(tmp_path, monkeypatch, capsys):
    ep = tmp_path / 'episode_a'
    ep.mkdir()
    def label(ep, target, **kw):
        record = {'parse_ok': False, 'labels': {'_raw': 'invalid'}, 'usage': {'est_cost_usd': .01}}
        target.write_text(json.dumps(record))
        return record
    monkeypatch.setattr(harness, 'label_episode', label)
    out = tmp_path / 'out'
    assert harness.run_batch([ep], out, keys=['sk-or-test'], concurrency=1, force=False) == 1
    assert 'done=0 failed=1' in capsys.readouterr().out
    assert json.loads((out / 'episode_a.json').read_text())['usage']['est_cost_usd'] == .01
    assert not (out / 'noreply_episode_a.json').exists()


def test_footage_hours_prefers_declared_span_and_validates_fallback(tmp_path):
    for name, context in [('declared', {'duration_s': 10, 'n_state_frames': 10, 'fps': 30}),
                           ('fallback', {'duration_s': None, 'n_state_frames': 60, 'fps': 30})]:
        ep = tmp_path / name
        ep.mkdir()
        (ep / 'context.json').write_text(json.dumps(context))
    assert run.footage_hours(tmp_path, ['declared', 'fallback']) == pytest.approx(12 / 3600)


def sensor_doc():
    return {'sensors': [{'id': 'sensor', 'name': 'sensor', 'kind': 'numeric',
                         'capabilities': ['response_change']}],
            'series': [{'id': 'series', 'sensor_id': 'sensor', 'times': [0., 80., 90., 100.],
                        'values': [1, 2, 3, 4], 'columns': [0]}], 'limitations': []}


def finding(times):
    return {'start_s': 80., 'end_s': 90., 'headline': 'Response rises', 'claim_type': 'response_change',
            'evidence': [{'sensor_id': 'sensor', 'series_id': 'series', 'time_s': times}]}


def test_unrelated_sensor_citation_cannot_back_late_event():
    bound = sensor_evidence.bind(sensor_doc(), [finding([0.])])
    assert not bound['findings'] and bound['unbound_findings']


@pytest.mark.parametrize('times', [[0., 80., 90.], [0., 100.], [80., 90.], [80.]])
def test_sensor_context_comparison_can_bracket_event(times):
    assert sensor_evidence.bind(sensor_doc(), [finding(times)])['findings']


def test_force_refresh_inspection_replaces_stale_cache_and_keeps_receipt(tmp_path):
    from test_evidence_access import access
    inspected = access(tmp_path)
    cache = tmp_path / 'inspection.json'
    response = {'usage': {'cost': .01}, 'choices': [{'message': {'content': '{"requests":[]}'}}]}
    before, _ = ea.discover(inspected, [], lambda _: response, cache, reserve_cost=.2)
    calls = []
    def call(_):
        calls.append(1)
        return response
    after, _ = ea.discover(inspected, [{'type': 'text', 'text': 'changed'}], call, cache,
                           reserve_cost=.2, force_refresh=True)
    assert calls == [1] and after['status'] == 'complete'
    assert after['digest'] != before['digest']
    assert any(p.read_bytes() for p in cache.parent.glob(cache.name + '.previous.*'))


def test_selection_reserve_uses_model_rates():
    content = [{'type': 'text', 'text': 'x' * 3000}]
    astra = harness.final_cost_bound(content, 2000, model='openai/gpt-6-astra')
    sol = harness.final_cost_bound(content, 2000, model='openai/gpt-6-sol')
    assert astra > sol
    with pytest.raises(ValueError, match='price|pricing|rate'):
        harness.final_cost_bound(content, 2000, model='unpriced/model')


def test_depth_presentation_origin_does_not_depend_on_rgb_times(tmp_path, monkeypatch):
    from test_camera_timing import recording
    ep = recording(tmp_path, np.arange(60) / 30)
    context = json.loads((ep / 'context.json').read_text())
    context.pop('real_times', None)
    context.update(depth_presentation_times='depth_presentation.npz', clock_zero_s=0.)
    (ep / 'context.json').write_text(json.dumps(context))
    (ep / 'depth.json').write_text(json.dumps({'exo': {'kmap': 'depth_kmap.npy'}}))
    np.save(ep / 'depth_kmap.npy', np.arange(60))
    np.savez(ep / 'depth_presentation.npz', exo=np.arange(60) / 30)
    monkeypatch.setattr(pieces, 'piece_max', lambda ctx: 1.)
    for part in pieces.write_pieces(ep, tmp_path / 'parts'):
        pc = json.loads((part / 'context.json').read_text())
        values = np.load(part / 'depth_presentation.npz')['exo']
        assert values[0] == pytest.approx(-pc['piece']['t0_s'])


def test_reparse_restores_native_inspected_and_camera_bindings(tmp_path, monkeypatch):
    from test_camera_timing import recording
    ep = recording(tmp_path, np.arange(12) / 30, np.arange(12) / 30)
    access = ea.Access(harness.me.load(ep))
    field = next(f for f in access.inventory() if f['name'] == 'force')
    receipt = access.inspect({'field_id': field['id'], 'mode': 'samples'})
    inspected = {'headline': 'Additional recorded response', 'start_s': 0., 'end_s': .3,
                 'adds_beyond_video': True,
                 'evidence': [{'evidence_id': receipt['id'], 'time_s': [0., .3]}]}
    cameras = [{'view': v, 'content': 'scene', 'evidence': 'The work surface is visible.'}
               for v in ('exo', 'left')]
    labels = {'timeline': [], 'sensor_findings': [finding([80., 90.])],
              'evidence_findings': [inspected], 'camera_observations': cameras}
    identity = harness.input_identity(ep, model=harness.DEFAULT_MODEL, reasoning='medium', max_tokens=100,
                                      cell_w=128, example_dir=None)
    record = {'episode_dir': str(ep), 'parse_ok': False, 'labels': {'_raw': json.dumps(labels)},
              'input_identity': identity, 'sensor_evidence': sensor_doc(), 'evidence_inspection': access.record(),
              'config': {'views': ['exo', 'left'], 'camera_observation_views': ['exo', 'left'],
                         'schema_fields': ['camera_observations']}}
    folder = tmp_path / 'run'
    (folder / 'out').mkdir(parents=True)
    target = folder / 'out' / 'episode_a.json'
    target.write_text(json.dumps(record))
    (folder / 'run.json').write_text('{}')
    monkeypatch.setattr(reparse, 'commit', lambda: ('offline', False))
    monkeypatch.setattr(harness, 'call_model', lambda *a, **kw: pytest.fail('reparse dispatched a paid call'))
    assert reparse.reparse(folder)['parsed_now'] == 1
    rebound = json.loads(target.read_text())
    assert len(rebound['sensor_evidence']['findings']) == 2
    assert rebound['camera_observations']['observations'] == cameras
    assert json.loads((folder / 'out' / 'reparsed_originals' / target.name).read_text()) == record


def test_unknown_model_without_cost_cannot_invent_astra_charge():
    assert harness._verified_final_cost({'prompt_tokens': 100, 'completion_tokens': 50}, 'custom/model') == (
        None, 'unverified')


def test_batch_retries_verified_quota_rejection_on_next_key(tmp_path, monkeypatch):
    from test_camera_timing import recording
    ep = recording(tmp_path, np.arange(12) / 30)
    calls = []
    def provider(request, timeout):
        key = request.get_header('Authorization')
        calls.append(key)
        if key.endswith('bad'):
            raise urllib.error.HTTPError(request.full_url, 429, 'quota', {},
                io.BytesIO(b'{"error":{"code":"insufficient_quota"}}'))
        payload = json.loads(request.data)
        text = json.dumps(payload['messages'])
        response = {'usage': {'cost': .01}, 'choices': [{'finish_reason': 'stop', 'message': {
            'content': '{"requests":[]}' if 'FIELD INVENTORY' in text else '{"timeline":[],"task_summary":"Wait beside the visible workspace."}'}}]}
        return io.BytesIO(json.dumps(response).encode())
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    out = tmp_path / 'out'
    assert harness.run_batch([ep], out, keys=['sk-proj-bad', 'sk-proj-good'], concurrency=1, force=False,
        model=harness.DEFAULT_MODEL, reasoning='medium', max_tokens=100, timeout=1, cell_w=128, max_spend=1) == 0
    result = json.loads((out / f'{ep.name}.json').read_text())
    assert result['parse_ok'] and harness.episode_cost(result) == pytest.approx(.02)
    assert calls == ['Bearer sk-proj-bad', 'Bearer sk-proj-good', 'Bearer sk-proj-good']
    inspection = result['evidence_inspection']
    assert inspection['rounds'][0]['rejected_attempts'][0]['rejection_code'] == 'insufficient_quota'


def test_force_refresh_retains_all_prior_inspection_and_final_spend(tmp_path, monkeypatch):
    from test_camera_timing import recording
    ep = recording(tmp_path, np.arange(12) / 30)
    calls = []
    def provider(content, *a, **kw):
        calls.append(1)
        selection = 'FIELD INVENTORY' in json.dumps(content)
        return {'usage': {'cost': .01}, 'choices': [{'finish_reason': 'stop', 'message': {
            'content': '{"requests":[]}' if selection else '{"timeline":[],"task_summary":"Wait beside the visible workspace."}'}}]}
    monkeypatch.setattr(harness, 'call_model', provider)
    out = tmp_path / 'out'
    args = dict(keys=['sk-or-test'], concurrency=1, model=harness.DEFAULT_MODEL, reasoning='medium',
                max_tokens=100, timeout=1, cell_w=128, max_spend=1)
    for index in range(3):
        assert harness.run_batch([ep], out, force=bool(index), **args) == 0
        assert run.billed_cost(tmp_path) == pytest.approx((index + 1) * .02)
    assert len(calls) == 6


def test_reparse_withholds_projections_after_native_source_changes(tmp_path):
    from test_camera_timing import recording
    ep = recording(tmp_path, np.arange(12) / 30, np.arange(12) / 30)
    identity = harness.input_identity(ep, model=harness.DEFAULT_MODEL, reasoning='medium', max_tokens=100,
                                      cell_w=128, example_dir=None)
    evidence = sensor_doc()
    evidence['source_proof'] = identity['source_proof']
    rows = [{'view': v, 'content': 'scene', 'evidence': 'The work surface is visible.'} for v in ('exo', 'left')]
    record = {'episode_dir': str(ep), 'input_identity': identity, 'sensor_evidence': evidence,
              'labels': {'sensor_findings': [finding([80.])], 'camera_observations': rows},
              'config': {'views': ['exo', 'left'], 'camera_observation_views': ['exo', 'left'],
                         'schema_fields': ['camera_observations']}}
    movie = tmp_path / 'exo.mp4'
    movie.write_bytes(movie.read_bytes() + b'changed source')
    reparse.rebind(record)
    assert not record['camera_observations']['observations']
    assert not record['sensor_evidence']['findings']
    assert record['sensor_evidence']['withheld_findings']


def test_token_only_cost_uses_selected_model_rate():
    usage = {'prompt_tokens': 100, 'completion_tokens': 50}
    expected = harness.openai_list_cost('openai/gpt-6-sol', usage)
    assert harness._verified_final_cost(usage, 'openai/gpt-6-sol') == (expected, 'estimate')


@pytest.mark.parametrize('times,accepted', [([0.], False), ([0., .3], True), ([.2], True)])
def test_inspected_citations_need_event_samples_or_bracketing_context(tmp_path, times, accepted):
    from test_evidence_access import access
    inspected = access(tmp_path)
    inspected.ep['signals']['pressure'] = np.arange(4).reshape(4, 1)
    inspected.ep['signal_meta'] = {'pressure': {}}
    inspected.refresh()
    field = next(f for f in inspected.inventory() if f['name'] == 'pressure')
    receipt = inspected.inspect({'field_id': field['id'], 'mode': 'samples'})
    claim = {'start_s': .2, 'end_s': .3, 'headline': 'Late response change', 'adds_beyond_video': True,
             'evidence': [{'evidence_id': receipt['id'], 'time_s': times}]}
    result = inspected.bind([claim])
    assert bool(result['findings']) is accepted


@pytest.mark.parametrize('reply', [
    {'timeline': []}, {'timeline': [{}]}, {'timeline': [{'start_s': 0, 'end_s': 1}]},
    {'tasks': [{}]}, {'completion': {'task_completed': None}},
    {'scene': {'objects': [{}]}}, {'data_issues': [{}]}, {'task_summary': '   '},
])
def test_placeholder_annotations_are_not_substantive(reply):
    labels, accepted = harness.parse_response(json.dumps(reply))
    assert not accepted
    assert labels['_raw'] == json.dumps(reply)


@pytest.mark.parametrize('reply', [
    {'timeline': [{'start_s': 0, 'end_s': 1, 'action': 'Wait beside the cup'}]},
    {'tasks': [{'task': 'Prepare a drink', 'outcome': 'partial'}]},
    {'completion': {'task_completed': 'unclear', 'reason': 'The camera remains covered.'}},
    {'timeline': [], 'tasks': [{'task': 'Place the cup', 'outcome': 'success'}],
     'completion': {'task_completed': None}},
    {'timeline': [], 'data_issues': [{'issue': 'The camera is covered.', 'severity': 'high'}]},
    {'operator_mistakes': [{'issue': 'The cup is dropped.', 'severity': 'medium'}]},
    {'performance_review': 'The hands remain idle throughout the visible recording.'},
    {'goal_alignment': {'matches_given': False, 'relation': 'unrelated', 'note': 'Only a blank screen is visible.'}},
])
def test_substantive_partial_schema_variants_remain_accepted(reply):
    assert harness.parse_response(json.dumps(reply)) == (reply, True)


@pytest.mark.parametrize('labels', [{'timeline': []}, {'timeline': [{}]}, {}])
def test_resume_revalidates_saved_placeholder_and_retains_previous_charge(tmp_path, monkeypatch, labels):
    from test_label import _packed_episode
    ep, _ = _packed_episode(tmp_path)
    out = tmp_path / 'out'
    out.mkdir()
    target = out / (ep.name + '.json')
    settings = dict(model=harness.DEFAULT_MODEL, reasoning='medium', max_tokens=100, cell_w=128, example_dir=None)
    identity = harness.input_identity(ep, **settings)
    target.write_text(json.dumps({'parse_ok': True, 'labels': labels, 'input_identity': identity,
                                 'usage': {'est_cost_usd': .01}, 'final_attempt_index': 1}))
    calls = []
    def label_episode(path, output, **kwargs):
        calls.append(1)
        record = {'parse_ok': True, 'labels': {'task_summary': 'The visible hands wait.'},
                  'input_identity': identity, 'usage': {'est_cost_usd': .02}, **harness._billing_fields(output)}
        output.write_text(json.dumps(record))
        return record
    monkeypatch.setattr(harness, 'label_episode', label_episode)
    assert harness.run_batch([ep], out, keys=['sk-or-test'], concurrency=1, force=False, **settings) == 1
    assert calls == []
    assert harness.run_batch([ep], out, keys=['sk-or-test'], concurrency=1, force=True, **settings) == 0
    assert calls == [1]
    result = json.loads(target.read_text())
    assert harness.episode_cost(result) == pytest.approx(.03)
    assert result['final_cost_history'][0]['usage']['est_cost_usd'] == .01


def test_resume_invalid_annotation_with_unknown_charge_fails_without_replaying(tmp_path, monkeypatch):
    from test_label import _packed_episode
    ep, _ = _packed_episode(tmp_path)
    out = tmp_path / 'out'
    out.mkdir()
    settings = dict(model=harness.DEFAULT_MODEL, reasoning='medium', max_tokens=100, cell_w=128, example_dir=None)
    identity = harness.input_identity(ep, **settings)
    record = {'parse_ok': True, 'labels': {'timeline': []}, 'input_identity': identity,
              'final_dispatch_outcome': 'unverified', 'final_reserved_usd': 1.}
    target = out / (ep.name + '.json')
    target.write_text(json.dumps(record))
    monkeypatch.setattr(harness, 'label_episode', lambda *a, **kw: pytest.fail('uncertain paid dispatch replayed'))
    assert harness.run_batch([ep], out, keys=['sk-or-test'], concurrency=1, force=False, **settings) == 1
    assert json.loads(target.read_text()) == record


def test_run_parsed_count_revalidates_saved_flags(tmp_path):
    (tmp_path / 'out').mkdir()
    for name, labels in [('empty', {'timeline': []}), ('placeholder', {'timeline': [{}]}),
                         ('quality', {'data_issues': [{'issue': 'The camera is covered.'}]})]:
        (tmp_path / 'out' / (name + '.json')).write_text(json.dumps({'parse_ok': True, 'labels': labels}))
    assert run.parsed(tmp_path, ['empty', 'placeholder', 'quality']) == 1


def test_resume_placeholder_without_billing_proof_fails_closed(tmp_path, monkeypatch):
    from test_label import _packed_episode
    ep, _ = _packed_episode(tmp_path)
    out = tmp_path / 'out'
    out.mkdir()
    settings = dict(model=harness.DEFAULT_MODEL, reasoning='medium', max_tokens=100, cell_w=128, example_dir=None)
    identity = harness.input_identity(ep, **settings)
    record = {'parse_ok': True, 'labels': {'timeline': []}, 'input_identity': identity}
    target = out / (ep.name + '.json')
    target.write_text(json.dumps(record))
    monkeypatch.setattr(harness, 'label_episode', lambda *a, **kw: pytest.fail('unverified charge replayed'))
    assert harness.run_batch([ep], out, keys=['sk-or-test'], concurrency=1, force=False, **settings) == 1
    assert json.loads(target.read_text()) == record
