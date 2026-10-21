"""Offline paid receipt authority across refresh, refusal, repair and consumers."""
import io
import json
import urllib.error

import numpy as np
import pytest

from label import harness, reparse, evidence_access
from label.dictionary_stage import output_spend
from board.to_board import label_outputs
from compare.metrics import response as compared
from test_camera_timing import recording
from test_dispatch_integrity_v5 import options, provider_kind, response
from test_execution_integrity_v6 import episodes
import complete_reply


def transport_fixture(monkeypatch, contents, cost=.003):
    calls = []
    def transport(request, timeout):
        kind = provider_kind(request); calls.append(kind)
        if kind == 'selection':
            return response('{"requests":[]}')
        content, finish = next(contents)
        result = json.loads(response(content, cost).read())
        result['choices'][0]['finish_reason'] = finish
        return io.BytesIO(json.dumps(result).encode())
    monkeypatch.setattr(harness.urllib.request, 'urlopen', transport)
    return calls


def reference_result(run, ep):
    from compare.metrics import compute
    manifest = {'datasets': [{'dataset': 'fixture', 'run': str(run), 'episodes': str(ep.parent)}],
                'comparisons': [{'key': 'copy', 'name': 'Copy', 'run': str(run), 'episodes': str(ep.parent)}]}
    return compute(manifest, run)['episodes'][0]['by'].get('board')


def test_cap_refusal_does_not_hide_received_refresh(tmp_path, monkeypatch):
    ep = recording(tmp_path, np.arange(12) / 30); out = tmp_path / 'out'
    calls = transport_fixture(monkeypatch, iter([
        (complete_reply.text('A hand reaches.'), 'stop'), ('{"task_summary":', 'length'),
        (complete_reply.text('A hand stops.'), 'stop')]))
    assert harness.run_batch([ep], out, **options()) == 0
    force = options(); force['force'] = True
    assert harness.run_batch([ep], out, **force) == 1
    low = options(); low['max_spend'] = .000001
    harness.run_batch([ep], out, retry_received=[ep.name], **low)
    assert calls.count('final') == 2 and output_spend(out) == pytest.approx(.012)
    assert harness.run_batch([ep], out, retry_received=[ep.name], **options()) == 0
    assert calls.count('final') == 3 and output_spend(out) == pytest.approx(.015)


def test_binding_repair_agrees_with_board_and_compare(tmp_path, monkeypatch):
    ep = recording(tmp_path, np.arange(12) / 30); out = tmp_path / 'out'
    transport_fixture(monkeypatch, iter([(complete_reply.text('A hand reaches.'), 'stop')]))
    bind = evidence_access.Access.bind
    monkeypatch.setattr(evidence_access.Access, 'bind', lambda *args: (_ for _ in ()).throw(RuntimeError('bind failed')))
    assert harness.run_batch([ep], out, **options()) == 1
    monkeypatch.setattr(evidence_access.Access, 'bind', bind)
    (tmp_path / 'run.json').write_text('{}')
    assert reparse.reparse(tmp_path)['parsed_now'] == 1
    outputs, _ = label_outputs(out)
    assert outputs[ep.name][1]['parse_ok'] is True
    assert compared({'run': tmp_path, 'status': 'done'}, ep.name)['status'] == 'parsed'
    assert output_spend(out) == pytest.approx(.006)


def test_newest_refresh_controls_readers_and_billing(tmp_path, monkeypatch):
    ep = recording(tmp_path, np.arange(12) / 30); out = tmp_path / 'out'
    transport_fixture(monkeypatch, iter([
        (complete_reply.text('A hand reaches.'), 'stop'), ('{"task_summary":', 'length')]))
    assert harness.run_batch([ep], out, **options()) == 0
    force = options(); force['force'] = True
    assert harness.run_batch([ep], out, **force) == 1
    result = compared({'run': tmp_path, 'status': 'exit 1'}, ep.name)
    assert result['status'] == 'cut_off'
    assert result['cost'] == pytest.approx(.012)


def test_completed_mixed_limits_resume_without_payment(tmp_path, monkeypatch):
    eps = episodes(tmp_path, 2); out = tmp_path / 'out'
    calls = transport_fixture(monkeypatch, iter([
        (complete_reply.text('A hand reaches.'), 'stop'), ('invalid', 'stop'),
        (complete_reply.text('A hand stops.'), 'stop')]))
    assert harness.run_batch(eps, out, **options()) == 1
    retry = options(); retry['max_tokens'] = 200
    assert harness.run_batch(eps, out, retry_received=[eps[1].name], **retry) == 0
    for limit in (100, 200):
        resume = options(); resume['max_tokens'] = limit
        assert harness.run_batch(eps, out, **resume) == 0
    assert calls.count('final') == 3


def test_unknown_charge_plain_resume_is_not_success(tmp_path, monkeypatch):
    ep = recording(tmp_path, np.arange(12) / 30); out = tmp_path / 'out'
    def transport(request, timeout):
        if provider_kind(request) == 'selection': return response('{"requests":[]}')
        raise TimeoutError('response may have been charged')
    monkeypatch.setattr(harness.urllib.request, 'urlopen', transport)
    monkeypatch.setattr(harness.time, 'sleep', lambda *args: None)
    assert harness.run_batch([ep], out, **options()) == 1
    original = harness.billing_record(out, ep.name + '.json')[1]
    monkeypatch.setattr(harness.urllib.request, 'urlopen', lambda *args, **kwargs: pytest.fail('unknown replay'))
    assert harness.run_batch([ep], out, **options()) == 1
    assert harness.billing_record(out, ep.name + '.json')[1] == original


@pytest.mark.parametrize('length', [20, 1000])
def test_structured_quota_rejection_is_parsed_before_display_truncation(monkeypatch, length):
    def transport(request, timeout):
        body = json.dumps({'error': {'code': 'insufficient_quota', 'message': 'x' * length}}).encode()
        raise urllib.error.HTTPError(harness.OPENAI_URL, 429, 'quota', {}, io.BytesIO(body))
    monkeypatch.setattr(harness.urllib.request, 'urlopen', transport)
    with pytest.raises(harness.KeyExhausted):
        harness._post(harness.OPENAI_URL, {}, 'sk-fixture', 1)


@pytest.mark.parametrize('refresh', [False, True])
def test_gate_counts_paid_unusable_reply_before_grading(tmp_path, monkeypatch, refresh):
    from gate import score
    ep = recording(tmp_path, np.arange(12) / 30); out = tmp_path / 'out'
    replies = ([(complete_reply.text('A hand reaches.'), 'stop')] if refresh else []) + [('invalid', 'length' if refresh else 'stop')]
    transport_fixture(monkeypatch, iter(replies), cost=.25)
    if refresh:
        assert harness.run_batch([ep], out, **options()) == 0
    args = options(); args['force'] = refresh
    assert harness.run_batch([ep], out, **args) == 1
    monkeypatch.setattr(score, 'load_selection', lambda: {'fixture': [{'name': ep.name, 'role': 'cost:fixture'}]})
    monkeypatch.setattr(score, 'load_cases', lambda: {})
    result = score.score([tmp_path])['fixture']
    assert result['parsed'] == 0
    assert result['cost_by_dataset']['fixture']['billed_usd'] == pytest.approx(output_spend(out))
    assert result['cost_by_dataset']['fixture']['episodes'] == 1
    if refresh:
        assert result['cost_sample']['cold_usd'] is None


def test_piece_reader_uses_repaired_reply_under_local_refusal(tmp_path):
    from label.pieces import _part_reply, _carry_part_billing
    out = tmp_path / 'out'; out.mkdir()
    paid = {'final_attempt_index': 1, 'final_dispatch_outcome': 'received', 'parse_ok': True,
            'labels': complete_reply.labels('A hand reaches.'), 'usage': {'est_cost_usd': .123},
            'config': {'resolution_route': {'cost_usd': .004}}}
    (out / 'episode_1.json').write_text(json.dumps(paid))
    refused = {'final_attempt_index': 2, 'final_dispatch_outcome': 'not dispatched',
               'usage': {}, 'parse_ok': False, 'final_cost_history': [paid],
               'auxiliary_cost_history': [{'kind': 'route', 'cost_usd': .004}]}
    (out / 'noreply_episode_1.json').write_text(json.dumps(refused))
    record, error = _part_reply(out, 'episode_1')
    assert error is None and record['labels'] == paid['labels']
    result = {'usage': {'est_cost_usd': .123}, 'config': {'resolution_route': {'cost_usd': .004}}}
    _carry_part_billing(result, [({'piece': {'index': 1}}, record)])
    assert harness.episode_cost(result) == pytest.approx(.127)


def test_review_named_retry_preserves_cap_and_history(tmp_path, monkeypatch):
    import sys
    import h5py
    from review import __main__ as review
    from label import dictionary_stage
    upload = tmp_path / 'upload'; upload.mkdir()
    with h5py.File(upload / 'recording.h5', 'w') as source:
        source.attrs['fps'] = 10
        source.create_dataset('images/top', data=np.full((12, 36, 64, 3), 60, np.uint8))
    job = tmp_path / 'job'
    calls = transport_fixture(monkeypatch, iter([('invalid', 'length'), (complete_reply.text('A hand reaches.'), 'stop')]))
    monkeypatch.setenv('OPENROUTER_API_KEYS', 'sk-or-fixture')
    monkeypatch.setattr(dictionary_stage, 'prepare', lambda *args, **kwargs: None)
    monkeypatch.setattr(review, 'build_review_board', lambda *args, **kwargs: {'incomplete': [], 'unlabelled': []})
    commands = []
    def step(job, kind, command, env, **kwargs):
        if kind in ('dry_run', 'label'):
            commands.append(command)
            with monkeypatch.context() as local:
                local.setattr(sys, 'argv', command[2:])
                for name in ('ARGUS_REVIEW_OUTPUT_LOCK_FD', 'ARGUS_REVIEW_OUTPUT_LOCK_PATH'):
                    if name in env:
                        local.setenv(name, env[name])
                return harness.main()
        return 0
    monkeypatch.setattr(review, 'run_step', step)
    argv = ['review', '--data', str(upload), '--out', str(job), '--rig', 'ego_head', '--cap', '2',
            '--concurrency', '1', '--max-tokens', '100']
    monkeypatch.setattr(sys, 'argv', argv)
    review.main()
    name = next((job / 'units').glob('episode_*')).name
    paid = output_spend(job / 'run' / 'out')
    retry = argv[:-1] + ['200', '--retry-received', name]
    monkeypatch.setattr(sys, 'argv', retry)
    review.main()
    assert calls.count('final') == 2
    assert output_spend(job / 'run' / 'out') == pytest.approx(paid + .003)
    label_command = commands[-1]
    assert float(label_command[label_command.index('--max-spend') + 1]) == pytest.approx(2 - paid)
    result = json.loads((job / 'run' / 'out' / (name + '.json')).read_text())
    assert result['final_cost_history'][-1]['finish_reason'] == 'length'


def test_reparse_received_history_retains_newer_financial_receipt(tmp_path, monkeypatch):
    ep = recording(tmp_path, np.arange(12) / 30); out = tmp_path / 'out'
    transport_fixture(monkeypatch, iter([(complete_reply.text('A hand reaches.'), 'stop')]))
    bind = evidence_access.Access.bind
    monkeypatch.setattr(evidence_access.Access, 'bind', lambda *args: (_ for _ in ()).throw(RuntimeError('bind failed')))
    assert harness.run_batch([ep], out, **options()) == 1
    monkeypatch.setattr(evidence_access.Access, 'bind', bind)
    low = options(); low['max_spend'] = .000001
    harness.run_batch([ep], out, retry_received=[ep.name], **low)
    primary = out / (ep.name + '.json'); primary.unlink()
    (tmp_path / 'run.json').write_text('{}')
    ledger_path, before = harness.billing_record(out, ep.name + '.json')
    # Legacy retained attempts can have the raw response without a labels placeholder.
    before['final_cost_history'][-1].pop('labels', None)
    ledger_path.write_text(json.dumps(before))
    assert reparse.reparse(tmp_path)['parsed_now'] == 1
    latest = harness.billing_record(out, ep.name + '.json')[1]
    assert latest['final_attempt_index'] == before['final_attempt_index']
    assert latest['final_dispatch_outcome'] == 'not dispatched'
    assert latest['final_cost_history'][-1]['response_text'] == complete_reply.text('A hand reaches.')
    assert latest['final_cost_history'][-1]['parse_ok'] is True
    assert output_spend(out) == pytest.approx(.006)
    outputs, _ = label_outputs(out)
    assert outputs[ep.name][1]['parse_ok'] is True
    reference = reference_result(tmp_path, ep)
    assert reference is not None and reference['status'] == 'parsed'
    assert reference['cost'] == pytest.approx(output_spend(out))
    monkeypatch.setattr(harness.urllib.request, 'urlopen', lambda *args, **kwargs: pytest.fail('repair repurchased'))
    assert harness.run_batch([ep], out, **options()) == 0


def test_actual_stitch_after_cap_refused_refresh_keeps_all_charges(inspect_every_field, tmp_path, monkeypatch):
    from test_pieces import _three_parts
    from label import pieces
    job, ep, parts = _three_parts(tmp_path, monkeypatch)
    proof = evidence_access.source_proof(json.loads((ep / 'context.json').read_text()), ep)
    (job / 'pieces' / f'.{ep.name}.source_proof.json').write_text(json.dumps(proof))
    out = job / 'run' / 'out'
    calls = transport_fixture(monkeypatch, iter([
        *[(complete_reply.text('A hand reaches.'), 'stop')] * 3,
        ('{"task_summary":', 'length'), (complete_reply.text('A hand stops.'), 'stop')]))
    assert harness.run_batch(parts, out, **options()) == 0
    force = options(); force['force'] = True
    assert harness.run_batch(parts[:1], out, **force) == 1
    low = options(); low['max_spend'] = .000001
    harness.run_batch(parts, out, retry_received=[parts[0].name], **low)
    final = tmp_path / 'final'
    state = pieces.stitch_run(job, ep.parent, {ep.name: [part.name for part in parts]}, final)
    assert state['incomplete'] == [ep.name]
    assert output_spend(final) == pytest.approx(output_spend(out)) == pytest.approx(.024)
    assert harness.run_batch(parts, out, retry_received=[parts[0].name], **options()) == 0
    state = pieces.stitch_run(job, ep.parent, {ep.name: [part.name for part in parts]}, final)
    assert not state['incomplete']
    assert output_spend(final) == pytest.approx(output_spend(out)) == pytest.approx(.027)
    assert calls.count('final') == 5


@pytest.mark.parametrize('body,attempts', [
    ({'error': {'code': 'insufficient_quota', 'message': 'x' * 70000}}, 1),
    ({'error': {'code': 'other', 'message': 'insufficient_quota'}}, 1),
    ({'error': {'code': 'insufficient_quota'}}, 2)])
def test_only_complete_first_dispatch_quota_proof_can_release_charge(monkeypatch, body, attempts):
    count = []
    def transport(request, timeout):
        count.append(1)
        if attempts == 2 and len(count) == 1:
            # a gateway timeout: the first attempt may have been billed, so a later quota answer proves nothing
            raise urllib.error.HTTPError(request.full_url, 504, 'unknown', {}, io.BytesIO(b'{}'))
        raise urllib.error.HTTPError(request.full_url, 429, 'quota', {}, io.BytesIO(json.dumps(body).encode()))
    monkeypatch.setattr(harness.urllib.request, 'urlopen', transport)
    monkeypatch.setattr(harness.time, 'sleep', lambda *args: None)
    with pytest.raises(RuntimeError) as error:
        harness._post(harness.OPENAI_URL, {}, 'sk-fixture', 1, attempts=attempts)
    assert not isinstance(error.value, harness.KeyExhausted)


def test_offline_rebind_preserves_human_rejection_of_same_finding():
    from test_tactile_contract import episode
    from label import sensor_evidence as se
    ep = episode({'contact': np.array([[0], [1], [1], [0], [0]])}, [{'name': 'contact', 'shape': [1]}],
                 {'contact': {'role': 'contact_state'}})
    doc = se.build(ep, {'n': 5, 'ks': [0, 2, 4]})
    finding = {'start_s': .1, 'end_s': .3, 'headline': 'Contact ends', 'claim_type': 'contact_change',
               'evidence': [{'sensor_id': doc['sensors'][0]['id'], 'series_id': doc['series'][0]['id'], 'time_s': [.2, .3]}]}
    bound = se.bind(doc, [finding]); bound['findings'][0]['review_status'] = 'rejected'
    different = {**finding, 'headline': 'A different finding'}
    raw = {**finding, 'time_origin_s': 999, 'start_label': 'invented'}
    record = {'labels': {'sensor_findings': [raw, different]}, 'sensor_evidence': bound}
    reparse.rebind(record)
    restored = record['sensor_evidence']['findings']
    assert restored[0]['review_status'] == 'rejected'
    assert 'review_status' not in restored[1]
    assert all('time_origin_s' not in finding and 'start_label' not in finding for finding in restored)


def test_compare_uses_financial_ledger_when_old_annotation_has_no_usage(tmp_path):
    out = tmp_path / 'out'; out.mkdir()
    primary = {'final_attempt_index': 1, 'parse_ok': True, 'labels': complete_reply.labels('A hand reaches.')}
    (out / 'episode_a.json').write_text(json.dumps(primary))
    ledger = {'final_attempt_index': 2, 'final_dispatch_outcome': 'not dispatched',
              'final_reserved_usd': 0.0, 'final_cost_history': [
                  {'final_attempt_index': 1, 'final_dispatch_outcome': 'received', 'usage': {'est_cost_usd': .1234567}}]}
    (out / 'noreply_episode_a.json').write_text(json.dumps(ledger))
    result = compared({'run': tmp_path, 'status': 'done'}, 'episode_a')
    assert result['status'] == 'parsed' and result['cost'] == pytest.approx(.1234567)


def test_reparse_valid_json_does_not_make_truncation_a_completed_reply(tmp_path, monkeypatch):
    ep = recording(tmp_path, np.arange(12) / 30); out = tmp_path / 'out'
    calls = transport_fixture(monkeypatch, iter([
        (complete_reply.text('A hand reaches.'), 'length'), (complete_reply.text('A hand stops.'), 'stop')]))
    assert harness.run_batch([ep], out, **options()) == 1
    (tmp_path / 'run.json').write_text('{}')
    assert reparse.reparse(tmp_path)['parsed_now'] == 1
    from board.to_board import label_failed
    from label.pieces import _part_reply
    from gate import score
    outputs, _ = label_outputs(out)
    assert label_failed(outputs[ep.name][1])['status'] == 'cut_off'
    assert compared({'run': tmp_path, 'status': 'failed'}, ep.name)['status'] == 'cut_off'
    assert 'cut off' in _part_reply(out, ep.name)[1]
    monkeypatch.setattr(score, 'load_selection', lambda: {'fixture': [{'name': ep.name, 'role': 'case'}]})
    monkeypatch.setattr(score, 'load_cases', lambda: {})
    assert score.score([tmp_path])['fixture']['parsed'] == 0
    assert harness.run_batch([ep], out, **options()) == 1
    assert calls.count('final') == 1
    assert harness.run_batch([ep], out, retry_received=[ep.name], **options()) == 0
    assert calls.count('final') == 2 and output_spend(out) == pytest.approx(.009)


def test_reference_cost_matches_spend_after_refused_refresh(tmp_path, monkeypatch):
    ep = recording(tmp_path, np.arange(12) / 30); out = tmp_path / 'out'
    calls = transport_fixture(monkeypatch, iter([(complete_reply.text('A hand reaches.'), 'stop')]))
    assert harness.run_batch([ep], out, **options()) == 0
    original = (out / (ep.name + '.json')).read_bytes()
    refresh = options(); refresh.update(force=True, max_tokens=1_000_000)
    harness.run_batch([ep], out, **refresh)
    # the refresh's final request cannot fit the cap, so no evidence selection is bought for it
    assert calls == ['selection', 'final']
    assert (out / (ep.name + '.json')).read_bytes() == original
    assert output_spend(out) == pytest.approx(.006)
    (tmp_path / 'run.json').write_text('{}')
    reference = reference_result(tmp_path, ep)
    assert reference['status'] == 'parsed'
    assert reference['cost'] == pytest.approx(output_spend(out))


def test_atomic_publication_cannot_pair_new_annotation_with_old_billing(tmp_path, monkeypatch):
    from pathlib import Path
    from label.atomic import write_atomic
    ep = recording(tmp_path, np.arange(12) / 30); out = tmp_path / 'out'
    transport_fixture(monkeypatch, iter([(complete_reply.text('A hand reaches.'), 'stop'),
                                       (complete_reply.text('A hand stops.'), 'stop')]))
    assert harness.run_batch([ep], out, **options()) == 0
    path = out / (ep.name + '.json'); old = path.read_bytes()
    refresh = options(); refresh['force'] = True
    assert harness.run_batch([ep], out, **refresh) == 0
    newer = json.loads(path.read_text()); path.write_bytes(old)
    read_text = Path.read_text; replaced = []
    def replace_between_reads(candidate, *args, **kwargs):
        text = read_text(candidate, *args, **kwargs)
        if candidate == path and not replaced:
            replaced.append(True)
            write_atomic(path, newer)
        return text
    monkeypatch.setattr(Path, 'read_text', replace_between_reads)
    _, selected = harness.annotation_record(out, path.name)
    ledger = harness.billing_view(selected)
    assert selected['final_attempt_index'] == ledger['final_attempt_index']
    assert harness.episode_cost(selected) == pytest.approx(.006 if selected['final_attempt_index'] == 1 else .012)


@pytest.mark.parametrize('mode', ['cap', 'timeout'])
def test_absent_final_response_is_not_a_schema_failure(tmp_path, monkeypatch, mode):
    from compare.metrics import _responses
    ep = recording(tmp_path, np.arange(12) / 30); out = tmp_path / 'out'
    calls = []
    def transport(request, timeout):
        kind = provider_kind(request); calls.append(kind)
        if kind == 'selection':
            return response('{"requests":[]}')
        raise TimeoutError('final response lost')
    monkeypatch.setattr(harness.urllib.request, 'urlopen', transport)
    settings = options()
    if mode == 'cap': settings['max_spend'] = .000001
    harness.run_batch([ep], out, **settings)
    result = compared({'run': tmp_path, 'status': 'exit 1'}, ep.name)
    counts = _responses('model', ['episode'], {'episode': {'by': {'model': result}}})
    assert result['status'] == 'no_response'
    assert counts['no_response'] == 1 and counts['unparsed'] == 0
    assert counts['parse_share'] is None
    # a lost response is sent again automatically, at most AUTO_RESENDS more times, and is still no_response
    assert calls == ([] if mode == 'cap' else ['selection'] + ['final'] * (1 + harness.AUTO_RESENDS))


def test_history_only_annotation_publishes_through_comparison_board(tmp_path):
    from board.build import build_comparisons
    from test_compare import _comparison
    spec = _comparison(tmp_path)
    primary = tmp_path / 'runs' / 'm2' / 'out' / 'episode_a.json'
    attempt = json.loads(primary.read_text())
    attempt.update(final_attempt_index=1, final_dispatch_outcome='received')
    attempt['episode_dir'] = str(tmp_path / 'slice' / 'episode_a')
    ledger = dict(episode_dir=attempt['episode_dir'], final_attempt_index=2,
                  final_dispatch_outcome='not dispatched', final_cost_history=[attempt], parse_ok=False)
    primary.with_name('noreply_' + primary.name).write_text(json.dumps(ledger))
    primary.unlink()
    qa = tmp_path / 'qa.new'; qa.mkdir()
    for name in ('a', 'b', 'c'):
        (qa / f'episode_{name}.json').write_text('{"dataset":"fixture"}')
    sources = {f'episode_{name}.json': tmp_path / 'runs' / 'ref' / 'out' / f'episode_{name}.json'
               for name in ('a', 'b', 'c')}
    result = build_comparisons(tmp_path, spec, qa, sources)
    published = json.loads((tmp_path / 'compare.new' / 'm2' / 'episode_a.json').read_text())
    assert published['_compare']['status'] == 'parsed'
    assert published['completion']['task_completed'] == 'success'
    assert result['models']['m2']['files'] == 3
    assert not primary.exists()


@pytest.mark.parametrize('finish', ['stop', 'length'])
def test_received_reply_without_invoice_remains_a_response(tmp_path, monkeypatch, finish):
    import io
    ep = recording(tmp_path, np.arange(12) / 30); out = tmp_path / 'out'
    def transport(request, timeout):
        if provider_kind(request) == 'selection': return response('{"requests":[]}')
        doc = json.loads(response(complete_reply.text('A hand reaches.')).read())
        doc.pop('usage', None)
        doc['choices'][0]['finish_reason'] = finish
        return io.BytesIO(json.dumps(doc).encode())
    monkeypatch.setattr(harness.urllib.request, 'urlopen', transport)
    harness.run_batch([ep], out, **options())
    result = compared({'run': tmp_path, 'status': 'exit 1'}, ep.name)
    assert result['status'] == ('parsed' if finish == 'stop' else 'cut_off')
    assert result['cost'] is None and result['cost_complete'] is False
    assert result['billed_usd'] == pytest.approx(.003)
    assert result['reserved_usd'] > 0


def test_mixed_round_invoices_and_final_history_do_not_become_billed_reservations():
    from label.receipts import billing_summary
    record = {'final_dispatch_outcome': 'not dispatched', 'final_cost_history': [
        {'final_dispatch_outcome': 'received', 'usage': {'est_cost_usd': .04}},
        {'final_dispatch_outcome': 'unverified', 'final_reserved_usd': .8}],
        'auxiliary_cost_history': [{'cost_usd': .2, 'cost_is_conservative_estimate': True,
            'rounds': [{'state': 'received', 'response': {'usage': {'cost': .003}}},
                       {'state': 'dispatched', 'reserved_cost_usd': .197}]}],
        'evidence_inspection': {'cost_usd': .01, 'previously_billed_usd': .004}}
    summary = billing_summary(record)
    assert summary['billed_usd'] == pytest.approx(.049)
    assert summary['reserved_usd'] == pytest.approx(.997)
    assert summary['cost_complete'] is False


def test_carrying_inspection_rounds_keeps_prior_invoice_attribution(tmp_path):
    from label.receipts import billing_summary
    out = tmp_path / 'out'; out.mkdir()
    record = {'final_dispatch_outcome': 'not dispatched', 'parse_ok': False,
        'auxiliary_cost_history': [{'kind': 'inspection', 'cost_usd': .003}],
        'evidence_inspection': {'cost_usd': .207, 'previously_billed_usd': .003,
            'cost_is_conservative_estimate': True, 'rounds': [
                {'state': 'received', 'response': {'usage': {'cost': .003}}},
                {'state': 'received', 'response': {'usage': {'cost': .004}}},
                {'state': 'dispatched', 'reserved_cost_usd': .2}]}}
    path = out / 'episode_x.json'; path.write_text(json.dumps(record))
    before = billing_summary(record)
    carried = billing_summary({'final_dispatch_outcome': 'not dispatched', **harness._billing_fields(path)})
    assert before['billed_usd'] == pytest.approx(.007)
    assert before['reserved_usd'] == pytest.approx(.2)
    assert carried == before
    from label.pieces import _carry_part_billing
    result = {'usage': {}, 'config': {'resolution_route': {}}}
    _carry_part_billing(result, [({'piece': {'index': 1}}, {'_billing_receipt': record})])
    assert billing_summary(result) == before


@pytest.mark.parametrize('previous', [0, .001])
def test_merging_inspection_parts_preserves_verified_rounds(previous):
    from label.receipts import billing_summary
    invoice = {'version': 1, 'cost_usd': .203, 'previously_billed_usd': previous,
        'cost_is_conservative_estimate': True, 'rounds': [
            {'state': 'received', 'response': {'usage': {'cost': .003}}},
            {'state': 'dispatched', 'reserved_cost_usd': .2}]}
    merged = evidence_access.merge([({'piece': {'index': 1, 't0_s': 0, 't1_s': 10}},
                                    {'evidence_inspection': invoice})])
    record = {'final_dispatch_outcome': 'not dispatched', 'evidence_inspection': merged}
    summary = billing_summary(record)
    assert summary['billed_usd'] == pytest.approx(.003 - previous)
    assert summary['reserved_usd'] == pytest.approx(.2)
    assert summary['cost_complete'] is False
