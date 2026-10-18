"""Offline provider and process regressions for dispatch, durable replies, and billing."""
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys

import numpy as np
import pytest

from label import evidence_access as ea, harness, run
from label.dictionary_stage import output_spend
from test_camera_timing import recording


def response(content, cost=.003):
    return io.BytesIO(json.dumps({'usage': {'cost': cost}, 'choices': [{
        'finish_reason': 'stop', 'message': {'content': content}}]}).encode())


def options(**extra):
    return dict(keys=['sk-or-fixture'], concurrency=1, force=False, model=harness.DEFAULT_MODEL,
                reasoning='medium', max_tokens=100, timeout=1, cell_w=128, max_spend=2., **extra)


def provider_kind(request):
    return 'selection' if 'FIELD INVENTORY' in json.dumps(json.loads(request.data)['messages']) else 'final'


def test_run_fractional_cent_cap_reaches_real_harness_without_rounding(tmp_path, monkeypatch):
    recording(tmp_path, np.arange(12) / 30)
    sent, commands = [], []
    monkeypatch.setattr(run, 'commit', lambda: ('fixture', False))
    monkeypatch.setenv('OPENROUTER_API_KEYS', 'sk-or-fixture')
    def transport(request, timeout):
        sent.append(provider_kind(request))
        return response(json.dumps({'requests': []} if sent[-1] == 'selection' else {'task_summary': 'A hand reaches.'}))
    monkeypatch.setattr(harness.urllib.request, 'urlopen', transport)
    def finish(folder, info, command, slice_dir, mode):
        commands.append(command)
        monkeypatch.setattr(sys, 'argv', command[2:])
        assert harness.main() == 0
    monkeypatch.setattr(run, 'finish', finish)
    monkeypatch.setattr(sys, 'argv', ['label', '--dataset', 'fixture', '--episodes', str(tmp_path),
        '--kind', 'full', '--cap', '.004', '--runs', str(tmp_path / 'runs'), '--', '--cell-w', '128', '--max-tokens', '100'])
    run.main()
    command = commands[0]
    assert float(command[command.index('--max-spend') + 1]) == .004
    assert sent == []


def test_received_unparsed_reply_is_not_replayed_and_force_keeps_original(tmp_path, monkeypatch):
    ep = recording(tmp_path, np.arange(12) / 30)
    sent = []
    def transport(request, timeout):
        kind = provider_kind(request)
        sent.append(kind)
        return response('{"requests":[]}' if kind == 'selection' else 'not valid JSON')
    monkeypatch.setattr(harness.urllib.request, 'urlopen', transport)
    out = tmp_path / 'out'
    assert harness.run_batch([ep], out, **options()) == 1
    original = (out / (ep.name + '.json')).read_bytes()
    assert harness.run_batch([ep], out, **options()) == 1
    assert sent == ['selection', 'final'] and (out / (ep.name + '.json')).read_bytes() == original
    assert output_spend(out) == pytest.approx(.006)
    monkeypatch.setattr(harness.urllib.request, 'urlopen', lambda request, timeout:
        response('{"requests":[]}' if provider_kind(request) == 'selection' else '{"task_summary":"A hand reaches."}'))
    args = options(); args['force'] = True
    assert harness.run_batch([ep], out, **args) == 0
    current = json.loads((out / (ep.name + '.json')).read_text())
    assert any(row.get('response_text') == 'not valid JSON' for row in current['final_cost_history'])
    assert output_spend(out) == pytest.approx(.012)


@pytest.mark.parametrize('evidence_id', [[], {}])
def test_unhashable_evidence_reference_keeps_paid_reply(tmp_path, monkeypatch, evidence_id):
    ep = recording(tmp_path, np.arange(12) / 30)
    final = {'task_summary': 'A hand reaches.', 'evidence_findings': [{
        'start_s': 0, 'end_s': .2, 'headline': 'Recorded reading', 'adds_beyond_video': True,
        'evidence': [{'evidence_id': evidence_id, 'time_s': [0]}]}]}
    sent = []
    def transport(request, timeout):
        kind = provider_kind(request); sent.append(kind)
        return response(json.dumps({'requests': []} if kind == 'selection' else final))
    monkeypatch.setattr(harness.urllib.request, 'urlopen', transport)
    out = tmp_path / 'out'
    assert harness.run_batch([ep], out, **options()) == 0
    result = json.loads((out / (ep.name + '.json')).read_text())
    assert result['parse_ok'] and result['labels'] == final
    assert result['sensor_evidence']['unbound_findings']
    assert output_spend(out) == pytest.approx(.006)
    assert harness.run_batch([ep], out, **options()) == 0 and sent == ['selection', 'final']


def test_batch_process_lock_prevents_duplicate_paid_dispatch(tmp_path, monkeypatch):
    ep = recording(tmp_path, np.arange(12) / 30)
    ctx = mp.get_context('fork')
    entered, release, results = ctx.Event(), ctx.Event(), ctx.Queue()
    log = tmp_path / 'calls'
    def transport(request, timeout):
        kind = provider_kind(request)
        with log.open('a') as stream: stream.write(kind + '\n')
        if kind == 'selection':
            entered.set()
            assert release.wait(5)
        return response('{"requests":[]}' if kind == 'selection' else '{"task_summary":"A hand reaches."}')
    monkeypatch.setattr(harness.urllib.request, 'urlopen', transport)
    def work():
        try: results.put(harness.run_batch([ep], tmp_path / 'out', **options()))
        except RuntimeError as error: results.put(str(error))
    first, second = ctx.Process(target=work), ctx.Process(target=work)
    first.start()
    try:
        assert entered.wait(5)
        second.start(); second.join(3)
        assert not second.is_alive(), 'concurrent batch must reject without waiting'
    finally:
        release.set()
        first.join(5)
        if second.is_alive(): second.terminate(); second.join()
    assert first.exitcode == 0 and second.exitcode == 0
    outcomes = [results.get(timeout=1), results.get(timeout=1)]
    assert 0 in outcomes and any(isinstance(value, str) and 'running' in value for value in outcomes)
    assert log.read_text().splitlines() == ['selection', 'final']
    assert output_spend(tmp_path / 'out') == pytest.approx(.006)


def test_interrupt_cancels_queued_episodes_and_retains_inflight_reply(tmp_path, monkeypatch):
    episodes = []
    for index in range(3):
        folder = tmp_path / str(index); folder.mkdir()
        episodes.append(recording(folder, np.arange(12) / 30))
        episodes[-1].rename(folder / ('episode_' + str(index)))
        episodes[-1] = folder / ('episode_' + str(index))
    ctx = mp.get_context('fork')
    entered, release, results = ctx.Event(), ctx.Event(), ctx.Queue()
    log = tmp_path / 'calls'
    def transport(request, timeout):
        kind = provider_kind(request)
        with log.open('a') as stream: stream.write(kind + '\n')
        if kind == 'final':
            entered.set()
            assert release.wait(5)
        return response('{"requests":[]}' if kind == 'selection' else '{"task_summary":"A hand reaches."}')
    monkeypatch.setattr(harness.urllib.request, 'urlopen', transport)
    def work():
        try: results.put(harness.run_batch(episodes, tmp_path / 'out', **options()))
        except KeyboardInterrupt: results.put(130)
    worker = ctx.Process(target=work); worker.start()
    try:
        assert entered.wait(5)
        os.kill(worker.pid, signal.SIGINT)
        # The parent observes cancellation before releasing the in-flight transport.
        import time
        time.sleep(.05)
        release.set(); worker.join(5)
    finally:
        release.set()
        if worker.is_alive(): worker.terminate(); worker.join()
    assert worker.exitcode == 0 and results.get(timeout=1) == 130
    assert log.read_text().splitlines() == ['selection', 'final']
    assert (tmp_path / 'out' / 'episode_0.json').exists()
    assert not (tmp_path / 'out' / 'episode_1.json').exists()
    assert not (tmp_path / 'out' / 'noreply_episode_1.json').exists()
    assert output_spend(tmp_path / 'out') == pytest.approx(.006)


def test_unexpected_binding_failure_retains_reparseable_received_response(tmp_path, monkeypatch):
    from label import reparse
    ep = recording(tmp_path, np.arange(12) / 30)
    monkeypatch.setattr(harness.urllib.request, 'urlopen', lambda request, timeout:
        response('{"requests":[]}' if provider_kind(request) == 'selection' else '{"task_summary":"A hand reaches."}'))
    original = ea.Access.bind
    monkeypatch.setattr(ea.Access, 'bind', lambda *args: (_ for _ in ()).throw(RuntimeError('binding failed')))
    out = tmp_path / 'out'
    assert harness.run_batch([ep], out, **options()) == 1
    record = json.loads((out / (ep.name + '.json')).read_text())
    assert record['response_text'] == '{"task_summary":"A hand reaches."}'
    assert record['usage']['est_cost_usd'] == .003 and record['final_dispatch_outcome'] == 'received'
    assert output_spend(out) == pytest.approx(.006)
    monkeypatch.setattr(ea.Access, 'bind', original)
    monkeypatch.setattr(harness.urllib.request, 'urlopen', lambda *args, **kwargs: pytest.fail('reparse dispatched'))
    (tmp_path / 'run.json').write_text('{}')
    assert reparse.reparse(tmp_path)['parsed_now'] == 1
    assert json.loads((out / (ep.name + '.json')).read_text())['parse_ok']


def test_model_finding_cannot_supply_presentation_overrides_or_time_offsets():
    from label import sensor_evidence as se
    from test_label_report_regressions import sensor_doc, finding
    raw = {**finding([80., 90.]), 'quoted_time_offset_s': ['n/a'],
           'overlay_headline': 'Invented overlay', 'overlay_inspect_s': 9000}
    bound = se.bind(sensor_doc(), [raw])
    assert len(bound['findings']) == 1
    assert all(key not in bound['findings'][0] for key in ('quoted_time_offset_s', 'overlay_headline', 'overlay_inspect_s'))


@pytest.mark.parametrize('dependency', ['label/evidence_access.py', 'prepare/display.py',
    'prepare/mcap_fields.py', 'prepare/hdf_metadata.py', 'checks/vendor/public_dataset_adapter_qc.py'])
def test_request_semantics_tracks_inspection_prompt_dependency(tmp_path, monkeypatch, dependency):
    ep = recording(tmp_path, np.arange(12) / 30)
    args = dict(model=harness.DEFAULT_MODEL, reasoning='medium', max_tokens=100, cell_w=128, example_dir=None)
    first = harness.input_identity(ep, **args)
    read = Path.read_bytes
    source = Path(harness.__file__).parent.parent / dependency
    def changed(path):
        return read(path) + b'\n# changed request dependency\n' if path == source else read(path)
    monkeypatch.setattr(Path, 'read_bytes', changed)
    second = harness.input_identity(ep, **args)
    assert not harness.same_input_identity(first, second)


def test_missing_source_is_rejected_before_paid_routing(tmp_path, monkeypatch):
    ep = recording(tmp_path, np.arange(12) / 30)
    sources = json.loads((ep / 'sources.json').read_text())
    Path(sources['exo']['packed']).unlink()
    sent = []
    monkeypatch.setattr(harness.urllib.request, 'urlopen', lambda *args, **kwargs: sent.append(1))
    args = options(); args['cell_w'] = 0
    assert harness.run_batch([ep], tmp_path / 'out', **args) == 1
    assert not sent


@pytest.mark.parametrize('prior_unknown', [False, True])
def test_stitch_keeps_paid_history_and_unknown_reservations(tmp_path, monkeypatch, prior_unknown):
    from compare import metrics
    from label import pieces
    from label.dictionary_stage import output_spend_complete
    ep = recording(tmp_path, np.arange(30) / 30)
    monkeypatch.setattr(pieces, 'piece_max', lambda ctx: .5)
    monkeypatch.setattr(pieces, 'CUT_GUARD_S', .05)
    job = tmp_path / 'job'
    parts = pieces.write_pieces(ep, job / 'pieces')
    assert len(parts) == 2
    context = json.loads((ep / 'context.json').read_text())
    (job / 'pieces' / ('.' + ep.name + '.source_proof.json')).write_text(json.dumps(ea.source_proof(context, ep)))
    unknown = [prior_unknown]
    def transport(request, timeout):
        kind = provider_kind(request)
        if kind == 'final' and unknown[0]:
            raise TimeoutError('unknown first final charge')
        return response('{"requests":[]}' if kind == 'selection' else '{"task_summary":"A hand reaches."}')
    monkeypatch.setattr(harness.urllib.request, 'urlopen', transport)
    source_out = job / 'run' / 'out'
    assert harness.run_batch(parts, source_out, **options()) == (1 if prior_unknown else 0)
    unknown[0] = False
    args = options(); args['force'] = True
    assert harness.run_batch(parts, source_out, **args) == 0
    expected = output_spend(source_out)
    if not prior_unknown: assert expected == pytest.approx(.024)
    final = tmp_path / 'final'
    pieces.stitch_run(job, ep.parent, {ep.name: [p.name for p in parts]}, final)
    assert output_spend(final) == pytest.approx(expected)
    assert output_spend_complete(final) is (not prior_unknown)
    assert metrics.read_output(final / (ep.name + '.json'))['cost'] == pytest.approx(expected, abs=1e-6)


def test_malformed_contact_and_task_objects_do_not_abort_stitch(tmp_path, monkeypatch):
    from label import pieces
    ep = recording(tmp_path, np.arange(30) / 30)
    monkeypatch.setattr(pieces, 'piece_max', lambda ctx: .5)
    monkeypatch.setattr(pieces, 'CUT_GUARD_S', .05)
    parts = pieces.write_pieces(ep, tmp_path / 'pieces')
    records = []
    for part in parts:
        context = json.loads((part / 'context.json').read_text())
        labels = {'task_summary': 'A hand reaches.', 'contacts': [{'id': ['invalid']}],
                  'tasks': [{'task': 'reach', 'objects': 4, 'start_s': 0, 'end_s': .5,
                             'note': 'continues from previous part; the clip ends mid-task'}]}
        parsed, ok = harness.parse_response(json.dumps(labels))
        assert ok
        records.append((context, {'episode_dir': str(part), 'parse_ok': True, 'labels': parsed,
                                  'usage': {'est_cost_usd': .003}}))
    stitched = pieces.stitch(ep, records)
    assert stitched['labels']['task_summary'] == 'A hand reaches.'
    assert not stitched['labels'].get('contacts')


def test_legacy_sensor_finding_offset_is_recomputed_from_part_clock():
    from label import sensor_evidence as se
    from test_label_report_regressions import sensor_doc, finding
    doc = sensor_doc()
    doc['version'] = 1
    doc['sensors'][0]['name'] = 'force'
    doc['findings'] = [{**finding([80., 90.]), 'quoted_time_offset_s': ['n/a'], 'overlay_headline': 'Wrong overlay'}]
    merged = se.merge([({'piece': {'index': 1, 't0_s': 3}}, {'sensor_evidence': doc})],
                      {'signals': [{'name': 'force'}]})
    result = merged['findings'] or [row['finding'] for row in merged['unbound_findings']]
    assert result[0]['quoted_time_offset_s'] == 3 and 'overlay_headline' not in result[0]


@pytest.mark.parametrize('force', [False, True])
def test_run_resume_before_primary_output_keeps_claim_and_finishes_untouched_work(tmp_path, monkeypatch, force):
    from types import SimpleNamespace
    root = tmp_path / 'episodes'; root.mkdir()
    episodes = []
    for index in range(2):
        folder = root / str(index); folder.mkdir()
        episode = recording(folder, np.arange(12) / 30)
        target = root / ('episode_' + str(index)); episode.rename(target); episodes.append(target)
    folder = tmp_path / 'run'; folder.mkdir()
    (folder / 'run.json').write_text(json.dumps({'dataset': 'fixture', 'kind': 'full', 'code': 'fixture',
        'slice': str(root), 'cap_usd': 2., 'status': 'exit 137'}))
    def killed(request, timeout):
        if provider_kind(request) == 'final': os._exit(137)
        return response('{"requests":[]}')
    monkeypatch.setattr(harness.urllib.request, 'urlopen', killed)
    worker = mp.get_context('fork').Process(target=lambda: harness.run_batch(episodes, folder / 'out', **options()))
    worker.start(); worker.join(5)
    assert worker.exitcode == 137
    claim = folder / 'out' / 'noreply_episode_0.json'
    before = claim.read_bytes()
    spent = output_spend(folder / 'out')
    assert not list((folder / 'out').glob('episode_*.json'))
    sent = []
    def transport(request, timeout):
        kind = provider_kind(request); sent.append(kind)
        return response('{"requests":[]}' if kind == 'selection' else '{"task_summary":"A hand reaches."}')
    monkeypatch.setattr(harness.urllib.request, 'urlopen', transport)
    monkeypatch.setattr(run, 'commit', lambda: ('fixture', False))
    monkeypatch.setenv('OPENROUTER_API_KEYS', 'sk-or-fixture')
    def finish(run_dir, info, command, slice_dir, mode):
        monkeypatch.setattr(sys, 'argv', command[2:])
        assert harness.main() == 0
    monkeypatch.setattr(run, 'finish', finish)
    args = SimpleNamespace(resume=folder, dataset='fixture', kind='full', why='killed child', concurrency=1,
        harness_args=['--cell-w', '128', '--max-tokens', '100'] + (['--force'] if force else []))
    run.resume(args, root)
    assert (folder / 'out' / 'episode_1.json').exists()
    if force:
        assert (folder / 'out' / 'episode_0.json').exists()
        assert output_spend(folder / 'out') == pytest.approx(spent + .012)
    else:
        assert claim.read_bytes() == before and not (folder / 'out' / 'episode_0.json').exists()
        assert sent == ['selection', 'final']
        assert output_spend(folder / 'out') == pytest.approx(spent + .006)


def test_inspection_replay_recomputes_failure_notes_without_rebilling(tmp_path, monkeypatch):
    from test_evidence_access import access
    first = access(tmp_path, new_fact=7)
    field = next(field for field in first.inventory() if field['name'] == 'new_fact')
    calls = []
    def select(content):
        calls.append(1)
        return {'usage': {'cost': .01}, 'choices': [{'message': {'content': json.dumps({
            'requests': [{'field_id': field['id'], 'mode': 'metadata'}]})}}]}
    monkeypatch.setattr(first, 'inspect', lambda request: (_ for _ in ()).throw(ValueError('temporarily unavailable')))
    cache = tmp_path / 'inspection.json'
    failed, _ = ea.discover(first, [], select, cache, rounds=1, reserve_cost=.2)
    assert failed['status'] == 'partial'
    current, _ = ea.discover(access(tmp_path, new_fact=7), [], select, cache, rounds=1, reserve_cost=.2)
    assert current['status'] == 'complete' and not current['limitations'] and calls == [1]
    for _ in range(2):
        broken = access(tmp_path, new_fact=7)
        monkeypatch.setattr(broken, 'inspect', lambda request: (_ for _ in ()).throw(ValueError('currently unavailable')))
        current, _ = ea.discover(broken, [], select, cache, rounds=1, reserve_cost=.2)
    assert current['status'] == 'partial' and len(current['limitations']) == 1 and calls == [1]


def test_numeric_citation_columns_require_integer_ids(tmp_path):
    from test_evidence_access import access
    inspected = access(tmp_path)
    inspected.ep['signals']['reading'] = np.arange(4)[:, None]
    inspected.ep['signal_meta']['reading'] = {}
    inspected.refresh()
    field = next(field for field in inspected.inventory() if field['name'] == 'reading')
    receipt = inspected.inspect({'field_id': field['id']})
    finding = {'start_s': 0, 'end_s': .3, 'headline': 'Recorded increase', 'adds_beyond_video': True,
               'evidence': [{'evidence_id': receipt['id'], 'time_s': [0, .3], 'columns': [0.0]}]}
    bound = inspected.bind([finding])
    assert not bound['findings'] and bound['unbound_findings']
