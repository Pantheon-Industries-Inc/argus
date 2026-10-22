"""Release journeys over converted uploads and durable provider receipts."""
import io
import json
import multiprocessing as mp
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from label import dictionary, dictionary_stage, harness, reparse, route
from review import __main__ as review
from test_dictionary import episode as dictionary_episode
from test_dispatch_integrity_v5 import options, provider_kind, response
from test_camera_timing import recording
import complete_reply


def review_fixture(tmp_path, monkeypatch):
    upload = tmp_path / 'upload'; upload.mkdir()
    with h5py.File(upload / 'take.h5', 'w') as source:
        source.attrs['fps'] = 10
        source.create_dataset('images/top', data=np.full((12, 36, 64, 3), 60, np.uint8))
        source.create_dataset('force', data=np.arange(12, dtype=float))
    job = tmp_path / 'job'
    monkeypatch.setenv('OPENROUTER_API_KEYS', 'sk-or-fixture')
    real_board = review.build_review_board
    def board(*args, **kwargs):
        from board import build
        with monkeypatch.context() as local:
            local.setattr(build, 'build', lambda job: None)
            return real_board(*args, **kwargs)
    monkeypatch.setattr(review, 'build_review_board', board)
    def step(job, kind, command, env, **kwargs):
        if kind in ('dry_run', 'label'):
            with monkeypatch.context() as local:
                local.setattr(sys, 'argv', command[2:])
                for name in ('ARGUS_REVIEW_OUTPUT_LOCK_FD', 'ARGUS_REVIEW_OUTPUT_LOCK_PATH'):
                    if name in env:
                        local.setenv(name, env[name])
                return harness.main()
        return 0
    monkeypatch.setattr(review, 'run_step', step)
    # these contracts cover the paid field dictionary's resume and billing, so it is turned on (default off)
    argv = ['review', '--data', str(upload), '--out', str(job), '--rig', 'ego_head',
            '--concurrency', '1', '--max-tokens', '100', '--field-dictionary', 'on']
    def run(*extra):
        monkeypatch.setattr(sys, 'argv', argv + list(extra))
        return review.main()
    return job, run


def funded_provider(monkeypatch, calls):
    def provider(request, timeout):
        text = json.dumps(json.loads(request.data)['messages'])
        if 'Interpret recorded field descriptors' in text:
            calls.append('dictionary')
            return response('{"entries":[]}')
        kind = provider_kind(request); calls.append(kind)
        return response('{"requests":[]}' if kind == 'selection' else complete_reply.text('A hand reaches.'))
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)


def test_unpaid_review_can_resume_with_dictionary_and_publish(tmp_path, monkeypatch):
    job, run = review_fixture(tmp_path, monkeypatch)
    calls = []; funded_provider(monkeypatch, calls)
    assert run('--cap', '.001') == 1
    assert calls == []
    assert run('--cap', '2') == 0
    assert calls[0] == 'dictionary'
    assert 'final' in calls
    assert json.loads((job / 'run' / 'run.json').read_text())['status'] == 'done'
    assert harness.annotation_complete(json.loads(next((job / 'run_final' / 'out').glob('episode_*.json')).read_text()))
    before = list(calls)
    assert run('--cap', '2') == 0
    assert calls == before


def test_free_then_paid_rebuilds_dry_requests_after_dictionary(tmp_path, monkeypatch):
    job, run = review_fixture(tmp_path, monkeypatch)
    calls = []; funded_provider(monkeypatch, calls)
    assert run('--free') == 0
    assert calls == []
    assert run('--cap', '2') == 0
    assert calls[0] == 'dictionary' and calls[-1] == 'final'
    assert json.loads((job / 'run' / 'run.json').read_text())['cost_complete']


def test_dictionary_inventory_does_not_change_when_review_cuts_pieces(tmp_path):
    ep = dictionary_episode(tmp_path, 'episode_x', [('force', np.ones((2, 1)), {})])
    before = dictionary.inventory([ep])['digest']
    path = ep / 'context.json'; ctx = json.loads(path.read_text())
    ctx['pieces'] = {'parts': ['episode_x__p01', 'episode_x__p02']}
    path.write_text(json.dumps(ctx))
    assert dictionary.inventory([ep])['digest'] == before


def test_review_rejects_competing_publisher_before_changing_metadata(tmp_path, monkeypatch):
    job, run = review_fixture(tmp_path, monkeypatch)
    calls = []; funded_provider(monkeypatch, calls)
    assert run('--cap', '2') == 0
    before = (job / 'run_final' / 'run.json').read_bytes()
    ctx = mp.get_context('fork'); entered, release = ctx.Event(), ctx.Event()
    def publisher():
        with harness.execution_lock(job):
            entered.set(); assert release.wait(5)
    worker = ctx.Process(target=publisher); worker.start()
    try:
        assert entered.wait(5)
        with pytest.raises(RuntimeError, match='Another label execution'):
            run('--cap', '2')
        assert (job / 'run_final' / 'run.json').read_bytes() == before
    finally:
        release.set(); worker.join(5)


def test_reparse_recovers_received_reply_without_releasing_unknown_charge(tmp_path, monkeypatch):
    ep = recording(tmp_path, np.arange(12) / 30)
    def provider(request, timeout):
        if provider_kind(request) == 'selection':
            return response('{"requests":[]}')
        return io.BytesIO(json.dumps({'choices': [{'finish_reason': 'stop', 'message': {
            'content': complete_reply.text('A hand reaches.')}}]}).encode())
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    original = harness.parse_response
    monkeypatch.setattr(harness, 'parse_response', lambda text: ({'_raw': text, '_parse_error': 'interrupted'}, False))
    assert harness.run_batch([ep], tmp_path / 'out', **options()) == 1
    before = dictionary_stage.output_spend(tmp_path / 'out')
    (tmp_path / 'run.json').write_text('{}')
    monkeypatch.setattr(harness, 'parse_response', original)
    assert reparse.reparse(tmp_path)['parsed_now'] == 1
    record = harness.annotation_record(tmp_path / 'out', ep.name + '.json')[1]
    assert harness.annotation_complete(record)
    assert dictionary_stage.output_spend(tmp_path / 'out') == before
    assert not dictionary_stage.output_spend_complete(tmp_path / 'out')


@pytest.mark.parametrize('finish,text', [('length', '{"fine_detail":false}'), ('stop', 'not json')])
def test_routing_keeps_unusable_paid_reply_and_uses_wide_cells(tmp_path, monkeypatch, finish, text):
    from test_route_accounting import observed_packed_episode
    ep = observed_packed_episode(tmp_path)
    monkeypatch.setattr(route, '_CACHE', {})
    def provider(content, model, *args, **kwargs):
        return {'usage': {'cost': .004}, 'choices': [{'finish_reason': finish, 'message': {'content': text}}]}
    monkeypatch.setattr(harness, 'call_model', provider)
    width, record = route.route_width(ep, 'sk-or-fixture', harness.call_model_once,
                                     out_path=tmp_path / 'episode_a.json')
    assert record['response_text'] == text and record['finish_reason'] == finish
    assert record['fine_detail'] is None
    assert width == max(harness.me.ROUTE_WIDTHS[harness.me.rig(harness.me.load(ep))])
    assert record['cost_usd'] == .004


def test_review_unknown_retry_names_fail_before_attempt_metadata(tmp_path, monkeypatch):
    job, run = review_fixture(tmp_path, monkeypatch)
    calls = []; funded_provider(monkeypatch, calls)
    assert run('--cap', '2') == 0
    before = (job / 'run' / 'run.json').read_bytes()
    with pytest.raises(SystemExit, match='unknown units.*available units'):
        run('--cap', '2', '--retry-received', 'mistyped')
    assert (job / 'run' / 'run.json').read_bytes() == before


def test_review_force_recovers_unknown_dispatch_with_original_hold(resends_used_up, tmp_path, monkeypatch):
    job, run = review_fixture(tmp_path, monkeypatch)
    calls = []; funded_provider(monkeypatch, calls)
    provider = harness.urllib.request.urlopen
    failed = []
    def lost(request, timeout):
        if provider_kind(request) == 'final' and 'Interpret recorded field descriptors' not in request.data.decode():
            failed.append(1)
            raise TimeoutError('fixture accepted request, response was lost')
        return provider(request, timeout)
    monkeypatch.setattr(harness.urllib.request, 'urlopen', lost)
    assert run('--cap', '2') == 1
    assert len(failed) == 1
    original_spend = dictionary_stage.label_spend(job)
    assert run('--cap', '2') == 1
    assert len(failed) == 1
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    assert run('--cap', '4', '--force') == 0
    assert dictionary_stage.label_spend(job) > original_spend
    assert not dictionary_stage.label_spend_complete(job)
    assert json.loads((job / 'run' / 'run.json').read_text())['cost_usd'] is None


def test_review_child_retains_publication_lock_after_wrapper_dies(tmp_path):
    ctx = mp.get_context('fork')
    ready = tmp_path / 'child-ready'
    def wrapper():
        with harness.execution_lock(tmp_path) as lock_fd, harness.execution_lock(tmp_path / 'run' / 'out') as output_fd:
            env = review.repo_env(ARGUS_REVIEW_LOCK_FD=str(lock_fd), ARGUS_REVIEW_OUTPUT_LOCK_FD=str(output_fd),
                                  ARGUS_REVIEW_OUTPUT_LOCK_PATH=str(tmp_path / 'run' / 'out' / '.execution.lock'))
            command = [sys.executable, '-c',
                'from pathlib import Path; import time; Path(' + repr(str(ready)) + ').touch(); time.sleep(4)']
            review.run_step(tmp_path, 'held', command, env)
    worker = ctx.Process(target=wrapper); worker.start()
    import time
    deadline = time.monotonic() + 3
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(.02)
    try:
        assert ready.exists()
        worker.kill(); worker.join(2)
        for folder in (tmp_path, tmp_path / 'run' / 'out'):
            with pytest.raises(RuntimeError, match='Another label execution'):
                with harness.execution_lock(folder):
                    pass
    finally:
        if worker.is_alive(): worker.kill(); worker.join()


def test_board_export_edit_does_not_repurchase_same_request(tmp_path, monkeypatch):
    ep = recording(tmp_path, np.arange(12) / 30)
    calls = []
    def provider(request, timeout):
        kind = provider_kind(request); calls.append(kind)
        return response('{"requests":[]}' if kind == 'selection' else complete_reply.text('A hand reaches.'))
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    out = tmp_path / 'out'
    assert harness.run_batch([ep], out, **options()) == 0
    original = Path.read_bytes
    target = Path(harness.__file__).resolve().parents[1] / 'board' / 'public_export.py'
    monkeypatch.setattr(Path, 'read_bytes', lambda path: original(path) + (b'\n# export-only edit\n' if path == target else b''))
    assert harness.run_batch([ep], out, **options()) == 0
    assert calls == ['selection', 'final']


def test_changed_reader_projection_stales_only_affected_request(tmp_path, monkeypatch):
    from board import public_export
    ep = recording(tmp_path, np.arange(12) / 30)
    context = ep / 'context.json'; value = json.loads(context.read_text())
    value['uploader_notes'] = {'recorded_metadata': {'private/path': {'fields': {'force': 'native value'}}}}
    context.write_text(json.dumps(value))
    settings = {k: v for k, v in options().items() if k in ('model', 'reasoning', 'max_tokens', 'cell_w')}
    settings['example_dir'] = None
    before = harness.input_identity(ep, **settings)
    original = public_export.reader_context_field
    monkeypatch.setattr(public_export, 'reader_context_field', lambda name, value:
                        {'changed': 'request-visible value'} if name == 'uploader_notes' else original(name, value))
    assert not harness.same_input_identity(before, harness.input_identity(ep, **settings))


def test_review_reuses_verified_preparation_across_producer_code_updates(tmp_path, monkeypatch):
    job, run = review_fixture(tmp_path, monkeypatch)
    calls = []; funded_provider(monkeypatch, calls)
    assert run('--cap', '2') == 0
    before = list(calls)
    producer = json.loads((job / 'preparation_receipt.json').read_text())['producer']
    monkeypatch.setattr(review, 'preparation_code_identity', lambda: {'new-reader-version': 'current'})
    assert run('--cap', '2') == 0
    assert calls == before
    assert json.loads((job / 'preparation_receipt.json').read_text())['producer'] == producer
    fresh = tmp_path / 'fresh-job'
    assert run('--cap', '2', '--out', str(fresh)) == 0
    assert json.loads((fresh / 'preparation_receipt.json').read_text())['producer'] == {'new-reader-version': 'current'}


def test_long_free_job_reuses_pieces_then_labels_without_rebuying_dictionary(tmp_path, monkeypatch):
    job, run = review_fixture(tmp_path, monkeypatch)
    with h5py.File(tmp_path / 'upload' / 'take.h5', 'w') as source:
        source.attrs['fps'] = 1
        source.create_dataset('images/top', data=np.full((452, 24, 32, 3), 70, np.uint8))
        source.create_dataset('force', data=np.arange(452, dtype=float))
    calls = []; funded_provider(monkeypatch, calls)
    assert run('--free') == 0
    assert len(list((job / 'units').glob('episode_*'))) >= 2
    assert run('--free') == 0
    assert calls == []
    assert run('--cap', '20') == 0
    assert calls.count('dictionary') == 1
    assert calls.count('final') >= 2
    before = list(calls)
    assert run('--cap', '20') == 0
    assert calls == before


def test_orphan_staged_harness_finalizes_run_after_wrapper_sigkill(tmp_path, monkeypatch):
    import time
    from label import run as staged
    ep = recording(tmp_path, np.arange(12) / 30)
    job = tmp_path / 'run'; job.mkdir()
    ready, release = tmp_path / 'ready', tmp_path / 'release'
    shim = tmp_path / 'shim'; shim.mkdir()
    (shim / 'sitecustomize.py').write_text('''import io, json, os, time, urllib.request
from pathlib import Path
def provider(request, timeout):
    selection = 'FIELD INVENTORY' in json.dumps(json.loads(request.data)['messages'])
    if not selection:
        Path(os.environ['READY']).write_text(str(os.getpid()))
        deadline = time.monotonic() + 8
        while not Path(os.environ['RELEASE']).exists() and time.monotonic() < deadline:
            time.sleep(.02)
    text = '{"requests":[]}' if selection else json.dumps({'task_summary': 'A hand reaches.',
        'timeline': [{'start_s': 0.0, 'end_s': 3600.0, 'action': 'reach'}], 'completion': {'task_completed': 'success'},
        'tasks': [{'task': 'reach', 'outcome': 'success'}]})
    return io.BytesIO(json.dumps({'usage': {'cost': .003}, 'choices': [
        {'finish_reason': 'stop', 'message': {'content': text}}]}).encode())
urllib.request.urlopen = provider
''')
    monkeypatch.setenv('PYTHONPATH', str(shim))
    monkeypatch.setenv('OPENROUTER_API_KEYS', 'sk-or-fixture')
    monkeypatch.setenv('READY', str(ready)); monkeypatch.setenv('RELEASE', str(release))
    info = {'run_id': 'r', 'kind': 'full', 'dataset': 'fixture', 'slice': str(ep.parent),
            'status': 'running', 'cap_usd': 2, 'code': 'fixture'}
    (job / 'run.json').write_text(json.dumps(info))
    command = [sys.executable, '-m', 'label.harness', '--episode-dir', str(ep), '--out',
               str(job / 'out' / (ep.name + '.json')), '--cell-w', '128', '--max-tokens', '100', '--max-spend', '2']
    ctx = mp.get_context('fork')
    wrapper = ctx.Process(target=staged.finish, args=(job, info, command, ep.parent, 'w'))
    wrapper.start()
    deadline = time.monotonic() + 5
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(.02)
    try:
        assert ready.exists()
        wrapper.kill(); wrapper.join(2)
        release.touch()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            saved = json.loads((job / 'run.json').read_text())
            if saved.get('status') != 'running':
                break
            time.sleep(.02)
        assert saved['status'] == 'done'
        assert saved['cost_usd'] == pytest.approx(.006)
        assert saved['episodes_parsed'] == 1
    finally:
        release.touch()
        if wrapper.is_alive(): wrapper.kill(); wrapper.join(2)


def test_competing_review_cannot_publish_while_paid_review_is_in_flight(tmp_path, monkeypatch):
    job, run = review_fixture(tmp_path, monkeypatch)
    calls = []; funded_provider(monkeypatch, calls)
    provider = harness.urllib.request.urlopen
    ctx = mp.get_context('fork'); entered, release, result = ctx.Event(), ctx.Event(), ctx.Queue()
    log = tmp_path / 'paid-calls'
    def held(request, timeout):
        kind = provider_kind(request)
        with log.open('a') as stream:
            stream.write(kind + '\n')
        if kind == 'final' and 'Interpret recorded field descriptors' not in request.data.decode():
            entered.set(); assert release.wait(5)
        return provider(request, timeout)
    monkeypatch.setattr(harness.urllib.request, 'urlopen', held)
    def active():
        result.put(run('--cap', '2'))
    first = ctx.Process(target=active); first.start()
    try:
        assert entered.wait(5)
        before = (job / 'run' / 'run.json').read_bytes()
        with pytest.raises(RuntimeError, match='Another label execution'):
            run('--cap', '2')
        assert (job / 'run' / 'run.json').read_bytes() == before
    finally:
        release.set(); first.join(5)
        if first.is_alive(): first.terminate(); first.join(2)
    assert first.exitcode == 0 and result.get(timeout=1) == 0
    assert json.loads((job / 'run_final' / 'run.json').read_text())['status'] == 'done'
    assert harness.annotation_complete(json.loads(next((job / 'run_final' / 'out').glob('episode_*.json')).read_text()))
    assert log.read_text().splitlines().count('final') == 2  # Dictionary and annotation have distinct persisted replies.


def test_completed_review_remains_successful_when_allowance_is_exhausted(tmp_path, monkeypatch):
    job, run = review_fixture(tmp_path, monkeypatch)
    calls = []; funded_provider(monkeypatch, calls)
    assert run('--cap', '2') == 0
    before = list(calls)
    # Lowering the cap cannot authorize new calls, but completed work needs none.
    assert run('--cap', '.001') == 0
    assert calls == before


def test_standalone_harness_cannot_enter_between_review_dry_run_and_publication(tmp_path, monkeypatch):
    import os
    import time
    job, run = review_fixture(tmp_path, monkeypatch)
    calls = []; funded_provider(monkeypatch, calls)
    provider, original_step = harness.urllib.request.urlopen, review.run_step
    ctx = mp.get_context('fork')
    entered, release, outside_done, result = ctx.Event(), ctx.Event(), ctx.Event(), ctx.Queue()
    owner = os.getpid()
    workers = []
    def held(request, timeout):
        if (os.getpid() != owner and provider_kind(request) == 'final'
                and 'Interpret recorded field descriptors' not in request.data.decode()):
            entered.set(); assert release.wait(8)
        return provider(request, timeout)
    monkeypatch.setattr(harness.urllib.request, 'urlopen', held)
    def external(environment):
        # A separate terminal does not inherit review's ownership descriptors.
        for name in ('ARGUS_REVIEW_LOCK_FD', 'ARGUS_REVIEW_OUTPUT_LOCK_FD'):
            if name in environment:
                os.close(int(environment[name]))
            os.environ.pop(name, None)
        os.environ.pop('ARGUS_REVIEW_OUTPUT_LOCK_PATH', None)
        try:
            value = harness.run_batch(list((job / 'units').glob('episode_*')), job / 'run' / 'out', **options())
            result.put(value)
        except RuntimeError as error:
            result.put(str(error))
        finally:
            outside_done.set()
    def step(job, kind, command, environment, **kwargs):
        if kind == 'dry_run':
            rc = original_step(job, kind, command, environment, **kwargs)
            worker = ctx.Process(target=external, args=(environment,)); worker.start(); workers.append(worker)
            deadline = time.monotonic() + 5
            while not outside_done.is_set() and not entered.is_set() and time.monotonic() < deadline:
                time.sleep(.01)
            assert outside_done.is_set() or entered.is_set()
            return rc
        if kind == 'label':
            try:
                return original_step(job, kind, command, environment, **kwargs)
            except RuntimeError as error:
                if 'Another label execution' in str(error):
                    return 1  # A refused real harness subprocess exits 1.
                raise
        return original_step(job, kind, command, environment, **kwargs)
    monkeypatch.setattr(review, 'run_step', step)
    try:
        rc = run('--cap', '2')
        assert not entered.is_set(), 'an unrelated harness dispatched while review owned publication'
        assert rc == 0
        assert 'Another label execution' in result.get(timeout=1)
        published = json.loads(next((job / 'run_final' / 'out').glob('episode_*.json')).read_text())
        assert harness.annotation_complete(published)
    finally:
        release.set()
        for worker in workers:
            worker.join(5)
            if worker.is_alive(): worker.terminate(); worker.join(2)


def test_output_lease_borrow_requires_same_inode_and_exclusive_description(tmp_path, monkeypatch):
    ep = recording(tmp_path, np.arange(12) / 30)
    calls = []
    def provider(request, timeout):
        kind = provider_kind(request); calls.append(kind)
        return response('{"requests":[]}' if kind == 'selection' else complete_reply.text('A hand reaches.'))
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    out = tmp_path / 'out'
    wrong = tmp_path / 'unrelated'; wrong.touch()
    with harness.execution_lock(out) as lease:
        for path in (wrong, out / '.execution.lock'):
            with path.open('a') as unowned:
                with pytest.raises(RuntimeError, match='lease|Another label execution'):
                    harness.run_batch([ep], out, execution_lease_fd=unowned.fileno(), **options())
        assert calls == []
        assert harness.run_batch([ep], out, execution_lease_fd=lease, **options()) == 0
        assert calls == ['selection', 'final']
        # Finishing a borrowed call must not unlock the encompassing transaction.
        with pytest.raises(RuntimeError, match='Another label execution'):
            with harness.execution_lock(out):
                pass
