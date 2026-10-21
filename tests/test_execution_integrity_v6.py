"""Offline writer, retry, and invoice boundaries for execution resumes."""
import json
import multiprocessing as mp
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from label import harness, pieces, reparse, run
from label.dictionary_stage import output_spend, output_spend_complete
from test_camera_timing import recording
from test_dispatch_integrity_v5 import options, provider_kind, response
import complete_reply


def episodes(tmp_path, count=3):
    result = []
    for i in range(count):
        parent = tmp_path / str(i); parent.mkdir()
        result.append(recording(parent, np.arange(12) / 30))
        target = parent / f'episode_{i}'
        result[-1].rename(target); result[-1] = target
    return result


@pytest.mark.parametrize('finish', ['stop', 'length'])
def test_explicit_received_retry_pays_only_for_named_unusable_reply(tmp_path, monkeypatch, finish):
    eps = episodes(tmp_path)
    calls = []
    def transport(request, timeout):
        kind = provider_kind(request); calls.append(kind)
        if kind == 'selection':
            return response('{"requests":[]}')
        content = 'not JSON' if calls.count('final') == 2 else complete_reply.text('A hand reaches.')
        result = response(content)
        if content == 'not JSON':
            data = json.loads(result.read()); data['choices'][0]['finish_reason'] = finish
            import io
            return io.BytesIO(json.dumps(data).encode())
        return result
    monkeypatch.setattr(harness.urllib.request, 'urlopen', transport)
    out = tmp_path / 'out'
    assert harness.run_batch(eps, out, **options()) == 1
    healthy = [(out / (ep.name + '.json')).read_bytes() for ep in (eps[0], eps[2])]
    paid = output_spend(out)
    assert harness.run_batch(eps, out, **options()) == 1
    assert calls.count('final') == 3
    retry_options = options(); retry_options['max_tokens'] = 200
    assert harness.run_batch(eps, out, retry_received=[eps[1].name], **retry_options) == 0
    assert calls.count('final') == 4 and calls.count('selection') == 3
    assert [(out / (ep.name + '.json')).read_bytes() for ep in (eps[0], eps[2])] == healthy
    saved = json.loads((out / (eps[1].name + '.json')).read_text())
    assert saved['final_cost_history'][-1]['response_text'] == 'not JSON'
    assert output_spend(out) == pytest.approx(paid + .003)


def test_reparse_excludes_a_harness_writer_before_reading_receipts(tmp_path, monkeypatch):
    ep = recording(tmp_path, np.arange(12) / 30)
    monkeypatch.setattr(harness.urllib.request, 'urlopen', lambda request, timeout:
        response('{"requests":[]}' if provider_kind(request) == 'selection' else complete_reply.text('A hand reaches.')))
    bind = harness.parse_response
    monkeypatch.setattr(harness, 'parse_response', lambda text: ({'_raw': text, '_parse_error': 'interrupted'}, False))
    out = tmp_path / 'out'
    assert harness.run_batch([ep], out, **options()) == 1
    (tmp_path / 'run.json').write_text('{}')
    monkeypatch.setattr(harness, 'parse_response', bind)
    ctx = mp.get_context('fork'); entered, release = ctx.Event(), ctx.Event()
    def held_parse(text):
        entered.set(); assert release.wait(5)
        return bind(text)
    monkeypatch.setattr(harness, 'parse_response', held_parse)
    worker = ctx.Process(target=reparse.reparse, args=(tmp_path,)); worker.start()
    try:
        assert entered.wait(5)
        args = options(); args['force'] = True
        with pytest.raises(RuntimeError, match='Another label execution'):
            harness.run_batch([ep], out, **args)
    finally:
        release.set(); worker.join(5)
        if worker.is_alive(): worker.terminate(); worker.join()
    assert worker.exitcode == 0
    assert output_spend(out) == pytest.approx(.006)


def test_received_retry_uses_newer_truncation_instead_of_older_healthy_primary(tmp_path, monkeypatch):
    import io
    ep = recording(tmp_path, np.arange(12) / 30)
    finals = []
    def transport(request, timeout):
        if provider_kind(request) == 'selection':
            return response('{"requests":[]}')
        finals.append(1)
        result = response(complete_reply.text('A hand reaches.') if len(finals) != 2 else '{"task_summary":')
        if len(finals) == 2:
            data = json.loads(result.read()); data['choices'][0]['finish_reason'] = 'length'
            return io.BytesIO(json.dumps(data).encode())
        return result
    monkeypatch.setattr(harness.urllib.request, 'urlopen', transport)
    out = tmp_path / 'out'
    assert harness.run_batch([ep], out, **options()) == 0
    args = options(); args['force'] = True
    assert harness.run_batch([ep], out, **args) == 1
    monkeypatch.setenv('OPENROUTER_API_KEYS', 'sk-or-fixture')
    monkeypatch.setattr(sys, 'argv', ['label.harness', '--episode-dir', str(ep), '--out',
        str(out / (ep.name + '.json')), '--retry-received', ep.name, '--cell-w', '128',
        '--max-tokens', '100', '--max-spend', '2'])
    assert harness.main() == 0
    assert len(finals) == 3 and output_spend(out) == pytest.approx(.015)
    latest = json.loads((out / (ep.name + '.json')).read_text())
    assert len(latest['final_cost_history']) == 2


def test_received_retry_cannot_repurchase_healthy_output(tmp_path, monkeypatch):
    ep = recording(tmp_path, np.arange(12) / 30)
    settings = dict(model=harness.DEFAULT_MODEL, reasoning='medium', max_tokens=100, cell_w=128, example_dir=None)
    record = {'input_identity': harness.input_identity(ep, **settings), 'parse_ok': True,
        'labels': complete_reply.labels('A hand reaches.'),
        'usage': {'est_cost_usd': .003}, 'final_dispatch_outcome': 'received', 'final_reserved_usd': 1.}
    out = tmp_path / 'out'; out.mkdir(); path = out / (ep.name + '.json')
    path.write_text(json.dumps(record)); original = path.read_bytes()
    monkeypatch.setattr(harness.urllib.request, 'urlopen', lambda *a, **kw: pytest.fail('unapproved paid retry'))
    # a healthy label is kept and nothing is bought; naming it is noted, and the finished job does not fail
    assert harness.run_batch([ep], out, retry_received=[ep.name], **options()) == 0
    assert path.read_bytes() == original
    assert not (out / f'stale_{ep.name}.json').exists()


def test_unknown_charge_past_its_automatic_resends_is_retried_only_when_named_and_stays_counted(
        resends_used_up, tmp_path, monkeypatch):
    # A dispatch whose outcome is unknown may have been billed: once its automatic resends are used up, a plain
    # resume never sends it again, and naming it sends that unit alone while its possible charge stays reserved.
    ep = recording(tmp_path, np.arange(12) / 30)
    settings = dict(model=harness.DEFAULT_MODEL, reasoning='medium', max_tokens=100, cell_w=128, example_dir=None)
    record = {'input_identity': harness.input_identity(ep, **settings), 'parse_ok': False,
        'labels': {'_raw': 'unusable'}, 'final_dispatch_outcome': 'unverified', 'final_reserved_usd': 1.}
    out = tmp_path / 'out'; out.mkdir(); path = out / ('noreply_' + ep.name + '.json')
    path.write_text(json.dumps(record))
    monkeypatch.setattr(harness.urllib.request, 'urlopen', lambda *a, **kw: pytest.fail('unapproved paid retry'))
    assert harness.run_batch([ep], out, **options()) == 1
    marker = json.loads((out / f'stale_{ep.name}.json').read_text())
    assert marker['refresh'] == f'--retry-received {ep.name}'
    calls = []
    def transport(request, timeout):
        calls.append(provider_kind(request))
        return response('{"requests":[]}' if calls[-1] == 'selection' else complete_reply.text('A hand reaches.'))
    monkeypatch.setattr(harness.urllib.request, 'urlopen', transport)
    assert harness.run_batch([ep], out, retry_received=[ep.name], **options()) == 0
    assert calls.count('final') == 1
    saved = json.loads((out / (ep.name + '.json')).read_text())
    assert any(h.get('final_dispatch_outcome') == 'unverified' and h.get('final_reserved_usd') == 1.
               for h in saved['final_cost_history'])
    assert output_spend(out) >= 1.


def test_changed_paid_inputs_are_isolated_without_being_reused(tmp_path, monkeypatch):
    eps = episodes(tmp_path, 2)
    calls = []
    def transport(request, timeout):
        kind = provider_kind(request); calls.append(kind)
        return response('{"requests":[]}' if kind == 'selection' else complete_reply.text('A hand reaches.'))
    monkeypatch.setattr(harness.urllib.request, 'urlopen', transport)
    out = tmp_path / 'out'
    assert harness.run_batch(eps[:1], out, **options()) == 0
    saved = (out / (eps[0].name + '.json')).read_bytes()
    source = json.loads((eps[0] / 'sources.json').read_text())['exo']['packed']
    Path(source).unlink()
    assert harness.run_batch(eps, out, **options()) == 1
    assert calls == ['selection', 'final'] * 2
    assert (out / (eps[0].name + '.json')).read_bytes() == saved
    assert (out / (eps[1].name + '.json')).exists()


@pytest.mark.parametrize('problem', ['missing', 'physical-state'])
def test_missing_never_dispatched_input_does_not_block_untouched_episodes(tmp_path, monkeypatch, problem):
    eps = episodes(tmp_path)
    if problem == 'missing':
        source = json.loads((eps[0] / 'sources.json').read_text())['exo']['packed']
        Path(source).unlink()
    else:
        path = eps[0] / 'context.json'; context = json.loads(path.read_text())
        context.update(state_kind='joints', source={'format': 'lerobot v3.0'})
        path.write_text(json.dumps(context))
        np.savez(eps[0] / 'state.npz', state=np.zeros((12, 7)))
    calls = []
    monkeypatch.setattr(harness.urllib.request, 'urlopen', lambda request, timeout:
        (calls.append(provider_kind(request)) or response('{"requests":[]}' if provider_kind(request) == 'selection'
                                                        else complete_reply.text('A hand reaches.'))))
    out = tmp_path / 'out'
    initial = eps[:2] if problem == 'physical-state' else eps[:1]
    assert harness.run_batch(initial, out, **options()) == 1
    claim = (out / ('noreply_' + eps[0].name + '.json')).read_bytes()
    assert harness.run_batch(eps, out, **options()) == 1
    assert calls == ['selection', 'final'] * 2
    assert (out / ('noreply_' + eps[0].name + '.json')).read_bytes() == claim
    assert all((out / (ep.name + '.json')).exists() for ep in eps[1:])


@pytest.mark.parametrize('mode', ['complete', 'mixed', 'unlabelled'])
def test_stitched_invoices_keep_current_history_and_route_precision(tmp_path, monkeypatch, mode):
    from test_pieces import _three_parts
    from compare.metrics import read_output
    job, ep, parts = _three_parts(tmp_path, monkeypatch)
    from label.evidence_access import source_proof
    proof = source_proof(json.loads((ep / 'context.json').read_text()), ep)
    (job / 'pieces' / f'.{ep.name}.source_proof.json').write_text(json.dumps(proof))
    out = job / 'run' / 'out'
    calls = []
    def transport(request, timeout):
        kind = provider_kind(request); calls.append(kind)
        return response('{"requests":[]}' if kind == 'selection' else complete_reply.text('A hand reaches.'), .0000123)
    monkeypatch.setattr(harness.urllib.request, 'urlopen', transport)
    assert harness.run_batch(parts, out, **options()) == 0
    args = options(); args['force'] = True
    assert harness.run_batch(parts, out, **args) == 0
    for i, part in enumerate(parts):
        path = out / (part.name + '.json'); record = json.loads(path.read_text())
        record['config']['resolution_route'] = {'cost_usd': .0000012, 'previously_billed': True}
        record.setdefault('auxiliary_cost_history', []).append({'kind': 'route', 'cost_usd': .0000012})
        if mode == 'unlabelled' or mode == 'mixed' and i == 1:
            record.update(parse_ok=False, labels={'_raw': 'unusable', '_parse_error': 'fixture'})
        path.write_text(json.dumps(record))
    raw = output_spend(out)
    final = tmp_path / 'final'
    pieces.stitch_run(job, ep.parent, {ep.name: [p.name for p in parts]}, final)
    path = final / (ep.name + '.json'); record = json.loads(path.read_text())
    assert record['usage']['est_cost_usd'] == pytest.approx(.0000123 * 3)
    assert harness.episode_cost(record) == pytest.approx(raw)
    assert output_spend(final) == pytest.approx(raw) and output_spend_complete(final)
    assert read_output(path)['cost'] == pytest.approx(raw)


def test_resume_typo_creates_no_folder(tmp_path):
    missing = tmp_path / 'typo'
    args = SimpleNamespace(resume=missing)
    with pytest.raises(SystemExit, match='run.json'):
        run.resume(args, tmp_path)
    assert not missing.exists()


def test_compare_cost_summary_preserves_small_invoices():
    from compare import metrics
    rows = {'episode': {'seconds': 1, 'by': {'model': {
        'status': 'parsed', 'violations': [], 'cost': .0000123}}}}
    assert metrics._responses('model', ['episode'], rows)['cost'] == .0000123


def test_resume_refuses_an_orphan_harness_without_mutating_run_metadata(tmp_path, monkeypatch):
    from test_label import _killed_run
    sl, folder = _killed_run(tmp_path)
    monkeypatch.setattr(run, 'commit', lambda: ('abc1234', False))
    args = SimpleNamespace(resume=folder, dataset='d', kind='smoke', why='parent died', concurrency=1, harness_args=[])
    before = (folder / 'run.json').read_bytes()
    ctx = mp.get_context('fork'); entered, release = ctx.Event(), ctx.Event()
    def active():
        with harness.execution_lock(folder / 'out'):
            entered.set(); assert release.wait(5)
    worker = ctx.Process(target=active); worker.start()
    try:
        assert entered.wait(5)
        monkeypatch.setattr(run, 'finish', lambda *args: pytest.fail('orphan already owns output execution'))
        with pytest.raises(RuntimeError, match='Another label execution'):
            run.resume(args, sl)
        assert (folder / 'run.json').read_bytes() == before
    finally:
        release.set(); worker.join(5)


def test_refused_child_exit_does_not_replace_an_active_harness_status(tmp_path, monkeypatch):
    from test_label import _killed_run
    sl, folder = _killed_run(tmp_path)
    prepared = tmp_path / 'prepared'; prepared.mkdir()
    recording(prepared, np.arange(12) / 30)
    before = (folder / 'run.json').read_bytes()
    info = json.loads(before)
    monkeypatch.setenv('OPENROUTER_API_KEYS', 'sk-or-fixture')
    ctx = mp.get_context('fork'); entered, release = ctx.Event(), ctx.Event()
    def active():
        with harness.execution_lock(folder / 'out'):
            entered.set(); assert release.wait(5)
    worker = ctx.Process(target=active); worker.start()
    try:
        assert entered.wait(5)
        command = [sys.executable, '-m', 'label.harness', '--episodes-root', str(prepared),
                   '--out-dir', str(folder / 'out')]
        with pytest.raises(RuntimeError, match='Another label execution'):
            run.finish(folder, info, command, sl, 'a')
        assert (folder / 'run.json').read_bytes() == before
        assert 'Another label execution' in (folder / 'log.txt').read_text()
    finally:
        release.set(); worker.join(5)


@pytest.mark.parametrize('dependency,changes', [('board/serve.py', False), ('board/build.py', False),
    ('board/to_board.py', False), ('board/clips.py', False), ('board/hands.py', False),
    ('label/episode.py', True), ('board/public_export.py', True),
    ('prepare/display.py', True), ('prepare/mcap_fields.py', True), ('prepare/hdf_metadata.py', True),
    ('label/evidence_access.py', True)])
def test_request_dependency_boundary_ignores_unrelated_board_code(tmp_path, monkeypatch, dependency, changes):
    ep = recording(tmp_path, np.arange(12) / 30)
    settings = dict(model=harness.DEFAULT_MODEL, reasoning='medium', max_tokens=100, cell_w=128, example_dir=None)
    before = harness.input_identity(ep, **settings)
    original = Path.read_bytes
    target = Path(harness.__file__).parent.parent / dependency
    monkeypatch.setattr(Path, 'read_bytes', lambda path: original(path) + (b'\n# changed\n' if path == target else b''))
    after = harness.input_identity(ep, **settings)
    assert harness.same_input_identity(before, after) is not changes


def test_import_walk_reuses_parsed_edges_but_reads_changed_imports(tmp_path, monkeypatch):
    import ast
    import os
    harness._request_imports.cache_clear()
    files = {'label/__init__.py': '', 'label/harness.py': 'from prepare import evidence\n',
             'prepare/__init__.py': '', 'prepare/evidence.py': 'from . import base\n',
             'prepare/base.py': 'VALUE = 1\n', 'prepare/next.py': 'VALUE = 2\n'}
    for name, text in files.items():
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)
    parsed = []
    parse = ast.parse
    def count_parse(*args, **kwargs):
        parsed.append(args[0])
        return parse(*args, **kwargs)
    monkeypatch.setattr(ast, 'parse', count_parse)
    original = harness.request_implementation(tmp_path)
    first_count = len(parsed)
    assert first_count > 0
    assert harness.request_implementation(tmp_path) == original
    assert len(parsed) == first_count
    source = tmp_path / 'prepare/evidence.py'; before = source.stat()
    source.write_text('from . import next\n')
    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert source.stat().st_size == before.st_size and source.stat().st_mtime_ns == before.st_mtime_ns
    current = harness.request_implementation(tmp_path)
    assert tmp_path / 'prepare/next.py' in current and tmp_path / 'prepare/base.py' not in current
    assert current[source] == b'from . import next\n'
    assert len(parsed) == first_count + 2
    source.unlink()
    package = source.with_suffix(''); package.mkdir()
    (package / '__init__.py').write_bytes(b'from . import next\n')
    (package / 'next.py').write_bytes(b'VALUE = 2\n')
    packaged = harness.request_implementation(tmp_path)
    assert package / 'next.py' in packaged and tmp_path / 'prepare/next.py' not in packaged
