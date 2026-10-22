"""Reviewing your own data (python -m review) and the pieces of it Data Review shares: the neighbour lag measured
inside a folder, the manifest entry for your own data, what a stitched recording carries onto the board, and the
outcome and severity values the board knows."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from board import build as board_build
from board import rules
from checks import timebase
from test_formats import recorder_folder

REPO = Path(__file__).resolve().parent.parent


def test_review_refuses_a_missing_data_path_in_one_line_and_creates_no_job(tmp_path, monkeypatch, capsys):
    """A mistyped --data path made a job folder and then stopped with a traceback ending in "the upload holds no ...
    (it is empty)". It is now refused with one line naming the path, before anything is written."""
    from review import __main__ as rv
    missing, job = tmp_path / 'no_such_upload', tmp_path / 'job'
    monkeypatch.setattr(sys, 'argv', ['review', '--data', str(missing), '--rig', 'teleop_arms', '--out', str(job),
                                      '--free'])
    with pytest.raises(SystemExit) as stop:
        rv.main()
    assert stop.value.code == f'--data {missing} does not exist'
    assert not job.exists()


@pytest.mark.parametrize('kind', ['video', 'mcap', 'hdf5'])
def test_review_direct_file_retains_owned_metadata_without_reading_neighbor_recordings(tmp_path, monkeypatch, kind):
    import hashlib
    import h5py
    from review import __main__ as rv
    from label import episode, harness
    from test_formats import _camera_mcap, _clip

    upload = tmp_path / 'collection'
    upload.mkdir()
    suffix = {'video': '.mp4', 'mcap': '.mcap', 'hdf5': '.h5'}[kind]
    source = upload / ('selected_clip' + suffix)
    clock = None
    if kind == 'video':
        _clip(source, 12)
        clock = upload / 'selected_clip-timestamps.npy'
        recorded_times = np.r_[np.arange(6) / 30, .2 + np.arange(6) / 3]
        np.save(clock, recorded_times)
        np.save(upload / 'other_clip-timestamps.npy', np.arange(18) / 15)
        corrupt_clock = upload / 'selected_clip-timestamp-corrupt.npy'
        corrupt_clock.write_bytes(b'Unusable publisher clock data.')
        (upload / 'selected_clip-timestamp-directory.npy').mkdir()
    elif kind == 'mcap':
        _camera_mcap(source, ['/top/image/compressed'], n=12)
    else:
        with h5py.File(source, 'w') as container:
            container.attrs['fps'] = 10
            container.attrs['recording_serial'] = 'selected-native-serial'
            container.create_dataset('images/top', data=np.full((12, 36, 64, 3), 70, np.uint8))
    _clip(upload / 'other_clip.mp4', 18)
    (upload / 'other_clip.txt').write_text('Handle the unrelated object.')
    (upload / 'task.txt').write_text('An ambiguous task for the surrounding collection.')
    own_note = upload / 'selected_clip.txt'
    own_note.write_text('Lift the selected object.')
    scope = upload / 'dataset.json'
    scope.write_text(json.dumps({'task_mode': 'session', 'collection_note': 'Recorded source collection.'}))
    before = {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
              for p in upload.iterdir() if p.is_file()}

    class ReachedChecks(Exception):
        pass
    monkeypatch.setattr(harness, 'call_model_once', lambda *a, **k: pytest.fail('unexpected provider call'))
    monkeypatch.setattr(rv, 'run_step', lambda *a, **k: (_ for _ in ()).throw(ReachedChecks()))
    job = tmp_path / 'job'
    monkeypatch.setattr(sys, 'argv', ['review', '--data', str(source), '--rig', 'ego_head', '--out', str(job), '--free'])
    with pytest.raises(ReachedChecks):
        rv.main()
    report = json.loads((job / 'report.json').read_text())
    assert len(report['episodes']) == 1 and not report['failed'] and not report['skipped']
    path = next((job / 'episodes').glob('episode_*'))
    context = json.loads((path / 'context.json').read_text())
    assert context['instruction'] == 'Lift the selected object.'
    assert context['task_mode'] == 'session' and 'Recorded source collection.' in context['collection_note']
    assert 'other_clip' not in json.dumps(context) and 'ambiguous task' not in json.dumps(context)
    receipt = json.loads((job / 'preparation_receipt.json').read_text())
    assert set(receipt['source']) == {source.name, own_note.name, scope.name} | (
        {clock.name, corrupt_clock.name} if clock else set())
    if kind == 'video':
        native = json.loads((path / 'sources.json').read_text())
        assert {view['packed'] for view in native.values()} == {str(source.resolve())}
    elif kind == 'hdf5':
        assert context['recorded_hdf5_metadata']['source_path'] == str(source.resolve())
        assert 'selected-native-serial' in json.dumps(context)
    loaded = episode.load(path)
    if clock:
        np.testing.assert_allclose(loaded['times'][episode.anchor(loaded)], recorded_times)
    selected = episode.plan(loaded)['ks']
    assert episode._decode_view(loaded, episode.anchor(loaded), selected)
    assert before == {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
                      for p in upload.iterdir() if p.is_file()}
    saved = (job / 'preparation_receipt.json').read_bytes()
    with pytest.raises(ReachedChecks):
        rv.main()
    assert (job / 'preparation_receipt.json').read_bytes() == saved
    # A changed unrelated recording note does not invalidate this file's preparation.
    (upload / 'other_clip.txt').write_text('Changed unrelated instructions.')
    with pytest.raises(ReachedChecks):
        rv.main()
    if clock:
        np.save(clock, recorded_times * 2)
        with pytest.raises(SystemExit, match='fresh job'):
            rv.main()
        np.save(clock, recorded_times)
        np.save(corrupt_clock, recorded_times * 2)
        with pytest.raises(SystemExit, match='fresh job'):
            rv.main()
        corrupt_clock.write_bytes(b'Unusable publisher clock data.')
    own_note.write_text('Changed selected instructions.')
    with pytest.raises(SystemExit, match='fresh job'):
        rv.main()
    own_note.write_text('Lift the selected object.')
    scope.write_text(json.dumps({'task_mode': 'session', 'collection_note': 'Changed recorded scope.'}))
    with pytest.raises(SystemExit, match='fresh job'):
        rv.main()


def test_review_keeps_mixed_nested_archives_immutable_and_reuses_preparation(tmp_path, monkeypatch):
    import hashlib
    import io
    import zipfile
    import h5py
    from review import __main__ as rv
    from label import harness

    upload = tmp_path / 'collection'
    upload.mkdir()
    raw = io.BytesIO()
    with h5py.File(raw, 'w') as source:
        source.attrs['fps'] = 10
        source.create_dataset('images/top', data=np.full((12, 36, 64, 3), 60, np.uint8))
    (upload / 'loose.h5').write_bytes(raw.getvalue())
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, 'w') as zipped:
        zipped.writestr('take/recorded.h5', raw.getvalue())
    with zipfile.ZipFile(upload / 'packed.zip', 'w') as zipped:
        zipped.writestr('bundle/episode.zip', inner.getvalue())
    publisher_partial = upload / 'bundle' / 'take' / 'recorded.h5.part'
    publisher_partial.parent.mkdir(parents=True)
    publisher_partial.write_bytes(b'Publisher-owned incomplete recording, preserved as supplied.')
    (upload / 'dataset.json').write_text(json.dumps({'task_mode': 'session', 'collection_note': 'One intact recording per take.'}))

    def source_files():
        return {p.relative_to(upload).as_posix(): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
                for p in upload.rglob('*') if p.is_file()}

    before = source_files()
    job = tmp_path / 'job'
    class ReachedChecks(Exception):
        pass
    monkeypatch.setattr(harness, 'call_model_once', lambda *a, **k: pytest.fail('unexpected provider call'))
    monkeypatch.setattr(rv, 'run_step', lambda *a, **k: (_ for _ in ()).throw(ReachedChecks()))
    monkeypatch.setattr(sys, 'argv', ['review', '--data', str(upload), '--rig', 'ego_head', '--out', str(job), '--free'])
    with pytest.raises(ReachedChecks):
        rv.main()
    assert source_files() == before
    receipt = (job / 'preparation_receipt.json').read_bytes()
    report = json.loads((job / 'report.json').read_text())
    assert len(report['episodes']) == 2 and not report['failed']
    contexts = [json.loads(p.read_text()) for p in (job / 'episodes').glob('*/context.json')]
    assert all(ctx['task_mode'] == 'session' for ctx in contexts)
    assert all('One intact recording per take.' in ctx['collection_note'] for ctx in contexts)
    with pytest.raises(ReachedChecks):
        rv.main()
    assert source_files() == before
    assert (job / 'preparation_receipt.json').read_bytes() == receipt


# the last case: an upload with a field the dictionary would read makes no call when the flag is left at its default
@pytest.mark.parametrize('extra,cap,flag,expected', [(False, 20.0, ['--field-dictionary', 'on'], 'skipped'),
                                                       (True, 20.0, ['--field-dictionary', 'on'], 'success'),
                                                       (True, .001, ['--field-dictionary', 'on'], 'skipped'),
                                                       (True, 20.0, [], 'skipped')])
def test_review_cli_raw_hdf_dictionary_gate(tmp_path, monkeypatch, extra, cap, flag, expected):
    import h5py
    from review import __main__ as review_main
    from label import dictionary, harness
    upload = tmp_path / 'upload'
    upload.mkdir()
    with h5py.File(upload / 'recording.h5', 'w') as source:
        source.attrs['fps'] = 10
        source.create_dataset('timestamps', data=np.int64(1790000000000000000)
                              + np.arange(12, dtype=np.int64) * 100000000)
        camera = source.create_dataset('images/top', data=np.full((12, 36, 64, 3), 60, np.uint8))
        if extra:
            camera.attrs['calibration'] = 'fixture calibration'
        from test_prepare import joint_names
        source.create_dataset('observations/qpos', data=np.zeros((12, 14))).attrs['names'] = joint_names()
    job = tmp_path / 'job'
    calls = []
    def provider(content, model, reasoning, api_key, max_tokens, timeout):
        paths = list((job / 'episodes').glob('episode_*'))
        plan = dictionary.request_plan(dictionary.inventory(paths))
        assert content == plan['content'] and max_tokens == plan['max_tokens']
        calls.append(1)
        ident = plan['field_ids'][0]
        return {'choices': [{'message': {'content': json.dumps({'entries': [
            {'id': ident, 'meaning': 'Reviewed recording descriptor', 'role': 'annotation'}]})}}],
                'usage': {'cost': .01}}
    class ReachedChecks(Exception):
        pass
    monkeypatch.setattr(harness, 'get_keys', lambda: ['sk-or-fixture'])
    monkeypatch.setattr(harness, 'call_model_once', provider)
    monkeypatch.setattr(review_main, 'run_step', lambda *a, **k: (_ for _ in ()).throw(ReachedChecks()))
    monkeypatch.setattr(sys, 'argv', ['review', '--data', str(upload), '--rig', 'teleop_arms',
                                      '--out', str(job), '--cap', str(cap), *flag])
    with pytest.raises(ReachedChecks):
        review_main.main()
    receipt = json.loads((job / 'dictionary_status.json').read_text())
    assert receipt['status'] == expected
    assert len(calls) == int(expected == 'success')
    assert (job / 'dictionary.claim').exists() is (expected == 'success')
    if expected == 'success':
        ctx = json.loads(next((job / 'episodes').glob('episode_*/context.json')).read_text())
        assert ctx['data_dictionary']['entries']


@pytest.mark.parametrize('status', [401, 402])
def test_review_native_dictionary_retires_only_verified_unpaid_key(tmp_path, monkeypatch, status):
    import io
    import hashlib
    import h5py
    import urllib.error
    from review import __main__ as rv
    from label import dictionary, dictionary_stage, harness
    from test_prepare import joint_names

    upload = tmp_path / 'upload'
    upload.mkdir()
    raw = upload / 'recording.h5'
    recorded_times = np.int64(1790000000000000000) + np.arange(12, dtype=np.int64) * 100000000
    with h5py.File(raw, 'w') as source:
        source.attrs['fps'] = 10
        source.create_dataset('timestamps', data=recorded_times)
        image = source.create_dataset('images/top', data=np.full((12, 36, 64, 3), 60, np.uint8))
        image.attrs['calibration'] = 'fixture camera calibration'
        source.create_dataset('observations/qpos', data=np.zeros((12, 14))).attrs['names'] = joint_names()
    raw_hash = hashlib.sha256(raw.read_bytes()).hexdigest()
    job = tmp_path / 'job'
    requests = []
    def transport(req, timeout):
        body = json.loads(req.data)
        key = req.get_header('Authorization').split()[-1]
        requests.append((key, body))
        assert body['model'] == 'openai/gpt-6-sol'
        if key == 'sk-or-empty':
            raise urllib.error.HTTPError(req.full_url, status, 'Rejected', {}, io.BytesIO(b'{}'))
        assert key == 'sk-or-funded'
        fields = dictionary.inventory(list((job / 'episodes').glob('episode_*')))['fields']
        native_state = next(field for field in fields if field['kind'] == 'state' and field['name'] == 'observations/qpos')
        answer = {'entries': [{'id': native_state['id'], 'meaning': 'Recorded joint positions from observations/qpos',
                               'role': 'joint_state'}]}
        return io.BytesIO(json.dumps({'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps(answer)}}], 'usage': {'cost': .01}}).encode())
    class ReachedChecks(Exception):
        pass
    monkeypatch.setenv('OPENROUTER_API_KEYS', 'sk-or-empty,sk-or-funded')
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    monkeypatch.setattr(harness.urllib.request, 'urlopen', transport)
    monkeypatch.setattr(rv, 'run_step', lambda *a, **k: (_ for _ in ()).throw(ReachedChecks()))
    monkeypatch.setattr(sys, 'argv', ['review', '--data', str(upload), '--rig', 'teleop_arms',
                                     '--out', str(job), '--cap', '20', '--field-dictionary', 'on'])
    with pytest.raises(ReachedChecks):
        rv.main()
    saved = json.loads((job / 'dictionary.json').read_text())
    assert saved['status'] == 'success' and saved['cost_usd'] == .01
    assert len(requests) == 2 and requests[0][1] == requests[1][1]
    assert [a['state'] for a in saved['dispatch_attempts']] == ['rejected unpaid', 'received']
    rejected = saved['dispatch_attempts'][0]
    assert rejected['attempt'] == 1 and rejected['http_status'] == status
    assert rejected['http_attempt'] == 1 and rejected['verified_unpaid'] is True and rejected['cost_usd'] == 0
    assert saved['dispatch_attempts'][1]['attempt'] == 2 and saved['dispatch_attempts'][1]['cost_usd'] == .01
    assert saved['request_digest']
    assert 'sk-or-empty' not in json.dumps(saved) and 'sk-or-funded' not in json.dumps(saved)
    assert dictionary_stage.dictionary_spend(job) == {'cost_usd': .01, 'reserved_usd': .01, 'complete': True}
    contexts = list((job / 'episodes').glob('*/context.json'))
    assert len(contexts) == 1
    context = json.loads(contexts[0].read_text())
    assert context['data_dictionary']['status'] == 'success'
    native_state = next(field for field in context['data_dictionary']['fields'] if field['kind'] == 'state')
    entry = context['data_dictionary']['entries'][native_state['id']]
    assert entry['meaning'] == 'Recorded joint positions from observations/qpos' and entry['role'] == 'joint_state'
    assert entry['provenance'] == 'machine' and native_state['episodes'] == [contexts[0].parent.name]
    assert context['real_times'] == 'times.npz'
    with np.load(contexts[0].parent / 'times.npz') as clocks:
        np.testing.assert_allclose(clocks['exo'], np.arange(12) / 10, atol=1e-12)
    with np.load(contexts[0].parent / context['recorded_container_times']['file']) as clocks:
        np.testing.assert_array_equal(clocks['timestamps'], recorded_times)
    assert any(field.get('object') == 'images/top' and field.get('attribute') == 'calibration'
               for field in context['recorded_hdf5_metadata']['fields'])
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == raw_hash
    receipt_before = (job / 'dictionary.json').read_bytes()
    with pytest.raises(ReachedChecks):
        rv.main()
    assert (job / 'dictionary.json').read_bytes() == receipt_before
    assert len(requests) == 2


@pytest.mark.parametrize('change', [None, 'source_in_job', 'source', 'resource', 'settings', 'preparation_settings',
                                     'producer', 'prepared', 'report', 'legacy', 'copied', 'same_stat_source'])
def test_review_reuses_real_preparation_and_refuses_unverified_paid_resume(tmp_path, monkeypatch, change):
    import h5py
    import hashlib
    import os
    from review import __main__ as rv
    from prepare import formats
    from label import dictionary_stage, evidence_access, harness
    # every file is identified as a real take's large videos are, whatever size a content hash once stopped at
    monkeypatch.setattr(evidence_access, 'SOURCE_HASH_LIMIT', 0, raising=False)
    upload = tmp_path / 'job' / 'upload' if change == 'source_in_job' else tmp_path / 'upload'
    upload.mkdir(parents=True)
    raw = upload / 'recording.h5'
    with h5py.File(raw, 'w') as source:
        source.attrs['fps'] = 10
        source.create_dataset('images/top', data=np.full((12, 36, 64, 3), 60, np.uint8))
    # A job inside its input must not become part of its own source identity.
    job = tmp_path / 'job' if change == 'source_in_job' else upload / 'review-job'
    conversions, dictionaries = [], []
    real_convert = formats.convert
    def convert(*args, **kwargs):
        conversions.append(1)
        return real_convert(*args, **kwargs)
    def dictionary(job, *args, **kwargs):
        dictionaries.append(1)
        # Dictionary overlays are valid derived state even after an interrupted run.
        context = next((job / 'episodes').glob('episode_*/context.json'))
        doc = json.loads(context.read_text())
        doc['data_dictionary'] = {'status': 'skipped', 'entries': {}}
        context.write_text(json.dumps(doc))
    class ReachedLabel(Exception):
        pass
    commands = []
    def reach_label(job, step, command, *args, **kwargs):
        if step in ('dry_run', 'label'):
            commands.append((step, command))
        if step == 'label':
            raise ReachedLabel()
        return 0
    monkeypatch.setattr(formats, 'convert', convert)
    monkeypatch.setattr(dictionary_stage, 'prepare', dictionary)
    monkeypatch.setattr(harness, 'call_model', lambda *a, **k: pytest.fail('unexpected provider call'))
    monkeypatch.setattr(rv, 'run_step', reach_label)
    timeout = [] if change == 'source_in_job' else ['--timeout', '900']
    argv = ['review', '--data', str(upload), '--rig', 'ego_head', '--out', str(job), *timeout]
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(ReachedLabel):
        rv.main()
    assert [step for step, _ in commands] == ['dry_run', 'label']
    for _, command in commands:
        assert command[:3] == [rv.PY, '-m', 'label.harness']
        if timeout:
            assert command[command.index('--timeout') + 1] == '900'
        else:
            assert '--timeout' not in command
    assert json.loads((job / 'preparation_receipt.json').read_text())['source']
    ep = next((job / 'episodes').glob('episode_*'))
    generated = next(ep.glob('*.mp4'))
    generated_proof = (generated.stat().st_mtime_ns, hashlib.sha256(generated.read_bytes()).hexdigest())
    out = job / 'run' / 'out'
    assert (out / '.execution.lock').is_file()
    out.mkdir(parents=True, exist_ok=True)
    invoice = out / f'{ep.name}.json'
    invoice.write_text(json.dumps({'usage': {'est_cost_usd': .125, 'cost_source': 'billed'},
                                   'generation_id': 'historical-generation', 'parse_ok': True}))
    original_invoice = invoice.read_bytes()
    raw_before = raw.read_bytes()
    assert dictionary_stage.label_spend(job) == pytest.approx(.125)
    if change == 'source':
        with h5py.File(raw, 'r+') as source:
            source['images/top'][0, 0, 0, 0] = 61
    elif change == 'resource':
        monkeypatch.setattr(sys, 'argv', ['review', '--data', 'https://example.org/other.h5',
                                          '--rig', 'ego_head', '--out', str(job)])
        monkeypatch.setattr(rv, 'fetch', lambda *a, **k: pytest.fail('changed resource was fetched'))
    elif change == 'settings':
        monkeypatch.setattr(sys, 'argv', argv + ['--dataset-note', 'Changed collection meaning'])
    elif change == 'preparation_settings':
        monkeypatch.setattr(sys, 'argv', argv + ['--max-minutes', '1'])
    elif change == 'producer':
        monkeypatch.setattr(rv, 'preparation_code_identity', lambda: {'changed': 'producer'})
    elif change == 'prepared':
        generated.write_bytes(generated.read_bytes() + b'changed')
    elif change == 'report':
        (job / 'report.json').write_text('{"episodes": []}')
    elif change == 'legacy':
        (job / 'preparation_receipt.json').unlink()
    elif change == 'copied':
        # a restarted container rebuilds the take and restores the job: the same bytes, new files and new times
        for f in sorted([*(p for p in upload.rglob('*') if not p.is_relative_to(job)), *(job / 'episodes').rglob('*')]):
            if f.is_file():
                copy = f.with_name(f.name + '.copy')
                shutil.copyfile(f, copy)
                os.replace(copy, f)
                os.utime(f, ns=(f.stat().st_mtime_ns + 10**10,) * 2)
    elif change == 'same_stat_source':
        # one changed byte under the size and modification time the receipt saw
        seen = raw.stat()
        with h5py.File(raw, 'r+') as source:
            source['images/top'][0, 0, 0, 0] = 61
        os.utime(raw, ns=(seen.st_atime_ns, seen.st_mtime_ns))
        assert (raw.stat().st_size, raw.stat().st_mtime_ns) == (seen.st_size, seen.st_mtime_ns)
    if change in (None, 'source_in_job', 'producer', 'copied'):
        with pytest.raises(ReachedLabel):
            rv.main()
        assert dictionaries == [1, 1]
    else:
        with pytest.raises(SystemExit, match='fresh.*job|new.*--out'):
            rv.main()
        assert dictionaries == [1]
    assert conversions == [1]
    assert invoice.read_bytes() == original_invoice
    assert dictionary_stage.label_spend(job) == pytest.approx(.125)
    if change not in ('source', 'same_stat_source'):
        assert raw.read_bytes() == raw_before
    if change not in ('prepared', 'copied'):
        assert (generated.stat().st_mtime_ns, hashlib.sha256(generated.read_bytes()).hexdigest()) == generated_proof
    if change == 'copied':
        # the copy passed on its content, and the receipt now holds the new times
        assert hashlib.sha256(generated.read_bytes()).hexdigest() == generated_proof[1]
        receipt = json.loads((job / 'preparation_receipt.json').read_text())
        assert receipt['source'][raw.name]['mtime_ns'] == raw.stat().st_mtime_ns


def test_unknown_outcomes_and_severities_are_shown_as_unclear():
    d = {"completion": {"task_completed": "mostly_done"}, "_meta": {"task_completed": "success"},
         "data_issues": [{"severity": "critical", "issue": "x"}, {"severity": "low", "issue": "<b>"}],
         "tasks": [{"outcome": "whatever"}], "operator_mistakes": [{"severity": None}]}
    out = board_build.normalize_enums(d)
    assert out["completion"]["task_completed"] == "unclear" and out["_meta"]["task_completed"] == "success"
    assert [i["severity"] for i in out["data_issues"]] == ["unclear", "low"] and out["data_issues"][1]["issue"] == "<b>"
    assert out["tasks"][0]["outcome"] == "whatever" and out["operator_mistakes"][0]["severity"] is None
    known = {"completion": {"task_completed": "success_then_undone"}, "data_issues": [{"severity": "high"}]}
    assert board_build.normalize_enums(known) == known


def test_a_stitched_recording_carries_its_parts_and_the_issues_set_aside_at_our_cuts():
    cut = {"category": "truncated_episode", "issue": "ends mid-task", "excluded_by": "piece_cut", "reason": "our cut"}
    d = {"_excluded": [{"category": "missing_instruction", "issue": "no text", "excluded_by": "no_task_text"}],
         "dataset_checks": {"timebase": {"sped_up_recording": False}}}
    r = {"stitched": {"parts": 2, "cuts_s": [300.0]}, "labels": {"_excluded": [cut]}}
    board_build.carry_pieces(d, r, {"timebase_neighbours_in_upload": 4})
    assert d["_stitched"] == {"parts": 2, "cuts_s": [300.0]}
    assert [x["excluded_by"] for x in d["_excluded"]] == ["no_task_text", "piece_cut"]
    assert d["dataset_checks"]["timebase"]["neighbours_in_upload"] == 4
    plain = {"data_issues": []}
    board_build.carry_pieces(plain, {"labels": {}}, {})
    assert plain == {"data_issues": []}                               # a label that was not stitched is unchanged


def test_own_data_entry_adds_fixed_window_only_for_fixed_length_files():
    e = rules.own_data_entry("mine", "run", "eps", "teleop_arms")
    assert e["rules"] == rules.rules_for("teleop_arms")
    w = rules.own_data_entry("mine", "run", "eps", "ego_head", {"fixed_window_s": 180.0})
    assert w["rules"][:-1] == rules.rules_for("ego_head") and w["rules"][-1]["kind"] == "fixed_window"
    assert w["rules"][-1]["window_s"] == 180.0


def _episode(root: Path, idx: int, lag: int, task: str = "stack", adapter: bool = False) -> None:
    d = root / f"episode_{idx:06d}"
    d.mkdir(parents=True)
    rng = np.random.default_rng(idx)
    a = np.cumsum(rng.normal(0, 0.01, (400, 14)), axis=0)
    s = np.roll(a, lag, axis=0)                      # the follower trails the leader by `lag` frames
    np.savez(d / "state.npz", state=s, action=a)
    ctx = {"profile": "teleop_arms", "state_kind": "joints", "task_label": [task]}
    from prepare.state_contract import declared_action_semantics
    ctx['action_semantics'] = declared_action_semantics('controlled leader fixture', 'command')
    if adapter:                                      # a dataset's own adapter keeps the index in its source
        ctx["source"] = {"folder": "upload", "episode_index": idx, "adapter": "galaxea"}
    else:
        ctx["episode_index"] = idx
    (d / "context.json").write_text(json.dumps(ctx))


def test_the_neighbour_lag_is_measured_inside_the_folder(tmp_path):
    for i in range(5):
        _episode(tmp_path, i, lag=3)
    _episode(tmp_path, 40, lag=3)                    # alone: measured, but no neighbour lag
    assert timebase.measure_folder(tmp_path) == 5
    ctx = lambda i: json.loads((tmp_path / f"episode_{i:06d}" / "context.json").read_text())
    assert abs(ctx(2)["timebase_neighbour_lag_frames"] - 3.0) < 0.2 and ctx(2)["timebase_neighbours_in_upload"] == 5
    assert "timebase_neighbour_lag_frames" not in ctx(40) and ctx(40)["timebase_neighbours_in_upload"] == 1


def test_the_neighbour_lag_is_measured_when_an_adapter_keeps_the_index_in_its_source(tmp_path):
    for i in range(4):
        _episode(tmp_path, i, lag=3, adapter=True)
    assert timebase.measure_folder(tmp_path) == 4
    ctx = json.loads((tmp_path / "episode_000001" / "context.json").read_text())
    assert abs(ctx["timebase_neighbour_lag_frames"] - 3.0) < 0.2 and ctx["timebase_neighbours_in_upload"] == 4


def test_review_runs_every_stage_on_a_recorders_folder_in_free_mode():
    """python -m review on a capture-stack folder reads it with its arm state, runs the checks and the clips, and
    builds every request without calling the model."""
    with tempfile.TemporaryDirectory() as t:
        root = Path(t) / "upload"
        recorder_folder(root, n=90, duplicate_stamps=False)
        job = Path(t) / "job"
        p = subprocess.run([sys.executable, "-m", "review", "--data", str(root), "--rig", "teleop_arms", "--out",
                            str(job), "--dataset", "mine", "--free"], cwd=REPO, capture_output=True, text=True)
        assert p.returncode == 0, p.stdout + p.stderr
        rep = json.loads((job / "report.json").read_text())
        assert len(rep["episodes"]) == 1 and rep["episodes"][0]["state_kind"] == "joints"
        dry = list((job / "dry").glob("episode_*.json"))
        assert len(dry) == 1 and json.loads(dry[0].read_text())["dry_run"] is True
        assert list((job / "clips").rglob("*.mp4"))
        media = {f: (f.stat().st_size, f.stat().st_mtime_ns) for f in (job / 'episodes').rglob('*.mp4')}
        sources = {f: f.read_bytes() for f in (job / 'episodes').rglob('sources.json')}
        repeated = subprocess.run([sys.executable, '-m', 'review', '--data', str(root), '--rig', 'teleop_arms',
                                   '--out', str(job), '--dataset', 'mine', '--free'], cwd=REPO,
                                  capture_output=True, text=True)
        assert repeated.returncode == 0, repeated.stdout + repeated.stderr
        assert 'verified preparation reused' in repeated.stdout
        assert {f: (f.stat().st_size, f.stat().st_mtime_ns) for f in media} == media
        assert {f: f.read_bytes() for f in sources} == sources


def test_review_records_the_paid_calls_it_was_told_to_skip(tmp_path):
    """--field-dictionary off and --inspection off reach the saved policy of every episode, and the dictionary stage
    says it was turned off instead of attempting a call."""
    from label.dictionary_stage import DICTIONARY_OFF
    root = tmp_path / "upload"
    recorder_folder(root, n=90, duplicate_stamps=False)
    job = tmp_path / "job"
    p = subprocess.run([sys.executable, "-m", "review", "--data", str(root), "--rig", "teleop_arms", "--out", str(job),
                        "--dataset", "mine", "--free", "--field-dictionary", "off", "--inspection", "off"],
                       cwd=REPO, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    policies = json.loads((job / "annotation_policy.json").read_text())
    assert policies and all(x["field_dictionary"] is False and x["inspection"] == "off" for x in policies.values())
    assert DICTIONARY_OFF in json.loads((job / "dictionary_status.json").read_text())["limitations"]


def _clip_episode(eps: Path, name: str, video: Path) -> None:
    d = eps / name
    d.mkdir(parents=True)
    (d / "sources.json").write_text(json.dumps({"exo": {"packed": str(video), "base_s": 0.0, "n_frames": 30}}))
    (d / "context.json").write_text(json.dumps({"fps": 30}))


def test_a_camera_file_that_does_not_decode_costs_only_its_own_episode(tmp_path):
    from board import clips
    good, bad = tmp_path / "good.mp4", tmp_path / "bad.mp4"
    subprocess.run([clips.find_ffmpeg(), "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30",
                    "-t", "1", "-pix_fmt", "yuv420p", str(good)], check=True)
    bad.write_bytes(good.read_bytes()[:600])                    # a truncated file: no frame decodes
    eps, out = tmp_path / "episodes", tmp_path / "clips"
    _clip_episode(eps, "episode_ok", good)
    _clip_episode(eps, "episode_broken", bad)
    run = lambda: subprocess.run([sys.executable, "-m", "board", "clips", "--episodes", str(eps), "--out", str(out)],
                                 cwd=REPO, capture_output=True, text=True)
    p = run()
    assert p.returncode == 0, p.stdout + p.stderr               # the upload goes on without the broken episode
    assert (out / "episode_ok.mp4").exists() and list(json.loads((out / "failed.json").read_text())) == ["episode_broken"]
    left = clips.set_aside_failed(eps, out)
    assert [f["name"] for f in left] == ["episode_broken"] and "could not be decoded" in left[0]["why"]
    assert sorted(d.name for d in eps.iterdir()) == ["episode_ok"]
    assert (tmp_path / "episodes_unclipped" / "episode_broken").is_dir()
    assert clips.set_aside_failed(eps, out) == []               # moved once, reported once
    rep = {"episodes": [{"name": "ok", "episode_id": "episode_ok", "seconds": 1.0},
                        {"name": "broken", "episode_id": "episode_broken", "seconds": 1.0}], "failed": [], "seconds": 2.0}
    clips.drop_from_report(rep, left)
    assert [e["name"] for e in rep["episodes"]] == ["ok"] and rep["seconds"] == 1.0
    assert rep["failed"] == [{"name": "broken", "why": left[0]["why"]}]
    (eps / "episode_ok").rename(tmp_path / "episode_ok")       # only the broken one left: nothing to show
    shutil.move(str(tmp_path / "episodes_unclipped" / "episode_broken"), str(eps / "episode_broken"))
    assert run().returncode == 1


def test_an_episode_left_off_the_board_is_explained_with_the_boards_camera_names(tmp_path):
    """Only an episode none of whose cameras decodes is left out, so its reason says that once for every camera, and a
    single camera is named as the board names it (never exo or extra1)."""
    from board import clips
    ep = tmp_path / "episode_x"
    ep.mkdir()
    views = ["exo", "left", "right", "extra1", "extra2"]
    (ep / "sources.json").write_text(json.dumps({v: {} for v in views}))
    ctx = {"profile": "teleop_arms", "cameras": {"extra1": {"name": "cam_low"}, "extra2": {"name": "cam_side"}}}
    (ep / "context.json").write_text(json.dumps(ctx))
    assert clips.failed_reason({v: "Command '...' returned non-zero exit status 183." for v in views}, ep) == \
        "every camera's video could not be decoded, so this episode was left out"
    assert clips.failed_reason({"extra1": "boom"}, ep) == "the cam_low camera video could not be decoded, so this " \
                                                          "episode was left out"
    (ep / "context.json").write_text(json.dumps({"profile": "handheld_gripper"}))
    assert clips.failed_reason({"left": "boom"}, ep) == "the left gripper camera video could not be decoded, so this episode was left out"


def _video(path: Path, frames: int) -> None:
    from board import clips
    subprocess.run([clips.find_ffmpeg(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc=size=160x120:rate=30", "-frames:v", str(frames), "-pix_fmt", "yuv420p", str(path)],
                   check=True)


def _two_camera_upload(tmp_path: Path, top: int = 30, wrist: int = 30) -> tuple[dict, Path, Path]:
    """A video upload of one episode filmed by a top camera and a left wrist camera, read as an upload is (the
    report, the episodes folder, the episode)."""
    from prepare import formats
    up = tmp_path / "up"
    up.mkdir()
    _video(up / "top.mp4", top)
    _video(up / "wrist_left.mp4", wrist)
    eps = tmp_path / "episodes"
    rep = formats.convert(up, "teleop_arms", eps, "mine", float("inf"), grouping={})
    return rep, eps, eps / rep["episodes"][0]["episode_id"]


def _clips(eps: Path, out: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "board", "clips", "--episodes", str(eps), "--out", str(out)],
                          cwd=REPO, capture_output=True, text=True)


def _board_episode(tmp_path: Path, eps: Path, ep: Path) -> dict:
    """The episode's file on a board built from a labelling run over it."""
    from test_board import _output
    run = tmp_path / "run"
    (run / "out").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"run_id": "r1", "code": "abc1234", "kind": "upload", "status": "done"}))
    (run / "out" / f"{ep.name}.json").write_text(json.dumps(_output(ep.name)))
    board = tmp_path / "board"
    board.mkdir()
    (board / "manifest.json").write_text(json.dumps({"board": "b", "datasets": [
        {"dataset": "mine", "run": str(run), "episodes": str(eps), "rules": rules.rules_for("teleop_arms")}]}))
    board_build.build(board)
    return json.loads((board / "qa" / f"{ep.name}.json").read_text())


def test_a_camera_clip_a_frame_short_keeps_the_episode_and_is_a_data_issue(tmp_path):
    """A camera whose clip comes out shorter than the episode (an upload's 63 s episode had 1890 frames in every
    camera's clip where the episode has 1891, and lost the episode) keeps its episode and its clip as cut. The
    mismatch is recorded in context.json, so it survives every rebuild, raises its data issue on the board, goes into
    the job's report notes, and the episode is labelled with that camera's cell empty past its last frame."""
    from board import clips
    from board import serve
    from label import episode as me
    rep, eps, ep = _two_camera_upload(tmp_path)
    _video(Path(json.loads((ep / "sources.json").read_text())["left"]["packed"]), 29)    # ends a frame early
    out = tmp_path / "clips"
    p = _clips(eps, out)
    assert p.returncode == 0, p.stdout + p.stderr
    assert clips.clip_frames(out / f"{ep.name}.mp4") == 30
    assert clips.clip_frames(out / "wrist_left" / f"{ep.name}.mp4") == 29
    assert json.loads((out / "failed.json").read_text()) == {}
    assert clips.set_aside_failed(eps, out) == [] and ep.is_dir()
    ctx = json.loads((ep / "context.json").read_text())
    what = ("The left wrist camera video ends at 0.97 s, before the episode ends at 1.00 s, so it shows nothing after "
            "that; 29 of the 30 frames its file lists could be read.")
    assert ctx["reader_issues"] == [{"kind": "camera_short", "camera": "wrist_left", "what": what, "t0_s": 0.967,
                                     "t1_s": 1.0, "clip_frames": 29, "episode_frames": 30}]
    assert set(json.loads((ep / "sources.json").read_text())) == {"exo", "left"}
    d = _board_episode(tmp_path, eps, ep)
    # the canned label's steps run past this 1 s episode, which the board flags beside it (model_steps_outside_episode)
    assert [x for x in d["dataset_checks"]["reader_issues"] if x["kind"] == "camera_short"] == [
        {**ctx["reader_issues"][0], "family": "clip-frames"}]
    assert "clip-frames" in serve._families(d)["families"]
    clips.note_camera_problems(rep, eps)
    assert rep["notes"] == ["episode_1: the left wrist camera video ends at 0.97 s, before the episode ends at "
                            "1.00 s, so it shows nothing after that; 29 of the 30 frames its file lists could be read."]
    assert rep["episodes"][0]["cameras"] == {"exo": "top", "left": "left"}
    # labelling decodes the files themselves: the left camera has no frame at the last instant, so its cell is empty
    req = me.build_request(ep)
    texts = [c["text"] for c in req["content"] if c.get("type") == "text"]
    last = [t for t in texts if t.startswith("=== detail view, last frame")]
    assert len(last) == 1 and "| camera top ===" in last[0], last
    assert ("Left's video ends before the episode does, so it has no frame at 0.97 s. Its cells at those times are "
            "empty"
            in req["prompt"])
    # a rerun keeps the clip as cut and the record as it was
    assert _clips(eps, out).returncode == 0
    assert json.loads((ep / "context.json").read_text())["reader_issues"] == ctx["reader_issues"]


def test_a_camera_that_does_not_decode_is_taken_out_and_the_episode_kept(tmp_path):
    """The main camera's file is not a video: the episode keeps its wrist camera for labelling and the board, the main
    camera is out of sources.json and the context's cameras (so nothing tries to decode it), the wrist camera, which
    was paired to the main camera by time, is the main camera now, and the camera is recorded as not decodable."""
    from board import clips
    from board import serve
    from label import episode as me
    rep, eps, ep = _two_camera_upload(tmp_path, top=30, wrist=33)
    assert json.loads((ep / "sources.json").read_text())["left"].get("kmap"), "the wrist must be paired by time"
    Path(json.loads((ep / "sources.json").read_text())["exo"]["packed"]).write_bytes(b"not a video at all" * 50)
    ctx = json.loads((ep / "context.json").read_text())
    ctx["reader_issues"] = [{"kind": "already", "what": "an entry another step wrote is kept"}]
    (ep / "context.json").write_text(json.dumps(ctx))
    out = tmp_path / "clips"
    p = _clips(eps, out)
    assert p.returncode == 0, p.stdout + p.stderr
    assert (out / "wrist_left" / f"{ep.name}.mp4").exists() and not (out / f"{ep.name}.mp4").exists()
    assert clips.set_aside_failed(eps, out) == [] and ep.is_dir()
    src = json.loads((ep / "sources.json").read_text())
    ctx = json.loads((ep / "context.json").read_text())
    assert list(src) == ["left"] and "kmap" not in src["left"] and list(ctx["cameras"]) == ["left"]
    what = "The main camera video could not be decoded, so this episode is shown and labelled without it."
    assert ctx["reader_issues"] == [{"kind": "already", "what": "an entry another step wrote is kept"},
                                    {"kind": "camera_not_decodable", "camera": "exo", "what": what}]
    d = _board_episode(tmp_path, eps, ep)
    fams = serve._families(d)["families"]
    assert "camera-undecodable" in fams and "d:Already" in fams        # a kind no family names is a data issue too
    clips.note_camera_problems(rep, eps)
    assert rep["notes"] == ["episode_1: an entry another step wrote is kept",
                            "episode_1: the main camera video could not be decoded, so this episode is shown and "
                            "labelled without it."]
    assert rep["episodes"][0]["cameras"] == {"left": "left"}
    req = me.build_request(ep)
    assert req["views"] == ["left"] and req["cam_labels"] == ["left"]
    assert "There is exactly 1 camera" in req["prompt"] and "\n- top:" not in req["prompt"]
    assert "\n- left:" in req["prompt"]


def test_an_episode_none_of_whose_cameras_decodes_is_set_aside_with_its_reason(tmp_path):
    from board import clips
    rep, eps, ep = _two_camera_upload(tmp_path)
    for v, s in json.loads((ep / "sources.json").read_text()).items():
        Path(s["packed"]).write_bytes(b"not a video at all" * 50)
    out = tmp_path / "clips"
    assert _clips(eps, out).returncode == 1                   # no episode came out whole
    left = clips.set_aside_failed(eps, out)
    assert left == [{"name": ep.name, "why": "every camera's video could not be decoded, so this episode was left out"}]
    assert not ep.exists() and (tmp_path / "episodes_unclipped" / ep.name).is_dir()
    clips.drop_from_report(rep, left)
    assert rep["episodes"] == [] and rep["failed"] == [{"name": "episode_1", "why": left[0]["why"]}]


def _machine_failures():
    import errno
    import os
    import signal
    full = "Error writing trailer of out.mp4: " + os.strerror(errno.ENOSPC)
    return {"a full disk": OSError(errno.ENOSPC, "No space left on device"),
            "ffmpeg on a full disk": subprocess.CalledProcessError(1, ["ffmpeg"], b"", full.encode()),
            "no progress": subprocess.TimeoutExpired(["ffmpeg"], 600.0),
            "ffmpeg missing": FileNotFoundError(errno.ENOENT, "No such file or directory", "/no/ffmpeg"),
            "ffmpeg killed": subprocess.CalledProcessError(-signal.SIGKILL, ["ffmpeg"], b"", b"")}


@pytest.mark.parametrize("how", list(_machine_failures()))
@pytest.mark.parametrize("which", ["every camera", "one camera beside one that does not decode"])
def test_a_cut_the_machine_stopped_never_takes_a_camera_or_episode_out(tmp_path, monkeypatch, how, which):
    """A full disk, a tool that stops making progress or is missing or killed says nothing about the video: the step
    fails, nothing is taken out, flagged or set aside, and a rerun on a sound machine cuts the episode. An episode
    whose every camera failed this way had been set aside as undecodable and never retried."""
    from board import clips
    rep, eps, ep = _two_camera_upload(tmp_path)
    if which != "every camera":
        Path(json.loads((ep / "sources.json").read_text())["exo"]["packed"]).write_bytes(b"not a video at all" * 50)
    before = {p.name: p.read_bytes() for p in ep.iterdir() if p.suffix == ".json"}
    out = tmp_path / "clips"
    real = clips.extract_one

    def machine(packed, base_s, n_frames, out_mp4, *args, **kwargs):
        if which == "every camera" or "wrist_left" in str(out_mp4):
            raise _machine_failures()[how]
        return real(packed, base_s, n_frames, out_mp4, *args, **kwargs)
    monkeypatch.setattr(clips, "extract_one", machine)
    monkeypatch.setattr(sys, "argv", ["clips", "--episodes", str(eps), "--out", str(out), "--jobs", "1"])
    assert clips.main() == 1
    assert json.loads((out / clips.FAILED).read_text()) == {}
    assert clips.set_aside_failed(eps, out) == [] and ep.is_dir()
    assert {p.name: p.read_bytes() for p in ep.iterdir() if p.suffix == ".json"} == before
    assert not list(out.rglob("*.mp4"))
    # the machine is sound again: the rerun cuts what decodes, and only a video that does not decode is taken out
    monkeypatch.setattr(clips, "extract_one", real)
    assert clips.main() == 0
    assert clips.set_aside_failed(eps, out) == [] and ep.is_dir()
    issues = [x["kind"] for x in json.loads((ep / "context.json").read_text()).get("reader_issues", [])]
    if which == "every camera":
        assert (out / f"{ep.name}.mp4").exists() and (out / "wrist_left" / f"{ep.name}.mp4").exists()
        assert "camera_not_decodable" not in issues
    else:
        assert (out / "wrist_left" / f"{ep.name}.mp4").exists() and not (out / f"{ep.name}.mp4").exists()
        assert issues.count("camera_not_decodable") == 1
        assert list(json.loads((ep / "sources.json").read_text())) == ["left"]


def test_only_the_video_is_blamed_for_a_clip_that_does_not_decode():
    import signal
    import av
    from board import clips
    for error in _machine_failures().values():
        assert clips.machine_stopped(error), error
    for error in (RuntimeError("out.mp4: no frame of in.mp4 decodes"), ValueError("no usable presentation interval"),
                  av.error.InvalidDataError(1094995529, "Invalid data found when processing input"),
                  subprocess.CalledProcessError(1, ["ffmpeg"], b"", b"moov atom not found"),
                  subprocess.CalledProcessError(-signal.SIGSEGV, ["ffmpeg"], b"", b"")):
        assert not clips.machine_stopped(error), error


def test_review_runs_the_sensor_checks_as_data_review_does(tmp_path, monkeypatch):
    """python -m review never ran checks.sensors, which Data Review runs after the capture checks on an upload whose
    episodes have other signals or depth, so a folder reviewed here had no sensor checks on its board."""
    import pytest
    import review.__main__ as rv
    root = tmp_path / "upload"
    recorder_folder(root, n=30, duplicate_stamps=False)
    steps = []

    def record(job, step, cmd, env, ok_codes=(0,)):
        steps.append((step, cmd[2] if len(cmd) > 2 else None))
        if step == "clips":
            raise SystemExit("stop after the checks")
        return 0
    monkeypatch.setattr(rv, "run_step", record)
    monkeypatch.setattr(sys, "argv", ["python -m review", "--data", str(root), "--rig", "teleop_arms", "--out",
                                      str(tmp_path / "job"), "--dataset", "mine", "--free"])
    with pytest.raises(SystemExit, match="stop after the checks"):
        rv.main()
    ep = next((tmp_path / "job" / "episodes").glob("episode_*"))
    assert (ep / "depth.json").exists() or (ep / "signals.npz").exists()
    names = [s for s, _ in steps]
    assert ("checks_sensors", "checks.sensors") in steps
    assert names.index("checks_capture") < names.index("checks_sensors") < names.index("clips")


def test_review_says_why_when_no_episode_can_be_put_on_the_board(tmp_path, monkeypatch):
    """board clips exits 1 when no episode came out whole: review says which camera file did not decode, the
    upload's own reason, instead of reporting the clips step as a crash."""
    import pytest
    import review.__main__ as rv
    root = tmp_path / "upload"
    recorder_folder(root, n=30, duplicate_stamps=False)
    real = rv.run_step

    def fake(job, step, cmd, env, ok_codes=(0,)):
        if step == "clips":
            assert 1 in ok_codes
            return 1
        return real(job, step, cmd, env, ok_codes)
    monkeypatch.setattr(rv, "run_step", fake)
    from board import clips as board_clips
    monkeypatch.setattr(board_clips, "set_aside_failed",
                        lambda eps, out: [{"name": p.name, "why": "the exo camera video could not be decoded"}
                                          for p in sorted(Path(eps).iterdir()) if p.name.startswith("episode_")])
    monkeypatch.setattr(board_clips, "drop_from_report", lambda rep, left: rep.update(episodes=[]))
    monkeypatch.setattr(sys, "argv", ["python -m review", "--data", str(root), "--rig", "teleop_arms", "--out",
                                      str(tmp_path / "job"), "--dataset", "mine", "--free"])
    with pytest.raises(SystemExit, match="no episode could be put on the board: the exo camera video could not be"):
        rv.main()
