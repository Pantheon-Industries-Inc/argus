import json
import runpy
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from checks import stream_pairing as pairing
from review import __main__ as review


@pytest.mark.parametrize('mode,flag', [('pairing', []), ('jumps', ['--jumps']), ('grippers', ['--grippers'])])
def test_pairing_modes_retry_errors_and_invalidate_old_results(tmp_path, monkeypatch, mode, flag):
    key, _, finding = pairing.MODES[mode]
    assessed = []
    repaired = False

    def assess(ep):
        assessed.append(ep.name)
        if ep.name == 'episode_bad' and not repaired:
            raise OSError('camera temporarily unavailable')
        return None if ep.name == 'episode_inapplicable' else {finding: False}

    for name in ('episode_bad', 'episode_good', 'episode_inapplicable', 'episode_legacy'):
        ep = tmp_path / name
        ep.mkdir()
        # A legacy positive finding must be recomputed, even though it has no error.
        ctx = {key: {finding: True}} if name == 'episode_legacy' else {}
        (ep / 'context.json').write_text(json.dumps(ctx))
    monkeypatch.setitem(pairing.MODES, mode, (key, assess, finding))
    monkeypatch.setattr(pairing, 'ProcessPoolExecutor', ThreadPoolExecutor)
    monkeypatch.setattr(sys, 'argv', ['checks.stream_pairing', '--jobs', '1', *flag, str(tmp_path)])
    assert pairing.main() == 3
    assert set(assessed) == {'episode_bad', 'episode_good', 'episode_inapplicable', 'episode_legacy'}
    saved = json.loads((tmp_path / 'episode_bad/context.json').read_text())
    assert saved[key]['error'] == 'OSError: camera temporarily unavailable'
    assert json.loads((tmp_path / 'episode_inapplicable/context.json').read_text())[key] is None
    repaired = True
    assessed.clear()
    assert pairing.main() == 0
    assert assessed == ['episode_bad']
    assert json.loads((tmp_path / 'episode_bad/context.json').read_text())[key] == {finding: False}
    assessed.clear()
    monkeypatch.setitem(pairing.ALGORITHM_VERSIONS, key, pairing.ALGORITHM_VERSIONS[key] + 1)
    assert pairing.main() == 0
    assert len(assessed) == 4


@pytest.mark.parametrize('failure_module,failure_code,expected_sensors', [
    ('checks.capture_qc', 3, True), ('checks.stream_pairing', 3, True), ('checks.capture_qc', 2, False)])
def test_checks_aggregate_continues_only_recorded_partial_failures(
        monkeypatch, failure_module, failure_code, expected_sensors):
    modules = []

    def child(cmd, **kwargs):
        modules.append(cmd[2])
        return SimpleNamespace(returncode=failure_code if cmd[2] == failure_module else 0)

    monkeypatch.setattr('subprocess.run', child)
    monkeypatch.setattr(sys, 'argv', ['checks', 'episodes'])
    with pytest.raises(SystemExit) as error:
        runpy.run_module('checks', run_name='__main__')
    assert error.value.code == failure_code
    assert ('checks.sensors' in modules) is expected_sensors


def test_review_continues_recorded_pairing_and_capture_failures(tmp_path, monkeypatch):
    from label import dictionary_stage
    from checks import timebase
    modules = []
    source = tmp_path / 'source.bin'
    source.write_bytes(b'data')

    def prepared(job, *args):
        ep = job / 'episodes/episode_good'
        ep.mkdir(parents=True)
        (ep / 'signals.npz').write_bytes(b'present')
        return {'episodes': [{'episode_id': ep.name, 'state_kind': 'joints'}]}, {'policies': {}}

    class ReachedClips(Exception):
        pass

    def child(cmd, **kwargs):
        modules.append(cmd[2])
        if cmd[2] == 'board':
            raise ReachedClips()
        return SimpleNamespace(returncode=3 if cmd[2] in ('checks.stream_pairing', 'checks.capture_qc') else 0)

    monkeypatch.setattr(review, 'verified_preparation', prepared)
    monkeypatch.setattr(dictionary_stage, 'prepare', lambda *a, **k: None)
    monkeypatch.setattr(timebase, 'measure_folder', lambda *a: None)
    monkeypatch.setattr(review.subprocess, 'run', child)
    monkeypatch.setattr(sys, 'argv', ['review', '--data', str(source), '--out', str(tmp_path / 'job'),
                                      '--rig', 'ego_head', '--free'])
    with pytest.raises(ReachedClips):
        review.main()
    assert modules == ['checks.stream_pairing'] * 3 + ['checks.capture_qc', 'checks.sensors', 'board']


def test_generated_check_versions_do_not_change_recorded_context_identity(tmp_path):
    context = tmp_path / 'context.json'
    context.write_text(json.dumps({'fps': 30, 'stream_pairing': None}))
    before = review.file_identity(context, context=True)
    context.write_text(json.dumps({'fps': 30, 'stream_pairing': None, 'check_versions': {'stream_pairing': 1}}))
    assert review.file_identity(context, context=True) == before
