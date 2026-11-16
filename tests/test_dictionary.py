import json
import multiprocessing
from pathlib import Path
import socket

import numpy as np
import pytest


def episode(root, name, signals, *, extra=None, state=None):
    ep = root / name
    ep.mkdir()
    arrays, descriptors = {}, []
    for i, (name, array, meta) in enumerate(signals):
        key = f"s{i}"
        arrays[key] = array
        descriptors.append({"name": name, "key": key, **meta})
    np.savez_compressed(ep / "signals.npz", **arrays)
    context = {"episode_id": ep.name, "state_kind": "none", "fps": 30,
               "signals": descriptors, "cameras": {}, **(extra or {})}
    if state is not None:
        np.savez(ep / "state.npz", state=state, action=state + 1)
        context.update(state_kind="recorded", source={"state": "robot.h5/qpos", "action": "robot.h5/ctrl"})
    (ep / "context.json").write_text(json.dumps(context))
    return ep


def test_inventory_deduplicates_compatible_fields_and_keeps_variant_owners(tmp_path):
    from label import dictionary as dd
    first = episode(tmp_path, "one", [("pressure", np.array([[1, 3]], dtype="f4"),
                                      {"shape": [2], "names": ["左", "right"], "source": "sensor.h5/pad"})],
                    state=np.zeros((2, 7), dtype="f4"))
    second = episode(tmp_path, "two", [("pressure", np.array([[5, 7]], dtype="f4"),
                                       {"shape": [2], "names": ["左", "right"], "source": "sensor.h5/pad"}),
                                      ("pressure", np.zeros((1, 3), dtype="f4"),
                                       {"shape": [3], "source": "other.h5/pad"})],
                     state=np.zeros((3, 7), dtype="f4"))
    inv = dd.inventory([second, first])
    pads = [f for f in inv["fields"] if f["name"] == "pressure"]
    assert len(pads) == 2
    shared = next(f for f in pads if f["shape"] == [2])
    assert shared["episodes"] == ["one", "two"]
    assert shared["names"] == ["左", "right"]
    assert shared["summary"] == {"minimum": 1, "maximum": 7, "mean": 4,
                                 "standard_deviation": pytest.approx(5 ** .5), "finite_fraction": 1}
    assert [b["context_path"] for b in shared["bindings"]] == ["signals/0", "signals/0"]
    assert all(f["episodes"] == ["one", "two"] for f in inv["fields"] if f["kind"] in ("state", "action"))
    assert dd.inventory([first, second])["digest"] == inv["digest"]
    assert str(tmp_path) not in json.dumps(inv)


def archived_state_episode(root, name, *, signal_columns=("observation.state", "action")):
    """An episode as prepare writes an unsupported LeRobot state (lerobot/aloha_static_coffee): state_kind none, the
    columns as signals and the raw rows archived in state.npz."""
    rng = np.random.default_rng(0)
    state, action = rng.normal(size=(5, 14)).astype("f4"), rng.normal(size=(5, 14)).astype("f4")
    names = [f"{side}_{joint}" for side in ("left", "right")
             for joint in ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate", "gripper")]
    values = {"observation.state": state, "action": action}
    ep = episode(root, name, [(column, values[column], {"dims": 14, "names": names, "source": "column " + column})
                              for column in signal_columns],
                 extra={"recorded_state_archive": {"file": "state.npz", "source_column": "observation.state",
                                                   "shape": [5, 14]}})
    np.savez(ep / "state.npz", state=state, action=action)
    return ep


def test_an_archived_state_is_described_once_by_the_signals_that_hold_it(tmp_path):
    from label import dictionary as dd
    inv = dd.inventory([archived_state_episode(tmp_path, "one")])
    numeric = [f for f in inv["fields"] if f.get("dtype") == "float32"]
    assert sorted(f["name"] for f in numeric) == ["action", "observation.state"], numeric
    assert all(f["kind"] == "signal" for f in numeric)
    bound = {f["name"]: [(b["file"], b["key"]) for b in f["bindings"]] for f in numeric}
    assert bound == {"observation.state": [("signals.npz", "s0"), ("state.npz", "state")],
                     "action": [("signals.npz", "s1"), ("state.npz", "action")]}
    # the request names each column's values once: no state.npz field and no unnamed bookkeeping copy of them
    request = json.loads(dd.request(inv)[0]["text"].split("\n", 1)[1])
    assert sorted(f["name"] for f in request["fields"] if f.get("dtype") == "float32") == ["action", "observation.state"]


def test_an_archive_column_no_signal_holds_keeps_its_own_field(tmp_path):
    from label import dictionary as dd
    inv = dd.inventory([archived_state_episode(tmp_path, "one", signal_columns=("observation.state",))])
    kinds = sorted((f["name"], f["kind"]) for f in inv["fields"] if f.get("dtype") == "float32")
    assert kinds == [("action", "action"), ("observation.state", "signal")], kinds


def test_summaries_are_bounded_and_requests_never_carry_rows_or_images(tmp_path, monkeypatch):
    from label import dictionary as dd
    values = np.arange(100_001, dtype="f4").reshape(-1, 1)
    ep = episode(tmp_path, "one", [("large", values, {"source": "values.npy"}),
                                   ("empty", np.zeros((0, 2), dtype="f4"), {}),
                                   ("gaps", np.array([[np.nan, 2, np.inf, 4]], dtype="f4"), {})],
                 extra={"state_unaligned": True, "annotation_subtasks": [{"label": "PRIVATE ROW TEXT"}],
                        "cameras": {"exo": {"name": "observation.images.top", "width": 1920, "height": 1080}},
                        "calibration": {"intrinsics": [[1, 0], [0, 1]]}})
    original = np.asarray
    def bounded(array, *args, **kwargs):
        if kwargs.get("dtype") in (np.float64, "float64"):
            assert np.size(array) <= 65_536
        return original(array, *args, **kwargs)
    monkeypatch.setattr(np, "asarray", bounded)
    inv = dd.inventory([ep])
    fields = {f["name"]: f for f in inv["fields"]}
    assert fields["large"]["summary"]["mean"] == 50_000
    assert fields["large"]["summary"]["standard_deviation"] == pytest.approx(28_867.802)
    assert fields["empty"]["summary"] == {"minimum": None, "maximum": None, "mean": None,
                                        "standard_deviation": None, "finite_fraction": 0}
    assert fields["gaps"]["summary"] == {"minimum": 2, "maximum": 4, "mean": 3,
                                       "standard_deviation": 1, "finite_fraction": .5}
    assert fields["calibration.intrinsics"]["kind"] == "calibration"
    assert fields["observation.images.top"]["kind"] == "camera"
    assert fields["annotation_subtasks"]["kind"] == "annotation"
    content = dd.request(inv)
    assert all(p["type"] == "text" for p in content)
    text = "\n".join(p["text"] for p in content)
    assert "PRIVATE ROW TEXT" not in text
    assert "100001" not in text
    assert "image_url" not in text
    assert len(text) < 15_000


def test_missing_numeric_files_and_bookkeeping_stay_named(tmp_path):
    from label import dictionary as dd
    ep = episode(tmp_path, "one", [("lost pressure", np.ones((2, 1), dtype="f4"), {"shape": [1]})])
    (ep / "signals.npz").unlink()
    inv = dd.inventory([ep])
    lost = next(f for f in inv["fields"] if f["name"] == "lost pressure")
    assert lost["kind"] == "signal"
    assert lost["shape"] == [1]
    assert lost["summary"]["finite_fraction"] == 0
    assert "signals.npz" in lost["limitations"][0]
    assert "unreadable" in lost["limitations"][0]


def fake_answer(entries, *, usage=None):
    answer = json.dumps({"entries": entries})
    def fake(content, model, reasoning, api_key, max_tokens, timeout):
        assert all(part["type"] == "text" for part in content)
        assert model == "openai/gpt-6-sol" and reasoning == "low"
        return {"choices": [{"message": {"content": answer}}], "usage": usage or {"cost": .001}}
    return fake


def test_receipt_retains_raw_damaged_unknown_and_missing_fields_without_retry(tmp_path):
    from label import dictionary as dd
    first = episode(tmp_path, "one", [("pad", np.ones((2, 1)), {})], state=np.zeros((2, 7)))
    second = episode(tmp_path, "two", [("pad", np.ones((2, 1)), {})], state=np.zeros((2, 7)))
    inv = dd.inventory([first, second])
    pad = next(f for f in inv["fields"] if f["name"] == "pad")["id"]
    state = next(f for f in inv["fields"] if f["kind"] == "state")["id"]
    entries = [{"id": pad, "meaning": "Measured pad activity", "role": "touch"},
               {"id": state, "meaning": "Recorded joint readings", "role": "joint_state"},
               {"id": "unknown", "meaning": "No recorded field", "role": "touch"},
               {"id": pad, "role": ["bad"]}]
    calls = []
    base = fake_answer(entries)
    def fake(*args, **kwargs):
        calls.append(1)
        return base(*args, **kwargs)
    job = tmp_path / "job"
    record = dd.prepare_upload(job, [first, second], "fake", fake)
    assert record["status"] == "success"
    assert set(record["entries"]) == {pad, state}
    assert record["unknown_entries"][0]["id"] == "unknown"
    assert len(record["damaged_entries"]) == 1
    assert json.loads(record["raw_text"])["entries"] == entries
    assert record["usage"] == {"cost": .001}
    assert record["cost_usd"] == .001
    assert record["missing_fields"]
    saved = (job / "dictionary.json").read_bytes()
    resumed = dd.prepare_upload(job, [second, first], "fake", fake)
    assert resumed["entries"] == record["entries"]
    assert len(calls) == 1
    assert (job / "dictionary.json").read_bytes() == saved
    np.savez(first / "signals.npz", s0=np.full((2, 1), 9.0))
    stale = dd.prepare_upload(job, [first, second], "fake", fake)
    assert stale["status"] == "stale" and stale["entries"] == {}
    assert len(calls) == 1
    assert (job / "dictionary.json").read_bytes() == saved


@pytest.mark.parametrize("response", ["failure", "malformed"])
def test_failure_receipts_survive_resume_and_free_runs_do_not_claim(tmp_path, response):
    from label import dictionary as dd
    ep = episode(tmp_path, "one", [("pad", np.ones((2, 1)), {})])
    calls = []
    def fake(*args, **kwargs):
        calls.append(1)
        if response == "failure":
            raise RuntimeError("fake provider failed")
        return {"choices": [{"message": {"content": "not JSON"}}], "usage": {"cost": .001}}
    job = tmp_path / "job"
    assert dd.prepare_upload(job, [ep], "fake", fake, dry_run=True)["status"] == "dry_run"
    assert dd.prepare_upload(job, [ep], None, fake)["status"] == "dry_run"
    assert not job.exists()
    failed = dd.prepare_upload(job, [ep], "fake", fake)
    assert failed["status"] == "failed" and failed["entries"] == {}
    assert failed["limitations"]
    if response == "malformed":
        assert failed["raw_text"] == "not JSON" and failed["cost_usd"] == .001
    assert dd.prepare_upload(job, [ep], "fake", fake)["status"] == "failed"
    assert len(calls) == 1


@pytest.mark.parametrize('outcome', ['timeout', 'http200_error', 'http504', 'bare_key_exhausted', 'untyped_key_proof', 'paid_malformed'])
def test_dictionary_unknown_or_paid_response_does_not_try_next_key(tmp_path, monkeypatch, outcome):
    import io
    import urllib.error
    from label import dictionary as dd, dictionary_stage, harness
    ep = episode(tmp_path, 'one', [('pressure', np.array([[1., 2.], [3., 4.]]),
                                  {'names': ['pad0', 'pad1'], 'source': 'native.h5/pressure'})],
                 extra={'recorded_metadata': {'operator_note': 'Original source task description.'}})
    job = tmp_path / 'job'
    job.mkdir()
    (job / 'dictionary_status.json').write_text(json.dumps({'projected_upper_usd': .37}))
    calls = []
    def transport(req, timeout):
        calls.append(req.get_header('Authorization').split()[-1])
        if outcome == 'timeout':
            raise TimeoutError('Reply may have been paid before the timeout')
        if outcome == 'untyped_key_proof':
            error = harness.KeyExhausted('Malformed unpaid proof')
            error.http_status = 402
            error.verified_unpaid = 'true'
            raise error
        if outcome == 'bare_key_exhausted':
            raise harness.KeyExhausted('Legacy exception without unpaid HTTP proof')
        if outcome == 'http504':
            # a gateway timeout: the request may still be running, and billed, behind the gateway
            raise urllib.error.HTTPError(req.full_url, 504, 'Gateway timeout', {}, io.BytesIO(b'{}'))
        if outcome == 'http200_error':
            return io.BytesIO(b'{"error":{"code":402,"message":"quota"}}')
        return io.BytesIO(json.dumps({'choices': [{'message': {'content': 'not JSON'}}],
                                    'usage': {'cost': .01}}).encode())
    monkeypatch.setattr(harness.urllib.request, 'urlopen', transport)
    record = dd.prepare_upload(job, [ep], 'sk-or-first', harness.call_model_once,
                               api_keys=['sk-or-first', 'sk-or-next'])
    assert record['status'] == 'failed' and record['entries'] == {}
    assert calls == ['sk-or-first']
    spend = dictionary_stage.dictionary_spend(job)
    if outcome == 'paid_malformed':
        assert record['raw_text'] == 'not JSON' and record['cost_usd'] == .01
        assert spend == {'cost_usd': .01, 'reserved_usd': .01, 'complete': True}
    else:
        assert record['cost_usd'] is None
        assert spend == {'cost_usd': None, 'reserved_usd': .37, 'complete': False}
    before = {name: (job / name).read_bytes() for name in ('dictionary.json', 'dictionary.claim')}
    resumed = dd.prepare_upload(job, [ep], 'sk-or-next', harness.call_model_once,
                                api_keys=['sk-or-next'])
    assert resumed['status'] == 'failed'
    assert calls == ['sk-or-first']
    assert all((job / name).read_bytes() == value for name, value in before.items())


@pytest.mark.parametrize('rejection', ['http429', 'http503', 'refused'])
def test_dictionary_rate_limits_and_unsent_requests_are_unpaid(tmp_path, monkeypatch, rejection):
    # A provider error status, or a connection that never carried the request, provably cost nothing: the call is
    # retried, then the next key is tried, and nothing stays reserved.
    import io
    import urllib.error
    from label import dictionary as dd, dictionary_stage, harness
    ep = episode(tmp_path, 'one', [('pressure', np.array([[1., 2.], [3., 4.]]),
                                  {'names': ['pad0', 'pad1'], 'source': 'native.h5/pressure'})])
    job = tmp_path / 'job'
    job.mkdir()
    (job / 'dictionary_status.json').write_text(json.dumps({'projected_upper_usd': .37}))
    calls = []
    def transport(req, timeout):
        calls.append(req.get_header('Authorization').split()[-1])
        if rejection == 'refused':
            raise urllib.error.URLError(ConnectionRefusedError(61, 'Connection refused'))
        raise urllib.error.HTTPError(req.full_url, int(rejection[4:]), 'Busy', {}, io.BytesIO(b'{}'))
    monkeypatch.setattr(harness.urllib.request, 'urlopen', transport)
    monkeypatch.setattr(harness.time, 'sleep', lambda seconds: None)
    record = dd.prepare_upload(job, [ep], 'sk-or-first', harness.call_model_once,
                               api_keys=['sk-or-first', 'sk-or-next'])
    assert record['status'] == 'failed' and record['entries'] == {}
    assert calls == ['sk-or-first'] * harness.UNPAID_ATTEMPTS + ['sk-or-next'] * harness.UNPAID_ATTEMPTS
    assert [a['state'] for a in record['dispatch_attempts']] == ['rejected unpaid', 'rejected unpaid']
    assert dictionary_stage.dictionary_spend(job) == {'cost_usd': 0., 'reserved_usd': 0., 'complete': True}


@pytest.mark.parametrize('terminal', ['all_rejected', 'next_unknown'])
def test_dictionary_only_proven_unpaid_attempts_clear_reservation(tmp_path, monkeypatch, terminal):
    import io
    import urllib.error
    from label import dictionary as dd, dictionary_stage, harness
    ep = episode(tmp_path, 'one', [('pressure', np.array([[1., 2.], [3., 4.]]),
                                  {'names': ['pad0', 'pad1'], 'source': 'native.h5/pressure'})])
    job = tmp_path / 'job'
    job.mkdir()
    (job / 'dictionary_status.json').write_text(json.dumps({'projected_upper_usd': .37}))
    calls = []
    def transport(req, timeout):
        key = req.get_header('Authorization').split()[-1]
        calls.append((key, json.loads(req.data)))
        if key == 'sk-or-empty':
            raise urllib.error.HTTPError(req.full_url, 402, 'Rejected', {}, io.BytesIO(b'{}'))
        assert key == 'sk-or-next'
        if terminal == 'all_rejected':
            raise urllib.error.HTTPError(req.full_url, 401, 'Rejected', {}, io.BytesIO(b'{}'))
        raise TimeoutError('Second dispatch outcome is unknown')
    monkeypatch.setattr(harness.urllib.request, 'urlopen', transport)
    record = dd.prepare_upload(job, [ep], 'sk-or-empty', harness.call_model_once,
                               api_keys=['sk-or-empty', 'sk-or-next'])
    assert record['status'] == 'failed' and record['entries'] == {}
    assert len(calls) == 2 and calls[0][1] == calls[1][1]
    assert record['dispatch_attempts'][0]['state'] == 'rejected unpaid'
    spend = dictionary_stage.dictionary_spend(job)
    if terminal == 'all_rejected':
        assert [a['state'] for a in record['dispatch_attempts']] == ['rejected unpaid', 'rejected unpaid']
        assert record['cost_usd'] == 0 and spend == {'cost_usd': 0., 'reserved_usd': 0., 'complete': True}
    else:
        assert record['dispatch_attempts'][1]['state'] != 'rejected unpaid'
        assert record['cost_usd'] is None and spend == {'cost_usd': None, 'reserved_usd': .37, 'complete': False}
    before = {p.name: p.read_bytes() for p in job.iterdir() if p.is_file()}
    assert dd.prepare_upload(job, [ep], 'sk-or-new', harness.call_model_once,
                             api_keys=['sk-or-new'])['status'] == 'failed'
    assert len(calls) == 2
    assert all((job / name).read_bytes() == value for name, value in before.items())


@pytest.mark.parametrize('crash_at', ['saved_refusal', 'next_dispatch'])
def test_dictionary_refusal_and_next_claim_survive_crash_without_resume_replay(tmp_path, monkeypatch, crash_at):
    import io
    import urllib.error
    from label import dictionary as dd, harness
    ep = episode(tmp_path, 'one', [('pressure', np.array([[1., 2.], [3., 4.]]),
                                  {'names': ['pad0', 'pad1'], 'source': 'native.h5/pressure'})],
                 extra={'recorded_metadata': {'operator_note': 'Retain the native task description.'}})
    job = tmp_path / 'job'
    class Crash(BaseException):
        pass
    original_save = dd._save
    def save_then_crash(path, value):
        original_save(path, value)
        attempts = value.get('dispatch_attempts') or []
        if crash_at == 'saved_refusal' and attempts and attempts[-1]['state'] == 'rejected unpaid':
            raise Crash()
    monkeypatch.setattr(dd, '_save', save_then_crash)
    calls = []
    def transport(req, timeout):
        key = req.get_header('Authorization').split()[-1]
        calls.append(key)
        if key == 'sk-or-empty':
            raise urllib.error.HTTPError(req.full_url, 402, 'Rejected', {}, io.BytesIO(b'{}'))
        assert key == 'sk-or-funded' and crash_at == 'next_dispatch'
        saved = json.loads((job / 'dictionary.json').read_text())
        assert [a['state'] for a in saved['dispatch_attempts']] == ['rejected unpaid', 'dispatched']
        retry = json.loads((job / 'dictionary.retry1.claim').read_text())
        assert retry['attempt'] == 2 and retry['request_digest'] == saved['request_digest']
        raise Crash()
    monkeypatch.setattr(harness.urllib.request, 'urlopen', transport)
    with pytest.raises(Crash):
        dd.prepare_upload(job, [ep], 'sk-or-empty', harness.call_model_once,
                          api_keys=['sk-or-empty', 'sk-or-funded'])
    saved = json.loads((job / 'dictionary.json').read_text())
    proof = saved['dispatch_attempts'][0]
    assert proof['attempt'] == 1 and proof['state'] == 'rejected unpaid'
    assert proof['http_status'] == 402 and proof['http_attempt'] == 1
    assert proof['verified_unpaid'] is True and proof['cost_usd'] == 0
    assert saved['request_digest']
    claim = json.loads((job / 'dictionary.claim').read_text())
    assert claim['attempt'] == 1 and claim['request_digest'] == saved['request_digest']
    assert ('sk-or-empty' not in json.dumps(saved) and 'sk-or-funded' not in json.dumps(saved))
    assert calls == (['sk-or-empty'] if crash_at == 'saved_refusal' else ['sk-or-empty', 'sk-or-funded'])
    before = {p.name: p.read_bytes() for p in job.iterdir() if p.is_file()}
    monkeypatch.setattr(dd, '_save', original_save)
    resumed = dd.prepare_upload(job, [ep], 'sk-or-funded', harness.call_model_once,
                               api_keys=['sk-or-funded'])
    assert resumed['status'] in ('pending', 'interrupted')
    assert all((job / name).read_bytes() == value for name, value in before.items())
    assert len(calls) == (1 if crash_at == 'saved_refusal' else 2)


def _claim_worker(job, ep, started, release, calls, output):
    from label import dictionary as dd
    def fake(*args, **kwargs):
        with calls.get_lock():
            calls.value += 1
        started.set()
        if not release.wait(5):
            raise RuntimeError("fixture release timed out")
        return fake_answer([])(*args, **kwargs)
    output.put(dd.prepare_upload(Path(job), [Path(ep)], "fake", fake)["status"])


def test_concurrent_processes_share_one_attempt_and_return_without_waiting(tmp_path):
    from label import dictionary as dd
    ep = episode(tmp_path, "one", [("pad", np.ones((2, 1)), {})])
    ctx = multiprocessing.get_context("spawn")
    started, release, calls, output = ctx.Event(), ctx.Event(), ctx.Value("i", 0), ctx.Queue()
    process = ctx.Process(target=_claim_worker, args=(str(tmp_path / "job"), str(ep), started, release, calls, output))
    process.start()
    try:
        assert started.wait(5)
        competing = dd.prepare_upload(tmp_path / "job", [ep], "fake", fake_answer([]))
        assert competing["status"] == "pending"
    finally:
        release.set()
        process.join(5)
    assert not process.is_alive() and process.exitcode == 0
    assert output.get(timeout=2) == "success"
    assert calls.value == 1
    assert dd.prepare_upload(tmp_path / "job", [ep], "fake", fake_answer([]))["status"] == "success"


def test_abandoned_pending_or_claim_only_never_dispatches(tmp_path):
    from label import dictionary as dd
    ep = episode(tmp_path, "one", [("pad", np.ones((2, 1)), {})])
    inv = dd.inventory([ep])
    for pending in (False, True):
        job = tmp_path / str(pending)
        job.mkdir()
        owner = {"pid": 2 ** 30, "host": socket.gethostname()}
        (job / "dictionary.claim").write_text(json.dumps({"schema": dd.SCHEMA, "inventory_digest": inv["digest"], **owner}))
        if pending:
            (job / "dictionary.json").write_text(json.dumps({"schema": dd.SCHEMA, "inventory_digest": inv["digest"],
                                                            "inventory": inv, "status": "pending", "owner": owner, "entries": {}}))
        def forbidden(*args, **kwargs):
            pytest.fail("abandoned attempt dispatched another model call")
        result = dd.prepare_upload(job, [ep], "fake", forbidden)
        assert result["status"] == "interrupted" and result["entries"] == {}


def test_human_edits_clear_role_and_validate_layout_without_changing_receipt(tmp_path):
    from label import dictionary as dd
    ep = episode(tmp_path, "one", [], state=np.zeros((2, 26)))
    field = next(f for f in dd.inventory([ep])["fields"] if f["kind"] == "state")["id"]
    layout = [{"start": 0, "count": 7, "name": "left arm"}, {"start": 7, "count": 7, "name": "right arm"},
              {"start": 14, "count": 6, "name": "left hand"}, {"start": 20, "count": 6, "name": "right hand"}]
    job = tmp_path / "job"
    record = dd.prepare_upload(job, [ep], "fake", fake_answer([{ "id": field, "meaning": "Recorded positions",
                                                               "role": "joint_state", "layout": layout}]))
    saved = (job / "dictionary.json").read_bytes()
    original = json.dumps(record, sort_keys=True)
    overrides = {"schema": dd.SCHEMA, "revision": 2, "entries": {field: {"role": "", "meaning": "Reviewed values"}},
                 "history": [{"revision": 1}, {"revision": 2}]}
    result = dd.effective(record, overrides)
    assert result["entries"][field]["role"] == ""
    assert result["entries"][field]["meaning"] == "Reviewed values"
    assert result["entries"][field]["layout"] == layout
    assert result["entries"][field]["provenance"] == "human"
    assert result["override_revision"] == 2
    assert dd.effective(record, {field: {"role": "event_flag"}})["entries"][field]["role"] == "event_flag"
    invalid = dd.effective(record, {field: {"layout": [{"start": 0, "count": 27, "name": "invented"}]},
                                    "unknown": {"role": "touch"}})
    assert invalid["entries"][field]["layout"] == layout
    assert invalid["override_limitations"]
    assert json.dumps(record, sort_keys=True) == original
    assert (job / "dictionary.json").read_bytes() == saved


@pytest.mark.parametrize("layout", [[{"start": 0, "count": 6, "name": "incomplete"}],
                                  [{"start": 0, "count": 7, "name": "one"}, {"start": 6, "count": 1, "name": "overlap"}],
                                  [{"start": True, "count": 7, "name": "boolean"}]])
def test_invalid_machine_layout_keeps_valid_meaning_and_names_limitation(tmp_path, layout):
    from label import dictionary as dd
    ep = episode(tmp_path, "one", [], state=np.zeros((2, 7)))
    field = next(f for f in dd.inventory([ep])["fields"] if f["kind"] == "state")["id"]
    result = dd.prepare_upload(tmp_path / "job", [ep], "fake", fake_answer([
        {"id": field, "meaning": "Recorded positions", "role": "joint_state", "layout": layout}]))
    assert result["entries"][field]["meaning"] == "Recorded positions"
    assert "layout" not in result["entries"][field]
    assert result["damaged_entries"] and result["limitations"]


def test_request_limit_keeps_full_inventory_and_failure_cache_without_dispatch(tmp_path, monkeypatch):
    from label import dictionary as dd
    name = "exact recorded name " + "z" * 1_000
    ep = episode(tmp_path, "one", [(name, np.ones((2, 1)), {"names": ["full original value name"]})])
    monkeypatch.setattr(dd, "REQUEST_BYTES", 100)
    def forbidden(*args, **kwargs):
        pytest.fail("oversize descriptors reached a model")
    result = dd.prepare_upload(tmp_path / "job", [ep], "fake", forbidden)
    assert result["status"] == "failed"
    field = next(f for f in result["inventory"]["fields"] if f["kind"] == "signal")
    assert field["name"] == name and field["names"] == ["full original value name"]
    assert any("100 bytes" in note and "no fields were omitted" in note for note in result["limitations"])
    assert result["cost_usd"] == 0
    assert dd.prepare_upload(tmp_path / "job", [ep], "fake", forbidden)["status"] == "failed"


def test_receipt_write_failure_after_dispatch_keeps_claim_and_unknown_failure_cost(tmp_path, monkeypatch):
    from label import dictionary as dd
    ep = episode(tmp_path, "one", [], extra={"recorded_metadata": {"operator_note": "recorded"}})
    calls = []
    def fake(*args, **kwargs):
        calls.append(1)
        raise TimeoutError("unknown billed provider timeout")
    save = dd._save
    def reject_final(path, value):
        if value["status"] != "pending":
            raise OSError("fixture final write failure")
        return save(path, value)
    monkeypatch.setattr(dd, "_save", reject_final)
    result = dd.prepare_upload(tmp_path / "job", [ep], "fake", fake)
    assert result["status"] == "failed" and result["cost_usd"] is None
    assert (tmp_path / "job" / "dictionary.claim").exists()
    assert dd.prepare_upload(tmp_path / "job", [ep], "fake", fake)["status"] == "pending"
    assert len(calls) == 1


def test_numeric_extremes_and_missing_member_keep_finite_summaries(tmp_path):
    from label import dictionary as dd
    ep = episode(tmp_path, "one", [("huge", np.array([[-1e308, 1e308]]), {}),
                                   ("tiny", np.array([[1e-300, 3e-300]]), {}),
                                   ("zero", np.zeros((1, 1)), {})])
    ctx = json.loads((ep / "context.json").read_text())
    ctx["signals"].append({"name": "missing member", "key": "absent", "shape": [7]})
    (ep / "context.json").write_text(json.dumps(ctx))
    fields = {f["name"]: f for f in dd.inventory([ep])["fields"]}
    assert fields["huge"]["summary"]["mean"] == 0
    assert fields["huge"]["summary"]["standard_deviation"] == 1e308
    assert fields["tiny"]["summary"]["mean"] == pytest.approx(2e-300, rel=1e-12, abs=0)
    assert fields["tiny"]["summary"]["standard_deviation"] == pytest.approx(1e-300, rel=1e-12, abs=0)
    assert fields["zero"]["summary"]["standard_deviation"] == 0
    assert fields["missing member"]["shape"] == [7] and fields["missing member"]["limitations"]


def test_camera_sidecar_and_original_bookkeeping_arrays_keep_descriptors(tmp_path):
    from label import dictionary as dd
    ep = episode(tmp_path, "one", [], extra={"cameras": {"exo": {"name": "top", "source": "camera topic"}},
                                            "source": {"bookkeeping": [{"name": "frame counter", "key": "raw",
                                                                        "shape": [1], "names": ["index"]}]}})
    np.savez(ep / "signals.npz", raw=np.array([[10], [20]], dtype="i4"))
    (ep / "sources.json").write_text(json.dumps({"exo": {"width": 1280, "height": 720, "pix_fmt": "yuv420p"}}))
    fields = dd.inventory([ep])["fields"]
    camera = next(f for f in fields if f["kind"] == "camera")
    counter = next(f for f in fields if f["name"] == "frame counter")
    assert camera["name"] == "top" and camera["shape"] == [720, 1280] and camera["dtype"] == "yuv420p"
    assert counter["kind"] == "bookkeeping" and counter["dtype"] == "int32" and counter["summary"]["mean"] == 15


def test_inventory_digest_ignores_generated_interpretation_and_checks(tmp_path):
    from label import dictionary as dd
    ep = episode(tmp_path, "one", [("pad", np.array([[1], [2]]), {})],
                 extra={"uploader_notes": {"calibration": "retained"}, "annotation_subtasks": [{"label": "recorded"}]})
    original = dd.inventory([ep])
    context = json.loads((ep / "context.json").read_text())
    context.update(data_dictionary={"entries": {"pad": {"role": "touch"}}}, checks={"passed": True},
                   contacts=[{"touch": False}], capture={"status": "checked"}, reader_display={"budget": 12})
    (ep / "context.json").write_text(json.dumps(context))
    resumed = dd.inventory([ep])
    assert resumed["digest"] == original["digest"]
    assert any(f["name"] == "uploader_notes.calibration" for f in resumed["fields"])
    assert any(f["name"] == "annotation_subtasks" for f in resumed["fields"])


@pytest.mark.parametrize("sources", [("/Users/private_a/sensor.h5", "/Users/private_b/sensor.h5"),
                                     ("/mnt/private_a/sensor.h5", "/mnt/private_b/sensor.h5"),
                                     ("C:\\private_a\\sensor.h5", "C:\\private_b\\sensor.h5")])
def test_external_sources_with_the_same_basename_keep_distinct_identities(tmp_path, sources):
    from label import dictionary as dd
    one = episode(tmp_path, "one", [("pad", np.ones((2, 1)), {"source": sources[0]})])
    two = episode(tmp_path, "two", [("pad", np.ones((2, 1)), {"source": sources[1]})])
    inventory = dd.inventory([one, two])
    pads = [field for field in inventory["fields"] if field["name"] == "pad"]
    assert len(pads) == 2
    assert {tuple(field["episodes"]) for field in pads} == {("one",), ("two",)}
    assert pads[0]["source"] != pads[1]["source"]
    assert "sensor.h5" in pads[0]["source"] and "sensor.h5" in pads[1]["source"]
    assert "/Users/" not in json.dumps(inventory)
    assert "private_a" not in json.dumps(dd.request(inventory))


def test_numeric_metadata_never_materialises_a_full_float64_array(tmp_path, monkeypatch):
    from label import dictionary as dd
    ep = episode(tmp_path, "one", [], extra={"calibration": {"recorded": [float(i) for i in range(131_073)]}})
    original = np.asarray
    def bounded(value, *args, **kwargs):
        if isinstance(value, list):
            assert len(value) <= 65_536, "numeric metadata was converted in bulk"
        return original(value, *args, **kwargs)
    monkeypatch.setattr(np, "asarray", bounded)
    field = next(f for f in dd.inventory([ep])["fields"] if f["name"] == "calibration.recorded")
    assert field["shape"] == [131_073] and field["dtype"] == "float64"
    assert field["summary"]["mean"] == pytest.approx(65_536)
    assert field["summary"]["minimum"] == 0 and field["summary"]["maximum"] == 131_072


@pytest.mark.parametrize('keys', [('/mnt/private_a/episodes.jsonl', '/mnt/private_b/episodes.jsonl'),
                                  (r'C:\private_a\episodes.jsonl', r'C:\private_b\episodes.jsonl')])
def test_metadata_path_keys_keep_private_names_out_of_descriptors_and_keep_distinct_bindings(tmp_path, keys):
    from label import dictionary as dd
    from label.dictionary_context import apply_context, field_interpretation
    ep = episode(tmp_path, 'one', [], extra={'recorded_metadata': {
        key: {'gain': [1, 3], 'rows': [{'text': 'PRIVATE ROW TEXT', 'image': 'PRIVATE IMAGE DATA'}]}
        for key in keys}})
    context = json.loads((ep / 'context.json').read_text())
    original = json.dumps(context, sort_keys=True)
    inv = dd.inventory([ep])
    fields = [field for field in inv['fields'] if field['name'].endswith('.gain')]
    assert len(fields) == 2 and len({field['id'] for field in fields}) == 2
    assert len({field['name'] for field in fields}) == len({field['source'] for field in fields}) == 2
    assert all('episodes.jsonl' in field['name'] for field in fields)
    text = dd.request(inv)[0]['text']
    assert 'private_a' not in text and 'private_b' not in text
    assert 'PRIVATE ROW TEXT' not in text and 'PRIVATE IMAGE DATA' not in text
    assert 'image_url' not in text
    for field in fields:
        assert field['summary']['mean'] == 2 and len(field['summary']) == 5
    record = {'schema': 1, 'inventory_digest': inv['digest'], 'inventory': inv, 'status': 'success',
              'entries': {field['id']: {'meaning': 'Recorded gain', 'role': 'calibration', 'provenance': 'machine'}
                          for field in fields}}
    reviewed = apply_context(context, record, {fields[0]['id']: {'meaning': 'Human reviewed gain', 'role': ''}})
    assert field_interpretation(reviewed, fields[0]['name'], fields[0]['kind']) == {
        'meaning': 'Human reviewed gain', 'role': '', 'provenance': 'human'}
    assert field_interpretation(reviewed, fields[1]['name'], fields[1]['kind'])['provenance'] == 'machine'
    assert json.dumps(context, sort_keys=True) == original


def test_pointer_escaping_keeps_slash_tilde_and_ordinary_metadata_keys_exact(tmp_path):
    from label import dictionary as dd
    from label.dictionary_context import apply_context, field_interpretation
    ep = episode(tmp_path, 'one', [], extra={
        'calibration/sensor~0': {'a/b': {'tilde~field': [11, 13]}, 'a~1b': {'tilde~field': [17, 19]}},
        'calibration': {'ordinary': [23, 25]}})
    context = json.loads((ep / 'context.json').read_text())
    inv = dd.inventory([ep])
    fields = {field['bindings'][0]['context_path']: field for field in inv['fields'] if field.get('summary')}
    paths = {'calibration~1sensor~00/a~1b/tilde~0field': 12,
             'calibration~1sensor~00/a~01b/tilde~0field': 18, 'calibration/ordinary': 24}
    assert all(path in fields for path in paths)
    for path, mean in paths.items():
        assert fields[path]['summary']['mean'] == mean
    entries = {fields[path]['id']: {'meaning': 'Original metadata', 'role': 'calibration', 'provenance': 'machine'}
               for path in paths}
    record = {'schema': 1, 'inventory_digest': inv['digest'], 'inventory': inv, 'status': 'success', 'entries': entries}
    changed = fields['calibration~1sensor~00/a~1b/tilde~0field']
    untouched = fields['calibration~1sensor~00/a~01b/tilde~0field']
    reviewed = apply_context(context, record, {changed['id']: {'meaning': 'Human selected exact slash key'}})
    assert field_interpretation(reviewed, changed['name'], 'calibration')['provenance'] == 'human'
    assert field_interpretation(reviewed, untouched['name'], 'calibration')['provenance'] == 'machine'
    ordinary = fields['calibration/ordinary']
    assert field_interpretation(reviewed, ordinary['name'], 'calibration')['meaning'] == 'Original metadata'
    assert context['calibration/sensor~0']['a/b']['tilde~field'] == [11, 13]


@pytest.mark.parametrize('value', [10 ** 400, [10 ** 400, 7]])
def test_unrepresentable_metadata_stays_qualified_without_aborting_dictionary(tmp_path, value):
    from label import dictionary as dd
    ep = episode(tmp_path, 'one', [], extra={'recorded_metadata': {'counter': value, 'gain': [1, 3]}})
    original = (ep / 'context.json').read_bytes()
    inv = dd.inventory([ep])
    fields = {field['name']: field for field in inv['fields']}
    counter = fields['recorded_metadata.counter']
    assert counter['limitations'] and 'summary range' in counter['limitations'][0]
    assert fields['recorded_metadata.gain']['summary']['mean'] == 2
    assert (ep / 'context.json').read_bytes() == original
    assert str(10 ** 400) not in dd.request(inv)[0]['text']
    calls = []
    def fake(*args, **kwargs):
        calls.append(args)
        return {'choices': [{'message': {'content': '{"entries": []}'}}], 'usage': {'cost': 0.001}}
    result = dd.prepare_upload(tmp_path / 'job', [ep], 'injected key', fake)
    assert len(calls) == 1 and result['status'] == 'success'


def test_a_dictionary_turned_off_makes_no_call_and_no_claim_and_says_why(tmp_path):
    from label import dictionary_stage
    ep = episode(tmp_path, 'one', [('pressure', np.array([[1., 2.], [3., 4.]]),
                                  {'names': ['pad0', 'pad1'], 'source': 'native.h5/pressure'})])
    job = tmp_path / 'job'
    job.mkdir()
    def forbidden(*a, **k):
        pytest.fail('a dictionary turned off must not be called')
    result = dictionary_stage.prepare(job, ep.parent, [ep.name], free=False, cap=10., api_key='sk-or-test',
                                      call_model=forbidden, enabled=False)
    assert result['cost_usd'] == 0. and dictionary_stage.DICTIONARY_OFF in result['limitations']
    assert not (job / 'dictionary.json').exists() and not (job / 'dictionary.claim').exists()
    assert dictionary_stage.dictionary_spend(job) == {'cost_usd': 0., 'reserved_usd': 0., 'complete': True}
    status = json.loads((job / 'dictionary_status.json').read_text())
    assert dictionary_stage.DICTIONARY_OFF in status['limitations']
