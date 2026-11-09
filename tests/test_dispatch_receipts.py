"""A grouping call (label/issue_groups.py) or a seam check (label/seam.py) killed after the provider took its request is
never bought twice. Each saved its receipt only after the reply, so a kill in between left no record: a restart paid
again, and the run's accounting counted only the second call. Now a pending receipt holding the call's full bound is
flushed to disk before dispatch, names the provider's generation as soon as the headers do, and is settled at what
OpenRouter billed on the next run."""
from __future__ import annotations

import json

import pytest

from label import harness, issue_groups as ig, seam
from label.errors import DispatchError
from test_issue_groups import reply, result_of
from test_seam import _episode, _halves, _result

GROUPS = '{"groups": [{"ids": [0, 1], "issue": "The wrist cameras are swapped.", "why": "same"}]}'
BILLED = {"total_cost": 0.0123, "cancelled": False, "finish_reason": "stop", "tokens_completion": 60}


class Killed(BaseException):
    """The process dies: nothing after it in the caller runs."""


def accepted_then_killed(sent):
    def call(content, *a, on_generation_id=None, **k):
        sent.append(content)
        if on_generation_id:            # the provider took the request and named its generation
            on_generation_id("gen-7")
        raise Killed()
    return call


def counting(sent, text=GROUPS):
    def call(content, *a, **k):
        sent.append(content)
        return reply(text)
    return call


def _ask(path, monkeypatch, call):
    monkeypatch.setattr(harness, "call_model_once", call)
    result = result_of("whole_take")
    return result, ig.model_asker(ig.settings(result), "sk-or-test", ig.Cache(path), ig.Budget(5.0))


def test_a_grouping_call_killed_after_dispatch_is_never_bought_again(tmp_path, monkeypatch):
    path, sent = tmp_path / "issue_groups.json", []
    result, ask = _ask(path, monkeypatch, accepted_then_killed(sent))
    with pytest.raises(Killed):
        ig.apply(result, ask)
    (saved,) = json.loads(path.read_text())["answers"].values()
    assert saved["pending"] and saved["request"]["generation_id"] == "gen-7"
    bound = saved["request"]["reserved_usd"]
    # the run's accounting holds the call at its full bound until it is known
    assert ig.spend(path) == {"cost_usd": pytest.approx(bound), "complete": False}
    result, ask = _ask(path, monkeypatch, counting(sent))
    ig.apply(result, ask)
    assert len(sent) == 1
    assert [f["reason"] for f in result["issue_groups"]["fallbacks"]] == [ig.PENDING_PROBLEM]
    assert len(result["labels"]["data_issues"]) == 2
    assert result["auxiliary_cost_history"][0]["cost_is_conservative_estimate"] is True


def test_a_pending_grouping_receipt_settles_at_what_openrouter_billed(tmp_path, monkeypatch):
    path, sent = tmp_path / "issue_groups.json", []
    result, ask = _ask(path, monkeypatch, accepted_then_killed(sent))
    with pytest.raises(Killed):
        ig.apply(result, ask)
    asked = []
    assert ig.reconcile(path, ["sk-or-test"], lookup=lambda gid, keys: asked.append(gid) or None) == \
        {"pending": 1, "resolved": 0}
    assert ig.spend(path)["complete"] is False      # no record yet: the hold stays
    assert ig.reconcile(path, ["sk-or-test"], lookup=lambda gid, keys: {"id": gid, **BILLED}) == \
        {"pending": 1, "resolved": 1}
    assert ig.spend(path) == {"cost_usd": pytest.approx(0.0123), "complete": True}
    assert asked == ["gen-7"]
    # still never sent again
    result, ask = _ask(path, monkeypatch, counting(sent))
    ig.apply(result, ask)
    assert len(sent) == 1


def test_two_writers_of_one_receipt_file_never_send_one_request_twice(tmp_path, monkeypatch):
    """Two processes that opened the file before either asked: the second finds the first's claim on disk."""
    path, sent = tmp_path / "issue_groups.json", []
    first, second = ig.Cache(path), ig.Cache(path)
    monkeypatch.setattr(harness, "call_model_once", accepted_then_killed(sent))
    result = result_of("whole_take")
    with pytest.raises(Killed):
        ig.apply(result, ig.model_asker(ig.settings(result), "sk-or-test", first, ig.Budget(5.0)))
    monkeypatch.setattr(harness, "call_model_once", counting(sent))
    again = result_of("whole_take")
    budget = ig.Budget(5.0)
    ig.apply(again, ig.model_asker(ig.settings(again), "sk-or-test", second, budget))
    assert len(sent) == 1 and budget.spent == 0.0
    # and an answer one writer kept is not lost when the other writes
    first.put("other", {"text": "{}", "request": {"cost_usd": 0.5}})
    assert set(json.loads(path.read_text())["answers"]) == {"other", *second.answers}


def test_a_grouping_call_that_provably_cost_nothing_leaves_no_receipt(tmp_path, monkeypatch):
    path, sent = tmp_path / "issue_groups.json", []

    def unpaid(content, *a, **k):
        sent.append(content)
        raise DispatchError("refused before sending", verified_unpaid=True, stage="not sent")
    result, ask = _ask(path, monkeypatch, unpaid)
    ig.apply(result, ask)
    assert (json.loads(path.read_text())["answers"] if path.exists() else {}) == {}
    result, ask = _ask(path, monkeypatch, counting(sent))
    ig.apply(result, ask)
    assert len(sent) == 2 and len(result["labels"]["data_issues"]) == 1


def _seam_job(tmp_path):
    src = _episode(tmp_path)
    from label import episode as me
    left, right = _halves(cut=1.5, end=3.0)
    left.update(start_s=0.1, end_s=1.5)
    right.update(end_s=2.8, completed_at_s=2.7)
    result = _result([left, right], [1.5], [1.5, 3.0], views=me.views(me.load(src)), label_schema='tasks',
                     given_prompt=None)
    result['labels']['completion'] = {'basis': 'tasks', 'from_tasks': 2}
    label = tmp_path / 'labels' / f'{src.name}.json'
    label.parent.mkdir()
    label.write_text(json.dumps(result))
    out = tmp_path / 'seams.json'
    return src, out, ['--label', str(label.parent), '--footage', str(src), '--out', str(out)]


def _verdict(content, *a, **k):
    return {'id': 'gen-2', 'usage': {'cost': 0.004}, 'choices': [{'finish_reason': 'stop', 'message': {
        'content': '{"same_task": true, "evidence": "The same blanket is spread across the cut."}'}}]}


def test_a_seam_check_killed_after_dispatch_is_never_bought_again(tmp_path, monkeypatch):
    src, out, argv = _seam_job(tmp_path)
    sent = []
    monkeypatch.setattr(harness, 'get_keys', lambda: ['sk-or-test'])
    monkeypatch.setattr(harness, 'openrouter_generation', lambda gid, keys: None)
    monkeypatch.setattr(harness, 'call_model_once', accepted_then_killed(sent))
    with pytest.raises(Killed):
        seam.main(argv)
    saved = json.loads(out.read_text())
    verdict = saved['seams'][src.name]['1.5']
    assert verdict['pending'] and verdict['same_task'] == 'unclear'
    assert verdict['request']['generation_id'] == 'gen-7'
    # the hold is in the file's cost until the call is known
    assert saved['cost_usd'] == pytest.approx(verdict['request']['reserved_usd'])
    monkeypatch.setattr(harness, 'call_model_once', lambda *a, **k: sent.append(a) or _verdict(*a))
    assert seam.main(argv) == 0
    assert len(sent) == 1
    assert json.loads(out.read_text())['seams'][src.name]['1.5']['pending'] is True


def test_a_pending_seam_receipt_settles_at_what_openrouter_billed(tmp_path, monkeypatch):
    src, out, argv = _seam_job(tmp_path)
    sent = []
    monkeypatch.setattr(harness, 'get_keys', lambda: ['sk-or-test'])
    monkeypatch.setattr(harness, 'call_model_once', accepted_then_killed(sent))
    with pytest.raises(Killed):
        seam.main(argv)
    monkeypatch.setattr(harness, 'openrouter_generation', lambda gid, keys: {'id': gid, **BILLED})
    monkeypatch.setattr(harness, 'call_model_once', lambda *a, **k: sent.append(a) or _verdict(*a))
    assert seam.main(argv) == 0
    saved = json.loads(out.read_text())
    assert len(sent) == 1
    assert saved['cost_usd'] == pytest.approx(0.0123)
    assert saved['seams'][src.name]['1.5']['request']['cost_source'] == 'generation lookup'


def test_a_second_seam_check_on_one_seams_file_sends_nothing(tmp_path, monkeypatch):
    import fcntl
    src, out, argv = _seam_job(tmp_path)
    monkeypatch.setattr(harness, 'get_keys', lambda: ['sk-or-test'])
    monkeypatch.setattr(harness, 'call_model_once', lambda *a, **k: pytest.fail('sent while another check runs'))
    with open(out.with_name(f'.{out.name}.lock'), 'a') as other:
        fcntl.flock(other, fcntl.LOCK_EX)
        with pytest.raises(SystemExit, match='another seam check'):
            seam.main(argv)
    assert not out.exists()
