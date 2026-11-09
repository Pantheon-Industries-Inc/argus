"""The calls that merge data issues after a long recording's parts (label/issue_groups.py) spend from the run's one
--cap, never from a limit of their own beside it. A review capped at about $0.65 once finished with $0.75 charged,
because grouping held its own $5."""
from __future__ import annotations

import io
import json
import sys

import h5py
import numpy as np
import pytest

import complete_reply
from label import harness, issue_groups as ig

ISSUE = {"issue": "", "category": "camera_view", "severity": "medium", "t_s": None, "end_s": None,
         "evidence": "The frames show mostly ceiling."}


def _reply(content, cost):
    return io.BytesIO(json.dumps({"id": "gen", "usage": {"cost": cost}, "choices": [
        {"finish_reason": "stop", "message": {"content": content}}]}).encode())


def _review(tmp_path, monkeypatch, cap, grouping_cost, second_part=(.003, .003)):
    """python -m review --cap cap on a 452s take cut into two parts, each reporting the one whole-take condition in its
    own words, every call answered in-process at a fixed charge; (exit code, calls, charges, job)."""
    from review import __main__ as review
    upload = tmp_path / "upload"
    upload.mkdir()
    with h5py.File(upload / "take.h5", "w") as source:
        source.attrs["fps"] = 1
        source.create_dataset("images/top", data=np.full((452, 24, 32, 3), 70, np.uint8))
        source.create_dataset("force", data=np.arange(452, dtype=float))
    job = tmp_path / "job"
    monkeypatch.setenv("OPENROUTER_API_KEYS", "sk-or-fixture")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(review, "build_review_board", _no_board(monkeypatch, review.build_review_board))

    def step(job, kind, command, env, **kwargs):
        if kind in ("dry_run", "label"):
            with monkeypatch.context() as local:
                local.setattr(sys, "argv", command[2:])
                for name in ("ARGUS_REVIEW_OUTPUT_LOCK_FD", "ARGUS_REVIEW_OUTPUT_LOCK_PATH"):
                    if name in env:
                        local.setenv(name, env[name])
                return harness.main()
        return 0
    monkeypatch.setattr(review, "run_step", step)
    calls, charges = [], []

    def provider(request, timeout):
        text = json.dumps(json.loads(request.data)["messages"])
        if "ISSUE GROUPING" in text:
            calls.append("grouping"); charges.append(grouping_cost)
            return _reply('{"groups": [{"ids": [0, 1], "issue": "The top camera faces the ceiling.", "why": "same"}]}',
                          grouping_cost)
        if "FIELD INVENTORY" in text:
            cost = second_part[0] if calls.count("selection") else .003
            calls.append("selection"); charges.append(cost)
            return _reply('{"requests":[]}', cost)
        cost = second_part[1] if calls.count("final") else .003
        calls.append("final"); charges.append(cost)
        issue = {**ISSUE, "issue": f"In stretch {calls.count('final')} the top camera stays tilted toward the ceiling "
                                   "throughout."}
        return _reply(json.dumps(complete_reply.labels("A hand reaches.", request=request, data_issues=[issue])), cost)
    monkeypatch.setattr(harness.urllib.request, "urlopen", provider)
    monkeypatch.setattr(sys, "argv", ["review", "--data", str(upload), "--out", str(job), "--rig", "ego_head",
                                      "--concurrency", "1", "--max-tokens", "100", "--cap", str(cap)])
    return review.main(), calls, charges, job


def _no_board(monkeypatch, real):
    def board(*args, **kwargs):
        from board import build
        with monkeypatch.context() as local:
            local.setattr(build, "build", lambda job: None)
            return real(*args, **kwargs)
    return board


def test_a_review_never_charges_past_its_cap_for_grouping(tmp_path, monkeypatch):
    # labelling a part holds about $1.49 until billed (its selection round and its final call at their output limits),
    # so the cap must cover that; the second part bills close to its holds ($1.30), which leaves about $0.19, and a
    # grouping call may cost up to about $0.21 (this one bills $0.20). Before, grouping asked anyway, under its own $5.
    rc, calls, charges, job = _review(tmp_path, monkeypatch, 1.50, 0.20, second_part=(.55, .75))
    assert calls.count("final") == 2
    assert "grouping" not in calls
    assert sum(charges) <= 1.50
    stitched = json.loads((job / "run_final" / "out" / "episode_take.json").read_text())
    # the issues stay as the parts reported them, and the record says why
    assert len(stitched["labels"]["data_issues"]) == 2
    reasons = [f["reason"] for f in stitched["issue_groups"]["fallbacks"]]
    assert reasons and all("--cap leaves" in r for r in reasons), reasons
    info = json.loads((job / "run_final" / "run.json").read_text())
    assert info["cost_usd"] == pytest.approx(sum(charges))


def test_a_review_with_room_under_its_cap_still_groups(tmp_path, monkeypatch):
    rc, calls, charges, job = _review(tmp_path, monkeypatch, 2, 0.002)
    assert rc == 0 and calls.count("grouping") == 1
    stitched = json.loads((job / "run_final" / "out" / "episode_take.json").read_text())
    assert [len(i["occurrences"]) for i in stitched["labels"]["data_issues"]] == [2]
    assert json.loads((job / "run_final" / "run.json").read_text())["cost_usd"] == pytest.approx(0.014)


def test_the_budget_counts_spent_and_open_reservations_against_the_smaller_limit():
    budget = ig.Budget(5.0, 0.30)
    assert budget.reserve(0.20)                 # open, not yet settled
    assert not budget.reserve(0.11)             # 0.20 held + 0.11 > 0.30
    budget.settle(0.20, 0.05)
    assert budget.reserve(0.25)                 # 0.05 spent + 0.25 = 0.30
    assert budget.refused == 1 and "--cap leaves $0.3000" in budget.refusal()
    own = ig.Budget(1.0, 7.0)                   # grouping keeps its own smaller limit
    assert not own.reserve(1.5) and "$1 cap for issue grouping" in own.refusal()
    assert ig.Budget(1.0).limit == 1.0


def test_a_staged_run_gives_grouping_only_what_its_cap_leaves(tmp_path, monkeypatch):
    from label import pieces, run as lrun
    run = tmp_path / "run"
    run.mkdir()
    ig.Cache(run / "issue_groups.json").put("earlier", {"text": "{}", "request": {"cost_usd": 0.05}})
    monkeypatch.setattr(lrun, "billed_cost", lambda run, info: 0.70)
    seen = {}
    monkeypatch.setattr(pieces, "stitch_run", lambda *a, allowance=None, **k: seen.setdefault("left", allowance) and {})
    lrun.stitch(run, {"kind": "full", "cap_usd": 1.0, "pieces": {"ep": ["ep__p1"]}}, tmp_path)
    assert seen["left"] == pytest.approx(0.25)
    seen.clear()
    lrun.stitch(run, {"kind": "full", "cap_usd": 0.5, "pieces": {"ep": ["ep__p1"]}}, tmp_path)
    assert seen["left"] == 0.0


def test_earlier_grouping_spends_from_the_review_allowance(tmp_path):
    from label import dictionary_stage
    job = tmp_path / "job"
    job.mkdir()
    assert dictionary_stage.remaining_label_budget(job, 1.0) == pytest.approx(1.0)
    ig.Cache(job / "issue_groups.json").put("earlier", {"text": "{}", "request": {"cost_usd": 0.4}})
    assert dictionary_stage.remaining_label_budget(job, 1.0) == pytest.approx(0.6)
