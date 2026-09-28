"""The gate: its selection and cases agree with each other and with the episode lists, each kind of fact is judged
as cases.json says, and the scorer counts parse failures, missing episodes and cold cost from run folders."""
from __future__ import annotations

import fnmatch
import json
from pathlib import Path

import pytest

from gate import score

REPO = Path(__file__).resolve().parent.parent


def test_every_case_names_gate_episodes_and_every_episode_is_in_its_list():
    sel, cases = score.load_selection(), score.load_cases()
    names = [x["name"] for xs in sel.values() for x in xs]
    assert len(names) == len(set(names))
    for key in cases:
        assert any(fnmatch.fnmatch(n, key) for n in names), key
    for xs in sel.values():
        for x in xs:
            lines = (REPO / "configs" / "slices" / f"{x['dataset']}.txt").read_text().splitlines()
            assert x["line"] in lines, x
            assert x["role"] == "case" or x["role"] == f"cost:{x['dataset']}"
    counts = {rig: sum(1 for x in xs if score.case_for(x["name"], cases)) for rig, xs in sel.items()}
    assert counts == {"teleop": 22, "handheld": 36, "ego": 7}
    assert sum(1 for x in sel["handheld"] if "Prepare_tableware" in x["name"]) == 32
    assert [x["name"] for x in sel["teleop"]].count("episode_001346_r6") == 1


def test_each_kind_of_fact_is_judged():
    case = {"outcome": ["failure"], "issue": ["instruction_mismatch"], "sev": "high"}
    good = {"completion": {"task_completed": "failure"},
            "data_issues": [{"category": "instruction_mismatch", "severity": "high"}]}
    assert score.judge(case, good) == (True, [])
    low = dict(good, data_issues=[{"category": "instruction_mismatch", "severity": "medium"}])
    assert score.judge(case, low) == (False, ["issue at ['medium']"])
    held = {"held": {"named": "fork", "not_named": "chopstick"}}
    fork = {"timeline": [{"action": "pick up", "object": "fork"}], "task_summary": "Set the table."}
    assert score.judge(held, fork)[0]
    assert score.judge(held, {"timeline": [{"object": "fork"}, {"action": "the fork, not chopsticks"}]})[0]
    wrong = score.judge(held, {"timeline": [{"object": "chopsticks"}]})
    assert not wrong[0] and wrong[1][0].startswith("claims chopstick") and "never names a fork" in wrong[1]
    ego = {"named": ["sew"], "max_issue_sev": "low"}
    labels = {"tasks": [{"task": "sew the fabric panels", "outcome": "success"}],
              "data_issues": [{"category": "camera_fault", "severity": "medium"}]}
    assert score.judge(ego, labels) == (False, ["issues above low: [('camera_fault', 'medium')]"])
    assert score.judge({"mistake": ["*slip*"]}, {"operator_mistakes": [{"category": "grasp_slip"}]})[0]
    assert score.judge({"align": ["aligned"]}, {"goal_alignment": {"relation": "broader"}}) == \
        (False, ["alignment broader"])


def _episode(tmp_path, name, seconds=60.0):
    d = tmp_path / "episodes" / name
    d.mkdir(parents=True)
    (d / "context.json").write_text(json.dumps({"duration_s": seconds, "fps": 30}))
    return d


def _out(run, name, ep, labels=None, cost=0.5, cached=0, parse_ok=True, route_cost=0.0):
    (run / "out").mkdir(parents=True, exist_ok=True)
    rec = {"episode_dir": str(ep), "parse_ok": parse_ok, "labels": labels or {},
           "usage": {"est_cost_usd": cost, "cached_tokens": cached},
           "config": {"resolution_route": {"routed": True, "cost_usd": route_cost}}}
    (run / "out" / f"{name}.json").write_text(json.dumps(rec))


def test_scorer_counts_failures_missing_episodes_and_cold_cost(tmp_path, monkeypatch):
    sel = {"teleop": [{"name": "episode_a", "role": "case"}, {"name": "episode_b", "role": "cost:molmo"},
                      {"name": "episode_c", "role": "cost:molmo"}, {"name": "episode_d", "role": "case"}]}
    monkeypatch.setattr(score, "load_selection", lambda: sel)
    monkeypatch.setattr(score, "load_cases", lambda: {"episode_a": {"outcome": ["failure"]},
                                                      "episode_d": {"outcome": ["success"]}})
    run = tmp_path / "run"
    _out(run, "episode_a", _episode(tmp_path, "a"), {"completion": {"task_completed": "failure"}})
    _out(run, "episode_b", _episode(tmp_path, "b", 1800), cost=10.0, cached=6000, route_cost=0.01)
    # a second send of an identical request: nearly everything read from the cache, which a real run never gets
    _out(run, "episode_c", _episode(tmp_path, "c", 1800), cost=4.0, cached=40000)
    p = score.score([run])["teleop"]
    assert (p["cases"], p["met"], p["asked"], p["parsed"]) == (2, 1, 4, 3)
    assert any("episode_d: no output" in f for f in p["failures"])
    c = p["cost_sample"]
    assert c["billed_usd"] == 14.01 and c["hours"] == 1.0 and c["shared_prefix_tokens"] == 6000
    assert c["cold_usd"] == pytest.approx(14.01 + 34000 * (score.CACHE_WRITE - score.CACHE_READ), abs=0.01)
