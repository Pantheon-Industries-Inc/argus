"""Grouping the issues of a long recording that report one condition (label/issue_groups.py). The fixtures are real
data issues of DiverseUMI takes labelled in parts, copied as the parts wrote them (tests/fixtures/issue_groups_real.json)."""
from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from label import issue_groups as ig

REAL = json.loads((Path(__file__).parent / "fixtures" / "issue_groups_real.json").read_text())


def issues(name):
    return copy.deepcopy(REAL[name]["issues"])


def result_of(name, parts=None):
    """A stitched result holding one fixture's issues, cut where its parts were (450s parts when not given)."""
    seconds = REAL[name]["seconds"]
    n = max(i.get("part") or 1 for i in REAL[name]["issues"])
    parts = parts or [{"part": k, "t0_s": (k - 1) * 435.0, "t1_s": min(seconds, k * 435.0 + 15.0)}
                      for k in range(1, n + 1)]
    return {"model": "openai/gpt-6-astra", "config": {"pieces": parts},
            "labels": {"data_issues": issues(name), "operator_mistakes": []}}


def answering(*groups, calls=None):
    """An ask() that answers every candidate with these groups ([(ids, sentence)]), and counts what it was asked."""
    def ask(category, members, context):
        if calls is not None:
            calls.append((category, copy.deepcopy(members)))
        return {"groups": [{"ids": list(ids), "issue": s, "why": "test"} for ids, s in groups]}, {"cost_usd": 0.01,
                                                                                                    "model": "m"}
    return ask


def flat(items):
    return [o for i in items for o in (i["occurrences"] if "occurrences" in i else [i])]


def test_discrete_events_of_one_category_are_never_asked_about():
    """Three failed grasps of one part, each short and in its own words: no issue reads as a condition and none
    comes from another part, so nothing is asked and the list is unchanged."""
    result = result_of("discrete")
    calls = []
    record = ig.apply(result, answering(((0, 1, 2), "merged"), calls=calls))
    assert calls == [] and record == {}
    assert result["labels"]["data_issues"] == issues("discrete")
    assert "issue_groups" not in result


def test_events_the_model_keeps_apart_stay_as_they_came():
    """Seven pauses of one take, one of them reported by both parts around a cut: when the model groups none, all
    seven stay unchanged and nothing falls back."""
    result = result_of("pauses_with_seam_copy")
    calls = []
    ig.apply(result, answering(calls=calls))
    assert result["labels"]["data_issues"] == ig._in_time_order(issues("pauses_with_seam_copy"))
    assert result["issue_groups"]["grouped"] == [] and result["issue_groups"]["fallbacks"] == []


def test_the_seam_copy_is_the_only_candidate_among_separate_pauses():
    """Two parts that both saw the pause at 278s around their cut are linked; the other five pauses are not."""
    items = issues("pauses_with_seam_copy")
    found = ig.candidates(items, {}, 450.0)
    assert [[items[n]["t_s"] for n in c["rows"]] for c in found] == [[278.178, 278.28]]


def test_a_whole_take_condition_reported_by_each_part_is_one_issue_with_every_report_kept():
    result = result_of("whole_take")
    ig.apply(result, answering(((0, 1), "The left and right wrist cameras are swapped.")))
    [one] = result["labels"]["data_issues"]
    assert one["issue"] == "The left and right wrist cameras are swapped."
    assert one["category"] == "camera_side_mismatch" and one["severity"] == "high"
    assert one["occurrences"] == issues("whole_take")          # verbatim, in time order (the untimed one last)
    assert (one["t_s"], one["end_s"]) == (0.0, 438.27) and one["parts"] == [1, 2]
    assert all(o["evidence"] in one["evidence"] for o in one["occurrences"])
    assert result["issue_groups"]["issues"]["data_issues"] == {"before": 2, "after": 1}
    assert result["issue_groups"]["grouped"][0]["signals"] == ["untimed", "wide"]


def test_a_recurring_condition_is_one_issue_spanning_its_reports():
    """The cable across the right camera at 959s and "again" at 1073s, both in part 3."""
    result = result_of("recurring")
    ig.apply(result, answering(((0, 1), "A cable intermittently crosses the right camera.")))
    [one] = result["labels"]["data_issues"]
    assert [o["t_s"] for o in one["occurrences"]] == [959.257, 1073.367]
    assert (one["t_s"], one["end_s"]) == (959.257, 1091.387)
    assert one["severity"] == max((i["severity"] for i in issues("recurring")), key=ig.SEVERITY_RANK.get)
    assert flat(result["labels"]["data_issues"]) == issues("recurring")


def test_severity_is_the_highest_of_the_members():
    items = issues("recurring")
    items[0]["severity"], items[1]["severity"] = "low", "medium"
    merged = ig.grouped_issue(items, "s", "w", "k")
    assert merged["severity"] == "medium"
    assert ig.verify(items, [merged]) is None


def test_the_check_refuses_a_group_that_drops_or_changes_an_occurrence():
    items = issues("whole_take")
    merged = ig.grouped_issue(items, "swapped", "w", "k")
    dropped = {**merged, "occurrences": merged["occurrences"][:1] + [copy.deepcopy(merged["occurrences"][0])]}
    assert "missing" in ig.verify(items, [dropped]) or "more than once" in ig.verify(items, [dropped])
    reworded = copy.deepcopy(merged)
    reworded["occurrences"][1]["evidence"] = "shortened"
    assert ig.verify(items, [reworded]) is not None
    lowered = {**merged, "severity": "low"}
    assert "severity" in ig.verify(items, [lowered])
    assert ig.verify(items, [merged]) is None


def test_a_group_the_check_refuses_is_kept_as_its_members(monkeypatch):
    real = ig.grouped_issue

    def losing(members, sentence, why, key):
        out = real(members, sentence, why, key)
        out["occurrences"] = out["occurrences"][:-1] + [{**out["occurrences"][-1], "evidence": None}]
        return out
    monkeypatch.setattr(ig, "grouped_issue", losing)
    result = result_of("whole_take")
    ig.apply(result, answering(((0, 1), "swapped")))
    assert result["labels"]["data_issues"] == issues("whole_take")
    [fallback] = result["issue_groups"]["fallbacks"]
    assert fallback["category"] == "camera_side_mismatch" and "missing" in fallback["reason"]


@pytest.mark.parametrize("reply", ['not json', '{"groups": [{"ids": [0, 7], "issue": "x"}]}',
                                   '{"groups": [{"ids": [0, 1]}]}', '{"groups": [{"ids": [0, 1], "issue": "a"}, '
                                   '{"ids": [1], "issue": "b"}]}', '{"answer": []}'])
def test_a_reply_that_cannot_be_read_keeps_every_issue(reply):
    with pytest.raises(ValueError):
        ig.read_answer(reply, 2)


def model_result(tmp_path, monkeypatch, behave):
    """apply() through model_asker with label/harness.py call_model_once replaced by behave."""
    from label import harness
    monkeypatch.setattr(harness, "call_model_once", behave)
    result = result_of("whole_take")
    ask = ig.model_asker(ig.settings(result), "sk-or-test", ig.Cache(tmp_path / "cache.json"), ig.Budget(5.0))
    return result, ask


def reply(text, cost=0.01):
    return {"id": "gen", "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"cost": cost, "prompt_tokens": 700, "completion_tokens": 60}}


def test_a_failed_call_keeps_every_issue_and_says_why(tmp_path, monkeypatch):
    from label.errors import DispatchError

    def fail(*a, **k):
        raise DispatchError("connection reset")
    result, ask = model_result(tmp_path, monkeypatch, fail)
    ig.apply(result, ask)
    assert result["labels"]["data_issues"] == issues("whole_take")
    assert "the request failed" in result["issue_groups"]["fallbacks"][0]["reason"]


def test_an_unreadable_reply_keeps_every_issue(tmp_path, monkeypatch):
    result, ask = model_result(tmp_path, monkeypatch, lambda *a, **k: reply("I think they are the same."))
    ig.apply(result, ask)
    assert result["labels"]["data_issues"] == issues("whole_take")
    assert "could not be read" in result["issue_groups"]["fallbacks"][0]["reason"]


def test_an_answer_is_paid_for_once_and_counted_in_the_recordings_cost(tmp_path, monkeypatch):
    sent = []

    def answer(content, *a, **k):
        sent.append(content)
        return reply('{"groups": [{"ids": [0, 1], "issue": "The wrist cameras are swapped.", "why": "same"}]}')
    result, ask = model_result(tmp_path, monkeypatch, answer)
    ig.apply(result, ask)
    assert len(result["labels"]["data_issues"]) == 1 and len(sent) == 1
    # the request holds only the candidate's issues, as JSON
    asked = json.loads(sent[0][0]["text"].split("Issues:\n", 1)[1])
    assert [i["issue"] for i in asked] == [i["issue"] for i in issues("whole_take")]
    assert result["auxiliary_cost_history"] == [{"kind": "issue_groups", "cost_usd": 0.01,
                                                 "previously_billed_usd": 0.0, "cost_is_conservative_estimate": False}]
    from label.harness import auxiliary_history_cost
    assert auxiliary_history_cost(result) == pytest.approx(0.01)
    again = result_of("whole_take")
    ask2 = ig.model_asker(ig.settings(again), "sk-or-test", ig.Cache(tmp_path / "cache.json"), ig.Budget(5.0))
    ig.apply(again, ask2)
    assert len(sent) == 1 and again["labels"]["data_issues"] == result["labels"]["data_issues"]
    assert again["issue_groups"]["calls"][0]["cached"] is True


def test_a_grouped_result_is_not_grouped_again():
    result = result_of("whole_take")
    ig.apply(result, answering(((0, 1), "swapped")))
    first = copy.deepcopy(result)
    ig.apply(result, answering(((0, 1), "again")))
    assert result == first


def test_the_step_can_be_turned_off(monkeypatch):
    monkeypatch.setenv(ig.ENV, "0")
    assert ig.enabled() is False
    monkeypatch.setenv(ig.ENV, "1")
    assert ig.enabled() is True
    monkeypatch.delenv(ig.ENV)
    assert ig.enabled() is ig.DEFAULT_ON


def test_stitch_run_with_no_model_key_leaves_the_result_as_it_is(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEYS", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = result_of("whole_take")
    before = copy.deepcopy(result)
    assert ig.apply_in_stitch(result, Path("/nonexistent/issue_groups.json"), ig.Budget(1.0)) == {}
    assert result == before


def test_stitch_run_groups_a_long_recordings_issues(monkeypatch, tmp_path):
    """stitch_run applies the step to each stitched recording through the labelling client and keeps the answers in
    job/issue_groups.json."""
    from label import pieces
    stitched = result_of("whole_take")
    monkeypatch.setattr(pieces, "stitch", lambda *a, **k: copy.deepcopy(stitched))
    monkeypatch.setattr(pieces, "_part_reply", lambda src, name: ({"labels": {}}, None))
    monkeypatch.setattr(pieces, "_carry_part_billing", lambda result, parts: None)
    monkeypatch.setattr(pieces, "_combine_request_config", lambda cfg: None)
    monkeypatch.setattr(pieces, "merge_jaws", lambda *a, **k: None)
    from label import evidence_access, harness
    monkeypatch.setattr(evidence_access, "merge", lambda *a, **k: {"parts": []})
    monkeypatch.setattr(harness, "saved_input_problem", lambda *a, **k: None)
    monkeypatch.setattr(harness, "get_keys", lambda: ["sk-or-test"])
    monkeypatch.setattr(harness, "call_model_once", lambda *a, **k: reply(
        '{"groups": [{"ids": [0, 1], "issue": "The wrist cameras are swapped.", "why": "same"}]}'))
    job = tmp_path / "job"
    for n in ("ep__p1", "ep__p2"):
        (job / "pieces" / n).mkdir(parents=True)
        (job / "pieces" / n / "context.json").write_text(json.dumps({"piece": {"index": int(n[-1]), "t0_s": 0,
                                                                               "t1_s": 1}}))
    (job / "run" / "out").mkdir(parents=True)
    stitched["usage"], stitched["decode_failed"] = {}, []
    stitched["config"]["resolution_route"] = {}
    out = tmp_path / "final"
    pieces.stitch_run(job, tmp_path / "eps", {"ep": ["ep__p1", "ep__p2"]}, out)
    written = json.loads((out / "ep.json").read_text())
    assert [len(i["occurrences"]) for i in written["labels"]["data_issues"]] == [2]
    assert (job / "issue_groups.json").is_file()
    monkeypatch.setenv(ig.ENV, "0")
    pieces.stitch_run(job, tmp_path / "eps", {"ep": ["ep__p1", "ep__p2"]}, out)
    # the issues as the parts reported them, with wording about the labelling taken out (label/wording.py)
    from label import wording
    published = wording.normalize_labels({"data_issues": issues("whole_take")})["data_issues"]
    assert json.loads((out / "ep.json").read_text())["labels"]["data_issues"] == published


@pytest.mark.skipif(not shutil.which("node"), reason="no node")
def test_the_board_lists_every_report_of_a_grouped_issue():
    repo = Path(__file__).resolve().parents[1]
    r = subprocess.run([shutil.which("node"), str(repo / "tests" / "issue_groups.js"),
                        str(repo / "board" / "index.html")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
