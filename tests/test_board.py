"""The board's build side: harness outputs to board files, the rules, the families, and a board built from its
manifest (runs, comparisons, hand pose)."""
from __future__ import annotations

import importlib
import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from board import build as board_build
from board import hands, rules, to_board
from board.families import Families

REPO = Path(__file__).resolve().parent.parent
NO_TASK_TEXT = next(r for r in rules.GENERAL if r["kind"] == "no_task_text")


# ---------------------------------------------------------------- harness output -> board file

def _output(ep: str, **labels) -> dict:
    """One harness output (label/harness.py) with a dense timeline."""
    return {"episode_dir": f"/x/{ep}", "parse_ok": True, "model": "some/model", "reasoning_effort": "medium",
            "given_prompt": "Put the cup on the plate.", "prompt_mode": "given",
            "usage": {"est_cost_usd": 0.12, "latency_s": 30.0, "prompt_tokens": 1000, "completion_tokens": 500},
            "config": {"timesteps_s": [0.0, 1.0, 2.0], "views": ["exo", "left"], "cam_labels": {"exo": "top"}},
            "dataset_checks": {"timebase": {"sped_up_recording": False}},
            "labels": {"task_summary": "The left arm puts the cup on the plate.",
                       "timeline": [{"start_s": 0.0, "end_s": 1.5, "arm": "left", "action": "reach", "object": "cup",
                                     "contribution": "advancing", "progress": 0.2},
                                    {"end_s": 2.0, "action": "no start time: not a marker"},
                                    {"start_s": 1.5, "end_s": 3.0, "arm": "left", "action": "place",
                                     "object": "cup", "destination": "plate", "progress": 1.0}],
                       "key_events": [{"t_s": 2.9, "label": "cup on plate", "kind": "goal"}, {"label": "no time"}],
                       "scene": {"objects": [{"name": "cup", "attributes": ["Red", "ceramic"]}, {"name": "plate"}]},
                       "completion": {"task_completed": "success", "completed_at_s": 2.9, "goal_reached_at_s": 2.9},
                       **labels}}


def test_convert_a_harness_output():
    d = to_board.convert(_output("episode_000007"), "demo")
    assert d["dataset"] == "demo" and d["_meta"]["episode_id"] == "episode_000007"
    # a step or key event with no time the board can read is kept, untimed (the page lists it after the timed ones)
    assert [(e["t_s"], e["verb_class"], e["event_idx"]) for e in d["event_labels"]] == [
        (0.0, "reach", 0), (None, "no start time: not a marker", 1), (1.5, "place", 2)]
    assert d["event_labels"][2]["carry_phase"] == "-> plate" and d["event_labels"][2]["end_s"] == 3.0
    assert [(k["t_s"], k["label"]) for k in d["key_events"]] == [(2.9, "cup on plate"), (None, "no time")]
    assert d["objects"] == [{"name": "cup", "color": "Red"}, {"name": "plate", "color": None}]
    assert d["completion"]["task_completed"] == "success" and d["_meta"]["engaged_events"] == 3
    assert d["_meta"]["given_prompt"] == "Put the cup on the plate." and d["_meta"]["model"] == "some/model"
    assert d["timesteps_s"] == [0.0, 1.0, 2.0] and d["camera_views"] == ["exo", "left"]
    assert d["_usage"]["est_cost_usd"] == 0.12
    assert d["data_issues"] == [] and d["operator_mistakes"] == [] and d["goal_alignment"] is None


def test_a_partial_outcome_is_a_failure_of_the_kind_partial():
    """The board's outcome is success or failure: a model's "partial" (part of the goal left undone) becomes a
    failure that keeps its kind, for the episode and for each task of a session, and the comparison agrees with it."""
    from compare import metrics
    d = to_board.convert(_output("episode_000010", completion={"task_completed": "partial", "reason": "3 of 4 done"}),
                         "demo")
    assert d["completion"]["task_completed"] == "failure" and d["completion"]["failure_kind"] == "partial"
    assert d["_meta"]["task_completed"] == "failure"
    s = to_board.convert(_output("episode_000011", tasks=[{"task": "pour", "outcome": "partial"},
                                                           {"task": "wipe", "outcome": "success"}]), "demo")
    assert [(t["outcome"], t.get("failure_kind")) for t in s["tasks"]] == [("failure", "partial"), ("success", None)]
    assert "failure_kind" not in to_board.convert(_output("episode_000012"), "demo")["completion"]
    L = _output("x")["labels"]
    assert metrics.measure({**L, "completion": {"task_completed": "partial"}}, "demo")["outcome"] == "failure"


def test_convert_a_reply_that_breaks_the_schema():
    """A parsed reply whose lists hold plain strings is shown without them, and the board file counts what was left
    out; an entry whose time is not a number is kept, untimed."""
    out = _output("episode_000008", key_events=["goal reached", {"t_s": "late", "label": "no number"},
                                                {"t_s": 2.0, "label": "kept"}],
                  data_issues=["the camera is dark", {"issue": "kept", "category": "camera_fault", "severity": "low"}],
                  tasks=[{"task": "pour", "outcome": None}], scene={"objects": ["cup", {"name": "plate",
                                                                                        "attributes": [3]}]})
    d = to_board.convert(out, "demo")
    assert [(k["t_s"], k["label"]) for k in d["key_events"]] == [(None, "no number"), (2.0, "kept")]
    assert [i["issue"] for i in d["data_issues"]] == ["kept"] and d["tasks"][0]["outcome"] == ""
    assert d["objects"] == [{"name": "plate", "color": None}]
    assert d["_off_schema"] == {"key_events": 1, "data_issues": 1}
    assert "_off_schema" not in to_board.convert(_output("episode_000009"), "demo")


def test_convert_run_keeps_a_reply_that_failed_and_skips_what_is_not_a_reply(tmp_path):
    """A reply that did not parse and one cut off at the output limit (failed_<episode>.json, with no output beside
    it) are each an episode on the board with no labels and the reply kept; a dry run and an unreadable file are not
    replies. A cut-off reply beside an output of the same episode never replaces it."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "episode_000000.json").write_text(json.dumps(_output("episode_000000")))
    (out / "failed_episode_000000.json").write_text(json.dumps({"episode_dir": "/x/episode_000000",
                                                                "finish_reason": "length"}))
    (out / "episode_000001.json").write_text(json.dumps({
        "episode_dir": "/x/episode_000001", "parse_ok": False, "model": "some/model",
        "labels": {"_raw": "not json " * 1000, "_parse_error": "JSONDecodeError: Expecting value"},
        "config": {"views": ["exo"], "timesteps_s": [0.0, 1.0]}, "usage": {"est_cost_usd": 0.2}}))
    (out / "episode_000002.json").write_text(json.dumps({"episode_dir": "/x/episode_000002", "dry_run": True}))
    (out / "episode_000003.json").write_text("{")
    (out / "failed_episode_000004.json").write_text(json.dumps({
        "episode_dir": "/x/episode_000004", "finish_reason": "length", "content_tail": '{"timeline": [[0.0, 1',
        "usage": {"est_cost_usd": 0.5, "completion_tokens": 64000}, "config": {"views": ["exo"]}}))
    n, skipped = to_board.convert_run(out, tmp_path / "board", "demo")
    assert n == 4 and skipped == ["episode_000002.json: dry run"]
    assert sorted(p.name for p in (tmp_path / "board").iterdir()) == [
        "episode_000000.json", "episode_000001.json", "episode_000003.json", "episode_000004.json"]
    # an output file that does not read is an episode with no labels, saying why
    bad = json.loads((tmp_path / "board" / "episode_000003.json").read_text())
    assert bad["_label_failed"]["status"] == "unreadable" and "episode_000003.json" in bad["_label_failed"]["error"]
    ok = json.loads((tmp_path / "board" / "episode_000000.json").read_text())
    assert "_label_failed" not in ok and ok["completion"]["task_completed"] == "success"
    un = json.loads((tmp_path / "board" / "episode_000001.json").read_text())
    assert un["_label_failed"]["status"] == "unparsed" and un["_label_failed"]["parse_error"].startswith("JSONDecode")
    assert un["_label_failed"]["raw_head"] == ("not json " * 1000)[:3000] and un["_label_failed"]["raw_chars"] == 9000
    assert un["event_labels"] == [] and un["camera_views"] == ["exo"] and un["_usage"]["est_cost_usd"] == 0.2
    cut = json.loads((tmp_path / "board" / "episode_000004.json").read_text())
    assert cut["_label_failed"] == {"status": "cut_off", "out_tokens": 64000, "tail": '{"timeline": [[0.0, 1'}
    assert cut["_meta"]["episode_id"] == "episode_000004" and cut["_usage"]["est_cost_usd"] == 0.5


# ---------------------------------------------------------------- rules

def _issue(cat: str, sev: str, text: str = "an issue") -> dict:
    return {"issue": text, "category": cat, "severity": sev}


def test_no_task_text_rule_only_where_no_text_was_given():
    issue = _issue("missing_instruction", "medium", "the clip ships no instruction")
    d = {"data_issues": [dict(issue)], "_meta": {"given_prompt": None}}
    board_build.apply_rules(d, {}, [NO_TASK_TEXT])
    assert d["data_issues"] == [] and d["_excluded"][0]["excluded_by"] == "no_task_text"
    d = {"data_issues": [dict(issue)], "_meta": {"given_prompt": "Pour the water."}}
    board_build.apply_rules(d, {}, [NO_TASK_TEXT])
    assert d["data_issues"] == [issue]


def test_cap_when_caps_only_what_another_finding_already_counts():
    unfinished = rules.GENERAL[0]
    d = {"data_issues": [_issue("instruction_mismatch", "medium")],
         "operator_mistakes": [_issue("incomplete_task", "high"), _issue("dropped_object", "high")]}
    board_build.apply_rules(d, {}, [unfinished])
    capped, other = d["operator_mistakes"]
    assert capped["severity"] == "low" and capped["model_severity"] == "high" and capped["capped_by"]
    assert other["severity"] == "high" and "model_severity" not in other
    d = {"data_issues": [_issue("instruction_mismatch", "low")],
         "operator_mistakes": [_issue("incomplete_task", "high")]}
    board_build.apply_rules(d, {}, [unfinished])
    assert d["operator_mistakes"][0]["severity"] == "high"          # a minor mismatch counts nothing
    undone = rules.GENERAL[1]
    d = {"completion": {"task_completed": "success_then_undone"},
         "operator_mistakes": [_issue("goal_undone", "medium")]}
    board_build.apply_rules(d, {}, [undone])
    assert d["operator_mistakes"][0]["severity"] == "low"
    d = {"completion": {"task_completed": "success"}, "operator_mistakes": [_issue("goal_undone", "medium")]}
    board_build.apply_rules(d, {}, [undone])
    assert d["operator_mistakes"][0]["severity"] == "medium"


def test_severity_cap_drop_check_fixed_window_and_unknown_kinds():
    idle = rules.BY_RIG["ego_head"][0]
    d = {"data_issues": [_issue("idle_stretch", "high"), _issue("camera_fault", "high"), _issue("idle_stretch", "low")]}
    board_build.apply_rules(d, {}, [idle])
    assert [i["severity"] for i in d["data_issues"]] == ["low", "high", "low"]
    assert d["data_issues"][0]["model_severity"] == "high" and "model_severity" not in d["data_issues"][2]
    d = {"dataset_checks": {"gripper_channels": {"flagged": True}, "stream_pairing": {"crossed": False}}}
    board_build.apply_rules(d, {}, [{"kind": "drop_check", "check": "gripper_channels", "reason": "one-armed tasks"}])
    assert list(d["dataset_checks"]) == ["stream_pairing"]
    assert d["_withheld_checks"]["gripper_channels"] == {"reason": "one-armed tasks", "result": {"flagged": True}}
    window = {"kind": "fixed_window", "tags": ["truncated_start"], "window_s": 180}
    for secs, excluded in ((179.0, True), (170.0, False)):
        d = {"data_issues": [_issue("truncated_start", "medium"), _issue("camera_fault", "medium")]}
        board_build.apply_rules(d, {"duration_s": secs}, [window])
        assert [i["category"] for i in d["data_issues"]] == (["camera_fault"] if excluded else
                                                             ["truncated_start", "camera_fault"])
        assert ("_packaging" in d) is excluded
    with pytest.raises(ValueError):
        board_build.apply_rules({}, {}, [{"kind": "no_such_rule"}])


def test_quickstart_manifest_lists_each_rigs_rules():
    """The template a user copies: every dataset carries exactly its rig's rules."""
    m = json.loads((REPO / "configs" / "quickstart" / "board.json").read_text())
    rig = {"molmo": "teleop_arms", "fastumi": "handheld_gripper", "openaoe": "ego_head"}
    assert [e["dataset"] for e in m["datasets"]] == list(rig)
    for e in m["datasets"]:
        assert e["rules"] == rules.rules_for(rig[e["dataset"]])
        assert e["run"] == f"../../runs/{e['dataset']}/latest"


# ---------------------------------------------------------------- families

def test_families_classify_and_the_counting_rule():
    fam = Families()
    d = {"dataset": "galaxea",
         "data_issues": [_issue("instruction_mismatch", "medium"), _issue("some_new_tag", "high"),
                         _issue("lighting", "low", "a rectangular blur patch over the table"),
                         _issue("camera_fault", "low"), {"category": "no_text", "severity": "high"}],
         "operator_mistakes": [_issue("dropped_object", "medium"), _issue("failed_grasp", "medium"),
                               _issue("failed_grasp", "high", "missed twice more")],
         "completion": {"task_completed": "success_then_undone"},
         "dataset_checks": {"stream_pairing": {"crossed": True}, "gripper_channels": {"flagged": True}}}
    c = fam.classify(d)
    counted = {k: [i["issue"] for i in v] for k, v in c["counted"].items()}
    assert counted == {"instruction-mismatch": ["an issue"], "d:Some new tag": ["an issue"],
                       "m:Dropped object": ["an issue"], "m:Missed grasp": ["missed twice more"],
                       "streams-crossed": [], "undone": []}
    # gripper-flat counts only on the datasets it lists; a family counted elsewhere is not also minor
    assert set(c["minor"]) == {"privacy-blur", "d:Camera fault"}
    assert fam.classify({**d, "dataset": "fastumi"})["counted"]["gripper-flat"] == []
    assert fam.counts("operator_mistakes", _issue("dropped_object", "medium"))
    assert not fam.counts("operator_mistakes", _issue("failed_grasp", "medium"))
    assert fam.counts("operator_mistakes", _issue("failed_grasp", "high"))
    assert fam.counts("data_issues", _issue("anything", "medium")) and not fam.counts("data_issues", _issue("x", "low"))
    assert fam.catalog()["recorded-jump"] == {"name": "Recorded pose jumps, video does not", "list": "data",
                                              "check": True}
    # one tag the model uses for several problems (camera_fault) is split by what the issue says; one that says
    # none of them keeps the tag's own name
    cam = lambda text: fam.family_of("data_issues", _issue("camera_fault", "medium", text), "egocentric100k")
    assert cam("The wearer takes the camera off and sets it down facing the ceiling.") == "camera-fault"
    assert cam("Fingers cover most of the lens for three seconds.") == "camera-fault"
    assert cam("The left camera repeats an unchanged image for the whole episode.") == "camera-image"
    assert cam("The right stream shows No Signal for a second.") == "camera-image"
    assert cam("Persistent glare obscures the circuit board.") == "low-light"
    assert cam("an issue") == "d:Camera fault"
    assert fam.catalog()["camera-fault"]["name"] == "Camera turned away or covered"
    ego = {"_rig": "ego_head", "event_labels": [{"t_s": 0.0, "end_s": 4.0, "hands_visible": False},
                                                {"t_s": 2.0, "end_s": 6.0, "hands_visible": False},
                                                {"t_s": 6.0, "end_s": 9.0, "hands_visible": True}]}
    assert fam.hands_hidden_seconds(ego) == 6.0 and fam.hands_hidden_seconds({"_rig": "teleop_arms"}) is None


def test_capture_check_names_come_from_the_code():
    from board.build import capture_names
    cq = {"checks": [{"check": "jump_return_event", "name": "an older name", "group": "Motion", "status": "fired"},
                     {"check": "not_a_check", "name": "kept", "group": "Other", "status": "clear"}], "flags": []}
    rows = capture_names(cq)["checks"]
    assert rows[0]["name"] == "Recorded pose jumps away and back" and rows[1]["name"] == "kept"
    assert capture_names({"source": "x"}) == {"source": "x"}


# ---------------------------------------------------------------- a board from its manifest

def _run(runs: Path, name: str, outcome: str, relation: str, kind: str = "full", status: str = "done") -> Path:
    run = runs / name
    (run / "out").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"run_id": name, "code": "abc1234", "kind": kind, "status": status,
                                              "slice": "demo", "cost_usd": 1.0}))
    for i in range(2):
        (run / "out" / f"episode_{i:06d}.json").write_text(json.dumps(_output(
            f"episode_{i:06d}", completion={"task_completed": outcome}, goal_alignment={"relation": relation},
            data_issues=[_issue("human_intervention", "medium", "a person moves the cup"),
                         _issue("missing_instruction", "medium", "no instruction")],
            operator_mistakes=[_issue("incomplete_task", "high", "left unfinished")])))
    (run / "out" / "failed_episode_000001.json").write_text(json.dumps({"episode_dir": "/x/episode_000001"}))
    return run


def _episodes(root: Path, profile: str = "teleop_arms", dataset: str = "you/your-own-dataset") -> Path:
    for i in range(2):
        ep = root / f"episode_{i:06d}"
        ep.mkdir(parents=True)
        (ep / "context.json").write_text(json.dumps({
            "dataset": dataset, "profile": profile, "fps": 30, "n_state_frames": 300,
            "stream_pairing": {"crossed": i == 1}, "gripper_channels": {"flagged": True},
            "annotation_subtasks": [{"t0": 0, "t1": 4.5, "label": "reach for the cup"}, {"t0": 5, "label": "open"}]}))
    return root


def test_board_builds_from_the_latest_run(tmp_path):
    runs = tmp_path / "runs" / "demo"
    _run(runs, "20260101-0000_full_abc1234", "failure", "aligned")
    _run(runs, "20260102-0000_full_abc1234", "success", "different")
    _run(runs, "20260103-0000_dry_abc1234", "success", "aligned", kind="dry")           # never the latest
    _run(runs, "20260104-0000_full_abc1234", "success", "aligned", status="running")    # not finished
    _episodes(tmp_path / "episodes" / "demo")
    board = tmp_path / "boards" / "demo"
    (board / "qa").mkdir(parents=True)
    (board / "qa" / "episode_999999.json").write_text("{}")                # a stale file from an earlier build
    manifest = {"board": "demo", "datasets": [
        {"dataset": "demo", "run": "../../runs/demo/latest", "episodes": "../../episodes/demo",
         "rules": rules.rules_for("teleop_arms") + [{"kind": "drop_check", "check": "gripper_channels",
                                                     "reason": "one-armed tasks"}]}]}
    (board / "manifest.json").write_text(json.dumps(manifest))
    built = board_build.build(board)
    assert built["counts"] == {"demo": {"run_id": "20260102-0000_full_abc1234", "episodes": 2}}
    on_disk = json.loads((board / "BUILT.json").read_text())
    assert on_disk == built
    assert on_disk["manifest"]["datasets"][0]["run"] == "../../runs/demo/20260102-0000_full_abc1234"
    assert on_disk["manifest"]["datasets"][0]["rules"] == manifest["datasets"][0]["rules"]
    assert sorted(p.name for p in (board / "qa").iterdir()) == ["episode_000000.json", "episode_000001.json"]
    assert not (board / "compare").exists() and not (board / "hands").exists() and not (board / "qa.new").exists()
    d = json.loads((board / "qa" / "episode_000001.json").read_text())
    assert d["completion"]["task_completed"] == "success"          # the label, not the cut-off reply beside it
    assert d["_run"] == {"run_id": "20260102-0000_full_abc1234", "code": "abc1234", "kind": "full", "slice": "demo"}
    assert d["duration_s"] == 10.0 and d["_rig"] == "teleop_arms" and "duration_estimated" not in d
    assert d["dataset_checks"]["stream_pairing"] == {"crossed": True} and "gripper_channels" not in d["dataset_checks"]
    assert d["_withheld_checks"]["gripper_channels"]["reason"] == "one-armed tasks"
    # a dataset label with no end time is a moment
    assert d["dataset_labels"] == [{"t0": 0.0, "t1": 4.5, "label": "reach for the cup"},
                                   {"t0": 5.0, "t1": 5.0, "label": "open"}]
    # the rules: an instruction was given, so a missing-instruction issue stays; nothing capped without a mismatch
    assert [i["category"] for i in d["data_issues"]] == ["human_intervention", "missing_instruction"]
    assert d["operator_mistakes"][0]["severity"] == "high"
    assert [f["rule"] for f in d["label_consistency"]] == ["outcome_vs_alignment"]
    assert d["_usage"]["est_cost_usd"] == 0.12
    assert "dataset_source" not in d          # a dataset of your own has no entry in board/dataset_sources.json


def test_a_run_whose_replies_all_failed_still_builds_a_board_of_its_episodes(tmp_path):
    """Every reply of the run failed, one not parsing and one cut off: the board still holds both episodes, each with
    its length, its checks from context.json and a data issue saying the model's reply did not parse, which raises
    its family at any severity, and the reply itself. A rerun whose reply failed never replaces a label that parsed."""
    run = tmp_path / "runs" / "demo" / "20260101-0000_full_abc1234"
    (run / "out").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"run_id": run.name, "code": "abc1234", "kind": "upload",
                                              "status": "done", "slice": "demo"}))
    eps = _episodes(tmp_path / "episodes" / "demo")
    (run / "out" / "episode_000000.json").write_text(json.dumps({
        "episode_dir": str(eps / "episode_000000"), "parse_ok": False, "model": "some/model",
        "labels": {"_raw": "{oops", "_parse_error": "JSONDecodeError: x"},
        "dataset_checks": {"timebase": {"sped_up_recording": False}},
        "decode_failed": [{"camera": "exo", "what": "The exo camera could not be decoded at 1 s.", "t0_s": 1.0}],
        "config": {"views": ["exo"], "timesteps_s": [0.0, 1.0]}, "usage": {"est_cost_usd": 0.2}}))
    (run / "out" / "failed_episode_000001.json").write_text(json.dumps({
        "episode_dir": str(eps / "episode_000001"), "finish_reason": "length", "model": "some/model",
        "content_tail": "...", "usage": {"est_cost_usd": 0.5, "completion_tokens": 64000},
        "config": {"views": ["exo"]}}))
    board = tmp_path / "boards" / "demo"
    board.mkdir(parents=True)
    (board / "manifest.json").write_text(json.dumps({"board": "demo", "datasets": [
        {"dataset": "demo", "run": str(run), "episodes": str(eps), "rules": rules.rules_for("teleop_arms")}]}))
    built = board_build.build(board)
    assert built["counts"]["demo"]["episodes"] == 2
    fam = Families()
    for name, kind, status in (("episode_000000", "model_reply_unparsed", "unparsed"),
                               ("episode_000001", "model_reply_cut_off", "cut_off")):
        d = json.loads((board / "qa" / f"{name}.json").read_text())
        assert d["_label_failed"]["status"] == status and d["duration_s"] == 10.0
        assert "stream_pairing" in d["dataset_checks"]
        issue = next(x for x in d["dataset_checks"]["reader_issues"] if x["kind"] == kind)
        assert issue["family"] == "label-failed" and "did not parse" in issue["what"]
        assert "label-failed" in fam.classify(d)["not_counted"] and "label-failed" not in fam.classify(d)["counted"]
    d0 = json.loads((board / "qa" / "episode_000000.json").read_text())
    assert [x["kind"] for x in d0["dataset_checks"]["reader_issues"]] == ["model_reply_unparsed",
                                                                          "camera_decode_failed"]
    # a later rerun whose reply failed leaves the earlier parsed label in place
    good = _run(tmp_path / "runs" / "good", "20260101-0000_full_abc1234", "success", "aligned")
    sl = tmp_path / "episodes" / "rr"
    sl.mkdir()
    (sl / "episode_000000").symlink_to(eps / "episode_000000")
    rr = tmp_path / "runs" / "rr" / "20260102-0000_full_abc1234"
    (rr / "out").mkdir(parents=True)
    (rr / "run.json").write_text(json.dumps({"run_id": rr.name, "code": "abc1234", "kind": "full", "status": "done",
                                             "slice": str(sl)}))
    (rr / "out" / "episode_000000.json").write_text(json.dumps({
        "episode_dir": str(sl / "episode_000000"), "parse_ok": False, "labels": {"_raw": "{", "_parse_error": "x"}}))
    (board / "manifest.json").write_text(json.dumps({"board": "demo", "datasets": [
        {"dataset": "demo", "run": str(good), "episodes": str(eps), "reruns": [{"run": str(rr), "why": "again"}]}]}))
    board_build.build(board)
    d0 = json.loads((board / "qa" / "episode_000000.json").read_text())
    assert "_label_failed" not in d0 and d0["_run"]["run_id"] == good.name


def test_public_datasets_carry_their_publisher_and_license(tmp_path):
    """Each episode of a public dataset names where its footage comes from, by the Hub repository its context.json
    records (not by the board's name for the dataset), so every download carries it."""
    _run(tmp_path / "runs" / "demo", "20260102-0000_full_abc1234", "success", "aligned")
    _episodes(tmp_path / "episodes" / "demo", dataset="RogersPyke/Galaxea-Open-World-Dataset_10K_20260123")
    board = tmp_path / "board"
    board.mkdir()
    (board / "manifest.json").write_text(json.dumps({"board": "demo", "datasets": [
        {"dataset": "anything", "run": "../runs/demo/latest", "episodes": "../episodes/demo"}]}))
    board_build.build(board)
    d = json.loads((board / "qa" / "episode_000000.json").read_text())
    assert d["dataset_source"] == {"name": "Galaxea Open-World", "publisher": "Galaxea", "license": "CC BY-NC-SA 4.0",
                                   "hub": "https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset",
                                   "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/"}
    sources = json.loads((REPO / "board" / "dataset_sources.json").read_text())
    adapters = {importlib.import_module(f"prepare.{n}").REPO for n in ("molmo", "abc130k", "galaxea", "habit",
                "fastumi", "realomin", "egocentric100k", "genhumanego", "openaoe")}
    for key, s in sources.items():
        if not key.startswith("_"):
            assert set(s) == {"name", "hub", "publisher", "license", "license_url"}
            assert s["hub"].startswith("https://huggingface.co/datasets/") and s["license_url"].startswith("https://")
    assert adapters <= set(sources), "every dataset prepare/ reads has its publisher and license"


def test_board_without_context_estimates_the_length(tmp_path):
    _run(tmp_path / "runs" / "demo", "20260102-0000_full_abc1234", "success", "aligned")
    board = tmp_path / "board"
    board.mkdir()
    (board / "manifest.json").write_text(json.dumps({"board": "demo", "datasets": [
        {"dataset": "demo", "run": str(tmp_path / "runs" / "demo" / "20260102-0000_full_abc1234"),
         "episodes": str(tmp_path / "no_episodes"), "file_prefix": "demo_"}]}))
    built = board_build.build(board)
    assert built["manifest"]["datasets"][0]["run"].endswith("20260102-0000_full_abc1234")
    d = json.loads((board / "qa" / "episode_demo_000000.json").read_text())
    assert d["duration_s"] == 3.0 and d["duration_estimated"] is True       # last timestep plus one step
    assert d["_meta"]["episode_id"] == "episode_demo_000000" and d["_meta"]["run_episode"] == "episode_000000"


def test_two_entries_cannot_write_one_episode(tmp_path):
    _run(tmp_path / "runs" / "demo", "20260102-0000_full_abc1234", "success", "aligned")
    entry = {"dataset": "demo", "run": "../runs/demo/latest", "episodes": "../episodes"}
    board = tmp_path / "board"
    board.mkdir()
    (board / "manifest.json").write_text(json.dumps({"board": "demo", "datasets": [entry, dict(entry)]}))
    with pytest.raises(RuntimeError, match="two manifest entries"):
        board_build.build(board)
    (board / "manifest.json").write_text(json.dumps({"board": "demo",
                                                     "datasets": [entry, dict(entry, file_prefix="b_")]}))
    assert board_build.build(board)["counts"]["demo"]["episodes"] == 2
    assert len(list((board / "qa").iterdir())) == 4


def test_comparisons_are_kept_apart_and_measured(tmp_path):
    """Two models over the board's episodes: their labels go to compare/, never qa/, with the metrics."""
    _run(tmp_path / "runs" / "demo", "20260102-0000_full_abc1234", "success", "aligned")
    eps = _episodes(tmp_path / "episodes" / "demo")
    other = tmp_path / "runs" / "compare" / "20260103-0000_full_abc1234_other"
    (other / "out").mkdir(parents=True)
    (other / "run.json").write_text(json.dumps({"run_id": other.name, "code": "abc1234", "kind": "full",
                                                "status": "done", "slice": str(eps),
                                                "command": ["--model", "some/model"]}))
    (other / "out" / "episode_000000.json").write_text(json.dumps({
        "episode_dir": str(eps / "episode_000000"), "parse_ok": True,
        "labels": {"completion": {"task_completed": "failure"}, "timeline": [], "data_issues": []},
        "usage": {"est_cost_usd": 0.1, "latency_s": 3.0}}))
    (other / "out" / "episode_000001.json").write_text(json.dumps({
        "episode_dir": str(eps / "episode_000001"), "parse_ok": False, "labels": {"_raw": "{", "_parse_error": "x"},
        "usage": {"est_cost_usd": 0.1}}))
    board = tmp_path / "boards" / "demo"
    board.mkdir(parents=True)
    (board / "manifest.json").write_text(json.dumps({"board": "demo", "datasets": [
        {"dataset": "demo", "run": "../../runs/demo/latest", "episodes": "../../episodes/demo"}],
        "comparisons": [{"key": "other", "name": "Other",
                         "run": "../../runs/compare/20260103-0000_full_abc1234_other"}]}))
    built = board_build.build(board)
    assert built["comparisons"]["models"] == {"other": {"run_id": other.name, "files": 2}}
    assert sorted(p.name for p in (board / "qa").iterdir()) == ["episode_000000.json", "episode_000001.json"]
    unparsed = json.loads((board / "compare" / "other" / "episode_000001.json").read_text())
    assert unparsed["_compare"]["status"] == "unparsed" and unparsed["_usage"]["est_cost_usd"] == 0.1
    parsed = json.loads((board / "compare" / "other" / "episode_000000.json").read_text())
    assert parsed["completion"]["task_completed"] == "failure" and parsed["duration_s"] == 10.0
    # the reference is the board's own labels: named after their model, with no files of its own under compare/
    index = json.loads((board / "compare" / "index.json").read_text())
    assert index["episodes"]["episode_000000.json"] == {"other": "parsed"}
    assert index["reference"]["key"] == "board" and index["reference"]["name"] == "model"   # some/model, unpinned
    assert [p.name for p in (board / "compare").iterdir() if p.is_dir()] == ["other"]
    m = json.loads((board / "compare" / "metrics.json").read_text())
    s = m["summary"]["all"]["responses"]
    assert s["other"]["parse_share"] == 0.5 and s["board"]["parse_share"] == 1.0 and s["board"]["asked"] == 2
    assert m["summary"]["all"]["agreement"]["board"]["other"]["outcome"] == 0.0     # success on the board, failure
    assert m["models"][0]["reference"] and m["main"] == ["board", "other"]
    # a manifest without comparisons leaves none behind
    manifest = json.loads((board / "manifest.json").read_text())
    (board / "manifest.json").write_text(json.dumps({k: v for k, v in manifest.items() if k != "comparisons"}))
    board_build.build(board)
    assert not (board / "compare").exists()


# ---------------------------------------------------------------- hand pose

def _clip(path: Path, n: int, w: int = 64, h: int = 36) -> None:
    import av
    c = av.open(str(path), "w")
    s = c.add_stream("mpeg4", rate=30)
    s.width, s.height, s.pix_fmt = w, h, "yuv420p"
    s.time_base = Fraction(1, 15360)
    for k in range(n):
        fr = av.VideoFrame.from_ndarray(np.full((h, w, 3), k % 256, np.uint8), format="rgb24")
        fr.pts = k * 512
        fr.time_base = s.time_base
        for pkt in s.encode(fr):
            c.mux(pkt)
    for pkt in s.encode():
        c.mux(pkt)
    c.close()


def _keypoint_run(src: Path, key: str, n: int) -> None:
    """A keypoint run in source pixels twice the board clip's size: a bent path, so some frames are stored in full,
    and the left hand gone for a while."""
    rng = np.random.default_rng(0)
    kp = [[float(64 + 3 * i + (7 * np.sin(i) if c % 2 else 0) + rng.normal(0, 0.2)) for c in range(42)]
          for i in range(n)]
    conf = [0.9] * 8 + [0.1] * 5 + [0.8] * (n - 13)
    (src / key).mkdir(parents=True)
    (src / key / "hands2d.json").write_text(json.dumps({
        "video": {"width": 128, "height": 72}, "joints": list(range(21)), "edges": [[0, 1]],
        "hands": {"left": {"kp": kp, "conf": conf}, "right": {"kp": kp, "conf": [0.0] * n}}}))
    (src / "index.json").write_text(json.dumps({key: {"hands2d": f"{key}/hands2d.json"}}))


@pytest.mark.skipif(not shutil.which("ffprobe"), reason="no ffprobe")
def test_hand_pose_files_round_trip(tmp_path):
    """Keypoints in source pixels become the board's hand pose file, and decode back within its tolerance."""
    n = 23
    clips = tmp_path / "clips"
    clips.mkdir()
    _clip(clips / "episode_x.mp4", n)
    qa = tmp_path / "qa"
    qa.mkdir()
    (qa / "episode_x.json").write_text(json.dumps({"dataset": "demo", "_rig": "ego_head",
                                                   "_meta": {"episode_id": "episode_x"}}))
    src = tmp_path / "run"
    _keypoint_run(src, "demo/episode_x", n)
    out = tmp_path / "hands"
    out.mkdir()
    res = hands.build(src, qa, clips, out, jobs=1)
    assert res["written"] == 1 and not res["skipped"]
    doc = json.loads((out / "episode_x.json").read_text())
    assert doc["left"]["spans"] == [[0, 8], [13, n]] and doc["right"]["spans"] == []
    assert "run" not in doc["source"]                        # published with the board: no run, no local path
    assert hands.verify(src, qa, out)["max_error_px"] <= doc["tol"]


@pytest.mark.skipif(not shutil.which("ffprobe"), reason="no ffprobe")
def test_board_build_writes_hand_pose_for_prefixed_head_camera_episodes(tmp_path):
    """The manifest's "hands" through board build: a head-camera dataset with a file_prefix, whose clips carry the
    board's name and whose keypoints carry the run's."""
    n = 23
    _run(tmp_path / "runs" / "ego", "20260102-0000_full_abc1234", "success", "aligned")
    eps = _episodes(tmp_path / "episodes" / "ego", profile="ego_head", dataset="inclusionAI/OpenAoE-2000h")
    # the dataset's own video, in the keypoints' pixels, which the download is timed against
    ep = eps / "episode_000000"
    _clip(ep / "raw_video.mp4", n, 128, 72)
    (ep / "sources.json").write_text(json.dumps({"exo": {"packed": str(ep / "raw_video.mp4"), "base_s": 0.0}}))
    ctx = json.loads((ep / "context.json").read_text())
    (ep / "context.json").write_text(json.dumps({**ctx, "source": {"clip": "raw_0001", "path": "/local/raw_0001"}}))
    clips = tmp_path / "clips"
    clips.mkdir()
    _clip(clips / "episode_ego_000000.mp4", n)
    _keypoint_run(tmp_path / "keypoints", "ego/episode_000000", n)
    board = tmp_path / "boards" / "ego"
    board.mkdir(parents=True)
    (board / "manifest.json").write_text(json.dumps({
        "board": "ego", "datasets": [{"dataset": "ego", "run": "../../runs/ego/latest",
                                      "episodes": "../../episodes/ego", "file_prefix": "ego_"}],
        "hands": {"src": "../../keypoints", "clips": "../../clips"}}))
    built = board_build.build(board)
    assert built["hands"]["written"] == 1 and built["hands"]["head_camera_files_without_keypoints"] == 1
    assert sorted(p.name for p in (board / "hands").iterdir()) == ["episode_ego_000000.json"]
    assert hands.verify(tmp_path / "keypoints", board / "qa", board / "hands")["files"] == 1
    assert json.loads((board / "hands" / "episode_ego_000000.json").read_text())["source"]["licence"] == hands.LICENCE
    # the download: every frame of the dataset's video in its own pixels, the licence and the dataset's license
    assert built["hands"]["keypoints"]["written"] == 1
    kidx = json.loads((board / "hand_keypoints" / "index.json").read_text())
    assert kidx["licence"] == hands.LICENCE and list(kidx["files"]) == ["episode_ego_000000.json"]
    doc = json.loads((board / "hand_keypoints" / "episode_ego_000000.json").read_text())
    assert doc["licence"] == hands.LICENCE and doc["joints"] == list(range(21))
    assert doc["episode"]["dataset"] == "OpenAoE-2000h (inclusionAI)" and doc["episode"]["dataset_license"] == {
        "name": "Open-AoE Dataset License",
        "url": "https://huggingface.co/datasets/inclusionAI/OpenAoE-2000h/blob/main/LICENSE"}
    assert doc["episode"]["board_file"] == "episode_ego_000000.json" and doc["video"]["source"] == {"clip": "raw_0001"}
    assert (doc["video"]["width"], doc["video"]["height"], doc["video"]["frames"]) == (128, 72, n)
    assert doc["frames"]["t"][:2] == [0.0, pytest.approx(1 / 30, abs=1e-6)] and "run" not in doc["model"]
    assert hands.verify_keypoints(tmp_path / "keypoints", board / "qa", {"episode_ego_000000.json": ep},
                                  board / "hand_keypoints") == {"files": 1, "exact": True}


def test_keypoint_licence_is_the_attribution_wording():
    """The notice in every hand pose file and download, word for word: the model's non-commercial term passed on
    (never a new restriction of ours), its creator, checkpoints, license and disclaimer, what was done to its output,
    MANO, and the video's own license."""
    assert hands.LICENCE == (
        "Non-commercial use only, because the model that predicted these keypoints is licensed for non-commercial use. "
        "They were predicted by ACE-Ego-Hand (Yufei Liu et al., ACE Robotics, arXiv:2608.20308) with its released "
        "checkpoints (https://huggingface.co/acerobotics2025/ACE-Ego-Hand), which are licensed CC BY-NC 4.0 "
        "(https://creativecommons.org/licenses/by-nc/4.0/) and provided as is, without warranties (see the license's "
        "disclaimer). Pantheon ran the model and processed its output, blending overlapping windows, smoothing over "
        "time and mapping the points back to the video's own pixels. The model uses MANO (Romero, Tzionas and Black, "
        "2017; https://mano.is.tue.mpg.de/license.html), which is licensed for non-commercial scientific research "
        "only. Credit ACE-Ego-Hand when you use these keypoints. The video belongs to its dataset and is under that "
        "dataset's license (episode.dataset_license).")


def test_clip_sizes_follow_where_the_page_shows_each_camera():
    from board import clips
    # never larger than the source: a small camera keeps its size; the main one is scaled to fit 1920x1080 when larger
    assert clips.clip_size(456, 256, True)[:2] == (456, 256)
    assert clips.clip_size(640, 480, True)[:2] == (640, 480)
    assert clips.clip_size(455, 255, True)[:2] == (454, 254)
    assert clips.clip_size(1920, 1080, True)[:2] == (1920, 1080)
    assert clips.clip_size(1920, 1200, True)[:2] == (1728, 1080)
    assert clips.clip_size(1080, 1920, True)[:2] == (608, 1080)
    # a side camera: never upscaled, at most 1280x1080
    assert clips.clip_size(640, 360, False)[:2] == (640, 360)
    assert clips.clip_size(1600, 1300, False)[:2] == (1280, 1040)
    for w, h in ((456, 256), (1600, 1300), (1920, 1200), (637, 479)):
        for main in (True, False):
            cw, ch, _ = clips.clip_size(w, h, main)
            assert cw % 2 == 0 and ch % 2 == 0
            assert cw <= w and ch <= h
            assert abs(cw / ch - w / h) <= 0.005 * w / h   # the hand overlay accepts 0.5%
    assert clips.main_cam({"left": {}, "right": {}}) == "left"
    assert clips.main_cam({"exo": {}, "left": {}}) == "exo"
    # a seek decodes from a keyframe at most KEY_S back; every frame keeps its source time
    assert "-vf" not in clips.video_args(456, 256, True, 2)          # a clip at its source size is not resampled
    assert "scale=1728:1080:flags=lanczos,setsar=1" in clips.video_args(1920, 1200, True, 2)
    args = clips.video_args(1920, 1080, True, 2)
    assert args[args.index("-enc_time_base") + 1] == "demux"
    assert f"expr:gte(t,n_forced*{clips.KEY_S})" in args


def test_a_camera_that_started_late_is_shifted_onto_the_episode_clock(tmp_path):
    from board import clips
    np.savez(tmp_path / "times.npz", left=np.array([0.0, 0.033, 0.067]), right=np.array([2.031, 2.064]),
             left_pts=np.array([0, 1, 2]), right_pts=np.array([0, 1]))
    sources = {"left": {}, "right": {}}
    assert clips.start_offsets(tmp_path, sources) == {"right": (2.031, 0)}
    # a camera that started first drops its frame from more than half a frame before the main camera's first (its
    # next is 10 ms from it, under half a frame), and one camera or no real times means nothing to shift
    np.savez(tmp_path / "times.npz", left=np.array([0.04, 0.07]), right=np.array([0.0, 0.03]))
    assert clips.start_offsets(tmp_path, sources) == {"right": (0.0, 1)}
    assert clips.start_offsets(tmp_path, {"left": {}}) == {}
    assert clips.start_offsets(tmp_path / "none", sources) == {}


def test_the_progress_readout_reaches_the_goal_when_it_is_reached():
    """tests/progress_points.js on the page's progressPoints, progressAt and progressPct: the chart reads 100% exactly
    from the goal frame the board shows and never before it or without one, on real episodes where the labels counted
    work past the goal (the rice cooker's lid, a second coffee filter) or finished it while the robot waited; an undone
    goal keeps its drop; a session of tasks reads 100% only when every task is done; a parked arm never drags it down."""
    import subprocess
    here = Path(__file__).resolve().parent
    r = subprocess.run([shutil.which("node"), str(here / "progress_points.js"), str(here.parent / "board" / "serve.py")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_a_new_episode_stops_the_old_episodes_videos():
    """Opening another episode empties the old episode's video elements before the new layout is built, so their
    downloads end at once instead of competing with the new footage until the browser collects them (a visitor
    clicking quickly through datasets left 7 videos loading at once and one superseded download running); switching
    source keeps the playing footage, so it is left alone there."""
    from board import serve
    page = serve.INDEX_HTML
    body = page[page.index("function renderEp(d, opts) {"):]
    keep, release = body.index("const keep = "), body.index("if (!keep) {")
    build = body.index("current-ep-src")
    assert keep < release < build
    block = body[release:build]
    for call in ("v.pause()", "v.removeAttribute('src')", "v.load()"):
        assert call in block


def test_the_dashboard_draws_no_issue_in_orange():
    """A problem is crimson and the operator's performance indigo, as on the blog; no literal colour in the page is an
    orange or orange-red hue (0 to 55 degrees, saturated, not near white)."""
    import colorsys
    import re
    src = (Path(__file__).resolve().parent.parent / "board" / "serve.py").read_text()
    found = []
    for m in re.finditer(r"#[0-9a-fA-F]{6}\b|rgba?\(\s*\d+\s*,\s*\d+\s*,\s*\d+", src):
        t = m.group(0)
        r, g, b = (int(t[i:i + 2], 16) for i in (1, 3, 5)) if t.startswith("#") else map(int, re.findall(r"\d+", t)[:3])
        h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
        if s > 0.3 and h * 360 <= 55 and l < 0.93:
            found.append(t)
    assert not found, found


# ---------------------------------------------------------------- the video download (board/serve.py footage)

def test_footage_layout_keeps_the_main_camera_and_never_enlarges_the_others():
    from board import serve
    # Rexair: a portrait scene camera and two landscape wrists share its height in a column beside it
    W, H, cells = serve.footage_layout([(480, 640), (640, 480), (640, 480)])
    assert cells == [(0, 0, 480, 640), (488, 0, 422, 316), (488, 324, 422, 316)]
    assert (W, H) == (910, 640)
    # a tall main camera: the wrists keep their own size, centred in the column
    W, H, cells = serve.footage_layout([(1920, 1080), (640, 480), (640, 480)])
    assert cells[1:] == [(1928, 56, 640, 480), (1928, 544, 640, 480)] and (W, H) == (2568, 1080)
    # one camera is its own frame
    assert serve.footage_layout([(640, 360)]) == (640, 360, [(0, 0, 640, 360)])
    for sizes in ([(456, 256), (640, 480)], [(1280, 720), (1280, 720), (1280, 720)], [(455, 255), (637, 479)]):
        W, H, cells = serve.footage_layout(sizes)
        assert W % 2 == 0 and H % 2 == 0
        for (w, h), (x, y, cw, ch) in zip(sizes, cells):
            assert cw <= w and ch <= h and x + cw <= W and y + ch <= H


def _flash_clip(path: Path, w: int, h: int, n: int, flash: int, start_s: float = 0.0) -> None:
    """n frames of grey at 30 fps with frame `flash` white, the first frame at start_s."""
    import av
    c = av.open(str(path), "w")
    s = c.add_stream("mpeg4", rate=30)
    s.width, s.height, s.pix_fmt = w, h, "yuv420p"
    s.time_base = Fraction(1, 15360)
    for k in range(n):
        fr = av.VideoFrame.from_ndarray(np.full((h, w, 3), 255 if k == flash else 60, np.uint8), format="rgb24")
        fr.pts = round((start_s + k / 30) * 15360)
        fr.time_base = s.time_base
        for pkt in s.encode(fr):
            c.mux(pkt)
    for pkt in s.encode():
        c.mux(pkt)
    c.close()


def _footage_board(tmp_path):
    from board import serve
    clips = tmp_path / "clips"
    (clips / "wrist_left").mkdir(parents=True)
    (clips / "wrist_right").mkdir()
    _flash_clip(clips / "episode_a.mp4", 60, 80, 90, 45)                         # exo: white at 1.5 s
    _flash_clip(clips / "wrist_left" / "episode_a.mp4", 80, 60, 90, 45, 0.5)    # started 0.5 s late: white at 2.0 s
    _flash_clip(clips / "wrist_right" / "episode_a.mp4", 80, 60, 90, 30)        # white at 1.0 s
    serve.MP4_DIR, serve.FOOTAGE_DIR = clips, tmp_path / "footage"
    return serve


def _cell_means(mp4: Path, cells) -> np.ndarray:
    """[frame, cell] mean brightness inside each cell (2 px in from its edges)."""
    import av
    out = []
    with av.open(str(mp4)) as c:
        for fr in c.decode(video=0):
            a = fr.to_ndarray(format="gray")
            out.append([a[y + 2:y + h - 2, x + 2:x + w - 2].mean() for x, y, w, h in cells])
    return np.array(out)


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="no ffmpeg")
def test_a_phone_portrait_video_gets_a_portrait_clip(tmp_path):
    """A phone stores portrait video as landscape frames with a display rotation (an iPhone's 3840x2160 with -90):
    the clip is the upright portrait size, never the frames squashed into the stored landscape size."""
    from board import clips
    src, rotated, out = tmp_path / "land.mp4", tmp_path / "IMG_0001.MOV", tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=2400x1350:rate=30", "-frames:v", "6",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-display_rotation", "-90", "-i", str(src), "-c", "copy", str(rotated)],
                   check=True)
    assert clips.source_size("ffmpeg", str(rotated)) == (1350, 2400, False)
    assert clips.source_size("ffmpeg", str(src)) == (2400, 1350, False)
    clips.extract_one(str(rotated), 0.0, 6, out, "ffmpeg", 1)
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                        "-of", "csv=p=0", str(out)], capture_output=True, text=True, check=True)
    assert tuple(int(x) for x in r.stdout.strip().split(",")[:2]) == (608, 1080)   # 1350x2400 fitted to 1080 tall


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="no ffmpeg")
def test_footage_shows_each_camera_at_its_own_time(tmp_path):
    """The downloaded video shows every camera at the time the page shows it: a camera that started late is black
    until it starts and then on its own clock, and a span starts at its first second."""
    serve = _footage_board(tmp_path)
    W, H, cells = serve.footage_layout([(60, 80), (80, 60), (80, 60)])
    mp4, name = serve.footage("episode_a", 0.5, 2.5)
    assert name == "episode_a_0.5-2.5s.mp4"
    m = _cell_means(mp4, cells)
    assert len(m) == 60                                 # 2 s at 30 fps
    # the flashes land on the frames of their times: exo 1.5 s, left 2.0 s, right 1.0 s, counted from 0.5 s
    assert [int(np.argmax(m[:, i])) for i in range(3)] == [30, 45, 15]
    whole, name = serve.footage("episode_a")
    assert name == "episode_a.mp4"
    m = _cell_means(whole, cells)
    assert len(m) == 90
    assert m[:14, 1].max() < 20 and m[16:40, 1].min() > 40     # the left wrist is black until it starts at 0.5 s
    assert [int(np.argmax(m[:, i])) for i in range(3)] == [45, 60, 30]
    # made once: asking again returns the same file
    assert serve.footage("episode_a")[0] == whole and len(list(serve.FOOTAGE_DIR.glob("*.mp4"))) == 2
    assert serve.footage("episode_none") is None
    assert serve.footage("episode_a", 2.0, 2.0) is None


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="no ffmpeg")
def test_the_server_hands_out_the_video_and_each_camera_as_files(tmp_path):
    import socketserver
    import threading
    import urllib.request
    serve = _footage_board(tmp_path)
    serve.HERE = tmp_path / "qa"
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), serve.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/api/footage?id=episode_a&t0=0.5&t1=2.5&prepare=1") as r:
            j = json.loads(r.read())
        assert j["name"] == "episode_a_0.5-2.5s.mp4" and j["bytes"] > 0
        with urllib.request.urlopen(base + "/api/footage?id=episode_a&t0=0.5&t1=2.5") as r:
            assert r.headers["Content-Disposition"] == 'attachment; filename="episode_a_0.5-2.5s.mp4"'
            assert len(r.read()) == j["bytes"]
        with urllib.request.urlopen(base + "/api/video?id=episode_a&cam=left&download=1") as r:
            assert r.headers["Content-Disposition"] == 'attachment; filename="episode_a_left.mp4"'
        with urllib.request.urlopen(base + "/api/video?id=episode_a&cam=left") as r:
            assert r.headers["Content-Disposition"] is None          # the page's own playback is not a download
        for bad in ("id=../x", "id=episode_a&t0=x", "id=episode_none"):
            try:
                urllib.request.urlopen(base + "/api/footage?" + bad)
                raise AssertionError(bad)
            except urllib.error.HTTPError as e:
                assert e.code in (400, 404), bad
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="no ffmpeg")
def test_an_extra_camera_is_cut_served_and_put_in_the_video_download(tmp_path):
    from board import clips, static
    serve = _footage_board(tmp_path)
    assert clips.clip_path(tmp_path, "e", "extra1") == tmp_path / "extra1" / "e.mp4"
    assert serve.clip_path(tmp_path, "e", "extra2") == tmp_path / "extra2" / "e.mp4"
    assert serve.clip_path(tmp_path, "e", "../x") == tmp_path / "e.mp4"          # anything else is the main camera
    assert static.media_key("extra1") == "extra1" and static.media_key("x") == "exo"
    assert clips.cams_of({"extra2": 1, "right": 1, "extra1": 1, "exo": 1}) == ["exo", "right", "extra1", "extra2"]
    (serve.MP4_DIR / "extra1").mkdir()
    _flash_clip(serve.MP4_DIR / "extra1" / "episode_a.mp4", 80, 60, 90, 75)         # white at 2.5 s
    assert [c for c, _ in serve.footage_cams(serve.MP4_DIR, "episode_a")] == ["exo", "left", "right", "extra1"]
    W, H, cells = serve.footage_layout([(60, 80), (80, 60), (80, 60), (80, 60)])
    mp4, _ = serve.footage("episode_a")
    m = _cell_means(mp4, cells)
    assert [int(np.argmax(m[:, i])) for i in range(4)] == [45, 60, 30, 75]


def _rerun(runs: Path, name: str, slice_dir: Path, outcomes: dict) -> Path:
    """A run over slice_dir (a folder of links to episode folders) whose outputs name their episode by its slice
    folder, as a comparison run's do."""
    run = runs / name
    (run / "out").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"run_id": name, "code": "def5678", "kind": "full", "status": "done",
                                              "slice": str(slice_dir)}))
    for ep, outcome in outcomes.items():
        (run / "out" / f"{ep}.json").write_text(json.dumps({**_output(ep, completion={"task_completed": outcome}),
                                                            "episode_dir": str(slice_dir / ep)}))
    return run


def test_reruns_replace_the_labels_of_the_episodes_they_labelled(tmp_path):
    """A rerun's label replaces the base run's for the episodes it labelled, matched by resolved episode folder
    (the comparison slice names the episode as the board file does), later reruns win, and the episodes of
    another dataset in the same slice are left alone. The reference compare.metrics measures against is the same
    label the board shows."""
    runs = tmp_path / "runs"
    _run(runs / "demo", "20260101-0000_full_abc1234", "failure", "aligned")
    eps = _episodes(tmp_path / "episodes" / "demo")
    other = _episodes(tmp_path / "episodes" / "other")
    sl = tmp_path / "episodes" / "cmp"
    sl.mkdir()
    (sl / "episode_demo_000001").symlink_to(eps / "episode_000001")
    (sl / "episode_000000").symlink_to(other / "episode_000000")         # another dataset's episode_000000
    first = _rerun(runs / "cmp", "20260102-0000_full_def5678_a", sl, {"episode_demo_000001": "partial",
                                                                        "episode_000000": "success"})
    later = _rerun(runs / "cmp", "20260103-0000_full_def5678_b", sl, {"episode_demo_000001": "success"})
    board = tmp_path / "boards" / "demo"
    board.mkdir(parents=True)
    manifest = {"board": "demo", "datasets": [
        {"dataset": "demo", "run": str(runs / "demo" / "20260101-0000_full_abc1234"), "episodes": str(eps),
         "rules": [], "reruns": [{"run": str(first), "why": "a first rerun"}, {"run": str(later), "why": "a later"}]}]}
    (board / "manifest.json").write_text(json.dumps(manifest))
    board_build.build(board)
    d0 = json.loads((board / "qa" / "episode_000000.json").read_text())
    d1 = json.loads((board / "qa" / "episode_000001.json").read_text())
    assert d0["completion"]["task_completed"] == "failure" and d0["_run"]["run_id"] == "20260101-0000_full_abc1234"
    assert d1["completion"]["task_completed"] == "success" and d1["_run"]["run_id"] == later.name
    src = board_build.label_sources(manifest, board)
    assert src["episode_000001.json"] == later / "out" / "episode_demo_000001.json"
    assert src["episode_000000.json"] == runs / "demo" / "20260101-0000_full_abc1234" / "out" / "episode_000000.json"


def test_the_readers_note_and_what_the_model_was_not_shown_reach_the_board(tmp_path):
    """state_note and the source's unused lists, which the prompt states only where they explain an absence, are on
    every episode's board file by kind; an episode the reader read whole carries none."""
    ctx = {"profile": "teleop_arms", "fps": 30, "n_state_frames": 30, "state_kind": "none",
           "state_note": "Labelled from the video: the recorded state has 16 values per frame.",
           "source": {"format": "lerobot v2.1", "unused_cameras": ["observation.images.cam_high_mask"],
                      "unused_signals": ["recorder_time_ns (a clock)"], "unused_arrays": []}}
    d = {}
    board_build.add_context(d, ctx, tmp_path)
    assert d["reader_notes"] == {"state_note": ctx["state_note"],
                                 "left_out": {"cameras": ["observation.images.cam_high_mask"],
                                              "signals": ["recorder_time_ns (a clock)"]}}
    whole = {}
    board_build.add_context(whole, {"profile": "teleop_arms", "fps": 30, "source": {"format": "video files"}}, tmp_path)
    assert "reader_notes" not in whole
    assert "reader_notes" in board_build.CONTEXT_KEYS
    # a value of the wrong type is ignored, never iterated into characters or called
    odd = {"state_note": 7, "source": {"unused_cameras": "observation.images.cam_high_mask", "unused_signals": 3}}
    assert board_build.reader_notes(odd) is None
    assert board_build.reader_notes({"state_note": ["x"], "source": {"unused_arrays": ("a", "b")}}) == {
        "left_out": {"arrays": ["a", "b"]}}


@pytest.mark.skipif(not shutil.which("node"), reason="no node")
def test_the_provenance_line_says_what_the_model_was_not_shown():
    r = subprocess.run([shutil.which("node"), str(REPO / "tests" / "reader_notes.js"),
                        str(REPO / "board" / "serve.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.skipif(not shutil.which("node"), reason="no node")
def test_the_problems_an_episode_was_kept_with_are_drawn_as_recording_checks():
    r = subprocess.run([shutil.which("node"), str(REPO / "tests" / "reader_issues.js"),
                        str(REPO / "board" / "serve.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_each_reader_issue_raises_its_family_at_any_severity(tmp_path):
    """A problem an episode was kept and flagged with (context.json reader_issues) reaches the board's data issues: a
    kind families.json names raises that family, any other kind a data family named after it, and an entry with no
    sentence is left out."""
    ctx = {"profile": "teleop_arms", "fps": 30, "reader_issues": [
        {"kind": "clip_frame_count", "camera": "left", "what": "The left wrist camera video has 29 frames."},
        {"kind": "signal_gap", "signal": "force", "what": "The force signal stops.", "t0_s": 1.0},
        {"kind": "no_sentence", "what": ""}, "not a dict"]}
    d = {}
    board_build.add_context(d, ctx, tmp_path)
    assert [(x["kind"], x["family"]) for x in d["dataset_checks"]["reader_issues"]] == [
        ("clip_frame_count", "clip-frames"), ("signal_gap", "d:Signal gap")]
    fam = Families()
    assert set(fam.classify(d)["counted"]) == {"clip-frames", "d:Signal gap"}
    assert fam.catalog()["camera-undecodable"] == {"name": "Camera video does not decode", "list": "data",
                                                   "check": True}
    none = {}
    board_build.add_context(none, {"profile": "teleop_arms", "fps": 30}, tmp_path)
    assert "dataset_checks" not in none


# every kind of problem a reader, board clips or the board build records (context.json reader_issues and
# board/build.py reader_issues): a fault in the recording counts as a data issue; a model reply that gave no labels
# and a limit of how we read or showed the recording are shown on the episode and never counted
READER_ISSUE_KINDS = {
    "data": ["camera_not_aligned", "camera_not_decodable", "camera_decode_failed", "clip_frame_count",
             "unshown_camera_not_decodable", "depth_not_read", "depth_not_decodable", "depth_clip_partial",
             "signal_bad_cells", "signal_gap", "signal_not_finite", "signal_partial_span", "signal_alignment_assumed",
             "state_filled", "state_unaligned", "state_partial", "table_short", "table_long"],
    "labelling": ["model_reply_unparsed", "model_reply_cut_off", "part_not_labelled", "label_output_unreadable",
                  "no_part_labelled"],
    "handling": ["table_downsampled", "signal_summarised", "camera_not_colour", "depth_clip_failed",
                 "depth_clip_timing", "camera_offset"],
}


def test_only_a_fault_in_the_recording_counts_as_a_data_issue():
    """A job whose replies all failed had every episode under Data issues as "Model reply gave no labels", and a table
    read every so many rows or a signal kept as its lowest, mean and highest value counted as a fault in the
    recording. Each kind lands in its list: a fault in the recording counts, a reply that gave no labels and a limit
    of how we read the recording are kept on the episode (not_counted) and never in the data issue rates."""
    fam = Families()
    for lst, kinds in READER_ISSUE_KINDS.items():
        for kind in kinds:
            d = {"dataset_checks": {"reader_issues": [{"kind": kind, "what": "x",
                                                       "family": fam.reader_family(kind)}]}}
            c = fam.classify(d)
            slug = fam.reader_family(kind)
            assert fam.list_of(slug) == lst, (kind, slug)
            if lst == "data":
                assert slug in c["counted"] and not c["not_counted"], kind
            else:
                assert not c["counted"] and slug in c["not_counted"], kind
    assert fam.catalog()["label-failed"]["list"] == "labelling"
    # a camera that is not colour is a property of the recording, named apart from a limit of ours, and says so
    nc = next(x for x in fam.defs if x["slug"] == fam.reader_family("camera_not_colour"))
    assert nc["slug"] != fam.reader_family("table_downsampled") and "property" in nc.get("why", ""), nc


def test_a_reply_that_gave_no_labels_is_shown_on_its_episode():
    r = subprocess.run([shutil.which("node"), str(REPO / "tests" / "label_failed.js"),
                        str(REPO / "board" / "serve.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_a_dataset_label_with_no_time_is_kept_untimed_and_never_breaks_the_build(tmp_path):
    ctx = {"profile": "ego_head", "fps": 30, "annotation_subtasks": [
        {"t0": None, "t1": None, "label": "wipe the table"}, {"t0": "soon", "t1": 3, "label": "odd start"},
        {"t0": 2, "label": "open"}, {"t0": 1, "t1": 2, "label": ""}]}
    d = {}
    board_build.add_context(d, ctx, tmp_path)
    assert d["dataset_labels"] == [{"t0": None, "t1": None, "label": "wipe the table"},
                                   {"t0": None, "t1": 3.0, "label": "odd start"},
                                   {"t0": 2.0, "t1": 2.0, "label": "open"}]


def test_an_untimed_step_never_breaks_the_hands_out_of_view_total():
    d = {"_rig": "ego_head", "event_labels": [{"t_s": None, "end_s": 4.0, "hands_visible": False},
                                              {"t_s": 1.0, "end_s": 3.0, "hands_visible": False}]}
    assert Families().hands_hidden_seconds(d) == 2.0


def test_the_page_lists_untimed_steps_rules_set_aside_and_withheld_checks():
    r = subprocess.run([shutil.which("node"), str(REPO / "tests" / "set_aside.js"),
                        str(REPO / "board" / "serve.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.skipif(not shutil.which("ffprobe"), reason="no ffprobe")
@pytest.mark.parametrize("n_kp", [21, 22, 24, 25])
def test_hand_keypoints_a_frame_or_two_off_the_clip_are_aligned_not_dropped(tmp_path, n_kp):
    """Keypoints that cover a frame or two more or fewer than the board clip are drawn from the clip's first frame:
    the frames past the keypoints' end have no hand, the keypoints past the clip's end are left out, and the file says
    how they were aligned. More than that is not the same video, and is skipped with the reason."""
    n = 23
    clips = tmp_path / "clips"
    clips.mkdir()
    _clip(clips / "episode_x.mp4", n)
    qa = tmp_path / "qa"
    qa.mkdir()
    (qa / "episode_x.json").write_text(json.dumps({"dataset": "demo", "_rig": "ego_head",
                                                   "_meta": {"episode_id": "episode_x"}}))
    src = tmp_path / "run"
    _keypoint_run(src, "demo/episode_x", n_kp)
    out = tmp_path / "hands"
    out.mkdir()
    res = hands.build(src, qa, clips, out, jobs=1)
    assert res["written"] == 1 and not res["skipped"]
    doc = json.loads((out / "episode_x.json").read_text())
    assert doc["aligned"] == {"keypoint_frames": n_kp, "clip_frames": n}
    assert doc["left"]["spans"] == [[0, 8], [13, min(n, n_kp)]]
    assert hands.verify(src, qa, out)["files"] == 1


@pytest.mark.skipif(not shutil.which("ffprobe"), reason="no ffprobe")
def test_hand_keypoints_of_another_length_are_skipped_with_the_reason(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    _clip(clips / "episode_x.mp4", 23)
    qa = tmp_path / "qa"
    qa.mkdir()
    (qa / "episode_x.json").write_text(json.dumps({"dataset": "demo", "_rig": "ego_head",
                                                   "_meta": {"episode_id": "episode_x"}}))
    _keypoint_run(tmp_path / "run", "demo/episode_x", 30)
    (tmp_path / "hands").mkdir()
    res = hands.build(tmp_path / "run", qa, clips, tmp_path / "hands", jobs=1)
    assert res["written"] == 0 and "keypoints cover 30 frames" in res["skipped"][0]["skip"]


def test_an_output_with_no_parse_flag_is_read_as_the_labels_it_holds():
    """An output written without parse_ok (by hand, or by an older harness) is its labels, never a failed reply; only
    parse_ok false, or a cut-off reply with no labels, is one."""
    out = _output("episode_000020")
    del out["parse_ok"]
    d = to_board.convert(out, "demo")
    assert "_label_failed" not in d and d["completion"]["task_completed"] == "success"
    assert to_board.label_failed({"labels": {}}) is None
    assert to_board.label_failed({"parse_ok": False, "labels": {"_raw": "{"}})["status"] == "unparsed"
    assert to_board.label_failed({"finish_reason": "length"})["status"] == "cut_off"


def test_an_unreadable_output_is_an_episode_with_a_data_issue_and_the_build_names_what_it_skipped(tmp_path):
    run = _run(tmp_path / "runs" / "demo", "20260101-0000_full_abc1234", "success", "aligned")
    (run / "out" / "episode_000001.json").write_text("{not json")
    (run / "out" / "episode_000002.json").write_text(json.dumps({"episode_dir": "/x/episode_000002",
                                                                 "dry_run": True}))
    eps = _episodes(tmp_path / "episodes" / "demo")
    board = tmp_path / "boards" / "demo"
    board.mkdir(parents=True)
    (board / "manifest.json").write_text(json.dumps({"board": "demo", "datasets": [
        {"dataset": "demo", "run": str(run), "episodes": str(eps)}]}))
    built = board_build.build(board)
    assert built["counts"]["demo"]["episodes"] == 2
    assert built["counts"]["demo"]["skipped"] == ["episode_000002.json: dry run"]
    d = json.loads((board / "qa" / "episode_000001.json").read_text())
    (iss,) = [x for x in d["dataset_checks"]["reader_issues"] if x["family"] == "label-failed"]
    assert iss["kind"] == "label_output_unreadable" and "does not read" in iss["what"]
    assert d["duration_s"] == 10.0


def test_the_page_says_when_hand_keypoints_were_laid_a_frame_or_two_off():
    r = subprocess.run([shutil.which("node"), str(REPO / "tests" / "hand_aligned.js"),
                        str(REPO / "board" / "serve.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_video_menu_opens_from_the_row_of_buttons_on_a_narrow_screen():
    """On a narrow screen the episode's buttons wrap to the pane's left edge, and the Video button can sit after another
    (Hand pose): its menu is placed against the row of buttons, never against its own button, so it opens inside the
    pane whatever comes before it (a menu 320 px wide opened from a button 138 px in ran 68 px past a 390 px screen)."""
    from board import serve
    page = serve.render_index("t", {"mode": "api"})
    # the phone's blocks (more than one share the query), where the buttons wrap to the pane's left edge
    narrow = "".join(x.split("\n}\n", 1)[0] for x in page.split("@media (max-width: 599px) {")[1:])
    assert ".vd { position: static; }" in narrow
    assert ".ep-head .ep-head-acts { position: relative;" in narrow
    assert ".vd-menu { right: auto; left: 0; }" in narrow
