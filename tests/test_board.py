"""The board's build side: harness outputs to board files, the rules, the families, and a board built from its
manifest (runs, comparisons, hand pose)."""
from __future__ import annotations

import json
import shutil
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
    assert [(e["t_s"], e["verb_class"], e["event_idx"]) for e in d["event_labels"]] == [(0.0, "reach", 0),
                                                                                        (1.5, "place", 2)]
    assert d["event_labels"][1]["carry_phase"] == "-> plate" and d["event_labels"][1]["end_s"] == 3.0
    assert [k["label"] for k in d["key_events"]] == ["cup on plate"]
    assert d["objects"] == [{"name": "cup", "color": "Red"}, {"name": "plate", "color": None}]
    assert d["completion"]["task_completed"] == "success" and d["_meta"]["engaged_events"] == 2
    assert d["_meta"]["given_prompt"] == "Put the cup on the plate." and d["_meta"]["model"] == "some/model"
    assert d["timesteps_s"] == [0.0, 1.0, 2.0] and d["camera_views"] == ["exo", "left"]
    assert d["_usage"]["est_cost_usd"] == 0.12
    assert d["data_issues"] == [] and d["operator_mistakes"] == [] and d["goal_alignment"] is None


def test_convert_a_reply_that_breaks_the_schema():
    """A parsed reply whose lists hold plain strings or whose times are not numbers is shown without them, and the
    board file counts what was left out."""
    out = _output("episode_000008", key_events=["goal reached", {"t_s": "late", "label": "no number"},
                                                {"t_s": 2.0, "label": "kept"}],
                  data_issues=["the camera is dark", {"issue": "kept", "category": "camera_fault", "severity": "low"}],
                  tasks=[{"task": "pour", "outcome": None}], scene={"objects": ["cup", {"name": "plate",
                                                                                        "attributes": [3]}]})
    d = to_board.convert(out, "demo")
    assert [k["label"] for k in d["key_events"]] == ["kept"]
    assert [i["issue"] for i in d["data_issues"]] == ["kept"] and d["tasks"][0]["outcome"] == ""
    assert d["objects"] == [{"name": "plate", "color": None}]
    assert d["_off_schema"] == {"key_events": 1, "data_issues": 1}
    assert "_off_schema" not in to_board.convert(_output("episode_000009"), "demo")


def test_convert_run_skips_what_is_not_a_label(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "episode_000000.json").write_text(json.dumps(_output("episode_000000")))
    (out / "episode_000001.json").write_text(json.dumps({"episode_dir": "/x/episode_000001", "parse_ok": False}))
    (out / "episode_000002.json").write_text(json.dumps({"episode_dir": "/x/episode_000002", "dry_run": True}))
    (out / "episode_000003.json").write_text("{")
    (out / "failed_episode_000004.json").write_text(json.dumps({"episode_dir": "/x/episode_000004"}))
    n, skipped = to_board.convert_run(out, tmp_path / "board", "demo")
    assert n == 1 and len(skipped) == 3
    assert sorted(p.name for p in (tmp_path / "board").iterdir()) == ["episode_000000.json"]


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
    """The template a user copies: every dataset carries its rig's rules, plus Galaxea's withheld gripper check."""
    m = json.loads((REPO / "configs" / "quickstart" / "board.json").read_text())
    rig = {"galaxea": "teleop_arms", "fastumi": "handheld_gripper", "openaoe": "ego_head"}
    for e in m["datasets"]:
        own = [r for r in e["rules"] if r not in rules.rules_for(rig[e["dataset"]])]
        assert e["rules"][:len(e["rules"]) - len(own)] == rules.rules_for(rig[e["dataset"]])
        assert own == ([{"kind": "drop_check", "check": "gripper_channels", "reason": own[0]["reason"]}]
                       if e["dataset"] == "galaxea" else [])
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
    assert set(c["minor"]) == {"privacy-blur", "camera-fault"}
    assert fam.classify({**d, "dataset": "fastumi"})["counted"]["gripper-flat"] == []
    assert fam.counts("operator_mistakes", _issue("dropped_object", "medium"))
    assert not fam.counts("operator_mistakes", _issue("failed_grasp", "medium"))
    assert fam.counts("operator_mistakes", _issue("failed_grasp", "high"))
    assert fam.counts("data_issues", _issue("anything", "medium")) and not fam.counts("data_issues", _issue("x", "low"))
    assert fam.catalog()["recorded-jump"] == {"name": "Recorded leap the camera never saw", "list": "data",
                                              "check": True}
    ego = {"_rig": "ego_head", "event_labels": [{"t_s": 0.0, "end_s": 4.0, "hands_visible": False},
                                                {"t_s": 2.0, "end_s": 6.0, "hands_visible": False},
                                                {"t_s": 6.0, "end_s": 9.0, "hands_visible": True}]}
    assert fam.hands_hidden_seconds(ego) == 6.0 and fam.hands_hidden_seconds({"_rig": "teleop_arms"}) is None


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


def _episodes(root: Path, profile: str = "teleop_arms") -> Path:
    for i in range(2):
        ep = root / f"episode_{i:06d}"
        ep.mkdir(parents=True)
        (ep / "context.json").write_text(json.dumps({
            "profile": profile, "fps": 30, "n_state_frames": 300,
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
    assert d["dataset_labels"] == [{"t0": 0.0, "t1": 4.5, "label": "reach for the cup"}]
    # the rules: an instruction was given, so a missing-instruction issue stays; nothing capped without a mismatch
    assert [i["category"] for i in d["data_issues"]] == ["human_intervention", "missing_instruction"]
    assert d["operator_mistakes"][0]["severity"] == "high"
    assert [f["rule"] for f in d["label_consistency"]] == ["outcome_vs_alignment"]
    assert d["_usage"]["est_cost_usd"] == 0.12


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
        "comparisons": [{"key": "ref", "name": "Reference", "run": "../../runs/demo/20260102-0000_full_abc1234",
                         "episodes": "../../episodes/demo", "reference": True},
                        {"key": "other", "name": "Other",
                         "run": "../../runs/compare/20260103-0000_full_abc1234_other"}]}))
    built = board_build.build(board)
    assert built["comparisons"]["models"]["other"]["files"] == 2
    assert sorted(p.name for p in (board / "qa").iterdir()) == ["episode_000000.json", "episode_000001.json"]
    unparsed = json.loads((board / "compare" / "other" / "episode_000001.json").read_text())
    assert unparsed["_compare"]["status"] == "unparsed" and unparsed["_usage"]["est_cost_usd"] == 0.1
    parsed = json.loads((board / "compare" / "other" / "episode_000000.json").read_text())
    assert parsed["completion"]["task_completed"] == "failure" and parsed["duration_s"] == 10.0
    index = json.loads((board / "compare" / "index.json").read_text())
    assert index["episodes"]["episode_000000.json"] == {"ref": "board", "other": "parsed"}
    m = json.loads((board / "compare" / "metrics.json").read_text())
    s = m["summary"]["all"]["responses"]
    assert s["other"]["parse_share"] == 0.5 and s["ref"]["parse_share"] == 1.0
    assert m["summary"]["all"]["agreement"]["ref"]["other"]["outcome"] == 0.0
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
    assert doc["source"]["run"] == "run"                     # the run's folder name, never a local path
    assert hands.verify(src, qa, out)["max_error_px"] <= doc["tol"]


@pytest.mark.skipif(not shutil.which("ffprobe"), reason="no ffprobe")
def test_board_build_writes_hand_pose_for_prefixed_head_camera_episodes(tmp_path):
    """The manifest's "hands" through board build: a head-camera dataset with a file_prefix, whose clips carry the
    board's name and whose keypoints carry the run's."""
    n = 23
    _run(tmp_path / "runs" / "ego", "20260102-0000_full_abc1234", "success", "aligned")
    _episodes(tmp_path / "episodes" / "ego", profile="ego_head")
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
