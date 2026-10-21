"""BOARD/additions: fields added to an episode's label by a later pass, beside the run's own output (board/build.py),
and the parser's rule that a goal_alignment saying the work is not the asked work comes with an instruction_mismatch
data issue (label/harness.py derive_instruction_mismatch)."""
from __future__ import annotations

import json
from pathlib import Path

from board import build as board_build
from label import harness

PROV = {"source": "text goal comparison", "model": "openai/gpt-6.1-sol", "reasoning": "medium",
        "request_id": "gen-1", "cost_usd": 0.0016, "argus_sha": "abc1234"}
ALIGN = {"matches_given": False, "relation": "different", "note": "A cloth is folded instead of zipping a bag."}
ISSUE = {"issue": "A cloth is folded instead of zipping a bag.", "category": "instruction_mismatch",
         "severity": "high", "t_s": None, "end_s": None, "evidence": "tasks 1 and 2"}


def _output(ep: Path, **labels) -> dict:
    return {"episode_dir": str(ep), "parse_ok": True, "model": "some/model", "reasoning_effort": "high",
            "given_prompt": "Zip the bag closed.", "prompt_mode": "session",
            "usage": {"est_cost_usd": 0.2}, "config": {"timesteps_s": [0.0, 1.0], "views": ["exo"]},
            "labels": {"task_summary": "The arm folds a cloth.",
                       # the work the request asked for (label/completeness.py): a timeline over its instants
                       "timeline": [{"start_s": 0.0, "end_s": 1.0, "arm": "right", "action": "fold",
                                     "contribution": "advancing", "progress": 1.0}],
                       "tasks": [{"start_s": 0.0, "end_s": 5.0, "task": "Fold the cloth.", "outcome": "success"}],
                       "data_issues": [{"issue": "a person reaches in", "category": "human_intervention",
                                        "severity": "low", "t_s": 1.0, "end_s": 2.0, "evidence": "exo 1 s"}],
                       **labels}}


def _board(tmp_path: Path, **labels) -> Path:
    eps = tmp_path / "episodes"
    run = tmp_path / "run"
    (run / "out").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"run_id": "r1", "code": "abc1234", "kind": "full", "status": "done",
                                              "slice": str(eps)}))
    for i in range(2):
        ep = eps / f"episode_{i:06d}"
        ep.mkdir(parents=True)
        (ep / "context.json").write_text(json.dumps({"dataset": "you/own", "profile": "teleop_arms", "fps": 30,
                                                      "n_state_frames": 300}))
        (run / "out" / f"episode_{i:06d}.json").write_text(json.dumps(_output(ep, **labels)))
    board = tmp_path / "board"
    board.mkdir()
    (board / "manifest.json").write_text(json.dumps({"board": "demo", "datasets": [
        {"dataset": "demo", "run": str(run), "episodes": str(eps)}]}))
    return board


def _add(board: Path, name: str, body) -> None:
    (board / "additions").mkdir(exist_ok=True)
    (board / "additions" / name).write_text(body if isinstance(body, str) else json.dumps(body))


def _qa(board: Path, name: str) -> dict:
    return json.loads((board / "qa" / name).read_text())


def test_an_addition_adds_fields_and_rows_with_provenance_and_leaves_the_run_output_alone(tmp_path):
    board = _board(tmp_path)
    out = tmp_path / "run" / "out" / "episode_000000.json"
    before = out.read_bytes()
    _add(board, "episode_000000.json", {"provenance": PROV, "fields": {"goal_alignment": ALIGN},
                                        "append": {"data_issues": [ISSUE]}})
    built = board_build.build(board)
    d = _qa(board, "episode_000000.json")
    assert d["goal_alignment"] == {**ALIGN, "provenance": PROV}
    assert [i["category"] for i in d["data_issues"]] == ["human_intervention", "instruction_mismatch"]
    assert d["data_issues"][1]["provenance"] == PROV and "provenance" not in d["data_issues"][0]
    assert d["_additions"] == [{"field": "goal_alignment", "provenance": PROV},
                               {"field": "data_issues", "rows": 1, "provenance": PROV}]
    assert "addition_conflicts" not in d
    other = _qa(board, "episode_000001.json")
    assert other["goal_alignment"] is None and "_additions" not in other and len(other["data_issues"]) == 1
    assert out.read_bytes() == before
    assert built["additions"] == {"files": 1, "episodes": 1, "conflicts": [], "unused": []}


def test_an_addition_never_overwrites_a_field_the_label_has_and_reports_it(tmp_path):
    own = {"matches_given": True, "relation": "aligned", "note": ""}
    board = _board(tmp_path, goal_alignment=own)
    _add(board, "episode_000000.json", {"provenance": PROV, "fields": {"goal_alignment": ALIGN,
                                                                       "duration_s": 99.0}})
    built = board_build.build(board)
    d = _qa(board, "episode_000000.json")
    assert d["goal_alignment"] == own and d["duration_s"] == 10.0
    assert sorted(c["field"] for c in d["addition_conflicts"]) == ["duration_s", "goal_alignment"]
    assert "_additions" not in d or all(a["field"] not in ("goal_alignment", "duration_s") for a in d["_additions"])
    assert sorted(c["field"] for c in built["additions"]["conflicts"]) == ["duration_s", "goal_alignment"]


def test_an_added_row_whose_category_the_label_already_reports_is_left_out_and_reported(tmp_path):
    own = {"issue": "The bag is never zipped.", "category": "instruction_mismatch", "severity": "high",
           "t_s": None, "end_s": None, "evidence": "tasks 1 and 2"}
    board = _board(tmp_path)
    out = tmp_path / "run" / "out" / "episode_000000.json"
    r = json.loads(out.read_text())
    r["labels"]["data_issues"].append(own)
    out.write_text(json.dumps(r))
    other = {**ISSUE, "issue": "A drop.", "category": "object_dropped"}
    _add(board, "episode_000000.json", {"provenance": PROV, "fields": {"goal_alignment": ALIGN},
                                        "append": {"data_issues": [ISSUE, other]}})
    built = board_build.build(board)
    d = _qa(board, "episode_000000.json")
    assert [i["category"] for i in d["data_issues"]] == ["human_intervention", "instruction_mismatch", "object_dropped"]
    assert d["data_issues"][1]["issue"] == own["issue"] and "provenance" not in d["data_issues"][1]
    assert d["goal_alignment"] == {**ALIGN, "provenance": PROV}
    assert {"field": "data_issues", "rows": 1, "provenance": PROV} in d["_additions"]
    assert [c["field"] for c in d["addition_conflicts"]] == ["data_issues"]
    assert "instruction_mismatch" in d["addition_conflicts"][0]["why"]
    assert [c["field"] for c in built["additions"]["conflicts"]] == ["data_issues"]


def test_a_changed_addition_rebuilds_its_episode_and_an_unchanged_one_is_reused(tmp_path, capsys):
    board = _board(tmp_path)
    _add(board, "episode_000000.json", {"provenance": PROV, "fields": {"goal_alignment": ALIGN}})
    board_build.build(board)
    capsys.readouterr()
    board_build.build(board)
    assert "2 of 2 episodes unchanged" in capsys.readouterr().err
    _add(board, "episode_000000.json", {"provenance": PROV, "fields": {"goal_alignment": {**ALIGN, "note": "new"}}})
    board_build.build(board)
    assert "1 of 2 episodes unchanged" in capsys.readouterr().err
    assert _qa(board, "episode_000000.json")["goal_alignment"]["note"] == "new"


def test_an_addition_that_does_not_read_or_names_no_episode_is_reported(tmp_path):
    board = _board(tmp_path)
    _add(board, "episode_000000.json", "{not json")
    _add(board, "episode_000001.json", {"fields": {"goal_alignment": ALIGN}})
    _add(board, "episode_999999.json", {"provenance": PROV, "fields": {"goal_alignment": ALIGN}})
    built = board_build.build(board)
    assert _qa(board, "episode_000000.json")["goal_alignment"] is None
    assert "no provenance" in _qa(board, "episode_000001.json")["addition_conflicts"][0]["why"]
    assert {c["file"] for c in built["additions"]["conflicts"]} == {"episode_000000.json", "episode_000001.json"}
    assert built["additions"]["unused"] == ["episode_999999.json"]


def test_a_board_without_additions_is_built_as_before(tmp_path):
    board = _board(tmp_path)
    built = board_build.build(board)
    assert "additions" not in built and "_additions" not in _qa(board, "episode_000000.json")


# ---------------------------------------------------------------- the parser holds the instruction mismatch rule

def _parse(**labels) -> dict:
    labels, ok = harness.parse_response(json.dumps({"task_summary": "x", **labels}))
    assert ok
    return labels


def test_a_different_or_unrelated_goal_alignment_without_a_mismatch_issue_gets_one_derived():
    for alignment in (ALIGN, {"matches_given": False, "relation": "unrelated", "note": ""},
                      {"matches_given": False, "relation": "narrower", "note": "no bag is seen"}):
        labels = _parse(goal_alignment=alignment, data_issues=[])
        (row,) = labels["data_issues"]
        assert row["category"] == "instruction_mismatch" and row["derived_from"] == "goal_alignment"
        assert row["issue"] == (alignment["note"] or row["issue"]) and row["issue"]
        # parsing the parsed labels again (annotation_complete, reparse) adds nothing
        assert harness.parse_response(json.dumps(labels))[0]["data_issues"] == labels["data_issues"]


def test_a_reply_that_reports_its_mismatch_or_matches_the_instruction_is_left_as_it_is():
    assert _parse(goal_alignment=ALIGN, data_issues=[ISSUE])["data_issues"] == [ISSUE]
    assert _parse(goal_alignment={"matches_given": True, "relation": "aligned", "note": ""}).get("data_issues") is None
    assert _parse(goal_alignment={"matches_given": True, "relation": "broader", "note": "half"},
                  data_issues=[])["data_issues"] == []
    assert "data_issues" not in _parse()
