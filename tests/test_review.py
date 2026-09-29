"""Reviewing your own data (python -m review) and the pieces of it Data Review shares: the neighbour lag measured
inside a folder, the manifest entry for your own data, what a stitched recording carries onto the board, and the
outcome and severity values the board knows."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from board import build as board_build
from board import rules
from checks import timebase
from test_formats import recorder_folder

REPO = Path(__file__).resolve().parent.parent


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


def _episode(root: Path, idx: int, lag: int, task: str = "stack") -> None:
    d = root / f"episode_{idx:06d}"
    d.mkdir(parents=True)
    rng = np.random.default_rng(idx)
    a = np.cumsum(rng.normal(0, 0.01, (400, 14)), axis=0)
    s = np.roll(a, lag, axis=0)                      # the follower trails the leader by `lag` frames
    np.savez(d / "state.npz", state=s, action=a)
    (d / "context.json").write_text(json.dumps({"profile": "teleop_arms", "state_kind": "joints", "episode_index": idx,
                                                "task_label": [task]}))


def test_the_neighbour_lag_is_measured_inside_the_folder(tmp_path):
    for i in range(5):
        _episode(tmp_path, i, lag=3)
    _episode(tmp_path, 40, lag=3)                    # alone: measured, but no neighbour lag
    assert timebase.measure_folder(tmp_path) == 5
    ctx = lambda i: json.loads((tmp_path / f"episode_{i:06d}" / "context.json").read_text())
    assert abs(ctx(2)["timebase_neighbour_lag_frames"] - 3.0) < 0.2 and ctx(2)["timebase_neighbours_in_upload"] == 5
    assert "timebase_neighbour_lag_frames" not in ctx(40) and ctx(40)["timebase_neighbours_in_upload"] == 1


def test_review_runs_every_stage_on_a_recorders_folder_in_free_mode():
    """python -m review on a capture-stack folder reads it with its arm state, runs the checks and the clips, and
    builds every request without calling the model."""
    with tempfile.TemporaryDirectory() as t:
        root = Path(t) / "upload"
        recorder_folder(root, n=90)
        job = Path(t) / "job"
        p = subprocess.run([sys.executable, "-m", "review", "--data", str(root), "--rig", "teleop_arms", "--out",
                            str(job), "--dataset", "mine", "--free"], cwd=REPO, capture_output=True, text=True)
        assert p.returncode == 0, p.stdout + p.stderr
        rep = json.loads((job / "report.json").read_text())
        assert len(rep["episodes"]) == 1 and rep["episodes"][0]["state_kind"] == "joints"
        dry = list((job / "dry").glob("episode_*.json"))
        assert len(dry) == 1 and json.loads(dry[0].read_text())["dry_run"] is True
        assert list((job / "clips").rglob("*.mp4"))
