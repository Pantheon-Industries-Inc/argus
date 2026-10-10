"""The model comparison: which episodes a selection names, the comparison folder, the manifest entries of the runs
`python -m compare label` starts, the metrics measured from run folders, and the board `python -m compare board`
makes of them. No network, no model call: the prepared episodes are empty folders with a context.json, and the
label subprocess is a fake that makes run folders.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import compare.__main__ as cm
from compare import metrics

SELECTIONS = cm.REPO / "configs" / "compare"
MODELS = json.loads((cm.REPO / "configs" / "models.json").read_text())


def _prepare_all(episodes: Path) -> None:
    """Every episode of main.json as `prepare` would leave it: EPISODES/<dataset>/compare_main/<episode>."""
    for dss in json.loads((SELECTIONS / "main.json").read_text())["rigs"].values():
        for ds, v in dss.items():
            for e in v["episodes"]:
                d = cm.prepared_dir(episodes, ds, "main") / e["episode"]
                d.mkdir(parents=True)
                (d / "context.json").write_text("{}")


# ---------------------------------------------------------------- selections

def test_main_selection_resolves_to_one_prepared_folder_per_episode(tmp_path):
    main = json.loads((SELECTIONS / "main.json").read_text())
    pairs = cm.selected(SELECTIONS / "main.json", tmp_path)
    n = sum(len(v["episodes"]) for dss in main["rigs"].values() for v in dss.values())
    assert len(pairs) == n == 193
    names = [name for name, _ in pairs]
    assert len(set(names)) == len(names)                                   # one name per episode in the folder
    assert all(src.parent.name == "compare_main" and src.parent.parent.parent == tmp_path for _, src in pairs)
    habit = [(name, src) for name, src in pairs if src.parent.parent.name == "habit"]
    assert habit and all(name == "episode_habit_" + src.name[len("episode_"):] for name, src in habit)
    assert all(name == src.name for name, src in pairs if src.parent.parent.name != "habit")


def test_every_third_name_resolves_to_a_main_episode(tmp_path):
    third = json.loads((SELECTIONS / "third.json").read_text())
    assert third["subset_of"] == "main"
    main = dict(cm.selected(SELECTIONS / "main.json", tmp_path))
    pairs = cm.selected(SELECTIONS / "third.json", tmp_path)
    assert len(pairs) == sum(len(v) for v in third["episodes"].values()) == 65
    for name, src in pairs:
        assert main[name] == src
    for ds, eps in third["episodes"].items():                             # each name is in its own dataset
        assert all(main[e].parent.parent.name == ds for e in eps)


def test_link_slice_links_every_selected_episode(tmp_path):
    episodes = tmp_path / "episodes"
    with pytest.raises(SystemExit, match="not prepared"):
        cm.link_slice(SELECTIONS / "third.json", episodes)
    _prepare_all(episodes)
    dest = cm.link_slice(SELECTIONS / "third.json", episodes)
    assert dest == episodes / "compare" / "third"
    links = sorted(dest.iterdir())
    assert len(links) == 65 and all(p.is_symlink() and (p / "context.json").exists() for p in links)
    habit = dest / "episode_habit_005709"
    assert habit.resolve() == (episodes / "habit" / "compare_main" / "episode_005709").resolve()
    assert cm.link_slice(SELECTIONS / "third.json", episodes) == dest          # linking again changes nothing
    habit.unlink()
    habit.symlink_to(episodes / "molmo" / "compare_main" / "episode_000209")
    with pytest.raises(SystemExit, match="points elsewhere"):
        cm.link_slice(SELECTIONS / "third.json", episodes)


def test_prepare_writes_each_datasets_list_and_runs_prepare_then_checks(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, cwd):
        calls.append(cmd)
        return cm.subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cm.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["python -m compare", "prepare", "--selection", str(SELECTIONS / "main.json"),
                                      "--episodes", str(tmp_path)])
    assert cm.main() == 0
    lst = (tmp_path / "molmo" / "compare_main.txt").read_text().splitlines()
    assert lst[0].startswith("#") and lst[1:] == [e["line"] for e in
                                                  json.loads((SELECTIONS / "main.json").read_text())["rigs"]
                                                  ["teleop"]["molmo"]["episodes"]]
    assert calls[0][1:] == ["-m", "prepare", "molmo", "prepare", "--episodes", str(tmp_path / "molmo" /
                            "compare_main.txt"), "--out", str(tmp_path / "molmo" / "compare_main")]
    assert calls[1][1:] == ["-m", "checks", str(tmp_path / "molmo" / "compare_main")]
    assert len(calls) == 2 * 9
    monkeypatch.setattr(sys, "argv", ["python -m compare", "prepare", "--selection", str(SELECTIONS / "third.json")])
    with pytest.raises(SystemExit, match="subset of main"):
        cm.main()


# ---------------------------------------------------------------- label: the manifest entries

class FakeLabel:
    """compare's label subprocess: makes the run folder label/run.py would, RUNS/compare/<run_id>."""
    started: list = []
    also: Path | None = None

    def __init__(self, cmd, cwd):
        def arg(flag):
            return cmd[cmd.index(flag) + 1]
        self.cmd = cmd
        FakeLabel.started.append(cmd)
        run = Path(arg("--runs")) / arg("--dataset") / f"20260927-1200_{arg('--kind')}_abc1234_{arg('--label')}"
        run.mkdir(parents=True)
        (run / "run.json").write_text(json.dumps({"run_id": run.name, "command": cmd,
                                                  "slice": str(Path(arg("--episodes")).resolve())}))
        if FakeLabel.also:
            # the same model's run of another compare label started at the same moment, over another slice
            other = Path(arg("--runs")) / arg("--dataset") / f"20260927-1201_{arg('--kind')}_abc1234_{arg('--label')}"
            other.mkdir(parents=True)
            (other / "run.json").write_text(json.dumps({"run_id": other.name, "slice": str(FakeLabel.also)}))

    def wait(self):
        return 0


def _label(tmp_path, monkeypatch, *extra):
    episodes, runs = tmp_path / "episodes", tmp_path / "runs"
    _prepare_all(episodes)
    FakeLabel.started, FakeLabel.also = [], None
    monkeypatch.setattr(cm.subprocess, "Popen", FakeLabel)
    monkeypatch.setattr(sys, "argv", ["python -m compare", "label", "--episodes", str(episodes), "--runs", str(runs),
                                      *extra])
    return episodes, runs


def test_label_writes_one_manifest_entry_per_model(tmp_path, monkeypatch):
    episodes, runs = _label(tmp_path, monkeypatch, "--selection", str(SELECTIONS / "main.json"), "--cap", "25")
    assert cm.main() == 0
    entries = json.loads((runs / "compare" / "main.json").read_text())
    assert [e["key"] for e in entries] == list(MODELS["models"])
    for e in entries:
        m = MODELS["models"][e["key"]]
        assert e["name"] == m["name"] and Path(e["run"]).name.endswith("_" + e["key"])
        assert e["episodes"] == str((episodes / "compare" / "main").resolve())
        assert e.get("reference", False) == (e["key"] == MODELS["reference"]) and "example" not in e
    for cmd in FakeLabel.started:
        key = cmd[cmd.index("--label") + 1]
        assert cmd[cmd.index("--model") + 1] == MODELS["models"][key]["model"]
        assert cmd[cmd.index("--reasoning") + 1] == MODELS["models"][key].get("reasoning", MODELS["reasoning"])
        assert cmd[cmd.index("--max-tokens") + 1] == str(MODELS["max_tokens"]) and "--example-dir" not in cmd
        assert cmd[cmd.index("--cap") + 1] == "25.0" and cmd[cmd.index("--kind") + 1] == "full"


# The command each model that has published results ran with (compare/__main__.py at e076229, when every model took
# the one top-level reasoning effort): a model's own "reasoning" in configs/models.json must never change them.
PUBLISHED = {"astra": "openai/gpt-6-astra", "opus55": "anthropic/claude-opus-5.5", "sol6": "openai/gpt-6-sol",
             "dsv41f": "deepseek/deepseek-v4.1-flash"}


def _published_command(key, model, slice_dir, runs, example):
    note = f"model comparison on {slice_dir.name}: {model}" + (", in-context learning with a reference trace"
                                                                if example else "")
    return ([sys.executable, "-m", "label", "--dataset", "compare", "--episodes", str(slice_dir), "--kind", "full",
             "--cap", "25.0", "--runs", str(runs), "--concurrency", "8", "--label", key + ("_ex" if example else ""),
             "--note", note, "--", "--model", model, "--reasoning", "medium", "--max-tokens", "64000"]
            + (["--example-dir", str(cm.REPO / "configs" / "examples")] if example else []))


@pytest.mark.parametrize("selection,example", [("main", False), ("third", True)])
def test_every_published_models_command_is_unchanged(tmp_path, monkeypatch, selection, example):
    episodes, runs = _label(tmp_path, monkeypatch, "--selection", str(SELECTIONS / f"{selection}.json"),
                            "--cap", "25", *(["--with-example"] if example else []))
    assert cm.main() == 0
    started = {cmd[cmd.index("--label") + 1].removesuffix("_ex"): cmd for cmd in FakeLabel.started}
    want = [k for k in PUBLISHED if k in (MODELS["with_example"] if example else MODELS["models"])]
    assert want == (["opus55", "sol6", "dsv41f"] if example else list(PUBLISHED))
    for key in want:
        assert started[key] == _published_command(key, PUBLISHED[key], episodes / "compare" / selection, runs, example)


def test_a_models_own_reasoning_effort_reaches_its_run_only(tmp_path, monkeypatch):
    _label(tmp_path, monkeypatch, "--selection", str(SELECTIONS / "main.json"), "--models", "sol6,sol61_high",
           "--cap", "150")
    assert cm.main() == 0
    effort = {cmd[cmd.index("--label") + 1]: cmd[cmd.index("--reasoning") + 1] for cmd in FakeLabel.started}
    assert effort == {"sol6": "medium", "sol61_high": "high"}
    assert MODELS["models"]["sol61_high"]["reasoning"] == "high" and "reasoning" not in MODELS["models"]["sol6"]


def test_label_with_example_marks_example_and_base_and_no_reference(tmp_path, monkeypatch):
    _, runs = _label(tmp_path, monkeypatch, "--selection", str(SELECTIONS / "third.json"), "--with-example",
                     "--kind", "smoke", "--cap", "5")
    assert cm.main() == 0
    entries = json.loads((runs / "compare" / "third_ex.json").read_text())
    assert [e["key"] for e in entries] == [k + "_ex" for k in MODELS["with_example"]]
    for e in entries:
        base = e["key"].removesuffix("_ex")
        assert e["example"] is True and e["base"] == base and "reference" not in e
        assert e["name"] == MODELS["models"][base]["name"] + ", in-context learning with an Astra trace"
    assert all(cmd[cmd.index("--example-dir") + 1] == str(cm.REPO / "configs" / "examples")
               for cmd in FakeLabel.started)


def test_label_refuses_unknown_models_and_a_paid_run_without_a_cap(tmp_path, monkeypatch):
    _label(tmp_path, monkeypatch, "--selection", str(SELECTIONS / "main.json"), "--models", "astra,nope", "--cap", "1")
    with pytest.raises(SystemExit, match="unknown models"):
        cm.main()
    _label(tmp_path / "b", monkeypatch, "--selection", str(SELECTIONS / "main.json"))
    with pytest.raises(SystemExit, match="needs --cap"):
        cm.main()
    assert FakeLabel.started == []


# ---------------------------------------------------------------- metrics on synthetic run folders

def _labels(outcome, segments, key_events=(), issues=(), arm="left"):
    """A response that follows the schema every model is given."""
    return {"scene": {}, "timeline": [{"start_s": float(i), "end_s": i + 1.0, "arm": arm, "contribution": "advancing",
                                       "progress": 0.5} for i in range(segments)],
            "task_summary": "", "key_events": list(key_events), "state_changes": [], "scene_graph": [],
            "recovery": [], "instruction_variants": [], "performance_review": "", "data_issues": list(issues),
            "completion": {"task_completed": outcome}}


CONFIG = {"views": ["exo", "left", "right"], "cam_labels": ["top", "left", "right"]}
SWAP_HIGH = {"issue": "left and right files swapped", "category": "camera_swap", "severity": "high"}
SWAP_MEDIUM = {**SWAP_HIGH, "severity": "medium"}
ODD_LOW = {"issue": "something small", "category": "odd_thing", "severity": "low"}
SUBGOAL = {"t_s": 1.0, "outcome": "success", "kind": "subgoal_complete"}
OTHER_EVENT = {"t_s": 2.0, "outcome": "success", "kind": "grasp"}


def _run(root: Path, key: str, sl: Path, outputs: dict, cut_off=(), example=False, status="done",
         model=None, route_cost=None) -> Path:
    """A run folder: outputs maps an episode to (labels or None for an unparsed reply, cost, latency); route_cost
    maps an episode to what its routing call cost."""
    run = root / key
    (run / "out").mkdir(parents=True)
    model = model or f"vendor/{key}"
    cmd = ["python", "-m", "label.harness", "--model", model] + (["--example-dir", "ex"] if example else [])
    (run / "run.json").write_text(json.dumps({"run_id": key, "code": "abc1234", "status": status, "slice": str(sl),
                                              "command": cmd}))
    for name, (labels, cost, latency) in outputs.items():
        config = dict(CONFIG, **({"resolution_route": {"cost_usd": route_cost[name]}}
                                 if name in (route_cost or {}) else {}))
        r = {"parse_ok": labels is not None, "config": config, "model": model,
             "labels": labels if labels is not None else {"_raw": "{", "_parse_error": "JSONDecodeError"},
             "usage": {"est_cost_usd": cost, "latency_s": latency, "completion_tokens": 100}}
        (run / "out" / f"{name}.json").write_text(json.dumps(r))
    for name, cost in cut_off:
        (run / "out" / f"failed_{name}.json").write_text(json.dumps(
            {"finish_reason": "length", "usage": {"cost": cost, "completion_tokens": 64000}, "content_tail": "{"}))
    return run


def _comparison(tmp_path) -> dict:
    """A board manifest over three teleop episodes of 20, 30 and 10 s: the board's own labels (by Astra), a second
    model, and that model with in-context learning.

      board   a parsed (success, 4 segments, 2 key events, 1 subgoal, a high camera swap)  b parsed
              c parsed ($0.30 and a $0.03 routing call)
      m2      a parsed (success, 2 segments, 1 key event, a medium camera swap and a low odd thing, arm "middle")
              b unparsed ($0.07, 7 s)   c cut off ($0.09)
      m2_ex   a parsed (failure, 3 segments)   b parsed   c none (the run finished)
    """
    sl = tmp_path / "slice"
    for name, frames in (("episode_a", 600), ("episode_b", 900), ("episode_c", 300)):
        (sl / name).mkdir(parents=True)
        (sl / name / "context.json").write_text(json.dumps({"dataset": "x", "profile": "teleop_arms", "fps": 30,
                                                            "n_state_frames": frames}))
    runs = tmp_path / "runs"
    _run(runs, "ref", sl, {"episode_a": (_labels("success", 4, [SUBGOAL, OTHER_EVENT], [SWAP_HIGH]), 0.10, 10),
                           "episode_b": (_labels("failure", 6, [OTHER_EVENT]), 0.20, 20),
                           "episode_c": (_labels("success", 2), 0.30, 30)},
         model="openai/gpt-6-astra", route_cost={"episode_c": 0.03})
    _run(runs, "m2", sl, {"episode_a": (_labels("success", 2, [OTHER_EVENT], [SWAP_MEDIUM, ODD_LOW], arm="middle"),
                                        0.05, 5),
                          "episode_b": (None, 0.07, 7)}, cut_off=[("episode_c", 0.09)])
    _run(runs, "m2_ex", sl, {"episode_a": (_labels("failure", 3), 0.04, 4),
                             "episode_b": (_labels("success", 5), 0.06, 6)}, example=True)
    return {"datasets": [{"dataset": "trial", "run": "runs/ref", "episodes": "slice"}],
            "comparisons": [{"key": "m2", "name": "Model two", "run": "runs/m2"},
                            {"key": "m2_ex", "name": "Model two, in-context learning with an Astra trace", "run": "runs/m2_ex"}]}


def test_load_models_reads_run_json_and_defaults(tmp_path):
    ms = {m["key"]: m for m in metrics.load_models(_comparison(tmp_path), tmp_path)}
    assert set(ms) == {"m2", "m2_ex"} and not ms["m2"]["reference"] and ms["m2"]["model"] == "vendor/m2"
    assert ms["m2_ex"]["example"] and ms["m2_ex"]["base"] == "m2"               # from --example-dir and the key
    assert not ms["m2"]["example"] and ms["m2"]["base"] is None
    assert ms["m2"]["episodes"] == tmp_path / "slice"                           # the slice run.json records
    assert ms["m2"]["episode_name"] == "Model two"
    assert ms["m2"]["reasoning"] is None                  # the command leaves the effort at the harness default
    manifest = _comparison(tmp_path / "b")
    run = tmp_path / "b" / "runs" / "m2" / "run.json"
    info = json.loads(run.read_text())
    run.write_text(json.dumps({**info, "command": info["command"] + ["--reasoning", "high"]}))
    assert {m["key"]: m["reasoning"] for m in metrics.load_models(manifest, tmp_path / "b")} \
        == {"m2": "high", "m2_ex": None}
    for key in ("board", "lists"):          # the reference's key, and the static build's folder of rail records
        bad = _comparison(tmp_path / key)
        bad["comparisons"][0]["key"] = key
        with pytest.raises(ValueError, match="no comparison may have the key"):
            metrics.load_models(bad, tmp_path / key)


def test_response_status_of_each_kind_of_output(tmp_path):
    ms = {m["key"]: m for m in metrics.load_models(_comparison(tmp_path), tmp_path)}
    parsed = metrics.response(ms["m2"], "episode_a")
    assert parsed["status"] == "parsed" and parsed["model"] == "vendor/m2" and parsed["cost"] == 0.05
    unparsed = metrics.response(ms["m2"], "episode_b")
    assert unparsed["status"] == "unparsed" and unparsed["cost"] == 0.07 and unparsed["error"] == "JSONDecodeError"
    cut = metrics.response(ms["m2"], "episode_c")
    assert cut["status"] == "cut_off" and cut["cost"] == 0.09 and cut["latency"] is None
    # the board's own label of a cut-off episode is its failed_ file, which reads as cut off
    assert metrics.read_output(cut["path"])["status"] == "cut_off"
    assert metrics.response(ms["m2_ex"], "episode_c")["status"] == "no_response"   # the run finished without it
    assert metrics.response({**ms["m2_ex"], "status": "running"}, "episode_c")["status"] == "pending"


def test_metrics_by_hand(tmp_path):
    """The reference is the board's own label of each episode, measured like every other model's response."""
    res = metrics.compute(_comparison(tmp_path), tmp_path)
    assert res["main"] == ["board", "m2"] and [m["key"] for m in res["models"]] == ["board", "m2", "m2_ex"]
    ref_model = res["models"][0]
    assert ref_model["reference"] and ref_model["name"] == "Astra" and ref_model["model"] == "openai/gpt-6-astra"
    s = res["summary"]["all"]
    assert res["summary"]["teleop"] == s and set(res["summary"]) == {"all", "teleop"}
    assert (s["episodes"], s["minutes"], s["common"], s["common_minutes"]) == (3, 1.0, 1, 0.33)
    ref, m2, ex = s["responses"]["board"], s["responses"]["m2"], s["responses"]["m2_ex"]
    assert (ref["asked"], ref["parsed"], ref["parse_share"], ref["violations"]) == (3, 3, 1.0, 0.0)
    assert (ref["cost"], ref["latency"]) == (0.21, 20.0)      # (0.10 + 0.20 + 0.30 + the 0.03 routing call) / 3
    assert (m2["parsed"], m2["unparsed"], m2["cut_off"], m2["parse_share"]) == (1, 1, 1, 0.3333)
    assert (m2["violations"], m2["violations_any"], m2["violations_n"]) == (2.0, 1.0, 1)    # 2 segments, arm "middle"
    assert (m2["cost"], m2["cost_n"], m2["latency"], m2["latency_n"]) == (0.07, 3, 6.0, 2)  # cut off: no latency
    assert (ex["parsed"], ex["no_response"], ex["parse_share"]) == (2, 1, 1.0)
    # per-episode numbers on the one episode both main models parsed (episode_a, 20 s)
    mr, mm = s["metrics"]["board"], s["metrics"]["m2"]
    assert (mr["n"], mr["minutes"], mr["events_per_min"], mr["key_events"], mr["subgoals"]) == (1, 0.33, 12.0, 2, 1)
    assert (mm["events_per_min"], mm["key_events"], mm["subgoals"]) == (6.0, 1, 0)
    assert (mr["data_issues"], mr["data_issues_minor"], mm["data_issues"], mm["data_issues_minor"]) == (1, 0, 1, 1)
    ag = s["agreement"]["board"]["m2"]
    assert (ag["outcome"], ag["outcome_n"]) == (1.0, 1)
    assert (ag["issues"], ag["issues_shared"], ag["issues_union"]) == (0.5, 1, 2)     # the camera swap is shared
    assert "m2_ex" in s["agreement"]["board"] and s["agreement"]["m2"]["board"] == ag
    # with an example: m2 and m2_ex were both asked a, b and c; both parsed only a, which the board has a label of
    (pair,) = res["paired"]["all"]
    assert (pair["base"], pair["with"], pair["asked"], pair["n"], pair["minutes"]) == ("m2", "m2_ex", 3, 1, 0.33)
    w, wo = pair["with_values"], pair["without_values"]
    assert (wo["parse_share"], wo["cost"], wo["events_per_min"], wo["agree_outcome"]) == (0.3333, 0.07, 6.0, 1.0)
    assert (w["parse_share"], w["cost"], w["events_per_min"], w["agree_outcome"]) == (1.0, 0.05, 9.0, 0.0)
    assert pair["reference"]["events_per_min"] == 12.0 and pair["reference_n"] == 1       # the board's own values
    by = {e["episode_id"]: e["by"] for e in res["episodes"]}
    assert by["episode_c"] == {"board": {"status": "parsed", "outcome": "success", "segments": 2, "cost": 0.33},
                               "m2": {"status": "cut_off", "cost": 0.09}, "m2_ex": {"status": "no_response"}}
    assert "_rows" not in metrics.public(res) and "_rows" in res


def test_the_reference_follows_the_boards_labels(tmp_path):
    """A board label rebuilt from another run changes the reference with it: the comparison never keeps a label of
    its own for the board's model."""
    comp = _comparison(tmp_path)
    _run(tmp_path / "runs", "ref2", tmp_path / "slice", {"episode_a": (_labels("failure", 1), 0.5, 1)},
         model="anthropic/claude-opus-5.5")
    comp["datasets"][0]["run"] = "runs/ref2"
    res = metrics.compute(comp, tmp_path)
    assert res["models"][0]["name"] == "Claude Opus 5.5"                  # the display name from configs/models.json
    by = {e["episode_id"]: e["by"] for e in res["episodes"]}
    assert by["episode_a"]["board"]["outcome"] == "failure" and list(by) == ["episode_a"]   # b and c: no board label
    assert res["summary"]["all"]["responses"]["board"]["asked"] == 1


def test_metrics_command_line(tmp_path, monkeypatch, capsys):
    (tmp_path / "manifest.json").write_text(json.dumps(_comparison(tmp_path)))
    monkeypatch.setattr(sys, "argv", ["python -m compare.metrics", str(tmp_path), "--json", str(tmp_path / "m.json")])
    assert metrics.main() == 0
    out = capsys.readouterr().out
    assert "== All footage: 3 episodes, 1 min; every model parsed 1" in out and " 33.3%" in out and "Astra" in out
    assert json.loads((tmp_path / "m.json").read_text())["summary"]["all"]["common"] == 1
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(SystemExit, match="names no comparisons"):
        metrics.main()


def test_board_manifest_builds_the_comparison_board(tmp_path, monkeypatch):
    """`compare board`: the reference run's labels are the board's, the other runs its comparisons."""
    from board import build as board_build
    comp = _comparison(tmp_path)
    sl = str((tmp_path / "slice").resolve())
    ref = {"key": "astra", "name": "Astra", "run": str((tmp_path / "runs" / "ref").resolve()), "episodes": sl,
           "reference": True}
    others = [dict(e, run=str((tmp_path / e["run"]).resolve()), episodes=sl) for e in comp["comparisons"]]
    for e in [ref] + others:
        rj = Path(e["run"]) / "run.json"
        rj.write_text(json.dumps({**json.loads(rj.read_text()), "kind": "full"}))
    (tmp_path / "main.json").write_text(json.dumps([ref, others[0]]))
    (tmp_path / "third_ex.json").write_text(json.dumps(others[1:]))
    board = tmp_path / "boards" / "compare"
    monkeypatch.setattr(sys, "argv", ["python -m compare", "board", "--entries", str(tmp_path / "main.json"),
                                      str(tmp_path / "third_ex.json"), "--out", str(board)])
    assert cm.main() == 0
    manifest = json.loads((board / "manifest.json").read_text())
    (ds,) = manifest["datasets"]
    assert ds["dataset"] == "compare" and ds["run"] == ref["run"]
    assert [e["key"] for e in manifest["comparisons"]] == ["m2", "m2_ex"]
    built = board_build.build(board)
    assert built["counts"]["compare"]["episodes"] == 3
    assert sorted(p.name for p in (board / "qa").glob("*.json")) == ["episode_a.json", "episode_b.json",
                                                                      "episode_c.json"]
    assert sorted(p.name for p in (board / "compare").iterdir() if p.is_dir()) == ["m2", "m2_ex"]
    index = json.loads((board / "compare" / "index.json").read_text())
    assert index["reference"]["name"] == "Astra" and [m["key"] for m in index["models"]] == ["m2", "m2_ex"]
    m = json.loads((board / "compare" / "metrics.json").read_text())
    assert m["main"] == ["board", "m2"] and m["summary"]["all"]["responses"]["board"]["parsed"] == 3
    # a set of entries without exactly one reference run is refused
    (tmp_path / "none.json").write_text(json.dumps(others))
    monkeypatch.setattr(sys, "argv", ["python -m compare", "board", "--entries", str(tmp_path / "none.json"),
                                      "--out", str(tmp_path / "b2")])
    with pytest.raises(SystemExit, match="0 reference runs"):
        cm.main()


def test_episodes_the_board_does_not_hold_are_left_out(tmp_path):
    """A comparison run over a slice that also holds an episode the board does not: that episode has no board label
    to set the model against, so no number includes it."""
    comp = _comparison(tmp_path)
    cmp_slice, extra = tmp_path / "cmp", tmp_path / "elsewhere" / "episode_z"
    cmp_slice.mkdir()
    extra.mkdir(parents=True)
    (extra / "context.json").write_text(json.dumps({"dataset": "x", "profile": "teleop_arms", "fps": 30,
                                                    "n_state_frames": 300}))
    (cmp_slice / "episode_a").symlink_to(tmp_path / "slice" / "episode_a")
    (cmp_slice / "episode_z").symlink_to(extra)
    _run(tmp_path / "runs", "m3", cmp_slice, {"episode_a": (_labels("success", 2), 0.01, 1),
                                              "episode_z": (_labels("failure", 2), 0.01, 1)})
    comp["comparisons"] = [{"key": "m3", "name": "Model three", "run": "runs/m3"}]
    res = metrics.compute(comp, tmp_path)
    assert [e["episode_id"] for e in res["episodes"]] == ["episode_a"]
    assert res["summary"]["all"]["episodes"] == 1 and res["summary"]["all"]["responses"]["m3"]["asked"] == 1


# ---------------------------------------------------------------- label: the reference run's routing answers

def _routed_slice(root: Path, instructions: dict) -> Path:
    """A slice of teleop episodes (a routed rig) and one head-camera episode, each with the files label.episode
    loads."""
    import numpy as np
    sl = root / "slice"
    for name, instr in {**instructions, "episode_ego": None}.items():
        d = sl / name
        d.mkdir(parents=True)
        (d / "sources.json").write_text(json.dumps({"exo": {"n_frames": 10}}))
        ctx = {"dataset": "some/dataset", "fps": 30, "n_state_frames": 10}
        if instr is None:
            ctx.update(profile="ego_head", state_kind="none")
        else:
            ctx.update(profile="teleop_arms", state_kind="joints", instruction=instr)
            np.savez(d / "state.npz", state=np.zeros((10, 14)))
        (d / "context.json").write_text(json.dumps(ctx))
    return sl


def test_unrouted_names_the_teleop_episodes_the_answers_do_not_cover(tmp_path):
    from label import episode as me
    from label import route
    sl = _routed_slice(tmp_path, {"episode_a": "Stack the cups.", "episode_b": "Read the label."})
    answers = {route.route_text(me.load(sl / "episode_a")): {"fine_detail": False, "why": "whole objects"}}
    assert cm.unrouted(sl, answers) == ["episode_b"]          # a head camera is never routed, so never missing


def test_label_with_routes_seeds_every_run_and_refuses_what_it_cannot_seed(tmp_path, monkeypatch):
    routes = tmp_path / "routes.json"
    routes.write_text("{}")
    _label(tmp_path, monkeypatch, "--selection", str(SELECTIONS / "main.json"), "--models", "sol61_high",
           "--cap", "150", "--routes", str(routes))
    monkeypatch.setattr(cm, "unrouted", lambda sl, answers: [])
    assert cm.main() == 0
    (cmd,) = FakeLabel.started
    assert cmd[cmd.index("--route-seeds") + 1] == str(routes.resolve())
    _label(tmp_path / "b", monkeypatch, "--selection", str(SELECTIONS / "main.json"), "--models", "sol61_high",
           "--cap", "150", "--routes", str(routes))
    monkeypatch.setattr(cm, "unrouted", lambda sl, answers: ["episode_x"])
    with pytest.raises(SystemExit, match="no routing answer for 1 routed episodes"):
        cm.main()
    assert FakeLabel.started == []


def test_published_routes_answer_every_task_text_true_or_false():
    from label import route
    answers = json.loads((cm.REPO / "configs" / "compare" / "routes_main.json").read_text())
    assert len(answers) == 39 and all(isinstance(a["fine_detail"], bool) for a in answers.values())
    assert all(t.startswith("Dataset: ") for t in answers)
    monkeypatch_cache = dict(route._CACHE)
    try:
        assert route.seed(answers, "routes_main.json") == 39
    finally:
        route._CACHE.clear()
        route._CACHE.update(monkeypatch_cache)


def test_label_names_its_own_run_when_another_starts_the_same_model(tmp_path, monkeypatch):
    """Two compare label invocations of one model at once (a tranche while the 193 still run): each entry names the
    run over its own slice, never the other's newer folder of the same key."""
    episodes, runs = _label(tmp_path, monkeypatch, "--selection", str(SELECTIONS / "main.json"), "--models", "sol6",
                            "--cap", "5")
    FakeLabel.also = tmp_path / "episodes" / "compare" / "tranche_10h"
    assert cm.main() == 0
    (entry,) = json.loads((runs / "compare" / "main.json").read_text())
    assert json.loads((Path(entry["run"]) / "run.json").read_text())["slice"] == entry["episodes"]
