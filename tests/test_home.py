"""The home view's numbers (board/home.py) and labels followed into a board while a run writes them (board/follow.py)."""
import json
import os
import time
from pathlib import Path

import pytest

from board import follow, home


def test_kinds_come_from_the_labels_own_words():
    assert [home.object_kind(n) for n in ["pebble container", "clear test tubes", "can of tuna", "pair of scissors",
                                          "10 of diamonds", "cloth and blue plastic piece", "black cherries"]] == \
        ["container", "tube", "can", "scissors", "card", "cloth", "cherry"]
    names = ["clear test tubes", "handled comb"]
    assert home.task_kind("Lift clear tubes from the rack and reseat them.", names) == "lift tube"
    assert home.task_kind("Try to stand the handled comb upright.", names) == "stand comb"
    assert home.task_kind("Gently lift the clear test tubes.", names) == "lift tube"


def _episode(qa: Path, name: str, **d):
    base = {"episode_prompt": "", "dataset": "ds", "duration_s": 60.0, "objects": [], "event_labels": [],
            "data_issues": [], "operator_mistakes": [], "_usage": {"est_cost_usd": 0.5}}
    (qa / f"{name}.json").write_text(json.dumps({**base, **d}))


def _counts_all(_key, _issue):
    return True


def test_objects_and_mistakes_count_per_subtask(tmp_path):
    """A subtask counts the objects it handles (its task's objects and its events' objects), never the episode's list
    of everything on the table; a timed mistake goes to the subtask its time falls in; data issues stay per episode."""
    qa = tmp_path / "qa"
    qa.mkdir()
    # a scripted episode: one task, nine objects on the table, one of them handled
    _episode(qa, "scripted", dataset="scripted", episode_prompt="Place the pebble container on the tray.",
             completion={"task_completed": "success"},
             objects=[{"name": n} for n in ["pebble container", "wood serving tray", "phone", "red cup", "green block",
                                             "sponge", "marker", "tape", "bowl"]],
             event_labels=[{"t_s": 3.0, "verb_class": "grasp", "object": "pebble container"},
                           {"t_s": 5.0, "verb_class": "lift and carry", "object": "pebble container"}],
             data_issues=[{"issue": "names the wrong object", "category": "instruction_mismatch", "severity": "high"}])
    # a session of two tasks, a mistake timed inside the second
    _episode(qa, "session", dataset="freeform", duration_s=120.0, objects=[{"name": "remote control"}],
             tasks=[{"start_s": 0, "end_s": 50, "task": "Fold the towel.", "objects": ["blue towel"], "outcome": "success"},
                    {"start_s": 50, "end_s": 120, "task": "Stack the cups.", "objects": ["red cups"], "outcome": "failure"}],
             event_labels=[{"t_s": 60.0, "verb_class": "grasp", "object": "green cup"}],
             operator_mistakes=[{"issue": "drops a cup", "category": "dropped_object", "t_s": 80.0, "severity": "medium"}])
    rows = [home.summarize(p, json.loads(p.read_text()), _counts_all) for p in sorted(qa.glob("*.json"))]
    s = home.stats(rows, {}, time.time(), {"towel": True, "cup": False, "container": False})
    by = {d["dataset"]: d for d in s["datasets"]}
    assert by["scripted"]["n_tasks"] == 1 and by["scripted"]["diversity"]["objects"]["distinct"] == 1
    assert by["freeform"]["n_tasks"] == 2 and by["freeform"]["subtasks_with_mistake"] == 1
    assert by["scripted"]["eps_with_issue"] == 1
    # the phone and the remote control sit on the tables and are never handled
    assert [k for k, *_ in s["total"]["top_objects"]] == ["container", "cup", "towel"]
    assert s["total"]["deformable"] == {"rigid": 2, "deformable": 1}
    assert by["freeform"]["outcomes"]["success"] == 1 and by["freeform"]["outcomes"]["failure"] == 1
    # timeline rows and key events are summed per dataset and in all
    assert s["total"]["n_timeline"] == 3 and by["scripted"]["n_timeline"] == 2 and s["total"]["n_key_events"] == 0
    # without the labeler's skill a subtask counts its task's first verb, and no actions
    assert sorted(k for k, *_ in s["total"]["top_skills"]) == ["fold", "place", "stack"]
    assert s["total"]["top_actions"] == []
    # the labeler's skill wins over the sentence's first word; each action counts once per subtask
    d = json.loads((qa / "session.json").read_text())
    d["tasks"][0].update(skill="Fold", actions=["flip", "Flip", "smooth"])
    d["tasks"][1].update(skill="stack", actions=["hand over"])
    named = home.summarize(qa / "session.json", d, _counts_all)
    assert named["skills"] == {"fold": 1, "stack": 1}
    assert named["actions"] == {"flip": 1, "smooth": 1, "hand over": 1}
    d["tasks"][1]["skill"] = "None"                     # the labeler saying no action happened counts no skill
    assert home.summarize(qa / "session.json", d, _counts_all)["skills"] == {"fold": 1}
    # names for one action count under the one name board/skills.py gave them
    merged = home.stats([named, {**named, "file": "b.json", "skills": {"stack up": 2}, "actions": {"pass": 1}}], {},
                        time.time(), names={"stack up": "stack", "pass": "hand over"})
    assert dict((k, n) for k, n, _ in merged["total"]["top_skills"]) == {"stack": 3, "fold": 1}
    assert dict((k, n) for k, n, _ in merged["total"]["top_actions"]) == {"hand over": 2, "flip": 1, "smooth": 1}


def test_the_numbers_keep_their_version_until_a_label_changes(tmp_path):
    qa = tmp_path / "qa"
    qa.mkdir()
    _episode(qa, "a", episode_prompt="Move the cup.")
    _, _, tag1 = home.home_json(qa, tmp_path / "plan.json", _counts_all)
    _, _, tag2 = home.home_json(qa, tmp_path / "plan.json", _counts_all)
    _episode(qa, "b", episode_prompt="Move the bowl.")
    raw, _, tag3 = home.home_json(qa, tmp_path / "plan.json", _counts_all)
    assert tag1 == tag2 != tag3 and json.loads(raw)["total"]["episodes"] == 2


def test_follow_puts_a_long_recording_on_the_board_once_its_last_part_is_in(tmp_path, monkeypatch):
    """Parts of a long recording never reach the board as episodes; the recording is stitched once every part is in,
    and a restarted follow does not write a label the board already has."""
    run, eps, board = tmp_path / "job" / "run", tmp_path / "job" / "episodes", tmp_path / "board"
    (run / "out").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"run_id": "r", "code": "c", "kind": "review"}))
    (eps / "episode_long").mkdir(parents=True)
    parts = ["episode_long__p01", "episode_long__p02"]
    (eps / "episode_long" / "context.json").write_text(json.dumps({"pieces": {"parts": parts}}))
    for n in parts:
        (tmp_path / "job" / "pieces" / n).mkdir(parents=True)
        (tmp_path / "job" / "pieces" / n / "context.json").write_text("{}")
    board.mkdir()
    (board / "manifest.json").write_text(json.dumps({"datasets": [{"dataset": "ds", "run": str(run), "episodes": str(eps)}]}))
    monkeypatch.setattr("label.pieces.stitch", lambda ep_dir, got: {"stitched_parts": len(got), "parse_ok": True})
    monkeypatch.setattr(follow, "board_label", lambda entry, manifest, fname, name, src, r, info, eps: (r, None))
    part = {"parse_ok": True}
    seen = {}
    (run / "out" / f"{parts[0]}.json").write_text(json.dumps(part))
    (run / "out" / "episode_short.json").write_text(json.dumps({"parse_ok": True, "episode_dir": "episode_short"}))
    assert follow.follow_once(board, seen) == 1                      # the short episode, not the part
    assert sorted(p.name for p in (board / "qa").glob("*.json")) == ["episode_short.json"]
    (run / "out" / f"{parts[1]}.json").write_text(json.dumps(part))
    assert follow.follow_once(board, seen) == 1
    assert json.loads((board / "qa" / "episode_long.json").read_text())["stitched_parts"] == 2
    assert follow.follow_once(board, {}) == 0                        # restarted: nothing is rewritten


def test_every_burst_of_jaw_moves_is_sent_densely_with_its_own_camera(monkeypatch):
    """Jaw moves no more than JAW_JOIN_S apart are one burst, whatever the measured spacing says (a miss, a slip and a
    grasp look alike there). Each burst gets instants every JAW_DENSE_STEP_S around it, showing the top camera and the
    moving gripper's own camera only, skipping instants a regular one covers; when the dense instants would pass
    JAW_DENSE_MAX_SHARE of the regular ones, the bursts with the most closes are kept. No moves, no text."""
    from label import episode as me
    ep = {"context": {"fps": 30, "cameras": {}}, "sources": {"exo": {}, "left": {}, "right": {}},
          "jaws": {"left": [{"t": 2.0, "kind": "close"}, {"t": 2.5, "kind": "open"}, {"t": 3.3, "kind": "close"},
                            {"t": 9.0, "kind": "open"}],
                   "right": [{"t": 12.0, "kind": "close"}], }}
    got = [(b["view"], b["t0"], b["t1"], b["closes"]) for b in me.jaw_bursts(ep)]
    assert got == [("left", 2.0, 3.3, 2), ("left", 9.0, 9.0, 0), ("right", 12.0, 12.0, 1)]
    ks = list(range(0, 600, 30))
    bursts, dense = me.jaw_dense(ep, ks, 600)
    assert len(bursts) == 3
    first = sorted(k for k in dense if k < 150)                   # 1.6 s to 3.9 s every 0.2 s, less those by a whole second
    assert first[0] == 48 and first[-1] == 114 and all(abs(k - r) > 3 for k in first for r in ks)
    assert all(dense[k] == {"exo", "left"} for k in first) and all(dense[k] == {"exo", "right"} for k in dense if k > 330)
    pl = {"ks": sorted(set(ks) | set(dense)), "dense": dense}
    imgs = {v: {k: None for k in pl["ks"]} for v in ("exo", "left", "right")}
    monkeypatch.setattr(me.mf, "to_jpeg", lambda im, w, q: b"j")
    steps = dict(me.timesteps(ep, pl, imgs, 64))
    assert [c for c, _ in steps[48 / 30]] == ["top", "left"] and [c for c, _ in steps[1.0]] == ["top", "left", "right"]
    monkeypatch.setattr(me, "JAW_DENSE_MAX_SHARE", 0.6)            # room for the two-close burst only
    bursts, dense = me.jaw_dense(ep, ks, 600)
    assert [(b["t0"], b["closes"]) for b in bursts] == [(2.0, 2)]
    assert me.jaw_desc(ep, {"bursts": []}) == ""
    assert "left gripper 2.00-3.30 s, 3 moves" in me.jaw_desc(ep, {"bursts": me.jaw_bursts(ep)})


def test_the_whole_run_cost_takes_each_tranche_at_its_own_rate_and_the_plan_rate_before_its_labels():
    """A labelled tranche counts at its measured cost per footage hour, an unlabelled one at the rate plan.json gives
    it, and with no rate for an unlabelled tranche there is no estimate at all."""
    from board import home
    ds = [{"dataset": "a", "plan_seconds": 7200, "seconds": 3600, "cost": 10.0},
          {"dataset": "b", "plan_seconds": 3600, "seconds": 0, "cost": 0.0}]
    assert home._projected(ds, {"a": {}, "b": {"cost_per_footage_h": 4.0}}) == 24
    assert home._projected(ds, {"a": {}, "b": {}}) is None


def test_time_left_needs_labelling_under_way():
    """A pace comes from the last hour's labels, first to last; none while labelling is paused or has barely begun."""
    now = 100_000.0
    row = lambda at: {"at": at, "seconds": 60.0}
    steady = [row(now - 1800 + 60 * i) for i in range(30)]          # one minute of footage a minute, last one now
    assert home._pace(steady, now)["footage_h_per_h"] == pytest.approx(60 * 30 / (29 * 60), rel=1e-3)
    burst = [row(now - 900 + i) for i in range(10)]                 # ten retries in ten seconds, then quiet
    assert home._pace(burst, now) is None
    assert home._pace(steady, now + 700) is None                    # nothing labelled for over ten minutes
