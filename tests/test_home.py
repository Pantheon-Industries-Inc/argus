"""The home view's numbers (board/home.py) and labels followed into a board while a run writes them (board/follow.py)."""
import json
import os
import time
from pathlib import Path

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
    # verbs are the tasks' own (place, fold, stack), one per subtask, never the events' grasp or lift
    assert sorted(k for k, *_ in s["total"]["top_verbs"]) == ["fold", "place", "stack"]
    # the labeler's own verb for a task wins over the sentence's first word
    d = json.loads((qa / "session.json").read_text())
    d["tasks"][0]["verb"] = "Fold in half"
    d["tasks"][1]["verb"] = "stack"
    named = home.summarize(qa / "session.json", d, _counts_all)
    assert named["verbs"] == {"fold in half": 1, "stack": 1}
    d["tasks"][1]["verb"] = "None"                      # the labeler saying no action happened counts no verb
    assert home.summarize(qa / "session.json", d, _counts_all)["verbs"] == {"fold in half": 1}
    # verbs naming one action count under the one name board/verbs.py gave them
    merged = home.stats([named, {**named, "file": "b.json", "verbs": {"stack up": 2}}], {}, time.time(),
                        names={"stack up": "stack", "fold in half": "fold"})
    assert dict((k, n) for k, n, _ in merged["total"]["top_verbs"]) == {"stack": 3, "fold": 1}


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


def test_only_a_quick_shut_reopen_reclose_reaches_the_facts():
    """A close where the jaws met, reopened soon and closed again on something is sent as a possible missed grasp; a
    clap that shuts, opens and shuts again in the air, a carry of something thin that is then released and the jaws
    shut, and a close on an object, are not. An episode with no such stretch gets no text at all, and no frame is
    added either way."""
    import numpy as np
    from label import episode as me
    from prepare import jaws
    fps = 30.0
    o = np.full(600, 160.0)                              # shut at rest
    o[30:60] = 205.0                                      # open, shut on nothing at 2s ...
    o[75:100] = 205.0                                     # ... reopen at 2.5s, close on something at 3.33s: a miss
    o[100:120] = 185.0                                    #     and its retry; held, released at 4s and kept open,
    o[120:180] = 205.0                                    # then a close on an object at 6s, held, released at 9s
    o[180:270] = 185.0
    o[270:280] = 205.0                                    # release, then shut the empty jaws at 9.33s
    o[330:360] = 205.0                                    # open, shut on something thin at 12s, carried 2s,
    o[420:430] = 205.0                                    # released at 14s and shut again at 14.33s
    o[450:470] = 205.0                                    # a clap: shut at 15.67s, open at 16s, shut again at 16.5s
    o[480:495] = 205.0
    evs = jaws.events(o, fps)
    jaws.mark_shut(evs, o)
    got = [[(round(e["t"], 1), e["kind"]) for e in st] for st in jaws.retries(evs)]
    assert got == [[(2.0, "close"), (2.5, "open"), (3.3, "close")]]
    ep = {"jaws": {"left": evs, "right": None}, "context": {"cameras": {"left": {"name": "left"}}}}
    text = me.jaw_block(ep)
    assert "left gripper: closed 2.00s shut, opened 2.50s, closed 3.33s on something" in text and "6.00" not in text
    assert me.jaw_block({"jaws": {"left": evs[3:], "right": None}}) == ""


def test_the_whole_run_cost_takes_each_tranche_at_its_own_rate_and_the_plan_rate_before_its_labels():
    """A labelled tranche counts at its measured cost per footage hour, an unlabelled one at the rate plan.json gives
    it, and with no rate for an unlabelled tranche there is no estimate at all."""
    from board import home
    ds = [{"dataset": "a", "plan_seconds": 7200, "seconds": 3600, "cost": 10.0},
          {"dataset": "b", "plan_seconds": 3600, "seconds": 0, "cost": 0.0}]
    assert home._projected(ds, {"a": {}, "b": {"cost_per_footage_h": 4.0}}) == 24
    assert home._projected(ds, {"a": {}, "b": {}}) is None
