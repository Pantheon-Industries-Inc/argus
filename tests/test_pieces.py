"""Labelling a long recording in parts (label/pieces.py): where it is cut, what each part holds, and how the parts'
labels are stitched back into one timeline."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from label import episode as me
from label import harness, pieces
from prepare import formats as f
from test_formats import recorder_folder


def test_only_a_recording_longer_than_the_limit_is_cut(tmp_path):
    for secs, cut in ((450.0, False), (472.0, False), (473.0, True)):
        d = tmp_path / f"episode_{int(secs)}"
        d.mkdir()
        (d / "context.json").write_text(json.dumps({"profile": "teleop_arms", "duration_s": secs}))
        assert pieces.needs_pieces(d) is cut, secs


def test_cuts_land_at_the_stillest_moment_near_each_target():
    t = np.arange(0, 1000, 0.1)
    m = np.ones_like(t)
    for still in (310.0, 690.0):                       # the only still stretches, near the thirds of 1000 s
        m[(t > still - 3) & (t < still + 3)] = 0.0
    cuts = pieces.choose_cuts(t, m, 450.0)
    assert len(cuts) == 2 and all(c["still"] for c in cuts)
    assert abs(cuts[0]["t_s"] - 310) <= 3 and abs(cuts[1]["t_s"] - 690) <= 3        # inside the still stretches
    assert pieces.choose_cuts(t[:4000], m[:4000], 450.0) == []          # 400 s fits in one request


def test_each_part_reads_the_recording_at_its_own_span(monkeypatch):
    """Every frame of the recording is in exactly one part, each part decodes the same pixels as the recording at
    the same moment on every camera, its state is the recording's, and it is told it is one part of a recording,
    with the recording's task text as context rather than its goal."""
    monkeypatch.setitem(pieces.PIECE_MAX_S, "teleop_arms", 1.0)
    with tempfile.TemporaryDirectory() as t:
        root = Path(t) / "upload"
        recorder_folder(root, n=90)
        rep = f.convert(root, "teleop_arms", Path(t) / "eps", "test", 900)
        src = Path(t) / "eps" / rep["episodes"][0]["episode_id"]
        whole = me.load(src)
        n = int(whole["sources"]["exo"]["n_frames"])
        parts = pieces.write_pieces(src, Path(t) / "pieces")
        assert [p.name for p in parts] == [f"{src.name}__p0{i}" for i in (1, 2, 3)]
        assert json.loads((src / "context.json").read_text())["pieces"]["parts"] == [p.name for p in parts]
        eps = [me.load(p) for p in parts]
        assert sum(int(e["sources"]["exo"]["n_frames"]) for e in eps) == n
        a = 0
        for i, e in enumerate(eps, 1):
            m = int(e["sources"]["exo"]["n_frames"])
            ctx = e["context"]
            assert ctx["piece"]["index"] == i and ctx["piece"]["count"] == 3 and ctx["piece"]["of"] == src.name
            assert "instruction" not in ctx and "Pick up the cube" in ctx["collection_note"]
            assert ctx["collection_note"].startswith(f"this clip is part {i} of 3 of one continuous")
            assert np.array_equal(e["state"], whole["state"][a:a + m])
            ks = [0, m - 1]
            for v in me.views(whole):
                got = me._decode_view(e, v, ks)
                want = me._decode_view(whole, v, [a + k for k in ks])
                for k in ks:
                    assert np.array_equal(np.asarray(got[k]), np.asarray(want[a + k])), (i, v, k)
            a += m


def test_stitching_puts_the_parts_back_on_the_recordings_clock(monkeypatch):
    """Times move onto the recording's clock, the lists are joined, each part's outcome becomes one task, and a
    cut-off issue at one of our cuts is set aside with the reason, while one at the recording's real end is kept."""
    monkeypatch.setitem(pieces.PIECE_MAX_S, "teleop_arms", 1.5)
    monkeypatch.setattr(pieces, "CUT_GUARD_S", 0.2)          # the guard, scaled to a 3-second recording
    with tempfile.TemporaryDirectory() as t:
        root = Path(t) / "upload"
        recorder_folder(root, n=90)
        rep = f.convert(root, "teleop_arms", Path(t) / "eps", "test", 900)
        src = Path(t) / "eps" / rep["episodes"][0]["episode_id"]
        parts = pieces.write_pieces(src, Path(t) / "pieces")
        assert len(parts) == 2
        pcs = [json.loads((p / "context.json").read_text()) for p in parts]
        cut = pcs[1]["piece"]["t0_s"]

        def result(i, end):
            return {"episode_dir": str(parts[i]), "parse_ok": True, "model": "m",
                    "config": {"timesteps_s": [0.0, 1.0], "resolution_route": {"cost_usd": 0.01 * (i + 1)}},
                    "usage": {"est_cost_usd": 0.5},
                    "labels": {"task_summary": f"part {i + 1}", "timeline": [{"start_s": 0.2, "end_s": 0.9,
                                                                              "action": "reach"}],
                               "key_events": [{"t_s": 0.5, "label": "grasp"}],
                               "completion": {"task_completed": "partial"},
                               "data_issues": [{"category": "truncated_episode", "severity": "medium", "t_s": end,
                                                "issue": "the episode ends mid-task"}]}}
        out = pieces.stitch(src, [(pcs[0], result(0, cut - 0.1)), (pcs[1], result(1, pcs[1]["piece"]["t1_s"] - cut))])
        lab = out["labels"]
        assert [s["start_s"] for s in lab["timeline"]] == [0.2, round(0.2 + cut, 3)]
        assert [k["t_s"] for k in lab["key_events"]] == [0.5, round(0.5 + cut, 3)]
        assert [x["outcome"] for x in lab["tasks"]] == ["partial", "partial"]
        assert [x["part"] for x in lab["data_issues"]] == [2]                  # the recording's own end stays
        assert [x["excluded_by"] for x in lab["_excluded"]] == ["piece_cut"]   # the cut after part 1 is ours
        assert out["stitched"]["parts"] == 2 and out["usage"]["est_cost_usd"] == 1.0
        # every part's routing call is billed, and no part was graded against the recording's task text
        assert out["config"]["resolution_route"]["cost_usd"] == 0.03 and len(out["config"]["resolution_route"]["parts"]) == 2
        assert harness.episode_cost(out) == 1.03 and out["prompt_mode"] == "inferred"



def test_a_task_carried_across_a_cut_is_one_task_again():
    """Both parts report the task they were cut in. Meeting at the cut and handling the same object, the two entries
    are one task: the first's start and description, the second's end and outcome, both entries kept. A task that
    only happens to start at a cut, with no object in common, stays its own; a task carried across two cuts is one."""
    t = lambda a, b, task, objs, outcome="success": {"start_s": a, "end_s": b, "task": task, "objects": objs,
                                                      "outcome": outcome, "completed_at_s": b}
    # the pilot's in-the-wild recording, cut at 304.833s
    tasks = [t(190.0, 304.8, "Unfold the blanket over the bed", ["floral fleece blanket", "penguin plush"]),
             t(304.833, 313.833, "Settle the floral blanket over the bed", ["floral blanket", "bed"]),
             t(313.833, 326.833, "Group the three plush toys", ["penguin plush"])]
    out = pieces.join_across_cuts(tasks, [304.833])
    assert [(x["start_s"], x["end_s"]) for x in out] == [(190.0, 313.833), (313.833, 326.833)]
    assert out[0]["task"] == "Unfold the blanket over the bed" and len(out[0]["joined_from"]) == 2
    assert out[0]["completed_at_s"] == 313.833

    # a new task starting at the cut, sharing no object with the one that ended there
    apart = [t(0, 100.0, "Fold the towel", ["towel"]), t(100.2, 150, "Stack the cups", ["red cup"])]
    assert len(pieces.join_across_cuts(apart, [100.0])) == 2

    # carried across two cuts, its outcome decided in the last part
    long = [t(0, 50.0, "Sort the blocks", ["blue block"]), t(50.1, 100.0, "Sort the blocks", ["blocks"]),
            t(100.0, 120, "Finish sorting", ["green block"], "failure")]
    one = pieces.join_across_cuts(long, [50.0, 100.0])
    assert len(one) == 1 and one[0]["end_s"] == 120 and one[0]["outcome"] == "failure"
    assert len(one[0]["joined_from"]) == 3


def test_a_recording_with_other_signals_can_be_labelled_in_parts(tmp_path, monkeypatch):
    """A LeRobot upload with one column the reader has no slot for (a base velocity) is longer than a part: each part
    loads, carrying its own rows of the signals its context lists."""
    from test_prepare import _lerobot_v21
    monkeypatch.setitem(pieces.PIECE_MAX_S, "teleop_arms", 0.5)
    n = 45
    root = tmp_path / "mobile"
    _lerobot_v21(root, n=n, extra={"observation.velocity": list(np.linspace(0, 1, n)[:, None] * [1.0, 0.0])})
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    src = tmp_path / "eps" / rep["episodes"][0]["episode_id"]
    assert json.loads((src / "context.json").read_text())["signals"]
    parts = pieces.write_pieces(src, tmp_path / "pieces")
    assert len(parts) > 1
    for p in parts:
        e = me.load(p)
        assert len(next(iter(e["signals"].values()))) == int(e["context"]["n_state_frames"])


def test_a_part_of_a_long_head_camera_recording_gets_its_subtasks_on_its_own_clock(tmp_path, monkeypatch):
    """The dataset's timed subtasks are on the recording's clock; a part's frames start at 0 at its cut."""
    from prepare import formats
    from test_formats import _clip
    (tmp_path / "v").mkdir()
    _clip(tmp_path / "v" / "a.mp4", 90)                      # 3 s at 30 fps
    ep = tmp_path / "eps" / "episode_a"
    subs = [{"t0": 0.0, "t1": 1.4, "label": "pick up the cup"}, {"t0": 1.6, "t1": 3.0, "label": "wipe the table"}]
    formats.video_views_episode(ep, {"exo": ("a", tmp_path / "v" / "a.mp4")}, "ego_head", "mine",
                                {"instruction": "clean up", "annotation_subtasks": subs})
    monkeypatch.setitem(pieces.PIECE_MAX_S, "ego_head", 1.6)
    parts = pieces.write_pieces(ep, tmp_path / "pieces")
    assert len(parts) == 2
    e = me.load(parts[1])
    t0 = e["context"]["piece"]["t0_s"]
    req = me.build_request(parts[1])
    block = req["prompt"].split("THE DATASET'S ANNOTATION FOR THIS EPISODE")[1]
    # the part's own clock runs 0..(t1 - t0); the second subtask is at 1.6-3.0 s of the recording
    assert f"{1.6 - t0:.1f}-{3.0 - t0:.1f}s  wipe the table" in block, block
    first = me.load(parts[0])["context"]["annotation_subtasks"]
    assert [x["label"] for x in first] == ["pick up the cup"] + (["wipe the table"] if t0 > 1.6 else [])
    assert all(x["t0"] >= 0 for x in e["context"]["annotation_subtasks"])
