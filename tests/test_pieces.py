"""Labelling a long recording in parts (label/pieces.py): where it is cut, what each part holds, and how the parts'
labels are stitched back into one timeline."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from label import episode as me
from label import pieces
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
                    "config": {"timesteps_s": [0.0, 1.0]}, "usage": {"est_cost_usd": 0.5},
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
