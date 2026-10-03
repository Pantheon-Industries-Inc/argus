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


def _long_press(tmp_path, monkeypatch, measured: bool = True) -> tuple[Path, list[Path]]:
    """A 10 s head camera recording whose right glove is pressed flat by the palm from 3 s to 6 s, written in parts of
    about 1 s: the parts from 3.7 s to 5.8 s fall inside the press. Without measured, its context has no contacts (an
    upload prepared before contacts were measured)."""
    from test_touch_depth import _hdf5
    root = tmp_path / "up"
    root.mkdir()
    _hdf5(root / "kitchen_p1.hdf5", demos=1, n=300, press=(90, 180), palm=True)
    rep = f.convert(root, "ego_head", tmp_path / "eps", "touchset", 900)
    src = tmp_path / "eps" / rep["episodes"][0]["episode_id"]
    if not measured:
        ctx = json.loads((src / "context.json").read_text())
        del ctx["contacts"]
        (src / "context.json").write_text(json.dumps(ctx))
    monkeypatch.setitem(pieces.PIECE_MAX_S, "ego_head", 1.0)
    return src, pieces.write_pieces(src, tmp_path / "pieces")


def test_a_recording_prepared_without_contacts_gives_each_part_the_whole_recordings_contacts(tmp_path, monkeypatch):
    """A recording whose context has no contacts has them found once, on the whole recording, and clipped into each
    part, so a part inside a long press keeps the contact that finding it on the part's own slice would miss."""
    from label import contacts as lc
    src, parts = _long_press(tmp_path, monkeypatch, measured=False)
    assert "contacts" not in json.loads((src / "context.json").read_text())      # the recording's own is unchanged
    inside = [p for p in parts if 3.0 < me.load(p)["context"]["piece"]["t0_s"]
              and me.load(p)["context"]["piece"]["t1_s"] < 6.0]
    assert inside
    records = {}
    for p in inside:
        e = me.load(p)
        assert [c["id"] for c in e["context"]["contacts"]] == ["c1"]
        assert lc.find(e["signals"], e["signal_meta"], np.arange(len(e["state"])) / 30.0) == []   # its slice alone
        r = me.build_request(p)
        assert "contacts" in r["blocks"] and "CONTACTS:" in r["prompt"] and r["contact_views"]["shown"] == ["c1"]
        records[p] = {"contacts": r["contacts"], "contact_views": {"shown": ["c1"], "strips": {}}}
    # the stitched record lists the contact the parts were asked about, on the recording's own clock
    out = pieces.stitch(src, [(json.loads((p / "context.json").read_text()),
                               {"episode_dir": str(p), "parse_ok": True, "model": "m", "usage": {},
                                "config": {"timesteps_s": [0.0]}, "labels": {"task_summary": p.name},
                                **records.get(p, {})}) for p in parts])
    (c,) = out["contacts"]
    assert c["id"] == "c1" and abs(c["start_s"] - 3.0) < 0.05 and abs(c["end_s"] - 6.0) < 0.05


def test_a_part_inside_a_long_press_keeps_the_contact_the_whole_recording_shows(tmp_path, monkeypatch):
    """Touch is judged once, on the whole recording: a part inside a long press has no rest of its own, so its slice
    alone does not read as touch, and the part takes the recording's verdict from its context."""
    from label import signals as sg
    src, parts = _long_press(tmp_path, monkeypatch)
    assert len(parts) >= 3
    assert all("touch" not in s for s in json.loads((src / "context.json").read_text())["signals"])
    inside = []
    for p in parts:
        e = me.load(p)
        assert {s["name"]: s["touch"] for s in e["context"]["signals"]} == {"right_pressure": True,
                                                                            "right_hand_landmarks": False}
        m = e["signal_meta"]["right_pressure"]
        a = np.asarray(e["signals"]["right_pressure"], dtype=np.float64)
        if e["context"]["contacts"] and not sg.is_touch("right_pressure", a, m.get("rest"), m.get("swing")):
            inside.append(p)
    assert len(inside) >= 2                          # parts the contact covers whose own numbers do not read as touch
    for p in inside:
        r = me.build_request(p)
        assert "contacts" in r["blocks"] and "CONTACTS:" in r["prompt"] and r["contact_views"]["shown"] == ["c1"]


def test_a_stitched_record_keeps_the_contacts_its_parts_showed_and_every_parts_blocks(tmp_path, monkeypatch):
    """The stitched record lists the recording's contacts some part showed, not the ones no part was asked about, and
    its config names every block, output field and check any part had, in the order they first appear."""
    src, parts = _long_press(tmp_path, monkeypatch)
    ctx = json.loads((src / "context.json").read_text())
    ctx["contacts"].append({**ctx["contacts"][0], "id": "c9", "signals": ["right_hand_landmarks"]})
    (src / "context.json").write_text(json.dumps(ctx))
    pcs = [json.loads((p / "context.json").read_text()) for p in parts]

    def result(i):
        shown = i == 1
        blocks = ["collection_note", "no_state", "signals"] + (["contacts"] if shown else [])
        return {"episode_dir": str(parts[i]), "parse_ok": True, "model": "m", "usage": {},
                "config": {"timesteps_s": [0.0], "prompt_blocks": blocks,
                           "schema_fields": ["contacts", "contacts_missing"] if shown else [],
                           "checks_implied": ["sensor_checks"] + (["contact_checks"] if shown else [])},
                "labels": {"task_summary": f"part {i + 1}"},
                **({"contacts": [ctx["contacts"][0]], "contact_views": {"shown": ["c1"], "strips": {}}}
                   if shown else {})}
    out = pieces.stitch(src, [(pc, result(i)) for i, pc in enumerate(pcs)])
    assert [c["id"] for c in out["contacts"]] == ["c1"]
    assert out["config"]["prompt_blocks"] == ["collection_note", "no_state", "signals", "contacts"]
    assert out["config"]["schema_fields"] == ["contacts", "contacts_missing"]
    assert out["config"]["checks_implied"] == ["sensor_checks", "contact_checks"]


def test_a_contact_cut_by_our_cuts_asks_each_part_only_for_the_strips_it_shows_and_stitches_them_back(tmp_path,
                                                                                                    monkeypatch):
    """The press from 3 s to 6 s crosses several cuts: the part where it begins shows only its begin strip, the part
    where it ends only its end strip, and the parts inside it neither, so none of them is asked for a frame of a strip
    it does not show. The stitched answer takes first_touch_frame from the part that showed the begin strip and
    last_touch_frame from the part that showed the end strip, each with that strip's times on the recording's clock."""
    src, parts = _long_press(tmp_path, monkeypatch)
    results, kinds = [], []
    for i, p in enumerate(parts):
        r = me.build_request(p)
        strips = (r.get("contact_views") or {}).get("strips", {}).get("c1")
        kinds.append(sorted(strips) if strips is not None else None)
        if strips is not None:
            assert ("first_touch_frame" in r["prompt"]) == ("begin" in strips)
            assert ("last_touch_frame" in r["prompt"]) == ("end" in strips)
        answer = {"id": "c1", "touch_seen": "yes", "first_touch_frame": 10 + i, "last_touch_frame": 20 + i}
        results.append({"episode_dir": str(p), "parse_ok": True, "model": "m", "usage": {},
                        "config": {"timesteps_s": [0.0]}, "labels": {"task_summary": p.name, "contacts": [answer]},
                        **({"contacts": r["contacts"], "contact_views": r["contact_views"]} if strips is not None
                           else {})})
    shown = [k for k in kinds if k is not None]
    assert shown[0] == ["begin"] and shown[-1] == ["end"] and [] in shown[1:-1]
    first, last = kinds.index(["begin"]), kinds.index(["end"])
    pcs = [json.loads((p / "context.json").read_text()) for p in parts]
    out = pieces.stitch(src, list(zip(pcs, results)))
    (c,) = out["labels"]["contacts"]
    assert c["first_touch_frame"] == 10 + first and c["last_touch_frame"] == 20 + last
    strips = out["contact_views"]["strips"]["c1"]
    shifted = lambda i, kind: [round(x + pcs[i]["piece"]["t0_s"], 3)
                               for x in results[i]["contact_views"]["strips"]["c1"][kind]]
    assert strips["begin"] == shifted(first, "begin") and strips["end"] == shifted(last, "end")
    assert abs(strips["begin"][2] - 3.0) < 0.05 and abs(strips["end"][1] - 6.0) < 0.05


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
