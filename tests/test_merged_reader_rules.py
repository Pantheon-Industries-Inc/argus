"""Reader and board rules retain the same evidence when used together."""
import json
import os
import zipfile

import numpy as np
import pytest

from board import build, clips
from board.families import Families
from label import atomic, episode
from prepare import formats


@pytest.mark.parametrize("writer", ["notes", "depth"])
def test_context_writers_keep_the_old_file_when_a_write_stops(tmp_path, monkeypatch, writer):
    p = tmp_path / "context.json"
    original = {"arbitrary": {"start_s": 37, "nested": [1, {"claim": "kept"}]}, "reader_issues": []}
    p.write_text(json.dumps(original))
    before = p.read_bytes()

    def stopped(fd):
        # the stop lands where every writer's bytes reach the disk, so no writer can step around it
        os.ftruncate(fd, 7)
        raise OSError("write stopped")

    def change():
        if writer == "notes":
            formats.add_table_notes(tmp_path, dict(original), [{"table": "notes.csv", "note": "claim"}])
        else:
            clips.record_depth(tmp_path, "exo", [{"kind": "depth_clip_failed", "what": "failed", "camera": "exo"}])

    with monkeypatch.context() as m:
        m.setattr(atomic.os, "fsync", stopped)
        with pytest.raises(OSError, match="write stopped"):
            change()
    assert p.read_bytes() == before
    assert sorted(x.name for x in tmp_path.iterdir()) == ["context.json"]   # no temporary file is left behind
    change()
    assert json.loads(p.read_text())["arbitrary"] == original["arbitrary"]


def test_catalog_reader_kinds_have_unique_families_and_reach_classification():
    families = Families()
    mapped = [kind for family in families.defs for kind in family.get("reader_issues", [])]
    assert mapped and len(mapped) == len(set(mapped))
    assert len(families.catalog()) == len(families.defs)
    for family in families.defs:
        for kind in family.get("reader_issues", []):
            slug = family["slug"]
            assert families.reader_family(kind) == slug
            doc = {}
            build.add_reader_issues(doc, {"reader_issues": [{"kind": kind, "what": "controlled finding"}]})
            assert doc["dataset_checks"]["reader_issues"][0]["family"] == slug
            group = "counted" if family["list"] in ("data", "mistake") else "not_counted"
            assert slug in families.classify(doc)[group]


@pytest.mark.parametrize("recording_fault", [False, True])
def test_depth_failure_emission_preserves_recording_fault_and_processing_failure(tmp_path, recording_fault):
    (tmp_path / "context.json").write_text(json.dumps({"reader_issues": []}))
    error = clips.DepthNotDecodable("controlled bad stream") if recording_fault else OSError("controlled cut failure")
    issue = clips.depth_failed(tmp_path, "exo", error)
    clips.record_depth(tmp_path, "exo", [issue])
    saved = json.loads((tmp_path / "context.json").read_text())
    doc = {}
    build.add_reader_issues(doc, saved)
    (finding,) = doc["dataset_checks"]["reader_issues"]
    assert finding["camera"] == "exo" and finding["what"] == issue["what"]
    families = Families()
    if recording_fault:
        assert finding["kind"] == "depth_not_decodable" and finding["family"] == "depth-missing"
        assert "depth-missing" in families.classify(doc)["counted"]
    else:
        assert finding["kind"] == "depth_clip_failed" and finding["family"] == "read-with-a-limit"
        assert "read-with-a-limit" in families.classify(doc)["not_counted"]
        assert "read-with-a-limit" not in families.classify(doc)["counted"]


def test_malformed_metadata_keeps_valid_rows_and_reaches_the_board(tmp_path):
    import pandas as pd
    from test_habit_generic import native, source_hashes
    native(tmp_path)
    table = tmp_path / "meta/tasks.jsonl"
    table.write_text(table.read_text() + "{controlled invalid row\n")
    before = source_hashes(tmp_path)
    parsed = formats.read_root(tmp_path, "")
    data_path = tmp_path / "data/chunk-000/episode_000000.parquet"
    frames = pd.read_parquet(data_path)
    ctx = {"episode_index": 0, "fps": 30, "n_state_frames": len(frames)}
    formats.lerobot_annotations(ctx, parsed, frames, data_path)
    doc = {}
    build.add_reader_issues(doc, ctx)
    findings = [issue for issue in doc["dataset_checks"]["reader_issues"]
                if issue["kind"] == "metadata_unreadable"]
    assert findings and all(issue["family"] == "metadata-unreadable" for issue in findings)
    assert any("tasks.jsonl" in issue["what"] for issue in findings)
    assert "metadata-unreadable" in Families().classify(doc)["counted"]
    # the task code holds one text over every row: the episode's claim about the whole recording, never a step
    assert any(span["label"] == "coarse recorded task" for span in ctx["annotation_task_claims"])
    assert source_hashes(tmp_path) == before


def test_reader_state_reasons_have_request_wording_and_times_do_not_print_negative_zero():
    assert set(episode.STATE_WORDING) == set(formats.STATE_WHY)
    assert episode.seconds(-0.001) == "0.00 s" and episode.tenths(-0.01) == "0.0s"


def test_bookkeeping_and_quiet_channels_keep_their_arrays_without_false_gaps(tmp_path):
    signals = formats.Signals()
    signals["system_info"] = np.array([[1, 2, np.inf], [1, 3, 4]], dtype=np.float32)
    signals.meta["system_info"] = {"names": ["pid", "memory_kb", "load"]}
    signals["camera_info"] = np.full((2, 1), np.nan, dtype=np.float32)
    signals.no_gaps.add("camera_info")
    ctx = {"n_state_frames": 2, "fps": 30}
    formats.write_signals(tmp_path, ctx, signals)
    with np.load(tmp_path / "signals.npz") as arrays:
        assert arrays["s0"].shape == (2, 3) and np.isnan(arrays["s0"][0, 2])
        assert arrays["s0_readings"].shape == (2, 1)
    assert ctx["source"]["bookkeeping"][0]["columns"] == [0, 1]
    assert not any(x.get("signal") == "camera_info" for x in ctx.get("reader_issues", []))


@pytest.mark.parametrize("why", ["layout", "not_recorded", "unreadable", "short", "assumed_clock"])
def test_reader_state_reasons_reach_the_request_with_placeholders(tmp_path, why):
    from test_board_sensors import _episode
    from test_reader_tables import _clip
    ep = _episode(tmp_path / "episodes")
    ctx = json.loads((ep / "context.json").read_text())
    _clip(ep / "exo.mp4", 300)
    ctx.update(state_kind="none", placeholder_frames={"exo": [[45, 45]]})
    formats.no_state(ctx, formats.StateNote("The contributor cannot establish the arm state", why))
    (ep / "context.json").write_text(json.dumps(ctx))
    loaded = episode.load(ep)
    plan = episode.plan(loaded)
    images = episode.frames(loaded, plan)
    assert 45 not in images["exo"]
    text = episode.episode_text(loaded, plan, 448, 252, (640, 480))
    assert "RECORDED STATE" in text
    assert "No camera could be decoded at the planned instants 1.50 s" in text
    assert loaded["decode_failed"] == {"exo": [45]}
    if why == "layout":
        assert "in the layout our checks read" in text
    else:
        assert episode.STATE_WORDING[why] in text
    doc = {}
    build.add_reader_issues(doc, loaded["context"], {})
    assert loaded["context"]["placeholder_frames"] == {"exo": [[45, 45]]}


@pytest.mark.parametrize("archive", [False, True])
def test_a_failed_camera_keeps_its_table_and_good_archived_footage(tmp_path, archive):
    from test_reader_tables import _clip
    upload = tmp_path / "upload"
    _clip(upload / "take" / "top.mp4", 60)
    (upload / "take" / "wrist_left.mp4").write_bytes(b"broken camera")
    (upload / "take" / "traj.csv").write_text("force\n" + "\n".join(str((k % 7) * 0.3) for k in range(60)))
    (upload / "take" / "instruction.txt").write_text("pick the cup")
    root = upload
    if archive:
        root = tmp_path / "upload.zip"
        with zipfile.ZipFile(root, "w", zipfile.ZIP_STORED) as zipped:
            for path in sorted(upload.rglob("*")):
                if path.is_file():
                    zipped.write(path, path.relative_to(upload))
            zipped.writestr("take/private.bin", b"not readable")
        raw = bytearray(root.read_bytes())
        raw[raw.rfind(b"PK\x01\x02") + 8] |= 1
        root.write_bytes(raw)
    report = formats.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    assert len(report["episodes"]) == 1 and not report["failed"]
    ep = tmp_path / "eps" / report["episodes"][0]["episode_id"]
    ctx = json.loads((ep / "context.json").read_text())
    assert any(x["kind"] == "camera_not_decodable" for x in ctx["reader_issues"])
    loaded = episode.load(ep)
    assert np.allclose(next(iter(loaded["signals"].values()))[:, 0], [(k % 7) * 0.3 for k in range(60)])
    assert episode.build_request(ep)["prompt"].count("pick the cup") >= 1
    if archive:
        assert any("private.bin" in line and "password" in line for line in report["used"])


def test_owned_task_text_survives_camera_reanchoring(tmp_path):
    from test_reader_tables import _clip
    upload = tmp_path / "upload"
    _clip(upload / "top.mp4", 60)
    _clip(upload / "wrist_left.mp4", 60)
    np.save(upload / "top_times.npy", np.arange(60) / 30)
    np.save(upload / "wrist_left_times.npy", 0.5 + np.arange(60) / 30)
    (upload / "instruction.txt").write_text("put the cup on the shelf")
    report = formats.convert(upload, "teleop_arms", tmp_path / "eps", "test", 900)
    ep = tmp_path / "eps" / report["episodes"][0]["episode_id"]
    before = json.loads((ep / "context.json").read_text())
    assert before["instruction"] == "put the cup on the shelf"
    clips.record_cameras(ep, {}, {"exo": "damaged"}, {"exo"})
    after = json.loads((ep / "context.json").read_text())
    assert after["instruction"] == before["instruction"]
    assert after["source"]["note_files"] == before["source"]["note_files"]
    assert after["clock_zero_s"] == 0
    assert after["clock_start_s"] == pytest.approx(0.5)
    request = episode.build_request(ep)
    assert request["timesteps"][0] == 0 and "put the cup on the shelf" in request["prompt"]


def test_a_short_episode_uses_the_same_edge_slack_for_cameras_and_signals():
    times = np.arange(30) / 30
    ctx = {}
    formats.camera_span_issues(ctx, {"exo": times, "left": times[12:]}, "exo", {"exo": "top", "left": "wrist_left"}, {"profile": "teleop_arms"})
    signal = np.ones((30, 1))
    signal[:12] = np.nan
    assert any(x["kind"] == "camera_coverage_partial" for x in ctx["reader_issues"])
    assert any(x["kind"] == "signal_partial_span" for x in formats.signal_gaps("force", signal, times))
    assert not formats.covers_footage(times[12], times[-1], times)


def test_trimming_placeholder_footage_keeps_recorded_state(tmp_path):
    from test_reader_cameras import _lerobot_images
    root = tmp_path / "lr"
    _lerobot_images(root, bad=(0, 45), n=90, wrist=True)
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"]["observation.state"].update(
        names=[name for side in ("left", "right")
               for name in [*[f"{side}_joint{index}" for index in range(1, 7)], f"{side}_gripper"]],
        units=(["rad"] * 6 + ["fraction"]) * 2)
    info_path.write_text(json.dumps(info))
    report = formats.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    ep = tmp_path / "eps" / report["episodes"][0]["episode_id"]
    before = json.loads((ep / "context.json").read_text())
    state = np.load(ep / "state.npz")["state"].copy()
    after = formats.trim_episode(ep, 1.0)
    assert after["placeholder_frames"] == {"exo": [[0, 0]]}
    assert after["state_kind"] == before["state_kind"] == "joints"
    np.testing.assert_array_equal(np.load(ep / "state.npz")["state"], state[:30])
    loaded = episode.load(ep)
    plan = episode.plan(loaded)
    assert plan["n"] == 30 and plan["state_usable"]
    images = episode.frames(loaded, plan)
    assert 0 not in images["exo"] and 0 in images["left"]
    assert "RECORDED MOTION" in episode.build_request(ep)["prompt"]
