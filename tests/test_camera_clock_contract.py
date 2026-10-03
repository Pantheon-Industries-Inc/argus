"""Stored captures and derived media share one consumer clock after every writer."""
import json

import av
import numpy as np
import pytest

from board import build, clips, sensors
from board.hands import probe_pts
from label import episode, pieces
from prepare import formats
from test_camera_timing import recording, video


def clock_recording(tmp_path, zero=0.5):
    ep = recording(tmp_path, zero + np.arange(90) / 30)
    ctx = json.loads((ep / "context.json").read_text())
    ctx.update(clock_zero_s=zero, arbitrary={"start_s": 37, "text": "captured at 37 s"})
    ctx["contacts"] = [{"id": "c1", "signals": [], "start_s": zero + 0.6, "end_s": zero + 2.7,
                        "peak_s": zero + 1.5, "dips_s": [zero + 1.6],
                        "from_start": False, "to_end": False}]
    ctx["annotation_subtasks"] = [{"t0": zero + 0.55, "t1": zero + 2.75, "text": "move"}]
    ctx["reader_issues"] = [{"kind": "signal_gap", "t0_s": zero + 1.2, "t1_s": zero + 1.3,
                             "what": "Uploader reports a gap at 37 s", "extra": {"t0_s": 37}}]
    ctx["unshown_cameras"] = [{"packed": str(tmp_path / "exo.mp4"), "n_frames": 90, "fps": 30,
                              "name": "thermal", "start_s": zero + 0.6}]
    (ep / "context.json").write_text(json.dumps(ctx))
    return ep, ctx


@pytest.mark.parametrize("zero", [0.0, 0.5])
def test_splitting_preserves_raw_parent_times_across_repeated_loads(tmp_path, monkeypatch, zero):
    ep, raw = clock_recording(tmp_path, zero)
    before = episode.load(ep)["context"]
    monkeypatch.setattr(pieces, "piece_max", lambda ctx: 1.0)
    for _ in range(2):
        parts = pieces.write_pieces(ep, tmp_path / "parts")
        assert len(parts) == 3
        saved = json.loads((ep / "context.json").read_text())
        assert {k: v for k, v in saved.items() if k != "pieces"} == raw
        loaded = episode.load(ep)["context"]
        assert {k: v for k, v in loaded.items() if k != "pieces"} == before


def test_parts_place_every_typed_contributor_on_their_own_capture_clock(tmp_path, monkeypatch):
    ep, raw = clock_recording(tmp_path)
    monkeypatch.setattr(pieces, "piece_max", lambda ctx: 1.0)
    for part in pieces.write_pieces(ep, tmp_path / "parts"):
        loaded = episode.load(part)
        ctx, t0 = loaded["context"], loaded["context"]["piece"]["t0_s"]
        assert "clock_zero_s" not in ctx and episode.frame_time(loaded, 0) == 0
        assert ctx["reader_issues"][0]["t0_s"] == pytest.approx(1.2 - t0, abs=0.001)
        assert ctx["reader_issues"][0]["t1_s"] == pytest.approx(1.3 - t0, abs=0.001)
        assert ctx["reader_issues"][0]["what"] == ("Full recording clock note. " if t0 else "") + raw["reader_issues"][0]["what"]
        assert ctx["reader_issues"][0]["extra"] == {"t0_s": 37}
        assert ctx["unshown_cameras"][0]["start_s"] == pytest.approx(0.6 - t0, abs=0.001)
        assert ctx["arbitrary"] == raw["arbitrary"]
        assert sensors.clip_times(part, ctx, len(loaded["state"]))[0] == 0
        if ctx["contacts"][0]["dips_s"]:
            assert ctx["contacts"][0]["dips_s"] == pytest.approx([1.6 - t0], abs=0.001)


@pytest.mark.parametrize("explicit_zero", [True, False])
def test_parts_keep_parent_join_offsets_when_its_first_capture_is_after_zero(tmp_path, monkeypatch, explicit_zero):
    ep, raw = clock_recording(tmp_path)
    if explicit_zero:
        raw["clock_zero_s"] = 0.0
    else:
        raw.pop("clock_zero_s")
    (ep / "context.json").write_text(json.dumps(raw))
    parent = episode.load(ep)
    before = parent["context"]
    monkeypatch.setattr(pieces, "piece_max", lambda ctx: 1.0)
    parts = pieces.write_pieces(ep, tmp_path / "parts")
    offset, stitched = 0, []
    for part in parts:
        loaded = episode.load(part)
        ctx, n = loaded["context"], len(loaded["state"])
        start = episode.frame_time(parent, offset)
        assert ctx["piece"]["t0_s"] == pytest.approx(start, abs=0.001)
        assert episode.frame_time(loaded, 0) == 0
        assert ctx["reader_issues"][0]["t0_s"] == pytest.approx(1.7 - start, abs=0.001)
        assert ctx["reader_issues"][0]["t1_s"] == pytest.approx(1.8 - start, abs=0.001)
        assert ctx["unshown_cameras"][0]["start_s"] == pytest.approx(1.1 - start, abs=0.001)
        assert sensors.clip_times(part, ctx, n)[0] == 0
        stitched.append((ctx, {"episode_dir": str(part), "parse_ok": True, "model": "m", "usage": {},
                              "config": {"timesteps_s": [0]}, "labels": {"task_summary": "move"},
                              "contacts": ctx["contacts"], "contact_views": {"shown": ["c1"], "strips": {}}}))
        offset += n
    assert offset == 90 and parts
    first = episode.load(parts[0])["context"]
    assert first["annotation_subtasks"][0]["t0"] == pytest.approx(0.55, abs=0.001)
    assert episode.load(parts[-1])["context"]["piece"]["t1_s"] == 3.5
    joined = pieces.stitch(ep, stitched)["contacts"][0]
    assert joined["start_s"] == 1.1 and joined["end_s"] == 3.2
    assert joined["peak_s"] == 2.0 and joined["dips_s"] == [2.1]
    saved = json.loads((ep / "context.json").read_text())
    assert {k: v for k, v in saved.items() if k != "pieces"} == raw
    assert {k: v for k, v in episode.load(ep)["context"].items() if k != "pieces"} == before


@pytest.mark.parametrize("zero,start,skip,first", [(0.5, 0.6, 0, 0.1), (0.5, 0.4, 3, 0.0),
                                                  (0.5, None, 15, 0.0), (0.0, 0.6, 0, 0.6),
                                                  (0.0, -0.1, 3, 0.0)])
def test_unshown_clips_trim_only_pre_zero_frames_and_encode_the_episode_start(tmp_path, zero, start, skip, first):
    ep, ctx = clock_recording(tmp_path, zero)
    if start is None:
        ctx["unshown_cameras"][0].pop("start_s")
    else:
        ctx["unshown_cameras"][0]["start_s"] = start
    (ep / "context.json").write_text(json.dumps(ctx))
    raw = (ep / "context.json").read_bytes()
    job = [j for j in clips.episode_jobs(ep, tmp_path / "clips", True) if j[-1] == "unshown1"][0]
    clips.extract_one(*job[:4], clips.find_ffmpeg(), 1, *job[4:12])
    _, _, tb, pts = probe_pts(job[3])
    assert job[7] == skip and job[6] == pytest.approx(first)
    assert len(pts) == 90 - skip and float(pts[0] * tb) == pytest.approx(first, abs=1e-6)
    with av.open(str(job[3])) as src:
        pixels = [f.to_ndarray(format="rgb24").mean() for f in src.decode(video=0)]
    assert pixels[0] == pytest.approx(2 * skip, abs=1.5)
    assert pixels[-1] == pytest.approx(178, abs=1.5)
    assert (ep / "context.json").read_bytes() == raw


def test_parts_move_depth_capture_times_with_colour_times_without_moving_packet_indices(tmp_path, monkeypatch):
    ep, raw = clock_recording(tmp_path)
    np.save(ep / "depth_kmap.npy", np.arange(90))
    (ep / "depth.json").write_text(json.dumps({"exo": {"packed": "depth.mkv", "kmap": "depth_kmap.npy"}}))
    depth_t = 0.5 + np.arange(90) / 30
    np.savez(ep / "depth_times.npz", depth_exo=depth_t, depth_exo_pts=np.arange(90))
    monkeypatch.setattr(pieces, "piece_max", lambda ctx: 1.0)
    for part in pieces.write_pieces(ep, tmp_path / "parts"):
        loaded = episode.load(part)
        with np.load(part / "depth_times.npz") as times:
            first = int(loaded["depth"]["exo"]["km"][0])
            assert times["depth_exo"][first] == pytest.approx(episode.frame_time(loaded, 0))
            assert np.array_equal(times["depth_exo_pts"], np.arange(90))
    with np.load(ep / "depth_times.npz") as times:
        assert np.array_equal(times["depth_exo"], depth_t)


@pytest.mark.parametrize("packed", [False, True])
def test_part_clips_decode_the_same_saved_packets_as_the_request(tmp_path, monkeypatch, packed):
    ep = recording(tmp_path, 0.5 + np.arange(90) / 30)
    source = tmp_path / "exo.mp4"
    prefix = 5 if packed else 0
    video(source, 90 + prefix, pattern=True)
    starts = 0.4 + np.cumsum(np.resize([0.021, 0.045, 0.034], 90 + prefix))
    clips.retime(source, starts, main=True)
    _, _, tb, pts = probe_pts(source)
    with np.load(ep / "times.npz") as z:
        times = {key: z[key] for key in z.files}
    times["exo_pts"] = np.asarray(pts[prefix:])
    np.savez(ep / "times.npz", **times)
    ctx = json.loads((ep / "context.json").read_text())
    ctx["clock_zero_s"] = 0.5
    (ep / "context.json").write_text(json.dumps(ctx))
    monkeypatch.setattr(pieces, "piece_max", lambda ctx: 1.0)
    for part in pieces.write_pieces(ep, tmp_path / "parts"):
        loaded = episode.load(part)
        n = int(loaded["sources"]["exo"]["n_frames"])
        selected = [0, n // 2, n - 1]
        expected = episode._decode_view(loaded, "exo", selected)
        job = clips.episode_jobs(part, tmp_path / "clips", True)[0]
        clips.extract_one(*job[:4], clips.find_ffmpeg(), 1, *job[4:12])
        with av.open(str(job[3])) as src:
            frames = list(src.decode(video=0))
        assert len(frames) == n
        for k in selected:
            got = frames[k].to_ndarray(format="rgb24")
            want = np.asarray(expected[k].convert("RGB"))
            assert np.abs(got.astype(float) - want).mean() < 12


def test_camera_cut_issues_round_trip_on_the_capture_clock_without_drift(tmp_path):
    ep, original = clock_recording(tmp_path)
    for _ in range(2):
        clips.record_cameras(ep, {"exo": {"clip_frames": 30, "episode_frames": 90}}, {}, {"exo"})
        raw = json.loads((ep / "context.json").read_text())
        issue = [x for x in episode.load(ep)["context"]["reader_issues"] if x.get("clip_frames")][0]
        stored = [x for x in raw["reader_issues"] if x.get("clip_frames")][0]
        assert stored["t0_s"] == 1.5 and stored["t1_s"] == 3.5
        assert issue["t0_s"] == 1.0 and issue["t1_s"] == 3.0
        assert "ends at 1.00 s" in issue["what"] and "episode ends at 3.00 s" in issue["what"]
        assert raw["reader_issues"][0] == original["reader_issues"][0]
        board = {}
        build.add_reader_issues(board, episode.load(ep)["context"], {})
        shown = [x for x in board["dataset_checks"]["reader_issues"] if "ends at 1.00" in x["what"]][0]
        assert shown["t0_s"] == 1.0 and shown["t1_s"] == 3.0


def test_remeasured_camera_spans_store_raw_markers_and_describe_the_consumer_clock(tmp_path):
    ep = recording(tmp_path, 0.5 + np.arange(90) / 30, 0.9 + np.arange(60) / 30)
    ctx = json.loads((ep / "context.json").read_text())
    ctx["clock_zero_s"] = 0.5
    src = json.loads((ep / "sources.json").read_text())
    for _ in range(2):
        clips.respan(ep, ctx, src)
        (ep / "context.json").write_text(json.dumps(ctx))
        shown = episode.load(ep)["context"]["reader_issues"]
        late = [x for x in shown if "starts at" in x["what"]][0]
        early = [x for x in shown if "ends at" in x["what"]][0]
        assert late["t0_s"] == 0 and late["t1_s"] == pytest.approx(0.4)
        assert early["t0_s"] == pytest.approx(2.4) and early["t1_s"] == 3.0
        assert "starts at 0.40 s" in late["what"] and "ends at 2.40 s" in early["what"]


@pytest.mark.parametrize("keep_clock,end", [(False, 1.0), (True, 1.4)])
def test_new_short_camera_issues_follow_a_main_camera_removed_in_the_same_run(tmp_path, keep_clock, end):
    ep = recording(tmp_path, 0.5 + np.arange(90) / 30, 0.9 + np.arange(90) / 30)
    ctx = json.loads((ep / "context.json").read_text())
    ctx["clock_zero_s"] = 0.5
    (ep / "context.json").write_text(json.dumps(ctx))
    clips.record_cameras(ep, {"left": {"clip_frames": 30, "episode_frames": 90}}, {"exo": "damaged"},
                         {"exo", "left"}, keep_clock=keep_clock)
    shown = [x for x in episode.load(ep)["context"]["reader_issues"] if x.get("clip_frames")][0]
    assert shown["t0_s"] == pytest.approx(end)
    assert f"ends at {end:.2f} s" in shown["what"]


def test_depth_decode_issues_round_trip_with_the_actual_colour_capture_times(tmp_path, monkeypatch):
    ep, original = clock_recording(tmp_path)
    depth = tmp_path / "depth.mkv"
    writer = formats.DepthWriter(depth)
    for k in range(90):
        writer.add(0.5 + k / 30, np.full((64, 96), 1000 + k, np.uint16), 0.001)
    writer.close()
    pr = formats.probe_depth(depth)
    np.save(ep / "depth_kmap.npy", np.arange(90))
    (ep / "depth.json").write_text(json.dumps({"exo": {"packed": str(depth), "kmap": "depth_kmap.npy",
                                                       "scale_m": 0.001}}))
    np.savez(ep / "depth_times.npz", depth_exo=0.5 + np.arange(90) / 30, depth_exo_pts=pr["pts"])
    job = clips.episode_jobs(ep, tmp_path / "clips", True)[0]
    clips.extract_one(*job[:4], clips.find_ffmpeg(), 1, *job[4:12])
    decode = clips.decoded_frames
    def damaged(src, stream):
        return (fr for fr in decode(src, stream) if fr.pts != pr["pts"][30])
    monkeypatch.setattr(clips, "decoded_frames", damaged)
    for _ in range(2):
        issues = clips.extract_depth(ep, "exo", job[3], tmp_path / "depth.mp4", 1)
        clips.record_depth(ep, "exo", issues)
        raw = json.loads((ep / "context.json").read_text())
        stored = [x for x in raw["reader_issues"] if x["kind"] == clips.DEPTH_NOT_DECODABLE][0]
        issue = [x for x in episode.load(ep)["context"]["reader_issues"] if x["kind"] == clips.DEPTH_NOT_DECODABLE][0]
        assert stored["t0_s"] == stored["t1_s"] == 1.5
        assert issue["t0_s"] == issue["t1_s"] == 1.0
        assert "from 1.00 s to 1.00 s" in issue["what"]
        assert raw["reader_issues"][0] == original["reader_issues"][0]
        assert probe_pts(tmp_path / "depth.mp4")[3] == probe_pts(job[3])[3]


@pytest.mark.parametrize("first", [0.0, 0.5])
def test_part_request_and_board_attribute_shifted_reader_notes_to_the_full_recording(tmp_path, monkeypatch, first):
    ep, raw = clock_recording(tmp_path, first)
    raw["clock_zero_s"] = 0.0
    raw["reader_issues"].extend([
        {"kind": "signal_gap", "what": "An untimed uploader note says 37 s"},
        {"kind": "signal_gap", "t0_s": "unknown", "what": "An unknown clock says 37 s"}])
    (ep / "context.json").write_text(json.dumps(raw))
    parent_request = episode.build_request(ep)["prompt"]
    monkeypatch.setattr(pieces, "piece_max", lambda ctx: 1.0)
    for part in pieces.write_pieces(ep, tmp_path / "parts"):
        ctx = json.loads((part / "context.json").read_text())
        t0 = ctx["piece"]["t0_s"]
        request = episode.build_request(part)
        board = {}
        build.add_context(board, ctx, part, {})
        issue = board["dataset_checks"]["reader_issues"][0]
        expected = ("Full recording clock note. " if t0 else "") + raw["reader_issues"][0]["what"]
        assert issue["what"] == expected
        assert issue["t0_s"] == pytest.approx(first + 1.2 - t0, abs=0.001)
        if t0:
            assert expected in request["prompt"]
        assert ctx["reader_issues"][1:] == raw["reader_issues"][1:]
        assert raw["reader_issues"][1]["what"] not in ctx["collection_note"]
        assert raw["reader_issues"][2]["what"] not in ctx["collection_note"]
        if first:
            assert "on the full recording clock" in request["prompt"]
            assert f"continuous {pieces.fmt_clock(3.0)} recording" in request["prompt"]
        else:
            assert "of it. The labelling pipeline" in ctx["collection_note"]
    saved = json.loads((ep / "context.json").read_text())
    assert {k: v for k, v in saved.items() if k != "pieces"} == raw
    assert episode.build_request(ep)["prompt"] == parent_request
