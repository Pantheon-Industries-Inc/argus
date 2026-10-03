"""Requests and displayed footage keep the same samples and clock."""
import json
import re
import subprocess
from pathlib import Path

import av
import numpy as np
import pytest

from board import clips, sensors, serve
from board.hands import probe_pts
from label import episode
from prepare import formats


def video(path, n, pattern=False):
    with av.open(str(path), "w") as dst:
        stream = dst.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        stream.options = {"bf": "0", "g": "1"}
        for k in range(n):
            image = np.full((64, 96, 3), k * 2, np.uint8)
            if pattern:
                tiles = np.random.default_rng(k).integers(0, 256, (4, 6, 3), dtype=np.uint8)
                image = np.repeat(np.repeat(tiles, 16, axis=0), 16, axis=1)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                dst.mux(packet)
        for packet in stream.encode():
            dst.mux(packet)


def recording(tmp_path, main, side=None):
    files, times = {}, {"exo": np.asarray(main)}
    if side is not None:
        times["left"] = np.asarray(side)
    for view, ts in times.items():
        path = tmp_path / (view + ".mp4")
        video(path, len(ts))
        files[view] = (view, path)
    ep = tmp_path / "episode_a"
    formats.video_views_episode(ep, files, "teleop_arms", "probe", {}, real=times,
                                signals={"force": np.arange(len(main))[:, None].astype(float)})
    with np.load(ep / "times.npz") as z:
        saved = {k: z[k] for k in z.files}
    np.savez(ep / "times.npz", **{**saved, **times})
    source = json.loads((ep / "sources.json").read_text())
    for view, ts in times.items():
        if view != "exo":
            source[view]["kmap"] = view + "_kmap.npy"
            np.save(ep / source[view]["kmap"], formats.nearest(ts, times["exo"]))
    (ep / "sources.json").write_text(json.dumps(source))
    return ep


@pytest.mark.parametrize("missing", [[0, 45], [0, 45, 89]])
def test_unavailable_planned_samples_keep_usable_footage_at_its_real_time(tmp_path, missing):
    ep = recording(tmp_path, np.arange(90) / 30)
    context = json.loads((ep / "context.json").read_text())
    context["placeholder_frames"] = {"exo": [[k, k] for k in missing]}
    (ep / "context.json").write_text(json.dumps(context))
    request = episode.build_request(ep)
    assert request["plan"]["ks"] and not set(missing) & set(request["plan"]["ks"])
    assert "No camera could be decoded at the planned instants" in request["prompt"]
    assert "0.00 s" in request["prompt"] and "1.50 s" in request["prompt"]
    details = [p["text"] for p in request["content"] if p.get("type") == "text" and "detail view" in p["text"]]
    assert "first available frame" in details[-2]
    assert "first frame of the episode" not in details[-2]


def test_a_paired_camera_keeps_its_existing_span_sentence():
    main, side = np.arange(90) / 30, 0.32 + np.arange(80) / 30
    ep = {"context": {"fps": 30}, "sources": {"exo": {}, "left": {}},
          "times": {"exo": main, "left": side}, "kmap": {"left": formats.nearest(side, main)}}
    assert episode._coverage_note(ep, {"ks": [0, 45, 89]}) == (
        " Left has frames only from 0.32 s to 2.95 s, so its cells are empty at the instants outside that time, "
        "and it is left out of a detail view there.")


def test_board_and_sensor_times_use_the_request_zero_when_the_first_capture_is_above_zero(tmp_path):
    main = 0.0122 + np.arange(90) / 30
    side = main - 0.008
    ep = recording(tmp_path, main, side)
    source = json.loads((ep / "sources.json").read_text())
    ctx = json.loads((ep / "context.json").read_text())
    assert np.array_equal(clips.clip_times(ep, source)["exo"], main)
    assert np.array_equal(sensors.clip_times(ep, ctx, 90), main)


def test_an_explicit_clock_zero_is_shared_by_request_and_board(tmp_path):
    main = 0.5 + np.arange(10) / 30
    ep = recording(tmp_path, main, main + 0.01)
    with np.load(ep / "times.npz") as z:
        ts = {k: z[k] for k in z.files}
    ts.update(exo=main, left=main + 0.01)
    np.savez(ep / "times.npz", **ts)
    ctx = json.loads((ep / "context.json").read_text())
    ctx["clock_zero_s"] = 0.5
    ctx["contacts"] = [{"id": "c1", "start_s": 0.6, "end_s": 0.7, "peak_s": 0.65, "dips_s": [0.66]}]
    ctx["annotation_subtasks"] = [{"t0": 0.55, "t1": 0.75}]
    (ep / "context.json").write_text(json.dumps(ctx))
    loaded = episode.load(ep)
    assert episode.frame_time(loaded, 0) == 0
    assert sensors.clip_times(ep, ctx, 10)[0] == 0
    assert loaded["context"]["contacts"][0]["start_s"] == pytest.approx(0.1)
    assert loaded["context"]["contacts"][0]["dips_s"] == pytest.approx([0.16])
    assert loaded["context"]["annotation_subtasks"][0]["t0"] == pytest.approx(0.05)
    assert json.loads((ep / "context.json").read_text())["contacts"][0]["start_s"] == 0.6


@pytest.mark.parametrize("offset", [0.5 / 30, 0.55 / 30, -0.008])
def test_encoded_boundaries_preserve_the_actual_request_map_at_float_ties(tmp_path, offset):
    main = 0.0122 + np.arange(90) / 30
    side = main + offset
    ep = recording(tmp_path, main, side)
    src = json.loads((ep / "sources.json").read_text())
    out = tmp_path / "clip.mp4"
    capture = clips.clip_times(ep, src)
    km = np.load(ep / src["left"]["kmap"])
    clips.extract_one(src["left"]["packed"], 0, 90, out, clips.find_ffmpeg(), 1,
                      main=False, times=capture["left"], query_s=capture["exo"], query_map=km)
    _, _, tb, pts = probe_pts(out)
    starts = np.asarray(pts) * float(tb)
    shown = np.searchsorted(starts, main, side="right") - 1
    assert np.array_equal(shown, km)
    midpoint = (side[45] + side[44]) / 2
    for t in (midpoint - 2e-6, midpoint + 2e-6):
        assert np.searchsorted(starts, t, side="right") - 1 == formats.nearest(side, np.array([t]))[0]


def test_composed_footage_uses_the_frame_already_on_screen_at_each_canvas_sample(tmp_path):
    path = tmp_path / "input.mp4"
    video(path, 90)
    capture = 0.019 + np.arange(90) / 30
    clips.retime(path, capture, main=False)
    out = tmp_path / "composed.mp4"
    subprocess.run(serve.footage_command([(path, (96, 64), 30)], 0.4, 2.0, out, 1), check=True)
    with av.open(str(path)) as src:
        original = [(float(f.time), f.to_ndarray(format="rgb24").mean()) for f in src.decode(video=0)]
    with av.open(str(out)) as src:
        for frame in src.decode(video=0):
            t = float(frame.time) + 0.4
            expected = [v for s, v in original if s <= t][-1]
            assert abs(frame.to_ndarray(format="rgb24").mean() - expected) < 1.2


def test_playback_panels_follow_the_capture_of_the_presented_frame(tmp_path):
    helpers = serve.INDEX_HTML.split("function snIndexAt", 1)[1].split("function snWhat", 1)[0]
    callbacks = re.findall(r"function onVF\(now, md\) \{[^\n]+\}", serve.INDEX_HTML)
    script = "function snIndexAt" + helpers + "\n"
    script += "const D={playback:{starts:[0,0.05,0.2],captures:[0,0.1,0.3]}};\n"
    script += "let rv=0, observed=-1; const alive=()=>true,watch=()=>{},sync=t=>observed=t;\n"
    for callback in callbacks[-2:]:
        script += callback + "\nonVF(0,{mediaTime:0.05}); if(observed!==0.1)throw Error(String(observed));\n"
        script += "onVF(0,{mediaTime:0.2}); if(observed!==0.3)throw Error(String(observed));\n"
    script += "if(snIndexAt([0,0.1,0.3],0.05)!==0)throw Error('paused seek moved');\n"
    path = tmp_path / "panels.js"
    path.write_text(script)
    result = subprocess.run(["node", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_timing_metadata_is_probed_once_per_clip_version(tmp_path, monkeypatch):
    path = tmp_path / "clip.mp4"
    video(path, 3)
    clips.retime(path, [0, 0.1, 0.3])
    calls, original = [], serve.subprocess.run
    def run(*args, **kw):
        calls.append(args[0])
        return original(*args, **kw)
    monkeypatch.setattr(serve.subprocess, "run", run)
    assert serve._shown_from_halfway(path)
    assert serve._shown_from_halfway(path)
    assert len(calls) == 1
    path.touch()
    assert serve._shown_from_halfway(path)
    assert len(calls) == 2


def test_goal_extraction_uses_absolute_clip_times_and_preserves_a_midpoint_tie(tmp_path):
    from PIL import Image
    import io
    path = tmp_path / "late.mp4"
    video(path, 30)
    capture = 0.3 + np.arange(30) / 30
    query = np.array([0.45])
    clips.retime(path, capture, main=False, query_s=query, query_map=np.array([4]))
    image = Image.open(io.BytesIO(serve.extract_frame(path, 0.45, 96)))
    assert abs(np.asarray(image).mean() - 8) < 1.5
    image = Image.open(io.BytesIO(serve.extract_frame(path, 0.450002, 96)))
    assert abs(np.asarray(image).mean() - 10) < 1.5


def test_removing_the_main_camera_moves_state_contacts_and_depth_on_one_clock(tmp_path):
    main = 0.0122 + np.arange(10) / 30
    side = main - 0.008
    ep = recording(tmp_path, main, side)
    src = json.loads((ep / "sources.json").read_text())
    ctx = json.loads((ep / "context.json").read_text())
    ctx["contacts"] = [{"start_s": float(side[3]), "end_s": float(side[5]), "peak_s": float(side[4]),
                        "dips_s": [float(side[4])]}]
    np.savez(ep / "state.npz", state=np.arange(10, dtype=float)[:, None])
    np.savez(ep / "depth_times.npz", depth_left=side, depth_left_pts=np.arange(10))
    np.save(ep / "depth_kmap.npy", np.arange(10))
    (ep / "depth.json").write_text(json.dumps({"left": {"kmap": "depth_kmap.npy"}}))
    with np.load(ep / "times.npz") as z:
        ts = {k: z[k] for k in z.files}
    src.pop("exo")
    clips.reanchor(ep, ctx, src, ts, "exo", "left", "top")
    (ep / "sources.json").write_text(json.dumps(src))
    (ep / "context.json").write_text(json.dumps(ctx))
    e = episode.load(ep)
    assert episode.frame_time(e, 0) == pytest.approx(0)
    with np.load(ep / "state.npz") as z:
        assert np.array_equal(z["state"][:, 0], np.arange(10))
    with np.load(ep / "depth_times.npz") as z:
        assert np.allclose(z["depth_left"], np.arange(10) / 30)
    assert ctx["contacts"][0]["start_s"] == pytest.approx(0.1)
    assert ctx["contacts"][0]["dips_s"] == pytest.approx([4 / 30], abs=0.001)
    assert np.allclose(clips.clip_times(ep, src)["left"], sensors.clip_times(ep, ctx, 10))


def test_composed_mixed_rate_footage_keeps_each_cameras_current_frame(tmp_path):
    inputs, original = [], []
    for name, rate in (("slow", 15), ("fast", 60)):
        path = tmp_path / (name + ".mp4")
        video(path, 90, pattern=True)
        clips.retime(path, 0.019 + np.arange(90) / rate, fps=rate, main=False)
        inputs.append((path, (96, 64), rate))
        with av.open(str(path)) as src:
            original.append([(float(f.time), f.to_ndarray(format="rgb24")) for f in src.decode(video=0)])
    out = tmp_path / "mixed.mp4"
    subprocess.run(serve.footage_command(inputs, 0.4, 1.4, out, 1), check=True)
    _, _, cells = serve.footage_layout([(96, 64), (96, 64)])
    with av.open(str(out)) as src:
        frames = list(src.decode(video=0))
    assert len(frames) == 60
    for frame in frames:
        t, pixels = float(frame.time) + 0.4, frame.to_ndarray(format="rgb24")
        for (x, y, w, h), samples in zip(cells, original):
            expected = [v for start, v in samples if start <= t][-1]
            assert np.abs(pixels[y:y + h, x:x + w].astype(float) - expected).mean() < 10
