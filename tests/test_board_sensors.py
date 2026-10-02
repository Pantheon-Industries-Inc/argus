"""The board's sensors files (board/sensors.py): what a recording's other signals and depth streams become on the
episode page, through board build, the server and the depth clips (board/clips.py)."""
from __future__ import annotations

import json
import shutil
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from board import build as board_build
from board import clips, sensors
from label import signals as S


# ---------------------------------------------------------------- quantization

def test_values_round_trip_within_half_a_step():
    rng = np.random.default_rng(0)
    a = np.c_[rng.normal(0, 1, 500), rng.normal(1000, 50, 500), np.full(500, 3.0)]
    a[17, 1] = np.nan
    blk = sensors.quantize(a)
    back = sensors.dequantize(blk, len(a))
    assert np.isnan(back[17, 1]) and np.isfinite(back[16, 1])
    for c in range(3):
        ok = np.isfinite(a[:, c])
        assert np.max(np.abs(back[ok, c] - a[ok, c])) <= blk["step"][c] / 2 + 1e-9
    assert np.all(back[:, 2] == 3.0)          # a value that never changes is stored exactly


def test_a_map_round_trips_in_one_byte_a_value():
    rng = np.random.default_rng(1)
    a = 3072 - np.clip(rng.normal(0, 400, (40, 256)), 0, None)
    a[3] = np.nan                              # a frame with no reading
    blk = sensors.quantize(a, bits=8, per_value=False)
    back = sensors.dequantize(blk, len(a), bits=8)
    assert np.isnan(back[3]).all()
    ok = np.isfinite(a)
    assert np.max(np.abs(back[ok] - a[ok])) <= blk["step"] / 2 + 1e-9


def test_times_round_trip_to_the_millisecond():
    t = np.cumsum(np.r_[0.0, np.full(60, 1 / 30), [0.1], np.full(20, 1 / 30)])
    back = sensors.decode_times(sensors.encode_times(t))
    assert len(back) == len(t) and np.max(np.abs(back - t)) <= 0.0005 + 1e-9


# ---------------------------------------------------------------- one episode

def _episode(root: Path, n: int = 300, fps: float = 30.0, depth: bool = False) -> Path:
    """A prepared episode with a force that rests at 0 and is pressed twice, a 4 x 4 pressure map that reads about
    3072 at rest and falls where it is pressed, a constant health flag, and the anchor camera's capture times starting
    at 12.5 s (a real clock, not the clip's)."""
    ep = root / "episode_000000"
    ep.mkdir(parents=True)
    t = 12.5 + np.arange(n) / fps
    force = np.zeros((n, 1), np.float32)
    force[60:90] = 5.0
    force[200:240] = 7.0
    # a real sensor's noise on every cell: a map that only ever takes a few readings is a setting, not touch
    rng = np.random.default_rng(0)
    pmap = (3072.0 + rng.normal(0, 2, (n, 16))).astype(np.float32)
    pmap[60:90, 5] += 2000.0 - 3072.0
    pmap[200:240, 10] += 1500.0 - 3072.0
    pmap[200:240, 11] += 2600.0 - 3072.0
    np.savez(ep / "signals.npz", s0=force, s1=pmap, s2=np.ones((n, 1), np.float32))
    np.savez(ep / "times.npz", exo=t, exo_pts=np.arange(n) * 512)
    ctx = {"dataset": "you/your-own-dataset", "profile": "teleop_arms", "fps": fps, "n_state_frames": n,
           "signals": [{"name": "fingertip force", "key": "s0", "dims": 1, "names": ["fz"], "rate_hz": 100.0},
                       {"name": "pressure", "key": "s1", "dims": 16, "shape": [4, 4]},
                       {"name": "health", "key": "s2", "dims": 1, "names": ["ok"]}]}
    sources = {"exo": {"packed": str(ep / "exo.mp4"), "base_s": 0.0, "n_frames": n}}
    if depth:
        np.save(ep / "depth_kmap_exo.npy", np.arange(n))
        (ep / "depth.json").write_text(json.dumps({"exo": {"packed": str(ep / "depth.mkv"), "base_s": 0.0,
                                                            "n_frames": n, "kmap": "depth_kmap_exo.npy",
                                                            "scale_m": 0.001, "width": 8, "height": 6}}))
        ctx["depth"] = {"exo": {"units": "metres", "scale_m": 0.001, "source": "depth.mkv"}}
    (ep / "context.json").write_text(json.dumps(ctx))
    (ep / "sources.json").write_text(json.dumps(sources))
    return ep


def test_an_episode_file_keeps_what_label_signals_reads(tmp_path):
    ep = _episode(tmp_path)
    doc = sensors.episode_doc(ep)
    assert doc["format"] == sensors.FORMAT and doc["frames"] == 300
    # 30 frames a second sampled every second frame: at most 15 a second, on the clip's clock (first frame 0)
    assert doc["stride"] == 2 and doc["n"] == 150
    t = sensors.decode_times(doc["times"])
    assert t[0] == 0.0 and t[-1] == pytest.approx(299 / 30, abs=1e-3)
    force, pmap, health = doc["signals"]
    assert health["constant"] is True and health["value"] == [1.0] and "values" not in health
    a = np.load(ep / "signals.npz")["s0"].astype(np.float64)
    clip_t = 12.5 + np.arange(300) / 30 - 12.5
    assert force["rests_and_rises"] is bool(S.rests_and_rises(a)) is True and force["direction"] == "up"
    # the spans are label/signals.py's, from every frame, on the clip's clock
    assert force["spans"] == [[round(s, 3), round(e, 3)] for s, e in S.active_spans(a, clip_t)]
    assert force["spans"] == [[2.0, 2.967], [6.667, 7.967]]
    v = sensors.dequantize(force["values"], doc["n"])
    assert np.max(np.abs(v[:, 0] - a[::2, 0])) <= force["values"]["step"][0] / 2 + 1e-9
    # the pressure map behaves like touch: falls when pressed, drawn as a heatmap against its rest and swing
    assert pmap["touch"] is True and pmap["direction"] == "down" and pmap["shape"] == [4, 4]
    assert np.allclose(pmap["rest"], 3072.0, atol=10) and pmap["swing_from"] == "episode" and pmap["swing"] > 0
    m = sensors.dequantize(pmap["map"], doc["n"], bits=8)
    raw = np.load(ep / "signals.npz")["s1"].astype(np.float64)
    assert m.shape == (150, 16) and abs(m[110, 10] - raw[220, 10]) <= pmap["map"]["step"] / 2 + 1e-6
    act = sensors.dequantize(pmap["activity"], doc["n"])[:, 0]
    assert act[10] == 0 and act[110] > act[40] > 0


def test_the_uploads_rest_and_swing_are_used_when_prepare_measured_them(tmp_path):
    ep = _episode(tmp_path)
    ctx = json.loads((ep / "context.json").read_text())
    ctx["signals"][1].update({"rest": [3100.0] * 16, "swing": 2000.0})
    (ep / "context.json").write_text(json.dumps(ctx))
    pmap = sensors.episode_doc(ep)["signals"][1]
    assert pmap["swing"] == 2000.0 and pmap["swing_from"] == "upload" and pmap["rest"] == [3100.0] * 16


def test_an_episode_without_signals_or_depth_has_no_file(tmp_path):
    ep = tmp_path / "episode_000000"
    ep.mkdir()
    (ep / "context.json").write_text(json.dumps({"profile": "ego_head", "fps": 30}))
    assert sensors.episode_doc(ep) is None


def test_depth_cameras_carry_their_units_and_colour_bar(tmp_path):
    doc = sensors.episode_doc(_episode(tmp_path, depth=True))
    d = doc["depth"]["exo"]
    assert d["units"] == "metres" and d["scale_m"] == 0.001
    assert len(d["bar"]) == sensors.BAR_STOPS and all(c.startswith("#") for c in d["bar"])
    labels = [label for _, label in d["ticks"]]
    assert labels[0] == "0.1 m" and labels[-1] == "10 m"              # near first, on the fixed metric scale
    assert [p for p, _ in d["ticks"]] == sorted(p for p, _ in d["ticks"])


# ---------------------------------------------------------------- board build

def _run(runs: Path, ep: Path) -> Path:
    run = runs / "r1"
    (run / "out").mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"run_id": "r1", "code": "abc1234", "kind": "full", "status": "done",
                                              "slice": str(ep.parent)}))
    (run / "out" / f"{ep.name}.json").write_text(json.dumps({
        "episode_dir": str(ep), "parse_ok": True, "model": "some/model",
        "labels": {"task_summary": "lift the cup", "timeline": [{"start_s": 0.0, "end_s": 5.0, "action": "lift"}],
                   "completion": {"task_completed": "success", "completed_at_s": 5.0}}}))
    return run


def _board(tmp: Path, ep: Path, **manifest) -> Path:
    board = tmp / "board"
    board.mkdir()
    (board / "manifest.json").write_text(json.dumps({"board": "b", "datasets": [
        {"dataset": "mine", "run": str(_run(tmp / "runs", ep)), "episodes": str(ep.parent)}], **manifest}))
    return board


def test_board_build_writes_sensors_only_for_episodes_that_have_them(tmp_path):
    ep = _episode(tmp_path / "eps")
    board = _board(tmp_path, ep)
    built = board_build.build(board)
    assert built["sensors"]["written"] == 1 and not built["sensors"]["skipped"]
    idx = json.loads((board / "sensors" / "index.json").read_text())
    assert idx["files"] == {"episode_000000.json": {"signals": 2, "constant": 1, "depth": []}}
    assert json.loads((board / "sensors" / "episode_000000.json").read_text())["format"] == sensors.FORMAT
    qa = json.loads((board / "qa" / "episode_000000.json").read_text())
    assert not any("sensor" in k for k in qa if k != "dataset_checks")     # nothing new in the label itself
    # turned off in the manifest: the folder goes with the next build
    m = json.loads((board / "manifest.json").read_text())
    (board / "manifest.json").write_text(json.dumps({**m, "sensors": False}))
    assert "sensors" not in board_build.build(board) and not (board / "sensors").exists()


def test_a_board_without_signals_builds_as_before(tmp_path):
    ep = tmp_path / "eps" / "episode_000000"
    ep.mkdir(parents=True)
    (ep / "context.json").write_text(json.dumps({"dataset": "you/your-own-dataset", "profile": "teleop_arms",
                                                 "fps": 30, "n_state_frames": 150}))
    board = _board(tmp_path, ep)
    built = board_build.build(board)
    first = {p.name: p.read_bytes() for p in (board / "qa").iterdir()}
    assert "sensors" not in built and not (board / "sensors").exists() and not (board / "sensors.new").exists()
    assert set(json.loads((board / "BUILT.json").read_text())) == {"manifest", "counts"}
    # and a second build writes the same bytes
    board_build.build(board)
    assert {p.name: p.read_bytes() for p in (board / "qa").iterdir()} == first


# ---------------------------------------------------------------- depth clips

def test_each_colour_frame_shows_the_nearest_depth_frame():
    ct = np.arange(10) / 30
    dt = np.r_[np.arange(5) / 30, 7 / 30 + np.arange(3) / 30] + 0.004     # a gap at frames 5 and 6
    assert clips.depth_frame_map(ct, dt) == [0, 1, 2, 3, 4, None, None, 5, 6, 7]


def _video(path: Path, n: int, codec: str, pix: str, frame, rate: int = 30) -> None:
    import av
    with av.open(str(path), "w") as c:
        s = c.add_stream(codec, rate=rate)
        s.width, s.height, s.pix_fmt = 32, 24, pix
        s.time_base = Fraction(1, 15360)
        for k in range(n):
            fr = frame(k)
            fr.pts, fr.time_base = k * 512, s.time_base
            for pkt in s.encode(fr):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="no ffmpeg")
@pytest.mark.parametrize("depth_times", ["depth_times.npz", "times.npz"])
def test_a_depth_clip_has_its_colour_clips_frames_and_timestamps(tmp_path, depth_times):
    """The depth stream's frame times are read from depth_times.npz (prepare writes them there), or from times.npz in an
    episode prepared before that file existed."""
    import av
    from board.hands import probe_pts
    n = 40
    ep = _episode(tmp_path / "eps", n=n, depth=True)
    _video(ep / "exo.mp4", n, "mpeg4", "yuv420p",
           lambda k: av.VideoFrame.from_ndarray(np.full((24, 32, 3), 40 + k, np.uint8), format="rgb24"))
    # metric depth: near (300 mm) on the left half, far (3 m) on the right, no reading in the top rows
    def depth(k):
        a = np.full((24, 32), 3000, np.uint16)
        a[:, :16] = 300
        a[:6] = 0
        return av.VideoFrame.from_ndarray(a, format="gray16le")
    _video(ep / "depth.mkv", n, "ffv1", "gray16le", depth)
    t = 12.5 + np.arange(n) / 30
    if depth_times == "times.npz":
        np.savez(ep / "times.npz", exo=t, exo_pts=np.arange(n) * 512, depth_exo=t, depth_exo_pts=np.arange(n) * 512)
    else:
        np.savez(ep / "times.npz", exo=t, exo_pts=np.arange(n) * 512)
        np.savez(ep / "depth_times.npz", depth_exo=t, depth_exo_pts=np.arange(n) * 512)
    out = tmp_path / "clips"
    jobs = clips.episode_jobs(ep, out, False)
    for (pk, b, du, o, fps, main, off, skip, _, _) in jobs:
        clips.extract_one(pk, b, du, o, clips.find_ffmpeg(), 1, fps, main, off, skip)
    (job,) = clips.depth_jobs(ep, out, False)
    clips.extract_depth(*job[:4], 1, job[4])
    colour, dclip = out / "episode_000000.mp4", out / "depth_exo" / "episode_000000.mp4"
    assert probe_pts(dclip)[3] == probe_pts(colour)[3]
    assert clips.depth_jobs(ep, out, False) == []                  # idempotent: a matching clip is not cut again
    with av.open(str(dclip)) as c:
        fr = next(c.decode(video=0)).to_ndarray(format="rgb24").astype(int)
    near, far, none = fr[16, 4], fr[16, 28], fr[1, 8]
    assert near[0] > near[2] and far[2] > far[0]                  # near warm, far cool on the turbo scale
    assert none.max() < 40                                        # no reading is black


# ---------------------------------------------------------------- the server

def test_the_server_hands_out_sensors_and_depth_clips(tmp_path, monkeypatch):
    import socketserver
    import threading
    import urllib.error
    import urllib.request
    from board import serve
    ep = _episode(tmp_path / "eps")
    board = _board(tmp_path, ep)
    board_build.build(board)
    clips_dir = tmp_path / "clips"
    (clips_dir / "depth_exo").mkdir(parents=True)
    (clips_dir / "depth_exo" / "episode_000000.mp4").write_bytes(b"depth")
    for k, v in (("HERE", board / "qa"), ("MP4_DIR", clips_dir), ("COMPARE_DIR", board / "compare"),
                 ("HANDS_DIR", board / "hands"), ("KEYPOINTS_DIR", board / "hand_keypoints"),
                 ("SENSORS_DIR", board / "sensors")):
        monkeypatch.setattr(serve, k, v.resolve())
    monkeypatch.setattr(serve, "_LIST_CACHE", {})
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), serve.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"

    def get(path):
        try:
            with urllib.request.urlopen(url + path) as r:
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read()
    try:
        page = get("/")[2].decode()
        assert '"sensors":true' in page
        assert json.loads(get("/api/sensors?file=index.json")[2])["files"]["episode_000000.json"]["signals"] == 2
        assert json.loads(get("/api/sensors?file=episode_000000.json")[2])["format"] == sensors.FORMAT
        for bad in ("../qa/episode_000000.json", "", "nope.json"):
            assert get("/api/sensors?file=" + bad)[0] == 404
        code, headers, body = get("/api/video?id=episode_000000&cam=depth_exo&download=1")
        assert code == 200 and body == b"depth" and "episode_000000_depth_exo.mp4" in headers["Content-Disposition"]
        assert get("/api/video?id=episode_000000&cam=depth_../x")[0] == 404
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_the_static_build_publishes_sensors_and_each_depth_clip(tmp_path):
    import argparse
    from board import static
    ep = _episode(tmp_path / "eps", depth=True)
    board = _board(tmp_path, ep)
    board_build.build(board)
    clips_dir = tmp_path / "clips"
    (clips_dir / "depth_exo").mkdir(parents=True)
    (clips_dir / "episode_000000.mp4").write_bytes(b"colour")
    (clips_dir / "depth_exo" / "episode_000000.mp4").write_bytes(b"depth")
    (e,) = static.plan(board / "qa", clips_dir)
    assert {k for k in e["media"] if k.startswith("depth_")} == {"depth_exo"} and e["media"]["depth_exo"]["rel"].startswith("v/depth_exo/")
    a = argparse.Namespace(board=board, qa=board / "qa", clips=clips_dir, compare=None, hands=None, keypoints=None,
                           out=tmp_path / "out", build_id="b1", force=False, public_base=None, title="Data Dashboard")
    assert static.cmd_site(a) == 0
    site = tmp_path / "out" / "b1"
    assert '"sensors":true' in (site / "index.html").read_text()
    assert (site / "data" / "sensors" / "episode_000000.json").read_bytes() == \
        (board / "sensors" / "episode_000000.json").read_bytes()
    assert list(json.loads((site / "data" / "sensors" / "index.json").read_text())["files"]) == ["episode_000000.json"]
    (rec,) = json.loads((site / "data" / "lists" / "mine.json").read_text())
    assert rec["_media"]["depth_exo"].startswith("v/depth_exo/episode_000000.")
