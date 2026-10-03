"""An episode keeps every camera that works: what board clips and labelling do when a camera is short, damaged or
does not decode at all, and what an episode whose main camera is taken out is timed on afterwards."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from board import clips
from label import episode as me
from prepare import formats

REPO = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="no ffmpeg")


def _video(path: Path, frames: int, size: str = "160x120") -> None:
    subprocess.run([clips.find_ffmpeg(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"testsrc=size={size}:rate=30", "-frames:v", str(frames), "-pix_fmt", "yuv420p", str(path)],
                   check=True)


def _damaged(path: Path, frames: int, damage: range) -> None:
    """A camera whose packets in damage have an impossible first NAL length: its index reads whole, and the frames
    there (to the next keyframe) do not decode."""
    import av
    with av.open(str(path), "w") as c:
        s = c.add_stream("libx264", rate=30)
        s.width, s.height, s.pix_fmt = 160, 120, "yuv420p"
        s.options = {"g": "10", "bf": "0"}
        rng = np.random.default_rng(0)
        for _ in range(frames):
            im = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
            for pk in s.encode(av.VideoFrame.from_ndarray(im, format="rgb24")):
                c.mux(pk)
        for pk in s.encode():
            c.mux(pk)
    with av.open(str(path)) as c:
        pos = [pk.pos for pk in c.demux(c.streams.video[0]) if pk.pos is not None and pk.size]
    b = bytearray(path.read_bytes())
    for i in damage:
        b[pos[i]:pos[i] + 4] = b"\x7f\xff\xff\xff"
    path.write_bytes(bytes(b))


def _clips(eps: Path, out: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "board", "clips", "--episodes", str(eps), "--out", str(out), *extra],
                          cwd=REPO, capture_output=True, text=True)


def _upload(tmp_path: Path, cams: dict) -> tuple[dict, Path, Path]:
    """A video upload of one episode, {file stem: frames}, read as an upload is."""
    up = tmp_path / "up"
    up.mkdir()
    for stem, n in cams.items():
        _video(up / f"{stem}.mp4", n)
    eps = tmp_path / "episodes"
    rep = formats.convert(up, "teleop_arms", eps, "mine", float("inf"), grouping={})
    return rep, eps, eps / rep["episodes"][0]["episode_id"]


T_EXO = np.arange(60) / 30.0
T_LEFT = 0.5 + np.arange(60) / 30.0
T_RIGHT = np.arange(75) / 30.0


def _recording(tmp_path: Path, size: str = "160x120", left_start: float = 0.5) -> tuple[Path, Path]:
    """One recording on one recorder clock: a top camera, a left wrist camera that starts left_start s after it with
    as many frames, and a right wrist camera with more; two arms' joints (14 values) recorded on the top camera's
    frames, row k holding k, and a force signal on the same frames."""
    up = tmp_path / "up"
    up.mkdir()
    t_left = left_start + np.arange(60) / 30.0
    for stem, t in (("top", T_EXO), ("wrist_left", t_left), ("wrist_right", T_RIGHT)):
        _video(up / f"{stem}.mp4", len(t), size)
    state = np.tile(np.arange(60, dtype=np.float32)[:, None], (1, 14))
    eps = tmp_path / "episodes"
    eps.mkdir()
    ep = eps / "episode_a"
    files = {"exo": ("top", up / "top.mp4"), "left": ("wrist_left", up / "wrist_left.mp4"),
             "right": ("wrist_right", up / "wrist_right.mp4")}
    formats.video_views_episode(ep, files, "teleop_arms", "probe", {}, real={"exo": T_EXO, "left": t_left,
                                                                             "right": T_RIGHT},
                                state=state, action=state.copy(), signals={"force": np.arange(60.0)[:, None]})
    return eps, ep


def _clip_timing(p: Path) -> tuple[int, float]:
    import av
    with av.open(str(p)) as c:
        s = c.streams.video[0]
        first = min(pk.pts for pk in c.demux(s) if pk.pts is not None and pk.size)
        return s.codec_context.width, float(first * s.time_base)


def test_a_main_camera_taken_out_moves_the_state_and_signals_onto_the_new_main_camera(tmp_path):
    """The top camera does not decode. The left wrist camera, 0.5 s later on the same clock, is the main camera now,
    so every array on the top camera's frames (state, action, signals) is moved onto its frames by capture time: row
    k is the top camera's frame nearest the left camera's frame k, and no row where no top frame is within half a
    frame. The right camera is paired to the new main camera, the context's length and rate are the new main
    camera's, and the episode stays on its own clock, so a rerun with --force cuts the same clips."""
    from checks import capture_qc
    eps, ep = _recording(tmp_path, size="1600x900")
    ctx = json.loads((ep / "context.json").read_text())
    ctx["capture_qc"] = capture_qc.run_episode(ep)
    (ep / "context.json").write_text(json.dumps(ctx))
    assert "exo" in ctx["capture_qc"]["metrics"]["cameras"]
    (eps.parent / "up" / "top.mp4").write_bytes(b"not a video at all" * 50)
    out = tmp_path / "clips"
    p = _clips(eps, out)
    assert p.returncode == 0, p.stdout + p.stderr
    src = json.loads((ep / "sources.json").read_text())
    ctx = json.loads((ep / "context.json").read_text())
    assert list(src) == ["left", "right"] and "kmap" not in src["left"]
    assert np.array_equal(np.load(ep / src["right"]["kmap"]), formats.nearest(T_RIGHT, T_LEFT))
    st = np.load(ep / "state.npz")
    assert st["state"].shape == (60, 14) and st["action"].shape == (60, 14)
    assert np.array_equal(st["state"][:45, 0], np.arange(15, 60)) and np.isnan(st["state"][45:]).all()
    assert np.array_equal(st["action"][:45, 3], np.arange(15, 60))
    sig = np.load(ep / "signals.npz")["s0"]
    assert np.array_equal(sig[:45, 0], np.arange(15, 60)) and np.isnan(sig[45:]).all()
    assert ctx["n_state_frames"] == 60 and ctx["fps"] == 30.0 and ctx["duration_s"] == 2.5
    assert "exo" not in ctx["capture_qc"]["metrics"]["cameras"]          # the checks were run again without it
    # the state covers the left camera's frames 0 to 44 only: limited to that stretch, said once, and never counted as
    # missing values by the capture checks
    assert ctx["state_span"] == [0, 45]
    partial = [x for x in ctx["reader_issues"] if x["kind"] == "state_partial"]
    assert len(partial) == 1 and partial[0]["what"] == (
        "The robot state was recorded with the main camera, which could not be decoded. The cameras left overlap it "
        "only from 0.50 s to 1.97 s, so the state is used for that part of the episode only.")
    assert not [f for f in ctx["capture_qc"]["flags"] if f["check"] == "nonfinite_signal"]
    e = me.load(ep)
    pl = me.plan(e)
    assert me.anchor(e) == "left" and pl["state_usable"] and pl["state_span"] == (0, 45) and 44 in pl["ks"]
    assert me.frame_time(e, 0) == pytest.approx(0.5) and e["state"][0][0] == 15
    req = me.build_request(ep)
    assert req["views"] == ["left", "right"]
    assert "nan" not in req["prompt"].lower().replace("nanosecond", "")
    assert "The recorded state covers only 0.50 s to 1.97 s of the episode" in req["prompt"]
    # the left clip, cut as a side camera on the first run, is cut again as the main camera, still 0.5 s in
    left = out / "wrist_left" / f"{ep.name}.mp4"
    assert _clip_timing(left) == (1600, pytest.approx(0.5, abs=0.02))
    right = _clip_timing(out / "wrist_right" / f"{ep.name}.mp4")
    assert _clips(eps, out, "--force").returncode == 0
    assert _clip_timing(left) == (1600, pytest.approx(0.5, abs=0.02))
    assert _clip_timing(out / "wrist_right" / f"{ep.name}.mp4") == right
    assert clips.start_offsets(ep, src) == {"left": (pytest.approx(0.5), 0)}


def test_placeholder_frames_move_onto_the_new_main_cameras_frames(tmp_path):
    """The placeholder frames the reader recorded are on the main camera's frames. When the main camera goes, its own
    entry goes with it and every other camera's moves onto the new main camera's frames by capture time, so labelling
    still leaves out exactly the frames that are placeholders."""
    eps, ep = _recording(tmp_path)
    files = {"exo": ("top", eps.parent / "up" / "top.mp4"), "left": ("wrist_left", eps.parent / "up" / "wrist_left.mp4"),
             "right": ("wrist_right", eps.parent / "up" / "wrist_right.mp4")}
    shutil.rmtree(ep)
    formats.video_views_episode(ep, files, "teleop_arms", "probe", {}, real={"exo": T_EXO, "left": T_LEFT,
                                                                             "right": T_RIGHT},
                                placeholders={"exo": [3], "left": [10], "right": [20]})
    ctx = json.loads((ep / "context.json").read_text())
    assert ctx["placeholder_frames"] == {"exo": [[3, 3]], "left": [[25, 25]], "right": [[20, 20]]}
    (eps.parent / "up" / "top.mp4").write_bytes(b"not a video at all" * 50)
    assert _clips(eps, tmp_path / "clips").returncode == 0
    ctx = json.loads((ep / "context.json").read_text())
    assert ctx["placeholder_frames"] == {"left": [[10, 10]], "right": [[5, 5]]}, ctx["placeholder_frames"]


def test_a_main_camera_taken_out_with_no_capture_times_leaves_the_state_unaligned(tmp_path):
    """The new main camera was paired to the old one by time (its kmap), and the capture times are gone: nothing can
    place the state on its frames, so the state and signals are not used and the episode says why. The checks that
    compare the state with the video are not assessed; those on the state alone still run."""
    from checks import capture_qc
    from checks import stream_pairing as sp
    eps, ep = _recording(tmp_path)
    assert json.loads((ep / "sources.json").read_text())["left"].get("kmap")
    ctx = json.loads((ep / "context.json").read_text())
    ctx.pop("real_times", None)
    (ep / "context.json").write_text(json.dumps(ctx))
    (ep / "times.npz").unlink()
    (eps.parent / "up" / "top.mp4").write_bytes(b"not a video at all" * 50)
    assert _clips(eps, tmp_path / "clips").returncode == 0
    ctx = json.loads((ep / "context.json").read_text())
    assert ctx["state_unaligned"]
    kinds = [x["kind"] for x in ctx["reader_issues"]]
    # measured again on the left camera, now main, by frame index: the right one goes on 0.5 s past it
    assert kinds == ["main_camera_short", "camera_not_decodable", "state_unaligned"]
    assert "main camera" in ctx["reader_issues"][2]["what"]
    e = me.load(ep)
    assert not me.plan(e)["state_usable"] and not e["signals"]
    req = me.build_request(ep)
    assert "state" not in req["blocks"] and "signals" not in req["blocks"]
    assert sp.pairing(ep)["not_assessed"] and "crossed" not in sp.pairing(ep)
    assert sp.jumps(ep)["not_assessed"] and "flagged" not in sp.jumps(ep)
    assert sp.grippers(ep)["actors"]                         # the state alone is still checked
    rows = {r["check"]: r for r in capture_qc.run_episode(ep)["checks"]}
    for c in ("camera_state_alignment_mismatch", "video_frozen_run"):
        assert rows[c]["status"] == "not_applicable" and "not on these cameras' frames" in rows[c]["why"], rows[c]
        assert ":" not in rows[c]["why"]                     # a reason is whole sentences, read on the job page
    assert ":" not in sp.pairing(ep)["not_assessed"]
    assert rows["invalid_state_shape"]["status"] == "clear"


def test_a_main_camera_taken_out_of_a_shared_frame_index_keeps_the_state(tmp_path):
    """LeRobot v3 style: every camera on one frame index (no kmap, no capture times, as many frames as the state).
    The state is on the new main camera's frames too, so it is kept as it is and used."""
    up = tmp_path / "up"
    up.mkdir()
    for stem in ("top", "wrist_left", "wrist_right"):
        _video(up / f"{stem}.mp4", 60)
    state = np.tile(np.arange(60, dtype=np.float32)[:, None], (1, 14))
    eps = tmp_path / "episodes"
    eps.mkdir()
    ep = eps / "episode_a"
    files = {"exo": ("top", up / "top.mp4"), "left": ("wrist_left", up / "wrist_left.mp4"),
             "right": ("wrist_right", up / "wrist_right.mp4")}
    formats.video_views_episode(ep, files, "teleop_arms", "probe", {}, state=state, action=state.copy())
    src = json.loads((ep / "sources.json").read_text())
    assert not any(s.get("kmap") for s in src.values())
    ctx = json.loads((ep / "context.json").read_text())
    ctx.pop("real_times", None)
    (ep / "context.json").write_text(json.dumps(ctx))
    (ep / "times.npz").unlink()
    (up / "top.mp4").write_bytes(b"not a video at all" * 50)
    assert _clips(eps, tmp_path / "clips").returncode == 0
    ctx = json.loads((ep / "context.json").read_text())
    assert "state_unaligned" not in ctx and [x["kind"] for x in ctx["reader_issues"]] == ["camera_not_decodable"]
    assert np.array_equal(np.load(ep / "state.npz")["state"], state)
    e = me.load(ep)
    assert me.anchor(e) == "left" and me.plan(e)["state_usable"]
    assert "state" in me.build_request(ep)["blocks"]


def test_the_cameras_spans_are_measured_again_on_the_new_main_camera(tmp_path):
    """A main camera whose index is whole and none of whose frames decode, beside a wrist camera half as long: the
    reader flagged the wrist camera as ending before the main one, and that issue had stayed after board clips took
    the main camera out, though the wrist camera is now the episode. The span issues are measured again on the cameras
    left: alone, the wrist camera covers its whole episode."""
    up = tmp_path / "up"
    up.mkdir()
    _video(up / "top.mp4", 60)
    _video(up / "wrist_left.mp4", 30)
    b = bytearray((up / "top.mp4").read_bytes())
    i = b.find(b"mdat")
    n = int.from_bytes(b[i - 4:i], "big") - 8
    b[i + 4:i + 4 + n] = np.random.default_rng(0).bytes(n)
    (up / "top.mp4").write_bytes(bytes(b))
    eps = tmp_path / "episodes"
    rep = formats.convert(up, "teleop_arms", eps, "mine", float("inf"), grouping={})
    ep = eps / rep["episodes"][0]["episode_id"]
    assert [x["kind"] for x in json.loads((ep / "context.json").read_text())["reader_issues"]] == ["camera_short"]
    assert _clips(eps, tmp_path / "clips").returncode == 0
    ctx = json.loads((ep / "context.json").read_text())
    assert [x["kind"] for x in ctx["reader_issues"]] == ["camera_not_decodable"], ctx["reader_issues"]


def test_a_camera_that_ends_early_is_one_issue_whichever_step_finds_it(tmp_path):
    """The reader flags a wrist camera that ends at 1 s of a 2 s episode, and board clips then finds its file holds
    fewer frames still: both had been recorded, under two kinds, for one camera that shows nothing past a time. It is
    one camera_short issue, at the end the clip has."""
    rep, eps, ep = _upload(tmp_path, {"top": 60, "wrist_left": 30})
    assert [x["kind"] for x in json.loads((ep / "context.json").read_text())["reader_issues"]] == ["camera_short"]
    _video(Path(json.loads((ep / "sources.json").read_text())["left"]["packed"]), 20)
    assert _clips(eps, tmp_path / "clips").returncode == 0
    (short,) = json.loads((ep / "context.json").read_text())["reader_issues"]
    assert short["kind"] == "camera_short" and short["camera"] == "wrist_left" and short["clip_frames"] == 20
    assert short["t0_s"] == pytest.approx(20 / 30, abs=0.01) and short["t1_s"] == 2.0, short


def test_the_board_shows_the_frame_the_model_is_sent_at_every_instant(tmp_path):
    """A wrist camera whose recorder dropped three frames at 1 s: its video plays at an even 30 fps, so its clip had
    shown frame 45 at 1.5 s while its capture times put frame 42 there, the frame the model is sent. Each clip frame
    plays at its capture time, so at every sampled instant the board's frame is the model's, on every camera."""
    from board.hands import probe_pts
    up = tmp_path / "up"
    up.mkdir()
    t_left = np.where(np.arange(57) < 30, np.arange(57), np.arange(57) + 3) / 30.0
    for stem, n in (("top", 60), ("wrist_left", 57)):
        _video(up / f"{stem}.mp4", n)
    eps = tmp_path / "episodes"
    ep = eps / "episode_a"
    files = {"exo": ("top", up / "top.mp4"), "left": ("wrist_left", up / "wrist_left.mp4")}
    formats.video_views_episode(ep, files, "teleop_arms", "probe", {}, real={"exo": T_EXO, "left": t_left})
    out = tmp_path / "clips"
    assert _clips(eps, out).returncode == 0
    e = me.load(ep)
    pl = me.plan(e)
    for v, clip in (("exo", out / f"{ep.name}.mp4"), ("left", out / "wrist_left" / f"{ep.name}.mp4")):
        _, _, tb, pts = probe_pts(clip)
        shown = np.asarray(pts, dtype=np.float64) * float(tb)
        assert np.allclose(shown, e["times"][v], atol=1e-3)
        for k in pl["ks"]:
            own = int(e["kmap"][v][k]) if v in e["kmap"] else k
            assert int(np.searchsorted(shown, me.frame_time(e, k) + 1e-4, side="right") - 1) == own, (v, k)
    assert 42 == int(e["kmap"]["left"][45])


def test_a_new_main_camera_that_started_earlier_moves_the_clock_to_its_first_frame(tmp_path):
    """The left wrist camera started 0.5 s before the top camera, which does not decode. The episode's clock now
    starts at the left camera's first frame, the earliest of the cameras left: frame times, clips and the length
    agree, no instant before 0 is sampled, and the state covers the stretch the top camera filmed. A reader issue
    already recorded on the old clock moves with it."""
    eps, ep = _recording(tmp_path, left_start=-0.5)
    ctx = json.loads((ep / "context.json").read_text())
    ctx["reader_issues"] = [{"kind": "signal_gap", "signal": "force", "what": "The force signal stops.",
                             "t0_s": 1.0, "t1_s": 1.2}]
    (ep / "context.json").write_text(json.dumps(ctx))
    (eps.parent / "up" / "top.mp4").write_bytes(b"not a video at all" * 50)
    out = tmp_path / "clips"
    assert _clips(eps, out).returncode == 0
    ctx = json.loads((ep / "context.json").read_text())
    gap = [x for x in ctx["reader_issues"] if x["kind"] == "signal_gap"]
    assert len(gap) == 1 and (gap[0]["t0_s"], gap[0]["t1_s"]) == (1.5, 1.7)
    assert ctx["clock_zero_s"] == 0.0 and ctx["clock_start_s"] == pytest.approx(-0.5)
    assert ctx["duration_s"] == 2.0 and ctx["state_span"] == [15, 60]
    with np.load(ep / "times.npz") as z:
        assert z["left"][0] == pytest.approx(0.0) and z["right"][0] == pytest.approx(0.5)
    e = me.load(ep)
    pl = me.plan(e)
    assert me.frame_time(e, 0) == pytest.approx(0.0) and min(me.frame_time(e, k) for k in pl["ks"]) >= 0
    assert e["state"][15][0] == 0
    assert _clip_timing(out / "wrist_left" / f"{ep.name}.mp4")[1] == pytest.approx(0.0, abs=0.02)
    assert _clip_timing(out / "wrist_right" / f"{ep.name}.mp4")[1] == pytest.approx(0.5, abs=0.02)
    assert clips.clip_frames(out / "wrist_left" / f"{ep.name}.mp4") == 60


def test_a_labelled_episode_keeps_its_clock_and_says_the_camera_starts_before_it(tmp_path):
    """The same episode, already labelled (its run's record is in run/out beside the episodes folder, as python -m
    review and Data Review lay a job out): the labels are on the episode's clock, so the clock does not move. The
    camera that started before it is cut from the clock's start, and a reader issue says how much of it is not
    shown."""
    eps, ep = _recording(tmp_path, left_start=-0.5)
    (tmp_path / "run" / "out").mkdir(parents=True)
    (tmp_path / "run" / "out" / f"{ep.name}.json").write_text("{}")
    (eps.parent / "up" / "top.mp4").write_bytes(b"not a video at all" * 50)
    out = tmp_path / "clips"
    assert _clips(eps, out).returncode == 0
    ctx = json.loads((ep / "context.json").read_text())
    with np.load(ep / "times.npz") as z:
        assert z["left"][0] == pytest.approx(-0.5) and z["right"][0] == pytest.approx(0.0)
    assert ctx["clock_start_s"] == pytest.approx(0.0) and ctx["state_span"] == [15, 60]
    # the length is the recording's, through the right camera's last frame (2.47 s), never the new main camera's span
    assert ctx["duration_s"] == 2.5
    # a relabel samples nothing before the clock's start: the first instant is the left camera's frame at 0 s
    e = me.load(ep)
    pl = me.plan(e)
    assert pl["ks"][0] == 15 and min(me.frame_time(e, k) for k in pl["ks"]) == pytest.approx(0.0)
    (off,) = [x for x in ctx["reader_issues"] if x["kind"] == "camera_offset"]
    assert off["camera"] == "left" and "0.50 s before" in off["what"]
    assert clips.clip_frames(out / "wrist_left" / f"{ep.name}.mp4") == 45


def test_every_camera_a_frame_short_is_labelled_to_the_last_frame_any_camera_has(tmp_path):
    """The real upload: every camera's file ends a frame before the episode does. The episode is labelled, and its
    last instant is the last frame the cameras have."""
    rep, eps, ep = _upload(tmp_path, {"top": 30, "wrist_left": 30, "wrist_right": 30})
    for s in json.loads((ep / "sources.json").read_text()).values():
        _video(Path(s["packed"]), 29)
    req = me.build_request(ep)
    texts = [c["text"] for c in req["content"] if c.get("type") == "text"]
    last = [t for t in texts if t.startswith("=== detail view, last frame")]
    assert len(last) == 1 and "t=0.93s | cameras top, left, right" in last[0], last
    assert req["timesteps"][-1] == pytest.approx(0.933, abs=0.001)
    assert ("Every camera's video ends before the episode does, so the last instant is the last frame they have, at "
            "0.93 s.") in req["prompt"]
    assert "plus its first frame and the last frame its cameras have." in req["prompt"]


def test_two_cameras_short_are_named_together(tmp_path):
    rep, eps, ep = _upload(tmp_path, {"top": 30, "wrist_left": 30, "wrist_right": 30})
    src = json.loads((ep / "sources.json").read_text())
    for v in ("left", "right"):
        _video(Path(src[v]["packed"]), 29)
    prompt = me.build_request(ep)["prompt"]
    assert "Left's video ends before the episode does, so it has no frame at 0.97 s; right's video ends" in prompt
    assert "Their cells at those times are empty, and they are left out of a detail view there." in prompt


def _cut_in_last_frame(path: Path, frames: int) -> None:
    """A camera of frames frames (MPEG-4, whose decoder fills in what a frame is missing), a keyframe every 10, its
    index at the start of the file, and the file cut in the middle of its last frame, as an upload cut off in transfer
    is: the index still lists every frame."""
    import av
    with av.open(str(path), "w", options={"movflags": "+faststart"}) as c:
        s = c.add_stream("mpeg4", rate=30)
        s.width, s.height, s.pix_fmt = 160, 120, "yuv420p"
        s.options = {"g": "10"}
        rng = np.random.default_rng(1)
        for _ in range(frames):
            im = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
            for pk in s.encode(av.VideoFrame.from_ndarray(im, format="rgb24")):
                c.mux(pk)
        for pk in s.encode():
            c.mux(pk)
    with av.open(str(path)) as c:
        last = max((pk for pk in c.demux(c.streams.video[0]) if pk.size), key=lambda pk: pk.pos)
        cut = last.pos + last.size // 2
    path.write_bytes(path.read_bytes()[:cut])


def test_a_frame_cut_short_is_never_sent_as_footage(tmp_path):
    """A camera file cut in the middle of its last frame, a keyframe: the decoder makes up the missing part of the
    picture, from the frame before it when decoded in order and from nothing after a seek, which came out smeared and
    was sent to the model as the camera's view. The decoder marks such a frame as damaged, and it is a frame that could
    not be decoded: its cell is empty and the prompt says so."""
    from label import frames as mf
    up = tmp_path / "up"
    up.mkdir()
    _video(up / "top.mp4", 21)
    _cut_in_last_frame(up / "wrist_left.mp4", 21)
    with pytest.raises(mf.DamagedFrame):
        mf.extract_frames(up / "wrist_left.mp4", 0.0, 21, [20])
    assert list(mf.extract_frames(up / "wrist_left.mp4", 0.0, 21, [19])) == [19]
    eps = tmp_path / "episodes"
    rep = formats.convert(up, "teleop_arms", eps, "mine", float("inf"), grouping={})
    ep = me.load(eps / rep["episodes"][0]["episode_id"])
    pl = me.plan(ep)
    assert pl["ks"][-1] == 20
    imgs = me.frames(ep, pl)
    assert 20 not in imgs["left"] and ep["decode_failed"] == {"left": [20]}, ep["decode_failed"]
    assert "left's video could not be decoded at 0.67 s" in me.build_request(ep["dir"])["prompt"].lower()


def test_a_camera_damaged_partway_leaves_only_its_own_cells_empty(tmp_path):
    """A camera whose file does not decode for a stretch in the middle: the episode is labelled from a read only
    folder, that camera's cells are empty where it does not decode, the prompt says so, and the request returns the
    stretch, which the run's record carries and the board shows as a reader issue. Nothing is written into the
    episode folder."""
    import os
    import stat
    from board import build as board_build
    from label import harness
    up = tmp_path / "up"
    up.mkdir()
    _damaged(up / "top.mp4", 300, range(0))
    _damaged(up / "wrist_left.mp4", 300, range(130, 150))
    eps = tmp_path / "episodes"
    rep = formats.convert(up, "teleop_arms", eps, "mine", float("inf"), grouping={})
    ep = eps / rep["episodes"][0]["episode_id"]
    before = (ep / "context.json").read_text()
    modes = {p: p.stat().st_mode for p in [ep, *ep.iterdir()]}
    for p in modes:
        p.chmod(stat.S_IMODE(modes[p]) & ~0o222)
    try:
        req = me.build_request(ep)
        out = tmp_path / "out.json"
        harness.label_episode(ep, out, model="m", reasoning="low", api_key="", max_tokens=1, timeout=5,
                              dry_run=True)
    finally:
        for p, m in modes.items():
            os.chmod(p, m)
    assert req["views"] == ["exo", "left"]
    assert "left's video could not be decoded at" in req["prompt"].lower()
    (bad,) = req["decode_failed"]
    assert bad["camera"] == "left" and 4.0 <= bad["t0_s"] <= bad["t1_s"] <= 5.5
    assert "could not be decoded" in bad["what"]
    assert (ep / "context.json").read_text() == before
    assert json.loads(out.read_text())["decode_failed"] == req["decode_failed"]
    d = {}
    board_build.add_context(d, json.loads(before), ep, {"decode_failed": req["decode_failed"]})
    (iss,) = [x for x in d["dataset_checks"]["reader_issues"] if x["kind"] == "camera_decode_failed"]
    assert iss["camera"] == "left" and iss["what"] == bad["what"] and iss["t0_s"] == bad["t0_s"]


def test_a_missing_camera_file_raises_and_a_garbage_one_is_named_not_decodable(tmp_path):
    """A camera file that is gone is a fault on our side and raises; a file that is not a video is named as one that
    does not decode, never as a video that ends early."""
    rep, eps, ep = _upload(tmp_path, {"top": 60, "wrist_left": 60})
    left = Path(json.loads((ep / "sources.json").read_text())["left"]["packed"])
    left.write_bytes(b"not a video at all" * 50)
    req = me.build_request(ep)
    assert "Left's video could not be decoded at any instant. Its cells are all empty, and it is left out of "\
        "every detail view." in req["prompt"]
    assert "ends before the episode" not in req["prompt"]
    assert [x["camera"] for x in req["decode_failed"]] == ["left"]
    left.unlink()
    with pytest.raises(FileNotFoundError):
        me.build_request(ep)


def test_the_job_notes_list_every_reader_issue_once(tmp_path):
    ep = tmp_path / "episode_1"
    ep.mkdir()
    (ep / "context.json").write_text(json.dumps({"reader_issues": [
        {"kind": "signal_gap", "signal": "force", "what": "The force signal stops for 2 s."},
        {"kind": "camera_not_decodable", "camera": "right", "what": "The right wrist camera video could not be "
                                                                     "decoded, so this episode is shown and labelled "
                                                                     "without it."},
        {"kind": "no_sentence", "what": ""}]}))
    rep = {"episodes": [{"name": "take 1", "episode_id": "episode_1", "cameras": {"exo": "top", "right": "r"}}],
           "notes": []}
    clips.note_camera_problems(rep, tmp_path)
    clips.note_camera_problems(rep, tmp_path)
    assert rep["notes"] == ["take 1: the force signal stops for 2 s.",
                            "take 1: the right wrist camera video could not be decoded, so this episode is shown and "
                            "labelled without it."]
    assert rep["episodes"][0]["cameras"] == {"exo": "top"}


def test_one_damaged_stretch_never_stops_a_long_recording_from_being_cut(tmp_path):
    """The motion a long recording is cut by (label/pieces.py video_motion) reads the main camera's image change: a
    stretch that does not decode has no motion measured, and the frames after it are read as usual."""
    from label import pieces
    up = tmp_path / "up"
    up.mkdir()
    _damaged(up / "top.mp4", 300, range(130, 150))
    rep = formats.convert(up, "teleop_arms", tmp_path / "eps", "mine", float("inf"), grouping={})
    ep = me.load(tmp_path / "eps" / rep["episodes"][0]["episode_id"])
    n = int(ep["sources"]["exo"]["n_frames"])
    m = pieces.video_motion(ep, n)
    assert len(m) == n and m[100:125].sum() > 0 and m[200:290].sum() > 0     # before and after the damage


def test_an_instant_no_camera_can_show_is_left_out_of_the_request(tmp_path, monkeypatch):
    """The main camera does not decode for a stretch and the other camera was not recording there (outside its own
    span): no camera can show those instants, so they leave the request, and the episode is labelled from the rest."""
    up = tmp_path / "up"
    up.mkdir()
    _damaged(up / "top.mp4", 300, range(130, 150))
    _damaged(up / "wrist_left.mp4", 300, range(0))
    rep = formats.convert(up, "teleop_arms", tmp_path / "eps", "mine", float("inf"), grouping={})
    ep = tmp_path / "eps" / rep["episodes"][0]["episode_id"]
    span = me._in_span
    monkeypatch.setattr(me, "_in_span", lambda e, v, k: span(e, v, k) and (v == "exo" or not 120 <= k <= 170))
    req = me.build_request(ep)
    (bad,) = [x for x in req["decode_failed"] if x["camera"] == "exo"]
    assert 4.0 <= bad["t0_s"] <= bad["t1_s"] <= 5.5
    assert req["timesteps"] and not [t for t in req["timesteps"] if bad["t0_s"] <= t <= bad["t1_s"]]


def test_a_single_cameras_damaged_stretch_is_flagged(tmp_path):
    """With one camera, the instants it cannot decode have no frame at all and leave the request, and the stretch is
    still in the record the board flags it from."""
    up = tmp_path / "up"
    up.mkdir()
    _damaged(up / "top.mp4", 300, range(130, 150))
    rep = formats.convert(up, "teleop_arms", tmp_path / "eps", "mine", float("inf"), grouping={})
    req = me.build_request(tmp_path / "eps" / rep["episodes"][0]["episode_id"])
    (bad,) = req["decode_failed"]
    assert bad["camera"] == "exo" and 4.0 <= bad["t0_s"] <= bad["t1_s"] <= 5.5



def test_a_camera_the_model_is_not_shown_is_cut_for_the_board(tmp_path):
    """A camera in context.json unshown_cameras (prepare/formats.py: one the model is not shown) is cut like any other
    camera, at its own offset on the episode's clock, into CLIPS/unshown<N>/; one whose video does not decode is
    flagged on the episode and costs nothing else. The episode's own cameras and the prompt are unchanged."""
    rep, eps, ep = _upload(tmp_path, {"top": 60})
    before = me.build_request(ep)["prompt"]
    _video(tmp_path / "ir.mp4", 45)
    (tmp_path / "mask.mp4").write_bytes(b"not a video" * 40)
    ctx = json.loads((ep / "context.json").read_text())
    ctx["unshown_cameras"] = [
        {"name": "cam_ir", "why": "an infrared video", "packed": str(tmp_path / "ir.mp4"), "base_s": 0.0,
         "n_frames": 45, "start_s": 0.5, "fps": 30.0},
        {"name": "cam_mask", "why": "a mask video", "packed": str(tmp_path / "mask.mp4"), "base_s": 0.0,
         "n_frames": 45, "start_s": 0.0, "fps": 30.0}]
    (ep / "context.json").write_text(json.dumps(ctx))
    jobs = {j[-1]: j for j in clips.episode_jobs(ep, tmp_path / "clips", False)}
    assert set(jobs) == {"exo", "unshown1", "unshown2"}
    assert jobs["unshown1"][3] == tmp_path / "clips" / "unshown1" / f"{ep.name}.mp4"
    assert jobs["unshown1"][5] is False and jobs["unshown1"][6] == 0.5        # a side camera, 0.5 s into the clock
    r = _clips(eps, tmp_path / "clips")
    assert r.returncode == 0, r.stderr
    ir = tmp_path / "clips" / "unshown1" / f"{ep.name}.mp4"
    assert clips.clip_frames(ir) == 45 and not (tmp_path / "clips" / "unshown2" / f"{ep.name}.mp4").exists()
    ri = json.loads((ep / "context.json").read_text())["reader_issues"]
    assert [(x["kind"], x["camera"]) for x in ri] == [("unshown_camera_not_decodable", "unshown2")]
    assert "cam_mask" in ri[0]["what"] and json.loads((ep / "sources.json").read_text()).keys() == {"exo"}
    assert me.build_request(ep)["prompt"] == before


def test_a_camera_the_model_is_not_shown_reaches_the_board_named_with_why(tmp_path):
    from board import build as board_build
    from board import serve, static
    d = {}
    board_build.add_context(d, {"profile": "ego_head", "fps": 30, "unshown_cameras": [
        {"name": "cam_ir", "why": "an infrared video", "packed": "/x/ir.mp4", "n_frames": 45},
        {"name": "no file", "why": "x"}]}, tmp_path)
    assert d["unshown_cameras"] == [{"view": "unshown1", "name": "cam_ir", "why": "an infrared video"}]
    assert serve.clip_path(tmp_path, "e", "unshown1") == tmp_path / "unshown1" / "e.mp4"
    assert static.media_key("unshown1") == "unshown1"
    assert static.shown_cams({**d, "_rig": "ego_head", "camera_views": ["exo"]}) == ["exo", "unshown1"]
    r = subprocess.run(["node", str(REPO / "tests" / "unshown_cameras.js"), str(REPO / "board" / "serve.py")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_a_camera_the_model_is_not_shown_moves_with_the_clock(tmp_path):
    """The main camera is taken out and the clock moves to the earliest camera left, 2 s later: a camera the model is
    not shown starts 2 s earlier on the new clock, and its clip is cut at that start."""
    d = tmp_path / "episode_000000"
    d.mkdir()
    t = {"exo": np.arange(0, 10, 1 / 30), "left": np.arange(2, 12, 1 / 30)}
    np.savez(d / "times.npz", **t)
    ctx = {"fps": 30, "state_kind": "none", "real_times": "times.npz",
           "unshown_cameras": [{"name": "ir", "why": "infrared", "packed": str(tmp_path / "ir.mp4"), "base_s": 0.0,
                                "n_frames": 300, "start_s": 5.0, "fps": 30}]}
    left = {"packed": str(tmp_path / "b.mp4"), "base_s": 0, "n_frames": 300}
    (d / "sources.json").write_text(json.dumps({"left": left}))
    clips.reanchor(d, ctx, {"left": left}, t, "exo", "left", "top camera")
    assert ctx["unshown_cameras"][0]["start_s"] == 3.0
    (d / "context.json").write_text(json.dumps(ctx))
    (job,) = [j for j in clips.episode_jobs(d, tmp_path / "clips", True) if j[-1] == "unshown1"]
    assert job[6] == 3.0


def test_board_clips_cuts_a_camera_the_model_is_not_shown_again_when_the_main_camera_goes(tmp_path):
    """board clips on an episode whose main camera does not decode: the clock moves to the left camera, which started
    2 s later, so a camera the model is not shown that started 5 s in starts 3 s in on the new clock, and its clip,
    cut first at 5 s, is cut again at 3 s."""
    from board.hands import probe_pts
    eps = tmp_path / "episodes"
    ep = eps / "episode_000000"
    ep.mkdir(parents=True)
    (ep / "top.mp4").write_bytes(b"not a video" * 50)
    _video(ep / "left.mp4", 60)
    _video(ep / "ir.mp4", 45)
    np.savez(ep / "times.npz", exo=np.arange(60) / 30.0, left=2.0 + np.arange(60) / 30.0)
    np.save(ep / "kmap_left.npy", np.zeros(60, dtype=np.int64))     # paired by time: its frame nearest each top frame
    (ep / "sources.json").write_text(json.dumps({
        "exo": {"packed": str(ep / "top.mp4"), "base_s": 0.0, "n_frames": 60},
        "left": {"packed": str(ep / "left.mp4"), "base_s": 0.0, "n_frames": 60, "kmap": "kmap_left.npy"}}))
    (ep / "context.json").write_text(json.dumps({
        "profile": "teleop_arms", "state_kind": "none", "fps": 30, "n_state_frames": 60, "real_times": "times.npz",
        "cameras": {"exo": {"name": "top"}, "left": {"name": "wrist_left"}},
        "unshown_cameras": [{"name": "cam_ir", "why": "an infrared video", "packed": str(ep / "ir.mp4"),
                             "base_s": 0.0, "n_frames": 45, "start_s": 5.0, "fps": 30.0}]}))
    r = _clips(eps, tmp_path / "clips")
    assert r.returncode == 0, r.stderr
    ctx = json.loads((ep / "context.json").read_text())
    assert list(json.loads((ep / "sources.json").read_text())) == ["left"]          # the main camera is out
    assert ctx["unshown_cameras"][0]["start_s"] == 3.0
    _, _, tb, pts = probe_pts(tmp_path / "clips" / "unshown1" / "episode_000000.mp4")
    assert len(pts) == 45 and abs(float(pts[0] * tb) - 3.0) < 0.02
    assert not [x for x in ctx.get("reader_issues") or [] if x.get("kind") == "unshown_camera_not_decodable"]
