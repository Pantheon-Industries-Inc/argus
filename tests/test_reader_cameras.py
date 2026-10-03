"""The reader keeps every camera that works (prepare/formats.py): a camera that cannot be opened, ends early, has
undecodable frames or a short depth stream is flagged on its episode and the rest of the episode is kept; a take of
many cameras is one episode, and every infrared, thermal or mask video is shown with its episode."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from prepare import formats as f
from test_formats import _issues, _lerobot


def _mp4(path: Path, n: int, shade: int = 60, w: int = 64, h: int = 48) -> Path:
    """n frames of an mpeg4 video at 30 fps, each frame a little brighter."""
    import av
    path.parent.mkdir(parents=True, exist_ok=True)
    c = av.open(str(path), "w", format="mov" if path.suffix in (".mp4", ".mov") else None)
    s = c.add_stream("mpeg4", rate=30)
    s.width, s.height, s.pix_fmt = w, h, "yuv420p"
    for k in range(n):
        fr = av.VideoFrame.from_ndarray(np.full((h, w, 3), (shade + 3 * k) % 256, np.uint8), format="rgb24")
        fr.pts = k
        for pkt in s.encode(fr):
            c.mux(pkt)
    for pkt in s.encode():
        c.mux(pkt)
    c.close()
    return path


def _ctx(out: Path, rep: dict, i: int = 0) -> dict:
    return json.loads((out / rep["episodes"][i]["episode_id"] / "context.json").read_text())


# ---------------------------------------------------------------- the container of a video named for another one

def test_a_matroska_video_named_mp4_is_read_by_its_own_container(tmp_path):
    """A valid Matroska video named .mp4 had been refused, since the extension forced the demuxer. The container is
    read from the file's own first bytes when the named one fails, and a file that is no video container (a concat
    script, which could make ffmpeg read other files) is still refused."""
    import pytest
    mkv = _mp4(tmp_path / "real.mkv", 12)
    named = tmp_path / "cam.mp4"
    named.write_bytes(mkv.read_bytes())
    assert len(f.probe(named)["pts"]) == 12
    script = tmp_path / "list.mp4"
    script.write_text("ffconcat version 1.0\nfile /etc/hosts\n")
    with pytest.raises(Exception):
        f.probe(script)


# ---------------------------------------------------------------- a camera that cannot be opened

def _three_cameras(root: Path, n: int = 30) -> Path:
    d = root / "ep1"
    for k, cam in enumerate(("top", "wrist_left", "wrist_right")):
        _mp4(d / f"{cam}.mp4", n, 40 * k)
    return d


def test_a_camera_that_cannot_be_opened_leaves_the_episode_with_the_others(tmp_path):
    """One empty and one garbage camera had dropped the whole episode ("no episode could be read"). Each is left out
    with a camera_not_decodable issue naming it and why, and the episode is labelled from the camera that opens."""
    root = tmp_path / "up"
    d = _three_cameras(root)
    (d / "wrist_left.mp4").write_bytes(b"")
    (d / "wrist_right.mp4").write_bytes(np.random.default_rng(0).bytes(20_000))
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    assert not rep["failed"] and len(rep["episodes"]) == 1, rep
    ctx = _ctx(tmp_path / "eps", rep)
    assert [c["key"] for c in ctx["cameras"].values()] == ["top"]
    bad = {i["camera"]: i["what"] for i in _issues(ctx, "camera_not_decodable")}
    assert set(bad) == {"wrist_left", "wrist_right"}, ctx.get("reader_issues")
    assert "empty" in bad["wrist_left"]
    assert any(u.startswith("wrist_left (") for u in ctx["source"]["unused_cameras"])


def test_when_the_main_camera_cannot_be_opened_another_camera_is_the_anchor(tmp_path):
    """The main camera empty: the next camera in row order is the episode's anchor, as board clips re anchors an
    episode whose main camera does not decode, and the episode keeps its length."""
    root = tmp_path / "up"
    d = _three_cameras(root)
    (d / "top.mp4").write_bytes(b"")
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    assert not rep["failed"] and len(rep["episodes"]) == 1, rep
    ctx = _ctx(tmp_path / "eps", rep)
    assert list(ctx["cameras"]) == ["left", "right"] and ctx["n_state_frames"] == 30
    assert [i["camera"] for i in _issues(ctx, "camera_not_decodable")] == ["top"]


def test_a_camera_the_model_is_not_shown_that_cannot_be_opened_is_named(tmp_path):
    """A mask video beside the cameras that cannot be opened had been left off the board without a word."""
    root = tmp_path / "up"
    d = _three_cameras(root)
    (d / "cam_mask.mp4").write_bytes(b"")
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    ctx = _ctx(tmp_path / "eps", rep)
    assert len(ctx["cameras"]) == 3 and not ctx.get("unshown_cameras")
    assert [i["camera"] for i in _issues(ctx, "camera_not_decodable")] == ["cam_mask"], ctx.get("reader_issues")


def test_an_episode_none_of_whose_cameras_opens_names_every_camera(tmp_path):
    root = tmp_path / "up"
    d = _three_cameras(root)
    for cam in ("top", "wrist_left", "wrist_right"):
        (d / f"{cam}.mp4").write_bytes(b"")
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    assert not rep["episodes"] and len(rep["failed"]) == 1, rep
    why = rep["failed"][0]["why"]
    assert all(cam in why for cam in ("top", "wrist_left", "wrist_right")), why


def test_a_lerobot_camera_that_cannot_be_opened_leaves_the_episode_and_its_state(tmp_path):
    """LeRobot v2: the main camera's file empty had dropped the episode. The wrist camera carries it, on the same
    frame index, with the state kept."""
    root = tmp_path / "lr"
    state = np.cumsum(np.random.default_rng(1).normal(0, 0.01, (30, 14)), axis=0)
    _lerobot(root, {0: {"observation.state": list(state)}},
             feats={"observation.images.cam_left_wrist": {"dtype": "video", "shape": [48, 64, 3]}})
    _mp4(root / "videos" / "chunk-000" / "observation.images.cam_left_wrist" / "episode_000000.mp4", 30)
    (root / "videos" / "chunk-000" / "observation.images.cam_high" / "episode_000000.mp4").write_bytes(b"")
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    assert not rep["failed"] and len(rep["episodes"]) == 1, rep
    ctx = _ctx(tmp_path / "eps", rep)
    assert [c["key"] for c in ctx["cameras"].values()] == ["observation.images.cam_left_wrist"]
    assert ctx["state_kind"] == "joints" and ctx["n_state_frames"] == 30
    assert [i["camera"] for i in _issues(ctx, "camera_not_decodable")] == ["observation.images.cam_high"]


# ---------------------------------------------------------------- cameras of unequal length

def test_a_lerobot_episode_is_as_long_as_its_main_camera_and_a_short_camera_is_flagged(tmp_path):
    """LeRobot v2: the episode had taken its shortest camera's length, so a side camera cut to half cut the labels,
    signals and timeline to half too. The episode is as long as its main camera, and the short camera is flagged with
    the stretch it does not cover."""
    root = tmp_path / "lr"
    _lerobot(root, {0: {"observation.state": [np.zeros(5)] * 30}},
             feats={"observation.state": {"dtype": "float32", "shape": [5]},
                    "observation.images.cam_left_wrist": {"dtype": "video", "shape": [48, 64, 3]}})
    _mp4(root / "videos" / "chunk-000" / "observation.images.cam_left_wrist" / "episode_000000.mp4", 9)
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    ctx = _ctx(tmp_path / "eps", rep)
    assert ctx["n_state_frames"] == 30, ctx["n_state_frames"]
    (short,) = _issues(ctx, "camera_short")
    assert short["camera"] == "observation.images.cam_left_wrist"
    assert abs(short["t0_s"] - 0.3) < 0.05 and abs(short["t1_s"] - 1.0) < 0.05, short


def test_a_side_camera_that_ends_early_is_flagged(tmp_path):
    root = tmp_path / "up"
    _mp4(root / "ep1" / "top.mp4", 60)
    _mp4(root / "ep1" / "wrist_left.mp4", 30)
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    ctx = _ctx(tmp_path / "eps", rep)
    assert ctx["n_state_frames"] == 60
    (short,) = _issues(ctx, "camera_short")
    assert short["camera"] == "wrist_left" and abs(short["t0_s"] - 1.0) < 0.05 and abs(short["t1_s"] - 2.0) < 0.05


def test_a_main_camera_that_ends_early_is_flagged_with_what_the_others_cover_past_it(tmp_path):
    """The main camera ending at 1 s of the side camera's 2 s had left the second half unlabelled with no word. The
    episode stays on the main camera's frames (its views name the cameras' roles, so a wrist camera is never made the
    scene camera), and the main camera is flagged with the stretch the other cameras cover past it."""
    root = tmp_path / "up"
    _mp4(root / "ep1" / "top.mp4", 30)
    _mp4(root / "ep1" / "wrist_left.mp4", 60)
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    ctx = _ctx(tmp_path / "eps", rep)
    (short,) = _issues(ctx, "main_camera_short")
    assert short["camera"] == "top" and abs(short["t0_s"] - 1.0) < 0.05 and abs(short["t1_s"] - 2.0) < 0.05, short
    assert "wrist_left" in short["what"] or "left wrist" in short["what"]
    assert not _issues(ctx, "camera_short")


def test_cameras_of_one_length_raise_no_issue(tmp_path):
    root = tmp_path / "up"
    _three_cameras(root)
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    assert not _issues(_ctx(tmp_path / "eps", rep))


# ---------------------------------------------------------------- a LeRobot camera with no video

def test_a_lerobot_camera_with_no_video_for_the_episode_is_listed_with_its_reason(tmp_path):
    root = tmp_path / "lr"
    _lerobot(root, {0: {"observation.state": [np.zeros(14)] * 30}},
             feats={"observation.images.cam_left_wrist": {"dtype": "video", "shape": [48, 64, 3]}})
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    ctx = _ctx(tmp_path / "eps", rep)
    (u,) = [u for u in ctx["source"]["unused_cameras"] if u.startswith("observation.images.cam_left_wrist")]
    assert "(" in u and "video" in u, u
