"""The reader keeps every camera that works (prepare/formats.py): a camera that cannot be opened, ends early, has
undecodable frames or a short depth stream is flagged on its episode and the rest of the episode is kept; a take of
many cameras is one episode, and every infrared, thermal or mask video is shown with its episode."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from prepare import formats as f
from test_formats import _issues, _jpeg, _lerobot, _png


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


# ---------------------------------------------------------------- undecodable frames inside a camera

def _lerobot_images(root: Path, bad: tuple = ()) -> None:
    """A LeRobot v2.1 episode of 30 frames whose camera is PNG images in the data file, with a 14 value state."""
    import pandas as pd
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    feats = {"observation.images.cam_high": {"dtype": "image", "shape": [72, 96, 3]},
             "observation.state": {"dtype": "float32", "shape": [14]}}
    (root / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v2.1", "fps": 30, "chunks_size": 1000, "features": feats,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"}))
    cells = [{"bytes": _png(np.full((72, 96, 3), 8 * k, np.uint8)) if k not in bad else b"\x89PNG\r\n\x1a\nbroken",
              "path": None} for k in range(30)]
    state = np.cumsum(np.random.default_rng(2).normal(0, 0.01, (30, 14)), axis=0)
    pd.DataFrame({"observation.images.cam_high": cells, "observation.state": list(state),
                  "frame_index": np.arange(30), "episode_index": np.zeros(30, int),
                  "timestamp": np.arange(30) / 30}).to_parquet(root / "data" / "chunk-000" / "episode_000000.parquet")


def test_one_undecodable_image_keeps_the_state_and_a_frame_for_every_row(tmp_path):
    """One undecodable PNG had made the camera one frame short of the table, which dropped the whole arm state. Every
    row has a frame (a blank one where the image fails), the state is kept, and the blank frames are flagged."""
    root = tmp_path / "lr"
    _lerobot_images(root, bad=(10, 11))
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    ctx = _ctx(tmp_path / "eps", rep)
    ep = tmp_path / "eps" / ctx["episode_id"]
    assert ctx["n_state_frames"] == 30 and ctx["state_kind"] == "joints", ctx.get("state_note")
    assert np.load(ep / "state.npz")["state"].shape == (30, 14)
    (bad,) = _issues(ctx, "frames_not_decodable")
    assert bad["camera"] == "observation.images.cam_high"
    assert abs(bad["t0_s"] - 10 / 30) < 0.01 and abs(bad["t1_s"] - 11 / 30) < 0.01, bad


def test_an_hdf5_camera_with_an_undecodable_frame_keeps_its_rows_on_their_frames(tmp_path):
    """An HDF5 camera of encoded JPEGs with frame 4 garbage had lost that frame, so every state row after it was one
    frame early. The frame is written blank, the state stays one row per frame, and the frame is flagged."""
    import h5py
    root = tmp_path / "h5"
    root.mkdir()
    n = 12
    with h5py.File(root / "episode_0.hdf5", "w") as h:
        o = h.create_group("observations")
        o["qpos"] = np.cumsum(np.random.default_rng(3).normal(0, 0.01, (n, 14)), axis=0)
        enc = [_jpeg(10 * k) for k in range(n)]
        enc[4] = b"\xff\xd8\xff" + bytes(300)
        ds = o.create_dataset("images/cam_high", (n,), dtype=h5py.vlen_dtype(np.dtype("uint8")))
        for i, b in enumerate(enc):
            ds[i] = np.frombuffer(b, np.uint8)
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    ctx = _ctx(tmp_path / "eps", rep)
    assert ctx["n_state_frames"] == n and ctx["state_kind"] == "joints", ctx.get("state_note")
    (bad,) = _issues(ctx, "frames_not_decodable")
    assert bad["camera"].endswith("cam_high") and abs(bad["t0_s"] - 4 / 30) < 0.01, bad


def _json_mcap(path: Path, chans: dict, n: int = 30, t0: float = 1_790_000_000.0, joints: bool = False) -> None:
    """chans {topic: (schema name, fn(k) -> message dict)} at 30 Hz as JSON messages, and an arm's joints at 100 Hz."""
    from mcap.writer import Writer
    msgs = [(k / 30, t, fn(k)) for t, (_, fn) in chans.items() for k in range(n)]
    if joints:
        msgs += [(k / 100, "/right_arm/joint_state", {"joint_pos": [0.01 * k] * 6, "gripper_pos": [0.5]})
                 for k in range(int(n / 30 * 100))]
    msgs.sort(key=lambda m: m[0])
    with open(path, "wb") as fh:
        w = Writer(fh)
        w.start()
        ids = {}
        for _, t, _m in msgs:
            if t not in ids:
                sid = w.register_schema(name=chans[t][0] if t in chans else "joint_state", encoding="jsonschema",
                                        data=b"{}")
                ids[t] = w.register_channel(topic=t, message_encoding="json", schema_id=sid)
        for s, t, m in msgs:
            ns = int((t0 + s) * 1e9)
            w.add_message(ids[t], log_time=ns, publish_time=ns, data=json.dumps(m).encode())
        w.finish()


def _jpg_msg(k: int) -> dict:
    return {"format": "jpeg", "data": base64.b64encode(_jpeg(8 * k % 256)).decode()}


def _depth_msg(k: int) -> dict:
    return {"width": 64, "height": 48, "encoding": "16UC1", "step": 128,
            "data": base64.b64encode(np.full((48, 64), 900 + 10 * k, np.uint16).tobytes()).decode()}


def test_undecodable_frames_inside_an_mcap_camera_are_flagged(tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    bad = lambda k: {"format": "jpeg", "data": base64.b64encode(b"\xff\xd8\xff" + bytes(200)).decode()} \
        if 10 <= k < 13 else _jpg_msg(k)
    _json_mcap(root / "run.mcap", {"/camera/front/image": ("foxglove.CompressedImage", bad)})
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    ctx = _ctx(tmp_path / "eps", rep)
    (iss,) = _issues(ctx, "frames_not_decodable")
    assert iss["camera"] == "/camera/front/image"
    assert abs(iss["t0_s"] - 10 / 30) < 0.01 and abs(iss["t1_s"] - 12 / 30) < 0.01, iss
    assert "3" in iss["what"]


# ---------------------------------------------------------------- depth shorter than its colour camera

def test_depth_that_ends_before_its_camera_gives_no_reading_past_its_end(tmp_path):
    """A depth video a third as long as its colour camera had its last frame given to the model for every later
    instant. Anchor frames with no depth frame within a frame of them get no depth reading, the prompt says so, and
    the stretch is a depth_partial issue."""
    from label import depth as dp
    from label import episode as me
    root = tmp_path / "up"
    d = root / "ep1"
    _mp4(d / "exo_cam-images-rgb.mp4", 60)
    dw = f.DepthWriter(d / "exo_cam-images-depth.mkv")
    for k in range(20):
        dw.add(k / 30, np.full((48, 64), 800 + k, np.uint16), 0.001)
    dw.close()
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    ctx = _ctx(tmp_path / "eps", rep)
    ep = tmp_path / "eps" / ctx["episode_id"]
    km = np.load(ep / "depth_kmap_exo.npy")
    assert (km[:20] == np.arange(20)).all() and (km[21:] < 0).all(), km
    (part,) = _issues(ctx, "depth_partial")
    assert abs(part["t0_s"] - 20 / 30) < 0.05 and abs(part["t1_s"] - 59 / 30) < 0.05, part
    e = me.load(ep)
    assert dp.at_anchor(e, dp.load(ep), "exo", [5, 50]).keys() == {5}
    assert "no depth" in me._depth_note({**e, "depth": dp.load(ep)})


def test_depth_as_long_as_its_camera_reads_at_every_frame(tmp_path):
    from label import depth as dp
    from label import episode as me
    root = tmp_path / "up"
    d = root / "ep1"
    _mp4(d / "exo_cam-images-rgb.mp4", 30)
    dw = f.DepthWriter(d / "exo_cam-images-depth.mkv")
    for k in range(30):
        dw.add(k / 30, np.full((48, 64), 800 + k, np.uint16), 0.001)
    dw.close()
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    ctx = _ctx(tmp_path / "eps", rep)
    ep = tmp_path / "eps" / ctx["episode_id"]
    assert (np.load(ep / "depth_kmap_exo.npy") == np.arange(30)).all()
    assert not _issues(ctx)
    note = me._depth_note({**me.load(ep), "depth": dp.load(ep)})
    assert note.endswith("at that instant.")


# ---------------------------------------------------------------- MCAP depth topics

def test_an_mcap_whose_only_camera_is_depth_is_labelled_from_its_depth_picture(tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    _json_mcap(root / "run.mcap", {"/camera/front/depth": ("foxglove.RawImage", _depth_msg)}, joints=True)
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    assert not rep["failed"] and len(rep["episodes"]) == 1, rep
    ctx = _ctx(tmp_path / "eps", rep)
    assert [c["key"] for c in ctx["cameras"].values()] == ["/camera/front/depth"]
    assert [i["camera"] for i in _issues(ctx, "camera_not_colour")] == ["/camera/front/depth"]
    assert ctx["state_kind"] == "joints" and ctx["n_state_frames"] == 30
    assert f.probe(tmp_path / "eps" / ctx["episode_id"] / "exo.mp4")["width"] == 64


def test_an_mcap_depth_topic_with_no_camera_of_its_own_goes_to_the_board(tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    _json_mcap(root / "run.mcap", {"/front/image": ("foxglove.CompressedImage", _jpg_msg),
                                   "/lidar/depth": ("foxglove.RawImage", _depth_msg)})
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    ctx = _ctx(tmp_path / "eps", rep)
    (u,) = ctx["unshown_cameras"]
    assert u["name"] == "/lidar/depth" and u["n_frames"] == 30 and "depth" in u["why"]


# ---------------------------------------------------------------- a take of many cameras

MANY = ["cam_front", "cam_wrist_left", "cam_wrist_right", "cam_side", "cam_back", "cam_top", "cam_low", "zed_left"]


def test_a_take_of_more_than_six_cameras_is_one_episode(tmp_path):
    """Eight colour cameras of one take had been split into eight episodes with no note, and the take's infrared and
    thermal videos then went with none. It is one episode: the cameras past those the model is shown go to the
    board as unshown cameras, and so do the infrared and thermal videos."""
    root = tmp_path / "up"
    for k, cam in enumerate(MANY + ["cam_infrared", "cam_thermal"]):
        _mp4(root / "ep1" / f"{cam}.mp4", 30, 20 * k)
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    assert len(rep["episodes"]) == 1, [e["name"] for e in rep["episodes"]]
    ctx = _ctx(tmp_path / "eps", rep)
    shown = [c["key"] for c in ctx["cameras"].values()]
    unshown = [u["name"] for u in ctx["unshown_cameras"]]
    assert sorted(shown + unshown) == sorted(MANY + ["cam_infrared", "cam_thermal"]), (shown, unshown)
    assert not any("left out" in u for u in rep["used"]), rep["used"]


def test_seven_cameras_of_one_take_in_a_flat_folder_are_one_episode(tmp_path):
    root = tmp_path / "up"
    for k, cam in enumerate(["front", "top", "side", "back", "wrist_left", "wrist_right", "head"]):
        _mp4(root / f"take1_{cam}.mp4", 30, 20 * k)
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    assert len(rep["episodes"]) == 1, [e["name"] for e in rep["episodes"]]


def test_a_folder_of_many_single_camera_episodes_stays_many_episodes(tmp_path):
    """The rule that tells a take from a folder of episodes: more than six videos are one take only when every name
    is a camera's and their lengths agree, as cameras of one take stop together. Videos of other lengths, or named
    for anything else, stay one episode each."""
    root = tmp_path / "up"
    for k, cam in enumerate(MANY):
        _mp4(root / "a" / f"{cam}.mp4", 20 + 40 * k)
    for k, task in enumerate(["pick", "place", "pour", "wipe", "stack", "open", "close", "push"]):
        _mp4(root / "b" / f"{task}.mp4", 30)
    det, items = f.plan(root)
    assert len(items) == 16, [it["name"] for it in items]


def test_an_infrared_video_with_no_take_goes_with_every_episode_of_its_folder(tmp_path):
    root = tmp_path / "up"
    for take in ("1", "2"):
        for k, cam in enumerate(("top", "wrist_left")):
            _mp4(root / f"{cam}_ep{take}.mp4", 30, 40 * k)
    _mp4(root / "infrared.mp4", 30)
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    assert len(rep["episodes"]) == 2
    for i in range(2):
        assert [u["name"] for u in _ctx(tmp_path / "eps", rep, i)["unshown_cameras"]] == ["infrared"]


def test_the_report_says_infrared_and_mask_videos_are_on_the_board(tmp_path):
    root = tmp_path / "up"
    for k, cam in enumerate(("top", "wrist_left", "cam_mask")):
        _mp4(root / "ep1" / f"{cam}.mp4", 30, 40 * k)
    det, items = f.plan(root)
    line = next(u for u in det["used"] if "mask" in u)
    assert "left out" not in line and "board" in line, line


# ---------------------------------------------------------------- a board clip that fails

def test_a_failed_cut_leaves_no_temporary_file(tmp_path, monkeypatch):
    import subprocess
    from board import clips

    def run(cmd, **kw):
        Path(cmd[-1]).write_bytes(b"part of a clip")
        raise subprocess.CalledProcessError(1, cmd)
    monkeypatch.setattr(clips, "source_size", lambda ffmpeg, path: (64, 48, False))
    monkeypatch.setattr(clips.subprocess, "run", run)
    out = tmp_path / "clips" / "episode_1.mp4"
    try:
        clips.extract_one(str(tmp_path / "src.mp4"), 0.0, 30, out, "ffmpeg", 1)
    except subprocess.CalledProcessError:
        pass
    assert not list(out.parent.glob("*.tmp.mp4"))
