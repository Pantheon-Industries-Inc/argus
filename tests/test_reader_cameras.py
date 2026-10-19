"""The reader keeps every camera that works (prepare/formats.py): a camera that cannot be opened, ends early, has
undecodable frames or a short depth stream is flagged on its episode and the rest of the episode is kept; a take of
many cameras is one episode, and every infrared, thermal or mask video is shown with its episode."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest

from prepare import formats as f
from test_formats import _issues, _jpeg, _lerobot, _png


from test_prepare import joint_names

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
    assert [i["camera"] for i in _issues(ctx, "unshown_camera_not_decodable")] == ["cam_mask"], ctx.get("reader_issues")


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
             feats={"observation.state": {"dtype": "float32", "shape": [14], "names": joint_names()},
                    "action": {"dtype": "float32", "shape": [14], "names": joint_names()},
                    "observation.images.cam_left_wrist": {"dtype": "video", "shape": [48, 64, 3]}})
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
    (short,) = _issues(ctx, "camera_coverage_partial")
    assert short["camera"] == "observation.images.cam_left_wrist"
    assert abs(short["t0_s"] - 0.3) < 0.05 and abs(short["t1_s"] - 1.0) < 0.05, short


def _lerobot_short_wrist(root: Path, n_main: int = 90, n_wrist: int = 30) -> None:
    """A LeRobot v2.1 episode of a moving 14 value state, one row per main camera frame, and a wrist camera whose video
    is shorter than the main camera's."""
    state = np.cumsum(np.random.default_rng(1).normal(0, 0.02, (n_main, 14)), axis=0)
    _lerobot(root, {0: {"observation.state": list(state), "action": list(state)}}, n_video=0,
             feats={"observation.state": {"dtype": "float32", "shape": [14], "names": joint_names()},
                    "action": {"dtype": "float32", "shape": [14], "names": joint_names()},
                    "observation.images.cam_left_wrist": {"dtype": "video", "shape": [48, 64, 3]}})
    videos = root / "videos" / "chunk-000"
    _mp4(videos / "observation.images.cam_high" / "episode_000000.mp4", n_main)
    _mp4(videos / "observation.images.cam_left_wrist" / "episode_000000.mp4", n_wrist)


def test_a_short_side_camera_never_cuts_what_is_labelled(tmp_path):
    """The labelling plan had taken the shortest camera's length, so a wrist camera of 1 s beside a 3 s main camera
    cut the request to 1 s and judged the state unusable, which dropped its recorded motion, while the prompt spoke of
    the whole episode. The request covers the main camera's frames with the state on them, and the wrist camera only
    loses its cells past its end, which the prompt names."""
    from label import episode as me
    root = tmp_path / "lr"
    _lerobot_short_wrist(root)
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    ep_dir = tmp_path / "eps" / rep["episodes"][0]["episode_id"]
    ep = me.load(ep_dir)
    pl = me.plan(ep)
    assert pl["n"] == 90 and pl["state_usable"], (pl["n"], pl["checks"])
    assert me.frame_time(ep, pl["ks"][-1]) > 2.5, pl["ks"]
    me.frames(ep, pl)
    assert ep["no_frame"]["left"] == {k for k in pl["ks"] if k >= 30}, ep["no_frame"]
    assert not ep["decode_failed"], ep["decode_failed"]
    req = me.build_request(ep_dir)
    assert "RECORDED MOTION" in req["prompt"]
    assert "Left's video ends before the episode does" in req["prompt"], req["prompt"]


def test_a_part_of_a_long_recording_past_a_short_cameras_end_has_no_frame_of_it(tmp_path, monkeypatch):
    """A part cut out of a long recording had given a camera that stopped before it a window of the anchor's length,
    so every instant read as a frame that could not be decoded. Each part keeps only the camera's own frames inside
    it: a part past its end has none, and its prompt says the video ended, not that it failed."""
    from label import episode as me
    from label import pieces
    monkeypatch.setattr(pieces, "piece_max", lambda ctx: 1.0)
    root = tmp_path / "lr"
    _lerobot_short_wrist(root)
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    parts = pieces.write_pieces(tmp_path / "eps" / rep["episodes"][0]["episode_id"], tmp_path / "pieces")
    assert len(parts) == 3
    windows = [int(json.loads((p / "sources.json").read_text())["left"]["n_frames"]) for p in parts]
    assert sum(windows) == 30 and windows[-1] == 0, windows
    ep = me.load(parts[-1])
    pl = me.plan(ep)
    me.frames(ep, pl)
    assert ep["no_frame"]["left"] == set(pl["ks"]) and not ep["decode_failed"], (ep["no_frame"], ep["decode_failed"])


def test_a_side_camera_that_ends_early_is_flagged(tmp_path):
    root = tmp_path / "up"
    _mp4(root / "ep1" / "top.mp4", 60)
    _mp4(root / "ep1" / "wrist_left.mp4", 30)
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    ctx = _ctx(tmp_path / "eps", rep)
    assert ctx["n_state_frames"] == 60
    (short,) = _issues(ctx, "camera_coverage_partial")
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
    assert not _issues(ctx, "camera_coverage_partial")


def test_the_edge_a_camera_may_miss_without_an_issue_is_at_most_a_tenth_of_the_episode(tmp_path):
    """A camera missing the last 0.2 s of a 1 s episode had raised no issue, since any edge under half a second was
    allowed. The edge a camera may miss is the smaller of half a second and a tenth of the episode (edge_slack), the
    rule a signal's edges follow, so it is flagged there and a camera missing 0.33 s of 20 s is not."""
    for secs, short, flagged in ((1.0, 24, True), (20.0, 590, False)):
        root = tmp_path / f"up{secs:g}"
        _mp4(root / "ep1" / "top.mp4", int(secs * 30))
        _mp4(root / "ep1" / "wrist_left.mp4", short)
        rep = f.convert(root, "teleop_arms", tmp_path / f"eps{secs:g}", "test", 900)
        assert bool(_issues(_ctx(tmp_path / f"eps{secs:g}", rep), "camera_coverage_partial")) is flagged, secs


def test_a_paired_camera_keeps_its_span_when_it_ends_early_or_starts_late(tmp_path):
    """A paired camera keeps its established span sentence, including both edges, so clean requests stay unchanged."""
    from label import episode as me
    up = tmp_path / "up"
    files = {"exo": ("top", _mp4(up / "top.mp4", 60)), "left": ("wrist_left", _mp4(up / "wrist_left.mp4", 30)),
             "right": ("wrist_right", _mp4(up / "wrist_right.mp4", 30))}
    ep_dir = tmp_path / "eps" / "episode_a"
    f.video_views_episode(ep_dir, files, "teleop_arms", "probe", {},
                          real={"exo": np.arange(60) / 30, "left": np.arange(30) / 30,
                                "right": 1.0 + np.arange(30) / 30})
    prompt = me.build_request(ep_dir)["prompt"]
    assert "left has frames only from 0.00 s to 0.97 s" in prompt.lower(), prompt
    assert "right has frames only from 1.00 s to 1.97 s" in prompt.lower(), prompt
    assert "instants outside that time" in prompt


def test_cameras_of_one_length_raise_no_issue(tmp_path):
    root = tmp_path / "up"
    _three_cameras(root)
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    assert not _issues(_ctx(tmp_path / "eps", rep))


# ---------------------------------------------------------------- a LeRobot camera with no video

def test_a_lerobot_chunk_size_that_is_not_a_whole_number_never_stops_the_conversion(tmp_path):
    """info.json chunks_size 0 had divided by zero and null had raised a type error, so the whole upload failed. The
    episodes are found by their file names, and the report says why the path templates were not used. A camera with
    no video under its own name is looked for through the template, which is where the error was raised."""
    for bad in (0, None, 2.5):
        root = tmp_path / f"lr{bad}"
        _lerobot(root, {0: {"observation.state": [np.zeros(14)] * 30}},
                 feats={"observation.images.cam_left_wrist": {"dtype": "video", "shape": [48, 64, 3]}})
        info = json.loads((root / "meta" / "info.json").read_text())
        info["chunks_size"] = bad
        (root / "meta" / "info.json").write_text(json.dumps(info))
        rep = f.convert(root, "teleop_arms", tmp_path / f"eps{bad}", "test", 900)
        assert len(rep["episodes"]) == 1, (bad, rep["failed"])
        assert any("chunks_size" in m for m in rep["missing"]), (bad, rep["missing"])


def test_a_lerobot_camera_with_no_video_for_the_episode_is_listed_with_its_reason(tmp_path):
    root = tmp_path / "lr"
    _lerobot(root, {0: {"observation.state": [np.zeros(14)] * 30}},
             feats={"observation.images.cam_left_wrist": {"dtype": "video", "shape": [48, 64, 3]}})
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    ctx = _ctx(tmp_path / "eps", rep)
    (u,) = [u for u in ctx["source"]["unused_cameras"] if u.startswith("observation.images.cam_left_wrist")]
    assert "(" in u and "video" in u, u


# ---------------------------------------------------------------- undecodable frames inside a camera

def _lerobot_images(root: Path, bad: tuple = (), n: int = 30, wrist: bool = False) -> None:
    """A LeRobot v2.1 episode of n frames whose scene camera (and, with wrist, a left wrist camera) is PNG images in
    the data file, with a 14 value state; the scene camera's images at the rows in bad do not decode."""
    from test_prepare import joint_names
    import pandas as pd
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    keys = ["observation.images.cam_high"] + (["observation.images.cam_left_wrist"] if wrist else [])
    feats = {**{k: {"dtype": "image", "shape": [72, 96, 3]} for k in keys},
             "observation.state": {"dtype": "float32", "shape": [14], "names": joint_names()}}
    (root / "meta" / "info.json").write_text(json.dumps({
        "codebase_version": "v2.1", "fps": 30, "chunks_size": 1000, "features": feats,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"}))
    cells = {k: [{"bytes": _png(np.full((72, 96, 3), (8 * r + 40 * i) % 256, np.uint8))
                  if i or r not in bad else b"\x89PNG\r\n\x1a\nbroken", "path": None} for r in range(n)]
             for i, k in enumerate(keys)}
    state = np.cumsum(np.random.default_rng(2).normal(0, 0.01, (n, 14)), axis=0)
    pd.DataFrame({**cells, "observation.state": list(state), "frame_index": np.arange(n),
                  "episode_index": np.zeros(n, int),
                  "timestamp": np.arange(n) / 30}).to_parquet(root / "data" / "chunk-000" / "episode_000000.parquet")


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


def test_embedded_images_and_state_use_the_same_sparse_row_order(tmp_path):
    import pandas as pd
    import av
    root = tmp_path / "lr"
    _lerobot_images(root, n=3)
    data = root / "data/chunk-000/episode_000000.parquet"
    df = pd.read_parquet(data)
    df["frame_index"] = [0, 2, 3]
    df["observation.state"] = [np.full(14, k, np.float32) for k in (10, 20, 30)]
    df.to_parquet(data)
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    assert not rep["failed"], rep
    ctx = _ctx(tmp_path / "eps", rep)
    ep = tmp_path / "eps" / ctx["episode_id"]
    np.testing.assert_array_equal(np.load(ep / "state.npz")["state"][:, 0], [10, 20, 30])
    with av.open(str(ep / "exo.mp4")) as clip:
        shades = [frame.to_ndarray(format="rgb24").mean() for frame in clip.decode(video=0)]
    assert len(shades) == 3 and shades[0] < shades[1] < shades[2]


def test_embedded_image_nonfinite_source_clock_uses_assumed_presentation(tmp_path):
    import pandas as pd
    root = tmp_path / "lr"
    _lerobot_images(root, n=12)
    data = root / "data/chunk-000/episode_000000.parquet"
    df = pd.read_parquet(data)
    df.loc[4, "timestamp"] = np.nan
    df.loc[7, "timestamp"] = np.inf
    df.to_parquet(data)
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    assert not rep["failed"], rep
    ctx = _ctx(tmp_path / "eps", rep)
    ep = tmp_path / "eps" / ctx["episode_id"]
    assert ctx["n_state_frames"] == 12 and ctx["state_kind"] == "none"
    assert ctx["state_why"] == "assumed_clock" and any(s["name"] == "recorded state" for s in ctx["signals"])
    assert _issues(ctx, "camera_timestamp_invalid")
    with np.load(ep / "recorded_container_times.npz") as raw:
        assert np.isnan(raw["timestamp"][4]) and np.isposinf(raw["timestamp"][7])
    with np.load(ep / "times.npz") as times:
        assert np.isnan(times["exo"][4]) and np.isposinf(times["exo"][7])


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
        o["qpos"].attrs["names"] = joint_names()
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
    assert ctx["placeholder_frames"] == {"exo": [[4, 4]]}, ctx.get("placeholder_frames")


def test_hdf_uint64_clock_reset_keeps_signed_capture_step(tmp_path):
    import h5py
    root = tmp_path / "h5"
    root.mkdir()
    stamps = np.array([1_700_000_000_000_000_000,
                       1_699_999_999_966_666_667,
                       1_699_999_999_983_333_334,
                       1_700_000_000_016_666_667], dtype=np.uint64)
    with h5py.File(root / "episode_0.hdf5", "w") as h:
        h.attrs["fps"] = 30
        h.create_dataset("timestamps", data=stamps).attrs["units"] = "ns"
        h.create_dataset("images/top", data=np.stack([
            np.full((48, 64, 3), 30 + 30 * k, np.uint8) for k in range(len(stamps))]))
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    assert not rep["failed"], rep
    ctx = _ctx(tmp_path / "eps", rep)
    ep = tmp_path / "eps" / ctx["episode_id"]
    with np.load(ep / "recorded_container_times.npz") as raw:
        np.testing.assert_array_equal(raw["timestamps"], stamps)
    with np.load(ep / "times.npz") as times:
        assert abs(times["exo"][1] + 1 / 30) < 1e-6
        assert abs(times["exo"][2] + 1 / 60) < 1e-6
    assert _issues(ctx, "camera_timestamp_invalid")


def test_a_placeholder_frame_is_never_shown_to_the_model_as_footage(tmp_path):
    """A black frame written where an image did not decode had been sent to the model as the camera's footage, with
    no word. The reader records it on the main camera's frames, and labelling treats it as a frame that did not
    decode: its grid cell is empty, the camera is left out of the detail view there, and the prompt names the instant,
    at a sampled instant and at the first frame alike."""
    from label import episode as me
    root = tmp_path / "lr"
    _lerobot_images(root, bad=(0, 45), n=90, wrist=True)
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    ctx = _ctx(tmp_path / "eps", rep)
    assert ctx["placeholder_frames"] == {"exo": [[0, 0], [45, 45]]}, ctx.get("placeholder_frames")
    ep_dir = tmp_path / "eps" / ctx["episode_id"]
    ep = me.load(ep_dir)
    pl = me.plan(ep)
    assert {0, 45} <= set(pl["ks"])
    imgs = me.frames(ep, pl)
    assert 0 not in imgs["exo"] and 45 not in imgs["exo"] and 0 in imgs["left"] and 45 in imgs["left"]
    assert not me.recording_at(ep, "exo", 0) and not me.recording_at(ep, "exo", 45)
    assert ep["decode_failed"] == {"exo": [0, 45]}, ep["decode_failed"]
    assert me.decode_failures(ep) == []        # flagged once, by the reader, over its stretch
    assert "Cam_high's video could not be decoded at 0.00 s, 1.50 s" in me.build_request(ep_dir)["prompt"]
    assert f.trim_episode(ep_dir, 1.0)["placeholder_frames"] == {"exo": [[0, 0]]}     # cut with the episode


def test_placeholder_frames_are_placed_on_the_main_cameras_frames_through_its_time_pairing():
    own = {"exo": [0], "left": [2]}
    kmaps = {"left": np.array([0, 1, 1, 2, 2, 3])}
    assert f.placeholder_frames(own, kmaps) == {"exo": [[0, 0]], "left": [[3, 4]]}


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
            w.add_message(ids[t], log_time=ns, publish_time=ns,
                          data=m if isinstance(m, bytes) else json.dumps(m).encode())
        w.finish()


def _jpg_msg(k: int) -> dict:
    return {"format": "jpeg", "data": base64.b64encode(_jpeg(8 * k % 256)).decode()}


def _depth_msg(k: int) -> dict:
    return {"width": 64, "height": 48, "encoding": "16UC1", "step": 128,
            "data": base64.b64encode(np.full((48, 64), 900 + 10 * k, np.uint16).tobytes()).decode()}


@pytest.mark.parametrize('rig,primary', [
    ('handheld_gripper', '/alpha/image_raw/compressed'),
    ('handheld_gripper', '/zeta/image_raw/compressed'),
    ('teleop_arms', '/alpha/image_raw/compressed'),
])
def test_mcap_presentation_camera_slots_do_not_invent_publisher_mount_claims(tmp_path, rig, primary):
    from label import episode as me
    root = tmp_path / 'upload'
    root.mkdir()
    other = '/other/image_raw/compressed'
    _json_mcap(root / 'capture.mcap', {primary: ('foxglove.CompressedImage', _jpg_msg),
                                     other: ('foxglove.CompressedImage', _jpg_msg)})
    report = f.convert(root, rig, tmp_path / 'episodes', 'native camera', 900)
    assert not report['failed']
    folder = tmp_path / 'episodes' / report['episodes'][0]['episode_id']
    ep = me.load(folder)
    camera = next(c for c in ep['context']['cameras'].values() if c['key'] == primary)
    assert camera['key'] == primary
    assert {c['key'] for c in ep['context']['cameras'].values()} == {primary, other}
    assert all(source['n_frames'] == 30 for source in ep['sources'].values())
    request = me.build_request(folder, inspect_evidence=False)
    prompt = '\n'.join(c.get('text', '') for c in request['content'])
    assert 'looking along its fingers' not in prompt
    assert 'a camera that is not mounted on any' not in prompt
    assert 'camera labels are presentation aliases' in prompt
    assert 'do not establish camera mounting' in prompt
    assert 'contradiction of a presentation alias is not a recording fault' in prompt
    assert 'camera_observations' in request['schema_fields']
    assert all(f'{view}: {me.cam_name(ep, view)}' in prompt for view in request['views'])
    # Legacy contexts must not turn the same generated description into a source claim.
    if rig == 'handheld_gripper':
        camera.pop('role_provenance', None)
        camera['name'] = 'gripper'
        camera['desc'] = 'the camera carried on the handheld gripper, looking along its fingers'
        camera.pop('mounting', None)
        legacy = me.camera_desc(ep)
        assert 'looking along its fingers' not in legacy
        assert 'do not establish camera mounting' in legacy


def test_recorded_mount_names_and_verified_adapter_descriptions_keep_viewpoint_guidance(tmp_path):
    from label import episode as me
    from prepare import fastumi
    root = tmp_path / 'upload'
    root.mkdir()
    _json_mcap(root / 'capture.mcap', {'/left_wrist/image_raw/compressed':
                                      ('foxglove.CompressedImage', _jpg_msg)})
    report = f.convert(root, 'teleop_arms', tmp_path / 'episodes', 'named wrist', 900)
    ep = me.load(tmp_path / 'episodes' / report['episodes'][0]['episode_id'])
    prompt = me.camera_desc(ep)
    assert 'camera mounted on the' in prompt
    assert 'recorded or verified camera-role claims' in prompt
    assert 'record it as a data issue' in prompt
    # Specialized adapters supply descriptions checked on native frames, beyond a generic slot.
    clip = _mp4(tmp_path / 'cam.mp4', 12)
    folder = tmp_path / 'verified'
    f.video_views_episode(folder, {'right': ('camera_rgb_image', clip)}, 'handheld_gripper', 'verified', {},
                          descs={'right': fastumi.CAMERA_DESC['gripper']})
    verified = me.load(folder)
    assert verified['context']['cameras']['right']['role_provenance'] == 'adapter'
    prompt = me.camera_desc(verified)
    assert fastumi.CAMERA_DESC['gripper'] in prompt
    assert 'recorded or verified camera-role claims' in prompt
    assert 'record it as a data issue' in prompt
    camera = verified['context']['cameras']['right']
    camera.update(role_provenance='human', mounting='fixed', desc='a fixed camera viewing the work bench')
    prompt = me.camera_desc(verified)
    assert 'a fixed camera viewing the work bench' in prompt
    assert 'No camera in this episode is off the gripper' not in prompt
    assert 'A mounted camera turns with its gripper' not in prompt
    for provenance, desc in [('adapter', 'Recorded camera channel /sensor/image; selected slots are presentation roles'),
                             ('adapter', 'the depth picture stored by the recorder as an ordinary image'),
                             ('recorded', None), ('human', None)]:
        camera.update(role_provenance=provenance)
        camera.pop('mounting', None)
        if desc:
            camera['desc'] = desc
        else:
            camera.pop('desc', None)
        prompt = me.camera_desc(verified)
        if desc:
            assert desc in prompt
        assert 'No camera in this episode is off the gripper' not in prompt
        assert 'A mounted camera turns with its gripper' not in prompt
        assert 'camera mounted on the handheld gripper' not in prompt
    camera.update(role_provenance='human', mounting='gripper')
    assert 'A mounted camera turns with its gripper' in me.camera_desc(verified)
    verified['context']['profile'] = 'ego_head'
    camera.pop('mounting')
    prompt = me.camera_desc(verified)
    assert "name the person's own left and right hands" in prompt
    verified['context']['profile'] = 'teleop_arms'
    camera.update(role_provenance='human', mounting='fixed', desc='a fixed camera viewing the work bench')
    verified['context']['cameras']['left'] = {**camera, 'name': 'left'}
    verified['sources']['left'] = dict(verified['sources']['right'])
    prompt = me.camera_desc(verified)
    assert "the left stream's own gripper makes the contact" not in prompt
    assert 'stream really sits on the side its name says' not in prompt


@pytest.mark.parametrize('rig', ['ego_head', 'handheld_gripper'])
def test_native_video_profile_intro_does_not_assert_mount_or_control(tmp_path, rig):
    from label import episode as me
    from label import prompts
    clip = _mp4(tmp_path / 'capture.mp4', 12)
    folder = tmp_path / 'native'
    view = 'exo' if rig == 'ego_head' else 'right'
    f.video_views_episode(folder, {view: ('capture', clip)}, rig, 'native', {})
    request = me.build_request(folder, inspect_evidence=True)
    intro = me._intro_head(me.load(folder))
    shared = prompts.what_this_is(rig)
    for text in (intro, shared, request['prompt']):
        assert 'a person wears a camera on their head' not in text
        assert 'a person wearing a head camera' not in text
        assert 'head camera points wherever' not in text
        assert 'a person holds the handheld gripper' not in text
        assert 'one per hand' not in text
        assert 'never a task object' not in text
    assert 'camera_observations' not in request['schema_fields']
    assert 'sensor_findings' not in request['schema_fields']
    assert request['evidence_access'] is None
    if rig == 'handheld_gripper':
        assert 'ownership' in shared
        assert 'gripper-only execution contract' in shared
        assert 'explicit source task or actor rules' in shared
        assert 'has only the gripper' not in shared
        assert 'other gripper' in shared and 'fingers' in shared and 'handle or strap' in shared
    else:
        assert 'wearer' not in shared
        assert "person's own left and right hands" in request['prompt']


def test_camera_observations_use_only_views_with_actual_supplied_native_frames(tmp_path):
    from label import episode as me
    root = tmp_path / 'upload'
    root.mkdir()
    _json_mcap(root / 'capture.mcap', {'/alpha/image': ('foxglove.CompressedImage', _jpg_msg),
                                     '/other/image': ('foxglove.CompressedImage', _jpg_msg)})
    report = f.convert(root, 'handheld_gripper', tmp_path / 'episodes', 'native camera', 900)
    folder = tmp_path / 'episodes' / report['episodes'][0]['episode_id']
    sources = json.loads((folder / 'sources.json').read_text())
    assert set(sources) == {'right', 'extra1'}
    Path(sources['extra1']['packed']).write_bytes(b'corrupted after conversion')
    request = me.build_request(folder, inspect_evidence=False)
    assert request['views'] == ['right', 'extra1']
    assert request['camera_observation_views'] == ['right']
    assert 'camera_observations' not in request['schema_fields']
    assert any(issue['camera'] == 'extra1' for issue in request['decode_failed'])


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


def test_depth_that_ends_early_is_not_also_called_offset_from_its_camera(tmp_path):
    """The depth check had measured the gap to a colour frame's depth frame at frames with no depth reading too
    (depth_kmap -1, which picked the stream's last frame), so depth that ends early was also reported as depth far
    from its colour frames. Only frames with a depth reading are measured; the missing stretch is depth_partial."""
    from checks import sensors
    from label import episode as me
    d = tmp_path / "up" / "ep1"
    _mp4(d / "exo_cam-images-rgb.mp4", 60)
    dw = f.DepthWriter(d / "exo_cam-images-depth.mkv")
    for k in range(20):
        dw.add(k / 30, np.full((48, 64), 800 + k, np.uint16), 0.001)
    dw.close()
    rep = f.convert(tmp_path / "up", "teleop_arms", tmp_path / "eps", "test", 900)
    ep = me.load(tmp_path / "eps" / rep["episodes"][0]["episode_id"])
    assert not [x for x in sensors.depth_findings(ep) if x["check"] == "depth_offset"]


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
    assert dp.load(ep)['exo']['paired_camera'] == 'exo'
    assert 'unpaired depth' not in note and 'assumed presentation times' not in note


def test_depth_that_misses_only_an_edge_within_the_slack_reads_no_issue(tmp_path):
    """Depth that stops three frames before its 2 s colour camera has no reading for the last two frames, and the
    prompt says so, but that tail is within the edge a stream may miss (edge_slack), so it is no data issue."""
    from label import depth as dp
    from label import episode as me
    d = tmp_path / "up" / "ep1"
    _mp4(d / "exo_cam-images-rgb.mp4", 60)
    dw = f.DepthWriter(d / "exo_cam-images-depth.mkv")
    for k in range(57):
        dw.add(k / 30, np.full((48, 64), 800 + k, np.uint16), 0.001)
    dw.close()
    rep = f.convert(tmp_path / "up", "teleop_arms", tmp_path / "eps", "test", 900)
    ctx = _ctx(tmp_path / "eps", rep)
    ep = tmp_path / "eps" / ctx["episode_id"]
    assert (np.load(ep / "depth_kmap_exo.npy")[-2:] < 0).all()
    assert not _issues(ctx, "depth_partial")
    assert "no depth" in me._depth_note({**me.load(ep), "depth": dp.load(ep)})


# ---------------------------------------------------------------- MCAP depth topics

@pytest.mark.parametrize('variant', [
    'registered', 'one_rgb', 'no_distortion', 'distortion', 'projection', 'rectification', 'crop', 'binning',
    'missing_calibration', 'changing_calibration', 'different_frame', 'different_clock',
    'malformed_camera_info', 'processed_candidate', 'nonidentity_mapping', 'bad_distortion_count',
])
def test_mcap_registration_uses_recorded_geometry_and_capture_clock(tmp_path, variant):
    """One recorded camera has raw/compressed aliases, infrared and a separate depth frame."""
    from label import depth as dp, episode as me, evidence_access as ea
    root = tmp_path / 'upload'
    root.mkdir()
    def header(row, frame='color_optical', offset=0):
        return {'frame_id': frame, 'stamp': {'sec': 1_790_000_000 + offset,
                                            'nanosec': int(row * 1e9 / 30)}}
    def info(row, frame='color_optical', color=False):
        value = {'header': header(row, frame), 'width': 64, 'height': 48,
                 'distortion_model': 'plumb_bob', 'd': [0., 0., 0., 0., 0.],
                 'k': [52., 0., 32., 0., 52., 24., 0., 0., 1.],
                 'r': [1., 0., 0., 0., 1., 0., 0., 0., 1.],
                 'p': [52., 0., 32., 0., 0., 52., 24., 0., 0., 0., 1., 0.],
                 'binning_x': 0, 'binning_y': 0,
                 'roi': {'x_offset': 0, 'y_offset': 0, 'height': 0, 'width': 0, 'do_rectify': False}}
        if frame == 'depth_optical':
            value['k'][0] = value['p'][0] = 40.
        if variant == 'no_distortion':
            value['distortion_model'], value['d'] = 'none', []
        if variant == 'nonidentity_mapping':
            value['d'][0] = .1
        if variant == 'bad_distortion_count':
            value['d'] = [0.]
        if color:
            if variant == 'distortion': value['d'][0] = .1
            if variant == 'projection': value['p'][3] = .1
            if variant == 'rectification': value['r'] = [0., -1., 0., 1., 0., 0., 0., 0., 1.]
            if variant == 'crop': value['roi']['x_offset'] = 1
            if variant == 'binning': value['binning_x'] = 2
            if variant == 'missing_calibration': value.pop('d')
        if variant == 'changing_calibration' and row >= 15:
            value['k'][0] += 1
        return value
    def color(row):
        value = {**_jpg_msg(row), 'header': header(row)}
        if variant == 'different_clock':
            value['header']['stamp']['nanosec'] += 1_000_000
        return value
    def infrared(row):
        value = color(row)
        value['header']['frame_id'] = 'infra_optical'
        return value
    def raw_color(row):
        value = color(row)
        value.pop('format')
        value.update(width=64, height=48, step=192, encoding='rgb8', is_bigendian=0,
                     data=base64.b64encode(np.full((48, 64, 3), 8 * row % 256, np.uint8).tobytes()).decode())
        return value
    def aligned(row):
        return {**_depth_msg(row), 'header': header(row, 'depth_optical' if variant == 'different_frame'
                                                   else 'color_optical')}
    aligned_topic = '/sensor/z_aligned_depth/image_raw' if variant == 'one_rgb' else '/sensor/aligned_depth/image_raw'
    channels = {
        '/sensor/color/image_raw': ('sensor_msgs/msg/Image', raw_color),
        '/sensor/color/image_raw/compressed': ('sensor_msgs/msg/CompressedImage', color),
        '/sensor/infra/image': ('sensor_msgs/msg/CompressedImage', infrared),
        aligned_topic: ('sensor_msgs/msg/Image', aligned),
        '/sensor/depth/image_raw': ('sensor_msgs/msg/Image',
                                  lambda row: {**_depth_msg(row), 'header': header(row, 'depth_optical')}),
        '/sensor/color/camera_info': ('sensor_msgs/msg/CameraInfo', lambda row:
                                     b'{' if variant == 'malformed_camera_info' and row == 15
                                     else info(row, color=True)),
        aligned_topic.rsplit('/', 1)[0] + '/camera_info': ('sensor_msgs/msg/CameraInfo', info),
        '/sensor/depth/camera_info': ('sensor_msgs/msg/CameraInfo', lambda row: info(row, 'depth_optical')),
    }
    if variant == 'processed_candidate':
        channels['/sensor/color/image_rect'] = ('sensor_msgs/msg/CompressedImage', color)
    if variant == 'one_rgb':
        channels.pop('/sensor/color/image_raw/compressed')
        channels.pop('/sensor/infra/image')
    _json_mcap(root / 'capture.mcap', channels)
    report = f.convert(root, 'handheld_gripper', tmp_path / 'episodes', 'recorded geometry', 900)
    assert not report['failed']
    ep = me.load(tmp_path / 'episodes' / report['episodes'][0]['episode_id'])
    assert all(source['n_frames'] == 30 for source in ep['sources'].values())
    all_depth = {**ep['depth'], **ep['additional_depth']}
    by_source = {entry['source']: (view, entry) for view, entry in all_depth.items()}
    assert len(all_depth) == 2
    view, entry = by_source[aligned_topic]
    raw_view = next(v for v, camera in ep['context']['cameras'].items()
                    if camera['key'] == '/sensor/color/image_raw')
    if variant in ('registered', 'one_rgb', 'no_distortion'):
        assert entry['paired_camera'] == raw_view
        proof = entry['calibration']['pixel_registration']
        assert proof['rgb_topics'] == (['/sensor/color/image_raw'] if variant == 'one_rgb' else
                                      ['/sensor/color/image_raw', '/sensor/color/image_raw/compressed'])
        assert proof['frame_id'] == 'color_optical' and proof['matched_capture_count'] == 30
        assert proof['camera_info']['roi']['do_rectify'] is False
        assert ep['context']['depth'][view]['calibration'] == entry['calibration']
        access = ea.Access(ep)
        field = next(field for field in access.inventory()
                     if field['kind'] == 'depth' and field['descriptor']['source'] == entry['source'])
        assert field['descriptor']['calibration'] == entry['calibration']
    else:
        assert entry['paired_camera'] is None
        assert entry['calibration']['pixel_registration']['status'] == 'unverified'
        if variant == 'malformed_camera_info':
            assert _issues(ep['context'], 'camera_calibration_unreadable')
    native_view, native = by_source['/sensor/depth/image_raw']
    assert native['paired_camera'] is None
    assert all_depth[view]['scale_m'] == native['scale_m'] == .001
    for source_view in (view, native_view):
        pixels = dp.at_anchor(ep, all_depth, source_view, [0, 29])
        assert int(pixels[0][0, 0]) == 900 and int(pixels[29][0, 0]) == 1190
    # Source headers, clocks and calibration remain independently inspectable.
    retained = ep['context']['mcap_field_inventory']
    assert any(ch['topic'] == '/sensor/color/camera_info' for ch in retained)
    camera_info = next(ch for ch in retained if ch['topic'] == '/sensor/color/camera_info')
    assert any(field['field'] == 'header.frame_id' for field in camera_info['exact_numeric_fields'])


def test_an_mcap_whose_only_camera_is_depth_is_labelled_from_its_depth_picture(tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    _json_mcap(root / "run.mcap", {
        "/camera/front/depth": ("foxglove.RawImage", _depth_msg),
        "/right/joint_states": ("sensor_msgs/msg/JointState", lambda k: {
            "name": joint_names(("right",)), "position": [0.01 * k] * 6 + [0.5]}),
    })
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    assert not rep["failed"] and len(rep["episodes"]) == 1, rep
    ctx = _ctx(tmp_path / "eps", rep)
    assert [c["key"] for c in ctx["cameras"].values()] == ["/camera/front/depth"]
    assert [i["camera"] for i in _issues(ctx, "camera_not_colour")] == ["/camera/front/depth"]
    assert ctx["state_kind"] == "joints" and ctx["n_state_frames"] == 30
    assert f.probe(tmp_path / "eps" / ctx["episode_id"] / "exo.mp4")["width"] == 64
    from label import depth as dp, episode as me
    ep = me.load(tmp_path / 'eps' / ctx['episode_id'])
    pixels = dp.at_anchor(ep, ep['depth'], 'exo', [0, 29])
    assert pixels[0].dtype == np.uint16
    assert int(pixels[29][0, 0]) == 1190
    # ROS compressed float depth stores inverse readings plus reconstruction parameters.
    import io, struct
    from PIL import Image
    compressed = tmp_path / 'compressed_depth'
    compressed.mkdir()
    def compressed_depth(row):
        inverse = np.full((48, 64), 900 + 10 * row, np.uint16)
        inverse[0, 0] = 0
        buffer = io.BytesIO()
        Image.fromarray(inverse).save(buffer, format='PNG')
        data = struct.pack('<iff', 0, 10100.0, -1009.0) + buffer.getvalue()
        return {'format': '32FC1; compressedDepth png', 'data': base64.b64encode(data).decode()}
    _json_mcap(compressed / 'run.mcap', {'/camera/front/depth': ('foxglove.CompressedImage', compressed_depth)})
    report = f.convert(compressed, 'ego_head', tmp_path / 'compressed_eps', 'test', 900)
    ep = me.load(tmp_path / 'compressed_eps' / report['episodes'][0]['episode_id'])
    decoded = dp.at_anchor(ep, ep['depth'], 'exo', [0, 29])
    assert ep['depth']['exo']['scale_m'] == 1.0 and decoded[0].dtype == np.float32
    assert np.isnan(decoded[0][0, 0])
    assert decoded[29][1, 1] == np.float32(10100.0) / np.float32(1190 + 1009)
    assert decoded[29][1, 1] < decoded[0][1, 1]


def test_an_mcap_depth_topic_with_no_camera_of_its_own_goes_to_the_board(tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    _json_mcap(root / "run.mcap", {"/front/image": ("foxglove.CompressedImage", _jpg_msg),
                                   "/lidar/depth": ("foxglove.RawImage", _depth_msg)})
    rep = f.convert(root, "teleop_arms", tmp_path / "eps", "test", 900)
    ctx = _ctx(tmp_path / "eps", rep)
    (u,) = ctx["unshown_cameras"]
    assert u["name"] == "/lidar/depth" and u["n_frames"] == 30 and "depth" in u["why"]
    from label import episode as me, evidence_access as ea
    ep = me.load(tmp_path / 'eps' / ctx['episode_id'])
    assert not ep['depth'] and len(ep['additional_depth']) == 1
    access = ea.Access(ep)
    field = next(row for row in access.inventory() if row['kind'] == 'depth')
    receipt = access.inspect({'field_id': field['id'], 'mode': 'regions'})
    from prepare.signal_alignment import RECORDER_ARRIVAL_LIMITATION
    assert receipt['presentation_clock'] == ep['context']['source']['presentation_clock']
    assert RECORDER_ARRIVAL_LIMITATION in receipt['limitations']
    assert receipt['values'][0] == [.9] and receipt['values'][-1] == [1.19]
    assert any('No recorded RGB camera pairing' in line for line in receipt['limitations'])
    assert ep['additional_depth'][field['name']]['paired_camera'] is None
    # Mixed capture availability retains its native values while common recorder logs place source rows.
    other = tmp_path / 'mixed_clock'
    other.mkdir()
    def captured_depth(row):
        return {**_depth_msg(row), 'timestamp': {'sec': 1_790_000_000, 'nanosec': int(row * 1e9 / 30)}}
    _json_mcap(other / 'run.mcap', {'/front/image': ('foxglove.CompressedImage', _jpg_msg),
                                  '/lidar/depth': ('foxglove.RawImage', captured_depth)})
    report = f.convert(other, 'teleop_arms', tmp_path / 'mixed_eps', 'test', 900)
    mixed = me.load(tmp_path / 'mixed_eps' / report['episodes'][0]['episode_id'])
    mixed_access = ea.Access(mixed)
    mixed_field = next(row for row in mixed_access.inventory() if row['kind'] == 'depth')
    inspected = mixed_access.inspect({'field_id': mixed_field['id'], 'mode': 'regions'})
    assert inspected['presentation_clock'] == mixed['context']['source']['presentation_clock']
    assert RECORDER_ARRIVAL_LIMITATION in inspected['limitations']
    assert inspected['values'][0] == [.9] and inspected['values'][-1] == [1.19]
    assert len(inspected['source_frames']) == 30
    assert mixed['context']['source']['presentation_clock']['basis'] == 'mcap_log_time'
    assert mixed['context']['source']['presentation_clock']['physical_synchronization_verified'] is False
    assert mixed['additional_depth'][mixed_field['name']]['paired_camera'] is None
    assert any('No recorded RGB camera pairing' in line for line in inspected['limitations'])
    from mcap.reader import make_reader
    from label import depth as dp
    with (other / 'run.mcap').open('rb') as stream:
        source = list(make_reader(stream).iter_messages(topics=['/lidar/depth']))
    with (other / 'run.mcap').open('rb') as stream:
        origin = next(make_reader(stream).iter_messages(topics=['/front/image']))[2].log_time
    log_ns = np.asarray([message.log_time for _, _, message in source], dtype=np.int64)
    bodies = [json.loads(message.data) for _, _, message in source]
    capture_ns = np.asarray([body['timestamp']['sec'] * 10**9 + body['timestamp']['nanosec'] for body in bodies],
                            dtype=np.int64)
    np.testing.assert_allclose(inspected['times_s'], (log_ns - origin) / 1e9, atol=1e-9)
    assert [row['source_frame'] for row in inspected['source_frames']] == list(range(30))
    with np.load(mixed['dir'] / mixed['context']['recorded_camera_clocks']) as clocks:
        np.testing.assert_array_equal(clocks['depth_' + mixed_field['name'] + '_capture_ns'], capture_ns)
        np.testing.assert_array_equal(clocks['depth_' + mixed_field['name'] + '_log_ns'], log_ns)
    entry = mixed['additional_depth'][mixed_field['name']]
    decoded = dp.decode(entry, entry['pts'], list(range(30)))
    assert len(decoded) == 30
    for row, picture in decoded.items():
        raw = np.frombuffer(base64.b64decode(bodies[row]['data']), dtype='<u2').reshape(48, 64)
        np.testing.assert_array_equal(picture, raw)


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


def test_an_infrared_video_naming_a_take_no_episode_has_goes_with_none_and_is_named(tmp_path):
    """An infrared video whose name gives take 3 beside the episodes of takes 1 and 2 had gone with both of them, and
    one whose name gives take 2 beside a folder's one episode, whose own videos give no take, with that one. Its
    take says it belongs to no episode there: it goes with none, and the report names it with the reason."""
    for files, ir in (({"top_ep1", "wrist_left_ep1", "top_ep2", "wrist_left_ep2"}, "infrared_ep3"),
                      ({"top", "wrist_left"}, "ir_2")):
        root = tmp_path / ir / "up"
        for k, stem in enumerate(sorted(files)):
            _mp4(root / f"{stem}.mp4", 30, 40 * k)
        _mp4(root / f"{ir}.mp4", 30)
        det, items = f.plan(root)
        assert items and not any(it["unshown"] for it in items), [(it["name"], it["unshown"]) for it in items]
        assert any(f"{ir}.mp4" in m and "no episode" in m for m in det["missing"]), det["missing"]
        assert not any("infrared" in u and "board" in u for u in det["used"]), det["used"]


def test_a_lone_infrared_video_beside_colour_episodes_is_shown_on_their_board_not_labelled(tmp_path):
    """An infrared video alone in a folder of its own, in an upload whose episodes are colour, had become an episode
    the model was shown as footage. It goes to the board with the episodes of the nearest folder above it, named as
    not shown to the model; an upload with no colour video at all is still labelled from its infrared one."""
    root = tmp_path / "up"
    for k, cam in enumerate(("top", "wrist_left")):
        _mp4(root / "ep1" / f"{cam}.mp4", 30, 40 * k)
    _mp4(root / "extras" / "cam_infrared.mp4", 30)
    det, items = f.plan(root)
    assert [(it["name"], [p.name for p in it["unshown"]]) for it in items] == [("ep1", ["cam_infrared.mp4"])]
    only = tmp_path / "ir_only"
    _mp4(only / "ep1" / "cam_infrared.mp4", 30)
    assert [list(it["cams"].values()) for it in f.plan(only)[1]] == [[only / "ep1" / "cam_infrared.mp4"]]


def test_the_report_says_infrared_and_mask_videos_are_on_the_board(tmp_path):
    root = tmp_path / "up"
    for k, cam in enumerate(("top", "wrist_left", "cam_mask")):
        _mp4(root / "ep1" / f"{cam}.mp4", 30, 40 * k)
    det, items = f.plan(root)
    line = next(u for u in det["used"] if "mask" in u)
    assert "left out" not in line and "board" in line, line


# ---------------------------------------------------------------- a board clip that fails

def test_a_failed_cut_leaves_no_temporary_file(tmp_path, monkeypatch):
    """An ffmpeg cut that fails after writing part of its clip had left the half written file in the clips folder."""
    import subprocess

    import pytest
    from board import clips

    def run(cmd, **kw):
        assert str(cmd[-1]).endswith(".tmp.mp4")    # the size probe runs as usual; the cut fails part way
        Path(cmd[-1]).write_bytes(b"part of a clip")
        raise subprocess.CalledProcessError(1, cmd)
    src = _mp4(tmp_path / "src.mp4", 30)
    monkeypatch.setattr(clips, "run_while_progressing", run)
    out = tmp_path / "clips" / "episode_1.mp4"
    with pytest.raises(subprocess.CalledProcessError):
        clips.extract_one(str(src), 0.0, 30, out, clips.find_ffmpeg(), 1)
    assert not list(out.parent.glob("*.tmp.mp4"))
