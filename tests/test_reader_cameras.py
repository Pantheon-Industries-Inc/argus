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


def _lerobot_short_wrist(root: Path, n_main: int = 90, n_wrist: int = 30) -> None:
    """A LeRobot v2.1 episode of a moving 14 value state, one row per main camera frame, and a wrist camera whose video
    is shorter than the main camera's."""
    state = np.cumsum(np.random.default_rng(1).normal(0, 0.02, (n_main, 14)), axis=0)
    _lerobot(root, {0: {"observation.state": list(state), "action": list(state)}}, n_video=0,
             feats={"observation.images.cam_left_wrist": {"dtype": "video", "shape": [48, 64, 3]}})
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
    monkeypatch.setitem(pieces.PIECE_MAX_S, "teleop_arms", 1.0)
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


def test_the_edge_a_camera_may_miss_without_an_issue_is_at_most_a_tenth_of_the_episode(tmp_path):
    """A camera missing the last 0.2 s of a 1 s episode had raised no issue, since any edge under half a second was
    allowed. The edge a camera may miss is the smaller of half a second and a tenth of the episode (edge_slack), the
    rule a signal's edges follow, so it is flagged there and a camera missing 0.33 s of 20 s is not."""
    for secs, short, flagged in ((1.0, 24, True), (20.0, 590, False)):
        root = tmp_path / f"up{secs:g}"
        _mp4(root / "ep1" / "top.mp4", int(secs * 30))
        _mp4(root / "ep1" / "wrist_left.mp4", short)
        rep = f.convert(root, "teleop_arms", tmp_path / f"eps{secs:g}", "test", 900)
        assert bool(_issues(_ctx(tmp_path / f"eps{secs:g}", rep), "camera_short")) is flagged, secs


def test_a_camera_paired_by_time_that_ends_early_or_starts_late_is_named_as_an_unpaired_one_is(tmp_path):
    """A camera paired by capture time that stopped first had been described as having frames only between two times,
    in other words than a camera that is not paired. Both say the video ends before the episode does, or starts after
    it, and the instants it has no frame at."""
    from label import episode as me
    up = tmp_path / "up"
    files = {"exo": ("top", _mp4(up / "top.mp4", 60)), "left": ("wrist_left", _mp4(up / "wrist_left.mp4", 30)),
             "right": ("wrist_right", _mp4(up / "wrist_right.mp4", 30))}
    ep_dir = tmp_path / "eps" / "episode_a"
    f.video_views_episode(ep_dir, files, "teleop_arms", "probe", {},
                          real={"exo": np.arange(60) / 30, "left": np.arange(30) / 30, "right": 1.0 + np.arange(30) / 30})
    prompt = me.build_request(ep_dir)["prompt"]
    assert "left's video ends before the episode does, so it has no frame at 1.50 s, 1.97 s" in prompt.lower(), prompt
    assert "right's video starts after the episode does, so it has no frame at 0.00 s" in prompt.lower(), prompt
    assert "has frames only" not in prompt


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
    import pandas as pd
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    keys = ["observation.images.cam_high"] + (["observation.images.cam_left_wrist"] if wrist else [])
    feats = {**{k: {"dtype": "image", "shape": [72, 96, 3]} for k in keys},
             "observation.state": {"dtype": "float32", "shape": [14]}}
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
    assert ctx["placeholder_frames"] == {"exo": [[4, 4]]}, ctx.get("placeholder_frames")


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
    assert note.endswith("at that instant.")


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

    real = subprocess.run

    def run(cmd, **kw):
        if not str(cmd[-1]).endswith(".tmp.mp4"):
            return real(cmd, **kw)                 # the size probe runs as usual; the cut fails part way
        Path(cmd[-1]).write_bytes(b"part of a clip")
        raise subprocess.CalledProcessError(1, cmd)
    src = _mp4(tmp_path / "src.mp4", 30)
    monkeypatch.setattr(clips.subprocess, "run", run)
    out = tmp_path / "clips" / "episode_1.mp4"
    with pytest.raises(subprocess.CalledProcessError):
        clips.extract_one(str(src), 0.0, 30, out, clips.find_ffmpeg(), 1)
    assert not list(out.parent.glob("*.tmp.mp4"))
