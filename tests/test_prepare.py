"""The prepare stage on synthetic inputs (no network): list parsing, episode naming, the shared command line,
the sidecar writers on small mp4s written here, the lerobot and videos commands for your own data (a LeRobot v2.1 folder and a
folder of videos) through to a harness request, and the adapters' pure functions (naming, samplers)."""
from __future__ import annotations

import contextlib
import io
import json
import sys
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from label import episode as me
from prepare import cli
from prepare import egocentric100k
from prepare import fastumi
from prepare import galaxea
from prepare import habit
from prepare import lerobot
from prepare import molmo
from prepare import openaoe
from prepare import realomin
from prepare import formats
from prepare import videos

av = pytest.importorskip("av")
FPS = 30
W, H = 64, 36


def _mp4(path, n_frames: int, pts=None, shade: int = 0) -> list[int]:
    """mp4 (mpeg4, 64x36) whose frame k is a flat grey; pts default k*512 on a 1/15360 time base (30 fps). Given
    pts must be multiples of 512: the encoder rounds them to its 1/30 s frame grid.
    Returns the pts written."""
    pts = list(pts) if pts is not None else [k * 512 for k in range(n_frames)]
    path.parent.mkdir(parents=True, exist_ok=True)
    c = av.open(str(path), "w")
    s = c.add_stream("mpeg4", rate=FPS)
    s.width, s.height, s.pix_fmt = W, H, "yuv420p"
    s.time_base = Fraction(1, 15360)
    s.codec_context.gop_size = 5
    for k, p in enumerate(pts):
        fr = av.VideoFrame.from_ndarray(np.full((H, W, 3), (shade + k * 7) % 256, np.uint8), format="rgb24")
        fr.pts, fr.time_base = p, s.time_base
        for pkt in s.encode(fr):
            c.mux(pkt)
    for pkt in s.encode():
        c.mux(pkt)
    c.close()
    return pts


def _main(module, argv: list[str]) -> tuple[int, str]:
    """Run an adapter's main() with argv; (exit code, stdout)."""
    out = io.StringIO()
    old = sys.argv
    sys.argv = [f"python -m prepare {module.__name__.rsplit('.', 1)[-1]}", *map(str, argv)]
    try:
        with contextlib.redirect_stdout(out):
            try:
                rc = module.main()
            except SystemExit as e:
                rc = e.code
    finally:
        sys.argv = old
    return rc, out.getvalue()


# ---- the shared command line ----

def test_read_list_skips_comments_and_blanks_and_keeps_spaces_inside_a_line(tmp_path):
    p = tmp_path / "list.txt"
    p.write_text("# header line\n\n  31  \nClutter Tidy-Up [Stage2]/00001/01751.mcap\n# another\n"
                 "factory004/worker025/part268.tar factory_004_worker_025_0045.mp4\n")
    assert cli.read_list(p) == ["31", "Clutter Tidy-Up [Stage2]/00001/01751.mcap",
                                "factory004/worker025/part268.tar factory_004_worker_025_0045.mp4"]


def test_stamp_names_the_adapter_and_commit_in_the_episodes_this_run_wrote(tmp_path):
    import os
    import time
    for name, src in (("old", {}), ("new", {"format": "lerobot v2.1"}), ("upload", {"adapter": "habit"})):
        (tmp_path / f"episode_{name}").mkdir()
        (tmp_path / f"episode_{name}" / "context.json").write_text(json.dumps({"source": src}))
    since = time.time()
    os.utime(tmp_path / "episode_old" / "context.json", (since - 60, since - 60))
    assert cli.stamp(tmp_path, "lerobot", since - 1) == 2
    read = lambda n: json.loads((tmp_path / f"episode_{n}" / "context.json").read_text())["source"]
    assert read("old") == {}
    assert read("new")["adapter"] == "lerobot" and read("new")["format"] == "lerobot v2.1"
    assert read("upload")["adapter"] == "habit"
    assert read("new")["adapter_commit"] == read("upload")["adapter_commit"] == cli.commit() != ""


def test_run_counts_ok_skip_and_failed_and_fails_the_exit_code(capsys):
    def one(x):
        if x == 3:
            raise ValueError("broken stream")
        return "skip" if x == 2 else "ok"
    assert cli.run([1, 2, 3, 4], one, jobs=2) == 1
    captured = capsys.readouterr()
    assert '"ok": 2, "skip": 1, "failed": 1' in captured.out
    assert "FAILED 3: ValueError: broken stream" in captured.err
    assert cli.run([1, 2], one, jobs=1) == 0


@pytest.mark.parametrize("name,downloads,sampler", [
    ("molmo", True, True), ("abc130k", True, True), ("galaxea", True, True), ("habit", True, False),
    ("fastumi", True, True), ("realomin", True, True), ("egocentric100k", True, True),
    ("genhumanego", True, False), ("openaoe", True, False), ("lerobot", False, False), ("videos", False, False)])
def test_every_adapter_has_the_same_prepare_flags(name, downloads, sampler):
    import importlib
    mod = importlib.import_module(f"prepare.{name}")
    rc, text = _main(mod, ["prepare", "--help"])
    assert rc == 0
    for flag in ("--episodes LIST", "--out FOLDER", "--jobs N", "--force"):
        assert flag in text, (name, flag)
    assert ("--raw RAW" in text) == downloads
    assert (_main(mod, ["sample", "--help"])[0] == 0) == sampler


# ---- episode naming ----

def test_episode_name_cleans_to_letters_digits_and_single_underscores():
    assert formats.episode_name("raw_019fd620-0759-42c1_seg_1") == "episode_raw_019fd620_0759_42c1_seg_1"
    assert formats.episode_name("000031") == "episode_000031"
    assert formats.episode_name("episode_7") == "episode_7"
    assert formats.episode_name("--") == "episode_0"
    assert len(formats.episode_name("x" * 500)) == len("episode_") + 120


def test_episode_dirs_suffixes_collisions_in_list_order_and_is_stable(tmp_path):
    names = ["run-1", "run_1", "other", "run 1"]
    dirs = formats.episode_dirs(tmp_path, names)
    assert [d.name for d in dirs] == ["episode_run_1", "episode_run_1_2", "episode_other", "episode_run_1_3"]
    for d in dirs:
        d.mkdir()
    assert formats.episode_dirs(tmp_path, names) == dirs     # a rerun maps every source to the same folder


def test_adapter_episode_folder_names():
    assert galaxea.episode_dir_name("Plug_Into_A_Fixed_Socket_20250710_006", 69) == \
        "episode_Plug_Into_A_Fixed_Socket_20250710_006_000069"
    assert fastumi.episode_dir_name("single_arm/open_toilet_lid/1638") == "episode_single_arm__open_toilet_lid__001638"
    assert realomin.episode_dir_name("Clutter Tidy-Up [Stage2]/00001/01751.mcap") == \
        "episode_Clutter_Tidy_Up_Stage2_00001_01751"
    assert realomin.task_of("Clutter Tidy-Up [Stage2]/00001/01751.mcap") == "Clutter Tidy-Up [Stage2]"


# ---- camera assignment and pairing ----

def test_cameras_are_assigned_to_views_by_name():
    views, unused = formats.assign_views(["observation.images.cam_high", "observation.images.cam_left_wrist",
                                          "observation.images.cam_right_wrist", "observation.images.cam_low"],
                                         "teleop_arms")
    assert views == {"left": "observation.images.cam_left_wrist", "right": "observation.images.cam_right_wrist",
                     "exo": "observation.images.cam_high", "extra1": "observation.images.cam_low"}
    assert unused == []
    # a fixed camera that carries a side is not a wrist camera
    assert formats.mounted_side("exterior_image_1_left") is None and formats.mounted_side("leftWrist") == "left"
    # the only camera of a single handheld gripper is the gripper's own
    assert formats.assign_views(["observation.images.camera"], "handheld_gripper")[0] == \
        {"right": "observation.images.camera"}
    assert formats.pick_cameras(["cam_side", "cam_head"], "ego_head") == ({"exo": "cam_head"}, ["cam_side"])


def test_nearest_pairs_each_query_with_the_closest_time_earlier_on_a_tie():
    src = np.array([0, 10, 20, 30])
    assert formats.nearest(src, np.array([-5, 4, 5, 6, 26, 99])).tolist() == [0, 0, 0, 1, 3, 3]
    assert formats.nearest(np.array([7]), np.array([0, 100])).tolist() == [0, 0]
    assert formats.nearest(src, np.array([1])).dtype == np.int32


def test_state_layout_reads_seven_values_per_actor_else_video_only():
    assert formats.state_layout(14, "teleop_arms") == ("joints", None)
    assert formats.state_layout(7, "handheld_gripper") == ("ee_pose", None)
    kind, note = formats.state_layout(9, "teleop_arms")
    assert kind == "none" and "9 values" in note
    assert formats.state_layout(14, "ego_head") == ("none", None)


# ---- the sidecar writer on a real mp4 ----

def test_video_views_episode_times_frames_by_their_pts(tmp_path):
    pts = [0, 512, 1024, 2048, 2560, 3072]          # one frame missing after the third
    _mp4(tmp_path / "v" / "a.mp4", len(pts), pts)
    ep = tmp_path / "out" / "episode_a"
    ctx = formats.video_views_episode(ep, {"exo": ("a", tmp_path / "v" / "a.mp4")}, "ego_head", "mine",
                                      {"instruction": None})
    t = np.load(ep / "times.npz")
    assert t["exo_pts"].tolist() == pts
    assert np.allclose(t["exo"], np.array(pts) / 15360)
    src = json.loads((ep / "sources.json").read_text())
    assert src == {"exo": {"packed": str((tmp_path / "v" / "a.mp4").resolve()), "base_s": 0.0, "n_frames": 6,
                           "camera_key": "a"}}
    assert json.loads((ep / "context.json").read_text()) == ctx
    assert ctx["real_times"] == "times.npz" and ctx["fps"] == 30.0 and ctx["n_state_frames"] == 6
    assert ctx["duration_s"] == round(3072 / 15360 + 1 / 30, 3)
    assert ctx["cameras"]["exo"]["name"] == "head" and ctx["cameras"]["exo"]["desc"] == formats.EGO_DESC
    assert not (ep / "state.npz").exists() and (ep / "instruction.txt").read_text() == "\n"


def test_video_views_episode_pairs_a_second_camera_by_time(tmp_path):
    _mp4(tmp_path / "top.mp4", 6)
    _mp4(tmp_path / "wrist.mp4", 3, [0, 1536, 3072])  # a third of the rate
    ep = tmp_path / "episode_x"
    formats.video_views_episode(ep, {"exo": ("top", tmp_path / "top.mp4"), "left": ("wrist", tmp_path / "wrist.mp4")},
                                "teleop_arms", "mine", {})
    src = json.loads((ep / "sources.json").read_text())
    assert src["left"]["kmap"] == "kmap_left.npy" and "kmap" not in src["exo"]
    assert np.load(ep / "kmap_left.npy").tolist() == [0, 0, 1, 1, 1, 2]


# ---- your own data: a LeRobot v2.1 folder ----

def _lerobot_v21(root, n: int = 45, task: str = "put the cube in the bowl", dims: int = 14, extra: dict | None = None):
    feats = {"observation.images.cam_high": {"dtype": "video", "shape": [H, W, 3],
                                             "info": {"video.width": W, "video.height": H, "video.codec": "mpeg4"}},
             "observation.images.cam_left_wrist": {"dtype": "video", "shape": [H, W, 3],
                                                   "info": {"video.width": W, "video.height": H,
                                                            "video.codec": "mpeg4"}},
             "observation.state": {"dtype": "float32", "shape": [dims]},
             "action": {"dtype": "float32", "shape": [dims]}}
    info = {"codebase_version": "v2.1", "fps": FPS, "chunks_size": 1000, "robot_type": "two test arms",
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": feats}
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info))
    (root / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": task}) + "\n")
    # episode 1 is listed in the metadata but has no files, as in a partly downloaded dataset
    (root / "meta" / "episodes.jsonl").write_text("".join(
        json.dumps({"episode_index": i, "tasks": [task], "length": n}) + "\n" for i in (0, 1)))
    rng = np.random.default_rng(0)
    state = np.cumsum(rng.normal(0, 0.01, (n, dims)), axis=0).astype(np.float32)
    df = pd.DataFrame({"observation.state": list(state), "action": list(state + 0.01),
                       "frame_index": np.arange(n), "episode_index": np.zeros(n, dtype=np.int64),
                       "timestamp": np.arange(n) / FPS, **(extra or {})})
    (root / "data" / "chunk-000").mkdir(parents=True)
    df.to_parquet(root / "data" / "chunk-000" / "episode_000000.parquet")
    for key, shade in (("observation.images.cam_high", 0), ("observation.images.cam_left_wrist", 100)):
        _mp4(root / "videos" / "chunk-000" / key / "episode_000000.mp4", n, shade=shade)
    return state


def test_lerobot_adapter_prepares_an_episode_the_harness_can_label(tmp_path):
    root = tmp_path / "my_dataset"
    state = _lerobot_v21(root)
    out = tmp_path / "episodes"
    rc, text = _main(lerobot, ["prepare", "--root", root, "--rig", "teleop_arms", "--out", out])
    assert rc == 0 and '"ok": 1' in text                  # episode 1 has no files and is not planned
    ep = out / "episode_000000"
    ctx = json.loads((ep / "context.json").read_text())
    assert ctx["dataset"] == "my_dataset" and ctx["profile"] == "teleop_arms" and ctx["state_kind"] == "joints"
    assert ctx["instruction"] == "put the cube in the bowl" and ctx["n_state_frames"] == 45
    assert ctx["stream_checks"]["frames_on_grid"] == {"exo": True, "left": True}
    assert set(ctx["cameras"]) == {"exo", "left"} and not (ep / "times.npz").exists()
    z = np.load(ep / "state.npz")
    assert np.array_equal(z["state"], state) and z["action"].shape == (45, 14)
    src = json.loads((ep / "sources.json").read_text())
    assert src["left"]["packed"].endswith("observation.images.cam_left_wrist/episode_000000.mp4")
    req = me.build_request(ep)
    assert req["views"] == ["exo", "left"] and req["given_prompt"] == "put the cube in the bowl"
    assert req["n_images"] >= 3 and "put the cube in the bowl" in req["prompt"]
    # prepared episodes are skipped unless --force; an absent listed episode is refused
    assert '"skip": 1' in _main(lerobot, ["prepare", "--root", root, "--rig", "teleop_arms", "--out", out])[1]
    lst = tmp_path / "list.txt"
    lst.write_text("1\n")
    rc, _ = _main(lerobot, ["prepare", "--root", root, "--rig", "teleop_arms", "--out", out, "--episodes", lst])
    assert rc != 0


def test_every_other_recorded_signal_reaches_the_model_under_its_own_name(tmp_path):
    """A column the reader has no slot for (a mobile base, joint velocities, a done flag) is kept under the dataset's
    name and shown at every sampled instant; bookkeeping columns are not; an episode with none gets no table."""
    n = 45
    base = np.stack([np.linspace(0, 0.9, n), np.zeros(n), np.linspace(0, 0.3, n)], axis=1)
    root = tmp_path / "mobile"
    _lerobot_v21(root, extra={"observation.state.chassis": list(base), "observation.state.torso": [[0.5, 0.25]] * n,
                              "next.done": [False] * (n - 1) + [True], "task_index": np.zeros(n, dtype=np.int64),
                              "index": np.arange(n)})
    out = tmp_path / "episodes"
    rc, _ = _main(lerobot, ["prepare", "--root", root, "--rig", "teleop_arms", "--out", out])
    ep = out / "episode_000000"
    ctx = json.loads((ep / "context.json").read_text())
    assert rc == 0 and ctx["state_kind"] == "joints"
    assert [s["name"] for s in ctx["signals"]] == ["observation.state.chassis", "observation.state.torso", "next.done"]
    z = np.load(ep / "signals.npz")
    assert np.allclose(z[ctx["signals"][0]["key"]], base) and z["s2"].shape == (n, 1)
    p = me.build_request(ep)["prompt"]
    table = p.split("OTHER RECORDED SIGNALS")[1].split("BETWEEN INSTANTS")[0]
    assert "observation.state.chassis (3 values): 0 to 0.9, 0, 0 to 0.3" in table
    # a value the same at every frame is named once, on one line with the others like it
    assert "The same at every frame: observation.state.torso [0.5, 0.25]" in table and "next.done (1 value): 0 to 1" in table
    assert "task_index" not in table and "  index" not in table and "timestamp" not in table
    # an episode that records nothing else gets exactly the prompt it had before
    plain = tmp_path / "plain"
    _lerobot_v21(plain)
    _main(lerobot, ["prepare", "--root", plain, "--rig", "teleop_arms", "--out", tmp_path / "plain_eps"])
    ctx = json.loads((tmp_path / "plain_eps" / "episode_000000" / "context.json").read_text())
    assert "signals" not in ctx and not (tmp_path / "plain_eps" / "episode_000000" / "signals.npz").exists()
    assert "OTHER RECORDED SIGNALS" not in me.build_request(tmp_path / "plain_eps" / "episode_000000")["prompt"]


def test_a_state_wider_than_the_arm_layout_is_shown_instead_of_dropped(tmp_path):
    root = tmp_path / "wide"
    _lerobot_v21(root, dims=16)
    out = tmp_path / "episodes"
    _main(lerobot, ["prepare", "--root", root, "--rig", "teleop_arms", "--out", out])
    ep = out / "episode_000000"
    ctx = json.loads((ep / "context.json").read_text())
    assert ctx["state_kind"] == "none" and [s["name"] for s in ctx["signals"]] == ["observation.state", "action"]
    p = me.build_request(ep)["prompt"]
    assert "observation.state (16 values)" in p and "the video is all there is" not in p


def test_the_other_signals_are_measured_over_each_still_span():
    """A base that drives while the arms hold still shows up where the still claim is made."""
    ep = {"signals": {"base": np.concatenate([np.zeros((60, 1)), np.linspace(0, 2, 60)[:, None]]),
                      "noise": np.full((120, 2), 0.5)}, "times": None, "kmap": {},
          "context": {"fps": 30}, "state": np.zeros((120, 14)), "sources": {}}
    t = me._signals_table(ep, {"n": 120, "spans": [(0, 59), (60, 119)], "ks": [0, 119]})
    assert "base (1 value): 0 to 2" in t and "The same at every frame: noise [0.5, 0.5]" in t
    assert "0.00-1.97s: none changed" in t and "2.00-3.97s: base 2" in t


def test_recorded_signals_skip_bookkeeping_and_what_an_adapter_holds_back():
    n = 4
    df = pd.DataFrame({"observation.velocity": [[1.0, 2.0]] * n, "frame_index": np.arange(n),
                       "coarse_quality_index": np.zeros(n), "timestamp": np.arange(n) / 30.0,
                       "is_error_segment": [0, 1, 1, 0], "note": ["a"] * n,
                       "picture": [list(range(formats.SIGNAL_MAX_VALUES + 1))] * n})
    got = formats.recorded_signals(df, set(habit.PUBLISHER_COLUMNS), n)
    assert list(got) == ["observation.velocity"] and got["observation.velocity"].shape == (n, 2)
    every = formats.recorded_signals(df, set(), n)
    assert list(every) == ["observation.velocity", "is_error_segment"]
    # a column too wide to be a signal is named with the reason, never dropped without a word
    assert [k for k, _ in every.left_out] == ["picture"]
    assert formats.recorded_signals(df, set(), n + 1) == {}          # a column shorter than the episode is not kept


def test_a_tactile_map_keeps_its_shape_and_its_value_names():
    """A glove's 16 x 16 pressure map (parquet's list of lists) is one signal of 256 values shaped 16 x 16, not dropped
    for being wide; a force sensor's values keep the names the dataset gives them."""
    n = 5
    grid = [[[float(r * 16 + c + k) for c in range(16)] for r in range(16)] for k in range(n)]
    df = pd.DataFrame({"observation.tactile.right": grid, "observation.force": [[0.1, 0.2, 9.8]] * n})
    feats = {"observation.force": {"dtype": "float32", "shape": [3], "names": {"axes": ["fx", "fy", "fz"]}}}
    got = formats.recorded_signals(df, set(), n, feats)
    assert got["observation.tactile.right"].shape == (n, 256)
    assert got.meta["observation.tactile.right"]["shape"] == [16, 16]
    assert got["observation.tactile.right"][2, 17] == 2 + 17            # row 1, column 1 of frame 2, in the map's order
    assert got.meta["observation.force"]["names"] == ["fx", "fy", "fz"] and "shape" not in got.meta["observation.force"]


def test_a_fourth_camera_is_sent_to_the_model_under_its_own_name(tmp_path):
    root = tmp_path / "four_cams"
    _lerobot_v21(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    info["features"]["observation.images.cam_low"] = dict(info["features"]["observation.images.cam_high"])
    (root / "meta" / "info.json").write_text(json.dumps(info))
    _mp4(root / "videos" / "chunk-000" / "observation.images.cam_low" / "episode_000000.mp4", 45, shade=200)
    out = tmp_path / "episodes"
    rc, _ = _main(lerobot, ["prepare", "--root", root, "--rig", "teleop_arms", "--out", out])
    ep = out / "episode_000000"
    ctx = json.loads((ep / "context.json").read_text())
    assert rc == 0 and set(ctx["cameras"]) == {"exo", "left", "extra1"} and ctx["source"]["unused_cameras"] == []
    req = me.build_request(ep)
    assert req["views"] == ["exo", "left", "extra1"] and req["cam_labels"][-1] == ctx["cameras"]["extra1"]["name"]
    assert f"- {ctx['cameras']['extra1']['name']}: another camera the recording has" in req["prompt"]
    assert "There are exactly 3" in req["prompt"]


def test_a_lerobot_camera_without_its_video_is_listed_as_unused(tmp_path):
    root = tmp_path / "my_dataset"
    _lerobot_v21(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    info["features"]["observation.images.cam_low"] = dict(info["features"]["observation.images.cam_high"])
    (root / "meta" / "info.json").write_text(json.dumps(info))
    out = tmp_path / "episodes"
    rc, _ = _main(lerobot, ["prepare", "--root", root, "--rig", "teleop_arms", "--out", out])
    ctx = json.loads((out / "episode_000000" / "context.json").read_text())
    assert rc == 0 and set(ctx["cameras"]) == {"exo", "left"}
    assert ctx["source"]["unused_cameras"] == ["observation.images.cam_low"]


# ---- your own data: a folder of videos ----

def test_videos_adapter_prepares_each_file_with_its_instruction(tmp_path):
    root = tmp_path / "clips"
    _mp4(root / "kitchen" / "run-1.mp4", 40)
    _mp4(root / "kitchen" / "run_1.mp4", 30, shade=50)
    (root / "notes.txt").write_text("not a video")
    ins = tmp_path / "instructions.json"
    towel = {"instruction": "dry the mug", "subtasks": [{"t0": 0, "t1": 0.5, "label": "pick up the towel"}]}
    ins.write_text(json.dumps({"kitchen/run-1.mp4": "wash the mug", "kitchen/run_1.mp4": towel}))
    out = tmp_path / "episodes"
    rc, _ = _main(videos, ["prepare", "--root", root, "--out", out, "--instructions", ins, "--dataset", "home"])
    assert rc == 0
    a = json.loads((out / "episode_kitchen_run_1" / "context.json").read_text())
    b = json.loads((out / "episode_kitchen_run_1_2" / "context.json").read_text())
    assert a["instruction"] == "wash the mug" and "annotation_subtasks" not in a
    assert a["task_label"] == ["kitchen/run-1"] and a["cameras"]["exo"]["key"] == "run-1"
    assert b["instruction"] == "dry the mug"
    assert b["annotation_subtasks"] == [{"t0": 0.0, "t1": 0.5, "label": "pick up the towel"}]
    assert a["profile"] == "ego_head" and a["state_kind"] == "none" and a["dataset"] == "home"
    assert a["n_state_frames"] == 40 and b["source"] == {"format": "video files", "file": "kitchen/run_1.mp4"}
    req = me.build_request(out / "episode_kitchen_run_1_2")
    assert req["views"] == ["exo"] and 'goal: "dry the mug"' in req["prompt"]
    assert "0.0-0.5s  pick up the towel" in req["prompt"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"kitchen/missing.mp4": "x"}))
    assert _main(videos, ["prepare", "--root", root, "--out", out, "--instructions", bad])[0] != 0


@pytest.mark.parametrize("rig,view,name,rule", [
    ("teleop_arms", "exo", "clip", "ABOUT THE TASK. This episode comes with no instruction"),
    ("handheld_gripper", "right", "gripper", "ABOUT THE TASK. This episode comes with no instruction"),
    ("ego_head", "exo", "head", "THE DATASET'S ANNOTATION FOR THIS EPISODE: none")])
def test_videos_adapter_takes_a_bare_video_of_any_rig(tmp_path, rig, view, name, rule):
    """One file with no instruction: the rig decides the camera's slot and name, and the model is told to infer the
    task (teleop and handheld) or the activities (head camera)."""
    _mp4(tmp_path / "clips" / "clip.mp4", 40)
    out = tmp_path / "episodes"
    rc, _ = _main(videos, ["prepare", "--root", tmp_path / "clips", "--out", out, "--rig", rig])
    assert rc == 0
    ctx = json.loads((out / "episode_clip" / "context.json").read_text())
    assert ctx["profile"] == rig and ctx["state_kind"] == "none" and list(ctx["cameras"]) == [view]
    assert ctx["cameras"][view]["name"] == name and ctx["task_label"] == ["clip"] and "instruction" not in ctx
    req = me.build_request(out / "episode_clip")
    assert req["views"] == [view] and rule in req["prompt"] and req["contact_s"] == []
    assert "ABOUT THE EPISODE'S INSTRUCTION" not in req["prompt"]
    with_subtasks = tmp_path / "subtasks.json"
    with_subtasks.write_text(json.dumps({"clip.mp4": {"instruction": "x", "subtasks": [{"t0": 0, "t1": 1, "label": "y"}]}}))
    rc, _ = _main(videos, ["prepare", "--root", tmp_path / "clips", "--out", tmp_path / "e2", "--rig", rig,
                           "--instructions", with_subtasks])
    assert (rc == 0) == (rig == "ego_head")


# ---- Egocentric-100K: one clip read out of a tar shard by byte ranges ----

def test_egocentric100k_downloads_one_clip_from_a_shard_by_range(tmp_path, monkeypatch):
    import tarfile
    for name, n in (("f_1_w_2_0001", 20), ("f_1_w_2_0002", 30)):
        _mp4(tmp_path / "src" / f"{name}.mp4", n)
        (tmp_path / "src" / f"{name}.json").write_text(json.dumps({"factory_id": "f1", "worker_id": "w2", "fps": 30}))
    shard = tmp_path / "part000.tar"
    with tarfile.open(shard, "w", format=tarfile.USTAR_FORMAT) as t:
        for f in sorted((tmp_path / "src").iterdir()):
            t.add(f, arcname=f.name)
    body = shard.read_bytes()
    calls = []

    def fake_get(url, rng=None):
        calls.append(rng)
        return (body[rng[0]:rng[1] + 1] if rng else body), {}
    monkeypatch.setattr(egocentric100k, "_get", fake_get)
    raw, out = tmp_path / "raw", tmp_path / "episodes"
    assert egocentric100k.prepare_one("f1/w2/part000.tar", "f_1_w_2_0002.mp4", raw, out) == "ok"
    mp4 = raw / "f1" / "w2" / "part000" / "f_1_w_2_0002.mp4"
    assert mp4.read_bytes() == (tmp_path / "src" / "f_1_w_2_0002.mp4").read_bytes()
    ctx = json.loads((out / "episode_f_1_w_2_0002" / "context.json").read_text())
    assert ctx["n_state_frames"] == 30 and ctx["task_label"] == ["f1 w2"] and "real_times" not in ctx
    assert ctx["source"] == {"shard": "f1/w2/part000.tar", "member": "f_1_w_2_0002.mp4",
                             "clip": {"factory_id": "f1", "worker_id": "w2", "fps": 30}}
    assert json.loads((out / "episode_f_1_w_2_0002" / "sources.json").read_text())["exo"]["packed"] == \
        str(mp4.resolve())
    calls.clear()
    assert egocentric100k.prepare_one("f1/w2/part000.tar", "f_1_w_2_0002.mp4", raw, out) == "skip"
    assert egocentric100k.prepare_one("f1/w2/part000.tar", "f_1_w_2_0002.mp4", raw, out, force=True) == "ok"
    assert calls == []                                   # the clip already in RAW is not downloaded again


# ---- the public-dataset adapters' pure functions ----

def _molmo_index() -> pd.DataFrame:
    """Two tasks; task a has 4 packs of 3 episodes, task b 2 packs of 2; every episode 60 s."""
    rows = []
    for task, packs, per in (("a", 4, 3), ("b", 2, 2)):
        for p in range(packs):
            for _ in range(per):
                rows.append({"eidx": len(rows), "task": task, "length": 1800, "top_chunk": 0,
                             "top_file": p + (10 if task == "b" else 0)})
    df = pd.DataFrame(rows)
    df["dur_s"] = df["length"] / molmo.FPS
    return df


def test_molmo_samplers_are_deterministic_and_take_whole_packs():
    df = _molmo_index()
    picks = molmo.sample_packs(df, 10 / 60)                  # 10 minutes
    assert picks == molmo.sample_packs(df, 10 / 60)
    assert sorted(picks) == sorted(molmo.sample_packs(df.sample(frac=1, random_state=3), 10 / 60))
    chosen = df[df["eidx"].isin(picks)]
    assert len(picks) == len(set(picks))
    for (task, f), g in chosen.groupby(["task", "top_file"]):
        assert len(g) == len(df[(df["task"] == task) & (df["top_file"] == f)])  # a pack is taken whole
    assert set(chosen["task"]) == {"a", "b"} and chosen["dur_s"].sum() >= 600
    eps = molmo.sample_episodes(df, 1.0, per_task_cap_h=3 / 60)
    assert eps == molmo.sample_episodes(df, 1.0, per_task_cap_h=3 / 60)
    assert (df[df["eidx"].isin(eps)].groupby("task").size() == 3).all()


def test_spread_permutation_prefixes_span_the_range():
    order = molmo._spread_permutation(9)
    assert sorted(order) == list(range(9))
    assert order[:3] == [4, 1, 6]


def test_galaxea_habit_and_openaoe_annotation_helpers():
    assert galaxea.spans(np.array([3, 3, 5, 5, 5, 3])) == [(0, 1, 3), (2, 4, 5), (5, 5, 3)]
    assert galaxea.english("拿起@pick up the cup") == "pick up the cup"
    assert habit.spans(np.array([0, 1, 1, 0, 1], dtype=bool), 10.0) == [[0.1, 0.3], [0.4, 0.5]]
    assert habit.timed_parts(np.array([-1, 0, 0, 1]), {0: "reach", 1: "grasp"}, 2.0) == \
        ["0.5-1.5 s: reach", "1.5-2.0 s: grasp"]
    subs = openaoe.subtasks([{"start_ts": 0, "end_ts": 5, "atomic_action": [
        {"verb": "pick up", "object": "cup", "hand": "left"}, {"verb": "wipe"}]}, {"start_ts": 5, "end_ts": 7}])
    assert subs == [{"t0": 0.0, "t1": 5.0, "label": "pick up cup (left hand); wipe", "ok": True},
                    {"t0": 5.0, "t1": 7.0, "label": "segment", "ok": True}]


def test_realomin_quaternion_to_roll_pitch_yaw():
    half = np.sqrt(0.5)
    rpy = realomin.quat_to_rpy(np.array([[0, 0, 0, 1], [half, 0, 0, half], [0, 0, half, half]], dtype=float))
    assert np.allclose(rpy, [[0, 0, 0], [np.pi / 2, 0, 0], [0, 0, np.pi / 2]])


def test_an_abc130k_remux_keeps_its_last_frame(tmp_path):
    """Each copied packet lasts until the next frame's time (the last repeats the step before it), so the mp4's edit
    list reaches the end of the last frame. Without a duration, a last frame that starts on a whole millisecond is where
    the edit list ends, and it does not decode."""
    from prepare import abc130k
    enc = av.CodecContext.create("libx264", "w")
    enc.width, enc.height, enc.pix_fmt = 64, 48, "yuv420p"
    enc.time_base = Fraction(1, 30)
    enc.options = {"preset": "ultrafast", "bframes": "0", "g": "30"}
    n, frames = 61, []
    for i in range(n):
        fr = av.VideoFrame.from_ndarray(np.full((48, 64, 3), (i * 4) % 256, np.uint8), format="rgb24")
        fr.pts = i
        frames += [bytes(p) for p in enc.encode(fr)]
    frames += [bytes(p) for p in enc.encode()]
    t = np.arange(n) * 0.033                     # every frame starts on a whole millisecond
    out = tmp_path / "top.mp4"
    abc130k.remux(frames, t, "h264", out)
    with av.open(str(out)) as c:
        st = c.streams.video[0]
        durs = [int(p.duration * p.time_base * abc130k.TIME_BASE_DEN) for p in c.demux(st) if p.size]
    with av.open(str(out)) as c:
        assert sum(1 for _ in c.decode(video=0)) == n
    assert durs == [33_000] * n


def _h264_frames(n: int) -> list[bytes]:
    """n Annex-B H.264 frames, the first carrying SPS, PPS and an IDR, as a recorder writes them one per message."""
    enc = av.CodecContext.create("libx264", "w")
    enc.width, enc.height, enc.pix_fmt = 64, 48, "yuv420p"
    enc.time_base = Fraction(1, 30)
    enc.options = {"preset": "ultrafast", "bframes": "0", "g": "30"}
    frames = []
    for i in range(n):
        fr = av.VideoFrame.from_ndarray(np.full((48, 64, 3), (i * 4) % 256, np.uint8), format="rgb24")
        fr.pts = i
        frames += [bytes(p) for p in enc.encode(fr)]
    return frames + [bytes(p) for p in enc.encode()]


def _file_times(mp4) -> np.ndarray:
    with av.open(str(mp4)) as c:
        st = c.streams.video[0]
        pts = np.sort([p.pts for p in c.demux(st) if p.size])
        return (pts - pts[0]) * float(st.time_base)


# a camera running a little slow (29.9 fps) that drops frame 40: nothing about its times is on a 30 fps grid
_OFF_GRID = np.delete(np.arange(81) / 29.9, 40)


def test_a_recorded_camera_is_written_at_its_recorded_times_not_at_a_constant_rate(tmp_path):
    from prepare import remux
    frames = _h264_frames(len(_OFF_GRID))
    remux.remux(frames, _OFF_GRID, "h264", tmp_path / "cam.mp4")
    assert np.allclose(_file_times(tmp_path / "cam.mp4"), _OFF_GRID, atol=2e-6)
    with av.open(str(tmp_path / "cam.mp4")) as c:
        assert sum(1 for _ in c.decode(video=0)) == len(_OFF_GRID)


def test_a_realomni_camera_keeps_its_capture_times(tmp_path):
    frames = _h264_frames(len(_OFF_GRID))
    ts = [int(round(1_700_000_000e9 + x * 1e9)) for x in _OFF_GRID]
    assert realomin.mux(frames, ts, tmp_path / "left.mp4") == ts
    assert np.allclose(_file_times(tmp_path / "left.mp4"), _OFF_GRID, atol=2e-6)   # was one median rate for all


def _genhumanego_raw(raw, times=None):
    from prepare import remux
    raw.mkdir(parents=True)
    t = _OFF_GRID if times is None else times
    remux.remux(_h264_frames(len(t)), t, "h264", raw / "cam2.mp4")
    (raw / "meta.json").write_text(json.dumps({"rel": "x.mcap", "dur": float(t[-1]), "ann": None, "fv_invalid": 0,
                                               "calib2": None}))


def test_a_genhumanego_episode_uses_its_recorded_frame_times(tmp_path):
    from prepare import genhumanego as gh
    raw, dst = tmp_path / "raw" / "abc", tmp_path / "episode_abc"
    _genhumanego_raw(raw)
    np.save(raw / gh.TIMES, _OFF_GRID)
    ctx = gh.write_sidecar(raw, dst)
    z = np.load(dst / "times.npz")
    assert ctx["real_times"] == "times.npz" and ctx["source"]["frame_times"] == gh.FRAME_TIMES_RECORDED
    assert np.array_equal(z["exo"], _OFF_GRID) and len(z["exo_pts"]) == len(_OFF_GRID)
    assert abs(ctx["fps"] - 29.9) < 0.01                                  # measured, not assumed to be 30
    assert ctx["n_state_frames"] == len(_OFF_GRID)


def test_a_genhumanego_copy_without_recorded_times_says_its_times_are_nominal(tmp_path):
    from prepare import genhumanego as gh
    raw, dst = tmp_path / "raw" / "old", tmp_path / "episode_old"
    _genhumanego_raw(raw, np.arange(60) / 30)                             # extracted before the times were kept
    ctx = gh.write_sidecar(raw, dst)
    assert "real_times" not in ctx and not (dst / "times.npz").exists()
    assert ctx["source"]["frame_times"] == gh.FRAME_TIMES_NOMINAL and ctx["fps"] == 30.0


def test_genhumanego_refuses_recorded_times_that_do_not_match_its_frames(tmp_path):
    from prepare import genhumanego as gh
    raw = tmp_path / "raw" / "bad"
    _genhumanego_raw(raw)
    np.save(raw / gh.TIMES, _OFF_GRID[:-1])
    with pytest.raises(RuntimeError, match="recorded frame times"):
        gh.write_sidecar(raw, tmp_path / "episode_bad")



def test_prepare_lerobot_measures_the_gripper_range_as_an_upload_does(tmp_path):
    """python -m prepare lerobot reads a dataset exactly as Data Review reads an upload, so a gripper recorded 0 to 100
    gets the range measured across the dataset (its still tolerance is 1% of 100), as an upload's does."""
    from label import state as ls
    n = 150
    root = tmp_path / "gripper100"
    _lerobot_v21(root, n=n)
    p = root / "data" / "chunk-000" / "episode_000000.parquet"
    df = pd.read_parquet(p)
    rng = np.random.default_rng(1)
    st = np.zeros((n, 14), dtype=np.float32)
    st[:, 6] = 0.2 * rng.random(n)                       # shut gripper: its reading wanders by 0.2
    st[:, 13] = 100 - 0.2 * rng.random(n)                # open gripper, the other end of its 0 to 100 range
    df["observation.state"] = list(st)
    df["action"] = list(st)
    df.to_parquet(p)
    rep = formats.convert(root, "teleop_arms", tmp_path / "upload_eps", "test", 900)
    up = json.loads((tmp_path / "upload_eps" / rep["episodes"][0]["episode_id"] / "context.json").read_text())
    argv = sys.argv
    sys.argv = ["x", "prepare", "--root", str(root), "--rig", "teleop_arms", "--out", str(tmp_path / "cli_eps")]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            lerobot.main()
    finally:
        sys.argv = argv
    cli_ctx = json.loads((tmp_path / "cli_eps" / "episode_000000" / "context.json").read_text())
    assert ls.still_spans(st, fps=30.0, kind="joints", grip_range=ls.gripper_full_range(up)) == [(0, n - 1)]
    assert cli_ctx.get("gripper_range") == up.get("gripper_range") is not None


def test_an_off_grid_file_without_times_fails_or_shows_the_frame_of_its_time(tmp_path):
    """FastUMI writes no times.npz: frame k is decoded at pts k * step. In a file missing one frame, the missing
    instant raises and every other instant decodes the frame whose pts is its own time (never a neighbour's)."""
    from label import frames as mf
    pts = [0, 512, 1024, 2048, 2560, 3072]                  # the frame at 3/30 s was never written
    _mp4(tmp_path / "gap.mp4", len(pts), pts=pts)
    got = mf.extract_frames(tmp_path / "gap.mp4", 0.0, len(pts), [0, 2, 4])
    assert sorted(got) == [0, 2, 4]
    with pytest.raises(mf.FrameError):
        mf.extract_frames(tmp_path / "gap.mp4", 0.0, len(pts), [3])
    _mp4(tmp_path / "shift.mp4", 4, pts=[512, 1024, 1536, 2048])  # the whole file one frame late
    with pytest.raises(mf.FrameError):
        mf.extract_frames(tmp_path / "shift.mp4", 0.0, 4, [0])
