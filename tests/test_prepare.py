"""The prepare stage on synthetic inputs (no network): list parsing, episode naming, the shared command line,
the sidecar writers on small mp4s written here, the two adapters for your own data (a LeRobot v2.1 folder and a
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
from prepare import sidecar
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
    assert sidecar.episode_name("raw_019fd620-0759-42c1_seg_1") == "episode_raw_019fd620_0759_42c1_seg_1"
    assert sidecar.episode_name("000031") == "episode_000031"
    assert sidecar.episode_name("episode_7") == "episode_7"
    assert sidecar.episode_name("--") == "episode_0"
    assert len(sidecar.episode_name("x" * 500)) == len("episode_") + 120


def test_episode_dirs_suffixes_collisions_in_list_order_and_is_stable(tmp_path):
    names = ["run-1", "run_1", "other", "run 1"]
    dirs = sidecar.episode_dirs(tmp_path, names)
    assert [d.name for d in dirs] == ["episode_run_1", "episode_run_1_2", "episode_other", "episode_run_1_3"]
    for d in dirs:
        d.mkdir()
    assert sidecar.episode_dirs(tmp_path, names) == dirs     # a rerun maps every source to the same folder


def test_adapter_episode_folder_names():
    assert galaxea.episode_dir_name("Plug_Into_A_Fixed_Socket_20250710_006", 69) == \
        "episode_Plug_Into_A_Fixed_Socket_20250710_006_000069"
    assert fastumi.episode_dir_name("single_arm/open_toilet_lid/1638") == "episode_single_arm__open_toilet_lid__001638"
    assert realomin.episode_dir_name("Clutter Tidy-Up [Stage2]/00001/01751.mcap") == \
        "episode_Clutter_Tidy_Up_Stage2_00001_01751"
    assert realomin.task_of("Clutter Tidy-Up [Stage2]/00001/01751.mcap") == "Clutter Tidy-Up [Stage2]"


# ---- camera assignment and pairing ----

def test_cameras_are_assigned_to_views_by_name():
    views, unused = sidecar.assign_views(["observation.images.cam_high", "observation.images.cam_left_wrist",
                                          "observation.images.cam_right_wrist", "observation.images.cam_low"],
                                         "teleop_arms")
    assert views == {"left": "observation.images.cam_left_wrist", "right": "observation.images.cam_right_wrist",
                     "exo": "observation.images.cam_high"}
    assert unused == ["observation.images.cam_low"]
    # a fixed camera that carries a side is not a wrist camera
    assert sidecar.mounted_side("exterior_image_1_left") is None and sidecar.mounted_side("leftWrist") == "left"
    # the only camera of a single handheld gripper is the gripper's own
    assert sidecar.assign_views(["observation.images.camera"], "handheld_gripper")[0] == \
        {"right": "observation.images.camera"}
    assert sidecar.pick_cameras(["cam_side", "cam_head"], "ego_head") == ({"exo": "cam_head"}, ["cam_side"])


def test_nearest_pairs_each_query_with_the_closest_time_earlier_on_a_tie():
    src = np.array([0, 10, 20, 30])
    assert sidecar.nearest(src, np.array([-5, 4, 5, 6, 26, 99])).tolist() == [0, 0, 0, 1, 3, 3]
    assert sidecar.nearest(np.array([7]), np.array([0, 100])).tolist() == [0, 0]
    assert sidecar.nearest(src, np.array([1])).dtype == np.int32


def test_state_layout_reads_seven_values_per_actor_else_video_only():
    assert sidecar.state_layout(14, "teleop_arms") == ("joints", None)
    assert sidecar.state_layout(7, "handheld_gripper") == ("ee_pose", None)
    kind, note = sidecar.state_layout(9, "teleop_arms")
    assert kind == "none" and "9 values" in note
    assert sidecar.state_layout(14, "ego_head") == ("none", None)


# ---- the sidecar writer on a real mp4 ----

def test_video_views_episode_times_frames_by_their_pts(tmp_path):
    pts = [0, 512, 1024, 2048, 2560, 3072]          # one frame missing after the third
    _mp4(tmp_path / "v" / "a.mp4", len(pts), pts)
    ep = tmp_path / "out" / "episode_a"
    ctx = sidecar.video_views_episode(ep, {"exo": ("a", tmp_path / "v" / "a.mp4")}, "ego_head", "mine",
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
    assert ctx["cameras"]["exo"]["name"] == "head" and ctx["cameras"]["exo"]["desc"] == sidecar.EGO_DESC
    assert not (ep / "state.npz").exists() and (ep / "instruction.txt").read_text() == "\n"


def test_video_views_episode_pairs_a_second_camera_by_time(tmp_path):
    _mp4(tmp_path / "top.mp4", 6)
    _mp4(tmp_path / "wrist.mp4", 3, [0, 1536, 3072])  # a third of the rate
    ep = tmp_path / "episode_x"
    sidecar.video_views_episode(ep, {"exo": ("top", tmp_path / "top.mp4"), "left": ("wrist", tmp_path / "wrist.mp4")},
                                "teleop_arms", "mine", {})
    src = json.loads((ep / "sources.json").read_text())
    assert src["left"]["kmap"] == "kmap_left.npy" and "kmap" not in src["exo"]
    assert np.load(ep / "kmap_left.npy").tolist() == [0, 0, 1, 1, 1, 2]


# ---- your own data: a LeRobot v2.1 folder ----

def _lerobot_v21(root, n: int = 45, task: str = "put the cube in the bowl"):
    feats = {"observation.images.cam_high": {"dtype": "video", "shape": [H, W, 3],
                                             "info": {"video.width": W, "video.height": H, "video.codec": "mpeg4"}},
             "observation.images.cam_left_wrist": {"dtype": "video", "shape": [H, W, 3],
                                                   "info": {"video.width": W, "video.height": H,
                                                            "video.codec": "mpeg4"}},
             "observation.state": {"dtype": "float32", "shape": [14]},
             "action": {"dtype": "float32", "shape": [14]}}
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
    state = np.cumsum(rng.normal(0, 0.01, (n, 14)), axis=0).astype(np.float32)
    df = pd.DataFrame({"observation.state": list(state), "action": list(state + 0.01),
                       "frame_index": np.arange(n), "episode_index": np.zeros(n, dtype=np.int64),
                       "timestamp": np.arange(n) / FPS})
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
    assert a["n_state_frames"] == 40 and b["source"] == {"file": "kitchen/run_1.mp4"}
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
