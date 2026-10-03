"""Synthetic recordings of the families of robot data people ship, one case per container (LeRobot v2.1 and v3.0,
HDF5, MCAP with JSON and with ROS 2 messages, a folder of videos with sidecars), built for the 2026-10-02 coverage
audit of prepare/formats.py. tests/test_audit_families.py builds every case and asserts where each family lands.
Each case is 3 s at 30 fps with 128 x 96 images. A case also writes _fields.json (family, field, search token, role)
beside its upload folder, for a by-hand audit run."""
from __future__ import annotations

import base64
import io
import json
import shutil
from fractions import Fraction
from pathlib import Path

import numpy as np

FPS = 30
N = 90                       # 3 s of footage
W, H = 128, 96
T0_NS = 1_790_000_000 * 10**9   # recorder clock: ns since 1970
rng = np.random.default_rng(0)
t = np.arange(N) / FPS


# ------------------------------------------------------------------ signal content

def frame(k, shade=60, n=N):
    a = np.full((H, W, 3), shade, np.uint8)
    x = int(10 + (W - 30) * k / max(n - 1, 1))
    a[30:50, x:x + 20] = (220, 40, 40)
    return a


def jpeg(a) -> bytes:
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(a).save(b, format="JPEG", quality=85)
    return b.getvalue()


def png16(a) -> bytes:
    from PIL import Image
    b = io.BytesIO()
    Image.fromarray(a.astype(np.uint16)).save(b, format="PNG")
    return b.getvalue()


def joints(tt, d, phase=0.0):
    return np.stack([0.3 * np.sin(0.8 * tt + j + phase) for j in range(d)], axis=1)


def gripper(tt, open_=0.08, shut=0.02):
    g = np.full(len(tt), open_)
    on = (tt >= 1.0) & (tt < 2.0)
    g[on] = shut
    return g


def wrench(tt):
    w = 0.05 * rng.normal(size=(len(tt), 6)) + np.array([0.3, -0.1, 2.0, 0.01, 0.02, 0.0])
    on = (tt >= 1.0) & (tt < 2.0)
    w[on, 2] += 15.0
    w[on, 3] += 0.8
    return w


def odom(tt):
    vx = np.where(tt < 1.5, 0.2, 0.0)
    x = np.cumsum(vx) / FPS
    yaw = 0.1 * tt
    return np.stack([x, 0.01 * tt, yaw, vx, np.full(len(tt), 0.1)], axis=1)


def imu(tt):
    a = np.stack([0.05 * rng.normal(size=len(tt)), 0.05 * rng.normal(size=len(tt)),
                  9.81 + 0.05 * rng.normal(size=len(tt))], 1)
    g = 0.02 * rng.normal(size=(len(tt), 3))
    q = np.stack([np.cos(0.05 * tt), np.zeros(len(tt)), np.zeros(len(tt)), np.sin(0.05 * tt)], 1)
    return np.concatenate([a, g, q], 1)


def wheels(tt):
    v = np.where(tt < 1.5, 4.0, 0.0)
    return np.stack([v, v, v + 0.1 * np.sin(tt), v - 0.1 * np.sin(tt)], 1)


def pressure(tt, rest=0.0):
    a = np.full((len(tt), 16, 16), rest) + 0.5 * rng.random((len(tt), 16, 16))
    on = (tt >= 1.0) & (tt < 2.0)
    a[on, 4:9, 5:11] += 60.0
    return a


def finger_forces(tt):
    f = 0.02 * rng.random((len(tt), 5, 3))
    on = (tt >= 1.0) & (tt < 2.0)
    f[on, :3, 2] += 3.0
    return f


def keypoints(tt, k=21):
    base = rng.normal(0, 0.05, (k, 3))
    return base[None] + 0.02 * np.sin(tt)[:, None, None] + 0.002 * rng.normal(size=(len(tt), k, 3))


def flag(tt, a=1.0, b=1.8):
    return ((tt >= a) & (tt < b)).astype(np.float64)


def rl(n):
    rew = np.zeros(n); rew[-1] = 1.0
    disc = np.ones(n); disc[-1] = 0.0
    first = np.zeros(n); first[0] = 1
    last = np.zeros(n); last[-1] = 1
    term = last.copy()
    succ = np.zeros(n); succ[-1] = 1
    return {"reward": rew, "discount": disc, "is_first": first, "is_last": last, "is_terminal": term, "success": succ}


def audio_chunks(n, sr=16000):
    per = sr // FPS
    x = 0.01 * rng.normal(size=(n, per))
    on = (np.arange(n) / FPS >= 2.2) & (np.arange(n) / FPS < 2.4)
    x[on] += 0.5 * np.sin(np.arange(per) * 0.3)
    return x


def write_mp4(path: Path, n: int, shade: int):
    import av
    path.parent.mkdir(parents=True, exist_ok=True)
    c = av.open(str(path), "w")
    s = c.add_stream("mpeg4", rate=FPS)
    s.width, s.height, s.pix_fmt = W, H, "yuv420p"
    for k in range(n):
        fr = av.VideoFrame.from_ndarray(frame(k % N, shade), format="rgb24")
        fr.pts = k
        for p in s.encode(fr):
            c.mux(p)
    for p in s.encode():
        c.mux(p)
    c.close()


def write_depth_mkv(path: Path, n: int):
    import av
    path.parent.mkdir(parents=True, exist_ok=True)
    c = av.open(str(path), "w", format="matroska")
    s = c.add_stream("ffv1", rate=FPS)
    s.width, s.height, s.pix_fmt = W, H, "gray16le"
    for k in range(n):
        fr = av.VideoFrame.from_ndarray(np.full((H, W), 800 + 5 * (k % N), np.uint16), format="gray16le")
        fr.pts = k
        for p in s.encode(fr):
            c.mux(p)
    for p in s.encode():
        c.mux(p)
    c.close()


def fields(case_dir: Path, rows: list):
    (case_dir / "_fields.json").write_text(json.dumps([dict(zip(("family", "field", "token", "role"), r))
                                                       for r in rows], indent=1))


def fresh(root: Path, name: str) -> tuple[Path, Path]:
    d = root / name
    if d.exists():
        shutil.rmtree(d)
    up = d / "upload"
    up.mkdir(parents=True)
    return d, up


def lst(a):
    return [np.asarray(x).tolist() for x in a]


# ------------------------------------------------------------------ 1. LeRobot v2.1, bimanual 7-DoF arms

def lerobot_v21_bimanual(root: Path, depth: bool = True):
    import pandas as pd
    d, up = fresh(root, "lerobot_v21_bimanual_7dof" + ("" if depth else "_nodepth"))
    arm_names = [f"{s}_joint{j}" for s in ("left", "right") for j in range(1, 8)]
    names16 = arm_names[:7] + ["left_gripper"] + arm_names[7:] + ["right_gripper"]
    cams = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist",
            "observation.images.cam_high_mask"]
    feats = {
        "observation.state": {"dtype": "float32", "shape": [16], "names": {"motors": names16}},
        "observation.velocity": {"dtype": "float32", "shape": [16], "names": {"motors": names16}},
        "observation.effort": {"dtype": "float32", "shape": [16], "names": {"motors": names16}},
        "action": {"dtype": "float32", "shape": [16], "names": {"motors": names16}},
        "observation.leader_state": {"dtype": "float32", "shape": [16]},
        "observation.ee_pose": {"dtype": "float32", "shape": [14],
                                "names": [f"{s}_{c}" for s in ("left", "right")
                                          for c in ("x", "y", "z", "qx", "qy", "qz", "qw")]},
        "observation.gripper_width": {"dtype": "float32", "shape": [2], "names": ["left", "right"]},
        "observation.wrench.left": {"dtype": "float32", "shape": [6], "names": ["fx", "fy", "fz", "tx", "ty", "tz"]},
        "teleop.intervention": {"dtype": "bool", "shape": [1]},
        "next.reward": {"dtype": "float32", "shape": [1]}, "next.done": {"dtype": "bool", "shape": [1]},
        "next.success": {"dtype": "bool", "shape": [1]}, "is_first": {"dtype": "bool", "shape": [1]},
        "is_last": {"dtype": "bool", "shape": [1]}, "is_terminal": {"dtype": "bool", "shape": [1]},
        "discount": {"dtype": "float32", "shape": [1]},
        "language_instruction": {"dtype": "string", "shape": [1]},
        "subtask_index": {"dtype": "int64", "shape": [1]},
        "recorder_time_ns": {"dtype": "int64", "shape": [1]},
        "observation.audio": {"dtype": "float32", "shape": [533]},
        "timestamp": {"dtype": "float32", "shape": [1]}, "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]}, "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
    }
    for c in cams:
        feats[c] = {"dtype": "video", "shape": [H, W, 3], "names": ["height", "width", "channel"],
                    "info": {"video.fps": 30, "video.codec": "mpeg4", "camera.intrinsics.fx": 400.0,
                             "camera.intrinsics.fy": 400.0, "camera.intrinsics.cx": 64.0, "camera.intrinsics.cy": 48.0}}
    if depth:
        feats["observation.images.cam_high_depth"] = {"dtype": "video", "shape": [H, W, 1],
                                                      "info": {"video.is_depth_map": True, "video.codec": "ffv1",
                                                               "depth_scale": 0.001}}
    info = {"codebase_version": "v2.1", "robot_type": "dual_franka_fr3", "fps": FPS, "total_episodes": 2,
            "chunks_size": 1000, "features": feats,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "camera_extrinsics": {"cam_high": {"T_base_cam": np.eye(4).tolist()}}}
    (up / "meta").mkdir()
    (up / "meta" / "info.json").write_text(json.dumps(info))
    (up / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "fold the towel in half"}) + "\n")
    (up / "meta" / "subtasks.jsonl").write_text("\n".join(json.dumps({"subtask_index": i, "subtask": s}) for i, s in
                                                          enumerate(["grasp corners", "fold over", "smooth"])) + "\n")
    eps_meta = []
    for e in range(2):
        q = joints(t, 14, e)
        st = np.concatenate([q[:, :7], gripper(t)[:, None], q[:, 7:], gripper(t)[:, None]], 1)
        vel = np.gradient(st, axis=0) * FPS
        eff = 0.5 * rng.normal(size=(N, 16)) + 2.0
        act = st + 0.01
        lead = st + 0.005
        ee = np.concatenate([np.c_[0.4 + 0.05 * np.sin(t), 0.2 + 0 * t, 0.3 + 0.02 * t, 0 * t, 0 * t, 0 * t,
                                   1 + 0 * t]] * 2, 1)
        r = rl(N)
        sub = np.where(t < 1.0, 0, np.where(t < 2.0, 1, 2))
        df = pd.DataFrame({
            "observation.state": lst(st), "observation.velocity": lst(vel), "observation.effort": lst(eff),
            "action": lst(act), "observation.leader_state": lst(lead), "observation.ee_pose": lst(ee),
            "observation.gripper_width": lst(np.c_[gripper(t), gripper(t)]), "observation.wrench.left": lst(wrench(t)),
            "teleop.intervention": flag(t).astype(bool), "next.reward": r["reward"],
            "next.done": r["is_last"].astype(bool),
            "next.success": r["success"].astype(bool), "is_first": r["is_first"].astype(bool),
            "is_last": r["is_last"].astype(bool), "is_terminal": r["is_terminal"].astype(bool),
            "discount": r["discount"],
            "language_instruction": ["fold the towel in half"] * N, "subtask_index": sub,
            "recorder_time_ns": T0_NS + (np.arange(N) * 33_333_333),
            "observation.audio": lst(audio_chunks(N)),
            "timestamp": t.astype(np.float32), "frame_index": np.arange(N), "episode_index": e,
            "index": np.arange(N) + e * N, "task_index": 0})
        p = up / "data" / "chunk-000" / f"episode_{e:06d}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p)
        for k, c in enumerate(cams):
            write_mp4(up / "videos" / "chunk-000" / c / f"episode_{e:06d}.mp4", N, 50 + 40 * k)
        if depth:
            write_depth_mkv(up / "videos" / "chunk-000" / "observation.images.cam_high_depth"
                            / f"episode_{e:06d}.mkv", N)
        eps_meta.append({"episode_index": e, "tasks": ["fold the towel in half"], "length": N,
                         "operator_id": "op_07", "success": e == 0, "quality_score": 0.82 - 0.4 * e,
                         "robot_serial": "SN-4471", "date": "2026-09-30",
                         "scene_tags": ["kitchen_island", "towel_blue"]})
    (up / "meta" / "episodes.jsonl").write_text("\n".join(json.dumps(x) for x in eps_meta) + "\n")
    fields(d, [
        ("joints (2x7-DoF + gripper)", "observation.state [16]", "observation.state", "state"),
        ("joints", "observation.velocity [16]", "observation.velocity", None),
        ("joints", "observation.effort [16]", "observation.effort", None),
        ("action", "action [16] commanded joint targets", "action", "action"),
        ("teleop", "observation.leader_state [16]", "observation.leader_state", None),
        ("teleop", "teleop.intervention (bool)", "teleop.intervention", None),
        ("EE pose", "observation.ee_pose [14] xyz+quat x2", "observation.ee_pose", None),
        ("gripper", "observation.gripper_width [2]", "observation.gripper_width", None),
        ("force-torque", "observation.wrench.left [6]", "observation.wrench.left", None),
        ("RL", "next.reward", "next.reward", None), ("RL", "next.done", "next.done", None),
        ("RL", "next.success", "next.success", None), ("RL", "is_first", "is_first", None),
        ("RL", "is_last", "is_last", None), ("RL", "is_terminal", "is_terminal", None),
        ("RL", "discount", "discount", None),
        ("language", "per-episode task (tasks.jsonl)", "fold the towel in half", None),
        ("language", "language_instruction column (string per step)", "language_instruction", None),
        ("language", "subtask_index + meta/subtasks.jsonl", "subtask", None),
        ("episode metadata", "operator_id (episodes.jsonl)", "op_07", None),
        ("episode metadata", "success (episodes.jsonl)", "\"success\"", None),
        ("episode metadata", "quality_score (episodes.jsonl)", "quality_score", None),
        ("episode metadata", "robot_serial (episodes.jsonl)", "SN-4471", None),
        ("episode metadata", "date (episodes.jsonl)", "2026-09-30", None),
        ("episode metadata", "scene_tags (episodes.jsonl)", "kitchen_island", None),
        ("episode metadata", "robot_type (info.json)", "dual_franka_fr3", None),
        ("timestamps", "recorder_time_ns column", "recorder_time_ns", None),
        ("audio", "observation.audio [533] per-frame chunk", "observation.audio", None),
        ("depth", "observation.images.cam_high_depth (ffv1 gray16)", "cam_high_depth", "depth"),
        ("camera calibration", "intrinsics in feature info", "camera.intrinsics", None),
        ("camera calibration", "camera_extrinsics in info.json", "camera_extrinsics", None),
        ("segmentation", "observation.images.cam_high_mask video", "cam_high_mask", "camera"),
    ])


# ------------------------------------------------------------------ 2. LeRobot v2.1, single arm 6-DoF + gripper

def lerobot_v21_single_arm(root: Path):
    import pandas as pd
    d, up = fresh(root, "lerobot_v21_single_arm_6dof")
    names7 = [f"joint{j}" for j in range(1, 7)] + ["gripper"]
    feats = {"observation.state": {"dtype": "float32", "shape": [7], "names": names7},
             "action": {"dtype": "float32", "shape": [7], "names": names7},
             "action.delta_ee": {"dtype": "float32", "shape": [7],
                                 "names": ["dx", "dy", "dz", "drx", "dry", "drz", "grip"]},
             "observation.ee_pos": {"dtype": "float32", "shape": [3], "names": ["x", "y", "z"]},
             "observation.ee_quat": {"dtype": "float32", "shape": [4], "names": ["qx", "qy", "qz", "qw"]},
             "observation.ft_wrist": {"dtype": "float32", "shape": [6], "names": ["fx", "fy", "fz", "tx", "ty", "tz"]},
             "observation.images.front": {"dtype": "video", "shape": [H, W, 3]},
             "observation.images.wrist": {"dtype": "video", "shape": [H, W, 3]},
             "timestamp": {"dtype": "float32", "shape": [1]}, "frame_index": {"dtype": "int64", "shape": [1]},
             "episode_index": {"dtype": "int64", "shape": [1]}, "index": {"dtype": "int64", "shape": [1]},
             "task_index": {"dtype": "int64", "shape": [1]}}
    info = {"codebase_version": "v2.1", "robot_type": "so100", "fps": FPS, "total_episodes": 1, "chunks_size": 1000,
            "features": feats,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"}
    (up / "meta").mkdir()
    (up / "meta" / "info.json").write_text(json.dumps(info))
    (up / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "put the cube in the bowl"}) + "\n")
    (up / "meta" / "episodes.jsonl").write_text(json.dumps({"episode_index": 0, "tasks": ["put the cube in the bowl"],
                                                            "length": N}) + "\n")
    st = np.c_[joints(t, 6), 30 * gripper(t)]
    dee = np.c_[0.01 * np.cos(t), 0.0 * t, -0.005 * t, 0 * t, 0 * t, 0.01 + 0 * t, gripper(t)]
    df = pd.DataFrame({"observation.state": lst(st), "action": lst(st + 0.02), "action.delta_ee": lst(dee),
                       "observation.ee_pos": lst(np.c_[0.3 + 0.05 * np.sin(t), 0 * t, 0.2 + 0 * t]),
                       "observation.ee_quat": lst(np.c_[0 * t, 0 * t, np.sin(0.1 * t), np.cos(0.1 * t)]),
                       "observation.ft_wrist": lst(wrench(t)),
                       "timestamp": t.astype(np.float32), "frame_index": np.arange(N), "episode_index": 0,
                       "index": np.arange(N), "task_index": 0})
    p = up / "data" / "chunk-000" / "episode_000000.parquet"
    p.parent.mkdir(parents=True)
    df.to_parquet(p)
    for k, c in enumerate(("observation.images.front", "observation.images.wrist")):
        write_mp4(up / "videos" / "chunk-000" / c / "episode_000000.mp4", N, 60 + 50 * k)
    fields(d, [
        ("joints (6-DoF + gripper)", "observation.state [7]", "observation.state", "state"),
        ("action", "action [7] joint targets", "action", "action"),
        ("action", "action.delta_ee [7] delta EE", "action.delta_ee", None),
        ("EE pose", "observation.ee_pos [3]", "observation.ee_pos", None),
        ("EE pose", "observation.ee_quat [4]", "observation.ee_quat", None),
        ("force-torque", "observation.ft_wrist [6]", "observation.ft_wrist", None),
        ("language", "per-episode task", "put the cube in the bowl", None),
    ])


# ---------------------------------------------------- 3. LeRobot v3.0, humanoid with dexterous hands on a base

def lerobot_v30_humanoid(root: Path):
    import pandas as pd
    d, up = fresh(root, "lerobot_v30_humanoid_base_ft")
    n_ep = 2
    names26 = ([f"{s}_arm_j{j}" for s in ("left", "right") for j in range(1, 8)]
               + [f"{s}_hand_{f}" for s in ("left", "right")
                  for f in ("thumb_yaw", "thumb_pitch", "index", "middle", "ring", "pinky")])
    cams = ["observation.images.head", "observation.images.left_wrist", "observation.images.right_wrist",
            "observation.images.gelsight_right", "observation.images.head_seg"]
    feats = {
        "observation.state": {"dtype": "float32", "shape": [26], "names": names26},
        "action": {"dtype": "float32", "shape": [26], "names": names26},
        "observation.base.odom": {"dtype": "float32", "shape": [5], "names": ["x", "y", "yaw", "vx", "wz"]},
        "observation.base.imu": {"dtype": "float32", "shape": [10],
                                 "names": ["ax", "ay", "az", "gx", "gy", "gz", "qw", "qx", "qy", "qz"]},
        "observation.base.wheel_speed": {"dtype": "float32", "shape": [4], "names": ["fl", "fr", "rl", "rr"]},
        "observation.wrench.left_wrist": {"dtype": "float32", "shape": [6],
                                          "names": ["fx", "fy", "fz", "tx", "ty", "tz"]},
        "observation.wrench.right_wrist": {"dtype": "float32", "shape": [6],
                                           "names": ["fx", "fy", "fz", "tx", "ty", "tz"]},
        "observation.tactile.left_palm": {"dtype": "float32", "shape": [16, 16]},
        "observation.finger_force.right": {"dtype": "float32", "shape": [5, 3]},
        "observation.hand_keypoints.left": {"dtype": "float32", "shape": [21, 3]},
        "observation.body_skeleton": {"dtype": "float32", "shape": [24, 3]},
        "observation.point_cloud": {"dtype": "float32", "shape": [1024, 3]},
        "timestamp": {"dtype": "float32", "shape": [1]}, "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]}, "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
    }
    for c in cams:
        feats[c] = {"dtype": "video", "shape": [H, W, 3]}
    info = {"codebase_version": "v3.0", "robot_type": "unitree_g1_dex3_on_wheeled_base", "fps": FPS,
            "total_episodes": n_ep, "chunks_size": 1000, "features": feats,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}
    (up / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (up / "meta" / "info.json").write_text(json.dumps(info))
    pd.DataFrame({"task_index": [0]}, index=pd.Index(["wipe the table and drive to the dock"], name="task")).to_parquet(
        up / "meta" / "tasks.parquet")
    frames, ep_rows = [], []
    for e in range(n_ep):
        st = joints(t, 26, e)
        fr = pd.DataFrame({
            "observation.state": lst(st), "action": lst(st + 0.01), "observation.base.odom": lst(odom(t)),
            "observation.base.imu": lst(imu(t)), "observation.base.wheel_speed": lst(wheels(t)),
            "observation.wrench.left_wrist": lst(wrench(t)), "observation.wrench.right_wrist": lst(wrench(t)),
            "observation.tactile.left_palm": [x.tolist() for x in pressure(t)],
            "observation.finger_force.right": [x.tolist() for x in finger_forces(t)],
            "observation.hand_keypoints.left": [x.tolist() for x in keypoints(t)],
            "observation.body_skeleton": [x.tolist() for x in keypoints(t, 24)],
            "observation.point_cloud": [x.tolist() for x in rng.normal(size=(N, 1024, 3))],
            "timestamp": t.astype(np.float32), "frame_index": np.arange(N), "episode_index": e,
            "index": np.arange(N) + e * N, "task_index": 0})
        frames.append(fr)
        row = {"episode_index": e, "length": N, "tasks": ["wipe the table and drive to the dock"],
               "data/chunk_index": 0, "data/file_index": 0, "operator_id": "op_11", "success": True}
        for c in cams:
            row.update({f"videos/{c}/chunk_index": 0, f"videos/{c}/file_index": 0,
                        f"videos/{c}/from_timestamp": e * N / FPS, f"videos/{c}/to_timestamp": (e + 1) * N / FPS})
        ep_rows.append(row)
    p = up / "data" / "chunk-000" / "file-000.parquet"
    p.parent.mkdir(parents=True)
    pd.concat(frames, ignore_index=True).to_parquet(p)
    pd.DataFrame(ep_rows).to_parquet(up / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    for k, c in enumerate(cams):
        write_mp4(up / "videos" / c / "chunk-000" / "file-000.mp4", N * n_ep, 40 + 35 * k)
    fields(d, [
        ("humanoid joints", "observation.state [26] arms + dex hands", "observation.state", "state"),
        ("action", "action [26]", "action", "action"),
        ("mobile base", "observation.base.odom [5] x y yaw vx wz", "observation.base.odom", None),
        ("mobile base", "observation.base.imu [10] accel gyro quat", "observation.base.imu", None),
        ("mobile base", "observation.base.wheel_speed [4]", "observation.base.wheel_speed", None),
        ("force-torque", "observation.wrench.left_wrist [6]", "observation.wrench.left_wrist", None),
        ("force-torque", "observation.wrench.right_wrist [6]", "observation.wrench.right_wrist", None),
        ("tactile", "observation.tactile.left_palm [16x16]", "observation.tactile.left_palm", None),
        ("tactile", "observation.finger_force.right [5x3]", "observation.finger_force.right", None),
        ("tactile", "observation.images.gelsight_right video", "gelsight_right", "camera"),
        ("mocap", "observation.hand_keypoints.left [21x3]", "observation.hand_keypoints.left", None),
        ("mocap", "observation.body_skeleton [24x3]", "observation.body_skeleton", None),
        ("point cloud", "observation.point_cloud [1024x3]", "observation.point_cloud", None),
        ("segmentation", "observation.images.head_seg video", "head_seg", "camera"),
        ("language", "per-episode task (tasks.parquet)", "wipe the table and drive to the dock", None),
        ("episode metadata", "operator_id in meta/episodes parquet", "op_11", None),
        ("episode metadata", "robot_type", "unitree_g1", None),
    ])


# ------------------------------------------------------------------ 4. LeRobot v2.1, handheld gripper (UMI-style)

def lerobot_v21_handheld(root: Path):
    import pandas as pd
    d, up = fresh(root, "lerobot_v21_handheld_quat")
    names8 = ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper_width"]
    feats = {"observation.state": {"dtype": "float32", "shape": [8], "names": names8},
             "action": {"dtype": "float32", "shape": [8], "names": names8},
             "observation.imu": {"dtype": "float32", "shape": [6], "names": ["ax", "ay", "az", "gx", "gy", "gz"]},
             "observation.images.gripper_cam": {"dtype": "video", "shape": [H, W, 3]},
             "timestamp": {"dtype": "float32", "shape": [1]}, "frame_index": {"dtype": "int64", "shape": [1]},
             "episode_index": {"dtype": "int64", "shape": [1]}, "index": {"dtype": "int64", "shape": [1]},
             "task_index": {"dtype": "int64", "shape": [1]}}
    info = {"codebase_version": "v2.1", "robot_type": "umi", "fps": FPS, "total_episodes": 1, "chunks_size": 1000,
            "features": feats,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"}
    (up / "meta").mkdir()
    (up / "meta" / "info.json").write_text(json.dumps(info))
    (up / "meta" / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "pick up the mug"}) + "\n")
    (up / "meta" / "episodes.jsonl").write_text(json.dumps({"episode_index": 0, "tasks": ["pick up the mug"],
                                                            "length": N}) + "\n")
    st = np.c_[0.3 + 0.05 * np.sin(t), 0 * t, 0.2 + 0.01 * t, 0 * t, 0 * t, np.sin(0.1 * t), np.cos(0.1 * t),
               gripper(t)]
    df = pd.DataFrame({"observation.state": lst(st), "action": lst(st), "observation.imu": lst(imu(t)[:, :6]),
                       "timestamp": t.astype(np.float32), "frame_index": np.arange(N), "episode_index": 0,
                       "index": np.arange(N), "task_index": 0})
    p = up / "data" / "chunk-000" / "episode_000000.parquet"
    p.parent.mkdir(parents=True)
    df.to_parquet(p)
    write_mp4(up / "videos" / "chunk-000" / "observation.images.gripper_cam" / "episode_000000.mp4", N, 90)
    fields(d, [
        ("EE pose (handheld)", "observation.state [8] xyz+quat+width", "observation.state", "state"),
        ("action", "action [8]", "action", "action"),
        ("IMU", "observation.imu [6]", "observation.imu", None),
    ])


# ---------------------------------------------------- 5. HDF5, ALOHA-style with many sensors and own clocks

def hdf5_aloha(root: Path):
    import h5py
    d, up = fresh(root, "hdf5_aloha_multirate")
    with h5py.File(up / "episode_0.hdf5", "w") as f:
        f.attrs.update({"sim": False, "operator": "op_03", "success": True, "quality_score": 0.9,
                        "robot_serial": "ALOHA-SN-22", "date": "2026-09-28", "scene_tags": "lab_bench,cups",
                        "language_instruction": "stack the red cup on the blue cup", "fps": 30})
        o = f.create_group("observations")
        q = joints(t, 12)
        st = np.c_[q[:, :6], gripper(t), q[:, 6:], gripper(t)]
        o["qpos"] = st
        o["qvel"] = np.gradient(st, axis=0) * FPS
        o["effort"] = 0.4 * rng.normal(size=(N, 14)) + 1.0
        f["action"] = st + 0.01
        o["timestamps"] = T0_NS + np.arange(N) * 33_333_333
        for k, cam in enumerate(("cam_high", "cam_left_wrist", "cam_right_wrist")):
            o.create_dataset(f"images/{cam}", data=np.stack([frame(i, 50 + 40 * k) for i in range(N)]))
        dd = o.create_dataset("depth/cam_high",
                              data=np.stack([np.full((H, W), 800 + 5 * i, np.uint16) for i in range(N)]))
        dd.attrs["depth_scale"] = 0.001
        o.create_dataset("images/gelsight_left", data=np.stack([frame(i, 140) for i in range(N)]))
        o.create_dataset("seg_mask", data=np.stack([(frame(i)[..., 0] > 100).astype(np.uint8) for i in range(N)]))
        o["point_cloud"] = rng.normal(size=(N, 2048, 3)).astype(np.float32)
        o["tactile_left"] = pressure(t)
        o["intervention"] = flag(t)
        # fast sensors on their own clocks
        tf = np.arange(3 * 1000) / 1000.0
        o["ft_left"] = wrench(tf)
        o["ft_timestamps"] = T0_NS + (tf * 1e9).astype(np.int64)
        ti = np.arange(3 * 200) / 200.0
        o["imu_base"] = imu(ti)
        o["imu_timestamps"] = T0_NS + (ti * 1e9).astype(np.int64)
        a = o.create_dataset("audio", data=(0.01 * rng.normal(size=48000)).astype(np.float32))
        a.attrs["sample_rate"] = 16000
        c = f.create_group("calibration/cam_high")
        c["intrinsics"] = np.array([[400, 0, 64], [0, 400, 48], [0, 0, 1]], np.float64)
        c["extrinsics"] = np.eye(4)
        r = rl(N)
        for k in ("reward", "discount", "is_first", "is_last", "is_terminal"):
            f[k] = r[k]
        f["subtasks"] = np.array([(0.0, 1.0, b"reach"), (1.0, 2.0, b"grasp"), (2.0, 3.0, b"stack")],
                                 dtype=[("start_s", "f8"), ("end_s", "f8"), ("label", "S16")])
    fields(d, [
        ("joints (2x 6-DoF + gripper)", "observations/qpos [14]", "qpos", "state"),
        ("joints", "observations/qvel [14]", "qvel", None),
        ("joints", "observations/effort [14]", "effort", None),
        ("action", "action [14]", "action", "action"),
        ("depth", "observations/depth/cam_high uint16 + depth_scale", "depth/cam_high", "depth"),
        ("tactile", "images/gelsight_left (uint8 frames)", "gelsight_left", "camera"),
        ("tactile", "tactile_left [16x16]", "tactile_left", None),
        ("segmentation", "seg_mask (N,H,W) uint8", "seg_mask", "camera"),
        ("point cloud", "point_cloud [2048x3]", "point_cloud", None),
        ("teleop", "intervention flag", "intervention", None),
        ("force-torque", "ft_left [6] @1 kHz, own clock", "ft_left", None),
        ("IMU", "imu_base [10] @200 Hz, own clock", "imu_base", None),
        ("audio", "audio 1-D 48000 samples + sample_rate attr", "audio", None),
        ("camera calibration", "calibration/cam_high/intrinsics 3x3", "intrinsics", None),
        ("camera calibration", "calibration/cam_high/extrinsics 4x4", "extrinsics", None),
        ("RL", "reward", "reward", None), ("RL", "discount", "discount", None),
        ("RL", "is_first", "is_first", None), ("RL", "is_last", "is_last", None),
        ("RL", "is_terminal", "is_terminal", None),
        ("language", "language_instruction attr", "stack the red cup", None),
        ("language", "subtasks compound (start,end,label)", "grasp", None),
        ("episode metadata", "operator attr", "op_03", None),
        ("episode metadata", "success attr", "attribute success", None),
        ("episode metadata", "quality_score attr", "quality_score", None),
        ("episode metadata", "robot_serial attr", "ALOHA-SN-22", None),
        ("episode metadata", "date attr", "2026-09-28", None),
        ("episode metadata", "scene_tags attr", "lab_bench", None),
        ("timestamps", "observations/timestamps (ns recorder clock)", "timestamps", None),
    ])


# ------------------------------------------------------------------ 6. HDF5, robomimic layout

def hdf5_robomimic(root: Path):
    import h5py
    d, up = fresh(root, "hdf5_robomimic")
    with h5py.File(up / "lift_ph.hdf5", "w") as f:
        g = f.create_group("data")
        g.attrs["env_args"] = json.dumps({"env_name": "Lift", "type": 1, "env_kwargs": {"control_freq": 20}})
        g.attrs["total"] = 2 * N
        for e in range(2):
            dg = g.create_group(f"demo_{e}")
            dg.attrs["num_samples"] = N
            dg.attrs["model_file"] = "<mujoco model='lift'>" + "x" * 200 + "</mujoco>"
            ob = dg.create_group("obs")
            ob["agentview_image"] = np.stack([frame(i, 70) for i in range(N)])
            ob["robot0_eye_in_hand_image"] = np.stack([frame(i, 150) for i in range(N)])
            ob["robot0_eef_pos"] = np.c_[0.1 * np.sin(t), 0 * t, 0.9 + 0 * t]
            ob["robot0_eef_quat"] = np.c_[0 * t, 1 + 0 * t, 0 * t, 0 * t]
            ob["robot0_gripper_qpos"] = np.c_[gripper(t) / 2, -gripper(t) / 2]
            ob["robot0_joint_pos"] = joints(t, 7)
            ob["robot0_joint_vel"] = np.gradient(joints(t, 7), axis=0) * FPS
            ob["object"] = rng.normal(size=(N, 10))
            dg["actions"] = np.c_[0.1 * np.cos(t), 0 * t, 0 * t, 0 * t, 0 * t, 0 * t,
                                  np.where(gripper(t) < 0.05, 1, -1)]
            r = rl(N)
            dg["rewards"] = r["reward"]
            dg["dones"] = r["is_last"]
            dg["states"] = rng.normal(size=(N, 45))
        f["mask/train"] = np.array([b"demo_0", b"demo_1"])
    fields(d, [
        ("EE pose", "obs/robot0_eef_pos [3]", "robot0_eef_pos", None),
        ("EE pose", "obs/robot0_eef_quat [4]", "robot0_eef_quat", None),
        ("gripper", "obs/robot0_gripper_qpos [2]", "robot0_gripper_qpos", None),
        ("joints (7-DoF)", "obs/robot0_joint_pos [7]", "robot0_joint_pos", "state"),
        ("joints", "obs/robot0_joint_vel [7]", "robot0_joint_vel", None),
        ("action", "actions [7] delta EE + grip", "actions", "action"),
        ("RL", "rewards", "rewards", None), ("RL", "dones", "dones", None),
        ("sim state", "states [45]", "states", None),
        ("episode metadata", "env_args attr (control_freq 20)", "env_args", None),
        ("episode metadata", "model_file attr", "model_file", None),
        ("episode metadata", "mask/train split", "mask/train", None),
        ("timestamps", "no clock, 20 Hz only in env_args", "clock_note", None),
    ])


# ------------------------------------------------------------------ 7. MCAP JSON (Foxglove schemas), mobile manipulator

def mcap_json(root: Path):
    from mcap.writer import Writer
    d, up = fresh(root, "mcap_json_mobile_manip")
    path = up / "run_0001.mcap"
    dur = 3.0
    msgs = []                               # (t_s, topic, dict)

    def every(rate, topic, fn, start=-0.02, end=dur + 0.02):
        k = 0
        while start + k / rate <= end:
            ts = start + k / rate
            msgs.append((ts, topic, fn(ts, k)))
            k += 1
    tt = lambda x: np.array([x])
    every(FPS, "/camera/front/image", lambda s, k: {"timestamp": {"sec": 0, "nsec": 0}, "frame_id": "front",
                                                    "format": "jpeg",
                                                    "data": base64.b64encode(jpeg(frame(min(k, N - 1), 60))).decode()},
          start=0.0, end=dur - 1e-3)
    every(FPS, "/camera/wrist_right/image",
          lambda s, k: {"format": "jpeg", "data": base64.b64encode(jpeg(frame(min(k, N - 1), 150))).decode()},
          start=0.0, end=dur - 1e-3)
    every(FPS, "/gelsight/right/image",
          lambda s, k: {"format": "jpeg", "data": base64.b64encode(jpeg(frame(min(k, N - 1), 200))).decode()},
          start=0.0, end=dur - 1e-3)
    every(FPS, "/camera/front/mask",
          lambda s, k: {"format": "jpeg", "data": base64.b64encode(jpeg(frame(min(k, N - 1), 0))).decode()},
          start=0.0, end=dur - 1e-3)
    every(FPS, "/camera/front/depth",
          lambda s, k: {"width": W, "height": H, "encoding": "16UC1", "step": W * 2,
                        "data": base64.b64encode(np.full((H, W), 900 + k, np.uint16).tobytes()).decode()},
          start=0.0, end=dur - 1e-3)
    every(FPS, "/camera/front/camera_info",
          lambda s, k: {"width": W, "height": H, "K": [400, 0, 64, 0, 400, 48, 0, 0, 1],
                        "D": [0.1, -0.05, 0, 0, 0], "distortion_model": "plumb_bob"})
    every(100, "/right_arm/joint_state", lambda s, k: {"joint_pos": joints(tt(s), 6)[0].tolist(),
                                                       "joint_vel": (0.24 * np.cos(0.8 * s + np.arange(6))).tolist(),
                                                       "joint_effort": (1 + 0.1 * rng.normal(size=6)).tolist(),
                                                       "gripper_pos": [float(gripper(tt(s))[0])]})
    every(100, "/right_arm/leader/joint_pos",
          lambda s, k: {"joint_pos": (joints(tt(s), 6)[0] + 0.01).tolist() + [float(gripper(tt(s))[0])]})
    every(100, "/right_arm/ee_pose", lambda s, k: {"position": {"x": 0.4 + 0.05 * np.sin(s), "y": 0.0, "z": 0.3},
                                                   "orientation": {"x": 0.0, "y": 0.0, "z": float(np.sin(0.05 * s)),
                                                                   "w": float(np.cos(0.05 * s))}})
    every(100, "/right_arm/gripper", lambda s, k: {"width": float(gripper(tt(s))[0])})
    every(1000, "/right_arm/wrist_ft",
          lambda s, k: (lambda w: {"force": dict(zip("xyz", map(float, w[:3]))),
                                   "torque": dict(zip("xyz", map(float, w[3:])))})(wrench(tt(s))[0]))
    every(50, "/base/odom", lambda s, k: {"pose": {"x": float(odom(tt(s))[0, 0] if s < 1.5 else 0.3), "y": 0.01 * s,
                                                   "yaw": 0.1 * s},
                                          "twist": {"linear_x": 0.2 if s < 1.5 else 0.0, "angular_z": 0.1}})
    every(200, "/base/imu", lambda s, k: (lambda m: {"linear_acceleration": dict(zip("xyz", map(float, m[:3]))),
                                                      "angular_velocity": dict(zip("xyz", map(float, m[3:6]))),
                                                      "orientation": dict(zip("wxyz", map(float, m[6:])))})(
        imu(tt(s))[0]))
    every(50, "/base/wheels", lambda s, k: {"speeds": wheels(tt(s))[0].tolist()})
    every(FPS, "/teleop/intervention", lambda s, k: {"active": bool(1.0 <= s < 1.8)})
    every(FPS, "/rl/step", lambda s, k: {"reward": 1.0 if s > 2.95 else 0.0, "discount": 0.0 if s > 2.95 else 1.0,
                                         "is_first": k == 0, "is_last": s > 2.95, "is_terminal": s > 2.95,
                                         "success": s > 2.95})
    every(31.25, "/audio/mic", lambda s, k: {"sample_rate": 16000,
                                             "samples": (0.01 * rng.normal(size=512)
                                                         + (0.5 if 2.2 < s < 2.4 else 0)).tolist()})
    every(FPS, "/hand/right/keypoints",
          lambda s, k: {"points": [dict(zip("xyz", map(float, p))) for p in keypoints(tt(s))[0]]})
    every(10, "/lidar/points", lambda s, k: {"points": rng.normal(size=(2048, 3)).tolist()})
    every(60, "/tactile/right/pressure", lambda s, k: {"pressure": pressure(tt(s))[0].tolist()})
    every(60, "/tactile/right/fingers",
          lambda s, k: {"forces": [dict(zip("xyz", map(float, p))) for p in finger_forces(tt(s))[0]]})
    every(FPS, "/recorder/stats", lambda s, k: {"seq": k, "recv_time_ns": int(T0_NS + s * 1e9), "dropped_frames": 0})
    msgs.append((0.0, "/task", {"data": "pick up the red cube and drop it in the bin"}))
    every(FPS, "/language_instruction", lambda s, k: {"data": "pick up the red cube"})
    for s, x in ((0.0, "drive to the table"), (1.0, "grasp the cube"), (2.0, "drop it in the bin")):
        msgs.append((s, "/task/subtask", {"data": x}))
    msgs.append((0.0, "/episode_info", {"operator_id": "op_21", "success": True, "quality_score": 0.75,
                                        "robot_serial": "MM-SN-0912", "date": "2026-09-29",
                                        "scene_tags": ["warehouse", "bin_blue"]}))
    msgs.sort(key=lambda m: m[0])
    with open(path, "wb") as fh:
        w = Writer(fh)
        w.start()
        chans = {}
        for _, topic, _ in msgs:
            if topic in chans:
                continue
            sname = ("foxglove.CompressedImage" if topic.endswith(("image", "mask")) else
                     "foxglove.RawImage" if topic.endswith("depth") else
                     "foxglove.CameraCalibration" if topic.endswith("camera_info") else
                     "std_msgs/String" if topic in ("/task", "/language_instruction", "/task/subtask") else
                     topic.strip("/").replace("/", "_"))
            sid = w.register_schema(name=sname, encoding="jsonschema", data=b"{}")
            chans[topic] = w.register_channel(topic=topic, message_encoding="json", schema_id=sid)
        for s, topic, m in msgs:
            ns = int(T0_NS + s * 1e9)
            w.add_message(chans[topic], log_time=ns, publish_time=ns, data=json.dumps(m).encode())
        w.add_metadata("episode", {"operator_id": "op_21_meta", "success": "true", "scene": "warehouse_meta"})
        w.finish()
    fields(d, [
        ("joints (6-DoF + gripper)", "/right_arm/joint_state joint_pos[6]+gripper_pos", "joint_pos", "state"),
        ("joints", "/right_arm/joint_state joint_vel", "joint_vel", None),
        ("joints", "/right_arm/joint_state joint_effort", "joint_effort", None),
        ("teleop", "/right_arm/leader/joint_pos [7] (leader arm)", "/right_arm/leader/joint_pos", "action"),
        ("teleop", "/teleop/intervention active (bool)", "/teleop/intervention", None),
        ("EE pose", "/right_arm/ee_pose position+orientation", "/right_arm/ee_pose", None),
        ("gripper", "/right_arm/gripper width", "/right_arm/gripper", None),
        ("force-torque", "/right_arm/wrist_ft @1 kHz", "/right_arm/wrist_ft", None),
        ("mobile base", "/base/odom @50 Hz", "/base/odom", None),
        ("mobile base", "/base/imu @200 Hz", "/base/imu", None),
        ("mobile base", "/base/wheels speeds[4]", "/base/wheels", None),
        ("RL", "/rl/step reward", "/rl/step reward", None),
        ("RL", "/rl/step is_first/is_last/is_terminal/success (bools)", "is_terminal", None),
        ("audio", "/audio/mic samples[512] + sample_rate", "/audio/mic", None),
        ("camera calibration", "/camera/front/camera_info K, D @30 Hz", "camera_info", None),
        ("mocap", "/hand/right/keypoints 21 x {x,y,z}", "/hand/right/keypoints", None),
        ("point cloud", "/lidar/points 2048x3 @10 Hz", "/lidar/points", None),
        ("tactile", "/tactile/right/pressure 16x16", "/tactile/right/pressure", None),
        ("tactile", "/tactile/right/fingers 5 x {x,y,z}", "/tactile/right/fingers", None),
        ("tactile", "/gelsight/right/image (CompressedImage)", "/gelsight/right/image", "camera"),
        ("depth", "/camera/front/depth RawImage 16UC1", "/camera/front/depth", "depth"),
        ("segmentation", "/camera/front/mask (CompressedImage)", "/camera/front/mask", "camera"),
        ("timestamps", "/recorder/stats seq, recv_time_ns", "/recorder/stats", None),
        ("language", "/task (once)", "pick up the red cube and drop it in the bin", None),
        ("language", "/language_instruction per step", "/language_instruction", None),
        ("language", "/task/subtask steps", "grasp the cube", None),
        ("episode metadata", "/episode_info JSON message (once)", "op_21", None),
        ("episode metadata", "MCAP metadata record 'episode'", "op_21_meta", None),
    ])


# ------------------------------------------------------------------ 8. MCAP ROS 2 (CDR), bimanual Franka + base

HDR = """================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""
V3 = """================================================================================
MSG: geometry_msgs/Vector3
float64 x
float64 y
float64 z
"""
QUAT = """================================================================================
MSG: geometry_msgs/Quaternion
float64 x
float64 y
float64 z
float64 w
"""
DEFS = {
    "sensor_msgs/msg/JointState": ("std_msgs/Header header\nstring[] name\nfloat64[] position\nfloat64[] velocity\n"
                                   "float64[] effort\n" + HDR),
    "sensor_msgs/msg/Imu": ("std_msgs/Header header\ngeometry_msgs/Quaternion orientation\n"
                            "float64[9] orientation_covariance\n"
                            "geometry_msgs/Vector3 angular_velocity\nfloat64[9] angular_velocity_covariance\n"
                            "geometry_msgs/Vector3 linear_acceleration\nfloat64[9] linear_acceleration_covariance\n"
                            + HDR + QUAT + V3),
    "geometry_msgs/msg/WrenchStamped": ("std_msgs/Header header\ngeometry_msgs/Wrench wrench\n" + HDR
                                        + "========================================"
                                          "========================================\n"
                                          "MSG: geometry_msgs/Wrench\ngeometry_msgs/Vector3 force\n"
                                          "geometry_msgs/Vector3 torque\n" + V3),
    "nav_msgs/msg/Odometry": ("std_msgs/Header header\nstring child_frame_id\ngeometry_msgs/PoseWithCovariance pose\n"
                              "geometry_msgs/TwistWithCovariance twist\n" + HDR
                              + "================================================================================\n"
                                "MSG: geometry_msgs/PoseWithCovariance\ngeometry_msgs/Pose pose\n"
                                "float64[36] covariance\n"
                                "================================================================================\n"
                                "MSG: geometry_msgs/Pose\ngeometry_msgs/Point position\n"
                                "geometry_msgs/Quaternion orientation\n"
                                "================================================================================\n"
                                "MSG: geometry_msgs/Point\nfloat64 x\nfloat64 y\nfloat64 z\n" + QUAT
                              + "================================================================================\n"
                                "MSG: geometry_msgs/TwistWithCovariance\ngeometry_msgs/Twist twist\n"
                                "float64[36] covariance\n"
                                "================================================================================\n"
                                "MSG: geometry_msgs/Twist\ngeometry_msgs/Vector3 linear\n"
                                "geometry_msgs/Vector3 angular\n" + V3),
    "sensor_msgs/msg/Image": ("std_msgs/Header header\nuint32 height\nuint32 width\nstring encoding\n"
                              "uint8 is_bigendian\nuint32 step\nuint8[] data\n" + HDR),
    "sensor_msgs/msg/CameraInfo": ("std_msgs/Header header\nuint32 height\nuint32 width\nstring distortion_model\n"
                                   "float64[] d\n"
                                   "float64[9] k\nfloat64[9] r\nfloat64[12] p\nuint32 binning_x\nuint32 binning_y\n"
                                   "sensor_msgs/RegionOfInterest roi\n" + HDR
                                   + "========================================"
                                     "========================================\n"
                                     "MSG: sensor_msgs/RegionOfInterest\nuint32 x_offset\nuint32 y_offset\n"
                                     "uint32 height\nuint32 width\nbool do_rectify\n"),
    "std_msgs/msg/String": "string data\n",
}


def mcap_ros2(root: Path):
    from mcap_ros2.writer import Writer
    d, up = fresh(root, "mcap_ros2_franka_base")
    path = up / "bag_0.mcap"
    dur = 3.0
    msgs = []

    def hdr(s):
        ns = int(T0_NS + s * 1e9)
        return {"stamp": {"sec": ns // 10**9, "nanosec": ns % 10**9}, "frame_id": "base"}

    def every(rate, topic, typ, fn, start=-0.02, end=dur + 0.02):
        k = 0
        while start + k / rate <= end:
            s = start + k / rate
            msgs.append((s, topic, typ, fn(s, k)))
            k += 1
    tt = lambda x: np.array([x])
    every(FPS, "/camera/color/image_raw", "sensor_msgs/msg/Image",
          lambda s, k: {"header": hdr(s), "height": H, "width": W, "encoding": "rgb8", "is_bigendian": 0, "step": W * 3,
                        "data": frame(min(k, N - 1), 80).tobytes()}, start=0.0, end=dur - 1e-3)
    every(FPS, "/camera/depth/image_rect_raw", "sensor_msgs/msg/Image",
          lambda s, k: {"header": hdr(s), "height": H, "width": W, "encoding": "16UC1", "is_bigendian": 0,
                        "step": W * 2,
                        "data": np.full((H, W), 700 + k, np.uint16).tobytes()}, start=0.0, end=dur - 1e-3)
    every(FPS, "/camera/color/camera_info", "sensor_msgs/msg/CameraInfo",
          lambda s, k: {"header": hdr(s), "height": H, "width": W, "distortion_model": "plumb_bob",
                        "d": [0.1, -0.05, 0, 0, 0],
                        "k": [400, 0, 64, 0, 400, 48, 0, 0, 1], "r": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        "p": [400, 0, 64, 0, 0, 400, 48, 0, 0, 0, 1, 0], "binning_x": 0, "binning_y": 0,
                        "roi": {"x_offset": 0, "y_offset": 0, "height": 0, "width": 0, "do_rectify": False}})
    for side, ph in (("left", 0.0), ("right", 1.0)):
        every(100, f"/{side}/joint_states", "sensor_msgs/msg/JointState",
              lambda s, k, ph=ph, side=side: {"header": hdr(s), "name": [f"fr3_{side}_joint{j}" for j in range(1, 8)],
                                              "position": joints(tt(s), 7, ph)[0].tolist(),
                                              "velocity": (0.24 * np.cos(0.8 * s + np.arange(7) + ph)).tolist(),
                                              "effort": (2 + 0.1 * rng.normal(size=7)).tolist()})
        every(100, f"/{side}/franka_gripper/joint_states", "sensor_msgs/msg/JointState",
              lambda s, k: {"header": hdr(s), "name": ["finger1", "finger2"],
                            "position": [float(gripper(tt(s))[0] / 2)] * 2, "velocity": [0.0, 0.0],
                            "effort": [0.0, 0.0]})
        every(100, f"/{side}/joint_command", "sensor_msgs/msg/JointState",
              lambda s, k, ph=ph: {"header": hdr(s), "name": [], "position": (joints(tt(s), 7, ph)[0] + 0.01).tolist(),
                                   "velocity": [], "effort": []})
    every(1000, "/left/ft_sensor/wrench", "geometry_msgs/msg/WrenchStamped",
          lambda s, k: (lambda w: {"header": hdr(s), "wrench": {"force": dict(zip("xyz", map(float, w[:3]))),
                                                                 "torque": dict(zip("xyz", map(float, w[3:])))}})(
              wrench(tt(s))[0]))
    every(200, "/base/imu", "sensor_msgs/msg/Imu",
          lambda s, k: (lambda m: {"header": hdr(s), "orientation": dict(zip("wxyz", map(float, m[6:]))),
                                   "orientation_covariance": [0.0] * 9,
                                   "angular_velocity": dict(zip("xyz", map(float, m[3:6]))),
                                   "angular_velocity_covariance": [0.0] * 9,
                                   "linear_acceleration": dict(zip("xyz", map(float, m[:3]))),
                                   "linear_acceleration_covariance": [0.0] * 9})(imu(tt(s))[0]))
    every(50, "/odom", "nav_msgs/msg/Odometry",
          lambda s, k: {"header": hdr(s), "child_frame_id": "base_link",
                        "pose": {"pose": {"position": {"x": 0.2 * min(s, 1.5), "y": 0.01 * s, "z": 0.0},
                                          "orientation": {"x": 0.0, "y": 0.0, "z": float(np.sin(0.05 * s)),
                                                          "w": float(np.cos(0.05 * s))}},
                                 "covariance": [0.0] * 36},
                        "twist": {"twist": {"linear": {"x": 0.2 if s < 1.5 else 0.0, "y": 0.0, "z": 0.0},
                                            "angular": {"x": 0.0, "y": 0.0, "z": 0.1}}, "covariance": [0.0] * 36}})
    msgs.append((0.0, "/instruction", "std_msgs/msg/String",
                 {"data": "hand the bottle from the left arm to the right arm"}))
    msgs.sort(key=lambda m: m[0])
    with open(path, "wb") as fh:
        w = Writer(fh)
        schemas = {typ: w.register_msgdef(typ, DEFS[typ]) for typ in sorted({m[2] for m in msgs})}
        for s, topic, typ, m in msgs:
            ns = int(T0_NS + s * 1e9)
            w.write_message(topic, schemas[typ], m, log_time=ns, publish_time=ns)
        w.finish()
    fields(d, [
        ("joints (2x 7-DoF, no gripper in vector)", "/left|right/joint_states position[7]", "/left/joint_states",
         "state"),
        ("joints", "/left/joint_states velocity[7]", "/left/joint_states velocity", None),
        ("joints", "/left/joint_states effort[7]", "/left/joint_states effort", None),
        ("gripper", "/left/franka_gripper/joint_states position[2]", "franka_gripper", None),
        ("action", "/left|right/joint_command position[7]", "joint_command", "action"),
        ("force-torque", "/left/ft_sensor/wrench @1 kHz", "/left/ft_sensor/wrench", None),
        ("mobile base", "/base/imu sensor_msgs/Imu @200 Hz", "/base/imu", None),
        ("mobile base", "/odom nav_msgs/Odometry @50 Hz", "/odom", None),
        ("depth", "/camera/depth/image_rect_raw 16UC1", "/camera/depth/image_rect_raw", "depth"),
        ("camera calibration", "/camera/color/camera_info @30 Hz", "camera_info", None),
        ("language", "/instruction std_msgs/String", "hand the bottle", None),
    ])


# ---------------------------------------------------- 9. plain video folder with sidecar metadata and CSVs

def video_folder(root: Path):
    d, up = fresh(root, "video_folder_sidecars")
    ep = up / "ep_001"
    write_mp4(ep / "exo_cam.mp4", N, 70)
    write_mp4(ep / "wrist_right_cam.mp4", N, 160)
    (ep / "meta.json").write_text(json.dumps({
        "task": "wipe the whiteboard", "operator_id": "op_42", "success": False, "quality_score": 0.4,
        "robot_serial": "VID-SN-7", "date": "2026-09-27", "scene_tags": ["office", "whiteboard"],
        "subtasks": [{"start_s": 0.0, "end_s": 1.5, "label": "grab eraser"},
                     {"start_s": 1.5, "end_s": 3.0, "label": "wipe"}],
        "intrinsics": {"fx": 400, "fy": 400, "cx": 64, "cy": 48}}))
    import pandas as pd
    tf = np.arange(3000) / 1000.0
    w = wrench(tf)
    pd.DataFrame({"timestamp": tf, **{k: w[:, i] for i, k in enumerate(("fx", "fy", "fz", "tx", "ty", "tz"))}}).to_csv(
        ep / "ft.csv", index=False)
    ti = np.arange(600) / 200.0
    m = imu(ti)
    pd.DataFrame({"t": ti, **{k: m[:, i] for i, k in enumerate(("ax", "ay", "az", "gx", "gy", "gz",
                                                                 "qw", "qx", "qy", "qz"))}}
                 ).to_csv(ep / "imu.csv", index=False)
    fields(d, [
        ("force-torque", "ft.csv @1 kHz (timestamp column)", "ft", None),
        ("IMU", "imu.csv @200 Hz (t column)", "imu", None),
        ("language", "meta.json task", "wipe the whiteboard", None),
        ("language", "meta.json subtasks", "grab eraser", None),
        ("episode metadata", "meta.json operator_id", "op_42", None),
        ("episode metadata", "meta.json success", "\"success\"", None),
        ("episode metadata", "meta.json quality_score", "quality_score", None),
        ("episode metadata", "meta.json robot_serial", "VID-SN-7", None),
        ("episode metadata", "meta.json scene_tags", "whiteboard", None),
        ("camera calibration", "meta.json intrinsics", "intrinsics", None),
    ])


# ------------------------------------------------------------------ 10. HDF5 head camera (ego) with hand and body mocap

def hdf5_ego(root: Path):
    import h5py
    d, up = fresh(root, "hdf5_ego_mocap")
    with h5py.File(up / "session_07.hdf5", "w") as f:
        f.attrs["task"] = "chop the carrot"
        f.attrs["participant_id"] = "P-0193"
        f["rgb"] = np.stack([frame(i, 90) for i in range(N)])
        f["timestamps_ns"] = T0_NS + np.arange(N) * 33_333_333
        f["hands/left_keypoints"] = keypoints(t)
        f["hands/right_keypoints"] = keypoints(t)
        f["body/skeleton"] = keypoints(t, 24)
        f["head_pose"] = np.c_[0.01 * t, 0 * t, 1.6 + 0 * t, np.cos(0.05 * t), 0 * t, 0 * t, np.sin(0.05 * t)]
        f["gaze"] = np.c_[0.5 + 0.1 * np.sin(t), 0.5 + 0 * t]
        f["audio_chunks"] = audio_chunks(N)
        f["right_glove_pressure"] = pressure(t)
        f["annotations/subtasks"] = np.array([(0.0, 1.2, b"pick up knife"), (1.2, 3.0, b"chop")],
                                             dtype=[("start", "f8"), ("end", "f8"), ("label", "S16")])
    fields(d, [
        ("mocap", "hands/left_keypoints [21x3]", "left_keypoints", None),
        ("mocap", "hands/right_keypoints [21x3]", "right_keypoints", None),
        ("mocap", "body/skeleton [24x3]", "body/skeleton", None),
        ("head pose", "head_pose [7]", "head_pose", None),
        ("gaze", "gaze [2]", "gaze", None),
        ("audio", "audio_chunks [533] per frame", "audio_chunks", None),
        ("tactile", "right_glove_pressure [16x16]", "right_glove_pressure", None),
        ("language", "task attr", "chop the carrot", None),
        ("language", "annotations/subtasks compound", "pick up knife", None),
        ("episode metadata", "participant_id attr", "P-0193", None),
    ])


CASES = {"lerobot_v21_bimanual_7dof": (lerobot_v21_bimanual, "teleop_arms"),
         "lerobot_v21_bimanual_7dof_nodepth": (lambda r: lerobot_v21_bimanual(r, depth=False), "teleop_arms"),
         "lerobot_v21_single_arm_6dof": (lerobot_v21_single_arm, "teleop_arms"),
         "lerobot_v30_humanoid_base_ft": (lerobot_v30_humanoid, "teleop_arms"),
         "lerobot_v21_handheld_quat": (lerobot_v21_handheld, "handheld_gripper"),
         "hdf5_aloha_multirate": (hdf5_aloha, "teleop_arms"),
         "hdf5_robomimic": (hdf5_robomimic, "teleop_arms"),
         "mcap_json_mobile_manip": (mcap_json, "teleop_arms"),
         "mcap_ros2_franka_base": (mcap_ros2, "teleop_arms"),
         "video_folder_sidecars": (video_folder, "teleop_arms"),
         "hdf5_ego_mocap": (hdf5_ego, "ego_head")}


def build(root: Path, name: str) -> tuple[Path, str]:
    """(upload folder, rig) of one case, built under root."""
    fn, rig = CASES[name]
    fn(root)
    return root / name / "upload", rig
