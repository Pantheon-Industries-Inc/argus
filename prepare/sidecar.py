"""Writing episode sidecars from LeRobot folders and plain video files, shared by the lerobot, videos, HABIT and
OpenAoE adapters.

An episode sidecar is one folder per episode holding
  context.json     the facts the harness may state: dataset, rig (profile), state kind, fps, cameras, instruction
  sources.json     per camera: the video file (packed), the episode's offset in it (base_s) and its frame count
  state.npz        state and action, one row per anchor-camera frame (absent when there is no usable state)
  times.npz        each camera's real frame times and exact pts, when frames are not on the k / fps grid
  kmap_<view>.npy  for a camera paired to the anchor camera by nearest time, its frame for each anchor frame
  instruction.txt  the instruction, for reading by eye

LeRobot v2.0 / v2.1 (one parquet and one mp4 per camera per episode) and v3.0 (episodes packed into shared
parquet and mp4 files): cameras are the features with dtype "video", assigned to the harness views (exo, left,
right) by name; state is observation.state and action is action. Recorded state is used when it has 7 values
per arm or gripper (6 joints plus gripper for teleop arms; x y z roll pitch yaw plus opening for handheld
grippers); other layouts are labelled from the video alone and the context says so. Plain video: real frame
times from each file's pts. Nothing is re-encoded.
"""
from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path

import numpy as np

STATE_KIND = {"teleop_arms": "joints", "handheld_gripper": "ee_pose"}
EGO_DESC = ("the camera worn on the person's head, looking forward and down at their hands and the work "
            "in front of them")
DEMUXER = {".mp4": "mov", ".mov": "mov", ".m4v": "mov", ".mkv": "matroska", ".webm": "matroska", ".avi": "avi"}


def open_video(path: Path):
    """A video opened only by the demuxer its extension names, so a file that is really a playlist or a
    concat script (which could make ffmpeg read other local files) fails instead of being followed."""
    import av
    fmt = DEMUXER.get(Path(path).suffix.lower())
    if fmt is None:
        raise ValueError(f"{Path(path).name}: not a video type this reads")
    return av.open(str(path), format=fmt)


def inside(root: Path, p: Path) -> Path:
    """p resolved, and required to be under root: dataset metadata may not point outside the dataset."""
    r, q = Path(root).resolve(), Path(p).resolve()
    if q != r and r not in q.parents:
        raise ValueError(f"the dataset refers to a file outside its folder ({p})")
    return q


def probe(path: Path) -> dict:
    """Container facts for one video file: its frames' pts sorted, time base, size, codec."""
    with open_video(path) as c:
        st = c.streams.video[0]
        pts = sorted(p.pts for p in c.demux(st) if p.size and p.pts is not None)
        rate = st.average_rate or st.guessed_rate
        return {"pts": np.asarray(pts, dtype=np.int64), "time_base": st.time_base,
                "width": st.codec_context.width, "height": st.codec_context.height,
                "codec": st.codec_context.name, "fps": float(rate) if rate else None}


def nearest(src_t: np.ndarray, q: np.ndarray) -> np.ndarray:
    """For each time in q, the index of the nearest time in the sorted src_t (the earlier one on a tie)."""
    if len(src_t) == 1:
        return np.zeros(len(q), dtype=np.int32)
    i = np.clip(np.searchsorted(src_t, q), 1, len(src_t) - 1)
    return np.where(np.abs(src_t[i - 1] - q) <= np.abs(src_t[i] - q), i - 1, i).astype(np.int32)


SCENE_WORDS = ("top", "high", "head", "overhead", "exo", "front", "scene", "main", "cam")
FIXED_WORDS = ("top", "high", "overhead", "exterior", "front", "scene", "exo", "zed", "stereo", "low", "static",
               "third")
MOUNTED_WORDS = ("wrist", "hand", "gripper", "arm", "eef", "ee")


def tokens(name: str) -> list[str]:
    """Words of a camera name: split on non-letters and camelCase (cam_left_wrist, leftCam)."""
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return [t for t in re.split(r"[^a-z0-9]+", spaced.lower()) if t]


def side_of(name: str) -> str | None:
    """left / right when a word of the name is or starts with the side (left, leftcam, l), else None."""
    for tk in tokens(name):
        for side in ("left", "right"):
            if tk == side[0] or tk.startswith(side):
                return side
    return None


def is_mount_named(name: str) -> bool:
    return any(t.startswith(w) for t in tokens(name) for w in MOUNTED_WORDS)


def mounted_side(name: str) -> str | None:
    """The side of a camera mounted on an arm or gripper: it names a side, and either names the mount
    (wrist, hand, gripper) or names nothing that says fixed (overhead_left, exterior_image_1_left and the
    left eye of a stereo pair are fixed cameras with a side, not wrist cameras)."""
    side = side_of(name)
    if side is None:
        return None
    if is_mount_named(name):
        return side
    return None if any(t.startswith(w) for t in tokens(name) for w in FIXED_WORDS) else side


def scene_rank(name: str) -> int:
    """Prefer an overhead / head / front camera as the scene camera when several are unnamed."""
    n = name.lower()
    for i, w in enumerate(SCENE_WORDS):
        if w in n:
            return i
    return 99


def assign_views(names: list[str], rig: str) -> tuple[dict, list]:
    """{view: camera name} for up to one scene and two mounted cameras, and the names left unused."""
    out = {}
    # names that also say wrist/hand/gripper take a side before names that only carry a side
    for nm in sorted(names, key=lambda n: (0 if is_mount_named(n) else 1, n)):
        v = mounted_side(nm)
        if v and v not in out:
            out[v] = nm
    rest = sorted((nm for nm in names if nm not in out.values()), key=lambda s: (scene_rank(s), s))
    if rest:
        out["exo"] = rest[0]
    unused = [nm for nm in names if nm not in out.values()]
    if rig == "handheld_gripper" and list(out) == ["exo"]:
        # the only camera of a single handheld gripper is the gripper's own camera, not a scene camera
        out = {"right": out["exo"]}
    return out, unused


def pick_cameras(names: list[str], rig: str) -> tuple[dict, list]:
    """{view: camera name} and the names left unused. A head-camera rig gets exactly one camera, the
    best-named scene camera."""
    if rig != "ego_head":
        return assign_views(names, rig)
    best = sorted(names, key=lambda t: (scene_rank(t), t))[0]
    return {"exo": best}, [t for t in names if t != best]


def camera_entry(view: str, name: str, pr: dict, rig: str) -> dict:
    return describe({"key": name, "name": _short(name, view), "width": pr["width"], "height": pr["height"],
                     "codec": pr["codec"]}, view, name, rig)


def describe(cam: dict, view: str, name: str, rig: str) -> dict:
    """Name and describe the cameras whose role follows from the rig alone: the head camera on a person,
    and the one camera of a single handheld gripper. Everything else keeps the prompt's fallback line
    for its slot (a scene camera, or the camera on the left / right arm or gripper)."""
    if rig == "ego_head" and view == "exo":
        cam.update(name="head", desc=EGO_DESC)
    elif rig == "handheld_gripper" and view == "right" and not re.search("right", name, re.I):
        cam.update(name="gripper", desc="the camera carried on the handheld gripper, looking along its fingers")
    return cam


def _short(name: str, view: str) -> str:
    base = re.sub(r"^(observation\.images\.|observation\.image\.|/)", "", name)
    base = re.sub(r"[^A-Za-z0-9_]+", "_", base).strip("_")
    return base[:32] or view


def state_layout(dims: int, rig: str) -> tuple[str, str | None]:
    """(state_kind, note). 7 or 14 values per frame are 1 or 2 actors of 6 + gripper; anything else is
    labelled from video."""
    if rig == "ego_head":
        return "none", None
    if dims in (7, 14):
        return STATE_KIND[rig], None
    return "none", (f"the recorded state has {dims} values per frame; this pipeline reads 7 per "
                    f"{'arm' if rig == 'teleop_arms' else 'gripper'} (6 + gripper), so the episode was "
                    "labelled from video only")


def finish_episode(ep: Path, ctx: dict, sources: dict, state=None, action=None, times: dict | None = None) -> dict:
    """Write the sidecar files of one episode folder; returns the context as written."""
    ep.mkdir(parents=True, exist_ok=True)
    if state is not None and ctx.get("state_kind") != "none":
        arrs = {"state": np.asarray(state, dtype=np.float32)}
        if action is not None and np.shape(action) == np.shape(state):
            arrs["action"] = np.asarray(action, dtype=np.float32)
        np.savez(ep / "state.npz", **arrs)
    if times:
        np.savez(ep / "times.npz", **times)
        ctx["real_times"] = "times.npz"
    (ep / "sources.json").write_text(json.dumps(sources, indent=1))
    (ep / "instruction.txt").write_text((ctx.get("instruction") or "") + "\n")
    (ep / "context.json").write_text(json.dumps(ctx, indent=1, default=str))
    return ctx


def video_views_episode(ep: Path, files: dict, rig: str, dataset: str, extra: dict) -> dict:
    """An episode made of video files {view: (camera name, path)}: real frame times from each file's pts,
    the first view in harness order as the anchor, the others paired to it by nearest time. Separate video
    files have no common clock, so each is timed from its own first frame."""
    from label.episode import VIEW_ORDER
    prs = {v: probe(p) for v, (_, p) in files.items()}
    order = [v for v in VIEW_ORDER if v in files]
    anchor = order[0]

    def seconds_of(pr):
        t = pr["pts"].astype(np.float64) * float(pr["time_base"])
        return t - t[0]
    ta = seconds_of(prs[anchor])
    ep.mkdir(parents=True, exist_ok=True)
    sources, times, cams = {}, {}, {}
    for v in order:
        name, path = files[v]
        pr = prs[v]
        t = seconds_of(pr)
        times[v], times[f"{v}_pts"] = t, pr["pts"]
        sources[v] = {"packed": str(Path(path).resolve()), "base_s": 0.0, "n_frames": int(len(pr["pts"])),
                      "camera_key": name}
        if v != anchor:
            km = nearest(t, ta)
            if not (len(t) == len(ta) and np.array_equal(km, np.arange(len(ta)))):
                np.save(ep / f"kmap_{v}.npy", km)
                sources[v]["kmap"] = f"kmap_{v}.npy"
        cams[v] = camera_entry(v, name, pr, rig)
    # the rate is measured from the frame times (a header can claim any rate); the length is the span of
    # the frames plus one frame, so it matches what the labeller samples
    step = float(np.median(np.diff(ta))) if len(ta) > 1 else 1 / 30
    fps = 1.0 / step if step > 0 else 30.0
    ctx = {"dataset": dataset, "profile": rig, "state_kind": "none", "episode_id": ep.name,
           "robot_type": None, "fps": round(float(fps), 3), "n_state_frames": int(len(ta)),
           "duration_s": round(float(ta[-1]) + step, 3) if len(ta) else 0.0, "cameras": cams, **extra}
    return finish_episode(ep, ctx, sources, times=times)


def episode_name(s: str) -> str:
    """The episode folder name for a source name: episode_ plus its letters and digits, runs of anything
    else as one underscore (an existing episode_ prefix is not doubled)."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    return "episode_" + (re.sub(r"^episode_", "", s)[:120] or "0")


def episode_dirs(out: Path, names: list[str]) -> list[Path]:
    """One episode folder per source name, in list order. run-1.mp4 and run_1.mp4 both clean to
    episode_run_1: the later one gets _2 (then _3, ...), so no episode overwrites another and a rerun over
    the same list maps every source to the same folder."""
    seen: set[str] = set()
    dirs = []
    for s in names:
        base = episode_name(s)
        d, k = base, 2
        while d in seen:
            d, k = f"{base}_{k}", k + 1
        seen.add(d)
        dirs.append(Path(out) / d)
    return dirs


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def scalar(v):
    """A LeRobot metadata value that may be stored as a one-element array (v3 parquet), as a scalar."""
    return v[0] if hasattr(v, "__len__") and not isinstance(v, str) else v


def plan_lerobot(root: Path) -> list[dict]:
    """The dataset's episodes present on disk, by index: {"name": 6-digit index, "row": its metadata row,
    "seconds": its length}. For v2, an episode listed in the metadata whose parquet is absent (a subset of a
    larger dataset was downloaded) is left out."""
    import pandas as pd
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    v3 = str(info.get("codebase_version", "")).startswith("v3")
    if v3:
        parts = sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))
        eps = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).to_dict("records") if parts else []
    else:
        eps = read_jsonl(root / "meta" / "episodes.jsonl")
    items = []
    for e in sorted(eps, key=lambda r: int(scalar(r["episode_index"]))):
        eidx = int(scalar(e["episode_index"]))
        if not v3:
            chunk = eidx // int(info.get("chunks_size", 1000))
            if not inside(root, root / info["data_path"].format(episode_chunk=chunk, episode_index=eidx)).exists():
                continue
        items.append({"name": f"{eidx:06d}", "row": e, "seconds": int(scalar(e["length"])) / fps})
    return items


def convert_lerobot(item: dict, root: Path, rig: str, out: Path, dataset: str) -> dict:
    """One episode of plan_lerobot into out/episode_<index>; returns its context."""
    import pandas as pd
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    v3 = str(info.get("codebase_version", "")).startswith("v3")
    row = item["row"]
    eidx = int(scalar(row["episode_index"]))
    chunk = eidx // int(info.get("chunks_size", 1000))
    feats = info.get("features", {})
    cams = [k for k, f in feats.items() if f.get("dtype") == "video"]
    bad = [k for k in cams if not re.fullmatch(r"[A-Za-z0-9._-]+", k)]
    if bad:
        raise ValueError(f"camera names with path characters are not accepted: {bad[:3]}")
    if not cams:
        raise ValueError("this LeRobot dataset stores no video features (image-in-parquet datasets are not supported)")
    vmap, unused = pick_cameras(cams, rig)
    if v3:
        dpath = inside(root, root / "data" / f"chunk-{int(scalar(row['data/chunk_index'])):03d}"
                       / f"file-{int(scalar(row['data/file_index'])):03d}.parquet")
    else:
        dpath = inside(root, root / info["data_path"].format(episode_chunk=chunk, episode_index=eidx))
    cols = [c for c in ("observation.state", "action", "frame_index", "episode_index")
            if c in feats or c in ("frame_index", "episode_index")]
    df = pd.read_parquet(dpath, columns=cols)
    df = df[df["episode_index"] == eidx].sort_values("frame_index")
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float32) if "observation.state" in df else None
    action = np.stack(df["action"].to_numpy()).astype(np.float32) if "action" in df else None
    kind, note = state_layout(state.shape[1] if state is not None and state.ndim == 2 else 0, rig)
    if state is None and rig != "ego_head":
        note = "the dataset records no observation.state, so the episode was labelled from video only"
    sources, cameras, grid, offgrid = {}, {}, {}, {}
    for v, key in vmap.items():
        if v3:
            ch = int(scalar(row[f"videos/{key}/chunk_index"]))
            fl = int(scalar(row[f"videos/{key}/file_index"]))
            rel = info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
            mp4 = inside(root, root / rel.format(video_key=key, chunk_index=ch, file_index=fl))
        else:
            mp4 = inside(root, root / info["video_path"].format(episode_chunk=chunk, video_key=key, episode_index=eidx))
        if not mp4.exists():
            raise FileNotFoundError(f"missing video {mp4.relative_to(root.resolve())}")
        if v3:
            base = float(scalar(row[f"videos/{key}/from_timestamp"]))
            to = float(scalar(row[f"videos/{key}/to_timestamp"]))
            n = int(round((to - base) * fps))
        else:
            pr = probe(mp4)
            n, base = int(len(pr["pts"])), 0.0
            step = Fraction(1) / Fraction(fps).limit_denominator(1000) / pr["time_base"]
            grid[v] = bool(step.denominator == 1 and pr["pts"][0] == 0
                           and np.array_equal(pr["pts"], np.arange(n) * int(step)))
            offgrid[v] = pr
        sources[v] = {"packed": str(mp4.resolve()), "base_s": base, "n_frames": n, "camera_key": key}
        vi = feats[key].get("info") or {}
        cameras[v] = describe({"key": key, "name": _short(key, v), "width": vi.get("video.width"),
                               "height": vi.get("video.height"), "codec": vi.get("video.codec")}, v, key, rig)
    tasks = row.get("tasks")
    tasks = ([str(t) for t in tasks] if hasattr(tasks, "__len__") and not isinstance(tasks, str)
             else ([str(tasks)] if tasks else []))
    instruction = "; ".join(t.strip() for t in tasks if t.strip())
    n_frames = len(state) if state is not None else min(s["n_frames"] for s in sources.values())
    ctx = {"dataset": dataset, "profile": rig, "state_kind": kind, "episode_id": episode_name(item["name"]),
           "episode_index": eidx, "robot_type": info.get("robot_type"), "fps": fps,
           "task_label": tasks or [item["name"]], "instruction": instruction,
           "instruction_note": "This instruction is the episode's task text in the dataset's LeRobot metadata.",
           "n_state_frames": int(n_frames), "cameras": cameras,
           "stream_checks": {"frames_on_grid": grid, "episode_length_meta": int(scalar(row["length"]))},
           "source": {"format": f"lerobot {info.get('codebase_version')}", "episode_index": eidx,
                      "unused_cameras": unused}}
    if note:
        ctx["state_note"] = note
    ep = out / ctx["episode_id"]
    times = None
    if not v3 and any(not ok for ok in grid.values()):
        # frames not on the k/fps grid from 0: decode each camera by its own pts, timed by frame index
        # (LeRobot's timestamps are frame_index / fps, and state rows follow frames)
        times = {}
        for v, pr in offgrid.items():
            times[v] = np.arange(len(pr["pts"])) / fps
            times[f"{v}_pts"] = pr["pts"]
    return finish_episode(ep, ctx, sources, state if kind != "none" else None, action, times=times)
