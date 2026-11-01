"""Prepare BitRobot/HIW-500-LeRobot episodes (a teleoperated Unitree G1 humanoid in homes) as episode sidecars
(rig teleop_arms, video plus signals).

    python -m prepare hiw500 sample --list LIST --hours 100 [--raw RAW]
    python -m prepare hiw500 prepare --episodes LIST --out EPISODES [--raw RAW] [--jobs N] [--force]

An episode list has one episode index per line. HIW-500 is LeRobot v3: episodes are packed back to back into shared
AV1 mp4s per camera (about 10 minutes each) and shared data parquets. Its meta/info.json counts 23,743 episodes and
40.8M frames at 30 fps, which is 378 hours, not the 500 the card names. `prepare` downloads, into RAW, the meta/
folder, the data parquets and the video packs the listed episodes need, then writes EPISODES/episode_<index>/.

The head camera is one 1280x480 video holding the two eyes of a stereo pair side by side. The two halves are the
same view a few centimetres apart, so only the left eye is shown. Each camera's episode is cut from its pack into
the episode's own H.264 file (head.mp4 cropped to the left half, left_wrist.mp4, right_wrist.mp4), every frame on the
30 fps grid and the count checked against the state length, so RAW can be deleted once a batch is prepared.

observation.state is the G1's 29 joint positions (legs, waist, arms), which our arm checks do not read, so the episode
is labelled from the video and every recorded column reaches the model as a named signal: the joints, and the
whole-body controller's state and command (observation.state.wbc and action: pivot_vx, pivot_vy and pivot_vyaw are
the commanded walking velocity of the base, pivot_height its height, then each hand's pose and its trigger and squeeze).

Annotation: each episode has one task (meta/tasks.parquet), given as the instruction, and LeRobot's
language_persistent column holds its timed subtasks (move to bed, pick pillow, place pillow to bed, ...), each with a
start time. The first entry repeats the task at time 0 and is the instruction, not a subtask; each other entry runs
until the next one starts, the last until the episode ends. They are given as the dataset's timed steps to check.

`sample` reads meta/ only and picks whole video packs spread across each task's pack range, one pack per task per
pass, until --hours, so every downloaded byte is used. It is deterministic. The dataset needs no token.
"""
from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from label.atomic import write_atomic
from label.presented import presented_pts
from prepare import cli
from prepare import hub
from prepare.formats import scalar
from prepare.molmo import _spread_permutation

REPO = "BitRobot/HIW-500-LeRobot"
FPS = 30
VKEYS = {"exo": "head", "left": "left_wrist", "right": "right_wrist"}
EYE_W = 640       # the left eye: x 0..639 of the 1280-wide head frame
CAMERA_DESC = {
    "exo": ("the robot's head camera (left eye of its stereo pair), looking forward and down from the humanoid's "
            "head. Both arms and grippers reach in from the bottom, and its own legs show when it looks down."),
    "left": ("the camera on the LEFT wrist, looking along that arm's two black rubber gripper fingers, which are at "
             "the bottom of its image."),
    "right": ("the camera on the RIGHT wrist, looking along that arm's two black rubber gripper fingers, which are at "
              "the bottom of its image."),
}
ROBOT = ("Unitree G1 humanoid: two legs, a waist and two 7-DoF arms, each ending in a two-finger gripper with black "
         "rubber fingers; teleoperated by a person in VR in real homes")
NOTE = ("This is the dataset's task for the whole episode. The dataset's timed steps name the walks to each place "
        "(move to the bed, move to the trash can) as steps of the task.")


def ensure_meta(raw: Path) -> Path:
    meta = raw / "meta"
    if not (meta / "info.json").exists():
        from huggingface_hub import snapshot_download
        snapshot_download(REPO, repo_type="dataset", local_dir=str(raw), allow_patterns=["meta/*", "meta/**"],
                          token=hub.token())
    return meta


def load_episodes(meta: Path) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p) for p in sorted((meta / "episodes").glob("chunk-*/*.parquet"))],
                     ignore_index=True)


def task_of(row) -> str:
    t = row["tasks"]
    return str(t[0]) if hasattr(t, "__len__") and not isinstance(t, str) else str(t)


def pack_of(row, view: str) -> tuple[int, int]:
    k = f"videos/observation.images.{VKEYS[view]}"
    return int(scalar(row[f"{k}/chunk_index"])), int(scalar(row[f"{k}/file_index"]))


def pack_rel(view: str, chunk: int, file: int) -> str:
    return f"videos/observation.images.{VKEYS[view]}/chunk-{chunk:03d}/file-{file:03d}.mp4"


def data_rel(row) -> str:
    return f"data/chunk-{int(scalar(row['data/chunk_index'])):03d}/file-{int(scalar(row['data/file_index'])):03d}.parquet"


def sample_packs(ep: pd.DataFrame, hours: float) -> list[int]:
    """Whole head packs spread across each task's pack range, round robin across tasks, until hours."""
    per_task = {}
    for _, r in ep.iterrows():
        per_task.setdefault(task_of(r), {}).setdefault(pack_of(r, "exo"), []).append(
            (int(scalar(r["episode_index"])), int(scalar(r["length"])) / FPS))
    state = {t: {"keys": sorted(p), "order": _spread_permutation(len(p)), "pos": 0} for t, p in per_task.items()}
    chosen, total, moved = [], 0.0, True
    while total < hours * 3600 and moved:
        moved = False
        for t in sorted(state):
            st = state[t]
            if st["pos"] >= len(st["keys"]):
                continue
            key = st["keys"][st["order"][st["pos"]]]
            st["pos"] += 1
            # a pack can hold episodes of several tasks; only this task's episodes in it are taken
            for eid, s in per_task[t][key]:
                chosen.append(eid)
                total += s
            moved = True
            if total >= hours * 3600:
                break
    return sorted(set(chosen))


def subtasks(entries, task: str, duration_s: float) -> tuple[list[dict], list[dict]]:
    """language_persistent's timed subtasks as annotation_subtasks, and the steps the recording ends before. The
    entry repeating the task at 0 is the instruction, not a step. A step runs until the next one starts; one that
    starts at or after the recording's end is not a timed step of it (a sample of 2,031 episodes had them in about
    13% of normal-speed episodes, the recording stopping before the last steps) and is returned apart, as the dataset
    wrote it."""
    items = sorted(((float(e["timestamp"]), str(e["content"]).strip()) for e in entries if e.get("content")),
                   key=lambda x: x[0])
    items = [x for x in items if not (x[0] == 0.0 and x[1] == task)]
    inside, after = [], []
    for k, (t0, label) in enumerate(items):
        if t0 >= duration_s:
            after.append({"t0": round(t0, 2), "label": label})
            continue
        t1 = items[k + 1][0] if k + 1 < len(items) else duration_s
        inside.append({"t0": round(t0, 2), "t1": round(min(t1, duration_s), 2), "label": label})
    return inside, after


def cut(pack: Path, base_s: float, n: int, out: Path, key: str, crop: str | None = None) -> dict:
    """The episode's n frames from base_s (cropped when crop is given) as H.264 on the 30 fps grid; refuses any other
    count. Each episode gets its own files, so the packs can be deleted once a batch is prepared."""
    import av
    tmp = out.with_suffix(".tmp.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{base_s:.6f}", "-i", str(pack), "-frames:v", str(n)]
                   + (["-vf", crop] if crop else []) +
                   ["-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-g", str(FPS),
                    "-pix_fmt", "yuv420p", "-an", str(tmp)], check=True)
    with av.open(str(tmp)) as c:
        s = c.streams.video[0]
        pts = presented_pts(c, s)
        tb, w, h = s.time_base, s.codec_context.width, s.codec_context.height
    step = 1 / (FPS * float(tb))
    on_grid = bool(pts) and all(abs((p - pts[0]) - k * step) < 0.5 for k, p in enumerate(pts))
    if len(pts) != n or not on_grid:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{key}: {len(pts)} frames (grid {on_grid}) for {n} state rows")
    os.replace(tmp, out)
    src = {"packed": str(out.resolve()), "base_s": 0.0, "n_frames": n, "camera_key": key, "codec": "h264",
           "width": w, "height": h, "source_offset_s": round(base_s, 6)}
    if crop:
        src["crop"] = f"left eye, x 0-{EYE_W - 1} of the 1280-wide frame"
    return src


def prepare_one(row, raw: Path, out_root: Path, info: dict, force: bool) -> str:
    eidx = int(scalar(row["episode_index"]))
    ep_dir = out_root / f"episode_{eidx:06d}"
    if not force and (ep_dir / "context.json").exists():
        return "skip"
    n = int(scalar(row["length"]))
    df = pd.read_parquet(raw / data_rel(row))
    df = df[df["episode_index"] == eidx].sort_values("frame_index").reset_index(drop=True)
    if not (df["frame_index"].to_numpy() == np.arange(len(df))).all() or len(df) != n:
        raise ValueError(f"episode {eidx}: {len(df)} rows, frame_index not 0..{n - 1}")
    ep_dir.mkdir(parents=True, exist_ok=True)
    sources = {}
    for view in VKEYS:
        k = f"videos/observation.images.{VKEYS[view]}"
        base_s = float(scalar(row[f"{k}/from_timestamp"]))
        pk = raw / pack_rel(view, *pack_of(row, view))
        sources[view] = cut(pk, base_s, n, ep_dir / f"{VKEYS[view]}.mp4", k,
                            f"crop={EYE_W}:480:0:0" if view == "exo" else None)
    task = task_of(row)
    steps, after = subtasks(df["language_persistent"].iloc[0], task, n / FPS)
    note = NOTE
    if after:
        note += (" The dataset's annotation also lists steps that start after this recording ends ("
                 + ", ".join(f"{s['label']} at {s['t0']:.1f}s" for s in after)
                 + "); the recording stops before them.")
    feats = info.get("features") or {}
    from prepare import formats
    names = formats.value_names(feats.get("observation.state", {}).get("names"), 29)
    kind, layout_note = formats.state_layout(29, "teleop_arms", names)
    ctx = {
        "dataset": REPO,
        "profile": "teleop_arms",
        "whole_body": True,        # walks, turns and crouches between places (label/prompts.py WHOLE_BODY_WORDING)
        "state_kind": kind,
        "episode_index": eidx,
        "robot_type": ROBOT,
        "fps": FPS,
        "task_label": [task],
        "instruction": task,
        "instruction_note": note,
        "annotation_subtasks": steps,
        "annotation_steps_after_end": after,
        "n_state_frames": n,
        "cameras": {v: {"key": f"observation.images.{VKEYS[v]}", "name": VKEYS[v],
                        "width": EYE_W if v == "exo" else 640, "height": 480, "desc": CAMERA_DESC[v]}
                    for v in VKEYS},
        "source": {"episode_index": eidx, "head_stereo": "left eye of the 1280x480 side-by-side pair"},
    }
    if layout_note:
        formats.no_state(ctx, formats.StateNote(layout_note, "layout"))
    skip = {"language_persistent", "language_events", "task_index"}
    formats.write_signals(ep_dir, ctx, formats.recorded_signals(df, skip, n, features=feats))
    (ep_dir / "sources.json").write_text(json.dumps(sources, indent=2))
    (ep_dir / "instruction.txt").write_text(task + "\n")
    write_atomic(ep_dir / "context.json", ctx, indent=2)
    return "ok"


def ensure_files(rows, raw: Path, jobs: int) -> None:
    rels = sorted({pack_rel(v, *pack_of(r, v)) for r in rows for v in VKEYS} | {data_rel(r) for r in rows})
    todo = [r for r in rels if not (raw / r).exists()]
    print(f"files needed {len(rels)}, to download {len(todo)}", flush=True)
    with ThreadPoolExecutor(jobs) as ex:
        for f in as_completed([ex.submit(hub.download, REPO, r, raw) for r in todo]):
            f.result()


def main() -> int:
    ap, sub = cli.parser("hiw500", __doc__)
    s = sub.add_parser("sample", help="choose whole video packs for about --hours and write their episode list")
    s.add_argument("--list", type=Path, required=True)
    s.add_argument("--raw", type=Path, default=Path("data/raw/hiw500"))
    s.add_argument("--hours", type=float, required=True)
    cli.add_prepare(sub, "hiw500", "one episode index per line", jobs=4)
    a = ap.parse_args()
    meta = ensure_meta(a.raw)
    ep = load_episodes(meta)
    if a.cmd == "sample":
        chosen = sample_packs(ep, a.hours)
        sel = ep[ep["episode_index"].isin(chosen)]
        a.list.write_text("\n".join(map(str, chosen)) + "\n")
        print(json.dumps({"episodes": len(chosen), "hours": round(sel["length"].sum() / FPS / 3600, 2),
                          "tasks": sel["tasks"].map(lambda t: str(t[0])).value_counts().to_dict()}))
        return 0
    want = {int(x) for x in cli.read_list(a.episodes)}
    rows = [r for _, r in ep[ep["episode_index"].isin(want)].iterrows()]
    info = json.loads((meta / "info.json").read_text())
    ensure_files(rows, a.raw, a.jobs)
    a.out.mkdir(parents=True, exist_ok=True)
    return cli.run(rows, lambda r: prepare_one(r, a.raw, a.out, info, a.force), a.jobs)


if __name__ == "__main__":
    raise SystemExit(main())
