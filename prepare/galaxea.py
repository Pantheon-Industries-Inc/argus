"""Prepare RogersPyke/Galaxea-Open-World-Dataset_10K_20260123 episodes (a teleoperated Galaxea R1 Lite) as
episode sidecars (rig teleop_arms, state joints).

    python -m prepare galaxea sample --out EPISODES --hours 5.5 --seed 71 [--list LIST] [--raw RAW] [--jobs N]
    python -m prepare galaxea prepare --episodes configs/slices/galaxea.txt --out EPISODES [--raw RAW] [--jobs N]
        [--force]

An episode list has one <collection folder>/<episode index> per line. The repo holds one LeRobot v2.1 dataset per
collection folder (task name plus date). For each episode this downloads, into RAW/<folder>/, the folder's
meta/info.json, meta/tasks.jsonl and meta/episodes.jsonl, the episode's parquet (per-frame state and action at
15 fps) and its three AV1 mp4s, and writes EPISODES/episode_<folder>_<index>/ with context.json, sources.json
(pointing at the downloaded mp4s), state.npz and instruction.txt. The views are the head camera (left eye of the
stereo pair) as the scene view and the two wrist cameras; each mp4 must be on the exact 15 fps pts grid and
cover every state row. The per-arm state is 6 joint positions plus the gripper.

Annotation: every frame carries a coarse task (the episode's task) and a fine task index (the timed sub-step
being performed, bilingual "Chinese@English"), plus a quality tag. The coarse task is given as the instruction;
the timed sub-steps and the quality tag are given as the dataset's claims to check.

`sample` visits the collection folders round robin in seeded order and prepares one random episode from each per
pass until --hours (an episode's length is known only once its parquet is read), then writes the list of what it
prepared to LIST (default EPISODES.txt). The dataset needs no token.
"""
from __future__ import annotations

import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from prepare import cli
from prepare import hub

REPO = "RogersPyke/Galaxea-Open-World-Dataset_10K_20260123"
FPS = 15
VIDEO_KEYS = {"exo": "observation.images.head_rgb", "left": "observation.images.left_wrist_rgb",
              "right": "observation.images.right_wrist_rgb"}
CAMERA_DESC = {
    "exo": ("the robot's head camera (left eye of its stereo pair), above and behind the two arms, looking down "
            "at the workspace. The head sits on the torso of a mobile robot, so this view moves when the torso "
            "tilts or the base drives."),
    "left": "the camera on the LEFT arm's wrist, looking along its gripper fingers.",
    "right": "the camera on the RIGHT arm's wrist, looking along its gripper fingers.",
}


def english(task: str) -> str:
    return task.split("@", 1)[1].strip() if "@" in task else task.strip()


def _dl(rel: str, raw: Path) -> Path:
    return Path(hub.download(REPO, rel, raw))


def folder_meta(folder: str, raw: Path) -> dict:
    tasks = {}
    for line in _dl(f"{folder}/meta/tasks.jsonl", raw).read_text().splitlines():
        if line.strip():
            t = json.loads(line)
            tasks[int(t["task_index"])] = t["task"]
    lines = _dl(f"{folder}/meta/episodes.jsonl", raw).read_text().splitlines()
    eps = [json.loads(s) for s in lines if s.strip()]
    info = json.loads(_dl(f"{folder}/meta/info.json", raw).read_text())
    return {"folder": folder, "tasks": tasks, "episodes": eps, "info": info}


def episode_dir_name(folder: str, index: int) -> str:
    """Plug_Into_A_Fixed_Socket_20250710_006, 69 -> episode_Plug_Into_A_Fixed_Socket_20250710_006_000069"""
    return f"episode_{re.sub(r'[^A-Za-z0-9]+', '_', folder).strip('_')}_{index:06d}"


def spans(idx: np.ndarray) -> list[tuple[int, int, int]]:
    """Runs of equal values: (start frame, end frame inclusive, value)."""
    out, a = [], 0
    for k in range(1, len(idx) + 1):
        if k == len(idx) or idx[k] != idx[a]:
            out.append((a, k - 1, int(idx[a])))
            a = k
    return out


def prepare_one(meta: dict, ep: dict, raw: Path, out_root: Path, force: bool = False) -> str:
    import av
    import pandas as pd
    folder, i = meta["folder"], int(ep["episode_index"])
    info = meta["info"]
    chunk = i // int(info.get("chunks_size") or 1000)
    name = episode_dir_name(folder, i)
    ep_dir = out_root / name
    if not force and (ep_dir / "context.json").exists():
        return "skip"
    rel = info["data_path"].format(episode_chunk=chunk, episode_index=i)
    df = pd.read_parquet(_dl(f"{folder}/{rel}", raw))
    n = len(df)
    sources, checks = {}, {"video_frames": {}}
    for v, key in VIDEO_KEYS.items():
        p = _dl(f"{folder}/" + info["video_path"].format(episode_chunk=chunk, video_key=key, episode_index=i), raw)
        with av.open(str(p)) as c:
            s = c.streams.video[0]
            pts = sorted(pk.pts for pk in c.demux(s) if pk.pts is not None)
            w, h, tb = s.codec_context.width, s.codec_context.height, s.time_base
        step = int(round(1 / (FPS * float(tb))))
        on_grid = bool(pts) and pts[0] == 0 and all(pp == k * step for k, pp in enumerate(pts))
        checks["video_frames"][v] = {"frames": len(pts), "state_frames": n, "on_15fps_grid": on_grid}
        if len(pts) < n or not on_grid:
            raise RuntimeError(f"{v}: {len(pts)} frames (grid {on_grid}) for {n} state rows")
        sources[v] = {"packed": str(p.resolve()), "base_s": 0.0, "n_frames": n, "camera_key": key,
                      "codec": "av1", "width": w, "height": h}
    col = lambda k: np.stack(df[k].values).astype(np.float64).reshape(len(df), -1)   # grippers are stored as scalars
    state = np.concatenate([col("observation.state.left_arm"), col("observation.state.left_gripper"),
                            col("observation.state.right_arm"), col("observation.state.right_gripper")], axis=1)
    action = np.concatenate([col("action.left_arm"), col("action.left_gripper"),
                             col("action.right_arm"), col("action.right_gripper")], axis=1)
    tasks = meta["tasks"]
    # a frame may point at a task index the folder's tasks.jsonl does not define; that is kept and
    # said, since an annotation that references nothing is itself a defect of the dataset
    name_of = lambda t: english(tasks[t]) if t in tasks else f"(task index {t}, missing from the dataset's task list)"
    coarse = name_of(int(df["coarse_task_index"].iloc[0]))
    subs = [{"t0": round(a / FPS, 2), "t1": round((b + 1) / FPS, 2), "label": name_of(t)}
            for a, b, t in spans(df["task_index"].to_numpy())]
    checks["missing_task_indices"] = sorted({int(t) for t in pd.unique(df["task_index"]) if int(t) not in tasks}
                                            | ({int(df["coarse_task_index"].iloc[0])} - set(tasks)))
    quality = sorted({tasks.get(int(q), str(q)) for q in pd.unique(df["quality_index"])})
    sub_txt = "; ".join(f"{s['t0']:.1f}-{s['t1']:.1f}s \"{s['label']}\"" for s in subs)
    ctx = {
        "dataset": REPO,
        "profile": "teleop_arms",
        "state_kind": "joints",
        "gripper_value": ("the measured gripper position, about 0 = jaws shut and about 100 = fully open "
                          "(checked against the wrist frames)"),
        "episode_id": name,
        "robot_type": "Galaxea R1 Lite (mobile base and torso, two 6-DoF arms with parallel-jaw grippers)",
        "fps": FPS,
        "task_label": [coarse],
        "instruction": coarse,
        "instruction_note": ("This is the dataset's task for the whole episode. The dataset also labels timed "
                             f"sub-steps for this episode, which are claims to check against the video: {sub_txt}. "
                             f"Its quality tag for the episode: {', '.join(quality)}."),
        "annotation_subtasks": subs,
        "quality_tag": quality,
        "n_state_frames": int(n),
        "source": {"folder": folder, "episode_index": i, "raw_file_name": ep.get("raw_file_name")},
        "cameras": {v: {"key": VIDEO_KEYS[v], "name": {"exo": "head", "left": "left", "right": "right"}[v],
                        "width": sources[v]["width"], "height": sources[v]["height"], "desc": CAMERA_DESC[v]}
                    for v in VIDEO_KEYS},
        "stream_checks": checks,
    }
    ep_dir.mkdir(parents=True, exist_ok=True)
    np.savez(ep_dir / "state.npz", state=state.astype(np.float32), action=action.astype(np.float32))
    (ep_dir / "sources.json").write_text(json.dumps(sources, indent=2))
    (ep_dir / "instruction.txt").write_text(coarse + "\n")
    (ep_dir / "context.json").write_text(json.dumps(ctx, indent=2))
    return "ok"


def drawn(meta: dict, ep: dict, raw: Path, out: Path) -> dict | None:
    """Prepare one drawn episode; its context, or None when it cannot be prepared (printed)."""
    try:
        prepare_one(meta, ep, raw, out)
        return json.loads((out / episode_dir_name(meta["folder"], int(ep["episode_index"])) / "context.json")
                          .read_text())
    except Exception as e:  # a drawn episode that cannot be read is a finding about the data; listed, not hidden
        print(f"FAILED {meta['folder']}/{ep['episode_index']}: {type(e).__name__}: {e}"[:400], file=sys.stderr,
              flush=True)
        return None


def cmd_sample(a) -> int:
    a.out.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(a.seed)
    folders = sorted(x["path"] for x in hub.ls(REPO) if x["type"] == "directory")
    with ThreadPoolExecutor(a.jobs) as ex:
        metas = list(ex.map(lambda f: folder_meta(f, a.raw), folders))
    print(f"{len(metas)} collection folders, {sum(len(m['episodes']) for m in metas)} episodes", flush=True)
    rnd.shuffle(metas)
    pools = {m["folder"]: rnd.sample(m["episodes"], len(m["episodes"])) for m in metas}
    got, secs = [], 0.0
    with ThreadPoolExecutor(a.jobs) as ex:
        while secs < a.hours * 3600 and any(pools.values()):
            batch = [(m, pools[m["folder"]].pop()) for m in metas if pools[m["folder"]]]
            for ctx in ex.map(lambda mp: drawn(mp[0], mp[1], a.raw, a.out), batch):
                if ctx:
                    got.append(ctx)
                    secs += ctx["n_state_frames"] / FPS
            print(f"episodes {len(got)} hours {secs / 3600:.2f}", flush=True)
    listing = sorted(f"{c['source']['folder']}/{c['source']['episode_index']}" for c in got)
    lst = a.list or Path(f"{a.out}.txt")
    lst.write_text("\n".join(listing) + "\n")
    print(json.dumps({"episodes": len(got), "hours": round(secs / 3600, 2), "list": str(lst),
                      "folders": len({c["source"]["folder"] for c in got})}), flush=True)
    return 0


def cmd_prepare(a) -> int:
    by_folder: dict[str, list[int]] = {}
    for it in cli.read_list(a.episodes):
        folder, idx = it.rsplit("/", 1)
        by_folder.setdefault(folder, []).append(int(idx))
    a.out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(a.jobs) as ex:
        metas = dict(zip(by_folder, ex.map(lambda f: folder_meta(f, a.raw), by_folder)))
    jobs = []
    for folder, idxs in by_folder.items():
        eps = {int(e["episode_index"]): e for e in metas[folder]["episodes"]}
        jobs += [(metas[folder], eps[i]) for i in idxs]
    return cli.run(jobs, lambda mp: prepare_one(mp[0], mp[1], a.raw, a.out, a.force), a.jobs)


def main() -> int:
    ap, sub = cli.parser("galaxea", __doc__)
    s = sub.add_parser("sample", help="draw and prepare episodes until about --hours, and write their list")
    s.add_argument("--out", type=Path, required=True, help="where the episode folders are written")
    s.add_argument("--list", type=Path, default=None, help="the episode list to write (default OUT.txt)")
    s.add_argument("--raw", type=Path, default=Path("data/raw/galaxea"), help="where downloaded files are kept")
    s.add_argument("--hours", type=float, default=5.5)
    s.add_argument("--jobs", type=int, default=6, help="episodes prepared at once (default 6)")
    s.add_argument("--seed", type=int, default=71)
    cli.add_prepare(sub, "galaxea", "one <collection folder>/<episode index> per line", jobs=6)
    a = ap.parse_args()
    return cmd_sample(a) if a.cmd == "sample" else cmd_prepare(a)


if __name__ == "__main__":
    raise SystemExit(main())
