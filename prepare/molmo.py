"""Prepare allenai/MolmoAct2-BimanualYAM-Dataset episodes as episode sidecars (rig teleop_arms, state joints),
without re-encoding video.

    python -m prepare molmo sample --list LIST --hours 25 [--mode pack|episode] [--raw RAW]
    python -m prepare molmo prepare --episodes configs/slices/molmo.txt --out EPISODES [--raw RAW] [--jobs N]
        [--force]

An episode list has one episode index per line. MolmoAct2 is LeRobot v3: episodes are packed back to back into
shared mp4s (one per camera, about 12 to 50 episodes each) and shared data parquets. `prepare` downloads, into
RAW, the dataset's meta/ folder, the data parquets and the video packs the listed episodes need (about 480 MB per
camera per pack), then writes EPISODES/episode_<index>/ with

  sources.json     per camera: the packed mp4, the episode's offset in it (from_timestamp) and its exact frame
                   count, so the harness decodes the episode's own frames out of the pack
  state.npz        state (T, 14) = observation.state and action (T, 14), exactly as shipped
  instruction.txt  the episode's annotated instruction (meta/tasks_annotated.parquet), which the dataset card
                   names as the per-episode instruction
  context.json     the facts the prompt may state: dataset, episode_index, robot_type, fps, the coarse task label
                   (meta/tasks.parquet), the instruction, and each camera's key, size and codec from meta/info.json

Nothing about the data is changed or cleaned. A camera window whose frame count disagrees with the state length
is written as is and the harness reports it. The sped-up recording check needs every episode of the dataset
(it compares neighbouring episodes), so it runs on its own afterwards, as
`python -m checks.timebase scan --raw RAW --out timebase.csv` and then
`python -m checks.timebase apply --timebase timebase.csv EPISODES`.

`sample` downloads meta/ and writes LIST. It is deterministic (no seed). --mode pack (the default) picks whole
video packs spread across each task's range, round robin across tasks, until --hours. Every downloaded byte is
used, and packs from across a task's range span different recording sessions. --mode episode picks single
episodes the same way. The dataset needs no token.
"""
from __future__ import annotations

import json
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from prepare import cli
from prepare import hub
from prepare.formats import scalar

# Data Review and python -m prepare folder hand an upload to an adapter that recognizes it (prepare/formats.py
# upload_adapters); this one reads only the published dataset
UPLOAD = None

REPO = "allenai/MolmoAct2-BimanualYAM-Dataset"
CAM_TO_VIEW = {"top": "exo", "left": "left", "right": "right"}
VKEYS = ("top", "left", "right")
FPS = 30# what each camera is, checked on the dataset's frames (per-episode facts handed to the model)
CAMERA_DESC = {
    "exo": "a fixed camera looking down at the table and both arms. Use it for the scene layout, object "
           "locations and where things end up",
    "left": "the camera mounted on the LEFT arm's gripper; that gripper's fingertips are at the bottom of its image",
    "right": "the camera mounted on the RIGHT arm's gripper; that gripper's fingertips are at the bottom of its image",
}


def ensure_meta(raw: Path) -> Path:
    """The dataset's meta/ folder (info.json, tasks, episode index), downloaded once."""
    meta = raw / "meta"
    if not (meta / "info.json").exists():
        from huggingface_hub import snapshot_download
        snapshot_download(REPO, repo_type="dataset", local_dir=str(raw), allow_patterns=["meta/*", "meta/**"],
                          token=hub.token())
    return meta


def load_index(meta_dir: Path) -> pd.DataFrame:
    ep = pd.concat([pd.read_parquet(p) for p in
                    sorted((meta_dir / "episodes").glob("chunk-*/*.parquet"))],
                   ignore_index=True)
    rows = {"eidx": [], "task": [], "length": []}
    for k in VKEYS:
        rows[f"{k}_chunk"] = []
        rows[f"{k}_file"] = []
    for _, r in ep.iterrows():
        rows["eidx"].append(int(scalar(r["episode_index"])))
        rows["task"].append(str(scalar(r["tasks"])))
        rows["length"].append(int(scalar(r["length"])))
        for k in VKEYS:
            rows[f"{k}_chunk"].append(int(scalar(r[f"videos/observation.images.{k}/chunk_index"])))
            rows[f"{k}_file"].append(int(scalar(r[f"videos/observation.images.{k}/file_index"])))
    df = pd.DataFrame(rows)
    df["dur_s"] = df["length"] / FPS
    return df


def sample_episodes(df: pd.DataFrame, target_h: float, per_task_cap_h: float | None) -> list[int]:
    """Episode-level sample: per task, episodes in an evenly spread order over the task's index range (so
    consecutive picks for a task are far apart, from different recording sessions), one pick per task per
    pass, until the target hours; a task stops at per_task_cap_h."""
    target_s = target_h * 3600
    cap_s = (per_task_cap_h * 3600) if per_task_cap_h else None
    per_task = {}
    for task, grp in df.groupby("task"):
        g = grp.sort_values("eidx").reset_index(drop=True)
        per_task[task] = {"g": g, "order": _spread_permutation(len(g)), "pos": 0, "acc_s": 0.0}
    tasks = sorted(per_task)
    chosen, total_s = [], 0.0
    progressing = True
    while total_s < target_s and progressing:
        progressing = False
        for task in tasks:
            st = per_task[task]
            if st["pos"] >= len(st["g"]):
                continue
            if cap_s is not None and st["acc_s"] >= cap_s:
                continue
            ridx = st["order"][st["pos"]]
            st["pos"] += 1
            row = st["g"].iloc[ridx]
            chosen.append(int(row["eidx"]))
            st["acc_s"] += float(row["dur_s"])
            total_s += float(row["dur_s"])
            progressing = True
            if total_s >= target_s:
                break
    return chosen


def sample_packs(df: pd.DataFrame, target_h: float) -> list[int]:
    """Pack-level sample: whole video packs (each about 12 to 50 episodes sharing one packed mp4) spread
    across each task's pack range, one pack per task per pass, until the target hours. Returns every episode
    index in the chosen packs. A pack is keyed by the top camera's (chunk, file); the three cameras of an
    episode share the same (chunk, file), so one key names the whole three-camera pack."""
    target_s = target_h * 3600
    per_task = {}
    for task, grp in df.groupby("task"):
        packs = {}
        for _, r in grp.iterrows():
            packs.setdefault((int(r["top_chunk"]), int(r["top_file"])),
                             []).append((int(r["eidx"]), float(r["dur_s"])))
        keys = sorted(packs)
        order = _spread_permutation(len(keys))
        per_task[task] = {"packs": packs, "keys": keys, "order": order, "pos": 0}
    tasks = sorted(per_task)
    chosen, total_s = [], 0.0
    progressing = True
    while total_s < target_s and progressing:
        progressing = False
        for task in tasks:
            st = per_task[task]
            if st["pos"] >= len(st["keys"]):
                continue
            key = st["keys"][st["order"][st["pos"]]]
            st["pos"] += 1
            eps = st["packs"][key]
            for eid, dur in eps:
                chosen.append(eid)
                total_s += dur
            progressing = True
            if total_s >= target_s:
                break
    return chosen


def _spread_permutation(n: int) -> list[int]:
    """A permutation of 0..n-1 whose prefix of any length is roughly evenly spread over the range (midpoints
    breadth first), so the first k picks sample the whole index span rather than a contiguous block."""
    if n <= 2:
        return list(range(n))
    order, seen = [], set()
    q = deque([(0, n - 1)])
    while q:
        lo, hi = q.popleft()
        if lo > hi:
            continue
        mid = (lo + hi) // 2
        if mid not in seen:
            seen.add(mid)
            order.append(mid)
        q.append((lo, mid - 1))
        q.append((mid + 1, hi))
    return order


def pack_of(row: dict, cam: str) -> tuple[int, int]:
    """(chunk, file) of the video pack holding this episode's frames of camera cam."""
    return (int(scalar(row[f"videos/observation.images.{cam}/chunk_index"])),
            int(scalar(row[f"videos/observation.images.{cam}/file_index"])))


def pack_rel(cam: str, chunk: int, file: int) -> str:
    return f"videos/observation.images.{cam}/chunk-{chunk:03d}/file-{file:03d}.mp4"


def packed_path(raw_root: Path, cam: str, chunk: int, file: int) -> Path:
    return raw_root / pack_rel(cam, chunk, file)


def data_path(raw_root: Path, chunk: int, file: int) -> Path:
    return raw_root / "data" / f"chunk-{chunk:03d}" / f"file-{file:03d}.parquet"


def ensure_data_parquet(raw_root: Path, chunk: int, file: int) -> Path:
    p = data_path(raw_root, chunk, file)
    if not p.exists():
        hub.download(REPO, f"data/chunk-{chunk:03d}/file-{file:03d}.parquet", raw_root)
    return p


def ensure_packs(rows: list[dict], raw_root: Path, jobs: int) -> None:
    """Download every video pack the episodes need, grouped by pack so all three cameras of a pack arrive
    together."""
    rels = sorted({pack_rel(cam, *pack_of(r, cam)) for r in rows for cam in VKEYS})
    todo = [r for r in rels if not (raw_root / r).exists()]
    print(f"video packs needed {len(rels)}, to download {len(todo)}", flush=True)
    with ThreadPoolExecutor(jobs) as ex:
        for f in as_completed([ex.submit(hub.download, REPO, r, raw_root) for r in todo]):
            f.result()


def episode_arrays(data_parquet: Path, episode_index: int) -> tuple[np.ndarray, np.ndarray]:
    """observation.state and action for one episode, in frame order, exactly as shipped."""
    d = pd.read_parquet(data_parquet, columns=["observation.state", "action", "frame_index",
                                               "episode_index"])
    e = d[d["episode_index"] == episode_index].sort_values("frame_index")
    if not (e["frame_index"].to_numpy() == np.arange(len(e))).all():
        raise ValueError(f"episode {episode_index}: frame_index is not 0..T-1")
    return (np.stack(e["observation.state"].to_numpy()).astype(np.float32),
            np.stack(e["action"].to_numpy()).astype(np.float32))


def prepare_episode(row: dict, raw_root: Path, out_root: Path, instruction: str, force: bool,
                    info: dict | None = None) -> str:
    eidx = int(scalar(row["episode_index"]))
    ep_dir = out_root / f"episode_{eidx:06d}"
    if not force and (ep_dir / "context.json").exists():
        return "skip"
    sources = {}
    for cam, view in CAM_TO_VIEW.items():
        pk = packed_path(raw_root, cam, *pack_of(row, cam))
        if not pk.exists():
            raise FileNotFoundError(f"missing packed video {pk}")
        base_s = float(scalar(row[f"videos/observation.images.{cam}/from_timestamp"]))
        to_s = float(scalar(row[f"videos/observation.images.{cam}/to_timestamp"]))
        sources[view] = {"packed": str(pk.resolve()), "base_s": base_s,
                         "n_frames": int(round((to_s - base_s) * FPS)),
                         "camera_key": f"observation.images.{cam}"}
    dch, dfl = int(scalar(row["data/chunk_index"])), int(scalar(row["data/file_index"]))
    dparq = ensure_data_parquet(raw_root, dch, dfl)
    state, action = episode_arrays(dparq, eidx)

    feats = (info or {}).get("features", {})
    context = {
        "dataset": REPO,
        "profile": "teleop_arms",
        "state_kind": "joints",
        "gripper_value": "0 = jaws shut, 1 = fully open (checked against the wrist frames)",
        "gripper_range": [0.0, 1.0],
        "episode_index": eidx,
        "robot_type": (info or {}).get("robot_type"),
        "fps": (info or {}).get("fps"),
        "task_label": [str(t) for t in row["tasks"]] if hasattr(row["tasks"], "__len__")
                      and not isinstance(row["tasks"], str) else [str(row["tasks"])],
        "instruction": (instruction or "").strip(),
        "n_state_frames": int(len(state)),
        "cameras": {view: {"key": f"observation.images.{cam}",
                           **{k.split(".")[-1]: v for k, v in
                              (feats.get(f"observation.images.{cam}", {}).get("info") or {}).items()
                              if k in ("video.width", "video.height", "video.codec", "video.fps")}}
                    for cam, view in CAM_TO_VIEW.items()},
    }
    for view, cam in context["cameras"].items():
        cam["desc"] = CAMERA_DESC[view]
    ep_dir.mkdir(parents=True, exist_ok=True)
    np.savez(ep_dir / "state.npz", state=state, action=action)
    (ep_dir / "sources.json").write_text(json.dumps(sources, indent=2))
    (ep_dir / "instruction.txt").write_text((instruction or "").strip() + "\n")
    (ep_dir / "context.json").write_text(json.dumps(context, indent=2))
    return "ok"


def load_rows(meta_dir: Path, episode_indices):
    ep = pd.concat([pd.read_parquet(p) for p in
                    sorted((meta_dir / "episodes").glob("chunk-*/*.parquet"))],
                   ignore_index=True)
    ta = pd.read_parquet(meta_dir / "tasks_annotated.parquet")
    ta_map = {int(i): str(t) for i, t in ta["task"].items()}
    if episode_indices is not None:
        want = set(int(i) for i in episode_indices)
        ep = ep[ep["episode_index"].apply(lambda v: int(scalar(v)) in want)]
    rows = []
    for _, r in ep.iterrows():
        d = r.to_dict()
        d["_instruction"] = ta_map.get(int(scalar(r["episode_index"])), "")
        rows.append(d)
    return rows


def cmd_sample(args) -> int:
    df = load_index(ensure_meta(args.raw))
    chosen = (sample_packs(df, args.hours) if args.mode == "pack"
              else sample_episodes(df, args.hours, args.per_task_cap_hours))
    sub = df[df["eidx"].isin(set(chosen))]
    args.list.write_text("\n".join(str(e) for e in sorted(chosen)) + "\n")
    print(json.dumps({"episodes": len(chosen), "hours": round(sub["dur_s"].sum() / 3600, 3),
                      "tasks": int(sub["task"].nunique()), "list": str(args.list)}))
    return 0


def cmd_prepare(args) -> int:
    meta = ensure_meta(args.raw)
    rows = load_rows(meta, [int(s) for s in cli.read_list(args.episodes)])
    info = json.loads((meta / "info.json").read_text())
    ensure_packs(rows, args.raw, args.jobs)
    args.out.mkdir(parents=True, exist_ok=True)
    return cli.run(rows, lambda r: prepare_episode(r, args.raw, args.out, r["_instruction"], args.force, info),
                   args.jobs)


def main() -> int:
    ap, sub = cli.parser("molmo", __doc__)
    s = sub.add_parser("sample", help="choose episodes for about --hours and write their list")
    s.add_argument("--list", type=Path, required=True, help="the episode list to write")
    s.add_argument("--raw", type=Path, default=Path("data/raw/molmo"), help="where downloaded files are kept")
    s.add_argument("--hours", type=float, required=True)
    s.add_argument("--mode", choices=["pack", "episode"], default="pack")
    s.add_argument("--per-task-cap-hours", type=float, default=None, help="episode mode: most hours from one task")
    cli.add_prepare(sub, "molmo", "one episode index per line", jobs=min(16, os.cpu_count() or 8))
    args = ap.parse_args()
    return cmd_sample(args) if args.cmd == "sample" else cmd_prepare(args)


if __name__ == "__main__":
    raise SystemExit(main())
