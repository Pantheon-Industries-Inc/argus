"""Prepare IPEC-COMMUNITY/FastUMI_100k_lerobot episodes as episode sidecars (rig handheld_gripper, state ee_pose).

    python -m prepare fastumi sample --list LIST --per-task 32 --seed 0 [--raw RAW]
    python -m prepare fastumi prepare --episodes configs/slices/fastumi.txt --out EPISODES [--raw RAW] [--jobs N]
        [--force]

An episode list has one <embodiment>/<task>/<episode index> per line (dual_arm/Fold_the_Jeans/12). `sample`
downloads every task folder's meta/episodes.jsonl, draws --per-task random episodes from each (seeded per task)
and writes LIST. `prepare` downloads, into RAW/<embodiment>/<task>/, the task's meta/info.json and
meta/episodes.jsonl, recorded task tables and the episode's parquet and camera mp4s, and writes EPISODES/episode_<embodiment>__<task>__
<index>/ with context.json, sources.json (pointing at the downloaded mp4s), state.npz and instruction.txt.

The dataset is LeRobot v2.1: per task a folder with one parquet and one mp4 per camera per episode, 20 fps. A
person holds a handheld gripper in each hand (dual_arm tasks) or in one hand (single_arm); each gripper carries
one camera and there is no fixed camera. The state is each gripper's pose (x y z in m, roll pitch yaw in rad) and
opening (0 to 1), and the action is the next state. Nothing is re-encoded. Frame k is decoded at its exact pts
(the files are on the 20 fps grid, checked per file into stream_checks). The dataset needs no token.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from label.atomic import write_atomic
from prepare import cli
from prepare import hub
from label.presented import presented_pts
from prepare.frame_rate import declared_fps


REPO = "IPEC-COMMUNITY/FastUMI_100k_lerobot"
CAMS = {"observation.images.left_camera_rgb_image": ("left", "left"),
        "observation.images.right_camera_rgb_image": ("right", "right"),
        "observation.images.camera_rgb_image": ("right", "gripper")}
# what each camera is, checked on the dataset's frames (per-episode facts handed to the model)
CAMERA_DESC = {
    "left": "the fisheye camera carried on the LEFT-hand gripper, looking along its fingers, which are at "
            "the bottom of the image",
    "right": "the fisheye camera carried on the RIGHT-hand gripper, looking along its fingers, which are at "
             "the bottom of the image",
    "gripper": "the fisheye camera carried on the handheld gripper, looking along its fingers, which are at "
               "the bottom of the image",
}


def _dl(path: str, raw_root: Path) -> Path:
    return Path(hub.download(REPO, path, raw_root))


def tasks() -> list[str]:
    from huggingface_hub import HfApi
    api = HfApi()
    out = []
    for emb in ("dual_arm", "single_arm"):
        for f in api.list_repo_tree(REPO, repo_type="dataset", path_in_repo=emb):
            out.append(f.path)
    return sorted(out)


def cmd_sample(args) -> int:
    picked, skipped = [], []
    for t in tasks():
        try:
            lines = _dl(f"{t}/meta/episodes.jsonl", args.raw).read_text().splitlines()
            eps = [json.loads(s) for s in lines if s.strip()]
        except Exception:  # noqa: BLE001 - a sample task whose metadata does not come is listed in skipped
            skipped.append(t)          # a task folder without LeRobot metadata (dual_arm/Unplug_the_Power_Strip)
            continue
        rng = random.Random(f"{args.seed}:{t}")
        for e in rng.sample(eps, min(args.per_task, len(eps))):
            picked.append(f"{t}/{int(e['episode_index'])}")
    Path(args.list).write_text("\n".join(sorted(picked)) + "\n")
    print(f"{len(picked)} episodes -> {args.list}; skipped task folders without metadata: {skipped}")
    return 0


def pts_on_grid(mp4: Path, fps: int) -> tuple[int, bool]:
    """Packet count, and whether frame k sits exactly at k * (1/fps) on the file's time base."""
    import av
    from fractions import Fraction
    with av.open(str(mp4)) as c:
        s = c.streams.video[0]
        pts = presented_pts(c, s)
        step = Fraction(1, fps) / s.time_base
    ok = step.denominator == 1 and pts == [k * int(step) for k in range(len(pts))]
    return len(pts), ok


def episode_dir_name(item: str) -> str:
    """dual_arm/Fold_the_Jeans/12 -> episode_dual_arm__Fold_the_Jeans__000012"""
    task, eidx = item.rsplit("/", 1)
    return f"episode_{task.replace('/', '__')}__{int(eidx):06d}"


def prepare_one(item: str, raw_root: Path, out_root: Path, force: bool) -> str:
    import pandas as pd
    from prepare import formats
    from prepare.lerobot_labels import TABLE_COLUMNS
    task, eidx = item.rsplit("/", 1)
    eidx = int(eidx)
    name = episode_dir_name(item)
    ep_dir = out_root / name
    if not force and (ep_dir / "context.json").exists():
        return "skip"
    info_path = _dl(f"{task}/meta/info.json", raw_root)
    info = json.loads(info_path.read_text())
    fps = int(declared_fps(info, required=True))
    chunk = eidx // int(info.get("chunks_size", 1000))
    _dl(f"{task}/meta/episodes.jsonl", raw_root)
    parq = _dl(f"{task}/" + info["data_path"].format(episode_chunk=chunk, episode_index=eidx), raw_root)
    df = pd.read_parquet(parq).sort_values("frame_index")
    tables, lookup_unavailable = [], False
    if any(column in df.columns for column in TABLE_COLUMNS.values()):
        try:
            tables = hub.ls(REPO, f"{task}/meta")
        except hub.HubReadError:
            lookup_unavailable = True
    for entry in tables:
        path = Path(entry["path"])
        if (entry["type"] == "file" and path.parent == Path(task) / "meta"
                and path.suffix in (".jsonl", ".parquet") and TABLE_COLUMNS.get(path.stem) in df.columns):
            _dl(entry["path"], raw_root)
    state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    action = np.stack(df["action"].to_numpy()).astype(np.float32)
    ep_dir.mkdir(parents=True, exist_ok=True)
    sources, cameras, grid = {}, {}, {}
    for key, feat in info["features"].items():
        if key not in CAMS:
            continue
        view, cname = CAMS[key]
        rel = info["video_path"].format(episode_chunk=chunk, video_key=key, episode_index=eidx)
        mp4 = _dl(f"{task}/{rel}", raw_root)
        n, ok = pts_on_grid(mp4, fps)
        grid[view] = ok
        sources[view] = {"packed": str(mp4.resolve()), "base_s": 0.0, "n_frames": n, "camera_key": key}
        vi = feat.get("info") or {}
        cameras[view] = {"key": key, "name": cname, "width": vi.get("video.width"), "height": vi.get("video.height"),
                         "codec": vi.get("video.codec"), "desc": CAMERA_DESC[cname]}
    if not sources:
        raise RuntimeError(f"{item}: no known camera in {list(info['features'])}")
    root = formats.read_root(info_path.parent.parent, task)
    meta = next(row for row in root["episodes"] if row["eidx"] == eidx)
    tasks = meta["tasks"]
    instruction_note = ("This instruction is the episode's task text in the dataset's "
                        "meta/episodes.jsonl; it is a recorded goal claim to check.")
    if not tasks and "task_index" in df:
        try:
            tasks = formats.constant_index_task(df["task_index"], root["tasks_by_index"])
        except (TypeError, ValueError):
            tasks = []
        instruction_note = "This instruction is constant recorded task_index resolved by the native task table."
    instruction = " ".join(tasks).strip()
    context = {
        "dataset": REPO,
        "profile": "handheld_gripper",
        "state_kind": "ee_pose",
        "episode_id": name,
        "episode_index": eidx,
        "robot_type": "handheld gripper (UMI-style)",
        "gripper_range": [0.0, 1.0],
        "fps": fps,
        "task_label": [task],
        "instruction": instruction,
        "instruction_note": instruction_note,
        "n_state_frames": int(len(state)),
        "cameras": cameras,
        "stream_checks": {"frames_on_grid": grid, "episode_length_meta": meta["length"] if meta["length"] is not None else -1},
    }
    formats.lerobot_annotations(context, root, df, parq)
    if lookup_unavailable:
        formats.add_issue(context, 'metadata_limit', 'The native task table listing was unavailable; available local '
                          'tables and all recorded annotation codes were retained. Additional upstream tables may '
                          'not be present in this cache.')
    for issue in root["metadata_issues"]:
        formats.add_issue(context, issue["kind"], issue["text"])
    instruction = context.get("instruction", "")
    from prepare.state_contract import declared_action_semantics, publisher_action_semantics
    context['action_semantics'] = declared_action_semantics('prepare.fastumi', 'target', relationship={
        'kind': 'next_observation', 'field': 'state', 'row_offset': 1, 'terminal': 'repeat_last',
        'scope': 'source_recording'})
    publisher = publisher_action_semantics(info['features'].get('action'), 'prepare.fastumi:publisher')
    if publisher is not None:
        context['publisher_action_semantics'] = publisher
    formats.record_state_identity(context, "observation.state",
                                  formats.value_names(info["features"].get("observation.state", {}).get("names"),
                                                      state.shape[1]), state.shape[1])
    np.savez(ep_dir / "state.npz", state=state, action=action)
    (ep_dir / "sources.json").write_text(json.dumps(sources, indent=2))
    (ep_dir / "instruction.txt").write_text(instruction + "\n")
    write_atomic(ep_dir / "context.json", context, indent=2)
    return "ok"


def main() -> int:
    ap, sub = cli.parser("fastumi", __doc__)
    s = sub.add_parser("sample", help="draw episodes from every task folder and write their list")
    s.add_argument("--list", type=Path, required=True, help="the episode list to write")
    s.add_argument("--raw", type=Path, default=Path("data/raw/fastumi"), help="where downloaded files are kept")
    s.add_argument("--per-task", type=int, default=32)
    s.add_argument("--seed", type=int, default=0)
    cli.add_prepare(sub, "fastumi", "one <embodiment>/<task>/<episode index> per line")
    a = ap.parse_args()
    if a.cmd == "sample":
        return cmd_sample(a)
    a.out.mkdir(parents=True, exist_ok=True)
    return cli.run(cli.read_list(a.episodes), lambda it: prepare_one(it, a.raw, a.out, a.force), a.jobs)


if __name__ == "__main__":
    raise SystemExit(main())
