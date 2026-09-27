"""Prepare configinc/HABIT episodes (bimanual Franka FR3 teleoperated with a Quest 3, human-robot
collaboration tasks, LeRobot v2.0, one file per episode) as episode sidecars (rig teleop_arms, state ee_pose).

    python -m prepare habit prepare --episodes configs/slices/habit.txt --out EPISODES [--raw RAW] [--jobs N]
        [--force]

The list has one episode index per line. This downloads into RAW/full the dataset's meta/ files (info.json,
episodes.jsonl, tasks.jsonl, subtasks.jsonl, human_subtasks.jsonl) and, per listed episode, its parquet and the
videos of the three cameras used. It writes EPISODES/episode_<index>/ with context.json, sources.json, state.npz and
instruction.txt (see prepare/sidecar.py, whose LeRobot reader this uses). HABIT specifics set here:
- observation.state is, per arm, the end effector's x y z (m) and three rotation values followed by the
  gripper opening, so the state kind is "ee_pose" (the LeRobot reader's teleop default, "joints", would read
  poses as joint angles).
- The instruction is the episode's high-level instruction. The instruction note says that a person shares the
  workspace in every episode, and gives the person's and the robot's timed parts as the operators marked them.
- The publisher's own labels (task_status, and the frames it marks as error, intervention and high-jerk
  segments) are kept in context["publisher_labels"] to score labels against afterwards. They are never put in
  the prompt.

The dataset needs no token.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare import cli
from prepare import hub
from prepare import sidecar

REPO = "configinc/HABIT"
META_FILES = ("info.json", "episodes.jsonl", "tasks.jsonl", "subtasks.jsonl", "human_subtasks.jsonl")
RIG = "teleop_arms"
COLLECTION_NOTE = "each episode is one teleoperated demonstration of one task, from its start to its end"
ROLE_NOTE = ("In every HABIT episode a person shares the workspace with the robot, by design. The task gives that "
             "person one of three roles: collaborator (they and the robot do the task together, with direct contact "
             "such as a handover or holding something together), coworker (each does its own part in a shared space, "
             "without contact), or supervisor (the person directs the robot with gestures or cues). The person's "
             "actions within their part are the task itself, not a data issue.")


def spans(flags: np.ndarray, fps: float) -> list[list[float]]:
    """Runs of true flags as [start s, end s]."""
    out, start = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append([round(start / fps, 2), round(i / fps, 2)])
            start = None
    return out


def timed_parts(idx: np.ndarray, texts: dict, fps: float) -> list[str]:
    """Consecutive runs of one subtask index as 't0-t1 s: text' (index < 0 means no active subtask)."""
    out, start = [], 0
    for i in range(1, len(idx) + 1):
        if i == len(idx) or idx[i] != idx[start]:
            k = int(idx[start])
            if k >= 0 and k in texts:
                out.append(f"{start / fps:.1f}-{i / fps:.1f} s: {texts[k]}")
            start = i
    return out


def _texts(p: Path) -> dict:
    return {json.loads(line)["task_index"]: json.loads(line)["task"] for line in open(p)}


def task_note(root: Path, df: pd.DataFrame, fps: float) -> str:
    hp = timed_parts(df["human_role_subtask_index"].to_numpy(), _texts(root / "meta" / "human_subtasks.jsonl"), fps)
    rp = timed_parts(df["low_level_task_index"].to_numpy(), _texts(root / "meta" / "subtasks.jsonl"), fps)
    return (ROLE_NOTE + "\nThe person's part in this episode, with the times the operators marked by foot pedal:\n  "
            + ("\n  ".join(hp) or "none recorded")
            + "\nThe robot's part, marked the same way (claims to check against the video, like the instruction):\n  "
            + ("\n  ".join(rp) or "none recorded"))


def download_meta(raw: Path) -> Path:
    for f in META_FILES:
        if not (raw / "full" / "meta" / f).exists():
            hub.download(REPO, f"full/meta/{f}", raw)
    return raw / "full"


def download_episode(root: Path, raw: Path, eidx: int) -> None:
    """The episode's parquet and the videos of the cameras the harness uses."""
    info = json.loads((root / "meta" / "info.json").read_text())
    feats = info.get("features", {})
    used, _ = sidecar.pick_cameras([k for k, f in feats.items() if f.get("dtype") == "video"], RIG)
    chunk = eidx // int(info.get("chunks_size", 1000))
    rels = [info["data_path"].format(episode_chunk=chunk, episode_index=eidx)]
    rels += [info["video_path"].format(episode_chunk=chunk, video_key=k, episode_index=eidx) for k in used.values()]
    for rel in rels:
        if not (root / rel).exists():
            hub.download(REPO, f"full/{rel}", raw)


def prepare_one(eidx: int, rows: dict, root: Path, raw: Path, out: Path, force: bool) -> str:
    ep = out / sidecar.episode_name(f"{eidx:06d}")
    if not force and (ep / "context.json").exists():
        return "skip"
    download_episode(root, raw, eidx)
    info = json.loads((root / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    r = rows[eidx]
    ctx = sidecar.convert_lerobot({"name": f"{eidx:06d}", "row": r}, root, RIG, out, REPO)
    df = pd.read_parquet(root / info["data_path"].format(episode_chunk=eidx // 1000, episode_index=eidx),
                         columns=["low_level_task_index", "human_role_subtask_index", "is_error_segment",
                                  "is_intervention_segment", "is_high_jerk_segment"])
    # the dataset's metadata gives robot_type "human", which would tell the model a person is the robot
    ctx.update(robot_type="two Franka FR3 arms teleoperated with a Meta Quest 3", state_kind="ee_pose",
               instruction=r.get("high_level_instruction") or ctx.get("instruction"),
               instruction_note=task_note(root, df, fps),
               collection_note=COLLECTION_NOTE,
               publisher_labels={"task_status": r.get("task_status"),
                                 "error_spans_s": spans(df["is_error_segment"].to_numpy(), fps),
                                 "intervention_spans_s": spans(df["is_intervention_segment"].to_numpy(), fps),
                                 "high_jerk_spans_s": spans(df["is_high_jerk_segment"].to_numpy(), fps),
                                 "sid": r.get("sid"), "unit_name": r.get("unit_name")})
    ctx.pop("state_note", None)
    (ep / "context.json").write_text(json.dumps(ctx, indent=1))
    return "ok"


def main() -> int:
    ap, sub = cli.parser("habit", __doc__)
    cli.add_prepare(sub, "habit", "one episode index per line", jobs=4)
    a = ap.parse_args()
    picks = [int(s) for s in cli.read_list(a.episodes)]
    a.out.mkdir(parents=True, exist_ok=True)
    root = download_meta(a.raw)
    rows = {int(r["episode_index"]): r for r in sidecar.read_jsonl(root / "meta" / "episodes.jsonl")}
    return cli.run(picks, lambda e: prepare_one(e, rows, root, a.raw, a.out, a.force), a.jobs)


if __name__ == "__main__":
    raise SystemExit(main())
