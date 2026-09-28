"""Prepare a folder of your own video files as episode sidecars, one episode per file, video only (no recorded
state), for any rig.

    python -m prepare videos prepare --root FOLDER --out EPISODES [--rig RIG] [--episodes LIST]
        [--instructions FILE] [--dataset NAME] [--jobs N] [--force]

Every file under FOLDER (searched recursively) with a video extension (.mp4 .mov .m4v .mkv .webm .avi) is one
episode. --rig says what the footage is: ego_head (a head camera on a person, the default), teleop_arms (one camera
watching teleoperated arms, taken as the scene camera) or handheld_gripper (the camera carried on one handheld
gripper). LIST, if given, has one file path relative to FOLDER per line; without it every file is prepared. FILE is
an optional JSON object keyed by those relative paths. A value is the episode's instruction (a string), or an
object with "instruction" and, for head cameras, "subtasks", a list of {"t0": s, "t1": s, "label": text} for the
timed steps you annotated. Both reach the model as a claim to check against the video, as a dataset's own
instruction or annotation does. A file with no entry is labelled with no instruction: the model infers the task
from the footage. --dataset is the name written into context.json and shown to the model (default: the folder's
name).

Writes EPISODES/episode_<path>/ with context.json, sources.json (pointing at the file itself), times.npz (the
frames' real times, from their pts) and instruction.txt. Nothing is copied or re-encoded. The camera is named
after the file, and the episode's task label is its path without the extension. Two files whose paths clean to
the same folder name (run-1.mp4 and run_1.mp4) get _2, _3 in path order.
"""
from __future__ import annotations

import json
from pathlib import Path

from prepare import cli
from prepare import sidecar

# The view one camera fills on each rig: a head camera and a camera watching the arms are the scene view; a
# handheld gripper's camera is the gripper's own.
VIEW_BY_RIG = {"ego_head": "exo", "teleop_arms": "exo", "handheld_gripper": "right"}
# Said with an instruction, so the model reads it as the task text given with the episode rather than a
# dataset's coarse task label.
INSTRUCTION_NOTE = "This instruction is the task text the uploader sent with the episode."


def find_videos(root: Path) -> list[str]:
    """Paths of the video files under root, relative to it, sorted."""
    return sorted(p.relative_to(root).as_posix() for p in Path(root).rglob("*")
                  if p.is_file() and p.suffix.lower() in sidecar.DEMUXER)


def read_instructions(path: Path | None, files: list[str]) -> dict[str, dict]:
    """{relative path: {"instruction": ..., "subtasks": [...]}} from the optional JSON file."""
    if path is None:
        return {}
    raw = json.loads(Path(path).read_text())
    unknown = sorted(set(raw) - set(files))
    if unknown:
        raise SystemExit(f"{path} names files that are not in the folder, e.g. {unknown[:5]}")
    return {k: (v if isinstance(v, dict) else {"instruction": v}) for k, v in raw.items()}


def prepare_one(root: Path, rel: str, ep: Path, rig: str, dataset: str, given: dict, force: bool) -> str:
    if not force and (ep / "context.json").exists():
        return "skip"
    name = rel.rsplit(".", 1)[0]
    # the file name is the task label, and "video files" tells label/route.py that it is only a file name
    extra = {"task_label": [name], "source": {"format": "video files", "file": rel}}
    instruction = (given.get("instruction") or "").strip()
    if instruction:
        extra["instruction"] = instruction
        extra["instruction_note"] = INSTRUCTION_NOTE
    subs = [{"t0": float(x["t0"]), "t1": float(x["t1"]), "label": str(x["label"])} for x in given.get("subtasks") or []]
    if subs:
        extra["annotation_subtasks"] = subs
    stem = rel.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    sidecar.video_views_episode(ep, {VIEW_BY_RIG[rig]: (stem, root / rel)}, rig, dataset, extra)
    return "ok"


def main() -> int:
    ap, sub = cli.parser("videos", __doc__)
    p = cli.add_prepare(sub, "videos", "one video path relative to --root per line (default: every video)",
                        raw=False, episodes_required=False)
    p.add_argument("--root", type=Path, required=True, help="the folder of video files")
    p.add_argument("--rig", choices=sorted(VIEW_BY_RIG), default="ego_head",
                   help="what the footage is (default ego_head)")
    p.add_argument("--instructions", type=Path, default=None, metavar="FILE",
                   help="JSON object: relative path -> instruction, or {\"instruction\": ..., \"subtasks\": [...]}")
    p.add_argument("--dataset", default=None, metavar="NAME",
                   help="the dataset name written into context.json (default: the folder's name)")
    a = ap.parse_args()
    files = find_videos(a.root)
    dirs = dict(zip(files, sidecar.episode_dirs(a.out, [f.rsplit(".", 1)[0] for f in files])))
    picks = cli.read_list(a.episodes) if a.episodes else files
    missing = [f for f in picks if f not in dirs]
    if missing:
        raise SystemExit(f"{len(missing)} listed files are not videos under {a.root}, e.g. {missing[:5]}")
    given = read_instructions(a.instructions, files)
    if a.rig != "ego_head" and any(g.get("subtasks") for g in given.values()):
        raise SystemExit("timed subtasks are read for head cameras only (--rig ego_head)")
    name = a.dataset or a.root.resolve().name
    a.out.mkdir(parents=True, exist_ok=True)
    return cli.run(picks, lambda f: prepare_one(a.root, f, dirs[f], a.rig, name, given.get(f, {}), a.force), a.jobs)


if __name__ == "__main__":
    raise SystemExit(main())
