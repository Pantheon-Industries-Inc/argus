"""Prepare a folder of your own video files as episode sidecars, one head-camera episode per file (rig ego_head,
no recorded state).

    python -m prepare videos prepare --root FOLDER --out EPISODES [--episodes LIST] [--instructions FILE]
        [--dataset NAME] [--jobs N] [--force]

Every file under FOLDER (searched recursively) with a video extension (.mp4 .mov .m4v .mkv .webm .avi) is one
episode. LIST, if given, has one file path relative to FOLDER per line; without it every file is prepared.
FILE is an optional JSON object keyed by those relative paths. A value is the episode's instruction (a string),
or an object with "instruction" and "subtasks", a list of {"t0": s, "t1": s, "label": text} for the timed steps
you annotated. Both reach the model as the annotation to check against the video, as a dataset's own annotation
does. --dataset is the name written into context.json and shown to the model (default: the folder's name).

Writes EPISODES/episode_<path>/ with context.json, sources.json (pointing at the file itself), times.npz (the
frames' real times, from their pts) and instruction.txt. Nothing is copied or re-encoded. Two files whose paths
clean to the same folder name (run-1.mp4 and run_1.mp4) get _2, _3 in path order.
"""
from __future__ import annotations

import json
from pathlib import Path

from prepare import cli
from prepare import sidecar


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


def prepare_one(root: Path, rel: str, ep: Path, dataset: str, given: dict, force: bool) -> str:
    if not force and (ep / "context.json").exists():
        return "skip"
    subs = [{"t0": float(x["t0"]), "t1": float(x["t1"]), "label": str(x["label"])} for x in given.get("subtasks") or []]
    extra = {"task_label": [rel], "instruction": (given.get("instruction") or "").strip() or None,
             "annotation_subtasks": subs, "source": {"file": rel}}
    sidecar.video_views_episode(ep, {"exo": (rel, root / rel)}, "ego_head", dataset, extra)
    return "ok"


def main() -> int:
    ap, sub = cli.parser("videos", __doc__)
    p = cli.add_prepare(sub, "videos", "one video path relative to --root per line (default: every video)",
                        raw=False, episodes_required=False)
    p.add_argument("--root", type=Path, required=True, help="the folder of video files")
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
    name = a.dataset or a.root.resolve().name
    a.out.mkdir(parents=True, exist_ok=True)
    return cli.run(picks, lambda f: prepare_one(a.root, f, dirs[f], name, given.get(f, {}), a.force), a.jobs)


if __name__ == "__main__":
    raise SystemExit(main())
