"""Prepare a folder of your own robot data as episode sidecars, read exactly as Data Review reads an upload.

    python -m prepare folder prepare --root FOLDER --rig RIG --out EPISODES [--dataset NAME] [--max-minutes M]

FOLDER holds one of: a LeRobot dataset (v2.0, v2.1 or v3.0, or a collection of them, one per folder; HABIT's and
Galaxea's go through their adapters, recognized by their columns), MCAP files
(one per episode), or video files (one per episode, or one folder per episode with a scene camera and a left and a
right mounted camera). Archives (.zip, .tar, .tar.gz, .tar.bz2, .tar.xz) are read as the folders they hold. An
MCAP in a layout a dataset adapter recognizes goes through that adapter (today ABC-130k and RealOmin, with their
recorded state, and Gen-HumanEgo, with its goal and timed steps); any other MCAP is read for its cameras, its arm
joint channels and its text channels (the task, and on a head camera its timed steps). MCAP files with no camera
beside an episode's videos are its recorded arm state.

--rig says what recorded it: teleop_arms (one or two robot arms), handheld_gripper (one or two grippers carried by a
person) or ego_head (a camera worn on a person's head). Your notes reach the model as claims to check against the
footage. They are a .txt, .json, .jsonl or .md named as a video, and in an episode's folder (or the folder of a
video that is the only episode there) the .txt or .json named for the episode, annotations.json, annotation.json,
meta.json, instruction.txt, task.txt, annotations.jsonl and notes.txt. The task comes from a JSON note's task key,
then instruction.txt or task.txt, then the .txt named for the episode, then a video's own .txt. A recorder's .json
in the folder that names the task gives it to the episodes its name names, or to every episode there when its name
names none. One naming an absent take keeps its task or note on its uploaded owners and reports the absent take;
one naming only absent takes is not read. --max-minutes stops after that much footage
(default: no limit). prepare/formats.py documents every layout it accepts and what it does when metadata is missing.

Writes EPISODES/episode_<name>/ with context.json, sources.json, state.npz when the recording has usable state,
times.npz and instruction.txt, and prints a report of what was read, used, skipped and why.
MCAP and HDF5 container notes apply to their contained episodes. LeRobot notes use the recorded episode index.
Outside files retain their filenames as uploader notes, and recorded instruction text keeps priority.
Owned text notes remain notes even when their filenames do not qualify as tasks. Camera notes accept every
own-note form on the same proven owners. LeRobot reserves official metadata at its actual meta paths;
recorder notes at the root retain their full relative filenames. Failed metadata reads and note size limits
are named in the report and episode issues, and successful format reads determine what was consumed.
Process diagnostics and standard logs stay in source bookkeeping and saved arrays, outside model sensor claims.

"""
from __future__ import annotations

import json
import math
from pathlib import Path

from prepare import cli
from prepare import formats


def main() -> int:
    ap, sub = cli.parser("folder", __doc__)
    p = sub.add_parser("prepare", help="prepare every episode in the folder")
    p.add_argument("--root", type=Path, required=True, help="the folder (or archive) holding your data")
    p.add_argument("--rig", required=True, choices=formats.RIGS, help="what recorded it")
    p.add_argument("--out", type=Path, required=True, help="where the episode folders are written")
    p.add_argument("--dataset", default=None, metavar="NAME",
                   help="the dataset name written into context.json (default: the folder's name)")
    p.add_argument("--max-minutes", type=float, default=None, help="stop after this much footage (default: all)")
    a = ap.parse_args()
    report = formats.convert(a.root, a.rig, a.out, a.dataset or a.root.resolve().name.split(".")[0],
                             a.max_minutes * 60 if a.max_minutes else math.inf)
    print(json.dumps(report, indent=1))
    return 1 if report["failed"] and not report["episodes"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
