"""Prepare a folder of your own robot data as episode sidecars, read exactly as Data Review reads an upload.

    python -m prepare folder prepare --root FOLDER --rig RIG --out EPISODES [--dataset NAME] [--max-minutes M]

FOLDER holds one of: a LeRobot dataset (v2.0, v2.1 or v3.0, or a collection of them, one per folder), MCAP files
(one per episode), or video files (one per episode, or one folder per episode with a scene camera and a left and a
right mounted camera). Archives (.zip, .tar, .tar.gz, .tar.bz2, .tar.xz) are read as the folders they hold. An
MCAP in the ABC-130k or RealOmin layout goes through that dataset's adapter, so its recorded state is used; any
other MCAP is read for its cameras and its text channels (the task, and on a head camera its timed steps).

--rig says what recorded it: teleop_arms (one or two robot arms), handheld_gripper (one or two grippers carried by
a person) or ego_head (a camera worn on a person's head). A .txt or .json beside a video, or instruction.txt or
annotations.json inside an episode folder, reaches the model as your annotation, a claim to check against the
footage. --max-minutes stops after that much footage (default: no limit). prepare/formats.py documents every
layout it accepts and what it does when metadata is missing.

Writes EPISODES/episode_<name>/ with context.json, sources.json, state.npz when the recording has usable state,
times.npz and instruction.txt, and prints a report of what was read, used, skipped and why.
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
