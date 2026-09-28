"""Prepare a LeRobot dataset on your own disk (v2.0, v2.1 or v3.0) as episode sidecars.

    python -m prepare lerobot prepare --root DATASET --rig teleop_arms --out EPISODES [--episodes LIST]
        [--dataset NAME] [--jobs N] [--force]

DATASET is the dataset's folder (meta/info.json, the episode metadata, data/ and videos/). LIST, if given, has one
episode index per line; without it every episode whose files are present is prepared. --rig says what recorded
it: teleop_arms (one or two robot arms), handheld_gripper (one or two grippers carried by a person) or ego_head
(a camera worn on a person's head). --dataset is the name written into context.json and shown to the model
(default: the folder's name).

It is read exactly as Data Review reads an uploaded LeRobot dataset (prepare/formats.py). Cameras are the
features of dtype "video" (or images stored in the data files), assigned to the harness views by name: a name
with a side and wrist, hand or gripper is that side's mounted camera, and the best-named other camera (top, head,
overhead, front) is the scene camera; the rest are listed in context["source"]["unused_cameras"]. The state is
observation.state and the action is action, used when there are 7 values per arm or gripper (6 joints plus
gripper for teleop arms, x y z roll pitch yaw plus opening for handheld grippers); any other layout is labelled
from the video alone and the context says so. The instruction is the episode's task text.

Writes EPISODES/episode_<index>/ with context.json, sources.json (pointing at the dataset's own mp4s), state.npz,
times.npz (only when a v2 video's frames are off the k / fps grid) and instruction.txt. Nothing is downloaded or
copied; only cameras stored as images in the data files are written to H.264, at their frame times.
"""
from __future__ import annotations

from pathlib import Path

from prepare import cli
from prepare import formats

RIGS = ("teleop_arms", "handheld_gripper", "ego_head")


def main() -> int:
    ap, sub = cli.parser("lerobot", __doc__)
    p = cli.add_prepare(sub, "lerobot", "one episode index per line (default: every episode present)",
                        raw=False, episodes_required=False)
    p.add_argument("--root", type=Path, required=True, help="the LeRobot dataset's folder")
    p.add_argument("--rig", required=True, choices=RIGS, help="what recorded the dataset")
    p.add_argument("--dataset", default=None, metavar="NAME",
                   help="the dataset name written into context.json (default: the folder's name)")
    a = ap.parse_args()
    det = formats.detect(a.root)
    if det["format"] != "lerobot":
        raise SystemExit(f"{a.root} holds no LeRobot dataset (it reads as {det['format']}); "
                         "python -m prepare folder reads any format")
    plan, used, missing, _ = formats.plan_lerobot(det, a.root)
    items = {it["name"]: it for it in plan}
    for line in used + missing:
        print(line)
    if a.episodes:
        want = [f"{int(s):06d}" if s.strip().isdigit() else s.strip() for s in cli.read_list(a.episodes)]
        picks = [n for n in items if n in want or n.rsplit("/", 1)[-1] in want]
        absent = sorted(set(want) - {n.rsplit("/", 1)[-1] for n in picks} - set(picks))
        if absent:
            raise SystemExit(f"{len(absent)} listed episodes are not in {a.root} or their files are absent, "
                             f"e.g. {absent[:5]}")
    else:
        picks = list(items)
    name = a.dataset or a.root.resolve().name
    a.out.mkdir(parents=True, exist_ok=True)

    def one(n: str) -> str:
        if not a.force and (a.out / formats.episode_name(n) / "context.json").exists():
            return "skip"
        it = items[n]
        (formats.convert_lerobot_item if it["kind"] == "lerobot" else formats.convert_recording)(it, a.rig, a.out, name)
        return "ok"
    return cli.run(picks, one, a.jobs)

if __name__ == "__main__":
    raise SystemExit(main())
