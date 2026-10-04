"""Prepare XDOF/ABC-130k episodes as episode sidecars (rig teleop_arms, state joints).

    python -m prepare abc130k sample --list LIST --per-task 5 --seed 0 [--split train] [--jobs N]
    python -m prepare abc130k prepare --episodes configs/slices/abc130k.txt --out EPISODES [--raw RAW] [--jobs N]
        [--force] [--keep-mcap]

An episode list has one repo path per line, data/<split>/<task>/<episode>. `sample` lists the repo's task
folders, draws --per-task random episodes from each (seeded per task) and writes LIST. `prepare` downloads each
listed episode's episode.mcap into RAW/<repo path>/, writes EPISODES/<episode>/ and then deletes the MCAP unless
--keep-mcap (every frame of the three cameras is in the episode's mp4s, and the arm streams are in state.npz,
interpolated onto the top camera's frames). The dataset asks you to accept its terms on Hugging Face, so
HF_TOKEN must be set.

ABC-130k ships one MCAP per episode: per-arm joint state and commanded joints (the operator's leader arm),
gripper aperture (0 shut, 1 open), and one compressed-video message per camera frame, every message with its
own absolute timestamp. Per episode this writes exo.mp4, left.mp4, right.mp4, times.npz, kmap_<view>.npy where
needed, state.npz, sources.json, context.json and instruction.txt. Nothing is re-encoded:

- Each camera's frames are copied packet for packet into an mp4 whose pts are the frames' real capture
  times (microsecond time base, relative to the first frame of any camera). times.npz keeps those times and
  exact pts, label/frames.py decodes frame k by its exact pts, and every time shown to the model or played in
  the viewer is real capture time.
- The top camera is the anchor. Each wrist camera frame is paired with the top frame nearest in real
  time (kmap in sources.json when the pairing is not the identity), and joint/gripper values are
  interpolated onto the top camera's timestamps, so row k of state.npz is the robot at top frame k.
- context["stream_checks"] gets the stream timing facts (gaps, pairing offsets) measured from the timestamps.
ZED-X stations have a stereo top camera; its left eye is used as the top view.
"""
from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from label.atomic import write_atomic
from prepare import cli
from prepare import hub
from prepare import formats
from prepare.remux import remux  # each frame's real capture time as its pts (shared with RealOmni, Gen-HumanEgo)

REPO = "XDOF/ABC-130k"
TIME_BASE_DEN = 1_000_000   # microseconds: some ABC frames are stamped only 1 us apart
NOMINAL_FPS = 30        # only a fallback and a one-frame tolerance; each episode's rate is measured (measured_fps)
TOP_TOPICS = ("/top-camera", "/top-left-camera")
VIEW_TOPIC = {"left": "/left-wrist-camera", "right": "/right-wrist-camera"}
ARM = ("/left-arm-state", "/left-ee-state", "/right-arm-state", "/right-ee-state")
ARM_ACT = ("/left-arm-action", "/left-ee-action", "/right-arm-action", "/right-ee-action")
# what each camera is, checked on the dataset's frames (per-episode facts handed to the model)
CAMERA_DESC = {
    "exo": "a fixed camera looking at the workspace and both arms from above. Use it for the scene "
           "layout, object locations and where things end up",
    "left": "the camera mounted on the LEFT arm's gripper; that gripper's fingertips are at the bottom of its image",
    "right": "the camera mounted on the RIGHT arm's gripper; that gripper's fingertips are at the bottom of its image",
}


def _ts(msg) -> int:
    return int(msg.timestamp.seconds) * 1_000_000_000 + int(msg.timestamp.nanos)


def cmd_sample(args) -> int:
    from huggingface_hub import HfApi
    api = HfApi(token=hub.token())
    tasks = sorted(f.path for f in api.list_repo_tree(REPO, repo_type="dataset", path_in_repo=f"data/{args.split}"))

    def one(t):
        eps = sorted(f.path for f in api.list_repo_tree(REPO, repo_type="dataset", path_in_repo=t))
        return random.Random(f"{args.seed}:{t}").sample(eps, min(args.per_task, len(eps)))

    picked = []
    with ThreadPoolExecutor(args.jobs) as ex:
        for eps in ex.map(one, tasks):
            picked += eps
    args.list.write_text("\n".join(sorted(picked)) + "\n")
    print(f"{len(tasks)} tasks, {len(picked)} episodes -> {args.list}")
    return 0


def read_mcap(path: Path) -> dict:
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory
    video, robot, meta, instr = {}, {}, {}, None
    with open(path, "rb") as fh:
        r = make_reader(fh, decoder_factories=[DecoderFactory()])
        for m in r.iter_metadata():
            meta.update(dict(m.metadata))
        for _, ch, _, dec in r.iter_decoded_messages():
            t = ch.topic
            if t.endswith("camera"):
                v = video.setdefault(t, {"t": [], "data": [], "format": dec.format})
                v["t"].append(_ts(dec))
                v["data"].append(bytes(dec.data))
            elif t in ARM or t in ARM_ACT:
                d = robot.setdefault(t, {"t": [], "pos": []})
                d["t"].append(_ts(dec))
                d["pos"].append(list(dec.position))
            elif t == "/instruction":
                instr = str(dec.data)
    return {"video": video, "robot": robot, "meta": meta, "instruction": instr}


def measured_fps(t_ns: np.ndarray) -> float:
    """The anchor camera's real frame rate from its capture times (ns). Stations record at 30 or 60 Hz, and the
    harness samples one frame per second by counting fps frames, so a fixed 30 sampled 60 Hz episodes every 0.5 s."""
    d = np.diff(np.asarray(t_ns, dtype=np.float64))
    d = d[d > 0]
    return round(1e9 / float(np.median(d)), 2) if len(d) else float(NOMINAL_FPS)


def interp(stream: dict, q: np.ndarray) -> tuple[np.ndarray | None, tuple[float, float] | None]:
    """A stream's rows on the frame times q (ns), as the readers place an arm (formats.fill_rows): (None, the gap in
    seconds) when it has no reading for longer than the slack the readers allow, which a line would draw as motion."""
    return formats.fill_rows(q / 1e9, np.asarray(stream["t"], dtype=np.float64) / 1e9,
                             np.asarray(stream["pos"], dtype=np.float64))


def read_fields(state, action) -> dict:
    """{topic: {"position"}} of the arm channels read as the state, and of the command channels when they were read as
    the action, for mcap_signals to leave out; positions that were not read stay signals (formats.state_fields)."""
    return {t: {"position"} for t in (ARM if state is not None else ()) + (ARM_ACT if action is not None else ())}


def gaps(t_ns: list[int]) -> dict:
    """Timing facts of one stream: frames missing (an interval over 1.5x the median), and frames
    stamped less than 1 ms after the previous one (a duplicated or glitched timestamp)."""
    d = np.diff(np.asarray(t_ns, dtype=np.int64)) / 1e6
    med = float(np.median(d)) if len(d) else 0.0
    return {"n": len(t_ns), "median_dt_ms": round(med, 2), "max_dt_ms": round(float(d.max()), 1) if len(d) else 0.0,
            "n_gaps": int((d > 1.5 * med).sum()) if len(d) else 0,
            "n_near_duplicate_stamps": int((d < 1.0).sum()) if len(d) else 0}


def prepare_one(ep_path: str, raw_root: Path, out_root: Path, force: bool, keep_mcap: bool = False) -> str:
    task, ep_name = ep_path.split("/")[-2], ep_path.split("/")[-1]
    ep_dir = out_root / ep_name
    if not force and (ep_dir / "context.json").exists():
        return "skip"
    mcap = Path(hub.download(REPO, f"{ep_path}/episode.mcap", raw_root))
    convert(mcap, ep_dir, ep_name, task, split=ep_path.split("/")[1])
    if not keep_mcap:
        mcap.unlink()
    return "ok"


# Uploads use structural MCAP fields. Published preparation and sampling stay available here.
UPLOAD = None


def recognizes(topics: list[str]) -> bool:
    return any(t in topics for t in TOP_TOPICS) and all(t in topics for t in VIEW_TOPIC.values()) \
        and all(t in topics for t in ARM)


def convert_upload(item: dict, ep: Path) -> dict:
    return convert(item["file"], ep, ep.name, task=item["name"])


def convert(mcap: Path, ep_dir: Path, ep_name: str, task: str, split: str | None = None) -> dict:
    """One ABC-layout MCAP on disk into an episode sidecar dir; returns its context."""
    d = read_mcap(mcap)
    top = next((t for t in TOP_TOPICS if t in d["video"]), None)
    if top is None or any(VIEW_TOPIC[v] not in d["video"] for v in VIEW_TOPIC):
        raise RuntimeError(f"{ep_name}: missing camera topics, has {sorted(d['video'])}")
    ep_dir.mkdir(parents=True, exist_ok=True)
    topics = {"exo": top, **VIEW_TOPIC}
    t_top = np.asarray(d["video"][top]["t"], dtype=np.int64)
    # time zero is the first frame of any camera, so every camera's times (and pts) are >= 0
    t0 = int(min(d["video"][t]["t"][0] for t in topics.values()))
    sources, times, checks = {}, {}, {"streams": {}}
    for v, topic in topics.items():
        vid = d["video"][topic]
        out = ep_dir / f"{v}.mp4"
        tv = np.asarray(vid["t"], dtype=np.int64)
        times[v] = (tv - t0) / 1e9
        times[f"{v}_pts"] = remux(vid["data"], times[v], vid["format"], out)
        n = len(tv)
        src = {"packed": str(out.resolve()), "base_s": 0.0, "n_frames": n, "camera_key": topic,
               "codec": vid["format"], "pts_from": "times.npz"}
        if v != "exo":
            idx = formats.nearest(tv, t_top)
            off = np.abs(tv[idx] - t_top) / 1e6
            checks["streams"][topic] = {**gaps(vid["t"]), "pair_offset_ms_max": round(float(off.max()), 1),
                                        "pair_offset_ms_median": round(float(np.median(off)), 2)}
            if not (len(tv) == len(t_top) and np.array_equal(idx, np.arange(len(t_top)))):
                np.save(ep_dir / f"kmap_{v}.npy", idx.astype(np.int32))
                src["kmap"] = f"kmap_{v}.npy"
        else:
            checks["streams"][topic] = gaps(vid["t"])
        sources[v] = src
    q = t_top.astype(np.float64)
    placed = {t: interp(d["robot"][t], q) for t in ARM + ARM_ACT}
    gap = next(((t, placed[t][1]) for t in ARM if placed[t][1]), None)
    state = action = None
    if gap is None:
        L, Lg, R, Rg = (placed[t][0] for t in ARM)
        state = np.concatenate([L[:, :6], Lg[:, :1], R[:, :6], Rg[:, :1]], axis=1).astype(np.float32)
        if not any(placed[t][1] for t in ARM_ACT):
            La, Lga, Ra, Rga = (placed[t][0] for t in ARM_ACT)
            action = np.concatenate([La[:, :6], Lga[:, :1], Ra[:, :6], Rga[:, :1]], axis=1).astype(np.float32)
    for t in ARM:
        checks["streams"][t] = gaps(d["robot"][t]["t"])
    meta = d["meta"]
    cams = {"exo": ("top", top), "left": ("left", VIEW_TOPIC["left"]), "right": ("right", VIEW_TOPIC["right"])}
    station = "ZED-X" if top == "/top-left-camera" else "RealSense"
    context = {
        "dataset": REPO,
        "profile": "teleop_arms",
        "state_kind": "joints" if state is not None else "none",
        # checked by eye against the wrist frames (fingers wide at 1.00, pinched at 0.08) and across 40
        # episodes (arms start at a median 0.99 and close to a median minimum of 0.04)
        "gripper_value": "0 = jaws shut, 1 = fully open (checked against the wrist frames)",
        "gripper_range": [0.0, 1.0],
        "episode_id": ep_name,
        "split": split,
        "robot_type": "bimanual YAM station (2x 6-DoF YAM arms, parallel-jaw grippers)",
        "fps": measured_fps(t_top),
        "task_label": [task],
        "instruction": (d["instruction"] or meta.get("task_name") or "").strip(),
        "instruction_note": ("This instruction is the task name the recording rig stored for the episode "
                             "(its /instruction message); it is the goal you grade against."),
        "operator_id": meta.get("operator_id"),
        "station": station,
        "n_state_frames": int(len(q)),
        "real_times": "times.npz",
        "cameras": {v: {"key": topic, "name": name,
                        "width": meta.get(f"{'top' if v == 'exo' else v}_camera_width"),
                        "height": meta.get(f"{'top' if v == 'exo' else v}_camera_height"),
                        "type": meta.get(f"{'top' if v == 'exo' else v}_camera_type"),
                        "desc": CAMERA_DESC[v]}
                    for v, (name, topic) in cams.items()},
        "stream_checks": checks,
    }
    # every other number the arms record (joint velocities and torques), under the dataset's names (formats.mcap_signals)
    formats.write_signals(ep_dir, context, formats.mcap_signals([mcap], t_top / 1e9, read_fields(state, action)))
    if gap is not None:
        formats.no_state(context, formats.StateNote(
            f"Labelled from the cameras, because the recorded arm state {gap[0]} "
            f"{formats.gap_words(gap[1], float(q[0]) / 1e9)}.", "short"))
    if state is not None:
        formats.record_state_groups(context, [(ARM[0], None, 7), (ARM[2], None, 7)])
        np.savez(ep_dir / "state.npz", **({"state": state, "action": action} if action is not None else
                                          {"state": state}))
    np.savez(ep_dir / "times.npz", **times)
    (ep_dir / "sources.json").write_text(json.dumps(sources, indent=2))
    (ep_dir / "instruction.txt").write_text(context["instruction"] + "\n")
    write_atomic(ep_dir / "context.json", context, indent=2)
    return context


def main() -> int:
    ap, sub = cli.parser("abc130k", __doc__)
    s = sub.add_parser("sample", help="draw episodes from every task folder and write their list")
    s.add_argument("--list", type=Path, required=True, help="the episode list to write")
    s.add_argument("--per-task", type=int, default=5)
    s.add_argument("--split", default="train")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--jobs", type=int, default=8, help="task folders listed at once (default 8)")
    p = cli.add_prepare(sub, "abc130k", "one repo path per line, data/<split>/<task>/<episode>", jobs=6)
    p.add_argument("--keep-mcap", action="store_true", help="keep each downloaded MCAP once its episode is written")
    a = ap.parse_args()
    if a.cmd == "sample":
        return cmd_sample(a)
    a.out.mkdir(parents=True, exist_ok=True)
    return cli.run(cli.read_list(a.episodes), lambda e: prepare_one(e, a.raw, a.out, a.force, a.keep_mcap), a.jobs)


if __name__ == "__main__":
    raise SystemExit(main())
