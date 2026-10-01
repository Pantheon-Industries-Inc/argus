"""Prepare genrobot2025/10Kh-RealOmin-OpenData episodes as episode sidecars (rig handheld_gripper, state ee_pose).

    python -m prepare realomin sample --out EPISODES --hours 5.5 --seed 61 [--list LIST] [--raw RAW] [--jobs N]
    python -m prepare realomin prepare --episodes configs/slices/realomin.txt --out EPISODES [--raw RAW] [--jobs N]
        [--force] [--keep-mcap]

An episode list has one MCAP repo path per line (the .mcap suffix is optional). Each episode is one MCAP holding,
per handheld DAS gripper (robot0 = left, robot1 = right): an H.264 fisheye camera stream, the VIO end-effector
pose, and the magnetic-encoder gripper opening. The topic map, H.264 muxing and pose layout follow the
public-dataset-adapter's adapter for this dataset (adapters/datasets/genrobot2025__10Kh-RealOmin-OpenData.py,
v1.3.0), written here as episode sidecars at source resolution.

`prepare` downloads each MCAP into RAW/<repo path>, writes EPISODES/episode_<repo path>/ and then deletes the MCAP
unless --keep-mcap. Per episode: left.mp4 and right.mp4 (the H.264 stream copied, not re-encoded), times.npz
(each camera's real capture times and mp4 pts), kmap_right.npy (the right camera frame nearest each left frame),
state.npz (per left frame, per gripper: x y z m, roll pitch yaw rad, opening), context.json and sources.json. The
dataset ships no per-episode instruction; the clip's task folder name is given as a coarse instruction, with a
note saying exactly that.

`sample` takes the same number of clips from each top-level task folder, each from a random sub-folder and a
random file in it, prepares them as it draws (a clip's length is known only once it is read) until --hours of
clip time, and writes the list of what it prepared to LIST (default EPISODES.txt). The dataset asks you to accept
its terms on Hugging Face, so HF_TOKEN must be set.
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
from prepare import formats
from prepare.remux import remux  # each frame's capture time as its pts

REPO = "genrobot2025/10Kh-RealOmin-OpenData"
CAMERA_TOPICS = {"/robot0/sensor/camera0/compressed": "left", "/robot1/sensor/camera0/compressed": "right"}
POSE_TOPICS = {"/robot0/vio/eef_pose": "left", "/robot1/vio/eef_pose": "right"}
GRIPPER_TOPICS = {"/robot0/sensor/magnetic_encoder": "left", "/robot1/sensor/magnetic_encoder": "right"}
CAMERA_DESC = {
    "left": ("the fisheye camera carried on the LEFT-hand gripper, looking along its fingers, which are at the "
             "bottom of the image."),
    "right": ("the fisheye camera carried on the RIGHT-hand gripper, looking along its fingers, which are at the "
              "bottom of the image."),
}


def ls(path: str) -> list[dict]:
    return hub.ls(REPO, path)


def task_of(path: str) -> str:
    parents = path.removesuffix(".mcap").split("/")[:-1]
    semantic = [p for p in parents if not p.isdigit() and not p.startswith("[STAGE")]
    return (semantic[-1] if semantic else parents[0]).replace("_", " ")


def quat_to_rpy(q: np.ndarray) -> np.ndarray:
    """(x, y, z, w) -> roll, pitch, yaw with R = Rz(yaw) Ry(pitch) Rx(roll), label.state's convention."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.stack([roll, pitch, yaw], axis=1)


def nal_types(b: bytes) -> set[int]:
    out, i = set(), 0
    while True:
        j = b.find(b"\x00\x00\x01", i)
        if j < 0 or j + 3 >= len(b):
            return out
        out.add(b[j + 3] & 0x1F)
        i = j + 3


def mux(packets: list[bytes], ts: list[int], out: Path) -> list[int]:
    """The H.264 packets from the first one carrying SPS, PPS and an IDR frame, stream-copied into out with each
    packet's own capture time as its pts (prepare/remux.py); returns the capture times of the packets written."""
    first = next(i for i, p in enumerate(packets) if {7, 8, 5} <= nal_types(p))
    packets, ts = packets[first:], ts[first:]
    t = np.asarray(ts, dtype=np.int64)
    remux(packets, (t - t[0]) / 1e9, "h264", out)
    return ts


# an uploaded MCAP in this layout (both grippers' cameras and poses) is read by this adapter
UPLOAD = "mcap"


def recognizes(topics: list[str]) -> bool:
    return all(t in topics for t in CAMERA_TOPICS) and all(t in topics for t in POSE_TOPICS)


def convert_upload(item: dict, ep: Path) -> dict:
    ctx = convert(item["file"], ep, "upload/" + item["name"])
    # the dataset's task text is its own folder path on Hugging Face; an uploader's folder name is not a task, so the
    # episode goes to the model with no instruction rather than an invented one
    for k in ("instruction", "instruction_note"):
        ctx.pop(k, None)
    ctx["task_label"] = [item["name"]]
    (ep / "instruction.txt").write_text("\n")
    return ctx


def convert(mcap_path: Path, ep: Path, rel: str) -> dict:
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory
    cam = {v: ([], []) for v in CAMERA_TOPICS.values()}
    pose = {v: ([], []) for v in POSE_TOPICS.values()}
    grip = {v: ([], []) for v in GRIPPER_TOPICS.values()}
    with mcap_path.open("rb") as f:
        for _, ch, msg, d in make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages(
                topics=list(CAMERA_TOPICS) + list(POSE_TOPICS) + list(GRIPPER_TOPICS)):
            t = msg.log_time
            if ch.topic in CAMERA_TOPICS:
                cam[CAMERA_TOPICS[ch.topic]][0].append(t)
                cam[CAMERA_TOPICS[ch.topic]][1].append(bytes(d.data))
            elif ch.topic in POSE_TOPICS:
                p = d.pose
                pose[POSE_TOPICS[ch.topic]][0].append(t)
                pose[POSE_TOPICS[ch.topic]][1].append([p.position.x, p.position.y, p.position.z, p.orientation.x,
                                                       p.orientation.y, p.orientation.z, p.orientation.w])
            else:
                grip[GRIPPER_TOPICS[ch.topic]][0].append(t)
                grip[GRIPPER_TOPICS[ch.topic]][1].append(float(d.value))
    ep.mkdir(parents=True, exist_ok=True)
    cts = {}
    for v in ("left", "right"):
        cts[v] = np.asarray(mux(cam[v][1], cam[v][0], ep / f"{v}.mp4"), dtype=np.int64)
    import av
    npts = {}
    for v in ("left", "right"):
        with av.open(str(ep / f"{v}.mp4")) as c:
            st = c.streams.video[0]
            npts[v] = np.asarray(sorted(p.pts for p in c.demux(st) if p.pts is not None and not p.is_discard),
                                 dtype=np.int64)
            wh = (st.codec_context.width, st.codec_context.height)
        if len(npts[v]) != len(cts[v]):         # remux wrote one packet per frame message, each at its capture time
            raise RuntimeError(f"{v}.mp4: {len(npts[v])} frames in the file but {len(cts[v])} capture times")
    t0 = int(cts["left"][0])
    left_t = cts["left"]
    state = []
    for v in ("left", "right"):
        pt = np.asarray(pose[v][0], dtype=np.int64)
        pv = np.asarray(pose[v][1], dtype=np.float64)
        gt = np.asarray(grip[v][0], dtype=np.int64)
        gv = np.asarray(grip[v][1], dtype=np.float64)
        ip, ig = formats.nearest(pt, left_t), formats.nearest(gt, left_t)
        state.append(np.concatenate([pv[ip, :3], quat_to_rpy(pv[ip, 3:7]), gv[ig, None]], axis=1))
    state = np.concatenate(state, axis=1).astype(np.float32)
    np.savez(ep / "state.npz", state=state)
    kmap = formats.nearest(cts["right"], left_t).astype(np.int64)
    np.save(ep / "kmap_right.npy", kmap)
    np.savez(ep / "times.npz", left=(left_t - t0) / 1e9, left_pts=npts["left"],
             right=(cts["right"] - t0) / 1e9, right_pts=npts["right"])
    task = task_of(rel)
    ctx = {"dataset": REPO, "profile": "handheld_gripper", "state_kind": "ee_pose", "episode_id": ep.name,
           "robot_type": "two handheld GenDAS grippers with wrist fisheye cameras",
           "fps": formats.measured_fps((left_t - t0) / 1e9) or 30.0,     # the left camera's real rate, not assumed
           "n_state_frames": int(len(left_t)), "real_times": "times.npz", "instruction": task,
           "instruction_note": ("This instruction is only the task category of the folder the dataset files the clip "
                                "under; the dataset ships no per-episode instruction, so judge alignment loosely."),
           "task_label": [task],
           # checked against the gripper camera (fingers pinched on a zipper tab at ~0, wide open at 0.103)
           "gripper_value": ("the measured opening width in metres, about 0 = jaws shut and about 0.10 = fully open "
                             "(checked against the gripper camera frames)"),
           "gripper_range": [0.0, 0.10],
           "cameras": {v: {"name": v, "width": wh[0], "height": wh[1], "desc": CAMERA_DESC[v]}
                       for v in ("left", "right")},
           "source": {"mcap": rel}}
    src = {"left": {"packed": str((ep / "left.mp4").resolve()), "base_s": 0.0, "n_frames": int(len(npts["left"]))},
           "right": {"packed": str((ep / "right.mp4").resolve()), "base_s": 0.0, "n_frames": int(len(npts["right"])),
                     "kmap": "kmap_right.npy"}}
    # every other number the grippers record (each one's IMU), under the dataset's names (formats.mcap_signals); the
    # pose and the opening are the state, and /robotN/sim/robot_info repeats the pose
    used = {**{t: {"pose.position", "pose.orientation"} for t in POSE_TOPICS}, **{t: {""} for t in GRIPPER_TOPICS},
            **{t.replace("/vio/eef_pose", "/sim/robot_info"): None for t in POSE_TOPICS}}
    formats.write_signals(ep, ctx, formats.mcap_signals([mcap_path], left_t / 1e9, used))
    (ep / "sources.json").write_text(json.dumps(src, indent=1))
    (ep / "context.json").write_text(json.dumps(ctx, indent=1))
    return ctx


def episode_dir_name(rel: str) -> str:
    """Clutter Tidy-Up [Stage2]/00001/01751.mcap -> episode_Clutter_Tidy_Up_Stage2_00001_01751"""
    return "episode_" + re.sub(r"[^A-Za-z0-9]+", "_", rel.removesuffix(".mcap")).strip("_")


def prepare_one(rel: str, raw: Path, out_root: Path, force: bool = False, keep_mcap: bool = False) -> str:
    ep = out_root / episode_dir_name(rel)
    if not force and (ep / "context.json").exists():
        return "skip"
    mcap = Path(hub.download(REPO, rel, raw))
    convert(mcap, ep, rel)
    if not keep_mcap:
        mcap.unlink()
    return "ok"


def drawn(rel: str, raw: Path, out_root: Path, keep_mcap: bool) -> dict | None:
    """Prepare one drawn clip; its context, or None when it cannot be prepared (printed)."""
    try:
        prepare_one(rel, raw, out_root, keep_mcap=keep_mcap)
        return json.loads((out_root / episode_dir_name(rel) / "context.json").read_text())
    except Exception as e:  # a drawn clip that cannot be read is a finding about the data; listed, not hidden
        print(f"FAILED {rel}: {type(e).__name__}: {e}"[:400], file=sys.stderr, flush=True)
        return None


def cmd_sample(a) -> int:
    a.out.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(a.seed)
    tops = sorted(x["path"] for x in ls("") if x["type"] == "directory")
    subdirs = {t: [x["path"] for x in ls(t) if x["type"] == "directory"] for t in tops}
    print({t: len(v) for t, v in subdirs.items()}, flush=True)

    def pick(t: str) -> str | None:
        path = t
        for _ in range(4):                      # descend to a folder that holds .mcap files
            kids = ls(path)
            files = [x["path"] for x in kids if x["type"] == "file" and x["path"].endswith(".mcap")]
            if files:
                return rnd.choice(files)
            dirs = [x["path"] for x in kids if x["type"] == "directory"]
            if not dirs:
                return None
            path = rnd.choice(dirs)
        return None

    got, secs, seen = [], 0.0, set()
    with ThreadPoolExecutor(a.jobs) as ex:
        while secs < a.hours * 3600:
            batch = []
            for t in tops:                          # the same number from every top-level task folder
                rel = pick(t)
                if rel and rel not in seen:
                    seen.add(rel)
                    batch.append(rel)
            for ctx in ex.map(lambda r: drawn(r, a.raw, a.out, a.keep_mcap), batch):
                if ctx:
                    got.append(ctx)
                    secs += ctx["n_state_frames"] / ctx["fps"]
            print(f"clips {len(got)} hours {secs / 3600:.2f}", flush=True)
    lst = a.list or Path(f"{a.out}.txt")
    lst.write_text("\n".join(sorted(c["source"]["mcap"] for c in got)) + "\n")
    print(json.dumps({"clips": len(got), "hours": round(secs / 3600, 2), "list": str(lst),
                      "tasks": sorted({c["task_label"][0] for c in got})}), flush=True)
    return 0


def main() -> int:
    ap, sub = cli.parser("realomin", __doc__)
    s = sub.add_parser("sample", help="draw and prepare clips until about --hours, and write their list")
    s.add_argument("--out", type=Path, required=True, help="where the episode folders are written")
    s.add_argument("--list", type=Path, default=None, help="the episode list to write (default OUT.txt)")
    s.add_argument("--raw", type=Path, default=Path("data/raw/realomin"), help="where downloaded files are kept")
    s.add_argument("--hours", type=float, default=5.5)
    s.add_argument("--jobs", type=int, default=4, help="clips prepared at once (default 4)")
    s.add_argument("--seed", type=int, default=61)
    s.add_argument("--keep-mcap", action="store_true", help="keep each downloaded MCAP once its episode is written")
    p = cli.add_prepare(sub, "realomin", "one MCAP repo path per line (.mcap optional)", jobs=4)
    p.add_argument("--keep-mcap", action="store_true", help="keep each downloaded MCAP once its episode is written")
    a = ap.parse_args()
    if a.cmd == "sample":
        return cmd_sample(a)
    rels = [r if r.endswith(".mcap") else r + ".mcap" for r in cli.read_list(a.episodes)]
    a.out.mkdir(parents=True, exist_ok=True)
    return cli.run(rels, lambda r: prepare_one(r, a.raw, a.out, a.force, a.keep_mcap), a.jobs)


if __name__ == "__main__":
    raise SystemExit(main())
