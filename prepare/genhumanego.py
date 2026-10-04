"""Prepare genrobot2025/Gen-HumanEgo episodes as episode sidecars (rig ego_head, no recorded state).

    python -m prepare genhumanego prepare --episodes configs/slices/genhumanego.txt --out EPISODES [--raw RAW]
        [--jobs N] [--force] [--keep-mcap]

An episode list has one MCAP repo path per line. Each episode is one MCAP recorded by a six-camera DAS-Ego
headset. Per episode this downloads the MCAP into RAW/mcap/, copies the forward camera's H.264 stream (camera2)
into RAW/<id>/cam2.mp4 without re-encoding (packets before the first keyframe are dropped, so mp4 frame i is
message i), each frame at the time the headset recorded it (the MCAP's log time of its message; the MCAP carries
no frame rate, so none is assumed), keeps those times in RAW/<id>/cam2_times.npy, and keeps in RAW/<id>/meta.json
the MCAP's own annotation (a goal and timed subtasks with success flags), its time-range validity and
frame-validity records, the camera's timing (gaps, and drift_s, the last frame's distance from a 30 fps grid) and
its calibration (which the board's hand-pose runner reads). The MCAP is then deleted unless --keep-mcap. It writes
EPISODES/episode_<id>/ with context.json (the goal as the instruction and the timed subtasks as
annotation_subtasks, both claims for the model to check), sources.json (pointing at cam2.mp4) and times.npz (the
recorded frame times). A copy extracted before the recorded times were kept has no times.npz, and its
context.source.frame_times says its times are nominal. The dataset asks you to accept its terms on Hugging Face,
so HF_TOKEN must be set.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from label.atomic import write_atomic
from prepare import cli
from prepare import formats
from prepare import hub
from prepare.remux import remux  # each frame's recorded time as its pts (shared with ABC-130k, RealOmni)

REPO = "genrobot2025/Gen-HumanEgo"
CAMERA_TOPIC = "/robot0/sensor/camera2/compressed"
CALIB_TOPIC = "/robot0/sensor/camera2/camera_info"
ANNOTATION_TOPIC = "/robot0/annotation_v2"
CAMERA_DESC = ("the forward-facing fisheye camera on the headset the person wears (camera2 of the six-camera "
               "DAS-Ego rig), looking out and down at their hands and the work in front of them")
TIMES = "cam2_times.npy"    # beside cam2.mp4: each kept frame's recorded time, s after the first
FRAME_TIMES_RECORDED = "recorded: each frame's MCAP log time"
FRAME_TIMES_NOMINAL = ("nominal: frame index / 30 fps (this copy was extracted before the recorded frame times were "
                       "kept, so they are not known)")


def _annotation(a) -> dict | None:
    if a is None:
        return None
    segs = []
    for s in a.segments_info:
        segs.append(dict(id=s.segment_id, t0=s.start_time_s, t1=s.end_time_s, label=s.fine_label,
                         subs=[dict(id=x.segment_id, t0=x.start_time_s, t1=x.end_time_s, label=x.fine_label,
                                    detail=x.fine_label_detail, ok=x.is_success) for x in s.sub_segments_info]))
    return dict(video_id=a.video_id, bold_mark=a.bold_mark, sst=a.sst, env=a.environment_description, segments=segs)


def extract(mcap: Path, raw: Path, rel: str) -> None:
    """cam2.mp4 and meta.json for one episode MCAP."""
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory
    raw.mkdir(parents=True, exist_ok=True)
    ts, frames, ann, trv, fv_invalid, calib = [], [], None, None, 0, None
    with open(mcap, "rb") as fh:
        r = make_reader(fh, decoder_factories=[DecoderFactory()])
        summ = r.get_summary()
        started = False
        for _, ch, msg, d in r.iter_decoded_messages(topics=[CAMERA_TOPIC, CALIB_TOPIC, ANNOTATION_TOPIC,
                                                             "/robot0/time_range_validity", "/robot0/frame_validity"]):
            if ch.topic == CAMERA_TOPIC:
                if not started:
                    b = d.data
                    o = 4 if b.startswith(b"\x00\x00\x00\x01") else 3 if b.startswith(b"\x00\x00\x01") else 0
                    if not b or (b[o] & 0x1F) != 7:
                        continue            # before the first SPS: dropped, so mp4 frame i is ts[i]
                    started = True
                frames.append(bytes(d.data))
                ts.append(msg.log_time)
            elif ch.topic == CALIB_TOPIC:
                # the head camera's calibration (Double Sphere: fx, fy, cx, cy, xi, alpha), used by the hand pose
                calib = dict(w=d.width, h=d.height, model=d.distortion_model, D=list(d.D), T=list(d.T_b_c),
                             frame=d.frame_id)
            elif ch.topic == ANNOTATION_TOPIC:
                ann = d
            elif ch.topic == "/robot0/time_range_validity":
                trv = d
            else:
                fv_invalid += 0 if d.is_valid else 1
    if not frames:
        raise RuntimeError(f"{rel}: the forward camera has no frame from its first SPS on")
    # each frame at the time the headset recorded it: the MCAP carries no frame rate, and the camera can drop a
    # frame or run off 30 fps, so a constant rate would put every later frame at the wrong time
    t = (np.asarray(ts, dtype=np.int64) - ts[0]) / 1e9
    remux(frames, t, "h264", raw / "cam2.mp4")
    np.save(raw / TIMES, t)
    meta = {"rel": rel, "dur": (summ.statistics.message_end_time - summ.statistics.message_start_time) / 1e9,
            "ann": _annotation(ann), "fv_invalid": fv_invalid, "calib2": calib}
    if len(ts) > 1:
        dt = np.diff(np.asarray(ts, dtype=np.int64)) / 1e6
        # drift_s: how far the last frame's recorded time is from where a 30 fps grid would put it
        meta["cam2_dt"] = dict(clock="MCAP log time of each camera message", med=float(np.median(dt)),
                               max=float(dt.max()), n_gap50=int((dt > 50).sum()), n_gap100=int((dt > 100).sum()),
                               span_s=float(t[-1]), drift_s=float(t[-1] - (len(t) - 1) / 30))
    if trv is not None:
        meta["invalid_ranges"] = [dict(t0=x.start_time_s, t1=x.end_time_s, code=x.invalid_code,
                                       msg=x.invalid_message) for x in trv.invalid_ranges]
    (raw / "meta.json").write_text(json.dumps(meta))


# an uploaded MCAP from this headset (its forward camera and its annotation) is read by this adapter
UPLOAD = None


def recognizes(topics: list[str]) -> bool:
    return CAMERA_TOPIC in topics and ANNOTATION_TOPIC in topics


def convert_upload(item: dict, ep: Path) -> dict:
    """The forward camera, and the recording's own goal and timed steps (claims for the model to check)."""
    extract(Path(item["file"]), ep / "source", item["name"])
    ctx = write_sidecar(ep / "source", ep)
    ctx["episode_id"] = ep.name
    if not any(ctx.get("task_label") or []):
        ctx["task_label"] = [item["name"]]
    (ep / "instruction.txt").write_text((ctx.get("instruction") or "") + "\n")
    return ctx


def write_sidecar(raw: Path, dst: Path) -> dict:
    import av
    meta = json.loads((raw / "meta.json").read_text())
    ann = meta.get("ann") or {}
    with av.open(str(raw / "cam2.mp4")) as c:
        st = c.streams.video[0]
        pts = sorted(p.pts for p in c.demux(st) if p.pts is not None and not p.is_discard)
        w, h = st.codec_context.width, st.codec_context.height
    n = len(pts)
    recorded = np.load(raw / TIMES) if (raw / TIMES).exists() else None
    if recorded is not None and len(recorded) != n:
        raise RuntimeError(f"{raw.name}: {n} frames in cam2.mp4 but {len(recorded)} recorded frame times")
    fps = (formats.measured_fps(recorded) if recorded is not None else None) or 30.0
    subs = []
    for seg in ann.get("segments") or []:
        for x in seg.get("subs") or []:
            subs.append({"t0": float(x["t0"]), "t1": float(x["t1"]), "label": x["label"],
                         "ok": bool(x.get("ok", True))})
    ctx = {"dataset": REPO, "profile": "ego_head", "state_kind": "none", "episode_id": f"episode_{raw.name}",
           "robot_type": None, "fps": fps, "n_state_frames": n,
           "instruction": (ann.get("bold_mark") or "").strip() or None,
           "annotation_subtasks": subs,
           "task_label": [ann.get("sst") or ""],
           "cameras": {"exo": {"name": "head", "width": w, "height": h, "desc": CAMERA_DESC}},
           "source": {"mcap": meta.get("rel"), "duration_s": meta.get("dur"), "camera_timing": meta.get("cam2_dt"),
                      "dataset_invalid_ranges": meta.get("invalid_ranges"),
                      "frame_valid_false": meta.get("fv_invalid"),
                      "frame_times": FRAME_TIMES_RECORDED if recorded is not None else FRAME_TIMES_NOMINAL}}
    dst.mkdir(parents=True, exist_ok=True)
    if recorded is not None:
        # the recorded times place each frame; the file's own pts (the same times, rounded to its time base) find it
        np.savez(dst / "times.npz", exo=np.asarray(recorded, dtype=np.float64), exo_pts=np.asarray(pts, dtype=np.int64))
        ctx["real_times"] = "times.npz"
    src = {"exo": {"packed": str((raw / "cam2.mp4").resolve()), "base_s": 0.0, "n_frames": n}}
    (dst / "sources.json").write_text(json.dumps(src, indent=1))
    write_atomic(dst / "context.json", ctx, indent=1)
    return ctx


def prepare_one(rel: str, raw_root: Path, out: Path, force: bool = False, keep_mcap: bool = False) -> str:
    rid = Path(rel).stem
    dst = out / f"episode_{rid}"
    if not force and (dst / "context.json").exists():
        return "skip"
    raw = raw_root / rid
    if not (raw / "meta.json").exists():
        mcap = Path(hub.download(REPO, rel, raw_root / "mcap"))
        extract(mcap, raw, rel)
        if not keep_mcap:
            mcap.unlink()
    write_sidecar(raw, dst)
    return "ok"


def main() -> int:
    ap, sub = cli.parser("genhumanego", __doc__)
    p = cli.add_prepare(sub, "genhumanego", "one MCAP repo path per line", jobs=4)
    p.add_argument("--keep-mcap", action="store_true", help="keep each downloaded MCAP once it is extracted")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    return cli.run(cli.read_list(a.episodes), lambda r: prepare_one(r, a.raw, a.out, a.force, a.keep_mcap), a.jobs)


if __name__ == "__main__":
    raise SystemExit(main())
