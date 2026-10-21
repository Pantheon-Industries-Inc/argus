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
so HF_TOKEN must be set. If camera2 cannot be read, a readable headset camera keeps its own identity and calibration;
publisher timed claims remain source metadata because their clock is not established for that fallback.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from label.atomic import write_atomic
from prepare import cli
from prepare import formats
from prepare import hub
from prepare.frame_rate import capture_seconds
from prepare.mcap_source import decoded_messages
from prepare.remux import RemuxSourceError, remux  # each frame's recorded time as its pts (shared with ABC-130k, RealOmni)
from label.presented import presented_pts

REPO = "genrobot2025/Gen-HumanEgo"
CAMERA_TOPIC = "/robot0/sensor/camera2/compressed"
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
    """Extract the forward camera, or a readable headset camera with its own identity and calibration."""
    from mcap.reader import make_reader
    raw.mkdir(parents=True, exist_ok=True)
    ann, trv, fv_invalid, calibrations = None, None, 0, {}
    source_claims, source_failures = [], {}
    with open(mcap, "rb") as fh:
        r = make_reader(fh)
        summ = r.get_summary()
        camera_topics = sorted({ch.topic for ch in summ.channels.values()
                                if re.fullmatch(r"/robot0/sensor/camera[0-9]+/compressed", ch.topic)})
        calib_topics = [topic.rsplit("/", 1)[0] + "/camera_info" for topic in camera_topics]
        fields = {topic: {"width": "integer", "height": "integer", "distortion_model": "text",
                          "D": "number[]", "T_b_c": "number[]", "frame_id": "text"} for topic in calib_topics}
        fields[ANNOTATION_TOPIC] = {"video_id": "text", "bold_mark": "text", "sst": "text",
            "environment_description": "text", "segments_info": "message[]",
            "segments_info.segment_id": "scalar", "segments_info.start_time_s": "number",
            "segments_info.end_time_s": "number", "segments_info.fine_label": "text",
            "segments_info.sub_segments_info": "message[]", "segments_info.sub_segments_info.segment_id": "scalar",
            "segments_info.sub_segments_info.start_time_s": "number", "segments_info.sub_segments_info.end_time_s": "number",
            "segments_info.sub_segments_info.fine_label": "text", "segments_info.sub_segments_info.fine_label_detail": "text",
            "segments_info.sub_segments_info.is_success": "bool"}
        fields["/robot0/time_range_validity"] = {"invalid_ranges": "message[]",
            "invalid_ranges.start_time_s": "number", "invalid_ranges.end_time_s": "number",
            "invalid_ranges.invalid_code": "scalar", "invalid_ranges.invalid_message": "text"}
        fields["/robot0/frame_validity"] = {"is_valid": "bool"}
        for _, ch, msg, d in decoded_messages(r, fields, source_failures):
            if ch.topic in calib_topics:
                calibrations[ch.topic.rsplit("/", 1)[0] + "/compressed"] = dict(
                    w=d.width, h=d.height, model=d.distortion_model, D=list(d.D), T=list(d.T_b_c), frame=d.frame_id)
            else:
                from prepare.mcap_claims import snapshot
                source_claims.append({"topic": ch.topic, "log_ns": int(msg.log_time),
                                      "publish_ns": int(msg.publish_time), "fields": snapshot(d)})
                if ch.topic == ANNOTATION_TOPIC:
                    ann = d
                elif ch.topic == "/robot0/time_range_validity":
                    trv = d
                else:
                    fv_invalid += 0 if d.is_valid else 1
    # An unreadable record invalidates its whole channel, including records seen before the failure.
    source_claims = [record for record in source_claims if record["topic"] not in source_failures]
    for topic in source_failures:
        if topic in calib_topics:
            calibrations.pop(topic.rsplit("/", 1)[0] + "/compressed", None)
    if ANNOTATION_TOPIC in source_failures:
        ann = None
    if "/robot0/time_range_validity" in source_failures:
        trv = None
    if "/robot0/frame_validity" in source_failures:
        fv_invalid = None
    import av
    failures, selected = {}, None
    for topic in [CAMERA_TOPIC, *(topic for topic in camera_topics if topic != CAMERA_TOPIC)]:
        ts, frames = [], []
        with mcap.open("rb") as stream:
            for _, _, msg, decoded in decoded_messages(make_reader(stream), {topic: {"data": "bytes"}}, source_failures):
                ts.append(msg.log_time)
                frames.append(bytes(decoded.data))
        if topic in source_failures:
            failures[topic] = source_failures[topic]
            continue
        first = None
        for index, packet in enumerate(frames):
            offset = 4 if packet.startswith(b"\x00\x00\x00\x01") else 3 if packet.startswith(b"\x00\x00\x01") else 0
            if len(packet) > offset and (packet[offset] & 0x1F) == 7:
                first = index
                break
        ts, frames = (ts[first:], frames[first:]) if first is not None else ([], [])
        if not frames:
            failures[topic] = "No frame from its first SPS on."
            continue
        name = topic.split("/")[-2].replace("camera", "cam")
        camera_file, camera_times = name + ".mp4", name + "_times.npy"
        t = (np.asarray(ts, dtype=np.int64) - ts[0]) / 1e9
        try:
            remux(frames, t, "h264", raw / camera_file)
        except (RemuxSourceError, av.error.FFmpegError) as error:
            failures[topic] = formats.plain_error(error)
            (raw / camera_file).unlink(missing_ok=True)
            continue
        np.save(raw / camera_times, t)
        selected = {"topic": topic, "file": camera_file, "times": camera_times,
                    "calibration": calibrations.get(topic), "is_forward_camera": topic == CAMERA_TOPIC}
        break
    if selected is None:
        raise RuntimeError(f"{rel}: no readable headset camera remains: {failures}")
    meta = {"rel": rel, "dur": (summ.statistics.message_end_time - summ.statistics.message_start_time) / 1e9,
            "ann": _annotation(ann), "fv_invalid": fv_invalid,
            "calib2": calibrations.get(CAMERA_TOPIC) if selected["is_forward_camera"] else None,
            "selected_camera": selected, "camera_calibrations": calibrations, "camera_failures": failures,
            "source_claims": source_claims,
            "source_failures": {topic: error for topic, error in source_failures.items() if topic not in camera_topics}}
    if len(ts) > 1:
        dt = np.diff(np.asarray(ts, dtype=np.int64)) / 1e6
        # drift_s: how far the last frame's recorded time is from where a 30 fps grid would put it
        meta["camera_timing"] = dict(clock="MCAP log time of each camera message", med=float(np.median(dt)),
                               max=float(dt.max()), n_gap50=int((dt > 50).sum()), n_gap100=int((dt > 100).sum()),
                               span_s=float(t[-1]), drift_s=float(t[-1] - (len(t) - 1) / 30))
    if selected["is_forward_camera"]:
        meta["cam2_dt"] = meta.get("camera_timing")
    if trv is not None:
        meta["invalid_ranges"] = [dict(t0=x.start_time_s, t1=x.end_time_s, code=x.invalid_code,
                                       msg=x.invalid_message) for x in trv.invalid_ranges]
    (raw / "meta.json").write_text(json.dumps(meta))


def write_sidecar(raw: Path, dst: Path) -> dict:
    import av
    meta = json.loads((raw / "meta.json").read_text())
    ann = meta.get("ann") or {}
    selected = meta.get("selected_camera") or {}
    camera_file = selected.get("file", "cam2.mp4")
    with av.open(str(raw / camera_file)) as c:
        st = c.streams.video[0]
        pts = presented_pts(c, st)
        w, h = st.codec_context.width, st.codec_context.height
    n = len(pts)
    times_file = raw / selected.get("times", TIMES)
    recorded = np.load(times_file) if times_file.exists() else None
    if recorded is not None and len(recorded) != n:
        raise RuntimeError(f"{raw.name}: {n} frames in {camera_file} but {len(recorded)} recorded frame times")
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
           "cameras": {"exo": {"name": "head" if selected.get("is_forward_camera", True) else "headset camera",
                "key": selected.get("topic", CAMERA_TOPIC), "width": w, "height": h,
                "desc": CAMERA_DESC if selected.get("is_forward_camera", True) else
                    f"a camera on the DAS-Ego headset ({selected['topic']}); its viewing direction is not established"}},
           "source": {"mcap": meta.get("rel"), "duration_s": meta.get("dur"), "camera_timing": meta.get("camera_timing", meta.get("cam2_dt")),
                      "dataset_invalid_ranges": meta.get("invalid_ranges"),
                      "frame_valid_false": meta.get("fv_invalid"),
                      "frame_times": FRAME_TIMES_RECORDED if recorded is not None else FRAME_TIMES_NOMINAL,
                      "selected_camera": selected.get("topic", CAMERA_TOPIC)}}
    if selected and not selected.get("is_forward_camera"):
        # The publisher annotation clock and frame-validity camera are not established by camera log times.
        ctx["annotation_subtasks"] = []
        ctx["source"].pop("dataset_invalid_ranges", None)
        ctx["source"].pop("frame_valid_false", None)
        ctx["source"]["claims_scope"] = {"source": "publisher recording", "camera": "unverified",
                                         "clock_origin": "unverified"}
        ctx["recorded_metadata"] = {"mcap_records": meta.get("source_claims") or []}
    for topic, error in meta.get("source_failures", {}).items():
        formats.add_issue(ctx, "mcap_channel_unreadable", f"Channel {topic} was left out: {error}", signal=topic)
    for topic, error in meta.get("camera_failures", {}).items():
        formats.add_issue(ctx, "camera_not_decodable", f"Camera {topic} was left out: {error}", camera=topic)
    dst.mkdir(parents=True, exist_ok=True)
    if recorded is not None:
        # the recorded times place each frame; the file's own pts (the same times, rounded to its time base) find it
        np.savez(dst / "times.npz", exo=np.asarray(recorded, dtype=np.float64), exo_pts=np.asarray(pts, dtype=np.int64))
        ctx["real_times"] = "times.npz"
        # the footage's length from the recorded times, so no reader counts it from the rate
        ctx["duration_s"] = round(capture_seconds([recorded], fps), 3)
    src = {"exo": {"packed": str((raw / camera_file).resolve()), "base_s": 0.0, "n_frames": n}}
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
