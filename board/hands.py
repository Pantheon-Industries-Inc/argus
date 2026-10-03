"""The 2D hand keypoints of head-camera episodes, as the board draws them over the footage (the episode page's "Hand
pose") and as a download ("Hand keypoints").

The keypoints come from an ACE-Ego-Hand run (arXiv 2608.20308; board/hand_pose/ runs it): a folder with index.json
and one <dataset>/<episode>/hands2d.json per episode, 21 OpenPose joints per hand per decoded frame in source-clip
pixels, with a presence probability per hand and frame. The weights are CC BY-NC 4.0 and the model uses MANO, which is
licensed for non-commercial research, so every file made here is for non-commercial use only and says so (LICENCE).
Two kinds of file, each in its own folder beside the board's label files (like compare/), never in qa/, so nothing
the board counts or exports as labels includes them:

  BOARD/hands/<label file>            the drawing the episode page lays over the footage (below), in the board
                                      clip's pixels, compact
  BOARD/hand_keypoints/<label file>   the download (keypoints_doc): every frame of the dataset's own video, in its
                                      own pixels and frame times, plain JSON that documents itself; index.json lists
                                      them with their frame counts and sizes

One drawing per board episode, named after its label file (BOARD/hands/<label file>):

  {"format": "board-hand-pose/1",
   "source": {...},                       the run and its licence
   "clip":   {"w", "h", "frames", "bytes", "tb": [num, den], "dur", "pts"},
                                          the clip the page plays, from ffprobe: every frame's timestamp, as VLQ
                                          integers in time-base units (tb seconds each): the first frame's, then per
                                          frame its gap to the previous frame minus dur (the usual gap), so a steady
                                          clip is a run of zeros. Timestamps are not assumed regular: 38 OpenAoE clips
                                          have a longer last frame, repeated timestamps or a first frame after 0.
                                          bytes is the clip's size, to tell a re-cut clip
   "joints": 21, "edges": [[0, 1], ...],  OpenPose hand skeleton
   "step": 4, "tol": 0.5,
   "aligned": {"keypoint_frames", "clip_frames"},   only when the keypoints were a frame or two off the clip's count
                                          (align_frames): laid from its first frame, frames past their end have no
                                          hand, and keypoints past its end are left out
   "left":  {"spans": [[s, e], ...], "data": "<VLQ>"},
   "right": {"spans": [...], "data": "..."}}

Coordinates are board-clip pixels, continuous (x = 0 is the left edge, pixel column j is centred on j + 0.5), so a
canvas laid exactly over the displayed image draws a joint at x * canvas_width / w. A hand is stored only on the
frames where its presence is at least 0.5; spans lists those frames as half-open runs [s, e).

`data` is one stream of integers in Base64 VLQ (the source-map encoding: zigzag sign, 5 bits per character, the
0x20 bit set on every character but a number's last; alphabet A-Z a-z 0-9 + /). The model predicts a pose every
`step` frames and interpolates linearly between them, so the stream stores those poses exactly and every frame in
between only when it differs. For each span in order, its knots are s, e - 1 and every frame in between that is a
multiple of `step`. For each knot: 42 integers, x0 y0 .. x20 y20 rounded to whole pixels, each the difference from
the previous knot's value (the previous knot in the stream, across spans; zero before the first). After every knot
but a span's first, one integer per frame f strictly between the previous knot a and this knot b: 0 when the frame
is the interpolation a + (b - a) * (f - ka) / (kb - ka), or 1 followed by 42 integers added to that interpolation
(the fisheye clips, whose projection bends a straight path). Every decoded value is within `tol` pixels of the
model's own.

  python -m board hands build --src KEYPOINT_RUN --qa BOARD/qa --clips CLIPS --out OUT
  python -m board hands verify --src KEYPOINT_RUN --qa BOARD/qa --out OUT   decode every file, compare with the run
  python -m board hands keypoints --src KEYPOINT_RUN --qa BOARD/qa --episodes EPISODES [EPISODES ...] --out OUT
  python -m board hands verify-keypoints --src KEYPOINT_RUN --qa BOARD/qa --episodes EPISODES ... --out OUT
                                          check every download against the run and the video, exactly

EPISODES are the folders of prepared episodes the board's runs labelled: a download is made from the episode's own
video, the one its sources.json names and the keypoint run read. board/build.py runs build() and
build_keypoints() when the board's manifest names "hands": {"src": ..., "clips": ...}.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

from board.to_board import dumps

FORMAT = "board-hand-pose/1"
STEP = 4
TOL = 0.5
B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
B64_INV = {c: i for i, c in enumerate(B64)}
HEAD_RIG = "ego_head"


def _ffprobe() -> str:
    for c in (os.environ.get("FFPROBE"), shutil.which("ffprobe")):
        if c and Path(c).exists():
            return c
    sys.exit("ffprobe not found")


# ---------------------------------------------------------------- VLQ

def vlq_append(v: int, out: list) -> None:
    v = ((-v) << 1) | 1 if v < 0 else v << 1
    while True:
        d = v & 31
        v >>= 5
        if v:
            d |= 32
        out.append(B64[d])
        if not v:
            return


def vlq_decode(s: str) -> list:
    out, v, shift = [], 0, 0
    for ch in s:
        d = B64_INV[ch]
        v |= (d & 31) << shift
        if d & 32:
            shift += 5
            continue
        out.append(-(v >> 1) if v & 1 else v >> 1)
        v, shift = 0, 0
    return out


# ---------------------------------------------------------------- encode / decode

def spans_of(present: list) -> list:
    spans, i, n = [], 0, len(present)
    while i < n:
        if present[i]:
            j = i
            while j < n and present[j]:
                j += 1
            spans.append([i, j])
            i = j
        else:
            i += 1
    return spans


def knots_of(s: int, e: int, step: int = STEP) -> list:
    first = s + (-s) % step
    return sorted({s, e - 1, *range(first, e - 1, step)})


def encode_hand(kp: list, present: list, scale_x: float, scale_y: float, step: int = STEP, tol: float = TOL) -> dict:
    """kp[i] = 42 source-pixel values (or None), present[i] = conf >= 0.5. Returns {"spans", "data"}."""
    spans = spans_of([bool(p) and kp[i] is not None for i, p in enumerate(present)])
    out: list = []
    prev = [0] * 42
    for s, e in spans:
        ks = knots_of(s, e, step)
        qk = {}
        for j, k in enumerate(ks):
            x = kp[k]
            q = [round(x[c] * (scale_x if c % 2 == 0 else scale_y)) for c in range(42)]
            for c in range(42):
                vlq_append(q[c] - prev[c], out)
            prev = q
            qk[k] = q
            if j == 0:
                continue
            ka = ks[j - 1]
            a = qk[ka]
            for f in range(ka + 1, k):
                xf = kp[f]
                p = [a[c] + (q[c] - a[c]) * (f - ka) / (k - ka) for c in range(42)]
                dev = [xf[c] * (scale_x if c % 2 == 0 else scale_y) - p[c] for c in range(42)]
                if max(abs(d) for d in dev) > tol:
                    vlq_append(1, out)
                    for d in dev:
                        vlq_append(round(d), out)
                else:
                    vlq_append(0, out)
    return {"spans": spans, "data": "".join(out)}


def decode_hand(h: dict, n_frames: int, step: int = STEP) -> list:
    """The page's decoder, in Python (for verify): a list of n_frames entries, 42 floats or None."""
    vals = vlq_decode(h["data"])
    pos = 0
    res = [None] * n_frames
    prev = [0] * 42
    for s, e in h["spans"]:
        ks = knots_of(s, e, step)
        for j, k in enumerate(ks):
            q = [prev[c] + vals[pos + c] for c in range(42)]
            pos += 42
            prev = q
            res[k] = [float(v) for v in q]
            if j == 0:
                continue
            ka = ks[j - 1]
            a = res[ka]
            for f in range(ka + 1, k):
                flag = vals[pos]
                pos += 1
                p = [a[c] + (q[c] - a[c]) * (f - ka) / (k - ka) for c in range(42)]
                if flag:
                    p = [p[c] + vals[pos + c] for c in range(42)]
                    pos += 42
                res[f] = p
    if pos != len(vals):
        raise ValueError(f"stream has {len(vals) - pos} unread values")
    return res


# ---------------------------------------------------------------- the board clip

def probe_pts(clip: Path) -> tuple:
    """(width, height, time base, every frame's presentation timestamp): packets sorted by pts, which is display
    order, so the i-th is the i-th decoded frame the keypoints are numbered by."""
    r = subprocess.run([_ffprobe(), "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,time_base:packet=pts,flags", "-of", "json", str(clip)],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe {clip}: {r.stderr.strip()[:200]}")
    j = json.loads(r.stdout)
    st = j["streams"][0]
    # a packet flagged D (discarded by an edit list) is not a frame
    pts = sorted(int(p["pts"]) for p in j.get("packets", [])
                 if p.get("pts") not in (None, "N/A") and "D" not in str(p.get("flags", "")))
    if not pts:
        raise RuntimeError(f"{clip}: no video frames")
    return int(st["width"]), int(st["height"]), Fraction(st["time_base"]), pts


def probe_clip(clip: Path) -> dict:
    """The "clip" block for this clip: width, height, size and every frame's presentation timestamp."""
    w, h, tb, pts = probe_pts(clip)
    gaps = [b - a for a, b in zip(pts, pts[1:])]
    dur = Counter(gaps).most_common(1)[0][0] if gaps else 0
    out: list = []
    vlq_append(pts[0], out)
    for g in gaps:
        vlq_append(g - dur, out)
    return {"w": w, "h": h, "frames": len(pts), "bytes": clip.stat().st_size,
            "tb": [tb.numerator, tb.denominator], "dur": dur, "pts": "".join(out)}


def frame_times(clip_block: dict) -> list:
    """Seconds at which each frame is shown, decoded from a "clip" block (as the page does)."""
    v = vlq_decode(clip_block["pts"])
    n, d = clip_block["tb"]
    p = v[0]
    out = [p * n / d]
    for g in v[1:]:
        p += clip_block["dur"] + g
        out.append(p * n / d)
    return out


def retime(doc: dict, clip: Path) -> dict:
    """The same keypoints timed against another copy of the clip (the static build's web copy, whose timestamps
    ffmpeg re-bases to start at 0). Refused unless it has the same frames at the same size."""
    cl = probe_clip(clip)
    old = doc["clip"]
    if (cl["frames"], cl["w"], cl["h"]) != (old["frames"], old["w"], old["h"]):
        raise ValueError(f"{clip} has {cl['frames']} frames at {cl['w']}x{cl['h']}, the keypoints "
                         f"{old['frames']} at {old['w']}x{old['h']}")
    return {**doc, "clip": cl}


# ---------------------------------------------------------------- one episode

def head_camera_files(qa: Path) -> dict:
    """Board label files of head-camera episodes: {"<dataset>/<episode name>": (label file, episode_id)}. The name
    is the run's own episode name when the board prefixes it (file_prefix), which is what the keypoint run keys on."""
    out = {}
    for f in sorted(qa.glob("episode_*.json")):
        d = json.loads(f.read_text())
        if d.get("_rig") != HEAD_RIG:
            continue
        meta = d.get("_meta") or {}
        eid = meta.get("episode_id") or f.stem
        for name in dict.fromkeys([meta.get("run_episode"), eid, f.stem]):
            if name:
                out.setdefault(f"{d.get('dataset')}/{name}", (f.name, eid))
    return out


FRAME_SLACK = 2      # keypoints this many frames off their video's frame count are aligned from its first frame


def align_frames(hands: dict, m: int) -> tuple[dict, dict | None]:
    """Both hands' keypoints and confidences on a video of m frames, and how they were aligned (None when they cover
    exactly its frames). Keypoints a frame or two off the video's count (FRAME_SLACK: a decoder that gave one frame more
    or fewer than the clip's) are laid from its first frame: the frames past their end have no hand, and the keypoints
    past the video's end are left out. Any other count is not the same video (ValueError)."""
    n = len(hands["left"]["kp"])
    if any(len(hands[h][k]) != n for h in ("left", "right") for k in ("kp", "conf")):
        raise ValueError("the two hands' keypoints cover different frames")
    if n == m:
        return hands, None
    if abs(n - m) > FRAME_SLACK:
        raise ValueError(f"keypoints cover {n} frames, the video has {m}")
    pad = [None] * max(0, m - n)
    return ({h: {**hands[h], "kp": (list(hands[h]["kp"]) + pad)[:m], "conf": (list(hands[h]["conf"]) + pad)[:m]}
             for h in ("left", "right")}, {"keypoint_frames": n, "clip_frames": m})


def build_one(job: tuple) -> dict:
    key, src_file, label_file, clip, out_dir, source = job
    try:
        if not clip.exists():
            return {"key": key, "skip": f"no clip {clip}"}
        cl = probe_clip(clip)
        d = json.loads(Path(src_file).read_text())
        v = d["video"]
        try:
            hands_, aligned = align_frames(d["hands"], cl["frames"])
        except ValueError as err:
            return {"key": key, "skip": f"{err}".replace("the video has", "the board clip has")}
        n = cl["frames"]
        sx, sy = cl["w"] / v["width"], cl["h"] / v["height"]
        if abs(sx - sy) > 0.005 * max(sx, sy):
            return {"key": key,
                    "skip": f"board clip {cl['w']}x{cl['h']} is not a uniform scale of {v['width']}x{v['height']}"}
        doc = {"format": FORMAT, "source": source,
               "clip": cl,
               "joints": len(d["joints"]), "edges": d["edges"], "step": STEP, "tol": TOL,
               **({"aligned": aligned} if aligned else {})}
        for h in ("left", "right"):
            hh = hands_[h]
            doc[h] = encode_hand(hh["kp"], [c is not None and c >= 0.5 for c in hh["conf"]], sx, sy)
        body = dumps(doc, separators=(",", ":"))
        tmp = out_dir / f".{label_file}.part"
        tmp.write_text(body)
        os.replace(tmp, out_dir / label_file)
        return {"key": key, "file": label_file, "bytes": len(body), "frames": n,
                "present": {h: sum(e - s for s, e in doc[h]["spans"]) / n for h in ("left", "right")}}
    except Exception as err:  # noqa: BLE001 - reported per episode, the build goes on
        return {"key": key, "skip": f"error: {err}"[:300]}


# ---------------------------------------------------------------- the download

KP_FORMAT = "pantheon-hand-keypoints/1"
# in every drawing and every download: the model's terms passed on, its credit, and what was done to its output
LICENCE = ("Non-commercial use only, because the model that predicted these keypoints is licensed for non-commercial "
           "use. They were predicted by ACE-Ego-Hand (Yufei Liu et al., ACE Robotics, arXiv:2608.20308) with its "
           "released checkpoints (https://huggingface.co/acerobotics2025/ACE-Ego-Hand), which are licensed CC BY-NC 4.0 "
           "(https://creativecommons.org/licenses/by-nc/4.0/) and provided as is, without warranties (see the "
           "license's disclaimer). Pantheon ran the model and processed its output, blending overlapping windows, "
           "smoothing over time and mapping the points back to the video's own pixels. The model uses MANO (Romero, "
           "Tzionas and Black, 2017; https://mano.is.tue.mpg.de/license.html), which is licensed for non-commercial "
           "scientific research only. Credit ACE-Ego-Hand when you use these keypoints. The video belongs to its "
           "dataset and is under that dataset's license (episode.dataset_license).")
# what the video is, per dataset: the frames the keypoints number are the dataset's own
VIDEO_ABOUT = {
    "egocentric100k": "The episode's clip as Egocentric-100K ships it (the mp4 named in source, inside its shard), "
                      "from the fisheye camera on the worker's head.",
    "genhumanego": "The forward head camera (camera 2) of the episode's Gen-HumanEgo MCAP recording named in source: "
                   "its H.264 stream copied into an mp4 without re-encoding, so the frames and pixels are the "
                   "dataset's own; the copy plays at 30 frames per second.",
    "openaoe": "The episode's raw_video.mp4 as OpenAoE-2000h ships it (the clip named in source), from the phone worn "
               "on the person's head.",
}
# the dataset's own names for the episode's video (context.json source), never a local path
SOURCE_KEYS = ("shard", "member", "mcap", "clip", "device", "resolution")


def _public_source(src: dict | None) -> dict:
    return {k: v for k, v in (src or {}).items() if k in SOURCE_KEYS and not (isinstance(v, str) and v.startswith("/"))}


def source_video(episode: Path) -> Path | None:
    """The head-camera video of a prepared episode, as its sources.json names it: the dataset's own file, the one
    the keypoint run read (board/hand_pose/specs.py)."""
    try:
        return Path(json.loads((episode / "sources.json").read_text())["exo"]["packed"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def keypoints_doc(key: str, d: dict, video: Path, label_file: str, eid: str, ctx: dict,
                  dataset_source: dict | None) -> dict:
    """The download for one episode: the run's keypoints of every frame of the dataset's video, in its own pixels,
    with each frame's timestamp probed from that video. Refused (ValueError) unless the video has exactly the
    keypoints' frames at the keypoints' size, or a frame or two more or fewer, which are aligned from its first frame
    (align_frames) and the download says so ("aligned"). dataset_source is the label file's
    (board/dataset_sources.json)."""
    w, h, tb, pts = probe_pts(video)
    v = d["video"]
    if (w, h) != (int(v["width"]), int(v["height"])):
        raise ValueError(f"{video.name} is {w}x{h}, the keypoints are in {v['width']}x{v['height']} pixels")
    hands_, aligned = align_frames(d["hands"], len(pts))
    n = len(pts)
    ds = key.split("/", 1)[0]
    joints = d["joints"]
    s = dataset_source or {}

    def hand(hh):
        kp = [None if x is None else [[x[2 * j], x[2 * j + 1]] for j in range(len(joints))] for x in hh["kp"]]
        return {"confidence": list(hh["conf"]), "keypoints": kp}
    return {
        "format": KP_FORMAT,
        "what": "2D keypoints of the camera wearer's two hands on every frame of this episode's head-camera video, "
                "predicted by the ACE-Ego-Hand model. They are model predictions, not checked by a person.",
        "licence": LICENCE,
        "episode": {"episode_id": eid, "board_file": label_file,
                    "dataset": f"{s['name']} ({s['publisher']})" if s else ds, "dataset_key": ds,
                    "dataset_id": ctx.get("dataset"), "dataset_episode": d.get("episode"),
                    "dataset_license": {"name": s["license"], "url": s["license_url"]} if s else None},
        "video": {"about": VIDEO_ABOUT.get(ds, "The episode's head-camera video as the dataset ships it."),
                  "source": _public_source(ctx.get("source")),
                  "camera": ((ctx.get("cameras") or {}).get("exo") or {}).get("desc"),
                  "width": w, "height": h, "frames": n, "time_base": [tb.numerator, tb.denominator]},
        "how_to_read": {
            "frames": "Frame i is the i-th frame decoded from the video, in display order. frames.pts[i] is its "
                      "presentation timestamp in time_base units (seconds = pts * numerator / denominator), and "
                      "frames.t[i] is seconds from the episode's first frame, (pts[i] - pts[0]) * time_base, the clock "
                      "the episode's labels on the board use.",
            "coordinates": "Pixels of the video at its full width and height, x to the right and y down, continuous: "
                           "x = 0 is the left edge of the image and pixel column j covers x = j to j + 1 (its centre "
                           "is j + 0.5); the same for y from the top edge. Fisheye footage is not undistorted: the "
                           "points sit on the image as it was recorded.",
            "keypoints": "hands.left.keypoints[i] and hands.right.keypoints[i] are 21 [x, y] pairs in the order of "
                         "joints (the OpenPose hand order; edges lists the skeleton's bones), or null where the model "
                         "gave none.",
            "confidence": "hands.left.confidence[i] and hands.right.confidence[i] are the model's probability, 0 to 1, "
                          "that the hand is in frame i. The model places both hands on every frame, so use a hand's "
                          "keypoints only where its confidence is high; the board draws a hand at 0.5 or more.",
            "hands": "left and right are the wearer's left and right hand as the model assigns them: its two output "
                     "slots, left first.",
        },
        "joints": joints, "edges": d["edges"],
        "model": {**{k: v for k, v in (d.get("model") or {}).items() if k != "code_commit"},
                  "temporal_smoothing": (d.get("temporal") or {}).get("smoothing")},
        "frames": {"pts": pts, "t": [round(float((p - pts[0]) * tb), 6) for p in pts]},
        "hands": {"left": hand(hands_["left"]), "right": hand(hands_["right"])},
        **({"aligned": {"keypoint_frames": aligned["keypoint_frames"], "video_frames": aligned["clip_frames"],
                        "how": "laid from the video's first frame: frames past the keypoints' end have none, and "
                               "keypoints past the video's end are left out"}} if aligned else {}),
    }


def keypoints_one(job: tuple) -> dict:
    key, src_file, label_path, eid, episode, out_dir = job
    try:
        video = source_video(episode)
        if video is None or not video.exists():
            return {"key": key, "skip": f"no head-camera video in {episode.name}/sources.json"}
        ctx = json.loads((episode / "context.json").read_text())
        doc = keypoints_doc(key, json.loads(Path(src_file).read_text()), video, label_path.name, eid, ctx,
                            json.loads(label_path.read_text()).get("dataset_source"))
        body = dumps(doc, separators=(",", ":"))
        tmp = out_dir / f".{label_path.name}.part"
        tmp.write_text(body)
        os.replace(tmp, out_dir / label_path.name)
        return {"key": key, "file": label_path.name, "frames": doc["video"]["frames"], "bytes": len(body)}
    except Exception as err:  # noqa: BLE001 - reported per episode, the build goes on
        return {"key": key, "skip": f"{type(err).__name__}: {err}"[:300]}


def build_keypoints(src: Path, qa: Path, episodes: dict, out_dir: Path, jobs: int = 8) -> dict:
    """Write the download of every head-camera board episode the run covers into out_dir (which must exist), and
    index.json: {"format", "licence", "files": {label file: {"frames", "bytes"}}}. episodes maps a label file to its
    prepared episode folder, whose video the run computed the keypoints on."""
    index = json.loads((src / "index.json").read_text())
    board = head_camera_files(qa)
    jobs_ = [(key, src / e["hands2d"], qa / board[key][0], board[key][1], Path(episodes[board[key][0]]), out_dir)
             for key, e in index.items() if key in board and board[key][0] in episodes]
    with cf.ProcessPoolExecutor(max(1, jobs)) as ex:
        res = list(ex.map(keypoints_one, jobs_))
    written = sorted((r for r in res if "file" in r), key=lambda r: r["file"])
    (out_dir / "index.json").write_text(dumps(
        {"format": KP_FORMAT, "licence": LICENCE,
         "files": {r["file"]: {"frames": r["frames"], "bytes": r["bytes"]} for r in written}}, separators=(",", ":")))
    return {"written": len(written), "skipped": [r for r in res if "skip" in r],
            "bytes": {"total": sum(r["bytes"] for r in written), "max": max((r["bytes"] for r in written), default=0)}}


def episode_folders(qa: Path, roots: list) -> dict:
    """{label file: prepared episode folder} for the head-camera label files, found by the run's episode name
    under any of roots (the command line's --episodes)."""
    out = {}
    for f in sorted(qa.glob("episode_*.json")):
        d = json.loads(f.read_text())
        if d.get("_rig") != HEAD_RIG:
            continue
        name = (d.get("_meta") or {}).get("run_episode") or f.stem
        hit = next((Path(r) / name for r in roots if (Path(r) / name / "context.json").exists()), None)
        if hit is not None:
            out[f.name] = hit
    return out


def build(src: Path, qa: Path, clips: Path, out_dir: Path, jobs: int = 8) -> dict:
    """Write one file per head-camera board episode the run covers into out_dir (which must exist). Episodes the
    run does not cover, and run episodes the board does not list, are simply absent; anything that does not line up
    with its board clip (a frame count more than FRAME_SLACK off, or another scale) is skipped with the reason, never
    written; a count a frame or two off is aligned from the clip's first frame (align_frames), and the file says so in
    "aligned"."""
    index = json.loads((src / "index.json").read_text())
    board = head_camera_files(qa)
    # these files are published with the board, so they name the model and its licence, and no run or storage path
    source = {"model": "ACE-Ego-Hand", "paper": "arXiv:2608.20308", "licence": LICENCE}
    jobs_ = []
    for key, e in index.items():
        if key not in board:
            continue
        label_file, eid = board[key]
        jobs_.append((key, src / e["hands2d"], label_file, clips / f"{eid}.mp4", out_dir, source))
    with cf.ProcessPoolExecutor(max(1, jobs)) as ex:
        res = list(ex.map(build_one, jobs_))
    written = [r for r in res if "file" in r]
    skipped = [r for r in res if "skip" in r]
    done = {r["file"] for r in written}
    return {"written": len(written), "skipped": skipped,
            "head_camera_files_without_keypoints": sorted({v[0] for v in board.values()} - done),
            "bytes": {"total": sum(r["bytes"] for r in written), "max": max((r["bytes"] for r in written), default=0)}}


def verify(src: Path, qa: Path, out_dir: Path) -> dict:
    """Decode every written file with the page's algorithm and compare with the source keypoints: presence frames
    must match exactly and every coordinate must be within tol of the source (rescaled to the board clip)."""
    index = json.loads((src / "index.json").read_text())
    by_file = {label_file: index[key] for key, (label_file, _) in head_camera_files(qa).items() if key in index}
    worst, n = 0.0, 0
    for f in sorted(out_dir.glob("episode_*.json")):
        doc = json.loads(f.read_text())
        d = json.loads((src / by_file[f.name]["hands2d"]).read_text())
        sx, sy = doc["clip"]["w"] / d["video"]["width"], doc["clip"]["h"] / d["video"]["height"]
        for h in ("left", "right"):
            dec = decode_hand(doc[h], doc["clip"]["frames"], doc["step"])
            hh = d["hands"][h]
            for i, (kp, c) in enumerate(zip(hh["kp"], hh["conf"])):
                if i >= len(dec):           # past the clip's end, left out (align_frames)
                    break
                want = c is not None and c >= 0.5 and kp is not None
                if want != (dec[i] is not None):
                    raise AssertionError(f"{f.name} {h} frame {i}: presence differs")
                if want:
                    err = max(abs(dec[i][k] - kp[k] * (sx if k % 2 == 0 else sy)) for k in range(42))
                    if err > doc["tol"] + 1e-9:
                        raise AssertionError(f"{f.name} {h} frame {i}: off by {err:.3f} px")
                    worst = max(worst, err)
        n += 1
    return {"files": n, "max_error_px": round(worst, 4)}


def verify_keypoints(src: Path, qa: Path, episodes: dict, out_dir: Path) -> dict:
    """Check every download against the run: it names its episode and carries the licence, has one entry per frame
    of the episode's video (the pts probed from that video again), and its keypoints and confidences are the
    run's, exactly."""
    index = json.loads((src / "index.json").read_text())
    by_file = {label_file: index[key] for key, (label_file, _) in head_camera_files(qa).items() if key in index}
    kidx = json.loads((out_dir / "index.json").read_text())["files"]
    for f, meta in sorted(kidx.items()):
        doc = json.loads((out_dir / f).read_text())
        if doc["episode"]["board_file"] != f or doc["licence"] != LICENCE:
            raise AssertionError(f"{f}: names {doc['episode']['board_file']}, or its licence is not LICENCE")
        d = json.loads((src / by_file[f]["hands2d"]).read_text())
        _, _, _, pts = probe_pts(source_video(Path(episodes[f])))
        d = {**d, "hands": align_frames(d["hands"], len(pts))[0]}
        if doc["frames"]["pts"] != pts or meta["frames"] != len(pts) or doc["video"]["frames"] != len(pts):
            raise AssertionError(f"{f}: frames differ from the video")
        for h in ("left", "right"):
            hh, got = d["hands"][h], doc["hands"][h]
            if got["confidence"] != hh["conf"] or len(got["keypoints"]) != len(pts):
                raise AssertionError(f"{f} {h}: confidence or length differs")
            for i, (kp, x) in enumerate(zip(hh["kp"], got["keypoints"])):
                if (kp is None) != (x is None) or (kp is not None and [v for xy in x for v in xy] != kp):
                    raise AssertionError(f"{f} {h} frame {i}: keypoints differ")
    return {"files": len(kidx), "exact": True}


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m board hands", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "verify", "keypoints", "verify-keypoints"])
    ap.add_argument("--src", type=Path, required=True,
                    help="the keypoint run (index.json and <dataset>/<episode>/hands2d.json)")
    ap.add_argument("--qa", type=Path, required=True, help="the board's label files (BOARD/qa)")
    ap.add_argument("--clips", type=Path, help="build: the clips the board serves (<episode_id>.mp4)")
    ap.add_argument("--episodes", type=Path, nargs="+", default=[],
                    help="keypoints, verify-keypoints: the folders of prepared episodes the board's runs labelled")
    ap.add_argument("--out", type=Path, required=True, help="the folder of hand pose files or downloads")
    ap.add_argument("--jobs", type=int, default=8, help="build, keypoints: episodes written at once")
    a = ap.parse_args()
    if a.cmd in ("build", "keypoints"):
        a.out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "build":
        res = build(a.src, a.qa, a.clips, a.out, a.jobs)
    elif a.cmd == "verify":
        res = verify(a.src, a.qa, a.out)
    elif a.cmd == "keypoints":
        res = build_keypoints(a.src, a.qa, episode_folders(a.qa, a.episodes), a.out, a.jobs)
    else:
        res = verify_keypoints(a.src, a.qa, episode_folders(a.qa, a.episodes), a.out)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
