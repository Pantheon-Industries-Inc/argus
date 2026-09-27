"""Web copy of the 2D hand keypoints the board draws over head-camera footage (the episode page's "Hand pose").

The keypoints come from an ACE-Ego-Hand run (arXiv 2608.20308; board/hand_pose/ runs it): a folder with index.json
and one <dataset>/<episode>/hands2d.json per episode, 21 OpenPose joints per hand per decoded frame in source-clip
pixels, with a presence probability per hand and frame. The weights are CC BY-NC 4.0 and the model uses MANO, which is
non-commercial, so these files are for display on the board only. They are written to their own folder beside the
board's label files (BOARD/hands/, like compare/), never into qa/, so nothing the board lists, counts, downloads or
exports can include them.

One output file per board episode, named after its label file (BOARD/hands/<label file>):

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

board/build.py runs build() when the board's manifest names "hands": {"src": ..., "clips": ...}.
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

def probe_clip(clip: Path) -> dict:
    """The "clip" block for this clip: width, height, size and every frame's presentation timestamp (packets
    sorted by pts, which is display order; the i-th is the i-th decoded frame the keypoints are numbered by)."""
    r = subprocess.run([_ffprobe(), "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,time_base:packet=pts", "-of", "json", str(clip)],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe {clip}: {r.stderr.strip()[:200]}")
    j = json.loads(r.stdout)
    st = j["streams"][0]
    tb = Fraction(st["time_base"])
    pts = sorted(int(p["pts"]) for p in j.get("packets", []) if p.get("pts") not in (None, "N/A"))
    if not pts:
        raise RuntimeError(f"{clip}: no video frames")
    gaps = [b - a for a, b in zip(pts, pts[1:])]
    dur = Counter(gaps).most_common(1)[0][0] if gaps else 0
    out: list = []
    vlq_append(pts[0], out)
    for g in gaps:
        vlq_append(g - dur, out)
    return {"w": int(st["width"]), "h": int(st["height"]), "frames": len(pts), "bytes": clip.stat().st_size,
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


def build_one(job: tuple) -> dict:
    key, src_file, label_file, clip, out_dir, source = job
    try:
        if not clip.exists():
            return {"key": key, "skip": f"no clip {clip}"}
        cl = probe_clip(clip)
        d = json.loads(Path(src_file).read_text())
        v = d["video"]
        n = len(d["hands"]["left"]["kp"])
        if not (n == cl["frames"] == len(d["hands"]["right"]["kp"])):
            return {"key": key, "skip": f"keypoints cover {n} frames, the board clip has {cl['frames']}"}
        sx, sy = cl["w"] / v["width"], cl["h"] / v["height"]
        if abs(sx - sy) > 0.005 * max(sx, sy):
            return {"key": key,
                    "skip": f"board clip {cl['w']}x{cl['h']} is not a uniform scale of {v['width']}x{v['height']}"}
        doc = {"format": FORMAT, "source": source,
               "clip": cl,
               "joints": len(d["joints"]), "edges": d["edges"], "step": STEP, "tol": TOL}
        for h in ("left", "right"):
            hh = d["hands"][h]
            doc[h] = encode_hand(hh["kp"], [c is not None and c >= 0.5 for c in hh["conf"]], sx, sy)
        body = json.dumps(doc, separators=(",", ":"))
        tmp = out_dir / f".{label_file}.part"
        tmp.write_text(body)
        os.replace(tmp, out_dir / label_file)
        return {"key": key, "file": label_file, "bytes": len(body), "frames": n,
                "present": {h: sum(e - s for s, e in doc[h]["spans"]) / n for h in ("left", "right")}}
    except Exception as err:  # noqa: BLE001 - reported per episode, the build goes on
        return {"key": key, "skip": f"error: {err}"[:300]}


def build(src: Path, qa: Path, clips: Path, out_dir: Path, jobs: int = 8) -> dict:
    """Write one file per head-camera board episode the run covers into out_dir (which must exist). Episodes the
    run does not cover, and run episodes the board does not list, are simply absent; anything that does not line up
    with its board clip (frame count or scale) is skipped with the reason, never written."""
    index = json.loads((src / "index.json").read_text())
    board = head_camera_files(qa)
    # the run by its folder name only: these files go wherever the board is served, and a local path means nothing there
    source = {"model": "ACE-Ego-Hand", "paper": "arXiv:2608.20308", "run": src.name,
              "licence": "weights CC BY-NC 4.0, MANO non-commercial: for display on the board only"}
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


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m board hands", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "verify"])
    ap.add_argument("--src", type=Path, required=True,
                    help="the keypoint run (index.json and <dataset>/<episode>/hands2d.json)")
    ap.add_argument("--qa", type=Path, required=True, help="the board's label files (BOARD/qa)")
    ap.add_argument("--clips", type=Path, help="build: the clips the board serves (<episode_id>.mp4)")
    ap.add_argument("--out", type=Path, required=True, help="the folder of hand pose files")
    ap.add_argument("--jobs", type=int, default=8, help="build: episodes encoded at once")
    a = ap.parse_args()
    if a.cmd == "build":
        a.out.mkdir(parents=True, exist_ok=True)
        print(json.dumps(build(a.src, a.qa, a.clips, a.out, a.jobs), indent=1))
    else:
        print(json.dumps(verify(a.src, a.qa, a.out), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
