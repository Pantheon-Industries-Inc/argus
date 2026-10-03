"""Per-episode, per-camera clips for the board (board/serve.py).

    python -m board clips --episodes EPISODES --out CLIPS [--name-prefix habit_]

The board plays each camera as its own synced <video> and expects one mp4 per episode and camera:
  fixed or head camera   CLIPS/<episode>.mp4
  left mounted camera    CLIPS/wrist_left/<episode>.mp4
  right mounted camera   CLIPS/wrist_right/<episode>.mp4
  any other camera       CLIPS/extra1/<episode>.mp4, CLIPS/extra2/..., as the reader numbers them
  a camera the model     CLIPS/unshown1/<episode>.mp4, ..., in the order of context.json unshown_cameras
  is not shown           (unshown_views); one that does not decode is flagged on the episode (record_unshown)
  a camera's depth      CLIPS/depth_<camera>/<episode>.mp4 (depth_exo, depth_left, ...), when the recording has a
                         depth stream for it (depth.json)

Some datasets keep their video packed (MolmoAct2: 12 to 50 episodes per mp4), and some cameras are HEVC or AV1,
which browsers do not all play. This cuts each episode's own frames out of its source file (sources.json: the
file, the episode's offset and its exact frame count) into a browser-native H.264 clip, once, sized for where the
page shows that camera (the recipe below) and timed on the episode's clock: a camera whose capture times the episode
keeps (times.npz) shows each frame from halfway after the capture before it to halfway to the next one (retime), so at
every time the board shows the camera's frame captured nearest it, the frame the request picks for that time, and a
labelled time shows the same instant on the board as in the request; any other camera's frames keep their source
times. A camera that started recording after the main one starts that much later (one with capture times as early as
the request shows it, retime). It is a viewing copy only: labelling decodes the source files directly and never
re-encodes. Idempotent and parallel.

An episode with any camera that decodes is always kept, labelled and put on the board from the cameras that work. A
camera whose clip comes out with fewer frames than the episode keeps its clip as cut, and the board plays it; a
camera whose video does not decode is taken out of the episode (drop_cameras; when it was the main camera, the
episode's state, signals and depth are moved onto the camera that is main now, reanchor, and the checks that read
its cameras run again, recheck). Either is recorded in the episode's
context.json, as an entry of reader_issues (record_cameras), which board/build.py copies into the episode's
dataset_checks, where it raises a data issue (board/families.py), and which note_camera_problems puts into an
upload's report notes. Only an episode none of whose cameras can be cut is listed in CLIPS/failed.json and left out
by set_aside_failed; the rest go on.

A depth clip is cut after its camera's colour clip and timed exactly like it: it has the colour clip's frames at the
colour clip's timestamps, each showing the depth frame recorded nearest that colour frame (depth_times.npz
depth_<camera> against times.npz <camera>), drawn by label/depth.py picture() (near red and far blue, metric depth
on one fixed scale, depth of unknown unit scaled across the upload, no reading black) at the colour clip's size. A
colour frame with no depth frame within a frame of it (depth_frame_map, the model's rule) is black, and so is one
whose depth frame does not decode, which is flagged with its stretch while every other depth frame is kept. The page
switches each camera between its colour and its depth clip. A depth clip that comes out imperfect (the camera's
capture times stop before its clip does, or the clip's timestamps differ from the colour clip's) is kept, and one
that cannot be cut is left out; either is recorded in the episode's reader_issues (record_depth), so the board flags
it.
A depth file that opens and does not decode, or holds no video stream, is depth_not_decodable. A missing or
unopenable file is depth_clip_failed, a failure of the cut.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from label.atomic import write_atomic

CAMS = ("exo", "left", "right")


def find_ffmpeg() -> str:
    """$FFMPEG, else the ffmpeg on PATH."""
    c = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if not c:
        raise SystemExit("ffmpeg not found: put it on PATH or set FFMPEG")
    return c


def clip_path(mp4_dir: Path, eid: str, cam: str) -> Path:
    """One camera's clip: the scene camera at the top, the mounted ones and any others in their own folders."""
    if cam in ("left", "right"):
        return mp4_dir / f"wrist_{cam}" / f"{eid}.mp4"
    if cam == "exo":
        return mp4_dir / f"{eid}.mp4"
    return mp4_dir / cam / f"{eid}.mp4"


def depth_clip(mp4_dir: Path, eid: str, cam: str) -> Path:
    """One camera's depth clip."""
    return mp4_dir / f"depth_{cam}" / f"{eid}.mp4"


def clip_paths(mp4_dir: Path, eid: str, cams=CAMS) -> dict:
    return {c: clip_path(mp4_dir, eid, c) for c in cams}


def cams_of(sources) -> list:
    """The episode's cameras in row order: the scene camera, the left and right mounted ones, then the others."""
    from label.episode import order_views
    return order_views(sources)


# The board's viewing copy of each camera: one recipe for every builder (this file and board/static.py, which
# Data Review also runs), so every copy of a clip is the same.
# A clip is never larger than its source: it shows the pixels the dataset shipped and no more. A camera larger than
# where the page shows it is scaled down to fit: the main camera (the fixed or head camera, else the first gripper
# camera) fills the screen in full screen, the others are shown 385 to 520 css px wide.
MAIN_BOX = (1920, 1080)      # the main camera: at most this, a 1080p screen in full screen
SIDE_BOX = (1280, 1080)      # a side camera: at most this
CRF = 20                     # with veryfast, the quality of CRF 22 at medium (VMAF at the shown size) in
PRESET = "veryfast"          # 40% of the time, which the whole board and every Data Review upload pay
KEY_S = 2                    # a keyframe every 2 s, so a seek decodes at most 2 s of video
DEPTH_CRF = 26               # a depth clip: its colour shading carries sensor noise that CRF 20 keeps at about
                             # five times the colour clip's size; 26 halves that and the shapes stay as clear
# names the recipe; board/static.py folds it into its media names, so a new recipe gets new names
ENC_TAG = f"h264-crf{CRF}-{PRESET}-main{MAIN_BOX[0]}x{MAIN_BOX[1]}-side{SIDE_BOX[0]}x{SIDE_BOX[1]}-kf{KEY_S}s-srcts-camclock-v4"


def main_cam(sources: dict) -> str:
    """The camera the page shows large: the fixed or head camera, else the first gripper camera."""
    return cams_of(sources)[0]


def clip_size(w: int, h: int, main: bool) -> tuple:
    """(width, height, scale) of the board clip of a w x h source: never larger than the source, both even, the
    aspect kept."""
    box = MAIN_BOX if main else SIDE_BOX
    s = min(1.0, box[0] / w, box[1] / h)
    even = lambda x, cap: min(2 * round(x / 2), cap - cap % 2)
    return even(w * s, w), even(h * s, h), s


def video_args(w: int, h: int, main: bool, threads: int, resample: bool = False, pre: tuple = ()) -> list:
    """The recipe's output arguments for a source shown w x h. resample: its pixels are not square
    (prepare/display.py), so it is scaled to its shown size and given square pixels like every other copy. pre: filters
    run before the scale (the clip's timing, extract_one)."""
    cw, ch, s = clip_size(w, h, main)
    # scaled only when it must shrink (or lose an odd row or column) or its pixels are not square, never enlarged.
    # cw x ch is the picture's own shape, so its pixels are square: without setsar, ffmpeg's scale records the rounding
    # of each side as a pixel shape (11557:11577 for 455x255 cut to 454x254)
    scale = [] if (cw, ch) == (w, h) and not resample else [f"scale={cw}:{ch}:flags=lanczos,setsar=1"]
    vf = ["-vf", ",".join([*pre, *scale])] if (pre or scale) else []
    # enc_time_base demux: every frame keeps its source timestamp exactly. The encoder's default time base is the
    # frame rate's, which rounds a variable-rate recording's times to a 1/30 s grid (up to half a frame off, and a
    # frame squeezed to 0 s where two round to the same tick)
    return [*vf, "-c:v", "libx264", "-preset", PRESET, "-crf", str(CRF), "-pix_fmt", "yuv420p",
            "-profile:v", "high", "-force_key_frames", f"expr:gte(t,n_forced*{KEY_S})", "-enc_time_base", "demux",
            "-movflags", "+faststart", "-threads", str(threads)]


def source_size(ffmpeg: str, path: str) -> tuple:
    """(width, height, resample) of a video's first stream as it is shown (prepare/display.py, the reader's rule):
    pixels that are not square made square (resample says they must be), and a rotation of 90 or 270 degrees
    swapping the stored size, since ffmpeg turns the frames upright before scaling. Scaling a portrait phone video to
    its stored landscape size squashed it."""
    from prepare import display
    probe = Path(ffmpeg).with_name("ffprobe")
    g = display.geometry(str(path), str(probe) if probe.exists() else None)
    if not g["stored"][0]:
        raise RuntimeError(f"ffprobe could not read the size of {path}")
    w, h = display.shown_size(g)
    return w, h, display.needs_resample(g)


def start_offsets(ep_dir: Path, sources: dict, fps: float = 30.0) -> dict:
    """{camera: (offset s, skip)} on the episode's clock, from times.npz, the real capture times some recordings keep
    per camera (ABC-130k, RealOmni, MCAP uploads). The request's clock starts at 0, or at
    context.json clock_zero_s once the main camera it was measured from was taken out of the episode (drop_cameras),
    so every run cuts the same clips whichever camera is main now. A camera whose recording started later plays its
    first frame offset s after it (RealOmni's right gripper camera starts up to 2 s after the left one); one that
    started earlier drops its skip frames from before the clock's start (more than half a frame before it), so it
    never plays early, and its next frame is at most half a frame from 0."""
    tp = ep_dir / "times.npz"
    cams = cams_of(sources)
    zero = _context(ep_dir).get("clock_zero_s")
    if not tp.exists() or not cams or (len(cams) < 2 and zero is None):
        return {}
    import numpy as np
    from prepare.camera_clock import load_times
    z = load_times(ep_dir, _context(ep_dir))
    t = {c: np.asarray(z[c], dtype=np.float64) for c in cams if c in z and len(z[c])}
    main = main_cam(sources)
    if main not in t:
        return {}
    ref, half = float(zero or 0.0), 0.5 / fps
    out = {}
    for c, tc in t.items():
        if c == main and zero is None:
            continue
        skip = int(np.searchsorted(tc, ref - half, side="left"))
        if skip >= len(tc):
            continue
        off = float(tc[skip]) - ref
        if skip or off >= half:
            out[c] = (off if off >= half else 0.0, skip)
    return out


def clip_times(ep_dir: Path, sources: dict, fps: float = 30.0) -> dict:
    """{camera: its frames' capture times on the episode's clock} for each camera times.npz holds one per frame of
    (sources.json n_frames): the times label/episode.py chooses the model's frames by (frame_time, and each paired
    camera's kmap, nearest in time). The clock starts where start_offsets puts it."""
    import numpy as np
    tp = ep_dir / "times.npz"
    cams = cams_of(sources)
    if not tp.exists() or not cams:
        return {}
    from prepare.camera_clock import load_times
    z = load_times(ep_dir, _context(ep_dir))
    t = {c: np.asarray(z[c], dtype=np.float64) for c in cams if c in z
         and len(z[c]) == int(sources[c].get("n_frames") or -1)}
    main = main_cam(sources)
    if main not in t:
        return {}
    zero = _context(ep_dir).get("clock_zero_s")
    ref = float(zero or 0.0)
    return {c: tc - ref for c, tc in t.items()}


def extract_one(packed: str, base_s: float, n_frames: int, out_mp4: Path,
                ffmpeg: str, threads: int, fps: float = 30.0, main: bool = True, offset_s: float = 0.0,
                skip: int = 0, times=None, query_s=None, query_map=None, source_pts=None) -> dict | None:
    """Exactly the episode's n_frames, starting at its first frame. Packed files are on an exact frame grid,
    so seeking half a frame before the episode's offset lands on its first frame whichever way the decimal
    rounds, and -frames:v stops after the last one (never a frame of the next episode). Per-episode files
    (ABC-130k, FastUMI) start at 0. Frame timestamps pass through unchanged, so real capture times stay the
    playback times. main is the camera the page shows large (main_cam); offset_s shifts every timestamp, for a
    camera that started recording after the main one, and skip drops a camera's first frames from before the main
    camera's first one (start_offsets).

    The clip is cut from the file's first video stream alone, and its first frame is put at 0 (then offset_s), so
    neither another stream that starts first (an audio track) nor the seek's half-frame lead shifts it off the
    episode's clock; the frames after it keep their own spacing, so a variable-rate recording plays as recorded. With
    times (the camera's capture times on the episode's clock, clip_times) the frames are placed by them instead
    (retime). source_pts gives the exact source packets of a part. Its cut seeks to the first packet's absolute
    timestamp, independent of the file's first timestamp, rather than using the parent's fixed-grid base_s.

    Returns None when the clip has the episode's frames. A clip with fewer (the camera's file ends before the
    episode does) is kept as cut and its counts returned, {"clip_frames", "episode_frames"}, for record_cameras; a
    video that gives no frame at all raises, as one that does not decode does."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    # a per-process temp name, so two builders on the same clip can never write one file at once
    tmp = out_mp4.with_suffix(f".{os.getpid()}.tmp.mp4")
    w, h, resample = source_size(ffmpeg, packed)
    want = int(n_frames) - int(skip)
    dense = False
    timing = ((f"select=gte(n\\,{int(skip)})",) if skip else ()) + ("setpts=PTS-STARTPTS",)
    seek = ["-ss", f"{max(0.0, base_s - 0.5 / fps):.6f}"]
    if source_pts is not None and len(source_pts):
        import av
        import math
        with av.open(str(packed)) as source:
            start = float(int(source_pts[0]) * source.streams.video[0].time_base)
            tb = source.streams.video[0].time_base
            dense = (len(source_pts) > 1 and min(int(b) - int(a) for a, b in zip(source_pts, source_pts[1:]))
                     * tb.numerator * DISPLAY_TICKS_PER_S <= tb.denominator)
            if dense:
                # Microsecond packet separation retains tied captures, but an accurate seek can discard the whole
                # short stream. Select its exact display indices from original PTS instead.
                packets = sorted(p.pts for p in source.demux(video=0) if p.size and p.pts is not None)
                first = packets.index(int(source_pts[0]))
        # Truncate the seek to microseconds so decimal rounding cannot put it after the wanted packet.
        seek = ["-seek_timestamp", "1", "-ss", f"{math.floor(start * 1_000_000) / 1_000_000:.6f}"]
        if dense:
            seek = []
            timing = (f"select=between(n\\,{first + int(skip)}\\,{first + len(source_pts) - 1})",
                      "setpts=PTS-STARTPTS")
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-threads", str(threads), *seek,
           "-i", packed, "-map", "0:v:0", "-frames:v", str(want), "-an", "-fps_mode", "passthrough",
           *video_args(w, h, main, threads, resample, pre=timing),
           *(["-movie_timescale", str(DISPLAY_TICKS_PER_S)] if dense else []),
           *(["-output_ts_offset", f"{offset_s:.6f}"] if offset_s >= 0.5 / fps else []), str(tmp)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        got = clip_frames(tmp)
        if not got:
            raise RuntimeError(f"{out_mp4.name}: no frame of {packed} decodes")
        if times is not None and len(times) >= skip + got:
            mapping = None if query_map is None else query_map - skip
            retime(tmp, times[skip:skip + got], fps, main, query_s, mapping)
        else:
            frame_lengths(tmp)
        os.replace(tmp, out_mp4)
    finally:
        tmp.unlink(missing_ok=True)       # a cut that failed part way leaves no half written clip behind
    return None if got == want else {"clip_frames": got, "episode_frames": want}


# the clip's comment when retime timed its frames: each frame is on screen from halfway after the capture before it,
# so the frame a time shows is the one on screen then (board/serve.py extract_frame), not the one starting nearest it
HALFWAY_TAG = "each frame shown from halfway after the capture before it"
# Microsecond ticks keep midpoint rounding below a video tick and agree with the composition filter's AVTB clock.
# Existing request maps resolve float ties without changing their images.
DISPLAY_TICKS_PER_S = 1_000_000


def shown_from(capture_s, fps: float = 30.0, lead_s: float = 0.0):
    """When each frame of a camera with capture times (capture_s, on the episode's clock) comes on screen, and when the
    last one goes off: from halfway after the capture before it, so at every time the board shows the frame captured
    nearest it, which is how label/episode.py picks a camera's frame for an instant (kmap, nearest in time). The
    first frame comes on lead_s before its capture, as early as the request shows it, and never before the clip's
    start at 0; the last stays on half its step after its capture (half a frame at fps for a camera with one frame).
    Returns (starts, end) in seconds."""
    import numpy as np
    t = np.asarray(capture_s, dtype=np.float64)
    starts = np.concatenate([[max(0.0, float(t[0]) - lead_s)], (t[1:] + t[:-1]) / 2])
    step = float(t[-1] - t[-2]) if len(t) > 1 else 1.0 / fps
    return starts, float(t[-1]) + step / 2


def display_ticks(capture_s, fps: float = 30.0, main: bool = True, query_s=None, query_map=None):
    """Display starts in fine ticks, constrained by the request's actual frame choices. A midpoint tie can resolve
    either way under float arithmetic, so its encoded boundary must keep that choice rather than recompute it.
    Other instants differ from the halfway boundary by at most one tick."""
    import numpy as np
    from label.episode import PAIRED_SPAN_SLACK_S
    starts, end = shown_from(capture_s, fps, 0.0 if main else PAIRED_SPAN_SLACK_S)
    ticks = np.floor(starts * DISPLAY_TICKS_PER_S).astype(np.int64) + 1
    ticks[0] = int(np.floor(starts[0] * DISPLAY_TICKS_PER_S))
    if query_s is not None and query_map is not None:
        q, km = np.asarray(query_s), np.asarray(query_map)
        valid = (km >= 0) & (km < len(ticks)) & (q >= 0)
        q, km = q[valid], km[valid]
        latest, earliest = np.full(len(ticks), -np.inf), np.full(len(ticks), np.inf)
        np.maximum.at(latest, km, q)
        np.minimum.at(earliest, km, q)
        latest = np.maximum.accumulate(latest)
        earliest = np.minimum.accumulate(earliest[::-1])[::-1]
        for i in range(1, len(ticks)):
            lo = int(np.floor(latest[i - 1] * DISPLAY_TICKS_PER_S)) + 1 if np.isfinite(latest[i - 1]) else 0
            hi = int(np.floor(earliest[i] * DISPLAY_TICKS_PER_S)) if np.isfinite(earliest[i]) else ticks[i]
            if lo > hi:
                raise ValueError("request frame choices are closer than the display clock can represent")
            ticks[i] = min(max(ticks[i], lo), hi)
    for i in range(1, len(ticks)):
        ticks[i] = max(ticks[i], ticks[i - 1] + 1)
    return ticks, max(int(np.ceil(end * DISPLAY_TICKS_PER_S)), int(ticks[-1]) + 1)


def retime(mp4: Path, capture_s, fps: float = 30.0, main: bool = True, query_s=None, query_map=None) -> None:
    """Put the frames of an encoded clip, in display order, on screen when shown_from says (capture_s, their capture
    times on the episode's clock, seconds), each lasting until the next one comes on: the times label/episode.py
    chooses the model's frames by, where the file's own timestamps can be an even rate its recorder never kept (a
    capture that dropped frames). Any camera but the main one comes on the pairing tolerance before its first capture
    (PAIRED_SPAN_SLACK_S), where label/episode.py _in_span first shows it; the request's instants are the main
    camera's own captures, so it comes on at its first. The packets are copied as encoded; decode times move with the
    display times they sit among (the same map, carried on before the first frame), so the decoder reads them in the
    same order. The clip's comment is HALFWAY_TAG."""
    import av
    import numpy as np
    from fractions import Fraction

    new, stop = display_ticks(capture_s, fps, main, query_s, query_map)
    tmp = mp4.with_name(mp4.stem + ".time.mp4")
    try:
        with av.open(str(mp4)) as src:
            ist = src.streams.video[0]
            tb = Fraction(1, DISPLAY_TICKS_PER_S)
            pkts = [p for p in src.demux(ist) if p.size and p.pts is not None]
            old = np.array(sorted(p.pts for p in pkts), dtype=np.float64)
            if len(old) != len(new):
                raise RuntimeError(f"{mp4.name}: {len(old)} frames for {len(new)} capture times")
            ticks = np.asarray(new, dtype=np.float64)

            def moved(x):
                if len(old) > 1 and x < old[0]:                 # before the first frame: the first step's slope
                    return ticks[0] - (old[0] - x) * (ticks[1] - ticks[0]) / max(old[1] - old[0], 1.0)
                return float(np.interp(x, old, ticks))
            at = {int(o): i for i, o in enumerate(old)}
            with av.open(str(tmp), "w", format="mp4", options={"movflags": "+faststart",
                                                            "movie_timescale": str(DISPLAY_TICKS_PER_S)}) as dst:
                dst.metadata["comment"] = HALFWAY_TAG
                ost = dst.add_stream_from_template(ist)
                ost.time_base = tb
                last_dts = None
                for p in pkts:
                    i = at[int(p.pts)]
                    p.duration = (new[i + 1] if i + 1 < len(new) else stop) - new[i]
                    dts = int(np.floor(moved(p.dts))) if p.dts is not None else new[i]
                    dts = min(dts if last_dts is None else max(dts, last_dts + 1), new[i])
                    p.pts, p.dts, last_dts = new[i], dts, dts
                    p.time_base = tb
                    p.stream = ost
                    dst.mux(p)
        os.replace(tmp, mp4)
    finally:
        tmp.unlink(missing_ok=True)


def depth_frame_map(colour_t, depth_t) -> list:
    """For each colour frame time, the index of the depth frame recorded nearest it, or None when none is within a
    frame of it: prepare/formats.py depth_kmap, the rule the model's depth follows, so the board and the request agree
    on where a camera has depth."""
    from prepare.formats import depth_kmap
    if not len(depth_t):
        return [None] * len(colour_t)
    return [int(k) if k >= 0 else None for k in depth_kmap(depth_t, colour_t)]


def decoded_frames(src, ist):
    """Every frame of a stream that decodes, packet by packet, so a damaged stretch costs only its own frames, as
    label/depth.py decode skips them. A file that cannot be read at all still raises (an OSError)."""
    import av
    for pkt in src.demux(ist):
        try:
            yield from pkt.decode()
        except av.error.FFmpegError as e:
            if isinstance(e, OSError):
                raise


def extract_depth(ep_dir: Path, cam: str, colour_mp4: Path, out_mp4: Path, threads: int = 2,
                  fps: float = 30.0) -> list[dict]:
    """The depth clip of one camera (module docstring): the colour clip's frames and timestamps, each the nearest
    depth frame drawn by label/depth.py picture(), encoded with the colour clip's recipe at its size (at DEPTH_CRF).

    A clip that comes out imperfect is kept, and what is wrong with it returned as reader issues for record_depth: the
    camera's frame times stop before its colour clip does (depth_clip_partial: the frames past them are black), the
    depth frames of some colour frames do not decode (depth_not_decodable, one per stretch: those frames are black),
    or the depth clip's timestamps differ from the colour clip's (depth_clip_timing). A clip that cannot be cut at all
    raises, and depth_failed records opened files with no decodable video as depth_not_decodable, and missing or
    unopenable files as depth_clip_failed."""
    from fractions import Fraction

    import av
    import numpy as np

    from board.hands import probe_pts
    from label import depth as dp
    entry = dp.load(ep_dir)[cam]
    w, h, tb, pts = probe_pts(colour_mp4)
    sources = json.loads((ep_dir / "sources.json").read_text())
    _, skip = start_offsets(ep_dir, sources, fps).get(cam, (0.0, 0))
    times = {}
    if (ep_dir / "times.npz").exists():
        with np.load(ep_dir / "times.npz") as z:
            times = {k: np.asarray(z[k]) for k in z.files}
    # the depth stream's frame times and pts: depth_times.npz (prepare/formats.py write_depth), which label/depth.py
    # load() puts on the entry; an episode prepared before that file existed has them in times.npz
    if entry.get("pts") is not None:
        times[f"depth_{cam}_pts"] = np.asarray(entry["pts"])
        if entry.get("t") is not None:
            times[f"depth_{cam}"] = np.asarray(entry["t"])
    dpts = times.get(f"depth_{cam}_pts")
    if dpts is None:
        raise RuntimeError(f"{ep_dir.name}: neither depth_times.npz nor times.npz has depth_{cam}_pts")
    issues, ctx = [], _context(ep_dir)
    if cam in times and f"depth_{cam}" in times:
        ct = times[cam][skip:skip + len(pts)]
        want = depth_frame_map(ct, times[f"depth_{cam}"]) + [None] * (len(pts) - len(ct))
        if len(ct) < len(pts):
            issues.append({"kind": DEPTH_CLIP_PARTIAL, "camera": cam, "what": (
                f"The {camera_label(cam, ctx)} has capture times for {len(ct)} of its {len(pts)} frames, so its depth "
                f"on the board is black after {len(ct)} frames.")})
    else:
        # no capture times: the depth stream's frames are the camera's own, one for one
        want = [skip + i if skip + i < len(dpts) else None for i in range(len(pts))]
    index_of = {int(p): i for i, p in enumerate(dpts)}
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_mp4.with_suffix(f".{os.getpid()}.tmp.mp4")
    black = np.zeros((h, w, 3), np.uint8)
    with depth_decoding():
        src = av.open(str(entry["packed"]))
    try:
        with av.open(str(colour_mp4)) as c:
            tag = c.metadata.get("comment")
        with src, av.open(str(tmp), "w", format="mp4",
            options={"movflags": "+faststart", **({"movie_timescale": str(tb.denominator)}
                                                   if tag == HALFWAY_TAG else {})}) as dst:
            if tag == HALFWAY_TAG:
                dst.metadata["comment"] = tag          # timed as its colour clip, so read the same way
            if not src.streams.video:
                raise DepthNotDecodable("the depth file holds no video stream")
            ist = src.streams.video[0]
            # a camera whose depth unit is not known is drawn on its upload-wide range (prepare writes
            # entry["range"]); an episode prepared before that was measured falls back to its own readings, never to
            # each frame's
            rng = None
            if not entry.get("scale_m") and not entry.get("range"):
                idx = sorted(set(np.linspace(0, len(dpts) - 1, min(24, len(dpts))).astype(int).tolist()))
                with depth_decoding():
                    got = dp.decode(entry, dpts, idx)
                rng = dp.scale_range(got.values())
            ist.codec_context.thread_count = threads
            ost = dst.add_stream("libx264", rate=Fraction(round(fps * 1000), 1000))
            ost.width, ost.height, ost.pix_fmt = w, h, "yuv420p"
            ost.time_base = tb
            ost.codec_context.time_base = tb
            ost.codec_context.thread_count = threads
            ost.options = {"crf": str(DEPTH_CRF), "preset": PRESET, "profile": "high", "forced-idr": "1"}
            frames = decoded_frames(src, ist)
            cur_i, cur, lost = -1, None, []
            next_key = 0.0
            for i, p in enumerate(pts):
                j = want[i]
                while j is not None and cur_i < j:
                    fr = next(frames, None)
                    if fr is None:
                        j = None
                        break
                    k = index_of.get(int(fr.pts)) if fr.pts is not None else None
                    if k is not None:
                        cur_i, cur = k, fr
                if j is not None and cur_i == j:
                    im = dp.picture(dp._array(cur), entry, rng)
                    if im.size != (w, h):
                        im = im.resize((w, h), resample=0)
                    rgb = np.asarray(im.convert("RGB"))
                else:
                    rgb = black
                    if want[i] is not None:
                        lost.append(i)
                vf = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                vf.pts, vf.time_base = int(p), tb
                if float(p * tb) >= next_key - 1e-9:
                    vf.pict_type = av.video.frame.PictureType.I
                    next_key += KEY_S
                for pkt in ost.encode(vf):
                    dst.mux(pkt)
            for pkt in ost.encode():
                dst.mux(pkt)
        frame_lengths(tmp)
        # a stretch is named by the capture times of its colour frames where the clip shows them by those (retime),
        # else by the clip's own times
        cap = clip_times(ep_dir, sources, fps).get(cam) if tag == HALFWAY_TAG else None
        at = cap[skip:skip + len(pts)] if cap is not None and len(cap) >= skip + len(pts) else \
            np.asarray(pts, dtype=np.float64) * float(tb)
        for run in np.split(np.asarray(lost, dtype=int), np.flatnonzero(np.diff(lost) > 1) + 1) if lost else []:
            t0, t1 = float(at[run[0]]), float(at[run[-1]])
            issues.append({"kind": DEPTH_NOT_DECODABLE, "camera": cam, "t0_s": round(t0, 3), "t1_s": round(t1, 3),
                           "what": f"The {camera_label(cam, ctx)}'s depth could not be decoded from {t0:.2f} s to "
                                   f"{t1:.2f} s, so the board shows no depth there."})
        got = probe_pts(tmp)[3]
        if got != list(pts):
            issues.append({"kind": DEPTH_CLIP_TIMING, "camera": cam, "what": (
                f"The {camera_label(cam, ctx)}'s depth clip came out with {len(got)} frames at timestamps that differ "
                f"from its video's {len(pts)}, so its depth on the board can be a frame off.")})
        os.replace(tmp, out_mp4)
    finally:
        tmp.unlink(missing_ok=True)
    return issues


@contextlib.contextmanager
def depth_decoding():
    """Reading the recording's depth file: an FFmpeg error that is not an OSError means the file opened and its data
    does not decode, a fault in the recording (DepthNotDecodable). A file that is gone or cannot be opened raises an
    OSError, which stays as it is, a failure of our cut, as label/depth.py decode treats it."""
    import av
    try:
        yield
    except av.error.FFmpegError as e:
        if isinstance(e, OSError):
            raise
        raise DepthNotDecodable(f"{type(e).__name__}: {e}") from e


DEPTH_CLIP_PARTIAL = "depth_clip_partial"     # a depth clip with black frames where the camera has no capture time
DEPTH_CLIP_TIMING = "depth_clip_timing"       # a depth clip whose timestamps differ from its colour clip's
DEPTH_CLIP_FAILED = "depth_clip_failed"       # a depth clip that could not be cut; the page offers no depth there
DEPTH_NOT_DECODABLE = "depth_not_decodable"   # the recording's depth file opens and does not decode
DEPTH_KINDS = (DEPTH_CLIP_PARTIAL, DEPTH_CLIP_TIMING, DEPTH_CLIP_FAILED, DEPTH_NOT_DECODABLE)


class DepthNotDecodable(Exception):
    """The recording's depth file opened and does not decode, or holds no video stream: a fault in the recording, not
    in our cut (depth_decoding)."""


def depth_failed(ep_dir: Path, cam: str, err: Exception) -> dict:
    """The reader issue of a camera whose depth clip could not be cut (extract_depth raised): depth_not_decodable when
    the recording's depth file opens and does not decode (DepthNotDecodable), else depth_clip_failed."""
    label = camera_label(cam, _context(ep_dir))
    if isinstance(err, DepthNotDecodable):
        return {"kind": DEPTH_NOT_DECODABLE, "camera": cam, "what": (
            f"The {label}'s depth video does not decode ({str(err)[:160]}), so the page shows no depth for it.")}
    return {"kind": DEPTH_CLIP_FAILED, "camera": cam, "what": (
        f"The {label}'s depth could not be drawn for the board ({str(err)[:160]}), so the page shows no depth for it.")}


def record_depth(ep_dir: Path, cam: str, issues: list[dict]) -> None:
    """A camera's depth clip issues from this run (extract_depth, depth_failed) in the episode's context.json
    reader_issues, in place of the ones an earlier run recorded for it; board/build.py copies them into the episode's
    dataset_checks, where each raises a data issue. Input markers are on the consumer clock and saved on the raw
    capture clock; their derived prose stays on the consumer clock. A clean cut takes the camera's entries away."""
    ctx = _context(ep_dir)
    from prepare.formats import shift_context_times
    issues = shift_context_times({"reader_issues": issues}, float(ctx.get("clock_zero_s") or 0.0))["reader_issues"]
    old = ctx.get("reader_issues") or []
    keep = [x for x in old if not (isinstance(x, dict) and x.get("kind") in DEPTH_KINDS and x.get("camera") == cam)]
    new = keep + list(issues)
    if new == old:
        return
    if new:
        ctx["reader_issues"] = new
    else:
        ctx.pop("reader_issues", None)
    write_atomic(ep_dir / "context.json", ctx, indent=1)


def frame_lengths(mp4: Path) -> bool:
    """Give every video frame of an encoded mp4 its length, so the file plays to the end of its last frame; True when
    it had to. Some ffmpeg builds (7.0.2 and 7.1.5 seen, on an MPEG-4 Part 2 source among others) hand the muxer
    encoded packets with no duration, so the last packet in decode order lasts 0 s, the file's edit list ends where
    the latest frame starts, and a player (and clip_frames) drops that frame: a 480-frame episode came out as 479.
    Each frame then lasts the step to the next frame's time and the last one the step before it, as FrameWriter
    (prepare/formats.py) gives copied frames; the packets, their times and every other stream are copied unchanged,
    so a camera that starts late still starts where it did. A file whose frames all have a length is left as it is."""
    import av
    with av.open(str(mp4)) as src:
        ist = src.streams.video[0]
        tag = src.metadata.get("comment")
        meta = [(p.pts, p.duration) for p in src.demux(ist) if p.size and p.pts is not None]
    if not meta or all(d for _, d in meta[:-1]) and meta[-1][1]:
        return False
    pts = sorted(p for p, _ in meta)
    step = {p: (pts[i + 1] - p if i + 1 < len(pts) else (p - pts[i - 1] if i else 0)) for i, p in enumerate(pts)}
    tmp = mp4.with_name(mp4.stem + ".len.mp4")
    try:
        with av.open(str(mp4)) as src, av.open(str(tmp), "w", format="mp4", options={"movflags": "+faststart"}) as dst:
            if tag == HALFWAY_TAG:
                dst.metadata["comment"] = tag          # how the clip is timed (retime) goes with it
            outs = {}
            for s in src.streams:
                if s.type in ("video", "audio"):
                    o = dst.add_stream_from_template(s)
                    o.time_base = s.time_base
                    outs[s.index] = o
            vi = src.streams.video[0].index
            for pkt in src.demux(*[src.streams[i] for i in outs]):
                if not pkt.size:
                    continue
                if pkt.stream.index == vi and pkt.pts is not None and step.get(pkt.pts):
                    pkt.duration = step[pkt.pts]
                pkt.stream = outs[pkt.stream.index]
                dst.mux(pkt)
        os.replace(tmp, mp4)
    finally:
        tmp.unlink(missing_ok=True)
    return True


def clip_frames(mp4: Path) -> int:
    """Frames in a finished clip (0 when it cannot be read)."""
    try:
        import av
        with av.open(str(mp4)) as c:
            return sum(1 for p in c.demux(c.streams.video[0]) if p.size and not p.is_discard)
    except Exception:
        return 0


def board_name(eid: str, prefix: str = "") -> str:
    """The episode's name on the board: a dataset whose episode names repeat another's (HABIT's
    episode_000494 and MolmoAct2's) takes the file_prefix of its board manifest entry, episode_habit_000494."""
    return eid.replace("episode_", f"episode_{prefix}", 1) if prefix else eid


UNSHOWN = "unshown"        # the view of a camera the model is not shown: unshown1, unshown2, ... (unshown_views)
UNSHOWN_NOT_DECODABLE = "unshown_camera_not_decodable"


def unshown_views(ctx: dict) -> list[tuple[str, dict]]:
    """[(view, entry)] of the cameras the model is not shown (context.json unshown_cameras, prepare/formats.py: more
    extra cameras than it is shown, a stereo camera's second eye, every camera but one on a head rig, an infrared,
    thermal or mask video) that have a video: unshown1, unshown2, ... in their order. The board cuts each like any
    other camera into CLIPS/unshown<N>/ (episode_jobs) and the page plays it, named as not shown to the model; the
    model's request never reads them."""
    out = []
    for e in ctx.get("unshown_cameras") or []:
        if isinstance(e, dict) and e.get("packed") and e.get("n_frames"):
            out.append((f"{UNSHOWN}{len(out) + 1}", e))
    return out


def record_unshown(ep_dir: Path, cut: set, broken: dict) -> None:
    """The cameras the model is not shown that this run cut (cut) or could not decode (broken: {view: why}) in the
    episode's context.json reader_issues: one entry of kind unshown_camera_not_decodable per camera that does not
    decode, in place of an earlier run's for the same camera, and none for one that cut. The episode keeps every
    other camera and its labels."""
    ctx = _context(ep_dir)
    names = {v: str(e.get("name") or v) for v, e in unshown_views(ctx)}
    old = ctx.get("reader_issues") or []
    redo = set(cut) | set(broken)
    new = [x for x in old if not (isinstance(x, dict) and x.get("kind") == UNSHOWN_NOT_DECODABLE
                                  and x.get("camera") in redo)]
    new += [{"kind": UNSHOWN_NOT_DECODABLE, "camera": v,
             "what": f"The {names.get(v, v)} camera, which the model is not shown, could not be decoded, so the board "
                     f"cannot play it ({str(why)[:160]})."} for v, why in sorted(broken.items())]
    if new == old:
        return
    if new:
        ctx["reader_issues"] = new
    else:
        ctx.pop("reader_issues", None)
    write_atomic(ep_dir / "context.json", ctx, indent=1)


def episode_jobs(ep_dir: Path, mp4_dir: Path, force: bool, prefix: str = ""):
    src_p = ep_dir / "sources.json"
    if not src_p.exists():
        return []
    eid = board_name(ep_dir.name, prefix)
    sources = json.loads(src_p.read_text())
    ctx_p = ep_dir / "context.json"
    ctx = json.loads(ctx_p.read_text()) if ctx_p.exists() else {}
    # half a frame before the episode's first frame, at the episode's own rate (packed LeRobot v3 files
    # at 50 fps put the previous episode's last frame closer than half a 30 fps frame)
    fps = float(ctx.get("fps") or 30.0)
    cams = cams_of(sources)           # FastUMI has no fixed camera; single-gripper tasks have one camera
    outs = clip_paths(mp4_dir, eid, cams)
    big = main_cam(sources) if cams else None
    offsets = start_offsets(ep_dir, sources, fps)
    times = clip_times(ep_dir, sources, fps)
    import numpy as np
    packets = {}
    if ctx.get("piece") and ctx.get("real_times"):
        with np.load(ep_dir / ctx["real_times"]) as z:
            packets = {cam: z[f"{cam}_pts"] for cam in cams if f"{cam}_pts" in z.files}
    query = times.get(big)
    jobs = []
    for cam in cams:
        o = outs[cam]
        if force or not (o.exists() and o.stat().st_size > 0 and clip_frames(o) > 0):
            s = sources[cam]
            off, skip = offsets.get(cam, (0.0, 0))
            km = (np.load(ep_dir / s["kmap"]) if s.get("kmap") else
                  np.arange(len(query)) if query is not None else None)
            jobs.append((s["packed"], float(s["base_s"]), int(s["n_frames"]), o, fps, cam == big, off, skip,
                         times.get(cam), query, km, packets.get(cam), ep_dir.name, cam))
    # the cameras the model is not shown, as side cameras at their own start on the episode's clock: one that started
    # before the clock drops its frames from before it, as start_offsets does
    for view, e in unshown_views(ctx):
        o = clip_path(mp4_dir, eid, view)
        if force or not (o.exists() and o.stat().st_size > 0 and clip_frames(o) > 0):
            f = float(e.get("fps") or fps)
            from prepare.camera_clock import unshown_times
            own_clock = unshown_times(ep_dir, e, float(ctx.get("clock_zero_s") or 0.0))
            if own_clock is not None:
                ts, pts = own_clock
                skip = int(np.searchsorted(ts, -0.5 / f, side="left"))
                if skip >= len(ts):
                    continue
                off = max(0.0, float(ts[skip]))
                from prepare.formats import nearest
                km = nearest(ts, query) if query is not None else None
                jobs.append((str(e["packed"]), float(e.get("base_s") or 0.0), len(ts), o, f, False,
                             off, skip, ts, query, km, pts, ep_dir.name, view))
                continue
            start = float(e.get("start_s") or 0.0) - float(ctx.get("clock_zero_s") or 0.0)
            skip = int(round(-start * f)) if start < 0 else 0
            jobs.append((str(e["packed"]), float(e.get("base_s") or 0.0), int(e["n_frames"]), o, f, False,
                         max(0.0, start + skip / f), skip, None, None, None, None, ep_dir.name, view))
    return jobs


def depth_jobs(ep_dir: Path, mp4_dir: Path, force: bool, prefix: str = "") -> list:
    """(episode folder, camera, colour clip, depth clip, fps) for each camera of the episode with a depth stream whose
    colour clip exists and whose depth clip is missing or does not match the colour clip's frame count."""
    dj = ep_dir / "depth.json"
    src_p = ep_dir / "sources.json"
    if not dj.exists() or not src_p.exists():
        return []
    eid = board_name(ep_dir.name, prefix)
    sources = json.loads(src_p.read_text())
    ctx_p = ep_dir / "context.json"
    fps = float((json.loads(ctx_p.read_text()) if ctx_p.exists() else {}).get("fps") or 30.0)
    jobs = []
    for cam in json.loads(dj.read_text()):
        if cam not in sources:
            continue
        colour, out = clip_path(mp4_dir, eid, cam), depth_clip(mp4_dir, eid, cam)
        n = clip_frames(colour) if colour.exists() else 0
        if not n:
            continue
        if force or not (out.exists() and clip_frames(out) == n):
            jobs.append((ep_dir, cam, colour, out, fps))
    return jobs


FAILED = "failed.json"      # in the clips folder: {episode folder: {camera: why its clip could not be cut}}, for the
                            # episodes none of whose cameras could be cut


def set_aside_failed(eps: Path, clips_dir: Path) -> list[dict]:
    """Move the episodes none of whose cameras could be cut (no camera file decodes) out of the episode folder, into
    <eps>_unclipped next to it, so the rest of the upload is labelled and put on the board without them. An episode
    with a camera that decodes is never here: board clips kept it with its other cameras (record_cameras). Returns
    them as the reader reports an episode it could not open, {"name", "why"}, with a plain reason. An episode moved on
    an earlier run is not reported again."""
    fp = clips_dir / FAILED
    failed = json.loads(fp.read_text()) if fp.exists() else {}
    out = []
    for ep, cams in sorted(failed.items()):
        src = eps / ep
        if not src.is_dir():
            continue
        dest = eps.parent / f"{eps.name}_unclipped"
        dest.mkdir(exist_ok=True)
        shutil.rmtree(dest / ep, ignore_errors=True)
        src.rename(dest / ep)
        out.append({"name": ep, "why": failed_reason(cams, dest / ep)})
    return out


def camera_label(view: str, ctx: dict) -> str:
    """A camera as the board names it: the main (or head) camera, the left or right wrist (or gripper), and any other
    camera by the dataset's own name for it."""
    rig = ctx.get("profile")
    if view == "exo":
        return "head camera" if rig == "ego_head" else "main camera"
    if view in ("left", "right"):
        return f"{view} gripper camera" if rig == "handheld_gripper" else f"{view} wrist camera"
    cam = (ctx.get("cameras") or {}).get(view) or {}
    return str(cam.get("name") or cam.get("key") or "other") + " camera"


def _context(ep_dir: Path) -> dict:
    try:
        return json.loads((ep_dir / "context.json").read_text())
    except (OSError, ValueError):
        return {}


def failed_reason(cams: dict, ep_dir: Path) -> str:
    """Why an episode was left out, in one plain sentence: {camera view: the error extract_one raised} for every
    camera it has, since an episode with one camera that decodes is kept. One camera is named as the board names
    it; several are said once."""
    if len(cams) > 1:
        return "every camera's video could not be decoded, so this episode was left out"
    view = next(iter(cams))
    return f"the {camera_label(view, _context(ep_dir))} video could not be decoded, so this episode was left out"


def drop_cameras(ep_dir: Path, views, keep_clock: bool = False) -> tuple[str | None, list[dict]]:
    """Take cameras out of a prepared episode, out of sources.json, context.json's cameras and depth.json, so labelling
    and the board never decode them. Returns (the new main camera when the main camera went, else None; the reader
    issues the drop leaves).

    The main camera is the one the episode's arrays are recorded on: the state and action rows (state.npz), the
    signals (signals.npz) and the depth frame for each frame (its depth kmap) are one per main camera frame. When it
    goes, the camera first in row order is the main one, and what happens to those arrays depends on how that camera
    was paired to the old one:
      - not paired by time (no kmap): it shares the old camera's frame index (LeRobot files, or capture times equal
        to the old camera's), so every array is on its frames already and is kept as it is;
      - paired by time, with both cameras' capture times in times.npz: everything is moved onto its frames (reanchor);
      - paired by time, with no capture times left: nothing places the arrays on its frames, so the state and signals
        are marked unaligned (context.json state_unaligned, which label/episode.py never treats as aligned and the
        checks that compare the state with the video do not assess) and a reader issue says why.
    The reader's issues of cameras that start late or end early are measured against the main camera, so they are
    measured again on the cameras left (respan). keep_clock: the episode is labelled already, so its clock never
    moves (reanchor)."""
    views = set(views)
    src = json.loads((ep_dir / "sources.json").read_text())
    ctx = _context(ep_dir)
    old_main = cams_of(src)[0] if src else None
    old_name = camera_label(old_main, ctx) if old_main else ""
    rest = {v: s for v, s in src.items() if v not in views}
    paired = bool(rest) and bool(rest[cams_of(rest)[0]].get("kmap"))      # the new main camera, paired by time
    for v in views:
        src.pop(v, None)
        (ctx.get("cameras") or {}).pop(v, None)
        (ctx.get("placeholder_frames") or {}).pop(v, None)
        (ctx.get("camera_clock") or {}).pop(v, None)
    dj = ep_dir / "depth.json"
    if dj.exists():
        depth = json.loads(dj.read_text())
        if views & set(depth):
            dj.write_text(json.dumps({v: e for v, e in depth.items() if v not in views}, indent=1))
    main, issues = None, []
    if src and old_main in views:
        main = cams_of(src)[0]
        import numpy as np
        tp = ep_dir / (ctx.get("real_times") or "times.npz")
        t = {}
        if tp.exists():
            from prepare.camera_clock import load_times
            t = load_times(ep_dir, ctx)
        if not paired:
            pass
        elif old_main in t and main in t and len(t[old_main]) and len(t[main]):
            issues += reanchor(ep_dir, ctx, src, t, old_main, main, old_name, keep_clock)
        else:
            for s in src.values():
                s.pop("kmap", None)
            if ctx.get("state_kind") not in (None, "none") or ctx.get("signals"):
                ctx["state_unaligned"] = (f"It was recorded on the frames of the {old_name}, which could not be "
                                          "decoded, and the episode has no capture times to place it on the other "
                                          "cameras' frames.")
                issues.append({"kind": "state_unaligned", "what": f"The recorded state and signals are on the frames "
                               f"of the {old_name}, which could not be decoded, and the episode has no capture times "
                               f"to place them on the {camera_label(main, ctx)}'s frames, so they are not used."})
    if main in (ctx.get("camera_clock") or {}):
        from prepare.formats import Signals, qualify_camera_signals, write_signals
        sig = Signals()
        if ctx.get("signals"):
            with np.load(ep_dir / "signals.npz") as z:
                for m in ctx["signals"]:
                    sig[m["name"]] = z[m["key"]]
                    sig.meta[m["name"]] = {k: v for k, v in m.items() if k not in ("key", "name")}
        state, action = None, None
        if ctx.get("state_kind") != "none" and (ep_dir / "state.npz").exists():
            with np.load(ep_dir / "state.npz") as z:
                state, action = z["state"], z["action"] if "action" in z.files else None
        sig = qualify_camera_signals(ctx, sig, state, action)
        write_signals(ep_dir, ctx, sig, load_times(ep_dir, ctx)[main])
    if src:
        respan(ep_dir, ctx, src)
    (ep_dir / "sources.json").write_text(json.dumps(src, indent=1))
    write_atomic(ep_dir / "context.json", ctx, indent=1)
    return main, issues


SPAN_KINDS = ("camera_short", "main_camera_short")    # the reader's camera span issues (camera_span_issues)


def respan(ep_dir: Path, ctx: dict, src: dict) -> None:
    """The reader's issues of cameras that start late or end early (prepare/formats.py camera_span_issues, kinds
    SPAN_KINDS), measured again in ctx on the cameras of src, the main camera first: each camera's frame times on the
    episode's clock (times.npz), or its frame index over the rate where the episode has no capture times, as the
    reader measures them. A short clip that board clips found itself (record_cameras) is kept."""
    import numpy as np

    from prepare.formats import camera_span_issues
    cams = cams_of(src)
    fps = float(ctx.get("fps") or 30.0)
    tp = ep_dir / (ctx.get("real_times") or "times.npz")
    t = {}
    if tp.exists():
        from prepare.camera_clock import load_times
        z = load_times(ep_dir, ctx)
        t = {v: np.asarray(z[v], dtype=np.float64) for v in cams if v in z}
    if set(t) != set(cams):
        t = {v: np.arange(int(src[v]["n_frames"])) / fps for v in cams}
    extra = {"reader_issues": [x for x in ctx.get("reader_issues") or []
                               if not (isinstance(x, dict) and x.get("kind") in SPAN_KINDS and "clip_frames" not in x)]}
    camera_span_issues(extra, t, cams[0], {v: src[v].get("camera_key") or v for v in cams}, ctx)
    if extra["reader_issues"]:
        ctx["reader_issues"] = extra["reader_issues"]
    else:
        ctx.pop("reader_issues", None)


def reanchor(ep_dir: Path, ctx: dict, src: dict, t: dict, old: str, new: str, old_name: str,
             keep_clock: bool = False) -> list[dict]:
    """Move an episode from the frames of its main camera old onto those of new, by capture time (t, times.npz), and
    return the reader issues it leaves.

    Each array with one row per old frame (state.npz, signals.npz) gets one row per new frame, the old frame nearest
    it in time, and no value (NaN) where no old frame is within half a frame. The rows that have a value are the
    stretch the state covers, context.json state_span [first, last + 1), which every use of the state keeps to
    (label/episode.py state_span), and a reader issue says so when it is not the whole episode. Every other camera and
    each depth stream is paired to new by nearest time, as the reader pairs them (prepare/formats.py nearest).

    The episode's clock then starts at the earliest first frame of the cameras left (a camera that started before the
    old main one would otherwise sit at negative times): times.npz and depth_times.npz, clock_start_s and every timed
    field of the context (prepare/formats.py CLOCK_TIME_KEYS) move with it, clock_zero_s holds the times.npz value of
    the clock's start, from which start_offsets times every clip, and the frame count, rate and length are new's. An
    episode labelled already (keep_clock) keeps its clock, since its labels are on it: a camera that started before it
    is cut from the clock's start, and a reader issue says how much of it is not shown."""
    import numpy as np

    from prepare.formats import CLOCK_TIME_KEYS, depth_kmap, nearest, placeholder_frames
    t_old, t_new = t[old], t[new]
    step_old = float(np.median(np.diff(t_old))) if len(t_old) > 1 else 1.0 / float(ctx.get("fps") or 30.0)
    idx = nearest(t_old, t_new)
    near = np.abs(t_old[idx] - t_new) <= step_old / 2 + 1e-6

    def move(a):
        a = np.asarray(a)
        if a.ndim == 0 or len(a) != len(t_old):
            return a
        out = a[idx].astype(a.dtype if np.issubdtype(a.dtype, np.floating) else np.float64)
        out[~near] = np.nan
        return out
    for name in ("state.npz", "signals.npz"):
        p = ep_dir / name
        if p.exists():
            with np.load(p) as z:
                arrs = {k: move(z[k]) for k in z.files}
            np.savez(p, **arrs)
    # each camera's placeholder frames (prepare/formats.py placeholder_frames) are on the old main camera's frames: back
    # to the camera's own frames through its old pairing, then onto the new main camera's through its new one
    own = {}
    for v, runs in (ctx.get("placeholder_frames") or {}).items():
        ks = [k for x, y in runs for k in range(int(x), int(y) + 1)]
        km = np.load(ep_dir / src[v]["kmap"]) if v in src and src[v].get("kmap") else None
        own[v] = sorted({int(km[k]) for k in ks if k < len(km)}) if km is not None else ks
    kmaps = {}
    for v, s in src.items():
        s.pop("kmap", None)
        if v == new or v not in t:
            continue
        km = nearest(t[v], t_new)
        if not (len(t[v]) == len(t_new) and np.array_equal(km, np.arange(len(km)))):
            np.save(ep_dir / f"kmap_{v}.npy", km)
            s["kmap"] = f"kmap_{v}.npy"
            kmaps[v] = km
    held = placeholder_frames({v: js for v, js in own.items() if v in src}, kmaps)
    ctx.pop("placeholder_frames", None)
    if held:
        ctx["placeholder_frames"] = held
    dj = ep_dir / "depth.json"
    dt = {}
    if (ep_dir / "depth_times.npz").exists():
        with np.load(ep_dir / "depth_times.npz") as z:
            dt = {k: np.asarray(z[k]) for k in z.files}
    if dj.exists():
        recorded_new = t_new
        if ctx.get("presentation_times"):
            from prepare.camera_clock import load_times
            recorded_new = load_times(ep_dir, ctx, recorded=True)[new]
        for v, e in json.loads(dj.read_text()).items():
            td = dt.get(f"depth_{v}", t.get(f"depth_{v}"))
            kp = ep_dir / e["kmap"]
            if td is not None and len(td):
                np.save(kp, depth_kmap(td, recorded_new))        # no reading where no depth frame is within a frame
            elif kp.exists():
                np.save(kp, np.load(kp)[idx])            # no depth times: the depth frame of the nearest old frame
    # the clock: its start is clock_zero_s, or 0 as in the request, moved
    # to the earliest first frame of the cameras left unless the episode is labelled already
    zero = float(ctx.get("clock_zero_s") or 0.0)
    first = {v: float(t[v][0]) for v in src if v in t and len(t[v])}
    issues = []
    shift = 0.0 if keep_clock else min(first.values()) - zero
    if shift:
        for entry in ctx.get("unshown_cameras") or []:
            if entry.get("camera_times"):
                entry["camera_clock_offset_s"] = float(entry.get("camera_clock_offset_s") or 0.0) + shift
                entry["start_s"] = float(entry.get("start_s") or 0.0) - shift
        moved = {k: (a if k.endswith("_pts") else a - shift) for k, a in t.items()}
        if ctx.get("presentation_times"):
            from prepare.camera_clock import load_times
            original = load_times(ep_dir, ctx, recorded=True)
            moved = {k: a if k.endswith("_pts") else np.asarray(a, dtype=np.float64) - shift
                     for k, a in original.items()}
            with np.load(ep_dir / ctx["presentation_times"]) as z:
                np.savez(ep_dir / ctx["presentation_times"], **{k: z[k] - shift for k in z.files})
        np.savez(ep_dir / (ctx.get("real_times") or "times.npz"), **moved)
        if dt:
            np.savez(ep_dir / "depth_times.npz", **{k: (a if k.endswith("_pts") else np.asarray(a, dtype=np.float64)
                                                         - shift) for k, a in dt.items()})
        if ctx.get("clock_start_s") is not None:
            ctx["clock_start_s"] = round(float(ctx["clock_start_s"]) + shift, 6)
        for key, fields in CLOCK_TIME_KEYS.items():
            for x in ctx.get(key) or []:
                for f in fields:
                    if isinstance(x, dict) and isinstance(x.get(f), (int, float)):
                        x[f] = round(float(x[f]) - shift, 3)
                if key == "contacts" and isinstance(x, dict) and x.get("dips_s"):
                    x["dips_s"] = [round(float(t) - shift, 3) for t in x["dips_s"]]
        t_new = t_new - shift
    half = step_old / 2
    for v, f0 in first.items():
        if keep_clock and f0 < zero - half:
            issues.append({"kind": "camera_offset", "camera": v, "what": f"The {camera_label(v, ctx)} video starts "
                           f"{zero - f0:.2f} s before the episode's clock, which its labels were made on, so that part "
                           "of it is not shown."})
    ctx["clock_zero_s"] = zero
    step = float(np.median(np.diff(t_new))) if len(t_new) > 1 else step_old
    duration = float(t_new[-1]) - zero + step
    if keep_clock:
        # the labels span the recording as it was: its length never shrinks to the new main camera's span, and reaches
        # every camera left's last frame
        duration = max([duration, float(ctx.get("duration_s") or 0.0)]
                       + [float(t[v][-1]) - zero + step for v in first])
    ctx.update(n_state_frames=int(len(t_new)), fps=round(1.0 / step, 3), duration_s=round(duration, 3))
    if ctx.get("state_kind") in (None, "none") and not ctx.get("signals"):
        return issues                            # nothing recorded on the frames to place
    if not near.any():
        ctx["state_unaligned"] = (f"It was recorded on the frames of the {old_name}, which could not be decoded, "
                                  "and no frame of the cameras left was filmed within half a frame of it.")
        issues.append({"kind": "state_unaligned", "what": f"The recorded state and signals are on the frames of the "
                       f"{old_name}, which could not be decoded, and no frame of the cameras left was filmed at the "
                       "same time, so they are not used."})
        return issues
    a, b = int(np.flatnonzero(near)[0]), int(np.flatnonzero(near)[-1]) + 1
    if (a, b) != (0, len(t_new)):
        ctx["state_span"] = [a, b]
        issues.append({"kind": "state_partial", "what": f"The robot state was recorded with the {old_name}, which "
                       f"could not be decoded. The cameras left overlap it only from {t_new[a] - zero:.2f} s to "
                       f"{t_new[b - 1] - zero:.2f} s, so the state is used for that part of the episode only."})
    return issues


def recheck(ep_dir: Path) -> None:
    """Run again, for an episode a camera was taken out of, the deterministic checks whose results in its context.json
    read its cameras or the frames its state is on (checks/stream_pairing.py, checks/capture_qc.py, checks/sensors.py),
    so no result names a camera the episode no longer has. A check that crashes now is recorded as errored, the way
    each check's own runner records it (stream_pairing._safe, capture_qc.run_episode, sensors._safe), never left
    describing the episode as it was and never taken out, so a finding that had fired never vanishes unseen."""
    from checks import capture_qc
    from checks import sensors
    from checks import stream_pairing
    ctx = _context(ep_dir)

    def capture(d):
        try:
            return capture_qc.run_episode(d)
        except Exception as e:  # noqa: BLE001 - recorded on every check, as run_episode records a crash inside it
            return capture_qc.errored_record(e)
    runs = {key: (lambda d, m=mode: stream_pairing._safe(m, str(d))[1])
            for mode, (key, _, _) in stream_pairing.MODES.items()}
    runs.update(capture_qc=capture, sensor_checks=lambda d: sensors._safe(str(d))[1])
    for key, fn in runs.items():
        if key not in ctx:
            continue
        old = ctx[key]
        new = fn(ep_dir)
        if key == "sensor_checks" and isinstance(new, dict) and isinstance(old, dict):
            # dead values are judged across a whole folder (checks/sensors.py main), not by run_episode: kept as found
            dead = [n for n in old.get("notes") or [] if isinstance(n, dict) and n.get("check") == "dead_values"]
            if dead:
                new["notes"] = (new.get("notes") or []) + dead
                for c in new.get("checks") or []:
                    if c.get("check") == "dead_values":
                        c["status"] = "fired"
        ctx[key] = new
    write_atomic(ep_dir / "context.json", ctx, indent=1)


# the kinds of reader issue (context.json reader_issues) board clips records
CAMERA_SHORT = "camera_short"                    # a camera's clip has fewer frames than its video lists, the reader's
                                                 # kind for a camera that ends before the episode (SPAN_KINDS)
CLIP_FRAME_COUNT = "clip_frame_count"            # the kind earlier runs gave that fact, taken out when cut again
CAMERA_NOT_DECODABLE = "camera_not_decodable"    # a camera's video does not decode; the episode goes on without it


def clip_end_s(ep_dir: Path, ctx: dict, src: dict, cam: str, got: int) -> float:
    """Where on the episode's clock a camera whose clip came out with got frames stops showing anything: the time of
    its first frame left out, from its capture times (times.npz) or its frame index over the rate, after the frames it
    skips from before the clock's start (start_offsets)."""
    import numpy as np
    fps = float(ctx.get("fps") or 30.0)
    off, skip = start_offsets(ep_dir, src, fps).get(cam, (0.0, 0))
    tp = ep_dir / (ctx.get("real_times") or "times.npz")
    if tp.exists():
        from prepare.camera_clock import load_times
        z = load_times(ep_dir, ctx)
        t = np.asarray(z[cam], dtype=np.float64) if cam in z else None
        main = np.asarray(z[main_cam(src)], dtype=np.float64) if main_cam(src) in z else None
        if t is not None and main is not None and len(t) > skip + got:
            zero = float(ctx.get("clock_zero_s") or 0.0)
            return float(t[skip + got]) - zero
    return off + got / fps


def is_end(x: dict, zero_s: float = 0.0) -> bool:
    """Whether a camera_short issue is one of a camera that ends early (its stretch starts inside the episode), not
    one that starts late (its stretch starts at the stored clock origin)."""
    return x.get("kind") == CAMERA_SHORT and float(x.get("t0_s") or 0.0) > zero_s


def record_cameras(ep_dir: Path, short: dict, broken: dict, cut, keep_clock: bool = False) -> str | None:
    """Record what board clips found wrong with an episode's cameras in its context.json, in reader_issues, the list of
    problems an episode was kept and flagged with ({"kind", "what", "camera"}: a short tag, one plain sentence a
    reviewer reads on the board, the camera's name as the reader gives it, or its view). short is {camera:
    extract_one's counts} for the clips that came out with fewer frames than the camera's video lists, which end
    before the episode does (kind camera_short, with its counts, clip_frames and episode_frames: one issue per
    camera, in place of the reader's of the camera ending early, as both say it shows nothing past a time), broken
    {camera: why} for the cameras whose video does not decode (kind camera_not_decodable), which are taken out of
    the episode (drop_cameras, whose own issues are added), and cut the cameras cut on this run. Other entries are
    never touched; a camera cut again on a later run is recorded as it came out then, and one not cut again keeps
    its entry. board/build.py copies the list into the episode's dataset_checks, where each entry raises a data
    issue (board/families.py), so every board build shows it. The cameras are taken out first, which can move the
    episode's clock and every time in the context with it, and the context's reader issues are read again after
    that. keep_clock: the episode is labelled already (drop_cameras). Returns the new main camera when the main
    camera was taken out, else None."""
    ctx = _context(ep_dir)
    src = json.loads((ep_dir / "sources.json").read_text())
    names = {v: str(s.get("camera_key") or v) for v, s in src.items()}
    name = lambda v: names.get(v, v)
    redo = set(cut) | set(broken)
    order = lambda v: (CAMS + (v,)).index(v)
    own = []                                   # named before a camera's entry leaves the context
    for v in sorted(broken, key=order):
        own.append({"kind": CAMERA_NOT_DECODABLE, "camera": v, "what": f"The {camera_label(v, ctx)} video could not "
                                                                          "be decoded, so this episode is shown and "
                                                                          "labelled without it."})
    main, more = None, []
    if broken:
        main, more = drop_cameras(ep_dir, broken, keep_clock)
        ctx = _context(ep_dir)
        src = json.loads((ep_dir / "sources.json").read_text())
    zero = float(ctx.get("clock_zero_s") or 0.0)
    fps = float(ctx.get("fps") or 30.0)
    episode_end = float(ctx.get("duration_s") or float(ctx.get("n_state_frames") or 0) / fps)
    for v in sorted(short, key=order):
        n = short[v]
        end = clip_end_s(ep_dir, ctx, src, v, int(n["clip_frames"]))
        before = f", before the episode ends at {episode_end:.2f} s" if episode_end > end else ""
        own.append({"kind": CAMERA_SHORT, "camera": name(v),
                    "what": f"The {camera_label(v, ctx)} video ends at {end:.2f} s{before}, so it shows nothing after "
                            f"that; {n['clip_frames']} of the {n['episode_frames']} frames its file lists could be "
                            "read.",
                    "t0_s": round(end + zero, 3),
                    **({"t1_s": round(episode_end + zero, 3)}
                       if episode_end > end else {}),
                    "clip_frames": n["clip_frames"], "episode_frames": n["episode_frames"]})
    ended = {name(v) for v in short}
    issues = [x for x in ctx.get("reader_issues") or []
              if not (isinstance(x, dict) and (x.get("kind") == CLIP_FRAME_COUNT and x.get("camera") in redo
                                               or x.get("kind") == CAMERA_SHORT and "clip_frames" in x
                                               and x.get("camera") in {name(v) for v in redo}
                                               or is_end(x, zero) and x.get("camera") in ended
                                               or x.get("kind") == CAMERA_NOT_DECODABLE and x.get("camera") in broken))]
    issues += own
    have = {(x.get("kind"), x.get("camera")) for x in issues if isinstance(x, dict)}
    issues += [x for x in more if (x["kind"], x.get("camera")) not in have]
    if issues != (ctx.get("reader_issues") or []):
        if issues:
            ctx["reader_issues"] = issues
        else:
            ctx.pop("reader_issues", None)
        write_atomic(ep_dir / "context.json", ctx, indent=1)
    if broken:
        recheck(ep_dir)
    return main


def drop_from_report(rep: dict, left_out: list[dict]) -> None:
    """Take the episodes set_aside_failed moved out of the reader's report (prepare.formats.convert): out of its
    episodes and footage, into its failed list under the name the reader gave them."""
    gone = {f["name"]: f["why"] for f in left_out}
    for e in [e for e in rep["episodes"] if e.get("episode_id") in gone]:
        rep["episodes"].remove(e)
        rep["failed"].append({"name": e["name"], "why": gone[e["episode_id"]]})
    rep["seconds"] = round(sum(e["seconds"] for e in rep["episodes"]), 2)


def note_camera_problems(rep: dict, eps: Path) -> None:
    """Put each kept episode's problems into the reader's report: one note per entry of its context.json reader_issues
    (whichever step recorded it, the reader or board clips; context.json is the one store), "<episode>: <its
    sentence>", which the job page lists, each once, and a camera taken out of the episode out of the episode's
    cameras."""
    for e in rep["episodes"]:
        for x in _context(eps / e["episode_id"]).get("reader_issues") or []:
            if not (isinstance(x, dict) and isinstance(x.get("what"), str) and x["what"].strip()):
                continue
            t = x["what"].strip()
            note = f"{e['name']}: {t[:1].lower()}{t[1:]}"
            if note not in rep["notes"]:
                rep["notes"].append(note)
            if x["kind"] == CAMERA_NOT_DECODABLE:
                (e.get("cameras") or {}).pop(x.get("camera"), None)


def labelled(ep_dir: Path, labels: Path) -> bool:
    """Whether a labelling run has a label of this episode (its record, or a part's of a long recording), whose times
    are on the episode's clock."""
    return (labels / f"{ep_dir.name}.json").exists() or any(labels.glob(f"{ep_dir.name}__p*.json"))


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m board clips", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=Path, required=True, help="a folder of prepared episode_* folders")
    ap.add_argument("--out", type=Path, required=True, help="the clips folder the board reads")
    ap.add_argument("--only", default=None, help="comma list of episode folder names (default: all)")
    ap.add_argument("--jobs", type=int, default=6, help="clips encoded at once")
    ap.add_argument("--clip-threads", type=int, default=2, help="ffmpeg threads per clip")
    ap.add_argument("--force", action="store_true", help="re-cut clips that already exist")
    ap.add_argument("--name-prefix", default="", help="the dataset's file_prefix in the board manifest, if any")
    ap.add_argument("--labels", type=Path, default=None,
                    help="a labelling run's out folder; an episode with a label there keeps its clock (default: "
                         "run/out beside the episodes folder, as python -m review and Data Review lay a job out)")
    args = ap.parse_args()

    ffmpeg = find_ffmpeg()
    labels = args.labels or args.episodes.parent / "run" / "out"
    if args.only:
        want = {x.strip() for x in args.only.split(",") if x.strip()}
        ep_dirs = [args.episodes / e for e in sorted(want) if (args.episodes / e).is_dir()]
    else:
        ep_dirs = sorted(d for d in args.episodes.glob("episode_*") if d.is_dir())

    jobs = []
    for d in ep_dirs:
        jobs += episode_jobs(d, args.out, args.force, args.name_prefix)
    print(f"clips: {len(ep_dirs)} episodes, {len(jobs)} cam-clips to extract "
          f"(jobs={args.jobs}, threads={args.clip_threads}) -> {args.out}")
    ok = fail = 0
    broken: dict[str, dict[str, str]] = {}      # episode folder -> {camera: why its video did not decode}
    short: dict[str, dict[str, dict]] = {}      # episode folder -> {camera: its clip's and the episode's frames}
    cut: dict[str, set] = {}                    # episode folder -> the cameras cut on this run
    unshown_cut: dict[str, set] = {}            # the same for the cameras the model is not shown (unshown_views)
    unshown_broken: dict[str, dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(extract_one, pk, b, du, o, ffmpeg, args.clip_threads, fps, is_main, off, skip, t, q, km, pts):
                (o, ep, cam) for (pk, b, du, o, fps, is_main, off, skip, t, q, km, pts, ep, cam) in jobs}
        for f in as_completed(futs):
            o, ep, cam = futs[f]
            if cam.startswith(UNSHOWN):
                # a camera the model is not shown: kept as cut, or flagged when it does not decode (record_unshown),
                # never taken out of the episode, whose cameras it is not among
                try:
                    f.result()
                    ok += 1
                    unshown_cut.setdefault(ep, set()).add(cam)
                except Exception as e:
                    fail += 1
                    unshown_broken.setdefault(ep, {})[cam] = str(e)[:400]
                    sys.stderr.write(f"clip FAIL {o}: {str(e)[:160]}\n")
                continue
            try:
                counts = f.result()
                ok += 1
                cut.setdefault(ep, set()).add(cam)
                if counts:
                    short.setdefault(ep, {})[cam] = counts
                    sys.stderr.write(f"clip SHORT {o}: {counts['clip_frames']} frames, episode has "
                                     f"{counts['episode_frames']}\n")
            except Exception as e:
                fail += 1
                broken.setdefault(ep, {})[cam] = str(e)[:400]
                sys.stderr.write(f"clip FAIL {o}: {str(e)[:160]}\n")
    for d in ep_dirs:
        if d.name in unshown_cut or d.name in unshown_broken:
            record_unshown(d, unshown_cut.get(d.name, set()), unshown_broken.get(d.name, {}))
    # an episode is left out only when none of its cameras can be cut; one with a camera that works keeps it, the
    # cameras that do not decode are taken out of it, and both kinds of problem are recorded in its context.json
    failed: dict[str, dict[str, str]] = {}
    for d in ep_dirs:
        if d.name not in cut and d.name not in broken:
            continue
        cams = set(json.loads((d / "sources.json").read_text()))
        if cams and cams <= set(broken.get(d.name, {})):
            failed[d.name] = broken[d.name]
            continue
        main = record_cameras(d, short.get(d.name, {}), broken.get(d.name, {}), cut.get(d.name, set()),
                              keep_clock=labelled(d, labels))
        if main:
            # every camera left is cut again: the camera that is main now was cut as a side camera and is cut at the
            # main camera's size, and the episode's clock may have moved to the earliest camera left (reanchor), so
            # every clip is timed on it (start_offsets); each camera's frames are recorded as they come out
            # the cameras the model is not shown too, at their start on the moved clock (CLOCK_TIME_KEYS)
            redo, again, u_cut, u_broken = {}, set(), set(), {}
            for (pk, b, du, o, fps, is_main, off, skip, t, q, km, pts, ep, cam) in episode_jobs(d, args.out, True,
                                                                                    args.name_prefix):
                try:
                    counts = extract_one(pk, b, du, o, ffmpeg, args.clip_threads, fps, is_main, off, skip, t, q, km, pts)
                    if cam.startswith(UNSHOWN):
                        u_cut.add(cam)
                        continue
                    again.add(cam)
                    if counts:
                        redo[cam] = counts
                except Exception as e:
                    if cam.startswith(UNSHOWN):
                        u_broken[cam] = str(e)[:400]
                    sys.stderr.write(f"clip FAIL {o} cut again after the main camera went: {str(e)[:160]}\n")
            record_cameras(d, redo, {}, again)
            record_unshown(d, u_cut, u_broken)
    # the depth clips, after the colour clips they are timed against; one that comes out imperfect is kept, and one
    # that cannot be cut is left out (the page then offers no depth for that camera), never costing the episode; either
    # is recorded on the episode as a reader issue (record_depth), which the board shows
    djobs = [j for d in ep_dirs if d.name not in failed for j in depth_jobs(d, args.out, args.force, args.name_prefix)]
    if djobs:
        print(f"clips: {len(djobs)} depth clips to cut")
        dok = 0
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(extract_depth, d, cam, colour, out, args.clip_threads, fps): (out, d, cam)
                    for (d, cam, colour, out, fps) in djobs}
            for f in as_completed(futs):
                out, d, cam = futs[f]
                try:
                    record_depth(d, cam, f.result())
                    dok += 1
                except Exception as e:
                    sys.stderr.write(f"depth clip FAIL {out}: {str(e)[:160]}\n")
                    record_depth(d, cam, [depth_failed(d, cam, e)])
        print(f"clips: depth ok={dok} fail={len(djobs) - dok}")
    # an episode none of whose cameras decodes costs only itself, never the rest (set_aside_failed); the step fails
    # only when no episode came out at all
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / FAILED).write_text(json.dumps(failed, indent=1))
    kept = len(set(broken) - set(failed))
    print(f"clips: ok={ok} fail={fail}" + (f", {len(short)} episode(s) with a short clip" if short else "")
          + (f", {kept} episode(s) kept without a camera that does not decode" if kept else "")
          + (f", {len(failed)} episode(s) left out" if failed else ""))
    return 1 if failed and len(failed) >= len(ep_dirs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
