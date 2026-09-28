"""Per-episode, per-camera clips for the board (board/serve.py).

    python -m board clips --episodes EPISODES --out CLIPS [--name-prefix habit_]

The board plays each camera as its own synced <video> and expects one mp4 per episode and camera:
  fixed or head camera   CLIPS/<episode>.mp4
  left mounted camera    CLIPS/wrist_left/<episode>.mp4
  right mounted camera   CLIPS/wrist_right/<episode>.mp4

Some datasets keep their video packed (MolmoAct2: 12 to 50 episodes per mp4), and some cameras are HEVC or AV1,
which browsers do not all play. This cuts each episode's own frames out of its source file (sources.json: the
file, the episode's offset and its exact frame count) into a browser-native H.264 clip, once. It is a viewing
copy only: labelling decodes the source files directly and never re-encodes. Idempotent and parallel.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CAMS = ("exo", "left", "right")


def find_ffmpeg() -> str:
    """$FFMPEG, else the ffmpeg on PATH."""
    c = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if not c:
        raise SystemExit("ffmpeg not found: put it on PATH or set FFMPEG")
    return c


def clip_paths(mp4_dir: Path, eid: str) -> dict:
    return {"exo": mp4_dir / f"{eid}.mp4",
            "left": mp4_dir / "wrist_left" / f"{eid}.mp4",
            "right": mp4_dir / "wrist_right" / f"{eid}.mp4"}


# The board's viewing copy of each camera: one recipe for every builder (this file and board/static.py, which
# Data Review also runs), so every copy of a clip is the same.
# Sizes are for the page as it lays the cameras out: the main camera (the fixed or head camera, else the first
# gripper camera) is shown 770 to 1040 css px wide on a 1440 to 1920 px screen, 1540 to 2080 device px on a retina
# one, and fills the screen in full screen; the others are shown 385 to 520 css px wide.
MAIN_MIN_W = 1280            # a main camera narrower than this is upscaled to it, once, with Lanczos and a light
UPSCALE = "lanczos"          # unsharp mask (as the article's clips are), which is sharper than the browser's
SHARPEN = "unsharp=5:5:0.55:5:5:0"   # bilinear stretch
MAIN_BOX = (1920, 1080)      # and a larger one scaled down to fit a 1080p screen in full screen
SIDE_BOX = (1280, 1080)      # a side camera: at most this, never upscaled
CRF = 20                     # with veryfast, the quality of CRF 22 at medium (VMAF at the shown size) in
PRESET = "veryfast"          # 40% of the time, which the whole board and every Data Review upload pay
KEY_S = 2                    # a keyframe every 2 s, so a seek decodes at most 2 s of video
# names the recipe; board/static.py folds it into its media names, so a new recipe gets new names
ENC_TAG = f"h264-crf{CRF}-{PRESET}-main{MAIN_MIN_W}-{MAIN_BOX[0]}x{MAIN_BOX[1]}-side{SIDE_BOX[0]}x{SIDE_BOX[1]}-kf{KEY_S}s-srcts-camclock-v3"


def main_cam(sources: dict) -> str:
    """The camera the page shows large: the fixed or head camera, else the first gripper camera."""
    return next(c for c in CAMS if c in sources)


def clip_size(w: int, h: int, main: bool) -> tuple:
    """(width, height, scale) of the board clip of a w x h source, both even, the aspect kept."""
    if main:
        s = min(max(1.0, MAIN_MIN_W / w), MAIN_BOX[0] / w, MAIN_BOX[1] / h)
    else:
        s = min(1.0, SIDE_BOX[0] / w, SIDE_BOX[1] / h)
    return 2 * round(w * s / 2), 2 * round(h * s / 2), s


def video_args(w: int, h: int, main: bool, threads: int) -> list:
    """The recipe's output arguments for a w x h source."""
    cw, ch, s = clip_size(w, h, main)
    vf = f"scale={cw}:{ch}:flags={UPSCALE},{SHARPEN}" if s > 1 else f"scale={cw}:{ch}:flags=lanczos"
    # enc_time_base demux: every frame keeps its source timestamp exactly. The encoder's default time base is the
    # frame rate's, which rounds a variable-rate recording's times to a 1/30 s grid (up to half a frame off, and a
    # frame squeezed to 0 s where two round to the same tick)
    return ["-vf", vf, "-c:v", "libx264", "-preset", PRESET, "-crf", str(CRF), "-pix_fmt", "yuv420p",
            "-profile:v", "high", "-force_key_frames", f"expr:gte(t,n_forced*{KEY_S})", "-enc_time_base", "demux",
            "-movflags", "+faststart", "-threads", str(threads)]


def source_size(ffmpeg: str, path: str) -> tuple:
    """(width, height) of a video's first stream."""
    probe = Path(ffmpeg).with_name("ffprobe")
    r = subprocess.run([str(probe) if probe.exists() else "ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
                       capture_output=True, text=True, check=True)
    w, h = r.stdout.strip().split(",")[:2]
    return int(w), int(h)


def start_offsets(ep_dir: Path, sources: dict) -> dict:
    """Each camera's first frame on the episode's clock, in seconds after the main camera's: from times.npz, the
    real capture times some datasets keep per camera (ABC-130k, RealOmni). A camera whose recording started later
    gets its clip's timestamps shifted by that much, so every camera plays on the one clock the page syncs them by
    (RealOmni's right gripper camera starts up to 2 s after the left one). One that started earlier is left at 0:
    on this board that is at most 46 ms, under the page's 0.1 s resync tolerance, and its frames before the main
    camera's first one would otherwise need a negative time."""
    tp = ep_dir / "times.npz"
    cams = [c for c in CAMS if c in sources]
    if not tp.exists() or len(cams) < 2:
        return {}
    import numpy as np
    with np.load(tp) as z:
        t0 = {c: float(z[c][0]) for c in cams if c in z.files and len(z[c])}
    ref = t0.get(main_cam(sources))
    return {c: t - ref for c, t in t0.items() if ref is not None and t - ref > 0}


def extract_one(packed: str, base_s: float, n_frames: int, out_mp4: Path,
                ffmpeg: str, threads: int, fps: float = 30.0, main: bool = True, offset_s: float = 0.0) -> None:
    """Exactly the episode's n_frames, starting at its first frame. Packed files are on an exact frame grid,
    so seeking half a frame before the episode's offset lands on its first frame whichever way the decimal
    rounds, and -frames:v stops after the last one (never a frame of the next episode). Per-episode files
    (ABC-130k, FastUMI) start at 0. Frame timestamps pass through unchanged, so real capture times stay the
    playback times. main is the camera the page shows large (main_cam); offset_s shifts every timestamp, for a
    camera that started recording after the main one (start_offsets)."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    # a per-process temp name, so two builders on the same clip can never write one file at once
    tmp = out_mp4.with_suffix(f".{os.getpid()}.tmp.mp4")
    w, h = source_size(ffmpeg, packed)
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-threads", str(threads), "-ss", f"{max(0.0, base_s - 0.5 / fps):.6f}",
           "-i", packed, "-frames:v", str(int(n_frames)), "-an", "-fps_mode", "passthrough",
           *video_args(w, h, main, threads),
           *(["-output_ts_offset", f"{offset_s:.6f}"] if offset_s >= 0.5 / fps else []), str(tmp)]
    subprocess.run(cmd, check=True, capture_output=True)
    got = clip_frames(tmp)
    if got != int(n_frames):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{out_mp4.name}: clip has {got} frames, episode has {n_frames}")
    os.replace(tmp, out_mp4)


def clip_frames(mp4: Path) -> int:
    """Frames in a finished clip (0 when it cannot be read)."""
    try:
        import av
        with av.open(str(mp4)) as c:
            return sum(1 for p in c.demux(c.streams.video[0]) if p.size)
    except Exception:
        return 0


def board_name(eid: str, prefix: str = "") -> str:
    """The episode's name on the board: a dataset whose episode names repeat another's (HABIT's
    episode_000494 and MolmoAct2's) takes the file_prefix of its board manifest entry, episode_habit_000494."""
    return eid.replace("episode_", f"episode_{prefix}", 1) if prefix else eid


def episode_jobs(ep_dir: Path, mp4_dir: Path, force: bool, prefix: str = ""):
    src_p = ep_dir / "sources.json"
    if not src_p.exists():
        return []
    eid = board_name(ep_dir.name, prefix)
    sources = json.loads(src_p.read_text())
    ctx_p = ep_dir / "context.json"
    # half a frame before the episode's first frame, at the episode's own rate (packed LeRobot v3 files
    # at 50 fps put the previous episode's last frame closer than half a 30 fps frame)
    fps = float((json.loads(ctx_p.read_text()) if ctx_p.exists() else {}).get("fps") or 30.0)
    outs = clip_paths(mp4_dir, eid)
    big = main_cam(sources) if any(c in sources for c in CAMS) else None
    offsets = start_offsets(ep_dir, sources)
    jobs = []
    for cam in CAMS:
        if cam not in sources:        # FastUMI has no fixed camera; single-gripper tasks have one camera
            continue
        o = outs[cam]
        if force or not (o.exists() and o.stat().st_size > 0 and clip_frames(o) > 0):
            s = sources[cam]
            jobs.append((s["packed"], float(s["base_s"]), int(s["n_frames"]), o, fps, cam == big, offsets.get(cam, 0.0)))
    return jobs


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
    args = ap.parse_args()

    ffmpeg = find_ffmpeg()
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
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(extract_one, pk, b, du, o, ffmpeg, args.clip_threads, fps, is_main, off): o
                for (pk, b, du, o, fps, is_main, off) in jobs}
        for f in as_completed(futs):
            try:
                f.result()
                ok += 1
            except Exception as e:
                fail += 1
                sys.stderr.write(f"clip FAIL {futs[f]}: {str(e)[:160]}\n")
    print(f"clips: ok={ok} fail={fail}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
