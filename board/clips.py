"""Per-episode, per-camera clips for the board (board/serve.py).

    python -m board clips --episodes EPISODES --out CLIPS [--name-prefix habit_]

The board plays each camera as its own synced <video> and expects one mp4 per episode and camera:
  fixed or head camera   CLIPS/<episode>.mp4
  left mounted camera    CLIPS/wrist_left/<episode>.mp4
  right mounted camera   CLIPS/wrist_right/<episode>.mp4
  any other camera       CLIPS/extra1/<episode>.mp4, CLIPS/extra2/..., as the reader numbers them

Some datasets keep their video packed (MolmoAct2: 12 to 50 episodes per mp4), and some cameras are HEVC or AV1,
which browsers do not all play. This cuts each episode's own frames out of its source file (sources.json: the
file, the episode's offset and its exact frame count) into a browser-native H.264 clip, once, sized for where the
page shows that camera (the recipe below) and timed on the episode's clock: every frame keeps its source time, and a
camera that started recording after the main one starts that much later. It is a viewing copy only: labelling
decodes the source files directly and never re-encodes. Idempotent and parallel. An episode with a camera file
that does not decode is listed in CLIPS/failed.json and left out by set_aside_failed; the rest go on.
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


def clip_path(mp4_dir: Path, eid: str, cam: str) -> Path:
    """One camera's clip: the scene camera at the top, the mounted ones and any others in their own folders."""
    if cam in ("left", "right"):
        return mp4_dir / f"wrist_{cam}" / f"{eid}.mp4"
    if cam == "exo":
        return mp4_dir / f"{eid}.mp4"
    return mp4_dir / cam / f"{eid}.mp4"


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
    # scaled only when it must shrink (or lose an odd row or column) or its pixels are not square, never enlarged
    scale = [] if (cw, ch) == (w, h) and not resample else \
        [f"scale={cw}:{ch}:flags=lanczos" + (",setsar=1" if resample else "")]
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
    per camera (ABC-130k, RealOmni, MCAP uploads). The main camera's first frame is 0. A camera whose recording
    started later plays its first frame offset s after it (RealOmni's right gripper camera starts up to 2 s after the
    left one); one that started earlier drops its skip frames from before the main camera's first frame (more than
    half a frame before it), so it never plays early, and its next frame is at most half a frame from 0."""
    tp = ep_dir / "times.npz"
    cams = cams_of(sources)
    if not tp.exists() or len(cams) < 2:
        return {}
    import numpy as np
    with np.load(tp) as z:
        t = {c: np.asarray(z[c], dtype=np.float64) for c in cams if c in z.files and len(z[c])}
    main = main_cam(sources)
    if main not in t:
        return {}
    ref, half = float(t[main][0]), 0.5 / fps
    out = {}
    for c, tc in t.items():
        if c == main:
            continue
        skip = int(np.searchsorted(tc, ref - half, side="left"))
        if skip >= len(tc):
            continue
        off = float(tc[skip]) - ref
        if skip or off >= half:
            out[c] = (off if off >= half else 0.0, skip)
    return out


def extract_one(packed: str, base_s: float, n_frames: int, out_mp4: Path,
                ffmpeg: str, threads: int, fps: float = 30.0, main: bool = True, offset_s: float = 0.0,
                skip: int = 0) -> None:
    """Exactly the episode's n_frames, starting at its first frame. Packed files are on an exact frame grid,
    so seeking half a frame before the episode's offset lands on its first frame whichever way the decimal
    rounds, and -frames:v stops after the last one (never a frame of the next episode). Per-episode files
    (ABC-130k, FastUMI) start at 0. Frame timestamps pass through unchanged, so real capture times stay the
    playback times. main is the camera the page shows large (main_cam); offset_s shifts every timestamp, for a
    camera that started recording after the main one, and skip drops a camera's first frames from before the main
    camera's first one (start_offsets).

    The clip is cut from the file's first video stream alone, and its first frame is put at 0 (then offset_s), so
    neither another stream that starts first (an audio track) nor the seek's half-frame lead shifts it off the
    episode's clock; the frames after it keep their own spacing, so a variable-rate recording plays as recorded."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    # a per-process temp name, so two builders on the same clip can never write one file at once
    tmp = out_mp4.with_suffix(f".{os.getpid()}.tmp.mp4")
    w, h, resample = source_size(ffmpeg, packed)
    want = int(n_frames) - int(skip)
    timing = (f"select=gte(n\\,{int(skip)})",) if skip else ()
    # the first frame is put at 0 after encoding (the setts bitstream filter), which keeps every frame's duration.
    # ffmpeg 7's setpts filter, which did this before, drops them: the last frame then had no duration, the clip's edit
    # list ended where that frame starts, and players and clip_frames drop it (a 453-frame FastUMI camera came out
    # with 452 and its episode was left off the board)
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-threads", str(threads), "-ss", f"{max(0.0, base_s - 0.5 / fps):.6f}",
           "-i", packed, "-map", "0:v:0", "-frames:v", str(want), "-an", "-fps_mode", "passthrough",
           *video_args(w, h, main, threads, resample, pre=timing),
           "-bsf:v", "setts=pts=PTS-STARTPTS:dts=DTS-STARTPTS",
           *(["-output_ts_offset", f"{offset_s:.6f}"] if offset_s >= 0.5 / fps else []), str(tmp)]
    subprocess.run(cmd, check=True, capture_output=True)
    got = clip_frames(tmp)
    if got != want:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{out_mp4.name}: clip has {got} frames, episode has {want}"
                           + (f" after the {skip} before the main camera's first" if skip else ""))
    os.replace(tmp, out_mp4)


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
    cams = cams_of(sources)           # FastUMI has no fixed camera; single-gripper tasks have one camera
    outs = clip_paths(mp4_dir, eid, cams)
    big = main_cam(sources) if cams else None
    offsets = start_offsets(ep_dir, sources, fps)
    jobs = []
    for cam in cams:
        o = outs[cam]
        if force or not (o.exists() and o.stat().st_size > 0 and clip_frames(o) > 0):
            s = sources[cam]
            off, skip = offsets.get(cam, (0.0, 0))
            jobs.append((s["packed"], float(s["base_s"]), int(s["n_frames"]), o, fps, cam == big, off, skip,
                         ep_dir.name, cam))
    return jobs


FAILED = "failed.json"      # in the clips folder: {episode folder: {camera: why its clip could not be cut}}


def set_aside_failed(eps: Path, clips_dir: Path) -> list[dict]:
    """Move the episodes whose clips could not be cut (a camera file that does not decode) out of the episode
    folder, into <eps>_unclipped next to it, so the rest of the upload is labelled and put on the board without them.
    Returns them as the reader reports an episode it could not open, {"name", "why"}, with a plain reason. An
    episode moved on an earlier run is not reported again."""
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
        names = " and ".join(("the fixed or head camera" if c == "exo" else f"the {c} camera") for c in sorted(cams))
        out.append({"name": ep, "why": f"{names} video could not be decoded, so this episode was left out"})
    return out


def drop_from_report(rep: dict, left_out: list[dict]) -> None:
    """Take the episodes set_aside_failed moved out of the reader's report (prepare.formats.convert): out of its
    episodes and footage, into its failed list under the name the reader gave them."""
    gone = {f["name"]: f["why"] for f in left_out}
    for e in [e for e in rep["episodes"] if e.get("episode_id") in gone]:
        rep["episodes"].remove(e)
        rep["failed"].append({"name": e["name"], "why": gone[e["episode_id"]]})
    rep["seconds"] = round(sum(e["seconds"] for e in rep["episodes"]), 2)


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
    failed: dict[str, dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(extract_one, pk, b, du, o, ffmpeg, args.clip_threads, fps, is_main, off, skip): (o, ep, cam)
                for (pk, b, du, o, fps, is_main, off, skip, ep, cam) in jobs}
        for f in as_completed(futs):
            o, ep, cam = futs[f]
            try:
                f.result()
                ok += 1
            except Exception as e:
                fail += 1
                failed.setdefault(ep, {})[cam] = str(e)[:400]
                sys.stderr.write(f"clip FAIL {o}: {str(e)[:160]}\n")
    # a camera file that does not decode costs its own episode, never the rest (set_aside_failed); the step
    # fails only when no episode came out whole
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / FAILED).write_text(json.dumps(failed, indent=1))
    print(f"clips: ok={ok} fail={fail}" + (f", {len(failed)} episode(s) left out" if failed else ""))
    return 1 if failed and len(failed) >= len(ep_dirs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
