"""The board: every episode's cameras synced to its annotation, with the dataset's issue profile and filters.

    python -m board serve --board BOARD --clips CLIPS [--port 8896] [--title "Data Dashboard"] [--header FILE]

One Python file, no framework: a threaded HTTP server and a single page (INDEX_HTML below). It reads the
episode files in BOARD/qa (board/build.py writes them) and plays the per-episode clips in CLIPS
(board/clips.py writes them: CLIPS/<episode>.mp4 for the fixed or head camera, CLIPS/wrist_left/ and
CLIPS/wrist_right/ for the mounted cameras). Goal frames are cut from the clips on request with ffmpeg and
cached. Each episode can be downloaded as JSON, and a filtered set as JSON Lines (/api/export); its video can be
downloaded with every camera in one frame, whole or a span of it, or one camera at a time (footage below). Other models'
labels of some episodes (BOARD/compare, when the manifest names comparisons) are a comparison: the "Labels by"
control switches the whole board to one model's labels, marked as such, and the comparison view sums them up;
they never enter the board's counts or its downloads. The hand pose files in BOARD/hands, when the manifest names
them, are drawn over the head-camera footage, and BOARD/hand_keypoints holds the same keypoints as a download of
their own. An episode whose recording has contacts (the spans its touch signals say a hand touches something,
label/contacts.py, joined with the model's answer for each by board/build.py) gets a Touch lane under the timeline,
one bar per hand, and a card for the contact under the playhead; the contact checks join the episode's other checks.
The sensors files in BOARD/sensors (board/sensors.py), for episodes whose recording has other signals or depth
streams, are drawn under the lanes as every recorded signal (folded away on an episode with contacts), and each
camera with depth plays its depth clip on request. The header shows the page title and the board's name from
BOARD/manifest.json, or, for a board that is part of a site, the site's own header (--header, an HTML file).

Endpoints (all GET but the export): / (the page), /api/episodes (one rail record per episode),
/api/episode?file=F[&download=1], /api/compare/index, /api/compare/metrics, /api/compare/list?key=K (one model's
rail records), /api/compare/episode?key=K&file=F, /api/hands?file=F, /api/sensors?file=F,
/api/keypoints?file=F[&download=1] (F is a label file, or index.json for the list),
/api/video?id=EPISODE&cam=exo|left|right|depth_<camera>[&download=1] (byte ranges),
/api/footage?id=EPISODE[&t0=S&t1=S][&prepare=1] (every camera in one video, see footage below),
/api/frame?id=EPISODE&cam=C&t=S&w=W (one JPEG), POST /api/export {"files": [...]} (JSON Lines).

Environment:
    BOARD_FRAME_CACHE          frames kept in memory (default 800); raise it so every goal frame of a large board
                               stays cached
    BOARD_FFMPEG_CONCURRENCY   ffmpeg processes cutting frames at once (default 4)
    BOARD_FRAME_DIR            a folder that keeps every frame cut, so a restarted server does not cut them again
    BOARD_FOOTAGE_DIR          where the videos made to download are kept (default BOARD/footage)
    BOARD_FOOTAGE_CONCURRENCY  videos made at once (default 1)
    BOARD_FOOTAGE_THREADS      threads each is made on (default 2)

board/static.py renders the same page with a static data source, so a CDN can serve the board with no server.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import http.server
import json
import math
import os
import re
import shutil
import socketserver
import subprocess
import threading
import urllib.parse
from collections import OrderedDict
from pathlib import Path

from board.families import Families
from compare.metrics import model_names, reasoning_effort


def _find_ffmpeg() -> str | None:
    """$FFMPEG (as board/clips.py reads it, so the frames come from the same binary as the clips), else the ffmpeg on
    PATH, else the usual install places."""
    if os.environ.get("FFMPEG") and os.access(os.environ["FFMPEG"], os.X_OK):
        return os.environ["FFMPEG"]
    found = shutil.which("ffmpeg")
    if found:
        return found
    # a server started outside a login shell may not have Homebrew or /usr/local on its PATH
    for c in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"):
        if os.access(c, os.X_OK):
            return c
    return None


FFMPEG = _find_ffmpeg()


# Frames cut from the clips (goal frames, per-task goal frames, posters) are asked for again and again, and each
# miss runs ffmpeg, so they are kept in a bounded LRU cache keyed by (clip, mtime, t, width). The server is
# threaded, so the cache has a lock, and a semaphore bounds the ffmpeg processes a burst of misses can start.
_FRAME_CACHE = OrderedDict()
_FRAME_CACHE_MAX = int(os.environ.get("BOARD_FRAME_CACHE", 800))
_FRAME_LOCK = threading.Lock()
_FFMPEG_SEM = threading.Semaphore(int(os.environ.get("BOARD_FFMPEG_CONCURRENCY", 4)))
# BOARD_FRAME_DIR keeps every frame cut on disk as well, so a restarted server (a deploy) serves the posters and goal
# frames it cut before without running ffmpeg again; the key holds the clip's mtime, so a rebuilt clip is cut anew
_FRAME_DIR = Path(os.environ["BOARD_FRAME_DIR"]) if os.environ.get("BOARD_FRAME_DIR") else None


_CLIP_TIMES: dict = {}


def _clip_times(mp4: Path) -> list:
    """The clip's frame times in seconds after its first frame (sorted presentation times), read once per clip
    version; [] when they cannot be read."""
    try:
        key = (str(mp4), mp4.stat().st_size, mp4.stat().st_mtime_ns)
    except OSError:
        return []
    hit = _CLIP_TIMES.get(key)
    if hit is not None:
        return hit
    probe = Path(FFMPEG).with_name("ffprobe") if FFMPEG else None
    try:
        r = subprocess.run([str(probe) if probe and probe.exists() else "ffprobe", "-v", "error", "-select_streams",
                            "v:0", "-show_entries", "packet=pts_time,flags", "-of", "csv=p=0", str(mp4)],
                           capture_output=True, text=True, timeout=60)
        ts = sorted(float(a) for a, *f in (ln.split(",") for ln in r.stdout.split() if ln)
                    if a not in ("", "N/A") and "D" not in "".join(f))
    except (OSError, subprocess.SubprocessError, ValueError):
        ts = []
    rel = [x - ts[0] for x in ts] if ts else []
    if len(_CLIP_TIMES) > 2048:
        _CLIP_TIMES.clear()
    _CLIP_TIMES[key] = rel
    return rel


def extract_frame(mp4: Path, t: float, max_w: int = 640) -> bytes | None:
    """One JPEG from mp4 at time t (seconds), at most max_w wide, or None when it cannot be cut."""
    if not FFMPEG or not mp4.exists():
        return None
    try:
        key = (str(mp4), int(mp4.stat().st_mtime), round(max(0.0, t), 3), int(max_w))
    except OSError:
        return None
    with _FRAME_LOCK:
        hit = _FRAME_CACHE.get(key)
        if hit is not None:
            _FRAME_CACHE.move_to_end(key)
            return hit
    disk = _FRAME_DIR / (hashlib.sha1(repr(key).encode()).hexdigest() + ".jpg") if _FRAME_DIR else None
    if disk is not None and disk.is_file():
        out = disk.read_bytes()
        _remember_frame(key, out)
        return out
    # Clips hold exactly the episode's own frames, their first at the clip's start. The frame shown is the clip's
    # own frame nearest t, from its frame times (any rate, variable or not), and the seek lands half a gap before
    # it (an input seek counts from the clip's start), so a 4-decimal seek can never round past it; a time at or
    # past the clip's end (a goal frame on the last instant) is its last frame, and a frame that cannot be cut
    # falls back to the one before.
    rel = _clip_times(mp4)
    if rel:
        import bisect
        i = bisect.bisect_left(rel, max(0.0, t))
        i = min(range(max(0, i - 1), min(len(rel), i + 1)), key=lambda j: (abs(rel[j] - t), j))

        def half_gap(j):
            gaps = [g for g in ([rel[j] - rel[j - 1]] if j > 0 else []) + ([rel[j + 1] - rel[j]] if j + 1 < len(rel)
                                                                           else []) if g > 0]
            return 0.5 * (min(gaps) if gaps else 1 / 30.0)
        tries = [rel[j] - half_gap(j) for j in (i, i - 1, i - 2) if j >= 0]
    else:
        k = max(0, int(round(max(0.0, t) * 30)))
        tries = [(kk - 0.5) / 30 for kk in (k, k - 1, k - 2) if kk >= 0]
    out = None
    for seek in tries:
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
               "-ss", f"{max(0.0, seek):.4f}", "-i", str(mp4), "-frames:v", "1",
               "-vf", f"scale='min({max_w},iw)':-2", "-q:v", "3",
               "-f", "image2pipe", "-vcodec", "mjpeg", "-"]
        try:
            with _FFMPEG_SEM:
                r = subprocess.run(cmd, capture_output=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            return None
        out = r.stdout if (r.returncode == 0 and r.stdout) else None
        if out:
            break
    if out:
        _remember_frame(key, out)
        if disk is not None:
            try:
                disk.parent.mkdir(parents=True, exist_ok=True)
                tmp = disk.with_suffix(f".{os.getpid()}.{threading.get_ident()}.part")
                tmp.write_bytes(out)
                tmp.replace(disk)
            except OSError:
                pass                      # the disk copy is only a cache
    return out


def _remember_frame(key, jpg: bytes) -> None:
    with _FRAME_LOCK:
        _FRAME_CACHE[key] = jpg
        _FRAME_CACHE.move_to_end(key)
        while len(_FRAME_CACHE) > _FRAME_CACHE_MAX:
            _FRAME_CACHE.popitem(last=False)


# any other camera the recording has, and a camera the model is not shown (board/clips.py clip_path, unshown_views)
EXTRA_CAM = re.compile(r"(?:extra|unshown)\d{1,2}")
DEPTH_CAM = re.compile(r"depth_(exo|left|right|extra\d{1,2})")   # a camera's depth clip (board/clips.py depth_clip)


def clip_path(clips: Path, eid: str, cam: str) -> Path:
    """The clip of one camera: cam "left" and "right" are the mounted cameras, extra1, extra2, ... the others,
    depth_<camera> a camera's depth clip, anything else the main one."""
    if cam == "left":
        return clips / "wrist_left" / f"{eid}.mp4"
    if cam == "right":
        return clips / "wrist_right" / f"{eid}.mp4"
    if EXTRA_CAM.fullmatch(cam or "") or DEPTH_CAM.fullmatch(cam or ""):
        return clips / cam / f"{eid}.mp4"
    return clips / f"{eid}.mp4"


def _under(base: Path, p: Path) -> bool:
    """True iff resolved p is inside base. Blocks ../ traversal on request params."""
    try:
        Path(p).resolve().relative_to(Path(base).resolve())
        return True
    except (ValueError, OSError):
        return False


# The episode's video as one file to download, play and cut: the main camera at its clip's size and the mounted
# cameras stacked in a column beside it, on black with a small gap. It is made from the board's clips, so every
# camera keeps its clip's timestamps (one that started recording late starts late here too) and a time on the page is
# the same time in the file. It is 30 fps at a constant rate, so an editor cuts it on whole frames. Each span is made
# once, on request, and kept in FOOTAGE_DIR; one is made at a time, niced, on a few threads, since a board can be
# served from a machine that also records.
FOOTAGE_GAP = 8
FOOTAGE_FPS = 30
FOOTAGE_TAG = f"footage-v2-h264-crf20-veryfast-{FOOTAGE_FPS}fps-gap{FOOTAGE_GAP}"   # a new recipe makes new files
_FOOTAGE_SEM = threading.Semaphore(int(os.environ.get("BOARD_FOOTAGE_CONCURRENCY", 1)))
_FOOTAGE_THREADS = int(os.environ.get("BOARD_FOOTAGE_THREADS", 2))
_FOOTAGE_LOCKS: dict = {}
_FOOTAGE_LOCKS_LOCK = threading.Lock()


def footage_cams(clips: Path, eid: str) -> list:
    """[(cam, clip)] of the episode's clips on disk, the main camera first: the fixed or head camera, else the first
    gripper camera, as the page shows them."""
    extra = sorted((d.name for d in clips.iterdir() if d.is_dir() and EXTRA_CAM.fullmatch(d.name)),
                   key=lambda n: (n.startswith("unshown"), int(re.sub(r"\D", "", n)))) if clips.is_dir() else []
    return [(c, clip_path(clips, eid, c)) for c in ("exo", "left", "right", *extra) if clip_path(clips, eid, c).is_file()]


def footage_layout(sizes: list, gap: int = FOOTAGE_GAP) -> tuple:
    """(width, height, [(x, y, w, h)] per camera) for clips of the given (w, h), the main one first. The main camera
    keeps its size; the others share its height in a column beside it, each at its own aspect (its width the even
    number nearest it, so within a pixel) and never enlarged, the column centred when they come out shorter."""
    even = lambda v: max(2, 2 * int(v / 2))
    nearest = lambda v, cap: max(2, min(2 * round(v / 2), cap - cap % 2))
    up = lambda v: v + v % 2           # the frame is even in both directions, as H.264 needs
    w0, h0 = sizes[0]
    cells = [(0, 0, w0, h0)]
    side = sizes[1:]
    if not side:
        return up(w0), up(h0), cells
    hs = even(min((h0 - gap * (len(side) - 1)) / len(side), *(h for _, h in side)))
    col_h = hs * len(side) + gap * (len(side) - 1)
    x, y, col_w = w0 + gap, 2 * int((h0 - col_h) / 4), 0
    for w, h in side:
        ws = nearest(w * hs / h, w)
        cells.append((x, y, ws, hs))
        y += hs + gap
        col_w = max(col_w, ws)
    return up(x + col_w), up(h0), cells


def _probe(p: Path) -> tuple:
    """(width, height, duration s, frame rate) of a clip. The duration runs to the end of its last frame (its
    time plus its own length), which a variable-rate recording's container duration can stop short of."""
    probe = Path(FFMPEG).with_name("ffprobe") if FFMPEG else None
    r = subprocess.run([str(probe) if probe and probe.exists() else "ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,avg_frame_rate:format=duration:packet=pts_time,duration_time",
                        "-of", "json", str(p)], capture_output=True, text=True, timeout=30, check=True)
    j = json.loads(r.stdout)
    s = j["streams"][0]
    dur = float(j["format"]["duration"])
    pk = [(float(x["pts_time"]), float(x.get("duration_time") or 0)) for x in j.get("packets", [])
          if x.get("pts_time") not in (None, "N/A")]
    if pk:
        last, d = max(pk)
        dur = max(dur, last + d)
    num, _, den = str(s.get("avg_frame_rate") or "0/1").partition("/")
    rate = float(num) / float(den) if den and float(den) else 0.0
    return int(s["width"]), int(s["height"]), dur, rate


def footage_command(inputs: list, t0: float, t1: float, out: Path, threads: int = 2) -> list:
    """The ffmpeg command that composes [(clip, (w, h))] (the main camera first) from t0 to t1 s into out. Timestamps
    are kept as the clips have them (-copyts), a black canvas runs from t0 to t1 at FOOTAGE_FPS, and each camera is
    laid on it at its place, so the file shows at each instant what the page shows then."""
    # inputs may carry each camera's frame rate; the canvas runs at FOOTAGE_FPS, or at the fastest camera's own rate
    # when that is faster, so no frame is dropped
    rates = [x[2] for x in inputs if len(x) > 2 and isinstance(x[2], (int, float)) and x[2] > 0]
    fps = max([FOOTAGE_FPS] + [int(-(-float(r) // 1)) for r in rates])
    inputs = [(x[0], x[1]) for x in inputs]
    W, H, cells = footage_layout([wh for _, wh in inputs])
    cmd = [FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-copyts"]
    for clip, _ in inputs:
        # from a few seconds early, so each camera has the frame showing at t0 (clips have a keyframe every 2 s)
        cmd += ["-ss", f"{max(0.0, t0 - 3.0):.3f}", "-i", str(clip)]
    g = [f"color=c=black:s={W}x{H}:r={fps}:d={t1 - t0:.3f},setpts=PTS+{t0:.3f}/TB[b0]"]
    # a camera keeps its last frame on to t1, as the page's video does once it has played to its end: the overlay
    # dropped a camera to black from its last frame's start (a 15 fps camera's last 1/30 s), so each is held there
    # (tpad), and the file keeps the canvas's own frames (those before t1), which the held frames would otherwise run past
    hold = f"tpad=stop_mode=clone:stop_duration={t1 - t0 + 1:.3f}"
    for i, ((_, wh), (x, y, w, h)) in enumerate(zip(inputs, cells)):
        g.append(f"[{i}:v]{'' if (w, h) == tuple(wh) else f'scale={w}:{h}:flags=lanczos,'}setsar=1,{hold}[c{i}]")
        g.append(f"[b{i}][c{i}]overlay={x}:{y}:eof_action=pass[b{i + 1}]")
    g.append(f"[b{len(inputs)}]setpts=PTS-STARTPTS,format=yuv420p[v]")
    n = math.ceil(round((t1 - t0) * fps, 6))
    return cmd + ["-filter_complex", ";".join(g), "-map", "[v]", "-frames:v", str(n), "-an", "-c:v", "libx264",
                  "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-g", str(2 * FOOTAGE_FPS), "-threads",
                  str(threads), "-filter_complex_threads", "1", "-movflags", "+faststart", str(out)]


def footage(eid: str, t0: float = 0.0, t1: float | None = None) -> tuple | None:
    """(file, download name) of the episode's video from t0 to t1 s (the whole episode by default), made now if it
    was not made before; None when the episode has no clip. Raises RuntimeError when ffmpeg fails."""
    cams = footage_cams(MP4_DIR, eid)
    if not cams or not FFMPEG:
        return None
    probes = [_probe(p) for _, p in cams]
    dur = probes[0][2]
    t0 = round(max(0.0, min(float(t0), dur)), 3)
    t1 = round(dur if t1 is None else max(0.0, min(float(t1), dur)), 3)
    if t1 - t0 < 1.0 / FOOTAGE_FPS:
        return None
    whole = t0 == 0 and t1 >= round(dur, 3)
    name = f"{eid}.mp4" if whole else f"{eid}_{t0:.1f}-{t1:.1f}s.mp4"
    key = hashlib.sha1(json.dumps([eid, t0, t1, FOOTAGE_TAG, [(str(p), p.stat().st_size, int(p.stat().st_mtime))
                                                             for _, p in cams]]).encode()).hexdigest()[:20]
    out = FOOTAGE_DIR / f"{key}.mp4"
    if out.is_file():
        return out, name
    with _FOOTAGE_LOCKS_LOCK:
        lock = _FOOTAGE_LOCKS.setdefault(key, threading.Lock())
    with lock:                       # a second request for the same span waits for the first one's file
        if out.is_file():
            return out, name
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(f".{os.getpid()}.{threading.get_ident()}.part.mp4")
        cmd = footage_command([(p, pr[:2], pr[3]) for (_, p), pr in zip(cams, probes)], t0, t1, tmp, _FOOTAGE_THREADS)
        nice = shutil.which("nice")
        with _FOOTAGE_SEM:
            r = subprocess.run(([nice, "-n", "19"] if nice else []) + cmd, capture_output=True, text=True)
        if r.returncode != 0 or not tmp.is_file():
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"the video could not be made: {(r.stderr or '').strip()[-300:]}")
        from board.clips import frame_lengths
        frame_lengths(tmp)           # some ffmpeg builds leave the last frame 0 s long, so it is never shown
        os.replace(tmp, out)
    return out, name


PORT = 8896
PAGE_TITLE = "Data Dashboard"
HERE = Path.cwd().resolve()           # the episode files (set by main from --board)
MP4_DIR = HERE / "clips"              # the clips (set by main from --clips)
COMPARE_DIR = HERE.parent / "compare" # other models' labels, beside qa/ (board/build.py writes them)
HANDS_DIR = HERE.parent / "hands"     # the hand pose drawn over head-camera footage (board/build.py)
KEYPOINTS_DIR = HERE.parent / "hand_keypoints"   # the same keypoints as a download, in the dataset video's pixels
SENSORS_DIR = HERE.parent / "sensors" # the episodes' other signals and depth (board/sensors.py)
FOOTAGE_DIR = HERE.parent / "footage" # the videos made to download (footage), or $BOARD_FOOTAGE_DIR (set by main)
BOARD_NAME = ""                     # the manifest's "board" (set by main)
HEADER = None                         # a site header in place of the title bar (--header)

_LIST_CACHE = {}         # folder -> (signature, rail records)
_LIST_LOCK = threading.Lock()

# Plain display names for the issue tags (board/families.py reads the same file). A tag with no entry shows as its
# words with the first letter capitalised.
TAG_NAMES_PATH = Path(__file__).resolve().parent / "tag_names.json"
FAMILIES = Families()


def load_tag_names() -> dict:
    try:
        d = json.loads(TAG_NAMES_PATH.read_text())
    except (OSError, ValueError):
        return {"data_issues": {}, "operator_mistakes": {}}
    return {k: v for k, v in d.items() if not k.startswith("_")}


def board_name(board: Path) -> str:
    """The board's name as its manifest gives it ("board"), or "" when it has none."""
    try:
        return str(json.loads((board / "manifest.json").read_text()).get("board") or "")
    except (OSError, ValueError, AttributeError):
        return ""


def labels_license(board: Path) -> dict | None:
    """The license the board's owner gives its labels, as its manifest names it ("labels_license": {"name", "url",
    "by"}), or None: a board states no license for labels it was not told of."""
    try:
        lic = json.loads((board / "manifest.json").read_text()).get("labels_license")
    except (OSError, ValueError, AttributeError):
        return None
    return lic if isinstance(lic, dict) and lic.get("name") else None


def list_episodes(here: Path | None = None) -> list:
    """The rail records of the board's own labels, or of one model's labels (compare/<key>). Cached per folder and
    rebuilt only when its set of json files or their mtimes change, so concurrent page loads don't each re-parse
    every episode file."""
    here = here or HERE
    try:
        # one stat per file: the list is signed on every page load, and a board has thousands of files
        sig = tuple(sorted((p.name, int(st.st_mtime), st.st_size) for p in here.glob("*.json") for st in [p.stat()]))
    except OSError:
        sig = None
    with _LIST_LOCK:
        hit = _LIST_CACHE.get(str(here))
        if sig is not None and hit and hit[0] == sig:
            return hit[1]
    out = rail_records(here)
    with _LIST_LOCK:
        _LIST_CACHE[str(here)] = (sig, out)
    return out


_LIST_BODY = {}          # folder -> (rail records, their JSON, gzipped): encoded once per list, not per request


def list_json(here: Path | None = None) -> tuple:
    """(JSON, gzipped JSON) of list_episodes(here). Every page load asks for the whole list, and encoding a large
    board's list (megabytes) on each request held the server's one interpreter for most of its time."""
    out = list_episodes(here)
    key = str(here or HERE)
    with _LIST_LOCK:
        hit = _LIST_BODY.get(key)
    if hit and hit[0] is out:
        return hit[1], hit[2]
    raw = json.dumps(out).encode()
    gz = gzip.compress(raw, compresslevel=5)
    with _LIST_LOCK:
        _LIST_BODY[key] = (out, raw, gz)
    return raw, gz


def _families(d: dict) -> dict:
    c = FAMILIES.classify(d)
    fams, minor = set(c["counted"]), set(c["minor"])
    # capture checks (checks/capture_qc.py): each kind that fired is a family of its own, keyed by the check's id so a
    # renamed check keeps its family (capture_catalog names it)
    cq = (d.get("dataset_checks") or {}).get("capture_qc") or {}
    for f in cq.get("flags") or []:
        if isinstance(f, dict) and f.get("check"):
            fams.add(CAPTURE_PREFIX + str(f["check"]))
    return {"families": sorted(fams), "minor_families": sorted(minor - fams)}


CAPTURE_PREFIX = "cq:"


def capture_catalog() -> dict:
    """The page's entry for each capture check's family: its name as checks/capture_qc.py gives it."""
    from checks.capture_qc import NAMES
    return {CAPTURE_PREFIX + k: {"name": name, "list": "data", "check": True} for k, (name, _group) in NAMES.items()}


# What a public copy of a label leaves out: how the label was made, rechecked or replaced (the run id and code
# commit, a replaced earlier label, frame verdicts, supplements, withheld checks). The label file keeps them, since
# board/build.py reads them; the page, the downloads and the exports never show them.
PRIVATE_KEYS = ("_run", "_replaced_label", "_supplements", "_verification", "_withheld_checks", "_carried_verdicts")
PRIVATE_RUN_KEYS = ("run_id", "code")


def public_label(d: dict) -> dict:
    """A label as anyone may see it: every field but PRIVATE_KEYS, and another model's label (_compare) without the
    run and code it came from."""
    d = {k: v for k, v in d.items() if k not in PRIVATE_KEYS}
    if isinstance(d.get("_compare"), dict):
        d["_compare"] = {k: v for k, v in d["_compare"].items() if k not in PRIVATE_RUN_KEYS}
    return d


def episode_view(d: dict) -> dict:
    """The episode as the page shows it: its public fields (public_label), each flagged issue carrying the family it
    is counted under and whether it counts (Families.counts), so the episode's own list names and counts a problem
    exactly as the filter does, and the checks a manifest rule withheld on this dataset (drop_check, kept in the
    label file's _withheld_checks) as set_aside_checks, each with the rule's reason and whether it fired, so the page
    lists them as set aside, never hides them. The label file itself is not changed."""
    held = d.get("_withheld_checks") if isinstance(d.get("_withheld_checks"), dict) else {}
    d = public_label(d)
    aside = [{"check": k, "reason": v.get("reason") or "",
              "flagged": any(bool(r.get(f)) for f in ("flagged", "crossed", "sped_up_recording"))}
             for k, v in held.items() if isinstance(v, dict) for r in [v.get("result") or {}] if isinstance(r, dict)]
    if aside:
        d["set_aside_checks"] = aside
    ds = d.get("dataset")
    for key in ("data_issues", "operator_mistakes"):
        for i in d.get(key) or []:
            if isinstance(i, dict) and i.get("issue"):
                i["family"] = FAMILIES.family_of(key, i, ds)
                i["counted"] = FAMILIES.counts(key, i)
    return d


def _duration_s(ts):
    try:
        ts = [float(t) for t in (ts or [])]
    except (TypeError, ValueError):
        return None
    if not ts:
        return None
    step = (ts[-1] - ts[0]) / (len(ts) - 1) if len(ts) > 1 else 0.0
    return round(ts[-1] + step, 3)


SEV_RANK = {"high": 3, "medium": 2, "low": 1}


def rail_records(here: Path | None = None) -> list:
    """One rail record per episode file in `here` (default: the served data dir), with what the rail's cards, the
    issue filter, the dataset tabs and the search read. board/static.py calls this too, so the static build's lists
    are the same records the live board serves."""
    out = []
    for p in sorted((here or HERE).glob("*.json")):
        try:
            d = json.loads(p.read_text())
            if isinstance(d, dict) and "episode_prompt" in d:
                out.append(_rail_record(p, d))
        except (OSError, ValueError, TypeError, AttributeError):
            continue                  # not an episode file, or one too malformed to list
    return out


def _rail_record(p: Path, d: dict) -> dict:
    # the issues that count (Families.counts) are summarised; the minor ones are only tallied
    all_issues = [i for i in (d.get("data_issues") or []) if i and i.get("issue")]
    issues = [i for i in all_issues if FAMILIES.counts("data_issues", i)]
    top = max(issues, key=lambda i: SEV_RANK.get(str(i.get("severity") or "").lower(), 0), default=None)
    max_sev = str(top.get("severity") or "").lower() if top else None
    # operator mistakes: the recording is faithful but the demonstration went wrong. Summarised apart from data
    # issues so the rail never counts a fumble as a fault of the data.
    all_mistakes = [i for i in (d.get("operator_mistakes") or []) if i and i.get("issue")]
    mistakes = [i for i in all_mistakes if FAMILIES.counts("operator_mistakes", i)]
    # head-camera sessions have no single completion: the rail shows "N/M tasks" from the per-task outcomes
    tasks = [t for t in (d.get("tasks") or []) if t and (t.get("task") or t.get("start_s") is not None)]
    return {
        "episode_id": (d.get("_meta") or {}).get("episode_id") or p.stem,
        "file": p.name,
        "dataset": d.get("dataset"),
        "task_completed": (d.get("completion") or {}).get("task_completed"),
        "failure_kind": (d.get("completion") or {}).get("failure_kind"),
        "episode_prompt": d.get("episode_prompt") or "",
        "n_issues": len(issues),
        "max_severity": max_sev if max_sev in SEV_RANK else None,
        "top_issue": (top.get("issue") or "") if top else "",
        "n_minor_issues": len(all_issues) - len(issues),
        "n_mistakes": len(mistakes),
        "n_minor_mistakes": len(all_mistakes) - len(mistakes),
        "n_tasks": len(tasks),
        "n_task_success": sum(1 for t in tasks if (t.get("outcome") or "").lower() == "success"),
        # the episode's length (board/build.py measures it); files without one fall back on the sampled times
        "duration_s": d.get("duration_s") or _duration_s(d.get("timesteps_s")),
        # which problems the episode has (board/families.py): the families that count, and those it has only
        # below the counting bar
        **_families(d),
        # head cameras only: seconds in which the wearer's hands are out of view (None on any other rig)
        "hands_hidden_s": FAMILIES.hands_hidden_seconds(d),
        # the board's own model reply gave no labels (board/to_board.py label_failed): unparsed or cut_off
        **({"label_failed": d["_label_failed"].get("status")} if isinstance(d.get("_label_failed"), dict) else {}),
        # another model's labels (compare/): how its response came out (parsed, unparsed, cut_off, no_response)
        **({"cmp_status": d["_compare"].get("status")} if isinstance(d.get("_compare"), dict) else {}),
    }


INDEX_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport"
  content="width=device-width, initial-scale=1"><title>__PAGE_TITLE__</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTExIiBoZWlnaHQ9IjEyNiIgdmlld0JveD0iMCAwIDExMSAxMjYiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0zNS40MDA0IDAuMDEwNzQyMkwzNS4yNDAyIDAuMDUwNzgxMkwzNS4wODk4IDAuMTEwMzUyTDM0Ljk0MDQgMC4yMDk5NjFMMzQuODQ5NiAwLjI5MDAzOUwzNC43Njk1IDAuMzc5ODgzTDM0LjY3OTcgMC41MjA1MDhMMzQuNjIwMSAwLjYyMDExN0wzNC41ODAxIDAuNzYwNzQyTDM0LjU0OTggMC45MTk5MjJMMzQuNTU5NiAxLjE2MDE2TDM0LjYyMDEgMS4zNzk4OEwzNC42Nzk3IDEuNDgwNDdMMzQuNzY5NSAxLjYyMDEyTDgwLjY3MDkgNTQuNjE4MkwwLjkxOTkyMiA2MS44MjAzTDAuNzY5NTMxIDYxLjg1MDZMMC42MTAzNTIgNjEuOTAwNEwwLjQ2OTcyNyA2MS45NzA3TDAuMzQ5NjA5IDYyLjA2MDVMMC4yNjk1MzEgNjIuMTQwNkwwLjE5MDQzIDYyLjIzMDVMMC4xMjk4ODMgNjIuMzMwMUwwLjA3MDMxMjUgNjIuNDQwNEwwLjA0MDAzOTEgNjIuNTRMMC4wMDk3NjU2MiA2Mi43MDAyTDAgNjIuODIwM0wwLjAwOTc2NTYyIDYyLjk0MDRMMC4wNDAwMzkxIDYzLjEwMDZMMC4wNzAzMTI1IDYzLjIwMDJMMC4xMjk4ODMgNjMuMzEwNUwwLjE5MDQzIDYzLjQxMDJMMC4yNjk1MzEgNjMuNUwwLjM0OTYwOSA2My41ODAxTDAuNDY5NzI3IDYzLjY2OTlMMC42MTAzNTIgNjMuNzQwMkwwLjc2OTUzMSA2My43OUwwLjkxOTkyMiA2My44MjAzTDgwLjY2OTkgNzEuMDIxNUwzNC43Njk1IDEyNC4wMjFMMzQuNzAwMiAxMjQuMTIxTDM0LjYyMDEgMTI0LjI2MUwzNC41ODAxIDEyNC40TDM0LjU0OTggMTI0LjY0MUwzNC41NTk2IDEyNC44MDFMMzQuNTk5NiAxMjQuOTVMMzQuNjc5NyAxMjUuMTIxTDM0Ljc0MDIgMTI1LjIzTDM0LjgzOTggMTI1LjM1MUwzNC45NDA0IDEyNS40MzFMMzUuMDg5OCAxMjUuNTNMMzUuMjQwMiAxMjUuNTkxTDM1LjM2MDQgMTI1LjYyMUwzNS41NDk4IDEyNS42NDFMMzUuNzAwMiAxMjUuNjMxTDM1Ljg2MDQgMTI1LjU5MUwzNi4wNzAzIDEyNS40OUw5MC43MDYxIDgzLjM3N0w5MC40NTAyIDEwMS4xMTFMOTAuNDgwNSAxMDEuMjYyTDkwLjUzMDMgMTAxLjQxMUw5MC41OTk2IDEwMS41NTFMOTAuNjkwNCAxMDEuNjgyTDkwLjc5OTggMTAxLjc5MUw5MC45Mjk3IDEwMS44ODFMOTEuMDcwMyAxMDEuOTUxTDkxLjIxOTcgMTAyLjAwMUw5MS4zMzAxIDEwMi4wMjFMOTEuNDUwMiAxMDIuMDMxTDkxLjU5OTYgMTAyLjAyMUw5MS43NTk4IDEwMS45ODFMOTEuODcwMSAxMDEuOTMyTDkyLjA0IDEwMS44NDFMOTIuMTUwNCAxMDEuNzQxTDkyLjI1OTggMTAxLjYxMUw5Mi4zMzk4IDEwMS40ODFMMTA5LjkzIDY3LjMyMTNMMTEwLjI1IDY2LjYyMTFMMTEwLjUxIDY1LjkxMTFMMTEwLjcyIDY1LjE5MTRMMTEwLjg3IDY0LjQyMDlMMTEwLjk3IDYzLjYzMDlMMTExIDYyLjgyMTNMMTEwLjk3IDYyLjAxMTdMMTEwLjg3IDYxLjIyMTdMMTEwLjcyIDYwLjQ1MTJMMTEwLjUxIDU5LjczMTRMMTEwLjI1IDU5LjAyMTVMMTA5LjkzIDU4LjMyMTNMOTIuMzM5OCAyNC4xNjAyTDkyLjI1OTggMjQuMDMwM0w5Mi4xNjAyIDIzLjkwMDRMOTIuMDQgMjMuODEwNUw5MS44NzAxIDIzLjcxTDkxLjc1OTggMjMuNjYwMkw5MS41OTk2IDIzLjYyOTlMOTEuNDUwMiAyMy42MTA0TDkxLjMzOTggMjMuNjIwMUw5MS4yMTk3IDIzLjY0MDZMOTEuMDcwMyAyMy42OTA0TDkwLjkyOTcgMjMuNzYwN0w5MC43OTk4IDIzLjg1MDZMOTAuNjkwNCAyMy45Nkw5MC41OTk2IDI0LjA4OThMOTAuNTMwMyAyNC4yMzA1TDkwLjQ4MDUgMjQuMzc5OUw5MC40NTAyIDI0LjUzMDNMOTAuNzA1MSA0Mi4yNTg4TDM2LjEzOTYgMC4xOTA0M0wzNS45Mjk3IDAuMDgwMDc4MUwzNS43ODAzIDAuMDMwMjczNEwzNS41NDk4IDBMMzUuNDAwNCAwLjAxMDc0MjJaIiBmaWxsPSIjMEEwQTBBIi8+Cjwvc3ZnPgo=">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700
&family=Geist+Mono:wght@400;500;600&display=swap">
<style>
/* ---------- design tokens ---------- */
:root {
  /* type: Geist and Geist Mono */
  --sans: "Geist", -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
  --mono: "Geist Mono", ui-monospace, "SF Mono", Menlo, monospace;

  /* surfaces: paper and ink; footage stays on dark panels */
  --bg:      #f3f4f6;
  --surface: #ffffff;
  --raised:  #ffffff;
  --border:  rgba(28, 28, 26, 0.22);
  --border-strong: rgba(28, 28, 26, 0.45);
  --row-divider: rgba(28, 28, 26, 0.18);

  --fg:          #1c1c1a;
  --fg-2:        #3d3d39;
  --fg-3:        #55544f;
  --fg-disabled: #8c8b86;
  --fg-muted: var(--fg-2);
  --fg-faint: var(--fg-3);

  /* one meaning per colour: crimson, something wrong with the data or the outcome; green, the task done; indigo, the
     operator's performance; teal, interactive; blue and purple, the left and right arm; ink and greys, everything
     else. Severity is never a hue: stronger fill is higher severity. On the dark video panels the crimson is #e58c9a
     and the indigo #a3aee0. No orange or amber anywhere. */
  --accent:    #45818e;   /* interactive: links, selection, the playhead */
  --success:   #2f7d52;   /* the task done */
  --warning:   #4d5aa0;   /* the operator's performance */
  --danger:    #b3263c;   /* something wrong with the data or the outcome */
  --arm-left:  #4f6d8f;
  --arm-right: #8a5a9a;

  --header-h: 44px;   /* the page header: title and board name */
  --cov-h: 62px;
  --rail-w: 240px;    /* the episode rail's column; the board totals above it take the same width */
  --top-h: calc(var(--header-h) + var(--cov-h));
  --r-sm: 3px;
  --r-md: 7px;
  --r-pill: 3px;
}

* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; }
/* paper grain over the whole page */
body::after { content: ""; position: fixed; inset: 0; z-index: 100; pointer-events: none; opacity: 0.16;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 180 180' xmlns='http://www.w3.org/2000/svg'%3E\
    %3Cfilter id='n'%3E\
    %3CfeTurbulence type='fractalNoise' baseFrequency='.78' numOctaves='4' seed='17' stitchTiles='stitch'/%3E\
    %3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E\
    %3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E\
    "); background-size: 200px 200px; mix-blend-mode: multiply; }
body {
  font-family: var(--sans);
  background: var(--bg); color: var(--fg);
  font-size: 13px; line-height: 1.45;
  font-variant-numeric: tabular-nums;
  -webkit-font-smoothing: antialiased;
}
code, .mono { font-family: var(--mono); font-variant-numeric: tabular-nums; }

/* ---------- header: the page title and the board's name ---------- */
.page-head { display: flex; align-items: center; gap: 12px; height: var(--header-h); padding: 0 18px;
  background: var(--bg); box-shadow: 0 1px 0 var(--border); position: sticky; top: 0; z-index: 11; }
.page-head h1 { margin: 0; font: 600 15px/1.2 var(--sans); color: var(--fg); letter-spacing: -0.005em; }
.ph-board { font: 500 12px/1.2 var(--mono); color: var(--fg-3); overflow-wrap: anywhere; }
.ph-board:empty { display: none; }
/* the open episode's name, at the top of its own pane, with its download beside it */
.ep-head { position: sticky; top: -16px; z-index: 5; margin: -16px -16px 14px; padding: 14px 16px 12px;
  display: flex; align-items: center; justify-content: space-between; gap: 14px;
  background: var(--surface); border-bottom: 1px solid var(--border); }
.ep-head-name { display: flex; flex-direction: column; gap: 4px; min-width: 0; }
.ep-head-k { font: 500 10.5px/1 var(--sans); color: var(--fg-3); }
#current-ep { font: 600 17px/1.25 var(--mono); color: var(--fg); overflow-wrap: anywhere; }
#current-ep-raw { font: 500 10.5px/1.3 var(--mono); color: var(--fg-3); overflow-wrap: anywhere; }
#current-ep-raw:empty { display: none; }
/* where the footage comes from: the dataset, its publisher and its license, linked (the licenses ask for it) */
#current-ep-src { font: 500 11px/1.35 var(--sans); color: var(--fg-3); }
#current-ep-src:empty { display: none; }
#current-ep-src a { color: var(--fg-2); text-decoration: none; border-bottom: 1px solid var(--border-strong); }
#current-ep-src a:hover { color: var(--fg); border-bottom-color: var(--fg-3); }
#current-ep-reader { font: 500 11px/1.35 var(--sans); color: var(--fg-3); }
#current-ep-reader:empty { display: none; }
/* with the reader's line (which opens to a tall list) under the name, the buttons stay at the top, beside the name,
   instead of sliding down the header's centre as the list opens */
.ep-head:has(#current-ep-reader:not(:empty)) { align-items: flex-start; }
/* The list of what the model was not shown is the board's own fold (.pub-fold). Opened, it scrolls inside the header
   so it never pushes the player off screen. */
#current-ep-reader .rn-note { overflow-wrap: anywhere; }
#current-ep-reader .pub-show { font-size: 11px; padding: 2px 0; text-align: left; }
#current-ep-reader .rn-body { max-height: 40vh; overflow: auto; padding: 4px 0 2px; overflow-wrap: anywhere; }
#current-ep-reader .rn-k { font-weight: 600; color: var(--fg-2); padding: 6px 0 2px; }
#current-ep-reader .rn-k:first-child { padding-top: 0; }
#current-ep-reader .rn-i { font: 500 10.5px/1.4 var(--mono); padding: 1px 0; }
.kp-lic a, .kp-note-in a { white-space: nowrap; color: inherit; text-decoration: underline;
  text-decoration-color: var(--border-strong); text-underline-offset: 2px; }
.ep-head-dl { flex: none; font: 500 12px/1 var(--sans); color: var(--fg-2); text-decoration: none; white-space: nowrap;
  padding: 8px 12px; border: 1px solid var(--border-strong); border-radius: var(--r-md); }
.ep-head-dl:hover { color: var(--fg); border-color: var(--fg-3); }
/* the open episode's buttons: its download, and on head-camera episodes the hand pose switch beside it */
.ep-head-acts { flex: none; display: flex; align-items: center; gap: 8px; }
/* the open episode's video to download: the whole episode or a span of it with every camera in one frame, or one
   camera on its own. The menu opens under its button, right-aligned to it */
.vd { position: relative; flex: none; }
.vd[hidden] { display: none; }
.vd-btn { display: inline-flex; align-items: center; gap: 7px; background: transparent; cursor: pointer; }
.vd-caret { flex: none; transition: transform 160ms ease; }
.vd.open .vd-caret { transform: rotate(180deg); }
.vd-menu { position: absolute; top: calc(100% + 6px); right: 0; z-index: 30; width: min(320px, calc(100vw - 32px));
  padding: 6px 0; background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-md);
  box-shadow: 0 10px 34px rgba(0,0,0,0.30); opacity: 0; transform: translateY(-4px); pointer-events: none;
  transition: opacity 120ms ease, transform 120ms ease; }
.vd.open .vd-menu { opacity: 1; transform: translateY(0); pointer-events: auto; }
.vd-opt { display: block; width: 100%; padding: 8px 14px; cursor: pointer; border: 0; background: none; text-align: left;
  font: 500 13px/1.35 var(--sans); color: var(--fg); text-decoration: none; }
.vd-opt:hover { background: rgba(28,28,26,0.06); }
.vd-opt:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
.vd-opt small, .vd-span small { display: block; margin-top: 1px; font: 400 11px/1.35 var(--sans); color: var(--fg-3); }
.vd-opt[aria-disabled="true"] { cursor: default; color: var(--fg-3); background: none; }
.vd-span { padding: 8px 14px 10px; border-top: 1px solid var(--border-strong); margin-top: 4px;
  font: 500 13px/1.35 var(--sans); color: var(--fg); }
.vd-span-row { display: flex; align-items: center; gap: 6px; margin-top: 7px; font: 400 12px/1 var(--sans);
  color: var(--fg-3); }
.vd-span-row input { width: 64px; padding: 6px 7px; border: 1px solid var(--border-strong); border-radius: var(--r-md);
  background: var(--surface); color: var(--fg); font: 500 12px/1 var(--mono); }
.vd-span-row input:focus-visible { outline: 2px solid var(--accent); outline-offset: -1px; }
.vd-span-row .vd-go { margin-left: auto; padding: 6px 10px; border: 1px solid var(--border-strong);
  border-radius: var(--r-md); background: transparent; cursor: pointer; font: 500 12px/1 var(--sans); color: var(--fg-2); }
.vd-span-row .vd-go:hover { color: var(--fg); border-color: var(--fg-3); }
.vd-cams { padding: 8px 14px 2px; border-top: 1px solid var(--border-strong); margin-top: 4px;
  font: 500 12px/1.3 var(--sans); color: var(--fg-3); }
.vd-cam-row { display: flex; flex-wrap: wrap; gap: 6px; padding: 6px 14px 6px; }
.vd-cam-row a { padding: 6px 10px; border: 1px solid var(--border-strong); border-radius: var(--r-md);
  font: 500 12px/1 var(--sans); color: var(--fg-2); text-decoration: none; }
.vd-cam-row a:hover { color: var(--fg); border-color: var(--fg-3); }
.vd-status { padding: 4px 14px 6px; font: 400 11.5px/1.4 var(--sans); color: var(--fg-2); }
.vd-status:empty { display: none; }
.hp-btn { display: inline-flex; align-items: center; gap: 8px; background: transparent; cursor: pointer;
  transition: opacity 200ms ease, color 160ms ease, border-color 160ms ease; }
.hp-btn[hidden] { display: none; }
.hp-sw { position: relative; flex: none; width: 22px; height: 12px; border-radius: 6px; box-sizing: border-box;
  border: 1px solid var(--border-strong); background: transparent; transition: background-color 160ms ease,
  border-color 160ms ease; }
.hp-sw::after { content: ""; position: absolute; top: 1px; left: 1px; width: 8px; height: 8px; border-radius: 50%;
  background: var(--fg-3); transition: transform 160ms ease, background-color 160ms ease; }
.hp-btn[aria-pressed="true"] { color: var(--fg); border-color: var(--fg-3); }
.hp-btn[aria-pressed="true"] .hp-sw { background: var(--fg); border-color: var(--fg); }
.hp-btn[aria-pressed="true"] .hp-sw::after { transform: translateX(10px); background: var(--surface); }
/* the hand pose, drawn on a canvas laid exactly over the displayed image (placed in JS). It sits above the video and
   below every note on it; while the player's own controls show, the strip they cover is dimmed so their text stays
   readable. */
@property --hp-band { syntax: "<number>"; inherits: false; initial-value: 1; }
.hp-canvas { position: absolute; left: 0; top: 0; width: 0; height: 0; pointer-events: none; opacity: 0;
  --hp-band: 1; transition: opacity 220ms ease, --hp-band 200ms ease;
  -webkit-mask-image: linear-gradient(to top, rgba(0,0,0,var(--hp-band)) var(--hp-bar, 0px),
    #000 calc(var(--hp-bar, 0px) + 12px));
  mask-image: linear-gradient(to top, rgba(0,0,0,var(--hp-band)) var(--hp-bar, 0px),
    #000 calc(var(--hp-bar, 0px) + 12px));
  }
.hp-canvas.on { opacity: 1; }
.hp-canvas.ctl { --hp-band: 0.18; }
/* ---------- the rail's controls ----------
   Top to bottom: whose labels the whole board shows (Labels by), then the dataset the list below holds, with its
   search and its issue filter, then the list, and under it the list's downloads. Every control is the same height, border and
   type, a small plain label names a control only where its own text does not, and teal marks what is chosen. */
.rail-k { display: block; margin: 0 0 6px; font: 500 12px/1.3 var(--sans); color: var(--fg-3); }
.rail-ds { flex: 0 0 auto; margin: 0 0 12px; }
.rd-name { display: block; font: 600 15px/1.25 var(--sans); color: var(--fg); letter-spacing: -0.005em;
  overflow-wrap: anywhere; }
.rd-sub { display: block; margin-top: 3px; font: 500 11.5px/1.3 var(--mono); color: var(--fg-3);
  font-variant-numeric: tabular-nums; }
.rd-sub em { font-style: normal; color: var(--fg); }
.dl-row { display: flex; flex-wrap: wrap; gap: 8px; }
.rail-export { flex: 0 0 auto; height: 32px; padding: 0 11px; cursor: pointer; white-space: nowrap;
  font: 500 12.5px/1 var(--sans); color: var(--fg-2); background: var(--surface);
  border: 1px solid var(--border-strong); border-radius: var(--r-md);
  transition: color 120ms ease, border-color 120ms ease; }
.rail-export:hover { color: var(--fg); border-color: var(--fg-3); }
.rail-export:disabled { cursor: default; color: var(--fg-3); }
.rail-export small { font: 500 11px/1 var(--mono); color: var(--fg-3); margin-left: 6px; }
.lb-btn .lb-sub { display: block; margin-top: 2px; font: 400 11.5px/1.3 var(--sans); color: inherit; opacity: 0.8; }
.cmpv-back { display: inline-flex; align-items: center; gap: 7px; margin: 0 0 18px; padding: 7px 12px; cursor: pointer;
  font: 500 12.5px/1 var(--sans); color: var(--fg-2); background: transparent; border: 1px solid var(--border-strong);
  border-radius: var(--r-md); }
.cmpv-back:hover { color: var(--fg); border-color: var(--fg-3); }
/* the downloads under the list: they fold away (eased) while a comparison is shown, since they are the board's */
.rail-dl, .kp-exp, .lb-note, .kp-note { display: grid; grid-template-rows: 1fr;
  transition: grid-template-rows 220ms ease, opacity 200ms ease, margin 220ms ease; }
/* the downloads of the list sit under it, at the foot of the rail */
/* the rail's two dividers (under "Labels by", over the downloads) run the rail's full width, to its border */
.rail-dl { flex: 0 0 auto; margin: 12px calc(-1 * var(--rail-pad)) 0; }
.rail-dl.off { margin-top: 0; }
.dl-box { padding: 12px var(--rail-pad) 0; border-top: 1px solid var(--border-strong); }
.rail-dl-in, .kp-exp-in, .lb-note-in, .kp-note-in { min-height: 0; overflow: hidden; }
.rail-dl.off, .kp-exp.off, .lb-note.off, .kp-note.off { grid-template-rows: 0fr; opacity: 0; }
.kp-exp { flex: 0 0 auto; }
.kp-lic, .dl-lic { margin: 8px 2px 0; font: 400 11px/1.4 var(--sans); color: var(--fg-3); }
.dl-lic a { color: inherit; text-underline-offset: 2px; }
/* whose labels the board shows: its own, or one other model's as a comparison. A comparison turns the button ink,
   the board's mark for labels that are not its own and are never counted */
.lb { flex: 0 0 auto; position: relative; margin: 0 calc(-1 * var(--rail-pad)) 12px; padding: 0 var(--rail-pad) 12px;
  border-bottom: 1px solid var(--border-strong); }
.lb[hidden] { display: none; }
.lb-btn { width: 100%; display: flex; align-items: center; gap: 9px; min-height: 40px; padding: 9px 12px;
  cursor: pointer; text-align: left; background: var(--surface); color: var(--fg); border: 1px solid var(--border-strong);
  border-radius: var(--r-md); font: 600 14px/1.25 var(--sans);
  transition: background-color 220ms ease, color 220ms ease, border-color 220ms ease; }
.lb-btn:hover { border-color: var(--fg-2); }
.lb-lab { flex: 1; min-width: 0; overflow-wrap: anywhere; }
.lb-caret { flex: none; transition: transform 160ms ease; }
.lb.open .lb-caret { transform: rotate(180deg); }
.lb.cmp .lb-btn { background: var(--fg); color: #f3f2ec; border-color: var(--fg); }
.lb.cmp .lb-btn:hover { background: #2e2e2b; }
.lb-note-in { padding: 7px 2px 0; font: 400 11.5px/1.4 var(--sans); color: var(--fg-2); }
.lb-note-in b { font-weight: 600; color: var(--fg); }
.lb-menu { position: fixed; left: var(--lb-x, 8px); top: var(--lb-y, 0px); width: var(--lb-w, 340px);
  max-height: var(--lb-h, 74vh); z-index: 60; overflow-y: auto; background: var(--surface);
  border: 1px solid var(--border); border-radius: var(--r-md); box-shadow: 0 10px 34px rgba(0,0,0,0.30); opacity: 0;
  transform: translateY(-4px); pointer-events: none; transition: opacity 120ms ease, transform 120ms ease; }
.lb.open .lb-menu { opacity: 1; transform: translateY(0); pointer-events: auto; }
.lb-group { padding: 10px 14px 4px; font: 500 12px/1.3 var(--sans); color: var(--fg-3); }
.lb-group + .lb-gnote { margin-top: -1px; }
.lb-gnote { padding: 0 14px 6px; font: 400 11px/1.4 var(--sans); color: var(--fg-3); }
.lb-sep { border-top: 1px solid var(--border-strong); margin-top: 4px; }
.lb-opt { position: relative; display: flex; align-items: baseline; gap: 12px; width: 100%; padding: 8px 14px 8px 32px;
  cursor: pointer; border: 0; background: none; text-align: left; font: 500 13px/1.35 var(--sans); color: var(--fg-2); }
.lb-opt:hover { background: rgba(28,28,26,0.06); color: var(--fg); }
.lb-opt:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
.lb-opt.on { color: var(--fg); font-weight: 600; }
.lb-opt.on::before { content: "\2713"; position: absolute; left: 13px; font-weight: 700; }
.lb-opt .lb-o-name { flex: 1; min-width: 0; }
.lb-opt small { display: block; font: 400 11px/1.35 var(--sans); color: var(--fg-3); margin-top: 1px; }
.lb-opt .lb-o-n { flex: none; font: 500 12px/1.35 var(--mono); color: var(--fg-3); }
.lb-go { display: flex; align-items: center; justify-content: space-between; gap: 12px; width: 100%;
  padding: 11px 14px 12px 32px; border: 0; border-top: 1px solid var(--border-strong); margin-top: 4px; background: none;
  cursor: pointer; text-align: left; font: 600 13px/1.35 var(--sans); color: var(--accent); }
.lb-go small { display: block; font: 400 11px/1.35 var(--sans); color: var(--fg-3); margin-top: 1px; }
.lb-go:hover { background: rgba(69,129,142,0.08); }
.lb-go:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
.lb-go .lb-arrow { flex: none; font-size: 15px; }
/* switching whose labels the board shows: the list, the filter and the strip's numbers ease out and back in */
#ep-list, .issue-filter, .coverage .cv-num, .coverage .cv-fig { transition: opacity 160ms ease; }
body.lb-swap #ep-list, body.lb-swap .issue-filter, body.lb-swap .coverage .cv-num,
  body.lb-swap .coverage .cv-fig { opacity: 0; }
/* the line in place of the label sections of an episode whose reply gave no labels */
.no-labels { margin: 18px 0 6px; padding: 12px 14px; border: 1px dashed var(--border-strong);
  border-radius: var(--r-md);
  font-size: 13px; line-height: 1.5; color: var(--fg-2); }
.ep-card .outcome-tag.fail { color: var(--fg-2); background: transparent; border: 1px dashed var(--border-strong); }
/* the header's buttons, and on head-camera episodes the keypoints' licence under them, never wider than the buttons
   (width 0, min-width 100%), so the episode's name keeps its room */
.ep-head-side { flex: none; display: flex; flex-direction: column; align-items: flex-end; }
.kp-note { width: 0; min-width: 100%; }
.kp-note-in { padding-top: 7px; text-align: right; font: 400 11px/1.4 var(--sans); color: var(--fg-3); }
#kp-dl[hidden] { display: none; }

/* ---------- main grid ---------- */
main {
  display: grid;
  grid-template-columns: var(--rail-w) minmax(640px, 56vw) 1fr;
  height: calc(100vh - var(--top-h));
}

/* ---------- episode rail (persistent picker) ---------- */
aside.rail {
  background: var(--bg); border-right: 1px solid var(--border);
  --rail-pad: 10px; padding: 12px var(--rail-pad);
  position: sticky; top: var(--top-h); height: calc(100vh - var(--top-h));
  /* sticky makes the rail its own stacking context; it sits above the main column so the filter menu, which is
     wider than the rail, is drawn over the video instead of under it */
  z-index: 20;
  /* the rail itself does not scroll; the dropdown stays pinned and only the
     episode list below it scrolls, so switching datasets never requires
     scrolling back up to reach the picker. */
  display: flex; flex-direction: column; overflow: hidden;
}
/* the scrolling episode list; flex child needs min-height:0 to actually scroll */
#ep-list { flex: 1 1 auto; min-height: 0; overflow-y: auto; }
/* data-issue filter: one compact dropdown (current filter + its episode count), whose menu lists
   every filter with its count. It never grows with the number of issue classes, so it cannot
   push into or overlap the episode list. */
.issue-filter { flex: 0 0 auto; position: relative; margin: 0 0 12px; }
/* episode ID search: narrows the rail as you type; every space-separated term must appear in the
   raw ID or the display name, so "1424", "rice 394" or a uuid prefix all work. Enter opens the
   first match. It sits with the other rail controls and never scrolls away. */
.ep-search { flex: 0 0 auto; position: relative; margin: 0 0 8px; }
.ep-search input { width: 100%; box-sizing: border-box; height: 40px; padding: 0 30px 0 34px; background: var(--surface);
  color: var(--fg); border: 1px solid var(--border-strong); border-radius: var(--r-md); font: 500 14px/1.2 var(--sans);
  outline: none;
  transition: border-color 120ms, background 120ms; }
.ep-search input::placeholder { color: var(--fg-3); }
.ep-search input:focus { border-color: rgba(69,129,142,0.8); background: rgba(69,129,142,0.08); }
.ep-search .es-ico { position: absolute; left: 11px; top: 50%; transform: translateY(-50%); color: var(--fg-3);
  pointer-events: none; }
.ep-search .es-clear { position: absolute; right: 8px; top: 50%; transform: translateY(-50%); width: 20px; height: 20px;
  border-radius: 50%; display: none; place-items: center; background: rgba(255,255,255,0.10); color: var(--fg);
  font-size: 13px; line-height: 1; cursor: pointer; border: 0; padding: 0; }
.ep-search.has-q .es-clear { display: grid; }
.ep-search .es-clear:hover { background: rgba(255,255,255,0.22); }
.rail-empty .es-jump { color: var(--accent); cursor: pointer; text-decoration: underline; }
.issue-filter:empty { display: none; }
/* the issue filter: a control like the search above it, reading "Filter by issue" until a problem is chosen, then
   naming it in teal, the colour of what is chosen */
.if-btn { width: 100%; display: flex; align-items: center; gap: 9px; min-height: 40px; padding: 9px 12px;
  background: var(--surface); color: var(--fg); border: 1px solid var(--border-strong);
  border-radius: var(--r-md); font: 600 14px/1.25 var(--sans); cursor: pointer; text-align: left;
  transition: background 120ms, border-color 120ms; }
.if-btn:hover { border-color: var(--fg-3); }
.if-btn .if-ico { color: var(--fg-3); flex: 0 0 auto; }
.if-btn .if-lab { flex: 1; min-width: 0; overflow-wrap: anywhere; }
.if-btn .if-ph { font-weight: 500; color: var(--fg-3); }
.if-btn .if-n { flex: 0 0 auto; font: 600 11.5px/1 var(--mono); color: var(--accent); padding: 4px 7px;
  border-radius: var(--r-pill); background: rgba(69,129,142,0.14); font-variant-numeric: tabular-nums; }
.issue-filter.filtered .if-btn { background: rgba(69,129,142,0.08); border-color: var(--accent); }
.issue-filter.filtered .if-ico, .issue-filter.filtered .if-lab { color: var(--accent); }
.if-clear { flex: 0 0 auto; width: 22px; height: 22px; border-radius: 50%; display: grid; place-items: center;
  background: rgba(28,28,26,0.090); color: var(--fg); font-size: 14px; line-height: 1; }
.if-clear:hover { background: rgba(28,28,26,0.198); }
.if-n { color: var(--fg-3); font-variant-numeric: tabular-nums; font-size: 12px; }
.if-menu { position: absolute; top: calc(100% + 4px); left: 0; right: 0; z-index: 40;
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--r-md);
  box-shadow: 0 10px 34px rgba(0,0,0,0.55); padding: 4px; max-height: 62vh; overflow-y: auto;
  opacity: 0; transform: translateY(-4px); pointer-events: none; transition: opacity 120ms ease, transform 120ms ease; }
.issue-filter.open .if-menu { opacity: 1; transform: translateY(0); pointer-events: auto; }
/* the menu is wider than the rail, which clips its children (overflow: hidden), so it opens as a fixed layer
   placed from the button by placeIssueMenu() and kept inside the viewport */
.if-menu { padding: 0; position: fixed; left: var(--if-x, 8px); top: var(--if-y, 0px); right: auto;
  width: var(--if-w, 340px); max-height: var(--if-h, 74vh); z-index: 60; }
/* one block per group: a header band in the group's colour, then its rows */
.if-sev { display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 9px 12px 9px 15px;
  border-bottom: 1px solid var(--border-strong); background: var(--surface); }
.if-sev-k { font: 500 12px/1.3 var(--sans); color: var(--fg-3); }
.if-sev-seg { display: inline-flex; border: 1px solid var(--border-strong); border-radius: 999px; padding: 2px;
  gap: 2px; }
.if-sev-seg button { border: 0; background: none; color: var(--fg-2); font: 500 12px/1 var(--sans); padding: 6px 10px;
  border-radius: 999px; cursor: pointer; transition: background .18s, color .18s; white-space: nowrap; }
.if-sev-seg button:hover { color: var(--fg); }
.if-sev-seg button[aria-checked="true"] { background: rgba(36,36,31,0.9); color: #fff; }
.if-group { --g: var(--fg-3); --gbg: rgba(255,255,255,0.04); border-top: 1px solid var(--border-strong); }
.if-group:first-child { border-top: 0; }
.if-group.g-data { --g: var(--danger); --gbg: rgba(179,38,60,0.07); }
.if-group.g-mistake { --g: var(--warning); --gbg: rgba(77,90,160,0.10); }
.if-ghead { position: sticky; top: 0; z-index: 2; display: flex; align-items: baseline; justify-content: space-between;
  gap: 8px;
  padding: 10px 12px 6px; background: color-mix(in srgb, var(--surface) 88%, var(--g));
  border-left: 3px solid var(--g); }
.if-group.g-overview .if-ghead { border-left-color: transparent; background: var(--surface); }
.if-gtitle { font: 700 12px/1.2 var(--sans); color: var(--g); }
.if-group.g-overview .if-gtitle { color: var(--fg-2); }
.if-gcount { font: 500 10.5px/1 var(--mono); color: var(--fg-3); white-space: nowrap; }
.if-gnote { padding: 0 12px 7px 15px; font: 400 11px/1.35 var(--sans); color: var(--fg-3);
  background: color-mix(in srgb, var(--surface) 88%, var(--g)); border-left: 3px solid var(--g); }
.if-row { display: grid; grid-template-columns: minmax(0, 1fr) 46px 40px; align-items: center; gap: 10px;
  padding: 7px 12px 7px 15px;
  cursor: pointer; font-size: 13px; color: var(--fg-2); border-left: 3px solid transparent; }
.if-group:not(.g-overview) .if-row { border-left-color: color-mix(in srgb, var(--g) 35%, transparent); }
.if-row .if-name { min-width: 0; overflow-wrap: break-word; display: flex; align-items: center; gap: 8px; }
.if-row .if-name:has(.if-sub) { flex-wrap: wrap; row-gap: 2px; }
.if-sub { flex-basis: 100%; font: 500 11px/1.3 var(--mono); color: var(--fg-3); }
.if-row .if-n { text-align: right; }
.if-track { height: 4px; border-radius: 2px; background: rgba(28,28,26,0.054); overflow: hidden; }
.if-group.g-overview .if-track { background: none; }
.if-track i { display: block; height: 100%; border-radius: 2px; background: var(--g); }
.if-dot { flex: none; width: 7px; height: 7px; border-radius: 2px; }
.if-dot.d-data { background: var(--danger); }
.if-dot.d-mistake { background: #4d5aa0; }
.if-row:hover { background: var(--gbg); color: var(--fg); }
.if-row.active { background: rgba(36,36,31,0.9); color: #fff; border-left-color: #fff; }
.if-row.active .if-n { color: rgba(255,255,255,0.9); }
.if-row.active .if-track i { background: #fff; }
/* what is on this board, at a glance: the board's totals, then one tab per dataset with its episodes and hours. The
   open dataset's tab is lit, and when a filter or search narrows the list it says how much of the dataset is listed. */
.coverage { position: sticky; top: var(--header-h); z-index: 9; height: var(--cov-h); box-sizing: border-box;
  display: grid; grid-template-columns: calc(var(--rail-w) + 1px) 1fr; align-items: stretch;
  background: var(--bg); }
/* the strip's bottom line, 1px above its edge so the open tab's 3px underline and the totals' 3px rule each sit
   centred on a 1px line: the underline on this one, the rule on the rail's border below it */
.coverage::after { content: ""; position: absolute; left: 0; right: 0; bottom: 1px; height: 1px;
  background: var(--border-strong); pointer-events: none; }
/* board totals: their own segment over the rail, set apart from the per-dataset tabs by a heavy rule centred on the
   rail's border (the segment is 1px wider than the rail) */
.cv-all { box-sizing: border-box; min-width: 0; display: flex; flex-direction: column; justify-content: center; gap: 7px; padding: 0 24px 0 10px;
  background: var(--surface); border-right: 3px solid var(--border-strong); }
.cv-all .cv-k { font: 600 11px/1 var(--sans); color: var(--fg-2); letter-spacing: 0.01em; white-space: nowrap; }
.cv-all .cv-fig { display: flex; align-items: baseline; gap: 14px; white-space: nowrap; }
.cv-all .cv-fig b { font: 600 18px/1 var(--mono); color: var(--fg); letter-spacing: -0.02em;
  font-variant-numeric: tabular-nums; }
.cv-all .cv-fig small { font: 500 11px/1 var(--sans); color: var(--fg-3); margin-left: 4px; }
/* one tab per dataset: click to open it */
/* each tab at least as wide as its text; more tabs than fit scroll sideways instead of truncating */
.cv-cells { display: grid; grid-auto-flow: column; grid-auto-columns: minmax(max-content, 1fr); min-width: 0;
  overflow-x: auto; scrollbar-width: thin; }
.cv-cell { display: flex; flex-direction: column; justify-content: center; gap: 7px; padding: 0 16px; min-width: 0;
  cursor: pointer; position: relative; transition: background 140ms; }
.cv-cell:hover { background: rgba(28,28,26,0.050); }
.cv-cell.on { background: rgba(69,129,142,0.08); }
.cv-cell.on::after { content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 2px; z-index: 1;
  background: var(--accent); }
.cv-name { font: 500 12.5px/1 var(--sans); color: var(--fg-3); white-space: nowrap; }
.cv-cell:hover .cv-name { color: var(--fg-2); }
.cv-cell.on .cv-name { color: var(--fg); font-weight: 600; }
.cv-num { font: 500 11.5px/1 var(--mono); color: var(--fg-3); white-space: nowrap;
  font-variant-numeric: tabular-nums; }
.cv-cell.on .cv-num { color: var(--fg-2); }
.cv-num em { font-style: normal; color: var(--accent); }
/* a dataset the chosen model did not label: dimmed and not clickable */
.cv-cell.none { cursor: default; }
.cv-cell.none .cv-name, .cv-cell.none .cv-num, .cv-cell.none:hover .cv-name { color: var(--fg-disabled); }
.cv-cell.none:hover { background: none; }
/* all nine tabs fit from 1600 px down to 1280 px (at 1440 px the last one ran past the edge) */
@media (max-width: 1600px) { .cv-cell { padding: 0 10px; } .cv-num { font-size: 10.5px; } .cv-all {
  padding: 0 16px 0 10px; } }
@media (max-width: 1280px) { .cv-num .u { display: none; } .cv-cell { padding: 0 11px; } .cv-all {
  padding: 0 16px 0 10px; } }
@media (max-width: 1200px) { .cv-num { font-size: 10.5px; } .cv-name { font-size: 12px; } .cv-cell { padding: 0 8px; }
  .cv-all .cv-fig b { font-size: 16px; } }
.rail-empty {
  padding: 18px 10px; color: var(--fg-3); font-size: 12.5px; line-height: 1.5;
}
.rail-empty b { font-weight: 600; color: var(--fg-2); }
.ep-card {
  display: block; padding: 10px 12px; margin-bottom: 4px;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r-md); cursor: pointer; transition: all 100ms;
  /* a card off screen is not laid out or painted; its last size holds its place in the scroll */
  content-visibility: auto; contain-intrinsic-size: auto 118px;
}
.ep-card:hover {
  background: var(--raised); border-color: var(--border-strong);
}
.ep-card.active {
  background: rgba(36,36,31,0.06);
  border-color: rgba(36,36,31,0.40);
}
.ep-card .id {
  font-family: var(--mono); font-size: 10px;
  color: var(--fg-faint); margin-bottom: 5px;
  overflow-wrap: anywhere;
}
.ep-card .preview {
  font-size: 12px; color: var(--fg); line-height: 1.4;
  margin-bottom: 8px; overflow-wrap: anywhere;
}
.ep-card .row {
  display: flex; flex-wrap: wrap; align-items: center; gap: 6px;
  font-family: var(--mono); font-size: 10px;
}
/* chips keep their text on one line and wrap as whole chips */
.ep-card .row > * { white-space: nowrap; flex: none; }
.ep-card .row .outcome-tag {
  padding: 1px 8px; border-radius: var(--r-pill);
  border: 1px solid transparent; font-weight: 700; letter-spacing: 0.02em;
  text-transform: lowercase;
}
.ep-card .row .outcome-tag.success { color: var(--success); background: rgba(78,194,127,0.14);
  border-color: rgba(78,194,127,0.30); }
.ep-card .row .outcome-tag.partial { color: var(--fg-2); background: rgba(28,28,26,0.050);
  border-color: var(--border); }
.ep-card .row .outcome-tag.success_then_undone { color: var(--danger); background: rgba(179,38,60,0.07);
  border-color: rgba(179,38,60,0.40); border-style: dashed; }
.ep-card .row .outcome-tag.failure { color: var(--danger);  background: rgba(179,38,60,0.14);
  border-color: rgba(179,38,60,0.30); }
.ep-card .row .outcome-tag.unclear,
.ep-card .row .outcome-tag.none { color: var(--fg-3); background: rgba(28,28,26,0.050); border-color: var(--border); }
/* data-issue chip in the rail preview, sits to the right of the outcome tag */
.ep-card .row .issue-tag {
  padding: 1px 8px; border-radius: var(--r-pill); font-weight: 700;
  letter-spacing: 0.02em; display: inline-flex; align-items: center; gap: 4px;
  border: 1px solid transparent;
}
.ep-card .row .issue-tag.high { color: var(--danger); background: rgba(179,38,60,0.16);
  border-color: rgba(179,38,60,0.45); }
.ep-card .row .issue-tag.medium { color: var(--danger); background: rgba(179,38,60,0.07);
  border-color: rgba(179,38,60,0.28); }
.ep-card .row .issue-tag.low { color: var(--fg-2); background: rgba(28,28,26,0.054); border-color: var(--border); }
/* operator-mistake chip: a different hue from the data-fault chip, since the recording itself is fine */
.ep-card .row .issue-tag.minor { color: var(--fg-3); background: transparent; border: 1px dashed var(--border-strong);
  font-weight: 500; }
.ep-card .row .issue-tag.op { color: var(--warning); background: rgba(77,90,160,0.12);
  border-color: rgba(77,90,160,0.38); font-weight: 600; }
/* one-line summary of the top data issue, so the rail says WHAT is wrong */
.ep-card .issue-note {
  margin-top: 5px; font-size: 10.5px; line-height: 1.35; overflow-wrap: anywhere;
  border-left: 2px solid transparent; padding-left: 7px;
}
.ep-card .issue-note.high { color: var(--danger); border-left-color: rgba(179,38,60,0.6); }
.ep-card .issue-note.medium { color: var(--danger); border-left-color: rgba(179,38,60,0.3); }
.ep-card .issue-note.low { color: var(--fg-3); border-left-color: var(--border); }

aside.left {
  background: var(--surface); border-right: 1px solid var(--border);
  padding: 16px; overflow-y: auto;
  position: sticky; top: var(--top-h); height: calc(100vh - var(--top-h));
}
aside.left .video-wrap {
  position: relative; width: 100%; background: #000;
  border-radius: var(--r-md); overflow: hidden;
  /* its own stacking context: an overlay's z-index (the fullscreen button, the toasts) ranks only inside the video,
     so nothing in it paints over the sticky episode header when the video scrolls under it */
  isolation: isolate;
}
/* Cam layout: big exo on top (full width), wrist L+R side-by-side below.
   The exo view is the most useful (third-person workspace); the wrist
   cams are reference views. Bigger video player means the scrubber bar
   is actually draggable. */
aside.left .cam-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: auto auto;
  gap: 4px;
  background: #000;
  width: 100%;
}
aside.left .cam-cell {
  position: relative;
  background: #000;
  min-width: 0;
  display: flex;
  align-items: center;
  justify-content: center;
}
aside.left .cam-cell.cam-exo {
  grid-column: 1 / span 2;  /* exo full width on top row */
}
aside.left .cam-cell video {
  display: block;
  width: 100%;
  height: auto;
  background: #000;
  object-fit: contain;
}
aside.left .cam-cell.cam-exo video {
  max-height: 55vh;
}
/* a head camera is a single panel with no wrist row, so it takes more height */
aside.left .cam-row-single .cam-cell.cam-exo video {
  max-height: 74vh;
}
aside.left .cam-cell.cam-wrist video {
  max-height: 28vh;
}
/* Gripper-only rigs (FastUMI): no fixed camera, so the gripper views sit side by side at equal size
   (one gripper: a single panel across the row). The recovery banner, action caption and state note
   render in .grip-strip under the whole row, centered, never over the image; caption and state keep
   a fixed height so plain playback never moves the page. The progress chip stays on the image. */
aside.left .cam-row-grippers .cam-cell.cam-exo { grid-column: auto; }
aside.left .cam-row-grippers.cam-row-single .cam-cell.cam-exo { grid-column: 1 / span 2; }
aside.left .cam-row-grippers .cam-cell video,
aside.left .cam-row-grippers .cam-cell.cam-wrist video { max-height: 50vh; }
aside.left .cam-row-grippers.cam-row-single .cam-cell.cam-exo video { max-height: 60vh; }
aside.left .cam-row-grippers .prog-overlay { top: calc(var(--fx-top, 0px) + 34px);
  left: calc(var(--fx-left, 0px) + 8px); }
aside.left .grip-strip { display: flex; flex-direction: column; align-items: center; gap: 8px; padding: 10px 0 12px;
  background: #000; }
aside.left .grip-strip .grip-note { order: 0; margin: 0; padding: 0 12px; text-align: center; font-family: var(--mono);
  font-size: 10px; color: rgba(255,255,255,0.62); text-wrap: balance; }
aside.left .grip-strip .recovery-overlay { position: static; order: 1; width: 92%; max-width: 92%; display: none; }
aside.left .grip-strip .recovery-overlay.active { display: block; }
aside.left .grip-strip .video-overlay,
aside.left .grip-strip .video-overlay.active { position: static; order: 2; transform: none; min-width: 0;
  max-width: 96%;
  padding: 11px 18px; font-size: 15px; text-align: center; box-sizing: border-box; min-height: 76px; }
aside.left .grip-strip .state-toast,
aside.left .grip-strip .state-toast.active { position: static; order: 3; transform: none; max-width: 92%;
  min-width: 40%;
  font-size: 14px; text-align: center; box-sizing: border-box; min-height: 70px; }
/* Fullscreen the exo CELL (video + overlays), not the bare <video>, so the
   overlays stay visible. Hide the native fullscreen button (it promotes only the
   video) and use the custom .fs-btn, which fullscreens the cell. */
#video::-webkit-media-controls-fullscreen-button, #video-wl::-webkit-media-controls-fullscreen-button,
  #video-wr::-webkit-media-controls-fullscreen-button { display: none; }
.cam-exo .fs-btn {
  position: absolute; top: calc(var(--fx-top, 0px) + 8px); right: calc(var(--fx-right, 0px) + 8px); z-index: 5;
  width: 26px; height: 26px; padding: 0; cursor: pointer;
  border: 1px solid rgba(255,255,255,0.18); border-radius: 5px;
  background: rgba(0,0,0,0.55); color: #fff; font-size: 15px; line-height: 1;
  display: flex; align-items: center; justify-content: center;
}
.cam-exo .fs-btn:hover { background: rgba(0,0,0,0.82); }
.cam-cell.cam-exo:fullscreen { background: #000; width: 100vw; height: 100vh; }
.cam-cell.cam-exo:fullscreen video, aside.left .cam-cell.cam-exo:fullscreen video,
  aside.left .cam-row-single .cam-cell.cam-exo:fullscreen video { max-height: 100vh; width: 100%; height: 100%;
  object-fit: contain; }
.cam-cell.cam-exo:-webkit-full-screen { background: #000; width: 100vw; height: 100vh; }
.cam-cell.cam-exo:-webkit-full-screen video { max-height: 100vh; width: 100%; height: 100%; object-fit: contain; }
/* each camera's label sits 8px in from the image's corner, the same line as the chips over the main camera's image */
/* the line under the cameras that names the ones the model was not shown, and why */
aside.left .unshown-note { margin: 0; padding: 6px 12px 8px; background: #000; font-family: var(--mono);
  font-size: 10px; line-height: 1.45; color: rgba(255,255,255,0.72); }
/* its label wraps inside a narrow cell instead of running past it */
aside.left .cam-unshown .cam-label { white-space: normal; max-width: calc(100% - 28px); border-radius: var(--r-sm); }
aside.left .cam-label {
  position: absolute; top: 8px; left: 8px; z-index: 2;
  padding: 1px 6px;
  font-family: var(--mono); font-size: 10px; white-space: nowrap;
  color: rgba(255,255,255,0.92); background: rgba(0,0,0,0.6);
  border-radius: var(--r-pill); pointer-events: none;
}
aside.left .cam-cell.cam-exo .cam-label { top: calc(var(--fx-top, 0px) + 8px); left: calc(var(--fx-left, 0px) + 8px); }

/* on-video subtitle-style overlay - pinned to the bottom of the EXO cell,
   lifted above the native controls bar; shows the currently-active event
   with a prominent timestamp readout. */
.video-overlay {
  position: absolute; left: 50%; bottom: calc(var(--fx-bottom, 0px) + 44px);
  max-width: 90%; min-width: 240px;
  background: rgba(0, 0, 0, 0.72);
  backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
  border-radius: var(--r-md); border: 1px solid rgba(255, 255, 255, 0.10);
  padding: 9px 14px; text-align: center;
  font-size: 13px; line-height: 1.4; color: #fff;
  opacity: 0; transform: translateX(-50%) translateY(6px);
  transition: opacity 130ms ease, transform 130ms ease;
  pointer-events: none;
  box-shadow: 0 4px 18px rgba(0, 0, 0, 0.45);
}
.video-overlay.active { opacity: 1; transform: translateX(-50%) translateY(0); }
.video-overlay.adv   { border-left: 3px solid rgba(255,255,255,0.55); box-shadow: 0 4px 18px rgba(0,0,0,0.45); }
.video-overlay.waste { border-left: 3px solid #a3aee0; box-shadow: 0 4px 18px rgba(0,0,0,0.45),
  inset 3px 0 12px -4px rgba(163,174,224,0.3); }
.video-overlay.idle  { border-left: 3px solid #8a8a94; box-shadow: 0 4px 18px rgba(0,0,0,0.45); }
.video-overlay.none  { border-left: 3px solid #8a8a94; box-shadow: 0 4px 18px rgba(0,0,0,0.45); }
.video-overlay .vo-time {
  display: inline-block; font-family: var(--mono);
  font-size: 13px; font-weight: 700; color: #fff;
  padding: 1px 8px; border-radius: var(--r-pill);
  background: rgba(0, 0, 0, 0.45);
  margin-right: 8px; vertical-align: 1px;
}
.video-overlay .vo-arm {
  display: inline-block; font-family: var(--mono);
  font-size: 10px; padding: 1px 6px; border-radius: var(--r-pill);
  background: rgba(255, 255, 255, 0.08);
  margin-right: 8px; vertical-align: 1px;
}
.video-overlay .vo-arm.left  { color: #9cc2ff; }
.video-overlay .vo-arm.right { color: #c9a3d6; }
.video-overlay .vo-phrase { color: #fff; font-weight: 500; }
.video-overlay .vo-phrase .dest-chip {
  display: inline-block; margin-left: 2px;
  font-size: 11px; padding: 1px 8px; border-radius: var(--r-pill);
  background: rgba(255, 255, 255, 0.10); color: #e6e8df; font-weight: 400;
}
.video-overlay .vo-phrase .dest-chip .prep { color: rgba(255, 255, 255, 0.72); margin-right: 3px; }
.video-overlay .vo-contrib {
  display: inline-block; margin-left: 8px;
  font-size: 10px; font-family: var(--mono); font-weight: 500;
  padding: 1px 8px; border-radius: var(--r-pill); vertical-align: 1px;
}
.video-overlay .vo-contrib.adv   { background: rgba(255, 255, 255, 0.14); color: #e6e8df; }
.video-overlay .vo-contrib.waste { background: rgba(163, 174, 224, 0.18); color: #bcc2e0; }
.video-overlay .vo-contrib.idle  { background: rgba(255,255,255,0.12); color: #cfcfd6; }
.video-overlay .vo-contrib.none  { background: rgba(255,255,255,0.10); color: #cfcfd6; }
/* head camera, hands out of view: a pulsing ring on the cell and a pill in the status strip below it, so a viewer
   sees at once that the annotation is inferred while the hands are not in shot rather than directly observed. */
.cam-cell.hands-hidden::after {
  content: ""; position: absolute; inset: 0; pointer-events: none; z-index: 6;
  border: 2px solid rgba(255,255,255,0.7); border-radius: inherit;
  box-shadow: inset 0 0 26px -6px rgba(255,255,255,0.35);
  animation: handsPulse 1.6s ease-in-out infinite;
}
@keyframes handsPulse { 0%,100% { opacity: 0.55; } 50% { opacity: 1; } }
/* head camera: one row across the top of the image, with the progress chip at its left, the viewpoint in its middle
   (centred on the image) and room for the full-screen button at its right. The row's three columns never overlap; a
   narrow image drops the chip's sparkline, then its caption, so the row fits at every width. The hand-state chip
   (hands out of view, gloves) fades in under the viewpoint as the playhead moves, and the notes that come and go at
   the top of the image (a state change, a recovery) are placed below the row (placeTop). */
.top-hud { position: absolute; z-index: 4; top: calc(var(--fx-top, 0px) + 8px);
  left: calc(var(--fx-left, 0px) + 8px); right: calc(var(--fx-right, 0px) + 8px); container: hud / inline-size;
  pointer-events: none; }
.top-hud-in { display: grid; grid-template-columns: minmax(max-content, 1fr) auto minmax(26px, 1fr); column-gap: 8px;
  align-items: start; }
.top-hud .prog-overlay { position: static; justify-self: start; }
@container hud (max-width: 480px) { .top-hud .prog-overlay .po-svg { display: none; } }
@container hud (max-width: 330px) { .top-hud .prog-overlay .po-sub { display: none; } }
.ego-status {
  display: flex; flex-direction: column; align-items: center; gap: 6px;
}
.ego-status .es-pov, .ego-status .es-hand {
  display: inline-flex; align-items: center; gap: 7px;
  padding: 4px 10px; border-radius: var(--r-pill);
  font-size: 12px; font-weight: 700; letter-spacing: 0.02em;
}
.ego-status .es-pov {
  background: rgba(20, 22, 18, 0.82); border: 1px solid rgba(255,255,255,0.3); color: #e6e8df; white-space: nowrap;
}
.ego-status .es-pov.exo {
  background: rgba(20, 22, 18, 0.85); border-color: rgba(255,255,255,0.4); color: #f3f2ec;
}
/* what the hands wear can be a long phrase: it wraps inside the row's middle column, never past the image */
.ego-status .es-hand {
  border: 1px solid transparent; text-align: center; max-width: 100%;
  opacity: 0; transform: translateY(4px); transition: opacity 140ms ease, transform 140ms ease;
}
.ego-status .es-hand.show { opacity: 1; transform: translateY(0); }
.ego-status .es-hand.hidden-hands { background: rgba(20, 22, 18, 0.9); border-color: rgba(255,255,255,0.45);
  color: #f3f2ec; }
.ego-status .es-hand.gloved { background: rgba(20, 22, 18, 0.9); border-color: rgba(255,255,255,0.3); color: #e6e8df; }
/* phones: the video is too small to carry the caption card on top of it, so it moves below the footage (as the
   handheld grippers' strip already does), its space kept while hidden so nothing jumps */
@media (max-width: 600px) {
  aside.left .cam-cell:has(> .video-overlay) { flex-wrap: wrap; }
  aside.left .cam-cell > video { flex: 1 0 100%; }
  aside.left .cam-cell .video-overlay { order: 2; flex: 1 0 100%; }
  aside.left .cam-cell .video-overlay, aside.left .cam-cell .video-overlay.active {
    position: static; transform: none; min-width: 0; max-width: none; margin: 6px 0 0; box-sizing: border-box;
    min-height: 64px; font-size: 13px; }
  /* the notes that come and go (a recovery, a state change) follow the caption below the footage, folding open and
     closed, since the small image has no room for them under its top row */
  aside.left .cam-cell > .recovery-overlay, aside.left .cam-cell > .state-toast {
    position: static; order: 3; flex: 1 0 100%; box-sizing: border-box; min-width: 0; max-width: none; margin: 0;
    transform: none; text-align: left; max-height: 0; overflow: hidden; padding-top: 0; padding-bottom: 0;
    border-top-width: 0; border-bottom-width: 0; box-shadow: none;
    transition: max-height 220ms ease, opacity 200ms ease, padding 220ms ease, margin 220ms ease; }
  aside.left .cam-cell > .state-toast { order: 4; }
  /* two gripper cameras side by side leave each cell too narrow for the progress chip's sparkline */
  aside.left .cam-row-grippers:not(.cam-row-single) .prog-overlay .po-svg { display: none; }
  aside.left .cam-cell > .recovery-overlay.active, aside.left .cam-cell > .state-toast.active {
    max-height: 220px; margin-top: 6px; padding-top: 7px; padding-bottom: 7px; border-top-width: 1px;
    border-bottom-width: 1px; }
}
section.right { overflow-y: auto; padding: 22px 28px; }

/* ---------- prompt banner ---------- */
.prompt-banner {
  background: var(--surface);
  border: 1px solid var(--border); border-left: 2px solid var(--fg);
  border-radius: var(--r-md);
  padding: 16px 20px; margin-bottom: 20px;
}
.prompt-banner .label {
  font-size: 11px; color: var(--fg-faint);
  letter-spacing: 0.01em; font-weight: 600;
  margin-bottom: 8px;
}
.prompt-banner .text {
  font-size: 16px; line-height: 1.5; color: var(--fg); font-weight: 400;
}

/* given-goal mode: the dataset's instruction is the anchor, the model's
   independent assessment sits beneath it so a divergence is visible at a glance. */
.prompt-banner.has-given { border-left-color: var(--fg); }
.prompt-banner .goal-given {
  display: flex; flex-direction: column; align-items: flex-start; gap: 8px; margin-bottom: 12px;
}
.prompt-banner .gg-badge {
  flex: none; font-size: 10px; font-weight: 700; letter-spacing: 0.03em;
  color: var(--bg); background: var(--fg-2);
  padding: 2px 7px; border-radius: 999px; line-height: 1.5;
}
.prompt-banner .gg-text {
  font-size: 16px; line-height: 1.5; color: var(--fg); font-weight: 500;
}
.prompt-banner .goal-read {
  display: flex; flex-direction: column; gap: 6px;
  padding-top: 10px; border-top: 1px dashed var(--border);
}
.prompt-banner .gr-badge {
  flex: none; font-size: 10px; font-weight: 600; letter-spacing: 0.02em;
  color: var(--fg-3); padding-top: 1px;
}
.prompt-banner .gr-text { font-size: 14px; line-height: 1.5; color: var(--fg-2); }
/* alignment chip: whether the model's independent assessment matches the given goal */
.align-chip {
  display: flex; flex-direction: column; align-items: flex-start; gap: 5px; margin-top: 10px;
  font-size: 12px; font-weight: 600; padding: 7px 10px; border-radius: var(--r-md);
  border: 1px solid var(--border);
}
/* a chip with only its label stays a compact pill */
.align-chip:not(:has(.ac-note)) { display: inline-flex; border-radius: 999px; padding: 3px 9px; }
.align-chip .ac-note { font-weight: 400; color: var(--fg-3); line-height: 1.45; }
.align-chip.match { color: var(--success); background: rgba(78,194,127,0.10);
  border-color: color-mix(in srgb, var(--success) 45%, transparent); }
.align-chip.nomatch { color: var(--danger); background: rgba(179,38,60,0.07);
  border-color: color-mix(in srgb, var(--danger) 45%, transparent); }

/* ---------- timeline ---------- */
.timeline {
  position: relative; height: 28px;
  background: var(--bg); border: 1px solid var(--border);
  border-radius: var(--r-sm); margin: 12px 0 4px; cursor: pointer;
}
/* lanes under the timeline share its box, so their time scale lines up with it exactly */
.lane { margin: 6px 0 0; }
.timeline + .lane { margin-top: 22px; }   /* clear the timeline's tick labels, which hang 16px below it */
.lane-title { font: 600 10.5px/1.3 var(--mono); letter-spacing: .04em; color: var(--fg-3); margin: 0 0 3px;
  display: flex; gap: 8px; align-items: baseline; min-width: 0; }
.lane-now { font: 500 12px/1.3 var(--sans); letter-spacing: 0; color: var(--fg); overflow-wrap: anywhere; }
.lane-sum { font: 500 10.5px/1.3 var(--mono); letter-spacing: 0; color: var(--fg-2); }
.lane-bar { position: relative; height: 14px; background: var(--bg); border: 1px solid var(--border);
  border-radius: var(--r-sm); cursor: pointer; }
.lane-seg { position: absolute; top: 2px; bottom: 2px; border-radius: 1px; }
.lane-seg.hands { background: #7c87c4; }
/* the hands lane: its name and share on the left, a stepper through its stretches on the right, one line over the
   bar; the stretch under the playhead is lit */
.lane-head { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 4px 12px;
  margin: 0 0 4px; }
.lane-head .lane-title { margin: 0; }
.lane-nav { display: inline-flex; align-items: center; gap: 2px; flex: none; margin-left: auto; }
.lane-step { width: 20px; height: 20px; padding: 0; display: inline-flex; align-items: center; justify-content: center;
  border: 1px solid var(--border); border-radius: var(--r-sm); background: var(--surface); color: var(--fg-2);
  cursor: pointer; transition: background .15s, color .15s, border-color .15s; }
.lane-step svg { width: 10px; height: 10px; }
.lane-step:hover { background: var(--bg); color: var(--fg); border-color: var(--border-strong); }
.lane-pos { min-width: 88px; padding: 0 4px; white-space: nowrap; text-align: center; font: 500 10.5px/1 var(--mono); color: var(--fg-2);
  font-variant-numeric: tabular-nums; }
.lane-seg.hands { cursor: pointer; opacity: .55; transition: opacity .2s, background-color .2s; }
.lane-seg.hands:hover { opacity: .85; }
.lane-seg.hands.now { opacity: 1; background: #4d5aa0; }
/* the last lane keeps clear of the next section's rule, so the bar's edge is never doubled */
.lane + section { margin-top: 22px; }
.lane-seg.pub { background: rgba(69,129,142,0.45); }
.lane-seg.pub.alt { background: rgba(69,129,142,0.28); }
.lane-seg.pub.now { background: #45818e; }
.lane-ph { position: absolute; top: -2px; bottom: -2px; width: 1px; background: var(--fg); pointer-events: none; }
/* ---------- sensors: the recording's other signals and depth (board/sensors.py) ----------
   Under the timeline and on its time scale. Each signal that changes is a lane as wide as the timeline, so its playhead
   stands under the timeline's: its name and the value at the playhead over a strip of its samples, the stretches it
   spends away from rest shaded. A 2-D array that is touch (label/signals.py is_touch, a pressure map) is drawn as a
   heatmap at the playhead beside the lanes' start. Lanes past the first few open with "Show all". */
/* clear the timeline's tick labels, which hang 16px below it, as the first lane does */
.timeline + #sn-slot > h3.sn-h, .lane + #sn-slot > h3.sn-h { margin-top: 22px; }
.sn { margin: 0 0 22px; }
.sn > .lane:first-child, .sn-maps + .lane { margin-top: 0; }
.sn .lane { margin-top: 10px; }
.sn-plot { height: 38px; overflow: hidden; }
.sn-plot svg { position: absolute; inset: 0; width: 100%; height: 100%; overflow: visible; pointer-events: none; }
.sn-plot path { fill: none; stroke: var(--fg); stroke-width: 1.25; vector-effect: non-scaling-stroke;
  stroke-linejoin: round; stroke-linecap: round; }
.sn-plot path.area { fill: rgba(28,28,26,0.10); stroke: none; }
.sn-span { position: absolute; top: 0; bottom: 0; background: rgba(28,28,26,0.07); pointer-events: none; }
.sn-hv { position: absolute; top: -2px; bottom: -2px; width: 1px; background: var(--fg-3); pointer-events: none;
  opacity: 0; transition: opacity 120ms ease; }
.sn-plot.hover .sn-hv { opacity: 1; }
.sn .lane-head { align-items: baseline; }
/* a long name and its summary wrap onto lines that start at the same left edge */
.sn .lane-title { flex-wrap: wrap; row-gap: 2px; }
.sn .lane-now { display: inline-flex; flex-wrap: wrap; justify-content: flex-end; gap: 2px 10px; margin-left: auto; }
.sn-v { display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; }
.sn-v i { flex: none; width: 10px; height: 2px; border-radius: 1px; background: var(--fg); }
.sn-maps { display: flex; flex-wrap: wrap; gap: 12px; margin: 0 0 14px; }
.sn-map { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 6px 14px; align-items: start;
  flex: 1 1 300px; min-width: 0; padding: 12px 14px; background: var(--raised); border: 1px solid var(--border);
  border-radius: var(--r-md); }
.sn-grid { position: relative; grid-row: 1 / span 3; width: 112px; line-height: 0; }
.sn-grid canvas { width: 100%; height: auto; image-rendering: pixelated; border-radius: var(--r-sm);
  box-shadow: 0 0 0 1px var(--border); background: var(--bg); }
.sn-peak { position: absolute; box-sizing: border-box; border: 1.5px solid var(--surface);
  box-shadow: 0 0 0 1px var(--fg); border-radius: 1px; pointer-events: none; opacity: 0;
  transition: opacity 160ms ease, left 90ms linear, top 90ms linear; }
.sn-peak.on { opacity: 1; }
.sn-map .lane-title { display: block; margin: 0; overflow-wrap: anywhere; }
.sn-map .lane-now { display: block; margin: 0; }
.sn-note { font-size: 12px; line-height: 1.45; color: var(--fg-3); min-width: 0; overflow-wrap: anywhere; }
.sn > .sn-note { margin-top: 10px; }
.sn .ck-all-in > .lane:first-child { margin-top: 10px; }
.sn.open .ck-all { grid-template-rows: 1fr; }
.sn.open .ck-all-in { opacity: 1; }
.sn-depth { margin-top: 14px; padding: 12px 14px; background: var(--raised); border: 1px solid var(--border);
  border-radius: var(--r-md); }
.sn-depth .lane-title { margin: 0 0 4px; }
.sn-depth + .sn-depth { margin-top: 12px; }
.sn-bar { position: relative; height: 10px; margin: 10px 0 0; border-radius: var(--r-sm);
  box-shadow: 0 0 0 1px var(--border); }
.sn-ticks { position: relative; height: 16px; margin-top: 3px; font: 500 10px/1 var(--mono); color: var(--fg-2); }
.sn-ticks span { position: absolute; top: 3px; white-space: nowrap; }
.sn-ticks span::before { content: ""; position: absolute; top: -6px; left: var(--tx, 50%); width: 1px; height: 4px;
  background: var(--border-strong); }
/* a camera with depth: a switch on its picture shows its depth clip in place of its colour one, faded over it and
   played in step with it. It sits under the full-screen button on the main camera and in the top right corner of a
   side camera, styled like the full-screen button. While the player's own controls show, the strip they cover is
   dimmed, as over the hand pose. */
.cam-dp { position: absolute; z-index: 5; top: 8px; right: 8px; height: 26px; padding: 0 8px; cursor: pointer;
  display: inline-flex; align-items: center; gap: 7px; border: 1px solid rgba(255,255,255,0.18); border-radius: 5px;
  background: rgba(0,0,0,0.55); color: #fff; font: 500 10px/1 var(--mono); white-space: nowrap;
  transition: background-color 160ms ease, border-color 160ms ease, opacity 200ms ease; }
.cam-dp:hover { background: rgba(0,0,0,0.82); }
.cam-dp[hidden] { display: none; }
.cam-exo .cam-dp { top: calc(var(--fx-top, 0px) + 40px); right: calc(var(--fx-right, 0px) + 8px); }
.cam-dp .hp-sw { border-color: rgba(255,255,255,0.45); }
.cam-dp .hp-sw::after { background: rgba(255,255,255,0.75); }
.cam-dp[aria-pressed="true"] { border-color: rgba(255,255,255,0.45); }
.cam-dp[aria-pressed="true"] .hp-sw { background: #fff; border-color: #fff; }
.cam-dp[aria-pressed="true"] .hp-sw::after { transform: translateX(10px); background: #000; }
@property --dp-band { syntax: "<number>"; inherits: false; initial-value: 1; }
.dp-vid { position: absolute; left: 0; top: 0; width: 0; height: 0; object-fit: contain; background: transparent;
  pointer-events: none; opacity: 0; --dp-band: 1; transition: opacity 220ms ease, --dp-band 200ms ease;
  -webkit-mask-image: linear-gradient(to top, rgba(0,0,0,var(--dp-band)) var(--dp-bar, 0px),
    #000 calc(var(--dp-bar, 0px) + 12px));
  mask-image: linear-gradient(to top, rgba(0,0,0,var(--dp-band)) var(--dp-bar, 0px),
    #000 calc(var(--dp-bar, 0px) + 12px)); }
.dp-vid.on { opacity: 1; }
.dp-vid.ctl { --dp-band: 0.18; }
@media (prefers-reduced-motion: reduce) { .dp-vid, .sn-peak, .sn-hv { transition: none; } }
#sn-slot { transition: opacity 220ms ease; }
#sn-slot.sn-wait { opacity: 0; transition: none; }
.sn-n { display: inline-block; min-width: 5ch; text-align: left; }
/* the panel of every recorded signal folds closed on an episode with contacts, which the Touch lane already shows */
.sn-fold { display: grid; grid-template-rows: 0fr; transition: grid-template-rows 320ms cubic-bezier(.32,.72,0,1); }
.sn-fold-in { overflow: hidden; min-height: 0; opacity: 0; transition: opacity 240ms ease; }
.sn.shown .sn-fold { grid-template-rows: 1fr; }
.sn.shown .sn-fold-in { opacity: 1; }
.sn-fold-in > .lane:first-child, .sn-fold-in > .sn-maps:first-child { margin-top: 10px; }
.sn-fold + .ck-more { margin-top: 0; }
@media (prefers-reduced-motion: reduce) { .sn-fold, .sn-fold-in { transition: none; } }
/* ---------- touch: the recording's contacts (label/contacts.py) and what the model saw at each ----------
   One bar per hand on the timeline's time scale. A contact is a box from its begin to its end with its strength drawn
   inside (its signals' activity over their swing, on one scale for the episode) and a tick where it is strongest; the
   box's colour is what the model found in the frames. A diamond is a moment the model saw a hand take hold of
   something that no recorded contact covers. */
.tc-row + .tc-row { margin-top: 6px; }
.tc-hand { display: block; margin: 0 0 3px; font: 500 10.5px/1.3 var(--mono); color: var(--fg-2); }
.lane-bar.tc-bar { height: 24px; }
.tc-seg { --c: var(--fg-2); position: absolute; top: 2px; bottom: 2px; box-sizing: border-box; overflow: hidden;
  border: 1px solid var(--c); border-radius: 2px; background: color-mix(in srgb, var(--c) 12%, transparent);
  cursor: pointer; opacity: .62; transition: opacity .2s, box-shadow .2s; }
.tc-seg:hover { opacity: .85; }
.tc-seg.now { opacity: 1; box-shadow: 0 0 0 1px var(--surface), 0 0 0 2px var(--c); z-index: 1; }
.tc-seg svg { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
.tc-seg path { fill: color-mix(in srgb, var(--c) 55%, transparent); stroke: none; opacity: 0; transition: opacity 220ms ease; }
.tc-seg path.in { opacity: 1; }
.st-yes { --c: var(--fg-2); }
.st-no { --c: var(--danger); }
.st-unclear, .st-unshown { --c: var(--fg-disabled); }
.tc-seg.st-unshown { border-style: dashed; }
.tc-peak { position: absolute; top: -3px; bottom: -3px; width: 1px; margin-left: -0.5px; background: var(--c);
  pointer-events: none; z-index: 2; }
.tc-miss { position: absolute; top: 50%; width: 9px; height: 9px; margin: -4.5px 0 0 -4.5px; z-index: 3;
  transform: rotate(45deg); background: var(--danger); border: 1.5px solid var(--surface); border-radius: 1px;
  box-sizing: border-box; cursor: pointer; transition: transform .15s; }
.tc-miss:hover { transform: rotate(45deg) scale(1.25); }
.tc-key { display: flex; flex-wrap: wrap; gap: 4px 14px; margin-top: 8px; font-size: 12px; line-height: 1.45;
  color: var(--fg-3); }
.tc-k { display: inline-flex; align-items: center; gap: 6px; }
.tc-k i { flex: none; width: 14px; height: 9px; box-sizing: border-box; border: 1px solid var(--c); border-radius: 2px;
  background: color-mix(in srgb, var(--c) 40%, transparent); }
.tc-k i.st-unshown { border-style: dashed; background: color-mix(in srgb, var(--c) 12%, transparent); }
.tc-k i.miss { width: 8px; height: 8px; margin: 0 3px; border: 0; transform: rotate(45deg); background: var(--danger); }
/* the contact card: every contact's card in one cell, so the space they take never changes while the footage plays and
   the card of the contact under the playhead fades in over the last */
.tc-cards { display: grid; margin: 12px 0 22px; }
.tc-card { grid-area: 1 / 1; min-width: 0; margin: 0; opacity: 0; visibility: hidden;
  transition: opacity 180ms ease, visibility 0s linear 180ms; }
.tc-card.on { opacity: 1; visibility: visible; transition: opacity 180ms ease, visibility 0s; }
.tc-card-head { display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; gap: 4px 12px; }
.tc-card-title { font: 700 13px/1.3 var(--sans); color: var(--fg); }
.tc-pill { font: 600 11px/1.3 var(--mono); color: var(--c); padding: 2px 8px; border-radius: var(--r-pill);
  border: 1px solid color-mix(in srgb, var(--c) 45%, transparent); background: color-mix(in srgb, var(--c) 8%, transparent);
  white-space: nowrap; }
.tc-pill.st-yes { color: var(--fg); }
.tc-times { margin-top: 6px; font: 500 12px/1.45 var(--mono); color: var(--fg-2); }
.tc-times [data-t] { cursor: pointer; color: var(--fg); text-decoration: underline; text-decoration-color: var(--border-strong);
  text-underline-offset: 3px; }
.tc-times [data-t]:hover { text-decoration-color: var(--fg); }
.tc-card .tc-kv { margin-top: 8px; padding: 0; }
.tc-card .kv { grid-template-columns: minmax(84px, 124px) minmax(0, 1fr); gap: 2px 14px; padding: 7px 0; }
.tc-card .kv .kv-v { overflow-wrap: anywhere; }
.tc-card .kv-v + .kv-v { grid-column: 2; }
.tc-plain { margin-top: 8px; font-size: 13px; line-height: 1.5; color: var(--fg-2); }
.tc-card .sn-maps { margin: 10px 0 0; }
.tc-card .sn-map { flex-basis: 260px; }
.tc-card > .sn-note { margin-top: 8px; }
.tc-empty .tc-plain { margin-top: 0; color: var(--fg-3); }
@media (prefers-reduced-motion: reduce) { .tc-card, .tc-seg, .tc-seg path { transition: none; } }
.pub-list .pub-note { font-size: 12px; color: var(--fg-3); margin: 0 0 8px; line-height: 1.45; }
.pub-list .pub-row { display: grid; grid-template-columns: 112px 1fr; gap: 10px; padding: 4px 6px; cursor: pointer;
  border-radius: 3px; font-size: 13px; line-height: 1.4; }
.pub-list .pub-row:hover { background: rgba(127,127,127,0.08); }
.pub-list .pub-row.now { background: rgba(69,129,142,0.14); }
.pub-list .pub-t { font-family: var(--mono); font-size: 11.5px; color: var(--fg-3); }
.pub-list .pub-ep { font-size: 13px; line-height: 1.6; margin-top: 4px; }
.pub-list .pub-k { color: var(--fg-3); font-family: var(--mono); font-size: 11px; margin-right: 6px; }
.pub-list .pub-group + .pub-group, .pub-list .pub-row + .pub-group, .pub-list .pub-ep + .pub-group,
.pub-list .pub-fold:not(:first-child) { margin-top: 12px; }
.pub-list .pub-gt { font-family: var(--mono); font-size: 11px; color: var(--fg-3); margin: 0 0 4px; overflow-wrap: anywhere; }
.pub-list .pub-group { display: grid; grid-template-columns: fit-content(42%) 1fr; column-gap: 10px; }
.pub-list .pub-gt { grid-column: 1 / -1; }
.pub-list .pub-kv { display: contents; font-size: 13px; line-height: 1.4; }
.pub-list .pub-kv .pub-k { min-width: 100px; margin: 0; padding: 4px 0 4px 6px; line-height: 1.6; overflow-wrap: anywhere; }
.pub-list .pub-v { padding: 4px 6px 4px 0; font-size: 13px; overflow-wrap: anywhere; }
.pub-list .pub-at { margin-left: 6px; padding: 0; background: none; border: 0; cursor: pointer; font-family: var(--mono);
  font-size: 11.5px; color: var(--accent); }
.pub-list .pub-at:hover { text-decoration: underline; }
.pub-fold.shown .sn-fold { grid-template-rows: 1fr; }
.pub-fold.shown .sn-fold-in { opacity: 1; }
.timeline .marker {
  position: absolute; top: 5px; bottom: 9px; width: 2px;
  border-radius: 1px; cursor: pointer; transition: width 100ms;
}
.timeline .marker.seg.adv   { background: var(--fg-2); }
.timeline .marker.seg.waste { background: #4d5aa0; }
.timeline .marker.seg.idle  { background: var(--fg-disabled); }
.timeline .marker.seg.none  { background: var(--fg-disabled); }
.timeline .marker.key {
  width: 4px; top: 0; bottom: 4px; background: var(--fg-2); z-index: 6;
  border-radius: 2px;
}
.timeline .marker.key:hover { width: 6px; z-index: 7; }
.key-events { display: flex; flex-direction: column; gap: 6px; }
.key-ev {
  display: grid; grid-template-columns: 20px 46px 1fr; gap: 11px; align-items: start;
  padding: 9px 11px; border-radius: var(--r-sm); cursor: pointer;
  border-left: 3px solid var(--fg-disabled);
  background: rgba(28,28,26,0.050);
  transition: background 100ms;
}
.key-ev.success { border-left-color: var(--border-strong); }
.key-ev.failure { border-left-color: var(--danger); }
.key-ev.unclear { border-left-color: var(--border-strong); }
.key-ev.goal { border-left-color: var(--success); }
.key-ev:hover { background: rgba(28,28,26,0.054); }
.key-ev .ke-time { color: var(--fg-3); font-family: var(--mono); font-size: 11.5px; padding-top: 1px; }
/* a key event or step with no time the board can read: listed after the timed ones, nothing to seek to */
.key-ev.untimed, .feed .ev.untimed { cursor: default; }
.key-ev.untimed .ke-time, .feed .ev.untimed .t { font-size: 10.5px; white-space: nowrap; }
.key-ev .ke-body { min-width: 0; }
.key-ev .ke-row1 { display: flex; align-items: baseline; gap: 10px; justify-content: space-between; }
.key-ev .ke-label { color: var(--fg); font-size: 13px; line-height: 1.4; }
.key-ev .outcome {
  flex: 0 0 auto; font-size: 10px; font-family: var(--mono); font-weight: 600;
  padding: 3px 9px; border-radius: var(--r-pill); border: 1px solid transparent; white-space: nowrap;
}
.key-ev .outcome.success { background: rgba(78,194,127,0.12); color: var(--success);
  border-color: rgba(78,194,127,0.22); }
.key-ev .outcome.failure { background: rgba(179,38,60,0.12); color: var(--danger);
  border-color: rgba(179,38,60,0.22); }
.key-ev .outcome.unclear { background: rgba(28,28,26,0.05); color: var(--fg-2); border-color: var(--border); }
.key-ev .ke-note { margin-top: 5px; font-size: 11.5px; color: var(--fg-3); font-style: italic; line-height: 1.45; }
.timeline .marker:hover { width: 4px; z-index: 5; }
.timeline .marker .tip {
  position: absolute; bottom: 30px; left: 50%; transform: translateX(-50%);
  background: var(--fg); color: var(--bg);
  padding: 4px 8px; border-radius: var(--r-sm);
  font-size: 10px; font-family: var(--mono);
  white-space: nowrap; display: none; z-index: 20;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.6);
}
.timeline .marker:hover .tip { display: block; }
.timeline .playhead {
  position: absolute; top: 0; bottom: 4px; width: 2px;
  background: var(--accent); pointer-events: none; z-index: 4;
}
.timeline .tick {
  position: absolute; bottom: 0; width: 1px; height: 4px;
  background: var(--border-strong);
}
.timeline .tick-label {
  position: absolute; bottom: -16px; font-size: 10px; color: var(--fg-2);
  font-family: var(--mono); transform: translateX(-50%);
}

/* ---------- section headers ---------- */
/* Real titles: bright, bold, larger, with an accent tick and a bottom rule
   so they read as headers, not dim inline labels. */
/* the rule sits ABOVE each heading, so it separates this section from the one
   before it (a divider), rather than underlining the title. */
h3.section {
  display: flex; align-items: center; gap: 9px;
  margin: 0 0 14px; padding-top: 24px;
  border-top: 1px solid var(--border-strong);
  font-size: 15px; font-weight: 700; color: var(--fg);
  letter-spacing: -0.005em;
}
h3.section::before {
  content: ""; flex: 0 0 auto;
  width: 3px; height: 15px; border-radius: 2px;
  background: var(--fg);
}
/* first heading in a column has nothing above it to divide from */
h3.section:first-child { margin-top: 0; padding-top: 0; border-top: none; }
/* the problem cards: one per kind, coloured to match the rail chips and the filter groups */
.ip-stack { display: flex; flex-direction: column; gap: 14px; margin-bottom: 22px; }
.ip { --g: var(--fg-3); border: 1px solid color-mix(in srgb, var(--g) 38%, var(--border));
  border-top: 3px solid var(--g);
  border-radius: var(--r-md); background: color-mix(in srgb, var(--surface) 94%, var(--g)); overflow: hidden; }
.ip-checks { --g: #2f7f76; }
.ip-data { --g: var(--danger); }
.ip-mistake { --g: var(--warning); }
.ip-head { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; padding: 12px 14px 2px; }
.ip-head h3 { margin: 0; font: 700 14px/1.25 var(--sans); color: var(--g); }
.ip-n { font: 600 12px/1 var(--mono); color: var(--g); background: color-mix(in srgb, var(--g) 16%, transparent);
  padding: 3px 8px; border-radius: 999px; white-space: nowrap; }
.ip-sub { margin: 4px 14px 10px; font-size: 12px; line-height: 1.4; color: var(--fg-3); }
.ip-body { padding: 0 10px 10px; }
.ip-body .info-block { margin: 0; }
.ip-body .info-block + .info-block { margin-top: 8px; }
.ip-none { margin: 0 0 22px; padding: 12px 14px; font-size: 13px; color: var(--fg-3);
  border: 1px dashed var(--border-strong); border-radius: var(--r-md); }
h3.section .count {
  color: var(--fg-3); font-weight: 500; letter-spacing: 0;
  font-size: 12px; font-family: var(--mono);
}

/* ---------- info blocks (left col) ---------- */
.info-block {
  background: var(--raised); border: 1px solid var(--border);
  border-radius: var(--r-md); padding: 12px 14px; margin-bottom: 12px;
}
/* aligned key/value block: label column + value column, so wrapped value
   lines hang-indent under the value (not back under the label). */
.kv-block { padding: 4px 14px; }
.kv { display: grid; grid-template-columns: 170px 1fr; gap: 4px 16px; padding: 9px 0; align-items: start; }
.kv + .kv { border-top: 1px solid var(--border); }
.kv .kv-k { color: var(--fg-2); font-weight: 600; font-size: 12px; }
.kv .kv-v { color: var(--fg); font-size: 13px; line-height: 1.5; }
.info-block .line { font-size: 12.5px; line-height: 1.75; }
.info-block .line .v { color: var(--fg); font-weight: 500; }
.completion-banner {
  display: flex; align-items: center; gap: 12px;
  padding: 12px 16px; border-radius: var(--r-md);
  margin-bottom: 8px;
}
.completion-banner .comp-detail {
  font-size: 12px; color: var(--fg-2); line-height: 1.4;
}

/* problem rows: a check, a data issue or an operator mistake, each with its severity */
.di-block { display: flex; flex-direction: column; gap: 0; }
.di-row { display: grid; grid-template-columns: 54px 1fr; gap: 10px; align-items: start; padding: 9px 2px;
  border-top: 1px solid var(--row-divider); }
.di-row:first-child { border-top: none; }
.di-row.minor { opacity: 0.72; }
.di-minor { font-family: var(--mono); font-size: 10px; color: var(--fg-3); border: 1px dashed var(--border-strong);
  padding: 1px 6px; border-radius: var(--r-pill); }
.ip-counts { display: inline-flex; align-items: baseline; gap: 8px; }
.ip-minor { font: 500 11px/1 var(--mono); color: var(--fg-3); white-space: nowrap; }
.di-row .di-sev { font-family: var(--mono); font-size: 9.5px; font-weight: 700;
  padding: 2px 6px; border-radius: var(--r-pill); text-align: center; }
.di-row.high .di-sev { color: var(--danger); background: rgba(179,38,60,0.14); border: 1px solid rgba(179,38,60,0.35); }
.di-row.low .di-sev { color: var(--fg-3); background: rgba(28,28,26,0.050); border: 1px solid var(--border); }
.di-issue { font-size: 13px; color: var(--fg); font-weight: 600; line-height: 1.4; }
.di-ev { font-size: 11.5px; color: var(--fg-3); font-style: italic; line-height: 1.4; margin-top: 3px; }
.di-tags { display: flex; gap: 8px; align-items: center; margin-top: 4px; flex-wrap: wrap; }
.di-cat { font-family: var(--mono); font-size: 10px; color: var(--fg-2); background: var(--bg);
  border: 1px solid var(--border); padding: 1px 6px; border-radius: var(--r-pill); }
.di-t { font-family: var(--mono); font-size: 10.5px; color: var(--accent); }
.di-verified { font-size: 10px; font-weight: 600; color: var(--fg-2); border: 1px solid var(--border-strong);
  background: rgba(28,28,26,0.05); padding: 1px 6px; border-radius: var(--r-pill); }
.di-block .di-row[data-t]:hover { background: rgba(127,127,127,0.06); }
.di-block.op .di-row .di-sev { color: var(--warning); background: rgba(77,90,160,0.10);
  border: 1px solid rgba(77,90,160,0.30); }
.di-block.op .di-row.high .di-sev { color: #36407a; background: rgba(77,90,160,0.22);
  border-color: rgba(77,90,160,0.55); }

/* a session's tasks panel, in place of the single completion */
.tasks-summary { font-size: 12.5px; color: var(--fg-2); margin: 2px 0 10px; }
.tasks-summary .ts-ok { color: var(--success); font-weight: 600; }
.tasks-summary .ts-part { color: var(--fg-2); font-weight: 600; }
.tasks-summary .ts-fail { color: var(--danger); font-weight: 600; }
.tasks-summary .tasks-note { font-size: 11px; color: var(--fg-3); margin-top: 4px; }
.tasks-panel { display: flex; flex-direction: column; gap: 0; }
.task-row {
  display: grid; grid-template-columns: 22px 1fr; gap: 10px; align-items: start;
  padding: 10px 2px; border-top: 1px solid var(--row-divider); cursor: pointer;
}
.task-row:first-child { border-top: none; }
.task-row:hover { background: var(--raised); }
.task-row .task-num {
  width: 20px; height: 20px; border-radius: 4px; display: inline-flex;
  align-items: center; justify-content: center; font-family: var(--mono);
  font-size: 11px; font-weight: 700; color: #0b0b0c; background: var(--fg-3);
}
.task-row .task-num.success { background: #2f7d52; }
.task-row .task-num.partial { background: #e7b0b8; }
.task-row .task-num.failure { background: #b3263c; color: #fff; }
.task-head { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; }
.task-head .task-name { font-size: 13px; font-weight: 600; color: var(--fg); }
.task-meta { font-family: var(--mono); font-size: 10.5px; color: var(--fg-3); margin-top: 3px; }
.task-pred { font-size: 12px; color: var(--fg-2); line-height: 1.45; margin-top: 4px; }
.task-note { font-size: 11.5px; color: var(--fg-3); font-style: italic; line-height: 1.4; margin-top: 3px; }
.task-row .outcome {
  font-size: 10px; font-family: var(--mono); font-weight: 700; letter-spacing: 0.02em;
  padding: 1px 8px; border-radius: var(--r-pill); border: 1px solid transparent;
}
.task-row .outcome.success { color: var(--success); background: rgba(78,194,127,0.12);
  border-color: rgba(78,194,127,0.22); }
.task-row .outcome.partial { color: var(--fg-2); background: rgba(28,28,26,0.05); border-color: var(--border); }
.task-row .outcome.failure { color: var(--danger);  background: rgba(179,38,60,0.12);
  border-color: rgba(179,38,60,0.22); }


.inv-list { display: flex; flex-direction: column; gap: 6px; font-size: 12px; }
.inv-list .it { display: flex; gap: 8px; align-items: baseline; }
.inv-list .it .name { color: var(--fg); }
.inv-list .it .color { color: var(--fg-faint); font-size: 11px; }

/* ---------- event feed ---------- */
/* Each row is one instant: time | arm | phrase | contribution, on a
   fixed 4-track grid so columns line up and the outcome never wraps. A rule
   between rows keeps the dense labels legible as discrete lines. */
/* dense-timeline count summary: total + color-coded contribution pills, on
   their own line so they never wrap into a misleading "N steps idle". */
.dense-head { margin-bottom: 6px; }
.dense-stats {
  display: flex; flex-wrap: wrap; gap: 6px 8px; align-items: center;
  margin: 0 0 12px; font-family: var(--mono); font-size: 11px;
}
.dense-stats .ds-total { color: var(--fg-2); font-weight: 600; }
.dense-stats .ds-chip {
  padding: 2px 9px; border-radius: var(--r-pill);
  border: 1px solid transparent; font-weight: 600;
}
.dense-stats .ds-chip.adv   { background: rgba(28,28,26,0.070); color: var(--fg); border-color: var(--border-strong); }
.dense-stats .ds-chip.waste { background: rgba(77,90,160,0.12); color: var(--warning);
  border-color: rgba(77,90,160,0.30); }
.dense-stats .ds-chip.idle  { background: rgba(28,28,26,0.050); color: var(--fg-3); border-color: var(--border); }

.feed { display: flex; flex-direction: column; gap: 0; }
.feed { container-type: inline-size; }
.feed .ev {
  display: grid;
  grid-template-columns: 50px 92px minmax(0, 1fr) 74px;
  column-gap: 14px;
  padding: 13px 12px 13px 14px;
  cursor: pointer; transition: background 100ms, box-shadow 100ms;
  border-left: 3px solid transparent;
  border-top: 1px solid var(--row-divider);
  /* baseline so the timestamp, arm, phrase and contribution pill all sit on the
     same first text line instead of each column starting at a different height */
  align-items: baseline;
}
.feed .ev:first-child { border-top: none; }
.feed .ev { scroll-margin-top: 12px; }
.feed .ev:hover { background: var(--raised); }

/* the row under the playhead, updated live during playback */
.feed .ev.active {
  background: rgba(36,36,31,0.10);
  border-left-color: var(--accent);
  box-shadow: inset 0 0 0 1px rgba(36,36,31,0.18);
}

.feed .ev .t {
  font-family: var(--mono); font-size: 11.5px;
  color: var(--fg-muted); text-align: right; padding-top: 1px;
}
.feed .ev.active .t { color: var(--accent); font-weight: 600; }
.feed .ev .who { display: flex; flex-direction: column; gap: 3px; align-items: flex-start; }
.feed .ev .arm {
  font-family: var(--mono); font-size: 10.5px; font-weight: 600;
  color: var(--fg-faint); letter-spacing: 0.02em;
}
.feed .ev .arm.left  { color: var(--arm-left); }
.feed .ev .arm.right { color: var(--arm-right); }
.feed .ev .arm.both  { color: var(--fg-2); }

.feed .ev .phrase { color: var(--fg); line-height: 1.5; font-size: 13px; }
.feed .ev.empty .phrase { color: var(--fg-faint); font-style: italic; }

/* destination chip - the carry_phase, rendered after the sentence */
.feed .ev .dest-chip {
  display: inline-flex; align-items: baseline; gap: 4px; margin: 7px 0 0;
  font-size: 11px; font-family: var(--mono);
  padding: 2px 9px; border-radius: var(--r-pill);
  background: rgba(36,36,31,0.08);
  color: var(--fg-2); border: 1px solid rgba(36,36,31,0.28);
}
.feed .ev .dest-chip .prep { color: var(--fg-3); }

/* right column pill (key-event outcome, or dense-step contribution) - fixed
   width, top-aligned with the time */
.feed .ev .outcome, .feed .ev .contrib {
  font-size: 10px; font-family: var(--mono);
  font-weight: 600; letter-spacing: 0.02em;
  padding: 3px 0; border-radius: var(--r-pill);
  text-align: center; white-space: nowrap;
  border: 1px solid transparent;
}
.feed .ev .outcome.success { background: rgba(78, 194, 127, 0.12);  color: var(--success);
  border-color: rgba(78, 194, 127, 0.22); }
.feed .ev .outcome.partial { background: rgba(28,28,26,0.05);  color: var(--fg-2); border-color: var(--border); }
.feed .ev .outcome.failure { background: rgba(179,38,60, 0.12); color: var(--danger);
  border-color: rgba(179,38,60, 0.22); }
.feed .ev .outcome.none    { background: transparent; color: var(--fg-faint); border-color: var(--border); }
.feed .ev .contrib.adv   { background: rgba(28,28,26,0.070); color: var(--fg); border-color: var(--border-strong); }
.feed .ev .contrib.waste { background: rgba(77,90,160,0.12); color: var(--warning);
  border-color: rgba(77,90,160,0.30); }
.feed .ev .contrib.idle  { background: rgba(28,28,26,0.050); color: var(--fg-3); border-color: var(--border); }
.feed .ev .contrib.none  { background: transparent; color: var(--fg-faint); border-color: var(--border); }

/* key-event milestone rows, interleaved into the feed by time */
.feed .ev.keyrow {
  /* inherits the dense grid (t | col2 | description | right pill) so the key
     label lines up with the dense phrases and the outcome with the contribs */
  background: rgba(28, 28, 26, 0.035);
  border-left-color: var(--border-strong);
  border-top: 1px solid var(--row-divider);
}
.feed .ev.keyrow + .ev { border-top: 1px solid var(--row-divider); }
.feed .ev.keyrow:hover { background: rgba(28, 28, 26, 0.05); }
.feed .ev.keyrow.active { background: rgba(69, 129, 142, 0.10); border-left-color: var(--accent);
  box-shadow: inset 0 0 0 1px rgba(69,129,142,0.28); }
/* col2: the "key event" badge */
.feed .ev.keyrow .ke-col {
  display: flex; flex-direction: column; align-items: flex-start; gap: 7px;
}
.feed .ev.keyrow .ke-badge {
  flex: 0 0 auto;
  font-family: var(--mono); font-size: 9.5px; font-weight: 700; letter-spacing: 0.02em;
  color: var(--fg-2); background: rgba(28,28,26,0.07);
  padding: 2px 8px; border-radius: var(--r-pill); border: 1px solid var(--border);
  white-space: nowrap;
}
/* the kind chip leads the event's own text, where there is room for the longest kinds
   ("processing pass complete"); in the narrow column beside it, a long kind ran into the text */
.feed .ev.keyrow .ke-kind {
  display: inline-block; margin-right: 8px; vertical-align: 1px;
  font-family: var(--mono); font-size: 9.5px; font-weight: 600; line-height: 1.5;
  color: var(--fg-2); background: rgba(28,28,26,0.090);
  padding: 0 7px; border-radius: var(--r-pill); border: 1px solid var(--border-strong);
  white-space: nowrap;
}
.feed .ev.keyrow .ke-text { min-width: 0; color: var(--fg); font-weight: 600; font-size: 13px; line-height: 1.5; }
/* a narrow feed: time, arm and pill on the first line, the phrase across the full width beneath,
   so nothing is squeezed into a sliver or pushed past the edge */
@container (max-width: 460px) {
  .feed .ev { grid-template-columns: 44px minmax(0, 1fr) max-content; row-gap: 7px; }
  .feed .ev > :nth-child(1) { grid-column: 1; grid-row: 1; }
  .feed .ev > :nth-child(2) { grid-column: 2; grid-row: 1; }
  .feed .ev > :nth-child(4) { grid-column: 3; grid-row: 1; padding-left: 8px; padding-right: 8px; }
  .feed .ev > :nth-child(3) { grid-column: 1 / -1; grid-row: 2; }
}

.perf { font-size: 13px; line-height: 1.5; color: var(--fg); }
.rec { padding: 10px 0; border-top: 1px solid var(--border); font-size: 13px; }
.rec:first-child { border-top: none; }
.rec-head { display: flex; align-items: baseline; flex-wrap: wrap; gap: 8px; }
.rec-t { color: var(--danger); font-family: var(--mono); font-size: 11.5px; font-weight: 700; cursor: pointer; }
.rec-t:hover { text-decoration: underline; }
.rec-fail { flex: 1; min-width: 140px; color: var(--fg); font-weight: 600; }
.rec-ok { color: var(--success); font-family: var(--mono); font-size: 11px; font-weight: 700; cursor: pointer;
  white-space: nowrap; }
.rec-ok:hover { text-decoration: underline; }
.rec-no { color: var(--danger); font-family: var(--mono); font-size: 11px; font-weight: 700; white-space: nowrap; }
.rec-line { margin-top: 5px; color: var(--fg-2); line-height: 1.5; }
.rec-line .rec-k { color: var(--fg-3); font-weight: 700; margin-right: 7px; }
.line .t { color: var(--fg-3); font-family: var(--mono); font-size: 11px; margin-right: 6px; }

/* top-left progress-to-goal HUD chip: compact one line, big % + inline
   sparkline + label, so it stays a short corner chip and never fights the
   recovery banner below it. */
.prog-overlay {
  position: absolute; top: calc(var(--fx-top, 0px) + 8px); left: calc(var(--fx-left, 0px) + 60px);
  z-index: 4; display: flex; align-items: center; gap: 8px;
  padding: 5px 11px; border-radius: var(--r-pill);
  background: rgba(0,0,0,0.55); border: 1px solid rgba(255,255,255,0.12);
  backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px); pointer-events: none;
}
.prog-overlay .po-pct { font-family: var(--mono); font-size: 17px; font-weight: 700; color: #f3f2ec; line-height: 1; }
.prog-overlay .po-sub { font-size: 10px; color: rgba(255,255,255,0.7); white-space: nowrap; }
.prog-overlay .po-svg { width: 62px; height: 18px; display: block; }
.prog-overlay .po-full { fill: none; stroke: rgba(255,255,255,0.25); stroke-width: 1;
  vector-effect: non-scaling-stroke; }
.prog-overlay .po-past { fill: none; stroke: #f3f2ec; stroke-width: 2; vector-effect: non-scaling-stroke; }
.prog-overlay .po-dot { fill: #f3f2ec; }
/* top-right transient toast: object state changes, below the fullscreen button (.fs-btn, 8 px + 26 px) */
.state-toast {
  position: absolute; top: calc(var(--fx-top, 0px) + 40px); right: calc(var(--fx-right, 0px) + 8px);
  z-index: 4; max-width: 210px;
  padding: 7px 11px; border-radius: var(--r-md); text-align: right;
  background: rgba(0,0,0,0.6); border: 1px solid rgba(255,255,255,0.22);
  backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px);
  opacity: 0; transform: translateY(-5px);
  transition: opacity 150ms ease, transform 150ms ease; pointer-events: none;
}
.state-toast.active { opacity: 1; transform: translateY(0); }
.state-toast .st-badge { font-family: var(--mono); font-size: 9px; font-weight: 700; letter-spacing: 0.04em;
  color: #d9dcd2; }
.state-toast .st-obj { font-size: 12px; font-weight: 700; color: #fff; margin-top: 2px; }
.state-toast .st-fromto { font-size: 12px; color: rgba(255,255,255,0.85); margin-top: 1px; }
.state-toast .st-from { color: rgba(255,255,255,0.6); }
.state-toast .st-arr { color: #d9dcd2; margin: 0 5px; font-weight: 700; }
.state-toast .st-to { color: #ffffff; font-weight: 600; }
.key-ev .ke-num {
  width: 20px; height: 20px; border-radius: 4px; background: #d6d5cc;
  display: inline-flex; align-items: center; justify-content: center;
  font-family: var(--mono); font-size: 11px; font-weight: 700; color: #0b0b0c;
}
/* the key event the playhead last passed, updated live during playback, as the dense timeline marks its row */
.key-ev.now { background: rgba(69,129,142,0.10); box-shadow: inset 0 0 0 1px rgba(69,129,142,0.28); }
.key-ev.now .ke-time { color: var(--accent); font-weight: 600; }

/* ---------- recovery overlay (top of exo) ----------
   Full-width banner spanning the frame, so verbose text stays SHORT (1-2 lines)
   instead of growing tall and covering the scene. It sits directly below the
   top-corner HUD chips (progress + state); its exact top is set in JS
   (placeTop) so it never overlaps them. The --fx-top value is only a
   fallback before JS runs. */
.recovery-overlay {
  position: absolute;
  left: calc(var(--fx-left, 0px) + 8px); right: calc(var(--fx-right, 0px) + 8px);
  top: calc(var(--fx-top, 0px) + 8px);
  background: rgba(18, 20, 34, 0.86);
  backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
  border-radius: var(--r-md); border: 1px solid rgba(163,174,224,0.35);
  border-left: 3px solid #a3aee0;
  padding: 8px 13px; color: #fff;
  /* Clean fade both ways (no translate yank), long enough to read as a fade and
     not a zap. `top` is set in JS as the corner chips come and go; ease it so the
     banner slides to its resting spot instead of snapping. */
  opacity: 0;
  transition: opacity 240ms ease, top 190ms cubic-bezier(0.22,0.61,0.36,1);
  pointer-events: none;
  box-shadow: 0 4px 18px rgba(0,0,0,0.5); z-index: 3;
}
.recovery-overlay.active { opacity: 1; }
.recovery-overlay.failed { border-left-color: #e58c9a; border-color: rgba(229,140,154,0.35);
  background: rgba(40, 20, 22, 0.86); }
.recovery-overlay .ro-row { margin-bottom: 5px; display: flex; align-items: center; gap: 8px; }
.recovery-overlay .ro-badge {
  display: inline-block;
  font-family: var(--mono); font-size: 10px; font-weight: 700; letter-spacing: 0.02em;
  color: #bcc2e0; background: rgba(163,174,224,0.20);
  padding: 2px 9px; border-radius: var(--r-pill); border: 1px solid rgba(163,174,224,0.4);
}
.recovery-overlay .ro-badge.failed { color: #e58c9a; background: rgba(229,140,154,0.18);
  border-color: rgba(229,140,154,0.45); }
.recovery-overlay .ro-status { font-family: var(--mono); font-size: 9.5px; font-weight: 700; }
.recovery-overlay .ro-status.ok { color: #7fd9a4; }
.recovery-overlay .ro-status.no { color: #e58c9a; }
.recovery-overlay .ro-fail { font-size: 12px; color: #f6d9de; line-height: 1.4; }
.recovery-overlay .ro-fix { font-size: 12px; color: rgba(255,255,255,0.92); line-height: 1.4; margin-top: 6px; }
.recovery-overlay .ro-fix-badge {
  display: inline-block; margin-right: 7px; vertical-align: 1px;
  font-family: var(--mono); font-size: 9.5px; font-weight: 700;
  color: #d5daf0; background: rgba(163,174,224,0.16);
  padding: 1px 8px; border-radius: var(--r-pill); border: 1px solid rgba(163,174,224,0.32);
}

/* goal-reached flag inside the bottom overlay */
.video-overlay .vo-goal { display: block; margin-bottom: 6px; }
.video-overlay .vo-goal-badge {
  display: inline-block;
  font-family: var(--mono); font-size: 10px; font-weight: 700; letter-spacing: 0.02em;
  color: #7fd9a4; background: rgba(78,194,127,0.20);
  padding: 2px 9px; border-radius: var(--r-pill); border: 1px solid rgba(78,194,127,0.45);
}

/* goal frame image in the completion block */
.goal-frame { margin: 6px 0 4px; cursor: pointer; }
.goal-frame .goal-frame-cap { font-size: 11.5px; color: var(--success); font-weight: 600; margin-bottom: 6px; }
.goal-frame img {
  display: block; width: 100%; max-width: 480px; border-radius: var(--r-md);
  border: 1.5px solid rgba(78,194,127,0.5);
}
/* a portrait frame (a phone held upright) would be 480 x 853 at full width: it is as tall as a square one instead */
.goal-frame img.tall { width: auto; max-width: 100%; max-height: 480px; }

/* goal marker on the scrubber */
.timeline .marker.goal {
  width: 3px; top: 0; bottom: 4px; background: var(--success); z-index: 7;
  border-radius: 2px; box-shadow: 0 0 6px rgba(78,194,127,0.85);
}
.timeline .marker.goal:hover { width: 5px; z-index: 8; }

/* key-event flag module inside the bottom segment overlay */
.video-overlay .vo-key { display: block; margin-bottom: 6px; }
.video-overlay .vo-key-badge {
  display: inline-block; margin-right: 8px; vertical-align: 1px;
  font-family: var(--mono); font-size: 10px; font-weight: 700; letter-spacing: 0.02em;
  color: #eef0e9; background: rgba(238,240,233,0.14);
  padding: 2px 9px; border-radius: var(--r-pill); border: 1px solid rgba(238,240,233,0.32);
}
.video-overlay .vo-key-label { font-size: 12px; font-weight: 600; color: #f3f2ec; }
.video-overlay .vo-key-outcome {
  display: inline-block; margin-left: 8px; vertical-align: 1px;
  font-family: var(--mono); font-size: 10px; font-weight: 700;
  padding: 1px 8px; border-radius: var(--r-pill);
}
.video-overlay .vo-key-outcome.success { background: rgba(78,194,127,0.2); color: #7fd9a4; }
.video-overlay .vo-key-outcome.failure { background: rgba(229,140,154,0.2); color: #e58c9a; }
.video-overlay .vo-key-outcome.unclear { background: rgba(255,255,255,0.12); color: #d9dcd2; }

/* ---------- scene-graph synced snapshot ---------- */
/* One snapshot at a time, tracking the playhead, so it reads as the current
   scene state instead of a wall of every snapshot at once. Each relation is
   subject | relation-pill | object. */
.sg-snap .sg-when {
  font-family: var(--mono); font-size: 11px; color: var(--fg-3);
  margin-bottom: 8px;
}
.sg-rel {
  display: flex; align-items: center; flex-wrap: wrap; gap: 6px;
  padding: 5px 0; font-size: 12.5px; line-height: 1.4;
  border-top: 1px solid var(--border);
}
.sg-rel:first-of-type { border-top: none; }
.sg-rel .subj { color: var(--fg); }
.sg-rel .obj  { color: var(--fg-2); }
.sg-rel .relpill {
  font-family: var(--mono); font-size: 10px; font-weight: 600;
  color: var(--fg-2); background: rgba(36,36,31,0.10);
  border: 1px solid rgba(36,36,31,0.22);
  padding: 1px 7px; border-radius: var(--r-pill); white-space: nowrap;
}

/* cost + generation time under the prompt */
.prompt-meta {
  display: flex; flex-wrap: wrap; gap: 10px 22px;
  margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--border-strong);
}
.prompt-meta .pm { font-size: 13px; color: var(--fg-2); display: inline-flex; gap: 7px; align-items: baseline; }
.prompt-meta .pm .pm-k { color: var(--fg-3); }
.prompt-meta .pm b { color: var(--fg); font-weight: 700; font-family: var(--mono); font-size: 14px;
  white-space: nowrap; }
.prompt-meta .pm-green b { color: var(--fg); }

/* prompt variants under the prompt banner */
.task-variants { margin-top: 12px; border-top: 1px solid var(--border); padding-top: 10px; }
.task-variants .tv-label {
  font-size: 10px; color: var(--fg-faint); letter-spacing: 0.01em;
  font-weight: 600; margin-bottom: 6px;
}
.task-variants ul { margin: 0; padding-left: 16px; }
.task-variants li { font-size: 12.5px; color: var(--fg-2); line-height: 1.5; margin: 3px 0; }

/* delineation on paper: strong rules between the columns and under the strip, white panels on grey paper */
.cv-all { border-right: 3px solid var(--fg); }
aside.rail { border-right: 1px solid var(--border-strong); }
aside.left { border-right: 1px solid var(--border-strong); background: var(--bg); }
section.right { background: var(--bg); }
.ep-card { background: var(--raised); border: 1px solid var(--border); }
.ep-card:hover { border-color: var(--border-strong); }
.ep-card.active { border-color: var(--fg); box-shadow: inset 3px 0 0 var(--fg); background: var(--raised); }
.info-block { background: var(--raised); border: 1px solid var(--border); }
.prompt-banner { background: var(--raised); border: 1px solid var(--border); }
.ep-head { background: var(--bg); border-bottom: 1px solid var(--border-strong); }
h3.section { border-top: 1px solid var(--border-strong); }
.ep-search input, .rail-export { background: var(--raised); border-color: var(--border-strong); }
.timeline { border: 1px solid var(--border-strong); }

/* dataset tabs on paper: one clear hours bar per tab, and a single ink underline for the open tab */
.cv-cell { gap: 8px; }
.cv-cell.on { background: var(--raised); }
.cv-cell.on::after { height: 3px; background: var(--fg); bottom: 0; }

/* every check run on the episode: ours, then public-dataset-adapter's, in one separated section */
.ck { display: grid; gap: 14px; margin-bottom: 22px; }
.ck-block { border: 1px solid var(--border); border-radius: var(--r-md); background: var(--raised);
  padding: 10px 14px 8px; }
.ck-head { display: flex; align-items: baseline; justify-content: space-between; gap: 6px 14px; flex-wrap: wrap;
  padding-bottom: 6px; }
.ck-title { font: 700 13px/1.3 var(--sans); color: var(--fg); }
.ck-sum { font: 500 11px/1.3 var(--mono); color: var(--fg-3); }
.ck-title a { color: inherit; text-underline-offset: 3px; }
.ck-row { display: grid; grid-template-columns: 14px minmax(0, 1fr) auto; align-items: baseline; gap: 2px 8px;
  padding: 5px 0; border-top: 1px solid var(--border); font-size: 12.5px; }
.ck-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--border-strong); transform: translateY(-1px); }
.ck-dot.issue { background: var(--danger); }
.ck-dot.note { background: transparent; box-shadow: inset 0 0 0 1.5px var(--fg-2); }
.ck-dot.na { background: transparent; box-shadow: inset 0 0 0 1px var(--border-strong); }
/* a check that stopped with an error: a fault of ours, never of the data, so no crimson */
.ck-dot.err { background: transparent; box-shadow: inset 0 0 0 1.5px var(--fg-3); }
.ck-name { color: var(--fg); min-width: 0; overflow-wrap: anywhere; }
.ck-row.clear .ck-name, .ck-row.na .ck-name { color: var(--fg-2); }
.ck-st { font: 500 11px/1 var(--mono); color: var(--fg-3); white-space: nowrap; }
.ck-row.issue .ck-st { color: var(--danger); }
.ck-text { grid-column: 2 / -1; font-size: 12px; line-height: 1.45; color: var(--fg-3); }
.ck-group { font: 500 12px/1.3 var(--sans); color: var(--fg-3); padding: 12px 0 4px; }
.ck-all { display: grid; grid-template-rows: 0fr; transition: grid-template-rows 320ms cubic-bezier(.32,.72,0,1); }
.ck-all-in { overflow: hidden; min-height: 0; opacity: 0; transition: opacity 240ms ease; }
.ck-theirs.open .ck-all { grid-template-rows: 1fr; }
.ck-theirs.open .ck-all-in { opacity: 1; }
.ck-more { margin-top: 6px; padding: 4px 0; background: none; border: 0; font: 500 12px/1.3 var(--sans);
  color: #2f7f76; cursor: pointer; text-decoration: underline; text-underline-offset: 3px; }
@media (prefers-reduced-motion: reduce) { .ck-all, .ck-all-in { transition: none; } }
/* capture-check notes: quiet facts about the recording, never counted as problems */
.cq-notes { margin: -8px 0 22px; padding: 10px 14px; border: 1px solid var(--border); border-radius: var(--r-md);
  background: var(--raised); }
.cq-notes-k { font: 600 11px/1.2 var(--sans); color: var(--fg-3); margin-bottom: 6px; }
.cq-note { font-size: 12.5px; color: var(--fg-2); line-height: 1.5; padding: 3px 0; }

/* ---------- other models' labels (compare/metrics.py), kept apart from the board's own ---------- */
/* the band over the episode's two columns: it opens while the episode shows another model's labels (the "Labels by"
   control), an ink band that says whose labels are shown, and folds away on the board's own */
main { grid-template-rows: auto minmax(0, 1fr); }
aside.rail { grid-column: 1; grid-row: 1 / span 2; }
aside.left { grid-column: 2; grid-row: 2; height: auto; min-height: 0; }
section.right { grid-column: 3; grid-row: 2; min-height: 0; }
.src-bar { grid-column: 2 / span 2; grid-row: 1; min-width: 0; height: 0; overflow: hidden; box-sizing: content-box;
  transition: height 240ms ease, background-color 240ms ease, color 240ms ease, border-color 240ms ease;
  background: var(--raised); color: var(--fg); border-bottom: 1px solid transparent; position: relative; z-index: 12; }
.src-bar.on { border-bottom-color: var(--border-strong); }
/* side by side (1230 px up) the band runs over the rail's column too, above the rail, whose own block starts under
   it (alignHeads), so the rail keeps no empty strip beside the band; its text starts on the rail's left edge */
@media (min-width: 1230px) { .src-bar { grid-column: 1 / span 3; z-index: 21; } .src-bar .sb-row { padding-left: 10px; }
  /* the line under the dataset tabs sits 1px above their row's edge; the open band meets it */
  .src-bar.on { margin-top: -1px; } }
.sb-row { display: flex; align-items: center; gap: 14px; padding: 10px 16px; min-height: 52px; box-sizing: border-box; }
.sb-note { min-width: 0; flex: 1; font: 400 12.5px/1.45 var(--sans); color: var(--fg-2); transition: color 240ms ease; }
.sb-note b { font-weight: 600; color: inherit; }
.sb-short { display: none; }
.sb-back { flex: none; font: 500 12px/1 var(--sans); padding: 8px 12px; cursor: pointer; border-radius: var(--r-md);
  background: transparent; color: inherit; border: 1px solid currentColor; opacity: 0; pointer-events: none;
  transition: opacity 200ms ease; }
.src-bar.cmp { background: var(--fg); color: #f3f2ec; border-bottom-color: var(--fg); }
.src-bar.cmp .sb-note { color: #f3f2ec; }
.src-bar.cmp .sb-back { opacity: 1; pointer-events: auto; }
.src-bar.cmp .sb-back:hover { background: rgba(243,242,236,0.12); }
/* the annotation panels ease out and back in when the labels change; the footage does not */
#left-col > *, #right-col, .ep-head { transition: opacity 170ms ease; }
main.src-fade #left-col > :not(.video-wrap), main.src-fade #right-col { opacity: 0; }
/* a labeller that did not label the open episode opens another one: its footage and header ease too */
main.src-fade.ep-fade #left-col > .video-wrap, main.src-fade.ep-fade .ep-head { opacity: 0; }
/* whose labels the subtitle on the footage shows, while it shows a comparison */
.video-overlay .vo-src { display: block; margin-bottom: 5px; font: 600 10.5px/1.2 var(--sans); letter-spacing: .02em;
  color: rgba(255,255,255,0.78); }
/* a response that did not parse, was cut off or never came */
.cmp-fail { background: var(--raised); border: 1px solid var(--border-strong); border-radius: var(--r-md);
  padding: 16px 20px; margin-bottom: 20px; }
.cmp-fail h4 { margin: 0 0 8px; font: 700 15px/1.3 var(--sans); color: var(--fg); }
.cmp-fail p { margin: 0 0 10px; font-size: 13px; line-height: 1.5; color: var(--fg-2); }
.cmp-fail pre { margin: 0; max-height: 320px; overflow: auto; padding: 10px 12px; background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--r-sm); font: 400 11.5px/1.5 var(--mono); color: var(--fg-2); white-space: pre-wrap;
  overflow-wrap: anywhere; }
/* a goal frame a static build has not extracted yet: the caption stays, the broken image does not */
.goal-frame.nofr img { display: none; }
.cmp-fail .cf-k { font: 600 11px/1.3 var(--sans); color: var(--fg-3); margin: 12px 0 6px; }

/* ---------- the comparison view ---------- */
#cmp-view { display: none; height: calc(100vh - var(--top-h)); overflow-y: auto; background: var(--bg); }
body.view-cmp main { display: none; }
body.view-cmp #cmp-view { display: block; }
#cmp-view, main { transition: opacity 180ms ease; }
body.view-fade #cmp-view, body.view-fade main { opacity: 0; }
.cmpv { max-width: 1320px; margin: 0 auto; padding: 28px 28px 64px; }
.cmpv-head h2 { margin: 0 0 12px; font: 700 24px/1.2 var(--sans); letter-spacing: -0.01em; color: var(--fg); }
.cmpv-head p { margin: 0 0 12px; max-width: 76ch; font-size: 14px; line-height: 1.6; color: var(--fg-2); }
.cmpv-head p:last-child { margin-bottom: 0; }
.cmpv-bar { position: sticky; top: 0; z-index: 5; display: flex; align-items: center; flex-wrap: wrap; gap: 10px 16px;
  margin: 24px -28px 0; padding: 12px 28px; background: var(--bg); border-bottom: 1px solid var(--border-strong); }
.cmpv-bar .if-sev-seg button { font-size: 13px; padding: 7px 12px; }
.cmpv-scope { font: 500 12px/1.4 var(--mono); color: var(--fg-3); }
.cmpv h3.section { margin-top: 46px; }
.cmpv-sub { margin: -4px 0 20px; max-width: 80ch; font-size: 13px; line-height: 1.55; color: var(--fg-3); }
.cmpv-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(480px, 1fr)); gap: 20px; }
.cmpv-bar + h3.section { border-top: 0; padding-top: 0; margin-top: 32px; }
.cc { background: var(--raised); border: 1px solid var(--border); border-radius: var(--r-md); padding: 16px 18px 14px;
  min-width: 0; }
.cc h4 { margin: 0 0 4px; font: 700 14px/1.3 var(--sans); color: var(--fg); }
.cc .cc-def { margin: 0 0 14px; font-size: 12px; line-height: 1.45; color: var(--fg-3); }
.cc .cc-foot { margin: 12px 0 0; font: 500 11px/1.4 var(--mono); color: var(--fg-3); }
.cc-wide { grid-column: 1 / -1; }
/* the main charts: each card's title, definition, key, bars and footnote are rows the cards in one grid row share, so
   two cards side by side start their bars on one line even when one has no definition or key */
.cmpv-main { row-gap: 0; }
.cmpv-main > .cc { display: grid; grid-template-rows: subgrid; grid-row: span 5; row-gap: 0; align-content: start;
  margin-bottom: 20px; }
.cmpv-main .cc-def:empty, .cmpv-main .cc-key:empty, .cmpv-main .cc-foot:empty { margin: 0; }
/* bar rows: one row per model, the reference model first and ruled off from the comparisons */
.br { display: grid; grid-template-columns: minmax(120px, 190px) minmax(0, 1fr) 88px; align-items: center;
  column-gap: 12px; padding: 5px 0; }
.br.ref { padding-bottom: 9px; margin-bottom: 4px; border-bottom: 1px solid var(--border); }
.br-name { font-size: 12.5px; line-height: 1.3; color: var(--fg-2); overflow-wrap: anywhere; }
.br.ref .br-name { color: var(--fg); font-weight: 600; }
.br-track { position: relative; height: 14px; }
.br-bar { position: absolute; left: 0; top: 0; bottom: 0; display: flex; gap: 2px;
  transition: width 420ms cubic-bezier(.2,.7,.2,1); min-width: 2px; }
.br-bar i { display: block; height: 100%; border-radius: 0 4px 4px 0;
  transition: flex-grow 420ms cubic-bezier(.2,.7,.2,1), opacity 300ms; flex-basis: 0; min-width: 0; }
.br-bar i + i { border-radius: 0 4px 4px 0; }
.br-bar i.seg0:not(:last-child) { border-radius: 0; }
.br-val { font: 600 12.5px/1.2 var(--mono); color: var(--fg); text-align: right; white-space: nowrap; }
.br-val small { display: block; font: 500 10.5px/1.3 var(--mono); color: var(--fg-3); margin-top: 2px; }
.c-ink { background: var(--fg-2); }
.c-red { background: var(--danger); } .c-red-l { background: rgba(179,38,60,0.30); }
.c-amb { background: #4d5aa0; } .c-amb-l { background: rgba(77,90,160,0.32); }
.c-grn { background: var(--success); }
.cc-key { display: flex; gap: 14px; flex-wrap: wrap; margin: -6px 0 12px; font-size: 11.5px; color: var(--fg-3); }
.cc-key span { display: inline-flex; align-items: center; gap: 6px; }
.cc-key i { display: inline-block; width: 12px; height: 8px; border-radius: 2px; }
/* agreement matrices: every pair of models, fill strength is agreement */
.mx-wrap { overflow-x: auto; }
table.mx { border-collapse: separate; border-spacing: 2px; font-variant-numeric: tabular-nums; }
table.mx th { font: 500 11px/1.25 var(--sans); color: var(--fg-2); text-align: left; padding: 4px 6px;
  vertical-align: bottom; min-width: 84px; max-width: 110px; }
table.mx th.mx-rh { text-align: right; vertical-align: middle; min-width: 150px; max-width: 200px;
  white-space: normal; }
table.mx th small, .br-name small { display: block; color: var(--fg-3); font-size: 10.5px; font-weight: 400; }
table.mx th.ref { color: var(--fg); font-weight: 700; }
table.mx td { height: 44px; min-width: 84px; text-align: center; border-radius: 3px; font: 600 12.5px/1.1 var(--mono);
  transition: background-color 380ms ease, color 380ms ease; }
table.mx td small { display: block; font: 500 10px/1.2 var(--mono); opacity: 0.8; margin-top: 3px; }
table.mx td.self { background: repeating-linear-gradient(135deg, rgba(28,28,26,0.05) 0 4px, transparent 4px 8px); }
table.mx td.na { color: var(--fg-3); font-weight: 500; background: rgba(28,28,26,0.03); }
.mx-scale { display: flex; align-items: center; gap: 8px; margin-top: 12px; font: 500 11px/1 var(--mono);
  color: var(--fg-3); }
.mx-scale i { width: 160px; height: 8px; border-radius: 2px;
  background: linear-gradient(90deg, rgba(28,28,26,0.04), rgba(28,28,26,0.86)); }
/* the paired view: each model without the example (open mark) and with it (filled), the reference model on the same
   episodes (rule) */
.pr { display: grid; grid-template-columns: minmax(120px, 170px) minmax(0, 1fr) 96px; align-items: center;
  column-gap: 12px; padding: 7px 0; }
.pr-track { position: relative; height: 22px; }
.pr-track::before { content: ""; position: absolute; left: 0; right: 0; top: 50%; height: 1px;
  background: var(--border); }
.pr-line { position: absolute; top: calc(50% - 1px); height: 2px; background: var(--fg-2);
  transition: left 420ms cubic-bezier(.2,.7,.2,1), width 420ms cubic-bezier(.2,.7,.2,1); }
.pr-dot { position: absolute; top: 50%; width: 12px; height: 12px; margin: -6px 0 0 -6px; border-radius: 50%;
  box-sizing: border-box;
  transition: left 420ms cubic-bezier(.2,.7,.2,1), opacity 300ms; box-shadow: 0 0 0 2px var(--raised); }
.pr-dot.without { background: var(--raised); border: 2px solid var(--fg-2); }
.pr-dot.with { background: var(--fg); }
.pr-ref { position: absolute; top: -4px; bottom: -4px; width: 2px; margin-left: -1px; background: var(--fg-disabled);
  opacity: 0.9;
  transition: left 420ms cubic-bezier(.2,.7,.2,1), opacity 300ms; }
.pr-val { font: 600 12px/1.25 var(--mono); color: var(--fg); text-align: right; white-space: nowrap; }
.pr-val small { display: block; font: 500 10.5px/1.3 var(--mono); color: var(--fg-3); }
.pr-key { display: flex; gap: 16px; flex-wrap: wrap; margin: 0 0 16px; font-size: 12px; color: var(--fg-3); }
.pr-key span { display: inline-flex; align-items: center; gap: 7px; }
.pr-key .k-open { width: 10px; height: 10px; border-radius: 50%; border: 2px solid var(--fg-2); }
.pr-key .k-fill { width: 12px; height: 12px; border-radius: 50%; background: var(--fg); }
.pr-key .k-ref { width: 2px; height: 14px; background: #8c8b86; }
/* episodes: full ID and dataset, each model's outcome, each a link to the episode with that source */
.et-wrap { overflow-x: auto; background: var(--raised); border: 1px solid var(--border); border-radius: var(--r-md); }
table.et { width: 100%; border-collapse: collapse; font-size: 12.5px; }
table.et th { position: sticky; top: 0; background: var(--raised); text-align: left; font: 600 11px/1.3 var(--sans);
  color: var(--fg-3);
  padding: 9px 10px; border-bottom: 1px solid var(--border-strong); vertical-align: bottom; }
table.et th small { display: block; font-weight: 400; }
table.et td { padding: 7px 10px; border-bottom: 1px solid var(--row-divider); vertical-align: top; }
table.et tr:last-child td { border-bottom: 0; }
table.et td.et-id { font: 500 11.5px/1.4 var(--mono); overflow-wrap: anywhere; min-width: 220px; max-width: 340px; }
table.et td.et-id a { color: var(--fg); text-decoration: none; }
table.et td.et-id a:hover, table.et td a.et-o:hover { text-decoration: underline;
  text-decoration-color: var(--accent); }
table.et td.et-ds { color: var(--fg-2); white-space: nowrap; }
table.et td.et-len { font-family: var(--mono); color: var(--fg-3); white-space: nowrap; }
table.et td.ref-col { border-right: 1px solid var(--border); }
table.et a.et-o { font: 600 11px/1.2 var(--mono); text-decoration: none; white-space: nowrap; padding: 2px 7px;
  border-radius: var(--r-pill);
  border: 1px solid transparent; display: inline-block; }
.et-o.success { color: var(--success); background: rgba(78,194,127,0.14);
  border-color: rgba(78,194,127,0.30) !important; }
.et-o.failure, .et-o.success_then_undone { color: var(--danger); background: rgba(179,38,60,0.08);
  border-color: rgba(179,38,60,0.30) !important; }
.et-o.partial, .et-o.unclear, .et-o.tasks { color: var(--fg-2); background: rgba(28,28,26,0.050);
  border-color: var(--border) !important; }
.et-o.fail { color: var(--fg-2); border: 1px dashed var(--border-strong) !important; background: transparent; }
.et-none { color: var(--fg-disabled); font: 500 11px/1.2 var(--mono); }
/* one tooltip for every chart mark: the value first, then what it is */
.cmp-tip { position: fixed; z-index: 90; pointer-events: none; max-width: 300px; padding: 8px 10px;
  border-radius: var(--r-sm);
  background: var(--fg); color: #f3f2ec; font-size: 12px; line-height: 1.4; box-shadow: 0 6px 20px rgba(0,0,0,0.3);
  opacity: 0; transition: opacity 120ms ease; }
.cmp-tip.show { opacity: 1; }
.cmp-tip b { display: block; font: 600 13px/1.3 var(--mono); }
@media (max-width: 1229px) {
  main { display: block; }
  .src-bar { position: sticky; top: var(--header-h); }
  aside.left, section.right { height: auto; }
  #cmp-view { height: auto; overflow: visible; }
  .cmpv { padding: 22px 16px 48px; }
  .cmpv-bar { margin: 20px -16px 0; padding: 10px 16px; top: var(--header-h); }
}
@media (max-width: 599px) {
  .cmpv-grid { grid-template-columns: minmax(0, 1fr); }
  .br { grid-template-columns: minmax(96px, 120px) minmax(0, 1fr) 84px; column-gap: 10px; }
  /* a pair's value ("100% → 100%") is wider than a bar's: its column fits it, and the track gives the room */
  .pr { grid-template-columns: minmax(96px, 120px) minmax(0, 1fr) max-content; column-gap: 10px; }
  .sb-row { flex-wrap: wrap; gap: 8px 12px; padding: 10px 16px; }
  .sb-note { flex-basis: 100%; order: 3; }
  .sb-long { display: none; } .sb-short { display: inline; }
  .sb-back { order: 2; }
  .cmpv-bar .if-sev-seg { flex-wrap: wrap; border-radius: 16px; }
  .cmpv-head h2 { font-size: 21px; }
}

/* narrow screens (tablets, phones, small laptops): the three columns stack. The tabs scroll sideways instead of
   cutting their names off, the episode list scrolls inside a capped box, and the episode and its timeline follow
   as part of the page. Three columns start where the side panel gets at least 300 px (240 + 56vw + 300 <= 1230);
   narrower, its timeline rows ran the arm name into the contribution chip. */
@media (max-width: 1229px) {
  .coverage { position: relative; top: 0; height: auto; grid-template-columns: 1fr; }
  .cv-all { flex-direction: row; align-items: baseline; justify-content: space-between; gap: 12px; padding: 12px 16px;
    border-right: 0; border-bottom: 1px solid var(--border-strong); }
  .cv-cells { grid-auto-columns: max-content; overflow-x: auto; scrollbar-width: thin; }
  .cv-cell { padding: 11px 16px; }
  .cv-name, .cv-num { overflow: visible; text-overflow: clip; }
  .cv-num { font-size: 11px; }
  .cv-num .u { display: inline; }
  main { display: block; height: auto; }
  aside.rail { position: static; height: auto; overflow: visible; --rail-pad: 16px; padding: 14px var(--rail-pad); border-right: 0;
    border-bottom: 1px solid var(--border-strong); }
  #ep-list { max-height: 46vh; }
  .if-menu { width: min(340px, calc(100vw - 32px)); }
  aside.left { position: static; height: auto; overflow: visible; border-right: 0;
    border-bottom: 1px solid var(--border-strong); }
  .ep-head { position: static; }
  section.right { overflow: visible; padding: 20px 16px; }
  /* in one column a long list scrolls inside its own panel, so the page never runs on for screens */
  #feed, .key-events, .di-block, #left-col > .info-block, .inv-list { max-height: 60vh; overflow-y: auto;
    overscroll-behavior: contain; }
}
@media (max-width: 599px) {
  /* on a phone the band naming the comparison is pinned right above the footage, and the small player has no room for
     a second line in its subtitle */
  .video-overlay .vo-src { display: none; }
  .ep-head { flex-wrap: wrap; row-gap: 10px; }
  .ep-head-side { flex: 1 0 100%; align-items: stretch; }
  .ep-head .ep-head-acts { flex-direction: row; flex-wrap: wrap; align-items: center; gap: 6px; }
  /* the buttons wrap to the pane's left edge here, so the video menu opens from its button's left */
  .vd-menu { right: auto; left: 0; }
  .kp-note-in { text-align: left; }
  .video-overlay { min-width: 0; max-width: 94%; padding: 7px 10px; font-size: 12px; }
  aside.left .cam-cell.cam-wrist video { max-height: none; }
  #ep-list { max-height: 36vh; }
}

</style>
</head><body>
<header class="page-head"><h1>__PAGE_TITLE__</h1><span class="ph-board">__BOARD_NAME__</span></header>
<nav class="coverage" id="coverage" aria-label="What is labelled on this dashboard"></nav>
<main>
  <aside class="rail" id="rail">
    <div class="lb" id="lb" hidden>
      <span class="rail-k" id="lb-k">Labels by</span>
      <button class="lb-btn" id="lb-btn" type="button" aria-haspopup="listbox" aria-expanded="false"
        aria-labelledby="lb-k lb-name">
        <span class="lb-lab"><span id="lb-name"></span></span>
        <svg class="lb-caret" width="12" height="12" viewBox="0 0 12 12"><path d="M2 4l4 4 4-4" stroke="currentColor"
          stroke-width="1.8" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </button>
      <div class="lb-note off" id="lb-note"><div class="lb-note-in" id="lb-note-in"></div></div>
      <div class="lb-menu" id="lb-menu" role="listbox" aria-label="Whose labels the dashboard shows"></div>
    </div>
    <div class="rail-ds" id="rail-ds"><span class="rd-name" id="rd-name"></span><span class="rd-sub"
      id="rd-sub"></span></div>
    <div class="ep-search" id="ep-search">
      <svg class="es-ico" width="14" height="14" viewBox="0 0 16 16"><circle cx="7" cy="7" r="4.8" fill="none"
        stroke="currentColor" stroke-width="1.7"/><path d="M10.6 10.6L14 14" stroke="currentColor" stroke-width="1.7"
        stroke-linecap="round"/></svg>
      <input id="ep-search-q" type="text" placeholder="Search by episode ID" autocomplete="off" spellcheck="false"
        aria-label="Search this dataset by episode ID">
      <button class="es-clear" id="ep-search-clear" title="Clear search" aria-label="Clear search">&times;</button>
    </div>
    <div id="issue-filter" class="issue-filter"></div>
    <div id="ep-list"></div>
    <div class="rail-dl" id="rail-dl"><div class="rail-dl-in"><div class="dl-box">
      <span class="rail-k" id="dl-k">Download the episodes shown</span>
      <div class="dl-row">
        <button id="export-jsonl" class="rail-export" type="button"
          title="every episode in the list below (dataset, filter and search) as JSON Lines, one episode per line"
          >JSON Lines</button>
        <button id="export-kp" class="rail-export" type="button" hidden
          title="the hand keypoints of every episode in the list below, one episode per line, in a file of their own"
          >Hand keypoints</button>
      </div>
      <p class="dl-lic" id="dl-lic" hidden></p>
      <div class="kp-exp off" id="kp-exp"><div class="kp-exp-in">
        <p class="kp-lic"><span id="kp-size"></span>Non-commercial use only. Predicted by <a
          href="https://huggingface.co/acerobotics2025/ACE-Ego-Hand" target="_blank" rel="noreferrer">ACE-Ego-Hand</a>
          (<a href="https://creativecommons.org/licenses/by-nc/4.0/" target="_blank" rel="noreferrer">CC BY-NC 4.0</a>),
          which uses <a href="https://mano.is.tue.mpg.de/license.html" target="_blank" rel="noreferrer">MANO</a>.</p>
      </div></div>
    </div></div></div>
  </aside>
  <div class="src-bar" id="src-bar"><div class="sb-in"><div class="sb-row">
    <span class="sb-note" id="sb-note"></span>
    <button class="sb-back" id="sb-back" type="button" tabindex="-1"></button>
  </div></div></div>
  <aside class="left" id="left-pane">
    <div class="ep-head">
      <div class="ep-head-name"><span class="ep-head-k">Episode</span><span id="current-ep"></span><span
        id="current-ep-raw"></span><span id="current-ep-src"></span><div id="current-ep-reader"></div></div>
      <div class="ep-head-side">
        <div class="ep-head-acts">
          <button id="hp-btn" class="ep-head-dl hp-btn" type="button" aria-pressed="true" hidden
            title="Draw the joints of each hand over the video"><span class="hp-sw"
            aria-hidden="true"></span>Hand pose</button>
          <a id="kp-dl" class="ep-head-dl" download hidden
            title="the model's 2D hand keypoints for every frame of this episode's video, as JSON, in a file of their own"
            >Hand keypoints</a>
          <div class="vd" id="vd" hidden>
            <button id="vd-btn" class="ep-head-dl vd-btn" type="button" aria-haspopup="menu" aria-expanded="false"
              title="Download this episode's video, with every camera in one frame">Video<svg class="vd-caret"
              width="10" height="10" viewBox="0 0 12 12" aria-hidden="true"><path d="M2 4l4 4 4-4" stroke="currentColor"
              stroke-width="1.8" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg></button>
            <div class="vd-menu" id="vd-menu" role="menu"></div>
          </div>
          <a id="dl-json" class="ep-head-dl" download
            title="this episode's full annotation and dataset checks as JSON">Episode JSON</a>
        </div>
        <div class="kp-note off" id="kp-note"><div class="kp-note-in">Hand keypoints for non-commercial use only. Predicted
          by <a href="https://huggingface.co/acerobotics2025/ACE-Ego-Hand" target="_blank"
          rel="noreferrer">ACE-Ego-Hand</a> (<a href="https://creativecommons.org/licenses/by-nc/4.0/" target="_blank"
          rel="noreferrer">CC BY-NC 4.0</a>), which uses <a href="https://mano.is.tue.mpg.de/license.html"
          target="_blank" rel="noreferrer">MANO</a>.</div></div>
      </div>
    </div>
    <div id="left-col"></div>
  </aside>
  <section class="right" id="right-col"></section>
</main>
<section id="cmp-view" aria-label="Model comparison"></section>
<div class="cmp-tip" id="cmp-tip" role="tooltip"></div>
<script>
const currentEp = document.getElementById('current-ep');

function fmtTok(n) { return n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1) + 'k' : String(n); }
function fmtDur(s) { return s == null ? '' : (s >= 60 ? (s / 60).toFixed(1) + ' min' : Math.round(s) + 's'); }
// what "both" means on the open episode's rig: a head camera films a person's hands, a handheld rig two grippers
let ARM_NOUN = 'arms';
function armLabel(a) { return a === 'both' ? `both ${ARM_NOUN}` : (a || '?'); }
function contribClass(c) {
  c = (c || '').toLowerCase();
  return c === 'advancing' ? 'adv' : (c === 'wasteful' ? 'waste' : (c === 'idle' ? 'idle' : 'none'));
}
const epListEl = document.getElementById('ep-list');
const issueFilterEl = document.getElementById('issue-filter');
const leftCol = document.getElementById('left-col');
const rightCol = document.getElementById('right-col');

// every string the page shows goes through here; the model's en and em dashes read as plain hyphens on the page
// (the episode JSON keeps the model's text as written)
function esc(s) { return String(s ?? '').replace(/[\u2013\u2014]/g, '-').replace(/[&<>"]/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function isNullish(v) { return v == null || (typeof v === 'string' && v.toLowerCase() === 'null'); }
function fmtT(t) { return (t == null) ? '-' : t.toFixed(1) + 's'; }
// a key event's kind as the model tagged it, in plain words ("subgoal_complete" reads "subgoal complete")
function kindName(k) { return String(k || '').replace(/_/g, ' ').trim(); }

let _activeFile = null;
// the files the rail lists, in order; the cards themselves are built in batches (renderRail), so anything that needs
// the whole list (the exports) reads this
let _railFiles = [];
let _railJob = 0;
const RAIL_FIRST = 40, RAIL_BATCH = 120;   // cards built at once, then per idle slice
let ALL_EPS = [];

// ---- data source ----
// This page runs two ways from this one template. Served by board/serve.py it reads the /api/*
// endpoints. Written out by board/static.py it reads plain files and needs no server:
// data/index.json (every episode's file, dataset and length, for the tabs, totals and search),
// data/lists/<dataset>.json (the full rail records of one dataset, fetched when its tab opens),
// data/ep/<file> (each episode file exactly as stored), data/compare/ (other models' labels, when BOARD.compare),
// data/hands/<file> (the hand pose drawn over a head camera, when BOARD.hands), data/keypoints/index.json (the hand
// keypoint downloads, when BOARD.keypoints) and web copies of the clips, goal frames and keypoint downloads under
// the media base, whose paths each rail record carries in _media and _frames. Relative bases keep a
// static build working under any sub-path and from any origin. BOARD.models names the pinned models
// (configs/models.json) by model id.
const BOARD = __BOARD_CONFIG__;
const STATIC = BOARD.mode === 'static';
// the name of the model whose labels an episode file holds: its display name, else its id without the provider
function modelName(id) {
  if (!id) return 'The model';
  return (BOARD.models || {})[id] || String(id).split('/').pop();
}

// ---- issue tags: plain names (tag_names.json) ----
const TAG_NAMES = __TAG_NAMES__;
function tagName(t, list) {
  const e = (TAG_NAMES[list] || {})[t] || (TAG_NAMES.data_issues || {})[t] || (TAG_NAMES.operator_mistakes || {})[t];
  if (e && e.name) return e.name;
  const w = String(t || '').replace(/_/g, ' ');
  return w.charAt(0).toUpperCase() + w.slice(1);
}
// whether an issue counts: the server marks each one (episode_view, from Families.counts); the rest stay visible,
// marked minor
const countsIssue = i => i.counted === true;
const _dsLoaded = new Set();
async function loadAllEpisodes() {
  if (!STATIC) return (await fetch('api/episodes', {cache: 'no-store'})).json();
  const r = await fetch(BOARD.data + 'index.json');
  if (!r.ok) throw new Error('index.json: ' + r.status);
  const idx = await r.json();
  // [dataset index, file, duration_s, episode_id if it is not the file name without .json]
  return idx.eps.map(([di, file, dur, id]) => ({episode_id: id || file.replace(/\.json$/, ''), file,
    dataset: idx.datasets[di], duration_s: dur}));
}
// static: swap one dataset's index stubs for its full rail records (once per dataset)
async function ensureDataset(ds) {
  if (!STATIC || _dsLoaded.has(ds)) return;
  const r = await fetch(BOARD.data + 'lists/' + encodeURIComponent(ds) + '.json');
  if (!r.ok) throw new Error('list ' + ds + ': ' + r.status);
  const full = new Map((await r.json()).map(e => [e.file, e]));
  ALL_EPS = ALL_EPS.map(e => full.get(e.file) || e);
  _dsLoaded.add(ds);
}
function episodeUrl(file) {
  return STATIC ? BOARD.data + 'ep/' + encodeURIComponent(file) : 'api/episode?file=' + encodeURIComponent(file);
}
// each episode's labels are fetched once per page view (a dataset tab fetches its first episode ahead, prefetchDataset)
const _epCache = new Map();          // file -> Promise of the labels, the most recently used last
const EP_CACHE_MAX = 60;
async function fetchEpisodeOnce(file) {
  if (!STATIC) return (await fetch(episodeUrl(file), {cache: 'no-store'})).json();
  const rec = ALL_EPS.find(e => e.file === file);
  if (rec) await ensureDataset(datasetOf(rec));
  return (await fetch(episodeUrl(file))).json();
}
function fetchEpisode(file) {
  let p = _epCache.get(file);
  if (p) { _epCache.delete(file); _epCache.set(file, p); return p; }
  p = fetchEpisodeOnce(file);
  p.catch(() => _epCache.delete(file));
  loadSensors(file);            // the episode's sensors file, when it has one, fetched beside its labels
  _epCache.set(file, p);
  while (_epCache.size > EP_CACHE_MAX) _epCache.delete(_epCache.keys().next().value);
  return p;
}
// other models' labels (compare/, never part of the board's lists or downloads)
function compareUrl(what) { return STATIC ? BOARD.data + 'compare/' + what + '.json' : 'api/compare/' + what; }
function compareEpUrl(key, file) {
  return STATIC ? BOARD.data + 'compare/' + encodeURIComponent(key) + '/' + encodeURIComponent(file)
    : 'api/compare/episode?key=' + encodeURIComponent(key) + '&file=' + encodeURIComponent(file);
}
// one model's rail records: the same records the board's own list is made of, from that model's labels
function cmpListUrl(key) {
  return STATIC ? BOARD.data + 'compare/lists/' + encodeURIComponent(key) + '.json'
    : 'api/compare/list?key=' + encodeURIComponent(key);
}
const CMP_LISTS = {};         // key -> Promise of that model's rail records (null when they cannot be had)
function loadCmpList(key) {
  if (!CMP_LISTS[key]) CMP_LISTS[key] = fetchJson(cmpListUrl(key)).then(r => Array.isArray(r) ? r : null);
  return CMP_LISTS[key];
}
// whose labels the board shows (the "Labels by" control): null is the board's own; otherwise one other model's, as a
// comparison. The rail, the strip under the header and the filter describe the episodes that labeller labelled.
let BY = null, BY_EPS = null;
function railEps() { return BY ? (BY_EPS || []) : ALL_EPS; }
// the hand keypoints download of a head-camera episode (hand_keypoints/, in the dataset video's own pixels and
// times); a static build lists them in data/keypoints/index.json and keeps the files with the media, named by their
// content
const kpIndexUrl = () => STATIC ? BOARD.data + 'keypoints/index.json' : 'api/keypoints?file=index.json';
function keypointsUrl(file) {
  return STATIC ? BOARD.media + ((KP_INDEX || {})[file] || {}).path : 'api/keypoints?file=' + encodeURIComponent(file);
}
function keypointsDownloadUrl(file) { return STATIC ? keypointsUrl(file) : keypointsUrl(file) + '&download=1'; }
let KP_INDEX = null;          // {file: {frames, bytes}}: the episodes that have one
function fmtBytes(b) {
  return b >= 1e9 ? (b / 1e9).toFixed(1) + ' GB' : b >= 1e6 ? (b / 1e6).toFixed(b >= 1e8 ? 0 : 1) + ' MB'
    : Math.max(1, Math.round(b / 1e3)) + ' KB';
}
async function fetchJson(url) {
  try {
    const r = await fetch(url, STATIC ? {} : {cache: 'no-store'});
    return r.ok ? await r.json() : null;
  } catch (e) { return null; }
}
function episodeDownloadUrl(file) { return STATIC ? episodeUrl(file) : episodeUrl(file) + '&download=1'; }
// the static build's media key of a camera value: left, right, extra1, extra2, ... and a camera's depth clip
// (depth_exo, depth_left, ...) are their own, anything else the main camera (board/static.py media_key)
function mediaKey(cam) {
  return (cam === 'left' || cam === 'right' || /^(extra|unshown)\d{1,2}$/.test(cam)
    || /^depth_(exo|left|right|extra\d{1,2})$/.test(cam)) ? cam : 'exo';
}
// web copy of one camera's clip; cam is the value the page asks /api/video for (left, right, or the top camera)
function videoSrc(eidEnc, cam) {
  if (!STATIC) return 'api/video?id=' + eidEnc + '&cam=' + cam;
  const rec = ALL_EPS.find(e => e.file === _activeFile) || {};
  return BOARD.media + ((rec._media || {})[mediaKey(cam)] || '');
}
// the camera's first frame as the video's poster attribute, or nothing where a static build has no such frame
function posterAttr(eidEnc, cam) {
  const src = posterSrc(_activeFile, eidEnc, cam);
  return src ? ` poster="${src}"` : '';
}
function posterSrc(file, eidEnc, cam) {
  if (STATIC) {
    const rec = ALL_EPS.find(e => e.file === file) || {};
    if (!(rec._frames || {})[mediaKey(cam) + '|0']) return '';
  }
  return frameSrcOf(file, eidEnc, cam, 0);
}
// a goal frame that loads taller than it is wide gets .tall, which caps its height (landscape ones are untouched)
function markTallGoalFrames() {
  document.querySelectorAll('.goal-frame img').forEach(im => {
    // kept for every load: the task goal frame shows another picture as the video plays
    const mark = () => im.classList.toggle('tall', im.naturalHeight > im.naturalWidth);
    im.addEventListener('load', mark);
    if (im.complete && im.naturalWidth) mark();
  });
}
function frameSrc(eidEnc, cam, t) { return frameSrcOf(_activeFile, eidEnc, cam, t); }
function frameSrcOf(file, eidEnc, cam, t) {
  if (!STATIC) return `api/frame?id=${eidEnc}&cam=${cam}&t=${t}&w=640`;
  const rec = ALL_EPS.find(e => e.file === file) || {};
  return BOARD.media + ((rec._frames || {})[mediaKey(cam) + '|'
    + Math.round(Number(t) * 1000)] || '');
}
// the cameras an episode shows: the main player's (the top camera when there is one, else the first gripper camera)
// and the side cells' (a head camera is shown alone)
function episodeCams(d) {
  const views = (d.camera_views && d.camera_views.length) ? d.camera_views : ['exo', 'left', 'right'];
  const main = views.includes('exo') ? 'exo' : views[0];
  return {views, main, side: d._rig === 'ego_head' ? [] : views.filter(v => v !== main && v !== 'exo')};
}
// the listed episodes as JSON Lines: the server joins them, a static build fetches and joins them here
async function exportBlob(files) {
  if (!STATIC) {
    const r = await fetch('api/export', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({files})});
    if (!r.ok) throw new Error('export failed: ' + r.status);
    return r.blob();
  }
  const lines = new Array(files.length);
  let next = 0;
  const worker = async () => {
    while (next < files.length) {
      const k = next++;
      const r = await fetch(episodeUrl(files[k]));
      if (!r.ok) throw new Error('export failed: ' + files[k] + ' ' + r.status);
      lines[k] = JSON.stringify(await r.json());
    }
  };
  await Promise.all(Array.from({length: Math.min(12, files.length)}, worker));
  return new Blob([lines.join('\n') + '\n'], {type: 'application/x-ndjson'});
}
function saveBlob(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
// a browser ignores <a download> across origins, so when a static build's data sits on another origin
// (the CDN) the episode JSON and the hand keypoints are fetched and saved instead of opened
if (STATIC) for (const id of ['dl-json', 'kp-dl']) document.getElementById(id).addEventListener('click', async (e) => {
  const a = e.currentTarget;
  if (new URL(a.href, location.href).origin === location.origin) return;
  e.preventDefault();
  try { saveBlob(await (await fetch(a.href)).blob(), a.getAttribute('download') || 'episode.json'); }
  catch (err) { alert(err.message); }
});

// The open episode's video to download (api/footage): every camera in one frame, the main one with the others beside
// it, for the whole episode or a span of it in seconds (set to the step under the playhead when the menu opens), made
// on the server the first time it is asked for; or one camera's own clip. Only a served board makes them.
const vdEl = document.getElementById('vd'), vdBtn = document.getElementById('vd-btn'),
  vdMenu = document.getElementById('vd-menu');
let _vdEp = null;            // the open episode: {eid, d}
function vdOpen(open) {
  if (open) buildVideoMenu();
  vdEl.classList.toggle('open', open);
  vdBtn.setAttribute('aria-expanded', open ? 'true' : 'false');
}
// the dense step at time t: {t0, t1, what}, or null before the first one
function vdStepAt(d, t) {
  const ev = (d.event_labels || []).filter(e => e && e.t_s != null).slice().sort((a, b) => a.t_s - b.t_s);
  let k = -1;
  for (let i = 0; i < ev.length; i++) if (ev[i].t_s <= t + 0.05) k = i;
  if (k < 0) return null;
  const e = ev[k];
  const t1 = (e.end_s != null && e.end_s > e.t_s) ? e.end_s : (ev[k + 1] ? ev[k + 1].t_s : e.t_s + 1);
  return {t0: Number(e.t_s), t1: Number(t1), what: e.verb_class || ''};
}
function buildVideoMenu() {
  if (!_vdEp) return;
  const {eid, d} = _vdEp, v = document.getElementById('video');
  const dur = (v && isFinite(v.duration) && v.duration > 0) ? v.duration : (d.duration_s || null);
  const st = vdStepAt(d, v ? v.currentTime : 0);
  const {views, main} = episodeCams(d);
  const cams = [main, ...views.filter(c => c !== main)];
  const hasTop = main === 'exo';
  // an extra camera is named as its cell is (camera_labels), never as a wrist
  const label = c => { const i = views.indexOf(c); return (d.camera_labels && i >= 0 && d.camera_labels[i]) || c; };
  const name = c => c === 'exo' ? 'exo' : (c !== 'left' && c !== 'right') ? label(c)
    : (hasTop ? `${c} wrist` : `${c} camera`);
  const eidEnc = encodeURIComponent(eid);
  vdMenu.innerHTML = `<button type="button" class="vd-opt" role="menuitem" data-whole="1">Whole episode<small>`
      + `${cams.length > 1 ? 'every camera in one frame' : 'the camera'}${dur ? `, 0.0s to ${fmtT(dur)}` : ''}</small>`
      + `</button>`
    + `<div class="vd-span">A part of it<small>${st ? `set to the step at the playhead: ${esc(st.what)}`
      : 'from and to, in seconds'}</small><div class="vd-span-row"><input id="vd-t0" inputmode="decimal" `
      + `aria-label="From, in seconds" value="${st ? st.t0.toFixed(1) : '0.0'}"> to <input id="vd-t1" `
      + `inputmode="decimal" aria-label="To, in seconds" value="${st ? st.t1.toFixed(1) : (dur ? dur.toFixed(1) : '')}">`
      + ` s<button type="button" class="vd-go">Download</button></div></div>`
    + (cams.length > 1 ? `<div class="vd-cams">Each camera on its own</div><div class="vd-cam-row">`
      + cams.map(c => `<a role="menuitem" href="api/video?id=${eidEnc}&cam=${c}&download=1" download>${esc(name(c))}`
        + `</a>`).join('') + `</div>` : '')
    + `<div class="vd-status" id="vd-status" role="status"></div>`;
}
async function vdDownload(t0, t1) {
  const status = document.getElementById('vd-status');
  const q = `api/footage?id=${encodeURIComponent(_vdEp.eid)}` + (t0 != null ? `&t0=${t0}` : '')
    + (t1 != null ? `&t1=${t1}` : '');
  status.textContent = 'Making the video. A long episode takes a few minutes.';
  try {
    const r = await fetch(q + '&prepare=1', {cache: 'no-store'});
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.error || ('HTTP ' + r.status));
    status.textContent = `Downloading ${j.name} (${fmtBytes(j.bytes)}).`;
    const a = document.createElement('a');
    a.href = q; a.download = j.name;
    document.body.appendChild(a); a.click(); a.remove();
  } catch (err) { status.textContent = 'The video could not be made: ' + err.message; }
}
vdBtn.addEventListener('click', (e) => { e.stopPropagation(); vdOpen(!vdEl.classList.contains('open')); });
vdMenu.addEventListener('click', (e) => {
  if (e.target.closest('[data-whole]')) { vdDownload(null, null); return; }
  if (e.target.closest('.vd-go')) {
    const t0 = parseFloat(document.getElementById('vd-t0').value), t1 = parseFloat(document.getElementById('vd-t1').value);
    if (!(t0 >= 0) || !(t1 > t0)) {
      document.getElementById('vd-status').textContent = 'The start must be a time before the end, in seconds.';
      return;
    }
    vdDownload(t0, t1);
  }
});
document.addEventListener('click', (e) => { if (!vdEl.contains(e.target)) vdOpen(false); });
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && vdEl.classList.contains('open')) { vdOpen(false); vdBtn.focus(); } });

// a readable name for an episode id, keeping every part of the id that identifies it (FastUMI's
// episode_<arm>__<task>__<index> reads "<task> #<index>"); any other id is shown as it is
function epName(id) {
  id = id || '';
  let m = /^episode_(?:dual_arm|single_arm)__(.+)__0*(\d+)$/.exec(id);                 // FastUMI
  if (m) return `${m[1].replace(/_/g, ' ')} #${m[2]}`;
  m = /^episode_(.+?)_(20\d{6})_(\d{3})_0*(\d+)$/.exec(id);                             // Galaxea
  if (m) return `${m[1].replace(/_/g, ' ')} \u00b7 ${m[2]}-${m[3]} #${m[4]}`;
  m = /^episode_factory_(\d+)_worker_(\d+)_0*(\d+)$/.exec(id);                          // Egocentric-100K
  if (m) return `Factory ${m[1]} \u00b7 worker ${m[2]} \u00b7 clip ${m[3]}`;
  m = /^episode_(.+?)_(\d{5})_0*(\d+)$/.exec(id);                                        // 10Kh-RealOmin
  if (m && /[A-Za-z]/.test(m[1])) return `${m[1].replace(/_/g, ' ')} \u00b7 ${m[2]} #${m[3]}`;
  return id;
}
// the dataset an episode belongs to: its manifest entry's name (board/build.py writes it on every file)
function datasetOf(ep) { return ep.dataset || ''; }

// ---- datasets: one tab each in the strip under the header ----
// Names for the datasets prepare/ can fetch; any other name (a user's own dataset) is shown as its words with each
// first letter capitalised.
const DS_LABELS = {molmo: 'MolmoAct2 (Molmo)', abc130k: 'ABC-130k (XDOF)', galaxea: 'Galaxea Open-World (R1 Lite)',
                   fastumi: 'FastUMI-100K', realomin: '10Kh-RealOmin (GenRobot)',
                   egocentric100k: 'Egocentric-100K (Build AI)', genhumanego: 'Gen-HumanEgo (GenRobot)',
                   habit: 'HABIT (Config)', openaoe: 'OpenAoE-2000h (inclusionAI)'};
// the tabs' order: these datasets first, then any other, alphabetically
const DS_ORDER = ['molmo', 'abc130k', 'galaxea', 'habit', 'fastumi', 'realomin', 'egocentric100k', 'genhumanego',
                  'openaoe'];
// where an episode's footage comes from and its license (board/dataset_sources.json, copied into each episode file
// by board/build.py); nothing for a dataset that has no entry, such as your own
function datasetSourceHtml(s) {
  if (!s || !s.name || !s.license) return '';
  const link = (url, text) => url ? `<a href="${esc(url)}" target="_blank" rel="noreferrer">${esc(text)}</a>`
    : esc(text);
  return `Footage: ${link(s.hub, s.name)}${s.publisher ? ` (${esc(s.publisher)})` : ''}, ${link(s.license_url,
    s.license)}`;
}
// The problems the episode was kept and flagged with (board/build.py reader_issues, from context.json): a camera whose
// video does not decode, which the episode is shown and labelled without, a camera whose clip has fewer frames than
// the episode, and any other kind a reader records. Each is a row of the recording checks card, in the sentence the
// entry carries, under the family it raises (board/families.py reader_family), with its camera or signal and its time
// when the entry has them. Nothing is drawn when the episode has none.
function readerIssueRows(d) {
  const ri = (d && d.dataset_checks || {}).reader_issues;
  const num = v => v != null && v !== '' && !isNaN(parseFloat(v));
  return (Array.isArray(ri) ? ri : []).filter(x => x && typeof x.what === 'string' && x.what.trim()).map(x => {
    const t = num(x.t0_s) ? parseFloat(x.t0_s) : null;
    const fam = x.family || ('d:' + String(x.kind || 'reader issue').replace(/_/g, ' '));
    return `<div class="di-row high"${t != null ? ` data-t="${t}"` : ''}>
      <span class="di-sev">check</span>
      <div class="di-body">
        <div class="di-issue">${esc(x.what)}</div>
        <div class="di-tags"><span class="di-cat">${esc(famName(fam))}</span>${t != null ? `<span class="di-t">@ `
          + `${esc(fmtT(t))}</span>` : ''}${x.signal ? `<span class="di-cat">${esc(x.signal)}</span>` : ''}</div>
      </div></div>`;
  });
}
// The steps and key events the model gave with no time the board can read (board/to_board.py keeps them with t_s
// null): rows of the dense timeline and of the key events, listed after the timed ones, with "no time" where the time
// goes and nothing to seek to; the key events are numbered on from the timed ones (first).
function untimedRows(d, first = 1) {
  const steps = (d.event_labels || []).filter(e => e && e.t_s == null).map(e => {
    const cl = (e.contribution || '').toLowerCase() || '-';
    return `<div class="ev untimed">
        <span class="t">no time</span>
        <span class="who"><span class="arm ${esc(e.arm || '')}">${esc(armLabel(e.arm))}</span></span>
        <span class="phrase">${buildPhrase(e)}</span>
        <span class="contrib ${contribClass(e.contribution)}">${esc(cl)}</span>
      </div>`;
  });
  const keys = (d.key_events || []).filter(k => k && k.t_s == null).map((k, i) => {
    const oc = (k.outcome || '').toLowerCase();
    return `<div class="key-ev untimed ${esc(oc)}">
      <span class="ke-num">${first + i}</span>
      <span class="ke-time">no time</span>
      <div class="ke-body">
        <div class="ke-row1"><span class="ke-label">${esc(k.label || '')}</span>${oc
          ? `<span class="outcome ${esc(oc)}">${esc(oc)}</span>` : ''}</div>
        ${k.note ? `<div class="ke-note">${esc(k.note)}</div>` : ''}
      </div>
    </div>`;
  });
  return {steps, keys};
}
// What this dataset's rules set aside: each issue a rule moved to _excluded (board/build.py apply_rules, and the
// issues label/pieces.py stitch set aside at our own cuts), with its text, tag, time and the rule's reason, and each
// check a rule withheld (set_aside_checks, board/serve.py episode_view), with whether it fired and the reason. None of
// them counts. They are rows of the problems' own kind, in a fold that is closed until opened and says how many it
// holds. Nothing is drawn when nothing was set aside.
function setAsideHtml(d) {
  const ex = (Array.isArray(d._excluded) ? d._excluded : []).filter(x => x && x.issue);
  const ck = (Array.isArray(d.set_aside_checks) ? d.set_aside_checks : []).filter(x => x && x.check);
  const n = ex.length + ck.length;
  if (!n) return '';
  const num = v => v != null && v !== '' && !isNaN(parseFloat(v));
  const why = t => { const w = String(t || '').trim(); return w ? w[0].toUpperCase() + w.slice(1).replace(/\.?$/, '.')
    : ''; };
  const fam = Object.fromEntries(OUR_CHECKS.map(([k, , f]) => [k, f]));
  const rows = ex.map(x => `<div class="di-row low minor"${num(x.t_s) ? ` data-t="${parseFloat(x.t_s)}"` : ''}>
      <span class="di-sev">${esc(x.severity || 'flag')}</span>
      <div class="di-body">
        <div class="di-issue">${esc(x.issue)}</div>
        <div class="di-tags">${x.category ? `<span class="di-cat">${esc(tagName(x.category, x.list || 'data_issues'))}`
          + `</span>` : ''}${num(x.t_s) ? `<span class="di-t">@ ${esc(fmtT(parseFloat(x.t_s)))}</span>` : ''}</div>
        ${x.reason ? `<div class="di-ev">${esc(why(x.reason))}</div>` : ''}
      </div></div>`).concat(ck.map(c => `<div class="di-row low minor">
      <span class="di-sev">check</span>
      <div class="di-body">
        <div class="di-issue">${esc(fam[c.check] ? famName(fam[c.check]) : tagName(c.check, 'data_issues'))}, ${
          c.flagged ? 'fired' : 'clear'}, not counted on this dataset</div>
        ${c.reason ? `<div class="di-ev">${esc(why(c.reason))}</div>` : ''}
      </div></div>`));
  const closed = `Show the ${n} set aside by this dataset's rules`;
  const opened = `Hide the ${n} set aside by this dataset's rules`;
  return `<div class="pub-fold sa-fold"><div class="sn-fold"><div class="sn-fold-in"><div class="info-block di-block">`
    + rows.join('') + `</div></div></div><button class="ck-more pub-show" type="button" aria-expanded="false" `
    + `data-closed="${closed}" data-open="${opened}">${closed}</button></div>`;
}
// The cameras the model is not shown (board/build.py unshown_cameras: more extra cameras than it is shown, a stereo
// camera's second eye, every camera but one on a head rig, an infrared, thermal or mask video). Each plays in a cell
// of its own after the other cameras, synced as a side camera, named as not shown to the model, and one line under the
// cameras says why for each. Nothing is drawn for an episode the model was shown whole.
function unshownCams(d) {
  return (Array.isArray(d.unshown_cameras) ? d.unshown_cameras : []).filter(u => u && u.view);
}
// kept: the footage playing now moves into the new cells (renderEp keepVideo), so they get no source of their own
function unshownCellsHtml(d, eidEnc, kept) {
  return unshownCams(d).map(u => `
        <div class="cam-cell cam-wrist cam-unshown">
          <span class="cam-label">${esc(u.name || u.view)}, not shown to the model</span>
          <video id="video-${esc(u.view)}" preload="auto" muted playsinline${kept ? ''
            : ` src="${videoSrc(eidEnc, u.view)}"${posterAttr(eidEnc, u.view)}`} `
            + `onloadedmetadata="this.currentTime=0.03"></video>
        </div>`).join('');
}
function unshownNote(d) {
  const us = unshownCams(d);
  if (!us.length) return '';
  const each = us.map(u => `${esc(u.name || u.view)}${u.why ? ` (${esc(u.why)})` : ''}`);
  const list = each.length > 1 ? `${each.slice(0, -1).join(', ')} and ${each[each.length - 1]}` : each[0];
  return `<p class="unshown-note">The model was not shown ${us.length === 1 ? 'the camera' : 'the cameras'} ${list}. `
    + `${us.length === 1 ? 'It plays' : 'They play'} here so every camera of the upload can be watched.</p>`;
}
// What the model was not shown of the upload, from board/build.py reader_notes. It draws the reader's note on the
// recorded state as text, then the cameras, signals, arrays and depth streams it did not read, each with the reason it
// gave, in the same fold as the notes in the files. The fold is closed until opened, since an upload can leave out
// dozens of signals. The button says the counts and flips its label when open. Nothing is drawn when the model was
// shown it all.
function readerNotesHtml(rn) {
  if (!rn) return '';
  const left = rn.left_out || {};
  const kinds = [['cameras', 'Cameras', 'camera', 'cameras'], ['signals', 'Signals', 'signal', 'signals'],
                 ['arrays', 'Arrays', 'array', 'arrays'], ['depth', 'Depth streams', 'depth stream', 'depth streams']]
    // a value that is not a list (an older or hand edited qa file) is passed over, as board/build.py reader_notes does
    .filter(([k]) => Array.isArray(left[k]) && left[k].length);
  const counts = kinds.map(([k, , one, many]) => `${left[k].length} ${left[k].length === 1 ? one : many}`);
  const n = counts.length > 1 ? `${counts.slice(0, -1).join(', ')} and ${counts[counts.length - 1]}` : counts[0];
  const closed = `The model was not shown ${n}`, opened = `Hide the ${n} the model was not shown`;
  return (rn.state_note ? `<div class="rn-note">${esc(rn.state_note)}</div>` : '')
    + (kinds.length ? `<div class="pub-fold rn-fold"><div class="sn-fold"><div class="sn-fold-in"><div class="rn-body">`
      + kinds.map(([k, head]) => `<div class="rn-k">${head}</div>`
        + left[k].map(x => `<div class="rn-i">${esc(x)}</div>`).join('')).join('')
      + `</div></div></div><button class="ck-more pub-show" type="button" aria-expanded="false" `
      + `data-closed="${closed}" data-open="${opened}">${closed}</button></div>` : '');
}
function dsLabel(ds) {
  return DS_LABELS[ds] || String(ds).replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}
function presentDatasets(eps) {
  const seen = Array.from(new Set(eps.map(datasetOf)));
  const known = DS_ORDER.filter(d => seen.includes(d));
  const extra = seen.filter(d => !DS_ORDER.includes(d)).sort();
  return [...known, ...extra];
}
let currentDataset = null;
let _dsToken = 0;
async function setDataset(ds, keepFile) {
  // a static build fetches the dataset's list first; a later tab click wins over a slower earlier one
  const token = ++_dsToken;
  await ensureDataset(ds);
  if (token !== _dsToken) return;
  currentDataset = ds;
  ISSUE_FILTER = null;             // a fresh dataset starts unfiltered
  buildIssueFilter(ds);
  return renderRail(ds, keepFile);
}

// the labels' license, when the board's owner gives one (its manifest), under the list's downloads
(() => {
  const lic = BOARD.labels_license, el = document.getElementById('dl-lic');
  if (!lic || !el) return;
  const name = lic.url ? `<a href="${esc(lic.url)}" target="_blank" rel="noreferrer">${esc(lic.name)}</a>` : esc(lic.name);
  el.innerHTML = `Labels${lic.by ? ` by ${esc(lic.by)}` : ''}, licensed ${name}. The footage keeps each dataset's license.`;
  el.hidden = false;
})();

// ---- export: the episodes currently listed in the rail, as JSON Lines ----
document.getElementById('export-jsonl').addEventListener('click', async () => {
  const files = _railFiles.slice();
  if (!files.length) return;
  const btn = document.getElementById('export-jsonl');
  const label = btn.textContent;
  btn.textContent = 'Exporting\u2026';
  try {
    const blob = await exportBlob(files);
    const tag = [currentDataset, ISSUE_FILTER, SEARCH].filter(Boolean).join('_').replace(/[^A-Za-z0-9_.-]+/g, '-');
    saveBlob(blob, `episodes_${tag || 'all'}_${files.length}.jsonl`);
  } catch (e) { alert(e.message); }
  btn.textContent = label;
});

// ---- export: the listed episodes' hand keypoints, as JSON Lines in a file of their own (never in the labels') ----
function kpListed() {
  return _railFiles.filter(f => KP_INDEX && KP_INDEX[f]);
}
const kpBtn = document.getElementById('export-kp');
let _kpBusy = false;
function updateKpExport() {
  const files = BY ? [] : kpListed();
  document.getElementById('kp-exp').classList.toggle('off', !files.length);
  kpBtn.hidden = !files.length;
  if (!files.length || _kpBusy) return;
  const bytes = files.reduce((a, f) => a + (KP_INDEX[f].bytes || 0), 0);
  kpBtn.textContent = 'Hand keypoints';
  document.getElementById('kp-size').textContent = `Hand keypoints of ${files.length.toLocaleString()} `
    + `${files.length === 1 ? 'episode' : 'episodes'}, ${fmtBytes(bytes)}. `;
}
kpBtn.addEventListener('click', async () => {
  const files = kpListed();
  if (!files.length || _kpBusy) return;
  _kpBusy = true; kpBtn.disabled = true;
  const parts = new Array(files.length);
  let next = 0, done = 0;
  const show = () => { kpBtn.innerHTML = `Exporting<small>${done.toLocaleString()} of `
    + `${files.length.toLocaleString()}</small>`; };
  show();
  try {
    const worker = async () => {
      while (next < files.length) {
        const k = next++;
        const r = await fetch(keypointsUrl(files[k]), STATIC ? {} : {cache: 'no-store'});
        if (!r.ok) throw new Error('export failed: ' + files[k] + ' ' + r.status);
        parts[k] = (await r.text()).trim();
        done++; show();
      }
    };
    await Promise.all(Array.from({length: Math.min(6, files.length)}, worker));
    const tag = [currentDataset, ISSUE_FILTER, SEARCH].filter(Boolean).join('_').replace(/[^A-Za-z0-9_.-]+/g, '-');
    saveBlob(new Blob(parts.flatMap(t => [t, '\n']), {type: 'application/x-ndjson'}),
      `hand_keypoints_${tag || 'export'}_${files.length}.jsonl`);
  } catch (e) { alert(e.message); }
  _kpBusy = false; kpBtn.disabled = false;
  updateKpExport();
});

// ---- episode ID search ----
const searchWrap = document.getElementById('ep-search');
const searchEl = document.getElementById('ep-search-q');
let SEARCH = '';
function searchTerms() { return SEARCH.toLowerCase().split(/\s+/).filter(Boolean); }
// An episode matches when every term appears in its raw ID, its file name, or its display name.
function matchesSearch(ep, terms) {
  if (!terms.length) return true;
  const hay = [ep.episode_id, ep.file, epName(ep.episode_id)].join(' ').toLowerCase();
  return terms.every(t => hay.includes(t));
}
let _searchT = null;
function applySearch(v) {
  SEARCH = v.trim();
  searchWrap.classList.toggle('has-q', !!SEARCH);
  renderRail(currentDataset, _activeFile, true);   // keeps the open episode only while it still matches
}
searchEl.addEventListener('input', () => { clearTimeout(_searchT);
  _searchT = setTimeout(() => applySearch(searchEl.value), 250); });
searchEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') { const c = epListEl.querySelector('.ep-card'); if (c) selectEp(c.dataset.file); }
  else if (e.key === 'Escape') { searchEl.value = ''; applySearch(''); }
});
document.getElementById('ep-search-clear').addEventListener('click', () => { searchEl.value = ''; applySearch('');
  searchEl.focus(); });
// "/" focuses the search, unless the user is already typing somewhere
document.addEventListener('keydown', (e) => {
  if (e.key === '/' && !/^(INPUT|TEXTAREA|SELECT)$/.test((document.activeElement || {}).tagName || '')) {
    e.preventDefault(); searchEl.focus(); }
});

// ---- data-issue class filter ----
let ISSUE_FILTER = null;   // null = show all; else an issue-class string
// which severities the filter counts: by default what the board counts (a data issue at medium or high, an operator
// mistake that changed the outcome at medium or high, or any at high), or everything the model reported, low severity
// included. Remembered per viewer.
let INCLUDE_MINOR = false;
let CHECKS_OPEN = false;       // the full list of capture checks, kept open or closed across episodes

// Every check run on this episode, fired or not, in one section: ours (recording and label) and the capture checks
// of public-dataset-adapter. A firing counted as a problem is also in the problem cards above; here each check
// shows only its status, so the reader sees what was tested as well as what was found.
// each of our checks is named by the problem it finds, its family (families.json), so the check list and the problem
// list always say the same thing
const OUR_CHECKS = [['stream_pairing', 'crossed', 'streams-crossed'], ['recorded_jumps', 'flagged', 'recorded-jump'],
                    ['gripper_channels', 'flagged', 'gripper-flat'], ['timebase', 'sped_up_recording', 'sped-up']];
// a check's reason as sentences: "no pose: joint-state teleop has none" reads "No pose. Joint-state teleop has none."
// a reason that holds an error message, as one sentence: its first letter raised and a full stop, its colons kept
const asSentence = t => { const x = String(t).trim(); return x.charAt(0).toUpperCase() + x.slice(1)
  + (/[.!?]$/.test(x) ? '' : '.'); };
const sentences = (t) => String(t).split(/:\s+/).map(x => x.charAt(0).toUpperCase() + x.slice(1)).join('. ')
  .replace(/\.?$/, '.');
// an outcome in words; a failure of the kind partial (a real part of the goal left undone) says it was partly done
const outcomeWords = (oc, kind) => oc === 'failure' && kind === 'partial' ? 'failure, partly done'
  : String(oc).replace(/_/g, ' ');
function checksSection(d) {
  const dc = d.dataset_checks || {}, rows = [];
  // a check not run on this episode says why (a state not on the cameras' frames, checks/stream_pairing.py unaligned),
  // and one that stopped with an error says it, with the error (checks/stream_pairing.py _safe)
  for (const [k, field, fam] of OUR_CHECKS) if (dc[k] && typeof dc[k] === 'object') rows.push({name: famName(fam),
    st: dc[k].error ? 'err' : dc[k].not_assessed ? 'na' : dc[k][field] ? 'issue' : 'clear',
    text: dc[k].error ? `The check stopped with an error (${dc[k].error}).`
      : dc[k].not_assessed ? sentences(dc[k].not_assessed) : ''});
  // the capture checks test a robot's recording (its state stream, grippers and camera timing); none of them applies to
  // footage from a person's head camera, so a head-camera episode lists none
  const cq = d._rig !== 'ego_head' && dc.capture_qc && Array.isArray(dc.capture_qc.checks) ? dc.capture_qc : null;
  // checks/sensors.py: the recording's other signals and depth streams, shown as notes (none counts as an issue yet)
  const sc = dc.sensor_checks && Array.isArray(dc.sensor_checks.checks) ? dc.sensor_checks : null;
  // checks/contacts.py: the recording's contacts against what the model saw at them, notes as well
  const tc = dc.contact_checks && typeof dc.contact_checks === 'object' ? dc.contact_checks : null;
  if (!rows.length && !cq && !sc && !tc) return '';
  const dot = st => `<span class="ck-dot ${st}" aria-hidden="true"></span>`;
  const word = {issue: 'fired', note: 'note', clear: 'clear', na: 'not applicable', err: 'error'};
  const row = r => `<div class="ck-row ${r.st}">${dot(r.st)}<span class="ck-name">${esc(r.name)}</span><span `
    + `class="ck-st">${word[r.st]}</span>${r.text ? `<div class="ck-text">${esc(r.text)}</div>` : ''}</div>`;
  const ours = rows.length ? `<div class="ck-block"><div class="ck-head"><span class="ck-title">Our checks</span><span `
    + `class="ck-sum">${rows.filter(r => r.st === 'issue').length} of ${rows.length} fired</span>`
    + `</div>${rows.map(row).join('')}</div>` : '';
  let theirs = '';
  if (cq) {
    const notes = {};
    // a note's evidence, each firing a sentence, then why the check is a note, once (an older record has only its text)
    for (const n of cq.notes || []) (notes[n.check] = notes[n.check] || []).push(n.evidence || n.text);
    const flags = {};
    for (const f of cq.flags || []) (flags[f.check] = flags[f.check] || []).push(f.evidence || f.title);
    const all = cq.checks.map(c => ({name: c.name, group: c.group, why: c.why,
      st: c.status === 'fired' ? (c.shown_as === 'issue' ? 'issue' : 'note') : c.status === 'clear' ? 'clear'
        : c.status === 'errored' ? 'err' : 'na',
      text: c.status === 'fired' ? [...new Set((c.shown_as === 'issue' ? flags[c.check] : notes[c.check]) || [])]
        .concat(c.shown_as === 'note' && c.why && (cq.notes || []).some(x => x.check === c.check && x.evidence)
          ? [c.why] : []).join(' ') : c.status === 'errored' && c.why ? asSentence(c.why) : ''}));
    // a check that stopped with an error is shown with the ones that fired, before the full list
    const fired = all.filter(c => c.st === 'issue' || c.st === 'note' || c.st === 'err');
    const n = st => all.filter(c => c.st === st).length;
    const groups = [...new Set(all.map(c => c.group))];
    theirs = `<div class="ck-block ck-theirs${CHECKS_OPEN ? ' open' : ''}">
      <div class="ck-head"><span class="ck-title">Checks from <a href="https://pantheon.inc/research/we-looked-at-the-data"
        target="_blank" rel="noopener">We Looked at the Data</a></span>
        <span class="ck-sum">${n('issue')} ${n('issue') === 1 ? 'issue'
          : 'issues'} &middot; ${n('note')} ${n('note') === 1 ? 'note' : 'notes'} &middot; ${n('clear')} clear `
          + `&middot; ${n('na')} not applicable${n('err') ? ` &middot; ${n('err')} ${n('err') === 1 ? 'error'
            : 'errors'}` : ''}</span></div>
      ${fired.map(row).join('')}
      <div class="ck-all"><div class="ck-all-in">${groups.map(g => `<div class="ck-group">${esc(g)}`
        + `</div>${all.filter(c => c.group === g).map(c => `<div class="ck-row ${c.st}">${dot(c.st)}<span `
        + `class="ck-name">${esc(c.name)}</span><span class="ck-st">${word[c.st]}</span>${(c.st === 'na'
        || c.st === 'err') && c.why
        ? `<div class="ck-text">${esc(c.st === 'err' ? asSentence(c.why) : sentences(c.why))}</div>`
        : ''}</div>`).join('')}`).join('')}</div></div>
      <button class="ck-more" type="button" onclick="CHECKS_OPEN = !CHECKS_OPEN; `
        + `this.closest('.ck-theirs').classList.toggle('open', CHECKS_OPEN); this.textContent = CHECKS_OPEN ? 'Hide `
        + `the full list' : 'Show all ${all.length} checks'">${CHECKS_OPEN ? 'Hide the full list'
        : `Show all ${all.length} checks`}</button>
    </div>`;
  }
  let sensors = '';
  if (sc) {
    const ev = {};
    for (const n of sc.notes || []) (ev[n.check] = ev[n.check] || []).push(n.evidence);
    const all = sc.checks.filter(c => c.status !== 'na').map(c => ({name: c.name,
      st: c.status === 'fired' ? 'note' : c.status === 'errored' ? 'err' : 'clear',
      text: c.status === 'errored' ? `The check stopped with an error (${c.error}).`
        : (ev[c.check] || []).map(sentences).join(' ')}));
    if (all.length) sensors = `<div class="ck-block"><div class="ck-head"><span class="ck-title">Sensor and depth checks`
      + `</span><span class="ck-sum">${all.filter(r => r.st === 'note').length} of ${all.length} noted</span></div>`
      + `${all.map(row).join('')}</div>`;
  }
  let touch = '';
  if (tc) {
    const ev = {};
    for (const n of tc.notes || []) if (n && n.evidence) (ev[n.check] = ev[n.check] || []).push(n.evidence);
    // each evidence sentence as checks/contacts.py wrote it, with its first letter raised and a full stop
    const asWritten = t => { const x = String(t).trim(); return x.charAt(0).toUpperCase() + x.slice(1)
      + (/[.!?]$/.test(x) ? '' : '.'); };
    const ran = (tc.checked || 0) > 0;
    const all = [['clock_offset', "Touch sensor's clock against the cameras", !!(tc.offset_ms && tc.offset_ms.n)],
      ['touch_not_seen', 'Contacts the frames show no touch at', ran],
      ['hand_mismatch', 'Contacts seen on the other hand', ran],
      ['contact_missing', 'Grasps no recorded contact covers', ran]]
      .map(([k, name, run]) => ({name, st: ev[k] ? 'note' : run ? 'clear' : 'na', text: (ev[k] || []).map(asWritten)
        .join(' ')}))
      .concat(Object.keys(ev).filter(k => !['clock_offset', 'touch_not_seen', 'hand_mismatch', 'contact_missing']
        .includes(k)).map(k => ({name: k.replace(/_/g, ' '), st: 'note', text: ev[k].map(asWritten).join(' ')})))
      .filter(r => r.st !== 'na');
    if (all.length) touch = `<div class="ck-block"><div class="ck-head"><span class="ck-title">Contact checks</span>`
      + `<span class="ck-sum">${tc.checked || 0} of ${tc.contacts || 0} contacts checked, ${all.filter(r => r.st
        === 'note').length} of ${all.length} noted</span></div>${all.map(row).join('')}</div>`;
  }
  if (!ours && !sensors && !touch && !theirs) return '';
  return `<h3 class="section">Checks <span class="count">every check run on this episode</span></h3><div `
    + `class="ck">${ours}${sensors}${touch}${theirs}</div>`;
}
try { INCLUDE_MINOR = localStorage.getItem('board.includeMinor') === '1'; } catch (e) {}
// every problem is one family (board/families.py, the classification the board counts with): a listed family
// under its own name, or one named after the model's tag ("d:" data issue, "m:" operator mistake). Our checks and the
// episode's outcome raise their families directly, so there is one row per problem, never one per tag or per source.
const FAMILIES = __FAMILIES__;
const famName = s => (FAMILIES[s] || {}).name || String(s).slice(2);
const famList = s => (FAMILIES[s] || {}).list || (String(s).startsWith('m:') ? 'mistake' : 'data');
const famsOf = e => INCLUDE_MINOR ? [...new Set([...(e.families || []), ...(e.minor_families || [])])]
  : (e.families || []);
// The filter dropdown: the problem families present in this dataset, each with the count of episodes that have it
// (an episode counts once per family). Rebuilt whenever the dataset, the filter or the severity switch changes.
// "any data issue" and "any operator mistake": the episode has a family of that list that counts
const hasDataIssue = e => famsOf(e).some(f => famList(f) === 'data');
const hasMistake = e => famsOf(e).some(f => famList(f) === 'mistake');
function matchesIssueFilter(e) {
  if (!ISSUE_FILTER) return true;
  if (ISSUE_FILTER === '__any__') return hasDataIssue(e);
  if (ISSUE_FILTER === '__anymistake__') return hasMistake(e);
  if (ISSUE_FILTER === '__hands__') return (e.hands_hidden_s || 0) > 0;
  return famsOf(e).includes(ISSUE_FILTER);
}
function buildIssueFilter(ds) {
  const eps = railEps().filter(e => datasetOf(e) === ds);
  const counts = new Map();      // family -> episode count
  let withAny = 0, withMistake = 0;
  for (const e of eps) {
    if (hasDataIssue(e)) withAny++;
    if (hasMistake(e)) withMistake++;
    for (const f of famsOf(e)) counts.set(f, (counts.get(f) || 0) + 1);
  }
  issueFilterEl.innerHTML = '';
  // nothing to filter
  if (!counts.size && !eps.some(e => (e.minor_families || []).length)) return;
  // one row per problem, in two groups, each with its own header and colour
  const groups = [];
  const overview = [['All episodes', null, eps.length, ''], ['Any data issue', '__any__', withAny, 'data']];
  if (withMistake) overview.push(['Any operator mistake', '__anymistake__', withMistake, 'mistake']);
  // head cameras: the episodes where the wearer's hands leave the view, and how much of the footage that is
  if (eps.some(e => e.hands_hidden_s != null)) {
    const hid = eps.filter(e => (e.hands_hidden_s || 0) > 0);
    const share = sumDur(eps) ? eps.reduce((a, e) => a + (e.hands_hidden_s || 0), 0) / sumDur(eps) : 0;
    const p = 100 * share;
    overview.push(['Hands out of view', '__hands__', hid.length, '', `${p === 0 ? '0' : p < 1 ? p.toFixed(2)
      : p.toFixed(1)}% of recorded hours`]);
  }
  groups.push({key: 'overview', title: 'Overview', items: overview});
  const rows = lst => Array.from(counts.entries()).filter(([f]) => famList(f) === lst)
    .map(([f, n]) => [famName(f), f, n]).sort((a, b) => b[2] - a[2]);
  const dataItems = rows('data'), mistakeItems = rows('mistake');
  const sevNote = INCLUDE_MINOR ? 'every severity, low included' : 'medium or high severity';
  if (dataItems.length) groups.push({key: 'data', title: 'Data issues',
    note: `faults in the recording, the scene or the label, ${BY ? `as ${cmpWho(BY)} reported them`
      : 'from the model or our checks'}; ${sevNote}`,
      items: dataItems});
  if (mistakeItems.length) groups.push({key: 'mistake', title: 'Operator mistakes', note: INCLUDE_MINOR
    ? 'the recording is faithful; the demonstration went wrong (every mistake reported, minor ones included)'
    : 'the recording is faithful; the demonstration went wrong (a changed outcome at medium or high, anything at high)',
      items: mistakeItems});
  const opts = groups.flatMap(g => g.items);
  const cur = opts.find(o => Array.isArray(o) && o[1] === ISSUE_FILTER) || opts[0];
  const btn = document.createElement('button');
  btn.className = 'if-btn';
  btn.setAttribute('aria-haspopup', 'listbox');
  const filtered = ISSUE_FILTER !== null;
  btn.setAttribute('aria-label', filtered ? `Filter by issue: ${cur[0]}` : 'Filter by issue');
  btn.innerHTML = '<svg class="if-ico" width="16" height="16" viewBox="0 0 16 16"><path d="M2 3h12l-4.6 5.4V13l-2.8 '
    + '1.2V8.4z" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/></svg>'
    + (filtered ? `<span class="if-lab">${esc(cur[0])}</span>` : '<span class="if-lab if-ph">Filter by issue</span>')
      + (filtered ? `<span class="if-n">${cur[2]}</span>` : '')
    + (filtered ? '<span class="if-clear" title="Clear filter">&times;</span>'
                : '<svg width="12" height="12" viewBox="0 0 12 12"><path d="M2 4l4 4 4-4" stroke="currentColor" '
                  + 'stroke-width="1.8" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>');
  issueFilterEl.classList.toggle('filtered', filtered);
  const menu = document.createElement('div');
  menu.className = 'if-menu';
  menu.setAttribute('role', 'listbox');
  // the severity switch: counts, the any-issue total and the list all follow it
  const sevBar = document.createElement('div');
  sevBar.className = 'if-sev';
  sevBar.innerHTML = '<span class="if-sev-k">Severity</span><span class="if-sev-seg" role="radiogroup" '
    + 'aria-label="Severity shown">'
    + `<button type="button" role="radio" aria-checked="${!INCLUDE_MINOR}" data-minor="0">Medium and high</button>`
    + `<button type="button" role="radio" aria-checked="${INCLUDE_MINOR}" data-minor="1">All, including low</button>`
      + `</span>`;
  sevBar.addEventListener('click', (ev) => {
    const b = ev.target.closest('button[data-minor]');
    ev.stopPropagation();
    if (!b) return;
    const next = b.dataset.minor === '1';
    if (next === INCLUDE_MINOR) return;
    INCLUDE_MINOR = next;
    try { localStorage.setItem('board.includeMinor', next ? '1' : '0'); } catch (e) {}
    buildIssueFilter(ds);
    renderRail(ds, _activeFile);
    issueFilterEl.classList.add('open');
    placeIssueMenu();
  });
  menu.appendChild(sevBar);
  for (const g of groups) {
    const sec = document.createElement('div');
    sec.className = 'if-group g-' + g.key;
    const maxN = Math.max(1, ...g.items.map(o => o[2]));
    sec.innerHTML = `<div class="if-ghead"><span class="if-gtitle">${esc(g.title)}</span>`
      + (g.key === 'overview' ? '' : `<span class="if-gcount">${g.items.length} ${g.items.length === 1 ? 'type'
        : 'types'}</span>`) + '</div>'
      + (g.note ? `<div class="if-gnote">${esc(g.note)}</div>` : '');
    for (const [label, val, n, dot, sub] of g.items) {
      const r = document.createElement('div');
      r.className = 'if-row' + (ISSUE_FILTER === val ? ' active' : '');
      r.setAttribute('role', 'option');
      const pct = eps.length ? Math.round(100 * n / eps.length) : 0;
      r.title = `${n} of ${eps.length} episodes (${pct < 1 && n ? '<1' : pct}%)`;
      const bar = g.key === 'overview' ? '<span class="if-track"></span>'
        : `<span class="if-track"><i style="width:${Math.max(4, 100 * n / maxN).toFixed(1)}%"></i></span>`;
      r.innerHTML = `<span class="if-name">${dot ? `<b class="if-dot d-${dot}"></b>` : ''}${esc(label)}${sub
        ? `<small class="if-sub">${esc(sub)}</small>` : ''}</span>${bar}<span class="if-n">${n}</span>`;
      r.addEventListener('click', (ev) => {
        ev.stopPropagation();
        ISSUE_FILTER = val;
        buildIssueFilter(ds);
        renderRail(ds, _activeFile);
      });
      sec.appendChild(r);
    }
    menu.appendChild(sec);
  }
  btn.addEventListener('click', (ev) => {
    ev.stopPropagation();
    if (ev.target.closest('.if-clear')) { ISSUE_FILTER = null; buildIssueFilter(ds); renderRail(ds, _activeFile);
      return; }
    lbOpen(false);             // the menus overlap in the rail: opening one closes the other
    placeIssueMenu();
    issueFilterEl.classList.toggle('open');
  });
  issueFilterEl.classList.remove('open');
  issueFilterEl.appendChild(btn);
  issueFilterEl.appendChild(menu);
}
// place the menu under the button, as wide as it needs (340px) but never past the viewport's edges
function placeIssueMenu() {
  const b = issueFilterEl.querySelector('.if-btn'), m = issueFilterEl.querySelector('.if-menu');
  if (!b || !m) return;
  const r = b.getBoundingClientRect(), pad = 8;
  const w = Math.min(Math.max(340, r.width), innerWidth - 2 * pad);
  const x = Math.min(Math.max(pad, r.left), innerWidth - w - pad);
  const y = r.bottom + 4;
  m.style.setProperty('--if-x', x + 'px');
  m.style.setProperty('--if-y', y + 'px');
  m.style.setProperty('--if-w', w + 'px');
  m.style.setProperty('--if-h', Math.max(160, innerHeight - y - pad) + 'px');
}
addEventListener('resize', () => { if (issueFilterEl.classList.contains('open')) placeIssueMenu(); });
addEventListener('scroll', () => { if (issueFilterEl.classList.contains('open')) placeIssueMenu(); }, {passive: true});
document.addEventListener('click', (e) => {
  if (!issueFilterEl.contains(e.target)) issueFilterEl.classList.remove('open'); });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') issueFilterEl.classList.remove('open'); });

// the model a URL asks for (?by=<key>), when this board has it
function urlLabeller(q) {
  const k = q.get('by');
  return k && CMP && cmpModel(k) ? k : null;
}
// the first dataset, in tab order, the labeller labelled
function firstDataset(want) {
  const has = d => railEps().some(e => datasetOf(e) === d);
  const datasets = presentDatasets(ALL_EPS);
  return want && datasets.includes(want) && has(want) ? want : (datasets.find(has) || null);
}
async function loadEpisodes() {
  const [eps, cmp, kp, sn] = await Promise.all([loadAllEpisodes(), BOARD.compare ? fetchJson(compareUrl('index'))
    : null, BOARD.keypoints ? fetchJson(kpIndexUrl()) : null, BOARD.sensors ? fetchJson(snIndexUrl()) : null]);
  ALL_EPS = eps;
  CMP = cmp && (cmp.models || []).length ? cmp : null;
  KP_INDEX = kp && kp.files ? kp.files : null;
  SN_INDEX = sn && sn.files ? sn.files : null;
  const q = new URLSearchParams(location.search);
  // ?ep=<file> reopens that episode, so a reload or a shared link keeps its place; it takes the file name or the
  // episode id (episode_habit_005733 or episode_habit_005733.json). ?by=<key> keeps a comparison's labels chosen
  // (a model that did not label the episode asked for gives way to the board's own), ?ds=<dataset> opens a
  // dataset's tab. Otherwise the first dataset opens.
  const wantFile = q.get('ep');
  const wantEp = wantFile && ALL_EPS.find(e => e.file === wantFile || e.file === wantFile + '.json');
  let by = urlLabeller(q);
  if (by && wantEp && !cmpHas(by, wantEp.file)) by = null;
  if (by) await applyLabeller(by);
  renderLabelsBy();
  if (CMP && q.get('view') === 'compare') { showCompare(true); return; }
  const ds = wantEp ? null : firstDataset(q.get('ds'));
  if (wantEp) await setDataset(datasetOf(wantEp), wantEp.file);
  else if (ds) await setDataset(ds);
  else renderRail(null);
  prefetchWhenIdle();
}

// ---- a dataset tab opens at once: its first episode is fetched ahead ----
// Once the page is idle, and again as soon as the pointer rests on a tab, each dataset's first episode is fetched
// ahead: its list (a static build), its labels and the first frame of every camera it shows (the players' posters).
// A tab click then renders from memory and the browser's cache. About a quarter of a megabyte per dataset; skipped
// when the browser asks to save data.
const _prefetched = new Set();
const _prefetchImgs = [];            // kept, so the browser finishes loading them
async function prefetchDataset(ds) {
  if (!ds || _prefetched.has(ds)) return;
  _prefetched.add(ds);
  try {
    await ensureDataset(ds);
    const terms = searchTerms();
    const first = railEps().find(e => datasetOf(e) === ds && matchesSearch(e, terms));
    if (!first) return;
    const d = BY && cmpHas(BY, first.file) ? await cmpEpisode(BY, first.file) : await fetchEpisode(first.file);
    if (!d) return;
    const eidEnc = encodeURIComponent((d._meta || {}).episode_id || '');
    const {main, side} = episodeCams(d);
    for (const cam of [main, ...side]) {
      const src = posterSrc(first.file, eidEnc, cam);
      if (!src) continue;
      const im = new Image();
      im.src = src;
      _prefetchImgs.push(im);
    }
  } catch (e) { _prefetched.delete(ds); }
}
function prefetchWhenIdle() {
  if (((globalThis.navigator || {}).connection || {}).saveData) return;
  const order = presentDatasets(ALL_EPS);
  const i = Math.max(0, order.indexOf(currentDataset));
  // the tabs beside the open one first, then outwards
  const queue = order.map((d, k) => [Math.abs(k - i), d]).sort((a, b) => a[0] - b[0]).map(x => x[1])
    .filter(d => d !== currentDataset);
  const idle = window.requestIdleCallback || (fn => setTimeout(fn, 400));
  const step = () => { const ds = queue.shift(); if (!ds) return;
    prefetchDataset(ds).finally(() => idle(step, {timeout: 2000})); };
  idle(step, {timeout: 2000});
}

// Back and forward between episodes (and labellers) update the view.
window.addEventListener('popstate', async () => {
  const q = new URLSearchParams(location.search);
  if (CMP && q.get('view') === 'compare') { showCompare(true); return; }
  const by = urlLabeller(q);
  if (document.body.classList.contains('view-cmp')) {
    const f = q.get('ep');
    const ep = f && ALL_EPS.find(e => e.file === f || e.file === f + '.json');
    leaveCompare(null, ep ? ep.file : null, by, true);
    return;
  }
  if (by !== BY) await setLabeller(by, {fromPop: true});
  const f = q.get('ep');
  const ep = f && railEps().find(e => e.file === f || e.file === f + '.json');
  if (ep) {
    if (currentDataset !== datasetOf(ep)) setDataset(datasetOf(ep), ep.file);
    else if (ep.file !== _activeFile) selectEp(ep.file, true);
  }
});

// ---- coverage: episodes and hours labelled on this board ----
const coverageEl = document.getElementById('coverage');
const DS_SHORT = {molmo: 'MolmoAct2', abc130k: 'ABC-130k', galaxea: 'Galaxea', fastumi: 'FastUMI-100K',
                  realomin: 'RealOmin', egocentric100k: 'Egocentric-100K', genhumanego: 'Gen-HumanEgo',
                  habit: 'HABIT', openaoe: 'OpenAoE'};
function fmtHours(sec) { const h = sec / 3600; return h >= 100 ? h.toFixed(0) : h.toFixed(1); }
// under an hour (an uploader's few episodes) the length reads in minutes, never as 0.0 h
function fmtSpan(sec, long) {
  return sec < 3600 ? [(sec / 60).toFixed(1), long ? 'minutes' : 'min'] : [fmtHours(sec), long ? 'hours' : 'h'];
}
function sumDur(list) { return list.reduce((a, e) => a + (e.duration_s || 0), 0); }
function renderCoverage(ds, shown) {
  if (!ALL_EPS.length) { coverageEl.innerHTML = ''; return; }
  // every dataset of the board keeps its tab; the numbers are the labeller's (a model that labelled none of a
  // dataset's episodes leaves its tab dimmed)
  const src = railEps();
  const per = presentDatasets(ALL_EPS).map(d => {
    const eps = src.filter(e => datasetOf(e) === d);
    return { d, n: eps.length, sec: sumDur(eps) };
  });
  const shownSec = sumDur(shown);
  const cur = per.find(p => p.d === ds) || { n: 0, sec: 0 };
  const narrowed = shown.length !== cur.n;
  const tabs = per.map(p => {
    const on = p.d === ds && !document.body.classList.contains('view-cmp');
    // the open dataset's tab also says how much of it the current filter or search keeps
    const num = on && narrowed
      ? `<em>${shown.length.toLocaleString()}</em> of ${p.n.toLocaleString()} &middot; <em>${fmtSpan(shownSec)[0]}`
        + `</em> ${fmtSpan(shownSec)[1]}`
      : `${p.n.toLocaleString()}<span class="u"> ${p.n === 1 ? 'ep' : 'eps'}</span> &middot; ${fmtSpan(p.sec)[0]} ${fmtSpan(p.sec)[1]}`;
    return `<div class="cv-cell${on ? ' on' : ''}${p.n ? '' : ' none'}" role="tab" aria-selected="${on}" `
      + `aria-disabled="${!p.n}" aria-label="${esc(dsLabel(p.d))}" data-ds="${esc(p.d)}">`
      + `<span class="cv-name">${esc(DS_SHORT[p.d] || dsLabel(p.d))}</span><span class="cv-num">${num}</span>`
      + '</div>';
  }).join('');
  coverageEl.innerHTML = `<div class="cv-all"><span class="cv-k">All datasets</span>`
    + `<span class="cv-fig"><span><b>${src.length.toLocaleString()}</b><small>${src.length === 1 ? 'episode' : 'episodes'}</small></span>`
    + `<span><b>${fmtSpan(sumDur(src), true)[0]}</b><small>${fmtSpan(sumDur(src), true)[1]}</small></span></span>`
    + `</div><div class="cv-cells" role="tablist">${tabs}</div>`;
  coverageEl.querySelectorAll('.cv-cell').forEach(el => {
    el.addEventListener('click', () => {
      if (el.classList.contains('none')) return;
      if (document.body.classList.contains('view-cmp')) { leaveCompare(el.dataset.ds); return; }
      if (el.dataset.ds !== currentDataset) setDataset(el.dataset.ds);
    });
    if (!el.classList.contains('none')) el.addEventListener('pointerenter', () => prefetchDataset(el.dataset.ds));
  });
  // when the tabs scroll, the open dataset's tab is always in view (a deep link to the last dataset included)
  const onTab = coverageEl.querySelector('.cv-cell.on'), strip = coverageEl.querySelector('.cv-cells');
  if (onTab && strip && strip.scrollWidth > strip.clientWidth) {
    const l = onTab.offsetLeft - strip.offsetLeft, r = l + onTab.offsetWidth;
    // a tab wider than the strip shows its start, so its name never begins mid-word
    if (l < strip.scrollLeft || r > strip.scrollLeft + strip.clientWidth) strip.scrollLeft = Math.max(0,
      onTab.offsetWidth > strip.clientWidth ? l : r - strip.clientWidth);
  }
}

// the heading over the list: the dataset it holds, and how many of its episodes the filter and the search leave
function renderRailHead(ds, eps) {
  const all = ds ? railEps().filter(e => datasetOf(e) === ds) : [];
  document.getElementById('rd-name').textContent = ds ? (DS_SHORT[ds] || dsLabel(ds)) : '';
  const [h, u] = fmtSpan(sumDur(all), true);
  document.getElementById('rd-sub').innerHTML = !ds ? ''
    : eps.length !== all.length ? `<em>${eps.length.toLocaleString()}</em> of ${all.length.toLocaleString()} episodes`
    : `${all.length.toLocaleString()} ${all.length === 1 ? 'episode' : 'episodes'} &middot; ${h} ${u}`;
  document.getElementById('dl-k').textContent = eps.length === 1 ? 'Download the episode shown'
    : `Download the ${eps.length.toLocaleString()} episodes shown`;
}

function renderRail(ds, keepFile, fromSearch) {
  const terms = searchTerms();
  const eps = railEps().filter(e => datasetOf(e) === ds && matchesSearch(e, terms) && matchesIssueFilter(e));
  renderCoverage(ds, eps);
  renderRailHead(ds, eps);
  epListEl.innerHTML = '';
  if (!eps.length) { _railFiles = []; _railJob++; }
  if (!eps.length && terms.length) {
    // the search looks in the dataset shown; nothing here matches, so say so and name the dataset, keep the open
    // episode on screen, and name any other dataset that holds a match rather than leaving a dead end
    const elsewhere = new Map();
    for (const e of railEps()) if (datasetOf(e) !== ds && matchesSearch(e, terms)) elsewhere.set(datasetOf(e),
      (elsewhere.get(datasetOf(e)) || 0) + 1);
    const where = Array.from(elsewhere.entries()).map(([d, n]) => `<span class="es-jump" `
      + `data-ds="${esc(d)}">${esc(DS_SHORT[d] || dsLabel(d))}${n > 1 ? ` (${n})` : ''}</span>`).join(', ');
    const one = elsewhere.size === 1 && [...elsewhere.values()][0] === 1;
    epListEl.innerHTML = `<div class="rail-empty"><b>No episode in ${esc(DS_SHORT[ds] || dsLabel(ds))} matches `
      + `&ldquo;${esc(SEARCH)}&rdquo;${ISSUE_FILTER ? ' with this issue filter' : ''}.</b>`
      + (where ? `<br>${one ? 'It is an episode of' : 'Episodes that match are in'} ${where}.` : '') + '</div>';
    epListEl.querySelectorAll('.es-jump').forEach(j => j.addEventListener('click', () => setDataset(j.dataset.ds)));
    updateKpExport();
    return;
  }
  if (!eps.length) {
    epListEl.innerHTML = '<div class="rail-empty">' +
      (ISSUE_FILTER ? 'No episodes with this issue.' : 'No episodes here yet.') + '</div>';
    leftCol.innerHTML = '';
    rightCol.innerHTML = '<div class="rail-empty" style="padding:48px 28px">No episodes in this dataset yet.</div>';
    currentEp.textContent = '';
    document.getElementById('current-ep-raw').textContent = '';
    document.getElementById('current-ep-src').innerHTML = '';
    document.getElementById('current-ep-reader').innerHTML = '';
    updateKpExport();
    return;
  }
  // the first screenful of cards (and the open or asked-for episode's) is built now; the rest follow in idle
  // batches, so switching dataset or typing a search never stops the page for a thousand cards at once
  _railFiles = eps.map(e => e.file);
  const job = ++_railJob;
  const makeCard = ep => {
    const card = document.createElement('div');
    card.className = 'ep-card';
    card.dataset.file = ep.file;
    // a session of tasks (head cameras) has no single verdict: the per-task success ratio ("8/9 tasks") instead of a
    // misleading "unrated"; a single task shows its completion verdict
    let outcomeHtml;
    if (ep.label_failed && !ep.cmp_status) {
      // the board's own reply gave no labels: no outcome to rate, which "unrated" would hide
      outcomeHtml = `<span class="outcome-tag fail">no labels</span>`;
    } else if (ep.cmp_status && ep.cmp_status !== 'parsed') {
      // a model's response that did not parse, was cut off or never came: a result, listed like any other
      outcomeHtml = `<span class="outcome-tag fail">${esc(ST_WORDS[ep.cmp_status] || ep.cmp_status)}</span>`;
    } else if (ep.n_tasks) {
      const allOk = ep.n_task_success === ep.n_tasks;
      const cls = allOk ? 'success' : 'partial';
      outcomeHtml = `<span class="outcome-tag ${cls}">${ep.n_task_success}/${ep.n_tasks} tasks</span>`;
    } else {
      const oc = (ep.task_completed || '').toLowerCase();
      outcomeHtml = `<span class="outcome-tag ${oc || 'none'}">${esc(oc ? outcomeWords(oc, ep.failure_kind)
        : 'unrated')}</span>`;
    }
    // flag data issues right of the outcome so a "success" with severe metadata
    // faults is not silently trusted, and say WHAT the top issue is (a mispaired
    // or mislabeled camera is foundational, not a footnote).
    const sev = (ep.max_severity || '').toLowerCase();
    const n = ep.n_issues || 0;
    const chipLabel = sev === 'high'
      ? (n > 1 ? `${n} data faults` : 'data fault')
      : `${n} data ${n > 1 ? 'issues' : 'issue'}`;
    const issueHtml = n ? `<span class="issue-tag ${sev || 'low'}">&#9888; ${chipLabel}</span>` : '';
    const nm = ep.n_mistakes || 0;
    const mistakeHtml = nm ? `<span class="issue-tag op" title="the recording is fine; the demonstration went `
      + `wrong">${nm} operator ${nm > 1 ? 'mistakes' : 'mistake'}</span>` : '';
    const nMinor = (ep.n_minor_issues || 0) + (ep.n_minor_mistakes || 0);
    const minorHtml = nMinor ? `<span class="issue-tag minor" title="${ep.n_minor_issues
      || 0} minor data issues and ${ep.n_minor_mistakes || 0} minor operator mistakes: shown on the episode, not `
      + `counted">${nMinor} minor</span>` : '';
    const noteHtml = (n && ep.top_issue)
      ? `<div class="issue-note ${sev || 'low'}" title="${esc(ep.top_issue)}">${esc(ep.top_issue)}</div>` : '';
    card.innerHTML = `
      <div class="id" title="${esc(ep.episode_id)}">${esc(epName(ep.episode_id))}</div>
      <div class="preview">${esc(ep.episode_prompt) || '<i>(no prompt)</i>'}</div>
      <div class="row">
        ${outcomeHtml}
        ${issueHtml}
        ${mistakeHtml}
        ${minorHtml}
      </div>
      ${noteHtml}`;
    card.addEventListener('click', () => selectEp(ep.file));
    if (ep.file === _activeFile) card.classList.add('active');
    return card;
  };
  const upTo = f => (f ? _railFiles.indexOf(f) : -1) + 1;
  let built = Math.min(eps.length, Math.max(RAIL_FIRST, upTo(keepFile) + 8, fromSearch ? upTo(_activeFile) + 8 : 0));
  const first = document.createDocumentFragment();
  for (let i = 0; i < built; i++) first.appendChild(makeCard(eps[i]));
  epListEl.appendChild(first);
  const later = window.requestIdleCallback ? (fn => requestIdleCallback(fn, {timeout: 120})) : (fn => setTimeout(fn, 16));
  const more = () => {
    if (job !== _railJob) return;
    const frag = document.createDocumentFragment();
    const end = Math.min(eps.length, built + RAIL_BATCH);
    for (; built < end; built++) frag.appendChild(makeCard(eps[built]));
    epListEl.appendChild(frag);
    if (built < eps.length) later(more);
  };
  if (built < eps.length) later(more);
  updateKpExport();
  if (fromSearch && eps.some(e => e.file === _activeFile)) {
    // the open episode is still in the results: keep it, no refetch while typing
    document.querySelectorAll('.ep-card').forEach(c => c.classList.toggle('active', c.dataset.file === _activeFile));
    return;
  }
  // otherwise the screen must never show an episode the list does not: open the first match
  const target = (keepFile && eps.some(e => e.file === keepFile)) ? keepFile : eps[0].file;
  return selectEp(target);
}

// the open episode, under whichever labels the board shows. When the labeller changes on the same episode
// (_lbSwapFile), the footage keeps playing: the same video elements move into the new layout while the panels ease
// out and in.
let _lbSwapFile = null;
function selectEp(file, fromPop) {
  const swap = _lbSwapFile === file && file === _activeFile && !!document.getElementById('video');
  _lbSwapFile = null;
  _activeFile = file;
  // Give every episode its own URL so it is shareable and reload-stable; the labeller rides along.
  if (!fromPop) {
    try {
      const u = new URL(location.href); u.searchParams.set('ep', file); u.searchParams.delete('view');
      if (BY) u.searchParams.set('by', BY); else u.searchParams.delete('by');
      history.replaceState(null, '', u);
    } catch (_) {}
  }
  document.querySelectorAll('.ep-card').forEach(c => {
    c.classList.toggle('active', c.dataset.file === file);
  });
  // the open episode's card stays in view in the list (a link or a reload can open one far down it); only the list
  // scrolls, never the page
  const card = epListEl.querySelector('.ep-card.active');
  if (card) {
    const top = card.offsetTop - epListEl.offsetTop, bot = top + card.offsetHeight;
    if (top < epListEl.scrollTop || bot > epListEl.scrollTop + epListEl.clientHeight) {
      epListEl.scrollTop = Math.max(0, top - 8);
    }
  }
  const tok = ++_epToken;
  const by = BY && cmpHas(BY, file) ? BY : null;
  return (by ? cmpEpisode(by, file) : fetchEpisode(file))
    .then(d => {
      if (tok !== _epToken || !d) return;
      SHOWN = by;
      if (swap) {
        const lp = document.getElementById('left-pane'), sl = lp.scrollTop, sr = rightCol.scrollTop;
        renderEp(d, {keepVideo: true});
        lp.scrollTop = sl; rightCol.scrollTop = sr;
      } else {
        renderEp(d);
        rightCol.scrollTop = 0; document.getElementById('left-pane').scrollTop = 0;
      }
      renderSourceBar();
    });
}

function buildPhrase(e) {
  // verb_class is the model's full phrase for the step (e.g. "Raise and rotate the empty wrists into working
  // orientations"), so it is rendered as it is. object usually repeats a noun already in the phrase, so it is not
  // appended. carry_phase (a destination, prefixed "-> ") becomes a separate chip without its arrow, so the chip
  // reads as a place.
  const v = e.verb_class || (e.object ? ('handle the ' + e.object) : 'move');
  let chipHtml = '';
  let carry = (e.carry_phase || '').trim();
  if (carry) {
    carry = carry.replace(/^->\s*/, '');
    // "<prep> <destination>" -> colored prep + destination
    const m = carry
      .match(/^(above|below|onto|into|in|on|to|toward|towards|over|under|near|away|beside|at|from)\s+(.+)$/i);
    if (m) {
      chipHtml = `<span class="dest-chip"><span class="prep">${esc(m[1])}</span> ${esc(m[2])}</span>`;
    } else if (carry) {
      chipHtml = `<span class="dest-chip">${esc(carry)}</span>`;
    }
  }
  return `<span class="verb">${esc(v)}</span>${chipHtml ? ' ' + chipHtml : ''}`;
}

// ================= hand pose over head-camera footage (board/hands.py) =================
// Each head-camera episode can have a file of 2D hand keypoints (<board>/hands/, served by api/hands or copied to a
// static build's data/hands/), drawn on a canvas laid exactly over the displayed image. It is only a drawing: nothing
// here touches the rail, the counts, the Episode JSON download or the JSON Lines export.
// the switch: on for a first-time visitor, and whatever the visitor last chose after that (stored in this browser; a
// browser that blocks storage gets the default on every load)
let HP_ON = true;
try { HP_ON = localStorage.getItem('board.handPose') !== '0'; } catch (e) {}
const _hpCache = new Map();         // file -> Promise of the decoded keypoints (or null when there are none)
const _hpData = new Map();          // file -> the decoded keypoints, once loaded
const HP_B64 = new Int8Array(128).fill(-1);
'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'.split('').forEach((c, i) => {
  HP_B64[c.charCodeAt(0)] = i; });
const HP_COLOR = {left: '#4f6d8f', right: '#8a5a9a'};    // the board's left and right (--arm-left, --arm-right)
const HP_FADE = 4;                  // frames over which a hand fades in where it appears and out where it goes
function handsUrl(file) {
  return STATIC ? BOARD.data + 'hands/' + encodeURIComponent(file) : 'api/hands?file=' + encodeURIComponent(file);
}
// Base64 VLQ integers (board/hands.py documents the format)
function hpReader(s) {
  let p = 0;
  const next = () => {
    let v = 0, sh = 0, d;
    do { d = HP_B64[s.charCodeAt(p++)]; if (d < 0 || p > s.length) throw new Error('bad hand pose data');
      v += (d & 31) * 2 ** sh; sh += 5; } while (d & 32);
    return v % 2 ? -(v - 1) / 2 : v / 2;
  };
  return {next, done: () => p === s.length};
}
function hpKnots(s, e, step) {
  const k = [s];
  for (let f = s + ((step - s % step) % step); f < e - 1; f += step) if (f > s) k.push(f);
  if (e - 1 > s) k.push(e - 1);
  return k;
}
function hpDecode(doc) {
  const n = doc.clip.frames, step = doc.step;
  // when each frame is shown: its timestamp, in seconds
  const tr = hpReader(doc.clip.pts), [tn, td] = doc.clip.tb, times = new Float64Array(n);
  let pts = tr.next();
  times[0] = pts * tn / td;
  for (let i = 1; i < n; i++) { pts += doc.clip.dur + tr.next(); times[i] = pts * tn / td; }
  if (!tr.done()) throw new Error('bad clip timing');
  const hands = {};
  for (const h of ['left', 'right']) {
    const H = doc[h];
    if (!H) continue;
    const xy = new Float32Array(n * 42), r = hpReader(H.data), prev = new Float64Array(42), a = new Float64Array(42);
    for (const [s, e] of H.spans) {
      let ka = -1;
      for (const k of hpKnots(s, e, step)) {
        for (let c = 0; c < 42; c++) prev[c] += r.next();
        xy.set(prev, k * 42);
        if (ka >= 0) {
          for (let f = ka + 1; f < k; f++) {
            const fix = r.next();
            for (let c = 0; c < 42; c++) {
              let v = a[c] + (prev[c] - a[c]) * (f - ka) / (k - ka);
              if (fix) v += r.next();
              xy[f * 42 + c] = v;
            }
          }
        }
        a.set(prev); ka = k;
      }
    }
    if (!r.done()) throw new Error('bad hand pose data');
    hands[h] = {xy, spans: H.spans};
  }
  return {n, w: doc.clip.w, h: doc.clip.h, times, edges: doc.edges, hands};
}
function loadHands(file) {
  if (!_hpCache.has(file)) {
    _hpCache.set(file, fetch(handsUrl(file), STATIC ? {} : {cache: 'no-store'})
      .then(r => r.ok ? r.json() : null).then(doc => doc ? hpDecode(doc) : null)
      .then(d => { if (d) _hpData.set(file, d); return d; }).catch(() => null));
  }
  return _hpCache.get(file);
}
// the frame on screen: rVFC's mediaTime is a frame's own timestamp (nearest); currentTime falls within one (the last
// frame at or before it, allowing the browser's microsecond rounding)
function hpFrameAt(times, t, exact) {
  let lo = 0, hi = times.length - 1;
  if (t <= times[0]) return 0;
  if (t >= times[hi]) return hi;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (times[m] <= t) lo = m; else hi = m; }
  if (exact) return (t - times[lo] <= times[hi] - t) ? lo : hi;
  return (times[hi] - t < 2e-5) ? hi : lo;
}
// how strongly to draw a hand on frame i: 0 when absent, ramping over HP_FADE frames where it appears or goes
function hpAlpha(spans, i, n) {
  let lo = 0, hi = spans.length - 1;
  while (lo <= hi) {
    const m = (lo + hi) >> 1, [s, e] = spans[m];
    if (i < s) hi = m - 1; else if (i >= e) lo = m + 1;
    else return Math.min(1, s === 0 ? 1 : (i - s + 1) / (HP_FADE + 1), e === n ? 1 : (e - i) / (HP_FADE + 1));
  }
  return 0;
}
const hpBtn = document.getElementById('hp-btn');
hpBtn.setAttribute('aria-pressed', String(HP_ON));
hpBtn.addEventListener('click', () => {
  HP_ON = !HP_ON;
  hpBtn.setAttribute('aria-pressed', String(HP_ON));
  try { localStorage.setItem('board.handPose', HP_ON ? '1' : '0'); } catch (e) {}
  if (window._hp) window._hp.set(HP_ON);
});
// wire the hand pose for the episode renderEp just laid out (its listeners go with the render's abort signal)
function setupHandPose(vid, cell, isEgo, on, file) {
  window._hp = null;
  const cv = document.getElementById('hp-canvas');
  if (!BOARD.hands || !isEgo || !vid || !cell || !cv) { hpBtn.hidden = true; return; }
  hpBtn.hidden = false;
  const ctx = cv.getContext('2d');
  let data = null, active = false, want = false, seen = true, rv = 0, raf = 0, last = '', frame = -1, box = null,
    hover = 0, hoverT = 0;
  // the same footage re-rendered (another annotation source keeps the playing video): no fade, the pose stays put
  const sameVideo = window._hpVid === vid;
  window._hpVid = vid;
  const alive = () => document.body.contains(cv);
  // lay the canvas over the image the video actually shows (object-fit: contain), in the cell's coordinates
  function place() {
    if (!vid.videoWidth || !vid.videoHeight || !vid.clientWidth || !vid.clientHeight) return false;
    const ar = vid.videoWidth / vid.videoHeight, elW = vid.clientWidth, elH = vid.clientHeight;
    let dW = elW, dH = elW / ar;
    if (dH > elH) { dH = elH; dW = elH * ar; }
    const cr = cell.getBoundingClientRect(), vr = vid.getBoundingClientRect();
    const x = vr.left - cr.left + vid.clientLeft + (elW - dW) / 2, y = vr.top - cr.top + vid.clientTop + (elH - dH) / 2;
    const dpr = window.devicePixelRatio || 1;
    const key = [x, y, dW, dH, dpr].map(v => v.toFixed(2)).join(',');
    if (box && box.key === key) return false;
    box = {key, x, y, w: dW, h: dH, dpr};
    Object.assign(cv.style, {left: x + 'px', top: y + 'px', width: dW + 'px', height: dH + 'px'});
    cv.width = Math.max(1, Math.round(dW * dpr)); cv.height = Math.max(1, Math.round(dH * dpr));
    // the player's controls sit along the bottom of the video element: the part of the image under them
    const bar = Math.max(0, Math.min(dH, 52 - (elH - dH) / 2));
    cv.style.setProperty('--hp-bar', bar.toFixed(1) + 'px');
    last = '';
    return true;
  }
  function draw(i) {
    if (!data || !active || !box) return;
    i = Math.max(0, Math.min(data.n - 1, i));
    frame = i;
    const al = {left: data.hands.left ? hpAlpha(data.hands.left.spans, i, data.n) : 0,
                right: data.hands.right ? hpAlpha(data.hands.right.spans, i, data.n) : 0};
    const key = `${i}|${al.left}|${al.right}|${box.key}`;
    if (key === last) return;           // the same picture: nothing to do
    last = key;
    const sx = cv.width / data.w, sy = cv.height / data.h;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, cv.width, cv.height);
    const lw = Math.max(1.4, Math.min(3, box.w / 480)) * box.dpr;   // thin: about 1.5 px on the board, 3 px full screen
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    for (const h of ['left', 'right']) {
      if (!al[h]) continue;
      const xy = data.hands[h].xy, o = i * 42;
      const X = j => xy[o + 2 * j] * sx, Y = j => xy[o + 2 * j + 1] * sy;
      ctx.globalAlpha = al[h];
      ctx.beginPath();
      for (const [p, q] of data.edges) { ctx.moveTo(X(p), Y(p)); ctx.lineTo(X(q), Y(q)); }
      ctx.strokeStyle = 'rgba(255,255,255,0.55)'; ctx.lineWidth = lw + 2 * box.dpr; ctx.stroke();
      ctx.strokeStyle = HP_COLOR[h]; ctx.lineWidth = lw; ctx.stroke();
      ctx.beginPath();
      for (let j = 0; j < 21; j++) { ctx.moveTo(X(j) + lw * 1.3, Y(j)); ctx.arc(X(j), Y(j), lw * 1.3, 0, 2 * Math.PI); }
      ctx.fillStyle = HP_COLOR[h]; ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.8)'; ctx.lineWidth = 0.8 * box.dpr; ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }
  const nowFrame = () => data ? hpFrameAt(data.times, vid.currentTime, false) : 0;
  // playing: one draw per presented frame (requestVideoFrameCallback), else per animation frame on currentTime
  function onVF(now, md) {
    rv = 0;
    if (!active || !seen || !alive()) return;
    draw(hpFrameAt(data.times, md.mediaTime, true)); watch();
  }
  // (while playing, currentTime runs about a tenth of a frame either side of the presented frame: nearest, not last)
  function onRaf() {
    raf = 0;
    if (!active || !seen || !alive() || vid.paused) return;
    draw(hpFrameAt(data.times, vid.currentTime, true)); raf = requestAnimationFrame(onRaf);
  }
  function watch() {
    if (!active || !seen) return;
    if (vid.requestVideoFrameCallback) { if (!rv) rv = vid.requestVideoFrameCallback(onVF); }
    else if (!raf && !vid.paused) raf = requestAnimationFrame(onRaf);
  }
  function stop() {
    if (rv && vid.cancelVideoFrameCallback) vid.cancelVideoFrameCallback(rv);
    if (raf) cancelAnimationFrame(raf);
    rv = 0; raf = 0; last = '';
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, cv.width, cv.height);
  }
  // the player's controls show while paused and for a moment after the pointer moves over it
  function controls() { cv.classList.toggle('ctl', vid.paused || hover > 0); }
  // the keypoints are fetched the first time the switch is on for this episode, never before
  function load() {
    loadHands(file).then(d => {
      if (!alive() || file !== _activeFile) return;
      if (!d) { hpBtn.hidden = true; return; }    // no hand pose for this episode after all
      data = d;
      if (want) set(true);
    });
  }
  function set(on_, instant) {
    want = on_;
    if (want && !data) { load(); return; }
    if (want && data) {
      active = true; place(); draw(nowFrame()); watch();
      if (instant) { cv.style.transition = 'none'; cv.classList.add('on'); void cv.offsetWidth;
        cv.style.transition = ''; }
      else requestAnimationFrame(() => { if (want) cv.classList.add('on'); });   // after the first picture: the fade
    } else if (cv.classList.contains('on')) {
      cv.classList.remove('on');     // the fade out; drawing stops when it ends (transitionend below)
    } else {
      active = false; stop();        // switched off before it began to show: nothing to fade
    }
  }
  on(cv, 'transitionend', e => {
    if (e.propertyName === 'opacity' && !cv.classList.contains('on')) { active = false; stop(); }
  });
  on(vid, 'play', () => { controls(); watch(); });
  on(vid, 'pause', () => { controls(); if (active) draw(nowFrame()); });
  on(vid, 'seeked', () => { if (active) draw(nowFrame()); });
  on(vid, 'loadedmetadata', () => { if (place() && active) draw(frame >= 0 ? frame : nowFrame()); });
  const pointer = () => {
    hover = 1; controls(); clearTimeout(hoverT);
    hoverT = setTimeout(() => { hover = 0; controls(); }, 2600);
  };
  on(cell, 'pointermove', pointer);
  on(cell, 'pointerdown', pointer);
  on(cell, 'pointerleave', () => { clearTimeout(hoverT); hover = 0; controls(); });
  const ro = new ResizeObserver(() => { if (place() && active) draw(frame >= 0 ? frame : nowFrame()); });
  ro.observe(vid); ro.observe(cell);
  // nothing is drawn while the video is scrolled out of view
  const io = new IntersectionObserver(es => {
    seen = es[es.length - 1].isIntersecting;
    if (seen && active) { draw(nowFrame()); watch(); }
  });
  io.observe(vid);
  const onFsHp = () => requestAnimationFrame(() => { if (place() && active) draw(frame >= 0 ? frame : nowFrame()); });
  document.addEventListener('fullscreenchange', onFsHp);
  document.addEventListener('webkitfullscreenchange', onFsHp);
  window._epCleanup.push(() => {
    ro.disconnect(); io.disconnect();
    document.removeEventListener('fullscreenchange', onFsHp);
    document.removeEventListener('webkitfullscreenchange', onFsHp);
    clearTimeout(hoverT); stop();
  });
  controls();
  window._hp = {set, get state() { return {active, frame, data: !!data, box}; }};
  if (_hpData.has(file)) data = _hpData.get(file);
  if (HP_ON) set(true, sameVideo && !!data);
}

// ================= the recording's other signals and depth (board/sensors.py) =================
// An episode whose recording has other signals (a force, joint velocities, a pressure map) or depth streams has a file
// of them (<board>/sensors/, served by api/sensors or copied to a static build's data/sensors/), and index.json lists
// those episodes with the cameras that have depth. The panel under the timeline and each camera's depth switch are
// drawn from it; nothing here touches the rail, the counts or the downloads.
let SN_INDEX = null;                 // {file: {signals, constant, depth: [camera view, ...]}}
const _snCache = new Map();          // file -> Promise of the decoded file (or null), the most recently used last
const _snData = new Map();           // file -> the decoded file, once loaded
let SN_OPEN = false;                 // every lane shown, kept across episodes
let SN_SHOWN = false;                // the folded panel of an episode with contacts opened, kept across episodes
const DP_ON = new Set();             // the cameras shown in depth, kept across episodes
const SN_FIRST = 4;                  // lanes shown before "Show all"
const SN_REST = 0.1;                 // label/signals.py REST_FRACTION: a cell this far from rest (of its swing) is active
const snIndexUrl = () => STATIC ? BOARD.data + 'sensors/index.json' : 'api/sensors?file=index.json';
function sensorsUrl(file) {
  return STATIC ? BOARD.data + 'sensors/' + encodeURIComponent(file) : 'api/sensors?file=' + encodeURIComponent(file);
}
// one block of samples (board/sensors.py quantize): rows x values floats, NaN where there is no reading
function snBlock(blk, rows, bits) {
  const b = atob(blk.data), u = new Uint8Array(b.length);
  for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i);
  const q = bits === 16 ? new Uint16Array(u.buffer, 0, u.length >> 1) : u, none = bits === 16 ? 65535 : 255;
  const lo = [].concat(blk.lo), st = [].concat(blk.step), dims = rows ? q.length / rows : 0;
  const v = new Float32Array(q.length);
  for (let i = 0; i < q.length; i++) {
    const c = lo.length > 1 ? i % dims : 0;
    v[i] = q[i] === none ? NaN : lo[c] + q[i] * st[c];
  }
  return {dims, v};
}
function snDecode(doc) {
  const out = {depth: doc.depth || {}, signals: [], constant: [], t: new Float64Array(0), errors: doc.errors || []};
  if (!doc.signals || !doc.frames) return out;
  const n = doc.n, stride = doc.stride || 1, ft = new Float64Array(doc.frames), tr = hpReader(doc.times.d);
  let ms = doc.times.ms0;
  ft[0] = ms / 1000;
  for (let i = 1; i < doc.frames; i++) { ms += doc.times.dur + tr.next(); ft[i] = ms / 1000; }
  out.t = new Float64Array(n);
  for (let i = 0; i < n; i++) out.t[i] = ft[Math.min(doc.frames - 1, i * stride)];
  for (const s of doc.signals) {
    if (s.constant) { out.constant.push(s); continue; }
    const g = Object.assign({}, s);
    if (s.values) g.vals = snBlock(s.values, n, 16);
    if (s.activity) g.act = snBlock(s.activity, n, 16).v;
    if (s.strength) g.str = snBlock(s.strength, n, 16).v;
    if (s.map && s.rest && s.swing > 0) { g.map = snBlock(s.map, n, 8).v; g.restArr = Float32Array.from(s.rest); }
    out.signals.push(g);
  }
  // what touches first, then what rests and rises, then the rest, each in the dataset's order
  const rank = s => s.touch ? 0 : s.rests_and_rises ? 1 : 2;
  out.signals = out.signals.map((s, i) => [s, i]).sort((a, b) => rank(a[0]) - rank(b[0]) || a[1] - b[1]).map(x => x[0]);
  return out;
}
function loadSensors(file) {
  if (!BOARD.sensors || !SN_INDEX || !SN_INDEX[file]) return Promise.resolve(null);
  let p = _snCache.get(file);
  if (p) { _snCache.delete(file); _snCache.set(file, p); return p; }
  p = fetchJson(sensorsUrl(file)).then(doc => {
    const d = doc && doc.format === 'board-sensors/1' ? snDecode(doc) : null;
    if (d) _snData.set(file, d);
    return d;
  }).catch(() => null);
  _snCache.set(file, p);
  while (_snCache.size > EP_CACHE_MAX) { const k = _snCache.keys().next().value; _snCache.delete(k); _snData.delete(k); }
  return p;
}
// the cameras of an episode with a depth clip to switch to
function depthViews(file) {
  if (STATIC) {
    const rec = ALL_EPS.find(e => e.file === file) || {};
    return Object.keys(rec._media || {}).filter(k => k.startsWith('depth_')).map(k => k.slice(6));
  }
  return BOARD.sensors && SN_INDEX && SN_INDEX[file] ? (SN_INDEX[file].depth || []) : [];
}
// on its own line after the camera's video, so a camera without depth draws exactly as before
function dpHtml(v, views) {
  if (!views.includes(v)) return '';
  return `\n          <video class="dp-vid" data-view="${esc(v)}" muted playsinline preload="none" aria-hidden="true"></video>`
    + `<button class="cam-dp" type="button" data-view="${esc(v)}" aria-pressed="false" title="Show this camera's depth `
    + `in place of its colour picture"><span class="hp-sw" aria-hidden="true"></span>Depth</button>`;
}
// three significant digits, as label/signals.py writes a value
const snNum = v => !isFinite(v) ? '-' : String(Number(v.toPrecision(3)));
const snOpacity = (i, n) => n > 1 ? (1 - 0.65 * i / (n - 1)).toFixed(2) : '1';
function snValueNames(s) {
  const d = s.vals ? s.vals.dims : 0;
  return s.names && s.names.length === d ? s.names : (d === 1 ? [''] : [...Array(d).keys()].map(i => `[${i}]`));
}
// the sample shown at time t: the last one at or before it
function snIndexAt(ts, t) {
  if (!ts.length || t < ts[0]) return ts.length ? 0 : -1;
  let lo = 0, hi = ts.length - 1;
  while (lo < hi) { const m = (lo + hi + 1) >> 1; if (ts[m] <= t + 1e-3) lo = m; else hi = m - 1; }
  return lo;
}
function snWhat(s) {
  const sh = s.shape && s.shape.length > 1 ? s.shape.join(' x ') : String(s.dims);
  let w = s.act ? `${sh} values, drawn as their activity` : `${sh} ${s.dims === 1 ? 'value' : 'values'}`;
  if (s.touch || s.rests_and_rises) {
    if (s.direction === 'down') w += ', falls when active';
    else if (s.direction === 'up') w += ', rises when active';
    const k = (s.spans || []).length;
    w += k ? `, away from rest ${k === 1 ? 'once' : k + ' times'}` : '';
  }
  // placed on the video from both starts, because the recording shares no clock with the cameras
  // (prepare/formats.py mark_assumed), as the prompt says it
  if (s.aligned_by) w += ', placed from both starts, as no clock is shared';
  return w;
}
// the signals the board could not draw (board/sensors.py "errors"), each named with the reason, under the lanes it drew
function snErrorsHtml(errors) {
  const xs = (Array.isArray(errors) ? errors : []).filter(x => x && x.name);
  if (!xs.length) return '';
  const each = xs.map(x => `${esc(x.name)} (${esc(x.error || 'no reason given')})`);
  const list = each.length > 1 ? `${each.slice(0, -1).join(', ')} and ${each[each.length - 1]}` : each[0];
  return `<div class="sn-note">${list} could not be drawn, so ${xs.length === 1 ? 'it has' : 'they have'} no lane `
    + `here.</div>`;
}
// one lane's strip: each value a line on one scale (an array, its activity filled from 0), on the timeline's time scale
function snPlot(s, ts, duration) {
  const W = 1000, H = 100, pad = 8, x = t => (W * t / duration).toFixed(1);
  const series = s.vals ? [...Array(s.vals.dims).keys()].map(c => i => s.vals.v[i * s.vals.dims + c]) : [i => s.act[i]];
  let lo = s.vals ? Infinity : 0, hi = -Infinity;
  for (let i = 0; i < ts.length; i++) for (const f of series) { const v = f(i); if (isFinite(v)) { lo = Math.min(lo, v);
    hi = Math.max(hi, v); } }
  if (!(hi > lo)) { hi = lo + 1; }
  const y = v => (H - pad - (H - 2 * pad) * (v - lo) / (hi - lo)).toFixed(1);
  const paths = series.map((f, c) => {
    let d = '', pen = false;
    for (let i = 0; i < ts.length && ts[i] <= duration; i++) {
      const v = f(i);
      if (!isFinite(v)) { pen = false; continue; }
      d += `${pen ? 'L' : 'M'}${x(ts[i])} ${y(v)}`;
      pen = true;
    }
    return d;
  });
  let svg = '';
  if (!s.vals && paths[0]) {
    const last = Math.min(ts.length, snIndexAt(ts, duration) + 1) - 1;
    svg += `<path class="area" d="${paths[0]}L${x(ts[Math.max(0, last)])} ${y(0)}L${x(ts[0])} ${y(0)}Z"></path>`;
  }
  svg += paths.map((d, c) => d ? `<path d="${d}" style="stroke-opacity:${snOpacity(c, paths.length)}"></path>` : '')
    .join('');
  const pct = t => (100 * Math.max(0, Math.min(duration, t)) / duration);
  const spans = (s.spans || []).filter(([a]) => a < duration).map(([a, b]) => `<div class="sn-span" style="left:`
    + `${pct(a)}%;width:max(2px, ${pct(b) - pct(a)}%)"></div>`).join('');
  return `${spans}<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">${svg}</svg>`
    + `<div class="sn-hv"></div><div class="lane-ph"></div>`;
}
// the readout under the playhead: each value with its name and the swatch of its line, or the array's activity
function snReadout(s, i) {
  if (i < 0) return '';
  if (s.vals) {
    const names = snValueNames(s), d = s.vals.dims;
    return names.map((nm, c) => `<span class="sn-v">${d > 1 ? `<i style="opacity:${snOpacity(c, d)}"></i>` : ''}`
      + `${esc(nm)}${nm ? ' ' : ''}<span class="sn-n">${snNum(s.vals.v[i * d + c])}</span></span>`).join('');
  }
  return `<span class="sn-v">activity <span class="sn-n">${snNum(s.act[i])}</span></span>`;
}
function snTipText(s, i) {
  if (s.vals) {
    const names = snValueNames(s), d = s.vals.dims;
    return names.map((nm, c) => `${nm ? esc(nm) + ' ' : ''}${snNum(s.vals.v[i * d + c])}`).join(', ');
  }
  return `activity ${snNum(s.act[i])}`;
}
const snAway = dir => dir === 'down' ? 'below' : dir === 'up' ? 'above' : 'away from';
// a 2-D array's cells at sample i: each cell's distance from its resting level in the signal's direction, over the
// swing (the upload's when prepare measured it), clipped to 0..1
function snCells(s, i) {
  const n = s.restArr.length, o = i * n, out = new Float32Array(n);
  let best = -1, bi = -1, none = true;
  for (let c = 0; c < n; c++) {
    const v = s.map[o + c];
    if (!isFinite(v)) { out[c] = NaN; continue; }
    none = false;
    const d = s.direction === 'down' ? s.restArr[c] - v : s.direction === 'up' ? v - s.restArr[c] : Math.abs(v
      - s.restArr[c]);
    out[c] = Math.max(0, Math.min(1, d / s.swing));
    if (d > best) { best = d; bi = c; }
  }
  return {cells: out, best, bi, none};
}
function snMapHtml(s, k) {
  const [rows, cols] = s.shape;
  const scope = s.swing_from === 'upload' ? 'the typical swing across the dataset' : 'the typical swing in this episode';
  // the longer side 112 px, so a long narrow array (21 x 3) stays as short as a square one
  const w = Math.max(16, Math.round(112 * cols / Math.max(rows, cols)));
  return `<div class="sn-map" data-k="${k}">
    <div class="sn-grid" style="width:${w}px"><canvas width="${cols}" height="${rows}"></canvas><div class="sn-peak" style="width:`
      + `${100 / cols}%;height:${100 / rows}%"></div></div>
    <span class="lane-title">${esc(s.name)}</span>
    <span class="lane-now"></span>
    <span class="sn-note">Each cell's distance from its resting level at the playhead. Darker is farther, and black `
      + `is ${snNum(s.swing)} ${snAway(s.direction)} rest or more, ${scope}.</span>
  </div>`;
}
function snDepthHtml(D, camName, order) {
  const groups = new Map();
  const rank = v => (order.indexOf(v) + 1) || 99;
  for (const [v, e] of Object.entries(D.depth || {}).sort((a, b) => rank(a[0]) - rank(b[0]))) {
    const key = JSON.stringify([e.units, e.kind, e.bar, e.ticks]);
    if (!groups.has(key)) groups.set(key, {e, views: []});
    groups.get(key).views.push(v);
  }
  return [...groups.values()].map(({e, views}) => {
    const names = views.map(camName);
    const cams = names.length === 1 ? names[0] : names.slice(0, -1).join(', ') + ' and ' + names[names.length - 1];
    const one = views.length === 1;
    const text = (e.units === 'metres'
      ? 'Red is near and blue is far, in metres on one fixed scale that is the same in every episode. Black is no '
        + 'reading.'
      : 'Red is near and blue is far. The recording does not give the unit, so the colours are relative to '
        + `${one ? "this camera's" : "each camera's"} readings across the upload. Black is no reading.`)
      + ` The Depth switch on ${one ? 'its picture shows it' : 'each picture shows it'} in place of the colour picture.`;
    const bar = (e.bar || []).length ? `<div class="sn-bar" style="background:linear-gradient(to right, `
      + `${e.bar.join(', ')})"></div><div class="sn-ticks">${(e.ticks || []).map(([p, label]) => `<span style="left:`
      + `${(100 * p).toFixed(2)}%;transform:translateX(-${(100 * p).toFixed(2)}%);--tx:${(100 * p).toFixed(2)}%">`
      + `${esc(label)}</span>`).join('')}</div>` : '';
    return `<div class="sn-depth"><span class="lane-title">Depth <span class="lane-sum">${esc(cams)} `
      + `${one ? 'camera' : 'cameras'}</span></span><div class="sn-note">${text}</div>${bar}</div>`;
  }).join('');
}
// the colours a heatmap is drawn in: the page's background where a cell is at rest, its ink where it is farthest
function snInk() {
  const css = getComputedStyle(document.documentElement);
  const rgb = h => { const m = String(h).trim().match(/^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i);
    return m ? [1, 2, 3].map(j => parseInt(m[j], 16)) : null; };
  return [rgb(css.getPropertyValue('--bg')) || [243, 244, 246], rgb(css.getPropertyValue('--fg')) || [28, 28, 26]];
}
// one heatmap (snMapHtml) bound to its signal, and drawn at sample i with its strongest cell marked and named
function snMapBind(el, s) {
  return {s, el, ctx: el.querySelector('canvas').getContext('2d'), img: null, peak: el.querySelector('.sn-peak'),
          now: el.querySelector('.lane-now')};
}
function snMapDraw(m, i, ink) {
  const s = m.s, [rows, cols] = s.shape, [c0, c1] = ink;
  if (!m.img) m.img = m.ctx.createImageData(cols, rows);
  if (i < 0) return;
  const {cells, best, bi, none} = snCells(s, i), px = m.img.data;
  for (let c = 0; c < cells.length; c++) {
    const x = cells[c], o = 4 * c;
    for (let k = 0; k < 3; k++) px[o + k] = isFinite(x) ? Math.round(c0[k] + (c1[k] - c0[k]) * x) : c0[k];
    px[o + 3] = 255;
  }
  m.ctx.putImageData(m.img, 0, 0);
  const active = !none && bi >= 0 && best / s.swing >= SN_REST;
  if (active) {
    m.peak.style.left = (100 * (bi % cols) / cols) + '%';
    m.peak.style.top = (100 * Math.floor(bi / cols) / rows) + '%';
  }
  m.peak.classList.toggle('on', active);
  m.now.textContent = none ? 'No reading at the playhead' : active ? `Row ${Math.floor(bi / cols) + 1}, column `
    + `${bi % cols + 1} is the strongest, ${snNum(best)} ${snAway(s.direction)} rest` : 'At rest';
}
// lay out the panel for the episode renderEp just drew, and wire it to the playhead (its listeners go with the render).
// On an episode with contacts the Touch lane above shows what matters, so the signals start folded away.
function setupSensors(file, duration, seek, on, vid, camName, order, hasContacts) {
  window._sn = null;
  const slot = document.getElementById('sn-slot');
  if (!slot || !BOARD.sensors || !SN_INDEX || !SN_INDEX[file]) return;
  const fill = (D, fade) => {
    if (!D || !document.body.contains(slot) || file !== _activeFile) return;
    const sigs = D.signals, maps = sigs.filter(s => s.map && s.shape && s.shape.length === 2);
    const nDepth = Object.keys(D.depth || {}).length;
    if (!sigs.length && !D.constant.length && !nDepth && !(D.errors || []).length) return;
    const counts = [];
    if (sigs.length || D.constant.length) counts.push(`${sigs.length} ${sigs.length === 1 ? 'signal changes'
      : 'signals change'}${D.constant.length ? `, ${D.constant.length} constant` : ''}`);
    if (nDepth) counts.push(`depth on ${nDepth} ${nDepth === 1 ? 'camera' : 'cameras'}`);
    const lane = (s, i) => `<div class="lane sn-lane">
        <div class="lane-head"><span class="lane-title">${esc(s.name)} <span class="lane-sum">${esc(snWhat(s))}`
          + `</span></span><span class="lane-now" data-i="${i}"></span></div>
        <div class="lane-bar sn-plot" data-i="${i}">${snPlot(s, D.t, duration)}</div>
      </div>`;
    const first = sigs.slice(0, SN_FIRST).map(lane).join('');
    const more = sigs.slice(SN_FIRST).map((s, j) => lane(s, j + SN_FIRST)).join('');
    const constHtml = D.constant.length ? `<div class="sn-note">Constant through this episode: ${D.constant.map(s =>
      `${esc(s.name)} (${s.dims > 4 || !s.value ? `${s.shape && s.shape.length > 1 ? s.shape.join(' x ')
        : s.dims} values` : s.value.map(snNum).join(', ')})`).join(', ')}.</div>` : '';
    const signalsHtml = `${maps.length ? `<div class="sn-maps">${maps.map((s) => snMapHtml(s, sigs.indexOf(s)))
        .join('')}</div>` : ''}
        ${first}
        ${more ? `<div class="ck-all"><div class="ck-all-in">${more}</div></div><button class="ck-more sn-more" `
          + `type="button">${SN_OPEN ? 'Hide the other signals' : `Show all ${sigs.length} signals`}</button>` : ''}
        ${constHtml}${snErrorsHtml(D.errors)}`;
    const fold = !!hasContacts && (sigs.length > 0 || D.constant.length > 0);
    const shown = !fold || SN_SHOWN;
    const foldWord = on_ => on_ ? 'Hide the recorded signals' : `Show all recorded signals`;
    slot.innerHTML = `<h3 class="section sn-h">All recorded signals <span class="count">${counts.join(', ')}</span></h3>
      <div class="sn${SN_OPEN ? ' open' : ''}${shown ? ' shown' : ''}">
        ${fold ? `<div class="sn-fold"><div class="sn-fold-in">${signalsHtml}</div></div><button class="ck-more `
          + `sn-show" type="button" aria-expanded="${shown}">${foldWord(shown)}</button>` : signalsHtml}
        ${snDepthHtml(D, camName, order || [])}
      </div>`;
    if (fade) { slot.classList.add('sn-wait'); void slot.offsetWidth; slot.classList.remove('sn-wait'); }
    const box = slot.querySelector('.sn');
    const moreBtn = slot.querySelector('.sn-more');
    if (moreBtn) moreBtn.addEventListener('click', () => {
      SN_OPEN = !SN_OPEN;
      box.classList.toggle('open', SN_OPEN);
      moreBtn.textContent = SN_OPEN ? 'Hide the other signals' : `Show all ${sigs.length} signals`;
    });
    const showBtn = slot.querySelector('.sn-show');
    if (showBtn) showBtn.addEventListener('click', () => {
      SN_SHOWN = !box.classList.contains('shown');
      box.classList.toggle('shown', SN_SHOWN);
      showBtn.textContent = foldWord(SN_SHOWN);
      showBtn.setAttribute('aria-expanded', String(SN_SHOWN));
    });
    const ink = snInk();
    const mapEls = [...slot.querySelectorAll('.sn-map')].map(el => snMapBind(el, sigs[+el.dataset.k]));
    const nows = [...slot.querySelectorAll('.sn-lane .lane-now')];
    const phs = [...slot.querySelectorAll('.sn-plot .lane-ph')];
    let last = -2;
    function draw(i) {
      nows.forEach(el => { el.innerHTML = snReadout(sigs[+el.dataset.i], i); });
      for (const m of mapEls) snMapDraw(m, i, ink);
    }
    function sync(t) {
      const pct = (100 * Math.max(0, Math.min(duration, t)) / duration) + '%';
      for (const p of phs) p.style.left = pct;
      const i = snIndexAt(D.t, t);
      if (i === last) return;
      last = i;
      draw(i);
    }
    slot.querySelectorAll('.sn-plot').forEach(plot => {
      const s = sigs[+plot.dataset.i], hv = plot.querySelector('.sn-hv');
      const tAt = e => { const r = plot.getBoundingClientRect();
        return Math.max(0, Math.min(1, (e.clientX - r.left) / Math.max(1, r.width))) * duration; };
      plot.addEventListener('pointermove', e => {
        const t = tAt(e), i = snIndexAt(D.t, t);
        hv.style.left = (100 * t / duration) + '%';
        plot.classList.add('hover');
        if (i < 0) return;
        plot.dataset.tip = `<b>${snTipText(s, i)}</b>${esc(s.name)} at ${D.t[i].toFixed(2)} s`;
        tipShow(plot, e.clientX, e.clientY);
      });
      plot.addEventListener('pointerleave', () => { plot.classList.remove('hover'); cmpTip.classList.remove('show'); });
      plot.addEventListener('click', e => { cmpTip.classList.remove('show'); seek(tAt(e)); });
    });
    // playing: one update per presented frame, as the hand pose does, so the heatmap keeps up with the footage
    let rv = 0, raf = 0;
    const alive = () => document.body.contains(slot) && file === _activeFile;
    function onVF(now, md) { rv = 0; if (!alive()) return; sync(md.mediaTime); watch(); }
    function onRaf() { raf = 0; if (!alive() || vid.paused) return; sync(vid.currentTime); raf = requestAnimationFrame(onRaf); }
    function watch() {
      if (!vid || vid.paused) return;
      if (vid.requestVideoFrameCallback) { if (!rv) rv = vid.requestVideoFrameCallback(onVF); }
      else if (!raf) raf = requestAnimationFrame(onRaf);
    }
    if (vid) {
      on(vid, 'play', watch);
      on(vid, 'seeked', () => sync(vid.currentTime));
    }
    window._epCleanup.push(() => {
      if (rv && vid && vid.cancelVideoFrameCallback) vid.cancelVideoFrameCallback(rv);
      if (raf) cancelAnimationFrame(raf);
      cmpTip.classList.remove('show');
    });
    window._sn = {sync};
    sync(vid ? vid.currentTime : 0);
    watch();
  };
  if (_snData.has(file)) fill(_snData.get(file), false);
  else loadSensors(file).then(D => fill(D, true));
}
// ================= touch: the recording's contacts and what the model saw at each (board/build.py add_contacts) =========
// d.contacts are the spans in which a hand's touch signals say it touches something (label/contacts.py), each with the
// model's answer when it was shown frames around it; d.contacts_missing the moments the model saw a hand take hold of
// something that no contact covers. The Touch lane draws them on the timeline's scale, one bar per hand, and the
// contact card under it says what is known about the contact under the playhead.
const TC_WORD = {yes: 'The frames show touch', no: 'The frames show no touch', unclear: 'Unclear in the frames',
                 unshown: 'Not shown to the model', unanswered: 'Shown, with no answer'};
// what the model found at a contact: yes, no and unclear as it answered, unshown when it was not shown the contact,
// unanswered when it was shown and left it out
function tcState(c) {
  if (!c.shown) return 'unshown';
  if (!c.seen) return 'unanswered';
  const v = String(c.seen.touch_seen || '').toLowerCase();
  return v === 'yes' || v === 'no' ? v : 'unclear';
}
const tcCls = st => st === 'unanswered' ? 'st-unclear' : 'st-' + st;
const tcHandKey = h => { const v = String(h || '').toLowerCase(); return v === 'left' || v === 'right' ? v : ''; };
const tcHandName = h => h === 'left' ? 'Left hand' : h === 'right' ? 'Right hand' : 'Hand not named';
// a list of times in words: 3.8s, 4.1s and 5.0s
const tcList = xs => xs.length < 2 ? xs.join('') : xs.slice(0, -1).join(', ') + ' and ' + xs[xs.length - 1];
// the episode's contacts in time order and the model's other grasps, both checked for times
function tcData(d) {
  const contacts = (d.contacts || []).filter(c => c && isFinite(c.start_s) && isFinite(c.end_s))
    .slice().sort((a, b) => a.start_s - b.start_s || String(a.hand).localeCompare(String(b.hand)));
  const missing = (d.contacts_missing || []).filter(x => x && x.t_s != null && isFinite(parseFloat(x.t_s)))
    .map(x => ({...x, t_s: parseFloat(x.t_s)})).sort((a, b) => a.t_s - b.t_s);
  // one row per hand the contacts name (or one row when they name none), and a row for a hand only a missed grasp names
  const named = new Set(contacts.map(c => tcHandKey(c.hand)));
  for (const x of missing) if (tcHandKey(x.hand) && named.size && !named.has('')) named.add(tcHandKey(x.hand));
  const rows = ['left', 'right', ''].filter(h => named.has(h));
  return {contacts, missing, rows};
}
// the rows a missed grasp is marked on: its hand's, or every row when it names none (or both)
const tcMissRows = (x, rows) => rows.includes(tcHandKey(x.hand)) ? [tcHandKey(x.hand)] : rows;
// the Touch lane, in the lane markup the Hands out of view lane uses
function touchLaneHtml(T, lanePct, chev) {
  const {contacts, missing, rows} = T;
  if (!contacts.length) return '';
  const n = contacts.length, st = contacts.map(tcState);
  const checked = contacts.filter(c => c.shown && c.seen).length, notSeen = st.filter(x => x === 'no').length;
  const sum = [`${n} ${n === 1 ? 'contact' : 'contacts'}`, `${checked} checked`, `${notSeen} not seen`]
    .concat(missing.length ? [`${missing.length} ${missing.length === 1 ? 'grasp' : 'grasps'} with no contact`] : []);
  const seg = (c, i) => {
    const a = lanePct(c.start_s), b = lanePct(c.end_s), cls = tcCls(st[i]);
    const tip = `${c.id ? c.id + ', ' : ''}${fmtT(c.start_s)} to ${fmtT(c.end_s)}: ${TC_WORD[st[i]].toLowerCase()}${c.seen && c.seen.object
      ? ', ' + String(c.seen.object) : ''}`;
    return `<div class="tc-seg ${cls}" data-c="${i}" title="${esc(tip)}" style="left:${a}%;width:max(3px, ${b - a}%)">`
      + `<svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true"><path d=""></path></svg></div>`
      + (isFinite(c.peak_s) ? `<div class="tc-peak ${cls}" style="left:${lanePct(c.peak_s)}%"></div>` : '');
  };
  const miss = (x, j) => `<div class="tc-miss" data-t="${x.t_s}" data-m="${j}" title="${esc(`${fmtT(x.t_s)}: the model `
    + `saw ${tcHandKey(x.hand) ? 'the ' + tcHandKey(x.hand) + ' hand' : 'a hand'} take hold of ${x.object || 'something'}`
    + `, and no recorded contact covers it`)}" style="left:${lanePct(x.t_s)}%"></div>`;
  const rowHtml = h => `<div class="tc-row">${rows.length > 1 || h ? `<span class="tc-hand">${tcHandName(h)}</span>` : ''}`
    + `<div class="lane-bar tc-bar">${contacts.map((c, i) => tcHandKey(c.hand) === h || (!rows.includes(tcHandKey(c.hand))
      && h === rows[0]) ? seg(c, i) : '').join('')}${missing.map((x, j) => tcMissRows(x, rows).includes(h) ? miss(x, j)
      : '').join('')}<div class="lane-ph"></div></div></div>`;
  const keys = ['yes', 'no', 'unclear', 'unshown'].filter(k => st.some(x => (x === 'unanswered' ? 'unclear' : x) === k));
  const key = keys.map(k => `<span class="tc-k"><i class="st-${k}" aria-hidden="true"></i>${k === 'unclear'
    && st.includes('unanswered') ? 'Unclear in the frames, or no answer' : TC_WORD[k]}</span>`).join('')
    + (missing.length ? `<span class="tc-k"><i class="miss" aria-hidden="true"></i>A hand takes hold with no recorded `
      + `contact</span>` : '');
  return `<div class="lane lane-touch" id="lane-touch">
      <div class="lane-head">
        <span class="lane-title">Touch <span class="lane-sum">${sum.join(', ')}</span></span>
        <span class="lane-nav" role="group" aria-label="Contacts">
          <button type="button" class="lane-step" data-dir="-1" aria-label="Previous contact">${chev('M6.5 2 3.5 5l3 3')}</button>
          <span class="lane-pos" id="lane-touch-pos">${n} ${n === 1 ? 'contact' : 'contacts'}</span>
          <button type="button" class="lane-step" data-dir="1" aria-label="Next contact">${chev('M3.5 2l3 3-3 3')}</button>
        </span>
      </div>
      ${rows.map(rowHtml).join('')}
      <div class="tc-key">${key}</div>
    </div>`;
}
// where a contact bears: each map's active cells at its strongest, and which of its signals are active
function tcRegions(c) {
  const out = [], r = c.regions || {};
  for (const [nm, x] of Object.entries(r)) {
    if (nm === 'active_signals' || !x || !Array.isArray(x.of)) continue;
    const span = (a, w) => a[0] === a[1] ? `${w} ${a[0] + 1}` : `${w}s ${a[0] + 1} to ${a[1] + 1}`;
    out.push(`${nm}: ${x.cells} of its ${x.of[0] * x.of[1]} cells, ${span(x.rows, 'row')} and ${span(x.columns,
      'column')} of ${x.of[0]} x ${x.of[1]}`);
  }
  if (Array.isArray(r.active_signals)) out.push(r.active_signals.length ? `Active at its strongest: `
    + `${tcList(r.active_signals)}` : 'None of its signals is active at its strongest');
  return out;
}
function tcCardHtml(c, i, st) {
  const s = c.seen || {};
  const tt = t => `<span data-t="${t}">${fmtT(t)}</span>`;
  const times = [c.from_start ? 'Already touching at the start' : `Begins ${tt(c.start_s)}`,
    isFinite(c.peak_s) ? `strongest ${tt(c.peak_s)}` : '',
    c.to_end ? 'still touching at the end' : `ends ${tt(c.end_s)}`].filter(Boolean).join(', ');
  const kv = (k, vs) => `<div class="kv"><div class="kv-k">${k}</div>${[].concat(vs).map(v => `<div class="kv-v">${v}`
    + `</div>`).join('')}</div>`;
  const rows = [];
  if (c.shown && c.seen) {
    const told = v => v != null && String(v).trim() !== '' && String(v).toLowerCase() !== 'null';
    if (told(s.object)) rows.push(kv('Object', esc(s.object)));
    if (told(s.grip)) rows.push(kv('Grip', esc(s.grip)));
    if (told(s.action)) rows.push(kv('What it does', esc(s.action)));
    if (told(s.slip)) rows.push(kv('Slip', esc(String(s.slip).toLowerCase() === 'yes' ? 'Yes, it slips'
      : String(s.slip).toLowerCase() === 'no' ? 'No' : 'Unclear')));
    const mh = String(s.hand || '').toLowerCase(), rh = tcHandKey(c.hand);
    if (mh && mh !== rh && (mh !== 'unclear' || rh)) rows.push(kv('Hand', esc(mh === 'both' ? 'The frames show both hands touching'
      : mh === 'left' || mh === 'right' ? `The frames show the ${mh} hand${rh ? `, and the signal's name says ${rh}` : ''}`
      : 'Unclear in the frames')));
    if (told(s.notes)) rows.push(kv('Notes', esc(s.notes)));
  }
  const where = tcRegions(c);
  rows.push(kv('Signals', esc(tcList(c.signals || []))));
  if (where.length) rows.push(kv('Where it bears', where.map(esc)));
  const dips = (c.dips_s || []).filter(isFinite);
  rows.push(kv('Dips', dips.length ? `${tcList(dips.map(fmtT))}: its strength falls under half its peak and comes back`
    : 'None: its strength stays above half its peak'));
  const plain = !c.shown ? `<div class="tc-plain">The model was not shown this contact, so nothing here says what the `
      + `frames show at it.</div>`
    : !c.seen ? `<div class="tc-plain">The model was shown this contact and gave no answer for it.</div>` : '';
  return `<div class="tc-card info-block" data-c="${i}">
      <div class="tc-card-head"><span class="tc-card-title">${tcHandKey(c.hand) ? tcHandName(tcHandKey(c.hand)) + ', '
        + 'contact' : 'Contact'} ${esc(c.id || String(i + 1))}</span><span class="tc-pill ${tcCls(st)}">${TC_WORD[st]}</span></div>
      <div class="tc-times">${times}</div>
      ${plain}
      <div class="kv-block tc-kv">${rows.join('')}</div>
      <div class="tc-maps"></div>
    </div>`;
}
function tcCardsHtml(T) {
  if (!T.contacts.length) return '';
  return `<div class="tc-cards" id="tc-cards">
      <div class="tc-card info-block tc-empty on" data-c="-1"><div class="tc-plain">No contact at the playhead. Click one `
        + `on the Touch lane, or step through them with its arrows.</div></div>
      ${T.contacts.map((c, i) => tcCardHtml(c, i, tcState(c))).join('')}
    </div>`;
}
// a contact's strength over its span, from the sensors file: its signals' strength summed per sample, as label/contacts.py
// sums them, as an area path in a 100 x 100 box scaled to top, the strongest contact of the episode
function tcCurve(c, D, top) {
  const sigs = (c.signals || []).map(nm => D.signals.find(s => s.name === nm && s.str)).filter(Boolean);
  if (!sigs.length || !(c.end_s > c.start_s)) return '';
  const pts = [];
  for (let i = 0; i < D.t.length; i++) {
    const t = D.t[i];
    if (t < c.start_s - 1e-3 || t > c.end_s + 1e-3) continue;
    let v = 0;
    for (const s of sigs) if (isFinite(s.str[i])) v += s.str[i];
    pts.push([100 * (t - c.start_s) / (c.end_s - c.start_s), 100 - 100 * Math.max(0, Math.min(1, v / top))]);
  }
  if (!pts.length) return '';
  if (pts.length === 1) pts.push([100, pts[0][1]]), pts[0][0] = 0;
  return `M${pts[0][0].toFixed(1)} 100` + pts.map(([x, y]) => `L${x.toFixed(1)} ${y.toFixed(1)}`).join('')
    + `L${pts[pts.length - 1][0].toFixed(1)} 100Z`;
}
// wire the lane and the cards renderEp drew: which contact the playhead is in, the stepper, clicks, the strength curves
// and the heatmaps of the shown card once the sensors file is in. Returns {sync(t)}.
function setupTouch(T, file, duration, seek, on, vid) {
  const lane = document.getElementById('lane-touch'), box = document.getElementById('tc-cards');
  if (!lane || !box) return null;
  const C = T.contacts, n = C.length;
  const segs = [...lane.querySelectorAll('.tc-seg')], cards = [...box.querySelectorAll('.tc-card')];
  const pos = document.getElementById('lane-touch-pos');
  let pinned = -1, shown = -2, D = null, ink = null, maps = [], lastI = -2;
  const inside = (c, t) => c.start_s <= t + 0.05 && t <= c.end_s + 0.05;
  // the contact shown: the one clicked while the playhead is in it, else the latest begun of those it is in, else the
  // one clicked
  function current(t) {
    if (pinned >= 0 && inside(C[pinned], t)) return pinned;
    let k = -1;
    for (let i = 0; i < n; i++) if (inside(C[i], t)) k = i;
    if (k >= 0) { if (k !== pinned) pinned = -1; return k; }
    return pinned;
  }
  // every card's heatmaps are laid out when the sensors file is in, so the cards' shared height is set once; only the
  // shown card's are drawn as the playhead moves
  function layMaps() {
    cards.slice(1).forEach((el, k) => {
      const ms = (C[k].signals || []).map(nm => D.signals.find(s => s.name === nm && s.map && s.shape
        && s.shape.length === 2)).filter(Boolean);
      el.querySelector('.tc-maps').innerHTML = ms.length ? `<div class="sn-maps">${ms.map(s => snMapHtml(s,
        D.signals.indexOf(s))).join('')}</div>` : '';
    });
  }
  function bindMaps(k) {
    maps = [];
    lastI = -2;
    const slot = k >= 0 && D ? cards[k + 1].querySelector('.tc-maps') : null;
    if (slot) maps = [...slot.querySelectorAll('.sn-map')].map(el => snMapBind(el, D.signals[+el.dataset.k]));
  }
  function sync(t) {
    const k = current(t);
    if (k !== shown) {
      shown = k;
      segs.forEach(g => g.classList.toggle('now', +g.dataset.c === k));
      cards.forEach(el => el.classList.toggle('on', +el.dataset.c === k));
      if (pos) pos.textContent = k >= 0 ? `${k + 1} of ${n}` : `${n} ${n === 1 ? 'contact' : 'contacts'}`;
      bindMaps(k);
    }
    if (maps.length) {
      const i = snIndexAt(D.t, t);
      if (i !== lastI) { lastI = i; for (const m of maps) snMapDraw(m, i, ink); }
    }
  }
  const go = (k, t) => { pinned = k; shown = -2; seek(t != null ? t : (C[k].from_start ? 0 : C[k].start_s)); };
  segs.forEach(g => g.addEventListener('click', e => { e.stopPropagation(); go(+g.dataset.c); }));
  lane.querySelectorAll('.tc-miss').forEach(m => m.addEventListener('click', e => { e.stopPropagation();
    pinned = -1; seek(m.dataset.t); }));
  box.querySelectorAll('.tc-times [data-t]').forEach(el => el.addEventListener('click', () => {
    go(+el.closest('.tc-card').dataset.c, el.dataset.t); }));
  lane.querySelectorAll('.lane-step').forEach(bt => bt.addEventListener('click', () => {
    const t = vid ? vid.currentTime : 0, starts = C.map(c => c.from_start ? 0 : c.start_s);
    // previous: the start of the contact shown once the playhead is over a second into it, else the one before
    let k;
    if (bt.dataset.dir === '1') k = shown >= 0 ? shown + 1 : starts.findIndex(a => a > t + 0.05);
    else k = shown >= 0 && t - starts[shown] > 1 ? shown : (shown >= 0 ? shown - 1 : starts.reduce((j, a, i) =>
      (a < t - 0.05 ? i : j), -1));
    if (k >= 0 && k < n) go(k);
  }));
  // playing: one update per presented frame, so the card's heatmap keeps up with the footage
  let rv = 0, raf = 0;
  const alive = () => document.body.contains(box) && file === _activeFile;
  function onVF(now, md) { rv = 0; if (!alive()) return; sync(md.mediaTime); watch(); }
  function onRaf() { raf = 0; if (!alive() || vid.paused) return; sync(vid.currentTime); raf = requestAnimationFrame(onRaf); }
  function watch() {
    if (!vid || vid.paused) return;
    if (vid.requestVideoFrameCallback) { if (!rv) rv = vid.requestVideoFrameCallback(onVF); }
    else if (!raf) raf = requestAnimationFrame(onRaf);
  }
  if (vid) on(vid, 'play', watch);
  window._epCleanup.push(() => {
    if (rv && vid && vid.cancelVideoFrameCallback) vid.cancelVideoFrameCallback(rv);
    if (raf) cancelAnimationFrame(raf);
  });
  // the sensors file draws each contact's strength in its bar and the card's heatmaps
  loadSensors(file).then(got => {
    if (!got || !alive()) return;
    D = got;
    ink = snInk();
    layMaps();
    // one scale for the episode: its strongest contact's peak (label/contacts.py sums the same strengths at every frame)
    const top = Math.max(1e-9, ...C.map(c => isFinite(c.peak_strength) ? c.peak_strength : 0));
    segs.forEach(g => {
      const p = g.querySelector('path'), d = tcCurve(C[+g.dataset.c], D, top);
      if (!d) return;
      p.setAttribute('d', d);
      requestAnimationFrame(() => p.classList.add('in'));
    });
    shown = -2;
    sync(vid ? vid.currentTime : 0);
  });
  watch();
  return {sync};
}

// each camera with depth: its switch, and its depth clip laid over its colour clip and played in step with it
function setupDepth(file, eidEnc, on) {
  for (const btn of document.querySelectorAll('.cam-dp')) {
    const view = btn.dataset.view, cell = btn.closest('.cam-cell');
    const dv = cell && cell.querySelector('.dp-vid'), cv = cell && cell.querySelector('video:not(.dp-vid)');
    if (!dv || !cv) { btn.hidden = true; continue; }
    let want = false, hover = 0, hoverT = 0;
    const place = () => {
      Object.assign(dv.style, {left: cv.offsetLeft + 'px', top: cv.offsetTop + 'px', width: cv.offsetWidth + 'px',
        height: cv.offsetHeight + 'px'});
      // the player's own controls sit along the bottom of a video with controls: the strip of depth over them is dimmed
      dv.style.setProperty('--dp-bar', cv.controls ? '52px' : '0px');
    };
    const controls = () => dv.classList.toggle('ctl', !!cv.controls && (cv.paused || hover > 0));
    const follow = () => {
      if (!want || !dv.getAttribute('src')) return;
      if (dv.playbackRate !== cv.playbackRate) dv.playbackRate = cv.playbackRate;
      if (dv.readyState >= 1 && Math.abs(dv.currentTime - cv.currentTime) > 0.1) {
        try { dv.currentTime = cv.currentTime; } catch (_) {}
      }
      if (cv.paused && !dv.paused) dv.pause();
      else if (!cv.paused && dv.paused) dv.play().catch(() => {});
    };
    const show = () => { if (want) requestAnimationFrame(() => { if (want) dv.classList.add('on'); }); };
    const set = (on_) => {
      want = on_;
      btn.setAttribute('aria-pressed', String(on_));
      if (on_) {
        place(); controls();
        if (!dv.getAttribute('src')) {
          dv.preload = 'auto';
          dv.src = videoSrc(eidEnc, 'depth_' + view);
          dv.addEventListener('loadeddata', () => { follow(); show(); }, {once: true});
        } else { follow(); show(); }
      } else dv.classList.remove('on');      // the fade out; the clip pauses when it ends (transitionend below)
    };
    btn.addEventListener('click', () => {
      const v = !want;
      if (v) DP_ON.add(view); else DP_ON.delete(view);
      set(v);
    });
    on(dv, 'transitionend', e => { if (e.propertyName === 'opacity' && !dv.classList.contains('on')) dv.pause(); });
    on(dv, 'loadedmetadata', follow);
    // no depth clip to play after all: the switch goes, the colour picture stays
    on(dv, 'error', () => { want = false; dv.classList.remove('on'); btn.hidden = true; });
    for (const ev of ['play', 'pause', 'seeked', 'ratechange']) on(cv, ev, () => { follow(); controls(); });
    const pointer = () => { hover = 1; controls(); clearTimeout(hoverT);
      hoverT = setTimeout(() => { hover = 0; controls(); }, 2600); };
    on(cell, 'pointermove', pointer);
    on(cell, 'pointerleave', () => { clearTimeout(hoverT); hover = 0; controls(); });
    const ro = new ResizeObserver(place);
    ro.observe(cv); ro.observe(cell);
    const iv = setInterval(() => { if (want && !cv.paused) follow(); }, 300);
    window._epCleanup.push(() => { ro.disconnect(); clearInterval(iv); clearTimeout(hoverT); });
    if (DP_ON.has(view)) set(true);
  }
}

// The rail's "Labels by" block and the episode's header sit side by side from 1230 px up: the divider under each is one
// line across the page. Their tops are put on one line and both take the taller height, so a header that wraps (a long
// name, the footage line) never leaves the two dividers at different heights. Stacked, below 1230 px, they are left alone.
function alignHeads() {
  const lb = document.querySelector('.lb'), eh = document.querySelector('.ep-head');
  if (!lb || !eh) return;
  for (const el of [lb, eh]) { el.style.marginTop = ''; el.style.paddingTop = ''; el.style.paddingBottom = ''; }
  if (!(window.innerWidth >= 1230) || lb.hidden || !eh.offsetParent || !lb.getBoundingClientRect) return;
  // same top and same inner top padding, so "Labels by" sits on the line of the header's "Episode"
  lb.style.paddingTop = getComputedStyle(eh).paddingTop;
  lb.style.marginTop = (eh.getBoundingClientRect().top - lb.getBoundingClientRect().top) + 'px';
  // the shorter of the two takes the difference at its bottom, so both keep their contents on the top line
  const hl = lb.getBoundingClientRect().height, he = eh.getBoundingClientRect().height;
  const short = hl < he ? lb : eh, d = Math.abs(hl - he);
  if (d > 0.01) short.style.paddingBottom = (parseFloat(getComputedStyle(short).paddingBottom) + d) + 'px';
}

// The progress chart's points, [{t, p}] in time order from {t: 0, p: 0}. The chart is tied to the goal frame the board
// shows: it reaches 100% exactly there and nowhere else, so the chart, the goal frame and the outcome never disagree.
// A step's progress is the level the task has reached when the step ends, so each value is plotted at its step's end
// and the readout moves linearly between points (progressAt).
// - One goal (goal.completedAt, or goal.reachedAt for a goal reached and later undone): before the goal frame the
//   labels' progress is divided by its own level there, so it rises to 100% at the goal frame however the labels
//   counted the work, and it is held under 100% until then. A demonstration that keeps working past the goal (more
//   items, or a step the instruction does not ask for) holds 100% from the goal frame on. An undone goal holds 100%
//   until goal.undoneAt and then follows the labels down. With no goal frame the chart never reaches 100%; it holds
//   the highest level the labels reach below it.
// - A session of tasks (goal.tasks, each with its own goal frame, a head camera's activities): every task is an equal
//   share. A task's share is full from its own goal frame on; before that, or if it never reaches one, it carries the
//   task's own progress, held under full. So the session reaches 100% only when every task is done.
// An idle step (an arm parked while the other works, or waiting while someone else finishes the task) places a point
// only when it raises the level. Read at its end, a parked arm's step spanning the episode would draw a drop to its own
// progress, while a goal completed during a wait (a collaborator closing a lid) would otherwise never reach the chart.
// The last level holds to the end of the episode. tests/progress_points.js runs this on real cases, and Data Review's
// QA viewer (teleop-labeler/scripts/serve_qa.py) carries the same two functions, checked identical by its tests.
function progressPoints(eventLabels, duration, goal) {
  goal = goal || {};
  const UNDER = 0.99;   // the highest level shown before a goal frame, so it never reads 100%
  const stepEnd = e => (e.end_s != null && Number(e.end_s) >= e.t_s ? Number(e.end_s) : e.t_s);
  const clamp = v => Math.max(0, Math.min(1, Number(v)));
  // the labels' own levels in time order; an idle step counts only when it raises the level
  const levels = steps => {
    const out = [];
    let cur = 0;
    const ordered = steps.filter(x => x.t_s != null && x.progress != null)
      .map((x, i) => ({x, i})).sort((a, b) => stepEnd(a.x) - stepEnd(b.x) || a.i - b.i).map(o => o.x);
    for (const e of ordered) {
      const p = clamp(e.progress);
      if (e.contribution === 'idle' && p <= cur) continue;
      out.push({t: stepEnd(e), p});
      cur = p;
    }
    return out;
  };
  const lastAt = (pts, t) => { let p = 0; for (const q of pts) { if (q.t <= t + 1e-6) p = q.p; else break; } return p; };
  const finish = pts => {
    pts.sort((a, b) => a.t - b.t);
    pts.unshift({t: 0, p: 0});
    if (pts.length >= 2 && pts[pts.length - 1].t < duration) pts.push({t: duration, p: pts[pts.length - 1].p});
    return pts;
  };
  const tasks = (goal.tasks || []).filter(t => t);
  if (tasks.length) {
    const N = tasks.length;
    // each step belongs to the task whose span holds its start (the last task also takes anything after its end)
    const spans = tasks.map((t, i) => ({s: i === 0 || t.start_s == null ? -Infinity : Number(t.start_s),
      e: i + 1 < N && tasks[i + 1].start_s != null ? Number(tasks[i + 1].start_s) : Infinity,
      g: t.completed_at_s != null ? Number(t.completed_at_s) : null}));
    const own = spans.map(sp => levels(eventLabels.filter(e => e.t_s != null && e.t_s >= sp.s && e.t_s < sp.e)));
    const share = (i, t) => (spans[i].g != null && t >= spans[i].g - 1e-6) ? 1 : Math.min(UNDER, lastAt(own[i], t));
    const times = new Set();
    own.forEach(o => o.forEach(q => times.add(q.t)));
    spans.forEach(sp => { if (sp.g != null) times.add(sp.g); });
    const pts = [...times].filter(t => t > 0).sort((a, b) => a - b)
      .map(t => ({t, p: spans.reduce((a, _, i) => a + share(i, t), 0) / N}));
    return finish(pts);
  }
  const raw = levels(eventLabels);
  // the goal frame the board shows: completion's goal frame, or for a goal later undone the frame it was reached
  const g = goal.completedAt != null ? Number(goal.completedAt) : (goal.reachedAt != null ? Number(goal.reachedAt) : null);
  if (g == null) {
    // no goal frame: never 100%, the highest level the labels reach below it
    const below = raw.filter(q => q.p < 1).reduce((a, q) => Math.max(a, q.p), 0);
    const cap = below > 0 ? below : UNDER;
    return finish(raw.map(q => ({t: q.t, p: Math.min(q.p, cap)})));
  }
  const atGoal = progressAt(finish(raw.map(q => ({t: q.t, p: q.p}))), g);
  const scale = atGoal > 0 ? atGoal : Math.max(0, ...raw.map(q => q.p));
  const lift = v => (scale > 0 ? Math.min(UNDER, v / scale) : 0);
  const pts = raw.filter(q => q.t < g - 1e-6).map(q => ({t: q.t, p: lift(q.p)}));
  pts.push({t: g, p: 1});
  const u = goal.undoneAt != null ? Number(goal.undoneAt) : null;
  if (u != null && u > g) {
    // a goal undone: 100% until it is undone, then the labels' own level, held under 100%
    pts.push({t: u, p: 1});
    const after = raw.filter(q => q.t > u + 1e-6).map(q => ({t: q.t, p: lift(q.p)}));
    if (after.length) pts.push(...after);
    else pts.push({t: Math.min(duration, u + 0.5), p: lift(raw.length ? raw[raw.length - 1].p : 0)});
  }
  return finish(pts);
}

// The readout's whole percent: 100 only when the level is 100% (the goal frame), so a level of 99.6% never rounds up
// to a goal the board has not reached.
function progressPct(p) {
  return p >= 1 - 1e-9 ? 100 : Math.min(99, Math.round(p * 100));
}

// The level the progress readout shows at time t: read off the straight line joining the points either side of t.
function progressAt(pts, t) {
  let p = 0;
  for (let i = 0; i < pts.length; i++) {
    if (pts[i].t <= t + 0.05) { p = pts[i].p; continue; }
    const prev = pts[i - 1];
    if (prev && pts[i].t > prev.t) p = prev.p + (pts[i].p - prev.p) * Math.max(0, Math.min(1, (t - prev.t) / (pts[i].t - prev.t)));
    break;
  }
  return p;
}

function renderEp(d, opts) {
  opts = opts || {};
  const meta = d._meta || {};
  // the video's listeners belong to this render: a re-render (another source, the same footage) drops them all
  if (window._epAbort) window._epAbort.abort();
  window._epAbort = new AbortController();
  const on = (el, ev, fn) => el.addEventListener(ev, fn, {signal: window._epAbort.signal});
  // another model's labels of this episode (compare/metrics.py), shown in place of the board's; never counted
  const cmpInfo = d._compare || null;
  const who = cmpInfo ? cmpWho(cmpInfo.key) : modelName(meta.model);
  const failed = !!cmpInfo && cmpInfo.status !== 'parsed';
  // the board's own reply gave no labels (board/to_board.py label_failed): the footage, checks, sensors and the
  // dataset's own labels are shown, and one line stands in for every section the model would have answered
  const noLabels = !cmpInfo && !!d._label_failed;
  // switching source keeps the playing footage: the same video elements move into the new layout
  const keep = opts.keepVideo && document.getElementById('video') ? {
    video: document.getElementById('video'),
    side: [...document.querySelectorAll('.cam-cell.cam-wrist video:not(.dp-vid)')].map(el => [el.id, el])} : null;
  // a new episode: stop the old episode's videos now, so their downloads end instead of running on until the detached
  // elements are collected and competing with the new episode's footage (a fast visitor left several loading at once)
  if (!keep) {
    for (const v of leftCol.querySelectorAll('video')) {
      try { v.pause(); v.removeAttribute('src'); v.querySelectorAll('source').forEach(s => s.remove()); v.load(); } catch (_) {}
    }
  }
  // a head camera (the rig board/build.py copies from the episode's context) is a single panel with no mounted
  // cameras beside it
  const isEgo = d._rig === 'ego_head';
  ARM_NOUN = isEgo ? 'hands' : (/handheld/.test(d._rig || '') ? 'grippers' : 'arms');
  const comp = d.completion || {};
  const inv = d.objects || [];
  const eventLabels = d.event_labels || [];

  // the episode's name at the top of its pane, and its raw id beneath when the two differ
  const eid = meta.episode_id || '';
  currentEp.textContent = epName(eid);
  document.getElementById('current-ep-raw').textContent = epName(eid) !== eid ? eid : '';
  document.getElementById('current-ep-src').innerHTML = datasetSourceHtml(d.dataset_source);
  document.getElementById('current-ep-reader').innerHTML = readerNotesHtml(d.reader_notes);
  document.getElementById('dl-json').href = episodeDownloadUrl(_activeFile);
  if (STATIC) document.getElementById('dl-json').setAttribute('download', _activeFile);
  // the episode's video to download, on a served board (a static build has no server to make it)
  _vdEp = {eid, d};
  vdEl.hidden = STATIC || !BOARD.footage || !eid;
  vdOpen(false);
  // the hand keypoints of a head-camera episode, a download of their own beside the labels'
  const kp = KP_INDEX && KP_INDEX[_activeFile], kpA = document.getElementById('kp-dl');
  kpA.hidden = !kp;
  document.getElementById('kp-note').classList.toggle('off', !kp);
  if (kp) {
    kpA.href = keypointsDownloadUrl(_activeFile);
    kpA.setAttribute('download', _activeFile.replace(/\.json$/, '') + '.hand_keypoints.json');
    kpA.title = `The model's 2D keypoints of both hands on all ${Number(kp.frames).toLocaleString()} frames of this `
      + `episode's video, in the dataset's own pixels and frame times (JSON, ${fmtBytes(kp.bytes)}). Non-commercial `
      + `use only.`;
  }

  // what the label call cost and took, as the harness recorded it (the provider's billed cost when it reports one),
  // under the prompt
  const usage = d._usage || {};
  let promptMetaHtml = '';
  if (usage.est_cost_usd != null) {
    const bits = [`<span class="pm pm-green"><span class="pm-k">cost</span><b>$${Number(usage.est_cost_usd).toFixed(2)}`
      + `</b></span>`];
    if (usage.latency_s != null) bits.push(`<span class="pm pm-green"><span class="pm-k">generated in</span>`
      + `<b>${fmtDur(usage.latency_s)}</b></span>`);
    const tk = [];
    if (usage.prompt_tokens != null) tk.push(fmtTok(usage.prompt_tokens) + ' in');
    if (usage.completion_tokens != null) tk.push(fmtTok(usage.completion_tokens) + ' out');
    if (tk.length) bits.push(`<span class="pm"><span class="pm-k">tokens</span><b>${tk.join(' / ')}</b></span>`);
    promptMetaHtml = `<div class="prompt-meta">${bits.join('')}</div>`;
  }

  // Outcome counts (for the dense-timeline count badge)
  const contribs = {advancing: 0, wasteful: 0, idle: 0};
  for (const e of eventLabels) {
    const c = (e.contribution || '').toLowerCase();
    if (c in contribs) contribs[c]++;
  }

  // the timeline's length: the last dense segment and every goal, task, key-event and recovery time, so no marker
  // lands past the right edge when a task's completion is timed apart from the motion segments
  let duration = 0;
  for (const e of eventLabels) {
    if (e.t_s != null) duration = Math.max(duration, e.t_s);
    if (e.end_s != null) duration = Math.max(duration, e.end_s);
  }
  for (const t of (d.tasks || [])) {
    if (t.completed_at_s != null) duration = Math.max(duration, t.completed_at_s);
    if (t.end_s != null) duration = Math.max(duration, t.end_s);
  }
  for (const k of (d.key_events || [])) { if (k.t_s != null) duration = Math.max(duration, k.t_s); }
  for (const r of (d.recovery || [])) {
    if (r.recovered_at_s != null) duration = Math.max(duration, r.recovered_at_s);
    if (r.failure_t_s != null) duration = Math.max(duration, r.failure_t_s);
  }
  if ((d.completion || {}).completed_at_s != null) duration = Math.max(duration, d.completion.completed_at_s);
  if ((d.completion || {}).goal_reached_at_s != null) duration = Math.max(duration, d.completion.goal_reached_at_s);
  // the recording's contacts and the grasps the model saw with none stay on the timeline too
  const touch = tcData(d);
  for (const c of touch.contacts) duration = Math.max(duration, c.end_s);
  for (const x of touch.missing) duration = Math.max(duration, x.t_s);
  // the recording's own length when the episode has one, so the timeline, its lanes and the video end together;
  // without it, the last labelled time with a second's margin
  duration = d.duration_s > 0 ? Math.max(duration, d.duration_s) : Math.max(duration + 1, 10);

  // a head-camera session is a sequence of self-directed tasks (d.tasks), each with its own goal frame; every other
  // episode has one completion and one goal
  const tasks = (d.tasks || []).filter(t => t && (t.task || t.start_s != null));
  const hasTasks = tasks.length > 0;
  const taskGoalTimes = tasks.map(t => t.completed_at_s).filter(t => t != null);

  // Given or inferred. When the dataset ships an instruction, the episode was graded against it (given mode): that
  // goal is the anchor, with the model's independent assessment beside it, since a divergence between the two is
  // itself a data-quality signal. Without one, the model's assessment is the one line.
  const givenMode = meta.prompt_mode === 'given' && !!meta.given_prompt;
  const bannerLabel = hasTasks ? 'Session summary'
    : givenMode ? `Given goal and ${whoOwn(who, 'independent assessment')}`
    : `Task, as ${esc(nameParts(who)[0])}${nameParts(who)[1] ? ` (${esc(nameParts(who)[1])})` : ''} assessed it`;
  // alignment chip: whether the model's independent assessment matches the given goal
  const ga = d.goal_alignment || null;
  let alignHtml = '';
  if (givenMode && ga && ga.relation) {
    const rel = String(ga.relation).toLowerCase();
    // green when the footage shows the given goal done (alone, or with more besides), red when it shows only part of
    // it or something else
    const matches = ga.matches_given !== false && (rel === 'aligned' || rel === 'narrower');
    const cls = matches ? 'match' : 'nomatch';
    const label = rel === 'narrower' && matches ? 'matches given goal and does more'
      : matches ? 'matches given goal'
      : rel === 'broader' && ga.matches_given !== false ? 'only part of the given goal'
      : 'does not match given goal';
    alignHtml = `<div class="align-chip ${cls}"><span>${esc(label)}</span>`
      + `${ga.note ? `<span class="ac-note">${esc(ga.note)}</span>` : ''}</div>`;
  }
  const bannerBody = givenMode
    ? `<div class="goal-given"><span class="gg-badge">given goal</span>`
        + `<span class="gg-text">${esc(meta.given_prompt)}</span></div>`
      + `<div class="goal-read"><span class="gr-badge">${whoOwn(who, 'independent assessment')}</span>`
        + `<span class="gr-text">${esc(d.episode_prompt || '(empty)')}</span></div>`
      + alignHtml
    : `<div class="text">${esc(d.episode_prompt || '(empty)')}</div>`;

  // Key events (milestone layer), sorted.
  const keyEvents = (d.key_events || []).filter(k => k.t_s != null)
      .slice().sort((a, b) => a.t_s - b.t_s);

  const progPts = progressPoints(eventLabels, duration, hasTasks ? {tasks}
    : {completedAt: comp.completed_at_s, reachedAt: comp.goal_reached_at_s, undoneAt: comp.undone_at_s});
  const progSubLabel = hasTasks ? 'tasks done' : 'to goal';
  let progOverlayHtml = '';
  if (progPts.length >= 2) {
    const W = 100, H = 24;
    const full = progPts.map(pt => `${(pt.t / duration * W).toFixed(2)},${((1 - pt.p) * H).toFixed(2)}`).join(' ');
    progOverlayHtml = `<div class="prog-overlay" id="prog-overlay">
      <span class="po-pct" id="po-pct">0%</span>
      <svg class="po-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
        <polyline class="po-full" points="${full}"></polyline>
        <polyline class="po-past" id="po-past" points=""></polyline>
        <circle class="po-dot" id="po-dot" r="2.4" cx="0" cy="${H}"></circle>
      </svg>
      <span class="po-sub">${progSubLabel}</span>
    </div>`;
  }

  // lanes under the timeline, on its time scale: when the wearer's hands leave a head camera (the model's
  // per-step hands_visible), and the dataset's own timed labels as the publisher ships them
  const lanePct = (t) => (100 * Math.max(0, Math.min(duration, t)) / Math.max(duration, 1e-6));
  const handSpans = isEgo ? eventLabels.filter(e => e.hands_visible === false && e.t_s != null)
    .map(e => [Number(e.t_s), Number(e.end_s != null ? e.end_s : e.t_s + 1)]) : [];
  const pubLabels = (d.dataset_labels || []).filter(x => x && x.label != null);
  const lane = (id, title, now, segs) => `<div class="lane" id="${id}"><div class="lane-title">${title}${now
    ? ` <span class="lane-now" id="${id}-now"></span>` : ''}</div>`
    + `<div class="lane-bar">${segs}<div class="lane-ph"></div></div></div>`;
  let laneHtml = '';
  // the stretches with the hands out of view: the model's spans merged where they are under 1.5 s apart, so one
  // stretch on the bar is one step of the stepper
  const handStretches = handSpans.slice().sort((x, y) => x[0] - y[0]).reduce((acc, [a, b]) => {
    const last = acc[acc.length - 1];
    if (last && a - last[1] < 1.5) last[1] = Math.max(last[1], b); else acc.push([a, b]);
    return acc; }, []);
  // the share counts the model's own spans (overlaps once), as the dataset's overview does, not the merged gaps
  const handSecs = (() => { let tot = 0, end = -1; for (const [a, b] of handSpans.slice().sort((x, y) => x[0] - y[0])) {
    if (b <= end) continue; tot += b - Math.max(a, end); end = b; } return tot; })();
  const handShare = handSecs / Math.max(duration, 1e-6);
  const nHand = handStretches.length;
  const chev = (d) => `<svg viewBox="0 0 10 10" aria-hidden="true"><path d="${d}" fill="none" stroke="currentColor" `
    + `stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
  if (nHand) laneHtml += `<div class="lane lane-hands" id="lane-hands">
      <div class="lane-head">
        <span class="lane-title">Hands out of view <span class="lane-sum">${(100 * handShare).toFixed(handShare < 0.1
          ? 1 : 0)}% of the clip</span></span>
        <span class="lane-nav" role="group" aria-label="Stretches with the hands out of view">
          <button type="button" class="lane-step" data-dir="-1" aria-label="Previous stretch">${chev('M6.5 2 3.5 5l3 3')}</button>
          <span class="lane-pos" id="lane-hands-pos">${nHand} ${nHand === 1 ? 'stretch' : 'stretches'}</span>
          <button type="button" class="lane-step" data-dir="1" aria-label="Next stretch">${chev('M3.5 2l3 3-3 3')}</button>
        </span>
      </div>
      <div class="lane-bar">${handStretches.map(([a, b], i) => `<div class="lane-seg hands" data-t="${a}" data-i="${i}" `
        + `title="${fmtT(a)} to ${fmtT(b)}" style="left:${lanePct(a)}%;width:max(3px, ${lanePct(b) - lanePct(a)}%)">`
        + `</div>`).join('')}<div class="lane-ph"></div></div>
    </div>`;
  // the recording's contacts, one bar per hand, and the card of the contact under the playhead below the lanes
  laneHtml += touchLaneHtml(touch, lanePct, chev);
  // a label with no time is listed below, never drawn on the lane
  if (pubLabels.some(x => x.t0 != null)) laneHtml += lane('lane-pub', "Dataset's labels", true, pubLabels.map((x, i) =>
    x.t0 == null ? '' : `<div class="lane-seg pub${i % 2 ? ' alt' : ''}" data-t="${x.t0}" data-i="${i}" `
      + `title="${esc(fmtT(x.t0) + ' to ' + fmtT(x.t1) + ': ' + x.label)}" `
      + `style="left:${lanePct(x.t0)}%;width:max(2px, `
      + `calc(${lanePct(x.t1) - lanePct(x.t0)}% - 1px))"></div>`).join(''));
  const pubEp = d.dataset_episode_labels || null;
  const spanText = (xs) => (xs && xs.length) ? xs.map(sp => Array.isArray(sp) ? `${fmtT(sp[0])} to ${fmtT(sp[1])}`
    : esc(String(sp))).join(', ') : 'none';
  let pubHtml = '';
  if (pubLabels.length || pubEp) {
    pubHtml = `<h3 class="section">The dataset's own labels${pubLabels.length
      ? ` <span class="count">${pubLabels.length}</span>` : ''}</h3><div class="info-block pub-list">`
      + (d.dataset_labels_note ? `<div class="pub-note">${esc(d.dataset_labels_note)}</div>` : '')
      + pubLabels.map((x, i) => `<div class="pub-row"${x.t0 != null ? ` data-t="${x.t0}"` : ''} data-i="${i}">`
        + `<span class="pub-t">${x.t0 == null ? 'no time' : x.t1 == null || x.t1 === x.t0 ? fmtT(x.t0)
          : `${fmtT(x.t0)} to ${fmtT(x.t1)}`}</span><span class="pub-l">${esc(x.label)}</span></div>`).join('')
      + (pubEp ? `<div class="pub-ep">`
        + (pubEp.task_status ? `<div><span class="pub-k">Task status</span> ${esc(String(pubEp.task_status))}</div>`
          : '')
        + ('error_spans_s' in pubEp ? `<div><span class="pub-k">Error spans</span> ${spanText(pubEp.error_spans_s)}`
          + `</div>` : '')
        + ('intervention_spans_s' in pubEp ? `<div><span class="pub-k">Intervention spans`
          + `</span> ${spanText(pubEp.intervention_spans_s)}</div>` : '')
        + ('high_jerk_spans_s' in pubEp ? `<div><span class="pub-k">High jerk spans`
          + `</span> ${spanText(pubEp.high_jerk_spans_s)}</div>` : '')
        + `</div>` : '')
      + `</div>`;
  }
  // the notes an upload sent with the episode, as sent: its table rows, then the notes read from its files, folded
  const upl = d.uploader_notes || [];
  if (upl.length) {
    const kv = (it) => `<div class="pub-kv"><span class="pub-k">${esc(it.name)}</span><span class="pub-v">`
      + `${esc(it.value)}${it.t != null ? `<button type="button" class="pub-at" data-t="${it.t}">at `
      + `${esc(fmtT(it.t))}</button>` : ''}</span></div>`;
    const grp = (g) => `<div class="pub-group"><div class="pub-gt">${esc(g.title)}</div>${g.items.map(kv).join('')}</div>`;
    const notes = upl.filter(g => g.kind !== 'row');
    const uplHtml = upl.filter(g => g.kind === 'row').map(grp).join('') + (notes.length
      ? `<div class="pub-fold"><div class="sn-fold"><div class="sn-fold-in">${notes.map(grp).join('')}</div></div>`
        + `<button class="ck-more pub-show" type="button" aria-expanded="false" `
        + `data-closed="Show the notes in the files" data-open="Hide the notes in the files">`
        + `Show the notes in the files</button></div>`
      : '');
    pubHtml = pubHtml ? pubHtml.slice(0, -'</div>'.length) + uplHtml + '</div>'
      : `<h3 class="section">The dataset's own labels</h3><div class="info-block pub-list">${uplHtml}</div>`;
  }
  let markersHtml = '';
  for (const e of eventLabels) {
    if (e.t_s == null) continue;
    const pct = (e.t_s / duration) * 100;
    const cls = contribClass(e.contribution);
    const tip = `${fmtT(e.t_s)}   ${esc(e.verb_class || '')}`;
    markersHtml += `<div class="marker seg ${cls}" style="left:${pct}%" data-t="${e.t_s}"><div class="tip">${esc(tip)}`
      + `</div></div>`;
  }
  let ticksHtml = '';
  // at most about eight labels whatever the length, so a long recording's labels never run into each other
  const labelStep = [1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1200, 1800].find(st => duration / st <= 8) || 3600;
  const tickLabel = t => t >= 60 && labelStep >= 60 ? `${Math.floor(t / 60)}:${String(t % 60).padStart(2, '0')}`
    : `${t}s`;
  for (let t = 0; t <= duration; t += labelStep / 2) {
    const pct = (t / duration) * 100;
    ticksHtml += `<div class="tick" style="left:${pct}%"></div>`;
    // the first label starts at the timeline's edge instead of centring on it, so it never sits outside the timeline
    if (t % labelStep === 0) ticksHtml += `<div class="tick-label" style="left:${pct}%${t === 0 ? ';transform:none' : ''}">`
      + `${tickLabel(t)}</div>`;
  }

  const eidEnc = encodeURIComponent(meta.episode_id || '');
  // cameras this episode actually has (FastUMI: grippers only, no top camera). The main player shows
  // the top camera when there is one, else the first gripper camera; side cells show the rest.
  const {views: camViews, main: mainCam} = episodeCams(d);
  const camNameOf = v => {
    const i = camViews.indexOf(v);
    return (d.camera_labels && i >= 0 && d.camera_labels[i]) || v;
  };
  const hasTop = mainCam === 'exo';
  const sideCams = camViews.filter(v => v !== mainCam && v !== 'exo');
  const camLabel = v => v === 'exo' ? 'exo' : (v !== 'left' && v !== 'right') ? camNameOf(v)
    : (hasTop ? `${v} wrist` : `${camNameOf(v)}${camNameOf(v) === 'gripper' ? '' : ' gripper'}`);
  // the side cells' video ids: the two mounted cameras keep theirs, any other camera is video-<its view>
  const sideId = v => v === 'left' ? 'video-wl' : v === 'right' ? 'video-wr' : `video-${v}`;
  const videoUrl = videoSrc(eidEnc, mainCam);
  const dpViews = depthViews(_activeFile);     // the cameras with a depth clip, each with its switch
  // under the lanes: the contact card, and the slot the sensors panel fills, only on an episode that has them, so every
  // other episode's page is what it was
  const belowLanes = [tcCardsHtml(touch), BOARD.sensors && SN_INDEX && SN_INDEX[_activeFile] ? '<div id="sn-slot"></div>'
    : ''].filter(Boolean).map(x => '\n    ' + x).join('');
  const videoUrlWL = videoSrc(eidEnc, 'left');
  const videoUrlWR = videoSrc(eidEnc, 'right');

  // LEFT COLUMN, in reading order: the footage and its timeline; what happened (key events) and whether the task was
  // done; how well it was done; what is wrong with the episode (its problems) and every check run on it; the
  // dataset's own labels; then the scene (state changes, object relationships, inventory)
  let invHtml = '';
  for (const o of inv) {
    invHtml += `<div class="it"><span class="name">${esc(o.name || '?')}</span>`
      + `${o.color ? `<span class="color">${esc(o.color)}</span>` : ''}</div>`;
  }

  // a key event's mark by kind: the goal and a subgoal green, a failure red, the rest ink
  const keyColor = k => (k.kind === 'goal_reached' || k.kind === 'subgoal_complete') ? 'var(--success)'
    : ((k.outcome || '').toLowerCase() === 'failure' ? 'var(--danger)' : 'var(--fg-2)');
  let keyMarkersHtml = '';
  for (const k of keyEvents) {
    const pct = (k.t_s / duration) * 100;
    const oc = (k.outcome || '').toLowerCase();
    keyMarkersHtml += `<div class="marker key" style="left:${pct}%;background:${keyColor(k)}" data-t="${k.t_s}"><div `
      + `class="tip">${esc(fmtT(k.t_s))}   ${k.kind ? esc(kindName(k.kind)) + ': ' : ''}${esc(k.label || '')}${oc
      ? '  [' + esc(oc) + ']' : ''}</div></div>`;
  }
  let keyPanelHtml = '';
  const untimed = untimedRows(d, keyEvents.length + 1);
  keyEvents.forEach((k, i) => {
    const oc = (k.outcome || '').toLowerCase();
    const goalish = k.kind === 'goal_reached' || k.kind === 'subgoal_complete';
    keyPanelHtml += `<div class="key-ev ${oc}${goalish ? ' goal' : ''}" data-t="${k.t_s}">
      <span class="ke-num">${i + 1}</span>
      <span class="ke-time">${fmtT(k.t_s)}</span>
      <div class="ke-body">
        <div class="ke-row1"><span class="ke-label">${esc(k.label || '')}</span>${oc
          ? `<span class="outcome ${oc}">${esc(oc)}</span>` : ''}</div>
        ${k.note ? `<div class="ke-note">${esc(k.note)}</div>` : ''}
      </div>
    </div>`;
  });
  keyPanelHtml += untimed.keys.join('');

  // the review of how the task was performed, and the state changes; the scene graph is a snapshot that follows the
  // playhead (renderSceneGraph)
  const perfHtml = d.performance_review
    ? `<div class="info-block"><div class="perf">${esc(d.performance_review)}</div></div>` : '';
  const scHtml = (d.state_changes || []).map(x =>
    `<div class="line"><span class="t">${fmtT(x.t_s)}</span> <span class="v">${esc(x.object || '')}</span>: ${esc(x.from
      || '')} <span style="color:var(--fg-3)">-&gt;</span> ${esc(x.to || '')}</div>`).join('');
  // recoveries: each failed subevent paired with its recovery; a reply that leaves out "recovered" but gives a
  // recovery time is read as recovered
  const recoveries = (d.recovery || []).map(r => ({
    failureT: r.failure_t_s,
    failure: r.failure || '',
    recovered: (r.recovered != null) ? r.recovered : (r.recovered_at_s != null ? true : null),
    recoveredAt: (r.recovered_at_s != null) ? r.recovered_at_s : null,
    correction: r.correction || '',
    howToFix: r.how_to_fix || '',
  })).filter(r => r.failureT != null).sort((a, b) => a.failureT - b.failureT);

  let rcHtml = '';
  for (const r of recoveries) {
    const recTag = r.recovered === true
      ? `<span class="rec-ok"${r.recoveredAt != null ? ` data-t="${r.recoveredAt}"`
        : ''}>recovered${r.recoveredAt != null ? ' at ' + fmtT(r.recoveredAt) : ''}</span>`
      : (r.recovered === false ? `<span class="rec-no">failed recovery</span>` : '');
    // Recovered -> show what it actually DID; only an unrecovered failure gets
    // the ideal "how to fix" prescription. Never both.
    const recDetail = r.recovered === true
      ? (r.correction ? `<div class="rec-line"><span class="rec-k">did</span> ${esc(r.correction)}</div>` : '')
      : (r.howToFix ? `<div class="rec-line"><span class="rec-k">how to fix</span> ${esc(r.howToFix)}</div>` : '');
    rcHtml += `<div class="rec">
      <div class="rec-head"><span class="rec-t" data-t="${r.failureT}">${fmtT(r.failureT)}</span><span `
        + `class="rec-fail">${esc(r.failure)}</span>${recTag}</div>
      ${recDetail}
    </div>`;
  }
  const ivHtml = (d.instruction_variants || []).map(v => `<li>${esc(v)}</li>`).join('');
  const verdict = (comp.task_completed || '').toLowerCase();
  // the verdict's colour: green when the task ends done, red when it does not (undone included), ink otherwise
  const vColor = verdict === 'success' ? 'var(--success)'
    : (verdict === 'failure' || verdict === 'success_then_undone') ? 'var(--danger)' : 'var(--fg-2)';
  const undone = verdict === 'success_then_undone';

  // a session of tasks: a Tasks panel, each task with its own outcome and goal frame, in place of the single
  // completion
  const tcount = {success: 0, failure: 0};
  tasks.forEach(t => { const o = (t.outcome || '').toLowerCase(); if (o in tcount) tcount[o]++; });
  const tPartly = tasks.filter(t => t.failure_kind === 'partial').length;
  let tasksHtml = '';
  tasks.forEach((t, i) => {
    const oc = (t.outcome || '').toLowerCase();
    const span = (t.start_s != null && t.end_s != null) ? `${fmtT(t.start_s)}-${fmtT(t.end_s)}` : '';
    const done = t.completed_at_s != null ? ` &middot; done ${fmtT(t.completed_at_s)}` : '';
    const jump = t.completed_at_s != null ? t.completed_at_s : t.start_s;
    tasksHtml += `<div class="task-row ${oc}"${jump != null ? ` data-t="${jump}"` : ''}>
      <span class="task-num ${oc}">${i + 1}</span>
      <div class="task-body">
        <div class="task-head"><span class="task-name">${esc(t.task || '?')}</span>${oc
          ? `<span class="outcome ${oc}">${esc(outcomeWords(oc, t.failure_kind))}</span>` : ''}</div>
        <div class="task-meta">${span}${done}</div>
        ${t.success_predicate ? `<div class="task-pred">${esc(t.success_predicate)}</div>` : ''}
        ${t.note ? `<div class="task-note">${esc(t.note)}</div>` : ''}
      </div>
    </div>`;
  });
  // the problems: deterministic checks that fired, data issues and operator mistakes
  const dataIssues = (d.data_issues || []).filter(x => x && x.issue);
  const tb = (d.dataset_checks || {}).timebase || null;
  const tbHtml = (tb && tb.sped_up_recording) ? `
    <div class="info-block di-block"><div class="di-row high">
      <span class="di-sev">check</span>
      <div class="di-body">
        <div class="di-issue">The video and robot state play back faster than real time.</div>
        <div class="di-tags"><span class="di-cat">${esc(famName('sped-up'))}</span></div>
        <div class="di-ev">The recorder skipped ${((tb.skipped_frac || 0) * 100).toFixed(1)}% and `
          + `repeated ${((tb.repeated_frac || 0) * 100).toFixed(1)}% of samples and the follower arm trails the `
          + `operator by only ${tb.follower_lag_frames} frames, so the recording loop ran below the 30 Hz its `
          + `timestamps claim. Rule: ${tb.rule || ''}.</div>
      </div></div></div>` : '';
  // the entries that count first, then the minor ones, marked; excluded entries (_excluded) are listed apart, in the
  // fold of what this dataset's rules set aside (setAsideHtml)
  const byCount = list => [...list.filter(countsIssue), ...list.filter(x => !countsIssue(x))];
  // an issue a person checked on the episode's frames, or a deterministic check confirmed, says so; one the model
  // gave in its free-text notes rather than its list of issues says that
  const verifiedChip = x => x.verified !== 'confirmed' ? '' : `<span class="di-verified" `
    + `title="${esc(x.verified_note || '')}">${x.verified_by === 'check' ? 'confirmed by a deterministic check'
      : 'checked on the frames'}</span>`;
  const issueRows = (list, key) => byCount(list).map(x => {
      const t = (x.t_s != null && !isNaN(parseFloat(x.t_s))) ? x.t_s : null;
      const minor = !countsIssue(x);
      return `
      <div class="di-row ${(x.severity || '').toLowerCase() === 'high' ? 'high' : 'low'}${minor ? ' minor'
        : ''}"${t != null ? ` data-t="${t}"` : ''}>
        <span class="di-sev">${esc((x.severity || 'flag'))}</span>
        <div class="di-body">
          <div class="di-issue">${esc(x.issue)}</div>
          <div class="di-tags">${minor ? '<span class="di-minor" title="low severity, left out of the totals">minor</span>'
            : ''}${x.family || x.category ? `<span class="di-cat" title="${esc(x.category || '')}">${esc(x.family
            ? famName(x.family) : tagName(x.category, key))}</span>` : ''}${t != null
            ? `<span class="di-t">@ ${esc(fmtT(t))}</span>` : ''}${verifiedChip(x)}${x.derived_from
            ? `<span class="di-cat" title="${esc(x.evidence || '')}">from the model's notes</span>` : ''}</div>
          ${x.evidence ? `<div class="di-ev">${esc(x.evidence)}</div>` : ''}
        </div>
      </div>`; }).join('');
  const sp = (d.dataset_checks || {}).stream_pairing || null;
  const spHtml = (sp && sp.crossed) ? `
    <div class="info-block di-block"><div class="di-row high">
      <span class="di-sev">check</span>
      <div class="di-body">
        <div class="di-issue">Each gripper camera moves with the other side's recorded motion.</div>
        <div class="di-tags"><span class="di-cat">${esc(famName('streams-crossed'))}</span></div>
        <div class="di-ev">The left stream's image change follows the right side's recorded speed (r `
          + `= ${sp.left_vs_right}) better than its own (${sp.left_vs_left}), and the right stream follows the left `
          + `side (${sp.right_vs_left}) better than its own (${sp.right_vs_right}). Either the stream names or the `
          + `state channels are swapped; the pixels tell which (where the other gripper appears in each view).</div>
      </div></div></div>` : '';
  const rj = (d.dataset_checks || {}).recorded_jumps || null;
  const rjEv = rj && rj.flagged ? (rj.events || []).filter(e => e.visual_jump === false) : [];
  const rjHtml = rjEv.length ? `
    <div class="info-block di-block">${rjEv.map(e => `<div class="di-row high" data-t="${e.t_s}">
      <span class="di-sev">check</span>
      <div class="di-body">
        <div class="di-issue">The ${esc(e.actor)} ${esc(e.unit === 'cm' ? 'gripper' : 'arm')}'s recorded motion leaps `
          + `${e.step} ${esc(e.unit)} in one frame, and its camera does not jump.</div>
        <div class="di-tags"><span class="di-cat">${esc(famName('recorded-jump'))}</span><span `
          + `class="di-t">@ ${esc(fmtT(e.t_s))}</span></div>
        <div class="di-ev">Its typical step is ${e.typical_p95} ${esc(e.unit)} (95th percentile). The ${esc(e.camera
          || '')} camera, which is mounted on it, changes ${e.image_change_at} at that frame against a median `
          + `of ${e.image_change_window_median} over the surrounding frames, so the camera did not move with it. `
          + `Rule: ${esc(rj.rule || '')}.</div>
      </div></div>`).join('')}</div>` : '';
  const gc = (d.dataset_checks || {}).gripper_channels || null;
  const gcFlat = gc && gc.flagged ? Object.entries(gc.actors || {}).filter(([, a]) => a.flat) : [];
  const gcHtml = gcFlat.length ? `
    <div class="info-block di-block">${gcFlat.map(([name, a]) => `<div class="di-row high">
      <span class="di-sev">check</span>
      <div class="di-body">
        <div class="di-issue">The ${esc(name)} gripper's recorded value never changes. It is exactly ${a.min} at `
          + `every frame.</div>
        <div class="di-tags"><span class="di-cat">${esc(famName('gripper-flat'))}</span></div>
        <div class="di-ev">Either this gripper was not used in the episode, or its sensor did not record. `
          + `The ${esc(name)} camera shows which. If the fingers open and close, the recorded gripper action `
          + `is missing. Rule: ${esc(gc.rule || '')}.</div>
      </div></div>`).join('')}</div>` : '';
  // capture checks (checks/capture_qc.py): flags are defects that held up on our verified datasets; notes
  // are quiet facts about the recording, never counted as problems
  const cqc = (d.dataset_checks || {}).capture_qc || null;
  const cqFlags = cqc ? (cqc.flags || []).filter(f => f && f.title) : [];
  const cqHtml = cqFlags.length ? `
    <div class="info-block di-block">${cqFlags.map(f => `<div class="di-row high"${f.t_s != null ? ` data-t="${f.t_s}"`
      : ''}>
      <span class="di-sev">check</span>
      <div class="di-body">
        <div class="di-issue">${esc(f.title)}</div>
        <div class="di-tags"><span class="di-cat">${esc(((cqc.checks || []).find(c => c.check === f.check) || {}).name
          || tagName(f.check || '', 'data_issues'))}</span>${f.t_s != null ? `<span class="di-t">@ ${esc(fmtT(f.t_s))}`
          + `</span>` : ''}${f.camera ? `<span class="di-cat">${esc(f.camera)}</span>` : ''}</div>
        ${f.evidence ? `<div class="di-ev">${esc(f.evidence)}</div>` : ''}
      </div></div>`).join('')}</div>` : '';
  const cqNotes = cqc ? (cqc.notes || []).filter(n => n && n.text) : [];
  const cqNotesHtml = cqNotes.length ? `<div class="cq-notes"><div class="cq-notes-k">Recording notes`
    + `</div>${cqNotes.map(n => `<div class="cq-note">${esc(n.text)}</div>`).join('')}</div>` : '';
  // the recording is faithful here; the demonstration itself went wrong in a way a model could copy
  const opMistakes = (d.operator_mistakes || []).filter(x => x && x.issue);
  // three kinds of problem, each in its own card with its own colour, so where one list ends and the
  // next begins is never in doubt
  const panel = (kind, title, sub, n, body, minor) => `
    <section class="ip ip-${kind}">
      <div class="ip-head"><h3>${title}</h3><span class="ip-counts">${minor ? `<span class="ip-minor">+ ${minor} minor`
        + `</span>` : ''}<span class="ip-n">${n}</span></span></div>
      <p class="ip-sub">${sub}</p>
      <div class="ip-body">${body}</div>
    </section>`;
  const riRows = readerIssueRows(d);
  const riHtml = riRows.length ? `<div class="info-block di-block">${riRows.join('')}</div>` : '';
  const nChecks = [tbHtml, spHtml, rjHtml, gcHtml].filter(Boolean).length + cqFlags.length + riRows.length;
  const checksPanel = nChecks ? panel('checks', 'Recording checks', 'Deterministic checks on the recorded data, run on '
    + 'every episode.', nChecks, `${riHtml}${tbHtml}${spHtml}${rjHtml}${gcHtml}${cqHtml}`) : '';
  const nData = dataIssues.filter(countsIssue).length;
  const nOp = opMistakes.filter(countsIssue).length;
  const diHtml = dataIssues.length ? panel('data', 'Data issues', 'Faults in the recording, the scene or the label, '
    + 'reported by the model. Low severity is shown as minor and left out of the totals.', nData,
    `<div class="info-block di-block">${issueRows(dataIssues, 'data_issues')}</div>`, dataIssues.length - nData) : '';
  const opHtml = opMistakes.length ? panel('mistake', 'Operator mistakes', 'The recording is faithful; the '
    + 'demonstration went wrong in a way a model could copy. Retries and wasted effort below high severity are shown '
    + 'as minor and left out of the totals.', nOp, `<div class="info-block di-block op">${issueRows(opMistakes,
    'operator_mistakes')}</div>`, opMistakes.length - nOp) : '';
  const problemsHtml = (checksPanel || diHtml || opHtml)
    ? `<h3 class="section">Problems in this episode</h3><div class="ip-stack">${checksPanel}${diHtml}${opHtml}</div>`
    : `<h3 class="section">Problems in this episode</h3><div class="ip-none">${cmpInfo ? `${esc(who)} reported no data `
      + `issue or operator mistake.`
        : 'No recording check fired, and the model reported no data issue or operator mistake.'}</div>`;
  const problemsAndNotes = problemsHtml + setAsideHtml(d) + (checksSection(d) || cqNotesHtml);

  // a session has a goal frame per task: one panel follows the playhead and shows the goal frame of the task nearest
  // the current time
  const taskGoalFrameHtml = (hasTasks && taskGoalTimes.length) ? `
    <div class="goal-frame" id="task-goal-frame" data-t="">
      <div class="goal-frame-cap" id="task-goal-cap"></div>
      <img id="task-goal-img" alt="task goal frame" loading="lazy">
    </div>` : '';
  const tasksSection = hasTasks ? `
    <h3 class="section">What tasks did the operator do?</h3>
    <div class="tasks-summary"><b>${tasks.length}</b> self-directed tasks &nbsp;
      <span class="ts-ok">${tcount.success} success</span> /
      <span class="ts-fail">${tcount.failure} failure${tPartly ? `, ${tPartly} of them partly done` : ''}</span>
      <div class="tasks-note">The episode has no single goal. Each task is graded on its own, and a messy final scene is fine.</div>
    </div>
    ${taskGoalFrameHtml}
    <div class="info-block tasks-panel">${tasksHtml}</div>
  ` : `
    <h3 class="section">Was the task completed?</h3>
    <div class="completion-banner" style="border-left:4px solid ${vColor}">
      <span class="comp-verdict" style="color:${vColor}">${esc(comp.task_completed
        ? outcomeWords(verdict, comp.failure_kind) : '-')}</span>
      <span class="comp-detail">${undone
        ? `goal reached ${comp.goal_reached_at_s != null ? fmtT(comp.goal_reached_at_s)
          : '?'}, undone ${comp.undone_at_s != null ? fmtT(comp.undone_at_s) : '?'}`
        : `goal frame ${comp.completed_at_s != null ? fmtT(comp.completed_at_s) : 'not reached'}`}</span>
    </div>
    <div class="info-block kv-block">
      ${comp.success_predicate ? `<div class="kv"><div class="kv-k">${esc(who)}'s success criterion</div><div `
        + `class="kv-v">${esc(comp.success_predicate)}</div></div>` : ''}
      ${comp.reason ? `<div class="kv"><div class="kv-k">What actually happened</div><div `
        + `class="kv-v">${esc(comp.reason)}</div></div>` : ''}
      ${undone && comp.undone_by ? `<div class="kv"><div class="kv-k">What undid it</div><div `
        + `class="kv-v">${esc(comp.undone_by)}</div></div>` : ''}
    </div>
    ${comp.completed_at_s != null ? `<div class="goal-frame" data-t="${comp.completed_at_s}"><div `
      + `class="goal-frame-cap">Goal frame at ${fmtT(comp.completed_at_s)} (click to jump)</div><img `
      + `src="${frameSrc(eidEnc, mainCam, comp.completed_at_s)}" alt="goal frame" loading="lazy" `
      + `onerror="this.parentElement.classList.add('nofr')"></div>` : ''}
    ${(undone && comp.goal_reached_at_s != null) ? `<div class="goal-frame" data-t="${comp.goal_reached_at_s}"><div `
      + `class="goal-frame-cap">Goal reached at ${fmtT(comp.goal_reached_at_s)}, later undone${comp.undone_at_s != null
      ? ` at ${fmtT(comp.undone_at_s)}` : ''} (click to jump)</div><img src="${frameSrc(eidEnc, mainCam,
      comp.goal_reached_at_s)}" alt="goal reached frame" loading="lazy" `
      + `onerror="this.parentElement.classList.add('nofr')"></div>` : ''}
  `;

  // gripper-only rigs (FastUMI): the notes go in a strip under the camera row, never over the image
  const gripOnly = !isEgo && !hasTop;
  const notesHtml = `
          <div class="state-toast" id="state-toast"></div>
          <div class="recovery-overlay" id="recovery-overlay"></div>
          <div class="video-overlay" id="video-overlay"></div>
  `;
  leftCol.innerHTML = `
    <div class="video-wrap">
      <div class="cam-row${isEgo ? ' cam-row-single' : (!hasTop ? ' cam-row-grippers' + (sideCams.length ? ''
        : ' cam-row-single') : '')}">
        <div class="cam-cell cam-exo">
          ${isEgo ? '' : `<span class="cam-label">${esc(camLabel(mainCam))}</span>`}
          <button class="fs-btn" id="fs-btn" title="fullscreen (keeps overlays)">&#9974;</button>
          <video id="video" controls controlslist="nofullscreen" preload="auto" playsinline${keep ? '' : ` src="${videoUrl}"${posterAttr(eidEnc,
            mainCam)}`}></video>${dpHtml(mainCam, dpViews)}
          ${isEgo ? '<canvas class="hp-canvas" id="hp-canvas" aria-hidden="true"></canvas>' : ''}
          ${isEgo ? `<div class="top-hud" id="top-hud"><div class="top-hud-in">
            <div class="th-l">${progOverlayHtml}</div>
            <div class="ego-status" id="ego-status">
              <span class="es-pov ${d.viewpoint === 'third_person' ? 'exo' : ''}">${
                d.viewpoint === 'third_person' ? '&#9673; exo &middot; third-person'
                : d.viewpoint === 'first_person' ? '&#9673; ego &middot; first-person'
                : '&#9673; ego view'}</span>
              <span class="es-hand" id="es-hand"></span>
            </div>
            <div class="th-r"></div>
          </div></div>` : progOverlayHtml}
          ${gripOnly ? '' : notesHtml}
        </div>
        ${isEgo ? '' : sideCams.map(v => `
        <div class="cam-cell cam-wrist">
          <span class="cam-label">${esc(camLabel(v))}</span>
          <video id="${sideId(v)}" preload="auto" muted playsinline${gripOnly
            ? ' controls' : ''}${keep ? '' : ` src="${v === 'left' ? videoUrlWL : v === 'right' ? videoUrlWR
            : videoSrc(eidEnc, v)}"${posterAttr(eidEnc, v)}`} onloadedmetadata="this.currentTime=0.03"></video>${dpHtml(v,
            dpViews)}
        </div>`).join('')}${unshownCellsHtml(d, eidEnc, !!keep)}
      </div>${unshownNote(d)}
      ${gripOnly ? `<div class="grip-strip">${sideCams.length ? '' : `<p class="grip-note">A single-arm task: the dataset records one gripper camera.</p>`}${notesHtml}</div>` : ''}
    </div>
    <div class="timeline" id="timeline">
      ${markersHtml}${keyMarkersHtml}${comp.completed_at_s != null ? `<div class="marker goal" `
        + `style="left:${(comp.completed_at_s / duration) * 100}%" data-t="${comp.completed_at_s}"><div `
        + `class="tip">goal reached ${fmtT(comp.completed_at_s)}</div></div>` : ''}${taskGoalTimes.map((gt,
        i) => `<div class="marker goal" style="left:${(gt / duration) * 100}%" data-t="${gt}"><div class="tip">task `
        + `done ${fmtT(gt)}</div></div>`).join('')}${ticksHtml}
      <div class="playhead" id="playhead" style="left:0%"></div>
    </div>
    ${laneHtml}${belowLanes}
    ${failed ? '' : noLabels ? `${noLabelsHtml(d._label_failed)}

    ${problemsAndNotes}

    ${pubHtml}
    ` : `
    <h3 class="section">Key events <span class="count">${keyEvents.length + untimed.keys.length}</span></h3>
    <div class="info-block"><div class="key-events">${keyPanelHtml || '<span style="color:var(--fg-3)">none</span>'}`
      + `</div></div>

    ${tasksSection}

    ${perfHtml ? `<h3 class="section">How well was the task executed?</h3>${perfHtml}` : ''}
    ${rcHtml ? `<h3 class="section">Recoveries</h3><div class="info-block">${rcHtml}</div>` : ''}

    ${problemsAndNotes}

    ${pubHtml}

    ${scHtml ? `<h3 class="section">What physically changed over the episode?</h3><div class="info-block">${scHtml}`
      + `</div>` : ''}

    <h3 class="section">Object relationships</h3>
    <div class="info-block sg-snap" id="sg-snap"></div>

    <h3 class="section">Workspace inventory</h3>
    <div class="info-block"><div class="inv-list">${invHtml || '<span style="color:var(--fg-3)">empty</span>'}</div>`
      + `</div>
    `}
  `;
  if (keep) {
    // the kept footage takes the place of the new, empty elements, so it plays on without reloading
    for (const [id, el] of [['video', keep.video], ...keep.side]) {
      const fresh = document.getElementById(id);
      if (fresh && el) fresh.replaceWith(el);
    }
  }

  // RIGHT COLUMN: task banner (+ instruction variants) + dense timeline.
  // Merge dense segments and key-event milestones into one time-ordered feed
  // so key events read clearly inline as the video plays.
  const feedItems = [];
  for (const e of eventLabels) if (e.t_s != null) feedItems.push({t: e.t_s, kind: 'seg', e});
  for (const k of keyEvents) feedItems.push({t: k.t_s, kind: 'key', k});
  feedItems.sort((a, b) => (a.t - b.t) || (a.kind === 'key' ? -1 : 1));

  let feedHtml = '';
  feedItems.forEach((it, idx) => {
    if (it.kind === 'key') {
      const k = it.k; const oc = (k.outcome || '').toLowerCase();
      const kindChip = k.kind ? `<span class="ke-kind">${esc(kindName(k.kind))}</span>` : '';
      feedHtml += `<div class="ev keyrow" data-t="${k.t_s}" data-idx="${idx}">
        <span class="t">${fmtT(k.t_s)}</span>
        <span class="ke-col"><span class="ke-badge">&#9670; key event</span></span>
        <span class="ke-text">${kindChip}${esc(k.label || '')}</span>
        ${oc ? `<span class="outcome ${oc}">${esc(oc)}</span>` : '<span></span>'}
      </div>`;
    } else {
      const e = it.e; const cc = contribClass(e.contribution);
      const cl = (e.contribution || '').toLowerCase() || '-';
      feedHtml += `<div class="ev" data-t="${e.t_s}" data-idx="${idx}">
        <span class="t">${fmtT(e.t_s)}</span>
        <span class="who"><span class="arm ${esc(e.arm || '')}">${esc(armLabel(e.arm))}</span></span>
        <span class="phrase">${buildPhrase(e)}</span>
        <span class="contrib ${cc}">${esc(cl)}</span>
      </div>`;
    }
  });

  feedHtml += untimed.steps.join('');
  rightCol.innerHTML = failed ? cmpFailHtml(cmpInfo, who, usage) : noLabels
    ? cmpFailHtml(d._label_failed, who, usage, true) : `
    <div class="prompt-banner${givenMode ? ' has-given' : ''}">
      <div class="label">${bannerLabel}</div>
      ${bannerBody}
      ${promptMetaHtml}
      ${ivHtml && !hasTasks ? `<div class="task-variants"><div class="tv-label">Prompt variants</div><ul>${ivHtml}`
        + `</ul></div>` : ''}
    </div>

    <h3 class="section dense-head">Dense timeline</h3>
    <div class="dense-stats">
      <span class="ds-total">${eventLabels.length} steps</span>
      <span class="ds-chip adv">${contribs.advancing} advancing</span>
      <span class="ds-chip waste">${contribs.wasteful} wasteful</span>
      <span class="ds-chip idle">${contribs.idle} idle</span>
    </div>
    <div class="feed" id="feed">${feedHtml || '<div class="ev empty"><span class="phrase">no events</span></div>'}</div>
  `;

  // wire interactions
  const vid = document.getElementById('video');
  const ph  = document.getElementById('playhead');
  const tl  = document.getElementById('timeline');

  // Fullscreen the exo cell (video + overlays), not the bare video, so overlays
  // stay on screen. The player's own fullscreen (its button, hidden via CSS, its menu item, off by controlslist, and
  // a double-click on the video, which Chrome takes as fullscreen) would promote only the video, so a double-click
  // fullscreens the cell as the button does.
  const fsBtn = document.getElementById('fs-btn');
  const exoCell = document.querySelector('.cam-cell.cam-exo');
  const fsToggle = () => {
    const fsEl = document.fullscreenElement || document.webkitFullscreenElement;
    if (fsEl) { (document.exitFullscreen || document.webkitExitFullscreen).call(document); }
    else { (exoCell.requestFullscreen || exoCell.webkitRequestFullscreen).call(exoCell); }
  };
  if (fsBtn && exoCell) {
    fsBtn.addEventListener('click', fsToggle);
    exoCell.addEventListener('dblclick', e => { if (e.target.closest('button, a')) return; e.preventDefault(); fsToggle(); });
  }

  // The exo <video> is object-fit:contain, so when the cell's aspect differs
  // from the frame's, the actual image is letterboxed with black bars and does
  // not fill the element. Overlays are positioned against the cell, so without
  // this they sit on the black bars. Compute the real displayed-image rect and
  // publish it as CSS vars (--fx-left/top/right/bottom insets) so every overlay
  // hugs the frame, not the padding. Recomputed on metadata/resize/fullscreen.
  function layoutFrame() {
    if (!vid || !exoCell || !vid.videoWidth || !vid.videoHeight) return;
    const elW = vid.clientWidth, elH = vid.clientHeight;
    if (!elW || !elH) return;
    const ar = vid.videoWidth / vid.videoHeight;
    let dW = elW, dH = elW / ar;
    if (dH > elH) { dH = elH; dW = elH * ar; }
    const cellR = exoCell.getBoundingClientRect(), vidR = vid.getBoundingClientRect();
    const elLeft = vidR.left - cellR.left, elTop = vidR.top - cellR.top;
    const fx = elLeft + (elW - dW) / 2, fy = elTop + (elH - dH) / 2;
    exoCell.style.setProperty('--fx-left', fx.toFixed(1) + 'px');
    exoCell.style.setProperty('--fx-top', fy.toFixed(1) + 'px');
    exoCell.style.setProperty('--fx-right', (cellR.width - (fx + dW)).toFixed(1) + 'px');
    exoCell.style.setProperty('--fx-bottom', (cellR.height - (fy + dH)).toFixed(1) + 'px');
  }
  if (vid) {
    on(vid, 'loadedmetadata', layoutFrame);
    on(vid, 'loadeddata', layoutFrame);
  }
  // Tear down the window/document listeners + interval from the PREVIOUS episode
  // before wiring this one. renderEp runs on every episode/dataset switch; without
  // this, each switch leaks another resize/fullscreen listener + 500ms interval
  // (all closing over the detached previous video), degrading a long-open session.
  if (window._epCleanup) { for (const fn of window._epCleanup) { try { fn(); } catch (_) {} } }
  window._epCleanup = [];
  const onResize = () => { layoutFrame(); alignHeads(); };
  const onFs = () => setTimeout(layoutFrame, 60);
  window.addEventListener('resize', onResize);
  alignHeads();
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(alignHeads);
  document.addEventListener('fullscreenchange', onFs);
  document.addEventListener('webkitfullscreenchange', onFs);
  window._epCleanup.push(() => window.removeEventListener('resize', onResize));
  window._epCleanup.push(() => { document.removeEventListener('fullscreenchange', onFs);
    document.removeEventListener('webkitfullscreenchange', onFs); });
  setTimeout(layoutFrame, 120);

  // Slave the two wrist videos to the exo (primary) video. Exo carries
  // controls; wrists track currentTime / play / pause / rate, no audio.
  // Hard-resync after each scrub to fight drift introduced by separate
  // video elements decoding at slightly different rates.
  // every camera beside the main one follows it: the two mounted cameras and any others
  const slaves = [...document.querySelectorAll('.cam-cell.cam-wrist video:not(.dp-vid)')];
  const SYNC_TOL = 0.10;  // seconds; tighter than this won't reseek
  function syncSlavesNow() {
    if (!vid) return;
    for (const s of slaves) {
      if (Math.abs(s.currentTime - vid.currentTime) > SYNC_TOL) {
        try { s._synced = true; s.currentTime = vid.currentTime; } catch (_) {}
      }
    }
  }
  // two UMI grippers side by side each carry the player's controls, and either one drives both: a play, pause,
  // scrub or speed change on the second camera goes to the first, which the second then follows as usual (a seek
  // the sync itself made is not sent back)
  if (vid && gripOnly) for (const s of slaves) {
    on(s, 'play', () => { if (vid.paused) vid.play().catch(()=>{}); });
    on(s, 'pause', () => { if (!vid.paused && !s.ended) vid.pause(); });
    on(s, 'seeked', () => {
      if (s._synced) { s._synced = false; return; }
      if (Math.abs(s.currentTime - vid.currentTime) > SYNC_TOL) vid.currentTime = s.currentTime;
    });
    on(s, 'ratechange', () => { if (vid.playbackRate !== s.playbackRate) vid.playbackRate = s.playbackRate; });
  }
  if (vid) {
    on(vid, 'play', () => {
      syncSlavesNow();
      for (const s of slaves) { s.play().catch(()=>{}); }
    });
    on(vid, 'pause', () => {
      for (const s of slaves) { s.pause(); }
      syncSlavesNow();
    });
    on(vid, 'seeked', syncSlavesNow);
    on(vid, 'ratechange', () => {
      for (const s of slaves) { s.playbackRate = vid.playbackRate; }
    });
    // Periodic drift correction during playback (torn down on the next episode).
    const _driftIv = setInterval(() => {
      if (!vid.paused) syncSlavesNow();
    }, 500);
    if (window._epCleanup) window._epCleanup.push(() => clearInterval(_driftIv));
  }

  const seek = (t) => {
    if (vid && t != null && t !== '' && !isNaN(parseFloat(t))) {
      vid.currentTime = parseFloat(t);
      // Auto-play from the clicked point (slaves follow via the 'play' handler).
      vid.play().catch(()=>{});
      // Bring the video back into view: a timestamp clicked from a section
      // scrolled far down (recoveries, scene graph) is useless if you then have
      // to scroll up manually while it plays past. The video is the first thing
      // in its column, so scroll that column to the top (deterministic; a smooth
      // scrollIntoView over a long distance overshoots and clips the top).
      // Instant, not smooth: a multi-second scroll animation would let the clip
      // play past before the video is even on screen.
      let sc = vid.parentElement;
      while (sc && sc !== document.body && !/(auto|scroll)/.test(getComputedStyle(sc).overflowY)) sc = sc.parentElement;
      if (sc && sc !== document.body) sc.scrollTop = 0;
      else vid.scrollIntoView({block: 'start'});
    }
  };
  document.querySelectorAll('.timeline .marker').forEach(m => {
    m.addEventListener('click', e => { e.stopPropagation(); seek(m.dataset.t); });
  });
  document.querySelectorAll('.feed .ev[data-t]').forEach(r => {
    r.addEventListener('click', () => seek(r.dataset.t));
  });
  document.querySelectorAll('.key-events .key-ev[data-t]').forEach(r => {
    r.addEventListener('click', () => seek(r.dataset.t));
  });
  document.querySelectorAll('#lane-hands .lane-step').forEach(bt => bt.addEventListener('click', () => {
    const t = vid ? vid.currentTime : 0;
    // previous: the start of the stretch the playhead is in once it is over a second into it, else the one before
    const i = bt.dataset.dir === '1' ? handStretches.findIndex(([a]) => a > t + 0.05)
      : handStretches.map(([a]) => a).reduce((k, a, j) => (a < t - 1 ? j : k), -1);
    if (i >= 0) seek(handStretches[i][0]);
  }));
  document.querySelectorAll('.lane .lane-seg[data-t], .pub-list .pub-row[data-t], .pub-list .pub-at[data-t]').forEach(r => {
    r.addEventListener('click', () => seek(r.dataset.t));
  });
  document.querySelectorAll('.pub-show').forEach(b => b.addEventListener('click', () => {
    const box = b.closest('.pub-fold'), on = !box.classList.contains('shown');
    box.classList.toggle('shown', on);
    b.setAttribute('aria-expanded', String(on));
    b.textContent = on ? b.dataset.open : b.dataset.closed;
  }));
  document.querySelectorAll('.rec [data-t]').forEach(r => {
    r.addEventListener('click', e => { e.stopPropagation(); seek(r.dataset.t); });
  });
  document.querySelectorAll('.goal-frame[data-t]').forEach(g => {
    g.addEventListener('click', () => seek(g.dataset.t));
  });
  markTallGoalFrames();
  document.querySelectorAll('.task-row[data-t]').forEach(r => {
    r.addEventListener('click', () => seek(r.dataset.t));
  });
  document.querySelectorAll('.di-block .di-row[data-t]').forEach(r => {
    r.style.cursor = 'pointer';
    r.addEventListener('click', () => seek(r.dataset.t));
  });
  if (tl) tl.addEventListener('click', e => {
    if (e.target.closest('.marker')) return;
    const rect = tl.getBoundingClientRect();
    const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    seek(pct * duration);
  });

  // ----- bottom overlay: current dense segment + key-event flag -----
  const segWindows = eventLabels
    .filter(e => e.t_s != null)
    .map(e => ({t0: e.t_s, t1: (e.end_s != null ? e.end_s : e.t_s + 1.0), ev: e}))
    .sort((a, b) => a.t0 - b.t0);
  const keyWindows = keyEvents.map(k => ({t0: k.t_s - 0.3, t1: k.t_s + 2.5, k}));
  function activeKey(t) { for (const w of keyWindows) if (t >= w.t0 && t <= w.t1) return w.k; return null; }
  const stateChanges = (d.state_changes || []).filter(x => x.t_s != null).sort((a, b) => a.t_s - b.t_s);
  function activeState(t) { for (const x of stateChanges) if (t >= x.t_s - 0.3 && t <= x.t_s + 2.5) return x;
    return null; }

  const overlay = document.getElementById('video-overlay');
  // Only rebuild the overlay when its content actually changes (segment, goal/key
  // window, contribution). Rebuilding every timeupdate (~4x/s) churned the DOM and
  // made the CSS transitions re-fire, which read as flicker.
  let _ovSig = null;
  function renderOverlay(t) {
    if (!overlay) return;
    let w = null;
    for (const s of segWindows) { if (s.t0 <= t + 0.05) w = s; else break; }
    if (!w) { if (_ovSig !== '') { _ovSig = ''; overlay.classList.remove('active', 'adv', 'waste', 'idle', 'none');
      } return; }
    const ev = w.ev;
    const cc = contribClass(ev.contribution);
    const cl = (ev.contribution || '').toLowerCase() || '-';
    const goalT = comp.completed_at_s;
    const atEpisodeGoal = goalT != null && t >= goalT - 0.3 && t <= goalT + 3.0;
    const atTaskGoal = taskGoalTimes.some(gt => t >= gt - 0.3 && t <= gt + 3.0);
    const k = activeKey(t);
    const kOc = k ? (k.outcome || '').toLowerCase() : '';
    const sig = [ev.event_idx != null ? ev.event_idx : ev.t_s, cc, atEpisodeGoal, atTaskGoal, k ? k.label : '',
      kOc].join('|');
    if (sig === _ovSig) return;
    _ovSig = sig;
    const goalHtml = (atEpisodeGoal || atTaskGoal)
      ? `<div class="vo-goal"><span class="vo-goal-badge">&#10003; ${atTaskGoal && !atEpisodeGoal ? 'task done'
        : 'goal reached'}</span></div>` : '';
    const keyHtml = k ? `<div class="vo-key"><span class="vo-key-badge">&#9670; ${esc(kindName(k.kind)
      || 'key event')}</span><span class="vo-key-label">${esc(k.label || '')}</span>${kOc
      ? `<span class="vo-key-outcome ${kOc}">${esc(kOc)}</span>` : ''}</div>` : '';
    overlay.innerHTML = `
      ${goalHtml}
      ${keyHtml}
      <span class="vo-time">${ev.t_s.toFixed(1)}s</span>
      <span class="vo-arm ${esc(ev.arm || '')}">${esc(armLabel(ev.arm))}</span>
      <span class="vo-phrase">${buildPhrase(ev)}</span>
      <span class="vo-contrib ${cc}">${esc(cl)}</span>`;
    overlay.classList.remove('adv', 'waste', 'idle', 'none');
    overlay.classList.add('active', cc);
  }

  // ----- head camera: hands out of view -----
  // A head camera often points away from the hands, so a step's action is inferred rather than seen. The model marks
  // this per step (hands_visible); the spans where it is false are highlighted, and what the hands wear is shown.
  const esHand = document.getElementById('es-hand');
  const isBare = w => !w || ['bare', 'none', 'null', 'nothing'].includes(String(w).trim().toLowerCase());
  let _handSig = null;
  function renderHands(t) {
    if (!isEgo || !exoCell) return;   // head cameras only: a wrist camera is bolted to its gripper
    // active segment (same rule as the bottom overlay)
    let ev = null;
    for (const s of segWindows) { if (s.t0 <= t + 0.05) ev = s.ev; else break; }
    const hidden = ev ? ev.hands_visible === false : false;
    const gloves = (ev && ev.hands_visible !== false && !isBare(ev.hands_wearing)) ? ev.hands_wearing : '';
    const sig = hidden ? 'H' : (gloves ? 'G:' + gloves : '');
    if (sig === _handSig) return;   // only touch the DOM on change
    _handSig = sig;
    exoCell.classList.toggle('hands-hidden', hidden);
    if (!esHand) return;
    esHand.classList.remove('hidden-hands', 'gloved', 'show');
    if (hidden) { esHand.textContent = 'Hands out of view'; esHand.classList.add('show', 'hidden-hands'); }
    else if (gloves) { esHand.textContent = 'Wearing ' + String(gloves).replace(/[\u2013\u2014]/g, '-');
      esHand.classList.add('show', 'gloved'); }
  }

  // ----- a session of tasks: the goal frame of the task nearest the playhead (switches as it plays) -----
  const taskGoals = tasks
    .filter(t => t.completed_at_s != null)
    .map((t, i) => ({t: t.completed_at_s, name: t.task || `task ${i + 1}`, oc: (t.outcome || '').toLowerCase(),
                     num: tasks.indexOf(t) + 1}))
    .sort((a, b) => a.t - b.t);
  const tgFrame = document.getElementById('task-goal-frame');
  const tgImg = document.getElementById('task-goal-img');
  const tgCap = document.getElementById('task-goal-cap');
  let _tgIdx = -1;
  function renderTaskGoal(t) {
    if (!tgFrame || !taskGoals.length) return;
    let bi = 0, bd = Infinity;
    for (let i = 0; i < taskGoals.length; i++) {
      const dd = Math.abs(taskGoals[i].t - t);
      if (dd < bd) { bd = dd; bi = i; }
    }
    if (bi === _tgIdx) return;   // only reload the frame when the nearest goal changes
    _tgIdx = bi;
    const g = taskGoals[bi];
    tgFrame.dataset.t = g.t;
    tgCap.innerHTML = `Task ${g.num} goal &middot; ${esc(g.name)} &middot; ${fmtT(g.t)} (click to jump)`;
    tgImg.src = frameSrc(eidEnc, mainCam, g.t);
  }

  // ----- top-right toast: object state-changes (transient, sparse) -----
  const stateToast = document.getElementById('state-toast');
  let _stSig = null;
  function renderState(t) {
    if (!stateToast) return;
    const sc = activeState(t);
    const sig = sc ? `${sc.t_s}|${sc.object}|${sc.from}|${sc.to}` : '';
    if (sig === _stSig) return;   // no change -> leave the DOM (and its transition) alone
    _stSig = sig;
    if (!sc) { stateToast.classList.remove('active'); return; }
    stateToast.innerHTML =
      `<div class="st-badge">state change</div>` +
      `<div class="st-obj">${esc(sc.object || '')}</div>` +
      `<div class="st-fromto"><span class="st-from">${esc(sc.from || '')}</span>` +
      `<span class="st-arr">&#8594;</span><span class="st-to">${esc(sc.to || '')}</span></div>`;
    stateToast.classList.add('active');
    placeTop();
  }

  // ----- top-middle overlay: recovery, shown for the whole recovery period
  // (from the failure until it is corrected), with the failure + the fix. -----
  const recWindows = recoveries
    .filter(r => r.failureT != null)
    .map(r => ({t0: r.failureT - 0.3,
                t1: (r.recoveredAt != null ? r.recoveredAt + 1.0 : r.failureT + 3.5), r}))
    .sort((a, b) => a.t0 - b.t0);
  const recOverlay = document.getElementById('recovery-overlay');
  const progOverlay = document.getElementById('prog-overlay');
  const topHud = document.getElementById('top-hud');
  // The notes that come and go at the top of the image stack under what is always there, so none covers another at
  // any width or text length: a state change goes under the progress chip (on a head camera, under the top row it
  // sits in), which a narrow image leaves no room beside; the recovery banner goes under both.
  function placeTop() {
    if (!exoCell) return;
    const cr = exoCell.getBoundingClientRect();
    const below = c => c.getBoundingClientRect().bottom - cr.top + 8;
    let top = 8;
    const anchor = topHud || progOverlay;
    // the main camera's depth switch sits under the full-screen button: the notes go under it too
    const dpBtn = exoCell.querySelector('.cam-dp:not([hidden])');
    if (anchor && stateToast && !stateToast.closest('.grip-strip')) {
      top = below(anchor);
      stateToast.style.top = Math.max(top, below(fsBtn || anchor), dpBtn ? below(dpBtn) : 0) + 'px';
    }
    if (!recOverlay) return;
    if (dpBtn) top = Math.max(top, below(dpBtn));
    for (const c of [progOverlay, stateToast]) {
      if (c && (c === progOverlay || c.classList.contains('active'))) top = Math.max(top, below(c));
    }
    recOverlay.style.top = top + 'px';
  }
  let _recSig = null;
  function renderRecovery(t) {
    if (!recOverlay) return;
    let w = null;
    for (const x of recWindows) { if (t >= x.t0 && t <= x.t1) { w = x; break; } }
    const sig = w ? `${w.r.failureT}|${w.r.recovered}|${w.r.recoveredAt}` : '';
    if (sig !== _recSig) {
      _recSig = sig;
      if (!w) { recOverlay.classList.remove('active'); }
      else {
        const r = w.r;
        const failed = r.recovered === false;
        // The badge carries the outcome so there is no contradictory "recovery:
        // not recovered". A failed attempt reads "failed recovery" (red); a
        // successful one keeps "recovery" + a green "recovered at" pill.
        const badge = failed
          ? `<span class="ro-badge failed">&#10007; failed recovery</span>`
          : `<span class="ro-badge">&#8635; recovery</span>`;
        const status = r.recovered === true
          ? `<span class="ro-status ok">recovered${r.recoveredAt != null ? ' at ' + fmtT(r.recoveredAt) : ''}</span>`
          : '';
        // If it recovered, the meaningful field is what it actually DID; the ideal
        // "how to fix" prescription only makes sense for a failure never recovered.
        // Never show both (a successful recovery does not need a prescription).
        const detail = r.recovered === true
          ? (r.correction ? `<div class="ro-fix"><span class="ro-fix-badge">did</span> <span `
            + `class="ro-fix-text">${esc(r.correction)}</span></div>` : '')
          : (r.howToFix ? `<div class="ro-fix"><span class="ro-fix-badge">how to fix</span> <span `
            + `class="ro-fix-text">${esc(r.howToFix)}</span></div>` : '');
        recOverlay.innerHTML =
          `<div class="ro-row">${badge}${status}</div>` +
          `<div class="ro-fail">${esc(r.failure || '')}</div>` +
          detail;
        recOverlay.classList.toggle('failed', failed);
        recOverlay.classList.add('active');
      }
    }
    // reposition every active frame (the top-corner chips it sits below come and
    // go independently); the CSS `top` transition turns each move into a slide.
    if (w) placeTop();
  }

  // ----- dense feed: highlight + auto-scroll the active row -----
  const feedEl = document.getElementById('feed');
  const rows = feedEl ? [...feedEl.querySelectorAll('.ev[data-t]')] : [];
  const rowTimes = rows.map(r => parseFloat(r.dataset.t));
  let lastRow = -1;
  function syncFeed(t) {
    let idx = -1;
    for (let i = 0; i < rowTimes.length; i++) { if (rowTimes[i] <= t + 0.05) idx = i; else break; }
    if (idx === lastRow) return;
    if (lastRow >= 0 && rows[lastRow]) rows[lastRow].classList.remove('active');
    lastRow = idx;
    if (idx >= 0 && rows[idx]) {
      rows[idx].classList.add('active');
      // Only auto-follow the feed while the video is actually playing. The feed
      // lives in the scrolling right column, so scrolling a row into view on
      // load/select would drag the column down past the prompt to timestamp 1.
      // When paused (just selected an episode, or scrubbed), leave the scroll be.
      // In the stacked narrow layout the column is part of the page, so following would scroll the page
      // and pull the video off screen; follow only when the column scrolls on its own. Only the column
      // scrolls: scrollIntoView would also scroll every page around the board (Data Review's job page
      // embeds it), pushing that page down as the rows advance.
      if (vid && !vid.paused && getComputedStyle(rightCol).overflowY !== 'visible') {
        const margin = parseFloat(getComputedStyle(rows[idx]).scrollMarginTop) || 0;
        const top = rows[idx].getBoundingClientRect().top - rightCol.getBoundingClientRect().top + rightCol.scrollTop - margin;
        rightCol.scrollTo({top: Math.max(0, top), behavior: 'smooth'});
      }
    }
  }

  // ----- key events: the one the playhead last passed is marked -----
  const keyEls = [...document.querySelectorAll('.key-events .key-ev')];
  let lastKey = -2;
  function syncKeyEvents(t) {
    let ai = -1;
    for (let i = 0; i < keyEvents.length && keyEvents[i].t_s <= t + 0.05; i++) ai = i;
    if (ai === lastKey) return;
    lastKey = ai;
    keyEls.forEach((el, i) => el.classList.toggle('now', i === ai));
  }

  // ----- scene graph: playhead-synced snapshot -----
  const REL_WORDS = ['next-to', 'next to', 'supported by', 'attached to', 'held by', 'resting on', 'resting against',
                     'leaning against', 'on top of', 'in front of', 'inside of', 'holding', 'supporting', 'grasping',
                     'gripping', 'against', 'between', 'above', 'below', 'beneath', 'onto', 'into', 'inside', 'in',
                     'on', 'at', 'near', 'under', 'over', 'beside', 'by'];
  const REL_RE = new RegExp(`\\s+(${REL_WORDS.join('|')})\\s+`, 'i');
  function parseRel(s) {
    const m = s.match(REL_RE);
    if (!m) return {subj: s, rel: '', obj: ''};
    const i = s.indexOf(m[0]);
    return {subj: s.slice(0, i), rel: m[1], obj: s.slice(i + m[0].length)};
  }
  const sgSnaps = (d.scene_graph || []).filter(g => g.t_s != null).slice().sort((a, b) => a.t_s - b.t_s);
  const sgEl = document.getElementById('sg-snap');
  let lastSg = -2;
  function renderSceneGraph(t) {
    if (!sgEl) return;
    if (!sgSnaps.length) { sgEl.innerHTML = '<span style="color:var(--fg-3)">none</span>'; return; }
    let si = 0;
    for (let i = 0; i < sgSnaps.length; i++) { if (sgSnaps[i].t_s <= t + 0.05) si = i; else break; }
    if (si === lastSg) return;
    lastSg = si;
    const s = sgSnaps[si];
    const rels = (s.relations || []).map(r => {
      const p = parseRel(r);
      return p.rel
        ? `<div class="sg-rel"><span class="subj">${esc(p.subj)}</span><span class="relpill">${esc(p.rel)}</span><span `
          + `class="obj">${esc(p.obj)}</span></div>`
        : `<div class="sg-rel"><span class="subj">${esc(p.subj)}</span></div>`;
    }).join('');
    sgEl.innerHTML = `<div class="sg-when">nearest snapshot at ${fmtT(s.t_s)} (the scene is captured only at key `
      + `moments)</div>${rels}`;
  }

  // ----- top-left progress chart: fill the curve up to now + big % readout -----
  const poPct = document.getElementById('po-pct');
  const poPast = document.getElementById('po-past');
  const poDot = document.getElementById('po-dot');
  const PO_W = 100, PO_H = 24;
  function renderProgress(t) {
    if (!poPct) return;
    // the level between two step ends is read off the straight line joining them (progressAt)
    const p = progressAt(progPts, t);
    const xy = (tt, pp) => `${(tt / duration * PO_W).toFixed(2)},${((1 - pp) * PO_H).toFixed(2)}`;
    const past = progPts.filter(pt => pt.t <= t + 0.05).map(pt => xy(pt.t, pt.p));
    if (progPts.some(pt => pt.t > t + 0.05)) past.push(xy(Math.min(t, duration), p));
    const cx = Math.max(0, Math.min(1, duration ? t / duration : 0)) * PO_W;
    if (poPast) poPast.setAttribute('points', past.join(' '));
    if (poDot) { poDot.setAttribute('cx', cx.toFixed(2)); poDot.setAttribute('cy', ((1 - p) * PO_H).toFixed(2)); }
    poPct.textContent = progressPct(p) + '%';
  }

  // the lanes' playheads, the dataset label under the playhead, and its row in the list
  let _pubIdx = -2, _handIdx = -2;
  function syncLanes(t) {
    document.querySelectorAll('.lane .lane-ph').forEach(p => { p.style.left = lanePct(t) + '%'; });
    if (handStretches.length) {
      const k = handStretches.findIndex(([a, b]) => a <= t + 0.05 && t < b);
      if (k !== _handIdx) {
        _handIdx = k;
        document.querySelectorAll('#lane-hands .lane-seg').forEach(g => g.classList.toggle('now', +g.dataset.i === k));
        const pos = document.getElementById('lane-hands-pos');
        if (pos) pos.textContent = k >= 0 ? `${k + 1} of ${handStretches.length}`
          : `${handStretches.length} ${handStretches.length === 1 ? 'stretch' : 'stretches'}`;
      }
    }
    if (!pubLabels.length) return;
    let idx = -1;
    // a moment (no end time) is the label under the playhead for a second after it
    for (let i = 0; i < pubLabels.length; i++) { const x = pubLabels[i]; if (x.t0 == null) continue;
      if (x.t0 <= t + 0.05 && t < (x.t1 != null && x.t1 > x.t0 ? x.t1 : x.t0 + 1)) { idx = i; break; } }
    if (idx === _pubIdx) return;
    _pubIdx = idx;
    const now = document.getElementById('lane-pub-now');
    if (now) now.textContent = idx >= 0 ? String(pubLabels[idx].label).replace(/[\u2013\u2014]/g, '-')
      : 'no label here';
    document.querySelectorAll('.lane-seg.pub, .pub-row').forEach(el => el.classList.toggle('now',
      Number(el.dataset.i) === idx));
  }
  let tcWire = null;            // the Touch lane and its card (setupTouch), on an episode with contacts
  function syncAll(t) {
    if (duration > 0 && ph) ph.style.left = (100 * t / duration) + '%';
    // progress + state first so the recovery banner can place itself below the
    // corner chips that are actually visible this frame.
    renderProgress(t); renderState(t); renderRecovery(t);
    renderOverlay(t); renderHands(t); renderTaskGoal(t); syncFeed(t); syncKeyEvents(t); renderSceneGraph(t);
      syncLanes(t);
    if (window._sn) window._sn.sync(t);
    if (tcWire) tcWire.sync(t);
  }
  if (vid) {
    on(vid, 'timeupdate', () => syncAll(vid.currentTime));
    on(vid, 'seeked', () => syncAll(vid.currentTime));
  }
  setupSensors(_activeFile, duration, seek, on, vid, v => v === mainCam && isEgo ? 'head' : camLabel(v), camViews,
    touch.contacts.length > 0);
  if (touch.contacts.length) tcWire = setupTouch(touch, _activeFile, duration, seek, on, vid);
  setupDepth(_activeFile, eidEnc, on);
  syncAll(vid && keep ? vid.currentTime : 0);
  setupHandPose(vid, exoCell, isEgo, on, _activeFile);
}

// ================= other models' labels, kept apart from the board's own (compare/metrics.py) =================
// compare/index.json names the reference (the model of the board's own labels) and the other models, and lists per
// episode which of them were asked to label it and how each response came out. The "Labels by" control beside the
// episode list switches the whole board to one model's labels, as a comparison: the list, the strip and the filter
// then describe the episodes that model labelled, and every episode shows its labels. The board's own labels are
// the reference's; nothing here touches the board's counts, its Episode JSON download or its JSON Lines export,
// which read the board's label files only. The comparison view charts compare/metrics.json.
let CMP = null;          // {reference: {...}, models: [...], episodes: {file: {key: status}}}
let SHOWN = null;        // whose labels the open episode shows: null is the board's own
let _epToken = 0;
const sleep = ms => new Promise(r => setTimeout(r, ms));
const cmpModel = k => ((CMP && CMP.models) || []).find(m => m.key === k) || null;
const cmpHas = (k, file) => !!(CMP && k && (CMP.episodes[file] || {})[k]);
const cmpCount = k => CMP ? Object.values(CMP.episodes).filter(e => e[k]).length : 0;
// the model of the board's own labels, the reference every other model is measured against
const refName = () => (CMP && CMP.reference && CMP.reference.name) || 'the dashboard';
const withAn = w => (/^[aeiou]/i.test(w) ? 'an ' : 'a ') + w;
const cap = w => w.charAt(0).toUpperCase() + w.slice(1);
// a run whose prompt also held one complete annotation of the reference (of another episode of the same rig): the
// model learns in context from that trace
const icl = () => `in-context learning with ${withAn(refName())} trace`;
// the model's own name, for "<model>'s independent assessment": a run with in-context learning is still that model
function cmpWho(k) {
  const m = cmpModel(k);
  if (!m) return 'The model';
  return ((m.example && cmpModel(m.base)) || m).name;
}
// the run's full name: the model, and for an in-context run, how it was prompted
function cmpFullName(k) {
  const m = cmpModel(k);
  return !m ? 'The model' : m.example ? `${shownName(cmpWho(k))}, ${icl()}` : shownName(m.name);
}
const ST_WORDS = {unparsed: 'did not parse', cut_off: 'cut off', no_response: 'no response', pending: 'not yet run'};
// every model in the menu's order: each model, then its in-context run
function cmpMenuOrder() {
  const ms = (CMP && CMP.models) || [], out = [];
  for (const m of ms.filter(m => !m.example)) { out.push(m); out.push(...ms.filter(x => x.example && x.base === m.key)); }
  return out.concat(ms.filter(m => !out.includes(m)));
}
// one model's labels of one episode, fetched once (a static build first loads the dataset's media paths)
const _cmpGot = new Map();
async function cmpEpisode(key, file) {
  const rec = ALL_EPS.find(e => e.file === file);
  if (rec) await ensureDataset(datasetOf(rec));
  const k = key + '|' + file;
  if (!_cmpGot.has(k)) _cmpGot.set(k, fetchJson(compareEpUrl(key, file)).then(d => { if (!d) _cmpGot.delete(k); return d; }));
  return _cmpGot.get(k);
}
// A response that gave no labels: another model's under Labels by (the comparison's wording), or with own the board's
// own label of the episode (d._label_failed, board/to_board.py label_failed), which keeps the episode's footage,
// checks and sensors on the page above and below this block.
// The line that stands in for the sections a model answers (key events, the outcome, recoveries, what changed, object
// relationships, the inventory) on an episode whose own reply gave no labels.
function noLabelsHtml(lf) {
  const why = lf && lf.status === 'cut_off' ? 'reply was cut off at the output limit'
    : lf && lf.status === 'unreadable' ? 'output file does not read'
    : lf && lf.status === 'no_part' ? 'replies gave no labels for any part of this long recording'
    : 'reply did not parse';
  return `<p class="no-labels">The model's ${why}, so this episode has no labels: no key events, outcome, `
    + `recoveries or scene. Its footage, checks, sensors and the dataset's own labels are shown as recorded.</p>`;
}
function cmpFailHtml(c, who, usage, own) {
  who = String(who || 'The model');
  who = who.charAt(0).toUpperCase() + who.slice(1);
  const cost = usage && usage.est_cost_usd != null
    ? ` The call cost $${Number(usage.est_cost_usd).toFixed(3)}${usage.latency_s != null
      ? ` and took ${fmtDur(usage.latency_s)}` : ''}.` : '';
  const rest = ' The footage, checks and sensors of the episode are shown as recorded.';
  if (c.status === 'unparsed') return `<div class="cmp-fail"><h4>${esc(who)}&rsquo;s response did not parse</h4>
    <p>The response is not valid JSON, so there are no labels to show. ${own ? 'It was not retried or repaired.'
      + rest : 'The comparison counts it as a response that did not parse; it was not retried or repaired.'}${cost}</p>
    ${c.parse_error ? `<div class="cf-k">Parser error</div><pre>${esc(c.parse_error)}</pre>` : ''}
    <div class="cf-k">Start of the response, ${Number(c.raw_chars || 0).toLocaleString()} characters in all</div>`
      + `<pre>${esc(c.raw_head || '(empty)')}</pre></div>`;
  if (c.status === 'no_part') return `<div class="cmp-fail"><h4>No part of the recording has labels</h4>
    <p>This long recording was labelled in parts, and no part's response gave labels.${rest}${cost}</p>
    <div class="cf-k">Each part</div><pre>${esc((c.parts || []).map(g => `part ${g.part}, ${Number(g.t0_s).toFixed(1)}`
      + ` to ${Number(g.t1_s).toFixed(1)} s: ${g.why}`).join('\n'))}</pre></div>`;
  if (c.status === 'unreadable') return `<div class="cmp-fail"><h4>${esc(who)}&rsquo;s output file does not read</h4>
    <p>The labelling run's output file for this episode does not read, so there are no labels to show.${rest}</p>
    ${c.error ? `<div class="cf-k">Error</div><pre>${esc(c.error)}</pre>` : ''}</div>`;
  if (c.status === 'cut_off') return `<div class="cmp-fail"><h4>${esc(who)}&rsquo;s response was cut off</h4>
    <p>The response reached the output limit${c.out_tokens ? ` after ${Number(c.out_tokens).toLocaleString()} output `
      + `tokens` : ''} before its JSON was complete, so there are no labels to show.${own ? rest
      : ' The comparison counts it as a response that did not parse.'}${cost}</p>
    ${c.tail ? `<div class="cf-k">End of the response</div><pre>${esc(c.tail)}</pre>` : ''}</div>`;
  return `<div class="cmp-fail"><h4>No response from ${esc(who)}</h4>
    <p>The call for this episode failed and returned nothing, so there are no labels to show. The comparison reports `
      + `it as no response and leaves it out of the share of responses that parse.</p></div>`;
}

// ---- "Labels by": whose labels the whole board shows ----
const lbEl = document.getElementById('lb'), lbBtn = document.getElementById('lb-btn');
const lbMenu = document.getElementById('lb-menu');
function lbOpen(open) {
  lbEl.classList.toggle('open', open);
  lbBtn.setAttribute('aria-expanded', open ? 'true' : 'false');
}
// a model's own name and the qualifier after its first comma ("GPT-6.1 Sol", "high reasoning")
function nameParts(n) {
  const i = String(n).indexOf(', ');
  return i < 0 ? [String(n), ''] : [n.slice(0, i), n.slice(i + 2)];
}
// the name a reader sees: the qualifier in brackets ("GPT-6.1 Sol (high reasoning)"), the same in every chart and line
function shownName(n) {
  const [h, t] = nameParts(n);
  return t ? `${h} (${t})` : h;
}
// "<model>'s <noun>", with a qualifier after the noun in brackets, never "<model>, <qualifier>'s <noun>"
function whoOwn(who, noun) {
  const [h, t] = nameParts(who);
  return `${esc(h)}&rsquo;s ${noun}${t ? ` (${esc(t)})` : ''}`;
}
function lbNameHtml(m) {
  if (!m) return esc(refName());
  const [head, tail] = nameParts(m.example ? cmpWho(m.key) : m.name);
  const sub = [tail, m.example ? icl() : ''].filter(Boolean).join(', ');
  return `${esc(head)}${sub ? `<small class="lb-sub">${esc(cap(sub))}</small>` : ''}`;
}
function renderLabelsBy() {
  lbEl.hidden = !CMP;
  if (!CMP) return;
  const m = BY ? cmpModel(BY) : null, ref = esc(refName());
  lbEl.classList.toggle('cmp', !!m);
  document.getElementById('lb-name').innerHTML = lbNameHtml(m);
  const n = m ? cmpCount(BY) : ALL_EPS.length;
  lbBtn.title = m ? `${cmpFullName(BY)}: its labels of the ${n} episodes it was asked to label, shown as a comparison`
    : `${refName()}, the dashboard’s own labels`;
  // a comparison is marked where it is chosen; the text stays while the note folds away, so it never jumps
  if (m) document.getElementById('lb-note-in').innerHTML = `<b>A comparison, not ${ref}&rsquo;s labels.</b> The `
    + `downloads stay ${ref}&rsquo;s.`;
  document.getElementById('lb-note').classList.add('off');
  document.getElementById('rail-dl').classList.toggle('off', !!m);
  updateKpExport();
}
function buildLabelsMenu() {
  const here = (_activeFile && CMP.episodes[_activeFile]) || {};
  const row = (key, name, sub, n, on) => `<button type="button" class="lb-opt${on ? ' on' : ''}" role="option" `
    + `aria-selected="${on}" data-k="${esc(key)}"><span class="lb-o-name">${esc(name)}${sub
      ? `<small>${esc(sub)}</small>` : ''}</span><span class="lb-o-n">${n.toLocaleString()}</span></button>`;
  lbMenu.innerHTML = `<div class="lb-group">This dashboard</div>`
    + row('', refName(), 'the dashboard’s own labels', ALL_EPS.length, !BY)
    + `<div class="lb-sep"></div><div class="lb-group">Comparisons</div>`
    + `<div class="lb-gnote">Other models&rsquo; labels of some of the same episodes, from the same prompt and `
      + `frames.</div>`
    + cmpMenuOrder().map(m => row(m.key, cmpWho(m.key), [m.example ? cap(icl()) : '',
      here[m.key] ? 'includes this episode' : ''].filter(Boolean).join(', '), cmpCount(m.key), BY === m.key)).join('')
    + `<button type="button" class="lb-go" data-go="compare"><span>How the models compare<small>Charts of every `
      + `model on the same episodes</small></span><span class="lb-arrow" aria-hidden="true">&rarr;</span></button>`;
}
// the menu is wider than the rail, which clips its children, so it opens as a fixed layer under the button
function placeLabelsMenu() {
  const r = lbBtn.getBoundingClientRect(), pad = 8;
  const w = Math.min(Math.max(340, r.width), innerWidth - 2 * pad);
  const x = Math.min(Math.max(pad, r.left), innerWidth - w - pad), y = r.bottom + 4;
  lbMenu.style.setProperty('--lb-x', x + 'px'); lbMenu.style.setProperty('--lb-y', y + 'px');
  lbMenu.style.setProperty('--lb-w', w + 'px');
  lbMenu.style.setProperty('--lb-h', Math.max(160, innerHeight - y - pad) + 'px');
}
lbBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  const open = !lbEl.classList.contains('open');
  if (open) { buildLabelsMenu(); placeLabelsMenu(); issueFilterEl.classList.remove('open'); }
  lbOpen(open);
});
lbMenu.addEventListener('click', (e) => {
  e.stopPropagation();
  if (e.target.closest('.lb-go')) { lbOpen(false); showCompare(); return; }
  const b = e.target.closest('.lb-opt');
  if (b) setLabeller(b.dataset.k || null);
});
document.addEventListener('click', (e) => { if (!lbEl.contains(e.target)) lbOpen(false); });
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && lbEl.classList.contains('open')) { lbOpen(false); lbBtn.focus(); } });
addEventListener('resize', () => { if (lbEl.classList.contains('open')) placeLabelsMenu(); });
addEventListener('scroll', () => { if (lbEl.classList.contains('open')) placeLabelsMenu(); }, {passive: true});
// load a labeller's records and make it the board's, without drawing anything (the caller redraws)
async function applyLabeller(key) {
  key = key && cmpModel(key) ? key : null;
  const list = key ? await loadCmpList(key) : null;
  if (key && !list) return false;
  BY = key; BY_EPS = list;
  return true;
}
// switch the whole board to another labeller: the list, the filter, the strip and the open episode ease out and back
// in; the open episode stays open (its footage playing) when the new labeller labelled it
let _lbToken = 0;
async function setLabeller(key, opts) {
  opts = opts || {};
  key = key && cmpModel(key) ? key : null;
  lbOpen(false);
  if (key === BY) return;
  const tok = ++_lbToken, main = document.querySelector('main');
  const stays = !key || cmpHas(key, _activeFile);
  document.body.classList.add('lb-swap'); main.classList.add('src-fade'); main.classList.toggle('ep-fade', !stays);
  const [list] = await Promise.all([key ? loadCmpList(key) : null, sleep(170)]);
  const done = () => { document.body.classList.remove('lb-swap'); void main.offsetHeight;
    requestAnimationFrame(() => main.classList.remove('src-fade', 'ep-fade')); };
  if (tok !== _lbToken) return;              // a later choice wins; only the latest one ever becomes the board's
  if (key && !list) { done(); return; }
  BY = key; BY_EPS = list;
  ISSUE_FILTER = null;
  renderLabelsBy();
  const has = d => railEps().some(e => datasetOf(e) === d);
  const ds = currentDataset && has(currentDataset) ? currentDataset : (firstDataset() || currentDataset);
  const open = _activeFile && railEps().find(e => e.file === _activeFile);
  const keep = open && datasetOf(open) === ds ? _activeFile : undefined;
  _lbSwapFile = keep || null;
  if (!opts.fromPop) {
    try { const u = new URL(location.href); if (BY) u.searchParams.set('by', BY); else u.searchParams.delete('by');
      history.replaceState(null, '', u); } catch (_) {}
  }
  await setDataset(ds, keep);
  if (tok === _lbToken) done();
}

// ---- the band over the open episode while it shows a comparison ----
const srcBar = document.getElementById('src-bar');
const sbNote = document.getElementById('sb-note'), sbBack = document.getElementById('sb-back');
// the band's height follows its content (eased by CSS): 0 while the episode shows the board's own labels
const sbRow = srcBar.querySelector('.sb-row');
function sbFit() { srcBar.style.height = (srcBar.classList.contains('on') ? sbRow.offsetHeight : 0) + 'px'; }
if (window.ResizeObserver) new ResizeObserver(sbFit).observe(sbRow);
// the header's top moves while the band opens or closes, so the rail's block is aligned again once it has
srcBar.addEventListener('transitionend', e => { if (e.target === srcBar && e.propertyName === 'height') alignHeads(); });
function renderSourceBar() {
  const m = SHOWN ? cmpModel(SHOWN) : null, ref = esc(refName());
  srcBar.classList.toggle('on', !!m);
  srcBar.classList.toggle('cmp', !!m);
  sbBack.tabIndex = m ? 0 : -1;
  // phones get the same statement in fewer words, so the band stays a band and not half the screen; the text stays
  // while the band folds away
  if (m) {
    sbBack.textContent = `Back to ${refName()}`;
    sbNote.innerHTML = `<span class="sb-long"><b>Comparison: labels by ${esc(cmpFullName(m.key))}.</b> These are `
      + `not ${ref}&rsquo;s labels. The Episode JSON download stays ${ref}&rsquo;s.</span><span class="sb-short">`
      + `<b>Comparison, not ${ref}&rsquo;s labels.</b> Downloads stay ${ref}&rsquo;s.</span>`;
  }
  sbFit();
}
sbBack.addEventListener('click', () => setLabeller(null));

// ---- the comparison view ----
const cmpView = document.getElementById('cmp-view');
const cmpTip = document.getElementById('cmp-tip');
let CMP_METRICS = null, CMP_RIG = 'all', _cmpBuilt = false;
async function setView(toCmp) {
  if (document.body.classList.contains('view-cmp') === toCmp) return;
  document.body.classList.add('view-fade');
  await sleep(180);
  document.body.classList.toggle('view-cmp', toCmp);
  renderCoverage(currentDataset, currentDataset ? railEps().filter(e => datasetOf(e) === currentDataset) : []);
  void document.body.offsetHeight;
  requestAnimationFrame(() => document.body.classList.remove('view-fade'));
}
async function showCompare(fromPop) {
  if (!CMP) return;
  if (!fromPop) {
    try { history.pushState(null, '', '?view=compare' + (BY ? '&by=' + encodeURIComponent(BY) : '')); } catch (_) {}
  }
  if (!CMP_METRICS) CMP_METRICS = await fetchJson(compareUrl('metrics'));
  if (!_cmpBuilt) buildCompare();
  await setView(true);
  cmpView.scrollTop = 0;
}
// back to the episodes: a dataset tab (ds), an episode link (file, with the labeller it asks for: by, null for the
// board's own), the view's own back button (nothing), or the browser's back button (fromPop: the URL already says
// where)
async function leaveCompare(ds, file, by, fromPop) {
  if (by !== undefined) await applyLabeller(by);
  renderLabelsBy();
  const target = file || (ds ? null : _activeFile);
  if (!fromPop) {
    const u = new URLSearchParams();
    if (target) u.set('ep', target); else if (ds) u.set('ds', ds);
    if (BY) u.set('by', BY);
    try { history.pushState(null, '', '?' + u.toString()); } catch (_) {}
  }
  await setView(false);
  const ep = target && railEps().find(e => e.file === target);
  if (ep) {
    if (file) { searchEl.value = ''; SEARCH = ''; searchWrap.classList.remove('has-q'); }
    if (currentDataset === datasetOf(ep) && !ISSUE_FILTER) renderRail(currentDataset, ep.file);
    else setDataset(datasetOf(ep), ep.file);
  } else {
    const first = firstDataset(ds || currentDataset);
    if (first) setDataset(first);
  }
}

// every chart mark carries its own tooltip (data-tip), on hover and on keyboard focus; the values are also on the
// page
function tipShow(el, x, y) {
  cmpTip.innerHTML = el.dataset.tip;
  const w = cmpTip.offsetWidth, h = cmpTip.offsetHeight;
  cmpTip.style.left = Math.max(8, Math.min(innerWidth - w - 8, x + 14)) + 'px';
  cmpTip.style.top = (y + h + 22 > innerHeight ? y - h - 12 : y + 16) + 'px';
  cmpTip.classList.add('show');
}
cmpView.addEventListener('pointermove', (e) => { const el = e.target.closest('[data-tip]'); if (el) tipShow(el,
  e.clientX, e.clientY); else cmpTip.classList.remove('show'); });
cmpView.addEventListener('pointerleave', () => cmpTip.classList.remove('show'));
cmpView.addEventListener('focusin', (e) => { const el = e.target.closest('[data-tip]'); if (el) {
  const r = el.getBoundingClientRect(); tipShow(el, r.left + r.width / 2, r.bottom - 10); } });
cmpView.addEventListener('focusout', () => cmpTip.classList.remove('show'));
cmpView.addEventListener('scroll', () => cmpTip.classList.remove('show'), {passive: true});

// numbers ease to their new value with the bars, from wherever they are
function tween(el, to, fmt) {
  const from = typeof el._v === 'number' ? el._v : to;
  el._v = to;
  if (el._raf) cancelAnimationFrame(el._raf);
  if (to == null || from == null || from === to) { el.textContent = to == null ? 'no data' : fmt(to); return; }
  const t0 = performance.now(), dur = 420;
  const step = (t) => {
    const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3);
    el.textContent = fmt(from + (to - from) * e);
    if (k < 1) el._raf = requestAnimationFrame(step);
  };
  el._raf = requestAnimationFrame(step);
}
const fPct = v => (100 * v).toFixed(v >= 0.995 || v === 0 ? 0 : 1) + '%';
const fNum = v => v >= 10 ? v.toFixed(1) : v.toFixed(2);
const fUsd = v => '$' + (v >= 1 ? v.toFixed(2) : v.toFixed(3));
const fSec = v => v >= 120 ? (v / 60).toFixed(1) + ' min' : Math.round(v) + ' s';
const fMin = m => m >= 90 ? (m / 60).toFixed(1) + ' hours'
  : Math.round(m) + (Math.round(m) === 1 ? ' minute' : ' minutes');
// models in the order every chart uses: the reference, then each model followed by its in-context run
function cmpOrder() {
  const ms = CMP_METRICS.models, out = ms.filter(m => m.reference);
  for (const m of ms.filter(m => !m.reference && !m.example)) {
    out.push(m);
    out.push(...ms.filter(x => x.example && x.base === m.key));
  }
  return out.concat(ms.filter(m => !out.includes(m)));
}
function cmpNameHtml(m) {
  if (m.example) { const b = CMP_METRICS.models.find(x => x.key === m.base); return `${esc(shownName((b
    || m).name))}<small>${esc(cap(icl()))}</small>`; }
  return esc(shownName(m.name));
}
// the same as text: an in-context run is its model and how it was prompted
function cmpNameText(m) {
  const b = m.example && CMP_METRICS.models.find(x => x.key === m.base);
  return b ? `${shownName(b.name)}, ${icl()}` : shownName(m.name);
}
const MAIN_CHARTS = [
  {id: 'parse', title: 'Responses that parse', fmt: fPct, max: () => 1,
   def: 'The share of responses that are valid JSON.',
   val: (s, k) => { const p = s.responses[k]; const n = p ? p.parsed + p.unparsed + p.cut_off : 0;
     return p && p.parse_share != null ? {v: p.parse_share, sub: `${p.parsed} of ${n}`,
       tip: `${p.parsed} parsed, ${p.unparsed} did not parse, ${p.cut_off} cut off at the output limit${p.no_response
         ? `; ${p.no_response} calls returned nothing` : ''}${p.pending ? `; ${p.pending} still to run` : ''}`} : null;
         },
   foot: s => ''},
  {id: 'events_per_min', title: 'Labelled events per minute', fmt: fNum, common: true},
  {id: 'key_events', title: 'Key events per episode', fmt: fNum, common: true,
   def: 'The milestones a person would mark to judge progress on the task.'},
  {id: 'subgoals', title: 'Subgoals per episode', fmt: fNum, common: true, colors: ['c-grn'],
   def: 'Key events the model marked as a completed subgoal.'},
  {id: 'data_issues', title: 'Data issues per episode', fmt: fNum, common: true, minor: 'data_issues_minor',
    colors: ['c-red', 'c-red-l'],
   keys: ['medium or high severity', 'low severity'], short: 'low',
   def: 'Faults in the recording, the scene or the label.'},
  {id: 'operator_mistakes', title: 'Operator mistakes per episode', fmt: fNum, common: true,
    minor: 'operator_mistakes_minor', colors: ['c-amb', 'c-amb-l'],
   keys: ['changed the outcome or high severity', 'minor'], short: 'minor'},
  {id: 'cost', title: 'Cost per episode', fmt: fUsd,
   val: (s, k) => { const p = s.responses[k]; return p && p.cost != null ? {v: p.cost, sub: `${p.cost_n} calls`} : null;
     },
   foot: s => ''},
  {id: 'latency', title: 'Time to annotate each episode', fmt: fSec,
   val: (s, k) => { const p = s.responses[k]; return p && p.latency != null ? {v: p.latency,
     sub: `${p.latency_n} calls`} : null; },
   foot: s => ''},
];
const ptsDelta = (a, b) => { const d = 100 * (b - a); return Math.abs(d) < 0.05 ? 'no change' : `${d > 0 ? '+'
  : '-'}${Math.abs(d).toFixed(1)} points`; };
const PAIR_CHARTS = [
  {id: 'parse_share', title: 'Responses that parse', fmt: fPct, max: 1, delta: (a, b) => ptsDelta(a, b)},
  {id: 'events_per_min', title: 'Labelled events per minute', fmt: fNum, ref: true},
  {id: 'key_events', title: 'Key events per episode', fmt: fNum, ref: true},
  {id: 'subgoals', title: 'Subgoals per episode', fmt: fNum, ref: true},
  {id: 'data_issues', title: 'Data issues per episode, medium or high severity', fmt: fNum, ref: true},
  {id: 'operator_mistakes', title: 'Operator mistakes per episode, minor ones left out', fmt: fNum, ref: true},
  {id: 'agree_outcome', title: 'Outcome agreement with {ref}', fmt: fPct, max: 1, delta: (a, b) => ptsDelta(a, b)},
  {id: 'agree_issues', title: 'Issue-type agreement with {ref}', fmt: fPct, max: 1, delta: (a, b) => ptsDelta(a, b)},
  {id: 'cost', title: 'Cost per episode', fmt: fUsd},
  {id: 'latency', title: 'Time to annotate each episode', fmt: fSec},
];
function buildCompare() {
  _cmpBuilt = true;
  const M = CMP_METRICS;
  if (!M || !M.summary) { cmpView.innerHTML = '<div class="cmpv"><p class="cmpv-sub">The comparison has no results yet.'
    + '</p></div>'; return; }
  const order = cmpOrder();
  const main = order.filter(m => M.main.includes(m.key));
  const others = M.models.filter(m => !m.reference && !m.example);
  const all = M.summary.all || {};
  const exN = new Set(M.episodes.filter(e => Object.keys(e.by).some(k => (M.models.find(m => m.key === k)
    || {}).example)).map(e => e.file)).size;
  const rigs = Object.keys(M.summary);
  const ref = refName(), refH = esc(ref);
  const andJoin = xs => xs.length > 1 ? xs.slice(0, -1).join(', ') + ' and ' + xs[xs.length - 1] : xs.join('');
  const serialJoin = xs => xs.length > 2 ? xs.slice(0, -1).join(', ') + ', and ' + xs[xs.length - 1] : andJoin(xs);
  // the footage the comparison covers, as its groups are named (teleoperated arms, UMI, human ego)
  const rigList = serialJoin(rigs.filter(r => r !== 'all').map(r => (M.rig_names[r] || r).toLowerCase()
    .replace(/^teleop$/, 'teleoperated arms').replace(/^umi$/, 'UMI')));
  // the episodes each model labelled with an Astra trace: the count most models share, then any model that differs
  const exBy = {};
  for (const m of M.models.filter(x => x.example)) {
    const n = M.episodes.filter(e => e.by && e.by[m.key]).length;
    (exBy[n] = exBy[n] || new Set()).add(m.name.split(',')[0]);
  }
  const exNs = Object.keys(exBy).map(Number).sort((a, b) => exBy[b].size - exBy[a].size || b - a);
  const exAlt = exNs.slice(1).map(n => `${n.toLocaleString()} for ${andJoin([...exBy[n]].map(esc))}`).join('; ');
  const exText = exNs.length ? `${exNs[0].toLocaleString()} of them${exAlt ? ` (${exAlt})` : ''}` : '';
  // each model with the reasoning effort it ran at (its run's own, else the pinned one), unless its name names one
  const effort = m => m.reasoning || BOARD.reasoning;
  // a qualified name ("GPT-6.1 Sol, high reasoning") reads as the model with its qualifier in brackets, like the rest
  const modelList = andJoin(others.map(m => { const [h, t] = nameParts(m.name);
    return esc(h) + (t ? ` (${esc(t)})` : effort(m) ? ` (${esc(effort(m))} reasoning)` : ''); }));
  const rigBtns = rigs.map(r => `<button type="button" role="radio" aria-checked="${r === CMP_RIG}" `
    + `data-rig="${r}">${esc(r === 'all' ? 'All footage' : M.rig_names[r] || r)}</button>`).join('');
  const card = (c, rowsHtml, wide) => `<div class="cc${wide ? ' cc-wide' : ''}" data-chart="${c.id}">`
    + `<h4>${esc(c.title)}</h4><p class="cc-def">${esc(c.def || '')}</p>`
    + `<div class="cc-key">${c.minor ? `<span><i class="${c.colors[0]}"></i>${esc(c.keys[0])}</span><span><i `
      + `class="${c.colors[1]}"></i>${esc(c.keys[1])}</span>` : ''}</div>`
    + `<div class="cc-rows">${rowsHtml}</div><p class="cc-foot"></p></div>`;
  const barRows = (c) => main.map(m => `<div class="br${m.reference ? ' ref' : ''}" data-k="${esc(m.key)}" `
    + `tabindex="0"><span class="br-name">${cmpNameHtml(m)}</span>`
    + `<span class="br-track"><span class="br-bar" style="width:0%">${(c.colors || ['c-ink']).map((col,
      i) => `<i class="${col} seg${i}" style="flex-grow:${i ? 0 : 1}"></i>`).join('')}</span></span>`
    + `<span class="br-val"><span class="bv">no data</span><small></small></span></div>`).join('');
  const mx = (id, title, def) => `<div class="cc cc-wide" data-mx="${id}"><h4>${esc(title)}</h4>${def ? `<p `
    + `class="cc-def">${esc(def)}</p>` : ''}<div class="mx-wrap"><table class="mx"><thead><tr><th></th>`
    + order.map(m => `<th class="${m.reference ? 'ref' : ''}">${cmpNameHtml(m)}</th>`).join('') + '</tr></thead><tbody>'
    + order.map(a => `<tr><th class="mx-rh${a.reference ? ' ref' : ''}">${cmpNameHtml(a)}</th>`
      + order.map(b => a.key === b.key ? '<td class="self"></td>'
      : `<td data-a="${esc(a.key)}" data-b="${esc(b.key)}" tabindex="0"></td>`).join('') + '</tr>').join('')
    + '</tbody></table></div><div class="mx-scale"><span>0%</span><i></i><span>100%</span></div><p class="cc-foot"></p>'
      + '</div>';
  const pairs = (M.paired.all || []);
  const pairCard = (c) => `<div class="cc" data-pair="${c.id}"><h4>${esc(c.title.replace('{ref}', ref))}</h4>`
    + pairs.map(p => { const m = M.models.find(x => x.key === p.base) || {};
      return `<div class="pr" data-k="${esc(p.base)}" tabindex="0"><span class="br-name">${esc(m.name
      ? shownName(m.name) : p.base)}</span>`
      + `<span class="pr-track"><span class="pr-ref" style="opacity:0"></span><span class="pr-line"></span><span `
        + `class="pr-dot without"></span><span class="pr-dot with"></span></span>`
      + `<span class="pr-val"><span class="pv"></span><small></small></span></div>`; }).join('') + '</div>';
  cmpView.innerHTML = `<div class="cmpv">
    <button type="button" class="cmpv-back" id="cmpv-back"><span aria-hidden="true">&larr;</span> Back to the
      episodes</button>
    <div class="cmpv-head"><h2>How the models compare</h2>
      <p>The comparison uses a ${(all.episodes || 0).toLocaleString()}-episode subset of the dashboard, meaning
        ${fMin(all.minutes || 0)} of footage spread across ${esc(rigList)}. ${modelList} each labelled these episodes and
        reported their data issues with the same harness as ${refH}.${exN ? ` On ${exText}, each
        labelled the episode a second time with in-context learning, its prompt including one complete ${refH} trace to
        show the density and reasoning expected.` : ''}</p></div>
    <div class="cmpv-bar"><span class="if-sev-seg" role="radiogroup" aria-label="Footage">${rigBtns}</span><span `
      + `class="cmpv-scope" id="cmpv-scope"></span></div>
    <h3 class="section">Each model on the same episodes <span class="count">(all medium reasoning unless otherwise specified)</span></h3>
    <div class="cmpv-grid cmpv-main">${MAIN_CHARTS.map(c => card(c, barRows(c))).join('')}</div>
    <h3 class="section">Agreement between models <span class="count">(all medium reasoning unless otherwise specified)</span></h3>
    <div class="cmpv-grid">${mx('outcome', 'Outcome agreement', 'The share of episodes on which two runs give the task '
      + 'the same outcome (success, success then undone, failure or unclear; a task partly done is a failure). Human ego '
      + 'sessions grade each '
      + 'task separately and have no single outcome, so they are left out.')}
      ${mx('issues', 'Issue-type agreement', '')}</div>
    ${pairs.length ? `<h3 class="section">${esc(cap(icl()))} <span class="count">(all medium reasoning unless otherwise specified)</span></h3>
    <p class="cmpv-sub" id="pr-sub"></p>
    <div class="pr-key"><span><i class="k-open"></i>without the trace</span><span><i class="k-fill"></i>with the `
      + `trace</span><span><i class="k-ref"></i>${refH} on the same episodes</span></div>
    <div class="cmpv-grid">${PAIR_CHARTS.map(pairCard).join('')}</div>` : ''}
    <h3 class="section">Episodes</h3>
    <div class="et-wrap" id="et-wrap"></div>
  </div>`;
  cmpView.querySelector('.cmpv-bar').addEventListener('click', (e) => {
    const b = e.target.closest('button[data-rig]');
    if (!b || b.dataset.rig === CMP_RIG) return;
    CMP_RIG = b.dataset.rig;
    cmpView.querySelectorAll('.cmpv-bar button').forEach(x => x.setAttribute('aria-checked',
      String(x.dataset.rig === CMP_RIG)));
    updateCompare();
  });
  cmpView.addEventListener('click', (e) => {
    if (e.target.closest('#cmpv-back')) { leaveCompare(); return; }
    const a = e.target.closest('a[data-file]');
    if (!a || e.metaKey || e.ctrlKey || e.shiftKey || e.button) return;
    e.preventDefault();
    leaveCompare(null, a.dataset.file, a.dataset.by || null);
  });
  updateCompare(true);
}
function updateCompare(first) {
  const M = CMP_METRICS, s = M.summary[CMP_RIG] || {};
  const order = cmpOrder();
  const nm = k => { const m = M.models.find(x => x.key === k); return m && m.name ? shownName(m.name) : k; };
  document.getElementById('cmpv-scope').textContent = `${(s.episodes || 0).toLocaleString()} episodes, ${fMin(s.minutes
    || 0)} of footage`;
  for (const c of MAIN_CHARTS) {
    const el = cmpView.querySelector(`[data-chart="${c.id}"]`);
    const vals = {};
    for (const k of M.main) {
      if (c.val) { vals[k] = c.val(s, k); continue; }
      const mv = (s.metrics || {})[k];
      if (!mv || mv[c.id] == null) { vals[k] = null; continue; }
      const minor = c.minor ? mv[c.minor] || 0 : 0;
      vals[k] = {v: mv[c.id], minor, sub: c.minor ? `+ ${fNum(minor)} ${c.short}` : '',
        tip: c.minor ? `${fNum(mv[c.id])} ${c.keys[0]} and ${fNum(minor)} ${c.keys[1]} per episode` : ''};
    }
    const max = c.max ? c.max() : Math.max(1e-9, ...Object.values(vals).filter(Boolean).map(x => x.v + (x.minor || 0)));
    el.querySelectorAll('.br').forEach(row => {
      const x = vals[row.dataset.k], bar = row.querySelector('.br-bar'), segs = bar.querySelectorAll('i');
      const tot = x ? x.v + (x.minor || 0) : 0;
      bar.style.width = (x ? 100 * tot / max : 0).toFixed(2) + '%';
      bar.style.opacity = x ? '1' : '0';
      if (segs[0]) segs[0].style.flexGrow = x ? String(x.v || 0.0001) : '1';
      if (segs[1]) segs[1].style.flexGrow = x ? String(x.minor || 0) : '0';
      tween(row.querySelector('.bv'), x ? x.v : null, c.fmt);
      row.querySelector('small').textContent = x ? x.sub || '' : '';
      row.dataset.tip = x ? `<b>${esc(c.fmt(x.v))}</b>${esc(nm(row.dataset.k))}${x.tip ? '<br>' + esc(x.tip) : ''}`
        : `<b>no data</b>${esc(nm(row.dataset.k))}`;
    });
    el.querySelector('.cc-foot').textContent = c.common
      ? (s.common ? '' : 'No episode here was parsed by every model yet.') : c.foot(s);
  }
  for (const id of ['outcome', 'issues']) {
    const card = cmpView.querySelector(`[data-mx="${id}"]`);
    let any = false;
    card.querySelectorAll('td[data-a]').forEach(td => {
      const g = ((s.agreement || {})[td.dataset.a] || {})[td.dataset.b] || {};
      const v = g[id], n = id === 'outcome' ? g.outcome_n : g.issues_n;
      if (v == null) {
        td.className = 'na'; td.style.background = ''; td.style.color = '';
        td.innerHTML = n ? 'none<small>n ' + n + '</small>' : 'no episodes';
        td.dataset.tip = `<b>${n ? 'nothing to compare' : 'no episodes'}</b>${esc(nm(td.dataset.a))} `
          + `and ${esc(nm(td.dataset.b))}${n ? `: neither reported a problem on the ${n} episodes both parsed` : ''}`;
        return;
      }
      any = true;
      td.className = '';
      td.style.background = `rgba(28,28,26,${(0.05 + 0.81 * v).toFixed(3)})`;
      td.style.color = v > 0.5 ? '#f3f2ec' : 'var(--fg)';
      td.innerHTML = `${fPct(v)}<small>n ${n}</small>`;
      td.dataset.tip = `<b>${fPct(v)}</b>${esc(nm(td.dataset.a))} and ${esc(nm(td.dataset.b))}, ${id === 'outcome'
        ? `the same outcome on ${Math.round(v * n)} of ${n} episodes`
        : `${g.issues_shared} of ${g.issues_union} kinds of problem in common over ${n} episodes`}`;
    });
    card.querySelector('.cc-foot').textContent = any ? '' : (id === 'outcome' && CMP_RIG === 'head_camera'
      ? 'Human ego sessions have no single outcome to compare.' : 'No pair of runs has an episode in common here '
        + 'yet.');
  }
  const pairs = (M.paired || {})[CMP_RIG] || [];
  const sub = document.getElementById('pr-sub');
  if (sub) sub.textContent = pairs.length
    ? `Each model with and without one complete ${refName()} trace in its prompt.`
    : 'No in-context run has episodes here yet.';
  for (const c of PAIR_CHARTS) {
    const el = cmpView.querySelector(`[data-pair="${c.id}"]`);
    if (!el) continue;
    const vs = [];
    for (const p of pairs) { vs.push(p.without_values[c.id], p.with_values[c.id]); if (c.ref
      && p.reference) vs.push(p.reference[c.id]); }
    const max = c.max || Math.max(1e-9, ...vs.filter(v => v != null)) * 1.08;
    el.querySelectorAll('.pr').forEach(row => {
      const p = pairs.find(x => x.base === row.dataset.k);
      const a = p ? p.without_values[c.id] : null, b = p ? p.with_values[c.id] : null, r = p && c.ref && p.reference
        ? p.reference[c.id] : null;
      const X = v => (100 * Math.max(0, Math.min(1, v / max))).toFixed(2) + '%';
      const dw = row.querySelector('.pr-dot.without'), df = row.querySelector('.pr-dot.with'),
        ln = row.querySelector('.pr-line'), rf = row.querySelector('.pr-ref');
      const ok = a != null && b != null;
      dw.style.opacity = df.style.opacity = ln.style.opacity = ok ? '1' : '0';
      if (ok) {
        dw.style.left = X(a); df.style.left = X(b);
        ln.style.left = X(Math.min(a, b)); ln.style.width = `calc(${X(Math.max(a, b))} - ${X(Math.min(a, b))})`;
      }
      rf.style.opacity = r != null ? '0.9' : '0';
      if (r != null) rf.style.left = X(r);
      row.querySelector('.pv').textContent = ok ? `${c.fmt(a)} → ${c.fmt(b)}` : 'no data';
      const dd = ok ? b - a : 0;
      row.querySelector('small').textContent = !ok ? '' : c.delta ? c.delta(a, b)
        : c.fmt(Math.abs(dd)) === c.fmt(0) ? 'no change' : `${dd > 0 ? '+' : '-'}${c.fmt(Math.abs(dd))}`;
      row.style.display = p ? '' : 'none';
      row.dataset.tip = ok ? `<b>${esc(c.fmt(a))} → ${esc(c.fmt(b))}</b>${esc(nm(p.base))}, without and with the `
        + `trace${r != null ? `<br>${esc(refName())} on the same ${p.reference_n} episodes: ${esc(c.fmt(r))}` : ''}`
        : `<b>no data</b>`;
    });
  }
  renderEpisodeTable(order, first);
}
async function renderEpisodeTable(order, first) {
  const M = CMP_METRICS, wrap = document.getElementById('et-wrap');
  const RIG_ORDER = ['teleop', 'handheld', 'head_camera'];
  const dsRank = d => { const i = DS_ORDER.indexOf(d); return i < 0 ? 99 : i; };
  const eps = M.episodes.filter(e => CMP_RIG === 'all' || e.rig === CMP_RIG).slice()
    .sort((a, b) => (RIG_ORDER.indexOf(a.rig) - RIG_ORDER.indexOf(b.rig)) || (dsRank(a.dataset) - dsRank(b.dataset))
      || String(a.episode_id).localeCompare(String(b.episode_id)));
  const cell = (e, m) => {
    const r = e.by[m.key];
    if (!r || !e.file) return '<td></td>';
    const by = !m.reference && cmpHas(m.key, e.file) ? m.key : '';
    let cls, txt;
    if (r.status === 'parsed') {
      if (r.tasks) { cls = 'tasks'; txt = `${r.tasks[1]} of ${r.tasks[0]} tasks`; }
      else { cls = r.outcome || 'unclear'; txt = (r.outcome || 'no outcome').replace(/_/g, ' '); }
    } else if (r.status === 'pending') return '<td><span class="et-none">not yet</span></td>';
    else { cls = 'fail'; txt = ST_WORDS[r.status] || r.status; }
    const href = `?ep=${encodeURIComponent(e.file)}${by ? '&by=' + encodeURIComponent(by) : ''}`;
    return `<td class="${m.reference ? 'ref-col' : ''}"><a class="et-o ${esc(cls)}" href="${href}" `
      + `data-file="${esc(e.file)}" data-by="${esc(by)}" title="open with ${esc((by ? cmpNameText(m)
      : refName()) + '’s labels')}">${esc(txt)}</a></td>`;
  };
  const html = `<table class="et"><thead><tr><th>Episode</th><th>Dataset</th><th>Length</th>`
    + order.map(m => `<th>${cmpNameHtml(m)}</th>`).join('') + '</tr></thead><tbody>'
    + eps.map(e => `<tr><td class="et-id">${e.file ? `<a href="?ep=${encodeURIComponent(e.file)}" `
      + `data-file="${esc(e.file)}">${esc(e.episode_id)}</a>` : esc(e.episode_id)}</td>`
      + `<td class="et-ds">${esc(DS_SHORT[e.dataset] || dsLabel(e.dataset
        || ''))}</td><td class="et-len">${e.seconds != null ? fmtT(e.seconds) : ''}</td>`
      + order.map(m => cell(e, m)).join('') + '</tr>').join('') + '</tbody></table>';
  if (first) { wrap.innerHTML = html; return; }
  // the rows change with the rig: the table eases out and back in, keeping its place on the page
  const tok = (wrap._tok = (wrap._tok || 0) + 1);
  wrap.style.transition = 'opacity 150ms ease';
  wrap.style.opacity = '0';
  await sleep(150);
  if (tok !== wrap._tok) return;
  wrap.innerHTML = html;
  void wrap.offsetHeight;
  wrap.style.opacity = '1';
}

// If the tab is restored from the back/forward cache (reopened closed tab, or
// browser session restore), the in-memory episode list can be stale (e.g. from a
// run before newer datasets were added), which shows only the old datasets and
// empty categories. Force a fresh load in that case.
window.addEventListener('pageshow', (e) => { if (e.persisted) location.reload(); });

loadEpisodes();
</script>
</body></html>
"""


# the title bar, which a site header (--header) replaces
PAGE_HEAD = ('<header class="page-head"><h1>__PAGE_TITLE__</h1>'
             '<span class="ph-board">__BOARD_NAME__</span></header>')


def _js(v) -> str:
    return json.dumps(v, separators=(",", ":")).replace("<", "\\u003c")


def read_header(path: Path | None) -> str | None:
    """A site header for the page (--header): the file's HTML, or None when no file is given."""
    return Path(path).read_text() if path else None


def render_index(title: str, board: dict, name: str = "", header: str | None = None) -> str:
    """The page with its title, the board's name and its data source filled in. `board` is {"mode": "api"} for this
    server, or {"mode": "static", "data": <base>, "media": <base>} for a static build (board/static.py); "compare":
    true, "hands": true, "sensors": true and "keypoints": true tell the page the board has other models' labels, hand
    pose drawings, sensors files or hand keypoint downloads to ask for, and "models" ({model id: name}) shows a model
    under a site's own name in place of its name in configs/models.json. `header` is a site's own header, for a board
    served as part of a site: its markup takes the place of the page's title bar and its <style> blocks go into the
    page's head (a header of another height sets --header-h, which the page's sticky offsets read)."""
    cfg = {**board, "models": {**model_names(), **(board.get("models") or {})}, "reasoning": reasoning_effort()}
    page = INDEX_HTML
    if header is not None:
        styles = "".join(re.findall(r"<style>.*?</style>", header, re.S))
        markup = re.sub(r"<style>.*?</style>", "", header, flags=re.S).strip()
        page = page.replace(PAGE_HEAD, markup, 1).replace("</head>", styles + "\n</head>", 1)
    return (page.replace("__PAGE_TITLE__", html.escape(title)).replace("__BOARD_NAME__", html.escape(name))
            .replace("__BOARD_CONFIG__", _js(cfg)).replace("__TAG_NAMES__", _js(load_tag_names()))
            .replace("__FAMILIES__", _js({**FAMILIES.catalog(), **capture_catalog()})))


class Handler(http.server.BaseHTTPRequestHandler):
    # keep-alive: a page asks for its labels, posters and three videos at once, and a proxy in front (or a browser
    # that talks to the board directly) reuses one connection for them instead of opening one per request; an idle
    # connection is closed after `timeout` seconds
    protocol_version = "HTTP/1.1"
    timeout = 30

    def log_message(self, *a, **kw):
        pass

    def _send(self, code, body, ctype="application/json", gzipped: bytes | None = None):
        """gzipped: the body already compressed (list_json), sent instead of compressing it again."""
        if isinstance(body, (dict, list)):
            body = json.dumps(body)
        if isinstance(body, str):
            body = body.encode()
        # compress large JSON and HTML: a large board's episode list is megabytes, about ten times smaller
        # gzipped, and the page waits for it before its first paint
        gz = len(body) > 16384 and "gzip" in (self.headers.get("Accept-Encoding") or "")
        if gz:
            body = gzipped if gzipped is not None else gzip.compress(body, compresslevel=5)
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        if gz:
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, p: Path, ctype: str, download: str | None = None):
        """Serve a file with proper HTTP Range support so the browser
        can stream/seek without downloading the whole thing first; download names the saved file."""
        try:
            st = p.stat()
        except FileNotFoundError:
            self._send(404, {"error": "not found"})
            return
        size = st.st_size

        rng = self.headers.get("Range") or self.headers.get("range")
        start = 0
        end = size - 1
        partial = False
        if rng and rng.startswith("bytes="):
            try:
                spec = rng[len("bytes="):].strip().split(",")[0]
                s, e = spec.split("-", 1)
                if s.strip():
                    start = int(s)
                    end = int(e) if e.strip() else size - 1
                else:
                    # suffix range: bytes=-N -> last N bytes
                    n = int(e)
                    start = max(0, size - n)
                    end = size - 1
                if start < 0 or start >= size or end >= size or start > end:
                    raise ValueError("range out of bounds")
                partial = True
            except Exception:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

        length = end - start + 1
        try:
            f = open(p, "rb")
        except FileNotFoundError:
            self._send(404, {"error": "not found"})
            return

        try:
            if partial:
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "public, max-age=3600")
            if download:
                self.send_header("Content-Disposition", f'attachment; filename="{download}"')
            self.end_headers()
            f.seek(start)
            remaining = length
            chunk = 64 * 1024
            while remaining > 0:
                buf = f.read(min(chunk, remaining))
                if not buf:
                    break
                try:
                    self.wfile.write(buf)
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True
                    break
                remaining -= len(buf)
        finally:
            f.close()

    def do_POST(self):
        """POST /api/export {"files": [...]} -> the listed episode files as JSON Lines, one per line,
        exactly as stored (annotation, dataset checks, run provenance)."""
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/export":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            files = json.loads(self.rfile.read(n) or b"{}").get("files") or []
        except (ValueError, json.JSONDecodeError):
            self._send(400, {"error": "bad request"})
            return
        lines = []
        for fname in files:
            p = HERE / str(fname)
            if not _under(HERE, p) or not p.is_file() or p.suffix != ".json":
                self._send(404, {"error": f"no such file: {fname}"})
                return
            lines.append(json.dumps(public_label(json.loads(p.read_text())), separators=(",", ":")))
        body = ("\n".join(lines) + "\n").encode()
        gz = "gzip" in (self.headers.get("Accept-Encoding") or "")
        if gz:
            body = gzip.compress(body, compresslevel=5)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Content-Disposition", 'attachment; filename="episodes.jsonl"')
        if gz:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            # the page asks for other models' labels, hand pose files and keypoint downloads only when this board has
            # them (no request that can only fail)
            cfg = {"mode": "api", "compare": (COMPARE_DIR / "index.json").is_file(), "hands": HANDS_DIR.is_dir(),
                   "sensors": (SENSORS_DIR / "index.json").is_file(),
                   "footage": FFMPEG is not None,
                   "keypoints": (KEYPOINTS_DIR / "index.json").is_file(), "labels_license": labels_license(HERE.parent)}
            self._send(200, render_index(PAGE_TITLE, cfg, BOARD_NAME, HEADER), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/episodes":
            raw, gz = list_json()
            self._send(200, raw, gzipped=gz)
            return
        if parsed.path == "/api/episode":
            q = urllib.parse.parse_qs(parsed.query)
            fname = (q.get("file") or [""])[0]
            p = HERE / fname
            # containment: never let ?file=../.. escape the data dir
            if not _under(HERE, p) or not p.is_file() or p.suffix != ".json":
                self._send(404, {"error": "no such file"})
                return
            if (q.get("download") or [""])[0] == "1":
                body = json.dumps(public_label(json.loads(p.read_text())), separators=(",", ":")).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Disposition", f'attachment; filename="{p.name}"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self._send(200, json.dumps(episode_view(json.loads(p.read_text()))), "application/json")
            return
        if parsed.path.startswith("/api/compare/"):
            # other models' labels (board/build.py compare/): kept beside the board's own and never in its lists
            what = parsed.path[len("/api/compare/"):]
            if what in ("index", "metrics"):
                p = COMPARE_DIR / f"{what}.json"
                if not p.is_file():
                    self._send(404, {"error": "no comparison on this dashboard"})
                    return
                self._send(200, p.read_bytes().decode(), "application/json")
                return
            if what == "list":
                # one model's rail records, the same records the board's own list is made of
                key = (urllib.parse.parse_qs(parsed.query).get("key") or [""])[0]
                d = COMPARE_DIR / key
                if not key or "/" in key or key.startswith(".") or not _under(COMPARE_DIR, d) or not d.is_dir():
                    self._send(404, {"error": "no such comparison"})
                    return
                raw, gz = list_json(d)
                self._send(200, raw, gzipped=gz)
                return
            if what == "episode":
                q = urllib.parse.parse_qs(parsed.query)
                key, fname = (q.get("key") or [""])[0], (q.get("file") or [""])[0]
                p = COMPARE_DIR / key / fname
                if not key or "/" in key or not _under(COMPARE_DIR, p) or not p.is_file() or p.suffix != ".json":
                    self._send(404, {"error": "no such comparison"})
                    return
                self._send(200, json.dumps(episode_view(json.loads(p.read_text()))), "application/json")
                return
            self._send(404, {"error": "unknown path"})
            return
        if parsed.path == "/api/hands":
            # one episode's hand pose (board/hands.py); read only by the page's overlay, never listed or exported
            fname = (urllib.parse.parse_qs(parsed.query).get("file") or [""])[0]
            p = HANDS_DIR / fname
            if not fname or "/" in fname or not _under(HANDS_DIR, p) or not p.is_file() or p.suffix != ".json":
                self._send(404, {"error": "no hand pose for this episode"})
                return
            self._send(200, p.read_bytes(), "application/json")
            return
        if parsed.path == "/api/sensors":
            # one episode's other signals and depth (board/sensors.py), or index.json for the list; read only by the
            # page's sensors panel and depth switches, never listed or exported
            fname = (urllib.parse.parse_qs(parsed.query).get("file") or [""])[0]
            p = SENSORS_DIR / fname
            if not fname or "/" in fname or not _under(SENSORS_DIR, p) or not p.is_file() or p.suffix != ".json":
                self._send(404, {"error": "no sensors for this episode"})
                return
            self._send(200, p.read_bytes(), "application/json")
            return
        if parsed.path == "/api/keypoints":
            # one head-camera episode's hand keypoints, a download of their own (never in the labels or their
            # export); index.json lists the episodes that have them
            q = urllib.parse.parse_qs(parsed.query)
            fname = (q.get("file") or [""])[0]
            p = KEYPOINTS_DIR / fname
            if not fname or "/" in fname or not _under(KEYPOINTS_DIR, p) or not p.is_file() or p.suffix != ".json":
                self._send(404, {"error": "no hand keypoints for this episode"})
                return
            if (q.get("download") or [""])[0] == "1" and fname != "index.json":
                body = p.read_bytes()
                gz = "gzip" in (self.headers.get("Accept-Encoding") or "")
                if gz:
                    body = gzip.compress(body, compresslevel=5)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Disposition", f'attachment; filename="{p.stem}.hand_keypoints.json"')
                if gz:
                    self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self._send(200, p.read_bytes(), "application/json")
            return
        if parsed.path == "/api/video":
            q = urllib.parse.parse_qs(parsed.query)
            eid, cam = (q.get("id") or [""])[0], (q.get("cam") or ["exo"])[0]
            mp4 = clip_path(MP4_DIR, eid, cam)
            if not _under(MP4_DIR, mp4) or not mp4.exists():
                self._send(404, {"error": "video not found"})
                return
            # download=1: one camera's clip saved as a file of its own, named after its camera, so an extra camera never
            # takes the main camera's name
            own = cam in ("left", "right") or EXTRA_CAM.fullmatch(cam) or DEPTH_CAM.fullmatch(cam)
            name = f"{eid}_{cam if own else 'main'}.mp4" \
                if (q.get("download") or [""])[0] == "1" \
                else None
            self._send_file(mp4, "video/mp4", name)
            return
        if parsed.path == "/api/footage":
            # the episode's cameras in one video (footage), whole or from t0 to t1 s; prepare=1 makes it and answers
            # with its size, so the page can say it is being made before the download starts
            q = urllib.parse.parse_qs(parsed.query)
            eid = (q.get("id") or [""])[0]
            try:
                t0 = float((q.get("t0") or ["0"])[0] or 0)
                t1 = float(q["t1"][0]) if (q.get("t1") or [""])[0] else None
            except ValueError:
                self._send(400, {"error": "t0 and t1 are seconds"})
                return
            if not eid or "/" in eid or eid.startswith(".") or not _under(MP4_DIR, MP4_DIR / f"{eid}.mp4"):
                self._send(404, {"error": "no such episode"})
                return
            try:
                got = footage(eid, t0, t1)
            except (RuntimeError, OSError, subprocess.SubprocessError, ValueError, KeyError) as e:
                self._send(500, {"error": str(e)[:400]})
                return
            if got is None:
                self._send(404, {"error": "no video for this episode and span"})
                return
            out, name = got
            if (q.get("prepare") or [""])[0] == "1":
                self._send(200, {"bytes": out.stat().st_size, "name": name})
                return
            self._send_file(out, "video/mp4", name)
            return
        if parsed.path == "/api/frame":
            q = urllib.parse.parse_qs(parsed.query)
            mp4 = clip_path(MP4_DIR, (q.get("id") or [""])[0], (q.get("cam") or ["exo"])[0])
            try:
                t = float((q.get("t") or ["0"])[0])
            except ValueError:
                t = 0.0
            try:
                w = int((q.get("w") or ["640"])[0])
            except ValueError:
                w = 640
            jpg = extract_frame(mp4, t, w) if _under(MP4_DIR, mp4) else None
            if not jpg:
                self._send(404, {"error": "frame not available"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            try:
                self.wfile.write(jpg)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        self._send(404, {"error": "unknown path"})


def main(argv=None) -> int:
    global HERE, MP4_DIR, COMPARE_DIR, HANDS_DIR, KEYPOINTS_DIR, SENSORS_DIR, FOOTAGE_DIR, PORT, PAGE_TITLE, BOARD_NAME
    global HEADER
    ap = argparse.ArgumentParser(prog="python -m board serve", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", type=Path, required=True, help="a board folder (its qa/ holds the episode files)")
    ap.add_argument("--clips", type=Path, required=True, help="the clips folder board clips wrote")
    ap.add_argument("--port", type=int, default=PORT, help=f"the port to listen on (default {PORT})")
    ap.add_argument("--title", default=PAGE_TITLE, help=f"the page title, in the header and the browser tab "
                                                        f"(default {PAGE_TITLE!r})")
    ap.add_argument("--header", type=Path, help="an HTML file with a site's own header, shown in place of the page's "
                                                "title bar (its <style> blocks go into the page's head)")
    a = ap.parse_args(argv)
    HERE = (a.board / "qa").resolve()
    MP4_DIR = a.clips.resolve()
    COMPARE_DIR = (a.board / "compare").resolve()
    HANDS_DIR = (a.board / "hands").resolve()
    KEYPOINTS_DIR = (a.board / "hand_keypoints").resolve()
    SENSORS_DIR = (a.board / "sensors").resolve()
    FOOTAGE_DIR = Path(os.environ.get("BOARD_FOOTAGE_DIR") or (a.board / "footage")).resolve()
    PORT, PAGE_TITLE, BOARD_NAME, HEADER = a.port, a.title, board_name(a.board), read_header(a.header)
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    # socketserver's default listen backlog is 5: a burst of visitors (or a proxy opening many connections at once)
    # overflows it and those connections wait on SYN retries for up to a minute
    socketserver.ThreadingTCPServer.request_queue_size = 128
    socketserver.ThreadingTCPServer.daemon_threads = True
    with socketserver.ThreadingTCPServer(("", PORT), Handler) as httpd:
        print(f"board at http://localhost:{PORT} ({len(list_episodes())} episodes from {HERE}, clips from {MP4_DIR})",
              flush=True)
        httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
