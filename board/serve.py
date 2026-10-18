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
    BOARD_FRAME_DIR            a bounded frame cache reused after a server restart
    BOARD_FRAME_DISK_BYTES     frame disk budget (default 512 MiB)
    BOARD_FRAME_DISK_FILES     maximum cached frame files (default 4000)
    BOARD_FOOTAGE_DISK_BYTES   footage disk budget (default 2 GiB)
    BOARD_FOOTAGE_DISK_FILES   maximum cached footage files (default 128)
    BOARD_FOOTAGE_DIR          where the videos made to download are kept (default BOARD/footage)
    BOARD_FOOTAGE_CONCURRENCY  videos made at once and concurrent source probes (default 1)
    BOARD_FOOTAGE_THREADS      threads each is made on (default 2)
    BOARD_FOOTAGE_STALL_TIMEOUT seconds without advancing media or output bytes (default 300)

board/static.py renders the same page with a static data source, so a CDN can serve the board with no server.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import http.server
import json
import logging
import math
import os
import re
import shutil
import socketserver
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager, ExitStack
from functools import lru_cache, wraps
import urllib.parse
import uuid
from collections import OrderedDict
from pathlib import Path

from board.families import Families
from board.public_export import project as public_json
from board.to_board import dumps, finite
from compare.metrics import model_names, reasoning_effort
from label.atomic import write_atomic


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
# BOARD_FRAME_DIR retains a bounded disk cache. File metadata invalidates replaced source clips.
_FRAME_DIR = Path(os.environ["BOARD_FRAME_DIR"]) if os.environ.get("BOARD_FRAME_DIR") else None


_CLIP_TIMES: dict = {}
_CLIP_HALFWAY: dict = {}


def _clip_times(mp4: Path) -> list:
    """The clip's frame times on the episode clock (sorted presentation times), read once per clip
    version; [] when they cannot be read."""
    try:
        st = mp4.stat()
        key = (str(mp4), st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino)
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
    if len(_CLIP_TIMES) > 2048:
        _CLIP_TIMES.clear()
    _CLIP_TIMES[key] = ts
    return ts


def _shown_from_halfway(mp4: Path) -> bool:
    """Whether the clip's frames come on screen halfway after the capture before each (board/clips.py retime, which
    sets HALFWAY_TAG as the clip's comment), so the frame a time shows is the one on screen then."""
    from board.clips import HALFWAY_TAG
    try:
        st = mp4.stat()
        key = (str(mp4), st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino)
    except OSError:
        return False
    if key in _CLIP_HALFWAY:
        return _CLIP_HALFWAY[key]
    probe = Path(FFMPEG).with_name("ffprobe") if FFMPEG else None
    try:
        r = subprocess.run([str(probe) if probe and probe.exists() else "ffprobe", "-v", "error", "-show_entries",
                            "format_tags=comment", "-of", "default=nw=1:nk=1", str(mp4)],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    hit = r.stdout.strip() == HALFWAY_TAG
    if len(_CLIP_HALFWAY) > 2048:
        _CLIP_HALFWAY.clear()
    _CLIP_HALFWAY[key] = hit
    return hit


class CacheBusy(RuntimeError):
    pass


def _bounded_work(semaphore):
    def decorate(fn):
        @wraps(fn)
        def run(*args, **kwargs):
            if not semaphore.acquire(blocking=kwargs.pop('_wait', False)):
                raise CacheBusy('Media preparation is busy; try again shortly')
            try:
                return fn(*args, **kwargs)
            finally:
                semaphore.release()
        return run
    return decorate


_DISK_CACHE_LOCK = threading.Lock()
_FRAME_DISK_BYTES = int(os.environ.get('BOARD_FRAME_DISK_BYTES', 512 * 1024**2))
_FRAME_DISK_FILES = int(os.environ.get('BOARD_FRAME_DISK_FILES', 4000))
_FOOTAGE_DISK_BYTES = int(os.environ.get('BOARD_FOOTAGE_DISK_BYTES', 2 * 1024**3))
_FOOTAGE_DISK_FILES = int(os.environ.get('BOARD_FOOTAGE_DISK_FILES', 128))


def _trim_disk_cache(folder: Path, suffix: str, *, max_bytes: int, max_files: int, protect=None):
    with _DISK_CACHE_LOCK:
        entries = []
        temporary = re.compile(r'(?:[a-f0-9]{40}\.[0-9]+\.[0-9]+\.part|'
                               r'[a-f0-9]{20}\.[0-9]+\.[0-9]+\.part(?:\.len)?\.mp4)')
        for path in folder.glob('*.part*'):
            if not temporary.fullmatch(path.name):
                continue
            try:
                if path.stat().st_mtime < time.time() - 600:
                    path.unlink(missing_ok=True)
            except OSError:
                continue
        length = {'.jpg': 40, '.mp4': 20}.get(suffix)
        if length is None:
            raise ValueError('Unknown media cache suffix')
        owned = re.compile(r'[a-f0-9]{' + str(length) + '}' + re.escape(suffix))
        for path in folder.glob('*' + suffix):
            if not owned.fullmatch(path.name):
                continue
            try:
                st = path.stat()
                entries.append((st.st_mtime_ns, path, st.st_size))
            except OSError:
                continue
        total, count = sum(x[2] for x in entries), len(entries)
        for _, path, size in sorted(entries):
            if total <= max_bytes and count <= max_files:
                break
            if path == protect:
                continue
            try:
                path.unlink(missing_ok=True)
                total -= size
                count -= 1
            except OSError:
                continue


_FRAME_WORK_SEM = threading.BoundedSemaphore(max(1, int(os.environ.get('BOARD_FFMPEG_CONCURRENCY', 4))))


def _cached_frame(key, disk: Path | None) -> bytes | None:
    with _FRAME_LOCK:
        hit = _FRAME_CACHE.get(key)
        if hit is not None:
            _FRAME_CACHE.move_to_end(key)
            return hit
    if disk is not None:
        try:
            out = disk.read_bytes()
            _remember_frame(key, out)
            return out
        except OSError:
            pass
    return None


def extract_frame(mp4: Path, t: float, max_w: int = 640, *, _wait=False) -> bytes | None:
    """Serve cached JPEGs immediately; admit only bounded uncached extraction work."""
    if not math.isfinite(t) or not 0 <= t <= 604800 or not 1 <= max_w <= 1920:
        raise ValueError('Frame time must be between 0 and 604800 seconds, and width between 1 and 1920')
    try:
        st = mp4.stat()
        key = (str(mp4), st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino, float(t), int(max_w))
    except OSError:
        return None
    disk = _FRAME_DIR / (hashlib.sha1(repr(key).encode()).hexdigest() + '.jpg') if _FRAME_DIR else None
    hit = _cached_frame(key, disk)
    if hit is not None:
        return hit
    if not FFMPEG:
        return None
    return _extract_uncached_frame(mp4, t, max_w, key, disk, _wait=_wait)


@_bounded_work(_FRAME_WORK_SEM)
def _extract_uncached_frame(mp4, t, max_w, key, disk):
    hit = _cached_frame(key, disk)
    if hit is not None:
        return hit
    # Clips hold exactly the episode's own frames, their first at the clip's start. The frame shown is the clip's
    # own frame nearest t, from its frame times (any rate, variable or not): for a clip timed by capture times, whose
    # frames come on halfway after the capture before each, the one on screen at t; for any other, the one whose time
    # is nearest t. The seek lands half a gap before it (an input seek counts from the clip's start), so a 4-decimal
    # seek can never round past it; a time at or past the clip's end (a goal frame on the last instant) is its last
    # frame, and a frame that cannot be cut falls back to the one before.
    rel = _clip_times(mp4)
    if rel:
        import bisect
        if _shown_from_halfway(mp4):
            i = max(0, bisect.bisect_right(rel, max(0.0, t)) - 1)
        else:
            i = bisect.bisect_left(rel, max(0.0, t))
            i = min(range(max(0, i - 1), min(len(rel), i + 1)), key=lambda j: (abs(rel[j] - t), j))

        def half_gap(j):
            gaps = [g for g in ([rel[j] - rel[j - 1]] if j > 0 else []) + ([rel[j + 1] - rel[j]] if j + 1 < len(rel)
                                                                           else []) if g > 0]
            return 0.5 * (min(gaps) if gaps else 1 / 30.0)
        tries = [rel[j] - rel[0] - half_gap(j) for j in (i, i - 1, i - 2) if j >= 0]
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
                if len(out) > _FRAME_DISK_BYTES:
                    return out
                disk.parent.mkdir(parents=True, exist_ok=True)
                tmp = disk.with_suffix(f".{os.getpid()}.{threading.get_ident()}.part")
                tmp.write_bytes(out)
                tmp.replace(disk)
                _trim_disk_cache(disk.parent, '.jpg', max_bytes=_FRAME_DISK_BYTES, max_files=_FRAME_DISK_FILES,
                                 protect=disk)
            except OSError:
                pass                      # the disk copy is only a cache
            finally:
                if 'tmp' in locals():
                    tmp.unlink(missing_ok=True)
    return out


def _remember_frame(key, jpg: bytes) -> None:
    with _FRAME_LOCK:
        if len(jpg) > 16 * 1024**2:
            return
        _FRAME_CACHE[key] = jpg
        _FRAME_CACHE.move_to_end(key)
        while len(_FRAME_CACHE) > max(0, _FRAME_CACHE_MAX) or sum(map(len, _FRAME_CACHE.values())) > 128 * 1024**2:
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
FOOTAGE_TAG = f"footage-v4-h264-crf20-veryfast-{FOOTAGE_FPS}fps-gap{FOOTAGE_GAP}"   # a new recipe makes new files
_FOOTAGE_SEM = threading.BoundedSemaphore(max(1, int(os.environ.get("BOARD_FOOTAGE_CONCURRENCY", 1))))
_FOOTAGE_PROBE_SEM = threading.BoundedSemaphore(max(1, int(os.environ.get('BOARD_FOOTAGE_CONCURRENCY', 1))))
_PROBE_ACTIVE = set()
_PROBE_ACTIVE_LOCK = threading.Lock()
_FOOTAGE_READY_LOCK = threading.Lock()
_CONTENT_PROBES = OrderedDict()
_CONTENT_PROBE_LOCK = threading.Lock()
_FOOTAGE_TICKET_SECONDS = 3600
_MEDIA_READERS = threading.local()
_FOOTAGE_DOWNLOAD_SEM = threading.BoundedSemaphore(4)
_FOOTAGE_THREADS = int(os.environ.get("BOARD_FOOTAGE_THREADS", 2))
_FOOTAGE_STALL_TIMEOUT = max(1.0, float(os.environ.get('BOARD_FOOTAGE_STALL_TIMEOUT', 300)))
_EXPORT_SEM = threading.BoundedSemaphore(2)
_EXPORT_MAX_BYTES = 256 * 1024**2
_EXPORT_LABEL_BYTES = 16 * 1024**2
_FOOTAGE_LOCKS: dict = {}
_FOOTAGE_LOCKS_LOCK = threading.Lock()


_LABEL_INDEX_LOCK = threading.Lock()
_LABEL_INDEX = {}
_LABEL_INDEX_ROOT = None
_LABEL_INDEX_CHECKED = 0.0


@contextmanager
def _footage_lock(key):
    with _FOOTAGE_LOCKS_LOCK:
        entry = _FOOTAGE_LOCKS.setdefault(key, [threading.Lock(), 0])
        entry[1] += 1
    try:
        with entry[0]:
            yield
    finally:
        with _FOOTAGE_LOCKS_LOCK:
            entry[1] -= 1
            if not entry[1]:
                _FOOTAGE_LOCKS.pop(key, None)


def footage_label(eid: str) -> dict:
    """Index episode IDs once and read only labels changed since the previous scan."""
    global _LABEL_INDEX_ROOT, _LABEL_INDEX_CHECKED
    with _LABEL_INDEX_LOCK:
        if _LABEL_INDEX_ROOT != HERE or time.monotonic() - _LABEL_INDEX_CHECKED >= 1:
            previous = dict(_LABEL_INDEX) if _LABEL_INDEX_ROOT == HERE else {}
            _LABEL_INDEX.clear()
            for path in sorted(HERE.glob('*.json')):
                if not _under(HERE, path):
                    continue
                try:
                    st = path.stat()
                    sig = (st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
                    old = previous.get(path.name)
                    if old is not None and old[0] == sig:
                        _LABEL_INDEX[path.name] = old
                        continue
                    label = json.loads(path.read_text())
                    identity = ((label.get('_meta') or {}).get('episode_id') or path.stem) if isinstance(label, dict) else None
                    _LABEL_INDEX[path.name] = (sig, identity)
                except (OSError, ValueError, TypeError, AttributeError):
                    continue
            _LABEL_INDEX_ROOT, _LABEL_INDEX_CHECKED = HERE, time.monotonic()
        path = next((HERE / name for name, (_, identity) in _LABEL_INDEX.items() if identity == eid), None)
    if path is not None:
        try:
            label = json.loads(path.read_text())
            if ((label.get('_meta') or {}).get('episode_id') or path.stem) == eid:
                return current_label(label, DICTIONARY_JOB, path.name)
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    return {}


def footage_cams(clips: Path, eid: str, label: dict | None = None) -> list:
    """Playable cameras in the current episode's presentation order, with legacy discovery as fallback."""
    extra = sorted((d.name for d in clips.iterdir() if d.is_dir() and EXTRA_CAM.fullmatch(d.name)),
                   key=lambda n: (n.startswith("unshown"), int(re.sub(r"\D", "", n)))) if clips.is_dir() else []
    cams = [(c, clip_path(clips, eid, c)) for c in ("exo", "left", "right", *extra)
            if clip_path(clips, eid, c).is_file()]
    label = footage_label(eid) if label is None else label
    if label.get('camera_views'):
        from board.static import shown_cams
        order = shown_cams(label)
        # The same current camera_presentation drives the player and static
        # export. Files from a withheld view are not evidence of availability.
        cams = [(cam, clip) for cam, clip in cams if cam in order]
        roles = label.get('camera_presentation') or {}
        if any((roles.get(cam) or {}).get('role') == 'scene' for cam, _ in cams):
            cams.sort(key=lambda item: order.index(item[0]))
    return cams


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
    version = next((version for reader, version, _ in getattr(_MEDIA_READERS, 'active', [])
                    if Path(reader.name) == p), None)
    return _probe_versioned(p, version or _footage_version(p))


def _probe_versioned(p: Path, version: tuple) -> tuple:
    from board.static import _durable_metadata
    if _durable_metadata(str(p), version):
        return _cached_probe_versioned(p, version)
    return _cached_probe_versioned.__wrapped__(p, version)


@contextmanager
def _probe_admission(key):
    with _PROBE_ACTIVE_LOCK:
        if key in _PROBE_ACTIVE or not _FOOTAGE_PROBE_SEM.acquire(blocking=False):
            raise CacheBusy('Media preparation is busy; try again shortly')
        _PROBE_ACTIVE.add(key)
    try:
        yield
    finally:
        with _PROBE_ACTIVE_LOCK:
            _PROBE_ACTIVE.remove(key)
            _FOOTAGE_PROBE_SEM.release()


@lru_cache(maxsize=2048)
def _cached_probe_versioned(p: Path, version: tuple) -> tuple:
    # Trusted cache hits bypass admission; cold probes and hashes share its bound.
    with _probe_admission((p, version)):
        return _probe_uncached(p)


_probe_versioned.cache_clear = _cached_probe_versioned.cache_clear


def _probe_uncached(p: Path) -> tuple:
    """(width, height, duration s, frame rate) of a clip. The duration runs to the end of its last frame (its
    time plus its own length), which a variable-rate recording's container duration can stop short of. The rate is
    one over the clip's usual step between frames: a clip timed by capture times (board/clips.py retime) starts with
    a frame half a step long, which puts its average rate above the camera's own and the footage canvas
    (footage_command) at a rate that samples on the frames' boundaries."""
    p = _source_path(p)
    probe = Path(FFMPEG).with_name("ffprobe") if FFMPEG else None
    r = subprocess.run([str(probe) if probe and probe.exists() else "ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,avg_frame_rate:format=duration",
                        "-of", "json", str(p)], capture_output=True, text=True, timeout=30, check=True,
                       **_source_process_options())
    j = json.loads(r.stdout)
    s = j["streams"][0]
    import av
    with av.open(str(p)) as c:
        pk = [(float(x.pts * x.time_base), float(x.duration * x.time_base))
              for x in c.demux(c.streams.video[0]) if x.size and x.pts is not None]
    # Some FFprobe builds omit packet lengths and stop their container duration at the final frame's start.
    dur = max(t + d for t, d in pk) if pk else float(j["format"]["duration"])
    num, _, den = str(s.get("avg_frame_rate") or "0/1").partition("/")
    rate = float(num) / float(den) if den and float(den) else 0.0
    steps = sorted(b[0] - a[0] for a, b in zip(sorted(pk), sorted(pk)[1:]) if b[0] > a[0])
    if steps:
        rate = round(1.0 / steps[len(steps) // 2], 3)
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
    g = [f"color=c=black:s={W}x{H}:r={fps}:d={t1 - t0:.3f},settb=AVTB,setpts=PTS+{t0:.3f}/TB[b0]"]
    # a camera keeps its last frame on to t1, as the page's video does once it has played to its end: the overlay
    # dropped a camera to black from its last frame's start (a 15 fps camera's last 1/30 s), so each is held there
    # (tpad), and the file keeps the canvas's own frames (those before t1), which the held frames would otherwise run past
    hold = f"tpad=stop_mode=clone:stop_duration={t1 - t0 + 1:.3f}"
    for i, ((_, wh), (x, y, w, h)) in enumerate(zip(inputs, cells)):
        g.append(f"[{i}:v]{'' if (w, h) == tuple(wh) else f'scale={w}:{h}:flags=lanczos,'}"
                 f"setsar=1,settb=AVTB,{hold}[c{i}]")
        g.append(f"[b{i}][c{i}]overlay={x}:{y}:eof_action=pass[b{i + 1}]")
    g.append(f"[b{len(inputs)}]setpts=PTS-STARTPTS,format=yuv420p[v]")
    n = math.ceil(round((t1 - t0) * fps, 6))
    return cmd + ["-filter_complex", ";".join(g), "-map", "[v]", "-frames:v", str(n),
                  "-r", str(fps), "-fps_mode", "cfr", "-an", "-c:v", "libx264",
                  "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-g", str(2 * FOOTAGE_FPS), "-threads",
                  str(threads), "-filter_complex_threads", "1", "-movflags", "+faststart", str(out)]


def _footage_version(path: Path) -> tuple:
    st = path.stat()
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def _stream_digest(source, size, destination=None):
    """Hash exactly the admitted byte count, never follow a growing source."""
    digest = hashlib.sha256()
    remaining = size
    while remaining:
        chunk = source.read(min(1024 * 1024, remaining))
        if not chunk:
            raise ValueError('Media changed while its snapshot was being read')
        digest.update(chunk)
        if destination is not None:
            destination.write(chunk)
        remaining -= len(chunk)
    if source.read(1):
        raise ValueError('Media grew while its snapshot was being read')
    return digest.hexdigest()


def _copy_source_snapshot(source: Path, target: Path, version: tuple) -> str:
    with source.open('rb') as incoming, target.open('xb') as outgoing:
        digest = _stream_digest(incoming, version[2], outgoing)
    if _footage_version(source) != version:
        raise ValueError('Media changed while its snapshot was being read')
    return digest


@contextmanager
def _file_lock(path, *, blocking=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as lock:
        try:
            if os.name == 'nt':
                import msvcrt
                lock.seek(0)
                if not lock.read(1):
                    lock.write(b'0')
                    lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as error:
            raise CacheBusy('Media preparation is busy; try again shortly') from error
        except OSError as error:
            if os.name == 'nt' and error.errno in (13, 36):
                raise CacheBusy('Media preparation is busy; try again shortly') from error
            raise
        try:
            yield
        finally:
            if os.name == 'nt':
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


@contextmanager
def _snapshot_workspace():
    """Portable fallback uses independent, crash-recoverable copy slots."""
    from board.build import remove_tree
    folder = FOOTAGE_DIR / '.sources'
    folder.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for slot in range(max(1, int(os.environ.get('BOARD_FOOTAGE_CONCURRENCY', 1)))):
            try:
                stack.enter_context(_file_lock(folder / f'.slot-{slot}.lock', blocking=False))
                break
            except CacheBusy:
                continue
        else:
            raise CacheBusy('Media snapshot preparation is busy; try again shortly')
        prefix = f'snapshot-{slot}-'
        for old in folder.glob(prefix + '*'):
            if old.is_dir() and not old.is_symlink():
                remove_tree(old)
        temporary = Path(tempfile.mkdtemp(prefix=prefix, dir=folder))
        try:
            yield temporary
        finally:
            remove_tree(temporary)


def _source_path(path):
    if os.name == 'posix':
        for reader, _, _ in getattr(_MEDIA_READERS, 'active', []):
            if Path(reader.name) == path:
                return Path('/dev/fd') / str(reader.fileno())
    return path


@contextmanager
def _encode_sources(cams):
    if os.name == 'posix':
        yield [(cam, _source_path(path)) for cam, path in cams]
        return
    readers = {Path(reader.name): (version, digest) for reader, version, digest
               in getattr(_MEDIA_READERS, 'active', [])}
    if not any(digest is not None for _, digest in readers.values()):
        yield cams
        return
    with _snapshot_workspace() as temporary:
        working = []
        for index, (cam, path) in enumerate(cams):
            version, digest = readers[path]
            if digest is not None:
                target = temporary / f'{index}.mp4'
                with _FOOTAGE_PROBE_SEM:
                    if _copy_source_snapshot(path, target, version) != digest:
                        raise ValueError('Media changed during footage preparation')
                path = target
            working.append((cam, path))
        yield working


def _source_process_options():
    readers = getattr(_MEDIA_READERS, 'active', [])
    return {'pass_fds': tuple(reader.fileno() for reader, _, _ in readers)} if os.name == 'posix' and readers else {}


def _validate_active_sources():
    # An open descriptor binds pathname replacements. Input bytes must remain
    # stable throughout preparation; before/after hashes detect persistent edits,
    # not a transient in-place edit restored while an encoder is reading.
    for reader, version, digest in getattr(_MEDIA_READERS, 'active', []):
        stat = os.fstat(reader.fileno())
        current = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if digest is not None:
            with _FOOTAGE_PROBE_SEM:
                reader.seek(0)
                if _stream_digest(reader, version[2]) != digest:
                    raise ValueError('Media changed during footage preparation')
        elif current != version:
            raise ValueError('Media changed during footage preparation')


@contextmanager
def _footage_sources(cams):
    """Retain source handles; untrusted files are content-validated on every prepare.

    POSIX probe and encode children inherit these same handles. A ready hit reads
    each untrusted source once, without copying it. Misses verify its content again
    before publication. Source media must not be edited during preparation.
    """
    from board.static import _durable_metadata
    if len(cams) > 64:
        raise ValueError('Too many footage cameras')
    with ExitStack() as stack:
        previous = getattr(_MEDIA_READERS, 'active', [])
        active, working, identities, digests = [], [], [], []
        _MEDIA_READERS.active = active
        try:
            for index, (cam, path) in enumerate(cams):
                reader = stack.enter_context(path.open('rb'))
                stat = os.fstat(reader.fileno())
                version = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                digest = None
                if not _durable_metadata(str(path), version):
                    with _probe_admission(('source', str(path), version)):
                        digest = _stream_digest(reader, version[2])
                    reader.seek(0)
                active.append((reader, version, digest))
                working.append((cam, path))
                identities.append([str(path), 'sha256', digest] if digest else [str(path), *version])
                digests.append(digest)
            yield working, identities, digests
        finally:
            _MEDIA_READERS.active = previous


@contextmanager
def _publication_lock(key):
    """A fixed set of cross-process locks protects short output transactions."""
    with _file_lock(FOOTAGE_DIR / '.locks' / f'{int(key[:2], 16) % 64:02x}'):
        yield


def _trim_footage_cache(protect):
    with _file_lock(FOOTAGE_DIR / '.locks' / 'cache'):
        _trim_disk_cache(FOOTAGE_DIR, '.mp4', max_bytes=_FOOTAGE_DISK_BYTES,
                         max_files=_FOOTAGE_DISK_FILES if _FOOTAGE_DISK_FILES > 0 else 2**63, protect=protect)


def _content_probe(path: Path, digest: str) -> tuple:
    # The digest identifies validated source content, independent of its FD pathname.
    with _CONTENT_PROBE_LOCK:
        hit = _CONTENT_PROBES.get(digest)
        if hit is not None:
            _CONTENT_PROBES.move_to_end(digest)
            return hit
    with _probe_admission(('content', digest)):
        result = _probe_uncached(path)
    with _CONTENT_PROBE_LOCK:
        _CONTENT_PROBES[digest] = result
        _CONTENT_PROBES.move_to_end(digest)
        while len(_CONTENT_PROBES) > 2048:
            _CONTENT_PROBES.popitem(last=False)
    return result


def _footage_receipt(result: tuple) -> dict:
    from board.static import _durable_metadata
    out, name = result
    version = _footage_version(out)
    if out.is_symlink() or not 0 < version[2] <= _FOOTAGE_DISK_BYTES:
        raise ValueError('Prepared footage exceeds the disk cache budget')
    record = {'format': 'argus-footage-ready/2', 'file': out.name, 'name': name, 'version': version}
    if not _durable_metadata(str(out), version):
        with _FOOTAGE_DOWNLOAD_SEM:
            with out.open('rb') as source:
                record['sha256'] = _stream_digest(source, version[2])
        if _footage_version(out) != version:
            raise ValueError('Prepared footage changed while its identity was being read')
    return record


def _discard_changed_footage(out: Path) -> None:
    try:
        out.unlink(missing_ok=True)
    except OSError as error:
        raise RuntimeError('Prepared footage changed and could not be discarded') from error


def _receipt_footage(record: dict, *, reader=None) -> tuple | None:
    from board.static import _durable_metadata
    if (record.get('format') != 'argus-footage-ready/2' or
            not re.fullmatch(r'[a-f0-9]{20}\.mp4', record['file']) or
            not isinstance(record['name'], str) or Path(record['name']).name != record['name']):
        return None
    out = FOOTAGE_DIR / record['file']
    version = _footage_version(out)
    if out.is_symlink() or list(version) != record['version'] or not 0 < version[2] <= _FOOTAGE_DISK_BYTES:
        return None
    if reader is not None:
        stat = os.fstat(reader.fileno())
        opened_version = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if opened_version != version:
            return None
    if 'sha256' in record:
        with _FOOTAGE_DOWNLOAD_SEM:
            if reader is None:
                with out.open('rb') as source:
                    digest = _stream_digest(source, version[2])
            else:
                digest = _stream_digest(reader, version[2])
                reader.seek(0)
        if digest != record['sha256'] or _footage_version(out) != version:
            return None
    elif not _durable_metadata(str(out), version):
        return None
    return out, record['name']


def _read_footage_receipt(receipt: Path) -> dict | None:
    if receipt.is_symlink():
        return None
    with receipt.open('rb') as source:
        raw = source.read(4097)
    if len(raw) > 4096:
        return None
    record = json.loads(raw)
    return record if isinstance(record, dict) else None


def _cached_footage_record(key):
    try:
        record = _read_footage_receipt(FOOTAGE_DIR / '.ready' / f'{key}.json')
        result = _receipt_footage(record) if record and record.get('file') == key + '.mp4' else None
        if result is not None:
            return record, result
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    # A surviving ticket is also a durable receipt for this exact output,
    # including when a canonical record is malformed or no longer validates.
    for path in (FOOTAGE_DIR / '.tickets').glob('*.json'):
        if not re.fullmatch(r'[a-f0-9]{32}\.json', path.name):
            continue
        try:
            record = _ticket_record(path.stem)
            if record and record.get('file') == key + '.mp4':
                result = _receipt_footage(record)
                if result is not None:
                    return record, result
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    return None, None


def _ready_footage(request: str) -> tuple | None:
    try:
        if len(request) == 20:
            return _cached_footage_record(request)[1]
        record = _read_footage_receipt(FOOTAGE_DIR / '.ready' / f'{request}.json')
        if record is not None and record.get('format') == 'argus-footage-alias/1':
            key = record.get('key', '')
            if not re.fullmatch(r'[a-f0-9]{20}', key):
                return None
            return _cached_footage_record(key)[1]
        return _receipt_footage(record) if record is not None else None
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def _trim_footage_receipts(folder: Path, pattern: str, *, expire=False):
    retained = []
    for path in folder.glob('*.json'):
        if not re.fullmatch(pattern, path.name):
            continue
        try:
            record = _read_footage_receipt(path)
            filename = record.get('file') if record else None
            if record and record.get('format') == 'argus-footage-alias/1':
                filename = str(record.get('key')) + '.mp4'
            expiry = record.get('expires_at') if expire and record else None
            valid = (isinstance(filename, str) and re.fullmatch(r'[a-f0-9]{20}\.mp4', filename)
                     and (FOOTAGE_DIR / filename).is_file())
            if expire:
                valid = valid and type(expiry) in (int, float) and math.isfinite(expiry) and expiry >= time.time()
            if not valid:
                path.unlink(missing_ok=True)
            elif expire or len(path.stem) == 40:
                retained.append((path.stat().st_mtime_ns, path))
        except FileNotFoundError:
            pass
        except (ValueError, TypeError, KeyError):
            path.unlink(missing_ok=True)
    # Canonical records follow movie lifetime. Request aliases and expiring
    # tickets have independent finite bounds, including when media count is zero.
    limit = ((_FOOTAGE_DISK_FILES if _FOOTAGE_DISK_FILES > 0 else 128) if expire
             else max(128, 2 * _FOOTAGE_DISK_FILES))
    for _, path in sorted(retained, reverse=True)[limit:]:
        path.unlink(missing_ok=True)


def _remember_ready_footage(request: str, result: tuple, record=None) -> None:
    folder = FOOTAGE_DIR / '.ready'
    if len(request) == 20:
        write_atomic(folder / f'{request}.json', record or _footage_receipt(result))
    else:
        write_atomic(folder / f'{request}.json', {'format': 'argus-footage-alias/1', 'key': result[0].stem})
    with _FOOTAGE_READY_LOCK:
        _trim_footage_receipts(folder, r'(?:[a-f0-9]{20}|[a-f0-9]{40})\.json')


def prepare_footage_ticket(result: tuple) -> str:
    record, _ = _cached_footage_record(result[0].stem)
    if record is None or list(_footage_version(result[0])) != record.get('version'):
        raise ValueError('Prepared footage changed before ticket publication')
    record['expires_at'] = time.time() + _FOOTAGE_TICKET_SECONDS
    token = uuid.uuid4().hex
    with _FOOTAGE_READY_LOCK:
        folder = FOOTAGE_DIR / '.tickets'
        write_atomic(folder / f'{token}.json', record)
        _trim_footage_receipts(folder, r'[a-f0-9]{32}\.json', expire=True)
    return token


def _ticket_record(token: str) -> dict | None:
    if not re.fullmatch(r'[a-f0-9]{32}', token):
        return None
    receipt = FOOTAGE_DIR / '.tickets' / f'{token}.json'
    record = _read_footage_receipt(receipt)
    if record is None:
        return None
    expiry = record.get('expires_at')
    if type(expiry) not in (int, float) or not math.isfinite(expiry) or expiry < time.time():
        receipt.unlink(missing_ok=True)
        return None
    return record


def ticket_footage(token: str) -> tuple | None:
    try:
        record = _ticket_record(token)
        return _receipt_footage(record) if record is not None else None
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


@contextmanager
def _ticket_reader(token):
    # Validate and send the same open file so a path replacement cannot switch
    # bytes between identity checking and Range delivery.
    reader, result = None, None
    try:
        record = _ticket_record(token)
        if record and re.fullmatch(r'[a-f0-9]{20}\.mp4', record.get('file', '')):
            reader = (FOOTAGE_DIR / record['file']).open('rb')
            result = _receipt_footage(record, reader=reader)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    try:
        yield (result, reader) if result is not None else (None, None)
    finally:
        if reader is not None:
            reader.close()


def footage(eid: str, t0: float = 0.0, t1: float | None = None) -> tuple | None:
    """Prepare the current sources; tickets separately serve the exact prepared output."""
    if not math.isfinite(t0) or (t1 is not None and not math.isfinite(t1)):
        raise ValueError('Footage times must be finite')
    label = footage_label(eid)
    cams = footage_cams(MP4_DIR, eid, label)
    if not cams or not FFMPEG:
        return None
    with _footage_sources(cams) as (working, identities, digests):
        return _footage_from_sources(eid, t0, t1, label, working, identities, digests)


def _footage_from_sources(eid, t0, t1, label, cams, identities, digests):
    request_parts = [eid, float(t0), None if t1 is None else float(t1), FOOTAGE_TAG,
                     label.get('duration_s'), label.get('camera_views'), identities]
    request = hashlib.sha1(json.dumps(request_parts).encode()).hexdigest()
    ready = _ready_footage(request)
    if ready is not None:
        _trim_footage_cache(ready[0])
        return ready
    probes = [_content_probe(path, digest) if digest else _probe(path)
              for (_, path), digest in zip(cams, digests)]
    clocks = label.get('camera_views') or ['exo', 'left', 'right']
    anchor = next((cam for cam in clocks if any(c == cam for c, _ in cams)), cams[0][0])
    dur = label.get('duration_s')
    if type(dur) not in (int, float) or not math.isfinite(dur) or dur <= 0:
        dur = next(pr[2] for (cam, _), pr in zip(cams, probes) if cam == anchor)
    t0 = round(max(0.0, min(float(t0), dur)), 3)
    t1 = round(dur if t1 is None else max(0.0, min(float(t1), dur)), 3)
    if t1 - t0 < 1.0 / FOOTAGE_FPS:
        return None
    if any(pr[0] * pr[1] > 16_777_216 or pr[3] > 120 for pr in probes):
        raise ValueError('Camera dimensions or frame rate exceed the footage limit')
    whole = t0 == 0 and t1 >= round(dur, 3)
    name = f'{eid}.mp4' if whole else f'{eid}_{t0:.1f}-{t1:.1f}s.mp4'
    key = hashlib.sha1(json.dumps([eid, t0, t1, FOOTAGE_TAG, identities]).encode()).hexdigest()[:20]
    out = FOOTAGE_DIR / f'{key}.mp4'
    with _footage_lock(key):
        ready = _ready_footage(key)
        if ready is not None:
            _remember_ready_footage(request, ready)
            return ready
        result = _uncached_footage(key, out, name, cams, probes, t0, t1)
        _remember_ready_footage(request, result)
        return result


def _run_footage(cmd: list) -> subprocess.CompletedProcess:
    """Bound stalled work while allowing a progressing encode of any duration."""
    progress_cmd = [*cmd[:-1], '-progress', 'pipe:1', '-stats_period', '5', cmd[-1]]
    last_progress, last_values = time.monotonic(), (0, 0)
    with subprocess.Popen(progress_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                          **_source_process_options()) as process:
        try:
            while True:
                try:
                    stdout, stderr = process.communicate(timeout=min(5.0, _FOOTAGE_STALL_TIMEOUT))
                    return subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)
                except subprocess.TimeoutExpired as error:
                    output = error.output or b''
                    if isinstance(output, str):
                        output = output.encode()
                    times = re.findall(rb'out_time_us=(\d+)', output)
                    try:
                        size = Path(cmd[-1]).stat().st_size
                    except OSError:
                        size = 0
                    values = (max((int(value) for value in times), default=0), size)
                    if any(value > previous for value, previous in zip(values, last_values)):
                        last_progress, last_values = time.monotonic(), values
                    if time.monotonic() - last_progress >= _FOOTAGE_STALL_TIMEOUT:
                        raise subprocess.TimeoutExpired(cmd, _FOOTAGE_STALL_TIMEOUT)
        except BaseException:
            process.kill()
            process.communicate()
            raise


@_bounded_work(_FOOTAGE_SEM)
def _uncached_footage(key, out, name, cams, probes, t0, t1):
    with _encode_sources(cams) as working:
        return _encode_and_publish(key, out, name, working, probes, t0, t1)


def _encode_and_publish(key, out, name, cams, probes, t0, t1):
    # A competing process may publish while this encoder runs. Publication adopts
    # its verified result instead of replacing a file held by a live ticket.
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.{threading.get_ident()}.part.mp4")
    cmd = footage_command([(p, pr[:2], pr[3]) for (_, p), pr in zip(cams, probes)], t0, t1, tmp, _FOOTAGE_THREADS)
    cmd[-1:-1] = ['-fs', str(_FOOTAGE_DISK_BYTES)]
    nice = shutil.which('nice')
    try:
        completed = _run_footage(([nice, '-n', '19'] if nice else []) + cmd)
        if completed.returncode != 0 or not tmp.is_file():
            raise RuntimeError(f"the video could not be made: {(completed.stderr or '').strip()[-300:]}")
        if tmp.stat().st_size >= _FOOTAGE_DISK_BYTES:
            raise ValueError('Prepared footage exceeds the disk cache budget')
        from board.clips import frame_lengths
        frame_lengths(tmp)
        _validate_active_sources()
        record = _footage_receipt((tmp, name))
        with _publication_lock(key):
            ready = _ready_footage(key)
            if ready is not None:
                return ready
            # Only the short rename/receipt/prune transaction shares a cache lock.
            # Source work, encoding and full output hashing run independently.
            with _file_lock(FOOTAGE_DIR / '.locks' / 'cache'):
                if out.exists():
                    _discard_changed_footage(out)
                os.replace(tmp, out)
                record.update(file=out.name, version=_footage_version(out))
                try:
                    _remember_ready_footage(key, (out, name), record)
                except BaseException:
                    _discard_changed_footage(out)
                    raise
                _trim_disk_cache(FOOTAGE_DIR, '.mp4', max_bytes=_FOOTAGE_DISK_BYTES,
                                 max_files=_FOOTAGE_DISK_FILES if _FOOTAGE_DISK_FILES > 0 else 2**63, protect=out)
        return out, name
    finally:
        tmp.unlink(missing_ok=True)


PORT = 8896
PAGE_TITLE = "Data Dashboard"
DICTIONARY_JOB = None                # local editing requires an explicit upload root
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
    raw = dumps(public_json(out)).encode()
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


def public_label(d: dict) -> dict:
    """Copy public JSON after source checks and current human interpretation."""
    return public_json(d)


def current_label(d: dict, job: Path | None, filename: str) -> dict:
    """Project current human interpretation for a public read or export."""
    if job is None:
        return d
    from board.dictionary_projection import project
    from label.dictionary_editor import for_episode
    import hashlib
    meta = d.get('_meta') or {}
    owner = (d.get('data_dictionary') or {}).get('episode_id') or meta.get('run_episode') or Path(filename).stem
    if not isinstance(owner, str) or owner in ('', '.', '..') or Path(owner).name != owner:
        return d
    episode_dir = Path(job) / 'episodes' / owner
    if not episode_dir.is_dir():
        units = Path(job) / 'units' / owner
        if units.is_dir():
            episode_dir = units
    if not _under(Path(job), episode_dir):
        return d
    # A filename alone is not source ownership. Fresh labels carry the prepared
    # context's location receipt; older dictionary labels retain an exact inventory.
    proofs = [d.get('source_identity')]
    proofs.extend((d.get(key) or {}).get('source_proof') for key in
                  ('sensor_evidence', 'grip_evidence', 'evidence_inspection')
                  if isinstance(d.get(key), dict))
    locations = {item.get('location_sha256') for proof in proofs if isinstance(proof, list)
                 for item in proof if isinstance(item, dict) and item.get('name') == 'prepared:context.json'}
    location = hashlib.sha256(str((episode_dir / 'context.json').resolve()).encode()).hexdigest()
    if locations:
        if location not in locations:
            return d
    else:
        old = d.get('data_dictionary') or {}
        current = for_episode(episode_dir, owner) or {}
        if (not old.get('inventory_digest') or old.get('episode_id') != owner
                or old.get('inventory_digest') != current.get('inventory_digest')):
            return d
    return project(d, episode_dir, owner)


def withheld_status(result: dict) -> dict:
    """What a check a rule withheld found, so the page says it as the check came out, never clear for a check that
    crashed, fired or measured nothing: {"status": "errored" | "not_assessed" | "fired" | "clear"}, with the error or
    the reason it was not assessed. A result that lists checks of its own (capture_qc, sensor_checks) also gives how
    many of them ran ("of") and how many of those fired and errored. The order is the page's for our own checks
    (checksSection): an error, then not assessed, then fired."""
    if result.get("error"):
        return {"status": "errored", "error": str(result["error"])}
    flagged = any(bool(result.get(f)) for f in ("flagged", "crossed", "sped_up_recording"))
    if isinstance(result.get("checks"), list):
        st = [c.get("status") for c in result["checks"] if isinstance(c, dict)]
        n = {"fired": st.count("fired"), "errored": st.count("errored"),
             "of": sum(s in ("fired", "clear", "errored") for s in st)}
        # a result that says it flagged fired, whatever its own list of checks holds
        return {"status": "errored" if n["errored"] else "fired" if n["fired"] or flagged else "clear" if n["of"]
                else "not_assessed", **n}
    if result.get("not_assessed"):
        return {"status": "not_assessed", "why": str(result["not_assessed"])}
    return {"status": "fired" if flagged else "clear"}


def episode_view(d: dict) -> dict:
    """The episode as the page shows it: its public fields (public_label), each flagged issue carrying the family it
    is counted under and whether it counts (Families.counts), so the episode's own list names and counts a problem
    exactly as the filter does, and the checks a manifest rule withheld on this dataset (drop_check, kept in the
    label file's _withheld_checks) as set_aside_checks, each with the rule's reason and what it found
    (withheld_status), so the page lists them as set aside, never hides them. The label file itself is not changed."""
    held = d.get("_withheld_checks") if isinstance(d.get("_withheld_checks"), dict) else {}
    d = public_label(d)
    aside = [{"check": k, "reason": v.get("reason") or "", **withheld_status(r)}
             for k, v in held.items() if isinstance(v, dict) for r in [v.get("result") or {}] if isinstance(r, dict)]
    if aside:
        d["set_aside_checks"] = aside
    ds = d.get("dataset")
    for key in ("data_issues", "operator_mistakes"):
        for i in d.get(key) or []:
            if isinstance(i, dict) and i.get("issue"):
                i["family"] = FAMILIES.family_of(key, i, ds)
                i["counted"] = FAMILIES.counts(key, i)
    return public_json(d)


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
        # a long recording labelled in parts, some of which gave no labels (label/pieces.py stitch_run)
        **({"parts_missing": len(d["_stitched"]["missing"]), "parts": d["_stitched"].get("parts")}
           if isinstance(d.get("_stitched"), dict) and d["_stitched"].get("missing") else {}),
        # another model's labels (compare/): how its response came out (parsed, unparsed, cut_off, no_response)
        **({"cmp_status": d["_compare"].get("status")} if isinstance(d.get("_compare"), dict) else {}),
    }


INDEX_HTML = Path(__file__).with_name("index.html").read_text()


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
    cfg = public_json({**board, "models": {**model_names(), **(board.get("models") or {})},
                       "reasoning": reasoning_effort()})
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
            body = dumps(body)
        if isinstance(body, str):
            body = body.encode()
        # compress large JSON and HTML: a large board's episode list is megabytes, about ten times smaller
        # gzipped, and the page waits for it before its first paint
        gz = len(body) > 16384 and "gzip" in (self.headers.get("Accept-Encoding") or "")
        if gz:
            body = gzipped if gzipped is not None else gzip.compress(body, compresslevel=5)
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        if self.close_connection:
            self.send_header("Connection", "close")
        if code == 503:
            self.send_header("Retry-After", "1")
        if gz:
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, p: Path, ctype: str, download: str | None = None, *, source=None):
        """Serve a file with proper HTTP Range support so the browser
        can stream/seek without downloading the whole thing first; download names the saved file."""
        try:
            st = os.fstat(source.fileno()) if source is not None else p.stat()
        except OSError:
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
            f = source if source is not None else open(p, "rb")
        except OSError:
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
            if source is None:
                f.close()

    def _dictionary(self, body=None):
        from label.dictionary_editor import public_dictionary, save_override, RevisionConflict
        parsed = urllib.parse.urlparse(self.path)
        if body is not None:
            host = self.headers.get('Host') or ''
            origin = self.headers.get('Origin')
            local = urllib.parse.urlparse('http://' + host).hostname in ('127.0.0.1', 'localhost', '::1')
            if DICTIONARY_JOB is None or not local or origin != 'http://' + host:
                return self._send(403, {'error': 'Dictionary editing is disabled.'})
            if (self.headers.get('Content-Type') or '').split(';')[0].strip().lower() != 'application/json':
                return self._send(400, {'error': 'The dictionary edit must be JSON.'})
            try:
                return self._send(200, public_json({**save_override(DICTIONARY_JOB, body, 'local'), 'editable': True}))
            except RevisionConflict as error:
                return self._send(409, {'error': str(error)})
            except ValueError as error:
                return self._send(400, {'error': str(error)})
            except OSError:
                return self._send(400, {'error': 'The dictionary edit could not be saved.'})
        q = urllib.parse.parse_qs(parsed.query)
        fname = (q.get('file') or [''])[0]
        p = HERE / fname
        if not fname or not _under(HERE, p) or not p.is_file() or p.suffix != '.json':
            return self._send(404, {'error': 'No such episode.'})
        try:
            d = json.loads(p.read_text())
            if not isinstance(d, dict) or not isinstance(d.get('data_dictionary') or {}, dict):
                raise ValueError('Invalid dictionary document')
        except (OSError, ValueError):
            return self._send(500, {'error': 'The saved data dictionary is unreadable.'})
        if DICTIONARY_JOB is None:
            return self._send(200, public_json({**(d.get('data_dictionary') or {}), 'editable': False}))
        try:
            return self._send(200, public_json({**public_dictionary(DICTIONARY_JOB, (d.get('data_dictionary') or {}).get('episode_id') or (d.get('_meta') or {}).get('episode_id') or p.stem), 'editable': True}))
        except (ValueError, OSError):
            return self._send(200, {'fields': [], 'limitations': ['The saved data dictionary is unreadable.'], 'editable': False})

    def _body_size(self, limit):
        lengths = self.headers.get_all('Content-Length') if hasattr(self.headers, 'get_all') else [self.headers.get('Content-Length')]
        if (self.headers.get('Transfer-Encoding') is not None or len(lengths or []) != 1
                or len(lengths[0] or '') > len(str(limit))
                or not re.fullmatch(r'[0-9]+', lengths[0] or '')
                or not 0 < int(lengths[0]) <= limit):
            self.close_connection = True
            raise ValueError('Unsupported request body framing')
        return int(lengths[0])

    def do_POST(self):
        """Export unique episodes through a bounded spool before sending headers."""
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path not in ('/api/export', '/api/dictionary'):
            self.close_connection = True
            return self._send(404, {'error': 'not found'})
        try:
            n = self._body_size(32768 if parsed.path == '/api/dictionary' else 1024 * 1024)
            raw = self.rfile.read(n)
            if len(raw) != n:
                self.close_connection = True
                raise ValueError('Incomplete request body')
            payload = json.loads(raw)
        except (ValueError, json.JSONDecodeError):
            return self._send(400, {'error': 'bad request'})
        if parsed.path == '/api/dictionary':
            return self._dictionary(payload)
        if (not isinstance(payload, dict) or not isinstance(payload.get('files'), list)
                or len(payload['files']) > 10000 or any(not isinstance(f, str) for f in payload['files'])):
            return self._send(400, {'error': 'bad request'})
        files = list(dict.fromkeys(payload['files']))
        if not _EXPORT_SEM.acquire(blocking=False):
            return self._send(503, {'error': 'Export is busy; try again shortly.'})
        try:
            with tempfile.TemporaryFile() as spool:
                compressed = 'gzip' in (self.headers.get('Accept-Encoding') or '')
                writer = gzip.GzipFile(fileobj=spool, mode='wb', compresslevel=5) if compressed else spool
                total = 0
                seen = set()
                try:
                    for fname in files:
                        path = HERE / fname
                        if not _under(HERE, path) or not path.is_file() or path.suffix != '.json':
                            return self._send(404, {'error': f'no such file: {fname}'})
                        identity = str(path.resolve())
                        if identity in seen:
                            continue
                        seen.add(identity)
                        try:
                            with path.open('rb') as source:
                                raw = source.read(_EXPORT_LABEL_BYTES + 1)
                            if len(raw) > _EXPORT_LABEL_BYTES:
                                return self._send(413, {'error': 'An episode exceeds the export size limit.'})
                            label = json.loads(raw)
                            if not isinstance(label, dict):
                                raise ValueError('Invalid label')
                            label = finite(public_label(current_label(label, DICTIONARY_JOB, path.name)))
                            for text in json.JSONEncoder(ensure_ascii=False, separators=(',', ':'), allow_nan=False).iterencode(label):
                                chunk = text.encode()
                                total += len(chunk)
                                if total > _EXPORT_MAX_BYTES:
                                    return self._send(413, {'error': 'The export exceeds the size limit.'})
                                writer.write(chunk)
                            total += 1
                            if total > _EXPORT_MAX_BYTES:
                                return self._send(413, {'error': 'The export exceeds the size limit.'})
                            writer.write(b'\n')
                        except (OSError, ValueError, TypeError, AttributeError):
                            return self._send(500, {'error': 'An episode could not be read for export.'})
                finally:
                    if compressed:
                        writer.close()
                size = spool.tell()
                spool.seek(0)
                self.send_response(200)
                self.send_header('Content-Type', 'application/x-ndjson')
                self.send_header('Content-Disposition', 'attachment; filename="episodes.jsonl"')
                if compressed:
                    self.send_header('Content-Encoding', 'gzip')
                self.send_header('Content-Length', str(size))
                self.end_headers()
                shutil.copyfileobj(spool, self.wfile, 64 * 1024)
        finally:
            _EXPORT_SEM.release()

    def do_GET(self):
        lengths = self.headers.get_all('Content-Length') or []
        if self.headers.get('Transfer-Encoding') is not None or (lengths and lengths != ['0']):
            self.close_connection = True
            return self._send(400, {'error': 'GET requests cannot carry a body.'})
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            # the page asks for other models' labels, hand pose files and keypoint downloads only when this board has
            # them (no request that can only fail)
            cfg = {"mode": "api", "compare": (COMPARE_DIR / "index.json").is_file(), "hands": HANDS_DIR.is_dir(),
                   "sensors": (SENSORS_DIR / "index.json").is_file(),
                   "grids": (HERE.parent / "grids").is_dir(),
                   "footage": FFMPEG is not None, "dictionary_url": "api/dictionary",
                   "keypoints": (KEYPOINTS_DIR / "index.json").is_file(), "labels_license": labels_license(HERE.parent)}
            self._send(200, render_index(PAGE_TITLE, cfg, BOARD_NAME, HEADER), "text/html; charset=utf-8")
            return
        if parsed.path == '/api/dictionary':
            return self._dictionary()
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
                body = dumps(public_label(current_label(json.loads(p.read_text()), DICTIONARY_JOB, p.name)), separators=(",", ":")).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Disposition", f'attachment; filename="{p.name}"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self._send(200, dumps(episode_view(current_label(json.loads(p.read_text()), DICTIONARY_JOB, p.name))), "application/json")
            return
        if parsed.path.startswith("/api/compare/"):
            # other models' labels (board/build.py compare/): kept beside the board's own and never in its lists
            what = parsed.path[len("/api/compare/"):]
            if what in ("index", "metrics"):
                p = COMPARE_DIR / f"{what}.json"
                if not p.is_file():
                    self._send(404, {"error": "no comparison on this dashboard"})
                    return
                self._send(200, public_json(json.loads(p.read_text())), "application/json")
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
                self._send(200, dumps(episode_view(current_label(json.loads(p.read_text()), DICTIONARY_JOB, p.name))), "application/json")
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
            self._send(200, public_json(json.loads(p.read_text())), "application/json")
            return
        if parsed.path in ("/api/grids", "/api/grid"):
            from board import grids as bg
            q = urllib.parse.parse_qs(parsed.query)
            fname = (q.get('file') or [''])[0]
            p = HERE / fname
            if not fname or Path(fname).name != fname or not _under(HERE, p) or not p.is_file():
                return self._send(404, {'error': 'no verified frame grids'})
            try:
                record = current_label(json.loads(p.read_text()), DICTIONARY_JOB, p.name)
                expected = (q.get('fingerprint') or [''])[0]
                if not expected or expected != (record.get('request_images') or {}).get('fingerprint'):
                    return self._send(404, {'error': 'frame grids belong to a different request'})
                idx = bg.read(HERE.parent / 'grids', fname, record)
                if idx is None:
                    return self._send(404, {'error': 'no verified frame grids'})
                if parsed.path == '/api/grids':
                    quoted = urllib.parse.quote(fname, safe='')
                    return self._send(200, bg.view(idx, lambda i, g: f'api/grid?file={quoted}&i={i}&fingerprint={expected}'))
                n = (q.get('i') or [''])[0]
                if not n.isdigit() or int(n) >= len(idx['grids']):
                    return self._send(404, {'error': 'no such frame grid'})
                entry = idx['grids'][int(n)]
                image = bg.image_path(HERE.parent / 'grids', fname, entry).read_bytes()
                from label.grids import digest
                if len(image) != entry['bytes'] or digest(image) != entry['sha256']:
                    return self._send(404, {'error': 'no verified frame grids'})
                return self._send(200, image, 'image/jpeg')
            except (OSError, ValueError, KeyError, TypeError):
                return self._send(404, {'error': 'no verified frame grids'})
        if parsed.path == "/api/sensors":
            # one episode's other signals and depth (board/sensors.py), or index.json for the list; read only by the
            # page's sensors panel and depth switches, never listed or exported
            fname = (urllib.parse.parse_qs(parsed.query).get("file") or [""])[0]
            p = SENSORS_DIR / fname
            if not fname or "/" in fname or not _under(SENSORS_DIR, p) or not p.is_file() or p.suffix != ".json":
                self._send(404, {"error": "no sensors for this episode"})
                return
            self._send(200, public_json(json.loads(p.read_text())), "application/json")
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
                body = dumps(public_json(json.loads(p.read_text())), separators=(",", ":")).encode()
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
            self._send(200, public_json(json.loads(p.read_text())), "application/json")
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
            if 'ticket' in q:
                with _ticket_reader(q['ticket'][0]) as (got, reader):
                    if got is None:
                        return self._send(404, {'error': 'Prepared video expired or changed; prepare it again.'})
                    return self._send_file(got[0], 'video/mp4', got[1], source=reader)
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
            except CacheBusy as error:
                return self._send(503, {'error': str(error)})
            except ValueError as error:
                return self._send(400, {'error': str(error)})
            except (RuntimeError, OSError, subprocess.SubprocessError, KeyError) as e:
                logging.getLogger(__name__).error("Footage preparation failed", exc_info=True)
                self._send(500, {"error": "The requested footage could not be prepared.",
                                 "stage": "footage preparation"})
                return
            if got is None:
                self._send(404, {"error": "no video for this episode and span"})
                return
            out, name = got
            if (q.get("prepare") or [""])[0] == "1":
                try:
                    ticket = prepare_footage_ticket(got)
                    size = out.stat().st_size
                except CacheBusy as error:
                    return self._send(503, {'error': str(error)})
                except (OSError, ValueError):
                    return self._send(409, {'error': 'Prepared video changed; prepare it again.'})
                self._send(200, {"bytes": size, "name": name, "ticket": ticket})
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
            try:
                jpg = extract_frame(mp4, t, w) if _under(MP4_DIR, mp4) else None
            except CacheBusy as error:
                return self._send(503, {'error': str(error)})
            except ValueError as error:
                return self._send(400, {'error': str(error)})
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
    global HEADER, DICTIONARY_JOB
    ap = argparse.ArgumentParser(prog="python -m board serve", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", type=Path, required=True, help="a board folder (its qa/ holds the episode files)")
    ap.add_argument("--clips", type=Path, required=True, help="the clips folder board clips wrote")
    ap.add_argument("--host", default="127.0.0.1", help="listen address (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=PORT, help=f"the port to listen on (default {PORT})")
    ap.add_argument("--title", default=PAGE_TITLE, help=f"the page title, in the header and the browser tab "
                                                        f"(default {PAGE_TITLE!r})")
    ap.add_argument("--header", type=Path, help="an HTML file with a site's own header, shown in place of the page's "
                                                "title bar (its <style> blocks go into the page's head)")
    ap.add_argument("--dictionary-job", type=Path, help="enable local dictionary editing for this upload")
    a = ap.parse_args(argv)
    DICTIONARY_JOB = a.dictionary_job.resolve() if a.dictionary_job else None
    HERE = (a.board / "qa").resolve()
    MP4_DIR = a.clips.resolve()
    COMPARE_DIR = (a.board / "compare").resolve()
    HANDS_DIR = (a.board / "hands").resolve()
    KEYPOINTS_DIR = (a.board / "hand_keypoints").resolve()
    SENSORS_DIR = (a.board / "sensors").resolve()
    FOOTAGE_DIR = Path(os.environ.get("BOARD_FOOTAGE_DIR") or (a.board / "footage")).resolve()
    PORT, PAGE_TITLE, BOARD_NAME, HEADER = a.port, a.title, board_name(a.board), read_header(a.header)
    if _FRAME_DIR is not None:
        _trim_disk_cache(_FRAME_DIR, '.jpg', max_bytes=_FRAME_DISK_BYTES, max_files=_FRAME_DISK_FILES)
    _trim_disk_cache(FOOTAGE_DIR, '.mp4', max_bytes=_FOOTAGE_DISK_BYTES, max_files=_FOOTAGE_DISK_FILES)
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    # socketserver's default listen backlog is 5: a burst of visitors (or a proxy opening many connections at once)
    # overflows it and those connections wait on SYN retries for up to a minute
    socketserver.ThreadingTCPServer.request_queue_size = 128
    socketserver.ThreadingTCPServer.daemon_threads = True
    with socketserver.ThreadingTCPServer(("127.0.0.1" if DICTIONARY_JOB else a.host, PORT), Handler) as httpd:
        print(f"board at http://localhost:{PORT} ({len(list_episodes())} episodes from {HERE}, clips from {MP4_DIR})",
              flush=True)
        httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
