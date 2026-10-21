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
import errno
import gzip
import hashlib
import html
import http.server
import json
import logging
import math
import os
import re
import selectors
import shutil
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager, ExitStack
from functools import lru_cache, wraps
import urllib.parse
import uuid
from collections import OrderedDict, deque
from pathlib import Path

from board.clips import DEPTH, clip_path    # one rule for where a camera's clip is, for the writer and this server
from board.families import Families
from board.public_export import project as public_json
from board.to_board import dumps, finite
from prepare import tools
from compare.metrics import model_names, reasoning_effort
from label.atomic import (cached_while_unchanged, file_version, remove_if, small_file, sweep_removals,
                          write_atomic)


# the ffmpeg board/clips.py cuts the clips with (prepare/tools.py), so the frames come from the same binary; None
# when there is none, and the board then serves no frames or footage
FFMPEG = tools.ffmpeg_or_none()


# Frames cut from the clips (goal frames, per-task goal frames, posters) are asked for again and again, and each
# miss runs ffmpeg, so they are kept in a bounded LRU cache keyed by (clip, mtime, t, width). The server is
# threaded, so the cache has a lock, and a semaphore bounds the ffmpeg processes a burst of misses can start.
_FRAME_CACHE = OrderedDict()
_FRAME_CACHE_MAX = int(os.environ.get("BOARD_FRAME_CACHE", 800))
_FRAME_LOCK = threading.Lock()
_FFMPEG_SEM = threading.Semaphore(int(os.environ.get("BOARD_FFMPEG_CONCURRENCY", 4)))
# BOARD_FRAME_DIR retains a bounded disk cache. File metadata invalidates replaced source clips.
_FRAME_DIR = Path(os.environ["BOARD_FRAME_DIR"]) if os.environ.get("BOARD_FRAME_DIR") else None


# Per-process memos of what ffprobe read from a clip, keyed by the clip's stat (_clip_key) and kept only when the clip
# still has that stat once the read is over (cached_while_unchanged): each stands for exactly the version it was read
# from, and a clip replaced during the read is read again.
def _clip_key(mp4: Path) -> tuple:
    st = mp4.stat()
    return str(mp4), st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino


def _still_clip(mp4, key: tuple) -> bool:
    try:
        return _clip_key(Path(mp4)) == tuple(key[:5])
    except OSError:
        return False


@cached_while_unchanged(2048, lambda *key: _still_clip(key[0], key))
def _read_clip_times(*key) -> list:
    try:
        r = subprocess.run([tools.ffprobe(), "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "packet=pts_time,flags", "-of", "csv=p=0", key[0]], capture_output=True, text=True,
                           timeout=60)
        return sorted(float(a) for a, *f in (ln.split(",") for ln in r.stdout.split() if ln)
                      if a not in ("", "N/A") and "D" not in "".join(f))
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


def _clip_times(mp4: Path) -> list:
    """The clip's frame times on the episode clock (sorted presentation times), read once per clip
    version; [] when they cannot be read."""
    try:
        return _read_clip_times(*_clip_key(mp4))
    except OSError:
        return []


@cached_while_unchanged(2048, lambda *key: _still_clip(key[0], key))
def _read_clip_comment(*key) -> str:
    # an ffprobe that cannot run raises, so its failure is never kept as the clip's answer
    r = subprocess.run([tools.ffprobe(), "-v", "error", "-show_entries", "format_tags=comment", "-of",
                        "default=nw=1:nk=1", key[0]], capture_output=True, text=True, timeout=60)
    return r.stdout.strip()


def _shown_from_halfway(mp4: Path) -> bool:
    """Whether the clip's frames come on screen halfway after the capture before each (board/clips.py retime, which
    sets HALFWAY_TAG as the clip's comment), so the frame a time shows is the one on screen then."""
    from board.clips import HALFWAY_TAG
    try:
        return _read_clip_comment(*_clip_key(mp4)) == HALFWAY_TAG
    except (OSError, subprocess.SubprocessError):
        return False


class CacheBusy(RuntimeError):
    pass


class FootageCapacityError(ValueError):
    pass


class _VerifiedFootage(tuple):
    def __new__(cls, out, name, record):
        result = super().__new__(cls, (out, name))
        result.receipt = _stored_record(record)
        return result


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


def _trim_disk_cache(folder: Path, suffix: str, *, max_bytes: int, max_files: int, protect=None, last=()):
    """Remove the least recently used cache files until the folder is within its budgets. Files in `protect` (a path or
    a collection of them) are never removed, and files in `last` only once nothing else is left to remove.

    Other server processes can share the folder, so the listing is only a plan: each file is removed through
    remove_if, only while it is still the version listed and nothing has used it since."""
    protect = {protect} if isinstance(protect, Path) else set(protect or ())
    last = set(last)
    with _DISK_CACHE_LOCK:
        entries = []
        sweep_removals(folder)
        temporary = re.compile(r'(?:[a-f0-9]{40}\.[0-9]+\.[0-9]+\.part|'
                               r'[a-f0-9]{20}\.[0-9]+\.[0-9]+(?:\.[a-f0-9]{32})?\.part(?:\.len)?\.mp4)')
        for path in folder.glob('*.part*'):
            if not temporary.fullmatch(path.name):
                continue
            try:
                if path.stat().st_mtime < time.time() - 600:
                    if suffix == '.mp4':
                        guard = _temporary_guard(path)
                        try:
                            with _file_lock(guard, blocking=False):
                                path.unlink(missing_ok=True)
                            guard.unlink(missing_ok=True)
                        except CacheBusy:
                            continue
                    else:
                        remove_if(path, lambda version: version[1] < time.time_ns() - 600 * 10**9)
            except OSError:
                continue
        if suffix == '.mp4':
            for guard in folder.glob('*.part.mp4.lock'):
                original = guard.with_name(guard.name[:-5])
                if not temporary.fullmatch(original.name):
                    continue
                try:
                    if guard.stat().st_mtime < time.time() - 600 and not original.exists():
                        with _file_lock(guard, blocking=False):
                            pass
                        guard.unlink(missing_ok=True)
                except CacheBusy:
                    pass
                except OSError as error:
                    _maintenance_error(error)
        length = {'.jpg': 40, '.mp4': 20}.get(suffix)
        if length is None:
            raise ValueError('Unknown media cache suffix')
        owned = re.compile(r'[a-f0-9]{' + str(length) + '}' + re.escape(suffix))
        for path in folder.glob('*' + suffix):
            if not owned.fullmatch(path.name):
                continue
            try:
                version = file_version(path)
                entries.append((_cache_recency(path, suffix, version), path, version))
            except OSError:
                continue
        total, count = sum(x[2][0] for x in entries), len(entries)
        for age, path, version in sorted(entries, key=lambda entry: (entry[1] in last, entry[0])):
            if total <= max_bytes and count <= max_files:
                break
            if path in protect:
                continue
            try:
                if remove_if(path, lambda now, listed=version, age=age, path=path:
                             now == listed and _cache_recency(path, suffix, now) <= age):
                    total -= version[0]
                    count -= 1
            except OSError:
                continue


def _cache_recency(path, suffix, version):
    return _footage_recency(path, version) if suffix == '.mp4' else version[1]


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
        key = (*_clip_key(mp4), float(t), int(max_w))
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
    if out and not _still_clip(mp4, key):
        return out                # cut from a clip replaced meanwhile: right for this request, kept for no other
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


# any other camera the recording has, and a camera the model is not shown (board/clips.py unshown_views)
EXTRA_CAM = re.compile(r"(?:extra|unshown)\d{1,2}")


def depth_clips(clips: Path, eid: str) -> dict:
    """Available paired and independent depth clips, keyed by their prepared stream view: every folder board/clips.py
    depth_clip writes, depth_<view>/."""
    if not clips.is_dir():
        return {}
    found = {folder.name[len(DEPTH):]: clip_path(clips, eid, folder.name) for folder in sorted(clips.iterdir())
             if folder.is_dir() and folder.name.startswith(DEPTH) and len(folder.name) > len(DEPTH)}
    return {view: path for view, path in found.items() if path.is_file()}


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
FOOTAGE_TAG = f"footage-v5-h264-crf20-veryfast-{FOOTAGE_FPS}fps-gap{FOOTAGE_GAP}"   # a new recipe makes new files
_FOOTAGE_SEM = threading.BoundedSemaphore(max(1, int(os.environ.get("BOARD_FOOTAGE_CONCURRENCY", 1))))
_FOOTAGE_PROBE_SEM = threading.BoundedSemaphore(max(1, int(os.environ.get('BOARD_FOOTAGE_CONCURRENCY', 1))))
_PROBE_ACTIVE = set()
_PROBE_ACTIVE_LOCK = threading.Lock()
_FOOTAGE_READY_LOCK = threading.Lock()
_FOOTAGE_MEMORY_LOCK = threading.Lock()
_FOOTAGE_RECORDS = OrderedDict()
_FOOTAGE_ACCESS = OrderedDict()
_FOOTAGE_TICKET_SECONDS = 3600
_MEDIA_READERS = threading.local()
_FOOTAGE_THREADS = int(os.environ.get("BOARD_FOOTAGE_THREADS", 2))
_FOOTAGE_STALL_TIMEOUT = max(1.0, float(os.environ.get('BOARD_FOOTAGE_STALL_TIMEOUT', 300)))
_EXPORT_SEM = threading.BoundedSemaphore(2)
_EXPORT_MAX_BYTES = 256 * 1024**2
_EXPORT_LABEL_BYTES = 16 * 1024**2
_FOOTAGE_LOCKS: dict = {}
_FOOTAGE_LOCKS_LOCK = threading.Lock()


# label file name -> (its trusted version, the episode ID it held then). Labels are rewritten by other processes (a
# labelling run, a rebuild), so an entry stands only for the version it was read from: every lookup checks each file's
# version, reads again only the files that changed, and reads the file it picks once more before using it.
_LABEL_INDEX_LOCK = threading.Lock()
_LABEL_INDEX = {}
_LABEL_INDEX_ROOT = None


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
    """The current label of an episode: the file named after it, else the one whose episode ID it is, found through an
    index that reads only the labels whose version changed since this process last read them."""
    global _LABEL_INDEX_ROOT
    direct = HERE / f'{eid}.json'
    if _under(HERE, direct) and direct.is_file():
        try:
            label = json.loads(direct.read_text())
            if isinstance(label, dict) and ((label.get('_meta') or {}).get('episode_id') or direct.stem) == eid:
                return current_label(label, DICTIONARY_JOB, direct.name)
        except (OSError, ValueError, TypeError, AttributeError, RecursionError):
            pass
    with _LABEL_INDEX_LOCK:
        previous = dict(_LABEL_INDEX) if _LABEL_INDEX_ROOT == HERE else {}
        _LABEL_INDEX.clear()
        for path in sorted(HERE.glob('*.json')):
            if not _under(HERE, path):
                continue
            try:
                sig = _label_identity(path)
                old = previous.get(path.name)
                if old is not None and old[0] == sig:
                    _LABEL_INDEX[path.name] = old
                    continue
                label = json.loads(path.read_text())
                identity = ((label.get('_meta') or {}).get('episode_id') or path.stem) if isinstance(label, dict) else None
                _LABEL_INDEX[path.name] = (sig, identity)
            except (OSError, ValueError, TypeError, AttributeError, RecursionError):
                continue
        _LABEL_INDEX_ROOT = HERE
        path = next((HERE / name for name, (_, identity) in _LABEL_INDEX.items() if identity == eid), None)
    if path is not None:
        try:
            label = json.loads(path.read_text())
            if ((label.get('_meta') or {}).get('episode_id') or path.stem) == eid:
                return current_label(label, DICTIONARY_JOB, path.name)
        except (OSError, ValueError, TypeError, AttributeError, RecursionError):
            pass
    return {}


def footage_cams(clips: Path, eid: str, label: dict | None = None) -> list:
    """Playable cameras in the current episode's presentation order, with legacy discovery as fallback."""
    extra = sorted((d.name for d in clips.iterdir() if d.is_dir() and EXTRA_CAM.fullmatch(d.name)),
                   key=lambda n: (n.startswith("unshown"), int(re.sub(r"\D", "", n)))) if clips.is_dir() else []
    label = footage_label(eid) if label is None else label
    # the label's own cameras too, so a camera of any name (board/clips.py clip_path) is found where it was written
    named = [str(v) for v in label.get('camera_views') or [] if isinstance(v, str)]
    cams = [(c, clip_path(clips, eid, c)) for c in dict.fromkeys(("exo", "left", "right", *extra, *named))
            if clip_path(clips, eid, c).is_file()]
    unpaired = {row.get('view') for row in label.get('unshown_cameras') or []
                if isinstance(row, dict) and row.get('clock_alignment') == 'unverified'}
    cams = [(cam, path) for cam, path in cams if cam not in unpaired]
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


def _trusted_version(path, version) -> tuple:
    """The part of a file's stat that stands for its content: all of it (device, inode, size, modification and change
    times) where the filesystem keeps change times (static._durable_metadata), and only size and modification time on
    exFAT, NFS, SMB and FUSE, whose inodes and change times are not reliable. A receipt records this when the file is
    made, and requests compare only this against it: content is never hashed on a request, so an edit by hand that
    keeps a file's size and modification time is not seen, as on most file servers."""
    from board.static import _durable_metadata
    version = tuple(version)
    return version if _durable_metadata(str(path), version) else (version[2], version[3])


def _probe(p: Path) -> tuple:
    version = next((version for reader, version in getattr(_MEDIA_READERS, 'active', [])
                    if Path(reader.name) == p), None)
    return _probe_versioned(p, version or _footage_version(p))


def _probe_versioned(p: Path, version: tuple) -> tuple:
    return _cached_probe_versioned(p, _trusted_version(p, version))


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
    # Cache hits bypass admission; cold probes share its bound.
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
    r = subprocess.run([tools.ffprobe(), "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,avg_frame_rate:format=duration",
                        "-of", "json", str(p)], capture_output=True, text=True, timeout=30, check=True,
                       **_source_process_options())
    j = json.loads(r.stdout)
    s = j["streams"][0]
    import av
    with av.open(str(p)) as c:
        from label.presented import presented
        pk = [(float(x.pts * x.time_base), float(x.duration * x.time_base)) for x in presented(c, c.streams.video[0])]
    # Some FFprobe builds omit packet lengths and stop their container duration at the final frame's start.
    dur = max(t + d for t, d in pk) if pk else float(j["format"]["duration"])
    num, _, den = str(s.get("avg_frame_rate") or "0/1").partition("/")
    rate = float(num) / float(den) if den and float(den) else 0.0
    steps = sorted(b[0] - a[0] for a, b in zip(sorted(pk), sorted(pk)[1:]) if b[0] > a[0])
    if steps:
        rate = round(1.0 / steps[len(steps) // 2], 3)
    return int(s["width"]), int(s["height"]), dur, rate


def _footage_seek(clip: Path, t0: float) -> float:
    """Seek relative to this camera, retaining its final GOP for EOF holds."""
    if t0 <= 0:
        return 0.0  # A whole-recording export reads from the native beginning without another probe.
    import av
    with av.open(str(_source_path(clip))) as container:
        stream = container.streams.video[0]
        if stream.start_time is None or stream.duration is None:
            return 0.0
        start = float(stream.start_time * stream.time_base)
        duration = float(stream.duration * stream.time_base)
    return max(0.0, min(t0 - start - 3.0, duration - 3.0))


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
        cmd += ["-ss", f"{_footage_seek(clip, t0):.3f}", "-i", str(clip)]
    g = [f"color=c=black:s={W}x{H}:r={fps}:d={t1 - t0:.3f},settb=AVTB,setpts=PTS+{t0:.3f}/TB[b0]"]
    # a camera keeps its last frame on to t1, as the page's video does once it has played to its end: the overlay
    # dropped a camera to black from its last frame's start (a 15 fps camera's last 1/30 s), so each is held there
    # (tpad), and the file keeps the canvas's own frames (those before t1), which the held frames would otherwise run past
    hold = f"tpad=stop_mode=clone:stop_duration={t1 + 1:.3f}"
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


def _source_path(path):
    if os.name == 'posix':
        for reader, _ in getattr(_MEDIA_READERS, 'active', []):
            if Path(reader.name) == path:
                return Path('/dev/fd') / str(reader.fileno())
    return path


def _source_process_options():
    readers = getattr(_MEDIA_READERS, 'active', [])
    return {'pass_fds': tuple(reader.fileno() for reader, _ in readers)} if os.name == 'posix' and readers else {}


def _validate_active_sources():
    # An open descriptor binds pathname replacements. A source edited in place while it is encoded changes its trusted
    # stat (_trusted_version), so the output is not published.
    for reader, version in getattr(_MEDIA_READERS, 'active', []):
        stat = os.fstat(reader.fileno())
        current = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if _trusted_version(reader.name, current) != _trusted_version(reader.name, version):
            raise ValueError('Media changed during footage preparation')


@contextmanager
def _footage_sources(cams):
    """Hold each source open for the probe and the encode, and name it by its trusted stat (_trusted_version).

    POSIX probe and encode children read these same descriptors, so a path replaced meanwhile cannot switch the bytes.
    A source is never hashed: its name in the output's key is its stat, so a prepared span is found from stats alone
    and never waits for an encode of anything else.
    """
    if len(cams) > 64:
        raise ValueError('Too many footage cameras')
    with ExitStack() as stack:
        previous = getattr(_MEDIA_READERS, 'active', [])
        active, identities = [], []
        for _, path in cams:
            reader = stack.enter_context(path.open('rb'))
            stat = os.fstat(reader.fileno())
            version = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            active.append((reader, version))
            trusted = _trusted_version(path, version)
            identities.append([str(path), *trusted] if len(trusted) > 2 else [str(path), 'size-mtime', *trusted])
        _MEDIA_READERS.active = active
        try:
            yield list(cams), identities
        finally:
            _MEDIA_READERS.active = previous


@contextmanager
def _publication_lock(key):
    """A fixed set of cross-process locks protects short output transactions."""
    with _file_lock(FOOTAGE_DIR / '.locks' / f'{int(key[:2], 16) % 64:02x}'):
        yield


def _maintenance_error(error):
    logging.getLogger(__name__).warning('Footage cache bookkeeping deferred: %s', error)


def _footage_recency(out, version):
    """When the movie at `out`, at `version` (file_version), was last used: its own modification time, the shared
    receipt's (which every server touches on use, _touch_footage) and this process's own record of using that version,
    never of an earlier version at the same name."""
    with _FOOTAGE_MEMORY_LOCK:
        age = _FOOTAGE_ACCESS.get((str(out), tuple(version)), version[1])
    try:
        age = max(age, (FOOTAGE_DIR / '.ready' / f'{out.stem}.json').stat().st_mtime_ns)
    except OSError:
        pass
    return max(age, version[1])


def _touch_footage(out):
    try:
        key = (str(out), file_version(out))
    except OSError:
        return
    with _FOOTAGE_MEMORY_LOCK:
        _FOOTAGE_ACCESS[key] = time.time_ns()
        _FOOTAGE_ACCESS.move_to_end(key)
        while len(_FOOTAGE_ACCESS) > max(128, _FOOTAGE_DISK_FILES):
            _FOOTAGE_ACCESS.popitem(last=False)
    try:
        os.utime(FOOTAGE_DIR / '.ready' / f'{out.stem}.json', None)
    except FileNotFoundError:
        pass
    except OSError as error:
        _maintenance_error(error)


def _stored_record(record: dict) -> dict:
    """A receipt in the shape its JSON file holds (a stat is a list, never a tuple). The memory copy, the returned
    receipt and the file are all this one shape, so the validator (_receipt_footage) cannot reject a record the server
    made itself only because it was never written and read back."""
    return json.loads(json.dumps(record))


def _retain_footage_record(out, record):
    with _FOOTAGE_MEMORY_LOCK:
        _FOOTAGE_RECORDS[str(out)] = _stored_record(record)
        _FOOTAGE_RECORDS.move_to_end(str(out))
        while len(_FOOTAGE_RECORDS) > max(128, _FOOTAGE_DISK_FILES):
            _FOOTAGE_RECORDS.popitem(last=False)


def _trim_footage_cache(protect=None, owner=None):
    try:
        with _file_lock(FOOTAGE_DIR / '.locks' / 'cache'):
            _trim_footage_files(protect, owner)
    except OSError as error:
        _maintenance_error(error)


def _trim_footage_files(protect, owner):
    """Trim prepared videos (with the cache lock held). A video a live download ticket names is kept, except that the
    client asking may lose its own oldest ones once nothing else is left: other clients' load never cancels a ticket."""
    held = _ticketed_footage()
    mine = {FOOTAGE_DIR / name for name, owners in held.items() if owners == {_ticket_owner(owner)}}
    others = {FOOTAGE_DIR / name for name in held} - mine
    _trim_disk_cache(FOOTAGE_DIR, '.mp4', max_bytes=_FOOTAGE_DISK_BYTES,
                     max_files=_FOOTAGE_DISK_FILES if _FOOTAGE_DISK_FILES > 0 else 2**63,
                     protect=others | ({protect} if protect is not None else set()), last=mine)


def _reserve_footage(owner, out: Path, size: int):
    """Refuse to keep `out` (`size` bytes) unless it fits in the cache budget beside the videos other clients' live
    tickets name. Those videos are never evicted (_trim_footage_files), so the budget is a reservation: the video about
    to be kept counts against it before it is kept, and one that does not fit is refused (503, retried later) instead
    of growing the cache past its limit."""
    held = _ticketed_footage()
    sizes = []
    for name, owners in held.items():
        if owners != {_ticket_owner(owner)} and name != out.name:
            try:
                sizes.append((FOOTAGE_DIR / name).stat().st_size)
            except OSError:
                continue
    if sum(sizes) + size > _FOOTAGE_DISK_BYTES or (_FOOTAGE_DISK_FILES > 0 and len(sizes) + 1 > _FOOTAGE_DISK_FILES):
        raise CacheBusy('Prepared video storage is full of downloads in progress; try again later.')


def _footage_room(owner, out: Path):
    """Refuse to start an encode that cannot be kept. Before encoding, the video's size is unknown but at least one
    byte; publication reserves its real size (_encode_and_publish)."""
    _reserve_footage(owner, out, 1)


def _receipt_guard(path):
    return FOOTAGE_DIR / '.locks' / f'receipt-{int(path.stem[:2], 16) % 64:02x}'


def _write_footage_record(path, record):
    with _file_lock(_receipt_guard(path)):
        write_atomic(path, record)


def _trim_receipt_temps(folder):
    # write_atomic names each writer's temporary file .<receipt>.<random>.tmp (older ones .<receipt>.tmp); a writer
    # holds the receipt's guard while its temporary file exists, so one whose guard is free is left by a dead writer
    for temporary in folder.glob('.*.tmp'):
        match = re.fullmatch(r'\.((?:[a-f0-9]{20}|[a-f0-9]{32}|[a-f0-9]{40})\.json)(?:\.[a-f0-9]{32})?\.tmp',
                             temporary.name)
        if match is None:
            continue
        try:
            with _file_lock(_receipt_guard(folder / match[1]), blocking=False):
                temporary.unlink(missing_ok=True)
        except CacheBusy:
            pass
        except OSError as error:
            _maintenance_error(error)


def _temporary_guard(path):
    return path.with_name(path.name.replace('.part.len.mp4', '.part.mp4') + '.lock')


def _footage_receipt(result: tuple) -> dict:
    """The receipt of a prepared video, written when it is made: its stat, which every later request compares
    (_receipt_footage)."""
    out, name = result
    version = _footage_version(out)
    if out.is_symlink() or not 0 < version[2] <= _FOOTAGE_DISK_BYTES:
        raise FootageCapacityError('Prepared footage exceeds the disk cache budget')
    return _stored_record({'format': 'argus-footage-ready/2', 'file': out.name, 'name': name, 'version': version})


def _discard_changed_footage(out: Path) -> None:
    try:
        out.unlink(missing_ok=True)
    except OSError as error:
        raise RuntimeError('Prepared footage changed and could not be discarded') from error


def _receipt_footage(record: dict, *, reader=None) -> tuple | None:
    if (record.get('format') != 'argus-footage-ready/2' or
            not re.fullmatch(r'[a-f0-9]{20}\.mp4', record['file']) or
            not isinstance(record['name'], str) or Path(record['name']).name != record['name']):
        return None
    out = FOOTAGE_DIR / record['file']
    version = _footage_version(out)
    trusted = _trusted_version(out, version)
    if (out.is_symlink() or not isinstance(record['version'], list) or len(record['version']) != 5
            or trusted != _trusted_version(out, record['version']) or not 0 < version[2] <= _FOOTAGE_DISK_BYTES):
        return None
    if reader is not None:
        stat = os.fstat(reader.fileno())
        opened_version = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if _trusted_version(out, opened_version) != trusted:
            return None
    _touch_footage(out)
    return _VerifiedFootage(out, record['name'], record)


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
    with _FOOTAGE_MEMORY_LOCK:
        record = _FOOTAGE_RECORDS.get(str(FOOTAGE_DIR / (key + '.mp4')))
    if record is not None:
        try:
            result = _receipt_footage(record)
            if result is not None:
                return record, result
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            pass
    # A surviving ticket is also a durable receipt for this exact output,
    # including when a canonical record is malformed or no longer validates.
    with _FOOTAGE_READY_LOCK:
        tokens = [token for token, entry in _ticket_index().items() if entry[1] == key + '.mp4']
    for token in tokens:
        try:
            record = _ticket_record(token)
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


def _snapshot_record(snapshot) -> dict | None:
    """The JSON object a receipt holds, from small_file's read of it, or None when it holds none (a symbolic link, a
    file over 4096 bytes, or anything but a JSON object)."""
    try:
        record = json.loads(snapshot[2]) if snapshot[2] is not None else None
    except (ValueError, RecursionError):
        return None
    return record if isinstance(record, dict) else None


_read_receipt_snapshot = small_file(4096)


def _receipt_lapsed(snapshot, expire=False) -> bool:
    """Whether a receipt or alias, as just read, no longer stands: it names no prepared movie that is there, or, for a
    ticket (`expire`), its expiry has passed."""
    record = _snapshot_record(snapshot)
    filename = record.get('file') if record else None
    if record and record.get('format') == 'argus-footage-alias/1':
        filename = str(record.get('key')) + '.mp4'
    if not (isinstance(filename, str) and re.fullmatch(r'[a-f0-9]{20}\.mp4', filename)
            and (FOOTAGE_DIR / filename).is_file()):
        return True
    return expire and _ticket_lapsed(snapshot)


def _remove_receipt(path, holds) -> bool:
    """Remove a receipt, alias or ticket only when `holds` is true for it as it is now, under the lock its writers hold
    (_write_footage_record), so a version another server wrote since this one read it is never the one removed."""
    return remove_if(path, holds, read=_read_receipt_snapshot, lock=lambda: _file_lock(_receipt_guard(path)))


def _trim_footage_receipts(folder: Path, pattern: str, *, expire=False):
    _trim_receipt_temps(folder)
    sweep_removals(folder)
    retained = []
    for path in folder.glob('*.json'):
        if not re.fullmatch(pattern, path.name):
            continue
        try:
            snapshot = _read_receipt_snapshot(path)
            if _receipt_lapsed(snapshot, expire):
                _remove_receipt(path, lambda now: _receipt_lapsed(now, expire))
            elif not expire and len(path.stem) == 40:
                retained.append((snapshot[1], path, snapshot))
        except FileNotFoundError:
            pass
        except OSError as error:
            _maintenance_error(error)
    # Canonical records follow movie lifetime and request aliases have a finite bound. Tickets are bounded when they
    # are issued, per client (prepare_footage_ticket), so no other client's load trims a live one here. An alias is
    # trimmed as one of the oldest only while it is still the version judged old: one rewritten since is in use.
    retained.sort(key=lambda entry: entry[0], reverse=True)
    for _, path, snapshot in retained[max(128, 2 * _FOOTAGE_DISK_FILES):]:
        try:
            _remove_receipt(path, lambda now, seen=snapshot: now == seen)
        except OSError as error:
            _maintenance_error(error)


def _remember_ready_footage(request: str, result: tuple, record=None) -> None:
    folder = FOOTAGE_DIR / '.ready'
    if len(request) == 20:
        record = record or _footage_receipt(result)
        _retain_footage_record(result[0], record)
    else:
        record = {'format': 'argus-footage-alias/1', 'key': result[0].stem}
    # Durable aliases accelerate reuse; their failure must not invalidate media
    # whose bytes and source identity have already been verified.
    try:
        _write_footage_record(folder / f'{request}.json', record)
        with _FOOTAGE_READY_LOCK:
            _trim_footage_receipts(folder, r'(?:[a-f0-9]{20}|[a-f0-9]{40})\.json')
    except (OSError, ValueError) as error:
        _maintenance_error(error)


_TICKET_CLIENT_MAX = 8      # live download tickets one client holds; another withdraws that client's oldest
_TICKET_MAX = 4096          # live tickets of all clients; past this a prepare is refused, never another's withdrawn
# FOOTAGE_DIR -> {token: (owner, file, version, expires_at, issued_at, stamp)}. Other server processes renew, withdraw
# and expire the same ticket files, so an entry stands only for the file at `stamp` (size, modification time and inode,
# taken before it was read): a file whose stamp differs is read again, and an expiry is acted on only through
# _remove_receipt, which reads the file again and keeps it when it was renewed.
_TICKETS = {}


def _ticket_owner(client) -> str:
    """The name a ticket records for the client that asked for it (client_key), not the address itself."""
    return hashlib.sha256(str(client).encode()).hexdigest()[:16] if client else ''


def _ticket_fields(record) -> tuple:
    """(owner, file, version, expires_at, issued_at) of a ticket record; an unreadable one expires at 0, so it is
    removed like an expired one."""
    expiry = record.get('expires_at') if record else None
    if type(expiry) not in (int, float) or not math.isfinite(expiry) or not isinstance(record.get('file'), str):
        return '', '', None, 0.0, 0.0
    issued = record.get('issued_at')
    return (str(record.get('owner') or ''), record['file'], record.get('version'), float(expiry),
            float(issued) if type(issued) in (int, float) else float(expiry) - _FOOTAGE_TICKET_SECONDS)


def _ticket_lapsed(snapshot) -> bool:
    return _ticket_fields(_snapshot_record(snapshot))[3] < time.time()


def _ticket_stamp(st) -> tuple:
    return st.st_size, st.st_mtime_ns, st.st_ino


def _read_ticket(index, folder, token, stamp=None) -> None:
    """Read one ticket file into the index as it is now (or drop it when it is gone). The stamp is taken before the
    read, so a write in between costs one more read later and never leaves an entry newer than its stamp says."""
    path = folder / f'{token}.json'
    try:
        stamp = stamp or _ticket_stamp(os.lstat(path))
        index[token] = (*_ticket_fields(_snapshot_record(_read_receipt_snapshot(path))), stamp)
    except FileNotFoundError:
        index.pop(token, None)
    except OSError as error:
        _maintenance_error(error)
        index.pop(token, None)


def _ticket_index() -> dict:
    """The live tickets in FOOTAGE_DIR/.tickets (with _FOOTAGE_READY_LOCK held). A ticket file is read again whenever
    its stamp changed since this process read it, so a renewal by another server is seen; expired ones are removed
    here, each only when it is still expired on a fresh read."""
    folder = FOOTAGE_DIR / '.tickets'
    index = _TICKETS.setdefault(str(FOOTAGE_DIR), {})
    stamps = {}
    try:
        entries = list(os.scandir(folder))
    except OSError:
        entries = []
    for entry in entries:
        if re.fullmatch(r'[a-f0-9]{32}\.json', entry.name):
            try:
                stamps[entry.name[:-5]] = _ticket_stamp(entry.stat(follow_symlinks=False))
            except OSError:
                continue
    for token in set(index) - set(stamps):
        del index[token]
    for token, stamp in stamps.items():
        if token not in index or index[token][5] != stamp:
            _read_ticket(index, folder, token, stamp)
    now = time.time()
    for token, entry in list(index.items()):
        if entry[3] < now:
            try:
                if _remove_receipt(folder / f'{token}.json', _ticket_lapsed):
                    del index[token]
                else:
                    _read_ticket(index, folder, token)      # renewed since it was read: keep its new expiry
            except OSError as error:
                _maintenance_error(error)
    return index


def _ticketed_footage() -> dict:
    """{video file name: owners of the live tickets that name it}."""
    held = {}
    with _FOOTAGE_READY_LOCK:
        for entry in _ticket_index().values():
            held.setdefault(entry[1], set()).add(entry[0])
    return held


def prepare_footage_ticket(result: tuple, client=None, *, reuse: bool = False) -> str:
    """A download ticket for a prepared video, owned by `client` (client_key). With `reuse`, the client's own live
    ticket for the same video is renewed for a full lifetime and returned instead. A client holds at most
    _TICKET_CLIENT_MAX tickets, and its own oldest is withdrawn past that; other clients' load never withdraws one.
    Past _TICKET_MAX live tickets in all, a new ticket is refused."""
    record = getattr(result, 'receipt', None)
    if record is None:
        record, _ = _cached_footage_record(result[0].stem)
    record = dict(record) if record is not None else None
    try:
        version = _footage_version(result[0])
    except FileNotFoundError as error:
        raise CacheBusy('Prepared video was evicted; prepare it again.') from error
    if record is None or list(version) != list(record.get('version') or []):
        raise ValueError('Prepared footage changed before ticket publication')
    owner, now = _ticket_owner(client), time.time()
    record.update(owner=owner, issued_at=now, expires_at=now + _FOOTAGE_TICKET_SECONDS)
    # A ticket protects its video from eviction, so issuing one is a reservation too. It holds the cache lock that
    # publication and trimming hold, so a trim cannot remove the video between this check and the ticket's write.
    with _file_lock(FOOTAGE_DIR / '.locks' / 'cache'):
        try:
            current = _footage_version(result[0])
        except FileNotFoundError as error:
            raise CacheBusy('Prepared video was evicted; prepare it again.') from error
        if list(current) != list(version):
            raise ValueError('Prepared footage changed before ticket publication')
        _reserve_footage(client, result[0], version[2])
        return _issue_footage_ticket(result, record, owner, version, now, reuse)


def _issue_footage_ticket(result, record, owner, version, now, reuse):
    with _FOOTAGE_READY_LOCK:
        folder = FOOTAGE_DIR / '.tickets'
        index = _ticket_index()
        token = next((token for token, entry in index.items() if reuse and entry[0] == owner
                      and entry[1] == result[0].name and entry[2] == list(version)), None)
        if token is None:
            mine = sorted((entry[4], token) for token, entry in index.items() if entry[0] == owner)
            excess = len(mine) - _TICKET_CLIENT_MAX + 1
            for issued, old in mine:
                if excess <= 0:
                    break
                # withdrawn only while it is still the ticket judged oldest: one renewed since is not, and stays
                if not _remove_receipt(folder / f'{old}.json',
                                       lambda now, issued=issued: _ticket_fields(_snapshot_record(now))[4] == issued):
                    _read_ticket(index, folder, old)
                    if old in index:
                        continue
                index.pop(old, None)
                excess -= 1
            if len(index) >= _TICKET_MAX:
                raise CacheBusy('Too many downloads are prepared at once; try again later.')
            token = uuid.uuid4().hex
        _write_footage_record(folder / f'{token}.json', record)
        _read_ticket(index, folder, token)
        _trim_receipt_temps(folder)
    return token


def _ticket_record(token: str) -> dict | None:
    if not re.fullmatch(r'[a-f0-9]{32}', token):
        return None
    receipt = FOOTAGE_DIR / '.tickets' / f'{token}.json'
    for _ in range(2):
        record = _read_footage_receipt(receipt)
        if record is None:
            return None
        expiry = record.get('expires_at')
        if type(expiry) in (int, float) and math.isfinite(expiry) and expiry >= time.time():
            return record
        # expired as read: removed only if it is still expired now, and read again when another server renewed it
        if _remove_receipt(receipt, _ticket_lapsed):
            return None
    return None


@contextmanager
def _ticket_reader(token):
    try:
        record = _ticket_record(token)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        record = None
    with _footage_reader(record) as opened:
        yield opened


@contextmanager
def _footage_reader(record):
    """(verified movie, open file) for a receipt, or (None, None). The file is opened first and its open descriptor is
    checked against the receipt, then the same descriptor is sent, so another server evicting or replacing the movie
    after it was verified can neither switch the bytes nor make this send a file the receipt does not name."""
    reader, result = None, None
    try:
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


def footage(eid: str, t0: float = 0.0, t1: float | None = None, client=None) -> tuple | None:
    """Prepare the current sources; tickets separately serve the exact prepared output."""
    if not math.isfinite(t0) or (t1 is not None and not math.isfinite(t1)):
        raise ValueError('Footage times must be finite')
    label = footage_label(eid)
    cams = footage_cams(MP4_DIR, eid, label)
    if not cams or not FFMPEG:
        return None
    if _FOOTAGE_DISK_BYTES <= 0:
        raise FootageCapacityError('Prepared footage disk cache has no capacity')
    with _footage_sources(cams) as (working, identities):
        return _footage_from_sources(eid, t0, t1, label, working, identities, client)


def _footage_from_sources(eid, t0, t1, label, cams, identities, client=None):
    request_parts = [eid, float(t0), None if t1 is None else float(t1), FOOTAGE_TAG,
                     label.get('duration_s'), label.get('camera_views'), identities]
    request = hashlib.sha1(json.dumps(request_parts).encode()).hexdigest()
    ready = _ready_footage(request)
    if ready is not None:
        _trim_footage_cache(ready[0], client)
        return ready
    probes = [_probe(path) for _, path in cams]
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
        _footage_room(client, out)
        result = _uncached_footage(key, out, name, cams, probes, t0, t1, client)
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
                        # the stall is the error; the poll's own timeout is how it was noticed
                        raise subprocess.TimeoutExpired(cmd, _FOOTAGE_STALL_TIMEOUT) from None
        except BaseException:
            process.kill()
            process.communicate()
            raise


def _uncached_footage(key, out, name, cams, probes, t0, t1, client=None):
    # An encode never waits for another: a request that finds every encoder busy is answered 503 and retries.
    if not _FOOTAGE_SEM.acquire(blocking=False):
        raise CacheBusy('Media preparation is busy; try again shortly')
    try:
        working = [(cam, _source_path(path)) for cam, path in cams]
        return _encode_and_publish(key, out, name, working, probes, t0, t1, client)
    finally:
        _FOOTAGE_SEM.release()


def _encode_and_publish(key, out, name, cams, probes, t0, t1, client=None):
    # A competing process may publish while this encoder runs. Publication adopts
    # its verified result instead of replacing a file held by a live ticket.
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.part.mp4")
    cmd = footage_command([(p, pr[:2], pr[3]) for (_, p), pr in zip(cams, probes)], t0, t1, tmp, _FOOTAGE_THREADS)
    cmd[-1:-1] = ['-fs', str(_FOOTAGE_DISK_BYTES)]
    nice = shutil.which('nice')
    guard = _temporary_guard(tmp)
    lease = _file_lock(guard)
    lease.__enter__()
    try:
        completed = _run_footage(([nice, '-n', '19'] if nice else []) + cmd)
        if completed.returncode != 0 or not tmp.is_file():
            if 'No space left on device' in (completed.stderr or '') or 'Disk quota exceeded' in (completed.stderr or ''):
                raise FootageCapacityError('Prepared footage storage is full')
            raise RuntimeError(f"the video could not be made: {(completed.stderr or '').strip()[-300:]}")
        if tmp.stat().st_size >= _FOOTAGE_DISK_BYTES:
            raise FootageCapacityError('Prepared footage exceeds the disk cache budget')
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
                _reserve_footage(client, out, record['version'][2])
                if out.exists():
                    _discard_changed_footage(out)
                os.replace(tmp, out)
                record = _stored_record({**record, 'file': out.name, 'version': _footage_version(out)})
                _remember_ready_footage(key, (out, name), record)
                _trim_footage_files(out, client)
        return _VerifiedFootage(out, name, record)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        finally:
            lease.__exit__(None, None, None)
            guard.unlink(missing_ok=True)


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
_LIST_REFRESHING = set()
_LIST_CHECKED = {}
_LIST_ROWS = {}
_LIST_REFRESH_INTERVAL = 1.0

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


def targets(board: Path) -> dict:
    """{dataset: {"episodes", "hours"}}: how much of each dataset a run in progress is meant to label, as its manifest
    entry gives it ("target": {"episodes": 2682, "hours": 101.9}). The page shows what is labelled against it; a
    dataset with no target, or a target that is not two positive numbers, shows only what is labelled."""
    try:
        entries = json.loads((board / "manifest.json").read_text()).get("datasets")
    except (OSError, ValueError, AttributeError):
        return {}
    out = {}
    for entry in entries if isinstance(entries, list) else []:
        target = entry.get("target") if isinstance(entry, dict) else None
        if not isinstance(target, dict) or not isinstance(entry.get("dataset"), str):
            continue
        values = [target.get(k) for k in ("episodes", "hours")]
        if all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values):
            out[entry["dataset"]] = {"episodes": int(values[0]), "hours": float(values[1])}
    return out


def board_version(board: Path) -> str:
    """Which build of the board this is, from BUILT.json (board/build.py writes it last, after the episode files are in
    place) and manifest.json (its targets): an open page asks for it to learn that the board was rebuilt, which costs
    two stats however many episodes the board holds."""
    parts = []
    for name in ("BUILT.json", "manifest.json"):
        try:
            st = (board / name).stat()
            parts.append(f"{st.st_mtime_ns}:{st.st_size}")
        except OSError:
            parts.append("-")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _label_identity(path):
    return _trusted_version(path, _footage_version(path))


def _opened_version(source) -> tuple:
    stat = os.fstat(source.fileno())
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _read_label(path):
    """Reuse bytes already read by an episode request to refresh its rail row."""
    with path.open('rb') as source:
        opened = _trusted_version(path, _opened_version(source))
        raw = source.read()
    label = json.loads(raw)
    key = str(path.parent)
    if isinstance(label, dict) and 'episode_prompt' in label:
        with _LIST_LOCK:
            known = key in _LIST_ROWS
        try:
            # the row stands for the file only while nothing replaced the bytes this request read
            known = known and _label_identity(path) == opened
        except OSError:
            known = False
        if known:
            row = (opened, _rail_record(path, label))
            with _LIST_LOCK:
                rows = _LIST_ROWS[key]
                rows[path.name] = row
                _LIST_CACHE[key] = (tuple((name, entry[0]) for name, entry in sorted(rows.items())),
                                    [entry[1] for name, entry in sorted(rows.items())])
    return label


def _refresh_weak_list(here: Path):
    """Read only the labels whose size or modification time changed, preserving last complete writer output."""
    key = str(here)
    with _LIST_LOCK:
        previous = dict(_LIST_ROWS.get(key, {}))
    rows = {}
    for path in sorted(here.glob('*.json')):
        try:
            if not _under(here, path):
                continue
            version = _label_identity(path)
            old = previous.get(path.name)
            if old and old[0] == version:
                rows[path.name] = old
                continue
            with path.open('rb') as source:
                opened = _trusted_version(path, _opened_version(source))
                raw = source.read()
            if opened != version or _label_identity(path) != version:
                raise ValueError('Label changed during snapshot')
            label = json.loads(raw)
            if isinstance(label, dict) and 'episode_prompt' in label:
                rows[path.name] = (version, _rail_record(path, label))
        except (OSError, ValueError, TypeError, AttributeError, RecursionError):
            if path.name in previous:
                rows[path.name] = previous[path.name]
    signature = tuple((name, row[0]) for name, row in rows.items())
    with _LIST_LOCK:
        hit = _LIST_CACHE.get(key)
        out = hit[1] if hit and hit[0] == signature else [row[1] for row in rows.values()]
        _LIST_ROWS[key] = rows
        _LIST_CACHE[key] = (signature, out)
        _LIST_CHECKED[key] = time.monotonic()
    return out


def _refresh_weak_list_worker(here):
    try:
        _refresh_weak_list(here)
    except OSError:
        pass
    finally:
        with _LIST_LOCK:
            _LIST_REFRESHING.discard(str(here))


def list_episodes(here: Path | None = None) -> list:
    """Serve complete rail snapshots; on a weak filesystem the scan runs off requests.

    Live weak-filesystem lists can lag writes by one refresh interval plus scan
    time, and see a label as changed when its size or modification time changes
    (_trusted_version). A malformed in-progress write retains its last complete row;
    a removed file disappears on the next complete scan. Episode exports read
    current files.
    """
    from board.static import _durable_metadata
    here = here or HERE
    key = str(here)
    try:
        versions = [(path, _footage_version(path)) for path in sorted(here.glob('*.json'))]
        weak = any(not _durable_metadata(str(path), version) for path, version in versions)
    except OSError:
        weak, versions = True, []
    if weak:
        with _LIST_LOCK:
            hit = _LIST_CACHE.get(key)
            if hit:
                if key not in _LIST_REFRESHING and time.monotonic()-_LIST_CHECKED.get(key, 0) >= _LIST_REFRESH_INTERVAL:
                    _LIST_REFRESHING.add(key)
                    threading.Thread(target=_refresh_weak_list_worker, args=(here,), daemon=True).start()
                return hit[1]
        return _refresh_weak_list(here)
    sig = tuple((path.name, version) for path, version in versions)
    with _LIST_LOCK:
        hit = _LIST_CACHE.get(key)
        if hit and hit[0] == sig:
            return hit[1]
    out = rail_records(here)
    with _LIST_LOCK:
        _LIST_CACHE[key] = (sig, out)
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
    return {"families": sorted(fams), "minor_families": sorted(minor - fams),
            "unrated_families": sorted(c["unrated"])}


CAPTURE_PREFIX = "cq:"


def capture_catalog() -> dict:
    """The page's entry for each capture check's family: its name as checks/capture_qc.py gives it."""
    from checks.capture_qc import NAMES
    return {CAPTURE_PREFIX + k: {"name": name, "list": "data", "check": True} for k, (name, _group) in NAMES.items()}


EXPORT_UPLOADER_VALUES = False      # --publish-uploader-values: downloads and exports carry uploader metadata values


def public_label(d: dict, *, export: bool = False) -> dict:
    """Copy public JSON after source checks and current human interpretation. A download or export leaves the board,
    so it carries uploader metadata values only when the operator published them (--publish-uploader-values)."""
    return public_json(d, uploader_values=EXPORT_UPLOADER_VALUES) if export else public_json(d)


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
        except (OSError, ValueError, TypeError, AttributeError, RecursionError):
            continue                  # not an episode file, or one too malformed to list
    return out


def _rail_record(p: Path, d: dict) -> dict:
    # Counted, minor and unrated issues retain separate totals in live and static rails.
    all_issues = [i for i in (d.get("data_issues") or []) if i and i.get("issue")]
    issues = [i for i in all_issues if FAMILIES.counts("data_issues", i)]
    top = max(issues, key=lambda i: SEV_RANK.get(str(i.get("severity") or "").lower(), 0), default=None)
    max_sev = str(top.get("severity") or "").lower() if top else None
    # operator mistakes: the recording is faithful but the demonstration went wrong. Summarised apart from data
    # issues so the rail never counts a fumble as a fault of the data.
    all_mistakes = [i for i in (d.get("operator_mistakes") or []) if i and i.get("issue")]
    mistakes = [i for i in all_mistakes if FAMILIES.counts("operator_mistakes", i)]
    unrated_issues = sum(not FAMILIES.rated(i) for i in all_issues)
    unrated_mistakes = sum(not FAMILIES.rated(i) for i in all_mistakes)
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
        "n_minor_issues": len(all_issues) - len(issues) - unrated_issues,
        "n_unrated_issues": unrated_issues,
        "n_mistakes": len(mistakes),
        "n_minor_mistakes": len(all_mistakes) - len(mistakes) - unrated_mistakes,
        "n_unrated_mistakes": unrated_mistakes,
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


@lru_cache(maxsize=1)
def machine_names() -> frozenset:
    """This machine's own host names, with and without their domain and as .local, which a board on a network address
    answers to."""
    names = set()
    for name in (socket.gethostname(), socket.getfqdn()):
        name = (name or '').lower().rstrip('.')
        if name:
            short = name.split('.')[0]
            names.update((name, short, short + '.local'))
    return frozenset(names)


def host_allowed(host: str, bound: str, allowed=()) -> bool:
    """Whether a board listening on `bound` answers to a request whose Host is `host`. Every board answers to localhost,
    the loopback addresses and the names given with --allow-host. A board on a network address also answers to any IP
    address, which DNS rebinding cannot forge since it needs a name, and to this machine's own names (machine_names); a
    loopback board answers to nothing else, since only a proxy on this machine forwards other names to it."""
    import ipaddress
    host = host.lower().rstrip('.')
    if host in ('localhost', '127.0.0.1', '::1') or host in {str(h).lower().rstrip('.') for h in allowed}:
        return True
    try:
        if ipaddress.ip_address(str(bound).split('%')[0]).is_loopback:
            return False
    except ValueError:
        pass
    try:
        ipaddress.ip_address(host.split('%')[0])
        return True
    except ValueError:
        return host in machine_names()


def media_tag(path: Path, version: tuple, *made) -> str:
    """An entity tag for a served clip, or for what was made from it (`made`: a frame's time, width and recipe): its
    trusted stat (_trusted_version), so a browser keeps unchanged media and revalidates it, and a replaced clip gets a
    new tag."""
    identity = repr((str(path), _trusted_version(path, version), made)).encode()
    return '"' + hashlib.sha1(identity).hexdigest()[:32] + '"'


REVALIDATE = 'no-cache'      # browsers may keep media, but ask each time whether it is still current


class Handler(http.server.BaseHTTPRequestHandler):
    # keep-alive: a page asks for its labels, posters and three videos at once, and a proxy in front (or a browser
    # that talks to the board directly) reuses one connection for them instead of opening one per request; an idle
    # connection is closed after `timeout` seconds
    protocol_version = "HTTP/1.1"
    timeout = 30
    request_read_timeout = 15

    def setup(self):
        # BoardHTTPServer hands over one request of a connection it holds, with the head it already received
        if isinstance(self.request, _HeldRequest):
            self.connection = self.request.sock
            self.rfile = self.request.reader
            self.wfile = socketserver._SocketWriter(self.connection)
            return
        super().setup()

    def handle(self):
        if isinstance(self.request, _HeldRequest):
            self.close_connection = True
            self.handle_one_request()
            self.request.close = self.close_connection
            return
        super().handle()

    def finish(self):
        if isinstance(self.request, _HeldRequest):
            return
        super().finish()

    def client(self) -> str:
        """Who this request comes from (client_key), for per-client bounds."""
        if isinstance(self.request, _HeldRequest):
            return self.request.client
        return client_key(str(self.client_address[0]) if self.client_address else '',
                          (self.headers.get_all('X-Forwarded-For') or [None])[-1] if self.headers else None)

    def _stop_read_deadline(self):
        deadline = getattr(self, '_read_deadline', None)
        if deadline is not None:
            deadline.cancel()
            self._read_deadline = None

    def parse_request(self):
        return super().parse_request() and not getattr(self, '_read_expired', False)

    def handle_one_request(self):
        self._read_expired = False
        def expire():
            self._read_expired = True
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self._read_deadline = threading.Timer(self.request_read_timeout, expire)
        self._read_deadline.daemon = True
        self._read_deadline.start()
        try:
            return super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        finally:
            self._stop_read_deadline()

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

    def _not_modified(self, tag: str) -> bool:
        """Answer 304 when the browser's copy carries this entity tag."""
        given = self.headers.get('If-None-Match')
        if not given or not ({t.strip().removeprefix('W/') for t in given.split(',')} & {tag, '*'}):
            return False
        self.send_response(304)
        self.send_header('ETag', tag)
        self.send_header('Cache-Control', REVALIDATE)
        self.end_headers()
        return True

    def _send_file(self, p: Path, ctype: str, download: str | None = None, *, source=None, validate=False):
        """Serve a file with proper HTTP Range support so the browser
        can stream/seek without downloading the whole thing first; download names the saved file. With validate, the
        response carries the file's entity tag (media_tag) so a browser revalidates its copy instead of fetching it
        again."""
        try:
            st = os.fstat(source.fileno()) if source is not None else p.stat()
        except OSError:
            self._send(404, {"error": "not found"})
            return
        size = st.st_size
        tag = media_tag(p, (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)) if validate else None
        if tag and self._not_modified(tag):
            return

        rng = self.headers.get("Range") or self.headers.get("range")
        if tag and rng and (self.headers.get('If-Range') or tag) != tag:
            rng = None          # the copy the browser is completing has changed: send the whole file
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
            except ValueError:            # a range header that does not parse or is out of bounds
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
            if tag:
                self.send_header('ETag', tag)
                self.send_header('Cache-Control', REVALIDATE)
            else:
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
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
            d = _read_label(p)
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

    def _browser_refusal(self) -> str | None:
        """Why a browser's request is refused, or None: another site's page asking (Sec-Fetch-Site), or a Host header
        this board does not answer to (host_allowed), which is how DNS rebinding reaches a board."""
        if self.headers.get('Sec-Fetch-Site') == 'cross-site':
            return 'Cross-site requests are disabled.'
        server = getattr(self, 'server', None)
        if server is None:
            return None
        raw = self.headers.get('Host') or ''
        try:
            host = urllib.parse.urlsplit('http://' + raw).hostname
        except ValueError:
            host = None
        if host and host_allowed(host, server.server_address[0], getattr(server, 'allowed_hosts', ())):
            return None
        shown = re.sub(r'[^A-Za-z0-9.:\[\]_-]', '', host or raw)[:100]
        if not shown:
            return 'The request names no host this board answers to.'
        return (f'This board does not answer to the host name {shown}. If it is reached by that name, for example '
                f'through a reverse proxy such as tailscale serve or Caddy, start it with --allow-host {shown}.')

    def do_POST(self):
        """Export unique episodes through a bounded spool before sending headers."""
        refusal = self._browser_refusal()
        if refusal:
            self.close_connection = True
            return self._send(403, {'error': refusal})
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
            self._stop_read_deadline()
            payload = json.loads(raw)
        except (ValueError, RecursionError):
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
                            label = finite(public_label(current_label(label, DICTIONARY_JOB, path.name), export=True))
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
                        except (OSError, ValueError, TypeError, AttributeError, RecursionError):
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
        self._stop_read_deadline()
        refusal = self._browser_refusal()
        if refusal:
            self.close_connection = True
            return self._send(403, {'error': refusal})
        try:
            return self._get()
        except (RecursionError, json.JSONDecodeError, UnicodeDecodeError):
            return self._send(500, {'error': 'The saved document could not be read.'})

    def _get(self):
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
                   "keypoints": (KEYPOINTS_DIR / "index.json").is_file(), "labels_license": labels_license(HERE.parent),
                   "targets": targets(HERE.parent)}
            self._send(200, render_index(PAGE_TITLE, cfg, BOARD_NAME, HEADER), "text/html; charset=utf-8")
            return
        if parsed.path == '/api/dictionary':
            return self._dictionary()
        if parsed.path == "/api/version":
            # an open page asks every 30s whether the board was rebuilt, and with what targets
            self._send(200, {"version": board_version(HERE.parent), "targets": targets(HERE.parent)})
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
                body = dumps(public_label(current_label(_read_label(p), DICTIONARY_JOB, p.name), export=True),
                             separators=(",", ":")).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Disposition", f'attachment; filename="{p.name}"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self._send(200, dumps(episode_view(current_label(_read_label(p), DICTIONARY_JOB, p.name))), "application/json")
            return
        if parsed.path.startswith("/api/compare/"):
            # other models' labels (board/build.py compare/): kept beside the board's own and never in its lists
            what = parsed.path[len("/api/compare/"):]
            if what in ("index", "metrics"):
                p = COMPARE_DIR / f"{what}.json"
                if not p.is_file():
                    self._send(404, {"error": "no comparison on this dashboard"})
                    return
                self._send(200, public_json(_read_label(p)), "application/json")
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
                self._send(200, dumps(episode_view(current_label(_read_label(p), DICTIONARY_JOB, p.name))), "application/json")
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
            self._send(200, public_json(_read_label(p)), "application/json")
            return
        if parsed.path in ("/api/grids", "/api/grid"):
            from board import grids as bg
            q = urllib.parse.parse_qs(parsed.query)
            fname = (q.get('file') or [''])[0]
            p = HERE / fname
            if not fname or Path(fname).name != fname or not _under(HERE, p) or not p.is_file():
                return self._send(404, {'error': 'no verified frame grids'})
            try:
                record = current_label(_read_label(p), DICTIONARY_JOB, p.name)
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
            self._send(200, public_json(_read_label(p)), "application/json")
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
                body = dumps(public_json(_read_label(p)), separators=(",", ":")).encode()
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
            self._send(200, public_json(_read_label(p)), "application/json")
            return
        if parsed.path == "/api/video":
            q = urllib.parse.parse_qs(parsed.query)
            eid, cam = (q.get("id") or [""])[0], (q.get("cam") or ["exo"])[0]
            mp4 = clip_path(MP4_DIR, eid, cam)
            if not _under(MP4_DIR, mp4) or not mp4.exists():
                self._send(404, {"error": "video not found"})
                return
            # download=1: one camera's clip saved as a file of its own, named after its camera, so another camera never
            # takes the main camera's name
            name = f"{eid}_{'main' if cam == 'exo' else cam}.mp4" \
                if (q.get("download") or [""])[0] == "1" \
                else None
            self._send_file(mp4, "video/mp4", name, validate=name is None)
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
                got = footage(eid, t0, t1, self.client())
            except CacheBusy as error:
                return self._send(503, {'error': str(error), 'retryable': True})
            except FootageCapacityError as error:
                return self._send(507, {'error': str(error)})
            except ValueError as error:
                return self._send(400, {'error': str(error)})
            except OSError as error:
                return self._send(507 if error.errno in (errno.ENOSPC, errno.EDQUOT) else 503,
                                  {'error': 'Prepared video storage is unavailable.', 'retryable': False})
            except (RuntimeError, subprocess.SubprocessError, KeyError):
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
                    ticket = prepare_footage_ticket(got, self.client(), reuse=True)
                    try:
                        size = out.stat().st_size
                    except FileNotFoundError as error:
                        raise CacheBusy('Prepared video was evicted; prepare it again.') from error
                except CacheBusy as error:
                    return self._send(503, {'error': str(error), 'retryable': True})
                except OSError as error:
                    return self._send(507 if error.errno in (errno.ENOSPC, errno.EDQUOT) else 503,
                                      {'error': 'Prepared video storage cannot save a download ticket.', 'retryable': False})
                except ValueError:
                    return self._send(409, {'error': 'Prepared video changed; prepare it again.'})
                self._send(200, {"bytes": size, "name": name, "ticket": ticket})
                return
            with _footage_reader(getattr(got, 'receipt', None)) as (verified, reader):
                if verified is None:
                    return self._send(503, {'error': 'Prepared video changed; prepare it again.', 'retryable': True})
                return self._send_file(out, "video/mp4", name, source=reader)
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
            tag = None
            try:
                if _under(MP4_DIR, mp4) and math.isfinite(t):
                    from board.static import FRAME_TAG
                    tag = media_tag(mp4, _footage_version(mp4), t, w, FRAME_TAG)
            except OSError:
                tag = None
            if tag and self._not_modified(tag):
                return
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
            if tag:
                self.send_header('ETag', tag)
            self.send_header("Cache-Control", REVALIDATE if tag else "no-store, no-cache, must-revalidate, max-age=0")
            self.end_headers()
            try:
                self.wfile.write(jpg)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        self._send(404, {"error": "unknown path"})


def client_key(peer: str, forwarded: str | None = None) -> str:
    """Who a request comes from, for sharing workers and download tickets between clients: the peer's address, the
    /64 network of an IPv6 peer (one host is routinely given a whole /64), or, for a loopback peer (a reverse proxy on
    this machine), the address that proxy put last in X-Forwarded-For. A forwarded address is trusted only from
    loopback: anything that can connect there can read the board's files anyway."""
    import ipaddress
    try:
        ip = ipaddress.ip_address(peer.split('%')[0])
    except ValueError:
        return peer
    if ip.is_loopback and forwarded:
        last = forwarded.split(',')[-1].strip()
        try:
            ip = ipaddress.ip_address(last.strip('[]').split('%')[0])
        except ValueError:
            return 'forwarded ' + last[:64] if last else str(ip)
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return str(ipaddress.ip_network(f'{ip}/64', strict=False)) if ip.version == 6 else str(ip)


_FORWARDED_FOR = re.compile(rb'\nx-forwarded-for[ \t]*:[ \t]*([^\r\n]*)', re.I)
_HEAD_END = re.compile(rb'\r?\n\r?\n')


class _Held:
    """A connection the server holds while no worker serves it: the bytes of its next request head so far, when it
    was accepted or handed back, and when that head's first byte came."""
    __slots__ = ('sock', 'address', 'peer', 'client', 'data', 'since', 'first', 'state', 'queued')

    def __init__(self, sock, address):
        self.sock, self.address, self.peer = sock, address, client_key(address[0])
        self.client, self.data, self.state, self.queued = self.peer, bytearray(), 'held', 0.0
        self.since, self.first = time.monotonic(), None


class _Reader:
    """A held connection's request bytes for a handler: what the server already received, then the socket. It never
    reads ahead of what the handler asks for, so bytes of a pipelined next request stay here for the server."""
    closed = False

    def __init__(self, sock, data):
        self.sock, self.buffer = sock, bytearray(data)

    def _fill(self) -> bool:
        chunk = self.sock.recv(65536)
        self.buffer += chunk
        return bool(chunk)

    def readline(self, limit=-1):
        while True:
            end = self.buffer.find(b'\n')
            if end >= 0 and (limit is None or limit < 0 or end < limit):
                end += 1
                break
            if limit is not None and 0 <= limit <= len(self.buffer):
                end = limit
                break
            if not self._fill():
                end = len(self.buffer)
                break
        out = bytes(self.buffer[:end])
        del self.buffer[:end]
        return out

    def read(self, n=-1):
        while (n is None or n < 0 or len(self.buffer) < n) and self._fill():
            pass
        end = len(self.buffer) if n is None or n < 0 else min(n, len(self.buffer))
        out = bytes(self.buffer[:end])
        del self.buffer[:end]
        return out

    def close(self):
        pass


class _HeldRequest:
    """What Handler is given in place of a socket for one request on a held connection."""
    def __init__(self, sock, reader, client):
        self.sock, self.reader, self.client, self.close = sock, reader, client, True


class BoardHTTPServer(socketserver.TCPServer):
    """Hold idle and slow connections without a thread each, and share the worker threads fairly between clients.

    One selector thread holds every connection between requests and reads request heads; a worker thread is given a
    connection only once its request head has arrived, for that one request, and the connection then comes back. A
    client that keeps sockets idle, or trickles a head byte by byte, so costs file descriptors, never the workers other
    viewers need. A waiting request goes to the client with the fewest requests in progress, and one client never has
    more than client_workers. Held connections are bounded: near the bound idle ones are closed after a second, and
    over it the oldest connection of whichever client holds the most is closed, so a new viewer is queued and served,
    never refused because someone else holds sockets."""
    allow_reuse_address = True
    request_queue_size = 128
    max_workers = 64
    client_workers = 16
    max_connections = 1024
    head_bytes = 256 * 1024
    worker_idle_seconds = 0.5

    def __init__(self, server_address, RequestHandlerClass, bind_and_activate=True):
        try:
            import resource
            soft = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
            if soft != resource.RLIM_INFINITY:
                # leave descriptors for clips, footage, locks and the encoders' pipes
                self.max_connections = max(16, min(self.max_connections, soft - 128))
        except (ImportError, OSError, ValueError):
            pass
        self._lock = threading.Lock()
        self._work = threading.Condition(self._lock)
        self._selector = selectors.DefaultSelector()
        self._wake_read, self._wake_write = socket.socketpair()
        self._wake_read.setblocking(False)
        self._wake_write.setblocking(False)
        self._by_peer = {}                  # peer -> its connections not being served (held or queued)
        self._held = set()                  # registered with the selector, reading a head
        self._ready = OrderedDict()         # client -> deque of connections whose request head has arrived
        self._busy = {}                     # client -> requests being served
        self._returned = []                 # connections workers hand back for their next request
        self._serving = self._workers = self._idle = self._queued = 0
        self._stopping = threading.Event()
        self._stopped = threading.Event()
        self._stopped.set()
        self._accept_paused_until = 0.0
        # bound last: a bind that fails (the port taken) closes the server, and server_close needs the state above, so
        # the caller sees the bind's own error ("Address already in use") and no descriptor is left open
        super().__init__(server_address, RequestHandlerClass, bind_and_activate)

    # selector thread ------------------------------------------------------------------------------------------------

    def serve_forever(self, poll_interval=0.25):
        self._stopping.clear()
        self._stopped.clear()
        self.socket.setblocking(False)
        self._selector.register(self.socket, selectors.EVENT_READ, 'listen')
        self._selector.register(self._wake_read, selectors.EVENT_READ, 'wake')
        try:
            while not self._stopping.is_set():
                for key, _ in self._selector.select(poll_interval):
                    if key.data == 'listen':
                        self._accept()
                    elif key.data == 'wake':
                        self._adopt()
                    else:
                        self._receive(key.data)
                self._sweep()
        finally:
            for fileobj in (self.socket, self._wake_read):
                try:
                    self._selector.unregister(fileobj)
                except (KeyError, ValueError):
                    pass
            self._close_all()
            self._stopped.set()

    def shutdown(self):
        self._stopping.set()
        self._wake()
        self._stopped.wait()

    def server_close(self):
        super().server_close()
        self._close_all()
        for end in (self._wake_read, self._wake_write):
            end.close()
        self._selector.close()

    def _connections(self) -> int:
        return sum(map(len, self._by_peer.values())) + self._serving

    def _accept(self):
        for _ in range(64):
            try:
                sock, address = self.socket.accept()
            except BlockingIOError:
                return
            except OSError as error:
                if error.errno in (errno.EMFILE, errno.ENFILE, errno.ENOBUFS, errno.ENOMEM):
                    with self._lock:
                        evicted = self._evict()
                    if not evicted:
                        # nothing to close: stop accepting a moment instead of spinning on the listening socket
                        self._selector.unregister(self.socket)
                        self._accept_paused_until = time.monotonic() + 0.1
                    return
                if error.errno in (errno.ECONNABORTED, errno.EPROTO, errno.EINTR):
                    continue
                raise
            sock.setblocking(False)
            held = _Held(sock, address if isinstance(address, tuple) else (str(address), 0))
            with self._lock:
                self._track(held)
                while self._connections() > self.max_connections and self._evict():
                    pass
            if held.state == 'held':
                self._selector.register(sock, selectors.EVENT_READ, held)

    def _track(self, held):
        self._by_peer.setdefault(held.peer, set()).add(held)
        self._held.add(held)
        held.state = 'held'

    def _untrack(self, held):
        peers = self._by_peer.get(held.peer)
        if peers is not None:
            peers.discard(held)
            if not peers:
                del self._by_peer[held.peer]
        self._held.discard(held)

    def _evict(self) -> bool:
        """Close one connection of the client holding the most: an idle one first, then one still sending its head,
        then one waiting for a worker; the oldest of each. Called with the lock held."""
        if not self._by_peer:
            return False
        peer = max(self._by_peer, key=lambda p: len(self._by_peer[p]))
        order = {'held': 0, 'queued': 2}
        victim = min(self._by_peer[peer], key=lambda h: (order[h.state] + (h.first is not None), h.since))
        if victim.state == 'queued':
            waiting = self._ready.get(victim.client)
            if waiting is not None:
                waiting.remove(victim)
                self._queued -= 1
                if not waiting:
                    del self._ready[victim.client]
        else:
            try:
                self._selector.unregister(victim.sock)
            except (KeyError, ValueError):
                pass
        self._untrack(victim)
        victim.state = 'closed'
        self._close(victim.sock)
        return True

    def _receive(self, held):
        try:
            chunk = held.sock.recv(65536)
        except (BlockingIOError, InterruptedError):
            return
        except OSError:
            chunk = b''
        if not chunk:
            return self._drop(held)
        if held.first is None:
            held.first = time.monotonic()
        held.data += chunk
        if _HEAD_END.search(held.data):
            self._selector.unregister(held.sock)
            with self._lock:
                self._queue(held)
        elif len(held.data) > self.head_bytes:
            self._drop(held)

    def _drop(self, held):
        try:
            self._selector.unregister(held.sock)
        except (KeyError, ValueError):
            pass
        with self._lock:
            self._untrack(held)
            held.state = 'closed'
        self._close(held.sock)

    def _adopt(self):
        try:
            while self._wake_read.recv(4096):
                pass
        except (BlockingIOError, InterruptedError):
            pass
        with self._lock:
            returned, self._returned = self._returned, []
            for held in returned:
                self._serving -= 1
                if self._stopping.is_set():
                    self._close(held.sock)
                    continue
                self._track(held)
                if _HEAD_END.search(held.data):
                    self._queue(held)
            while self._connections() > self.max_connections and self._evict():
                pass
        for held in returned:
            if held.state == 'held':
                held.sock.setblocking(False)
                self._selector.register(held.sock, selectors.EVENT_READ, held)

    def _sweep(self):
        now = time.monotonic()
        if self._accept_paused_until and now >= self._accept_paused_until:
            self._accept_paused_until = 0.0
            self._selector.register(self.socket, selectors.EVENT_READ, 'listen')
        deadline = float(getattr(self.RequestHandlerClass, 'request_read_timeout', 15))
        with self._lock:
            crowded = self._connections() > 0.75 * self.max_connections
            expired = [h for h in self._held
                       if now - (h.first if h.first is not None else h.since) >
                       (1.0 if crowded and h.first is None else deadline)]
        for held in expired:
            self._drop(held)

    def _close_all(self):
        with self._lock:
            pending = [h for peers in self._by_peer.values() for h in peers] + self._returned
            self._serving -= len(self._returned)
            self._returned = []
            self._by_peer.clear()
            self._held.clear()
            self._ready.clear()
            self._queued = 0
            self._work.notify_all()
        for held in pending:
            try:
                self._selector.unregister(held.sock)
            except (KeyError, ValueError, RuntimeError):
                pass
            self._close(held.sock)

    @staticmethod
    def _close(sock):
        try:
            sock.shutdown(socket.SHUT_WR)
        except OSError:
            pass
        sock.close()

    def _wake(self):
        try:
            self._wake_write.send(b'.')
        except (BlockingIOError, OSError):
            pass

    # workers --------------------------------------------------------------------------------------------------------

    def _queue(self, held):
        """A connection whose request head has arrived waits for a worker. Called with the lock held."""
        forwarded = _FORWARDED_FOR.findall(bytes(held.data[:_HEAD_END.search(held.data).end()]))
        held.client = client_key(held.address[0], forwarded[-1].decode('latin-1') if forwarded else None)
        self._held.discard(held)
        held.state, held.queued = 'queued', time.monotonic()
        self._ready.setdefault(held.client, deque()).append(held)
        self._queued += 1
        if self._queued > self._idle and self._workers < self.max_workers:
            self._workers += 1
            threading.Thread(target=self._work_loop, name='board-worker', daemon=True).start()
        self._work.notify()

    def _take(self):
        """The waiting connection of the client with the fewest requests in progress (the longest waiting of those),
        skipping clients at their share. Called with the lock held."""
        best = None
        for client, waiting in self._ready.items():
            busy = self._busy.get(client, 0)
            if busy < self.client_workers and (best is None or (busy, waiting[0].queued) < best[0]):
                best = ((busy, waiting[0].queued), client)
        if best is None:
            return None
        waiting = self._ready[best[1]]
        held = waiting.popleft()
        if not waiting:
            del self._ready[best[1]]
        self._queued -= 1
        self._untrack(held)
        held.state = 'serving'
        self._serving += 1
        self._busy[held.client] = self._busy.get(held.client, 0) + 1
        return held

    def _work_loop(self):
        with self._work:
            try:
                while True:
                    held = self._take()
                    if held is None:
                        if self._stopping.is_set():
                            return
                        self._idle += 1
                        try:
                            notified = self._work.wait(self.worker_idle_seconds)
                        finally:
                            self._idle -= 1
                        held = self._take()
                        if held is None:
                            if notified and not self._stopping.is_set():
                                continue
                            return
                    client = held.client
                    self._work.release()
                    try:
                        self._serve(held)
                    finally:
                        self._work.acquire()
                        busy = self._busy.get(client, 1) - 1
                        if busy:
                            self._busy[client] = busy
                        else:
                            self._busy.pop(client, None)
            finally:
                self._workers -= 1

    def _serve(self, held):
        sock, keep = held.sock, False
        request = None
        try:
            sock.setblocking(True)
            sock.settimeout(getattr(self.RequestHandlerClass, 'timeout', None))
            request = _HeldRequest(sock, _Reader(sock, held.data), held.client)
            self.RequestHandlerClass(request, held.address, self)
            keep = not request.close and not self._stopping.is_set()
        except Exception:  # noqa: BLE001 - socketserver's pattern: handle_error logs the request's error, the server goes on
            self.handle_error(sock, held.address)
        if not keep:
            with self._lock:
                self._serving -= 1
            held.state = 'closed'
            self._close(sock)
            return
        held.data = request.reader.buffer
        held.since, held.first = time.monotonic(), (time.monotonic() if held.data else None)
        with self._lock:
            self._returned.append(held)
        self._wake()


def main(argv=None) -> int:
    global HERE, MP4_DIR, COMPARE_DIR, HANDS_DIR, KEYPOINTS_DIR, SENSORS_DIR, FOOTAGE_DIR, PORT, PAGE_TITLE, BOARD_NAME, EXPORT_UPLOADER_VALUES
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
    ap.add_argument('--allow-host', action='append', default=[],
                    help="another host name the board answers to, such as the name a reverse proxy that keeps the "
                         "visitor's Host header (tailscale serve, Caddy) or a LAN DNS name gives it; repeatable. "
                         "Without it the board answers to localhost and, when --host is a network address, to IP "
                         "addresses and this machine's own names")
    ap.add_argument('--publish-uploader-values', action='store_true',
                    help="show and export the uploader's own metadata values (file attributes, sidecar fields) to "
                         "everyone who can reach the board; by default only a loopback board with no --allow-host "
                         "shows them, and downloads and exports never carry them")
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
    _trim_footage_cache()
    for folder in ('.ready', '.tickets'):
        _trim_receipt_temps(FOOTAGE_DIR / folder)
    try:
        # idle connections cost only a descriptor each (BoardHTTPServer), so allow enough of them
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = 4096 if hard == resource.RLIM_INFINITY else min(hard, 4096)
        if soft != resource.RLIM_INFINITY and soft < want:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    except (ImportError, OSError, ValueError):
        pass
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    # socketserver's default listen backlog is 5: a burst of visitors (or a proxy opening many connections at once)
    # overflows it and those connections wait on SYN retries for up to a minute
    socketserver.ThreadingTCPServer.request_queue_size = 128
    socketserver.ThreadingTCPServer.daemon_threads = True
    bind = "127.0.0.1" if DICTIONARY_JOB else a.host
    EXPORT_UPLOADER_VALUES = a.publish_uploader_values
    from board.public_export import publish_uploader_values
    # a loopback board nobody else is let in to (no --allow-host for a proxy) is the uploader's own view
    publish_uploader_values(a.publish_uploader_values or (bind in ('127.0.0.1', 'localhost', '::1') and not a.allow_host))
    with BoardHTTPServer((bind, PORT), Handler) as httpd:
        httpd.allowed_hosts = a.allow_host
        machine_names()      # resolve this machine's names once, before the first request waits on it
        if not list_episodes():
            print('No episodes are available. A free review stops before board generation; build a board with labels.', file=sys.stderr, flush=True)
        print(f"board at http://localhost:{PORT} ({len(list_episodes())} episodes from {HERE}, clips from {MP4_DIR})",
              flush=True)
        httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
