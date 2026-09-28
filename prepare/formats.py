"""Read a folder of your own robot data into episode sidecars (the layout label/episode.py reads). Data Review
reads every upload with this module, and `python -m prepare folder` runs it on a folder on disk.

Accepted uploads, in the order they are recognised:

1. LeRobot, v2.0 / v2.1 (one parquet and one mp4 per camera per episode) or v3.0 (episodes packed
   into shared parquet and mp4 files). Every folder that holds a LeRobot dataset is read, so a collection
   with one sub-dataset per task directory (as FastUMI ships) is one upload. A dataset is recognised by
   meta/info.json, or without it by its data/ and videos/ layout (RoboPRO ships no info.json). Missing
   metadata degrades instead of failing:
     - no meta/info.json: cameras are the videos/ folders, the frame rate comes from the data's
       timestamps or the frames' own times, and the version from the file layout;
     - v2 without meta/episodes.jsonl: episodes are the per-episode files;
     - v3 without meta/episodes: episodes are recovered from the data files' episode_index and accepted
       only if their frame counts add up exactly to every packed video's frame count; otherwise each
       packed video is kept whole, as one recording;
     - no data parquet: the episode is labelled from video.
   Cameras stored as images inside the parquet are written to H.264 at their frame times.
2. MCAP, one file per episode. XDOF ABC-130k and GenRobot RealOmin topic layouts get their full
   adapters (robot state included). Any other layout is read as video: every compressed-image or
   compressed-video channel becomes a camera.
3. Plain video (mp4, mov, mkv, webm, avi). Either one file per episode, or one folder per episode
   holding up to three files (a scene camera plus a left and a right mounted camera, told apart by
   "left" / "right" in the file name). A file is never split: an unsplit recording is one episode, and the
   pipeline labels a long one in pieces and stitches the labels back into one timeline. When
   many files share one length, the recorder cut continuous footage into fixed-length files; the episodes
   say so, so a file that starts or ends mid-activity is read as packaging, not a truncated episode.

An archive (.zip, .tar, .tar.gz, .tar.bz2, .tar.xz) is read as the folder it holds (open_archives). Data
Review's upload page opens archives in the browser and sends their files; this is for archives on disk.

Anything else the uploader sends next to an episode (a .txt or .json with the same name as the
video, or instruction.txt / annotations.json inside an episode folder) is passed to the model as
the uploader's own annotation, a claim to check against the video, never as truth.

Recorded state is used when it has 7 values per arm or gripper (6 joints plus gripper for teleop
arms; x y z roll pitch yaw plus opening for handheld grippers). Other layouts are labelled from the
video alone and the report says so. Nothing in the footage is re-encoded except image frames (MCAP
image channels, LeRobot image features), which are written to H.264 at their real capture times.

Every episode's duration is known before conversion (metadata or container headers), so the
footage cap is applied before any heavy work: episodes are taken in order until the cap is reached
and the rest are listed as skipped. The report lists, in plain words, what was read and used and what
could not be read and what was done instead.
"""
from __future__ import annotations

import json
import os
import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np


VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
RIGS = ("teleop_arms", "handheld_gripper", "ego_head")
STATE_KIND = {"teleop_arms": "joints", "handheld_gripper": "ee_pose"}
TIME_BASE_DEN = 1_000_000
ANNOTATION_MAX_CHARS = 4000
EGO_DESC = ("the camera worn on the person's head, looking forward and down at their hands and the work "
            "in front of them")
FIXED_WINDOW_TOLERANCE_S = 2.5
FIXED_WINDOW_MIN_FILES = 3
PACKAGING_TAGS = ["truncated_episode", "incomplete_demonstration"]


SYSTEM_FILES = {"thumbs.db", "ehthumbs.db", "desktop.ini", "icon\r"}


def hidden_part(s: str) -> bool:
    """A path part that is never read: hidden files and tool caches (.DS_Store, .cache/huggingface), __MACOSX, and
    the files operating systems leave in folders (Thumbs.db, desktop.ini). read.js hiddenPath, the same rule."""
    return s.startswith(".") or s == "__MACOSX" or s.lower() in SYSTEM_FILES


def hidden(p: Path, root: Path) -> bool:
    try:
        parts = p.relative_to(root).parts
    except ValueError:
        parts = p.parts
    return any(hidden_part(s) for s in parts)


def files_under(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.is_file() and not hidden(p, Path(root)))


# ---------------------------------------------------------------- archives

ARCHIVE_RE = re.compile(r"\.(zip|tar|tar\.gz|tgz|tar\.bz2|tbz2|tar\.xz|txz)$", re.I)
UNPACK_MAX_BYTES = int(float(os.environ.get("UPLOAD_UNPACK_MAX_GB", 40)) * 1e9)
UNPACK_MAX_FILES = int(os.environ.get("UPLOAD_MAX_FILES", 20000))


def _member_parts(name: str) -> list[str] | None:
    """The parts of an archive member's path, or None for a path that could leave the folder it is unpacked
    into (absolute, a drive, "..") or that could not be uploaded (server.clean_path's rules)."""
    name = name.replace("\\", "/")
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name) or "\0" in name:
        return None
    parts = [x for x in name.split("/") if x and x != "."]
    if not parts or len(parts) > 32 or len("/".join(parts).encode()) > 1024 \
            or any(x == ".." or len(x.encode()) > 250 for x in parts):
        return None
    return parts


def _normalized(parts: list[str]) -> list[str] | None:
    """A member path with ".." parts resolved; None when it climbs out of the archive."""
    out = []
    for x in parts:
        if x == "..":
            if not out:
                return None
            out.pop()
        elif x not in ("", "."):
            out.append(x)
    return out


def zip_name(info) -> str:
    """A zip member's name as its maker meant it. zipfile reads a name without the UTF-8 flag as CP437; macOS and
    Linux write UTF-8 without setting the flag, so such a name is read as UTF-8 when it is valid UTF-8 (read.js
    zipName, the same rule; Windows Explorer's CP437 names are not valid UTF-8 and stay CP437)."""
    if info.flag_bits & 0x800:
        return info.filename
    x = info.extra
    while len(x) >= 4:                          # Info-ZIP Unicode path field
        tag, size = int.from_bytes(x[:2], "little"), int.from_bytes(x[2:4], "little")
        if tag == 0x7075 and size > 5 and x[4] == 1:
            return x[9:4 + size].decode("utf-8", "replace")
        x = x[4 + size:]
    try:
        return info.orig_filename.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return info.filename


ZIP_METHODS = {9: "Deflate64", 93: "Zstandard", 95: "XZ", 98: "PPMd"}


def _archive_members(path: Path):
    """[(parts, size, open)] for every regular file in the archive, and how many members were left out.
    Links, devices, folders, hidden files and __MACOSX are never unpacked."""
    import stat
    import tarfile
    import zipfile
    out, skipped = [], 0
    if zipfile.is_zipfile(path):              # by its bytes, not its name: a tar named .zip is opened as a tar
        zf = zipfile.ZipFile(path)
        infos = zf.infolist()
        regular = {"/".join(p) for p in (_member_parts(zip_name(i)) for i in infos
                                        if not i.is_dir() and stat.S_IFMT(i.external_attr >> 16) in (0, stat.S_IFREG)) if p}
        by_parts = {"/".join(p): i for i in infos for p in [_member_parts(zip_name(i))] if p}
        for info in infos:
            if info.is_dir():
                continue
            kind = stat.S_IFMT(info.external_attr >> 16)      # 0 when the zip was made without Unix modes
            if kind == stat.S_IFLNK and info.file_size < 4096 and not info.flag_bits & 1:
                # a symbolic link stored as one (ditto, zip -y) is the member it names, when the archive has it;
                # read.js listZip, the same rule
                parts = _member_parts(zip_name(info))
                target = zf.read(info).decode("utf-8", "replace")
                norm = None if target.startswith("/") or parts is None else _normalized(parts[:-1] + target.split("/"))
                tpath = "/".join(norm) if norm else None
                if parts is None or tpath not in regular:
                    skipped += 1
                    continue
                ti = by_parts[tpath]
                out.append((parts, ti.file_size, lambda i=ti: zf.open(i)))
                continue
            if kind and kind != stat.S_IFREG:
                skipped += 1
                continue
            if info.flag_bits & 1:
                raise ValueError(f"{path.name} is password-protected")
            if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA):
                # Deflate64 (Windows Explorer, for large files) and others; the upload page decodes Deflate64 itself
                method = ZIP_METHODS.get(info.compress_type, f"method {info.compress_type}")
                raise ValueError(f"{path.name} uses {method} compression, which Python's zip reader does not open; "
                                 "unzip it and give the folder")
            parts = _member_parts(zip_name(info))
            if parts is None:
                skipped += 1
                continue
            out.append((parts, info.file_size, lambda i=info: zf.open(i)))
        return out, skipped, zf
    tf = tarfile.open(path, "r:*")
    for m in tf:
        if m.isdir():
            continue
        if m.islnk() or m.issym():
            # a link is the member it points to when that member is in the archive (tarfile resolves it among the
            # archive's own members only, never on disk); read.js resolveLinks, the same rule
            try:
                target = tf._find_link_target(m)
            except KeyError:
                target = None
            if target is None or not target.isreg():
                skipped += 1
                continue
            parts = _member_parts(m.name)
            if parts is None:
                skipped += 1
                continue
            out.append((parts, target.size, lambda mm=target: tf.extractfile(mm)))
            continue
        if not m.isreg():
            skipped += 1
            continue
        parts = _member_parts(m.name)
        if parts is None:
            skipped += 1
            continue
        out.append((parts, m.size, lambda mm=m: tf.extractfile(mm)))
    return out, skipped, tf


def open_archives(root: Path, dest: Path) -> tuple[Path, list[str]]:
    """Unpack the upload's archives the way Data Review's upload page opens them (its read.js openArchives): an
    archive whose files sit in one folder unpacks to that folder, one of loose files to a folder named after it.
    An upload that is itself an archive unpacks into dest, which becomes the upload; archives inside a folder
    unpack beside themselves. Archives inside archives are opened in turn, three levels deep (a zip of episode
    zips). Returns the upload's root and what was done, in words.

    Every member is checked before a byte is written: paths that could leave the folder, links and devices are
    skipped, as are hidden files and __MACOSX; an upload holds at most UNPACK_MAX_FILES files and
    UNPACK_MAX_BYTES unpacked, counted from what each member really yields, so an archive whose members hold
    more than they declare stops at the limit."""
    import shutil
    root = Path(root)
    if root.is_file():
        if not ARCHIVE_RE.search(root.name):
            return root, []
        archives, base_of = [root], {root: Path(dest)}
    else:
        archives = [p for p in files_under(root) if ARCHIVE_RE.search(p.name)]
        base_of = {a: a.parent for a in archives}
    notes, total, count, depth = [], 0, 0, {a: 0 for a in archives}
    for a in archives:                          # grows as archives inside archives are unpacked
        done = base_of[a] / f".{a.name}.unpacked"
        if done.exists():
            continue
        try:
            members, skipped, handle = _archive_members(a)
        except ValueError as e:
            notes.append(f"{e}, so it was not opened.")
            continue
        except Exception:
            notes.append(f"{a.name} could not be opened as an archive; it may be damaged.")
            continue
        with handle:
            members = [m for m in members if not any(hidden_part(x) for x in m[0])]
            tops = {m[0][0] if len(m[0]) > 1 else "" for m in members}
            base = base_of[a] if len(tops) == 1 and "" not in tops else base_of[a] / ARCHIVE_RE.sub("", a.name)
            base.mkdir(parents=True, exist_ok=True)
            broot = base.resolve()
            wrote = 0
            for parts, size, opener in members:
                if count >= UNPACK_MAX_FILES or total + size > UNPACK_MAX_BYTES:
                    raise ValueError(f"{a.name} holds more than one upload can: at most {UNPACK_MAX_FILES:,} files "
                                     f"and {UNPACK_MAX_BYTES / 1e9:.0f} GB unpacked")
                target = base.joinpath(*parts)
                if broot not in target.resolve().parents:
                    skipped += 1
                    continue
                inner = ARCHIVE_RE.search(target.name) and depth[a] < 3
                if inner and target not in depth:
                    archives.append(target)
                    base_of[target], depth[target] = target.parent, depth[a] + 1
                if target.exists():            # unpacked by an earlier run that stopped part way
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(target.name + ".part")
                with opener() as src, open(tmp, "wb") as dst:
                    while True:
                        buf = src.read(1 << 20)
                        if not buf:
                            break
                        total += len(buf)
                        if total > UNPACK_MAX_BYTES:
                            dst.close()
                            tmp.unlink()
                            raise ValueError(f"{a.name} unpacks to more than {UNPACK_MAX_BYTES / 1e9:.0f} GB")
                        dst.write(buf)
                tmp.replace(target)
                count += 1
                wrote += 1
        done.write_text("")
        notes.append(f"Opened {a.name}: {wrote} files" + (f"; {skipped} left out (links, or names that could leave "
                                                            "the archive's folder)" if skipped else "") + ".")
    if root.is_file():
        return Path(dest), notes
    return root, notes


# ---------------------------------------------------------------- detection

def detect(root: Path) -> dict:
    """{"format": "lerobot" | "mcap" | "video", ...} for an upload folder, or raises with what was found."""
    root = Path(root)
    files = files_under(root)
    roots = lerobot_roots(root, files)
    if roots:
        return {"format": "lerobot", "roots": roots}
    mcaps = [p for p in files if p.suffix.lower() == ".mcap"]
    if mcaps:
        return {"format": "mcap", "files": [str(p) for p in mcaps]}
    vids = [p for p in files if p.suffix.lower() in VIDEO_EXT]
    if vids:
        return {"format": "video", "files": [str(p) for p in vids]}
    seen = sorted({p.suffix.lower() or p.name for p in files})[:12]
    raise ValueError("the upload holds no LeRobot dataset, MCAP file or video"
                     + (f" (only {', '.join(seen)} files)" if seen else " (it is empty)"))


V2_DATA = re.compile(r"^episode_(\d+)\.parquet$")
V3_DATA = re.compile(r"^file-(\d+)\.parquet$")
V2_VIDEO = re.compile(r"^episode_(\d+)\.(mp4|mkv|mov|webm|avi|m4v)$", re.I)
V3_VIDEO = re.compile(r"^file-(\d+)\.(mp4|mkv|mov|webm|avi|m4v)$", re.I)
CHUNK = re.compile(r"^chunk-(\d+)$")


def lerobot_roots(root: Path, files: list[Path]) -> list[str]:
    """Every folder holding a LeRobot dataset: the parent of each meta/info.json, and folders whose data/ and
    videos/ are laid out the LeRobot way without one. Nested datasets are separate roots."""
    found = set()
    for p in files:
        if p.name == "info.json" and p.parent.name == "meta":
            found.add(p.parent.parent)
    for p in files:
        parts = p.parts
        if V2_DATA.match(p.name) or V3_DATA.match(p.name):
            if p.parent.parent.name == "data" and CHUNK.match(p.parent.name):
                found.add(p.parent.parent.parent)
        elif V2_VIDEO.match(p.name) and CHUNK.match(p.parent.parent.name) and p.parent.parent.parent.name == "videos":
            found.add(p.parent.parent.parent.parent)          # videos/chunk-000/<camera>/episode_000000.mp4
        elif V3_VIDEO.match(p.name) and CHUNK.match(p.parent.name) and len(parts) >= 4 \
                and p.parent.parent.parent.name == "videos":
            found.add(p.parent.parent.parent.parent)          # videos/<camera>/chunk-000/file-000.mp4
    # a folder found only by layout that sits inside a dataset found by its info.json is part of it
    with_info = {r for r in found if (r / "meta" / "info.json").exists()}
    out = [r for r in found if r in with_info or not any(a in with_info for a in r.parents)]
    return [str(r) for r in sorted(out)]


# ---------------------------------------------------------------- shared helpers

DEMUXER = {".mp4": "mov", ".mov": "mov", ".m4v": "mov", ".mkv": "matroska", ".webm": "matroska", ".avi": "avi"}


def open_video(path: Path):
    """A video opened only by the demuxer its extension names, so a file that is really a playlist or a
    concat script (which could make ffmpeg read other local files) fails instead of being followed."""
    import av
    fmt = DEMUXER.get(Path(path).suffix.lower())
    if fmt is None:
        raise ValueError(f"{Path(path).name}: not a video type we accept")
    return av.open(str(path), format=fmt)


def inside(root: Path, p: Path) -> Path:
    """p resolved, and required to be under root: dataset metadata may not point outside the upload."""
    r, q = Path(root).resolve(), Path(p).resolve()
    if q != r and r not in q.parents:
        raise ValueError(f"the dataset refers to a file outside the upload ({p})")
    return q


def video_stream(c, path: Path):
    """The container's first video stream, or a plain reason why there is none."""
    if not c.streams.video:
        raise ValueError("it has no video, only audio" if c.streams.audio else "it holds no video")
    return c.streams.video[0]


def open_checked(path: Path):
    if Path(path).stat().st_size == 0:
        raise ValueError("the file is empty")
    return open_video(path)


def display_rotation(path: Path) -> int:
    """The display rotation of a video's frames in degrees (a phone's portrait video is stored landscape with
    90 or 270), read from its first decoded frame; 0 when there is none."""
    try:
        with open_video(path) as c:
            fr = next(c.decode(c.streams.video[0]), None)
            return int(round(getattr(fr, "rotation", 0) or 0)) % 360 if fr is not None else 0
    except Exception:
        return 0


def probe(path: Path) -> dict:
    """Container facts for one video file: frame pts in decode order sorted, time base, size as shown (a
    display rotation of 90 or 270 degrees swaps the stored width and height; label/frames.py upright turns the
    decoded frames the same way), codec."""
    with open_checked(path) as c:
        st = video_stream(c, path)
        pts = sorted(p.pts for p in c.demux(st) if p.size and p.pts is not None)
        rate = st.average_rate or st.guessed_rate
        w, h, tb, codec = st.codec_context.width, st.codec_context.height, st.time_base, st.codec_context.name
    rot = display_rotation(path)
    if rot in (90, 270):
        w, h = h, w
    return {"pts": np.asarray(pts, dtype=np.int64), "time_base": tb, "width": w, "height": h,
            "codec": codec, "fps": float(rate) if rate else None, "rotation": rot}


def seconds(pr: dict) -> np.ndarray:
    t = pr["pts"].astype(np.float64) * float(pr["time_base"])
    return t - t[0] if len(t) else t


def measured_fps(t: np.ndarray) -> float | None:
    d = np.diff(np.asarray(t, dtype=np.float64))
    d = d[d > 1e-6]
    return round(1.0 / float(np.median(d)), 3) if len(d) else None


def nearest(src_t: np.ndarray, q: np.ndarray) -> np.ndarray:
    if len(src_t) == 1:
        return np.zeros(len(q), dtype=np.int32)
    i = np.clip(np.searchsorted(src_t, q), 1, len(src_t) - 1)
    return np.where(np.abs(src_t[i - 1] - q) <= np.abs(src_t[i] - q), i - 1, i).astype(np.int32)


SCENE_WORDS = ("top", "high", "head", "overhead", "exo", "front", "scene", "main", "cam")
FIXED_WORDS = ("top", "high", "overhead", "exterior", "front", "scene", "exo", "zed", "stereo", "low", "static", "third")
MOUNTED_WORDS = ("wrist", "hand", "gripper", "arm", "eef", "ee")


def tokens(name: str) -> list[str]:
    """Words of a camera name: split on non-letters and camelCase (cam_left_wrist, leftCam, /robot0/wrist_l)."""
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return [t for t in re.split(r"[^a-z0-9]+", spaced.lower()) if t]


def side_of(name: str) -> str | None:
    """left / right when a word of the name is or starts with the side (left, leftcam, l), else None."""
    for tk in tokens(name):
        for side in ("left", "right"):
            if tk == side[0] or tk.startswith(side):
                return side
    return None


def is_mount_named(name: str) -> bool:
    return any(t.startswith(w) for t in tokens(name) for w in MOUNTED_WORDS)


def mounted_side(name: str) -> str | None:
    """The side of a camera mounted on an arm or gripper: it names a side, and either names the mount
    (wrist, hand, gripper) or names nothing that says fixed (overhead_left, exterior_image_1_left and a
    stereo pair's /zed/left are fixed cameras with a side, not wrist cameras)."""
    side = side_of(name)
    if side is None:
        return None
    if is_mount_named(name):
        return side
    return None if any(t.startswith(w) for t in tokens(name) for w in FIXED_WORDS) else side


def scene_rank(name: str) -> int:
    """Prefer an overhead / head / front camera as the scene camera when several are unnamed."""
    n = name.lower()
    for i, w in enumerate(SCENE_WORDS):
        if w in n:
            return i
    return 99


def assign_views(names: list[str], rig: str) -> tuple[dict, list]:
    """{view: camera name} for up to one scene and two mounted cameras, and the names left unused."""
    out = {}
    # names that also say wrist/hand/gripper take a side before names that only carry a side
    for nm in sorted(names, key=lambda n: (0 if is_mount_named(n) else 1, n)):
        v = mounted_side(nm)
        if v and v not in out:
            out[v] = nm
    rest = sorted((nm for nm in names if nm not in out.values()), key=lambda s: (scene_rank(s), s))
    if rest:
        out["exo"] = rest[0]
    unused = [nm for nm in names if nm not in out.values()]
    if rig == "handheld_gripper" and list(out) == ["exo"]:
        # one camera on a handheld rig is the gripper's own camera (FastUMI single_arm), not a scene camera
        out = {"right": out["exo"]}
    return out, unused


NOT_RGB = re.compile(r"depth|conf|disparity|mask|seg|thermal|infrared|(^|/)ir(/|$)|vis_", re.I)
TEXT_TOPIC = re.compile(r"instruction|task|annotation|language|prompt", re.I)
TEXT_MSGS_MAX = 50           # a text topic with more distinct messages than this is a log (a heartbeat), not notes
# a topic named for the task itself (/task, /instruction), which a sub-topic (/task/subtask, /task/health) is not
TASK_TOPIC_NAME = re.compile(r"(language_)?(instruction|task|task_description|prompt|goal)", re.I)
# a topic named for the task's timed steps (MicroAGI's /task/subtask)
STEP_TOPIC_NAME = re.compile(r"sub_?tasks?|steps?", re.I)
# camera channels: compressed images or video (Foxglove, ROS CompressedImage), and raw images, which ROS 2 records
# by default (sensor_msgs/Image, foxglove.RawImage). read.js mcapSummary, the same schemas.
RAW_IMAGE_SCHEMA = re.compile(r"(^|/)(msg/)?Image$|RawImage$", re.I)
CAMERA_SCHEMA = re.compile(r"compressedvideo|compressedimage|(^|/)(msg/)?Image$|RawImage$", re.I)


def raw_image(msg):
    """A raw image message (sensor_msgs/Image, foxglove.RawImage) as an RGB PIL image, or None for an encoding that
    is not a colour or grey picture (depth, 16-bit)."""
    from PIL import Image
    w, h = int(_field(msg, "width") or 0), int(_field(msg, "height") or 0)
    enc = str(_field(msg, "encoding") or "").lower()
    data = _field(msg, "data", binary=True)
    data = bytes(data) if data is not None else b""
    step = int(_field(msg, "step") or 0)
    ch = {"rgb8": 3, "bgr8": 3, "8uc3": 3, "rgba8": 4, "bgra8": 4, "8uc4": 4, "mono8": 1, "8uc1": 1}.get(enc)
    if not (w and h):
        return None
    if ch:
        step = step or w * ch
        if len(data) < step * h:
            return None
        a = np.frombuffer(data, np.uint8, count=step * h).reshape(h, step)[:, :w * ch].reshape(h, w, ch)
        if enc in ("bgr8", "8uc3", "bgra8", "8uc4"):
            a = a[:, :, [2, 1, 0]]
        elif ch == 4:
            a = a[:, :, :3]
        return Image.fromarray(np.ascontiguousarray(a if ch != 1 else a[:, :, 0]), "RGB" if ch != 1 else "L").convert("RGB")
    if enc in ("yuv422", "uyvy", "yuyv", "yuv422_yuy2"):
        import av
        fmt = "uyvy422" if enc in ("yuv422", "uyvy") else "yuyv422"
        step = step or w * 2
        if len(data) < step * h:
            return None
        a = np.frombuffer(data, np.uint8, count=step * h).reshape(h, step)[:, :w * 2]
        return av.VideoFrame.from_ndarray(np.ascontiguousarray(a).reshape(h, w, 2), format=fmt).to_image()
    return None


def pick_cameras(topics: list[str], rig: str, all_topics: list[str] | None = None) -> tuple[dict, list]:
    """{view: topic} among an episode's cameras. A head-camera rig gets exactly one camera: the one the
    recording computes depth for when there is one (its primary camera), else the best-named one."""
    rgb = [t for t in topics if not NOT_RGB.search(t)] or list(topics)
    skipped = [t for t in topics if t not in rgb]
    if rig != "ego_head":
        vm, unused = assign_views(rgb, rig)
        return vm, unused + skipped

    def rank(t):
        base = t.rsplit("/", 1)[0] if "/" in t else ""
        has_depth = bool(base) and any(o != t and o.startswith(base + "/") and "depth" in o.lower()
                                       for o in (all_topics if all_topics is not None else topics))
        return (0 if has_depth else 1, scene_rank(t), t)
    best = sorted(rgb, key=rank)[0]
    return {"exo": best}, [t for t in topics if t != best]


def camera_entry(view: str, name: str, pr: dict, rig: str) -> dict:
    return describe({"key": name, "name": _short(name, view), "width": pr["width"], "height": pr["height"],
                     "codec": pr["codec"]}, view, name, rig)


def describe(cam: dict, view: str, name: str, rig: str) -> dict:
    """Name and describe the cameras whose role follows from the rig alone: the head camera on a person,
    and the cameras carried on handheld grippers. Everything else keeps the prompt's fallback line for its
    slot (a scene camera, or the camera on the left / right arm)."""
    if view in ("left", "right") and rig != "ego_head":
        cam["name"] = view          # the side, as our boards name mounted cameras; the dataset's key stays in "key"
    if rig == "ego_head" and view == "exo":
        cam.update(name="head", desc=EGO_DESC)
    elif rig == "handheld_gripper" and view in ("left", "right"):
        if view == "right" and not re.search("right", name, re.I):
            cam.update(name="gripper", desc="the camera carried on the handheld gripper, looking along its fingers")
        else:
            cam.update(desc=f"the camera carried on the {view.upper()}-hand gripper, looking along its fingers")
    return cam


def _short(name: str, view: str) -> str:
    base = re.sub(r"^(observation\.images\.|observation\.image\.|/)", "", name)
    base = re.sub(r"[^A-Za-z0-9_]+", "_", base).strip("_")
    return base[:32] or view


def annotation_text(obj) -> str | None:
    """The uploader's notes for an episode as text for the prompt (label/episode.py shows them as claims)."""
    if obj in (None, "", {}, []):
        return None
    txt = (obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)).strip()
    if len(txt) > ANNOTATION_MAX_CHARS:
        txt = txt[:ANNOTATION_MAX_CHARS] + " [truncated]"
    return txt or None


def instruction_from(obj) -> str | None:
    """A task text from an uploaded annotation, if it names one plainly."""
    if isinstance(obj, str):
        s = obj.strip()
        return s if s and len(s) < 400 and "\n" not in s else None
    if isinstance(obj, dict):
        for k in ("instruction", "task", "task_description", "language_instruction", "prompt", "goal"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def read_annotation(paths: list[Path]):
    for p in paths:
        if not p.exists():
            continue
        try:
            if p.suffix == ".json":
                return json.loads(p.read_text())
            if p.suffix == ".jsonl":
                return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
            return p.read_text(errors="replace")
        except Exception:
            return p.read_text(errors="replace")
    return None


def state_layout(dims: int, rig: str) -> tuple[str, str | None]:
    """(state_kind, note). 7 or 14 values per frame are 1 or 2 actors of 6 + gripper; anything else is
    labelled from video."""
    if rig == "ego_head":
        return "none", None
    if dims in (7, 14):
        return STATE_KIND[rig], None
    return "none", (f"Labelled from the video: the recorded state has {dims} values per frame, and our checks expect 7 per "
                    + ("arm (six joints and a gripper)." if rig == "teleop_arms" else "gripper (a 6D pose and an opening)."))


def finish_episode(ep: Path, ctx: dict, sources: dict, state=None, action=None, times: dict | None = None) -> dict:
    ep.mkdir(parents=True, exist_ok=True)
    if state is not None and ctx.get("state_kind") != "none":
        arrs = {"state": np.asarray(state, dtype=np.float32)}
        if action is not None and np.shape(action) == np.shape(state):
            arrs["action"] = np.asarray(action, dtype=np.float32)
        np.savez(ep / "state.npz", **arrs)
    if times:
        np.savez(ep / "times.npz", **times)
        ctx["real_times"] = "times.npz"
    (ep / "sources.json").write_text(json.dumps(sources, indent=1))
    (ep / "instruction.txt").write_text((ctx.get("instruction") or "") + "\n")
    (ep / "context.json").write_text(json.dumps(ctx, indent=1, default=str))
    return ctx


def video_views_episode(ep: Path, files: dict, rig: str, dataset: str, extra: dict, shared_clock: bool = False,
                        prs: dict | None = None) -> dict:
    """An episode made of video files {view: (camera name, path)}: real frame times from each file's pts,
    the first view in harness order as the anchor, the others paired by nearest time. Separate video files
    have no common clock, so each starts at its own first frame; cameras written from one MCAP
    (shared_clock) keep their offsets and are measured from the anchor's first frame."""
    from label import episode as me
    prs = prs or {v: probe(p) for v, (_, p) in files.items()}
    order = [v for v in me.VIEW_ORDER if v in files]
    anchor = order[0]
    zero = float(prs[anchor]["pts"][0] * prs[anchor]["time_base"]) if shared_clock else None

    def seconds_of(pr):
        t = pr["pts"].astype(np.float64) * float(pr["time_base"])
        return t - (zero if shared_clock else t[0])
    ta = seconds_of(prs[anchor])
    ep.mkdir(parents=True, exist_ok=True)
    sources, times, cams = {}, {}, {}
    for v in order:
        name, path = files[v]
        pr = prs[v]
        t = seconds_of(pr)
        times[v], times[f"{v}_pts"] = t, pr["pts"]
        sources[v] = {"packed": str(Path(path).resolve()), "base_s": 0.0, "n_frames": int(len(pr["pts"])),
                      "camera_key": name}
        if v != anchor:
            km = nearest(t, ta)
            if not (len(t) == len(ta) and np.array_equal(km, np.arange(len(ta)))):
                np.save(ep / f"kmap_{v}.npy", km)
                sources[v]["kmap"] = f"kmap_{v}.npy"
        cams[v] = camera_entry(v, name, pr, rig)
    # the rate is measured from the frame times (a header can claim any rate); the length is the span of
    # the frames plus one frame, so it matches what the labeller samples
    step = float(np.median(np.diff(ta))) if len(ta) > 1 else 1 / 30
    fps = 1.0 / step if step > 0 else 30.0
    ctx = {"dataset": dataset, "profile": rig, "state_kind": "none", "episode_id": ep.name,
           "robot_type": None, "fps": round(float(fps), 3), "n_state_frames": int(len(ta)),
           "duration_s": round(float(ta[-1]) + step, 3) if len(ta) else 0.0, "cameras": cams, **extra}
    return finish_episode(ep, ctx, sources, times=times)


def episode_name(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    return "episode_" + (re.sub(r"^episode_", "", s)[:120] or "0")


def episode_dirs(out: Path, names: list[str]) -> list[Path]:
    """One episode folder per source name, in list order. run-1.mp4 and run_1.mp4 both clean to
    episode_run_1: the later one gets _2 (then _3, ...), so no episode overwrites another and a rerun over
    the same list maps every source to the same folder."""
    seen: set[str] = set()
    dirs = []
    for x in names:
        base = episode_name(x)
        d, k = base, 2
        while d in seen:
            d, k = f"{base}_{k}", k + 1
        seen.add(d)
        dirs.append(Path(out) / d)
    return dirs


def unique_dir(out: Path, name: str) -> Path:
    """run-1.mp4 and run_1.mp4 both clean to episode_run_1: the second gets a suffix, never overwrites."""
    d, k = out / name, 2
    while d.exists():
        d, k = out / f"{name}_{k}", k + 1
    return d


def _duration(p: Path) -> float:
    with open_checked(p) as c:
        st = video_stream(c, p)
        if st.duration and st.time_base:
            return float(st.duration * st.time_base)
        return float(c.duration or 0) / 1e6


def _frame_count(p: Path) -> int:
    with open_video(p) as c:
        st = c.streams.video[0]
        return sum(1 for pk in c.demux(st) if pk.size and pk.pts is not None)


# ---------------------------------------------------------------- plain video

CAMERA_TOKENS = {"top", "front", "side", "back", "rear", "wrist", "head", "high", "low", "exo", "overhead", "scene", "rgb",
                 "color", "colour", "gripper", "hand", "third", "ego", "zed", "realsense", "depth", "fisheye", "external",
                 "exterior"}
MAX_CAMERAS = 6
# the parts of a file name that number a take rather than name a camera: bare numbers (dates and times included),
# and a take word with or without its number (take1, ep_3, trial, run2)
TAKE_WORD = re.compile(r"^(take|ep|episode|trial|run|demo|seq|sequence|clip|part|seg|segment|shot|rec|recording|"
                       r"session|chunk)(\d*)$")
SAME_LENGTH_S, SAME_LENGTH_FRAC = 1.5, 0.03       # cameras of one take stop within this of each other


def camera_named(name: str) -> bool:
    return any(re.fullmatch(r"(cam|camera|view)\d*", t) or t in CAMERA_TOKENS for t in tokens(name))


def name_parts(stem: str) -> dict:
    """A video's name split into what names its camera and what numbers its take: cam_high_ep3 gives the camera
    "cam_high" and the take "ep3"; top_2 gives "top" and "2"; cam0 is a camera on its own. read.js nameParts."""
    cam, take, explicit = [], [], False
    for t in tokens(stem):
        if t.isdigit():
            take.append(t)
        elif TAKE_WORD.match(t):
            take.append(t)
            explicit = True
        else:
            cam.append(t)
    return {"cam": "_".join(cam), "take": "_".join(take), "explicit": explicit}


def _says_cameras(cams: list[str]) -> bool:
    return any(re.search(r"left|right", c, re.I) for c in cams) or all(camera_named(c) for c in cams)


def one_take(stems: list[str]) -> bool:
    """Whether videos with these names, in one folder, are the cameras of one episode by their names alone: two to
    six of them, one take number (or none), every camera named differently, and the names say cameras (a side, or
    every name a camera's: cam0, cam_high, top, wrist)."""
    parts = [name_parts(s) for s in stems]
    return 2 <= len(stems) <= MAX_CAMERAS and len({p["take"] for p in parts}) == 1 \
        and len({p["cam"] for p in parts}) == len(parts) and _says_cameras([p["cam"] for p in parts])


def _same_length(lengths: list) -> bool:
    if not lengths or any(x is None for x in lengths):
        return True                       # unknown: the lengths cannot tell takes from cameras
    lo, hi = min(lengths), max(lengths)
    return hi - lo <= max(SAME_LENGTH_S, SAME_LENGTH_FRAC * hi)


def group_videos(rels: list[str], length_of=None, grouping: dict | None = None) -> tuple[list[dict], list[dict]]:
    """Plain videos (paths relative to the upload) grouped into episodes, as their names and folders say:
    [{"name", "dir", "cams": [(camera name, path)]}] in episode order, and the folders the files cannot settle,
    [{"dir", "files", "camera"}]. read.js groupVideos, the same rules:

    - Camera folders (top/ep1.mp4, wrist_left/ep1.mp4): two to six sibling folders named as cameras, sharing file
      names, give one episode per shared name, with the folders as its cameras.
    - A folder's own videos group by take number (top_ep1, wrist_left_ep1, top_ep2, ...); a group whose cameras are
      all named differently and named as cameras is one episode (top, wrist_left, wrist_right; cam0, cam1).
    - The same camera name numbered (cam0_take1, cam0_take2; top_1, top_2) is separate takes when a take word says
      so or their lengths differ. When the numbers are bare and the lengths agree, the files cannot tell takes from
      cameras: the folder is returned as unsettled, and read as grouping[folder] says ("takes" or "cameras"), or as
      separate takes, which never merges footage that is not one take.
    - Anything else (IMG_1234, GX010042, episode_000) is one episode per file.

    length_of(path) gives a file's length in seconds (or None); it is asked only for unsettled folders."""
    grouping = grouping or {}
    by_dir: dict[str, list[str]] = {}
    for r in rels:
        d = r.rsplit("/", 1)[0] if "/" in r else ""
        by_dir.setdefault(d, []).append(r)
    stem = lambda r: r.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    base = lambda r: r.rsplit("/", 1)[-1]
    parent = lambda d: d.rsplit("/", 1)[0] if "/" in d else ""
    eps, unsettled, used = [], [], set()
    # camera folders
    kids: dict[str, list[str]] = {}
    for d in by_dir:
        if d:
            kids.setdefault(parent(d), []).append(d)
    for p, ds in kids.items():
        names = [base(d) for d in ds]
        if not one_take(names):
            continue
        by_stem: dict[str, list[tuple[str, str]]] = {}
        for d in ds:
            for r in by_dir[d]:
                by_stem.setdefault(stem(r), []).append((base(d), r))
        if not any(len(v) > 1 for v in by_stem.values()):
            continue                          # no file name shared across the folders: not camera folders
        for st, cams in by_stem.items():
            if len({c for c, _ in cams}) != len(cams):
                continue                      # the same name twice in one folder (top.mp4 and top.mov): read below
            eps.append({"name": f"{p}/{st}" if p else st, "dir": p, "cams": sorted(cams)})
            used.update(r for _, r in cams)
    for d, fs in by_dir.items():
        fs = sorted(r for r in fs if r not in used)
        if not fs:
            continue
        single = lambda r: {"name": r.rsplit(".", 1)[0], "dir": None, "cams": [(stem(r), r)]}
        parts = {r: name_parts(stem(r)) for r in fs}
        takes: dict[str, list[str]] = {}
        for r in fs:
            takes.setdefault(parts[r]["take"], []).append(r)
        good = [g for g in takes.values() if len(g) >= 2 and len(g) <= MAX_CAMERAS and _says_cameras([parts[r]["cam"] for r in g])]
        distinct = all(len({parts[r]["cam"] for r in g}) == len(g) for g in takes.values())
        if good and distinct and all(len(g) <= MAX_CAMERAS for g in takes.values()):
            for k, g in takes.items():
                nm = d or "episode_1"
                eps.append({"name": f"{nm}/{k}" if len(takes) > 1 and k else nm, "dir": d,
                            "cams": [(stem(r), r) for r in g]})
            continue
        cams = {parts[r]["cam"] for r in fs}
        if 2 <= len(fs) <= MAX_CAMERAS and len(cams) == 1 and all(len(g) == 1 for g in takes.values()) \
                and camera_named(next(iter(cams))) and not any(parts[r]["explicit"] for r in fs) \
                and _same_length([length_of(r) if length_of else None for r in fs]):
            unsettled.append({"dir": d, "files": fs, "camera": next(iter(cams))})
            if grouping.get(d) == "cameras":
                eps.append({"name": d or "episode_1", "dir": d, "cams": [(stem(r), r) for r in fs]})
                continue
        eps += [single(r) for r in fs]
    eps.sort(key=lambda e: tuple(e["name"].split("/")))
    return eps, unsettled


def plan_video(det: dict, root: Path, grouping: dict | None = None) -> list[dict]:
    """One item per episode, grouped as group_videos says. Files are never split. Fixed-length packaging (a
    recorder that cuts continuous footage into files of one length) is found here and recorded on every item it
    applies to."""
    root = Path(root)
    rels = [Path(f).relative_to(root).as_posix() for f in det["files"]]
    durations = {}

    def length_of(r):
        if r not in durations:
            try:
                durations[r] = _duration(root / r)
            except Exception:
                durations[r] = None
        return durations[r]
    eps, unsettled = group_videos(rels, length_of, grouping)
    for u in unsettled:
        choice = (grouping or {}).get(u["dir"])
        what = f"{u['dir']}: " if u["dir"] else ""
        det.setdefault("used", []).append(
            f"{what}{', '.join(r.rsplit('/', 1)[-1] for r in u['files'])} share one camera name and one length; "
            + ("they were read as the cameras of one episode, as chosen." if choice == "cameras" else
               "they were read as separate episodes" + (", as chosen." if choice == "takes" else
                                                        ", since nothing said they are one take.")))
    items = []
    for e in eps:
        files = [root / r for _, r in e["cams"]]
        items.append({"kind": "video", "name": e["name"], "files": files,
                      "dir": (root / e["dir"]) if e["dir"] is not None else None,
                      "cams": {c: root / r for c, r in e["cams"]}})
    for it in items:
        try:
            it["seconds"] = max((durations.get(Path(f).relative_to(root).as_posix()) or _duration(f)) for f in it["files"])
        except Exception:
            it["seconds"] = None          # unreadable header: measured on conversion, or reported as unreadable
    return items


def fixed_window(lengths: list) -> float | None:
    """The file length a recorder cuts continuous footage into, when at least FIXED_WINDOW_MIN_FILES files
    and at least half of all files share one length (within the tolerance); else None."""
    ls = sorted(float(x) for x in lengths if x)
    if len(ls) < FIXED_WINDOW_MIN_FILES:
        return None
    best, best_n = None, 0
    for x in ls:
        n = sum(1 for y in ls if abs(y - x) <= FIXED_WINDOW_TOLERANCE_S)
        if n > best_n:
            best, best_n = x, n
    if best_n >= FIXED_WINDOW_MIN_FILES and best_n * 2 >= len(ls) and best >= 30:
        near = [y for y in ls if abs(y - best) <= FIXED_WINDOW_TOLERANCE_S]
        return round(float(np.median(near)), 1)
    return None


def packaging_note(window: float) -> str:
    return (f"the recorder cuts continuous footage into consecutive files of about {window:g} seconds wherever "
            "the clock falls; a file is a window of ongoing work, so starting or ending in the middle of an "
            "activity is how the footage is packaged, not a fault of the episode.")


def convert_video(item: dict, rig: str, out: Path, dataset: str) -> dict:
    fs = item["files"]
    if item["dir"] is None:
        f = fs[0]
        view = {"ego_head": "exo", "teleop_arms": "exo", "handheld_gripper": "right"}[rig]
        files = {view: (f.stem, f)}
        ann = read_annotation([f.with_suffix(s) for s in (".json", ".txt", ".jsonl", ".md")])
    else:
        by = item.get("cams") or {f.stem: f for f in fs}
        vmap, unused = pick_cameras(list(by), rig, list(by))
        files = {v: (nm, by[nm]) for v, nm in vmap.items()}
        d = item["dir"]
        own = [f.with_suffix(x) for f in fs for x in (".json", ".txt")]      # camera folders: ep1.txt beside a camera
        ep_name = item["name"].rsplit("/", 1)[-1]
        ann = read_annotation([d / f"{ep_name}{x}" for x in (".json", ".txt")] + [d / n for n in (
            "annotations.json", "annotation.json", "meta.json", "instruction.txt", "task.txt", "annotations.jsonl",
            "notes.txt")] + (own if len({f.stem for f in fs}) == 1 else []))
    extra = {"task_label": [item["name"]], "source": {"format": "video files", "upload": item["name"]}}
    instr = instruction_from(ann)
    if instr:
        extra["instruction"] = instr
        extra["instruction_note"] = "This instruction is the task text the uploader sent with the episode."
    if ann is not None and not (isinstance(ann, str) and ann.strip() == instr):
        extra["uploader_annotation"] = annotation_text(ann)
    if item.get("fixed_window_s"):
        extra["collection_note"] = packaging_note(item["fixed_window_s"])
        extra["packaging"] = {"fixed_window_s": item["fixed_window_s"]}
    ep = unique_dir(out, episode_name(item["name"]))
    return video_views_episode(ep, files, rig, dataset, extra)


# ---------------------------------------------------------------- LeRobot: reading a dataset root

def read_jsonl(p: Path) -> list[dict]:
    out = []
    if not p.exists():
        return out
    for line in p.read_text(errors="replace").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue            # one damaged line never loses the rest of the metadata
    return out


def scalar(v):
    return v[0] if hasattr(v, "__len__") and not isinstance(v, str) else v


def _read_json(p: Path) -> dict | None:
    try:
        d = json.loads(p.read_text())
        return d if isinstance(d, dict) else None
    except Exception:
        return None


ANNOT_TEXT_COLS = ("instruction", "language_instruction", "task", "annotation", "prompt", "description")


def annotated_instructions(meta: Path) -> tuple[dict, str | None]:
    """{episode_index: text} from a per-episode annotation table the dataset ships next to its coarse tasks
    (MolmoAct2's meta/tasks_annotated.parquet, which its card names as the per-episode instruction), and
    the file it came from. Only tables keyed by episode index with one text column count."""
    import pandas as pd
    if not meta.is_dir():
        return {}, None
    for p in sorted(meta.glob("*annotat*")):
        try:
            if p.suffix == ".parquet":
                df = pd.read_parquet(p)
            elif p.suffix == ".jsonl":
                df = pd.DataFrame(read_jsonl(p))
            else:
                continue
        except Exception:
            continue
        if "episode_index" in df.columns:
            df = df.set_index("episode_index")
        elif df.index.name != "episode_index":
            continue
        col = next((c for c in ANNOT_TEXT_COLS if c in df.columns), None)
        if col is None:
            continue
        out = {}
        for k, v in df[col].items():
            v = scalar(v)
            if isinstance(v, str) and v.strip():
                out[int(k)] = v.strip()
        if out:
            return out, p.relative_to(meta.parent).as_posix()
    return {}, None


def read_root(rdir: Path, rel: str) -> dict:
    """One LeRobot dataset folder, normalised: cameras, frame rate, and one row per episode with its video
    paths (or packed windows), its data file and its task text, plus plain-words notes on what was read and
    what was missing."""
    import pandas as pd
    info = _read_json(rdir / "meta" / "info.json")
    used, missing = [], []
    where = f" in {rel}" if rel else ""
    data_v2 = sorted(p for p in (rdir / "data").glob("chunk-*/episode_*.parquet")) if (rdir / "data").is_dir() else []
    data_v3 = sorted(p for p in (rdir / "data").glob("chunk-*/file-*.parquet")) if (rdir / "data").is_dir() else []
    vids_v2 = sorted(p for p in (rdir / "videos").glob("chunk-*/*/episode_*.*") if p.suffix.lower() in VIDEO_EXT) \
        if (rdir / "videos").is_dir() else []
    vids_v3 = sorted(p for p in (rdir / "videos").glob("*/chunk-*/file-*.*") if p.suffix.lower() in VIDEO_EXT) \
        if (rdir / "videos").is_dir() else []
    version = str((info or {}).get("codebase_version") or "")
    if not version:
        version = "v3.0" if (data_v3 or vids_v3) and not (data_v2 or vids_v2) else "v2.1"
    v3 = version.startswith("v3")
    feats = (info or {}).get("features") or {}
    if info is None:
        missing.append(f"No meta/info.json{where}, so the episodes were found from its folders (LeRobot {version[:2]}) and the "
                       "frame rate from its timestamps.")
    else:
        used.append(f"LeRobot {version} metadata{where} (meta/info.json).")
    cams = [k for k, f in feats.items() if f.get("dtype") == "video"]
    image_cams = [k for k, f in feats.items() if f.get("dtype") == "image"]
    if not cams:
        on_disk = sorted({p.parent.name for p in vids_v2} if not v3 else {p.parent.parent.name for p in vids_v3})
        if on_disk:
            if feats:
                missing.append(f"meta/info.json{where} lists no video cameras, so the cameras in its videos folder were used.")
            cams = on_disk
    fps = float(info["fps"]) if info and info.get("fps") else None
    tasks_by_index = {}
    for t in read_jsonl(rdir / "meta" / "tasks.jsonl"):
        if "task_index" in t and "task" in t:
            tasks_by_index[int(t["task_index"])] = str(t["task"])
    if (rdir / "meta" / "tasks.parquet").exists():
        try:
            tp = pd.read_parquet(rdir / "meta" / "tasks.parquet")
            if "task_index" in tp.columns:
                for task, row in tp.iterrows():
                    tasks_by_index[int(row["task_index"])] = str(row.get("task", task))
        except Exception:
            pass
    annotated, annot_src = annotated_instructions(rdir / "meta")
    root = {"dir": str(rdir), "rel": rel, "info": info, "version": version, "v3": v3, "fps": fps,
            "robot_type": (info or {}).get("robot_type"), "features": feats, "cams": cams, "image_cams": image_cams,
            "tasks_by_index": tasks_by_index, "annotated": annotated, "annot_src": annot_src,
            "used": used, "missing": missing, "episodes": [], "recordings": []}
    if v3:
        _episodes_v3(root, rdir, data_v3, vids_v3)
    else:
        _episodes_v2(root, rdir, data_v2, vids_v2)
    if annotated and root["episodes"]:
        used.append(f"Per-episode instructions{where} from {annot_src}.")
    return root


def _episodes_v2(root: dict, rdir: Path, data_files: list[Path], vids: list[Path]) -> None:
    info, where = root["info"] or {}, (f" in {root['rel']}" if root["rel"] else "")
    rows = {int(scalar(e["episode_index"])): e for e in read_jsonl(rdir / "meta" / "episodes.jsonl") if "episode_index" in e}
    data_by = {int(V2_DATA.match(p.name).group(1)): p for p in data_files}
    vid_by: dict[int, dict] = {}
    for p in vids:
        vid_by.setdefault(int(V2_VIDEO.match(p.name).group(1)), {})[p.parent.name] = p
    if rows:
        root["used"].append(f"Episode list{where} from meta/episodes.jsonl ({len(rows)} listed).")
    elif info:
        root["missing"].append(f"No meta/episodes.jsonl{where}, so each uploaded episode file is one episode.")
    idx = sorted(set(rows) | set(data_by) | set(vid_by))
    absent = 0
    for e in idx:
        row = rows.get(e, {})
        vids_e = {}
        for key in root["cams"]:
            p = vid_by.get(e, {}).get(key)
            if p is None and info.get("video_path"):
                try:
                    cand = rdir / info["video_path"].format(episode_chunk=e // int(info.get("chunks_size", 1000)),
                                                            video_key=key, episode_index=e)
                    p = inside(rdir, cand) if cand.exists() else None
                except (KeyError, IndexError, ValueError):
                    p = None
            if p is not None:
                vids_e[key] = p
        data = data_by.get(e)
        if data is None and info.get("data_path"):
            try:
                cand = rdir / info["data_path"].format(episode_chunk=e // int(info.get("chunks_size", 1000)), episode_index=e)
                data = inside(rdir, cand) if cand.exists() else None
            except (KeyError, IndexError, ValueError):
                data = None
        if not vids_e and not (root["image_cams"] and data):
            absent += e in rows
            continue
        tasks = row.get("tasks")
        tasks = [str(t) for t in tasks] if isinstance(tasks, list) else ([str(tasks)] if tasks else [])
        root["episodes"].append({"eidx": e, "length": int(scalar(row["length"])) if row.get("length") else None,
                                 "tasks": tasks, "data": data, "videos": vids_e})
    if absent:
        root["used"].append(f"The metadata{where} lists {absent} more episodes than were uploaded; the uploaded ones were labelled.")


def _episodes_v3(root: dict, rdir: Path, data_files: list[Path], vids: list[Path]) -> None:
    import pandas as pd
    info, where = root["info"] or {}, (f" in {root['rel']}" if root["rel"] else "")
    parts = sorted((rdir / "meta" / "episodes").glob("chunk-*/*.parquet")) if (rdir / "meta" / "episodes").is_dir() else []
    eps = []
    if parts:
        try:
            eps = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).to_dict("records")
        except Exception:
            eps = []
            root["missing"].append(f"meta/episodes{where} could not be opened, so the episodes were found from the data files.")
    if eps:
        root["used"].append(f"Episode list{where} from meta/episodes ({len(eps)} listed).")
        rel_tpl = info.get("video_path", "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
        data_tpl = info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
        skipped = 0
        for e in sorted(eps, key=lambda r: int(scalar(r["episode_index"]))):
            eidx = int(scalar(e["episode_index"]))
            vids_e = {}
            for key in root["cams"]:
                try:
                    mp4 = inside(rdir, rdir / rel_tpl.format(video_key=key, chunk_index=int(scalar(e[f"videos/{key}/chunk_index"])),
                                                             file_index=int(scalar(e[f"videos/{key}/file_index"]))))
                except (KeyError, ValueError):
                    continue
                if mp4.exists():
                    vids_e[key] = (mp4, float(scalar(e[f"videos/{key}/from_timestamp"])), float(scalar(e[f"videos/{key}/to_timestamp"])))
            if not vids_e:
                skipped += 1
                continue
            data = None
            try:
                dp = inside(rdir, rdir / data_tpl.format(chunk_index=int(scalar(e["data/chunk_index"])),
                                                         file_index=int(scalar(e["data/file_index"]))))
                data = dp if dp.exists() else None
            except (KeyError, ValueError):
                pass
            tasks = e.get("tasks")
            tasks = [str(t) for t in tasks] if hasattr(tasks, "__len__") and not isinstance(tasks, str) else ([str(tasks)] if tasks else [])
            root["episodes"].append({"eidx": eidx, "length": int(scalar(e["length"])) if e.get("length") is not None else None,
                                     "tasks": tasks, "data": data, "videos": vids_e})
        if skipped:
            root["used"].append(f"The metadata{where} lists {skipped} more episodes than there is uploaded video for; the "
                                "uploaded ones were labelled.")
        return
    # no episode metadata: recover episodes from the data files, accepted only on an exact frame-count match
    root["missing"].append(f"No meta/episodes{where}, which says where each episode sits in the packed videos.")
    by_key: dict[str, list[Path]] = {}
    for p in vids:
        by_key.setdefault(p.parent.parent.name, []).append(p)
    keys = [k for k in root["cams"] if k in by_key] or sorted(by_key)
    if not keys:
        return
    order = lambda p: (int(CHUNK.match(p.parent.name).group(1)), int(V3_VIDEO.match(p.name).group(1)))
    for k in keys:
        by_key[k].sort(key=order)
    lengths, tasks_of, data_of, fps_ts = {}, {}, {}, []
    for dp in data_files:
        try:
            import pyarrow.parquet as pq
            have = pq.ParquetFile(dp).schema_arrow.names
            df = pd.read_parquet(dp, columns=[c for c in ("episode_index", "frame_index", "timestamp", "task_index")
                                              if c in have])
        except Exception:
            continue
        if "episode_index" not in df.columns:
            continue
        for e, g in df.groupby("episode_index"):
            lengths[int(e)] = len(g)
            data_of[int(e)] = dp
            if "task_index" in g.columns:
                ti = int(g["task_index"].iloc[0])
                if ti in root["tasks_by_index"]:
                    tasks_of[int(e)] = [root["tasks_by_index"][ti]]
            if "timestamp" in g.columns and len(g) > 2:
                fps_ts.append(measured_fps(np.sort(g["timestamp"].to_numpy(dtype=np.float64))))
    if root["fps"] is None and fps_ts:
        root["fps"] = float(np.median([x for x in fps_ts if x]))
    fps = root["fps"]
    places = {k: _place_episodes(by_key[k], sorted(lengths.items())) for k in keys} if lengths else {}
    common = sorted(set.intersection(*[set(p) for p in places.values()])) if places and all(places.values()) else []
    if common and fps:
        for eidx in common:
            vids_e = {k: (places[k][eidx][0], places[k][eidx][1] / fps, (places[k][eidx][1] + lengths[eidx]) / fps) for k in keys}
            root["episodes"].append({"eidx": eidx, "length": lengths[eidx], "tasks": tasks_of.get(eidx, []),
                                     "data": data_of.get(eidx), "videos": vids_e})
        root["used"].append(f"{len(common)} episodes{where} found in the data files and matched to every camera's packed "
                            "videos frame for frame.")
        return
    # no certain placement: each packed video of the scene camera is one recording; another camera joins it only
    # when its file has the same name and the same number of frames (otherwise its footage is not the same time)
    root["missing"].append(f"The episodes{where} could not be matched to the packed videos exactly, so each packed video "
                           "was labelled as one recording.")
    vm, _ = assign_views(keys, "teleop_arms")
    lead = vm.get("exo") or keys[0]
    counts = {}
    dropped = set()
    for p in by_key[lead]:
        files = {lead: p}
        for k in keys:
            if k == lead:
                continue
            q = next((x for x in by_key[k] if x.parent.name == p.parent.name and x.name == p.name), None)
            if q is not None:
                a = counts.setdefault(p, _frame_count(p))
                b = counts.setdefault(q, _frame_count(q))
                if abs(a - b) <= 2:
                    files[k] = q
                    continue
            dropped.add(k)
        root["recordings"].append({"name": p.relative_to(rdir).with_suffix("").as_posix(), "files": files})
    if dropped:
        root["missing"].append(f"The {', '.join(sorted(dropped))} videos{where} are packed differently from the {lead} "
                               "videos and, without the episode list, cannot be lined up with them in time, so they were "
                               "not used.")


def _place_episodes(files: list[Path], lengths: list[tuple[int, int]]) -> dict:
    """{episode_index: (file, first_frame)} for the data's episodes inside one camera's packed files (in file
    order), or {} when the placement is not certain. LeRobot v3 never splits an episode across files, so every
    file boundary (the stream's start and end included) that falls inside the run of consecutive episodes must
    fall exactly on an episode boundary, and at least two must. An offset is accepted only when exactly one does; the data
    may hold episodes outside the uploaded videos and the videos episodes outside the data."""
    counts = [_frame_count(p) for p in files]
    bounds = [int(b) for b in np.cumsum([0] + counts)]
    total = bounds[-1]
    idx = [e for e, _ in lengths]
    runs, cur = [], []
    for e in idx:
        if cur and e == cur[-1] + 1:
            cur.append(e)
        else:
            if cur:
                runs.append(cur)
            cur = [e]
    if cur:
        runs.append(cur)
    n_of = dict(lengths)
    best = {}
    for run in runs:
        cum = [int(c) for c in np.cumsum([0] + [n_of[e] for e in run])]
        cset, span = set(cum), cum[-1]
        ok = []
        for o in sorted({b - c for b in bounds for c in cum}):
            inside = [b for b in bounds if o <= b <= o + span]
            if len(inside) < 2:            # one boundary pins nothing: any episode edge could sit on it
                continue
            if all((b - o) in cset for b in inside):
                ok.append(o)
        if len(ok) != 1:
            continue
        o = ok[0]
        out = {}
        for e, c in zip(run, cum[:-1]):
            a, z = o + c, o + c + n_of[e]
            if a < 0 or z > total:
                continue
            fi = max(k for k in range(len(files)) if bounds[k] <= a)
            if z > bounds[fi + 1]:
                return {}                  # an episode across a file boundary: the placement is wrong
            out[e] = (files[fi], a - bounds[fi])
        if len(out) > len(best):
            best = out
    return best


def plan_lerobot(det: dict, root: Path) -> tuple[list[dict], list[str], list[str], list[dict]]:
    """Items for every episode of every dataset root, and the used / missing notes."""
    roots_dirs = [Path(r) for r in det["roots"]]
    base = Path(root)
    if len(roots_dirs) > 1:
        import os
        base = Path(os.path.commonpath([str(r) for r in roots_dirs]))
    items, used, missing, roots = [], [], [], []
    if len(roots_dirs) > 1:
        used.append(f"A collection of {len(roots_dirs)} LeRobot datasets, one per folder; each was read on its own.")
    for rdir in roots_dirs:
        rel = rdir.relative_to(base).as_posix() if len(roots_dirs) > 1 and rdir != base else ""
        r = read_root(rdir, rel)
        roots.append(r)
        used += r["used"]
        missing += r["missing"]
        prefix = (r["rel"] + "/") if r["rel"] else ""
        for e in r["episodes"]:
            fps = r["fps"]
            secs = e["length"] / fps if e.get("length") and fps else None
            if secs is None and e["videos"]:
                v = next(iter(e["videos"].values()))
                secs = (v[2] - v[1]) if isinstance(v, tuple) else _safe_duration(v)
            items.append({"kind": "lerobot", "name": f"{prefix}{e['eidx']:06d}", "root": r, "row": e, "seconds": secs})
        for rec in r["recordings"]:
            items.append({"kind": "recording", "name": prefix + rec["name"], "root": r, "files": rec["files"],
                          "seconds": max((_safe_duration(p) or 0) for p in rec["files"].values()) or None})
    return items, used, missing, roots


def _safe_duration(p: Path) -> float | None:
    try:
        return _duration(p)
    except Exception:
        return None


# ---------------------------------------------------------------- LeRobot: converting one episode

def _read_episode_table(path: Path, eidx: int, columns: list[str]):
    import pandas as pd
    import pyarrow.parquet as pq
    have = pq.ParquetFile(path).schema_arrow.names
    cols = [c for c in columns if c in have]
    df = pd.read_parquet(path, columns=cols or None)
    if "episode_index" in df.columns:
        df = df[df["episode_index"] == eidx]
    if "frame_index" in df.columns:
        df = df.sort_values("frame_index")
    return df


def _stack(col) -> np.ndarray | None:
    try:
        a = np.stack([np.asarray(x, dtype=np.float64).reshape(-1) for x in col.to_numpy()])
        return a if a.ndim == 2 and np.isfinite(a).all() else None
    except Exception:
        return None


def convert_lerobot(item: dict, rig: str, out: Path, dataset: str) -> dict:
    r, row = item["root"], item["row"]
    rdir = Path(r["dir"])
    eidx = row["eidx"]
    info = r["info"] or {}
    feats = r["features"]
    ep = out / episode_name(item["name"])
    notes = []
    df = None
    if row.get("data") is not None:
        try:
            need_images = [k for k in r["image_cams"]] if not any(k in row["videos"] for k in r["cams"]) else []
            df = _read_episode_table(row["data"], eidx, ["observation.state", "action", "frame_index", "episode_index",
                                                         "timestamp", "task_index", *need_images])
            if not len(df):
                df = None
        except Exception as e:
            notes.append("Labelled from the video: the episode's data file could not be opened.")
            df = None
    fps = r["fps"]
    if fps is None and df is not None and "timestamp" in df.columns and len(df) > 2:
        fps = measured_fps(np.sort(df["timestamp"].to_numpy(dtype=np.float64)))
    state = _stack(df["observation.state"]) if df is not None and "observation.state" in df.columns else None
    action = _stack(df["action"]) if df is not None and "action" in df.columns else None
    tasks = list(row.get("tasks") or [])
    if not tasks and df is not None and "task_index" in df.columns and len(df):
        ti = int(df["task_index"].iloc[0])
        if ti in r["tasks_by_index"]:
            tasks = [r["tasks_by_index"][ti]]
    annotated = r["annotated"].get(eidx)
    extra = {"task_label": tasks or [item["name"]], "episode_index": eidx, "robot_type": r["robot_type"],
             "source": {"format": f"lerobot {r['version']}", "episode_index": eidx, "dataset_folder": r["rel"] or None}}
    if annotated:
        # the coarse task stays the task label; the per-episode annotation is the goal (label/episode.py states both)
        extra["instruction"] = annotated
    elif tasks:
        extra["instruction"] = "; ".join(t.strip() for t in tasks if t.strip())
        extra["instruction_note"] = "This instruction is the episode's task text in the dataset's LeRobot metadata."

    video_cams = [k for k in r["cams"] if k in row["videos"]]
    if not video_cams and r["image_cams"] and df is not None:
        return _convert_image_episode(item, rig, ep, dataset, df, fps or 30.0, state, action, extra, notes)
    vmap, unused = pick_cameras(video_cams, rig, list(feats) or video_cams)
    if r["image_cams"]:
        unused = unused + [f"{k} (images in the data file)" for k in r["image_cams"]]
    extra["source"]["unused_cameras"] = unused
    kind, note = state_layout(state.shape[1] if state is not None else 0, rig)
    if state is None and rig != "ego_head":
        note = ("Labelled from the video: the dataset records no observation.state."
                if df is not None or row.get("data") is None else note)
        if row.get("data") is None:
            note = "Labelled from the video: no data file came with this episode."
        if notes:
            note = None                  # the read failure already says why, in notes
    packed = all(isinstance(row["videos"][k], tuple) for k in vmap.values())
    if packed:
        if not fps:
            # no meta/info.json and no timestamps: the frame rate is the packed video's own
            k0 = next(iter(vmap.values()))
            pr0 = probe(row["videos"][k0][0])
            fps = measured_fps(seconds(pr0)) or pr0["fps"] or 30.0
        sources, cameras = {}, {}
        for v, key in vmap.items():
            mp4, base, to = row["videos"][key]
            n = int(round((to - base) * fps))
            sources[v] = {"packed": str(mp4), "base_s": base, "n_frames": n, "camera_key": key}
            vi = (feats.get(key) or {}).get("info") or {}
            w, h, codec = vi.get("video.width"), vi.get("video.height"), vi.get("video.codec")
            if not w:
                pr = _stream_facts(mp4)
                w, h, codec = pr["width"], pr["height"], pr["codec"]
            cameras[v] = describe({"key": key, "name": _short(key, v), "width": w, "height": h, "codec": codec}, v, key, rig)
        n_frames = min(s["n_frames"] for s in sources.values())
        if state is not None and kind != "none" and len(state) != n_frames:
            if abs(len(state) - n_frames) <= 1:
                n = min(len(state), n_frames)          # one frame of rounding in the packed window
                state, action = state[:n], (action[:n] if action is not None else None)
                for s in sources.values():
                    s["n_frames"] = min(s["n_frames"], n)
            else:
                kind, note = "none", (f"Labelled from the video: the recorded state has {len(state)} frames and the "
                                      f"video {n_frames}, so the two cannot be lined up.")
        ctx = {"dataset": dataset, "profile": rig, "state_kind": kind, "episode_id": ep.name, "fps": fps,
               "n_state_frames": int(len(state)) if state is not None and kind != "none" else int(min(s["n_frames"] for s in sources.values())),
               "cameras": cameras, "stream_checks": {"episode_length_meta": row.get("length")}, **extra}
        if note or notes:
            ctx["state_note"] = " ".join([x for x in [note, *notes] if x])
        return finish_episode(ep, ctx, sources, state if kind != "none" else None, action)
    # one file per camera per episode (v2). LeRobot's timestamps are frame_index / fps and state rows follow
    # frames, so frames on the exact k/fps grid need no times; frames off it are decoded by their own pts and
    # timed by frame index, as the dataset defines them
    prs = {v: probe(row["videos"][key]) for v, key in vmap.items()}
    from label import episode as me
    anchor = next(v for v in me.VIEW_ORDER if v in prs)
    if not fps:
        fps = measured_fps(prs[anchor]["pts"].astype(np.float64) * float(prs[anchor]["time_base"])) or 30.0
    sources, cameras, grid = {}, {}, {}
    for v, key in vmap.items():
        pr = prs[v]
        n = int(len(pr["pts"]))
        step = Fraction(1) / Fraction(fps).limit_denominator(1000) / pr["time_base"]
        grid[v] = bool(n and step.denominator == 1 and pr["pts"][0] == 0 and
                       np.array_equal(pr["pts"], np.arange(n) * int(step)))
        sources[v] = {"packed": str(Path(row["videos"][key]).resolve()), "base_s": 0.0, "n_frames": n, "camera_key": key}
        cameras[v] = camera_entry(v, key, pr, rig)
    times = None
    if not all(grid.values()):
        times = {}
        for v, pr in prs.items():
            times[v] = np.arange(len(pr["pts"])) / fps
            times[f"{v}_pts"] = pr["pts"]
    n_video = min(s["n_frames"] for s in sources.values())
    if state is not None and kind != "none" and len(state) != sources[anchor]["n_frames"]:
        if abs(len(state) - sources[anchor]["n_frames"]) <= 1:
            n = min(len(state), sources[anchor]["n_frames"])
            state, action = state[:n], (action[:n] if action is not None else None)
        else:
            kind, note = "none", (f"Labelled from the video: the recorded state has {len(state)} frames and the {anchor} "
                                  f"camera {sources[anchor]['n_frames']}, so the two cannot be lined up.")
    ctx = {"dataset": dataset, "profile": rig, "state_kind": kind, "episode_id": ep.name, "fps": fps,
           "n_state_frames": int(len(state)) if state is not None and kind != "none" else int(n_video),
           "cameras": cameras, "stream_checks": {"frames_on_grid": grid, "episode_length_meta": row.get("length")},
           **extra}
    if note or notes:
        ctx["state_note"] = " ".join([x for x in [note, *notes] if x])
    return finish_episode(ep, ctx, sources, state if kind != "none" else None, action, times=times)


def _stream_facts(p: Path) -> dict:
    with open_video(p) as c:
        st = c.streams.video[0]
        return {"width": st.codec_context.width, "height": st.codec_context.height, "codec": st.codec_context.name}


def _convert_image_episode(item, rig, ep, dataset, df, fps, state, action, extra, notes) -> dict:
    """Cameras stored as encoded images inside the data file: each written to H.264 at its frame time."""
    r = item["root"]
    vmap, unused = pick_cameras(list(r["image_cams"]), rig, list(r["features"]))
    ep.mkdir(parents=True, exist_ok=True)
    fi = df["frame_index"].to_numpy() if "frame_index" in df.columns else np.arange(len(df))
    t = df["timestamp"].to_numpy(dtype=np.float64) if "timestamp" in df.columns else fi / fps
    t = t - t[0]
    files = {}
    for v, key in vmap.items():
        w = FrameWriter(ep / f"{v}.mp4", "")
        for ts, cell in zip(t, df[key].to_numpy()):
            b = cell.get("bytes") if isinstance(cell, dict) else cell
            if isinstance(b, (bytes, bytearray)) and b:
                w.add(float(ts), bytes(b))
        if not w.close():
            raise ValueError(f"the {key} images in the data file could not be decoded")
        files[v] = (key, ep / f"{v}.mp4")
    extra = {**extra}
    extra["source"]["unused_cameras"] = unused
    extra["source"]["images_in_parquet"] = True
    ctx = video_views_episode(ep, files, rig, dataset, extra)
    kind, note = state_layout(state.shape[1] if state is not None else 0, rig)
    if state is not None and kind != "none" and len(state) == ctx["n_state_frames"]:
        ctx["state_kind"] = kind
        return finish_episode(ep, ctx, json.loads((ep / "sources.json").read_text()), state, action,
                              times={k: v for k, v in np.load(ep / "times.npz").items()})
    if rig != "ego_head":
        ctx["state_note"] = note or "Labelled from the video: the recorded state and the image frames cannot be lined up."
        (ep / "context.json").write_text(json.dumps(ctx, indent=1, default=str))
    return ctx


def convert_recording(item: dict, rig: str, out: Path, dataset: str) -> dict:
    """A packed LeRobot video whose episodes could not be placed with certainty: the whole file is one
    recording, labelled from video."""
    keys = list(item["files"])
    vmap, unused = pick_cameras(keys, rig, keys)
    files = {v: (k, item["files"][k]) for v, k in vmap.items()}
    extra = {"task_label": [item["name"]],
             "source": {"format": f"lerobot {item['root']['version']} (packed video kept whole)", "file": item["name"],
                        "unused_cameras": unused},
             "unsplit": True}
    if rig != "ego_head":
        extra["state_note"] = ("Labelled from the video, as one recording: its episodes could not be matched to the "
                               "packed video exactly.")
    return video_views_episode(unique_dir(out, episode_name(item["name"])), files, rig, dataset, extra)


# ---------------------------------------------------------------- MCAP

def plan_mcap(det: dict, root: Path) -> list[dict]:
    from mcap.reader import make_reader
    items = []
    for f in det["files"]:
        try:
            with open(f, "rb") as fh:
                s = make_reader(fh).get_summary()
        except Exception:
            s = None
        st = s.statistics if s else None
        secs = (st.message_end_time - st.message_start_time) / 1e9 if st and st.message_count else None
        topics = sorted({c.topic for c in s.channels.values()}) if s else []
        items.append({"kind": "mcap", "name": Path(f).relative_to(root).with_suffix("").as_posix(), "file": Path(f),
                      "seconds": secs, "topics": topics})
    return items


def mcap_layout(topics: list[str]) -> str:
    from prepare import abc130k as abc
    from prepare import realomin as ro
    if any(t in topics for t in abc.TOP_TOPICS) and all(t in topics for t in abc.VIEW_TOPIC.values()) \
            and all(t in topics for t in abc.ARM):
        return "abc130k"
    if all(t in topics for t in ro.CAMERA_TOPICS) and all(t in topics for t in ro.POSE_TOPICS):
        return "realomin"
    return "generic"


MCAP_MAGIC = b"\x89MCAP0\r\n"


def convert_mcap(item: dict, rig: str, out: Path, dataset: str) -> dict:
    with open(item["file"], "rb") as fh:
        if fh.read(len(MCAP_MAGIC)) != MCAP_MAGIC:
            raise ValueError("it is not a readable MCAP file; it may be damaged, or another kind of file renamed")
    if not item["topics"]:
        # no summary section (a recording cut off before its footer): read it as a stream of messages
        item["topics"] = _mcap_topics_by_scan(item["file"])
    layout = mcap_layout(item["topics"])
    ep = out / episode_name(item["name"])
    if layout != "generic" and item["seconds"] is None:
        layout = "generic"          # a cut-off file: the full adapters need its summary; its cameras are still read
        item.setdefault("notes", []).append("The file ends early, before its index, so it was read from its cameras.")
    if layout == "abc130k":
        from prepare import abc130k as abc
        ctx = abc.convert(item["file"], ep, ep.name, task=item["name"])
    elif layout == "realomin":
        from prepare import realomin as ro
        ctx = ro.convert(item["file"], ep, "upload/" + item["name"])
        # RealOmin's task text is the dataset's own folder path on Hugging Face; an uploader's folder name is
        # not a task, so the episode goes to the model with no instruction rather than an invented one
        for k in ("instruction", "instruction_note"):
            ctx.pop(k, None)
        ctx["task_label"] = [item["name"]]
        (ep / "instruction.txt").write_text("\n")
    else:
        return convert_mcap_generic(item, rig, ep, dataset)
    ctx.update({"dataset": dataset, "source": {"format": f"mcap ({layout} layout)", "file": item["name"]}})
    # the adapters assume a nominal rate (ABC: 30 Hz), but stations record at 30 or 60 Hz; the rate and the
    # length come from the anchor camera's real capture times, so sampling is one instant per second of
    # real time and the footage cap counts real minutes
    from label import episode as me
    t = np.load(ep / ctx["real_times"])[next(v for v in me.VIEW_ORDER if v in ctx["cameras"])]
    step = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30
    ctx["fps"] = round(1.0 / step, 3)
    ctx["duration_s"] = round(float(t[-1] - t[0]) + step, 3)
    if ctx.get("profile") != rig:
        ctx["source"]["rig_note"] = f"the upload was marked {rig}; the {layout} layout is {ctx.get('profile')}"
    (ep / "context.json").write_text(json.dumps(ctx, indent=1, default=str))
    return ctx


def _mcap_stream(path: Path, topics: set | None = None):
    """(schema, channel, message) for every message in file order, read record by record, so a recording cut off
    before its summary (or mid-chunk) yields every message written before the cut."""
    from mcap.records import Channel, Message, Schema
    from mcap.stream_reader import StreamReader
    schemas, chans = {}, {}
    with open(path, "rb") as fh:
        try:
            for r in StreamReader(fh, skip_magic=False).records:
                if isinstance(r, Schema):
                    schemas[r.id] = r
                elif isinstance(r, Channel):
                    chans[r.id] = r
                elif isinstance(r, Message):
                    ch = chans.get(r.channel_id)
                    if ch is not None and (topics is None or ch.topic in topics):
                        yield schemas.get(ch.schema_id), ch, r
        except Exception:
            return                  # the cut: everything before it has been yielded


def _mcap_channels_by_scan(path: Path) -> list[tuple[str, str]]:
    from mcap.records import Channel, Schema
    from mcap.stream_reader import StreamReader
    schemas, seen = {}, {}
    with open(path, "rb") as fh:
        try:
            for r in StreamReader(fh, skip_magic=False).records:
                if isinstance(r, Schema):
                    schemas[r.id] = r.name
                elif isinstance(r, Channel):
                    seen.setdefault(r.topic, schemas.get(r.schema_id, ""))
        except Exception:
            pass
    return sorted(seen.items())


def _mcap_topics_by_scan(path: Path) -> list[str]:
    return [t for t, _ in _mcap_channels_by_scan(path)]


def _decoder_for(encoding: str, schema, facs: list):
    """A function decoding one message payload, or None: JSON channels by json, the rest by whichever
    installed factory (protobuf, ROS 2) knows the encoding."""
    if encoding == "json":
        return lambda b: json.loads(b)
    for f in facs:
        d = f.decoder_for(encoding, schema)
        if d is not None:
            return d
    return None


def _field(msg, key, binary: bool = False):
    if isinstance(msg, dict):
        v = msg.get(key)
        if binary and isinstance(v, str):
            import base64
            return base64.b64decode(v)          # JSON messages carry bytes as base64
        return v
    return getattr(msg, key, None)


def _decoders():
    facs = []
    try:
        from mcap_protobuf.decoder import DecoderFactory as P
        facs.append(P())
    except ImportError:
        pass
    try:
        from mcap_ros2.decoder import DecoderFactory as R
        facs.append(R())
    except ImportError:
        pass
    try:
        from mcap_ros1.decoder import DecoderFactory as R1
        facs.append(R1())
    except ImportError:
        pass
    return facs


def _title_of(x):
    """A record in protobuf text form that is only a title (MicroAGI's task messages): the title, else the text."""
    m = re.fullmatch(r'\s*title:\s*"([^"]+)"\s*', x) if isinstance(x, str) else None
    return m.group(1) if m else (x.strip() if isinstance(x, str) else x)


def mcap_task_texts(texts: dict, counts: dict, t0: int | None) -> tuple[str | None, dict]:
    """(instruction, uploader notes) from an MCAP's text topics, each a list of its distinct messages in time order as
    (log time ns, text). The instruction is the one message of a topic named for the task (/task, /instruction)
    before any sub-topic (/task/subtask, /task/health); MicroAGI's /task titles the fragment and its /task/subtask
    names each step, so the first step is never the task. A topic with several messages goes to the notes as a
    timeline on the episode's clock (the vendor's steps, claims to check); a topic with more than TEXT_MSGS_MAX
    distinct messages (a heartbeat) keeps only its first, with the count."""
    def named(t):
        return bool(TASK_TOPIC_NAME.fullmatch(t.rstrip("/").rsplit("/", 1)[-1]))
    # a timeline of steps (several messages on a sub-topic) is never read as the task: with no task topic the
    # episode goes to the model with no instruction rather than its first step
    cands = [t for t in texts if re.search("instruction|task", t, re.I) and texts[t] and isinstance(texts[t][0][1], str)
             and 0 < len(texts[t][0][1].strip()) < 400 and (named(t) or len(texts[t]) == 1)]
    cands.sort(key=lambda t: (not named(t), len(texts[t]) != 1))
    it = cands[0] if cands else None
    notes = {}
    base = t0 if t0 is not None else min((s[0][0] for s in texts.values() if s), default=0)
    for t, seq in texts.items():
        if t == it and len(seq) == 1:
            continue                  # the instruction itself (a task topic whose text changes keeps its timeline)
        if len(seq) == 1:
            notes[t] = seq[0][1]
        elif len(seq) > TEXT_MSGS_MAX:
            notes[t] = f"{_title_of(seq[0][1])} (the first of {counts.get(t, len(seq))} messages)"
        else:
            notes[t] = [f"{max(0.0, (ts - base) / 1e9):.1f} s: {_title_of(x)}" for ts, x in seq]
    if it is None:
        return None, notes
    instr = texts[it][0][1]
    # a task message in protobuf text form (MicroAGI's /task: title, success, tools, confidence) gives its title as
    # the instruction; the rest of the record stays with the uploader's notes as claims to check
    title = re.search(r'\btitle:\s*"([^"]+)"', instr)
    if title:
        notes["task record"] = instr
        instr = title.group(1)
    return instr.strip(), notes


def mcap_step_subtasks(texts: dict, t0: int | None, end_s: float | None) -> tuple[str | None, list]:
    """(topic, [{t0, t1, label}]) of a topic named for steps (MicroAGI's /task/subtask), in the shape the dataset
    adapters give a head camera's timed subtasks (Gen-HumanEgo's, read by the harness as the dataset's annotation):
    each step lasts until the next one begins, the last until the episode ends."""
    t = next((t for t in texts if STEP_TOPIC_NAME.fullmatch(t.rstrip("/").rsplit("/", 1)[-1])
              and 0 < len(texts[t]) <= TEXT_MSGS_MAX and all(isinstance(x, str) for _, x in texts[t])), None)
    if t is None:
        return None, []
    base = t0 if t0 is not None else texts[t][0][0]
    starts = [max(0.0, (ts - base) / 1e9) for ts, _ in texts[t]]
    ends = starts[1:] + [max(starts[-1], end_s if end_s is not None else starts[-1])]
    return t, [{"t0": round(a, 3), "t1": round(b, 3), "label": _title_of(x)}
               for a, b, (_, x) in zip(starts, ends, texts[t])]


def convert_mcap_generic(item: dict, rig: str, ep: Path, dataset: str) -> dict:
    """Every compressed-image / compressed-video channel that carries a colour camera is a candidate camera;
    up to three are used (one for a head camera). A text or annotation channel becomes the task text or the
    uploader's notes. No robot state: the layout is not one we know."""
    from mcap.reader import make_reader
    with open(item["file"], "rb") as fh:
        try:
            summ = make_reader(fh).get_summary()
        except Exception:
            summ = None
    if summ is not None:
        chans = list(summ.channels.values())
        schema_of = {c.id: (summ.schemas.get(c.schema_id).name if summ.schemas.get(c.schema_id) else "") for c in chans}
        chan_topics = [(c.topic, schema_of[c.id]) for c in chans]
    else:
        chan_topics = _mcap_channels_by_scan(item["file"])
    video_topics = sorted({t for t, s in chan_topics if CAMERA_SCHEMA.search(s) and not NOT_RGB.search(t)})
    raw_topics = {t for t, s in chan_topics if RAW_IMAGE_SCHEMA.search(s)}
    if not video_topics:
        raise ValueError("the file has no colour camera channel"
                         + (f" (its channels: {', '.join(item['topics'][:12])})" if item["topics"] else ""))
    all_topics = [t for t, _ in chan_topics]
    vmap, unused = pick_cameras(video_topics, rig, all_topics)
    text_topics = sorted({t for t in all_topics if TEXT_TOPIC.search(t) and t not in video_topics})
    want = set(vmap.values()) | set(text_topics)
    view_of_topic = {t: v for v, t in vmap.items()}
    writers: dict[str, FrameWriter] = {}
    texts: dict[str, list] = {}               # topic: its distinct messages in time order, (log time ns, text)
    n_text: dict[str, int] = {}
    facs, decs, undecodable, t0 = _decoders(), {}, set(), None
    ep.mkdir(parents=True, exist_ok=True)
    with open(item["file"], "rb") as fh:
        try:
            msgs = make_reader(fh).iter_messages(topics=sorted(want), log_time_order=True) if summ is not None \
                else _mcap_stream(item["file"], want)
            for schema, ch, msg in msgs:
                if ch.topic in texts and len(texts[ch.topic]) > TEXT_MSGS_MAX:
                    n_text[ch.topic] += 1
                    continue
                if ch.id not in decs:
                    decs[ch.id] = _decoder_for(ch.message_encoding, schema, facs)
                if decs[ch.id] is None:
                    undecodable.add(f"{ch.topic} ({ch.message_encoding})")
                    continue
                dec = decs[ch.id](msg.data)
                if ch.topic in view_of_topic:
                    t0 = int(msg.log_time) if t0 is None else t0          # every camera on the recording's one clock
                    w = writers.get(ch.topic)
                    if w is None:
                        w = writers[ch.topic] = FrameWriter(ep / f"{view_of_topic[ch.topic]}.mp4",
                                                            "raw" if ch.topic in raw_topics else str(_field(dec, "format") or "").lower())
                    if ch.topic in raw_topics:
                        im = raw_image(dec)
                        if im is not None:
                            w.add_image((int(msg.log_time) - t0) / 1e9, im)
                    else:
                        w.add((int(msg.log_time) - t0) / 1e9, bytes(_field(dec, "data", binary=True)))
                else:
                    d = _field(dec, "data")
                    x = d if isinstance(d, str) else (dec if isinstance(dec, dict) else str(dec))
                    seq = texts.setdefault(ch.topic, [])
                    n_text[ch.topic] = n_text.get(ch.topic, 0) + 1
                    if not seq or seq[-1][1] != x:          # a message repeated until the next one is one message
                        seq.append((int(msg.log_time), x))
        except Exception as e:
            if not writers:
                raise
            # a damaged tail (recording cut off): keep every frame read before it
            item.setdefault("notes", []).append("The file ends early; every frame before the cut was used.")
    counts = {t: w.close() for t, w in writers.items()}
    missing = [t for t in vmap.values() if not counts.get(t)]
    if missing and len(missing) == len(vmap):
        encodings = sorted({u.rsplit(" (", 1)[-1].rstrip(")") for u in undecodable})
        if encodings:
            raise ValueError(f"its camera channels ({', '.join(missing)}) are in the {' and '.join(encodings)} message "
                             "encoding, which we do not decode; MCAP messages in JSON, Protobuf, ROS 1 or ROS 2 work")
        raise ValueError(f"no frame could be decoded from {', '.join(missing)}")
    for t in missing:              # a camera with no readable frames is left out; the others are labelled
        v = view_of_topic[t]
        unused.append(f"{t} (no readable frames)")
        del vmap[v]
    files = {v: (t, ep / f"{v}.mp4") for v, t in vmap.items()}
    extra = {"task_label": [item["name"]], "source": {"format": "mcap (camera channels only)", "file": item["name"],
                                                      "unused_cameras": unused}}
    instr, notes = mcap_task_texts(texts, n_text, t0)
    if instr:
        extra.update(instruction=instr, instruction_note="This instruction is the task text stored in the MCAP.")
    if rig == "ego_head":
        # a head camera's timed steps are the dataset's subtasks (claims to check, as Gen-HumanEgo's are), shown to
        # the model once, as its annotation, not again among the notes
        end_s = max(((w.pts[-1] + (w.pts[-1] - w.pts[-2] if len(w.pts) > 1 else 0)) / TIME_BASE_DEN
                     for w in writers.values() if w.pts), default=None)
        st, subs = mcap_step_subtasks(texts, t0, end_s)
        if subs:
            notes.pop(st, None)
            extra["annotation_subtasks"] = subs
    if notes:
        extra["uploader_annotation"] = annotation_text({t: x for t, x in notes.items()})
    # motion the file records but this reader does not use yet (hand, body or camera poses in a human recording,
    # joints in an unknown robot layout): named on the job page, so a missing check is never a silent gap
    motion = [t for t in item["topics"] if re.search(r"hand|pose|slam|body|joint|odom|/tf$", t, re.I)
              and not re.search(r"health|info|meta|static|image|mask", t, re.I)]
    if item["seconds"] is None and mcap_layout(item["topics"]) != "generic":
        extra["state_note"] = ("Labelled from the cameras: the file ends before the index its robot state is read from.")
    elif motion:
        shown = ", ".join(motion[:4]) + (f" and {len(motion) - 4} more" if len(motion) > 4 else "")
        extra["state_note"] = (f"Labelled from the cameras. The checks on recorded motion did not run on its motion channels "
                               f"({shown}): they read joint and gripper state from LeRobot datasets and from ABC-130k and "
                               "RealOmin recordings.")
    elif rig != "ego_head":
        extra["state_note"] = "Labelled from the cameras: the file records no robot state."
    if item.get("notes"):
        extra["source"]["notes"] = item["notes"]
    if item.get("fixed_window_s"):
        extra["collection_note"] = packaging_note(item["fixed_window_s"])
        extra["packaging"] = {"fixed_window_s": item["fixed_window_s"]}
    return video_views_episode(ep, files, rig, dataset, extra, shared_clock=True)


def nal_units(b: bytes):
    """Header bytes of each NAL unit in an Annex-B buffer."""
    i = 0
    while True:
        j = b.find(b"\x00\x00\x01", i)
        if j < 0 or j + 3 >= len(b):
            return
        yield b[j + 3]
        i = j + 3


def codec_of(frame: bytes, fmt: str) -> str:
    """hevc or h264: the format field when it says, else the NAL header (HEVC parameter sets are types
    32-34, whose header byte is 0x40-0x44; an H.264 SPS is 0x67)."""
    if fmt in ("h265", "hevc"):
        return "hevc"
    if fmt == "h264":
        return "h264"
    first = next(nal_units(frame), None)
    return "hevc" if first is not None and (first >> 1) & 0x3F in (32, 33, 34, 19, 20) and first & 0x81 == 0 else "h264"


def is_keyframe(frame: bytes, codec: str) -> bool:
    types = {(h >> 1) & 0x3F for h in nal_units(frame)} if codec == "hevc" else {h & 0x1F for h in nal_units(frame)}
    return bool(types & ({19, 20, 21} if codec == "hevc" else {5}))


class FrameWriter:
    """One camera's frames streamed to an mp4 at their real times as they are read, so an MCAP is never
    held in memory. Annex-B H.264/H.265 is copied without re-encoding, starting at the first keyframe
    (earlier frames cannot be decoded); JPEG/PNG frames are encoded to H.264 at the first frame's size."""

    def __init__(self, out: Path, fmt: str):
        self.out, self.fmt, self.pts, self.kind = out, fmt, [], None
        self.raw = self.enc = self.dst = None

    def add(self, t_s: float, frame: bytes) -> None:
        if self.kind is None:
            head = frame[:4]
            annexb = head.startswith(b"\x00\x00\x00\x01") or head[:3] == b"\x00\x00\x01"
            self.kind = codec_of(frame, self.fmt) if (annexb or self.fmt in ("h264", "h265", "hevc")) else "image"
        p = int(round(t_s * TIME_BASE_DEN))
        if self.pts and p <= self.pts[-1]:
            p = self.pts[-1] + 1          # a repeated or backwards stamp stays in order, one microsecond on
        if self.kind == "image":
            try:
                self._image(p, frame)
            except Exception:
                pass                      # one undecodable image is skipped, never the whole camera
            return
        if self.raw is None:
            if not is_keyframe(frame, self.kind):
                return
            self.raw = open(self.out.with_suffix("." + self.kind), "wb")
        self.raw.write(frame)
        self.pts.append(p)

    def add_image(self, t_s: float, im) -> None:
        """A picture already decoded (a raw image message), encoded to H.264 like JPEG frames."""
        self.kind = "image"
        p = int(round(t_s * TIME_BASE_DEN))
        if self.pts and p <= self.pts[-1]:
            p = self.pts[-1] + 1
        self._image(p, im)

    def _image(self, p: int, frame) -> None:
        import io
        import av
        from PIL import Image
        im = frame.convert("RGB") if isinstance(frame, Image.Image) else Image.open(io.BytesIO(frame)).convert("RGB")
        if self.enc is None:
            self.dst = av.open(str(self.out), "w")
            self.enc = self.dst.add_stream("libx264", rate=30, time_base=Fraction(1, TIME_BASE_DEN))
            self.enc.width, self.enc.height = im.width - im.width % 2, im.height - im.height % 2
            self.enc.pix_fmt = "yuv420p"
            self.enc.codec_context.time_base = Fraction(1, TIME_BASE_DEN)
            self.enc.options = {"crf": "18", "preset": "veryfast"}
        if im.size != (self.enc.width, self.enc.height):
            im = im.resize((self.enc.width, self.enc.height))
        fr = av.VideoFrame.from_image(im)
        fr.pts, fr.time_base = p, Fraction(1, TIME_BASE_DEN)
        for pkt in self.enc.encode(fr):
            self.dst.mux(pkt)
        self.pts.append(p)

    def close(self) -> int:
        """Finish the file; returns the number of frames written."""
        import av
        if self.kind == "image" and self.enc is not None:
            for pkt in self.enc.encode():
                self.dst.mux(pkt)
            self.dst.close()
        elif self.raw is not None:
            self.raw.close()
            raw = self.out.with_suffix("." + self.kind)
            try:
                with av.open(str(raw), format=self.kind) as src, av.open(str(self.out), "w") as dst:
                    ist = src.streams.video[0]
                    ost = dst.add_stream_from_template(ist)
                    ost.time_base = Fraction(1, TIME_BASE_DEN)
                    i = 0
                    pts = self.pts
                    for pkt in src.demux(ist):
                        if not pkt.size or i >= len(pts):
                            continue
                        pkt.stream, pkt.time_base = ost, ost.time_base
                        pkt.pts = pkt.dts = pts[i]
                        # the step to the next frame, the last repeating the one before: the raw stream's own
                        # duration (0, or a 25 fps guess) could end the edit list where the last frame starts
                        pkt.duration = (pts[i + 1] - pts[i] if i + 1 < len(pts) else
                                        pts[i] - pts[i - 1] if i else TIME_BASE_DEN // 30)
                        dst.mux(pkt)
                        i += 1
                self.pts = self.pts[:i]
            finally:
                raw.unlink(missing_ok=True)
        return len(self.pts)


def trim_episode(ep: Path, max_s: float) -> dict:
    """Cut an episode's sidecar down to its first max_s seconds (anchor frames, every camera, state, times)."""
    from label import episode as me
    e = me.load(ep)
    ctx = e["context"]
    a = me.anchor(e)
    n = int(e["sources"][a]["n_frames"])
    t = np.array([me.frame_time(e, k) for k in range(n)])
    keep = max(1, int(np.searchsorted(t, max_s - 1e-6)))
    src = e["sources"]
    for v, s_ in src.items():
        km = e["kmap"].get(v)
        if v == a:
            s_["n_frames"] = keep
        elif km is not None:
            s_["n_frames"] = int(km[keep - 1]) + 1
            np.save(ep / s_["kmap"], np.asarray(km[:keep], dtype=np.int32))
        else:
            s_["n_frames"] = min(int(s_["n_frames"]), keep)
    if e.get("times") is not None:
        tz = dict(e["times"])
        for v in src:
            if v in tz:
                tz[v] = tz[v][:src[v]["n_frames"]]
            if f"{v}_pts" in tz:
                tz[f"{v}_pts"] = tz[f"{v}_pts"][:src[v]["n_frames"]]
        np.savez(ep / ctx["real_times"], **tz)
    if (ep / "state.npz").exists():
        z = np.load(ep / "state.npz")
        np.savez(ep / "state.npz", **{k: z[k][:keep] for k in z.files})
    step = float(np.median(np.diff(t))) if n > 1 else 1 / 30
    ctx["trimmed"] = {"to_s": round(float(t[keep - 1]) + step, 3), "of_s": round(float(t[-1]) + step, 3)}
    ctx["n_state_frames"] = keep
    ctx["duration_s"] = ctx["trimmed"]["to_s"]
    (ep / "sources.json").write_text(json.dumps(src, indent=1))
    (ep / "context.json").write_text(json.dumps(ctx, indent=1, default=str))
    return ctx


# ---------------------------------------------------------------- entry point

def plan(root: Path, grouping: dict | None = None) -> tuple[dict, list[dict]]:
    root = Path(root)
    det = detect(root)
    det.setdefault("used", [])
    det.setdefault("missing", [])
    if det["format"] == "lerobot":
        items, used, missing, roots = plan_lerobot(det, root)
        det["used"] += used
        det["missing"] += missing
        det["version"] = ", ".join(sorted({r["version"] for r in roots}))
        return det, items
    if det["format"] == "mcap":
        items = plan_mcap(det, root)
        det["used"].append(f"{len(items)} MCAP files, one episode each." if len(items) != 1 else "1 MCAP file, one episode.")
        if any(it["seconds"] is None for it in items):
            det["missing"].append("Some MCAP files end early, before their index; they were scanned message by message.")
        _mark_packaging(det, items)
        return det, items
    items = plan_video(det, root, grouping)
    det["used"].append(f"{len(items)} video episodes." if len(items) != 1
                       else "1 video episode.")
    _mark_packaging(det, items)
    return det, items


def _mark_packaging(det: dict, items: list[dict]) -> None:
    w = fixed_window([it["seconds"] for it in items])
    if not w:
        return
    for it in items:
        if it["seconds"] is not None and abs(it["seconds"] - w) <= FIXED_WINDOW_TOLERANCE_S:
            it["fixed_window_s"] = w
    det["used"].append(f"The files share one length ({w:g} s), so they are read as continuous footage cut into files, "
                       "and a file that starts or ends mid-activity is not reported as cut off.")
    det["packaging"] = {"fixed_window_s": w}


def plain_error(e: Exception) -> str:
    """Why an episode could not be read, in words for the uploader. Our own ValueErrors are written that way;
    library errors are named by what they mean for the file."""
    msg = str(e).strip()
    mod = type(e).__module__ or ""
    if isinstance(e, FileNotFoundError):
        return "a file it refers to was not in the upload"
    if mod.startswith("av") or "Invalid data found" in msg or "moov atom" in msg:
        return "its video could not be decoded; the file may be damaged or cut short"
    if mod.startswith("pyarrow") or "Parquet" in msg or "parquet" in msg:
        return "its data file could not be opened; the file may be damaged or cut short"
    if mod.startswith("mcap") or "magic" in msg.lower():
        return "the MCAP file could not be opened; it may be damaged or not an MCAP file"
    if isinstance(e, ValueError) and msg and len(msg) < 300:
        return msg
    return "it could not be opened (" + type(e).__name__ + ")"


def convert(root: Path, rig: str, out: Path, dataset: str, max_seconds: float, grouping: dict | None = None) -> dict:
    """Convert an upload into episode sidecars under out/, taking episodes in order until max_seconds of
    footage. Returns a report of what was accepted, skipped and why, and what was read and what was not."""
    if rig not in RIGS:
        raise ValueError(f"rig must be one of {RIGS}")
    root, out = Path(root), Path(out)
    root, opened = open_archives(root, out.parent / "upload_unpacked")
    det, items = plan(root, grouping)
    out.mkdir(parents=True, exist_ok=True)
    total = 0.0
    report = {"format": det["format"], "version": det.get("version"), "rig": rig,
              "episodes": [], "skipped": [], "failed": [], "notes": [],
              "used": opened + list(det.get("used") or []), "missing": list(det.get("missing") or [])}
    if det.get("packaging"):
        report["packaging"] = det["packaging"]
    for i, it in enumerate(items):
        known = it["seconds"] if it["seconds"] is not None else 0.0      # unknown until converted; measured below
        first = not report["episodes"]
        if total + known > max_seconds and not first:
            # episodes are taken in order until the cap, as the page shows; the rest are listed
            for rest in items[i:]:
                report["skipped"].append({"name": rest["name"], "why": f"past the first {max_seconds / 60:g} minutes"})
            break
        try:
            if it["kind"] == "lerobot":
                ctx = convert_lerobot(it, rig, out, dataset)
            elif it["kind"] == "recording":
                ctx = convert_recording(it, rig, out, dataset)
            elif it["kind"] == "mcap":
                ctx = convert_mcap(it, rig, out, dataset)
            else:
                ctx = convert_video(it, rig, out, dataset)
        except Exception as e:
            report["failed"].append({"name": it["name"], "why": plain_error(e)})
            print(f"convert: {it['name']}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        secs = float(ctx.get("duration_s") or ctx["n_state_frames"] / float(ctx["fps"]))
        if first and secs > max_seconds + 1:
            # one recording longer than the whole limit: its first max_seconds are labelled, never nothing
            ctx = trim_episode(out / ctx["episode_id"], max_seconds)
            report["used"].append(f"{it['name']} is {secs / 60:.1f} minutes long; its first {max_seconds / 60:g} minutes "
                                  "were labelled.")
            secs = float(ctx["duration_s"])
        if total + secs > max_seconds + 1:
            # the measured length is longer than the file's header claimed: never label past the cap
            import shutil
            shutil.rmtree(out / ctx["episode_id"], ignore_errors=True)
            for rest in items[i:]:
                report["skipped"].append({"name": rest["name"], "why": f"past the first {max_seconds / 60:g} minutes"})
            break
        total += secs
        report["episodes"].append({"name": it["name"], "episode_id": ctx["episode_id"], "seconds": round(secs, 2),
                                   "cameras": {v: c.get("name") for v, c in ctx["cameras"].items()},
                                   "state_kind": ctx["state_kind"], "state_note": ctx.get("state_note"),
                                   "instruction": ctx.get("instruction"), "fps": ctx.get("fps"),
                                   "unsplit": bool(ctx.get("unsplit")), "packaging": ctx.get("packaging")})
    notes = sorted({e["state_note"] for e in report["episodes"] if e.get("state_note")})
    report["notes"] += notes
    report["seconds"] = round(total, 2)
    return report

