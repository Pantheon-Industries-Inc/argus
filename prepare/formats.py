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
   Cameras stored as images inside the parquet are written to H.264 at their frame times. A dataset an adapter
   recognizes by its columns goes through that adapter (the prepare/*.py that declare UPLOAD = "lerobot": HABIT
   and Galaxea), which knows what the columns mean.
2. MCAP, one file per episode. A layout a dataset adapter recognizes goes through that adapter (the
   prepare/*.py that declare UPLOAD = "mcap", found by upload_adapters: ABC-130k and RealOmin with their
   robot state, Gen-HumanEgo with its forward camera, goal and timed steps). Any other layout is read for
   its cameras (every compressed-image or compressed-video channel), its depth image channels (each with the camera
   whose topic it shares), its arm joints on a teleoperated rig (joint_state: any channel whose messages carry a joint
   vector and a gripper reading, its side from the topic, a leader or command channel as the action), every other
   number it records as a signal (mcap_signals), and its text channels (the task topic, and on a head camera a step
   topic as the timed steps).
3. HDF5 (h5, hdf5), read by what each array is (plan_hdf5, h5_kind): sibling groups that share one layout are one
   episode each, image frames or encoded images are cameras, a large 16-bit or float picture is depth, a rising array
   named for time is a clock, every other numeric array is a signal, and strings, records and attributes are the
   uploader's notes.
4. Plain video (mp4, mov, mkv, webm, avi). Either one file per episode, or one folder per episode
   holding the cameras of one episode (group_videos says how names and folders group them). MCAP and HDF5 files with
   no camera beside the videos are the episode's recorded state and sensor data, read as in 2 and 3 and placed on the
   frames' capture times from the timestamp file a recorder writes beside each video (frame_times), and so is a CSV
   table of numbers beside them (table_signals). A depth video beside a colour one goes with that camera
   (depth_videos); an infrared or mask video is left out. A file is never split: an unsplit recording is one episode,
   and the pipeline labels a long one in pieces and stitches the labels back into one timeline. When
   many files share one length, the recorder cut continuous footage into fixed-length files; the episodes
   say so, so a file that starts or ends mid-activity is read as packaging, not a truncated episode. A folder
   in a layout a dataset adapter recognizes goes through that adapter (the prepare/*.py that declare
   UPLOAD = "video": OpenAoE's clip, with its action segments and device).

An archive (.zip, .tar, .tar.gz, .tar.bz2, .tar.xz) is read as the folder it holds (open_archives). Data
Review's upload page opens archives in the browser and sends their files; this is for archives on disk.

Anything else the uploader sends next to an episode (a .txt or .json with the same name as the
video, or instruction.txt / annotations.json inside an episode folder) is passed to the model as
the uploader's own annotation, a claim to check against the video, never as truth. A recorder's metadata
file in an episode folder that names the task (its prompt, instruction or task) gives the instruction.

Recorded state is used when it has 7 values per arm or gripper (6 joints plus gripper for teleop
arms; x y z roll pitch yaw plus opening for handheld grippers). When the dataset names the values, the names settle
it (state_layout): seven named joints and no gripper stay signals. Every other number the recording keeps is a signal
(Signals), under its own name, with its shape (a 16 x 16 pressure map stays 16 x 16) and its values' names: counters
and clocks are bookkeeping, a topic that names its sensor per message is split, a sensor faster than the camera is
summarised per frame (place_on_frames), and a frame with no reading near it is NaN. Anything read but not kept is
listed with the reason, never dropped without a word. After every episode is read, the upload's depth ranges, each
signal's resting level and swing, and each episode's contacts (label/contacts.py) are measured across the upload.
Nothing in the footage is re-encoded except image frames (MCAP image channels, LeRobot image features), which are
written to H.264 at their real capture times.

Every episode's duration is known before conversion (metadata or container headers), so the
footage cap is applied before any heavy work: episodes are taken in order until the cap is reached
and the rest are listed as skipped. The report lists, in plain words, what was read and used and what
could not be read and what was done instead.

What every adapter writes, one folder per episode (label/episode.py reads it):
  context.json     the facts the harness may state: dataset, rig (profile), state kind, fps, cameras, instruction,
                   signals and contacts
  signals.npz      the other signals, one row per anchor-camera frame (absent when there are none)
  depth.json       each camera's depth stream, with depth_kmap_<view>.npy and depth_times.npz
  sources.json     per camera: the video file (packed), the episode's offset in it (base_s) and its frame count
  state.npz        state and action, one row per anchor-camera frame (absent when there is no usable state)
  times.npz        each camera's real frame times and exact pts, when frames are not on the k / fps grid
  kmap_<view>.npy  for a camera paired to the anchor camera by nearest time, its frame for each anchor frame
  instruction.txt  the instruction, for reading by eye

context.json's fields: dataset, profile (the rig: teleop_arms, handheld_gripper or ego_head), state_kind (joints,
ee_pose or none), fps, n_state_frames, cameras (per view exo, left, right: its name, width, height and desc, what
the camera is), task_label, and the task text when there is one, instruction or annotation_subtasks (a list of
{"t0", "t1", "label"}, the dataset's timed steps); uploader_annotation, notes sent with the episode in whatever form
they came (the model is shown them as claims to check); real_times names times.npz when frames carry real capture
times. sources.json gives per view the video file (packed), the episode's offset in it in seconds (base_s), its
exact frame count (n_frames) and, for a camera paired to the anchor camera by time, its kmap file.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
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
    h5s = [p for p in files if p.suffix.lower() in H5_EXT]
    vids = [p for p in files if p.suffix.lower() in VIDEO_EXT]
    h5_cams = [p for p in h5s if h5_has_camera(p)]
    if vids and not h5_cams and (mcaps or h5s) and not any(mcap_has_camera(p) for p in mcaps):
        # MCAP and HDF5 files with no camera (arm joints, grippers, a glove's pressure and hand pose) beside videos are
        # the videos' recorded state and sensor data, not episodes
        return {"format": "video", "files": [str(p) for p in vids], "state": [str(p) for p in mcaps + h5s]}
    if mcaps:
        return {"format": "mcap", "files": [str(p) for p in mcaps]}
    if h5_cams:
        return {"format": "hdf5", "files": [str(p) for p in h5_cams]}
    if vids:
        return {"format": "video", "files": [str(p) for p in vids]}
    seen = sorted({p.suffix.lower() or p.name for p in files})[:12]
    if h5s:
        raise ValueError("the HDF5 files hold no camera (no image frames or encoded images), and there are no videos "
                         "beside them to place their data against")
    raise ValueError("the upload holds no LeRobot dataset, MCAP file, HDF5 file or video"
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
    """Container facts for one video file: frame pts in decode order sorted, time base, size as shown (pixels
    that are not square made square, prepare/display.py; a display rotation of 90 or 270 degrees swaps the stored
    width and height; label/frames.py upright turns the decoded frames the same way), codec. A file stored with
    pixels that are not square, or mirrored, also says so ("sar", "mirror")."""
    from prepare import display
    with open_checked(path) as c:
        st = video_stream(c, path)
        pts = sorted(p.pts for p in c.demux(st) if p.size and p.pts is not None and not p.is_discard)
        rate = st.average_rate or st.guessed_rate
        w, h, tb, codec = st.codec_context.width, st.codec_context.height, st.time_base, st.codec_context.name
    geom = display.geometry(str(path))
    extra = {}
    if tuple(geom["stored"]) == (w, h) and display.needs_resample(geom):
        w, h = display.square_size(geom)
        extra["sar"] = f"{geom['sar'].numerator}:{geom['sar'].denominator}"
    if geom["mirror"]:
        extra["mirror"] = True
    rot = display_rotation(path)
    if rot in (90, 270):
        w, h = h, w
    return {"pts": np.asarray(pts, dtype=np.int64), "time_base": tb, "width": w, "height": h,
            "codec": codec, "fps": float(rate) if rate else None, "rotation": rot, **extra}


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


MAX_EXTRA_CAMERAS = 4  # cameras beyond the scene and two mounted ones, sent to the model too; more are listed unused
# a view whose name says it senses touch (a GelSight or DIGIT image, a tactile heatmap) is never the scene camera or the
# camera on a gripper: it is another view, named as the dataset names it, and the pair of them (left and right) are two
# sensors, not the two eyes of one stereo camera. A camera and a signal say touch with the words of TOUCH_CORE; a camera
# also with a tactile camera's brand or picture (gelsight, digit, xense, heatmap), and a signal with what it measures (a
# pressure, a contact flag, a force, a force-sensing resistor or a piezo pad). A signal never uses the camera words,
# since digit also names a finger in a hand pose (hand.digits, digit_1_tip).
TOUCH_CORE = ("tactile", "touch", "visuotactile", "haptic", "taxel", "skin")
SENSING_WORDS = TOUCH_CORE + ("gelsight", "digit", "xense", "heatmap")
TOUCH_WORDS = TOUCH_CORE + ("pressure", "contact", "force", "fsr", "piezo")
# a name with one of these words is a command, not a measurement (action.gripper_force, gripper_force_cmd)
COMMAND_WORDS = ("action", "cmd", "command", "target", "setpoint", "goal", "desired")


def _names_word(name: str, words) -> bool:
    """Whether a word of a name (tokens) is one of words, alone, plural or numbered (tactile, tactiles, digit_0, fsr0),
    never as part of a longer word: digital is not digit and reinforcement is not force."""
    for t in tokens(name):
        stem = t.rstrip("0123456789")
        if stem in words or (stem.endswith("s") and stem[:-1] in words):
            return True
    return False


def is_sensing(name: str) -> bool:
    """Whether a camera's name says it senses touch (SENSING_WORDS): a GelSight or DIGIT image, a tactile heatmap."""
    return _names_word(name, SENSING_WORDS)


def names_touch(name: str) -> bool:
    """Whether a signal's name says it measures touch (TOUCH_WORDS: right_pressure, tactile_left_raw, right_contact)
    and is not a command (COMMAND_WORDS): a commanded force is not a measured one."""
    return _names_word(name, TOUCH_WORDS) and not _names_word(name, COMMAND_WORDS)


def assign_views(names: list[str], rig: str) -> tuple[dict, list]:
    """{view: camera name}: one scene and two mounted cameras by role, then every other camera as extra1, extra2, ...
    (up to MAX_EXTRA_CAMERAS, in the scene cameras' rank order), and the names left unused."""
    out = {}
    sensing = [nm for nm in names if is_sensing(nm)]
    # names that also say wrist/hand/gripper take a side before names that only carry a side
    for nm in sorted((n for n in names if n not in sensing), key=lambda n: (0 if is_mount_named(n) else 1, n)):
        v = mounted_side(nm)
        if v and v not in out:
            out[v] = nm
    rest = sorted((nm for nm in names if nm not in out.values() and nm not in sensing),
                  key=lambda s: (scene_rank(s), s))
    if rest:
        out["exo"] = rest[0]
    if rig == "handheld_gripper" and list(out) == ["exo"]:
        # one camera on a handheld rig is the gripper's own camera (FastUMI single_arm), not a scene camera
        out = {"right": out["exo"]}
    # the other eye of a stereo camera already shown (/zed/left/image and /zed/right/image) adds a near-copy of its
    # picture, so it is left out; every other camera is sent
    for nm in rest[1:] + sorted(sensing):
        n_extra = sum(1 for k in out if k.startswith("extra"))
        if n_extra < MAX_EXTRA_CAMERAS and (nm in sensing or not any(_stereo_twin(nm, c) for c in out.values())):
            out[f"extra{n_extra + 1}"] = nm
    unused = [nm for nm in names if nm not in out.values()]
    return out, unused


def _stereo_twin(a: str, b: str) -> bool:
    """True when two camera names differ only by a left/right word (the two eyes of one stereo camera)."""
    swap = lambda s: re.sub(r"left|right", lambda m: {"left": "right", "right": "left"}[m.group(0).lower()], s,
                            flags=re.I)
    return a != b and swap(a).lower() == b.lower()


NOT_RGB = re.compile(r"depth|conf|disparity|mask|seg|thermal|infrared|(^|/)ir(/|$)|vis_", re.I)
TEXT_TOPIC = re.compile(r"instruction|task|annotation|language|prompt", re.I)
TEXT_MSGS_MAX = 50           # a text topic with more distinct messages than this is a log (a heartbeat), not notes
# a topic named for the task itself (/task, /instruction), which a sub-topic (/task/subtask, /task/health) is not
TASK_TOPIC_NAME = re.compile(r"(language_)?(instruction|task|task_description|prompt|goal)", re.I)
# a topic named for the task's timed steps (MicroAGI's /task/subtask)
STEP_TOPIC_NAME = re.compile(r"sub_?tasks?|steps?", re.I)


def _is_step(topic: str) -> bool:
    return bool(STEP_TOPIC_NAME.fullmatch(topic.rstrip("/").rsplit("/", 1)[-1]))


def add_text(texts: dict, counts: dict, topic: str, t_ns: int, x) -> None:
    """One text message into texts[topic], (log time ns, text) in time order. A step topic keeps every message as the
    dataset sent it (MicroAGI sends one per short window, so ten windows of the same step are ten steps); any other
    topic keeps a message repeated until the next one (a re-sent instruction, a heartbeat) once."""
    seq = texts.setdefault(topic, [])
    counts[topic] = counts.get(topic, 0) + 1
    if not seq or seq[-1][1] != x or _is_step(topic):
        seq.append((t_ns, x))
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


DEPTH_TOPIC = re.compile(r"depth", re.I)


def depth_image(msg) -> tuple[np.ndarray, float | None] | None:
    """(depth array, metres per unit) of a depth image message: a raw image in 16UC1 / mono16 (millimetres, as ROS
    defines it) or 32FC1 (metres), or a compressed one (a 16-bit PNG, with the 12-byte header ROS's compressedDepth
    adds); None for anything else."""
    import io
    enc = str(_field(msg, "encoding") or _field(msg, "format") or "").lower()
    data = _field(msg, "data", binary=True)
    data = bytes(data) if data is not None else b""
    w, h = int(_field(msg, "width") or 0), int(_field(msg, "height") or 0)
    if w and h and data:
        step = int(_field(msg, "step") or 0)
        if enc in ("16uc1", "mono16") and len(data) >= h * (step or w * 2):
            step = step or w * 2
            a = np.frombuffer(data, np.uint16, count=h * step // 2).reshape(h, step // 2)[:, :w]
            return a.copy(), 0.001
        if enc == "32fc1" and len(data) >= h * (step or w * 4):
            step = step or w * 4
            a = np.frombuffer(data, np.float32, count=h * step // 4).reshape(h, step // 4)[:, :w]
            return a.copy(), 1.0
        return None
    i = data.find(b"\x89PNG")
    if i < 0:
        return None
    from PIL import Image
    try:
        im = Image.open(io.BytesIO(data[i:]))
        a = np.asarray(im)
    except Exception:
        return None
    if a.ndim != 2:
        return None
    return a.astype(np.uint16), (0.001 if a.dtype == np.uint16 else None)


def depth_partner(depth_topic: str, cams: dict) -> str | None:
    """The view whose camera topic shares the longest leading path with a depth topic (/realsense/aligned_depth_to_color
    goes with /realsense/color), or None when none shares any."""
    parts = [x for x in depth_topic.split("/") if x]
    best, n = None, 0
    for v, t in cams.items():
        q = [x for x in t.split("/") if x]
        k = 0
        while k < min(len(parts), len(q)) and parts[k] == q[k]:
            k += 1
        if k > n:
            best, n = v, k
    return best


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
    # the words every video file of a camera carries (exo_cam-images-rgb is the camera exo_cam)
    words = [w for w in base.split("_") if w]
    kept = [w for w in words if w.lower() not in GENERIC_VIDEO_WORDS]
    return "_".join(kept or words)[:32] or view


def set_uploader_notes(ctx: dict, obj) -> None:
    """The uploader's notes for an episode, as text for the prompt (uploader_annotation) and as sent (uploader_notes,
    which the board shows beside the labels)."""
    text = annotation_text(obj)
    if not text:
        return
    ctx["uploader_annotation"] = text
    try:
        json.dumps(obj)
        ctx["uploader_notes"] = obj
    except (TypeError, ValueError):
        ctx["uploader_notes"] = text


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


# A state's value names settle what the 7 values of one actor are, when the dataset gives them. A seventh value named
# for a gripper (left_gripper.pos, gripper) is six values and a gripper, and six named for a position and an
# orientation (x, y, z, roll, pitch, yaw) make that a pose whatever the rig; seven joints and no gripper (a Franka arm,
# fr3_left_joint1..7, read as six joints and a gripper until the 2026-10-02 audit) or a quaternion are not the layout
# the checks read. Each name is read as words (state_words), split at its separators and camelCase, lowercased, and
# a unit after the last word dropped, so x_m, eefPosX and roll_rad end in their axis. A name with a word for a
# quantity other than a position (joint1_vel, joint3_effort, force_x) is never a state the checks read, and a name
# whose frame or orientation word is followed by a number (cartesian_position_0, robot0_eef_pos_0, wrist_rot_0) is a
# pose whose axes the rule cannot read, never joints by width. A frame word with no number after it names a joint as
# often (base_rotation, tool_roll, flange, end_effector), and a position word alone with its index (HABIT's position_0
# to position_13) says no frame, so names that say none of these (position_0, motor_3) leave the width rule.
STATE_GRIPPER_NAME = re.compile(r"grip|finger|jaw|claw|opening", re.I)
STATE_JOINT_NAME = re.compile(r"joint|(^|[^a-z])j\d|waist|shoulder|elbow|forearm|wrist", re.I)
STATE_QUAT_NAME = re.compile(r"(^|[._/ -])q[._/ -]?[wxyz]$|(^|[^a-z])quat", re.I)
STATE_WORD_SPLIT = re.compile(r"[._/ -]+|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Za-z])(?=[0-9])")
STATE_UNIT_WORDS = {"m", "mm", "cm", "rad", "deg", "degree", "degrees"}
STATE_AXIS_WORDS = {"x", "y", "z", "roll", "pitch", "yaw", "rx", "ry", "rz"}
STATE_POSITION_AXES = {"x", "y", "z"}
STATE_POSITION_WORDS = {"pos", "position", "positions"}
STATE_NOT_POSITION_WORDS = {"vel", "velocity", "velocities", "speed", "effort", "efforts", "torque", "torques",
                            "current", "currents", "force", "forces", "acc", "accel", "acceleration"}
STATE_FRAME_WORDS = {"cartesian", "eef", "ee", "tcp", "pose", "effector", "flange", "tool", "rot", "rotation",
                     "orientation"}


def state_words(name: str) -> list[str]:
    """The words of a state value's name for state_layout: split at . _ / space and hyphen, at a camelCase boundary
    and before a number, lowercased, with a unit word after the last word dropped (x_m is x, eefPosX is eef pos x)."""
    words = [w.lower() for w in STATE_WORD_SPLIT.split(str(name)) if w]
    return words[:-1] if len(words) > 1 and words[-1] in STATE_UNIT_WORDS else words


def state_layout(dims: int, rig: str, names: list[str] | None = None) -> tuple[str, str | None]:
    """(state_kind, note). 7 or 14 values per frame are 1 or 2 actors of 6 + gripper; anything else is labelled from
    video. names, one per value when the dataset gives them, settle the layout (STATE_GRIPPER_NAME above); names that
    say neither a gripper, joints nor a pose keep the width rule. Six names are a pose when every one's last word
    (state_words) is an axis and at least one names a position, its last word exactly x, y or z (wrist_x, ee.pos.x,
    x_m, eefPosX; not rx). Joints are named for the axis they turn about and never for a position, so an arm's
    shoulder_yaw, elbow_pitch and wrist_roll (or a humanoid's waist_yaw), joint angles in radians, stay joints rather
    than be read as metres, while a pose of the wrist frame (wrist_x .. wrist_yaw) stays a pose. A name of a velocity,
    an effort or a force is not a position on any rig, and a name of a frame followed by a number, without axes the
    rule reads (cartesian_position_0, wrist_rot_0), is a pose: never joints on an arm rig, joint words or not, and the
    pose it already is on a gripper rig."""
    if rig == "ego_head":
        return "none", None
    per = "arm (six joints and a gripper)." if rig == "teleop_arms" else "gripper (a 6D pose and an opening)."
    if dims not in (7, 14):
        return "none", (f"Labelled from the video: the recorded state has {dims} values per frame, and our checks "
                        "expect 7 per " + per)
    kind = STATE_KIND[rig]
    if not names or len(names) != dims:
        return kind, None
    names = [str(x) for x in names]
    words = {x: state_words(x) for x in names}
    groups = [names[i:i + 7] for i in range(0, dims, 7)]
    if any(STATE_QUAT_NAME.search(x) for x in names):
        return "none", ("Labelled from the video: the recorded state's value names give a quaternion, and our checks "
                        "read a position, a roll, pitch and yaw and an opening per gripper.")
    # a word for a present position (current_pos) is not a current
    other = next((x for x in names if set(words[x]) & STATE_NOT_POSITION_WORDS
                  and not set(words[x]) & STATE_POSITION_WORDS), None)
    if other:
        return "none", (f"Labelled from the video: the recorded state's value names ({other}) give a velocity, an "
                        "effort or another quantity that is not a position, and our checks read the positions of each "
                        + per)
    framed = next((x for x in names if set(words[x]) & STATE_FRAME_WORDS and words[x][-1].isdigit()
                   and not STATE_GRIPPER_NAME.search(x)), None)
    by_width = ("none", f"Labelled from the video: the recorded state's value names give a position or a pose without "
                        f"the axes our checks read ({framed}), not six joints and a gripper per arm.") \
        if framed and kind != "ee_pose" else (kind, None)
    seventh = all(STATE_GRIPPER_NAME.search(g[6]) for g in groups)
    if seventh and not any(STATE_GRIPPER_NAME.search(x) for g in groups for x in g[:6]):
        if all(words[x][-1] in STATE_AXIS_WORDS for g in groups for x in g[:6]) and \
                all(any(words[x][-1] in STATE_POSITION_AXES for x in g[:6]) for g in groups):
            return "ee_pose", None
        if by_width[0] == "none":
            return by_width
        if all(STATE_JOINT_NAME.search(x) for g in groups for x in g[:6]):
            return "joints", None
        return by_width
    if any(STATE_GRIPPER_NAME.search(x) for x in names):
        return "none", ("Labelled from the video: the recorded state's value names put a gripper elsewhere than "
                        "seventh in each group of seven, and our checks read six values and then the gripper.")
    if all(STATE_JOINT_NAME.search(x) for x in names):
        return "none", (f"Labelled from the video: the recorded state's value names give {dims} joints and no gripper, "
                        "and our checks read six joints and a gripper per arm.")
    return by_width


def state_value_names(feats: dict, state) -> list[str] | None:
    """The names a LeRobot dataset gives observation.state's values in meta/info.json, one per value, for
    state_layout; None when there is no state or the names do not give one per value."""
    if state is None:
        return None
    return value_names((feats.get("observation.state") or {}).get("names"), state.shape[1])


# Per-frame columns that are the table's bookkeeping, not a recording: never kept as a signal
SIGNAL_SKIP = re.compile(r"(^|\.)(index|timestamp)$|_index$")
# A per-frame array of more values than this (a point cloud, a flattened image) is a picture, not numbers a person or a
# model reads one by one; it is listed among the signals left out, never dropped without a word. A tactile pressure map
# of 64 x 64 cells is still a signal.
SIGNAL_MAX_VALUES = 4096
SIGNAL_MIN_READINGS = 0.5         # the share of its rows a column must have a reading at (any finite value) to be kept


class Signals(dict):
    """{name: (n, values) array} of the numbers a recording keeps beside its cameras and arm state, under the dataset's
    own names, with what is known about each (meta[name]: "shape", the array's own shape per frame when it is not a flat
    vector, such as a 16 x 16 pressure map; "names", one per value, when the dataset names them; "source", where it was
    read) and the numbers that were not kept (left_out: [(name, why)]). A plain dict of arrays is read the same way."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.meta: dict[str, dict] = {}
        self.left_out: list[tuple[str, str]] = []
        self.clocks: dict[str, np.ndarray] = {}      # per-frame clocks the recording keeps (write_signals)

    def add(self, name: str, a: np.ndarray, shape=None, names=None, source: str | None = None) -> None:
        self[name] = a
        m = {}
        if shape is not None and tuple(int(x) for x in shape) != (a.shape[1],):
            m["shape"] = [int(x) for x in shape]
        if names is not None and len(names) == a.shape[1]:
            m["names"] = [str(x) for x in names]
        if source:
            m["source"] = source
        self.meta[name] = m


def is_clock(x: np.ndarray) -> bool:
    """Whether a column rises like a clock: it never falls, rises at 90% of its rows or more, and by a steady step (the
    spread of its steps under half their median). A base driving at a steady speed rises the same way, so a column is a
    clock only when its name says time as well (is_named_clock)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < COUNTER_MIN_MESSAGES:
        return False
    d = np.diff(x)
    if (d < 0).any() or (d > 0).mean() < 0.9:
        return False
    up = d[d > 0]
    return float(np.std(up)) < 0.5 * float(np.median(up))


TIME_WORDS = {"time", "times", "timestamp", "timestamps", "stamp", "stamps", "ts", "t", "epoch"}
TIME_UNITS = {"s", "ns", "us", "ms", "sec", "secs", "nsec", "nsecs", "usec", "usecs", "msec", "msecs", "nanos",
              "micros", "millis", "nanosec", "nanosecs", "utc"}


def is_time_name(name) -> bool:
    """Whether a name says it holds times: its last word is a time word (sensorTimestamp, recv_time, t), or a unit
    after one (epoch_ns, stamp.nanosec, timestampUtc)."""
    words = tokens(str(name))
    if words and words[-1] in TIME_UNITS:
        words = words[:-1]
    return bool(words) and words[-1] in TIME_WORDS


def is_named_clock(name, values) -> bool:
    """Whether a column is a clock: its name says time (is_time_name) and it rises like one (is_clock)."""
    return is_time_name(name) and is_clock(values)


COUNTER_NOTE = "counts rows one by one, so it is bookkeeping"


def is_counter(values) -> bool:
    """Whether a column counts rows one by one (a sequence number, seq, frame_id): whole numbers that never fall and
    step by exactly 1 at 90% of its rows or more, since a recorder can stamp a row twice. Such a column is bookkeeping,
    as an MCAP channel's counters are (_counters), so it is left out with COUNTER_NOTE."""
    x = np.asarray(values, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if len(x) < COUNTER_MIN_MESSAGES or not np.all(x == np.round(x)):
        return False
    d = np.diff(x)
    return bool((d >= 0).all() and (d == 1).mean() >= 0.9)


def value_names(names, dims: int) -> list[str] | None:
    """One name per value from a dataset's own naming of a feature: a list (["x", "y", "z"]), a dict holding one list
    ({"motors": [...]}), or nested lists in the feature's shape; None when they do not give exactly dims names."""
    if isinstance(names, dict):
        vals = [v for v in names.values() if isinstance(v, (list, tuple))]
        names = vals[0] if len(vals) == 1 else None
    if not isinstance(names, (list, tuple)):
        return None
    flat = []

    def walk(x):
        if isinstance(x, (list, tuple)):
            for y in x:
                walk(y)
        else:
            flat.append(str(x))
    walk(names)
    return flat if len(flat) == dims else None


def recorded_signals(df, used, n: int, features: dict | None = None) -> Signals:
    """Every other numeric per-frame column of an episode's table, under the dataset's own name, as (n, values) arrays:
    the columns that are not bookkeeping (SIGNAL_SKIP), not already read (used: the state, the action, the cameras)
    and at least n rows long. A reading missing at some frames is still a reading, NaN at those frames, while a column
    with a reading at fewer than SIGNAL_MIN_READINGS of its rows is left out. An array per frame keeps its shape (a
    pressure map is 16 x 16, not 256 numbers in a row) and the names the dataset gives its values (features:
    meta/info.json's, with "shape" and "names"). The harness shows them to the model as they are (label/episode.py), so
    nothing a dataset records is dropped because our checks do not know what it means: a mobile robot's base and torso,
    joint velocities, forces, a tactile glove's pressure map. A column left out (wider than SIGNAL_MAX_VALUES, with too
    few readings, a counter) is listed in left_out with the reason."""
    out = Signals()
    if df is None:
        return out
    features = features or {}
    for c in df.columns:
        if c in used or SIGNAL_SKIP.search(str(c)) or str(c).startswith("observation.images"):
            continue
        if (features.get(c) or {}).get("dtype") == "video":
            continue
        if (features.get(c) or {}).get("dtype") == "image":
            a = _image_cells(df[c])               # only the small per-pad images reach here (read_root image_signals)
            if a is None:
                continue
            out.add(str(c), a[0][:n] if len(a[0]) >= n else a[0], shape=a[1], source="image column " + str(c))
            continue
        a = _cells(df[c])
        if a is None or not a.shape[1] or len(a) < n:
            continue
        # a row has a reading when any of its values does, as checks/sensors.py counts it: a pressure map with one dead
        # cell still reads at every frame
        has_reading = np.isfinite(a).any(axis=1)
        if has_reading.mean() < SIGNAL_MIN_READINGS:
            out.left_out.append((str(c), "no reading at most frames"))
            continue
        a = np.where(np.isfinite(a), a, np.nan)
        if a.shape[1] > SIGNAL_MAX_VALUES:
            out.left_out.append((str(c), f"{a.shape[1]} values per frame, more than the {SIGNAL_MAX_VALUES} a signal "
                                         "holds"))
            continue
        if a.shape[1] == 1 and is_named_clock(c, a[:, 0]):
            # a clock: kept for the sync check, not shown
            out.clocks[str(c)] = a[:n, 0]
            continue
        if a.shape[1] == 1 and is_counter(a[:, 0]):
            out.left_out.append((str(c), COUNTER_NOTE))
            continue
        f = features.get(c) or {}
        shape = f.get("shape") if isinstance(f.get("shape"), (list, tuple)) and int(np.prod(f["shape"])) == a.shape[1] \
            else _cell_shape(df[c], a.shape[1])
        out.add(str(c), a[:n], shape=shape, names=value_names(f.get("names"), a.shape[1]), source="column " + str(c))
    return out


def _image_cells(col):
    """(rows, (h, w)) of a column of small encoded images (PNG bytes, as LeRobot stores an image feature), each read as
    grey values; None when they do not decode to one size."""
    import io
    from PIL import Image
    rows, shape = [], None
    for cell in col.to_numpy():
        b = cell.get("bytes") if isinstance(cell, dict) else cell
        try:
            a = np.asarray(Image.open(io.BytesIO(bytes(b))).convert("L"), dtype=np.float64)
        except Exception:
            return None
        if shape is None:
            shape = a.shape
        if a.shape != shape:
            return None
        rows.append(a.ravel())
    return (np.stack(rows), list(shape)) if rows else None


def _cell_shape(col, dims: int):
    """The shape of one cell of a table column (a 16 x 16 list of lists), when it holds exactly dims numbers."""
    try:
        s = np.asarray(_nested(col.iloc[0]), dtype=np.float64).shape
    except Exception:
        return None
    return s if s and int(np.prod(s)) == dims else None


def _nested(x):
    """A cell as nested lists: parquet gives a list of lists as an object array of arrays."""
    if isinstance(x, np.ndarray) and x.dtype == object:
        return [_nested(y) for y in x]
    return x


def write_signals(ep: Path, ctx: dict, signals: dict | None) -> None:
    """signals.npz beside the state (keys s0, s1, ...) and ctx["signals"], [{name, key, dims, shape, names, source}],
    trimmed to the episode's n_state_frames; nothing when there are none. What a reader read but did not keep
    (Signals.left_out) is written to ctx["source"]["unused_signals"], so the report names it."""
    n = int(ctx.get("n_state_frames") or 0)
    meta = getattr(signals, "meta", {}) or {}
    left = list(getattr(signals, "left_out", []) or [])
    keep = {}
    for k, v in (signals or {}).items():
        if n and len(v) >= n:
            keep[k] = v
        else:
            left.append((k, f"{len(v)} samples, fewer than the episode's {n} frames"))
    if left:
        ctx.setdefault("source", {})["unused_signals"] = [f"{k} ({why})" for k, why in left]
    clocks = getattr(signals, "clocks", None) or {}
    if len(clocks) >= 2 and n:
        # each clock's offset from the first, its median and spread in milliseconds (checks/sensors.py reads them)
        names_ = list(clocks)
        base = np.asarray(clocks[names_[0]][:n], dtype=np.float64)
        to_s = _seconds_scale(base[:1000])
        ctx["clocks"] = [{"name": k, "offset_ms": round(float(np.nanmedian((np.asarray(v[:n], dtype=np.float64) - base)
                                                                          * to_s)) * 1000, 2),
                          "spread_ms": round(float(np.nanstd((np.asarray(v[:n], dtype=np.float64) - base)
                                                             * to_s)) * 1000, 2)}
                         for k, v in clocks.items()]
    if not keep:
        ctx.pop("signals", None)
        return
    np.savez(ep / "signals.npz", **{f"s{i}": np.asarray(v[:n], dtype=np.float32) for i, v in enumerate(keep.values())})
    ctx["signals"] = [{"name": k, "key": f"s{i}", "dims": int(v.shape[1]), **(meta.get(k) or {})}
                      for i, (k, v) in enumerate(keep.items())]


# ---------------------------------------------------------------- depth

# A depth stream is a per-pixel distance picture on its own clock, kept beside the colour camera it belongs to: never a
# camera of its own, never re-encoded when it is already a file (a RealSense FFV1 16-bit video), and in metres only when
# the recording says what one unit is (ROS 16UC1 is millimetres and 32FC1 metres by the message's own definition, or a
# depth_scale the recorder wrote). Otherwise its values are relative distances, and everything downstream says so.
DEPTH_SCALE_KEY = re.compile(r"depth[_ ]?(scale|units?)$", re.I)


def depth_scale_from(folder: Path) -> float | None:
    """Metres per depth unit from a recorder's metadata in the episode's folder: a JSON file (at most 1 MB) holding a
    number under a key named depth_scale or depth_units (a RealSense's 0.001). None when nothing says."""
    found = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if DEPTH_SCALE_KEY.search(str(k)) and isinstance(v, (int, float)) and 0 < v < 10:
                    found.append(float(v))
                walk(v)
        elif isinstance(x, list):
            for v in x[:200]:
                walk(v)
    for p in sorted(Path(folder).glob("*.json")):
        if p.stat().st_size <= 1_000_000:
            walk(_read_json(p) or {})
    return found[0] if found and len(set(found)) == 1 else None


def depth_camera(depth_name: str, cameras: dict[str, str], scene: str) -> tuple[str, str]:
    """(view, source note) for a depth stream of an HDF5 file or a LeRobot dataset: the one camera ({view: name}) whose
    name has every word of the depth's name, leaving out the words for what a file holds (depth, images) and, when
    there are several cameras, the words every camera's name shares (the dataset's own prefix, observation). So
    observations/depth/cam_high goes with observations/images/cam_high, and observation.depth.left_wrist with
    observation.images.cam_left_wrist. A depth stream whose words fit no camera or several (observation.depth.wrist
    beside a left and a right wrist camera) goes with the scene camera, with the reason added to the note, as
    depth_videos asks for exactly one match."""
    names = {v: set(tokens(name)) for v, name in cameras.items()}
    shared = set.intersection(*names.values()) if len(names) > 1 else set()
    words = set(tokens(depth_name)) - NON_COLOUR - GENERIC_VIDEO_WORDS - shared
    fits = [v for v, name in names.items() if words and words <= name]
    if len(fits) == 1:
        return fits[0], depth_name
    return scene, f"{depth_name} (its words name no one camera, so it goes with the scene camera)"


def depth_entry(ep: Path, view: str, path: Path, t_depth: np.ndarray, t_anchor: np.ndarray, pts: np.ndarray,
                scale_m: float | None, source: str) -> tuple[dict, dict]:
    """(depth.json entry, times) for one camera's depth stream: its file, its frame for each anchor frame (nearest in
    time, depth_kmap_<view>.npy), its frame times and exact pts, and metres per unit when known."""
    ep.mkdir(parents=True, exist_ok=True)     # a LeRobot episode writes depth before finish_episode makes its folder
    km = nearest(np.asarray(t_depth, dtype=np.float64), np.asarray(t_anchor, dtype=np.float64))
    np.save(ep / f"depth_kmap_{view}.npy", km)
    entry = {"packed": str(Path(path).resolve()), "base_s": 0.0, "n_frames": int(len(t_depth)),
             "kmap": f"depth_kmap_{view}.npy", "scale_m": scale_m, "source": source,
             # a disparity (or inverse depth) stream reads larger where nearer; label/depth.py draws it reversed
             "kind": "disparity" if re.search(r"disparit|inverse", source or "", re.I) else "depth"}
    return entry, {f"depth_{view}": np.asarray(t_depth, dtype=np.float64), f"depth_{view}_pts": np.asarray(pts)}


def write_depth(ep: Path, ctx: dict, depth: dict | None, times: dict | None = None) -> None:
    """depth.json ({view: entry}), depth_times.npz (each stream's frame times and pts, depth_entry) and ctx["depth"]
    ({view: {units, source}}); nothing when there is no depth."""
    if not depth:
        ctx.pop("depth", None)
        return
    (ep / "depth.json").write_text(json.dumps(depth, indent=1))
    if times:
        np.savez(ep / "depth_times.npz", **times)
    ctx["depth"] = {v: {"units": ("metres" if d.get("scale_m") else "relative"), "scale_m": d.get("scale_m"),
                        "kind": d.get("kind", "depth"), "source": d.get("source")} for v, d in depth.items()}


def probe_depth(path: Path) -> dict:
    """A depth video's frame pts, time base and pixel format (gray16le for a 16-bit depth stream)."""
    with open_checked(path) as c:
        st = video_stream(c, path)
        pts = sorted(p.pts for p in c.demux(st) if p.size and p.pts is not None and not p.is_discard)
        return {"pts": np.asarray(pts, dtype=np.int64), "time_base": st.time_base, "width": st.codec_context.width,
                "height": st.codec_context.height, "pix_fmt": st.codec_context.pix_fmt}


class DepthWriter:
    """One camera's depth frames streamed to a lossless 16-bit FFV1 mkv at their real times (an MCAP's or an HDF5
    file's depth images). Values are kept as recorded: 16-bit integers as they are, metres (float) as millimetres, so
    scale_m says what one stored unit is."""

    def __init__(self, out: Path):
        self.out, self.pts, self.dst, self.st, self.scale_m = out, [], None, None, None

    def add(self, t_s: float, a: np.ndarray, scale_m: float | None) -> None:
        import av
        a = np.asarray(a)
        if a.ndim == 3:
            a = a[..., 0]
        if a.dtype.kind == "f":
            a = np.nan_to_num(a * 1000.0, nan=0.0, posinf=0.0, neginf=0.0)
            a = np.clip(a, 0, 65535).astype(np.uint16)
            scale_m = 0.001 if scale_m is None or scale_m == 1.0 else scale_m * 0.001
        a = a.astype(np.uint16)
        if self.dst is None:
            self.scale_m = scale_m
            self.dst = av.open(str(self.out), "w", format="matroska")
            self.st = self.dst.add_stream("ffv1", rate=30)
            self.st.width, self.st.height = a.shape[1], a.shape[0]
            self.st.pix_fmt = "gray16le"
            self.st.time_base = Fraction(1, TIME_BASE_DEN)
            self.st.codec_context.time_base = Fraction(1, TIME_BASE_DEN)
        if a.shape != (self.st.height, self.st.width):
            return                        # a frame of another size cannot join this stream
        p = int(round(t_s * TIME_BASE_DEN))
        if self.pts and p <= self.pts[-1]:
            p = self.pts[-1] + 1
        fr = av.VideoFrame.from_ndarray(np.ascontiguousarray(a), format="gray16le")
        fr.pts, fr.time_base = p, Fraction(1, TIME_BASE_DEN)
        self.pts.append(p)
        for pkt in self.st.encode(fr):
            self.dst.mux(pkt)

    def close(self) -> int:
        if self.dst is not None:
            for pkt in self.st.encode():
                self.dst.mux(pkt)
            self.dst.close()
        return len(self.pts)


def finish_episode(ep: Path, ctx: dict, sources: dict, state=None, action=None, times: dict | None = None,
                   signals: dict | None = None) -> dict:
    ep.mkdir(parents=True, exist_ok=True)
    if state is not None and ctx.get("state_kind") != "none":
        arrs = {"state": np.asarray(state, dtype=np.float32)}
        if action is not None and np.shape(action) == np.shape(state):
            arrs["action"] = np.asarray(action, dtype=np.float32)
        np.savez(ep / "state.npz", **arrs)
    write_signals(ep, ctx, signals)
    if times:
        np.savez(ep / "times.npz", **times)
        ctx["real_times"] = "times.npz"
    (ep / "sources.json").write_text(json.dumps(sources, indent=1))
    (ep / "instruction.txt").write_text((ctx.get("instruction") or "") + "\n")
    (ep / "context.json").write_text(json.dumps(ctx, indent=1, default=str))
    return ctx


def video_views_episode(ep: Path, files: dict, rig: str, dataset: str, extra: dict, shared_clock: bool = False,
                        prs: dict | None = None, real: dict | None = None, state=None, action=None,
                        descs: dict | None = None, signals: dict | None = None, depth: dict | None = None,
                        state_names: list | None = None) -> dict:
    """An episode made of video files {view: (camera name, path)}: real frame times from each file's pts,
    the first view in harness order as the anchor, the others paired by nearest time. Separate video files
    have no common clock, so each starts at its own first frame; cameras written from one MCAP
    (shared_clock) keep their offsets and are measured from the anchor's first frame. real gives every view's
    capture times on one recorder clock (frame_times), which then place the cameras against each other; state and
    action are rows on the anchor's frames (joint_state). descs {view: text} describes a camera the reader knows
    more about than its slot says (the prompt's camera line). depth {view: {"path", "real" (its own capture times on
    the same recorder clock, or None), "scale_m", "source"}} is each camera's depth stream (depth_entry), timed as its
    colour camera is. state_names gives the state's value names for state_layout."""
    from label import episode as me
    prs = prs or {v: probe(p) for v, (_, p) in files.items()}
    order = me.order_views(files)
    anchor = order[0]
    use_real = bool(real) and all(real.get(v) is not None for v in order)
    zero = float(real[anchor][0]) if use_real else \
        float(prs[anchor]["pts"][0] * prs[anchor]["time_base"]) if shared_clock else None
    # the recorder's time of the episode's first frame, so times the uploader gives on that clock land on the episode's
    origin = extra.pop("clock_origin_s", None)
    if use_real:
        extra["clock_start_s"] = zero
    elif shared_clock and origin is not None:
        extra["clock_start_s"] = origin + zero

    def seconds_of(v):
        if use_real:
            return np.asarray(real[v], dtype=np.float64) - zero
        t = prs[v]["pts"].astype(np.float64) * float(prs[v]["time_base"])
        return t - (zero if shared_clock else t[0])
    ta = seconds_of(anchor)
    ep.mkdir(parents=True, exist_ok=True)
    sources, times, cams = {}, {}, {}
    for v in order:
        name, path = files[v]
        pr = prs[v]
        t = seconds_of(v)
        times[v], times[f"{v}_pts"] = t, pr["pts"]
        sources[v] = {"packed": str(Path(path).resolve()), "base_s": 0.0, "n_frames": int(len(pr["pts"])),
                      "camera_key": name}
        if v != anchor:
            km = nearest(t, ta)
            if not (len(t) == len(ta) and np.array_equal(km, np.arange(len(ta)))):
                np.save(ep / f"kmap_{v}.npy", km)
                sources[v]["kmap"] = f"kmap_{v}.npy"
        cams[v] = camera_entry(v, name, pr, rig)
        if (descs or {}).get(v):
            cams[v]["desc"] = descs[v]
    dep, dtimes = {}, {}
    for v, d in (depth or {}).items():
        if v not in files:
            continue
        try:
            pd_ = probe_depth(Path(d["path"]))
        except Exception:
            extra.setdefault("source", {}).setdefault("unused_depth", []).append(
                f"{Path(d['path']).name} (could not be read)")
            continue
        if not len(pd_["pts"]):
            continue
        if use_real and d.get("real") is not None and len(d["real"]) == len(pd_["pts"]):
            td = np.asarray(d["real"], dtype=np.float64) - zero
        else:
            # no capture times of its own: the depth stream is timed by its file, from the colour camera's first frame
            td = pd_["pts"].astype(np.float64) * float(pd_["time_base"])
            td = (td - (zero if shared_clock and zero is not None else td[0])
                  + (times[v][0] if not shared_clock else 0.0))
        entry, tz = depth_entry(ep, v, Path(d["path"]), td, ta, pd_["pts"], d.get("scale_m"), d.get("source") or "")
        entry.update(width=pd_["width"], height=pd_["height"], pix_fmt=pd_["pix_fmt"])
        dep[v] = entry
        dtimes.update(tz)
    # the rate is measured from the frame times (a header can claim any rate); the length is the span of
    # the frames plus one frame, so it matches what the labeller samples
    step = float(np.median(np.diff(ta))) if len(ta) > 1 else 1 / 30
    fps = 1.0 / step if step > 0 else 30.0
    ctx = {"dataset": dataset, "profile": rig, "state_kind": "none", "episode_id": ep.name,
           "robot_type": None, "fps": round(float(fps), 3), "n_state_frames": int(len(ta)),
           "duration_s": round(float(ta[-1]) + step, 3) if len(ta) else 0.0, "cameras": cams, **extra}
    if state is not None:
        ctx["state_kind"] = state_layout(state.shape[1], rig, state_names)[0]
    write_depth(ep, ctx, dep, dtimes)
    return finish_episode(ep, ctx, sources, state=state, action=action, times=times, signals=signals)


def episode_name(s: str) -> str:
    # an accented letter keeps its base letter (Día -> Dia), so a name in Spanish or French stays readable; other
    # letters and symbols (CJK, emoji) have no ASCII form and are dropped like any other separator
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
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
        return sum(1 for pk in c.demux(st) if pk.size and pk.pts is not None and not pk.is_discard)


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


NON_COLOUR = {"depth", "disparity", "confidence", "conf", "mask", "infrared", "ir", "thermal"}
GENERIC_VIDEO_WORDS = {"images", "image", "video", "videos", "rgb", "color", "colour", "raw"}


DEPTH_WORDS = {"depth", "disparity"}


def colour_videos(rels: list[str]) -> tuple[list[str], list[str]]:
    """(kept, left out): a folder's depth, infrared and mask videos are not cameras of their own when a colour video
    is beside them (exo_cam-images-depth.mkv beside exo_cam-images-rgb.mp4): a depth video goes with its colour camera
    (depth_videos), and the rest are left out. read.js colourVideos, the same rule."""
    dirs = lambda r: r.rsplit("/", 1)[0] if "/" in r else ""
    non = lambda r: bool(set(tokens(r.rsplit("/", 1)[-1].rsplit(".", 1)[0])) & NON_COLOUR)
    colour_dirs = {dirs(r) for r in rels if not non(r)}
    out = [r for r in rels if non(r) and dirs(r) in colour_dirs]
    return [r for r in rels if r not in out], out


def camera_words(rel: str) -> frozenset:
    """The words of a video's name that name its camera, not what the file holds (exo_cam-images-depth gives exo, cam).
    Only a video extension is taken off the name: a LeRobot feature key's dots are part of it
    (observation.images.cam_high gives cam, high)."""
    name = Path(rel)
    words = tokens(name.stem if name.suffix.lower() in VIDEO_EXT else name.name)
    return frozenset(set(words) - NON_COLOUR - GENERIC_VIDEO_WORDS)


def depth_videos(rels: list[str]) -> dict[str, str]:
    """{colour video: its depth video} for each depth video in a folder whose camera words (camera_words) match exactly
    one colour video beside it (exo_cam-images-depth.mkv goes with exo_cam-images-rgb.mp4). read.js depthVideos, the
    same rule."""
    dirs = lambda r: r.rsplit("/", 1)[0] if "/" in r else ""
    kind = lambda r: set(tokens(r.rsplit("/", 1)[-1].rsplit(".", 1)[0]))
    colour = [r for r in rels if not kind(r) & NON_COLOUR]
    out = {}
    for d in (r for r in rels if kind(r) & DEPTH_WORDS):
        match = [c for c in colour if dirs(c) == dirs(d) and camera_words(c) == camera_words(d)]
        if len(match) == 1 and match[0] not in out:
            out[match[0]] = d
    return out


def frame_times(video: Path, n: int, depth: bool = False) -> np.ndarray | None:
    """A video's frames' capture times in seconds on its recorder's clock, from the per-frame timestamp array a
    recorder writes beside it (exo_cam-rgb-timestamp.npy beside exo_cam-images-rgb.mp4): a .npy in its folder whose
    name says time and has the video's camera words, holding n rising numbers (seconds, or ms, us or ns, read
    from their size). Colour over depth when both fit, and depth over colour for a depth video (depth=True, whose
    stamps exo_cam-depth-timestamp.npy are its own). None when there is none."""
    cam = set(tokens(video.stem)) - GENERIC_VIDEO_WORDS - (NON_COLOUR if depth else set())
    cands = []
    for p in sorted(video.parent.glob("*.npy")):
        tk = set(tokens(p.stem))
        if not any(t.startswith(("time", "stamp")) for t in tk) or not cam <= tk:
            continue
        try:
            a = np.load(p, allow_pickle=False)
        except Exception:
            continue
        if a.ndim != 1 or len(a) != n or not np.issubdtype(a.dtype, np.number):
            continue
        a = a.astype(np.float64)
        if n > 1 and not (np.all(np.diff(a) >= 0) and a[-1] > a[0]):
            continue                      # never backwards (a recorder can stamp two frames alike)
        cands.append((bool(tk & NON_COLOUR) != depth, -len(os.path.commonprefix([p.stem, video.stem])), p.name, a))
    if not cands:
        return None
    a = min(cands, key=lambda c: c[:3])[3]
    # the unit from the frame step (a camera runs at 1 to 1000 fps), so stamps counted from the recording's start
    # (0, 33.3, 66.7 ms) read as well as stamps since 1970; a single frame falls back to the stamp's own size
    step = float(np.median(np.diff(a))) if n > 1 else 0.0
    if step > 0:
        return a * (1e-9 if step > 1.5e6 else 1e-6 if step > 1.5e3 else 1e-3 if step > 1.5 else 1.0)
    return a * (1e-9 if a[0] > 1e17 else 1e-6 if a[0] > 1e14 else 1e-3 if a[0] > 1e11 else 1.0)


def table_signals(paths: list[Path], real_anchor, pr_anchor: dict, extra: dict) -> Signals:
    """The numbers of CSV tables beside an episode's videos as signals, one per table under its own name (the file's
    name without its take: "traj"), its numeric columns as the named values. A table is placed on the anchor camera's
    frames by its time column (a rising column named for time) on the recorder's clock when the videos carry capture
    times, row by row when it has exactly one row per frame, and otherwise by its time column from the videos' start,
    which the source notes say; a table that cannot be placed is left out with the reason."""
    import pandas as pd
    out = Signals()
    n = int(len(pr_anchor["pts"]))
    t_vid = pr_anchor["pts"].astype(np.float64) * float(pr_anchor["time_base"])
    t_vid = t_vid - t_vid[0]
    for p in paths:
        try:
            df = pd.read_csv(p, sep="\t" if p.suffix.lower() == ".tsv" else ",")
        except Exception:
            out.left_out.append((p.name, "could not be read as a table"))
            continue
        num = df.select_dtypes("number")
        if num.shape[1] == 0 or len(num) < 2:
            continue                      # text only: the uploader's notes, read by annotation_tables
        clocks = [c for c in num.columns if is_named_clock(c, num[c].to_numpy())]
        tcol = clocks[0] if clocks else None
        skip = [c for c in num.columns if c in clocks or SIGNAL_SKIP.search(str(c))]
        counters = [c for c in num.columns if c not in skip and is_counter(num[c].to_numpy())]
        out.left_out += [(f"{c} in {p.name}", COUNTER_NOTE) for c in counters]
        vals = num.drop(columns=skip + counters)
        if vals.shape[1] == 0:
            continue
        v = vals.to_numpy(dtype=np.float64)
        parts = name_parts(p.stem)
        name = parts["cam"] or p.stem
        if tcol is not None:
            t = _seconds(num[tcol].to_numpy(dtype=np.float64))
        if tcol is not None and real_anchor is not None and t[0] < real_anchor[-1] and t[-1] > real_anchor[0]:
            a, var, gaps = place_on_frames(t, v, np.asarray(real_anchor, dtype=np.float64))
        elif len(v) == n:
            a, gaps = v, 0
        elif tcol is not None:
            a, var, gaps = place_on_frames(t - t[0], v, t_vid)
            extra.setdefault("source", {})["series_note"] = (f"{p.name} was placed on the video from both starts, "
                                                             "since the videos carry no capture times on its clock")
        else:
            out.left_out.append((p.name, f"{len(v)} rows and no time column, while the video has {n} frames"))
            continue
        out.add(name, a, names=[str(c) for c in vals.columns], source=f"table {p.name}")
        if gaps:
            out.meta[name]["gaps"] = gaps
    return out


def plan_video(det: dict, root: Path, grouping: dict | None = None) -> list[dict]:
    """One item per episode, grouped as group_videos says. Files are never split. Fixed-length packaging (a
    recorder that cuts continuous footage into files of one length) is found here and recorded on every item it
    applies to. MCAP files of recorded state (det["state"]) go with the episode of their folder, when the folder
    holds one episode."""
    root = Path(root)
    all_rels = [Path(f).relative_to(root).as_posix() for f in det["files"]]
    rels, left_out = colour_videos(all_rels)
    pairs = depth_videos(all_rels)
    with_colour = set(pairs.values())
    if with_colour:
        det.setdefault("used", []).append(
            f"{len(with_colour)} depth video{'s were' if len(with_colour) != 1 else ' was'} read with the colour "
            f"camera {'they belong' if len(with_colour) != 1 else 'it belongs'} to.")
    left_out = [r for r in left_out if r not in with_colour]
    if left_out:
        det.setdefault("used", []).append(
            f"{len(left_out)} infrared, mask or unmatched depth video{'s were' if len(left_out) != 1 else ' was'} left "
            "out, since the labeller reads the colour video of each camera.")
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
                      "cams": {c: root / r for c, r in e["cams"]},
                      "depth": {str(root / r): root / pairs[r] for _, r in e["cams"] if r in pairs}})
    for it in items:
        try:
            it["seconds"] = max((durations.get(Path(f).relative_to(root).as_posix()) or _duration(f)) for f in it["files"])
        except Exception:
            it["seconds"] = None          # unreadable header: measured on conversion, or reported as unreadable
    # a table of numbers beside an episode's videos (FreeTacMan's Hold_10_traj.csv beside Hold_10_camera1.mp4): the CSVs
    # in a folder of one episode, or the CSVs whose name gives the same take as the episode's videos (name_parts)
    csvs = {}
    for it in items:
        d = Path(it["files"][0]).parent
        csvs.setdefault(d, sorted(p for p in d.glob("*") if p.suffix.lower() in (".csv", ".tsv")
                                  and p.stat().st_size <= TABLE_MAX_BYTES))
    eps_in = {}
    for it in items:
        eps_in[Path(it["files"][0]).parent] = eps_in.get(Path(it["files"][0]).parent, 0) + 1
    for it in items:
        d = Path(it["files"][0]).parent
        take = name_parts(Path(it["files"][0]).stem)["take"]
        it["series"] = [p for p in csvs.get(d, []) if eps_in[d] == 1 or (take and name_parts(p.stem)["take"] == take)]
    folder = lambda it: Path(it["files"][0]).parent
    per_folder: dict[Path, int] = {}
    for it in items:
        per_folder[folder(it)] = per_folder.get(folder(it), 0) + 1
    for it in items:
        it["state"] = [Path(p) for p in det.get("state") or [] if Path(p).parent == folder(it) and per_folder[folder(it)] == 1]
    if det.get("state"):
        n = sum(1 for p in det["state"] if any(Path(p) in it["state"] for it in items))
        det["used"].append(f"{n} MCAP file{'s' if n != 1 else ''} of recorded state and sensor data, read with the "
                           f"videos beside {'them' if n != 1 else 'it'}." if n else
                           "The MCAP files hold no camera, and no folder holds them with the videos of one episode, so "
                           "their recorded data was not read.")
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
    mod = next((m for m in upload_adapters("video") if m.recognizes(item)), None)
    if mod is not None:
        # a folder in a dataset's own layout (OpenAoE's clip with its action segments) goes through its adapter
        ctx = mod.convert_upload(item, rig, out, dataset)
        ctx.setdefault("source", {})["adapter"] = mod.__name__.rsplit(".", 1)[-1]
        return ctx
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
    if item["dir"] is not None:
        extra["source"]["unused_cameras"] = unused
    instr = instruction_from(ann)
    if not instr and item["dir"] is not None:
        # a recorder's own metadata file in the episode folder (session_meta.json) that names the task
        instr = next((x for x in (instruction_from(_read_json(p)) for p in sorted(item["dir"].glob("*.json"))
                                  if p.stat().st_size <= 1_000_000) if x), None)
    if instr:
        extra["instruction"] = instr
        extra["instruction_note"] = "This instruction is the task text the uploader sent with the episode."
    if ann is not None and not (isinstance(ann, str) and ann.strip() == instr):
        set_uploader_notes(extra, ann)
    if item.get("fixed_window_s"):
        extra["collection_note"] = packaging_note(item["fixed_window_s"])
        extra["packaging"] = {"fixed_window_s": item["fixed_window_s"]}
    prs = {v: probe(p) for v, (_, p) in files.items()}
    real = {v: frame_times(Path(p), len(prs[v]["pts"])) for v, (_, p) in files.items()}
    state = action = state_names = None
    descs, signals = {}, {}
    if item.get("state"):
        from label import episode as me
        anchor = me.order_views(files)[0]
        mcap_files = [p for p in item["state"] if Path(p).suffix.lower() == ".mcap"]
        h5_files = [Path(p) for p in item["state"] if Path(p).suffix.lower() in H5_EXT]
        if real[anchor] is None:
            extra["state_note"] = ("Labelled from the cameras, because the videos carry no capture times to place the "
                                   "recorded " + ("arm state" if rig == "teleop_arms" else "sensor data") + " against.")
        elif rig != "teleop_arms" or not mcap_files:
            # a glove's pressure and hand pose beside a head camera, a handheld gripper's IMU: every number the MCAP
            # files record, on the videos' clock (mcap_signals); an MCAP arm channel is read as the state on an arm rig
            # only, and an HDF5 array named as the state below (h5_state)
            signals = mcap_signals(mcap_files, real[anchor]) if mcap_files else Signals()
            extra["source"]["sensors"] = [Path(p).name for p in item["state"]]
        else:
            streams = mcap_joint_streams(mcap_files)
            state, action, note = joint_state(streams, real[anchor])
            signals = mcap_signals(mcap_files, real[anchor], state_fields(streams, state, action))
            if note:
                extra["state_note"] = note
            elif state is not None:
                extra["source"]["state"] = [Path(p).name for p in mcap_files]
                third = third_arms(streams)
                if third:
                    extra["state_note"] = (f"The recording has a third arm ({', '.join(third)}) beside the left and "
                                           "right arms; it is read as neither working arm.")
                    # a scene camera whose name says it is on an arm, beside a third arm, is carried by that arm
                    if "exo" in files and is_mount_named(files["exo"][0]):
                        descs["exo"] = third_arm_camera_desc(third)
        if real[anchor] is not None and h5_files:
            more = h5_file_signals(h5_files, real[anchor], len(real[anchor]))
            if state is None and rig != "ego_head":
                # an HDF5 array named as the state (a robot.h5's qpos beside the videos) is read by the rule an HDF5
                # episode's is (h5_state), and leaves the signals when it is read
                state, action, state_names, state_src, h5_note = h5_state(more, rig, real[anchor])
                if state is not None:
                    extra["source"]["state"] = state_src
                    extra.pop("state_note", None)        # a note on the MCAP arm channels, which are not the state
                elif h5_note:
                    extra.setdefault("state_note", h5_note)
            if not hasattr(signals, "meta"):
                signals = Signals(signals)
            for k, v in more.items():
                signals[k] = v
                signals.meta[k] = more.meta[k]
            signals.left_out += more.left_out
    if item.get("series"):
        from label import episode as me
        anchor = me.order_views(files)[0]
        more = table_signals(item["series"], real.get(anchor), prs[anchor], extra)
        if not hasattr(signals, "meta"):
            signals = Signals(signals)
        for k, v in more.items():
            signals[k] = v
            signals.meta[k] = more.meta[k]
        signals.left_out += more.left_out
    # each camera's depth video, when the folder has one beside its colour video (depth_videos), on the same clock. A
    # depth video stored as an ordinary 8-bit picture (yuv, rgb) holds the recorder's shading, not distances: it is
    # shown as a camera of its own and never decoded as depth
    depth = {}
    for v, (_, p) in list(files.items()):
        dp = (item.get("depth") or {}).get(str(p))
        if dp is None:
            continue
        try:
            pr_d = probe_depth(dp)
            n_d = len(pr_d["pts"])
        except Exception:
            pr_d, n_d = None, 0
        if pr_d is not None and not str(pr_d["pix_fmt"] or "").startswith("gray"):
            k = 1 + sum(1 for x in files if x.startswith("extra"))
            files[f"extra{k}"] = (Path(dp).stem, Path(dp))
            descs[f"extra{k}"] = (f"the depth of {files[v][0]}, stored by the recorder as an ordinary picture, with "
                                  "its own shading of near and far")
            prs[f"extra{k}"] = probe(Path(dp))
            real[f"extra{k}"] = frame_times(Path(dp), len(prs[f"extra{k}"]["pts"]), depth=True)
            continue
        depth[v] = {"path": dp, "real": frame_times(Path(dp), n_d, depth=True) if n_d else None,
                    "scale_m": depth_scale_from(Path(dp).parent), "source": Path(dp).name}
    ep = unique_dir(out, episode_name(item["name"]))
    return video_views_episode(ep, files, rig, dataset, extra, prs=prs, real=real, state=state, action=action,
                               descs=descs, signals=signals, depth=depth, state_names=state_names)


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
    feats = {k: dict(f) for k, f in feats.items()}
    # an array's shape stated elsewhere in info.json (OpenTouch's export: "tactile": {"key": ..., "shape": [16, 16]})
    def shapes(x):
        if isinstance(x, dict):
            if isinstance(x.get("key"), str) and isinstance(x.get("shape"), list):
                feats.setdefault(x["key"], {}).setdefault("shape", x["shape"])
            for y in x.values():
                shapes(y)
    shapes({k: v for k, v in (info or {}).items() if k != "features"})
    def side(f):
        """The shorter side of an image feature, its channel axis (1, 3 or 4, first or last) left out; 0 when
        unknown."""
        sh = [int(x) for x in (f.get("shape") or [])]
        if len(sh) == 3:
            sh = sh[1:] if sh[0] in (1, 3, 4) else sh[:2] if sh[2] in (1, 3, 4) else sh[:2]
        return min(sh) if len(sh) == 2 else 0
    # an image feature smaller than CAMERA_MIN_PX on a side is a reading, not a camera (the Inspire hand's per-pad
    # pressure images of 3 x 3 to 14 x 8 cells): it is kept as a signal (recorded_signals)
    image_signals = [k for k, f in feats.items() if f.get("dtype") == "image" and 0 < side(f) < CAMERA_MIN_PX]
    is_depth = lambda k, f: bool((f.get("info") or {}).get("video.is_depth_map")) or bool(DEPTH_TOPIC.search(k))
    depth_cams = [k for k, f in feats.items() if f.get("dtype") == "video" and is_depth(k, f)]
    cams = [k for k, f in feats.items() if f.get("dtype") == "video" and k not in depth_cams]
    image_cams = [k for k, f in feats.items() if f.get("dtype") == "image" and k not in image_signals]
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
            "image_signals": image_signals, "depth_cams": depth_cams,
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
        for key in root["cams"] + root["depth_cams"]:
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
            for key in root["cams"] + root["depth_cams"]:
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

def _read_episode_table(path: Path, eidx: int, columns: list[str] | None, exclude=()):
    """The episode's rows of a LeRobot data file, in frame order: the named columns, or every column but exclude."""
    import pandas as pd
    import pyarrow.parquet as pq
    have = pq.ParquetFile(path).schema_arrow.names
    cols = [c for c in have if c not in exclude] if columns is None else [c for c in columns if c in have]
    df = pd.read_parquet(path, columns=cols or None)
    if "episode_index" in df.columns:
        df = df[df["episode_index"] == eidx]
    if "frame_index" in df.columns:
        df = df.sort_values("frame_index")
    return df


def _cells(col) -> np.ndarray | None:
    """A column as (rows, values), each cell flattened; a cell that is a list of lists (a 16 x 16 pressure map, which
    parquet gives as an array of arrays) is flattened in its own order. None when the cells are not numbers of one
    size."""
    try:
        a = np.stack([np.asarray(_nested(x), dtype=np.float64).reshape(-1) for x in col.to_numpy()])
        return a if a.ndim == 2 else None
    except Exception:
        return None


def _stack(col) -> np.ndarray | None:
    """A column as (rows, values) (_cells) when every value is a finite number, else None."""
    a = _cells(col)
    return a if a is not None and np.isfinite(a).all() else None


def convert_lerobot_item(item: dict, rig: str, out: Path, dataset: str) -> dict:
    """A LeRobot episode through the adapter that recognizes its dataset (prepare/*.py with UPLOAD = "lerobot", found
    by upload_adapters), else the generic reading below. Returns its context."""
    mod = next((m for m in upload_adapters("lerobot") if m.recognizes(item["root"])), None)
    if mod is None:
        return convert_lerobot(item, rig, out, dataset)
    ctx = mod.convert_upload(item, rig, out, dataset)
    ctx["dataset"] = dataset
    ctx.setdefault("source", {})["adapter"] = mod.__name__.rsplit(".", 1)[-1]
    if ctx.get("profile") != rig:
        ctx["source"]["rig_note"] = f"the upload was marked {rig}; this dataset's layout is {ctx.get('profile')}"
    (out / ctx["episode_id"] / "context.json").write_text(json.dumps(ctx, indent=1, default=str))
    return ctx


def convert_lerobot(item: dict, rig: str, out: Path, dataset: str, hold_back=()) -> dict:
    """hold_back: columns an adapter keeps out of the prompt (a publisher's own labels, kept to score against)."""
    r, row = item["root"], item["row"]
    eidx = row["eidx"]
    feats = r["features"]
    ep = out / episode_name(item["name"])
    notes = []
    df = None
    if row.get("data") is not None:
        try:
            need_images = [k for k in r["image_cams"]] if not any(k in row["videos"] for k in r["cams"]) else []
            # every column, so the recording's other signals are kept too (recorded_signals); images only when the
            # cameras are stored in the data file
            df = _read_episode_table(row["data"], eidx, None,
                                     exclude=[k for k in r["image_cams"] if k not in need_images])
            if not len(df):
                df = None
        except Exception:
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
    # a camera the metadata lists whose video is not on disk (an adapter downloads only the cameras it uses) is unused too
    unused = unused + [k for k in r["cams"] if k not in video_cams]
    if r["image_cams"]:
        unused = unused + [f"{k} (images in the data file)" for k in r["image_cams"]]
    extra["source"]["unused_cameras"] = unused
    kind, note = state_layout(state.shape[1] if state is not None else 0, rig, state_value_names(feats, state))
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
            # the size as the file is shown (prepare/display.py, as probe gives it): info.json's video.width and
            # video.height are the stored frame's, which a display rotation turns (a wrist camera mounted on its side)
            vi = (feats.get(key) or {}).get("info") or {}
            w, h = _shown_size(mp4)
            codec = vi.get("video.codec")
            if not w or not codec:
                pr = _stream_facts(mp4)
                w, h, codec = w or pr["width"], h or pr["height"], codec or pr["codec"]
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
        write_depth(ep, ctx, *lerobot_depth(ep, r, row, vmap, fps, ctx["n_state_frames"], unused))
        return finish_episode(ep, ctx, sources, state if kind != "none" else None, action,
                              signals=recorded_signals(df, _used_columns(kind) | set(hold_back), ctx["n_state_frames"],
                                                       feats))
    # one file per camera per episode (v2). LeRobot's timestamps are frame_index / fps and state rows follow
    # frames, so frames on the exact k/fps grid need no times; frames off it are decoded by their own pts and
    # timed by frame index, as the dataset defines them
    prs = {v: probe(row["videos"][key]) for v, key in vmap.items()}
    from label import episode as me
    anchor = me.order_views(prs)[0]
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
    write_depth(ep, ctx, *lerobot_depth(ep, r, row, vmap, fps, ctx["n_state_frames"], unused))
    return finish_episode(ep, ctx, sources, state if kind != "none" else None, action, times=times,
                          signals=recorded_signals(df, _used_columns(kind) | set(hold_back), ctx["n_state_frames"],
                                                   feats))


def lerobot_depth(ep: Path, r: dict, row: dict, vmap: dict, fps: float, n: int, unused: list) -> tuple[dict, dict]:
    """(depth.json entries, depth times) of a LeRobot episode's depth videos (features marked video.is_depth_map or
    named depth), each with its camera as the HDF5 reader pairs them (depth_camera). A second depth stream for a camera
    is added to unused (the episode's unused cameras), as convert_hdf5 lists it. Frames are timed as LeRobot defines
    them, frame index over fps, from the episode's own window of a packed file."""
    dep, tz = {}, {}
    if not r.get("depth_cams") or not vmap:
        return dep, tz
    from label import episode as me
    anchor = me.order_views(vmap)[0]
    ta = np.arange(n) / float(fps or 30.0)
    for key in r["depth_cams"]:
        src = row["videos"].get(key)
        if src is None:
            continue
        path, base, to = (src if isinstance(src, tuple) else (src, None, None))
        v, source = depth_camera(key, vmap, anchor)
        if v in dep:
            unused.append(f"{key} (depth with no camera of its own)")
            continue
        try:
            pr = probe_depth(Path(path))
        except Exception:
            continue
        if not str(pr["pix_fmt"] or "").startswith("gray"):
            continue                          # a depth feature stored in colour is a rendered picture, not distances
        t_all = pr["pts"].astype(np.float64) * float(pr["time_base"])
        sel = (t_all >= base - 1e-6) & (t_all < to - 1e-6) if base is not None else np.ones(len(t_all), dtype=bool)
        if not sel.any():
            continue
        td = t_all[sel] - (base if base is not None else t_all[sel][0])
        info = (r["features"].get(key) or {}).get("info") or {}
        scale = next((float(x) for k, x in info.items() if DEPTH_SCALE_KEY.search(str(k).split(".")[-1])
                      and isinstance(x, (int, float)) and 0 < x < 10), None)
        e, t = depth_entry(ep, v, Path(path), td, ta, pr["pts"][sel], scale, source)
        e.update(width=pr["width"], height=pr["height"], pix_fmt=pr["pix_fmt"])
        dep[v] = e
        tz.update(t)
    return dep, tz


def _used_columns(kind: str) -> set:
    """The columns already read as the state and action. A state that does not fit the arm layout (kind "none") is not
    read as one, so it stays a signal the model is shown."""
    return {"observation.state", "action"} if kind != "none" else set()


def _stream_facts(p: Path) -> dict:
    with open_video(p) as c:
        st = c.streams.video[0]
        return {"width": st.codec_context.width, "height": st.codec_context.height, "codec": st.codec_context.name}


def _shown_size(p: Path) -> tuple[int, int]:
    """(width, height) of a video as it is shown (prepare/display.py; read once per file, so a packed file shared by
    many episodes is probed once); (0, 0) when ffprobe cannot read it."""
    from prepare import display
    g = display.geometry(str(p))
    return display.shown_size(g) if g["stored"][0] else (0, 0)


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
    kind, note = state_layout(state.shape[1] if state is not None else 0, rig, state_value_names(r["features"], state))
    # the table's other numbers go with the frames as they do beside videos (recorded_signals)
    signals = recorded_signals(df, _used_columns(kind) | set(r["image_cams"]), 0, r["features"])
    ctx = video_views_episode(ep, files, rig, dataset, extra, signals=signals)
    if state is not None and kind != "none" and len(state) == ctx["n_state_frames"]:
        ctx["state_kind"] = kind
        return finish_episode(ep, ctx, json.loads((ep / "sources.json").read_text()), state, action,
                              times={k: v for k, v in np.load(ep / "times.npz").items()}, signals=signals)
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


# ---------------------------------------------------------------- HDF5

# HDF5 has no schema for robot data, so it is read by what each array is, never by a list of dataset names:
#   - a camera: uint8 frames (N, H, W, 3) or (N, H, W) at least CAMERA_MIN_PX on a side, or N encoded images (JPEG, PNG
#     bytes); its name decides its slot as any camera's does (assign_views)
#   - depth: (N, H, W) of 16-bit integers or floats, too large to be a signal (more than SIGNAL_MAX_VALUES pixels), or
#     any such array named depth; it goes with the camera whose name it shares, else the scene camera
#   - a clock: a rising 1-D (or N x 1) array named for time (time_s, timestamps, stamp, ts), read in seconds, ms, us or
#     ns by its step; it times every array of its length in its group, then in the groups above it
#   - a signal: every other numeric array with one row per sample (a 16 x 16 pressure map, 21 x 3 hand landmarks, a
#     force vector), resampled onto the anchor camera's frames as an MCAP channel is (mcap_signals' rules)
#   - text: strings, one-row records and attributes become the uploader's notes, and one named for the task
#     (instruction, task, prompt, language) the instruction
# Episodes: sibling groups that share one layout (OpenTouch's data/demo_287, robomimic's data/demo_0) are one episode
# each; otherwise the file is one episode (ALOHA's episode_0.hdf5). A file with no camera beside videos holds the
# videos' sensor data (a glove's pressure recorded beside a head camera), as an MCAP with no camera does.
H5_EXT = {".h5", ".hdf5", ".hdf"}
CAMERA_MIN_PX = 64
H5_CONSTANT_MAX = 64              # a numeric array this small with no clock of its length is a setting, kept as a note
TASK_KEY = re.compile(r"(^|_)(instruction|task|task_description|language_instruction|language|prompt|goal)$", re.I)


def _is_image_bytes(b) -> bool:
    b = bytes(b[:8]) if b is not None else b""
    return b.startswith(b"\xff\xd8") or b.startswith(b"\x89PNG")


def _h5_bytes(x) -> bytes | None:
    if isinstance(x, (bytes, bytearray, memoryview)):
        return bytes(x)
    if isinstance(x, np.ndarray) and x.dtype == np.uint8 and x.ndim == 1:
        return x.tobytes()
    if isinstance(x, np.void):
        return bytes(x)
    return None


EPOCH_S = (1.0e9, 4.1e9)       # 2001 to 2099: a recorder clock counting from the epoch lands here in its own unit
# A stream read as an epoch clock steps at 1 Hz to 100 kHz in that unit. Epoch clocks slower than 1 Hz are rare, and a
# top of 10 s would read a 200 Hz clock in nanoseconds from 23 days of uptime as microseconds from the epoch.
EPOCH_STEP_S = (1e-5, 1.0)


def _clock_facts(a: np.ndarray) -> tuple[np.ndarray, float, float]:
    """(finite values, median step, size) of a clock. Its size is the median of its finite values, so a 0 a recorder
    writes before its first stamp does not decide it."""
    ok = np.asarray(a, dtype=np.float64).ravel()
    ok = ok[np.isfinite(ok)]
    step = float(np.median(np.diff(ok))) if len(ok) > 1 else 0.0
    size = abs(float(np.median(ok))) if len(ok) else 0.0
    return ok, step, size


def _epoch_scale(a: np.ndarray) -> float | None:
    """The unit a clock's size settles: a clock that counts from the epoch says its unit by its size (only nanoseconds
    put 1.79e18 in this century), provided its step in that unit falls between 10 us and 1 s (EPOCH_STEP_S). A clock
    from boot can also be that size in another unit. OpenTouch's 2.6e12 ns after 43 minutes up is 2.6e9 read as
    milliseconds, which would step 9 hours at 30 fps, and 2e15 ns after 23 days up is 2e9 read as microseconds, which
    would step 5 s at 200 Hz. None when the size settles nothing."""
    ok, step, size = _clock_facts(a)
    for scale in (1e-9, 1e-6, 1e-3, 1.0):
        on_epoch = EPOCH_S[0] <= size * scale <= EPOCH_S[1]
        steps_like_a_stream = len(ok) == 1 or EPOCH_STEP_S[0] <= step * scale <= EPOCH_STEP_S[1]
        if on_epoch and steps_like_a_stream:
            return scale
    return None


def _seconds_scale(a: np.ndarray) -> float:
    """Seconds per unit of a clock: the unit its size settles (_epoch_scale), else the unit its step says (one sample
    every 1 ms to 10 s), as frame_times reads a recorder's stamps, and for a clock that never steps the unit its size
    says alone."""
    scale = _epoch_scale(a)
    if scale is not None:
        return scale
    _, step, size = _clock_facts(a)
    if step > 0:
        return 1e-9 if step > 1.5e6 else 1e-6 if step > 1.5e3 else 1e-3 if step > 1.5 else 1.0
    return 1e-9 if size > 1e17 else 1e-6 if size > 1e14 else 1e-3 if size > 1e11 else 1.0


def _seconds(a: np.ndarray) -> np.ndarray:
    """A clock in seconds, in the unit _seconds_scale reads from it."""
    a = np.asarray(a, dtype=np.float64).ravel()
    return a * _seconds_scale(a)


CLOCK_SCALES = (1e-9, 1e-6, 1e-3, 1.0)
FAR_FROM_ZERO_STEPS = 1000     # a clock counts from far from zero when its median is more than this many of its steps
RANGE_PERCENTILES = (1, 99)    # a clock's time range, so a stray stamp (a 0 before the first) does not stretch it
SPAN_MATCH = 2.0               # a clock from near zero takes the unit that puts its span within this factor


def _clocks_in_seconds(raw: dict[str, np.ndarray], reference: str | None = None) -> dict[str, np.ndarray]:
    """{path: seconds} for the clocks of one episode. Each driver stamps in its own unit: a 1 kHz pad beside a 30 fps
    camera steps by 1e6 in nanoseconds or 1e3 in microseconds, which its step alone reads as microseconds or
    milliseconds. So each clock is read against a reference clock, which is read with its own _seconds_scale.
    - A clock that counts from far from zero (its median more than FAR_FROM_ZERO_STEPS of its own steps, a clock from
      boot or the epoch), beside a reference that does too, takes the unit among CLOCK_SCALES under which its time
      range (its RANGE_PERCENTILES) overlaps the reference's. The units are 1000 apart, so at most one does, whether
      the clock's log runs longer than the camera's clip or covers only part of it. The median and the percentiles
      keep a 0 a recorder writes before its first stamp from putting the clock near zero or stretching its range.
    - A clock or a reference that starts near zero says nothing about its unit by its start, so a sampled stream (at
      least COUNTER_MIN_MESSAGES finite values) takes the unit that puts its span within SPAN_MATCH of the
      reference's. A few event stamps are not a stream and need not span the episode, so they keep their own reading.
    - Otherwise, and when it has fewer than two finite values, a clock keeps its own _seconds_scale.
    The reference is the camera's clock (reference). Without one, the clock with the largest step is a guess, so it is
    used only when its size settles its unit (_epoch_scale), and otherwise every clock keeps its own reading. A
    reference that does not step forward (one value, or stuck at one) says nothing about units, so then too every
    clock keeps its own reading."""
    facts = {p: _clock_facts(a) for p, a in raw.items()}
    if reference not in raw:
        guess = max(raw, key=lambda p: facts[p][1], default=None)
        reference = guess if guess is not None and _epoch_scale(raw[guess]) is not None else None
    if reference is None or facts[reference][1] <= 0:
        return {p: a * _seconds_scale(a) for p, a in raw.items()}

    def far_from_zero(p):
        _, step, size = facts[p]
        return size > FAR_FROM_ZERO_STEPS * abs(step)

    def bounds(p):
        lo, hi = np.percentile(facts[p][0], RANGE_PERCENTILES)
        return float(lo), float(hi)
    ref_scale = _seconds_scale(raw[reference])
    ref_lo, ref_hi = (x * ref_scale for x in bounds(reference))
    out = {}
    for p, a in raw.items():
        scale = _seconds_scale(a)
        if p != reference and len(facts[p][0]) > 1:
            lo, hi = bounds(p)
            if far_from_zero(p) and far_from_zero(reference):
                overlapping = [s for s in CLOCK_SCALES if lo * s <= ref_hi and hi * s >= ref_lo]
                if overlapping:
                    scale = overlapping[0]
            elif len(facts[p][0]) >= COUNTER_MIN_MESSAGES and hi > lo and ref_hi > ref_lo:
                off = {s: abs(np.log((hi - lo) * s / (ref_hi - ref_lo))) for s in CLOCK_SCALES}
                closest = min(off, key=off.get)
                if off[closest] <= np.log(SPAN_MATCH):
                    scale = closest
        out[p] = a * scale
    return out


def h5_kind(name: str, ds) -> str | None:
    """camera, depth, time, signal, text or None (empty, or nothing we read) for one HDF5 dataset."""
    shape, dt = ds.shape, ds.dtype
    if not shape or shape[0] < 1:
        return "text" if dt.kind in "SUO" or dt.names else None
    n = shape[0]
    leaf = name.rsplit("/", 1)[-1]
    if dt.kind == "O" or dt.kind == "V" and not dt.names or (dt.kind == "u" and dt.itemsize == 1 and len(shape) == 1
                                                              and not is_time_name(leaf)):
        try:
            first = _h5_bytes(ds[0])
        except Exception:
            first = None
        if first is not None and _is_image_bytes(first):
            return "camera"
        if dt.kind == "O":
            try:
                return "text" if isinstance(ds[0], (str, bytes)) else None
            except Exception:
                return None
    if dt.kind in "SU" or dt.names:
        return "text"
    if dt.kind not in "biuf":
        return None
    per = shape[1:]
    if len(per) <= 1 and (not per or per[0] == 1) and n > 1 and is_time_name(leaf):
        a = np.asarray(ds[: min(n, 4096)], dtype=np.float64).ravel()
        if np.all(np.diff(a) >= 0) and a[-1] > a[0]:
            return "time"
    if len(per) == 3 and per[2] in (3, 4) and dt == np.uint8 and min(per[:2]) >= CAMERA_MIN_PX:
        return "camera"
    if len(per) == 2 and min(per) >= CAMERA_MIN_PX:
        wide_numeric_picture = int(np.prod(per)) > SIGNAL_MAX_VALUES and dt.kind in "uf" and dt.itemsize >= 2
        if "depth" in leaf.lower() or "depth" in name.lower() or wide_numeric_picture:
            return "depth"
        if dt == np.uint8:
            return "camera"
    if int(np.prod(per)) <= SIGNAL_MAX_VALUES:
        return "signal"
    return None


def _h5_datasets(g, base: str = "") -> list[tuple[str, object]]:
    import h5py
    out = []
    for k in g:
        try:
            o = g[k]
        except Exception:
            continue
        p = f"{base}/{k}" if base else k
        if isinstance(o, h5py.Dataset):
            out.append((p, o))
        elif isinstance(o, h5py.Group):
            out += _h5_datasets(o, p)
    return out


def h5_episodes(f) -> list[str]:
    """The group paths of the file's episodes: the first level, from the root down, holding two or more sibling groups
    with one layout (the same dataset paths under each); else [""], the whole file."""
    import h5py

    def layout(g):
        return frozenset(p for p, _ in _h5_datasets(g))
    queue = [("", f)]
    while queue:
        path, g = queue.pop(0)
        kids = [(k, g[k]) for k in g if isinstance(g[k], h5py.Group)]
        lays = {k: layout(kg) for k, kg in kids}
        lays = {k: lay for k, lay in lays.items() if lay}
        # siblings that share most of one layout are episodes: a demo that also records one more array (or one fewer)
        # is still a demo, never dropped for it
        similar = lambda a, b: len(a & b) * 2 >= max(len(a), len(b))
        best = max(([k2 for k2, l2 in lays.items() if similar(l1, l2)] for l1 in lays.values()), key=len, default=[])
        if len(best) >= 2:
            return [f"{path}/{k}" if path else k for k in sorted(best, key=_natural)]
        queue += [(f"{path}/{k}" if path else k, kg) for k, kg in kids]
    return [""]


def _natural(s: str):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", s)]


FRAME_RATE_NAME = re.compile(r"^(fps|frame_?rate|control_?freq|control_?frequency)$", re.I)
GENERIC_RATE_NAME = re.compile(r"^(frequency|freq|hz|rate)$", re.I)
FPS_RANGE = (1, 1000)


def h5_fps(f, group: str) -> float | None:
    """A frame rate the file states in the attributes of the episode's group or a group above it. In order, the first
    found wins: an attribute with a frame rate's name (fps, frame_rate, control_freq), nearest group first up to the
    root; then a frame rate's name inside an attribute that holds JSON (robomimic's env_args keeps
    env_kwargs.control_freq), nearest group first; then an attribute with a generic name (rate, hz, freq), nearest
    group first. An attribute named for the frame rate is the file's own statement about its frames, JSON holds a
    configuration, and a generic name only counts where the file says it plainly: a configuration's {"sensors": {"imu":
    {"rate": 200}}} is a sensor's rate, never the frames'. A rate counts only within FPS_RANGE, a bool is not one, and
    per-camera rates ({"fps": {"cam_high": 30, "cam_wrist": 30}}) count when they all agree."""
    def number(v):
        try:
            x = np.asarray(v).ravel()[0]
            if isinstance(x, (bool, np.bool_)):
                return None
            x = float(x)
        except Exception:
            return None
        return x if FPS_RANGE[0] <= x <= FPS_RANGE[1] else None

    def rate_value(v):
        if isinstance(v, dict):
            rates = {number(x) for x in v.values()}
            return rates.pop() if len(rates) == 1 and None not in rates else None
        return number(v)

    def rate_in(key, value, names):
        # the first value under one of names, in document order; a list's items go under the list's own name
        todo = [(key, value)]
        while todo:
            k, v = todo.pop()
            x = rate_value(v) if names.match(str(k)) else None
            if x is not None:
                return x
            if isinstance(v, dict):
                todo += reversed(list(v.items()))
            elif isinstance(v, list):
                todo += [(k, item) for item in reversed(v)]
        return None

    def attrs(g):
        # {"direct": each attribute as it is, "in JSON": each attribute that holds JSON (a string, bytes, or a
        # one-element string array), parsed}
        direct, in_json = [], []
        for k, v in g.attrs.items():
            text = v.ravel()[0] if isinstance(v, np.ndarray) and v.size == 1 and v.dtype.kind in "OSU" else v
            if isinstance(text, bytes):
                text = text.decode("utf-8", "replace")
            if isinstance(text, str) and text.lstrip()[:1] in ("{", "["):
                try:
                    in_json.append((k, json.loads(text)))
                except (ValueError, RecursionError):
                    pass                          # malformed, or nested deeper than the parser goes
            else:
                direct.append((k, v))
        return {"direct": direct, "in JSON": in_json}

    parts = [p for p in group.split("/") if p]
    nearest_first = [attrs(f["/".join(parts[:i])] if i else f) for i in range(len(parts), -1, -1)]
    precedence = (("direct", FRAME_RATE_NAME), ("in JSON", FRAME_RATE_NAME), ("direct", GENERIC_RATE_NAME))
    for kind, names in precedence:
        for group_attrs in nearest_first:
            rates = (rate_in(k, v, names) for k, v in group_attrs[kind])
            x = next((x for x in rates if x is not None), None)
            if x is not None:
                return x
    return None


def h5_streams(f, group: str) -> dict:
    """What one episode's group holds: {"camera": [...], "depth": [...], "signal": [...], "text": [...], "clock": {path:
    seconds}}, each stream {"path", "name" (its path inside the episode), "n", "clock" (the path of the clock that times
    it, or None)}."""
    g = f[group] if group else f
    items = _h5_datasets(g)
    kinds = {p: h5_kind(p, ds) for p, ds in items}
    # raw until the streams are paired with their clocks by length, then in seconds read against a camera's clock
    clocks = {p: np.asarray(ds[()], dtype=np.float64).ravel() for p, ds in items if kinds[p] == "time"}
    out = {"camera": [], "depth": [], "signal": [], "text": [], "clock": clocks, "unused": []}
    for p, ds in items:
        k = kinds[p]
        if k == "time":
            continue
        if k is None:
            if ds.shape and ds.shape[0] > 1 and ds.dtype.kind in "biuf":
                out["unused"].append(f"{p} ({' x '.join(map(str, ds.shape[1:]))} values per sample, more than the "
                                     f"{SIGNAL_MAX_VALUES} a signal holds)")
            continue
        n = int(ds.shape[0]) if ds.shape else 0
        if k == "signal" and not any(len(t) == n for t in clocks.values()) and ds.size <= H5_CONSTANT_MAX:
            k = "text"                    # a calibration or a setting (a 4 x 4 transform), not a sample per frame
        # the clock beside it with its length, in its own group first, then in each group above it
        clock = None
        parts = p.split("/")[:-1]
        parent = lambda c: c.rsplit("/", 1)[0] if "/" in c else ""
        for i in range(len(parts), -1, -1):
            pre = "/".join(parts[:i])
            cands = [c for c, t in clocks.items() if len(t) == n and parent(c) == pre]
            if cands:
                clock = sorted(cands)[0]
                break
        out[k].append({"path": f"{group}/{p}" if group else p, "name": p, "n": n, "clock": clock})
    camera_clock = next((c["clock"] for c in out["camera"] if c["clock"]), None)
    out["clock"] = _clocks_in_seconds(clocks, reference=camera_clock)
    # names inside the episode: a group every camera and signal sits under (OpenTouch's data/demo_023/ in a file of one
    # demo) says nothing about any one of them, so it is left off their names
    timed = out["camera"] + out["depth"] + out["signal"]
    if timed:
        common = os.path.commonprefix([x["name"].split("/")[:-1] for x in timed])
        if common:
            cut = len("/".join(common)) + 1
            for x in timed:
                x["name"] = x["name"][cut:]
    return out


def h5_text(f, group: str, texts: list[dict]) -> tuple[str | None, dict]:
    """(instruction, notes) from an episode's strings and records and the attributes of its group and the file."""
    notes, instr = {}, None

    def val(x):
        if isinstance(x, bytes):
            return x.decode("utf-8", "replace")
        if isinstance(x, np.ndarray):
            if x.dtype.names:
                return [{n: val(r[n]) for n in x.dtype.names} for r in x.ravel()[:50]]
            return [val(y) for y in x.ravel()[:50]] if x.size > 1 else val(x.ravel()[0]) if x.size else None
        if isinstance(x, np.void) and x.dtype.names:
            return {n: val(x[n]) for n in x.dtype.names}
        if isinstance(x, np.generic):
            return x.item()
        return x
    for s in texts:
        try:
            v = val(f[s["path"]][()])
        except Exception:
            continue
        notes[s["name"]] = v
        if instr is None and TASK_KEY.search(s["name"].rsplit("/", 1)[-1]) and isinstance(v, str) and 0 < len(v) < 400:
            instr = v.strip()
    if group:
        # what the file keeps beside its episodes (a camera's calibration, a setting): small arrays and strings outside
        # every episode group, as notes of each episode
        eps = set(h5_episodes(f))
        for p, ds in _h5_datasets(f):
            if any(p == e or p.startswith(e + "/") for e in eps) or ds.size > H5_CONSTANT_MAX:
                continue
            try:
                notes[f"file {p}"] = val(ds[()])
            except Exception:
                continue
    for where, g in (("episode", f[group] if group else None), ("file", f)):
        if g is None:
            continue
        for k, v in g.attrs.items():
            v = val(v)
            notes[f"{where} attribute {k}"] = v
            if instr is None and TASK_KEY.search(str(k)) and isinstance(v, str) and 0 < len(v) < 400:
                instr = v.strip()
    return instr, notes


def plan_hdf5(det: dict, root: Path) -> list[dict]:
    """One item per episode of every HDF5 file (h5_episodes), with its length from its clocks or its stated rate."""
    import h5py
    items = []
    for fpath in det["files"]:
        try:
            with h5py.File(fpath, "r") as f:
                groups = h5_episodes(f)
                for g in groups:
                    st = h5_streams(f, g)
                    cams = st["camera"]
                    secs = None
                    if cams:
                        c0 = cams[0]
                        if c0["clock"]:
                            t = st["clock"][c0["clock"]]
                            secs = float(t[-1] - t[0]) + (float(np.median(np.diff(t))) if len(t) > 1 else 0.0)
                        else:
                            secs = c0["n"] / (h5_fps(f, g) or 30.0)
                    rel = Path(fpath).relative_to(root).with_suffix("").as_posix()
                    items.append({"kind": "hdf5", "name": f"{rel}/{g.rsplit('/', 1)[-1]}" if g else rel,
                                  "file": Path(fpath), "group": g, "seconds": secs, "has_camera": bool(cams)})
        except Exception as e:
            items.append({"kind": "hdf5", "name": Path(fpath).relative_to(root).with_suffix("").as_posix(),
                          "file": Path(fpath), "group": "", "seconds": None, "has_camera": False,
                          "error": f"the HDF5 file could not be opened ({type(e).__name__})"})
    return items


NOT_FINITE = "values that are not finite numbers"    # why a signal is left out when it holds an inf (h5_state reads it)


def h5_signals(f, streams: dict, q_abs: np.ndarray, fps: float | None, n_anchor: int) -> Signals:
    """Every signal of an episode on the anchor camera's frames: by its own clock, as mcap_signals places a channel
    (nearest sample, NaN where none is near, left out when it does not cover the footage), or, with no clock, one row
    per anchor frame when it has as many rows as the anchor has frames."""
    out = Signals()
    q = np.asarray(q_abs, dtype=np.float64)
    span = float(q[-1] - q[0]) if len(q) > 1 else 0.0
    for s in streams["signal"]:
        ds = f[s["path"]]
        if ds.dtype.kind not in "biuf":
            continue
        a = np.asarray(ds[()], dtype=np.float64)
        shape = list(a.shape[1:])
        a = a.reshape(len(a), -1) if a.ndim > 1 else a[:, None]
        if not a.shape[1]:
            continue
        names = None
        for key in ("names", "columns", "labels", "fields"):
            if key in ds.attrs:
                stated = [x.decode() if isinstance(x, bytes) else str(x) for x in np.asarray(ds.attrs[key]).ravel()]
                names = value_names(stated, a.shape[1])
                break
        if s["clock"]:
            t = streams["clock"][s["clock"]]
            if span <= 0 or len(t) < 2:
                continue
            if t[0] > q[0] + STATE_EDGE_SLACK_S or t[-1] < q[-1] - STATE_EDGE_SLACK_S:
                out.left_out.append((s["name"], f"recorded from {t[0] - q[0]:.1f} s to {t[-1] - q[0]:.1f} s, not over "
                                                "the whole footage"))
                continue
            v, var, n_gaps = place_on_frames(t, a, q)
            far = np.zeros(len(q), dtype=bool)
            far[:n_gaps] = True               # only its count is kept (meta "gaps")
            rate = len(t) / max(float(t[-1] - t[0]), 1e-9)
            if var is not None and a.shape[1] <= VARIATION_MAX_VALUES and _variation_matters(s["name"], v, var):
                out.add(f"{s['name']} variation within each frame", var, shape=shape if len(shape) > 1 else None,
                        names=names, source=f"HDF5 dataset {s['path']}")
                out.meta[f"{s['name']} variation within each frame"]["variation_of"] = s["name"]
        elif len(a) == n_anchor:
            v, rate, far, var = a, fps, None, None
        else:
            out.left_out.append((s["name"], f"{len(a)} rows and no clock, while the camera has {n_anchor} frames"))
            continue
        if not np.isfinite(v[~np.isnan(v).all(axis=1)] if np.isnan(v).any() else v).all():
            out.left_out.append((s["name"], NOT_FINITE))
            continue
        out.add(s["name"], v, shape=shape if len(shape) > 1 else None, names=names, source=f"HDF5 dataset {s['path']}")
        if rate:
            out.meta[s["name"]]["rate_hz"] = round(float(rate), 2)
        if far is not None and far.any():
            out.meta[s["name"]]["gaps"] = int(far.sum())
    return out


# An HDF5 array is the episode's recorded state when its own name says so: ALOHA's observations/qpos, a recorder's
# state or robot_state, or joint_positions. It is then read by the same rule as a LeRobot observation.state
# (state_layout). An array named for joint positions names every value a joint, so DROID's seven Franka joints are not
# taken for six joints and a gripper; robomimic's obs/robot0_joint_pos is not named as the state and stays a signal.
# An array under a group named for the action (DROID's action/joint_position) is a command, never the state. Beside
# videos, the arrays of several HDF5 files carry their file's name first ("robot qpos", h5_file_signals).
H5_STATE_NAME = re.compile(r"(^|[/ ])(qpos|state|states|robot_state|joint_positions?|joint_pos)$", re.I)
H5_JOINT_ARRAY = re.compile(r"(^|[/ ])joint_pos(itions?)?$", re.I)
H5_ACTION_NAME = re.compile(r"(^|[/ ])actions?$", re.I)
H5_ACTION_GROUP = re.compile(r"(^|[/ ])actions?/", re.I)


def h5_state(signals: Signals, rig: str, q: np.ndarray) -> tuple:
    """(state, action, value names, the state array's name, note) of an HDF5 episode, from its signals (h5_signals,
    already on the anchor camera's frames, at times q). The arrays named as the state (H5_STATE_NAME, outside an action
    group) are tried shortest name first, and the first that state_layout lays out with the names the file gives its
    values is the state; the array named as the action goes with it when it has the state's shape. A frame with no
    reading (a clocked array that starts or ends within STATE_EDGE_SLACK_S of the footage, which h5_signals keeps) is
    filled as joint_state fills an MCAP arm's frames (fill_rows), so an HDF5 state is accepted wherever an MCAP one is,
    and a gap longer than STATE_EDGE_SLACK_S leaves it unread with the gap's time in the note.
    Both leave the signals when the state is read; otherwise they stay, and note gives the first array's reason, named.
    All None on a head camera, which has no state and no note about one, or when no array is named as the state."""
    if rig == "ego_head":
        return None, None, None, None, None
    meta = getattr(signals, "meta", {}) or {}
    left_out = dict(getattr(signals, "left_out", []) or [])
    named = lambda k: H5_STATE_NAME.search(k) and not H5_ACTION_GROUP.search(k)
    cands = sorted({k for k in [*signals, *left_out] if named(k)}, key=lambda k: (len(k), k))

    q = np.asarray(q, dtype=np.float64)

    def filled(a):
        # frames with no reading take the readings around them across no gap longer than the slack (fill_rows), else
        # (None, why): no frame has a reading, or a gap is longer
        ok = np.isfinite(a).all(axis=1)
        if not ok.any():
            return None, "has no reading on any frame"
        if ok.all():
            return a, None
        rows, gap = fill_rows(q, q[ok], a[ok])
        return rows, (gap_words(gap, q[0]) if gap else None)
    notes = []
    for name in cands:
        if name not in signals and left_out[name] == NOT_FINITE:
            notes.append(f"Labelled from the video, because the recorded state {name} has values that are not all "
                         "finite numbers.")
            continue
        if name not in signals:
            notes.append(f"Labelled from the video, because the recorded state {name} could not be placed on the "
                         f"camera's frames ({left_out[name]}).")
            continue
        a = np.asarray(signals[name], dtype=np.float64)
        names = (meta.get(name) or {}).get("names")
        dims = a.shape[1]
        if names is None and H5_JOINT_ARRAY.search(name) and dims in (7, 14):
            # the file names no value, but the array's name says every value is a joint, so there is no gripper
            notes.append(f"Labelled from the video: the array's name says every value is a joint, so its {dims} values "
                         f"are {dims} joints and no gripper, and our checks read "
                         + ("six joints and a gripper per arm." if rig == "teleop_arms" else
                            "a 6D pose and an opening per gripper.") + f" The recorded state is the HDF5 array {name}.")
            continue
        kind, note = state_layout(dims, rig, names)
        if kind == "none":
            notes.append(f"{note} The recorded state is the HDF5 array {name}." if note else None)
            continue
        a, why = filled(a)
        if a is None:
            notes.append(f"Labelled from the video: the recorded state {name} has no reading on any frame."
                         if why == "has no reading on any frame" else
                         f"Labelled from the video, because the recorded state {name} {why}.")
            continue
        act = next((k for k in signals if H5_ACTION_NAME.search(k) and np.shape(signals[k]) == a.shape
                    and filled(np.asarray(signals[k], dtype=np.float64))[0] is not None), None)
        action = filled(np.asarray(signals.pop(act), dtype=np.float64))[0] if act else None
        signals.pop(name)
        for k in (name, act):
            meta.pop(k, None)
        return a, action, names, name, None
    return None, None, None, None, (notes[0] if notes else None)


def convert_hdf5(item: dict, rig: str, out: Path, dataset: str) -> dict:
    """One HDF5 episode: its cameras written to H.264 at their frame times (encoded images as they are decoded, raw
    frames as they are), its depth to 16-bit FFV1, every signal on the anchor camera's frames, and its text as the
    instruction and the uploader's notes (h5_streams says what each array is). An array named as the state is read as
    the state by the same rule as LeRobot's (h5_state)."""
    import h5py
    from PIL import Image
    if item.get("error"):
        raise ValueError(item["error"])
    ep = unique_dir(out, episode_name(item["name"]))
    ep.mkdir(parents=True, exist_ok=True)
    with h5py.File(item["file"], "r") as f:
        g = item["group"]
        st = h5_streams(f, g)
        if not st["camera"]:
            raise ValueError("the HDF5 episode has no camera (no image frames or encoded images)")
        fps = h5_fps(f, g)
        names = [c["name"] for c in st["camera"]]
        vmap, unused = pick_cameras(names, rig, names + [d["name"] for d in st["depth"]])
        by_name = {c["name"]: c for c in st["camera"]}

        def times_of(s):
            if s["clock"]:
                return st["clock"][s["clock"]]
            return np.arange(s["n"]) / (fps or 30.0)
        chosen = {v: by_name[nm] for v, nm in vmap.items()}
        t0 = min(float(times_of(s)[0]) for s in chosen.values())
        files, real = {}, {}
        for v, s in chosen.items():
            ds = f[s["path"]]
            t = times_of(s)
            w = FrameWriter(ep / f"{v}.mp4", "")
            for i in range(s["n"]):
                x = ds[i]
                b = _h5_bytes(x) if (ds.dtype.kind in "OV" or ds.ndim == 1) else None
                if b is not None:
                    w.add(float(t[i] - t0), b)
                else:
                    a = np.asarray(x)
                    im = Image.fromarray(a if a.ndim == 3 else a.astype(np.uint8)).convert("RGB")
                    w.add_image(float(t[i] - t0), im)
            if not w.close():
                unused.append(f"{s['name']} (no frame could be decoded)")
                continue
            files[v] = (s["name"], ep / f"{v}.mp4")
        if not files:
            raise ValueError("no frame of the HDF5 episode's cameras could be decoded")
        from label import episode as me
        anchor = me.order_views(files)[0]
        q_abs = times_of(chosen[anchor])
        # depth: with the camera whose name has its words, else the anchor camera
        depth = {}
        for d in st["depth"]:
            v, source = depth_camera(d["name"], {v: nm for v, (nm, _) in files.items()}, anchor)
            if v in depth:
                unused.append(f"{d['name']} (depth with no camera of its own)")
                continue
            ds = f[d["path"]]
            t = times_of(d)
            scale = 1.0 if ds.dtype.kind == "f" else depth_scale_attr(ds)
            dw = DepthWriter(ep / f"depth_{v}.mkv")
            for i in range(d["n"]):
                dw.add(float(t[i] - t0), ds[i], scale)
            if dw.close():
                depth[v] = {"path": ep / f"depth_{v}.mkv", "real": None, "scale_m": dw.scale_m, "source": source}
        signals = h5_signals(f, st, q_abs, fps, chosen[anchor]["n"])
        state, action, state_names, state_src, state_note = h5_state(signals, rig, q_abs)
        instr, notes = h5_text(f, g, st["text"])
    extra = {"task_label": [item["name"]],
             "source": {"format": "hdf5", "file": item["file"].name, "group": g or None, "unused_cameras": unused}}
    if st["unused"]:
        extra["source"]["unused_arrays"] = st["unused"]
    if state_src:
        extra["source"]["state"] = state_src
    if state_note:
        extra["state_note"] = state_note
    if not chosen[anchor]["clock"]:
        extra["source"]["clock_note"] = (f"the file gives no frame times, so frames are {fps:g} per second as it states"
                                         if fps else "the file gives no frame times or rate, so 30 frames per second "
                                                     "was assumed")
    if instr:
        extra.update(instruction=instr, instruction_note="This instruction is the task text stored in the HDF5 file.")
    if notes:
        set_uploader_notes(extra, notes)
    if chosen[anchor]["clock"]:
        extra["clock_origin_s"] = t0         # the recorder's time at the clips' zero
    return video_views_episode(ep, files, rig, dataset, extra, shared_clock=True, signals=signals, depth=depth,
                               state=state, action=action, state_names=state_names)


def depth_scale_attr(ds) -> float | None:
    """Metres per unit from a depth dataset's own attributes (depth_scale, units "mm" or "m"), else None."""
    for k, v in ds.attrs.items():
        if DEPTH_SCALE_KEY.search(str(k)):
            try:
                x = float(np.asarray(v).ravel()[0])
                if 0 < x < 10:
                    return x
            except Exception:
                pass
        if str(k).lower() in ("unit", "units"):
            u = (v.decode() if isinstance(v, bytes) else str(v)).strip().lower()
            if u in ("mm", "millimeter", "millimeters", "millimetre", "millimetres"):
                return 0.001
            if u in ("m", "meter", "meters", "metre", "metres"):
                return 1.0
    return None


def h5_has_camera(path: Path) -> bool:
    import h5py
    try:
        with h5py.File(path, "r") as f:
            return any(h5_kind(p, ds) == "camera" for p, ds in _h5_datasets(f))
    except Exception:
        return False


def h5_file_signals(paths: list[Path], q_abs: np.ndarray, n_anchor: int) -> Signals:
    """Every signal of HDF5 files that hold no camera, placed on videos' frame times (q_abs, the recorder's clock):
    a glove's pressure and hand pose recorded beside a head camera. Only arrays with their own clock can be placed."""
    import h5py
    out = Signals()
    for p in paths:
        try:
            with h5py.File(p, "r") as f:
                st = h5_streams(f, "")
                got = h5_signals(f, {**st, "signal": [s for s in st["signal"] if s["clock"]]}, q_abs, None, n_anchor)
                for k, v in got.items():
                    name = f"{p.stem} {k}" if len(paths) > 1 else k
                    out[name] = v
                    out.meta[name] = got.meta[k]
                out.left_out += got.left_out
                out.left_out += [(s["name"], "no clock to place it against the videos") for s in st["signal"]
                                 if not s["clock"]]
        except Exception as e:
            out.left_out.append((p.name, f"could not be read ({type(e).__name__})"))
    return out


# modules of prepare/ that are the reader and its tools, not dataset adapters
NOT_ADAPTERS = {"__main__", "cli", "display", "folder", "formats", "hub", "lerobot", "remux", "videos"}


def upload_adapters(kind: str) -> list:
    """The dataset adapters in prepare/ that read an upload of this kind ("mcap", "lerobot" or "video"), found rather
    than listed: an adapter
    declares UPLOAD = kind with recognizes() and convert_upload(), or UPLOAD = None when it reads only its published
    dataset (tests/test_formats.py holds every adapter to one or the other). Adding a dataset is adding its adapter."""
    import importlib
    import pkgutil
    import prepare
    out = []
    for m in sorted(pkgutil.iter_modules(prepare.__path__), key=lambda m: m.name):
        if m.name in NOT_ADAPTERS:
            continue
        mod = importlib.import_module(f"prepare.{m.name}")
        if getattr(mod, "UPLOAD", None) == kind:
            out.append(mod)
    return out


def mcap_layout(topics: list[str]) -> str:
    """The adapter whose layout these topics are (its module name), or "generic"."""
    return next((m.__name__.rsplit(".", 1)[-1] for m in upload_adapters("mcap") if m.recognizes(topics)), "generic")


MCAP_MAGIC = b"\x89MCAP0\r\n"


def mcap_layout_context(item: dict, layout: str, ep: Path, dataset: str) -> dict:
    """The context a recognised MCAP layout's reader writes (prepare/<layout>.py), with the format and file added to
    its source and the dataset to the context, so the reader's notes on what it left out stay."""
    import importlib
    ctx = importlib.import_module(f"prepare.{layout}").convert_upload(item, ep)
    if not ctx:
        raise ValueError(f"prepare.{layout} returned no context")
    src = ctx.get("source") if isinstance(ctx.get("source"), dict) else {}
    ctx.update({"dataset": dataset, "source": {**src, "format": f"mcap ({layout} layout)", "adapter": layout,
                                                "file": item["name"]}})
    return ctx


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
    if layout == "generic":
        return convert_mcap_generic(item, rig, ep, dataset)
    ctx = mcap_layout_context(item, layout, ep, dataset)
    # the rate and the length come from the anchor camera's real capture times (stations record at 30 or 60 Hz,
    # and no camera runs at exactly its nominal rate), so sampling is one instant per second of real time and the
    # footage cap counts real minutes
    from label import episode as me
    if ctx.get("real_times"):
        t = np.load(ep / ctx["real_times"])[me.order_views(ctx["cameras"])[0]]
        step = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30
        ctx["fps"] = round(1.0 / step, 3)
        ctx["duration_s"] = round(float(t[-1] - t[0]) + step, 3)
    else:                                   # frames exactly on the adapter's grid: its rate is the real one
        ctx["duration_s"] = round(ctx["n_state_frames"] / float(ctx["fps"]), 3)
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


def mcap_channels(path: Path) -> list[tuple[str, str]]:
    """[(topic, schema name)] of an MCAP file, from its summary, or scanned when it has none (a cut-off file)."""
    from mcap.reader import make_reader
    try:
        with open(path, "rb") as fh:
            s = make_reader(fh).get_summary()
    except Exception:
        s = None
    if s is None:
        return _mcap_channels_by_scan(path)
    return sorted({(c.topic, s.schemas[c.schema_id].name if c.schema_id in s.schemas else "") for c in s.channels.values()})


def mcap_has_camera(path: Path) -> bool:
    return any(CAMERA_SCHEMA.search(s) for _, s in mcap_channels(path))


# ---------------------------------------------------------------- recorded joint state, from any MCAP

# A channel is an arm's joints when its messages carry a joint vector under one of these names (sensor_msgs/JointState's
# position included), with the gripper's reading beside it or as the vector's last value. Its topic names the arm's
# side, and a leader arm or a command channel is the action rather than the state.
JOINT_KEYS = ("joint_pos", "joint_positions", "joint_position", "positions", "position", "qpos")
GRIPPER_KEYS = ("gripper_pos", "gripper_position", "gripper", "gripper_width", "gripper_opening")
ACTION_TOPIC = re.compile(r"leader|action|command|cmd|target|teleop", re.I)
JOINT_DIMS = 7                     # six joints and a gripper per arm, as state_layout and the checks read state


def _vector(x) -> list[float] | None:
    if x is None or isinstance(x, (str, bytes, dict)):
        return None
    try:
        v = [float(a) for a in (x if hasattr(x, "__len__") else [x])]
    except (TypeError, ValueError):
        return None
    return v or None


def _joint_row(msg) -> list[float] | None:
    joints = next((v for v in (_vector(_field(msg, k)) for k in JOINT_KEYS) if v), None)
    if joints is None:
        return None
    grip = next((v for v in (_vector(_field(msg, k)) for k in GRIPPER_KEYS) if v), None)
    return joints + grip[:1] if grip else joints


def _name_lists(items) -> list[list[str]]:
    """The lists of names among a message's fields (_msg_items): any sequence of nonempty strings, as _vector reads
    values, so a Protobuf repeated field or a ROS string[] is read like a JSON list. A field _numbers leaves out (the
    message's own stamps and bookkeeping, _skipped_field) is never a list of names."""
    out = []
    for k, v, _set in items:
        if (_skipped_field(str(k)) or v is None or isinstance(v, (str, bytes, bytearray, memoryview, dict))
                or not hasattr(v, "__len__")):
            continue
        v = list(v)
        if v and all(isinstance(x, str) and x for x in v):
            out.append(v)
    return out


def _sibling_names(lists: list[list[str]], n: int) -> list[str] | None:
    """The names of the n values of a numeric array, from the name lists beside it in the same (sub)message
    (_name_lists): the one list there with exactly n distinct entries. sensor_msgs/JointState's name names its
    position, velocity and effort this way, and a vendor's names, joint_names or labels beside its own list does the
    same. None when no list has n distinct entries (a units list of "rad" three times names nothing), or two different
    ones do, since either could be the names; a list in another message or under another parent never names the
    array."""
    found = {tuple(v) for v in lists if len(v) == n and len(set(v)) == n}
    return list(found.pop()) if len(found) == 1 else None


def _joint_names(msg, n: int) -> list[str] | None:
    """The names a joint message gives the values of its row (_sibling_names, so sensor_msgs/JointState's name), with
    "gripper" for a gripper reading _joint_row appended from its own field; None when the message names none, or not
    one per value."""
    lists = _name_lists(_msg_items(msg))
    names = _sibling_names(lists, n)
    if names is None and next((g for g in (_vector(_field(msg, k)) for k in GRIPPER_KEYS) if g), None):
        names = _sibling_names(lists, n - 1)
        return names + ["gripper"] if names else None
    return names


def _joint_fields(msg) -> set:
    """{(field, its value names as a frozenset, or None)} of the joint vector and the gripper reading _joint_row reads
    from a message, as mcap_signals names those fields (_numbers), so a channel's other name sets stay signals."""
    lists, out = _name_lists(_msg_items(msg)), set()
    for keys in (JOINT_KEYS, GRIPPER_KEYS):
        k, v = next(((k, v) for k, v in ((k, _vector(_field(msg, k))) for k in keys) if v), (None, None))
        if k is not None:
            names = _sibling_names(lists, len(v))
            out.add((k, frozenset(names) if names else None))
    return out


# A channel's messages need not all name the same values: ROS's /joint_states carries every driver's joints, each in
# messages of its own, and a merged JointState can list the same joints in another order message by message. Rows are
# read by their names (name_group), never by position under the first message's names; a field whose messages name
# their values in more than NAME_SETS_MAX ways (a detector's labels) holds no value that is one reading over time.
NAME_SETS_MAX = 8


class NameSets(list):
    """One channel's (or one field's) name sets [(names, width)] in the order first seen (name_group), with each set's
    place by its frozenset of names, and overflow once its messages named more than NAME_SETS_MAX sets."""

    def __init__(self):
        super().__init__()
        self.index: dict = {}
        self.overflow = False


def name_group(groups: NameSets, names, vals: list) -> tuple[int | None, list]:
    """(the index in groups of one message's row, its values in that group's order), extending groups. A row whose
    names give one distinct name per value is grouped with the rows that name the same set (looked up by set, so a
    detector's labels cost one lookup a message) and reordered by name to the group's first order; a row without such
    names is grouped with the unnamed rows of its width. None for a row naming a new set once groups has
    NAME_SETS_MAX sets, when groups.overflow is set and grouping stops."""
    n = len(vals)
    if names is not None and len(names) == n and len(set(names)) == n:
        key = frozenset(names)
        i = groups.index.get(key)
        if i is None:
            if len(groups.index) >= NAME_SETS_MAX:
                groups.overflow = True
                return None, vals
            groups.index[key] = i = len(groups)
            groups.append((list(names), n))
            return i, vals
        gn = groups[i][0]
        if list(names) != gn:
            at = {x: j for j, x in enumerate(names)}
            vals = [vals[at[x]] for x in gn]
        return i, vals
    i = groups.index.get(n)
    if i is None:
        groups.index[n] = i = len(groups)
        groups.append((None, n))
    return i, vals


def merge_unnamed(groups: NameSets, rows: dict, lists: tuple) -> tuple[dict, int]:
    """({group index: row} of the name sets read, how many sets the channel or field has once its unnamed rows are
    placed). rows {group index: {"t": times, and each of lists: values}} come from name_group. A message without names
    is read as the reader read it before name sets: its rows join the only named set of their width when there is
    exactly one (a JointState with an empty name list for its first second), and are left with that set when the set
    is read elsewhere (the state). Otherwise the unnamed rows of the first unnamed message's width are one set and
    rows of any other width are dropped, counted in that set's "dropped" so a reason given for it can say so."""
    named: dict = {}
    for i, (gn, w) in enumerate(groups):
        if gn is not None:
            named.setdefault(w, []).append(i)
    home = lambda i: named[groups[i][1]][0] if len(named.get(groups[i][1], [])) == 1 else None
    rest = [i for i in rows if groups[i][0] is None and home(i) is None]
    keep = min(rest, key=lambda i: rows[i]["t"][0]) if rest else None
    out = {i: r for i, r in rows.items() if groups[i][0] is not None or i == keep}
    for i in sorted(rows):
        if groups[i][0] is not None or i == keep:
            continue
        h, r = home(i), rows[i]
        if h is None:
            out[keep]["dropped"] = out[keep].get("dropped", 0) + len(r["t"])
        elif h in out:
            m = out[h]
            order = np.argsort(np.concatenate([m["t"], r["t"]]), kind="stable")
            for k in ("t",) + lists:
                both = list(m[k]) + list(r[k])
                m[k] = [both[j] for j in order]
            if "set" in m:
                m["set"] = m["set"] or r["set"]
            if "fields" in m:
                m["fields"] = m["fields"] | r["fields"]
    return out, sum(1 for gn, _ in groups if gn is not None) + (keep is not None)


def group_label(group: tuple) -> str:
    """How a name set (name_group) is told apart from the others on its channel: its names, the first two and a count
    past three, or its width when it names none."""
    names, n = group
    if names is None:
        return f" ({n} values)"
    return " (" + (", ".join(names) if len(names) <= 3 else f"{names[0]}, {names[1]} and {len(names) - 2} more") + ")"


def _join_gripper(groups: list[dict]) -> list[dict]:
    """A channel's two name sets as one arm when one names only a gripper and the other names joints and no gripper
    (an arm's driver and its gripper's driver both publishing /joint_states): the gripper's readings, placed at the
    arm's message times, follow the joints as _joint_row puts a gripper field after them. Kept apart when the gripper
    leaves a gap in the arm's time longer than STATE_EDGE_SLACK_S (fill_rows), or the channel has any other name
    set."""
    grip = [g for g in groups if g["names"] and all(STATE_GRIPPER_NAME.search(x) for x in g["names"])]
    if len(groups) != 2 or len(grip) != 1:
        return groups
    grip = grip[0]
    arm = groups[1] if groups[0] is grip else groups[0]
    if not arm["names"] or any(STATE_GRIPPER_NAME.search(x) for x in arm["names"]) or len(arm["t"]) < 2:
        return groups
    at_arm, gap = fill_rows(arm["t"], grip["t"], grip["pos"])
    if gap:
        return groups
    pos = np.concatenate([arm["pos"], at_arm], axis=1)
    return [{"t": arm["t"], "pos": pos, "names": arm["names"] + grip["names"],
             "fields": arm["fields"] | grip["fields"]}]


def mcap_joint_streams(paths: list[Path]) -> dict:
    """{key: {"t": seconds on the recording's clock, "pos": rows, "names": the value names its messages give, or None,
    "topic": its channel}} for every channel of these MCAP files that carries an arm's joints (JOINT_KEYS); cameras and
    text are not read. A channel's rows are grouped by the names their messages give (name_group), each in its group's
    order; a channel of one group (or of an arm and its gripper, _join_gripper) is keyed by its topic, and one of
    several by its topic and each group's label (group_label), with the fields its messages fill ("fields",
    _joint_fields) so that only the group joint_state reads leaves the signals."""
    from mcap.reader import make_reader
    found, facs = {}, _decoders()
    for p in paths:
        chans = [(t, s) for t, s in mcap_channels(p) if not CAMERA_SCHEMA.search(s) and not TEXT_TOPIC.search(t)]
        decs, skip = {}, set()
        with open(p, "rb") as fh:
            try:
                msgs = make_reader(fh).iter_messages(topics=[t for t, _ in chans], log_time_order=True)
                for schema, ch, msg in msgs:
                    if ch.topic in skip:
                        continue
                    if ch.id not in decs:
                        decs[ch.id] = _decoder_for(ch.message_encoding, schema, facs)
                    try:
                        m = decs[ch.id](msg.data) if decs[ch.id] else None
                        row = _joint_row(m) if m is not None else None
                    except Exception:
                        m, row = None, None
                    if row is None:
                        if ch.topic not in found:
                            skip.add(ch.topic)        # not a joint channel (health, status, poses)
                        continue
                    c = found.setdefault(ch.topic, {"groups": NameSets(), "rows": []})
                    i, row = name_group(c["groups"], _joint_names(m, len(row)), row)
                    if i is None:
                        continue                      # a channel of more name sets than an arm has: not read here
                    if i == len(c["rows"]):
                        c["rows"].append({"t": [], "pos": [], "fields": set()})
                    r = c["rows"][i]
                    r["t"].append(msg.log_time / 1e9)
                    r["pos"].append(row)
                    r["fields"] |= _joint_fields(m)
            except Exception:
                pass                                  # a cut-off file: the messages before the cut are kept
    out = {}
    for topic, c in found.items():
        if c["groups"].overflow:
            continue                                  # mcap_signals names it as left out, with the reason
        rows, n_sets = merge_unnamed(c["groups"], dict(enumerate(c["rows"])), ("pos",))
        read = [{"t": np.asarray(r["t"]), "pos": np.asarray(r["pos"], dtype=np.float64),
                 "names": c["groups"][i][0], "fields": r["fields"], "label": group_label(c["groups"][i])}
                for i, r in sorted(rows.items())]
        groups = _join_gripper(read)
        apart = len(groups) == len(read) and n_sets > 1
        for g in groups:
            if len(g["t"]) > 1:
                s = {"t": g["t"], "pos": g["pos"], "names": g["names"], "topic": topic}
                if apart:
                    s["fields"] = g["fields"]
                out[topic + (g["label"] if apart else "")] = s
    return out


# Every other number an MCAP records (a gripper's IMU, an arm's joint velocities and torques, a base's odometry), read
# as recorded_signals reads a LeRobot table's other columns: per channel, each numeric field under the dataset's own
# name, placed on the anchor camera's frames. A channel with fewer messages than SIGNAL_MIN_HZ per second (a
# calibration, a process's CPU report) is not a per-frame record, and one that does not span the footage to within
# STATE_EDGE_SLACK_S at both ends is left out rather than held flat where nothing was recorded.
SIGNAL_MIN_HZ = 1.0
SIGNAL_SKIP_PARTS = {"header", "timestamp", "stamp"}      # a message's own time and sequence bookkeeping
STATE_EDGE_SLACK_S = 0.5


def lerp_rows(q: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """y's rows, read at times t, at times q: each column linearly interpolated, and a time before the first reading or
    after the last one holding that reading (np.interp). abc130k's arms are placed on the frames this way, and the
    state readers do it through fill_rows, which fills no gap longer than STATE_EDGE_SLACK_S."""
    return np.stack([np.interp(q, t, y[:, j]) for j in range(y.shape[1])], axis=1)


def fill_rows(q: np.ndarray, t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray | None, tuple[float, float] | None]:
    """(y's rows at times q by lerp_rows, None), or (None, (start, end) of the longest gap) when two readings in a row
    leave more than STATE_EDGE_SLACK_S of q's span without a reading, or the first or last reading is further than
    that from q's ends. A gap is measured inside the footage, so a message latched seconds before the first frame
    does not make the stretch before the footage a gap. A straight line across a recorder that stopped for 2 s would
    be shown as recorded motion, and a reading held past the ends as stillness, where no still span can tell, so a
    state is filled across no longer gap than the slack its edges are allowed. Both state readers place an arm this
    way: an MCAP arm's channel (joint_state) and an HDF5 state's frames with a reading (h5_state)."""
    t = np.asarray(t, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if len(t) and len(q):
        lo, hi = np.maximum(t[:-1], q[0]), np.minimum(t[1:], q[-1])
        gaps = [(float(lo[i]), float(hi[i])) for i in np.flatnonzero(hi - lo > STATE_EDGE_SLACK_S)]
        gaps += [(float(q[0]), float(t[0]))] if t[0] > q[0] + STATE_EDGE_SLACK_S else []
        gaps += [(float(t[-1]), float(q[-1]))] if t[-1] < q[-1] - STATE_EDGE_SLACK_S else []
        if gaps:
            return None, max(gaps, key=lambda g: g[1] - g[0])
    return lerp_rows(q, t, y), None


def gap_words(gap: tuple[float, float], zero: float) -> str:
    """A gap fill_rows will not fill, in seconds of the footage (zero its first frame), for a state note."""
    return (f"has no reading from {gap[0] - zero:.1f} s to {gap[1] - zero:.1f} s, a gap longer than the "
            f"{STATE_EDGE_SLACK_S:g} s the reader fills")


def _msg_items(m) -> list:
    """(name, value, set) for each field of a decoded message. A Protobuf scalar left at its default is not set: proto3
    cannot tell a 0 the recorder wrote from a field it never filled in."""
    if isinstance(m, dict):
        return [(k, v, True) for k, v in m.items()]
    desc = getattr(m, "DESCRIPTOR", None)
    if desc is not None and hasattr(desc, "fields"):          # protobuf: every declared field, zeros included
        present = {fd.name for fd, _ in m.ListFields()}
        return [(fd.name, getattr(m, fd.name), fd.name in present) for fd in desc.fields
                if fd.message_type is None or fd.name in present]
    slots = getattr(type(m), "__slots__", None)
    if slots:
        return [(s, getattr(m, s, None), True) for s in slots]
    return [(k, v, True) for k, v in vars(m).items()] if hasattr(m, "__dict__") else []


def _is_num(x) -> bool:
    return isinstance(x, (bool, int, float, np.number))


def _skipped_field(k: str) -> bool:
    """Whether _numbers leaves a message's field out: its own time and sequence bookkeeping (SIGNAL_SKIP_PARTS,
    SIGNAL_SKIP) or a private field."""
    return k in SIGNAL_SKIP_PARTS or k.startswith("_") or bool(SIGNAL_SKIP.search(k))


def _numbers(m, path: str = "") -> dict:
    """{name: (values, set, value names, shape)} for the numbers of one decoded message (JSON, Protobuf, ROS): a
    repeated numeric field is one vector under its path, the scalars of one (sub)message one vector under that message's
    path (each value named by its field, so a position reads x, y, z), set when any of them was written (_msg_items). A
    repeated field of messages (the 21 joints of a tracked hand as a list of poses, the taxels of a glove as a list of
    readings) is one array under its path, shaped (count, numbers per element), when every element carries the same
    numbers. A list of names beside a numeric array with one name per value names its values (_sibling_names, a
    JointState's name for its position, velocity and effort). Strings, bytes and the message's own stamps are left
    out."""
    out, items, lists = {}, _msg_items(m), None
    for k, v, was_set in items:
        k = str(k)
        if _skipped_field(k):
            continue
        p = f"{path}.{k}" if path else k
        if _is_num(v):
            vals, seen, names, _ = out.get(path, ([], False, [], None))
            out[path] = (vals + [float(v)], seen or was_set, names + [k], None)
        elif v is None or isinstance(v, (str, bytes, bytearray, memoryview)):
            continue
        elif hasattr(v, "__len__") and not isinstance(v, dict) and not hasattr(v, "DESCRIPTOR"):
            vals = list(v)
            if vals and all(_is_num(x) for x in vals):
                lists = _name_lists(items) if lists is None else lists
                out[p] = ([float(x) for x in vals], True, _sibling_names(lists, len(vals)), None)
            elif vals and all(_is_num(y) for x in vals if isinstance(x, (list, tuple)) for y in x) \
                    and all(isinstance(x, (list, tuple)) and len(x) == len(vals[0]) and len(x) for x in vals):
                out[p] = ([float(y) for x in vals for y in x], True, None, (len(vals), len(vals[0])))
            elif vals:
                rows = [_flat_numbers(x) for x in vals]
                if rows[0] and all(r is not None and [n for n, _ in r] == [n for n, _ in rows[0]] for r in rows):
                    per = [n for n, _ in rows[0]]
                    out[p] = ([x for r in rows for _, x in r], True,
                              [f"{i}.{n}" for i in range(len(rows)) for n in per], (len(rows), len(per)))
        else:
            out.update(_numbers(v, p))
    return out


def _flat_numbers(m) -> list[tuple[str, float]] | None:
    """[(name, value)] of every number in one element of a repeated field of messages, in field order (position.x,
    position.y, ...), or None when it holds none."""
    if _is_num(m):
        return [("value", float(m))]
    flat = []
    for path, (vals, _seen, names, _shape) in _numbers(m).items():
        flat += [(f"{path}.{n}" if path else str(n), x) for n, x in zip(names or range(len(vals)), vals)]
    return flat or None


SAMPLE_RATE_KEY = re.compile(r"^(sample_?rate|sampling_?rate|sample_?freq(uency)?|fs)$", re.I)
SAMPLES_MIN = 32           # a vector this long beside a sample rate is samples in time (audio), not channels
MUX_MAX_VALUES = 8         # a field with at most this many integer values that keeps switching names the sensor
VARIATION_MAX_VALUES = 64  # a fast signal this wide or narrower also gets its variation within each frame
VARIATION_MIN_SHARE = 0.05  # kept only when it reaches this share of the signal's own range somewhere in the episode
COUNTER_MIN_MESSAGES = 20  # a value must rise at this many messages in a row to be taken for a counter or a clock


def place_on_frames(t: np.ndarray, v: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, int]:
    """(values, variation within each frame or None, frames with no reading) of samples (t seconds, v rows) on frame
    times q. A sensor no faster than twice the frame rate gives each frame its nearest sample, and a frame with no
    sample near it is NaN rather than the last value held. A faster sensor gives each frame the mean of the samples in
    its interval and their standard deviation, so a vibration or a slip between two frames is not lost by picking one
    sample; a frame whose interval holds no sample is NaN."""
    t, q = np.asarray(t, dtype=np.float64), np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    dq = float(np.median(np.diff(q))) if len(q) > 1 else 1 / 30
    rate = len(t) / max(float(t[-1] - t[0]), 1e-9) if len(t) > 1 else 0.0
    if rate <= 2.0 / dq:
        idx = nearest(t, q)
        a = v[idx].copy()
        far = np.abs(t[idx] - q) > max(2.5 * float(np.median(np.diff(t))) if len(t) > 1 else 0.0, 2.5 * dq)
        a[far] = np.nan
        return a, None, int(far.sum())
    edges = np.concatenate([[q[0] - dq / 2], (q[1:] + q[:-1]) / 2, [q[-1] + dq / 2]])
    bin_ = np.searchsorted(edges, t, side="right") - 1
    ok = (bin_ >= 0) & (bin_ < len(q))
    n = np.bincount(bin_[ok], minlength=len(q)).astype(np.float64)
    tot = np.zeros((len(q), v.shape[1]))
    sq = np.zeros((len(q), v.shape[1]))
    np.add.at(tot, bin_[ok], v[ok])
    np.add.at(sq, bin_[ok], v[ok] ** 2)
    with np.errstate(all="ignore"):
        mean = tot / n[:, None]
        std = np.sqrt(np.maximum(sq / n[:, None] - mean ** 2, 0.0))
    mean[n == 0] = np.nan
    std[n == 0] = np.nan
    return mean, std, int((n == 0).sum())


def _variation_matters(name: str, a: np.ndarray, var: np.ndarray, samples: bool = False) -> bool:
    """Whether a fast signal's variation within frames is worth a signal of its own: it is where a vibration or a slip
    shows, so it is kept for a touch signal (label/signals.py is_touch, by its name and its numbers) or the loudness of
    samples in time (a contact microphone), and there only when it reaches VARIATION_MIN_SHARE of the signal's own
    range somewhere in the episode. An IMU's or a pose's jitter between frames is left out."""
    from label import signals as sg
    if not samples and not sg.is_touch(name, a):
        return False
    with np.errstate(all="ignore"):
        rng = np.nanmax(a, axis=0) - np.nanmin(a, axis=0)
        top = np.nanmax(var, axis=0)
    ok = np.isfinite(rng) & np.isfinite(top) & (rng > 0)
    return bool(ok.any() and (top[ok] >= VARIATION_MIN_SHARE * rng[ok]).any())


def _counters(r: dict) -> list[int]:
    """The values of a row that count up message by message (a device's clock or sequence number): bookkeeping."""
    v = np.asarray(r["v"], dtype=np.float64)
    if len(v) < COUNTER_MIN_MESSAGES:
        return []
    d = np.diff(v, axis=0)
    # whole numbers that never fall and rise at nearly every message (a device clock can stamp two messages alike)
    return [int(i) for i in range(v.shape[1]) if np.all(v[:, i] == np.round(v[:, i])) and (d[:, i] >= 0).all()
            and (d[:, i] > 0).mean() >= 0.9]


def _multiplexed(r: dict) -> int | None:
    """The value of a row that says which sensor each message is from (an IMU topic whose `type` is 1 for the
    accelerometer, 2 for the gyroscope): integers, at most MUX_MAX_VALUES of them, switching between at least a tenth of
    consecutive messages. None when no value does."""
    v = np.asarray(r["v"], dtype=np.float64)
    if len(v) < 10 or v.shape[1] < 2:
        return None
    for i in range(v.shape[1]):
        c = v[:, i]
        vals = np.unique(c)
        if 2 <= len(vals) <= MUX_MAX_VALUES and np.all(vals == np.round(vals)) and (np.diff(c) != 0).mean() >= 0.1:
            return i
    return None


def mcap_signals(paths: list[Path], q: np.ndarray, used: dict | None = None) -> Signals:
    """{name: (len(q), values) array} of every numeric field these MCAP files record on channels that are not cameras
    or text, sampled at the recorded message nearest each anchor frame time q (seconds, the files' log-time clock).
    used {topic: fields already read, or None for the whole channel} keeps out what the reader already shows as the
    state. The name is the topic, then the field path ("/robot0/sensor/imu angular_velocity"); each signal keeps its
    values' names and its shape (Signals.meta), and a field that is read but not kept is named with the reason
    (Signals.left_out): too few messages to be a per-frame record, not covering the footage, values that are not
    finite, or wider than SIGNAL_MAX_VALUES. A field's messages are grouped by the names they give its values
    (name_group), each put in its group's order, and a field of several groups is one signal per group, its name
    followed by the group's label (group_label). used names a field, or a (field, its value names) pair for a field
    of that name set only (state_fields)."""
    from mcap.reader import make_reader
    used, facs = used or {}, _decoders()
    q = np.asarray(q, dtype=np.float64)
    rows: dict[tuple, dict] = {}
    sets: dict[tuple, NameSets] = {}      # (topic, field): its name sets (name_group)
    for p in paths:
        chans = [t for t, s in mcap_channels(p) if not CAMERA_SCHEMA.search(s) and not TEXT_TOPIC.search(t)
                 and not (t in used and used[t] is None)]
        decs = {}
        with open(p, "rb") as fh:
            try:
                try:
                    has_summary = make_reader(fh).get_summary() is not None
                except Exception:
                    has_summary = False
                fh.seek(0)
                # a recording cut off before its index is read message by message (_mcap_stream)
                msgs = make_reader(fh).iter_messages(topics=chans, log_time_order=True) if has_summary \
                    else _mcap_stream(p, set(chans))
                for schema, ch, msg in msgs:
                    if ch.id not in decs:
                        decs[ch.id] = _decoder_for(ch.message_encoding, schema, facs)
                    try:
                        nums = _numbers(decs[ch.id](msg.data)) if decs[ch.id] else {}
                    except Exception:
                        nums = {}
                    for field, (vals, was_set, names, shape) in list(nums.items()):
                        # a field the reader already shows (the state) is still grouped, so the name sets beside it
                        # keep the label that tells them from it
                        u = used.get(ch.topic) or ()
                        if field in u or (field, frozenset(names) if names else None) in u:
                            name_group(sets.setdefault((ch.topic, field), NameSets()), names, vals)
                            continue
                        # samples in time beside a sample rate (a contact microphone's 512 samples at 48 kHz): kept as
                        # their loudness, root mean square and peak, not as 512 channels
                        parent = field.rsplit(".", 1)[0] if "." in field else ""
                        sib = nums.get(parent) if parent != field else None
                        rate = next((x for n_, x in zip((sib or [None, None, []])[2] or [], (sib or [[]])[0])
                                     if SAMPLE_RATE_KEY.match(str(n_))), None) if sib else None
                        kind = None
                        if rate and len(vals) >= SAMPLES_MIN and shape is None:
                            x = np.asarray(vals, dtype=np.float64)
                            vals, names, kind = [float(np.sqrt(np.mean(x ** 2))), float(np.max(np.abs(x)))], \
                                ["root mean square", "peak"], {"kind": "samples", "sample_rate": float(rate)}
                        i, vals = name_group(sets.setdefault((ch.topic, field), NameSets()), names, vals)
                        if i is None:
                            continue
                        r = rows.setdefault((ch.topic, field, i),
                                            {"t": [], "v": [], "d": len(vals), "set": False, "names": names,
                                             "shape": shape, "topic": ch.topic, "kind": kind})
                        r["t"].append(msg.log_time / 1e9)
                        r["v"].append(vals)
                        r["set"] |= was_set
            except Exception:
                pass                                  # a cut-off file: the messages before the cut are kept
    out = Signals()
    named, by_field = {}, {}
    for (topic, field, i), r in rows.items():
        by_field.setdefault((topic, field), {})[i] = r
    for (topic, field), field_rows in by_field.items():
        name, groups = f"{topic} {field}".strip(), sets[(topic, field)]
        if groups.overflow:
            out.left_out.append((name, f"its messages name its values in more than {NAME_SETS_MAX} different "
                                       "ways, so no value is one reading over time"))
            continue
        field_rows, n_sets = merge_unnamed(groups, field_rows, ("v",))
        for i, r in sorted(field_rows.items()):
            named[name + (group_label(groups[i]) if n_sets > 1 else "")] = r
    rows = named
    # bookkeeping values (a counter, a device clock) leave the row; a row that names its sensor per message is split

    def no_counters(name, r):
        cnt = _counters(r)
        if not cnt:
            return r
        nm = r["names"] or [f"[{i}]" for i in range(r["d"])]
        keep = [i for i in range(r["d"]) if i not in cnt]
        out.left_out.append((f"{name} " + ", ".join(str(nm[i]) for i in cnt),
                             "rises at every message, so a counter or a clock rather than a reading"))
        if not keep:
            return None
        return {**r, "v": [[row[i] for i in keep] for row in r["v"]], "d": len(keep),
                "names": [nm[i] for i in keep] if r["names"] else None, "shape": None}
    split = {}
    for name, r in rows.items():
        if not r["v"]:
            split[name] = r
            continue
        r = no_counters(name, r)
        if r is None:
            continue
        mux = _multiplexed(r)
        if mux is None:
            split[name] = r
            continue
        nm = r["names"] or [f"[{i}]" for i in range(r["d"])]
        v = np.asarray(r["v"])
        t = np.asarray(r["t"])
        for val in np.unique(v[:, mux]):
            sel = v[:, mux] == val
            keep = [i for i in range(r["d"]) if i != mux]
            sub = f"{name} ({nm[mux]} {int(val)})"
            part = no_counters(sub, {**r, "t": list(t[sel]), "v": v[sel][:, keep].tolist(), "d": len(keep),
                                     "names": [nm[i] for i in keep] if r["names"] else None, "shape": None})
            if part is not None:
                split[sub] = part
    rows = split
    span = float(q[-1] - q[0]) if len(q) > 1 else 0.0
    sparse = []
    for name, r in rows.items():
        t = np.asarray(r["t"])
        if not r["set"] or not r["d"] or span <= 0:
            continue
        # rows of another width than its first message's were dropped (merge_unnamed): a reason says so
        some = (f"only {len(t)} of its {len(t) + r['dropped']} messages carry the {r['d']} values of its first "
                "message, ") if r.get("dropped") else ""
        if len(t) < 2 or len(t) / span < SIGNAL_MIN_HZ:
            if some:
                out.left_out.append((name, f"{some}fewer than {SIGNAL_MIN_HZ:g} a second"))
            else:
                sparse.append(name)           # a setting, a calibration or a status report, not a per-frame record
            continue
        if r["d"] > SIGNAL_MAX_VALUES:
            out.left_out.append((name, f"{r['d']} values per message, more than the {SIGNAL_MAX_VALUES} a signal "
                                       "holds"))
            continue
        if t[0] > q[0] + STATE_EDGE_SLACK_S or t[-1] < q[-1] - STATE_EDGE_SLACK_S:
            out.left_out.append((name, f"{some}recorded from {t[0] - q[0]:.1f} s to {t[-1] - q[0]:.1f} s, not over "
                                       "the whole footage"))
            continue
        v = np.asarray(r["v"], dtype=np.float64)
        if not np.isfinite(v).all():
            out.left_out.append((name, NOT_FINITE))
            continue
        rate = len(t) / max(float(t[-1] - t[0]), 1e-9)
        # a frame with no message near it (a hand the tracker lost, a sensor that paused) is NaN, not the last value
        # held; a sensor faster than the camera gives each frame the mean of its interval (place_on_frames)
        a, var, gaps = place_on_frames(t, v, q)
        out.add(name, a, shape=r["shape"], names=r["names"], source=f"MCAP channel {r['topic']}")
        out.meta[name]["rate_hz"] = round(rate, 2)
        if r.get("kind"):
            out.meta[name].update(r["kind"])
        if gaps:
            out.meta[name]["gaps"] = gaps
        if (var is not None and a.shape[1] <= VARIATION_MAX_VALUES
                and _variation_matters(name, a, var, bool(r.get("kind")))):
            vn = f"{name} variation within each frame"
            out.add(vn, var, shape=r["shape"], names=r["names"], source=f"MCAP channel {r['topic']}")
            out.meta[vn].update(rate_hz=round(rate, 2), variation_of=name)
    if sparse:
        out.left_out.append((", ".join(sparse[:6]) + (f" and {len(sparse) - 6} more" if len(sparse) > 6 else ""),
                             f"fewer than {SIGNAL_MIN_HZ:g} message per second, so settings or reports rather than a "
                             "per-frame record"))
    return out


def joint_state(streams: dict, q: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """(state, action, note): the arms' joints and grippers interpolated onto the anchor camera's frame times q (the
    same clock as the streams), 7 values per arm, left arm first; the leader or command channels, when they match, as
    the action. None with a note when the streams are not a layout the checks read or do not cover the footage."""
    st, act = arm_streams(streams, False), arm_streams(streams, True)
    if not st:
        return None, None, None
    order = [s for s in ("left", "right", "only") if s in st]
    if "only" in order and len(order) > 1:
        if "left" not in st or "right" not in st:
            return None, None, "Labelled from the cameras, because the recorded arm channels do not say which arm is which."
        # an arm that names a side and one that does not could be the same side's arm twice, but a left and a right
        # arm are the two working arms, and an arm beside them that names no side is a third one (third_arms)
        order = ["left", "right"]
    dims = [streams[st[s]]["pos"].shape[1] for s in order]
    if any(d != JOINT_DIMS for d in dims):
        return None, None, (f"Labelled from the cameras, because the recorded arms have {' and '.join(map(str, dims))} "
                            "values per frame and our checks read six joints and a gripper per arm.")
    for s in order:
        # the channel's own value names settle six joints and a gripper against seven joints (a Franka arm)
        kind, why = state_layout(JOINT_DIMS, "teleop_arms", streams[st[s]].get("names"))
        if kind != "joints":
            return None, None, why or ("Labelled from the cameras, because the recorded arm channels name their values "
                                       "as a pose, not six joints and a gripper.")
    span = float(q[-1] - q[0]) if len(q) > 1 else 0.0

    def covers(topic):
        # the arm's samples must span the footage: a frame np.interp places past the first or last sample holds that
        # sample's value, so the state would seem to record an arm standing still where nothing was recorded
        t = streams[topic]["t"]
        return span > 0 and t[0] <= q[0] + STATE_EDGE_SLACK_S and t[-1] >= q[-1] - STATE_EDGE_SLACK_S
    if not all(covers(st[s]) for s in order):
        return None, None, "Labelled from the cameras, because the recorded arm state does not cover the footage's time."

    def fill(topic):
        # the arm's readings on the frames, across no gap longer than the slack (fill_rows)
        return fill_rows(q, streams[topic]["t"], streams[topic]["pos"])
    rows = [fill(st[s]) for s in order]
    gap = next(((st[s], g) for s, (_, g) in zip(order, rows) if g), None)
    if gap:
        return None, None, (f"Labelled from the cameras, because the recorded arm state {gap[0]} "
                            f"{gap_words(gap[1], float(q[0]))}.")
    state = np.concatenate([r for r, _ in rows], axis=1)
    action = None
    if all(s in act and streams[act[s]]["pos"].shape[1] == JOINT_DIMS and covers(act[s]) for s in order):
        cmd = [fill(act[s])[0] for s in order]
        action = None if any(c is None for c in cmd) else np.concatenate(cmd, axis=1)
    return state, action, None


def _topic(streams: dict, key: str) -> str:
    # the channel a stream was read from: its key, or the key without the label of its name set (mcap_joint_streams)
    return streams[key].get("topic", key)


def _arm_rank(streams: dict, key: str) -> tuple:
    # six joints and a gripper first, then the widest; within a channel, the name set that fits the layout, a named one
    # over an unnamed one, and the one with the most readings, never the first label
    s = streams[key]
    dims, names = s["pos"].shape[1], s.get("names")
    fits = names is not None and state_layout(dims, "teleop_arms", names)[0] == "joints"
    return dims != JOINT_DIMS, -dims, _topic(streams, key), not fits, names is None, -len(s["t"]), key


def arm_streams(streams: dict, role: bool) -> dict:
    """{side: key} of the arm streams joint_state reads, the commands when role: per side its channel names (left,
    right, or "only"), a stream of six joints and a gripper when there is one (an arm can also record other vectors),
    else the widest, so a channel's arm is chosen over the gripper published apart from it."""
    by_side = {}
    for t in sorted(streams, key=lambda t: _arm_rank(streams, t)):
        if bool(ACTION_TOPIC.search(_topic(streams, t))) == role:
            by_side.setdefault(side_of(_topic(streams, t)) or "only", t)
    return by_side


def third_arms(streams: dict) -> list[str]:
    """The recorded arm channels that are neither working arm: those whose topic names no side, beside a left and a
    right arm (a third arm that carries the scene camera, as on a rig whose camera an operator moves). joint_state
    reads the two sided arms and leaves these out."""
    st = [t for t in streams if not ACTION_TOPIC.search(_topic(streams, t))]
    if not {"left", "right"} <= {side_of(_topic(streams, t)) for t in st}:
        return []
    return sorted(t for t in st if side_of(_topic(streams, t)) is None)


def state_fields(streams: dict, state, action) -> dict:
    """{topic: fields} of the joint channels joint_state read as the state (and the action, when it matched), for
    mcap_signals to leave out; a third arm's joints, and every joint channel when no state was read, stay signals. Of
    a channel whose messages name several sets of values (mcap_joint_streams), only the sets joint_state read leave,
    each as (field, its value names) pairs, so a gripper or wheels named apart from the arm stay signals."""
    if state is None:
        return {}
    third = set(third_arms(streams))
    read = set(arm_streams(streams, False).values()) | (set(arm_streams(streams, True).values()) if action is not None
                                                        else set())
    out = {}
    for t, s in streams.items():
        topic = _topic(streams, t)
        if t in third or (action is None and ACTION_TOPIC.search(topic)):
            continue
        if "fields" not in s:
            out[topic] = set(JOINT_KEYS) | set(GRIPPER_KEYS)
        elif t in read:
            out.setdefault(topic, set()).update(s["fields"])
    return out


def third_arm_camera_desc(topics: list[str]) -> str:
    return (f"a camera on a third arm, which is not either working arm (its joints are recorded as {', '.join(topics)}). "
            "That arm can move, so this view can pan and tilt, and a change of view while it moves is the camera moving, "
            "not the scene changing. Use it for the scene layout, object locations and where things end up, working "
            "out from each frame where it looks")


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
    """(instruction, uploader notes) from an MCAP's text topics, each a list of its messages in time order as
    (log time ns, text), as add_text keeps them. The instruction is the one message of a topic named for the task
    (/task, /instruction) before any sub-topic (/task/subtask, /task/health); MicroAGI's /task titles the fragment
    and its /task/subtask names each step, so the first step is never the task. A task topic whose text changes is
    the instruction whole, as a timeline, with none of its texts chosen over the others. Any other topic with several
    messages goes to the notes as a timeline on the episode's clock (the vendor's steps, claims to check). A topic
    other than a step topic with more than TEXT_MSGS_MAX distinct messages (a heartbeat) keeps only its first, with
    the count."""
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
        if t == it:
            continue                  # the instruction itself, all of it (below)
        if len(seq) == 1:
            notes[t] = seq[0][1]
        elif len(seq) > TEXT_MSGS_MAX and not _is_step(t):
            notes[t] = f"{_title_of(seq[0][1])} (the first of {counts.get(t, len(seq))} messages)"
        else:
            notes[t] = [f"{max(0.0, (ts - base) / 1e9):.1f} s: {_title_of(x)}" for ts, x in seq]
    if it is None:
        return None, notes
    if len(texts[it]) > 1:
        # a task topic whose text changes is given whole, each text with the time it was sent, and no one of them is
        # chosen: MicroAGI sends a placeholder ("The agent is idle") at the start of some fragments, the title after
        # it, and a new title when the work changes
        return "; ".join(f"{max(0.0, (ts - base) / 1e9):.1f} s: {_title_of(x)}" for ts, x in texts[it]), notes
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
    t = next((t for t in texts if _is_step(t) and texts[t] and all(isinstance(x, str) for _, x in texts[t])), None)
    if t is None:
        return None, []
    base = t0 if t0 is not None else texts[t][0][0]
    starts = [max(0.0, (ts - base) / 1e9) for ts, _ in texts[t]]
    ends = starts[1:] + [max(starts[-1], end_s if end_s is not None else starts[-1])]
    return t, [{"t0": round(a, 3), "t1": round(b, 3), "label": _title_of(x)}
               for a, b, (_, x) in zip(starts, ends, texts[t])]


def convert_mcap_generic(item: dict, rig: str, ep: Path, dataset: str) -> dict:
    """Every compressed-image / compressed-video channel that carries a colour camera is a candidate camera;
    they are assigned as assign_views does (a head camera gets one). A text or annotation channel becomes the task text or the
    uploader's notes. Channels of arm joints become the recorded state on a teleoperated rig (joint_state)."""
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
    # depth image channels, each with the camera whose topic it shares the most of (depth_partner), one per camera
    depth_of = {}
    for t, sname in sorted(chan_topics):
        if CAMERA_SCHEMA.search(sname) and DEPTH_TOPIC.search(t) and t not in vmap.values():
            v = depth_partner(t, vmap)
            if v is not None and v not in depth_of.values():
                depth_of[t] = v
            else:
                unused.append(f"{t} (depth with no camera of its own)")
    want = set(vmap.values()) | set(text_topics) | set(depth_of)
    view_of_topic = {t: v for v, t in vmap.items()}
    dwriters: dict[str, DepthWriter] = {}
    writers: dict[str, FrameWriter] = {}
    texts: dict[str, list] = {}               # topic: its messages in time order, (log time ns, text) (add_text)
    n_text: dict[str, int] = {}
    facs, decs, undecodable, t0 = _decoders(), {}, set(), None
    ep.mkdir(parents=True, exist_ok=True)
    with open(item["file"], "rb") as fh:
        try:
            msgs = make_reader(fh).iter_messages(topics=sorted(want), log_time_order=True) if summ is not None \
                else _mcap_stream(item["file"], want)
            for schema, ch, msg in msgs:
                if ch.topic in texts and len(texts[ch.topic]) > TEXT_MSGS_MAX and not _is_step(ch.topic):
                    n_text[ch.topic] += 1
                    continue
                if ch.id not in decs:
                    decs[ch.id] = _decoder_for(ch.message_encoding, schema, facs)
                if decs[ch.id] is None:
                    undecodable.add(f"{ch.topic} ({ch.message_encoding})")
                    continue
                dec = decs[ch.id](msg.data)
                if ch.topic in depth_of:
                    if t0 is None:
                        continue                  # depth before the first colour frame has no frame to go with
                    got = depth_image(dec)
                    if got is not None:
                        dw = dwriters.get(ch.topic)
                        if dw is None:
                            dw = dwriters[ch.topic] = DepthWriter(ep / f"depth_{depth_of[ch.topic]}.mkv")
                        dw.add((int(msg.log_time) - t0) / 1e9, got[0], got[1])
                    continue
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
                    add_text(texts, n_text, ch.topic, int(msg.log_time),
                             d if isinstance(d, str) else (dec if isinstance(dec, dict) else str(dec)))
        except Exception:
            if not writers:
                raise
            # a damaged tail (recording cut off): keep every frame read before it
            item.setdefault("notes", []).append("The file ends early; every frame before the cut was used.")
    counts = {t: w.close() for t, w in writers.items()}
    depth = {}
    for t, dw in dwriters.items():
        if dw.close() and depth_of[t] in vmap:
            depth[depth_of[t]] = {"path": ep / f"depth_{depth_of[t]}.mkv", "real": None, "scale_m": dw.scale_m,
                                  "source": t}
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
    extra = {"task_label": [item["name"]], "source": {"format": "mcap (cameras and text channels)", "file": item["name"],
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
        set_uploader_notes(extra, {t: x for t, x in notes.items()})
    if t0 is not None:
        extra["clock_origin_s"] = t0 / 1e9   # the recording's log time at the clips' zero
    # the arms' joints, on the cameras' clock, when the file records them (joint_state)
    state = action = note = None
    prs = {v: probe(p) for v, (_, p) in files.items()}
    from label import episode as me
    pr = prs[me.order_views(files)[0]]
    q = t0 / 1e9 + pr["pts"].astype(np.float64) * float(pr["time_base"])          # its frames were written from t0
    used = {}
    if rig == "teleop_arms":
        streams = mcap_joint_streams([item["file"]])
        state, action, note = joint_state(streams, q)
        used = state_fields(streams, state, action)
    # every other number the file records, under its own name (mcap_signals)
    signals = mcap_signals([item["file"]], q, used)
    # motion the file records but this reader does not use (hand, body or camera poses in a human recording, joints
    # in a layout the checks do not read): named on the job page, so a missing check is never a silent gap
    motion = [t for t in item["topics"] if re.search(r"hand|pose|slam|body|joint|odom|/tf$", t, re.I)
              and not re.search(r"health|info|meta|static|image|mask", t, re.I)]
    shown_topics = {n.split(" ", 1)[0] for n in signals}
    motion = [t for t in motion if t not in shown_topics]            # kept as signals, so shown to the model
    if state is not None:
        extra["source"]["state"] = "joint channels"
    elif note:
        extra["state_note"] = note
    elif item["seconds"] is None and mcap_layout(item["topics"]) != "generic":
        extra["state_note"] = ("Labelled from the cameras, because the file ends before the index its robot state is read "
                               "from.")
    elif motion and rig == "ego_head":
        shown = ", ".join(motion[:4]) + (f" and {len(motion) - 4} more" if len(motion) > 4 else "")
        extra["state_note"] = (f"Labelled from the camera. The hand, body and camera tracks the file records ({shown}) are "
                               "not read yet.")
    elif motion:
        shown = ", ".join(motion[:4]) + (f" and {len(motion) - 4} more" if len(motion) > 4 else "")
        extra["state_note"] = (f"Labelled from the cameras. The checks on recorded motion read six joints and a gripper per "
                               f"arm, so they did not run on this file's motion channels ({shown}).")
    elif rig != "ego_head":
        extra["state_note"] = "Labelled from the cameras, because the file records no robot state."
    if item.get("notes"):
        extra["source"]["notes"] = item["notes"]
    if item.get("fixed_window_s"):
        extra["collection_note"] = packaging_note(item["fixed_window_s"])
        extra["packaging"] = {"fixed_window_s": item["fixed_window_s"]}
    return video_views_episode(ep, files, rig, dataset, extra, shared_clock=True, prs=prs, state=state, action=action,
                               signals=signals, depth={v: d for v, d in depth.items() if v in files})


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
        self.held = []                    # encoded packets waiting for the next frame's time (_mux)

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
        self.pts.append(p)
        self._mux(self.enc.encode(fr))

    def _mux(self, packets, last: bool = False) -> None:
        """Encoded packets into the mp4 in the order the encoder gives them, each lasting the step to the next frame's
        time and the last frame the step before it, as copied frames do (close): the encoder gives every frame 1/30 s
        at any rate, which ended a 15 fps camera's file half a frame early. A packet waits until the frame after it
        has come in."""
        import bisect
        self.held += list(packets)
        while self.held:
            pkt = self.held[0]
            i = min(bisect.bisect_left(self.pts, int(pkt.pts)), len(self.pts) - 1)
            if i + 1 < len(self.pts):
                pkt.duration = self.pts[i + 1] - self.pts[i]
            elif last:
                pkt.duration = self.pts[i] - self.pts[i - 1] if i else TIME_BASE_DEN // 30
            else:
                return
            self.dst.mux(self.held.pop(0))

    def close(self) -> int:
        """Finish the file; returns the number of frames written."""
        import av
        if self.kind == "image" and self.enc is not None:
            self._mux(self.enc.encode(), last=True)
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
    if det["format"] == "hdf5":
        items = plan_hdf5(det, root)
        nf = len(det["files"])
        det["used"].append(f"{nf} HDF5 file{'s' if nf != 1 else ''}, "
                           f"{len(items)} episode{'s' if len(items) != 1 else ''}.")
        _mark_packaging(det, items)
        return det, items
    items = plan_video(det, root, grouping)
    det["used"].append(f"{len(items)} video episodes." if len(items) != 1
                       else "1 video episode.")
    _mark_packaging(det, items)
    return det, items


TABLE_EXT = {".csv", ".tsv", ".jsonl"}
TABLE_MAX_BYTES = 20_000_000
TABLE_MAX_ROWS_PER_EPISODE = 20


def annotation_tables(root: Path) -> list[tuple[str, list[dict]]]:
    """[(file name, rows)] of every table in the upload (CSV, TSV, JSON Lines, at most TABLE_MAX_BYTES each): a
    dataset's per-episode metadata kept beside its data (OpenTouch's final_annotations/eat_ygf_p1_merged.csv, one row
    per clip with its object, action, grip and description)."""
    import csv
    out = []
    for p in files_under(root):
        if p.suffix.lower() not in TABLE_EXT or p.stat().st_size > TABLE_MAX_BYTES:
            continue
        try:
            if p.suffix.lower() == ".jsonl":
                rows = [r for r in read_jsonl(p) if isinstance(r, dict)]
            else:
                with open(p, newline="", errors="replace") as fh:
                    rows = list(csv.DictReader(fh, delimiter="\t" if p.suffix.lower() == ".tsv" else ","))
        except Exception:
            continue
        if rows:
            out.append((p.relative_to(root).as_posix(), rows))
    return out


def table_rows_for(tables: list, item: dict) -> list[dict]:
    """The rows of the upload's tables that name this episode: a cell holding its file's name and, for an episode that
    is one group of a file, that group's name too (eat_ygf_p1::demo_00 names demo_00 of eat_ygf_p1.hdf5), as a whole
    word. Rows that name every episode alike (a file shared by all) are not one episode's."""
    parts = [x for x in str(item["name"]).split("/") if x]
    if not parts:
        return []
    need = [parts[-1]] if len(parts) == 1 else [parts[-2], parts[-1]]
    if item.get("kind") == "hdf5" and item.get("group"):
        need = [Path(item["file"]).stem, item["group"].rsplit("/", 1)[-1]]
    pats = [re.compile(r"(^|[^A-Za-z0-9])" + re.escape(w) + r"($|[^A-Za-z0-9])") for w in need]
    got = []
    for name, rows in tables:
        for r in rows:
            cells = [str(v) for v in r.values() if isinstance(v, (str, int, float)) and str(v)]
            if all(any(pt.search(c) for c in cells) for pt in pats):
                got.append({"table": name, **{k: v for k, v in r.items() if v not in (None, "")}})
    return got[:TABLE_MAX_ROWS_PER_EPISODE] if 0 < len(got) <= TABLE_MAX_ROWS_PER_EPISODE * 5 else []


def add_table_notes(ep: Path, ctx: dict, rows: list[dict]) -> dict:
    """The table rows that name an episode, added to its uploader notes (claims the model checks against the video),
    and a task text among them (a column named for the task) as its instruction when it has none."""
    if not rows:
        return ctx
    prev = ctx.get("uploader_notes", ctx.get("uploader_annotation"))
    if isinstance(prev, str):
        try:
            prev = json.loads(prev)                        # notes already written as JSON stay one object, not a string
        except ValueError:
            pass
    set_uploader_notes(ctx, {"notes": prev, "table rows": rows} if prev else {"table rows": rows})
    if not ctx.get("instruction"):
        task = next((str(v).strip() for r in rows for k, v in r.items() if TASK_KEY.search(str(k))
                     and isinstance(v, str) and 0 < len(v.strip()) < 400), None)
        if task:
            ctx["instruction"] = task
            ctx["instruction_note"] = "This instruction is the task text in a table the uploader sent with the episode."
            (ep / "instruction.txt").write_text(task + "\n")
    ctx.setdefault("source", {})["tables"] = sorted({r["table"] for r in rows})
    (ep / "context.json").write_text(json.dumps(ctx, indent=1, default=str))
    return ctx


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
    tables = annotation_tables(root)
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
                ctx = convert_lerobot_item(it, rig, out, dataset)
            elif it["kind"] == "recording":
                ctx = convert_recording(it, rig, out, dataset)
            elif it["kind"] == "mcap":
                ctx = convert_mcap(it, rig, out, dataset)
            elif it["kind"] == "hdf5":
                ctx = convert_hdf5(it, rig, out, dataset)
            else:
                ctx = convert_video(it, rig, out, dataset)
        except Exception as e:
            report["failed"].append({"name": it["name"], "why": plain_error(e)})
            print(f"convert: {it['name']}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        if tables:
            ctx = add_table_notes(out / ctx["episode_id"], ctx, table_rows_for(tables, it))
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
    ids = [e["episode_id"] for e in report["episodes"]]
    measure_gripper_range(out, ids)
    measure_depth_ranges(out, ids)
    measure_signal_scales(out, ids)
    measure_contacts(out, ids)
    return report


def measure_contacts(out: Path, ids: list[str]) -> int:
    """Each episode's contacts (label/contacts.py), found with the upload's signal scales and written as
    context["contacts"], so the labeller, the checks and the board read one list. Returns how many were found."""
    from label import contacts as lc
    from label import episode as me
    total = 0
    for i in ids:
        d = Path(out) / i
        if not (d / "signals.npz").exists():
            continue
        try:
            ep = me.load(d)
            sig = ep.get("signals") or {}
            n = len(next(iter(sig.values()))) if sig else 0
            found = lc.find(sig, ep.get("signal_meta") or {}, np.array([me.frame_time(ep, k) for k in range(n)]))
        except Exception as e:
            print(f"contacts: {i}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        ctx = json.loads((d / "context.json").read_text())
        ctx["contacts"] = found
        (d / "context.json").write_text(json.dumps(ctx, indent=1, default=str))
        total += len(found)
    return total


DEPTH_RANGE_FRAMES = 6        # depth frames per episode read to measure a camera's range across the upload


def measure_depth_ranges(out: Path, ids: list[str]) -> dict:
    """For each camera whose depth has no stated unit, the range of its readings across every episode of the upload (the
    2nd to 98th percentile, label/depth.scale_range), written as "range" into each episode's depth.json, so a shade of
    grey is the same reading in every episode (label/depth.py). Metric depth needs none: it has one fixed scale."""
    from label import depth as dp
    from label import episode as me
    frames, where = {}, {}
    for i in ids:
        d = Path(out) / i
        if not (d / "depth.json").exists():
            continue
        try:
            ep = me.load(d)
        except Exception:
            continue
        for v, e in (ep.get("depth") or {}).items():
            if e.get("scale_m") and e.get("kind") != "disparity":
                continue
            key = (ep["sources"][v].get("camera_key") or v)
            n = int(ep["sources"][me.anchor(ep)]["n_frames"])
            ks = sorted({int(x) for x in np.linspace(0, max(n - 1, 0), DEPTH_RANGE_FRAMES)})
            frames.setdefault(key, []).extend(dp.at_anchor(ep, ep["depth"], v, ks).values())
            where.setdefault(key, []).append((d, v))
    ranges = {}
    for key, fs in frames.items():
        r = dp.scale_range(fs)
        if r is None:
            continue
        ranges[key] = [round(r[0], 6), round(r[1], 6)]
        for d, v in where[key]:
            dj = json.loads((d / "depth.json").read_text())
            dj[v]["range"] = ranges[key]
            dj[v]["range_note"] = f"measured across the {len(where[key])} episodes of this upload"
            (d / "depth.json").write_text(json.dumps(dj, indent=1))
    return ranges


SIGNAL_SCALE_ROWS = 200_000   # rows per signal read across the upload to measure its rest and swing


def measure_signal_scales(out: Path, ids: list[str]) -> dict:
    """For each signal (by name), its resting level per value and its typical swing measured across every episode of the
    upload (label/signals.py), written into each episode's context.json signal entry as "rest" and "swing", so the
    activity of a glove's pressure, and the colour of its map on the board, mean the same raw distance from rest in
    every episode: a light touch in one episode never looks like a hard grip in another."""
    from label import signals as sg
    rows, where = {}, {}
    for i in ids:
        d = Path(out) / i
        p = d / "context.json"
        if not (d / "signals.npz").exists() or not p.exists():
            continue
        ctx = json.loads(p.read_text())
        with np.load(d / "signals.npz") as z:
            for s in ctx.get("signals") or []:
                if s["key"] in z.files:
                    rows.setdefault(s["name"], []).append(np.asarray(z[s["key"]], dtype=np.float64))
                    where.setdefault(s["name"], []).append(d)
    scales = {}
    for name, arrs in rows.items():
        if len({a.shape[1] for a in arrs}) != 1:
            continue                       # one name with different widths in different episodes: no shared scale
        a = np.concatenate(arrs)
        if len(a) > SIGNAL_SCALE_ROWS:
            a = a[:: int(np.ceil(len(a) / SIGNAL_SCALE_ROWS))]
        rest = sg.resting_level(a)
        _, swing = sg._distance(a, rest)
        scales[name] = {"rest": [round(float(x), 6) for x in rest], "swing": round(float(swing), 6),
                        "episodes": len(arrs)}
    for name, sc in scales.items():
        for d in set(where[name]):
            p = d / "context.json"
            ctx = json.loads(p.read_text())
            for s in ctx.get("signals") or []:
                if s["name"] == name:
                    s.update(rest=sc["rest"], swing=sc["swing"], scale_note=f"measured across the {sc['episodes']} "
                                                                             "episodes of this upload")
            p.write_text(json.dumps(ctx, indent=1, default=str))
    return scales


def measure_gripper_range(out: Path, ids: list[str]) -> list | None:
    """The gripper's full range [shut, open] in the upload's own unit, measured across all its episodes (the 0.5th
    to 99.5th percentile of every gripper reading, so a stray sample does not stretch it), written as
    "gripper_range" into each episode with recorded state whose reader did not declare one. The still-span test
    (label/state.py) takes 1% of it as the gripper's tolerance, so a gripper recorded 0 to 100 or in metres is judged
    as one recorded 0 to 1 is. One episode's own range would not do: a gripper that never moves in it has a range of
    only its noise. Returns the range, or None when no episode has state."""
    eps = [out / i for i in ids if (out / i / "state.npz").exists()]
    ctxs = {d: json.loads((d / "context.json").read_text()) for d in eps}
    todo = [d for d, c in ctxs.items() if c.get("state_kind") in ("joints", "ee_pose") and "gripper_range" not in c]
    if not todo:
        return None
    vals = []
    for d in todo:
        with np.load(d / "state.npz") as z:
            st = z["state"]
        if st.ndim == 2 and st.shape[1] % 7 == 0:
            vals.append(np.asarray(st[:, 6::7], dtype=np.float64).ravel())
    if not vals:
        return None
    v = np.concatenate(vals)
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    rng = [round(float(np.percentile(v, 0.5)), 6), round(float(np.percentile(v, 99.5)), 6)]
    for d in todo:
        c = ctxs[d]
        c["gripper_range"] = rng
        c["gripper_range_note"] = f"measured across the {len(todo)} episodes of this upload"
        (d / "context.json").write_text(json.dumps(c, indent=1, default=str))
    return rng

