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

An upload that holds several of these (plain videos beside an HDF5 file with a camera, a LeRobot dataset beside loose
MCAP recordings) is read in every one of them (detect), and the files no reader opens are named in the report. MCAP and
HDF5 files with no camera go with the episodes of their folder, whatever format those are (assign_sensors).

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

No drop. When part of an upload is imperfect (a signal with a bad value or a gap, a table shorter than the video, a
camera that does not decode, a sensor placed on the video by an assumed common start), the usable rest is kept and the
problem is recorded on the episode, never used as a reason to drop it. context.json's reader_issues is that record, a
list of {"kind": a short snake_case tag (signal_not_finite, signal_partial_span, signal_gap, table_short,
camera_not_decodable, ...), "what": one plain sentence a reviewer reads on the board, and when they apply "camera",
"signal", "t0_s" and "t1_s" (seconds of the episode)}. add_issue appends one entry and never overwrites the others;
the board shows them as the episode's data issues. What could not be used at all is still listed with its reason
(context["source"] unused_*), and an episode with nothing to label is listed in the report with why. A signal placed
from both starts, because the reader had no clock in common to place it by, also carries "aligned_by": "assumed start"
in its context.json signal entry (mark_assumed), and a table placed one row per frame "aligned_by": "row per frame"
(ALIGNED_ROWS), which the prompt's signal line states; neither is ever read as the arm state. A signal past the
episode's SIGNAL_EPISODE_BYTES is kept as its lowest, mean and highest value at each frame, with "summary_of" giving
its width.

Every camera reaches the board. A camera the model is not shown (more extra cameras than MAX_EXTRA_CAMERAS, the second
eye of a stereo camera, every camera but one on a head rig, an infrared, thermal or mask video) is listed in
context.json's unshown_cameras, [{"name", "why" (the reason the model is not shown it), "packed" (its video file),
"base_s" (the episode's offset in it), "n_frames", "start_s" (its first frame on the episode's clock), "fps"}], so the
board plays it named as not shown to the model; an MCAP or HDF5 camera is written to a video of its own for it.
"""
from __future__ import annotations

import functools
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


# ---------------------------------------------------------------- data issues

def add_issue(ctx: dict, kind: str, what: str, camera: str | None = None, signal: str | None = None,
              t0_s: float | None = None, t1_s: float | None = None) -> dict:
    """One entry appended to ctx["reader_issues"] (the module docstring's No drop): kind, a short snake_case tag; what,
    one plain sentence for the board; the camera or signal it is about and its time span in seconds of the episode
    when known. An entry already there is not added twice, so a reader that writes an episode's context twice
    (finish_episode after video_views_episode) records each problem once."""
    entry = {"kind": str(kind), "what": str(what)}
    for k, v in (("camera", camera), ("signal", signal)):
        if v is not None:
            entry[k] = str(v)
    for k, v in (("t0_s", t0_s), ("t1_s", t1_s)):
        if v is not None and np.isfinite(v):
            entry[k] = round(float(v), 3)
    issues = ctx.setdefault("reader_issues", [])
    if entry not in issues:
        issues.append(entry)
    return entry


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


class UnpackLimit(ValueError):
    """An upload that unpacks to more than one upload may hold: it stops the unpacking, never only one member."""


def _archive_members(path: Path):
    """([(parts, size, open)] for every regular file in the archive, how many members were left out, the open archive,
    the members that cannot be read, each named with why). Links, devices, folders, hidden files and __MACOSX are
    never unpacked. A password-protected member or one in a compression Python does not open is left out by itself,
    and a tar cut short keeps every member whose header comes before the cut, as the upload page does: one bad member
    had cost the whole archive."""
    import stat
    import tarfile
    import zipfile
    out, skipped, bad = [], 0, []
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
                bad.append(f"{zip_name(info)} (password-protected)")
                continue
            if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA):
                # Deflate64 (Windows Explorer, for large files) and others; the upload page decodes Deflate64 itself
                method = ZIP_METHODS.get(info.compress_type, f"method {info.compress_type}")
                bad.append(f"{zip_name(info)} ({method} compression, which Python's zip reader does not open; unzip "
                           "it and give the folder)")
                continue
            parts = _member_parts(zip_name(info))
            if parts is None:
                skipped += 1
                continue
            out.append((parts, info.file_size, lambda i=info: zf.open(i)))
        return out, skipped, zf, bad
    tf = tarfile.open(path, "r:*")
    members = []
    try:
        for m in tf:
            members.append(m)
    except Exception:
        # cut short: every member whose header came before the cut is listed, and one whose bytes the cut reaches
        # fails as it is unpacked (open_archives)
        bad.append(f"{path.name} is cut short, so only the files before the cut were unpacked")
    for m in members:
        if m.isdir():
            continue
        if m.islnk() or m.issym():
            # a link is the member it points to when that member is in the archive (tarfile resolves it among the
            # archive's own members only, never on disk); read.js resolveLinks, the same rule
            try:
                target = tf._find_link_target(m)
            except Exception:                   # not in the archive, or after the cut of one cut short
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
    return out, skipped, tf, bad


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
            members, skipped, handle, bad = _archive_members(a)
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
                    raise UnpackLimit(f"{a.name} holds more than one upload can: at most {UNPACK_MAX_FILES:,} files "
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
                got = 0
                try:
                    with opener() as src, open(tmp, "wb") as dst:
                        while True:
                            buf = src.read(1 << 20)
                            if not buf:
                                break
                            got += len(buf)
                            if total + got > UNPACK_MAX_BYTES:
                                dst.close()
                                tmp.unlink()
                                raise UnpackLimit(f"{a.name} unpacks to more than {UNPACK_MAX_BYTES / 1e9:.0f} GB")
                            dst.write(buf)
                except UnpackLimit:
                    raise
                except Exception:
                    # a member cut short or damaged (a tar cut inside it, a zip member that fails its check) is left
                    # out by itself and named; the members before and after it are unpacked
                    tmp.unlink(missing_ok=True)
                    bad.append(f"{'/'.join(parts)} (cut short or damaged)")
                    continue
                total += got
                tmp.replace(target)
                count += 1
                wrote += 1
        done.write_text("")
        cut = [b for b in bad if b.startswith(a.name + " is cut short")]
        members_bad = [b for b in bad if b not in cut]
        notes.append(f"Opened {a.name}: {wrote} files" + (f"; {skipped} left out (links, or names that could leave "
                                                            "the archive's folder)" if skipped else "")
                     + (f"; {len(members_bad)} left out because {'they' if len(members_bad) != 1 else 'it'} could "
                        f"not be unpacked: {_and_words(members_bad[:12])}"
                        + (f" and {len(members_bad) - 12} more" if len(members_bad) > 12 else "")
                        if members_bad else "") + "." + "".join(f" {c}." for c in cut))
    if root.is_file():
        return Path(dest), notes
    return root, notes


# ---------------------------------------------------------------- detection

def detect(root: Path) -> dict:
    """{"format": "lerobot" | "mcap" | "hdf5" | "video", ...} for an upload folder, or {"format": "mixed", "parts":
    [one such dict per format]} when it holds several, or raises with what was found. Every format present is read:
    an upload of plain videos beside an HDF5 file with a camera had been read as HDF5 alone, its videos never
    mentioned. Files inside a LeRobot dataset belong to it. MCAP and HDF5 files with no camera (arm joints, grippers, a
    glove's pressure and hand pose) are sensor files ("state"), read with the episodes of their folder (assign_sensors),
    never episodes of their own; an MCAP a dataset adapter recognizes is an episode whatever its channels."""
    root = Path(root)
    files = files_under(root)
    roots = lerobot_roots(root, files)
    parts = [{"format": "lerobot", "roots": roots}] if roots else []
    rdirs = [Path(r) for r in roots]
    rest = [p for p in files if not any(r in p.parents for r in rdirs)]
    mcaps = [p for p in rest if p.suffix.lower() == ".mcap"]
    h5s = [p for p in rest if p.suffix.lower() in H5_EXT]
    vids = [p for p in rest if p.suffix.lower() in VIDEO_EXT]
    mcap_eps = [p for p in mcaps if mcap_has_camera(p) or mcap_layout([t for t, _ in mcap_channels(p)]) != "generic"]
    h5_cams = [p for p in h5s if h5_has_camera(p)]
    sensors = [str(p) for p in mcaps + h5s if p not in mcap_eps and p not in h5_cams]
    if mcap_eps:
        parts.append({"format": "mcap", "files": [str(p) for p in mcap_eps]})
    if h5_cams:
        parts.append({"format": "hdf5", "files": [str(p) for p in h5_cams]})
    if vids:
        parts.append({"format": "video", "files": [str(p) for p in vids]})
    for part in parts:
        if part["format"] != "lerobot" and sensors:
            part["state"] = sensors
    if len(parts) == 1:
        return parts[0]
    if parts:
        return {"format": "mixed", "parts": parts, "state": sensors}
    seen = sorted({p.suffix.lower() or p.name for p in files})[:12]
    if sensors:
        # the board shows an episode beside its footage, so recorded data with no camera cannot be shown or labelled
        # yet: the refusal names every file and why, so the uploader knows what was found
        raise ValueError(no_camera_words(root, files, [Path(x) for x in sensors]))
    raise ValueError("the upload holds no LeRobot dataset, MCAP file, HDF5 file or video"
                     + (f" (only {', '.join(seen)} files)" if seen else " (it is empty)"))


NO_CAMERA_FILES_MAX = 200


def no_camera_words(root: Path, files: list[Path], sensors: list[Path]) -> str:
    """Why an upload of recorded data with no camera is refused: every file counted by kind, and each named with why it
    gives nothing to label (up to NO_CAMERA_FILES_MAX names)."""
    def why(p):
        if p in sensors:
            return ("an HDF5 file with no image frames or encoded images" if p.suffix.lower() in H5_EXT
                    else "an MCAP file with no camera channel")
        return "not a recording" if p.suffix.lower() in NOTE_EXT else "not a kind of file a reader opens"
    named = [f"{p.relative_to(root).as_posix()} ({why(p)})" for p in files[:NO_CAMERA_FILES_MAX]]
    kinds = {"HDF5 file": 0, "MCAP file": 0, "other file": 0}
    for p in files:
        kinds["HDF5 file" if p.suffix.lower() in H5_EXT else "MCAP file" if p.suffix.lower() == ".mcap"
              else "other file"] += 1
    count = ", ".join(f"{k_} {name}{'s' if k_ != 1 else ''}" for name, k_ in kinds.items() if k_)
    more = f"; and {len(files) - NO_CAMERA_FILES_MAX} more" if len(files) > NO_CAMERA_FILES_MAX else ""
    return ("the upload holds recorded data but no camera, and an episode is shown and labelled beside its video, so "
            f"nothing could be labelled. It holds {len(files)} file{'s' if len(files) != 1 else ''} ({count}): "
            + "; ".join(named) + more)


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


def unshown_why(name: str, rig: str, shown: list[str]) -> str:
    """Why the model is not shown a camera pick_cameras left unused, in words for the board."""
    if not_rgb(name):
        return "not a colour camera (depth, infrared, thermal or a mask), so the model is not shown it"
    if rig == "ego_head":
        return "the model is shown one head camera"
    if any(_stereo_twin(name, c) for c in shown):
        return "the other eye of a stereo camera the model is shown"
    return f"beyond the {MAX_EXTRA_CAMERAS} extra cameras the model is shown"


def unshown_entry(name: str, path: Path, why: str, base_s: float = 0.0, n_frames: int | None = None,
                  start_s: float = 0.0, fps: float | None = None) -> dict | None:
    """One camera the model is not shown, as context.json's unshown_cameras holds it (the module docstring); its frame
    count and rate from the file when not given. None when the file cannot be read."""
    if n_frames is None or fps is None:
        try:
            pr = probe(Path(path))
        except Exception:
            return None
        n_frames = len(pr["pts"]) if n_frames is None else n_frames
        fps = fps or measured_fps(seconds(pr)) or pr["fps"]
    if not n_frames:
        return None
    return {"name": str(name), "why": why, "packed": str(Path(path).resolve()), "base_s": round(float(base_s), 6),
            "n_frames": int(n_frames), "start_s": round(float(start_s), 6),
            "fps": round(float(fps), 3) if fps else None}


def _stereo_twin(a: str, b: str) -> bool:
    """True when two camera names differ only by a left/right word (the two eyes of one stereo camera)."""
    swap = lambda s: re.sub(r"left|right", lambda m: {"left": "right", "right": "left"}[m.group(0).lower()], s,
                            flags=re.I)
    return a != b and swap(a).lower() == b.lower()


# a camera whose name says it is not a colour picture: depth, confidence, disparity, a mask or segmentation, thermal,
# infrared, or a visualisation. Matched by whole words (not_rgb): the pattern it replaced matched inside words, so
# segway_cam, a conference room and visual_top were taken for masks and confidence maps. A RealSense infra1 stream is
# a grey picture of the scene the model has been shown as a camera, and stays one
NOT_RGB_WORDS = ("depth", "conf", "confidence", "disparity", "mask", "seg", "segmentation", "thermal", "infrared",
                 "ir", "vis")


def not_rgb(name: str) -> bool:
    """Whether a camera's name says it is not a colour picture (NOT_RGB_WORDS), by whole words (_names_word)."""
    return _names_word(name, NOT_RGB_WORDS)


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
    rgb = [t for t in topics if not not_rgb(t)] or list(topics)
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
STATE_PART_POSE_NOTE = ("Labelled from the video: the recorded state's value names give a position for some values "
                        "({}) but not a pose our checks read, and our checks read six joints and a gripper per arm or "
                        "a 6D pose and an opening per gripper.")
STATE_UNAXED_NOTE = ("Labelled from the video: the recorded state's value names give a position or a pose without the "
                     "axes our checks read ({}), not six joints and a gripper per arm.")


def state_words(name: str) -> list[str]:
    """The words of a state value's name for state_layout: split at . _ / space and hyphen, at a camelCase boundary
    and before a number, lowercased, with a unit word after the last word dropped (x_m is x, eefPosX is eef pos x)."""
    words = [w.lower() for w in STATE_WORD_SPLIT.split(str(name)) if w]
    return words[:-1] if len(words) > 1 and words[-1] in STATE_UNIT_WORDS else words


# Why an episode has no arm state, in one word, written to context.json "state_why" beside its state_note (no_state):
# the note says it in full for a reviewer, and labelling chooses its one line on the recorded state from this word, so
# that line is true for the episode.
STATE_WHY = {
    "layout": "a state is recorded, but not in a layout our checks read: its width, its value names, which arm is "
              "which, or rows that cannot be lined up with the frames",
    "not_recorded": "the recording holds no state at all",
    "unreadable": "a file that holds the state, or may hold it, could not be read, or was damaged before any of its "
                  "messages",
    "short": "an arm's state does not cover the footage: it starts late, stops early or stops inside it",
    "assumed_clock": "the state's channels are only on a clock placed on the footage from both starts, an alignment "
                     "that is assumed",
}


class StateNote(str):
    """A state note, the sentence that says why an episode has no arm state, carrying that reason in one word (why,
    one of STATE_WHY), so the reason travels with the sentence from the rule that decided it to no_state."""

    def __new__(cls, text: str, why: str):
        if why not in STATE_WHY:
            raise ValueError(f"{why!r} is not one of {sorted(STATE_WHY)}")
        note = super().__new__(cls, text)
        note.why = why
        return note


def no_state(ctx: dict, note: StateNote) -> None:
    """An episode with no arm state: its state_note, and its state_why (STATE_WHY), always written together."""
    ctx["state_note"] = str(note)
    ctx["state_why"] = note.why


def drop_no_state(ctx: dict) -> None:
    """An episode whose arm state was read after all: neither its state_note nor its state_why stays."""
    ctx.pop("state_note", None)
    ctx.pop("state_why", None)


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
    # a name with no words ("" or "_") has no last word, so it says neither an axis, a joint nor a gripper
    last = {x: words[x][-1] if words[x] else "" for x in names}
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
    # a pose whose axes the rule cannot read is never joints by width on an arm rig, and is the pose on a gripper rig
    framed = next((x for x in names if set(words[x]) & STATE_FRAME_WORDS and last[x].isdigit()
                   and not STATE_GRIPPER_NAME.search(x)), None)
    unaxed = framed is not None and kind != "ee_pose"
    seventh = all(STATE_GRIPPER_NAME.search(g[6]) for g in groups)
    if seventh and not any(STATE_GRIPPER_NAME.search(x) for g in groups for x in g[:6]):
        if all(last[x] in STATE_AXIS_WORDS for g in groups for x in g[:6]) and \
                all(any(last[x] in STATE_POSITION_AXES for x in g[:6]) for g in groups):
            return "ee_pose", None
        if unaxed:
            return "none", STATE_UNAXED_NOTE.format(framed)
        if all(STATE_JOINT_NAME.search(x) for g in groups for x in g[:6]):
            return "joints", None
        # a name of a position axis (x, y or z) in a group that is no full pose is never read by width
        placed = next((x for g in groups for x in g[:6] if last[x] in STATE_POSITION_AXES), None)
        if placed:
            return "none", STATE_PART_POSE_NOTE.format(placed)
        return kind, None
    if any(STATE_GRIPPER_NAME.search(x) for x in names):
        return "none", ("Labelled from the video: the recorded state's value names put a gripper elsewhere than "
                        "seventh in each group of seven, and our checks read six values and then the gripper.")
    if unaxed:
        return "none", STATE_UNAXED_NOTE.format(framed)
    if all(STATE_JOINT_NAME.search(x) for x in names):
        return "none", (f"Labelled from the video: the recorded state's value names give {dims} joints and no gripper, "
                        "and our checks read six joints and a gripper per arm.")
    placed = next((x for g in groups for x in g[:6] if last[x] in STATE_POSITION_AXES), None)
    if placed:
        return "none", STATE_PART_POSE_NOTE.format(placed)
    return kind, None


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
# An array wider than SIGNAL_MAX_VALUES is still kept, as a map the prompt summarises (label/signals.py), read and
# placed in float32 so it is never copied as float64. What is bounded is the signals a reader keeps: as they are read,
# a signal that would take the running total of an episode's kept signals (as float32) past SIGNAL_EPISODE_BYTES is
# kept as a summary per frame (summarise_rows), a data issue (Signals.add, merge_signals), and an HDF5 array that
# would is summarised a block of rows at a time without being read whole (h5_signals); write_signals applies the same
# bound once more to what it writes. Not bounded: the source a reader already holds to read it (a LeRobot table, an
# MCAP channel's messages before they are placed), nor one signal's own size while it is placed (a few times its
# float32 size at most).
SIGNAL_EPISODE_BYTES = 1_000_000_000
SUMMARY_NAMES = ["lowest", "mean", "highest"]
SUMMARY_CHUNK_ROWS = 256


def float_rows(a) -> np.ndarray:
    """Rows of numbers as floats without a copy when they already are: float64 for a signal of up to SIGNAL_MAX_VALUES
    values, float32 for a wider one."""
    a = np.asarray(a)
    wide = a.ndim > 1 and int(np.prod(a.shape[1:])) > SIGNAL_MAX_VALUES
    if a.dtype.kind == "f" and (wide or a.dtype == np.float64):
        return a
    return a.astype(np.float32 if wide else np.float64)


def summarise_rows(a) -> np.ndarray:
    """(n, 3) float32: each row's lowest, mean and highest finite value (SUMMARY_NAMES), NaN where a row has none,
    computed a block of rows at a time."""
    n = len(a)
    out = np.full((n, 3), np.nan, dtype=np.float32)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for i in range(0, n, SUMMARY_CHUNK_ROWS):
                b = np.asarray(a[i:i + SUMMARY_CHUNK_ROWS], dtype=np.float32)
                b = b.reshape(len(b), -1)
                out[i:i + len(b)] = np.stack([np.nanmin(b, 1), np.nanmean(b, 1), np.nanmax(b, 1)], axis=1)
    return out


def summary_issue(name: str, values: int) -> dict:
    return {"kind": "signal_summarised", "signal": name,
            "what": f"{name} has {values:,} values per frame, more than an episode's signals hold together, so it is "
                    "kept as its lowest, mean and highest value at each frame"}


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
        self.issues: list[dict] = []                 # problems with what was kept, as add_issue's entries
        self.no_gaps: set[str] = set()               # signals whose frames with no reading are no issue (write_signals)
        self.bytes = 0                               # the kept signals' size as float32, for SIGNAL_EPISODE_BYTES
        self._sizes: dict[str, int] = {}             # what keep counted for each name, given back when it leaves

    def fits(self, rows: int, values: int) -> bool:
        """Whether a signal of rows x values keeps the running total within SIGNAL_EPISODE_BYTES."""
        return self.bytes + int(rows) * int(values) * 4 <= SIGNAL_EPISODE_BYTES

    def keep(self, name: str, a: np.ndarray, m: dict) -> None:
        """a kept under name with its meta m, as its summary per frame (summarise_rows, a data issue) when it would take
        the running total past SIGNAL_EPISODE_BYTES. A signal kept under a name already held replaces it, and gives
        its size back first."""
        self._release(name)
        a = np.asarray(a)
        width = int(a.shape[1]) if a.ndim > 1 else 1
        if not self.fits(len(a), width) and width > len(SUMMARY_NAMES):
            a = summarise_rows(a)
            m = {k: v for k, v in m.items() if k != "shape"}
            m.update(names=list(SUMMARY_NAMES), summary_of=width)
            self.issues.append(summary_issue(name, width))
            width = len(SUMMARY_NAMES)
        self._sizes[name] = len(a) * width * 4
        self.bytes += self._sizes[name]
        self[name] = a
        self.meta[name] = m

    def _release(self, name: str) -> None:
        """The size keep counted for name given back to the running total."""
        self.bytes -= self._sizes.pop(name, 0)

    def pop(self, name, *default):
        """A signal taken out (h5_state reads it as the state) gives its size back to the running total."""
        self._release(name)
        return super().pop(name, *default)

    def __delitem__(self, name) -> None:
        self._release(name)
        super().__delitem__(name)

    def add(self, name: str, a: np.ndarray, shape=None, names=None, source: str | None = None) -> None:
        m = {}
        if shape is not None and tuple(int(x) for x in shape) != (a.shape[1],):
            m["shape"] = [int(x) for x in shape]
        if names is not None and len(names) == a.shape[1]:
            m["names"] = [str(x) for x in names]
        if source:
            m["source"] = source
        self.keep(name, a, m)


def merge_signals(into: Signals, more: Signals) -> Signals:
    """more's signals, notes and clocks added to into (a plain dict of arrays is read as Signals), each counting toward
    into's running total (Signals.keep)."""
    for k, v in more.items():
        into.keep(k, v, dict((getattr(more, "meta", {}) or {}).get(k) or {}))
    into.left_out += list(getattr(more, "left_out", []) or [])
    into.clocks.update(getattr(more, "clocks", {}) or {})
    into.issues += list(getattr(more, "issues", []) or [])
    into.no_gaps |= set(getattr(more, "no_gaps", ()) or ())
    return into


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


def short_rising(values) -> bool:
    """Whether a column too short to be judged a counter (under COUNTER_MIN_MESSAGES readings) never falls and rises."""
    x = np.asarray(values, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    return 2 <= len(x) < COUNTER_MIN_MESSAGES and bool((np.diff(x) >= 0).all()) and x[-1] > x[0]


def is_named_clock(name, values) -> bool:
    """Whether a column is a clock: its name says time (is_time_name) and it rises like one (is_clock), or, with too
    few readings to judge, never falls and rises (a slow sensor's own stamp at 1.7 Hz)."""
    return is_time_name(name) and (is_clock(values) or short_rising(values))


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
    the columns that are not bookkeeping (SIGNAL_SKIP) and not already read (used: the state, the action, the
    cameras), each row on its frame (frame_rows, on_frames), so a table shorter than the video is NaN past its last
    row (a data issue, table_issue) rather than dropped. A reading missing at some frames is still a reading, NaN at
    those frames, however few frames have one (the 2026-10-03 audit found a column dropped for reading at fewer than
    half), and a cell or image that cannot be read is NaN at its frame (_cells_rows, _image_cells, a data issue).
    An array per frame keeps its shape (a pressure map is 16 x 16, not 256 numbers in a row) and the names the dataset
    gives its values (features: meta/info.json's, with "shape" and "names"). The harness shows them to the model as
    they are (label/episode.py), so nothing a dataset records is dropped because our checks do not know what it means:
    a mobile robot's base and torso, joint velocities, forces, a tactile glove's pressure map. A column left out
    (wider than SIGNAL_MAX_VALUES, with no reading at all, a counter) is listed in left_out with the reason."""
    out = Signals()
    if df is None:
        return out
    features = features or {}
    at = frame_rows(df, n)
    if n and len(df):
        table_issue(out, df, at, n)
    for c in df.columns:
        if c in used or SIGNAL_SKIP.search(str(c)) or str(c).startswith("observation.images"):
            continue
        if (features.get(c) or {}).get("dtype") == "video":
            continue
        if (features.get(c) or {}).get("dtype") == "image":
            got = _image_cells(df[c])             # only the small per-pad images reach here (read_root image_signals)
            if got is None:
                out.left_out.append((str(c), "none of its images could be decoded"))
                continue
            a, shape, bad = got
            if bad:
                bad_cells_issue(out, str(c), bad, len(a), "images that could not be decoded")
            out.add(str(c), on_frames(a, at, n), shape=shape, source="image column " + str(c))
            continue
        a, bad = _cells_rows(df[c])
        if a is None or not a.shape[1]:
            continue                      # text (a task, a note): read by the task and note readers, not a signal
        if bad:
            bad_cells_issue(out, str(c), bad, len(a), "cells that are empty or not of its "
                                                      f"{a.shape[1]} value{'s' if a.shape[1] != 1 else ''}")
        inf = int(np.isinf(a).any(axis=1).sum())
        if inf:
            bad_cells_issue(out, str(c), inf, len(a), "cells holding a value that is not a finite number")
        a = on_frames(a, at, n)
        # a row has a reading when any of its values does, as checks/sensors.py counts it: a pressure map with one dead
        # cell still reads at every frame. A column with few readings is kept, NaN where it has none (write_signals
        # records the gaps); only one with no reading at all says nothing
        has_reading = np.isfinite(a).any(axis=1)
        if not has_reading.any():
            out.left_out.append((str(c), "no reading at any frame"))
            continue
        a = np.where(np.isfinite(a), a, np.nan)
        if a.shape[1] == 1 and is_named_clock(c, a[:, 0]):
            # a clock: kept for the sync check, not shown
            out.clocks[str(c)] = a[:n, 0] if n else a[:, 0]
            continue
        if a.shape[1] == 1 and is_counter(a[:, 0]):
            out.left_out.append((str(c), COUNTER_NOTE))
            continue
        f = features.get(c) or {}
        shape = f.get("shape") if isinstance(f.get("shape"), (list, tuple)) and int(np.prod(f["shape"])) == a.shape[1] \
            else _cell_shape(df[c], a.shape[1])
        out.add(str(c), a[:n] if n else a, shape=shape, names=value_names(f.get("names"), a.shape[1]),
                source="column " + str(c))
    return out


def frame_rows(df, n: int) -> np.ndarray | None:
    """The frame each row of an episode's table belongs to, from its frame_index when that numbers frames once each
    (a LeRobot table, where a row is a frame), else None (rows are frames in order)."""
    if not n or df is None or "frame_index" not in df.columns:
        return None
    try:
        fi = np.asarray(df["frame_index"].to_numpy(), dtype=np.int64)
    except (TypeError, ValueError):
        return None
    # as given: a table whose first rows are missing starts at a later frame, never moved to frame 0
    return fi if len(np.unique(fi)) == len(fi) else None


def on_frames(a: np.ndarray, at: np.ndarray | None, n: int) -> np.ndarray:
    """A table's rows on n frames: each at its frame (frame_rows) or in order, a frame no row reaches NaN, and rows past
    the last frame not shown. n 0 keeps the rows as they are."""
    if not n:
        return a
    out = np.full((n, a.shape[1]), np.nan)
    if at is None:
        out[:min(n, len(a))] = a[:n]
    else:
        keep = (at >= 0) & (at < n)
        out[at[keep]] = a[keep]
    return out


def table_issue(out: Signals, df, at: np.ndarray | None, n: int) -> None:
    """A data issue when an episode's table and its video do not cover the same frames: a table shorter than the video
    (table_short), whose signals are NaN where it has no row, or longer (table_long), whose rows past the video's last
    frame have no picture to show them with."""
    rows = np.arange(len(df)) if at is None else at
    have = len(np.unique(rows[(rows >= 0) & (rows < n)]))
    past = int(np.count_nonzero((rows >= n) | (rows < 0)))
    if have < n:
        out.issues.append({"kind": "table_short", "what": f"The episode's data table has rows for {have} of its {n} "
                                                          "video frames, so its signals have no reading at the "
                                                          f"other {n - have}."})
    if past:
        out.issues.append({"kind": "table_long", "what": f"The episode's data table has {past} row"
                                                         f"{'s' if past != 1 else ''} past the video's last frame, "
                                                         "which have no picture to be shown with."})


def bad_cells_issue(out: Signals, name: str, bad: int, rows: int, what: str) -> None:
    out.issues.append({"kind": "signal_bad_cells", "signal": name,
                       "what": f"{name} has {bad} of {rows} {what}; those frames are kept as missing readings"})


def _image_cells(col):
    """(rows, (h, w), how many images could not be read) of a column of small encoded images (PNG bytes, as LeRobot
    stores an image feature), each read as grey values at the size most of them have; an image that does not decode
    or has another size is a row of NaN, never a reason to drop the column. None when no image decodes."""
    import io
    from PIL import Image
    got = []
    for cell in col.to_numpy():
        b = cell.get("bytes") if isinstance(cell, dict) else cell
        try:
            got.append(np.asarray(Image.open(io.BytesIO(bytes(b))).convert("L"), dtype=np.float64))
        except Exception:
            got.append(None)
    shapes = [a.shape for a in got if a is not None]
    if not shapes:
        return None
    shape = max(set(shapes), key=shapes.count)
    rows = np.full((len(got), int(np.prod(shape))), np.nan)
    for i, a in enumerate(got):
        if a is not None and a.shape == shape:
            rows[i] = a.ravel()
    return rows, list(shape), sum(1 for a in got if a is None or a.shape != shape)


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


# A stream's lead or tail with no reading at the footage's ends is a recorder starting up or stopping, not data
# missing, while it is within both of these: half a second is nothing in a minute of footage, but 40 percent of a 1 s
# episode is a sensor that missed it. One rule for every stream: a signal's lead within it is no data issue
# (signal_gaps), and an arm state is held from its first or last reading across it and never further (fill_rows,
# joint_state), so no stretch the signals call missing is shown as recorded stillness.
EDGE_SLACK_S = 0.5
EDGE_SLACK_SHARE = 0.1


def edge_slack(span_s: float) -> float:
    """The lead or tail with no reading that a stream is allowed in footage of span_s seconds: the smaller of
    EDGE_SLACK_S and EDGE_SLACK_SHARE of the footage."""
    return min(EDGE_SLACK_S, EDGE_SLACK_SHARE * span_s)


def covers_footage(t0: float, t1: float, q: np.ndarray) -> bool:
    """Whether readings from t0 to t1 cover the footage whose frames are at times q (the same clock): they start and
    end within the edge slack (edge_slack) of its first and last frame."""
    span = float(q[-1] - q[0]) if len(q) > 1 else 0.0
    return span > 0 and t0 <= q[0] + edge_slack(span) and t1 >= q[-1] - edge_slack(span)


def span_on_footage(t0: float, t1: float, zero: float, length: float) -> tuple[float, float]:
    """Readings from t0 to t1 as seconds of the footage, whose first frame is at zero on their clock and which lasts
    length seconds, held within it: a reading before the first frame starts the span at 0, one after the last frame
    ends it at the footage's end, so a data issue never names a time the footage does not have."""
    return min(max(t0 - zero, 0.0), length), min(max(t1 - zero, 0.0), length)


def signal_gaps(name: str, a: np.ndarray, t: np.ndarray) -> list[dict]:
    """The data issues of a kept signal's frames with no reading (a row with no finite value), in seconds of the
    episode (t, its frames' times): before its first reading or after its last one, when longer than the edge slack
    (edge_slack), a signal_partial_span each (a sensor started late or stopped early), and every other frame
    without a reading counted in one signal_gap with its longest run. A lead or a tail within the slack is a recorder
    starting up or stopping, no issue: a camera's calibration first sent 0.2 s in had made a gap issue of every value
    it holds."""
    none = ~np.isfinite(np.asarray(a)).any(axis=1)
    n = len(none)
    if not none.any() or none.all() or len(t) != n:
        return []
    t = np.asarray(t, dtype=np.float64) - float(t[0])
    slack = edge_slack(float(t[-1]))
    read = np.flatnonzero(~none)
    first, last = int(read[0]), int(read[-1])
    out = []
    if first and t[first] > slack:
        out.append({"kind": "signal_partial_span", "signal": name, "t0_s": 0.0, "t1_s": float(t[first]),
                    "what": f"{name} has no reading before {t[first]:.1f} s, so it is missing over the start of the "
                            "footage"})
    none[:first] = False
    if last < n - 1 and t[-1] - t[last] > slack:
        out.append({"kind": "signal_partial_span", "signal": name, "t0_s": float(t[last]), "t1_s": float(t[-1]),
                    "what": f"{name} has no reading after {t[last]:.1f} s, so it is missing over the end of the "
                            "footage"})
    none[last + 1:] = False
    if none.any():
        edges = np.diff(np.concatenate([[0], none.astype(np.int8), [0]]))
        runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))
        a0, a1 = max(runs, key=lambda r: r[1] - r[0])
        k = int(none.sum())
        out.append({"kind": "signal_gap", "signal": name, "t0_s": float(t[a0]), "t1_s": float(t[a1]),
                    "what": f"{name} has no reading at {k} of its {n} frames"
                            + (f"; the longest gap runs from {t[a0]:.2f} s to {t[a1]:.2f} s" if a1 > a0 else "")})
    return out


def write_signals(ep: Path, ctx: dict, signals: dict | None, t: np.ndarray | None = None) -> None:
    """signals.npz beside the state (keys s0, s1, ...) and ctx["signals"], [{name, key, dims, shape, names, source}],
    trimmed to the episode's n_state_frames; nothing when there are none. What a reader read but did not keep
    (Signals.left_out) is written to ctx["source"]["unused_signals"], so the report names it, and the problems with
    what it kept (Signals.issues) to ctx["reader_issues"] (add_issue)."""
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
    for i in getattr(signals, "issues", []) or []:
        add_issue(ctx, **i)
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
    t = np.asarray(t, dtype=np.float64)[:n] if t is not None and len(t) >= n else \
        np.arange(n) / float(ctx.get("fps") or 30.0)
    # past SIGNAL_EPISODE_BYTES together, the widest signals are kept as a summary per frame, a data issue each
    meta = {k: dict(m) for k, m in meta.items()}
    size = lambda v: int(np.prod(np.shape(v)[1:])) * n * 4
    while sum(size(v) for v in keep.values()) > SIGNAL_EPISODE_BYTES:
        k = max((k for k in keep if np.shape(keep[k])[1] > len(SUMMARY_NAMES)), key=lambda k: np.shape(keep[k])[1],
                default=None)
        if k is None:
            break
        width = int(np.shape(keep[k])[1])
        keep[k] = summarise_rows(keep[k][:n])
        m = meta.setdefault(k, {})
        m.pop("shape", None)
        m.update(names=list(SUMMARY_NAMES), summary_of=width)
        add_issue(ctx, **summary_issue(k, width))
    stored, quiet = {}, set(getattr(signals, "no_gaps", ()) or ())
    for k, v in keep.items():
        # an inf is never stored: it is a missing reading (NaN), as each reader reads it, and a data issue here when
        # the reader had not already made it NaN, so no range shown for the signal reads "- to -"
        a = np.asarray(v[:n], dtype=np.float32)
        inf = np.isinf(a)
        if inf.any():
            a = np.where(inf, np.float32(np.nan), a)
            rows = np.flatnonzero(inf.reshape(n, -1).any(axis=1))
            add_issue(ctx, "signal_not_finite", f"{k} has {int(inf.sum())} value{'s' if inf.sum() != 1 else ''} that "
                                                f"{'are' if inf.sum() != 1 else 'is'} not a finite number in "
                                                f"{len(rows)} frame{'s' if len(rows) != 1 else ''}; read as missing",
                      signal=k, t0_s=float(t[rows[0]] - t[0]), t1_s=float(t[rows[-1]] - t[0]))
        stored[k] = a
        if k in quiet:
            continue
        for i in signal_gaps(k, a.reshape(n, -1), t):
            add_issue(ctx, **i)
    np.savez(ep / "signals.npz", **{f"s{i}": a for i, a in enumerate(stored.values())})
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


# The fields of context.json that hold times on the episode's clock, {field: the time keys of each entry}. A step that
# moves the clock (board/clips.py reanchor, when the camera the clock was measured from is taken out) moves every one of
# them, so a reader that writes a new timed field registers it here, where the context is written.
CLOCK_TIME_KEYS = {"annotation_subtasks": ("t0", "t1"), "contacts": ("start_s", "end_s", "peak_s"),
                   "reader_issues": ("t0_s", "t1_s"), "unshown_cameras": ("start_s",)}


def finish_episode(ep: Path, ctx: dict, sources: dict, state=None, action=None, times: dict | None = None,
                   signals: dict | None = None) -> dict:
    ep.mkdir(parents=True, exist_ok=True)
    if state is not None and ctx.get("state_kind") != "none":
        arrs = {"state": np.asarray(state, dtype=np.float32)}
        if action is not None and np.shape(action) == np.shape(state):
            arrs["action"] = np.asarray(action, dtype=np.float32)
        np.savez(ep / "state.npz", **arrs)
    from label import episode as me
    anchor = next(iter(me.order_views(ctx.get("cameras") or sources or {})), None)
    write_signals(ep, ctx, signals, (times or {}).get(anchor))
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


TABLE_CHUNK_ROWS = 200_000
TABLE_STREAM_MAX_ROWS = 2_000_000   # a table longer than this keeps every so many rows, still finer than the frames


TABLE_SEPARATORS = (",", ";", "\t", "|")    # in the order a tie between them is settled
TABLE_SNIFF_LINES = 20       # a table's separator is judged on its header and the lines after it, up to this many
DECIMAL_COMMA = re.compile(r"^\s*[-+]?\d*,\d+\s*$")                       # a number with a decimal comma: 0,033
# Thousands grouped by dots (1.234,56). A first group of 0 is a decimal point (0.896), never thousands, and one dot
# group alone (1.234) can be a decimal point too, so a column's dots are thousands only when one of its cells has dots
# that cannot be one (THOUSANDS_PROOF: 1.234,56 or 1.234.567; column_numbers).
GROUPED_NUMBER = re.compile(r"^\s*[-+]?[1-9]\d{0,2}(\.\d{3})+(,\d+)?\s*$")
THOUSANDS_PROOF = re.compile(r"^\s*[-+]?[1-9]\d{0,2}((\.\d{3}){2,}(,\d+)?|(\.\d{3})+,\d+)\s*$")


def table_format(p: Path) -> tuple[str, str]:
    """(separator, decimal mark) of a CSV or TSV table, judged on its first TABLE_SNIFF_LINES lines split as a CSV
    reader splits them, so a separator inside a quoted name ("force [N; x]") splits nothing. A .tsv whose header
    splits on tabs is tab separated. Otherwise the separator is the one of TABLE_SEPARATORS that
    splits every one of those lines into the same number of fields, the most fields when several do (a table written
    with semicolons, as spreadsheets in many locales write it, had been read as one column of text, and counting
    separators in the header alone split a .tsv whose names hold commas on its commas), or when none does, the one
    that splits its header into the most fields; a comma (a tab for a .tsv) for a table of one column. A semicolon
    table whose cells are numbers written with a decimal comma (0,033 or 1.234,56) reads them with it, as those
    spreadsheets write them, or none of its numbers would read as one. Dots between thousands are a column's, not
    the table's (column_numbers)."""
    import csv
    with open(p, newline="", errors="replace") as fh:
        lines = [line for line in (fh.readline() for _ in range(TABLE_SNIFF_LINES)) if line.strip()]
    rows = {s: [r for r in csv.reader(lines, delimiter=s) if r] for s in TABLE_SEPARATORS}
    tsv = Path(p).suffix.lower() == ".tsv"
    if tsv and rows["\t"] and len(rows["\t"][0]) > 1:
        return "\t", "."
    widths = {s: {len(r) for r in rs} for s, rs in rows.items()}
    steady = [s for s in TABLE_SEPARATORS if len(widths[s]) == 1 and min(widths[s]) > 1]
    if steady:
        sep = max(steady, key=lambda s: min(widths[s]))
    else:
        head = {s: len(rs[0]) if rs else 0 for s, rs in rows.items()}
        sep = max(TABLE_SEPARATORS, key=lambda s: head[s])
        if head[sep] < 2:
            sep = "\t" if tsv else ","
    cells = [c for r in rows[sep][1:] for c in r] if sep == ";" else []
    return sep, "," if any(DECIMAL_COMMA.match(c) or GROUPED_NUMBER.match(c) and "," in c for c in cells) else "."


def table_separator(p: Path) -> str:
    """The separator of a CSV or TSV table (table_format)."""
    return table_format(p)[0]


# A column is numbers when at least this share of its filled cells read as numbers. A stray cell of text among them
# (an "ERR" a logger wrote for a dropped reading) is a bad cell of a column of numbers, flagged where it is.
NUMBER_COLUMN_SHARE = 0.9
# A column whose only text is one mark repeated ("-", "ERR") is a reading with that mark where a reading is missing,
# however often it drops out, unless its numbers take at most this many values: then it is a column of codes, a phase
# written as 1, 2 or "grasp", text and never a signal of numbers with bad cells.
CODE_VALUES_MAX = 8
UNNAMED_COLUMN = re.compile(r"^Unnamed: \d+$")      # pandas' name for a column whose header cell is blank


def proves_thousands(col) -> bool:
    """Whether a table column has a cell whose dots can only be between thousands (THOUSANDS_PROOF)."""
    return bool(col.astype(str).str.strip().str.match(THOUSANDS_PROOF).any())


def column_numbers(col, decimal: str = ".", thousands: bool | None = None):
    """A table column's cells as numbers (NaN where a cell is not one), read as pandas reads a number, and with two
    marks it does not know. Dots between thousands (GROUPED_NUMBER) are read as such in a column whose cells prove
    them (proves_thousands, or thousands when the column was judged already), never across a table, so a column of
    0.896 beside one of 1.234,56 keeps its decimal point. A decimal comma (DECIMAL_COMMA) is read in a table whose
    decimal mark is a comma (table_format) and in a column with thousands dots, also in a column holding a cell of
    text, which pandas leaves as text whole. A column pandas read as numbers is kept as it is."""
    import pandas as pd
    x = pd.to_numeric(col, errors="coerce")
    if pd.api.types.is_numeric_dtype(col):
        return x
    if thousands is None:
        thousands = proves_thousands(col)
    if not thousands and decimal != ",":
        return x
    cells = col.astype(str).str.strip()
    grouped = cells.str.match(GROUPED_NUMBER) if thousands else pd.Series(False, index=col.index)
    fix = grouped | (cells.str.match(DECIMAL_COMMA) & x.isna())
    plain = cells.where(~grouped, cells.str.replace(".", "", regex=False)).str.replace(",", ".", regex=False)
    return x.where(~fix, pd.to_numeric(plain, errors="coerce"))


def number_columns(df, decimal: str = ".") -> tuple:
    """(the columns of a table that hold numbers, as floats, [(name, cells that are not numbers, filled cells)] of
    each filled column that does not), its cells read by column_numbers with the table's decimal mark. A column holds
    numbers when its filled cells read as numbers at least NUMBER_COLUMN_SHARE of the time, or when its only text is
    one mark repeated and its numbers take more than CODE_VALUES_MAX values (a sensor's "-" where it dropped out); each
    cell that is not a number (a stray "ERR", the mark, an empty cell) is NaN, so a text cell never drops its column
    (the table's bad cells are flagged where it is placed, table_signals). A column of text or codes (a task, a note, a
    phase) is left to annotation_tables, and returned so the reader can name it. A named column with no cell filled
    stays, as one with no reading, flagged where it is placed. A column with neither a name nor a filled cell is no
    column of the data: a separator at the end of every line, as some writers put one, leaves it, so it is ignored
    without a word."""
    import pandas as pd
    keep, text = {}, []
    for c in df.columns:
        col = df[c]
        filled = col.notna() & (col.astype(str).str.strip() != "")
        if not filled.any() and UNNAMED_COLUMN.match(str(c)):
            continue
        if pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col):
            keep[c] = col.astype(np.float64)
            continue
        x = column_numbers(col, decimal)
        words = col[filled & x.isna()].astype(str).str.strip()
        n_filled = int(filled.sum())
        mark = words.nunique() == 1 and x.nunique() > CODE_VALUES_MAX
        if not n_filled or n_filled - len(words) >= NUMBER_COLUMN_SHARE * n_filled or mark:
            keep[c] = x.astype(np.float64)
        else:
            text.append((str(c), len(words), n_filled))
    return pd.DataFrame(keep, index=df.index), text


def read_number_table(p: Path):
    """(the number columns of a CSV or TSV table, number_columns, the columns of it that are not numbers with how many
    of their cells are not, the stride its rows were kept at, how many rows it has), read with its own separator,
    decimal mark (table_format). A table up to TABLE_MAX_BYTES is read whole; a larger one in chunks of
    TABLE_CHUNK_ROWS, keeping the columns its first chunk holds numbers in, each read with the thousands its first
    chunk proves (column_numbers; a cell that is not a number is NaN), and once more than TABLE_STREAM_MAX_ROWS rows
    are kept, every other kept row is dropped and the stride doubled, so memory stays bounded and the rows stay spread
    over the whole recording. A table over 20 MB beside the videos had been ignored."""
    import pandas as pd
    sep, decimal = table_format(p)
    if Path(p).stat().st_size <= TABLE_MAX_BYTES:
        df, text = number_columns(pd.read_csv(p, sep=sep, decimal=decimal), decimal)
        return df, text, 1, len(df)
    parts, cols, text, stride, rows, kept = [], None, [], 1, 0, 0
    for ch in pd.read_csv(p, sep=sep, decimal=decimal, chunksize=TABLE_CHUNK_ROWS):
        if cols is None:
            first, text = number_columns(ch, decimal)
            cols = list(first.columns)
            grouped = {c: proves_thousands(ch[c]) for c in cols}
        start = (-rows) % stride
        rows += len(ch)
        # a time column keeps float64: float32 holds about 7 digits, so epoch seconds at 100 Hz collapse together
        ch = pd.DataFrame({c: column_numbers(ch[c], decimal, grouped[c]) for c in cols}, index=ch.index)
        ch = ch.astype({c: np.float64 if is_time_name(c) else np.float32 for c in cols}).iloc[start::stride]
        parts.append(ch)
        kept += len(ch)
        while kept > TABLE_STREAM_MAX_ROWS:
            # keep every other row of what is kept so far, counted across the parts as if they were one table
            joined = pd.concat(parts, ignore_index=True).iloc[::2]
            parts, kept, stride = [joined], len(joined), stride * 2
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return df, text, stride, rows


def table_seconds(raw: np.ndarray, real_anchor, t_vid: np.ndarray) -> np.ndarray:
    """A table's time column in seconds. Beside capture times (real_anchor, seconds on the recorder's clock) it is read
    against them as the clocks of one recording are (_clocks_in_seconds: the unit under which its range overlaps
    theirs), so a table on the epoch clock at 0.5 Hz, whose step alone says milliseconds, is read in seconds. Without
    them it is read by its own unit (_seconds), unless that puts its span more than SPAN_MATCH away from the
    footage's (t_vid) and another of CLOCK_SCALES puts it within: the units are 1000 apart, so at most one can."""
    raw = np.asarray(raw, dtype=np.float64)
    if real_anchor is not None:
        return _clocks_in_seconds({"table": raw, "camera": np.asarray(real_anchor, dtype=np.float64)}, "camera")["table"]
    t = _seconds(raw)
    ok = raw[np.isfinite(raw)]
    span, vspan = (float(ok[-1] - ok[0]) if len(ok) > 1 else 0.0), float(t_vid[-1] - t_vid[0]) if len(t_vid) > 1 else 0.0
    if span > 0 and vspan > 0 and abs(np.log(span * _seconds_scale(raw) / vspan)) > np.log(SPAN_MATCH):
        fits = [s for s in CLOCK_SCALES if abs(np.log(span * s / vspan)) <= np.log(SPAN_MATCH)]
        if fits:
            return raw * fits[0]
    return t


ALIGNED_ROWS = "row per frame"   # a signal's meta "aligned_by" when a table was placed one row per frame
ROWS_ASSUMED = ("{} (from {}) has as many rows as the video has frames, so each row was placed on one frame; its "
                "timing assumes one row per frame")
SPAN_DIFFERS_MIN_S = 0.5         # a table placed one row per frame whose own times span this much more or less than
SPAN_DIFFERS_SHARE = 0.05        # the footage, and this share of it, is said to disagree with the footage's frame rate


def table_signals(paths: list[Path], real_anchor, pr_anchor: dict, extra: dict) -> Signals:
    """The numbers of CSV tables beside an episode's videos as signals, one per table under its own name (the file's
    name without its take: "traj"), its number columns (number_columns) as the named values. A table is placed:
    - by its time column (a rising column named for time, read in seconds by table_seconds) on the recorder's clock,
      when the videos carry capture times on a recorder's clock (recorder_clock) and its own times are on one too and
      overlap them: recorded timing;
    - otherwise, when it has as many rows as the video has frames, one row per frame, whatever its time column says:
      a recorder that writes a row as it writes each frame stamps the row on its own clock, which may run at another
      rate than the video's nominal one (rows that span 10.3 s beside 8 s of video at 30 fps), and placed by those
      stamps its readings drift from the motion they record; a time column that counts frames reads as one second
      per row the same way. The placement is assumed, marked (ALIGNED_ROWS) and a data issue, and a time column whose
      span disagrees with the footage's is a data issue too (table_span_differs: by its own times the video plays
      fast or slow);
    - otherwise by its time column from the videos' start, an alignment that is assumed: marked (mark_assumed) and a
      data issue. A row with no time is left out of this placement and said.
    A table placed by its time records its rate (rate_hz), so a slow one is said to be held between readings. Every
    value's bad cells (empty, not a number, not finite) are missing readings and a data issue each (signal_bad_cells),
    and a value with no reading at all is left out and flagged. A table that cannot be placed, or has a single row (a
    setting, not a reading over time), is left out with the reason."""
    out = Signals()
    n = int(len(pr_anchor["pts"]))
    t_vid = pr_anchor["pts"].astype(np.float64) * float(pr_anchor["time_base"])
    t_vid = t_vid - t_vid[0]
    for p in paths:
        try:
            num, text, stride, rows = read_number_table(p)
        except Exception:
            out.left_out.append((p.name, "could not be read as a table"))
            continue
        if stride > 1:
            out.issues.append({"kind": "table_downsampled", "what": f"{p.name} has {rows:,} rows, more than the "
                                                                    f"{TABLE_STREAM_MAX_ROWS:,} read whole, so every "
                                                                    f"{stride}th row was read"})
        if num.shape[1] == 0:
            continue                      # text only: the uploader's notes, read by annotation_tables
        # a column of text or codes beside the numbers is named, never dropped without a word (number_columns)
        out.left_out += [(f"{c} in {p.name}", f"{k} of its {m} filled cells are not numbers, so it is not read as a "
                                              "signal") for c, k, m in text]
        if len(num) < 2:
            vals = ", ".join(f"{c} {x:g}" for c, x in zip(num.columns[:8], num.iloc[0, :8])) if len(num) else ""
            out.left_out.append((p.name, f"one row ({vals}), so a setting or a report rather than a reading over time"
                                 if len(num) else "no rows"))
            continue
        clocks = [c for c in num.columns if is_named_clock(c, num[c].to_numpy())]
        tcol = clocks[0] if clocks else None
        skip = [c for c in num.columns if c in clocks or SIGNAL_SKIP.search(str(c))]
        counters = [c for c in num.columns if c not in skip and is_counter(num[c].to_numpy())]
        out.left_out += [(f"{c} in {p.name}", COUNTER_NOTE) for c in counters]
        vals = num.drop(columns=skip + counters)
        if vals.shape[1] == 0:
            continue
        parts = name_parts(p.stem)
        name = parts["cam"] or p.stem
        v = vals.to_numpy(dtype=np.float64)
        t = timed = None
        if tcol is not None:
            raw = num[tcol].to_numpy(dtype=np.float64)
            timed = np.isfinite(raw)
            t = table_seconds(raw[timed], real_anchor, t_vid) if timed.sum() > 1 else None
        # recorded timing only when both are on a recorder's clock, as split_sensors places a sensor file: capture times
        # and a time column that both count from 0 overlap whatever they are (a frame count read as 1 Hz)
        on_clock = (t is not None and real_anchor is not None and recorder_clock(real_anchor) and recorder_clock(t)
                    and t[0] < real_anchor[-1] and t[-1] > real_anchor[0])
        assumed, aligned, rate, span_note = False, None, None, None
        if not on_clock and len(v) == n:
            a, gaps, row_t, aligned = v, 0, t_vid, ALIGNED_ROWS
            span, vspan = (float(t[-1] - t[0]) if t is not None else 0.0), float(t_vid[-1])
            if t is not None and abs(span - vspan) > max(SPAN_DIFFERS_MIN_S, SPAN_DIFFERS_SHARE * vspan):
                span_note = (f"{p.name} was placed one row per video frame, but its time column {tcol} says its {n} "
                             f"rows span {span:.1f} s, not the video's {vspan:.1f} s; by its own times the video "
                             f"plays {'fast' if span > vspan else 'slow'}, unless {tcol} does not count seconds")
        elif t is not None:
            if not timed.all():
                # a row with no time cannot be placed by time: left out, and said
                out.issues.append({"kind": "signal_bad_cells", "signal": name,
                                   "what": f"{name} has {int((~timed).sum())} of {len(timed)} rows with no time in "
                                           f"its time column {tcol}, so those rows could not be placed"})
                v = v[timed]
            if on_clock:
                a, var, gaps = place_on_frames(t, v, np.asarray(real_anchor, dtype=np.float64))
                row_t = t - float(real_anchor[0])
            else:
                a, var, gaps = place_on_frames(t - t[0], v, t_vid)
                row_t = t - t[0]
                assumed = True
            if len(t) > 1 and t[-1] > t[0]:
                rate = (len(t) - 1) / float(t[-1] - t[0])
        else:
            out.left_out.append((p.name, f"{len(v)} rows and no time column, while the video has {n} frames"))
            continue
        names = [str(c) for c in vals.columns]
        # every value's bad cells, where they are in the episode; a value with no reading at all is left out
        bad = ~np.isfinite(v)
        empty = [j for j in range(v.shape[1]) if bad[:, j].all()]
        for j in range(v.shape[1]):
            k = int(bad[:, j].sum())
            if not k:
                continue
            if j in empty:
                what = f"{name} {names[j]} has no reading in any of its {len(v)} rows, so it is left out"
                out.issues.append({"kind": "signal_bad_cells", "signal": name, "what": what})
                continue
            rows_bad = np.flatnonzero(bad[:, j])
            out.issues.append({"kind": "signal_bad_cells", "signal": name,
                               "what": f"{name} {names[j]} has {k} of {len(v)} cells that are empty, not a number or "
                                       "not finite; they are read as missing",
                               "t0_s": float(row_t[rows_bad[0]]), "t1_s": float(row_t[rows_bad[-1]])})
        if empty:
            out.left_out += [(f"{names[j]} in {p.name}", "no reading in any row") for j in empty]
            keep = [j for j in range(v.shape[1]) if j not in empty]
            if not keep:
                continue
            a = np.asarray(a)[:, keep]
            names = [names[j] for j in keep]
        a = np.where(np.isfinite(a), a, np.nan)
        out.add(name, a, names=names, source=f"table {p.name}")
        if gaps:
            out.meta[name]["gaps"] = gaps
        if rate:
            out.meta[name]["rate_hz"] = round(rate, 2)
        if assumed:
            # its time column shares no clock with the videos: placed from both starts, marked and flagged
            one = Signals()
            one.add(name, a)
            mark_assumed(one, extra, p.name)
            out.meta[name]["aligned_by"] = ALIGNED_ASSUMED
        if aligned:
            out.meta[name]["aligned_by"] = aligned
            add_issue(extra, "signal_alignment_assumed", ROWS_ASSUMED.format(name, p.name), signal=name)
        if span_note:
            add_issue(extra, "table_span_differs", span_note, signal=name)
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
                      "depth": {str(root / r): root / pairs[r] for _, r in e["cams"] if r in pairs}, "unshown": []})
    # an infrared, mask or unmatched depth video goes to the board with the episode of its folder (the one there, or
    # the one whose take its name gives), never to the model
    for r in left_out:
        here = [it for it in items if item_folder(it) == (root / r).parent]
        take = name_parts(Path(r).stem)["take"]
        for it in here if len(here) == 1 else [it for it in here if take and item_take(it) == take]:
            it["unshown"].append(root / r)
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
        csvs.setdefault(d, sorted(p for p in d.glob("*") if p.suffix.lower() in (".csv", ".tsv")))
    eps_in = {}
    for it in items:
        eps_in[Path(it["files"][0]).parent] = eps_in.get(Path(it["files"][0]).parent, 0) + 1
    for it in items:
        d = Path(it["files"][0]).parent
        take = name_parts(Path(it["files"][0]).stem)["take"]
        it["series"] = [p for p in csvs.get(d, []) if eps_in[d] == 1 or (take and name_parts(p.stem)["take"] == take)]
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
    descs = {}
    signals = Signals()
    from label import episode as me
    anchor = me.order_views(files)[0]
    # sensor files of the episode's folder (assign_sensors): by their clock on the capture times, from both starts when
    # they share no clock with the footage, or listed (split_sensors)
    by_clock, assumed, unplaced = split_sensors(item, real[anchor], real[anchor] is not None)
    note_sensors(extra, signals, by_clock, assumed, unplaced, q=real[anchor])
    mcap_files = [p for p in by_clock if p.suffix.lower() == ".mcap"]
    h5_files = [p for p in by_clock if p.suffix.lower() in H5_EXT]
    arm_rig = rig == "teleop_arms"
    # an arm state only when every arm the sensor files may hold was read on the footage's clock (state_blockers): a
    # file that could not be read, or was damaged where the footage needs it, and an arm placed from both starts
    # beside files on the footage's clock leave the episode with no arm state, every channel a signal
    lost: list = []
    streams = mcap_joint_streams(mcap_files, real[anchor], lost) if arm_rig and mcap_files else {}
    bad = unreadable_sensors(unplaced) + damaged_files(lost) if rig != "ego_head" else []
    others = "the other sensor files" if len(by_clock) + len(assumed) + len(unplaced) > len(bad) else None
    blocked = state_blockers(bad, assumed_arms(assumed, streams) if arm_rig and by_clock else [], others)
    if by_clock:
        if not arm_rig or not mcap_files:
            # a glove's pressure and hand pose beside a head camera, a handheld gripper's IMU: every number the MCAP
            # files record, on the videos' clock (mcap_signals); an MCAP arm channel is read as the state on an arm rig
            # only, and an HDF5 array named as the state below (h5_state)
            merge_signals(signals, mcap_signals(mcap_files, real[anchor]) if mcap_files else Signals())
        else:
            state, action, note = joint_state(streams, real[anchor]) if not blocked else (None, None, None)
            merge_signals(signals, mcap_signals(mcap_files, real[anchor], state_fields(streams, state, action)))
            signals.left_out += joint_left_out(streams, state, action)
            if note:
                no_state(extra, note)
            elif state is not None:
                extra["source"]["state"] = [Path(p).name for p in mcap_files]
                third = third_arms(streams)
                if third:
                    extra["state_note"] = (f"The recording has a third arm ({', '.join(third)}) beside the left and "
                                           "right arms; it is read as neither working arm.")
                    # a scene camera whose name says it is on an arm, beside a third arm, is carried by that arm
                    if "exo" in files and is_mount_named(files["exo"][0]):
                        descs["exo"] = third_arm_camera_desc(third)
        if h5_files:
            more = h5_file_signals(h5_files, real[anchor], len(real[anchor]))
            if state is None and rig != "ego_head" and not blocked:
                # an HDF5 array named as the state (a robot.h5's qpos beside the videos) is read by the rule an HDF5
                # episode's is (h5_state), and leaves the signals when it is read
                state, action, state_names, state_src, h5_note = h5_state(
                    more, rig, real[anchor], [p.stem for p in h5_files] if len(h5_files) > 1 else None)
                if state is not None:
                    extra["source"]["state"] = state_src
                    drop_no_state(extra)                 # a note on the MCAP arm channels, which are not the state
                elif h5_note and "state_note" not in extra:
                    no_state(extra, h5_note)
            merge_signals(signals, more)
    if blocked:
        no_state(extra, blocked)
    if assumed:
        t_rel = (np.asarray(real[anchor], dtype=np.float64) - float(real[anchor][0])) if real[anchor] is not None \
            else seconds(prs[anchor])
        if arm_rig and state is None and "state_note" not in extra:
            # no sensor file on the footage's clock: an arm the files record has nothing to place it against
            no_state(extra, StateNote(
                "Labelled from the cameras, because the sensor files share no clock with the videos to place a "
                "recorded arm state against; their channels are kept as signals.", "assumed_clock")
                if assumed_arms(assumed, {}) else StateNote(
                "Labelled from the cameras, because the sensor files beside the videos record no arm state; their "
                "channels are kept as signals.", "not_recorded"))
        merge_signals(signals, sensors_from_start(assumed, t_rel, extra))
    if item.get("series"):
        more = table_signals(item["series"], real.get(anchor), prs[anchor], extra)
        if not hasattr(signals, "meta"):
            signals = Signals(signals)
        merge_signals(signals, more)
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
    # the cameras the model is not shown go to the board (unshown_cameras), timed on the recorder's clock when the
    # videos carry capture times, else from their own first frame as every separate video file is
    zero = real.get(anchor)[0] if real.get(anchor) is not None else None

    def start_of(path):
        if zero is None:
            return 0.0
        try:
            t = frame_times(Path(path), _frame_count(Path(path)))
        except Exception:
            t = None
        return float(t[0] - zero) if t is not None and len(t) else 0.0
    shown = [nm for nm, _ in files.values()]
    unshown = [unshown_entry(nm, by[nm], unshown_why(nm, rig, shown), start_s=start_of(by[nm]))
               for nm in (unused if item["dir"] is not None else [])]
    unshown += [unshown_entry(Path(q).stem, q, unshown_why(Path(q).stem, rig, shown) if not_rgb(Path(q).stem) else
                              "an infrared, mask or unmatched depth video, not the colour picture the model reads",
                              start_s=start_of(q)) for q in item.get("unshown") or []]
    if any(unshown):
        extra["unshown_cameras"] = [u for u in unshown if u]
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
    absent, no_video = 0, []
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
            if data is not None:
                no_video.append(data)
            else:
                absent += e in rows
            continue
        tasks = row.get("tasks")
        tasks = [str(t) for t in tasks] if isinstance(tasks, list) else ([str(tasks)] if tasks else [])
        root["episodes"].append({"eidx": e, "length": int(scalar(row["length"])) if row.get("length") else None,
                                 "tasks": tasks, "data": data, "videos": vids_e})
    if absent:
        root["used"].append(f"The metadata{where} lists {absent} more episodes than were uploaded; the uploaded ones were labelled.")
    no_video_note(root, rdir, no_video)


def no_video_note(root: dict, rdir: Path, data_files: list[Path]) -> None:
    """The episodes of a LeRobot dataset with a data file but no video, named by their data files in the report: an
    episode is shown and labelled beside its footage, so these cannot be yet, and had been dropped without a word."""
    if not data_files:
        return
    where = f" in {root['rel']}" if root["rel"] else ""
    names = sorted({Path(p).relative_to(rdir).as_posix() for p in data_files})
    shown = ", ".join(names[:20]) + (f" and {len(names) - 20} more" if len(names) > 20 else "")
    root["missing"].append(f"{len(data_files)} episode{'s' if len(data_files) != 1 else ''}{where} "
                           f"{'have' if len(data_files) != 1 else 'has'} a data file but no video ({shown}); an "
                           "episode is shown and labelled beside its video, so "
                           f"{'they were' if len(data_files) != 1 else 'it was'} not labelled.")


PACKED_UNALIGNED = "its packed videos could not be lined up with this episode's frames"


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
        skipped, no_video = 0, []
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
            data = None
            try:
                dp = inside(rdir, rdir / data_tpl.format(chunk_index=int(scalar(e["data/chunk_index"])),
                                                         file_index=int(scalar(e["data/file_index"]))))
                data = dp if dp.exists() else None
            except (KeyError, ValueError):
                pass
            if not vids_e:
                if data is not None:
                    no_video.append(data)
                else:
                    skipped += 1
                continue
            tasks = e.get("tasks")
            tasks = [str(t) for t in tasks] if hasattr(tasks, "__len__") and not isinstance(tasks, str) else ([str(tasks)] if tasks else [])
            root["episodes"].append({"eidx": eidx, "length": int(scalar(e["length"])) if e.get("length") is not None else None,
                                     "tasks": tasks, "data": data, "videos": vids_e})
        if skipped:
            root["used"].append(f"The metadata{where} lists {skipped} more episodes than there is uploaded video for; the "
                                "uploaded ones were labelled.")
        no_video_note(root, rdir, sorted(set(no_video)))
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
    # an episode is kept with the cameras whose packed videos place it frame for frame; a camera that does not place it
    # is listed on the episode with the reason (convert_lerobot), never the reason to drop the episode
    placed = sorted(set().union(*[set(p) for p in places.values()])) if places else []
    if placed and fps:
        part = 0
        for eidx in placed:
            have = [k for k in keys if eidx in places[k]]
            vids_e = {k: (places[k][eidx][0], places[k][eidx][1] / fps, (places[k][eidx][1] + lengths[eidx]) / fps)
                      for k in have}
            part += len(have) < len(keys)
            root["episodes"].append({"eidx": eidx, "length": lengths[eidx], "tasks": tasks_of.get(eidx, []),
                                     "data": data_of.get(eidx), "videos": vids_e,
                                     "unplaced": {k: PACKED_UNALIGNED for k in keys if k not in have}})
        root["used"].append(f"{len(placed)} episodes{where} found in the data files and matched to the packed videos "
                            "frame for frame" + (f"; {part} of them by only some of the cameras, and each lists the "
                                                 "cameras that could not be lined up." if part else "."))
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
        files, unaligned = {lead: p}, {}
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
                unaligned[k] = f"its packed file has {b} frames and the {lead} file {a}, so the two cannot be lined up"
            else:
                unaligned[k] = f"it has no packed file of the same name as the {lead} file"
            dropped.add(k)
        # the data file packed under the same name, read when it holds one row per frame (convert_recording)
        dp = rdir / "data" / p.parent.name / (p.stem + ".parquet")
        root["recordings"].append({"name": p.relative_to(rdir).with_suffix("").as_posix(), "files": files,
                                   "unaligned": unaligned, "data": dp if dp.exists() else None})
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
                continue                   # an episode across a file boundary is not placed; the others are
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
                          "unaligned": rec.get("unaligned") or {}, "data": rec.get("data"),
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


def _cells_rows(col) -> tuple[np.ndarray | None, int]:
    """(a column as (rows, values), how many of its cells could not be read). Each cell is flattened, a list of lists
    (a 16 x 16 pressure map, which parquet gives as an array of arrays) in its own order; the width is the size most
    cells have, and a cell that is empty, not numbers or of another size is a row of NaN, never a reason to drop the
    column (the 2026-10-03 audit found a glove dropped for one None cell in a hundred). None when no cell holds
    numbers (a text column)."""
    flat, sizes = [], {}
    for x in col.to_numpy():
        try:
            v = np.asarray(_nested(x), dtype=np.float64).reshape(-1)
        except Exception:
            v = None
        if v is not None and v.size == 0:
            v = None
        flat.append(v)
        if v is not None:
            sizes[v.size] = sizes.get(v.size, 0) + 1
    if not sizes:
        return None, len(flat)
    w = max(sizes, key=lambda k: (sizes[k], k))
    a = np.full((len(flat), w), np.nan, dtype=np.float32 if w > SIGNAL_MAX_VALUES else np.float64)
    bad = 0
    for i, v in enumerate(flat):
        if v is not None and v.size == w:
            a[i] = v
        else:
            bad += 1
    return a, bad


def _cells(col) -> np.ndarray | None:
    """A column as (rows, values) (_cells_rows), a cell it cannot read a row of NaN; None for a text column."""
    return _cells_rows(col)[0]


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
    # every row as it is, NaN where a cell cannot be read; a frame with no reading is filled or the state is left a
    # signal (state_on_frames), never dropped for one NaN frame
    state = _cells(df["observation.state"]) if df is not None and "observation.state" in df.columns else None
    action = _cells(df["action"]) if df is not None and "action" in df.columns else None
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
    unshown_keys = list(unused)
    descs = colour_depth_views(r, row, vmap)
    # a camera the metadata lists whose video is not on disk (an adapter downloads only the cameras it uses) is unused
    # too, and one whose packed video could not be placed on this episode says why (_episodes_v3)
    unplaced = row.get("unplaced") or {}
    unused = unused + [f"{k} ({unplaced[k]})" if k in unplaced else k for k in r["cams"] if k not in video_cams]
    for k in unplaced:
        add_issue(extra, "camera_not_aligned", f"The camera {k} is not shown: {unplaced[k]}.", camera=k)
    if r["image_cams"]:
        unused = unused + [f"{k} (images in the data file)" for k in r["image_cams"]]
    extra["source"]["unused_cameras"] = unused
    kind, note = state_layout(state.shape[1] if state is not None else 0, rig, state_value_names(feats, state))
    note = StateNote(note, "layout") if note else None
    if state is None and rig != "ego_head":
        # no observation.state, or one with no numbers, is a state not recorded; a data file that did not come is
        # one that could not be read
        note = (StateNote("Labelled from the video: the dataset's observation.state holds no numbers.", "not_recorded")
                if df is not None and "observation.state" in df.columns else
                StateNote("Labelled from the video: the dataset records no observation.state.", "not_recorded")
                if df is not None or row.get("data") is None else
                StateNote(note, "not_recorded") if note else None)
        if row.get("data") is None:
            note = StateNote("Labelled from the video: no data file came with this episode.", "unreadable")
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
            if key in descs:
                cameras[v]["desc"] = descs[key]
        n_frames = min(s["n_frames"] for s in sources.values())
        state, action, kind, note, fixes = state_on_frames(df, state, action, n_frames, fps, kind, note)
        ctx = {"dataset": dataset, "profile": rig, "state_kind": kind, "episode_id": ep.name, "fps": fps,
               "n_state_frames": int(len(state)) if state is not None and kind != "none" else int(min(s["n_frames"] for s in sources.values())),
               "cameras": cameras, "stream_checks": {"episode_length_meta": row.get("length")}, **extra}
        un = lerobot_unshown(row, unshown_keys, rig, list(vmap.values()), fps)
        if un:
            ctx["unshown_cameras"] = un
        if note or notes:
            no_state(ctx, lerobot_state_note(note, notes))
        for i in fixes:
            add_issue(ctx, **i)
        write_depth(ep, ctx, *lerobot_depth(ep, r, row, vmap, fps, ctx["n_state_frames"], unused, ctx))
        return finish_episode(ep, ctx, sources, state if kind != "none" else None, action,
                              signals=recorded_signals(df, _used_columns(kind, state, action) | set(hold_back),
                                                       ctx["n_state_frames"], feats))
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
        if key in descs:
            cameras[v]["desc"] = descs[key]
    times = None
    if not all(grid.values()):
        times = {}
        for v, pr in prs.items():
            times[v] = np.arange(len(pr["pts"])) / fps
            times[f"{v}_pts"] = pr["pts"]
    n_video = min(s["n_frames"] for s in sources.values())
    state, action, kind, note, fixes = state_on_frames(df, state, action, sources[anchor]["n_frames"], fps, kind, note)
    ctx = {"dataset": dataset, "profile": rig, "state_kind": kind, "episode_id": ep.name, "fps": fps,
           "n_state_frames": int(len(state)) if state is not None and kind != "none" else int(n_video),
           "cameras": cameras, "stream_checks": {"frames_on_grid": grid, "episode_length_meta": row.get("length")},
           **extra}
    un = lerobot_unshown(row, unshown_keys, rig, list(vmap.values()), fps)
    if un:
        ctx["unshown_cameras"] = un
    if note or notes:
        no_state(ctx, lerobot_state_note(note, notes))
    for i in fixes:
        add_issue(ctx, **i)
    write_depth(ep, ctx, *lerobot_depth(ep, r, row, vmap, fps, ctx["n_state_frames"], unused, ctx))
    return finish_episode(ep, ctx, sources, state if kind != "none" else None, action, times=times,
                          signals=recorded_signals(df, _used_columns(kind, state, action) | set(hold_back),
                                                   ctx["n_state_frames"], feats))


def colour_depth_views(r: dict, row: dict, vmap: dict) -> dict:
    """{depth feature: its camera line} for each of an episode's depth videos stored as an ordinary picture (an 8-bit
    colour stream holds the recorder's shading of near and far, not distances), each added to vmap as a camera of its
    own (extra1, extra2, ...), as convert_video shows such a video beside its colour camera. They had been skipped
    without a word."""
    out = {}
    cams = dict(vmap)
    if not cams:
        return out
    from label import episode as me
    anchor = me.order_views(cams)[0]
    for key in r.get("depth_cams") or []:
        src = row["videos"].get(key)
        if src is None:
            continue
        try:
            pf = str(probe_depth(Path(src[0] if isinstance(src, tuple) else src))["pix_fmt"] or "")
        except Exception:
            continue                          # lerobot_depth lists it with the reason
        if pf.startswith("gray"):
            continue
        v, _ = depth_camera(key, cams, anchor)
        vmap[f"extra{1 + sum(1 for x in vmap if x.startswith('extra'))}"] = key
        out[key] = (f"the depth of {cams[v]}, stored by the dataset as an ordinary picture, with its own shading of "
                    "near and far")
    return out


def lerobot_unshown(row: dict, keys: list[str], rig: str, shown: list[str], fps: float | None) -> list[dict]:
    """The episode's cameras the model is not shown (pick_cameras' unused), for the board (unshown_cameras): a packed
    camera by its window of the shared file, a per episode file whole."""
    out = []
    for k in keys:
        src = row["videos"].get(k)
        if src is None:
            continue
        why = unshown_why(k, rig, shown)
        if isinstance(src, tuple):
            mp4, base, to = src
            e = unshown_entry(k, mp4, why, base_s=base, n_frames=int(round((to - base) * float(fps or 30.0))),
                              fps=fps)
        else:
            e = unshown_entry(k, src, why)
        if e:
            out.append(e)
    return out


def lerobot_depth(ep: Path, r: dict, row: dict, vmap: dict, fps: float, n: int, unused: list,
                  ctx: dict | None = None) -> tuple[dict, dict]:
    """(depth.json entries, depth times) of a LeRobot episode's depth videos (features marked video.is_depth_map or
    named depth), each with its camera as the HDF5 reader pairs them (depth_camera). A depth video that is not read
    (a second stream for one camera, one that will not open, has no frame in the episode's window, or is stored in
    colour and is not already a camera, colour_depth_views) is added to unused (the episode's unused cameras) with the
    reason and is a data issue on ctx (depth_not_read), never skipped without a word. Frames are timed as LeRobot
    defines them, frame index over fps, from the episode's own window of a packed file."""
    dep, tz = {}, {}
    if not r.get("depth_cams") or not vmap:
        return dep, tz
    from label import episode as me
    cams = {v: k for v, k in vmap.items() if k not in r["depth_cams"]}
    anchor = me.order_views(cams or vmap)[0]
    ta = np.arange(n) / float(fps or 30.0)

    def leave(key, why):
        unused.append(f"{key} ({why})")
        if ctx is not None:
            add_issue(ctx, "depth_not_read", f"The depth video {key} is not used: {why}.", camera=key)
    for key in r["depth_cams"]:
        src = row["videos"].get(key)
        if src is None or key in vmap.values():
            continue
        path, base, to = (src if isinstance(src, tuple) else (src, None, None))
        v, source = depth_camera(key, cams or vmap, anchor)
        if v in dep:
            leave(key, "depth with no camera of its own")
            continue
        try:
            pr = probe_depth(Path(path))
        except Exception:
            leave(key, "the depth video could not be opened")
            continue
        if not str(pr["pix_fmt"] or "").startswith("gray"):
            leave(key, "stored in colour, so it holds no distances")
            continue
        t_all = pr["pts"].astype(np.float64) * float(pr["time_base"])
        sel = (t_all >= base - 1e-6) & (t_all < to - 1e-6) if base is not None else np.ones(len(t_all), dtype=bool)
        if not sel.any():
            leave(key, "no depth frame falls inside this episode's window of its packed file")
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


def _used_columns(kind: str, state=None, action=None) -> set:
    """The columns already read as the state and action. A state that does not fit the arm layout (kind "none") is not
    read as one, so it stays a signal the model is shown, and so does an action finish_episode does not keep (not read,
    or of another shape than the state), which had been dropped without a word."""
    if kind == "none":
        return set()
    kept = action is not None and (state is None or np.shape(action) == np.shape(state))
    return {"observation.state"} | ({"action"} if kept else set())


def state_filled_issue(what: str, k: int, n: int, t0: float, t1: float) -> dict:
    """The data issue of a state whose frames with no reading were filled (fill_rows)."""
    return {"kind": "state_filled", "signal": what, "t0_s": t0, "t1_s": t1,
            "what": f"{what} has no reading at {k} of its {n} frames; they were filled from the readings around them, "
                    f"across no gap longer than {STATE_GAP_S:g} s"}


def lerobot_state_note(note: StateNote | None, notes: list[str]) -> StateNote:
    """A LeRobot episode's state note: why its state was not read (note) followed by its data file's read failures
    (notes), which alone leave it "unreadable"."""
    return StateNote(" ".join(x for x in [note, *notes] if x), note.why if note else "unreadable")


def state_on_frames(df, state, action, n: int, fps: float | None, kind: str, note: str | None) -> tuple:
    """(state, action, kind, note, data issues) of a LeRobot episode's state and action rows (_cells, NaN where a cell
    could not be read) on its n video frames: each row at its frame (frame_rows, on_frames), and a frame with no
    reading filled from the readings around it across no gap longer than STATE_GAP_S (fill_rows), as the MCAP
    and HDF5 state readers fill an arm's frames, with a state_filled issue. A state with a longer gap (a table shorter
    than its video) is not read as the state, with the gap in the note; its column stays a signal (_used_columns). A
    state with one NaN frame had been dropped and the episode told the dataset records no observation.state."""
    issues = []
    if state is None or kind == "none" or not n:
        return state, action, kind, note, issues
    at = frame_rows(df, n)
    t = np.arange(n) / float(fps or 30.0)

    def fit(a, what):
        a = on_frames(np.asarray(a, dtype=np.float64), at, n)
        ok = np.isfinite(a).all(axis=1)
        if ok.all():
            return a, None
        if not ok.any():
            return None, "has no reading on any frame"
        rows, gap = fill_rows(t, t[ok], a[ok])
        if gap:
            return None, gap_words(gap, 0.0)
        miss = np.flatnonzero(~ok)
        issues.append(state_filled_issue(what, len(miss), n, float(t[miss[0]]), float(t[miss[-1]])))
        return rows, None
    st, why = fit(state, "observation.state")
    if st is None:
        return state, None, "none", StateNote(f"Labelled from the video, because the recorded observation.state {why}; "
                                              "it is kept among the signals.", "short"), issues
    act = fit(action, "action")[0] if action is not None else None
    return st, act, kind, note, issues


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
    # a camera meta/info.json lists whose column is not in this episode's data file is listed, never a failure
    absent = [k for k in r["image_cams"] if k not in df.columns]
    vmap, unused = pick_cameras([k for k in r["image_cams"] if k in df.columns], rig, list(r["features"]))
    not_shown = list(unused)
    unused = unused + [f"{k} (listed in meta/info.json, not in the data file)" for k in absent]
    ep.mkdir(parents=True, exist_ok=True)
    fi = df["frame_index"].to_numpy() if "frame_index" in df.columns else np.arange(len(df))
    t = df["timestamp"].to_numpy(dtype=np.float64) if "timestamp" in df.columns else fi / fps
    t = t - t[0]
    files, undecoded = {}, []
    for v, key in vmap.items():
        w = FrameWriter(ep / f"{v}.mp4", "")
        for ts, cell in zip(t, df[key].to_numpy()):
            b = cell.get("bytes") if isinstance(cell, dict) else cell
            if isinstance(b, (bytes, bytearray)) and b:
                w.add(float(ts), bytes(b))
        if not w.close():
            # one camera whose images do not decode leaves the others, never fails the episode
            undecoded.append(key)
            unused.append(f"{key} (none of its images in the data file could be decoded)")
            continue
        files[v] = (key, ep / f"{v}.mp4")
    if not files:
        raise ValueError(f"the {', '.join(undecoded)} images in the data file could not be decoded")
    extra = {**extra}
    extra["source"]["unused_cameras"] = unused
    # the cameras the model is not shown are written too, for the board (unshown_cameras)
    un = []
    for i, key in enumerate(not_shown):
        w = FrameWriter(ep / f"unshown{i + 1}.mp4", "")
        for ts, cell in zip(t, df[key].to_numpy()):
            b = cell.get("bytes") if isinstance(cell, dict) else cell
            if isinstance(b, (bytes, bytearray)) and b:
                w.add(float(ts), bytes(b))
        n_u = w.close()
        if n_u:
            un.append(unshown_entry(key, ep / f"unshown{i + 1}.mp4", unshown_why(key, rig, list(vmap.values())),
                                    n_frames=n_u, start_s=w.pts[0] / TIME_BASE_DEN,
                                    fps=measured_fps(np.asarray(w.pts, dtype=np.float64) / TIME_BASE_DEN)))
    if any(un):
        extra["unshown_cameras"] = [u for u in un if u]
    for key in undecoded:
        add_issue(extra, "camera_not_decodable", f"None of the {key} images in the data file could be decoded, so the "
                                                 "camera is not shown.", camera=key)
    extra["source"]["images_in_parquet"] = True
    kind, note = state_layout(state.shape[1] if state is not None else 0, rig, state_value_names(r["features"], state))
    note = StateNote(note, "layout") if note else None
    # the state's frames with no reading filled as in convert_lerobot (state_on_frames), one row per image frame
    state, action, kind, note, fixes = state_on_frames(df, state, action, len(df), fps, kind, note)
    for i in fixes:
        add_issue(extra, **i)
    # the table's other numbers go with the frames as they do beside videos (recorded_signals)
    signals = recorded_signals(df, _used_columns(kind, state, action) | set(r["image_cams"]), 0, r["features"])
    ctx = video_views_episode(ep, files, rig, dataset, extra, signals=signals)
    if state is not None and kind != "none" and len(state) == ctx["n_state_frames"]:
        ctx["state_kind"] = kind
        return finish_episode(ep, ctx, json.loads((ep / "sources.json").read_text()), state, action,
                              times={k: v for k, v in np.load(ep / "times.npz").items()}, signals=signals)
    if rig != "ego_head":
        # no observation.state at all is a state not recorded, whatever the layout note says of its width
        no_state(ctx, StateNote(note, "not_recorded" if state is None else note.why) if note else StateNote(
            "Labelled from the video: the recorded state and the image frames cannot be lined up.", "layout"))
        (ep / "context.json").write_text(json.dumps(ctx, indent=1, default=str))
    return ctx


def convert_recording(item: dict, rig: str, out: Path, dataset: str) -> dict:
    """A packed LeRobot video whose episodes could not be placed with certainty: the whole file is one
    recording, labelled from video. A camera packed differently is listed with the reason and is a data issue, and
    the data file packed under the same name gives the recording its signals when it holds exactly one row per frame
    of the scene camera's file (a row is a frame, in order); otherwise it is listed with its row count."""
    import pandas as pd
    keys = list(item["files"])
    vmap, unused = pick_cameras(keys, rig, keys)
    files = {v: (k, item["files"][k]) for v, k in vmap.items()}
    unaligned = item.get("unaligned") or {}
    extra = {"task_label": [item["name"]],
             "source": {"format": f"lerobot {item['root']['version']} (packed video kept whole)", "file": item["name"],
                        "unused_cameras": unused + [f"{k} ({why})" for k, why in unaligned.items()]},
             "unsplit": True}
    for k, why in unaligned.items():
        add_issue(extra, "camera_not_aligned", f"The camera {k} is not shown: {why}.", camera=k)
    un = [unshown_entry(k, item["files"][k], unshown_why(k, rig, list(vmap.values()))) for k in unused]
    if any(un):
        extra["unshown_cameras"] = [u for u in un if u]
    if rig != "ego_head":
        # the state is recorded, but its rows are not matched to this packed video's episodes, a layout we do not read
        no_state(extra, StateNote("Labelled from the video, as one recording: its episodes could not be matched to "
                                  "the packed video exactly.", "layout"))
    signals = None
    if item.get("data") is not None:
        from label import episode as me
        lead = files[me.order_views(files)[0]][1]
        signals = Signals()
        try:
            df = pd.read_parquet(item["data"])
            order = [c for c in ("episode_index", "frame_index") if c in df.columns]
            df = df.sort_values(order, kind="stable") if order else df
            n = _frame_count(lead)
        except Exception as e:
            signals.left_out.append((Path(item["data"]).name, f"could not be read ({type(e).__name__})"))
        else:
            if len(df) == n:
                signals = recorded_signals(df.drop(columns=["frame_index"], errors="ignore"), set(), n,
                                           item["root"]["features"])
            else:
                signals.left_out.append((Path(item["data"]).name, f"{len(df)} rows while the video has {n} frames, so "
                                                                  "its rows cannot be placed on the frames"))
    return video_views_episode(unique_dir(out, episode_name(item["name"])), files, rig, dataset, extra,
                               signals=signals)


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
    writes before its first stamp does not decide it. Its step is the median between distinct times: a file whose
    channels are written at the same instants (an arm's joints and its health, each stamped alike) repeats every
    time, so half its steps are 0 and their median had been 0, as if it had no clock at all."""
    ok = np.asarray(a, dtype=np.float64).ravel()
    ok = ok[np.isfinite(ok)]
    steps = np.diff(ok)
    steps = steps[steps != 0]
    step = float(np.median(steps)) if len(steps) else 0.0
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
    if _picture_axes(per) is not None and (dt == np.uint8 or (dt.kind == "f" and _picture_values(ds))):
        return "camera"
    if len(per) == 2 and min(per) >= CAMERA_MIN_PX:
        wide_numeric_picture = int(np.prod(per)) > SIGNAL_MAX_VALUES and dt.kind in "uf" and dt.itemsize >= 2
        if "depth" in leaf.lower() or "depth" in name.lower() or wide_numeric_picture:
            return "depth"
        if dt == np.uint8:
            return "camera"
    # an array wider than SIGNAL_MAX_VALUES (a point cloud, a flattened picture) is a signal kept as a map, when it fits
    # an array wider than SIGNAL_MAX_VALUES (a point cloud, a flattened picture) is a signal kept as a map or, past the
    # episode's budget, as a summary (h5_signals)
    return "signal"


H5_PICTURE_MIN_PX = 16     # a colour picture (three or four channels) at least this many pixels on a side is a camera


def _picture_axes(per) -> tuple[int, int, bool] | None:
    """(height, width, channel first) of one sample's shape when it is a colour picture: three or four channels last
    (H, W, 3) or first (3, H, W), and at least H5_PICTURE_MIN_PX pixels on a side. A camera had needed 64 pixels and
    uint8 channels last, so a 48 x 48 picture or one stored channel first was no camera at all."""
    per = tuple(int(x) for x in per)
    if len(per) != 3:
        return None
    if per[2] in (3, 4) and min(per[:2]) >= H5_PICTURE_MIN_PX:
        return per[0], per[1], False
    if per[0] in (3, 4) and min(per[1:]) >= H5_PICTURE_MIN_PX:
        return per[1], per[2], True
    return None


PICTURE_SAMPLE_FRAMES = 5


def _picture_frames(ds) -> np.ndarray | None:
    """PICTURE_SAMPLE_FRAMES samples spread over an array (the first, the last and between), as float64."""
    try:
        n = len(ds)
        idx = sorted({int(x) for x in np.linspace(0, n - 1, PICTURE_SAMPLE_FRAMES)}) if n else []
        return np.stack([np.asarray(ds[i], dtype=np.float64) for i in idx]) if idx else None
    except Exception:
        return None


def _picture_values(ds) -> bool:
    """Whether a float array reads as a picture: finite values from 0 to 1, or 0 to 255, in frames spread over it, not
    its first alone (a flow field can start still and then reach hundreds)."""
    a = _picture_frames(ds)
    return bool(a is not None and a.size and np.isfinite(a).all() and a.min() >= 0 and a.max() <= 255)


def picture_scale(ds) -> float:
    """255 for a float picture stored 0 to 1 (judged on frames spread over it), else 1."""
    if ds.dtype.kind != "f":
        return 1.0
    a = _picture_frames(ds)
    return 255.0 if a is not None and a.size and float(np.nanmax(a)) <= 1.0 else 1.0


def picture(a: np.ndarray, scale: float = 1.0):
    """One sample of an HDF5 camera as an RGB or grey PIL image: channels moved last when stored first, floats times
    scale (255 for a picture stored 0 to 1) and clipped to 0 to 255, a fourth channel dropped."""
    from PIL import Image
    a = np.asarray(a)
    if a.ndim == 3 and a.shape[0] in (3, 4) and a.shape[2] not in (3, 4):
        a = np.moveaxis(a, 0, -1)
    if a.dtype != np.uint8:
        a = np.clip(np.nan_to_num(a.astype(np.float64) * scale), 0, 255).astype(np.uint8)
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[:, :, :3]
    return Image.fromarray(np.ascontiguousarray(a)).convert("RGB")


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
                out["unused"].append(f"{p} ({' x '.join(map(str, ds.shape[1:]))} values per sample, which no "
                                     "reader reads)")
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


def overlaps(t: np.ndarray, q: np.ndarray) -> bool:
    """Whether readings at times t fall anywhere in the footage's frame times q."""
    return len(t) > 0 and len(q) > 0 and float(t[-1]) >= float(q[0]) and float(t[0]) <= float(q[-1])


OUTSIDE_FOOTAGE = "outside the footage"      # the end of why a stream recorded none of the footage (outside_words)


def outside_words(t: np.ndarray, q: np.ndarray) -> str:
    return f"recorded from {t[0] - q[0]:.1f} s to {t[-1] - q[0]:.1f} s, {OUTSIDE_FOOTAGE}"


def not_finite(name: str, a: np.ndarray, t: np.ndarray, zero: float, out: Signals) -> np.ndarray:
    """a (rows of readings at times t) with every value that is not a finite number NaN, and, when any is an inf or a
    NaN beside finite values in its row, a data issue on out (signal_not_finite) giving how many and from the first to
    the last such reading in seconds after zero (the footage's first frame). One bad value had dropped a whole
    signal; now only that value is missing. A row with no finite value is a gap, which write_signals records. A wide
    signal stays float32 (float_rows), and a fresh array of floats is changed in place, never copied."""
    a = float_rows(a)
    fin = np.isfinite(a)
    if fin.all():
        return a
    bad = ~fin & (np.isinf(a) | fin.any(axis=1, keepdims=True))
    if bad.any():
        rows = np.flatnonzero(bad.any(axis=1))
        t = np.asarray(t, dtype=np.float64)
        out.issues.append({"kind": "signal_not_finite", "signal": name,
                           "what": f"{name} has {int(bad.sum())} value{'s' if bad.sum() != 1 else ''} that "
                                   f"{'are' if bad.sum() != 1 else 'is'} not a finite number in {len(rows)} reading"
                                   f"{'s' if len(rows) != 1 else ''}; {'they were' if bad.sum() != 1 else 'it was'} "
                                   "read as missing",
                           "t0_s": float(t[rows[0]] - zero), "t1_s": float(t[rows[-1]] - zero)})
    a[~fin] = np.nan
    return a


def h5_signals(f, streams: dict, q_abs: np.ndarray, fps: float | None, n_anchor: int) -> Signals:
    """Every signal of an episode on the anchor camera's frames: by its own clock, as mcap_signals places a channel
    (nearest sample, NaN where none is near, so a signal that starts or ends inside the footage is NaN outside its
    readings, and left out only when it records nothing inside the footage), or, with no clock, one row per anchor
    frame when it has as many rows as the anchor has frames. A value that is not finite is NaN (not_finite)."""
    out = Signals()
    q = np.asarray(q_abs, dtype=np.float64)
    span = float(q[-1] - q[0]) if len(q) > 1 else 0.0
    for s in streams["signal"]:
        ds = f[s["path"]]
        if ds.dtype.kind not in "biuf":
            continue
        shape = list(ds.shape[1:])
        width = int(np.prod(shape)) if shape else 1
        summarised = None
        if not out.fits(len(ds), width) and width > len(SUMMARY_NAMES):
            # past the running total an episode's signals hold: read a block of rows at a time into its summary
            a, summarised, shape = summarise_rows(ds), width, [len(SUMMARY_NAMES)]
        else:
            a = float_rows(ds[()])
        a = a.reshape(len(a), -1) if a.ndim > 1 else a[:, None]
        if not a.shape[1]:
            continue
        names = list(SUMMARY_NAMES) if summarised else None
        for key in ("names", "columns", "labels", "fields") if not summarised else ():
            if key in ds.attrs:
                stated = [x.decode() if isinstance(x, bytes) else str(x) for x in np.asarray(ds.attrs[key]).ravel()]
                names = value_names(stated, a.shape[1])
                break
        if s["clock"]:
            t = streams["clock"][s["clock"]]
            if span <= 0 or len(t) < 2:
                continue
            if not overlaps(t, q):
                out.left_out.append((s["name"], outside_words(t, q)))
                continue
            # a signal that starts or ends inside the footage is NaN outside its readings (write_signals records the
            # span), and a value that is not finite is NaN where it is (not_finite)
            a = not_finite(s["name"], a, t, q[0], out)
            v, var, n_gaps = place_on_frames(t, a, q)
            far = np.zeros(len(q), dtype=bool)
            far[:n_gaps] = True               # only its count is kept (meta "gaps")
            rate = len(t) / max(float(t[-1] - t[0]), 1e-9)
            if var is not None and a.shape[1] <= VARIATION_MAX_VALUES and _variation_matters(s["name"], v, var):
                out.add(f"{s['name']} variation within each frame", var, shape=shape if len(shape) > 1 else None,
                        names=names, source=f"HDF5 dataset {s['path']}")
                out.meta[f"{s['name']} variation within each frame"]["variation_of"] = s["name"]
        elif len(a) == n_anchor:
            v, rate, far, var = not_finite(s["name"], a, q, q[0], out), fps, None, None
        else:
            out.left_out.append((s["name"], f"{len(a)} rows and no clock, while the camera has {n_anchor} frames"))
            continue
        out.add(s["name"], v, shape=shape if len(shape) > 1 else None, names=names, source=f"HDF5 dataset {s['path']}")
        if summarised:
            out.meta[s["name"]]["summary_of"] = summarised
            out.issues.append(summary_issue(s["name"], summarised))
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
# videos, the arrays of several HDF5 files carry their file's name first ("robot qpos", h5_file_signals); h5_state
# reads the name after that file name, so a space inside an array's own name ("gripper state") is never a separator.
H5_STATE_NAME = re.compile(r"(^|/)(qpos|state|states|robot_state|joint_positions?|joint_pos)$", re.I)
H5_JOINT_ARRAY = re.compile(r"(^|/)joint_pos(itions?)?$", re.I)
H5_ACTION_NAME = re.compile(r"(^|/)actions?$", re.I)
H5_ACTION_GROUP = re.compile(r"(^|/)actions?/", re.I)


def h5_state(signals: Signals, rig: str, q: np.ndarray, files: list[str] | None = None) -> tuple:
    """(state, action, value names, the state array's name, note) of an HDF5 episode, from its signals (h5_signals,
    already on the anchor camera's frames, at times q). The arrays named as the state (H5_STATE_NAME, outside an action
    group) are tried shortest name first, and the first that state_layout lays out with the names the file gives its
    values is the state; the array named as the action goes with it when it has the state's shape. A frame with no
    reading (a clocked array that starts or ends within the edge slack of the footage, edge_slack, which h5_signals
    keeps) is filled as joint_state fills an MCAP arm's frames (fill_rows), so an HDF5 state is accepted wherever an
    MCAP one is, and a longer gap leaves it unread with the gap's time in the note.
    Both leave the signals when the state is read; otherwise they stay, and note gives the first array's reason, named.
    All None on a head camera, which has no state and no note about one, or when no array is named as the state.
    files are the names of the HDF5 files whose arrays carry them first (h5_file_signals), passed over to read each
    array's own name."""
    if rig == "ego_head":
        return None, None, None, None, None
    meta = getattr(signals, "meta", {}) or {}
    left_out = dict(getattr(signals, "left_out", []) or [])

    def own(k):
        # an array's own name, after the HDF5 file's name that h5_file_signals puts first
        stem = next((f for f in files or () if k.startswith(f"{f} ")), None)
        return k[len(stem) + 1:] if stem else k
    named = lambda k: H5_STATE_NAME.search(own(k)) and not H5_ACTION_GROUP.search(own(k))
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
    notes, read, failed, filled_at = [], [], {}, {}

    def unsided(name):
        # an array's own name without its side words, so left/qpos and right/qpos in one file, like left.h5's and
        # right.h5's qpos, are the same array of each side
        return " ".join(w for w in tokens(own(name)) if side_of(w) is None)

    def fail(name, text, why):
        # why an array named as the state is not read (a StateNote, STATE_WHY), kept by the side its name says and its
        # name without that side, so only the other side's same array of the arm read (qpos against qpos) cancels a
        # two arm state
        note = StateNote(text, why) if text else None
        notes.append(note)
        failed.setdefault((side_of(name), unsided(name)), note)
    for name in cands:
        if name not in signals and left_out[name] == NOT_FINITE:
            fail(name, f"Labelled from the video, because the recorded state {name} has values that are not all "
                         "finite numbers.", "unreadable")
            continue
        if name not in signals:
            # a state recorded outside the footage covers none of it ("short"); rows with no clock to line them up
            # by are not a layout the checks read ("layout")
            fail(name, f"Labelled from the video, because the recorded state {name} could not be placed on the "
                         f"camera's frames ({left_out[name]}).",
                 "short" if left_out[name].endswith(OUTSIDE_FOOTAGE) else "layout")
            continue
        a = np.asarray(signals[name], dtype=np.float64)
        names = (meta.get(name) or {}).get("names")
        dims = a.shape[1]
        if names is None and H5_JOINT_ARRAY.search(own(name)) and dims in (7, 14):
            # the file names no value, but the array's name says every value is a joint, so there is no gripper
            fail(name, f"Labelled from the video: the array's name says every value is a joint, so its {dims} values "
                         f"are {dims} joints and no gripper, and our checks read "
                         + ("six joints and a gripper per arm." if rig == "teleop_arms" else
                            "a 6D pose and an opening per gripper.") + f" The recorded state is the HDF5 array {name}.",
                 "layout")
            continue
        kind, note = state_layout(dims, rig, names)
        if kind == "none":
            fail(name, f"{note} The recorded state is the HDF5 array {name}." if note else None, "layout")
            continue
        miss = np.flatnonzero(~np.isfinite(a).all(axis=1))
        a, why = filled(a)
        if a is None:
            fail(name, f"Labelled from the video: the recorded state {name} has no reading on any frame."
                         if why == "has no reading on any frame" else
                         f"Labelled from the video, because the recorded state {name} {why}.", "short")
            continue
        if len(miss):
            filled_at[name] = state_filled_issue(name, len(miss), len(a), float(q[miss[0]] - q[0]),
                                                 float(q[miss[-1]] - q[0]))
        read.append((name, a, names))
    if not read:
        return None, None, None, None, (notes[0] if notes else None)
    # one arm's array whose name says its side (left.h5's qpos, observations/left/qpos), beside the other side's, is
    # half of a two arm state, left first, as joint_state reads two sided arm channels
    first = read[0]
    other = next((r for r in read[1:] if side_of(first[0]) and side_of(r[0]) not in (None, side_of(first[0]))), None)
    arms = sorted([first, other], key=lambda r: side_of(r[0]) != "left") \
        if rig == "teleop_arms" and other and first[1].shape[1] == other[1].shape[1] == JOINT_DIMS else [first]
    # one side's arm whose other side's array failed is half a state, so neither is read, as joint_state reads no arm
    # when one side's channel is not the layout; the note gives the side that failed and why
    lost = ({"left": "right", "right": "left"}.get(side_of(first[0])), unsided(first[0]))
    if rig == "teleop_arms" and len(arms) == 1 and lost[0] and failed.get(lost):
        return None, None, None, None, failed[lost]

    def command(name, shape):
        # the array named as the action beside it: of its shape, of its side when it is one side's arm
        side = side_of(name) if len(arms) > 1 else None
        return next((k for k in signals if H5_ACTION_NAME.search(own(k)) and np.shape(signals[k]) == shape
                     and (side is None or side_of(k) == side)
                     and filled(np.asarray(signals[k], dtype=np.float64))[0] is not None), None)
    acts = [command(name, a.shape) for name, a, _ in arms]
    action = None
    if all(acts):
        action = np.concatenate([filled(np.asarray(signals.pop(k), dtype=np.float64))[0] for k in acts], axis=1)
    for name, _, _ in arms:
        signals.pop(name)
        if name in filled_at and hasattr(signals, "issues"):
            signals.issues.append(filled_at[name])
    for k in [name for name, _, _ in arms] + [k for k in acts if k and action is not None]:
        meta.pop(k, None)
    state = np.concatenate([a for _, a, _ in arms], axis=1)
    names = [x for _, _, nm in arms for x in nm] if all(nm for _, _, nm in arms) else None
    return state, action, names, " and ".join(name for name, _, _ in arms), None


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
        undecoded = []
        for v, s in chosen.items():
            ds = f[s["path"]]
            t = times_of(s)
            w = FrameWriter(ep / f"{v}.mp4", "")
            # a float picture stored 0 to 1 is scaled to 0 to 255, judged on its first frame
            scale = picture_scale(ds)
            for i in range(s["n"]):
                x = ds[i]
                b = _h5_bytes(x) if (ds.dtype.kind in "OV" or ds.ndim == 1) else None
                if b is not None:
                    w.add(float(t[i] - t0), b)
                else:
                    w.add_image(float(t[i] - t0), picture(x, scale))
            if not w.close():
                unused.append(f"{s['name']} (no frame could be decoded)")
                undecoded.append(s["name"])
                continue
            files[v] = (s["name"], ep / f"{v}.mp4")
        if not files:
            raise ValueError("no frame of the HDF5 episode's cameras could be decoded")
        # the cameras the model is not shown are written too, for the board (unshown_cameras)
        unshown = []
        for i, nm in enumerate(x for x in unused if x in by_name):
            s = by_name[nm]
            ds, t = f[s["path"]], times_of(s)
            w = FrameWriter(ep / f"unshown{i + 1}.mp4", "")
            scale = picture_scale(ds)
            for k in range(s["n"]):
                b = _h5_bytes(ds[k]) if (ds.dtype.kind in "OV" or ds.ndim == 1) else None
                if b is not None:
                    w.add(float(t[k] - t0), b)
                else:
                    w.add_image(float(t[k] - t0), picture(ds[k], scale))
            n_u = w.close()
            if n_u:
                unshown.append(unshown_entry(nm, ep / f"unshown{i + 1}.mp4", unshown_why(nm, rig, names_shown(files)),
                                             n_frames=n_u, start_s=w.pts[0] / TIME_BASE_DEN,
                                             fps=measured_fps(np.asarray(w.pts, dtype=np.float64) / TIME_BASE_DEN)))
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
        # sensor files of the episode's folder (assign_sensors): on the camera's clock when both are on a recorder's
        # clock, which may give the state; else from both starts, after the state is read, so an assumed alignment
        # is never the recorded state (split_sensors)
        by_clock, assumed, unplaced = split_sensors(item, q_abs, chosen[anchor]["clock"] is not None)
        sensor_h5s = [p for p in by_clock if p.suffix.lower() in H5_EXT]
        sensor_mcaps = [p for p in by_clock if p.suffix.lower() == ".mcap"]
        if sensor_h5s:
            merge_signals(signals, h5_file_signals(sensor_h5s, q_abs, len(q_abs)))
        if sensor_mcaps:
            merge_signals(signals, mcap_signals(sensor_mcaps, q_abs))
        state, action, state_names, state_src, state_note = h5_state(signals, rig, q_abs)
        sensor_extra: dict = {}
        if assumed:
            merge_signals(signals, sensors_from_start(assumed, q_abs - q_abs[0], sensor_extra))
        instr, notes = h5_text(f, g, st["text"])
    extra = {"task_label": [item["name"]],
             "source": {"format": "hdf5", "file": item["file"].name, "group": g or None, "unused_cameras": unused}}
    for nm in undecoded:
        add_issue(extra, "camera_not_decodable", f"No frame of the camera {nm} could be decoded, so it is not shown.",
                  camera=nm)
    if any(unshown):
        extra["unshown_cameras"] = [u for u in unshown if u]
    if st["unused"]:
        extra["source"]["unused_arrays"] = st["unused"]
    for i in sensor_extra.get("reader_issues") or []:
        add_issue(extra, **i)
    note_sensors(extra, signals, by_clock, assumed, unplaced, q=q_abs)
    if state_src:
        extra["source"]["state"] = state_src
    if state_note:
        no_state(extra, state_note)
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


def names_shown(files: dict) -> list[str]:
    return [nm for nm, _ in files.values()]


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
                # with several files each array, kept or left out, carries its file's name first
                named = (lambda k: f"{p.stem} {k}") if len(paths) > 1 else (lambda k: k)
                for k, v in got.items():
                    out[named(k)] = v
                    out.meta[named(k)] = got.meta[k]
                out.left_out += [(named(k), why) for k, why in got.left_out]
                for i in got.issues:
                    k = i.get("signal")
                    out.issues.append({**i, "signal": named(k), "what": named(k) + i["what"][len(k):]}
                                      if k and i["what"].startswith(k) else i)
                out.left_out += [(named(s["name"]), "no clock to place it against the videos") for s in st["signal"]
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
    # the layout's reader places only the recording's own channels: its folder's sensor files are listed on it
    extra_files = [Path(x) for x in (item.get("state") or []) + (item.get("state_shared") or [])]
    if extra_files:
        why = f"the {layout} reader of this recording places only its own channels"
        ctx["source"].setdefault("unused_signals", []).extend(f"{p.name} ({why})" for p in extra_files)
        ctx["source"]["sensors"] = [p.name for p in extra_files]
        ctx["source"]["sensor_files"] = {p.name: f"not placed: {why}" for p in extra_files}
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


def mcap_messages(fh, path: Path, topics, damaged: list | None = None):
    """(schema, channel, message) of these topics in an open MCAP file fh (at path): by its index in log time order
    when it has a summary that can be read, and record by record otherwise (_mcap_stream), a file cut short, whose
    messages before the cut are then all it holds: its index had made every reader of it throw, so its arms and
    signals were lost. An indexed file damaged inside (a chunk that does not read, its summary whole) gives the
    messages before the damage, and (its path, the first and last log time read in seconds, None when none was) is
    appended to damaged, so the reader can say so: the file is whole by every other sign (sensor_cut)."""
    from mcap.reader import make_reader
    try:
        indexed = make_reader(fh).get_summary() is not None
    except Exception:
        indexed = False
    fh.seek(0)
    if not indexed:
        yield from _mcap_stream(path, set(topics))
        return
    first = last = None
    try:
        for schema, ch, msg in make_reader(fh).iter_messages(topics=sorted(topics), log_time_order=True):
            first, last = first if first is not None else msg.log_time / 1e9, msg.log_time / 1e9
            yield schema, ch, msg
    except Exception:
        if damaged is not None:
            damaged.append((Path(path), first, last))


def damaged_issue(p: Path, t0: float | None, t1: float | None, q: np.ndarray) -> dict | None:
    """The data issue of an MCAP file whose messages stop at damage inside it (mcap_messages), with the span of the
    footage (q, its frame times on the file's clock) its messages read cover (span_on_footage). None when they cover
    the whole footage (covers_footage): the damage is past what the footage needs, so nothing of it was lost."""
    what = f"{Path(p).name} is damaged inside, though its index is whole"
    if t0 is None:
        return {"kind": "mcap_file_damaged", "what": what + ", and none of its messages could be read"}
    q = np.asarray(q, dtype=np.float64)
    if covers_footage(t0, t1, q):
        return None
    a, b = span_on_footage(t0, t1, float(q[0]) if len(q) else t0, float(q[-1] - q[0]) if len(q) else 0.0)
    return {"kind": "mcap_file_damaged", "t0_s": a, "t1_s": b,
            "what": what + f", so only the messages before the damage were read; they cover {a:.1f} s to {b:.1f} s "
                           "of the footage"}


def mcap_message_counts(path: Path) -> dict[str, int]:
    """{topic: how many messages} of an MCAP file by its summary's statistics; {} when it has none to read."""
    from mcap.reader import make_reader
    try:
        with open(path, "rb") as fh:
            s = make_reader(fh).get_summary()
    except Exception:
        return {}
    if s is None or s.statistics is None:
        return {}
    out: dict[str, int] = {}
    for cid, n in s.statistics.channel_message_counts.items():
        if cid in s.channels:
            out[s.channels[cid].topic] = out.get(s.channels[cid].topic, 0) + int(n)
    return out


def damaged_unread(p: Path, chans: list[tuple[str, str]], seen: set, damaged: list, q) -> list[str]:
    """The channels of an indexed MCAP file damaged inside (damaged, as mcap_messages appends it) that its summary
    declares with messages (mcap_message_counts; every declared one when it gives no count), that are not a recorder's
    bookkeeping (bookkeeping_why), and none of whose messages was read (seen). None when no damage was met, or when
    the messages read cover the footage (q, its frame times on the file's clock; covers_footage): the damage is then
    past what the footage needs. Such a channel could be an arm, so a state read beside it could lack one."""
    if not damaged:
        return []
    _, first, last = damaged[0]
    if first is not None and q is not None and covers_footage(first, last, np.asarray(q, dtype=np.float64)):
        return []
    counts = mcap_message_counts(p)
    return sorted(t for t, s in chans if t not in seen and counts.get(t, 1) and not bookkeeping_why(t, s))


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
    place by its frozenset of names (an unnamed set's by its width), how many sets are named, and overflow once its
    messages named more than NAME_SETS_MAX sets. Unnamed widths never count toward that limit."""

    def __init__(self):
        super().__init__()
        self.index: dict = {}
        self.named = 0
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
            if groups.named >= NAME_SETS_MAX:
                groups.overflow = True
                return None, vals
            groups.named += 1
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


def merge_unnamed(groups: NameSets, rows: dict, lists: tuple, keep_all: bool = False) -> tuple[dict, int]:
    """({group index: row} of the name sets read, how many sets the channel or field has once its unnamed rows are
    placed). rows {group index: {"t": times, and each of lists: values}} come from name_group. A message without names
    is read as the reader read it before name sets: its rows join the only named set of their width when there is
    exactly one (a JointState with an empty name list for its first second), and are left with that set when the set
    is read elsewhere (the state). Otherwise the unnamed rows of the first unnamed message's width are one set, and
    rows of any other width are each a set of their own when keep_all (mcap_signals, so a message of another width is
    a signal labelled by its width, never dropped), else dropped and counted in that set's "dropped" (the joint
    reader, which reads one arm's width)."""
    named: dict = {}
    for i, (gn, w) in enumerate(groups):
        if gn is not None:
            named.setdefault(w, []).append(i)
    home = lambda i: named[groups[i][1]][0] if len(named.get(groups[i][1], [])) == 1 else None
    rest = [i for i in rows if groups[i][0] is None and home(i) is None]
    keep = min(rest, key=lambda i: rows[i]["t"][0]) if rest else None
    kept = set(rest) if keep_all else {keep}
    out = {i: r for i, r in rows.items() if groups[i][0] is not None or i in kept}
    for i in sorted(rows):
        if groups[i][0] is not None or i in kept:
            continue
        h, r = home(i), rows[i]
        if h is None:
            out[keep]["dropped"] = out[keep].get("dropped", 0) + len(r["t"])
            out[keep].setdefault("dropped_widths", set()).add(groups[i][1])
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
    return out, sum(1 for gn, _ in groups if gn is not None) + (len(rest) if keep_all else keep is not None)


def group_label(group: tuple) -> str:
    """How a name set (name_group) is told apart from the others on its channel: its names, the first two and a count
    past three, or its width when it names none."""
    names, n = group
    if names is None:
        return f" ({n} values)"
    return " (" + (", ".join(names) if len(names) <= 3 else f"{names[0]}, {names[1]} and {len(names) - 2} more") + ")"


def _join_gripper(groups: list[dict], q: np.ndarray | None = None) -> list[dict]:
    """A channel's two name sets as one arm when one names only a gripper and the other names joints and no gripper
    (an arm's driver and its gripper's driver both publishing /joint_states): the gripper's first value (one finger
    of two, as _joint_row takes a gripper field's first value), placed at the arm's message times, follows the joints.
    Kept apart when the gripper leaves a gap fill_rows does not fill in the footage's frame times q,
    or in the arm's message times without q, or the channel has any other name set."""
    grip = [g for g in groups if g["names"] and all(STATE_GRIPPER_NAME.search(x) for x in g["names"])]
    if len(groups) != 2 or len(grip) != 1:
        return groups
    grip = grip[0]
    arm = groups[1] if groups[0] is grip else groups[0]
    if not arm["names"] or any(STATE_GRIPPER_NAME.search(x) for x in arm["names"]) or len(arm["t"]) < 2:
        return groups
    one = grip["pos"][:, :1]
    if fill_rows(arm["t"] if q is None else q, grip["t"], one)[1]:
        return groups
    pos = np.concatenate([arm["pos"], lerp_rows(arm["t"], grip["t"], one)], axis=1)
    return [{"t": arm["t"], "pos": pos, "names": arm["names"] + grip["names"][:1],
             "fields": arm["fields"] | grip["fields"]}]


def mcap_joint_streams(paths: list[Path], q: np.ndarray | None = None, unread: list | None = None) -> dict:
    """{key: {"t": seconds on the recording's clock, "pos": rows, "names": the value names its messages give, or None,
    "topic": its channel}} for every channel of these MCAP files that carries an arm's joints (JOINT_KEYS); cameras and
    text are not read. A channel's rows are grouped by the names their messages give (name_group), each in its group's
    order; a channel of one group (or of an arm and its gripper, _join_gripper, judged on the footage's frame times q
    when given) is keyed by its topic, and one of several by its topic and each group's label (group_label), with
    the fields its messages fill ("fields", _joint_fields) so that only the group joint_state reads leaves the
    signals. A file cut short is read up to its cut (mcap_messages), its rows put in time order, so an arm in it is
    read as far as it was recorded and joint_state judges whether that covers the footage. An indexed file damaged
    inside is read up to the damage, and when unread is given, each such file that declares channels none of whose
    messages could be read, where the footage q needs them (damaged_unread), is appended to it as (its path, those
    channels, whether any of its messages was read): what they hold is unknown, so no arm state is read beside them
    (state_blockers)."""
    found, facs = {}, _decoders()
    for p in paths:
        chans = [(t, s) for t, s in mcap_channels(p) if not CAMERA_SCHEMA.search(s) and not TEXT_TOPIC.search(t)]
        decs, skip, seen, damaged = {}, set(), set(), []
        with open(p, "rb") as fh:
            try:
                for schema, ch, msg in mcap_messages(fh, p, [t for t, _ in chans], damaged):
                    seen.add(ch.topic)
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
        lost = damaged_unread(p, chans, seen, damaged, q) if unread is not None else []
        if lost:
            unread.append((Path(p), lost, bool(seen)))
    out = {}
    for topic, c in found.items():
        if c["groups"].overflow:
            continue                                  # mcap_signals names it as left out, with the reason
        rows, n_sets = merge_unnamed(c["groups"], dict(enumerate(c["rows"])), ("pos",))
        read = []
        for i, r in sorted(rows.items()):
            order = np.argsort(np.asarray(r["t"]), kind="stable")    # file order, for a file read record by record
            read.append({"t": np.asarray(r["t"])[order], "pos": np.asarray(r["pos"], dtype=np.float64)[order],
                         "names": c["groups"][i][0], "fields": r["fields"], "label": group_label(c["groups"][i]),
                         "dropped": r.get("dropped", 0), "dropped_widths": r.get("dropped_widths") or set()})
        groups = _join_gripper(read, q)
        apart = len(groups) == len(read) and n_sets > 1
        for g in groups:
            if len(g["t"]) > 1:
                s = {"t": g["t"], "pos": g["pos"], "names": g["names"], "topic": topic,
                     "field": next((k for k, _ in sorted(g["fields"], key=str) if k in JOINT_KEYS), "joints")}
                if g.get("dropped"):
                    s["dropped"], s["dropped_widths"] = g["dropped"], sorted(g["dropped_widths"])
                if apart:
                    s["fields"] = g["fields"]
                out[topic + (g["label"] if apart else "")] = s
    return out


# Every other number an MCAP records (a gripper's IMU, an arm's joint velocities and torques, a base's odometry), read
# as recorded_signals reads a LeRobot table's other columns: per channel, each numeric field under the dataset's own
# name, placed on the anchor camera's frames. A channel slower than SIGNAL_MIN_HZ (a battery report) is kept with its
# rate (rate_hz), each frame its nearest message; one with a single message is a setting, listed with its values,
# not a reading over time. One that starts or ends inside the footage is NaN
# outside its messages, never held flat where nothing was recorded, and the span it misses is a data issue.
SIGNAL_MIN_HZ = 1.0
SIGNAL_SKIP_PARTS = {"header", "timestamp", "stamp"}      # a message's own time and sequence bookkeeping
# A gap between two readings inside the footage longer than STATE_GAP_S is a stop the state is never filled across,
# when it is also longer than STATE_STOP_STEPS of the stream's own median steps: a 30 Hz recorder that stops for 2 s
# is caught, and an arm logged at 1 Hz is read as it was before (fill_rows). The footage's ends are edge_slack's.
STATE_GAP_S = 0.5
STATE_STOP_STEPS = 3


def lerp_rows(q: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """y's rows, read at times t, at times q: each column linearly interpolated, and a time before the first reading or
    after the last one holding that reading (np.interp). abc130k's arms are placed on the frames this way, and the
    state readers do it through fill_rows, which fills no gap longer than STATE_GAP_S."""
    return np.stack([np.interp(q, t, y[:, j]) for j in range(y.shape[1])], axis=1)


def fill_rows(q: np.ndarray, t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray | None, tuple | None]:
    """(y's rows at times q by lerp_rows, None), or (None, (start, end, the longest gap allowed there) of the longest
    gap) when two readings in a row leave more of q's span without a reading than both STATE_GAP_S and STATE_STOP_STEPS
    of the stream's median step, or the first or last reading is further than the edge slack (edge_slack of q's span)
    from q's ends. A gap is measured inside the footage, so a message latched seconds before the first frame does not
    make the stretch before the footage a gap.
    A straight line across a recorder that stopped for 2 s would be shown as recorded motion, and a reading held past
    the ends as stillness, where no still span can tell, so a state is filled across no longer gap than the slack its
    edges are allowed, unless the stream always reads that far apart (an arm logged at 1 Hz). Both state readers place
    an arm this way: an MCAP arm's channel (joint_state) and an HDF5 state's frames with a reading (h5_state)."""
    t = np.asarray(t, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if len(t) and len(q):
        lo, hi = np.maximum(t[:-1], q[0]), np.minimum(t[1:], q[-1])
        step = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
        longest, edge = max(STATE_GAP_S, STATE_STOP_STEPS * step), edge_slack(float(q[-1] - q[0]))
        gaps = [(float(lo[i]), float(hi[i]), longest) for i in np.flatnonzero(hi - lo > longest)]
        gaps += [(float(q[0]), float(t[0]), edge)] if t[0] > q[0] + edge else []
        gaps += [(float(t[-1]), float(q[-1]), edge)] if t[-1] < q[-1] - edge else []
        if gaps:
            return None, max(gaps, key=lambda g: g[1] - g[0])
    return lerp_rows(q, t, y), None


def gap_words(gap: tuple, zero: float) -> str:
    """A gap fill_rows will not fill, (start, end, the longest gap allowed there), in seconds of the footage (zero its
    first frame), for a state note."""
    return (f"has no reading from {gap[0] - zero:.1f} s to {gap[1] - zero:.1f} s, a gap longer than the "
            f"{round(gap[2], 2):g} s the reader fills")


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
    v = float_rows(v)
    acc = v.dtype
    dq = float(np.median(np.diff(q))) if len(q) > 1 else 1 / 30
    rate = len(t) / max(float(t[-1] - t[0]), 1e-9) if len(t) > 1 else 0.0
    if rate <= 2.0 / dq:
        idx = nearest(t, q)
        far = np.abs(t[idx] - q) > max(2.5 * float(np.median(np.diff(t))) if len(t) > 1 else 0.0, 2.5 * dq)
        if len(idx) == len(v) and not far.any() and np.array_equal(idx, np.arange(len(v))):
            return v, None, 0             # a sample at every frame, in order: the samples are the frames, no copy
        a = v[idx].copy()
        a[far] = np.nan
        return a, None, int(far.sum())
    edges = np.concatenate([[q[0] - dq / 2], (q[1:] + q[:-1]) / 2, [q[-1] + dq / 2]])
    bin_ = np.searchsorted(edges, t, side="right") - 1
    ok = (bin_ >= 0) & (bin_ < len(q))
    n = np.bincount(bin_[ok], minlength=len(q)).astype(np.float64)
    # a value with no finite reading in a sample (a NaN the recorder wrote) counts toward no mean, so one missing
    # value never empties a frame's other readings
    fin = np.isfinite(v)
    vz = np.where(fin, v, 0.0)
    cnt = np.zeros((len(q), v.shape[1]), dtype=acc)
    tot = np.zeros((len(q), v.shape[1]), dtype=acc)
    sq = np.zeros((len(q), v.shape[1]), dtype=acc)
    np.add.at(cnt, bin_[ok], fin[ok].astype(np.float64))
    np.add.at(tot, bin_[ok], vz[ok])
    np.add.at(sq, bin_[ok], vz[ok] ** 2)
    with np.errstate(all="ignore"):
        mean = tot / cnt
        std = np.sqrt(np.maximum(sq / cnt - mean ** 2, 0.0))
    mean[cnt == 0] = np.nan
    std[cnt == 0] = np.nan
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


# A recorder's own log and health reports (ROS's /rosout and /diagnostics, by message type or by name) are its
# bookkeeping, not a reading of the task: each is named among the signals left out with BOOKKEEPING_NOTE, never shown
# as a signal and never dropped without a word.
BOOKKEEPING_SCHEMA = re.compile(r"(^|[/.])(msg/)?Log$|rosgraph_msgs|rcl_interfaces|diagnostic_msgs", re.I)
BOOKKEEPING_TOPIC = re.compile(r"^/?(rosout(_agg)?|diagnostics(_agg|_toplevel_state)?)$", re.I)
BOOKKEEPING_NOTE = "the recorder's own log or diagnostics, bookkeeping rather than a reading"
# A camera's calibration (sensor_msgs CameraInfo) repeats one set of values: it stays a signal, listed with its
# values, and a stretch with no message of it is no data issue, since nothing it says changes over the episode.
CALIBRATION_SCHEMA = re.compile(r"CameraInfo$", re.I)
CALIBRATION_TOPIC = re.compile(r"camera_info$", re.I)


def bookkeeping_why(topic: str, schema: str) -> str | None:
    """BOOKKEEPING_NOTE for a log or diagnostics topic, by its message type or its name; None for any other."""
    return BOOKKEEPING_NOTE if BOOKKEEPING_SCHEMA.search(schema or "") or BOOKKEEPING_TOPIC.search(topic) else None


def mcap_signals(paths: list[Path], q: np.ndarray, used: dict | None = None) -> Signals:
    """{name: (len(q), values) array} of every numeric field these MCAP files record on channels that are not cameras
    or text, sampled at the recorded message nearest each anchor frame time q (seconds, the files' log-time clock).
    used {topic: fields already read, or None for the whole channel} keeps out what the reader already shows as the
    state. The name is the topic, then the field path ("/robot0/sensor/imu angular_velocity"); each signal keeps its
    values' names and its shape (Signals.meta), and a field that is read but not kept is named with the reason
    (Signals.left_out): too few messages to be a per-frame record, nothing inside the footage, or wider than
    SIGNAL_MAX_VALUES. A value that is not finite is NaN where it is (not_finite), and a channel that starts or ends
    inside the footage is NaN outside its messages. A field's messages are grouped by the names they give its values
    (name_group), each put in its group's order, and a field of several groups is one signal per group, its name
    followed by the group's label (group_label). used names a field, or a (field, its value names) pair for a field
    of that name set only (state_fields)."""
    used, facs = used or {}, _decoders()
    q = np.asarray(q, dtype=np.float64)
    rows: dict[tuple, dict] = {}
    sets: dict[tuple, NameSets] = {}      # (topic, field): its name sets (name_group)
    books: dict[str, str] = {}            # a recorder's own log and diagnostics topics, named and not read
    kinds: dict[str, str] = {}            # each topic's message type
    damaged: list = []                    # files whose index is whole but whose messages stop at damage (mcap_messages)
    for p in paths:
        everything = mcap_channels(p)
        books.update({t: bookkeeping_why(t, s) for t, s in everything if bookkeeping_why(t, s)})
        kinds.update(everything)
        chans = [t for t, s in everything if not CAMERA_SCHEMA.search(s) and not TEXT_TOPIC.search(t)
                 and not (t in used and used[t] is None) and t not in books]
        decs = {}
        with open(p, "rb") as fh:
            try:
                for schema, ch, msg in mcap_messages(fh, p, chans, damaged):
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
    out.left_out += sorted(books.items())
    out.issues += [i for i in (damaged_issue(p, t0, t1, q) for p, t0, t1 in damaged) if i]
    named, by_field = {}, {}
    for (topic, field, i), r in rows.items():
        by_field.setdefault((topic, field), {})[i] = r
    for (topic, field), field_rows in by_field.items():
        name, groups = f"{topic} {field}".strip(), sets[(topic, field)]
        if groups.overflow:
            out.left_out.append((name, f"its messages name its values in more than {NAME_SETS_MAX} different "
                                       "ways, so no value is one reading over time"))
            continue
        field_rows, n_sets = merge_unnamed(groups, field_rows, ("v",), keep_all=True)
        for i, r in sorted(field_rows.items()):
            named[name + (group_label(groups[i]) if n_sets > 1 else "")] = r
    rows = named
    # bookkeeping values (a counter, a device clock) leave the row; a row that names its sensor per message is split

    def no_counters(name, r):
        nm = r["names"] or [f"[{i}]" for i in range(r["d"])]
        v_ = np.asarray(r["v"], dtype=np.float64) if r["v"] else np.zeros((0, r["d"]))
        # a counter, or a value whose name says time and that rises like a clock (is_named_clock)
        cnt = sorted(set(_counters(r)) | {i for i in range(r["d"]) if len(v_) and is_named_clock(nm[i], v_[:, i])})
        if not cnt:
            return r
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
        some = ""
        if len(t) < 2:
            # one message is a setting or a calibration, not a reading over time: listed with its values
            vals = ", ".join(f"{x:g}" for x in r["v"][0][:8]) + (" and more" if r["d"] > 8 else "")
            sparse.append(f"{name} ({vals})")
            continue
        if not overlaps(t, q):
            out.left_out.append((name, some + outside_words(t, q)))
            continue
        # a channel that starts or ends inside the footage is NaN outside its messages (write_signals records the
        # span), and a value that is not finite is NaN where it is (not_finite)
        v = not_finite(name, r["v"], t, float(q[0]), out)
        rate = len(t) / max(float(t[-1] - t[0]), 1e-9)
        # a frame with no message near it (a hand the tracker lost, a sensor that paused) is NaN, not the last value
        # held; a sensor faster than the camera gives each frame the mean of its interval (place_on_frames)
        a, var, gaps = place_on_frames(t, v, q)
        out.add(name, a, shape=r["shape"], names=r["names"], source=f"MCAP channel {r['topic']}")
        out.meta[name]["rate_hz"] = round(rate, 2)
        if CALIBRATION_SCHEMA.search(kinds.get(r["topic"], "")) or CALIBRATION_TOPIC.search(r["topic"]):
            out.no_gaps.add(name)
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
                             "one message each, so settings or reports rather than a reading over time"))
    return out


def joint_state(streams: dict, q: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, StateNote | None]:
    """(state, action, note): the arms' joints and grippers interpolated onto the anchor camera's frame times q (the
    same clock as the streams), 7 values per arm, left arm first; the leader or command channels, when they match, as
    the action. None with a note when the streams are not a layout the checks read ("layout") or an arm does not
    cover the footage ("short", StateNote)."""
    st, act = arm_streams(streams, False), arm_streams(streams, True)
    if not st:
        return None, None, None
    order = [s for s in ("left", "right", "only") if s in st]
    if "only" in order and len(order) > 1:
        if "left" not in st or "right" not in st:
            return None, None, StateNote("Labelled from the cameras, because the recorded arm channels do not say "
                                         "which arm is which.", "layout")
        # an arm that names a side and one that does not could be the same side's arm twice, but a left and a right
        # arm are the two working arms, and an arm beside them that names no side is a third one (third_arms)
        order = ["left", "right"]
    dims = [streams[st[s]]["pos"].shape[1] for s in order]
    if any(d != JOINT_DIMS for d in dims):
        return None, None, StateNote(f"Labelled from the cameras, because the recorded arms have "
                                     f"{' and '.join(map(str, dims))} values per frame and our checks read six joints "
                                     "and a gripper per arm.", "layout")
    for s in order:
        # the channel's own value names settle six joints and a gripper against seven joints (a Franka arm)
        kind, why = state_layout(JOINT_DIMS, "teleop_arms", streams[st[s]].get("names"))
        if kind != "joints":
            return None, None, StateNote(why or ("Labelled from the cameras, because the recorded arm channels name "
                                                 "their values as a pose, not six joints and a gripper."), "layout")
    span = float(q[-1] - q[0]) if len(q) > 1 else 0.0

    def covers(topic):
        # the arm's samples must span the footage: a frame np.interp places past the first or last sample holds that
        # sample's value, so the state would seem to record an arm standing still where nothing was recorded
        return covers_footage(streams[topic]["t"][0], streams[topic]["t"][-1], q)
    short = [st[s] for s in order if not covers(st[s])]
    if short:
        # each arm that falls short is named with the span of the footage it covers, so an arm whose file was cut
        # short says so
        def covered(k):
            t0, t1 = (max(float(x) - float(q[0]), 0.0) for x in (streams[k]["t"][0], streams[k]["t"][-1]))
            return f"{k} has readings from {t0:.1f} s to {t1:.1f} s of the footage's {span:.1f} s"
        return None, None, StateNote("Labelled from the cameras, because the recorded arm state does not cover the "
                                     "footage's time: " + "; ".join(covered(k) for k in short) + ".", "short")

    def fill(topic):
        # the arm's readings on the frames, across no gap longer than the slack (fill_rows)
        return fill_rows(q, streams[topic]["t"], streams[topic]["pos"])
    rows = [fill(st[s]) for s in order]
    gap = next(((st[s], g) for s, (_, g) in zip(order, rows) if g), None)
    if gap:
        return None, None, StateNote(f"Labelled from the cameras, because the recorded arm state {gap[0]} "
                                     f"{gap_words(gap[1], float(q[0]))}.", "short")
    state = np.concatenate([r for r, _ in rows], axis=1)
    action = None
    if all(s in act and streams[act[s]]["pos"].shape[1] == JOINT_DIMS and covers(act[s]) for s in order):
        cmd = [fill(act[s])[0] for s in order]
        action = None if any(c is None for c in cmd) else np.concatenate(cmd, axis=1)
    return state, action, None


def _topic(streams: dict, key: str) -> str:
    # the channel a stream was read from: its key, or the key without the label of its name set (mcap_joint_streams)
    return streams[key].get("topic", key)


def _sides(streams: dict, role: bool) -> dict:
    """{key: left, right or None} of the arm streams (the commands when role): the side each channel's topic names
    (side_of). An arm of six joints and a gripper whose topic names none takes the side every one of its value names
    says (left_joint1 .. left_gripper) only when that completes one left and one right arm the topics did not, so two
    arms in messages of their own on one /joint_states are two arms, as two topics are. Otherwise the topics' sides
    stand: one arm named right_* beside a base's left and right wheels is still the only arm."""
    keys = [t for t in streams if bool(ACTION_TOPIC.search(_topic(streams, t))) == role]
    sides = {t: side_of(_topic(streams, t)) for t in keys}
    named = dict(sides)
    for t in keys:
        names = streams[t].get("names")
        # only a set whose names state_layout reads as six joints and a gripper is an arm (never a hand of 7 joints)
        if named[t] is None and streams[t]["pos"].shape[1] == JOINT_DIMS and names \
                and state_layout(JOINT_DIMS, "teleop_arms", names)[0] == "joints":
            said = {side_of(str(n)) for n in names}
            named[t] = said.pop() if len(said) == 1 else None
    pair = lambda by: {"left", "right"} <= {by[t] for t in keys if streams[t]["pos"].shape[1] == JOINT_DIMS}
    return named if pair(named) and not pair(sides) else sides


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
    by_side, sides = {}, _sides(streams, role)
    for t in sorted(sides, key=lambda t: _arm_rank(streams, t)):
        by_side.setdefault(sides[t] or "only", t)
    return by_side


def third_arms(streams: dict) -> list[str]:
    """The recorded arm channels that are neither working arm: those whose topic names no side, beside a left and a
    right arm (a third arm that carries the scene camera, as on a rig whose camera an operator moves). joint_state
    reads the two sided arms and leaves these out."""
    sides = _sides(streams, False)
    if not {"left", "right"} <= set(sides.values()):
        return []
    return sorted(t for t, side in sides.items() if side is None)


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


def joint_left_out(streams: dict, state, action) -> list[tuple[str, str]]:
    """[(name, why)] for the messages of another number of values on the arm channels read as the state (and the
    action): the field read as the state is not a signal, so without this they would be dropped without a word."""
    if state is None:
        return []
    read = set(arm_streams(streams, False).values()) | (set(arm_streams(streams, True).values()) if action is not None
                                                        else set())
    out = []
    for k in sorted(read):
        s = streams[k]
        if s.get("dropped"):
            widths = " or ".join(str(w) for w in s.get("dropped_widths") or [])
            out.append((f"{_topic(streams, k)} {s.get('field', 'joints')}",
                        f"{s['dropped']} of its messages carry {widths} values, not the {s['pos'].shape[1]} of the arm "
                        "read as the state, so they were not read"))
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
    cam_topics = sorted({t for t, s in chan_topics if CAMERA_SCHEMA.search(s)})
    # every camera channel but depth (distances, read as depth below) is chosen among (pick_cameras): one whose name
    # says it is not colour (thermal, a mask) goes to the unused cameras and the board, never nowhere, and a recording
    # whose only cameras are not colour is labelled from them, as a data issue
    video_topics = [t for t in cam_topics if not DEPTH_TOPIC.search(t)]
    raw_topics = {t for t, s in chan_topics if RAW_IMAGE_SCHEMA.search(s)}
    if not video_topics:
        raise ValueError("the file has no colour camera channel"
                         + (f" (its channels: {', '.join(item['topics'][:12])})" if item["topics"] else ""))
    all_topics = [t for t, _ in chan_topics]
    vmap, unused = pick_cameras(video_topics, rig, all_topics)
    not_colour = [t for t in vmap.values() if not_rgb(t)]
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
    # the cameras the model is not shown are written too, for the board (unshown_cameras)
    unshown_of = {t: f"unshown{i + 1}.mp4" for i, t in enumerate(x for x in unused if x in cam_topics
                                                                  and not DEPTH_TOPIC.search(x))}
    want = set(vmap.values()) | set(text_topics) | set(depth_of) | set(unshown_of)
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
                if ch.topic in unshown_of and t0 is None:
                    continue                      # before the first frame the model is shown: no clock to place it on
                if ch.topic in view_of_topic or ch.topic in unshown_of:
                    if ch.topic in view_of_topic:
                        t0 = int(msg.log_time) if t0 is None else t0      # every camera on the recording's one clock
                    w = writers.get(ch.topic)
                    if w is None:
                        out_mp4 = ep / (f"{view_of_topic[ch.topic]}.mp4" if ch.topic in view_of_topic
                                        else unshown_of[ch.topic])
                        w = writers[ch.topic] = FrameWriter(out_mp4, "raw" if ch.topic in raw_topics
                                                            else str(_field(dec, "format") or "").lower())
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
    for t in missing:
        add_issue(extra, "camera_not_decodable", f"No frame of the camera {t} could be decoded, so it is not shown.",
                  camera=t)
    un = []
    for t, name in unshown_of.items():
        w = writers.get(t)
        if counts.get(t) and w is not None:
            fps_u = measured_fps(np.asarray(w.pts, dtype=np.float64) / TIME_BASE_DEN)
            un.append(unshown_entry(t, ep / name, unshown_why(t, rig, list(vmap.values())), n_frames=counts[t],
                                    start_s=w.pts[0] / TIME_BASE_DEN, fps=fps_u))
    if any(un):
        extra["unshown_cameras"] = [u for u in un if u]
    for t in (t for t in vmap.values() if t in not_colour):
        add_issue(extra, "camera_not_colour", f"{t} is the recording's only camera and its name says it is not a "
                                              "colour camera (infrared, thermal or a mask); the episode is labelled "
                                              "from it.", camera=t)
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
    # sensor files of the episode's folder (assign_sensors): on the recording's log clock when they share it, else
    # from both starts and never the state (split_sensors)
    by_clock, assumed, unplaced = split_sensors(item, q, True)
    sensor_mcaps = [p for p in by_clock if p.suffix.lower() == ".mcap"]
    sensor_h5s = [p for p in by_clock if p.suffix.lower() in H5_EXT]
    if rig == "teleop_arms":
        # an arm state only when every arm the recording and its sensor files may hold was read on its clock
        # (state_blockers), as beside videos
        lost: list = []
        streams = mcap_joint_streams([item["file"]] + sensor_mcaps, q, lost)
        bad = unreadable_sensors(unplaced) + damaged_files(lost)
        others = "the recording's other files" if 1 + len(by_clock) + len(assumed) + len(unplaced) > len(bad) else None
        blocked = state_blockers(bad, assumed_arms(assumed, streams), others)
        state, action, note = joint_state(streams, q) if not blocked else (None, None, blocked)
        used = state_fields(streams, state, action)
    # every other number the file records, under its own name (mcap_signals)
    signals = mcap_signals([item["file"]] + sensor_mcaps, q, used)
    if rig == "teleop_arms":
        signals.left_out += joint_left_out(streams, state, action)
    if sensor_h5s:
        merge_signals(signals, h5_file_signals(sensor_h5s, q, len(q)))
    if assumed:
        merge_signals(signals, sensors_from_start(assumed, q - q[0], extra))
    note_sensors(extra, signals, by_clock, assumed, unplaced, q=q)
    # motion the file records but this reader does not use (hand, body or camera poses in a human recording, joints
    # in a layout the checks do not read): named on the job page, so a missing check is never a silent gap
    motion = [t for t in item["topics"] if re.search(r"hand|pose|slam|body|joint|odom|/tf$", t, re.I)
              and not re.search(r"health|info|meta|static|image|mask", t, re.I)]
    shown_topics = {n.split(" ", 1)[0] for n in signals}
    motion = [t for t in motion if t not in shown_topics]            # kept as signals, so shown to the model
    if state is not None:
        extra["source"]["state"] = "joint channels"
    elif note:
        no_state(extra, note)
    elif item["seconds"] is None and mcap_layout(item["topics"]) != "generic":
        # cut short (no summary), and its messages before the cut hold no arm (mcap_joint_streams reads up to the cut)
        no_state(extra, StateNote("Labelled from the cameras, because the file is cut short before any of its robot "
                                  "state.", "unreadable"))
    elif motion and rig == "ego_head":
        shown = ", ".join(motion[:4]) + (f" and {len(motion) - 4} more" if len(motion) > 4 else "")
        no_state(extra, StateNote(f"Labelled from the camera. The hand, body and camera tracks the file records "
                                  f"({shown}) are not read yet.", "layout"))
    elif motion:
        shown = ", ".join(motion[:4]) + (f" and {len(motion) - 4} more" if len(motion) > 4 else "")
        no_state(extra, StateNote(f"Labelled from the cameras. The checks on recorded motion read six joints and a "
                                  f"gripper per arm, so they did not run on this file's motion channels ({shown}).",
                                  "layout"))
    elif rig != "ego_head":
        no_state(extra, StateNote("Labelled from the cameras, because the file records no robot state.",
                                  "not_recorded"))
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
    # a data issue that starts past the new end is about footage no longer labelled; one across it ends there
    issues = []
    for i in ctx.get("reader_issues") or []:
        if i.get("t0_s") is not None and i["t0_s"] >= ctx["trimmed"]["to_s"]:
            continue
        if i.get("t1_s") is not None and i["t1_s"] > ctx["trimmed"]["to_s"]:
            i = {**i, "t1_s": ctx["trimmed"]["to_s"]}
        issues.append(i)
    if "reader_issues" in ctx:
        ctx["reader_issues"] = issues
    for u in ctx.get("unshown_cameras") or []:
        # a camera the model is not shown keeps its frames up to the same time, at its own rate
        rate = float(u.get("fps") or ctx.get("fps") or 30.0)
        u["n_frames"] = int(min(u["n_frames"], max(0, np.ceil((ctx["trimmed"]["to_s"] - u["start_s"]) * rate))))
    ctx["n_state_frames"] = keep
    ctx["duration_s"] = ctx["trimmed"]["to_s"]
    (ep / "sources.json").write_text(json.dumps(src, indent=1))
    (ep / "context.json").write_text(json.dumps(ctx, indent=1, default=str))
    return ctx


# ---------------------------------------------------------------- entry point

def plan(root: Path, grouping: dict | None = None) -> tuple[dict, list[dict]]:
    """(what was detected, one item per episode) for every format the upload holds (detect), each planned by its own
    reader, then the sensor files no episode took and the files no reader opened named in det["missing"], so nothing
    in the upload goes unmentioned."""
    root = Path(root)
    det = detect(root)
    det.setdefault("used", [])
    det.setdefault("missing", [])
    parts = det["parts"] if det["format"] == "mixed" else [det]
    if len(parts) > 1:
        det["used"].append("The upload holds " + _and_words([PART_WORDS[p["format"]] for p in parts])
                           + "; each was read.")
    items = []
    for part in parts:
        part.setdefault("used", [])
        part.setdefault("missing", [])
        items += _plan_part(part, root, grouping)
        if part is not det:
            det["used"] += part["used"]
            det["missing"] += part["missing"]
            if part.get("version"):
                det["version"] = part["version"]
    sensors = [Path(p) for p in det.get("state") or []]
    # a folder's episodes are counted across every format, so one sensor file never joins two episodes as its own
    assign_sensors(items, sensors)
    if sensors:
        # what became of each sensor file is said after conversion (convert, sensor_lines), once it is known
        taken = {Path(p) for it in items for p in (it.get("state") or []) + (it.get("state_shared") or [])}
        lost = [p for p in sensors if p not in taken]
        if lost:
            det["missing"].append(f"{_and_words([p.relative_to(root).as_posix() for p in lost])} "
                                  f"{'hold' if len(lost) != 1 else 'holds'} no camera, and no episode's footage shares "
                                  f"{'their' if len(lost) != 1 else 'its'} folder to place the data against, so "
                                  f"{'they were' if len(lost) != 1 else 'it was'} not read.")
    det["used"] += possible_duplicates(root, items)
    tables = unread_tables(root, det, items)
    if tables:
        det["missing"].append(f"{len(tables)} table{'s' if len(tables) != 1 else ''} of numbers that no episode "
                              "reads: " + _and_words(tables[:12])
                              + (f" and {len(tables) - 12} more" if len(tables) > 12 else "") + ".")
    unread = unread_files(root, det, items)
    if unread:
        det["missing"].append(f"{len(unread)} file{'s' if len(unread) != 1 else ''} that no reader opens: "
                              + _and_words(unread[:12]) + (f" and {len(unread) - 12} more" if len(unread) > 12 else "")
                              + ".")
    if len(parts) > 1:
        det["format"] = " and ".join(p["format"] for p in parts)
    return det, items


def possible_duplicates(root: Path, items: list[dict]) -> list[str]:
    """Notes on MCAP recordings with videos of the same length beside them (SAME_LENGTH_S, SAME_LENGTH_FRAC): a
    recorder can export an MCAP's cameras as mp4 files, so they may be the same footage read twice as separate
    episodes. Both are kept; the note says so."""
    out = []
    for m in (it for it in items if it.get("kind") == "mcap" and it.get("seconds")):
        same = [it for it in items if it.get("kind") == "video" and it.get("seconds")
                and item_folder(it) == item_folder(m) and _same_length([m["seconds"], it["seconds"]])]
        if same:
            vids = _and_words(sorted(Path(f).name for it in same for f in it["files"]))
            out.append(f"{Path(m['file']).relative_to(root).as_posix()} and {vids} beside it have the same length and "
                       "may be the same footage, an MCAP and its video exports; both were read, as separate episodes.")
    return out


PART_WORDS = {"lerobot": "a LeRobot dataset", "mcap": "MCAP recordings", "hdf5": "HDF5 recordings",
              "video": "plain videos"}
# what a reader reads beside the data files it plans: notes, metadata and tables (the uploader's annotations,
# annotation_tables), each video's frame times (frame_times) and archives (open_archives)
NOTE_EXT = {".json", ".jsonl", ".txt", ".md", ".yaml", ".yml", ".csv", ".tsv", ".xml", ".toml", ".ini", ".cfg"}


def _and_words(xs: list[str]) -> str:
    xs = [str(x) for x in xs]
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1] if xs else ""


NOTE_NAMES = ("annotations.json", "annotation.json", "meta.json", "instruction.txt", "task.txt", "annotations.jsonl",
              "notes.txt")


def opened_notes(items: list[dict]) -> set[Path]:
    """The notes convert_video opens beside an episode's videos: a video's own .json, .txt, .jsonl and .md, and in an
    episode folder every .json (a recorder's metadata, depth_scale_from) and the notes named NOTE_NAMES. Every file of
    a folder a dataset adapter reads (upload_adapters, OpenAoE's clip) is its adapter's."""
    out = set()
    adapters = upload_adapters("video")
    for it in items:
        if it.get("kind") != "video":
            continue
        if any(m.recognizes(it) for m in adapters):
            # a folder in a dataset's own layout is read by its adapter (OpenAoE's annotation and video_info.json)
            out |= {p for p in item_folder(it).rglob("*") if p.is_file()}
            continue
        for f in it["files"]:
            out |= {Path(f).with_suffix(x) for x in (".json", ".txt", ".jsonl", ".md")}
        d = it.get("dir")
        if d is not None:
            ep_name = it["name"].rsplit("/", 1)[-1]
            out |= set(Path(d).glob("*.json")) | {Path(d) / n for n in NOTE_NAMES} | \
                {Path(d) / f"{ep_name}{x}" for x in (".json", ".txt")}
        for dp in (it.get("depth") or {}).values():
            out |= set(Path(dp).parent.glob("*.json"))
    return out


LEROBOT_PARTS = ("meta", "data", "videos", "images")


def lerobot_own(p: Path, rdirs: list[Path]) -> bool:
    """Whether a file is one a LeRobot dataset's reader opens: under its meta, data, videos or images folder. Any other
    file of its root (a README, a dataset card, a script) is not read by us, and is named so, as the upload page names
    it as not sent."""
    return any(r in p.parents and p.relative_to(r).parts[0] in LEROBOT_PARTS for r in rdirs)


TABLE_TEXT_ROWS = 1000     # the rows of a table looked at to tell notes (a cell of text) from a table of numbers


def _table_has_text(p: Path) -> bool:
    """Whether one of a CSV or TSV table's first TABLE_TEXT_ROWS rows holds text (_has_text): notes, which
    annotation_tables reads for the rows that name an episode."""
    import csv
    try:
        with open(p, newline="", errors="replace") as fh:
            reader = csv.DictReader(fh, delimiter=table_separator(p))
            return any(_has_text(r) for _, r in zip(range(TABLE_TEXT_ROWS), reader))
    except Exception:
        return False


def unread_tables(root: Path, det: dict, items: list[dict]) -> list[str]:
    """The upload's CSV and TSV tables of numbers that no episode reads, each "path (why)": a table is read as signals
    beside the videos of its episode (plan_video "series"), so one in a folder with no video episode, or whose name
    gives none of the takes of the video episodes beside it, is read nowhere; it had gone unmentioned, as
    unread_files counts every table as read by annotation_tables, which reads only the rows that hold text. A
    table of a LeRobot dataset is the dataset's."""
    root = Path(root)
    parts = det["parts"] if det.get("parts") else [det]
    rdirs = [Path(r) for p in parts for r in p.get("roots") or []]
    taken = {Path(p) for it in items for p in it.get("series") or []}
    video_dirs = {item_folder(it) for it in items if it.get("kind") == "video"}
    out = []
    for p in files_under(root):
        if p.suffix.lower() not in (".csv", ".tsv") or p in taken or lerobot_own(p, rdirs):
            continue
        if _table_has_text(p):
            continue
        why = ("its name gives none of the takes of the video episodes in its folder" if p.parent in video_dirs else
               "no video episode is in its folder, where a table is read as the episode's signals")
        out.append(f"{p.relative_to(root).as_posix()} ({why})")
    return out


def unread_files(root: Path, det: dict, items: list[dict]) -> list[str]:
    """The upload's files (relative paths) that no reader opens. Opened are the files of a LeRobot dataset, videos,
    MCAP and HDF5 files (a sensor file no episode takes is named apart), archives, tables (annotation_tables reads
    every CSV, TSV and JSON Lines file), a video's frame times (a .npy whose name says time beside it) and the notes
    convert_video reads beside an episode (opened_notes). Any other file, a notes file included, is listed, and so is a
    file of a LeRobot dataset's root that its reader does not open (lerobot_own)."""
    root = Path(root)
    parts = det["parts"] if det.get("parts") else [det]
    rdirs = [Path(r) for p in parts for r in p.get("roots") or []]
    vid_dirs = {Path(p).parent for p in files_under(root) if p.suffix.lower() in VIDEO_EXT}
    notes = {p.resolve() for p in opened_notes(items)}
    out = []
    for p in files_under(root):
        x = p.suffix.lower()
        if lerobot_own(p, rdirs) or x in VIDEO_EXT or x in H5_EXT or x in TABLE_EXT or x == ".mcap" \
                or ARCHIVE_RE.search(p.name) or p.resolve() in notes:
            continue
        if x == ".npy" and p.parent in vid_dirs and any(t.startswith(("time", "stamp")) for t in tokens(p.stem)):
            continue
        out.append(p.relative_to(root).as_posix())
    return out


def _plan_part(det: dict, root: Path, grouping: dict | None) -> list[dict]:
    """The items of one format of the upload (detect), with its notes added to det."""
    if det["format"] == "lerobot":
        items, used, missing, roots = plan_lerobot(det, root)
        det["used"] += used
        det["missing"] += missing
        det["version"] = ", ".join(sorted({r["version"] for r in roots}))
        return items
    if det["format"] == "mcap":
        items = plan_mcap(det, root)
        det["used"].append(f"{len(items)} MCAP files, one episode each." if len(items) != 1 else "1 MCAP file, one episode.")
        if any(it["seconds"] is None for it in items):
            det["missing"].append("Some MCAP files end early, before their index; they were scanned message by message.")
    elif det["format"] == "hdf5":
        items = plan_hdf5(det, root)
        nf = len(det["files"])
        det["used"].append(f"{nf} HDF5 file{'s' if nf != 1 else ''}, "
                           f"{len(items)} episode{'s' if len(items) != 1 else ''}.")
    else:
        items = plan_video(det, root, grouping)
        det["used"].append(f"{len(items)} video episodes." if len(items) != 1
                           else "1 video episode.")
    _mark_packaging(det, items)
    return items


def item_folder(it: dict) -> Path:
    """The folder an episode's files sit in: a video episode's first video's, an MCAP's or an HDF5 file's own."""
    return Path(it["files"][0]).parent if it.get("files") else Path(it["file"]).parent


def item_take(it: dict) -> str:
    """The take an episode's file name gives (name_parts): cam_high_ep3 gives ep3, run2 gives run2."""
    return name_parts(Path(it["files"][0] if it.get("files") else it["file"]).stem)["take"]


def assign_sensors(items: list[dict], sensors: list[Path]) -> None:
    """Each episode's sensor files (MCAP or HDF5 with no camera, detect), as table_signals pairs a table: it["state"]
    holds the sensor files of its folder when the folder holds that one episode, and in a folder of several episodes
    the files whose name gives its take (glove_run2.h5 goes with run2.mp4). A file of such a folder whose name gives
    no episode's take is it["state_shared"] of every episode there: its own clock places it on the one it recorded,
    and without a clock in common it is listed with the reason (split_sensors). The episodes of a folder are counted
    across every format, so a folder of run.mp4 and cam.h5 holds two."""
    per_folder: dict[Path, list[dict]] = {}
    for it in items:
        if it.get("kind") in ("video", "mcap", "hdf5"):    # a LeRobot dataset's own files are its own
            per_folder.setdefault(item_folder(it), []).append(it)
    for it in items:
        it.setdefault("state", [])
        it.setdefault("state_shared", [])
    for p in sensors:
        eps = per_folder.get(p.parent) or []
        if len(eps) == 1:
            eps[0]["state"].append(p)
            continue
        take = name_parts(p.stem)["take"]
        mine = [it for it in eps if take and item_take(it) == take]
        for it in mine:
            it["state"].append(p)
        if not mine:
            for it in eps:
                it["state_shared"].append(p)


SENSOR_TIMES_KEPT = 8       # the sensor files whose times are kept once read: an episode's own and shared ones


def sensor_times(p: Path) -> np.ndarray | None:
    """A sensor file's own times in seconds, sorted: an MCAP's message log times (from its summary's first and last
    time and count, or scanned when it has no summary or its summary cannot be read, a file cut short, whose
    messages before the cut are then its times), an HDF5 file's longest clock (h5_streams); None when it has none.
    The times of a file are read once while it is unchanged (_file_times, by its inode, size, and modification and
    change times, so a file copied over it at the same size and modification time is read again), since placing it,
    flagging its cut and placing it from both starts each ask for them, and a cut file is scanned whole; they come
    read only, so no caller changes them for the next."""
    try:
        st = os.stat(p)
    except OSError:
        return None
    return _file_times(str(Path(p).resolve()), st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


@functools.lru_cache(maxsize=SENSOR_TIMES_KEPT)
def _file_times(p: str, inode: int, size: int, mtime_ns: int, ctime_ns: int) -> np.ndarray | None:
    """sensor_times of the file at p, as it is with this inode, size, and modification and change times."""
    t = _read_file_times(Path(p))
    if t is not None:
        t.setflags(write=False)
    return t


def _read_file_times(p: Path) -> np.ndarray | None:
    try:
        if Path(p).suffix.lower() == ".mcap":
            from mcap.reader import make_reader
            try:
                with open(p, "rb") as fh:
                    s = make_reader(fh).get_summary()
            except Exception:
                s = None                  # cut short: no footer to find the summary by
            st = s.statistics if s is not None else None
            if st and st.message_count:
                return np.linspace(st.message_start_time, st.message_end_time, max(int(st.message_count), 1)) / 1e9
            t = np.sort(np.array([m.log_time for _, _, m in _mcap_stream(Path(p))], dtype=np.float64)) / 1e9
            return t if len(t) else None
        import h5py
        with h5py.File(p, "r") as f:
            clocks = [np.sort(t[np.isfinite(t)]) for t in h5_streams(f, "")["clock"].values() if len(t)]
        clocks = [t for t in clocks if len(t)]
        return max(clocks, key=len) if clocks else None
    except Exception:
        return None


def no_time_why(p: Path) -> str:
    """Why a sensor file gave no times (sensor_times): it could not be opened at all, it is cut short before its first
    message (sensor_cut), or it opened and holds none."""
    try:
        if Path(p).suffix.lower() == ".mcap":
            with open(p, "rb") as fh:
                if fh.read(len(MCAP_MAGIC)) != MCAP_MAGIC:
                    raise ValueError("not an MCAP file")
            if sensor_cut(p):
                return "it is cut short before its first message, so nothing in it could be read"
        else:
            import h5py
            with h5py.File(p, "r"):
                pass
    except Exception:
        return "it could not be opened; it may be damaged or cut short"
    return NO_TIME_WHY


NO_TIME_WHY = "no time in it to place it on the footage by"     # a whole sensor file that holds no time (no_time_why)


def unreadable_sensors(unplaced: list) -> list[tuple[str, str]]:
    """[(name, what happened to it)] of an episode's sensor MCAP files that could not be read at all (unplaced, as
    split_sensors gives them; no_time_why: it could not be opened, or it is cut short before its first message). A
    whole file that holds no message (NO_TIME_WHY) holds nothing a state could lack, and is not one."""
    return [(Path(p).name, why.removeprefix("it ")) for p, why in unplaced
            if Path(p).suffix.lower() == ".mcap" and why != NO_TIME_WHY and sensor_times(p) is None]


def damaged_files(lost: list) -> list[tuple[str, str]]:
    """[(name, what happened to it)] of the MCAP files damaged inside whose channels could not be read where the
    footage needs them (lost, as mcap_joint_streams appends them: path, channels, whether any message was read)."""
    head = "is damaged inside, though its index is whole, so "
    return [(Path(p).name, head + (f"nothing on {_and_words(chans)} could be read" if read else
                                   "none of its messages could be read")) for p, chans, read in lost]


def unread_sensors_note(bad: list[tuple[str, str]], others: str | None) -> StateNote | None:
    """The state note of an episode with a sensor file whose content is unknown (bad: each file's name and what
    happened to it, unreadable_sensors and damaged_files), or None. What such a file held is unknown, so a state of the
    other files (others, as the note names them, or None when there are none) may lack an arm it recorded, and a state
    of one arm on a two arm rig tells the model one arm works: the episode has no arm state and is labelled from the
    cameras, every channel of the other files kept as a signal ("unreadable"). The note does not say what the file
    held, since a tactile pad, a leader's commands or a log is no arm."""
    if not bad:
        return None
    note = ("Labelled from the cameras, because " + "; ".join(f"{name} {why}" for name, why in bad)
            + f". What {'they record' if len(bad) > 1 else 'it records'} is unknown")
    if others:
        note += f", so {others} are not read as the arm state, and their channels are given as signals"
    return StateNote(note + ".", "unreadable")


def h5_state_arrays(p: Path) -> list[str]:
    """The arrays of an HDF5 file that are named as the state (H5_STATE_NAME, outside an action group)."""
    import h5py
    try:
        with h5py.File(p, "r") as f:
            return sorted(s["name"] for s in h5_streams(f, "")["signal"]
                          if H5_STATE_NAME.search(s["name"]) and not H5_ACTION_GROUP.search(s["name"]))
    except Exception:
        return []


def assumed_arms(paths: list[Path], clocked: dict) -> list[tuple[str, list[str]]]:
    """[(name, its arm channels)] of the sensor files placed from both starts (split_sensors) that record a working
    arm: an MCAP channel of an arm's joints (mcap_joint_streams) that is not a command channel (ACTION_TOPIC, which
    gives the action, never the state) and not a third arm beside a left and a right arm (third_arms, with the arms
    of the files on the footage's clock, clocked), or an HDF5 array named as the state (h5_state_arrays). Such an arm
    is never the recorded state, so a state of the other files would lack it (state_blockers)."""
    out = []
    for p in map(Path, paths):
        if p.suffix.lower() == ".mcap":
            own = {f"{k} ({p.name})": v for k, v in mcap_joint_streams([p]).items()}
            third = set(third_arms({**clocked, **own}))
            arms = sorted({_topic(own, k) for k in own if k not in third and not ACTION_TOPIC.search(_topic(own, k))})
        else:
            arms = h5_state_arrays(p)
        if arms:
            out.append((p.name, arms))
    return out


def assumed_arms_note(arms: list[tuple[str, list[str]]], others: str | None, lead: bool = True) -> StateNote | None:
    """The state note of an episode with a working arm only on a clock placed from both starts (assumed_arms), beside
    files on the footage's clock (others, as the note names them): that arm is never the state, and a state of the
    others alone would lack it, so the episode has no arm state ("assumed_clock"). lead starts the note as every state
    note starts, and is left off when it follows another (state_blockers)."""
    if not arms:
        return None
    many = len(arms) > 1
    note = (_and_words([f"{name} records an arm ({', '.join(chans)})" for name, chans in arms])
            + f" on {'clocks' if many else 'a clock'} the footage does not share, so "
            + f"{'they were' if many else 'it was'} placed on the footage from both starts, an assumed alignment that "
            "is never read as the arm state.")
    if others:
        note += (f" A state of {others} alone would lack {'those arms' if many else 'that arm'}, so every arm channel "
                 "is given as a signal.")
    return StateNote(("Labelled from the cameras, because " if lead else "Also, ") + note, "assumed_clock")


def state_blockers(bad: list[tuple[str, str]], arms: list[tuple[str, list[str]]],
                   others: str | None) -> StateNote | None:
    """Why no arm state is read beside an episode's sensor files, or None: a file whose content is unknown (bad,
    unread_sensors_note) or a working arm only on an assumed clock (arms, assumed_arms_note). An arm state is given
    only when every arm the episode's sensor files may hold was read on the footage's clock, since a state that lacks
    one tells the model the arms it has did all the work. Both are named when both apply; the reason is the first."""
    first = unread_sensors_note(bad, others)
    then = assumed_arms_note(arms, others, lead=first is None)
    if first and then:
        return StateNote(f"{first} {then}", first.why)
    return first or then


def sensor_cut(p: Path) -> bool:
    """Whether an MCAP sensor file is cut short: a whole one ends with the MCAP magic after its footer."""
    if Path(p).suffix.lower() != ".mcap":
        return False
    try:
        with open(p, "rb") as fh:
            fh.seek(0, 2)
            if fh.tell() < 2 * len(MCAP_MAGIC):
                return True
            fh.seek(-len(MCAP_MAGIC), 2)
            return fh.read() != MCAP_MAGIC
    except OSError:
        return False


def recorder_clock(t) -> bool:
    """Whether times count on a recorder's clock, from boot or the epoch (far from zero in their own steps), so the
    times of another file on that clock can be compared with them. Times that count from their own start (0, 0.033,
    ...) share no clock with anything: two such files are lined up only by assuming they started together."""
    if t is None:
        return False
    ok, step, size = _clock_facts(np.asarray(t, dtype=np.float64))
    return len(ok) > 1 and step > 0 and size > FAR_FROM_ZERO_STEPS * step


def split_sensors(item: dict, q, clocked: bool) -> tuple[list[Path], list[Path], list[tuple[Path, str]]]:
    """(placed by their own clock, placed from both starts, not placed with why) for an episode's sensor files
    (assign_sensors): its own files (item["state"]) and those it shares with the other episodes of its folder
    (item["state_shared"]). A file is placed by its clock when the footage's frame times q are on a recorder's clock
    (clocked: capture times, an MCAP's log times, an HDF5 camera's clock) and the file's times are on one too and
    overlap the footage; when both are on a recorder's clock and the file covers none of the footage, it recorded
    something else and is listed as recorded outside the footage. Otherwise an own file is placed from both starts, an
    alignment that is assumed (its signals are marked so and never read as the arm state), while a shared file is
    listed on the episode with the reason: it could be any of the folder's episodes' recording, and placing it from
    both starts on each would be a guess."""
    by_clock, assumed, unplaced = [], [], []
    on_clock = bool(clocked) and q is not None and recorder_clock(q)
    own = [Path(x) for x in item.get("state") or []]
    for p in own + [Path(x) for x in item.get("state_shared") or [] if Path(x) not in own]:
        t = sensor_times(p)
        if t is None:
            unplaced.append((p, no_time_why(p)))
        elif on_clock and recorder_clock(t) and overlaps(t, np.asarray(q, dtype=np.float64)):
            by_clock.append(p)
        elif on_clock and recorder_clock(t):
            # both clocks are real and comparable: a file that covers none of the footage recorded something else
            unplaced.append((p, outside_words(t, np.asarray(q, dtype=np.float64))))
        elif p in own:
            assumed.append(p)
        else:
            unplaced.append((p, "several episodes share its folder, its name gives none of their takes, and it shares "
                                "no clock with this episode's footage to tell whether it recorded it"))
    return by_clock, assumed, unplaced


def note_sensors(extra: dict, signals: Signals, by_clock: list, assumed: list, unplaced: list,
                 q: np.ndarray | None = None) -> None:
    """What became of an episode's sensor files: their names (source "sensors", which says sensor data was read or
    tried, label/episode.py), how each was placed or why it was not (source "sensor_files", which convert gathers into
    the report), and each file not placed listed among the signals left out with the reason. A placed MCAP file cut
    short (sensor_cut) is read up to the cut, a data issue (sensor_file_cut) giving the span of the footage its
    messages cover (span_on_footage; q, the footage's frame times on the file's clock when it was placed by its clock,
    and from its own start when it was placed from both starts)."""
    files = [*by_clock, *assumed, *(p for p, _ in unplaced)]
    if not files:
        return
    for p in [*by_clock, *assumed]:
        if not sensor_cut(p):
            continue
        t = sensor_times(p)
        issue = {"kind": "sensor_file_cut", "what": f"{Path(p).name} is cut short, so only the messages written "
                                                    "before the cut were read"}
        if t is not None:
            # the span its messages cover, in seconds of the footage and within it (span_on_footage)
            zero = float(q[0]) if p in by_clock and q is not None and len(q) else float(t[0])
            length = float(q[-1] - q[0]) if q is not None and len(q) else float(t[-1] - t[0])
            t0, t1 = span_on_footage(float(t[0]), float(t[-1]), zero, length)
            issue.update(t0_s=t0, t1_s=t1, what=issue["what"] + f"; its messages cover {t0:.1f} s to {t1:.1f} s of "
                                                                 "the footage")
        signals.issues.append(issue)
    extra.setdefault("source", {})["sensors"] = [Path(p).name for p in files]
    how = {Path(p).name: "placed by its own clock" for p in by_clock}
    how.update({Path(p).name: "placed from both starts" for p in assumed})
    how.update({Path(p).name: f"not placed: {why}" for p, why in unplaced})
    extra["source"]["sensor_files"] = how
    signals.left_out += [(Path(p).name, why) for p, why in unplaced]


ASSUMED_START = ("{} (from {}) was placed on the footage from both starts, since the two share no clock, so its "
                 "alignment assumes a common start")
ALIGNED_ASSUMED = "assumed start"     # a signal's meta "aligned_by" when it was placed from both starts


def mark_assumed(sig: Signals, extra: dict, source: str) -> Signals:
    """Every signal of sig marked as placed from both starts: meta "aligned_by" ALIGNED_ASSUMED, which the prompt and
    the board state beside it, and one data issue per signal (signal_alignment_assumed) naming it and its source."""
    for k in sig:
        sig.meta.setdefault(k, {})["aligned_by"] = ALIGNED_ASSUMED
        add_issue(extra, "signal_alignment_assumed", ASSUMED_START.format(k, source), signal=k)
    return sig


def sensors_from_start(paths: list[Path], t_video: np.ndarray, extra: dict) -> Signals:
    """The signals of sensor files that share no clock with the footage, each file placed from both starts as
    table_signals places a table: its own first reading at the anchor camera's first frame (t_video, the anchor's
    seconds from its first frame). Every signal so placed is marked (mark_assumed), and an arm's channels stay
    signals, never the recorded state: a state is read only where the alignment is recorded."""
    out = Signals()
    t_video = np.asarray(t_video, dtype=np.float64)
    for p in paths:
        p = Path(p)
        t = sensor_times(p)
        if t is None:
            out.left_out.append((p.name, no_time_why(p)))
            continue
        q = float(t[0]) + t_video
        got = mcap_signals([p], q) if p.suffix.lower() == ".mcap" else h5_file_signals([p], q, len(q))
        merge_signals(out, mark_assumed(got, extra, p.name))
    return out


TABLE_EXT = {".csv", ".tsv", ".jsonl"}
TABLE_MAX_BYTES = 20_000_000
TABLE_MAX_ROWS_PER_EPISODE = 20


def annotation_tables(root: Path) -> list[tuple[str, list[dict]]]:
    """[(file name, rows)] of every table in the upload (CSV, TSV, JSON Lines): a dataset's per-episode metadata kept
    beside its data (OpenTouch's final_annotations/eat_ygf_p1_merged.csv, one row per clip with its object, action,
    grip and description). A table over TABLE_MAX_BYTES had been ignored; its rows that hold text are read now, up to
    TABLE_NOTE_ROWS_MAX of them."""
    import csv
    out = []
    for p in files_under(root):
        if p.suffix.lower() not in TABLE_EXT:
            continue
        big = p.stat().st_size > TABLE_MAX_BYTES
        try:
            if p.suffix.lower() == ".jsonl":
                rows = [r for r in read_jsonl(p) if isinstance(r, dict)] if not big else \
                    [r for r in _jsonl_rows(p) if _has_text(r)][:TABLE_NOTE_ROWS_MAX]
            else:
                with open(p, newline="", errors="replace") as fh:
                    reader = csv.DictReader(fh, delimiter=table_separator(p))
                    # a table over TABLE_MAX_BYTES is read row by row for the rows that hold text, the ones that can
                    # name an episode; a large table of numbers is a recording (table_signals), not notes
                    rows = list(reader) if not big else [r for _, r in zip(range(TABLE_NOTE_ROWS_MAX), (
                        r for r in reader if _has_text(r)))]
        except Exception:
            continue
        if rows:
            out.append((p.relative_to(root).as_posix(), rows))
    return out


TABLE_NOTE_ROWS_MAX = 100_000


def _jsonl_rows(p: Path):
    with open(p, errors="replace") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict):
                yield r


def _has_text(row: dict) -> bool:
    """Whether a table row holds a cell of text that is not a number (a name, a task, a note). A number written with a
    decimal comma (DECIMAL_COMMA, 0,033) or with dots that can only be between thousands (THOUSANDS_PROOF, 1.234,56
    or 1.100.000) is a number, as the table reader reads it in any table (column_numbers), never a note."""
    for v in row.values():
        if isinstance(v, str) and v.strip() and not (DECIMAL_COMMA.match(v) or THOUSANDS_PROOF.match(v)):
            try:
                float(v)
            except ValueError:
                return True
    return False


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
    try:
        det, items = plan(root, grouping)
    except ValueError as e:
        if not opened:
            raise
        # what the archives said (a member left out, an archive cut short) is why the upload holds nothing readable
        if not files_under(root):
            raise ValueError("nothing in the upload could be unpacked: " + " ".join(opened)) from e
        raise ValueError(f"{e}. {' '.join(opened)}") from e
    out.mkdir(parents=True, exist_ok=True)
    total = 0.0
    report = {"format": det["format"], "version": det.get("version"), "rig": rig,
              "episodes": [], "skipped": [], "failed": [], "notes": [],
              "used": opened + list(det.get("used") or []), "missing": list(det.get("missing") or [])}
    if det.get("packaging"):
        report["packaging"] = det["packaging"]
    tables = annotation_tables(root)
    past = f"past the first {max_seconds / 60:g} minutes"
    full = False
    for it in items:
        known = it["seconds"] if it["seconds"] is not None else 0.0      # unknown until converted; measured below
        first = not report["episodes"]
        if full or (total + known > max_seconds + 1 and not first):
            # an episode that does not fit what is left of the minutes is listed, and the later ones are still tried,
            # in upload order, as the upload page chooses them (read.js chooseEpisodes)
            report["skipped"].append({"name": it["name"], "why": past})
            continue
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
            # the measured length is longer than the file's header claimed: never label past the cap, and never delete
            # what fits under it; its first minutes up to the cap are labelled, as the first episode's are
            room = max_seconds - total
            if room >= TRIM_MIN_S:
                ctx = trim_episode(out / ctx["episode_id"], room)
                report["used"].append(f"{it['name']} is {secs / 60:.1f} minutes long, longer than its file's header "
                                      f"says; its first {float(ctx['duration_s']):.1f} seconds were labelled, up to "
                                      f"the {max_seconds / 60:g} minute limit.")
                secs = float(ctx["duration_s"])
                total += secs
                report["episodes"].append(episode_row(it, ctx, secs))
                full = True               # the minutes are used up: every later episode is listed
            else:
                import shutil
                shutil.rmtree(out / ctx["episode_id"], ignore_errors=True)
                report["skipped"].append({"name": it["name"], "why": past})
            continue
        total += secs
        report["episodes"].append(episode_row(it, ctx, secs))
        # within a second of the limit the minutes are used up, as the upload page counts them (read.js
        # chooseEpisodes): a later episode is listed, never converted and then deleted for not fitting
        full = full or total >= max_seconds - 1
    used, missing = sensor_lines(out, report["episodes"])
    report["used"] += used
    report["missing"] += missing
    notes = sorted({e["state_note"] for e in report["episodes"] if e.get("state_note")})
    report["notes"] += notes
    report["seconds"] = round(total, 2)
    ids = [e["episode_id"] for e in report["episodes"]]
    measure_gripper_range(out, ids)
    measure_depth_ranges(out, ids)
    measure_signal_scales(out, ids)
    measure_contacts(out, ids)
    return report


def sensor_lines(out: Path, episodes: list[dict]) -> tuple[list[str], list[str]]:
    """(used, missing) report lines saying exactly what became of each sensor file (source "sensor_files", written by
    note_sensors): on how many episodes it was placed by its own clock, placed from both starts, or listed with the
    reason. A file listed on every episode it went with is in missing, any other in used."""
    by: dict[str, dict[str, list[str]]] = {}
    for e in episodes:
        try:
            ctx = json.loads((Path(out) / e["episode_id"] / "context.json").read_text())
        except Exception:
            continue
        for name, how in ((ctx.get("source") or {}).get("sensor_files") or {}).items():
            by.setdefault(name, {}).setdefault(how, []).append(e["name"])
    used, missing = [], []
    count = lambda xs: f"{len(xs)} episode{'s' if len(xs) != 1 else ''}"
    for name, hows in sorted(by.items()):
        parts = []
        for how, eps in sorted(hows.items()):
            if how == "placed by its own clock":
                parts.append(f"placed by its own clock on {count(eps)}")
            elif how == "placed from both starts":
                parts.append(f"placed from both starts on {count(eps)}, so its alignment there is assumed")
            else:
                parts.append(f"listed on {count(eps)} and not read, since {how.split(': ', 1)[-1]}")
        line = f"{name}, a sensor file with no camera: " + "; ".join(parts) + "."
        (missing if all(h.startswith("not placed") for h in hows) else used).append(line)
    return used, missing


TRIM_MIN_S = 1.0       # an episode is trimmed to the room left under the cap only when at least this much is left


def episode_row(it: dict, ctx: dict, secs: float) -> dict:
    """An accepted episode's line in the conversion report."""
    return {"name": it["name"], "episode_id": ctx["episode_id"], "seconds": round(secs, 2),
            "cameras": {v: c.get("name") for v, c in ctx["cameras"].items()},
            "state_kind": ctx["state_kind"], "state_note": ctx.get("state_note"),
            "instruction": ctx.get("instruction"), "fps": ctx.get("fps"),
            "unsplit": bool(ctx.get("unsplit")), "packaging": ctx.get("packaging")}


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
                # a wide signal is summarised in the prompt, and a rest per value of thousands is not one to show
                if s["key"] in z.files and int(s.get("dims") or 0) <= SIGNAL_MAX_VALUES:
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

