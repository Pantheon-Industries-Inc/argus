"""Where the repo finds ffmpeg and ffprobe. Every caller (the reader, board clips, the board server, the static build,
the hand pose overlay) uses these two, so a probe, a frame and a clip all come from one install.

  ffmpeg    $FFMPEG, else ffmpeg on PATH, else the usual install places
  ffprobe   $FFPROBE, else the ffprobe beside that ffmpeg, else ffprobe on PATH, else the usual install places

The install places are there because a server started outside a login shell may not have Homebrew or /usr/local on
its PATH. A variable that is set but names no program is an error, never skipped for another copy. A tool that cannot
be found raises Missing, a FileNotFoundError, so a step that treats a missing tool as the machine's fault
(board/clips.py machine_stopped) still does, with a message that says how to fix it.

Cutting a clip needs ffmpeg 6.1 or newer, because a clip keeps its source timestamps with -enc_time_base demux
(board/clips.py video_args). Every clip fails on an older ffmpeg, with "Invalid time base: demux", so ffmpeg_for_clips
stops the step first and names the version it found. Reading frames and probing work on older installs, so ffmpeg() and
ffprobe() do not check it."""
from __future__ import annotations

import errno
import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

INSTALL_DIRS = ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin")
# The first release whose -enc_time_base takes "demux": fftools/ffmpeg_mux_init.c parses the word from tag n6.1, and at
# n6.0 and n5.1.6 hands it to av_parse_ratio, which refuses it
FFMPEG_MINIMUM = (6, 1)


class Missing(FileNotFoundError):
    """ffmpeg or ffprobe cannot be found."""

    def __init__(self, tool: str, why: str):
        super().__init__(errno.ENOENT, f"{tool} not found: {why}", tool)


class TooOld(Missing):
    """ffmpeg is there and is older than clip cutting needs. Still the machine's fault, like a missing tool."""

    def __init__(self, path: str, found: tuple):
        need = ".".join(map(str, FFMPEG_MINIMUM))
        FileNotFoundError.__init__(self, errno.ENOENT, f"ffmpeg {'.'.join(map(str, found))} at {path} is too old: "
                                   f"cutting clips needs ffmpeg {need} or newer (-enc_time_base demux). "
                                   "Install a newer one, and set FFMPEG if it is not on PATH", path)


def _runnable(path) -> bool:
    return bool(path) and os.path.isfile(path) and os.access(path, os.X_OK)


def _from_env(var: str, tool: str) -> str | None:
    given = os.environ.get(var)
    if not given:
        return None
    if not _runnable(given):
        raise Missing(tool, f"${var} is {given}, which is not a program")
    return given


def _installed(tool: str) -> str | None:
    found = shutil.which(tool)
    if found:
        return found
    return next((str(Path(d) / tool) for d in INSTALL_DIRS if _runnable(str(Path(d) / tool))), None)


def ffmpeg() -> str:
    """The ffmpeg every caller runs (module docstring)."""
    found = _from_env("FFMPEG", "ffmpeg") or _installed("ffmpeg")
    if not found:
        raise Missing("ffmpeg", "put it on PATH or set FFMPEG")
    return found


def ffprobe() -> str:
    """The ffprobe every caller runs (module docstring)."""
    found = _from_env("FFPROBE", "ffprobe")
    if found:
        return found
    try:
        beside = str(Path(ffmpeg()).with_name("ffprobe"))
    except Missing:
        beside = None
    found = beside if _runnable(beside) else _installed("ffprobe")
    if not found:
        raise Missing("ffprobe", "put it on PATH or set FFPROBE")
    return found


@lru_cache(maxsize=16)
def _version_of(path: str, stamp: tuple) -> tuple | None:
    # stamp is only part of the key, so an ffmpeg upgraded in place is read again
    try:
        said = subprocess.run([path, "-version"], capture_output=True, text=True, errors="replace", timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    found = re.match(r"ffmpeg version n?(\d+)\.(\d+)", said)
    return (int(found[1]), int(found[2])) if found else None


def ffmpeg_version(path: str) -> tuple | None:
    """(major, minor) of the ffmpeg at path. None when its version line names no release (a git build says
    N-113007-g8d24a28d06) or it cannot be run: nothing shows it is too old, so the cut itself decides."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return _version_of(path, (st.st_size, st.st_mtime_ns))


def ffmpeg_for_clips(path: str | None = None) -> str:
    """The ffmpeg at path, or the one every caller runs, once it is known not to be too old to cut clips (module
    docstring). Raises TooOld, or Missing when there is none."""
    path = path or ffmpeg()
    found = ffmpeg_version(path)
    if found is not None and found < FFMPEG_MINIMUM:
        raise TooOld(path, found)
    return path


def ffmpeg_or_none() -> str | None:
    """ffmpeg, or None when there is none, for a caller that works without it (the board server's frames)."""
    try:
        return ffmpeg()
    except Missing:
        return None


def ffprobe_or_none() -> str | None:
    """ffprobe, or None when there is none."""
    try:
        return ffprobe()
    except Missing:
        return None
