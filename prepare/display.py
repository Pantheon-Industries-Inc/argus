"""How a video file is meant to be shown: its display matrix (a rotation, and a mirror when the matrix flips) and its
sample aspect ratio (pixels that are not square, as DV and anamorphic footage store them), read with ffprobe, and the
size those give.

Every copy of a picture takes its shown size from here: the reader's camera sizes (prepare/formats.py probe), the
model's frames (label/frames.py upright), the board's clips (board/clips.py) and the hand tracker
(board/hand_pose/core.py). A file's own declaration is followed exactly; nothing is assumed about orientation or pixel
shape. A sample aspect ratio so close to 1 that the shown width rounds to the stored one (Egocentric-100K's 512:513
on 456 px) leaves the size as stored."""
from __future__ import annotations

import json
import os
import subprocess
from fractions import Fraction

from label.atomic import cached_while_unchanged


def even(x: float) -> int:
    return max(2, int(2 * round(x / 2)))


def geometry(path: str, ffprobe: str | None = None) -> dict:
    """{"stored": (w, h), "sar": Fraction, "rotation": degrees as ffprobe reports it, "matrix": [a, b, c, d] or None,
    "mirror": bool} for the first video stream; a file ffprobe cannot read is taken as stored (sar 1, no rotation).
    Read once per file version (its whole stat, label/content_identity.py: an edit that keeps the size and puts the
    modification time back still changes the change time), and kept only for the version it was read from."""
    try:
        version = _version(path)
    except OSError:
        version = None
    return dict(_geometry(str(path), version, ffprobe))


def _version(path) -> tuple:
    # the whole stat, as label/content_identity.py stat_version keys it (the Modal image carries this file alone)
    st = os.stat(path)
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


@cached_while_unchanged(4096, lambda path, version, ffprobe: version is not None and _version(path) == version)
def _geometry(path: str, version, ffprobe: str | None) -> dict:
    try:
        # the one ffprobe every caller runs (prepare/tools.py); none found reads as no size, like one that fails
        from prepare import tools
        exe = ffprobe or tools.ffprobe()
        r = subprocess.run([exe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height,sample_aspect_ratio:stream_tags=rotate:"
                            "stream_side_data=rotation,displaymatrix", "-of", "json", str(path)],
                           capture_output=True, text=True, timeout=60)
        st = (json.loads(r.stdout or "{}").get("streams") or [{}])[0]
    except (OSError, subprocess.SubprocessError, ValueError):
        st = {}
    w, h = int(st.get("width") or 0), int(st.get("height") or 0)
    sar = Fraction(1)
    num, _, den = str(st.get("sample_aspect_ratio") or "1:1").partition(":")
    if num.isdigit() and den.isdigit() and int(num) > 0 and int(den) > 0:
        sar = Fraction(int(num), int(den))
    rot, matrix = 0.0, None
    for sd in st.get("side_data_list") or []:
        if "rotation" in sd:
            rot = float(sd["rotation"])
        if sd.get("displaymatrix"):
            vals = [int(x) for line in str(sd["displaymatrix"]).splitlines() if ":" in line
                    for x in line.split(":", 1)[1].split()]
            if len(vals) == 9:
                matrix = [vals[0], vals[1], vals[3], vals[4]]
    if not rot and (st.get("tags") or {}).get("rotate"):
        rot = float(st["tags"]["rotate"])
    mirror = bool(matrix) and (matrix[0] * matrix[3] - matrix[1] * matrix[2]) < 0
    return {"stored": (w, h), "sar": sar, "rotation": rot, "matrix": matrix, "mirror": mirror}


def square_size(g: dict) -> tuple:
    """(w, h) of the stored frame with its pixels made square, before any rotation: the width scaled by the sample
    aspect ratio, rounded to even, unless that rounds back to the stored width."""
    w, h = g["stored"]
    if g["sar"] != 1 and w:
        ws = even(w * g["sar"])
        if ws != even(w):
            return ws, h
    return w, h


def quarter_turn(g: dict) -> bool:
    return int(round(g["rotation"])) % 180 == 90


def shown_size(g: dict) -> tuple:
    """(w, h) as a player shows the file: square pixels, then the display rotation."""
    w, h = square_size(g)
    return (h, w) if quarter_turn(g) else (w, h)


def needs_resample(g: dict) -> bool:
    """Whether the stored pixels must be resampled to show the file right (pixels that are not square and do not
    round away)."""
    return square_size(g) != tuple(g["stored"])
