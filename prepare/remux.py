"""A camera's recorded frames written into an mp4 whose timestamps are the frames' real capture times.

Datasets that ship their cameras as messages in an MCAP (ABC-130k, RealOmni, Gen-HumanEgo) give each frame the
time it was recorded, and no frame rate. A camera does not run at exactly its nominal rate: it can drop frames or
run slightly fast or slow. So the mp4 must carry each frame's own recorded time as its pts, never a constant rate
chosen for it: the labeller shows the model each frame at that time, and the board plays the clip by it, so a
constant rate would place every frame after a drop, and every frame of a camera running off its rate, at the wrong
time. remux() copies the Annex-B frames packet for packet (no re-encode), sets each one's pts to its capture time
and checks that the written file carries exactly those times.
"""
from __future__ import annotations

import os
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np

TIME_BASE_DEN = 1_000_000   # microseconds: some ABC frames are stamped only 1 us apart
NOMINAL_FPS = 30            # only a one-frame tolerance for the muxer's own shift


def remux(frames: list[bytes], t_rel_s: np.ndarray, fmt: str, out: Path) -> np.ndarray:
    """Copy Annex-B frames packet for packet into an mp4 whose pts are the frames' real capture times
    (rounded to the microsecond time base). No re-encode. Returns the pts in the written file, one per frame."""
    import av
    with tempfile.NamedTemporaryFile(suffix="." + fmt, dir=out.parent, delete=False) as f:
        for b in frames:
            f.write(b)
        raw = f.name
    pts = np.round(np.asarray(t_rel_s, dtype=np.float64) * TIME_BASE_DEN).astype(np.int64)
    # a frame stamped at or before the previous one (a recorder glitch, counted in stream_checks) is
    # placed 1 us after it so the mp4 stays valid; no frame is dropped or reordered
    for i in range(1, len(pts)):
        if pts[i] <= pts[i - 1]:
            pts[i] = pts[i - 1] + 1
    try:
        with av.open(raw, format={"h264": "h264", "h265": "hevc"}[fmt]) as src, av.open(str(out), "w") as dst:
            ist = src.streams.video[0]
            ost = dst.add_stream_from_template(ist)
            ost.time_base = Fraction(1, TIME_BASE_DEN)
            i = 0
            for pkt in src.demux(ist):
                if not pkt.size:
                    continue
                if i >= len(pts):
                    raise RuntimeError(f"{out.name}: more packets than frame messages")
                pkt.stream = ost
                pkt.time_base = ost.time_base
                pkt.pts = pkt.dts = int(pts[i])
                # the step to the next frame, the last repeating the one before: the raw stream's own duration
                # (0, or a 25 fps guess) could end the mp4's edit list where the last frame starts, and that frame
                # would not decode
                pkt.duration = int(pts[i + 1] - pts[i] if i + 1 < len(pts) else
                                   pts[i] - pts[i - 1] if i else TIME_BASE_DEN // 30)
                dst.mux(pkt)
                i += 1
        if i != len(pts):
            raise RuntimeError(f"{out.name}: {len(pts)} frame messages but {i} packets")
    finally:
        os.unlink(raw)
    with av.open(str(out)) as c:
        got = np.asarray([p.pts for p in c.demux(c.streams.video[0]) if p.size], dtype=np.int64)
    # the mp4 muxer shifts the whole track so its first frame sits at 0 (by an edit list); accept only
    # a constant shift under one frame and return the pts actually in the file, which is what the
    # decoder sees. The real capture times stay in times.npz.
    shift = np.unique(got - pts) if len(got) == len(pts) else np.array([10**9])
    if len(shift) != 1 or abs(int(shift[0])) >= TIME_BASE_DEN // NOMINAL_FPS:
        raise RuntimeError(f"{out.name}: pts in the written mp4 differ from the capture times ({shift[:5]})")
    return got
