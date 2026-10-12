"""An episode's footage, decoded once onto an 8 fps grid that every pass reads from.

The three passes (events/passes.py) see 4 fps, 1 fps and 8 fps windows of the same recording. Decoding each pass's
instants separately would decode most of the file three times, so the footage is decoded once, front to back, and the
frame shown at each grid instant t = k / GRID_FPS is kept, downscaled to at most CELL_MAX_W wide (never enlarged) and
stored as JPEG bytes: a five-minute head-camera clip at 8 fps is 2,400 cells, about 40 MB. The 4 fps and 1 fps
instants are subsets of the grid, so every pass sees exactly the same pictures of the same instants.

The frame kept for t is the last frame whose presentation time is at or before t (what a player shows at t), from the
frame's own pts, so variable-rate phone video is placed correctly. Display rotation is applied (label.frames.upright).
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from pathlib import Path

GRID_FPS = 8
CELL_MAX_W = 448
JPEG_QUALITY = 88


class FootageError(RuntimeError):
    """A camera could not be decoded onto the grid."""


@dataclass
class Camera:
    name: str
    path: Path
    start_s: float = 0.0  # the episode's start in the file (packed files); 0 for one file per episode
    end_s: float | None = None  # the episode's end in the file, or the file's end


@dataclass
class Footage:
    """{camera: {grid index: jpeg}} for one episode, plus its duration in seconds."""
    duration_s: float
    cells: dict[str, dict[int, bytes]] = field(default_factory=dict)
    sizes: dict[str, tuple[int, int]] = field(default_factory=dict)  # native (w, h) per camera

    @property
    def cameras(self) -> list[str]:
        return list(self.cells)

    def index(self, t_s: float) -> int:
        """The grid index shown at t (clamped to the episode)."""
        last = max(0, int(math.floor(self.duration_s * GRID_FPS)))
        return min(last, max(0, int(round(t_s * GRID_FPS))))

    def at(self, t_s: float) -> dict[str, bytes]:
        """Every camera's cell at the grid instant nearest t."""
        k = self.index(t_s)
        out = {}
        for cam, by_k in self.cells.items():
            if not by_k:
                continue
            j = k if k in by_k else max((i for i in by_k if i <= k), default=min(by_k))
            out[cam] = by_k[j]
        return out

    def times(self, fps: float, t0: float = 0.0, t1: float | None = None) -> list[float]:
        """Grid instants at `fps` (a divisor of GRID_FPS) from t0 to t1 inclusive, always including both ends."""
        t1 = self.duration_s if t1 is None else t1
        t0, t1 = max(0.0, t0), min(self.duration_s, t1)
        if t1 < t0:
            return []
        step = max(1, int(round(GRID_FPS / fps)))
        k0, k1 = self.index(t0), self.index(t1)
        ks = list(range(k0, k1 + 1, step))
        if ks[-1] != k1:
            ks.append(k1)
        return [k / GRID_FPS for k in ks]


def decode(cameras: list[Camera], cell_max_w: int = CELL_MAX_W) -> Footage:
    """Decode every camera once onto the 8 fps grid."""
    import av
    from label import frames as mf
    from prepare import display

    footage = Footage(duration_s=0.0)
    durations = []
    for cam in cameras:
        geom = display.geometry(str(cam.path))
        by_k: dict[int, bytes] = {}
        with av.open(str(cam.path)) as c:
            s = c.streams.video[0]
            s.codec_context.thread_count = 2
            if cam.start_s > 0:
                c.seek(int(cam.start_s / s.time_base), stream=s, backward=True, any_frame=False)
            pending = None  # [frame, jpeg or None]: the latest frame decoded, encoded only once it is kept
            next_k, last_t = 0, 0.0

            def keep(k: int) -> None:
                if pending[1] is None:
                    im = mf.upright(pending[0], geom)
                    footage.sizes.setdefault(cam.name, im.size)
                    pending[1] = _jpeg(im, cell_max_w)
                by_k[k] = pending[1]

            for fr in c.decode(s):
                if fr.pts is None:
                    continue
                t = float(fr.pts * s.time_base) - cam.start_s
                if t < -1e-6:
                    continue
                if cam.end_s is not None and t > cam.end_s - cam.start_s + 1e-6:
                    break
                # the grid instants before this frame show the previous one (the first frame stands in for any
                # instant before it)
                while next_k / GRID_FPS < t - 1e-6:
                    if pending is not None:
                        keep(next_k)
                        next_k += 1
                    else:
                        break
                pending = [fr, None]
                last_t = t
                while next_k / GRID_FPS <= t + 1e-6:
                    keep(next_k)
                    next_k += 1
            if pending is not None:
                # the last frame is on screen for one frame interval
                while next_k / GRID_FPS <= last_t + 1.0 / max(1.0, float(s.average_rate or 30)) + 1e-6:
                    keep(next_k)
                    next_k += 1
        if not by_k:
            raise FootageError(f"no frames decoded from {cam.path}")
        footage.cells[cam.name] = by_k
        durations.append(max(by_k) / GRID_FPS)
    footage.duration_s = max(durations)
    return footage


def _jpeg(im, width: int) -> bytes:
    from label.frames import downscaled
    im = downscaled(im.convert("RGB"), width)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()
