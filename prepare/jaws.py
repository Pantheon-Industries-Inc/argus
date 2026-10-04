"""The opening of a handheld gripper's jaws, measured frame by frame from its own wrist camera, and the moments it
closes and opens.

A UMI gripper records no gripper signal, so a grasp that misses and is retried within a second falls between two of
the harness's once-a-second instants. This measures the jaws on every frame of the wrist video instead, and the
harness lists each close and open in the episode's facts (label/episode.py jaw_block); it adds no frame.

The method is umi-action-deltas' grip6 (Pantheon-Industries-Inc/umi-action-deltas, umi/gripper/grip6.py), applied to
the flat 640x480 view the UMI release renders from the wrist fisheye. Each jaw carries two orange dots, one above the
other. The wrist camera is bolted to the gripper, so the dots appear only in two horizontal bands of the image, each
pair a mirror image about the camera's centre column, and nothing is searched anywhere else:
  - dot response: the orange score minus its morphological opening with a disk larger than a dot, so a large orange
    object (a cloth, a cup) is removed and a dot on the dark jaw survives;
  - per clip: the two rows (the most frequent height of response peaks inside each band) and one vertical symmetry
    axis (voted by the midpoints of same-row pairs from both rows);
  - per frame: for each row, the symmetric pair with the largest summed response over its spacing and a small shared
    vertical offset (the dots drop a few pixels as the jaws open);
  - a frame is kept only when its top-row and bottom-row spacings agree with the clip's own relation between them
    (the pixel stand-in for grip6's fin-depth check: a wrong dot breaks it), and gaps up to 0.5s are interpolated.
The opening is reported in pixels of dot spacing. Closes and opens are found against the clip's own shut spacing (where
the jaws meet), so no lens model or rig constant is needed. A close that ends at the shut spacing has nothing between
the jaws, or only something thin. A video in which the dots cannot be found (another gripper, another
view) gives nothing, and the harness then samples as before.

    python -m prepare.jaws VIDEO           prints the clip's rows, axis, coverage and its closes and opens
"""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
from scipy import ndimage

W, H = 640, 480                          # the release's wrist view; a video of another size is scaled to it
TOP_Y, BOT_Y = (225, 290), (335, 405)    # the dot rows' bands in that view, measured on the pilot's wrist clips
AXIS_X = (295, 350)                      # the symmetry axis' range
BAND = 5                                 # +- px around a row
DY = np.arange(-3, 9)                    # px a row's dots may sit below (+) or above (-) their row as the jaws open
S_MIN, S_MAX = 60, 330                   # a row's dot spacing in px, closed to fully open, generous
OPEN_K = 19                              # window for the opening: larger than a dot (about 10px), smaller than a cup
ROW_TOL = 0.08                           # a frame's bottom/top spacing may differ by this share from the clip's relation
GAP_S = 0.5                              # gaps up to this long are interpolated
EVENT_MIN = 0.08                         # a close or open moves the spacing by at least this share of the closed spacing
EVENT_WITHIN_S = 0.7                     # ... within this long
SHUT_TOL = 0.06                          # a close ending within this share of the clip's shut spacing has nothing
                                         # between the jaws, or only something thin (cloth, a band, paper)

def read_frames(video: str):
    """Every frame of the video as BGR at W x H, with the video's frame rate."""
    p = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                   "stream=width,height,r_frame_rate", "-of", "json", video],
                                  capture_output=True, text=True, check=True).stdout)["streams"][0]
    num, den = (int(x) for x in p["r_frame_rate"].split("/"))
    fps = num / den if den else 30.0
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", video, "-vf", f"scale={W}:{H}", "-f", "rawvideo",
                          "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3), fps


def response(img):
    """Compact-orange response over the two bands (float32, full-frame coordinates, zero elsewhere). The opening uses
    a square window, which scipy computes as two fast one-dimensional passes."""
    y0, y1 = TOP_Y[0] - BAND - 10, BOT_Y[1] + BAND + 10
    crop = img[y0:y1].astype(np.float32)
    b, g, r = crop[..., 0], crop[..., 1], crop[..., 2]
    o = np.clip(r - np.maximum(g, b) * 0.9, 0, None) * (r > 60)
    o = ndimage.gaussian_filter(o, 0.8)
    top = o - ndimage.grey_opening(o, size=(OPEN_K, OPEN_K))
    out = np.zeros((H, W), np.float32)
    out[y0:y1] = top
    return out


def _peaks(resp, thr):
    m = (resp == ndimage.maximum_filter(resp, size=9)) & (resp > thr)
    ys, xs = np.nonzero(m)
    return np.stack([xs, ys], 1)


def calibrate(frames) -> dict | None:
    """The clip's two dot rows, its symmetry axis and its response threshold, from 48 frames across the clip."""
    pick = frames[np.linspace(0, len(frames) - 1, min(48, len(frames))).astype(int)]
    R = [response(im) for im in pick]
    pos = np.concatenate([r[r > 0] for r in R if (r > 0).any()]) if any((r > 0).any() for r in R) else np.array([])
    if pos.size < 50:
        return None
    thr = float(np.percentile(pos, 99.0))
    P = [_peaks(r, thr) for r in R]
    allp = np.vstack([p for p in P if len(p)]) if any(len(p) for p in P) else np.zeros((0, 2))
    rows = []
    for lo, hi in (TOP_Y, BOT_Y):
        h, e = np.histogram(allp[:, 1], bins=np.arange(lo, hi + 4, 4))
        if h.max() < 5:
            return None
        rows.append(float(e[np.argmax(h)] + 2))
    mids = []
    for p in P:
        for y in rows:
            q = p[np.abs(p[:, 1] - y) <= BAND + 2]
            for i in range(len(q)):
                for j in range(i + 1, len(q)):
                    if S_MIN <= abs(q[i, 0] - q[j, 0]) <= S_MAX:
                        mids.append((q[i, 0] + q[j, 0]) / 2)
    mids = np.array([m for m in mids if AXIS_X[0] <= m <= AXIS_X[1]])
    if len(mids) < 5:
        return None
    h, e = np.histogram(mids, bins=np.arange(AXIS_X[0], AXIS_X[1] + 3, 3))
    c0 = e[np.argmax(h)] + 1.5
    return {"rows": rows, "cx": float(np.median(mids[np.abs(mids - c0) <= 4])), "thr": thr}


def _row_pair(respd, y, cal):
    """The symmetric pair with the largest summed response on one row: (spacing px, score) or None."""
    ss = np.arange(S_MIN, S_MAX + 1, 1.0)
    xl, xr = np.round(cal["cx"] - ss / 2).astype(int), np.round(cal["cx"] + ss / 2).astype(int)
    keep = (xl >= 0) & (xr < W)
    ss, xl, xr = ss[keep], xl[keep], xr[keep]
    ys = np.clip(np.round(y + DY).astype(int), 0, H - 1)
    v0, v1 = respd[ys[:, None], xl[None, :]], respd[ys[:, None], xr[None, :]]
    tot = np.where(np.minimum(v0, v1) >= 0.25 * cal["thr"], v0 + v1, 0)
    best = tot.max(0)
    if best.max() <= 0:
        return None
    i = int(best.argmax())
    return float(ss[i]), float(best[i])


def measure(frames, cal) -> np.ndarray:
    """Per frame (top spacing, bottom spacing) in px, NaN where a row's pair is not found."""
    out = np.full((len(frames), 2), np.nan)
    for k, im in enumerate(frames):
        respd = ndimage.maximum_filter(response(im), size=3)
        for r, y in enumerate(cal["rows"]):
            got = _row_pair(respd, y, cal)
            if got:
                out[k, r] = got[0]
    return out


def opening(sp: np.ndarray, fps: float) -> np.ndarray:
    """One opening per frame (the bottom row's spacing, the wider and steadier pair), kept only where the two rows
    agree with the clip's own linear relation between them, smoothed (median of 5, mean of 3) and with gaps up to
    GAP_S interpolated."""
    top, bot = sp[:, 0], sp[:, 1]
    ok = np.isfinite(top) & np.isfinite(bot)
    o = np.full(len(sp), np.nan)
    if ok.sum() < 10:
        return o
    a, b = np.polyfit(top[ok], bot[ok], 1)
    agree = ok & (np.abs(bot - (a * top + b)) <= ROW_TOL * np.abs(a * top + b))
    o[agree] = bot[agree]
    med = o.copy()
    for i in np.flatnonzero(np.isfinite(o)):
        w = o[max(0, i - 2):i + 3]
        w = w[np.isfinite(w)]
        if len(w) >= 3:
            med[i] = np.median(w)
    sm = med.copy()
    for i in np.flatnonzero(np.isfinite(med)):
        w = med[max(0, i - 1):i + 2]
        if np.isfinite(w).all() and len(w) == 3:
            sm[i] = w.mean()
    good = np.flatnonzero(np.isfinite(sm))
    for a_, b_ in zip(good, good[1:]):
        if 1 < b_ - a_ <= GAP_S * fps:
            sm[a_ + 1:b_] = np.interp(np.arange(a_ + 1, b_), [a_, b_], [sm[a_], sm[b_]])
    return sm


def events(o: np.ndarray, fps: float) -> list[dict]:
    """The moments the jaws close or open: the spacing falls (a close) or rises (an open) by at least EVENT_MIN of
    the clip's closed spacing within EVENT_WITHIN_S. Each event is timed where the move ends, when the jaws settle."""
    fin = np.isfinite(o)
    if fin.sum() < 10:
        return []
    closed = float(np.nanpercentile(o, 3))
    step = EVENT_MIN * closed
    win = max(2, int(round(EVENT_WITHIN_S * fps)))
    out, i, n = [], 0, len(o)
    while i < n - 1:
        if not fin[i]:
            i += 1
            continue
        seg = o[i:min(n, i + win + 1)]
        j_lo, j_hi = int(np.nanargmin(seg)), int(np.nanargmax(seg))
        if o[i] - seg[j_lo] >= step:
            kind, j = "close", i + j_lo
        elif seg[j_hi] - o[i] >= step:
            kind, j = "open", i + j_hi
        else:
            i += 1
            continue
        # follow the move to where it settles (no further move of a quarter step within a tenth of a second)
        settle = max(1, int(round(0.1 * fps)))
        while j + settle < n and np.isfinite(o[j + settle]) and (
                (kind == "close" and o[j] - o[j + settle] > step / 4) or (kind == "open" and o[j + settle] - o[j] > step / 4)):
            j += settle
        out.append({"t": round(j / fps, 3), "kind": kind, "from_px": round(float(o[i]), 1), "to_px": round(float(o[j]), 1)})
        i = j + 1
    return out


def mark_shut(evs: list[dict], o: np.ndarray) -> float:
    """Marks each close "shut" when it ends at the clip's shut spacing (the 5th percentile of the opening, where the
    jaws meet), and returns that spacing. A close that stops wider has something between the jaws."""
    shut = float(np.nanpercentile(o, 5))
    for e in evs:
        if e["kind"] == "close":
            e["shut"] = bool(e["to_px"] <= shut * (1 + SHUT_TOL))
    return shut


def analyse(video: str) -> dict | None:
    """{"fps", "rows", "cx", "coverage", "shut_px", "opening_px", "events"} for one wrist video, or None when its dots
    are not found."""
    frames, fps = read_frames(video)
    cal = calibrate(frames)
    if cal is None:
        return None
    o = opening(measure(frames, cal), fps)
    if not np.isfinite(o).any():
        return None
    evs = events(o, fps)
    shut = mark_shut(evs, o)
    return {"fps": fps, "rows": cal["rows"], "cx": round(cal["cx"], 1), "coverage": round(float(np.isfinite(o).mean()), 3),
            "shut_px": round(shut, 1), "opening_px": [None if not np.isfinite(x) else round(float(x), 1) for x in o],
            "events": evs}


if __name__ == "__main__":
    r = analyse(sys.argv[1])
    if r is None:
        print("no jaw dots found")
    else:
        print(json.dumps({k: v for k, v in r.items() if k != "opening_px"}, indent=1))
