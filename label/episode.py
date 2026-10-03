"""One episode's request: which frames, decoded exactly, and a prompt that states only facts we are sure of.

An episode is a sidecar folder written by a preparer (prepare/): context.json (the dataset's facts: rig, state
kind, fps, cameras and what each one is, the instruction or annotation), sources.json (per camera, the video
file, the episode's offset in it and its frame count), state.npz (recorded state and action, absent for
video-only rigs) and, where the dataset keeps real capture times, times.npz and a kmap per camera paired to the
anchor camera by time. Nothing about the data is changed: the model sees the episode as the dataset ships it.

What the model is told about the episode, and where each fact comes from:
- dataset, robot type, fps, camera names and resolution: the dataset's own metadata (context.json).
- what each camera is: checked by eye on the dataset's frames at prepare time, stated as the dataset's
  camera identity, and the model is told to verify it against the pixels.
- the instruction or annotation: the dataset's per-episode text, as a claim to check.
- still spans: from the recorded state, true by construction (state.still_spans), as the recording's claim.
- recorded motion between consecutive instants: from the recorded state, as a claim to check.
- sampling: exactly what state.sample_frames did.
- which of these are said at all: each part that depends on data the episode may not hold is a block (BLOCKS) with a
  test of the episode folder, so an episode without that data gets no word of it and is asked for no field of it.
- a fisheye lens: only where the camera's own frames show a circular image with black corners (label/lens.py).
Nothing else is said about field of view or the lens, nor about lighting, object identities, or what a gripper
reading implies.

Layout (fixed, not flags):
- Grid cells. A teleop episode's width is chosen from its task text alone (label/route.py): a task that needs
  fine detail (lettering, numbers, symbols or a display, which face of an object is up, small objects of
  similar shape) keeps every camera at 448 px, because at narrower cells a wrist camera's oblique view cannot
  show which face is up and the model reads a change of viewing direction as a change of state; every other
  teleop episode is sent at 224 px. Handheld cells are 320 px: a gripper camera has the held object in every
  frame, and at 256 px or less a thin utensil held upright was misread. Head-camera cells are 256 px. An
  episode whose grids would pass the request's image-size cap (a long ABC-130k episode at 448 px) is sent at
  the largest of CELL_W_STEPS that fits, instead of being refused.
- Contact detail views, on teleop episodes sent at narrow cells: at the sampled instants just after the
  recorded gripper value changed sharply, the scene camera and the acting arm's own camera are sent again at
  detail size, because that is where what is held, and how it is left, is decided.
- After the grids, the episode's first and last instant are sent again larger (all cameras stacked, at most
  DETAIL_MAX_W wide each), because completion is judged on the end state and small detail (lettering, a
  display, fine alignment) only reads at native resolution.
"""
from __future__ import annotations

import base64
import io
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from checks import timebase
from label import frames as mf
from label import lens
from label import prompts
from label import state as ms

FPS = 30
VIEW_ORDER = ("exo", "left", "right")          # harness view keys with a role: the scene camera, the two mounted ones
MOUNTED = ("left", "right")                     # the cameras mounted on the left and right arm or gripper
EXTRA_VIEW = re.compile(r"extra(\d+)")          # any other camera the recording has (extra1, extra2, ...)
CAM_NAME = {"exo": "top", "left": "left", "right": "right"}   # dataset camera names
GRID_CELL_W_BY_RIG = {"teleop_arms": 448, "handheld_gripper": 320, "ego_head": 256}
CELL_W_STEPS = (448, 384, 320, 288, 256, 224, 192)   # widths tried, largest first, when a request is too big
DETAIL_VIEW_BYTES_MAX = 2_000_000   # room kept under the cap for the two full-resolution first and last frames
# The provider caps a request's total image size at 50 MB. Its counted size is not the raw JPEG bytes: on a
# measured request 34 MB of JPEG counted as 60.14 MB (about 1.77x, base64 on the wire plus overhead), so the
# inflated size is bounded with a margin.
IMAGE_LIMIT_BYTES = 50 * 1024 * 1024
IMAGE_SIZE_INFLATION = 1.85
# The first and last instant are also sent larger, for small detail (lettering, a display, fine
# alignment). Image tokens scale with pixel area, so each camera is capped at this width: 768 px is above
# MolmoAct2's native 640 px (where lettering and displays read fine), while native 1920 px frames would
# cost about 6x as much for little gain.
DETAIL_MAX_W = 768
# Resolution routing (label/route.py): the (narrow, wide) cell width of a routed rig. A task that needs fine detail
# gets the wide cells, every other task the narrow ones plus contact detail views.
ROUTE_WIDTHS = {"teleop_arms": (224, 448)}
CONTACT_BELOW_W = 384   # contact detail views only for episodes sent at narrower cells than this
# Contact detail views: the sampled instants just after the recorded gripper value changed sharply (a close or an
# open between two instants), where what is held, or how it is left, is decided. The value only chooses where to
# look closer; what is held is still read from the frames. One view per CONTACT_EVERY_S of footage at most, never
# more than CONTACT_MAX, and none for a flat channel, video only or head cameras. Teleop only: a handheld episode is
# one or two gripper cameras already sent at 320 px, which settles what is held at lower cost.
CONTACT_RIGS = ("teleop_arms",)
CONTACT_EVERY_S = 4.0
CONTACT_MAX = 8
CONTACT_MIN_CHANGE = 0.25    # of the channel's own range over the episode, between consecutive instants
CONTACT_MIN_GAP_S = 2.0
PAIRED_SPAN_SLACK_S = 0.1   # a paired camera is shown at an instant up to this far outside its own first and last frame
GRID_GUTTER = 84
GRID_HEADER = 30


def is_episode_dir(ep_dir: Path) -> bool:
    return (Path(ep_dir) / "context.json").exists() and (Path(ep_dir) / "sources.json").exists()


def load(ep_dir: Path) -> dict:
    ep_dir = Path(ep_dir)
    ctx = json.loads((ep_dir / "context.json").read_text())
    src = json.loads((ep_dir / "sources.json").read_text())
    for v, d in src.items():
        if "n_frames" not in d:
            raise RuntimeError(f"{ep_dir}: sources.json has no n_frames for {v}; prepare the episode again")
    if ctx.get("state_kind") == "none":
        # video-only rigs (a head camera on a person): no recorded state; the state array is empty
        # columns over the anchor camera's frames so frame counts and times work unchanged
        first = order_views(src)[0]
        state, action = np.zeros((int(src[first]["n_frames"]), 0)), None
    else:
        z = np.load(ep_dir / "state.npz")
        if "state" not in z.files:
            raise RuntimeError(f"{ep_dir}: state.npz has no state array; prepare the episode again")
        state, action = z["state"], (z["action"] if "action" in z.files else None)
    ep = {"dir": ep_dir, "context": ctx, "sources": src, "state": state,
          "action": action, "times": None, "kmap": {}, "signals": {}}
    if ctx.get("signals") and not ctx.get("state_unaligned"):
        # the recording's other per-frame numbers, under the dataset's names (prepare/formats.py recorded_signals), with
        # each one's shape and value names (a 16 x 16 pressure map; fx, fy, fz) and everything else its reader wrote,
        # so a field a reader adds reaches the checks without being listed here. Signals on the frames of a camera
        # taken out of the episode, with nothing to place them on the others (state_unaligned), are not read. One whose
        # rows stop short of the episode's frames has no reading past them (label/signals.py pad_rows)
        from label import signals as sg
        z = np.load(ep_dir / "signals.npz")
        ep["signals"] = {s["name"]: sg.pad_rows(sg.columns(z[s["key"]]), len(state)) for s in ctx["signals"]}
        ep["signal_meta"] = {s["name"]: {k: v for k, v in s.items() if k not in ("name", "key")}
                             for s in ctx["signals"]}
    if ctx.get("real_times"):
        # datasets with real per-frame capture times (ABC-130k, RealOmin): every time shown uses them, and
        # each camera's frames are decoded by their exact pts
        t = np.load(ep_dir / ctx["real_times"])
        ep["times"] = {k: t[k] for k in t.files}
    for v, d in src.items():
        if d.get("kmap"):
            ep["kmap"][v] = np.load(ep_dir / d["kmap"])
    # each camera's depth stream, when the recording has one (label/depth.py)
    from label import depth as dp
    ep["depth"] = {v: e for v, e in dp.load(ep_dir).items() if v in src}
    return ep


def ep_fps(ep: dict) -> float:
    return float(ep["context"].get("fps") or FPS)


# a reply's time may lie this far past the edge of the footage it labels (a reply rounding its last time up) before the
# board flags it (board/build.py steps_outside, label/pieces.py outside_part)
STEP_SLACK_S = 0.5
# a number written as text: plain ASCII digits with an optional sign, point and exponent. Python's float also reads
# "1_000" and digits of other scripts, which no dataset or reply means as a time
NUMBER_TEXT = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", re.ASCII)


def number(x) -> float | None:
    """A number as a dataset or the model writes it: a finite number, or text that reads as one (NUMBER_TEXT: "12.5",
    or a time "12.5s"). Anything else (none, a word, NaN, true or false) is no number, None, and a time that is none is
    shown untimed. Negative zero is zero, so no time prints as "-0.0s". One rule, so the prompt, the parts of a long
    recording, the reply's parse and the board read every time alike."""
    if isinstance(x, bool):
        return None
    if isinstance(x, str):
        x = x.strip().removesuffix("s").strip()
        if not NUMBER_TEXT.fullmatch(x):
            return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v + 0.0 if math.isfinite(v) else None


def order_views(keys) -> list[str]:
    """View keys in row order: the scene camera, the left and right mounted ones, then the extra cameras by number."""
    keys = list(keys)
    extra = sorted((k for k in keys if EXTRA_VIEW.fullmatch(str(k))), key=lambda k: int(EXTRA_VIEW.fullmatch(k)[1]))
    return [v for v in VIEW_ORDER if v in keys] + extra


def views(ep: dict) -> list[str]:
    """The episode's cameras in row order (top, then left, then right gripper, then any other cameras)."""
    have = ep.get("sources") or ep["context"].get("cameras") or dict.fromkeys(VIEW_ORDER)
    return order_views(have)


def anchor(ep: dict) -> str:
    return views(ep)[0]


def cam_name(ep: dict, v: str) -> str:
    return ((ep["context"].get("cameras") or {}).get(v) or {}).get("name") or CAM_NAME.get(v, v)


def frame_time(ep: dict, k: int) -> float:
    """Seconds from the episode's start for anchor frame k: the real capture time when the dataset
    has one, otherwise k / fps (MolmoAct2's and FastUMI's timestamps are exactly frame_index / fps)."""
    if ep.get("times") is not None:
        return float(ep["times"][anchor(ep)][k])
    return k / ep_fps(ep)


def describe_spans(ep: dict, spans) -> list[dict]:
    last = len(ep["state"]) - 1
    return [{"start_s": round(frame_time(ep, a), 2), "end_s": round(frame_time(ep, min(b, last)), 2)} for a, b in spans]


# Rigs this harness knows. The episode's context.json declares one as "profile"; nothing is assumed from the
# dataset name or the camera set.
RIGS = ("teleop_arms", "handheld_gripper", "ego_head")
STATE_KINDS = ("joints", "ee_pose", "none")
# One sampled instant every N s for the whole episode, still spans included (a still span is exactly
# where a stopped recording would hide). Teleop arms move slowly and are seen by up to three cameras, so one
# instant every 1.5 s is enough. Handheld demonstrations are short and fast, so denser. Egocentric footage has one
# low-resolution head camera and the hands are the whole point, so it is sampled twice as densely again.
SAMPLE_EVERY_S = {"teleop_arms": 1.5, "handheld_gripper": 1.0, "ego_head": 0.5}


def rig(ep: dict) -> str:
    p = ep["context"].get("profile")
    if p not in RIGS:
        raise RuntimeError(f"{ep.get('dir', '?')}: context.json must declare profile as one of {RIGS}, got {p!r}")
    return p


def state_kind(ep: dict) -> str:
    k = ep["context"].get("state_kind")
    if k not in STATE_KINDS:
        raise RuntimeError(f"{ep.get('dir', '?')}: context.json must declare state_kind as one of {STATE_KINDS}, "
                           f"got {k!r}")
    return k


def state_span(ep: dict) -> tuple[int, int]:
    """The anchor frames [a, b) the recorded state covers: all of them, or context.json state_span when the state was
    moved onto another camera's frames and covers only part of them (board/clips.py reanchor, which leaves no value
    outside it). Every use of the state stays inside it."""
    T = int(len(ep["state"]))
    sp = ep["context"].get("state_span")
    if isinstance(sp, (list, tuple)) and len(sp) == 2:
        a, b = max(0, int(sp[0])), min(T, int(sp[1]))
        if a < b:
            return a, b
    return 0, T


def plan(ep: dict) -> dict:
    """Frames to send plus the deterministic checks we report ourselves. With no usable arm state, the quiet spans of
    the signals choose the extra instants (pl["quiet_spans"])."""
    T = int(len(ep["state"]))
    r, kind, fps = rig(ep), state_kind(ep), ep_fps(ep)
    windows = {v: int(ep["sources"][v]["n_frames"]) for v in views(ep)}
    # the state follows the anchor camera's frames; a camera paired to the anchor by real time (kmap) has
    # its own frame count and is matched through the map, so only unpaired cameras must equal the state
    a = anchor(ep)
    paired = {v for v in windows if v != a and v in ep["kmap"] and len(ep["kmap"][v]) >= windows[a]}
    # state_unaligned: the camera the state was recorded on was taken out of the episode, and nothing places the state
    # on the cameras left (board/clips.py drop_cameras), so it is never treated as aligned
    sa, sb = state_span(ep)
    checks = {"state_frames": T, "camera_frames": windows,
              "camera_windows_match_state": not ep["context"].get("state_unaligned")
              and all(n == T for v, n in windows.items() if v not in paired)}
    if ep.get("action") is not None and r == "teleop_arms" and kind == "joints" and ep["state"].shape[1] == 14:
        # sped-up recording (the rig's loop ran below the rate its samples are stamped at): a report
        # field computed from the leader/follower joint lag, not a claim made to the model. It reads the 12 arm
        # joints of two arms (timebase.JOINTS), as measure_folder does, so a one-arm recording is not measured
        checks["timebase"] = timebase.timebase_check(ep["state"][sa:sb], ep["action"][sa:sb],
                                               ep["context"].get("timebase_neighbour_lag_frames"), ep_fps(ep))
    if kind != "none" and checks["camera_windows_match_state"]:
        spans = [(x + sa, y + sa) for x, y in ms.still_spans(ep["state"][sa:sb], fps=fps, kind=kind,
                                                            grip_range=ms.gripper_full_range(ep["context"]))]
        n = T
    else:
        # video only, or a dataset defect (the cameras do not cover the same frames as the state): label the
        # video as shipped over the anchor frames every camera not paired to it by time also has, and make no
        # state claims (the defect is reported in checks)
        spans = []
        n = min([windows[a]] + [w for v, w in windows.items() if v != a and v not in paired])
    if ep["context"].get("stream_checks"):
        checks["streams"] = ep["context"]["stream_checks"].get("streams")
    if ep["context"].get("stream_pairing"):
        # checks/stream_pairing.py: whether each mounted stream follows its own actor's recorded motion (a
        # report field, never a claim made to the model)
        checks["stream_pairing"] = ep["context"]["stream_pairing"]
    every = SAMPLE_EVERY_S[r]
    sample_spans, quiet = spans, None
    if not (kind != "none" and checks["camera_windows_match_state"]) and ep.get("signals"):
        # no arm state to find still spans in: the instant every moving signal falls quiet, and the instant it moves
        # again, are sent as an arm's still span would give them (label/signals.py quiet_spans), never as a claim
        from label import signals as sg
        quiet = sg.quiet_spans({k: np.asarray(v)[:n] for k, v in ep["signals"].items()},
                               int(round(ms.MIN_STILL_S * fps)))
        sample_spans = quiet
    ks = ms.sample_frames(n, sample_spans, fps=fps, moving_every_s=every, still_every_s=every)
    zero = ep["context"].get("clock_zero_s")
    if zero is not None and ep.get("times") is not None:
        # a camera that started before the episode's clock (kept for an episode labelled already, board/clips.py
        # reanchor) is main: nothing before the clock's start is sampled, and its first frame at the start is
        before = [k for k in range(n) if frame_time(ep, k) < float(zero) - 0.5 / fps]
        if before and len(before) < n:
            ks = sorted({k for k in ks if k > before[-1]} | {before[-1] + 1})
    if kind != "none" and checks["camera_windows_match_state"] and (sa, sb) != (0, T):
        # a state that covers part of the episode: its first and last frame are instants too, so the recorded motion
        # covers all of it and stops there (_motion_table)
        ks = sorted(set(ks) | {sa, sb - 1})
    pl = {"n": n, "ks": ks, "spans": spans, "checks": checks, "state_span": (sa, sb),
          "state_usable": checks["camera_windows_match_state"], "touch": touch_verdicts(ep, n)}
    if quiet is not None:
        pl["quiet_spans"] = quiet
    pl["contact"] = contact_instants(ep, pl)
    return pl


def contact_instants(ep: dict, pl: dict) -> list[int]:
    """Sampled instants just after the recorded gripper value changed sharply, where an object is usually picked
    up or put down: the largest changes first, at least CONTACT_MIN_GAP_S apart, never the first or last instant
    (already sent in detail). Empty for head cameras, video only, a state that does not line up with the cameras,
    and a flat channel (it carries no timing)."""
    if rig(ep) not in CONTACT_RIGS or state_kind(ep) == "none" or not pl["state_usable"]:
        return []
    ks = pl["ks"]
    st = np.asarray(ep["state"][:pl["n"]], dtype=np.float64)
    cands = []
    for g in range(st.shape[1] // 7):
        v = st[:, 7 * g + 6]
        rng = float(np.nanmax(v) - np.nanmin(v)) if np.isfinite(v).any() else 0.0
        if not rng > 1e-6:
            continue
        for a, b in zip(ks, ks[1:]):
            d = abs(float(v[b] - v[a])) / rng
            if d >= CONTACT_MIN_CHANGE and b != ks[-1]:
                cands.append((d, b))
    n_max = min(CONTACT_MAX, max(1, int(round(frame_time(ep, ks[-1]) / CONTACT_EVERY_S))))
    gap = CONTACT_MIN_GAP_S * ep_fps(ep)
    chosen = []
    for d, k in sorted(cands, key=lambda x: (-x[0], x[1])):
        if len(chosen) >= n_max:
            break
        if all(abs(k - c) >= gap for c in chosen):
            chosen.append(k)
    return sorted(chosen)


def contact_views(ep: dict, pl: dict) -> list[tuple[int, list[str]]]:
    """[(k, views)]: each contact instant with the scene camera (it shows how the object is left) and the mounted
    cameras of the arms whose gripper value changed sharply into it (a wrist camera shows what its own gripper
    holds)."""
    mounted = [v for v in views(ep) if v in MOUNTED]
    if not mounted:
        return [(k, list(views(ep))) for k in pl["contact"]]
    st = np.asarray(ep["state"][:pl["n"]], dtype=np.float64)
    ks = pl["ks"]
    out = []
    for k in pl["contact"]:
        i = ks.index(k)
        a = ks[i - 1] if i > 0 else k
        who = []
        for g in range(st.shape[1] // 7):
            v = st[:, 7 * g + 6]
            rng = float(np.nanmax(v) - np.nanmin(v))
            if rng > 1e-6 and abs(float(v[k] - v[a])) / rng >= CONTACT_MIN_CHANGE:
                who.append(g)
        vs = [mounted[g] for g in who if g < len(mounted)] if len(mounted) > 1 else mounted
        out.append((k, (["exo"] if "exo" in views(ep) else []) + (vs or mounted)))
    return out


def actors(ep: dict) -> list[str]:
    """Names of the arms or grippers in state order (7 values each): left then right, or the one.
    On a person (ego), the actors are their own two hands."""
    if rig(ep) == "ego_head":
        return ["left", "right"]
    if state_kind(ep) == "none":
        # video only: the actors are the mounted cameras' own, or both when no single mounted camera names one
        mounted = [v for v in views(ep) if v in MOUNTED]
        return [mounted[0]] if len(mounted) == 1 else ["left", "right"]
    if ep["state"].shape[1] == 14:
        return ["left", "right"]
    # the one arm or gripper is named by its own mounted camera, never by an extra camera that sorts after it
    mounted = [v for v in views(ep) if v in MOUNTED]
    return [cam_name(ep, (mounted or views(ep)[:1])[-1])]


def _decode_error(e: Exception) -> bool:
    """Whether e is the decoder failing on a file that is there (PyAV's errors, the frame reader's own): a fault of the
    recording, which costs only the frames it hits. A file that is missing or cannot be opened (OSError, including
    PyAV's FileNotFoundError and PermissionError) is a fault on our side and is never one of these."""
    import av
    return isinstance(e, (av.error.FFmpegError, mf.FrameError)) and not isinstance(e, OSError)


def _decode_view(ep: dict, v: str, ks: list[int], gate=None, widths=None, detail_ks=(), failed: set | None = None):
    """Frames for anchor indices ks. A camera paired to the anchor by real time (kmap) is decoded at
    its own frame nearest each anchor frame; results are keyed by the anchor index. With widths, a frame wider
    than the widest of them is kept full size only at detail_ks, and otherwise only at those widths.

    A camera never fails its episode over its own data. Its file may end before the episode does (an upload's every
    camera one frame short), and then the instants after its last frame have no frame. Its file may not decode, or be
    damaged partway, and then each instant is decoded on its own, so only the instants it cannot decode lose its
    frame; those go into failed, when given. A missing file still raises (_decode_error)."""
    s = ep["sources"][v]
    km = ep["kmap"].get(v)
    own = [int(km[k]) for k in ks] if km is not None else list(ks)
    pts = ep["times"].get(f"{v}_pts") if ep.get("times") is not None else None
    keep = None
    if widths:
        full = {int(km[k]) if km is not None else int(k) for k in detail_ks}
        top = max(widths)
        keep = lambda j, im: im if (j in full or im.width <= top) else mf.Shrunk(im, widths)

    def run(js):
        def go():
            return mf.extract_frames(s["packed"], s["base_s"], int(s["n_frames"]), js, pts=pts, fps=ep_fps(ep),
                                     keep=keep, tail_ok=True)
        if gate is None:
            return go()
        with gate:
            return go()
    try:
        got = run(own)
    except Exception as e:
        if not _decode_error(e):
            raise
        got, bad = {}, set()
        for j in sorted(set(own)):
            try:
                got.update(run([j]))
            except Exception as e1:
                if not _decode_error(e1):
                    raise
                bad.add(j)
        if failed is not None:
            failed.update(k for k, j in zip(ks, own) if j in bad)
    return {k: got[j] for k, j in zip(ks, own) if j in got}


def frames(ep: dict, pl: dict, gate=None, widths=None, detail_ks=()) -> dict:
    """{view: {k: PIL image}} for every planned k a camera has a frame at. With widths (the cell widths a request can
    be built at), frames outside detail_ks are kept only at those widths (label/frames.py Shrunk).

    An instant no camera has a frame at is dropped from pl["ks"], and when the episode's last instants are past every
    camera's last frame (an upload whose every camera's file ends a frame before the episode does), the last frame any
    camera has takes their place, so the last detail view is the end of the footage (ep["footage_end"]). The episode
    keeps what it found for the prompt and the request: ep["no_frame"], the instants each camera has no frame at,
    which recording_at then reports as not recording, so every grid and view leaves it out there; ep["decode_failed"],
    the instants a camera's file could not be decoded at, before its last frame or, for a file none of whose frames
    decodes, all of them (_coverage_note, decode_failures). Raises only when no camera has any frame."""
    vs = views(ep)
    failed = {v: set() for v in vs}
    with ThreadPoolExecutor(max_workers=len(vs)) as ex:
        futs = {v: ex.submit(_decode_view, ep, v, pl["ks"], gate, widths, detail_ks, failed[v]) for v in vs}
        got = {v: f.result() for v, f in futs.items()}
    ks = sorted(set(pl["ks"]))
    # an instant some camera can show: decoded there and inside its own recording (a camera paired by time that was
    # not recording has only its nearest frame, from another time, which is never shown)
    keep = [k for k in ks if any(k in got[v] and _in_span(ep, v, k) for v in vs)]
    if not keep:
        raise mf.FrameError(f"{ep.get('dir', '?')}: no camera has a frame at any instant")
    ep.pop("footage_end", None)
    if len(keep) < len(ks):
        past = [k for k in ks if k > keep[-1]]
        if past:
            # the frames before the first instant past every camera's end, back to the last instant kept (decoded
            # again at full size, for the detail view): the latest any camera has takes the end's place
            for k in range(min(past) - 1, keep[-1] - 1, -1):
                more = {v: _decode_view(ep, v, [k], gate) for v in vs}
                if any(k in more[v] and _in_span(ep, v, k) for v in vs):
                    for v in vs:
                        got[v].update(more[v])
                    keep = sorted(set(keep) | {k})
                    ep["footage_end"] = k
                    break
        pl["ks"] = keep
        if pl.get("contact"):
            pl["contact"] = [k for k in pl["contact"] if k in keep]
    ep["no_frame"] = {v: {k for k in keep if k not in got[v]} for v in vs if any(k not in got[v] for k in keep)}
    # a damaged stretch is an instant the camera could not decode before its last frame (the instants after it are
    # where its file ended); a camera with no frame at all that failed to decode does not decode anywhere. It is
    # recorded whether or not another camera shows the instant, so an instant that left the request is still flagged
    bad = {v: sorted(k for k in failed[v] if not got[v] or k < max(got[v])) for v in vs}
    ep["decode_failed"] = {v: ks_ for v, ks_ in bad.items() if ks_}
    ep["undecodable"] = {v for v in ep["decode_failed"] if not got[v]}
    return got


def decode_failures(ep: dict) -> list[dict]:
    """The stretches each camera's file could not be decoded at (frames), for the run's record and the board
    (board/build.py reader_issues): [{"camera", "t0_s", "t1_s", "what"}], on the recording's clock (a part of a long
    recording adds where it starts, label/pieces.py). Labelling never writes into the episode folder."""
    from board.clips import camera_label
    off = float((ep["context"].get("piece") or {}).get("t0_s") or 0.0)
    out = []
    for v, ks in (ep.get("decode_failed") or {}).items():
        t0, t1 = round(frame_time(ep, min(ks)) + off, 3), round(frame_time(ep, max(ks)) + off, 3)
        name = camera_label(v, ep["context"])
        if v in (ep.get("undecodable") or ()):
            what = f"The {name} video could not be decoded, so the labels have no frame of it."
        else:
            when = f"at {t0:.2f} s" if t0 == t1 else f"from {t0:.2f} s to {t1:.2f} s"
            what = f"The {name} video could not be decoded {when}, so the labels have no frame of it there."
        out.append({"camera": v, "t0_s": t0, "t1_s": t1, "what": what})
    return out


def _in_span(ep: dict, v: str, k: int) -> bool:
    """Whether anchor instant k is within camera v's own recording (within PAIRED_SPAN_SLACK_S of its first and last
    frame). A camera paired to the anchor by real time (kmap) that started later or stopped earlier was not recording
    there: its nearest frame is its first or last, taken at another time, so it is not shown under this instant's
    time."""
    km = (ep.get("kmap") or {}).get(v)
    t = ep["times"] if ep.get("times") is not None else None
    if km is None or t is None or v not in t:
        return True
    # only the camera's own span counts: inside it, a dropped frame leaves the nearest frame a few hundredths of a
    # second away, which is still the view at that moment; and a camera whose first frame comes a frame or two after
    # the anchor's (ABC-130k's wrists, 0.034 s) is shown as usual
    tk = frame_time(ep, k)
    return float(t[v][0]) - PAIRED_SPAN_SLACK_S <= tk <= float(t[v][-1]) + PAIRED_SPAN_SLACK_S


def recording_at(ep: dict, v: str, k: int) -> bool:
    """Whether camera v has a frame to show at anchor instant k: inside its own recording (_in_span), and decoded
    there (frames, ep["no_frame"])."""
    if k in (ep.get("no_frame") or {}).get(v, ()):
        return False
    return _in_span(ep, v, k)


def timesteps(ep: dict, pl: dict, imgs: dict, cell_w: int, quality: int = 90):
    """[(t_s, [(camera_name, jpeg bytes), ...]), ...] in time order, cameras in a fixed order; a camera with no
    frame at an instant (recording_at) is left out there, so its grid cell stays empty."""
    vs = order_views(imgs)
    out = []
    for k in pl["ks"]:
        out.append((frame_time(ep, k), [(cam_name(ep, v), mf.to_jpeg(imgs[v][k], cell_w, quality)) for v in vs
                                        if recording_at(ep, v, k)]))
    return out


def detail_size(w: int, h: int) -> tuple[int, int]:
    if w <= DETAIL_MAX_W:
        return w, h
    return DETAIL_MAX_W, int(round(h * DETAIL_MAX_W / w / 2)) * 2


def fullres_stack(ep: dict, imgs: dict, k: int, label: str, t_s: float, only: list[str] | None = None) -> bytes:
    """All cameras (or `only` these) at instant k, at detail size (native, capped at DETAIL_MAX_W wide), stacked
    top to bottom with a name strip."""
    from PIL import Image, ImageDraw
    vs = [v for v in order_views(imgs) if only is None or v in only]
    ims = [imgs[v][k] for v in vs]
    ims = [im if im.width <= DETAIL_MAX_W else im.resize(detail_size(im.width, im.height), Image.LANCZOS)
           for im in ims]
    w = max(i.width for i in ims)
    strip = 26
    g = Image.new("RGB", (w, sum(i.height + strip for i in ims)), (18, 18, 20))
    d = ImageDraw.Draw(g)
    y = 0
    for v, im in zip(vs, ims):
        d.text((6, y + 5), f"{cam_name(ep, v)}   {label}   t={t_s:.2f}s", fill=(255, 220, 0))
        g.paste(im, (0, y + strip))
        y += im.height + strip
    buf = io.BytesIO()
    g.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


DEPTH_SCALE_INSTANTS = 12   # sampled instants whose depth readings set a camera's colour scale


def _depth_frames(ep: dict, pl: dict) -> dict:
    """{view: {k: depth array}} at the detail instants (first, contact, last), and ep["depth_range"] {view: (near, far)}
    from those and up to DEPTH_SCALE_INSTANTS sampled instants spread over the episode."""
    from label import depth as dp
    d = ep.get("depth") or {}
    if not d:
        return {}
    ks = pl["ks"]
    detail = sorted({ks[0], ks[-1], *(pl.get("contact") or [])})
    spread = [ks[int(i)] for i in np.linspace(0, len(ks) - 1, min(DEPTH_SCALE_INSTANTS, len(ks)))]
    out, rng = {}, {}
    for v in order_views(d):
        got = dp.at_anchor(ep, d, v, sorted(set(detail) | set(spread)))
        # the upload's range for depth with no stated unit (prepare/formats.py measure_depth_ranges), else the
        # range of this episode's readings
        r = tuple(d[v]["range"]) if d[v].get("range") else dp.scale_range(got.values())
        if r is None:
            continue
        rng[v] = r
        out[v] = {k: got[k] for k in detail if k in got}
    ep["depth_range"] = rng
    return out


def depth_stack(ep: dict, depth_at: dict, vs: list[str], k: int, label: str) -> bytes:
    """The depth pictures of cameras vs at instant k, coloured on each camera's episode scale (label/depth.py), at
    detail size, stacked top to bottom with a name strip that gives the scale."""
    from PIL import Image, ImageDraw
    from label import depth as dp
    ims = []
    for v in vs:
        im = dp.picture(depth_at[v][k], ep["depth"][v], ep["depth_range"].get(v))
        if im.width > DETAIL_MAX_W:
            im = im.resize(detail_size(im.width, im.height), Image.NEAREST)
        ims.append((v, im))
    w = max(im.width for _, im in ims)
    strip = 26
    g = Image.new("RGB", (w, sum(im.height + strip for _, im in ims)), (18, 18, 20))
    dr = ImageDraw.Draw(g)
    y = 0
    for v, im in ims:
        dr.text((6, y + 5), f"{cam_name(ep, v)} depth   {label}   t={frame_time(ep, k):.2f}s", fill=(255, 220, 0))
        g.paste(im, (0, y + strip))
        y += im.height + strip
    return dp.to_jpeg(g)


def _contact_views(ep: dict, c: dict) -> tuple[list[str], list[str], list[str]]:
    """(the camera the strips show, the cameras the strongest moment shows, those of them with depth). The strips show
    the camera closest to the touch: the head camera on a person, the camera on the contact's own arm or gripper when
    there is one, else the scene camera. The strongest moment adds the scene camera, so where the object is stays
    clear."""
    vs = views(ep)
    if rig(ep) == "ego_head":
        near = vs[:1]
    else:
        own = [v for v in vs if v in MOUNTED and c.get("hand") == v]
        mounted = [v for v in vs if v in MOUNTED]
        near = own or (mounted[:1] if len(mounted) == 1 and c.get("hand") is None else []) or vs[:1]
    peak = near + [v for v in vs[:1] if v not in near]
    return near, peak, [v for v in peak if v in (ep.get("depth") or {})]


def _frame_at(ep: dict, t_s: float) -> int:
    n = int(len(ep["state"]))
    ts = np.array([frame_time(ep, k) for k in range(n)])
    return int(np.clip(np.argmin(np.abs(ts - t_s)), 0, n - 1))


def contact_image(ep: dict, c: dict, gate=None) -> tuple[bytes, dict] | None:
    """(picture, strips) of one contact: for each camera shown (_contact_views) five frames around the signal's begin
    (STRIP_OFFSETS_S) and three around its end (END_OFFSETS_S), each strip only when the contact has it in the clip
    (contact_strips), then the strongest moment with the cameras' depth and
    the touch sensor's maps (the contact's 2-D signals, bright away from rest on the upload's scale). strips gives each
    strip's frame times ({"begin": [...], "end": [...]}), which checks/contacts.py reads the model's frame numbers
    against. None when its frames cannot be decoded."""
    from PIL import Image, ImageDraw
    from label import depth as dp
    vs, pvs, dvs = _contact_views(ep, c)
    t_end = frame_time(ep, len(ep["state"]) - 1)
    strips = []
    begin, end = contact_strips(c)
    # a contact placed from both starts says so on its picture: its begin and end are the placement's, not recorded
    by = ", placed from both starts" if c.get("aligned_by") else ""
    if begin:
        strips.append((f"touch begins by the recording{by}",
                       [min(max(c["start_s"] + o, 0.0), t_end) for o in STRIP_OFFSETS_S]))
    if end:
        strips.append((f"touch ends by the recording{by}",
                       [min(max(c["end_s"] + o, 0.0), t_end) for o in END_OFFSETS_S]))
    kp = _frame_at(ep, c["peak_s"])
    ks = sorted({_frame_at(ep, t) for _, ts in strips for t in ts} | {kp})
    try:
        got = {v: _decode_view(ep, v, ks if v in vs else [kp], gate) for v in dict.fromkeys(vs + pvs)}
    except Exception:
        return None
    font = mf._grid_font(16)
    pad, strip_h = 8, 24
    blocks = []                                   # (title, [(label, PIL)] rows)
    for title, ts in strips:
        rows = []
        for v in vs:
            cells = []
            for i, t in enumerate(ts):
                k = _frame_at(ep, t)
                if not recording_at(ep, v, k) or k not in got[v]:
                    continue
                im = got[v][k]
                im = im.resize((STRIP_CELL_W, int(round(im.height * STRIP_CELL_W / im.width))))
                cells.append((f"{i + 1}   {frame_time(ep, k):.2f}s", im))
            rows.append((cam_name(ep, v), cells))
        blocks.append((title, rows))
    peak = []
    for v in pvs:
        im = got[v].get(kp)
        if im is None or not recording_at(ep, v, kp):      # past the camera's last frame, or where it does not decode
            continue
        peak.append((f"{cam_name(ep, v)}   {frame_time(ep, kp):.2f}s",
                     im.resize((PEAK_CELL_W, int(round(im.height * PEAK_CELL_W / im.width))))))
    if dvs:
        d = ep.get("depth") or {}
        rng = ep.get("depth_range") or {}
        for v in dvs:
            fr = dp.at_anchor(ep, d, v, [kp]).get(kp)
            if fr is not None:
                im = dp.picture(fr, d[v], rng.get(v))
                size = (PEAK_CELL_W, int(round(im.height * PEAK_CELL_W / im.width)))
                peak.append((f"{cam_name(ep, v)} depth", im.resize(size, Image.NEAREST)))
    for nm in c["signals"]:
        tile = _map_tile(ep, nm, kp)
        if tile is not None:
            peak.append((nm, tile))
    blocks.append(("strongest", [("", peak)]))
    widths = [sum(im.width for _, im in cells) + pad * max(len(cells) - 1, 0)
              for _, rows in blocks for _, cells in rows]
    w = max(widths + [640]) + 2 * pad
    h = sum(strip_h + sum(strip_h + max((im.height for _, im in cells), default=0) for _, cells in rows)
            for _, rows in blocks) + strip_h
    g = Image.new("RGB", (w, h), (18, 18, 20))
    dr = ImageDraw.Draw(g)
    dr.text((pad, 4), f"contact {c['id']}   {c.get('hand') or 'hand not named'}   {', '.join(c['signals'])}   "
                      f"{c['start_s']:.2f}-{c['end_s']:.2f}s", font=font, fill=(255, 220, 0))
    y = strip_h
    for title, rows in blocks:
        dr.text((pad, y + 4), title, font=font, fill=(255, 220, 0))
        y += strip_h
        for cam, cells in rows:
            x = pad
            for lab, im in cells:
                dr.text((x + 2, y + 4), (cam + "   " if cam and x == pad else "") + lab, font=font,
                        fill=(230, 230, 230))
                g.paste(im, (x, y + strip_h))
                x += im.width + pad
            y += strip_h + max((im.height for _, im in cells), default=0)
    times = {("begin" if "begins" in title else "end"): [round(frame_time(ep, _frame_at(ep, t)), 3) for t in ts]
             for title, ts in strips}
    return mf.to_jpeg(g, None, 88), times


def _map_tile(ep: dict, name: str, k: int):
    """A 2-D touch signal at frame k as grey cells, bright away from rest on the upload's scale; None for a signal that
    is not a map."""
    from PIL import Image
    from label import signals as sg
    m = (ep.get("signal_meta") or {}).get(name) or {}
    shape = m.get("shape") or []
    if len(shape) != 2 or name not in (ep.get("signals") or {}):
        return None
    a = ep["signals"][name]
    d, swing = sg.distance_at(a, min(k, len(a) - 1), m.get("rest"), m.get("swing"))
    row = d.reshape(int(shape[0]), int(shape[1]))
    g = np.where(np.isfinite(row), np.clip(row / swing, 0, 1) if swing > 0 else 0.0, 0.0)
    cell = max(1, MAP_TILE_PX // max(int(shape[0]), int(shape[1])))
    return Image.fromarray((g * 255).astype(np.uint8), "L").resize((int(shape[1]) * cell, int(shape[0]) * cell),
                                                                    Image.NEAREST).convert("RGB")


def _rig_nouns(r: str) -> dict:
    if r == "ego_head":
        return {"actor": "hand", "an_actor": "a hand", "actors": "hands", "gripper_of": "hand",
                "who": "a person wears a camera on their head and works with their own two hands"}
    if r == "teleop_arms":
        return {"actor": "arm", "an_actor": "an arm", "actors": "arms", "gripper_of": "arm's gripper",
                "who": "a person teleoperates the robot arms"}
    return {"actor": "gripper", "an_actor": "a gripper", "actors": "grippers", "gripper_of": "handheld gripper",
            "who": "a person holds handheld grippers and does the task with them"}


def _camera_line(ep: dict, v: str) -> str:
    """One camera's facts, and its lens when its frames show a circular image (label/lens.py)."""
    line = _camera_facts(ep, v)
    if ((ep.get("lens") or {}).get(v) or {}).get("circular"):
        line += f" It has {lens.DESC}."
    return line


def _camera_facts(ep: dict, v: str) -> str:
    """A description written at prep time (verified for that dataset) wins; the fallback says only what the
    camera's slot implies: on the left/right gripper, or not on one."""
    cam = (ep["context"].get("cameras") or {}).get(v) or {}
    if cam.get("desc"):
        return f"- {cam_name(ep, v)}: {cam['desc'].rstrip('.')}."
    n = _rig_nouns(rig(ep))
    if v == "exo" and rig(ep) == "ego_head":
        return f"- {cam_name(ep, v)}: the camera worn on the person's head."
    if v == "exo":
        return (f"- {cam_name(ep, v)}: a camera that is not mounted on any {n['actor']}. Use it for the "
                "scene layout, object locations and where things end up.")
    if v not in MOUNTED:
        return (f"- {cam_name(ep, v)}: another camera the recording has; the dataset does not say where it is "
                "mounted, so read that from its frames.")
    side = "" if len(views(ep)) == 1 or v not in ("left", "right") else f"{v.upper()} "
    return (f"- {cam_name(ep, v)}: the camera mounted on the {side}{n['gripper_of']}. When its own gripper's fingers "
            "are in view, they sit in the same place in every frame, usually along the bottom edge, and change only by "
            "opening and closing; "
            "the rest of the image moves whenever the gripper moves.")


def camera_desc(ep: dict, recorded: bool = True) -> str:
    vs, r = views(ep), rig(ep)
    n = _rig_nouns(r)
    cams = ep["context"].get("cameras") or {}
    res = sorted({f"{c.get('width')}x{c.get('height')}" for c in cams.values() if c.get("width")})
    rec = f" (recording {', '.join(res)} at {ep_fps(ep):g} fps)" if res else ""
    count = "There is exactly 1 camera; every grid row and every image strip is labelled with its name" \
        if len(vs) == 1 else (f"There are exactly {len(vs)}; every grid row and every image strip is labelled "
                              "with one of these names")
    s = (f"Cameras in this episode, as named in the dataset{rec}. {count}:\n"
         + "\n".join(_camera_line(ep, v) for v in vs) + "\n")
    if not any(v not in MOUNTED for v in vs):
        mounted = [v for v in vs if v in MOUNTED]
        s += (f"No camera in this episode is off the {n['actor'] if len(mounted) == 1 else n['actors']}: the "
              f"whole scene is seen only through {'that camera' if len(mounted) == 1 else 'those cameras'}, so "
              "reconstruct the layout and where things end up from what it shows.\n")
    if r == "ego_head":
        s += ("This camera description is dataset metadata, not guaranteed truth: check it against the pixels. "
              "If the stream contradicts it (a view that stays fixed instead of moving with the person's head, "
              "a black, frozen or corrupted stream), describe what the view actually is and record it as a data "
              "issue. Where the camera is worn and where it points are read from the frames: a head camera turns "
              "and tilts with the head, so it shows wherever the person looks, and a camera that is really on the "
              "chest or held in the hand, or a mount that slips and tilts the view partway through, is worth "
              "recording.")
    else:
        s += ("These camera identities are dataset metadata, not guaranteed truth: check them against the "
              "pixels. If a stream's content contradicts its name (a mounted camera that shows a fixed view "
              f"or the reverse, two {n['actor']} streams swapped or identical, a black or frozen stream), "
              "describe what the view actually is and record it as a data issue.")
    if r != "ego_head" and any(v in MOUNTED for v in vs):
        s += (f" A mounted camera turns with its {n['actor']}, so where it looks changes through the episode: "
              "sometimes down onto the work, sometimes along or across it. Work out its direction at each instant "
              "from the frame itself (the perspective of the table or floor, which faces of an object are in view, "
              f"where walls or the room appear) and from where that {n['actor']} is in the other views at the same "
              "instant, and read heights, contacts and which face of an object is up from that geometry, never from "
              "an assumed direction. The "
              "same object seen from another direction shows other faces although nothing about it changed: a face "
              "that fills a view looking along the table is a side, and only a view looking down on an object shows "
              "its top. So an object changed state only when views from comparable directions, or the camera that "
              "is not mounted, show the change, never because a mounted camera now sees it from elsewhere.")
    if "left" in vs and "right" in vs:
        s += (f" In the output, \"left\", \"right\" and \"both\" name the streams: an action is \"left\" when "
              "the left stream's own gripper makes the contact, \"right\" likewise, "
              f"\"both\" when the two act together. The other {n['actor']} often appears inside a view, and an "
              "object lying between open fingers is not yet held, so neither is a contact of that camera's own. "
              "This naming is bookkeeping only; it does not settle whether the names are right. Whether each "
              "stream really sits on the side its name says is a separate question for the pixels: where the "
              f"other {n['actor']} and the scene appear in it once you have worked out from the frame how that camera "
              "is turned at that instant" + (", and which recorded motion its view follows." if recorded else "."))
    elif r == "ego_head":
        s += (" In the output, \"left\", \"right\" and \"both\" name the person's own left and right hands, "
              "as seen from their head. Which hand is which follows the person's body (the forearm it belongs to, "
              "the thumb side), not which half of the image it is in, because hands cross the midline and reach "
              "across.")
    elif len([v for v in vs if v in MOUNTED]) == 1:
        s += f" In the output, the \"arm\" field always names the one {n['actor']}: \"{actors(ep)[0]}\"."
    return s


def _detail_desc(native: tuple) -> str:
    try:
        w, h = int(native[0]), int(native[1])
    except (TypeError, ValueError):
        return f"up to {DETAIL_MAX_W} px wide"
    dw, dh = detail_size(w, h)
    return f"the full {w}x{h}" if (dw, dh) == (w, h) else f"{dw}x{dh} (the recording is {w}x{h})"


def _cell_sizes(ep: dict, cell_w: int, cell_h: int) -> str:
    """The grid cell size: every camera is cut to cell_w wide with its own aspect kept (label/frames.py to_jpeg), so
    cameras of different aspect get cells of different height, and each is named then. A camera narrower than
    cell_w is sent at its own size, never enlarged, and said to be."""
    cams = ep["context"].get("cameras") or {}
    sizes, small = {}, set()
    for v in views(ep):
        c = cams.get(v) or {}
        try:
            w, h = int(c["width"]), int(c["height"])
        except (KeyError, TypeError, ValueError):
            return f"{cell_w}x{cell_h}"
        if w < cell_w:
            sizes[cam_name(ep, v)] = (w, h)
            small.add(cam_name(ep, v))
        else:
            sizes[cam_name(ep, v)] = (cell_w, int(round(h * cell_w / w / 2)) * 2)
    native = lambda n: " at its own size, not enlarged" if n in small else ""
    if not small:
        if len({h for _w, h in sizes.values()}) <= 1:
            return f"{cell_w}x{cell_h}"
        return f"{cell_w} px wide (" + ", ".join(f"{n} {w}x{h}" for n, (w, h) in sizes.items()) + ")"
    if len(sizes) == 1:
        (n, (w, h)), = sizes.items()
        return f"{w}x{h}{native(n)}"
    return f"at most {cell_w} px wide (" + ", ".join(f"{n} {w}x{h}{native(n)}" for n, (w, h) in sizes.items()) + ")"


def _coverage_note(ep: dict, pl: dict) -> str:
    """A camera that has no frame at some instants (recording_at), said so its empty cells are read as what they are:
    when it records (_in_span), the instants after its file ends (frames), and the instants its file could not be
    decoded at (all of them for a file none of whose frames decodes). One camera can have more than one of these, and
    each is said. When every camera's file ends before the episode does, the last instant is the last frame they have
    (frames, ep["footage_end"]), which is said too."""
    gaps, ended, broken, never = [], [], [], []
    at = lambda ks: ", ".join(f"{frame_time(ep, k):.2f} s" for k in sorted(ks))
    for v in views(ep):
        if all(recording_at(ep, v, k) for k in pl["ks"]):
            continue
        name = cam_name(ep, v)
        if not all(_in_span(ep, v, k) for k in pl["ks"]):
            t = ep["times"][v]
            gaps.append(f"{name} has frames only from {float(t[0]):.2f} s to {float(t[-1]):.2f} s")
        # the instants of the request it could not decode (one no camera could show has left the request)
        bad = set((ep.get("decode_failed") or {}).get(v) or ()) & set(pl["ks"])
        if v in (ep.get("undecodable") or ()):
            never.append(f"{name}'s video could not be decoded at any instant")
        elif bad:
            broken.append(f"{name}'s video could not be decoded at {at(bad)}")
        ends = {k for k in (ep.get("no_frame") or {}).get(v, ()) if k not in bad and _in_span(ep, v, k)}
        if ends:
            ended.append(f"{name}'s video ends before the episode does, so it has no frame at {at(ends)}")
    out = ""
    span_tail = ("so {its} cells are empty at the instants outside that time, and {it} {is_} left out of a detail view "
                 "there.")
    gone_tail = "{Its} cells at those times are empty, and {it} {is_} left out of a detail view there."
    never_tail = "{Its} cells are all empty, and {it} {is_} left out of every detail view."
    for parts, tail in ((gaps, span_tail), (ended + broken, gone_tail), (never, never_tail)):
        if not parts:
            continue
        one = len(parts) == 1
        words = {"its": "its" if one else "their", "Its": "Its" if one else "Their", "it": "it" if one else "they",
                 "is_": "is" if one else "are"}
        s = "; ".join(parts)
        lead = " " + s[0].upper() + s[1:]
        out += lead + (", " if tail.startswith("so") else ". ") + tail.format(**words)
    if ep.get("footage_end") is not None:
        out += (" Every camera's video ends before the episode does, so the last instant is the last frame they have, "
                f"at {frame_time(ep, ep['footage_end']):.2f} s.")
    return out


def _num(x: float) -> str:
    return f"{float(x):.3g}"


def _depth_note(ep: dict) -> str:
    """Which cameras also record depth, and that each detail view is followed by their depth: what the input is, never
    how to read it."""
    d = ep.get("depth") or {}
    if not d:
        return ""
    names = ", ".join(cam_name(ep, v) for v in order_views(d))
    return (f"\nDEPTH: {names} {'records' if len(d) == 1 else 'record'} depth as well as colour. Each detail view is "
            "followed by the depth from the same "
            + ("camera's depth sensor" if len(d) == 1 else "cameras' depth sensors") + " at that instant.")


CONTACT_VIEWS_MAX = 8             # contacts shown per episode, the strongest first, at most one per CONTACT_EVERY_S
STRIP_OFFSETS_S = (-0.3, -0.15, 0.0, 0.15, 0.3)   # the frames of the begin strip, around the signal's begin
END_OFFSETS_S = (-0.15, 0.0, 0.15)                # the frames of the end strip: the begin strip pins the clock
STRIP_CELL_W = 192
PEAK_CELL_W = 288
MAP_TILE_PX = 160


def contact_strips(c: dict) -> tuple[bool, bool]:
    """(begin, end): whether a contact's picture (contact_image) has a begin strip and an end strip, and so whether the
    prompt describes each and asks for its frame (contacts_block). A contact already touching at the first frame
    (from_start) has no begin in the clip, and one still touching at the last (to_end) no end: label/pieces.py
    write_pieces sets them for every contact that crosses one of our cuts."""
    return not c.get("from_start"), not c.get("to_end")


def chosen_contacts(ep: dict, contacts: list[dict]) -> list[dict]:
    """The contacts shown to the model: the strongest, at most CONTACT_VIEWS_MAX and one per CONTACT_EVERY_S of
    footage, in time order."""
    if not contacts:
        return []
    dur = frame_time(ep, len(ep["state"]) - 1) if len(ep["state"]) else 0.0
    cap = min(CONTACT_VIEWS_MAX, max(1, int(round(dur / CONTACT_EVERY_S))))
    best = sorted(contacts, key=lambda c: -float(c.get("peak_strength") or 0))[:cap]
    return sorted(best, key=lambda c: c["start_s"])


# a contact timed by a signal placed from both starts (label/contacts.py mark_aligned): its times are the placement's
CONTACT_ASSUMED = (" (these times are placed from both starts, as the touch signal shares no clock with the cameras, "
                   "so they are not recorded times)")


def _contact_line(c: dict) -> str:
    hand = f"{c['hand']} hand" if c.get("hand") else "hand not named by the recording"
    when = (f"{c['start_s']:.2f} s" + (" (already touching at the first frame)" if c.get("from_start") else "")
            + f" to {c['end_s']:.2f} s" + (" (still touching at the last frame)" if c.get("to_end") else "")
            + f", strongest at {c['peak_s']:.2f} s"
            + (CONTACT_ASSUMED if c.get("aligned_by") else ""))
    where = []
    for nm, r in (c.get("regions") or {}).items():
        if nm == "active_signals":
            continue
        where.append(f"{nm}: {r['cells']} cells, rows {r['rows'][0]}-{r['rows'][1]} and columns "
                     f"{r['columns'][0]}-{r['columns'][1]} of {r['of'][0]} x {r['of'][1]}")
    act = (c.get("regions") or {}).get("active_signals")
    if act:
        where.append("active: " + ", ".join(act))
    return (f"  {c['id']}: {hand}, from {', '.join(c['signals'])}, {when}"
            + (f"; at its strongest {'; '.join(where)}" if where else "")
            + (f"; it weakens and comes back at {', '.join(f'{x:.2f}' for x in c['dips_s'])} s"
               if c.get("dips_s") else ""))


def touch_verdicts(ep: dict, n: int) -> frozenset:
    """The names of the episode's signals that measure touch, the one rule labelling uses for it (the contacts shown
    and the per-instant readout), judged once per plan (plan()["touch"]): one judgement of a 16 x 16 glove takes over a
    second, and judging again at every presence test cost a 450 s episode about two minutes per request. A part of a
    long recording carries the whole recording's verdict in its signal entry ("touch", label/pieces.py write_pieces),
    because touch is judged once, on the whole recording: a part that falls inside a long press has no rest of its
    own, and its slice alone would not read as touch. Any other signal is judged by label/signals.py is_touch (its name
    says so and its numbers behave like touch, on the upload's scale) over its first n frames, the frames the prompt
    covers (plan()["n"])."""
    from label import signals as sg
    meta = ep.get("signal_meta") or {}
    out = set()
    for name, a in (ep.get("signals") or {}).items():
        m = meta.get(name) or {}
        # read as stored (a float32 skin is never copied whole as float64, label/signals.py CHUNK_VALUES)
        if m["touch"] if "touch" in m else sg.is_touch(name, a[:n], m.get("rest"), m.get("swing")):
            out.add(name)
    return frozenset(out)


def _touch(ep: dict, pl: dict) -> frozenset:
    """plan()["touch"], or for a plan made by hand without it, the verdicts made now."""
    return pl["touch"] if "touch" in pl else touch_verdicts(ep, pl["n"])


def touch_contacts(ep: dict, pl: dict, contacts) -> list[dict]:
    """The contacts timed by at least one touch signal of the episode (touch_verdicts). A context.json prepared before
    is_touch can hold a contact found from a signal that only behaved like touch (an intervention flag, odometry); it
    is not shown."""
    touch = _touch(ep, pl)
    return [c for c in contacts or [] if any(nm in touch for nm in c.get("signals") or [])]


def contacts_block(ep: dict, pl: dict) -> str:
    """The episode's contacts as the recording gives them (label/contacts.py), what each contact picture shows, and
    what to return for them. Empty when no contact is timed by a touch signal (touch_contacts)."""
    shown = touch_contacts(ep, pl, ep.get("contacts_shown"))
    if not shown:
        return ""
    rest = [c for c in touch_contacts(ep, pl, ep.get("contacts")) if c["id"] not in {x["id"] for x in shown}]
    depth = any(_contact_views(ep, c)[2] for c in shown)
    # each strip is described, and its frame asked for, only for the contacts whose picture has it (contact_strips)
    begin = [c["id"] for c in shown if contact_strips(c)[0]]
    end = [c["id"] for c in shown if contact_strips(c)[1]]
    lacking = lambda have, why: ("" if len(have) == len(shown) else
                                 f" (not for {_and_list([c['id'] for c in shown if c['id'] not in have])}, {why})")
    only = lambda have: "" if len(have) == len(shown) else f"for {_and_list(have)} only, "
    picture = ((["five frames around the time the signal says the touch begins, numbered 1 to 5"
                 + lacking(begin, "already touching at the first frame")] if begin else [])
               + ([f"three{'' if begin else ' frames'} around the time {'it' if begin else 'the signal'} says the "
                   "touch ends, numbered 1 to 3" + lacking(end, "still touching at the last frame")] if end else []))
    return ("\nCONTACTS: the recording's touch signals say a hand is touching something in these spans. They are the "
            "recording's claims, to check against the frames:\n" + "\n".join(_contact_line(c) for c in shown) + "\n"
            + (("  The signals record more contacts that are not shown: "
                + "; ".join(f"{c['id']} {c['start_s']:.2f}-{c['end_s']:.2f} s"
                            + (" placed from both starts" if c.get("aligned_by") else "") for c in rest)
                + ".\n") if rest else "")
            + "After the detail views, each contact above has one picture: " + "".join(p + ", " for p in picture)
            + ("and " if picture else "") + "the moment it is strongest"
            + (" with that camera's depth" if depth else "")
            + " and the touch sensor's reading, each frame with its time. Return, beside the other fields:\n"
            '  "contacts": [{"id": "<c1, ...>", "touch_seen": "yes" | "no" | "unclear", '
            + (f'"first_touch_frame": <{only(begin)}1-5, the first frame of the begin strip in which the hand is '
               'touching, or null>, ' if begin else "")
            + (f'"last_touch_frame": <{only(end)}1-3, the last frame of the end strip in which it is still touching, '
               'or null>, ' if end else "")
            + '"hand": "left" | "right" | "both" | "unclear", "object": "<what it touches>", '
            '"grip": "<how the hand holds or presses it>", "action": "<what the contact does in the task>", '
            '"slip": "yes" | "no" | "unclear", "notes": "<or null>"}], one per contact shown,\n'
            '  "contacts_missing": [{"t_s": <float>, "hand": "left" | "right" | "unclear", "object": "<name>"}], each '
            "moment a hand clearly takes hold of or presses something that no contact of the recording covers.\n")


SIGNAL_TABLE_MAX_CHARS = 12000     # the values at each instant stay under this; the rows that move most are kept


def _signals_table(ep: dict, pl: dict) -> str:
    """The recording's other per-frame numbers (ep["signals"], under the dataset's own names): every one listed once
    with its shape, its value names and the range its values take (label/signals.py describe); over each recorded
    still span how much each one changed (the claim they bear on: a mobile base can drive while the arms are still);
    and the values at every sampled instant (_signal_readout), except a touch signal's, whose timing is given once as
    the episode's contacts (contacts_block). They are shown, not interpreted: the model reads what each is from its
    name and the robot's description."""
    from label import signals as sg
    sig = ep.get("signals") or {}
    if not sig:
        return ""
    meta = ep.get("signal_meta") or {}
    n = pl["n"]
    # as stored: every number below is the one a float64 copy would give (largest and smallest readings are exact in
    # any precision, and their differences are taken in float64), with no whole copy of a large signal
    arrs = {k: a[:n] for k, a in sig.items()}
    lines, still, wherever = [], [], []
    for name, a in arrs.items():
        try:
            if not len(a):
                lines.append(f"  {name}: no rows, so no reading at any frame")
                continue
            if _constant(a):
                # a value repeated wherever it reads (a setting, a calibration, or a sensor that sent nothing new):
                # named once, as the same at every frame only when it reads at every frame
                v = a[np.isfinite(a).all(axis=1)][0] if np.isfinite(a).all(axis=1).any() else np.nanmax(a, axis=0)
                said = name + (f" {_num(v[0])}" if len(v) == 1 else
                               " [" + ", ".join(_num(x) for x in v) + "]" if len(v) <= sg.PER_VALUE_MAX else "")
                gaps = int((~np.isfinite(a)).all(axis=1).sum())
                if gaps:
                    wherever.append(f"{said} (no reading at {gaps} of {len(a)} frames)")
                else:
                    still.append(said)
                continue
            m = meta.get(name) or {}
            lines.append(sg.describe(name, a, m.get("shape"), m.get("names"), rate_hz=m.get("rate_hz"),
                                     fps=ep_fps(ep), aligned_by=m.get("aligned_by")))
        except Exception as e:  # noqa: BLE001 - one signal that cannot be read is named, the others are shown
            lines.append(f"  {name}: could not be read ({type(e).__name__})")
    if still:
        lines.append("  The same at every frame: " + "; ".join(still))
    if wherever:
        lines.append("  The same wherever it reads: " + "; ".join(wherever))
    if pl["spans"]:
        lines.append("  Over each recorded still span, the largest change of any one value of each signal (a signal "
                     "that did not change is left out):")
        for a0, b0 in pl["spans"]:
            ch = []
            for name, a in arrs.items():
                seg = a[a0:b0 + 1]
                with np.errstate(all="ignore"):
                    c = (float(np.nanmax(np.nanmax(seg, axis=0).astype(np.float64)
                                         - np.nanmin(seg, axis=0).astype(np.float64)))
                         if np.isfinite(seg).any() else 0.0)
                if c > 0:
                    ch.append(f"{name} {_num(c)}")
            lines.append(f"    {frame_time(ep, a0):.2f}-{frame_time(ep, min(b0, n - 1)):.2f}s: "
                         + ("; ".join(ch) if ch else "none changed"))
    lines += _readout_of(ep, pl)[0]
    return ("\nOTHER RECORDED SIGNALS: every other number the dataset records per frame, under the dataset's own "
            "name, with the range each of its values takes over the episode (one that never changes is given as its "
            "value). They are not interpreted for you: read what each is from its name and the robot's description "
            "above. Like the rest of the recording they are claims to check against the video; a camera carried by "
            "something they show moving (a mobile base, a torso) moves with it.\n" + "\n".join(lines))


def _signal_readout(ep: dict, pl: dict) -> tuple[list[str], frozenset]:
    """(lines, whole): the values of the signals at every sampled instant, and the signals every row of which is in
    them. A touch signal has no rows (its timing is given once as the episode's contacts, contacts_block, so the frames
    are read on their own first and the contacts are checked against them; touch is the rule the contacts shown follow,
    touch_verdicts), nor has a signal that never changes or has no reading. The rows are ranked by how much their
    values move over the episode (label/signals.py movements) and added in that order until the first that does not
    fit SIGNAL_TABLE_MAX_CHARS, then printed in the signals' own order, so the rows shown are always the ones that move
    most; every signal left out, whole or in part, is named in one line with its size and rate. Until the 2026-10-02
    audit the readout was dropped whole past the budget, so a 66 s ego MCAP with IMU, hand, body and SLAM streams
    showed none of its 15 signals over time."""
    from label import signals as sg
    sig = ep.get("signals") or {}
    meta = ep.get("signal_meta") or {}
    n = pl["n"]
    arrs = {k: a[:n] for k, a in sig.items()}      # as stored; label/signals.py reads them in float64 pieces
    touch = _touch(ep, pl)
    ks = pl["ks"]
    rows = []             # (how much the row's values move, the signal's place, the row's place, signal, label, values)
    for i, (name, a) in enumerate(arrs.items()):
        try:
            if name in touch or not len(a) or not np.isfinite(a).any() or _constant(a):
                continue
            m = meta.get(name) or {}
            got = sg.summary_rows(name, a, ks, m.get("shape"), m.get("names"))
            mv = sg.movements(a)
            by_value = sg.per_value(name, a.shape[1], m.get("shape"), m.get("names"))
        except Exception:  # noqa: BLE001 - named as not read in the signals' list (_signals_table), the rest are given
            continue
        for j, (lb, v) in enumerate(got):
            rows.append((float(mv[j]) if by_value else float(np.median(mv)), i, j, name, lb, v))
    if not rows:
        return [], frozenset()
    lines = []
    head = "    at: " + " ".join(f"{frame_time(ep, k):.2f}" for k in ks)
    text = {(r[1], r[2]): f"    {r[4]}: " + " ".join(r[5]) for r in rows}
    room, chosen = SIGNAL_TABLE_MAX_CHARS - len(head), set()
    for r in sorted(rows, key=lambda r: (-r[0], r[1], r[2])):
        # the first row that does not fit ends the readout: a shorter row that moves less, kept after it, would leave
        # the line below false in saying the rows left out move least
        if len(text[r[1], r[2]]) > room:
            break
        chosen.add((r[1], r[2]))
        room -= len(text[r[1], r[2]])
    if chosen:
        lines.append("  Each signal that changes, at every instant you receive (seconds in the first row; \"-\" is "
                     "no reading):")
        lines += [head] + [text[r[1], r[2]] for r in rows if (r[1], r[2]) in chosen]
    left = {}
    for r in rows:
        if (r[1], r[2]) not in chosen:
            left.setdefault(r[3], []).append(r)
    if left:
        named = _and_list([_left_out(nm, arrs[nm], meta.get(nm) or {}, len(left[nm]), sum(r[3] == nm for r in rows))
                           for nm in left])
        lines.append("  The values at each instant leave out " + named
                     + (", because these move least and there is no more room." if chosen else
                        ", because not even one row fits."))
    return lines, frozenset(r[3] for r in rows if r[3] not in left)


def _readout_of(ep: dict, pl: dict) -> tuple[list[str], frozenset]:
    """The readout episode_text made once for this prompt (pl["readout"]), or, for a block text called on its own,
    made now. Only episode_text sets pl["readout"], and only on its own copy of the plan, never on plan()'s."""
    return pl["readout"] if "readout" in pl else _signal_readout(ep, pl)


def _constant(a: np.ndarray) -> bool:
    """Whether a signal has a reading and each of its values never changes over the episode wherever it reads: named
    once with its value, and given no rows at each instant (_signals_table says where it has no reading)."""
    with np.errstate(all="ignore"):
        return bool(np.isfinite(a).any() and (np.nanmax(a, axis=0) == np.nanmin(a, axis=0)).all())


def _left_out(name: str, a: np.ndarray, m: dict, n_left: int, n_rows: int) -> str:
    """One signal left out of the values at each instant: its name, its size, its rate when known, and how many of its
    rows were left out when some of them were shown."""
    shape = m.get("shape")
    size = (" x ".join(str(int(x)) for x in shape) + " values" if shape and len(shape) > 1
            else f"{a.shape[1]} value{'s' if a.shape[1] > 1 else ''}")
    rate = f", {_num(m['rate_hz'])} Hz" if m.get("rate_hz") else ""
    part = f", {n_left} of its {n_rows} rows" if n_left < n_rows else ""
    return f"{name} ({size}{rate}{part})"


def _and_list(xs: list[str]) -> str:
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


BETWEEN_INSTANTS = (
    "\nBETWEEN INSTANTS: reconstruct, do not smooth. A brief event can fall between two "
    "instants. When the scene in the frames differs between consecutive instants (something held "
    "is released, moved or gone; a contact is made or broken), an event happened in that "
    "interval: place it there and say in its notes that it is inferred. Do not fill an unseen interval with the "
    "expected, competent version of the task; a change that does not fit smooth progress may "
    "be a slip, drop, knock or failed grasp, and should be weighed against the frames before "
    "and after, the other side, and where the object ends up. Where the evidence does not "
    "settle it, say so in a note instead of defaulting to the charitable "
    "reading, and never invent an event when consecutive instants are consistent. An event you "
    "infer between instants but cannot confirm from the frames around it goes in the timeline as "
    "inferred; it is not by itself an operator mistake or a data issue.")


def _motion_table(ep: dict, pl: dict) -> str:
    """The recorded motion between consecutive sampled instants, as a claim to check against the
    cameras. Mounted cameras are rigid on their gripper, so a real move shows in that camera."""
    names, kind, n = actors(ep), state_kind(ep), _rig_nouns(rig(ep))
    st = ep["state"][:pl["n"]]
    sa, sb = pl.get("state_span") or (0, len(st))
    ks = [k for k in pl["ks"] if sa <= k < sb]
    rows = []
    if kind == "ee_pose":
        for r in ms.recorded_motion(st, ks, names):
            parts = [f"{a} {g['move_cm']:.1f}, {g['max_step_cm']:.1f}, {g['turn_deg']:.0f}, "
                     f"{g['open_a']:.2f}>{g['open_b']:.2f}" for a, g in r["grippers"].items()]
            rows.append(f"  {frame_time(ep, r['a']):.2f}-{frame_time(ep, r['b']):.2f}s | " + " | ".join(parts))
        what = ("Each row gives, per gripper: moved (cm, the straight-line distance between the recorded "
                "positions at the two instants), largest single-frame step (cm, the biggest recorded jump "
                "between two consecutive frames inside the interval), turned (deg, the recorded rotation "
                "between the two instants), and the opening at the two instants written start>end")
        clear = ("a view that clearly shifts or turns over an interval where the recorded move is near 0 cm "
                 "AND the recorded turn is near 0 deg (the pose stopped updating); a recorded single-frame "
                 "step of several cm with no jump in the view at that moment")
    else:
        for r in ms.recorded_joint_motion(st, ks, names):
            parts = [f"{a} {g['max_deg']:.1f}, {g['max_step_deg']:.1f}, {g['grip_a']:.2f}>{g['grip_b']:.2f}"
                     for a, g in r["arms"].items()]
            rows.append(f"  {frame_time(ep, r['a']):.2f}-{frame_time(ep, r['b']):.2f}s | " + " | ".join(parts))
        what = ("Each row gives, per arm: joints moved up to (deg, the largest change of any one joint "
                "between the two instants), largest single-frame step (deg, the biggest change of any joint "
                "between two consecutive frames inside the interval), and the gripper value at the two "
                "instants written start>end. Joint angles tell you whether and when an arm moved, not where "
                "its gripper is")
        clear = ("an arm that clearly moves in the video over an interval where every joint is recorded as "
                 "near 0 deg (the recording stopped updating), or the reverse; a single-frame joint step "
                 "of many degrees with no matching jump in the video")
    step_ms = 1000.0 / ep_fps(ep)
    grip = (ep["context"].get("gripper_value") or
            "the dataset's own number (units and direction not documented)").rstrip(".")
    return (
        f"\nRECORDED MOTION, from the dataset's state, one row per interval between consecutive instants "
        f"you receive ({step_ms:.0f} ms per recorded frame). {what}; the gripper/opening value is "
        f"{grip}. This is the recording's claim, not a "
        f"fact: check it against the video. A camera mounted on {n['an_actor']} shifts when that "
        f"{n['actor']} moves. The gripper value is the recorded jaw position; whether anything is held is "
        "read from the frames, never from the value. A disagreement between the recording and the video is "
        f"a data issue only when it is clear: {clear}; or fingers that clearly open or close while the value "
        "stays flat, or a value that swings from open to shut while the fingers stay still. Do not judge "
        "whether a recorded motion is too small or too large from how much a view changes: apparent motion "
        "depends on the lens, the distance to the scene and the direction of travel. Likewise never compare "
        "how open the fingers look with the gripper number: its scale is not a picture of how wide the "
        "fingers look, so only the timing of a change can be compared with the video.\n"
        + (f"The recorded state covers only {frame_time(ep, sa):.2f} s to {frame_time(ep, sb - 1):.2f} s of the "
           "episode, so the rows stop there and nothing is recorded outside it.\n" if (sa, sb) != (0, len(st)) else "")
        + "\n".join(rows))


def ego_annotation_block(ctx: dict) -> str:
    goal = (ctx.get("instruction") or "").strip()
    subs = ctx.get("annotation_subtasks") or []
    if not goal and not subs:
        return ("\nTHE DATASET'S ANNOTATION FOR THIS EPISODE: none; the dataset ships no task description for this "
                "clip. Infer the activities from the footage alone and leave goal_alignment out.\n")
    def when(x):            # a step with no end time is a moment, one with no time is listed without one
        t0, t1 = number(x.get("t0")), number(x.get("t1"))
        return ("no time" if t0 is None else f"{t0:.1f}s" if t1 is None or t1 == t0 else f"{t0:.1f}-{t1:.1f}s")
    lines = [f"  {when(x)}  {x['label']}" + ("" if x.get("ok", True) else "  (marked unsuccessful)")
             for x in subs if isinstance(x, dict)]
    return ("\nTHE DATASET'S ANNOTATION FOR THIS EPISODE (claims to check, see ABOUT THE DATASET'S ANNOTATION above):\n"
            + (f"  goal: \"{goal}\"\n" if goal else "")
            + ("  subtasks, with the times the dataset gives:\n" + "\n".join(lines) + "\n" if lines else "")
            + (f"  about these annotations: {ctx['annotation_note'].strip()}\n" if ctx.get("annotation_note") else ""))


# ---------------------------------------------------------------- the episode prompt: a base and its blocks

# Where a block's text goes in the episode prompt. The base fills the rest: the intro (who and what), the camera
# paragraph, the frames paragraph, the sampling line, BETWEEN_INSTANTS and the task.
PROMPT_SLOTS = ("intro", "frames_detail", "frames", "state", "signals", "after_frames", "after_task")


@dataclass(frozen=True)
class Block:
    """One part of the episode prompt that exists only when the episode holds its data. present(ep, pl) tests the
    loaded episode folder and the plan made from it, never the rig or the dataset's name; text(ep, pl) is what the
    model is told; schema_fields are the output fields the text asks for beyond the rig's shared schema
    (label/prompts.py); checks are the deterministic checks the same data feeds (reported, never told to the model),
    under their keys in the episode's dataset_checks (plan's own, and those board/build.py add_context copies from
    context.json and add_contacts writes).
    An episode without the data gets neither the text nor the fields."""
    name: str
    slot: str
    present: Callable[[dict, dict], bool]
    text: Callable[[dict, dict], str]
    schema_fields: tuple = ()
    checks: tuple = ()


def _has_signals(ep: dict, pl: dict) -> bool:
    return bool(ep.get("signals"))


def _has_state(ep: dict, pl: dict) -> bool:
    return state_kind(ep) != "none" and bool(pl.get("state_usable", True))


def _state_unaligned(ep: dict, pl: dict) -> bool:
    return state_kind(ep) != "none" and not pl.get("state_usable", True)


def _has_contacts(ep: dict, pl: dict) -> bool:
    return bool(touch_contacts(ep, pl, ep.get("contacts_shown")))


def _no_state(ep: dict, pl: dict) -> bool:
    return state_kind(ep) == "none"


def _has_collection_note(ep: dict, pl: dict) -> bool:
    return bool(ep["context"].get("collection_note"))


def _has_contact_views(ep: dict, pl: dict) -> bool:
    return bool(pl.get("contact"))


def _has_coverage(ep: dict, pl: dict) -> bool:
    return bool(_coverage_note(ep, pl))


def _has_depth(ep: dict, pl: dict) -> bool:
    return bool(ep.get("depth"))


def _has_uploader_notes(ep: dict, pl: dict) -> bool:
    return bool(ep["context"].get("uploader_annotation"))


def _collection_text(ep: dict, pl: dict) -> str:
    return f"How the dataset cuts its recordings into episodes: {ep['context']['collection_note'].strip()}\n"


def _contact_views_text(ep: dict, pl: dict) -> str:
    return (" So are the instants just after a gripper's recorded value changes sharply, where something is "
            "usually picked up or put down, from the scene camera and that gripper's own camera ("
            + ", ".join(f"{frame_time(ep, k):.2f}" for k in pl["contact"]) + " s): use them to read what "
            "is held and how it is left. The value only chose where to look closer; what is held is read from "
            "the frames.")


def _state_text(ep: dict, pl: dict) -> str:
    """The recorded state, as the recording's claims: its still spans and the motion between consecutive instants."""
    r, kind = rig(ep), state_kind(ep)
    n = _rig_nouns(r)
    src = ("joint encoders" if kind == "joints" else
           "recorded end-effector poses" if r == "teleop_arms" else "tracked gripper poses")
    if pl["spans"]:
        sp = ", ".join(f"{d['start_s']:.2f}-{d['end_s']:.2f}s" for d in describe_spans(ep, pl["spans"]))
        s = (f"\nRECORDED STILL SPANS, from the dataset's {src}: {sp}. Over each span the recording says "
             f"no {n['actor']} moved and none opened or closed. This is the recording's claim, not a "
             f"fact: check it. {'An' if n['actor'][0] in 'aeiou' else 'A'} {n['actor']} that is really still "
             "shows a steady view in its own camera"
             + (" unless something that carries it moves, which the other recorded signals below may show; the "
                f"claim covers only the {n['actors']}" if _has_signals(ep, pl) else "")
             + ". If the views show motion during a span, the recording is "
             f"wrong there. If the views hold steady and the scene still changes, the {n['actors']} did not do it: "
             "say what you see.")
    else:
        s = f"\nRECORDED STILL SPANS, from the dataset's {src}: none."
    return s + _motion_table(ep, pl)


def _state_unaligned_text(ep: dict, pl: dict) -> str:
    return ("\nRECORDED STATE: not given. This episode's cameras do not cover the same frames "
            "as its recorded state, so the state cannot be aligned to the video.")


# What in context["source"] says the reader found sensor data it did not read (prepare/formats.py write_signals,
# convert_hdf5, plan_video): the arrays and signals it left out, and "sensors", the sensor files whose signals it read
# (prepare/formats.py convert_video), so an episode with no signal from them has none read. With any of it, the
# episode is never told its dataset records no state.
UNREAD_SOURCE_KEYS = ("unused_signals", "unused_arrays", "sensors")


# Why an episode has no arm state, as the reader records it in context.json state_why beside its state_note, each with
# the reason the RECORDED STATE line gives for it (_no_state_text):
#   layout         the state is recorded, but not in a layout our checks read: the layout line, no reason
#   not_recorded   the recording holds no state at all
#   unreadable     a sensor file holding the state could not be read, or was damaged before any message
#   short          an arm's state does not cover the footage (one arm's file cut short leaves no state at all)
#   assumed_clock  the state's channels are only on a clock placed from both starts, so they never become state
STATE_WHY = {
    "layout": None,
    "not_recorded": "as the recording holds none",
    "unreadable": "as a file holding it could not be read",
    "short": "as it does not cover the footage",
    "assumed_clock": "as it is recorded only on a clock placed from both starts, not shared with the cameras",
}


def _no_state_text(ep: dict, pl: dict) -> str:
    """No arm state. With other signals the line says why, as the reader recorded it (state_why, STATE_WHY): "layout"
    says none is in the layout our checks read, and every other reason that no state was read, why, and the reader's
    note on it (not for "not_recorded", whose note only says the same). A context written before the reader recorded
    state_why says the layout line, unless a signal whose name says joints or a state stops short of the episode: then
    the layout is not why (an arm sensor file cut before the footage ends), and the line gives the reader's note on the
    state, which says why. With no other signal, that the dataset records none, unless the reader wrote a note on the
    state or left sensor data unread: then only that none was read, since "records no hand, head or device tracking"
    was false for an MCAP whose hand tracks the reader did not read yet (2026-10-02 audit). There the note and the lists
    of unread channels go to the board, never to the model: they name the checks and channels that did not run ("the
    checks on recorded motion ..."), which would put
    the words about a recorded motion back into a video only prompt; with other signals the prompt is a recording's
    already. A state_why other than "not_recorded" counts as a note: the state is there but was not read, so only that
    none was read; "not_recorded" and no state_why read as before."""
    r = rig(ep)
    n = _rig_nouns(r)
    ctx = ep["context"]
    if _has_signals(ep, pl):
        from label import signals as sg
        meta = ep.get("signal_meta") or {}
        # a signal of several values whose name says joints or a state (label/signals.py names_joints_or_state), and
        # only one whose every value is at each instant in the readout below (_signal_readout): one left out of it in
        # whole or in part, or with no rows there (constant, no reading, touch), is not named. The line says only what
        # is true by construction: what the names say, and "under their own names" only when every value has one
        joints = [nm for nm, a in ep["signals"].items() if nm in _readout_of(ep, pl)[1] and np.shape(a)[1] > 1
                  and sg.names_joints_or_state(nm)
                  and sg.per_value(nm, np.shape(a)[1], (meta.get(nm) or {}).get("shape"),
                                   (meta.get(nm) or {}).get("names"))]
        one = len(joints) == 1
        named = all(len((meta.get(nm) or {}).get("names") or []) == np.shape(ep["signals"][nm])[1] for nm in joints)
        # a joints or state signal with no reading over part of the episode says the state's data stops short (a sensor
        # file cut before the footage ends), so the layout is not why none was read: the reader's own note says why
        short = [nm for nm, a in ep["signals"].items() if sg.names_joints_or_state(nm) and len(a[:pl["n"]])
                 and np.isnan(np.asarray(a[:pl["n"]], dtype=np.float64)).all(axis=1).any()]
        note = (ctx.get("state_note") or "").strip()
        why = ctx.get("state_why")
        if why == "layout" or why is None and not (short and note):
            head = f"no {n['actor']} state in the layout our checks read."
        else:
            reason = STATE_WHY.get(why) if why is not None else None
            # a recording that holds no state needs no note: the reader's note can only say so again (the board
            # shows it)
            head = (f"no {n['actor']} state was read{f', {reason}' if reason else ''}."
                    + (f" The reader's note on it: {note}" if note and why != "not_recorded" else ""))
        return (f"\nRECORDED STATE: {head}"
                + (f" The signal{'' if one else 's'} whose name{' says' if one else 's say'} joints or a state "
                   f"({', '.join(joints)}) {'is' if one else 'are'} given value by value"
                   + (" under their own names" if named else "") + " among the other recorded signals below."
                   if joints else ""))
    src = ctx.get("source") if isinstance(ctx.get("source"), dict) else {}
    # depth pictures follow the detail views of an episode with depth (the depth block), so there the video is not all
    all_there_is = (("the cameras' colour and depth images are" if _has_depth(ep, pl) else "the video is")
                    + " all there is.")
    if ((ctx.get("state_note") or "").strip() or any(src.get(k) for k in UNREAD_SOURCE_KEYS)
            or ctx.get("state_why") not in (None, "not_recorded")):
        return f"\nRECORDED STATE: none was read from this episode, so {all_there_is}"
    what = "no hand, head or device tracking" if r == "ego_head" else "no robot or gripper state"
    return f"\nRECORDED STATE: none; this dataset records {what}, so {all_there_is}"


def _uploader_text(ep: dict, pl: dict) -> str:
    # notes the person who uploaded the episode sent with it (a note file beside a video, an annotation channel in an
    # MCAP), in whatever form they came
    return ("\nTHE UPLOADER'S OWN NOTES FOR THIS EPISODE, as sent. They are claims to check against the "
            "video, not ground truth; where the video contradicts them, record it as a data issue:\n"
            + ep["context"]["uploader_annotation"].strip() + "\n")


BLOCKS = (
    Block("collection_note", "intro", _has_collection_note, _collection_text),
    Block("contact_views", "frames_detail", _has_contact_views, _contact_views_text),
    Block("coverage", "frames", _has_coverage, _coverage_note),
    Block("depth", "frames", _has_depth, lambda ep, pl: _depth_note(ep), checks=("sensor_checks",)),
    Block("state", "state", _has_state, _state_text,
          checks=("timebase", "stream_pairing", "recorded_jumps", "gripper_channels", "capture_qc")),
    Block("state_unaligned", "state", _state_unaligned, _state_unaligned_text, checks=("camera_windows_match_state",)),
    Block("no_state", "state", _no_state, _no_state_text),
    Block("signals", "signals", _has_signals, _signals_table, checks=("sensor_checks",)),
    Block("contacts", "after_frames", _has_contacts, contacts_block,
          schema_fields=("contacts", "contacts_missing"),
          checks=("contact_checks",)),
    Block("uploader_notes", "after_task", _has_uploader_notes, _uploader_text),
)


# The blocks that make an episode a recording rather than video only: with none of them, the shared instructions and
# the camera paragraph say nothing of a recorded motion (label/prompts.py VIDEO_ONLY_CONTRACT_WORDING).
RECORDED_BLOCKS = ("state", "state_unaligned", "signals")


def present_blocks(ep: dict, pl: dict) -> list[Block]:
    """The blocks whose data this episode holds, in prompt order."""
    return [b for b in BLOCKS if b.present(ep, pl)]


def is_recorded(ep: dict, pl: dict, blocks: list[Block] | None = None) -> bool:
    """Whether the episode is a recording rather than video only, from its present blocks (given, or found now)."""
    return any(b.name in RECORDED_BLOCKS for b in (present_blocks(ep, pl) if blocks is None else blocks))


def requested_schema(ep: dict, pl: dict, blocks: list[Block] | None = None) -> tuple:
    """The output fields this episode's blocks ask for beyond the rig's shared schema."""
    return tuple(f for b in (present_blocks(ep, pl) if blocks is None else blocks) for f in b.schema_fields)


def implied_checks(ep: dict, pl: dict, blocks: list[Block] | None = None) -> tuple:
    """The deterministic checks the data of this episode's blocks feeds, each named once (depth and the signals both
    feed sensor_checks)."""
    return tuple(dict.fromkeys(c for b in (present_blocks(ep, pl) if blocks is None else blocks) for c in b.checks))


def _intro_head(ep: dict) -> str:
    ctx = ep["context"]
    r = rig(ep)
    n = _rig_nouns(r)
    robot = ctx.get("robot_type")
    what = f" ({robot})" if robot else ""
    k = len(actors(ep))
    who = n["who"] if r in ("teleop_arms", "ego_head") else (
        # one camera says nothing about how many grippers the rig has: a two-gripper rig's upload can carry one
        # gripper's footage, and its other gripper then appears in that camera, held in the other hand
        "a person does the task with one or two handheld grippers, and this recording has one gripper's camera"
        if k == 1
        else f"a person holds {k} handheld grippers, one per hand, and does the task with them")
    kind_of = ("one clip of first-person human video from the {d} dataset, collected to train robots and world "
               "models" if r == "ego_head" else "one episode of a robot-learning demonstration from the {d} dataset{w}")
    return (f"You are labelling {kind_of.format(d=ctx.get('dataset'), w=what)}: {who}. You see the episode exactly as "
            "the dataset ships it, from its first recorded frame to its last; nothing was trimmed, cleaned or "
            "edited.\n")


def _frames_head(ep: dict, cell_w: int, cell_h: int, native: tuple) -> str:
    names = ", ".join(cam_name(ep, v) for v in views(ep))
    return (
        f"\nFRAMES. You receive the episode as grid images: ROWS are the cameras ({names}, top to "
        "bottom), COLUMNS are instants left to right, and each column is headed with its exact time in "
        "seconds from the episode's first frame. Read each grid left to right and the grids in order. "
        "The times are exact: use them, do not invent your own. Each grid cell is the camera frame "
        f"downscaled to {_cell_sizes(ep, cell_w, cell_h)}. After the grids, the episode's first and last instant are "
        f"repeated larger, at {_detail_desc(native)}; use them for the start and end state and any "
        "small detail (lettering, a display, fine alignment).")


def _instants_line(ep: dict) -> str:
    # when every camera's file ends before the episode does, the last instant is the last frame they have (frames)
    last = "frame and the last frame its cameras have" if ep.get("footage_end") is not None else "and last frame"
    return (f"Which instants you get: one every {SAMPLE_EVERY_S[rig(ep)]:g} s for the whole episode, plus its first "
            f"{last}.")


def task_block(ep: dict) -> str:
    """The task, part of the base: the dataset's instruction (robot rigs) or annotation (head camera). A head camera
    with no annotation is told there is none, because its shared instructions always ask for goal_alignment."""
    ctx = ep["context"]
    if rig(ep) == "ego_head":
        return ego_annotation_block(ctx)
    given = (ctx.get("instruction") or "").strip()
    if not given:
        return ""
    label = "; ".join(ctx.get("task_label") or [])
    # the rules for using the instruction are the same for every episode and live in the cached
    # instructions (instruction_rules); only the instruction itself belongs to the episode
    s = ("\nTHE TASK FOR THIS EPISODE WAS GIVEN TO YOU as the dataset's per-episode instruction:\n"
         f"  \"{given}\"\nHow to use it is set out under ABOUT THE EPISODE'S INSTRUCTION above.\n")
    if ctx.get("instruction_note"):
        s += ctx["instruction_note"].strip() + "\n"
    elif label:
        s += (f"The dataset's coarse task label for this episode is \"{label}\"; the "
              "instruction above is the dataset's per-episode annotation of it, and "
              "the outcome is graded against it.\n")
    return s


def episode_text(ep: dict, pl: dict, cell_w: int, cell_h: int, native: tuple,
                 blocks: list[Block] | None = None) -> str:
    """The episode's part of the prompt: the base, with each present block (given, or found now) in its slot."""
    got = dict.fromkeys(PROMPT_SLOTS, "")
    blocks = present_blocks(ep, pl) if blocks is None else blocks
    if _has_signals(ep, pl):
        # the readout of the signals is made once: the signals block prints it and the state line names only the
        # joint readings it shows whole
        pl = {**pl, "readout": _signal_readout(ep, pl)}
    for b in blocks:
        got[b.slot] += b.text(ep, pl)
    return (EPISODE_HEADER + _intro_head(ep) + got["intro"] + "\n" + camera_desc(ep, is_recorded(ep, pl, blocks))
            + "\n"
            + _frames_head(ep, cell_w, cell_h, native) + got["frames_detail"] + "\n" + _instants_line(ep)
            + got["frames"] + got["state"] + got["signals"] + BETWEEN_INSTANTS + got["after_frames"] + "\n"
            + task_block(ep) + got["after_task"])


def build_prompt(ep: dict, pl: dict, *, cell_w: int, cell_h: int, example_dir=None,
                 blocks: list[Block] | None = None) -> tuple[str, str]:
    """(fixed, episode): the shared instructions (output schema, what the episode is, the data contract),
    identical for every episode of the same rig, instruction presence and recorded or video only variant (is_recorded),
    then the facts about THIS episode (episode_text). The present blocks are found once, unless the caller has them."""
    ctx = ep["context"]
    r = rig(ep)
    c0 = (ctx.get("cameras") or {}).get(anchor(ep), {})
    native = (c0.get("width") or "native", c0.get("height") or "resolution")
    given = (ctx.get("instruction") or "").strip()
    blocks = present_blocks(ep, pl) if blocks is None else blocks
    fixed = prompts.fixed_instructions(r, has_instruction=bool(given), recorded=is_recorded(ep, pl, blocks))
    return fixed + prompts.example_block(r, example_dir), episode_text(ep, pl, cell_w, cell_h, native, blocks)


EPISODE_HEADER = "\n\nTHE EPISODE TO LABEL.\n\n"


def build_request(ep_dir: Path, *, detail: str = "high", gate=None, cell_w: int | None = None,
                  max_cell_w: int | None = None, grid_cols: int = 4, grid_quality: int = 80,
                  example_dir=None) -> dict:
    """Everything the harness sends for one episode (content parts), and what it records about it. cell_w fixes
    the cell width; max_cell_w (the routed width, label/route.py) replaces the rig's default widest cell."""
    ep = load(ep_dir)
    pl = plan(ep)
    from label import contacts as lc
    ep["contacts"] = touch_contacts(ep, pl, lc.of_episode(ep, _touch(ep, pl)))
    ep["contacts_shown"] = chosen_contacts(ep, ep["contacts"])
    if ep["contacts_shown"]:
        pl["contact"] = []   # the touch signals' own contacts replace the views chosen from the gripper's value
    # An explicit cell width is used as given. The rig's default is the largest width whose grids fit the
    # request's image-size cap: a long episode at 448 px can exceed it, and is then sent at the next step
    # down rather than refused. Episodes that fit are unchanged.
    widths = [cell_w] if cell_w else [w for w in CELL_W_STEPS if w <= (max_cell_w or GRID_CELL_W_BY_RIG[rig(ep)])]
    if max(widths) >= CONTACT_BELOW_W:
        pl["contact"] = []   # wide cells already show contact in detail
    # full size is kept only where a detail view shows it: the first and last instant and the contact instants
    imgs = frames(ep, pl, gate, widths=widths, detail_ks={pl["ks"][0], pl["ks"][-1], *(pl.get("contact") or [])})
    any_img = next(im[pl["ks"][0]] for im in imgs.values() if pl["ks"][0] in im)
    # a circular image with black corners names a fisheye lens in that camera's line (label/lens.py)
    ep["lens"] = {v: lens.circular_image(imgs[v]) for v in order_views(imgs)}
    # depth at the detail instants, and one colour scale per camera from its readings at the sampled instants
    depth_at = _depth_frames(ep, pl)
    if len(views(ep)) == 1:
        # one camera: a grid row holds 6 instants (still under 2048 px wide), halving the per-image overhead
        grid_cols = max(grid_cols, 6)
    cam_labels = [cam_name(ep, v) for v in order_views(imgs)]
    contact = [(k, vs, fullres_stack(ep, imgs, k, "just after a sharp gripper change", frame_time(ep, k), vs))
               for k, vs in ((k, [v for v in vs if recording_at(ep, v, k)]) for k, vs in contact_views(ep, pl)) if vs]
    budget = (IMAGE_LIMIT_BYTES / IMAGE_SIZE_INFLATION - DETAIL_VIEW_BYTES_MAX
              - sum(len(j) for _, _, j in contact))
    # the blocks follow from the episode and its plan, never from the cell width: found once, for every width and the
    # record
    blocks = present_blocks(ep, pl)
    for cell_w in widths:
        steps = timesteps(ep, pl, imgs, cell_w)
        cell_h = int(round(any_img.height * cell_w / any_img.width / 2)) * 2
        fixed, episode = build_prompt(ep, pl, cell_w=cell_w, cell_h=cell_h, example_dir=example_dir, blocks=blocks)
        content, n_grids, grid_bytes = mf.build_content(fixed, episode, steps, cam_labels, grid_cols, detail,
                                                        grid_quality, gutter=GRID_GUTTER, header=GRID_HEADER)
        if grid_bytes <= budget:
            break
    prompt = fixed + episode
    extra_bytes = 0
    # the first frame, the contact views in time order, then the last frame
    views_sent = []
    for k, name in ((pl["ks"][0], "first frame"), (pl["ks"][-1], "last frame")):
        here = [v for v in order_views(imgs) if recording_at(ep, v, k)]
        views_sent.append((k, f"{name} of the episode",
                           cam_labels if len(here) == len(imgs) else [cam_name(ep, v) for v in here],
                           fullres_stack(ep, imgs, k, name, frame_time(ep, k), None if len(here) == len(imgs) else here)))
    views_sent[1:1] = [(k, "just after a sharp change of the recorded gripper value",
                        [cam_name(ep, v) for v in order_views(vs)], jpg) for k, vs, jpg in contact]
    depth_sent = []
    for k, what, names, jpg in views_sent:
        extra_bytes += len(jpg)
        # the first and last views name every camera; a contact view names only its own cameras
        cams = (f"cameras {', '.join(names)} stacked top to bottom" if names is cam_labels or len(names) > 1
                else f"camera {names[0]}")
        content.append({"type": "text", "text": f"=== detail view, {what}, t={frame_time(ep, k):.2f}s | {cams} ==="})
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode("ascii"),
            "detail": detail}})
        dv = [v for v in order_views(depth_at) if cam_name(ep, v) in names and k in depth_at[v]]
        if dv:
            djpg = depth_stack(ep, depth_at, dv, k, what)
            extra_bytes += len(djpg)
            depth_sent.append(round(frame_time(ep, k), 3))
            dn = ", ".join(cam_name(ep, v) for v in dv)
            content.append({"type": "text", "text": f"=== depth, {what}, t={frame_time(ep, k):.2f}s | "
                                                    f"{'cameras' if len(dv) > 1 else 'camera'} {dn} ==="})
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(djpg).decode("ascii"), "detail": detail}})
    contacts_sent, strips = [], {}
    for c in ep.get("contacts_shown") or []:
        got = contact_image(ep, c, gate)
        if got is None:
            continue
        cjpg, strips[c["id"]] = got
        extra_bytes += len(cjpg)
        contacts_sent.append(c["id"])
        content.append({"type": "text", "text": f"=== contact {c['id']}, {c.get('hand') or 'hand not named'}, "
                                                f"{c['start_s']:.2f}-{c['end_s']:.2f}s ==="})
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(cjpg).decode("ascii"), "detail": detail}})
    return {"content": content, "prompt": prompt, "plan": pl, "n_grids": n_grids,
            "n_images": n_grids + len(views_sent) + len(depth_sent) + len(contacts_sent),
            "image_bytes": grid_bytes + extra_bytes,
            "contact_s": [round(frame_time(ep, k), 3) for k in pl["contact"]],
            "given_prompt": (ep["context"].get("instruction") or "").strip() or None,
            "task_label": ep["context"].get("task_label"), "cam_labels": cam_labels,
            "cell": [cell_w, cell_h], "timesteps": [round(frame_time(ep, k), 3) for k in pl["ks"]], "lens": ep["lens"],
            "grid_cols": grid_cols, "decode_failed": decode_failures(ep),
            "still_spans": describe_spans(ep, pl["spans"]), "views": views(ep),
            "sampling": f"{rig(ep)}-every-{SAMPLE_EVERY_S[rig(ep)]:g}s",
            "blocks": [b.name for b in blocks],
            "schema_fields": list(requested_schema(ep, pl, blocks)),
            "checks_implied": list(implied_checks(ep, pl, blocks)),
            **({"depth_s": depth_sent, "depth_views": order_views(depth_at)} if depth_sent else {}),
            **({"contact_views": {"shown": contacts_sent, "strips": strips}, "contacts": ep.get("contacts")}
               if contacts_sent else {})}
