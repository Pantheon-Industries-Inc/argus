"""One episode's request: which frames, decoded exactly, and a prompt that states only facts we are sure of.

An episode is a sidecar folder written by a preparer (prepare/): context.json (the dataset's facts: rig, state
kind, fps, cameras and what each one is, the instruction or annotation), sources.json (per camera, the video
file, the episode's offset in it and its frame count), state.npz (recorded state and action, absent for
video-only rigs) and, where the dataset keeps real capture times, times.npz and a kmap per camera paired to the
anchor camera by time. Source and preparation boundaries are kept separate from sampled evidence.

What the model is told about the episode, and where each fact comes from:
- dataset, robot type, fps and camera resolution: recorded metadata and preparation facts (context.json).
- camera labels: presentation aliases, separately from recorded mount identifiers, explicit mounting facts
  and reader-verified descriptions. Profiles and slots do not establish physical camera placement.
- the instruction or annotation: the dataset's per-episode text, as a claim to check.
- still spans: from the recorded state, true by construction (state.still_spans), as the recording's claim.
- recorded motion between consecutive instants: from the recorded state, as a claim to check.
- sampling: exactly what state.sample_frames did.
- which of these are said at all: each part that depends on data the episode may not hold is a block (BLOCKS) with a
  test of the episode folder, so an episode without that data gets no word of it and is asked for no field of it.
- a circular image area: only where its pixels have persistent dark corners (label/lens.py), without asserting optics.
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
from label import prompts, policy
from label import sensor_evidence as se
from label import state as ms
from prepare.signal_alignment import ALIGNED_CAMERA, ALIGNED_ASSUMED, recorder_arrival, placement_text
from prepare.state_notes import ASSUMED_CLOCK, LAYOUT, NOT_RECORDED, SHORT, UNREADABLE

FPS = 30
VIEW_ORDER = ("exo", "left", "right")          # stable presentation slots; their order does not establish content
MOUNTED = ("left", "right")                     # actor-side slots; physical mounting needs separate evidence
EXTRA_VIEW = re.compile(r"extra(\d+)")          # any other camera the recording has (extra1, extra2, ...)
CAM_NAME = {"exo": "top", "left": "left", "right": "right"}   # fallback presentation labels
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
# A camera paired by time is shown at an instant up to this far outside its own first and last frame: a pairing
# tolerance, whether its nearest frame still stands for that instant (a wrist camera whose first frame comes 0.034 s
# after the anchor's), not the edge a stream may miss without a data issue (prepare/formats.py edge_slack)
PAIRED_SPAN_SLACK_S = 0.1
GRID_GUTTER = 84
GRID_HEADER = 30


def is_episode_dir(ep_dir: Path) -> bool:
    return (Path(ep_dir) / "context.json").exists() and (Path(ep_dir) / "sources.json").exists()


def load(ep_dir: Path) -> dict:
    ep_dir = Path(ep_dir)
    ctx = json.loads((ep_dir / "context.json").read_text())
    from prepare.formats import clock_context
    ctx = clock_context(ctx)
    from label.dictionary_editor import for_episode
    owner = (ctx.get("piece") or {}).get("of") or (ctx.get("data_dictionary") or {}).get("episode_id") or ep_dir.name
    resolved = for_episode(ep_dir, owner, private_bindings=True)
    if resolved is not None:
        ctx["data_dictionary"] = resolved
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
    from prepare.state_contract import prepared_state_problem
    problem = prepared_state_problem(ctx, state.shape[1] if state.ndim == 2 else None,
                                     has_action=action is not None)
    if problem:
        raise RuntimeError(f"{ep_dir}: {problem}")
    ep = {"dir": ep_dir, "context": ctx, "sources": src, "state": state,
          "action": action, "times": None, "kmap": {}, "signals": {}}
    if ctx.get("signals") and not ctx.get("state_unaligned"):
        # the recording's other per-frame numbers, under the dataset's names (prepare/formats.py recorded_signals), with
        # each one's shape and value names (a 16 x 16 pressure map; fx, fy, fz) and everything else its reader wrote,
        # so a field a reader adds reaches the checks without being listed here. Signals on the frames of a camera
        # taken out of the episode, with nothing to place them on the others (state_unaligned), are not read. One whose
        # rows stop short of the episode's frames has no reading past them (label/signals.py pad_rows)
        # an image stored with its colour channels by an older preparation reads as the planes a new one stores
        # (prepare/formats.py stored_planes, which the board's signal reader uses too)
        from label import signals as sg
        from prepare.formats import stored_planes
        with np.load(ep_dir / "signals.npz") as z:
            ctx["signals"], rows = stored_planes(ctx["signals"], {s["key"]: z[s["key"]] for s in ctx["signals"]},
                                                 ctx.get("reader_issues"))
        ep["signals"] = {s["name"]: sg.pad_rows(sg.columns(rows[s["name"]]), len(state)) for s in ctx["signals"]}
        ep["signal_meta"] = {s["name"]: {k: v for k, v in s.items() if k not in ("name", "key")}
                             for s in ctx["signals"]}
        from label.dictionary_context import field_interpretation
        for name, meta in ep["signal_meta"].items():
            if recorder_arrival(ctx):
                from prepare.signal_alignment import RECORDER_ARRIVAL
                meta.setdefault('aligned_by', RECORDER_ARRIVAL)
            interpretation = field_interpretation(ctx, name)
            if interpretation:
                meta["dictionary"] = interpretation
    if ctx.get("real_times"):
        # datasets with real per-frame capture times (ABC-130k, RealOmin): every time shown uses them, and
        # each camera's frames are decoded by their exact pts
        from prepare.camera_clock import load_times
        t = load_times(ep_dir, ctx)
        zero = float(ctx.get("clock_zero_s") or 0.0)
        ep["times"] = {k: t[k] if k.endswith("_pts") else t[k] - zero for k in t}
        if ctx.get("presentation_times"):
            original = load_times(ep_dir, ctx, recorded=True)
            ep["recorded_times"] = {k: original[k] if k.endswith("_pts") else original[k] - zero for k in original}
    for v, d in src.items():
        if d.get("kmap"):
            ep["kmap"][v] = np.load(ep_dir / d["kmap"])
    # each camera's depth stream, when the recording has one (label/depth.py)
    from label import depth as dp
    recorded_depth = dp.load(ep_dir)
    ep["depth"] = {v: e for v, e in recorded_depth.items() if v in src}
    ep['additional_depth'] = {v: e for v, e in recorded_depth.items() if v not in src}
    return ep


def request_duration(ep: dict) -> float:
    """Actual anchor duration, including the final frame interval and retained duration claim."""
    view = anchor(ep)
    count = int(ep['sources'][view]['n_frames'])
    if count <= 0:
        raise ValueError('recording has no anchor frames')
    times = (ep.get('times') or {}).get(view)
    if times is not None:
        times = np.asarray(times, dtype=float)
        differences = np.diff(times)
        if len(times) != count or not np.isfinite(times).all() or np.any(differences < 0):
            raise ValueError('recording presentation clock cannot establish the request duration')
        positive = differences[differences > 0]
        # Legacy capture clocks may tie without opting into a separate display clock.
        # Bound their final stamp interval without rewriting the saved label instants.
        interval = float(np.median(positive)) if len(positive) else count / ep_fps(ep)
        if 'last_frame_interval_s' in ep['sources'][view]:
            interval = number(ep['sources'][view]['last_frame_interval_s'])
            if interval is None or interval <= 0:
                raise ValueError('recording has no usable final frame presentation interval')
        actual = max(0.0, float(times[-1]) + interval)
    else:
        actual = count / ep_fps(ep)
    stated = number(ep['context'].get('duration_s'))
    return max(actual, stated or 0.0)


def require_request_window(ep_dir: Path, ctx: dict) -> None:
    """Check source window metadata without decoding images or loading sensor arrays."""
    from prepare.formats import clock_context
    from prepare.camera_clock import load_times
    ctx = clock_context(ctx)
    source_path = Path(ep_dir) / 'sources.json'
    if not source_path.is_file():
        return  # normal request construction reports missing source files
    sources = json.loads(source_path.read_text())
    ordered = order_views(sources)
    if not ordered or any(not isinstance(sources[view], dict) or 'n_frames' not in sources[view] for view in ordered):
        return  # normal loading reports unsupported or incomplete source entries
    times = load_times(ep_dir, ctx) if ctx.get('real_times') else None
    if times is not None:
        zero = float(ctx.get('clock_zero_s') or 0.0)
        times = {key: value if key.endswith('_pts') else value - zero for key, value in times.items()}
    ep = {'dir': Path(ep_dir), 'context': ctx, 'sources': sources, 'times': times}
    policy.require_request_window(ctx, request_duration(ep))


def frame_rate(ep: dict) -> tuple:
    """(the anchor's frames per second, a note or None), from prepare/frame_rate.py episode_fps: the stated fps, or the
    frames' own rate when the anchor's capture times contradict it, which turns every window and lag in seconds into
    frames. plan names the note in the checks."""
    from prepare.frame_rate import episode_fps
    shown = views(ep) or list(ep.get("sources") or {})    # a lone camera of its own (an image ep) is its own anchor
    own = (ep.get("recorded_times") or ep.get("times") or {}).get(shown[0]) if shown else None
    return episode_fps(ep["context"], times=own, default=FPS)


def ep_fps(ep: dict) -> float:
    """The anchor's frames per second (frame_rate)."""
    return float(frame_rate(ep)[0])


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
    except (TypeError, ValueError, OverflowError):      # an integer too large for a float is no time either
        return None
    return v + 0.0 if math.isfinite(v) else None


def _seconds_number(t: float, places: int) -> str:
    """Round displayed seconds once, removing the sign when the rounded value is zero."""
    return f"{round(float(t), places) + 0.0:.{places}f}"


def tenths(t: float) -> str:
    """A time as the prompt gives a dataset's label time, to the tenth of a second, never as -0.0s: a time that rounds
    to zero is zero, not a time before the episode."""
    return _seconds_number(t, 1) + "s"


def order_views(keys) -> list[str]:
    """View keys in row order: the scene camera, the left and right mounted ones, then the extra cameras by number."""
    keys = list(keys)
    extra = sorted((k for k in keys if EXTRA_VIEW.fullmatch(str(k))), key=lambda k: int(EXTRA_VIEW.fullmatch(k)[1]))
    return [v for v in VIEW_ORDER if v in keys] + extra


def views(ep: dict) -> list[str]:
    """The shown camera slots in stable presentation order, without inferring physical placement."""
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


def frame_window(ep: dict, n: int | None = None) -> list[float]:
    """Actual anchor-frame window, including the final interval, without retained duration claims."""
    view = anchor(ep)
    sources = ep.get('sources') or {}
    source = sources.get(view) or {}
    if n is None:
        n = int(source.get('n_frames', len(ep.get('state', []))))
        n = min([n, *(len(mapping) for name, mapping in (ep.get('kmap') or {}).items()
                      if name != view and name in sources)])
    if n <= 0:
        return [0., 0.]
    times = (ep.get('times') or {}).get(view)
    if times is None:
        return [0., n / ep_fps(ep)]
    times = np.asarray(times, dtype=float)
    start = max(0., float(times[0]))
    if n < len(times):
        return [start, float(times[n])]
    steps = np.diff(times[:n])
    interval = float(np.median(steps[steps > 0])) if np.any(steps > 0) else 1 / ep_fps(ep)
    interval = float(source.get('last_frame_interval_s', interval))
    return [start, float(times[n - 1]) + interval]


def camera_sample(ep: dict, view: str, k: int, *, source_frame: int | None = None) -> dict | None:
    """One displayed RGB cell's source capture, separately from its anchor placement."""
    mapping = (ep.get('kmap') or {}).get(view)
    if k < 0 or (source_frame is None and mapping is not None and k >= len(mapping)):
        return None
    if source_frame is None:
        source_frame = int(mapping[k]) if mapping is not None else int(k)
    clocks = ep.get('times')
    recorded = (ep.get('recorded_times') or {}).get(view)
    own = (clocks or {}).get(view) if recorded is None else recorded
    count = (ep.get('sources') or {}).get(view, {}).get('n_frames')
    if source_frame < 0 or (count is not None and source_frame >= int(count)):
        return None
    capture = None
    if own is not None and source_frame < len(own):
        value = float(own[source_frame])
        capture = value if math.isfinite(value) else None
    elif clocks is None:
        capture = source_frame / ep_fps(ep)
    display = frame_time(ep, k)
    notes = ep['context'].get('camera_clock') or {}
    sample = {'row': int(k), 'source_frame': source_frame, 'display_time_s': display,
              'source_time_s': capture, 'capture_offset_s': capture - display if capture is not None else None,
              'episode_alignment_verified': capture is not None and not (notes.get(view) or notes.get(anchor(ep)))}
    pts = (clocks or {}).get(view + '_pts')
    if pts is not None and source_frame < len(pts):
        sample['source_pts'] = int(pts[source_frame])
    return sample


def seconds(t: float) -> str:
    """A time as the prompt says it, to the hundredth of a second. A recorder's clock can put a frame a hair before
    the episode's zero, and that instant is said as 0.00 s, never as -0.00 s, which reads as a time before the
    episode."""
    return _seconds_number(t, 2) + " s"


def describe_spans(ep: dict, spans) -> list[dict]:
    last = len(ep["state"]) - 1
    return [{"start_s": round(frame_time(ep, a), 2), "end_s": round(frame_time(ep, min(b, last)), 2)} for a, b in spans]


# Rigs this harness knows. The episode's context.json declares one as "profile"; nothing is assumed from the
# dataset name or the camera set.
RIGS = ("teleop_arms", "handheld_gripper", "ego_head")
STATE_KINDS = ("joints", "ee_pose", "none")


# One sampled instant every N s (label/policy.py PERIODS, per rig) for the whole episode, still spans included (a
# still span is exactly where a stopped recording would hide). Teleop arms move slowly and are seen by up to three
# cameras, so one instant every 1.5 s is enough. Handheld demonstrations are short and fast, so denser. Egocentric
# footage has one low-resolution head camera and the hands are the whole point, so it is sampled twice as densely
# again.
def every_s(ep: dict) -> float:
    return policy.effective(ep['context'])['sample_every_s']


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
    # the episode is the anchor camera's frames, which the state's rows follow. A camera paired to the anchor by real
    # time (kmap) is matched through its map, which must cover every anchor frame; any other camera only shows its own
    # frames, and one that ends before the anchor has no frame past its end (_decode_view), so it never decides how
    # long the episode is or whether the state lines up
    a = anchor(ep)
    short_maps = [len(km) for v, km in ep["kmap"].items() if v != a and v in windows and len(km) < windows[a]]
    # state_unaligned: the camera the state was recorded on was taken out of the episode, and nothing places the state
    # on the cameras left (board/clips.py drop_cameras), so it is never treated as aligned
    sa, sb = state_span(ep)
    checks = {"state_frames": T, "camera_frames": windows,
              "camera_windows_match_state": not ep["context"].get("state_unaligned")
              and windows[a] == T and not short_maps}
    if frame_rate(ep)[1]:
        checks["frame_rate"] = frame_rate(ep)[1]
    from prepare.state_contract import action_semantics
    if (ep.get("action") is not None and action_semantics(ep['context'])['role'] == 'command'
            and r == "teleop_arms" and kind == "joints" and ep["state"].shape[1] == 14):
        # sped-up recording (the rig's loop ran below the rate its samples are stamped at): a report
        # field computed from the leader/follower joint lag, not a claim made to the model. It reads the 12 arm
        # joints of two arms (timebase.JOINTS), as measure_folder does, so a one-arm recording is not measured
        checks["timebase"] = timebase.timebase_check(ep["state"][sa:sb], ep["action"][sa:sb],
                                               ep["context"].get("timebase_neighbour_lag_frames"), ep_fps(ep),
                                               context=ep['context'])
    if kind != "none" and checks["camera_windows_match_state"]:
        spans = [(x + sa, y + sa) for x, y in ms.still_spans(ep["state"][sa:sb], fps=fps, kind=kind,
                                                            grip_range=ms.gripper_full_range(ep["context"]))]
        n = T
    else:
        # video only, or a dataset defect (the anchor's frames are not the state's rows): label the video as shipped
        # over the anchor's frames that every camera paired to it by time has a map for, and make no state claims (the
        # defect is reported in checks)
        spans = []
        n = min([windows[a]] + short_maps)
    if ep["context"].get("stream_checks"):
        checks["streams"] = ep["context"]["stream_checks"].get("streams")
    if ep["context"].get("stream_pairing"):
        # checks/stream_pairing.py: whether each mounted stream follows its own actor's recorded motion (a
        # report field, never a claim made to the model)
        checks["stream_pairing"] = ep["context"]["stream_pairing"]
    every = every_s(ep)
    sample_spans, quiet = spans, None
    if not (kind != "none" and checks["camera_windows_match_state"]) and ep.get("signals"):
        # no arm state to find still spans in: the instant every moving signal falls quiet, and the instant it moves
        # again, are sent as an arm's still span would give them (label/signals.py quiet_spans), never as a claim
        from label import signals as sg
        quiet = sg.quiet_spans({k: np.asarray(v)[:n] for k, v in ep["signals"].items()},
                               int(round(ms.MIN_STILL_S * fps)))
        sample_spans = quiet
    presentation = (ep.get('times') or {}).get(a)
    ks = ms.sample_frames(n, sample_spans, fps=fps, moving_every_s=every, still_every_s=every,
                          times_s=np.asarray(presentation)[:n] if presentation is not None else None)
    zero = ep["context"].get("clock_zero_s")
    if zero is not None and ep.get("times") is not None:
        # a camera that started before the episode's clock (kept for an episode labelled already, board/clips.py
        # reanchor) is main: nothing before the clock's start is sampled, and its first frame at the start is
        before = [k for k in range(n) if frame_time(ep, k) < -0.5 / fps]
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
        mapped = actor_views(ep)
        vs = [mapped[g] for g in who if g < len(mapped) and mapped[g] is not None]
        out.append((k, (["exo"] if "exo" in views(ep) else []) + (vs or mounted)))
    return out


def actors(ep: dict) -> list[str]:
    """Unique names of the native seven value groups in state order, using each group's recorded claims.
    Only an old context without native group metadata retains the legacy left then right convention.
    On a person (ego), the actors are their own two hands."""
    if rig(ep) == "ego_head":
        return ["left", "right"]
    if state_kind(ep) == "none":
        # video only: the actors are the mounted cameras' own, or both when no single mounted camera names one
        mounted = [v for v in views(ep) if v in MOUNTED]
        return [mounted[0]] if len(mounted) == 1 else ["left", "right"]
    ctx = ep["context"]
    count = ep["state"].shape[1] // 7
    identities = ctx.get("state_identities")
    noun = "gripper" if rig(ep) == "handheld_gripper" else "arm"
    if isinstance(identities, list) and len(identities) == count and all(isinstance(i, dict) for i in identities):
        if count > 1 or any(i.get("status") != "absent" for i in identities):
            from prepare.formats import state_contract_actors
            contract = state_contract_actors(ctx, count)
            sides = [i.get("side") if i.get("status") == "known" else
                     contract[g] if contract and i.get("status") == "absent" else None
                     for g, i in enumerate(identities)]
            return [side if side and sides.count(side) == 1 else
                    f"{side} (recorded group {g + 1})" if side else
                    f"recorded {noun}" + (f" {g + 1}" if count > 1 else "") + " (side unknown)"
                    for g, side in enumerate(sides)]
    if (ctx.get("state_identity") or {}).get("status") == "conflict":
        return [f"recorded {noun} (side unknown)"]
    recorded = ep["context"].get("state_actors")
    if isinstance(recorded, list) and len(recorded) == count \
            and all(isinstance(name, str) and name for name in recorded) and len(set(recorded)) == count:
        return recorded
    if count == 2:
        return ["left", "right"]
    from prepare.formats import recorded_state_side
    source = ctx.get("source") or {}
    side = ctx.get("state_side") if "state_side" in ctx else recorded_state_side(
        source.get("state") if source.get("format") == "hdf5" else None)
    if side in MOUNTED:
        return [side]
    # A unique mounted view names the one actor. Two mounted views establish no side for a single state stream.
    mounted = [v for v in views(ep) if v in MOUNTED]
    if len(mounted) == 1:
        return [cam_name(ep, mounted[0])]
    return [f"recorded {noun} (side unknown)"]


def actor_views(ep: dict) -> list[str | None]:
    """Mounted view in recorded group order, with no wrist assigned to unknown or duplicate side claims."""
    names = actors(ep)
    vs = views(ep)
    if len(names) == 2:
        return [name if name in MOUNTED and name in vs else None for name in names]
    if state_kind(ep) != "none" and (ep["context"].get("state_identity") or {}).get("status") == "conflict":
        return [None]
    mounted = [v for v in vs if v in MOUNTED]
    name = names[0]
    if name in MOUNTED:
        return [name if name in mounted else None]
    return [mounted[0] if len(mounted) == 1 and name == cam_name(ep, mounted[0]) else None]


def _decode_error(e: Exception) -> bool:
    """Whether e is the decoder failing on a file that is there (PyAV's errors, the frame reader's own): a fault of the
    recording, which costs only the frames it hits. A file that is missing or cannot be opened (OSError, including
    PyAV's FileNotFoundError and PermissionError) is a fault on our side and is never one of these."""
    import av
    return isinstance(e, (av.error.FFmpegError, mf.FrameError)) and not isinstance(e, OSError)


def _decode_view(ep: dict, v: str, ks: list[int], gate=None, widths=None, detail_ks=(), failed: set | None = None,
                 damaged: set | None = None):
    """Frames for anchor indices ks. A camera paired to the anchor by real time (kmap) is decoded at
    its own frame nearest each anchor frame; results are keyed by the anchor index. With widths, a frame wider
    than the widest of them is kept full size only at detail_ks, and otherwise only at those widths.

    A camera never fails its episode over its own data. It may have fewer frames than the anchor (a camera that stopped
    first), or its file may end before its own last frame (an upload's every camera one frame short), and then the
    instants after its last frame have no frame. Its file may not decode, or be damaged partway, and then each instant
    is decoded on its own, so only the instants it cannot decode lose its frame; those go into failed, when given, and
    those whose frame the decoder marks as damaged (label/frames.py DamagedFrame) into damaged too. A missing file
    still raises (_decode_error)."""
    s = ep["sources"][v]
    km = ep["kmap"].get(v)
    own = [int(km[k]) for k in ks] if km is not None else list(ks)
    mine = [j for j in own if j < int(s["n_frames"])]      # the instants past the camera's last frame have none
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
        got = run(mine)
    except Exception as e:
        if not _decode_error(e):
            raise
        got, bad, hurt = {}, set(), set()
        for j in sorted(set(mine)):
            try:
                got.update(run([j]))
            except Exception as e1:
                if not _decode_error(e1):
                    raise
                bad.add(j)
                if isinstance(e1, mf.DamagedFrame):
                    hurt.add(j)
        if failed is not None:
            failed.update(k for k, j in zip(ks, own) if j in bad)
        if damaged is not None:
            damaged.update(k for k, j in zip(ks, own) if j in hurt)
    held = placeholder_instants(ep, ks).get(v, set())
    if failed is not None:
        failed.update(held)
    if damaged is not None:
        damaged.update(held)
    return {k: got[j] for k, j in zip(ks, own) if j in got and k not in held}


class _CaptureSearch:
    """Nearest remaining capture by time, with earlier indices winning ties. Removed failures are skipped through
    compressed successor and predecessor links, so each missing sample cannot walk the same failed stretch again.
    Captures and equal-time group starts are indexed once; each unsuccessful decode removes its index once."""

    def __init__(self, captures):
        self.captures = np.asarray(captures)
        self.order = np.argsort(self.captures, kind="stable")
        self.times = self.captures[self.order]
        self.starts = np.searchsorted(self.times, self.times)
        self.positions = np.empty(len(self.order), dtype=int)
        self.positions[self.order] = np.arange(len(self.order))
        self.after = list(range(len(self.order) + 1))
        self.before = list(range(len(self.order) + 1))

    @staticmethod
    def _find(links, i):
        while links[i] != i:
            links[i] = links[links[i]]
            i = links[i]
        return i

    def discard(self, k):
        i = int(self.positions[k])
        self.after[i] = self._find(self.after, i + 1)
        self.before[i + 1] = self._find(self.before, i)

    def nearest(self, k):
        target = self.captures[k]
        bound = int(np.searchsorted(self.times, target))
        right = self._find(self.after, bound)
        left = self._find(self.before, bound) - 1
        # A predecessor points to the last surviving index of its time group. Choose the group's first survivor,
        # since stable sorting makes it the earliest original index, including ties across two distinct times.
        left = self._find(self.after, int(self.starts[left])) if left >= 0 else -1
        candidates = [i for i in (left, right) if 0 <= i < len(self.order)]
        if not candidates:
            return None
        i = min(candidates, key=lambda i: (abs(self.times[i] - target), int(self.order[i])))
        return int(self.order[i])


def frames(ep: dict, pl: dict, gate=None, widths=None, detail_ks=()) -> dict:
    """{view: {k: PIL image}} for every planned k a camera has a frame at. With widths (the cell widths a request can
    be built at), frames outside detail_ks are kept only at those widths (label/frames.py Shrunk).

    An instant no camera has a frame at is named and replaced by a nearby readable frame at its own time. When the
    episode's last instants are past every
    camera's last frame (an upload whose every camera's file ends a frame before the episode does), the last frame any
    camera has takes their place, so the last detail view is the end of the footage (ep["footage_end"]). The episode
    keeps what it found for the prompt and the request: ep["no_frame"], the instants each camera has no frame at,
    which recording_at then reports as not recording, so every grid and view leaves it out there; ep["decode_failed"],
    the instants inside a camera's own recording (_in_span) its file could not be decoded at, before its last frame
    or, for a file none of whose frames decodes, all of them (_coverage_note, decode_failures). A frame the decoder
    marks as damaged and a placeholder frame (placeholder_instants) did not decode, wherever they are in the file, so
    the model is never shown either as footage. Raises only when no camera has any frame."""
    vs = views(ep)
    failed, damaged = {v: set() for v in vs}, {v: set() for v in vs}
    with ThreadPoolExecutor(max_workers=len(vs)) as ex:
        futs = {v: ex.submit(_decode_view, ep, v, pl["ks"], gate, widths, detail_ks, failed[v], damaged[v])
                for v in vs}
        got = {v: f.result() for v, f in futs.items()}
    for v, hit in placeholder_instants(ep, pl["ks"]).items():
        if v in got:
            for k in hit:
                got[v].pop(k, None)
            failed[v] |= hit
            damaged[v] |= hit
    ks = sorted(set(pl["ks"]))
    # an instant some camera can show: decoded there and inside its own recording (a camera paired by time that was
    # not recording has only its nearest frame, from another time, which is never shown)
    keep = [k for k in ks if any(k in got[v] and _in_span(ep, v, k) for v in vs)]
    unavailable = [k for k in ks if k not in keep]
    ep["unavailable_instants"] = unavailable
    ep["fallback_instants"] = {}

    def usable(k):
        more = {v: _decode_view(ep, v, [k], gate, widths, detail_ks, failed[v], damaged[v])
                if _in_span(ep, v, k) else {} for v in vs}
        for v, hit in placeholder_instants(ep, [k]).items():
            more.get(v, {}).pop(k, None)
            failed[v] |= hit
            damaged[v] |= hit
        for v in vs:
            got[v].update(more[v])
        return any(k in more[v] and _in_span(ep, v, k) for v in vs)

    # The sampling schedule can land entirely on damaged rows. Search the episode's real frame indices before
    # concluding that it has no footage. Keep unknown and usable captures in a nearest-time index; every known
    # failure leaves it once, so later missing targets skip whole damaged stretches.
    if unavailable:
        search = _CaptureSearch(np.fromiter((frame_time(ep, k) for k in range(pl["n"])), dtype=float))
        for k in unavailable:
            search.discard(k)
    kept = set(keep)
    for missing in unavailable:
        while (k := search.nearest(missing)) is not None:
            if k in kept or usable(k):
                kept.add(k)
                ep["fallback_instants"][missing] = k
                break
            search.discard(k)
        if not kept:
            break
    keep = sorted(kept)
    if not keep:
        raise mf.FrameError(f"{ep.get('dir', '?')}: no camera has a decodable frame")
    ep.pop("footage_end", None)
    if unavailable:
        past = [k for k in ks if k > keep[-1] and not any(k in damaged[v] for v in vs)]
        if past:
            # the frames before the first instant past every camera's end, back to the last instant kept (decoded
            # again at full size, for the detail view): the latest any camera has takes the end's place
            for k in range(min(past) - 1, keep[-1] - 1, -1):
                more = {v: _decode_view(ep, v, [k], gate) for v in vs}
                for v in placeholder_instants(ep, [k]):
                    more.get(v, {}).pop(k, None)
                if any(k in more[v] and _in_span(ep, v, k) for v in vs):
                    for v in vs:
                        got[v].update(more[v])
                    keep = sorted(set(keep) | {k})
                    ep["footage_end"] = k
                    break
        pl["ks"] = keep
        if pl.get("contact"):
            pl["contact"] = [k for k in pl["contact"] if k in keep]
        # Replacement grid cells stay small; only the new first and last detail views need their native pixels.
        if widths:
            for k in (keep[0], keep[-1]):
                if k not in detail_ks:
                    for v in vs:
                        if k in got[v]:
                            got[v].update(_decode_view(ep, v, [k], gate))
    ep["no_frame"] = {v: {k for k in keep if k not in got[v]} for v in vs if any(k not in got[v] for k in keep)}
    # a damaged stretch is an instant the camera could not decode before its last frame (the instants after it are
    # where its file ended), or a damaged or placeholder frame anywhere; a camera with no frame at all that failed to
    # decode does not decode anywhere. Only an instant inside the camera's own recording counts: a camera paired by
    # time that was not recording there has only its nearest frame, from another time, so the instant is where its
    # video starts or ends (_coverage_note), even when that frame does not decode. It is recorded whether or not
    # another camera shows the instant, so an instant that left the request is still flagged
    bad = {v: sorted(k for k in failed[v] if _in_span(ep, v, k) and (not got[v] or k < max(got[v]) or k in damaged[v]))
           for v in vs}
    ep["decode_failed"] = {v: ks_ for v, ks_ in bad.items() if ks_}
    ep["undecodable"] = {v for v in ep["decode_failed"] if not got[v]}
    return got


def frame_runs(ks) -> list[list[int]]:
    """Frame indices as runs [[first, last], ...] of consecutive ones, how context.json keeps a set of frames short
    (placeholder_frames)."""
    ks = np.unique(np.asarray(list(ks), dtype=np.int64))
    if not len(ks):
        return []
    cuts = np.flatnonzero(np.diff(ks) > 1) + 1
    return [[int(r[0]), int(r[-1])] for r in np.split(ks, cuts)]


def runs_within(runs, a: int, b: int) -> list[list[int]]:
    """Runs of frames (frame_runs) cut to the frames [a, b) and counted from a, for an episode cut to those frames (a
    part of a long recording, label/pieces.py; an episode trimmed to the minutes cap, prepare/formats.py)."""
    out = []
    for x, y in runs:
        x, y = max(int(x), a), min(int(y), b - 1)
        if x <= y:
            out.append([x - a, y - a])
    return out


def placeholder_instants(ep: dict, ks) -> dict:
    """{view: the instants of ks at which the camera's frame is a placeholder}: a black frame the reader wrote where an
    image did not decode, so a camera of images one per table row keeps the rows on their frames (context.json
    placeholder_frames, runs of anchor frames, prepare/formats.py placeholder_frames). It is no footage, and frames
    treats it as a frame that did not decode."""
    out = {}
    for v, runs in (ep["context"].get("placeholder_frames") or {}).items():
        hit = {int(k) for k in ks if any(x <= int(k) <= y for x, y in runs)}
        if hit:
            out[v] = hit
    return out


def camera_label(view: str, ctx: dict) -> str:
    """A camera as the board names it: the main (or head) camera, the left or right wrist (or gripper), and any other
    camera by the dataset's own name for it."""
    rig = ctx.get("profile")
    if view == "exo":
        return "head camera" if rig == "ego_head" else "main camera"
    if view in ("left", "right"):
        return f"{view} gripper camera" if rig == "handheld_gripper" else f"{view} wrist camera"
    cam = (ctx.get("cameras") or {}).get(view) or {}
    return str(cam.get("name") or cam.get("key") or "other") + " camera"


def decode_failures(ep: dict) -> list[dict]:
    """The stretches each camera's file could not be decoded at (frames), for the run's record and the board
    (board/build.py reader_issues): [{"camera", "t0_s", "t1_s", "what"}], on the recording's clock (a part of a long
    recording adds where it starts, label/pieces.py). Labelling never writes into the episode folder."""
    off = float((ep["context"].get("piece") or {}).get("t0_s") or 0.0)
    out = []
    for v, ks in (ep.get("decode_failed") or {}).items():
        # a placeholder is flagged by the reader already, over its whole stretch (frames_not_decodable)
        held = placeholder_instants(ep, ks).get(v, set())
        ks = [k for k in ks if k not in held]
        if not ks:
            continue
        t0, t1 = round(frame_time(ep, min(ks)) + off, 3), round(frame_time(ep, max(ks)) + off, 3)
        name = camera_label(v, ep["context"])
        if v in (ep.get("undecodable") or ()):
            what = f"The {name} video could not be decoded, so the labels have no frame of it."
        else:
            when = f"at {seconds(t0)}" if t0 == t1 else f"from {seconds(t0)} to {seconds(t1)}"
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


def depth_label(ep: dict, view: str) -> str:
    """A camera pairing must be recorded; an anchor slot is only a display placement."""
    entry = (ep.get('depth') or {}).get(view) or (ep.get('additional_depth') or {}).get(view) or {}
    metadata = (ep['context'].get('depth') or {}).get(view) or {}
    paired = entry.get('paired_camera', metadata.get('paired_camera'))
    if paired is not None and paired in (ep['context'].get('cameras') or {}):
        return cam_name(ep, paired) + ' depth'
    source = str(entry.get('source_key') or metadata.get('source_key') or entry.get('source')
                 or metadata.get('source') or view)
    from label.dictionary_context import _public_source
    return _public_source(source) + ' (unpaired depth)'


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
        dr.text((6, y + 5), f"{depth_label(ep, v)}   {label}   t={frame_time(ep, k):.2f}s", fill=(255, 220, 0))
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
    by = ", " + contact_placement(c) if c.get("aligned_by") else ""
    event = 'touch' if c.get('evidence_kind') == 'recorded_contact_state' else 'signal activity'
    if begin:
        strips.append((f"{event} begins by the recording{by}",
                       [min(max(c["start_s"] + o, 0.0), t_end) for o in STRIP_OFFSETS_S]))
    if end:
        strips.append((f"{event} ends by the recording{by}",
                       [min(max(c["end_s"] + o, 0.0), t_end) for o in END_OFFSETS_S]))
    kp = _frame_at(ep, c["peak_s"])
    ks = sorted({_frame_at(ep, t) for _, ts in strips for t in ts} | {kp})
    try:
        got = {v: _decode_view(ep, v, ks if v in vs else [kp], gate) for v in dict.fromkeys(vs + pvs)}
    except Exception as e:
        # _decode_view already keeps the frames it can; a fault on our side (a missing file) or in our code raises
        if not _decode_error(e):
            raise
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
                peak.append((depth_label(ep, v), im.resize(size, Image.NEAREST)))
    for nm in c["signals"]:
        tile = _map_tile(ep, nm, kp)
        if tile is not None:
            peak.append((nm, tile))
    blocks.append(("largest recorded response", [("", peak)]))
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
    reference = sg.declared_zero_reference(m, a.shape[1])
    if reference is None:
        d, swing = sg.distance_at(a, min(k, len(a) - 1), m.get("rest"), m.get("swing"))
    else:
        level, sign = reference
        d = np.maximum(sign * (a[min(k, len(a) - 1)] - level), 0)
        swing = sg.swing_of(a, level)
    row = d.reshape(int(shape[0]), int(shape[1]))
    g = np.where(np.isfinite(row), np.clip(row / swing, 0, 1) if swing > 0 else 0.0, 0.0)
    cell = max(1, MAP_TILE_PX // max(int(shape[0]), int(shape[1])))
    return Image.fromarray((g * 255).astype(np.uint8), "L").resize((int(shape[1]) * cell, int(shape[0]) * cell),
                                                                    Image.NEAREST).convert("RGB")


def _rig_nouns(r: str) -> dict:
    if r == "ego_head":
        return {"actor": "hand", "an_actor": "a hand", "actors": "hands", "gripper_of": "hand",
                "who": "the person's hands are the acting parts in this annotation profile"}
    if r == "teleop_arms":
        return {"actor": "arm", "an_actor": "an arm", "actors": "arms", "gripper_of": "arm's gripper",
                "who": "robot arms are the acting parts; the profile alone does not establish the control method"}
    return {"actor": "gripper", "an_actor": "a gripper", "actors": "grippers", "gripper_of": "handheld gripper",
            "who": "grippers are the acting parts; the profile alone does not establish human holding or the control method"}


def _camera_line(ep: dict, v: str) -> str:
    """One camera's facts, and its lens when its frames show a circular image (label/lens.py)."""
    line = _camera_facts(ep, v)
    if ((ep.get("lens") or {}).get(v) or {}).get("circular"):
        line += f" It has {lens.DESC}."
    return line


def _camera_facts(ep: dict, v: str) -> str:
    """A recorded or verified role can guide geometry; a presentation slot cannot establish it."""
    from prepare.formats import camera_role_provenance, camera_mounting
    cam = (ep["context"].get("cameras") or {}).get(v) or {}
    provenance = camera_role_provenance(cam, v, rig(ep))
    mounting = camera_mounting(cam, v, rig(ep))
    if cam.get("desc") and provenance not in ('presentation', 'recorded_name'):
        return f"- {cam_name(ep, v)}: {cam['desc'].rstrip('.')}."
    if mounting == 'unspecified':
        return (f"- {cam_name(ep, v)}: recorded view"
                + (f" {cam['key']}" if cam.get('key') else '')
                + "; its presentation slot and profile do not establish camera mounting or viewpoint. "
                  "Read the scene or sensor content and placement from the pixels and explicit source claims.")
    if mounting in ('fixed', 'scene'):
        return f"- {cam_name(ep, v)}: explicitly described as a {mounting} camera; read its direction from the pixels."
    n = _rig_nouns(rig(ep))
    if mounting == 'head':
        return (f"- {cam_name(ep, v)}: explicitly described as head-mounted; read its wearer or platform "
                "and direction from the pixels.")
    side = "" if len(views(ep)) == 1 or v not in ("left", "right") else f"{v.upper()} "
    return (f"- {cam_name(ep, v)}: the camera mounted on the {side}{n['gripper_of']}. When its own gripper's fingers "
            "are in view, they sit in the same place in every frame, usually along the bottom edge, and change only by "
            "opening and closing; "
            "the rest of the image moves whenever the gripper moves.")


def camera_desc(ep: dict, recorded: bool = True) -> str:
    from prepare.formats import camera_role_provenance, camera_mounting
    vs, r = views(ep), rig(ep)
    n = _rig_nouns(r)
    cams = ep["context"].get("cameras") or {}
    provenance = {v: camera_role_provenance(cams.get(v, {}), v, r) for v in vs}
    mounting = {v: camera_mounting(cams.get(v, {}), v, r) for v in vs}
    unknown_mount = {v for v in vs if mounting[v] == 'unspecified'}
    mounted = [v for v in vs if mounting[v] in ('actor', 'gripper', 'wrist')]
    res = sorted({f"{c.get('width')}x{c.get('height')}" for c in cams.values() if c.get("width")})
    rec = f" (recording {', '.join(res)} at {ep_fps(ep):g} fps)" if res else ""
    count = "There is exactly 1 camera; every grid row and every image strip is labelled with its name" \
        if len(vs) == 1 else (f"There are exactly {len(vs)}; every grid row and every image strip is labelled "
                              "with one of these names")
    s = (f"Cameras in this episode{rec}. {count}:\n"
         + "\n".join(_camera_line(ep, v) for v in vs) + "\n")
    s += ("The camera labels are presentation aliases identifying the supplied views, not publisher assertions "
          "about mounting or actor identity. A contradiction of a presentation alias is not a recording fault. "
          "Report observable corruption or missing evidence independently of those aliases. ")
    if mounted and len(mounted) == len(vs):
        s += (f"No camera in this episode is off the {n['actor'] if len(mounted) == 1 else n['actors']}: the "
              f"whole scene is seen only through {'that camera' if len(mounted) == 1 else 'those cameras'}, so "
              "reconstruct the layout and where things end up from what it shows.\n")
    claims = [cam_name(ep, v) for v in vs if mounting[v] != 'unspecified'
              or (provenance[v] != 'presentation' and cams.get(v, {}).get('desc'))]
    if claims:
        s += ("Check explicit recorded or verified camera-role claims in the descriptions or structured facts for "
              + ', '.join(claims) + ". A description alone does not establish mounting: check only the mounting "
              "or image-type assertions it actually makes, not its presentation alias. If an explicit assertion "
              "contradicts the pixels, describe what the view actually is and record it as a data issue.")
    if r == "ego_head" and any(mounting[v] == 'head' for v in vs):
        s += ("The recorded head-camera mount claims for " + ', '.join(claims)
              + " are not guaranteed truth: check them against the pixels. "
              "If the stream contradicts it (a view that stays fixed instead of moving with the person's head, "
              "a black, frozen or corrupted stream), describe what the view actually is and record it as a data "
              "issue. Where the camera is worn and where it points are read from the frames: a head camera turns "
              "and tilts with the head, so it shows wherever the person looks, and a camera that is really on the "
              "chest or held in the hand, or a mount that slips and tilts the view partway through, is worth "
              "recording.")
    if r != "ego_head" and mounted:
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
    if unknown_mount:
        s += (" Camera labels identify views; they do not establish which arm owns a view. "
              "Determine camera placement and arm identity from the pixels and explicit recorded identities. "
              "Do not report a camera metadata mismatch merely because a side-named camera is fixed: "
              "its mounting was not declared.")
    identity_known = all(mounting[v] in ('actor', 'gripper', 'wrist') for v in vs if v in MOUNTED)
    if r == "ego_head":
        s += (" In the output, \"left\", \"right\" and \"both\" name the person's own left and right hands. "
              "Which hand is which follows the person's body (the forearm it belongs to, the thumb side), "
              "not which half of the image it is in, because hands cross the midline and reach across.")
    elif identity_known and "left" in vs and "right" in vs:
        s += (f" In the output, \"left\", \"right\" and \"both\" name the streams: an action is \"left\" when "
              "the left stream's own gripper makes the contact, \"right\" likewise, "
              f"\"both\" when the two act together. The other {n['actor']} often appears inside a view, and an "
              "object lying between open fingers is not yet held, so neither is a contact of that camera's own. "
              "This naming is bookkeeping only; it does not settle whether the names are right. Whether each "
              "stream really sits on the side its name says is a separate question for the pixels: where the "
              f"other {n['actor']} and the scene appear in it once you have worked out from the frame how that camera "
              "is turned at that instant" + (", and which recorded motion its view follows." if recorded else "."))
    elif (identity_known and state_kind(ep) != "none" and len(actors(ep)) == 1
          and len([v for v in vs if v in MOUNTED]) == 1 and actor_views(ep)[0] is not None):
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
    the instants before its video starts or after it ends (a camera paired by time that was not recording, _in_span,
    or a camera whose file ends first, frames), each said the same way for both, and the instants its file could not be
    decoded at (all of them for a file none of whose frames decodes). One camera can have more than one of these, and
    each is said; the sentence after them speaks of one camera or of several by how many cameras they name. When every
    camera's file ends before the episode does, the last instant is the last frame they have (frames,
    ep["footage_end"]), which is said too."""
    gaps, ended, broken, never = [], [], [], []
    at = lambda ks: ", ".join(seconds(frame_time(ep, k)) for k in sorted(ks))
    for v in views(ep):
        if all(recording_at(ep, v, k) for k in pl["ks"]):
            continue
        name = cam_name(ep, v)
        if not all(_in_span(ep, v, k) for k in pl["ks"]):
            t = ep["times"][v]
            gaps.append(f"{name} has frames only from {seconds(t[0])} to {seconds(t[-1])}")
        if v in (ep.get("undecodable") or ()):
            never.append((name, f"{name}'s video could not be decoded at any instant"))
            continue
        # the instants of the request it could not decode (one no camera could show has left the request)
        bad = set((ep.get("decode_failed") or {}).get(v) or ()) & set(pl["ks"])
        out_of = [k for k in pl["ks"] if not recording_at(ep, v, k) and k not in bad]
        t = (ep.get("times") or {}).get(v) if (ep.get("kmap") or {}).get(v) is not None else None
        before = [k for k in out_of if t is not None and frame_time(ep, k) < float(t[0])]
        after = [k for k in out_of if k not in before]
        after = [k for k in after if _in_span(ep, v, k)]
        if after:
            ended.append((name, f"{name}'s video ends before the episode does, so it has no frame at {at(after)}"))
        if bad:
            broken.append((name, f"{name}'s video could not be decoded at {at(bad)}"))
    out = ""
    if gaps:
        one = len(gaps) == 1
        s = "; ".join(gaps)
        out += (" " + s[0].upper() + s[1:] + f", so {'its' if one else 'their'} cells are empty at the instants "
                f"outside that time, and {'it is' if one else 'they are'} left out of a detail view there.")
    gone_tail = "{Its} cells at those times are empty, and {it} {is_} left out of a detail view there."
    never_tail = "{Its} cells are all empty, and {it} {is_} left out of every detail view."
    for parts, tail in ((ended + broken, gone_tail), (never, never_tail)):
        if not parts:
            continue
        one = len({name for name, _ in parts}) == 1
        words = {"its": "its" if one else "their", "Its": "Its" if one else "Their", "it": "it" if one else "they",
                 "is_": "is" if one else "are"}
        s = "; ".join(said for _, said in parts)
        out += " " + s[0].upper() + s[1:] + ". " + tail.format(**words)
    if ep.get("footage_end") is not None:
        out += (" Every camera's video ends before the episode does, so the last instant is the last frame they have, "
                f"at {seconds(frame_time(ep, ep['footage_end']))}.")
    if ep.get("unavailable_instants"):
        out += f" No camera could be decoded at the planned instants {at(ep['unavailable_instants'])}."
        replacements = ep.get("fallback_instants") or {}
        if replacements:
            out += " The available replacement frames are shown at their own times: " + "; ".join(
                f"{seconds(frame_time(ep, k))} for the unavailable instant {seconds(frame_time(ep, missing))}"
                for missing, k in replacements.items()) + "."
    return out


def _depth_note(ep: dict) -> str:
    """Which cameras also record depth, and that each detail view is followed by their depth: what the input is, never
    how to read it."""
    d = ep.get("depth") or {}
    if not d:
        return ""
    paired = [v for v in order_views(d) if d[v].get('paired_camera') is not None
              and d[v]['paired_camera'] in (ep['context'].get('cameras') or {})]
    unpaired = [v for v in order_views(d) if v not in paired]
    names = ", ".join(cam_name(ep, d[v]['paired_camera']) for v in paired)
    # a depth stream that stops before its camera has no reading at the instants past it (prepare/formats.py
    # depth_kmap -1), where no depth view follows
    gaps = [depth_label(ep, v) for v in order_views(d) if "km" in d[v] and (np.asarray(d[v]["km"]) < 0).any()]
    clocks = ep["context"].get("camera_clock") or {}
    depth_clocks = ep["context"].get("depth_camera_clock") or {}
    assumed = bool(clocks or depth_clocks)
    invalid = any(isinstance(note, dict) and note.get("clock_problem")
                  for notes in (clocks, depth_clocks) for note in notes.values())
    if invalid:
        # A malformed side clock does not qualify depth paired only to a valid anchor and its own valid clock.
        relevant = [clocks.get(anchor(ep)), *(clocks.get(v) for v in d), *depth_clocks.values()]
        assumed = any(relevant)
        invalid = any(isinstance(note, dict) and note.get("clock_problem") for note in relevant)
    qualification = "".join(f" {depth_label(ep, v)}. {note['what']}" for v, note in depth_clocks.items()
                            if v in d and isinstance(note, dict) and note.get("clock_problem"))
    precision = (" Depth is placed using assumed presentation times. "
                 + ("An unusable recorded clock does not establish exact capture instants." if invalid else
                    "Shared timestamps do not establish exact capture instants.") + qualification) if assumed else ""
    description = (f"{names} {'records' if len(paired) == 1 else 'record'} depth as well as colour. " if paired else '')
    if unpaired:
        description += (', '.join(depth_label(ep, v) for v in unpaired)
                        + (' has' if len(unpaired) == 1 else ' have')
                        + ' no recorded RGB camera pairing. Display slots do not establish common sensors '
                        'or image coordinates with the colour views. ')
    return ("\nDEPTH: " + description + 'Detail views are followed by available recorded depth at the displayed instants.'
            + precision
            + (f" {', '.join(gaps)} {'has' if len(gaps) == 1 else 'have'} no depth for part of the episode; a detail "
               "view there is followed by no depth." if gaps else ""))


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


def contact_placement(c: dict) -> str:
    """Name the particular assumption a contact inherits, preserving existing common start wording."""
    return placement_text(c.get("aligned_by"))


def _contact_timing(c: dict) -> str:
    by = c.get("aligned_by")
    if by == ALIGNED_ASSUMED:
        return CONTACT_ASSUMED
    if by == ALIGNED_CAMERA:
        return " (these times use the assumed camera presentation clock, not measured capture times)"
    return f" (these times are {placement_text(by)}, not measured contact times)" if by else ""


def _contact_line(c: dict) -> str:
    hand = f"{c['hand']} hand" if c.get("hand") else "hand not named by the recording"
    physical = c.get('evidence_kind') == 'recorded_contact_state'
    state = 'touching' if physical else 'active'
    when = (f"{c['start_s']:.2f} s" + (f" (already {state} at the first frame)" if c.get("from_start") else "")
            + f" to {c['end_s']:.2f} s" + (f" (still {state} at the last frame)" if c.get("to_end") else "")
            + f", largest response at {c['peak_s']:.2f} s"
            + _contact_timing(c))
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
            + (f"; at its largest response {'; '.join(where)}" if where else "")
            + (f"; it weakens and comes back at {', '.join(f'{x:.2f}' for x in c['dips_s'])} s"
               if c.get("dips_s") else "")
            + (f". {c['reference_note']}" if c.get('reference_note') else
               '. Response activity has no verified unloaded reference or physical-contact interpretation.' if not physical else ''))


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
    from label.dictionary_context import field_interpretation
    meta = ep.get("signal_meta") or {}
    out = set()
    for name, a in (ep.get("signals") or {}).items():
        m = meta.get(name) or {}
        role = field_interpretation(ep["context"], name).get("role")
        permission = sg.touch_cache_permission(name, role, m.get("touch_role"))
        if permission is False:
            continue
        from prepare.formats import names_touch
        if (sg.declared_zero_reference(m, a.shape[1]) is not None and
                (names_touch(name) or sg.signal_role(role) in sg.TOUCH_QUANTITY_ROLES)):
            out.add(name)
            continue
        # read as stored (a float32 skin is never copied whole as float64, label/signals.py CHUNK_VALUES)
        stored = "touch" in m and (permission is None or m.get("touch_role") == "touch")
        if m["touch"] if stored else sg.is_touch(name, a[:n], m.get("rest"), m.get("swing"), role=role):
            out.add(name)
    return frozenset(out)


def _touch(ep: dict, pl: dict) -> frozenset:
    """plan()["touch"], or for a plan made by hand without it, the verdicts made now."""
    if "touch" not in pl:
        return touch_verdicts(ep, pl["n"])
    from label import signals as sg
    from label.dictionary_context import field_interpretation
    return frozenset(name for name in pl["touch"] if sg.touch_cache_permission(
        name, field_interpretation(ep["context"], name).get("role"),
        (ep.get("signal_meta") or {}).get(name, {}).get("touch_role")) is not False)


def touch_contacts(ep: dict, pl: dict, contacts) -> list[dict]:
    """The contacts timed by at least one touch signal of the episode (touch_verdicts). A context.json prepared before
    is_touch can hold a contact found from a signal that only behaved like touch (an intervention flag, odometry); it
    is not shown."""
    touch = _touch(ep, pl)
    return [c for c in contacts or [] if any(nm in touch for nm in c.get("signals") or [])]


def _contact_coverage(ep: dict) -> dict:
    from label.contacts import coverage
    from label.dictionary_context import field_interpretation
    return coverage({name: {**entry, 'dictionary': field_interpretation(ep['context'], name)}
                     for name, entry in (ep.get('signal_meta') or {}).items()})


def contacts_block(ep: dict, pl: dict) -> str:
    """Recorded contacts and instrumented-hand coverage, including quiet touch streams."""
    covered = _contact_coverage(ep)
    missing = ('  "contacts_missing": [{"t_s": <float>, "hand": "left" | "right", "object": "<name>"}], '
               'only a visible grasp or press on an instrumented hand that no supplied sensor-activity interval covers. '
               'An uninstrumented hand or an unknown hand cannot establish a missing tactile response. '
               'A discrepancy is a review candidate, not proof of a failed sensor or no physical contact; '
               'sensor placement, contact detection and timing can limit the comparison.\n')
    arrival = recorder_arrival(ep['context'])
    if arrival:
        missing = ('  "contacts_missing": [], because recorder-arrival placement does not establish when a '
                   'physical tactile response should coincide with a visible grasp.\n')
    instruments = ('\nCONTACTS: instrumented hands and their recorded touch signals: '
                   + '; '.join(f'{hand}: {", ".join(names)}' for hand, names in covered.items()) + '.\n') if covered else (
                   '\nCONTACTS: no instrumented hand is identified by the recorded touch signals. '
                   'Return an empty contacts_missing list.\n')
    shown = touch_contacts(ep, pl, ep.get("contacts_shown"))
    if not shown:
        return instruments + '  "contacts": [], no sensor-activity intervals were supplied.\n' + missing if covered else ""
    rest = [c for c in touch_contacts(ep, pl, ep.get("contacts")) if c["id"] not in {x["id"] for x in shown}]
    physical = all(c.get('evidence_kind') == 'recorded_contact_state' for c in shown)
    event = 'touch' if physical else 'signal activity'
    state = 'touching' if physical else 'active'
    depth = any(_contact_views(ep, c)[2] for c in shown)
    # each strip is described, and its frame asked for, only for the contacts whose picture has it (contact_strips)
    begin = [c["id"] for c in shown if contact_strips(c)[0]]
    end = [c["id"] for c in shown if contact_strips(c)[1]]
    lacking = lambda have, why: ("" if len(have) == len(shown) else
                                 f" (not for {_and_list([c['id'] for c in shown if c['id'] not in have])}, {why})")
    only = lambda have: "" if len(have) == len(shown) else f"for {_and_list(have)} only, "
    picture = (([f"five frames around the time {event} begins, numbered 1 to 5"
                 + lacking(begin, f"already {state} at the first frame")] if begin else [])
               + ([f"three{'' if begin else ' frames'} around the time {event} ends, numbered 1 to 3"
                   + lacking(end, f"still {state} at the last frame")] if end else []))
    claim = ("The following spans describe recorded touch-signal activity in recorder-arrival coordinates. "
             "Describe the pictured frames independently. Their visual touch and hand observations do not "
             "establish tactile event timing, a sensor fault or swapped sensors.\n" if arrival else
             "Recorded contact-state channels report touch in these spans; check their states against the frames:\n" if physical else
             "These spans describe sensor response activity, not confirmed physical contact. An inferred response reference "
             "is not a verified unloaded baseline. Even a declared calibration zero does not independently verify touch. "
             "Describe visible touch from the pictured frames independently; sensor activity boundaries do not establish touch onset or release.\n")
    return (instruments + claim + "\n".join(_contact_line(c) for c in shown) + "\n"
            + (("  More sensor-activity intervals are not shown: "
                + "; ".join(f"{c['id']} {c['start_s']:.2f}-{c['end_s']:.2f} s"
                            + (" " + contact_placement(c) if c.get("aligned_by") else "") for c in rest)
                + ".\n") if rest else "")
            + "After the detail views, each contact above has one picture: " + "".join(p + ", " for p in picture)
            + ("and " if picture else "") + "the moment of its largest recorded response"
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
            + missing)


SIGNAL_TABLE_MIN_CHARS = 12000
SIGNAL_TABLE_CHARS_PER_INSTANT = 400
SIGNAL_TABLE_MAX_CHARS = 96000


def signal_readout_budget(instants: int) -> int:
    """Keep the small readout allowance and grow with selected instants, up to a fixed character cap.
    Retained 28 to 134 instant recordings need more room than the old fixed 12000 characters; 400 per instant
    permits dozens of numeric rows without allowing a long recording to grow the prompt without limit.
    This counts characters, not provider tokens, and excludes descriptor and omission text.
    """
    return min(SIGNAL_TABLE_MAX_CHARS, max(SIGNAL_TABLE_MIN_CHARS,
                                         SIGNAL_TABLE_CHARS_PER_INSTANT * instants))


def _signals_table(ep: dict, pl: dict) -> str:
    """The recording's other per-frame numbers (ep["signals"], under the dataset's own names): every one listed once
    with its shape, its value names and the range its values take (label/signals.py describe); over each recorded
    still span how much each one changed (the claim they bear on: a mobile base can drive while the arms are still);
    and the values at every sampled instant (_signal_readout), except a touch signal's, whose timing is given once as
    the episode's contacts (contacts_block). They are shown, not interpreted: the model reads what each is from its
    name and the robot's description."""
    from label import signals as sg
    from label.dictionary_context import field_interpretation
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
        interpretation = field_interpretation(ep["context"], name)
        wording = _interpretation_words(interpretation)
        try:
            if not len(a):
                lines.append(f"  {name}: no rows, so no reading at any frame" + wording)
                continue
            if _constant(a):
                # a value repeated wherever it reads (a setting, a calibration, or a sensor that sent nothing new):
                # named once, as the same at every frame only when it reads at every frame
                complete = np.isfinite(a).all(axis=1)
                v = a[complete][0] if complete.any() else sg.finite_range(a)[0]
                said = name + (f" {sg.reading_text(v[0])}" if len(v) == 1 else
                               " [" + ", ".join(sg.reading_text(x) for x in v) + "]" if len(v) <= sg.PER_VALUE_MAX else "")
                said += wording
                gaps = sg.gap_words(a)
                if gaps:
                    wherever.append(f"{said} ({gaps})")
                else:
                    still.append(said)
                continue
            m = meta.get(name) or {}
            lines.append(sg.describe(name, a, m.get("shape"), m.get("names"), rate_hz=m.get("rate_hz"),
                                     fps=ep_fps(ep), aligned_by=m.get("aligned_by"),
                                     camera_aligned_by=m.get("camera_aligned_by"),
                                     clock_problem=m.get("clock_problem"), source_rows=m.get("source_rows"),
                                     camera_frames=m.get("camera_frames"), unnamed=m.get("unnamed"),
                                     names_source=m.get("names_source")) + wording)
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
                lo, hi = sg.finite_range(seg)
                c = float(np.nanmax(hi - lo)) if np.isfinite(seg).any() else 0.0
                if c > 0:
                    ch.append(f"{name} {sg.reading_text(c)}")
            lines.append(f"    {frame_time(ep, a0):.2f}-{frame_time(ep, min(b0, n - 1)):.2f}s: "
                         + ("; ".join(ch) if ch else "none changed"))
    lines += _readout_of(ep, pl)[0]
    interpretation_note = ("Dictionary meanings and roles are attributed interpretations, not recorded facts. "
                           if any(field_interpretation(ep["context"], name) for name in arrs) else
                           "They are not interpreted for you: read what each is from its name and the robot's description above. ")
    return ("\nOTHER RECORDED SIGNALS: every other number the dataset records per frame, under the dataset's own "
            "name, with the range each of its values takes over the episode (one that never changes is given as its "
            "value). " + interpretation_note + "Like the rest of the recording they are claims to check against the video; a camera carried by "
            "something they show moving (a mobile base, a torso) moves with it.\n" + "\n".join(lines))


def _signal_readout(ep: dict, pl: dict) -> tuple[list[str], frozenset]:
    """(lines, whole): the values of the signals at every sampled instant, and the signals every row of which is in
    them. A touch signal has no rows (its timing is given once as the episode's contacts, contacts_block, so the frames
    are read on their own first and the contacts are checked against them; touch is the rule the contacts shown follow,
    touch_verdicts), nor has a signal that never changes or has no reading. The rows are ranked by how much their
    values move over the episode (label/signals.py movements) and kept when they fit the allowance for the selected
    instants, then printed in the signals' own order. An oversized row does not prevent smaller rows being shown.
    A row whose readings at the instants do not resolve its value (label/signals.py resolved), or track a row already
    sent (_drop_tracking), is left out and named with why.
    Every signal left out, whole or in part, is named with its size and rate; original value names remain in metadata.
    """
    from label import signals as sg
    from label.dictionary_context import field_interpretation
    sig = ep.get("signals") or {}
    meta = ep.get("signal_meta") or {}
    n = pl["n"]
    arrs = {k: a[:n] for k, a in sig.items()}      # as stored; label/signals.py reads them in float64 pieces
    touch = _touch(ep, pl)
    ks = pl["ks"]
    rows = []             # (how much the row's values move, the signal's place, the row's place, signal, label, values)
    unread = {}           # signal: the error that kept its rows out, named below
    aliased = {}          # signal: [(label, readings at every frame) of rows the instants do not resolve]
    counts = {}           # signal: how many rows it has
    wcols = {}            # (signal's place, row's place): readings at every frame of a row sent as window means
    series = {}           # (signal's place, row's place): the value's readings at every frame, for rows of one value
    for i, (name, a) in enumerate(arrs.items()):
        try:
            if name in touch or not len(a) or not np.isfinite(a).any() or _constant(a):
                continue
            m = meta.get(name) or {}
            role = field_interpretation(ep["context"], name).get("role")
            got = sg.summary_rows(name, a, ks, m.get("shape"), m.get("names"), role=role)
            mv = sg.movements(a)
            by_value = sg.per_value(name, a.shape[1], m.get("shape"), m.get("names"), role=role)
            # a row of a value's own readings is sent only when those readings resolve it at this sampling
            # (label/signals.py resolved); rows that summarise a large array are not readings of one value
            if by_value:
                follow, stand = sg.resolution(a, ks)
                ok = sg.is_resolved(follow, stand)
            else:
                ok, stand = np.ones(len(got), dtype=bool), np.ones(len(got))
        except Exception as e:  # noqa: BLE001 - one signal whose rows cannot be made is named below, the rest are given
            unread[name] = type(e).__name__
            continue
        counts[name] = len(got)
        cols = sg.columns(a) if by_value else None
        wmeans, wkept, wnoise = sg.window_means(a, ks) if by_value and not ok.all() else (None, None, None)
        for j, (lb, v) in enumerate(got):
            if not ok[j] and wkept is not None and wkept[j] >= sg.WINDOW_SIGNAL * wnoise:
                # the readings do not stand for their instants, but the means over each instant's frames keep the
                # row's trend: those are sent, marked, and the row takes no part in tracking
                rows.append(((float(wkept[j]), float(mv[j])), i, j, name, f"{lb} (window mean)",
                             [sg.reading_text(m) if np.isfinite(m) else "-" for m in wmeans[:, j]]))
                wcols[i, j] = cols[:, j]
                continue
            if not ok[j]:
                aliased.setdefault(name, []).append((lb, cols[:, j] if by_value else None))
                continue
            if by_value:
                series[i, j] = cols[:, j]
            # ranked for the budget by how well each reading stands for its instant, then by movement: movement alone
            # ranks a gripper that opens and closes a few times (a few large steps) below a noisy effort, so a budget
            # cut the gripper first. A summary row's numbers are made at each instant, so they stand for it.
            rows.append(((round(float(stand[j]), 2), float(mv[j]) if by_value else float(np.median(mv))),
                         i, j, name, lb, v))
    # a signal whose rows could not be made is named, so a signal missing from the values is never unexplained
    not_read = ([f"  The values at each instant leave out {_and_list([f'{nm} ({why})' for nm, why in unread.items()])}"
                 f", because {'its' if len(unread) == 1 else 'their'} rows could not be read."] if unread else [])
    # rows left out because their readings at the instants do not resolve them are named, with why
    step = float(np.median(np.diff([frame_time(ep, k) for k in ks]))) if len(ks) > 1 else every_s(ep)
    unresolved = ([f"  The values at each instant leave out "
                   + _and_list([lb + _extremes_words(ep, ks, col) for lbs in aliased.values() for lb, col in lbs])
                   + f", because one instant every {_seconds_number(step, 2)} s does not resolve their readings: they "
                     "swing faster than that or only jitter by their noise, so a reading at one instant does not stand "
                     "for the time around it. Their ranges are given above, and when each reached its largest and "
                     "smallest reading."] if aliased else [])
    # a row whose readings track a row already sent is sent once (label/signals.py tracking)
    rows, tracked = _drop_tracking(ep, rows, series, ks)
    whole_copies = [(r, s) for r, s, a in tracked if not a]
    partly = [(r, s, a) for r, s, a in tracked if a]
    repeated = ([f"  The values at each instant leave out these rows, because each follows a row that is sent, as "
                 "fitted here (\"later\" means the sent row reaches the same value that much later), so each is sent "
                 "once; how closely the fit holds is its largest miss at any instant, as a share of the row's range:"]
                + [f"  {_relation_words(ep, ks, r, s, series)}{_extremes_words(ep, ks, series[r[1], r[2]])}"
                   for r, s in sorted(whole_copies, key=lambda p: (p[0][1], p[0][2]))] if whole_copies else [])
    # a row read only where it comes apart is charged to the same budget as the rows, ranked with them
    apart_head = ("  These rows follow a sent row as fitted here except where they come apart, so they are given "
                  "only there, with the time of each reading:")
    unresolved = unresolved + repeated
    win_note = ("  A row marked (window mean) gives, at each instant, the mean of the frames that instant stands for, "
                "because its single readings at the instants do not resolve it.")
    partial = [r for r, _, _ in partly]
    if not rows and not partial:
        return unresolved + not_read, frozenset()
    lines = []
    head = "    at: " + " ".join(f"{frame_time(ep, k):.2f}" for k in ks)
    text = {(r[1], r[2]): f"    {r[4]}: " + " ".join(r[5]) for r in rows}
    text.update({(r[1], r[2]): _apart_line(ep, ks, r, s, a, series) for r, s, a in partly})
    budget = signal_readout_budget(len(ks))
    # what is said about the rows left out (when their extremes fell, the relation a copy follows) is paid first
    room, chosen = budget - sum(len(x) + 1 for x in unresolved) - (len(win_note) + 1 if wcols else 0), set()
    # rows sent as window means come after every other row and share a room of their own within the budget
    wroom = int(sg.WINDOW_SHARE * budget)
    skipped, kept_after_skip = False, False
    is_partial = {(r[1], r[2]) for r in partial}
    # a sent row that other rows are read from (their relation names it) comes first: without it they say nothing
    partners = {(s[1], s[2]) for _, s, _ in tracked}
    for r in sorted(rows + partial, key=lambda r: ((r[1], r[2]) not in partners, (r[1], r[2]) in wcols, -r[0][0],
                                                   -r[0][1], r[1], r[2])):
        key = (r[1], r[2])
        # the times row and the heading of the lines read only where apart are paid once, by the first of each
        extra = (len(apart_head) if not any(k in is_partial for k in chosen) else 0) if key in is_partial else \
            (len(head) if not any(k not in is_partial for k in chosen) else 0)
        if key in wcols and len(text[key]) + extra > min(room, wroom):
            continue
        if len(text[key]) + extra > room:
            skipped = True
            continue
        kept_after_skip |= skipped
        chosen.add(key)
        room -= len(text[key]) + extra
        wroom -= len(text[key]) + extra if key in wcols else 0
    table = [r for r in rows if (r[1], r[2]) in chosen]
    if table:
        lines.append("  Each signal that changes, at every instant you receive (seconds in the first row; \"-\" is "
                     "no reading):")
        lines += [head] + [text[r[1], r[2]] for r in table]
    shown_apart = [text[r[1], r[2]] for r in partial if (r[1], r[2]) in chosen]
    if shown_apart:
        unresolved = unresolved + [apart_head] + shown_apart
    if any(k in wcols for k in chosen):
        unresolved = unresolved + [win_note]
    cut = [r for r in rows if (r[1], r[2]) in wcols and (r[1], r[2]) not in chosen]
    if cut:
        unresolved = unresolved + [
            "  The values at each instant leave out "
            + _and_list([r[4].removesuffix(" (window mean)") + _extremes_words(ep, ks, wcols[r[1], r[2]]) for r in cut])
            + f", because one instant every {_seconds_number(step, 2)} s does not resolve their readings and rows given "
              f"as window means take at most {int(sg.WINDOW_SHARE * budget)} of the {budget} character budget, kept by "
              "how much of their variance the means keep. Their ranges are given above, and when each reached its "
              "largest and smallest reading."]
    left = {}
    for r in rows + partial:
        if (r[1], r[2]) not in chosen and (r[1], r[2]) not in wcols:
            left.setdefault(r[3], []).append(r)
    if left:
        named = _and_list([_left_out(nm, arrs[nm], meta.get(nm) or {}, len(left[nm]), counts[nm])
                           for nm in left])
        reason = (", because these rank lowest and there is no more room." if chosen else
                  ", because not even one row fits.")
        if kept_after_skip or budget != SIGNAL_TABLE_MIN_CHARS:
            reason = (", because rows are ranked by how well each reading stands for its instant, then by movement, "
                      f"and kept when they fit the {budget} character budget."
                      if chosen else f", because no row fits the {budget} character budget after the times row.")
        lines.append("  The values at each instant leave out " + named + reason)
    # a signal any of whose rows is sent: its other rows are said above as copies, window means or unresolved
    return lines + unresolved + not_read, frozenset(r[3] for r in rows if r[3] not in left)


def _drop_tracking(ep: dict, rows: list, series: dict, ks: list[int]) -> tuple[list, list]:
    """(rows to send, [(row left out, the sent row it tracks)]). Rows of one value's readings are taken in the budget's
    rank order; one whose readings track a row already kept (label/signals.py tracking) is sent once. Of such a pair the
    row whose values come later is kept, read from the lag of the best fit: a robot's measured joints follow the
    command that moves them, so the measured side is kept whatever either is named. With no lag either way the
    higher ranked row is kept."""
    from label import signals as sg
    order = sorted(rows, key=lambda r: (-r[0][0], -r[0][1], r[1], r[2]))
    kept, tracked, fps = [], [], ep_fps(ep)
    for r in order:
        key = (r[1], r[2])
        if key not in series:
            kept.append(r)
            continue
        hit = None
        for q in kept:
            if (q[1], q[2]) not in series:
                continue
            if q[3] == r[3]:
                continue    # values of one signal (cells of one sensor, joints of one state) are distinct readings
            same, lag = sg.tracking(series[q[1], q[2]], series[key], ks, fps)
            if not same:
                continue
            # where the two come apart (a stall, a slip, a grasp the gripper did not reach) the row is still read
            apart = sg.apart_instants(series[q[1], q[2]], series[key], ks, lag, fps)
            hit = (q, lag, apart)
            break
        if hit is None:
            kept.append(r)
            continue
        q, lag, apart = hit
        # the row left out must follow its partner along a line outside the instants where they come apart; when the
        # readings there carry none (the partner never moves outside them), the pair is not a copy and both are sent
        left, partner = (q, r) if lag < 0 else (r, q)
        if _relation_fit(ep, ks, left, partner, series, apart) is None:
            kept.append(r)
            continue
        if lag < 0:
            # the new row's values come later: it is the measured side, so it takes the kept row's place
            kept[kept.index(q)] = r
            tracked = [(t, r if s is q else s, a) for t, s, a in tracked] + [(q, r, apart)]
        else:
            tracked.append((r, q, apart))
    sent = {id(r) for r in kept}
    return [r for r in rows if id(r) in sent], tracked


def _apart_line(ep: dict, ks: list[int], row: tuple, other: tuple, apart: list[int], series: dict) -> str:
    """One row that tracks another except at some instants: the relation, the stretches, and its readings there."""
    runs, start = [], apart[0]
    for a, b in zip(apart, apart[1:] + [None]):
        if b != a + 1:
            runs.append((start, a))
            start = b
    stretches = _and_list([f"{_seconds_number(frame_time(ep, ks[a]), 1)} to "
                           f"{_seconds_number(frame_time(ep, ks[b]), 1)} s" for a, b in runs])
    readings = ", ".join(f"{frame_time(ep, ks[i]):.2f} {row[5][i]}" for i in apart)
    return (f"  {_relation_words(ep, ks, row, other, series, apart)}{_extremes_words(ep, ks, series[row[1], row[2]])}"
            f", except {stretches}, where its readings are given: {readings}.")


def _relation_fit(ep: dict, ks: list[int], row: tuple, other: tuple, series: dict, apart=()):
    """(lag, the sent row's readings, the left out row's readings, (slope, intercept)) of the line the left out row
    follows at the instants outside its apart instants, the sent row read lag frames later, or None when those readings
    carry no line (label/signals.py line_through): then the row does not follow the sent row and is sent whole."""
    from label import signals as sg
    x, y = series[other[1], other[2]], series[row[1], row[2]]
    _, lag = sg.tracking(x, y, ks, ep_fps(ep))
    k = np.asarray(ks, dtype=int)
    keep = np.ones(len(k), dtype=bool)
    keep[list(apart)] = False
    xa, ya = x[np.clip(k + lag, 0, len(x) - 1)][keep], y[k][keep]
    line = sg.line_through(xa, ya)
    return None if line is None else (lag, xa, ya, line)


def _relation_words(ep: dict, ks: list[int], row: tuple, other: tuple, series: dict, apart=()) -> str:
    """"row ≈ a + b x other, 0.10 s later (within 2.1%)": the straight line through the sent row that the left out row
    follows at the instants (outside its apart instants), in the left out row's own units, the lag at which it holds best,
    and its largest miss at any of those instants as a share of the row's range."""
    from label import signals as sg
    fps = ep_fps(ep)
    lag, xa, ya, (b, a) = _relation_fit(ep, ks, row, other, series, apart)
    good = np.isfinite(xa) & np.isfinite(ya)
    span = float(np.ptp(ya[good])) or 1.0
    miss = float(np.max(np.abs(ya[good] - (a + b * xa[good])))) / span
    # the offset is said to the nearest 1% of the row's range (the fit is no closer than that), the scale to three
    # figures; an offset that rounds to nothing is left out
    step = 10.0 ** np.floor(np.log10(0.01 * span))
    a = float(np.round(a / step) * step)
    a = 0.0 if abs(a) < step / 2 else a
    when = ("" if lag == 0 else f", {_seconds_number(abs(lag) / fps, 2)} s {'later' if lag > 0 else 'earlier'}")
    scale = f"{sg.reading_text(abs(b))} x {other[4]}"
    fit = (f"{sg.reading_text(a)} {'-' if b < 0 else '+'} {scale}" if a else f"{'-' if b < 0 else ''}{scale}")
    return f"{row[4]} ≈ {fit}{when} (within {100 * miss:.1f}%)"


def _extremes_words(ep: dict, ks: list[int], col) -> str:
    """" (max 3.21 at 41.32 s, min -0.05 at 12.10 s)": when a row left out reached its largest and smallest reading,
    read at every frame from the first instant to the last, so a short pulse between or on an instant keeps its time."""
    from label import signals as sg
    if col is None:
        return ""
    lo, hi = int(ks[0]), int(ks[-1]) + 1
    v = np.asarray(col[lo:hi], dtype=np.float64)
    if not np.isfinite(v).any():
        return ""
    i, j = int(np.nanargmax(v)), int(np.nanargmin(v))
    return (f" (max {sg.reading_text(v[i])} at {_seconds_number(frame_time(ep, lo + i), 2)} s, "
            f"min {sg.reading_text(v[j])} at {_seconds_number(frame_time(ep, lo + j), 2)} s)")


def _readout_of(ep: dict, pl: dict) -> tuple[list[str], frozenset]:
    """The readout episode_text made once for this prompt (pl["readout"]), or, for a block text called on its own,
    made now. Only episode_text sets pl["readout"], and only on its own copy of the plan, never on plan()'s."""
    return pl["readout"] if "readout" in pl else _signal_readout(ep, pl)


def _constant(a: np.ndarray) -> bool:
    """Whether a signal has a reading and each of its values never changes over the episode wherever it reads: named
    once with its value, and given no rows at each instant (_signals_table says where it has no reading)."""
    from label import signals as sg
    lo, hi = sg.finite_range(a)
    return bool(np.isfinite(a).any() and (hi == lo).all())


def _left_out(name: str, a: np.ndarray, m: dict, n_left: int, n_rows: int) -> str:
    """One signal left out of the values at each instant: its name, its size, its rate when known, and how many of its
    rows were left out when some of them were shown."""
    from label import signals as sg
    shape = m.get("shape")
    size = (" x ".join(str(int(x)) for x in shape) + " values" if shape and len(shape) > 1
            else f"{a.shape[1]} value{'s' if a.shape[1] > 1 else ''}")
    rate = f", {sg.reading_text(m['rate_hz'])} Hz" if m.get("rate_hz") else ""
    part = f", {n_left} of its {n_rows} rows" if n_left < n_rows else ""
    return f"{name} ({size}{rate}{part})"


def _and_list(xs: list[str]) -> str:
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


BETWEEN_INSTANTS = (
    "\nBETWEEN INSTANTS: reconstruct, do not smooth. The following rule concerns reconstructing visible actions "
    "from sampled RGB frames, not separate sensor_findings supported by supplied readings. A brief event can fall between two "
    "instants. When the scene in the frames differs between consecutive instants (something held "
    "is released, moved or gone; a contact is made or broken), an event happened in that "
    "interval: place it there and say in its notes that it is inferred. Do not fill an unseen interval with the "
    "expected, competent version of the task; a change that does not fit smooth progress may "
    "be a slip, drop, knock or failed grasp, and should be weighed against the frames before "
    "and after, the other side, and where the object ends up. Where the evidence does not "
    "settle it, say so in a note instead of defaulting to the charitable "
    "reading, and never invent a visual action merely because it would be expected between consistent frames. "
    "Consistent RGB frames do not negate a change measured by a supplied sensor. An event you "
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
                "instants written start>end. " + ("Joint angles describe recorded configuration, not physical "
                "event times or gripper location" if recorder_arrival(ep['context']) else
                "Joint angles tell you whether and when an arm moved, not where its gripper is"))
        clear = ("an arm that clearly moves in the video over an interval where every joint is recorded as "
                 "near 0 deg (the recording stopped updating), or the reverse; a single-frame joint step "
                 "of many degrees with no matching jump in the video")
    step_ms = 1000.0 / ep_fps(ep)
    grip = (ep["context"].get("gripper_value") or
            "the dataset's own number (units and direction not documented)").rstrip(".")
    if recorder_arrival(ep['context']):
        return ("\nRECORDED MOTION, from the dataset's state in recorder-arrival presentation coordinates. "
                + what + f"; the gripper/opening value is {grip}. "
                "The rows describe changes between recorded samples, including any interpolation used for display. "
                "Their times do not measure physical event times. Do not infer a physical state/video mismatch, "
                "stopped sensor, follower delay or synchronization fault by comparing these intervals with footage. "
                "Recorded motion and gripper changes remain evidence about the recorded channels. Read visible "
                "actions and held objects from the frames independently.\n" + "\n".join(rows))
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
        + (f"The recorded state covers only {seconds(frame_time(ep, sa))} to {seconds(frame_time(ep, sb - 1))} of "
           "the episode, so the rows stop there and nothing is recorded outside it.\n"
           if (sa, sb) != (0, len(st)) else "")
        + "\n".join(rows))


def annotation_lines(ctx: dict) -> list[str]:
    """Every dataset step with its declared time and success flag."""
    def when(x):            # a step with no end time is a moment, one with no time is listed without one
        t0, t1 = number(x.get("t0")), number(x.get("t1"))
        return ("no time" if t0 is None else tenths(t0) if t1 is None or t1 == t0
                else f"{tenths(t0)[:-1]}-{tenths(t1)}")
    def label(x):
        if "label" in x:
            return x["label"]
        if "text" in x:
            return x["text"]
        return "recorded annotation without a label or text: " + json.dumps(x, ensure_ascii=False, default=str)
    def window(x):
        piece = ctx.get('piece')
        w = x.get('request_window')
        start = number(piece.get('t0_s')) if isinstance(piece, dict) else None
        end = number(piece.get('t1_s')) if isinstance(piece, dict) else None
        if (start is None or end is None or end <= start
                or not isinstance(x.get('source_annotation'), dict) or not isinstance(w, dict)):
            return ''
        source = when({'t0': w.get('source_start_s'), 't1': w.get('source_end_s')})
        visible = when({'t0': w.get('visible_start_s'), 't1': w.get('visible_end_s')})
        edges = [text for key, text in (('continues_before', 'continues before this request'),
                                       ('continues_after', 'continues after this request')) if w.get(key)]
        return (f"\n    Original recording span: {source}. Visible request span: {visible}. "
                + (', '.join(edges) + '. ' if edges else '')
                + 'Only the visible portion can be checked; an internal request boundary does not change '
                  'the annotation deadline or establish that its goal failed.')
    lines = [f"  {when(x)}  {label(x)}" + ("" if x.get("ok", True) else "  (marked unsuccessful)") + window(x)
             for x in (ctx.get("annotation_subtasks") or []) if isinstance(x, dict)]
    return lines


def ego_annotation_block(ctx: dict) -> str:
    goal = (ctx.get("instruction") or "").strip()
    subs = ctx.get("annotation_subtasks") or []
    if not goal and not subs:
        return ("\nTHE DATASET'S ANNOTATION FOR THIS EPISODE: none; the dataset ships no task description for this "
                "clip. Infer the activities from the footage alone and leave goal_alignment out.\n")
    lines = annotation_lines(ctx)
    goal_label = "recording context goal" if policy.effective(ctx)["task_mode"] == "session" else "goal"
    return ("\nTHE DATASET'S ANNOTATION FOR THIS EPISODE (claims to check, see ABOUT THE DATASET'S ANNOTATION above):\n"
            + (f"  {goal_label}: \"{goal}\"\n" if goal else "")
            + ("  subtasks, with the times the dataset gives:\n" + "\n".join(lines) + "\n" if lines else "")
            + (f"  about these annotations: {ctx['annotation_note'].strip()}\n" if ctx.get("annotation_note") else ""))


# ---------------------------------------------------------------- the episode prompt: a base and its blocks

# Where a block's text goes in the episode prompt. The base fills the rest: the intro (who and what), the camera
# paragraph, the frames paragraph, the sampling line, BETWEEN_INSTANTS and the task.
PROMPT_SLOTS = ("intro", "frames_detail", "frames", "state", "signals", "after_frames", "after_task", "evidence")


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
    return bool(touch_contacts(ep, pl, ep.get("contacts_shown"))) or bool(_contact_coverage(ep))


def _no_state(ep: dict, pl: dict) -> bool:
    return state_kind(ep) == "none"


def _has_collection_note(ep: dict, pl: dict) -> bool:
    return bool(ep["context"].get("collection_note") or policy.effective(ep["context"])["dataset_note"])


def _has_contact_views(ep: dict, pl: dict) -> bool:
    return bool(pl.get("contact"))


def _has_coverage(ep: dict, pl: dict) -> bool:
    return bool(_coverage_note(ep, pl))


def _has_depth(ep: dict, pl: dict) -> bool:
    return bool(ep.get("depth"))


def _has_uploader_notes(ep: dict, pl: dict) -> bool:
    return bool(ep["context"].get("uploader_annotation"))


def _collection_text(ep: dict, pl: dict) -> str:
    source = (ep['context'].get('collection_note') or '').strip()
    supplied = policy.effective(ep['context'])['dataset_note'].strip()
    lines = [f'How the dataset cuts its recordings into episodes: {source}'] if source else []
    if supplied and supplied != source:
        lines.append(f'Dataset description supplied by the caller: {supplied}')
    return '\n'.join(lines) + '\n'


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
        if recorder_arrival(ep['context']):
            s = (f"\nRECORDED STILL SPANS, from the dataset's {src}: {sp}. The recorded channels remain within "
                 "the stillness threshold over these recorder-arrival intervals. This does not establish "
                 "physical stillness at the pictured instants; describe visual activity independently.")
        else:
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
    return s + _motion_table(ep, pl) + _dictionary_text(ep, pl)


def _state_unaligned_text(ep: dict, pl: dict) -> str:
    return ("\nRECORDED STATE: not given. This episode's cameras do not cover the same frames "
            "as its recorded state, so the state cannot be aligned to the video." + _dictionary_text(ep, pl))


# What in context["source"] says the reader found sensor data it did not read (prepare/formats.py write_signals,
# convert_hdf5, plan_video): the arrays and signals it left out, and "sensors", the sensor files whose signals it read
# (prepare/formats.py convert_video), so an episode with no signal from them has none read. With any of it, the
# episode is never told its dataset records no state.
UNREAD_SOURCE_KEYS = ("unused_signals", "unused_arrays", "sensors")


# Request wording for the reader's shared reasons (prepare.state_notes). The note names the particular channel or
# clock limitation; a leading sentence must hold even when other recorded channels use the footage clock.
STATE_WORDING = {
    LAYOUT: None,
    NOT_RECORDED: "as the recording holds none",
    UNREADABLE: "as a sensor file that may hold it could not be read",
    SHORT: "as it does not cover the footage",
    ASSUMED_CLOCK: "as a contributing state channel needs an assumed alignment with the cameras",
}


def _no_state_base_text(ep: dict, pl: dict) -> str:
    """No arm state. With other signals the line says why, as the reader recorded it (state_why, STATE_WHY): "layout"
    says none is in the layout our checks read, and every other reason that no state was read, why, and the reader's
    note on it (not for "not_recorded", whose note only says the same). A context written before the reader recorded
    state_why says the layout line, unless a signal whose name says joints or a state stops short of the episode: then
    the layout is not why (an arm sensor file cut before the footage ends), and the line gives the reader's note on the
    state, which says why. With no other signal, that the dataset records none, unless the reader wrote a note on the
    state or left sensor data unread: then only that none was read, since "records no hand, head or device tracking"
    was false for an MCAP whose hand tracks the reader did not read yet (2026-10-02 audit). There the note and the lists
    of unread channels go to the board, never to the model: they name the checks and channels that did not run ("the
    checks on recorded motion ..."), which would put the words about a recorded motion back into a video only prompt;
    with other signals the prompt is a recording's already. A state_why other than "not_recorded" counts as a note:
    the state is there but was not read, so only that none was read; "not_recorded" and no state_why read as before."""
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
        # An absent designated state field does not prove absence in another recorded observation layout. Inspect
        # preserved values even when their readout is constant or outside the table budget; commands are not state.
        if why == NOT_RECORDED and any((sg.names_joints_or_state(nm) if np.shape(a)[1] > 1
                                       else sg.names_scalar_observed_state(nm))
                                      and np.isfinite(a[:pl["n"]]).any()
                                      for nm, a in ep["signals"].items()):
            why = LAYOUT
        if r == "ego_head" and (why == LAYOUT or why is None and not (short and note)):
            head = "no tracked actor state was read; the recorded signals below retain their own shapes and names."
        elif why == LAYOUT or why is None and not (short and note):
            head = f"no {n['actor']} state in the layout our checks read."
        else:
            reason = STATE_WORDING.get(why) if why is not None else None
            # a recording that holds no state needs no note: the reader's note can only say so again (the board
            # shows it)
            head = (f"no {n['actor']} state was read{f', {reason}' if reason else ''}."
                    + (f" The reader's note on it: {note}" if note and why != NOT_RECORDED else ""))
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
            or ctx.get("state_why") not in (None, NOT_RECORDED)):
        return f"\nRECORDED STATE: none was read from this episode, so {all_there_is}"
    what = "no hand, head or device tracking" if r == "ego_head" else "no robot or gripper state"
    return f"\nRECORDED STATE: none; this dataset records {what}, so {all_there_is}"


def _no_state_text(ep: dict, pl: dict) -> str:
    interpretation = _dictionary_text(ep, pl)
    if interpretation and not _has_signals(ep, pl) and any(field.get("kind") == "state" for field in
            (ep["context"].get("data_dictionary") or {}).get("fields", [])):
        actor = _rig_nouns(rig(ep))["actor"]
        return f"\nRECORDED STATE: no {actor} state in the layout our checks read." + interpretation
    return _no_state_base_text(ep, pl) + interpretation


def _uploader_text(ep: dict, pl: dict) -> str:
    # notes the person who uploaded the episode sent with it (a note file beside a video, an annotation channel in an
    # MCAP), in whatever form they came
    ctx = ep['context']
    text = ctx['uploader_annotation'].strip()
    from board.public_export import reader_annotation_projection
    projected = reader_annotation_projection(text, ctx.get('uploader_notes'))
    if projected is not None:
        text = projected
    return ("\nTHE UPLOADER'S OWN NOTES FOR THIS EPISODE, as sent. They are claims to check against the "
            "video, not ground truth; where the video contradicts them, record it as a data issue:\n"
            + text + "\n")


def _metadata_issues(ep: dict, pl: dict) -> str:
    """Metadata failures and reader limits stay visible beside the claims that could be read."""
    issues = [i["what"] for i in ep["context"].get("reader_issues", [])
              if i.get("kind") in ("metadata_unreadable", "metadata_limit")]
    return "\nUPLOADED METADATA COULD NOT BE FULLY READ:\n" + "\n".join(issues) + "\n" if issues else ""


def _has_metadata_issues(ep: dict, pl: dict) -> bool:
    """Only actual metadata failures or limits add their warning block."""
    return any(i.get("kind") in ("metadata_unreadable", "metadata_limit")
               for i in ep["context"].get("reader_issues", []))


def _state_identity_issues(ep: dict, pl: dict) -> str:
    issues = [i["what"] for i in ep["context"].get("reader_issues", [])
              if i.get("kind") == "state_identity_conflict"]
    text = "\nRECORDED ACTOR IDENTITY DISAGREES:\n" + "\n".join(issues) + "\n" if issues else ""
    note = ep["context"].get("state_identity_note")
    return text + ("\nRECORDED ACTOR GROUP IDENTITY:\n" + note + "\n" if note else "")


def _has_state_identity_issues(ep: dict, pl: dict) -> bool:
    return bool(ep["context"].get("state_identity_note")) or any(
        i.get("kind") == "state_identity_conflict" for i in ep["context"].get("reader_issues", []))


def _table_number_notes(ep: dict) -> list[str]:
    """Numeric interpretation assumptions that qualify the table values shown to the model."""
    return [issue["what"].strip() for issue in ep["context"].get("reader_issues") or []
            if isinstance(issue, dict) and issue.get("kind") == "table_number_ambiguous"
            and isinstance(issue.get("what"), str) and issue["what"].strip()]


def _table_numbers_text(ep: dict, pl: dict) -> str:
    return ("\nTABLE NUMBER INTERPRETATION, inferred while reading the tables:\n"
            + "\n".join("  " + note for note in _table_number_notes(ep)) + "\n")


def _has_table_numbers(ep: dict, pl: dict) -> bool:
    return bool(_table_number_notes(ep))


def _signal_clock_notes(ep: dict) -> list[str]:
    """Unusable signal timing stays visible even when unequal row counts cannot be placed on footage."""
    return [issue["what"] for issue in ep["context"].get("reader_issues") or []
            if isinstance(issue, dict) and issue.get("kind") == "signal_timestamp_invalid" and issue.get("what")]


def _signal_clocks_text(ep: dict, pl: dict) -> str:
    return "\nRECORDED SIGNAL CLOCK LIMITS:\n" + "\n".join(_signal_clock_notes(ep)) + "\n"


def _has_signal_clock_notes(ep: dict, pl: dict) -> bool:
    return bool(_signal_clock_notes(ep))


def _interpretation_words(entry):
    if not entry:
        return ""
    role = "role " + entry["role"] if entry.get("role") else "role cleared"
    meaning = entry.get("meaning") or ""
    return f" [{entry.get('provenance', 'machine')} interpretation, {role}] {meaning}".rstrip()


def _dictionary_text(ep: dict, pl: dict) -> str:
    from label.dictionary_context import field_interpretation
    lines = []
    for field in (ep["context"].get("data_dictionary") or {}).get("fields", []):
        if field.get("kind") not in ("state", "action"):
            continue
        entry = field_interpretation(ep["context"], field["name"], field["kind"])
        if not entry:
            continue
        line = "  " + field["name"] + _interpretation_words(entry)
        groups = [f"{group['name']} columns {group['start']} to {group['start'] + group['count'] - 1}"
                  for group in entry.get("layout") or []]
        if groups:
            line += ". Proposed layout " + "; ".join(groups) + "."
        lines.append(line)
    return "\nDATA DICTIONARY INTERPRETATIONS\n" + "\n".join(lines) if lines else ""


def _has_sensor_evidence(ep: dict, pl: dict) -> bool:
    return bool((pl.get("sensor_evidence") or {}).get("sensors"))


def _camera_observation_views(ep: dict, pl: dict) -> list[str]:
    return pl.get('camera_observation_views', views(ep))


def _needs_camera_observations(ep: dict, pl: dict) -> bool:
    from label.dictionary_context import camera_content_role
    shown = _camera_observation_views(ep, pl)
    return len(shown) > 1 and any(camera_content_role(ep['context'], view) is None for view in shown)


def _camera_observations_text(ep: dict, pl: dict) -> str:
    return prompts.camera_observation_block([(view, cam_name(ep, view)) for view in _camera_observation_views(ep, pl)])


def _action_semantics_text(ep: dict, pl: dict) -> str:
    from prepare.state_contract import action_semantics
    semantics = action_semantics(ep['context'])
    relationship = semantics.get('relationship') or {}
    if relationship.get('kind') == 'next_observation':
        detail = (f"The source defines action as the next observed state, {relationship['row_offset']} source row(s) "
                  "ahead of state. The source declares that its final row repeats its observed state and its targets "
                  "derive from observations. This declaration supplies no independent control-command evidence. The relationship describes "
                  "the original source recording; at a cut the final target can refer to a state outside this request.")
        from prepare.state_contract import redundant_observation_target
        if not redundant_observation_target(ep):
            detail += ' The available paired rows do not verify this relationship; inspect action values separately.'
    elif semantics['role'] == 'command':
        detail = "The preparation records the source action array as control commands."
    elif semantics['role'] == 'target':
        detail = "The source declares action values as targets; their derivation and control relationship are unspecified."
    elif semantics['role'] == 'observation':
        detail = "The source declares the action array as observations."
    else:
        detail = ("The source action semantics are unknown. The array name alone does not establish controller "
                  "commands, observed motion, or the derivation of training targets.")
    return ('\nRECORDED ACTION SEMANTICS: ' + detail + ' Preserve the recorded row times. Differences from state '
            'alone do not establish control failure or physical latency. Source declarations take precedence '
            'over dictionary interpretations.\n')


BLOCKS = (
    Block("collection_note", "intro", _has_collection_note, _collection_text),
    Block('camera_observations', 'frames_detail', _needs_camera_observations, _camera_observations_text,
          schema_fields=('camera_observations',)),
    Block("action_semantics", "state", lambda ep, pl: ep.get('action') is not None and ep['action'].size > 0,
          _action_semantics_text),
    Block("contact_views", "frames_detail", _has_contact_views, _contact_views_text),
    Block("coverage", "frames", _has_coverage, _coverage_note),
    Block("depth", "frames", _has_depth, lambda ep, pl: _depth_note(ep), checks=("sensor_checks",)),
    Block("state", "state", _has_state, _state_text,
          checks=("timebase", "stream_pairing", "recorded_jumps", "gripper_channels", "capture_qc")),
    Block("state_unaligned", "state", _state_unaligned, _state_unaligned_text, checks=("camera_windows_match_state",)),
    Block("no_state", "state", _no_state, _no_state_text),
    Block("table_numbers", "signals", _has_table_numbers, _table_numbers_text),
    Block("signal_clocks", "signals", _has_signal_clock_notes, _signal_clocks_text),
    Block("signals", "signals", _has_signals, _signals_table, checks=("sensor_checks",)),
    Block("contacts", "after_frames", _has_contacts, contacts_block,
          schema_fields=("contacts", "contacts_missing"),
          checks=("contact_checks",)),
    Block("sensor_evidence", "evidence", _has_sensor_evidence,
          lambda ep, pl: se.prompt(pl["sensor_evidence"]),
          schema_fields=("sensor_findings",)),
    Block("uploader_notes", "after_task", _has_uploader_notes, _uploader_text),
    Block("metadata_issues", "after_task", _has_metadata_issues, _metadata_issues),
    Block("state_identity_issues", "after_task", _has_state_identity_issues, _state_identity_issues),
)


# The blocks that make an episode a recording rather than video only: with none of them, the shared instructions and
# the camera paragraph say nothing of a recorded motion (label/prompts.py VIDEO_ONLY_CONTRACT_WORDING).
RECORDED_BLOCKS = ("state", "state_unaligned", "signals")


def present_blocks(ep: dict, pl: dict) -> list[Block]:
    """The blocks whose data this episode holds, in prompt order."""
    return [b for b in BLOCKS if b.present(ep, pl)]


def is_recorded(ep: dict, pl: dict, blocks: list[Block] | None = None) -> bool:
    """Whether shared motion wording has state or other signals behind it. Touch alone measures contact,
    not actor motion; its sensor and contact blocks still remain present with their checks and schema fields.
    """
    return any(b.name in RECORDED_BLOCKS and (b.name != "signals" or any(
        name not in _touch(ep, pl) for name in ep.get("signals", {})))
        for b in (present_blocks(ep, pl) if blocks is None else blocks))


def requested_schema(ep: dict, pl: dict, blocks: list[Block] | None = None) -> tuple:
    """The output fields this episode's blocks ask for beyond the rig's shared schema."""
    return tuple(f for b in (present_blocks(ep, pl) if blocks is None else blocks) for f in b.schema_fields)


def implied_checks(ep: dict, pl: dict, blocks: list[Block] | None = None) -> tuple:
    """The deterministic checks the data of this episode's blocks feeds, each named once (depth and the signals both
    feed sensor_checks)."""
    return tuple(dict.fromkeys(c for b in (present_blocks(ep, pl) if blocks is None else blocks) for c in b.checks))


def recording_window_note(ep: dict) -> str:
    """Describe preparation-owned selection bounds without changing retained source metadata."""
    ctx = ep["context"]
    piece, trimmed = ctx.get("piece"), ctx.get("trimmed")

    def number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    scope = ""
    if isinstance(piece, dict):
        start, end = piece.get("t0_s"), piece.get("t1_s")
        index, count = piece.get("index"), piece.get("count")
        if (number(start) and number(end) and 0 <= start < end
                and type(index) is int and type(count) is int and 1 <= index <= count
                and number(ctx.get("duration_s")) and abs(end - start - ctx["duration_s"]) <= 0.002):
            scope = (f"This request is part {index} of {count}, from {start:g} to {end:g} s on the "
                     "parent recording clock. Its displayed times start on the part's own clock. ")
    elif isinstance(trimmed, dict):
        end, total = trimmed.get("to_s"), trimmed.get("of_s")
        if number(end) and number(total) and 0 < end < total and number(ctx.get("duration_s")):
            if abs(end - ctx["duration_s"]) <= 0.002:
                scope = f"This request covers only the first {end:g} s of a {total:g} s recording. "
        elif number(end) and number(total) and end == total:
            return ""
    if not scope:
        return ("Preparation records a selected request window with unverified bounds; do not assume this "
                "request covers the whole source recording.\n" if piece or trimmed else "")
    return (scope + "Source metadata may describe the full recording, including its larger frame count, "
            "duration and annotations outside this request. That difference alone is not a recording fault. "
            "Judge only the supplied window; an activity unfinished at a preparation cut is not by itself an "
            "operator mistake, failed task or truncated source recording. Use unknown when an outcome lies "
            "outside this window. Still report faults and mistakes supported within the supplied evidence.\n")


def _intro_head(ep: dict) -> str:
    ctx = ep["context"]
    r = rig(ep)
    n = _rig_nouns(r)
    robot = ctx.get("robot_type")
    what = f" ({robot})" if robot else ""
    who = n["who"]
    kind_of = ("one clip of human activity from the {d} dataset, collected to train robots and world "
               "models" if r == "ego_head" else "one episode of a robot-learning demonstration from the {d} dataset{w}")
    window = recording_window_note(ep)
    if window:
        availability = (" Unavailable instants and replacement frames are named below."
                        if ep.get("unavailable_instants") else "")
        return (f"You are labelling {kind_of.format(d=ctx.get('dataset'), w=what)}: {who}. "
                f"You see the sampled footage for this request.{availability} " + window)
    if ep.get("unavailable_instants"):
        return (f"You are labelling {kind_of.format(d=ctx.get('dataset'), w=what)}: {who}. You see the available "
                "sampled footage; unavailable instants and replacement frames are named below.\n")
    return (f"You are labelling {kind_of.format(d=ctx.get('dataset'), w=what)}: {who}. "
            "You see sampled footage from the supplied recording. Source claims and any stated recording "
            "or preparation boundaries are described below.\n")


def _frames_head(ep: dict, cell_w: int, cell_h: int, native: tuple, *, capture_mapped=False) -> str:
    names = ", ".join(cam_name(ep, v) for v in views(ep))
    ends = "first and last available instant" if ep.get("unavailable_instants") else "first and last instant"
    # A retained parent zero may precede its first capture. Name that zero rather than claim the headings start
    # at the first frame, while keeping the existing text for clocks whose first frame is zero.
    first = frame_time(ep, 0)
    origin = ("the episode clock's zero" if round(first, 2) != 0 else "the episode's first frame")
    start = f"The first recorded frame is at {seconds(first)} on this clock. " if round(first, 2) != 0 else ""
    assumed = ep["context"].get("camera_clock") or {}
    precision = ("The column times are assumed presentation instants, not measured capture times. "
                 "Use these times for this presentation. " if anchor(ep) in assumed else
                 "Column times give the anchor camera's display placements. Other cameras can reuse captures "
                 "or have different capture times; use the capture mapping below for image citations and "
                 "comparisons with recorded measurements. " if capture_mapped else
                 "The times are exact: use them, do not invent your own. ")
    qualification = "".join(f"{cam_name(ep, v)}. {note['what']} " for v, note in assumed.items() if v in views(ep))
    time_word = "assumed presentation time" if anchor(ep) in assumed else "anchor time" if capture_mapped else "exact time"
    return (
        f"\nFRAMES. You receive the episode as grid images: ROWS are the cameras ({names}, top to "
        f"bottom), COLUMNS are instants left to right, and each column is headed with its {time_word} in "
        f"seconds from {origin}. {start}Read each grid left to right and the grids in order. "
        f"{precision}{qualification}Each grid cell is the camera frame "
        f"downscaled to {_cell_sizes(ep, cell_w, cell_h)}. After the grids, the episode's {ends} are "
        f"repeated larger, at {_detail_desc(native)}; use them for the start and end state and any "
        "small detail (lettering, a display, fine alignment)." + _excluded_camera_clock_notes(ep))


def camera_frame_interval_s(ep: dict, view: str) -> float:
    """A camera's own frame interval: the median positive step of its capture clock, or one frame at the episode's
    fps when it has none."""
    recorded = (ep.get('recorded_times') or {}).get(view)
    own = (ep.get('times') or {}).get(view) if recorded is None else recorded
    if own is not None:
        t = np.asarray(own, dtype=np.float64)
        steps = np.diff(t[np.isfinite(t)])
        steps = steps[steps > 0]
        if len(steps):
            return float(np.median(steps))
    return 1.0 / ep_fps(ep)


def _ms(x: float | None) -> float | None:
    return None if x is None else round(float(x), 3)


def _camera_capture_rows(ep: dict, pl: dict) -> tuple[dict, dict]:
    """({view: [[display s, source frame, capture s, offset s], ...]}, {view: largest offset left out, in s}).
    A capture offset is listed only when it is at least half a frame interval, of the camera's own clock or of the
    anchor's, whichever is shorter: a smaller one is finer than the column times themselves resolve and finer than the
    camera's own frames, so it cannot change which capture a cell shows or which column an event falls in. Listing
    those (a wrist camera stamped 10 ms after the anchor at every instant) repeated one sub-frame number per instant.
    Times are rounded to the millisecond. An unavailable capture time is always listed."""
    mapping, small = {}, {}
    base = camera_frame_interval_s(ep, anchor(ep))
    for view in views(ep):
        half = 0.5 * min(camera_frame_interval_s(ep, view), base)
        rows = []
        for k in pl.get('ks') or []:
            if not recording_at(ep, view, k):
                continue
            sample = camera_sample(ep, view, k)
            if sample is None:
                continue
            delta = sample['capture_offset_s']
            if delta is None or abs(delta) >= half:
                rows.append([_ms(sample['display_time_s']), sample['source_frame'], _ms(sample['source_time_s']),
                             _ms(delta)])
            elif abs(delta) > 1e-6:
                small[view] = max(small.get(view, 0.0), abs(delta))
        if rows:
            mapping[view] = rows
    return mapping, small


def _small_offsets_text(ep: dict, small: dict) -> str:
    if not small:
        return ''
    each = _and_list([f"{1000 * s:.1f} ms for {cam_name(ep, v)}" for v, s in small.items()])
    return (f"Capture offsets shorter than half a frame interval are not listed; the largest left out is {each}.")


def _camera_capture_text(ep: dict, pl: dict) -> str:
    """Expose nonidentity cell timing without repeating clocks shared by every row (_camera_capture_rows), and say how
    large the sub-frame offsets left out were, so that fact is kept."""
    mapping, small = _camera_capture_rows(ep, pl)
    if not mapping:
        if not small:
            return ''
        each = _and_list([f"{1000 * s:.1f} ms for {cam_name(ep, v)}" for v, s in small.items()])
        return ("\nCAMERA CAPTURE TIMES: every camera's captures are within half a frame interval of their column "
                f"times, at most {each}.")
    return ('\nCAMERA CAPTURE MAPPING\n'
            + (_small_offsets_text(ep, small) + ' ' if small else '') +
            'Each row is [anchor display seconds, source frame, capture seconds, capture minus display seconds]. '
            'Only differing or unavailable capture times are listed. Null capture times cannot establish synchronized '
            'events. Camera clock qualifications still apply. Reused source frames are one capture, not separate '
            'observations. A change between captures is bounded by those capture times; their gap does not establish '
            'when within it the change happened or continuous movement.\n' + json.dumps(mapping, separators=(',', ':')))


def _excluded_camera_clock_notes(ep: dict) -> str:
    """Keep an excluded stream's clock issue scoped to its actual source, without adding it to sampled views."""
    ctx = ep["context"]
    notes = []
    for entry in ctx.get("unshown_cameras") or []:
        clock = entry.get("camera_clock") or {}
        name = entry.get("name")
        domain_reason = (entry.get('presentation_clock') or {}).get('reason')
        clock_note = clock.get('what') or domain_reason
        if not name or not clock_note:
            continue
        issues = [i["what"] for i in ctx.get("reader_issues") or []
                  if i.get("kind") == "camera_timestamp_repeated" and i.get("camera") == name and i.get("what")]
        reason = str(entry.get("why") or "")
        suffix = ". " + clock_note
        if reason.endswith(suffix):
            reason = reason[:-len(suffix)]
        omitted = f"{name} is not shown to the model" + (f" because {reason}" if reason else "") + ". "
        notes.append(omitted + " ".join(dict.fromkeys([*(issues or [clock_note]),
                                                       *([domain_reason] if domain_reason else [])])))
    return " " + " ".join(notes) if notes else ""


def _instants_line(ep: dict, pl: dict | None = None) -> str:
    """Describe the selector's cadence and boundary instants without modifying its picks."""
    spans = (pl.get("quiet_spans") or pl.get("spans") or []) if pl else []
    boundaries = sorted({k for a, b in spans for k in (a, b + 1) if k in pl["ks"]}) if pl else []
    extra = ""
    if spans:
        what = "quiet signal spans" if pl.get("quiet_spans") else "recorded still spans"
        times = ", ".join(f"{frame_time(ep, k):.2f}" for k in boundaries)
        extra = (f" Sampling restarts at the boundary instants of {what} ({times} s), at each span's start and "
                 "the first frame after its end where available.")
        if pl.get("quiet_spans"):
            extra += " Signal quiet does not establish that an arm was still."
    times = (ep.get('times') or {}).get(anchor(ep))
    if times is not None:
        times = np.asarray(times)[:pl['n']] if pl else np.asarray(times)
        intervals = np.diff(times)
        tolerance = max(1e-9, float(np.spacing(np.abs(times).max())) * 4) if len(times) else 1e-9
        if (len(intervals) > 1 and np.all(intervals > 0)
                and not np.allclose(intervals, intervals[0], rtol=1e-6, atol=tolerance)):
            extra += (" Cadence targets select the nearest available source frames on this presentation clock. "
                      "Gaps in that clock remain gaps; frames are not invented between recorded instants.")
    # when every camera's file ends before the episode does, the last instant is the last frame they have (frames)
    last = "frame and the last frame its cameras have" if ep.get("footage_end") is not None else "and last frame"
    cadence = ("within each stretch between boundaries" if spans else "for the whole episode")
    if ep.get("unavailable_instants"):
        return (f"Which instants are planned: one every {every_s(ep):g} s {cadence}, plus its "
                "first and last frame. Unavailable instants and available replacements are named below." + extra)
    return (f"Which instants you get: one every {every_s(ep):g} s {cadence}, plus its first "
            f"{last}." + extra)


def robot_annotation_block(ctx: dict) -> str:
    """Dataset step claims, independent of the episode instruction."""
    if not ctx.get("annotation_subtasks"):
        return ""
    lines = annotation_lines(ctx)
    return ("\nTHE DATASET'S STEP ANNOTATIONS FOR THIS EPISODE (claims to check against the footage):\n"
            + "These are the dataset's claims, not observed actions or an instruction. Check them against the footage; "
              "an unsuccessful flag is the dataset's claim about the outcome.\n"
            + ("  subtasks, with the times the dataset gives:\n" + "\n".join(lines) + "\n" if lines else "")
            + (f"  about these annotations: {ctx['annotation_note'].strip()}\n" if ctx.get("annotation_note") else ""))


def task_block(ep: dict) -> str:
    """The task, part of the base: the dataset's instruction (robot rigs) or annotation (head camera). A head camera
    with no annotation is told there is none, because its shared instructions always ask for goal_alignment."""
    ctx = ep["context"]
    if rig(ep) == "ego_head":
        return ego_annotation_block(ctx)
    given = (ctx.get("instruction") or "").strip()
    annotations = robot_annotation_block(ctx)
    if not given:
        return annotations
    label = "; ".join(ctx.get("task_label") or [])
    # the rules for using the instruction are the same for every episode and live in the cached
    # instructions (instruction_rules); only the instruction itself belongs to the episode
    if policy.effective(ctx)["task_mode"] == "session":
        return ("\nTHE INSTRUCTION FOR THIS RECORDING, as the dataset gives it:\n"
                f"  \"{given}\"\nHow to check it is set out under ABOUT THE RECORDING'S INSTRUCTION above.\n"
                + annotations)
    s = ("\nTHE TASK FOR THIS EPISODE WAS GIVEN TO YOU as the dataset's per-episode instruction:\n"
         f"  \"{given}\"\nHow to use it is set out under ABOUT THE EPISODE'S INSTRUCTION above.\n")
    if ctx.get("instruction_note"):
        s += ctx["instruction_note"].strip() + "\n"
    elif label:
        s += (f"The dataset's coarse task label for this episode is \"{label}\"; the "
              "instruction above is the dataset's per-episode annotation of it, and "
              "the outcome is graded against it.\n")
    return s + annotations


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
    capture_text = _camera_capture_text(ep, pl)
    return (EPISODE_HEADER + _intro_head(ep) + got["intro"] + "\n" + camera_desc(ep, is_recorded(ep, pl, blocks))
            + "\n"
            + _frames_head(ep, cell_w, cell_h, native, capture_mapped="CAMERA CAPTURE MAPPING" in capture_text)
            + capture_text + got["frames_detail"] + "\n" + _instants_line(ep, pl)
            + got["frames"] + got["state"] + got["signals"] + BETWEEN_INSTANTS + got["after_frames"] + "\n"
            + task_block(ep) + got["after_task"] + got['evidence']
            + (prompts.EVIDENCE_REVIEW if got['evidence'] else ''))


def build_prompt(ep: dict, pl: dict, *, cell_w: int, cell_h: int, example_dir=None,
                 blocks: list[Block] | None = None) -> tuple[str, str]:
    """(fixed, episode): the shared instructions (output schema, what the episode is, the data contract),
    identical for every episode of the same rig, instruction presence, sensor-evidence presence and recorded or video only variant (is_recorded),
    then the facts about THIS episode (episode_text). The present blocks are found once, unless the caller has them."""
    ctx = ep["context"]
    r = rig(ep)
    c0 = (ctx.get("cameras") or {}).get(anchor(ep), {})
    native = (c0.get("width") or "native", c0.get("height") or "resolution")
    given = (ctx.get("instruction") or "").strip()
    blocks = present_blocks(ep, pl) if blocks is None else blocks
    fixed = prompts.fixed_instructions(r, has_instruction=bool(given), recorded=is_recorded(ep, pl, blocks),
                                      sessions=policy.effective(ctx)["task_mode"] == "session")
    fixed += prompts.example_block(r, example_dir)
    if any(block.name == 'sensor_evidence' for block in blocks):
        fixed = (prompts.EVIDENCE_CONTRACT + '\nVISUAL ANNOTATION CONTRACT\n' + fixed
                 + '\nEND VISUAL ANNOTATION CONTRACT\n')
    from prepare import jaws
    return fixed, episode_text(ep, pl, cell_w, cell_h, native, blocks) + jaws.prompt(ep.get('jaw_evidence') or {})


EPISODE_HEADER = "\n\nTHE EPISODE TO LABEL.\n\n"


def build_request(ep_dir: Path, *, detail: str = "high", gate=None, cell_w: int | None = None,
                  max_cell_w: int | None = None, grid_cols: int = 4, grid_quality: int = 80,
                  example_dir=None, inspect_evidence=False, model=None, reasoning=None,
                  sample_fps=None, task_mode=None, dataset_note=None, derived_dir=None, read_only=False,
                  inspection=None) -> dict:
    """Everything the harness sends for one episode (content parts), and what it records about it. cell_w fixes
    the cell width; max_cell_w (the routed width, label/route.py) replaces the rig's default widest cell."""
    ep = load(ep_dir)
    ep['context']['annotation_policy'] = policy.resolve(ep['context'], model=model, reasoning=reasoning,
        sample_fps=sample_fps, task_mode=task_mode, dataset_note=dataset_note, inspection=inspection)
    policy.require_request_window(ep['context'], request_duration(ep))
    from prepare import jaws
    ep['jaw_evidence'] = (jaws.for_episode(ep, derived_dir, read_only=True) if read_only
                          else jaws.for_episode(ep, derived_dir))
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
    from label import tactile_capture, sensor_copy
    native_images, capture_selection = tactile_capture.collect(ep, pl, gate)
    native_images, native_parts, native_bytes = tactile_capture.bounded_parts(ep, native_images, detail=detail)
    for view, receipt in capture_selection.items():
        supplied = sorted(native_images.get(view) or {})
        receipt['withheld_by_image_budget'] = len(receipt['selected_source_frames']) - len(supplied)
        receipt['selected_source_frames'] = supplied
    pl['camera_observation_views'] = [v for v in order_views(imgs)
                                    if any(k in imgs[v] and recording_at(ep, v, k) for k in pl['ks'])]
    any_img = next(im[pl["ks"][0]] for im in imgs.values() if pl["ks"][0] in im)
    # A circular area with dark corners is an image observation, not a lens-type or calibration fact.
    ep["lens"] = {v: lens.circular_image(imgs[v]) for v in order_views(imgs)}
    pl["sensor_evidence"] = se.build(ep, pl, shown={v: [k for k in pl["ks"] if k in im and recording_at(ep, v, k)]
                                                  for v, im in imgs.items()}, images=imgs,
                                   native_images=native_images, capture_selection=capture_selection)
    array_parts = se.image_parts(pl['sensor_evidence'])
    array_bytes = sum(len(base64.b64decode(p['image_url']['url'].split(',', 1)[1]))
                      for p in array_parts if p['type'] == 'image_url')
    # depth at the detail instants, and one colour scale per camera from its readings at the sampled instants
    depth_at = _depth_frames(ep, pl)
    if len(views(ep)) == 1:
        # one camera: a grid row holds 6 instants (still under 2048 px wide), halving the per-image overhead
        grid_cols = max(grid_cols, 6)
    cam_labels = [cam_name(ep, v) for v in order_views(imgs)]
    contact = [(k, vs, fullres_stack(ep, imgs, k, "just after a sharp gripper change", frame_time(ep, k), vs))
               for k, vs in ((k, [v for v in vs if recording_at(ep, v, k)]) for k, vs in contact_views(ep, pl)) if vs]
    from label import sensor_scene
    pairing_shown = {v: [k for k in pl['ks'] if k in images and recording_at(ep, v, k)]
                     for v, images in imgs.items()}
    for k, cameras, _ in contact:
        for v in cameras:
            pairing_shown.setdefault(v, []).append(k)
    scene_parts, scene_pairing, scene_bytes = sensor_scene.collect(ep, pl['sensor_evidence'],
        shown=pairing_shown, gate=gate, width=max(widths), grid_cols=grid_cols,
        quality=grid_quality, detail=detail)
    budget = (IMAGE_LIMIT_BYTES / IMAGE_SIZE_INFLATION - DETAIL_VIEW_BYTES_MAX - array_bytes - native_bytes
              - scene_bytes - sum(len(j) for _, _, j in contact))
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
    content.extend(array_parts)
    content.extend(native_parts)
    content.extend(scene_parts)
    extra_bytes = array_bytes + native_bytes + scene_bytes
    # the first frame, the contact views in time order, then the last frame
    views_sent = []
    for k, name in ((pl["ks"][0], "first frame"), (pl["ks"][-1], "last frame")):
        if ep.get("unavailable_instants") and ((name == "first frame" and k != 0)
                                               or (name == "last frame" and k != pl["n"] - 1)):
            name = name.replace("frame", "available frame")
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
            dn = ", ".join(depth_label(ep, v) for v in dv)
            content.append({"type": "text", "text": f"=== depth, {what}, t={frame_time(ep, k):.2f}s | "
                                                    f"{'streams' if len(dv) > 1 else 'stream'} {dn} ==="})
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
    from label.evidence_access import Access, needs_inspection
    mode = policy.inspection_mode(policy.effective(ep['context']))
    access = Access(ep, pl['sensor_evidence'], plan=pl, copy_contract_version=sensor_copy.VERSION) if inspect_evidence and needs_inspection(ep, pl['sensor_evidence'], mode) else None
    return {**({"evidence_access": access} if inspect_evidence else {}), "sensor_evidence": pl["sensor_evidence"],
            "content": content, "prompt": prompt, "plan": pl, "n_grids": n_grids + len(native_parts) // 2,
            'native_tactile_frames': sum(len(frames) for frames in native_images.values()),
            'sensor_scene_pairing': scene_pairing,
            'sensor_copy_contract_version': sensor_copy.VERSION,
            **({"jaw_evidence": ep["jaw_evidence"]} if ep.get("jaw_evidence") else {}),
            "n_images": n_grids + len(views_sent) + len(depth_sent) + len(contacts_sent) + len(array_parts) // 2 + len(native_parts) // 2 + len(scene_parts) // 2,
            "image_bytes": grid_bytes + extra_bytes,
            "contact_s": [round(frame_time(ep, k), 3) for k in pl["contact"]],
            "given_prompt": (ep["context"].get("instruction") or "").strip() or None,
            "task_label": ep["context"].get("task_label"), "cam_labels": cam_labels,
            "cell": [cell_w, cell_h], "timesteps": [round(frame_time(ep, k), 3) for k in pl["ks"]], "lens": ep["lens"],
            "grid_cols": grid_cols, "decode_failed": decode_failures(ep),
            "still_spans": describe_spans(ep, pl["spans"]), "views": views(ep),
            'camera_observation_views': pl['camera_observation_views'],
            'camera_observation_labels': {v: cam_name(ep, v) for v in pl['camera_observation_views']},
            "sampling": f"{rig(ep)}-every-{every_s(ep):g}s",
            "annotation_policy": ep['context']['annotation_policy'],
            "blocks": [b.name for b in blocks],
            "schema_fields": list(requested_schema(ep, pl, blocks)),
            "checks_implied": list(implied_checks(ep, pl, blocks)),
            **({"depth_s": depth_sent, "depth_views": order_views(depth_at)} if depth_sent else {}),
            **({"contact_views": {"shown": contacts_sent, "strips": strips}, "contacts": ep.get("contacts")}
               if contacts_sent else {})}
