"""Capture checks from Sambhav Gupta's public-dataset-adapter (commit 99d0a9e), run on episode sidecars as a
deterministic stage.

    python -m checks.capture_qc [--jobs N] [--force] EPISODES [EPISODES ...]

EPISODES are folders of prepared episode_* folders. For each episode this reads context.json, sources.json,
state.npz, the real-times file and kmap_*.npy files the sources name, and every camera's video, and writes
context["capture_qc"]. An episode that already has a result is skipped unless --force, or unless its result was
written because its worker stopped (needs_check). It exits EXIT_FAILED when some episode's checks could not run.
`python -m board build` copies the result into the episode's dataset_checks on the board.

The checks themselves are the vendored upstream functions (checks/vendor/public_dataset_adapter_qc.py).
This module does three things around them:

1. Adapter from our sidecars to their inputs:
   - End-effector states: handheld ee_pose becomes xyz + rotation vector, the rotation built from our
     roll/pitch/yaw with the convention of label.state._rot_zyx (R = Rz(yaw) Ry(pitch) Rx(roll), which is
     scipy's Rotation.from_euler("xyz", rpy) and upstream's euler_xyz_to_rotvec). Joint-state teleop has
     no forward kinematics in our pipeline, so every check that needs an end-effector pose is reported
     as not assessed there; the gripper column is still used.
   - Timestamps: the anchor camera's real capture times from the real-times file, otherwise k / fps.
   - Actions: derived from the states with the vendored delta_actions, never from state.npz["action"].
   - Video: every camera is decoded on its own native frames inside the episode window, each frame
     matched by its exact pts (as label.frames.extract_frames does), to 64x64 grey with the same
     fast-bilinear scaling upstream uses (PyAV and upstream's ffmpeg command agree within one grey
     level). Frame statistics (exposure, contrast, duplicates, frozen runs) use those native frames.
     Checks that compare video with motion aggregate the camera's own frame-to-frame change to anchor
     intervals through its kmap with cumulative sums, as stream_pairing.stream_motion does; an anchor
     interval in which that camera recorded no new frame is NaN (no observation), never zero. Nothing is
     decoded through kmap duplicates.
   - Gripper unit: an open fraction only where the episode says so (context "gripper_unit", or the
     verified "gripper_value" text "0 = jaws shut, 1 = fully open"). Otherwise the unit-dependent
     gripper checks are not assessed.
   - No native-rate layer: raw_row is never built, so native_rate_qc_unavailable is never emitted.
2. The stage: assess() runs every upstream check, then DISPOSITION turns the fired checks into flags
   (shown as issues), notes (quiet facts) or nothing (excluded), with the reason kept in code. The
   thresholds come from a calibration on our frame-verified datasets (the rule: a check is a flag only if
   every episode it fires on, on every verified dataset of its rig, is a confirmed defect); the finding
   behind each decision is written next to it below.
3. The command line above.

Output, context["capture_qc"]:
    {"source": "public-dataset-adapter@99d0a9e", "version": 2,
     "flags": [{"check", "title", "t_s", "camera", "actor", "evidence"}],
     "notes": [{"check", "text"}],
     "not_assessed": {check: why},
     "checks": [{"check", "name", "group", "status", ...}],
     "metrics": {"cameras": {...}, "actors": {...}, "episode": {...}}}
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from checks.vendor import public_dataset_adapter_qc as up
from label import episode as me
from label import frames as mf
from label.atomic import write_atomic
from prepare.state_notes import STATE_WHY, NOT_RECORDED, LAYOUT

SOURCE = up.SOURCE
VERSION = 2          # the format version of context["capture_qc"]
GRAY = 64

# ------------------------------------------------------------------------------------------ thresholds
#
# Upstream's FilterPolicy defaults stay in the vendored file and are used unchanged. EXTRA holds the few rules
# upstream does not have, per rig, each one found necessary by the calibration and explained where it is used.
# EXTRA_DEFAULT is upstream's behaviour for each of them. None of them is a quantile quota.

EXTRA_DEFAULT = {
    # a pair of consecutive frames counts as a duplicate below this mean absolute grey change
    # (upstream hard-codes 0.25 inside inspect_processed_video)
    "duplicate_pair_mad": 0.25,
    # a clock interval is a gap only if it is also at least this long (upstream: none)
    "gap_min_s": 0.0,
    # frames darker / brighter than these means count as extreme exposure (upstream 5 / 250)
    "black_mean": 5.0,
    "white_mean": 250.0,
    # a frame with grey standard deviation below this is low contrast (upstream 3)
    "low_contrast_std": 3.0,
    # frozen run: longest run of duplicate pairs in seconds (None: upstream's policy.maximum_frozen_run_s)
    "frozen_run_s": None,
    # largest_action_not_in_video: compare each actor's largest step with its own rigidly mounted camera
    # only (upstream: every camera must agree, so the other hand's camera moving hides every spike)
    "largest_action_own_camera": False,
    # duplicates count only where the camera otherwise moves: a run of at most duplicate_max_run repeated
    # pairs between two frame steps of at least this many grey levels (upstream: 0, every pair counts)
    "duplicate_motion_mad": 0.0,
    "duplicate_max_run": 4,
    # jump_return_event: both leap steps at least this many times the neighbouring steps (upstream: 0, no test)
    "jump_isolation": 0.0,
    # video_frozen_run: only while the recorded state says the view must change (upstream: always)
    "frozen_needs_motion": False,
}

EXTRA: dict[str, dict] = {
    "teleop_arms": {"duplicate_pair_mad": 0.1, "duplicate_motion_mad": 3.0, "gap_min_s": 0.25,
                    "frozen_needs_motion": True},
    "handheld_gripper": {"duplicate_pair_mad": 0.1, "duplicate_motion_mad": 3.0, "largest_action_own_camera": True,
                         "jump_isolation": 3.0, "gap_min_s": 0.25, "frozen_needs_motion": True},
    "ego_head": {"duplicate_pair_mad": 0.1, "duplicate_motion_mad": 3.0, "gap_min_s": 0.25,
                 "frozen_needs_motion": True},
}


# the checks that compare the recorded state with the video (video_frozen_run too where a frozen picture is claimed only
# while the state says the view must change, frozen_needs_motion)
STATE_VS_VIDEO = ("camera_state_alignment_mismatch", "largest_action_not_in_video",
                  "visual_change_unexplained_by_action", "pixel_action_corr_mismatch")


def policy_for(rig: str) -> tuple[up.FilterPolicy, dict]:
    """Upstream's FilterPolicy and our per-rig rules for this rig."""
    policy = up.FilterPolicy()
    extra = {**EXTRA_DEFAULT, **EXTRA.get(rig, {})}
    if extra["frozen_run_s"] is None:
        extra["frozen_run_s"] = policy.maximum_frozen_run_s
    return policy, extra


# ------------------------------------------------------------------------------------------ checks
#
# Every check upstream's evaluate_episode can emit, the native_* family as one entry and the per-side gripper and
# correlation reasons merged into one name each. DISPOSITION (below assess) says what is done with each.

CHECKS = [
    "missing_canonical_signal", "invalid_state_shape", "invalid_action_shape", "state_action_count_mismatch",
    "nonfinite_signal", "episode_too_short", "invalid_rotation_matrix", "nonprincipal_rotation_state",
    "normalized_gripper_out_of_range", "gripper_action_integral_out_of_range", "gripper_never_acts",
    "gripper_sensor_bug", "state_time_non_monotonic_or_duplicate", "action_time_non_monotonic_or_duplicate",
    "state_timestamp_gap", "native_camera_timestamp_gap", "missing_camera", "camera_state_alignment_mismatch",
    "video_decode_failure", "video_decode_frame_count_mismatch", "video_extreme_exposure", "video_low_contrast",
    "video_duplicate_frames", "video_frozen_run", "se3_translation_round_trip_failure",
    "se3_rotation_round_trip_failure", "state_time_too_short", "action_time_too_short",
    "action_smoothness_discontinuity", "jump_return_event", "gross_umi_speed", "over_95_percent_static",
    "largest_action_not_in_video",
    "visual_change_unexplained_by_action", "pixel_action_corr_mismatch", "native_rate_qc_unavailable",
    "native_signal_checks", "processing_failure",
]


# ------------------------------------------------------------------------------------------ sidecars

def gripper_unit(ctx: dict) -> str:
    """normalized_open_fraction only where the episode states it; otherwise the stated unit or unknown."""
    u = ctx.get("gripper_unit")
    if isinstance(u, dict):
        u = u.get("unit")
    if isinstance(u, str) and u:
        return u
    text = str(ctx.get("gripper_value") or "")
    if re.search(r"\b0\s*=\s*jaws shut,\s*1\s*=\s*fully open", text):
        return "normalized_open_fraction"
    if "metre" in text or "meter" in text:
        return "metres"
    if text:
        return "source_units"
    return "unknown"


def anchor_times(ep: dict) -> np.ndarray:
    """Seconds from the episode start for every anchor frame."""
    n = len(ep["state"])
    if ep.get("times") is not None:
        t = np.asarray(ep["times"][me.anchor(ep)], dtype=np.float64)
        return t[:n]
    return np.arange(n, dtype=np.float64) / me.ep_fps(ep)


def camera_times(ep: dict, v: str) -> np.ndarray | None:
    """Camera v's selected clock, including explicitly qualified presentation times. Clock defect checks read
    recorded_times instead, so assumed placement cannot hide repeated recorded timestamps."""
    if ep.get("times") is not None and v in ep["times"]:
        return np.asarray(ep["times"][v], dtype=np.float64)
    return None


def actor_views(ep: dict, names: list[str]) -> list[str | None]:
    """The camera view mounted on each actor in native state order, or None when that mapping is unknown."""
    return me.actor_views(ep)


def canonical_states(ep: dict) -> dict:
    """Upstream's [T,14] layout (x y z m, rotation vector rad, gripper per group) in native state order with its
    validity mask. Upstream calls the first and second slots left and right; those are positional keys, whose
    recorded actor names and wrist views come from the episode rather than the slot labels.
    Joint-state rigs get only the gripper column (no forward kinematics here). A frame with a missing
    or infinite value is listed in "nonfinite" and left invalid for its actor, and every other frame is checked. Only
    the frames the
    state covers are read (label/episode.py state_span): a state moved onto another camera's frames has no value
    outside them (board/clips.py reanchor), which is not a missing value of the recording, so they are left invalid."""
    from scipy.spatial.transform import Rotation
    s = np.asarray(ep["state"], dtype=np.float64)
    T = len(s)
    kind = me.state_kind(ep)
    states = np.zeros((T, 14))
    valid = np.zeros((T, 14), dtype=bool)
    names = me.actors(ep)
    out = {"states": states, "valid": valid, "actors": names, "kind": kind, "nonfinite": [], "shape_ok": True}
    if kind == "none":
        return out
    if s.ndim != 2 or s.shape[1] not in (7, 14):
        out["shape_ok"] = False
        return out
    n_act = s.shape[1] // 7
    a, b = me.state_span(ep)
    for g in range(n_act):
        block = s[a:b, 7 * g:7 * g + 7]
        # a row with a missing (NaN) or infinite value is reported and left out (invalid); every other row is checked
        ok = np.isfinite(block).all(axis=1)
        if not ok.all():
            out["nonfinite"].append({"actor": names[g], "rows": int((~ok).sum()),
                                     "first_row": a + int(np.flatnonzero(~ok)[0])})
        rows = a + np.flatnonzero(ok)
        o = 7 * g
        if kind == "ee_pose":
            states[rows, o:o + 3] = block[ok, 0:3]
            if len(rows):
                states[rows, o + 3:o + 6] = Rotation.from_euler("xyz", block[ok, 3:6]).as_rotvec()
            valid[rows, o:o + 6] = True
        states[rows, o + 6] = block[ok, 6]
        valid[rows, o + 6] = True
    return out


# ------------------------------------------------------------------------------------------ video

def _targets(ep: dict, v: str, stream) -> list[int] | None:
    """Exact pts of camera v's frames 0..n-1 inside its file, or None to take frames in decode order."""
    s = ep["sources"][v]
    n = int(s["n_frames"])
    pts = ep["times"].get(f"{v}_pts") if ep.get("times") is not None else None
    if pts is not None:
        return [int(p) for p in pts[:n]]
    fps = me.ep_fps(ep)
    try:
        b0, step = mf.base_frame(float(s.get("base_s") or 0.0), fps), mf.frame_pts_step(stream.time_base, fps)
        return [(b0 + k) * step for k in range(n)]
    except mf.FrameError:
        if float(s.get("base_s") or 0.0) == 0.0:
            # the episode's own file, not a packed one: frame k is the k-th decoded frame
            return None
        raise


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "")
    return int(v) if v.isdigit() and int(v) > 0 else default


# Decoding is most of this stage's time. An episode's cameras decode at the same time, each with a few frame
# threads, and the frames decoded are the same with any setting; both are bounded for shared hosts
# (RDA_CAPTURE_DECODE_THREADS per camera, RDA_CAPTURE_CAMERA_THREADS cameras at once).
DECODE_THREADS = _env_int("RDA_CAPTURE_DECODE_THREADS", 2)
CAMERA_THREADS = _env_int("RDA_CAPTURE_CAMERA_THREADS", 3)
_FEATURES = threading.Lock()


def decode_gray(ep: dict, v: str) -> dict:
    """Camera v's own frames of this episode as [n,64,64] grey ("frames"), with "got" marking the frames that
    decoded at their recorded times.

    A missing file raises (an infrastructure fault, not a data defect). A decoder error on an existing
    file is returned as "error" (upstream's video_decode_failure)."""
    import av
    from av.video.reformatter import VideoReformatter
    s = ep["sources"][v]
    n = int(s["n_frames"])
    path = Path(s["packed"])
    # one scaler for every frame: frame.reformat() builds a new one per frame, which was most of the stage's time;
    # the same scaler reused gives byte-identical frames
    scaler = VideoReformatter()

    def to_gray(fr) -> np.ndarray:
        return scaler.reformat(fr, width=GRAY, height=GRAY, format="gray", interpolation="FAST_BILINEAR").to_ndarray()
    if not path.exists():
        raise FileNotFoundError(f"camera {v}: {path} not found")
    frames = np.zeros((n, GRAY, GRAY), dtype=np.uint8)
    got = np.zeros(n, dtype=bool)
    placeholder = np.zeros(n, dtype=bool)
    km = ep["kmap"].get(v)
    for first, last in (ep["context"].get("placeholder_frames") or {}).get(v, []):
        if km is None:
            placeholder[max(0, first):min(n, last + 1)] = True
        else:
            own = np.asarray(km[max(0, first):last + 1], dtype=int)
            placeholder[own[(own >= 0) & (own < n)]] = True
    info = {"n_expected": n, "error": None}
    try:
        with av.open(str(path)) as c:
            st = c.streams.video[0]
            # frame threading returns the same frames in the same order
            st.codec_context.thread_count = DECODE_THREADS
            if DECODE_THREADS > 1:
                st.thread_type = "AUTO"
            targets = _targets(ep, v, st)
            if targets is None:
                k = 0
                for fr in c.decode(st):
                    if k >= n:
                        break
                    if not fr.is_corrupt and not placeholder[k]:
                        frames[k] = to_gray(fr)
                        got[k] = True
                    k += 1
            else:
                index: dict[int, int] = {}
                for k, p in enumerate(targets):
                    index.setdefault(p, k)
                first, last = min(targets), max(targets)
                c.seek(first, stream=st, backward=True, any_frame=False)
                for fr in c.decode(st):
                    if fr.pts is None or fr.pts < first:
                        continue
                    if fr.pts > last:
                        break
                    k = index.get(fr.pts)
                    if k is None:
                        continue
                    if not fr.is_corrupt and not placeholder[k]:
                        frames[k] = to_gray(fr)
                        got[k] = True
                    if got[-1] and k == n - 1:
                        break
    except FileNotFoundError:
        raise
    except Exception as e:  # decoder failure on an existing file
        info["error"] = f"{type(e).__name__}: {e}"[:300]
    info["frames"] = frames
    info["got"] = got
    return info


def cam_fps(ep: dict, v: str, n: int) -> float:
    """Camera v's frame rate: the mean rate over its window when it has a real clock (a median interval
    is useless on clocks that stamp frames in bursts: some ABC-130k cameras put half their intervals at
    1 us and the rest near 150 ms), otherwise the episode fps."""
    ct = camera_times(ep, v)
    if ct is not None and len(ct) > 1:
        m = min(n, len(ct))
        if ct[m - 1] > ct[0]:
            return float((m - 1) / (ct[m - 1] - ct[0]))
    return me.ep_fps(ep)


def camera_features(ep: dict, v: str) -> dict:
    """Everything the video checks need from one camera: per-frame grey mean and standard deviation, and per
    consecutive pair the mean absolute change and a tile-median change that ignores a global brightness change."""
    d = decode_gray(ep, v)
    with _FEATURES:   # one camera's float arithmetic at a time: a 15-minute camera needs over 1 GB here
        return _features(ep, v, d)


def _features(ep: dict, v: str, d: dict) -> dict:
    n = d["n_expected"]
    got = d["got"]
    fr = d["frames"].astype(np.float32)
    means = fr.mean(axis=(1, 2))
    stds = fr.std(axis=(1, 2))
    if n > 1:
        delta = fr[1:] - fr[:-1]
        pair = np.abs(delta).mean(axis=(1, 2))
        delta -= np.median(delta, axis=(1, 2), keepdims=True)
        tiles = np.abs(delta).reshape(-1, 8, 8, 8, 8).mean(axis=(2, 4)).reshape(-1, 64)
        pchange = 0.75 * np.median(tiles, axis=1) + 0.25 * np.quantile(tiles, 0.75, axis=1)
        both = got[1:] & got[:-1]
        pair[~both] = np.nan
        pchange[~both] = np.nan
    else:
        pair = np.zeros(0, np.float32)
        pchange = np.zeros(0, np.float32)
    means[~got] = np.nan
    stds[~got] = np.nan
    fps = cam_fps(ep, v, n)
    return {"n": n, "decoded": int(got.sum()), "error": d["error"], "fps": fps,
            "means": means.astype(np.float32), "stds": stds.astype(np.float32),
            "pair": pair.astype(np.float32), "pchange": pchange.astype(np.float32)}


def to_anchor(ep: dict, v: str, own: np.ndarray, T: int) -> np.ndarray | None:
    """Per-anchor-interval values (len T-1) from a camera's own per-frame-pair values (len n-1).
    A camera without a kmap must have exactly the anchor's frames. With a kmap, interval i sums the
    camera's own changes from frame km[i] to km[i+1]; NaN when the camera has no new frame there."""
    km = ep["kmap"].get(v)
    n = len(own) + 1
    if km is None:
        return own[:T - 1].astype(np.float64) if n == T else None
    km = np.asarray(km, dtype=np.int64)[:T]
    if len(km) < T:
        return None
    km = np.clip(km, 0, n - 1)
    finite = np.isfinite(own)
    cum = np.concatenate([[0.0], np.cumsum(np.where(finite, own, 0.0))])
    bad = np.concatenate([[0], np.cumsum(~finite)])
    a, b = km[:-1], km[1:]
    out = cum[b] - cum[a]
    out[(b <= a) | ((bad[b] - bad[a]) > 0)] = np.nan
    return out


def _camera_or_failure(ep: dict, v: str) -> dict:
    """camera_features, or, when the camera's frames cannot be read at all (its file gone, a crash computing them),
    {"failed": the exception, "n": its frame count}, which assess records as that camera's crash: the checks not run
    on it name it with the error and the other cameras are checked."""
    try:
        return camera_features(ep, v)
    except Exception as e:  # noqa: BLE001 - recorded on the camera by assess
        return {"failed": e, "n": int(ep["sources"][v]["n_frames"])}


def extract(ep_dir: Path) -> dict:
    """Load the sidecars and decode every camera once. The result is all assess() needs. The cameras decode at the
    same time (the decoder runs outside Python's lock), each on its own frames, so the result is the same."""
    ep = me.load(ep_dir)
    T = len(ep["state"])
    views = me.views(ep)
    with ThreadPoolExecutor(max_workers=max(1, min(len(views), CAMERA_THREADS))) as pool:
        feats = list(pool.map(lambda v: _camera_or_failure(ep, v), views))
    return {"ep": ep, "T": T, "cams": dict(zip(views, feats))}


# ------------------------------------------------------------------------------------------ assess

def _r(status: str, why: str | None = None, events=None, metrics=None) -> dict:
    return {"status": status, "why": why, "events": events or [], "metrics": metrics or {}}


def _na(why: str) -> dict:
    return _r("not_assessed", why)


def _errored(e: Exception) -> dict:
    """A check that stopped with an error: status errored, the error, and why in words (format_result lists it)."""
    msg = f"{type(e).__name__}: {e}"[:300]
    return {**_r("errored", f"the check stopped with an error ({msg})"), "error": msg}


class _guard:
    """A block of checks in assess: a crash inside it records each of its checks that has no result yet as errored,
    with the error, and every other check of the episode runs on, so no check costs another."""

    def __init__(self, R: dict, names):
        self.R, self.names = R, tuple(names)

    def __enter__(self):
        return self

    def __exit__(self, et, e, tb) -> bool:
        if not isinstance(e, Exception):
            return False
        for c in self.names:
            self.R.setdefault(c, _errored(e))
        return True


VIDEO_CHECKS = ("missing_camera", "camera_state_alignment_mismatch", "video_decode_failure",
                "video_decode_frame_count_mismatch", "video_extreme_exposure", "video_low_contrast",
                "video_duplicate_frames", "video_frozen_run")
PIXEL_CHECKS = ("video_extreme_exposure", "video_low_contrast", "video_duplicate_frames", "video_frozen_run")
# the motion checks that compare every camera's picture with the recorded motion
CAMERA_MOTION_CHECKS = ("largest_action_not_in_video", "visual_change_unexplained_by_action",
                        "pixel_action_corr_mismatch")
CLOCK_CHECKS = ("state_time_too_short", "state_time_non_monotonic_or_duplicate", "state_timestamp_gap",
                "native_camera_timestamp_gap")
MOTION_CHECKS = ("action_smoothness_discontinuity", "jump_return_event", "gross_umi_speed", "over_95_percent_static",
                 "largest_action_not_in_video", "visual_change_unexplained_by_action", "pixel_action_corr_mismatch")
STRUCTURE_CHECKS = ("missing_canonical_signal", "invalid_state_shape", "nonfinite_signal")
GRIPPER_CHECKS = ("normalized_gripper_out_of_range", "gripper_action_integral_out_of_range", "gripper_never_acts",
                  "gripper_sensor_bug")
# the checks that read the recorded state (canonical_states): when it cannot be read, these are errored and the rest run
STATE_CHECKS = STRUCTURE_CHECKS + GRIPPER_CHECKS + MOTION_CHECKS


def _ev(evidence: str, t_s: float | None = None, camera: str | None = None, actor: str | None = None) -> dict:
    """One firing: where it is (seconds, camera, actor; None when it is the whole episode) and what was measured."""
    return {"t_s": t_s, "camera": camera, "actor": actor, "evidence": evidence}


def _fired(events: list[dict], metrics=None) -> dict:
    """fired when there is at least one event, else clear."""
    return _r("fired" if events else "clear", events=events, metrics=metrics)


def _t(ts: np.ndarray, i: int) -> float | None:
    if 0 <= i < len(ts):
        return round(float(ts[i]), 2)
    return None


def _cam_t(ep: dict, v: str, j: int, fps: float) -> float:
    ct = camera_times(ep, v)
    if ct is not None and 0 <= j < len(ct):
        return round(float(ct[j]), 2)
    return round(j / fps, 2)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index runs where mask is true."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def _no_state_reason(ep: dict) -> str:
    """Canonical state can be withheld while its recorded channels remain qualified signals."""
    from label.signals import names_joints_or_state, names_scalar_observed_state

    ctx = ep["context"]
    why = ctx.get("state_why")
    note = (ctx.get("state_note") or "").strip()
    recorded = any((names_joints_or_state(name) if np.ndim(a) > 1 and np.shape(a)[1] > 1
                    else names_scalar_observed_state(name)) for name, a in (ep.get("signals") or {}).items())
    if recorded and why in (None, NOT_RECORDED):
        why = LAYOUT
    if why == NOT_RECORDED or not (why or note or recorded):
        return "the recording has no robot state, only video"
    reason = STATE_WHY.get(why)
    return ("no usable robot state was read for this check" + (f": {reason}" if reason else "")
            + (f". The reader's note on it: {note}" if note else ""))


def assess(feats: dict) -> dict:
    """Run every upstream check on one episode (feats from extract()). Returns {"checks": {check: {status, why,
    events, metrics}}, "cameras", "actors", "episode"}, status one of fired, clear, not_assessed. Each block names
    the upstream lines it reproduces (filtering.py, processing.py of public-dataset-adapter)."""
    ep, T, cams = feats["ep"], feats["T"], feats["cams"]
    ctx = ep["context"]
    assumed_clocks = ctx.get("camera_clock") or {}
    anchor = me.anchor(ep) if cams else None
    # The same frame-window eligibility as label.episode.plan, without running its sampling work here.
    state_window_ok = (anchor is not None and not ctx.get("state_unaligned")
                       and int(ep["sources"][anchor]["n_frames"]) == T
                       and all(len(km) >= T for v, km in ep["kmap"].items() if v != anchor and v in cams))
    comparison_cameras = ({v for v in cams if v not in assumed_clocks and anchor not in assumed_clocks}
                          if state_window_ok else set())
    rig = me.rig(ep)
    kind = me.state_kind(ep)
    policy, extra = policy_for(rig)
    R: dict[str, dict] = {}
    ts = anchor_times(ep)
    real_times = ep.get("times") is not None
    unit = gripper_unit(ctx)
    normalized = "normalized_open_fraction" in unit
    # a crash reading the state errors only the checks that read it (STATE_CHECKS, set at the end), and the camera,
    # clock and length checks run on as for a recording with no usable state
    state_error = None
    try:
        cs = canonical_states(ep)
    except Exception as e:  # noqa: BLE001 - recorded on the checks that read the state
        state_error = e
        cs = {"states": np.zeros((T, 14)), "valid": np.zeros((T, 14), dtype=bool), "actors": me.actors(ep),
              "kind": kind, "nonfinite": [], "shape_ok": True}
    names = cs["actors"]
    states, valid = cs["states"], cs["valid"]
    has_state = kind != "none"
    has_pose = kind == "ee_pose"
    no_state_why = _no_state_reason(ep)
    no_pose_why = ("the recording has joint angles but no gripper pose, which this check needs"
                   if kind == "joints" else no_state_why)

    # constant exclusions: these upstream reasons test upstream's own processing or a layer we do not build
    R["invalid_action_shape"] = _na("we compute the actions from the recorded state, so their shape is always valid")
    R["state_action_count_mismatch"] = _na("we compute the actions from the recorded state, so there is one for every state frame")
    R["se3_translation_round_trip_failure"] = _na("this tests how the actions are computed, not the recording")
    R["se3_rotation_round_trip_failure"] = _na("this tests how the actions are computed, not the recording")
    R["action_time_too_short"] = _na("action times are taken from the frame times, which are checked on their own")
    R["native_rate_qc_unavailable"] = _na("this flags a missing full-rate copy of the signals, which the pipeline does not use")
    R["native_signal_checks"] = _na("the pipeline keeps no separate full-rate copy of the signals to check")
    R["processing_failure"] = _na("an episode that fails to process is reported by the step that failed")
    R["action_time_non_monotonic_or_duplicate"] = _na("action times are taken from the frame times, which are checked on their own")

    # ---- structure (filtering.py:1461-1488)
    with _guard(R, STRUCTURE_CHECKS):
        if not has_state:
            for c in STRUCTURE_CHECKS:
                R[c] = _na(no_state_why)
        else:
            shape = np.shape(ep["state"])
            width = shape[-1] if len(shape) == 2 else list(shape)
            R["invalid_state_shape"] = _fired([] if cs["shape_ok"] else [_ev(
                f"the recorded state has {width} values per frame, a layout these checks do not read (they read 7 per "
                f"arm or gripper), so the state checks were skipped")])
            R["nonfinite_signal"] = _fired([_ev(f"{e['rows']} frames of the {e['actor']} state contain NaN or infinite "
                                                f"values", _t(ts, e["first_row"]), actor=e["actor"])
                                            for e in cs["nonfinite"]])
            empty = cs["shape_ok"] and T > 1 and not valid.any() and not cs["nonfinite"]
            R["missing_canonical_signal"] = _fired([_ev("no valid state value in the episode")] if empty else [])

    # ---- duration (filtering.py:1490)
    duration = float(ts[-1]) if len(ts) else 0.0
    with _guard(R, ("episode_too_short",)):
        last = [float(ts[-1])] if len(ts) else [0.0]
        for v, c in cams.items():
            if "failed" in c:
                continue        # a camera whose frames could not be read gives no length
            ct = camera_times(ep, v)
            last.append(float(ct[c["n"] - 1]) if ct is not None and len(ct) >= c["n"] else (c["n"] - 1) / c["fps"])
        duration = max(last)
        short = duration < policy.minimum_episode_duration_s
        R["episode_too_short"] = _fired([_ev(f"the episode lasts {duration:.1f} s; the rule is under "
                                             f"{policy.minimum_episode_duration_s:g} s")] if short else [],
                                        metrics={"duration_s": round(duration, 2)})

    # actions (processing.py:1637), from our states on the anchor clock
    ts_ns = np.round((ts - (ts[0] if len(ts) else 0.0)) * 1e9).astype(np.int64)
    # a frame with a missing value is left out (canonical_states) and reported (nonfinite_signal); the state is checked
    # on the frames that have readings
    usable_state = has_state and cs["shape_ok"] and bool(valid.any()) and T > 2
    unusable_why = ("the state layout is not 7 values per arm or gripper" if has_state and not cs["shape_ok"]
                    else "the state has no frame without a missing (NaN) or infinite value" if not valid.any()
                    else "fewer than 3 state frames")
    try:
        if usable_state:
            local, global_, avalid, _ = up.delta_actions(states, valid, ts_ns)
            local = local.astype(np.float64)
            global_ = global_.astype(np.float64)
    except Exception as err:  # noqa: BLE001 - the checks that need the actions say why, the rest run on
        usable_state, unusable_why = False, f"the actions could not be computed ({type(err).__name__}: {err})"[:300]
    if not usable_state:
        local = global_ = np.zeros((max(T - 1, 0), 14))
        avalid = np.zeros((max(T - 1, 0), 14), dtype=bool)
    robot = {"states": states, "state_valid": valid, "actions_local": local, "action_valid_local": avalid,
             "actions_global": global_, "action_valid_global": avalid, "gripper_unit": unit}

    # ---- rotations (filtering.py:1306-1324)
    why_rot = "we compute the rotations from roll, pitch and yaw, so they are always valid"
    R["invalid_rotation_matrix"] = _na(why_rot if has_pose else no_pose_why)
    R["nonprincipal_rotation_state"] = _na(why_rot if has_pose else no_pose_why)

    # ---- grippers (filtering.py:1327-1333, 936-996, 999-1052, 1131-1158)
    unit_why = ((f"the gripper reading is in {unit}, not a verified opening from 0 to 1" if unit != "unknown"
                 else "the gripper reading's unit is not known for this dataset")
                + ", and this check needs an opening from 0 to 1")
    with _guard(R, GRIPPER_CHECKS):
        if not usable_state:
            for c in GRIPPER_CHECKS:
                R[c] = _na(no_state_why if not has_state else unusable_why)
        elif not normalized:
            for c in GRIPPER_CHECKS:
                R[c] = _na(unit_why)
        else:
            ev = []
            for g, name in enumerate(names):
                gv, gok = states[:, 7 * g + 6], valid[:, 7 * g + 6]
                bad = np.flatnonzero(gok & ((gv < -0.05) | (gv > 1.05)))
                if len(bad):
                    ev.append(_ev(f"{name} gripper reads {gv[gok].min():.3f} to {gv[gok].max():.3f} ({len(bad)} "
                                  f"frames outside "
                                  f"-0.05..1.05 of a 0-1 open fraction)", _t(ts, int(bad[0])), actor=name))
            R["normalized_gripper_out_of_range"] = _fired(ev)

            smoothed, _ = up.smooth_canonical_trajectory(states, valid, ts_ns)
            s_local, _, _, _ = up.delta_actions(smoothed.astype(np.float64), valid, ts_ns)
            robot_i = dict(robot, smoothed_actions_local=s_local.astype(np.float64))
            integ, _ = up._gripper_integral_checks(robot_i, states, valid, policy)
            ev = []
            for key, m in integ.items():
                variant, arm = key.split("_", 1)
                g = 0 if arm == "left" else 1
                if m["out_of_range_count"] and g < len(names):
                    i = int(m["out_of_range_indices"][0])
                    ev.append(_ev(f"replaying the {variant} {names[g]} gripper deltas reaches "
                                  f"{m['minimum_replayed_open_fraction']:.3f} to "
                                  f"{m['maximum_replayed_open_fraction']:.3f}, "
                                  f"outside 0..1 by more than {policy.gripper_integral_tolerance:g} on "
                                  f"{m['out_of_range_count']} steps", _t(ts, i + 1), actor=names[g]))
            R["gripper_action_integral_out_of_range"] = _fired(ev)

            inactive, _ = up._inactive_gripper_checks(robot, policy)
            ev = []
            for arm, m in inactive.items():
                g = 0 if arm == "left" else 1
                if m["status"] == "recorded" and m["active_action_count"] == 0 and g < len(names):
                    ev.append(_ev(f"the {names[g]} gripper reading never changes by more than "
                                  f"{policy.static_gripper_delta:g} between frames (largest change "
                                  f"{m['maximum_absolute_delta']:.2g})", actor=names[g]))
            R["gripper_never_acts"] = _fired(ev)

            _, _, sev = up.gripper_sensor_bug_checks(robot, policy)
            ev = []
            for e in sev:
                g = 0 if e["arm"] == "left" else 1
                if g < len(names):
                    ev.append(_ev(f"the {names[g]} gripper jumps {e['first_delta']:+.2f} then {e['second_delta']:+.2f} "
                                  f"(open fraction) on consecutive frames; the rule is two opposite steps of at least "
                                  f"{policy.gripper_sensor_bug_minimum_delta:g}", _t(ts, e["pivot_frame"]),
                                  actor=names[g]))
            R["gripper_sensor_bug"] = _fired(ev)

    # ---- clocks (filtering.py:463-504, 1530-1541, 1774-1790)
    gap_min = float(extra["gap_min_s"])

    def clock(values_s: np.ndarray, label: str, camera: str | None):
        """Repeat-or-backwards events, gap events and metrics of one clock (seconds per frame)."""
        dup_ev, gap_ev = [], []
        values_s = np.asarray(values_s, dtype=np.float64)
        if not np.isfinite(values_s).all():
            return [], [], {"status": "not_assessed", "why": f"the {camera or 'state'} capture clock contains nonfinite values"}
        # Reject unrepresentable seconds before multiplying or casting them into the upstream int64 clock.
        limit = np.iinfo(np.int64).max / 1e9
        if (np.abs(values_s) >= limit).any():
            return [], [], {"status": "not_assessed",
                            "why": f"the {camera or 'state'} capture clock exceeds the int64 nanosecond range"}
        tns = np.round(values_s * 1e9).astype(np.int64)
        reasons: list[str] = []
        dts = up._strict_timestamps(tns, label, reasons)
        if any(r.endswith("_non_monotonic_or_duplicate") for r in reasons):
            bad = np.flatnonzero(dts <= 0)
            dup_ev.append(_ev(f"{len(bad)} frame times on the {camera or 'state'} clock repeat or go backwards",
                              round(float(values_s[bad[0] + 1]), 2), camera))
        # Repeated stamps describe clock resolution, not zero-length capture intervals. Keep their events above
        # and compare positive intervals using the same upstream gap rule, retaining the original frame indices.
        positive = np.flatnonzero(dts > 0)
        gm = up._gap_metrics(dts[positive])
        idx = [int(positive[i]) for i in gm["gap_interval_indices"] if dts[positive[i]] >= gap_min]
        for i in idx[:5]:
            floor = f" and at least {gap_min * 1000:.0f} ms" if gap_min > 0 else ""
            gap_ev.append(_ev(f"{dts[i] * 1000:.0f} ms between frames where the clock's median is "
                              f"{gm['median_dt_s'] * 1000:.1f} ms (rule: over {gm['gap_threshold_s'] * 1000:.1f} ms"
                              f"{floor})", round(float(values_s[i]), 2), camera))
        if len(idx) > 5:
            gap_ev[-1]["evidence"] += f"; {len(idx)} such gaps in total"
        mets = {"median_dt_ms": round(gm["median_dt_s"] * 1000, 2) if gm["median_dt_s"] is not None else None,
                "max_dt_ms": round(gm["maximum_dt_s"] * 1000, 1) if gm["maximum_dt_s"] is not None else None,
                "gaps": len(idx)}
        return dup_ev, gap_ev, mets

    with _guard(R, CLOCK_CHECKS):
        R["state_time_too_short"] = _fired([_ev(f"the recording has {len(ts)} frame time{'s' if len(ts) != 1 else ''}, "
                                                f"fewer than two")] if len(ts) < 2 else [])
        if real_times:
            recorded = ep.get("recorded_times") or ep.get("times")
            raw_anchor = np.asarray(recorded[me.anchor(ep)], dtype=np.float64)[:len(ts)]
            dup, gap, m = clock(raw_anchor, "state_time", me.anchor(ep))
            anchor_bad = m.get("status") == "not_assessed"
            bad = [m["why"]] if anchor_bad else []
            R["state_timestamp_gap"] = _na(m["why"]) if anchor_bad else _fired(gap, metrics=m)
            nev, nm, assessed = [], {}, 0
            for v in me.views(ep):
                if v == me.anchor(ep):
                    continue
                ct = recorded.get(v)
                if ct is None:
                    continue
                d2, g2, m2 = clock(ct[:cams[v]["n"]], f"native_camera_{v}", v)
                dup.extend(d2)
                nev.extend(g2)
                nm[v] = m2
                if m2.get("status") == "not_assessed":
                    bad.append(m2["why"])
                else:
                    assessed += 1
            duplicate = _fired(dup, metrics=m)
            if bad:
                duplicate["why"] = "; ".join(bad)
                if not dup:
                    duplicate["status"] = "not_assessed"
                duplicate["metrics"] = {"anchor": m, "cameras": nm}
            R["state_time_non_monotonic_or_duplicate"] = duplicate
            if nm and assessed:
                result = _fired(nev, metrics=nm)
                invalid = [x["why"] for x in nm.values() if x.get("status") == "not_assessed"]
                if invalid:
                    result["why"] = "; ".join(invalid)
                    if not nev:
                        result["status"] = "not_assessed"
                R["native_camera_timestamp_gap"] = result
            elif nm:
                R["native_camera_timestamp_gap"] = {**_na("; ".join(x["why"] for x in nm.values())), "metrics": nm}
            else:
                R["native_camera_timestamp_gap"] = _na("no camera has a clock of its own besides the main one")
        else:
            why = "the dataset has no capture clock, so frame times are the frame number divided by the frame rate"
            R["state_time_non_monotonic_or_duplicate"] = _na(why)
            R["state_timestamp_gap"] = _na(why)
            R["native_camera_timestamp_gap"] = _na(why)

    # ---- cameras (filtering.py:1543-1639), each on its own native frames
    anchor_pair: dict[str, np.ndarray] = {}
    anchor_pc: dict[str, np.ndarray] = {}
    cam_metrics = {}
    cam_err: dict[str, str] = {}      # camera -> the error its checks stopped with
    cam_unchecked: dict[str, set] = {}  # camera -> the video checks it crashed before finishing
    align_ev, dec_ev, cnt_ev = [], [], []
    exp_ev, low_ev, dup_ev, frz_ev = [], [], [], []
    lists = (align_ev, dec_ev, cnt_ev, exp_ev, low_ev, dup_ev, frz_ev)
    list_checks = VIDEO_CHECKS[1:]    # the check each evidence list is for, in the same order
    pixel_observed = {check: 0 for check in PIXEL_CHECKS}
    pixel_limited = {check: [] for check in PIXEL_CHECKS}
    motion_limited = []
    with _guard(R, ("missing_camera",)):
        R["missing_camera"] = _fired([] if cams else [_ev("the episode has no camera")])
    # of the video checks only the frozen picture needs the camera each actor is mounted on (to tell it from a still
    # scene), so failing to work it out errors that check alone (None)
    av_all = None
    if state_error is not None and extra["frozen_needs_motion"]:
        R["video_frozen_run"] = _errored(state_error)     # it needs the recorded motion here, which could not be read
    else:
        with _guard(R, ("video_frozen_run",)):
            av_all = actor_views(ep, names) if usable_state and extra["frozen_needs_motion"] else []

    def expected_motion(v: str, t0: float, t1: float) -> str:
        """Why camera v's picture must change between t0 and t1, from the recorded state, or "" when the
        recording gives no such reason (video-only rigs, or nothing recorded moving)."""
        if not usable_state or av_all is None or v not in comparison_cameras:
            return ""
        k = np.flatnonzero((ts >= t0) & (ts <= t1))
        if len(k) < 2:
            return ""
        a, b = int(k[0]), int(k[-1])
        mounted = [g for g, mv in enumerate(av_all) if mv == v]
        movers = mounted if mounted else (list(range(len(names))) if v == "exo" and rig == "teleop_arms" else [])
        s_raw = np.asarray(ep["state"], dtype=np.float64)
        for g in movers:
            o = 7 * g
            if kind == "ee_pose":
                # summed over the steps with a reading (a missing row never hides a frozen camera)
                path = float(np.nansum(np.linalg.norm(np.diff(s_raw[a:b + 1, o:o + 3], axis=0), axis=1)))
                turn = float(np.linalg.norm(global_[a:b, o + 3:o + 6], axis=1).sum())
                if path >= 0.02 or turn >= 0.1:
                    return (f"the recorded {names[g]} pose moves {path * 100:.0f} cm and turns "
                            f"{np.degrees(turn):.0f} deg")
            elif kind == "joints":
                jt = float(np.nansum(np.abs(np.diff(s_raw[a:b + 1, o:o + 6], axis=0))))
                if jt >= 0.1:
                    return f"the recorded {names[g]} arm joints move {np.degrees(jt):.0f} deg in total"
        return ""

    for v, c in cams.items():
        # one camera's checks: a crash costs only its own evidence, never the other cameras' (cam_err). done holds each
        # check whose work on this camera completed, so a crash after it leaves that check's finding (or its clear
        # result) standing for this camera
        done = set()
        try:
            if "failed" in c:
                raise c["failed"]        # its frames could not be read (_camera_or_failure)
            n = c["n"]
            km = ep["kmap"].get(v)
            if km is None and n != T and has_state:
                align_ev.append(_ev(f"camera {v} has {n} frames and the state {T}, with no time pairing between them",
                                    camera=v))
            elif km is not None and (len(km) < T or int(np.max(km[:T])) >= n or int(np.min(km[:T])) < 0):
                align_ev.append(_ev(f"camera {v}'s frame pairing covers {len(km)} of {T} state frames or points "
                                    f"outside its {n} frames", camera=v))
            done.add("camera_state_alignment_mismatch")
            if c["error"]:
                dec_ev.append(_ev(f"camera {v} fails to decode: {c['error']}", camera=v))
            done.add("video_decode_failure")
            if c["decoded"] != n:
                missing = np.flatnonzero(~np.isfinite(c["means"]))
                cnt_ev.append(_ev(f"camera {v}: {c['decoded']} of its {n} frames decode at their recorded times",
                                  _cam_t(ep, v, int(missing[0]), c["fps"]) if len(missing) else None, v))
            done.add("video_decode_frame_count_mismatch")
            means, stds, pair = c["means"], c["stds"], c["pair"]
            ok = np.isfinite(means)
            pok = np.isfinite(pair)
            for check in PIXEL_CHECKS:
                seen, expected = (int(ok.sum()), n) if check in PIXEL_CHECKS[:2] else (int(pok.sum()), max(n - 1, 0))
                if seen < expected or not seen:
                    pixel_limited[check].append(f"camera {v}: {seen} of {expected} usable "
                                                + ("frames" if check in PIXEL_CHECKS[:2] else "frame pairs"))
                if check == "video_frozen_run" and extra["frozen_needs_motion"] and v not in comparison_cameras:
                    pixel_limited[check].append(f"camera {v}: state and video timing cannot be compared")
                else:
                    pixel_observed[check] += seen > 0
            if v in comparison_cameras and int(pok.sum()) < max(n - 1, 0):
                motion_limited.append(f"camera {v}: {int(pok.sum())} of {max(n - 1, 0)} usable frame pairs")
            if ok.any():
                black = ok & (means < extra["black_mean"])
                white = ok & (means > extra["white_mean"])
                frac = float((black | white)[ok].mean())
                if frac > policy.maximum_extreme_exposure_fraction:
                    runs = _runs(black | white)
                    longest = max(runs, key=lambda r: r[1] - r[0])
                    exp_ev.append(_ev(f"{frac:.0%} of camera {v}'s frames are nearly black or white (mean grey under "
                                      f"{extra['black_mean']:g} or over {extra['white_mean']:g}); the rule is over "
                                      f"{policy.maximum_extreme_exposure_fraction:.0%}; longest run "
                                      f"{(longest[1] - longest[0] + 1) / c['fps']:.1f} s",
                                      _cam_t(ep, v, longest[0], c["fps"]), v))
                done.add("video_extreme_exposure")
                low = ok & (stds < extra["low_contrast_std"])
                lfrac = float(low[ok].mean())
                if lfrac > policy.maximum_low_contrast_fraction:
                    runs = _runs(low)
                    longest = max(runs, key=lambda r: r[1] - r[0])
                    low_ev.append(_ev(f"{lfrac:.0%} of camera {v}'s frames are nearly uniform (grey standard deviation "
                                      f"under {extra['low_contrast_std']:g}); the rule is over "
                                      f"{policy.maximum_low_contrast_fraction:.0%}",
                                      _cam_t(ep, v, longest[0], c["fps"]), v))
            done.update(("video_extreme_exposure", "video_low_contrast"))    # nothing to judge with no frame decoded
            dupm = pok & (pair < extra["duplicate_pair_mad"])
            if extra["duplicate_motion_mad"]:
                # count repeats only inside motion: a run of at most duplicate_max_run repeated pairs whose neighbouring
                # steps on both sides are real motion (at least duplicate_motion_mad grey levels). In a still scene a
                # repeated picture cannot be told from a live one: a Galaxea wrist camera on an idle arm alternates
                # 0.04 / 0.6 grey-level steps from encoding alone, a still RealOmin view shows 29 identical frames
                # between 2.2-level keyframe steps, and MolmoAct2's AV1 top views alternate 0.02 / 1.5 in slow motion.
                # A short repeat between two clearly moving steps is unambiguous.
                m = extra["duplicate_motion_mad"]
                counted = np.zeros_like(dupm)
                for a_, b_ in _runs(dupm):
                    if b_ - a_ + 1 <= extra["duplicate_max_run"] and a_ > 0 and b_ + 1 < len(pair) \
                            and pair[a_ - 1] >= m and pair[b_ + 1] >= m:
                        counted[a_:b_ + 1] = True
                base = counted | (pok & (pair >= m))
                dupm_count = counted
            else:
                base = pok
                dupm_count = dupm
            min_n = max(2, int(2 * c["fps"])) if extra["duplicate_motion_mad"] else 1   # at least 2 s of moving frames
            dfrac = float(dupm_count[base].mean()) if base.sum() >= min_n else (0.0 if pok.any() else None)
            if dfrac is not None and dfrac > policy.maximum_duplicate_pair_fraction:
                where = ("of its frame steps while it moves (a short repeat between two steps that each change by at "
                         f"least {extra['duplicate_motion_mad']:g} grey levels)" if extra["duplicate_motion_mad"]
                         else "of consecutive frames")
                dup_ev.append(_ev(f"camera {v} repeats the same picture on {dfrac:.0%} {where} (mean grey change under "
                                  f"{extra['duplicate_pair_mad']:g}); the rule is over "
                                  f"{policy.maximum_duplicate_pair_fraction:.0%}", camera=v))
            done.add("video_duplicate_frames")
            runs = _runs(dupm)
            ct = camera_times(ep, v)

            def span_s(r):
                # pairs a..b cover frames a..b+1: real clock span when there is one, else pairs / fps (upstream)
                if ct is not None and r[1] + 1 < len(ct):
                    return float(ct[r[1] + 1] - ct[r[0]])
                return (r[1] - r[0] + 1) / c["fps"]
            longest_run = max(runs, key=span_s) if runs else None
            longest_s = span_s(longest_run) if longest_run else 0.0
            frozen = [r for r in runs if span_s(r) > extra["frozen_run_s"]]
            if extra["frozen_needs_motion"]:
                # a still scene can decode to identical frames (a RealOmin view shows 29 identical frames between
                # keyframes), so a frozen picture is claimed only while the recording says the view must change
                kept = []
                for r in frozen:
                    why = expected_motion(v, _cam_t(ep, v, r[0], c["fps"]), _cam_t(ep, v, r[1] + 1, c["fps"]))
                    if why:
                        kept.append((r, why))
            else:
                kept = [(r, "") for r in frozen]
            if kept:
                r, why = max(kept, key=lambda x: span_s(x[0]))
                # a recorder that re-encodes a frozen camera changes the picture a little at each keyframe; one such
                # step (well under real motion) between two frozen runs does not end the freeze, so the run that fired
                # is reported from where the freeze starts to where it ends. Runs that fire on their own are the only
                # ones extended, so no new firing can come of it.
                m_ = extra["duplicate_motion_mad"] or float("inf")
                starts = {a_: b_ for a_, b_ in runs}
                ends = {b_: a_ for a_, b_ in runs}
                a0, b0, steps = r[0], r[1], 0
                while a0 - 2 in ends and np.isfinite(pair[a0 - 1]) and pair[a0 - 1] < m_:
                    a0, steps = ends[a0 - 2], steps + 1
                while b0 + 2 in starts and np.isfinite(pair[b0 + 1]) and pair[b0 + 1] < m_:
                    b0, steps = starts[b0 + 2], steps + 1
                r = (a0, b0)
                t0 = _cam_t(ep, v, r[0], c["fps"])
                but = (f", apart from {steps} single keyframe step{'s' if steps > 1 else ''} under {m_:g}"
                       if steps else "")
                frz_ev.append(_ev(f"camera {v} shows the same picture for {span_s(r):.1f} s from {t0:.1f} s (every "
                                  f"consecutive frame changes by under {extra['duplicate_pair_mad']:g} grey "
                                  f"levels{but})" + (f" while {why}" if why else "")
                                  + f"; the rule is over {extra['frozen_run_s']:g} s", t0, v))
            done.add("video_frozen_run")
            extreme = (means < extra["black_mean"]) | (means > extra["white_mean"])
            cam_metrics[v] = {"frames": n, "decoded": c["decoded"], "fps": round(c["fps"], 2),
                              "extreme_exposure_fraction": round(float(extreme[ok].mean()), 4) if ok.any() else None,
                              "low_contrast_fraction": (round(float((stds < extra["low_contrast_std"])[ok].mean()), 4)
                                                        if ok.any() else None),
                              "repeated_frame_fraction": round(dfrac, 4) if dfrac is not None else None,
                              "longest_still_run_s": round(longest_s, 2) if pok.any() else None,
                              "mean_grey_p01_p99": ([round(float(np.nanquantile(means, q)), 1) for q in (0.01, 0.99)]
                                                    if ok.any() else None)}
            ap = to_anchor(ep, v, pair.astype(np.float64), T)
            pc = to_anchor(ep, v, c["pchange"].astype(np.float64), T)
            if ap is not None and pc is not None and v in comparison_cameras \
                    and np.isfinite(ap).any() and np.isfinite(pc).any():
                anchor_pair[v] = ap
                anchor_pc[v] = np.concatenate([[0.0], pc])
        except Exception as e:  # noqa: BLE001 - recorded on the camera and on the checks that need it
            # a check whose work on this camera completed before the crash (done) stands, its defect or its clear
            # result; only the checks that did not finish on it were not run there (cam_unchecked)
            cam_unchecked[v] = set(list_checks) - done
            for d_ in (anchor_pair, anchor_pc):
                d_.pop(v, None)
            cam_err[v] = f"{type(e).__name__}: {e}"[:300]
            dec_err = c.get("error") if isinstance(c, dict) else None
            cam_metrics[v] = {"error": cam_err[v], **({"decode_error": dec_err} if dec_err else {})}
    with _guard(R, VIDEO_CHECKS):
        for chk, x in zip(list_checks, lists):
            if chk not in R:    # video_frozen_run is errored already when the actors' cameras were not worked out
                R[chk] = _fired(x)
    for check in PIXEL_CHECKS:
        r = R.get(check)
        if r and r["status"] in ("clear", "fired"):
            why = "; ".join(pixel_limited[check])
            if not pixel_observed[check] and not cam_err:
                R[check] = _na(f"No usable video observations for this check. {why}")
            elif why:
                R[check] = {**r, "why": why}

    # ---- motion (filtering.py:1654-1700, 1796-1861), end-effector pose only
    actor_metrics: dict[str, dict] = {}
    with _guard(R, MOTION_CHECKS):
        if not has_pose or not usable_state:
            for c in MOTION_CHECKS:
                R[c] = _na(no_pose_why if not has_pose else unusable_why)
        else:
            pm = up._motion_metrics(
                global_, avalid, ts_ns,
                static_translation_m=policy.static_translation_m, static_rotation_rad=policy.static_rotation_rad,
                static_gripper_delta=policy.static_gripper_delta, jump_relative_robust_z=policy.jump_relative_robust_z,
                minimum_jump_translation_m=policy.minimum_jump_translation_m,
                minimum_jump_rotation_rad=policy.minimum_jump_rotation_rad,
                maximum_jump_interval_ratio=policy.maximum_jump_interval_ratio,
                minimum_smoothness_translation_m=policy.minimum_smoothness_translation_m,
                minimum_smoothness_rotation_rad=policy.minimum_smoothness_rotation_rad,
                maximum_interleaved_hold_ratio=policy.maximum_interleaved_hold_ratio)
            hold_ev, jump_ev, speed_ev = [], [], []
            holds = 0
            for arm, am in pm["arms"].items():
                g = 0 if arm == "left" else 1
                if g >= len(names):
                    continue
                name = names[g]
                for sig in ("translation", "rotation"):
                    hi = am[f"{sig}_interleaved_hold_indices"]
                    holds += len(hi)
                    if hi:
                        hold_ev.append(_ev(f"{len(hi)} times the recorded {name} {sig} steps, nearly stops for one "
                                           f"frame (under 10% of the steps around it), then steps again in the same "
                                           f"direction (first at {_t(ts, hi[0])} s)", _t(ts, hi[0]), actor=name))
                    ji = am[f"{sig}_jump_return_indices"]
                    o = 7 * g + (0 if sig == "translation" else 3)
                    mag = np.linalg.norm(global_[:, o:o + 3], axis=1)
                    iso = extra["jump_isolation"]
                    if iso:
                        # a leap, not fast motion: both steps of the leap-and-return are at least `iso` times the steps
                        # just before and after it (the rule recorded_jumps uses; upstream has none and fires on the
                        # back-and-forth of a fast zipping motion in RealOmin)
                        ji = [i for i in ji if min(mag[i], mag[i + 1]) >= iso * max(
                            [mag[j] for j in (i - 1, i + 2) if 0 <= j < len(mag)] or [0.0])]
                    if ji:
                        i = ji[0]
                        size = (f"{mag[i] * 100:.1f} cm out and {mag[i + 1] * 100:.1f} cm back" if sig == "translation"
                                else f"{np.degrees(mag[i]):.0f} deg out and {np.degrees(mag[i + 1]):.0f} deg back")
                        least = (f"{policy.minimum_jump_translation_m * 100:g} cm" if sig == "translation"
                                 else f"{np.degrees(policy.minimum_jump_rotation_rad):.0f} deg")
                        jump_ev.append(_ev(f"the recorded {name} {sig} leaps {size} within two frames at "
                                           f"{_t(ts, i + 1)} s" + (f" ({len(ji)} such leaps)" if len(ji) > 1 else "")
                                           + f"; rule: both steps over {least} and 10 robust deviations above the "
                                             f"typical step, nearly cancelling"
                                           + (f", and at least {iso:g} x the steps around them" if iso else ""),
                                           _t(ts, i + 1), actor=name))
                dt_s = np.diff(ts_ns) / 1e9
                tr = np.linalg.norm(global_[:, 7 * g:7 * g + 3], axis=1) / np.maximum(dt_s, 1e-12)
                rr = np.linalg.norm(global_[:, 7 * g + 3:7 * g + 6], axis=1) / np.maximum(dt_s, 1e-12)
                ftr = np.flatnonzero(tr > policy.maximum_umi_translation_speed_m_s)
                frr = np.flatnonzero(rr > policy.maximum_umi_rotation_speed_rad_s)
                if len(ftr) or len(frr):
                    i = int(ftr[0]) if len(ftr) else int(frr[0])
                    n_fast = len(np.union1d(ftr, frr))
                    speed_ev.append(_ev(f"the {name} pose moves at up to {tr.max():.2f} m/s and {rr.max():.1f} rad/s "
                                        f"between two frames, over the limit of "
                                        f"{policy.maximum_umi_translation_speed_m_s:g} m/s or "
                                        f"{policy.maximum_umi_rotation_speed_rad_s:g} rad/s in {n_fast} "
                                        f"interval{'' if n_fast == 1 else 's'}", _t(ts, i + 1), actor=name))
                actor_metrics[name] = {"max_speed_m_s": round(float(tr.max()), 3) if len(tr) else None,
                                       "max_turn_rad_s": round(float(rr.max()), 2) if len(rr) else None}
            enough = holds >= policy.minimum_interleaved_hold_events
            R["action_smoothness_discontinuity"] = _fired(hold_ev if enough else [], metrics={"events": holds})
            R["jump_return_event"] = _fired(jump_ev)
            R["gross_umi_speed"] = _fired(speed_ev)
            sf = pm["static_fraction"]
            still = sf > policy.maximum_static_fraction
            R["over_95_percent_static"] = _fired([_ev(f"{sf:.1%} of frame steps move under "
                                                      f"{policy.static_translation_m * 1000:g} mm, "
                                                      f"{policy.static_rotation_rad * 1000:g} mrad and the gripper "
                                                      f"under "
                                                      f"{policy.static_gripper_delta:g}")] if still else [],
                                                 metrics={"static_fraction": round(sf, 4)})

            # largest action vs video (filtering.py:1161-1238), every camera on the anchor interval grid: on handheld
            # rigs each actor against its own mounted camera only (largest_action_own_camera)
            av = actor_views(ep, names)
            if extra["largest_action_own_camera"]:
                vc = {}
                for g, v in enumerate(av):
                    if v not in anchor_pair:
                        continue
                    part, _ = up._largest_action_video_checks(global_, avalid, {v: anchor_pair[v]}, policy)
                    arm = ("left", "right")[g]
                    vc.update({k: m for k, m in part.items() if k.startswith(arm + "_")})
            else:
                vc, _ = up._largest_action_video_checks(global_, avalid, anchor_pair, policy)
            ev = []
            for key, m in vc.items():
                if m.get("status") != "unsupported":
                    continue
                arm, sig = key.split("_", 1)
                g = 0 if arm == "left" else 1
                if g >= len(names):
                    continue
                i = m["action_index"]
                mag = m.get("magnitude_m", m.get("magnitude_rad"))
                unit_s = f"{mag * 100:.1f} cm" if sig == "translation" else f"{np.degrees(mag):.1f} deg"
                camtxt = ", ".join(f"{cv} {cm['mean_absolute_luma_difference']:.2f}" for cv, cm in m["cameras"].items())
                ev.append(_ev(f"the {names[g]} gripper's largest single-frame {sig} ({unit_s}, over 10 robust "
                              f"deviations above its typical step) shows no image change on any camera (mean grey "
                              f"change {camtxt}; "
                              f"rule: every camera in its lowest {policy.largest_action_visual_percentile:.0%}, under "
                              f"{policy.largest_action_visual_median_ratio:g} x its median and under "
                              f"{policy.largest_action_visual_absolute_difference:g})", _t(ts, i + 1), actor=names[g]))
            R["largest_action_not_in_video"] = _fired(ev)

            uv, _ = up._unexplained_visual_change_checks(global_, avalid, anchor_pair, policy)
            ev = []
            for cv, cm in uv["cameras"].items():
                if cm["flagged_count"]:
                    i = cm["flagged_indices"][0]
                    ev.append(_ev(f"camera {cv} changes by at least "
                                  f"{policy.minimum_unexplained_visual_difference:g} grey levels on "
                                  f"{cm['flagged_count']} frame steps while every gripper holds still",
                                  _t(ts, i + 1), cv))
            R["visual_change_unexplained_by_action"] = _fired(ev)

            # upstream reads these as lists (it tests them with `or []`)
            row = {"robot": {"actions_global": global_.tolist(), "action_valid_global": avalid.tolist()}}
            row.update(up.action_intensity_features(row))
            wrist = {slot: v for slot, v in zip(("left", "right"), av) if v is not None}
            for slot, v in wrist.items():
                row[f"{slot}_pixel_change_amount"] = anchor_pc.get(v)
            row["overhead_pixel_change_amount"] = anchor_pc.get("exo") if rig == "teleop_arms" else None
            corr, _ = up.visual_action_correlation_checks(row, policy)
            ev, cm_out = [], {}
            for slot, rec in corr.items():
                actor = dict(zip(("left", "right"), names)).get(slot, slot)
                cm_out[actor] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()
                                if k in ("status", "correlation", "r_squared", "pairs")}
                if rec.get("status") == "mismatch":
                    cam = wrist.get(slot, "exo")
                    ev.append(_ev(f"camera {cam}'s frame-to-frame change correlates {rec['correlation']:+.2f} "
                                  f"(r squared {rec['r_squared']:.3f}) with the recorded motion; the rule is r <= 0 "
                                  f"or r squared "
                                  f"under {policy.minimum_visual_action_r_squared:g}", camera=cam))
            R["pixel_action_corr_mismatch"] = _fired(ev, metrics=cm_out)

    if cam_err:
        # a camera whose checks crashed is left out of anchor_pair and anchor_pc, so a check that compares every camera
        # with the motion would judge the others alone; it says why instead
        err = "; ".join(f"camera {v}: {m}" for v, m in cam_err.items())
        for c in CAMERA_MOTION_CHECKS:
            if (R.get(c) or {}).get("status") in ("fired", "clear"):
                R[c] = {**_r("errored", f"the check stopped with an error ({err})"), "error": err}
        # a per-camera video check never reads clear for a camera nobody checked: one that did not fire is errored,
        # and one that fired on another camera stays fired and names the camera that was not checked
        for c in VIDEO_CHECKS:
            r = R.get(c) or {}
            err_c = "; ".join(f"camera {v}: {m}" for v, m in cam_err.items() if c in cam_unchecked[v])
            if not err_c or r.get("status") not in ("fired", "clear"):
                continue
            if r["status"] == "clear":
                R[c] = {**_r("errored", f"the check stopped with an error ({err_c})"), "error": err_c}
            else:
                R[c] = {**r, "why": f"it was not run on every camera, as the check stopped with an error ({err_c})"}
    if ctx.get("state_unaligned"):
        # the recorded state is not on these cameras' frames (the camera it was recorded on was taken out,
        # board/clips.py drop_cameras): every check that compares it with the video is not assessed; the checks on the
        # state alone stand
        why = f"The recorded state is not on these cameras' frames. {ctx['state_unaligned']}"
        for c in STATE_VS_VIDEO + (("video_frozen_run",) if extra["frozen_needs_motion"] else ()):
            R[c] = _na(why)
    if assumed_clocks or not state_window_ok:
        skipped = sorted(set(cams) - comparison_cameras)
        why = ("The recorded state does not match the anchor camera's frame window."
               if not state_window_ok else
               f"State and video motion were not compared for {', '.join(skipped)}: camera placement uses assumed presentation timing.")
        for c in CAMERA_MOTION_CHECKS + (("video_frozen_run",) if extra["frozen_needs_motion"] else ()):
            r = R.get(c)
            if r and r["status"] in ("clear", "fired"):
                if (c in CAMERA_MOTION_CHECKS and not anchor_pair) or (c == "video_frozen_run" and not comparison_cameras):
                    R[c] = _na(why)
                else:
                    R[c] = {**r, "why": "; ".join(x for x in (r.get("why"), why) if x)}
    if comparison_cameras and not anchor_pair:
        why = "No usable video frame pairs remain for state and video motion comparison."
        for c in CAMERA_MOTION_CHECKS:
            if (R.get(c) or {}).get("status") in ("clear", "fired"):
                R[c] = _na(why)
    elif motion_limited:
        why = "; ".join(motion_limited)
        for c in CAMERA_MOTION_CHECKS:
            r = R.get(c)
            if r and r["status"] in ("clear", "fired"):
                R[c] = {**r, "why": "; ".join(x for x in (r.get("why"), why) if x)}
    if state_error is not None:
        R.update({c: _errored(state_error) for c in STATE_CHECKS})
    return {"checks": R, "cameras": cam_metrics, "actors": actor_metrics,
            "episode": {"duration_s": round(duration, 2), "rig": rig, "state_kind": kind, "gripper_unit": unit,
                        "clock": ("assumed camera presentation" if ctx.get("camera_clock") else
                                  "capture times" if real_times else "frame_index / fps")}}


# ------------------------------------------------------------------------------------------ disposition
#
# What is done with each check when it fires. flag: shown as an issue, on the rigs in "applies" (on any other rig a
# firing is shown as a note). note: shown as a quiet fact. excluded: never an issue; a firing is shown as a note
# that says why it does not count (NOT_COUNTED). "reason" is the calibration finding behind the decision, from the
# check's firings on our frame-verified datasets.

ALL = ("teleop_arms", "handheld_gripper", "ego_head")
ARMS = ("teleop_arms", "handheld_gripper")
HANDHELD = ("handheld_gripper",)


def _d(disposition: str, reason: str, title: str = "", note: str = "", applies: tuple[str, ...] = ALL) -> dict:
    return {"disposition": disposition, "reason": reason, "title": title, "note": note, "applies": tuple(applies)}


DISPOSITION: dict[str, dict] = {
    # ---- structure: definitional defects; none fired on our frame-verified datasets
    "nonfinite_signal": _d(
        "flag", "a NaN or infinite recorded value is a broken recording by definition; fired on none",
        "The recorded state contains missing (NaN or infinite) values.", applies=ARMS),
    "missing_canonical_signal": _d(
        "flag", "no valid state value at all in an episode that declares a state is a broken recording; fired on none",
        "The episode declares a recorded state but contains no valid state values.", applies=ARMS),
    "invalid_state_shape": _d(
        "note", "fires on 2 Galaxea episodes whose state has 16 values per frame (a different arm variant): a layout "
                "our checks do not read, not a defect of the data; shown so the skipped state checks are explained",
        applies=ARMS),
    "missing_camera": _d(
        "flag", "an episode with no camera stream is broken by definition; fired on none",
        "The episode has no camera stream."),
    "camera_state_alignment_mismatch": _d(
        "flag", "a camera that covers different frames than the state, with no time pairing, cannot be matched to the "
                "recording; fired on none (the same fact as dataset_checks.camera_windows_match_state)",
        "A camera covers a different number of frames than the recorded state and cannot be matched to it."),
    "video_decode_failure": _d(
        "flag", "a camera file that fails to decode is broken by definition; fired on none (a missing file is an "
                "infrastructure fault and fails the episode instead)",
        "A camera video fails to decode."),
    "video_decode_frame_count_mismatch": _d(
        "flag", "frames that do not decode at their recorded times are missing video; fired on none",
        "Some camera frames are missing, because they do not decode at their recorded times."),
    "state_time_non_monotonic_or_duplicate": _d(
        "flag", "a capture clock that repeats or runs backwards is broken; fired on none of the datasets with capture "
                "clocks (ABC-130k, RealOmin; the published Gen-HumanEgo copies were extracted with nominal 30 fps "
                "times, so they were not checked on a capture clock)",
        "The capture clock repeats a time or runs backwards."),
    # ---- video on each camera's native frames
    "video_duplicate_frames": _d(
        "flag", "upstream (any pair under 0.25 grey levels on 64 px frames, over 20%) fires on quiet but live scenes "
                "(Galaxea 16 of 222, MolmoAct2 44 of 78 decoded, RealOmin 9). A still scene can decode to "
                "near-identical frames: MolmoAct2's AV1 top views alternate 0.02 / 1.5 grey-level steps in slow "
                "motion, a RealOmin view shows 29 identical frames between keyframes. So we count a repeat only as a "
                "run of at most 4 pairs under 0.1 between two frame steps of at least 3 grey levels (clear motion), "
                "over 20% of the camera's moving steps. It then fires only on ABC-130k cameras that deliver the same "
                "picture several times, stamped 0 ms apart (3 episodes, all checked on the frames: full-resolution "
                "difference 0.01 to 0.1 grey levels against 3 to 13 for real steps)",
        "A camera repeats the same picture on many of its frames while the scene is moving (over 20% of its moving "
        "frame steps)."),
    "video_frozen_run": _d(
        "flag", "at upstream's 0.25 it fires on 1 s pauses of a static view (ABC-130k 4, Egocentric-100K 2, live "
                "pictures); a still scene can even decode to identical frames (RealOmin: 29 identical frames between "
                "keyframes), so we require a change under 0.1 grey levels on every frame for over 1 s while the "
                "recorded state says the view must change (the camera's own gripper or arm moves, or any arm moves in "
                "front of a fixed top camera); the only firing is FastUMI 433's frozen left camera (confirmed camera "
                "fault, 16.8 s); never claimed on video-only rigs, where a still scene cannot be told from a frozen "
                "camera",
        "A camera shows the same frozen picture for over a second while the recording says it was moving.",
        applies=ARMS),
    "video_extreme_exposure": _d(
        "flag", "a camera nearly black or white for over a quarter of the episode; every firing was checked on the "
                "frames",
        "A camera is nearly black or nearly white for over a quarter of the episode."),
    "video_low_contrast": _d(
        "note", "a wrist camera pressed against cloth or a table, or a head camera facing a blank wall, is uniform "
                "without any fault; a fact, not a defect claim",
        note="Low contrast:"),
    # ---- clocks
    "state_timestamp_gap": _d(
        "flag", "upstream's rule alone (over 1.5 x the median interval) fires on 86% of ABC-130k, whose RealSense "
                "clocks stamp frames in bursts (half the intervals 1 us, the rest 30 to 150 ms), and on 4% of "
                "RealOmin for single dropped frames (50 to 100 ms); we add a 0.25 s floor, above the longest interval "
                "of any verified dataset (200 ms), so a firing is a recording that lost at least a quarter second; "
                "fired on none",
        "The recording's clock has a gap of at least a quarter second (frames lost)."),
    "native_camera_timestamp_gap": _d(
        "flag", "the same rule and 0.25 s floor on each non-anchor camera's own clock (upstream alone: 89% of "
                "ABC-130k, 1% of RealOmin, all stamping pattern or single dropped frames); fired on none",
        "A camera's clock has a gap of at least a quarter second (frames lost)."),
    "action_time_non_monotonic_or_duplicate": _d(
        "excluded", "action times are midpoints of the state times; checked there"),
    "episode_too_short": _d(
        "note", "policy, not a defect (legitimate short demonstrations exist); fired on none",
        note="Short episode:"),
    # ---- grippers (only where the unit is a verified 0-1 open fraction: MolmoAct2, ABC-130k)
    "normalized_gripper_out_of_range": _d(
        "flag", "a 0-1 open fraction outside -0.05..1.05 is a broken reading; fired on none of MolmoAct2's 1284 or "
                "ABC-130k's 183 episodes (readings stay in 0.001..1.000)",
        "The gripper reading goes outside its 0 to 1 range.", applies=ARMS),
    "gripper_action_integral_out_of_range": _d(
        "excluded", "for the original deltas this replays the recorded reading exactly, so it is the range check with "
                    "a 0.01 tolerance (a 1.02 reading at full open would fire without any fault); the smoothed replay "
                    "tests upstream's own smoothing; fired on none"),
    "gripper_never_acts": _d(
        "note", "a gripper that never moves is either unused (one-handed tasks) or not recording; which one needs the "
                "frames, and our gripper_channels check already reports exact flatness; fired on none",
        note="Gripper reading never changes:", applies=ARMS),
    "gripper_sensor_bug": _d(
        "flag", "two opposite gripper steps of half the range on consecutive frames cannot be a real jaw movement; "
                "fired on none of MolmoAct2 or ABC-130k",
        "The gripper reading jumps by half its range and back on consecutive frames (a sensor glitch).",
        applies=ARMS),
    # ---- end-effector motion (handheld grippers only; joint teleop has no forward kinematics here)
    "gross_umi_speed": _d(
        "excluded", "a single interval over 2 m/s or 8 rad/s: fires on 3% of FastUMI and includes real fast sweeps "
                    "(Clean_Desktop 001476 and 001452 on the frames: dustpan sweeps with motion blur); the real "
                    "glitches it finds are also found by jump_return_event and largest_action_not_in_video"),
    "jump_return_event": _d(
        "flag", "upstream computes it but does not count it against a stored episode; with our added isolation test "
                "(both leap steps at least 3 x the steps around them) it fires on 3 FastUMI episodes, each a recorded "
                "pose that leaps and returns while its camera stays smooth (3 of 3 on the frames); without it, it also "
                "fires on a fast back-and-forth zipping motion in RealOmin (zip_clothes 00041) and on a FastUMI leap "
                "inside fast motion",
        "The recorded gripper pose leaps away and back within two frames while its camera shows no such movement.",
        applies=HANDHELD),
    "largest_action_not_in_video": _d(
        "flag", "each gripper's largest single-frame step, when it is over 2 cm or 0.15 rad and 10 robust deviations "
                "above its typical step, against its own rigidly mounted camera (upstream requires every camera to "
                "be still, so the other hand's camera hides the glitch); every firing is a recorded leap the camera "
                "does not show (FastUMI 1 confirmed by recorded_jumps, RealOmin 2 of 2 on the frames)",
        "The gripper's recorded pose makes a large single-frame jump that its own camera does not show.",
        applies=HANDHELD),
    "visual_change_unexplained_by_action": _d(
        "excluded", "fires on 51% of RealOmin: its per-frame hold thresholds (0.5 mm, 5 mrad) count a slow handheld "
                    "drift as still, and a bright window then changes the picture by 8 grey levels "
                    "(unscrew_bottle_cap 00178 on the frames); on FastUMI all 10 firings were real frozen poses, "
                    "which pixel_action_corr_mismatch and largest_action_not_in_video report more directly"),
    "pixel_action_corr_mismatch": _d(
        "note", "absolute r squared under 0.04 between a camera's frame change and the recorded motion: on FastUMI "
                "most firings are real (frozen pose, crossed streams) but Open_Double_Door_Shoe_Cabinet 000379 is "
                "normal content with a weak correlation (0.13), so it is shown as a measured fact next to "
                "stream_pairing, not as a defect",
        note="Camera motion follows the recorded motion only weakly:", applies=HANDHELD),
    "action_smoothness_discontinuity": _d(
        "note", "fires on 9% of RealOmin, whose state is nearest-sampled onto the camera clock by our prepare step, so "
                "a pose stream that updates less often than the camera shows as one-frame stops; a true fact about "
                "the pose stream, not a defect claim; fired on no FastUMI episode",
        note="The recorded pose updates less often than the camera:", applies=HANDHELD),
    "over_95_percent_static": _d(
        "note", "policy, not a defect", note="Almost no recorded motion:", applies=HANDHELD),
    # ---- upstream reasons that test upstream's own processing or a layer we never build
    "invalid_action_shape": _d(
        "excluded", "actions are derived here from the states; their shape cannot be wrong"),
    "state_action_count_mismatch": _d(
        "excluded", "actions are derived here from the states; the counts cannot differ"),
    "invalid_rotation_matrix": _d(
        "excluded", "rotations are built here from roll/pitch/yaw: always proper, so this tests our conversion"),
    "nonprincipal_rotation_state": _d(
        "excluded", "rotation vectors from scipy are always within pi, so this cannot fire"),
    "se3_translation_round_trip_failure": _d(
        "excluded", "tests upstream's own action derivation, which we reuse unchanged"),
    "se3_rotation_round_trip_failure": _d(
        "excluded", "tests upstream's own action derivation, which we reuse unchanged"),
    "state_time_too_short": _d(
        "flag", "a recording with fewer than two frame times has no timeline; broken by definition; fired on none",
        "The recording has fewer than two frame times."),
    "action_time_too_short": _d(
        "excluded", "action times are midpoints of the state times; checked there"),
    "native_rate_qc_unavailable": _d(
        "excluded", "a process flag that fires on every episode without a native-rate row; we never build one"),
    "native_signal_checks": _d(
        "excluded", "the native_* reasons (native_invalid_rotation_matrix, native_nonprincipal_rotation_state, "
                    "native_normalized_gripper_out_of_range, native_state_time_* and native_camera_<name>_* clocks, "
                    "source_stream_<name>_non_monotonic_or_duplicate, native_signal_invalid, "
                    "native_stream_shape_mismatch, native_stream_nonfinite) need a separate native-rate telemetry "
                    "layer our sidecars do not have; the camera clocks are checked on their own frames by "
                    "native_camera_timestamp_gap"),
    "processing_failure": _d(
        "excluded", "an episode the stage cannot process is reported by the command line per episode, not as a data "
                    "issue"),
}


# ------------------------------------------------------------------------------------------ output

# The plain name of each check (after the headings of public-dataset-adapter's docs/data-quality-findings-and-
# examples.md where it names one), which names the problem a firing finds, and the group it is shown under.
NAMES: dict[str, tuple[str, str]] = {
    "missing_canonical_signal": ("Recorded state missing", "Structure"),
    "invalid_state_shape": ("State in a layout these checks cannot parse", "Structure"),
    "invalid_action_shape": ("Malformed action arrays", "Structure"),
    "state_action_count_mismatch": ("State and action counts differ", "Structure"),
    "nonfinite_signal": ("Missing (NaN) values in the state", "Structure"),
    "episode_too_short": ("Too-short episode", "Structure"),
    "invalid_rotation_matrix": ("Invalid rotation", "Structure"),
    "nonprincipal_rotation_state": ("Rotation outside its principal range", "Structure"),
    "se3_translation_round_trip_failure": ("Actions do not replay to the recorded position", "Structure"),
    "se3_rotation_round_trip_failure": ("Actions do not replay to the recorded rotation", "Structure"),
    "state_time_too_short": ("Too few frame times", "Clocks"),
    "action_time_too_short": ("Too few action times", "Clocks"),
    "state_time_non_monotonic_or_duplicate": ("Clock repeats or runs backwards", "Clocks"),
    "action_time_non_monotonic_or_duplicate": ("Action clock repeats or runs backwards", "Clocks"),
    "state_timestamp_gap": ("Gap in the recording's clock", "Clocks"),
    "native_camera_timestamp_gap": ("Gap in a camera's clock", "Clocks"),
    "missing_camera": ("Missing camera", "Video"),
    "camera_state_alignment_mismatch": ("Camera not aligned with the state", "Video"),
    "video_decode_failure": ("Video that cannot be decoded", "Video"),
    "video_decode_frame_count_mismatch": ("Frames missing from a video", "Video"),
    "video_extreme_exposure": ("Extreme exposure", "Video"),
    "video_low_contrast": ("Low contrast", "Video"),
    "video_duplicate_frames": ("Duplicate adjacent frames", "Video"),
    "video_frozen_run": ("Frozen camera", "Video"),
    "normalized_gripper_out_of_range": ("Gripper aperture outside its range", "Grippers"),
    "gripper_action_integral_out_of_range": ("Replayed gripper leaves its range", "Grippers"),
    "gripper_never_acts": ("A gripper never acts", "Grippers"),
    "gripper_sensor_bug": ("Gripper sensor glitch", "Grippers"),
    "jump_return_event": ("Recorded pose jumps away and back", "Motion"),
    "largest_action_not_in_video": ("Large action with no camera motion", "Motion"),
    "visual_change_unexplained_by_action": ("Large visual change while the arms hold", "Motion"),
    "pixel_action_corr_mismatch": ("Weak camera and motion correlation", "Motion"),
    "action_smoothness_discontinuity": ("Recorded pose updates less often than the camera", "Motion"),
    "gross_umi_speed": ("Implausible gripper speed", "Motion"),
    "over_95_percent_static": ("Mostly static episode", "Motion"),
    "native_rate_qc_unavailable": ("No native-rate data", "Pipeline"),
    "native_signal_checks": ("Native-rate signal faults", "Pipeline"),
    "processing_failure": ("Processing failure", "Pipeline"),
}

# Why a check that fired is shown as a note rather than an issue: its firings did not all hold up on the frames
# of our verified datasets (the disposition reasons above hold the numbers).
NOT_COUNTED = {
    "gross_umi_speed": "It is shown as a note, not an issue, because it also fires on real fast sweeps.",
    "visual_change_unexplained_by_action": "It is shown as a note, not an issue, because slow handheld drift looks like "
                                           "holding still to this check, so it also fires on sound recordings.",
    "gripper_action_integral_out_of_range": "It is shown as a note, not an issue, because it repeats the aperture range "
                                            "check with a looser tolerance.",
}
SETUP_WORDS = {"teleop_arms": "teleoperated-arm", "handheld_gripper": "UMI", "ego_head": "human ego"}
# where a stored note's reason starts, in its earlier and its current wording (refresh_notes cuts there)
WHY_STARTS = ("Not counted as an issue", "It is shown as a note")


def note_why(check: str, rig: str) -> str:
    """Why a firing of this check on this setup is shown as a note, or "" for a check that is a note by design."""
    d = DISPOSITION.get(check) or _d("excluded", "unknown check")
    if check in NOT_COUNTED:
        return NOT_COUNTED[check]
    if d["disposition"] == "note":
        return ""
    return ("It is shown as a note, not an issue, because its firings were not confirmed on the frames of our "
            f"verified {SETUP_WORDS.get(rig, rig.replace('_', ' '))} datasets.")


def note_row_why(check: str, rig: str, unchecked: str | None) -> str:
    """A note row's reason: why it is a note (note_why) and the cameras it was not run on (unchecked), each a
    sentence, so the second never runs on lowercase after the first's full stop."""
    return " ".join(_sentence(x) for x in (note_why(check, rig), unchecked) if x)


def _sentence(t: str) -> str:
    """A note's lead or evidence as one sentence: capitalized, its trailing colon or full stop made one full stop."""
    t = t.strip().rstrip(":.").strip()
    return t[:1].upper() + t[1:] + "." if t else ""


def _lead(check: str) -> str:
    d = DISPOSITION.get(check) or _d("excluded", "unknown check")
    return d["note"] or d["title"] or NAMES.get(check, (check.replace("_", " ").capitalize(), ""))[0]


def note_record(check: str, evidence: str, rig: str) -> dict:
    """One firing shown as a note: its evidence as a sentence, and the text a download carries on its own (the
    check's lead, the evidence and why it is a note)."""
    ev, why = _sentence(evidence), note_why(check, rig)
    return {"check": check, "evidence": ev, "text": " ".join(x for x in (_sentence(_lead(check)), ev, why) if x)}


# the pose-speed evidence as it was stored before it was reworded, and its count before it took the right plural
OLD_SPEED = re.compile(r" \(rule: over ([\d.]+) m/s or ([\d.]+) rad/s in any one interval; (\d+) intervals\)")


def refresh_notes(cq: dict) -> dict:
    """A stored result with each note's reason worded as note_why words it now, so a reworded reason reaches the
    board on the next build without rerunning the checks."""
    rig = ((cq.get("metrics") or {}).get("episode") or {}).get("rig")
    if not rig:
        return cq
    notes = []
    for n in cq.get("notes") or []:
        if not (isinstance(n, dict) and isinstance(n.get("text"), str) and n.get("check")):
            notes.append(n)
            continue
        if "evidence" in n:
            ev = n["evidence"]
        else:
            # an earlier record: its text is the lead, the evidence and the reason run together
            cuts = [n["text"].find(s) for s in WHY_STARTS if s in n["text"]]
            ev = (n["text"][:min(cuts)] if cuts else n["text"]).strip()
            lead = _lead(n["check"]).strip()
            ev = ev[len(lead):] if ev.startswith(lead) else ev
        ev = OLD_SPEED.sub(lambda m: f", over the limit of {m[1]} m/s or {m[2]} rad/s in {m[3]} "
                                     f"interval{'' if m[3] == '1' else 's'}", ev)
        notes.append({**n, **note_record(n["check"], ev, rig)})
    # a note row's reason is note_why followed by the cameras it was not run on (unchecked), kept apart for this
    rows = [{**r, "why": note_row_why(r["check"], rig, r.get("unchecked"))}
            if isinstance(r, dict) and r.get("shown_as") == "note" and r.get("why") else r
            for r in cq.get("checks") or []]
    return {**cq, **({"notes": notes} if "notes" in cq else {}), **({"checks": rows} if "checks" in cq else {})}


def format_result(a: dict) -> dict:
    """assess() output -> the compact context["capture_qc"] record. Every check appears in "checks" with its status
    (fired, clear, not_applicable, or errored with why it stopped) and how a firing is shown (issue or note); the
    rule that a check is an issue only where its firings held up on verified datasets decides the label, never
    whether the check is listed."""
    rig = a["episode"].get("rig")
    flags, notes, not_assessed, listing = [], [], {}, []
    for check in CHECKS + [c for c in a["checks"] if c not in CHECKS]:
        r = a["checks"].get(check) or _na("not computed for this episode")
        d = DISPOSITION.get(check) or _d("excluded", "unknown check")
        name, group = NAMES.get(check, (check.replace("_", " ").capitalize(), "Other"))
        row = {"check": check, "name": name, "group": group}
        if r["status"] == "not_assessed":
            not_assessed[check] = r["why"]
            listing.append({**row, "status": "not_applicable", "why": r["why"]})
            continue
        if r["status"] == "errored":
            listing.append({**row, "status": "errored", "why": r["why"]})
            continue
        if r["status"] != "fired":
            listing.append({**row, "status": "clear", **({"why": r["why"]} if r.get("why") else {})})
            continue
        counted = d["disposition"] == "flag" and rig in d["applies"]
        if counted:
            for e in r["events"]:
                flags.append({"check": check, "title": d["title"], "t_s": e["t_s"], "camera": e["camera"],
                              "actor": e["actor"], "evidence": e["evidence"]})
            listing.append({**row, "status": "fired", "shown_as": "issue", "events": len(r["events"]),
                            **({"why": r["why"]} if r.get("why") else {})})
            continue
        why = note_row_why(check, rig, r.get("why"))
        for e in r["events"]:
            notes.append(note_record(check, e["evidence"], rig))
        listing.append({**row, "status": "fired", "shown_as": "note", "events": len(r["events"]),
                        **({"why": why} if why else {}), **({"unchecked": r["why"]} if r.get("why") else {})})
    metrics = {"cameras": a["cameras"], "actors": a["actors"], "episode": a["episode"]}
    clock_metrics = (a["checks"].get("state_time_non_monotonic_or_duplicate") or {}).get("metrics") or {}
    if "anchor" in clock_metrics:
        # A partial clock assessment retains both the unusable clock and each other clock's available evidence.
        def availability(m):
            return {"status": "assessed", **m}
        metrics["clocks"] = {"anchor": availability(clock_metrics["anchor"]),
                             "cameras": {v: availability(m) for v, m in clock_metrics["cameras"].items()}}
    return {"source": SOURCE, "version": VERSION, "flags": flags, "notes": notes, "not_assessed": not_assessed,
            "checks": listing, "metrics": metrics}


def errored_record(e: Exception) -> dict:
    """The context["capture_qc"] record of an episode whose checks could not run at all: every check errored, with
    the error."""
    return format_result({"checks": {c: _errored(e) for c in CHECKS}, "cameras": {}, "actors": {}, "episode": {}})


def run_episode(ep_dir: Path | str) -> dict:
    """Decode, assess and format one episode: its context["capture_qc"] record. An episode that cannot be read or
    decoded at all still gets one, with every check errored and the error, never no record."""
    try:
        return format_result(assess(extract(Path(ep_dir))))
    except Exception as e:  # noqa: BLE001 - recorded on every check, the episode keeps a record
        return errored_record(e)


# ------------------------------------------------------------------------------------------ CLI

# the exit status when some episode's checks could not run because its worker stopped: every episode still has a
# record (those with every check errored), so a caller goes on and reads the errored checks from the records
EXIT_FAILED = 3


def _one(d: str) -> tuple[str, dict, str | None, float]:
    """(episode, its record, the error when its worker stopped, seconds). A worker that stops outside run_episode still
    gives the episode a record, every check errored with the error, so it is never left with none."""
    t0 = time.time()
    try:
        return d, run_episode(Path(d)), None, time.time() - t0
    except Exception as e:  # reported per episode and recorded on every check, never silently skipped
        return d, errored_record(e), f"{type(e).__name__}: {e}"[:300], time.time() - t0


def _pool(eps: list[str], jobs: int, record) -> list[str]:
    """Run each episode's _one on one pool of jobs worker processes and record each answer as it finishes. Returns the
    episodes the pool never finished: a worker that dies (killed by a signal, out of memory) breaks the whole pool, so
    the episode it was on and every one still waiting come back unfinished, not only the one that killed it."""
    lost = []
    with ProcessPoolExecutor(jobs) as ex:
        futures = {ex.submit(_one, d): d for d in eps}
        for fut in as_completed(futures):
            try:
                res = fut.result()
            except Exception:  # noqa: BLE001 - the pool broke: retried alone (_alone)
                lost.append(futures[fut])
                continue
            record(*res)
    return sorted(lost)


def _alone(d: str) -> tuple[str, dict, str | None, float]:
    """_one for one episode in a worker process of its own, so a worker that dies costs that episode only: then it gets
    a record with every check errored and the error."""
    try:
        with ProcessPoolExecutor(1) as ex:
            return ex.submit(_one, d).result()
    except Exception as e:  # noqa: BLE001 - its worker died again
        return d, errored_record(e), f"{type(e).__name__}: {e}"[:300], 0.0


def needs_check(ctx: dict) -> bool:
    """Whether an episode is checked without --force: it has no record, or its record is one written because its
    worker stopped (worker_stopped), which says nothing about the episode, so a rerun checks it again."""
    cq = ctx.get("capture_qc")
    return not isinstance(cq, dict) or bool(cq.get("worker_stopped"))


def main() -> int:
    """Check every episode and write its record into its context.json. Episodes run on a pool of --jobs worker
    processes; the ones a worker's death left unfinished run again, each in a process of its own, so only an episode
    whose worker dies on that retry too is recorded with every check errored, marked worker_stopped. Exits EXIT_FAILED
    when some episode's checks could not run, else 0."""
    ap = argparse.ArgumentParser(prog="python -m checks.capture_qc", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", type=Path, metavar="EPISODES", help="folders of prepared episode_* folders")
    ap.add_argument("--jobs", type=int, default=2, help="episodes decoded in parallel")
    ap.add_argument("--force", action="store_true", help="recompute episodes that already have a result")
    args = ap.parse_args()
    eps = []
    for root in args.roots:
        for d in sorted(root.glob("episode_*")):
            if not (d / "context.json").exists():
                continue
            if args.force or needs_check(json.loads((d / "context.json").read_text())):
                eps.append(str(d))
    print(f"episodes to check: {len(eps)}", flush=True)
    done = flagged = failed = 0

    def record(d: str, r: dict, err: str | None, secs: float) -> None:
        nonlocal done, flagged, failed
        done += 1
        if err:
            failed += 1
            r = {**r, "worker_stopped": True}
            print(f"FAILED {Path(d).name}: {err}", flush=True)
        p = Path(d) / "context.json"
        ctx = json.loads(p.read_text())
        ctx["capture_qc"] = r
        write_atomic(p, ctx, indent=1)
        if r["flags"]:
            flagged += 1
            print(f"FLAGGED {Path(d).name} " + "; ".join(f"{f['check']} {f['t_s']}" for f in r["flags"]), flush=True)
        if done % 25 == 0:
            print(f"progress {done}/{len(eps)} flagged={flagged} failed={failed} last={secs:.1f}s", flush=True)

    lost = _pool(eps, args.jobs, record)
    if lost:
        print(f"retrying {len(lost)} episodes a worker's death left unfinished, each in a process of its own",
              flush=True)
        with ThreadPoolExecutor(args.jobs) as ex:
            for res in ex.map(_alone, lost):
                record(*res)
    print(f"done {done} flagged={flagged} failed={failed}", flush=True)
    return EXIT_FAILED if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
