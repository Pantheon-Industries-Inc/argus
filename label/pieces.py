"""Label a long recording in pieces and stitch the labels back into one timeline.

A recording longer than its part limit (piece_max: the part_max_s that label/policy.py resolve gives, 450s at a
rig's default sampling and at most 330s at denser sampling) is one episode on the board, with one clip and one
timeline, but the model labels it in parts: the recording is cut at frame boundaries within the request limit, preferring the least motion near each evenly
spaced target (from recorded state when available and anchor-camera image change otherwise), each
part is labelled as its own request, and the parts' labels are shifted back onto the recording's clock and
joined. Nothing about the recording's episode boundaries is invented: an unsplit recording stays one episode,
and only actual model-emitted activities become tasks on its one timeline. Request-local
goals, outcomes and progress remain attributed assessments, not recording-level judgments.

Each part is told it is one part of a continuous recording cut at a still moment, so activity carrying across
a cut is expected; an issue a part reports about being cut off at one of our cuts (a truncated start or end)
describes our cut, not the recording, and is set aside with that reason. The recording's task text describes
the whole recording, so a part is shown it as context, not as its goal.

    python -m label.pieces plan EP_DIR            # print the cuts

The path for your own data (python -m review) cuts every episode that needs it, labels the parts and stitches
them, and Data Review runs the same functions on uploads. No episode on the published board is longer than 450s,
so none of them is cut.
"""
from __future__ import annotations

import copy
import errno
import json
import math
import os
import re
import shutil
from pathlib import Path

import numpy as np

from board.label_projection import LABEL_SCHEMA, recording_completion
from label.atomic import write_atomic
from label import policy
from label.task_claims import claims as task_claims
from prepare.frame_rate import episode_seconds

# Not read here: the part limit is piece_max (policy part_max_s). Kept because Data Review's vendor check
# (upload/vendor_repo.py NEEDS in the Data repo) requires this name in this file.
PIECE_MAX_S = {"teleop_arms": 450.0, "handheld_gripper": 450.0, "ego_head": 450.0}
SEARCH = 0.3                # a cut is sought within +-30% of a part's length around its target
SMOOTH_S = 2.0              # motion is averaged over 2 s so a cut lands in a still stretch, not a still frame
CUT_GUARD_S = 10.0          # maximum distance for an explicitly described issue at its owning internal boundary
CUT_TAGS = ("truncat", "incomplete", "cut_off", "starts_mid", "ends_mid", "mid_task")


def piece_max(ctx: dict) -> float:
    return policy.effective(ctx)["part_max_s"]


def duration(ctx: dict, ep_dir: Path | None = None) -> float:
    return float(episode_seconds(ctx, ep_dir=ep_dir))


def needs_pieces(ep_dir: Path) -> bool:
    """Use the same surviving anchor span as request validation, without loading sensor arrays."""
    from label import episode as me
    from prepare.formats import clock_context
    from prepare.camera_clock import load_times
    ep_dir = Path(ep_dir)
    ctx = clock_context(json.loads((ep_dir / "context.json").read_text()))
    source_path = ep_dir / 'sources.json'
    if not source_path.is_file():
        return duration(ctx, ep_dir) > piece_max(ctx)
    sources = json.loads(source_path.read_text())
    times = load_times(ep_dir, ctx) if ctx.get('real_times') else None
    if times is not None:
        zero = float(ctx.get('clock_zero_s') or 0.0)
        times = {key: value if key.endswith('_pts') else value - zero for key, value in times.items()}
    ep = {'dir': ep_dir, 'context': ctx, 'sources': sources, 'times': times}
    return me.request_duration(ep) > piece_max(ctx)


def motion(ep: dict) -> tuple[np.ndarray, np.ndarray]:
    """(times of anchor frames, motion per frame). From the recorded state when the episode has one (joint or
    pose speed summed over actors), else the anchor camera's grey-level change between consecutive frames."""
    from label import episode as me
    n = int(ep["sources"][me.anchor(ep)]["n_frames"])
    t = np.array([me.frame_time(ep, k) for k in range(n)], dtype=np.float64)
    if not ep['context'].get('state_unaligned') and me.state_kind(ep) != "none" and len(ep["state"]) == n:
        s = np.asarray(ep["state"], dtype=np.float64)
        d = np.abs(np.diff(s, axis=0)).sum(axis=1)
        return t, np.concatenate([[0.0], d])
    return t, video_motion(ep, n)


def video_motion(ep: dict, n: int) -> np.ndarray:
    """The anchor camera's grey-level change between consecutive frames, per frame. A stretch of the video that does
    not decode has no change measured (0, as a still stretch); decoding goes on after it, seeking further ahead each
    time the decoder fails again, so one damaged stretch never stops the recording from being cut."""
    import av
    from label import episode as me
    from label import frames as mf
    v = me.anchor(ep)
    s = ep["sources"][v]
    pts = ep["times"].get(f"{v}_pts") if ep.get("times") is not None else None
    out = np.zeros(n)
    with av.open(str(s["packed"])) as c:
        st = c.streams.video[0]
        st.thread_type = "AUTO"
        if pts is not None:
            targets = [int(p) for p in pts]
        else:
            b0, step = mf.base_frame(s["base_s"], me.ep_fps(ep)), mf.frame_pts_step(st.time_base, me.ep_fps(ep))
            targets = [(b0 + k) * step for k in range(n)]
        index = {p: k for k, p in enumerate(targets)}
        start, seen, skip = 0, -1, 1
        while start < n:
            prev = None
            try:
                c.seek(targets[start], stream=st, backward=True, any_frame=False)
                for fr in c.decode(st):
                    k = index.get(fr.pts)
                    if k is None:
                        if fr.pts is not None and fr.pts > targets[-1]:
                            break
                        continue
                    g = fr.reformat(width=64, height=max(2, int(round(fr.height * 64 / fr.width))),
                                    format="gray").to_ndarray().astype(np.int16)
                    if prev is not None and k > seen:
                        out[k] = np.abs(g - prev).mean()
                    prev, seen = g, max(seen, k)
                    if k == n - 1:
                        break
                break
            except av.error.FFmpegError as e:
                if isinstance(e, OSError):      # a file that is gone or cannot be opened is our fault, never data's
                    raise
                start, skip = max(start, seen + 1) + skip, skip * 2
    return out


def choose_cuts(t: np.ndarray, m: np.ndarray, max_s: float, end_s: float | None = None) -> list[dict]:
    """Choose still frame boundaries while reserving enough capacity for every remaining part, including the tail."""
    t, m = np.asarray(t, dtype=float), np.asarray(m, dtype=float)
    if not len(t) or len(t) != len(m) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Request pieces need finite, increasing presentation times")
    step = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30
    end = float(t[-1] + step if end_s is None else end_s)
    if not math.isfinite(max_s) or max_s <= 0 or not math.isfinite(end) or end <= t[-1]:
        raise ValueError("Request pieces need a positive duration and part limit")
    boundaries = np.append(t, end)
    # The furthest reachable boundary minimizes the number of remaining parts. Keep this constraint during the
    # low-motion search, so choosing an early quiet frame cannot leave an oversized last window.
    reach = np.searchsorted(boundaries, t + max_s + 1e-8, side="right") - 1
    remaining = np.zeros(len(t) + 1, dtype=int)
    for k in range(len(t) - 1, -1, -1):
        if reach[k] <= k:
            raise ValueError("No frame boundary can keep this recording within the request part limit")
        remaining[k] = 1 + remaining[reach[k]]
    parts = int(remaining[0])
    if parts == 1:
        return []
    length = (end - t[0]) / parts
    w = min(len(m), max(1, int(round(SMOOTH_S / max(step, 1e-3)))))
    sm = np.convolve(np.where(np.isfinite(m), m, np.inf), np.ones(w) / w, mode="same")
    finite = sm[np.isfinite(sm)]
    still_level = float(np.percentile(finite, 25)) if len(finite) else -math.inf
    cuts, prev = [], 0
    for i in range(1, parts):
        target = float(t[0]) + i * length
        candidates = np.arange(prev + 1, min(int(reach[prev]), len(t) - 1) + 1)
        candidates = candidates[remaining[candidates] <= parts - i]
        if not len(candidates):
            raise ValueError("No frame boundary can keep this recording within the request part limit")
        near = candidates[np.abs(t[candidates] - target) <= SEARCH * length]
        candidates = near if len(near) else candidates
        # Equal-motion frames nearest the balanced target avoid arbitrary early cuts in a quiet recording.
        order = np.lexsort((np.abs(t[candidates] - target), sm[candidates]))
        k = int(candidates[order[0]])
        cuts.append({"frame": k, "t_s": round(float(t[k]), 3), "still": bool(sm[k] <= still_level),
                     "motion": round(float(sm[k]), 4) if math.isfinite(sm[k]) else None,
                     "target_s": round(target, 1)})
        prev = k
    return cuts


def fmt_clock(s: float) -> str:
    s = max(0, int(round(s)))
    return f"{s // 60}:{s % 60:02d}"


def _reuse_array(source: Path, target: Path) -> None:
    """Share immutable prepared arrays locally, copying only across filesystem boundaries."""
    try:
        os.link(source.resolve(), target)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        shutil.copy2(source, target)


def write_pieces(ep_dir: Path, pieces_root: Path) -> list[Path]:
    """Write one sidecar folder per part under pieces_root (named <episode>__pNN) and record the cuts in the
    episode's context (context["pieces"]). Returns the part folders."""
    from label import episode as me
    from prepare.formats import CLOCK_TIME_KEYS, shift_context_times
    ep_dir = Path(ep_dir)
    stored = json.loads((ep_dir / "context.json").read_text())
    ep = me.load(ep_dir)
    ctx = ep["context"]
    read = task_claims(ctx)
    t, m = motion(ep)
    n = len(t)
    # Cut the same actual source span that request admission validates, including the last frame interval.
    total = me.request_duration(ep)
    unaligned = bool(ctx.get('state_unaligned'))
    max_s = piece_max(ctx)
    cuts = choose_cuts(t, m, max_s, total)
    bounds = [0] + [c["frame"] for c in cuts] + [n]
    a = me.anchor(ep)
    fps = me.ep_fps(ep)
    src = ep["sources"]
    z = np.load(ep_dir / "state.npz") if (ep_dir / "state.npz").exists() else None
    zs = np.load(ep_dir / "signals.npz") if ctx.get("signals") else None
    tz = dict(np.load(ep_dir / ctx["real_times"])) if ctx.get("real_times") else None
    presentation = dict(np.load(ep_dir / ctx["presentation_times"])) if ctx.get("presentation_times") else None
    touch = {}
    if zs is not None and not unaligned:
        # touch is judged once, on the whole recording, and each part's signal entries carry the verdict
        # (label/episode.py touch_verdicts): a part that falls inside a long press has no rest of its own, so its slice
        # alone would not read as touch and the part would lose the contact the recording shows
        from label import signals as sg
        from label.dictionary_context import field_interpretation
        touch = {s["name"]: bool(sg.is_touch(s["name"], np.asarray(zs[s["key"]][:n], dtype=np.float64), s.get("rest"),
                                             s.get("swing"), role=field_interpretation(ctx, s["name"]).get("role")))
                 for s in ctx["signals"] if s["key"] in zs.files}
    # the recording's contacts, found once on the whole recording when its context has none (prepared before contacts
    # were measured, or measuring them failed), so no part's labelling finds contacts on its own slice; the recording's
    # context.json is not given them. They are found with the verdicts just made, so no signal is judged twice
    contacts = None
    if not unaligned and ("contacts" in ctx or ctx.get("signals")):
        from label import contacts as lc
        contacts = lc.of_episode(ep, frozenset(nm for nm, v in touch.items() if v))
    out = []
    count = len(bounds) - 1
    for i in range(count):
        k0, k1 = bounds[i], bounds[i + 1]
        name = f"{ep_dir.name}__p{i + 1:02d}"
        d = pieces_root / name
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        t0, t1 = float(t[k0]), (float(t[k1]) if k1 < n else total)
        new_src, new_times, new_presentation = {}, {}, {}
        for v, s in src.items():
            s2 = {kk: vv for kk, vv in s.items() if kk != "kmap"}
            km = ep["kmap"].get(v)
            if v == a:
                j0, j1 = k0, k1
            elif km is None:
                # a camera not paired by time shares the anchor's frame index up to its own last frame, so a part past
                # that has none of it (label/episode.py _decode_view)
                own = int(s["n_frames"])
                j0, j1 = min(k0, own), min(k1, own)
            else:
                j0, j1 = int(km[k0]), int(km[k1 - 1]) + 1
                np.save(d / f"kmap_{v}.npy", (np.asarray(km[k0:k1]) - j0).astype(np.int32))
                s2["kmap"] = f"kmap_{v}.npy"
            s2["n_frames"] = int(j1 - j0)
            if s.get('last_frame_interval_s') is not None and tz is not None and v in tz and j1 < len(tz[v]) and j1 > j0:
                s2['last_frame_interval_s'] = float(tz[v][j1] - tz[v][j1 - 1])
            if tz is not None and f"{v}_pts" in tz:
                new_times[f"{v}_pts"] = tz[f"{v}_pts"][j0:j1]
            else:
                s2["base_s"] = round(float(s["base_s"]) + j0 / fps, 9)
            if tz is not None and v in tz:
                origin = float(presentation.get(a, tz[a])[k0]) if presentation else float(tz[a][k0])
                new_times[v] = tz[v][j0:j1] - origin
                if presentation and v in presentation:
                    new_presentation[v] = presentation[v][j0:j1] - origin
            new_src[v] = s2
        c2 = {kk: vv for kk, vv in ctx.items() if kk not in (
            "stream_pairing", "recorded_jumps", "gripper_channels", "capture_qc", "stream_checks", "pieces",
            "real_times", "timebase_neighbour_lag_frames",
            "clock_zero_s")}  # a part's times start at its own first frame, never on the recording's clock zero
        held = {v: r for v, r in ((v, me.runs_within(r, k0, k1))
                                  for v, r in (ctx.get("placeholder_frames") or {}).items()) if r}
        c2.pop("placeholder_frames", None)
        if held:
            c2["placeholder_frames"] = held       # on the part's own frames
        c2.update(episode_id=name, n_state_frames=int(k1 - k0), duration_s=round(t1 - t0, 3),
                  piece={"of": ep_dir.name, "index": i + 1, "count": count, "t0_s": round(t0, 3), "t1_s": round(t1, 3), "max_s": max_s})
        if ctx.get('recorded_mcap_fields') or ctx.get('recorded_table_fields') or ctx.get('recorded_sensor_fields'):
            c2['retained_evidence_root'] = str(Path(ctx.get('retained_evidence_root') or ep_dir).resolve())
        if ctx.get("recorded_camera_ns"):
            c2["recorded_camera_ns"] = str((ep_dir / ctx["recorded_camera_ns"]).resolve())
        if ctx.get("recorded_container_times"):
            c2["recorded_container_times"] = {**ctx["recorded_container_times"],
                "file": str((ep_dir / ctx["recorded_container_times"]["file"]).resolve())}
        if ctx.get("unshown_cameras"):
            from prepare.camera_clock import unshown_span
            c2["unshown_cameras"] = unshown_span(ep_dir, ctx["unshown_cameras"], t0, t1,
                                                float(ctx.get("clock_zero_s") or 0.0), shift=True)
        if unaligned:
            # Contacts on the lost anchor's clock cannot be rebased onto this part's camera clock.
            c2.pop('contacts', None)
        c2 = shift_context_times(c2, -t0)
        parent_notes = []
        if t0:
            for issue in c2.get("reader_issues") or []:
                if (isinstance(issue, dict) and isinstance(issue.get("what"), str) and issue["what"]
                        and any(isinstance(issue.get(key), (int, float)) and not isinstance(issue.get(key), bool)
                                for key in CLOCK_TIME_KEYS["reader_issues"])):
                    # The typed markers move to the part clock. Copied prose remains the recording's evidence.
                    issue["what"] = "Full recording clock note. " + issue["what"]
                    parent_notes.append(issue["what"])
        interval = (f"{fmt_clock(t0)} to {fmt_clock(t1)} on the full recording clock" if float(t[0]) else
                    f"{fmt_clock(t0)} to {fmt_clock(t1)} of it")
        note = (f"this clip is part {i + 1} of {count} of one continuous {fmt_clock(total - float(t[0]))} recording, from "
                f"{interval}. The labelling pipeline cut the recording into parts at "
                "frame boundaries, preferring moments of little motion, to label it; activity across a cut is expected, and a part "
                "that starts or ends in the middle of an activity is how we cut it, not a truncated or cut-off "
                "recording.")
        if read["instruction"]:
            # the part is checked against the recording's instruction, on what the part shows
            note += (" The instruction given with this part describes the whole recording, not this part: check this "
                     "part against it only on what this part shows. Asked work that falls outside this part is not a "
                     "failure, a partial outcome or an instruction mismatch here, and when this part ends before the "
                     "asked end state could be reached, its outcome is unclear.")
        if ctx.get("collection_note"):
            note += " " + ctx["collection_note"].strip()
        if parent_notes:
            note += " " + " ".join(parent_notes)
        c2["collection_note"] = note
        if new_times:
            np.savez(d / "times.npz", **new_times)
            c2["real_times"] = "times.npz"
        if new_presentation:
            np.savez(d / ctx["presentation_times"], **new_presentation)
        if contacts is not None:
            # the recording's contacts that overlap the part, on its clock and clipped to it, keeping their ids so the
            # parts' answers join back into one list
            c2["contacts"] = [{**c, "start_s": round(max(c["start_s"], t0) - t0, 3),
                               "end_s": round(min(c["end_s"], t1) - t0, 3),
                               "peak_s": round(min(max(c["peak_s"], t0), t1) - t0, 3),
                               "from_start": c["from_start"] or c["start_s"] < t0,
                               "to_end": c["to_end"] or c["end_s"] > t1,
                               "dips_s": [round(x - t0, 3) for x in c.get("dips_s") or [] if t0 <= x <= t1]}
                              for c in contacts if c["end_s"] >= t0 and c["start_s"] < t1]
        if (ep_dir / "depth.json").exists():
            # each camera's depth frames for the part's anchor frames; the depth files and their times are the
            # recording's own pts; their scalar capture times move with the part's colour capture times
            dj = json.loads((ep_dir / "depth.json").read_text())
            for v, e in dj.items():
                np.save(d / e["kmap"], np.asarray(np.load(ep_dir / e["kmap"])[k0:k1]))
            write_atomic(d / "depth.json", dj, indent=1)
            origin = float(t[k0]) + float(ctx.get("clock_zero_s") or 0.0)
            if (ep_dir / "depth_times.npz").exists():
                with np.load(ep_dir / "depth_times.npz") as dt:
                    np.savez(d / "depth_times.npz", **{key: dt[key] if key.endswith("_pts") else dt[key] - origin
                                                      for key in dt.files})
            if ctx.get("depth_presentation_times"):
                with np.load(ep_dir / ctx["depth_presentation_times"]) as dt:
                    np.savez(d / ctx["depth_presentation_times"], **{key: dt[key] - origin for key in dt.files})
        if z is not None:
            if unaligned:
                # Original source rows remain untimed evidence; camera indices cannot select their rows.
                _reuse_array(ep_dir / 'state.npz', d / 'state.npz')
            else:
                np.savez(d / "state.npz", **{kk: z[kk][k0:k1] for kk in z.files})
        if zs is not None:
            if unaligned:
                _reuse_array(ep_dir / 'signals.npz', d / 'signals.npz')
            else:
                np.savez(d / "signals.npz", **{kk: zs[kk][k0:k1] for kk in zs.files})
            # copies, so the recording's own context.json keeps its entries as prepare wrote them
            c2["signals"] = [{**s, "touch": touch[s["name"]],
                              **({"touch_role": "touch"} if sg.touch_permission(
                                  s["name"], field_interpretation(ctx, s["name"]).get("role")) is True else {})}
                             if s["name"] in touch else s for s in ctx["signals"]]
        # the recording's task texts as label/task_claims.py reads them: a text held over the whole recording is the
        # instruction (when it has none) or the dataset's task for the whole episode in every part, never a step
        if read["instruction_from"] == "claim":
            c2["instruction"] = read["instruction"]
        if read["whole"]:
            c2["dataset_task_claims"] = read["whole"]
        if ctx.get("annotation_subtasks"):
            # Keep the publisher's full claim, even when its interval crosses a request boundary.
            # Its shifted interval may start before zero or end after this request. The separate
            # visible window describes what this request can check, never a new publisher deadline. Each step keeps
            # its number in the whole recording (step), the number a task judged against it names.
            subs = [(x, me.number(x.get("t0")),
                     me.number(x["t1"]) if x.get("t1") is not None else me.number(x.get("t0")))
                    for x in read["steps"]]
            c2["annotation_subtasks"] = [
                {**x, 'source_annotation': {k: copy.deepcopy(v) for k, v in x.items() if k not in ('step', 'joined_rows')}, "t0": None,
                 **({"t1": None} if "t1" in x else {})} if a is None else
                {**x, 'source_annotation': {k: copy.deepcopy(v) for k, v in x.items() if k not in ('step', 'joined_rows')}, "t0": round(a - t0, 3),
                 **({"t1": round(b - t0, 3) if b is not None else None} if "t1" in x else {}),
                 'request_window': {'source_start_s': a, 'source_end_s': b,
                    'visible_start_s': round(max(a, t0) - t0, 3),
                    'visible_end_s': round(min(a if b is None else b, t1) - t0, 3),
                    'continues_before': a < t0, 'continues_after': b is not None and b > t1}}
                for x, a, b in subs if a is None or ((a if b is None else b) >= t0 and a < t1)]
        write_atomic(d / "sources.json", new_src, indent=1)
        write_atomic(d / "context.json", c2, indent=1, default=str)
        (d / "instruction.txt").write_text("\n")
        out.append(d)
    stored["pieces"] = {"max_s": piece_max(ctx), "cuts": cuts, "parts": [p.name for p in out]}
    write_atomic(ep_dir / "context.json", stored, indent=1, default=str)
    return out


def write_units(job: Path, eps: Path, *, refresh_unpaid_interpretation=False) -> dict:
    """The requests to send for a folder of prepared episodes: short episodes as they are, long recordings as their
    parts. job/units holds one link per request so one labelling run covers both, and job/pieces the parts.
    Returns {long episode: [its part folders' names]}."""
    units = Path(job) / "units"
    if units.exists():
        shutil.rmtree(units)
    units.mkdir()
    proot = Path(job) / "pieces"
    long_eps = {}
    for d in sorted(Path(eps).glob("episode_*")):
        if needs_pieces(d):
            parts = _write_or_reuse(d, proot, refresh_unpaid_interpretation=refresh_unpaid_interpretation)
            long_eps[d.name] = [p.name for p in parts]
            for p in parts:
                (units / p.name).symlink_to(p.resolve())
        else:
            (units / d.name).symlink_to(d.resolve())
    return long_eps


def _write_or_reuse(d: Path, proot: Path, *, refresh_unpaid_interpretation=False) -> list[Path]:
    from label.evidence_access import same_source_proof, source_proof
    ctx = json.loads((d / "context.json").read_text())
    names = (ctx.get("pieces") or {}).get("parts") or []
    proof_path = proot / f".{d.name}.source_proof.json"
    if names and not proof_path.exists() and not any((proot / n).exists() for n in names):
        # A different job may have cut this source before. This job has no saved parts or paid replies to reuse.
        parts = write_pieces(d, proot)
        ctx = json.loads((d / "context.json").read_text())
        write_atomic(proof_path, source_proof(ctx, d))
        return parts
    if names:
        try:
            saved = json.loads(proof_path.read_text())
        except (OSError, ValueError) as error:
            raise RuntimeError(f"{d.name}: existing pieces have no verifiable source proof; use an explicit fresh job") \
                from error
        if not same_source_proof(saved, ctx, d):
            from label.harness import billing_record, has_dispatch_receipt
            from label.evidence_access import equivalent_source_proof
            current = source_proof(ctx, d)
            unchanged_sources = equivalent_source_proof(
                [row for row in saved if row['name'] != 'prepared:context.json'],
                [row for row in current if row['name'] != 'prepared:context.json'])
            out = proot.parent / 'run' / 'out'
            receipts = [billing_record(out, name + '.json') for name in names]
            if (refresh_unpaid_interpretation and unchanged_sources
                    and not any(receipt and has_dispatch_receipt(receipt[1]) for receipt in receipts)):
                # Review verified the preparation before applying dictionary overlays.
                # No request used these parts, so rebuild their derived interpretation.
                parts = write_pieces(d, proot)
                ctx = json.loads((d / 'context.json').read_text())
                write_atomic(proof_path, source_proof(ctx, d))
                return parts
            raise RuntimeError(f"{d.name}: prepared source changed since pieces were cut; use an explicit fresh job")
        if not all((proot / n / "context.json").exists() for n in names):
            raise RuntimeError(f"{d.name}: a saved piece is missing; use an explicit fresh job")
        for name in names:
            if needs_pieces(proot / name):
                raise RuntimeError(f"{name}: saved part exceeds the request duration limit; use an explicit fresh job")
        return [proot / n for n in names]
    parts = write_pieces(d, proot)
    ctx = json.loads((d / "context.json").read_text())
    write_atomic(proof_path, source_proof(ctx, d))
    return parts


def _read(p: Path) -> tuple[dict | None, str | None]:
    """(the output file's JSON, None), or (None, why) when it does not read."""
    try:
        r = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        return None, f"its output file does not read ({type(e).__name__}: {e})"[:300]
    return (r, None) if isinstance(r, dict) else (None, "its output file is not a JSON object")


def _part_reply(src: Path, name: str) -> tuple[dict | None, str | None]:
    """A part's reply from a run's out/ folder: (its result, None) when it is a complete annotation of the part's own
    request (label/harness.py annotation_problems) and its labels can be read (their fields of the wrong type left
    out, typed_labels), else (what came back, why it gave no labels): not parsing, lacking work the part's request
    asked for, cut off at the output limit (the harness's failed_<part>.json), an output file that does not read, or
    no reply at all, with why when the harness recorded it (noreply_<part>.json)."""
    from label.harness import typed_labels, annotation_problems, annotation_record, reply_truncated
    try:
        selected = annotation_record(src, name + '.json')
    except (OSError, ValueError, TypeError, AttributeError):
        present = [src / (prefix + name + '.json') for prefix in ('', 'failed_', 'noreply_')
                   if (src / (prefix + name + '.json')).exists()]
        if len(present) == 1:
            _, error = _read(present[0])
            if error:
                return None, error
        return None, 'saved billing receipts could not be verified'
    if selected is None:
        return None, 'the model gave no reply'
    _, record = selected
    if reply_truncated(record):
        return record, "the model's reply was cut off at the output limit"
    # each part is judged against its own request, as any unit is (label/completeness.py)
    problems = annotation_problems(record)
    if not problems:
        return {**record, 'labels': typed_labels(copy.deepcopy(record['labels']))}, None
    labels = record.get('labels')
    if record.get('parse_ok') and not isinstance(labels, dict):
        return record, 'its labels are not a JSON object'
    if record.get('no_reply'):
        return record, f"the model gave no reply ({record['no_reply']})"
    if record.get('parse_ok') or isinstance(labels, dict) and labels.get('_incomplete'):
        return record, "the model's reply is incomplete: " + '; '.join(problems)
    # the parser's own error says why, such as a JSON reply with no annotation in it, which the gap would otherwise
    # describe as JSON that did not read
    error = labels.get('_parse_error') if isinstance(labels, dict) else None
    return record, "the model's reply did not parse" + (f" ({error})" if isinstance(error, str) and error.strip() else "")


def _raw_head(r: dict | None) -> str:
    """The start of a part's reply that gave no labels, as the board keeps one (board/to_board.py RAW_HEAD)."""
    lab = (r or {}).get("labels") if isinstance((r or {}).get("labels"), dict) else {}
    raw = (lab.get("_raw") if lab.get("_raw") is not None else (r or {}).get("content_tail")
           or (r or {}).get("response_text") or "")
    return str(raw)[:3000]


def _part_dispatch(record: dict | None) -> dict:
    """Only dispatch states established by the part's own durable receipt."""
    from label.harness import client_timeout
    record = record or {}
    outcome = record.get('final_dispatch_outcome')
    reserve = record.get('final_reserved_usd')
    usage = record.get('usage') if isinstance(record.get('usage'), dict) else {}
    if outcome == 'not dispatched' and (type(reserve) not in (int, float) or reserve != 0
            or record.get('parse_ok') is not False or usage.get('est_cost_usd') not in (None, 0)
            or record.get('finish_reason') == 'length' or _raw_head(record)):
        outcome = None
    return {**({'final_dispatch_outcome': outcome}
               if outcome in ('claimed', 'unverified', 'not dispatched') else {}),
            **({'finish_reason': 'length'} if record.get('finish_reason') == 'length' else {}),
            **({'no_reply_kind': 'spend_cap'} if outcome == 'not dispatched'
               and record.get('no_reply_kind') == 'spend_cap'
               and record.get('parse_ok') is False and record.get('final_reserved_usd') == 0 else {}),
            **({'no_reply_kind': 'client_timeout'} if outcome in ('claimed', 'unverified')
               and client_timeout(record) else {})}


def _carry_part_billing(result: dict, parts: list) -> None:
    """Carry cumulative invoices separately from each part's current usage."""
    from label.harness import billing_view, _history_cost, auxiliary_cost
    if any(record and '_billing_receipt' in record for _, record in parts):
        # Annotation authority may predate a local refusal. Preserve the complete
        # financial ledgers as invoices instead of counting the old reply twice.
        final, auxiliary = [], []
        for context, annotation in parts:
            if not annotation:
                continue
            record = billing_view(annotation)
            part = (context.get('piece') or {}).get('index')
            final.extend({**copy.deepcopy(invoice), 'part': part} for invoice in record.get('final_cost_history', []))
            if record.get('final_dispatch_outcome') != 'not dispatched' and not record.get('dry_run'):
                _history_cost(record)
                final.append({**{key: copy.deepcopy(record[key]) for key in (
                    'usage', 'final_dispatch_outcome', 'final_reserved_usd', 'final_attempt_index',
                    'model', 'response', 'response_text') if key in record}, 'part': part})
            auxiliary.extend({**copy.deepcopy(invoice), 'part': part}
                             for invoice in record.get('auxiliary_cost_history', []))
            for kind, value in (('route', (record.get('config') or {}).get('resolution_route') or {}),
                                ('inspection', record.get('evidence_inspection') or {})):
                cost = auxiliary_cost(value, routing=kind == 'route')
                if cost:
                    auxiliary.append({**copy.deepcopy(value), 'kind': kind, 'part': part,
                                      'cost_usd': cost, 'previously_billed_usd': 0.0,
                                      'rounds_cost_offset_usd': float(value.get('rounds_cost_offset_usd') or 0)
                                          + max(0.0, float(value.get('cost_usd', 0)) - cost)})
                    auxiliary[-1].pop('previously_billed', None)
        result['final_cost_history'], result['auxiliary_cost_history'] = final, auxiliary
        result['usage']['est_cost_usd'] = 0.0
        for value in ((result.get('config') or {}).get('resolution_route') or {},
                      result.get('evidence_inspection') or {}):
            value.pop('previously_billed', None)
            value['previously_billed_usd'] = value.get('cost_usd', 0.0)
        return
    final, auxiliary, routes = [], [], []
    for context, record in parts:
        if not record:
            continue
        part = (context.get('piece') or {}).get('index')
        final.extend({**copy.deepcopy(invoice), 'part': part} for invoice in record.get('final_cost_history', []))
        auxiliary.extend({**copy.deepcopy(invoice), 'part': part} for invoice in record.get('auxiliary_cost_history', []))
        if record.get('final_dispatch_outcome') in ('claimed', 'unverified'):
            final.append({key: copy.deepcopy(record[key]) for key in (
                'model', 'generation_id', 'usage', 'final_dispatch_outcome', 'final_reserved_usd',
                'response_text', 'response') if key in record} | {'part': part})
        routes.append((record.get('config') or {}).get('resolution_route') or {})
    result['final_cost_history'], result['auxiliary_cost_history'] = final, auxiliary
    route = result['config']['resolution_route']
    route.pop('previously_billed', None)
    route['previously_billed_usd'] = math.fsum(float(item.get('cost_usd') or 0) if item.get('previously_billed')
        else float(item.get('previously_billed_usd') or 0) for item in routes)
    if any(item.get('cost_is_conservative_estimate') for item in routes):
        route['cost_is_conservative_estimate'] = True


def stitch_run(job: Path, eps: Path, long_eps: dict, out: Path) -> dict:
    """out/: the short episodes' replies from job/run/out as they came, and one stitched result per long recording.

    A recording is stitched from the parts that parsed; a part that did not parse, was cut off or never came is a gap,
    named in the result's stitched["missing"] with its span and why (the board flags it at that span, board/build.py
    reader_issues), and the recording is listed under "incomplete". A recording none of whose parts parsed is still
    written, as a reply that gave no labels, so the board shows its footage, checks and sensors and says why; it is
    listed under "unlabelled"."""
    src = Path(job) / "run" / "out"
    short_outputs = [p for p in [*src.glob("episode_*.json"), *src.glob("failed_episode_*.json"),
                                  *src.glob("noreply_episode_*.json")] if "__p" not in p.stem]
    # Upload resumes can skip the harness when every saved reply parsed. Verify
    # the active reply for each unit before any result reaches a new export.
    from label.harness import saved_input_problem
    units = {p.stem.removeprefix('failed_').removeprefix('noreply_'):
             Path(eps) / p.stem.removeprefix('failed_').removeprefix('noreply_') for p in short_outputs}
    units.update({name: Path(job) / 'pieces' / name for parts in long_eps.values() for name in parts})
    records = {}
    for name, directory in sorted(units.items()):
        record, _ = _part_reply(src, name)
        records[name] = record
        problem = saved_input_problem(record, directory) if record else None
        if problem:
            raise RuntimeError(f'{name}: {problem}; use an explicit fresh job to label changed inputs')
    for parent, parts in long_eps.items():
        proof_path = Path(job) / 'pieces' / f'.{parent}.source_proof.json'
        if not proof_path.exists() and not any((records.get(name) or {}).get('input_identity') for name in parts):
            continue
        from label.evidence_access import same_source_proof
        try:
            proof = json.loads(proof_path.read_text())
            directory = Path(eps) / parent
            context = json.loads((directory / 'context.json').read_text())
            current = same_source_proof(proof, context, directory)
        except (OSError, ValueError, KeyError, TypeError):
            current = False
        if not current:
            raise RuntimeError(f'{parent}: saved pieces have changed or missing input; '
                               'use an explicit fresh job to label changed inputs')
    out.mkdir(parents=True, exist_ok=True)
    res = {"stitched": 0, "incomplete": [], "unlabelled": []}
    # A short episode's files are published as the run holds them: each is copied, and one the run no longer holds
    # is removed, so a copy left from an earlier run never stands beside the label that replaced it. Readers choose
    # the active record from the same files in either folder (annotation_record), and a newer receipt beside an
    # older label carries billing that choosing one file would lose.
    held = {p.name for p in short_outputs}
    for name in sorted({p.stem.removeprefix('failed_').removeprefix('noreply_') for p in short_outputs}):
        for prefix in ('', 'failed_', 'noreply_'):
            target = out / f'{prefix}{name}.json'
            if target.name in held:
                shutil.copy(src / target.name, target)
            else:
                target.unlink(missing_ok=True)
    for ep, parts in long_eps.items():
        got, missing, failed = [], [], []
        for n in parts:
            pc = json.loads((Path(job) / "pieces" / n / "context.json").read_text())
            r, why = _part_reply(src, n)
            if why is None:
                got.append((pc, r))
                continue
            piece = pc["piece"]
            # with the start of what came back, so the board can show it (board/to_board.py label_failed)
            missing.append({"part": piece["index"], "t0_s": piece["t0_s"], "t1_s": piece["t1_s"], "why": why,
                            **_part_dispatch(r),
                            **({"raw_head": _raw_head(r)} if _raw_head(r) else {})})
            failed.append((pc, r))
        if got:
            result = stitch(Path(eps) / ep, got, missing)
            from label.evidence_access import merge as merge_inspection
            inspection = merge_inspection(got + [(pc, r) for pc, r in failed if r], Path(eps) / ep)
            if inspection['parts']:
                result['evidence_inspection'] = inspection
            jaw_evidence = merge_jaws(Path(eps) / ep, got + [(pc, r) for pc, r in failed if r], missing)
            if jaw_evidence:
                result['jaw_evidence'] = jaw_evidence
            for pc, r in failed:
                if not r:
                    continue
                u = r.get('usage') or {}
                for key in ('prompt_tokens', 'completion_tokens', 'reasoning_tokens', 'cached_tokens',
                            'cache_write_tokens', 'latency_s', 'est_cost_usd'):
                    total = result['usage'][key] + float(u.get(key) or 0)
                    result['usage'][key] = total if key == 'est_cost_usd' else round(total, 4)
                part_record = result['config']['pieces'][int(pc['piece']['index']) - 1]
                part_record.update(
                    episode_dir=r.get('episode_dir'), example_dir=r.get('example_dir'),
                    cost_usd=u.get('est_cost_usd'),
                    annotation_policy=r.get('annotation_policy'),
                    usage=u, provider=r.get('provider'), generation_id=r.get('generation_id'),
                    finish_reason=r.get('finish_reason'), resolution_route=(r.get('config') or {}).get('resolution_route'),
                    config=copy.deepcopy(r.get('config') or {}))
                part_record.update(_part_dispatch(r))
            _combine_request_config(result['config'])
            routes = [((r or {}).get('config') or {}).get('resolution_route') or {}
                      for _, r in sorted(got + failed, key=lambda item: item[0]['piece']['index'])]
            result['config']['resolution_route']['parts'] = routes
            result['config']['resolution_route']['cost_usd'] = math.fsum(
                float(route.get('cost_usd') or 0) for route in routes)
            result['decode_failed'] += [x for _, r in failed for x in (r or {}).get('decode_failed') or []]
            _carry_part_billing(result, got + failed)
            write_atomic(out / f"{ep}.json", result, indent=None)
            res["stitched"] += 1
            if missing:
                res["incomplete"].append(ep)
            continue
        write_atomic(out / f"{ep}.json", unlabelled(Path(eps) / ep, failed, missing), indent=None)
        res["unlabelled"].append(ep)
    return res


def unlabelled(ep_dir: Path, failed: list[tuple[dict, dict | None]], missing: list[dict]) -> dict:
    """The result of a long recording none of whose parts gave labels: no labels, and stitched["missing"] with each
    part's span and why (board/to_board.py label_failed reads it as no_part, and the board says why for each part),
    with the first part's reply that came back, every part's undecodable stretches and cost, and the parts it was cut
    into."""
    first = next((r for _, r in failed if r), {}) or {}
    asked = {r.get('label_schema') for _, r in failed if r}
    raw = (first.get("labels") or {}).get("_raw") if isinstance(first.get("labels"), dict) else None
    raw = raw if raw is not None else first.get("content_tail") or ""
    whys = "; ".join(f"part {g['part']}: {g['why']}" for g in missing)
    usage = {key: (math.fsum(float(((r or {}).get('usage') or {}).get(key) or 0) for _, r in failed)
                   if key == 'est_cost_usd' else
                   round(sum(float(((r or {}).get('usage') or {}).get(key) or 0) for _, r in failed), 4))
             for key in ('prompt_tokens', 'completion_tokens', 'reasoning_tokens', 'cached_tokens',
                         'cache_write_tokens', 'latency_s', 'est_cost_usd')}
    count = len(missing)
    from label.evidence_access import merge as merge_inspection
    inspection = merge_inspection([(pc, r) for pc, r in failed if r], ep_dir)
    from label.sensor_evidence import merge as merge_sensor
    ctx = json.loads((Path(ep_dir) / 'context.json').read_text())
    sensor = merge_sensor([(pc, r) for pc, r in failed if r], ctx, ep_dir=ep_dir)
    jaw_evidence = merge_jaws(ep_dir, [(pc, r) for pc, r in failed if r], missing)
    selected = policy.effective(ctx)
    cfg = dict(first.get('config') or {})
    routes = [((r or {}).get('config') or {}).get('resolution_route') or {} for _, r in failed]
    cfg['resolution_route'] = {'parts': routes,
                               'cost_usd': math.fsum(float(route.get('cost_usd') or 0) for route in routes)}
    cfg['pieces'] = [{**pc.get('piece', {}), 'episode_dir': (r or {}).get('episode_dir'),
                      'example_dir': (r or {}).get('example_dir'),
                      **_part_dispatch(r),
                      'annotation_policy': (r or {}).get('annotation_policy'),
                      'usage': (r or {}).get('usage') or {}, 'provider': (r or {}).get('provider'),
                      'generation_id': (r or {}).get('generation_id'), 'finish_reason': (r or {}).get('finish_reason'),
                      'resolution_route': ((r or {}).get('config') or {}).get('resolution_route'),
                      'config': copy.deepcopy((r or {}).get('config') or {})}
                     for pc, r in failed]
    _combine_request_config(cfg)
    result = {"episode_dir": str(ep_dir), "model": first.get("model"), "reasoning_effort": first.get("reasoning_effort"),
            'annotation_policy': selected,
            'prompt_mode': ('session' if selected.get('task_mode') == 'session' and not asked & {LABEL_SCHEMA}
                            else 'given' if task_claims(ctx)['instruction'] else 'inferred'),
            **({'label_schema': LABEL_SCHEMA} if asked == {LABEL_SCHEMA} else {}),
            **({'jaw_evidence': jaw_evidence} if jaw_evidence else {}),
            **({'evidence_inspection': inspection} if inspection['parts'] else {}),
            **({'sensor_evidence': sensor} if sensor['sensors'] else {}),
            "config": cfg, "parse_ok": False, "no_part": True, "usage": usage,
            "labels": {"_raw": raw, "_parse_error": f"no part of the recording returned labels that parse ({whys})"},
            "stitched": {"parts": count, "cuts_s": [g["t0_s"] for g in missing[1:]], "missing": missing},
            "decode_failed": [x for _, r in failed for x in (r or {}).get("decode_failed") or []]}
    _carry_part_billing(result, failed)
    return result


# ---------------------------------------------------------------- stitching

def _combine_request_config(cfg: dict) -> None:
    """Aggregate available request receipts; unknown counts stay unknown, not zero.

    Per-part configs retain camera/layout differences and the original local clock.
    Top-level sampled/detail times use the recording clock, including failed replies
    whose request receipt is available. They describe supplied input, not label coverage.
    """
    parts = cfg['pieces']
    configs = [part.get('config') or {} for part in parts]
    for field in ('n_frames_sent', 'n_image_parts'):
        values = [config.get(field) for config in configs]
        cfg[field] = (sum(values) if all(isinstance(value, int) and not isinstance(value, bool)
                                       and value >= 0 for value in values) else None)
    for field in ('timesteps_s', 'contact_detail_s'):
        cfg[field] = [round(float(t) + float(part['t0_s']), 3)
                      for part, config in zip(parts, configs) for t in config.get(field) or []]
    cfg['n_timesteps'] = (len(cfg['timesteps_s']) if all('timesteps_s' in config for config in configs) else None)
    cfg['request_config_scope'] = 'available_part_records'
    from label.grids import joined_receipt
    receipt = joined_receipt(parts)
    cfg.pop('request_images', None)
    if receipt is not None:
        cfg['request_images'] = receipt

T_KEYS = ("t_s", "start_s", "end_s", "completed_at_s", "goal_reached_at_s", "undone_at_s", "failure_t_s",
          "recovered_at_s")
# a contact's strips and the answer that counts frames in each (label/episode.py contacts_block, checks/contacts.py)
STRIP_FIELDS = {"begin": "first_touch_frame", "end": "last_touch_frame"}


def _shift(x, dt: float):
    """Every time field of a label structure shifted by dt seconds, one written as text that reads as a number
    (label/episode.py number) as well; one that reads as none is left as it is."""
    from label.episode import number
    if isinstance(x, list):
        return [_shift(v, dt) for v in x]
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k in T_KEYS and number(v) is not None:
                out[k] = round(number(v) + dt, 3)
            else:
                out[k] = _shift(v, dt)
        return out
    return x


def outside_part(lab: dict, t0: float, t1: float) -> dict:
    """{"steps": n, "key_events": k} of one part's labels, already on the recording's clock: how many timeline steps
    start or end, and how many key events lie, more than STEP_SLACK_S outside the part's own footage (t0 to t1). The
    stitch keeps them where the model put them, in a neighbouring part's footage, so the board flags them
    (board/build.py steps_outside). Empty when every time lies within the part."""
    from label.episode import STEP_SLACK_S, number
    lo, hi = t0 - STEP_SLACK_S, t1 + STEP_SLACK_S

    def out(x) -> bool:
        return x is not None and not lo <= x <= hi
    n = sum(any(out(number(s.get(k))) for k in ("start_s", "end_s"))
            for s in lab.get("timeline") or [] if isinstance(s, dict))
    k = sum(out(number(e.get("t_s"))) for e in lab.get("key_events") or [] if isinstance(e, dict))
    return {**({"steps": n} if n else {}), **({"key_events": k} if k else {})}


JOIN_GUARD_S = 2.0
KEY_CUT_GUARD_S = 1.0
OUTCOME_FIELDS = ("outcome", "success_predicate", "completed_at_s", "goal_reached_at_s", "undone_at_s", "undone_by",
                  "judged_against", "subtask")


def _cut_sides(entry: dict) -> set[str]:
    """Only explicit start/end wording establishes which source edge a boundary claim describes."""
    text = " ".join(str(entry.get(k) or "") for k in ("category", "issue", "note", "reason", "label"))
    text = text.lower().replace("_", " ").replace("-", " ")
    sides = set()
    if re.search(r"\b(?:starts?|begins?) (?:in the middle|mid)\b|"
                 r"\bcontinues?\b.{0,60}\bfrom (?:the )?(?:previous|prior) (?:part|clip)\b|"
                 r"\balready (?:under way|underway|in progress) at (?:the )?(?:start|first frame)\b", text):
        sides.add("start")
    if re.search(r"\b(?:ends?|ending|ended) (?:in the middle|mid)\b|"
                 r"\b(?:clip|part|video|recording|episode) (?:ends?|ended|cuts? off|is cut off) (?:before|mid)\b|"
                 r"\b(?:clip|part|video|recording|episode) (?:ends?|ended) while\b[^.]*\bstill (?:under way|underway|in progress)\b|"
                 r"\bstill (?:under way|underway|in progress) at (?:the )?(?:end|last frame|cut)\b", text):
        sides.add("end")
    return sides


def _internal_boundary(source: dict, side: str, cuts_s: list[float]) -> float | None:
    from label.episode import number
    edge = number(source.get("t0_s" if side == "start" else "t1_s"))
    return next((cut for cut in cuts_s if edge is not None and abs(edge - cut) <= .001), None)


def _boundary_only_text(text) -> bool:
    """Require boundary assessments throughout the prose; mixed or unknown claims stay attributed."""
    clauses = [clause.strip().lower() for clause in re.split(r'[;\n]|\.(?=\s|$)', str(text or ''))
               if clause.strip()]
    subject = r'(?:(?:the|this) )?(?:clip|part|video|recording|episode)'
    activity = r'(?:(?:a|the|this) )?(?:task|activity|action|work|recovery)'
    underway = r'(?:under way|underway|in progress)'
    middle = r'(?:mid[- ](?:task|activity|action)|in the middle of ' + activity + r')'
    # Full clause matching deliberately permits only abstract temporal statements.
    # Extra causal, relative or physical-event clauses require keeping the source judgment.
    temporal = re.compile('|'.join([
        subject + r' (?:starts?|begins?|ends?|ended) ' + middle,
        subject + r' (?:ends?|ended|cuts? off|is cut off) before ' + activity +
        r' (?:can finish|finishes|is finished|is completed)',
        subject + r' (?:ends?|ended|cuts? off|is cut off) before (?:(?:a|the|this) )?[a-z]+ can finish',
        subject + r' (?:ends?|ended) while ' + activity + r' is still ' + underway,
        r'(?:starts?|begins?) ' + middle,
        r'already ' + underway + r' at (?:the )?(?:start|first frame)',
        r'still ' + underway + r' at (?:the )?(?:end|last frame|cut)',
        r'continues? (?:' + activity + r' )?from (?:the )?(?:previous|prior) (?:part|clip)',
        r'continues across the internal request cut at [0-9]+(?:\.[0-9]+)?s']))
    return bool(clauses) and all(temporal.fullmatch(clause) for clause in clauses)


def _observed_incident(entry: dict, labels: dict, cuts_s: list[float]) -> bool:
    """Structured failure or wasted-motion onset within this assessment is independent of its cut note."""
    from label.episode import number
    start = number(entry.get('start_s') if 'start_s' in entry else entry.get('t_s'))
    end = number(entry.get('end_s'))
    if start is None:
        return False
    end = start if end is None else end
    owner = entry.get('source_piece') or {}

    def within(row, key):
        instant = number(row.get(key))
        source = row.get('source_piece') or {}
        return (instant is not None and start - .001 <= instant <= end + .001
                and (not owner or not source or owner.get('index') == source.get('index')))

    return (any(event.get('outcome') == 'failure' and within(event, 't_s')
                and not is_cut_key_event(event, cuts_s) for event in labels.get('key_events') or [])
            or any(step.get('contribution') == 'wasteful' and within(step, 'start_s')
                   for step in labels.get('timeline') or [])
            or any(row.get('failure') and within(row, 'failure_t_s')
                   for row in labels.get('recovery') or []))


def is_cut_artifact(iss: dict, part: int, count: int, t0: float, t1: float, cuts_s: list[float],
                    labels: dict | None = None) -> bool:
    """Only a boundary-only assessment at an internal edge, never an observed incident or original edge."""
    from label.episode import number
    cat = str((iss or {}).get("category") or "").lower()
    text = str((iss or {}).get("issue") or "").lower()
    if not any(tag in cat for tag in CUT_TAGS) and not any(w in text for w in ("cut off", "truncat", "starts mid", "ends mid")):
        return False
    if not _boundary_only_text(iss.get('issue')) or _observed_incident(iss, labels or {}, cuts_s):
        return False
    ts = number(iss.get("t_s"))
    if ts is not None and not t0 - .001 <= ts <= t1 + .001:
        return False
    source = {"t0_s": t0, "t1_s": t1}
    for side in _cut_sides(iss):
        if (side == "start" and part == 1) or (side == "end" and part == count):
            continue
        edge = _internal_boundary(source, side, cuts_s)
        if edge is not None and (ts is None or abs(ts - edge) <= CUT_GUARD_S):
            return True
    return False


def _object_tokens(objects) -> list[frozenset]:
    """Full object-name token sets permit omitted modifiers; shared final nouns alone do not match two modifiers."""
    out = []
    for obj in objects if isinstance(objects, list) else []:
        words = re.findall(r"[a-z0-9]+", str(obj.get("name") if isinstance(obj, dict) else obj).lower())
        words = [word[:-1] if word.endswith("s") and not word.endswith("ss") and len(word) > 3 else word
                 for word in words if word not in ("a", "an", "the")]
        if words:
            out.append(frozenset(words))
    return out


def _last_task(task):
    return (task.get("joined_from") or [task])[-1]


def join_scene_objects(objects: list, tasks: list) -> list:
    """The recording's scene objects from its parts' [(part number, part object)], each part object carrying its own
    source_piece. Parts are labelled by separate requests, so a name proves nothing across them: two parts' "cup" can be
    two cups, and one cup can be named differently in each. Objects are joined only where the parts show they are one:
    a task joined across a cut (join_across_cuts, which already requires its halves to share an object) names the
    object in both halves, the name in each half answers to exactly one name in the other, and in each half's part
    exactly one listed object answers to that name. A join never holds two objects of one part: part 1's red cup and
    blue cup are two objects, so part 2's cup cannot join both of them into one. That holds because an object joins at
    most one object of the next part and one of the part before (two pairings of one object contradict each other and
    neither joins), so a joined object is a chain of one record per part. A joined object keeps its first part's
    record, with every later part's own record under "continues"; every other object stays its own entry."""
    parent = list(range(len(objects)))

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def join(x, y):
        parent[root(y)] = root(x)

    def listed(part, name_tokens):
        hits = [k for k, (n, obj) in enumerate(objects) if n == part
                and any(a <= name_tokens or name_tokens <= a for a in _object_tokens([obj]))]
        return hits[0] if len(hits) == 1 else None

    pairs = set()
    for task in tasks:
        halves = task.get("joined_from") or []
        for left, right in zip(halves, halves[1:]):
            a_part, b_part = (left.get("source_piece") or {}).get("index"), (right.get("source_piece") or {}).get("index")
            if not isinstance(a_part, int) or b_part != a_part + 1:
                continue
            # each half's distinct names: a half naming one object twice (cup and the cup, cups and cup) names it once
            names_a, names_b = (list(dict.fromkeys(_object_tokens(h.get("objects")))) for h in (left, right))
            fits = lambda a, b: a <= b or b <= a
            for a in names_a:
                for b in names_b:
                    if fits(a, b) and sum(fits(a, o) for o in names_b) == 1 and sum(fits(o, b) for o in names_a) == 1:
                        x, y = listed(a_part, a), listed(b_part, b)
                        if x is not None and y is not None:
                            pairs.add((x, y))
    # two tasks that pair one object with two different objects of the next part contradict each other: neither joins.
    # This alone keeps each joined object to one record per part (the docstring), in whatever order the tasks come.
    for x, y in sorted(pairs):
        if sum(p[0] == x for p in pairs) == 1 and sum(p[1] == y for p in pairs) == 1:
            join(x, y)
    out, entry = [], {}
    for k, (_, obj) in enumerate(objects):
        r = root(k)
        if r not in entry:
            entry[r] = copy.deepcopy(obj)
            out.append(entry[r])
        else:
            entry[r].setdefault("continues", []).append(copy.deepcopy(obj))
    return out


def join_across_cuts(tasks: list, cuts_s: list[float]) -> list:
    """Join unambiguous adjacent-part continuations, keeping every source judgment and its outcome evidence."""
    from label.episode import number
    tasks = sorted(tasks, key=lambda task: (number(task.get("start_s")) is None, number(task.get("start_s")) or 0))
    for cut in cuts_s:
        pairs = []
        for left in tasks:
            end = number(left.get("end_s"))
            tail = _last_task(left)
            source = tail.get("source_piece") or {}
            if end is None or end > cut + .001 or abs(end - cut) > JOIN_GUARD_S \
                    or _internal_boundary(source, "end", [cut]) is None:
                continue
            if "end" not in _cut_sides({"note": tail.get("note")}):
                continue
            for right in tasks:
                start = number(right.get("start_s"))
                other = right.get("source_piece") or {}
                if right is left or start is None or start < cut - .001 or abs(start - cut) > JOIN_GUARD_S:
                    continue
                if _internal_boundary(other, "start", [cut]) is None or not isinstance(source.get("index"), int) \
                        or other.get("index") != source["index"] + 1:
                    continue
                if "start" not in _cut_sides({"note": right.get("note")}):
                    continue
                if any(a <= b or b <= a for a in _object_tokens(left.get("objects"))
                       for b in _object_tokens(right.get("objects"))):
                    pairs.append((left, right))
        # Ambiguous simultaneous tasks stay separate; do not select the first shared object in a list.
        for left, right in pairs:
            if sum(a is left or b is right for a, b in pairs) != 1:
                continue
            objects = []
            for obj in [*(left.get("objects") or []), *(right.get("objects") or [])]:
                if obj not in objects:
                    objects.append(copy.deepcopy(obj))
            originals = copy.deepcopy((left.get("joined_from") or [left]) + (right.get("joined_from") or [right]))
            joined = {**left, "end_s": right.get("end_s"), "objects": objects,
                      **{key: right.get(key) for key in OUTCOME_FIELDS},
                      "source_piece": copy.deepcopy(right["source_piece"]), "joined_from": originals,
                      "note": f"Continues across the internal request cut at {cut:g}s. " + str(right.get("note") or "")}
            tasks = [joined if task is left else task for task in tasks if task is not right]
    return tasks


def mark_cut_tasks(tasks: list, cuts_s: list[float], labels: dict | None = None) -> list:
    """Qualify an unfinished task at its owning internal end, retaining the part's original judgment. The task's own
    span establishes that our cut ended its view, so an unclear outcome there is qualified whatever its note says: it
    holds no judgment to lose, and an incident inside the task stays recorded as its own event or mistake. A partial or
    failure outcome is a judgment, so it is qualified only when the note attributes it to the cut alone and no incident
    is observed in the task."""
    from label.episode import number
    out = []
    for task in tasks:
        tail = _last_task(task)
        end = number(task.get("end_s"))
        cut = _internal_boundary(tail.get("source_piece") or {}, "end", cuts_s)
        at_cut = (cut is not None and end is not None and cut - JOIN_GUARD_S <= end <= cut + .001
                  and number(task.get("goal_reached_at_s")) is None and number(task.get("undone_at_s")) is None)
        if at_cut and (task.get("outcome") == "unclear" or (
                task.get("outcome") in ("partial", "failure")
                and "end" in _cut_sides({"note": tail.get("note")})
                and _boundary_only_text(tail.get('note'))
                and not _observed_incident(tail, labels or {}, cuts_s))):
            task = {**task, "outcome": "continues_past_cut", "outcome_in_part": task.get("outcome"),
                    "completed_at_s_in_part": task.get("completed_at_s"), "completed_at_s": None,
                    "cut_continuation": {"side": "end", "t_s": cut, "evidence": tail.get("note")},
                    "note": str(task.get("note") or "") + f" Still under way at the internal request cut at {cut:g}s."}
        out.append(task)
    return out


def is_cut_key_event(event: dict, cuts_s: list[float]) -> bool:
    """Exclude a boundary-only event with explicit truncation evidence, never a nearby physical failure."""
    from label.episode import number
    ts = number(event.get("t_s"))
    source = event.get("source_piece") or {}
    lo, hi = number(source.get("t0_s")), number(source.get("t1_s"))
    if ts is None or lo is None or hi is None or not lo - .001 <= ts <= hi + .001 \
            or event.get("outcome") not in ("failure", "unclear"):
        return False
    label = str(event.get("label") or "").strip().lower()
    # A real event with an incidental end-of-clip note keeps its label and outcome.
    if not _boundary_only_text(label):
        return False
    if any(value is not None and str(value).strip() and not _boundary_only_text(value)
           for value in (event.get('note'), event.get('evidence'))):
        return False
    if not re.match(r"^(?:the )?(?:clip|part|video|recording|episode) (?:starts?|begins?|ends?|ended|cuts? off|is cut off)\b", label):
        return False
    for side in _cut_sides({"label": label}):
        cut = _internal_boundary(source, side, cuts_s)
        if cut is not None and abs(ts - cut) <= KEY_CUT_GUARD_S:
            return True
    return False


def merge_jaws(ep_dir, parts, missing=()):
    """Retain independent part calibrations and proof, with explicit parent-clock coordinates."""
    from label.evidence_access import piece_reference, source_proof
    from label.episode import number
    views = sorted({view for _, result in parts for view in result.get('jaw_evidence') or {}})
    if not views:
        return {}
    context = json.loads((Path(ep_dir) / 'context.json').read_text())
    parent_proof = source_proof(context, ep_dir)
    merged = {}
    for view in views:
        rows = []
        for pc, result in sorted(parts, key=lambda item: item[0]['piece']['index']):
            piece = pc.get('piece') or {}
            start, end = number(piece.get('t0_s')), number(piece.get('t1_s'))
            record = copy.deepcopy((result.get('jaw_evidence') or {}).get(view) or {
                'status': 'abstained', 'reason': 'no_saved_measurement_for_part', 'events': []})
            owner = {key: piece.get(key) for key in ('index', 't0_s', 't1_s')}
            row = {'part': piece.get('index'), 'source_piece': owner, 'record': record,
                   'parse_ok': result.get('parse_ok'),
                   'source_proof': copy.deepcopy((result.get('input_identity') or {}).get('source_proof'))}
            location = piece_reference(ep_dir, result.get('episode_dir'), piece)
            if location is not None:
                row['source_location'] = location
            if start is not None and end is not None and end >= start:
                row['time_origin_s'] = start
                row['times_s'] = [number(value) + start if number(value) is not None else None
                                  for value in record.get('times_s') or []]
                row['events'] = [{**copy.deepcopy(event),
                                  **{key: number(event[key]) + start for key in ('start_s', 'end_s')
                                     if number(event.get(key)) is not None},
                                  'source_piece': dict(owner)} for event in record.get('events') or []]
            else:
                row['timing_reason'] = 'No exact owning part window; original local times retained only.'
            rows.append(row)
        merged[view] = {'status': 'segmented', 'source_proof': parent_proof, 'parts': rows,
                        'missing': copy.deepcopy(list(missing)),
                        'limitations': ['Each part retains its own calibration and coverage qualification.',
                                        'Original record times are part-local; mapped times add time_origin_s.',
                                        'No marker motion is inferred across a request cut or missing part.']}
    return merged


def merge_camera_observations(ep_dir: Path, parts: list[tuple[dict, dict]], missing, ep: dict) -> dict | None:
    """Only source-current, agreeing part observations describe a whole recording's view."""
    if not any(isinstance(r.get('camera_observations'), dict) for _, r in parts):
        return None
    from label.evidence_access import source_proof, piece_reference
    from label.harness import camera_observation_sources_match
    from label import episode as me
    from board.dictionary_projection import current_camera_observations
    ep_dir = Path(ep_dir)
    ctx = ep['context']
    views = me.views(ep)
    views = [view for view in views if any(
        isinstance(record.get('camera_observations'), dict)
        and view in (record['camera_observations'].get('views') or []) for _, record in parts)]
    result = {'version': 1, 'views': views, 'source_proof': source_proof(ctx, ep_dir),
              'observations': [], 'withheld_observations': [], 'parts': [],
              'provenance': {'method': 'agreement of source-bound final annotation parts'}}
    by_view = {view: [] for view in views}
    for pc, record in parts:
        owner = copy.deepcopy(pc.get('piece') or {})
        child = copy.deepcopy(record.get('camera_observations') or {})
        directory = Path(record['episode_dir']) if record.get('episode_dir') else None
        location = piece_reference(ep_dir, directory, owner)
        valid = False
        if location:
            actual_ctx = json.loads((directory / 'context.json').read_text())
            current_camera_observations(child, actual_ctx, directory)
            valid = (child.get('source_current') is True and camera_observation_sources_match(
                result['source_proof'], child.get('source_proof'), child.get('views')))
        result['parts'].append({'source_piece': {key: owner.get(key) for key in ('index', 't0_s', 't1_s')},
                                'source_location': location, 'record': child})
        if valid:
            for row in child.get('observations') or []:
                if row['view'] in by_view:
                    by_view[row['view']].append(row)
    expected = len(parts) + len(missing)
    for view in views:
        rows = by_view[view]
        if len(rows) == expected and len({r['content'] for r in rows}) == 1:
            result['observations'].append({'view': view, 'content': rows[0]['content'],
                'evidence': ' | '.join(dict.fromkeys(r['evidence'] for r in rows))})
        else:
            result['withheld_observations'].append({'view': view,
                'why': 'Request parts disagree or do not all supply a verified observation for this view.'})
    if result['withheld_observations']:
        result['withheld_observations'].extend({'observation': row,
            'why': 'Camera interpretation requires agreement for every supplied view ID.'}
            for row in result['observations'])
        result['observations'] = []
    return result


def stitch(ep_dir: Path, parts: list[tuple[dict, dict]], missing: list[dict] = ()) -> dict:
    """One labelling result for the whole recording from its parts' results [(part context, part result), ...]
    in order. Times are shifted onto the recording's clock; observations and actual tasks are joined.
    Independently inferred request goals/outcomes/progress retain their scope under part_assessments,
    never becoming episode tasks or whole-recording progress. Cut issues are set aside in _excluded. missing names
    the parts that gave no labels ({"part", "t0_s", "t1_s", "why"}, stitch_run): the recording is stitched from the
    rest, its cuts are still every cut, and the record keeps the gaps under stitched["missing"]."""
    from label import episode as me
    ep_dir = Path(ep_dir)
    ep = me.load(ep_dir)
    pl = me.plan(ep)
    missing = sorted(missing, key=lambda g: g["part"])
    count = len(parts) + len(missing)
    starts = sorted([float(pc["piece"]["t0_s"]) for pc, _ in parts] + [float(g["t0_s"]) for g in missing])
    cuts_s = starts[1:]
    L = {"scene": {"objects": [], "setting": ""}, "timeline": [], "key_events": [], "state_changes": [],
         "scene_graph": [], "recovery": [], "instruction_variants": [], "data_issues": [], "operator_mistakes": [],
         "tasks": [], "part_assessments": [], "progress_scope": "request_parts"}
    contacts_model, contacts_missing, views, strip_part = {}, [], {"shown": [], "strips": {}}, {}
    excluded, summaries, reviews, scene_objects = [], [], [], []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "est_cost_usd": 0.0,
             "cached_tokens": 0, "cache_write_tokens": 0, "latency_s": 0.0}
    still, part_info, dropped, outside = [], [], [], []
    for n, (pc, r) in enumerate(parts, start=1):
        i = int(pc["piece"].get("index") or n)      # the part's own number, with a gap where a part gave no labels
        t0, t1 = float(pc["piece"]["t0_s"]), float(pc["piece"]["t1_s"])
        from board.label_projection import project
        projected, omitted = project(r.get('labels') or {})
        lab = _shift(projected, t0)
        owner = {"index": i, "t0_s": t0, "t1_s": t1}
        for field in ("tasks", "key_events", "timeline"):
            lab[field] = [{**item, "source_piece": dict(owner)} for item in lab.get(field) or []]
        L['part_assessments'].append({
            'source_piece': dict(owner),
            **{field: copy.deepcopy(lab.get(field)) for field in
               ('task_summary', 'completion', 'performance_review', 'goal_alignment')},
            'progress': [copy.deepcopy(step) for step in lab['timeline'] if step.get('progress') is not None],
        })
        for step in lab['timeline']:
            # a step's progress is toward the sub-task under way in its own part: no reading of the whole recording,
            # so it leaves the recording's progress, and stays with the step for the task it belongs to
            # (board/scores.py, a task's own progress)
            if step.get('progress') is not None:
                step['progress_in_part'] = step['progress']
            step['progress'] = None
        off = outside_part(lab, t0, t1)
        if off:
            outside.append({"part": i, "t0_s": t0, "t1_s": t1, **off})
        scene_objects += [(i, {**o, "source_piece": dict(owner)})
                          for o in (lab.get("scene") or {}).get("objects") or [] if isinstance(o, dict)]
        L["scene"]["setting"] = L["scene"]["setting"] or (lab.get("scene") or {}).get("setting") or ""
        for k in ("timeline", "key_events", "state_changes", "scene_graph", "recovery"):
            L[k] += lab.get(k) or []
        for v in lab.get("instruction_variants") or []:
            if v not in L["instruction_variants"] and len(L["instruction_variants"]) < 6:
                L["instruction_variants"].append(v)
        for k in ("data_issues", "operator_mistakes"):
            for iss in lab.get(k) or []:
                iss = {**iss, "part": i}
                semantic = {key: value for key, value in iss.items() if key not in ('part', 'parts')}
                same = next((x for x in L[k] if x.get("t_s") is None and iss.get("t_s") is None
                             and {key: value for key, value in x.items() if key not in ('part', 'parts')} == semantic), None)
                if same is not None:
                    # an issue about a whole part, reported by several parts, is one issue about the recording
                    same["parts"] = sorted(set(same.get("parts") or [same["part"]]) | {i})
                    continue
                if is_cut_artifact(iss, i, count, t0, t1, cuts_s, lab):
                    excluded.append({**iss, "list": k, "excluded_by": "piece_cut",
                                     "reason": "the labelling pipeline cut the recording into parts here; being cut "
                                               "off at this point describes our cut, not the recording"})
                else:
                    L[k].append(iss)
        # a contact cut by one of our cuts is shown by each part it reaches, with the begin strip only in the part where
        # it begins and the end strip only where it ends (label/episode.py contact_strips); each strip's times come from
        # the part that showed it, on the recording's clock
        cv = r.get("contact_views") or {}
        for cid in cv.get("shown") or []:
            if cid not in views["shown"]:
                views["shown"].append(cid)
                views["strips"][cid] = {}
            for k, ts in ((cv.get("strips") or {}).get(cid) or {}).items():
                if k not in views["strips"][cid]:
                    views["strips"][cid][k] = [round(float(x) + t0, 3) for x in ts]
                    strip_part[cid, k] = i
        # the first answer that saw a contact is kept, with its first and last touch frame from the parts whose strips
        # they count in
        for c in lab.get("contacts") or []:
            if not isinstance(c, dict) or not isinstance(c.get('id'), str):
                continue
            m = contacts_model.setdefault(c.get("id"), {k: v for k, v in c.items() if k not in STRIP_FIELDS.values()})
            for k, field in STRIP_FIELDS.items():
                if strip_part.get((c.get("id"), k)) == i and field in c:
                    m[field] = c[field]
        contacts_missing += [c for c in lab.get("contacts_missing") or [] if isinstance(c, dict)]
        summ = (lab.get("task_summary") or "").strip()
        if summ:
            summaries.append(summ)
        if lab.get("performance_review"):
            reviews.append(f"Part {i}: {lab['performance_review'].strip()}")
        if lab.get("tasks"):
            L["tasks"] += lab["tasks"]
        # what each part's reply broke of the output format, left out of its labels (label/harness.py typed_labels)
        dropped += [{**item, 'part': i} for item in omitted]
        u = r.get("usage") or {}
        for k in usage:
            total = usage[k] + float(u.get(k) or 0)
            usage[k] = total if k == 'est_cost_usd' else round(total, 4)
        cfg = r.get("config") or {}
        still += _shift(r.get("arm_still_spans") or [], t0)
        part_info.append({"part": i, "t0_s": t0, "t1_s": t1, "episode_dir": r.get("episode_dir"),
                          "example_dir": r.get("example_dir"),
                          **_part_dispatch(r),
                          **({'annotation_policy': copy.deepcopy(r['annotation_policy'])}
                             if r.get('annotation_policy') else {}),
                          'config': copy.deepcopy(cfg),
                          "cost_usd": u.get("est_cost_usd"), "parse_ok": r.get("parse_ok")})
    uniq = list(dict.fromkeys(summaries))
    L["task_summary"] = uniq[0] if len(uniq) == 1 else " ".join(f"({i}) {s}" for i, s in enumerate(uniq, 1))
    L["performance_review"] = " ".join(reviews)
    # the output format each part was asked in (label_schema): in the tasks format the recording's completion follows
    # what the recording gives (board/label_projection.py derive_completion), from its tasks joined across the cuts
    # below; a part asked before it was asked in the episode or session mode of its policy, and a mix of the two has
    # no single reading
    formats = {r.get('label_schema') for _, r in parts}
    tasks_format = formats == {LABEL_SCHEMA}
    session = any((r.get('annotation_policy') or {}).get('task_mode') == 'session' for _, r in parts)
    gaps = ("; " + "; ".join(f"part {g['part']} has no labels, as {g['why']}" for g in missing)) if missing else ""
    empty = {"task_completed": None, "success_predicate": None, "completed_at_s": None, "goal_reached_at_s": None,
             "undone_at_s": None, "undone_by": None}
    parted = (f"Labelled in {count} requests with independently inferred goals. Local outcomes and progress are under "
              "part_assessments; they do not establish a whole-recording outcome.")
    rig = me.rig(ep)
    instruction = bool(task_claims(ep["context"])["instruction"])
    if len(formats) > 1:
        L["completion"] = {**empty, "reason": "parts labelled with different output formats" + gaps}
    elif not tasks_format:
        # a session has no recording-wide goal in one request or in many: its outcome is its tasks', joined across the
        # cuts below; an episode labelled in parts lacks a whole-recording verdict
        L["completion"] = {**empty, "task_completed": None if session else "unclear",
                           "reason": ("a session of several tasks; each task's outcome is under tasks" if session
                                      else parted) + gaps}
    elif rig != "ego_head" and instruction:
        # no part sees the whole recording, so none can judge it against the instruction: unclear, as an episode
        # labelled in parts always was, and every reader reads that verdict as main does; its tasks joined across the
        # cuts are listed beside it (basis "parts", board/label_projection.py shown_tasks)
        L["completion"] = {**empty, "task_completed": "unclear", "reason": parted + gaps, "basis": "parts"}
    for k in ("timeline",):
        L[k].sort(key=lambda s: (s.get("start_s") is None, s.get("start_s") or 0))      # an untimed step last
    L["tasks"] = mark_cut_tasks(join_across_cuts(L["tasks"], cuts_s), cuts_s, L)
    if tasks_format and not (rig != "ego_head" and instruction):
        # the tasks joined across the cuts give the completion as one request's would; a part with no labels may hold
        # more of the work, so then there is no verdict
        steps = len(me.annotation_lines(ep["context"]))
        n = len(L["tasks"])
        if rig == "ego_head" and (n != 1 or (steps and not instruction) or missing):
            L["completion"] = {**empty, "reason": f"a head-camera clip labelled in {count} requests; each task's "
                                                  "outcome is under tasks" + gaps}
        elif missing:
            L["completion"] = {**empty, "judged_against": None, "from_tasks": n, "basis": "tasks",
                               "reason": f"{n} task{'' if n == 1 else 's'} in the parts with labels" + gaps}
        else:
            L["completion"] = recording_completion(L["tasks"], task_prompts=0 if instruction else steps)
    L["scene"]["objects"] = join_scene_objects(scene_objects, L["tasks"])
    events = []
    for event in L["key_events"]:
        if is_cut_key_event(event, cuts_s):
            excluded.append({**event, "list": "key_events", "excluded_by": "piece_cut",
                             "reason": "This event explicitly describes the owning part's internal request boundary."})
        else:
            events.append(event)
    L["key_events"] = events
    first = parts[0][1]
    ctx = ep["context"]
    cfg = dict(first.get("config") or {})
    part_info = sorted(part_info + [{**g, "parse_ok": False} for g in missing], key=lambda x: x["part"])
    cfg['pieces'] = part_info
    _combine_request_config(cfg)
    # each part routed its own cell width; the recording's route records every part's, and its cost is theirs summed
    routes = [(r.get("config") or {}).get("resolution_route") or {} for _, r in parts]
    cfg["resolution_route"] = {**(routes[0] or {}), "parts": routes,
                               "cost_usd": math.fsum(float(x.get("cost_usd") or 0) for x in routes)}
    # each part's prompt had its own blocks (a contact or depth only some parts hold); the recording's record names
    # every block, output field and implied check any part had, in the order they first appear
    for k in ("prompt_blocks", "schema_fields", "checks_implied"):
        got = [x for _, r in parts for x in (r.get("config") or {}).get(k) or []]
        if got or k in cfg:
            cfg[k] = list(dict.fromkeys(got))
    # a contact of the recording is kept when the record of some part lists it; a part's record lists all its touch
    # contacts, shown or left out by the cap, so one timed by a signal that is not touch (touch_contacts) is left out
    asked = {c.get("id") for _, r in parts for c in r.get("contacts") or [] if isinstance(c, dict)}
    # the recording's contacts as write_pieces gave them to the parts: stored, or found once on the whole recording
    # when its context has none, with the touch verdicts write_pieces wrote into every part's signal entries, so the
    # ids match and no signal is judged again (the plan's verdicts for parts written without them)
    from label import contacts as lc
    entries = parts[0][0].get("signals") or []
    verdicts = (frozenset(s["name"] for s in entries if s["touch"]) if entries and all("touch" in s for s in entries)
                else pl["touch"])
    recorded = lc.of_episode(ep, verdicts) if asked else []
    if excluded:
        L["_excluded"] = excluded
    if dropped:
        L["_dropped"] = dropped
    if views["shown"] or contacts_missing:
        L["contacts"], L["contacts_missing"] = list(contacts_model.values()), contacts_missing
    from label.sensor_evidence import merge
    sensor_evidence = merge(parts, ctx, ep_dir=ep_dir)
    from label.evidence_access import merge as merge_inspection
    evidence_inspection = merge_inspection(parts, ep_dir)
    jaw_evidence = merge_jaws(ep_dir, parts, missing)
    camera_observations = merge_camera_observations(ep_dir, parts, missing, ep)
    selected = policy.effective(ctx)
    result = {"episode_dir": str(ep_dir), "model": first.get("model"), "reasoning_effort": first.get("reasoning_effort"),
            'annotation_policy': selected,
            **({'camera_observations': camera_observations} if camera_observations is not None else {}),
            **({'jaw_evidence': jaw_evidence} if jaw_evidence else {}),
            **({"sensor_evidence": sensor_evidence} if sensor_evidence["sensors"] else {}),
            **({"evidence_inspection": evidence_inspection} if evidence_inspection['parts'] else {}),
            # each part was checked against the recording's instruction on what it shows (write_pieces); the parts'
            # comparisons are under part_assessments, since no part sees the whole asked work
            "given_prompt": task_claims(ctx)["instruction"],
            "prompt_mode": ("session" if selected.get('task_mode') == 'session' and not formats & {LABEL_SCHEMA}
                            else "given" if task_claims(ctx)["instruction"] else "inferred"),
            **({"label_schema": LABEL_SCHEMA} if tasks_format else {}),
            "task_label": ctx.get("task_label"), "sampling": first.get("sampling"),
            "arm_still_spans": still, "dataset_checks": pl["checks"], "config": cfg,
            "provider": first.get("provider"), "parse_ok": True, "labels": L, "usage": usage,
            "stitched": {"parts": count, "cuts_s": cuts_s, **({"missing": missing} if missing else {}),
                         **({"outside_part": outside} if outside else {})},
            # each part's undecodable stretches, already on the recording's clock (label/episode.py decode_failures)
            "decode_failed": [x for _, r in parts for x in r.get("decode_failed") or []],
            **({"contacts": [c for c in recorded if c.get("id") in asked], "contact_views": views}
               if views["shown"] else {})}
    _carry_part_billing(result, parts)
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan"])
    ap.add_argument("ep_dir", type=Path)
    a = ap.parse_args()
    from label import episode as me
    e = me.load(a.ep_dir)
    tt, mm = motion(e)
    print(json.dumps(choose_cuts(tt, mm, piece_max(e["context"])), indent=1))
