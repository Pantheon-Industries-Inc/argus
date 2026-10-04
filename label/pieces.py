"""Label a long recording in pieces and stitch the labels back into one timeline.

A recording longer than PIECE_MAX_S is one episode on the board, with one clip and one timeline, but the model
labels it in parts: the recording is cut at natural still points (the least motion near each evenly spaced
target, from the recorded state when there is one and from the anchor camera's image change otherwise), each
part is labelled as its own request, and the parts' labels are shifted back onto the recording's clock and
joined. Nothing about the recording's episode boundaries is invented: an unsplit recording stays one episode,
and its subtasks are the parts' tasks on its one timeline.

Each part is told it is one part of a continuous recording cut at a still moment, so activity carrying across
a cut is expected; an issue a part reports about being cut off at one of our cuts (a truncated start or end)
describes our cut, not the recording, and is set aside with that reason. A task that carries across a cut is
reported by both parts, one ending at the cut and one starting there; when the two handle the same object they
are joined back into one task (join_across_cuts). The recording's task text describes
the whole recording, so a part is shown it as context, not as its goal.

    python -m label.pieces plan EP_DIR            # print the cuts

The path for your own data (python -m review) cuts every episode that needs it, labels the parts and stitches
them, and Data Review runs the same functions on uploads. No episode on the published board is longer than
PIECE_MAX_S, so none of them is cut.
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np

# the longest recording the published board labels in one request is 443 s (OpenAoE); a recording longer than
# this is labelled in parts, so an upload is never sent to the model in a longer request than the board's
PIECE_MAX_S = {"teleop_arms": 450.0, "handheld_gripper": 450.0, "ego_head": 450.0}
SEARCH = 0.3                # a cut is sought within +-30% of a part's length around its target
SMOOTH_S = 2.0              # motion is averaged over 2 s so a cut lands in a still stretch, not a still frame
CUT_GUARD_S = 10.0          # an issue this close to one of our cuts, of a cut-off kind, describes the cut
CUT_TAGS = ("truncat", "incomplete", "cut_off", "starts_mid", "ends_mid", "mid_task")
JOIN_GUARD_S = 2.0          # a task ending this close to a cut, and one starting this close after it, may be one task


def piece_max(ctx: dict) -> float:
    return PIECE_MAX_S.get(ctx.get("profile"), PIECE_MAX_S["teleop_arms"])


def duration(ctx: dict) -> float:
    return float(ctx.get("duration_s") or ctx["n_state_frames"] / float(ctx["fps"]))


def needs_pieces(ep_dir: Path) -> bool:
    ctx = json.loads((Path(ep_dir) / "context.json").read_text())
    return duration(ctx) > piece_max(ctx) * 1.05


def motion(ep: dict) -> tuple[np.ndarray, np.ndarray]:
    """(times of anchor frames, motion per frame). From the recorded state when the episode has one (joint or
    pose speed summed over actors), else the anchor camera's grey-level change between consecutive frames."""
    from label import episode as me
    n = int(ep["sources"][me.anchor(ep)]["n_frames"])
    t = np.array([me.frame_time(ep, k) for k in range(n)], dtype=np.float64)
    if me.state_kind(ep) != "none" and len(ep["state"]) == n:
        s = np.asarray(ep["state"], dtype=np.float64)
        d = np.abs(np.diff(s, axis=0)).sum(axis=1)
        return t, np.concatenate([[0.0], d])
    return t, video_motion(ep, n)


def video_motion(ep: dict, n: int) -> np.ndarray:
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
        c.seek(targets[0], stream=st, backward=True, any_frame=False)
        prev = None
        for fr in c.decode(st):
            k = index.get(fr.pts)
            if k is None:
                if fr.pts is not None and fr.pts > targets[-1]:
                    break
                continue
            g = fr.reformat(width=64, height=max(2, int(round(fr.height * 64 / fr.width))), format="gray").to_ndarray()
            g = g.astype(np.int16)
            if prev is not None:
                out[k] = np.abs(g - prev).mean()
            prev = g
            if k == n - 1:
                break
    return out


def choose_cuts(t: np.ndarray, m: np.ndarray, max_s: float) -> list[dict]:
    """Cut frames (anchor indices) splitting the recording into equal-ish parts no longer than max_s, each cut at
    the stillest moment near its target."""
    total = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    parts = max(1, math.ceil(total / max_s))
    if parts == 1:
        return []
    L = total / parts
    step = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30
    w = max(1, int(round(SMOOTH_S / max(step, 1e-3))))
    sm = np.convolve(m, np.ones(w) / w, mode="same")
    still_level = float(np.percentile(sm, 25))
    cuts, prev_t = [], float(t[0])
    for i in range(1, parts):
        target = float(t[0]) + i * L
        lo = max(target - SEARCH * L, prev_t + 0.5 * L)
        hi = min(target + SEARCH * L, prev_t + max_s, float(t[-1]) - 0.5 * L)
        idx = np.flatnonzero((t >= lo) & (t <= hi))
        if not len(idx):
            idx = np.array([int(np.searchsorted(t, target))])
        k = int(idx[np.argmin(sm[idx])])
        cuts.append({"frame": k, "t_s": round(float(t[k] - t[0]), 3), "still": bool(sm[k] <= still_level),
                     "motion": round(float(sm[k]), 4), "target_s": round(target - float(t[0]), 1)})
        prev_t = float(t[k])
    return cuts


def fmt_clock(s: float) -> str:
    s = max(0, int(round(s)))
    return f"{s // 60}:{s % 60:02d}"


def write_pieces(ep_dir: Path, pieces_root: Path) -> list[Path]:
    """Write one sidecar folder per part under pieces_root (named <episode>__pNN) and record the cuts in the
    episode's context (context["pieces"]). Returns the part folders."""
    from label import episode as me
    ep_dir = Path(ep_dir)
    ep = me.load(ep_dir)
    ctx = ep["context"]
    t, m = motion(ep)
    cuts = choose_cuts(t, m, piece_max(ctx))
    n = len(t)
    bounds = [0] + [c["frame"] for c in cuts] + [n]
    total = duration(ctx)
    a = me.anchor(ep)
    fps = me.ep_fps(ep)
    src = ep["sources"]
    z = np.load(ep_dir / "state.npz") if (ep_dir / "state.npz").exists() else None
    zs = np.load(ep_dir / "signals.npz") if ctx.get("signals") else None
    tz = dict(np.load(ep_dir / ctx["real_times"])) if ctx.get("real_times") else None
    out = []
    count = len(bounds) - 1
    for i in range(count):
        k0, k1 = bounds[i], bounds[i + 1]
        name = f"{ep_dir.name}__p{i + 1:02d}"
        d = pieces_root / name
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        t0, t1 = float(t[k0] - t[0]), (float(t[k1] - t[0]) if k1 < n else total)
        new_src, new_times = {}, {}
        for v, s in src.items():
            s2 = {kk: vv for kk, vv in s.items() if kk != "kmap"}
            km = ep["kmap"].get(v)
            if v == a or km is None:
                j0, j1 = k0, k1
            else:
                j0, j1 = int(km[k0]), int(km[k1 - 1]) + 1
                np.save(d / f"kmap_{v}.npy", (np.asarray(km[k0:k1]) - j0).astype(np.int32))
                s2["kmap"] = f"kmap_{v}.npy"
            s2["n_frames"] = int(j1 - j0)
            if tz is not None and f"{v}_pts" in tz:
                new_times[f"{v}_pts"] = tz[f"{v}_pts"][j0:j1]
            else:
                s2["base_s"] = round(float(s["base_s"]) + j0 / fps, 9)
            if tz is not None and v in tz:
                new_times[v] = tz[v][j0:j1] - float(tz[a][k0])
            new_src[v] = s2
        c2 = {kk: vv for kk, vv in ctx.items() if kk not in (
            "stream_pairing", "recorded_jumps", "gripper_channels", "capture_qc", "stream_checks", "pieces",
            "instruction", "instruction_note", "real_times", "timebase_neighbour_lag_frames")}
        c2.update(episode_id=name, n_state_frames=int(k1 - k0), duration_s=round(t1 - t0, 3),
                  piece={"of": ep_dir.name, "index": i + 1, "count": count, "t0_s": round(t0, 3), "t1_s": round(t1, 3)})
        note = (f"this clip is part {i + 1} of {count} of one continuous {fmt_clock(total)} recording, from "
                f"{fmt_clock(t0)} to {fmt_clock(t1)} of it. The labelling pipeline cut the recording into parts at "
                "moments of little motion to label it; activity that carries across a cut is expected, and a part "
                "that starts or ends in the middle of an activity is how we cut it, not a truncated or cut-off "
                "recording.")
        if ctx.get("instruction"):
            note += (f" The recording's task text, which describes the whole recording rather than this part, is: "
                     f"\"{ctx['instruction'].strip()}\". This part may show only some of it.")
        if ctx.get("collection_note"):
            note += " " + ctx["collection_note"].strip()
        c2["collection_note"] = note
        if new_times:
            np.savez(d / "times.npz", **new_times)
            c2["real_times"] = "times.npz"
        if z is not None:
            arrs = {kk: z[kk][k0:k1] for kk in z.files}
            np.savez(d / "state.npz", **arrs)
        if zs is not None:          # the context lists the recording's other signals, so the part carries its rows
            np.savez(d / "signals.npz", **{kk: zs[kk][k0:k1] for kk in zs.files})
        if ctx.get("annotation_subtasks"):
            # the dataset's timed subtasks are on the recording's clock; the part is shown those that overlap it, on
            # its own clock and clipped to it
            # (a step with no end time is a moment)
            subs = [(x, float(x.get("t0") or 0.0), float(x["t1"] if x.get("t1") is not None else x.get("t0") or 0.0))
                    for x in ctx["annotation_subtasks"] if isinstance(x, dict)]
            c2["annotation_subtasks"] = [
                {**x, "t0": round(max(a, t0) - t0, 3), **({"t1": round(min(b, t1) - t0, 3)} if "t1" in x else {})}
                for x, a, b in subs if b >= t0 and a < t1]
        (d / "sources.json").write_text(json.dumps(new_src, indent=1))
        (d / "context.json").write_text(json.dumps(c2, indent=1, default=str))
        (d / "instruction.txt").write_text("\n")
        out.append(d)
    ctx["pieces"] = {"max_s": piece_max(ctx), "cuts": cuts, "parts": [p.name for p in out]}
    (ep_dir / "context.json").write_text(json.dumps(ctx, indent=1, default=str))
    return out


def write_units(job: Path, eps: Path) -> dict:
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
            parts = _write_or_reuse(d, proot)
            long_eps[d.name] = [p.name for p in parts]
            for p in parts:
                (units / p.name).symlink_to(p.resolve())
        else:
            (units / d.name).symlink_to(d.resolve())
    return long_eps


def _write_or_reuse(d: Path, proot: Path) -> list[Path]:
    ctx = json.loads((d / "context.json").read_text())
    names = (ctx.get("pieces") or {}).get("parts") or []
    if names and all((proot / n / "context.json").exists() for n in names):
        return [proot / n for n in names]          # a resumed job reuses its parts (their labels depend on them)
    return write_pieces(d, proot)


def stitch_run(job: Path, eps: Path, long_eps: dict, out: Path) -> dict:
    """out/: the short episodes' results from job/run/out as labelled, and one stitched result per long recording.
    A recording with a part that is missing or did not parse is left out and listed under "incomplete"."""
    src = Path(job) / "run" / "out"
    out.mkdir(parents=True, exist_ok=True)
    res = {"stitched": 0, "incomplete": []}
    for p in src.glob("episode_*.json"):
        if "__p" not in p.stem:
            shutil.copy(p, out / p.name)
    for ep, parts in long_eps.items():
        got = []
        for n in parts:
            q = src / f"{n}.json"
            r = json.loads(q.read_text()) if q.exists() else None
            if not r or not r.get("parse_ok"):
                break
            got.append((json.loads((Path(job) / "pieces" / n / "context.json").read_text()), r))
        if len(got) != len(parts):
            res["incomplete"].append(ep)
            continue
        (out / f"{ep}.json").write_text(json.dumps(stitch(Path(eps) / ep, got)))
        res["stitched"] += 1
    return res


# ---------------------------------------------------------------- stitching

T_KEYS = ("t_s", "start_s", "end_s", "completed_at_s", "goal_reached_at_s", "undone_at_s", "failure_t_s",
          "recovered_at_s")


def _shift(x, dt: float):
    """Every time field of a label structure shifted by dt seconds."""
    if isinstance(x, list):
        return [_shift(v, dt) for v in x]
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k in T_KEYS and isinstance(v, (int, float)) and not isinstance(v, bool):
                out[k] = round(float(v) + dt, 3)
            else:
                out[k] = _shift(v, dt)
        return out
    return x


def is_cut_artifact(iss: dict, part: int, count: int, t0: float, t1: float, cuts_s: list[float]) -> bool:
    """An issue of a cut-off kind (truncated start or end, task left incomplete) that sits at one of our cuts,
    or spans a part that is not the recording's last: it describes our cut."""
    cat = str((iss or {}).get("category") or "").lower()
    text = str((iss or {}).get("issue") or "").lower()
    if not any(tag in cat for tag in CUT_TAGS) and not any(w in text for w in ("cut off", "truncat", "starts mid", "ends mid")):
        return False
    ts = iss.get("t_s")
    if not isinstance(ts, (int, float)):
        return part < count          # a part-wide "incomplete" in any part but the last is our cut
    return any(abs(float(ts) - c) <= CUT_GUARD_S for c in cuts_s)


def _heads(objects) -> set:
    """The last word of each object name, singular: "floral fleece blanket" and "floral blanket" are both a blanket."""
    out = set()
    for o in objects or []:
        words = str(o.get("name") if isinstance(o, dict) else o).lower().replace("-", " ").split()
        if words:
            w = words[-1]
            out.add(w[:-1] if w.endswith("s") and not w.endswith("ss") and len(w) > 3 else w)
    return out


def join_across_cuts(tasks: list, cuts_s: list[float]) -> list:
    """The recording's tasks with each task that carries across a cut made one again. Each part reports the task it
    was cut in: the earlier part's ends at the cut, the later part's starts there. Two tasks that meet at a cut
    (within JOIN_GUARD_S) and handle the same object are one task: it runs from the first's start to the second's
    end, keeps the first's description (it saw the task begin), and takes the second's outcome (it saw the task
    end). joined_from keeps every part's own entry. Tasks that meet at a cut but share no object stay apart: a new
    task can begin exactly where we cut."""
    tasks = sorted(tasks, key=lambda t: (t.get("start_s") or 0))
    for c in cuts_s:
        a = next((t for t in tasks if isinstance(t.get("end_s"), (int, float))
                  and abs(t["end_s"] - c) <= JOIN_GUARD_S), None)
        b = next((t for t in tasks if t is not a and isinstance(t.get("start_s"), (int, float))
                  and abs(t["start_s"] - c) <= JOIN_GUARD_S), None)
        if a is None or b is None or not (_heads(a.get("objects")) & _heads(b.get("objects"))):
            continue
        own = lambda t: {k: v for k, v in t.items() if k != "joined_from"}
        joined = {**a, "end_s": b.get("end_s"),
                  "objects": list(dict.fromkeys([*(a.get("objects") or []), *(b.get("objects") or [])])),
                  "outcome": b.get("outcome") or a.get("outcome"), "completed_at_s": b.get("completed_at_s"),
                  "note": f"Carries across the cut at {c:g}s, where the recording was labelled in two parts.",
                  "joined_from": (a.get("joined_from") or [own(a)]) + [own(b)]}
        tasks = [joined if t is a else t for t in tasks if t is not b]
    return tasks


def stitch(ep_dir: Path, parts: list[tuple[dict, dict]]) -> dict:
    """One labelling result for the whole recording from its parts' results [(part context, part result), ...]
    in order. Times are shifted onto the recording's clock; lists are joined; each part's task and outcome
    become one entry of tasks, and a task carried across a cut becomes one again (join_across_cuts); issues
    describing our cuts are set aside in _excluded with the reason."""
    from label import episode as me
    ep_dir = Path(ep_dir)
    ep = me.load(ep_dir)
    pl = me.plan(ep)
    count = len(parts)
    cuts_s = [float(pc["piece"]["t0_s"]) for pc, _ in parts[1:]]
    L = {"scene": {"objects": [], "setting": ""}, "timeline": [], "key_events": [], "state_changes": [],
         "scene_graph": [], "recovery": [], "instruction_variants": [], "data_issues": [], "operator_mistakes": [],
         "tasks": []}
    excluded, summaries, reviews, seen_obj = [], [], [], set()
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "est_cost_usd": 0.0,
             "cached_tokens": 0, "cache_write_tokens": 0, "latency_s": 0.0}
    timesteps, still, part_info = [], [], []
    for i, (pc, r) in enumerate(parts, start=1):
        t0, t1 = float(pc["piece"]["t0_s"]), float(pc["piece"]["t1_s"])
        lab = _shift(r.get("labels") or {}, t0)
        for o in (lab.get("scene") or {}).get("objects") or []:
            key = str(o.get("name", "")).strip().lower()
            if key and key not in seen_obj:
                seen_obj.add(key)
                L["scene"]["objects"].append(o)
        L["scene"]["setting"] = L["scene"]["setting"] or (lab.get("scene") or {}).get("setting") or ""
        for k in ("timeline", "key_events", "state_changes", "scene_graph", "recovery"):
            L[k] += lab.get(k) or []
        for v in lab.get("instruction_variants") or []:
            if v not in L["instruction_variants"] and len(L["instruction_variants"]) < 6:
                L["instruction_variants"].append(v)
        for k in ("data_issues", "operator_mistakes"):
            for iss in lab.get(k) or []:
                iss = {**iss, "part": i}
                same = next((x for x in L[k] if x.get("t_s") is None and iss.get("t_s") is None
                             and x.get("category") == iss.get("category")), None)
                if same is not None:
                    # an issue about a whole part, reported by several parts, is one issue about the recording
                    same["parts"] = sorted(set(same.get("parts") or [same["part"]]) | {i})
                    continue
                if is_cut_artifact(iss, i, count, t0, t1, cuts_s):
                    excluded.append({**iss, "list": k, "excluded_by": "piece_cut",
                                     "reason": "the labelling pipeline cut the recording into parts here; being cut "
                                               "off at this point describes our cut, not the recording"})
                else:
                    L[k].append(iss)
        summ = (lab.get("task_summary") or "").strip()
        if summ:
            summaries.append(summ)
        if lab.get("performance_review"):
            reviews.append(f"Part {i}: {lab['performance_review'].strip()}")
        if lab.get("tasks"):
            L["tasks"] += lab["tasks"]
        else:
            comp = lab.get("completion") or {}
            L["tasks"].append({"start_s": round(t0, 3), "end_s": round(t1, 3), "task": summ or f"Part {i}",
                               "objects": [], "outcome": (comp.get("task_completed") or "unclear"),
                               "success_predicate": comp.get("success_predicate") or "",
                               "completed_at_s": comp.get("completed_at_s"), "note": comp.get("reason") or ""})
        u = r.get("usage") or {}
        for k in usage:
            usage[k] = round(usage[k] + float(u.get(k) or 0), 4)
        cfg = r.get("config") or {}
        timesteps += [round(float(x) + t0, 3) for x in cfg.get("timesteps_s") or []]
        still += _shift(r.get("arm_still_spans") or [], t0)
        part_info.append({"part": i, "t0_s": t0, "t1_s": t1, "episode_dir": r.get("episode_dir"),
                          "cost_usd": u.get("est_cost_usd"), "parse_ok": r.get("parse_ok")})
    uniq = list(dict.fromkeys(summaries))
    L["task_summary"] = uniq[0] if len(uniq) == 1 else " ".join(f"({i}) {s}" for i, s in enumerate(uniq, 1))
    L["performance_review"] = " ".join(reviews)
    L["completion"] = {"task_completed": None, "success_predicate": None, "completed_at_s": None,
                       "goal_reached_at_s": None, "undone_at_s": None, "undone_by": None,
                       "reason": f"a long recording labelled in {count} parts; each part's outcome is under tasks"}
    for k in ("timeline",):
        L[k].sort(key=lambda s: (s.get("start_s") or 0))
    L["tasks"] = join_across_cuts(L["tasks"], cuts_s)
    first = parts[0][1]
    ctx = ep["context"]
    cfg = dict(first.get("config") or {})
    cfg.update(timesteps_s=timesteps, n_timesteps=len(timesteps), pieces=part_info)
    # each part routed its own cell width; the recording's route records every part's, and its cost is theirs summed
    routes = [(r.get("config") or {}).get("resolution_route") or {} for _, r in parts]
    cfg["resolution_route"] = {**(routes[0] or {}), "parts": routes,
                               "cost_usd": round(sum(float(x.get("cost_usd") or 0) for x in routes), 6)}
    if excluded:
        L["_excluded"] = excluded
    return {"episode_dir": str(ep_dir), "model": first.get("model"), "reasoning_effort": first.get("reasoning_effort"),
            # each part inferred its own task, with the recording's task text given only as context (write_pieces), so
            # no part was graded against that text and the recording is not either
            "given_prompt": (ctx.get("instruction") or "").strip() or None, "prompt_mode": "inferred",
            "task_label": ctx.get("task_label"), "sampling": first.get("sampling"),
            "arm_still_spans": still, "dataset_checks": pl["checks"], "config": cfg,
            "provider": first.get("provider"), "parse_ok": True, "labels": L, "usage": usage,
            "stitched": {"parts": count, "cuts_s": cuts_s}}


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
