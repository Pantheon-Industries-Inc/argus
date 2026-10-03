"""Model comparison: the same episodes labelled by several models through one harness, measured from their runs.

A board manifest names the comparison runs under "comparisons" (board/build.py reads the same list; python -m
compare writes the entries for the runs it starts). Paths are absolute or relative to the board folder:

    "comparisons": [
      {"key": "opus55", "name": "Claude Opus 5.5", "run": RUN_DIR},
      {"key": "opus55_ex", "name": "Claude Opus 5.5, in-context learning with an Astra trace", "run": RUN_DIR,
       "example": true, "base": "opus55"},
      ...]

The reference every model is set against, and listed first, is the board's own labels: for every episode a
comparison run was asked, the run output the board's label of that episode was built from (board/build.py
label_sources), measured the same way as every other model's response. So every number compares a model with
exactly the label the board shows, and the board holds one label per episode. It is named after the model that
made those labels (configs/models.json names the pinned models) and has the key "board". "example" says the run was
given one complete annotation of another episode of the same rig (configs/examples); it defaults to whether the
run's command has --example-dir. "base" names the run of the same model without the example (default: the key
without "_ex"). "episodes" names the slice folder the run labelled (default: the slice its run.json records).

Every number comes from the run folders: out/<episode>.json (parse_ok, labels, usage), out/failed_<episode>.json
(a response cut off at the output limit) and the slice's context.json files (rig, length). Nothing is retried,
repaired or re-prompted; a response that did not parse is a result. None of the board's checks or rules enter a
number: they are applied to the board's labels on the page, not to any model's response.

    status   parsed | unparsed | cut_off | no_response (a finished run has no output for it)
             | pending (the run is running or was interrupted, or it was a dry run)

Which episodes each number uses (only episodes the board has a label of: any other episode a comparison run labelled
has no board label to set it against, so it is left out):
  - parse share: every episode the model was asked (pending ones aside), parsed / (parsed + unparsed + cut off);
    a call that returned nothing is reported beside it, not counted as a response. The reference is asked every
    episode the board has a label of, and a board label is always a response that parsed
  - schema violations: every parsed response of that model
  - cost and latency: every response returned (an unparsed response is still paid for)
  - events per minute, key events, subgoals, data issues, operator mistakes: the episodes every model in the
    chart parsed, so no model's number moves because it failed on other episodes than another
  - agreement: each pair on the episodes both parsed
  - in-context learning: each model with and without the trace, on the episodes both of its runs parsed, beside the
    reference's numbers on those of them the board has a label of

    python -m compare.metrics BOARD          # print the summary
    python -m compare.metrics BOARD --json OUT
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from collections import defaultdict
from pathlib import Path

from board.families import Families
from label.harness import episode_cost

FAMILIES = Families()
MODELS_PATH = Path(__file__).resolve().parent.parent / "configs" / "models.json"
RIGS ={"teleop_arms": "teleop", "handheld_gripper": "handheld", "ego_head": "head_camera"}
RIG_NAMES = {"all": "All footage", "teleop": "Teleop", "handheld": "UMI", "head_camera": "Human ego"}
RESPONDED = ("parsed", "unparsed", "cut_off")
OUTCOMES = ("success", "success_then_undone", "failure", "partial", "unclear")
TASK_OUTCOMES = ("success", "partial", "failure")
LISTS = ("data_issues", "operator_mistakes")
PER_EPISODE = ("events_per_min", "key_events", "subgoals", "data_issues", "data_issues_minor", "operator_mistakes",
               "operator_mistakes_minor")


# ---------------------------------------------------------------- the runs

def _run_arg(info: dict, flag: str) -> str | None:
    """A harness option a run was started with, from its command (None when the command leaves it at the harness
    default)."""
    cmd = info.get("command") or []
    if flag in cmd and cmd.index(flag) + 1 < len(cmd):
        return cmd[cmd.index(flag) + 1]
    return None


def _run_model(info: dict) -> str | None:
    """The model id a run was started with: its harness command's --model, or None (the harness default)."""
    return _run_arg(info, "--model")


def _path(p: str, base: Path | None) -> Path:
    return Path(p) if Path(p).is_absolute() or base is None else (Path(base) / p).resolve()


def model_names() -> dict:
    """Model id -> display name, from configs/models.json ({} when it is not there)."""
    try:
        d = json.loads(MODELS_PATH.read_text())
    except (OSError, ValueError):
        return {}
    return {m["model"]: m.get("name") or m["model"] for m in (d.get("models") or {}).values() if m.get("model")}


def reasoning_effort() -> str | None:
    """The reasoning effort a model is run with unless its configs/models.json entry names its own (None when the
    file is not there). Each comparison run's own effort is read from its command (load_models)."""
    try:
        return json.loads(MODELS_PATH.read_text()).get("reasoning")
    except (OSError, ValueError):
        return None


REFERENCE_KEY = "board"
# a static build keeps each model's rail records under compare/lists/, so no model may be called that either
RESERVED_KEYS = (REFERENCE_KEY, "lists")


def reference_model(model_ids: set) -> dict:
    """The reference, the board's own labels, named after the model that made them (their outputs' "model"): its
    display name, else its id without the provider; the names joined with "and" when several models made them."""
    ids = sorted(i for i in model_ids if i)
    names = [model_names().get(i) or i.split("/")[-1] for i in ids] or ["The board"]
    name = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    return {"key": REFERENCE_KEY, "name": name, "episode_name": name, "reference": True, "example": False,
            "base": None, "model": ids[0] if len(ids) == 1 else None}


def load_models(manifest: dict, base: Path | None = None) -> list[dict]:
    """The manifest's comparison runs with what their run.json says about them. base: the board folder, which
    relative paths are read against."""
    out = []
    for c in manifest.get("comparisons") or []:
        run = _path(c["run"], base)
        info = json.loads((run / "run.json").read_text())
        cmd = info.get("command") or []
        eps = _path(c["episodes"], base) if c.get("episodes") else Path(info["slice"])
        example = bool(c["example"]) if "example" in c else "--example-dir" in cmd
        key = c["key"]
        out.append({"key": key, "name": c["name"], "run": run, "episodes": eps, "reference": False,
                    "example": example, "base": c.get("base") or (re.sub(r"_ex$", "", key) if example else None),
                    "run_id": info.get("run_id"), "code": info.get("code"), "model": _run_model(info),
                    "reasoning": _run_arg(info, "--reasoning"),
                    "status": info.get("status"), "slice": info.get("slice"), "episode_name": c["name"]})
    keys = [m["key"] for m in out]
    if len(set(keys)) != len(keys):
        raise ValueError("comparison keys must be unique")
    if set(keys) & set(RESERVED_KEYS):
        raise ValueError(f"no comparison may have the key {' or '.join(map(repr, RESERVED_KEYS))}")
    return out


def slice_episodes(eps: Path) -> list[str]:
    """The episodes a run over this slice was asked to label: its folders with a context.json."""
    return sorted(p.name for p in eps.iterdir() if (p / "context.json").exists())


def read_output(p: Path) -> dict:
    """A run output file's response: status, raw labels when parsed, usage, and the model that gave it."""
    r = json.loads(p.read_text())
    if r.get("dry_run"):
        return {"status": "pending"}
    if p.name.startswith("failed_"):
        # a failed_<episode>.json: the board's own label of an episode whose reply was cut off (board/to_board.py)
        u = r.get("usage") or {}
        cost = round(episode_cost(r), 6) if u.get("est_cost_usd") is not None else u.get("cost")
        return {"status": "cut_off", "cost": cost, "latency": None, "out_tokens": u.get("completion_tokens"),
                "finish_reason": r.get("finish_reason"), "tail": r.get("content_tail") or "", "path": p,
                "model": r.get("model")}
    u = r.get("usage") or {}
    # the episode's billed cost: its model call and, for a routed episode, the routing call made for it
    cost = round(episode_cost(r), 6) if u.get("est_cost_usd") is not None else None
    base = {"cost": cost, "latency": u.get("latency_s"), "out_tokens": u.get("completion_tokens"), "path": p,
            "model": r.get("model")}
    if r.get("parse_ok"):
        return {"status": "parsed", "labels": r.get("labels") or {}, "config": r.get("config") or {}, **base}
    lab = r.get("labels") or {}
    return {"status": "unparsed", "error": lab.get("_parse_error"), "raw": lab.get("_raw") or "", **base}


def response(model: dict, name: str) -> dict:
    """One episode's response: status, raw labels when parsed, usage."""
    out = model["run"] / "out"
    p, f = out / f"{name}.json", out / f"failed_{name}.json"
    if p.exists():
        return read_output(p)
    if f.exists():
        # the harness records a cut-off reply's billed cost and its routing call like any other episode's
        return read_output(f)
    # only a run that finished (label/run.py writes "done", or "exit N" when some calls failed) has episodes it
    # will never answer; a running or interrupted run (killed from outside, then resumed) has episodes to come
    st = str(model.get("status") or "")
    return {"status": "no_response" if st == "done" or st.startswith("exit") else "pending"}


# ---------------------------------------------------------------- which board episode each one is

def board_index(manifest: dict, base: Path | None = None) -> dict:
    """Resolved episode folder -> (dataset, the run's episode name, the board's file name) for every episode
    of the board's own slices. A comparison slice is symlinks into these, so resolving both sides identifies
    an episode exactly, where names alone would not (MolmoAct2 and HABIT share episode_000494)."""
    idx = {}
    for e in manifest.get("datasets") or []:
        root = _path(e["episodes"], base)
        pre = e.get("file_prefix") or ""
        if not root.exists():
            continue
        for p in root.iterdir():
            if not p.name.startswith("episode_"):
                continue
            pub = p.name.replace("episode_", f"episode_{pre}", 1) if pre else p.name
            idx[p.resolve()] = {"dataset": e["dataset"], "run_episode": p.name, "file": pub + ".json"}
    return idx


def episode_identity(eps: Path, name: str, bidx: dict) -> dict:
    """The comparison episode's board identity and context: dataset, board file, rig, length."""
    from board.build import episode_seconds   # board.build imports this module, so imported here
    d = eps / name
    ctx = json.loads((d / "context.json").read_text())
    hit = bidx.get(d.resolve()) or {}
    return {"name": name, "dataset": hit.get("dataset") or ctx.get("dataset"), "file": hit.get("file"),
            "run_episode": hit.get("run_episode"), "episode_id": (hit.get("file") or name + ".json")[:-5],
            "rig": RIGS.get(ctx.get("profile"), ctx.get("profile") or "other"), "seconds": episode_seconds(ctx, d)}


# ---------------------------------------------------------------- per response

def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _list(L: dict, key: str) -> list:
    """L[key] when it is a list, else an empty list."""
    return L[key] if isinstance(L.get(key), list) else []


def schema_violations(L: dict, config: dict | None = None) -> list[str]:
    """Where a parsed response departs from the schema every model was given. Head-camera responses carry
    tasks instead of one completion. An episode with a single camera, named for its actor
    (FastUMI's one-gripper tasks: the prompt names the camera "gripper" and still lists left, right or both),
    may name that actor: the ambiguity is the prompt's, and counting it would score our wording as the model's
    failure to follow it."""
    v = []
    arms = {"left", "right", "both"}
    labels = (config or {}).get("cam_labels") or []
    if len((config or {}).get("views") or []) == 1 and labels:
        arms |= {str(labels[0])}
    req = {"scene": dict, "timeline": list, "task_summary": str, "key_events": list, "state_changes": list,
           "scene_graph": list, "recovery": list, "instruction_variants": list, "performance_review": str,
           "data_issues": list}
    for k, t in req.items():
        if k not in L:
            v.append(f"{k} missing")
        elif not isinstance(L[k], t):
            v.append(f"{k} is not a {t.__name__}")
    if "tasks" in L:
        if not isinstance(L["tasks"], list):
            v.append("tasks is not a list")
        else:
            for i, t in enumerate(L["tasks"]):
                if not isinstance(t, dict) or str(t.get("outcome") or "").lower() not in TASK_OUTCOMES:
                    v.append(f"task {i}: outcome")
    else:
        c = L.get("completion")
        if not isinstance(c, dict):
            v.append("completion missing")
        elif str(c.get("task_completed") or "").lower() not in OUTCOMES:
            v.append(f"completion: task_completed {c.get('task_completed')!r}")
    for i, s in enumerate(_list(L, "timeline")):
        if not isinstance(s, dict):
            v.append(f"timeline {i}: not a segment")
            continue
        if not (_num(s.get("start_s")) and _num(s.get("end_s"))):
            v.append(f"timeline {i}: times")
        elif s["end_s"] < s["start_s"]:
            v.append(f"timeline {i}: ends before it starts")
        if s.get("arm") not in arms:
            v.append(f"timeline {i}: arm {s.get('arm')!r}")
        if s.get("contribution") not in ("advancing", "wasteful", "idle"):
            v.append(f"timeline {i}: contribution {s.get('contribution')!r}")
        if s.get("progress") is not None and not (_num(s["progress"]) and 0 <= s["progress"] <= 1):
            v.append(f"timeline {i}: progress {s.get('progress')!r}")
    for i, k in enumerate(_list(L, "key_events")):
        if not isinstance(k, dict) or not _num(k.get("t_s")):
            v.append(f"key event {i}: time")
        elif str(k.get("outcome") or "").lower() not in ("success", "failure", "unclear"):
            v.append(f"key event {i}: outcome {k.get('outcome')!r}")
    for key in LISTS:
        lst = L.get(key)
        if lst is None:
            continue
        if not isinstance(lst, list):
            v.append(f"{key} is not a list")
            continue
        for i, x in enumerate(lst):
            if not isinstance(x, dict) or not str(x.get("issue") or "").strip():
                v.append(f"{key} {i}: no description")
            elif str(x.get("severity") or "").lower() not in ("low", "medium", "high"):
                v.append(f"{key} {i}: severity {x.get('severity')!r}")
    return v


def measure(L: dict, dataset: str | None, config: dict | None = None) -> dict:
    """The counts one parsed response contributes."""
    tl = [s for s in _list(L, "timeline") if isinstance(s, dict) and s.get("start_s") is not None]
    ke = [k for k in _list(L, "key_events") if isinstance(k, dict)]
    m = {"segments": len(tl), "key_events": len(ke),
         "subgoals": sum(1 for k in ke if str(k.get("kind") or "").lower() == "subgoal_complete"),
         "violations": schema_violations(L, config)}
    fams = set()
    for key in LISTS:
        items = [i for i in _list(L, key) if isinstance(i, dict) and str(i.get("issue") or "").strip()]
        n = sum(1 for i in items if FAMILIES.counts(key, {**i, "severity": str(i.get("severity") or "low").lower()}))
        m[key] = n
        m[key + "_minor"] = len(items) - n
        fams |= {FAMILIES.family_of(key, i, dataset) for i in items}
    m["families"] = sorted(fams)
    tasks = [t for t in _list(L, "tasks") if isinstance(t, dict)]
    if tasks:
        m["outcome"] = None
        m["tasks"] = [len(tasks), sum(1 for t in tasks if str(t.get("outcome") or "").lower() == "success")]
    else:
        c = L.get("completion")
        oc = str(c.get("task_completed") or "").lower() if isinstance(c, dict) else ""
        # partial is a kind of failure (board/to_board.py), so two runs agree when both say the task was not done
        oc = "failure" if oc == "partial" else oc
        m["outcome"] = oc if oc in OUTCOMES else (oc or None)
    return m


# ---------------------------------------------------------------- the summary

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def _r(x, n=4):
    return None if x is None else round(x, n)


def metric_values(models: list, eps: list, rows: dict) -> dict:
    """The per-episode metrics of these models on these episodes (every model parsed each one)."""
    out = {}
    mins = sum((rows[e]["seconds"] or 0) for e in eps) / 60
    for m in models:
        rs = [rows[e]["by"][m] for e in eps]
        segs = sum(r["segments"] for r in rs)
        out[m] = {"n": len(eps), "minutes": _r(mins, 2),
                  "events_per_min": _r(segs / mins if mins else None, 3),
                  "key_events": _r(_mean([r["key_events"] for r in rs]), 3),
                  "subgoals": _r(_mean([r["subgoals"] for r in rs]), 3),
                  "data_issues": _r(_mean([r["data_issues"] for r in rs]), 3),
                  "data_issues_minor": _r(_mean([r["data_issues_minor"] for r in rs]), 3),
                  "operator_mistakes": _r(_mean([r["operator_mistakes"] for r in rs]), 3),
                  "operator_mistakes_minor": _r(_mean([r["operator_mistakes_minor"] for r in rs]), 3)}
    return out


def agreement(a: str, b: str, eps: list, rows: dict) -> dict:
    """Outcome agreement and issue-type agreement of two models on the episodes both parsed."""
    both = [e for e in eps if rows[e]["by"].get(a, {}).get("status") == "parsed"
            and rows[e]["by"].get(b, {}).get("status") == "parsed"]
    oc = [(rows[e]["by"][a]["outcome"], rows[e]["by"][b]["outcome"]) for e in both]
    oc = [(x, y) for x, y in oc if x and y]
    inter = union = 0
    for e in both:
        fa, fb = set(rows[e]["by"][a]["families"]), set(rows[e]["by"][b]["families"])
        inter += len(fa & fb)
        union += len(fa | fb)
    return {"outcome": _r(sum(x == y for x, y in oc) / len(oc)) if oc else None, "outcome_n": len(oc),
            "issues": _r(inter / union) if union else None, "issues_n": len(both), "issues_shared": inter,
            "issues_union": union}


def _asked(rec: dict | None) -> bool:
    """Whether a model's record of an episode is a finished request (anything but pending or absent)."""
    return (rec or {}).get("status") not in (None, "pending")


def _minutes(eps: list, rows: dict) -> float | None:
    return _r(sum(rows[e]["seconds"] or 0 for e in eps) / 60, 2)


def _responses(k: str, eps: list, rows: dict) -> dict:
    """One model's response counts, parse share, schema violations, cost and latency over these episodes."""
    mine = [rows[e]["by"][k] for e in eps if k in rows[e]["by"]]
    st = defaultdict(int)
    for rec in mine:
        st[rec["status"]] += 1
    resp = sum(st[x] for x in RESPONDED)
    costs = [rec.get("cost") for rec in mine if rec["status"] in RESPONDED]
    lats = [rec.get("latency") for rec in mine if rec["status"] in RESPONDED]
    viol = [len(rec["violations"]) for rec in mine if rec["status"] == "parsed"]
    return {"asked": len(mine), **{x: st[x] for x in RESPONDED + ("no_response", "pending")},
            "parse_share": _r(st["parsed"] / resp) if resp else None,
            "violations": _r(_mean(viol), 3),
            "violations_any": _r(sum(1 for x in viol if x) / len(viol)) if viol else None,
            "violations_n": len(viol),
            "cost": _r(_mean(costs), 4), "cost_n": len([c for c in costs if c is not None]),
            "latency": _r(_mean(lats), 1), "latency_n": len([x for x in lats if x is not None])}


def _with_example(m: dict, ref: str | None, eps: list, rows: dict) -> dict | None:
    """A with-example run against the same model's run without the example, on the episodes both were asked, and
    the reference on those of them it has a label of."""
    ex, base = m["key"], m["base"]
    asked = [e for e in eps if _asked(rows[e]["by"].get(ex)) and _asked(rows[e]["by"].get(base))]
    if not asked:
        return None
    both = [e for e in asked if rows[e]["by"][ex]["status"] == "parsed" and rows[e]["by"][base]["status"] == "parsed"]
    with_ref = [e for e in both if ref and rows[e]["by"].get(ref, {}).get("status") == "parsed"]
    mv = metric_values([base, ex], both, rows) if both else {}

    def side(k):
        st = [rows[e]["by"][k]["status"] for e in asked]
        resp = sum(1 for x in st if x in RESPONDED)
        rs = [rows[e]["by"][k] for e in asked if rows[e]["by"][k]["status"] in RESPONDED]
        viol = [len(rows[e]["by"][k]["violations"]) for e in both]
        ag = agreement(k, ref, with_ref, rows) if with_ref else {}
        return {"parse_share": _r(st.count("parsed") / resp) if resp else None,
                "violations": _r(_mean(viol), 3),
                "cost": _r(_mean([r.get("cost") for r in rs]), 4),
                "latency": _r(_mean([r.get("latency") for r in rs]), 1),
                **{f: (mv.get(k) or {}).get(f) for f in PER_EPISODE},
                "agree_outcome": ag.get("outcome"), "agree_issues": ag.get("issues")}
    ref_values = metric_values([ref], with_ref, rows)[ref] if with_ref else None
    # the reference's values on the same episodes, for the comparison view's "with an example" chart
    return {"base": base, "with": ex, "asked": len(asked), "n": len(both), "minutes": _minutes(both, rows),
            "without_values": side(base), "with_values": side(ex),
            "reference": {f: ref_values.get(f) for f in PER_EPISODE} if ref_values else None,
            "reference_n": len(with_ref)}


def compute(manifest: dict, base: Path | None = None, sources: dict | None = None) -> dict:
    """Every number of the comparison view, from the manifest's comparison runs and the board's own labels (empty
    when it names no comparison). sources is {board file: the run output its label was built from}; board/build.py
    passes the one it built, and otherwise board/build.py's label_sources works it out from the manifest. The
    "_rows" entry is per episode and per model, for board/build.py; public() leaves it out."""
    models = load_models(manifest, base)
    if not models:
        return {}
    if sources is None:
        from board.build import label_sources   # board.build imports this module, so imported here
        sources = label_sources(manifest, base or Path.cwd())
    bidx = board_index(manifest, base)
    rows, idents = {}, {}          # rows: board file name -> the episode and each model's record
    for m in models:
        for name in slice_episodes(m["episodes"]):
            d = (m["episodes"] / name).resolve()
            if d not in idents:
                idents[d] = episode_identity(m["episodes"], name, bidx)
            ident = {**idents[d], "name": name}
            if ident["file"] not in sources:
                continue           # the board has no label of this episode to set the model against
            row = rows.setdefault(ident["file"], {**ident, "by": {}})
            r = response(m, name)
            rec = {"status": r["status"],
                   **{f: r[f] for f in ("cost", "latency", "out_tokens") if r.get(f) is not None}}
            if r["status"] == "parsed":
                rec.update(measure(r["labels"], row["dataset"], r.get("config")))
            row["by"][m["key"]] = rec
    # the reference: the board's own label of every episode a comparison run was asked
    seen = set()
    for row in rows.values():
        src = sources.get(row["file"])
        if src is None or not Path(src).exists():
            continue
        r = read_output(Path(src))
        seen.add(r.get("model"))
        rec = {"status": r["status"], **{f: r[f] for f in ("cost", "latency", "out_tokens") if r.get(f) is not None}}
        if r["status"] == "parsed":
            rec.update(measure(r["labels"], row["dataset"], r.get("config")))
        row["by"][REFERENCE_KEY] = rec
    models = [reference_model(seen)] + models
    ref = REFERENCE_KEY
    order = [m["key"] for m in models]
    main = [k for k in order if not next(m for m in models if m["key"] == k)["example"]]
    keys = [m["key"] for m in models]
    by_key = {m["key"]: m for m in models}

    def rig_eps(rig, want=None):
        return sorted(k for k, r in rows.items() if (rig == "all" or r["rig"] == rig)
                      and (want is None or any(_asked(r["by"].get(x)) for x in want)))

    summary, paired = {}, {}
    for rig in RIG_NAMES:
        eps = rig_eps(rig, main)
        if not eps:
            continue
        common = [e for e in eps if all(rows[e]["by"].get(k, {}).get("status") == "parsed" for k in main)]
        summary[rig] = {
            "episodes": len(eps), "minutes": _minutes(eps, rows),
            "responses": {k: _responses(k, rig_eps(rig), rows) for k in keys},
            "common": len(common), "common_minutes": _minutes(common, rows),
            "metrics": metric_values(main, common, rows),
            "agreement": {a: {b: agreement(a, b, rig_eps(rig), rows) for b in order if b != a} for a in order}}
        pairs = [p for m in models if m["example"] and m["base"] in by_key
                 for p in [_with_example(m, ref, rig_eps(rig), rows)] if p]
        if pairs:
            paired[rig] = pairs
    kept = ("status", "outcome", "tasks", "segments", "cost")
    episodes = [{"file": r["file"], "episode_id": r["episode_id"], "dataset": r["dataset"], "rig": r["rig"],
                 "seconds": _r(r["seconds"], 2),
                 "by": {m: {x: v for x, v in rec.items() if x in kept} for m, rec in r["by"].items()}}
                for k, r in sorted(rows.items(), key=lambda kv: (kv[1]["rig"], str(kv[1]["dataset"]), kv[0]))]
    return {"generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "models": [{k: (str(v) if isinstance(v, Path) else v) for k, v in by_key[key].items()
                        if k not in ("run", "episodes")} for key in order],
            "main": main, "rig_names": RIG_NAMES,
            "summary": summary, "paired": paired, "episodes": episodes, "_rows": rows}


def public(metrics: dict) -> dict:
    """What the board serves: everything but the internal per-episode rows."""
    return {k: v for k, v in metrics.items() if not k.startswith("_")}


def _fmt(x, fmt: str) -> str:
    return fmt % x if x is not None else "-"


def print_summary(m: dict) -> None:
    """Per rig, one line per model: parse share, schema violations per parsed response, then the per-episode
    counts on the episodes every model parsed, cost and latency per response."""
    names = {x["key"]: x["name"] for x in m["models"]}
    for rig, s in m["summary"].items():
        print(f"\n== {RIG_NAMES[rig]}: {s['episodes']} episodes, {s['minutes']:.0f} min; "
              f"every model parsed {s['common']}")
        print(f"{'model':46} {'parse':>7} {'viol':>6} {'ev/min':>7} {'key':>5} {'subg':>5} {'data':>5} {'mist':>5} "
              f"{'$/ep':>7} {'s/ep':>6}")
        for k, p in s["responses"].items():
            mv = s["metrics"].get(k) or {}
            share = None if p["parse_share"] is None else 100 * p["parse_share"]
            print(f"{names[k][:46]:46} {_fmt(share, '%6.1f%%'):>7} {_fmt(p['violations'], '%.2f'):>6} "
                  f"{_fmt(mv.get('events_per_min'), '%.2f'):>7} {_fmt(mv.get('key_events'), '%.1f'):>5} "
                  f"{_fmt(mv.get('subgoals'), '%.1f'):>5} {_fmt(mv.get('data_issues'), '%.2f'):>5} "
                  f"{_fmt(mv.get('operator_mistakes'), '%.2f'):>5} {_fmt(p['cost'], '%.3f'):>7} "
                  f"{_fmt(p['latency'], '%.0f'):>6}")


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m compare.metrics", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("board", type=Path, help="a board folder whose manifest.json names comparisons")
    ap.add_argument("--json", type=Path, help="also write the metrics (what the board's comparison view shows)")
    a = ap.parse_args()
    res = compute(json.loads((a.board / "manifest.json").read_text()), a.board)
    if not res:
        raise SystemExit(f"{a.board / 'manifest.json'} names no comparisons")
    print_summary(res)
    if a.json:
        a.json.write_text(json.dumps(public(res), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
