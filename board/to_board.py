"""Convert a run's per-episode outputs (label/harness.py) into the board's episode files.

    python -m board to_board --in-dir RUN/out --out-dir BOARD_FILES --dataset galaxea

The board plays each episode's cameras and syncs its annotation to them: the dense timeline (each segment
becomes a marker at its start, with its end, arm, action, object, contribution and progress), key events,
state changes, the scene graph, recovery, the outcome and goal times, the goal alignment, data issues and
operator mistakes, plus what the harness recorded about the request (sampled instants, cameras, deterministic
checks, still spans, usage). A reply that did not parse, or was cut off at the output limit (failed_<episode>.json),
is still the episode's file: no labels, the reply kept in _label_failed, and the episode's footage, checks and sensors
shown as for any other. A dry run's outputs are skipped, since a dry run asked nothing. board/build.py converts the
replies of every run a board's manifest names (label_outputs), then adds the episode's context.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path


def _time(x) -> float | None:
    try:
        t = round(float(x), 3)
    except (TypeError, ValueError):
        return None
    return t if math.isfinite(t) else None


def finite(x):
    """x with every number that is not finite (NaN, inf) as null, at any depth. JSON has no such number: json.dumps
    writes NaN, which the page cannot parse, so one NaN in a reply or a check would stop the whole episode loading."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite(v) for v in x]
    return x


def dumps(x, **kw) -> str:
    """Every board file and every response the board serves as JSON: non finite numbers written as null (finite), and
    allow_nan off, so anything that still slips through fails here rather than on the page."""
    return json.dumps(finite(x), allow_nan=False, **kw)


RAW_HEAD = 3000       # characters of a reply that did not parse kept on the board (the run's output keeps it whole)
TAIL = 1500           # characters of the end of a cut-off reply kept on the board


def label_failed(result: dict) -> dict | None:
    """How a reply that gave no labels came back, for the page (board/serve.py cmpFailHtml): cut off at the output limit
    (the harness's failed_<episode>.json, with the tokens it ran to and the end of the reply), or not parsing (the
    parser's error, the start of the reply and its length), or an output file that does not read (unreadable, with
    the error), or a long recording none of whose parts gave labels (no_part, each part with why and the start of its
    reply), or an episode the model gave no reply for (no_reply, with why: no answer, the spend cap reached, a request
    that could not be built; label/harness.py run_batch), or a reply the board could not read although it parsed
    (not_shown, with the error and the start of the reply; board/build.py). None for any other output, a reply that
    parsed or one written without parse_ok (by hand, or by an older harness), which is its labels."""
    if result.get("unreadable"):
        return {"status": "unreadable", "error": str(result["unreadable"])[:RAW_HEAD]}
    if result.get("no_reply"):
        return {"status": "no_reply", "why": str(result["no_reply"])[:RAW_HEAD]}
    if result.get("board_error"):
        raw = str(result.get("raw") or "")
        return {"status": "not_shown", "error": str(result["board_error"])[:RAW_HEAD], "raw_head": raw[:RAW_HEAD],
                "raw_chars": len(raw)}
    st = result.get("stitched") if isinstance(result.get("stitched"), dict) else {}
    if result.get("no_part") and st.get("missing"):
        # a long recording none of whose parts gave labels (label/pieces.py unlabelled): each part and why
        return {"status": "no_part", "parts": [{k: g.get(k) for k in ("part", "t0_s", "t1_s", "why", "raw_head")
                                                if k in g} for g in st["missing"] if isinstance(g, dict)]}
    if result.get("finish_reason") == "length" and "labels" not in result:
        return {"status": "cut_off", "out_tokens": (result.get("usage") or {}).get("completion_tokens"),
                "tail": (result.get("content_tail") or "")[-TAIL:]}
    if result.get("parse_ok") is not False:
        return None
    lab = result.get("labels") if isinstance(result.get("labels"), dict) else {}
    raw = str(lab.get("_raw") or "")
    return {"status": "unparsed", "parse_error": lab.get("_parse_error"), "raw_head": raw[:RAW_HEAD],
            "raw_chars": len(raw)}


def typed(result: dict) -> tuple[dict, list[dict]]:
    """(labels, dropped): a parsed reply's labels with every field of the type the output format gives it
    (label/harness.py typed_labels, which a reply parsed before it was written never went through), and what was left
    out for breaking the format ({"field", "row"?, "why"} per field or row). The run's own output is unchanged."""
    from label.harness import typed_labels
    labels = typed_labels(copy.deepcopy(result.get("labels") if isinstance(result.get("labels"), dict) else {}))
    return labels, [x for x in labels.pop("_dropped", None) or [] if isinstance(x, dict)]


# the fields of the output format whose plain words are plural (the key events are, the timeline is), and the fields
# whose plain words are not their own words with spaces
PLURAL_FIELDS = frozenset(("key_events", "state_changes", "data_issues", "operator_mistakes", "tasks", "contacts",
                           "contacts_missing", "instruction_variants", "scene.objects"))
FIELD_WORDS = {"contacts_missing": "missing contacts"}


def field_words(field: str) -> str:
    """A field of the output format in plain words for the page: key_events is "key events", scene.objects is
    "scene objects", contacts_missing is "missing contacts"."""
    return FIELD_WORDS.get(str(field)) or str(field).replace("_", " ").replace(".", " ")


def and_list(items: list[str]) -> str:
    """Items as a sentence lists them: "a", "a and b", "a, b and c"."""
    return ", ".join(items[:-1]) + f" and {items[-1]}" if len(items) > 1 else items[0]


def off_schema_text(dropped: list[dict]) -> str | None:
    """What of a parsed reply the board leaves out for breaking the output format (typed's dropped), as the subject and
    verb of a sentence, or None when nothing was: "2 rows of the key events and the task summary (a list, not text)
    are". One field takes the verb its own number takes (PLURAL_FIELDS: the tasks are, the timeline is)."""
    rows, whole = {}, []
    for x in dropped:
        if "row" in x:
            rows[x["field"]] = rows.get(x["field"], 0) + 1
        else:
            whole.append(x)
    each = [f"{n} {'row' if n == 1 else 'rows'} of the {field_words(k)}" for k, n in rows.items()]
    each += [f"the {field_words(x['field'])} ({x['why']})" for x in whole]
    if not each:
        return None
    if len(each) > 1:
        verb = "are"
    elif rows:
        verb = "is" if next(iter(rows.values())) == 1 else "are"
    else:
        verb = "are" if whole[0]["field"] in PLURAL_FIELDS else "is"
    return f"{and_list(each)} {verb}"


def convert(result: dict, dataset: str | None = None) -> dict:
    """One harness output as a board episode file. A reply that gave no labels (label_failed) is an episode file with
    empty lists and the reply under _label_failed.

    Every model is given the same schema, but a parsed reply can still break it (key events written as plain
    strings, a list written as an object, a time that is not a number). A row or field of the wrong type is left out
    (typed), and `_off_schema` counts per field what was left out, so a reply that breaks the schema never breaks the
    board and never hides that it did (board/build.py flags it). A step or key event whose time is not a number is kept
    with t_s null, and the page lists it untimed after the timed ones. The run's own output is unchanged."""
    failed = label_failed(result)
    labels, dropped = ({}, []) if failed else typed(result)
    off = {}
    for x in dropped:
        off[str(x.get("field"))] = off.get(str(x.get("field")), 0) + 1
    scene = labels.get("scene") if isinstance(labels.get("scene"), dict) else {}
    timeline = labels.get("timeline") or []
    eid = Path(result.get("episode_dir", "")).name

    objects = []
    for o in scene.get("objects") or []:
        if not isinstance(o, dict):
            continue
        attrs = [str(a) for a in o.get("attributes") or []]
        color = next((a for a in attrs
                      if any(c in a.lower() for c in
                             ("red", "blue", "green", "black", "white",
                              "clear", "yellow", "grey", "gray", "orange"))),
                     None)
        objects.append({"name": o.get("name") or "?", "color": color})

    event_labels = []
    for i, s in enumerate(timeline):
        t0 = _time(s.get("start_s"))          # None: kept, untimed
        dest = s.get("destination") or s.get("spatial_relation")
        event_labels.append({
            "t_s": t0,
            "event_idx": i,
            "arm": s.get("arm") or "",
            "verb_class": s.get("action") or "",
            "object": s.get("object") or "",
            "carry_phase": (f"-> {dest}" if dest else ""),
            "contribution": s.get("contribution") or "",
            "progress": s.get("progress"),
            "end_s": s.get("end_s"),
            "confidence": s.get("confidence"),
            # head camera: per step whether the person's hands are in the frame, so the board can mark
            # stretches inferred while unseen, and what covers them (gloves) when visible; absent elsewhere
            "hands_visible": s.get("hands_visible"),
            "hands_wearing": s.get("hands_wearing"),
        })

    raw_comp = labels.get("completion") if isinstance(labels.get("completion"), dict) else {}
    completion = {
        "task_completed": raw_comp.get("task_completed"),
        "success_predicate": raw_comp.get("success_predicate"),
        "completed_at_s": raw_comp.get("completed_at_s"),
        # success_then_undone: the goal WAS reached (goal_reached_at_s) and later undone
        # (undone_at_s, undone_by); for a plain success goal_reached_at_s == completed_at_s.
        "goal_reached_at_s": raw_comp.get("goal_reached_at_s"),
        "undone_at_s": raw_comp.get("undone_at_s"),
        "undone_by": raw_comp.get("undone_by"),
        "reason": raw_comp.get("reason"),
    }
    # the outcome is success or failure. "partial" (a real part of the goal left undone) is a kind of failure: the board
    # records it as a failure and keeps the kind, as success_then_undone is kept as a kind of success
    if str(completion["task_completed"] or "").lower() == "partial":
        completion["task_completed"], completion["failure_kind"] = "failure", "partial"
    # head-camera clips have no single completion: they carry a `tasks` list of activities, each with its
    # own outcome and goal frame, which the board shows as a Tasks panel
    tasks = []
    for t in labels.get("tasks") or []:
        oc = str(t.get("outcome") or "").lower()
        tasks.append({
            "start_s": t.get("start_s"), "end_s": t.get("end_s"),
            "task": t.get("task") or "",
            "objects": t.get("objects") or [],
            # a task partly done is a failure of that task, of the kind partial (as the episode's outcome above)
            "outcome": "failure" if oc == "partial" else oc,
            **({"failure_kind": "partial"} if oc == "partial" else {}),
            "success_predicate": t.get("success_predicate") or "",
            "completed_at_s": t.get("completed_at_s"),
            "note": t.get("note") or "",
        })
    key_events = []
    for k in labels.get("key_events") or []:
        t = _time(k.get("t_s"))               # None: kept, untimed
        key_events.append({"t_s": t,
                           "label": k.get("label") or "",
                           "kind": k.get("kind") or "",
                           "outcome": k.get("outcome") or "",
                           "note": k.get("note") or ""})

    return {
        "dataset": dataset,
        "episode_prompt": labels.get("task_summary") or "",
        "viewpoint": labels.get("viewpoint"),  # ego: first_person | third_person
        "objects": objects,
        "event_labels": event_labels,
        "key_events": key_events,
        "state_changes": labels.get("state_changes") or [],
        "scene_graph": labels.get("scene_graph") or [],
        "recovery": labels.get("recovery") or [],
        "instruction_variants": labels.get("instruction_variants") or [],
        "performance_review": labels.get("performance_review") or "",
        # with an instruction or annotation: how the model's independent reading relates to it (aligned,
        # narrower, broader, different, unrelated); None when there is nothing to compare against
        "goal_alignment": labels.get("goal_alignment") if isinstance(labels.get("goal_alignment"), dict) else None,
        "data_issues": labels.get("data_issues") or [],
        # recording is faithful but the demonstration was performed poorly (failed grasp, drop,
        # struggle); kept apart from data_issues, which are faults of the recording or its label
        "operator_mistakes": labels.get("operator_mistakes") or [],
        "completion": completion,
        "tasks": tasks,
        **({"evidence_inspection": finite(copy.deepcopy(result["evidence_inspection"]))}
           if isinstance(result.get("evidence_inspection"), dict) and result["evidence_inspection"].get("version") == 1 else {}),
        # Supplementary sensor analysis keeps its own model and source provenance.
        **({"sensor_evidence": finite(copy.deepcopy(result["sensor_evidence"]))}
           if isinstance(result.get("sensor_evidence"), dict) and result["sensor_evidence"].get("version") == 1 else {}),
        **({"grip_evidence": finite(copy.deepcopy(result["grip_evidence"]))}
           if isinstance(result.get("grip_evidence"), dict) and result["grip_evidence"].get("version") == 1 else {}),
        **({"tactile_qc": finite(copy.deepcopy(result["tactile_qc"]))}
           if isinstance(result.get("tactile_qc"), dict) and result["tactile_qc"].get("version") == 1 else {}),
        # the model's answer for each contact it was shown, the moments it saw a hand take hold of something no
        # recorded contact covers, and the strips it was shown (board/build.py joins them to the recording's contacts)
        **({"contacts_model": [c for c in labels.get("contacts") or [] if isinstance(c, dict)],
            "contacts_missing": [c for c in labels.get("contacts_missing") or [] if isinstance(c, dict)],
            "contact_views": result.get("contact_views")} if result.get("contact_views") else {}),
        # the still spans the model was told about, the deterministic checks, and the exact instants it was shown
        "arm_still_spans": result.get("arm_still_spans"),
        "dataset_checks": result.get("dataset_checks"),
        "task_label": result.get("task_label"),
        "timesteps_s": (result.get("config") or {}).get("timesteps_s"),
        # which cameras the episode has and their dataset names (FastUMI has no top camera), so the
        # board lays out only the streams that exist
        "camera_views": (result.get("config") or {}).get("views"),
        "camera_labels": (result.get("config") or {}).get("cam_labels"),
        "_meta": {
            "episode_id": eid,
            "engaged_events": len(event_labels),
            "task_completed": completion.get("task_completed"),
            "completed_at_s": completion.get("completed_at_s"),
            "goal_reached_at_s": completion.get("goal_reached_at_s"),
            "model": result.get("model"),
            "reasoning_effort": result.get("reasoning_effort"),
            # the instruction the episode was graded against, if the dataset ships one (else None: the task
            # was inferred); the board shows it beside the model's own task_summary
            "given_prompt": result.get("given_prompt"),
            "prompt_mode": result.get("prompt_mode"),
        },
        "_usage": result.get("usage"),
        **({"_off_schema": off} if off else {}),
        **({"_label_failed": failed} if failed else {}),
    }


def label_outputs(in_dir: Path) -> tuple[dict, list[str]]:
    """Every reply in a run's out/ folder, {episode folder name: (output file, output)}: each episode_*.json that is not
    a dry run, parsed or not, and for an episode with none, the reply cut off at the output limit that the harness kept
    as failed_<episode>.json (read with parse_ok false), else the record of why it got no reply (noreply_<episode>.json,
    label/harness.py run_batch). A file that does not read is its episode's output too, with no labels and the error
    (unreadable, label_failed). Also one line per file skipped: a dry run, which asked nothing."""
    outs, skipped = {}, []
    for f in sorted(in_dir.glob("episode_*.json")):
        try:
            r = json.loads(f.read_text())
        except (OSError, ValueError) as e:
            outs[f.stem] = (f, {"parse_ok": False, "unreadable": f"{f.name}: {type(e).__name__}: {e}"})
            continue
        if r.get("dry_run"):
            skipped.append(f"{f.name}: dry run")
            continue
        outs[Path(r.get("episode_dir", f.stem)).name] = (f, r)
    for f in sorted(in_dir.glob("failed_episode_*.json")):
        name = f.stem[len("failed_"):]
        if (in_dir / f"{name}.json").exists():
            continue                      # the episode has an output, so this is an earlier cut-off reply
        try:
            r = json.loads(f.read_text())
        except (OSError, ValueError) as e:
            outs.setdefault(name, (f, {"parse_ok": False, "unreadable": f"{f.name}: {type(e).__name__}: {e}"}))
            continue
        outs.setdefault(Path(r.get("episode_dir") or name).name, (f, {**r, "parse_ok": False}))
    for f in sorted(in_dir.glob("noreply_episode_*.json")):
        name = f.stem[len("noreply_"):]
        if name in outs or (in_dir / f"{name}.json").exists():
            continue                      # an earlier attempt's record; the episode has a reply now
        try:
            r = json.loads(f.read_text())
        except (OSError, ValueError) as e:
            r = {"no_reply": f"{f.name} does not read ({type(e).__name__}: {e})"}
        outs.setdefault(name, (f, {**r, "parse_ok": False, "no_reply": r.get("no_reply") or "no reason recorded"}))
    return outs, skipped


def convert_run(in_dir: Path, out_dir: Path, dataset: str) -> tuple[int, list[str]]:
    """Every reply in a run's out/ folder (label_outputs) as a board file in out_dir, named after the episode
    folder. Returns the count written and one line per file skipped."""
    out_dir.mkdir(parents=True, exist_ok=True)
    outs, skipped = label_outputs(in_dir)
    for eid, (_, r) in outs.items():
        (out_dir / f"{eid}.json").write_text(dumps(convert(r, dataset), indent=2))
    return len(outs), skipped


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m board to_board", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, required=True, help="a run's out/ folder")
    ap.add_argument("--out-dir", type=Path, required=True, help="where the board's episode files go")
    ap.add_argument("--dataset", required=True, help="the dataset's short name on the board, e.g. galaxea")
    args = ap.parse_args()
    n, skipped = convert_run(args.in_dir, args.out_dir, args.dataset)
    for line in skipped:
        print(f"skip {line}")
    print(f"converted {n} -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
