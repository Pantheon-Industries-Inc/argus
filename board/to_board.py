"""Convert a run's per-episode outputs (label/harness.py) into the board's episode files.

    python -m board to_board --in-dir RUN/out --out-dir BOARD_FILES --dataset galaxea

The board plays each episode's cameras and syncs its annotation to them: the dense timeline (each segment
becomes a marker at its start, with its end, arm, action, object, contribution and progress), key events,
state changes, the scene graph, recovery, the outcome and goal times, the goal alignment, data issues and
operator mistakes, plus what the harness recorded about the request (sampled instants, cameras, deterministic
checks, still spans, usage). A dry run's outputs and replies that did not parse are skipped: neither is a label.
board/build.py converts the labels of every run a board's manifest names (label_outputs), then adds the episode's
context.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


LISTS = ("timeline", "key_events", "state_changes", "scene_graph", "recovery", "data_issues", "operator_mistakes",
         "tasks")


def _time(x) -> float | None:
    try:
        return round(float(x), 3)
    except (TypeError, ValueError):
        return None


def convert(result: dict, dataset: str | None = None) -> dict:
    """One harness output as a board episode file.

    Every model is given the same schema, but a parsed reply can still break it (key events written as plain
    strings, a time that is not a number). Such an entry is not shown: each list keeps only its objects, and
    `_off_schema` counts per list what was left out, so a reply that breaks the schema never breaks the board and
    never hides that it did. The run's own output is unchanged."""
    labels = dict(result.get("labels") or {})
    off = {}
    for key in LISTS:
        v = labels.get(key)
        if isinstance(v, list):
            kept = [x for x in v if isinstance(x, dict)]
            if len(kept) < len(v):
                off[key] = len(v) - len(kept)
            labels[key] = kept
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
        t0 = _time(s.get("start_s"))
        if t0 is None:
            continue
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
            # the labeler's main verb for the task (label/prompts.py), which the home page counts
            "verb": t.get("verb"),
            "objects": t.get("objects") or [],
            # a task partly done is a failure of that task, of the kind partial (as the episode's outcome above)
            "outcome": "failure" if oc == "partial" else oc,
            **({"failure_kind": "partial"} if oc == "partial" else {}),
            "success_predicate": t.get("success_predicate") or "",
            "completed_at_s": t.get("completed_at_s"),
            "note": t.get("note") or "",
            # a task carried across one of our cuts keeps each part's own entry (label/pieces.py join_across_cuts)
            **({"joined_from": t["joined_from"]} if t.get("joined_from") else {}),
        })
    key_events = []
    for k in labels.get("key_events") or []:
        t = _time(k.get("t_s"))
        if t is None:
            continue
        key_events.append({"t_s": t,
                           "label": k.get("label") or "",
                           "kind": k.get("kind") or "",
                           "outcome": k.get("outcome") or "",
                           "note": k.get("note") or ""})

    return {
        "dataset": dataset,
        "episode_prompt": labels.get("task_summary") or "",
        "task_verb": labels.get("task_verb"),   # the labeler's main verb for the whole task, where it is one task
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
    }


def label_outputs(in_dir: Path) -> tuple[dict, list[str]]:
    """The outputs in a run's out/ folder that are labels, {episode folder name: (output file, output)}: every
    episode_*.json that is neither a dry run nor an unparsed reply. Also one line per output skipped. failed_*.json
    files keep a cut-off reply and are never read."""
    outs, skipped = {}, []
    for f in sorted(in_dir.glob("episode_*.json")):
        try:
            r = json.loads(f.read_text())
        except ValueError as e:
            skipped.append(f"{f.name}: {e}")
            continue
        if r.get("dry_run") or r.get("parse_ok") is False:
            skipped.append(f"{f.name}: {'dry run' if r.get('dry_run') else 'unparsed model reply'}")
            continue
        outs[Path(r.get("episode_dir", f.stem)).name] = (f, r)
    return outs, skipped


def convert_run(in_dir: Path, out_dir: Path, dataset: str) -> tuple[int, list[str]]:
    """Every label in a run's out/ folder (label_outputs) as a board file in out_dir, named after the episode
    folder. Returns the count written and one line per output skipped."""
    out_dir.mkdir(parents=True, exist_ok=True)
    outs, skipped = label_outputs(in_dir)
    for eid, (_, r) in outs.items():
        (out_dir / f"{eid}.json").write_text(json.dumps(convert(r, dataset), indent=2))
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
