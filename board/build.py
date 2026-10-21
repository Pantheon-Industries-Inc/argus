"""Build a board from exactly the runs its manifest names.

    python -m board build BOARD          BOARD/manifest.json in; BOARD/qa/, BOARD/BUILT.json out

A board serves the episode files in BOARD/qa. Those files are derived: every build regenerates them from the runs
listed in BOARD/manifest.json and nothing else. Publication exchanges each derived directory when supported;
otherwise per-file replacements keep it available, with rollback on failure. Concurrent readers of that fallback
can briefly see files from both generations. Each file is
the run's label (board/to_board.py) with its provenance (`_run`: run id, code commit, kind, slice), the episode's
rig and length, the deterministic dataset checks and the dataset's own labels from its context.json, where the
footage comes from and its license (`dataset_source`, from board/dataset_sources.json, for the public datasets
prepare/ reads), the manifest's rules applied, and the label consistency check (checks/label_consistency.py:
annotations that contradict themselves are reported in label_consistency, never used to edit a label). A
recording labelled in parts carries the parts it was stitched from and the issues set aside at our cuts
(carry_pieces), and an outcome or severity outside its known values is shown as "unclear" (normalize_enums). An
episode whose model reply did not parse or was cut off is on the board too, with its footage, checks and sensors, no
labels, the reply itself, and a data issue saying so (label_failure); a rerun's reply of that kind never replaces a
label that parsed. So is every episode that got no reply at all (the spend cap reached, a request that could not be
built), saying why, and on an entry whose run labels every episode of its folder (labels_every_episode) every
prepared episode the run never reached; one whose reply the board cannot read, with the reply and the error; and one
whose context.json does not read, with its labels and an issue saying so (context_unreadable): one episode never stops
the build.

manifest.json. Paths are absolute or relative to the board folder; a run given as RUNS/<dataset>/latest is that
dataset's newest finished run that is not a dry run (run ids start with their start time).

    {"board": "quickstart",                                   the board's name
     "datasets": [                                            one entry per dataset, in tab order
       {"dataset": "molmo",                                   the dataset's short name on the board
        "run": "../../runs/molmo/latest",                     a run folder: run.json and out/
        "episodes": "../../episodes/molmo/quickstart",        the prepared episodes the run labelled
        "rules": [...],                                       optional: definitions applied after labelling
        "file_prefix": "...",                                 optional: a prefix for its board file names
        "labels_every_episode": true,                         optional: the run labels every episode of the folder,
                                                              so one it has no reply for is listed (own_data_entry)
        "target": {"episodes": 2682, "hours": 101.9}},        optional: what a run in progress is meant to label;
                                                              the page shows what is labelled against it
       ...],
     "comparisons": [...],                                    optional: other models' runs (compare/metrics.py)
     "hands": {"src": KEYPOINT_RUN, "clips": CLIPS},          optional: hand pose overlay (board/hands.py)
     "sensors": false}                                        optional: no sensors files (board/sensors.py)

configs/quickstart/board.json is a complete manifest for the quickstart runs.

Rules change how a label is shown and counted without any model call. Nothing is deleted: an issue a rule
excludes moves to "_excluded" with the rule's reason, and a capped issue keeps its own severity as
"model_severity". The rules every dataset of a rig needs are in board/rules.py (rules_for); a manifest lists them
explicitly, so the board's inputs are all in one file.
  {"kind": "cap_when", "list": "operator_mistakes", "tags": [...], "severity": "low", "reason": "...",
   "when_tags": [...] | "when_outcome": [...]}
      caps those tags in an episode that also has a medium or high data issue in when_tags (a task "left
      unfinished" judged against an instruction the footage does not match: the instruction mismatch already
      counts the episode), or whose outcome is one of when_outcome (a goal "undone" in an episode whose outcome
      success_then_undone already counts it as a data issue). "when_list" names another list to look in.
  {"kind": "severity_cap", "tags": [...], "severity": "low", "reason": "..."}
      caps those data issue tags on a rig where they are normal (a head-camera wearer pausing between tasks).
  {"kind": "no_task_text", "pattern": REGEX, "reason": "..."}
      in an episode given no task text, moves data issues whose tag matches REGEX (a missing instruction) to
      "_excluded": the model inferred the task, and the missing text is not a fault of the recording.
  {"kind": "drop_check", "check": "gripper_channels", "reason": "..."}
      withholds a deterministic check whose flags are not defects on this dataset, keeping its result under
      "_withheld_checks".
  {"kind": "fixed_window", "tags": [...], "window_s": 180, "tolerance_s": 2.5}
      for a dataset whose recorder cuts continuous footage into fixed-length files (Egocentric-100K's 3-minute
      clips): on an episode within tolerance of the window, issues with those tags (a clip starting or ending
      mid-task) describe how the dataset is packaged, so they move to "_excluded".

"reruns": [{"run": RUN, "why": "..."}] on a dataset entry names later runs over some of its episodes: for each episode
a rerun labelled, its label replaces the base run's (later reruns win), and an episode only a rerun labelled joins
the board from it. A rerun's episode is matched to the entry's by its resolved episode folder, not its name, so a
rerun over another slice (a model comparison's, whose names are the board's file names) replaces exactly that
entry's episodes of it. Each label's _run names the run it came from.

BOARD/additions/<board file>.json, when that folder exists, adds fields to an episode's label without touching the
run's own output: a later pass over the same episodes (a text-only goal comparison) whose results belong beside the
paid label. Each file is
  {"provenance": {"source": "...", "model": "...", ...},     required: where the added content came from
   "fields": {"goal_alignment": {...}},                      optional: fields set where the label has none
   "append": {"data_issues": [{...}]}}                       optional: rows added to the label's list fields
An added field never overwrites one the label (or the build) already has, and an added row is left out when the label's
list already has a row of its category: the label's value stands and the conflict is listed in the episode's addition_conflicts and in BUILT.json, as is an addition that does not read or names no episode
of the board. Every added field that is an object and every added row carries the provenance, and _additions lists
what was added. The addition is part of the episode's build key, so changing it rebuilds that episode.

"file_prefix" is for a dataset whose episode names repeat another dataset's on the same board (HABIT and
MolmoAct2 both have episode_000494): its board files become episode_<prefix>000494.json, and `board clips
--name-prefix` names its clips the same way.

"comparisons" lists other models' runs over some of the board's episodes, in the format compare/metrics.py
describes (python -m compare writes the entries). Their labels go to BOARD/compare/, never into qa/, so nothing
the board counts or exports includes them; the page's "Labels by" control switches the board to one of those
models' labels, marked as a comparison, and its comparison view shows BOARD/compare/metrics.json. The reference
every model is measured against is the board's own label of each episode (label_sources), so every number
compares a model with what the board shows.

"hands" names a folder of 2D hand keypoints of head-camera episodes (board/hand_pose/ makes it) and the clips
folder the board plays. They go to BOARD/hands/, one file per label file, timed against the board's clip, which
the episode page draws over the footage, and to BOARD/hand_keypoints/, a download in the dataset video's own
pixels and frame times (board/hands.py). Neither goes into qa/, so nothing the board counts or exports as labels
includes them. The keypoints are for non-commercial use only, which every file says.

Episodes whose prepared folder has other signals (signals.npz: a force, joint velocities, a pressure map) or depth
streams (depth.json) get a sensors file each in BOARD/sensors/, with BOARD/sensors/index.json listing them
(board/sensors.py), which the episode page draws under its timeline. Nothing in the manifest is needed: the build reads
the dataset entries' "episodes" folders, and a board none of whose episodes has either gets no sensors/ and the same
BUILT.json as before. "sensors": false in the manifest turns it off. Like hands/, nothing that reads qa/ reads it.
"""
from __future__ import annotations

import argparse
import errno
import glob
import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

from board.to_board import (and_list, convert, dumps, field_words, label_failed, label_outputs, off_schema_text,
                            typed)
from checks import label_consistency
from board import scores
from label.atomic import write_atomic
from prepare.frame_rate import episode_fps, episode_seconds as frame_episode_seconds

SEVERITIES = ["low", "medium", "high"]
# the deterministic checks copied from context.json into the episode's dataset_checks
CONTEXT_CHECKS = ("stream_pairing", "recorded_jumps", "gripper_channels", "capture_qc", "sensor_checks")
# the public datasets' publishers and licenses, by the Hub repository an episode's context.json names
SOURCES = {k: v for k, v in json.loads((Path(__file__).resolve().parent / "dataset_sources.json").read_text()).items()
           if not k.startswith("_")}


def capture_names(cq: dict) -> dict:
    """A stored capture-check result with each check named and grouped as checks/capture_qc.py names it now, so a
    renamed check, or a reworded note, reaches the board on the next build without rerunning the checks."""
    from checks.capture_qc import NAMES, refresh_notes
    cq = refresh_notes(cq)
    rows = [{**r, "name": NAMES[r["check"]][0], "group": NAMES[r["check"]][1]}
            if isinstance(r, dict) and r.get("check") in NAMES else r for r in cq.get("checks") or []]
    return {**cq, "checks": rows} if "checks" in cq else cq


def episode_seconds(ctx: dict, ep_dir: Path | None = None) -> float | None:
    """The episode's length (prepare/frame_rate.py episode_seconds: its duration_s, else the span of its capture
    times, else its frame count at its frame rate), or None without one."""
    return frame_episode_seconds(ctx, ep_dir=ep_dir, default=None) or None


def _cap(issues: list, tags: list, cap: str, reason: str) -> list:
    """Issues with one of these tags above the cap, lowered to it; the model's own severity kept."""
    return [{**iss, "severity": cap, "model_severity": iss.get("severity"), "capped_by": reason}
            if (iss or {}).get("category") in tags and iss.get("severity") in SEVERITIES
            and SEVERITIES.index(iss["severity"]) > SEVERITIES.index(cap) else iss
            for iss in issues or []]


def _exclude(d: dict, match, kind: str, reason: str) -> None:
    """Moves the data issues match() selects to d["_excluded"], each with the rule and its reason."""
    keep = []
    for iss in d.get("data_issues") or []:
        if match(iss or {}):
            d.setdefault("_excluded", []).append({**iss, "excluded_by": kind, "reason": reason})
        else:
            keep.append(iss)
    d["data_issues"] = keep


def apply_rules(d: dict, ctx: dict, rules: list, ep_dir: Path | None = None) -> None:
    """The manifest entry's rules on one board file, in order (the module docstring describes each kind)."""
    for rule in rules:
        kind = rule.get("kind")
        if kind == "no_task_text":
            if not ((d.get("_meta") or {}).get("given_prompt") or "").strip():
                rx = re.compile(rule["pattern"], re.I)
                _exclude(d, lambda i: bool(rx.search(str(i.get("category") or ""))), kind, rule["reason"])
        elif kind == "drop_check":
            dc = d.get("dataset_checks") or {}
            if rule["check"] in dc:
                d.setdefault("_withheld_checks", {})[rule["check"]] = {"reason": rule["reason"],
                                                                        "result": dc.pop(rule["check"])}
        elif kind == "cap_when":
            outcome = str((d.get("completion") or {}).get("task_completed") or "").lower()
            when = outcome in (rule.get("when_outcome") or []) or any(
                (i or {}).get("category") in (rule.get("when_tags") or []) and i.get("severity") in ("medium", "high")
                for i in d.get(rule.get("when_list", "data_issues")) or [])
            if when:
                key = rule.get("list", "operator_mistakes")
                d[key] = _cap(d.get(key), rule["tags"], rule["severity"], rule["reason"])
        elif kind == "severity_cap":
            d["data_issues"] = _cap(d.get("data_issues"), rule["tags"], rule["severity"], rule["reason"])
        elif kind == "fixed_window":
            secs = episode_seconds(ctx, ep_dir)
            if secs is None or abs(secs - rule["window_s"]) > rule.get("tolerance_s", 2.5):
                continue
            d["_packaging"] = {"fixed_window_s": rule["window_s"]}
            _exclude(d, lambda i: i.get("category") in rule["tags"], kind,
                     f"the dataset cuts continuous footage into {rule['window_s']:g} s files; "
                     "a file starting or ending mid-task is how it is packaged")
        else:
            raise ValueError(f"unknown rule kind {kind!r}")


# what the reader did not read of an upload, by kind (prepare/formats.py writes these into context["source"])
READER_LEFT_OUT = (("unused_cameras", "cameras"), ("unused_signals", "signals"), ("unused_arrays", "arrays"),
                   ("unused_depth", "depth"))


def reader_notes(ctx: dict) -> dict | None:
    """What the reader says about an episode beyond what it read: an episode read again with its cameras alone, why
    and the parts that left unread (context["read_cameras_only"], prepare/formats.py read_episode), its note on the
    recorded state (state_note) and what of the upload it did not read, each with its reason (context["source"]
    unused_*). The model never saw these. The prompt states them only where they explain an absence (label/episode.py,
    the no state block); the board always shows them, so a field the model never saw is never a silent gap. None when
    there is nothing to say."""
    src = ctx.get("source") if isinstance(ctx.get("source"), dict) else {}
    left = {kind: [str(x) for x in src[key]] for key, kind in READER_LEFT_OUT
            if isinstance(src.get(key), (list, tuple)) and src[key]}
    note = ctx["state_note"].strip() if isinstance(ctx.get("state_note"), str) else ""
    again = ctx.get("read_cameras_only") if isinstance(ctx.get("read_cameras_only"), dict) else None
    if again is not None:
        unread = again.get("left_unread")
        again = {"why": str(again.get("why") or ""),
                 "left_unread": [str(x) for x in unread] if isinstance(unread, (list, tuple)) else []}
    if not note and not left and again is None:
        return None
    return {**({"cameras_only": again} if again is not None else {}), **({"state_note": note} if note else {}),
            **({"left_out": left} if left else {})}


_FAMILIES = None


def label_failure(result: dict | None) -> list[dict]:
    """The data issue of an episode whose model reply gave no labels (board/to_board.py label_failed): one entry of
    kind model_reply_cut_off or model_reply_unparsed, which raises the family label-failed, so the episode is on the
    board with its footage, checks and sensors and the filter finds it. A long recording stitched from the parts that
    parsed (label/pieces.py stitch_run) has one entry of kind part_not_labelled per part that gave none, at its span,
    with the start of that part's reply (reply_head) when it gave one. Nothing for a reply that parsed whole."""
    if not isinstance(result, dict):
        return []
    st = result.get("stitched") if isinstance(result.get("stitched"), dict) else {}
    gaps = [{"kind": "part_not_labelled", "t0_s": g.get("t0_s"), "t1_s": g.get("t1_s"),
             "what": f"Part {g.get('part')} of {st.get('parts')} of this long recording, from {float(g['t0_s']):.1f} s "
                     f"to {float(g['t1_s']):.1f} s, has no labels, as {g.get('why') or 'its reply gave none'}; the "
                     "labels come from the other parts.",
             **({"reply_head": g["raw_head"]} if g.get("raw_head") else {})}
            for g in st.get("missing") or [] if isinstance(g, dict) and g.get("t0_s") is not None
            and g.get("t1_s") is not None] if label_failed(result) is None else []
    lf = label_failed(result)
    if lf is None:
        return gaps + off_schema(result)
    rest = "so this episode has no labels; its footage, checks and sensors are shown as recorded"
    if lf["status"] == "no_part":
        def said(why: str) -> str:
            w = str(why or "")
            # a reason the harness recorded for no reply (the spend cap reached) is kept in brackets
            detail = w[w.index("("):] if "no reply (" in w else ""
            return ("was cut off at the output limit" if "cut off" in w else
                    f"never answered {detail}".strip() if "no reply" in w else
                    "has an output file that does not read" if "does not read" in w else
                    "did not parse" if "parse" in w else w or "gave no labels")
        each = [f"part {g.get('part')} {said(g.get('why'))}" for g in lf["parts"]]
        listed = ", ".join(each[:-1]) + f" and {each[-1]}" if len(each) > 1 else each[0]
        return [{"kind": "no_part_labelled",
                 "what": f"No part of this long recording has labels: {listed}. It is shown with its footage, checks "
                         "and sensors as recorded."}]
    if lf["status"] == "no_reply":
        return [{"kind": "model_no_reply",
                 "what": "The model gave no reply for this episode "
                         f"({str(lf.get('why') or 'no reason recorded').rstrip('.')}), {rest}."}]
    if lf["status"] == "not_shown":
        return [{"kind": "model_reply_not_shown",
                 "what": f"The model's reply parsed but the board could not read it ({lf.get('error')}), {rest}; the "
                         "reply is kept as it came."}]
    if lf["status"] == "unreadable":
        return [{"kind": "label_output_unreadable",
                 "what": f"The labelling run's output file for this episode does not read ({lf.get('error')}), "
                         f"{rest}."}]
    if lf["status"] == "incomplete":
        return [{"kind": "model_reply_incomplete",
                 "what": f"The model's reply lacks work its request asked for ({'; '.join(lf.get('why') or [])}), "
                         f"{rest}."}]
    if lf["status"] == "cut_off":
        n = lf.get("out_tokens")
        return [{"kind": "model_reply_cut_off",
                 "what": f"The model's reply was cut off at the output limit{f' after {n:,} tokens' if n else ''} and "
                         f"did not parse, {rest}."}]
    return [{"kind": "model_reply_unparsed", "what": f"The model's reply did not parse as JSON, {rest}."}]


# the fields every rig's output format asks for (label/prompts.py), named when a reply with no timeline leaves them out
SCHEMA_KEYS = ("timeline", "task_summary", "key_events", "data_issues", "operator_mistakes")


def off_schema(result: dict) -> list[dict]:
    """The data issues of a reply that parsed but broke the output format, each field in plain words: one of kind
    model_reply_off_schema when it has no timeline, or one that is not a list (it gave none of the steps every rig's
    format asks for, so the episode has no steps; what else it left out and what it gave are named, and what it gave
    is shown), and one of kind model_reply_fields_dropped naming the other rows and fields left out of what is shown
    for being of the wrong type (board/to_board.py typed), for a long recording those of every part. Nothing for a
    reply that keeps to the format."""
    if "labels" not in result:
        return []
    out = []
    raw = result["labels"] if isinstance(result["labels"], dict) else {}
    labels, dropped = typed(result)
    if not isinstance(labels.get("timeline"), list) and not result.get("stitched"):
        from label.harness import json_kind
        given = raw.get("timeline")
        lead = ("The model's reply has no timeline" if given is None else
                f"The model's reply gives its timeline as {json_kind(given)}, not a list")
        missing = [field_words(k) for k in SCHEMA_KEYS if k != "timeline" and raw.get(k) is None]
        # what it gives in the output format: a field given as null is not given, and the timeline is named above
        gave = sorted(field_words(k) for k, v in labels.items()
                      if not str(k).startswith("_") and k != "timeline" and v is not None)
        wrong = [x for x in dropped if "row" not in x and x.get("field") != "timeline"]
        rest = (f"gives only the {and_list(gave)}; what it gives is shown." if gave else
                "nothing else it gives keeps to the output format." if wrong else "gives nothing else.")
        said = f"It leaves out the {and_list(missing)}, and {rest}" if missing else (
            f"It {rest}" if gave or not wrong else rest[0].upper() + rest[1:])
        out.append({"kind": "model_reply_off_schema", "what": f"{lead}, so this episode has no steps. {said}"})
        # the timeline is named there once, never again among the fields left out
        dropped = [x for x in dropped if x.get("field") != "timeline" or "row" in x]
    left = off_schema_text(dropped)
    if left:
        out.append({"kind": "model_reply_fields_dropped",
                    "what": f"The model's reply broke the output format, so {left} left out of what is shown; the rest "
                            "of its labels are shown."})
    return out


def steps_outside(d: dict, result: dict | None = None) -> list[dict]:
    """The data issue of a reply whose times do not fit the episode, every time the page draws at the episode's edge:
    kind model_steps_outside_episode, naming how many steps lie past the episode's end, start before it and end before
    they start, and how many key events lie outside it, and for a long recording labelled in parts (result, its
    stitched record) how many steps and key events of each part lie outside that part's own footage, which the stitch
    put in a neighbouring part's (label/pieces.py outside_part). They are kept as the model gave them; the page draws
    the timeline to the episode's length with each time outside it at its edge, which the sentence says only when there
    is one. A time within STEP_SLACK_S of an edge is a rounding. The episode's edges are judged only when its length
    is a known number (not estimated)."""
    from label.episode import STEP_SLACK_S
    dur = d.get("duration_s")
    num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
    one = lambda xs, a, b: a if len(xs) == 1 else b
    said, edge, parts = [], False, []
    if num(dur) and math.isfinite(dur) and dur > 0 and not d.get("duration_estimated"):
        lo, hi = -STEP_SLACK_S, dur + STEP_SLACK_S
        steps = [e for e in d.get("event_labels") or [] if isinstance(e, dict)]
        keys = [k for k in d.get("key_events") or [] if isinstance(k, dict)]
        past = [e for e in steps if any(num(e.get(k)) and e[k] > hi for k in ("t_s", "end_s"))]
        early = [e for e in steps if any(num(e.get(k)) and e[k] < lo for k in ("t_s", "end_s"))]
        back = [e for e in steps if num(e.get("t_s")) and num(e.get("end_s")) and e["end_s"] < e["t_s"]]
        stray = [k for k in keys if num(k.get("t_s")) and not lo <= k["t_s"] <= hi]
        said = [f"{len(past)} of its {len(steps)} steps {one(past, 'lies', 'lie')} past the episode's end at "
                f"{dur:.1f} s" if past else "",
                f"{len(early)} {one(early, 'starts', 'start')} before the episode does" if early else "",
                f"{len(back)} {one(back, 'ends before it starts', 'end before they start')}" if back else "",
                f"{len(stray)} of its {len(keys)} key events {one(stray, 'lies', 'lie')} outside the episode"
                if stray else ""]
        edge = bool(past or early or stray)
        for field, name, time_keys, start_key, end_key in (
                ('data_issues', 'data issue', ('t_s', 'end_s'), 't_s', 'end_s'),
                ('operator_mistakes', 'operator mistake', ('t_s', 'end_s'), 't_s', 'end_s'),
                ('tasks', 'task', ('start_s', 'end_s', 'completed_at_s', 'goal_reached_at_s', 'undone_at_s'),
                 'start_s', 'end_s'),
                ('completion', 'completion', ('completed_at_s', 'goal_reached_at_s', 'undone_at_s'), None, None),
                ('state_changes', 'state change', ('t_s',), None, None),
                ('recovery', 'recovery', ('failure_t_s', 'recovered_at_s'), 'failure_t_s', 'recovered_at_s')):
            value = d.get(field)
            rows = [value] if isinstance(value, dict) else [row for row in value or [] if isinstance(row, dict)]
            outside = [row for row in rows if any(num(row.get(key)) and not lo <= row[key] <= hi
                                                 for key in time_keys)]
            reversed_rows = [row for row in rows if start_key and num(row.get(start_key)) and num(row.get(end_key))
                             and row[end_key] < row[start_key]]
            if outside:
                said.append(f"{len(outside)} {name}{'' if len(outside) == 1 else 's'} "
                            f"{one(outside, 'lies', 'lie')} outside the episode")
            if reversed_rows:
                said.append(f"{len(reversed_rows)} {name}{'' if len(reversed_rows) == 1 else 's'} "
                            f"{one(reversed_rows, 'ends', 'end')} before it starts")
            edge = edge or bool(outside)
    st = (result or {}).get("stitched") if isinstance((result or {}).get("stitched"), dict) else {}
    for p in st.get("outside_part") or []:
        if not isinstance(p, dict) or not (num(p.get("t0_s")) and num(p.get("t1_s"))):
            continue
        n, k = int(p.get("steps") or 0), int(p.get("key_events") or 0)
        what = [f"{n} {'step' if n == 1 else 'steps'}" if n else "", f"{k} key {'event' if k == 1 else 'events'}"
                if k else ""]
        what = [x for x in what if x]
        if what:
            parts.append(f"{' and '.join(what)} of part {p.get('part')} {'lies' if n + k == 1 else 'lie'} outside that "
                         f"part's footage, from {p['t0_s']:.1f} s to {p['t1_s']:.1f} s")
    said = [x for x in said if x] + parts
    if not said:
        return []
    # a part's span holds commas, so a list with one is told with semicolons
    listed = (and_list(said) if not parts else said[0] if len(said) == 1
              else "; ".join(said[:-1]) + f"; and {said[-1]}")
    return [{"kind": "model_steps_outside_episode",
             "what": f"The model's timeline does not fit the episode: {listed}. They are kept as given"
                     + (", and the timeline is drawn to the episode's length with the times outside it at its edge."
                        if edge else ".")}]


def reader_issues(ctx: dict, result: dict | None = None) -> list[dict]:
    """The problems an episode was kept and flagged with, each with the family it raises (board/families.py
    reader_family), so the page shows each as a data issue under that family's name: context.json reader_issues
    ({"kind", "what", and optionally "camera", "signal", "t0_s", "t1_s"}: a camera clip shorter than the episode or a
    camera that does not decode among them, board/clips.py record_cameras), then a model reply that gave no labels
    (label_failure), then the stretches the labelling run could not decode a camera's file at, from its record
    (decode_failed, label/episode.py decode_failures), as kind camera_decode_failed. An entry without a kind and a
    sentence says nothing and is left out."""
    global _FAMILIES
    if _FAMILIES is None:
        from board.families import Families
        _FAMILIES = Families()
    from board.public_export import reader_failure_what
    failures = label_failure(result)
    for issue in failures:
        if public_what := reader_failure_what(issue['kind']):
            issue['_reader_public_what'] = public_what
    found = list(ctx.get("reader_issues") or []) + failures
    found += [{"kind": "camera_decode_failed", **x} for x in (result or {}).get("decode_failed") or []
              if isinstance(x, dict)]
    return [{**x, "family": _FAMILIES.reader_family(str(x["kind"]))} for x in found
            if isinstance(x, dict) and x.get("kind") and isinstance(x.get("what"), str) and x["what"].strip()]


def add_reader_issues(d: dict, ctx: dict, result: dict | None = None) -> None:
    """The episode's reader_issues (reader_issues) into its dataset_checks, when it has any."""
    issues = reader_issues(ctx, result)
    if issues:
        d["dataset_checks"] = d.get("dataset_checks") or {}
        d["dataset_checks"]["reader_issues"] = issues


def dataset_label(s: dict) -> dict:
    """One of the dataset's timed labels for the board: its start and end in seconds, a label with no end time a
    moment (its end its start), and a time that is not a number null, which the page shows untimed."""
    from label.episode import number
    t0, t1 = number(s.get("t0")), number(s.get("t1"))
    # OpenAoE labels stored before prepare/openaoe.py hand_phrase
    return {"t0": t0, "t1": t1 if t1 is not None or s.get("t1") is not None else t0,
            "label": str(s["label"]).replace("(both hand)", "(both hands)")}


def add_context(d: dict, ctx: dict, ep_dir: Path, result: dict | None = None) -> None:
    """What the episode's context.json adds to its label: the rig, the real length, the deterministic checks, the
    dataset's own labels (timed segments, as OpenAoE, Galaxea and Gen-HumanEgo ship them, and episode-level status
    and spans, as HABIT does), so a claim that they disagree with the footage can be judged on the board, and the
    dataset's publisher and license, which travel with its labels into every download. result is the labelling run's
    own output, for the contacts it found when the context has none (add_contacts)."""
    from prepare.formats import clock_context
    ctx = clock_context(ctx)
    d["_rig"] = ctx.get("profile")
    mounting = {view: camera['mounting'] for view, camera in (ctx.get('cameras') or {}).items()
                if camera.get('mounting')}
    if mounting:
        d['camera_mounting'] = mounting
    if ctx.get("dataset") in SOURCES:
        d["dataset_source"] = SOURCES[ctx["dataset"]]
    # the sampled timesteps end before the last frame and are sparse in still spans, so the length comes from
    # the episode itself
    secs = episode_seconds(ctx, ep_dir)
    if secs:
        d["duration_s"] = round(secs, 3)
    for key in CONTEXT_CHECKS:
        if ctx.get(key) is not None:
            d["dataset_checks"] = d.get("dataset_checks") or {}
            d["dataset_checks"][key] = capture_names(ctx[key]) if key == "capture_qc" else ctx[key]
    add_reader_issues(d, ctx, result)
    outside = [{**x, "family": _FAMILIES.reader_family(x["kind"])} for x in steps_outside(d, result)]
    if outside:
        d["dataset_checks"] = d.get("dataset_checks") or {}
        d["dataset_checks"]["reader_issues"] = (d["dataset_checks"].get("reader_issues") or []) + outside
    subs = [s for s in ctx.get("annotation_subtasks") or [] if isinstance(s, dict) and s.get("label")]
    if subs:
        d["dataset_labels"] = [dataset_label(s) for s in subs]
        if ctx.get("annotation_note"):
            d["dataset_labels_note"] = ctx["annotation_note"]
    if isinstance(ctx.get("publisher_labels"), dict):
        d["dataset_episode_labels"] = ctx["publisher_labels"]
    notes = ctx.get("uploader_notes")
    if notes is None and ctx.get("uploader_annotation"):
        try:                       # an episode prepared before uploader_notes kept them only as the prompt's text
            notes = json.loads(ctx["uploader_annotation"])
        except ValueError:
            notes = ctx["uploader_annotation"]
    groups = uploader_groups(notes, ctx.get("clock_start_s"), d.get("duration_s"))
    if groups:
        d["uploader_notes"] = groups
    rn = reader_notes(ctx)
    if rn:
        d["reader_notes"] = rn
    # the cameras the model is not shown, which board clips cut like any other (board/clips.py unshown_views): the page
    # plays each, named as not shown to the model, with why; one board clips could not cut (record_unshown) has no
    # clip, so it is left out of what the page plays and of its note, and its problem stays on the episode
    from board.clips import UNSHOWN_NOT_DECODABLE, unshown_views
    uncut = {x.get("camera") for x in ctx.get("reader_issues") or []
             if isinstance(x, dict) and x.get("kind") == UNSHOWN_NOT_DECODABLE}
    unshown = [{"view": v, "name": str(e.get("name") or v), "why": str(e.get("why") or ""),
                **{key: e[key] for key in ('clock_alignment', 'presentation_clock') if key in e}}
               for v, e in unshown_views(ctx) if v not in uncut]
    if unshown:
        d["unshown_cameras"] = unshown
    add_contacts(d, ctx, result, ep_dir=ep_dir)
    if ctx.get('signals') and (Path(ep_dir) / 'signals.npz').exists():
        from checks.tactile_quality import for_episode
        try:
            # Preserve reviewed saved warnings until explicitly refreshed.
            if 'tactile_qc' not in d:
                d['tactile_qc'] = for_episode(Path(ep_dir), d.get('contacts') or [],
                                            ((result or {}).get('contact_views') or {}).get('strips') or {})
        except (OSError, ValueError, KeyError) as error:
            if 'tactile_qc' not in d:
                d['tactile_qc'] = {'version': 1, 'warnings': [],
                                   'checks': {'readings': 'errored'}, 'error': type(error).__name__}
    from board.dictionary_projection import apply as apply_dictionary
    apply_dictionary(d, ctx, ep_dir)


UPLOADER_LIST_MAX = 24       # a list of more numbers than this (a calibration matrix is 16) is summarised by its length


def uploader_groups(notes, start_s, dur_s) -> list[dict]:
    """The notes an upload sent with an episode (prepare/formats.py uploader_notes), as sent, for the board: each table
    row that names the episode as one group, the notes read from its files as another, every value under its own
    name. A number that is a time on the recorder's clock (seconds, ms, us or ns from its size, landing within the
    episode once its first frame's time clock_start_s is taken off) also carries that moment in the episode, so the
    board can jump to it. Only a recorder clock far from zero is read this way, so a count or an index is never taken
    for a time."""
    from board.public_export import location_list_receipt
    if notes in (None, "", {}, []):
        return []
    near = start_s is not None and dur_s and abs(float(start_s)) > 100 * float(dur_s)

    def when(v):
        if not near or isinstance(v, bool):
            return None
        try:
            x = float(v)
        except (TypeError, ValueError):
            return None
        for scale in (1.0, 1e-3, 1e-6, 1e-9):
            t = x * scale - float(start_s)
            if -0.5 <= t <= float(dur_s) + 0.5:
                return round(max(0.0, min(t, float(dur_s))), 3)
        return None

    def flat(x, path, out):
        if isinstance(x, dict):
            for k, v in x.items():
                flat(v, path + [str(k)], out)
        elif isinstance(x, list) and x and any(isinstance(i, dict) for i in x):
            for i, v in enumerate(x):
                flat(v, path + [str(i + 1)] if len(x) > 1 else path, out)
        else:
            if isinstance(x, list):
                v = (", ".join(str(i) for i in x) if len(x) <= UPLOADER_LIST_MAX
                     else f"{len(x)} values, {', '.join(str(i) for i in x[:4])}, ...")
            else:
                v = str(x)
            item = {"name": " / ".join(path) or "note", "value": v,
                    "_uploader_field_path": list(path)}
            if locations := location_list_receipt(path, x, UPLOADER_LIST_MAX):
                item['_uploader_location_values'] = locations
            t = when(x)
            if t is not None:
                item["t"] = t
            out.append(item)

    groups = []
    if isinstance(notes, dict) and isinstance(notes.get("table rows"), list):
        for r in notes["table rows"]:
            r = dict(r) if isinstance(r, dict) else {"row": r}
            table = r.pop("table", None)
            out = []
            flat(r, [], out)
            groups.append({"title": f"Row of {table}" if table else "Table row", "kind": "row", "items": out})
        notes = notes.get("notes")
    if notes not in (None, "", {}, []):
        out = []
        flat(notes, [], out)
        groups.append({"title": "Notes in the files", "kind": "notes", "items": out})
    return groups


def add_contacts(d: dict, ctx: dict, result: dict | None = None, ep_dir: Path | None = None) -> None:
    """The recording's contacts (context["contacts"], label/contacts.py), each with the model's answer when it was shown
    ("seen"), and the check of one against the other (checks/contacts.py) in dataset_checks["contact_checks"]. An
    episode prepared before contacts were measured has none in its context; labelling found them on the recording
    (label/episode.py build_request, label/pieces.py write_pieces) and its result holds them, so those are used."""
    recorded = ctx.get("contacts") or (result or {}).get("contacts") or []
    from label import contacts as lc
    from prepare.signal_alignment import recorder_arrival, RECORDER_ARRIVAL, RECORDER_ARRIVAL_LIMITATION
    from label.dictionary_context import field_interpretation
    meta = {s['name']: {**s, 'dictionary': field_interpretation(ctx, s['name'])}
            for s in ctx.get('signals') or [] if isinstance(s, dict) and s.get('name')}
    if recorder_arrival(ctx):
        meta = {name: {'aligned_by': RECORDER_ARRIVAL, **value} for name, value in meta.items()}
    covered = {} if ctx.get('state_unaligned') else lc.coverage(meta)
    d.pop('contact_sensor_coverage', None)
    if covered:
        d['contact_sensor_coverage'] = covered
    missing = d.get('contacts_missing') or []
    if missing:
        def covered_hand(row):
            hand = row.get('hand')
            return isinstance(hand, str) and hand.strip().lower() in covered
        d['contacts_missing'] = [{**m, 'withheld_from_final': True,
                                 'withheld_reason': ('Instrumentation coverage is unknown for the reported hand.'
                                    if not covered_hand(m) else RECORDER_ARRIVAL_LIMITATION)}
                                if isinstance(m, dict) and (not covered_hand(m) or recorder_arrival(ctx))
                                else m for m in missing]
    if not recorded and not missing:
        for k in ("contacts_model", "contact_views"):
            d.pop(k, None)
        return
    from checks import contacts as cc
    # a contact timed by a signal placed from both starts says so, also one found before contacts carried it, and a
    # contact a context prepared earlier found on a foot's contact points is no hand or gripper contact
    recorded = lc.mark_aligned(lc.hand_contacts(recorded, meta), meta)
    seen = {c.get("id"): c for c in d.pop("contacts_model", None) or []}
    views = d.pop("contact_views", None) or {}
    shown = set(views.get("shown") or [])
    from label.signal_class import place_words
    d["contacts"] = [{**c, **({"seen": seen[c["id"]]} if c["id"] in seen else {}), "shown": c["id"] in shown,
                      "place_words": place_words(c.get("hand"), c.get("place"), len(c.get("signals") or []))}
                     for c in recorded]
    res = cc.check({"contacts": list(seen.values()), "contacts_missing": d.get("contacts_missing") or []}, recorded,
                   views.get("strips") or {}, float(episode_fps(ctx, ep_dir=ep_dir)[0]),
                   instrumented_hands=set(covered),
                   timing_context=ctx)
    if res is not None:
        d["dataset_checks"] = d.get("dataset_checks") or {}
        d["dataset_checks"]["contact_checks"] = res


def carry_pieces(d: dict, r: dict, ctx: dict) -> None:
    """What a label of your own data carries beyond the run's output: for a recording labelled in parts
    (label/pieces.py), the parts it was stitched from and the issues the stitcher set aside at our own cuts, kept in
    _excluded with their reason; and how many neighbours the sped-up check had inside the folder
    (checks.timebase measure_folder)."""
    if ctx.get("timebase_neighbours_in_upload") is not None and (d.get("dataset_checks") or {}).get("timebase"):
        d["dataset_checks"]["timebase"]["neighbours_in_upload"] = ctx["timebase_neighbours_in_upload"]
    if r.get("stitched"):
        d["_stitched"] = r["stitched"]
        have = {(x.get("category"), x.get("issue")) for x in d.get("_excluded") or []}
        ex = [x for x in (r.get("labels") or {}).get("_excluded") or []
              if (x.get("category"), x.get("issue")) not in have]
        if ex:
            d["_excluded"] = (d.get("_excluded") or []) + ex


# fields with a fixed set of values: a model's word outside the set is shown as "unclear", so the board never counts
# or files an outcome or a severity it does not know
ENUMS = {"task_completed": {"success", "partial", "failure", "unclear", "success_then_undone", None},
         "severity": {"high", "medium", "low", None}}


def normalize_enums(x, key: str | None = None):
    """x with every string under an ENUMS key that is outside its set replaced by "unclear", at any depth."""
    if isinstance(x, dict):
        return {k: normalize_enums(v, k) for k, v in x.items()}
    if isinstance(x, list):
        return [normalize_enums(v, key) for v in x]
    if isinstance(x, str) and key in ENUMS and x not in ENUMS[key]:
        return "unclear"
    return x


# the episode's context a comparison label carries from the board's own label, so the page lays out the same player
# (length, rig, cameras, the dataset's own labels, where the footage comes from); none of the checks or rules
CONTEXT_KEYS = ("dataset", "_rig", "duration_s", "duration_estimated", "dataset_labels", "dataset_labels_note",
                "dataset_episode_labels", "uploader_notes", "dataset_source", "camera_views", "camera_labels", "camera_mounting",
                "timesteps_s", "task_label", "reader_notes", "unshown_cameras", "data_dictionary", "camera_presentation")


def build_comparisons(board: Path, manifest: dict, qa_new: Path, board_src: dict) -> dict:
    """Other models' labels of some of the board's episodes, kept apart from the board's own. Written to
    BOARD/compare.new, renamed to compare/ with qa/:
      <key>/<episode file>   one model's labels of one episode, converted like the board's; a response that did
                             not parse, was cut off or never came is a file too, saying so
      index.json             the reference (the model of the board's own labels) and every other model, and per
                             episode which of them were asked to label it and how each response came out
      metrics.json           compare.metrics.compute: every chart of the comparison view, with the reference
                             measured on the board's own labels (board_src: the run output each one was built from)
    The reference has no files here: its label of every episode is the board's own, in qa/. Nothing the board
    counts reads these: its lists, filters, exports and downloads read qa/ only."""
    from compare import metrics as mc
    out = board / "compare.new"
    if out.exists():
        remove_tree(out)
    out.mkdir()
    models = [m for m in mc.load_models(manifest, board) if not m["reference"]]
    metrics = mc.compute(manifest, board, board_src)
    rows = metrics.pop("_rows")
    # published with the board: each model by its name and settings, never the run, slice or code it came from
    for m in metrics["models"]:
        for k in ("run_id", "code", "slice"):
            m.pop(k, None)
    index = {"reference": next(m for m in metrics["models"] if m["reference"]),
             "models": [m for m in metrics["models"] if not m["reference"]], "episodes": {}}
    written = {m["key"]: 0 for m in models}
    for row in rows.values():
        f = row["file"]
        bd = json.loads((qa_new / f).read_text())
        ctx = {k: bd[k] for k in CONTEXT_KEYS if k in bd}
        bmeta = bd.get("_meta") or {}
        meta = {k: bmeta[k] for k in ("episode_id", "run_episode", "given_prompt", "prompt_mode")
                if bmeta.get(k) is not None}
        srcs = {}
        for m in models:
            if m["key"] not in row["by"]:
                continue
            r = mc.response(m, row["name"])
            if r["status"] == "pending":
                continue
            if r["status"] == "parsed":
                d = convert(r['_record'], bd.get("dataset"))
                d.update(ctx)
                d["_meta"] = {**(d.get("_meta") or {}), **meta}
            else:
                d = {**ctx, "episode_prompt": "", "event_labels": [], "_meta": dict(meta)}
                if r.get("cost") is not None:
                    d["_usage"] = {"est_cost_usd": r.get("cost"), "latency_s": r.get("latency"),
                                   "completion_tokens": r.get("out_tokens")}
            info = {"key": m["key"], "name": m["episode_name"], "status": r["status"], "model": m["model"],
                    "example": m["example"]}
            if 'cost_complete' in r:
                d.setdefault('_usage', {}).update({key: r[key] for key in
                    ('billed_usd', 'reserved_usd', 'cost_complete')})
            if r["status"] == "unparsed":
                info.update({"parse_error": r.get("error"), "raw_head": (r.get("raw") or "")[:3000],
                             "raw_chars": len(r.get("raw") or "")})
            if r["status"] == "incomplete":
                info.update({"why": r.get("why") or [], "raw_head": (r.get("raw") or "")[:3000],
                             "raw_chars": len(r.get("raw") or "")})
            if r["status"] == "cut_off":
                info.update({"out_tokens": r.get("out_tokens"), "tail": (r.get("tail") or "")[-1500:]})
            d["_compare"] = info
            (out / m["key"]).mkdir(exist_ok=True)
            write_atomic(out / m["key"] / f, d, indent=None, serializer=dumps)
            srcs[m["key"]] = r["status"]
            written[m["key"]] += 1
        if srcs:
            index["episodes"][f] = srcs
    write_atomic(out / "index.json", index, indent=None, serializer=dumps, separators=(",", ":"))
    write_atomic(out / "metrics.json", mc.public(metrics), indent=None, serializer=dumps, separators=(",", ":"))
    return {"models": {m["key"]: {"run_id": m["run_id"], "files": written[m["key"]]} for m in models},
            "episodes": len(index["episodes"])}


def build_hands(board: Path, spec: dict, qa_new: Path, episodes: dict) -> dict:
    """The hand pose of the head-camera episodes (board/hands.py): the drawing, written to BOARD/hands.new, and the
    download in the dataset video's own pixels and frame times, written to BOARD/hand_keypoints.new; both renamed
    into place with qa/. episodes maps a board file to its prepared episode folder. Nothing that reads qa/ reads
    these."""
    from board import hands as hands_overlay
    out, kout = board / "hands.new", board / "hand_keypoints.new"
    for d in (out, kout):
        if d.exists():
            remove_tree(d)
        d.mkdir()
    here = board.resolve()
    src, clips = (Path(p) if Path(p).is_absolute() else (here / p).resolve() for p in (spec["src"], spec["clips"]))
    res = hands_overlay.build(src, qa_new, clips, out)
    for sk in res["skipped"]:
        print(f"hands: skipped {sk['key']}: {sk['skip']}", file=sys.stderr)
    kres = hands_overlay.build_keypoints(src, qa_new, episodes, kout)
    for sk in kres["skipped"]:
        print(f"hand keypoints: skipped {sk['key']}: {sk['skip']}", file=sys.stderr)
    return {"src": spec["src"], "written": res["written"], "skipped": len(res["skipped"]),
            "head_camera_files_without_keypoints": len(res["head_camera_files_without_keypoints"]),
            "bytes": res["bytes"],
            "keypoints": {"written": kres["written"], "skipped": len(kres["skipped"]), "bytes": kres["bytes"]}}


def build_sensors(board: Path, episodes: dict, last: dict | None = None, keys: dict | None = None) -> dict | None:
    """The sensors files of the episodes with signals or depth (board/sensors.py), written to BOARD/sensors.new and
    renamed into place with qa/; None, and nothing written, when no episode has either. episodes maps a board file to
    its prepared episode folder."""
    from board import sensors as bs
    have = {f: ep for f, ep in episodes.items()
            if (Path(ep) / "signals.npz").exists() or (Path(ep) / "depth.json").exists()}
    if not have:
        return None
    out = board / "sensors.new"
    if out.exists():
        remove_tree(out)
    out.mkdir()
    keys = {} if keys is None else keys
    for f, ep in have.items():
        keys[f] = sensors_key(Path(ep))
    # a sensors file built from the same episode folder by the same code is the last build's, as it is
    res = bs.build(have, out, reuse=lambda f: cached_file(board / "sensors", f, keys[f], last or {}, drop=()))
    for sk in res["skipped"]:
        print(f"sensors: skipped {sk['file']}: {sk['skip']}", file=sys.stderr)
    return {"written": res["written"], "skipped": len(res["skipped"]), "bytes": res["bytes"],
            "reused": res.get("reused", 0)}


def resolve_run(p: Path) -> Path:
    """A run folder; RUNS/<dataset>/latest (when no folder has that name) is the newest finished run that is
    not a dry run."""
    if p.name != "latest" or p.exists():
        return p
    done = []
    for rj in p.parent.glob("*/run.json"):
        info = json.loads(rj.read_text())
        if info.get("status") == "done" and info.get("kind") != "dry":
            done.append(rj.parent)
    if not done:
        raise SystemExit(f"no finished run under {p.parent}")
    return sorted(done)[-1]


def _path(p: str, here: Path) -> Path:
    """A manifest path: absolute, or relative to the board folder."""
    return Path(p) if Path(p).is_absolute() else (here / p).resolve()


def _rerun_folder(r: dict, run: Path, name: str) -> Path | None:
    """The resolved episode folder a rerun's output labelled: its episode_dir, else the run's slice / name."""
    for p in (r.get("episode_dir"), (json.loads((run / "run.json").read_text()).get("slice") or "") + "/" + name):
        if p and Path(p).exists():
            return Path(p).resolve()
    return None


def entry_labels(entry: dict, here: Path, skipped: list | None = None) -> tuple[Path, dict]:
    """A manifest entry's run folder and {board file: (the entry's episode name, output file, output, the run it
    came from)} for every label it holds (board/to_board.py label_outputs), each rerun's label in place of the base
    run's for the episodes it labelled (the module docstring). here is the board folder, resolved. skipped, when
    given, gets one line per file of the runs that is no reply (label_outputs), which BUILT.json names."""
    run = resolve_run(_path(entry["run"], here))
    pre = entry.get("file_prefix") or ""
    outs, skip = label_outputs(run / "out")
    if skipped is not None:
        skipped += skip
    by_name = {name: (f, r, run) for name, (f, r) in outs.items()}
    if entry.get("reruns"):
        eps = _path(entry["episodes"], here)
        own = {p.resolve(): p.name for p in eps.iterdir() if p.name.startswith("episode_")}
        for rr in entry["reruns"]:
            rrun = resolve_run(_path(rr["run"], here))
            routs, rskip = label_outputs(rrun / "out")
            if skipped is not None:
                skipped += [f"{rrun.name}/{x}" for x in rskip]
            for rname, (f, r) in routs.items():
                folder = _rerun_folder(r, rrun, rname)
                if folder not in own:
                    continue
                # a rerun's reply that gave no labels never replaces a label that parsed
                if label_failed(r) is None or label_failed((by_name.get(own[folder]) or (None, {}))[1]) is not None:
                    by_name[own[folder]] = (f, r, rrun)
    return run, {(name.replace("episode_", f"episode_{pre}", 1) if pre else name) + ".json": (name, f, r, src)
                 for name, (f, r, src) in by_name.items()}


def label_sources(manifest: dict, board: Path) -> dict:
    """{board file: the run output its label is built from} for every episode of the board. build() builds exactly
    these, and compare/metrics.py measures the reference from them, so the comparison's reference is the board's
    own label of each episode."""
    here = Path(board).resolve()
    return {f: out for e in manifest.get("datasets", []) for f, (_, out, _, _) in entry_labels(e, here)[1].items()}


def exchange_directories(left: Path, right: Path) -> None:
    """Exchange existing directory names atomically, preserving both on failure."""
    import ctypes
    import sys
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == 'darwin' and hasattr(libc, 'renamex_np'):
        exchange = libc.renamex_np
        exchange.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        result = exchange(os.fsencode(left), os.fsencode(right), 2)
    elif sys.platform.startswith('linux') and hasattr(libc, 'renameat2'):
        exchange = libc.renameat2
        exchange.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        result = exchange(-100, os.fsencode(left), -100, os.fsencode(right), 2)
    else:
        raise OSError(errno.ENOTSUP, 'Atomic directory exchange is unavailable on this platform')
    if result:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def remove_tree(path: Path) -> None:
    """Remove derived trees when macOS also removes their AppleDouble entries."""
    def onerror(function, name, error):
        if not isinstance(error[1], FileNotFoundError):
            raise error[1]
    shutil.rmtree(path, onerror=onerror)


def _discard_publication(root: Path) -> None:
    # Keep the cleanup phase recorded until its backup has been removed.
    # A killed recursive cleanup must not orphan the remaining backup files.
    previous = root / 'previous'
    if previous.exists():
        remove_tree(previous)
    remove_tree(root)


def recover_publications(board: Path, name: str | None = None) -> None:
    """Restore interrupted installs; retain unrecognized backups for inspection."""
    for root in sorted(board.glob(f'.{glob.escape(name)}.rollback-*' if name else '.*.rollback-*')):
        if not root.is_dir():
            continue
        marker = root / 'publication.json'
        try:
            state = json.loads(marker.read_text())
            target = state['target']
            if (state.get('format') != 'argus-publication/1' or not isinstance(target, str) or
                    Path(target).name != target or target in ('', '.', '..') or
                    not root.name.startswith(f'.{target}.rollback-') or root.is_symlink()):
                raise ValueError('Unrecognized publication record')
            phase = state['phase']
            if phase not in ('copying', 'ready', 'published', 'restored'):
                raise ValueError('Unrecognized publication phase')
        except (OSError, ValueError, KeyError, TypeError):
            if root.is_dir() and not root.is_symlink() and not any(root.iterdir()):
                root.rmdir()
                continue
            print(f'board: preserved unrecognized publication backup at {root}', file=sys.stderr)
            continue
        if phase == 'ready':
            backup, live = root / 'previous', board / target
            if live.is_symlink() or not backup.is_dir() or any(p.is_symlink() for p in [backup, *backup.rglob('*')]):
                raise RuntimeError(f'Publication recovery needs inspection; previous data is preserved at {root}')
            live.mkdir(exist_ok=True)
            try:
                _install_tree(backup, live)
            except BaseException as error:
                raise RuntimeError(f'Publication recovery failed; previous data is preserved at {root}') from error
            write_atomic(marker, {**state, 'phase': 'restored'})
        _cleanup_publication(root)


def _cleanup_publication(root: Path) -> None:
    try:
        _discard_publication(root)
    except OSError as error:
        print(f'board: publication cleanup deferred at {root}: {error}', file=sys.stderr)


def _install_tree(source: Path, destination: Path) -> None:
    """Replace regular files atomically without removing the live directory."""
    incoming = {p.relative_to(source): p for p in source.rglob('*') if p.is_file()}
    for rel, path in sorted(incoming.items()):
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix='.argus-publish-', dir=target.parent)
        os.close(fd)
        temporary = Path(name)
        try:
            shutil.copy2(path, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    for path in destination.rglob('*'):
        if path.is_file() and path.relative_to(destination) not in incoming:
            path.unlink(missing_ok=True)
    for path in sorted(destination.rglob('*'), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def publish_directory(old: Path, new: Path) -> None:
    """Prefer a generation exchange; fall back to per-file publication and rollback.

    Unsupported filesystems keep the live directory and every open file reader.
    New readers can observe mixed generations during the portable fallback.
    A complete backup restores the previous generation if publication fails.
    """
    recover_publications(old.parent, old.name)
    try:
        exchange_directories(old, new)
        return
    except OSError as error:
        if error.errno not in {errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EINVAL, errno.EXDEV}:
            raise
    # Derived outputs contain regular files. Reject links before touching live data.
    if any(p.is_symlink() for root in (old, new) for p in [root, *root.rglob('*')]):
        raise OSError(errno.ENOTSUP, 'Portable directory publication requires regular files')
    backup_root = Path(tempfile.mkdtemp(prefix=f'.{old.name}.rollback-', dir=old.parent))
    backup = backup_root / 'previous'
    marker = backup_root / 'publication.json'
    state = {'format': 'argus-publication/1', 'target': old.name, 'phase': 'copying'}
    try:
        write_atomic(marker, state)
        shutil.copytree(old, backup)
        write_atomic(marker, {**state, 'phase': 'ready'})
    except BaseException:
        _discard_publication(backup_root)
        raise
    try:
        _install_tree(new, old)
        write_atomic(marker, {**state, 'phase': 'published'})
    except BaseException:
        try:
            _install_tree(backup, old)
        except BaseException as restore_error:
            raise RuntimeError(f'Publication rollback failed; previous data is preserved at {backup}') from restore_error
        write_atomic(marker, {**state, 'phase': 'restored'})
        _cleanup_publication(backup_root)
        raise
    _cleanup_publication(backup_root)


def _swap(board: Path, name: str, keep: bool) -> None:
    """Publish derived files without removing the live directory before replacement."""
    old, new = board / name, board / f'{name}.new'
    recover_publications(board, name)
    if not keep:
        if old.exists():
            remove_tree(old)
        return
    if old.exists():
        publish_directory(old, new)
        try:
            remove_tree(new)
        except OSError as error:
            print(f'board: publication cleanup deferred at {new}: {error}', file=sys.stderr)
    else:
        new.rename(old)


def unlabelled(entry: dict, eps: Path, run: Path, labels: dict) -> dict:
    """{board file: (episode name, None, output, run)} for every prepared episode of an entry whose run labels every
    episode of its folder (labels_every_episode, board/rules.py own_data_entry) that no run of it has an output for:
    the run never recorded a reply for it (stopped before it, or written before the harness kept why), so it is shown
    with its footage, checks and sensors and says so (label_failed no_reply). Nothing for an entry whose run labelled
    a chosen few of the folder's episodes: the others were never sent."""
    if not entry.get("labels_every_episode"):
        return {}
    pre = entry.get("file_prefix") or ""
    have = {name for name, _, _, _ in labels.values()}
    out = {}
    for d in sorted(eps.glob("episode_*")) if eps.is_dir() else []:
        if d.name in have or not (d / "context.json").exists():
            continue
        fname = (d.name.replace("episode_", f"episode_{pre}", 1) if pre else d.name) + ".json"
        out[fname] = (d.name, None, {"episode_dir": str(d), "parse_ok": False,
                                     "no_reply": "the labelling run recorded no reply for it"}, run)
    return out


def not_shown(r: dict, e: Exception) -> dict:
    """An output whose reply parsed but which the board could not read: what the request recorded, the reply as text,
    and the error (board/to_board.py label_failed not_shown), so the episode is still built."""
    keep = ("episode_dir", "model", "reasoning_effort", "config", "usage", "dataset_checks", "decode_failed",
            "given_prompt", "prompt_mode", "task_label", "arm_still_spans")
    try:
        raw = json.dumps(r.get("labels"), indent=1, default=str)
    except (TypeError, ValueError):
        raw = str(r.get("labels"))
    return {**{k: r[k] for k in keep if k in r and isinstance(r[k], (dict, list, str, int, float))}, "parse_ok": False,
            "board_error": f"{type(e).__name__}: {e}"[:300], "raw": raw}


def read_context(ep_dir: Path) -> tuple[dict, str | None]:
    """(the episode's context.json, why it does not read): {} and None when it has none, {} and the error when it does
    not parse as an object (a file cut mid write)."""
    p = ep_dir / "context.json"
    if not p.exists():
        return {}, None
    try:
        ctx = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        return {}, f"{type(e).__name__}: {e}"[:300]
    return (ctx, None) if isinstance(ctx, dict) else ({}, f"it holds {type(ctx).__name__}, not an object")


def context_unreadable(why: str) -> dict:
    """The issue of an episode whose context.json the board could not use (kind context_unreadable): it is shown with
    the model's reply as it came, labels or not, and without what the context gives (its length, checks, the reader's
    issues and notes). The error is kept apart from the sentence, for whoever looks into it."""
    from board.public_export import reader_failure_what
    return {"kind": "context_unreadable", "error": why,
            "_reader_public_error": reader_failure_what('context_unreadable'),
            "what": "This episode's recorded details (its length, checks and the reader's notes) could not be read, so "
                    "they are not shown; the model's reply is shown as it came."}


ADDITIONS = "additions"


def read_additions(board: Path) -> dict:
    """{board file: (its text, the addition or None, why it does not read or None)} for every file of
    BOARD/additions/, or {} when the board has no such folder."""
    folder = board / ADDITIONS
    out = {}
    for f in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        try:
            text = f.read_text()
            a = json.loads(text)
        except (OSError, ValueError) as e:
            out[f.name] = ("", None, f"{type(e).__name__}: {e}"[:300])
            continue
        why = addition_problem(a)
        out[f.name] = (text, None if why else a, why)
    return out


def addition_problem(a) -> str | None:
    """Why an addition cannot be applied, or None: it is an object with a non-empty provenance object, fields as an
    object and append as an object of lists of objects."""
    if not isinstance(a, dict):
        return "it is not an object"
    if not isinstance(a.get("provenance"), dict) or not a["provenance"]:
        return "it has no provenance object"
    if not isinstance(a.get("fields", {}), dict):
        return "its fields is not an object"
    if not isinstance(a.get("append", {}), dict) or not all(
            isinstance(rows, list) and all(isinstance(r, dict) for r in rows) for rows in a.get("append", {}).values()):
        return "its append is not an object of lists of objects"
    if not a.get("fields") and not a.get("append"):
        return "it adds nothing"
    return None


def _empty(v) -> bool:
    return v is None or v == "" or v == [] or v == {}


def apply_addition(d: dict, a: dict) -> dict:
    """Merges an addition (read_additions) into a converted label, in place. Each field of fields is set only where the
    label has none (absent, null or empty), and each row of append is added to that list field of the label unless the
    label's list already has a row of its category; every
    added field that is an object and every added row carries the addition's provenance, unless it brings its own, and
    _additions lists what was added. A field the label already has is never overwritten: it is kept, and the conflict
    is listed in addition_conflicts. Returns {field: value added}, so the caller can check the build left it alone."""
    prov = a["provenance"]
    added, done, conflicts = {}, [], []
    for field, value in (a.get("fields") or {}).items():
        if field.startswith("_") or not _empty(d.get(field)):
            conflicts.append({"field": field, "why": "the label already has this field, so the addition is not applied"
                              if not field.startswith("_") else "fields starting with _ belong to the build"})
            continue
        if isinstance(value, dict):
            value = {**value, "provenance": value.get("provenance") or prov}
        d[field] = added[field] = value
        done.append({"field": field, "provenance": prov})
    for field, rows in (a.get("append") or {}).items():
        have = d.get(field)
        if field.startswith("_") or (have is not None and not isinstance(have, list)):
            conflicts.append({"field": field, "why": "the label's field is not a list, so the rows are not added"
                              if not field.startswith("_") else "fields starting with _ belong to the build"})
            continue
        # a row whose category the label already reports is a second report of the same thing: the label's stands
        own = {r.get("category") for r in have or [] if isinstance(r, dict) and r.get("category")}
        for r in rows:
            if r.get("category") in own:
                conflicts.append({"field": field, "why": f"the label already has a {r['category']} row, so the "
                                  "addition's is not added"})
        rows = [{**r, "provenance": r.get("provenance") or prov} for r in rows if r.get("category") not in own]
        if not rows:
            continue
        d[field] = list(have or []) + rows
        done.append({"field": field, "rows": len(rows), "provenance": prov})
    if done:
        d["_additions"] = done
    if conflicts:
        d["addition_conflicts"] = conflicts
    return added


def episode_file(entry: dict, manifest: dict, fname: str, name: str, r: dict, info: dict, eps: Path, ctx: dict,
                 unread: str | None = None, addition: dict | None = None) -> dict:
    """One episode's board file: its label (board/to_board.py convert) with its addition merged in (apply_addition),
    provenance, context (ctx, its context.json, or {} with why it did not read in unread), rules, consistency check
    and the parts it was stitched from."""
    d = convert(r, entry["dataset"])
    added = apply_addition(d, addition) if addition else {}
    d["_run"] = {"run_id": info["run_id"], "code": info["code"], "kind": info["kind"], "slice": info.get("slice")}
    if manifest.get("labels_license"):
        d["labels_license"] = manifest["labels_license"]    # travels with the label into every download
    if ctx:
        add_context(d, ctx, eps / name, r)
    else:
        # a reply that gave no labels is flagged with or without a context, beside a context that did not read
        add_reader_issues(d, {"reader_issues": [context_unreadable(unread)]} if unread else {}, r)
    # after the checks are in; a rule that needs the context (fixed_window) skips where there is none
    apply_rules(d, ctx, entry.get("rules") or [], eps / name)
    d["label_consistency"] = label_consistency.check(d, d.get("duration_s"))
    if "duration_s" not in d and d.get("timesteps_s"):
        # no context: the last sampled time plus one sampling step, marked as an estimate
        ts = [float(t) for t in d["timesteps_s"]]
        step = (ts[-1] - ts[0]) / (len(ts) - 1) if len(ts) > 1 else 0.0
        d["duration_s"], d["duration_estimated"] = round(ts[-1] + step, 3), True
    if fname != name + ".json":
        # the board finds clips by episode_id; the run's own name stays for the hand pose keypoints
        d["_meta"] = {**(d.get("_meta") or {}), "episode_id": Path(fname).stem, "run_episode": name}
    carry_pieces(d, r, ctx)
    for field, value in added.items():
        if d.get(field) != value:
            # the build sets this field itself (the context, the checks): its own value stands, and the addition's
            # is reported, never silently lost
            d.setdefault("addition_conflicts", []).append(
                {"field": field, "why": "the build sets this field, so the addition is not applied"})
            d["_additions"] = [x for x in d.get("_additions") or [] if x.get("field") != field]
            if not d["_additions"]:
                d.pop("_additions")
    return normalize_enums(d)


def guarded_episode_file(entry: dict, manifest: dict, fname: str, name: str, r: dict, info: dict, eps: Path,
                         addition: dict | None = None) -> dict:
    """episode_file, which never stops the build. The context is read once (read_context). When the file cannot be
    built, the context is tried alone, with no reply: one it fails on (a capture record that is not an object) does not
    read either, and the episode is built with its labels and a context_unreadable issue, so a fault of the context is
    never blamed on the reply. Otherwise the reply is what the board could not read, and the episode is built with the
    reply as text and the error (not_shown)."""
    ctx, unread = read_context(eps / name)
    try:
        return episode_file(entry, manifest, fname, name, r, info, eps, ctx, unread, addition)
    except Exception as e:  # noqa: BLE001 - the context or the reply: found below, the build goes on
        err = e
    if ctx:
        try:
            add_context({}, ctx, eps / name)
        except Exception as e:  # noqa: BLE001 - the context does not read
            print(f"board: {fname}: the context could not be read ({type(e).__name__}: {e})", file=sys.stderr)
            ctx, unread = {}, f"{type(e).__name__}: {e}"[:300]
            try:
                return episode_file(entry, manifest, fname, name, r, info, eps, ctx, unread, addition)
            except Exception as e2:  # noqa: BLE001 - the reply does not read either
                err = e2
    print(f"board: {fname}: the reply could not be read ({type(err).__name__}: {err})", file=sys.stderr)
    return episode_file(entry, manifest, fname, name, not_shown(r, err), info, eps, ctx, unread, addition)


def add_scores(qa: Path, files: list[str]) -> None:
    """Each episode's quality scores (board/scores.py), written into its record. They compare an episode with the
    board's other demonstrations of the same task, so they are computed once every record is written, holding only
    the fields they read."""
    records = {}
    for f in files:
        d = json.loads((qa / f).read_text())
        records[f] = {k: d.get(k) for k in scores.FIELDS}
    for f, s in scores.score_board(records).items():
        d = json.loads((qa / f).read_text())
        d["scores"] = s
        write_atomic(qa / f, d, indent=None, serializer=dumps)



# ---- the build cache: an episode whose inputs are unchanged keeps the files the last build made of it ----
BUILD_KEYS = "build_keys.json"      # board file -> the key of the inputs its live qa/ and sensors/ files were built from
_CODE_ID = None


def code_identity() -> str:
    """One hash of the code and configuration a build runs (board/, label/, checks/, prepare/ and configs/), so any
    change to them rebuilds every episode."""
    global _CODE_ID
    if _CODE_ID is None:
        import hashlib
        repo = Path(__file__).resolve().parent.parent
        h = hashlib.sha256()
        for top in ("board", "label", "checks", "prepare", "configs"):
            for f in sorted((repo / top).rglob("*")):
                if f.is_file() and f.suffix in (".py", ".json", ".html", ".js", ".yaml", ".yml"):
                    h.update(f.relative_to(repo).as_posix().encode() + b"\0" + f.read_bytes() + b"\0")
        _CODE_ID = h.hexdigest()
    return _CODE_ID


def _content_text(path) -> str:
    """The digest of a file's content (label/content_identity.py, whose stat check only spares reading it again), so a
    file copied or touched with the same bytes keeps its key and one edited under the same size and time changes it."""
    from label import content_identity
    try:
        return content_identity.digest(path)
    except (OSError, ValueError):
        return "-"


def folder_identity(ep_dir: Path) -> list:
    """Every file of a prepared episode folder by name and content, and the source videos its sources.json names (read
    by tools the build starts, outside its own file reads)."""
    out = []
    if not ep_dir.is_dir():
        return out
    for f in sorted(ep_dir.rglob("*")):
        if f.is_file():
            out.append(f"{f.relative_to(ep_dir).as_posix()}={_content_text(f)}")
    try:
        sources = json.loads((ep_dir / "sources.json").read_text())
    except (OSError, ValueError):
        sources = {}
    for view, src in sorted(sources.items()) if isinstance(sources, dict) else []:
        for k, v in sorted(src.items()) if isinstance(src, dict) else []:
            if isinstance(v, str) and v.startswith("/"):
                out.append(f"source {view}.{k}={_content_text(v)}")
    return out


def _without_targets(value):
    """The manifest, or an entry, without its targets: a target is drawn by the page and changes no board file."""
    if isinstance(value, dict):
        return {k: _without_targets(v) for k, v in value.items() if k != "target"}
    if isinstance(value, list):
        return [_without_targets(v) for v in value]
    return value


def episode_key(manifest: dict, entry: dict, fname: str, name: str, src, r, info: dict, ep_dir: Path,
                addition_text: str | None = None) -> str:
    """The key of everything a board file is made from (traced: the episode folder, the run's output and run.json,
    the manifest, the code and its file in BOARD/additions when it has one). Equal keys make equal files, so the last
    build's file is used as it is."""
    import hashlib
    label = _content_text(src) if src is not None else json.dumps(r, sort_keys=True, default=str)
    parts = [code_identity(), json.dumps(_without_targets(manifest), sort_keys=True, default=str),
             json.dumps(_without_targets(entry), sort_keys=True, default=str), fname, name, str(src), label,
             json.dumps(info, sort_keys=True, default=str), *folder_identity(ep_dir)]
    if addition_text is not None:
        parts.append("addition:" + addition_text)
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def sensors_key(ep_dir: Path) -> str:
    """The key of what a sensors file is made from: its prepared episode folder and the code."""
    import hashlib
    return hashlib.sha256("\n".join([code_identity(), str(ep_dir), *folder_identity(Path(ep_dir))]).encode()).hexdigest()


def read_build_keys(board: Path) -> dict:
    try:
        keys = json.loads((board / BUILD_KEYS).read_text())
    except (OSError, ValueError):
        return {"qa": {}, "sensors": {}}
    return keys if isinstance(keys, dict) else {"qa": {}, "sensors": {}}


def cached_file(folder: Path, fname: str, key: str, keys: dict, drop=("scores",)):
    """The last build's file for fname when it was built from the same key, without what the build adds afterwards
    (scores, which compare an episode with the others), or None."""
    if not key or keys.get(fname) != key or not (folder / fname).is_file():
        return None
    try:
        d = json.loads((folder / fname).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict):
        return None
    for k in drop:
        d.pop(k, None)
    return d


def build(board: Path, cache: bool = True) -> dict:
    recover_publications(board)
    manifest = json.loads((board / "manifest.json").read_text())
    # the last build's keys, read before anything is replaced; --no-cache builds every episode again
    last = read_build_keys(board) if cache else {"qa": {}, "sensors": {}}
    keys = {"qa": {}, "sensors": {}}
    reused = 0
    here = board.resolve()
    new = board / "qa.new"
    if new.exists():
        remove_tree(new)
    new.mkdir(parents=True)
    counts, built_entries, written = {}, [], []
    board_src = {}      # board file -> the run output its label came from
    episodes = {}       # board file -> its prepared episode folder
    additions = read_additions(board)     # board file -> its addition, taken as each file is built
    added = {"files": len(additions), "episodes": 0, "conflicts": []}
    for entry in manifest.get("datasets", []):
        skipped: list = []
        run, labels = entry_labels(entry, here, skipped)
        infos = {}
        eps = _path(entry["episodes"], here)
        labels.update(unlabelled(entry, eps, run, labels))
        for fname, (name, src, r, from_run) in sorted(labels.items()):
            info = infos.get(from_run) or infos.setdefault(from_run, json.loads((from_run / "run.json").read_text()))
            text, addition, unusable = additions.pop(fname, (None, None, None))
            key = episode_key(manifest, entry, fname, name, src, r, info, eps / name,
                              None if text is None else text + (unusable or ""))
            d = cached_file(board / "qa", fname, key, last.get("qa") or {})
            if d is None:
                d = guarded_episode_file(entry, manifest, fname, name, r, info, eps, addition)
                if unusable:
                    d.setdefault("addition_conflicts", []).append(
                        {"field": None, "why": f"the addition {ADDITIONS}/{fname} is not applied: {unusable}"})
            else:
                reused += 1
            if d.get("_additions"):
                added["episodes"] += 1
            for c in d.get("addition_conflicts") or []:
                added["conflicts"].append({"file": fname, **c})
            keys["qa"][fname] = key
            if (eps / name / "context.json").exists():
                episodes[fname] = eps / name
            dest = new / fname
            if dest.exists():
                raise RuntimeError(f"{dest.name} comes from two manifest entries; a board holds one label per "
                                   "episode (file_prefix separates datasets whose episode names repeat)")
            write_atomic(dest, d, indent=None, serializer=dumps)
            written.append(fname)
            if src is not None:
                board_src[fname] = src
        counts[entry["dataset"]] = {"run_id": json.loads((run / "run.json").read_text())["run_id"],
                                    "episodes": len(labels), **({"skipped": skipped} if skipped else {})}
        # BUILT.json names the run that was used, so the board's inputs stay traceable
        built_entries.append(dict(entry, run=os.path.relpath(run, here)) if run != _path(entry["run"], here)
                             else entry)
    add_scores(new, written)
    if reused:
        print(f"board: {reused} of {len(written)} episodes unchanged since the last build, reused", file=sys.stderr)
    # an addition that names no episode of the board is reported, never dropped quietly
    added["unused"] = sorted(additions)
    for c in added["conflicts"]:
        print(f"board: {c['file']}: {c['why']}" + (f" ({c['field']})" if c.get("field") else ""), file=sys.stderr)
    if added["unused"]:
        print(f"board: {len(added['unused'])} files of {ADDITIONS}/ name no episode of this board, not applied",
              file=sys.stderr)
    compared = build_comparisons(board, manifest, new, board_src) if manifest.get("comparisons") else None
    hands = build_hands(board, manifest["hands"], new, episodes) if manifest.get("hands") else None
    sensors = (build_sensors(board, episodes, last.get("sensors") or {}, keys["sensors"])
               if manifest.get("sensors", True) is not False else None)
    _swap(board, "qa", True)
    _swap(board, "compare", compared is not None)
    _swap(board, "hands", hands is not None)
    _swap(board, "hand_keypoints", hands is not None)
    _swap(board, "sensors", sensors is not None)
    # the keys describe the files now live, so they are written once qa/ and sensors/ are in place
    write_atomic(board / BUILD_KEYS, keys, indent=None, serializer=dumps)
    built = {"manifest": {**manifest, "datasets": built_entries}, "counts": counts,
             **({"comparisons": compared} if compared else {}), **({"hands": hands} if hands else {}),
             **({"sensors": sensors} if sensors else {}), **({ADDITIONS: added} if added["files"] else {})}
    write_atomic(board / "BUILT.json", built, indent=1, serializer=dumps)
    return built




def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m board build", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("board", type=Path, help="the board folder, with manifest.json")
    ap.add_argument("--no-cache", action="store_true",
                    help="build every episode again, even one whose inputs and code are unchanged")
    a = ap.parse_args()
    if not (a.board / 'manifest.json').is_file():
        ap.error('No board manifest.json exists. A free review stops before board generation; build a labelled board first.')
    print(json.dumps(build(a.board, cache=not a.no_cache)["counts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
