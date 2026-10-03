"""Build a board from exactly the runs its manifest names.

    python -m board build BOARD          BOARD/manifest.json in; BOARD/qa/, BOARD/BUILT.json out

A board serves the episode files in BOARD/qa. Those files are derived: every build regenerates them from the runs
listed in BOARD/manifest.json and nothing else, so a board never shows a stale or mixed set of labels. Each file is
the run's label (board/to_board.py) with its provenance (`_run`: run id, code commit, kind, slice), the episode's
rig and length, the deterministic dataset checks and the dataset's own labels from its context.json, where the
footage comes from and its license (`dataset_source`, from board/dataset_sources.json, for the public datasets
prepare/ reads), the manifest's rules applied, and the label consistency check (checks/label_consistency.py:
annotations that contradict themselves are reported in label_consistency, never used to edit a label). A
recording labelled in parts carries the parts it was stitched from and the issues set aside at our cuts
(carry_pieces), and an outcome or severity outside its known values is shown as "unclear" (normalize_enums). An
episode whose model reply did not parse or was cut off is on the board too, with its footage, checks and sensors, no
labels, the reply itself, and a data issue saying so (label_failure); a rerun's reply of that kind never replaces a
label that parsed. So is every prepared episode of the entry that got no reply at all (the spend cap reached, a
request that could not be built, a run stopped before it), saying why, and one whose reply the board cannot read,
with the reply and the error: one episode never stops the build.

manifest.json. Paths are absolute or relative to the board folder; a run given as RUNS/<dataset>/latest is that
dataset's newest finished run that is not a dry run (run ids start with their start time).

    {"board": "quickstart",                                   the board's name
     "datasets": [                                            one entry per dataset, in tab order
       {"dataset": "molmo",                                   the dataset's short name on the board
        "run": "../../runs/molmo/latest",                     a run folder: run.json and out/
        "episodes": "../../episodes/molmo/quickstart",        the prepared episodes the run labelled
        "rules": [...],                                       optional: definitions applied after labelling
        "file_prefix": "..."},                                optional: a prefix for its board file names
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
import json
import os
import re
import shutil
import sys
from pathlib import Path

from board.to_board import convert, dumps, label_failed, label_outputs, off_schema_text, typed
from checks import label_consistency

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
    """The episode's length: the sidecar's own duration (from the video's timestamps) when it has one, then
    the span of its real per-frame capture times (ABC-130k stations record below the 30 fps their files
    declare, so frame count over fps understates them), otherwise its frame count over its frame rate."""
    if ctx.get("duration_s"):
        return float(ctx["duration_s"])
    if ep_dir is not None and ctx.get("real_times") and (ep_dir / ctx["real_times"]).exists():
        import numpy as np
        z = np.load(ep_dir / ctx["real_times"])
        cam = next((k for k in ("exo", "left", "right") if k in z.files), None)
        if cam is not None and len(z[cam]) > 1:
            t = z[cam].astype(float)
            return float(t[-1] - t[0] + np.median(np.diff(t)))
    fps = ctx.get("fps")
    n = ctx.get("n_state_frames")
    return n / fps if fps and n else None


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


def apply_rules(d: dict, ctx: dict, rules: list) -> None:
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
            secs = episode_seconds(ctx)
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
    """What the reader says about an episode beyond what it read: its note on the recorded state (state_note) and what
    of the upload it did not read, each with its reason (context["source"] unused_*). The model never saw these. The
    prompt states them only where they explain an absence (label/episode.py, the no state block); the board always
    shows them, so a field the model never saw is never a silent gap. None when there is nothing to say."""
    src = ctx.get("source") if isinstance(ctx.get("source"), dict) else {}
    left = {kind: [str(x) for x in src[key]] for key, kind in READER_LEFT_OUT
            if isinstance(src.get(key), (list, tuple)) and src[key]}
    note = ctx["state_note"].strip() if isinstance(ctx.get("state_note"), str) else ""
    if not note and not left:
        return None
    return {**({"state_note": note} if note else {}), **({"left_out": left} if left else {})}


_FAMILIES = None


def label_failure(result: dict | None) -> list[dict]:
    """The data issue of an episode whose model reply gave no labels (board/to_board.py label_failed): one entry of
    kind model_reply_cut_off or model_reply_unparsed, which raises the family label-failed, so the episode is on the
    board with its footage, checks and sensors and the filter finds it. A long recording stitched from the parts that
    parsed (label/pieces.py stitch_run) has one entry of kind part_not_labelled per part that gave none, at its span.
    Nothing for a reply that parsed whole."""
    if not isinstance(result, dict):
        return []
    st = result.get("stitched") if isinstance(result.get("stitched"), dict) else {}
    gaps = [{"kind": "part_not_labelled", "t0_s": g.get("t0_s"), "t1_s": g.get("t1_s"),
             "what": f"Part {g.get('part')} of {st.get('parts')} of this long recording, from {float(g['t0_s']):.1f} s "
                     f"to {float(g['t1_s']):.1f} s, has no labels, as {g.get('why') or 'its reply gave none'}; the "
                     "labels come from the other parts."}
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
                 "what": f"The model gave no reply for this episode ({str(lf.get('why') or 'no reason recorded').rstrip('.')}"
                         f"), {rest}."}]
    if lf["status"] == "not_shown":
        return [{"kind": "model_reply_not_shown",
                 "what": f"The model's reply parsed but the board could not read it ({lf.get('error')}), {rest}; the "
                         "reply is kept as it came."}]
    if lf["status"] == "unreadable":
        return [{"kind": "label_output_unreadable",
                 "what": f"The labelling run's output file for this episode does not read ({lf.get('error')}), "
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
    """The data issues of a reply that parsed but broke the output format: one of kind model_reply_off_schema when it
    has no timeline (it gave none of the steps every rig's format asks for, so the episode has no steps, and what it did
    give is shown), and one of kind model_reply_fields_dropped naming the rows and fields left out of what is shown for
    being of the wrong type (board/to_board.py typed), for a long recording those of every part. Nothing for a reply
    that keeps to the format."""
    if "labels" not in result:
        return []
    out = []
    labels, _ = typed(result)
    if not isinstance(labels.get("timeline"), list) and not result.get("stitched"):
        missing = [k for k in SCHEMA_KEYS if k not in labels]
        gave = sorted(k for k in labels if not str(k).startswith("_"))
        out.append({"kind": "model_reply_off_schema",
                    "what": "The model's reply has no timeline, so this episode has no steps. It leaves out "
                            + ", ".join(missing) + (f" and gives only {', '.join(gave)}" if gave else " and gives nothing")
                            + "; what it gives is shown."})
    left = off_schema_text(result)
    if left:
        out.append({"kind": "model_reply_fields_dropped",
                    "what": f"The model's reply broke the output format, so {left} are left out of what is shown; the "
                            "rest of its labels are shown."})
    return out


STEP_SLACK_S = 0.5      # a step may end this far past the episode's end (a reply rounding its last time up)


def steps_outside(d: dict) -> list[dict]:
    """The data issue of a timeline whose steps lie past the episode's end, or end before they start: kind
    model_steps_outside_episode, naming how many of each. The steps are kept as the model gave them; the page draws
    the timeline to the episode's length with each such step at its edge. Nothing when the episode's length is not
    known (or only estimated) or every step lies within it."""
    dur = d.get("duration_s")
    if not dur or d.get("duration_estimated"):
        return []
    steps = [e for e in d.get("event_labels") or [] if isinstance(e, dict)]
    num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool)
    past = [e for e in steps if any(num(e.get(k)) and e[k] > dur + STEP_SLACK_S for k in ("t_s", "end_s"))]
    back = [e for e in steps if num(e.get("t_s")) and num(e.get("end_s")) and e["end_s"] < e["t_s"]]
    if not past and not back:
        return []
    one = lambda xs, a, b: a if len(xs) == 1 else b
    said = []
    if past:
        said.append(f"{len(past)} of its {len(steps)} steps {one(past, 'lies', 'lie')} past the episode's end at "
                    f"{dur:.1f} s")
    if back:
        said.append(f"{len(back)} {one(back, 'ends before it starts', 'end before they start')}")
    return [{"kind": "model_steps_outside_episode",
             "what": "The model's timeline does not fit the episode: " + " and ".join(said) + ". They are kept as "
                     "given, and the timeline is drawn to the episode's length with them at its edge."}]


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
    found = list(ctx.get("reader_issues") or []) + label_failure(result)
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
    d["_rig"] = ctx.get("profile")
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
    outside = [{**x, "family": _FAMILIES.reader_family(x["kind"])} for x in steps_outside(d)]
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
    unshown = [{"view": v, "name": str(e.get("name") or v), "why": str(e.get("why") or "")}
               for v, e in unshown_views(ctx) if v not in uncut]
    if unshown:
        d["unshown_cameras"] = unshown
    add_contacts(d, ctx, result)


UPLOADER_LIST_MAX = 24       # a list of more numbers than this (a calibration matrix is 16) is summarised by its length


def uploader_groups(notes, start_s, dur_s) -> list[dict]:
    """The notes an upload sent with an episode (prepare/formats.py uploader_notes), as sent, for the board: each table
    row that names the episode as one group, the notes read from its files as another, every value under its own
    name. A number that is a time on the recorder's clock (seconds, ms, us or ns from its size, landing within the
    episode once its first frame's time clock_start_s is taken off) also carries that moment in the episode, so the
    board can jump to it. Only a recorder clock far from zero is read this way, so a count or an index is never taken
    for a time."""
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
        elif isinstance(x, list) and x and all(isinstance(i, (dict, list)) for i in x):
            for i, v in enumerate(x):
                flat(v, path + [str(i + 1)] if len(x) > 1 else path, out)
        else:
            if isinstance(x, list):
                v = (", ".join(str(i) for i in x) if len(x) <= UPLOADER_LIST_MAX
                     else f"{len(x)} values, {', '.join(str(i) for i in x[:4])}, ...")
            else:
                v = str(x)
            item = {"name": " / ".join(path) or "note", "value": v}
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


def add_contacts(d: dict, ctx: dict, result: dict | None = None) -> None:
    """The recording's contacts (context["contacts"], label/contacts.py), each with the model's answer when it was shown
    ("seen"), and the check of one against the other (checks/contacts.py) in dataset_checks["contact_checks"]. An
    episode prepared before contacts were measured has none in its context; labelling found them on the recording
    (label/episode.py build_request, label/pieces.py write_pieces) and its result holds them, so those are used."""
    recorded = ctx.get("contacts") or (result or {}).get("contacts") or []
    if not recorded:
        for k in ("contacts_model", "contact_views"):
            d.pop(k, None)
        return
    from checks import contacts as cc
    from label import contacts as lc
    # a contact timed by a signal placed from both starts says so, also one found before contacts carried it
    recorded = lc.mark_aligned(recorded, {s["name"]: s for s in ctx.get("signals") or []
                                          if isinstance(s, dict) and s.get("name")})
    seen = {c.get("id"): c for c in d.pop("contacts_model", None) or []}
    views = d.pop("contact_views", None) or {}
    shown = set(views.get("shown") or [])
    d["contacts"] = [{**c, **({"seen": seen[c["id"]]} if c["id"] in seen else {}), "shown": c["id"] in shown}
                     for c in recorded]
    res = cc.check({"contacts": list(seen.values()), "contacts_missing": d.get("contacts_missing") or []}, recorded,
                   views.get("strips") or {}, float(ctx.get("fps") or 30))
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
                "dataset_episode_labels", "uploader_notes", "dataset_source", "camera_views", "camera_labels",
                "timesteps_s", "task_label", "reader_notes", "unshown_cameras")


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
        shutil.rmtree(out)
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
                d = convert(json.loads(r["path"].read_text()), bd.get("dataset"))
                d.update(ctx)
                d["_meta"] = {**(d.get("_meta") or {}), **meta}
            else:
                d = {**ctx, "episode_prompt": "", "event_labels": [], "_meta": dict(meta)}
                if r.get("cost") is not None:
                    d["_usage"] = {"est_cost_usd": r.get("cost"), "latency_s": r.get("latency"),
                                   "completion_tokens": r.get("out_tokens")}
            info = {"key": m["key"], "name": m["episode_name"], "status": r["status"], "model": m["model"],
                    "example": m["example"]}
            if r["status"] == "unparsed":
                info.update({"parse_error": r.get("error"), "raw_head": (r.get("raw") or "")[:3000],
                             "raw_chars": len(r.get("raw") or "")})
            if r["status"] == "cut_off":
                info.update({"out_tokens": r.get("out_tokens"), "tail": (r.get("tail") or "")[-1500:]})
            d["_compare"] = info
            (out / m["key"]).mkdir(exist_ok=True)
            (out / m["key"] / f).write_text(dumps(d))
            srcs[m["key"]] = r["status"]
            written[m["key"]] += 1
        if srcs:
            index["episodes"][f] = srcs
    (out / "index.json").write_text(dumps(index, separators=(",", ":")))
    (out / "metrics.json").write_text(dumps(mc.public(metrics), separators=(",", ":")))
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
            shutil.rmtree(d)
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


def build_sensors(board: Path, episodes: dict) -> dict | None:
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
        shutil.rmtree(out)
    out.mkdir()
    res = bs.build(have, out)
    for sk in res["skipped"]:
        print(f"sensors: skipped {sk['file']}: {sk['skip']}", file=sys.stderr)
    return {"written": res["written"], "skipped": len(res["skipped"]), "bytes": res["bytes"]}


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


def _swap(board: Path, name: str, keep: bool) -> None:
    """Replaces BOARD/<name> with BOARD/<name>.new (derived files only; the runs they came from are untouched).
    Without keep, the old folder is removed and nothing replaces it."""
    old, new = board / name, board / f"{name}.new"
    if old.exists():
        shutil.rmtree(old)
    if keep:
        new.rename(old)


def unlabelled(entry: dict, eps: Path, run: Path, labels: dict) -> dict:
    """{board file: (episode name, None, output, run)} for every prepared episode of the entry (an episode_* folder with
    a context.json) that no run of it has an output for: the run never recorded a reply for it (stopped before it, or
    written before the harness kept why), so it is shown with its footage, checks and sensors and says so
    (label_failed no_reply)."""
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


def episode_file(entry: dict, manifest: dict, fname: str, name: str, r: dict, info: dict, eps: Path) -> dict:
    """One episode's board file: its label (board/to_board.py convert), provenance, context, rules, consistency
    check and the parts it was stitched from."""
    d = convert(r, entry["dataset"])
    d["_run"] = {"run_id": info["run_id"], "code": info["code"], "kind": info["kind"], "slice": info.get("slice")}
    ctx_p = eps / name / "context.json"
    ctx = json.loads(ctx_p.read_text()) if ctx_p.exists() else {}
    if manifest.get("labels_license"):
        d["labels_license"] = manifest["labels_license"]    # travels with the label into every download
    if ctx:
        add_context(d, ctx, eps / name, r)
    else:
        add_reader_issues(d, {}, r)       # a reply that gave no labels is flagged with or without a context
    # after the checks are in; a rule that needs the context (fixed_window) skips where there is none
    apply_rules(d, ctx, entry.get("rules") or [])
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
    return normalize_enums(d)


def build(board: Path) -> dict:
    manifest = json.loads((board / "manifest.json").read_text())
    here = board.resolve()
    new = board / "qa.new"
    if new.exists():
        shutil.rmtree(new)
    new.mkdir(parents=True)
    counts, built_entries = {}, []
    board_src = {}      # board file -> the run output its label came from
    episodes = {}       # board file -> its prepared episode folder
    for entry in manifest.get("datasets", []):
        skipped: list = []
        run, labels = entry_labels(entry, here, skipped)
        infos = {}
        eps = _path(entry["episodes"], here)
        labels.update(unlabelled(entry, eps, run, labels))
        for fname, (name, src, r, from_run) in sorted(labels.items()):
            info = infos.get(from_run) or infos.setdefault(from_run, json.loads((from_run / "run.json").read_text()))
            try:
                d = episode_file(entry, manifest, fname, name, r, info, eps)
            except Exception as e:  # noqa: BLE001 - this episode is shown with its reply and the error, the build goes on
                print(f"board: {fname}: the reply could not be read ({type(e).__name__}: {e})", file=sys.stderr)
                d = episode_file(entry, manifest, fname, name, not_shown(r, e), info, eps)
            if (eps / name / "context.json").exists():
                episodes[fname] = eps / name
            dest = new / fname
            if dest.exists():
                raise RuntimeError(f"{dest.name} comes from two manifest entries; a board holds one label per "
                                   "episode (file_prefix separates datasets whose episode names repeat)")
            dest.write_text(dumps(d))
            if src is not None:
                board_src[fname] = src
        counts[entry["dataset"]] = {"run_id": json.loads((run / "run.json").read_text())["run_id"],
                                    "episodes": len(labels), **({"skipped": skipped} if skipped else {})}
        # BUILT.json names the run that was used, so the board's inputs stay traceable
        built_entries.append(dict(entry, run=os.path.relpath(run, here)) if run != _path(entry["run"], here)
                             else entry)
    compared = build_comparisons(board, manifest, new, board_src) if manifest.get("comparisons") else None
    hands = build_hands(board, manifest["hands"], new, episodes) if manifest.get("hands") else None
    sensors = build_sensors(board, episodes) if manifest.get("sensors", True) is not False else None
    _swap(board, "qa", True)
    _swap(board, "compare", compared is not None)
    _swap(board, "hands", hands is not None)
    _swap(board, "hand_keypoints", hands is not None)
    _swap(board, "sensors", sensors is not None)
    built = {"manifest": {**manifest, "datasets": built_entries}, "counts": counts,
             **({"comparisons": compared} if compared else {}), **({"hands": hands} if hands else {}),
             **({"sensors": sensors} if sensors else {})}
    (board / "BUILT.json").write_text(dumps(built, indent=1))
    return built




def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m board build", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("board", type=Path, help="the board folder, with manifest.json")
    a = ap.parse_args()
    print(json.dumps(build(a.board)["counts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
