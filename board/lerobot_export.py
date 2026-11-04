"""Write Argus labels into a LeRobot dataset's two language columns, language_persistent and language_events.

    python -m board export-lerobot --dataset DATASET --labels LABELS --out OUT [--report REPORT]

DATASET is a LeRobot dataset on disk (meta/info.json, data/, videos/). LABELS is a folder of Argus episode files,
the JSON the board serves and downloads (board/to_board.py), one per labelled episode. OUT is a new folder that
receives a copy of the dataset with every data file rewritten to carry the two columns (videos and the other
files are hard-linked when the disk allows, else copied). The dataset itself is never written to.

An episode file belongs to the LeRobot episode whose index its name carries (episode_<index>, as
`python -m prepare lerobot` names the episodes it prepares), read from _meta.run_episode, then _meta.episode_id,
then the file name. Argus times are seconds from the episode's first frame.

The columns follow LeRobot's schema (src/lerobot/datasets/language.py upstream) and its writer's layout
(src/lerobot/annotations/steerable_pipeline/writer.py): every frame of an episode carries the episode's whole
persistent list, a frame carries the event rows whose time is that frame's own timestamp, each episode is its own
row group, and the older subtask_index and tools columns are dropped (from meta/info.json too).

- language_persistent holds `subtask` rows. The Argus timeline can run a step for each side at once, while LeRobot
  keeps exactly one subtask active on every frame, so a new subtask row starts at every moment a step starts or ends,
  and its text names every step running then (one side's step alone, or "left: ...; right: ..."). A stretch with no
  step is the subtask "idle", which is what Argus calls a stretch where nothing advances the task. The first row
  starts on the episode's first frame, so the rows cover the episode with no gap.
- language_events holds one row per timed Argus finding, in five styles LeRobot does not have: `outcome` (the
  episode's verdict and each task's, at the time the goal was reached, else the task's end), `mistake` (each
  operator mistake, at its start), `recovery` (each recovery, when it was recovered, else when it failed),
  `key_event` and `state_change` (at their own times). Each row's content is the finding as JSON, as LeRobot's vqa
  rows carry JSON. A finding with no time is left out and counted in the report.
- Argus writes no plan and no memory rows, since it labels neither. LeRobot's validator only warns about that.

Every time is snapped to the nearest frame timestamp read from the data file, never computed from a frame index and
a rate. A time between the episode's first and last frame lands on the nearer frame, however far apart the two are. A
time more than one frame period before the first frame or after the last is refused: its row is left out and listed
in the report with its distance, and one written more than a frame period from its frame, across dropped frames
for example, is listed in far_snaps with its episode, text and distance. The period is the shorter of 1 / the fps in meta/info.json
and the median gap between the episode's frames (frame_period). When two subtask boundaries round to the same frame,
the later one is kept and the earlier subtask, left with no frame, is listed in subtasks_lost; repeated texts in a row
merge. After writing, the copy is checked (validate below), and
any failure stops the command with exit 1.

The five event styles must be registered in LeRobot before its loaders route them, the way its docs register a
project style: add them to lerobot.datasets.language.EXTENDED_STYLES and EVENT_ONLY_STYLES (EVENT_STYLES below).
This works only for recordings that started as LeRobot datasets. MCAP, HDF5 and plain video have no LeRobot dataset
to write into.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from bisect import bisect_left
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

LANGUAGE_PERSISTENT, LANGUAGE_EVENTS = "language_persistent", "language_events"
# LeRobot's own styles (language.py PERSISTENT_STYLES, EVENT_ONLY_STYLES) and the five this exporter adds
PERSISTENT_STYLES = frozenset({"subtask", "plan", "memory", "motion", "task_aug"})
LEROBOT_EVENT_STYLES = frozenset({"interjection", "vqa", "trace"})
EVENT_STYLES = ("outcome", "mistake", "recovery", "key_event", "state_change")
VIEW_DEPENDENT_STYLES = frozenset({"vqa", "trace"})
IDLE = "idle"
# columns LeRobot's annotation writer drops from a data file when it writes the language columns
DROPPED = ("subtask_index", "tools")
SIDE_ORDER = {"left": 0, "right": 1, "both": 2}


def _row_type(timestamp: bool, json_type) -> pa.StructType:
    fields = [pa.field("role", pa.string(), nullable=False), pa.field("content", pa.string()),
              pa.field("style", pa.string())]
    if timestamp:
        fields.append(pa.field("timestamp", pa.float32(), nullable=False))
    return pa.struct(fields + [pa.field("camera", pa.string()), pa.field("tool_calls", pa.list_(json_type))])


def column_types() -> dict[str, pa.DataType]:
    """The two columns' Arrow types as LeRobot declares them (language.py language_*_arrow_type)."""
    j = pa.json_() if hasattr(pa, "json_") else pa.string()
    return {LANGUAGE_PERSISTENT: pa.list_(_row_type(True, j)), LANGUAGE_EVENTS: pa.list_(_row_type(False, j))}


def _array(rows: list[list[dict]], column: str) -> pa.Array:
    # pyarrow cannot build the JSON extension type from Python values, so the rows are built with string tool calls
    # and cast to LeRobot's exact type (its writer infers the type instead, which loses it on an all-null column)
    plain = pa.list_(_row_type(column == LANGUAGE_PERSISTENT, pa.string()))
    return pa.array(rows, type=plain).cast(column_types()[column])


def episode_index_of(label: dict, path: Path) -> int | None:
    meta = label.get("_meta") if isinstance(label.get("_meta"), dict) else {}
    for name in (meta.get("run_episode"), meta.get("episode_id"), path.stem):
        m = re.search(r"episode_0*(\d+)$", str(name or ""))
        if m:
            return int(m.group(1))
    return None


def _num(x) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def step_text(step: dict) -> str:
    """A timeline step as the board's caption says it (board/index.html buildPhrase): the model's phrase for the step,
    and its destination when it has one."""
    text = str(step.get("verb_class") or "").strip() or (
        f"handle the {step['object']}" if step.get("object") else "move")
    dest = re.sub(r"^->\s*", "", str(step.get("carry_phase") or "").strip())
    return f"{text} -> {dest}" if dest else text


def subtask_spans(label: dict, duration: float) -> list[tuple[float, str]]:
    """(start, text) of each subtask, in order, the first at 0: a new one wherever a step starts or ends, naming every
    step running then, and IDLE where none runs. Consecutive spans with the same text are one span."""
    steps = []
    for e in label.get("event_labels") or []:
        if not isinstance(e, dict):
            continue
        t0, t1 = _num(e.get("t_s")), _num(e.get("end_s"))
        if t0 is None or t1 is None or t1 <= t0:
            continue
        steps.append((max(t0, 0.0), min(t1, duration), str(e.get("arm") or ""), step_text(e)))
    cuts = sorted({0.0, *(s[0] for s in steps), *(s[1] for s in steps)} - {duration})
    spans: list[tuple[float, str]] = []
    for t in cuts:
        if t >= duration:
            continue
        on = sorted((s for s in steps if s[0] <= t < s[1]), key=lambda s: (SIDE_ORDER.get(s[2], 3), s[0], s[3]))
        if not on:
            text = IDLE
        elif len(on) == 1:
            text = on[0][3]
        else:
            text = "; ".join(f"{s[2] or 'step'}: {s[3]}" for s in on)
        if not spans or spans[-1][1] != text:
            spans.append((t, text))
    return spans or [(0.0, IDLE)]


def findings(label: dict) -> list[tuple[str, float | None, dict]]:
    """(style, time, the finding) of every Argus finding that becomes an event row; time None when it has none."""
    out = []
    comp = label.get("completion") if isinstance(label.get("completion"), dict) else {}
    if comp.get("task_completed"):
        t = _num(comp.get("goal_reached_at_s"))
        t = _num(comp.get("completed_at_s")) if t is None else t
        out.append(("outcome", t, {"scope": "episode", **{k: comp[k] for k in (
            "task_completed", "failure_kind", "completed_at_s", "goal_reached_at_s") if comp.get(k) is not None}}))
    for task in label.get("tasks") or []:
        if not isinstance(task, dict) or not task.get("outcome"):
            continue
        t = next((v for v in (_num(task.get("goal_reached_at_s")), _num(task.get("completed_at_s")),
                              _num(task.get("end_s"))) if v is not None), None)
        out.append(("outcome", t, {"scope": "task", **{k: task[k] for k in (
            "task", "outcome", "failure_kind", "start_s", "end_s", "completed_at_s") if task.get(k) not in (None, "")}}))
    for m in label.get("operator_mistakes") or []:
        if isinstance(m, dict):
            out.append(("mistake", _num(m.get("t_s")), m))
    for r in label.get("recovery") or []:
        if isinstance(r, dict):
            t = _num(r.get("recovered_at_s")) if r.get("recovered") else None
            out.append(("recovery", _num(r.get("failure_t_s")) if t is None else t, r))
    for k in label.get("key_events") or []:
        if isinstance(k, dict):
            out.append(("key_event", _num(k.get("t_s")), k))
    for s in label.get("state_changes") or []:
        if isinstance(s, dict):
            out.append(("state_change", _num(s.get("t_s")), s))
    return out


def finding_text(item: dict) -> str:
    """A finding's own words, for the report: its label, issue, task, object or failure, whichever it has first."""
    for k in ("label", "issue", "task", "failure", "object"):
        if isinstance(item.get(k), str) and item[k].strip():
            return item[k].strip()
    return ""


def _snap(times: list[float], t: float) -> tuple[float, float]:
    """(nearest frame timestamp, its distance from t)."""
    i = bisect_left(times, t)
    best = min((times[j] for j in (i - 1, i) if 0 <= j < len(times)), key=lambda f: (abs(f - t), f))
    return best, abs(best - t)


def frame_period(fps, frame_ts: list[float]) -> float:
    """One frame period, the farthest a time before the first frame or after the last may sit from it and still be
    placed on it (a time between them always lands on the nearer frame): the shorter of 1 / the
    dataset's declared fps and the median positive gap between the episode's frames. Each bound covers a failure of the
    other. A declared 10 fps on frames that run at 30 fps would allow 100ms and place a time 2.5 frames past the end on
    the last frame (issue 196), and the median gap catches it. Dropped frames lower the rate over the recorded span
    (30% dropped at 30 fps reads as about 21 fps), but not the median gap, so the period stays 33ms; that is why the
    span rate of prepare/frame_rate.py measured_rate is not used here. The smallest gap is never used either, since
    one late frame shrinks it to a millisecond. Without a declared fps the median gap alone sets it."""
    gaps = np.diff(np.asarray(frame_ts, dtype=np.float64))
    gaps = gaps[np.isfinite(gaps) & (gaps > 0)]
    bounds = [float(np.median(gaps))] if len(gaps) else []
    declared = fps if isinstance(fps, (int, float)) and not isinstance(fps, bool) and math.isfinite(fps) and fps > 0 \
        else None
    if declared:
        bounds.append(1.0 / declared)
    return min(bounds) if bounds else 0.0


def episode_rows(label: dict, frame_ts: list[float], report: dict, name: str,
                 fps=None) -> tuple[list[dict], dict[float, list]]:
    """The episode's persistent rows and its event rows by frame timestamp. frame_ts are the episode's frame
    timestamps from the data file, sorted, and fps the dataset's declared rate (None when it declares none)."""
    period = frame_period(fps, frame_ts)
    start, duration = frame_ts[0], frame_ts[-1] - frame_ts[0] + period

    def place(t: float, what: str, text: str):
        # a time between the first and last frame always has a frame on each side and lands on the nearer one, even
        # across dropped frames or under a declared rate above the frames' own; only one past either end by more than
        # a frame period is refused. One written more than a frame period from its frame is listed in far_snaps.
        at, off = _snap(frame_ts, start + t)
        row = {"episode": name, "what": what, "text": text, "t_s": t, "nearest_frame_s": at - start,
               "off_s": round(off, 6)}
        if off > period + 1e-6:
            if not frame_ts[0] <= start + t <= frame_ts[-1]:
                report["refused"].append(row)
                return None
            report["far_snaps"].append(row)
        report["max_snap_s"] = max(report["max_snap_s"], off)       # of the rows written
        return at

    # Two boundaries that round to one frame (a step ending at 2.0s and the next starting at 2.01s) keep the later
    # one, which is the subtask in force from that frame on, and the earlier one, left with no frame, is listed in
    # subtasks_lost; then repeated texts in a row merge into one subtask.
    on_frame: dict[float, tuple[float, str]] = {}
    for t, text in subtask_spans(label, duration):
        at = place(t, "subtask", text)
        if at is None:
            continue
        if at in on_frame and on_frame[at][1] != text:
            report["subtasks_lost"].append({"episode": name, "text": on_frame[at][1], "t_s": on_frame[at][0],
                                            "frame_s": at - start, "replaced_by": text})
        on_frame[at] = (t, text)
    on_frame = {at: text for at, (_, text) in on_frame.items()}
    persistent = []
    for at in sorted(on_frame):
        if persistent and persistent[-1]["content"] == on_frame[at]:
            continue
        persistent.append({"role": "assistant", "content": on_frame[at], "style": "subtask", "timestamp": at,
                           "camera": None, "tool_calls": None})
    if persistent and persistent[0]["timestamp"] != frame_ts[0]:
        persistent[0]["timestamp"] = frame_ts[0]
    events: dict[float, list] = {}
    for style, t, item in findings(label):
        if t is None:
            report["untimed"][style] = report["untimed"].get(style, 0) + 1
            continue
        at = place(t, style, finding_text(item))
        if at is None:
            continue
        events.setdefault(at, []).append({"role": "assistant", "style": style, "camera": None, "tool_calls": None,
                                          "content": json.dumps(item, sort_keys=True, ensure_ascii=False)})
        report["events"][style] = report["events"].get(style, 0) + 1
    for rows in events.values():
        rows.sort(key=lambda r: (r["style"], r["content"]))
    return persistent, events


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_one_row_group_per_episode(table: pa.Table, path: Path) -> None:
    """One row group per run of rows of one episode, as LeRobot's writer re-emits a data file
    (io_utils.write_table_one_row_group_per_episode), through a sibling file renamed into place."""
    eps = table.column("episode_index").to_pylist()
    starts = [i for i in range(len(eps)) if i == 0 or eps[i] != eps[i - 1]] + [len(eps)]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with pq.ParquetWriter(tmp, table.schema) as w:
        for a, b in zip(starts, starts[1:]):
            w.write_table(table.slice(a, b - a), row_group_size=b - a)
    tmp.replace(path)


def data_files(root: Path) -> list[Path]:
    return sorted(p for p in (root / "data").rglob("*.parquet"))


def export(dataset: Path, labels: Path, out: Path) -> dict:
    """Copy DATASET to OUT with both language columns written from the episode files in LABELS. The report."""
    dataset, out = Path(dataset), Path(out)
    if not (dataset / "meta" / "info.json").is_file():
        raise SystemExit(f"{dataset} has no meta/info.json: not a LeRobot dataset")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    by_episode, report = {}, {"labels": 0, "unmatched_labels": [], "episodes_written": 0, "events": {},
                              "untimed": {}, "refused": [], "far_snaps": [], "subtasks_lost": [],
                              "max_snap_s": 0.0}
    for p in sorted(Path(labels).glob("*.json")):
        label = json.loads(p.read_text())
        idx = episode_index_of(label, p) if isinstance(label, dict) else None
        if idx is None:
            report["unmatched_labels"].append(p.name)
            continue
        if idx in by_episode:
            raise SystemExit(f"two episode files for LeRobot episode {idx}: {by_episode[idx][1].name} and {p.name}")
        by_episode[idx] = (label, p)
        report["labels"] += 1
    for src in dataset.rglob("*"):
        rel = src.relative_to(dataset)
        if src.is_file() and not (rel.parts[0] == "data" and src.suffix == ".parquet"):
            _link_or_copy(src, out / rel)
    from prepare.frame_rate import declared_fps
    fps = declared_fps(json.loads((dataset / "meta" / "info.json").read_text()))
    found = set()
    for src in data_files(dataset):
        table = pq.read_table(src)
        if "episode_index" not in table.column_names or "timestamp" not in table.column_names:
            raise SystemExit(f"{src} has no episode_index or timestamp column")
        eps, ts = table.column("episode_index").to_pylist(), table.column("timestamp").to_pylist()
        frames: dict[int, list[float]] = {}
        for e, t in zip(eps, ts):
            frames.setdefault(int(e), []).append(float(t))
        rows = {}
        for e, times in frames.items():
            if e in by_episode:
                label, p = by_episode[e]
                rows[e] = episode_rows(label, sorted(times), report, p.name, fps)
                found.add(e)
        persistent = [rows[int(e)][0] if int(e) in rows else [] for e in eps]
        events = [rows[int(e)][1].get(float(t), []) if int(e) in rows else [] for e, t in zip(eps, ts)]
        keep = [n for n in table.column_names if n not in (LANGUAGE_PERSISTENT, LANGUAGE_EVENTS, *DROPPED)]
        new = table.select(keep).append_column(LANGUAGE_PERSISTENT, _array(persistent, LANGUAGE_PERSISTENT))
        new = new.append_column(LANGUAGE_EVENTS, _array(events, LANGUAGE_EVENTS))
        dst = out / src.relative_to(dataset)
        dst.parent.mkdir(parents=True, exist_ok=True)
        write_one_row_group_per_episode(new, dst)
    report["episodes_written"] = len(found)
    report["labels_without_episode"] = sorted(by_episode.keys() - found)
    info_path = out / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    for name in DROPPED:
        info.setdefault("features", {}).pop(name, None)
    info.setdefault("features", {}).update({c: {"dtype": "language", "shape": [1], "names": None}
                                            for c in (LANGUAGE_PERSISTENT, LANGUAGE_EVENTS)})
    info_path.unlink()          # a hard link to the source's file: replace it, never write through it
    info_path.write_text(json.dumps(info, indent=4))
    report["max_snap_s"] = round(report["max_snap_s"], 6)
    problems = validate(out)
    report["validation_errors"] = problems
    return report


def validate(root: Path) -> list[str]:
    """Checks on the written data. From LeRobot's validator (steerable_pipeline/validator.py) and its writer's
    invariants: every row routes to the column its style belongs in, only view-dependent styles carry a camera, every
    event lies on the frame whose timestamp it fires at, an episode's persistent list is the same on every frame, and
    the columns have LeRobot's types. That the subtasks start on the episode's first frame and at distinct frame
    timestamps (one active subtask on every frame) is the exporter's own rules, not LeRobot's: its validator does not
    check either."""
    errors = []
    events_ok = LEROBOT_EVENT_STYLES | set(EVENT_STYLES)
    for path in data_files(Path(root)):
        t = pq.read_table(path)
        if t.schema.field(LANGUAGE_PERSISTENT).type != column_types()[LANGUAGE_PERSISTENT] or \
                t.schema.field(LANGUAGE_EVENTS).type != column_types()[LANGUAGE_EVENTS]:
            errors.append(f"{path.name}: the language columns do not have LeRobot's types")
        first: dict[int, tuple[float, list]] = {}
        frame_ts: dict[int, set] = {}
        for e, ts, per, ev in zip(t.column("episode_index").to_pylist(), t.column("timestamp").to_pylist(),
                                  t.column(LANGUAGE_PERSISTENT).to_pylist(), t.column(LANGUAGE_EVENTS).to_pylist()):
            # in float32, the type language_persistent stores a subtask start in, whatever type the column has
            frame_ts.setdefault(e, set()).add(float(np.float32(ts)))
            if e not in first:
                first[e] = (ts, per)
            elif per != first[e][1]:
                errors.append(f"ep={e}: the persistent list differs between frames")
            for row in per:
                if row["style"] not in PERSISTENT_STYLES:
                    errors.append(f"ep={e}: style {row['style']!r} in {LANGUAGE_PERSISTENT}")
                if (row["camera"] is not None) != (row["style"] in VIEW_DEPENDENT_STYLES):
                    errors.append(f"ep={e}: camera field on style {row['style']!r}")
            for row in ev:
                if row["style"] not in events_ok:
                    errors.append(f"ep={e}: style {row['style']!r} in {LANGUAGE_EVENTS}")
                if (row["camera"] is not None) != (row["style"] in VIEW_DEPENDENT_STYLES):
                    errors.append(f"ep={e}: camera field on style {row['style']!r}")
                if row["content"] is None and row["tool_calls"] is None:
                    errors.append(f"ep={e}: event row with neither content nor tool calls")
        for e, (ts0, per) in first.items():
            starts = [r["timestamp"] for r in per if r["style"] == "subtask"]
            if not starts:
                continue
            if min(starts) != min(frame_ts[e]):
                errors.append(f"ep={e}: the first subtask starts at {min(starts)}, not on the first frame")
            if len(set(starts)) != len(starts) or any(s not in frame_ts[e] for s in starts):
                errors.append(f"ep={e}: subtask starts are not distinct frame timestamps")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m board export-lerobot", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, required=True, help="the LeRobot dataset's folder (read only)")
    ap.add_argument("--labels", type=Path, required=True, help="a folder of Argus episode files (the board's JSON)")
    ap.add_argument("--out", type=Path, required=True, help="a new folder for the copy with the language columns")
    ap.add_argument("--report", type=Path, default=None, help="also write the report here as JSON")
    a = ap.parse_args()
    report = export(a.dataset, a.labels, a.out)
    text = json.dumps(report, indent=1)
    if a.report:
        a.report.write_text(text)
    print(text)
    return 1 if report["validation_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
