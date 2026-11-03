"""Settling a label run's metadata once its worker ends: how many episodes finished and parsed, their footage hours and
cost, and run.json's final status. label/harness.py calls finalize_worker when a batch ends, so a killed wrapper never
leaves paid work marked as running. It lives apart from label/run.py so the request identity (label/harness.py
request_implementation), which follows every import of the harness, never reaches the long-recording cutter in
label/pieces.py or the board modules that cutter reads."""
from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path

from label.atomic import write_atomic
from label.dictionary_stage import output_spend, output_spend_complete
from label.harness import annotation_complete, annotation_record
from prepare.frame_rate import episode_seconds


def parsed(run: Path, eps: list[str]) -> int:
    """How many of the episodes with an output have a finished annotation, a reply that parsed and holds the work its
    request asked for (label/harness.py annotation_complete; episodes_done counts every output)."""
    n = 0
    for e in eps:
        try:
            selected = annotation_record(run / 'out', f'{e}.json')
            record = selected[1] if selected else {}
            n += isinstance(record, dict) and annotation_complete(record)
        except (OSError, ValueError):
            pass
    return n


def harness_out(run: Path, info: dict) -> Path:
    """Where the harness writes its replies: run/parts when the run cut long recordings (run/out then holds the
    stitched results, label/pieces.py stitch_run), else run/out."""
    return run / ('parts' if info.get('pieces') else 'out')


def billed_cost(run: Path, info: dict | None = None) -> float:
    """Cumulative episode charges and outstanding reservations, counted once: from the harness's own replies, since a
    stitched result in run/out carries the same charges as its parts."""
    return output_spend(harness_out(run, info or {}))


def footage_hours(slice_dir: Path, names: list[str]) -> float:
    """Hours of recording, each episode's length from prepare/frame_rate.py episode_seconds (its duration_s, else the
    span of its capture times, else its frame count at its frame rate)."""
    s = 0.0
    for n in names:
        c = json.loads((slice_dir / n / "context.json").read_text())
        s += episode_seconds(c, ep_dir=slice_dir / n)
    return s / 3600


def finalize(run: Path, info: dict, slice_dir: Path, rc: int) -> None:
    """Publish settled worker results even if the wrapper was killed."""
    tail = (run / "log.txt").read_text().strip().splitlines()
    last = next((l for l in reversed(tail) if l.startswith("done=")), "")
    m = dict(re.findall(r"(\w+)=\$?([\d.]+)", last))
    done_eps = sorted(p.stem for p in (run / "out").glob("episode_*.json")) if (run / "out").exists() else []
    hours = footage_hours(slice_dir, done_eps) if done_eps else 0.0
    reserved = billed_cost(run, info)
    complete = output_spend_complete(harness_out(run, info))
    cost = reserved if complete else None
    info.update({"status": "done" if rc == 0 else f"exit {rc}",
                 "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
                 "cost_usd": cost, "reserved_cost_usd": reserved, "cost_complete": complete, "episodes_done": len(done_eps), "episodes_parsed": parsed(run, done_eps),
                 "episodes_failed": int(m.get("failed", 0)),
                 "footage_hours": round(hours, 3), "usd_per_hour": round(cost / hours, 2) if hours and cost else None})
    write_atomic(run / "run.json", info, indent=1)


def finalize_worker(out_dir: Path, rc: int) -> None:
    metadata = os.environ.get('ARGUS_STAGED_RUN_METADATA')
    if not metadata:
        return
    path = Path(metadata)
    if path.name != 'run.json' or path.parent.resolve() != Path(out_dir).resolve().parent:
        return
    info = json.loads(path.read_text())
    if info.get('kind') in ('dry', 'smoke', 'full') and info.get('status') == 'running':
        finalize(path.parent, info, Path(info['slice']), rc)
    elif info.get('kind') == 'review' and info.get('status') == 'running':
        # A review whose wrapper was killed still has its labels settled here; the wrapper, if alive, writes its own
        # final status after this, and a rerun of review builds the board.
        from label.dictionary_stage import dictionary_spend, label_spend, label_spend_complete
        job = path.parent.parent
        spend = dictionary_spend(job)
        complete = spend['complete'] and label_spend_complete(job)
        total = label_spend(job) + (spend['cost_usd'] or 0.0)
        info.update(status='labelled; board not built (rerun review to build it)' if rc == 0 else f'labelling exit {rc}',
                    cost_usd=total if complete else None, reserved_cost_usd=label_spend(job) + spend['reserved_usd'],
                    cost_complete=complete, labels_finished_at=dt.datetime.now().isoformat(timespec='seconds'))
        write_atomic(path, info, indent=1)
