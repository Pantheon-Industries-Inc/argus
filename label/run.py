"""Start a labelling run: the harness over one slice of prepared episodes, into a run folder of its own.

    python -m label --dataset molmo --episodes data/episodes/molmo/quickstart --kind smoke --cap 2

- The run folder is RUNS/<dataset>/<run_id>, run_id = <YYYYmmdd-HHMM>_<kind>_<commit>[_<label>]. It never
  existed before; a run is never rerun into. RUNS defaults to data/runs.
- kind is dry (build every request, call nothing, free), smoke (a small paid run to measure cost and read the
  output first) or full.
- run.json records the commit of this checkout, the slice, the exact command (with the model settings), the cap
  and --note, and at the end the billed cost, episode counts, footage hours and cost per hour. log.txt holds the
  harness's output. A paid run refuses to start (or resume) from a checkout with uncommitted changes to tracked
  files, so every label can be traced to the code that made it.
- --cap is the most a paid run may spend, in USD; it is required unless the kind is dry.
- --resume RUN_DIR --why TEXT finishes a run that was killed from outside: same folder, commit, slice and kind,
  and what is left of its cap; episodes already labelled are skipped, the log is appended to, and run.json
  records each resume and why. --dataset, --episodes and --kind must be given again and match run.json.
- Keys come from OPENROUTER_API_KEYS, or when it holds none from OPENAI_API_KEY (label.harness.get_keys).

Anything after -- goes to the harness (label/harness.py), e.g. -- --model anthropic/claude-opus-5.5.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

from label.harness import annotation_complete, annotation_record, execution_lock, get_keys, is_openrouter_key
from label.dictionary_stage import output_spend, output_spend_complete
from label.atomic import write_atomic
from prepare.frame_rate import frames_seconds

REPO = Path(__file__).resolve().parent.parent
PY = Path(sys.executable)


def _keys() -> str:
    keys = get_keys()
    if not keys:
        raise SystemExit("set OPENROUTER_API_KEYS (comma-separated OpenRouter keys) or OPENAI_API_KEY")
    return ",".join(keys)


def commit() -> tuple[str, bool]:
    """The checkout's commit (7 characters) and whether any tracked file differs from it (True when git fails)."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"], cwd=REPO, capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                                    capture_output=True, text=True, check=True).stdout.strip())
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def parsed(run: Path, eps: list[str]) -> int:
    """How many of the episodes with an output have a reply that parsed (episodes_done counts every output)."""
    n = 0
    for e in eps:
        try:
            selected = annotation_record(run / 'out', f'{e}.json')
            record = selected[1] if selected else {}
            n += isinstance(record, dict) and annotation_complete(record)
        except (OSError, ValueError):
            pass
    return n


def billed_cost(run: Path) -> float:
    """Cumulative episode charges and outstanding reservations, counted once."""
    return output_spend(run / 'out')


def footage_hours(slice_dir: Path, names: list[str]) -> float:
    """Hours of recording from declared spans, with legacy frame-count fallback."""
    s = 0.0
    for n in names:
        c = json.loads((slice_dir / n / "context.json").read_text())
        duration = c.get('duration_s')
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
            duration = frames_seconds(c, ep_dir=slice_dir / n)
        s += duration
    return s / 3600


def harness_cmd(slice_dir: Path, run: Path, kind: str, cap: float, concurrency: int, extra: list[str]) -> list[str]:
    cmd = [str(PY), "-m", "label.harness", "--episodes-root", str(slice_dir.resolve()), "--out-dir",
           str((run / "out").resolve()), "--concurrency", str(concurrency)]
    cmd += ["--dry-run"] if kind == "dry" else ["--max-spend", repr(cap)]
    return cmd + [x for x in extra if x != "--"]


def finish(run: Path, info: dict, cmd: list, slice_dir: Path, mode: str) -> int:
    env = dict(os.environ)
    keys = _keys().split(",") if info["kind"] != "dry" else []
    env["OPENROUTER_API_KEYS"] = ",".join(k for k in keys if is_openrouter_key(k))
    env["OPENAI_API_KEY"] = ",".join(k for k in keys if not is_openrouter_key(k))
    env["ARGUS_STAGED_RUN_METADATA"] = str((run / "run.json").resolve())
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    # few malloc arenas, so a long multithreaded run does not keep freed frame buffers in per-thread heaps
    env.setdefault("MALLOC_ARENA_MAX", "2")
    with open(run / "log.txt", mode) as log:
        rc = subprocess.run(cmd, cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        # An orphan may own the output lock after this wrapper's child refused it.
        # Do not replace its running metadata with the refused child's exit status.
        with execution_lock(run / 'out'):
            pass
    finalize(run, info, slice_dir, rc)
    print(json.dumps({k: info[k] for k in ("run_id", "status", "cost_usd", "episodes_done", "episodes_failed",
                                           "footage_hours", "usd_per_hour")} | {"run": str(run)}))
    return rc



def finalize(run: Path, info: dict, slice_dir: Path, rc: int) -> None:
    """Publish settled worker results even if the wrapper was killed."""
    tail = (run / "log.txt").read_text().strip().splitlines()
    last = next((l for l in reversed(tail) if l.startswith("done=")), "")
    m = dict(re.findall(r"(\w+)=\$?([\d.]+)", last))
    done_eps = sorted(p.stem for p in (run / "out").glob("episode_*.json")) if (run / "out").exists() else []
    hours = footage_hours(slice_dir, done_eps) if done_eps else 0.0
    reserved = billed_cost(run)
    complete = output_spend_complete(run / "out")
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


def resume(a, slice_dir: Path) -> None:
    folder = a.resume.resolve()
    if not (folder / 'run.json').is_file():
        raise SystemExit(f'--resume: no run.json in {folder}')
    with execution_lock(folder):
        # A killed wrapper can leave its harness alive and spending.
        with execution_lock(folder / 'out'):
            pass
        return _resume(a, slice_dir)


def _resume(a, slice_dir: Path) -> None:
    run = a.resume.resolve()
    info = json.loads((run / "run.json").read_text())
    sha, dirty = commit()
    for k, have in (("dataset", a.dataset), ("kind", a.kind), ("code", sha)):
        if str(info[k]) != str(have):
            raise SystemExit(f"--resume: {k} is {info[k]!r} in run.json, not {have!r}")
    if info["kind"] != "dry" and dirty:
        raise SystemExit("--resume refused: the checkout has uncommitted changes; commit or stash them first")
    if str(Path(info["slice"]).resolve()) != str(slice_dir.resolve()):
        raise SystemExit(f"--resume: the run's slice is {info['slice']}, not {slice_dir}")
    if not a.why:
        raise SystemExit("--resume needs --why")
    spent = billed_cost(run)
    if info["kind"] != "dry" and getattr(a, "cap", 0):
        # a new --cap on a resume replaces the run's cap, so a run the cap stopped can be finished
        if not math.isfinite(a.cap) or a.cap <= 0:
            raise SystemExit("--resume: --cap must be a positive amount")
        info.setdefault("cap_changes", []).append({"at": dt.datetime.now().isoformat(timespec="seconds"),
                                                   "from": info["cap_usd"], "to": a.cap})
        info["cap_usd"] = a.cap
    left = float(info["cap_usd"]) - spent
    if info["kind"] != "dry" and left <= 0:
        raise SystemExit(f"--resume: the run's cap ${info['cap_usd']} is already spent (${spent:.2f})")
    if not any((run / 'out').glob('*episode_*.json')) and not any((run / 'out' / '.evidence').glob('*.inspection.json')):
        raise SystemExit(f"--resume: no outputs in {run / 'out'}; refusing (wrong folder?)")
    cmd = harness_cmd(slice_dir, run, info["kind"], left, a.concurrency, a.harness_args)
    info.setdefault("resumes", []).append({"at": dt.datetime.now().isoformat(timespec="seconds"), "why": a.why,
                                           "spent_before": spent, "command": cmd})
    info["status"] = "running"
    write_atomic(run / "run.json", info, indent=1)
    return finish(run, info, cmd, slice_dir, "a")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="the dataset's short name, e.g. molmo, fastumi, openaoe")
    ap.add_argument("--episodes", type=Path, required=True, help="the slice: a folder of prepared episode_* folders")
    ap.add_argument("--kind", required=True, choices=["dry", "smoke", "full"], help="dry calls no model")
    ap.add_argument("--cap", type=float, default=0.0, help="most this run may spend in USD (required unless dry); "
                                                           "with --resume, a new cap for the run")
    ap.add_argument("--runs", type=Path, default=REPO / "data" / "runs", help="where run folders go")
    ap.add_argument("--concurrency", type=int, default=16, help="episodes in flight (default 16)")
    ap.add_argument("--note", default="", help="free text recorded in run.json")
    ap.add_argument("--label", default="", help="appended to the run id, e.g. the model of a comparison run")
    ap.add_argument("--resume", type=Path, default=None, help="finish this killed run")
    ap.add_argument("--why", default="", help="with --resume: what killed the run")
    ap.add_argument("harness_args", nargs=argparse.REMAINDER, help="after --: flags for label/harness.py")
    a = ap.parse_args()
    if not a.episodes.is_dir():
        raise SystemExit(f"no slice folder {a.episodes}")
    if a.resume:
        return resume(a, a.episodes)
    sha, dirty = commit()
    if a.kind != "dry":
        if not math.isfinite(a.cap) or a.cap <= 0:
            raise SystemExit("a paid run needs --cap")
        if dirty:
            raise SystemExit("refused: the checkout has uncommitted changes, so run.json could not name the code "
                             "that made these labels; commit or stash them first")
    now = dt.datetime.now()
    base_id = f"{now:%Y%m%d-%H%M}_{a.kind}_{sha}" + (f"_{a.label}" if a.label else "")
    run_id = base_id
    (a.runs / a.dataset).mkdir(parents=True, exist_ok=True)
    for n in range(2, 1000):
        try:
            (a.runs / a.dataset / run_id).mkdir()
            break
        except FileExistsError:
            run_id = f"{base_id}-{n}"          # two runs started in the same minute get their own folders
    run = a.runs / a.dataset / run_id
    cmd = harness_cmd(a.episodes, run, a.kind, a.cap, a.concurrency, a.harness_args)
    info = {"run_id": run_id, "kind": a.kind, "dataset": a.dataset, "slice": str(a.episodes.resolve()),
            "code": sha, "code_dirty": dirty, "started_at": now.isoformat(timespec="seconds"), "cap_usd": a.cap,
            "status": "running", "command": cmd, "note": a.note}
    with execution_lock(run):
        write_atomic(run / "run.json", info, indent=1)
        return finish(run, info, cmd, a.episodes, "w")


if __name__ == "__main__":
    from label.harness import cli
    raise SystemExit(cli(main))
