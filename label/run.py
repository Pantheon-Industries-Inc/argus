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
import os
import re
import subprocess
import sys
from pathlib import Path

from label.harness import episode_cost, get_keys, is_openrouter_key

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
            n += bool(json.loads((run / "out" / f"{e}.json").read_text()).get("parse_ok"))
        except (OSError, ValueError):
            pass
    return n


def billed_cost(run: Path) -> float:
    """The billed cost of every episode in out/, replies cut off at the output limit included (each output records
    its own calls' cost)."""
    total = 0.0
    for p in sorted((run / "out").glob("*episode_*.json")) if (run / "out").exists() else []:
        total += episode_cost(json.loads(p.read_text()))
    return total


def footage_hours(slice_dir: Path, names: list[str]) -> float:
    """Hours of recording in these episodes of the slice, from each context.json's state frame count and fps."""
    s = 0.0
    for n in names:
        c = json.loads((slice_dir / n / "context.json").read_text())
        s += (c.get("n_state_frames") or 0) / float(c.get("fps") or 30)
    return s / 3600


def harness_cmd(slice_dir: Path, run: Path, kind: str, cap: float, concurrency: int, extra: list[str]) -> list[str]:
    cmd = [str(PY), "-m", "label.harness", "--episodes-root", str(slice_dir.resolve()), "--out-dir",
           str((run / "out").resolve()), "--concurrency", str(concurrency)]
    cmd += ["--dry-run"] if kind == "dry" else ["--max-spend", f"{cap:.2f}"]
    return cmd + [x for x in extra if x != "--"]


def finish(run: Path, info: dict, cmd: list, slice_dir: Path, mode: str) -> None:
    env = dict(os.environ)
    keys = _keys().split(",") if info["kind"] != "dry" else []
    env["OPENROUTER_API_KEYS"] = ",".join(k for k in keys if is_openrouter_key(k))
    env["OPENAI_API_KEY"] = ",".join(k for k in keys if not is_openrouter_key(k))
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    # few malloc arenas, so a long multithreaded run does not keep freed frame buffers in per-thread heaps
    env.setdefault("MALLOC_ARENA_MAX", "2")
    with open(run / "log.txt", mode) as log:
        rc = subprocess.run(cmd, cwd=run, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    tail = (run / "log.txt").read_text().strip().splitlines()
    last = next((l for l in reversed(tail) if l.startswith("done=")), "")
    m = dict(re.findall(r"(\w+)=\$?([\d.]+)", last))
    done_eps = sorted(p.stem for p in (run / "out").glob("episode_*.json")) if (run / "out").exists() else []
    hours = footage_hours(slice_dir, done_eps) if done_eps else 0.0
    cost = round(billed_cost(run), 2)
    info.update({"status": "done" if rc == 0 else f"exit {rc}",
                 "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
                 "cost_usd": cost, "episodes_done": len(done_eps), "episodes_parsed": parsed(run, done_eps),
                 "episodes_failed": int(m.get("failed", 0)),
                 "footage_hours": round(hours, 3), "usd_per_hour": round(cost / hours, 2) if hours and cost else None})
    (run / "run.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: info[k] for k in ("run_id", "status", "cost_usd", "episodes_done", "episodes_failed",
                                           "footage_hours", "usd_per_hour")} | {"run": str(run)}))


def resume(a, slice_dir: Path) -> None:
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
    left = float(info["cap_usd"]) - spent
    if info["kind"] != "dry" and left <= 0:
        raise SystemExit(f"--resume: the run's cap ${info['cap_usd']} is already spent (${spent:.2f})")
    if not any((run / "out").glob("episode_*.json")):
        raise SystemExit(f"--resume: no outputs in {run / 'out'}; refusing (wrong folder?)")
    cmd = harness_cmd(slice_dir, run, info["kind"], left, a.concurrency, a.harness_args)
    info.setdefault("resumes", []).append({"at": dt.datetime.now().isoformat(timespec="seconds"), "why": a.why,
                                           "spent_before": round(spent, 2), "command": cmd})
    info["status"] = "running"
    (run / "run.json").write_text(json.dumps(info, indent=1))
    finish(run, info, cmd, slice_dir, "a")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="the dataset's short name, e.g. molmo, fastumi, openaoe")
    ap.add_argument("--episodes", type=Path, required=True, help="the slice: a folder of prepared episode_* folders")
    ap.add_argument("--kind", required=True, choices=["dry", "smoke", "full"], help="dry calls no model")
    ap.add_argument("--cap", type=float, default=0.0, help="most this run may spend in USD (required unless dry)")
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
        if a.cap <= 0:
            raise SystemExit("a paid run needs --cap")
        if dirty:
            raise SystemExit("refused: the checkout has uncommitted changes, so run.json could not name the code "
                             "that made these labels; commit or stash them first")
    now = dt.datetime.now()
    run_id = f"{now:%Y%m%d-%H%M}_{a.kind}_{sha}" + (f"_{a.label}" if a.label else "")
    run = a.runs / a.dataset / run_id
    run.mkdir(parents=True, exist_ok=False)
    cmd = harness_cmd(a.episodes, run, a.kind, a.cap, a.concurrency, a.harness_args)
    info = {"run_id": run_id, "kind": a.kind, "dataset": a.dataset, "slice": str(a.episodes.resolve()),
            "code": sha, "code_dirty": dirty, "started_at": now.isoformat(timespec="seconds"), "cap_usd": a.cap,
            "status": "running", "command": cmd, "note": a.note}
    (run / "run.json").write_text(json.dumps(info, indent=1))
    finish(run, info, cmd, a.episodes, "w")


if __name__ == "__main__":
    main()
