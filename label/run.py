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
- A recording longer than one request allows (label/pieces.py piece_max: 450s at a rig's default sampling) is cut
  into parts at still moments, the parts are labelled with the other episodes, and their labels are stitched back
  onto the recording's timeline, as python -m review does. The run's policy (the harness's --model, --sample-fps and
  the rest) is written into the slice's sidecars first (label/policy.py persist), so the parts are cut to the limit
  of the model that labels them. Replies go to run/parts and run/out holds one result per recording; run.json's
  "pieces" names each cut recording's parts. A slice with no long recording runs as before, into run/out.

Anything after -- goes to the harness (label/harness.py), e.g. -- --model anthropic/claude-opus-5.5.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from label.harness import execution_lock, get_keys, is_openrouter_key, resolve_unanswered_charges
from label.atomic import write_atomic
from label.checkout import commit
from label.finalize import billed_cost, finalize, finalize_worker, footage_hours, harness_out, parsed

REPO = Path(__file__).resolve().parent.parent
PY = Path(sys.executable)


def _keys() -> str:
    keys = get_keys()
    if not keys:
        raise SystemExit("set OPENROUTER_API_KEYS (comma-separated OpenRouter keys) or OPENAI_API_KEY")
    return ",".join(keys)


def run_policy(extra: list[str]) -> dict:
    """The policy settings the harness is given after -- (--model, --reasoning, --sample-fps, --dataset-note,
    --inspection), as label/policy.py resolve takes them."""
    from label import policy
    ap = argparse.ArgumentParser(add_help=False)
    policy.add_arguments(ap)
    known, _ = ap.parse_known_args([x for x in extra if x != "--"])
    return {k: getattr(known, k) for k in ("model", "reasoning", "sample_fps", "dataset_note", "inspection")}


def plan_pieces(run: Path, slice_dir: Path, extra: list[str]) -> dict:
    """Cut every recording of the slice that one request cannot hold into parts, as python -m review does
    (label/pieces.py write_units): {recording: [its parts]}. The run's policy is written into the slice's sidecars
    first, so the parts are cut to the limit of the model that labels them; a slice with no long recording is left as
    it is and gets {}."""
    from label import pieces, policy
    overrides = run_policy(extra)
    longer = False
    for d in sorted(slice_dir.glob("episode_*")):
        try:
            ctx = json.loads((d / "context.json").read_text())
        except (OSError, ValueError):
            continue
        if pieces.duration(ctx, d) > policy.resolve(ctx, **overrides)["part_max_s"]:
            longer = True
            break
    if not longer:
        return {}
    policy.persist(slice_dir, **overrides)
    return pieces.write_units(run, slice_dir)


def stitch(run: Path, info: dict, slice_dir: Path) -> None:
    """One result per recording in run/out: the short episodes' replies as they came and each cut recording stitched
    from its parts (label/pieces.py stitch_run). A stitch that fails is recorded, never hidden."""
    if not info.get("pieces"):
        return
    from label import pieces
    try:
        info["stitched"] = pieces.stitch_run(run, slice_dir, info["pieces"], run / "out", src=run / "parts")
        info.pop("stitch_error", None)
    except Exception as error:  # noqa: BLE001 - recorded in run.json, and a resume stitches again
        info["stitch_error"] = repr(error)[:500]


def harness_cmd(slice_dir: Path, run: Path, kind: str, cap: float, concurrency: int, extra: list[str],
                split: bool = False) -> list[str]:
    """The harness over the slice, or over run/units (each short episode and each part of a cut recording) into
    run/parts when the run cut long recordings."""
    root, out = (run / "units", run / "parts") if split else (slice_dir, run / "out")
    cmd = [str(PY), "-m", "label.harness", "--episodes-root", str(root.resolve()), "--out-dir",
           str(out.resolve()), "--concurrency", str(concurrency)]
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
        with execution_lock(harness_out(run, info)):
            pass
    stitch(run, info, slice_dir)
    finalize(run, info, slice_dir, rc)
    print(json.dumps({k: info[k] for k in ("run_id", "status", "cost_usd", "episodes_done", "episodes_failed",
                                           "footage_hours", "usd_per_hour")} | {"run": str(run)}))
    return rc


def resume(a, slice_dir: Path) -> None:
    folder = a.resume.resolve()
    if not (folder / 'run.json').is_file():
        raise SystemExit(f'--resume: no run.json in {folder}')
    with execution_lock(folder):
        # A killed wrapper can leave its harness alive and spending.
        with execution_lock(harness_out(folder, json.loads((folder / 'run.json').read_text()))):
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
    if info.get("pieces"):
        # the same cuts again (label/pieces.py reuses parts whose source is unchanged)
        info["pieces"] = plan_pieces(run, slice_dir, a.harness_args)
    replies = harness_out(run, info)
    if info["kind"] != "dry" and replies.exists():
        # calls killed with the last run hold their full bound until OpenRouter says what it billed for them; a run
        # that cut long recordings keeps its harness replies, and so its holds, in run/parts
        settled = resolve_unanswered_charges(replies, get_keys())
        if settled["resolved"]:
            print(f"resume: {settled['resolved']} unanswered calls settled at OpenRouter's billed cost, "
                  f"${settled['released_usd']:.2f} of holds released", flush=True)
    spent = billed_cost(run, info)
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
    replies = harness_out(run, info)
    if not any(replies.glob('*episode_*.json')) and not any((replies / '.evidence').glob('*.inspection.json')):
        raise SystemExit(f"--resume: no outputs in {replies}; refusing (wrong folder?)")
    cmd = harness_cmd(slice_dir, run, info["kind"], left, a.concurrency, a.harness_args, split=bool(info.get("pieces")))
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
    long_eps = plan_pieces(run, a.episodes, a.harness_args)
    cmd = harness_cmd(a.episodes, run, a.kind, a.cap, a.concurrency, a.harness_args, split=bool(long_eps))
    info = {"run_id": run_id, "kind": a.kind, "dataset": a.dataset, "slice": str(a.episodes.resolve()),
            "code": sha, "code_dirty": dirty, "started_at": now.isoformat(timespec="seconds"), "cap_usd": a.cap,
            "status": "running", "command": cmd, "note": a.note, **({"pieces": long_eps} if long_eps else {})}
    with execution_lock(run):
        write_atomic(run / "run.json", info, indent=1)
        return finish(run, info, cmd, a.episodes, "w")


if __name__ == "__main__":
    from label.harness import cli
    raise SystemExit(cli(main))
