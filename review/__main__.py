"""Review your own robot data in one command: read it, check it, label it and build its board.

    python -m review --data PATH_OR_URL --rig teleop_arms|handheld_gripper|ego_head --out JOB \\
        [--dataset NAME] [--free] [--cap 20] [--concurrency 8] [--max-minutes M] [--grouping JSON]

PATH_OR_URL is a folder, a file or an archive, or an http(s) URL of a file or an archive, which is downloaded into
JOB/upload first. Everything else is written under JOB, and the data itself is never edited. These are the stages
Data Review runs on an upload, in its order and with its settings, so a folder reviewed here and the same folder
uploaded to Data Review get the same requests and the same board:

  1. convert  the data into episode sidecars (prepare/formats.py, the reader python -m prepare folder runs), with
              a report of what was read, used and left out (JOB/report.json)
  2. dictionary  interpret source fields when recorded descriptors leave their meanings uncertain
  3. checks   crossed camera streams, recorded jumps and flat gripper channels where the data has arm state
              (checks.stream_pairing), sped-up recordings measured against their neighbours in the same folder
              (checks.timebase measure_folder), the capture checks (checks.capture_qc), and the checks on the
              other signals and depth streams where the data has them (checks.sensors)
  3. clips    browser-playable copies of each camera for the board (python -m board clips)
  4. dry run  every request built exactly as it would be sent, free. A recording longer than label/pieces.py's
              PIECE_MAX_S is labelled in parts cut at still moments
  5. label    the model labels every episode or part, stopping at --cap dollars (skipped with --free)
  6. board    parts stitched back into one timeline per recording, and the board's files (board.build with the
              rig's rules, plus fixed_window when the reader found fixed-length files)

Serve the result with `python -m board serve --board JOB --clips JOB/clips`. The model key comes from
OPENROUTER_API_KEYS, as for python -m label.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable


def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def repo_env(**extra) -> dict:
    """The environment every stage runs in: this checkout on PYTHONPATH, and few malloc arenas, as python -m label
    sets them, so a long threaded run hands freed frame buffers back."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("MALLOC_ARENA_MAX", "2")
    env.update(extra)
    return env


def run_step(job: Path, step: str, cmd: list[str], env: dict, ok_codes=(0,)) -> int:
    log = job / "logs" / f"{step}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"== {step}", flush=True)
    with open(log, "a") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=job).returncode
    if rc not in ok_codes:
        tail = log.read_text().strip().splitlines()[-6:]
        raise SystemExit(f"{step} exited {rc} (log {log}): " + " | ".join(tail))
    return rc


def fetch(data: str, job: Path) -> Path:
    """The data as a local path: a URL is downloaded into JOB/upload under its own file name."""
    if not data.startswith(("http://", "https://")):
        return Path(data).resolve()
    dest = job / "upload" / (Path(urllib.parse.urlparse(data).path).name or "data")
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"== download {data}", flush=True)
        tmp = dest.with_name(dest.name + ".part")
        with urllib.request.urlopen(data) as r, open(tmp, "wb") as fh:
            shutil.copyfileobj(r, fh)
        tmp.replace(dest)
    return dest


def commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m review", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="a folder, file or archive, or an http(s) URL of one")
    ap.add_argument("--rig", required=True, choices=["teleop_arms", "handheld_gripper", "ego_head"])
    ap.add_argument("--out", type=Path, required=True, metavar="JOB", help="where everything is written")
    ap.add_argument("--dataset", default=None, help="name shown on the board and told to the model")
    ap.add_argument("--free", action="store_true", help="stop after the dry run (no model calls)")
    ap.add_argument("--cap", type=float, default=20.0, help="most the labelling may spend, in USD (default 20)")
    ap.add_argument("--concurrency", type=int, default=8, help="requests in flight (default 8)")
    ap.add_argument("--jobs", type=int, default=2, help="decode processes per step (default 2)")
    ap.add_argument("--max-minutes", type=float, default=0, help="read at most this much footage (default: all)")
    ap.add_argument("--grouping", type=json.loads, default={},
                    help='JSON {folder: "takes" | "cameras"} for folders whose files cannot tell takes from cameras')
    a = ap.parse_args()

    from board import build as build_board
    from board import clips as board_clips
    from board import rules as board_rules
    from checks import capture_qc, timebase
    from label import pieces, dictionary_stage
    from prepare import formats

    job = a.out.resolve()
    job.mkdir(parents=True, exist_ok=True)
    data = fetch(a.data, job)
    dataset = a.dataset or data.stem
    eps = job / "episodes"
    jobs = str(max(1, a.jobs))
    env = repo_env()

    print("== convert", flush=True)
    if eps.exists():
        shutil.rmtree(eps)
    rep = formats.convert(data, a.rig, eps, dataset, a.max_minutes * 60 if a.max_minutes else float("inf"),
                          grouping=a.grouping)
    (job / "report.json").write_text(json.dumps(rep, indent=1))
    if not rep["episodes"]:
        why = "; ".join(f"{f['name']}: {f['why']}" for f in rep["failed"][:3])
        raise SystemExit("no episode could be read" + (f" ({why})" if why else ""))

    print('== dictionary', flush=True)
    dictionary_stage.prepare(job, eps, [e['episode_id'] for e in rep['episodes']], free=a.free, cap=a.cap)

    if any(e["state_kind"] != "none" for e in rep["episodes"]):
        for flag in ([], ["--jumps"], ["--grippers"]):
            run_step(job, f"checks{''.join(flag).replace('--', '_')}",
                     [PY, "-m", "checks.stream_pairing", *flag, "--jobs", jobs, str(eps)], env)
        timebase.measure_folder(eps)
    # EXIT_FAILED is some episode's capture checks not running (its worker died twice): that episode's record has every
    # check errored with the error, which the board shows, and the other episodes go on
    if run_step(job, "checks_capture", [PY, "-m", "checks.capture_qc", "--jobs", jobs, str(eps)], env,
                ok_codes=(0, capture_qc.EXIT_FAILED)):
        print("capture checks could not run on some episodes; each is shown with its checks errored "
              f"(log {job / 'logs' / 'checks_capture.log'})", flush=True)
    if any((eps / e["episode_id"] / "signals.npz").exists() or (eps / e["episode_id"] / "depth.json").exists()
           for e in rep["episodes"]):
        run_step(job, "checks_sensors", [PY, "-m", "checks.sensors", "--jobs", jobs, str(eps)], env)
    # exit 1 is board clips saying no episode came out at all (clips/failed.json lists why), told below as the upload's
    # own reason; any other failure of the step is still an error. An episode with a camera that decodes is kept, and
    # what was wrong with its other cameras goes into the report's notes
    rc = run_step(job, "clips", [PY, "-m", "board", "clips", "--episodes", str(eps), "--out", str(job / "clips"),
                                 "--jobs", jobs, "--clip-threads", "1"], env, ok_codes=(0, 1))
    left_out = board_clips.set_aside_failed(eps, job / "clips")
    if left_out:
        board_clips.drop_from_report(rep, left_out)
        if not rep["episodes"]:
            raise SystemExit("no episode could be put on the board: " + "; ".join(f["why"] for f in left_out[:3]))
        print(f"left out, no camera file decodes: {', '.join(f['name'] for f in left_out)}", flush=True)
    board_clips.note_camera_problems(rep, eps)
    (job / "report.json").write_text(json.dumps(rep, indent=1))
    if rc != 0:
        raise SystemExit(f"clips exited {rc} with episodes still to put on the board (log {job / 'logs' / 'clips.log'})")

    long_eps = pieces.write_units(job, eps)
    env = repo_env(RDA_DECODE_CONCURRENCY=os.environ.get("RDA_DECODE_CONCURRENCY") or str(2 * int(jobs)))
    run_step(job, "dry_run", [PY, "-m", "label.harness", "--episodes-root", str(job / "units"), "--out-dir",
                              str(job / "dry"), "--concurrency", jobs, "--dry-run"], env)
    if a.free:
        print(f"free run: requests built in {job / 'dry'}, model not called", flush=True)
        return 0

    run_dir = job / "run"
    run_dir.mkdir(exist_ok=True)
    info = {"run_id": job.name, "kind": "review", "dataset": dataset, "slice": str(eps),
            "code": f"argus@{commit()}", "started_at": now(), "cap_usd": a.cap, "status": "running"}
    (run_dir / "run.json").write_text(json.dumps(info, indent=1))
    remaining = round(a.cap - dictionary_stage.dictionary_spend(job)['reserved_usd'], 6)
    if remaining > 0:
        run_step(job, "label", [PY, "-m", "label.harness", "--episodes-root", str(job / "units"), "--out-dir",
                                str(run_dir / "out"), "--concurrency", str(a.concurrency),
                                "--max-spend", str(remaining)], env, ok_codes=(0, 1))
    else:
        print('label budget exhausted by dictionary reservation', flush=True)
    spend = dictionary_stage.dictionary_spend(job)
    label_cost = dictionary_stage.label_spend(job)
    complete = spend['complete'] and dictionary_stage.label_spend_complete(job)
    total = label_cost + (spend['cost_usd'] or 0.0)
    info.update(status="done", finished_at=now(), dictionary=spend,
                cost_usd=round(total, 6) if complete else None,
                reserved_cost_usd=round(label_cost + spend['reserved_usd'], 6),
                cost_complete=complete)
    (run_dir / "run.json").write_text(json.dumps(info, indent=1))

    print("== board", flush=True)
    final = job / "run_final"
    if final.exists():
        shutil.rmtree(final)
    st = pieces.stitch_run(job, eps, long_eps, final / "out")
    (final / "run.json").write_text(json.dumps(info, indent=1))
    entry = board_rules.own_data_entry(dataset, str(final), str(eps), a.rig, rep.get("packaging"))
    (job / "manifest.json").write_text(json.dumps({"board": job.name, "datasets": [entry]}, indent=1))
    build_board.build(job)
    if st["incomplete"]:
        print(f"on the board with a part not labelled, flagged at its span: {', '.join(st['incomplete'])}", flush=True)
    if st["unlabelled"]:
        print(f"on the board with no labels, no part was labelled: {', '.join(st['unlabelled'])}", flush=True)
    print(f"done: python -m board serve --board {job} --clips {job / 'clips'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
