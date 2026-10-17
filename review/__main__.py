"""Review your own robot data in one command: read it, check it, label it and build its board.

    python -m review --data PATH_OR_URL --rig teleop_arms|handheld_gripper|ego_head --out JOB \\
        [--dataset NAME] [--free] [--cap 20] [--concurrency 8] [--max-minutes M] [--grouping JSON]

PATH_OR_URL is a folder, a file or an archive, or an http(s) URL of a file or an archive, which is downloaded into
JOB/upload first. Everything else is written under JOB, and the data itself is never edited. These are the stages
Data Review shares this path. Upload limits and explicit local overrides can differ; the resolved policy is saved
in the job and each prepared episode:

  1. convert  the data into episode sidecars (prepare/formats.py, the reader python -m prepare folder runs), with
              a report of what was read, used and left out (JOB/report.json)
  2. dictionary  interpret source fields when recorded descriptors leave their meanings uncertain
  3. checks   crossed camera streams, recorded jumps and flat gripper channels where the data has arm state
              (checks.stream_pairing), sped-up recordings measured against their neighbours in the same folder
              (checks.timebase measure_folder), the capture checks (checks.capture_qc), and the checks on the
              other signals and depth streams where the data has them (checks.sensors)
  4. clips    browser-playable copies of each camera for the board (python -m board clips)
  5. dry run  every request built exactly as it would be sent, free. A recording longer than label/pieces.py's
              PIECE_MAX_S is labelled in parts cut at still moments
  6. label    the model labels every episode or part, stopping at --cap dollars (skipped with --free)
  7. board    parts stitched back into one timeline per recording, and the board's files (board.build with the
              rig's rules, plus fixed_window when the reader found fixed-length files)

Serve the result with `python -m board serve --board JOB --clips JOB/clips`. The model key comes from
OPENROUTER_API_KEYS, as for python -m label. Downloads allow 24 hours by default;
REVIEW_DOWNLOAD_DEADLINE sets a different finite duration in seconds. A stalled read still times out after 30 seconds.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

from label import policy
from label.atomic import write_atomic

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable
DOWNLOAD_MAX_BYTES = int(os.environ.get('REVIEW_DOWNLOAD_MAX_BYTES', 20 * 1024**3))
DOWNLOAD_TIMEOUT = 30
DOWNLOAD_DEADLINE = 24 * 3600


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


def fetch(data: str, job: Path, *, allow_download=True) -> Path:
    """Keep different URL resources distinct without exposing access parameters."""
    if not data.startswith(('http://', 'https://')):
        return Path(data).resolve()
    deadline = float(os.environ.get('REVIEW_DOWNLOAD_DEADLINE', DOWNLOAD_DEADLINE))
    if not math.isfinite(deadline) or deadline <= 0:
        raise ValueError('REVIEW_DOWNLOAD_DEADLINE must be positive finite seconds')
    resource = hashlib.sha256(data.encode()).hexdigest()
    basename = Path(urllib.parse.urlsplit(data).path).name or 'data'
    dest = job / 'upload' / resource / basename
    if not dest.exists():
        if not allow_download:
            raise SystemExit('Retained download is missing; use a fresh job with a new --out. Existing receipts are retained.')
        dest.parent.mkdir(parents=True, exist_ok=True)
        print('== download', flush=True)
        temporary = None
        try:
            started = time.monotonic()
            with urllib.request.urlopen(data, timeout=min(DOWNLOAD_TIMEOUT, deadline)) as response, tempfile.NamedTemporaryFile(
                    dir=dest.parent, prefix='.download-', delete=False) as output:
                temporary = Path(output.name)
                length = response.headers.get('Content-Length')
                if length and int(length) > DOWNLOAD_MAX_BYTES:
                    raise ValueError('Download exceeds the configured size limit')
                total = 0
                # HTTPResponse.read waits for the entire requested chunk even while bytes trickle in.
                read = getattr(response, 'read1', response.read)
                raw = getattr(getattr(response, 'fp', None), 'raw', None)
                connection = getattr(raw, '_sock', None)
                while True:
                    remaining = deadline - (time.monotonic() - started)
                    if remaining <= 0:
                        raise ValueError('Download exceeds the configured time limit')
                    if connection is not None and connection.fileno() >= 0:
                        connection.settimeout(min(DOWNLOAD_TIMEOUT, remaining))
                    try:
                        chunk = read(64 * 1024)
                    except TimeoutError as error:
                        if time.monotonic() - started >= deadline:
                            raise ValueError('Download exceeds the configured time limit') from error
                        raise
                    if time.monotonic() - started >= deadline:
                        raise ValueError('Download exceeds the configured time limit')
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > DOWNLOAD_MAX_BYTES:
                        raise ValueError('Download exceeds the configured size limit')
                    output.write(chunk)
                if length and total != int(length):
                    raise ValueError('Download ended before its declared length')
            temporary.replace(dest)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return dest


def commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def preparation_code_identity() -> dict:
    """The reader and the runtime helpers it uses, independent of a Git checkout's dirty state."""
    return {p.relative_to(REPO).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for package in ('prepare', 'label', 'checks', 'board', 'review')
            for p in sorted((REPO / package).rglob('*.py')) if '__pycache__' not in p.parts}


def file_identity(path: Path, *, context=False) -> dict:
    from label.evidence_access import SOURCE_HASH_LIMIT
    stat = path.stat()
    result = {'location': hashlib.sha256(str(path.resolve()).encode()).hexdigest(), 'size': stat.st_size}
    if context:
        from label.dictionary import DERIVED_CONTEXT
        document = json.loads(path.read_text())
        # These stages add results to the context without preparing its source again.
        document = {k: v for k, v in document.items() if k not in DERIVED_CONTEXT | {'pieces'}}
        content = json.dumps(document, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()
        result['size'] = len(content)
        result['sha256'] = hashlib.sha256(content).hexdigest()
    elif stat.st_size <= SOURCE_HASH_LIMIT:
        result['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        result['mtime_ns'] = stat.st_mtime_ns
    return result


def source_identity(data: Path, job: Path) -> dict:
    from prepare.formats import files_under, single_file_members
    if data.is_file():
        members = single_file_members(data)
        files = list(dict.fromkeys(members['files'] + members['scope']))
    else:
        files = files_under(data)
    # In particular, a CLI job nested below its input is not another uploaded recording.
    if data.is_dir() and job.is_relative_to(data):
        files = [p for p in files if not p.resolve().is_relative_to(job)]
    base = data.parent if data.is_file() else data
    return {p.relative_to(base).as_posix(): file_identity(p) for p in files}


def prepared_identity(job: Path) -> dict:
    from prepare.formats import files_under
    return {p.relative_to(job).as_posix(): file_identity(p, context=p.name == 'context.json')
            for name in ('episodes', 'episodes_unclipped') for p in files_under(job / name)}


def preparation_checkpoint(job: Path, receipt: dict) -> None:
    receipt.update(prepared=prepared_identity(job), report=file_identity(job / 'report.json'))
    write_atomic(job / 'preparation_receipt.json', receipt)


def verified_preparation(job: Path, data: Path, args, dataset: str, overrides: dict) -> tuple[dict, dict]:
    """Reuse a proven preparation or require a new job, retaining every old artifact."""
    from prepare import formats
    receipt_path = job / 'preparation_receipt.json'
    resource = args.data if args.data.startswith(('http://', 'https://')) else str(data)
    settings = {'rig': args.rig, 'dataset': dataset, 'max_minutes': args.max_minutes, 'grouping': args.grouping}
    current = {'version': 1, 'resource': hashlib.sha256(resource.encode()).hexdigest(),
               'source': source_identity(data, job), 'settings': settings, 'producer': preparation_code_identity()}
    refuse = 'Preparation cannot be verified for this resume; use a fresh job with a new --out. Existing data and receipts are retained.'
    if receipt_path.exists():
        try:
            saved = json.loads(receipt_path.read_text())
            if any(saved.get(key) != value for key, value in current.items()):
                raise ValueError('source, settings or producer changed')
            if saved['prepared'] != prepared_identity(job) or saved['report'] != file_identity(job / 'report.json'):
                raise ValueError('prepared artifacts or report changed')
            selected = {}
            for name in saved['policies']:
                ep = job / 'episodes' / name
                if not ep.is_dir():
                    ep = job / 'episodes_unclipped' / name
                selected[name] = policy.resolve(json.loads((ep / 'context.json').read_text()), **overrides)
            if selected != saved['policies']:
                raise ValueError('annotation settings changed')
            rep = json.loads((job / 'report.json').read_text())
            if not isinstance(rep.get('episodes'), list):
                raise ValueError('preparation report is invalid')
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            raise SystemExit(refuse) from error
        print('== convert (verified preparation reused)', flush=True)
        return rep, saved
    if any((job / name).exists() for name in ('episodes', 'episodes_unclipped', 'report.json', 'run',
                                             'dictionary.json', 'dictionary.claim')):
        raise SystemExit(refuse)
    print('== convert', flush=True)
    rep = formats.convert(data, args.rig, job / 'episodes', dataset,
                          args.max_minutes * 60 if args.max_minutes else float('inf'), grouping=args.grouping)
    if current['source'] != source_identity(data, job) or current['producer'] != preparation_code_identity():
        raise SystemExit(refuse)
    write_atomic(job / 'report.json', rep, indent=1)
    current['policies'] = policy.persist(job / 'episodes', **overrides)
    preparation_checkpoint(job, current)
    return rep, current


def build_review_board(job: Path, eps: Path, long_eps, info: dict, dataset: str, rig: str, packaging) -> dict:
    """Build the stitched board while the encompassing review still has running status."""
    from board import build as build_board, rules as board_rules
    from label import pieces
    print("== board", flush=True)
    final = job / 'run_final'
    result = pieces.stitch_run(job, eps, long_eps, final / 'out')
    write_atomic(final / 'run.json', info, indent=1)
    entry = board_rules.own_data_entry(dataset, str(final), str(eps), rig, packaging)
    write_atomic(job / 'manifest.json', {'board': job.name, 'datasets': [entry]}, indent=1)
    build_board.build(job)
    return result


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
    ap.add_argument("--timeout", type=int, default=None,
                    help="seconds per labeling request (default: the harness's 600 seconds)")
    ap.add_argument("--max-minutes", type=float, default=0, help="read at most this much footage (default: all)")
    ap.add_argument("--grouping", type=json.loads, default={},
                    help='JSON {folder: "takes" | "cameras"} for folders whose files cannot tell takes from cameras')
    policy.add_arguments(ap)
    a = ap.parse_args()
    if not math.isfinite(a.cap) or a.cap <= 0:
        ap.error("--cap must be finite and positive")
    if not math.isfinite(a.max_minutes) or a.max_minutes < 0:
        ap.error("--max-minutes must be finite and nonnegative")
    if a.timeout is not None and a.timeout <= 0:
        ap.error("--timeout must be positive")
    overrides = {key: getattr(a, key) for key in ("model", "reasoning", "sample_fps", "task_mode", "dataset_note")}
    policy.resolve({"profile": a.rig}, **overrides)

    from board import clips as board_clips
    from checks import EXIT_PARTIAL, timebase
    from label import pieces, dictionary_stage
    from prepare import formats

    job = a.out.resolve()
    job.mkdir(parents=True, exist_ok=True)
    receipt_path = job / 'preparation_receipt.json'
    refuse = 'Preparation cannot be verified for this resume; use a fresh job with a new --out. Existing data and receipts are retained.'
    if receipt_path.exists():
        resource = a.data if a.data.startswith(('http://', 'https://')) else str(Path(a.data).resolve())
        try:
            if json.loads(receipt_path.read_text()).get('resource') != hashlib.sha256(resource.encode()).hexdigest():
                raise ValueError('input resource changed')
        except (OSError, ValueError, AttributeError) as error:
            raise SystemExit(refuse) from error
    elif any((job / name).exists() for name in ('episodes', 'episodes_unclipped', 'report.json', 'run',
                                               'dictionary.json', 'dictionary.claim')):
        raise SystemExit(refuse)
    data = fetch(a.data, job, allow_download=not receipt_path.exists())
    if data == job:
        raise SystemExit('--out must be a separate job directory; it may be nested below --data.')
    dataset = a.dataset or data.stem
    eps = job / "episodes"
    jobs = str(max(1, a.jobs))
    env = repo_env()

    rep, preparation = verified_preparation(job, data, a, dataset, overrides)
    if not rep["episodes"]:
        why = "; ".join(f"{f['name']}: {f['why']}" for f in rep["failed"][:3])
        raise SystemExit("no episode could be read" + (f" ({why})" if why else ""))

    selected = preparation['policies']
    write_atomic(job / "annotation_policy.json", selected)

    print('== dictionary', flush=True)
    dictionary_stage.prepare(job, eps, [e['episode_id'] for e in rep['episodes']], free=a.free, cap=a.cap)

    if any(e["state_kind"] != "none" for e in rep["episodes"]):
        for flag in ([], ["--jumps"], ["--grippers"]):
            step = f"checks{''.join(flag).replace('--', '_')}"
            if run_step(job, step, [PY, "-m", "checks.stream_pairing", *flag, "--jobs", jobs, str(eps)], env,
                        ok_codes=(0, EXIT_PARTIAL)):
                print(f"some recorded-state checks could not finish; errors are retained (log {job / 'logs' / (step + '.log')})",
                      flush=True)
        timebase.measure_folder(eps)
    # A recorded partial failure leaves the usable checks available and names every check that could not finish.
    if run_step(job, "checks_capture", [PY, "-m", "checks.capture_qc", "--jobs", jobs, str(eps)], env,
                ok_codes=(0, EXIT_PARTIAL)):
        print("capture checks could not run on some episodes; each is shown with its checks errored "
              f"(log {job / 'logs' / 'checks_capture.log'})", flush=True)
    if any((eps / e["episode_id"] / "signals.npz").exists() or (eps / e["episode_id"] / "depth.json").exists()
           for e in rep["episodes"]):
        if run_step(job, "checks_sensors", [PY, "-m", "checks.sensors", "--jobs", jobs, str(eps)], env,
                    ok_codes=(0, EXIT_PARTIAL)):
            print(f"sensor checks could not finish on some episodes; usable results and errors are retained "
                  f"(log {job / 'logs/checks_sensors.log'})", flush=True)
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
    write_atomic(job / "report.json", rep, indent=1)
    preparation_checkpoint(job, preparation)
    if rc != 0:
        raise SystemExit(f"clips exited {rc} with episodes still to put on the board (log {job / 'logs' / 'clips.log'})")

    long_eps = pieces.write_units(job, eps)
    timeout_args = ["--timeout", str(a.timeout)] if a.timeout is not None else []
    env = repo_env(RDA_DECODE_CONCURRENCY=os.environ.get("RDA_DECODE_CONCURRENCY") or str(2 * int(jobs)))
    run_step(job, "dry_run", [PY, "-m", "label.harness", "--episodes-root", str(job / "units"), "--out-dir",
                              str(job / "dry"), "--concurrency", jobs, "--dry-run", *timeout_args], env)
    if a.free:
        print(f"free run: requests built in {job / 'dry'}, model not called", flush=True)
        return 0

    if preparation['source'] != source_identity(data, job):
        raise SystemExit('Source changed during review; use a fresh job with a new --out. Existing receipts are retained.')

    run_dir = job / "run"
    run_dir.mkdir(exist_ok=True)
    prior_run = json.loads((run_dir / 'run.json').read_text()) if (run_dir / 'run.json').exists() else {}
    info = {"run_id": job.name, "kind": "review", "dataset": dataset, "slice": str(eps),
            "code": f"argus@{commit()}", "started_at": prior_run.get('started_at') or now(), "cap_usd": a.cap, "status": "running",
            "annotation_policy": selected}
    write_atomic(run_dir / "run.json", info, indent=1)
    label_rc = 0
    try:
        remaining = dictionary_stage.remaining_label_budget(job, a.cap)
        if remaining > 0:
            label_rc = run_step(job, "label", [PY, "-m", "label.harness", "--episodes-root", str(job / "units"), "--out-dir",
                                    str(run_dir / "out"), "--concurrency", str(a.concurrency),
                                    "--max-spend", str(remaining), *timeout_args], env, ok_codes=(0, 1))
        else:
            print('label budget exhausted by prior spending or outstanding reservations', flush=True)
            from label.harness import record_budget_refusal
            for dry in sorted((job / 'dry').glob('episode_*.json')):
                record_budget_refusal(run_dir / 'out' / dry.name, json.loads(dry.read_text()))
        spend = dictionary_stage.dictionary_spend(job)
        label_cost = dictionary_stage.label_spend(job)
        complete = spend['complete'] and dictionary_stage.label_spend_complete(job)
        total = label_cost + (spend['cost_usd'] or 0.0)
        info.update(dictionary=spend,
                    cost_usd=round(total, 6) if complete else None,
                    reserved_cost_usd=round(label_cost + spend['reserved_usd'], 6),
                    cost_complete=complete)
        write_atomic(run_dir / "run.json", info, indent=1)
        st = build_review_board(job, eps, long_eps, info, dataset, a.rig, rep.get('packaging'))
        info.update(status='failed' if label_rc else 'done', finished_at=now())
        write_atomic(run_dir / 'run.json', info, indent=1)
        write_atomic(job / 'run_final/run.json', info, indent=1)
    except BaseException as error:
        info.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', finished_at=now(),
                    error=f'{type(error).__name__}: {error}'[:500])
        write_atomic(run_dir / 'run.json', info, indent=1)
        if (job / 'run_final/run.json').exists():
            write_atomic(job / 'run_final/run.json', info, indent=1)
        raise
    if st["incomplete"]:
        print(f"on the board with a part not labelled, flagged at its span: {', '.join(st['incomplete'])}", flush=True)
    if st["unlabelled"]:
        print(f"on the board with no labels, no part was labelled: {', '.join(st['unlabelled'])}", flush=True)
    outcome = 'review finished with label failures' if label_rc else 'done'
    print(f"{outcome}: python -m board serve --board {job} --clips {job / 'clips'}", flush=True)
    return 1 if label_rc else 0


if __name__ == "__main__":
    raise SystemExit(main())
