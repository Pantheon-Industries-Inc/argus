"""What every adapter's command line shares, the episode list format, the `prepare` flags and the run loop.

An episode list is a text file with one episode per line, in the adapter's own form (an episode index, a repo
path, a clip name); blank lines and lines starting with # are skipped. `prepare` takes

    --episodes LIST   the episodes to prepare
    --out FOLDER      where the episode folders are written, one per episode
    --raw RAW         where downloaded files are kept (adapters that download; default data/raw/<dataset>)
    --jobs N          episodes prepared at once
    --force           write episodes again that are already prepared (downloads in RAW are reused)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable


def read_list(path: Path) -> list[str]:
    """The episode lines of a list file, stripped, in file order."""
    return [s for s in (line.strip() for line in Path(path).read_text().splitlines()) if s and not s.startswith("#")]


def parser(dataset: str, doc: str) -> tuple[argparse.ArgumentParser, argparse._SubParsersAction]:
    ap = argparse.ArgumentParser(prog=f"python -m prepare {dataset}", description=doc,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    return ap, ap.add_subparsers(dest="cmd", required=True)


def add_prepare(sub, dataset: str, episodes_help: str, *, jobs: int = 8, raw: bool = True,
                episodes_required: bool = True) -> argparse.ArgumentParser:
    """The `prepare` subcommand with the shared flags; an adapter adds its own after these."""
    p = sub.add_parser("prepare", help="prepare exactly the listed episodes")
    p.add_argument("--episodes", type=Path, required=episodes_required, metavar="LIST", help=episodes_help)
    p.add_argument("--out", type=Path, required=True, metavar="FOLDER", help="where the episode folders are written")
    if raw:
        p.add_argument("--raw", type=Path, default=Path("data/raw") / dataset, metavar="RAW",
                       help=f"where downloaded files are kept (default data/raw/{dataset})")
    p.add_argument("--jobs", type=int, default=jobs, metavar="N", help=f"episodes prepared at once (default {jobs})")
    p.add_argument("--force", action="store_true", help="write episodes again that are already prepared")
    return p


def run(items: list, prepare_one: Callable[[object], str], jobs: int) -> int:
    """prepare_one(item) for every item, `jobs` at a time. It returns "ok" or "skip" (already prepared); an
    exception is a failed episode, printed with its item and counted. Exit code 1 when any failed."""
    counts = {"ok": 0, "skip": 0, "failed": 0}
    with ThreadPoolExecutor(max(1, jobs)) as ex:
        futs = {ex.submit(prepare_one, it): it for it in items}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                counts[f.result()] += 1
            except Exception as e:  # an episode that cannot be read is a finding about the data: listed, not hidden
                counts["failed"] += 1
                print(f"FAILED {futs[f]}: {type(e).__name__}: {e}"[:400], file=sys.stderr, flush=True)
            if i % 25 == 0 or i == len(futs):
                print(f"{i}/{len(futs)} {json.dumps(counts)}", flush=True)
    return 1 if counts["failed"] else 0


def commit() -> str:
    """The commit this code is: a vendored copy's COMMIT file, else the checkout's HEAD (with -dirty when a tracked
    file differs from it), else "unknown"."""
    root = Path(__file__).resolve().parent.parent
    try:
        return json.loads((root / "COMMIT").read_text())["short"]
    except (OSError, ValueError, KeyError):
        pass
    try:
        git = lambda *a: subprocess.run(["git", *a], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        return git("rev-parse", "--short=7", "HEAD") + ("-dirty" if git("status", "--porcelain", "--untracked-files=no")
                                                         else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def stamp(out: Path, adapter: str, since: float) -> int:
    """Record in every episode context under `out` written at or after `since` (a time.time()) which adapter wrote it
    and at which commit: source.adapter (kept when the adapter already named a more specific one) and
    source.adapter_commit. Returns how many were stamped."""
    sha, n = commit(), 0
    for c in sorted(Path(out).glob("episode_*/context.json")):
        if c.stat().st_mtime < since:
            continue                     # prepared by an earlier run and skipped by this one
        ctx = json.loads(c.read_text())
        src = ctx.setdefault("source", {})
        src.setdefault("adapter", adapter)
        src["adapter_commit"] = sha
        c.write_text(json.dumps(ctx, indent=1, default=str))
        n += 1
    return n
