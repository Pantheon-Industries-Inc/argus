"""The checkout's commit, which every run and reparse records beside its labels. It lives apart from label/run.py so
label/reparse.py, which the request identity reaches through label/harness.py, never imports the run wrapper and the
long-recording cutter it calls."""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


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
