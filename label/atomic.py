"""Writing a JSON file whole or not at all, with nothing else loaded: the readers, the checks, the board and the
labelling harness all write through it."""
from __future__ import annotations

import json
import os
from pathlib import Path


def write_atomic(out_path: Path, result, indent: int = 2, default=None) -> None:
    """Write JSON through a temporary file in the same folder, then replace the file with it, so a kill mid-write never
    leaves a truncated file: a resume would take it for a result, and a context.json cut short would stop everything
    that reads the episode."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f".{out_path.name}.tmp")
    tmp.write_text(json.dumps(result, indent=indent, default=default))
    os.replace(tmp, out_path)
