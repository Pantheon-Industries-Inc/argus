"""Re-read a run's stored replies that did not parse, with the harness's current parser (no model call).

    python -m label.reparse RUN [RUN ...]

For every out/episode_*.json whose reply did not parse, the stored raw reply goes through
label.harness.parse_response again. A reply that now parses replaces its labels (parse_ok true, the earlier
error kept under _reparsed); one that still does not is left as it is. The original file is copied to
out/reparsed_originals/ first, and run.json lists what was re-read, by which code, and why. Prints one JSON
line per run with the counts.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path

from label import harness
from label.atomic import write_atomic
from label.run import commit


def reparse(run: Path) -> dict:
    out, done, still = run / "out", [], []
    code = commit()[0]
    for p in sorted(out.glob("episode_*.json")):
        r = json.loads(p.read_text())
        raw = (r.get("labels") or {}).get("_raw")
        if r.get("parse_ok") or not raw:
            continue
        labels, ok = harness.parse_response(raw)
        if not ok:
            still.append({"episode": p.stem, "error": labels["_parse_error"]})
            continue
        keep = out / "reparsed_originals"
        keep.mkdir(exist_ok=True)
        shutil.copy2(p, keep / p.name)
        labels["_reparsed"] = {"was": r["labels"].get("_parse_error"), "code": code}
        r["labels"], r["parse_ok"] = labels, True
        write_atomic(p, r)
        done.append({"episode": p.stem, "was": labels["_reparsed"]["was"]})
    rj = run / "run.json"
    meta = json.loads(rj.read_text())
    meta.setdefault("reparsed", []).append({"at": dt.datetime.now().isoformat(timespec="seconds"), "code": code,
                                            "why": "stored replies re-read with the current parser, no model call",
                                            "parsed_now": done, "still_unparsed": still})
    write_atomic(rj, meta, indent=1)
    return {"run": run.name, "parsed_now": len(done), "still_unparsed": len(still)}


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m label.reparse", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, nargs="+", help="run folders (each with run.json and out/)")
    for run in ap.parse_args().runs:
        print(json.dumps(reparse(run)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
