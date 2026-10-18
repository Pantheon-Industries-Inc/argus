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
import copy
import datetime as dt
import json
import shutil
from pathlib import Path

from label import harness
from label.atomic import write_atomic
from label.run import commit


def rebind(record: dict) -> None:
    """Rebuild derived projections from saved request evidence without a provider call."""
    from label import sensor_evidence as se, evidence_access as ea
    labels = record['labels']
    problem = harness.saved_input_problem(record, Path(record['episode_dir'])) if record.get('episode_dir') else None
    provenance = {'model': record.get('model_served') or record.get('model'),
                  'generation_id': record.get('generation_id'), 'method': 'stored episode reply reparsed'}
    if 'camera_observations' in labels:
        fields = {**record, 'input_identity': {}} if problem else record
        record['camera_observations'] = harness.bind_camera_observations(labels['camera_observations'], fields)
        record['camera_observations']['provenance'] = provenance
    evidence = record.get('sensor_evidence')
    if isinstance(evidence, dict):
        # Inspected fields have their separate request receipts and binder below.
        native = copy.deepcopy(evidence)
        native['sensors'] = [s for s in native.get('sensors', []) if 'access_field_id' not in s]
        sensor_ids = {s['id'] for s in native['sensors']}
        native['series'] = [r for r in native.get('series', []) if r['sensor_id'] in sensor_ids]
        record['sensor_evidence'] = se.bind(native, labels.get('sensor_findings'))
        record['sensor_evidence']['provenance'] = provenance
        if problem:
            bound = record['sensor_evidence']
            bound.setdefault('withheld_findings', []).extend(bound['findings'])
            bound['findings'] = []
            bound.setdefault('limitations', []).append(problem)
    inspection = record.get('evidence_inspection')
    if not isinstance(inspection, dict):
        return
    try:
        ep = harness.me.load(Path(record['episode_dir']))
        if not ea.same_source_proof(inspection.get('source_proof'), ep['context'], ep['dir']):
            raise ValueError('saved inspected source proof changed or is unavailable')
        access = ea.Access(ep)
        receipts = copy.deepcopy(inspection.get('inspections') or [])
        if any(r.get('field_id') not in access.fields for r in receipts):
            raise ValueError('saved inspection field is no longer available')
        access.receipts = receipts
        additional = access.bind(labels.get('evidence_findings'))
        inspection['untimed_findings'] = additional['untimed_findings']
        ea.withhold_stale_untimed(inspection, ep['context'], ep['dir'])
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        inspection['untimed_findings'] = []
        inspection.setdefault('limitations', []).append(f'Reparse evidence binding withheld: {error}')
        additional = {'sensors': [], 'series': [], 'findings': [], 'coverage': [], 'limitations': [],
                      'unbound_findings': [{'finding': copy.deepcopy(f), 'reason': str(error)}
                                           for f in labels.get('evidence_findings') or []]}
    evidence = record.setdefault('sensor_evidence', {'version': 1})
    for key in ('sensors', 'series', 'findings', 'unbound_findings', 'coverage', 'limitations'):
        evidence.setdefault(key, []).extend(additional[key])
    evidence['provenance'] = provenance


def reparse(run: Path) -> dict:
    run = Path(run)
    if not (run / 'run.json').is_file() or not (run / 'out').is_dir():
        raise SystemExit(f'{run}: reparse needs an existing run.json and out folder')
    with harness.execution_lock(run), harness.execution_lock(run / 'out'):
        return _reparse(run)


def _reparse(run: Path) -> dict:
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
        rebind(r)
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
