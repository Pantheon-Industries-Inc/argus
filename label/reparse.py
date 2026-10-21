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
    from label import sensor_evidence as se, evidence_access as ea, sensor_copy, sensor_analysis
    labels = record['labels']
    copy_version = (record.get('config') or {}).get('sensor_copy_contract_version')
    def finding_key(finding):
        fields = se.finding_fields(finding)
        fields.pop('review_status', None)
        if sensor_copy.shared(finding.get('copy_contract_version')):
            fields.pop('overlay_detail', None)
        return json.dumps(fields, sort_keys=True, separators=(',', ':'), allow_nan=False)
    def unbound_findings(doc):
        return [row['finding'] for row in doc.get('unbound_findings') or []
                if isinstance(row, dict) and isinstance(row.get('finding'), dict)]
    reviewed, overlay_overrides = {}, {}
    for doc in (record.get('sensor_evidence') or {}, record.get('evidence_inspection') or {}):
        bound = [*doc.get('findings', []), *doc.get('untimed_findings', []), *doc.get('withheld_findings', [])]
        for finding in [*bound, *unbound_findings(doc)]:
            if isinstance(finding, dict) and finding.get('review_status') is not None:
                reviewed[finding_key(finding)] = finding['review_status']
        for finding in bound:
            if (isinstance(finding, dict) and sensor_copy.shared(finding.get('copy_contract_version'))
                    and isinstance(finding.get('overlay_detail'), str)):
                overlay_overrides[finding_key(finding)] = finding['overlay_detail']
    def retain_reviews(findings):
        for finding in findings:
            status = reviewed.get(finding_key(finding))
            if status is not None:
                finding['review_status'] = status
            if sensor_copy.shared(finding.get('copy_contract_version')):
                key = finding_key(finding)
                if key in overlay_overrides:
                    finding['overlay_detail'] = overlay_overrides[key]
    problem = harness.saved_input_problem(record, Path(record['episode_dir'])) if record.get('episode_dir') else None
    current_context = None
    def restore_analysis():
        receipt = record.get('sensor_analysis') or (record.get('evidence_inspection') or {}).get('sensor_analysis')
        if not isinstance(receipt, dict):
            return
        try:
            request = receipt['request']
            stage = sensor_analysis.bind(request, receipt)
            if problem or current_context is None or not se.compatible(stage, current_context, record.get('episode_dir')):
                stage['withheld_findings'] = stage.pop('findings')
                stage['findings'] = []
                stage.setdefault('limitations', []).append(problem or 'Regional analysis source or interpretation changed.')
            sensor_analysis.attach(record, stage)
            retain_reviews(record['sensor_evidence']['findings'])
            retain_reviews(record['sensor_evidence'].get('withheld_findings') or [])
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            record.setdefault('sensor_evidence', {'version': 1}).setdefault('limitations', []).append(
                f'Regional analysis binding withheld: {error}')
    provenance = {'model': record.get('model_served') or record.get('model'),
                  'generation_id': record.get('generation_id'), 'method': 'stored episode reply reparsed'}
    if 'camera_observations' in labels:
        fields = {**record, 'input_identity': {}} if problem else record
        record['camera_observations'] = harness.bind_camera_observations(labels['camera_observations'], fields)
        record['camera_observations']['provenance'] = provenance
    evidence = record.get('sensor_evidence')
    inspection = record.get('evidence_inspection')
    native = None
    if isinstance(evidence, dict):
        # Inspected fields have their separate request receipts and binder below.
        native = sensor_analysis.without_attached(evidence)
        native['sensors'] = [s for s in native.get('sensors', []) if 'access_field_id' not in s]
    routes = se.route_findings(labels, sensor_evidence=bool(native and native['sensors']),
                               inspection=isinstance(inspection, dict))
    if native is not None:
        sensor_ids = {s['id'] for s in native['sensors']}
        native['series'] = [r for r in native.get('series', []) if r['sensor_id'] in sensor_ids]
        source_context = None
        if not problem and record.get('episode_dir'):
            try:
                source_context = json.loads((Path(record['episode_dir']) / 'context.json').read_text())
                current_context = source_context
            except (OSError, ValueError):
                pass
        record['sensor_evidence'] = se.bind(native, routes['sensor_findings'],
                                            source_context=source_context, copy_contract_version=copy_version)
        retain_reviews(record['sensor_evidence']['findings'])
        retain_reviews(unbound_findings(record['sensor_evidence']))
        record['sensor_evidence']['provenance'] = provenance
        if problem:
            bound = record['sensor_evidence']
            bound.setdefault('withheld_findings', []).extend(bound['findings'])
            bound['findings'] = []
            bound.setdefault('limitations', []).append(problem)
    if routes['unbound'] or routes['routed'] or isinstance(record.get('sensor_evidence'), dict):
        if not isinstance(record.get('sensor_evidence'), dict):
            record['sensor_evidence'] = {**se.empty_doc(), 'provenance': provenance}
        se.keep_routing(record['sensor_evidence'], routes)
        retain_reviews(row['finding'] for row in routes['unbound'] if isinstance(row['finding'], dict))
    if not isinstance(inspection, dict):
        restore_analysis()
        return
    try:
        ep = harness.me.load(Path(record['episode_dir']))
        current_context = ep['context']
        if not ea.same_source_proof(inspection.get('source_proof'), ep['context'], ep['dir']):
            raise ValueError('saved inspected source proof changed or is unavailable')
        access = ea.Access(ep, copy_contract_version=copy_version)
        receipts = copy.deepcopy(inspection.get('inspections') or [])
        if any(r.get('field_id') not in access.fields for r in receipts):
            raise ValueError('saved inspection field is no longer available')
        access.receipts = receipts
        additional = access.bind(routes['evidence_findings'])
        retain_reviews(additional['findings'])
        retain_reviews(additional['untimed_findings'])
        inspection['untimed_findings'] = additional['untimed_findings']
        ea.withhold_stale_untimed(inspection, ep['context'], ep['dir'])
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        inspection['untimed_findings'] = []
        inspection.setdefault('limitations', []).append(f'Reparse evidence binding withheld: {error}')
        additional = {'sensors': [], 'series': [], 'findings': [], 'coverage': [], 'limitations': [],
                      'unbound_findings': [{'finding': copy.deepcopy(f), 'reason': str(error)}
                                           for f in routes['evidence_findings']]}
    retain_reviews(unbound_findings(additional))
    evidence = record.setdefault('sensor_evidence', {'version': 1})
    for key in ('sensors', 'series', 'findings', 'unbound_findings', 'coverage', 'limitations'):
        evidence.setdefault(key, []).extend(additional[key])
    evidence['provenance'] = provenance
    restore_analysis()
    receipt = inspection.get('sensor_cards')
    if isinstance(receipt, dict) and receipt.get('state') == 'received':
        from label import sensor_cards
        try:
            sensor_cards.apply(evidence, receipt['request'], receipt)
        except (ValueError, KeyError, TypeError) as error:
            evidence.setdefault('limitations', []).append(f'Video cards withheld: {error}')


def refresh_received(record: dict) -> dict:
    """Regrade a compatible raw reply, retaining human decisions and financial fields."""
    raw = record.get('response_text')
    if not isinstance(raw, str):
        raise ValueError('saved reply has no complete retained response text')
    updated = copy.deepcopy(record)
    labels, ok = harness.parse_response(raw)
    if '_reparsed' in (record.get('labels') or {}):
        labels['_reparsed'] = copy.deepcopy(record['labels']['_reparsed'])
    updated.update(labels=labels, parse_ok=ok)
    if ok:
        if 'camera_observations' not in labels:
            updated.pop('camera_observations', None)
        rebind(updated)
        for name in ('sensor_evidence', 'camera_observations'):
            if isinstance(updated.get(name), dict) and isinstance(record.get(name), dict):
                if 'provenance' in record[name]:
                    updated[name]['provenance'] = copy.deepcopy(record[name]['provenance'])
                else:
                    updated[name].pop('provenance', None)
    else:
        updated.pop('camera_observations', None)
        for name, fields in (('sensor_evidence', ('findings',)), ('evidence_inspection', ('untimed_findings',))):
            if isinstance(updated.get(name), dict):
                for field in fields:
                    updated[name].setdefault('withheld_findings', []).extend(updated[name].get(field) or [])
                    updated[name][field] = []
    return updated


def save_annotation(out: Path, name: str, path: Path, record: dict) -> None:
    """Repair selected reply authority without replacing a newer billing ledger."""
    if record.get('_authority_history_index') is not None:
        ledger = harness.billing_record(out, name)
        invoice = ledger[1]['final_cost_history'][record['_authority_history_index']]
        invoice.update({key: value for key, value in record.items()
                        if key not in ('_billing_receipt', '_authority_history_index')})
        write_atomic(ledger[0], ledger[1])
    else:
        write_atomic(path, {key: value for key, value in record.items() if key != '_billing_receipt'})


def reparse(run: Path) -> dict:
    run = Path(run)
    if not (run / 'run.json').is_file() or not (run / 'out').is_dir():
        raise SystemExit(f'{run}: reparse needs an existing run.json and out folder')
    with harness.execution_lock(run), harness.execution_lock(run / 'out'):
        return _reparse(run)


def _reparse(run: Path) -> dict:
    out, done, still = run / "out", [], []
    code = commit()[0]
    names = {p.name.removeprefix('failed_').removeprefix('noreply_') for p in out.glob('*.json')
             if p.name.startswith(('episode_', 'failed_episode_', 'noreply_episode_'))}
    for name in sorted(names):
        selected = harness.annotation_record(out, name)
        if selected is None:
            continue
        p, r = selected
        if r.get('final_dispatch_outcome') in ('claimed', 'unverified', 'not dispatched') and not (
                r.get('response') is not None or r.get('response_text') is not None):
            continue
        raw = (r.get("labels") or {}).get("_raw") or r.get('response_text')
        if r.get("parse_ok") or not raw:
            continue
        labels, ok = harness.parse_response(raw)
        if not ok:
            still.append({"episode": p.stem, "error": labels["_parse_error"]})
            continue
        keep = out / "reparsed_originals"
        keep.mkdir(exist_ok=True)
        ledger = harness.billing_record(out, name)
        original = ledger[0] if r.get('_authority_history_index') is not None else p
        shutil.copy2(original, keep / original.name)
        previous_labels = r.get("labels") if isinstance(r.get("labels"), dict) else {}
        labels["_reparsed"] = {"was": previous_labels.get("_parse_error"), "code": code}
        r["labels"], r["parse_ok"] = labels, True
        rebind(r)
        save_annotation(out, name, p, r)
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
    from label.harness import cli
    raise SystemExit(cli(main))
