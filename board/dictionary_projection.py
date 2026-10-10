"""Current public dictionary and evidence projection over an unchanged saved label."""
from __future__ import annotations

import copy
import json
from pathlib import Path


def project(label: dict, episode_dir: Path, owner: str | None = None) -> dict:
    """Return the current interpretation without writing to the label or prepared source."""
    shown = copy.deepcopy(label)
    episode_dir = Path(episode_dir)
    try:
        ctx = json.loads((episode_dir / 'context.json').read_text())
    except (OSError, ValueError):
        ctx = {}
    return apply(shown, ctx, episode_dir, owner)


def apply(shown: dict, ctx: dict, episode_dir: Path, owner: str | None = None) -> dict:
    """Apply the same projection while board/build already has the prepared context."""
    from label.dictionary_editor import context_dictionary, for_episode

    owner = owner or (ctx.get('piece') or {}).get('of') or ctx.get('episode_id') or episode_dir.name
    dictionary = for_episode(episode_dir, owner)
    if dictionary is None and isinstance(ctx.get('data_dictionary'), dict):
        dictionary = context_dictionary(ctx['data_dictionary'])
    old = shown.get('data_dictionary') or ctx.get('data_dictionary') or {}
    if dictionary is not None:
        shown['data_dictionary'] = dictionary
    effective_ctx = {**ctx, **({'data_dictionary': dictionary} if dictionary is not None else {})}
    if isinstance(shown.get('evidence_inspection'), dict):
        from label.evidence_access import same_source_proof, withhold_stale_untimed, resolve_piece
        inspection = shown['evidence_inspection']
        withhold_stale_untimed(inspection, effective_ctx, episode_dir)
        if inspection.get('inspections') or inspection.get('parts'):
            inspection['source_current'] = same_source_proof(
                inspection.get('source_proof'), effective_ctx, episode_dir)
            if not inspection['source_current']:
                inspection.setdefault('limitations', []).append(
                    'Saved inspection regions use changed or missing source data; receipts withheld pending refresh.')
        for part in inspection.get('parts') or []:
            if not isinstance(part, dict) or not isinstance(part.get('record'), dict):
                continue
            record = part['record']
            directory = resolve_piece(episode_dir, part.get('source_location'), {'index': part.get('part')}, ctx)
            try:
                piece_ctx = json.loads((directory / 'context.json').read_text()) if directory else None
            except (OSError, ValueError):
                piece_ctx = None
            record['source_current'] = bool(
                inspection.get('source_current') and isinstance(piece_ctx, dict)
                and same_source_proof(record.get('source_proof'), piece_ctx, directory))
    if isinstance(shown.get('sensor_evidence'), dict) or isinstance(shown.get('grip_evidence'), dict):
        from label.sensor_evidence import compatible
    if isinstance(shown.get('sensor_evidence'), dict):
        if not compatible(shown['sensor_evidence'], effective_ctx, episode_dir):
            saved = shown['sensor_evidence']
            withheld = saved.setdefault('withheld_findings', [])
            withheld.extend(f for f in saved.get('findings') or [] if f not in withheld)
            saved['findings'] = []
            note = 'Sensor source, interpretation or timing changed since labeling; saved findings withheld pending refresh.'
            if note not in saved.setdefault('limitations', []):
                saved['limitations'].append(note)
    if isinstance(shown.get('grip_evidence'), dict) and shown['grip_evidence'].get('version') == 1:
        if not compatible(shown['grip_evidence'], effective_ctx, episode_dir):
            shown['withheld_grip_evidence'] = shown.pop('grip_evidence')
            shown.setdefault('dictionary_projection_limitations', []).append(
                'Saved legacy grip findings have no verifiable current source and interpretation proof.')
    changed = (isinstance(old.get('entries'), dict) and
               (dictionary is None or old['entries'] != dictionary.get('entries')))
    if changed:
        for key in ('contacts', 'contacts_missing', 'tactile_qc'):
            if shown.get(key):
                shown['withheld_' + key] = shown.pop(key)
        checks = shown.get('dataset_checks')
        if isinstance(checks, dict) and 'contact_checks' in checks:
            shown['withheld_contact_checks'] = checks.pop('contact_checks')
        if any(key in shown for key in ('withheld_contacts', 'withheld_contacts_missing',
                                         'withheld_contact_checks', 'withheld_tactile_qc')):
            shown.setdefault('dictionary_projection_limitations', []).append(
                'Saved contact claims use an earlier sensor interpretation and are withheld pending refresh.')
    return shown
