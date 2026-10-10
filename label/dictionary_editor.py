"""Separate human dictionary edits and the public interpretation projection."""
from __future__ import annotations

import copy
import fcntl
import json
import re
import time
from pathlib import Path, PureWindowsPath

from label.atomic import write_atomic
from label.dictionary import effective


class RevisionConflict(ValueError):
    """The saved human revision changed since the visitor loaded it."""


def _read(path: Path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def _record(job: Path) -> dict:
    record = _read(job / 'dictionary.json')
    if not isinstance(record, dict) or not isinstance(record.get('inventory'), dict):
        raise ValueError('The data dictionary is unavailable.')
    fields = record['inventory'].get('fields')
    if not isinstance(fields, list) or any(not isinstance(f, dict) or not isinstance(f.get('id'), str) for f in fields):
        raise ValueError('The data dictionary fields are unreadable.')
    return record


def _overrides(job: Path) -> dict:
    value = _read(job / 'dictionary_overrides.json', {})
    if not isinstance(value, dict):
        raise ValueError('The saved dictionary edits are unreadable.')
    if 'entries' not in value:
        return {'schema': 1, 'revision': 0, 'entries': value, 'history': []}
    if (value.get('schema') != 1 or type(value.get('revision')) is not int or value['revision'] < 0
            or not isinstance(value.get('entries'), dict) or not isinstance(value.get('history'), list)):
        raise ValueError('The saved dictionary edits are unreadable.')
    return value


def _source(value):
    """Public source attribution never publishes a private absolute path."""
    if isinstance(value, str):
        windows = PureWindowsPath(value)
        if windows.is_absolute():
            return windows.name
        if Path(value).is_absolute():
            return Path(value).name
        return re.sub(r'(?<![\w])/(?:Users|home|data|app|private|tmp|var)/[^\s<>"\']+',
                      lambda m: Path(m.group()).name, value)
    if isinstance(value, list):
        return [_source(v) for v in value]
    if isinstance(value, dict):
        return {k: _source(v) for k, v in value.items() if k in ('file', 'key', 'episode', 'context_path')}
    return value


def _entry(value):
    return {k: copy.deepcopy(value[k]) for k in ('meaning', 'role', 'layout', 'provenance') if k in value}


def _history(rows):
    return [{k: _entry(row[k]) if k in ('before', 'after') else row[k]
             for k in ('revision', 'field_id', 'at', 'before', 'after', 'after_labelling') if k in row}
            for row in rows if isinstance(row, dict)]


def context_dictionary(overlay: dict) -> dict:
    """Project an already scoped reader overlay without its private human attribution."""
    result = {k: copy.deepcopy(overlay[k]) for k in
              ('schema', 'inventory_digest', 'episode_id', 'status', 'model', 'fields', 'entries',
               'machine_entries', 'limitations', 'override_limitations', 'missing_fields',
               'request_field_ids', 'deferred_fields') if k in overlay}
    for field in result.get('fields', []):
        for key in ('source', 'bindings', 'limitations'):
            if key in field:
                field[key] = _source(field[key])
    result['limitations'] = _source(list(overlay.get('limitations') or []) + list(overlay.get('override_limitations') or []))
    result['revision'] = overlay.get('override_revision', 0)
    result['history'] = _history(overlay.get('override_history') or [])
    result['after_labelling'] = any(row.get('after_labelling') is True for row in result['history'])
    result['editable'] = False
    return result


def public_dictionary(job: Path, episode_id: str | None = None) -> dict:
    """Original descriptors and machine entries beside the current human interpretation."""
    job = Path(job)
    record, overrides = _record(job), _overrides(job)
    resolved = effective(record, overrides)
    fields = [{k: _source(f[k]) if k in ('source', 'bindings', 'limitations') else copy.deepcopy(f[k])
               for k in ('id', 'name', 'kind', 'shape', 'dtype', 'names', 'rate_hz', 'source', 'episodes',
                         'bindings', 'summary', 'limitations', 'unit', 'units', 'response_direction', 'calibration', 'coordinate_frame', 'sensor_type', 'description') if k in f}
              for f in record['inventory']['fields'] if episode_id is None or episode_id in (f.get('episodes') or [])]
    ids = {f['id'] for f in fields}
    history = _history([row for row in overrides['history']
                        if isinstance(row, dict) and row.get('field_id') in ids])
    return {'schema': record.get('schema'), 'inventory_digest': record.get('inventory_digest'),
            'status': record.get('status'), 'model': record.get('model'), 'episode_id': episode_id, 'fields': fields,
            'entries': {k: _entry(v) for k, v in resolved.get('entries', {}).items() if k in ids},
            'machine_entries': {k: _entry(v) for k, v in record.get('entries', {}).items() if k in ids},
            'revision': overrides['revision'], 'history': history,
            'after_labelling': any(row.get('after_labelling') is True for row in history),
            'limitations': _source(list(record.get('limitations') or []) + list(resolved.get('override_limitations') or [])),
            'missing_fields': [k for k in resolved.get('missing_fields', []) if k in ids],
            **{key: [ident for ident in record[key] if ident in ids]
               for key in ('request_field_ids', 'deferred_fields') if key in record},
            'unknown_entries': len(record.get('unknown_entries') or []),
            'damaged_entries': len(record.get('damaged_entries') or []), 'editable': False}


def for_episode(episode: Path, owner: str | None = None) -> dict | None:
    """Find the upload receipt beside a prepared episode without changing its label."""
    episode = Path(episode)
    for parent in list(episode.parents)[:4]:
        if (parent / 'dictionary.json').is_file():
            try:
                return public_dictionary(parent, owner or episode.name)
            except (ValueError, OSError):
                return {'status': 'unreadable', 'fields': [], 'entries': {}, 'machine_entries': {},
                        'limitations': ['The saved data dictionary is unreadable.'], 'editable': False}
    return None


def _labelled(job: Path) -> bool:
    paths = list((job / 'run' / 'out').glob('*.json'))
    for path in paths:
        try:
            record = _read(path)
        except (ValueError, OSError):
            continue
        if isinstance(record, dict) and not record.get('dry_run') and not record.get('no_reply'):
            if any(k in record for k in ('labels', 'raw', 'raw_text', 'response', 'parse_ok')):
                return True
    return False if paths else any((job / 'qa').glob('*.json'))


def save_override(job: Path, body: dict, actor: str) -> dict:
    """Validate and atomically append one revision under an upload file lock."""
    job = Path(job)
    if not isinstance(body, dict) or set(body) - {'revision', 'field_id', 'meaning', 'role', 'layout'}:
        raise ValueError('The dictionary edit is not valid.')
    revision, ident = body.get('revision'), body.get('field_id')
    if type(revision) is not int or revision < 0 or not isinstance(ident, str) or len(ident) > 128:
        raise ValueError('The dictionary revision or field is not valid.')
    edit = {k: body[k] for k in ('meaning', 'role', 'layout') if k in body}
    if not edit:
        raise ValueError('Choose a dictionary value to edit.')
    with (job / '.dictionary_overrides.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record, overrides = _record(job), _overrides(job)
        if overrides['revision'] != revision:
            raise RevisionConflict('The dictionary changed. Reload it before saving.')
        if ident not in {f['id'] for f in record['inventory']['fields']}:
            raise ValueError('The dictionary field is unknown.')
        validated = effective(record, {'schema': 1, 'revision': revision, 'entries': {ident: edit}, 'history': []})
        if validated.get('override_limitations'):
            raise ValueError(' '.join(validated['override_limitations']))
        clean = {k: validated['entries'][ident][k] for k in edit}
        previous = copy.deepcopy(overrides['entries'].get(ident) or {})
        overrides['entries'][ident] = {**previous, **clean}
        overrides['revision'] += 1
        overrides['history'].append({'revision': overrides['revision'], 'field_id': ident, 'at': time.time(),
                                     'actor': actor, 'before': previous, 'after': copy.deepcopy(overrides['entries'][ident]),
                                     'after_labelling': _labelled(job)})
        write_atomic(job / 'dictionary_overrides.json', overrides)
        return public_dictionary(job)
