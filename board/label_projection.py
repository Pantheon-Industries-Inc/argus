"""Project partial labels into consumer-safe values without changing the saved reply."""
from __future__ import annotations

import copy
import json
import math
import re

OUTCOMES = ('success', 'success_then_undone', 'failure', 'partial', 'unclear')
TASK_OUTCOMES = OUTCOMES + ('continues_past_cut',)
EVENT_OUTCOMES = ('success', 'failure', 'unclear')


def effective_outcome(value):
    return 'failure' if value == 'partial' else value


def _arm_names(config=None):
    arms = {arm: arm for arm in ('left', 'right', 'both')}
    config = config or {}
    labels = config.get('cam_labels') or []
    if len(config.get('views') or []) == 1 and labels:
        name = str(labels[0])
        arms[name.casefold()] = name
    return arms


def allowed_arms(config=None):
    return set(_arm_names(config).values()) | {'left', 'right', 'both'}
TEXT = {
    'timeline': ('arm', 'action', 'object', 'destination', 'spatial_relation', 'contribution', 'hands_wearing', 'notes'),
    'key_events': ('label', 'kind', 'outcome', 'note'),
    'state_changes': ('object', 'from', 'to', 'seen_in'),
    'completion': ('task_completed', 'success_predicate', 'undone_by', 'reason'),
    'tasks': ('task', 'outcome', 'success_predicate', 'note', 'undone_by'),
    'recovery': ('failure', 'correction', 'how_to_fix'),
    'data_issues': ('issue', 'category', 'severity', 'evidence', 'deliberate'),
    'operator_mistakes': ('issue', 'category', 'severity', 'evidence', 'deliberate'),
    'goal_alignment': ('relation', 'reason', 'observed_goal', 'given_goal', 'evidence', 'note'),
    'scene': ('setting',), 'scene.objects': ('name', 'location'),
    'contacts': ('id', 'hand', 'object', 'action', 'touch_seen', 'slip', 'note', 'evidence'),
    'contacts_missing': ('hand', 'object', 'action', 'note', 'evidence'),
}
ENUMS = {'task_completed': OUTCOMES, 'outcome': OUTCOMES,
         'severity': ('low', 'medium', 'high'), 'contribution': ('advancing', 'wasteful', 'idle'),
         'relation': ('aligned', 'narrower', 'broader', 'different', 'unrelated'),
         'deliberate': ('yes', 'no', 'unclear')}
BOOLS = {'timeline': ('hands_visible',), 'recovery': ('recovered',),
         'operator_mistakes': ('recovered',), 'goal_alignment': ('matches_given',)}
TEXT_LISTS = {'tasks': ('objects',), 'scene.objects': ('attributes',), 'scene_graph': ('relations',)}


def field_identity(field: str, row: int | None = None) -> str:
    if row is None:
        return field
    scope, dot, key = field.rpartition('.')
    return f'{scope}[{row}].{key}' if dot else f'{field}[{row}]'


def source_row_indices(labels: dict, scope: str, dropped: list | None = None) -> list[int]:
    """Derive provenance from actual omitted rows, never a reply-supplied index map."""
    rows = labels
    for key in scope.split('.'):
        rows = rows.get(key) if isinstance(rows, dict) else None
    size = len(rows) if isinstance(rows, list) else 0
    omitted = dropped if dropped is not None else labels.get('_dropped')
    omitted = {item['row'] for item in (omitted if isinstance(omitted, list) else [])
               if isinstance(item, dict) and item.get('field') == scope
               and type(item.get('row')) is int and item['row'] >= 0}
    total = size + len(omitted)
    if any(index >= total for index in omitted):
        return list(range(size))
    return [index for index in range(total) if index not in omitted]


def source_row_index(labels: dict, scope: str, index: int, dropped: list | None = None) -> int:
    mapping = source_row_indices(labels, scope, dropped)
    return mapping[index] if 0 <= index < len(mapping) else index


def diagnostics(labels: dict, dropped: list[dict]) -> dict[str, str]:
    """One diagnostic per source field, independent of the layer reporting it."""
    entries = {}
    mappings = {}
    for message in labels.get('_schema_violations') or []:
        row = re.match(r'^(\w+) row (\d+): (?:its )?(\w+)', message)
        field = re.match(r'^([\w.\[\]]+):', message)
        if row:
            if row[1] not in mappings:
                mappings[row[1]] = source_row_indices(labels, row[1], dropped)
            mapping = mappings[row[1]]
            index = int(row[2])
            source_index = mapping[index] if index < len(mapping) else index
            identity = field_identity(row[1] + '.' + row[3], source_index)
            if source_index != int(row[2]):
                message = identity + ':' + message.split(':', 1)[1]
        elif field:
            identity = field[1]
        elif message.startswith('completion.task_completed '):
            identity = 'completion.task_completed'
        else:
            identity = 'diagnostic:' + message
        entries.setdefault(identity, message)
    for item in dropped:
        identity = field_identity(item['field'], item.get('row'))
        value = f" ({item['value']!r})" if 'value' in item else ''
        entries.setdefault(identity, f"{identity}: {item['why']}{value}")
    return entries


def project(labels: dict, config: dict | None = None) -> tuple[dict, list[dict]]:
    """Keep valid fields, omit unsafe nested values, and report every omission with its source value."""
    labels = copy.deepcopy(labels)
    dropped = []
    # Replies can supply these internal keys too. Validate their shapes before the
    # structural parser appends diagnostics or consumers count them.
    for key in ('_schema_violations', '_dropped'):
        if key not in labels:
            continue
        value = labels[key]
        def valid(item):
            if key == '_schema_violations':
                return isinstance(item, str)
            return (isinstance(item, dict) and isinstance(item.get('field'), str)
                    and isinstance(item.get('why'), str)
                    and ('row' not in item or type(item['row']) is int and item['row'] >= 0))
        if not isinstance(value, list) or any(not valid(item) for item in value):
            dropped.append({'field': key, 'why': 'invalid diagnostic metadata', 'value': copy.deepcopy(value)})
            labels[key] = [item for item in value if valid(item)] if isinstance(value, list) else []
    from label.harness import typed_labels
    labels = typed_labels(labels)
    dropped.extend(labels.pop('_dropped', []))
    supplied_mapping = labels.pop('_schema_row_indices', None)
    mappings = {}
    for scope, rows in labels.items():
        if isinstance(rows, list) and any(item.get('field') == scope and 'row' in item for item in dropped):
            mappings[scope] = source_row_indices(labels, scope, dropped)
    if supplied_mapping is not None and supplied_mapping != mappings:
        dropped.append({'field': '_schema_row_indices', 'why': 'row map does not match omitted source rows',
                        'value': supplied_mapping})
    if mappings:
        labels['_schema_row_indices'] = mappings

    def invalid(row, key, scope, index, why):
        value = row.pop(key)
        dropped.append({'field': scope + '.' + key, **({'row': index} if index is not None else {}),
                        'why': why, 'value': value})

    def row_fields(row, scope, index=None):
        for key in TEXT.get(scope, ()):
            value = row.get(key)
            if value is not None and not isinstance(value, str):
                invalid(row, key, scope, index, 'not text')
            elif key == 'issue' and isinstance(value, str) and not value.strip():
                invalid(row, key, scope, index, 'empty text')
            elif value is not None and (key in ENUMS or key == 'arm'):
                normalized = value.strip().lower()
                allowed = (allowed_arms(config) if key == 'arm' else
                           TASK_OUTCOMES if scope == 'tasks' and key == 'outcome' else
                           EVENT_OUTCOMES if scope == 'key_events' and key == 'outcome' else ENUMS[key])
                if key == 'arm':
                    canonical = _arm_names(config)
                    normalized = canonical.get(value.strip().casefold(), normalized)
                if normalized in allowed:
                    row[key] = normalized
                else:
                    invalid(row, key, scope, index, 'unknown value')
        for key in BOOLS.get(scope, ()):
            if row.get(key) is not None and type(row[key]) is not bool:
                invalid(row, key, scope, index, 'not true or false')
                # Explicit unknown must not become a missing-field inference.
                row[key] = None
        for key in TEXT_LISTS.get(scope, ()):
            value = row.get(key)
            if value is not None and not isinstance(value, list):
                invalid(row, key, scope, index, 'not a list of text')
                row[key] = []
            elif isinstance(value, list):
                if any(not isinstance(item, str) for item in value):
                    dropped.append({'field': scope + '.' + key, **({'row': index} if index is not None else {}),
                                    'why': 'non-text list entries', 'value': copy.deepcopy(value)})
                row[key] = [item for item in value if isinstance(item, str)]
        if scope == 'timeline' and row.get('progress') is not None:
            value = row['progress']
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
                invalid(row, 'progress', scope, index, 'not a finite number from 0 to 1')
        if scope == 'contacts' and isinstance(row.get('seen'), dict):
            row_fields(row['seen'], 'contacts', index)

    for scope in sorted(set(TEXT) | set(TEXT_LISTS) | set(BOOLS)):
        if scope == 'scene.objects':
            rows = (labels.get('scene') or {}).get('objects') or []
        else:
            rows = labels.get(scope)
        if isinstance(rows, dict):
            row_fields(rows, scope)
        elif isinstance(rows, list):
            indices = source_row_indices(labels, scope, dropped)
            for index, row in enumerate(rows):
                if isinstance(row, dict):
                    row_fields(row, scope, indices[index])
    # Local assessments use the same annotation schema and retain their own attribution.
    assessments = labels.get('part_assessments')
    if assessments is not None and not isinstance(assessments, list):
        dropped.append({'field': 'part_assessments', 'why': 'not a list', 'value': assessments})
        labels.pop('part_assessments')
    elif isinstance(assessments, list):
        kept = []
        for index, assessment in enumerate(assessments):
            if not isinstance(assessment, dict):
                dropped.append({'field': 'part_assessments', 'row': index, 'why': 'not an object',
                                'value': assessment})
                continue
            local, omitted = project(assessment, config)
            kept.append(local)
            dropped.extend({**item, 'field': f'part_assessments[{index}].' + item['field']} for item in omitted)
        labels['part_assessments'] = kept
    # Nested assessments return their inherited omissions too. Preserve each
    # source omission once when their projection is revisited by another consumer.
    unique = {}
    for item in dropped:
        unique.setdefault(json.dumps(item, sort_keys=True, ensure_ascii=True), item)
    dropped = list(unique.values())
    issues = diagnostics(labels, dropped)
    if issues:
        labels['_schema_violations'] = list(issues.values())
    if dropped:
        labels['_dropped'] = dropped
    return labels, dropped
