"""Project partial labels into consumer-safe values without changing the saved reply."""
from __future__ import annotations

import copy
import math

OUTCOMES = ('success', 'success_then_undone', 'failure', 'partial', 'unclear')
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


def project(labels: dict) -> tuple[dict, list[dict]]:
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

    def invalid(row, key, scope, index, why):
        value = row.pop(key)
        dropped.append({'field': scope + '.' + key, **({'row': index} if index is not None else {}),
                        'why': why, 'value': value})

    def row_fields(row, scope, index=None):
        for key in TEXT.get(scope, ()):
            value = row.get(key)
            if value is not None and not isinstance(value, str):
                invalid(row, key, scope, index, 'not text')
            elif value is not None and key in ENUMS and value not in ENUMS[key]:
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
            for index, row in enumerate(rows):
                if isinstance(row, dict):
                    row_fields(row, scope, index)
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
            local, omitted = project(assessment)
            kept.append(local)
            dropped.extend({**item, 'field': f'part_assessments[{index}].' + item['field']} for item in omitted)
        labels['part_assessments'] = kept
    return labels, dropped
