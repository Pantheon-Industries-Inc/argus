"""Recorded LeRobot task tables and publisher claims keep their original field ownership."""
from copy import deepcopy
from pathlib import Path

from prepare.annotations import index_tables, original


UPLOAD = None
TABLE_COLUMNS = {'tasks': 'task_index', 'subtasks': 'low_level_task_index',
                 'human_subtasks': 'human_role_subtask_index'}
PUBLISHER_COLUMNS = frozenset({'is_error_segment', 'is_intervention_segment', 'is_high_jerk_segment'})
PUBLISHER_FIELDS = PUBLISHER_COLUMNS | {'task_status'}


def publisher_columns(columns):
    """Exact recorded QC flag columns belong to publisher claims, outside numeric judgment signals."""
    return {column for column in columns if column in PUBLISHER_COLUMNS}


def scoped_index_tables(meta, columns, read_jsonl):
    """Named format tables retain their task_index key while binding to their recorded frame column."""
    columns = list(columns)
    tables = index_tables(meta, list(dict.fromkeys(columns + ['task_index'])), read_jsonl)
    out = {}
    for (column, path), table in tables.items():
        owner = TABLE_COLUMNS.get(Path(path).stem)
        if column == 'task_index' and owner is not None:
            if owner not in columns:
                continue
            if owner != column and (owner, path) in tables:
                continue
            table = deepcopy(table)
            for record in table['labels'].values():
                for claim in record['claims']:
                    claim['table_key'] = 'task_index'
            out[(owner, path)] = table
        elif column is None or column in columns:
            out[(column, path)] = table
    return out


def _judgment_notes(value, *, annotation_spans=False):
    if isinstance(value, dict):
        projected = {key: _judgment_notes(cell, annotation_spans=annotation_spans or key == 'recorded annotation spans')
                     for key, cell in value.items() if key not in PUBLISHER_FIELDS}
        notes = projected.get('notes')
        if annotation_spans and isinstance(notes, dict) and isinstance(projected.get('label'), str) \
                and notes.get('task') == projected['label']:
            notes.pop('task')
            if not notes:
                projected.pop('notes')
        return projected
    if isinstance(value, list):
        return [_judgment_notes(cell, annotation_spans=annotation_spans) for cell in value]
    return value


def _flag_spans(values, fps):
    result, start = [], None
    for row, value in enumerate(list(values) + [False]):
        if value and start is None:
            start = row
        elif not value and start is not None:
            result.append([round(start / fps, 2), round(row / fps, 2)])
            start = None
    return result


def retain_metadata(ctx, root, df, data_path):
    """Retain owned instructions and typed publisher rows while projecting safe judgment notes."""
    from prepare import formats
    owner = ctx.get('episode_index')
    owned = root.get('episode_metadata', {}).get(owner, {})
    high_level = []
    publisher = deepcopy(ctx.get('publisher_labels') or {})
    claims = {}
    for path, rows in owned.items():
        for row in rows:
            fields = row.get('fields') or {}
            instruction = fields.get('high_level_instruction')
            if isinstance(instruction, str) and instruction.strip():
                high_level.append((instruction.strip(), path))
            selected = {key: original(value) for key, value in fields.items() if key in PUBLISHER_FIELDS}
            if selected:
                claims.setdefault(path, []).append({**original(row), 'fields': selected})
    unique = set(text for text, _ in high_level)
    if len(unique) == 1:
        ctx['instruction'] = high_level[0][0]
        ctx['instruction_note'] = 'The instruction is the owned episode high_level_instruction field, a recorded claim to check.'
        ctx['instruction_source'] = high_level[0][1]
    elif high_level:
        ctx.pop('instruction', None)
        ctx.pop('instruction_note', None)
        formats.add_issue(ctx, 'metadata_unreadable', 'Owned high_level_instruction claims conflict; original metadata is retained without an inferred instruction.')
    if claims:
        publisher['metadata'] = claims
        statuses = [row['fields']['task_status'] for rows in claims.values() for row in rows
                    if 'task_status' in row['fields']]
        if statuses and all(status == statuses[0] for status in statuses):
            publisher['task_status'] = statuses[0]
    recorded = ctx.setdefault('recorded_metadata', {})
    for stem in TABLE_COLUMNS:
        path = Path(root['dir']) / 'meta' / (stem + '.jsonl')
        if path.exists():
            recorded[str(path)] = original(formats.read_jsonl(path, root))
            root['metadata_read'].add(path.resolve())
    if df is not None:
        columns = {}
        for column in sorted(publisher_columns(df.columns)):
            values = df[column].to_numpy()
            columns[column] = {'source': str(data_path), 'column': column, 'dtype': str(values.dtype),
                               'values': original(values)}
        if columns:
            publisher['columns'] = columns
            publisher['row_clock'] = {key: original(df[key].to_numpy()) for key in ['timestamp', 'frame_index'] if key in df}
            fps = ctx.get('fps') or root.get('fps')
            if fps:
                for column, target in [('is_error_segment', 'error_spans_s'),
                                       ('is_intervention_segment', 'intervention_spans_s'),
                                       ('is_high_jerk_segment', 'high_jerk_spans_s')]:
                    if column in columns:
                        publisher[target] = _flag_spans(columns[column]['values'], fps)
                publisher['computed_span_note'] = 'Span seconds are computed from row positions and nominal fps; original row clocks and flags are retained separately.'
    if publisher:
        ctx['publisher_labels'] = publisher
    notes = ctx.get('uploader_notes')
    if notes:
        safe = _judgment_notes(notes)
        if safe != notes:
            ctx['recorded_uploader_notes'] = deepcopy(notes)
            ctx['uploader_notes'] = safe
            ctx.pop('uploader_annotation', None)
            formats.set_uploader_notes(ctx, safe)
    return ctx
