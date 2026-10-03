"""Recorded index tables supply attributed annotations, never inferred labels or arm state."""
from collections import defaultdict
import json
import math
from pathlib import Path


UPLOAD = None
BOOKKEEPING_COLUMNS = frozenset({'frame_index', 'episode_index', 'index', 'timestamp'})


def original(value):
    """Keep scalar types and nested recorded fields when moving numpy cells into JSON metadata."""
    if hasattr(value, 'tolist'):
        return original(value.tolist())
    if isinstance(value, dict):
        return {str(k): original(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [original(v) for v in value]
    return value


def _label(claims):
    fields = [c['fields'] for c in claims]
    different = {json.dumps(x, sort_keys=True, default=str) for x in fields}
    texts = [v.strip() for v in fields[0].values() if isinstance(v, str) and v.strip()]
    why = 'conflicting recorded rows for this index' if len(different) > 1 else \
          'no unique nonempty text field' if len(texts) != 1 else None
    return {'label': texts[0] if why is None else None, 'notes': fields[0], 'claims': claims, 'why': why}


def index_tables(meta, columns, read_jsonl):
    """Tables keyed by (exact recorded column, original path), with every nonkey field and conflicting row.

    A recorded column or named parquet index establishes ownership. Positional row order never does.
    Only a unique nonempty text field supplies a display label; other cells remain attributed notes.
    """
    import pandas as pd
    from prepare.formats import recorded_index
    columns = [c for c in columns if c not in BOOKKEEPING_COLUMNS]
    out = {}
    for path in sorted(Path(meta).glob('*')):
        if path.suffix not in ('.jsonl', '.parquet'):
            continue
        try:
            if path.suffix == '.jsonl':
                rows = read_jsonl(path)
                keys = [c for c in columns if any(c in row for row in rows)]
                cells = [(i, {k: original(v) for k, v in row.items()}) for i, row in enumerate(rows)]
            else:
                df = pd.read_parquet(path)
                keys = [c for c in columns if c in df.columns or c == df.index.name]
                cells = []
                for i, index in enumerate(df.index):
                    row = {k: original(df[k].iloc[i]) for k in df.columns}
                    if df.index.name is not None:
                        row.setdefault(df.index.name, original(index))
                    elif not isinstance(df.index, pd.RangeIndex):
                        row['table_index'] = original(index)
                    cells.append((i, row))
        except Exception as error:
            out[(None, str(path))] = {'labels': {}, 'issues': [{'why': f'table read failed ({error})'}]}
            continue
        for column in keys:
            claims, issues = defaultdict(list), []
            for row_number, row in cells:
                value = row.get(column)
                fields = {k: v for k, v in row.items() if k != column}
                claim = {'row': row_number, 'value': value, 'fields': fields, 'source': str(path)}
                try:
                    claims[recorded_index(value)].append(claim)
                except (TypeError, ValueError) as error:
                    issues.append({**claim, 'why': str(error)})
            labels = {index: _label(rows) for index, rows in claims.items()}
            issues.extend({'value': index, 'claims': record['claims'], 'why': record['why']}
                          for index, record in labels.items() if record['why'])
            out[(column, str(path))] = {'labels': labels, 'issues': issues}
    return out


def resolved_labels(tables, column):
    """Combine all claims for one exact column so a later table cannot silently replace an earlier owner."""
    claims = defaultdict(list)
    for (key, _), table in tables.items():
        if key == column:
            for index, record in table['labels'].items():
                claims[index].extend(record['claims'])
    return {index: _label(rows) for index, rows in claims.items()}


def _finite(value):
    try:
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def annotation_spans(values, times, labels, source, column, *, final_boundary=None, frame_indices=None):
    """Equal consecutive codes form attributed spans on their recorded row clock.

    A next recorded transition ends a span; only an explicitly measured final boundary ends the last.
    Missing times, backwards clocks and missing frames break continuity. Original row claims stay visible.
    """
    from prepare.formats import recorded_index
    values = [original(v) for v in values]
    times = [original(t) for t in times] if times is not None else [None] * len(values)
    frames = [original(v) for v in frame_indices] if frame_indices is not None else None
    spans, unresolved = [], []
    start, selected = None, None

    def time_at(row):
        return float(times[row]) if row < len(times) and _finite(times[row]) else None

    def end(row, reason=None):
        nonlocal start, selected
        if start is None:
            return
        at = time_at(start)
        stop = time_at(row) if row < len(values) else final_boundary
        step = {'source': str(source), 'column': column, 'value': values[start],
                'values': values[start:row], 'row_start': start, 'row_end': row - 1,
                'raw_times': times[start:row],
                'label': selected['label'], 'notes': selected.get('notes', {}),
                'claims': selected.get('claims', [])}
        if at is not None:
            step['t0'] = float(at)
        if at is not None and _finite(stop) and float(stop) > float(at) and reason is None:
            step['t1'] = float(stop)
        else:
            step['timing_reason'] = reason or ('timestamp is missing or nonfinite' if at is None else
                                              'no measured end boundary on the recorded clock')
        spans.append(step)
        start, selected = None, None

    for row, value in enumerate(values):
        discontinuity = None
        if row:
            a, b = time_at(row - 1), time_at(row)
            if a is None or b is None or b <= a:
                discontinuity = 'timestamps do not establish continuity between these rows'
            if frames is not None:
                try:
                    contiguous = recorded_index(frames[row]) == recorded_index(frames[row - 1]) + 1
                except (IndexError, TypeError, ValueError):
                    contiguous = False
                if not contiguous:
                    discontinuity = 'recorded frame indices have a gap or are invalid'
        record = None
        try:
            index = recorded_index(value)
            record = labels.get(index)
            if isinstance(record, str):
                record = {'label': record} if record.strip() else None
            if not record or not record.get('label'):
                raise ValueError((record or {}).get('why') or 'no recorded table label for this index')
        except (TypeError, ValueError) as error:
            end(row, discontinuity)
            unresolved.append({'source': str(source), 'column': column, 'row': row, 'value': value,
                               'raw_time': times[row] if row < len(times) else None,
                               'why': str(error), 'claims': (record or {}).get('claims', [])})
            continue
        if start is not None and (discontinuity or value != values[start]):
            end(row, discontinuity)
        if start is None:
            start, selected = row, record
    end(len(values))
    return spans, unresolved
