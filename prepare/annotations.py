"""Recorded index tables supply attributed annotations, never inferred labels or arm state."""
from collections import defaultdict
import json
import math
from pathlib import Path


BOOKKEEPING_COLUMNS = frozenset({'frame_index', 'episode_index', 'index', 'timestamp'})
SCOPE_FILES = ('info.json', 'metadata.json', 'dataset.json', 'meta/info.json')
DESCRIPTION_FIELDS = ('dataset_note', 'dataset_description', 'description', 'collection_note')


def declared_context(declarations, rig: str | None) -> dict:
    """Resolve explicit declarations without inferring scope from descriptive text. A declared task mode (task_mode,
    or sessions true) is kept as provenance when the declarations agree on one; it never changes how a recording is
    labelled, since every recording is labelled as a list of tasks (label/prompts.py), so declarations that disagree,
    or a mode that is neither episode nor session, keep none and refuse nothing."""
    context = {'profile': rig} if rig is not None else {}
    modes, notes = set(), []
    for data in declarations:
        mode = data.get('task_mode')
        if mode is not None:
            modes.add(mode if isinstance(mode, str) else json.dumps(mode, sort_keys=True, default=str))
        elif data.get('sessions') is True:
            modes.add('session')
        for key in DESCRIPTION_FIELDS:
            note = data.get(key)
            if isinstance(note, str) and note.strip() and note not in notes:
                notes.append(note)
    if len(modes) == 1 and next(iter(modes)) in ('episode', 'session'):
        context['task_mode'] = modes.pop()
    if notes:
        context['collection_note'] = '\n'.join(notes)
    return context


def merge_source_contexts(rig: str, *contexts) -> dict:
    return declared_context([{'task_mode': context.get('task_mode'),
                              'collection_note': context.get('collection_note')} for context in contexts], rig)


def source_root(upload: Path) -> Path:
    """A common directory wrapper is transparent, but scope declarations and sibling entries end descent."""
    from prepare.formats import hidden_part
    root = Path(upload)
    if root.is_file():
        return root.parent
    while root.is_dir() and not any((root / relative).is_file() for relative in SCOPE_FILES):
        children = [path for path in root.iterdir() if not hidden_part(path.name)]
        if len(children) != 1 or not children[0].is_dir() or children[0].is_symlink():
            break
        root = children[0]
    return root


def scope_records(upload: Path):
    """Native declarations, retaining their original file paths and complete JSON objects."""
    root = source_root(upload)
    for relative in SCOPE_FILES:
        path = root / relative
        try:
            if not path.is_file() or path.stat().st_size > 1_000_000:
                continue
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            yield path, data


def source_context(upload: Path, rig: str | None) -> dict:
    """Read explicit upload task scope and description for preparation and Data policy resolution."""
    return declared_context([data for _, data in scope_records(upload)], rig)


def hdf_source_context(f, group: str, rig: str, coverage: dict) -> dict:
    """Only the file root and the selected recording group can declare recording scope."""
    from prepare.hdf_metadata import attribute_values, value
    from label.source_eligibility import mark_hdf
    owners = [('', f), *([(group, f[group])] if group else [])]
    declarations = [{key: value(raw) for key, raw in attribute_values(obj, ('task_mode', 'sessions', *DESCRIPTION_FIELDS))}
                    for _, obj in owners]
    context = declared_context(declarations, rig)
    for (name, _), fields in zip(owners, declarations):
        for key, field in fields.items():
            # a declared task mode is read as provenance, never offered as metadata, whether or not they agree
            consumed = key == 'task_mode' and field in ('episode', 'session')
            consumed |= key == 'sessions' and field is True and fields.get('task_mode') is None
            consumed |= key in DESCRIPTION_FIELDS and isinstance(field, str) and bool(field.strip())
            if consumed:
                mark_hdf(coverage, str(Path(f.filename).resolve()), name, key)
    return context


def scope_pointers(fields, context):
    """Only native leaves already read as task scope (provenance, declared_context) or supplied as recording
    description."""
    pointers = []
    mode = fields.get('task_mode')
    if mode in ('episode', 'session'):
        pointers.append('/task_mode')
    if mode is None and fields.get('sessions') is True:
        pointers.append('/sessions')
    for key in DESCRIPTION_FIELDS:
        value = fields.get(key)
        if isinstance(value, str) and value.strip() and value in context.get('collection_note', ''):
            pointers.append('/' + key)
    return pointers


def mark_source_context_read(root, items):
    """Record exact source-scope paths opened during planning, without claiming unrelated notes."""
    records = list(scope_records(root))
    context = declared_context([fields for _, fields in records], None)
    paths = [str(path) for path, fields in records if scope_pointers(fields, context)]
    for item in items:
        item['scope_metadata_read'] = paths


def represented_scope(ctx, roots):
    """Mark only original scope leaves already supplied through this prepared context."""
    from label.source_eligibility import mark_json
    records = {str(path): fields for root in roots for path, fields in scope_records(root)}
    used = []
    for source, fields in records.items():
        pointers = scope_pointers(fields, ctx)
        if not pointers:
            continue
        # Keep complete native fields; only the leaves already supplied as recording context are represented.
        ctx.setdefault('recorded_metadata', {}).setdefault(source, fields)
        represented = ctx.setdefault('represented_source_members', {})
        for pointer in pointers:
            mark_json(represented, source, pointer)
        used.append(source)
    return used


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


def _episode_catalog(path, meta):
    """Canonical episode tables require an episode owner, including damaged or missing owner cells."""
    return path.stem == 'episodes' or Path(meta) / 'episodes' in path.parents


def index_tables(meta, columns, read_jsonl):
    """Tables keyed by (exact recorded column, original path), with every nonkey field and conflicting row.

    A recorded column or named parquet index establishes ownership. Positional row order never does.
    Episode-owned rows contain references to codes, not global definitions of those codes.
    Only a unique nonempty text field supplies a display label; other cells remain attributed notes.
    """
    import pandas as pd
    from prepare.formats import recorded_index
    columns = [c for c in columns if c not in BOOKKEEPING_COLUMNS]
    out = {}
    for path in sorted(Path(meta).glob('*')):
        if path.suffix not in ('.jsonl', '.parquet'):
            continue
        if _episode_catalog(path, meta):
            continue
        try:
            if path.suffix == '.jsonl':
                rows = read_jsonl(path)
                cells = [(i, {k: original(v) for k, v in row.items()}) for i, row in enumerate(rows)]
            else:
                df = pd.read_parquet(path)
                cells = []
                for i, index in enumerate(df.index):
                    row = {k: original(df[k].iloc[i]) for k in df.columns}
                    if df.index.name is not None:
                        row.setdefault(df.index.name, original(index))
                    elif not isinstance(df.index, pd.RangeIndex):
                        row['table_index'] = original(index)
                    cells.append((i, row))
        except Exception as error:  # noqa: BLE001 - any table that does not read is named in its issues
            out[(None, str(path))] = {'labels': {}, 'issues': [{'why': f'table read failed ({error})'}]}
            continue
        # Presence establishes the ownership domain even when the episode owner is invalid.
        # Those claims are retained by episode_metadata, without acquiring a global task owner.
        cells = [(i, row) for i, row in cells if 'episode_index' not in row]
        keys = [c for c in columns if any(c in row for _, row in cells)]
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


def index_issue_text(column, issue):
    """One readable reason for an index table problem, naming the row or index it concerns, never the raw record."""
    if column is None:
        return issue['why']
    if 'row' in issue:
        return f"row {issue['row']} has {column} {issue['value']!r}, which is not an index ({issue['why']})"
    return f"{column} {issue['value']} has {issue['why']}"


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


def language_spans(values, times, source, column, *, frame_indices=None):
    """Recorded strings are their own labels. Empty or nonstring cells retain rejected original claims."""
    values = [original(x) for x in values]
    texts = list(dict.fromkeys(x for x in values if isinstance(x, str) and x.strip()))
    codes = {text: i for i, text in enumerate(texts)}
    encoded = [codes.get(x) if isinstance(x, str) else None for x in values]
    spans, unresolved = annotation_spans(encoded, times, {i: text.strip() for text, i in codes.items()},
                                         source, column, frame_indices=frame_indices)
    for step in spans:
        step['value'] = values[step['row_start']]
        step['values'] = values[step['row_start']:step['row_end'] + 1]
    for claim in unresolved:
        claim['value'] = values[claim['row']]
        claim['why'] = 'recorded text is empty or is not a string'
    return spans, unresolved


def metadata_value(value):
    """JSON representations of recorded metadata types, with binary and datetime types kept explicit."""
    import datetime
    import decimal
    value = original(value)
    if isinstance(value, dict):
        return {k: metadata_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [metadata_value(v) for v in value]
    if isinstance(value, bytes):
        return {'recorded_type': 'bytes', 'hex': value.hex()}
    if isinstance(value, (datetime.date, datetime.time)):
        result = {'recorded_type': type(value).__name__, 'iso8601': value.isoformat()}
        if type(value).__name__ == 'Timestamp':
            try:
                result['integer_ns'] = value.value
            except OverflowError:
                pass
        return result
    if isinstance(value, decimal.Decimal):
        return {'recorded_type': 'Decimal', 'decimal_text': str(value)}
    return value


def episode_metadata(meta, read_jsonl):
    """Full metadata claims by explicit episode owner. Unassigned malformed rows never acquire an owner."""
    import pandas as pd
    from prepare.formats import names_annotation, recorded_index
    meta = Path(meta)
    paths = set(meta.glob('*.jsonl')) | set(meta.glob('*.parquet'))
    paths |= set((meta / 'episodes').rglob('*.parquet'))
    owned, unassigned = {}, []
    for path in sorted(paths):
        try:
            if path.suffix == '.jsonl':
                read_jsonl(path)
                rows = []
                # Retain malformed physical lines independently of the readable object rows.
                for number, line in enumerate(path.read_text(encoding='utf-8-sig', errors='replace').splitlines()):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                        valid = isinstance(value, dict)
                    except ValueError:
                        valid = False
                    if valid:
                        rows.append((number, value))
                    else:
                        unassigned.append({'source': str(path), 'line': number + 1, 'raw_text': line,
                                           'why': 'not a JSON object; no episode owner can be established'})
            else:
                with pd.option_context("future.infer_string", False):
                    df = pd.read_parquet(path)
                rows = []
                for number, index in enumerate(df.index):
                    row = {k: original(df[k].iloc[number]) for k in df.columns}
                    if df.index.name is not None:
                        row.setdefault(df.index.name, original(index))
                    rows.append((number, row))
        except Exception as error:  # noqa: BLE001 - any table that does not read is named in unassigned
            unassigned.append({'source': str(path), 'why': f'metadata table read failed ({error})'})
            continue
        for number, row in rows:
            if 'episode_index' not in row and not (_episode_catalog(path, meta) or names_annotation(path.name)):
                continue
            claim = {'source': str(path), 'row': number, 'fields': metadata_value(row)}
            try:
                owner = recorded_index(row.get('episode_index'))
            except (TypeError, ValueError) as error:
                unassigned.append({**claim, 'why': f'{error}; no episode owner can be established'})
                continue
            owned.setdefault(owner, {}).setdefault(str(path), []).append(claim)
    unassigned.sort(key=lambda x: (x["source"], x.get("row", x.get("line", 1) - 1)))
    return owned, unassigned
