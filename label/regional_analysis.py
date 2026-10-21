"""Restore the original regional producer outputs for the independent broad pass."""
from __future__ import annotations

import copy

import numpy as np

from label import derived_measurements as dm, semantic_evidence as semantic, sensor_evidence as se, signals
from label.dictionary_context import field_interpretation

MAX_OUTPUT_SERIES = 64
MAX_ROWS = 512
MAX_VALUES = 8192


def _note(doc, text):
    if text not in doc.setdefault('limitations', []):
        doc['limitations'].append(text)


def _columns(value):
    return list(range(value['start'], value['start'] + value['count'])) if isinstance(value, dict) else value


def _compact_columns(columns):
    if len(columns) > 256 and columns == list(range(columns[0], columns[0] + len(columns))):
        return {'start': columns[0], 'count': len(columns)}
    return columns


def _source(ep, sensor, derivation):
    """Use the original derivation method only with matching owned source metadata."""
    name, n = sensor.get('name'), derivation['native_samples']
    meta = next((m for m in ep['context'].get('signals') or [] if m.get('name') == name), None)
    if meta is None or sensor.get('descriptor_digest') != se._descriptor_digest(meta):
        raise ValueError('source descriptor does not match the regional derivation')
    a = np.asarray((ep.get('signals') or {}).get(name))[:n]
    if len(a) != n or a.size > semantic.MAX_SOURCE_VALUES or a.dtype.kind not in 'biuf':
        raise ValueError('regional source is absent or exceeds the existing source-value bound')
    entry = field_interpretation(ep['context'], name)
    if (entry.get('provenance') == 'human' and 'role' in entry
            and signals.signal_role(entry['role']) not in signals.TOUCH_QUANTITY_ROLES):
        raise ValueError('current human interpretation withdraws the regional sensor role')
    resolved = semantic._semantics(ep['context'], meta, entry, meta.get('shape') or list(a.shape[1:]))
    if resolved is not None and 'regions' in resolved:
        resolved['regions'] = resolved['regions'][:semantic.MAX_REGIONS]
    saved = derivation.get('semantics')
    if resolved != saved:
        raise ValueError('current source interpretation differs from the saved regional derivation')
    times = se._times(ep, n)
    result = dm.regional_response(a.reshape(n, -1), times, resolved)
    # A same-shaped different recording must not silently replace saved evidence.
    for audit in derivation.get('traces') or []:
        if 'raw_means' not in audit:
            continue
        columns = _columns(audit['source_columns'])
        match = next((t for t in result['traces'] if t['indices'] == columns), None)
        if match is None:
            match = next((t['sensitivity'] for t in result['traces']
                          if (t.get('sensitivity') or {}).get('indices') == columns), None)
        if match is None:
            raise ValueError('saved regional columns differ from the current source')
        for row, time, value in zip(audit['rows'], audit['times_s'], audit['raw_means']):
            if row >= n or times[row] != time or match['raw_means'][row] != value:
                raise ValueError('saved regional readings or clocks differ from the current source')
    return result


def _saved(derivation, ids):
    return next((t for t in derivation.get('traces') or []
                 if _columns(t.get('source_columns')) == ids and 'raw_means' in t and 'zero_counts' in t), None)


def _outputs(derivation, source, series):
    n = derivation['native_samples']
    for index, region in enumerate((derivation.get('semantics') or {}).get('regions') or []):
        # Regions absent from the original admitted derivation stay absent.
        ids = region['indices']
        allowed = (derivation.get('semantics') or {}).get('valid_indices')
        if allowed is not None:
            ids = [i for i in ids if i in allowed]
        main = _saved(derivation, ids)
        if main is None:
            continue
        if source is not None:
            fresh = next(t for t in source['traces'] if t['name'] == region['name'])
            times, rows, main = source['times_s'], list(range(n)), fresh
            sensitivity = fresh.get('sensitivity')
        else:
            times, rows = main['times_s'], main['rows']
            sensitivity = next((t for t in derivation['traces']
                if 'raw_means' in t and 'zero_counts' in t and _columns(t['source_columns']) != ids
                and set(_columns(t['source_columns'])).issubset(ids)
                and (series.get(t['series_id']) or {}).get('label') == region['name'] + ' zero-exclusion sensitivity'), None)
            if sensitivity is None and rows == list(range(n)) and not any(main['zero_counts']):
                sensitivity = main
        common = {'region': region['name'], 'index': index, 'times': times, 'rows': rows,
                  'columns': ids, 'unit': region.get('unit', main.get('unit')),
                  'counts': main.get('raw_valid_counts', main.get('valid_counts'))}
        yield {**common, 'kind': 'raw_mean', 'values': main['raw_means'],
               'operation': 'mean recorded raw regional codes, without reference subtraction'}
        yield {**common, 'kind': 'zero_cells', 'values': main['zero_counts'], 'unit': 'cells',
               'operation': 'count finite recorded regional cells equal to zero; zero remains a valid reading'}
        if sensitivity is None:
            continue
        sub = {**common, 'columns': sensitivity.get('indices', _columns(sensitivity.get('source_columns'))),
               'times': source['times_s'] if source is not None else sensitivity['times_s'],
               'rows': list(range(n)) if source is not None else sensitivity['rows'],
               'counts': sensitivity.get('raw_valid_counts', sensitivity.get('valid_counts'))}
        yield {**sub, 'kind': 'raw_never_zero_mean', 'values': sensitivity['raw_means'],
               'operation': 'mean recorded raw codes excluding cells that ever equal zero over the source period'}
        if 'signed_values' in sensitivity:
            yield {**sub, 'kind': 'signed_never_zero',
                   'counts': sensitivity.get('valid_counts'),
                   'values': [None if v is None else -v for v in sensitivity['signed_values']],
                   'operation': 'mean signed initial per-cell reference minus raw, excluding cells that ever equal zero; code differences, not calibrated force'}


def augment(doc, *, ep=None, max_rows=512, max_values=8192):
    """Return an augmented copy; this does not consume primary derivation trace slots.

    Saved audits retain their exact existing rows. Missing original outputs are
    recomputed only through the existing derivation function when owned ep data
    is supplied. All newly citable values share a separate bounded value budget.
    """
    if type(max_rows) is not int or type(max_values) is not int or max_rows < 2 or max_values < 2:
        raise ValueError('regional output budgets must admit at least two samples')
    max_rows, max_values = min(max_rows, MAX_ROWS), min(max_values, MAX_VALUES)
    result = copy.deepcopy(doc)
    old_ids = {r['id'] for r in result.get('series') or [] if r.get('regional_output')}
    result['series'] = [r for r in result.get('series') or [] if r['id'] not in old_ids]
    for sensor in result.get('sensors') or []:
        for summary in sensor.get('derivations') or []:
            summary['trace_ids'] = [i for i in summary.get('trace_ids') or [] if i not in old_ids]
    sensors = {s['id']: s for s in result.get('sensors') or []}
    queues = []
    series = {r['id']: r for r in result.get('series') or []}
    for derivation in result.get('derivations') or []:
        derivation['traces'] = [t for t in derivation.get('traces') or [] if t['series_id'] not in old_ids]
        sensor = sensors.get(derivation.get('sensor_id'))
        if derivation.get('kind') != 'regional_response' or sensor is None:
            continue
        source = None
        if ep is not None:
            try:
                source = _source(ep, sensor, derivation)
            except (ValueError, TypeError, IndexError) as error:
                _note(result, f'{sensor["id"]}: original regional outputs unavailable: {error}.')
                continue
        outputs = list(_outputs(derivation, source, series))
        if not any(row['kind'] == 'signed_never_zero' for row in outputs):
            _note(result, f'{sensor["id"]}: original signed zero-exclusion regional output unavailable in saved audit; no value was inferred.')
        queues.append([(sensor, derivation, output) for output in outputs])
    candidates = [queue[index] for index in range(max(map(len, queues), default=0))
                  for queue in queues if index < len(queue)]
    admitted = candidates[:min(MAX_OUTPUT_SERIES, max_values // 2)]
    quota = min(max_rows, max_values // max(1, len(admitted)))
    labels = {'raw_mean': 'raw regional mean', 'raw_never_zero_mean': 'raw never-zero regional mean',
              'zero_cells': 'zero-cell count', 'signed_never_zero': 'signed never-zero initial reference minus raw'}
    for sensor, derivation, output in admitted:
        ts = np.asarray(output['times'], dtype=float)
        values = np.asarray([np.nan if x is None else x for x in output['values']], dtype=float)
        times, kept_values = se._samples(ts, values, quota)
        if len(times) > quota:
            keep = np.linspace(0, len(times) - 1, quota, dtype=int)
            times, kept_values = [times[i] for i in keep], [kept_values[i] for i in keep]
        positions = np.searchsorted(ts, times)
        rows = [output['rows'][i] for i in positions]
        ident = derivation['id'] + ':regional-output:' + str(output['index']) + ':' + output['kind']
        coverage = {'native_samples': derivation['native_samples'], 'available_samples': len(ts),
                    'supplied_samples': len(times), 'omitted_samples': derivation['native_samples'] - len(times)}
        row = {'id': ident, 'sensor_id': sensor['id'], 'derivation_id': derivation['id'],
               'regional_output': output['kind'], 'label': output['region'] + ' ' + labels[output['kind']],
               'source_columns': _compact_columns(output['columns']), 'columns': [], 'source_rows': rows,
               'times': times, 'values': kept_values, 'operation': output['operation'],
               'unit': 'cells' if output['kind'] == 'zero_cells' else sensor.get('unit', output['unit']),
               'native_samples': derivation['native_samples'], 'sampled': bool(coverage['omitted_samples']),
               'coverage': coverage}
        if output['counts'] is not None:
            row['valid_counts'] = [output['counts'][i] for i in positions]
        result['series'].append(row)
        derivation['traces'].append({'series_id': ident, 'rows': rows, 'times_s': times,
            'values': kept_values, 'source_columns': row['source_columns'], 'operation': output['operation']})
        for summary in sensor.get('derivations') or []:
            if summary['id'] == derivation['id']:
                summary.setdefault('trace_ids', []).append(ident)
        if coverage['omitted_samples']:
            _note(result, 'Original regional output rows exceed the independent pass budget or saved audit coverage; only supplied exact rows may support citations.')
    if len(admitted) != len(candidates):
        _note(result, f'Original regional output series budget omitted {len(candidates) - len(admitted)} series; omitted values cannot support citations.')
    return result
