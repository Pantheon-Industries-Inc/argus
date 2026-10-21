"""Bounded derived traces from explicit regions and recorded point-set coordinates."""
from __future__ import annotations

import copy
import math

import numpy as np

from label import derived_measurements as dm
from label.dictionary_context import field_interpretation
from prepare.sensor_semantics import validate
from prepare.signal_alignment import video_timing

MAX_SOURCE_VALUES = 16_000_000  # At most 128 MB when a real-valued source is converted to float64.
MAX_DERIVED_SERIES = 24
MAX_REGIONS = 8
MAX_SAMPLES = 48
MAX_NATIVE_ROWS = 512
MAX_NATIVE_VALUES = 8192
POINT_ROLES = frozenset(('landmarks', 'hand_pose', 'pose'))


def _semantics(ctx, meta, entry, shape):
    declared = validate(meta.get('sensor_semantics'), shape)
    semantics = copy.deepcopy(declared or {'version': 1})
    human = entry.get('provenance') == 'human'
    layout = entry.get('layout') if human else None
    point_layout = entry.get('role') in POINT_ROLES or semantics.get('chains') or semantics.get('normalization_pair')
    if layout and not point_layout:
        regions = []
        references = 0
        for group in layout:
            if not isinstance(group, dict) or type(group.get('start')) is not int or type(group.get('count')) is not int:
                return None
            references += group['count']
            if (group['start'] < 0 or group['count'] <= 0 or group['start'] + group['count'] > math.prod(shape)
                    or references > 8192):
                return None
            regions.append({'name': group.get('name'),
                            'indices': list(range(group['start'], group['start'] + group['count']))})
        semantics['regions'] = regions
        if (semantics.get('provenance') or {}).get('authority') == 'publisher_candidate':
            semantics.pop('valid_indices', None)
            semantics.pop('positions', None)
        semantics['layout_authority'] = 'human dictionary layout'
    provenance = semantics.get('provenance') or {}
    if human and provenance.get('authority') == 'publisher_candidate' and (layout or entry.get('meaning')):
        fields = [field for field in (ctx.get('data_dictionary') or {}).get('fields') or []
                  if field.get('name') == meta['name'] and field.get('kind') == 'signal']
        machine = ((ctx.get('data_dictionary') or {}).get('machine_entries') or {}).get(fields[0]['id']) if len(fields) == 1 else None
        if layout or not isinstance(machine, dict) or entry.get('meaning') != machine.get('meaning'):
            semantics.pop('chains', None)
            semantics.pop('normalization_pair', None)
            semantics['candidate_point_layout_withheld'] = 'The dataset publisher changed the candidate point semantics.'
        changed_meaning = bool(entry.get('meaning')) and (not isinstance(machine, dict) or entry['meaning'] != machine.get('meaning'))
        if changed_meaning:
            for key in ('valid_indices', 'positions', 'response_direction'):
                semantics.pop(key, None)
            if not layout:
                semantics.pop('regions', None)
            semantics['candidate_regional_layout_withheld'] = 'Publisher candidate regional semantics and response direction were withdrawn by the dataset publisher field interpretation.'
            semantics.setdefault('limitations', []).append(semantics['candidate_regional_layout_withheld'])
    for key in ('unit', 'units', 'response_direction'):
        if meta.get(key) is not None and (key not in semantics or provenance.get('authority') == 'publisher_candidate'):
            semantics[key] = copy.deepcopy(meta[key])
    if (provenance.get('authority') != 'publisher_candidate'
            and semantics.get('response_direction') in ('up', 'down')
            and meta.get('response_direction') in ('up', 'down')
            and semantics['response_direction'] != meta['response_direction']):
        semantics['response_direction_conflict'] = {
            'sensor_semantics': semantics.pop('response_direction'), 'metadata': meta['response_direction']}
        semantics.setdefault('limitations', []).append(
            'Conflicting declared response directions; stronger-response direction is unresolved.')
    if entry.get('role') in POINT_ROLES and len(shape) == 2:
        units = semantics.get('units')
        if isinstance(units, list) and len(units) == shape[-1]:
            semantics['units'] = units * shape[0]
    return validate(semantics, shape)


def _feature(label, values, columns, operation, unit=None, counts=None, source_rows=None, **audit):
    return {'label': label, 'values': values, 'source_columns': columns, 'operation': operation,
            'unit': unit, 'counts': counts, 'source_rows': source_rows, 'audit': audit}


def _bounded_feature(feature, times, native_quota=MAX_SAMPLES):
    from label import sensor_evidence as se
    source_rows = feature.get('source_rows')
    feature_times = np.asarray(times) if source_rows is None else np.asarray(times)[source_rows]
    complete = source_rows is None and len(feature_times) <= min(MAX_NATIVE_ROWS, native_quota)
    limit = len(feature_times) if complete else max(2, min(MAX_SAMPLES, native_quota))
    ts, values = se._samples(feature_times,
        np.asarray([np.nan if value is None else value for value in feature['values']]), limit)
    positions = np.searchsorted(feature_times, ts).tolist()
    rows = positions if source_rows is None else [source_rows[i] for i in positions]
    counts = feature['counts'] or [0] * len(feature_times)
    columns = feature['source_columns']
    if len(columns) > 256 and columns == list(range(columns[0], columns[0] + len(columns))):
        columns = {'start': columns[0], 'count': len(columns)}
    plot = se.native_plot(feature_times, feature['values']) if source_rows is None else None
    return {**feature, 'values': values, 'times': ts, 'rows': rows, 'source_columns': columns,
            **({'plot': plot} if plot is not None else {}),
            'coverage': {'scope': 'all native derivation rows' if source_rows is None else 'selected endpoint comparisons', 'valid_count_min': min(counts),
                         'valid_count_max': max(counts), 'valid_samples': sum(value is not None for value in feature['values']),
                         'native_samples': len(feature_times), 'supplied_samples': len(ts),
                         'omitted_samples': len(feature_times) - len(ts)},
            'counts': [counts[i] for i in positions],
            'audit': {key: [items[i] for i in positions] for key, items in feature['audit'].items()}}


def _regional(a, times, semantics):
    semantics = copy.deepcopy(semantics)
    semantics['regions'] = semantics['regions'][:MAX_REGIONS]
    result = dm.regional_response(a, times, semantics)
    primary, extra = [], []
    for trace in result['traces']:
        common = dict(raw_means=trace['raw_means'], zero_counts=trace['zero_counts'],
                      signed_values=trace['signed_values'], positive_values=trace['positive_values'],
                      negative_values=trace['negative_values'])
        primary.append(_feature(trace['name'] + ' relative response', trace['values'], trace['indices'],
                                trace['operation'], trace['unit'], trace['valid_counts'], **common))
        sensitivity = trace.get('sensitivity')
        if sensitivity and sensitivity['indices'] != trace['indices']:
            extra.append(_feature(trace['name'] + ' zero-exclusion sensitivity', sensitivity['values'],
                                  sensitivity['indices'], sensitivity['operation'], trace['unit'],
                                  sensitivity['valid_counts'], raw_means=sensitivity['raw_means'],
                                  zero_counts=sensitivity['zero_counts']))
        if result['response_direction'] is not None:
            extra.append(_feature(trace['name'] + ' signed raw minus initial reference',
                                  trace['signed_values'], trace['indices'],
                                  'mean signed raw minus initial reference', trace['unit'], trace['valid_counts']))
        else:
            for direction in ('positive', 'negative'):
                extra.append(_feature(trace['name'] + ' ' + direction + ' change', trace[direction + '_values'],
                                      trace['indices'], 'mean ' + direction + ' per-cell change from initial reference',
                                      trace['unit'], trace['valid_counts']))
    result.pop('traces')
    return result, primary + extra


def _pointset(a, times, semantics):
    semantics = copy.deepcopy(semantics)
    semantics['chains'] = (semantics.get('chains') or [])[:MAX_REGIONS]
    result = dm.pointset_shape(a, times, semantics)
    columns = list(range(math.prod(a.shape[1:])))
    features = [_feature('Point-set shape from first frame', result.pop('first_rms'), columns,
                         result['operation'], result['unit'], result['pair_valid_counts']['first']),
                _feature('Point-set shape from previous frame', result.pop('adjacent_rms'), columns,
                         result['operation'] + '; directly compares adjacent source rows', result['unit'],
                         result['pair_valid_counts']['adjacent'])]
    for chain in result.pop('curls'):
        source_columns = [point * a.shape[2] + axis for point in chain['indices'] for axis in range(a.shape[2])]
        features.append(_feature(chain['name'] + ' curl', chain['values'], source_columns,
                                 chain['operation'], chain['unit'], chain['valid_counts']))
    # The native-adjacent trace cannot compare widely separated later poses.
    # Preserve bounded extrema across the available shape/curl traces, then
    # compare those selected endpoint poses directly on the same initial scale.
    from label import sensor_evidence as se
    selected = {0, len(times) - 1}
    quota = max(4, (MAX_SAMPLES - 2) // len(features))
    for feature in features:
        sampled, _ = se._samples(np.asarray(times),
            np.asarray([np.nan if value is None else value for value in feature['values']]), quota)
        selected.update(np.searchsorted(times, sampled).tolist())
    # Other modalities can change while the estimated pose is quiet. Fill the
    # largest temporal gaps rather than supplying only pose extrema.
    while len(selected) < min(MAX_SAMPLES, len(times)):
        ordered = sorted(selected)
        gaps = [(start, stop) for start, stop in zip(ordered, ordered[1:]) if stop - start > 1]
        if not gaps:
            break
        start, stop = max(gaps, key=lambda pair: times[pair[1]] - times[pair[0]])
        midpoint = (times[start] + times[stop]) / 2
        row = min(stop - 1, max(start + 1, int(np.searchsorted(times, midpoint))))
        selected.add(row)
    source_rows = sorted(selected)
    selected_times = np.asarray(times)[source_rows]
    comparison = dm.pointset_shape(a[source_rows], selected_times, semantics)
    features.insert(2, _feature('Point-set shape between selected poses', comparison['adjacent_rms'], columns,
        result['operation'] + '; directly compares the stated selected endpoint poses, not distances from the first pose',
        result['unit'], comparison['pair_valid_counts']['adjacent'], source_rows=source_rows,
        comparison_from_s=[None, *selected_times[:-1].tolist()]))
    result['limitations'].append('Selected-pose comparisons measure only their stated endpoints; intervening missing or repeated estimates do not establish continuous stability.')
    result.pop('pair_valid_counts')
    result.pop('repeated_previous')
    return result, features


def append(ep, pl, doc, times):
    """Add shared sampled values before raw-cell traces, without changing native source identities."""
    from label import sensor_evidence as se, signals as sg
    from label.episode import anchor
    ctx, n = ep['context'], int(pl['n'])
    sensors = {sensor['id']: sensor for sensor in doc['sensors']}
    candidates = []
    for meta in ctx.get('signals') or []:
        name = meta['name']
        entry = field_interpretation(ctx, name)
        role = sg.signal_role(entry.get('role'))
        if entry.get('provenance') == 'human' and 'role' in entry and role not in (
                *sg.TOUCH_QUANTITY_ROLES, *POINT_ROLES):
            continue
        value = (ep.get('signals') or {}).get(name)
        if value is None or meta.get('summary_of'):
            continue
        a = np.asarray(value)[:n]
        shape = meta.get('shape') or list(a.shape[1:])
        if a.size > MAX_SOURCE_VALUES:
            doc['limitations'].append(f'{name}: regional and point-set derivations exceed the source-value budget; available raw evidence is retained.')
            continue
        if (a.dtype.kind not in 'biuf' or len(a) != n
                or not shape or validate({'version': 1}, shape) is None
                or math.prod(shape) != math.prod(a.shape[1:])):
            continue
        semantics = _semantics(ctx, meta, entry, shape)
        if semantics is None:
            doc['limitations'].append(f'{name}: invalid measurement semantics; derived comparisons withheld.')
            continue
        doc['limitations'].extend(note for note in semantics.get('limitations') or []
                                 if isinstance(note, str) and note not in doc['limitations'])
        pointset = len(shape) == 2 and shape[-1] in (2, 3) and (
            role in POINT_ROLES or (semantics and (semantics.get('chains') or semantics.get('normalization_pair'))))
        sensor_id = 'signal:' + name
        sensor = sensors.get(sensor_id)
        if not pointset and (not sensor or not semantics or not semantics.get('regions')):
            continue
        if len(candidates) >= MAX_DERIVED_SERIES:
            doc['limitations'].append(f'{name}: derived sensor budget reached; native evidence remains available.')
            continue
        if sensor is None:
            if len(doc['sensors']) >= se.MAX_SENSORS:
                doc['limitations'].append(f'{name}: no descriptor budget for point-set comparisons.')
                continue
            timing = (ep.get('signal_meta') or {}).get(name, meta)
            sensor = {'id': sensor_id, 'name': name, 'kind': 'numeric', 'quantity': 'landmark_shape',
                      'descriptor_digest': se._descriptor_digest(meta), 'shape': shape,
                      'capabilities': ['response_change'], 'interpretation': entry,
                      'source': str(meta.get('source') or name), 'columns': math.prod(shape), 'supplied_columns': 0,
                      'timing': {key: timing[key] for key in ('aligned_by', 'camera_aligned_by', 'clock_problem', 'rate_hz')
                                 if timing.get(key) is not None},
                      'episode_alignment_verified': video_timing(timing, ctx, anchor(ep))['verified']}
            sensors[sensor_id] = sensor
            doc['sensors'].append(sensor)
            doc['coverage'].append({'id': sensor_id, 'name': name, 'status': 'sampled'})
        try:
            result, features = (_pointset(a.reshape(n, *shape), times, semantics or {'version': 1}) if pointset
                                else _regional(a.reshape(n, -1), times, semantics))
        except (ValueError, np.linalg.LinAlgError) as error:
            doc['limitations'].append(f'{name}: derived comparison unavailable: {error}')
            continue
        result.pop('times_s')
        result['limitations'].extend(note for note in (semantics or {}).get('limitations') or []
                                     if isinstance(note, str) and note not in result['limitations'])
        if len(result.get('stale_intervals') or []) > 16:
            result['stale_intervals_total'] = len(result['stale_intervals'])
            result['stale_intervals'] = result['stale_intervals'][:16]
        result['id'] = sensor_id + ':derivation:' + result['kind']
        result['sensor_id'] = sensor_id
        result['native_samples'] = n
        result['traces'] = []
        result['source'] = sensor_id
        summary = {key: copy.deepcopy(result[key]) for key in ('id', 'kind', 'operation', 'provenance', 'limitations')}
        summary['response_direction'] = (semantics or {}).get('response_direction')
        summary['reference'] = ({key: copy.deepcopy(result['reference'][key]) for key in ('times_s', 'method')}
                                if 'reference' in result else copy.deepcopy(result['normalization']))
        summary['coverage'] = copy.deepcopy(result.get('coverage') or {'frames': n})
        if 'stale_intervals' in result:
            summary['stale_intervals'] = copy.deepcopy(result['stale_intervals'])
            summary['stale_intervals_total'] = result.get('stale_intervals_total', len(result['stale_intervals']))
        summary['layout_authority'] = (semantics or {}).get('layout_authority')
        summary['semantic_limits'] = {key: copy.deepcopy(semantics[key]) for key in (
            'hardware_revision_verified', 'force_calibration_verified', 'anatomical_claims_conditional',
            'response_convention_note', 'candidate_point_layout_withheld', 'candidate_regional_layout_withheld') if key in (semantics or {})}
        summary['trace_ids'] = []
        if len((semantics or {}).get('regions') or []) > MAX_REGIONS:
            summary['limitations'].append('Only the first bounded declared regions are derived; other raw readings remain available.')
        candidates.append((sensor, result, summary, features))
    emitted = []
    feature_count = min(MAX_DERIVED_SERIES, sum(len(item[3]) for item in candidates))
    native_quota = MAX_NATIVE_VALUES // max(1, feature_count)
    positions = [0] * len(candidates)
    while len(emitted) < MAX_DERIVED_SERIES:
        progress = False
        for index, (sensor, result, summary, features) in enumerate(candidates):
            position = positions[index]
            if position >= len(features) or len(emitted) >= MAX_DERIVED_SERIES:
                continue
            progress = True
            feature = _bounded_feature(features[position], times, native_quota)
            positions[index] += 1
            if feature.get('source_rows') is None and feature['coverage']['omitted_samples']:
                note = 'Native derived trajectories exceed the shared row/value budget; only retained exact source samples may support findings.'
                if note not in summary['limitations']:
                    summary['limitations'].append(note)
                    result['limitations'].append(note)
            ts, values, rows = feature['times'], feature['values'], feature['rows']
            trace_id = result['id'] + ':' + str(position)
            row = {'id': trace_id, 'sensor_id': sensor['id'], 'derivation_id': result['id'],
                   'label': feature['label'], 'source_columns': feature['source_columns'], 'columns': [],
                   'unit': feature['unit'], 'times': ts, 'values': values, 'operation': feature['operation'],
                   'coverage': feature['coverage'], 'valid_counts': feature['counts'],
                   'native_samples': n, 'sampled': len(ts) < n,
                   'sample_period_s': float(np.median(np.diff(times))) if n > 1 else None}
            if feature.get('plot'):
                row['plot'] = feature['plot']
            if 'comparison_from_s' in feature['audit']:
                row['comparison_from_s'] = feature['audit']['comparison_from_s']
                row['source_rows'] = rows
            emitted.append(row)
            result['traces'].append({'series_id': trace_id, 'rows': rows, 'times_s': ts, 'values': values,
                'source_columns': feature['source_columns'], 'operation': feature['operation'],
                'valid_counts': feature['counts'],
                **({'source_rows': rows} if 'comparison_from_s' in feature['audit'] else {}), **feature['audit']})
            summary['trace_ids'].append(trace_id)
        if not progress:
            break
    for index, (sensor, result, summary, features) in enumerate(candidates):
        if summary['trace_ids']:
            if positions[index] < len(features):
                summary['withheld_trace_count'] = len(features) - positions[index]
                summary['limitations'].append('Derived trace budget reached; omitted comparisons cannot support findings.')
            sensor.setdefault('derivations', []).append(summary)
            doc.setdefault('derivations', []).append(result)
            doc['limitations'].extend(note for note in result['limitations'] if note not in doc['limitations'])
    doc['series'] = emitted + doc['series']
