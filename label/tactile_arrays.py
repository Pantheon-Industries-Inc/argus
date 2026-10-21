"""Whole-array tactile comparisons in source coordinates, without guessed anatomy or polarity."""
from __future__ import annotations

import base64
import copy
import io
import math

import numpy as np

MAX_ARRAYS = 4
MAX_VALUES = 8192
MAX_FRAMES = 32
MAX_SCALE_VALUES = 16_000_000
MAX_NATIVE_VALUES = 65536
MAX_NATIVE_ROWS = 4096
# A snapshot's selection_priority says why it was chosen: 0 an endpoint, 1 a
# reference-change maximum, SPACED a row spaced along the native clock. Only SPACED
# rows may be dropped to fit a prompt. Priority 2 once meant a spaced RGB instant;
# selection stopped depending on RGB sampling, so nothing is given 2 any more.
SPACED = 3
# Every tactile change image uses one colour convention, so a colour means the same
# sign on every sensor in a request. Captions and legends quote these words.
SIGNED_CHANGE_KEY = 'Orange is negative, dark is zero, blue is positive.'
SIGNED_CHANGE_LEGEND = 'Orange: lower codes | Blue: higher codes'


def signed_change_color(q):
    """RGB for a change scaled to -1..1: orange below the reference, dark at it, blue above."""
    q = float(np.clip(q, -1, 1))
    return (int(70 + 185*max(-q, 0)), int(90 + 80*abs(q)), int(110 + 145*max(q, 0)))


def comparison(a, shape, times, ks, value_budget, semantics=None, native_value_budget=None):
    """Keep complete snapshots, including extrema outside the ordinary RGB samples."""
    if (len(shape) != 2 or any(type(d) is not int or d <= 0 for d in shape)
            or math.prod(shape) != a.shape[1] or max(shape) > 32 or a.shape[1] > 1024):
        return None
    limit = min(MAX_FRAMES, len(a), value_budget // a.shape[1] - 1)
    if limit < min(3, len(a)):
        return None
    initial = np.asarray(a[:min(3, len(a))], dtype=float)
    # Incomplete initial cells have no reference; zero is a reading, not a validity mask.
    valid_reference = np.isfinite(initial).all(axis=0)
    reference = np.median(np.where(np.isfinite(initial), initial, 0), axis=0)
    reference[~valid_reference] = np.nan
    peak = np.zeros(len(a))
    mean = np.zeros(len(a))
    raw_low, raw_high, delta_high = np.inf, -np.inf, 0.0
    for start in range(0, len(a), 128):
        block = np.asarray(a[start:start + 128], dtype=float)
        valid = np.isfinite(block)
        if valid.any():
            raw_low = min(raw_low, float(block[valid].min()))
            raw_high = max(raw_high, float(block[valid].max()))
        delta = np.abs(block - reference)
        finite = np.isfinite(delta)
        values = np.where(finite, delta, 0)
        peak[start:start + len(block)] = values.max(axis=1)
        mean[start:start + len(block)] = values.sum(axis=1) / np.maximum(finite.sum(axis=1), 1)
        delta_high = max(delta_high, float(values.max()))
    if not math.isfinite(raw_low):
        return None
    selected = {0, len(a) - 1}
    priority = {0: 0, len(a) - 1: 0}
    # Preserve local excursions as well as large distributed changes. These select
    # source rows only; they are not claims of contact, force or failure.
    for score in (mean, peak):
        if len(selected) < limit:
            k = int(np.argmax(score))
            selected.add(k)
            priority.setdefault(k, 1)
    # Reserve native-clock coverage before RGB instants. Local responses may
    # cancel regional averages and need not coincide with an array extremum.
    for target in np.linspace(times[0], times[-1], max(2, limit - 2)):
        k = int(np.argmin(np.abs(times - target)))
        if len(selected) < limit:
            selected.add(k)
            priority.setdefault(k, SPACED)
    while len(selected) < limit:
        available = [k for k in range(len(a)) if k not in selected]
        if not available:
            break
        k = max(available, key=lambda k: min(abs(times[k] - times[j]) for j in selected))
        selected.add(k)
        priority[k] = SPACED
    def values(row):
        return [float(v) if np.isfinite(v) else None for v in row]
    from prepare.sensor_semantics import validate
    declared = validate(semantics, shape)
    shown_columns = (declared or {}).get('valid_indices', list(range(a.shape[1])))
    result = {'shape': list(shape), 'coordinates': 'Source array row and column indices; physical layout is not established.',
            'reference': {'times_s': [float(t) for t in times[:len(initial)]], 'values': values(reference),
                          'method': 'Per-cell median of the first up to three samples, requiring all to be finite. An initial reference, not a verified unloaded baseline.'},
            'raw_range': [raw_low, raw_high], 'signed_change_range': [-delta_high, delta_high],
            'selection': 'Endpoints, whole-array and single-cell reference-change maxima, and spaced native-clock snapshots. These do not depend on scene RGB sampling. Unshown transients remain possible.',
            'native_samples': len(a),
            'frames': [{'time_s': float(times[k]), 'values': values(a[k]),
                        'valid_cells': int(np.isfinite(a[k, shown_columns]).sum()),
                        'signed_change_valid_cells': int(np.isfinite(a[k, shown_columns] - reference[shown_columns]).sum()),
                        'selection_priority': priority[k],
                        'raw_quantiles': values(np.nanquantile(np.where(np.isfinite(a[k, shown_columns]), a[k, shown_columns], np.nan), [.1, .5, .9]))
                            if np.isfinite(a[k, shown_columns]).any() else [None] * 3,
                        'signed_change_quantiles': values(np.nanquantile(np.where(
                            np.isfinite(a[k, shown_columns] - reference[shown_columns]),
                            a[k, shown_columns] - reference[shown_columns], np.nan), [.1, .5, .9]))
                            if np.isfinite(a[k, shown_columns] - reference[shown_columns]).any() else [None] * 3}
                       for k in sorted(selected)]}
    if declared is not None and 'valid_indices' in declared:
        result['valid_indices'] = declared['valid_indices']
    if declared and declared.get('response_direction'):
        result['response_convention'] = {
            'direction': declared['response_direction'],
            'authority': (declared.get('provenance') or {}).get('authority')}
    if declared and declared.get('response_direction_conflict'):
        result['response_direction_conflict'] = copy.deepcopy(declared['response_direction_conflict'])
    native_budget = MAX_NATIVE_VALUES if native_value_budget is None else native_value_budget
    count = min(len(a), MAX_NATIVE_ROWS, native_budget // len(shown_columns)) if shown_columns else 0
    required_rows = [0, len(a) - 1, *[row for row in sorted(selected) if priority[row] == 1]]
    source_rows = native_rows(times, count, required_rows)
    result['native_constituents'] = {
        'source_columns': shown_columns, 'source_rows': source_rows,
        'preserved_rows': sorted(set(required_rows) & set(source_rows)),
        'times_s': [float(times[row]) for row in source_rows],
        'values': [[(value.item() if isinstance(value, np.generic) else value)
                    if np.isfinite(value) else None for value in a[row, shown_columns]] for row in source_rows],
        'native_samples': len(a), 'omitted_rows': len(a) - len(source_rows),
        'sampled': len(source_rows) < len(a),
        'selection': ('Every native source row.' if len(source_rows) == len(a) else
                      'Spaced source-clock rows under the native row/value budget; omitted readings cannot support findings.')}
    layout = _physical_layout(semantics, shape)
    if layout:
        result['physical_layout'] = layout
        result['physical_scale'] = _physical_scale(a, reference, layout, delta_high)
        result['coordinates'] = 'Declared physical coordinates with flattened source cell IDs; anatomical names retain their source authority.'
    return result


def native_rows(times, count, required=()):
    """Select actual clock rows, retaining endpoints and supplied extrema."""
    if count >= len(times):
        return list(range(len(times)))
    if not count:
        return []
    keep = set(list(dict.fromkeys(required))[:count])
    targets = np.linspace(times[0], times[-1], count)
    right = np.minimum(np.searchsorted(times, targets), len(times) - 1)
    left = np.maximum(0, right - 1)
    selected = keep | set(np.where(np.abs(times[left] - targets) <= np.abs(times[right] - targets), left, right).tolist())
    while len(selected) > count:
        ordered = sorted(selected)
        removable = [i for i in range(1, len(ordered) - 1) if ordered[i] not in keep]
        if not removable:
            removable = [i for i, row in enumerate(ordered) if row not in keep]
        remove = min(removable, key=lambda i: times[ordered[min(i + 1, len(ordered) - 1)]] - times[ordered[max(0, i - 1)]])
        selected.remove(ordered[remove])
    return sorted(selected)


def _physical_layout(semantics, shape):
    from prepare.sensor_semantics import validate
    descriptor = validate(semantics, shape)
    if not descriptor or 'positions' not in descriptor:
        return None
    authority = (descriptor.get('provenance') or {}).get('authority')
    if authority not in ('recorded', 'publisher_candidate', 'recording_verified'):
        return None
    positions = descriptor['positions']
    if len(positions[0]) != 2:
        return None
    ids = descriptor.get('valid_indices', list(range(math.prod(shape))))
    if len(ids) < 2 or len({tuple(positions[i]) for i in ids}) != len(ids):
        return None
    if any(max(positions[i][axis] for i in ids) == min(positions[i][axis] for i in ids) for axis in (0, 1)):
        return None
    regions = {i: group['name'] for group in descriptor.get('regions') or [] for i in group['indices']}
    return {'authority': authority, 'limitations': descriptor.get('limitations') or [],
            'positions': [{'index': i, 'x': positions[i][0], 'y': positions[i][1],
                           **({'region': regions[i]} if i in regions else {})} for i in ids]}


def _physical_scale(a, reference, layout, fallback):
    """Keep one full-period valid-cell display scale without changing readings."""
    ids = [p['index'] for p in layout['positions']]
    result = {'maximum': fallback or 1., 'valid_cells': len(ids),
              'method': 'Full-array maximum absolute change; exact valid-cell percentile exceeds the 16000000 source-value budget.'}
    if a.size > MAX_SCALE_VALUES:
        return result
    # The transposed column selection is contiguous; subtract and take absolute
    # values in place before the percentile partitions this bounded buffer.
    delta = np.asarray(a[:, ids].T, dtype=float)
    delta -= reference[ids, None]
    np.abs(delta, out=delta)
    delta[~np.isfinite(delta)] = np.nan
    finite = int(np.isfinite(delta).sum())
    if not finite:
        return {**result, 'maximum': 1., 'finite_values': 0, 'method': 'No finite valid-cell reference changes; fixed unit display scale.'}
    maximum = float(np.nanpercentile(delta.reshape(-1), 99.5, overwrite_input=True))
    # nanpercentile may compact missing values in its scratch buffer. Count
    # clipped source values independently instead of reading that buffer again.
    clipped = sum(int(np.count_nonzero(np.abs(
        np.asarray(a[start:start + 128, ids], dtype=float) - reference[ids]) > maximum))
        for start in range(0, len(a), 128))
    return {'maximum': maximum or 1., 'valid_cells': len(ids), 'finite_values': finite,
            'clipped_values': clipped,
            'percentile': 99.5,
            'method': '99.5th percentile of absolute raw-minus-reference changes over the complete source period and declared valid cells; display colors clipped beyond this fixed scale.'}


def _render_sheet(sensor, snapshots):
    """Render raw and signed changes on scales shared across every supplied instant."""
    from PIL import Image, ImageDraw, ImageFont
    from label.frames import GRID_FONT
    array = sensor['array_evidence']
    rows, cols = array['shape']
    width, height = 210, 294
    per_row = min(4, len(snapshots))
    image = Image.new('RGB', (per_row * width, math.ceil(len(snapshots) / per_row) * height), '#101923')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(GRID_FONT), 13)
    raw_low, raw_high = array['raw_range']
    delta_max = array['signed_change_range'][1]
    reference = np.array([np.nan if v is None else v for v in array['reference']['values']])
    for i, frame in enumerate(snapshots):
        x, y = (i % per_row) * width, (i // per_row) * height
        draw.text((x + 8, y + 3), f"{frame['time_s']:.4f} s", fill='white', font=font)
        raw = np.array([np.nan if v is None else v for v in frame['values']])
        for offset, label, values in ((24, 'Raw', raw), (157, 'Raw - initial reference', raw - reference)):
            draw.text((x + 8, y + offset), label, fill='white', font=font)
            for cell, value in enumerate(values):
                r, c = divmod(cell, cols)
                if not math.isfinite(value):
                    color = (105, 105, 105)
                elif offset == 24:
                    q = float(np.clip((value - raw_low) / (raw_high - raw_low or 1), 0, 1))
                    color = (int(25 + 230*q), int(35 + 180*q), int(55 + 50*q))
                else:
                    color = signed_change_color(value / (delta_max or 1))
                left, top = x + 24 + c * 160 / cols, y + offset + 18 + r * 104 / rows
                draw.rectangle((left, top, left + 160/cols, top + 104/rows), fill=color)
            draw.text((x + 8, y + offset + 17), '0', fill='white', font=font)
            draw.text((x + 8, y + offset + 108), str(rows - 1), fill='white', font=font)
            draw.text((x + 24, y + offset + 122), f'columns 0 to {cols-1}', fill='white', font=font)
    stream = io.BytesIO()
    image.save(stream, format='JPEG', quality=88)
    raw = stream.getvalue()
    caption = (f"TACTILE ARRAY SNAPSHOTS | {sensor['id']} | raw scale {raw_low:g} to {raw_high:g}; "
               f"signed change scale {-delta_max:g} to {delta_max:g}. "
               'All frames use the same scales. Raw is dark to yellow. For signed change, ' + SIGNED_CHANGE_KEY[0].lower() + SIGNED_CHANGE_KEY[1:] + ' '
               'Grey means missing. Axes are source array indices, not hand anatomy. '
               + _timing_caption(sensor) + _response_caption(sensor) + ' '
               'The numeric summary gives 10th, 50th and 90th percentiles across finite cells. Read individual spatial patterns from these images; cite sensor_id without series_id for these full-array snapshots.')
    return [{'type': 'text', 'text': caption}, {'type': 'image_url', 'image_url': {
        'url': 'data:image/jpeg;base64,' + base64.b64encode(raw).decode('ascii'), 'detail': 'high'}}]


def render(sensor):
    """Use bounded sheets so native-clock spatial comparisons remain readable."""
    array = sensor['array_evidence']
    parts = []
    for start in range(0, len(array['frames']), 8):
        frames = array['frames'][start:start + 8]
        parts.extend(_render_physical(sensor, frames) if array.get('physical_layout')
                     else _render_sheet(sensor, frames))
    return parts


def _timing_caption(sensor):
    """What the times printed on a sensor's snapshots are, by the one timing rule (prepare/signal_alignment.py
    video_timing): nothing for a sensor on a recorded clock the cameras share."""
    from prepare.signal_alignment import video_timing
    timing = video_timing(sensor.get('timing'))
    if timing['timed'] and sensor.get('episode_alignment_verified') is not False:
        return ''
    if not timing['timed']:
        return ('The times on these snapshots place a recording that shares no clock with the cameras from both '
                'starts, so they say nothing about the frames at those times. ')
    return 'The times on these snapshots are placements without verified alignment to the video. '


def _response_caption(sensor):
    convention = sensor['array_evidence'].get('response_convention') or {}
    if sensor['array_evidence'].get('response_direction_conflict') or sensor.get('response_direction_conflict'):
        return ('Response polarity declarations conflict; code changes do not establish stronger or weaker sensor response. '
                'Raw change is not a physical force calibration.')
    direction = convention.get('direction', sensor.get('response_direction'))
    if direction not in ('up', 'down'):
        return ('Response polarity is undeclared; code changes do not establish stronger or weaker sensor response. '
                'Raw change is not a physical force calibration.')
    word = 'Lower' if direction == 'down' else 'Higher'
    candidate = convention.get('authority') == 'publisher_candidate'
    return (("Publisher candidate response convention; the recording hardware match is unverified. " if candidate else
             "Declared response convention. ") + word + ' raw values indicate stronger sensor response under that convention. '
            'This is not a physical force calibration.')


def _render_physical(sensor, snapshots):
    from PIL import Image, ImageDraw, ImageFont
    from label.frames import GRID_FONT
    array = sensor['array_evidence']
    layout = array['physical_layout']
    positions = layout['positions']
    width, height, columns = 320, 720, min(4, len(snapshots))
    image = Image.new('RGB', (columns * width, math.ceil(len(snapshots) / columns) * height), '#101923')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(GRID_FONT), 12)
    small = ImageFont.truetype(str(GRID_FONT), 10)
    lo = np.min([[p['x'], p['y']] for p in positions], axis=0)
    hi = np.max([[p['x'], p['y']] for p in positions], axis=0)
    scale = min(260 / (hi[0] - lo[0]), 260 / (hi[1] - lo[1]))
    coordinates = {p['index']: (30 + (p['x'] - lo[0]) * scale, 44 + (p['y'] - lo[1]) * scale) for p in positions}
    distinct = np.array(list(coordinates.values()))
    spacing = np.linalg.norm(distinct[:, None] - distinct[None], axis=2)
    np.fill_diagonal(spacing, np.inf)
    radius = max(2., min(4., float(np.median(spacing.min(axis=1))) * .42))
    reference = array['reference']['values']
    color_scale = array.get('physical_scale') or {'maximum': array['signed_change_range'][1] or 1, 'method': 'Full-array maximum absolute change.'}
    maximum = color_scale['maximum']
    regions = sorted({p['region'] for p in positions if p.get('region')})
    raw_low, raw_high = array['raw_range']
    for i, frame in enumerate(snapshots):
        ox, base_y = (i % columns) * width, (i // columns) * height
        for offset, kind in ((0, 'Raw'), (360, 'Raw - reference')):
            oy = base_y + offset
            draw.text((ox + 8, oy + 4), f"{frame['time_s']:.4f} s | {kind}", fill='white', font=font)
            for region in regions:
                group = [coordinates[p['index']] for p in positions if p.get('region') == region]
                x0, y0 = np.min(group, axis=0)
                x1, y1 = np.max(group, axis=0)
                draw.rectangle((ox + x0 - 6, oy + y0 - 6, ox + x1 + 6, oy + y1 + 6), outline='#536272')
                draw.text((ox + x0 - 4, oy + y0 - 19), region[:32], fill='#c0cbd7', font=small)
            for p in positions:
                cell = p['index']
                raw, ref = frame['values'][cell], reference[cell]
                if raw is None:
                    color = (105, 105, 105)
                elif offset == 0:
                    q = float(np.clip((raw - raw_low) / (raw_high - raw_low or 1), 0, 1))
                    color = (int(25 + 230*q), int(35 + 180*q), int(55 + 50*q))
                elif ref is None:
                    color = (110, 85, 125)
                else:
                    color = signed_change_color((raw - ref) / maximum)
                x, y = coordinates[cell]
                draw.ellipse((ox + x - radius, oy + y - radius, ox + x + radius, oy + y + radius),
                             fill=color, outline='white' if raw == 0 else None)
            legend = 'Dark to yellow: lower to higher raw codes' if offset == 0 else SIGNED_CHANGE_LEGEND
            draw.text((ox + 8, oy + 334), legend, fill='white', font=small)
            draw.text((ox + 8, oy + 348), layout['authority'], fill='#b7c6d5', font=small)
    stream = io.BytesIO()
    image.save(stream, format='JPEG', quality=90)
    caption = (f"TACTILE ARRAY SNAPSHOTS | {sensor['id']} | Declared physical layout, authority {layout['authority']}. "
               + _timing_caption(sensor) +
               f"Raw scale {raw_low:g} to {raw_high:g}, dark to yellow. Fixed raw-minus-reference color scale {-maximum:g} to {maximum:g} for every instant. "
               f"{color_scale['method']} Underlying raw values and full signed ranges are unchanged. "
               + SIGNED_CHANGE_KEY + ' White rings mark raw zero. Grey means missing raw; purple means valid raw with a missing reference, so no change can be computed. '
               'Source cell IDs and physical coordinates are retained in physical_layout.positions. '
               'Region outlines group declared cells, not measured contact areas. '
               'Regions: ' + ', '.join(regions) + '. ' + ' '.join(layout['limitations']) + ' '
               'A publisher_candidate layout is conditional, not a verified recording hardware match. '
               + _response_caption(sensor) + ' '
               'Cite sensor_id and exact supplied snapshot times without series_id.')
    return [{'type': 'text', 'text': caption}, {'type': 'image_url', 'image_url': {
        'url': 'data:image/jpeg;base64,' + base64.b64encode(stream.getvalue()).decode('ascii'), 'detail': 'high'}}]
