"""Quiet tactile warnings from full-rate readings and saved visual contact boundaries.

Warnings are review candidates, never validated dataset verdicts. An empty warning
list does not certify alignment when the video has no bracketed contact changes.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

MIN_CONTACTS = 3
MIN_GAP_S = 1.0
MIN_OFFSET_S = .15
MAX_BRACKET_S = .25


def _number(value):
    return isinstance(value, (int, float, np.number)) and not isinstance(value, bool) and np.isfinite(value)


def _allowed(name, meta):
    from label.signals import touch_cache_permission
    m = meta.get(name) or {}
    role = (m.get('dictionary') or {}).get('role')
    permission = touch_cache_permission(name, role, m.get('touch_role'))
    from label.sensor_evidence import _quantity
    return permission is not False and (_quantity(name, m.get('dictionary') or {}) is not None)


def _bracket(strip, index, first):
    if isinstance(index, bool) or not isinstance(index, int) or not isinstance(strip, list):
        return None
    if len(strip) < 2 or not all(_number(t) for t in strip) or np.any(np.diff(strip) <= 0):
        return None
    i = index - 1
    pair = (i - 1, i) if first else (i, i + 1)
    if not 0 <= pair[0] < pair[1] < len(strip):
        return None
    lo, hi = (float(strip[k]) for k in pair)
    return (lo, hi) if hi - lo <= MAX_BRACKET_S else None


def _boundaries(contacts, strips, meta):
    groups = defaultdict(list)
    used = set()
    for c in contacts:
        seen = c.get('seen') or {}
        if (not c.get('shown') or c.get('aligned_by') or c.get('review_status') == 'rejected'
                or seen.get('review_status') == 'rejected' or seen.get('touch_seen') != 'yes'
                or c.get('hand') not in ('left', 'right') or seen.get('hand') != c.get('hand')):
            continue
        names = tuple(sorted(n for n in c.get('signals') or [] if _allowed(n, meta)))
        if any((meta.get(n) or {}).get('aligned_by') for n in names):
            continue
        if not names or not all(_number(c.get(k)) for k in ('start_s', 'end_s')) or c['end_s'] < c['start_s']:
            continue
        group = (c['hand'], names)
        identity = (group, c['start_s'], c['end_s'])
        if identity in used:
            continue
        used.add(identity)
        st = strips.get(c.get('id')) or {}
        for first, side, field, timefield, censor in (
                (True, 'begin', 'first_touch_frame', 'start_s', 'from_start'),
                (False, 'end', 'last_touch_frame', 'end_s', 'to_end')):
            if c.get(censor):
                continue
            pair = _bracket(st.get(side), seen.get(field), first)
            if pair:
                groups[group].append({'id': c['id'], 'recorded_s': float(c[timefield]),
                                      'visual_bounds_s': list(pair),
                                      'offset_bounds_s': [float(x - c[timefield]) for x in pair]})
    return groups


def _runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return zip(edges[::2], edges[1::2])


def check(contacts, strips, signals, times, meta=None, fps=30):
    """Return versioned review warnings without changing labels or source readings."""
    meta = meta or {}
    groups = _boundaries(contacts or [], strips or {}, meta)
    warnings = []
    enough = [rows for rows in groups.values() if len({r['id'] for r in rows}) >= MIN_CONTACTS]
    timing_status = 'checked' if enough else 'insufficient_evidence'
    frame_s = 1 / fps if _number(fps) and fps > 0 else 1 / 30
    threshold = max(MIN_OFFSET_S, 4 * frame_s)
    for (hand, names), rows in groups.items():
        ids = list(dict.fromkeys(r['id'] for r in rows))
        if len(ids) < MIN_CONTACTS:
            continue
        lo = max(r['offset_bounds_s'][0] for r in rows)
        hi = min(r['offset_bounds_s'][1] for r in rows)
        # Every boundary must agree, including release. Opposite or dispersed lags stay quiet.
        if lo > hi or not (lo >= threshold or hi <= -threshold):
            continue
        offset = float(np.median([np.mean(r['offset_bounds_s']) for r in rows]))
        lead = 'after' if offset > 0 else 'before'
        warnings.append({'kind': 'timing_mismatch', 'headline': 'Possible tactile/video timing mismatch',
                         'start_s': min(r['recorded_s'] for r in rows),
                         'end_s': max(r['recorded_s'] for r in rows), 'hand': hand,
                         'signals': list(names), 'contact_ids': ids, 'offset_s': round(offset, 4),
                         'offset_bounds_s': [round(lo, 4), round(hi, 4)], 'boundaries': rows,
                         'detail': f'Video contact changes consistently appear about {abs(offset):.2f}s {lead} '
                                   f'the recorded signal across {len(ids)} contacts. Check the frames before changing timing.'})
    t = np.asarray(times, dtype=float)
    valid_clock = t.ndim == 1 and len(t) >= 2 and np.isfinite(t).all() and np.all(np.diff(t) > 0)
    readings_status = 'not_available'
    frozen_status = 'insufficient_evidence'
    if valid_clock:
        step = float(np.median(np.diff(t)))
        edge = np.r_[t, t[-1] + step]
        for name, stored in (signals or {}).items():
            if not _allowed(name, meta):
                continue
            a = np.asarray(stored)
            if a.ndim < 1 or not np.issubdtype(a.dtype, np.number):
                continue
            a = a.reshape(len(a), -1) if len(a) else np.empty((0, 1))
            # Reader-padded absent tails are missing, but zeros and dead cells are real codes.
            present = np.zeros(len(t), dtype=bool)
            count = min(len(t), len(a))
            present[:count] = np.isfinite(a[:count]).any(axis=1)
            readings_status = 'checked'
            for begin, end in _runs(~present):
                if edge[end] - edge[begin] < MIN_GAP_S:
                    continue
                warnings.append({'kind': 'missing_readings', 'headline': 'Tactile readings missing',
                                 'signals': [name], 'start_s': float(edge[begin]), 'end_s': float(edge[end]),
                                 'detail': f'{name} has no finite readings for {edge[end] - edge[begin]:.1f}s.'})
            events = [r for (hand, names), rows in groups.items() if name in names for r in rows]
            if len({r['id'] for r in events}) < MIN_CONTACTS or count < 2:
                continue
            frozen_status = 'checked'
            rate = (meta.get(name) or {}).get('rate_hz')
            min_hold = max(MIN_GAP_S, 8 / rate) if _number(rate) and rate > 0 else MIN_GAP_S
            same = np.isfinite(a[:count]).all(axis=1)
            same[1:] &= np.all(a[1:count] == a[:count - 1], axis=1)
            same[0] = False
            for begin, end in _runs(same):
                begin -= 1
                if edge[end] - edge[begin] < min_hold:
                    continue
                inside = [r for r in events if edge[begin] <= r['visual_bounds_s'][0]
                          and r['visual_bounds_s'][1] <= t[end - 1]]
                ids = list(dict.fromkeys(r['id'] for r in inside))
                if len(ids) < MIN_CONTACTS:
                    continue
                warnings.append({'kind': 'frozen_readings', 'headline': 'Tactile readings may be frozen',
                                 'signals': [name], 'contact_ids': ids, 'start_s': float(edge[begin]),
                                 'end_s': float(edge[end]), 'detail': f'{name} repeats exactly the same complete '
                                 f'reading through {len(ids)} visible contact changes. Inspect the recording.'})
    return {'version': 1, 'warnings': warnings,
            'checks': {'timing': timing_status, 'readings': readings_status, 'frozen': frozen_status},
            'policy': {'min_contacts': MIN_CONTACTS, 'min_offset_s': threshold, 'min_gap_s': MIN_GAP_S,
                       'max_bracket_s': MAX_BRACKET_S}, 'validated': False}


def for_episode(ep_dir: Path, contacts, strips):
    """Use canonical prepared clocks and effective dictionary roles, including human edits."""
    from label import episode as me
    from label.dictionary_context import field_interpretation
    # Avoid loading large unrelated pose arrays on episodes without tactile signals.
    import json
    ctx = json.loads((Path(ep_dir) / 'context.json').read_text())
    from label.dictionary_editor import for_episode as effective_dictionary
    owner = (ctx.get('piece') or {}).get('of') or (ctx.get('data_dictionary') or {}).get('episode_id')
    dictionary = effective_dictionary(Path(ep_dir), owner)
    if dictionary is not None:
        ctx['data_dictionary'] = dictionary
    declared = {s['name']: {**s, 'dictionary': field_interpretation(ctx, s['name'])}
                for s in ctx.get('signals') or []}
    if not any(_allowed(name, declared) for name in declared):
        return check([], {}, {}, [], declared)
    ep = me.load(ep_dir)
    meta = ep.get('signal_meta') or {}
    for name in ep.get('signals') or {}:
        entry = field_interpretation(ctx, name)
        if entry:
            meta.setdefault(name, {})['dictionary'] = entry
    n = len(ep['state'])
    times = np.array([me.frame_time(ep, i) for i in range(n)])
    return check(contacts, strips, ep.get('signals') or {}, times, meta, me.ep_fps(ep))
