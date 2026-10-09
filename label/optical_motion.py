"""Image-space dot movement from tactile frames already supplied to the model.

No force conversion, unloaded reference assumption or removal of coherent motion.
Tracking declines when the first image lacks a distributed, repeated dot pattern.
"""
from __future__ import annotations

import numpy as np
from PIL import ImageFilter

ANALYSIS_WIDTH = 384
MIN_MARKERS = 24
MAX_COMPARISONS = 16


def _image(im):
    from label.frames import Shrunk, downscaled
    if isinstance(im, Shrunk):
        widths = [w for w in im.by_width if w <= ANALYSIS_WIDTH]
        im = im.at(max(widths) if widths else min(im.by_width))
    return downscaled(im, ANALYSIS_WIDTH)


def _contrast(im):
    gray = im.convert('L')
    return (np.asarray(gray.filter(ImageFilter.BoxBlur(max(2, im.width // 100))), dtype=float)
            - np.asarray(gray, dtype=float))


def _dots(contrast):
    h, w = contrast.shape
    mask = contrast > 12
    margin = max(4, int(min(h, w) * .05))
    mask[:margin] = mask[-margin:] = False
    mask[:, :margin] = mask[:, -margin:] = False
    todo = set(map(tuple, np.argwhere(mask)))
    points, areas = [], []
    while todo:
        seed = todo.pop(); component = [seed]; stack = [seed]
        while stack:
            y, x = stack.pop()
            for q in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if q in todo:
                    todo.remove(q); stack.append(q); component.append(q)
        if not 2 <= len(component) <= max(20, w*h*.0015):
            continue
        ys, xs = np.asarray(component).T
        if not .45 < (np.ptp(xs)+1)/(np.ptp(ys)+1) < 2.2:
            continue
        weights = contrast[ys, xs]
        points.append([float(np.average(xs, weights=weights)), float(np.average(ys, weights=weights))])
        areas.append(len(component))
    p = np.asarray(points)
    if len(p) < MIN_MARKERS or len(p) > 512:
        return None
    distance = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    np.fill_diagonal(distance, np.inf)
    nearest = np.min(distance, axis=1)
    spacing = float(np.median(nearest))
    q25, q75 = np.percentile(nearest, [25, 75])
    if (spacing < 4 or spacing > min(h, w)*.2 or q75-q25 > spacing*.6
            or np.ptp(p[:, 0]) < w*.45 or np.ptp(p[:, 1]) < h*.45
            or np.percentile(areas, 90) > np.median(areas)*3):
        return None
    occupied = {(min(2, int(x*3/w)), min(2, int(y*3/h))) for x, y in p}
    return (p, spacing) if len(occupied) >= 6 else None


def _positions(contrast, reference, radius):
    offsets = np.arange(-radius, radius+1)
    dx, dy = np.meshgrid(offsets, offsets)
    x, y = np.rint(reference).astype(int).T
    xx, yy = x[:, None, None]+dx, y[:, None, None]+dy
    inside = ((xx >= 0) & (xx < contrast.shape[1]) & (yy >= 0) & (yy < contrast.shape[0])).all(axis=(1, 2))
    patch = contrast[np.clip(yy, 0, contrast.shape[0]-1), np.clip(xx, 0, contrast.shape[1]-1)]
    positions, strength = [], []
    for threshold in (8, 16):
        weights = np.maximum(patch-threshold, 0)
        total = weights.sum(axis=(1, 2)); strength.append(total)
        positions.append(np.column_stack([(weights*xx).sum(axis=(1, 2))/np.maximum(total, 1),
                                          (weights*yy).sum(axis=(1, 2))/np.maximum(total, 1)]))
    agreement = np.linalg.norm(positions[0]-positions[1], axis=1)
    valid = (inside & (strength[0] > 20) & (strength[1] > 8) & (agreement < .35)
             & (np.linalg.norm(positions[0]-reference, axis=1) < radius*.9))
    return positions[0], valid, agreement


def _measurement(reference, a, b, valid, agreement):
    n = int(valid.sum())
    row = {'matched': n, 'tracked_fraction': round(n/len(reference), 3)}
    if n < MIN_MARKERS or n/len(reference) < .6:
        return {**row, 'p95_px': None}
    delta = b[valid]-a[valid]
    design = np.column_stack([reference[valid]-np.mean(reference[valid], axis=0), np.ones(n)])
    residual = delta-design@np.linalg.lstsq(design, delta, rcond=None)[0]
    row.update({'mean_dx_px': round(float(np.mean(delta[:, 0])), 4),
                'mean_dy_px': round(float(np.mean(delta[:, 1])), 4),
                'p95_px': round(float(np.percentile(np.linalg.norm(delta, axis=1), 95)), 4),
                'local_p95_px': round(float(np.percentile(np.linalg.norm(residual, axis=1), 95)), 4),
                'threshold_agreement_p95_px': round(float(np.percentile(agreement[valid], 95)), 4)})
    return row


def measure(frames, times):
    """Evidence at supplied instants, with explicit quality gaps and image coordinates."""
    ks = sorted(k for k in frames if k in times)
    if len(ks) < 2:
        return None
    first = _image(frames[ks[0]])
    if min(first.size) < 64:
        return None
    detected = _dots(_contrast(first))
    if detected is None:
        return None
    reference, spacing = detected
    radius = max(2, min(8, int(spacing*.32)))
    baseline, base_valid, base_agreement = _positions(_contrast(first), reference, radius)
    samples, pairs = [], []
    previous = None
    for k in ks:
        im = _image(frames[k])
        if im.size == first.size:
            position, valid, agreement = _positions(_contrast(im), reference, radius)
        else:
            position = baseline; valid = np.zeros(len(reference), dtype=bool); agreement = base_agreement
        samples.append({'time_s': times[k], **_measurement(reference, baseline, position, base_valid & valid,
                                                         np.maximum(base_agreement, agreement))})
        if previous:
            pk, pa, pv, pagreement = previous
            pairs.append({'from_s': times[pk], 'to_s': times[k],
                          **_measurement(reference, pa, position, pv & valid,
                                         np.maximum(pagreement, agreement))})
        previous = k, position, valid, agreement
    if not any(p['p95_px'] is not None for p in pairs):
        return None
    # Keep large changes and quiet comparisons within a fixed payload budget.
    chosen = set(np.rint(np.linspace(0, len(pairs)-1, min(8, len(pairs)))).astype(int))
    chosen.update(sorted(range(len(pairs)), key=lambda i: pairs[i].get('p95_px') or 0,
                         reverse=True)[:MAX_COMPARISONS-len(chosen)])
    selected = [pairs[i] for i in sorted(chosen)]
    used_times = {times[ks[0]], *(p['from_s'] for p in selected), *(p['to_s'] for p in selected)}
    return {'version': 1, 'analysis_size': list(first.size), 'reference_s': times[ks[0]],
            'reference_markers': len(reference), 'samples': [r for r in samples if r['time_s'] in used_times],
            'pairs': selected, 'comparisons_total': len(pairs),
            'method': 'Contrast-weighted dot centers at two thresholds. Pairs compare the same valid markers. Coherent movement is retained; local_p95_px separately removes a fitted affine field.',
            'coordinates': 'Analysis-image pixels, x right and y down. No world direction, pressure or force calibration. The first supplied frame is a reference, not a certified unloaded pad.',
            'coverage': 'Only supplied image instants. Large movement, lost dots and ambiguous centers produce quality gaps. Sparse comparisons cannot establish continuous slip or complete transient coverage.'}
