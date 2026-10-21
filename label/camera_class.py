"""What a camera records: a scene, a tactile sensor's own image, or something no evidence decides.

One function (camera_class) makes the decision for every reader of it (prepare's view slots and anchor,
prepare/camera_slots.py; native tactile capture selection, sensor
evidence, scene pairing), from evidence the recording carries, in this order:

- An explicit content role (a person's dictionary edit, or recorded content_role, recorded_role or role metadata,
  label/dictionary_context.py camera_content_role) and a recorded sensor_type decide as stated.
- The field dictionary's role for the camera (label/dictionary_context.py camera_role, which keeps the reader's
  placement over a machine guess) decides tactile or not; a non-touch role keeps it a camera.
- With no role, the recorded name decides when it says touch (prepare/formats.py is_sensing, the reader's vocabulary
  for camera names), and otherwise the camera's recorded device: a camera whose topic or key shares a device namespace
  with recorded touch measurements (a /d360_0/image_raw topic beside /d360_0/pressure_topic) is that device's image
  when its pictures show a fixed view. The namespace is a device only when no other camera of the recording sits in
  it and none of its channels is the recording's state, joints or a command, so a robot or a dataset root never
  makes its cameras tactile.
- Whether a tactile image is an optical capture (a camera looking at its own contact surface, GelSight or DIGIT 360
  style) or a rendering of other readings (a heatmap) comes from a declared optical role or sensor type, a declared
  rendering in the camera's description, or the image statistics: an optical sensor's camera never moves relative to
  its gel, so its pictures hold a fixed view (fixed_view).

Image statistics never make a camera tactile on their own: a fixed scene camera also holds a fixed view. When nothing
decides, the camera stays a camera and the result says its class is unknown, so the request can say so.
"""
from __future__ import annotations

import numpy as np

# A view is fixed when, at a typical pixel, the frames vary by less than this share of the picture's own spatial
# contrast (the same share label/signals.py REST_FRACTION uses for a value at rest). Measured on the audited
# recordings: GelSight gel views 0.0, DIGIT 360 views 0.0 to 0.1, scene and wrist cameras 0.3 to 1.0, rendered
# tactile heatmaps above 0.9.
FIXED_SHARE = 0.1
# Fewer sampled frames than this cannot show whether a view moves.
MIN_FRAMES = 2
RENDER_WORDS = ('heatmap', 'heat map', 'thermal')


def fixed_view(images) -> dict | None:
    """{frames, temporal, spatial, fixed}: the median per-pixel standard deviation across up to label/lens.py
    MAX_FRAMES sampled frames (temporal), the median spatial standard deviation of one frame (spatial), and whether
    the temporal variation is under FIXED_SHARE of the spatial contrast. None with fewer than MIN_FRAMES frames or a
    picture with no spatial contrast (a blank frame shows no view at all)."""
    if not images:
        return None
    from label import lens
    keys = sorted(images)
    keys = [keys[i] for i in lens.spread(len(keys))]
    if len(keys) < MIN_FRAMES:
        return None
    thumbs = [lens.thumb(images[k]).astype(float) for k in keys]
    if len({t.shape for t in thumbs}) != 1:
        return None
    stack = np.stack(thumbs)
    spatial = float(np.median([t.std() for t in thumbs]))
    if not spatial > 0:
        return None
    temporal = float(np.median(stack.std(axis=0)))
    return {'frames': len(keys), 'temporal': round(temporal, 3), 'spatial': round(spatial, 3),
            'fixed': temporal <= FIXED_SHARE * spatial}


def _segments(key) -> list[str]:
    key = str(key or '')
    return [s for s in key.split('/') if s] if '/' in key else []


def touch_device(ctx: dict, view: str, meta: dict | None = None) -> dict | None:
    """{namespace, measurements}: the recorded device namespace this camera shares with touch measurements, or None.
    A measurement is touch by its effective dictionary role or, without one, its recorded name (label/sensor_evidence.py
    _quantity), and sits directly in the namespace (/d360_0/pressure_topic beside /d360_0/image_raw, not
    /robot/gripper/pressure beside /robot/camera). The namespace is the longest leading path both share, and counts
    only when no other camera of the recording (shown or retained) lies under it, and none of its channels is the
    recording's state, joints or a command (label/signals.py names_joints_or_state, prepare/formats.py COMMAND_WORDS):
    a namespace holding those is a robot or a dataset, not one sensor device."""
    from label.dictionary_context import field_interpretation
    from label.sensor_evidence import _quantity
    from label.signals import names_joints_or_state
    from prepare.formats import COMMAND_WORDS, _names_word
    meta = meta if meta is not None else (ctx.get('cameras') or {}).get(view) or {}
    own = _segments(meta.get('key') or meta.get('name'))
    if len(own) < 2:
        return None
    others = [_segments(m.get('key') or m.get('name')) for v, m in (ctx.get('cameras') or {}).items() if v != view]
    others += [_segments(m.get('name')) for m in ctx.get('unshown_cameras') or [] if isinstance(m, dict)]
    found = {}
    for signal in ctx.get('signals') or []:
        name = signal.get('name')
        theirs = _segments(name)
        depth = 0
        for a, b in zip(own[:-1], theirs[:-1]):
            if a != b:
                break
            depth += 1
        if not depth or len(theirs) != depth + 1 or _quantity(name, field_interpretation(ctx, name)) is None:
            continue
        namespace = tuple(own[:depth])
        if any(tuple(other[:depth]) == namespace for other in others if len(other) > depth):
            continue
        members = [str(s.get('name')) for s in ctx.get('signals') or []
                   if tuple(_segments(s.get('name'))[:depth]) == namespace]
        if any(names_joints_or_state(m) or _names_word(m, COMMAND_WORDS + ('action', 'actions')) for m in members):
            continue
        found.setdefault(namespace, []).append(name)
    if not found:
        return None
    namespace = max(found, key=len)
    return {'namespace': '/' + '/'.join(namespace), 'measurements': sorted(found[namespace])[:8],
            'measurement_count': len(found[namespace])}


def camera_class(ctx: dict, view: str, images=None, meta: dict | None = None) -> dict:
    """{class, tactile, optical, known, basis, evidence}. class is optical_tactile, tactile_image or camera; known is
    False when no evidence decided it (the camera is then kept a camera). images, the view's sampled frames, let the
    image statistics decide what only they can (fixed_view)."""
    from label import signals as sg
    from label.dictionary_context import (camera_content_role, camera_role, prepared_interpretation,
                                          TACTILE_CONTENT_ROLES)
    from prepare.formats import is_sensing
    meta = meta if meta is not None else (ctx.get('cameras') or {}).get(view) or {}
    name = str(meta.get('name') or view)
    entry = prepared_interpretation(ctx, view, 'camera') if view in (ctx.get('cameras') or {}) else {}
    evidence = {}

    def result(kind, basis, known=True, rendered=False):
        return {'class': kind, 'tactile': kind != 'camera', 'optical': kind == 'optical_tactile',
                'rendered': rendered, 'known': known, 'basis': basis, 'evidence': evidence}

    content = camera_content_role(ctx, view) if view in (ctx.get('cameras') or {}) else None
    if content and content['role'] in ('scene', 'other'):
        return result('camera', f'{content["provenance"]} content role {content["role"]}')
    sensor_type = sg._role(meta.get('sensor_type'))
    declared = [sg._role(meta.get(key)) for key in ('content_role', 'recorded_role', 'role')]
    role = camera_role(ctx, view, entry, meta)
    stats = fixed_view(images)
    if stats is not None:
        evidence['view'] = stats
    if role is None and entry.get('role'):
        evidence['dictionary_role_dropped'] = entry['role']
    if bool(content and content['role'] == 'tactile') or sensor_type == 'optical_tactile':
        tactile, basis = True, (f'{content["provenance"]} content role tactile' if content
                                else 'recorded sensor_type optical_tactile')
    elif role in TACTILE_CONTENT_ROLES:
        tactile, basis = True, f'dictionary role {role}'
    elif role:
        return result('camera', f'dictionary role {role}')
    elif is_sensing(name):
        tactile, basis = True, 'recorded name says touch'
    else:
        device = touch_device(ctx, view, meta)
        if device is None:
            return result('camera', 'no recorded role, touch name or touch device', known=False)
        evidence['device'] = device
        if stats is None or not stats['fixed']:
            return result('camera', f'shares device {device["namespace"]} with touch measurements, but '
                          + ('no sampled frames show whether its view is fixed' if stats is None
                             else 'its view is not fixed'), known=False)
        tactile, basis = True, f'image of device {device["namespace"]}, which records touch'
    if not tactile:
        return result('camera', basis)
    description = ' '.join(str(value or '') for value in
                           (entry.get('meaning'), meta.get('description'), name, meta.get('sensor_type'))).lower()
    if any(word in description for word in RENDER_WORDS):
        return result('tactile_image', basis + '; described as a rendering', rendered=True)
    if sensor_type == 'optical_tactile' or role == 'optical_tactile' or 'optical_tactile' in declared:
        return result('optical_tactile', basis + '; declared optical')
    if stats is not None and stats['fixed']:
        return result('optical_tactile', basis + '; its frames hold a fixed view, as a camera inside an optical '
                                                 'tactile sensor does')
    return result('tactile_image', basis + ('; no sampled frames to tell an optical capture from a rendering'
                                            if stats is None else '; its view changes, as a rendering of readings does'))
