"""Verify saved final requests from their actual ordered text and image receipts."""
from __future__ import annotations

import re
from pathlib import Path

IMPLEMENTATION_SETTINGS = frozenset({'request_semantics_sha256', 'reader_projection_sha256',
                                      'jaw_algorithm_sha256'})


def only_implementation_changed(saved: dict, current: dict) -> bool:
    from label.harness import same_input_identity
    if not isinstance(saved, dict) or saved.get('version') != 2 or current.get('version') != 2:
        return False
    if not all(isinstance(identity.get('settings'), dict) for identity in (saved, current)):
        return False
    def without_code(identity):
        return {**identity, 'settings': {key: value for key, value in identity['settings'].items()
                                        if key not in IMPLEMENTATION_SETTINGS}}
    return same_input_identity(without_code(saved), without_code(current))


def matches(record: dict, current: dict, ep_dir: Path, out_dir: Path, *, example_dir=None) -> bool:
    """Replay proven selections locally; no provider, routing, or cache mutation."""
    from label import harness, grids, evidence_access as ea, sensor_analysis as sa
    saved = record.get('input_identity')
    if not only_implementation_changed(saved, current):
        return False
    proof = (record.get('config') or {}).get('request_images') or {}
    recipe = proof.get('recipe')
    if (proof.get('version') != 1 or proof.get('coverage') != 'one_request'
            or not isinstance(proof.get('fingerprint'), str)
            or not re.fullmatch('[0-9a-f]{64}', proof['fingerprint']) or not isinstance(recipe, dict)
            or not harness.same_input_identity(recipe.get('input_identity'), saved)):
        return False
    try:
        settings = current['settings']
        policy = settings['annotation_policy']
        if recipe.get('model') != settings['model'] or recipe.get('reasoning') != settings['reasoning']:
            return False
        route = (record.get('config') or {}).get('resolution_route') or {}
        route_width = None
        if not settings['cell_w'] and route.get('routed'):
            rig = harness.me.rig(harness.me.load(ep_dir))
            low, high = harness.me.ROUTE_WIDTHS[rig]
            # the width routing chose, as it recorded it (an episode with no task text takes the narrow one with
            # fine_detail unset); only an older receipt without it is derived from fine_detail
            recorded = route.get('cell_w')
            route_width = recorded if type(recorded) is int and recorded > 0 else (
                low if route.get('fine_detail') is False else high)
        request = harness.me.build_request(ep_dir, detail=harness.DETAIL, gate=harness.DECODE_GATE,
            grid_cols=harness.GRID_COLS, cell_w=settings['cell_w'] or None, max_cell_w=route_width,
            example_dir=example_dir, inspect_evidence=True, model=settings['model'], reasoning=settings['reasoning'],
            sample_fps=policy.get('sample_fps'), task_mode=policy.get('task_mode'), dataset_note=policy.get('dataset_note'),
            inspection=harness.policy.inspection_mode(policy),
            derived_dir=Path(out_dir) / '.derived' / 'jaws', read_only=True)
        for key, field in (('cell', 'cell'), ('grid_cols', 'grid_cols'), ('timesteps_s', 'timesteps'),
                           ('views', 'views'), ('cam_labels', 'cam_labels')):
            if recipe.get(key) != request[field]:
                return False
        content = request['content']
        access = request.get('evidence_access')
        if access is not None:
            doc, extra = ea.replay_selection(access, record.get('evidence_inspection'))
            content, _, _ = ea.final_content(content, request['image_bytes'], access, doc, extra,
                image_limit=harness.MAX_IMAGES, image_bytes_limit=harness.IMAGE_LIMIT_BYTES,
                image_inflation=harness.IMAGE_SIZE_INFLATION)
        analysis = sa.build(content, request['sensor_evidence'], config={
            **(record.get('config') or {}), 'model': settings['model'], 'reasoning': settings['reasoning'],
            'max_tokens': settings['max_tokens'], 'input_identity': saved},
            ep=access.ep if access is not None else None)
        receipt = record.get('sensor_analysis')
        if analysis is not None:
            if not isinstance(receipt, dict) or receipt.get('digest') != analysis['digest']:
                return False
            sa.bind(analysis, receipt)
            content = sa.local_content(content)
        elif receipt is not None:
            return False
        rebuilt = grids.receipt(content, {}, reference_recipe=recipe)
        return rebuilt['fingerprint'] == proof['fingerprint']
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, RuntimeError):
        return False
