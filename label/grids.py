"""Hash actual image inputs; rebuild base frame grids only when every byte matches."""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import math
import re
from pathlib import Path

from label import frames as mf
from label.atomic import write_atomic, write_bytes_atomic


class GridError(RuntimeError):
    pass


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def fingerprint(value) -> str:
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode())


def receipt(content: list, record: dict, *, reference_recipe: dict | None = None) -> dict:
    """An ordered final-image receipt, without storing images or prompt prose."""
    from PIL import Image
    images, grids, parts = [], [], []
    cfg = record.get('config') or {}
    caption = ''
    for part in content:
        if part.get('type') == 'text':
            caption = part.get('text') or ''
            parts.append({'type': 'text', 'sha256': digest(caption.encode())})
        elif part.get('type') == 'image_url':
            url = part['image_url']['url']
            if not url.startswith('data:image/') or ';base64,' not in url:
                raise GridError('image input is not an embedded image')
            raw = base64.b64decode(url.split(',', 1)[1], validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                width, height = image.size
            entry = {'sha256': digest(raw), 'bytes': len(raw), 'width': width, 'height': height,
                     'detail': part['image_url'].get('detail'), 'index': len(images)}
            match = re.match(r'^=== grid t=([0-9.eE+-]+)-([0-9.eE+-]+)s \|', caption)
            if match:
                entry = {**entry, 'kind': 'frame_grid', 't0_s': float(match[1]), 't1_s': float(match[2])}
                ts, cols = cfg.get('timesteps_s') or [], cfg.get('grid_cols') or 4
                first = len(grids) * cols
                if first < len(ts):
                    entry.update(t0_s=ts[first], t1_s=ts[min(len(ts), first + cols) - 1])
                grids.append(copy.deepcopy(entry))
            else:
                entry['kind'] = 'supplemental'
            images.append(entry)
            parts.append({'type': 'image_url', 'sha256': entry['sha256'], 'detail': entry['detail']})
            caption = ''
    identity = record.get('input_identity') or {}
    cfg = record.get('config') or {}
    recipe = {'model': record.get('model'), 'reasoning': record.get('reasoning_effort'),
              'annotation_policy': copy.deepcopy(record.get('annotation_policy')),
              'cell': cfg.get('cell'), 'grid_cols': cfg.get('grid_cols'), 'grid_quality': 80,
              'detail': cfg.get('detail'), 'timesteps_s': cfg.get('timesteps_s'),
              'views': cfg.get('views'), 'cam_labels': cfg.get('cam_labels'),
              'input_identity': copy.deepcopy(identity),
              'renderer_sha256': digest(Path(mf.__file__).read_bytes() + mf.GRID_FONT.read_bytes())}
    if reference_recipe is not None:
        recipe = copy.deepcopy(reference_recipe)
    proof = fingerprint({'parts': parts, 'recipe': recipe})
    return {'version': 1, 'fingerprint': proof, 'images': images, 'grid_images': grids,
            'supplemental_images': len(images) - len(grids), 'dispatch_status': 'intended',
            'coverage': 'one_request', 'recipe': recipe}


def public_receipt(value: dict | None) -> dict | None:
    if not isinstance(value, dict):
        return None
    return {k: copy.deepcopy(value[k]) for k in
            ('version', 'fingerprint', 'images', 'grid_images', 'supplemental_images',
             'dispatch_status', 'coverage', 'parts', 'unavailable_parts') if k in value}


def joined_receipt(parts: list) -> dict | None:
    """Keep all known request inputs and explicitly identify parts without receipts."""
    if not parts:
        return None
    images, grids, scope = [], [], []
    for p in parts:
        r = (p.get('config') or {}).get('request_images')
        available = isinstance(r, dict) and bool(r.get('fingerprint')) and bool(r.get('grid_images'))
        status = r.get('dispatch_status', 'unknown') if available else (
            'not_dispatched' if p.get('final_dispatch_outcome') == 'not dispatched' else 'unknown')
        owner = {'part': p.get('part', p.get('index')), 't0_s': p['t0_s'], 't1_s': p['t1_s'],
                 'receipt_available': available, 'dispatch_status': status}
        if available:
            owner['fingerprint'] = r['fingerprint']
            for src, target in ((r.get('images') or [], images), (r['grid_images'], grids)):
                for image in src:
                    row = {**image, 'part': owner['part'], 'dispatch_status': status}
                    if 't0_s' in image:
                        row.update(local_t0_s=image['t0_s'], local_t1_s=image['t1_s'],
                                   t0_s=round(image['t0_s'] + float(p['t0_s']), 3),
                                   t1_s=round(image['t1_s'] + float(p['t0_s']), 3))
                    target.append(row)
        scope.append(owner)
    return {'version': 1, 'fingerprint': fingerprint([{k: v for k, v in p.items() if k != 'dispatch_status'}
            for p in scope]), 'images': images, 'grid_images': grids,
            'supplemental_images': len(images) - len(grids), 'coverage': 'available_part_receipts',
            'parts': scope, 'unavailable_parts': [p for p in scope if not p['receipt_available']]}


def _same_grid_inputs(saved: dict, current: dict) -> bool:
    """Keep source/settings proof; final JPEG hashes verify historical image identity."""
    from label import harness
    identities = []
    for identity in (saved, current):
        if not isinstance(identity, dict) or not isinstance(identity.get('settings'), dict):
            return False
        # Annotation prose can change without changing any frame-grid bytes.
        identities.append({**identity, 'settings': {key: value for key, value in identity['settings'].items()
                           if key != 'request_semantics_sha256'}})
    return harness.same_input_identity(*identities)


def _one(record: dict, ep_dir: Path) -> list[dict]:
    from label import episode as me, harness, policy
    saved = (record.get('config') or {}).get('request_images') or {}
    if not isinstance(saved, dict) or not isinstance(saved.get('recipe'), dict):
        raise GridError('this label has no verified frame-grid input receipt')
    recipe = saved['recipe']
    expected = saved.get('grid_images') or []
    if not isinstance(expected, list) or not all(isinstance(row, dict) for row in expected):
        raise GridError('the saved frame-grid receipt is invalid')
    identity = recipe.get('input_identity')
    selected = recipe.get('annotation_policy')
    if (not expected or not isinstance(identity, dict) or not identity.get('settings')
            or not isinstance(identity.get('settings'), dict) or not isinstance(selected, dict)):
        raise GridError('this label has no verified frame-grid input receipt')
    period = selected.get('sample_every_s')
    if type(period) not in (int, float) or not math.isfinite(period) or period <= 0:
        raise GridError('the saved request sampling policy is invalid')
    if recipe.get('renderer_sha256') != digest(Path(mf.__file__).read_bytes() + mf.GRID_FONT.read_bytes()):
        raise GridError('the frame-grid renderer differs from the recorded request')
    settings = identity['settings']
    examples = record.get('example_dir')
    if settings.get('examples') and not examples:
        raise GridError('the original example inputs are unavailable')
    try:
        current = harness.input_identity(ep_dir, model=recipe['model'], reasoning=recipe['reasoning'],
            max_tokens=settings['max_tokens'], cell_w=settings['cell_w'], example_dir=examples,
            sample_fps=selected.get('sample_fps'), task_mode=selected.get('task_mode'),
            dataset_note=selected.get('dataset_note'),
            inspection=policy.inspection_mode(selected))
        if not _same_grid_inputs(identity, current):
            raise GridError('the source, dictionary or request settings differ from the recorded request')
        req = me.build_request(ep_dir, cell_w=recipe['cell'][0], grid_cols=recipe['grid_cols'],
            grid_quality=recipe['grid_quality'], detail=recipe['detail'], example_dir=examples,
            model=recipe['model'], reasoning=recipe['reasoning'], sample_fps=1 / selected['sample_every_s'],
            task_mode=selected.get('task_mode'), dataset_note=selected.get('dataset_note'), inspect_evidence=False)
    except GridError:
        raise
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        raise GridError('the original frame-grid inputs cannot be verified') from error
    if (req['cell'] != recipe['cell'] or req['grid_cols'] != recipe['grid_cols']
            or req['timesteps'] != recipe['timesteps_s'] or req['views'] != recipe['views']
            or req['cam_labels'] != recipe['cam_labels']):
        raise GridError('the original camera layout or sampling instants differ')
    images = mf.grid_jpegs(req['content'])
    if [digest(raw) for raw in images] != [row['sha256'] for row in expected]:
        raise GridError('the rebuilt frame grids are not the image bytes shown to the model')
    return [{**row, 'jpeg': raw} for row, raw in zip(expected, images)]


def rebuild(record: dict, ep_dir: Path | None = None) -> list[dict]:
    cfg = record.get('config') or {}
    if cfg.get('pieces'):
        out = []
        for p in cfg['pieces']:
            local = {'config': p.get('config') or {}}
            receipt = (p.get('config') or {}).get('request_images') or {}
            if not receipt.get('fingerprint') or not receipt.get('grid_images'):
                continue
            local['example_dir'] = p.get('example_dir')
            if not p.get('episode_dir'):
                raise GridError('a proven request part has no original episode input')
            for row in _one(local, Path(p['episode_dir'])):
                out.append({**row, 'part': p.get('part', p.get('index')),
                            'local_t0_s': row['t0_s'], 'local_t1_s': row['t1_s'],
                            'dispatch_status': receipt.get('dispatch_status', 'unknown'),
                            't0_s': round(row['t0_s'] + float(p['t0_s']), 3),
                            't1_s': round(row['t1_s'] + float(p['t0_s']), 3)})
        if not out:
            raise GridError('no part has a verified frame-grid input receipt')
        return out
    folder = ep_dir or record.get('episode_dir')
    if not folder:
        # Path('') is the working directory, which would read whatever episode happens to be there
        raise GridError('the record names no episode folder')
    return _one(record, Path(folder))


def write(out_dir: Path, record: dict, grids: list[dict]) -> dict:
    from PIL import Image
    saved = (record.get('config') or {}).get('request_images') or record.get('request_images')
    if not saved or not grids:
        raise GridError('no verified frame grids to write')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for i, g in enumerate(grids, 1):
        raw = g['jpeg']
        with Image.open(io.BytesIO(raw)) as image:
            width, height = image.size
        name = f'grid_{i:02d}.jpg'
        write_bytes_atomic(out_dir / name, raw)
        entries.append({**{k: g[k] for k in ('t0_s', 't1_s', 'local_t0_s', 'local_t1_s', 'part', 'dispatch_status') if k in g},
                        'file': name, 'sha256': digest(raw), 'bytes': len(raw), 'width': width, 'height': height})
    if [g['sha256'] for g in entries] != [g['sha256'] for g in saved.get('grid_images') or []]:
        raise GridError('frame grids do not match the label receipt')
    idx = {'version': 1, 'check': 'sha256', 'request_images': public_receipt(saved), 'grids': entries}
    write_atomic(out_dir / 'grids.json', idx)
    return idx
