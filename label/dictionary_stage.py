"""Shared interpretation stage and spend reservation for local and upload review."""
from __future__ import annotations

import json
import math
from pathlib import Path

DICTIONARY_MAX_USD = 3.0
UNKNOWN_LABEL_RESERVE_USD = 1e300


def dictionary_spend(job: Path) -> dict:
    receipt = job / 'dictionary.json'
    status = job / 'dictionary_status.json'
    if not receipt.exists() and not (job / 'dictionary.claim').exists():
        return {'cost_usd': 0.0, 'reserved_usd': 0.0, 'complete': True}
    try:
        record = json.loads(receipt.read_text())
        cost = record.get('cost_usd')
        if isinstance(cost, (int, float)) and not isinstance(cost, bool) and math.isfinite(cost) and cost >= 0:
            return {'cost_usd': float(cost), 'reserved_usd': float(cost), 'complete': True}
        if record.get('attempted') is False and record.get('status') != 'pending':
            return {'cost_usd': 0.0, 'reserved_usd': 0.0, 'complete': True}
    except (OSError, ValueError, AttributeError):
        pass
    try:
        bound = json.loads(status.read_text()).get('projected_upper_usd', DICTIONARY_MAX_USD)
        if not isinstance(bound, (float, int)) or isinstance(bound, bool) or not math.isfinite(bound) or bound <= 0:
            bound = DICTIONARY_MAX_USD
    except (OSError, ValueError, AttributeError):
        bound = DICTIONARY_MAX_USD
    return {'cost_usd': None, 'reserved_usd': float(bound), 'complete': False}


def label_spend(job: Path) -> float:
    return sum(cost for cost, _ in _label_receipts(job))


def label_spend_complete(job: Path) -> bool:
    return all(complete for _, complete in _label_receipts(job))


def _label_receipts(job: Path):
    from label.harness import episode_cost
    from label.evidence_access import recover_selection_reservations
    out = job / 'run' / 'out'
    if not out.exists():
        return
    names = {p.name.removeprefix('failed_').removeprefix('noreply_') for p in out.glob('*episode_*.json')}
    cache_dir = out / '.evidence'
    names.update(p.name.removesuffix('.inspection.json') for p in cache_dir.glob('episode_*.json.inspection.json'))
    for name in names:
        try:
            value, complete = 0.0, True
            primary_inspection = None
            p = next((p for p in (out / name, out / f'failed_{name}', out / f'noreply_{name}') if p.exists()), None)
            if p is not None:
                record = json.loads(p.read_text())
                if not _known_label_cost(record, p):
                    raise ValueError('saved label cost is not verified')
                value = episode_cost(record)
                primary_inspection = record.get('evidence_inspection')
                complete = (record.get('final_dispatch_outcome') not in ('claimed', 'unverified')
                            and not (record.get('evidence_inspection') or {}).get('cost_is_conservative_estimate'))
            cache = cache_dir / f'{name}.inspection.json'
            if cache.exists():
                inspection = json.loads(cache.read_text())
                recover_selection_reservations(inspection)
                reserve = inspection['cost_usd']
                if type(reserve) not in (int, float) or not math.isfinite(reserve) or reserve < 0:
                    raise ValueError('saved inspection cost is invalid')
                if (isinstance(primary_inspection, dict) and inspection.get('digest')
                        and primary_inspection.get('digest') == inspection['digest']):
                    value = max(value, float(reserve))
                else:
                    value += float(reserve)
                complete = complete and not inspection.get('cost_is_conservative_estimate')
            if not math.isfinite(value) or value < 0:
                raise ValueError('saved label cost is invalid')
            yield value, complete
        except (OSError, ValueError, TypeError, AttributeError, OverflowError):
            yield UNKNOWN_LABEL_RESERVE_USD, False


def _known_label_cost(record: dict, path: Path) -> bool:
    if not isinstance(record, dict):
        return False
    outcome = record.get('final_dispatch_outcome')
    if outcome in ('claimed', 'unverified'):
        cost = record.get('final_reserved_usd')
    elif outcome == 'not dispatched' or record.get('dry_run'):
        return True
    elif path.name.startswith('noreply_'):
        return False
    else:
        cost = (record.get('usage') or {}).get('est_cost_usd')
    return (isinstance(cost, (int, float)) and not isinstance(cost, bool)
            and math.isfinite(cost) and cost >= 0)


def prepare(job: Path, eps: Path, ids: list[str], *, free: bool, cap: float,
            api_key=None, call_model=None) -> dict:
    """Interpret a converted upload once, with a durable pre-dispatch budget reservation."""
    from label import dictionary as dd, harness
    from label.atomic import write_atomic
    job = Path(job)
    paths = [eps / eid for eid in ids]
    status_p = job / 'dictionary_status.json'
    prior_error = False
    try:
        prior = json.loads(status_p.read_text()) if status_p.exists() else {}
        if not isinstance(prior, dict):
            raise ValueError('dictionary status is not an object')
    except (OSError, ValueError):
        prior, prior_error = {}, True
    bound = prior.get('projected_upper_usd', 0.0)
    if not isinstance(bound, (int, float)) or isinstance(bound, bool) or not math.isfinite(bound) or bound < 0:
        bound = 0.0
    saved_labels = any((job / 'run' / 'out').glob('*episode_*.json'))
    result = {'status': 'skipped', 'cost_usd': 0.0, 'attempted': False, 'limitations': []}
    try:
        if saved_labels:
            result['limitations'].append('saved model replies retain their original request interpretation')
        else:
            cached = (job / 'dictionary.json').exists() or (job / 'dictionary.claim').exists()
            eligible = dd.needs_interpretation(paths) if not cached else True
            keys = harness.get_keys() if api_key is None and not free and not cached and eligible else []
            key = api_key if api_key is not None else (keys[0] if keys else None)
            plan = None
            refused = False
            if not free and not cached and eligible and key:
                inv = dd.inventory(paths)
                plan = dd.request_plan(inv)
                n_in = len(json.dumps(plan['content'], ensure_ascii=False).encode('utf-8')) + 1024
                prices = harness.OPENAI_PRICES[dd.MODEL.split('/', 1)[-1]][n_in >= harness.LONG_PROMPT_TOKENS]
                bound = n_in * prices[0] + plan['max_tokens'] * prices[2]
                available = max(0.0, min(DICTIONARY_MAX_USD, cap - label_spend(job)))
                if bound > available:
                    result['limitations'].append('dictionary request exceeds the reserved upload budget')
                    refused = True
            if not refused:
                write_atomic(status_p, {'status': 'pending', 'projected_upper_usd': bound}, indent=1)
                result = dd.prepare_upload(job, paths, key, call_model or harness.call_model_once,
                                           dry_run=free, planned_request=plan)
                from label.dictionary_context import apply_context
                overrides_p = job / 'dictionary_overrides.json'
                overrides = json.loads(overrides_p.read_text()) if overrides_p.exists() else None
                errors = []
                for path in paths:
                    try:
                        ctx_p = path / 'context.json'
                        ctx = json.loads(ctx_p.read_text())
                        updated = apply_context(ctx, result, overrides, episode_id=path.name)
                        if updated != ctx:
                            write_atomic(ctx_p, updated, indent=1)
                        if (updated.get('data_dictionary') or {}).get('entries'):
                            from label import episode as me, contacts as lc
                            import numpy as np
                            ep = me.load(path)
                            sig = ep.get('signals') or {}
                            n = len(next(iter(sig.values()))) if sig else 0
                            verdicts = me.touch_verdicts(ep, n)
                            updated['contacts'] = lc.find(sig, ep.get('signal_meta') or {},
                                np.array([me.frame_time(ep, k) for k in range(n)]), verdicts)
                            if updated != json.loads(ctx_p.read_text()):
                                write_atomic(ctx_p, updated, indent=1)
                    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
                        errors.append({'episode': path.name, 'error': type(error).__name__})
                if errors:
                    result = dict(result, context_errors=errors)
    except Exception as error:
        result = {'status': 'failed', 'cost_usd': None if (job / 'dictionary.claim').exists() else 0.0,
                  'limitations': [f'dictionary unavailable ({type(error).__name__}); recorded fields remain usable']}
    if prior_error:
        result = dict(result, limitations=list(result.get('limitations') or []) +
                      ['saved dictionary status was unreadable; original receipt and claim are retained'])
    status = {k: result.get(k) for k in ('status', 'model', 'reasoning', 'cost_usd', 'attempted',
                                        'inventory_digest', 'limitations', 'context_errors')}
    status['projected_upper_usd'] = bound
    status['spend'] = dictionary_spend(job)
    write_atomic(status_p, status, indent=1)
    return result
