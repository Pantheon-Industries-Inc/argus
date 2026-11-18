"""Reporting views of saved invoices. Budget enforcement keeps its separate liability ledger."""
from __future__ import annotations

import math


def _amount(value):
    return float(value) if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def has_response(record: dict) -> bool:
    """A response and its invoice are independent facts, including empty returned text."""
    return (isinstance(record.get('response_text'), str) or isinstance(record.get('response'), dict)
            or record.get('parse_ok') is True or record.get('final_dispatch_outcome') == 'received'
            or record.get('final_dispatch_outcome') is None and isinstance(record.get('labels'), dict))


def _verified_auxiliary(invoice):
    """Gross known round charges before subtracting this invoice's carried allocation."""
    if not invoice.get('cost_is_conservative_estimate'):
        return _amount(invoice.get('cost_usd')) or 0.0
    if invoice.get('kind') and 'rounds_cost_offset_usd' not in invoice:
        # Older retained invoices stored a net cost with cumulative rounds but
        # omitted their prior allocation. Its partition cannot be reconstructed.
        return 0.0
    known = 0.0
    for row in invoice.get('rounds') or []:
        usage = (row.get('response') or {}).get('usage') or {}
        if row.get('state') == 'received':
            known += _amount(usage.get('cost', usage.get('list_cost'))) or 0.0
    for part in invoice.get('parts') or []:
        # inspection parts wrap their invoice as {part, record}; a stitched route lists the part invoices themselves
        if isinstance(part, dict):
            known += _verified_auxiliary(part['record'] if isinstance(part.get('record'), dict) else part)
    return known


def billing_summary(annotation: dict, path=None) -> dict:
    """Separate verified charges from unresolved holds, including cumulative retry history.

    Missing invoices are unknown, never free. A partially verified inspection reports
    only charges backed by its round receipts; the remainder stays reserved.
    """
    from label.harness import billing_view, auxiliary_cost
    record = billing_view(annotation)
    billed, reserved, complete = 0.0, 0.0, True

    def final(invoice):
        nonlocal billed, reserved, complete
        outcome = invoice.get('final_dispatch_outcome')
        if outcome == 'not dispatched' or invoice.get('dry_run'):
            return
        if outcome in ('claimed', 'unverified') and invoice.get('final_reserved_source') == 'generation lookup':
            # the hold was replaced by what OpenRouter billed for the generation (harness.resolve_unanswered_charges)
            billed += _amount(invoice.get('final_reserved_usd')) or 0.0
            return
        if outcome in ('claimed', 'unverified'):
            complete = False
            reserved += _amount(invoice.get('final_reserved_usd')) or 0.0
            return
        usage = invoice.get('usage') or {}
        value = _amount(usage.get('est_cost_usd', usage.get('cost')))
        if value is None:
            complete = False
        else:
            billed += value

    def auxiliary(invoice, routing=False):
        nonlocal billed, reserved, complete
        total = auxiliary_cost(invoice, routing=routing)
        if not invoice.get('cost_is_conservative_estimate'):
            billed += total
            return
        complete = False
        # Round receipts preserve actual invoices even if a later round was lost.
        known = _verified_auxiliary(invoice)
        previous = ((_amount(invoice.get('cost_usd')) or 0.0) - total
                    + (_amount(invoice.get('rounds_cost_offset_usd')) or 0.0))
        verified = min(total, max(0.0, known - previous))
        billed += verified
        reserved += total - verified

    final(record)
    for invoice in record.get('final_cost_history') or []:
        final(invoice)
    for invoice in record.get('auxiliary_cost_history') or []:
        auxiliary(invoice)
    auxiliary((record.get('config') or {}).get('resolution_route') or {}, routing=True)
    auxiliary(record.get('evidence_inspection') or {})
    if path is not None:
        # The spend ledger (dictionary_stage._label_receipts) also counts evidence selections paid before a run was
        # killed, kept only in the inspection cache; the reports read the same receipts, so they never miss them.
        import json
        from pathlib import Path
        from label.harness import inspection_billed
        from label.evidence_access import recover_selection_reservations
        name = Path(path).name.removeprefix('failed_').removeprefix('noreply_')
        cache = Path(path).parent / '.evidence' / f'{name}.inspection.json'
        if cache.exists():
            try:
                inspection = json.loads(cache.read_text())
                recover_selection_reservations(inspection)
                # reconciled against the same receipt the totals above came from, as the spend ledger does
                extra = max(0.0, float(inspection['cost_usd']) - inspection_billed(record, inspection.get('digest')))
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                complete = False
            else:
                if inspection.get('cost_is_conservative_estimate'):
                    complete = False
                    reserved += extra
                else:
                    billed += extra
    return {'billed_usd': billed, 'reserved_usd': reserved, 'cost_complete': complete}
