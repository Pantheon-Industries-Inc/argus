"""The service tier a run asks for, and how often it got it (--service-tier, --flex-fallback).

Flex is the same model and the same request at half the price, with lower processing priority. A provider that has no
flex capacity rejects the request unpaid, so harness._post retries it with backoff. What happens after that is the
fallback setting. Off, the unit stays unlabelled and unpaid and a resume sends it again, never above the flex price. On,
the request goes once more at the standard tier, at twice the price, and the cap reserves that rate for every request.

The tier is a run setting and is left out of input_identity: it changes what a request costs, never what it asks.
"""
from __future__ import annotations

import json
import threading
from collections import Counter
from pathlib import Path

from label.atomic import write_atomic

COUNTS = '.service_tier/counts.json'

_lock = threading.Lock()
_requested: str | None = None
_fallback = False
_rejected = 0
_fell_back = 0
_replies: Counter = Counter()


def configure(service_tier: str | None, fallback: bool) -> None:
    global _requested, _fallback, _rejected, _fell_back, _replies
    if service_tier not in (None, 'flex', 'default'):
        raise ValueError(f'unknown service tier: {service_tier}')
    with _lock:
        _requested, _fallback, _rejected, _fell_back, _replies = service_tier, fallback, 0, 0, Counter()


def requested() -> str | None:
    return _requested


def flex() -> bool:
    return _requested == 'flex'


def reserves_flex() -> bool:
    """Whether every request of this run can only be billed at flex rates."""
    return flex() and not _fallback


def falls_back() -> bool:
    return flex() and _fallback


def body_fields() -> dict:
    """The provider's default tier is what a request gets when it names none."""
    return {'service_tier': 'flex'} if flex() else {}


def is_capacity_rejection(response: dict) -> bool:
    """A 200 with no choices and no usage that says the provider has no flex capacity, which is how OpenRouter sent one."""
    error = response.get('error') if isinstance(response.get('error'), dict) else response
    metadata = error.get('metadata') if isinstance(error.get('metadata'), dict) else {}
    return error.get('code') == 502 and metadata.get('error_type') == 'provider_unavailable' and not response.get('usage')


def note_rejection() -> None:
    global _rejected
    with _lock:
        _rejected += 1


def note_fallback() -> None:
    global _fell_back
    with _lock:
        _fell_back += 1


def note_reply(response: dict) -> None:
    if _requested is not None:
        with _lock:
            _replies[str(response.get('service_tier'))] += 1


def counts() -> dict:
    with _lock:
        return {'requested': _requested, 'fallback': _fallback, 'rejected': _rejected, 'fell_back': _fell_back,
                'replies': dict(_replies)}


def write(out_dir: Path) -> None:
    """Add this run's counts to the folder's, since a resumed run writes to the same folder. They go in a subfolder,
    as the inspection caches do, because board.lerobot_export, label.catalog, label.seam and label.issue_groups read
    every *.json at the top of a labels folder as a label. The run's paid work is already saved, so a failure here
    only warns."""
    if _requested is None:
        return
    now = counts()
    path = Path(out_dir) / COUNTS
    try:
        if path.exists():
            before = json.loads(path.read_text())
            now['rejected'] += before.get('rejected', 0)
            now['fell_back'] += before.get('fell_back', 0)
            now['replies'] = dict(Counter(before.get('replies', {})) + Counter(now['replies']))
        path.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(path, now, indent=1)
    except (OSError, ValueError, AttributeError) as error:
        print(f'WARN could not record the service tier counts: {error}', flush=True)
