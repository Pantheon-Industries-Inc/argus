"""Resolution routing: which cell width a teleop episode is sent at, chosen from its task text alone.

Routing is off by default (ARGUS_RESOLUTION_ROUTE, set by --resolution-route on python -m review and python -m
label.harness): a teleop episode is then sent at the narrow cells with contact detail views and no routing call, which
is what routing chose for the tasks it has seen, since almost every task is judged from whole objects. With routing
on, the rest of this note applies.

Most tasks read well at low resolution (whole objects, where they are, whether they moved); some can only be
judged with fine detail. Before an episode of a routed rig (label/episode.py ROUTE_WIDTHS) is labelled, a small
model reads the episode's own task text, never its frames, and says which. A task that needs fine detail gets the
wide cells, every other task the narrow cells plus contact detail views. The call costs a fraction of a cent, its
answer is recorded in the episode's output (config.resolution_route). Missing task text uses narrow cells without
a routing call; a failed call, an unclear answer or a dry run with task text uses wide cells. Episodes with the
same task text share one answer within a run; the cost is recorded on the episode whose call it was
(label.harness.episode_cost).

The answer is sampled, so two runs over the same episodes can send some of them at different widths. A model
comparison seeds the answers instead (seed(), label/harness.py --route-seeds): every model's run starts with the
reference run's recorded answer for each task text, so every model sees the same frames and makes no routing call.
"""
from __future__ import annotations

import json
import copy
import fcntl
import hashlib
import os
import threading
from contextlib import nullcontext
from pathlib import Path

from label import episode as me
from label import policy

ROUTE_MODEL = "openai/gpt-6-sol"
ROUTE_REASONING = "low"
# The answer is a short JSON object; the limit adds reasoning room at ROUTE_REASONING.
ROUTE_MAX_TOKENS = policy.output_limit(500, ROUTE_REASONING)
ROUTE_PROMPT = (
    "You choose the image resolution for labelling one episode of a robot-learning dataset, from its text alone. "
    "The labeller sees the episode's cameras as small frames, about 200 pixels wide. That is enough for most tasks: "
    "whole objects, where they are, whether they moved, opened or closed. Answer true only when judging the task "
    "needs finer detail than that: reading lettering, numbers, symbols or a display; telling which face or side of "
    "an object is up; or telling apart small objects of similar shape. Reply with JSON only: "
    '{"fine_detail": true or false, "why": "<a short phrase>"}.\n\n')

# One answer per task text for the life of the process: the episodes of one task share it.
_CACHE: dict = {}
_LOCK = threading.Lock()
_TASK_LOCKS: dict = {}


def seed(answers: dict, source: str) -> int:
    """Hold another run's routing answers, {task text as route_text gives it: {"fine_detail": true or false, "why":
    ...}}, as this process's own; returns how many. An answer that is not true or false is refused, since the wide
    cells it would fall back to need not be what the other run sent."""
    held = {}
    for text, a in answers.items():
        if not isinstance(a.get("fine_detail"), bool):
            raise ValueError(f"the seeded answer for {text[:80]!r} is not true or false")
        held[text] = {"fine_detail": a["fine_detail"], "why": str(a.get("why") or "")[:200], "cost_usd": 0.0,
                      "seeded_from": source}
    with _LOCK:
        _CACHE.update(held)
    return len(held)


def route_text(ep: dict) -> str:
    """The episode's task text as the router reads it: dataset, robot, instruction, task label, timed sub-steps and
    uploader notes, whichever the episode has."""
    ctx = ep["context"]
    parts = [f"Dataset: {ctx.get('dataset')}"]
    if ctx.get("robot_type"):
        parts.append(f"Robot: {ctx['robot_type']}")
    if (ctx.get("instruction") or "").strip():
        parts.append(f"Instruction: {ctx['instruction'].strip()}")
    if ctx.get("task_label"):
        parts.append("Task label: " + "; ".join(ctx["task_label"]))
    if ctx.get("annotation_subtasks"):
        parts.append(f"Timed sub-steps: {str(ctx['annotation_subtasks'])[:1500]}")
    if ctx.get("uploader_annotation"):
        parts.append(f"Uploader's notes: {str(ctx['uploader_annotation'])[:1500]}")
    return "\n".join(parts)


def request(ep: dict) -> list:
    """The router's message content for this episode."""
    return [{"type": "text", "text": ROUTE_PROMPT + route_text(ep)}]


def _paid_route(ep, api_key, call_model, timeout, out_path, identity, reserve_dispatch, settle_dispatch, billing_fields, force_refresh):
    from label.errors import DispatchError, SpendCap
    from label.harness import OPENAI_PRICES, _verified_final_cost, persist_final_claim, same_input_identity
    text = route_text(ep)
    routing_identity = {'version': 1, 'task_sha256': hashlib.sha256(text.encode()).hexdigest(),
                        'rig': me.rig(ep), 'model': ROUTE_MODEL, 'reasoning': ROUTE_REASONING,
                        'max_tokens': ROUTE_MAX_TOKENS}
    # the output the receipts belong to is given, never recovered from a receipt's name
    receipt_path = out_path.with_name('noreply_' + out_path.name) if out_path is not None else None
    if out_path is not None:
        mismatched = False
        for previous_path in (out_path, out_path.with_name('failed_' + out_path.name), receipt_path):
            if not previous_path.exists():
                continue
            previous = json.loads(previous_path.read_text())
            saved = (previous.get('config') or {}).get('resolution_route') or {}
            if saved.get('dispatch_outcome') in ('claimed', 'unverified', 'received', 'cached'):
                matched = (saved['routing_identity'] == routing_identity if 'routing_identity' in saved
                           else same_input_identity(previous.get('input_identity'), identity))
                if matched:
                    return dict(saved, previously_billed=True)
                mismatched = True
        if mismatched and not force_refresh:
            raise ValueError('saved routing receipt has changed or missing input')
    with _LOCK:
        hit = _CACHE.get(text)

    def persist(record):
        if receipt_path is not None:
            persist_final_claim(receipt_path, {
                'episode_dir': str(ep['dir']), 'input_identity': identity,
                **(billing_fields or {}),
                'config': {'resolution_route': {**record, 'model': ROUTE_MODEL,
                                                'routing_identity': routing_identity}}, 'parse_ok': False,
                'no_reply': 'annotation request not dispatched; routing receipt retained',
                'final_dispatch_outcome': 'not dispatched', 'final_reserved_usd': 0.0})

    if hit is not None:
        hit = dict(hit, cost_usd=0.0, cached=True, dispatch_outcome='cached', previously_billed=False,
                   cost_is_conservative_estimate=False, routing_identity=routing_identity)
        persist(hit)
        return hit
    content = request(ep)
    rates = OPENAI_PRICES[ROUTE_MODEL.split('/', 1)[-1]][1]
    bound = (len(content[0]['text'].encode('utf-8')) + 1024) * rates[0] + ROUTE_MAX_TOKENS * rates[2]
    if reserve_dispatch is not None and not reserve_dispatch(bound):
        persist({'fine_detail': None, 'why': 'routing exceeds the remaining batch spend cap',
                 'cost_usd': 0.0, 'dispatch_outcome': 'not dispatched'})
        raise SpendCap('routing request exceeds the remaining batch spend cap')
    hit = {'fine_detail': None, 'why': 'routing response unverified; using wide cells',
           'cost_usd': bound, 'reserved_usd': bound, 'dispatch_outcome': 'claimed',
           'cost_is_conservative_estimate': True, 'routing_identity': routing_identity}
    try:
        persist(hit)
    except BaseException:
        if settle_dispatch is not None:
            settle_dispatch(bound, 0.0)
        raise
    try:
        resp = call_model(content, ROUTE_MODEL, ROUTE_REASONING, api_key,
                          max_tokens=ROUTE_MAX_TOKENS, timeout=timeout)
    except DispatchError as error:
        if error.verified_unpaid:
            persist({'fine_detail': None, 'why': 'routing request rejected before processing', 'cost_usd': 0.0,
                     'dispatch_outcome': 'not dispatched'})
            if settle_dispatch is not None:
                settle_dispatch(bound, 0.0)
            raise
        hit.update(why=f'routing failed: {str(error)[:120]}', dispatch_outcome='unverified')
    except Exception as error:  # noqa: BLE001 - recorded as the route's why, its outcome unverified
        hit.update(why=f'routing failed: {str(error)[:120]}', dispatch_outcome='unverified')
    else:
        usage = resp.get('usage') or {}
        cost, source = _verified_final_cost(usage, ROUTE_MODEL)
        hit.update(cost_usd=bound if cost is None else cost, usage=usage, cost_source=source,
                   generation_id=resp.get('id'), dispatch_outcome='received',
                   cost_is_conservative_estimate=cost is None)
        hit['response'] = copy.deepcopy(resp)
        try:
            choice = (resp.get('choices') or [{}])[0]
            hit.update(response_text=(choice.get('message') or {}).get('content'),
                       finish_reason=choice.get('finish_reason') or choice.get('native_finish_reason'))
            if hit['finish_reason'] == 'length':
                raise ValueError('routing reply cut off; using wide cells')
            msg = choice['message']['content']
            ans = json.loads(msg[msg.find('{'):msg.rfind('}') + 1])
            fine = ans.get('fine_detail')
            hit.update(fine_detail=fine if isinstance(fine, bool) else None,
                       why=str(ans.get('why') or '')[:200])
        except (ValueError, KeyError, TypeError, AttributeError, IndexError) as error:
            hit['why'] = f'routing failed: {str(error)[:120]}'
    persist(hit)
    if settle_dispatch is not None:
        settle_dispatch(bound, hit['cost_usd'])
    with _LOCK:
        _CACHE[text] = hit
    return hit


def enabled() -> bool:
    """Whether teleop episodes are routed: ARGUS_RESOLUTION_ROUTE=on. Read at each call, so the harness process a
    review starts follows the flag it was given."""
    return os.environ.get("ARGUS_RESOLUTION_ROUTE", "off").strip().lower() == "on"


def route_width(ep_dir: Path, api_key: str | None, call_model, timeout: int = 120, *,
                out_path=None, input_identity=None, reserve_dispatch=None,
                settle_dispatch=None, billing_fields=None, force_refresh=False) -> tuple[int | None, dict]:
    """(widest cell width, record) for an episode of a routed rig; (None, {"routed": False}) for the others.
    call_model is label.harness.call_model_once; api_key None means a dry run."""
    ep = me.load(ep_dir)
    r = me.rig(ep)
    if r not in me.ROUTE_WIDTHS:
        return None, {"routed": False}
    low, high = me.ROUTE_WIDTHS[r]
    if not enabled():
        return low, {"routed": False, "why": "resolution routing off", "cell_w": low}
    ctx = ep["context"]
    # a plain video's task label is only its file name, which says nothing about the task
    real = any(value.strip() if isinstance(value, str) else value
               for value in (ctx.get(k) for k in ("instruction", "annotation_subtasks", "uploader_annotation")))
    from_video = (ctx.get("source") or {}).get("format") == "video files"
    has_task = real or (not from_video and any(label.strip() for label in ctx.get("task_label") or []))
    if not has_task:
        return low, {"routed": True, "fine_detail": None, "why": "no task text", "cell_w": low}
    if not api_key:
        return high, {"routed": True, "fine_detail": None, "why": "no key (dry run)", "cell_w": high}
    text = route_text(ep)
    with _LOCK:
        task_lock = _TASK_LOCKS.setdefault(text, threading.Lock())
    out_path = Path(out_path) if out_path is not None else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    # A process restart reuses the durable claim, including an ambiguous dispatch.
    with task_lock, (out_path.with_name('.noreply_' + out_path.name + '.route.lock').open('a')
                     if out_path is not None else nullcontext()) as lock:
        if lock is not None:
            fcntl.flock(lock, fcntl.LOCK_EX)
        hit = _paid_route(ep, api_key, call_model, timeout, out_path, input_identity,
                          reserve_dispatch, settle_dispatch, billing_fields, force_refresh)
    w = low if hit["fine_detail"] is False else high
    return w, {"routed": True, "model": ROUTE_MODEL, **hit, "cell_w": w}
