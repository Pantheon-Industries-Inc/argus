"""Label episodes with a vision-language model: one request per episode, one JSON file per episode.

    python -m label.harness --episode-dir EPISODE [--out OUT.json] [--dry-run]
    python -m label.harness --episodes-root SLICE --out-dir OUT [--concurrency N] [--max-spend USD] [--dry-run]

Runs normally start through `python -m label` (label/run.py), which gives each run its own folder, records the
code commit and settings, and enforces a spend cap. This module is what it calls.

Each episode is a sidecar folder (label/episode.py builds the request from it). The request goes to OpenRouter,
or straight to OpenAI with an OpenAI key (the same request in OpenAI's fields, with no cache breakpoint, since
OpenAI caches a repeated prefix itself, and the cost priced from OpenAI's list prices, since OpenAI reports tokens
but no cost). It carries the model id, the reasoning effort (medium by default), max_completion_tokens,
response_format json_object, a cache_control breakpoint at the end of the shared instructions, and usage
include (so the billed cost comes back). No temperature, top_p or seed is sent and no provider is pinned: each
model runs with its provider's own sampling defaults. Each output records what served the request:
provider_name, generation_id, model_served and system_fingerprint (None when the provider gives none). Before a
teleop episode's request is built, a text-only routing call chooses its cell width (label/route.py); its answer and
cost are recorded in config.resolution_route, and episode_cost() counts both calls.

Output per episode (out.json, or OUT/<episode>.json): the model's labels (`labels`, parsed; `_raw` and
`_parse_error` when the reply did not parse), `parse_ok` (false as well for a reply that parses but lacks work its
request asked for, label/completeness.py, with what it lacks under labels._incomplete), what was sent (`config`:
cameras, cell size, the exact instants), the deterministic checks that ran on the episode (`dataset_checks`), the
still spans the model was told about, the instruction it was graded against, and the billed usage and cost. A reply
cut off at the output limit is kept as failed_<episode>.json, with what was sent, and counts as a failure; the board
shows the episode with that reply, as it shows one whose reply did not parse. An episode that got no reply at all (the spend
cap reached, every key out of credit, a request that could not be built, a call that failed) has
noreply_<episode>.json saying why, so the board shows it too.

Keys: OPENROUTER_API_KEYS, a comma-separated list, or when it holds none OPENAI_API_KEY, which sends every call
straight to OpenAI and so only runs OpenAI models; keys are used round robin, and a key that runs out of
credit is retired for the rest of the batch. RDA_DECODE_CONCURRENCY (default 12) bounds the frame decodes
running at once across the batch. A dry run builds every request exactly as it would be sent (frames decoded,
grids composed, prompt assembled, caps checked) and writes it with a token estimate, without calling the model.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import http.client
import json
import math
import os
import re
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from functools import lru_cache

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from board.label_projection import LABEL_SCHEMA, derive_completion
from label import episode as me, policy
from label.atomic import write_atomic
from label.errors import DispatchError, KeyExhausted, SpendCap, Truncated
from label.route import route_width

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-6-astra"
DETAIL = "high"
GRID_COLS = 4

# gpt-6-astra's input list price (USD per token); gate/score.py prices prompt cache reads and writes from it
PRICE_IN = 1.0e-5
# OpenAI list prices (USD per token: input, cached input, output), and from LONG_PROMPT_TOKENS prompt tokens the
# long-context ones, for a call sent straight to OpenAI, whose response carries tokens but no cost
OPENAI_PRICES = {"gpt-6-astra": ((1.0e-5, 1.0e-6, 5.0e-5), (2.0e-5, 2.0e-6, 7.5e-5)),
                 "gpt-6-sol": ((2.0e-6, 2.0e-7, 1.0e-5), (4.0e-6, 4.0e-7, 1.5e-5)),
                 # Standard pricing: https://developers.openai.com/api/docs/models/gpt-6.1-sol
                 "gpt-6.1-sol": ((2.0e-6, 1.0e-7, 1.0e-5), (4.0e-6, 2.0e-7, 1.5e-5))}
LONG_PROMPT_TOKENS = 272000

# Requests carry at most 500 images, and at most IMAGE_LIMIT_BYTES of images as the provider counts them
# (label/episode.py steps the cell width down to fit).
IMAGE_LIMIT_BYTES = me.IMAGE_LIMIT_BYTES
IMAGE_SIZE_INFLATION = me.IMAGE_SIZE_INFLATION
MAX_IMAGES = 500

# Frame decoding is CPU-heavy (AV1, HEVC), and many episodes are prepared at once, so the total number of
# concurrent decodes is bounded here, whatever the API concurrency. It does not change which frames are sent.
_dg = os.environ.get("RDA_DECODE_CONCURRENCY")
DECODE_GATE = threading.BoundedSemaphore(int(_dg) if (_dg or "").isdigit() and int(_dg) > 0 else 12)


def is_key_exhausted(code: int, body: str) -> bool:
    b = body.lower()
    # 402 Payment Required (insufficient credits), or 401 on a revoked or disabled key
    return (code == 402 or ("insufficient_quota" in b) or ("insufficient credits" in b)
            or ("credit" in b and code in (402, 403)) or code == 401)


def is_openrouter_key(key: str) -> bool:
    return key.startswith("sk-or-")


def openai_list_cost(model: str, usage: dict) -> float | None:
    """What a call straight to OpenAI costs at list price, cached input at its own price (None for a model not in
    OPENAI_PRICES)."""
    tiers = OPENAI_PRICES.get(model.split("/", 1)[-1])
    if tiers is None or not _complete_token_usage(usage):
        return None
    n_in = usage["prompt_tokens"]
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    p_in, p_cached, p_out = tiers[n_in >= LONG_PROMPT_TOKENS]
    return (n_in - cached) * p_in + cached * p_cached + usage["completion_tokens"] * p_out


def _complete_token_usage(usage: dict) -> bool:
    if not isinstance(usage, dict) or any(type(usage.get(key)) is not int or usage[key] < 0
                                          for key in ("prompt_tokens", "completion_tokens")):
        return False
    details = usage.get("prompt_tokens_details")
    if details is not None and not isinstance(details, dict):
        return False
    cached = (details or {}).get("cached_tokens", 0)
    return type(cached) is int and 0 <= cached <= usage["prompt_tokens"]


def _verified_final_cost(usage: dict, model: str) -> tuple[float | None, str]:
    if not isinstance(usage, dict):
        return None, "unverified"
    for key, source in (("cost", "billed"), ("list_cost", "list")):
        value = usage.get(key)
        try:
            if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                return float(value), source
        except OverflowError:
            pass
    if (not _complete_token_usage(usage) or not model.startswith("openai/")
            or model.split("/", 1)[-1] not in OPENAI_PRICES):
        return None, "unverified"
    try:
        listed = openai_list_cost(model, usage)
        cost = listed
        if cost is not None and math.isfinite(cost) and cost >= 0:
            return cost, "estimate"
    except OverflowError:
        pass
    return None, "unverified"


# What a request body asks besides its messages, model, reasoning effort and output limit, which input_identity
# records on their own. usage.include asks for the billed cost (cache reads are about 1/10 price, cache writes about
# 1.25x), so every recorded cost and spend cap uses what was actually charged.
OPENROUTER_BODY = {"response_format": {"type": "json_object"}, "usage": {"include": True}}
OPENAI_BODY = {"response_format": {"type": "json_object"}}
# These bodies as every saved label so far was requested with. A body that differs is named in input_identity, so
# a label requested with another body is never taken as current (request_attestation).
FIRST_BODY_SHA256 = '879adaf179bcedfd6686f0ab4547e79e1862b71b439fc9815cbb789b43a0d50c'


def request_body_sha256() -> str:
    return hashlib.sha256(json.dumps({'openai': OPENAI_BODY, 'openrouter': OPENROUTER_BODY},
                                     sort_keys=True).encode()).hexdigest()


def call_model(content: list, model: str, reasoning: str, api_key: str, max_tokens: int, timeout: int,
               *, attempts: int = 6, pricing_quote=None) -> dict:
    if not is_openrouter_key(api_key):
        return _call_openai(content, model, reasoning, api_key, max_tokens, timeout, attempts=attempts)
    if content and content[0].get("type") == "text" and len(content[0].get("text", "")) > 4000:
        # mark the end of the shared instructions as a cache breakpoint: the provider caches whole prompts up to
        # a breakpoint, so without one, episodes that share only the instructions never hit the cache (measured:
        # 0 cached tokens; with it, the instruction tokens are read at about 1/10 price on every later call)
        content = [dict(content[0], cache_control={"type": "ephemeral"})] + list(content[1:])
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_completion_tokens": max_tokens,
        "reasoning": {"effort": reasoning},
        **OPENROUTER_BODY,
    }
    if pricing_quote is not None:
        if pricing_quote.get('model') != model:
            raise ValueError('pricing quote does not match the requested model')
        body['provider'] = {'max_price': pricing_quote['provider_max_price']}
    return _post(OPENROUTER_URL, body, api_key, timeout, attempts=attempts)


def call_model_once(content: list, model: str, reasoning: str, api_key: str, max_tokens: int, timeout: int,
                    *, pricing_quote=None) -> dict:
    """One HTTP attempt for a paid call. A failed dispatch may already have been billed."""
    return call_model(content, model, reasoning, api_key, max_tokens, timeout, attempts=1,
                      **({'pricing_quote': pricing_quote} if pricing_quote is not None else {}))


def _call_openai(content: list, model: str, reasoning: str, api_key: str, max_tokens: int, timeout: int,
                 *, attempts: int = 6) -> dict:
    """The same request straight to OpenAI. The response is recorded as OpenRouter's is: its provider is OpenAI
    and its usage carries list_cost, the list-price cost, where OpenRouter's carries the billed cost."""
    if not model.startswith("openai/"):
        raise RuntimeError(f"{model} is not an OpenAI model, so it needs an OpenRouter key")
    body = {"model": model.split("/", 1)[1], "messages": [{"role": "user", "content": content}],
            "max_completion_tokens": max_tokens, "reasoning_effort": reasoning, **OPENAI_BODY}
    resp = _post(OPENAI_URL, body, api_key, timeout, attempts=attempts)
    resp.setdefault("provider", "OpenAI")
    if isinstance(resp.get("usage"), dict):
        resp["usage"]["list_cost"] = openai_list_cost(model, resp["usage"])
    return resp


# What an HTTP failure says about billing. An error status is the provider declining to produce a completion, so
# nothing was billed, except a gateway timeout, behind which the request may still be running and billed.
UNKNOWN_OUTCOME_STATUS = frozenset({504, 520, 521, 522, 523, 524})
TRANSIENT_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 529})
UNPAID_ATTEMPTS = 4          # a provably unpaid failure is retried even on a paid call: retrying it cannot pay twice


def _retry_after(headers, attempt: int) -> float:
    try:
        stated = float((headers or {}).get('Retry-After'))
        if math.isfinite(stated) and stated >= 0:
            return min(60.0, stated)
    except (TypeError, ValueError):
        pass
    return min(60.0, 4.0 * 2 ** attempt)


# A request the provider failed on its side (the reply never came or was cut off, or the provider reported an error
# inside the reply), or one whose connection our own client closed as dead after SILENT_CONNECTION_LIMIT_S with no
# bytes (client_timeout, recorded as no_reply_kind client_timeout and never as the provider's failure), is sent again
# automatically, up to AUTO_RESENDS more times per unit, after AUTO_RESEND_DELAYS_S.
# Each attempt that may have been billed keeps its charge in the unit's billing history and against the cap, as an
# explicit --retry-received does; past the bound only --retry-received sends it again. A reply the model wrote that
# does not parse, a reply cut at the output limit and a complete annotation are never resent automatically.
AUTO_RESENDS = 2
AUTO_RESEND_DELAYS_S = (30.0, 120.0)


# How a record written before client timeouts were named (no_reply_kind) says our own client gave up: the exception
# our socket raised, never a provider's HTTP status or error text.
_CLIENT_TIMEOUT_TEXT = re.compile(r'\b(TimeoutError|ReadTimeout|socket\.timeout)\b')


def client_timeout(record: dict) -> str | None:
    """Why a unit's last attempt ended on our side, or None: our client closed a connection that carried no bytes
    for longer than SILENT_CONNECTION_LIMIT_S (no_reply_kind client_timeout), or, in a record written before that was
    named, the socket timeout our client raised. The provider never failed such a request; it may still have
    finished and billed it."""
    if not isinstance(record, dict) or record.get('dry_run') or record.get('parse_ok') is True:
        return None
    if record.get('response') is not None or record.get('final_dispatch_outcome') not in ('claimed', 'unverified'):
        return None
    why = str(record.get('no_reply') or '')
    if record.get('no_reply_kind') == 'client_timeout' or (
            record.get('no_reply_kind') is None and _CLIENT_TIMEOUT_TEXT.search(why)
            and not re.search(r'HTTP \d{3}:', why)):
        return why[:200] or 'our client closed a silent connection'
    return None


def provider_failure(record: dict) -> str | None:
    """Why a unit's last attempt failed on the provider's side, or None: no verified response (a request cut off or
    whose machine stopped), or the provider's own error inside a reply it delivered. Our own client timeout is never
    one (client_timeout)."""
    if not isinstance(record, dict) or record.get('dry_run') or record.get('parse_ok') is True:
        return None
    resp = record.get('response')
    if resp is None and record.get('final_dispatch_outcome') in ('claimed', 'unverified'):
        if client_timeout(record):
            return None
        return str(record.get('no_reply') or 'the request has no verified response')[:200]
    if isinstance(resp, dict) and record.get('finish_reason') != 'length':
        failure = reply_error(resp)
        if failure is not None:
            return f'the provider returned an error in the reply ({failure})'
    return None


def dispatched_attempts(record: dict) -> int:
    """How many times this unit's final request was sent: the attempts in its billing history and the last one,
    never counting an attempt that was not dispatched."""
    # an attempt was sent when its request was claimed for dispatch or anything came back (a completed record names no
    # dispatch outcome, but holds the response); one refused before dispatch or built in a dry run was not
    sent = lambda r: (isinstance(r, dict) and not r.get('dry_run')
                      and r.get('final_dispatch_outcome') != 'not dispatched'
                      and (r.get('final_dispatch_outcome') in ('claimed', 'unverified', 'received')
                           or r.get('response') is not None or r.get('finish_reason') is not None))
    history = record.get('final_cost_history') if isinstance(record, dict) else None
    return sum(1 for r in history or [] if sent(r)) + (1 if sent(record) else 0)


def auto_resend(record: dict) -> str | None:
    """Why to send this unit again, while it has been sent at most AUTO_RESENDS times, or None: a provider failure,
    or a connection our client found dead, each named as what it was."""
    failure = provider_failure(record)
    ours = None if failure else client_timeout(record)
    why = f'provider failure: {failure}' if failure else f'our client closed a dead connection: {ours}' if ours else None
    return why if why and dispatched_attempts(record) <= AUTO_RESENDS else None


def reply_error(resp: dict) -> str | None:
    """The provider's own failure carried inside a reply it delivered with HTTP 200 (finish_reason error, or an error
    on the choice, eg an overloaded upstream), or None. Such a reply's text is a fragment, never an answer to parse."""
    choices = resp.get('choices') if isinstance(resp, dict) else None
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    error = choice.get('error')
    if choice.get('finish_reason') != 'error' and not error:
        return None
    if isinstance(error, dict):
        return ' '.join(str(error.get(key)) for key in ('code', 'message') if error.get(key) is not None)[:300] or 'error'
    return str(error)[:300] if error else 'finish_reason error'


# How long the connection may carry no bytes at all before our client calls it dead and closes it. It never ends a reply
# the provider is still sending, however long the reply takes in total: it is the longest silence a live reply can
# have, taken from the provider's published worst case. OpenRouter and api.openai.com both answer through Cloudflare
# (every response carries server: cloudflare and a cf-ray), and Cloudflare ends a request whose origin sends nothing
# for its Proxy Read Timeout, 125s by default and at most 6000s on any plan
# (https://developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-5xx-errors/error-524/).
# OpenRouter says it sends keepalive comments on a stream but publishes no spacing, and nothing for a request that is
# not streamed, so the bound is Cloudflare's maximum, plus a minute so that Cloudflare's own close reaches us first
# and is recorded as the provider's. --timeout can raise it, never lower it.
CLOUDFLARE_MAX_PROXY_READ_S = 6000
SILENT_CONNECTION_LIMIT_S = CLOUDFLARE_MAX_PROXY_READ_S + 60


def silent_connection_limit(timeout) -> float:
    """The socket timeout for a model request: the caller's timeout, never below SILENT_CONNECTION_LIMIT_S."""
    return max(float(timeout or 0), float(SILENT_CONNECTION_LIMIT_S))


def _post(url: str, body: dict, api_key: str, timeout: int, *, attempts: int = 6) -> dict:
    """POST one request. attempts bounds retries of failures whose billing is unknown, so a paid call passes 1 and
    is never replayed after it may have been billed. A failure that provably cost nothing (the request never fully
    left the machine, or the provider answered with an error status) is retried up to UNPAID_ATTEMPTS times, and
    when it persists it raises DispatchError with verified_unpaid True, so its caller releases the reservation and a
    plain resume tries the unit again."""
    if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 1:
        raise ValueError("HTTP attempts must be a positive integer")
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    unknown_left, unpaid_left = attempts, max(attempts, UNPAID_ATTEMPTS)
    silence = silent_connection_limit(timeout)
    clean = True          # no earlier attempt of this call may have been billed
    tries = 0
    while True:
        tries += 1
        try:
            # urllib raises URLError only while connecting and sending the request; every error after the request
            # was sent (waiting for or reading the response) propagates as itself. The socket timeout bounds each
            # wait for bytes, never the whole reply, so a reply that keeps arriving is never cut off.
            with urllib.request.urlopen(req, timeout=silence) as r:
                resp = json.loads(r.read().decode())
            if not resp.get("choices"):
                # HTTP 200 carrying an error body and no completion: fail loudly, never record an empty parse
                err = json.dumps(resp.get("error") or resp)[:300]
                raise DispatchError(f"200 response with no choices: {err}", http_status=200, stage='unknown')
            failure = reply_error(resp)
            cost = (resp.get('usage') or {}).get('cost') if isinstance(resp.get('usage'), dict) else None
            if failure is not None and type(cost) in (int, float) and cost == 0:
                # the provider failed inside a 200 reply (an overloaded upstream) and billed nothing: it cost nothing,
                # so it is retried like an error status; a billed or unpriced one is kept as received, and only
                # --retry-received sends it again
                last = f"provider error in the reply: {failure}"
                unpaid_left -= 1
                if unpaid_left > 0:
                    time.sleep(min(30.0, 2.0 ** (tries - 1)))
                    continue
                raise DispatchError(f"model call failed after {tries} HTTP attempts: {last}", http_status=200,
                                    verified_unpaid=clean, stage='rejected')
            return resp
        except urllib.error.HTTPError as e:
            # Read a bounded complete envelope before shortening its display message.
            raw_error = e.read(65537)
            text = raw_error.decode(errors="replace")
            complete_error = len(raw_error) <= 65536
            last = f"HTTP {e.code}: {text[:300]}"
            if is_key_exhausted(e.code, text):
                quota = None
                try:
                    quota = json.loads(text).get('error', {}).get('code') if complete_error else None
                except (ValueError, AttributeError):
                    pass
                # a key rejection proves the key is spent only when no earlier attempt of this call may have been
                # billed; after one, it is just a failure whose cost is unknown
                if clean and (e.code in (401, 402) or (url == OPENAI_URL and e.code == 429
                                                         and quota == 'insufficient_quota')):
                    raise KeyExhausted(last, http_status=e.code, first_attempt=True, rejection_code=quota) from e
                raise DispatchError(last, http_status=e.code, verified_unpaid=clean, stage='rejected') from e
            if e.code in UNKNOWN_OUTCOME_STATUS:
                clean = False
                unknown_left -= 1
                if unknown_left > 0:
                    time.sleep(_retry_after(None, tries - 1))
                    continue
                raise DispatchError(f"model call failed after {tries} HTTP attempts: {last}", http_status=e.code,
                                    stage='unknown') from e
            if e.code in TRANSIENT_STATUS:
                unpaid_left -= 1
                if unpaid_left > 0:
                    time.sleep(_retry_after(e.headers, tries - 1))
                    continue
            raise DispatchError(f"model call failed after {tries} HTTP attempts: {last}", http_status=e.code,
                                verified_unpaid=clean, stage='rejected') from e
        except urllib.error.URLError as e:
            last = f"request not sent: {e.reason}"
            unpaid_left -= 1
            if unpaid_left > 0:
                time.sleep(min(30.0, 2.0 ** (tries - 1)))
                continue
            raise DispatchError(f"model call failed after {tries} HTTP attempts: {last}", verified_unpaid=clean,
                                stage='not sent') from e
        except DispatchError:
            raise
        except (OSError, http.client.HTTPException, ValueError) as e:
            # sent, then timed out, reset or answered with an unreadable body: the outcome is unknown. A timeout is
            # our own client closing a connection silent for longer than any live reply can be, never the provider's
            ours = isinstance(e, TimeoutError)
            last = (f"our client closed the connection after {silence:g}s with no bytes from the provider "
                    f"({type(e).__name__}: {e})" if ours else f"{type(e).__name__}: {e}")
            clean = False
            unknown_left -= 1
            if unknown_left > 0:
                time.sleep(min(30.0, 2.0 ** (tries - 1)))
                continue
            raise DispatchError(f"model call failed after {tries} HTTP attempts: {last}", stage='unknown',
                                client_timeout=ours) from e


def label_episode(ep_dir: Path, out_path: Path, *, model: str, reasoning: str, api_key: str, max_tokens: int,
                  timeout: int, cell_w: int = 0, example_dir: str | None = None, dry_run: bool = False,
                  reserve_final=None, reserve_selection=None, settle_selection=None, force_refresh=False,
                  settle_final=None,
                  sample_fps=None, task_mode=None, dataset_note=None, require_known_pricing=False,
                  inspection_refresh_id=None, inspection=None) -> dict:
    context = json.loads((Path(ep_dir) / 'context.json').read_text())
    resolved = policy.resolve(context, model=model, reasoning=reasoning, sample_fps=sample_fps,
                              task_mode=task_mode, dataset_note=dataset_note, inspection=inspection)
    model, reasoning = resolved['model'], resolved['reasoning']
    if not dry_run and not is_openrouter_key(api_key) and not model.startswith('openai/'):
        raise ValueError(f'{model} requires an OpenRouter key; no request was dispatched.')
    pricing_quote = None
    if not dry_run and require_known_pricing and (not model.startswith('openai/')
            or model.split('/', 1)[-1] not in OPENAI_PRICES):
        if not is_openrouter_key(api_key):
            raise ValueError(f'No verified pricing rates for {model}; a spend cap requires known rates')
        from label import pricing
        pricing_quote = pricing.quote(model, timeout)
    identity = input_identity(ep_dir, model=model, reasoning=reasoning, max_tokens=max_tokens,
                              cell_w=cell_w, example_dir=example_dir, sample_fps=sample_fps,
                              task_mode=task_mode, dataset_note=dataset_note, inspection=inspection)
    if any(item.get('missing') for item in identity['source_proof']):
        raise RuntimeError('Request source inputs are missing; no paid request was dispatched.')
    billing = _billing_fields(out_path)
    # an explicit cell width skips the routing; otherwise a routed rig's widest cell comes from its task text
    route_w, route = (None, {"routed": False}) if cell_w else route_width(
        ep_dir, None if dry_run else api_key, call_model_once, timeout=timeout,
        out_path=out_path, input_identity=identity,
        reserve_dispatch=reserve_selection, settle_dispatch=settle_selection, billing_fields=billing,
        force_refresh=force_refresh)
    req = me.build_request(ep_dir, detail=DETAIL, gate=DECODE_GATE, grid_cols=GRID_COLS, cell_w=cell_w or None,
                           max_cell_w=route_w, example_dir=example_dir, inspect_evidence=True,
                           model=model, reasoning=reasoning, sample_fps=sample_fps,
                           task_mode=task_mode, dataset_note=dataset_note, inspection=policy.inspection_mode(resolved),
                           derived_dir=Path(out_path).parent / ".derived" / "jaws")
    pl = req["plan"]
    fields = {
        "input_identity": identity,
        "annotation_policy": resolved,
        **({"jaw_evidence": req["jaw_evidence"]} if req.get("jaw_evidence") else {}),
        **billing,
        **({"sensor_evidence": req["sensor_evidence"]} if req.get("sensor_evidence", {}).get("sensors") else {}),
        "given_prompt": req["given_prompt"],
        # graded against the instruction it was given, or against tasks it inferred; every request asks for a list of
        # tasks (label_schema), so the recording's completion is derived from them (read_reply)
        "prompt_mode": "given" if req["given_prompt"] else "inferred",
        "label_schema": LABEL_SCHEMA,
        "task_prompts": req["task_prompts"],
        "task_label": req["task_label"],
        "sampling": req["sampling"],
        "arm_still_spans": req["still_spans"],
        "dataset_checks": pl["checks"],
        # the stretches a camera's file could not be decoded at (label/episode.py decode_failures), which the board
        # shows as reader issues (board/build.py); labelling never writes into the episode folder
        "decode_failed": req["decode_failed"],
        "example_dir": str(example_dir) if example_dir else None,
        # the recording's contacts (label/contacts.py) and the strips each one shown was drawn with (checks/contacts.py)
        **({"contacts": req["contacts"], "contact_views": req["contact_views"]} if req.get("contact_views") else {}),
        "config": {"views": req["views"], "cam_labels": req["cam_labels"], "layout": "grid",
                   **({'sensor_copy_contract_version': req['sensor_copy_contract_version']}
                      if 'sensor_copy_contract_version' in req else {}),
                   **({'sensor_scene_pairing': req['sensor_scene_pairing']}
                      if (req.get('sensor_scene_pairing', {}).get('snapshot_times')
                          or any(req.get('sensor_scene_pairing', {}).get('omissions', {}).values())) else {}),
                   **({'camera_observation_views': req['camera_observation_views'],
                       'camera_observation_labels': req['camera_observation_labels']}
                      if 'camera_observations' in req['schema_fields'] else {}),
                   "grid_cols": req["grid_cols"], "cell": req["cell"],
                   "n_timesteps": len(pl["ks"]), "n_frames_sent": len(pl["ks"]) * len(req["cam_labels"]) + req.get("native_tactile_frames", 0)
                   + req.get('sensor_scene_pairing', {}).get('supplemental_captures', 0),
                   "n_image_parts": req["n_images"], "fullres_frames": ["first", "last"],
                   "contact_detail_s": req["contact_s"], "resolution_route": route,
                   "timesteps_s": req["timesteps"], "detail": DETAIL, "circular_image": req["lens"],
                   "prompt_blocks": req["blocks"], "schema_fields": req["schema_fields"],
                   "checks_implied": req["checks_implied"]},
    }
    if pricing_quote is not None:
        fields['config']['pricing_quote'] = pricing_quote
    return _call_and_record(ep_dir, out_path, req["content"], req["image_bytes"], fields, model=model,
                            reasoning=reasoning, api_key=api_key, max_tokens=max_tokens, timeout=timeout,
                            dry_run=dry_run, prompt_text=req["prompt"], evidence_access=req["evidence_access"],
                            reserve_final=reserve_final, reserve_selection=reserve_selection,
                            settle_selection=settle_selection, settle_final=settle_final, force_refresh=force_refresh,
                            require_known_pricing=require_known_pricing, inspection_refresh_id=inspection_refresh_id,
                            pricing_quote=pricing_quote)


@lru_cache(maxsize=128)
def _request_imports(module: str, is_package: bool, source: bytes) -> tuple[str, ...]:
    """Cache only parsed import edges, keyed by module semantics and exact current bytes."""
    import ast
    names = module.split('.')
    imports = ['.'.join(names[:size]) for size in range(1, len(names))]
    for node in ast.walk(ast.parse(source, filename=module)):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ''
            if node.level:
                parent = names if is_package else names[:-1]
                base = '.'.join(parent[:len(parent) - node.level + 1] + ([base] if base else []))
            imports.append(base)
            imports.extend(base + '.' + alias.name for alias in node.names if alias.name != '*')
    return tuple(imports)


def request_implementation(repository: Path) -> dict[Path, bytes]:
    """Follow first-party imports from the paid request and binding entry point.

    Include imports inside functions and package initializers, so native evidence,
    drawing and reader projections keep their transitive guards. Unrelated board
    publication and server modules do not shape requests and stay outside this proof.
    """
    pending, sources = ['label.harness'], {}
    packages = {'label', 'prepare', 'checks', 'board'}
    while pending:
        module = pending.pop()
        if module.split('.')[0] not in packages:
            continue
        path = repository.joinpath(*module.split('.')).with_suffix('.py')
        if not path.is_file():
            path = repository.joinpath(*module.split('.'), '__init__.py')
        if not path.is_file() or path in sources:
            continue
        raw = path.read_bytes()
        sources[path] = raw
        pending.extend(_request_imports(module, path.name == '__init__.py', raw))
    return sources


def input_identity(ep_dir: Path, *, model: str, reasoning: str, max_tokens: int, cell_w: int,
                   example_dir: str | None, sample_fps=None, task_mode=None, dataset_note=None,
                   inspection=None) -> dict:
    """Bounded proof for source files and the settings that shape a paid annotation request."""
    from label.evidence_access import source_proof, reader_projection_identity
    ep_dir = Path(ep_dir)
    ctx = json.loads((ep_dir / "context.json").read_text())
    from prepare.state_contract import prepared_state_problem
    dims, has_action = None, False
    state_path = ep_dir / 'state.npz'
    if ctx.get('state_kind') in ('joints', 'ee_pose') and state_path.is_file():
        with np.load(state_path) as arrays:
            state = arrays['state']
            dims = state.shape[1] if state.ndim == 2 else None
            has_action = 'action' in arrays.files
    problem = prepared_state_problem(ctx, dims, has_action=has_action)
    if problem:
        raise RuntimeError(f'{ep_dir}: {problem}')
    resolved = policy.resolve(ctx, model=model, reasoning=reasoning, sample_fps=sample_fps,
                              task_mode=task_mode, dataset_note=dataset_note, inspection=inspection)
    me.require_request_window(ep_dir, {**ctx, 'annotation_policy': resolved})
    from label.dictionary_editor import for_episode
    owner = (ctx.get('piece') or {}).get('of') or (ctx.get('data_dictionary') or {}).get('episode_id') or ep_dir.name
    dictionary = for_episode(ep_dir, owner)
    dictionary = dictionary if dictionary is not None else ctx.get('data_dictionary') or {}
    meanings = {name: {key: entry[key] for key in ('role', 'meaning', 'layout') if key in entry}
                for name, entry in (dictionary.get('entries') or {}).items() if isinstance(entry, dict)}
    from prepare import jaws
    jaw_algorithm = hashlib.sha256(Path(jaws.__file__).read_bytes()).hexdigest() if any(
        isinstance(camera, dict) and camera.get('jaw_geometry') for camera in (ctx.get('cameras') or {}).values()) else None
    examples = []
    if example_dir:
        from label import content_identity
        for path in sorted(Path(example_dir).glob("example_*.json")):
            # by content: an example copied with new times is the same example
            examples.append({"name": path.name, **content_identity.content(content_identity.identity(path))})
    repository = Path(__file__).parent.parent
    request_sources = request_implementation(repository)
    semantics = hashlib.sha256()
    for path in sorted(request_sources):
        semantics.update(str(path.relative_to(repository)).encode() + b'\0' + request_sources[path] + b'\0')
    settings = {"model": resolved["model"], "reasoning": resolved["reasoning"], "max_tokens": max_tokens, "cell_w": cell_w,
                "detail": DETAIL, "grid_cols": GRID_COLS, "examples": examples,
                "annotation_policy": resolved, "jaw_algorithm_sha256": jaw_algorithm,
                "interpretation_sha256": hashlib.sha256(json.dumps(meanings, sort_keys=True,
                    separators=(',', ':'), ensure_ascii=True).encode()).hexdigest(),
                "request_semantics_sha256": semantics.hexdigest(),
                "reader_projection_sha256": reader_projection_identity(ctx)}
    if request_body_sha256() != FIRST_BODY_SHA256:
        settings["request_body_sha256"] = request_body_sha256()
    return {"version": 2, "source_proof": source_proof(ctx, ep_dir), "settings": settings}


def _content_settings(settings):
    """Settings with each example by its content alone (an earlier identity also kept its modification time)."""
    from label import content_identity
    if isinstance(settings, dict) and isinstance(settings.get('examples'), list):
        settings = {**settings, 'examples': [content_identity.content(e) if isinstance(e, dict) else e
                                             for e in settings['examples']]}
    return settings


def same_input_identity(saved: dict, current: dict) -> bool:
    from label.evidence_access import equivalent_source_proof
    return (isinstance(saved, dict) and saved.get('version') == current.get('version')
            and _content_settings(saved.get('settings')) == _content_settings(current.get('settings'))
            and equivalent_source_proof(saved.get('source_proof'), current.get('source_proof')))


def saved_input_problem(record: dict, ep_dir: Path) -> str | None:
    """Check proven source inputs before republishing a saved reply, without regrading historical labels."""
    identity = record.get('input_identity')
    if identity is None:
        return None
    if not isinstance(identity, dict) or identity.get('version') not in (1, 2):
        return 'saved output input proof cannot be verified'
    from label.evidence_access import same_source_proof
    try:
        ctx = json.loads((Path(ep_dir) / 'context.json').read_text())
        if not isinstance(ctx, dict) or not same_source_proof(identity.get('source_proof'), ctx, ep_dir):
            return 'saved output has changed or missing input'
    except (OSError, ValueError, KeyError, TypeError):
        return 'saved output has changed or missing input'
    return None


def _receipt_snapshot(out_dir: Path, name: str) -> list:
    """Read and validate each candidate once for a consistent annotation and billing selection."""
    records = []
    for path in (Path(out_dir) / name, Path(out_dir) / f'failed_{name}', Path(out_dir) / f'noreply_{name}'):
        if not path.exists():
            continue
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            raise ValueError(f'{path.name}: saved billing receipt cannot be read: {error}') from error
        if not isinstance(record, dict):
            raise ValueError(f'{path.name}: saved billing receipt is not an object')
        index = record.get('final_attempt_index', 0)
        if type(index) is not int or index < 0:
            raise ValueError(f'{path.name}: saved final attempt index is invalid')
        records.append((path, record))
    return records


def billing_record(out_dir: Path, name: str):
    """Select the latest durable paid attempt without discarding earlier label files."""
    return max(_receipt_snapshot(out_dir, name), key=lambda item: item[1].get('final_attempt_index', 0), default=None)


def annotation_record(out_dir: Path, name: str):
    """Select final reply authority separately from the latest cumulative financial ledger.

    A local refusal cannot supersede a received reply or an unresolved dispatch. Indexed
    histories retain authority when a later local receipt replaced that dispatch's file.
    Equal-index files preserve primary, failed, no-reply legacy precedence.
    """
    records = _receipt_snapshot(out_dir, name)
    latest = max(records, key=lambda item: item[1].get('final_attempt_index', 0), default=None)
    if latest is None:
        return None
    selected = None
    for path, record in records:
        if record.get('final_dispatch_outcome') == 'not dispatched':
            continue
        if selected is None or record.get('final_attempt_index', 0) > selected[1].get('final_attempt_index', 0):
            selected = path, record
    for history_index, invoice in enumerate(latest[1].get('final_cost_history', [])):
        index = invoice.get('final_attempt_index', 0)
        if type(index) is not int or index < 0:
            raise ValueError('saved final attempt index is invalid')
        if invoice.get('final_dispatch_outcome') == 'not dispatched':
            continue
        if selected is None or index > selected[1].get('final_attempt_index', 0):
            prefix = ('failed_' if invoice.get('finish_reason') == 'length' else
                      'noreply_' if invoice.get('final_dispatch_outcome') in ('claimed', 'unverified') else '')
            selected = (Path(out_dir) / (prefix + name),
                        {'episode_dir': latest[1].get('episode_dir'), **copy.deepcopy(invoice),
                         '_authority_history_index': history_index})
    selected = selected or latest
    if selected[1] != latest[1]:
        selected = selected[0], {**selected[1], '_billing_receipt': latest[1]}
    return selected


def billing_view(record: dict) -> dict:
    """The financial receipt attached to an annotation selected from an older attempt."""
    return record.get('_billing_receipt', record)


def _history_cost(invoice: dict) -> float:
    if not isinstance(invoice, dict):
        raise ValueError('saved final invoice is not an object')
    outcome = invoice.get('final_dispatch_outcome')
    if outcome == 'not dispatched' or invoice.get('dry_run'):
        return 0.0
    value = invoice.get('final_reserved_usd') if outcome in ('claimed', 'unverified') else (
        invoice.get('usage') or {}).get('est_cost_usd')
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('saved final invoice cost is not verified')
    return float(value)


def final_history_cost(record: dict) -> float:
    history = record.get('final_cost_history', [])
    if not isinstance(history, list):
        raise ValueError('saved final invoice history is invalid')
    return sum(_history_cost(invoice) for invoice in history)


def final_history_complete(record: dict) -> bool:
    final_history_cost(record)
    return all(invoice.get('final_dispatch_outcome') not in ('claimed', 'unverified')
               for invoice in record.get('final_cost_history', []))


def auxiliary_cost(value: dict, *, routing=False) -> float:
    cost = value.get('cost_usd', 0.0)
    previous = cost if routing and value.get('previously_billed') is True else value.get('previously_billed_usd', 0.0)
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in (cost, previous)) or previous > cost:
        raise ValueError('saved auxiliary invoice cost is invalid')
    return float(cost - previous)


def auxiliary_history_cost(record: dict) -> float:
    history = record.get('auxiliary_cost_history', [])
    if not isinstance(history, list) or any(not isinstance(invoice, dict) for invoice in history):
        raise ValueError('saved auxiliary invoice history is invalid')
    return sum(auxiliary_cost(invoice) for invoice in history)


def auxiliary_history_complete(record: dict) -> bool:
    auxiliary_history_cost(record)
    return not any(invoice.get('cost_is_conservative_estimate') for invoice in record.get('auxiliary_cost_history', []))


def inspection_billed(record: dict, digest) -> float:
    """Inspection caches are cumulative; count only their costs not already represented in this receipt."""
    if not digest:
        return 0.0
    auxiliary_history_cost(record)
    cost = sum(auxiliary_cost(invoice) for invoice in record.get('auxiliary_cost_history', [])
               if invoice.get('kind') == 'inspection' and invoice.get('digest') == digest)
    active = record.get('evidence_inspection') or {}
    if active.get('digest') == digest:
        cost += auxiliary_cost(active)
    return cost


def _final_cost_history(out_path: Path) -> list:
    saved = billing_record(Path(out_path).parent, Path(out_path).name)
    if saved is None:
        return []
    record = saved[1]
    final_history_cost(record)
    history = copy.deepcopy(record.get('final_cost_history', []))
    if record.get('final_dispatch_outcome') == 'not dispatched' or record.get('dry_run'):
        return history
    _history_cost(record)
    # Retain the full attempt for offline repair, without recursively copying its histories.
    history.append({key: copy.deepcopy(value) for key, value in record.items()
                    if key not in ('final_cost_history', 'auxiliary_cost_history') and not key.startswith('_')})
    return history


def _billing_fields(out_path: Path) -> dict:
    final = _final_cost_history(out_path)
    current = billing_record(Path(out_path).parent, Path(out_path).name)
    history = []
    def retain(kind, value, cost):
        if cost:
            history.append({'kind': kind, 'cost_usd': cost,
                'rounds_cost_offset_usd': float(value.get('rounds_cost_offset_usd') or 0)
                    + max(0.0, float(value.get('cost_usd', 0)) - cost),
                **{key: copy.deepcopy(value[key]) for key in (
                    'model', 'generation_id', 'usage', 'cost_source', 'dispatch_outcome',
                    'reserved_usd', 'digest', 'rounds', 'cost_is_conservative_estimate') if key in value}})
    if current is not None:
        record = current[1]
        auxiliary_history_cost(record)
        history = copy.deepcopy(record.get('auxiliary_cost_history', []))
        for kind, value in (('route', (record.get('config') or {}).get('resolution_route') or {}),
                            ('inspection', record.get('evidence_inspection') or {})):
            cost = auxiliary_cost(value, routing=kind == 'route')
            retain(kind, value, cost)
    cache = Path(out_path).parent / '.evidence' / (Path(out_path).name + '.inspection.json')
    if cache.exists():
        from label.evidence_access import recover_selection_reservations
        inspection = json.loads(cache.read_text())
        recover_selection_reservations(inspection)
        carried = inspection_billed({'auxiliary_cost_history': history}, inspection.get('digest'))
        retain('inspection', inspection, max(0.0, inspection['cost_usd'] - carried))
    return {'final_cost_history': final, 'auxiliary_cost_history': history,
            'final_attempt_index': current[1].get('final_attempt_index', 0) + 1 if current else 1}


def episode_cost(result: dict) -> float:
    """What labelling one episode was billed: its model call plus, for a routed episode, the routing call made for
    it (label/route.py)."""
    result = billing_view(result)
    route = (result.get("config") or {}).get("resolution_route") or {}
    inspection = result.get('evidence_inspection') or {}
    final = (result.get("usage") or {}).get("est_cost_usd")
    if final is None:
        final = result.get('final_reserved_usd') or 0.0
    return (float(final) + final_history_cost(result) + auxiliary_history_cost(result)
            + auxiliary_cost(route, routing=True) + auxiliary_cost(inspection))


def final_cost_bound(content: list, max_tokens: int, *, model=DEFAULT_MODEL, require_known_pricing=True,
                     pricing_quote=None) -> float:
    """Reserve a generous input allowance and the full output limit before dispatch."""
    if pricing_quote is not None:
        if pricing_quote.get('model') != model:
            raise ValueError('pricing quote does not match the requested model')
        from label import pricing
        return pricing.cost_bound(content, max_tokens, pricing_quote)
    tiers = OPENAI_PRICES.get(model.split('/', 1)[-1]) if model.startswith('openai/') else None
    if tiers is None and require_known_pricing:
        raise ValueError(f'No verified pricing rates for {model}; a spend cap requires known rates')
    # Uncapped custom models retain a provisional hold if their provider loses the cost receipt.
    # This fallback is never used to authorize a capped dispatch.
    rates = tiers or OPENAI_PRICES[DEFAULT_MODEL.split('/', 1)[-1]]
    import io
    from PIL import Image
    text_bytes = sum(len(c.get('text', '').encode('utf-8')) for c in content if c.get('type') == 'text')
    images = 0.0
    for part in content:
        if part.get('type') == 'image_url':
            raw = base64.b64decode(part['image_url']['url'].split(',', 1)[1])
            with Image.open(io.BytesIO(raw)) as image:
                images += estimate_image_tokens(*image.size)
    # an upper bound on the prompt's tokens (a token is at least one byte, images at four times their estimate)
    prompt_bound = text_bytes + 4 * images + 1024
    # The long-context rates apply only from LONG_PROMPT_TOKENS prompt tokens, so a prompt bounded below that is
    # billed at the standard rates, input and output alike. Reserving the long rates for it held half again too much.
    if prompt_bound < LONG_PROMPT_TOKENS:
        price_in, price_out = rates[0][0], rates[0][2]
    else:
        price_in, price_out = max(t[0] for t in rates), max(t[2] for t in rates)
    return math.ceil((prompt_bound * price_in + max_tokens * price_out) * 1_000_000) / 1_000_000


def persist_final_claim(path: Path, claim: dict) -> None:
    """Make the claim durable before a network dispatch can consume money."""
    write_atomic(path, claim)
    with path.open('rb') as stream:
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def has_dispatch_receipt(record: dict) -> bool:
    """Whether saved interpretation is backed by a received or uncertain dispatch."""
    if record.get('dry_run'):
        return False
    if record.get('final_dispatch_outcome') != 'not dispatched':
        return True
    if record.get('response') is not None or record.get('response_text') is not None:
        return True
    if record.get('final_cost_history') or record.get('auxiliary_cost_history'):
        return True
    route = (record.get('config') or {}).get('resolution_route') or {}
    if route.get('dispatch_outcome') in ('claimed', 'unverified', 'received', 'cached'):
        return True
    inspection = record.get('evidence_inspection') or {}
    if inspection.get('rounds') or inspection.get('cost_is_conservative_estimate'):
        return True
    return episode_cost(record) != 0


def record_budget_refusal(path: Path, request: dict) -> bool:
    """Record an unattempted final annotation without replacing any paid receipt."""
    path = Path(path)
    if billing_record(path.parent, path.name) is not None:
        return False
    if request.get('dry_run') is not True:
        raise ValueError('a budget refusal needs the verified dry request')
    fields = {key: request[key] for key in ('episode_dir', 'model', 'reasoning_effort',
              'input_identity', 'annotation_policy') if key in request}
    write_atomic(path.with_name('noreply_' + path.name), {
        **fields, **_billing_fields(path), 'parse_ok': False,
        'final_dispatch_outcome': 'not dispatched', 'final_reserved_usd': 0.0,
        'no_reply_kind': 'spend_cap', 'no_reply': 'No remaining spend allowance for the final annotation.'})
    return True


def estimate_image_tokens(w: int, h: int) -> float:
    """Input tokens per image, fitted on 2,572 gpt-6-astra requests (about 1.1 tokens per 32x32 px patch plus 69
    per image). Used only for the dry-run estimate; billing always comes from the API's own usage."""
    return 68.8 + 1.115 * (-(-w // 32)) * (-(-h // 32))


TIMELINE_COLUMNS = ["start_s", "end_s", "arm", "action", "object", "destination", "spatial_relation",
                    "contribution", "progress", "notes"]


def _model_diagnostics(labels: dict) -> None:
    """Only the parser may author diagnostic provenance, including local assessments."""
    supplied = [key for key in ('_dropped', '_schema_violations', '_schema_row_indices') if key in labels]
    for key in supplied:
        labels.pop(key)
    for key in supplied:
        _drop(labels, key, None, 'reserved parser metadata supplied in the reply')
    assessments = labels.get('part_assessments')
    for assessment in assessments if isinstance(assessments, list) else []:
        if isinstance(assessment, dict):
            _model_diagnostics(assessment)


def parse_response(text: str) -> tuple[dict, bool]:
    """A model's reply as labels, or the raw text and the reason it could not be read. The one parser for every
    model and for re-reading stored replies (label/reparse.py). A reply that is a JSON object but breaks the output
    format in places keeps the rest of its labels (normalize_timeline, typed_labels)."""
    try:
        # some models ignore json_object and wrap the JSON in a markdown fence
        fenced = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", text, re.S)
        labels = json.loads(fenced.group(1) if fenced else text)
        if not isinstance(labels, dict):
            raise ValueError(f"the reply is {json_kind(labels)}, not an object")
        _model_diagnostics(labels)
        had_rows = bool(labels.get('timeline'))
        labels = typed_labels(normalize_timeline(labels))
        derive_instruction_mismatch(labels)
        completion = labels.get('completion')
        if isinstance(completion, dict) and completion.get('task_completed') is not None and completion['task_completed'] not in (
                'success', 'success_then_undone', 'failure', 'partial', 'unclear'):
            outcome = completion['task_completed']
            if isinstance(outcome, str) and outcome.strip().lower() in (
                    'success', 'success_then_undone', 'failure', 'partial', 'unclear'):
                completion['task_completed'] = outcome.strip().lower()
            else:
                _violation(labels, 'completion.task_completed has an unknown outcome')
        for index, row in enumerate(labels.get('timeline') or []):
            progress = row.get('progress')
            if isinstance(progress, str) and me.number(progress) is not None:
                progress = row['progress'] = me.number(progress)
            if progress is not None and (type(progress) not in (int, float) or not math.isfinite(progress)
                                         or not 0 <= progress <= 1):
                _violation(labels, f'timeline row {index}: progress {progress!r} must be a finite number from 0 to 1')
        if had_rows and not labels.get('timeline') and not any(labels.get(k) for k in LIST_FIELDS[1:] + DICT_FIELDS + TEXT_FIELDS):
            raise ValueError('no usable timeline rows remain')
        if not substantive_annotation(labels):
            raise ValueError('the reply contains no substantive annotation content')
        return labels, True
    except Exception as e:  # noqa: BLE001 - any reply that does not parse is kept raw, with why
        return {"_raw": text, "_parse_error": f"{type(e).__name__}: {e}"[:300]}, False


def derive_instruction_mismatch(labels: dict) -> None:
    """The prompt's rule that a reply whose goal_alignment says the work is not the asked work also reports it as an
    instruction_mismatch data issue (label/prompts.py INSTRUCTION_RULES), held in code.
    When goal_alignment has relation different or unrelated, or matches_given false, and no data issue has the
    category instruction_mismatch, one is added from the goal_alignment note, marked derived_from goal_alignment. A
    reply that already reports one keeps it as it is. Running this again on its output changes nothing."""
    alignment = labels.get('goal_alignment')
    if not isinstance(alignment, dict):
        return
    relation = alignment.get('relation')
    if relation not in ('different', 'unrelated') and alignment.get('matches_given') is not False:
        return
    issues = labels.get('data_issues')
    if issues is not None and not isinstance(issues, list):
        return
    if any(isinstance(row, dict) and row.get('category') == 'instruction_mismatch' for row in issues or []):
        return
    note = alignment.get('note')
    what = note.strip() if isinstance(note, str) and note.strip() else (
        f"The recorded work does not match the instruction (goal_alignment relation {relation}, "
        f"matches_given {str(alignment.get('matches_given')).lower()}).")
    labels['data_issues'] = list(issues or []) + [{
        'issue': what, 'category': 'instruction_mismatch', 'severity': 'high', 't_s': None, 'end_s': None,
        'evidence': 'goal_alignment of this reply', 'derived_from': 'goal_alignment'}]


def json_kind(x) -> str:
    """What a JSON value is, in the output format's words, for the board's reason a field was left out."""
    if isinstance(x, bool):
        return "true or false"
    return ("null" if x is None else "a number" if isinstance(x, (int, float)) else "text" if isinstance(x, str)
            else "a list" if isinstance(x, list) else "an object" if isinstance(x, dict) else type(x).__name__)


def _drop(labels: dict, field: str, row, why: str) -> None:
    """Records a field or a row of the reply left out because it breaks the output format."""
    labels.setdefault("_dropped", []).append({"field": field, **({"row": row} if row is not None else {}), "why": why})


def _violation(labels: dict, why: str) -> None:
    violations = labels.setdefault('_schema_violations', [])
    if why not in violations:
        violations.append(why)


def normalize_timeline(labels: dict) -> dict:
    """The output format sends each timeline segment as one array in the column order of timeline_columns.
    Convert it back to one object per segment, validating every row, so everything downstream sees the usual
    format. A time or progress written as numeric text ("2.0") is read as that number. Shortened rows retain
    only field assignments shared by every compatible column mapping; ambiguous values are preserved in
    _dropped. Rows with unparseable times are omitted. Full-width rows retain malformed noncritical fields
    with schema violations. A value that is missing or outside a
    field's allowed set (a progress or a contribution of null) is kept as the model wrote it and listed in
    _schema_violations, so one bad field is counted, not a reason to discard a valid answer."""
    tl = labels.get("timeline")
    if not isinstance(tl, list) or not any(isinstance(r, list) for r in tl):
        labels.pop("timeline_columns", None)
        return labels
    cols = labels.pop("timeline_columns", None)
    if cols is None:
        cols = TIMELINE_COLUMNS
    if (not isinstance(cols, list) or not cols or any(not isinstance(c, str) or not c for c in cols)
            or len(set(cols)) != len(cols)):
        for i, row in enumerate(tl):
            if isinstance(row, list):
                _drop(labels, 'timeline', i, 'timeline columns must be unique nonempty names')
        labels['timeline'] = [r for r in tl if isinstance(r, dict)]
        return labels
    out, source_rows = [], []
    first_diagnostic = len(labels.get('_schema_violations') or [])
    for i, row in enumerate(tl):
        if isinstance(row, dict):
            out.append(row)
            source_rows.append(i)
            continue
        missing_notes = isinstance(row, list) and cols[-1] == 'notes' and len(row) == len(cols) - 1
        if not isinstance(row, list) or (len(row) != len(cols) and not missing_notes):
            n = len(row) if isinstance(row, list) else "?"
            _drop(labels, "timeline", i, f"it has {n} values for {len(cols)} columns")
            continue
        seg = dict(zip(cols, row))
        bad = None
        for key in ('arm', 'action', 'object', 'destination', 'spatial_relation', 'contribution', 'hands_wearing', 'notes'):
            if seg.get(key) is not None and not isinstance(seg[key], str):
                why = f'its {key} is not text ({seg[key]!r})'
                if missing_notes:
                    bad = why
                    break
                _violation(labels, f'timeline row {i}: {why}')
        if 'hands_visible' in seg and seg['hands_visible'] is not None and type(seg['hands_visible']) is not bool:
            why = f'its hands_visible is not true or false ({seg["hands_visible"]!r})'
            if missing_notes:
                bad = why
            else:
                _violation(labels, f'timeline row {i}: {why}')
        if missing_notes and not bad:
            def compatible(candidate):
                for key, value in candidate.items():
                    if value is None and key not in ('start_s', 'end_s'):
                        continue
                    if key in ('start_s', 'end_s', 'progress'):
                        if isinstance(value, bool) or me.number(value) is None:
                            return False
                    elif key == 'hands_visible':
                        if type(value) is not bool:
                            return False
                    elif key in ('arm', 'action', 'object', 'destination', 'spatial_relation',
                                 'contribution', 'hands_wearing', 'notes') and not isinstance(value, str):
                        return False
                return True
            supplied = {key: value for key, value in seg.items() if value is not None}
            alternatives = [dict(zip(cols[:gap] + cols[gap + 1:], row)) for gap in range(len(cols) - 1)]
            candidates = [seg] + [candidate for candidate in alternatives if compatible(candidate)]
            if any({key: value for key, value in candidate.items() if value is not None} != supplied
                   for candidate in candidates):
                # Keep only fields proved by every compatible column mapping.
                # Free text cannot establish whether it is handwear or a note.
                seg = {key: value for key, value in seg.items()
                       if all(key in candidate and candidate[key] == value for candidate in candidates)}
                _drop(labels, 'timeline.columns', i, 'ambiguous shortened row; only unambiguous fields are shown')
                labels['_dropped'][-1]['value'] = row[:]
        for k in ("start_s", "end_s", "progress"):
            if bad:
                break
            if k == "progress" and k in seg and seg[k] is None:
                labels.setdefault("_schema_violations", []).append(f"timeline row {i}: progress null")
            elif k in seg and (isinstance(seg[k], bool) or not isinstance(seg[k], (int, float))):
                v = me.number(seg[k])
                if v is None:
                    if k == 'progress' and not missing_notes:
                        _violation(labels, f'timeline row {i}: progress {seg[k]!r} must be a finite number from 0 to 1')
                        continue
                    bad = f"its {k} is not a number ({seg[k]!r})"
                    break
                seg[k] = v
        if bad:
            _drop(labels, "timeline", i, bad)
            continue
        if 'contribution' in seg and str(seg.get("contribution") or '').strip().lower() not in ("advancing", "wasteful", "idle"):
            labels.setdefault("_schema_violations", []).append(
                f"timeline row {i}: contribution {seg.get('contribution')!r}")
        if seg.get("notes") in (None, ""):
            seg.pop("notes", None)
        out.append(seg)
        source_rows.append(i)
    if len(out) != len(tl):
        mappings = labels.get('_schema_row_indices')
        labels['_schema_row_indices'] = {**(mappings if isinstance(mappings, dict) else {}),
                                         'timeline': source_rows}
        diagnostics = labels.get('_schema_violations') or []
        for index in range(first_diagnostic, len(diagnostics)):
            match = re.match(r'^timeline row (\d+): (?:its )?(\w+)(.*)', diagnostics[index])
            if match:
                diagnostics[index] = f'timeline[{match[1]}].{match[2]}:{match[3]}'
    labels["timeline"] = out
    return labels


# the reply's fields by the type the output format gives them (label/prompts.py)
LIST_FIELDS = ("timeline", "key_events", "state_changes", "scene_graph", "recovery", "data_issues", "operator_mistakes",
               "tasks", "contacts", "contacts_missing", "sensor_findings", "evidence_findings", "camera_observations")
DICT_FIELDS = ("scene", "completion", "goal_alignment", "goal_outcome")
TEXT_FIELDS = ("task_summary", "performance_review", "viewpoint")
TIME_FIELDS = ("t_s", "start_s", "end_s", "completed_at_s", "goal_reached_at_s", "undone_at_s", "failure_t_s",
               "recovered_at_s")


def substantive_annotation(labels: dict) -> bool:
    """Require an observed action, task, result, description, or quality finding from a supported schema.

    Partial replies remain readable. Empty containers, time-only rows, and classification metadata
    cannot establish that an episode was annotated.
    """
    def text(row, key):
        return isinstance(row, dict) and isinstance(row.get(key), str) and bool(row[key].strip())

    if any(text(labels, key) for key in ('task_summary', 'performance_review')):
        return True
    content_fields = {'timeline': 'action', 'tasks': 'task', 'key_events': 'label',
                      'data_issues': 'issue', 'operator_mistakes': 'issue', 'recovery': 'failure'}
    if any(text(row, key) for field, key in content_fields.items() for row in labels.get(field) or []):
        return True
    completion = labels.get('completion') or {}
    if completion.get('task_completed') in ('success', 'success_then_undone', 'failure', 'partial', 'unclear'):
        return True
    alignment = labels.get('goal_alignment') or {}
    if (type(alignment.get('matches_given')) is bool
            and alignment.get('relation') in ('aligned', 'narrower', 'broader', 'different', 'unrelated')):
        return True
    scene = labels.get('scene') or {}
    if text(scene, 'setting') or any(text(row, 'name') for row in scene.get('objects') or []):
        return True
    if any(text(row, 'object') and text(row, 'from') and text(row, 'to')
           for row in labels.get('state_changes') or []):
        return True
    if any(isinstance(row.get('relations'), list) and any(isinstance(value, str) and value.strip()
               for value in row['relations']) for row in labels.get('scene_graph') or []):
        return True
    if any(text(row, 'headline') and isinstance(row.get('evidence'), list) and row['evidence']
           for field in ('sensor_findings', 'evidence_findings') for row in labels.get(field) or []):
        return True
    if any(text(row, 'view') and text(row, 'evidence') and row.get('content') in ('scene', 'sensor', 'unclear')
           for row in labels.get('camera_observations') or []):
        return True
    return any(text(row, 'object') for field in ('contacts', 'contacts_missing') for row in labels.get(field) or [])


def reply_truncated(record: dict) -> bool:
    """A syntactically parsed fragment still lacks the provider's completed reply."""
    return record.get('finish_reason') == 'length'


def read_reply(text: str, request: dict) -> tuple[dict, bool]:
    """A model's reply as labels, judged against the request it answers: parse_response, then whether it holds the
    work the request asked for (label/completeness.py). A reply that parses but lacks some of that work keeps its
    labels, with what it lacks under _incomplete, and is not a usable annotation (False). request is the record or
    the request fields it is built from."""
    labels, ok = parse_response(text)
    if ok and request.get('label_schema') == LABEL_SCHEMA:
        # a reply asked for a list of tasks: the recording's completion is the reply's own, against the instruction,
        # when the request gave one to a robot rig, and otherwise derived from the tasks
        selected = request.get('annotation_policy') if isinstance(request.get('annotation_policy'), dict) else {}
        given = request.get('given_prompt')
        labels = derive_completion(labels, rig=selected.get('rig') or 'teleop_arms',
                                   instruction=isinstance(given, str) and bool(given.strip()),
                                   task_prompts=request.get('task_prompts') or 0)
    if ok:
        from label.completeness import annotation_gaps
        gaps = annotation_gaps(labels, request)
        if gaps:
            labels['_incomplete'] = gaps
            return labels, False
    return labels, ok


def annotation_problems(record: dict) -> list[str]:
    """Why a saved record is not a finished annotation, or [] when it is: a reply cut off at the output limit, one
    that did not parse, or one without the work its request asked for, judged again with the current parser and
    label/completeness.py, never from the saved parse_ok alone. The one judgement every site that calls a unit done,
    reusable or skippable reads (annotation_complete)."""
    if reply_truncated(record):
        return ["the model's reply was cut off at the output limit"]
    labels = record.get('labels')
    if not isinstance(labels, dict):
        return [f"the model gave no reply ({record['no_reply']})" if record.get('no_reply')
                else 'its labels are not a JSON object']
    if record.get('parse_ok') is not True:
        return list(labels.get('_incomplete') or []) or [str(labels.get('_parse_error') or record.get('no_reply')
                                                             or "the model's reply did not parse")]
    checked, ok = read_reply(json.dumps({k: v for k, v in labels.items() if k != '_incomplete'}), record)
    return [] if ok else list(checked.get('_incomplete') or []) or [str(checked.get('_parse_error'))]


def annotation_complete(record: dict) -> bool:
    """Whether a saved record is a finished annotation (annotation_problems)."""
    return not annotation_problems(record)


def _times(x: dict, where: str, labels: dict) -> None:
    """The time fields of one row or object as numbers: a number in text ("12.5", "12.5s") is that number, and a time
    that is no number ("late", NaN) is null, listed in _schema_violations, so the row is kept and shown untimed."""
    for k in TIME_FIELDS:
        v = x.get(k)
        if v is None or (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)):
            continue
        x[k] = me.number(v)
        if x[k] is None:
            labels.setdefault("_schema_violations", []).append(f"{where}: {k} {v!r} is not a time, kept untimed")


def _rows(labels: dict, field: str, rows: list, kind=dict, what: str = "an object") -> list:
    """The rows of a list field of this kind; each other row is recorded in _dropped."""
    if all(isinstance(r, kind) for r in rows):
        return rows
    for i, r in enumerate(rows):
        if not isinstance(r, kind):
            _drop(labels, field, i, f"{json_kind(r)}, not {what}")
    return [r for r in rows if isinstance(r, kind)]


def typed_labels(labels: dict) -> dict:
    """The reply with every field of the type the output format gives it. A list field that is not a list, an object
    field that is not an object or a text field that is not text is left out, and so is a row of a list that is not
    an object (a key event written as a plain string), an instruction variant that is not text and an entry of
    scene.objects that is not an object; each is recorded in _dropped ({"field", "row", "why"}), which the board counts
    and shows, and the rest of the reply is kept. Times are read as numbers (_times). A reply that keeps to the format
    comes back unchanged, and running this again on its output changes nothing."""
    for fields, kind, what in ((LIST_FIELDS + ("instruction_variants",), list, "a list"),
                               (DICT_FIELDS, dict, "an object"), (TEXT_FIELDS, str, "text")):
        for k in fields:
            if k in labels and labels[k] is not None and not isinstance(labels[k], kind):
                _drop(labels, k, None, f"{json_kind(labels[k])}, not {what}")
                labels.pop(k)
    if isinstance(labels.get("instruction_variants"), list):
        labels["instruction_variants"] = _rows(labels, "instruction_variants", labels["instruction_variants"], str,
                                               "text")
    for k in LIST_FIELDS:
        if isinstance(labels.get(k), list):
            labels[k] = _rows(labels, k, labels[k])
            for i, r in enumerate(labels[k]):
                _times(r, f"{k} row {i}", labels)
    for k in ("completion", "goal_alignment", "goal_outcome"):
        if isinstance(labels.get(k), dict):
            _times(labels[k], k, labels)
    sc = labels.get("scene")
    if isinstance(sc, dict) and sc.get("objects") is not None:
        if not isinstance(sc["objects"], list):
            _drop(labels, "scene.objects", None, f"{json_kind(sc['objects'])}, not a list")
            sc.pop("objects")
        else:
            sc["objects"] = _rows(labels, "scene.objects", sc["objects"])
    return labels


def camera_observation_source(proof, views) -> bool:
    """Require native packed-camera identities before a model view can drive presentation."""
    if (not isinstance(proof, list) or not proof or not views
            or any(not isinstance(item, dict) or item.get('missing') for item in proof)):
        return False
    for view in views:
        matches = [item for item in proof if item.get('name') == f'rgb:{view}:packed']
        if (len(matches) != 1 or not matches[0].get('location_sha256')
                or not matches[0].get('size') or not (matches[0].get('sha256') or matches[0].get('mtime_ns'))):
            return False
    return True


def bind_camera_observations(rows, fields: dict) -> dict:
    """Bind exact request IDs or unambiguous recorded captions to immutable native inputs."""
    from collections import Counter
    config = fields.get('config') or {}
    views = config.get('camera_observation_views') or []
    proof = (fields.get('input_identity') or {}).get('source_proof')
    supplied = (isinstance(views, list) and len(views) > 1 and all(isinstance(v, str) for v in views)
                and len(views) == len(set(views)) and all(v in (config.get('views') or []) for v in views)
                and 'camera_observations' in (config.get('schema_fields') or []))
    native_verified = supplied and camera_observation_source(proof, views)
    has_labels = 'camera_observation_labels' in config
    displayed = config.get('camera_observation_labels')
    labels_valid = (not has_labels or (supplied and isinstance(displayed, dict)
                    and set(displayed) == set(views)
                    and all(isinstance(label, str) and label.strip() for label in displayed.values())))
    result = {'version': 1, 'views': list(views) if isinstance(views, list) else [],
              'source_proof': copy.deepcopy(proof), 'observations': [], 'withheld_observations': []}
    if has_labels:
        result.update(displayed_labels=copy.deepcopy(displayed), reported_observations=copy.deepcopy(rows),
                      address_resolutions=[])

    def resolve(value):
        if not isinstance(value, str) or not labels_valid or not supplied:
            return None
        matches = [view for view, label in displayed.items() if label == value] if has_labels else []
        if value in views:
            return None if any(view != value for view in matches) else value
        return matches[0] if len(matches) == 1 else None

    addressed = [(row, resolve(row.get('view')) if isinstance(row, dict) else None)
                 for row in rows] if isinstance(rows, list) else [(rows, None)]
    counts = Counter(view for _, view in addressed if view is not None)
    for row, view in addressed:
        why = None
        if not supplied:
            why = 'Camera observations were not requested with exact supplied view IDs.'
        elif not native_verified:
            why = 'Native camera source identities are missing or cannot be verified.'
        elif not labels_valid:
            why = 'Recorded camera display labels do not match the supplied view IDs.'
        elif view is None:
            why = 'The observation does not name an exact supplied ID or unique recorded display label.'
        elif counts[view] != 1:
            why = 'More than one observation names this view; no observation is selected.'
        elif row.get('content') not in ('scene', 'sensor', 'unclear') or not isinstance(row.get('evidence'), str) or not row['evidence'].strip():
            why = 'The observation needs a known content class and visible evidence.'
        if why:
            result['withheld_observations'].append({'observation': copy.deepcopy(row), 'why': why})
        else:
            result['observations'].append({'view': view, 'content': row['content'], 'evidence': row['evidence']})
            if has_labels:
                result['address_resolutions'].append({'reported_view': row['view'], 'view': view,
                    'basis': 'stable_id' if row['view'] == view else 'displayed_label'})
    if (result['withheld_observations'] or not supplied
            or {row['view'] for row in result['observations']} != set(views)):
        result['withheld_observations'].extend({'observation': row,
            'why': 'Camera interpretation requires one valid observation for every supplied view ID.'}
            for row in result['observations'])
        result['observations'] = []
        missing = [view for view in result['views'] if isinstance(view, str) and counts.get(view) != 1]
        if missing:
            result['withheld_observations'].append({'views': missing,
                'why': 'Supplied view IDs are missing or do not have exactly one observation.'})
    return result


def camera_observation_sources_match(parent, child, views) -> bool:
    """A request part's view IDs must still name the parent's native camera files."""
    from label.evidence_access import equivalent_source_proof
    if not camera_observation_source(parent, views) or not camera_observation_source(child, views):
        return False
    names = {f'rgb:{view}:packed' for view in views}
    return equivalent_source_proof([p for p in parent if p['name'] in names],
                                   [p for p in child if p['name'] in names])


def _call_and_record(ep_dir: Path, out_path: Path, content: list, img_bytes: int, fields: dict, *, model: str,
                     reasoning: str, api_key: str, max_tokens: int, timeout: int, dry_run: bool = False,
                     prompt_text: str = "", evidence_access=None, reserve_final=None,
                     reserve_selection=None, settle_selection=None, settle_final=None, force_refresh=False,
                     require_known_pricing=False, inspection_refresh_id=None, pricing_quote=None) -> dict:
    """Send one request (or, in a dry run, record exactly what would be sent) and write the result."""
    fields = {**fields, **(_billing_fields(out_path) if 'final_cost_history' not in fields else {})}
    def verify_inputs():
        saved = fields.get('input_identity')
        if saved is None:
            return  # Raw request callers may have no prepared-source identity.
        try:
            settings = saved['settings']
            selected = settings['annotation_policy']
            current = input_identity(ep_dir, model=model, reasoning=reasoning, max_tokens=max_tokens,
                cell_w=settings['cell_w'], example_dir=fields.get('example_dir'),
                sample_fps=selected.get('sample_fps'), task_mode=selected.get('task_mode'),
                dataset_note=selected.get('dataset_note'),
                inspection=policy.inspection_mode(selected))
            valid = same_input_identity(saved, current) and not any(item.get('missing') for item in current['source_proof'])
        except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError):
            valid = False
        if not valid:
            reason = 'Request inputs changed during assembly or inspection; no final annotation was dispatched.'
            write_atomic(Path(out_path).with_name(f'noreply_{Path(out_path).name}'), {
                'episode_dir': str(ep_dir), 'model': model, 'reasoning_effort': reasoning, **fields,
                'parse_ok': False, 'no_reply': reason, 'final_dispatch_outcome': 'not dispatched',
                'final_reserved_usd': 0.0})
            raise RuntimeError(reason)
    verify_inputs()
    if img_bytes * IMAGE_SIZE_INFLATION > IMAGE_LIMIT_BYTES:
        raise RuntimeError(f"payload {img_bytes} B exceeds the image-size cap; refusing to send")
    if evidence_access is not None:
        from label import evidence_access as ea
        evidence_access.copy_contract_version = (fields.get('config') or {}).get('sensor_copy_contract_version')
        fields = copy.deepcopy(fields)
        fields['evidence_inspection'] = evidence_access.record()
        if not dry_run:
            def select(parts):
                response = call_model_once(parts, model, reasoning, api_key, ea.selection_tokens(reasoning), timeout,
                    **({'pricing_quote': pricing_quote} if pricing_quote is not None else {}))
                usage = response.get('usage') or {}
                cost, _ = _verified_final_cost(usage, model)
                if cost is not None:
                    response.setdefault('usage', {})['cost'] = cost
                return response
            # Reserve the maximum output and a conservative input allowance before each selection dispatch.
            def reserve(parts):
                return final_cost_bound(parts, ea.selection_tokens(reasoning), model=model,
                                        require_known_pricing=require_known_pricing,
                                        **({'pricing_quote': pricing_quote} if pricing_quote is not None else {}))
            cache = Path(out_path).parent / '.evidence' / (Path(out_path).name + '.inspection.json')
            if force_refresh and cache.exists():
                persist_final_claim(Path(out_path).with_name(f'noreply_{Path(out_path).name}'), {
                    'episode_dir': str(ep_dir), 'model': model, 'reasoning_effort': reasoning, **fields,
                    'parse_ok': False, 'final_dispatch_outcome': 'not dispatched', 'final_reserved_usd': 0.0,
                    'no_reply': 'Inspection refresh requested; earlier receipts retained in billing history.'})
            selection_dispatch = reserve_selection
            if reserve_selection is not None and settle_selection is not None:
                final_floor = final_cost_bound(content, max_tokens, model=model,
                                               require_known_pricing=require_known_pricing,
                                               **({'pricing_quote': pricing_quote} if pricing_quote is not None else {}))

                def selection_dispatch(amount):
                    # Evidence selection serves the final request, so it is paid only when the final request can
                    # still be paid after it; a cap that cannot cover the final buys no selection.
                    if not reserve_selection(amount + final_floor):
                        return False
                    settle_selection(final_floor, 0.0)
                    return True
            # inspection off keeps the inspection record (a tactile array's regional analysis is owned by it) and
            # buys no selection
            off = policy.inspection_mode(fields.get('annotation_policy') or {}) == 'off'
            rounds = 0 if off else 2
            try:
                doc, extra = ea.discover(evidence_access, content, select, cache, rounds=rounds, reserve_cost=reserve,
                                        reserve_dispatch=selection_dispatch, settle_dispatch=settle_selection,
                                        force_refresh=force_refresh, refresh_id=inspection_refresh_id,
                                        ledger_managed=(callable(reserve_selection) and callable(settle_selection)),
                                        image_limit=MAX_IMAGES, image_bytes_limit=IMAGE_LIMIT_BYTES / IMAGE_SIZE_INFLATION)
            except Exception as error:
                doc = evidence_access.record()
                if cache.exists():
                    try:
                        saved = json.loads(cache.read_text())
                        doc.update({k: saved[k] for k in ('rounds', 'cost_usd', 'digest') if k in saved})
                    except (OSError, ValueError):
                        pass
                doc.update(status='incomplete', limitations=[f'Inspection failed: {type(error).__name__}: {error}'])
                doc['previously_billed_usd'] = inspection_billed(fields, doc.get('digest'))
                auxiliary_cost(doc)
                fields['evidence_inspection'] = doc
                failed = {'episode_dir': str(ep_dir), 'model': model, 'reasoning_effort': reasoning,
                          **fields, 'parse_ok': False, 'no_reply': f'{type(error).__name__}: {error}',
                          'final_dispatch_outcome': 'not dispatched'}
                write_atomic(Path(out_path).with_name(f'noreply_{Path(out_path).name}'), failed)
                raise
            doc['previously_billed_usd'] = inspection_billed(fields, doc.get('digest'))
            auxiliary_cost(doc)
            fields['evidence_inspection'] = doc
            if doc.get('pause_reason') == 'spend_cap':
                reason = 'Inspection selection exceeds the remaining batch spend cap; final annotation deferred.'
                write_atomic(Path(out_path).with_name(f'noreply_{Path(out_path).name}'), {
                    'episode_dir': str(ep_dir), 'model': model, 'reasoning_effort': reasoning, **fields,
                    'parse_ok': False, 'no_reply': reason, 'no_reply_kind': 'spend_cap',
                    'final_dispatch_outcome': 'not dispatched', 'final_reserved_usd': 0.0})
                raise SpendCap(reason)
            content, img_bytes, suffix = ea.final_content(content, img_bytes, evidence_access, doc, extra,
                image_limit=MAX_IMAGES, image_bytes_limit=IMAGE_LIMIT_BYTES, image_inflation=IMAGE_SIZE_INFLATION)
            prompt_text += '\n' + suffix
    verify_inputs()
    n_images = sum(1 for c in content if c.get("type") == "image_url")
    if n_images > MAX_IMAGES:
        raise RuntimeError(f"{n_images} images exceeds the {MAX_IMAGES}-image cap; refusing to send")
    from label import grids as input_grids
    fields = copy.deepcopy(fields)
    fields.setdefault('config', {})['request_images'] = input_grids.receipt(content,
        {**fields, 'model': model, 'reasoning_effort': reasoning})
    bound_sensor_analysis = None
    if not dry_run and isinstance(fields.get('sensor_evidence'), dict):
        from label import sensor_analysis, sensor_analysis_run
        analysis_request = sensor_analysis.build(content, fields['sensor_evidence'], config={
            **fields['config'], 'model': model, 'reasoning': reasoning, 'max_tokens': max_tokens,
            'input_identity': fields.get('input_identity')},
            ep=evidence_access.ep if evidence_access is not None else None)
        if analysis_request is not None:
            content = sensor_analysis.local_content(content)
            prompt_text = '\n'.join(part['text'] for part in content if part.get('type') == 'text')
            fields['config']['request_images'] = input_grids.receipt(content,
                {**fields, 'model': model, 'reasoning_effort': reasoning})
            analysis_doc = fields.get('evidence_inspection')
            if not isinstance(analysis_doc, dict) or not analysis_doc.get('digest'):
                raise ValueError('Required sensor analysis has no durable inspection owner; no dispatch.')
            analysis_cache = Path(out_path).parent / '.evidence' / (Path(out_path).name + '.inspection.json')
            def analysis_call(parts):
                return call_model_once(parts, model, reasoning, api_key, max_tokens, timeout,
                    **({'pricing_quote': pricing_quote} if pricing_quote is not None else {}))
            def analysis_reserve(parts):
                return final_cost_bound(parts, max_tokens, model=model, require_known_pricing=require_known_pricing,
                    **({'pricing_quote': pricing_quote} if pricing_quote is not None else {}))
            analysis_dispatch = reserve_selection
            if callable(reserve_selection) and callable(settle_selection):
                analysis_final_floor = analysis_reserve(content)
                def analysis_dispatch(amount):
                    if not reserve_selection(amount + analysis_final_floor):
                        return False
                    settle_selection(analysis_final_floor, 0.0)
                    return True
            try:
                analysis_receipt = sensor_analysis_run.execute(analysis_request, analysis_doc, analysis_cache,
                    call=analysis_call, reserve_cost=analysis_reserve, model=model,
                    reserve_dispatch=analysis_dispatch, settle_dispatch=settle_selection,
                    verify_inputs=verify_inputs)
                fields['sensor_analysis'] = copy.deepcopy(analysis_receipt)
                bound_sensor_analysis = sensor_analysis.bind(analysis_request, analysis_receipt)
            except Exception as error:
                if isinstance(analysis_doc.get('sensor_analysis'), dict):
                    fields['sensor_analysis'] = copy.deepcopy(analysis_doc['sensor_analysis'])
                persist_final_claim(Path(out_path).with_name(f'noreply_{Path(out_path).name}'), {
                    'episode_dir': str(ep_dir), 'model': model, 'reasoning_effort': reasoning, **fields,
                    'parse_ok': False, 'final_dispatch_outcome': 'not dispatched', 'final_reserved_usd': 0.0,
                    'no_reply': f'Sensor analysis incomplete: {error}',
                    **({'no_reply_kind': 'spend_cap'} if isinstance(error, SpendCap) else {})})
                raise
            verify_inputs()
    if dry_run:
        fields['config']['request_images']['dispatch_status'] = 'dry_run'
        import io
        from PIL import Image
        est_img = 0.0
        for c in content:
            if c.get("type") == "image_url":
                raw = base64.b64decode(c["image_url"]["url"].split(",", 1)[1])
                w, h = Image.open(io.BytesIO(raw)).size
                est_img += estimate_image_tokens(w, h)
        est_text = sum(len(c["text"]) for c in content if c.get("type") == "text") / 4.0
        result = {"episode_dir": str(ep_dir), "model": model, "reasoning_effort": reasoning, **fields,
                  "dry_run": True, "prompt_text": prompt_text,
                  "request": {"n_parts": len(content), "n_images": n_images, "image_bytes": img_bytes,
                              "est_input_tokens": round(est_img + est_text)}}
        if model.startswith('openai/') and model.split('/', 1)[-1] in OPENAI_PRICES:
            # what a paid run reserves for this final call, and the least any final call can reserve (its output limit,
            # with no prompt at all), since routing and evidence selection change the prompt; review refuses a cap up
            # front only when it cannot cover even that, never one that routing would make fit
            result['request']['final_cost_bound_usd'] = final_cost_bound(content, max_tokens, model=model)
            result['request']['final_cost_floor_usd'] = final_cost_bound([], max_tokens, model=model)
        write_atomic(out_path, result)
        return result
    bound = final_cost_bound(content, max_tokens, model=model, require_known_pricing=require_known_pricing,
                             **({'pricing_quote': pricing_quote} if pricing_quote is not None else {}))
    claim_path = Path(out_path).with_name(f'noreply_{Path(out_path).name}')
    fields['config']['request_images']['dispatch_status'] = 'dispatch_claimed'
    claim = {"episode_dir": str(ep_dir), "model": model, "reasoning_effort": reasoning, **fields,
             "parse_ok": False, "no_reply": "final model call has no verified response",
             "final_dispatch_outcome": "claimed", "final_reserved_usd": bound}
    if reserve_final is None:
        persist_final_claim(claim_path, claim)
    else:
        try:
            reserve_final(episode_cost(claim), claim_path, claim)
        except SpendCap as error:
            fields['config']['request_images']['dispatch_status'] = 'not_dispatched'
            write_atomic(claim_path, {**claim, "no_reply": str(error),
                                      "final_dispatch_outcome": "not dispatched", "final_reserved_usd": 0.0})
            raise
    t_start = time.time()
    try:
        resp = call_model_once(content, model, reasoning, api_key, max_tokens, timeout,
            **({'pricing_quote': pricing_quote} if pricing_quote is not None else {}))
    except DispatchError as error:
        if error.verified_unpaid:
            fields['config']['request_images']['dispatch_status'] = 'not_dispatched'
            sent = error.stage != 'not sent'
            write_atomic(claim_path, {**claim, 'final_dispatch_outcome': 'not dispatched',
                'final_reserved_usd': 0.0,
                'no_reply': ('final request rejected before processing' if sent else
                             'final request could not be sent') + f'; nothing was billed ({str(error)[:200]})',
                'final_rejection': {'http_status': error.http_status, 'http_attempt': 1,
                                    'state': 'rejected unpaid' if sent else 'not sent', 'cost_usd': 0.0,
                                    **({'rejection_code': error.rejection_code} if error.rejection_code else {})}})
        else:
            fields['config']['request_images']['dispatch_status'] = 'outcome_unknown'
            write_atomic(claim_path, {**claim, 'final_dispatch_outcome': 'unverified',
                                      'no_reply': f'{type(error).__name__}: {error}',
                                      **({'no_reply_kind': 'client_timeout'} if error.client_timeout else {})})
        raise
    except Exception as error:
        fields['config']['request_images']['dispatch_status'] = 'outcome_unknown'
        failed = dict(claim, no_reply=f"{type(error).__name__}: {error}",
                      final_dispatch_outcome="unverified")
        write_atomic(claim_path, failed)
        raise
    elapsed = time.time() - t_start
    fields['config']['request_images']['dispatch_status'] = 'response_received'

    choice = (resp.get("choices") or [{}])[0]
    finish = choice.get("finish_reason") or choice.get("native_finish_reason")
    msg = choice.get("message") or {}
    text = msg.get("content") or ""
    raw_usage = resp.get('usage')
    received_usage = dict(raw_usage) if isinstance(raw_usage, dict) else {}
    received_cost, received_source = _verified_final_cost(received_usage, model)
    received_usage.update(est_cost_usd=received_cost, cost_source=received_source)
    received = {**claim, 'parse_ok': False, 'response': copy.deepcopy(resp), 'response_text': text,
                'labels': {'_raw': text, '_parse_error': 'Received response awaiting parsing and evidence binding'},
                'finish_reason': finish, **_served(resp), 'usage': received_usage,
                'final_dispatch_outcome': 'received' if received_cost is not None else 'unverified'}
    received.pop('no_reply', None)
    received_path = Path(out_path).with_name(f'failed_{Path(out_path).name}') if finish == 'length' else Path(out_path)
    persist_final_claim(received_path, received)
    claim_path.unlink(missing_ok=True)
    if received_cost is not None and settle_final is not None:
        # The reply is billed: hold its real cost, not its upper bound, so later calls see the true allowance.
        settle_final(received_cost)
    if finish == "length":
        # the output hit max_tokens, so the JSON is cut off: keep what came back beside the outputs (never as an
        # episode_*.json), then fail. Paid retry requires --retry-received for this episode.
        raw_usage = resp.get("usage")
        usage = dict(raw_usage) if isinstance(raw_usage, dict) else {}
        cost, source = _verified_final_cost(usage, model)
        usage["est_cost_usd"] = cost
        usage["cost_source"] = source
        # with every field of the request, so the board shows the episode's checks and undecodable stretches beside
        # the cut-off reply (board/to_board.py label_failed)
        failed = {"episode_dir": str(ep_dir), "model": model, "reasoning_effort": reasoning, **fields,
                  "finish_reason": finish, **_served(resp), "usage": usage, "content_tail": text[-4000:],
                  "reasoning_tail": (msg.get("reasoning") or "")[-8000:],
                  'response': copy.deepcopy(resp), 'response_text': text}
        if cost is None:
            failed.update(final_dispatch_outcome="unverified", final_reserved_usd=bound,
                          usage_reported=raw_usage)
        write_atomic(Path(out_path).with_name(f"failed_{Path(out_path).name}"), failed)
        claim_path.unlink(missing_ok=True)
        raise Truncated(f"response truncated (finish_reason=length) at max_tokens={max_tokens}", episode_cost(failed))
    # an unparseable or incomplete reply is recorded as it came, never repaired or resent automatically
    labels, parse_ok = read_reply(text, fields)
    failure = reply_error(resp)
    if failure is not None:
        # the provider failed partway (billed or unpriced, or _post would have retried it): its text is a fragment,
        # so it is named as the provider's error, never as a reply that did not parse; --retry-received resends it
        labels, parse_ok = {'_raw': text, '_parse_error': f'the provider returned an error in the reply ({failure}); '
                                                          'resend it with --retry-received'}, False

    raw_usage = resp.get("usage")
    usage = raw_usage if isinstance(raw_usage, dict) else {}
    n_in = usage.get("prompt_tokens")
    n_out = usage.get("completion_tokens")
    completion_details = usage.get("completion_tokens_details")
    n_reason = completion_details.get("reasoning_tokens") if isinstance(completion_details, dict) else None
    ptd = usage.get("prompt_tokens_details")
    ptd = ptd if isinstance(ptd, dict) else {}
    cost, cost_source = _verified_final_cost(usage, model)
    result = {
        "episode_dir": str(ep_dir),
        "model": model,
        "reasoning_effort": reasoning,
        **fields,
        "provider": "openrouter" if is_openrouter_key(api_key) else "openai",
        **_served(resp),
        "finish_reason": finish,
        "parse_ok": parse_ok,
        "labels": labels,
        'response': copy.deepcopy(resp), 'response_text': text,
        "usage": {"prompt_tokens": n_in, "completion_tokens": n_out,
                  "reasoning_tokens": n_reason, "est_cost_usd": cost,
                  "cost_source": cost_source,
                  "cached_tokens": ptd.get("cached_tokens"), "cache_write_tokens": ptd.get("cache_write_tokens"),
                  "latency_s": round(elapsed, 1)},
    }
    if cost is None:
        result.update(final_dispatch_outcome="unverified", final_reserved_usd=bound,
                      usage_reported=raw_usage)
    if 'camera_observations' in labels:
        result['camera_observations'] = bind_camera_observations(labels['camera_observations'], fields)
        result['camera_observations']['provenance'] = {'model': result.get('model_served') or model,
            'generation_id': result.get('generation_id'), 'method': 'same episode annotation request'}
    from label import sensor_evidence as se
    request_doc = fields.get("sensor_evidence")
    routes = se.route_findings(labels, inspection=evidence_access is not None,
                               sensor_evidence=isinstance(request_doc, dict) and bool(request_doc.get('sensors')))
    if isinstance(request_doc, dict):
        result["sensor_evidence"] = se.bind(request_doc, routes['sensor_findings'],
                                         ep=evidence_access.ep if evidence_access is not None else None,
                                         copy_contract_version=(fields.get('config') or {}).get('sensor_copy_contract_version'))
        result["sensor_evidence"]["provenance"] = {"model": result.get("model_served") or model,
            "generation_id": result.get("generation_id"), "method": "same episode annotation request"}
    if evidence_access is not None:
        additional = evidence_access.bind(routes['evidence_findings'])
        result['evidence_inspection']['untimed_findings'] = additional['untimed_findings']
        evidence = result.setdefault('sensor_evidence', additional)
        if evidence is not additional:
            for key in ('sensors', 'series', 'findings', 'unbound_findings', 'coverage', 'limitations'):
                evidence.setdefault(key, []).extend(additional[key])
        evidence.setdefault('provenance', {}).update(model=result.get('model_served') or model,
            generation_id=result.get('generation_id'), method='episode annotation with bounded evidence inspection')
    if routes['unbound'] or routes['routed']:
        if not isinstance(result.get('sensor_evidence'), dict):
            result['sensor_evidence'] = {**se.empty_doc(), 'provenance': {
                'model': result.get('model_served') or model, 'generation_id': result.get('generation_id'),
                'method': 'same episode annotation request'}}
        se.keep_routing(result['sensor_evidence'], routes)
    if bound_sensor_analysis is not None:
        sensor_analysis.attach(result, bound_sensor_analysis)
    if msg.get("reasoning"):
        # the reasoning text or summary the provider returned, when it returns one
        result["reasoning_text"] = msg["reasoning"][:20000]
    write_atomic(out_path, result)
    if parse_ok and isinstance(result.get('sensor_evidence'), dict) and isinstance(result.get('evidence_inspection'), dict) \
            and result['evidence_inspection'].get('digest'):
        # The paid annotation is saved first, so nothing that happens during the card call can cost it.
        _video_cards(result, out_path, model=model, reasoning=reasoning, api_key=api_key, timeout=timeout,
                     pricing_quote=pricing_quote, require_known_pricing=require_known_pricing,
                     reserve_selection=reserve_selection, settle_selection=settle_selection, verify_inputs=verify_inputs)
        write_atomic(out_path, result)
    claim_path.unlink(missing_ok=True)
    c = fields.get("config") or {}
    cost_text = f"cost=${cost:.3f}" if cost is not None else f"reserved=${bound:.3f} cost=unverified"
    print(f"[{Path(ep_dir).name}] timesteps={c.get('n_timesteps')} images={n_images} "
          f"in={n_in} out={n_out} {cost_text} {elapsed:.0f}s parse_ok={parse_ok} -> {out_path}", flush=True)
    if labels.get('_incomplete'):
        print(f"INCOMPLETE {Path(ep_dir).name}: {'; '.join(labels['_incomplete'])}", file=sys.stderr, flush=True)
    return result


def _video_cards(result, out_path, *, model, reasoning, api_key, timeout, pricing_quote, require_known_pricing,
                 reserve_selection, settle_selection, verify_inputs):
    """Choose and write the video cards from the bound sensor findings (label/sensor_cards.py), billed into the
    inspection invoice. The annotation is already paid for and bound, so a failure here is recorded and the record
    is still written: its findings stay in the list, and the board shows them as it did before cards existed."""
    from label import sensor_analysis_run, sensor_cards
    evidence, inspection = result['sensor_evidence'], result['evidence_inspection']
    request = sensor_cards.build(evidence, task=(result.get('labels') or {}).get('task_summary'),
                                 config={'model': model, 'reasoning': reasoning})
    if request is None:
        return
    cache = Path(out_path).parent / '.evidence' / (Path(out_path).name + '.inspection.json')
    quote = {'pricing_quote': pricing_quote} if pricing_quote is not None else {}
    try:
        receipt = sensor_analysis_run.execute(
            request, inspection, cache, key='sensor_cards', model=model,
            call=lambda parts: call_model_once(parts, model, reasoning, api_key, request['config']['max_tokens'], timeout,
                                               **quote),
            reserve_cost=lambda parts: final_cost_bound(parts, request['config']['max_tokens'], model=model,
                                                        require_known_pricing=require_known_pricing, **quote),
            reserve_dispatch=reserve_selection, settle_dispatch=settle_selection, verify_inputs=verify_inputs)
        sensor_cards.apply(evidence, receipt['request'], receipt)
    except Exception as error:  # noqa: BLE001 - the cards are optional; named in the evidence's limitations
        evidence.setdefault('limitations', []).append(f'Video cards were not written: {error}')


def _served(resp: dict) -> dict:
    """What the response says served the request: the upstream provider, OpenRouter's generation id, the model
    id that answered and the provider's system fingerprint (None when a field is absent)."""
    return {"provider_name": resp.get("provider"), "generation_id": resp.get("id"),
            "model_served": resp.get("model"), "system_fingerprint": resp.get("system_fingerprint")}


def get_keys() -> list[str]:
    """OpenRouter keys from OPENROUTER_API_KEYS (comma-separated), or when it holds none OpenAI keys from
    OPENAI_API_KEY (comma-separated). The key decides where a call goes: anything in OPENROUTER_API_KEYS that is
    not an OpenRouter key is ignored, so no other provider's key is ever sent to OpenRouter, and an OpenRouter key
    in OPENAI_API_KEY is ignored too."""
    def split(var):
        return [k.strip() for k in os.environ.get(var, "").split(",") if k.strip()]
    return ([k for k in split("OPENROUTER_API_KEYS") if is_openrouter_key(k)]
            or [k for k in split("OPENAI_API_KEY") if not is_openrouter_key(k)])


def discover_episodes(root: Path) -> list[Path]:
    return [d for d in sorted(Path(root).glob("episode_*")) if d.is_dir() and me.is_episode_dir(d)]


class KeyPool:
    """Keys round robin. A key that runs out of credit is retired for the rest of the batch and its episode is
    retried on the next live key; when every key is retired the batch stops starting new episodes."""

    def __init__(self, keys: list[str]):
        self.keys = list(keys)
        self.dead: set[str] = set()
        self.i = 0
        self.lock = threading.Lock()

    def take(self) -> str | None:
        with self.lock:
            live = [k for k in self.keys if k not in self.dead]
            if not live:
                return None
            k = live[self.i % len(live)]
            self.i += 1
            return k

    def retire(self, key: str, why: str) -> None:
        with self.lock:
            if key not in self.dead:
                self.dead.add(key)
                print(f"KEY RETIRED ...{key[-4:]}: {why[:200]} ({len(self.keys) - len(self.dead)} live)",
                      file=sys.stderr, flush=True)


def _release_memory() -> None:
    """Hand freed memory back to the system after each episode. Decoding many episodes' frames on many threads
    leaves glibc's per-thread heaps full of freed image buffers the process keeps, so a long batch grew to about
    17 GB per process; malloc_trim returns them. A no-op where glibc is absent."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


class ExecutionBusy(RuntimeError):
    """Another process owns a folder's execution lock."""


def cli(main) -> int:
    """A command's main(), where a folder another process is working in is refused in one line, not a traceback."""
    try:
        return main()
    except ExecutionBusy as busy:
        raise SystemExit(f'refused, {busy}; let it finish or stop it first') from None


@contextmanager
def execution_lock(folder: Path, *, inherited_fd: int | None = None):
    """One process owns execution until dispatched work settles, where flock is shared.

    Cross-host writers require a filesystem with working shared locks. Client-local
    NFS nolock/local_lock mounts do not provide that guarantee.
    """
    import fcntl
    import stat
    Path(folder).mkdir(parents=True, exist_ok=True)
    lock_path = Path(folder) / '.execution.lock'
    if inherited_fd is not None:
        try:
            held, expected = os.fstat(inherited_fd), lock_path.stat()
            if (not stat.S_ISREG(held.st_mode)
                    or (held.st_dev, held.st_ino) != (expected.st_dev, expected.st_ino)):
                raise ValueError('descriptor does not identify this output lock')
            # A different open description cannot borrow an existing flock. This
            # succeeds for the description inherited from the owning review.
            fcntl.flock(inherited_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, ValueError) as error:
            raise RuntimeError(f'Invalid inherited execution lease for {folder}: {error}') from error
        yield inherited_fd
        return  # The encompassing review still owns the inherited description.
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ExecutionBusy(f'Another label execution is running in {folder}') from error
        yield lock.fileno()


def run_batch(episodes: list[Path], out_dir: Path, *, execution_lease_fd=None, **kwargs) -> int:
    with execution_lock(out_dir, inherited_fd=execution_lease_fd):
        result = _run_batch(episodes, out_dir, **kwargs)
        # The worker owns its receipts until metadata is settled. A killed wrapper
        # must not leave completed paid work permanently marked as running.
        from label.run import finalize_worker
        finalize_worker(out_dir, result)
        return result


def _run_batch(episodes: list[Path], out_dir: Path, *, keys: list[str], concurrency: int, force: bool,
              max_spend: float = 0.0, output_path: Path | None = None,
              retry_received: list[str] | None = None, **label_kw) -> int:
    """Resume verified outputs and reserve each call within this invocation's allowance."""
    if not math.isfinite(max_spend) or max_spend < 0:
        raise ValueError('spend allowance must be finite and nonnegative')
    if output_path is not None and (len(episodes) != 1 or Path(output_path).parent != out_dir):
        raise ValueError('a custom output path requires one episode in its output folder')
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = KeyPool(keys)
    dry = bool(label_kw.get("dry_run"))
    retry_received = set(retry_received or [])
    if force and retry_received:
        raise ValueError('--force and --retry-received cannot be combined')
    unknown = retry_received - {ep.name for ep in episodes}
    if unknown:
        raise ValueError(f'--retry-received names unknown episodes: {sorted(unknown)}')

    def out_for(ep: Path) -> Path:
        return Path(output_path) if output_path is not None else out_dir / f"{ep.name}.json"

    def is_done(ep: Path) -> bool:
        p = out_for(ep)
        if force:
            return False
        try:
            selected = annotation_record(out_dir, p.name)
        except (OSError, ValueError, TypeError, AttributeError) as error:
            stale[ep.name] = f'saved receipts cannot be verified: {error}'
            return True
        if selected is None:
            if ep.name in retry_received:
                stale[ep.name] = 'explicit received retry needs an existing received reply'
                return True
            return False
        records = [selected]
        try:
            never_dispatched = all(not has_dispatch_receipt(record) for _, record in records)
        except (ValueError, TypeError, AttributeError):
            never_dispatched = False
        if len(records) == 1 and records[0][0] == p and records[0][1].get('input_identity') is None:
            if not dry and not annotation_complete(records[0][1]):
                stale[ep.name] = f'saved output {p.name} has no usable annotation or input proof; use --force to refresh'
            else:
                unverified[ep.name] = f"saved output {p.name} predates input proof"
            return True
        if len(records) == 1 and records[0][0].name.startswith('noreply_') and (
                records[0][1].get('input_identity') is None and
                records[0][1].get('final_dispatch_outcome') is None):
            unverified[ep.name] = 'legacy no-reply record has no source proof or dispatch receipt'
            return False
        try:
            current = input_identity(ep, model=label_kw.get("model"),
                                     reasoning=label_kw.get("reasoning"),
                                     max_tokens=label_kw.get("max_tokens", 64000), cell_w=label_kw.get("cell_w", 0),
                                     example_dir=label_kw.get("example_dir"),
                                     sample_fps=label_kw.get('sample_fps'), task_mode=label_kw.get('task_mode'),
                                     dataset_note=label_kw.get('dataset_note'),
                        inspection=label_kw.get('inspection'))
        except (OSError, ValueError, RuntimeError) as error:
            if never_dispatched:
                saved_failures.add(ep.name)
                unverified[ep.name] = f'input cannot be verified; no paid dispatch: {error}'
                return True
            stale[ep.name] = f"input cannot be verified: {type(error).__name__}: {error}"
            return True
        if never_dispatched:
            if ep.name in retry_received:
                saved_failures.add(ep.name)
                unverified[ep.name] = 'received retry refused: no received paid reply exists'
                return True
            if any(item.get('missing') for item in current['source_proof']):
                saved_failures.add(ep.name)
                unverified[ep.name] = 'source input is missing; no paid dispatch'
                return True
            if ep.name not in retry_received:
                return False  # No paid inputs exist to reuse; assemble a fresh request.
        for q, record in records:
            saved = record.get("input_identity")
            if saved is None:
                unverified[ep.name] = f"saved output {q.name} predates input proof"
                return True
            comparison = current
            if isinstance(saved, dict) and isinstance(saved.get('settings'), dict):
                # Completion limits belong to each saved request. Changing a future
                # request limit does not invalidate a durably received reply.
                comparison = {**current, 'settings': {**current['settings'],
                    'max_tokens': saved['settings'].get('max_tokens')}}
            matches = same_input_identity(saved, comparison)
            if not matches and not dry:
                from label.request_attestation import matches as actual_request_matches
                matches = actual_request_matches(record, comparison, ep, out_dir,
                                                 example_dir=label_kw.get('example_dir'))
                if matches:
                    from label.reparse import refresh_received, save_annotation
                    try:
                        updated = refresh_received(record)
                        if updated != record:
                            save_annotation(out_dir, p.name, q, updated)
                            record.clear()
                            record.update(updated)
                    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError):
                        matches = False
            if not matches or any(item.get("missing") for item in current["source_proof"]):
                hint = ''
                saved_policy = ((saved or {}).get('settings') or {}).get('annotation_policy')
                if (isinstance(saved_policy, dict) and 'inspection' not in saved_policy
                        and policy.inspection_mode((current.get('settings') or {}).get('annotation_policy') or {})
                        != policy.LEGACY_INSPECTION):
                    # labelled before inspection could be set, when every recorded field was inspected
                    hint = (f'; it was labelled before inspection could be set, so add --inspection '
                            f'{policy.LEGACY_INSPECTION} to resume it as it was paid for')
                mode = saved_policy.get('task_mode') if isinstance(saved_policy, dict) else None
                if mode in policy.TASK_MODES:
                    # labelled in the episode or session mode, before every recording became a list of tasks
                    hint += (f'; it was labelled in the {mode} mode, before every recording was labelled as a list '
                             'of tasks, so --force relabels it as one')
                stale[ep.name] = f"saved output {q.name} has changed or missing input; use --force to refresh{hint}"
                return True
        if any(record.get('final_dispatch_outcome') in ('claimed', 'unverified') for _, record in records):
            why = None if dry or annotation_complete(records[0][1]) else auto_resend(records[0][1])
            if why and ep.name not in retry_received:
                auto_resent[ep.name] = (why, dispatched_attempts(records[0][1]))
                return False
            if ep.name in retry_received and not dry:
                # The caller named this unit, accepting that its unknown charge may have been real. The unresolved
                # attempt stays in the billing history with its reservation (_final_cost_history), so the cap still
                # counts it, and nothing else is sent again.
                return False
            saved_failures.add(ep.name)
            retryable_received.add(ep.name)
            unverified[ep.name] = ('saved dispatch has an unresolved charge (it may have been billed)'
                                   + (f' after {dispatched_attempts(records[0][1])} attempts'
                                      if provider_failure(records[0][1]) or client_timeout(records[0][1]) else '')
                                   + f'; --retry-received {ep.name} sends this unit alone again')
            if not dry and any(record.get('parse_ok') is True and not annotation_complete(record) for _, record in records):
                stale[ep.name] = 'saved annotation is unusable and its charge is unresolved; use --force to refresh explicitly'
            return True
        for path, record in records:
            received = (path == p or record.get('response') is not None or
                        record.get('finish_reason') == 'length')
            if not dry and received and not record.get('dry_run') and not annotation_complete(record):
                try:
                    _history_cost(record)
                except (ValueError, TypeError):
                    stale[ep.name] = 'saved reply billing cannot be verified; reconcile it before retrying'
                    return True
                if ep.name in retry_received:
                    return False
                why = auto_resend(record)
                if why:
                    auto_resent[ep.name] = (why, dispatched_attempts(record))
                    return False
                saved_failures.add(ep.name)
                retryable_received.add(ep.name)
                unverified[ep.name] = ('saved paid reply needs reparse or explicit '
                                       f'--retry-received {ep.name}'
                                       + (f'; sent {dispatched_attempts(record)} times' if provider_failure(record)
                                          else '; a reply the model wrote is never resent automatically'))
                return True
        record = records[0][1]
        if dry:
            return bool(record.get('dry_run'))
        complete = annotation_complete(record)
        if ep.name in retry_received and complete:
            # nothing to retry: the unit is labelled, and naming it changes nothing (it never fails the run)
            print(f"NOTE {ep.name}: --retry-received skipped, it already has a usable annotation", file=sys.stderr,
                  flush=True)
            return True
        if ep.name in retry_received:
            saved_failures.add(ep.name)
            unverified[ep.name] = 'received retry refused: this episode already has a usable annotation'
            return True
        inspection = record.get('evidence_inspection') or {}
        internally_limited = ('Inspection selection stopped at the cost or dispatch boundary.'
                              in inspection.get('limitations', []))
        if complete and (internally_limited or inspection.get('pause_reason') == 'spend_cap' or
                'Inspection selection exceeds the remaining batch spend cap.' in inspection.get('limitations', [])):
            unverified[ep.name] = 'saved paid annotation omitted budget-blocked inspection; use --force to refresh explicitly'
            if internally_limited:
                print(f'LIMITED {ep.name}: saved paid annotation has incomplete inspection; '
                      'use --force to refresh explicitly', file=sys.stderr, flush=True)
        if record.get('parse_ok') is True and not complete:
            try:
                _billing_fields(p)
            except (OSError, ValueError, TypeError, AttributeError):
                stale[ep.name] = 'saved annotation is unusable and its billing cannot be verified; reconcile it before retrying'
                return True
        return complete

    stale, unverified, saved_failures, retryable_received = {}, {}, set(), set()
    auto_resent = {}       # unit -> (the failure it is sent again for, provider's or ours, how many times it was sent)
    todo = [ep for ep in episodes if not is_done(ep)]
    for name, (why, sent) in sorted(auto_resent.items()):
        print(f'RESEND {name}: sent {sent} time{"s" if sent != 1 else ""}, {why}; sending it again '
              f'(automatic, at most {AUTO_RESENDS + 1} sends)', file=sys.stderr, flush=True)
    # a marker says what this run found; one left by an earlier run for a unit now current would say it twice
    for name in {ep.name for ep in episodes} - set(stale) - set(unverified):
        (out_dir / f"stale_{name}.json").unlink(missing_ok=True)
    for name, reason in {**unverified, **stale}.items():
        write_atomic(out_dir / f"stale_{name}.json", {"episode": name, "status": "stale" if name in stale else
                     "unverified", "reason": reason,
                     "refresh": f'--retry-received {name}' if name in retryable_received else '--force'})
    if stale:
        for name, reason in stale.items():
            print(f"STALE {name}: {reason}", file=sys.stderr, flush=True)
        saved_failures.update(stale)
    skipped = len(episodes) - len(todo)
    print(f"episodes={len(episodes)} skipped(done)={skipped} todo={len(todo)} "
          f"keys={len(keys)} concurrency={concurrency} dry_run={dry}", flush=True)
    if not todo:
        print(f'done=0 failed={len(saved_failures)} skipped={skipped} total_cost=$0.00 out_dir={out_dir}', flush=True)
        return 1 if saved_failures else 0
    done, failed, cap_refused = 0, len(saved_failures), 0
    total_cost = 0.0
    fails = []
    spent = {"usd": 0.0}
    spend_lock = threading.Condition()
    pending = {}
    stopping = threading.Event()

    def allowance(ep, amount):
        # Only live dispatches can release a temporary reservation. Settled costs and
        # unknown charges from finished workers remain committed to the cap.
        if stopping.is_set():
            raise InterruptedError('Label execution interrupted before dispatch')
        if max_spend > 0 and amount > max_spend:
            return False, False
        waited = False
        while max_spend > 0 and spent['usd'] + amount > max_spend:
            if stopping.is_set():
                raise InterruptedError('Label execution interrupted before dispatch')
            if not any(value > 0 for owner, value in pending.items() if owner != ep):
                return False, waited
            spend_lock.wait()
            waited = True
        return True, waited

    def work_claimed(ep):
        prior_cost = episode_cost(_billing_fields(out_for(ep)))
        inspection_refresh_id = uuid.uuid4().hex if force else None
        reserved = [0.0]
        selection_accounted = [0.0]
        def reserve_selection(amount):
            with spend_lock:
                pending[ep] = 0.0
                spend_lock.notify_all()
                allowed, _ = allowance(ep, amount)
                if not allowed:
                    return False
                spent['usd'] += amount
                selection_accounted[0] += amount
                pending[ep] = amount
                return True
        def settle_selection(reserve, actual):
            with spend_lock:
                spent['usd'] += actual - reserve
                selection_accounted[0] += actual - reserve
                pending[ep] = max(0.0, pending.get(ep, 0.0) - reserve)
                spend_lock.notify_all()
        def reserve_final(amount, path, claim):
            with spend_lock:
                pending[ep] = 0.0
                spend_lock.notify_all()
                new_amount = max(0.0, amount - selection_accounted[0] - prior_cost)
                allowed, waited = allowance(ep, new_amount)
                if not allowed:
                    raise SpendCap(f"final request reserve ${new_amount:.2f} exceeds remaining spend cap")
                if waited:
                    current = input_identity(ep, model=label_kw.get('model'),
                        reasoning=label_kw.get('reasoning'), max_tokens=label_kw.get('max_tokens', 64000),
                        cell_w=label_kw.get('cell_w', 0), example_dir=label_kw.get('example_dir'),
                        sample_fps=label_kw.get('sample_fps'), task_mode=label_kw.get('task_mode'),
                        dataset_note=label_kw.get('dataset_note'),
                        inspection=label_kw.get('inspection'))
                    if not same_input_identity(claim.get('input_identity'), current) or any(
                            item.get('missing') for item in current['source_proof']):
                        raise RuntimeError('Request inputs changed while waiting for spend allowance; no dispatch.')
                persist_final_claim(path, claim)
                spent['usd'] += new_amount
                reserved[0] = new_amount
                pending[ep] = new_amount
        def settle_final(cost):
            with spend_lock:
                held = max(0.0, cost - selection_accounted[0] - prior_cost)
                spent['usd'] += held - reserved[0]
                reserved[0] = held
                pending[ep] = 0.0
                spend_lock.notify_all()
        while True:
            key = pool.take() if not dry else "dry-run"
            if key is None:
                return ("fail", ep, "all keys retired (out of credit)")
            try:
                r = label_episode(ep, out_for(ep), api_key=key, reserve_final=reserve_final,
                                  reserve_selection=reserve_selection, settle_selection=settle_selection,
                                  settle_final=settle_final,
                                  force_refresh=force, inspection_refresh_id=inspection_refresh_id,
                                  require_known_pricing=max_spend > 0, **label_kw)
                _release_memory()
                c = episode_cost(r)
                with spend_lock:
                    spent["usd"] += c - reserved[0] - selection_accounted[0] - prior_cost
                return ("invalid" if r.get("parse_ok") is False and not r.get("dry_run") else "ok", ep, c)
            except KeyExhausted as e:
                if not e.verified_unpaid:
                    failure = out_for(ep).with_name(f'noreply_{out_for(ep).name}')
                    if failure.exists() and not reserved[0]:
                        with spend_lock:
                            spent['usd'] += episode_cost(json.loads(failure.read_text())) - selection_accounted[0] - prior_cost
                    return ('fail', ep, f'KeyExhausted with unverified dispatch outcome: {e}')
                with spend_lock:
                    spent['usd'] -= reserved[0]
                    pending[ep] = max(0.0, pending.get(ep, 0.0) - reserved[0])
                    reserved[0] = 0.0
                    spend_lock.notify_all()
                pool.retire(key, str(e))
                continue
            except DispatchError as e:
                if not e.verified_unpaid:
                    failure = out_for(ep).with_name(f'noreply_{out_for(ep).name}')
                    if failure.exists() and not reserved[0]:
                        with spend_lock:
                            spent['usd'] += episode_cost(json.loads(failure.read_text())) - selection_accounted[0] - prior_cost
                    return ('fail', ep, f'{type(e).__name__}: {e}')
                # the provider refused it or it never left the machine: nothing was billed, so its reservation is
                # released and a plain resume sends it again
                with spend_lock:
                    spent['usd'] -= reserved[0]
                    pending[ep] = max(0.0, pending.get(ep, 0.0) - reserved[0])
                    reserved[0] = 0.0
                    spend_lock.notify_all()
                return ('fail', ep, f'not billed, a resume retries it: {e}')
            except Truncated as e:
                with spend_lock:
                    spent["usd"] += e.cost - reserved[0] - selection_accounted[0] - prior_cost
                return ("fail", ep, str(e))
            except SpendCap as e:
                return ("skip", ep, str(e))
            except Exception as e:  # noqa: BLE001 - one episode's failure is returned with why, the batch goes on
                failure = out_for(ep).with_name(f"noreply_{out_for(ep).name}")
                if failure.exists() and not reserved[0]:
                    with spend_lock:
                        spent['usd'] += episode_cost(json.loads(failure.read_text())) - selection_accounted[0] - prior_cost
                # the failure line names the error; the stage log keeps where it was raised, which that line cannot say
                print(f"TRACEBACK {ep.name}:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
                return ("fail", ep, f"{type(e).__name__}: {e}")

    def work(ep):
        try:
            if stopping.is_set():
                return ('cancelled', ep, 'Label execution interrupted before work started')
            result = work_claimed(ep)
            for delay in AUTO_RESEND_DELAYS_S[:AUTO_RESENDS]:
                if dry or result[0] not in ('fail', 'invalid'):
                    break
                saved = billing_record(out_dir, out_for(ep).name)
                why = auto_resend(saved[1]) if saved is not None and not annotation_complete(saved[1]) else None
                if not why:
                    break
                print(f'RESEND {ep.name}: sent {dispatched_attempts(saved[1])} times, {why}; sending it again in '
                      f'{delay:g}s (automatic, at most {AUTO_RESENDS + 1} sends)', file=sys.stderr, flush=True)
                if stopping.wait(delay):
                    break
                with spend_lock:
                    pending[ep] = 0.0
                    spend_lock.notify_all()
                result = work_claimed(ep)
            return result
        finally:
            with spend_lock:
                pending.pop(ep, None)
                spend_lock.notify_all()

    def no_reply(ep: Path, why: str | None, *, kind=None) -> None:
        """noreply_<episode>.json beside the outputs: why the episode got no reply (the spend cap reached, every key out
        of credit, a request that could not be built or a call that failed), so the board shows the episode and says
        why (board/to_board.py label_outputs). Removed once the episode has a reply; a dry run writes none."""
        p = out_for(ep).with_name(f"noreply_{out_for(ep).name}")
        if dry:
            return
        if why is None:
            p.unlink(missing_ok=True)
            (out_dir / f"stale_{ep.name}.json").unlink(missing_ok=True)
            return
        previous = json.loads(p.read_text()) if p.exists() else {}
        current = billing_record(out_dir, out_for(ep).name)
        if current is not None and current[1].get('final_dispatch_outcome') != 'not dispatched':
            # Keep post-receipt processing errors on that attempt, without inventing a final dispatch.
            previous = current[1]
        if not previous:
            previous = {**_billing_fields(out_for(ep)), 'final_dispatch_outcome': 'not dispatched',
                        'final_reserved_usd': 0.0}
            try:
                previous['input_identity'] = input_identity(ep, model=label_kw.get('model'),
                    reasoning=label_kw.get('reasoning'),
                    max_tokens=label_kw.get('max_tokens', 64000), cell_w=label_kw.get('cell_w', 0),
                    example_dir=label_kw.get('example_dir'), sample_fps=label_kw.get('sample_fps'),
                    task_mode=label_kw.get('task_mode'), dataset_note=label_kw.get('dataset_note'),
                        inspection=label_kw.get('inspection'))
            except (OSError, ValueError, RuntimeError):
                pass
        requested_model = (previous.get('model') or
                           (previous.get('input_identity') or {}).get('settings', {}).get('model') or
                           label_kw.get('model'))
        write_atomic(p, {**previous, "episode_dir": str(ep), "model": requested_model,
                         **({'no_reply_kind': kind} if kind else {}),
                         "parse_ok": False, "no_reply": why})

    ex = ThreadPoolExecutor(max_workers=concurrency)
    futures = [ex.submit(work, ep) for ep in todo]
    try:
        for f in as_completed(futures):
            status, ep, info = f.result()
            no_reply(ep, None if status in ("ok", "invalid") else str(info), kind='spend_cap' if status == 'skip' else None)
            if status == "ok":
                done += 1
                total_cost += float(info or 0)
            elif status == "invalid":
                failed += 1
                total_cost += float(info or 0)
                fails.append(f"FAIL {ep.name}: annotation did not parse")
                print(f"FAIL {ep.name}: annotation did not parse", file=sys.stderr, flush=True)
            elif status == "skip":
                skipped += 1
                cap_refused += 1
                try:
                    total_cost += episode_cost(json.loads(out_for(ep).with_name(f"noreply_{out_for(ep).name}").read_text()))
                except (OSError, ValueError, TypeError, AttributeError):
                    pass
                print(f"SKIP {ep.name}: {info}", file=sys.stderr, flush=True)
            else:
                failed += 1
                fails.append(f"FAIL {ep.name}: {info}")
                print(f"FAIL {ep.name}: {info}", file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        stopping.set()
        with spend_lock:
            spend_lock.notify_all()
        for future in futures:
            future.cancel()
        ex.shutdown(wait=True, cancel_futures=True)
        print('Interrupted: queued episodes cancelled; dispatched replies and reservations retained.', file=sys.stderr)
        return 130
    finally:
        ex.shutdown(wait=True, cancel_futures=True)
    print(f"\ndone={done} failed={failed} skipped={skipped} "
          f"total_cost=${total_cost:.2f} out_dir={out_dir} retired_keys={len(pool.dead)}", flush=True)
    if fails:
        print("failures:\n  " + "\n  ".join(fails), file=sys.stderr)
    return 1 if failed or cap_refused else 0


def inherited_review_lease(folder: Path) -> int | None:
    """Match the declared output path before validating the inherited kernel lock."""
    lease_path = os.environ.get('ARGUS_REVIEW_OUTPUT_LOCK_PATH')
    if not lease_path or Path(lease_path).resolve() != (Path(folder) / '.execution.lock').resolve():
        return None
    try:
        return int(os.environ['ARGUS_REVIEW_OUTPUT_LOCK_FD'])
    except (KeyError, ValueError) as error:
        raise RuntimeError('Invalid inherited execution lease descriptor') from error


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--episode-dir", type=Path, help="label one episode")
    g.add_argument("--episodes-root", type=Path, help="label every episode_* folder under here")
    ap.add_argument("--out", type=Path, help="output JSON with --episode-dir (default EPISODE/labels.json)")
    ap.add_argument("--out-dir", type=Path, help="output folder with --episodes-root, one <episode>.json each")
    policy.add_arguments(ap)
    ap.add_argument("--max-tokens", type=int, default=64000, help="output tokens per episode (default 64000)")
    ap.add_argument("--timeout", type=int, default=SILENT_CONNECTION_LIMIT_S,
                    help="seconds a request's connection may carry no bytes before it is closed as dead; never below "
                         f"{SILENT_CONNECTION_LIMIT_S} (the provider's longest possible silence), and a reply still "
                         "arriving is never cut off")
    ap.add_argument("--concurrency", type=int, default=0, help="episodes in flight (default 4 per key)")
    ap.add_argument("--limit", type=int, default=0, help="label at most the first N episodes (0 = all)")
    ap.add_argument("--force", action="store_true", help="relabel episodes that already have a parsed output")
    ap.add_argument('--retry-received', action='append', default=[], metavar='EPISODE',
                    help='explicitly retry this received unusable or truncated reply; repeat for each episode')
    ap.add_argument("--cell-w", type=int, default=0,
                    help="grid cell width in px for every episode, with no routing; 0 keeps the per-rig width (teleop "
                         "224 or 448 by its task text, handheld 320, head camera 256), stepped down for an episode "
                         "whose grids would pass the image-size cap")
    ap.add_argument("--example-dir", default=None,
                    help="folder with example_<rig>.json: one complete annotation per rig, shown to the model as an "
                         "example of the density expected (the model comparison's with-example runs); off by default")
    ap.add_argument("--route-seeds", type=Path, default=None,
                    help="JSON {task text: routing answer} of another run (a model comparison's reference run), used "
                         "instead of routing, so this run sends the same widths and frames (label/route.py)")
    ap.add_argument("--max-spend", type=float, default=0.0,
                    help="remaining USD allowance for this invocation, reserved before every call (0 = no cap)")
    ap.add_argument("--dry-run", action="store_true", help="build and record every request without calling the model")
    args = ap.parse_args()

    keys = get_keys() or (["dry-run"] if args.dry_run else [])
    if not keys:
        print("set OPENROUTER_API_KEYS (comma-separated OpenRouter keys) or OPENAI_API_KEY", file=sys.stderr)
        return 2
    if not args.dry_run and not is_openrouter_key(keys[0]) and args.model and not args.model.startswith("openai/"):
        print(f"{args.model} is not an OpenAI model, so it needs OPENROUTER_API_KEYS", file=sys.stderr)
        return 2
    if args.route_seeds:
        from label import route
        n = route.seed(json.loads(args.route_seeds.read_text()), args.route_seeds.name)
        print(f"routing answers seeded for {n} task texts from {args.route_seeds}", flush=True)
    label_kw = dict(model=args.model, reasoning=args.reasoning, max_tokens=args.max_tokens, timeout=args.timeout,
                    cell_w=args.cell_w, example_dir=args.example_dir, dry_run=args.dry_run,
                    sample_fps=args.sample_fps, dataset_note=args.dataset_note,
                    inspection=args.inspection)
    if args.episode_dir:
        target = args.out or (args.episode_dir / 'labels.json')
        claim = target.with_name(f'noreply_{target.name}')
        if claim.exists() and not args.force and args.episode_dir.name not in args.retry_received:
            print(f'{claim}: existing no-reply claim cannot authorize a new call; use --force explicitly',
                  file=sys.stderr)
            return 1
        return run_batch([args.episode_dir], target.parent, keys=keys, concurrency=1,
                         force=args.force, max_spend=args.max_spend, output_path=target,
                         execution_lease_fd=inherited_review_lease(target.parent),
                         retry_received=args.retry_received, **label_kw)
    if not args.out_dir:
        print("--out-dir is required with --episodes-root", file=sys.stderr)
        return 2
    episodes = discover_episodes(args.episodes_root)
    if not episodes:
        print(f"no episode folders under {args.episodes_root}", file=sys.stderr)
        return 2
    if args.limit > 0:
        episodes = episodes[:args.limit]
    return run_batch(episodes, args.out_dir, keys=keys, concurrency=args.concurrency or 4 * len(keys),
                     force=args.force, max_spend=args.max_spend, retry_received=args.retry_received,
                     execution_lease_fd=inherited_review_lease(args.out_dir), **label_kw)


if __name__ == "__main__":
    raise SystemExit(cli(main))
