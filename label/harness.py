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
`_parse_error` when the reply did not parse), `parse_ok`, what was sent (`config`: cameras, cell size, the
exact instants), the deterministic checks that ran on the episode (`dataset_checks`), the still spans the model
was told about, the instruction it was graded against, and the billed usage and cost. A reply cut off at the
output limit is kept as failed_<episode>.json, with what was sent, and counts as a failure; the board shows the
episode with that reply, as it shows one whose reply did not parse. An episode that got no reply at all (the spend
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
import json
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from label import episode as me
from label.atomic import write_atomic
from label.route import route_width

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-6-astra"
DEFAULT_REASONING = "medium"
DETAIL = "high"
GRID_COLS = 4

# gpt-6-astra list prices (USD per token), used only when a response carries no billed cost.
PRICE_IN = 1.0e-5
PRICE_OUT = 5.0e-5
# OpenAI list prices (USD per token: input, cached input, output), and from LONG_PROMPT_TOKENS prompt tokens the
# long-context ones, for a call sent straight to OpenAI, whose response carries tokens but no cost
OPENAI_PRICES = {"gpt-6-astra": ((1.0e-5, 1.0e-6, 5.0e-5), (2.0e-5, 2.0e-6, 7.5e-5)),
                 "gpt-6-sol": ((2.0e-6, 2.0e-7, 1.0e-5), (4.0e-6, 4.0e-7, 1.5e-5))}
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


class KeyExhausted(RuntimeError):
    """The key has no credit left; the batch retires it and moves on."""


class Truncated(RuntimeError):
    """The reply was cut off at the output limit. It was billed, so it carries what the episode cost."""

    def __init__(self, message: str, cost: float):
        super().__init__(message)
        self.cost = cost


class SpendCap(RuntimeError):
    """The final request cannot fit under the remaining batch budget."""


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
    if not _complete_token_usage(usage):
        return None, "unverified"
    try:
        listed = openai_list_cost(model, usage)
        estimated = usage["prompt_tokens"] * PRICE_IN + usage["completion_tokens"] * PRICE_OUT
        cost = max(estimated, listed) if listed is not None else estimated
        if math.isfinite(cost) and cost >= 0:
            return cost, "estimate"
    except OverflowError:
        pass
    return None, "unverified"


def call_model(content: list, model: str, reasoning: str, api_key: str, max_tokens: int, timeout: int,
               *, attempts: int = 6) -> dict:
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
        "response_format": {"type": "json_object"},
        # ask for the billed cost (cache reads are about 1/10 price, cache writes about 1.25x), so every
        # recorded cost and spend cap uses what was actually charged
        "usage": {"include": True},
    }
    return _post(OPENROUTER_URL, body, api_key, timeout, attempts=attempts)


def call_model_once(content: list, model: str, reasoning: str, api_key: str, max_tokens: int, timeout: int) -> dict:
    """One HTTP attempt for a paid call. A failed dispatch may already have been billed."""
    return call_model(content, model, reasoning, api_key, max_tokens, timeout, attempts=1)


def _call_openai(content: list, model: str, reasoning: str, api_key: str, max_tokens: int, timeout: int,
                 *, attempts: int = 6) -> dict:
    """The same request straight to OpenAI. The response is recorded as OpenRouter's is: its provider is OpenAI
    and its usage carries list_cost, the list-price cost, where OpenRouter's carries the billed cost."""
    if not model.startswith("openai/"):
        raise RuntimeError(f"{model} is not an OpenAI model, so it needs an OpenRouter key")
    body = {"model": model.split("/", 1)[1], "messages": [{"role": "user", "content": content}],
            "max_completion_tokens": max_tokens, "reasoning_effort": reasoning,
            "response_format": {"type": "json_object"}}
    resp = _post(OPENAI_URL, body, api_key, timeout, attempts=attempts)
    resp.setdefault("provider", "OpenAI")
    if isinstance(resp.get("usage"), dict):
        resp["usage"]["list_cost"] = openai_list_cost(model, resp["usage"])
    return resp


def _post(url: str, body: dict, api_key: str, timeout: int, *, attempts: int = 6) -> dict:
    if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts < 1:
        raise ValueError("HTTP attempts must be a positive integer")
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    last = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.loads(r.read().decode())
            if not resp.get("choices"):
                # HTTP 200 carrying an error body and no completion: fail loudly, never record an empty parse
                err = json.dumps(resp.get("error") or resp)[:300]
                code = (resp.get("error") or {}).get("code") if isinstance(resp.get("error"), dict) else None
                if is_key_exhausted(code if isinstance(code, int) else 0, err):
                    raise KeyExhausted(f"200 with error body: {err}")
                raise RuntimeError(f"200 response with no choices: {err}")
            return resp
        except urllib.error.HTTPError as e:
            text = e.read().decode(errors="replace")[:500]
            last = f"HTTP {e.code}: {text[:300]}"
            if is_key_exhausted(e.code, text):
                raise KeyExhausted(last)
            if e.code in (429, 500, 502, 503, 504, 529):
                if attempt + 1 == attempts:
                    break
                time.sleep(min(60, 4 * 2 ** attempt))
                continue
            raise RuntimeError(last)
        except (urllib.error.URLError, TimeoutError) as e:
            last = str(e)
            if attempt + 1 == attempts:
                break
            time.sleep(2 ** attempt)
    raise RuntimeError(f"model call failed after {attempts} HTTP attempts: {last}")


def label_episode(ep_dir: Path, out_path: Path, *, model: str, reasoning: str, api_key: str, max_tokens: int,
                  timeout: int, cell_w: int = 0, example_dir: str | None = None, dry_run: bool = False,
                  reserve_final=None, reserve_selection=None, settle_selection=None) -> dict:
    # an explicit cell width skips the routing; otherwise a routed rig's widest cell comes from its task text
    route_w, route = (None, {"routed": False}) if cell_w else route_width(
        ep_dir, None if dry_run else api_key, call_model, timeout=min(timeout, 120))
    req = me.build_request(ep_dir, detail=DETAIL, gate=DECODE_GATE, grid_cols=GRID_COLS, cell_w=cell_w or None,
                           max_cell_w=route_w, example_dir=example_dir, inspect_evidence=True)
    pl = req["plan"]
    fields = {
        "input_identity": input_identity(ep_dir, model=model, reasoning=reasoning, max_tokens=max_tokens,
                                          cell_w=cell_w, example_dir=example_dir),
        **({"sensor_evidence": req["sensor_evidence"]} if req.get("sensor_evidence", {}).get("sensors") else {}),
        "given_prompt": req["given_prompt"],
        "prompt_mode": "given" if req["given_prompt"] else "inferred",
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
                   "grid_cols": req["grid_cols"], "cell": req["cell"],
                   "n_timesteps": len(pl["ks"]), "n_frames_sent": len(pl["ks"]) * len(req["cam_labels"]),
                   "n_image_parts": req["n_images"], "fullres_frames": ["first", "last"],
                   "contact_detail_s": req["contact_s"], "resolution_route": route,
                   "timesteps_s": req["timesteps"], "detail": DETAIL, "circular_image": req["lens"],
                   "prompt_blocks": req["blocks"], "schema_fields": req["schema_fields"],
                   "checks_implied": req["checks_implied"]},
    }
    return _call_and_record(ep_dir, out_path, req["content"], req["image_bytes"], fields, model=model,
                            reasoning=reasoning, api_key=api_key, max_tokens=max_tokens, timeout=timeout,
                            dry_run=dry_run, prompt_text=req["prompt"], evidence_access=req["evidence_access"],
                            reserve_final=reserve_final, reserve_selection=reserve_selection,
                            settle_selection=settle_selection)


def input_identity(ep_dir: Path, *, model: str, reasoning: str, max_tokens: int, cell_w: int,
                   example_dir: str | None) -> dict:
    """Bounded proof for source files and the settings that shape a paid annotation request."""
    from label.evidence_access import source_proof
    ep_dir = Path(ep_dir)
    ctx = json.loads((ep_dir / "context.json").read_text())
    examples = []
    if example_dir:
        for path in sorted(Path(example_dir).glob("example_*.json")):
            stat = path.stat()
            examples.append({"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    settings = {"model": model, "reasoning": reasoning, "max_tokens": max_tokens, "cell_w": cell_w,
                "detail": DETAIL, "grid_cols": GRID_COLS, "examples": examples}
    return {"version": 1, "source_proof": source_proof(ctx, ep_dir), "settings": settings}


def same_input_identity(saved: dict, current: dict) -> bool:
    from label.evidence_access import equivalent_source_proof
    return (isinstance(saved, dict) and saved.get('version') == current.get('version')
            and saved.get('settings') == current.get('settings')
            and equivalent_source_proof(saved.get('source_proof'), current.get('source_proof')))


def episode_cost(result: dict) -> float:
    """What labelling one episode was billed: its model call plus, for a routed episode, the routing call made for
    it (label/route.py)."""
    route = (result.get("config") or {}).get("resolution_route") or {}
    inspection = result.get('evidence_inspection') or {}
    final = (result.get("usage") or {}).get("est_cost_usd")
    if final is None:
        final = result.get('final_reserved_usd') or 0.0
    return (float(final) + float(route.get("cost_usd") or 0.0)
            + float(inspection.get('cost_usd') or 0.0))


def final_cost_bound(content: list, max_tokens: int) -> float:
    """Reserve a generous input allowance and the full output limit before dispatch."""
    import io
    from PIL import Image
    text_bytes = sum(len(c.get('text', '').encode('utf-8')) for c in content if c.get('type') == 'text')
    images = 0.0
    for part in content:
        if part.get('type') == 'image_url':
            raw = base64.b64decode(part['image_url']['url'].split(',', 1)[1])
            with Image.open(io.BytesIO(raw)) as image:
                images += estimate_image_tokens(*image.size)
    return round((text_bytes + 4 * images + 1024) * max(PRICE_IN, 2.0e-5)
                 + max_tokens * max(PRICE_OUT, 7.5e-5), 6)


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


def estimate_image_tokens(w: int, h: int) -> float:
    """Input tokens per image, fitted on 2,572 gpt-6-astra requests (about 1.1 tokens per 32x32 px patch plus 69
    per image). Used only for the dry-run estimate; billing always comes from the API's own usage."""
    return 68.8 + 1.115 * (-(-w // 32)) * (-(-h // 32))


TIMELINE_COLUMNS = ["start_s", "end_s", "arm", "action", "object", "destination", "spatial_relation",
                    "contribution", "progress", "notes"]


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
        return typed_labels(normalize_timeline(labels)), True
    except Exception as e:
        return {"_raw": text, "_parse_error": f"{type(e).__name__}: {e}"[:300]}, False


def json_kind(x) -> str:
    """What a JSON value is, in the output format's words, for the board's reason a field was left out."""
    if isinstance(x, bool):
        return "true or false"
    return ("null" if x is None else "a number" if isinstance(x, (int, float)) else "text" if isinstance(x, str)
            else "a list" if isinstance(x, list) else "an object" if isinstance(x, dict) else type(x).__name__)


def _drop(labels: dict, field: str, row, why: str) -> None:
    """Records a field or a row of the reply left out because it breaks the output format."""
    labels.setdefault("_dropped", []).append({"field": field, **({"row": row} if row is not None else {}), "why": why})


def normalize_timeline(labels: dict) -> dict:
    """The output format sends each timeline segment as one array in the column order of timeline_columns.
    Convert it back to one object per segment, validating every row, so everything downstream sees the usual
    format. A time or a progress written as a number in text ("2.0") is read as that number. A row with the wrong
    number of values, or a time or a progress that is not a number, is left out and recorded in _dropped (a shifted
    column would silently corrupt the labels), and the other rows are kept. A value that is missing or outside a
    field's allowed set (a progress or a contribution of null) is kept as the model wrote it and listed in
    _schema_violations, so one bad field is counted, not a reason to discard a valid answer."""
    tl = labels.get("timeline")
    if not isinstance(tl, list) or not any(isinstance(r, list) for r in tl):
        labels.pop("timeline_columns", None)
        return labels
    cols = labels.pop("timeline_columns", None) or TIMELINE_COLUMNS
    out = []
    for i, row in enumerate(tl):
        if isinstance(row, dict):
            out.append(row)
            continue
        if not isinstance(row, list) or not (len(cols) - 1 <= len(row) <= len(cols)):
            n = len(row) if isinstance(row, list) else "?"
            _drop(labels, "timeline", i, f"it has {n} values for {len(cols)} columns")
            continue
        seg = dict(zip(cols, row))
        bad = None
        for k in ("start_s", "end_s", "progress"):
            if k == "progress" and seg.get(k) is None:
                labels.setdefault("_schema_violations", []).append(f"timeline row {i}: progress null")
            elif k in seg and (isinstance(seg[k], bool) or not isinstance(seg[k], (int, float))):
                v = me.number(seg[k])
                if v is None:
                    bad = f"its {k} is not a number ({seg[k]!r})"
                    break
                seg[k] = v
        if bad:
            _drop(labels, "timeline", i, bad)
            continue
        if seg.get("contribution") not in ("advancing", "wasteful", "idle"):
            labels.setdefault("_schema_violations", []).append(
                f"timeline row {i}: contribution {seg.get('contribution')!r}")
        if seg.get("notes") in (None, ""):
            seg.pop("notes", None)
        out.append(seg)
    labels["timeline"] = out
    return labels


# the reply's fields by the type the output format gives them (label/prompts.py)
LIST_FIELDS = ("timeline", "key_events", "state_changes", "scene_graph", "recovery", "data_issues", "operator_mistakes",
               "tasks", "contacts", "contacts_missing", "sensor_findings", "evidence_findings")
DICT_FIELDS = ("scene", "completion", "goal_alignment")
TEXT_FIELDS = ("task_summary", "performance_review", "viewpoint")
TIME_FIELDS = ("t_s", "start_s", "end_s", "completed_at_s", "goal_reached_at_s", "undone_at_s", "failure_t_s",
               "recovered_at_s")


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
    for k in ("completion", "goal_alignment"):
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


def _call_and_record(ep_dir: Path, out_path: Path, content: list, img_bytes: int, fields: dict, *, model: str,
                     reasoning: str, api_key: str, max_tokens: int, timeout: int, dry_run: bool = False,
                     prompt_text: str = "", evidence_access=None, reserve_final=None,
                     reserve_selection=None, settle_selection=None) -> dict:
    """Send one request (or, in a dry run, record exactly what would be sent) and write the result."""
    if img_bytes * IMAGE_SIZE_INFLATION > IMAGE_LIMIT_BYTES:
        raise RuntimeError(f"payload {img_bytes} B exceeds the image-size cap; refusing to send")
    if evidence_access is not None:
        from label import evidence_access as ea
        fields = copy.deepcopy(fields)
        fields['evidence_inspection'] = evidence_access.record()
        if not dry_run:
            def select(parts):
                response = call_model_once(parts, model, reasoning, api_key, ea.MAX_SELECTION_TOKENS, min(timeout, 120))
                usage = response.get('usage') or {}
                if any(type(usage.get(k)) in (int, float) for k in ('cost', 'list_cost')) or all(
                        type(usage.get(k)) in (int, float) for k in ('prompt_tokens', 'completion_tokens')):
                    response.setdefault('usage', {})['cost'] = _cost(usage)
                return response
            # Reserve the maximum output and a conservative input allowance before each selection dispatch.
            def reserve(parts):
                import io
                from PIL import Image
                text_tokens = sum(len(c.get('text', '')) for c in parts) / 3
                image_tokens = 0
                for part in parts:
                    if part.get('type') == 'image_url':
                        raw = base64.b64decode(part['image_url']['url'].split(',', 1)[1])
                        with Image.open(io.BytesIO(raw)) as image:
                            image_tokens += estimate_image_tokens(*image.size)
                return (text_tokens + image_tokens) * PRICE_IN + ea.MAX_SELECTION_TOKENS * PRICE_OUT
            cache = Path(out_path).parent / '.evidence' / (Path(out_path).name + '.inspection.json')
            try:
                doc, extra = ea.discover(evidence_access, content, select, cache, reserve_cost=reserve,
                                        reserve_dispatch=reserve_selection, settle_dispatch=settle_selection,
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
                fields['evidence_inspection'] = doc
                failed = {'episode_dir': str(ep_dir), 'model': model, 'reasoning_effort': reasoning,
                          **fields, 'parse_ok': False, 'no_reply': f'{type(error).__name__}: {error}',
                          'final_dispatch_outcome': 'not dispatched'}
                write_atomic(Path(out_path).with_name(f'noreply_{Path(out_path).name}'), failed)
                raise
            fields['evidence_inspection'] = doc
            addition = [{'type': 'text', 'text': ea.FINAL + '\nINSPECTION COVERAGE\n' +
                        ea.packed(ea.Access.coverage_summary(doc))}] + extra
            # Extra images must fit the existing request cap. Their receipts alone cannot authorize image claims.
            extra_images = [p for p in addition if p.get('type') == 'image_url']
            extra_bytes = sum(len(base64.b64decode(p['image_url']['url'].split(',', 1)[1])) for p in extra_images)
            if (img_bytes + extra_bytes) * IMAGE_SIZE_INFLATION > IMAGE_LIMIT_BYTES or (
                    sum(p.get('type') == 'image_url' for p in content) + len(extra_images) > MAX_IMAGES):
                fields['evidence_inspection']['limitations'].append('Inspected image batch exceeds final request cap; image findings withheld.')
                for r in evidence_access.receipts:
                    if r['mode'] in ('images', 'regions'):
                        r['withheld_from_final'] = True
                fields['evidence_inspection']['inspections'] = ea.clean(evidence_access.receipts)
                addition = [p for p in addition if p.get('type') != 'image_url' and not (
                    p.get('type') == 'text' and (p.get('text', '').startswith('Inspected ') or (
                        p.get('text', '').startswith('INSPECTED EVIDENCE\n') and
                        json.loads(p['text'].split('\n', 1)[1]).get('mode') in ('images', 'regions'))))]
                extra_bytes = 0
            content = content + addition
            img_bytes += extra_bytes
            prompt_text += '\n' + '\n'.join(p['text'] for p in addition if p.get('type') == 'text')
    n_images = sum(1 for c in content if c.get("type") == "image_url")
    if n_images > MAX_IMAGES:
        raise RuntimeError(f"{n_images} images exceeds the {MAX_IMAGES}-image cap; refusing to send")
    if dry_run:
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
        write_atomic(out_path, result)
        return result
    bound = final_cost_bound(content, max_tokens)
    claim_path = Path(out_path).with_name(f'noreply_{Path(out_path).name}')
    claim = {"episode_dir": str(ep_dir), "model": model, "reasoning_effort": reasoning, **fields,
             "parse_ok": False, "no_reply": "final model call has no verified response",
             "final_dispatch_outcome": "claimed", "final_reserved_usd": bound}
    if reserve_final is None:
        persist_final_claim(claim_path, claim)
    else:
        try:
            reserve_final(episode_cost(claim), claim_path, claim)
        except SpendCap as error:
            write_atomic(claim_path, {**claim, "no_reply": str(error),
                                      "final_dispatch_outcome": "not dispatched", "final_reserved_usd": 0.0})
            raise
    t_start = time.time()
    try:
        resp = call_model_once(content, model, reasoning, api_key, max_tokens, timeout)
    except KeyExhausted:
        claim_path.unlink(missing_ok=True)
        raise
    except Exception as error:
        failed = dict(claim, no_reply=f"{type(error).__name__}: {error}",
                      final_dispatch_outcome="unverified")
        write_atomic(claim_path, failed)
        raise
    elapsed = time.time() - t_start

    choice = (resp.get("choices") or [{}])[0]
    finish = choice.get("finish_reason") or choice.get("native_finish_reason")
    msg = choice.get("message") or {}
    text = msg.get("content") or ""
    if finish == "length":
        # the output hit max_tokens, so the JSON is cut off: keep what came back beside the outputs (never as an
        # episode_*.json, so a resumed run still counts the episode as not done), then fail
        raw_usage = resp.get("usage")
        usage = dict(raw_usage) if isinstance(raw_usage, dict) else {}
        cost, source = _verified_final_cost(usage, model)
        usage["est_cost_usd"] = cost
        usage["cost_source"] = source
        # with every field of the request, so the board shows the episode's checks and undecodable stretches beside
        # the cut-off reply (board/to_board.py label_failed)
        failed = {"episode_dir": str(ep_dir), "model": model, "reasoning_effort": reasoning, **fields,
                  "finish_reason": finish, **_served(resp), "usage": usage, "content_tail": text[-4000:],
                  "reasoning_tail": (msg.get("reasoning") or "")[-8000:]}
        if cost is None:
            failed.update(final_dispatch_outcome="unverified", final_reserved_usd=bound,
                          usage_reported=raw_usage)
        write_atomic(Path(out_path).with_name(f"failed_{Path(out_path).name}"), failed)
        claim_path.unlink(missing_ok=True)
        raise Truncated(f"response truncated (finish_reason=length) at max_tokens={max_tokens}", episode_cost(failed))
    # an unparseable reply is recorded as it came, never repaired or retried
    labels, parse_ok = parse_response(text)

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
        "usage": {"prompt_tokens": n_in, "completion_tokens": n_out,
                  "reasoning_tokens": n_reason, "est_cost_usd": cost,
                  "cost_source": cost_source,
                  "cached_tokens": ptd.get("cached_tokens"), "cache_write_tokens": ptd.get("cache_write_tokens"),
                  "latency_s": round(elapsed, 1)},
    }
    if cost is None:
        result.update(final_dispatch_outcome="unverified", final_reserved_usd=bound,
                      usage_reported=raw_usage)
    if isinstance(fields.get("sensor_evidence"), dict):
        from label.sensor_evidence import bind
        result["sensor_evidence"] = bind(fields["sensor_evidence"], labels.get("sensor_findings"))
        result["sensor_evidence"]["provenance"] = {"model": result.get("model_served") or model,
            "generation_id": result.get("generation_id"), "method": "same episode annotation request"}
    if evidence_access is not None:
        additional = evidence_access.bind(labels.get('evidence_findings'))
        result['evidence_inspection']['untimed_findings'] = additional['untimed_findings']
        evidence = result.setdefault('sensor_evidence', additional)
        if evidence is not additional:
            for key in ('sensors', 'series', 'findings', 'unbound_findings', 'coverage', 'limitations'):
                evidence.setdefault(key, []).extend(additional[key])
        evidence.setdefault('provenance', {}).update(model=result.get('model_served') or model,
            generation_id=result.get('generation_id'), method='episode annotation with bounded evidence inspection')
    if msg.get("reasoning"):
        # the reasoning text or summary the provider returned, when it returns one
        result["reasoning_text"] = msg["reasoning"][:20000]
    write_atomic(out_path, result)
    claim_path.unlink(missing_ok=True)
    c = fields.get("config") or {}
    cost_text = f"cost=${cost:.3f}" if cost is not None else f"reserved=${bound:.3f} cost=unverified"
    print(f"[{Path(ep_dir).name}] timesteps={c.get('n_timesteps')} images={n_images} "
          f"in={n_in} out={n_out} {cost_text} {elapsed:.0f}s parse_ok={parse_ok} -> {out_path}", flush=True)
    return result


def _cost(usage: dict) -> float:
    """The billed cost of a call, or the list-price estimate when the response carries none (list_cost on a call
    straight to OpenAI)."""
    if usage.get("cost") is not None:
        return float(usage["cost"])
    if usage.get("list_cost") is not None:
        return float(usage["list_cost"])
    return usage.get("prompt_tokens", 0) * PRICE_IN + usage.get("completion_tokens", 0) * PRICE_OUT


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


def run_batch(episodes: list[Path], out_dir: Path, *, keys: list[str], concurrency: int, force: bool,
              max_spend: float = 0.0, **label_kw) -> int:
    """Label many episodes concurrently. Resumable: an episode with a parsed output is skipped unless force. Once
    the recorded cost reaches max_spend no new episode starts (episodes in flight finish, so the overshoot is at
    most `concurrency` episodes)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = KeyPool(keys)
    dry = bool(label_kw.get("dry_run"))

    def out_for(ep: Path) -> Path:
        return out_dir / f"{ep.name}.json"

    def is_done(ep: Path) -> bool:
        p = out_for(ep)
        if force:
            return False
        candidates = [p, out_dir / f"failed_{ep.name}.json", out_dir / f"noreply_{ep.name}.json"]
        if p.exists():
            try:
                primary = json.loads(p.read_text())
                if primary.get("parse_ok") or primary.get("dry_run"):
                    candidates = [p]
            except (OSError, ValueError, AttributeError):
                pass
        existing = [q for q in candidates if q.exists() and q.stat().st_size]
        if not existing:
            return False
        records = []
        for q in existing:
            try:
                record = json.loads(q.read_text())
            except (OSError, ValueError):
                stale[ep.name] = f"saved output {q.name} cannot be verified"
                return True
            if not isinstance(record, dict):
                stale[ep.name] = f"saved output {q.name} is not a JSON object"
                return True
            records.append((q, record))
        if len(records) == 1 and records[0][0] == p and records[0][1].get('input_identity') is None:
            unverified[ep.name] = f"saved output {p.name} predates input proof"
            return True
        if len(records) == 1 and records[0][0].name.startswith('noreply_') and (
                records[0][1].get('input_identity') is None and
                records[0][1].get('final_dispatch_outcome') is None):
            unverified[ep.name] = 'legacy no-reply record has no source proof or dispatch receipt'
            return False
        try:
            current = input_identity(ep, model=label_kw.get("model", DEFAULT_MODEL),
                                     reasoning=label_kw.get("reasoning", DEFAULT_REASONING),
                                     max_tokens=label_kw.get("max_tokens", 64000), cell_w=label_kw.get("cell_w", 0),
                                     example_dir=label_kw.get("example_dir"))
        except (OSError, ValueError) as error:
            stale[ep.name] = f"input cannot be verified: {type(error).__name__}: {error}"
            return True
        for q, record in records:
            saved = record.get("input_identity")
            if saved is None:
                unverified[ep.name] = f"saved output {q.name} predates input proof"
                return True
            if not same_input_identity(saved, current) or any(item.get("missing") for item in current["source_proof"]):
                stale[ep.name] = f"saved output {q.name} has changed or missing input; use --force to refresh"
                return True
        if any(record.get('final_dispatch_outcome') in ('claimed', 'unverified') for _, record in records):
            return True
        if not p.exists():
            return False
        record = json.loads(p.read_text())
        return bool(record.get("dry_run")) if dry else bool(record.get("parse_ok"))

    stale, unverified = {}, {}
    todo = [ep for ep in episodes if not is_done(ep)]
    for name, reason in {**unverified, **stale}.items():
        write_atomic(out_dir / f"stale_{name}.json", {"episode": name, "status": "stale" if name in stale else
                     "unverified", "reason": reason, "refresh": "--force"})
    if stale:
        for name, reason in stale.items():
            print(f"STALE {name}: {reason}", file=sys.stderr, flush=True)
        return 1
    skipped = len(episodes) - len(todo)
    print(f"episodes={len(episodes)} skipped(done)={skipped} todo={len(todo)} "
          f"keys={len(keys)} concurrency={concurrency} dry_run={dry}", flush=True)
    if not todo:
        return 0
    done = failed = 0
    total_cost = 0.0
    fails = []
    spent = {"usd": 0.0}
    spend_lock = threading.Lock()

    def work(ep):
        if max_spend > 0:
            with spend_lock:
                if spent["usd"] >= max_spend:
                    return ("skip", ep, f"spend cap ${max_spend:.2f} reached")
        reserved = [0.0]
        selection_accounted = [0.0]
        def reserve_selection(amount):
            with spend_lock:
                if max_spend > 0 and spent['usd'] + amount > max_spend:
                    return False
                spent['usd'] += amount
                selection_accounted[0] += amount
                return True
        def settle_selection(reserve, actual):
            with spend_lock:
                spent['usd'] += actual - reserve
                selection_accounted[0] += actual - reserve
        def reserve_final(amount, path, claim):
            with spend_lock:
                new_amount = max(0.0, amount - selection_accounted[0])
                if max_spend > 0 and spent['usd'] + new_amount > max_spend:
                    raise SpendCap(f"final request reserve ${new_amount:.2f} exceeds remaining spend cap")
                persist_final_claim(path, claim)
                spent['usd'] += new_amount
                reserved[0] = new_amount
        while True:
            key = pool.take() if not dry else "dry-run"
            if key is None:
                return ("fail", ep, "all keys retired (out of credit)")
            try:
                r = label_episode(ep, out_for(ep), api_key=key, reserve_final=reserve_final,
                                  reserve_selection=reserve_selection, settle_selection=settle_selection, **label_kw)
                _release_memory()
                c = episode_cost(r)
                with spend_lock:
                    spent["usd"] += c - reserved[0] - selection_accounted[0]
                return ("ok", ep, c)
            except KeyExhausted as e:
                with spend_lock:
                    spent['usd'] -= reserved[0]
                    reserved[0] = 0.0
                pool.retire(key, str(e))
                continue
            except Truncated as e:
                with spend_lock:
                    spent["usd"] += e.cost - reserved[0] - selection_accounted[0]
                return ("fail", ep, str(e))
            except SpendCap as e:
                return ("skip", ep, str(e))
            except Exception as e:
                failure = out_dir / f"noreply_{ep.name}.json"
                if failure.exists() and not reserved[0]:
                    with spend_lock:
                        spent['usd'] += episode_cost(json.loads(failure.read_text()))
                return ("fail", ep, f"{type(e).__name__}: {e}")

    def no_reply(ep: Path, why: str | None) -> None:
        """noreply_<episode>.json beside the outputs: why the episode got no reply (the spend cap reached, every key out
        of credit, a request that could not be built or a call that failed), so the board shows the episode and says
        why (board/to_board.py label_outputs). Removed once the episode has a reply; a dry run writes none."""
        p = out_dir / f"noreply_{ep.name}.json"
        if dry:
            return
        if why is None:
            p.unlink(missing_ok=True)
            (out_dir / f"stale_{ep.name}.json").unlink(missing_ok=True)
            return
        previous = json.loads(p.read_text()) if p.exists() else {}
        write_atomic(p, {**previous, "episode_dir": str(ep), "model": label_kw.get("model"),
                         "parse_ok": False, "no_reply": why})

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for f in as_completed([ex.submit(work, ep) for ep in todo]):
            status, ep, info = f.result()
            no_reply(ep, None if status == "ok" else str(info))
            if status == "ok":
                done += 1
                total_cost += float(info or 0)
            elif status == "skip":
                skipped += 1
                try:
                    total_cost += episode_cost(json.loads((out_dir / f"noreply_{ep.name}.json").read_text()))
                except (OSError, ValueError, TypeError, AttributeError):
                    pass
                print(f"SKIP {ep.name}: {info}", file=sys.stderr, flush=True)
            else:
                failed += 1
                fails.append(f"FAIL {ep.name}: {info}")
                print(f"FAIL {ep.name}: {info}", file=sys.stderr, flush=True)
    print(f"\ndone={done} failed={failed} skipped={skipped} "
          f"total_cost=${total_cost:.2f} out_dir={out_dir} retired_keys={len(pool.dead)}", flush=True)
    if fails:
        print("failures:\n  " + "\n  ".join(fails), file=sys.stderr)
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--episode-dir", type=Path, help="label one episode")
    g.add_argument("--episodes-root", type=Path, help="label every episode_* folder under here")
    ap.add_argument("--out", type=Path, help="output JSON with --episode-dir (default EPISODE/labels.json)")
    ap.add_argument("--out-dir", type=Path, help="output folder with --episodes-root, one <episode>.json each")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenRouter model id (default {DEFAULT_MODEL}); an "
                                                          "OpenAI key runs only openai/ models")
    ap.add_argument("--reasoning", default=DEFAULT_REASONING, help="reasoning effort (default medium)")
    ap.add_argument("--max-tokens", type=int, default=64000, help="output tokens per episode (default 64000)")
    ap.add_argument("--timeout", type=int, default=600, help="seconds per request (default 600)")
    ap.add_argument("--concurrency", type=int, default=0, help="episodes in flight (default 4 per key)")
    ap.add_argument("--limit", type=int, default=0, help="label at most the first N episodes (0 = all)")
    ap.add_argument("--force", action="store_true", help="relabel episodes that already have a parsed output")
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
                    help="stop starting new episodes once this many USD are spent (0 = no cap)")
    ap.add_argument("--dry-run", action="store_true", help="build and record every request without calling the model")
    args = ap.parse_args()

    keys = get_keys() or (["dry-run"] if args.dry_run else [])
    if not keys:
        print("set OPENROUTER_API_KEYS (comma-separated OpenRouter keys) or OPENAI_API_KEY", file=sys.stderr)
        return 2
    if not args.dry_run and not is_openrouter_key(keys[0]) and not args.model.startswith("openai/"):
        print(f"{args.model} is not an OpenAI model, so it needs OPENROUTER_API_KEYS", file=sys.stderr)
        return 2
    if args.route_seeds:
        from label import route
        n = route.seed(json.loads(args.route_seeds.read_text()), args.route_seeds.name)
        print(f"routing answers seeded for {n} task texts from {args.route_seeds}", flush=True)
    label_kw = dict(model=args.model, reasoning=args.reasoning, max_tokens=args.max_tokens, timeout=args.timeout,
                    cell_w=args.cell_w, example_dir=args.example_dir, dry_run=args.dry_run)
    if args.episode_dir:
        target = args.out or (args.episode_dir / 'labels.json')
        claim = target.with_name(f'noreply_{target.name}')
        if claim.exists() and not args.force:
            print(f'{claim}: existing no-reply claim cannot authorize a new call; use --force explicitly',
                  file=sys.stderr)
            return 1
        label_episode(args.episode_dir, target, api_key=keys[0], **label_kw)
        return 0
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
                     force=args.force, max_spend=args.max_spend, **label_kw)


if __name__ == "__main__":
    raise SystemExit(main())
