"""Label episodes with a vision-language model: one request per episode, one JSON file per episode.

    python -m label.harness --episode-dir EPISODE [--out OUT.json] [--dry-run]
    python -m label.harness --episodes-root SLICE --out-dir OUT [--concurrency N] [--max-spend USD] [--dry-run]

Runs normally start through `python -m label` (label/run.py), which gives each run its own folder, records the
code commit and settings, and enforces a spend cap. This module is what it calls.

Each episode is a sidecar folder (label/episode.py builds the request from it). The request goes to OpenRouter
and carries the model id, the reasoning effort (medium by default), max_completion_tokens,
response_format json_object, a cache_control breakpoint at the end of the shared instructions, and usage
include (so the billed cost comes back). No temperature, top_p or seed is sent and no provider is pinned: each
model runs with its provider's own sampling defaults. Each output records what served the request:
provider_name, generation_id, model_served and system_fingerprint (None when the provider gives none).

Output per episode (out.json, or OUT/<episode>.json): the model's labels (`labels`, parsed; `_raw` and
`_parse_error` when the reply did not parse), `parse_ok`, what was sent (`config`: cameras, cell size, the
exact instants), the deterministic checks that ran on the episode (`dataset_checks`), the still spans the model
was told about, the instruction it was graded against, and the billed usage and cost. A reply cut off at the
output limit is kept as failed_<episode>.json for diagnosis and counts as a failure.

Keys: OPENROUTER_API_KEYS, a comma-separated list; keys are used round robin, and a key that runs out of
credit is retired for the rest of the batch. RDA_DECODE_CONCURRENCY (default 12) bounds the frame decodes
running at once across the batch. A dry run builds every request exactly as it would be sent (frames decoded,
grids composed, prompt assembled, caps checked) and writes it with a token estimate, without calling the model.
"""
from __future__ import annotations

import argparse
import base64
import json
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

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-6-astra"
DEFAULT_REASONING = "medium"
DETAIL = "high"
GRID_COLS = 4

# gpt-6-astra list prices (USD per token), used only when a response carries no billed cost.
PRICE_IN = 1.0e-5
PRICE_OUT = 5.0e-5

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


def is_key_exhausted(code: int, body: str) -> bool:
    b = body.lower()
    # 402 Payment Required (insufficient credits), or 401 on a revoked or disabled key
    return (code == 402 or ("insufficient_quota" in b) or ("insufficient credits" in b)
            or ("credit" in b and code in (402, 403)) or code == 401)


def call_model(content: list, model: str, reasoning: str, api_key: str, max_tokens: int, timeout: int) -> dict:
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
    req = urllib.request.Request(OPENROUTER_URL, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    last = None
    for attempt in range(6):
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
                time.sleep(min(60, 4 * 2 ** attempt))
                continue
            raise RuntimeError(last)
        except (urllib.error.URLError, TimeoutError) as e:
            last = str(e)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"model call failed after retries: {last}")


def label_episode(ep_dir: Path, out_path: Path, *, model: str, reasoning: str, api_key: str, max_tokens: int,
                  timeout: int, cell_w: int = 0, example_dir: str | None = None, dry_run: bool = False) -> dict:
    req = me.build_request(ep_dir, detail=DETAIL, gate=DECODE_GATE, grid_cols=GRID_COLS, cell_w=cell_w or None,
                           example_dir=example_dir)
    pl = req["plan"]
    fields = {
        "given_prompt": req["given_prompt"],
        "prompt_mode": "given" if req["given_prompt"] else "inferred",
        "task_label": req["task_label"],
        "sampling": req["sampling"],
        "arm_still_spans": req["still_spans"],
        "dataset_checks": pl["checks"],
        "example_dir": str(example_dir) if example_dir else None,
        "config": {"views": req["views"], "cam_labels": req["cam_labels"], "layout": "grid",
                   "grid_cols": GRID_COLS, "cell": req["cell"],
                   "n_timesteps": len(pl["ks"]), "n_frames_sent": len(pl["ks"]) * len(req["cam_labels"]),
                   "n_image_parts": req["n_images"], "fullres_frames": ["first", "last"],
                   "timesteps_s": req["timesteps"], "detail": DETAIL},
    }
    return _call_and_record(ep_dir, out_path, req["content"], req["image_bytes"], fields, model=model,
                            reasoning=reasoning, api_key=api_key, max_tokens=max_tokens, timeout=timeout,
                            dry_run=dry_run, prompt_text=req["prompt"])


def estimate_image_tokens(w: int, h: int) -> float:
    """Input tokens per image, fitted on 2,572 gpt-6-astra requests (about 1.1 tokens per 32x32 px patch plus 69
    per image). Used only for the dry-run estimate; billing always comes from the API's own usage."""
    return 68.8 + 1.115 * (-(-w // 32)) * (-(-h // 32))


TIMELINE_COLUMNS = ["start_s", "end_s", "arm", "action", "object", "destination", "spatial_relation",
                    "contribution", "progress", "notes"]


def parse_response(text: str) -> tuple[dict, bool]:
    """A model's reply as labels, or the raw text and the reason it could not be read. The one parser for every
    model and for re-reading stored replies (label/reparse.py)."""
    try:
        # some models ignore json_object and wrap the JSON in a markdown fence
        fenced = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", text, re.S)
        return normalize_timeline(json.loads(fenced.group(1) if fenced else text)), True
    except Exception as e:
        return {"_raw": text, "_parse_error": f"{type(e).__name__}: {e}"[:300]}, False


def normalize_timeline(labels: dict) -> dict:
    """The output format sends each timeline segment as one array in the column order of timeline_columns.
    Convert it back to one object per segment, validating every row, so everything downstream sees the usual
    format. A row with the wrong number of values, a time that is not a number, or a progress that is text
    raises: a shifted column would silently corrupt the labels. A value that is missing or outside a field's
    allowed set (a progress or a contribution of null) is kept as the model wrote it and listed in
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
            raise ValueError(f"timeline row {i} has {n} values for {len(cols)} columns")
        seg = dict(zip(cols, row))
        for k in ("start_s", "end_s", "progress"):
            if k == "progress" and seg.get(k) is None:
                labels.setdefault("_schema_violations", []).append(f"timeline row {i}: progress null")
            elif k in seg and not isinstance(seg[k], (int, float)):
                raise ValueError(f"timeline row {i}: {k} is not a number ({seg[k]!r})")
        if seg.get("contribution") not in ("advancing", "wasteful", "idle"):
            labels.setdefault("_schema_violations", []).append(
                f"timeline row {i}: contribution {seg.get('contribution')!r}")
        if seg.get("notes") in (None, ""):
            seg.pop("notes", None)
        out.append(seg)
    labels["timeline"] = out
    return labels


def _call_and_record(ep_dir: Path, out_path: Path, content: list, img_bytes: int, fields: dict, *, model: str,
                     reasoning: str, api_key: str, max_tokens: int, timeout: int, dry_run: bool = False,
                     prompt_text: str = "") -> dict:
    """Send one request (or, in a dry run, record exactly what would be sent) and write the result."""
    if img_bytes * IMAGE_SIZE_INFLATION > IMAGE_LIMIT_BYTES:
        raise RuntimeError(f"payload {img_bytes} B exceeds the image-size cap; refusing to send")
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
    t_start = time.time()
    resp = call_model(content, model, reasoning, api_key, max_tokens, timeout)
    elapsed = time.time() - t_start

    choice = (resp.get("choices") or [{}])[0]
    finish = choice.get("finish_reason") or choice.get("native_finish_reason")
    msg = choice.get("message") or {}
    text = msg.get("content") or ""
    if finish == "length":
        # the output hit max_tokens, so the JSON is cut off: keep what came back beside the outputs (never as an
        # episode_*.json, so a resumed run still counts the episode as not done), then fail
        write_atomic(Path(out_path).with_name(f"failed_{Path(out_path).name}"), {
            "episode_dir": str(ep_dir), "model": model, "reasoning_effort": reasoning, "finish_reason": finish,
            **_served(resp), "usage": resp.get("usage"), "content_tail": text[-4000:],
            "reasoning_tail": (msg.get("reasoning") or "")[-8000:]})
        raise RuntimeError(f"response truncated (finish_reason=length) at max_tokens={max_tokens}")
    # an unparseable reply is recorded as it came, never repaired or retried
    labels, parse_ok = parse_response(text)

    usage = resp.get("usage") or {}
    n_in = usage.get("prompt_tokens", 0)
    n_out = usage.get("completion_tokens", 0)
    n_reason = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
    ptd = usage.get("prompt_tokens_details") or {}
    billed = usage.get("cost")
    cost = float(billed) if billed is not None else n_in * PRICE_IN + n_out * PRICE_OUT
    result = {
        "episode_dir": str(ep_dir),
        "model": model,
        "reasoning_effort": reasoning,
        **fields,
        "provider": "openrouter",
        **_served(resp),
        "finish_reason": finish,
        "parse_ok": parse_ok,
        "labels": labels,
        "usage": {"prompt_tokens": n_in, "completion_tokens": n_out,
                  "reasoning_tokens": n_reason, "est_cost_usd": round(cost, 4),
                  "cost_source": "billed" if billed is not None else "estimate",
                  "cached_tokens": ptd.get("cached_tokens", 0), "cache_write_tokens": ptd.get("cache_write_tokens", 0),
                  "latency_s": round(elapsed, 1)},
    }
    if msg.get("reasoning"):
        # the reasoning text or summary the provider returned, when it returns one
        result["reasoning_text"] = msg["reasoning"][:20000]
    write_atomic(out_path, result)
    c = fields.get("config") or {}
    print(f"[{Path(ep_dir).name}] timesteps={c.get('n_timesteps')} images={n_images} "
          f"in={n_in} out={n_out} cost=${cost:.3f} {elapsed:.0f}s parse_ok={parse_ok} -> {out_path}", flush=True)
    return result


def _served(resp: dict) -> dict:
    """What the response says served the request: the upstream provider, OpenRouter's generation id, the model
    id that answered and the provider's system fingerprint (None when a field is absent)."""
    return {"provider_name": resp.get("provider"), "generation_id": resp.get("id"),
            "model_served": resp.get("model"), "system_fingerprint": resp.get("system_fingerprint")}


def write_atomic(out_path: Path, result: dict) -> None:
    """Write JSON through a temporary file, so a kill mid-write never leaves a truncated file that a resume would
    take for a result."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f".{out_path.name}.tmp")
    tmp.write_text(json.dumps(result, indent=2))
    os.replace(tmp, out_path)


def get_keys() -> list[str]:
    """OpenRouter keys from OPENROUTER_API_KEYS (comma-separated). Anything that is not an OpenRouter key is
    ignored, so no other provider's key is ever sent to OpenRouter."""
    return [k.strip() for k in os.environ.get("OPENROUTER_API_KEYS", "").split(",") if k.strip().startswith("sk-or-")]


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
        if force or not p.exists() or p.stat().st_size == 0:
            return False
        try:
            r = json.loads(p.read_text())
            return bool(r.get("dry_run")) if dry else bool(r.get("parse_ok"))
        except Exception:
            return False

    todo = [ep for ep in episodes if not is_done(ep)]
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
        while True:
            key = pool.take() if not dry else "dry-run"
            if key is None:
                return ("fail", ep, "all keys retired (out of credit)")
            try:
                r = label_episode(ep, out_for(ep), api_key=key, **label_kw)
                _release_memory()
                c = float((r.get("usage") or {}).get("est_cost_usd", 0.0) or 0.0)
                with spend_lock:
                    spent["usd"] += c
                return ("ok", ep, c)
            except KeyExhausted as e:
                pool.retire(key, str(e))
                continue
            except Exception as e:
                return ("fail", ep, f"{type(e).__name__}: {e}")

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for f in as_completed([ex.submit(work, ep) for ep in todo]):
            status, ep, info = f.result()
            if status == "ok":
                done += 1
                total_cost += float(info or 0)
            elif status == "skip":
                skipped += 1
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
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenRouter model id (default {DEFAULT_MODEL})")
    ap.add_argument("--reasoning", default=DEFAULT_REASONING, help="reasoning effort (default medium)")
    ap.add_argument("--max-tokens", type=int, default=64000, help="output tokens per episode (default 64000)")
    ap.add_argument("--timeout", type=int, default=600, help="seconds per request (default 600)")
    ap.add_argument("--concurrency", type=int, default=0, help="episodes in flight (default 4 per key)")
    ap.add_argument("--limit", type=int, default=0, help="label at most the first N episodes (0 = all)")
    ap.add_argument("--force", action="store_true", help="relabel episodes that already have a parsed output")
    ap.add_argument("--cell-w", type=int, default=0,
                    help="grid cell width in px; 0 keeps the per-rig width (teleop 448, handheld and head camera 256), "
                         "stepped down for an episode whose grids would pass the image-size cap")
    ap.add_argument("--example-dir", default=None,
                    help="folder with example_<rig>.json: one complete annotation per rig, shown to the model as an "
                         "example of the density expected (the model comparison's with-example runs); off by default")
    ap.add_argument("--max-spend", type=float, default=0.0,
                    help="stop starting new episodes once this many USD are spent (0 = no cap)")
    ap.add_argument("--dry-run", action="store_true", help="build and record every request without calling the model")
    args = ap.parse_args()

    keys = get_keys() or (["dry-run"] if args.dry_run else [])
    if not keys:
        print("set OPENROUTER_API_KEYS (comma-separated OpenRouter keys)", file=sys.stderr)
        return 2
    label_kw = dict(model=args.model, reasoning=args.reasoning, max_tokens=args.max_tokens, timeout=args.timeout,
                    cell_w=args.cell_w, example_dir=args.example_dir, dry_run=args.dry_run)
    if args.episode_dir:
        label_episode(args.episode_dir, args.out or (args.episode_dir / "labels.json"), api_key=keys[0], **label_kw)
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
