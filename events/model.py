"""The one seam between the passes and a model: a Caller takes the request content and a purpose and returns a Reply.

On a laptop or in the bench, `openrouter(...)` sends each call through label.harness.call_model (OpenRouter, or
OpenAI with an OpenAI key) and reads the billed cost back. In production the platform passes its own Caller, which
sends the same content through its metered, traced and rate-limited effect door, so nothing here knows about it.
"""
from __future__ import annotations

import http.client
import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol


@dataclass
class Reply:
    text: str
    usage: dict = field(default_factory=dict)
    cost: float | None = None
    served: dict = field(default_factory=dict)
    seconds: float = 0.0


class Caller(Protocol):
    def __call__(self, content: list, purpose: str) -> Reply: ...


class SpendCapReached(RuntimeError):
    pass


def parse_json(text: str):
    """The reply's JSON object: the whole text, or the outermost {...} inside a fenced or chatty reply."""
    text = (text or "").strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if fenced:
        text = fenced.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        a, b = text.find("{"), text.rfind("}")
        if a >= 0 and b > a:
            try:
                return json.loads(text[a:b + 1])
            except json.JSONDecodeError:
                return None
    return None


class Spend:
    """A shared spend counter across threads; a call that would start past the cap raises SpendCapReached."""

    def __init__(self, cap_usd: float | None):
        self.cap = cap_usd
        self.spent = 0.0
        self.lock = threading.Lock()

    def check(self) -> None:
        with self.lock:
            if self.cap is not None and self.spent >= self.cap:
                raise SpendCapReached(f"spend cap ${self.cap:.2f} reached (${self.spent:.4f} spent)")

    def add(self, usd: float | None) -> None:
        with self.lock:
            self.spent += usd or 0.0


def openrouter(model: str, *, reasoning: str | None = "low", max_tokens: int = 32000, timeout: int = 600,
               spend: Spend | None = None, on_reply: Callable[[str, list, Reply], None] | None = None) -> Caller:
    from label import harness
    keys = harness.get_keys()
    if not keys:
        raise SystemExit("set OPENROUTER_API_KEYS (or OPENAI_API_KEY for OpenAI models)")

    def call(content: list, purpose: str) -> Reply:
        if spend is not None:
            spend.check()
        t0 = time.monotonic()
        for attempt in range(3):
            try:
                resp = harness.call_model(content, model, reasoning or "low", keys[0], max_tokens, timeout)
                break
            except (http.client.HTTPException, ConnectionError) as e:
                # a connection dropped mid-answer (long image requests through a proxy): the call may have been
                # billed, so this is a laptop and bench convenience; production calls go through the platform's
                # effect door, which never resends an unsettled paid call
                if attempt == 2:
                    raise
                time.sleep(5 * (attempt + 1))
        usage = resp.get("usage") or {}
        msg = (resp.get("choices") or [{}])[0].get("message") or {}
        reply = Reply(text=msg.get("content") or "", usage=usage, cost=harness._cost(usage),
                      served=harness._served(resp), seconds=time.monotonic() - t0)
        if spend is not None:
            spend.add(reply.cost)
        if on_reply is not None:
            on_reply(purpose, content, reply)
        return reply

    return call
