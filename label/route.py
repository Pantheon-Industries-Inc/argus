"""Resolution routing: which cell width a teleop episode is sent at, chosen from its task text alone.

Most tasks read well at low resolution (whole objects, where they are, whether they moved); some can only be
judged with fine detail. Before an episode of a routed rig (label/episode.py ROUTE_WIDTHS) is labelled, a small
model reads the episode's own task text, never its frames, and says which. A task that needs fine detail gets the
wide cells, every other task the narrow cells plus contact detail views. The call costs a fraction of a cent, its
answer is recorded in the episode's output (config.resolution_route), and a failed call, an unclear answer, a dry
run or an episode with no task text all get the wide cells. Episodes with the same task text share one answer
within a run; the cost is recorded on the episode whose call it was (label.harness.episode_cost).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from label import episode as me

ROUTE_MODEL = "openai/gpt-6-sol"
ROUTE_REASONING = "low"
ROUTE_MAX_TOKENS = 2000
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


def _call_cost(usage: dict) -> float:
    """The billed cost, or on a call straight to OpenAI its list-price cost."""
    return usage.get("cost") or usage.get("list_cost") or 0.0


def route_width(ep_dir: Path, api_key: str | None, call_model, timeout: int = 120) -> tuple[int | None, dict]:
    """(widest cell width, record) for an episode of a routed rig; (None, {"routed": False}) for the others.
    call_model is label.harness.call_model; api_key None means a dry run."""
    ep = me.load(ep_dir)
    r = me.rig(ep)
    if r not in me.ROUTE_WIDTHS:
        return None, {"routed": False}
    low, high = me.ROUTE_WIDTHS[r]
    ctx = ep["context"]
    # a plain video's task label is only its file name, which says nothing about the task
    real = any((ctx.get(k) or "") for k in ("instruction", "annotation_subtasks", "uploader_annotation"))
    from_video = (ctx.get("source") or {}).get("format") == "video files"
    has_task = real or (bool(ctx.get("task_label")) and not from_video)
    if not has_task:
        return high, {"routed": True, "fine_detail": None, "why": "no task text", "cell_w": high}
    if not api_key:
        return high, {"routed": True, "fine_detail": None, "why": "no key (dry run)", "cell_w": high}
    text = route_text(ep)
    with _LOCK:
        hit = _CACHE.get(text)
    if hit is not None:
        # the answer of an earlier episode with the same task text: nothing was called, nothing was billed
        hit = dict(hit, cost_usd=0.0, cached=True)
    else:
        try:
            resp = call_model(request(ep), ROUTE_MODEL, ROUTE_REASONING, api_key, max_tokens=ROUTE_MAX_TOKENS,
                              timeout=timeout)
            msg = resp["choices"][0]["message"]["content"]
            ans = json.loads(msg[msg.find("{"):msg.rfind("}") + 1])
            fine = ans.get("fine_detail")
            hit = {"fine_detail": fine if isinstance(fine, bool) else None, "why": str(ans.get("why") or "")[:200],
                   "cost_usd": float(_call_cost(resp.get("usage") or {}))}
        except Exception as e:   # the wide cells are always safe; a failure is not cached, so the next episode retries
            hit = {"fine_detail": None, "why": f"routing failed: {str(e)[:120]}", "cost_usd": 0.0}
        else:
            with _LOCK:
                _CACHE[text] = hit
    w = low if hit["fine_detail"] is False else high
    return w, {"routed": True, "model": ROUTE_MODEL, **hit, "cell_w": w}
