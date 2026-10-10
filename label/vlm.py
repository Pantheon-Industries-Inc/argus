"""The VLMs this harness prescribes, and how densely each one is shown an episode.

Two models are prescribed. Astra (openai/gpt-6-astra) is the model the harness was built and tuned on, and it keeps
the per-rig sampling it was tuned at (label/episode.py SAMPLE_EVERY_S, label/pieces.py PIECE_MAX_S). GPT-6.1 Sol
(openai/gpt-6.1-sol, at high reasoning) is the other. Every model other than Astra is sent 2 instants per second on
every rig, and a long recording is labelled in parts of at most 330 s. At 1 instant per second, GPT-6.1 Sol invented
operator mistakes that the frames between its samples disproved (on UMI handheld footage, 7 of 20 blind-checked
claims were false at 1 per second and 1 of 20 at 2 per second), and 330 s parts at 2 per second are what that run
used. Other models still run (the model comparison, compare/), under the same rule as GPT-6.1 Sol.
"""
from __future__ import annotations

ASTRA = "openai/gpt-6-astra"
SOL61 = "openai/gpt-6.1-sol"
PRESCRIBED = {
    ASTRA: {"name": "Astra", "reasoning": "medium"},
    SOL61: {"name": "GPT-6.1 Sol", "reasoning": "high"},
}
OTHER_EVERY_S = 0.5         # 2 instants per second, on every rig, for every model but Astra
OTHER_PART_MAX_S = 330.0    # the longest part such a model labels in one request


def is_astra(model: str | None) -> bool:
    """Astra under any of its names: the OpenRouter id, a dated id it is served as, or the bare OpenAI one. No model
    given means the harness default, which is Astra."""
    return model is None or "gpt-6-astra" in model


def reasoning_for(model: str, default: str) -> str:
    """The reasoning effort a prescribed model runs at; any other model gets the caller's default."""
    return PRESCRIBED.get(model, {}).get("reasoning", default)
