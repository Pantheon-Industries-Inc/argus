"""One name for each task verb, for the home page's verb counts and diversity chart.

    python -m board verbs BOARD              merges every verb not merged yet (board/follow.py runs it as labels land)

The labeler names each task's predominant action in one or two words (label/prompts.py: an episode's task_verb, a
session task's verb). Freeform tasks are named in the labeler's own words, so the same action can come back as
"pick" and "pick up", or "flip" and "turn over". A small model is given the list of distinct verbs, never the tasks,
and gives each the name its action is counted under, choosing among names already in use where one fits. A verb once
merged is never asked again, so the names stay fixed as labels land; the answers are kept in BOARD/verbs.json, and
merging every verb of a 100 hour run costs about a cent. A verb the model leaves out, or a failed call, counts under
its own words until the next run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

MODEL = "openai/gpt-6-sol"
REASONING = "low"
MAX_TOKENS = 16000
BATCH = 300                 # verbs per call
PROMPT = (
    "Each line below is a verb a labeler used to name the main action of a task in tabletop and household "
    "manipulation footage. Several lines can name the same action in different words ('pick' and 'pick up', 'stand' "
    "and 'stand up', 'turn over' and 'flip'). Give each line the one name, in one or two words, that its action "
    "should be counted under, so that lines naming the same action get the same name and lines naming different "
    "actions keep different names. Choose among the lines' own words, and use a name already in use (listed first) "
    "whenever it names the same action. Reply with JSON only: "
    '{"verbs": {"<line>": "<name>"}}, every line spelled exactly as given.\n\n')


NO_VERB = {"none", "null", "n/a", "na", "no action", "nothing", "idle"}   # what a labeler writes when no action happened


def verb_of(raw) -> str | None:
    """A verb as the labeler wrote it, lower case and single spaced; None when it wrote none, or wrote that no action
    happened ("none": an episode where the operator does nothing has no verb to count)."""
    v = " ".join(str(raw).lower().split()) if isinstance(raw, str) else ""
    return v if v and v not in NO_VERB else None


def verbs_seen(qa: Path) -> set:
    """Every verb the labeler wrote in BOARD/qa: each session task's verb, or the episode's task_verb."""
    out = set()
    for p in qa.glob("*.json"):
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(d, dict):
            continue
        tasks = [t for t in d.get("tasks") or [] if isinstance(t, dict)]
        for raw in ([t.get("verb") for t in tasks] if tasks else [d.get("task_verb")]):
            v = verb_of(raw)
            if v:
                out.add(v)
    return out


def load(path: Path) -> dict:
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"verbs": {}, "cost_usd": 0.0}
    return {"verbs": d.get("verbs") or {}, "cost_usd": float(d.get("cost_usd") or 0.0)}


def load_names(path: Path) -> dict:
    """{verb as written: the name it is counted under} from BOARD/verbs.json."""
    return load(path)["verbs"]


def merge(board: Path, timeout: int = 180) -> tuple[int, float]:
    """Gives a name to each verb in BOARD/qa that BOARD/verbs.json lacks. Returns (verbs merged, dollars spent)."""
    from label.harness import _cost, call_model, get_keys
    path = board / "verbs.json"
    have = load(path)
    todo = sorted(verbs_seen(board / "qa") - set(have["verbs"]))
    keys = get_keys()
    if not todo or not keys:
        return 0, 0.0
    merged, spent = 0, 0.0
    for i in range(0, len(todo), BATCH):
        part = todo[i:i + BATCH]
        in_use = sorted(set(have["verbs"].values()))
        text = PROMPT + (f"Names already in use: {', '.join(in_use)}\n\n" if in_use else "") + "\n".join(part)
        try:
            resp = call_model([{"type": "text", "text": text}], MODEL, REASONING, keys[0], max_tokens=MAX_TOKENS,
                              timeout=timeout)
            msg = resp["choices"][0]["message"]["content"]
            ans = json.loads(msg[msg.find("{"):msg.rfind("}") + 1]).get("verbs") or {}
            spent += _cost(resp.get("usage") or {})
        except Exception as e:      # verbs left unmerged are asked again next time
            print(f"verbs: a call failed ({str(e)[:160]}); its verbs stay unmerged")
            continue
        for v in part:
            name = verb_of(ans.get(v))
            if name and len(name.split()) <= 2:
                have["verbs"][v] = name
                merged += 1
    have["cost_usd"] = round(have["cost_usd"] + spent, 6)
    have["model"] = MODEL
    tmp = path.with_name(".verbs.json.tmp")
    tmp.write_text(json.dumps(have, indent=1, sort_keys=True))
    os.replace(tmp, path)
    return merged, spent


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m board verbs", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("board", type=Path, help="the board folder")
    a = ap.parse_args()
    n, usd = merge(a.board)
    print(f"merged {n} verbs for ${usd:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
