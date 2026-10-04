"""The main verb of each task: one verb for what the hands do to the object, for the home page.

    python -m board verbs BOARD              names the verb of every task sentence not named yet (board/follow.py runs it)

A task's first word is often a word that fits any relocation ("Move the bracelet beside the book", "Pick up the block,
pass it to the left gripper, and place it beside the cup"), and the events' own verbs (approach, grasp, lower,
release) are the steps of nearly every task. Which verb a sentence is about is a reading of the whole sentence, so a
small model is asked once per distinct task sentence, never per episode, with no frames: the block above is handed
over, "Use the sponge to push the orange" is push, and a task that only takes an object from one place to another is
move. Sentences go in batches; the answers are kept in BOARD/verbs.json and a sentence once named is never asked again.
A sentence the model leaves out, or a failed call, stays unnamed and is asked on the next run; until then the home page
counts its first verb (board/home.py task_kind).
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

MODEL = "openai/gpt-6-sol"
REASONING = "low"
MAX_TOKENS = 32000
BATCH = 300                 # sentences per call
PROMPT = (
    "Each numbered line below is one task a person did with two hand-held grippers at a table, as its label wrote "
    "it. Give the verb that names most specifically what the hands do to the object in that task, in its base form "
    'and lower case, one or two words (for example "hand over", "stand", "flip", "fold", "stack", "insert", "push", '
    '"slide", "squeeze", "pour", "wipe", "open", "lay"). Choose the action the task is about, not its first step '
    'or how it ends: in "Pick up the block, pass it to the left gripper, and place it beside the cup" the block is '
    'handed over, in "Use the sponge to push the orange left" the hands push, and in "Turn the card face down and '
    'set it on the table" they flip it. When the task only takes an object from one place to another, with nothing '
    'more said about how, the verb is "move". Reply with JSON only: {"verbs": {"<line number>": "<verb>"}}.\n\n')


def task_sentences(qa: Path) -> set:
    """Every task sentence in BOARD/qa: a session's tasks, or the episode's prompt where an episode is one task."""
    out = set()
    for p in qa.glob("*.json"):
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(d, dict):
            continue
        tasks = [t for t in d.get("tasks") or [] if isinstance(t, dict) and t.get("task")]
        for s in ([t["task"] for t in tasks] if tasks else [d.get("episode_prompt")]):
            s = " ".join(str(s or "").split())
            if s:
                out.add(s)
    return out


def load(path: Path) -> dict:
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"verbs": {}, "cost_usd": 0.0}
    return {"verbs": d.get("verbs") or {}, "cost_usd": float(d.get("cost_usd") or 0.0)}


def load_verbs(path: Path) -> dict:
    """{task sentence: verb} from BOARD/verbs.json."""
    return load(path)["verbs"]


def _clean(v) -> str | None:
    words = re.findall(r"[a-z][a-z'-]*", str(v or "").lower())
    return " ".join(words) if 1 <= len(words) <= 2 else None


def name(board: Path, timeout: int = 300, limit: int | None = None) -> tuple[int, float]:
    """Names the verb of the task sentences in BOARD/qa that BOARD/verbs.json lacks, at most LIMIT of them.
    Returns (sentences named, dollars spent)."""
    from label.harness import _cost, call_model, get_keys
    path = board / "verbs.json"
    have = load(path)
    todo = sorted(s for s in task_sentences(board / "qa") if s not in have["verbs"])[:limit]
    keys = get_keys()
    if not todo or not keys:
        return 0, 0.0
    named, spent = 0, 0.0
    for i in range(0, len(todo), BATCH):
        part = todo[i:i + BATCH]
        text = PROMPT + "\n".join(f"{j + 1}. {s}" for j, s in enumerate(part))
        try:
            resp = call_model([{"type": "text", "text": text}], MODEL, REASONING, keys[0], max_tokens=MAX_TOKENS,
                              timeout=timeout)
            msg = resp["choices"][0]["message"]["content"]
            ans = json.loads(msg[msg.find("{"):msg.rfind("}") + 1]).get("verbs") or {}
            usd = _cost(resp.get("usage") or {})
        except Exception as e:      # unnamed sentences are asked again next time
            print(f"verbs: a call failed ({str(e)[:160]}); its sentences stay unnamed")
            continue
        for j, s in enumerate(part):
            v = _clean(ans.get(str(j + 1)))
            if v:
                have["verbs"][s] = v
                named += 1
        # saved after every batch, so a stopped run keeps what it paid for
        spent += usd
        have["cost_usd"] = round(have["cost_usd"] + usd, 6)
        have["model"] = MODEL
        tmp = path.with_name(".verbs.json.tmp")
        tmp.write_text(json.dumps(have, indent=1, sort_keys=True))
        os.replace(tmp, path)
    return named, spent


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m board verbs", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("board", type=Path, help="the board folder")
    ap.add_argument("--limit", type=int, help="name at most this many sentences (to measure the cost first)")
    a = ap.parse_args()
    n, usd = name(a.board, limit=a.limit)
    print(f"named the verb of {n} task sentences for ${usd:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
