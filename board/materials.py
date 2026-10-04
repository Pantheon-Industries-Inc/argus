"""Rigid or deformable: one tag for each kind of object a board's labels name, for the home page.

    python -m board materials BOARD          tags every kind not tagged yet (board/follow.py runs it as labels land)

The labels name objects in their own words, and board/home.py reduces each name to its kind (the head noun: "clear
test tubes" is a tube). A small model is asked once per kind, never per episode, whether that kind of object is
deformable (it bends, folds, drapes or squashes when handled: cloth, bags, sponges, dough, cables) or rigid. The kinds
go in batches with a few of the names each was given and no frames, so tagging every kind of a 100 hour run costs
about a cent. The answers are kept in BOARD/materials.json and a kind once tagged is never asked again; a kind the
model leaves out, or a failed call, stays untagged and is asked on the next run.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from board.home import object_kind

MODEL = "openai/gpt-6-sol"
REASONING = "low"
MAX_TOKENS = 16000
BATCH = 300                 # kinds per call
EXAMPLES = 3                # names sent with each kind, for context
PROMPT = (
    "Each line below is a kind of object seen in tabletop and household manipulation footage, with some of the names "
    "it was given. Say whether that kind of object is deformable, meaning it bends, folds, drapes, stretches or "
    "squashes in normal handling (cloth, towels, clothes, bags, sponges, dough, cables, paper, plush toys), or rigid, "
    "meaning it keeps its shape (cups, blocks, tools, bottles, boxes, devices). Reply with JSON only: "
    '{"kinds": {"<kind>": true if deformable, false if rigid}}, every kind spelled exactly as given.\n\n')


def kinds_seen(qa: Path) -> dict:
    """{kind: up to EXAMPLES of the names it was given} over every episode file in qa."""
    out = defaultdict(list)
    for p in qa.glob("*.json"):
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(d, dict):
            continue
        named = [o.get("name") if isinstance(o, dict) else None for o in d.get("objects") or []]
        named += [o.get("name") if isinstance(o, dict) else o for t in d.get("tasks") or [] if isinstance(t, dict)
                  for o in t.get("objects") or []]
        named += [e.get("object") for e in d.get("event_labels") or [] if isinstance(e, dict)]
        for raw in named:
            name = str(raw or "").strip().lower()
            k = object_kind(name)
            if k and name and len(out[k]) < EXAMPLES and name not in out[k]:
                out[k].append(name)
    return dict(out)


def load(path: Path) -> dict:
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"kinds": {}, "cost_usd": 0.0}
    return {"kinds": d.get("kinds") or {}, "cost_usd": float(d.get("cost_usd") or 0.0)}


def tag(board: Path, timeout: int = 180) -> tuple[int, float]:
    """Tags the kinds in BOARD/qa that BOARD/materials.json lacks. Returns (kinds tagged, dollars spent)."""
    from label.harness import _cost, call_model, get_keys
    path = board / "materials.json"
    have = load(path)
    todo = {k: v for k, v in kinds_seen(board / "qa").items() if k not in have["kinds"]}
    keys = get_keys()
    if not todo or not keys:
        return 0, 0.0
    tagged, spent, items = 0, 0.0, sorted(todo.items())
    for i in range(0, len(items), BATCH):
        part = items[i:i + BATCH]
        text = PROMPT + "\n".join(f"{k}: {', '.join(names)}" for k, names in part)
        try:
            resp = call_model([{"type": "text", "text": text}], MODEL, REASONING, keys[0], max_tokens=MAX_TOKENS,
                              timeout=timeout)
            msg = resp["choices"][0]["message"]["content"]
            ans = json.loads(msg[msg.find("{"):msg.rfind("}") + 1]).get("kinds") or {}
            spent += _cost(resp.get("usage") or {})
        except Exception as e:      # untagged kinds are asked again next time
            print(f"materials: a call failed ({str(e)[:160]}); its kinds stay untagged")
            continue
        for k, _ in part:
            if isinstance(ans.get(k), bool):
                have["kinds"][k] = {"deformable": ans[k], "examples": todo[k]}
                tagged += 1
    have["cost_usd"] = round(have["cost_usd"] + spent, 6)
    have["model"] = MODEL
    tmp = path.with_name(".materials.json.tmp")
    tmp.write_text(json.dumps(have, indent=1, sort_keys=True))
    os.replace(tmp, path)
    return tagged, spent


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m board materials", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("board", type=Path, help="the board folder")
    a = ap.parse_args()
    n, usd = tag(a.board)
    print(f"tagged {n} kinds for ${usd:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
