"""One name for each skill and each action the labels give, for the home page's counts and diversity chart.

    python -m board skills BOARD             names every skill and action not named yet (board/follow.py runs it)

The labeler gives each task the skill it exercises in one to three words, and the other actions inside it (each a word
or two), in label/prompts.py: an episode's task_skill and task_actions, a session task's skill and actions. Freeform
tasks are named in the labeler's own words, so one action can come back as "pick" and "pick up", or "flip" and "turn
over". A small model is given the list of distinct names, never the tasks, and gives each the name it is counted
under, choosing among names already in use where one fits. Only different words for one action are merged, never a
specific action into a general one (laying an object flat is not placing it), so the counts keep the detail the
labeler gave. A name once merged is never asked again, so the counts stay fixed as labels land; the answers are kept
in BOARD/skills.json, and naming every skill and action of a 100 hour run costs about a cent. A name the model leaves
out, or a failed call, counts under its own words until the next run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

MODEL = "openai/gpt-6-sol"
REASONING = "low"
MAX_TOKENS = 16000
BATCH = 300                 # names per call
PROMPT = (
    "Each line below is a name a labeler gave to an action in tabletop and household manipulation footage, either "
    "the skill a task exercises or an action inside a task. Several lines can name the same action in different words "
    "('pick' and 'pick up', 'stand' and 'stand up', 'turn over' and 'flip'). Give each line the one name, in one to "
    "three words, that its action should be counted under, so that lines naming the same action get the same name and "
    "lines naming different actions keep different names. Merge only different words for one action; never fold a "
    "more specific action into a more general or a merely similar one (laying an object flat is not placing it, "
    "tipping it over is not leaning it), so a line with no other words for its action keeps its own. Choose among the "
    "lines' own words, and use a name already in use (listed first) whenever it names the same action. Reply with "
    'JSON only: {"names": {"<line>": "<name>"}}, every line spelled exactly as given.\n\n')
# what a labeler writes when a task had no action
NO_ACTION = {"none", "null", "n/a", "na", "no action", "no op", "no-op", "noop", "nothing", "idle"}


def name_of(raw) -> str | None:
    """A skill or action as the labeler wrote it, lower case and single spaced; None when it wrote none, or wrote that
    no action happened (an episode where the operator does nothing has no skill to count)."""
    v = " ".join(str(raw).lower().split()) if isinstance(raw, str) else ""
    return v if v and v not in NO_ACTION else None


def task_fields(d: dict) -> list:
    """[(skill, [actions])] for each subtask of one board episode file: a session's tasks, or the episode itself
    where it is one task. A skill is None where the label has the field but names no action, and the field itself is
    missing (not None) in a label written before the labeler gave skills."""
    tasks = [t for t in d.get("tasks") or [] if isinstance(t, dict)]
    rows = [(t, "skill", "actions") for t in tasks] if tasks else [(d, "task_skill", "task_actions")]
    return [(r.get(sk) if sk in r else ..., [a for a in r.get(ac) or [] if isinstance(a, str)]) for r, sk, ac in rows]


def names_seen(qa: Path) -> set:
    """Every skill and action named in BOARD/qa."""
    out = set()
    for p in qa.glob("*.json"):
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(d, dict):
            continue
        for skill, actions in task_fields(d):
            out.update(n for n in (name_of(x) for x in [skill, *actions]) if n)
    return out


def load(path: Path) -> dict:
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"names": {}, "cost_usd": 0.0}
    return {"names": d.get("names") or {}, "cost_usd": float(d.get("cost_usd") or 0.0)}


def load_names(path: Path) -> dict:
    """{skill or action as written: the name it is counted under} from BOARD/skills.json."""
    return load(path)["names"]


def merge(board: Path, timeout: int = 180) -> tuple[int, float]:
    """Gives a name to each skill and action in BOARD/qa that BOARD/skills.json lacks. Returns (names given, dollars
    spent)."""
    from label.harness import _cost, call_model, get_keys
    path = board / "skills.json"
    have = load(path)
    todo = sorted(names_seen(board / "qa") - set(have["names"]))
    keys = get_keys()
    if not todo or not keys:
        return 0, 0.0
    named, spent = 0, 0.0
    for i in range(0, len(todo), BATCH):
        part = todo[i:i + BATCH]
        in_use = sorted(set(have["names"].values()))
        text = PROMPT + (f"Names already in use: {', '.join(in_use)}\n\n" if in_use else "") + "\n".join(part)
        try:
            resp = call_model([{"type": "text", "text": text}], MODEL, REASONING, keys[0], max_tokens=MAX_TOKENS,
                              timeout=timeout)
            msg = resp["choices"][0]["message"]["content"]
            ans = json.loads(msg[msg.find("{"):msg.rfind("}") + 1]).get("names") or {}
            spent += _cost(resp.get("usage") or {})
        except Exception as e:      # names left out are asked again next time
            print(f"skills: a call failed ({str(e)[:160]}); its names stay unmerged")
            continue
        for v in part:
            n = name_of(ans.get(v))
            if n and len(n.split()) <= 3:
                have["names"][v] = n
                named += 1
    have["cost_usd"] = round(have["cost_usd"] + spent, 6)
    have["model"] = MODEL
    tmp = path.with_name(".skills.json.tmp")
    tmp.write_text(json.dumps(have, indent=1, sort_keys=True))
    os.replace(tmp, path)
    return named, spent


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m board skills", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("board", type=Path, help="the board folder")
    a = ap.parse_args()
    n, usd = merge(a.board)
    print(f"named {n} skills and actions for ${usd:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
