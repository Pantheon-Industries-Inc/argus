"""A task catalog of labelled recordings: which tasks a dataset holds and how much of each, from Argus labels alone.

Every labelled recording carries labels.tasks, its units of work in time order, each with a sentence, the objects it
acts on, an outcome and its start and end. A freeform recording holds many of them, so the catalog counts tasks, never
recordings. It sorts every task into three levels:

- a task family, a broad kind of activity ("building and assembly", "pouring and scooping");
- a coarse task, one of the few hundred tasks a reader browses ("fold towel", "assemble jigsaw puzzle", or a generic one
  such as "place object" for a task that only moves one thing);
- a specific task, one verb on one specific kind of object ("pick and place rubber duck", "slide playing card"), so a
  rubber duck and a stuffed animal are different specific tasks, and so are sliding a card and picking it up.

Each row gives its hours (the summed length of its tasks, labelled time rather than footage), its success hours (the
tasks whose outcome is success), how many tasks it holds and in how many recordings, and three example spans.

How a task gets its names. A model names every distinct task sentence twice, CALL_SIZE sentences a call: once with a
coarse task, once with a verb and the kind of object. Every sentence is named on its own; naming a whole group of
similar sentences gives one action to tasks that differ in action, since sentences that look alike mostly share their
objects. Calls hold sentences in sorted order, and the specific calls in the order of their coarse names, so similar
tasks share a call and get the same names. One call then writes the coarse task list from the names holding the most
hours (LIST_NAMES), and every name is mapped onto it, LIST_ASSIGN names a call, with new coarse tasks only for names that
fit none. A failed call's names are asked again in smaller calls, never counted as having no task.

How specific tasks merge. Only the same verb on the same kind of object said in other words merges. Verbs merge where
they name one manipulation ("put", "set" and "lay" into "pick and place") and stay apart where the motion or the rule
differs ("slide" from "pick and place", "sort by color" from "sort by shape"). Each object name is offered the more used
names that share a word with it, the model proposes a match, and a second call accepts it only after reading two
example tasks of each name. merge_allowed refuses any merge where one name has words the other lacks ("tube" and
"toothpaste tube", "card" and "playing card"), whatever the model says, since the general name may be another kind, and
reorder_merges joins names with the same words in any order or number ("duck toy" and "toy duck", "screw" and "screws")
without a call.

A later catalog. --base OUT of an earlier catalog keeps its vocabulary (its coarse tasks, the names mapped onto them, its
verb and object merges) and its cache, so the new catalog names only sentences it has not seen, places them on the
existing coarse tasks, and merges new names into the existing ones. On a base, only a name holding at least
ADD_MIN_HOURS may found a new coarse task; the rest are listed under "(no coarse task)", where they keep their specific
tasks. A catalog grown this way keeps its task names from one version to the next.

The checks. --check N judges a fixed sample (seed SEED) of N tasks against their coarse and specific task, and N random
merges as the same task or not, each a small call to --judge-model.

    python -m label.catalog --label RESULTS --out OUT [--base EARLIER_OUT] [--cap 50] [--check 100] [--dry-run]

RESULTS is a stitched result, a folder of them, or a JSON Lines file of results (a line may also hold the result under
"label", with the recording's name under "take"). OUT gets catalog.md (one row per coarse task), catalog_specific.csv
(one row per specific task under its coarse task), catalog.json (both levels nested), check.json, vocab.json (what a
later --base reads) and cache.jsonl (every answer by the fingerprint of what was asked, so a rerun never pays twice).
The input is never touched.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VERSION = 1
PROMPT_VERSION = 1
SEED = 20261031
CALL_SIZE = 100
LIST_NAMES = 6000
LIST_ASSIGN = 400
ADD_MIN_HOURS = 0.5
REASONING = "low"
TIMEOUT_S = 900
NO_COARSE = "(no coarse task)"
SETUPS = {"handheld_gripper": ("an operator holding a handheld gripper in each hand", "grippers"),
          "teleop_arms": ("an operator teleoperating robot arms", "arms"),
          "ego_head": ("a person wearing a head camera and working with their own hands", "hands")}


def norm(s) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


# ---------------------------------------------------------------- reading results

def tasks_of(result: dict, name: str) -> dict:
    """The tasks of one stitched result, each with the distinct non-idle timeline actions inside its span, which the
    naming calls read as context and the catalog never counts."""
    labels = result.get("labels") if isinstance(result.get("labels"), dict) else {}
    steps = [s for s in labels.get("timeline") or [] if isinstance(s, dict)
             and isinstance(s.get("start_s"), (int, float)) and isinstance(s.get("end_s"), (int, float))]
    out = []
    for i, t in enumerate(labels.get("tasks") or []):
        if not isinstance(t, dict): continue
        a, b = t.get("start_s"), t.get("end_s")
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or b <= a: continue
        acts = list(dict.fromkeys(norm(s.get("action")) for s in steps
                                  if s["start_s"] >= a - 0.5 and s["end_s"] <= b + 0.5 and s.get("contribution") != "idle"))
        out.append({"i": i, "a": round(float(a), 3), "b": round(float(b), 3), "task": str(t.get("task") or "").strip(),
                    "objects": [str(o) for o in (t.get("objects") or [])][:8], "outcome": t.get("outcome"),
                    "acts": [x for x in acts if x][:8]})
    rig = (result.get("annotation_policy") or {}).get("rig")
    return {"name": name, "rig": rig if rig in SETUPS else None, "tasks": out}


def read_results(path: Path) -> list[dict]:
    """Every recording under path: a result file, a folder of them (failed_ and noreply_ files skipped), or JSON
    Lines of results or of {"take", "label": result}. A recording read twice keeps its first result."""
    path = Path(path); recs, seen = [], set()
    def add(result, name):
        if isinstance(result, dict) and name not in seen:
            seen.add(name); recs.append(tasks_of(result, name))
    if path.is_dir():
        for p in sorted(path.glob("*.json")):
            if not p.name.startswith(("failed_", "noreply_")): add(json.loads(p.read_text()), p.stem)
    elif path.suffix == ".jsonl":
        for n, line in enumerate(path.read_text().splitlines()):
            if not line.strip(): continue
            row = json.loads(line)
            if isinstance(row.get("label"), dict): add(row["label"], str(row.get("take") or row.get("name") or n))
            else: add(row, str(row.get("take") or row.get("name") or n))
    else:
        add(json.loads(path.read_text()), path.stem)
    return recs


def text_key(task: dict) -> str:
    """What a task is named from: its sentence and its objects."""
    return norm(task["task"]) + " | " + ", ".join(sorted(norm(o) for o in task["objects"][:6]))


def batches(ids: list, size: int) -> list[list]:
    """Calls of at most size items, each item in exactly one call."""
    ids = list(dict.fromkeys(ids))
    return [ids[i:i + size] for i in range(0, len(ids), size)]


# ---------------------------------------------------------------- replies

def parse_rows(text: str) -> list:
    """The rows of a reply: the whole reply as JSON when it parses, else every flat JSON object, or when there are none
    every flat JSON array, parsed on its own, so one malformed row loses only itself."""
    text = text or ""
    starts = [x for x in (text.find("["), text.find("{")) if x >= 0]
    if starts:
        try:
            value = json.loads(text[min(starts):max(text.rfind("]"), text.rfind("}")) + 1])
            if isinstance(value, list) and value and all(isinstance(x, (dict, list)) for x in value): return value
            return [value]
        except ValueError:
            pass
    def each(pattern):
        out = []
        for m in re.finditer(pattern, text):
            try: out.append(json.loads(m.group(0)))
            except ValueError: pass
        return out
    return each(r"\{[^{}]*\}") or each(r"\[[^\[\]{}]*\]")


# ---------------------------------------------------------------- merges

PLURAL_KEEPS = {"glasses", "pliers", "scissors", "tongs", "pants", "shorts", "jeans", "tweezers", "clippers", "goggles"}


def singular(w: str) -> str:
    """A word's singular for comparing names ("screws" and "screw"); a word whose plural is its own thing stays."""
    if w in PLURAL_KEEPS or len(w) <= 3 or w.endswith("ss"): return w
    if w.endswith("ies"): return w[:-3] + "y"
    if w.endswith(("ches", "shes", "xes", "sses", "zes")): return w[:-2]
    return w[:-1] if w.endswith("s") else w


def words(name: str) -> set:
    return {singular(w) for w in norm(name).split()}


def merge_allowed(a: str, b: str) -> bool:
    """Whether a may merge into b at all: never when one has words the other lacks, a qualifier ("sort by color" and
    "sort") or a more general name that may be another kind ("tube" and "toothpaste tube")."""
    wa, wb = words(a), words(b)
    return bool(wa) and bool(wb) and a != b and not (wa < wb or wb < wa)


def resolve(m: dict, x):
    seen = set()
    while x in m and x not in seen and m[x] != x:
        seen.add(x); x = m[x]
    return x


def reorder_merges(counts: dict) -> dict:
    """{name: the most used name with the same words in any order or number}, merged without a call."""
    groups = collections.defaultdict(list)
    for n in counts: groups[tuple(sorted(words(n)))].append(n)
    out = {}
    for g in groups.values():
        if len(g) > 1:
            top = max(g, key=lambda n: (counts[n], -len(n), n))      # on a tie the shorter name ("towel", not "towels")
            out.update({n: top for n in g if n != top})
    return out


def candidates(names: list, k: int = 8) -> dict:
    """{index: indices of up to k more used names sharing a word with it}, nearest by shared words and letters first.
    names is in order of use, most used first."""
    by_word = collections.defaultdict(list)
    for i, n in enumerate(names):
        for w in words(n): by_word[w].append(i)
    def grams(s): s = f" {norm(s)} "; return {s[i:i + 3] for i in range(len(s) - 2)}
    g = [grams(n) for n in names]; out = {}
    for i, n in enumerate(names):
        near = {j for w in words(n) for j in by_word[w] if j < i}
        if near:
            out[i] = sorted(near, key=lambda j: (-len(g[i] & g[j]) / len(g[i] | g[j]), j))[:k]
    return out


# ---------------------------------------------------------------- mapping onto the coarse list

def assign_round(todo: list, blocks: list, replies: list, task_ids: set) -> tuple[dict, set, list]:
    """One round of mapping names onto coarse task ids: (assigned, names answered null, names to ask again). A failed
    call (reply None) or a name a reply left out is asked again, never counted as having no task."""
    assigned, no_task, answered = {}, set(), set()
    for b, rows in zip(blocks, replies):
        if rows is None: continue
        for r in rows:
            if not (isinstance(r, list) and len(r) == 2 and isinstance(r[0], int) and r[0] in b): continue
            name = todo[r[0]]; answered.add(name)
            if r[1] in task_ids: assigned[name] = r[1]
            elif r[1] is None: no_task.add(name)
    return assigned, no_task, [n for n in todo if n not in answered]


def add_candidates(left: list, hours: dict, base: bool, min_hours: float = ADD_MIN_HOURS, cap: int = 3000) -> list:
    """The names with no coarse task that may found one, most hours first; on a base only those with min_hours, so a
    few small names never grow an established list."""
    return sorted((n for n in left if not base or hours.get(n, 0) >= min_hours), key=lambda n: -hours.get(n, 0))[:cap]


# ---------------------------------------------------------------- the levels

def levels(tasks: list, seed: int = SEED) -> list:
    """The coarse tasks with their hours, success hours, tasks, recordings and examples, and under each its specific
    tasks; a task with no coarse task sits under None. Each task is {name, a, b, outcome, coarse, family, verb, obj}."""
    rng = random.Random(seed)
    def blank(): return {"s": 0.0, "ok": 0.0, "n": 0, "recs": set(), "ex": []}
    coarse, fine, family = collections.defaultdict(blank), collections.defaultdict(blank), {}
    for t in tasks:
        dur = t["b"] - t["a"]; ok = dur if t.get("outcome") == "success" else 0.0
        groups = [coarse[t.get("coarse")]] + ([fine[(t.get("coarse"), t["verb"], t["obj"])]] if t.get("verb") and t.get("obj") else [])
        for g in groups:
            g["s"] += dur; g["ok"] += ok; g["n"] += 1; g["recs"].add(t["name"]); g["ex"].append(f"{t['name']}@{t['a']:.1f}-{t['b']:.1f}s")
        family[t.get("coarse")] = t.get("family")
    def row(g):
        ex = sorted(g["ex"])
        return {"hours": round(g["s"] / 3600, 3), "success_hours": round(g["ok"] / 3600, 3), "tasks": g["n"],
                "recordings": len(g["recs"]), "examples": rng.sample(ex, 3) if len(ex) > 3 else ex}
    kids = collections.defaultdict(list)
    for (c, v, o), g in fine.items(): kids[c].append((v, o, g))
    out = []
    for c, g in sorted(coarse.items(), key=lambda kv: -kv[1]["s"]):
        out.append({"task": c, "family": family.get(c), **row(g),
                    "specific": [{"task": f"{v} {o}", "verb": v, "object": o, **row(fg)}
                                 for v, o, fg in sorted(kids[c], key=lambda x: -x[2]["s"])]})
    return out


def levels_id(out: list) -> str:
    """A hash of what the checks judge, so a check of other levels is never reused."""
    rows = sorted((str(t["task"]), f["verb"], f["object"], f["tasks"]) for t in out for f in t["specific"])
    return hashlib.sha1(json.dumps(rows).encode()).hexdigest()[:12]


# ---------------------------------------------------------------- the prompts

def setup_of(recs: list) -> tuple[str, str]:
    """(who does the work, what passes an object) for the prompts, from the recordings' rigs."""
    rigs = {r["rig"] for r in recs if r["rig"]}
    if len(rigs) == 1: return SETUPS[rigs.pop()]
    return ("a demonstrator, with robot arms, handheld grippers or their own hands", "hands or grippers")


def prompts(setup: tuple[str, str]) -> dict:
    who, between = setup
    ctx = (f"You are building a task catalog over recorded demonstrations: {who} does the work, and an automatic "
           "labeller wrote one sentence per task, with its objects and the actions seen inside it.")
    return {
        "coarse": ctx + f""" For each task give its coarse task, a short lowercase phrase "<verb> <object>[ <short qualifier>]", 2 to 5 words
("fold towel", "stack cups", "write on paper", "pour beans into bowl"). Drop colours, sizes, counts, positions, brand names
and which side. An undoing action is its own task ("unfold towel", "untie knot"). A task that only moves, places, turns over
or hands over one object is named for exactly that ("place cup", "turn over remote", "pass object between {between}").
Answer with a JSON array only, one object per task in the same order: [{{"id": <id>, "task": "..."}}]

Tasks:
""",
        "specific": ctx + f""" For each task give its verb and its object:
- "verb": the manipulation, a short lowercase phrase: "pick and place" (picking an object up and putting it down anywhere,
  also what "put", "set", "lay" and "return" mean), "slide", "push", "roll", "turn over", "stand upright", "pass between
  {between}", "toss", "stack", "unstack", "insert", "remove", "pour", "scoop", "fold", "unfold", "tie", "untie",
  "assemble", "take apart", "sort by color", "wipe", and so on, the verb of the task's main manipulation; an undoing
  action is its own verb.
- "object": the specific kind of object, singular, lowercase ("rubber duck", "stuffed giraffe", "playing card", "towel");
  keep the kind, drop colours, sizes, counts and positions; "mixed objects" for a group of different things, and the
  material for loose material ("rice", "dough").
Answer with one JSON array per task, in the same order and nothing else: [<id>, "<verb>", "<object>"]

Tasks:
""",
        "list": """Below is every coarse task name of a task catalog with its hours, written task by task, so it holds synonyms and
near-duplicates. Write the final list a reader would browse: one task per distinct manipulation (the same action on the
same kind of object is one task); an undoing action is its own task; a name too vague to filter by belongs to a family's
"other" task; every task belongs to one family, a broad kind of activity that fits both its action and its object; name
tasks lowercase "<verb> <object>[ <qualifier>]"; rarer names are mapped onto this list afterwards, so never fold a name
into a task it is not.
Answer with a JSON array only: [{"id": <int from 1>, "family": "...", "task": "...", "definition": "<at most 15 words>"}]

Names (name<TAB>hours):
""",
        "assign": """Below is the task list of a task catalog, then numbered task names. Map each name to the id of the task it is (the
same action on the same kind of object, synonyms counting as the same; an undoing action is a different task; a vague
name goes to its family's "other" task). Answer null when no task is that task; never force a name into a task it is not.
Answer with a JSON array only, one pair per name: [[<name number>, <task id or null>], ...]

Task list (id<TAB>family<TAB>task<TAB>definition):
{tasks}

Names:
{names}""",
        "add": """Below is the task list of a task catalog, then names that fit none of its tasks. Add the fewest new tasks that cover
them, each with a family (an existing one where it fits); a name is covered only by a task with its action and its kind
of object. Answer with a JSON array of the new tasks only:
[{{"id": <int, continuing after the last id>, "family": "...", "task": "...", "definition": "<at most 15 words>"}}]

Task list (id<TAB>family<TAB>task<TAB>definition):
{tasks}

Names with no task (name<TAB>hours):
{names}""",
        "verbs": """Below are verbs of a task catalog with their counts. Merge a verb into another only when both name the same
manipulation in other words ("put down" and "set" into "pick and place"). Keep verbs apart when the motion differs
("slide" vs "pick and place", "pour" vs "scoop", "stack" vs "pick and place"), keep an undoing verb apart from what it
undoes, and keep a qualifier that changes the task ("sort by color" and "sort by shape" are not "sort").
Answer with one JSON array per verb that changes, and nothing else: ["<verb>", "<verb it merges into>"]

Verbs (verb<TAB>count):
""",
        "propose": """Below are object names of a task catalog, each numbered with its count and an example task, then candidate names used
more often. For each, pick the candidate that is exactly the same kind of object in other words ("toy duck" and "rubber
duck"), or null. Never a different kind, even a close one (a rubber duck and a stuffed animal, a marble and a bead), and
never a more general or more specific name ("card" and "playing card"). When in doubt, null.
Answer with one JSON array per name, and nothing else: [<name number>, <candidate number or null>]

Names:
""",
        "verify": """Below are proposed merges of two object names, each with two example tasks that use it. Accept a merge only when both
names, as the examples use them, are exactly the same kind of object in other words. Reject a part, group, container or
variant of the other, a more general or more specific name, or a close but different kind ("jar" and "bottle"). When in
doubt, reject.
Answer with one JSON array per merge, and nothing else: [<merge number>, true | false]

Merges:
""",
        "judge": """A task catalog sorts each labelled task into a coarse task and a specific task (one verb on one specific kind of
object). A task that only moves one object belongs to a generic coarse task such as "place object". Judge both for this task.

Task: {text}
Objects: {objects}
Coarse task: "{coarse}"
Specific task: verb "{verb}", object "{obj}"

Answer with one JSON object only: {{"coarse_correct": true | false, "specific_correct": true | false, "reason": "<one sentence>"}}""",
        "judge_merge": """A task catalog merged two names into one specific task, which should happen only when the verb and the kind of
object are the same, in other words. Is this merge right?

A: verb "{va}", object "{oa}", e.g. "{ta}"
B: verb "{vb}", object "{ob}", e.g. "{tb}"

Answer with one JSON object only: {{"same_task": true | false, "reason": "<one sentence>"}}"""}


# ---------------------------------------------------------------- the model

class Cache:
    """The answers by fingerprint, appended to a JSON Lines file as they arrive, so a catalog of thousands of calls
    writes each answer once (label/issue_groups.py Cache rewrites its whole file per answer, which suits its few)."""

    def __init__(self, path=None):
        self.path, self.lock, self.answers = (Path(path) if path else None), threading.Lock(), {}
        if self.path and self.path.is_file():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line); self.answers[row["fp"]] = row["answer"]

    def get(self, fp: str):
        with self.lock: return self.answers.get(fp)

    def put(self, fp: str, answer: dict) -> None:
        with self.lock:
            self.answers[fp] = answer
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a") as f: f.write(json.dumps({"fp": fp, "answer": answer}, ensure_ascii=False) + "\n")


class Asker:
    """Calls through the labelling client under one cap (label/issue_groups.py Budget), each answer kept in the cache by
    the fingerprint of what was asked."""

    def __init__(self, model: str, reasoning: str, cache, budget, keys: list, dry_run: bool = False):
        self.model, self.reasoning, self.cache, self.budget, self.keys, self.dry = model, reasoning, cache, budget, keys, dry_run
        self.n, self.lock, self.problems, self.examples = 0, threading.Lock(), collections.Counter(), []

    def note(self, problem: str, detail: str = "") -> None:
        """Count a problem, and keep the first few with their detail for the summary."""
        with self.lock:
            self.problems[problem] += 1
            if len(self.examples) < 5 and detail: self.examples.append(f"{problem}: {detail}"[:400])

    def __call__(self, text: str, answer_tokens: int, model: str | None = None):
        """The reply's rows, or None when there is no usable reply."""
        from label import harness, policy
        from label.errors import DispatchError
        model = model or self.model
        max_tokens = policy.output_limit(answer_tokens, self.reasoning)
        fp = hashlib.sha1(json.dumps([PROMPT_VERSION, model, self.reasoning, text]).encode()).hexdigest()
        saved = self.cache.get(fp)
        if saved is None:
            content = [{"type": "text", "text": text}]
            bound = harness.final_cost_bound(content, max_tokens, model=model, require_known_pricing=False)
            if self.dry: self.problems["dry run"] += 1; return None
            if not self.keys: self.problems["no model key"] += 1; return None
            if not self.budget.reserve(bound): self.problems["cap reached"] += 1; return None
            with self.lock: first = self.n; self.n += 1
            resp = None
            for k in range(len(self.keys)):          # a key refused unpaid (out of credit, say) hands the call to the next
                try:
                    resp = harness.call_model_once(content, model, self.reasoning, self.keys[(first + k) % len(self.keys)],
                                                   max_tokens, TIMEOUT_S)
                    break
                except DispatchError as error:
                    if getattr(error, "verified_unpaid", False) and k + 1 < len(self.keys): continue
                    unpaid = bool(getattr(error, "verified_unpaid", False))
                    self.budget.settle(bound, 0.0 if unpaid else bound); self.note("call failed", str(error))
                    return None
            cost, _ = harness._verified_final_cost(resp.get("usage") or {}, model)
            self.budget.settle(bound, bound if cost is None else cost)
            choice = (resp.get("choices") or [{}])[0]
            saved = {"text": (choice.get("message") or {}).get("content") or "", "finish_reason": choice.get("finish_reason"),
                     "cost_usd": bound if cost is None else cost}
            self.cache.put(fp, saved)
        rows = parse_rows(saved.get("text") or "")
        if not rows: self.note("unreadable reply", f"finish {saved.get('finish_reason')}: {(saved.get('text') or '')[:200]!r}")
        return rows or None

    def many(self, texts: list, answer_tokens: int, concurrency: int, model: str | None = None) -> list:
        with ThreadPoolExecutor(max(1, concurrency)) as pool:
            return list(pool.map(lambda t: self(t, answer_tokens, model), texts))


# ---------------------------------------------------------------- the run

def build(recs: list, ask: Asker, *, base: dict | None = None, concurrency: int = 8, check: int = 0,
          judge_model: str | None = None) -> dict:
    """The catalog of recs: {summary, levels, vocab, check}."""
    P = prompts(setup_of(recs))
    tasks = [{"name": r["name"], **t} for r in recs for t in r["tasks"]]
    texts = {}
    for t in tasks: t["key"] = text_key(t); texts.setdefault(t["key"], t)
    vocab = base or {}

    def name_all(kind, prompt, row_of, order, tokens):
        named = dict(vocab.get(f"{kind}_names_by_text") or {})
        for size in (CALL_SIZE, 20):
            todo = [k for k in order if k not in named]
            if not todo: break
            ids = {k: i for i, k in enumerate(todo)}
            calls = batches(todo, size)
            rows = ask.many([prompt + "\n".join(json.dumps({"id": ids[k], "text": texts[k]["task"], "objects": texts[k]["objects"][:6],
                                                             "actions": texts[k]["acts"]}, ensure_ascii=False) for k in b) for b in calls],
                            tokens * size + 500, concurrency)
            for r in (x for rs in rows if rs for x in rs):
                got = row_of(r)
                if got and 0 <= got[0] < len(todo): named.setdefault(todo[got[0]], got[1])
        return named
    coarse_names = name_all("coarse", P["coarse"], lambda r: (r["id"], norm(r["task"])) if isinstance(r, dict)
                            and isinstance(r.get("id"), int) and r.get("task") else None, sorted(texts), 25)
    fine_order = sorted(texts, key=lambda k: (coarse_names.get(k, ""), k))
    fine_names = name_all("specific", P["specific"], lambda r: (r[0], [norm(r[1]), norm(r[2])]) if isinstance(r, list)
                          and len(r) == 3 and isinstance(r[0], int) and r[1] and r[2] else None, fine_order, 30)
    hours = collections.Counter()
    for t in tasks:
        t["cname"] = coarse_names.get(t["key"]); f = fine_names.get(t["key"])
        t["verb0"], t["obj0"] = (f[0], f[1]) if f else (None, None)
        if t["cname"]: hours[t["cname"]] += (t["b"] - t["a"]) / 3600

    # the coarse list, kept from the base or written by one call, and every name placed on it
    clist = [dict(x) for x in vocab.get("coarse_tasks") or []]; cmap = dict(vocab.get("coarse_map") or {})
    if not clist:
        rows = ask(P["list"] + "\n".join(f"{n}\t{h:.2f}" for n, h in hours.most_common(LIST_NAMES)), 32000)
        clist = [x for x in rows or [] if isinstance(x, dict) and x.get("task") and isinstance(x.get("id"), int)]
        for x in clist: x["task"] = norm(x["task"])
    tab = lambda: "\n".join(f"{x['id']}\t{x.get('family')}\t{x['task']}\t{x.get('definition', '')}" for x in clist)
    for rnd in range(3):
        if not clist: break
        nulls = set()
        for size in (LIST_ASSIGN, 100, 25):
            todo = [n for n, _ in hours.most_common() if n not in cmap and n not in nulls]
            if not todo: break
            blocks = [range(a, min(a + size, len(todo))) for a in range(0, len(todo), size)]
            replies = ask.many([P["assign"].format(tasks=tab(), names="\n".join(f"{j}\t{todo[j]}" for j in b)) for b in blocks],
                               24 * size + 500, concurrency)
            got, null, _ = assign_round(todo, blocks, replies, {x["id"] for x in clist})
            cmap.update(got); nulls |= null
        left = add_candidates([n for n in hours if n not in cmap], hours, bool(base))
        if not left or rnd == 2: break
        new = ask(P["add"].format(tasks=tab(), names="\n".join(f"{n}\t{hours[n]:.2f}" for n in left)), 32000) or []
        top = max(x["id"] for x in clist)
        clist += [{**x, "task": norm(x["task"])} for x in new if isinstance(x, dict) and x.get("task")
                  and isinstance(x.get("id"), int) and x["id"] > top]
    byid = {x["id"]: x for x in clist}
    for t in tasks:
        x = byid.get(cmap.get(t["cname"])) if t["cname"] else None
        t["coarse"], t["family"] = (x["task"], x.get("family")) if x else (None, None)

    # verb and object merges, the base's kept
    named = [t for t in tasks if t["verb0"]]
    vc = collections.Counter(t["verb0"] for t in named); oc = collections.Counter(t["obj0"] for t in named)
    vmap = dict(vocab.get("verbs") or {}); omap = dict(vocab.get("objects") or {})
    new_v = [v for v, _ in vc.most_common() if v not in set(vocab.get("known_verbs") or [])]
    if new_v:
        anchors = [v for v, _ in vc.most_common(300)]
        for rows in ask.many([P["verbs"] + "\n".join(f"{v}\t{vc[v]}" for v in anchors) + "\n\nVerbs to place:\n"
                              + "\n".join(f"{v}\t{vc[v]}" for v in new_v[i:i + 400]) for i in range(0, len(new_v), 400)], 8000, concurrency):
            for r in rows or []:
                if isinstance(r, list) and len(r) == 2 and norm(r[0]) in new_v and norm(r[1]) in vc and merge_allowed(norm(r[0]), norm(r[1])):
                    vmap.setdefault(norm(r[0]), norm(r[1]))
    objs = [o for o, _ in oc.most_common()]; known = set(vocab.get("known_objects") or [])
    cand = {i: c for i, c in candidates(objs).items() if objs[i] not in known}
    if cand:
        ex = collections.defaultdict(list)
        for t in named:
            if len(ex[t["obj0"]]) < 2 and t["task"][:110] not in ex[t["obj0"]]: ex[t["obj0"]].append(t["task"][:110])
        items = sorted(cand.items())
        prop = {}
        for rows in ask.many([P["propose"] + "\n".join(f"{i}: {objs[i]} ({oc[objs[i]]}), e.g. \"{(ex[objs[i]] or [''])[0]}\" | candidates: "
                                                       + "; ".join(f"{j} {objs[j]} ({oc[objs[j]]})" for j in c) for i, c in items[s:s + 80])
                              for s in range(0, len(items), 80)], 30 * 80 + 500, concurrency):
            for r in rows or []:
                if (isinstance(r, list) and len(r) == 2 and isinstance(r[0], int) and isinstance(r[1], int)
                        and r[1] in cand.get(r[0], ()) and merge_allowed(objs[r[0]], objs[r[1]])):
                    prop.setdefault(objs[r[0]], objs[r[1]])
        pl = sorted(prop.items())
        for rows in ask.many([P["verify"] + "\n".join(f'{j}: "{a}" e.g. {" / ".join(ex[a])}  ||  "{b}" e.g. {" / ".join(ex[b])}'
                                                      for j, (a, b) in list(enumerate(pl))[s:s + 40]) for s in range(0, len(pl), 40)], 800, concurrency):
            for r in rows or []:
                if isinstance(r, list) and len(r) == 2 and isinstance(r[0], int) and 0 <= r[0] < len(pl) and r[1] is True:
                    omap.setdefault(*pl[r[0]])
    for t in named: t["verb"] = resolve(vmap, t["verb0"]); t["obj"] = resolve(omap, t["obj0"])
    rv = reorder_merges(collections.Counter(t["verb"] for t in named)); ro = reorder_merges(collections.Counter(t["obj"] for t in named))
    for t in named: t["verb"] = rv.get(t["verb"], t["verb"]); t["obj"] = ro.get(t["obj"], t["obj"])
    vmap.update({a: b for a, b in rv.items() if a not in vmap}); omap.update({a: b for a, b in ro.items() if a not in omap})

    out = levels(tasks)
    defs = {x["task"]: x.get("definition") for x in clist}
    for x in out: x["definition"] = defs.get(x["task"])
    summary = {"recordings": len(recs), "tasks": len(tasks), "task_hours": round(sum(t["b"] - t["a"] for t in tasks) / 3600, 2),
               "families": len({x["family"] for x in out if x["family"]}), "coarse_tasks": sum(1 for x in out if x["task"]),
               "specific_tasks": len({(f["verb"], f["object"]) for x in out for f in x["specific"]}),
               "tasks_with_no_coarse_task": sum(1 for t in tasks if not t["coarse"]), "levels_id": levels_id(out),
               "model": ask.model, "reasoning": ask.reasoning, "spent_usd": round(ask.budget.spent, 4),
               "problems": dict(ask.problems), "problem_examples": list(getattr(ask, "examples", []))}
    new_vocab = {"version": VERSION, "coarse_tasks": clist, "coarse_map": cmap, "verbs": vmap, "objects": omap,
                 "known_verbs": sorted(set(vc) | set(vocab.get("known_verbs") or [])),
                 "known_objects": sorted(set(oc) | set(vocab.get("known_objects") or [])),
                 "coarse_names_by_text": coarse_names, "specific_names_by_text": fine_names}
    result = {"summary": summary, "levels": out, "vocab": new_vocab}
    if check:
        result["check"] = run_check(tasks, named, P, ask, check, concurrency, judge_model, summary["levels_id"])
    return result


def run_check(tasks, named, P, ask, n, concurrency, judge_model, lid) -> dict:
    """A fixed sample of n tasks judged on both levels, and n random merges judged as the same task or not."""
    rng = random.Random(SEED)
    pop = [t for t in tasks if t["coarse"] and t.get("verb")]
    sub = rng.sample(pop, min(n, len(pop)))
    by = collections.defaultdict(list)
    for t in named: by[(t["verb"], t["obj"])].append(t)
    merged = [t for t in named if (t["verb0"], t["obj0"]) != (t["verb"], t["obj"])]
    pairs = []
    for t in rng.sample(merged, min(len(merged), 4 * n)):
        others = [x for x in by[(t["verb"], t["obj"])] if (x["verb0"], x["obj0"]) != (t["verb0"], t["obj0"])]
        if others: pairs.append((t, rng.choice(others)))
        if len(pairs) == n: break
    js = ask.many([P["judge"].format(text=t["task"], objects=", ".join(t["objects"]), coarse=t["coarse"], verb=t["verb"], obj=t["obj"]) for t in sub]
                  + [P["judge_merge"].format(va=a["verb0"], oa=a["obj0"], ta=a["task"], vb=b["verb0"], ob=b["obj0"], tb=b["task"]) for a, b in pairs],
                  400, concurrency, judge_model)
    first = lambda rows: next((r for r in rows or [] if isinstance(r, dict)), None)
    jt, jm = [first(r) for r in js[:len(sub)]], [first(r) for r in js[len(sub):]]
    return {"levels_id": lid, "judge_model": judge_model or ask.model,
            "coarse": {"judged": sum(1 for j in jt if j), "correct": sum(1 for j in jt if j and j.get("coarse_correct") is True)},
            "specific": {"judged": sum(1 for j in jt if j), "correct": sum(1 for j in jt if j and j.get("specific_correct") is True)},
            "merges": {"judged": sum(1 for j in jm if j), "correct": sum(1 for j in jm if j and j.get("same_task") is True)},
            "rows": [{"task": t["task"], "coarse": t["coarse"], "specific": f"{t['verb']} {t['obj']}", "judge": j} for t, j in zip(sub, jt)]
                    + [{"a": [a["verb0"], a["obj0"], a["task"]], "b": [b["verb0"], b["obj0"], b["task"]], "judge": j} for (a, b), j in zip(pairs, jm)]}


# ---------------------------------------------------------------- the files

def write(result: dict, out: Path) -> None:
    from label.atomic import write_atomic
    out.mkdir(parents=True, exist_ok=True)
    s, rows = result["summary"], [{**x, "task": x["task"] or NO_COARSE} for x in result["levels"]]
    rows.sort(key=lambda x: (x["task"] == NO_COARSE, -x["hours"]))
    write_atomic(out / "catalog.json", {"summary": s, "check": {k: v for k, v in (result.get("check") or {}).items() if k != "rows"},
                                        "coarse_tasks": rows}, indent=1)
    write_atomic(out / "vocab.json", result["vocab"], indent=None)
    if result.get("check"): write_atomic(out / "check.json", result["check"], indent=1)
    with open(out / "catalog_specific.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["coarse task", "task family", "specific task", "verb", "object", "hours", "success hours", "tasks", "recordings", "examples"])
        for x in rows:
            for f in x["specific"]:
                w.writerow([x["task"], x["family"] or "", f["task"], f["verb"], f["object"], f"{f['hours']:.3f}", f"{f['success_hours']:.3f}",
                            f["tasks"], f["recordings"], " ".join(f["examples"])])
    lines = ["# Task catalog", "",
             f"{s['families']} task families, {s['coarse_tasks']} coarse tasks and {s['specific_tasks']:,} specific tasks (one verb on one "
             f"specific kind of object), over {s['recordings']:,} labelled recordings and {s['tasks']:,} tasks ({s['task_hours']:,.1f} "
             "labelled hours). The specific tasks are in catalog_specific.csv.", "",
             "| coarse task | task family | hours | success hours | tasks | recordings | specific tasks | largest specific tasks |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for x in rows:
        top = ", ".join(f"{f['task']} ({f['hours']:.1f}h)" for f in x["specific"][:3])
        lines.append(f"| {x['task']} | {x['family'] or ''} | {x['hours']:,.2f} | {x['success_hours']:,.2f} | {x['tasks']:,} | "
                     f"{x['recordings']:,} | {len(x['specific']):,} | {top} |")
    (out / "catalog.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    from label import harness, policy
    from label.issue_groups import Budget
    ap = argparse.ArgumentParser(prog="python -m label.catalog", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", type=Path, required=True, help="a stitched result, a folder of them, or a JSON Lines file of results")
    ap.add_argument("--out", type=Path, required=True, help="the folder to write (never the input)")
    ap.add_argument("--base", type=Path, default=None, help="the OUT of an earlier catalog, whose vocabulary and cache this one keeps")
    ap.add_argument("--cap", type=float, default=50.0, help="most the calls may spend, in USD (default 50)")
    ap.add_argument("--model", default=policy.ASTRA, help=f"default: {policy.ASTRA}")
    ap.add_argument("--reasoning", default=REASONING, help=f"default: {REASONING}")
    ap.add_argument("--judge-model", default=None, help="the checks' model (default: --model)")
    ap.add_argument("--check", type=int, default=100, help="tasks and merges in each check sample (0: no checks)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="read and count; ask nothing")
    a = ap.parse_args(argv)
    if a.base and a.base.resolve() == a.out.resolve(): raise SystemExit("--out must differ from --base")
    recs = read_results(a.label)
    base = json.loads((a.base / "vocab.json").read_text()) if a.base else None
    cache = Cache(a.out / "cache.jsonl")
    if a.base:
        for k, v in Cache(a.base / "cache.jsonl").answers.items(): cache.answers.setdefault(k, v)
    keys = harness.get_keys()
    if not keys and not a.dry_run: raise SystemExit("no model key: set OPENROUTER_API_KEYS or OPENAI_API_KEY")
    ask = Asker(a.model, a.reasoning, cache, Budget(a.cap), keys, a.dry_run)
    result = build(recs, ask, base=base, concurrency=a.concurrency, check=0 if a.dry_run else a.check, judge_model=a.judge_model)
    write(result, a.out)
    print(json.dumps({**result["summary"], "check": {k: v for k, v in (result.get("check") or {}).items() if k != "rows"}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
