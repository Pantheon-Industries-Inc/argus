"""The numbers on the dashboard's home view: how far a labelling run has got and what the labelled footage holds.

    GET /api/home      (board/serve.py)

Units. A subtask is one task of a session (an entry of the label's tasks list), or a whole episode where the episode
is one task, as a scripted clip is. An object counts for a subtask when the subtask handles it, meaning the subtask's
task names it or one of the subtask's events acts on it; the episode's own object list, which holds everything on
the table, is never counted. Success and operator mistakes are counted per subtask, each mistake going to the
subtask its time falls in (a mistake with no time counts for one subtask). Data issues describe the recording, such
as swapped cameras or an instruction that names the wrong object, so they are counted per episode.

For the whole board and for each dataset the page shows
- progress: episodes and footage hours labelled against the run's plan (BOARD/plan.json);
- pace: footage hours labelled per hour of labelling, over the last hour of labelling, and the time left at that
  pace (the plan's hours not yet labelled, divided by the pace);
- cost: the labels' billed cost so far (each label's _usage.est_cost_usd), the cost per footage hour, and that rate
  times the plan's hours as the cost of the whole run;
- subtasks, subtasks per minute of footage, the share of subtasks that succeeded, the share of episodes with a data
  issue and the share of subtasks with an operator mistake (the issues the episode list counts, Families.counts);
- diversity: how many kinds of object are handled, of verb and of task, and how evenly they are spread (the
  effective number, e to the Shannon entropy of the counts: the number of equally common kinds that would give the
  same spread, so 40 kinds where one fills nine tenths of the footage counts as far fewer than 40);
- whether new kinds still turn up: distinct kinds against labelled hours, in the order the episodes were labelled;
- the five commonest kinds of object (subtasks that handle each) and task verbs (subtasks with each), each split
  by dataset, and the share of handled objects that are deformable (each kind's tag, board/materials.py).

Kinds come from the labels' own words, never from a fixed list. An object's kind is the head noun of its name
("pebble container" is a container, "clear test tubes" a tube, "10 of diamonds" a card); a task is its verb and the
kind of the first object it names ("Place the closed pebble container upright on the tray" is place container). The
rules are simple and the same for every dataset, so the datasets compare with each other even where a rule misreads a
name. A task's verb is the one the labeler names for it (an episode's task_verb, a session task's verb,
label/prompts.py): its predominant action in one or two words, never move. Verbs naming the same action ("pick" and
"pick up") are counted under one name that a small model gives each distinct verb (board/verbs.py). A label written
before the labeler named verbs counts the task sentence's first verb.

plan.json, written when a run starts:
    {"datasets": {"umi_scripted": {"name": "Scripted", "episodes": 6224, "seconds": 88474.8}, ...}}
Its order is the page's order and its names are the page's names. Without it the page shows what is labelled and
no progress bar, time left or whole-run cost.

Each episode file is read once and its summary kept until the file changes, so a board that gains a label every
few seconds reads one new file, not thousands.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

from board.verbs import load_names, verb_of

PACE_WINDOW_S = 3600        # the pace is the footage labelled in the last hour of labelling
PACE_MIN_LABELS = 3         # fewer labels than this in the window give no pace and no time left
TOP_N = 12                  # bars per chart
CURVE_POINTS = 48           # points per line on the "new kinds" chart
FEED_N = 14                 # latest labels shown
LENGTH_BINS = [(0, 15, "under 15s"), (15, 30, "15 to 30s"), (30, 60, "30 to 60s"), (60, 120, "1 to 2 min"),
               (120, 300, "2 to 5 min"), (300, 600, "5 to 10 min"), (600, None, "over 10 min")]

# ---- kinds from the labels' own words ----

_PAREN = re.compile(r"\([^)]*\)")
_TOKEN = re.compile(r"[a-z0-9][a-z0-9'-]*")
# "pair of scissors" is scissors, but "can of tuna" is a can: the words that name an amount of the thing after "of"
_AMOUNT = {"pair", "set", "piece", "pieces", "bunch", "cluster", "stack", "pile", "sheet", "strip", "handful", "slice"}
_SUITS = {"heart", "diamond", "club", "spade"}
_SKIP_VERB = {"try", "attempt", "begin", "start", "continue"}   # "Try to stand the comb upright" is stand


# nouns that are plural in form only
_PLURAL_ONLY = {"scissors", "glasses", "sunglasses", "goggles", "tongs", "pliers", "tweezers", "pants", "jeans",
                "shorts", "trousers", "headphones", "earphones", "binoculars", "clothes"}


def singular(w: str) -> str:
    if len(w) <= 3 or w.endswith(("ss", "us", "is")) or w in _PLURAL_ONLY:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ches", "shes", "xes", "sses", "zes")):
        return w[:-2]
    return w[:-1] if w.endswith("s") else w


def object_kind(name: str) -> str | None:
    """The head noun of an object's name, singular: "clear test tubes" -> tube, "can of tuna" -> can."""
    s = _PAREN.sub(" ", str(name or "").lower())
    s = re.split(r"\s+(?:and|with|or)\s+|,", s)[0]          # "cloth and blue plastic piece" is a cloth
    if " of " in s:
        head, rest = s.split(" of ", 1)
        words = _TOKEN.findall(head)
        if _TOKEN.findall(rest)[-1:] and singular(_TOKEN.findall(rest)[-1]) in _SUITS:
            return "card"                   # "10 of diamonds", "queen of spades": a playing card
        if not words or words[-1].isdigit() or words[-1] in _AMOUNT:
            s = rest
        else:
            s = head
    words = [w for w in _TOKEN.findall(s) if not w.isdigit()]
    return singular(words[-1]) if words else None


def task_kind(sentence: str, names: list) -> str | None:
    """A task's verb and the kind of the first object it names: reading on from the verb, the first place where one
    of the episode's object names starts, or the first word that is one of their kinds ("Lift clear tubes from the
    rack" with an object "clear test tubes" is lift tube). Only the verb when it names none of them."""
    words = _TOKEN.findall(str(sentence or "").lower())
    # "Gently place the cup" is place: a leading adverb (a word of six letters or more ending in ly) is skipped
    while len(words) > 1 and (words[0] in _SKIP_VERB or words[0] == "to"
                              or (len(words[0]) > 5 and words[0].endswith("ly"))):
        words = words[1:]
    if not words:
        return None
    verb = words[0]
    full = sorted({tuple(_TOKEN.findall(str(n).lower())): n for n in names if str(n).strip()}.items(),
                  key=lambda kv: -len(kv[0]))
    kinds = {k for k in (object_kind(n) for n in names) if k}
    for i in range(1, len(words)):
        for toks, n in full:
            if toks and tuple(words[i:i + len(toks)]) == toks:
                k = object_kind(n)
                return f"{verb} {k}" if k else verb
        if singular(words[i]) in kinds:
            return f"{verb} {singular(words[i])}"
    return verb


# ---- one episode file, summarised ----

def summarize(p: Path, d: dict, counts, name=None) -> dict:
    """What the home page needs from one episode file. counts(list_key, issue) is Families.counts: whether an issue
    counts, the same rule as the rail and the issue filter. name(list_key, issue, dataset) is the issue's family as the
    episode list's filter names it (serve.py issue_family_name); without it an issue goes by its tag."""
    tag = (lambda key, i: name(key, i, d.get("dataset"))) if name else \
        (lambda key, i: str(i.get("category") or i.get("issue"))[:60])
    names = [str(o.get("name") or "") for o in (d.get("objects") or []) if isinstance(o, dict) and o.get("name")]
    events = [e for e in (d.get("event_labels") or []) if isinstance(e, dict)]
    tasks = [t for t in (d.get("tasks") or []) if isinstance(t, dict) and t.get("task")]
    # the objects each subtask handles: the ones the task names and the ones its events act on. The episode's own
    # object list holds everything on the table, distractors included, so it is not counted.
    def ev_names(lo, hi):
        return [str(e["object"]) for e in events if e.get("object") and isinstance(e.get("t_s"), (int, float))
                and lo - 0.5 <= e["t_s"] <= hi + 0.5]
    if tasks:
        spans = [(float(t.get("start_s") or 0), float(t.get("end_s") or 0)) for t in tasks]
        handled = [[*(o.get("name") if isinstance(o, dict) else o for o in (t.get("objects") or [])), *ev_names(*sp)]
                   for t, sp in zip(tasks, spans)]
    else:
        spans = [(float("-inf"), float("inf"))]
        handled = [ev_names(*spans[0])]
    sub_kinds = [sorted({k for k in (object_kind(n) for n in hs if n) if k}) for hs in handled]
    handled_names = sorted({str(n).strip().lower() for hs in handled for n in hs if n and str(n).strip()})
    if tasks:
        task_rows = [(task_kind(t["task"], [*(t.get("objects") or []), *names]), (t.get("outcome") or "").lower())
                     for t in tasks]
    else:
        task_rows = [(task_kind(d.get("episode_prompt") or "", names),
                      ((d.get("completion") or {}).get("task_completed") or "").lower())]
    # each subtask's verb: the labeler's (label/prompts.py), or the task sentence's first verb in a label written
    # before it named one. The events' own verbs (approach, grasp, lower, release) are the steps of nearly every task,
    # so they tell tasks apart poorly and are not counted.
    # a label with the field counts the labeler's verb, or none where it says no action happened
    named = [t.get("verb") for t in tasks] if tasks else [d.get("task_verb")]
    verbs = Counter(verb_of(v) if isinstance(v, str) else (k.split()[0] if k else None)
                    for v, (k, _) in zip(named, task_rows))
    verbs.pop(None, None)
    issues = [i for i in (d.get("data_issues") or []) if isinstance(i, dict) and i.get("issue")
              and counts("data_issues", i)]
    mistakes = [i for i in (d.get("operator_mistakes") or []) if isinstance(i, dict) and i.get("issue")
                and counts("operator_mistakes", i)]
    # each mistake goes to the subtask its time falls in; one with no time counts for one subtask
    hit = {next((j for j, (lo, hi) in enumerate(spans) if lo - 0.5 <= m["t_s"] <= hi + 0.5), None)
           for m in mistakes if isinstance(m.get("t_s"), (int, float))} - {None}
    untimed = sum(1 for m in mistakes if not isinstance(m.get("t_s"), (int, float)))
    mistake_subtasks = min(len(spans), len(hit) + untimed)
    usage = d.get("_usage") or {}
    st = p.stat()
    return {
        "file": p.name,
        "dataset": d.get("dataset") or "",
        "seconds": float(d.get("duration_s") or 0.0),
        "cost": float(usage.get("est_cost_usd") or 0.0),
        # when the label was written: board/follow.py stamps the run output's time; a built board has the file's
        "at": float(d.get("_labelled_at") or st.st_mtime),
        "prompt": d.get("episode_prompt") or "",
        "sessions": bool(tasks),
        "outcome": ((d.get("completion") or {}).get("task_completed") or "").lower() or None,
        "object_names": handled_names,
        "object_kinds": sorted({k for ks in sub_kinds for k in ks}),
        "subtask_objects": sub_kinds,
        "mistake_subtasks": mistake_subtasks,
        "verbs": dict(verbs),
        "tasks": task_rows,
        "issue_tags": [tag("data_issues", i) for i in issues],
        "mistake_tags": [tag("operator_mistakes", i) for i in mistakes],
    }


# ---- the numbers ----

def effective_number(c: Counter) -> float:
    n = sum(c.values())
    if not n:
        return 0.0
    h = -sum((v / n) * math.log(v / n) for v in c.values() if v)
    return round(math.exp(h), 1)


def _top(c: Counter, n: int = TOP_N) -> list:
    return [[k, v] for k, v in c.most_common(n)]


def _diversity(rows: list) -> dict:
    objs = Counter(k for r in rows for ks in r["subtask_objects"] for k in ks)   # subtasks that handle each kind
    names = {n for r in rows for n in r["object_names"]}
    verbs = Counter()
    for r in rows:
        verbs.update(r["verbs"])
    tasks = Counter(k for r in rows for k, _ in r["tasks"] if k)
    return {
        "objects": {"distinct": len(objs), "names": len(names), "effective": effective_number(objs), "top": _top(objs)},
        "verbs": {"distinct": len(verbs), "effective": effective_number(verbs), "top": _top(verbs)},
        "tasks": {"distinct": len(tasks), "effective": effective_number(tasks), "top": _top(tasks)},
    }


def _curve(rows: list) -> dict:
    """Distinct object kinds, tasks and task verbs against labelled hours, in the order the episodes were labelled:
    points are [hours, objects, tasks, verbs]."""
    rows = sorted(rows, key=lambda r: (r["at"], r["file"]))
    seen_o, seen_t, seen_m, h, pts = set(), set(), set(), 0.0, []
    for r in rows:
        h += r["seconds"] / 3600
        seen_o.update(r["object_kinds"])
        seen_t.update(k for k, _ in r["tasks"] if k)
        seen_m.update(r["verbs"])
        pts.append([round(h, 3), len(seen_o), len(seen_t), len(seen_m)])
    if len(pts) > CURVE_POINTS:
        step = (len(pts) - 1) / (CURVE_POINTS - 1)
        pts = [pts[round(i * step)] for i in range(CURVE_POINTS)]
    return {"points": pts}


def _outcomes(rows: list) -> dict:
    """Success and failure: per episode where an episode is one task, per task where it is a session."""
    c = Counter()
    for r in rows:
        for _, o in (r["tasks"] if r["sessions"] else [(None, r["outcome"] or "")]):
            c[o if o in ("success", "failure") else "other"] += 1
    return {"unit": "tasks" if any(r["sessions"] for r in rows) else "episodes", **c}


def _lengths(rows: list) -> list:
    out = []
    for lo, hi, name in LENGTH_BINS:
        out.append([name, sum(1 for r in rows if r["seconds"] >= lo and (hi is None or r["seconds"] < hi))])
    return out


def _pace(rows: list, now: float) -> dict | None:
    """Footage hours labelled per hour of labelling, over the last hour (or since the first label, if later)."""
    recent = [r for r in rows if r["at"] >= now - PACE_WINDOW_S]
    if len(recent) < PACE_MIN_LABELS:
        return None
    span = now - min(r["at"] for r in recent)
    if span < 300:                          # under five minutes of labelling says little about the next hours
        return None
    return {"footage_h_per_h": round(sum(r["seconds"] for r in recent) / span, 3),
            "labels_per_h": round(len(recent) * 3600 / span, 1), "window_s": round(span)}


def _top_split(counts: dict, n: int = 5) -> list:
    """The n commonest kinds over every dataset, each with its count per dataset: [[kind, total, {dataset: n}]]."""
    tot = sorted(counts.items(), key=lambda kv: (-sum(kv[1].values()), kv[0]))[:n]
    return [[k, sum(c.values()), dict(c)] for k, c in tot]


def _deformable(rows: list, tags: dict) -> dict:
    """Of the objects the subtasks handle (one count per kind per subtask), how many are deformable, how many rigid,
    and how many have no tag yet (board/materials.py)."""
    c = Counter()
    for r in rows:
        for k in (k for ks in r["subtask_objects"] for k in ks):
            t = tags.get(k)
            c["deformable" if t is True else "rigid" if t is False else "untagged"] += 1
    return dict(c)


def _projected(datasets: list, pds: dict) -> float | None:
    """The whole run's cost: each planned dataset's footage hours at its measured cost per footage hour, or at its
    plan.json rate while it has no labels yet; None while a planned dataset has neither."""
    total = 0.0
    for d in datasets:
        if not d["plan_seconds"]:
            continue
        rate = (d["cost"] / (d["seconds"] / 3600) if d["seconds"]
                else (pds.get(d["dataset"]) or {}).get("cost_per_footage_h"))
        if rate is None:
            return None
        total += rate * d["plan_seconds"] / 3600
    return round(total, 0)


def _named(rows: list, names: dict) -> list:
    """Each row with its verbs counted under the names board/verbs.py gave them; a verb not merged yet keeps its own."""
    out = []
    for r in rows:
        c = Counter()
        for v, n in r["verbs"].items():
            c[names.get(v) or v] += n
        out.append({**r, "verbs": dict(c)})
    return out


def stats(rows: list, plan: dict, now: float, tags: dict | None = None, names: dict | None = None) -> dict:
    tags = tags or {}
    rows = _named(rows, names or {})
    pds = (plan or {}).get("datasets") or {}
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)
    order = list(pds) + sorted(d for d in by_ds if d not in pds)
    datasets = []
    for ds in order:
        rs = by_ds.get(ds, [])
        p = pds.get(ds) or {}
        sec = sum(r["seconds"] for r in rs)
        cost = sum(r["cost"] for r in rs)
        pace = _pace(rs, now)
        left_s = (p.get("seconds") or 0) - sec if p.get("seconds") else None
        datasets.append({
            "dataset": ds, "name": p.get("name") or ds,
            "episodes": len(rs), "seconds": round(sec, 1),
            "plan_episodes": p.get("episodes"), "plan_seconds": p.get("seconds"),
            "cost": round(cost, 2), "cost_per_footage_h": round(cost / (sec / 3600), 2) if sec else None,
            "pace": pace,
            "eta_s": round(left_s / pace["footage_h_per_h"]) if pace and left_s and left_s > 0 else None,
            "diversity": _diversity(rs), "curve": _curve(rs), "outcomes": _outcomes(rs), "lengths": _lengths(rs),
            "issues_per_h": round(sum(len(r["issue_tags"]) for r in rs) / (sec / 3600), 2) if sec else None,
            "mistakes_per_h": round(sum(len(r["mistake_tags"]) for r in rs) / (sec / 3600), 2) if sec else None,
            "deformable": _deformable(rs, tags),
            # episodes with at least one data issue, and with at least one operator mistake, that the episode list counts
            # subtasks: a session's tasks, or one per episode where an episode is one task
            "n_tasks": sum(len(r["tasks"]) for r in rs),
            "eps_with_issue": sum(1 for r in rs if r["issue_tags"]),
            "eps_with_mistake": sum(1 for r in rs if r["mistake_tags"]),
            "subtasks_with_mistake": sum(r["mistake_subtasks"] for r in rs),
            # data issues by the episodes that have each, operator mistakes by how often each happened
            "top_issues": _top(Counter(t for r in rs for t in set(r["issue_tags"])), 5),
            "top_mistakes": _top(Counter(t for r in rs for t in r["mistake_tags"]), 5),
        })
    sec = sum(r["seconds"] for r in rows)
    cost = sum(r["cost"] for r in rows)
    plan_sec = sum((p.get("seconds") or 0) for p in pds.values()) or None
    plan_eps = sum((p.get("episodes") or 0) for p in pds.values()) or None
    pace = _pace(rows, now)
    left_s = plan_sec - sec if plan_sec else None
    latest = sorted(rows, key=lambda r: (r["at"], r["file"]), reverse=True)[:FEED_N]
    objs, verbs = defaultdict(Counter), defaultdict(Counter)
    for r in rows:
        for k in (k for ks in r["subtask_objects"] for k in ks):
            objs[k][r["dataset"]] += 1
        for v, n in r["verbs"].items():
            verbs[v][r["dataset"]] += n
    return {
        "now": now,
        "has_plan": bool(pds),
        "total": {
            "episodes": len(rows), "seconds": round(sec, 1), "plan_episodes": plan_eps, "plan_seconds": plan_sec,
            "cost": round(cost, 2), "cost_per_footage_h": round(cost / (sec / 3600), 2) if sec else None,
            # each planned dataset at its own cost per footage hour once it has labels, so a cheap tranche labelled
            # first does not set the price of the rest, and before that at the rate plan.json gives it
            # ("cost_per_footage_h", from a pilot); none while a planned dataset has neither
            "projected_cost": _projected(datasets, pds) if plan_sec else None,
            "pace": pace,
            "eta_s": round(left_s / pace["footage_h_per_h"]) if pace and left_s and left_s > 0 else None,
            "last_at": max((r["at"] for r in rows), default=None),
            "diversity": _diversity(rows),
            "deformable": _deformable(rows, tags),
            "outcomes": _outcomes(rows),
            "n_tasks": sum(len(r["tasks"]) for r in rows),
            "eps_with_issue": sum(1 for r in rows if r["issue_tags"]),
            "eps_with_mistake": sum(1 for r in rows if r["mistake_tags"]),
            "subtasks_with_mistake": sum(r["mistake_subtasks"] for r in rows),
            "top_issues": _top(Counter(t for r in rows for t in set(r["issue_tags"])), 5),
            "top_mistakes": _top(Counter(t for r in rows for t in r["mistake_tags"]), 5),
            # the five commonest kinds of object (with their tag) and task verbs, split by dataset
            "top_objects": [[k, n, by, tags.get(k)] for k, n, by in _top_split(objs)],
            "top_verbs": _top_split(verbs),
        },
        "datasets": datasets,
        "latest": [{"file": r["file"], "dataset": r["dataset"], "prompt": r["prompt"], "at": r["at"],
                    "seconds": r["seconds"], "outcome": r["outcome"],
                    "tasks": len(r["tasks"]) if r["sessions"] else None,
                    "tasks_done": sum(1 for _, o in r["tasks"] if o == "success") if r["sessions"] else None}
                   for r in latest],
    }


# ---- cached per folder ----

_LOCK = threading.Lock()
_SUMMARIES: dict = {}       # folder -> {file name: (mtime_ns, size, summary)}
_BODY: dict = {}            # folder -> (signature, raw JSON, gzipped JSON, etag, built at)
REBUILD_EVERY_S = 15        # the pace and the times on the page move with the clock even when no label lands


def _summaries(here: Path, counts, name=None) -> list:
    with _LOCK:
        known = dict(_SUMMARIES.get(str(here)) or {})
    fresh = {}
    for p in here.glob("*.json"):
        try:
            st = p.stat()
        except OSError:
            continue
        hit = known.get(p.name)
        if hit and hit[0] == st.st_mtime_ns and hit[1] == st.st_size:
            fresh[p.name] = hit
            continue
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue                        # half-written or not an episode file: read again next time
        if isinstance(d, dict) and "episode_prompt" in d:
            fresh[p.name] = (st.st_mtime_ns, st.st_size, summarize(p, d, counts, name))
    with _LOCK:
        _SUMMARIES[str(here)] = fresh
    return [v[2] for v in fresh.values()]


def load_tags(path: Path) -> dict:
    """{kind: True if deformable, False if rigid} from BOARD/materials.json (board/materials.py)."""
    try:
        kinds = json.loads(path.read_text()).get("kinds") or {}
    except (OSError, ValueError, AttributeError):
        return {}
    return {k: v.get("deformable") for k, v in kinds.items() if isinstance(v, dict)}


def home_json(here: Path, plan_path: Path, counts, name=None) -> tuple:
    """(JSON, gzipped JSON, ETag) of the home page's numbers. Rebuilt when a label lands, the plan changes or
    REBUILD_EVERY_S passes; between those every visitor gets the same bytes, and a visitor whose copy is current
    gets 304 from serve.py."""
    try:
        sig_files = tuple(sorted((p.name, p.stat().st_mtime_ns) for p in here.glob("*.json")))
    except OSError:
        sig_files = ()
    tags_path, names_path = plan_path.with_name("materials.json"), plan_path.with_name("verbs.json")
    plan_m = tuple(p.stat().st_mtime_ns if p.exists() else None for p in (plan_path, tags_path, names_path))
    sig = (hashlib.sha1(repr(sig_files).encode()).hexdigest(), plan_m)
    now = time.time()
    with _LOCK:
        hit = _BODY.get(str(here))
    if hit and hit[0] == sig and now - hit[4] < REBUILD_EVERY_S:
        return hit[1], hit[2], hit[3]
    try:
        plan = json.loads(plan_path.read_text())
    except (OSError, ValueError):
        plan = {}
    raw = json.dumps(stats(_summaries(here, counts, name), plan, now, load_tags(tags_path), load_names(names_path)),
                     separators=(",", ":")).encode()
    gz = gzip.compress(raw, compresslevel=5)
    etag = '"' + hashlib.sha1(raw).hexdigest()[:20] + '"'
    with _LOCK:
        _BODY[str(here)] = (sig, raw, gz, etag, now)
    return raw, gz, etag


def plan_names(plan_path: Path) -> dict:
    """{dataset: {"name", "order"}} from plan.json, so the dataset tabs use the plan's names and order."""
    try:
        pds = json.loads(plan_path.read_text()).get("datasets") or {}
    except (OSError, ValueError, AttributeError):
        return {}
    return {ds: {"name": (v or {}).get("name") or ds, "order": i} for i, (ds, v) in enumerate(pds.items())}
