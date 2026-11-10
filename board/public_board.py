"""The rules every public board gets from `python -m board static site`, each an option of the site build.

--public-header      one title top left (--title), with the dataset tabs and the repeated board name hidden
--customer           no data-issue trace on the page or in its data, checked by an audit that fails the build
--split-episodes F   a JSON list of the board's episode ids that are subtasks already split out of a longer recording:
                     their cards show the subtask prompt, every other card the recording's summary
--episode-order F    a JSON list of episode ids shown first, in that order, in data/index.json, the lists and the rail
--audit-deny RE      a pattern that must not appear anywhere in a customer build (repeatable), such as an internal
                     host or folder name

A public board (--public-header or --customer) never shows an empty card: the build fails when a prompt or summary is
empty.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

from board.families import Families, check_hit

HIDE_REPEATS = "<style>#coverage,#rd-name{display:none!important}</style>"
ISSUE_COUNTS = ("n_issues", "n_minor_issues", "n_unrated_issues")
FAMILY_LISTS = ("families", "minor_families", "unrated_families")
# words that only a data-issue trace puts in a build's data: a state note's wording and the page's data family prefix,
# with every data-issue tag (board/tag_names.json) added by data_flag_words
FLAG_TEXT = (r"Kinematic checks unavailable", r'"d:')
# folders a build's machine names, which a public page never shows
INTERNAL_PATHS = r"(?<![\w.])/(?:vol|tmp|root|home|Users|mnt|private|var/folders)/"

_FAMILIES = None


def families() -> Families:
    global _FAMILIES
    if _FAMILIES is None:
        from board.serve import FAMILIES
        _FAMILIES = FAMILIES
    return _FAMILIES


def header_markup(title: str) -> str:
    """The public header: the title alone, top left, in the page's own title bar style."""
    return f'<header class="page-head"><h1>{html.escape(title)}</h1></header>'


def with_public_header(page: str) -> str:
    """The rendered page with the dataset tabs and the rail's repeated board name hidden."""
    return page.replace("</head>", HIDE_REPEATS + "</head>", 1)


def read_ids(path: Path | None, option: str) -> list:
    if path is None:
        return []
    ids = json.loads(Path(path).read_text())
    if not isinstance(ids, list) or not all(isinstance(x, str) and x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{option} must be a JSON list of distinct episode ids")
    return ids


def _data_family(fam) -> bool:
    return isinstance(fam, str) and families().list_of(fam) == "data"


def _reader_family(x: dict) -> str:
    # the page's own rule (board/index.html readerIssueEntries): the issue's family, else "d:" and its kind's words
    return x.get("family") or families().reader_family(str(x.get("kind") or "reader issue"))


def _without_data_checks(dc: dict) -> dict:
    out = dict(dc)
    if isinstance(out.get("reader_issues"), list):
        out["reader_issues"] = [x for x in out["reader_issues"]
                                if not (isinstance(x, dict) and _data_family(_reader_family(x)))]
    # a deterministic check that raised a data family (board/families.json "checks") goes with the family it raised
    for f in families().defs:
        if f.get("list") == "data":
            for key in f.get("checks") or []:
                if check_hit({"dataset_checks": out}, key):
                    out.pop(key.split(".")[0], None)
    if isinstance(out.get("capture_qc"), dict) and out["capture_qc"].get("flags"):
        out["capture_qc"] = {**out["capture_qc"], "flags": []}
    return out


def without_issue_traces(x, key: str | None = None):
    """x (an episode view, a rail record or the index) with every data-issue trace removed: data issues emptied, their
    counts zeroed, no top issue or severity, no data family in any list, no reader issue or check the page counts as a
    data issue, and no state note above the clips."""
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k == "data_issues":
                out[k] = []
            elif k in ISSUE_COUNTS:
                out[k] = 0
            elif k in ("max_severity", "top_issue"):
                out[k] = None
            elif k == "dataset_checks" and isinstance(v, dict):
                out[k] = without_issue_traces(_without_data_checks(v), k)
            elif k == "reader_notes" and isinstance(v, dict):
                out[k] = without_issue_traces({n: w for n, w in v.items() if n != "state_note"}, k)
            else:
                out[k] = without_issue_traces(v, k)
        return out
    if isinstance(x, list):
        return [without_issue_traces(v) for v in x
                if not (isinstance(v, str) and (v.startswith("d:") or key in FAMILY_LISTS and _data_family(v)))]
    return x


def subtask_prompt(view: dict) -> str:
    """A split episode's card text: its subtasks' prompts in order, numbered when there are several."""
    tasks = list(dict.fromkeys(str(t.get("task") or "").strip() for t in view.get("tasks") or []
                               if isinstance(t, dict) and str(t.get("task") or "").strip()))
    return tasks[0] if len(tasks) == 1 else " ".join(f"({i}) {t}" for i, t in enumerate(tasks, 1))


def card_text(view: dict, split: bool) -> str:
    return subtask_prompt(view) if split else str(view.get("episode_prompt") or "").strip()


def ordered(items: list, order: list, eid) -> list:
    """items with the episodes named in order first, in that order; the rest keep theirs."""
    rank = {e: i for i, e in enumerate(order)}
    return sorted(items, key=lambda x: rank.get(eid(x), len(order)))


def data_flag_words(categories=()) -> list:
    """Patterns for the data-issue tags a customer build's data must not name: Argus's own (board/tag_names.json and
    board/families.json) and the categories the board's own labels gave their data issues (a dataset's labels can coin
    their own, such as DiverseUMI-10k's face_visible). One-word tags are ordinary words in prose, so only tags with an
    underscore are searched."""
    names = json.loads((Path(__file__).resolve().parent / "tag_names.json").read_text())
    tags = {*names["data_issues"], *(t for f in families().defs if f.get("list") == "data" for t in f.get("tags") or []),
            *categories}
    return [r"\b" + re.escape(t) + r"\b" for t in sorted(tags) if isinstance(t, str) and "_" in t] + list(FLAG_TEXT)


def label_categories(labels) -> set:
    """The categories of the data issues in these labels (the board's episode files, before the customer rules)."""
    return {str(i.get("category")) for d in labels for i in d.get("data_issues") or []
            if isinstance(i, dict) and i.get("category")}


def audit(site: Path, deny: list, categories=()) -> list:
    """Every file of a customer build (its page, data and BUILD.json) that names an internal folder or a deny pattern,
    or whose data names a data-issue tag, a state note or a data family, and every episode or rail record that still
    carries a data issue: [(file, what)]. Empty means the build may be published."""
    hits = []
    anywhere = [re.compile(p) for p in [INTERNAL_PATHS, *deny]]
    in_data = [re.compile(p) for p in data_flag_words(categories)]
    for p in sorted(q for q in site.rglob("*") if q.is_file()):
        text = p.read_text(errors="replace")
        rel = p.relative_to(site).as_posix()
        for rx in anywhere + (in_data if rel.startswith("data/") else []):
            m = rx.search(text)
            if m:
                hits.append((rel, m.group(0)))
    for p in sorted((site / "data/ep").glob("*.json")):
        d = json.loads(p.read_text())
        counted = families().classify(d)["counted"]
        left = sorted(f for f in counted if _data_family(f))
        if d.get("data_issues") or left or (d.get("reader_notes") or {}).get("state_note"):
            hits.append((p.relative_to(site).as_posix(), f"data issue left: {left or 'data_issues or state_note'}"))
    for p in sorted((site / "data/lists").glob("*.json")):
        for r in json.loads(p.read_text()):
            if any(r.get(k) for k in ISSUE_COUNTS) or any(_data_family(f) for k in FAMILY_LISTS for f in r.get(k) or []):
                hits.append((p.relative_to(site).as_posix(), f"{r.get('episode_id')} counts a data issue"))
    return hits
