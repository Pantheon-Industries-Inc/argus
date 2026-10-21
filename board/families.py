"""Which problem each flagged issue is: the classification behind the board's problem filter.

families.json lists the families. Every flagged issue (a data issue or an operator mistake) belongs to exactly one
family: the first listed family whose tags, plain names (tag_names.json) or text it matches, else a family named
after its tag's plain name ("d:<name>" or "m:<name>"). A family can also be raised by one of the deterministic
checks or by the episode's outcome, and then counts at any severity. So does each problem an episode was kept and
flagged with (context.json reader_issues, copied into dataset_checks by board/build.py): the listed family whose
"reader_issues" names its kind, else a data family named after the kind ("d:<words of the kind>"). Only a family of a
list in COUNTED_LISTS (a fault in the recording, an operator mistake) counts; a reader issue of a model reply that
gave no labels (list "labelling") or of how we read or showed the recording (list "handling": a limit of ours, such as
a table read every so many rows or a signal kept as its lowest, mean and highest value, or a property of the recording
that is not a fault in it, such as a camera that is not colour) is shown on the episode and returned under
not_counted, never in the counts or the filter. A family limited to some datasets ("datasets") is only matched on
those.

A family's text is read only where the tag says too little: a tag the model files several distinct problems under
(a family's "among", such as camera_fault: a camera turned away, a frozen image, glare) or a tag it coined (no entry
in tag_names.json). A known tag keeps its own name whatever its prose mentions. A family with "among" matches its text
only on issues whose tag reads as one of those plain names. The text claims a family only where its clause asserts the
problem (board/polarity.py: not where it is denied, retracted, asked about or supposed), the issue sentence before
its evidence, and a word two swapped letters away from a family's word counts as that word.

What counts (Families.counts, the one statement of the rule):
  - a data issue at medium or high severity;
  - an operator mistake that changes the outcome (its tag's kind in tag_names.json is "outcome") at medium or high;
  - any operator mistake at high.
Rated issues below these thresholds are minor. Issues without a valid severity are unrated.
Both remain visible on the episode and are excluded from severity totals.

    fam = Families()
    c = fam.classify(episode)      # {"counted": {slug: [issues]}, "minor": {...}, "unrated": {...}, "not_counted": {...}}
"""
from __future__ import annotations

import collections
import json
import re
from re import _constants as _C, _parser
from pathlib import Path

from board.label_projection import severity as _sev, tasks_format
from board.polarity import denied_spans

HERE = Path(__file__).resolve().parent
LISTS = {"data_issues": "data", "operator_mistakes": "mistake"}
COUNTED_LISTS = set(LISTS.values())
# characters around a word that spell correction reads: far wider than the longest family wording
# ("fills much of the kitchen camera"), so a family word overlapping the word always lies inside
_SPELL_REACH = 64
_WORD_CHAR = re.compile(r"\w")
# a character that any word character fills, in the strings a family's pattern matches (_wordings)
_WILD = "\x00"


def union_seconds(spans) -> float:
    """Seconds covered by a set of (start, end) spans, overlaps counted once."""
    total, end = 0.0, None
    for a, b in sorted((a, b) for a, b in spans if b > a):
        if end is None or a > end:
            total += b - a
            end = b
        elif b > end:
            total += b - end
            end = b
    return total


def check_hit(d: dict, key: str) -> bool:
    """key is "check.field": the deterministic check's flag on this episode."""
    name, field = key.split(".")
    return bool(((d.get("dataset_checks") or {}).get(name) or {}).get(field))


def _wordings(items, limit=50000):
    """Every string a parsed pattern matches, with _WILD for a character any word character fills; None when the
    pattern uses a form this does not spell out or matches more than limit strings."""
    out = ['']
    for op, av in items:
        if op is _C.LITERAL:
            alts = [chr(av)]
        elif op is _C.IN:
            alts = [chr(v) for o, v in av] if all(o is _C.LITERAL for o, v in av) else [_WILD]
        elif op is _C.ANY:
            alts = [_WILD]
        elif op is _C.AT:
            alts = ['']
        elif op is _C.SUBPATTERN:
            alts = _wordings(av[-1], limit)
        elif op is _C.BRANCH:
            alts = [s for b in av[1] for s in (_wordings(b, limit) or [None])]
        elif op in (_C.MAX_REPEAT, _C.MIN_REPEAT):
            lo, hi, sub = av
            inner = _wordings(sub, limit)
            if inner is None:
                return None
            if hi <= 1:
                alts = ([''] if lo == 0 else []) + inner
            elif all(len(s) == 1 for s in inner):
                alts = [_WILD]          # a run of any length of characters ("\w+") reads as one wild character
            else:
                return None
        else:
            return None
        if alts is None or None in alts:
            return None
        out = [a + b for a in out for b in dict.fromkeys(alts)]
        if len(out) > limit:
            return None
    return out


def _vocabulary(patterns):
    """The pieces of letters the patterns can match: (every three letters in a run of letters, every whole run of one
    or two letters), or None when some pattern cannot be spelled out or puts a wild character beside a letter."""
    tri, short = set(), set()
    for r in patterns:
        strings = _wordings(_parser.parse(r.pattern, r.flags))
        if strings is None or any(re.search(r"[^\W\d_]" + _WILD + "|" + _WILD + r"[^\W\d_]", x) for x in strings):
            return None
        for x in strings:
            for run in re.findall(r"[^\W\d_]+", x.lower()):
                tri.update(run[i:i + 3] for i in range(len(run) - 2))
                if len(run) <= 2:
                    short.add(run)
    return tri, short


def _may_hold(vocab, word, i, j) -> bool:
    """Whether a family's word can cover one of letters i to j of word, a run of letters at least three long. The
    letters of word that a match covers are a run of letters the pattern matches, or part of one, so some three of
    them around those letters are three letters of such a run, or they are a whole run of two letters, or of one at
    an edge of word. Without a vocabulary (a pattern this cannot spell out) every word may."""
    if vocab is None or not word.isascii():
        return True
    tri, short = vocab
    low = word.lower()
    return (any(low[x:x + 3] in tri for x in range(max(0, i - 2), min(j + 1, len(low) - 2)))
            or any(low[x:x + 2] in short for x in range(max(0, i - 1), min(j + 1, len(low) - 1)))
            or (i == 0 and low[0] in short) or (j == len(low) - 1 and low[-1] in short))


def _overlaps(patterns, pre, word, post) -> bool:
    """Whether a match of one of the patterns in pre + word + post overlaps word."""
    text, a, b = pre + word + post, len(pre), len(pre) + len(word)
    for r in patterns:
        for x in r.finditer(text):
            if x.start() >= b:
                break
            if x.end() > a:
                return True
    return False


class Families:
    def __init__(self, path: Path | None = None, tag_names_path: Path | None = None):
        spec = json.loads(Path(path or HERE / "families.json").read_text())
        tn_path = Path(tag_names_path or HERE / "tag_names.json")
        tn = json.loads(tn_path.read_text()) if tn_path.exists() else {}
        self.plain = {k: {t: v.get("name") for t, v in d.items()} for k, d in tn.items() if not k.startswith("_")}
        self.kinds = {t: v.get("kind") for t, v in tn.get("operator_mistakes", {}).items()}
        self.defs = spec["families"]
        self._re = {f["slug"]: re.compile(f["text"], re.I) for f in self.defs if f.get("text")}
        self._spell_cache = {}
        self._vocab = _vocabulary(self._re.values())

    def tag_name(self, key: str, tag: str | None) -> str:
        """The tag's plain name (tag_names.json), else its words with the first letter capitalised."""
        n = self.plain.get(key, {}).get(tag or "")
        if n:
            return n
        w = (tag or "untagged").replace("_", " ")
        return w[:1].upper() + w[1:]

    def catalog(self) -> dict:
        """The listed families, in order, for a page: slug -> name, list, and whether only a check raises it."""
        return {f["slug"]: {"name": f["name"], "list": f["list"],
                            "check": bool(f.get("checks") or f.get("reader_issues"))
                            and not (f.get("names") or f.get("tags") or f.get("text"))}
                for f in self.defs}

    def reader_family(self, kind: str) -> str:
        """The family a reader issue of this kind raises: the listed family that names the kind, else "d:" and the
        kind's words."""
        for f in self.defs:
            if kind in (f.get("reader_issues") or []):
                return f["slug"]
        w = str(kind or "reader issue").replace("_", " ")
        return "d:" + w[:1].upper() + w[1:]

    def list_of(self, slug: str) -> str:
        """The list a family is on: a listed family's own, else data for "d:" and mistake for "m:"."""
        f = next((f for f in self.defs if f["slug"] == slug), None)
        return f["list"] if f else "mistake" if str(slug).startswith("m:") else "data"

    @staticmethod
    def rated(i: dict) -> bool:
        return _sev(i) is not None

    def counts(self, key: str, i: dict) -> bool:
        """Whether an issue of this list ("data_issues" or "operator_mistakes") counts (the module docstring)."""
        s = _sev(i)
        if key == "data_issues":
            return s in ("medium", "high")
        return s == "high" or (s == "medium" and self.kinds.get(i.get("category")) == "outcome")

    def family_of(self, key: str, i: dict, dataset: str | None = None) -> str:
        lst = LISTS[key]
        cat = i.get("category") or ""
        plain = self.tag_name(key, cat)
        eligible = [f for f in self.defs if f['list'] == lst
                    and (not f.get('datasets') or dataset in f['datasets'])]
        # Structured categories outrank incidental words anywhere in the prose.
        for f in eligible:
            if cat in (f.get("tags") or []) or plain in (f.get("names") or []):
                return f["slug"]
        # Words refine only a tag that names no one problem: one the model files several problems under (a family's
        # "among") or one it coined (no entry in tag_names.json). A known tag keeps its own name.
        if not (self._broad(plain) or cat not in self.plain.get(key, {})):
            return ("d:" if lst == "data" else "m:") + plain
        # The issue and its evidence are read as one note, each sentence of it on its own grammar (board/polarity.py):
        # nothing in one sentence retracts another. The issue says what the problem is; its evidence decides only
        # when the issue asserts none of the families.
        issue, evidence = (self._spelled(str(i.get(field) or '')) for field in ('issue', 'evidence'))
        text = issue + '\n' + evidence if evidence else issue
        spans = denied_spans(text)
        claimed = {}
        for f in eligible:
            if f["slug"] in self._re and (not f.get('among') or plain in f['among']):
                start = next((m.start() for m in self._re[f["slug"]].finditer(text)
                              if not any(a <= m.start() < b for a, b in spans)), None)
                if start is not None:
                    claimed[f["slug"]] = start
        for in_issue in (True, False):
            for f in eligible:
                if f["slug"] in claimed and (claimed[f["slug"]] <= len(issue)) == in_issue:
                    return f["slug"]
        return ("d:" if lst == "data" else "m:") + plain

    def _spelled(self, text: str) -> str:
        """text with each word that two swapped letters keep from being a family's word put right ("dorpout"). A
        swap keeps every character offset, so the clause reading is unchanged."""
        if text in self._spell_cache:
            return self._spell_cache[text]
        out = list(text)
        n = len(out)
        patterns = list(self._re.values())
        for m in re.finditer(r"[^\W\d_]{4,}", text):
            a, b = m.span()
            w = m.group()
            # no family word is that long, and trying every swap in it would cost its length squared
            if len(w) > _SPELL_REACH:
                continue
            # A family word that overlaps this word lies within _SPELL_REACH of it, so only that stretch of the text
            # is read; reading the whole text for every word and every swap made a long note take minutes. The
            # stretch ends at word edges, so a "\b" in a family's pattern reads as it would in the whole text.
            lo, hi = max(0, a - _SPELL_REACH), min(n, b + _SPELL_REACH)
            while lo < a and lo > 0 and _WORD_CHAR.match(out[lo - 1]):
                lo += 1
            while hi > b and hi < n and _WORD_CHAR.match(out[hi]):
                hi -= 1
            pre, post = ''.join(out[lo:a]), ''.join(out[b:hi])
            if _may_hold(self._vocab, w, 0, len(w) - 1) and _overlaps(patterns, pre, w, post):
                continue
            for k in range(len(w) - 1):
                if w[k] == w[k + 1]:
                    continue
                cand = w[:k] + w[k + 1] + w[k] + w[k + 2:]
                # a family's word the swap brings in covers one of the two letters it moved
                if _may_hold(self._vocab, cand, k, k + 1) and _overlaps(patterns, pre, cand, post):
                    out[a:b] = cand
                    break
        out = ''.join(out)
        if len(self._spell_cache) > 4096:
            self._spell_cache.clear()
        self._spell_cache[text] = out
        return out

    def _broad(self, plain: str) -> bool:
        return any(plain in (f.get('among') or []) for f in self.defs)

    def classify(self, d: dict) -> dict:
        """The episode's families: counted ones with the issues that count under each (empty when a check or the
        outcome raised it), minor and unrated ones with no counted issue, and not_counted, the families of reader issues that are
        no fault in the recording (a list outside COUNTED_LISTS), shown on the episode and never counted."""
        counted, minor, unrated = (collections.defaultdict(list) for _ in range(3))
        not_counted = {}
        ds = d.get("dataset")
        for key in LISTS:
            for i in d.get(key) or []:
                if i and i.get("issue"):
                    group = counted if self.counts(key, i) else minor if self.rated(i) else unrated
                    group[self.family_of(key, i, ds)].append(i)
        # the episode's outcome, and each part's of a long recording labelled in parts (label/pieces.py puts those
        # under tasks), so a part reached and then undone counts as a short episode's does
        outcomes = {str((d.get("completion") or {}).get("task_completed") or "").lower()}
        if not tasks_format(d):
            outcomes |= {str(t.get("outcome") or "").lower() for t in d.get("tasks") or [] if isinstance(t, dict)}
        elif outcomes == {""}:
            # in the tasks format: the recording's verdict, else only the tasks judged against the instruction; an
            # unasked side task never stands for the recording
            outcomes |= {str(t.get("outcome") or "").lower() for t in d.get("tasks") or []
                         if isinstance(t, dict) and t.get("judged_against") == "instruction"}
        outcomes.discard("")
        for f in self.defs:
            if f.get("datasets") and ds not in f["datasets"]:
                continue
            if (any(check_hit(d, k) for k in f.get("checks") or [])
                    or outcomes & set(f.get("completion") or [])):
                counted.setdefault(f["slug"], [])
        for x in (d.get("dataset_checks") or {}).get("reader_issues") or []:
            if isinstance(x, dict) and x.get("kind"):
                fam = self.reader_family(str(x["kind"]))
                (counted if self.list_of(fam) in COUNTED_LISTS else not_counted).setdefault(fam, [])
        return {"counted": dict(counted), "minor": {k: v for k, v in minor.items() if k not in counted},
                "unrated": {k: v for k, v in unrated.items() if k not in counted},
                "not_counted": {k: v for k, v in not_counted.items() if k not in counted}}

    def hands_hidden_seconds(self, d: dict) -> float | None:
        """For head-camera footage, the seconds in which the wearer's hands are out of view (the dense timeline marks
        each segment); None for any other rig, where the question does not apply."""
        if d.get("_rig") != "ego_head":
            return None
        return union_seconds([(float(e["t_s"]), float(e["end_s"])) for e in d.get("event_labels") or []
                              if e.get("hands_visible") is False and e.get("t_s") is not None
                              and e.get("end_s") is not None])
