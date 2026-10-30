"""Labels without wording about how they were made.

A long recording is labelled in parts, one request each, and the model writes about the part it was given: "the window
begins with the cloth already folded", "still underway at the request boundary", "release inferred between samples",
"the left-stream gripper", "Part 2: ...". A reader of the labels sees one recording and none of those parts, so such
wording is about the labelling, not the recording. normalize_labels rewrites it, deterministically, after the parts are
stitched (label/pieces.py stitch_run), and records VERSION in the result (text_normalizer), so a reader can tell
normalized labels from earlier ones.

The rules, in order:
  - a camera called a stream is named as the view or the gripper it is ("the left-stream gripper" -> "the left
    gripper", "the exo stream" -> "the exo view"); a "Part 2:" or "(part 3)" marker goes; "most of the window is
    spent" reads "most of the recording is spent"
  - a clause (a sentence, or a part of one between semicolons) whose subject or predicate is the labelling goes whole:
    "The window begins with the lid open.", "The pour is still underway at the request boundary.", "No instruction is
    supplied.", "Release occurs between samples.", "The sampled views do not resolve every extraction."
  - a phrase that only places a claim in the labelling ("in this window", "between samples", "at the request
    boundary", "in the supplied frames") is cut and the claim kept ("The cards shift between sampled frames." -> "The
    cards shift."); a clause left with nothing a reader needs ("Grasp inferred between samples.") goes
  - a name, an object, a place or another short field keeps its words with the phrase cut out, and is never dropped

The word window is the labelling's only where its context says so: a qualifier (this, supplied, request, preparation),
a temporal preposition (during, throughout, until, beyond), an edge (the window begins, the window's end, the end of the
window) or a task still running at it ("still underway at the window end"). A physical window keeps every word: "at the
window end of the table", "the window sill", "near the window", "open the window". "the window" with no such context is
left as it is. A bare "part" or "clip" is a physical one (a toy's part, a binder clip) unless a scope word says
otherwise ("outside this part", "not shown in this part"), and bare "samples" can be fabric samples, so they count only
between samples, in these samples, or as what a time was inferred from."""
from __future__ import annotations

import re

VERSION = 'argus-wording-1'

I = re.I
# words that make "window" the labelling's
_QUAL = (r'(?:supplied|provided|given|request|requested|labell?ed|labell?ing|annotation|annotated|current|sampled|'
         r'analy[sz]ed|observed|preparation|prepared|time|clip|part|recording)')
# what follows a physical window ("the window end of the table", "the window sill"): "window" before it is physical
_PHYS_AFTER = (r'(?!\s*(?:-\s*)?(?:sills?|panes?|ledges?|glass|blinds?|shades?|curtains?|seat|box|boxes|latch|handle|'
               r'cleaner|wiper|frame\b(?!s?\s+(?:of|in)\s+(?:the|this)\s+(?:part|clip|request))|screen|bay|light|'
               r'(?:end|side|edge|corner)\s+of)\b)')
_WIN = r'window' + _PHYS_AFTER
_EDGE = r'(?:start|end|beginning|boundary|boundaries|edge|edges|opening|close|closing|cut)'
_PART_EDGE = r'(?:preparation\s+)?(?:boundary|boundaries|cut)'   # a toy's part has an end and an edge, not these
_DET = r'(?:the|this|that|each|its|the\s+current)'
_ENDS = r'(?:begins|began|starts|started|ends(?!\s+up\b)|ended|cuts\s+off|stops|finishes|is\s+cut)'  # "ends up" is a place
_OPENS = r'(?:' + _ENDS[3:-1] + r'|opens|opened|closes|closed)'   # a named unit, never a bare window, also opens

# a labelling unit, named: "this window", "the supplied window", "the request", "this supplied part", "this part of the
# recording", "the supplied clip"
UNIT = (r'(?:' + _DET + r'\s+(?:' + _QUAL + r'[\s-]+)+' + _WIN +
        r'|(?:this|that|each|the\s+current)\s+' + _WIN +
        r'|' + _DET + r'\s+(?:' + _QUAL + r'\s+)*request\b(?!\s+(?:of|for|to)\b)' +
        r'|(?:this|that|the\s+current)\s+(?:segment|excerpt)\b(?!\s+of\s+(?:the|a)\s+(?!recording|session|footage|video))'
        r'|(?:this|that|the\s+current)\s+part\s+of\s+the\s+(?:\w+\s+)?(?:recording|session|footage|video)\b'
        r'|(?:the|this|that)\s+(?:supplied|provided|given|labell?ed|requested)\s+(?:part|clip|segment|excerpt|footage|'
        r'video|frames?|views?|images?|recording)\b)')
# a stretch of samples: "between samples", "between these sampled frames", "within the sampled interval"
_SDET = (r'(?:(?:the|these|those|this|that|two|first|last|final|same|adjacent|consecutive|neighbouring|neighboring|'
         r'available|supplied|successive|displayed|one|one-second|[\w-]+-facing)\s+)*')
SAMPLES = (r'(?:between\s+' + _SDET + r'samples\b'
           r'|(?:between|within|in|inside|during|across|by|from)\s+' + _SDET + r'(?:sampled\s+(?:frames?|instants?|'
           r'moments?|times?|images?|views?|stills?|interval|intervals|footage)|sampling\s+interval|sample\s+interval)\b'
           r'|(?:in|by|from|within|across)\s+(?:these|those|the\s+adjacent|adjacent|the\s+consecutive|consecutive|'
           r'the\s+displayed)\s+samples\b'
           r'|(?:across|from)\s+the\s+samples\b|in\s+(?:every|each|many|several|some|all)\s+sampled\s+\w+'
           r'|between\s+(?:the\s+)?supplied\s+(?:frames|samples|images|views))')
_TIMING = (r'(?:occurs?|occurred|falls?|fell|happens?|happened|takes?\s+place|took\s+place|lies?|is|are|was|were|'
           r'(?:(?:is|are|was|were)\s+)?(?:inferred|interpolated|estimated))')

# clauses about the labelling: dropped from prose, cut from names
# the unit's edges as an event: "the window begins with", "the supplied part ends before", "Window ends with"
_EDGE_AS_EVENT = (r'\b(?:' + UNIT + r')\s+' + _OPENS + r'\b|\bthe\s+' + _WIN + r'\s+' + _ENDS + r'\b|^\s*window\s+' +
                  _OPENS + r'\b')
# a task still running at an edge: "is still underway at the request boundary", "continues at the preparation cut"
_AT_EDGE = (r'\b(?:(?:is|are|was|were|remains?|remained)\s+)?(?:still\s+|already\s+)?(?:under\s*way|in\s+progress|ongoing|'
            r'continuing|continues|continue|active|being\s+worked(?:\s+on)?|held|unfinished|incomplete|not\s+finished|'
            r'not\s+complete|mid[- ]?(?:task|action|activity))\s+(?:at|by|when|as|until|through|across|before)\s+(?:the\s+|this\s+|'
            r'both\s+)?(?:(?:' + _QUAL + r'|internal)[\s-]+)*(?:(?:' + _WIN + r'|request)(?:\'s)?(?:\s+' + _EDGE +
            r')?|(?:part|clip)(?:\'s)?\s+' + _PART_EDGE + r'|cut|boundary|boundaries)\b')
EDGE_EVENT = re.compile(_EDGE_AS_EVENT + '|' + _AT_EDGE, I)
ABOUT = re.compile('|'.join([
    _EDGE_AS_EVENT,
    _AT_EDGE,
    # what the unit shows or lacks: "this window does not establish", "the supplied frames only show"
    r'\b(?:' + UNIT + r')\s+(?:does\s+not|do\s+not|doesn\'t|don\'t|cannot|can\'t|never|only|shows?|reveals?|'
    r'establish(?:es)?|contains?|covers?|includes?|is\s+spent|continues|follows|picks\s+up)\b',
    r'\b(?:lies?|lie|is|are|was|were|remains?|remained)\s+outside\s+the\s+' + _WIN + r'\b',
    r'\b(?:this|the)\s+(?:internal\s+)?(?:preparation|request|labell?ing)\s+cut\s+(?:is|was|does|did|shows?)\b',
    r'\b(?:is|was)\s+an?\s+(?:internal\s+)?(?:preparation|request)\s+cut\b|\boutside\s+the\s+supplied\s+evidence\b',
    r'\b(?:begins?|began|starts?|started|continues?|continued|extends?|runs?|carries|carry|goes|go|lies?|lie|falls?|'
    r'happens?|occurs?|is|are|was|were|remains?)\s+(?:on\s+)?(?:before|after|beyond|past|outside|into|across|'
    r'over)\s+(?:' + UNIT + r'|the\s+request\s+' + _EDGE + r'|the\s+part\s+' + _PART_EDGE + r')',
    r'\b(?:predat|preced|postdat)\w*\s+(?:' + UNIT + r')',
    r'\b(?:outside|beyond)\s+(?:this|the\s+supplied|this\s+supplied)\s+part\b',
    r'\b(?:continues?|continuing|extends?|carries|runs?)\s+(?:on\s+)?into\s+the\s+next\s+(?:supplied\s+)?'
    r'(?:part|request|window|clip)\b',
    r'\brequest[\s-]+(?:windows?|boundar(?:y|ies)|cuts?|edges?|starts?|ends?)\b',
    # the input, as the model got it: "no instruction is supplied", "the sampled views do not resolve"
    r'\b(?:no|an?)\s+(?:explicit\s+|written\s+|task\s+|source\s+)*(?:instruction|description|note|dataset\s+note|task)s?\s+'
    r'(?:is|was|are|were)\s+(?:supplied|provided|given)\b',
    r'\b(?:is|are|was|were)\s+not\s+(?:supplied|provided|given)\b|\bno\s+supplied\b',
    r'\b(?:the|these|this|those|available|adjacent|sparse)\s+(?:sampled|supplied)\s+(?:views?|frames?|images?|instants?|'
    r'samples?|footage|stills?|interval)\s+(?:do|does|did|show|shows|suggest|cannot|can\'t|never|only|are|is|leave|'
    r'leaves|resolve|resolves|make|makes|bound|bounds|establish|establishes|settle|settles)\b',
    r'\b(?:the|this)\s+(?:last|first|final)\s+(?:supplied|sampled)\s+(?:frames?|views?|images?|samples?|instants?)\b',
    r'\b(?:the\s+)?samples?\s+(?:do|does|did|cannot|can\'t|never)\s+(?:not\s+)?(?:show|resolve|establish|reveal|capture|'
    r'settle|distinguish)\b',
    r'\b(?:the|these)\s+samples\s+(?:show|shows|establish|establishes|suggest|suggests|bound|bounds|do\s+not\s+support)\b',
    r'\b(?:is|are|was|were)\s+not\s+(?:fully\s+|densely\s+)?sampled\b|\bsparsely\s+sampled\b',
    # a time placed only between samples: "release occurs between samples", "grasp inferred within the sampled interval"
    r'\b' + _TIMING + r'\s+' + SAMPLES,
    r'\b(?:inferred|interpolated|estimated|bounded)\s+(?:by|between|within|from|in)\s+(?:the\s+)?(?:samples?|sampled\s+\w+)\b',
]), I)
ABOUT_CUT = re.compile(r'\s*,?\s*(?:' + ABOUT.pattern + r')', I)     # the same, with its comma, for cutting from a name

# phrases that only place a claim in the labelling: cut out, the claim kept
CUT = re.compile(r'\s*,?\s*(?:' + '|'.join([
    SAMPLES,
    r'at\s+(?:sample|(?:the\s+)?(?:next|following|previous|last|first)\s+sample)\b',
    r'(?:at|by|with)\s+(?:this\s+|the\s+)?(?:supplied\s+)?(?:(?:one|1|two|2|half)[- ]?(?:second|s)\s+)?sampl(?:ing|e)\s+'
    r'(?:rate|interval)\b',
    r'(?:at|by|with)\s+(?:this\s+|the\s+)?(?:one|1|two|2|half)[- ]?(?:second|s)\s+sampling\b',
    r'in\s+the\s+(?:following|next|previous|last|first|later|earlier)\s+(?:sample|sampled\s+\w+)\b',
    # "in this window", "during the supplied window", "throughout most of the window", "by the end of this request"
    r'(?:in|within|during|throughout|through|across|over|for|from|until|till|by|at|before|after|beyond|past|outside)\s+'
    r'(?:(?:all|most|much|some|part|half|the\s+rest|the\s+remainder|the\s+(?:start|end|beginning|middle|close|first\s+'
    r'part|last\s+part|later\s+part|earlier\s+part|latter\s+part|first\s+half|second\s+half))\s+of\s+)?' + UNIT +
    r'(?:\'s)?(?:\s+' + _EDGE + r'\b(?!\s+of\b))?',
    # "throughout the window", "until the window ends": bare "the window" after a temporal preposition ("across the
    # window" and "before the window" can be places, so they are not among them)
    r'(?:during|throughout|within|until|till|beyond|past)\s+(?:(?:all|most|much|the\s+rest|the\s+'
    r'(?:start|end|beginning))\s+of\s+)?the\s+' + _WIN + r'(?:\'s)?(?:\s+' + _EDGE + r'\b(?!\s+of\b))?',
    r'(?:at|by|near|toward|towards)\s+the\s+(?:start|end|beginning|close)\s+of\s+the\s+' + _WIN + r'\b',
    r'(?:early|earlier|late|later)\s+in\s+(?:the|this)\s+(?:(?:supplied|labell?ed|current)\s+)?' + _WIN + r'\b',
    r'(?:at|by|near|toward|towards|through|throughout|during|for)\s+(?:the\s+)?(?:start|end|beginning|close|rest|remainder|'
    r'most|latter\s+part|later\s+part|first\s+part|last\s+part)\s+of\s+(?:the|this)\s+(?:(?:supplied|current|labell?ed|'
    r'preparation)\s+)?' + _WIN + r'\b',
    r'(?:at|by|in|on)\s+the\s+window\'s\s+(?:first|last|final|opening|closing)\s+frame\b',
    r'(?:before|by|in|at|until)\s+the\s+next\s+(?:sample|sampled\s+(?:frame|view|image))\b',
    r'for\s+(?:most|much|all|part|half)\s+of\s+the\s+' + _WIN + r'\b',
    r'inside\s+(?:this|the\s+supplied)\s+' + _WIN + r'\b',
    r'by\s+the\s+(?:next|following)\s+(?:sample|sampled\s+(?:frame|view|image))\b',
    r'(?:in|into|of|from)\s+the\s+(?:next|previous|prior|following|preceding)\s+supplied\s+(?:part|clip|window|segment)\b',
    # an edge: "at the request boundary", "at the window end" (not "at the window end of the table"), "at this part's
    # preparation boundary" (not "at the part's end", a toy's)
    r'(?:at|by|near|toward|towards|until|till|up\s+to|past|across|around|before|after|over|through|spanning|'
    r'crossing)\s+(?:the\s+|this\s+|'
    r'that\s+|each\s+|both\s+)?(?:(?:' + _QUAL + r'|internal)[\s-]+)*(?:(?:window|request)(?:\'s)?\s+' + _EDGE +
    r'|(?:part|clip)(?:\'s)?\s+' + _PART_EDGE + r')\b(?!\s+of\b)',
    r'(?:at|across|around|past|near|over|before|after|beyond|until|till)\s+(?:the|this|that|an?)\s+(?:internal\s+)?(?:request|part|'
    r'labell?ing|preparation)\s+cut\b',
    r'(?:in|from|across|within|throughout)\s+(?:all\s+|the\s+|these\s+)?(?:supplied|provided|given|sampled)\s+(?:frames?|'
    r'views?|images?|footage|clip|video|stills?|instants?|samples)\b',
    r'(?:in|of|from)\s+(?:the\s+)?(?:previous|prior|next|earlier|later|following|preceding)\s+(?:request|' + _WIN + r')\b',
]) + r')', I)

# a claim scoped to the part or the samples keeps its predicate: "not shown in this part" -> "not shown"
SCOPED = re.compile(r'\b((?:resolved|visible|shown|seen|established|confirmed|identified|captured|separable|'
                    r'distinguishable|evident|reached|completed|finished|observed|supplied|clear|settled|isolated|'
                    r'verifiable|distinguished|resolvable))\s+(?:with)?(?:in|for|by|from)\s+(?:this|the|these|those)\s+'
                    r'(?:supplied\s+)?(?:part|samples)\b', I)
# rewrites that keep the sentence
_QTY = (r'(?:most|much|some|all|part|half|the\s+rest|the\s+remainder|the\s+middle|(?:a\s+)?(?:substantial\s+|large\s+|'
        r'small\s+)?(?:portion|share|fraction))')
REWRITE = [
    (re.compile(r"\b(left|right)[- ]stream(?:'s)?\s+(grippers?|fingers?|jaws?|pads?|tips?)\b", I), r'\1 \2'),
    (re.compile(r"\b(left|right|exo|top|scene)[- ]streams?(?:'s)?\b", I), r'\1 view'),
    (re.compile(r"\b(?:wrist|wrist-camera|gripper|close-view|handheld)\s+streams\b", I), 'wrist views'),
    (re.compile(r"\b(?:wrist|wrist-camera|gripper|close-view|handheld)\s+stream\b", I), 'wrist view'),
    (re.compile(r"\bstream's\s+(fingers|jaws|pads|tips)\b", I), r"gripper's \1"),
    # "most of the window is spent" reads "most of the recording is spent"; a bare "the window" only where a verb
    # of time follows or precedes it, so "wipe most of the window" stays
    (re.compile(r'\b(' + _QTY + r')\s+of\s+(?:this|the\s+(?:supplied|labell?ed|current|preparation))\s+(remaining\s+)?'
                + _WIN, I), r'\1 of the \2recording'),
    (re.compile(r'\b(' + _QTY + r')\s+of\s+the\s+(remaining\s+)?' + _WIN + r'(?=\s+(?:is|was|consists|goes|involves|'
                r'shows|features|covers)\b)', I), r'\1 of the \2recording'),
    (re.compile(r'\b(spends?|spent|spending|occup(?:y|ies|ied|ying)|takes?|took|consum(?:e|es|ed|ing))\s+(' + _QTY +
                r')\s+of\s+the\s+(remaining\s+)?' + _WIN, I), r'\1 \2 of the \3recording'),
    (re.compile(r'\bsupplied\s+(views?|cameras?)\b', I), r'\1'),
]
# a part's marker: "Part 2: ..." at a sentence start, "(part 3)" after a time
PART_MARK = re.compile(r'(?:(?<=^)|(?<=[.;!?]\s)|(?<=\n))\s*Part\s+\d+\s*[:.\-]\s*|\s*\((?:request\s+)?part\s+\d+\)', I)
SENTENCE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9("\'])')
CLAUSE = re.compile(r'(?<=;)\s+')
# a clause with nothing a reader needs once its process phrase is gone: "Release", "The grasp timing is inferred"
_BARE = re.compile(r'^(?:the\s+)?(?:(?:exact|brief|final|small|fine|right|left|precise)\s+)*(?:\w+\s+){0,2}?'
                   r'(?:release|grasp|pickup|pick-up|placement|closure|contact|handoff|transfer|timing|instant|moment|'
                   r'onset|time)?\s*(?:timing|instants?|moment|time)?\s*(?:is|are|was|were)?\s*(?:only\s+)?(?:inferred|'
                   r'interpolated|estimated|approximate|bounded|uncertain)?\s*[.;]?$', I)

# keys whose values are enums or ids, never prose: left as they are
ENUM_KEYS = frozenset({'arm', 'contribution', 'kind', 'outcome', 'category', 'severity', 'deliberate', 'basis', 'list',
                       'excluded_by', 'by', 'side', 'task_completed', 'view', 'content', 'fingerprint', 'signals',
                       'progress_scope', 'id', 'sha256', 'status'})
# subtrees Argus writes itself about its stitching: left as they are
OWN_KEYS = frozenset({'cut_joins', 'grouped', 'stitched', 'windows'})
# a field Argus writes itself inside an item it set aside or marked at a cut ("both parts labelled these seconds
# around the cut ..."): left as it is
OWN_FIELDS = frozenset({('_excluded', 'reason'), ('_dropped', 'reason'), ('cut_continuation', 'reason')})
# short fields: a phrase is cut from them, never the whole value
NAME_KEYS = frozenset({'action', 'object', 'objects', 'destination', 'spatial_relation', 'relations', 'name',
                       'attributes', 'location', 'task', 'success_predicate', 'from', 'to', 'label', 'task_summary',
                       'instruction_variants', 'setting', 'how_to_fix', 'failure', 'correction'})


def _tidy(s: str) -> str:
    s = re.sub(r'\s{2,}', ' ', s)
    s = re.sub(r'\s+([.;,:!?)])', r'\1', s)
    s = re.sub(r'\(\s*\)', '', s)
    s = re.sub(r'([,;:])\s*([.;!?])', r'\2', s)
    s = re.sub(r'^[\s,;:]+', '', s)
    return s.strip()


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s and s[0].islower() else s


def _rewritten(s: str) -> str:
    y = PART_MARK.sub(' ', s) if PART_MARK.search(s) else s
    for rx, to in REWRITE:
        y = rx.sub(to, y)
    y = SCOPED.sub(r'\1', y)
    return _tidy(y) if y != s else s


def mentions_labelling(s: str) -> bool:
    """Whether a text still holds wording about the labelling (a clause about it, or a phrase placing a claim in it)."""
    return bool(ABOUT.search(s) or CUT.search(s))


def _clause(c: str) -> str | None:
    """One clause, without its closing punctuation, without its process wording: None when it is about the
    labelling, or nothing a reader needs is left once the phrase placing it in the labelling is cut."""
    if ABOUT.search(c):
        return None
    if not CUT.search(c):
        return c
    y = _tidy(CUT.sub('', c)).rstrip(' ,')
    if not y or ABOUT.search(y) or _BARE.match(y) or len(re.findall(r'[A-Za-z]{2,}', y)) < 3:
        return None
    return y


def _prose(y: str) -> str | None:
    out = []
    for sentence in SENTENCE.split(y):
        s = sentence.strip()
        end = s[-1] if s[-1:] in '.!?' else ''
        body = s[:-1] if end else s
        kept = []
        for c in CLAUSE.split(body):
            c = c.rstrip()
            semi = c.endswith(';')
            k = _clause(c[:-1].rstrip() if semi else c) if c.strip(' ;') else None
            if k:
                kept.append(k)
        if kept:
            out.append(_cap('; '.join(kept)) + end)
    return ' '.join(out) or None


def normalize_text(s: str, name: bool = False) -> str | None:
    """s without wording about the labelling. A name (name=True) keeps its words with the phrase cut out, or comes back
    unchanged when nothing would be left; prose loses each clause about the labelling and is None when none is left."""
    if not isinstance(s, str) or not s.strip():
        return s
    y = _rewritten(s)
    if not mentions_labelling(y):
        return _late(y)
    if name:
        z = _tidy(CUT.sub('', ABOUT_CUT.sub('', y))).strip(' ,;:')
        return _late(z) if re.search(r'[A-Za-z]{2,}', z) else y
    z = _prose(y)
    return _late(z) if z else z


# after the clauses are judged (they read "supplied frames" as the labelling's): the frames a reader sees are frames
LATE = re.compile(r'\bsupplied\s+(?:samples|images|frames)\b|\b(?:sampled\s+)?supplied\s+(footage|video)\b', I)


def _late(s: str) -> str:
    return LATE.sub(lambda m: m.group(1) or 'frames', s)


def edge_event(label) -> bool:
    """Whether a key event's label is a part's edge, not an event of the recording: "Window ends with the border
    incomplete", "Knot work still underway at the request boundary"."""
    return isinstance(label, str) and bool(EDGE_EVENT.search(label))


def normalize_labels(labels, counts: dict | None = None, log: list | None = None):
    """A copy of a result's labels (any depth) with every text normalized (normalize_text): a short field as a name,
    every other as prose; enum fields and Argus's own stitch records are left as they are. A key event that is a
    part's edge (edge_event) leaves its list. A prose field left with no
    clause becomes None, and a list entry left empty leaves the list. counts, when given, gets 'rewritten' and
    'dropped' added up; log, a list, gets (field, before, after) for every text changed."""
    counts = {} if counts is None else counts

    def keep(parent, k):
        return k in OWN_KEYS or k in ENUM_KEYS or (parent, k) in OWN_FIELDS

    def walk(v, key):
        if isinstance(v, dict):
            return {k: (x if keep(key, k) else walk(x, k)) for k, x in v.items()}
        if isinstance(v, list):
            out = []
            for x in v:
                if key == 'key_events' and isinstance(x, dict) and edge_event(x.get('label')):
                    counts['dropped'] = counts.get('dropped', 0) + 1
                    if log is not None:
                        log.append(('key_events', x.get('label'), None))
                    continue
                y = walk(x, key)
                if isinstance(x, str) and y is None:
                    continue
                out.append(y)
            return out
        if not isinstance(v, str):
            return v
        y = normalize_text(v, name=key in NAME_KEYS)
        if y != v:
            k = 'dropped' if y is None else 'rewritten'
            counts[k] = counts.get(k, 0) + 1
            if log is not None:
                log.append((key, v, y))
        return y

    return walk(labels, '')


def normalize_result(result: dict) -> dict:
    """The result with its labels normalized in place and text_normalizer recording VERSION and what changed."""
    counts = {'rewritten': 0, 'dropped': 0}
    if isinstance(result.get('labels'), dict):
        result['labels'] = normalize_labels(result['labels'], counts)
    result['text_normalizer'] = {'version': VERSION, **counts}
    return result
