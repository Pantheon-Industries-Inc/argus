"""Labels without wording about how they were made.

A long recording is labelled in parts, one request each, and the model writes about the part it was given: "the window
begins with the cloth already folded", "still underway at the request boundary", "release inferred between samples",
"the left-stream gripper", "Part 2: ...". A reader of the labels sees one recording and none of those parts, so such
wording is about the labelling, not the recording. normalize_labels rewrites it, deterministically, after the parts are
stitched (label/pieces.py stitch_run), and records VERSION in the result (text_normalizer), so a reader can tell
normalized labels from earlier ones.

The rules under-cut: losing a physical fact (an object, a motion, a state, a count, a time, a failure) is worse than
leaving a phrase about the labelling in place, and when a rule cannot tell, the text stays as the model wrote it.
In order:
  - a camera called a stream is named as the view or the gripper it is ("the left-stream gripper" -> "the left
    gripper", "the exo stream" -> "the exo view"); a "(part 3)" after a time goes. In a recording labelled in one part
    (whole=True) a "Part 1:" marker goes and "most of the window is spent" reads "most of the recording is spent"; in a
    stitched one both stay, since "Part 2:" is what tells one part's claims from the next and the window is one part
  - a clause (a sentence, or a part of one between semicolons) about the labelling goes only when every word of it is
    the labelling's: "No instruction is supplied.", "Inferred between samples." A clause that only places named events
    between samples keeps the names: "Release occurs between samples." -> "Release.". A clause with anything else in it
    ("The window begins with the lid open.", "The pour is still underway at the request boundary.") stays, losing at
    most the samples phrase ("The fall is inferred between samples: the pencil ..." -> "The fall is inferred: the
    pencil ...")
  - a phrase that only places a claim in the labelling ("in this window", "between samples", "at the internal request
    boundary", "in the supplied frames") is cut and the claim kept ("The cards shift between sampled frames." -> "The
    cards shift."). A phrase that places it in time against a part ("before this window", "at the window end", "early
    in the window", "for most of the window") is never cut: at the recording's own edge that time is real
  - a name, an object, a place or another short field keeps its words with the phrase cut out, and is never dropped;
    a key event goes only when it is nothing but a part's edge ("Window ends")

The word window is the labelling's only where its context says so: a qualifier (this, supplied, request, preparation),
a temporal preposition (during, throughout, until, beyond), an edge (the window begins, the window's end, the end of the
window) or a task still running at it ("still underway at the window end"). A physical window keeps every word: "at the
window end of the table", "the window sill", "near the window", "open the window". "the window" with no such context is
left as it is. A bare "part" or "clip" is a physical one (a toy's part, a binder clip) unless a scope word says
otherwise ("outside this part", "not shown in this part"), and bare "samples" can be fabric samples, so they count only
between samples, in these samples, or as what a time was inferred from."""
from __future__ import annotations

import re

VERSION = 'argus-wording-2'

I = re.I
# words that make "window" the labelling's
_QUAL = (r'(?:supplied|provided|given|request|requested|labell?ed|labell?ing|annotation|annotated|current|sampled|'
         r'analy[sz]ed|observed|preparation|prepared|time|clip|part|recording)')
# what follows a physical window ("the window end of the table", "the window sill"): "window" before it is physical
_PHYS_AFTER = (r'(?!\s*(?:-\s*)?(?:sills?|panes?|ledges?|glass|blinds?|shades?|curtains?|seat|box|boxes|latch|handle|'
               r'cleaner|wiper|frame\b(?!s?\s+(?:of|in)\s+(?:the|this)\s+(?:part|clip|request))|screen|bay|light|'
               r'(?:end|side|edge|corner)\s+of)\b)')
_WIN = r'window' + _PHYS_AFTER
_ENDPHRASE = r'(?=\s*(?:[.,;:!?)]|$|(?:and|but|so|while|when|where|then)\b))'   # a bare window that ends its phrase (not "the window cleaning task")
_EDGE = r'(?:start|end|beginning|boundary|boundaries|edge|edges|opening|close|closing|cut)'
_WEDGE = r'(?:start|end|beginning|boundary|boundaries|cut)'     # a real window has an edge, an opening and a close
_PART_EDGE = r'(?:preparation\s+)?(?:boundary|boundaries|cut)'   # a toy's part has an end and an edge, not these
_QW = r'(?:(?:' + _QUAL + r'|internal)[\s-]+)'                    # a word that makes the window or the part the labelling's
# what follows an edge that is a physical one: "the window end of the table", "the part cut line"
_NOTPHYS = r'(?!\s+(?:of|line|lines|mark|marks|edge|edges|piece|pieces|end|ends|side|surface)\b)'
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
# a labelling unit with a word that makes it the labelling's: "from this window" can be a real window, "from the
# supplied window" cannot
UNIT_Q = UNIT.replace(r'|(?:this|that|each|the\s+current)\s+' + _WIN, '')
assert UNIT_Q != UNIT
# a stretch of samples: "between samples", "between these sampled frames", "within the sampled interval"
_SDET = (r'(?:(?:the|these|those|this|that|two|first|last|final|same|adjacent|consecutive|neighbouring|neighboring|'
         r'available|supplied|successive|displayed|one|one-second|[\w-]+-facing)\s+)*')
# bare "samples" only between samples of the footage: "between the two samples" can be fabric samples
SAMPLES = (r'(?:between\s+(?:(?:these|those)\s+|(?:the\s+)?(?:(?:first|last|final|adjacent|consecutive|successive|'
           r'neighbouring|neighboring|displayed|supplied|available|one-second|[\w-]+-facing)\s+)+(?:two\s+)?)?samples\b(?!\s+of\b)'
           r'|(?:between|within|in|inside|during|across|by|from)\s+' + _SDET + r'(?:sampled\s+(?:frames?|instants?|'
           r'moments?|times?|images?|views?|stills?|interval|intervals|footage)|sampling\s+interval|sample\s+interval)\b'
           r'|(?:in|by|from|within|across)\s+(?:these|those|the\s+adjacent|adjacent|the\s+consecutive|consecutive|'
           r'the\s+displayed)\s+samples\b'
           r'|(?:across|from)\s+the\s+samples\b|in\s+(?:every|each|all)\s+sampled\s+\w+'
           r'|between\s+(?:the\s+)?supplied\s+(?:frames|samples|images|views))')
# a time or a number after it belongs to the claim: "release occurs between samples at 3.2 s and 4.1 s" is two times
SAMPLES = r'(?:' + SAMPLES + r')(?!\s*(?:\d|(?:at|from|of|near|around)\s+(?:t\s*=\s*)?\d|t\s*=|\())'
_TIMING = (r'(?:occurs?|occurred|falls?|fell|happens?|happened|takes?\s+place|took\s+place|lies?|is|are|was|were|'
           r'(?:(?:is|are|was|were)\s+)?(?:inferred|interpolated|estimated))')

# clauses about the labelling: dropped from prose, cut from names
# the unit's edges as an event: "the window begins with", "the supplied part ends before", "Window ends with"
_EDGE_AS_EVENT = (r'\b(?:' + UNIT + r')\s+' + _OPENS + r'\b|\bthe\s+' + _WIN + r'\s+' + _ENDS + r'\b|^\s*window\s+(?:' +
                  _ENDS[3:-1] + r'|(?:opens|closes)\s+(?:during|while|mid|as|before|after)\b)')
# a task still running at an edge: "is still underway at the request boundary", "continues at the preparation cut"
_AT_EDGE = (r'\b(?:(?:is|are|was|were|remains?|remained)\s+)?(?:still\s+|already\s+)?(?:under\s*way|in\s+progress|ongoing|'
            r'continuing|continues|continue|active|being\s+worked(?:\s+on)?|held|unfinished|incomplete|not\s+finished|'
            r'not\s+complete|mid[- ]?(?:task|action|activity))\s+(?:at|by|when|as|until|through|across|before)\s+(?:the\s+|this\s+|'
            r'both\s+)?(?:' + _QW + r'+window(?:\'s)?(?:\s+' + _EDGE + r')?|' + _QW + r'*request(?:\'s)?(?:\s+' + _EDGE +
            r')?|window(?:\'s)?\s+' + _WEDGE + r'|' + _QW + r'*(?:part|clip)(?:\'s)?\s+' + _PART_EDGE +
            r'|cut|boundary|boundaries)\b' + _NOTPHYS)
EDGE_EVENT = re.compile(_EDGE_AS_EVENT + '|' + _AT_EDGE, I)
# about the labelling's parts and their edges
_ABOUT_EDGE = [
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
]
# about what the sampled input shows or lacks
_ABOUT_EVIDENCE = [
    # the input, as the model got it: "no instruction is supplied", "the sampled views do not resolve"
    # (what a person is given, "no task was given to the second operator", is not the labelling's)
    r'\bno\s+(?:explicit\s+|written\s+|task\s+|source\s+)*(?:instruction|description|note|dataset\s+note|task)s?\s+'
    r'(?:is|was|are|were)\s+(?:supplied|provided|given)\b(?!\s+to\b)',
    r'\b(?:instruction|target|intended|specification|requirement|reference|goal|description|note|count|length|'
    r'number|image)s?\b[^.;]{0,120}?\b(?:is|are|was|were)\s+not\s+(?:supplied|provided|given)\b(?!\s+to\b)'
    r'|\bno\s+supplied\b',
    r'\b(?:the|these|this|those|available|adjacent|sparse)\s+(?:sampled|supplied)\s+(?:views?|frames?|images?|instants?|'
    r'samples?|footage|stills?|interval)\s+(?:do|does|did|cannot|can\'t|never|leave|leaves|bound|bounds)\b',
    r'\b(?:the|this)\s+(?:last|first|final)\s+(?:supplied|sampled)\s+(?:frames?|views?|images?|samples?|instants?)\b',
    r'\b(?:the\s+)?samples?\s+(?:do|does|did|cannot|can\'t|never)\s+(?:not\s+)?(?:show|resolve|establish|reveal|capture|'
    r'settle|distinguish)\b',
    r'\bthese\s+samples\s+(?:show|shows|establish|establishes|suggest|suggests|bound|bounds)\b'
    r'|\b(?:the|these)\s+samples\s+do\s+not\s+support\b',
    r'\b(?:is|are|was|were)\s+not\s+(?:fully\s+|densely\s+)?sampled\b|\bsparsely\s+sampled\b',
    # a time placed only between samples: "release occurs between samples", "grasp inferred within the sampled interval"
    r'\b' + _TIMING + r'\s+' + SAMPLES,
    r'\b(?:inferred|interpolated|estimated|bounded)\s+(?:by|between|within|from|in)\s+(?:the\s+)?(?:samples?|sampled\s+\w+)\b',
]
ABOUT = re.compile('|'.join(_ABOUT_EDGE + _ABOUT_EVIDENCE), I)
ABOUT_EVIDENCE = re.compile('|'.join(_ABOUT_EVIDENCE), I)
ABOUT_CUT = re.compile(r'\s*,?\s*(?:' + ABOUT.pattern + r')', I)     # the same, with its comma, for cutting from a name

# phrases that only place a claim in the labelling: cut out, the claim kept
CUT = re.compile(r'\s*,?\s*(?:' + '|'.join([
    SAMPLES,
    r'at\s+(?:sample|(?:the\s+)?(?:next|following|previous|last|first)\s+sample)\b',
    r'(?:at|by|with)\s+(?:this\s+|the\s+)?(?:supplied\s+)?(?:(?:one|1|two|2|half)[- ]?(?:second|s)\s+)?sampl(?:ing|e)\s+'
    r'(?:rate|interval)\b',
    r'(?:at|by|with)\s+(?:this\s+|the\s+)?(?:one|1|two|2|half)[- ]?(?:second|s)\s+sampling\b',
    r'in\s+the\s+(?:following|next|previous|last|first|later|earlier)\s+(?:sample|sampled\s+\w+)\b',
    # "in this window", "during the supplied window", "throughout most of the window": the claim's scope, cut. A time
    # placed against the part ("before this window", "at the end of the window", "early in the window",
    # "beyond the preparation cut") is never cut: at the recording's own edge it is a real time, and cutting it turns
    # "folding occurred before this window" into "folding occurred" (audit of argus-wording-1, 2026-10-30)
    r'(?:in|within|during|throughout|through|across|over|for|from)\s+'
    r'(?:all\s+of\s+)?' + UNIT_Q + r"(?!'s|\s+" + _EDGE + r'\b)',
    # "this window" with no qualifier is the labelling's only after a preposition of scope: "light comes in through
    # that window" and "the cat jumps from this window" are real ones
    # and only where its phrase ends there: "Scrub the frame in this window corner." names a real corner
    r'(?:in|within|during|throughout)\s+(?:all\s+of\s+)?(?:this|that|each|the\s+current)\s+' + _WIN +
    r"(?!'s|\s+" + _EDGE + r'\b)(?=\s*(?:[.,;:!?)]|$|(?:and|but|so|or|while|when|where|then|as|before|after|until|'
    r'is|are|was|were|does|do|did|remains?|stays?|shows?|appears?|seems?|has|have|had)\b|\w+ed\b))',
    # bare "the window" after "during" or "throughout" is never cut: "Clean throughout the window." is a real one
    r'(?:before|by|in|at|until)\s+the\s+next\s+(?:sample|sampled\s+(?:frame|view|image))\b',
    r'inside\s+(?:this|the\s+supplied)\s+' + _WIN + r'\b',
    r'by\s+the\s+(?:next|following)\s+(?:sample|sampled\s+(?:frame|view|image))\b',
    r'(?:in|of|from)\s+the\s+(?:next|previous|prior|following|preceding)\s+supplied\s+(?:part|clip|window|segment)\b',
    # an edge between two parts, which is never the recording's: "at the internal request boundary", "at the
    # preparation cut", "across this part's preparation boundary". A window's or a request's start, end or boundary
    # can be the recording's own, so it stays
    r'(?:at|across|around|over|through|spanning|crossing|near)\s+(?:the\s+|this\s+|that\s+|each\s+|both\s+)?'
    r'(?:(?:internal|preparation)[\s-]+(?:\w+[\s-]+)?(?:window|request|part|clip)(?:\'s)?\s+(?:boundary|boundaries|cut)'
    r'|(?:' + _QW + r'+)?(?:part|clip)(?:\'s)?\s+(?:preparation\s+(?:boundary|boundaries)|cut))\b' + _NOTPHYS,
    r'(?:at|across|around|near|over)\s+(?:the|this|that|an?)\s+(?:internal\s+)?(?:request|part|'
    r'labell?ing|preparation)\s+cut\b' + _NOTPHYS,
    r'(?:in|from|across|within|throughout)\s+(?:all\s+|the\s+|these\s+)?(?:supplied|provided|given|sampled)\s+(?:frames?|'
    r'views?|images?|footage|clip|video|stills?|instants?|samples)\b',
]) + r')', I)

# a claim scoped to the part or the samples keeps its predicate: "not shown in this part" -> "not shown". "the part"
# alone is a physical one ("the hole is visible in the part"), so only this part, the supplied part and the samples
SCOPED = re.compile(r'\b((?:resolved|visible|shown|seen|established|confirmed|identified|captured|separable|'
                    r'distinguishable|evident|reached|completed|finished|observed|supplied|clear|settled|isolated|'
                    r'verifiable|distinguished|resolvable))\s+(?:with)?(?:in|for|by|from)\s+(?:(?:this|these|those)\s+'
                    r'(?:supplied\s+)?(?:part|samples)|the\s+(?:supplied\s+)?samples|the\s+supplied\s+part)\b', I)
# rewrites that keep the sentence
_QTY = (r'(?:most|much|some|all|part|half|the\s+rest|the\s+remainder|the\s+middle|(?:a\s+)?(?:substantial\s+|large\s+|'
        r'small\s+)?(?:portion|share|fraction))')
REWRITE = [
    (re.compile(r"\b(left|right)[- ]stream(?:'s)?\s+(grippers?|fingers?|jaws?|pads?|tips?)\b", I), r'\1 \2'),
    # not "a stream of water", and not "left-stream-supported cloth", which "left view-supported" would misread
    (re.compile(r"\b(left|right|exo|top|scene)[- ]streams(?:')?(?!\w|\s+of\b|-)", I), r'\1 views'),
    (re.compile(r"\b(left|right|exo|top|scene)[- ]stream(?:'s)?\b(?!\s+of\b|-)", I), r'\1 view'),
    (re.compile(r"\b(?:wrist|wrist-camera|gripper|close-view|handheld)\s+streams\b", I), 'wrist views'),
    (re.compile(r"\b(?:wrist|wrist-camera|gripper|close-view|handheld)\s+stream\b", I), 'wrist view'),
    (re.compile(r"\bstream's\s+(fingers|jaws|pads|tips)\b", I), r"gripper's \1"),
    (re.compile(r'\bsupplied\s+(views?|cameras?)\b', I), r'\1'),
]
# "most of the window is spent" reads "most of the recording is spent", but only in a recording labelled in one part:
# in a stitched one the window is one part, and "most of the recording" would claim a share of time it does not know.
# A bare "the window" only where a verb of time follows or precedes it, so "wipe most of the window" stays
WHOLE_REWRITE = [
    (re.compile(r'\b(' + _QTY + r')\s+of\s+(?:this|the)\s+(?:supplied|labell?ed|current|preparation)\s+(remaining\s+)?'
                + _WIN + r"(?!')", I), r'\1 of the \2recording'),
    (re.compile(r'\b(' + _QTY + r')\s+of\s+the\s+(remaining\s+)?' + _WIN + r'(?=\s+(?:is\s+spent|was\s+spent|consists|'
                r'goes)\b)', I), r'\1 of the \2recording'),
    (re.compile(r'\b(spends?|spent|spending|occup(?:y|ies|ied|ying)|consum(?:e|es|ed|ing))\s+(' + _QTY +
                r')\s+of\s+the\s+(remaining\s+)?' + _WIN + r"(?!')", I), r'\1 \2 of the \3recording'),
]
# a part's marker after a time: "665.10-690.13s (part 3):" keeps its time. "Part 2:" at a sentence start stays: it is
# all that tells a stitched review's parts apart, and without it part 1's "no drop is seen" reads as a claim about the
# whole recording next to part 2's slips
PART_MARK = re.compile(r'(?<=\ds)\s*\((?:request\s+)?part\s+\d+\)', I)
# in a recording labelled in one part, "Part 1:" at a sentence start tells nothing apart, so it goes
PART_HEAD = re.compile(r'(?:(?<=^)|(?<=[.;!?]\s)|(?<=\n))\s*Part\s+\d+\s*[:.\-]\s*', I)
SENTENCE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9("\'])')
SENTENCE_SEP = re.compile(r'((?<=[.!?])\s+(?=[A-Z0-9("\']))')      # the same, keeping each separator (a newline stays)
CLAUSE = re.compile(r'(?<=;)\s+')
# The gate on every drop. Measured on 1,000 published lines (2026-10-30), argus-wording-1 dropped whole clauses with a
# physical fact in them: "Inferred between samples: the bolt is at the opposing tips before and lies unsupported on the
# mat", "The window begins during ongoing sewing, with several buttons already on the cloth". A clause about the
# labelling now goes only when every word of it is a word of the labelling (_LABELLING_WORDS); a clause that only
# places named events (_EVENT_WORDS) between samples keeps them as their names ("Release inferred between samples."
# reads "Release."). Anything else in it (an object, a place, a state, a count, a time, a failure) keeps the
# clause, with at most the samples phrase cut. When unsure the clause stays as the model wrote it.
_LABELLING_WORDS = frozenset('''
a an the this that these those its it each every both all any some most much several one two of in at by on to from
within between during throughout through across over for and or but as is are was were be been being does do did not
no only can cannot can't never may already still exact exactly precise precisely explicit sample samples sampled
sampling frame frames view views image images stills instant instants moment moments interval intervals footage clip
clips video window windows window's request request's requests requested boundary boundaries edge edges start
end beginning segment excerpt part part's parts labelling labeling labelled labeled annotation annotated
supplied provided given evidence rate time timing instruction instructions description descriptions note notes dataset
written source current internal next previous prior following preceding adjacent consecutive successive neighbouring
neighboring available displayed sparse sparsely densely fully first last final same occurs occur occurred happens
happen happened inferred interpolated estimated bounded reconstructed resolve resolves resolved establish establishes
established distinguish distinguishes isolate isolated begins began starts started ends ended one-second
'''.split())
_EVENT_WORDS = frozenset('''
release releases placement placements pickup pickups pick-up grasp grasps transfer transfers closure closures handoff
handoffs handover contact contacts set-down setdown regrasp regrip separation acquisition deposit deposits transition
transitions approach lift lifts exchange emptying lowering reach extraction seating withdrawal detachment disengagement
insertion removal landing engagement turnover retrieval reacquisition brief short fine small rapid fast quick itself
individual initial subsequent intervening
'''.split())
_WORDS = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+")
_EDGE_ABOUT = re.compile('|'.join(_ABOUT_EDGE), I)
# a time placed only between samples; what comes before it names the events: "Release" in "Release inferred between
# samples"
TIMED = re.compile(r'\b' + _TIMING + r'\s+' + SAMPLES +
                   r'|\b(?:inferred|interpolated|estimated|bounded)\s+(?:by|between|within|from|in)\s+(?:the\s+)?'
                   r'(?:samples?|sampled\s+\w+)\b', I)
# a verb left with nothing once "between samples" is cut: "the exact contact falls" would read as a fall, so such a
# clause stays as written
_DANGLING = re.compile(r'\b(?:occurs?|occurred|falls?|fell|happens?|happened|lies?|lay|place|is|are|was|were)'
                       r'(?:\s+\w+ly)?(?=\s*(?:[,;:.!?]|$|(?:and|but|or|while|as|so|then|during|before|after)\b))', I)
# the timing words around a name of an event: "Grasp timing is inferred" -> "Grasp"
_TIMING_WORDS = re.compile(r'\b(?:timing|instants?|moments?|time|is|are|was|were|only|inferred|interpolated|estimated|'
                           r'bounded|reconstructed|approximate|uncertain)\b', I)

# keys whose values are enums or ids, never prose: left as they are
ENUM_KEYS = frozenset({'arm', 'contribution', 'kind', 'outcome', 'category', 'severity', 'deliberate', 'basis', 'list',
                       'excluded_by', 'by', 'side', 'task_completed', 'view', 'content', 'fingerprint', 'signals',
                       'progress_scope', 'id', 'sha256', 'status'})
# task text stays as Argus wrote it, "in this window" included: a task name names the time window it was labelled in
TASK_TEXT_KEYS = frozenset({'task', 'task_summary', 'instruction_variants'})
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


def _rewritten(s: str, whole: bool = False) -> str:
    y = PART_MARK.sub('', s)          # "665.10-690.13s (part 3): ..." -> "665.10-690.13s: ..."
    if whole and PART_HEAD.search(y):  # "Part 1: pick the cup." -> "Pick the cup."
        y = re.sub(r'(^|[.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), _tidy(PART_HEAD.sub(' ', y)))
    for rx, to in REWRITE + (WHOLE_REWRITE if whole else []):
        y = rx.sub(to, y)
    y = SCOPED.sub(r'\1', y)
    return _tidy(y) if y != s else s


def mentions_labelling(s: str) -> bool:
    """Whether a text still holds wording about the labelling (a clause about it, or a phrase placing a claim in it)."""
    return bool(ABOUT.search(s) or CUT.search(s))


# "preparation" and "cut" are the labelling's only as its cuts and windows: "the internal preparation cut", "the request
# cut", "this preparation window". "The window ends during preparation" is a person preparing, and "the cut is inferred
# between samples" can be scissors
_LABELLING_CUT = re.compile(r"\b(?:internal|preparation|request|part|labell?ing)(?:[\s-]+(?:window|request|part)(?:'s)?)?"
                            r"[\s-]+cuts?\b|\bpreparation(?=[\s-]+(?:windows?|boundar(?:y|ies))\b)", I)


def _rest(s: str) -> list[str]:
    """The words of s that are not the labelling's."""
    return [w for w in _WORDS.findall(_LABELLING_CUT.sub(' ', s).lower()) if w not in _LABELLING_WORDS]


# a negation or an auxiliary before the samples phrase: "Release does not occur between samples" says the release was
# seen, and "Release does not." would say it never happened
_AUX_BEFORE = re.compile(r"\b(?:not|never|no|does|do|did|is|are|was|were|can|cannot|can't|will|would|may|might)\s*$", I)
_NEGATION = re.compile(r"\b(?:not|never|no|nor|neither|without|cannot|can't|does|do|did|doesn't|don't|didn't)\b", I)
# a name that ends on one of these lost its verb to the cut: "Release isn't inferred between samples"
# a claim that something did not happen: cutting its scope widens it ("No grasp in this window." in part 1 of a
# stitched review is not "No grasp." over the recording), and a leading "No" or "Neither" names what did not happen
_NEG_ANY = re.compile(r"\b(?:not|never|no|nor|neither|none|nothing|nobody|without|cannot|can't|isn't|aren't|wasn't|"
                      r"weren't|doesn't|don't|didn't|won't|absent|lacks?)\b", I)
_UNIT_WORD = re.compile(r'\b(?:window|part|clip|request|segment)\b', I)   # a scope that is one part's, not the samples'
_LEAD_NEG = re.compile(r'\s*(?:no|neither|none|nothing|never)\b', I)
# a thing put somewhere: "Place the plant in this window." names where it went
_PLACED = re.compile(r'\b(?:plac|put|set|hang|hung|hangs|mount|lean|rest|stand|sit|sat|mov|carr|bring|brought|insert|'
                     r'position|stack|lay|laid|drop|store|leav|left)\w*\b[^.;:]*\b(?:in|into|within|onto|on)\s+'
                     r'(?:this|that|each|the)\s+window\b', I)
_HANGING = re.compile(r"\b(?:not|never|no|nor|isn't|aren't|wasn't|weren't|doesn't|don't|didn't|can't|cannot|won't|"
                      r"wouldn't|couldn't|shouldn't|is|are|was|were|be|been|being|can|could|will|would|may|might|"
                      r"must|should|does|do|did|has|have|had)\s*$", I)


def _timed_events(c: str) -> re.Match | None:
    """The samples phrase of a clause that only places named events between samples ("Release inferred between
    samples"), every event named before it; None for any other clause, a negated one included."""
    m = TIMED.search(c)
    rest = _rest(c)
    if not (m and rest and not _EDGE_ABOUT.search(c) and all(w in _EVENT_WORDS for w in rest)):
        return None
    return None if _rest(c[m.start():]) or _NEGATION.search(c[:m.start()]) else m


def _event_name(head: str) -> str | None:
    """The events a phrase names, without their timing words: "Grasp timing is" -> "Grasp"; None when it names none."""
    y = _tidy(_TIMING_WORDS.sub(' ', head)).strip(' ,;:')
    y = re.sub(r'^(?:the|an?)\s+', '', y, flags=I)
    return y if any(w in _EVENT_WORDS for w in _rest(y)) else None


def _clause(c: str, whole: bool = False) -> str | None:
    k = _clause_cut(c)
    if k and k != c and (_PLACED.search(c) or _LEAD_NEG.match(c) or
                         (not whole and _NEG_ANY.search(c) and _UNIT_WORD.search(c))):
        return c
    return k


def _clause_cut(c: str) -> str | None:
    """One clause, without its closing punctuation, without its process wording. None only when every word of it is
    the labelling's; a clause about the samples that only names events keeps their names ("Release inferred between
    samples" -> "Release"); a clause with anything else in it keeps it, losing at most a phrase that places it in the
    labelling, and stays as written when that phrase cannot be cut alone."""
    if ABOUT.search(c):
        if not _rest(c):
            return None
        m = _timed_events(c)
        if m:
            return _event_name(c[:m.start()]) or c
        y = _tidy(CUT.sub('', c)).rstrip(' ,')
        return _named(y) if y != c and _rest(y) and not ABOUT.search(y) and not _DANGLING.search(y) else c
    if not CUT.search(c):
        return c
    y = _tidy(CUT.sub('', c)).rstrip(' ,')
    rest = _rest(y)
    if not rest:
        return None
    if _DANGLING.search(y) and not _DANGLING.search(c):   # "the transition is partly between samples"
        return c
    return _named(y)


def _named(y: str) -> str:
    """y, or the events it names when nothing but their timing is left: "Release is bounded" -> "Release"."""
    if all(w in _EVENT_WORDS for w in _rest(y)) and _TIMING_WORDS.search(y) and not _NEGATION.search(y):
        return _event_name(y) or y
    return y


def _prose(y: str, whole: bool = False) -> str | None:
    out = []
    parts = SENTENCE_SEP.split(y)
    for i in range(0, len(parts), 2):
        s = parts[i].strip()
        sep = parts[i - 1] if i else ''
        end = s[-1] if s[-1:] in '.!?' else ''
        body = s[:-1] if end else s
        kept, first = [], None
        for c in CLAUSE.split(body):
            c = c.rstrip()
            semi = c.endswith(';')
            c = c[:-1].rstrip() if semi else c
            first = c if first is None else first
            k = _clause(c, whole) if c.strip(' ;') else None
            if k:
                kept.append(k)
        if kept:
            text = '; '.join(kept)
            out.append((sep, (text if kept[0] == first else _cap(text)) + end))
    if not out:
        return None
    return out[0][1] + ''.join(sep + text for sep, text in out[1:])


def normalize_text(s: str, name: bool = False, whole: bool = False) -> str | None:
    """s without wording about the labelling. A name (name=True) loses a phrase about the labelling only when the
    phrase holds nothing else, and comes back unchanged when nothing would be left; prose loses each clause that is only
    about the labelling (_clause) and is None when none is left. whole: the recording was labelled in one part, so
    "most of the window" is "most of the recording"."""
    if not isinstance(s, str) or not s.strip():
        return s
    y = _rewritten(s, whole)
    if not mentions_labelling(y):
        return _late(y)
    if name:
        # a part's edge stays in a name: "Window ends with the border incomplete" without "Window ends" loses its time
        z = ABOUT_CUT.sub(lambda m: m.group(0) if _rest(m.group(0)) or _EDGE_ABOUT.search(m.group(0)) or
                          _NEGATION.search(m.group(0)) or _AUX_BEFORE.search(y[:m.start()]) else '', y)
        z = _tidy(CUT.sub('', z)).strip(' ,;:')
        # "Release isn't", "Neither grasp nor release", "Put the cup", one part's "No grasp" in a stitched review
        if z != y and (_HANGING.search(z) or _LEAD_NEG.match(y) or _PLACED.search(y) or
                       (not whole and _NEG_ANY.search(y) and _UNIT_WORD.search(y))):
            return _late(y)
        return _late(z) if re.search(r'[A-Za-z]{2,}', z) else y
    z = _prose(y, whole)
    if z and z == _cap(y):          # only a capital letter changed: nothing was cut
        return _late(y)
    return _late(z) if z else z


# after the clauses are judged (they read "supplied frames" as the labelling's): the frames a reader sees are frames
LATE = re.compile(r'\bsupplied\s+(?:samples|images|frames)\b|\b(?:sampled\s+)?supplied\s+(footage|video)\b', I)


def _late(s: str) -> str:
    return LATE.sub(lambda m: m.group(1) or 'frames', s)


def edge_event(label) -> bool:
    """Whether a key event's label is only a part's edge, with nothing of the recording in it: "Window ends". One that
    names a state at the edge ("Window ends with the border incomplete", "Knot work still underway at the request
    boundary") is kept: at the recording's own edge that state is a fact."""
    return isinstance(label, str) and bool(EDGE_EVENT.search(label)) and not _rest(label)


def normalize_labels(labels, counts: dict | None = None, log: list | None = None, whole: bool = False):
    """A copy of a result's labels (any depth) with every text normalized (normalize_text): a short field as a name,
    every other as prose; enum fields and Argus's own stitch records are left as they are. A key event that is only a
    part's edge (edge_event) leaves its list. A prose field left with no clause becomes None, and a list entry left
    empty leaves the list. counts, when given, gets 'rewritten' and 'dropped' added up; log, a list, gets (field,
    before, after) for every text changed. whole: the recording was labelled in one part (normalize_text)."""
    counts = {} if counts is None else counts

    def keep(parent, k):
        return k in OWN_KEYS or k in ENUM_KEYS or k in TASK_TEXT_KEYS or (parent, k) in OWN_FIELDS

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
        y = normalize_text(v, name=key in NAME_KEYS, whole=whole)
        if y != v:
            k = 'dropped' if y is None else 'rewritten'
            counts[k] = counts.get(k, 0) + 1
            if log is not None:
                log.append((key, v, y))
        return y

    return walk(labels, '')


def normalize_result(result: dict, whole: bool = False) -> dict:
    """The result with its labels normalized in place and text_normalizer recording VERSION and what changed. whole:
    the recording was labelled in one part (a short episode's reply), not stitched from several."""
    counts = {'rewritten': 0, 'dropped': 0}
    if isinstance(result.get('labels'), dict):
        result['labels'] = normalize_labels(result['labels'], counts, whole=whole)
    result['text_normalizer'] = {'version': VERSION, **counts}
    return result
