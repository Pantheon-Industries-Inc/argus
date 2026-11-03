"""The rule that an idle stretch is never more than a low data issue (label/prompts.py data_contract), held in code:
idleness is trimmed from the dataset, it is not a defect of the footage. The parser (label/harness.py parse_response),
the stitch of a recording labelled in parts (label/pieces.py) and the head-camera board cap (board/build.py
severity_cap) all read it from here, and it imports nothing of Argus, so every one of them can."""
from __future__ import annotations

import re

# An idle data issue is one whose category names the work pausing: idle, idling, pause, stall or inactivity in any
# spelling ("idle_stretch", "operator_paused", "Stalled", "inactivity"), read word by word so "install" is not a stall.
# A category that also names the camera or recording ("camera_stall", "video_paused", "frozen_frame") is a recording
# fault, a frozen or black camera, and not idleness.
IDLE_WORD = re.compile(r'^(?:idl|paus|stall|inactiv)')
RECORDING_WORDS = frozenset({'recording', 'camera', 'cam', 'video', 'frame', 'frames', 'stream', 'feed', 'sensor',
                             'frozen', 'freeze', 'black', 'dropout', 'fault'})


def idle_category(category) -> bool:
    """Whether a data issue's category names idleness (IDLE_WORD) and not a recording fault (RECORDING_WORDS)."""
    if not isinstance(category, str):
        return False
    words = [w for w in re.split(r'[^a-z0-9]+', re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', category).lower()) if w]
    return any(IDLE_WORD.match(w) for w in words) and not RECORDING_WORDS.intersection(words)


# Idle that is really other activity, in the words of PHONES, BARE HANDS, PARKED GRIPPERS AND FACES (label/prompts.py):
# the operator holding or using a phone, another person reaching in or handling the objects, bare-hand work, a
# gripper set down, parked or taken off while the work goes on. Each is at least medium there, so an idle item whose
# issue or evidence describes one keeps the model's severity. The text is read clause by clause (other_activity).
_GRIPPER = r'(?:(?:left|right|other|one|a|the|both|his|her|their)\s+)*grippers?(?!\s+(?:jaws?|tips?|fingers?)\b)'
# a person's own hand or fingers, never the gripper's ("empty-finger movements", "finger adjustments")
_HUMAN_HAND = (r"(?:bare|free|human|own|operator's|demonstrator's|person's|participant's|wearer's|his|her|their)\s+"
               r"(?:(?:left|right)\s+)?(?:fingers?|hands?)")
# what another person does to the work: a person merely nearby, at a doorway or moving in the background is not it
_PERSON = (r'(?:\b(?:another|other|second|third|different)\s+(?:person|people|operator|individual|human|worker)s?\b'
           r"|\bsomeone(?:\s+else)?\b|\bsomebody\b|\banyone\s+else\b|\bbystanders?\b|\bhelpers?\b)(?:'s)?")
_PERSON_ACTS = (r'\b(?:reach|enter|step|hand|touch|mov|pick|grab|place|plac|take|took|lift|adjust|help|assist|'
                r'interven|pass|giv|gave|hold|held|handl|carr|push|pull|remov|add|bring|brought|arm|insert|tak)\w*')
OTHER_ACTIVITY = {
    'phone': re.compile(r'\b(?:i|smart|cell\s?|mobile\s)?phones?\b(?!-like)|\b(?:answers?|answering|takes?|taking|'
                        r'makes?|making|on)\s+a\s+(?:phone\s+)?call\b', re.I),
    'other_person': re.compile(_PERSON + r'[^.;]*?' + _PERSON_ACTS + '|' + _PERSON_ACTS + r'[^.;]*?\bby\s+' + _PERSON +
                               r'|\bexternal(?:ly)?[- ]assist\w*', re.I),
    'bare_hand': re.compile(r'\bbare[- ]?hand(?:s|ed)?\b|\b' + _HUMAN_HAND + r'\s+(?:clearly\s+)?(?:fiddl|mov|push|'
                            r'slid|pick|grab|reposition|adjust|handl|touch)\w*|\bby\s+hand\b', re.I),
    'parked_gripper': re.compile(
        r'\b(?:set|sets|setting|put|puts|putting|lay|lays|laid|laying|place|places|placed|placing|rest|rests|'
        r'rested|resting|park|parks|parked|parking)[\s-]+(?:down\s+)?' + _GRIPPER + r'(?:\s+(?:down|aside))?\b'
        r'|\b(?:take|takes|took|taken|taking|remove|removes|removed|removing|unstrap\w*|detach\w*)\s+(?:off\s+)?'
        + _GRIPPER +
        r'|\bgrippers?\s+(?:is\s+|are\s+|was\s+|were\s+|being\s+|lies\s+|lying\s+|sits\s+|sitting\s+|left\s+)?'
        r'(?:set\s+down|put\s+down|laid\s+down|parked(?!\s+(?:above|over|in\s+the\s+air|mid-?air))|taken\s+off|'
        r'removed|detached|aside|down\s+on)\b'
        # the same states through more linking verbs and an adverb ("remains parked", "has been set down", "is visibly
        # parked"), and a gripper resting on a surface ("rests on the table", "is left resting on the table"); a gripper
        # that remains aside, raised or near the table, or is parked above it, is held still, which is idle
        r'|\bgrippers?\s+(?:(?:is|are|was|were|being|been|has|have|had|remains?|remained|remaining|stays?|stayed|'
        r'staying|left)\s+)+(?:\w+ly\s+)?(?:set\s+down|put\s+down|laid\s+down|parked(?!\s+(?:above|over|in\s+the\s+air|'
        r'mid-?air))|taken\s+off|removed|detached)\b'
        r'|\bgrippers?\s+(?:(?:is|are|was|were|has|have|had|been|remains?|remained|remaining|stays?|stayed|staying|'
        r'left)\s+)*(?:(?:rests?|rested|resting|lies|lying|sits|sitting|placed|put|set|laid)\s+on|(?:remains?|remained|'
        r'stays?|stayed|left|lies|lying|sits|sitting)\s+on\s+the\s+(?:table|tabletop|surface|mat|floor|desk|bench))\b',
        re.I),
}
FOB = re.compile(r'\bfob\b', re.I)
# a clause that says the other person does not act ("without visibly touching", "does not move") names no other activity
NEGATION = re.compile(r"\b(?:not|no|without|never|nor)\b|n't\b", re.I)
# a denial whose object is the operator is about the operator, not the other person ("without the operator noticing",
# "someone else, not the operator, handles the towel")
ABOUT_OPERATOR = re.compile(r"\s+(?:the\s+|a\s+|an\s+)?(?:operator|wearer|demonstrator|camera\s+wearer)s?\b", re.I)
# after the person's verb, a denial is of the person's action only when an action or contact follows it ("without
# visibly touching", "does not touch", "no contact"), never "without the operator noticing" or "no task progress"
DENIED_ACT = re.compile(r"\s*(?:\w+ly\s+|ever\s+|even\s+)?(?:" + _PERSON_ACTS[2:] + r"|contact\w*|interact\w*)", re.I)
# a denial whose subject is the operator ("the operator does not move", "the operator doesn't react")
OPERATOR_SUBJECT = re.compile(r"\b(?:operator|wearer|demonstrator)s?\s+(?:does|did|do|is|was|are|were|has|have|had|"
                              r"can|could|will|would|remains|stays)\s*$", re.I)
# clauses end at a sentence stop or semicolon followed by a space, never at a decimal point ("335.33 s")
CLAUSE = re.compile(r'[.;!?](?=\s|$)')


def _counts(kind: str, clause: str, m: re.Match) -> bool:
    """Whether one match of a kind counts. A phone or a gripper set down in the same comma-separated part as the
    recording's fob is the fob being pressed; another person counts unless their part, from the person on, denies
    what they do ("the other person moves beside the workspace without touching", not "the grippers do not move
    while another person adjusts the bowl"). Only a denial of the person's action counts (_denies_person)."""
    start = clause.rfind(',', 0, m.start()) + 1
    end = clause.find(',', m.end())
    end = len(clause) if end < 0 else end
    if kind in ('phone', 'parked_gripper'):
        return not FOB.search(clause[start:end])
    if kind == 'other_person':
        return not any(_denies_person(clause, m, n) for n in NEGATION.finditer(clause, m.start(), end))
    return True


def _denies_person(clause: str, m: re.Match, n: re.Match) -> bool:
    """Whether a negation n in another person's part denies what that person does (issue 185). Between the person and
    the verb the match found, it does, unless it is an aside naming the operator ("someone else, not the operator,
    handles the towel"): there the verb may be the operator's ("a bystander watches without the operator moving the
    cup"). After that verb it does only when an action or contact follows it (DENIED_ACT) and the operator is not its
    subject (OPERATOR_SUBJECT), so "without the operator noticing", "no task progress" and "the operator does not
    move" leave the person's action standing."""
    if n.start() < m.end():
        op = ABOUT_OPERATOR.match(clause, n.end())
        aside = op and re.search(r',\s*$', clause[:n.start()]) and re.match(r'\s*,', clause[op.end():])
        return not aside
    return bool(DENIED_ACT.match(clause, n.end())) and not OPERATOR_SUBJECT.search(clause[:n.start()])


def other_activity(row: dict) -> list[str]:
    """The kinds of other activity (OTHER_ACTIVITY) an issue's text and evidence describe, read clause by clause, each
    match judged where it stands (_counts)."""
    text = ' '.join(v for v in (row.get('issue'), row.get('evidence')) if isinstance(v, str))
    found = set()
    for clause in CLAUSE.split(text):
        for kind, pattern in OTHER_ACTIVITY.items():
            if kind not in found and any(_counts(kind, clause, m) for m in pattern.finditer(clause)):
                found.add(kind)
    return [kind for kind in OTHER_ACTIVITY if kind in found]


def idle_issues_low(labels: dict) -> None:
    """Every data issue with an idle category (idle_category) gets severity low, and one the model rated otherwise keeps
    that rating in severity_as_labelled, unless its text describes other activity (other_activity): then it keeps its
    severity and its category. Operator mistakes are left as they are. Running this again on its output changes
    nothing."""
    for row in labels.get('data_issues') or []:
        if (isinstance(row, dict) and idle_category(row.get('category')) and row.get('severity') != 'low'
                and not other_activity(row)):
            row.setdefault('severity_as_labelled', row.get('severity'))
            row['severity'] = 'low'
