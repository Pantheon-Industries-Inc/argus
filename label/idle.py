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
# clauses end at a sentence stop or semicolon followed by a space, never at a decimal point ("335.33 s")
CLAUSE = re.compile(r'[.;!?](?=\s|$)')


def other_activity(row: dict) -> list[str]:
    """The kinds of other activity (OTHER_ACTIVITY) an issue's text and evidence describe, read clause by clause: in a
    clause that names the recording's fob, a phone or a gripper set down is the fob being pressed, not one of them, and
    another person counts only in a clause that does not deny what they do."""
    text = ' '.join(v for v in (row.get('issue'), row.get('evidence')) if isinstance(v, str))
    found = set()
    for clause in CLAUSE.split(text):
        fob, denied = bool(FOB.search(clause)), bool(NEGATION.search(clause))
        for kind, pattern in OTHER_ACTIVITY.items():
            if kind in found or (fob and kind in ('phone', 'parked_gripper')) or (denied and kind == 'other_person'):
                continue
            if pattern.search(clause):
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
