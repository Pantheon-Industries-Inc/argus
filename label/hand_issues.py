"""The rule that bare-hand work is a medium or high data issue only when it is egregious, held in code (label/prompts.py
PHONES, BARE HANDS, PARKED GRIPPERS AND FACES): a bare hand with no gripper on it, a hand reaching below the table for
an object, something in the mouth, another person acting on the work, or fingers or a palm plainly grasping a task object
with the gripper on. A bare-hand item that only appears, seems or may be so, or that describes fingers helping while the
gripper stays worn, is low: the benefit of the doubt (Eric, 2026-11-11, after a page of 19 dropped clips where most
medium items were guesses such as "bare fingertips appear to assist"). The parser (label/harness.py parse_response) and
the stitch of a recording labelled in parts (label/pieces.py) apply it after idle_issues_low."""
from __future__ import annotations

import re

from label.idle import other_activity

# A data issue about the operator's own hands: human_intervention (the tag Argus reuses for it), bare-hand tags and
# finger tags. face_visible and the rest are other rules.
HAND_CATEGORY = re.compile(r'human[_ -]?intervention|bare[_ -]?hand|hand[_ -]?(?:work|assist|use)|finger', re.I)

# Wording that states a guess, not a certainty.
HEDGE = re.compile(r'\b(?:appears?|appeared|appearing|seems?|seemed|seeming|may|might|possibly|possible|probably|'
                   r'likely|apparently|perhaps|presumably|suggests?|suggesting|could\s+be|looks?\s+like|unclear)\b', re.I)

# The gripper stays on while the fingers help: the case the benefit of the doubt covers unless the text is egregious.
GRIPPER_WORN = re.compile(r'\bgrippers?\s+(?:\w+\s+){0,2}(?:remains?|remained|stays?|stayed|is|are|still|being)\s+'
                          r'(?:\w+\s+)?(?:worn|on|strapped|used|in\s+use)\b'
                          r'|\bwhile\s+(?:the\s+)?(?:\w+\s+){0,2}grippers?\s+(?:hold|holds|continue|continues|work|works|'
                          r'grip|grips|keep|keeps|are|is|remain|remains)\b'
                          r'|\balongside\s+(?:the\s+)?(?:\w+\s+){0,2}grippers?\b'
                          r'|\bin\s+addition\s+to\s+(?:the\s+)?grippers?\b', re.I)

# The egregious cases, which keep their severity whatever the wording.
EGREGIOUS = {
    'mouth': re.compile(r'\b(?:mouth|lips?|teeth|bites?|biting|bit|chews?|chewing)\b', re.I),
    'no_gripper': re.compile(r'\bgrippers?\s+(?:\w+\s+){0,2}(?:off|removed|taken\s+off|not\s+worn|unstrapped|set\s+down|'
                             r'put\s+down|parked)\b|\bwithout\s+(?:the\s+|a\s+|any\s+)?grippers?\b|\bno\s+grippers?\b|'
                             r'\b(?:takes?|took|taking|removes?|removed|removing)\s+(?:off\s+)?(?:the\s+|a\s+|their\s+)?'
                             r'(?:\w+\s+)?grippers?\b', re.I),
    'below_table': re.compile(r'\b(?:floor|ground)\b|\b(?:beneath|under|underneath|below)\s+the\s+(?:table|desk|bench)\b',
                              re.I),
    'grasp': re.compile(r'\b(?:palm|whole\s+hand|full[- ]hand|fist)\b'
                        r'|\b(?:fingers?|hands?)\s+(?:\w+\s+){0,3}(?:grasps?|grasped|grasping|grips?|gripped|gripping|'
                        r'holds?|held|holding|clutch\w*)\b'
                        r'|\b(?:held|grasped|gripped|carried|operated)\s+(?:\w+\s+){0,3}by\s+(?:the\s+|their\s+|his\s+|'
                        r'her\s+)?(?:bare\s+)?(?:fingers?|hands?)\b', re.I),
}


def egregious(row: dict) -> list:
    """The egregious kinds the issue or evidence of a data issue names, in EGREGIOUS order, plus another person or a phone
    (label/idle.py other_activity), which are never bare-hand guesses."""
    text = f"{row.get('issue') or ''} {row.get('evidence') or ''}"
    found = [k for k, p in EGREGIOUS.items() if p.search(text)]
    return found + [k for k in other_activity(row) if k in ('other_person', 'phone')]


def hand_issues_low(labels: dict) -> None:
    """Every medium or high data issue about the operator's own hands (HAND_CATEGORY) that names no egregious case
    (egregious) and is worded as a guess (HEDGE) or describes the fingers helping while the gripper stays on
    (GRIPPER_WORN) gets severity low, keeping the model's rating in severity_as_labelled. Running it again changes
    nothing."""
    for row in labels.get('data_issues') or []:
        if not isinstance(row, dict) or str(row.get('severity') or '').lower() not in ('medium', 'high'):
            continue
        if not HAND_CATEGORY.search(str(row.get('category') or '')) or egregious(row):
            continue
        text = f"{row.get('issue') or ''} {row.get('evidence') or ''}"
        if HEDGE.search(text) or GRIPPER_WORN.search(text):
            row.setdefault('severity_as_labelled', row.get('severity'))
            row['severity'] = 'low'
