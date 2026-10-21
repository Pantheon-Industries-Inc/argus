"""Problem families decided by clause polarity (board/polarity.py), measured over corpora rather than one wording at a
time, and the consumers of a label agreeing on every value they derive from it (board/label_projection.py)."""
from __future__ import annotations

import collections
import copy
import importlib.util
import json
import math
import random
from pathlib import Path

import pytest

from board import to_board
from board.families import Families
from board.label_projection import max_severity, severity
from checks import label_consistency
from compare import metrics
from gate.score import judge
from label.harness import annotation_problems, parse_response

_spec = importlib.util.spec_from_file_location('family_wordings',
                                               Path(__file__).parent / 'fixtures' / 'family_wordings.py')
wordings = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wordings)

FAMILIES = Families()
BROAD = 'd:Camera fault'
CODE = {'P': 'privacy-blur', 'L': 'low-light', 'F': 'camera-fault', 'I': 'camera-image', 'D': BROAD}


def family(issue, evidence=''):
    return FAMILIES.family_of('data_issues', {'category': 'camera_fault', 'severity': 'high', 'issue': issue,
                                              'evidence': evidence})


def worst_precision_recall(items):
    """The lowest per-family precision or recall over (issue, evidence, expected family) items."""
    pred = [family(issue, evidence) for issue, evidence, _ in items]
    worst = 1.0
    for fam in CODE.values():
        tp = sum(1 for (_, _, want), got in zip(items, pred) if want == fam and got == fam)
        npred, nwant = pred.count(fam), sum(1 for _, _, want in items if want == fam)
        worst = min(worst, tp / npred if npred else 1.0, tp / nwant if nwant else 1.0)
    wrong = [(issue, evidence, got, want) for (issue, evidence, want), got in zip(items, pred) if got != want]
    return worst, wrong


# ---------------------------------------------------------------- the wordings of the reviews

@pytest.mark.parametrize('text,wanted', [
    # bug 21: wordings the previous fix broke
    ('Glare never appears in any frame', BROAD),
    ('Dropout is never an issue here', BROAD),
    ('Occlusion never happens', BROAD),
    ('It looks like the view is frozen but it is just a static scene', BROAD),
    # a correction whose subject is another noun phrase states something about that subject, not the impression
    ('At first it seems frozen, but the arm is just still', 'camera-image'),
    ('Looks like glare, actually a white tablecloth', BROAD),
    # a question asserts nothing, and a sentence with something of its own to say is no answer to it
    ('Glare? No question, it hides the cup', BROAD),
    ('Free of glare until 5 s, then a finger covers the lens', 'camera-fault'),
    ('The view is free of glare except near the window where the plate is clipped', 'low-light'),
    ('The footage is free of noise and a finger covers the lens', 'camera-fault'),
    # bug 22: negated remedies, multiword answers, lists
    ('The occlusion is not removed until 9 s', 'camera-fault'),
    ('Overexposure is not corrected by auto exposure', 'low-light'),
    ('The dropout has not ended by the last frame', 'camera-image'),
    ('A rectangular blur is not lifted at any point', 'privacy-blur'),
    ('Clipping is not limited to the plate', 'low-light'),
    ('Glare never clears from the table', 'low-light'),
    ('Frozen frames never recover after 4 s', 'camera-image'),
    ('Washed-out highlights? No.', BROAD),
    ('Finger over the lens? No.', BROAD),
    ('Frame freeze? No.', BROAD),
    ('No occlusion, camera rotation or glare', BROAD),
    ('No dropout, video corruption or streaking', BROAD),
    ('No glare, image corruption or occlusion', BROAD),
    ('No glare, the image is clean', BROAD),
    ('Occlusion: none', BROAD),
    ('Glare: none', BROAD),
    ('Glare? None.', BROAD),
    ('Checked 0 to 30 s and ruled out occlusion', BROAD),
    ('The exo view is frozen, and the wrist looks like glare but it is not', 'camera-image'),
    ('Is the exo view frozen? Yes, from 4 s to 9 s', 'camera-image'),
    ('No glare. The exo view is frozen from 4 s', 'camera-image'),
    ('nto glare anywhere', BROAD),
    ('A rcetangular blur covers the cutting board', 'privacy-blur'),
    # notes that search, judge a name wrong, or answer in field notation; a retraction in a later sentence retracts
    # nothing (board/polarity.py), so those keep what the earlier words assert, while a withdrawal that names the
    # impression in its own sentence retracts it
    ('Earlier I wrote that the lens was covered. Correction: it never was.', 'camera-fault'),
    ('I initially suspected glare. Nothing is clipped. I am withdrawing the glare flag.', 'low-light'),
    ('I initially suspected glare and I am withdrawing the glare flag.', BROAD),
    # a refused verdict or withdrawal, a denied other explanation and a state that ended keep the impression (review
    # of 2ab6857)
    ('I initially suspected glare, but I have not retracted that finding.', 'low-light'),
    ('I initially suspected glare, but actually I am not withdrawing the glare flag.', 'low-light'),
    ('The exo camera appears frozen, but actually it is not just a static scene.', 'camera-image'),
    ('Glare was reported and has not been retracted.', 'low-light'),
    ('The camera looked frozen, but it is actually no longer frozen.', 'camera-image'),
    ('I initially suspected glare and I am withdrawing the occlusion flag.', 'low-light'),
    ('The camera seems frozen, but this is not the case.', BROAD),
    ('Looks like glare, actually a white tablecloth', BROAD),
    ('I am withdrawing the glare flag.', BROAD),
    ('Three things checked: glare, occlusion, and freezing. Glare is absent. Freezing occurs twice.', 'camera-image'),
    ('It would be inaccurate to call this footage overexposed.', BROAD),
    ('Unobstructed view of the workspace for the full clip.', BROAD),
    ('Frozen: no | Clipping: no | Rotation: yes, from 0:15', 'camera-fault'),
    ('Black frames: 4. Glare: 0.', 'camera-image'),
    ('No change in lighting fixes the overexposure; it persists through the end.', 'low-light'),
    ('None of the usual suspects (glare, clipping) show up; instead the feed lost its video at 0:45.', 'camera-image'),
])
def test_reported_wordings(text, wanted):
    assert family(text) == wanted


def test_text_refines_only_a_tag_that_names_no_one_problem():
    # a known tag that names a problem of its own keeps it, whatever its prose mentions
    known = {'category': 'missing_instruction', 'severity': 'high', 'issue': 'A blur patch hides the instruction card'}
    assert FAMILIES.family_of('data_issues', known) == 'd:Instruction missing'
    # a tag the model coined names nothing, so a privacy blur's own words still claim it
    coined = dict(known, category='lighting')
    assert FAMILIES.family_of('data_issues', coined) == 'privacy-blur'
    # the issue says what the problem is; evidence is read only when the issue asserts no family
    assert family('The exo view is frozen', 'glare on the plate at 2 s') == 'camera-image'
    assert family('No glare', 'the exo view is frozen from 4 s to 9 s') == 'camera-image'


# ---------------------------------------------------------------- spelling

def _spelled_whole_text(fam, text):
    """The spelling rule read over the whole text for every word and every swap, as it was first written: slow, and
    the reference the windowed reading must agree with."""
    import re
    out = text
    hit = lambda s, a, b: any(x.start() < b and x.end() > a for r in fam._re.values() for x in r.finditer(s))
    for m in re.finditer(r"[^\W\d_]{4,}", text):
        a, b = m.span()
        if hit(out, a, b):
            continue
        w = m.group()
        for k in range(len(w) - 1):
            if w[k] != w[k + 1]:
                cand = out[:a] + w[:k] + w[k + 1] + w[k] + w[k + 2:] + out[b:]
                if hit(cand, a, b):
                    out = cand
                    break
    return out


def test_spelling_reads_the_same_as_the_whole_text_reading():
    """Spelling correction reads only the stretch of text around each word and tries only the swaps that can spell a
    piece of a family's word; on notes made of family words, their swapped spellings and filler it puts right exactly
    the words the whole-text reading does."""
    rng = random.Random(5)
    words = ['frozen', 'freezes', 'stale', 'corrupt', 'dropout', 'glare', 'clipping', 'overexposed', 'washed-out',
             'removed', 'lifted', 'rotated', 'sideways', 'upside', 'covered', 'occluded', 'finger', 'anonymised',
             'black screen', 'no signal', 'loses its image', 'fills much of the kitchen lens', 'turned away',
             'faces the wearer', 'blurred patch', 'images are identical', 'repeats the same', 'the', 'operator',
             'holds', 'kitchen', 'table', 'not', 'looks', 'like', 'at', '3s', 'and', 'it', 'is', 'item_4', 'a']
    def swap(x):
        if len(x) < 2:
            return x
        k = rng.randrange(len(x) - 1)
        return x[:k] + x[k + 1] + x[k] + x[k + 2:]

    swapped = 0
    for _ in range(300):
        note = []
        for _ in range(rng.randrange(2, 40)):
            w = rng.choice(words)
            if rng.random() < 0.5:
                w = ' '.join(map(swap, w.split(' ')))
            note.append(w + rng.choice([' ', ' ', ', ', '. ', '\n', '-']))
        text = ''.join(note)
        want = _spelled_whole_text(FAMILIES, text)
        swapped += want != text
        assert FAMILIES._spelled(text) == want, text
    assert swapped > 150


def test_spelling_reads_a_long_note_in_time_linear_in_its_length():
    """A label's evidence has no length limit, and the board classifies every issue when it loads. Spelling
    correction read the whole note for every word and every swap, so 13 KB of evidence took 19 s and 44 KB 212 s.
    It now reads a bounded stretch per word: the characters its patterns scan grow with the note, not its square,
    and the family is the same."""
    class Counting:
        scanned = 0

        def __init__(self, r):
            self.r, self.pattern, self.flags = r, r.pattern, r.flags

        def finditer(self, s, *a):
            Counting.scanned += len(s)
            return self.r.finditer(s, *a)

    fam = Families()
    fam._re = {k: Counting(r) for k, r in fam._re.items()}
    per_byte = []
    for n in (100, 1000):
        for issue, tail in [('The video is frozen.', ''), ('The video is forzen.', ''),
                            ('The camera is on the table.', ' Then a dorpout.')]:
            evidence = 'The operator holds the cup above the table. ' * n + tail
            Counting.scanned = 0
            got = fam.family_of('data_issues', {'category': 'camera_fault', 'severity': 'high', 'issue': issue,
                                                'evidence': evidence})
            assert got == 'camera-image', (issue, n)
            per_byte.append(Counting.scanned / len(evidence))
    assert max(per_byte) < 200 and max(per_byte[3:]) < 1.2 * max(per_byte[:3]), per_byte


# ---------------------------------------------------------------- corpora

def reviewer_corpus():
    """The ninth review's family corpus (744 texts): each family's problems wrapped in unrelated negations, and
    denials of each family in fifteen grammatical forms."""
    pos = {
        'privacy-blur': ["A rectangular blur covers the cutting board", "A blur patch hides the cup during the grasp",
                         "Anonymization blurs the mug and the operator's hands", "A blur mask sits over the drawer front",
                         "A bounded blur region hides the screen", "The anonymized region hides the label on the bottle",
                         "A blurred patch hides the faucet", "Anonymisation hides the oven display",
                         "A rectangular blur sits over the sink", "The blur mask hides the knife"],
        'low-light': ["Glare from the window hides the table", "The image is washed out by the overhead light",
                      "Highlights are clipped on the white plate", "Strong overexposure on the exo camera",
                      "Sun glare hides the cup", "The exo view is overexposed near the sink",
                      "Glare on the countertop hides the spoon", "The white bowl is clipped to pure white",
                      "The wrist view is washed-out near the lamp", "The exo frame is overexposed for 4 s"],
        'camera-fault': ["The wrist camera is rotated sideways", "The head camera is covered by the operator's sleeve",
                         "A finger blocks part of the lens", "The exo camera points down at the floor",
                         "The exo camera is displaced mid-episode", "The operator adjusts the head camera",
                         "The head camera is obstructed by a cabinet door", "The camera turns away from the table",
                         "A towel occludes the wrist camera", "The operator removes the head camera"],
        'camera-image': ["The exo view is frozen", "The wrist feed goes black", "Frames are corrupted with green streaks",
                         "The head camera loses its signal", "Video dropout on the wrist camera",
                         "The image freezes on one frame", "The exo frames are corrupt for 2 s",
                         "The wrist stream shows a black screen", "The head view is stale for 3 s",
                         "A dropout blanks the exo view"],
    }
    terms = {
        'privacy-blur': ["blur patch", "rectangular blur", "anonymization", "blur mask", "bounded blur",
                         "anonymised region"],
        'low-light': ["glare", "clipping", "overexposure", "washed-out highlights", "clipped highlights", "glare"],
        'camera-fault': ["occlusion", "obstruction", "rotated camera", "finger over the lens", "cover on the lens",
                         "lens covering"],
        'camera-image': ["frozen frame", "dropout", "corruption", "black frame", "streaking", "frame freeze"],
    }
    other = {'privacy-blur': ('camera-image', "the exo view is frozen"),
             'low-light': ('camera-fault', "a finger blocks the lens"),
             'camera-fault': ('low-light', "glare hides the cup"),
             'camera-image': ('camera-fault', "the head camera is rotated sideways")}
    neutral = ["The wrist camera is slightly out of focus", "Motion blur on the fast reach",
               "Auto white balance shifts the colours", "Compression artifacts around the gripper",
               "A lens smudge softens the image", "The exo view is slightly tilted"]
    ev = 'exo 4.0 s to 9.5 s'

    def lc(s):
        return s[0].lower() + s[1:]

    def typo(s):
        w = s.split()
        i = 1 if len(w) > 2 else 0
        x = w[i]
        w[i] = x[:1] + x[2:3] + x[1:2] + x[3:] if len(x) > 3 else x
        return ' '.join(w)

    items = []
    for fam, sentences in pos.items():
        for s in sentences:
            items += [(f"The grasp is not visible because {lc(s)}", ev, fam), (f"{s} so the grasp is not visible", ev, fam),
                      (f"{s}, and the grasp is not visible", ev, fam), (f"{s}. The grasp is not visible.", ev, fam),
                      (s, "the cup is not visible from 4.0 s to 9.5 s", fam),
                      (f"No task progress for 3 s while {lc(s)}", ev, fam),
                      (f"{s}, which does not affect the outcome", ev, fam), (typo(s), ev, fam),
                      (f"Not motion blur but {lc(s)}", ev, fam)]
        for k in range(6):
            t = terms[fam]
            a, b, c = t[k % 6], t[(k + 1) % 6], t[(k + 2) % 6]
            n = neutral[k]
            ofam, oth = other[fam]
            items += [(f"{n}; no {a} is present", ev, BROAD), (n, f"No {a} in any frame", BROAD),
                      (n, f"{a[0].upper() + a[1:]} is not present in any frame", BROAD),
                      (n, f"checked 0 to 30 s: there is no {a}", BROAD), (f"{n}, without any {a}", ev, BROAD),
                      (f"{n}; the footage is free of {a}", ev, BROAD), (n, f"{a[0].upper() + a[1:]}? No.", BROAD),
                      (n, f"No {a}, {b} or {c}", BROAD), (n, f"Initially looks like {a} but it is not", BROAD),
                      (n, f"{a[0].upper() + a[1:]} was not observed", BROAD),
                      (f"{n}. There is no sign of {a}.", ev, BROAD), (f"Not {a} but {oth}", ev, ofam),
                      (n, f"{a[0].upper() + a[1:]} absent throughout", BROAD),
                      (n, f"Neither {a} nor {b} appears", BROAD), (n, f"nto {a} anywhere", BROAD),
                      (f"The exo view is not without {a}", ev, fam)]
    return items


@pytest.mark.parametrize('name,items,bound', [
    ('reviewer corpus', reviewer_corpus(), 0.97),
    ('written before the matcher', [(i, e, CODE[x]) for _, i, e, x in wordings.WRITTEN], 0.97),
    ('written independently', [(i, e, CODE[x]) for _, i, e, x in wordings.INDEPENDENT], 0.95),
    ('third independent set', [(i, e, CODE[x]) for _, i, e, x in wordings.THIRD], 0.95),
    ('fourth independent set', [(i, e, CODE[x]) for _, i, e, x in wordings.FOURTH], 0.94),
    ('second looks', [(i, e, CODE[x]) for _, i, e, x in wordings.SECOND_LOOK], 1.0),
])
def test_family_precision_and_recall_over_corpora(name, items, bound):
    worst, wrong = worst_precision_recall(items)
    assert worst >= bound, (name, worst, wrong[:10])


def test_a_later_sentence_never_removes_a_family():
    """Reviews of f050fa2, 2514ce5: a later sentence retracted a fault when it was a normal result for another camera, a
    pronoun for something else, a withdrawal that was negated, asked, supposed or about another camera, a recovery, or
    a correction limited in time. Telling what a later sentence is about needs its meaning, so a later sentence retracts
    nothing (board/polarity.py): every fault keeps its family whatever follows it, in one field or split across issue and
    evidence."""
    from board.polarity import denied_spans
    faults = {'camera-image': ['The exo camera appears frozen.', 'The wrist view looks frozen.',
                               'The exo view is frozen from 4 s.', 'The wrist feed goes black.'],
              'low-light': ['Glare appears on the plate.', 'The image looks overexposed.',
                            'Highlights are clipped on the white bowl.'],
              'camera-fault': ['The lens appears covered.', 'The wrist camera seems rotated sideways.',
                               'A finger blocks part of the lens.'],
              'privacy-blur': ['A rectangular blur appears over the sink.', 'A blurred patch hides the faucet.']}
    later = ['The wrist camera plays normally throughout.', 'The grasp proceeds normally.',
             'The right view works correctly throughout.', 'The timestamps advance normally.', 'The audio is fine.',
             'The operator does not touch it.', 'Nobody notices it.', 'It does not affect the grasp.',
             'That is not the main problem.', 'On review, it is just a static scene.',
             'Actually, nothing is wrong with it.', 'On closer inspection, it is not frozen after 9 s.',
             'On closer inspection, the timestamps advance while the same frame is repeated.',
             'On closer inspection, it is not.', 'Correction: it never was.', 'That was wrong.',
             'It is no longer frozen.', 'It was not frozen at the start.', 'It is not covered anymore.',
             'The wrist camera plays normally. It is not frozen.',
             'I am not withdrawing the glare flag.', 'I will never dismiss the frozen-frame finding.',
             'Do not retract the covered-lens finding.', 'Should I withdraw the blur finding?',
             'If we withdraw the glare flag, the report will be wrong.',
             'I am withdrawing the glare flag.', 'I am withdrawing the frozen-frame finding for the wrist camera.',
             'The covered-lens flag was withdrawn.', 'Scratch that.',
             'I am withdrawing the wrist camera glare flag.', 'Retracting my earlier note about a blur patch.']
    right = total = 0
    wrong = []
    for fam, sentences in faults.items():
        for s in sentences:
            for t in later:
                for issue, evidence in ((s + ' ' + t, ''), (s, t)):
                    total += 1
                    got = family(issue, evidence)
                    right += got == fam
                    if got != fam:
                        wrong.append((issue, evidence, got))
            note = s + ' ' + later[0]
            assert not any(a <= len(s) - 2 < b for a, b in denied_spans(note)), note
    assert right == total, wrong[:10]
    # within its own sentence a correction still retracts
    assert family('The view looks frozen, but it is not.') == BROAD
    assert family('Looks like glare, actually a white tablecloth') == BROAD


def test_reading_a_note_takes_time_in_proportion_to_its_length():
    """Review of 2514ce5: a pass that compared every clause with every later one took 19 s on a 54 KB evidence field and
    stalled the episode list. Reading a note is one pass over its clauses, so ten times the text takes about ten times
    as long, never the hundredfold a comparison of every pair would."""
    import gc
    import time
    from board.polarity import denied_spans
    unit = 'The operator appears to hold the cup above the table. '
    timings = {}
    # the fastest of several runs, with collection paused, so a stall of a shared machine never decides the ratio
    gc.disable()
    try:
        for n in (100, 1000):
            text = 'The exo camera appears frozen. ' + unit * n
            runs = []
            for _ in range(5):
                start = time.perf_counter()
                denied_spans(text)
                runs.append(time.perf_counter() - start)
            timings[n] = min(runs)
    finally:
        gc.enable()
    assert timings[1000] < 1.0, timings
    assert timings[1000] < 30 * max(timings[100], 1e-3), timings


def test_a_correction_in_the_same_sentence_retracts_only_what_it_refers_to():
    """A correction clause reaches the impression before it in its sentence only when it refers to it: its subject is
    a pronoun, it has none of its own, or it is the impression's subject again. One whose subject is another noun
    phrase (another camera, another object, the timestamps rather than the picture) states something about that and
    retracts nothing, after an impression with a named subject, a pronoun or none."""
    impressions = {'camera-image': ['The exo camera appears frozen', 'The wrist view looks frozen', 'It looks frozen',
                                    'At first the feed seems to freeze', 'Looks frozen'],
                   'low-light': ['Glare appears on the plate', 'The image looks overexposed', 'It looks like glare'],
                   'camera-fault': ['The lens appears covered', 'The head camera looks obstructed',
                                    'At first the camera seems turned away'],
                   'privacy-blur': ['A rectangular blur appears over the sink', 'A blur mask seems to hide the knife']}
    referring = ["but it isn't", 'but it is not', 'but it is just a static scene', 'but that is not the case',
                 'but it is actually a reflection', 'actually just a white tablecloth', 'but this is merely a shadow',
                 'actually a white tablecloth']
    other = ['but the wrist camera is just fine', 'but the grasp is simply slow', 'but the timestamps actually advance',
             'but the operator is just waiting', 'but the other view is actually fine', 'but the cup is merely hidden',
             'but the right camera really works', 'but the audio is just quiet']
    wrong = []
    for fam, sentences in impressions.items():
        for s in sentences:
            wrong += [(s, c, family(f'{s}, {c}.')) for c in referring if family(f'{s}, {c}.') != BROAD]
            wrong += [(s, c, family(f'{s}, {c}.')) for c in other if family(f'{s}, {c}.') != fam]
    assert not wrong, wrong[:10]
    # the impression's own subject again ("the camera" after "the head camera") still refers to it
    assert family('The head camera looks obstructed, but the camera is just tilted.') == BROAD


def test_withdrawing_and_searching_verbs_read_in_every_inflection():
    """The withdrawing, searching and naming verbs are given as base forms and every inflection is derived, so no form
    of one is missing ("withdrawn" was, and "the overexposure I flagged earlier is withdrawn" kept the family)."""
    from board import polarity
    for participle in sorted(polarity.VERDICTS - {'unfounded'}):
        assert family(f'Suspected glare, {participle}.') == BROAD, participle
        assert family(f'The glare flag was {participle}.') == BROAD, participle
    for form in ('retract', 'retracts', 'retracting', 'withdraw', 'withdraws', 'withdrawing', 'withdrew',
                 'dismisses', 'cancelling', 'discards'):
        assert form in polarity.WITHDRAW and family(f'{form.capitalize()} the glare flag.') == BROAD, form
    for form in sorted(polarity.SEARCH):
        assert family(f'The operator {form} for glare.') == BROAD, form
    assert family('The overexposure I flagged earlier is withdrawn.') == BROAD
    assert family('It would be wrong to term this overexposed.') == BROAD


def test_corpus_has_the_reviewed_size():
    assert len(reviewer_corpus()) == 744
    assert all(len(rows) >= 150 for rows in (wordings.WRITTEN, wordings.INDEPENDENT, wordings.THIRD, wordings.FOURTH))


# ---------------------------------------------------------------- gate

def test_gate_severity_bound_is_the_boards_rated_maximum():
    labels = {'data_issues': [{'issue': 'camera blur', 'category': 'camera_fault', 'severity': None},
                              {'issue': 'glare', 'category': 'lighting', 'severity': 'low'}]}
    for bound in ('low', 'medium', 'high'):
        met, why = judge({'max_issue_sev': bound}, labels)
        assert met and why[0].startswith('needs review:')
    labels['data_issues'][1]['severity'] = 'High'
    assert not judge({'max_issue_sev': 'medium'}, labels)[0]
    assert max_severity(labels['data_issues']) == 'high' and severity({'severity': 'unknown'}) is None


def test_gate_named_reads_the_same_polarity_as_the_families():
    labels = {'task_summary': 'The operator never sews the cloth',
              'timeline': [{'action': 'does not sew', 'object': 'cloth'}], 'completion': {'task_completed': 'failure'}}
    assert not judge({'named': ['sew']}, labels)[0]
    assert not judge({'named': ['sews']}, labels)[0]
    labels['timeline'].append({'action': 'sew', 'object': 'cloth'})
    assert judge({'named': ['sew']}, labels)[0]


# ---------------------------------------------------------------- every consumer of one label agrees

SEVS = ['low', 'medium', 'high']
WEIRD = [None, True, False, 0, -1, 1.5, 'NaN', '', ' ', [], {}, [1], {'a': 1}, 'SUCCESS', ' Success ', 'High',
         'partial', 'Partial ', 'success_then_undone', 'failure', 'unclear', 'medium', 'low', 'Aligned', 2, 0.5,
         '0.5', 'aligned ', 'broader', 'Different', 'high', ['high'], 7, 61.0, -0.2]


def _reply(rng):
    r = {
        'scene': {'objects': [{'attributes': ['red'], 'name': 'red cup', 'location': 'left'}], 'setting': 'a table'},
        'timeline_columns': ['start_s', 'end_s', 'arm', 'action', 'object', 'destination', 'spatial_relation',
                             'contribution', 'progress', 'notes'],
        'timeline': [[0.0, 5.0, 'left', 'reach', 'red cup', None, None, 'advancing', 0.1, None],
                     [5.0, 12.0, 'left', 'grasp', 'red cup', None, None, 'advancing', 0.4, 'ok'],
                     [12.0, 30.0, 'both', 'place', 'red cup', 'plate', 'onto the plate', 'advancing', 1.0, None]],
        'task_summary': 'Place the red cup on the plate.',
        'key_events': [{'t_s': 12.0, 'label': 'cup grasped', 'kind': 'contact', 'outcome': 'success', 'note': ''}],
        'state_changes': [{'t_s': 30.0, 'object': 'red cup', 'from': 'table', 'to': 'plate', 'seen_in': 'exo'}],
        'completion': {'task_completed': rng.choice(['success', 'failure', 'partial', 'unclear',
                                                     'success_then_undone']),
                       'success_predicate': 'cup on plate', 'completed_at_s': 30.0, 'goal_reached_at_s': 30.0,
                       'undone_at_s': None, 'undone_by': None, 'reason': ''},
        'goal_alignment': {'matches_given': True, 'relation': rng.choice(['aligned', 'narrower', 'broader',
                                                                          'different']), 'reason': 'same'},
        'scene_graph': [], 'instruction_variants': ['cup to plate'], 'performance_review': 'Clean.', 'recovery': [],
        'data_issues': [{'issue': 'camera blur', 'category': rng.choice(['camera_fault', 'instruction_mismatch',
                                                                         'lighting']),
                         'severity': rng.choice(SEVS), 't_s': 3.0, 'end_s': 4.0, 'evidence': 'exo 3s'}
                        for _ in range(rng.randint(0, 3))],
        'operator_mistakes': [{'issue': 'regrasp', 'category': 'failed_grasp', 'severity': rng.choice(SEVS),
                               't_s': 6.0, 'end_s': 7.0, 'recovered': True, 'deliberate': 'no', 'evidence': 'left 6s'}],
    }
    if rng.random() < 0.3:
        r['tasks'] = [{'start_s': 0.0, 'end_s': 30.0, 'task': 'place cup', 'objects': ['red cup'],
                       'outcome': rng.choice(['success', 'failure', 'partial', 'success_then_undone', 'unclear']),
                       'completed_at_s': 30.0} for _ in range(rng.randint(1, 2))]
        if rng.random() < 0.5:
            r.pop('completion')
    paths = []

    def walk(x, pre=()):
        if pre:
            paths.append(pre)
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, pre + (k,))
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, pre + (i,))
    for _ in range(rng.randint(1, 4)):
        paths.clear()
        walk(r)
        path = rng.choice(paths)
        parent = r
        for k in path[:-1]:
            parent = parent[k]
        k = path[-1]
        op = rng.random()
        if op < 0.4:
            parent[k] = copy.deepcopy(rng.choice(WEIRD))
        elif op < 0.6 and isinstance(parent[k], str):
            parent[k] = rng.choice([parent[k].upper(), ' ' + parent[k] + ' ', parent[k].title()])
        elif op < 0.75:
            parent.pop(k)
        else:
            parent[k] = rng.choice([[parent[k]], str(parent[k])])
    return r


def test_board_compare_gate_and_checks_agree_on_every_derived_value():
    rng = random.Random(20261018)
    seen = collections.Counter()
    for _ in range(400):
        labels, ok = parse_response(json.dumps(_reply(rng)))
        record = {'episode_dir': '/x/ep', 'parse_ok': True, 'labels': labels, 'config': {}}
        # a reply that lacks a timeline or an outcome is not labelled on any of them (label/completeness.py)
        if not ok or annotation_problems(record):
            continue
        seen['parsed'] += 1
        d = to_board.convert(record, 'probe')
        m = metrics.measure(labels, 'probe', {})
        board_outcome = (d.get('completion') or {}).get('task_completed') or None
        assert m['outcome'] == board_outcome
        gate_outcome = judge({'outcome': ['\x00']}, labels)[1][0].split(' ', 1)[1]
        if board_outcome:
            assert gate_outcome == board_outcome
        elif d.get('tasks'):
            assert gate_outcome == 'tasks:' + ','.join(sorted({str(t.get('outcome') or None) for t in d['tasks']}))
        relation = (d.get('goal_alignment') or {}).get('relation')
        assert judge({'align': ['\x00']}, labels)[1][0] == f'alignment {relation}'
        for key in ('data_issues', 'operator_mistakes'):
            shown = [i for i in d.get(key) or [] if i and i.get('issue')]
            assert m[key] == sum(1 for i in shown if FAMILIES.counts(key, i))
            assert m[key + '_unrated'] == sum(1 for i in shown if not FAMILIES.rated(i))
        shown = [i for i in d.get('data_issues') or [] if i and i.get('issue')]
        top = max_severity(shown)
        for bound in SEVS:
            board_view = top is None or SEVS.index(top) <= SEVS.index(bound)
            assert judge({'max_issue_sev': bound}, labels)[0] == board_view
        assert sorted(d.get('_schema_violations') or []) == sorted(m['violations'])
        assert sorted(metrics.measure(labels, None, {})['violations']) == sorted(m['violations'])
        assert (sorted(f['rule'] for f in label_consistency.check(labels, 60.0))
                == sorted(f['rule'] for f in label_consistency.check(d, 60.0)))
        for e in d.get('event_labels') or []:
            p = e.get('progress')
            assert p is None or (type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1)
    assert seen['parsed'] > 300
