"""Labels without wording about how they were made (label/wording.py), applied where results are published
(label/pieces.py stitch_run)."""
import json

import pytest

from label import pieces, wording


@pytest.mark.parametrize('text, name, want', [
    # a physical window keeps every word; the labelling's window goes
    ('Stand it at the window end of the table.', False, 'Stand it at the window end of the table.'),
    ('stand it at the window end of the table', True, 'stand it at the window end of the table'),
    ('The pour is still underway at the window end.', False, 'The pour is still underway at the window end.'),
    ('Pour the water, still underway at the window end', True, 'Pour the water, still underway at the window end'),
    ('Wipe the window sill with the cloth.', False, 'Wipe the window sill with the cloth.'),
    ('Place the cup near the window.', False, 'Place the cup near the window.'),
    ('Open the window and close it again.', False, 'Open the window and close it again.'),
    ('The cup stays on the table in this window.', False, 'The cup stays on the table.'),
    ('Black, open throughout the supplied window', True, 'Black, open'),
    ('The window begins with the lid already open. The right gripper lifts the cup.', False,
     'The window begins with the lid already open. The right gripper lifts the cup.'),
    ('The activity begins before this preparation window. Repeated transfers support a single task.', False,
     'The activity begins before this preparation window. Repeated transfers support a single task.'),
    # request edges and sampling
    ('The fold is still underway at the request boundary.', False, 'The fold is still underway at the request boundary.'),
    ('The lid is released between samples.', False, 'The lid is released.'),
    ('Release inferred between samples.', False, 'Release.'),
    ('Release inferred between samples', True, 'Release'),
    ('The sampled views do not resolve every extraction.', False, 'The sampled views do not resolve every extraction.'),
    ('The sampled views do not resolve every instant.', False, None),
    ('No instruction is supplied. Opposed pulling suggests disassembly.', False, 'Opposed pulling suggests disassembly.'),
    # cameras called streams, part markers
    ('The left-stream gripper closes on the cup.', False, 'The left gripper closes on the cup.'),
    ('The exo stream shows both hands.', False, 'The exo view shows both hands.'),
    ('Left and right streams at 335.13 s show the needle.', False, 'Left and right views at 335.13 s show the needle.'),
    ('Part 1: The demonstrator picks up blocks. Part 2: The demonstrator sorts them.', False,
     'Part 1: The demonstrator picks up blocks. Part 2: The demonstrator sorts them.'),
    ('665.10-690.13s (part 3): exo shows the demonstrator resting.', False, '665.10-690.13s: exo shows the demonstrator resting.'),
    # clauses between semicolons, and the forms measured on 500 published labels (2026-10-30)
    ('Grasp is inferred between samples; the cord is visibly raised afterward.', False,
     'Grasp; the cord is visibly raised afterward.'),
    ('The bag and loose cards shift between sampled frames; precise contact is inferred.', False,
     'The bag and loose cards shift; precise contact is inferred.'),
    ('Placement and release occur within the sampled interval.', False, 'Placement and release.'),
    ('Most of the window is spent on repeated fitting attempts.', False,
     'Most of the window is spent on repeated fitting attempts.'),
    ('The drawing and pencil grasp predate this supplied window.', False,
     'The drawing and pencil grasp predate this supplied window.'),
    ('The small assembly is still held; its final placement is outside this part.', False,
     'The small assembly is still held; its final placement is outside this part.'),
    ('The finished seam is not shown in this part.', False, 'The finished seam is not shown.'),
    ('This is one ongoing knitting activity spanning both request boundaries.', False,
     'This is one ongoing knitting activity spanning both request boundaries.'),
    ('Fine shaping of the layered model is still continuing at the part boundary', True,
     'Fine shaping of the layered model is still continuing at the part boundary'),
    ('Needlework continues at the supplied-window boundary; the finish is not visible.', False,
     'Needlework continues at the supplied-window boundary; the finish is not visible.'),
    ('The right-fin assembly falls off early in the window and must be rebuilt.', False,
     'The right-fin assembly falls off early in the window and must be rebuilt.'),
    ('The demonstrator spends most of the window on fine wire work.', False,
     'The demonstrator spends most of the window on fine wire work.'),
    ('Wipe most of the window with the cloth.', False, 'Wipe most of the window with the cloth.'),
    ('Blue separation is achieved, but the yellow outcome lies outside the window.', False,
     'Blue separation is achieved, but the yellow outcome lies outside the window.'),
    ('The right view ends before the other supplied views.', False, 'The right view ends before the other views.'),
    ('Completion beyond this preparation cut is unknown.', False, 'Completion beyond this preparation cut is unknown.'),
    ('Final candidate in this window seated and released', True, 'Final candidate seated and released'),
    ('Placement timing interpolated between supplied frames.', False, 'Placement.'),
    ('Several consecutive supplied samples appear to repeat the same images.', False,
     'Several consecutive frames appear to repeat the same images.'),
    ('An extended inactive wait interrupts the task near the end of this window.', False,
     'An extended inactive wait interrupts the task near the end of this window.'),
    ('The nut is already between the tips at the window\'s first frame.', False,
     'The nut is already between the tips at the window\'s first frame.'),
    # a real window, stream, sample or part keeps its words (review of PR 159)
    ('The arm wipes the window during the window cleaning task.', False,
     'The arm wipes the window during the window cleaning task.'),
    ('Wipe the window until the window is clean.', False, 'Wipe the window until the window is clean.'),
    ('The cloth is still held at the window edge.', False, 'The cloth is still held at the window edge.'),
    ('Tape held at the window edge', True, 'Tape held at the window edge'),
    ('The robot places the left part between the two samples.', False,
     'The robot places the left part between the two samples.'),
    ('Place the samples in the tray; the samples show color differences.', False,
     'Place the samples in the tray; the samples show color differences.'),
    ('Insert the part at the part cut line.', False, 'Insert the part at the part cut line.'),
    ('No task was given to the second operator.', False, 'No task was given to the second operator.'),
    ('The cups are not supplied.', False, 'The cups are not supplied.'),
    ('The left stream of water hits the cup.', False, 'The left stream of water hits the cup.'),
    ('Part 1: pick the cup. Part 2: place the cup.', False, 'Part 1: pick the cup. Part 2: place the cup.'),
    ('Clean the window at the start of the window cleaning task.', False,
     'Clean the window at the start of the window cleaning task.'),
    ('Compare fabric samples between samples of cloth.', False, 'Compare fabric samples between samples of cloth.'),
    ('The sampled views show the cup moved.', False, 'The sampled views show the cup moved.'),
    # a physical part or clip is not the labelling's
    ('Subsequent work removes, handles and reseats this part.', False,
     'Subsequent work removes, handles and reseats this part.'),
    ('approach and grasp next part', True, 'approach and grasp next part'),
    ('Pick up this part and attach it.', False, 'Pick up this part and attach it.'),
    ('The pink part ends up on the table.', False, 'The pink part ends up on the table.'),
    ('Clip the towel; the clip opens and closes.', False, 'Clip the towel; the clip opens and closes.'),
    ('Sort the fabric samples into the bin.', False, 'Sort the fabric samples into the bin.'),
    # argus-wording-1 dropped or cut each of these with a physical fact in it (audit on 1,000 published lines,
    # 2026-10-30); a clause goes only when nothing but the labelling is in it, and an event it names is kept
    ('Inferred between samples: the bolt is at the opposing tips before and lies unsupported on the mat at 302.30 s.',
     False, 'Inferred: the bolt is at the opposing tips before and lies unsupported on the mat at 302.30 s.'),
    ('The fall is inferred between samples: the pencil is across the jaws at 131.13 s and loose on the sheet.', False,
     'The fall is inferred: the pencil is across the jaws at 131.13 s and loose on the sheet.'),
    ('The window begins during ongoing sewing, with several buttons already on the cloth.', False,
     'The window begins during ongoing sewing, with several buttons already on the cloth.'),
    ('Work began before this window, and transparent tape makes individual strip boundaries difficult to distinguish.',
     False, 'Work began before this window, and transparent tape makes individual strip boundaries difficult to '
     'distinguish.'),
    ('There are two phone-use pauses, but the sampled views do not establish a definite accidental drop.', False,
     'There are two phone-use pauses, but the sampled views do not establish a definite accidental drop.'),
    ('The final supplied frame still shows the cords held and tensioned; a completed knot is not established in this '
     'part.', False, 'The final supplied frame still shows the cords held and tensioned; a completed knot is not '
     'established.'),
    ('The separation appears unintended, although the exact contact falls between samples.', False,
     'The separation appears unintended, although the exact contact falls between samples.'),
    ('Release is inferred between samples and is followed by a regrasp.', False,
     'Release is inferred and is followed by a regrasp.'),
    ('Tool release is inferred between samples.', False, 'Tool release is inferred.'),
    ('Manipulation continues at the preparation cut.', False, 'Manipulation continues.'),
    ('Contact ends between sampled frames.', False, 'Contact ends.'),
    ('Release is bounded by these samples.', False, 'Release.'),
    ('Initial folding occurred before this supplied window; the first placement follows.', False,
     'Initial folding occurred before this supplied window; the first placement follows.'),
    ('Connection is not shown before this preparation cut.', False, 'Connection is not shown before this preparation cut.'),
    ('Embroidery is still being worked at the window end', True, 'Embroidery is still being worked at the window end'),
    ('The cloth is left-stream-supported.', False, 'The cloth is left-stream-supported.'),
    ('The hole is visible in the part.', False, 'The hole is visible in the part.'),
    ('Attach the bracket (part 2) to the frame.', False, 'Attach the bracket (part 2) to the frame.'),
    ('The pour is still underway at the internal request boundary.', False, 'The pour is still underway.'),
    ('Cup held in this window.', False, 'Cup held.'),
    # review of the under-cut rules: a negation, a real window after "from" or "through", a time after the samples,
    # a real part, a part number
    ('Release does not occur between samples.', False, 'Release does not occur between samples.'),
    ('Release is not bounded by these samples.', False, 'Release is not bounded.'),
    ('Grasp occurs between samples; release is inferred between samples.', False, 'Grasp; release.'),
    ('The cat jumps from this window.', False, 'The cat jumps from this window.'),
    ('Light comes in through that window onto the table.', False, 'Light comes in through that window onto the table.'),
    ('The arm reaches within the window and grabs the cup.', False, 'The arm reaches within the window and grabs the cup.'),
    ('Release occurs between samples at 3.2 s and 4.1 s.', False, 'Release occurs between samples at 3.2 s and 4.1 s.'),
    ('The lid is released between samples at 10 s and 11 s.', False,
     'The lid is released between samples at 10 s and 11 s.'),
    ('Spare part is not provided', True, 'Spare part is not provided'),
    ('The last part is not provided.', False, 'The last part is not provided.'),
    ('Attach bracket 3 (part 2) to the frame.', False, 'Attach bracket 3 (part 2) to the frame.'),
    ('Cup lifted.\nRelease between samples.\nCup placed.', False, 'Cup lifted.\nRelease.\nCup placed.'),
    ('iPad stays on the table. No instruction is supplied.', False, 'iPad stays on the table.'),
    ('The pickup-to-transport transition is partly between samples.', False,
     'The pickup-to-transport transition is partly between samples.'),
    # names argus-wording-1 garbled ("Final hair-fitting pass in retain connection", "remains loose at the")
    ('Final hair-fitting pass in this window does not retain connection', True,
     'Final hair-fitting pass does not retain connection'),
    ('A further working loop remains loose at the request boundary', True,
     'A further working loop remains loose at the request boundary'),
    ('Much of the later window consists of repeated fitting attempts.', False,
     'Much of the later window consists of repeated fitting attempts.'),
    # "preparation" and "cut" are the labelling's only as its cuts and windows
    ('The window ends during preparation; no assembled column is shown.', False,
     'The window ends during preparation; no assembled column is shown.'),
    ('The cut is inferred between samples.', False, 'The cut is inferred.'),
    ('This is an internal preparation cut. The cup is lifted.', False, 'The cup is lifted.'),
    # a share of the part is a time, and events named inside the samples phrase are events
    ('The cup is held for most of the window.', False, 'The cup is held for most of the window.'),
    ('The lid stays shut through the rest of the window.', False, 'The lid stays shut through the rest of the window.'),
    ('The cup is visible in many sampled frames.', False, 'The cup is visible in many sampled frames.'),
    ('Release inferred within the sampled placement interval.', False,
     'Release inferred within the sampled placement interval.'),
    ('Short contact and lift are bounded by the final sampled frames.', False, 'Short contact and lift.'),
    ('The release itself is not sampled.', False, 'The release itself is not sampled.'),
])
def test_normalize_text(text, name, want):
    assert wording.normalize_text(text, name=name) == want


def test_normalize_text_leaves_clean_text_unchanged_and_is_idempotent():
    for s in ('The right gripper lifts the red cup onto the shelf.', 'stack the blocks', ''):
        assert wording.normalize_text(s) == s
    once = wording.normalize_text('No instruction is supplied. The lid is released between samples.')
    assert once == 'The lid is released.' and wording.normalize_text(once) == once
    kept = 'The window begins with the lid open.'          # a state at the part's start: at the recording's, a fact
    assert wording.normalize_text(kept) == kept


@pytest.mark.parametrize('text, want', [
    ('Most of the window is spent on repeated fitting attempts.',
     'Most of the recording is spent on repeated fitting attempts.'),
    ('The demonstrator spends most of the window on fine wire work.',
     'The demonstrator spends most of the recording on fine wire work.'),
    ('Part 1: pick the cup between samples.', 'Pick the cup.'),
    ('Wipe most of this window with the cloth.', 'Wipe most of this window with the cloth.'),
    ('Most of the window is covered in soap.', 'Most of the window is covered in soap.'),
    ('The squeegee took most of the window\'s grime.', 'The squeegee took most of the window\'s grime.'),
])
def test_a_recording_labelled_in_one_part_reads_its_window_as_the_recording(text, want):
    assert wording.normalize_text(text, whole=True) == want
    if 'recording' in want or 'Pick' in want:
        assert wording.normalize_text(text) != want             # a stitched part's window is not the recording


def test_normalize_labels_keeps_enums_and_argus_records_and_counts_changes():
    labels = {
        'timeline': [{'action': 'release the cup between samples', 'arm': 'left', 'contribution': 'advancing',
                      'notes': 'The window ends before the cup lands. No instruction is supplied.'}],
        'tasks': [{'task': 'Stack the cups at the window end of the table', 'outcome': 'success'}],
        '_excluded': [{'reason': 'both parts labelled these seconds around the cut; kept over its own half of the window',
                       'note': 'Release is inferred between the sampled frames.'}],
        'instruction_variants': ['Stack the cups', 'Stack the cups in this window'],
    }
    counts, log = {}, []
    out = wording.normalize_labels(labels, counts, log)
    assert out['timeline'][0] == {'action': 'release the cup', 'arm': 'left', 'contribution': 'advancing',
                                  'notes': 'The window ends before the cup lands.'}
    assert out['tasks'] == labels['tasks']
    assert out['_excluded'][0]['reason'] == labels['_excluded'][0]['reason']
    assert out['_excluded'][0]['note'] == 'Release.'
    assert out['instruction_variants'] == labels['instruction_variants']      # task text stays as Argus wrote it
    assert counts == {'rewritten': 3} and len(log) == 3
    assert labels['timeline'][0]['notes'].endswith('No instruction is supplied.')     # the input is not changed
    assert wording.normalize_labels({'notes': 'No instruction is supplied.'}, counts) == {'notes': None}
    assert counts == {'rewritten': 3, 'dropped': 1}


def test_a_key_event_that_is_only_a_parts_edge_leaves_the_list():
    labels = {'key_events': [{'label': 'Window ends', 'kind': 'phase'},
                             # a state at the edge: at the recording's own edge it is a fact, so the event stays
                             {'label': 'Window ends with the puzzle border still incomplete', 'kind': 'phase'},
                             {'label': 'Knot work still underway at the request boundary', 'kind': 'phase'},
                             {'label': 'Window opens during screwdriver contact with the board', 'kind': 'phase'},
                             {'label': 'Release the cup at the window end of the table', 'kind': 'placement'},
                             {'label': 'Window opens fully', 'kind': 'state_change'}]}
    counts = {}
    out = wording.normalize_labels(labels, counts)
    assert out['key_events'] == labels['key_events'][1:] and counts == {'dropped': 1}


def test_normalize_result_records_the_version():
    r = wording.normalize_result({'labels': {'performance_review': 'The towel is folded in this window.',
                                             'task_summary': 'Fold the towel in this window.'}})
    assert r['labels'] == {'performance_review': 'The towel is folded.', 'task_summary': 'Fold the towel in this window.'}
    assert r['text_normalizer'] == {'version': wording.VERSION, 'rewritten': 1, 'dropped': 0}


def test_stitch_run_publishes_short_replies_normalized_and_keeps_the_saved_reply(tmp_path):
    """A short episode's published reply carries normalized labels and the normalizer version; the reply saved under
    run/out is left as the model's, and a reply with no labels is copied as it is."""
    src = tmp_path / 'job' / 'run' / 'out'
    src.mkdir(parents=True)
    reply = {'parse_ok': True, 'labels': {'performance_review': 'Part 1: The cup is placed between samples.'}}
    (src / 'episode_000000.json').write_text(json.dumps(reply))
    (src / 'failed_episode_000001.json').write_text(json.dumps({'finish_reason': 'length'}))
    out = tmp_path / 'final' / 'out'
    pieces.stitch_run(tmp_path / 'job', tmp_path / 'eps', {}, out)
    got = json.loads((out / 'episode_000000.json').read_text())
    assert got['labels']['performance_review'] == 'The cup is placed.'
    assert got['text_normalizer']['version'] == wording.VERSION
    assert json.loads((src / 'episode_000000.json').read_text()) == reply
    assert json.loads((out / 'failed_episode_000001.json').read_text()) == {'finish_reason': 'length'}
