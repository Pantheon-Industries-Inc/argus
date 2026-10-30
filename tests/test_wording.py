"""Labels without wording about how they were made (label/wording.py), applied where results are published
(label/pieces.py stitch_run)."""
import json

import pytest

from label import pieces, wording


@pytest.mark.parametrize('text, name, want', [
    # a physical window keeps every word; the labelling's window goes
    ('Stand it at the window end of the table.', False, 'Stand it at the window end of the table.'),
    ('stand it at the window end of the table', True, 'stand it at the window end of the table'),
    ('The pour is still underway at the window end.', False, None),
    ('Pour the water, still underway at the window end', True, 'Pour the water'),
    ('Wipe the window sill with the cloth.', False, 'Wipe the window sill with the cloth.'),
    ('Place the cup near the window.', False, 'Place the cup near the window.'),
    ('Open the window and close it again.', False, 'Open the window and close it again.'),
    ('The cup stays on the table in this window.', False, 'The cup stays on the table.'),
    ('Black, open throughout the supplied window', True, 'Black, open'),
    ('The window begins with the lid already open. The right gripper lifts the cup.', False,
     'The right gripper lifts the cup.'),
    ('The activity begins before this preparation window. Repeated transfers support a single task.', False,
     'Repeated transfers support a single task.'),
    # request edges and sampling
    ('The fold is still underway at the request boundary.', False, None),
    ('The lid is released between samples.', False, 'The lid is released.'),
    ('Release inferred between samples.', False, None),
    ('Release inferred between samples', True, 'Release'),
    ('The sampled views do not resolve every extraction.', False, None),
    ('No instruction is supplied. Opposed pulling suggests disassembly.', False, 'Opposed pulling suggests disassembly.'),
    # cameras called streams, part markers
    ('The left-stream gripper closes on the cup.', False, 'The left gripper closes on the cup.'),
    ('The exo stream shows both hands.', False, 'The exo view shows both hands.'),
    ('Part 1: The demonstrator picks up blocks. Part 2: The demonstrator sorts them.', False,
     'The demonstrator picks up blocks. The demonstrator sorts them.'),
    ('665.10-690.13s (part 3): exo shows the demonstrator resting.', False, '665.10-690.13s: exo shows the demonstrator resting.'),
    # clauses between semicolons, and the forms measured on 500 published labels (2026-10-30)
    ('Grasp is inferred between samples; the cord is visibly raised afterward.', False,
     'The cord is visibly raised afterward.'),
    ('The bag and loose cards shift between sampled frames; precise contact is inferred.', False,
     'The bag and loose cards shift; precise contact is inferred.'),
    ('Placement and release occur within the sampled interval.', False, None),
    ('Most of the window is spent on repeated fitting attempts.', False,
     'Most of the recording is spent on repeated fitting attempts.'),
    ('The drawing and pencil grasp predate this supplied window.', False, None),
    ('The small assembly is still held; its final placement is outside this part.', False,
     'The small assembly is still held.'),
    ('The finished seam is not shown in this part.', False, 'The finished seam is not shown.'),
    ('This is one ongoing knitting activity spanning both request boundaries.', False, None),
    ('Fine shaping of the layered model is still continuing at the part boundary', True,
     'Fine shaping of the layered model'),
    ('Needlework continues at the supplied-window boundary; the finish is not visible.', False,
     'The finish is not visible.'),
    ('The right-fin assembly falls off early in the window and must be rebuilt.', False,
     'The right-fin assembly falls off and must be rebuilt.'),
    ('The demonstrator spends most of the window on fine wire work.', False,
     'The demonstrator spends most of the recording on fine wire work.'),
    ('Wipe most of the window with the cloth.', False, 'Wipe most of the window with the cloth.'),
    ('Blue separation is achieved, but the yellow outcome lies outside the window.', False, None),
    ('The right view ends before the other supplied views.', False, 'The right view ends before the other views.'),
    ('Completion beyond this preparation cut is unknown.', False, None),
    ('Final candidate in this window seated and released', True, 'Final candidate seated and released'),
    ('Placement timing interpolated between supplied frames.', False, None),
    ('Several consecutive supplied samples appear to repeat the same images.', False,
     'Several consecutive frames appear to repeat the same images.'),
    ('An extended inactive wait interrupts the task near the end of this window.', False,
     'An extended inactive wait interrupts the task.'),
    ('The nut is already between the tips at the window\'s first frame.', False, 'The nut is already between the tips.'),
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
    ('Part 1: pick the cup. Part 2: place the cup.', False, 'Pick the cup. Place the cup.'),
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
])
def test_normalize_text(text, name, want):
    assert wording.normalize_text(text, name=name) == want


def test_normalize_text_leaves_clean_text_unchanged_and_is_idempotent():
    for s in ('The right gripper lifts the red cup onto the shelf.', 'stack the blocks', ''):
        assert wording.normalize_text(s) == s
    once = wording.normalize_text('The window begins with the lid open. The lid is released between samples.')
    assert once == 'The lid is released.' and wording.normalize_text(once) == once


def test_normalize_labels_keeps_enums_and_argus_records_and_counts_changes():
    labels = {
        'timeline': [{'action': 'release the cup between samples', 'arm': 'left', 'contribution': 'advancing',
                      'notes': 'The window ends before the cup lands.'}],
        'tasks': [{'task': 'Stack the cups at the window end of the table', 'outcome': 'success'}],
        '_excluded': [{'reason': 'both parts labelled these seconds around the cut; kept over its own half of the window',
                       'note': 'Subsequent handling is outside this window.'}],
        'instruction_variants': ['Stack the cups', 'Stack the cups in this window'],
    }
    counts, log = {}, []
    out = wording.normalize_labels(labels, counts, log)
    assert out['timeline'][0] == {'action': 'release the cup', 'arm': 'left', 'contribution': 'advancing', 'notes': None}
    assert out['tasks'] == labels['tasks']
    assert out['_excluded'][0]['reason'] == labels['_excluded'][0]['reason']
    assert out['_excluded'][0]['note'] is None
    assert out['instruction_variants'] == labels['instruction_variants']      # task text stays as Argus wrote it
    assert counts == {'rewritten': 1, 'dropped': 2} and len(log) == 3
    assert labels['timeline'][0]['notes'] == 'The window ends before the cup lands.'     # the input is not changed


def test_a_key_event_that_is_a_parts_edge_leaves_the_list():
    labels = {'key_events': [{'label': 'Window ends with the puzzle border still incomplete', 'kind': 'phase'},
                             {'label': 'Knot work still underway at the request boundary', 'kind': 'phase'},
                             {'label': 'Window opens during screwdriver contact with the board', 'kind': 'phase'},
                             {'label': 'Release the cup at the window end of the table', 'kind': 'placement'},
                             {'label': 'Window opens fully', 'kind': 'state_change'}]}
    counts = {}
    out = wording.normalize_labels(labels, counts)
    assert out['key_events'] == labels['key_events'][3:] and counts == {'dropped': 3}


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
