"""A data issue's severity is its consequence for the footage, and its footage says what the span shows
(label/prompts.py FOOTAGE_TYPES). Footage other than task is never usable task data, so the parser holds such an issue
at medium or above (label/harness.py footage_floor), and a request records the issue schema it was asked in
(issue_schema), which the stitch carries. The issues below are real DiverseUMI data issues: a lit phone held for 9s
that a part filed as a low idle_stretch, and a tap on the recording controller."""
import copy
import json

import pytest

from label import harness, issue_groups as ig, pieces, prompts
from test_pieces import _part_result, _three_parts
from test_tasks_format import TASK, _label, _reply

PHONE = {'issue': 'Painting pauses while the right gripper is parked and the demonstrator looks at a handheld phone.',
         'category': 'idle_stretch', 'severity': 'low', 'footage': 'operator_phone', 't_s': 112.1, 'end_s': 121.1,
         'evidence': 'exo 112s to 121s'}
CONTROLLER = {'issue': 'The operator taps the recording controller with the right gripper to stop the recording.',
              'category': 'setup_change', 'severity': 'low', 'footage': 'rig_handling', 't_s': 640.2, 'end_s': 642.0,
              'evidence': 'right wrist 640s'}
BLUR = {'issue': 'The left wrist view blurs during a fast reach.', 'category': 'camera_fault', 'severity': 'low',
        'footage': 'task', 't_s': 30.0, 'end_s': 31.0, 'evidence': 'left wrist 30s'}


def parsed(*issues):
    labels, ok = harness.parse_response(json.dumps({'task_summary': 'paint the bowl',
                                                    'data_issues': copy.deepcopy(list(issues))}))
    assert ok
    return labels


# ---- the prompt ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('rig', ['teleop_arms', 'handheld_gripper', 'ego_head'])
def test_every_rig_asks_each_data_issue_for_its_footage_from_its_own_words(rig):
    text = prompts.fixed_instructions(rig)
    line = next(l for l in text.splitlines() if l.strip().startswith('"footage"'))
    offered = line.split('what the span shows: ')[1].split(', or your own')[0].split(' | ')
    assert tuple(offered) == prompts.FOOTAGE_TYPES[rig] and offered[0] == prompts.FOOTAGE_TASK
    assert ('grippers_off' in offered) == (rig == 'handheld_gripper')
    assert ('bare_hand' in offered) == (rig != 'ego_head')
    assert text.count('"footage"') == 1 and '<<' not in text
    flat = ' '.join(text.split())
    assert 'Footage other than task is never usable task data, so its severity is at least medium' in flat
    assert 'a phone-like recording controller in the operator\'s hand is rig_handling, not operator_phone' in flat
    for word in offered[1:]:
        assert f'{word}:' in flat, word


@pytest.mark.parametrize('rig', ['teleop_arms', 'handheld_gripper', 'ego_head'])
def test_severity_tiers_are_defined_by_what_happens_to_the_footage(rig):
    flat = ' '.join(prompts.fixed_instructions(rig).split())
    assert 'SEVERITY IS THE CONSEQUENCE FOR THE FOOTAGE' in flat
    assert 'high: the episode is unusable, whatever is trimmed' in flat
    assert 'medium: the span from t_s to end_s is not usable task data and is cut out' in flat
    assert 'low: cosmetic; the footage in the span is still usable task data as it is' in flat


# ---- the rule in code ----------------------------------------------------------------------------------------------

def test_a_low_issue_whose_footage_is_not_task_is_raised_to_medium():
    phone, controller, blur = parsed(PHONE, CONTROLLER, BLUR)['data_issues']
    assert (phone['severity'], phone['severity_raised_from']) == ('medium', 'low')
    assert (controller['severity'], controller['severity_raised_from']) == ('medium', 'low')
    assert blur['severity'] == 'low' and 'severity_raised_from' not in blur


def test_a_coined_footage_is_read_as_a_word_and_counts_as_not_task():
    (row,) = parsed({**PHONE, 'footage': 'Eating Lunch', 'severity': None})['data_issues']
    assert (row['footage'], row['severity'], row['severity_raised_from']) == ('eating_lunch', 'medium', None)
    (row,) = parsed({**BLUR, 'footage': ' Task '})['data_issues']
    assert (row['footage'], row['severity']) == ('task', 'low')


def test_a_medium_or_high_issue_keeps_its_severity_and_footage_that_is_no_text_is_null():
    high = {**PHONE, 'severity': 'high', 't_s': None, 'end_s': None}
    labels = parsed(high, {**BLUR, 'footage': 3})
    assert labels['data_issues'][0]['severity'] == 'high' and 'severity_raised_from' not in labels['data_issues'][0]
    assert labels['data_issues'][1]['footage'] is None and labels['data_issues'][1]['severity'] == 'low'
    assert labels['_schema_violations'] == ['data_issues row 1: footage 3 is not text, kept as null']


def test_an_issue_asked_without_footage_is_read_as_it_was_and_the_floor_runs_once():
    old = {k: v for k, v in PHONE.items() if k != 'footage'}
    (row,) = parsed(old)['data_issues']
    assert row == old
    labels = parsed(PHONE, CONTROLLER)
    assert harness.parse_response(json.dumps(labels))[0]['data_issues'] == labels['data_issues']


def test_a_labelled_episode_records_its_issue_schema_and_its_floored_issues(tmp_path, monkeypatch):
    reply = _reply([TASK], data_issues=[{**PHONE, 't_s': 0.2, 'end_s': 0.8},
                                        {k: v for k, v in BLUR.items() if k != 'footage'}])
    record, fixed = _label(tmp_path, monkeypatch, reply)
    assert record['issue_schema'] == harness.ISSUE_SCHEMA == 2 and record['parse_ok']
    phone, blur = record['labels']['data_issues']
    assert (phone['footage'], phone['severity'], phone['severity_raised_from']) == ('operator_phone', 'medium', 'low')
    assert blur['footage'] is None and blur['severity'] == 'low'
    assert 'data_issues row 1: no footage, kept as null' in record['labels']['_schema_violations']
    assert '"footage"' in fixed
    # read again by the current parser (annotation_problems, reparse), the record is unchanged and complete
    assert harness.annotation_problems(record) == []
    again, ok = harness.read_reply(json.dumps(record['labels']), record)
    assert ok and again['data_issues'] == record['labels']['data_issues']


def test_a_record_asked_before_the_issue_schema_gets_no_footage():
    reply = _reply([TASK], data_issues=[{k: v for k, v in BLUR.items() if k != 'footage'}])
    for record in ({}, {'issue_schema': 1}):
        labels, ok = harness.read_reply(json.dumps(reply), record)
        assert ok and 'footage' not in labels['data_issues'][0] and '_schema_violations' not in labels
    labels, ok = harness.read_reply(json.dumps(reply), {'issue_schema': harness.ISSUE_SCHEMA})
    assert ok and labels['data_issues'][0]['footage'] is None


# ---- stitching and grouping ----------------------------------------------------------------------------------------

def test_the_stitch_carries_the_issue_schema_only_when_every_part_was_asked_in_it(tmp_path, monkeypatch):
    _, src, parts = _three_parts(tmp_path, monkeypatch)
    contexts = [json.loads((p / 'context.json').read_text()) for p in parts]
    records = [_part_result(p, 'paint the bowl') for p in parts]
    for r in records:
        r['issue_schema'] = harness.ISSUE_SCHEMA
    records[1]['labels']['data_issues'] = [{**PHONE, 't_s': 0.2, 'end_s': 0.5, 'severity': 'medium',
                                            'severity_raised_from': 'low'}]
    stitched = pieces.stitch(src, list(zip(contexts, records)))
    assert stitched['issue_schema'] == harness.ISSUE_SCHEMA
    (phone,) = stitched['labels']['data_issues']
    assert (phone['footage'], phone['severity'], phone['severity_raised_from']) == ('operator_phone', 'medium', 'low')
    del records[2]['issue_schema']
    assert 'issue_schema' not in pieces.stitch(src, list(zip(contexts, records)))


def _answer_all(category, members, context):
    return ({'groups': [{'ids': list(range(len(members))), 'issue': 'one condition', 'why': 'test'}]},
            {'cost_usd': 0.0, 'model': 'm'})


def test_issues_whose_spans_show_different_footage_are_never_one_group():
    """Two parts each report an idle_stretch across the whole part; one span is the operator on a phone and the other
    an idle pause. Without footage they would be one condition; with it, only issues of one footage are linked."""
    whole = {'category': 'idle_stretch', 'severity': 'medium', 't_s': None, 'end_s': None}
    items = [{**whole, 'issue': 'The operator uses a phone.', 'footage': 'operator_phone', 'evidence': 'a', 'part': 1},
             {**whole, 'issue': 'The operator waits.', 'footage': 'idle', 'evidence': 'b', 'part': 2},
             {**whole, 'issue': 'The operator uses a phone again.', 'footage': 'operator_phone', 'evidence': 'c',
              'part': 3}]
    cands = ig.candidates(items, {}, 450.0)
    assert [c['rows'] for c in cands] == [[0, 2]]
    result = {'model': 'm', 'config': {'pieces': []}, 'labels': {'data_issues': copy.deepcopy(items),
                                                                  'operator_mistakes': []}}
    ig.apply(result, _answer_all)
    grouped = [x for x in result['labels']['data_issues'] if 'occurrences' in x]
    assert len(grouped) == 1 and grouped[0]['footage'] == 'operator_phone'
    assert [x['footage'] for x in result['labels']['data_issues'] if 'occurrences' not in x] == ['idle']


def test_the_check_refuses_a_group_that_mixes_footage():
    a = {**PHONE, 'part': 1, 'severity': 'medium'}
    b = {**PHONE, 'footage': 'idle', 'part': 2, 'severity': 'medium', 'evidence': 'other'}
    group = ig.grouped_issue([a, b], 'one', 'test', 'k')
    group['footage'] = 'operator_phone'
    assert ig.verify([a, b], [group]) == "a grouped issue's footage is not its occurrences' footage"
