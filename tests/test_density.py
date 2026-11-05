"""Annotation density must not fall when the timeline's rules change. The rules are held to a paired measurement on real
footage (tests/fixtures/density_current.json), so a change to them fails here until it brings its own measurement, and a
measurement that shows a drop fails too. tests/fixtures/density_pr187.json is the drop PR 187 and PR 188 caused."""
import json
from pathlib import Path

import pytest

from checks import density
from label import prompts

HERE = Path(__file__).parent / 'fixtures'
COLUMNS = ['start_s', 'end_s', 'arm', 'action', 'object', 'destination', 'spatial_relation', 'contribution', 'progress',
           'notes']


def label(steps=60, seconds=120.0):
    rows, t = [], 0.0
    width = seconds / steps
    for i in range(steps):
        rows.append([round(t, 2), round(t + width, 2), ['left', 'right', 'both'][i % 3], f'action {i}', f'object {i % 7}',
                     None, None, 'idle' if i % 11 == 0 else 'advancing', 0.1, 'a note'])
        t += width
    return {'timeline_columns': COLUMNS, 'timeline': rows,
            'key_events': [{'t_s': i * 5.0, 'label': 'x', 'kind': k}
                           for i, k in enumerate(['contact', 'grasp', 'release', 'subgoal_complete'] * 4)],
            'scene': {'objects': [{'name': 'a'}, {'name': 'b'}]}, 'tasks': [{'task': 'one'}],
            'data_issues': [{'issue': 'x', 'category': 'face_visible', 'occurrences': [{}, {}]},
                            {'issue': 'y', 'category': 'surface_glare'}],
            'state_changes': [{'a': 1}], 'operator_mistakes': []}


def every_second_step_gone(labels):
    return {**labels, 'timeline': labels['timeline'][::2]}


def second_steps_gone_and_each_written_for_both_sides(labels):
    rows = []
    for row in labels['timeline'][::2]:
        rows.append(row[:2] + ['left'] + row[3:])
        rows.append(row[:2] + ['right'] + row[3:])
    return {**labels, 'timeline': rows}


def neighbours_merged_into_a_both_step(labels):
    rows, source = [], labels['timeline']
    for a, b in zip(source[::2], source[1::2]):
        rows.append([a[0], b[1], 'both'] + a[3:])
    return {**labels, 'timeline': rows}


def test_counts_are_per_minute_and_a_side_counts_a_both_step():
    m = density.metrics(label(), 120.0)
    assert m['items_per_min'] == 30
    assert m['left_items_per_min'] == 20
    assert m['contact_per_min'] == 2 and m['grasp_per_min'] == 2 and m['release_per_min'] == 2
    assert m['issues_per_min'] == 1.5             # a group of two counts two
    assert m['issues_nonface_per_min'] == 0.5     # surface_glare holds the letters of face but not the word
    assert m['objects_per_min'] == 1
    assert abs(m['coverage_any'] - 1) < 1e-3


def test_removing_every_second_step_lowers_the_step_numbers_and_only_those():
    a = density.metrics(label(), 120.0)
    b = density.metrics(every_second_step_gone(label()), 120.0)
    assert b['items_per_min'] == a['items_per_min'] / 2
    assert b['moments_per_min'] == a['moments_per_min'] / 2
    assert b['coverage_left'] < a['coverage_left']
    assert b['text_chars_per_min'] < a['text_chars_per_min']
    assert b['key_events_per_min'] == a['key_events_per_min']


def test_a_drop_cannot_hide_behind_steps_split_per_side():
    a = density.metrics(label(), 120.0)
    b = density.metrics(second_steps_gone_and_each_written_for_both_sides(label()), 120.0)
    assert b['items_per_min'] == a['items_per_min']
    assert b['moments_per_min'] == a['moments_per_min'] / 2


def test_a_step_that_hides_two_actions_lengthens_the_median_step():
    a = density.metrics(label(), 120.0)
    b = density.metrics(neighbours_merged_into_a_both_step(label()), 120.0)
    assert b['median_step_s'] > a['median_step_s']


def episodes(n, plant=None):
    return {f'e{i}': [density.metrics((plant or (lambda x: x))(label(steps=50 + i % 20)), 120.0)] for i in range(n)}


def test_the_comparison_finds_a_planted_drop_and_no_drop_between_identical_sides():
    assert density.drops(density.compare(episodes(30), episodes(30), n_boot=500)) == []
    planted = density.compare(episodes(30), episodes(30, every_second_step_gone), n_boot=500)
    assert {'items_per_min', 'moments_per_min'} <= set(density.drops(planted))
    assert planted['key_events_per_min']['verdict'] == 'holds'
    merged = density.compare(episodes(30), episodes(30, neighbours_merged_into_a_both_step), n_boot=500)
    assert 'median_step_s' in density.drops(merged)


def test_the_comparison_refuses_different_footage_on_the_two_sides():
    with pytest.raises(ValueError):
        density.compare(episodes(3), {'other': episodes(1)['e0']}, n_boot=10)


def record(name):
    return json.loads((HERE / name).read_text())


def test_the_measured_drop_of_pr_187_and_188_fails_the_guard():
    """30 real episodes, the same model and settings on both sides, the commit before PR 187 against main after PR 188."""
    dropped = density.record_drops(record('density_pr187.json'))
    assert 'items_per_min' in dropped and 'moments_per_min' in dropped


def test_the_timeline_rules_in_use_were_measured_and_density_held():
    """A change to the timeline's rules changes the fingerprint: measure it on paired footage and replace the record."""
    current = record('density_current.json')
    assert current['prompt_sha256'] == density.prompt_fingerprint(prompts)
    assert density.record_drops(current) == []
