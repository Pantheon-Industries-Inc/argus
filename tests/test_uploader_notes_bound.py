"""A LeRobot root's metadata reaches every episode as notes: one real folder made 282,215 values and an 89 MB board file
for a 20 minute episode, and each open spent over 45 s projecting them. The board's copy is bounded, the notes written
about the episode survive beside the dataset's, redaction still applies, and what a saved label was paid for stays."""
from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import pytest

import board.public_export as pe
from board.build import UPLOADER_CHARS_MAX, UPLOADER_VALUE_MAX, _note_shares, uploader_groups
from board.public_export import UPLOADER_ITEMS_MAX, UPLOADER_VALUE_WITHHELD, project
from prepare import formats

LEAVES = 282_215
OWN_NOTE = 'operator bumped the wrist camera'
SPAN = 'pick up the block'


def _large_notes(leaves=LEAVES) -> dict:
    """The shape prepare/formats.py lerobot_metadata writes, with the dataset's files before the episode's own notes."""
    per_feature = 12
    features = (leaves // 2) // per_feature
    rows = leaves - leaves // 2
    return {
        'recorded episode metadata': {
            'meta/info.json': {'features': {f'feature_{i}': {f'stat_{j}': i * j for j in range(per_feature)}
                                            for i in range(features)}},
            'meta/episodes/chunk-000/file-000.parquet': [{'source': 'row', 'row': 3,
                                                          'fields': {'operator_note': OWN_NOTE}}],
        },
        'unassigned dataset metadata claims': [{'column': i} for i in range(rows)],
        'recorded annotation spans': {'phase': [{'t0': 1.5, 't1': 2.5, 'label': SPAN}]},
    }


def _leaves(x) -> int:
    if isinstance(x, dict):
        return sum(map(_leaves, x.values()))
    return sum(map(_leaves, x)) if isinstance(x, list) else 1


def _items(groups) -> list:
    return [item for group in groups for item in group['items']]


def test_a_normal_recording_keeps_every_note_exactly_and_says_nothing_was_left_out():
    notes = {'table rows': [{'table': 'runs.csv', 'operator': 'A', 'success': True}],
             'notes': {'meta/info.json': {'features': {f'f{i}': {'dtype': 'float32', 'names': ['x', 'y'],
                                                                 'matrix': list(range(16))} for i in range(400)}},
                       'long list': list(range(100)), 'prose': 'p' * UPLOADER_VALUE_MAX,
                       'rows': [{'a': 1}, {'a': 2}], 'one row': [{'b': 3}]}}
    groups = uploader_groups(notes, None, 10)
    assert [g['kind'] for g in groups] == ['row', 'notes'] and not any('omitted' in g for g in groups)
    assert groups[0]['title'] == 'Row of runs.csv'
    assert [(i['name'], i['value']) for i in groups[0]['items']] == [('operator', 'A'), ('success', 'True')]
    shown = {i['name']: i['value'] for i in groups[1]['items']}
    assert len(shown) == 400 * 3 + 5
    assert shown['meta/info.json / features / f399 / names'] == 'x, y'
    assert shown['meta/info.json / features / f0 / matrix'] == ', '.join(map(str, range(16)))
    assert shown['long list'] == '100 values, 0, 1, 2, 3, ...'
    assert shown['prose'] == 'p' * UPLOADER_VALUE_MAX       # the longest value kept is kept whole
    assert shown['rows / 1 / a'] == '1' and shown['rows / 2 / a'] == '2' and shown['one row / b'] == '3'
    assert list(notes) == ['table rows', 'notes'] and notes['table rows'][0]['table'] == 'runs.csv'


@pytest.mark.parametrize('extra', [0, 1])
def test_the_bound_starts_just_past_it(extra):
    groups = uploader_groups({f'k{i}': i for i in range(UPLOADER_ITEMS_MAX + extra)}, None, 10)
    assert len(_items(groups)) == UPLOADER_ITEMS_MAX
    assert groups[0].get('omitted', 0) == extra


def test_a_recording_time_still_jumps_from_a_kept_note():
    groups = uploader_groups({'stamp': 1_000_000.5}, 1_000_000.0, 10)
    assert _items(groups)[0]['t'] == 0.5


def test_a_large_note_set_is_bounded_in_size_and_time_and_keeps_the_episodes_own_notes():
    notes = _large_notes()
    untouched = copy.deepcopy(notes)
    started = time.perf_counter()
    groups = uploader_groups(notes, None, 1200)
    public = project({'uploader_notes': groups})
    published = project({'uploader_notes': groups}, uploader_values=True)
    took = time.perf_counter() - started
    items = _items(groups)
    assert notes == untouched                    # context.json's notes are only read
    assert len(items) == UPLOADER_ITEMS_MAX
    assert LEAVES - 10 < _leaves(notes) <= LEAVES
    assert groups[0]['omitted'] == _leaves(notes) - UPLOADER_ITEMS_MAX
    assert len(json.dumps(groups)) < 1_000_000   # 89 MB before
    assert took < 10, took                       # over 45 s for one open before, on this size
    values = [i['value'] for i in items]
    # the small notes survive whichever side of the dataset's files they sit on
    assert OWN_NOTE in values and SPAN in values
    # and each large file is still represented, with neither taking the other's share
    names = [i['name'] for i in items]
    info = sum('meta/info.json' in n for n in names)
    claims = sum('unassigned dataset metadata claims' in n for n in names)
    assert abs(info - claims) <= 4 and info > 900
    # the page gets the count on both routes
    assert public['uploader_notes'][0]['omitted'] == groups[0]['omitted']
    assert published['uploader_notes'][0]['omitted'] == groups[0]['omitted']


def test_one_open_projects_each_kept_value_a_bounded_number_of_times(monkeypatch):
    calls = []
    real = pe._credential_value
    monkeypatch.setattr(pe, '_credential_value', lambda value, *a: calls.append(1) or real(value, *a))
    project({'uploader_notes': uploader_groups(_large_notes(50_000), None, 1200)}, uploader_values=True)
    assert len(calls) < 20 * UPLOADER_ITEMS_MAX


def test_a_board_built_before_the_bound_opens_quickly_without_a_rebuild():
    items = [{'name': f'a / {i}', 'value': str(i), '_uploader_field_path': ['a', str(i)]} for i in range(LEAVES)]
    started = time.perf_counter()
    public = project({'uploader_notes': [{'title': 'Notes in the files', 'kind': 'notes', 'items': items}]})
    assert time.perf_counter() - started < 10
    group = public['uploader_notes'][0]
    assert len(group['items']) == UPLOADER_ITEMS_MAX and group['omitted'] == LEAVES - UPLOADER_ITEMS_MAX
    assert {i['value'] for i in group['items']} == {UPLOADER_VALUE_WITHHELD}


def test_rows_that_overflow_leave_room_for_the_notes_and_every_value_is_accounted_for():
    rows = [{'table': 'runs.csv', 'a': i, 'b': i} for i in range(5000)]
    groups = uploader_groups({'table rows': rows, 'notes': {'note': OWN_NOTE}}, None, 10)
    assert len(_items(groups)) == UPLOADER_ITEMS_MAX
    assert sum(len(g['items']) + g.get('omitted', 0) for g in groups) == 10_001
    assert groups[-1]['kind'] == 'notes' and groups[-1]['items'][0]['value'] == OWN_NOTE
    assert groups[-2]['kind'] == 'row' and groups[-2]['omitted'] > 0


def test_total_text_is_bounded_when_each_value_is_long_but_allowed():
    notes = {f'k{i}': 'v' * UPLOADER_VALUE_MAX for i in range(1000)}
    groups = uploader_groups(notes, None, 10)
    kept = _items(groups)
    assert sum(len(i['value']) for i in kept) <= UPLOADER_CHARS_MAX
    assert len(kept) + groups[0]['omitted'] == 1000 and all(len(i['value']) == UPLOADER_VALUE_MAX for i in kept)


def test_a_value_too_long_to_keep_shows_no_part_of_itself():
    # a token straddling any cut would no longer read as a token, so nothing of the value is shown
    secret = 'hf_' + 'a1B2' * 10
    value = 'x' * UPLOADER_VALUE_MAX + secret
    groups = uploader_groups({'note': value}, None, 10)
    assert _items(groups)[0]['value'] == f'{len(value)} characters, not shown here'
    assert 'a1B2' not in json.dumps(groups) and 'xxx' not in json.dumps(groups)


def test_redaction_and_public_withholding_still_apply_to_what_is_kept():
    token = 'a' * 32
    bearer = 'Bearer ' + 'c0d3' * 10
    private = 'https://box.tailnet.ts.net/data'
    public_note = 'public words only'
    notes = _large_notes(20_000)
    notes['kaggle'] = {'key': token}
    notes['api_key'] = 'b' * 32
    notes['header'] = bearer
    notes['where'] = private
    notes['said'] = public_note
    groups = uploader_groups(notes, None, 10)
    assert {'kaggle / key', 'api_key', 'header', 'where', 'said'} <= {i['name'] for i in _items(groups)}
    published = project({'uploader_notes': groups}, uploader_values=True)
    said = json.dumps(published)
    for secret in (token, 'b' * 32, bearer, private):
        assert secret not in said
    assert public_note in said
    public = project({'uploader_notes': groups})
    said = json.dumps(public)
    for secret in (token, 'b' * 32, bearer, private, public_note, OWN_NOTE, SPAN):
        assert secret not in said
    assert {i['value'] for i in public['uploader_notes'][0]['items']} == {UPLOADER_VALUE_WITHHELD}
    assert '_uploader_field_path' not in said


def test_a_secret_past_the_bound_is_not_in_the_board_at_all():
    secret = 'b' * 32
    notes = {f'k{i}': i for i in range(UPLOADER_ITEMS_MAX)}
    notes['zz'] = secret
    groups = uploader_groups(notes, None, 10)
    assert secret not in json.dumps(groups) and groups[0]['omitted'] == 1


def test_a_deep_path_is_read_in_linear_work(monkeypatch):
    calls = []
    real = pe._name
    monkeypatch.setattr(pe, '_name', lambda key: calls.append(1) or real(key))
    depth = 4000
    path = [f'p{i}' for i in range(depth)]
    pe._uploader_item({'name': ' / '.join(path), 'value': 'x', '_uploader_field_path': path})
    assert len(calls) < 6 * depth        # about depth * depth / 2 before


@pytest.mark.parametrize('path,value,kept', [
    # below a source record's fields is the publisher's own schema, so 'token' is an ordinary word there...
    (['meta/a.parquet', 'fields', 'token'], 'plain', True),
    (['meta/a.parquet', 'fields', 'tokens'], 'plain', True),
    # ...except under a provider's owner
    (['meta/a.parquet', 'fields', 'openai', 'token'], 'plain', False),
    # above any fields, the access words are access words
    (['meta/a.parquet', 'token'], 'plain', False),
    (['token', 'fields', 'x'], 'plain', False),
    (['password'], 'plain', False),
    (['fields', 'password'], 'plain', False),
    (['a', 'fields', 'b', 'fields', 'tokens'], 'plain', True),
    # the 'fields' part itself is native, so its own name is read as a native one
    (['fields'], 'plain', True),
])
def test_the_native_boundary_is_where_it_was(path, value, kept):
    out = pe._uploader_item({'name': ' / '.join(path), 'value': value, '_uploader_field_path': path})
    assert (out['value'] == value) is kept, out


@pytest.mark.parametrize('path,name', [
    # a private source key is renamed above the native boundary, and on the boundary part itself
    (['recorded_metadata', '/home/op/run/a.json'], 'recorded_metadata / a.json'),
    (['recorded_metadata', '/home/op/run/a.json', 'fields', 'x'], 'recorded_metadata / a.json / fields / x'),
    # a part below fields is the publisher's and is left as written
    (['recorded_metadata', 'fields', 'recorded_metadata', '/home/op/run/a.json'],
     'recorded_metadata / fields / recorded_metadata / /home/op/run/a.json'),
])
def test_source_names_are_shown_where_they_were(path, name):
    assert pe._uploader_item({'name': 'x', 'value': 'v', '_uploader_field_path': path})['name'] == name


def test_a_part_named_fields_is_still_read_as_a_source_name_and_a_last_one_opens_no_native_schema():
    # '/fields' reads as the word 'fields', and as the part that starts the native schema it is still renamed
    item = pe._uploader_item({'name': 'x', 'value': 'v', '_uploader_field_path': ['recorded_metadata', '/fields']})
    assert item['name'] == 'recorded_metadata / fields'
    assert pe._source_name('/fields') == 'fields'
    # a last part named fields has no fields below it, so a private location in its value is still shortened
    path = ['file attribute url', 'fields']
    item = pe._uploader_item({'name': 'x', 'value': '/home/op/run/a.json', '_uploader_field_path': path})
    assert item['value'] == 'a.json'


def test_a_credential_in_a_value_is_withheld_by_the_item_itself():
    item = pe._uploader_item({'name': 'x', 'value': 'Bearer ' + 'c0d3' * 10, '_uploader_field_path': ['note']})
    assert item['value'] == 'withheld'


def test_shares_keep_small_subtrees_whole_and_never_exceed_the_budget():
    assert _note_shares([1, 1_000_000, 3], 100) == [1, 96, 3]
    assert _note_shares([1, 2, 3], 6) == [1, 2, 3]
    assert _note_shares([500, 500], 101) == [51, 50]
    assert _note_shares([1] * 10, 3) == [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    assert _note_shares([], 5) == []
    assert _note_shares([10, 0], 5) == [5, 0]


def test_the_page_says_how_many_it_left_out():
    page = (Path(__file__).resolve().parents[1] / 'board' / 'index.html').read_text()
    assert '${g.omitted ? ' in page and 'more recorded values are not shown' in page


# What a saved, paid label is proved from: bounding the board's copy must change none of it.
def test_the_prompt_text_and_context_notes_are_what_they_were():
    notes = _large_notes(5_000)
    ctx = {}
    formats.set_uploader_notes(ctx, notes)
    assert ctx['uploader_notes'] == notes            # context.json keeps every note
    # the digest of the prompt's text on the commit before the bound: a change here relabels paid work
    assert hashlib.sha256(ctx['uploader_annotation'].encode()).hexdigest() == PROMPT_TEXT_SHA256
    before = json.dumps(ctx, sort_keys=True)
    uploader_groups(ctx['uploader_notes'], None, 10)
    assert json.dumps(ctx, sort_keys=True) == before


def test_a_saved_label_on_a_large_note_set_survives_a_code_only_change(tmp_path, monkeypatch):
    """A changed request module changes request_semantics_sha256 (board/public_export.py is one). A resumed review then
    keeps a paid label only if the request rebuilt from the same context gives the same fingerprint."""
    from label import harness, request_attestation
    from test_label import _packed_episode
    ep, _ = _packed_episode(tmp_path)
    path = Path(ep) / 'context.json'
    ctx = json.loads(path.read_text())
    formats.set_uploader_notes(ctx, _large_notes(20_000))
    path.write_text(json.dumps(ctx))
    settings = dict(model='openai/gpt-6.1-sol', reasoning='high', max_tokens=2000, cell_w=224, sample_fps=2)
    saved = harness.label_episode(ep, tmp_path / 'label.json', api_key='dry-run', timeout=60, dry_run=True, **settings)
    context_bytes = path.read_bytes()
    uploader_groups(ctx['uploader_notes'], ctx.get('clock_start_s'), 10)
    assert path.read_bytes() == context_bytes
    current = harness.input_identity(ep, example_dir=None, **settings)
    assert harness.same_input_identity(saved['input_identity'], current)
    current['settings']['request_semantics_sha256'] = '0' * 64
    assert not harness.same_input_identity(saved['input_identity'], current)
    assert request_attestation.matches(saved, current, ep, tmp_path)
    # and the attestation can fail: a different note in the prompt is a different request
    ctx['uploader_annotation'] = 'another note'
    path.write_text(json.dumps(ctx))
    assert not request_attestation.matches(saved, harness.input_identity(ep, example_dir=None, **settings), ep, tmp_path)


PROMPT_TEXT_SHA256 = '3cb9ca6665d6f470ab9a813204269bddd7311f8e236e9d426ed7274ce6e277fb'
