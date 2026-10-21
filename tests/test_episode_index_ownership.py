"""Episode-owned task references cannot become dataset-wide annotation dictionaries."""
from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import pytest

from prepare import annotations, formats
from prepare.lerobot_labels import scoped_index_tables
from test_habit_generic import native, source_hashes


SAMPLES = json.loads((Path(__file__).parent / 'fixtures/episode_index_ownership.json').read_text())
COLUMNS = ['task_index', 'low_level_task_index', 'human_role_subtask_index']


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def test_large_rich_episode_catalog_does_not_poison_named_task_tables(tmp_path):
    # A large controlled catalog must stay episode-owned instead of becoming per-frame claims.
    catalog_size = 12_000
    rows = [dict(deepcopy(SAMPLES['episodes.jsonl'][i % 3]), episode_index=i) for i in range(catalog_size)]
    write_rows(tmp_path / 'episodes.jsonl', rows)
    for name in ['tasks.jsonl', 'subtasks.jsonl', 'human_subtasks.jsonl']:
        write_rows(tmp_path / name, SAMPLES[name])
    before = source_hashes(tmp_path)
    tables = scoped_index_tables(tmp_path, COLUMNS, formats.read_jsonl)
    labels = annotations.resolved_labels(tables, 'task_index')
    assert labels[11]['label'] == 'Arrange the foam blocks on the practice tray.'
    assert set(labels) == {11, 17, 23}
    assert all(len(label['claims']) == 1 for label in labels.values())
    assert {Path(path).name for _, path in tables} == {'tasks.jsonl', 'subtasks.jsonl', 'human_subtasks.jsonl'}
    owned, unassigned = annotations.episode_metadata(tmp_path, formats.read_jsonl)
    assert len(owned) == catalog_size and not unassigned
    assert owned[0][str(tmp_path / 'episodes.jsonl')][0]['fields'] == rows[0]
    notes = owned[0][str(tmp_path / 'episodes.jsonl')][0]['fields']['notes']
    assert notes['reviewed'] is False and type(notes['counter']) is int
    assert notes['counter'] == 9007199254740993
    assert owned[catalog_size - 1][str(tmp_path / 'episodes.jsonl')][0]['fields'] == rows[-1]
    assert source_hashes(tmp_path) == before


def test_rich_owned_metadata_and_530_frame_annotations_keep_distinct_sources(tmp_path):
    root = tmp_path / 'upload'
    native(root)
    recording = SAMPLES['controlled_recording']
    write_rows(root / 'meta/episodes.jsonl', SAMPLES['episodes.jsonl'] + [recording['episode']])
    for name, rows in recording['tables'].items():
        write_rows(root / 'meta' / name, rows)
    before = source_hashes(root)
    parsed = formats.read_root(root, '')
    frames = pd.DataFrame(recording['columns']).astype(recording['column_dtypes'])
    assert frames['timestamp'].dtype == 'float32'
    assert all(frames[column].dtype == 'int64' for column in COLUMNS)
    ctx = {'episode_index': 904, 'fps': 10, 'n_state_frames': 530}
    data_path = root / 'data/chunk-000/episode_000904.parquet'
    formats.lerobot_annotations(ctx, parsed, frames, data_path)
    assert not ctx['annotation_unresolved']
    assert ctx['instruction'] == 'Arrange the foam blocks on the practice tray.'
    assert ctx['publisher_labels']['task_status'] == 'recovered'
    path = str(root / 'meta/episodes.jsonl')
    assert ctx['recorded_metadata'][path][0]['fields'] == recording['episode']
    assert len(ctx['recorded_metadata'][path]) == 1
    assert path not in ctx['annotation_tables']
    for column, name, ranges in [
            ('task_index', 'tasks.jsonl', [(0, 529)]),
            ('low_level_task_index', 'subtasks.jsonl', [(0, 172), (173, 280), (281, 529)]),
            ('human_role_subtask_index', 'human_subtasks.jsonl', [(0, 128), (129, 241), (242, 366), (367, 529)])]:
        # the task code over every row names the episode's task, given once as its instruction, so its recorded span
        # is kept as a task claim and never shown as a step (prepare/formats.py lerobot_annotations)
        spans = [span for span in ctx['annotation_subtasks' if column != 'task_index' else 'annotation_task_claims']
                 if span['column'] == column]
        assert [(span['row_start'], span['row_end']) for span in spans] == ranges
        assert [span['label'] for span in spans] == [row['task'] for row in recording['tables'][name]]
        assert all(span['source'] == str(root / 'meta' / name) for span in spans)
        assert all(len(span['claims']) == 1 for span in spans)
        assert spans[-1]['raw_times'][-1] == 52.900001525878906 and 't1' not in spans[-1]
        for span, (first, last) in zip(spans, ranges):
            assert span['raw_times'] == recording['columns']['timestamp'][first:last + 1]
            assert span['t0'] == pytest.approx(first / 10)
            if last < 529:
                assert span['t1'] == pytest.approx((last + 1) / 10)
    assert path not in ctx['represented_source_members'].get('json', {})
    assert len(json.dumps(ctx)) < 200_000
    assert source_hashes(root) == before


@pytest.mark.parametrize('owner', [0, None, True, 0.5, 'broken'])
@pytest.mark.parametrize('suffix', ['jsonl', 'parquet'])
def test_custom_mixed_table_preserves_owned_rows_without_global_label_leakage(tmp_path, owner, suffix):
    path = tmp_path / ('custom_codes.' + suffix)
    row = {'episode_index': owner, 'phase_index': 7, 'label': 'episode-only phase',
           'notes': {'flag': False, 'counter': 9007199254740993}, 'task_index': 0}
    global_row = {'phase_index': 7, 'label': 'recorded global phase'}
    if suffix == 'jsonl':
        write_rows(path, [row, global_row])
    else:
        # Parquet columns establish an episode ownership field for every physical row.
        global_row.update(episode_index=None, notes={'flag': True, 'counter': 1}, task_index=0)
        pd.DataFrame([row, global_row]).to_parquet(path)
    tables = annotations.index_tables(tmp_path, ['phase_index', 'task_index'], formats.read_jsonl)
    labels = annotations.resolved_labels(tables, 'phase_index')
    if suffix == 'jsonl':
        assert labels[7]['label'] == 'recorded global phase'
        assert len(labels[7]['claims']) == 1 and labels[7]['claims'][0]['row'] == 1
    else:
        assert not labels
    owned, unassigned = annotations.episode_metadata(tmp_path, formats.read_jsonl)
    if type(owner) is int:
        assert owned[0][str(path)][0]['fields'] == row
    else:
        assert not owned
        value = unassigned[0]['fields']['episode_index']
        assert pd.isna(value) if suffix == 'parquet' and owner is None else value == owner
        assert unassigned[0]['fields']['label'] == 'episode-only phase'
    assert (not unassigned) if suffix == 'jsonl' and type(owner) is int else bool(unassigned)


def test_episode_catalog_rows_without_a_valid_owner_never_gain_global_ownership(tmp_path):
    path = tmp_path / 'episodes.jsonl'
    rows = [{'task_index': 0, 'task': 'not a global dictionary'},
            {'episode_index': None, 'task_index': 0, 'task': 'unknown owner'}]
    write_rows(path, rows)
    tables = scoped_index_tables(tmp_path, ['task_index'], formats.read_jsonl)
    assert not annotations.resolved_labels(tables, 'task_index')
    owned, unassigned = annotations.episode_metadata(tmp_path, formats.read_jsonl)
    assert not owned and [claim['fields'] for claim in unassigned] == rows


def test_conflicting_global_claims_stay_unresolved_after_episode_partition(tmp_path):
    write_rows(tmp_path / 'tasks.jsonl', SAMPLES['tasks.jsonl'])
    other = tmp_path / 'custom_codes.jsonl'
    write_rows(other, [dict(SAMPLES['episodes.jsonl'][0]), {'task_index': 11, 'task': 'competing controlled task'}])
    tables = scoped_index_tables(tmp_path, ['task_index'], formats.read_jsonl)
    labels = annotations.resolved_labels(tables, 'task_index')
    assert labels[11]['label'] is None
    assert len(labels[11]['claims']) == 2
    spans, unresolved = annotations.annotation_spans([11, 11], [0, 1], labels, 'recording', 'task_index')
    assert not spans and len(unresolved) == 2
    assert all(len(claim['claims']) == 2 for claim in unresolved)
