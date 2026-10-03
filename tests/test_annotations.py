"""Recorded table ownership and annotation intervals never invent task labels or time boundaries."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prepare import formats as f
from test_formats import _jpeg, _lerobot, _lerobot_v3


def annotations():
    assert importlib.util.find_spec('prepare.annotations') is not None, 'recorded annotation interface is absent'
    from prepare import annotations as module
    return module


def test_task_transitions_keep_the_recorded_clock_and_attribution():
    spans, unresolved = annotations().annotation_spans(
        [0, 0, 1, 1], [2, 3, 4, 5], {0: 'pick', 1: 'place'}, 'meta/tasks.jsonl', 'task_index',
        final_boundary=6, frame_indices=[0, 1, 2, 3])
    assert not unresolved
    assert [(x['t0'], x['t1'], x['label']) for x in spans] == [(2, 4, 'pick'), (4, 6, 'place')]
    assert all(x['source'] == 'meta/tasks.jsonl' and x['column'] == 'task_index' for x in spans)
    ctx = {'clock_zero_s': 2, 'annotation_subtasks': spans}
    shown = f.clock_context(ctx)
    assert [(x['t0'], x['t1']) for x in shown['annotation_subtasks']] == [(0, 2), (2, 4)]
    assert ctx['annotation_subtasks'][0]['t0'] == 2
    assert f.clock_context(ctx) == shown


@pytest.mark.parametrize('value', [7, 0.5, True])
def test_unresolved_indices_keep_the_original_row_and_value(value):
    spans, unresolved = annotations().annotation_spans([value], [2], {0: 'pick'},
                                                      'meta/tasks.jsonl', 'task_index', final_boundary=3)
    assert spans == []
    assert unresolved[0]['row'] == 0 and unresolved[0]['value'] == value
    assert type(unresolved[0]['value']) is type(value)
    assert unresolved[0]['source'] == 'meta/tasks.jsonl' and unresolved[0]['why']


def test_nan_time_does_not_bridge_two_recorded_runs():
    spans, unresolved = annotations().annotation_spans([0, 0, 0], [2, float('nan'), 4], {0: 'pick'},
                                                      'meta/tasks.jsonl', 'task_index', final_boundary=5)
    assert [(x['row_start'], x['row_end']) for x in spans] == [(0, 0), (1, 1), (2, 2)]
    assert spans[0]['t0'] == 2 and 't1' not in spans[0] and 't0' not in spans[1]
    assert spans[0]['timing_reason'] and spans[1]['timing_reason']
    assert (spans[2]['t0'], spans[2]['t1']) == (4, 5)
    assert not unresolved


def test_sparse_frames_do_not_invent_annotation_continuity():
    spans, _ = annotations().annotation_spans([0, 0, 0], [2, 3, 5], {0: 'pick'},
                                             'meta/tasks.jsonl', 'task_index', final_boundary=6,
                                             frame_indices=[0, 1, 3])
    assert [(x['row_start'], x['row_end']) for x in spans] == [(0, 1), (2, 2)]
    assert spans[0]['t0'] == 2 and 't1' not in spans[0] and spans[0]['timing_reason']
    assert (spans[1]['t0'], spans[1]['t1']) == (5, 6)


def test_an_unmeasured_final_boundary_stays_untimed():
    spans, _ = annotations().annotation_spans([0, 0], [2, 3], {0: 'pick'}, 'meta/tasks.jsonl', 'task_index')
    assert spans[0]['t0'] == 2 and 't1' not in spans[0]
    assert spans[0]['timing_reason']


def test_duplicate_table_indices_keep_every_conflicting_claim(tmp_path):
    meta = tmp_path / 'meta'
    meta.mkdir()
    path = meta / 'subtasks.jsonl'
    path.write_text('{"phase_index":1,"label":"pick","score":0.4}\n'
                    '{"phase_index":1,"label":"place","score":0.8}\n')
    tables = annotations().index_tables(meta, ['phase_index'], f.read_jsonl)
    table = tables[('phase_index', str(path))]
    assert table['labels'][1]['label'] is None
    assert [x['fields'] for x in table['labels'][1]['claims']] == [
        {'label': 'pick', 'score': 0.4}, {'label': 'place', 'score': 0.8}]
    assert table['issues']
    spans, unresolved = annotations().annotation_spans([1], [2], table['labels'], str(path),
                                                      'phase_index', final_boundary=3)
    assert spans == [] and len(unresolved[0]['claims']) == 2


def test_a_named_parquet_index_supplies_ownership_without_using_row_order(tmp_path):
    meta = tmp_path / 'meta'
    meta.mkdir()
    path = meta / 'subtasks.parquet'
    pd.DataFrame({'label': ['place', 'pick'], 'confidence': [0.8, 0.4]},
                 index=pd.Index([9, 2], name='phase_index')).to_parquet(path)
    tables = annotations().index_tables(meta, ['phase_index', 'frame_index'], f.read_jsonl)
    assert set(tables) == {('phase_index', str(path))}
    assert tables[('phase_index', str(path))]['labels'][2]['notes'] == {'label': 'pick', 'confidence': 0.4}


def test_unnamed_table_row_order_is_not_an_owner(tmp_path):
    meta = tmp_path / 'meta'
    meta.mkdir()
    pd.DataFrame({'label': ['pick', 'place']}).to_parquet(meta / 'subtasks.parquet')
    assert annotations().index_tables(meta, ['task_index'], f.read_jsonl) == {}


def test_multiple_text_fields_remain_notes_without_a_guessed_display_label(tmp_path):
    meta = tmp_path / 'meta'
    meta.mkdir()
    path = meta / 'subtasks.jsonl'
    path.write_text('{"phase_index":1,"task":"pick","note":"camera loose"}\n')
    table = annotations().index_tables(meta, ['phase_index'], f.read_jsonl)[('phase_index', str(path))]
    assert table['labels'][1]['label'] is None
    assert table['labels'][1]['notes'] == {'task': 'pick', 'note': 'camera loose'}


def _native_case(root, mode):
    state = np.arange(56, dtype=np.float32).reshape(4, 14) / 100
    data = {'observation.state': list(state), 'task_index': [0, 0, 1, 1], 'sensor': [3, 4, 5, 6]}
    if mode == 'packed':
        _lerobot_v3(root, {'observation.images.cam_high': 4},
                    {**data, 'episode_index': [0] * 4, 'frame_index': list(range(4)),
                     'timestamp': np.arange(4) / 30})
    else:
        feats = None
        if mode == 'image':
            data['observation.images.cam_high'] = [{'bytes': _jpeg(k * 20, 64, 64)} for k in range(4)]
            feats = {'observation.images.cam_high': {'dtype': 'image', 'shape': [64, 64, 3]}}
        _lerobot(root, {0: data}, feats=feats, n_video=0 if mode == 'image' else 4)
    (root / 'meta/tasks.jsonl').write_text('{"task_index":0,"task":"pick"}\n'
                                         '{"task_index":1,"task":"place"}\n')
    return state


@pytest.mark.parametrize('mode', ['video', 'packed', 'image'])
def test_every_lerobot_camera_path_keeps_changing_indices_and_native_bytes(tmp_path, mode):
    root = tmp_path / 'upload'
    state = _native_case(root, mode)
    originals = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
    output = tmp_path / 'episodes'
    report = f.convert(root, 'teleop_arms', output, 'test', 900)
    assert not report['failed'] and len(report['episodes']) == 1, report
    ep = output / report['episodes'][0]['episode_id']
    ctx = json.loads((ep / 'context.json').read_text())
    assert 'instruction' not in ctx, 'first row task must not become an episode instruction'
    assert [x['label'] for x in ctx['annotation_subtasks']] == ['pick', 'place']
    assert ctx['annotation_subtasks'][0]['t0'] == 0
    assert ctx['annotation_subtasks'][0]['t1'] == 2 / 30
    assert ctx['annotation_subtasks'][0]['column'] == 'task_index'
    assert ctx['annotation_subtasks'][0]['source'] == str(root / 'meta/tasks.jsonl')
    with np.load(ep / 'state.npz') as saved:
        assert saved['state'].dtype == state.dtype and saved['state'].tobytes() == state.tobytes()
    with np.load(ep / 'signals.npz') as saved:
        key = next(s['key'] for s in ctx['signals'] if s['name'] == 'sensor')
        expected = np.array([3, 4, 5, 6], dtype=saved[key].dtype).reshape(saved[key].shape)
        assert saved[key].tobytes() == expected.tobytes()
    assert {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()} == originals


def test_declared_episode_instruction_wins_over_changing_indices(tmp_path):
    root = tmp_path / 'upload'
    _native_case(root, 'video')
    (root / 'meta/episodes.jsonl').write_text('{"episode_index":0,"tasks":["declared goal"],"length":4}\n')
    report = f.convert(root, 'teleop_arms', tmp_path / 'episodes', 'test', 900)
    assert not report['failed'], report
    ctx = json.loads((tmp_path / 'episodes' / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    assert ctx['instruction'] == 'declared goal'
    assert [x['label'] for x in ctx['annotation_subtasks']] == ['pick', 'place']


def test_constant_duplicate_task_claims_cannot_supply_an_instruction(tmp_path):
    root = tmp_path / 'upload'
    _lerobot(root, {0: {'observation.state': [np.zeros(14)] * 4, 'task_index': [1] * 4}}, n_video=4)
    (root / 'meta/tasks.jsonl').write_text('{"task_index":1,"task":"pick"}\n'
                                         '{"task_index":1,"task":"place"}\n')
    report = f.convert(root, 'teleop_arms', tmp_path / 'episodes', 'test', 900)
    assert not report['failed'], report
    ctx = json.loads((tmp_path / 'episodes' / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    assert 'instruction' not in ctx
    assert not ctx.get('annotation_subtasks')
    assert len(ctx['annotation_unresolved'][0]['claims']) == 2
    assert any(i['kind'] == 'metadata_unreadable' for i in ctx['reader_issues'])


def test_an_index_column_without_a_table_keeps_all_claims(tmp_path):
    root = tmp_path / 'upload'
    _lerobot(root, {0: {'observation.state': [np.zeros(14)] * 4, 'phase_index': [7, 7, 8, 8]}}, n_video=4)
    report = f.convert(root, 'teleop_arms', tmp_path / 'episodes', 'test', 900)
    assert not report['failed'], report
    ctx = json.loads((tmp_path / 'episodes' / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    assert [x['value'] for x in ctx['annotation_unresolved']] == [7, 7, 8, 8]
    assert all(x['column'] == 'phase_index' for x in ctx['annotation_unresolved'])
    assert 'phase_index' in ctx['uploader_annotation']


@pytest.mark.parametrize('value', [0.5, True])
def test_invalid_table_keys_keep_their_original_claims(tmp_path, value):
    meta = tmp_path / 'meta'
    meta.mkdir()
    path = meta / 'subtasks.jsonl'
    path.write_text(json.dumps({'phase_index': value, 'label': 'pick'}) + '\n')
    table = annotations().index_tables(meta, ['phase_index'], f.read_jsonl)[('phase_index', str(path))]
    assert table['labels'] == {}
    assert table['issues'][0]['value'] == value and type(table['issues'][0]['value']) is type(value)


def test_nonzero_image_timestamps_map_once_and_keep_the_raw_annotation_clock(tmp_path):
    root = tmp_path / 'upload'
    _native_case(root, 'image')
    path = root / 'data/chunk-000/episode_000000.parquet'
    df = pd.read_parquet(path)
    df['timestamp'] += 2
    df.to_parquet(path)
    raw = path.read_bytes()
    report = f.convert(root, 'teleop_arms', tmp_path / 'episodes', 'test', 900)
    assert not report['failed'], report
    ctx = json.loads((tmp_path / 'episodes' / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    step = ctx['annotation_subtasks'][0]
    assert step['t0'] == 0 and step['t1'] == pytest.approx(2 / 30)
    assert step['raw_t0'] == 2 and step['raw_t1'] == 2 + 2 / 30
    assert step['clock_offset_s'] == -2
    assert 'clock_zero_s' not in ctx
    assert f.clock_context(ctx)['annotation_subtasks'] == ctx['annotation_subtasks']
    assert path.read_bytes() == raw


def test_mixed_numeric_table_rows_preserve_large_integer_keys(tmp_path):
    meta = tmp_path / 'meta'
    meta.mkdir()
    path = meta / 'subtasks.parquet'
    pd.DataFrame({'phase_index': np.array([9007199254740993], dtype=np.int64),
                  'confidence': [0.5]}).to_parquet(path)
    tables = annotations().index_tables(meta, ['phase_index'], f.read_jsonl)
    assert list(tables[('phase_index', str(path))]['labels']) == [9007199254740993]


def test_timestamp_claims_keep_original_types_with_numeric_continuity():
    spans, _ = annotations().annotation_spans([0, 0, 1], ['2', 3, '10'], {0: 'pick', 1: 'place'},
                                             'tasks.jsonl', 'task_index')
    assert (spans[0]['t0'], spans[0]['t1']) == (2, 10)
    assert spans[0]['raw_times'] == ['2', 3]
    spans, _ = annotations().annotation_spans([0], [float('nan')], {0: 'pick'},
                                             'tasks.jsonl', 'task_index')
    assert np.isnan(spans[0]['raw_times'][0]) and 't0' not in spans[0]


def test_each_table_owner_retains_the_other_recorded_key_as_a_note(tmp_path):
    meta = tmp_path / 'meta'
    meta.mkdir()
    path = meta / 'subtasks.jsonl'
    path.write_text('{"task_index":1,"quality_index":2,"label":"pick"}\n')
    tables = annotations().index_tables(meta, ['task_index', 'quality_index'], f.read_jsonl)
    assert tables[('task_index', str(path))]['labels'][1]['notes']['quality_index'] == 2
    assert tables[('quality_index', str(path))]['labels'][2]['notes']['task_index'] == 1


def test_jsonl_rows_keep_exact_scalar_types_across_missing_cells(tmp_path):
    meta = tmp_path / 'meta'
    meta.mkdir()
    path = meta / 'subtasks.jsonl'
    path.write_text('{"phase_index":9007199254740993,"label":"pick","count":1}\n'
                    '{"phase_index":null,"label":"place","count":null}\n')
    table = annotations().index_tables(meta, ['phase_index'], f.read_jsonl)[('phase_index', str(path))]
    assert list(table['labels']) == [9007199254740993]
    assert type(table['labels'][9007199254740993]['notes']['count']) is int
    assert table['issues'][0]['value'] is None


def test_scanning_a_table_without_index_keys_does_not_mark_its_notes_consumed(tmp_path):
    root = tmp_path / 'upload'
    _native_case(root, 'video')
    path = root / 'meta/recorder_notes.jsonl'
    path.write_text('{"note":"camera loose","count":1}\n')
    original = path.read_bytes()
    before = f.read_root(root, '')
    df = pd.read_parquet(root / 'data/chunk-000/episode_000000.parquet')
    ctx = {'fps': 30, 'n_state_frames': 4}
    f.lerobot_annotations(ctx, before, df, root / 'data/chunk-000/episode_000000.parquet')
    assert path.resolve() not in before['metadata_read']
    assert path.read_bytes() == original


def _galaxea_case(root, coarse):
    import av
    from prepare import galaxea
    cols = {'task_index': [0, 0, 1, 1], 'coarse_task_index': [coarse] * 4, 'quality_index': [7] * 4}
    for prefix in ('observation.state', 'action'):
        for side in ('left', 'right'):
            cols[f'{prefix}.{side}_arm'] = [np.zeros(6)] * 4
            cols[f'{prefix}.{side}_gripper'] = [0.5] * 4
    _lerobot(root, {0: cols}, feats={c: {'dtype': 'float32', 'shape': [1]} for c in cols}, n_video=0)
    info = json.loads((root / 'meta/info.json').read_text())
    info['fps'] = 15
    info['features'].pop('observation.images.cam_high')
    for key in galaxea.VIDEO_KEYS.values():
        info['features'][key] = {'dtype': 'video', 'shape': [36, 64, 3]}
        path = root / 'videos/chunk-000' / key / 'episode_000000.mp4'
        path.parent.mkdir(parents=True)
        with av.open(str(path), 'w') as container:
            stream = container.add_stream('mpeg4', rate=15)
            stream.width, stream.height, stream.pix_fmt = 64, 36, 'yuv420p'
            for row in range(4):
                frame = av.VideoFrame.from_ndarray(np.full((36, 64, 3), row * 20, np.uint8), format='rgb24')
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    (root / 'meta/info.json').write_text(json.dumps(info))
    (root / 'meta/tasks.jsonl').write_text('{"task_index":0,"task":"pick"}\n{"task_index":1,"task":"place"}\n')
    (root / 'meta/episodes.jsonl').write_text('{"episode_index":0,"length":4,"tasks":[]}\n')
    path = root / 'data/chunk-000/episode_000000.parquet'
    df = pd.read_parquet(path)
    df['timestamp'] = np.arange(4) / 15
    df.to_parquet(path)


@pytest.mark.parametrize('coarse', [0, 0.5, True])
def test_adapter_boundary_uses_attributed_spans_and_never_truncates_coarse_codes(tmp_path, coarse):
    root = tmp_path / 'upload'
    _galaxea_case(root, coarse)
    originals = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
    output = tmp_path / 'episodes'
    report = f.convert(root, 'teleop_arms', output, 'test', 900)
    assert not report['failed'] and len(report['episodes']) == 1, report
    ctx = json.loads((output / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    assert ctx.get('instruction') == ('pick' if type(coarse) is int else None)
    assert len(ctx['annotation_subtasks']) == 2
    assert all(s['column'] == 'task_index' and s['raw_times'] for s in ctx['annotation_subtasks'])
    assert 't1' not in ctx['annotation_subtasks'][-1]
    assert 'computed' in ctx['uploader_annotation'] and 'approximation' in ctx['uploader_annotation']
    assert any(s['column'] == 'quality_index' and s['value'] == 7 for s in ctx['annotation_unresolved'])
    assert {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()} == originals


def test_whole_packed_recording_retains_unmapped_annotation_clocks(tmp_path):
    root = tmp_path / 'upload'
    _lerobot_v3(root, {'observation.images.cam_high': 4},
                {'task_index': [0, 0, 1, 1], 'timestamp': np.arange(4) / 30})
    (root / 'meta/tasks.jsonl').write_text('{"task_index":0,"task":"pick"}\n{"task_index":1,"task":"place"}\n')
    report = f.convert(root, 'teleop_arms', tmp_path / 'episodes', 'test', 900)
    assert not report['failed'] and len(report['episodes']) == 1, report
    ctx = json.loads((tmp_path / 'episodes' / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    assert [x['label'] for x in ctx['annotation_subtasks']] == ['pick', 'place']
    assert all('t0' not in x and 't1' not in x and x['timing_reason'] for x in ctx['annotation_subtasks'])
    assert ctx['annotation_subtasks'][1]['raw_times'] == [2 / 30, 3 / 30]
    assert ctx['uploader_notes']['recorded annotation spans']['task_index'][1]['raw_times'] == [2 / 30, 3 / 30]


def test_failed_annotation_table_retains_its_source_and_decoder_reason_without_consumption(tmp_path):
    root = tmp_path / 'upload'
    state = np.arange(56, dtype=np.float32).reshape(4, 14) / 100
    _lerobot(root, {0: {'observation.state': list(state), 'phase_index': [0, 0, 1, 1],
                       'sensor': [3, 4, 5, 6]}}, n_video=4)
    clean_out = tmp_path / 'clean'
    clean = f.convert(root, 'teleop_arms', clean_out, 'test', 900)
    assert not clean['failed'] and len(clean['episodes']) == 1
    path = root / 'meta/subtasks.parquet'
    path.write_bytes(b'not a parquet table')
    originals = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
    output = tmp_path / 'failed_table'
    report = f.convert(root, 'teleop_arms', output, 'test', 900)
    assert not report['failed'] and len(report['episodes']) == 1, report
    ep = output / report['episodes'][0]['episode_id']
    ctx = json.loads((ep / 'context.json').read_text())
    issue = next(x for x in ctx['reader_issues'] if 'subtasks.parquet' in x['what'])
    assert issue['kind'] == 'metadata_unreadable' and 'table read failed' in issue['what']
    failed = ctx['annotation_table_failures'][str(path)]
    assert failed['labels'] == {} and 'table read failed' in failed['issues'][0]['why']
    assert ctx['uploader_notes']['unreadable annotation table sources'][str(path)] == failed
    parsed = f.read_root(root, '')
    assert path.resolve() not in parsed['metadata_read']
    assert any('subtasks.parquet' in x['text'] and 'table read failed' in x['text'] for x in parsed['metadata_issues'])
    assert str(path) not in ctx['annotation_tables'] and not ctx['annotation_subtasks']
    clean_ep = clean_out / clean['episodes'][0]['episode_id']
    for name in ('state.npz', 'signals.npz'):
        with np.load(clean_ep / name) as before, np.load(ep / name) as after:
            assert before.files == after.files
            for key in before.files:
                assert before[key].dtype == after[key].dtype and before[key].shape == after[key].shape
                assert before[key].tobytes() == after[key].tobytes()
    assert (clean_ep / 'sources.json').read_bytes() == (ep / 'sources.json').read_bytes()
    assert {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()} == originals


def _task2_metadata(root, mode, instruction=None):
    row = {'episode_index': 0, 'length': 4, 'tasks': [instruction] if instruction else [],
           'operator': {'name': 'recorded operator', 'notes': ['literal note', '7']},
           'success_claim': False, 'custom': {'value': 9007199254740993}}
    neighbor = {'episode_index': 1, 'length': 'damaged', 'tasks': [27], 'operator': 'neighbor only'}
    if mode == 'packed':
        camera = 'observation.images.cam_high'
        row.update({'data/chunk_index': 0, 'data/file_index': 0,
                    f'videos/{camera}/chunk_index': 0, f'videos/{camera}/file_index': 0,
                    f'videos/{camera}/from_timestamp': 0.0, f'videos/{camera}/to_timestamp': 4 / 30})
        path = root / 'meta/episodes/chunk-000/file-000.parquet'
        path.parent.mkdir(parents=True)
        # Distinct native rows without dataframe integer coercion across the damaged neighbor.
        pd.DataFrame([row]).to_parquet(path)
        pd.DataFrame([neighbor]).to_parquet(path.with_name('file-001.parquet'))
    else:
        path = root / 'meta/episodes.jsonl'
        path.write_text(json.dumps(row) + '\n' + json.dumps(neighbor) + '\n{damaged json\n')
    info_path = root / 'meta/info.json'
    info = json.loads(info_path.read_text())
    calibration = {'intrinsics': [[501.25, 0, 32], [0, 502.5, 32], [0, 0, 1]],
                   'distortion': [-0.1, 0.002, 0, 0, 0], 'other': {'serial': 'native serial'}}
    info['features']['observation.images.cam_high']['calibration'] = calibration
    info_path.write_text(json.dumps(info))
    return path, row, calibration


@pytest.mark.parametrize('mode', ['video', 'packed', 'image'])
def test_full_selected_metadata_and_intrinsics_keep_native_ownership_and_arrays(tmp_path, mode):
    root = tmp_path / 'upload'
    _native_case(root, mode)
    clean_out = tmp_path / 'clean'
    clean = f.convert(root, 'teleop_arms', clean_out, 'test', 900)
    assert not clean['failed'] and len(clean['episodes']) == 1
    path, row, calibration = _task2_metadata(root, mode)
    originals = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
    output = tmp_path / 'with_metadata'
    report = f.convert(root, 'teleop_arms', output, 'test', 900)
    assert not report['failed'], report
    episode = next(e for e in report['episodes'] if e['episode_id'] == clean['episodes'][0]['episode_id'])
    ep = output / episode['episode_id']
    ctx = json.loads((ep / 'context.json').read_text())
    assert ctx['recorded_metadata'][str(path)][0]['fields'] == row
    assert ctx['recorded_metadata'][str(root / 'meta/info.json')]['features']['observation.images.cam_high']['calibration'] == calibration
    assert 'neighbor only' not in json.dumps(ctx['recorded_metadata'])
    assert 'recorded operator' in ctx['uploader_annotation'] and '501.25' in ctx['uploader_annotation']
    clean_ep = clean_out / clean['episodes'][0]['episode_id']
    assert sorted(p.name for p in clean_ep.glob('*.npz')) == sorted(p.name for p in ep.glob('*.npz'))
    for before_path in clean_ep.glob('*.npz'):
        with np.load(before_path) as before, np.load(ep / before_path.name) as after:
            assert before.files == after.files
            for key in before.files:
                assert before[key].dtype == after[key].dtype and before[key].shape == after[key].shape
                assert before[key].tobytes() == after[key].tobytes()
    import av
    def packets(path):
        with av.open(str(path)) as container:
            return [(packet.pts, packet.dts, str(packet.time_base), hashlib.sha256(bytes(packet)).hexdigest())
                    for packet in container.demux(video=0) if packet.size]
    media = sorted(p.relative_to(clean_ep) for p in clean_ep.rglob('*') if p.suffix in ('.mkv', '.mp4'))
    # Only these two owned output prefixes differ; native packed input paths remain literal.
    assert (clean_ep / 'sources.json').read_text().replace(str(clean_ep.resolve()), '<owned episode>') == (
        ep / 'sources.json').read_text().replace(str(ep.resolve()), '<owned episode>')
    assert media or any('packed' in entry for entry in json.loads((ep / 'sources.json').read_text()).values())
    assert media == sorted(p.relative_to(ep) for p in ep.rglob('*') if p.suffix in ('.mkv', '.mp4'))
    for name in media:
        assert packets(clean_ep / name) == packets(ep / name)
    assert {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()} == originals


@pytest.mark.parametrize('mode', ['video', 'packed', 'image'])
@pytest.mark.parametrize('declared', [None, 'declared goal'])
def test_constant_row_language_preserves_declared_instruction_priority(tmp_path, mode, declared):
    root = tmp_path / 'upload'
    _native_case(root, mode)
    _task2_metadata(root, mode, declared)
    path = root / ('data/chunk-000/file-000.parquet' if mode == 'packed' else 'data/chunk-000/episode_000000.parquet')
    df = pd.read_parquet(path)
    df['language_instruction'] = ['recorded language'] * 4
    df.to_parquet(path)
    raw = path.read_bytes()
    output = tmp_path / 'episodes'
    report = f.convert(root, 'teleop_arms', output, 'test', 900)
    assert not report['failed'], report
    ctx = json.loads((output / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    assert ctx['instruction'] == (declared or 'recorded language')
    language = [s for s in ctx['annotation_subtasks'] if s['column'] == 'language_instruction']
    assert language[0]['value'] == 'recorded language' and language[0]['source'] == str(path)
    assert path.read_bytes() == raw


@pytest.mark.parametrize('mode', ['video', 'packed', 'image'])
def test_changing_language_is_timed_claims_and_never_a_first_row_goal(tmp_path, mode):
    root = tmp_path / 'upload'
    _native_case(root, mode)
    path = root / ('data/chunk-000/file-000.parquet' if mode == 'packed' else 'data/chunk-000/episode_000000.parquet')
    df = pd.read_parquet(path)
    df['language_instruction'] = ['pick', 'pick', 'place', 'place']
    df.to_parquet(path)
    output = tmp_path / 'episodes'
    report = f.convert(root, 'teleop_arms', output, 'test', 900)
    assert not report['failed'], report
    ctx = json.loads((output / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    assert 'instruction' not in ctx
    language = [s for s in ctx['annotation_subtasks'] if s['column'] == 'language_instruction']
    assert [(s['label'], s['t0']) for s in language] == [('pick', 0), ('place', 2 / 30)]
    assert language[0]['t1'] == 2 / 30 and 't1' not in language[1]


@pytest.mark.parametrize('kind', ['duplicate', 'files', 'columns'])
def test_competing_annotated_episode_instructions_have_no_winner(tmp_path, kind):
    root = tmp_path / 'upload'
    _native_case(root, 'video')
    path = root / 'meta/tasks_annotated.jsonl'
    rows = [{'episode_index': 0, 'instruction': 'pick'}, {'episode_index': 1, 'instruction': 'neighbor only'}]
    if kind == 'duplicate':
        rows.append({'episode_index': 0, 'instruction': 'place', 'detail': 17})
    elif kind == 'columns':
        rows[0]['language_instruction'] = 'place'
    else:
        (root / 'meta/other_annotated.jsonl').write_text('{"episode_index":0,"instruction":"place"}\n')
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    output = tmp_path / 'episodes'
    report = f.convert(root, 'teleop_arms', output, 'test', 900)
    assert not report['failed'], report
    ctx = json.loads((output / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    assert 'instruction' not in ctx
    assert 'pick' in ctx['uploader_annotation'] and 'place' in ctx['uploader_annotation']
    assert 'neighbor only' not in json.dumps(ctx['recorded_metadata'])
    assert any(i['kind'] == 'metadata_unreadable' and 'competing' in i['what'] for i in ctx['reader_issues'])


def test_empty_malformed_and_competing_frame_text_remains_source_attributed(tmp_path):
    root = tmp_path / 'upload'
    _native_case(root, 'video')
    path = root / 'data/chunk-000/episode_000000.parquet'
    df = pd.read_parquet(path)
    df['language_instruction'] = ['pick', '', None, 'place']
    df['instruction'] = ['other goal'] * 4
    df.to_parquet(path)
    output = tmp_path / 'episodes'
    report = f.convert(root, 'teleop_arms', output, 'test', 900)
    assert not report['failed'], report
    ctx = json.loads((output / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    assert 'instruction' not in ctx
    missing = [x for x in ctx['annotation_unresolved'] if x['column'] == 'language_instruction']
    assert [x['value'] for x in missing] == ['', None]
    assert all(x['source'] == str(path) and x['why'] for x in missing)
    assert 'other goal' in ctx['uploader_annotation'] and 'language_instruction' in ctx['uploader_annotation']


def test_unassigned_episode_claims_never_attach_valid_neighbors(tmp_path):
    root = tmp_path / 'upload'
    _native_case(root, 'video')
    path = root / 'meta/episodes.jsonl'
    path.write_text('\n'.join(json.dumps(x) for x in [
        {'episode_index': 0, 'length': 4, 'tasks': ['declared'], 'native': {'owner': 0}},
        {'episode_index': 9, 'length': 4, 'tasks': ['neighbor only'], 'native': {'owner': 9}},
        {'episode_index': False, 'instruction': 'unassigned text', 'unknown': [13, False]},
    ]) + '\n{broken row\n')
    report = f.convert(root, 'teleop_arms', tmp_path / 'episodes', 'test', 900)
    ctx = json.loads((tmp_path / 'episodes' / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    assert ctx['instruction'] == 'declared'
    assert 'neighbor only' not in json.dumps(ctx['recorded_metadata'])
    bad = ctx['unassigned_metadata']
    assert bad[0]['fields'] == {'episode_index': False, 'instruction': 'unassigned text', 'unknown': [13, False]}
    assert bad[0]['source'] == str(path) and 'no episode owner' in bad[0]['why']
    assert bad[1]['raw_text'] == '{broken row' and bad[1]['line'] == 4
    assert 'unassigned text' in ctx['uploader_annotation']


def test_duplicate_episode_tasks_cannot_supply_a_last_row_instruction(tmp_path):
    root = tmp_path / 'upload'
    _native_case(root, 'video')
    path = root / 'meta/episodes.jsonl'
    path.write_text('\n'.join(json.dumps(x) for x in [
        {'episode_index': 0, 'length': 4, 'tasks': ['first claim'], 'operator': 'first'},
        {'episode_index': 0, 'length': 4, 'tasks': ['second claim'], 'operator': 'second'},
    ]) + '\n')
    report = f.convert(root, 'teleop_arms', tmp_path / 'episodes', 'test', 900)
    ctx = json.loads((tmp_path / 'episodes' / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    assert 'instruction' not in ctx
    rows = ctx['recorded_metadata'][str(path)]
    assert [x['fields']['tasks'] for x in rows] == [['first claim'], ['second claim']]
    assert any('competing' in x['what'] for x in ctx['reader_issues'])


def test_complete_frame_language_precedes_only_the_coarse_index_fallback(tmp_path):
    root = tmp_path / 'upload'
    _native_case(root, 'video')
    path = root / 'data/chunk-000/episode_000000.parquet'
    df = pd.read_parquet(path)
    df['task_index'] = [0] * 4
    df['language_instruction'] = ['specific recorded language'] * 4
    df.to_parquet(path)
    report = f.convert(root, 'teleop_arms', tmp_path / 'episodes', 'test', 900)
    ctx = json.loads((tmp_path / 'episodes' / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    assert ctx['instruction'] == 'specific recorded language'
    assert ctx['task_label'] == ['pick']


def test_partial_language_never_becomes_a_whole_instruction(tmp_path):
    root = tmp_path / 'upload'
    _native_case(root, 'video')
    path = root / 'data/chunk-000/episode_000000.parquet'
    df = pd.read_parquet(path)
    df['language_instruction'] = ['pick', None, 'pick', 'pick']
    df.to_parquet(path)
    report = f.convert(root, 'teleop_arms', tmp_path / 'episodes', 'test', 900)
    ctx = json.loads((tmp_path / 'episodes' / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    assert 'instruction' not in ctx
    assert ctx['annotation_unresolved'][0]['value'] is None
    assert ctx['annotation_unresolved'][0]['source'] == str(path)


@pytest.mark.parametrize('bad', [None, ''])
def test_incomplete_annotated_owner_rows_never_supply_a_surviving_winner(tmp_path, bad):
    root = tmp_path / 'upload'
    _native_case(root, 'video')
    path = root / 'meta/tasks_annotated.jsonl'
    rows = [{'episode_index': 0, 'instruction': 'first claim'}, {'episode_index': 0, 'instruction': bad}]
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    report = f.convert(root, 'teleop_arms', tmp_path / 'episodes', 'test', 900)
    ctx = json.loads((tmp_path / 'episodes' / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    assert 'instruction' not in ctx
    assert [x['fields'] for x in ctx['recorded_metadata'][str(path)]] == rows
    assert any('malformed' in x['what'] for x in ctx['reader_issues'])


def test_invalid_episode_task_claim_does_not_block_complete_native_language(tmp_path):
    root = tmp_path / 'upload'
    _native_case(root, 'video')
    (root / 'meta/episodes.jsonl').write_text('{"episode_index":0,"length":4,"tasks":[123]}\n')
    path = root / 'data/chunk-000/episode_000000.parquet'
    df = pd.read_parquet(path)
    df['language_instruction'] = ['recorded language'] * 4
    df.to_parquet(path)
    report = f.convert(root, 'teleop_arms', tmp_path / 'episodes', 'test', 900)
    ctx = json.loads((tmp_path / 'episodes' / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    assert ctx['instruction'] == 'recorded language'
    assert ctx['recorded_metadata'][str(root / 'meta/episodes.jsonl')][0]['fields']['tasks'] == [123]
    assert any(x['kind'] == 'metadata_unreadable' for x in ctx['reader_issues'])


def test_full_parquet_metadata_retains_native_binary_and_timestamp_fields(tmp_path):
    root = tmp_path / 'upload'
    _native_case(root, 'video')
    path = root / 'meta/tasks_annotated.parquet'
    stamp = pd.Timestamp('2026-10-03T12:34:56.123456789')
    pd.DataFrame([{'episode_index': 0, 'instruction': 'declared goal',
                   'recorded_at': stamp, 'payload': b'\x00\xffliteral'}]).to_parquet(path)
    original_bytes = path.read_bytes()
    report = f.convert(root, 'teleop_arms', tmp_path / 'episodes', 'test', 900)
    assert not report['failed'], report
    ctx = json.loads((tmp_path / 'episodes' / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    fields = ctx['recorded_metadata'][str(path)][0]['fields']
    assert fields['payload'] == {'recorded_type': 'bytes', 'hex': '00ff6c69746572616c'}
    assert fields['recorded_at']['iso8601'] == stamp.isoformat()
    assert fields['recorded_at']['recorded_type'] == 'Timestamp'
    assert fields['recorded_at']['integer_ns'] == stamp.value
    assert ctx['instruction'] == 'declared goal' and path.read_bytes() == original_bytes
