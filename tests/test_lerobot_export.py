"""board/lerobot_export.py on a small packed LeRobot v3 dataset laid out as LeRobot writes one (meta/info.json, meta/
episodes, data/chunk-000/file-000.parquet with two episodes, a video file), whose frame timestamps are uneven, as a
recorder's are, so a time computed from a frame index and the rate lands between frames and only a time read from the
data file lands on one."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from board import lerobot_export as lx

FPS = 10
# episode 0: 30 frames over 3s, episode 1: 20 frames; a few frames arrive late, as a recorder's do
JITTER = {3: .004, 12: .007, 17: .003, 24: .006}


def frame_times(n):
    return [np.float32(k / FPS + JITTER.get(k, 0.)).item() for k in range(n)]


def dataset(root: Path, ep0_times=None, fps=True, rate=FPS) -> Path:
    """ep0_times replaces episode 0's 30 frame timestamps; rate is the fps meta/info.json declares, and fps=False
    leaves it out."""
    (root / 'meta' / 'episodes' / 'chunk-000').mkdir(parents=True)
    (root / 'data' / 'chunk-000').mkdir(parents=True)
    (root / 'videos' / 'observation.images.front' / 'chunk-000').mkdir(parents=True)
    (root / 'videos' / 'observation.images.front' / 'chunk-000' / 'file-000.mp4').write_bytes(b'\0' * 64)
    info = {'codebase_version': 'v3.0', 'robot_type': 'so100', 'fps': rate, 'total_episodes': 2, 'total_frames': 50,
            'chunks_size': 1000, 'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
            'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4',
            'features': {'observation.state': {'dtype': 'float32', 'shape': [6], 'names': None},
                         'action': {'dtype': 'float32', 'shape': [6], 'names': None},
                         'observation.images.front': {'dtype': 'video', 'shape': [480, 640, 3]},
                         'timestamp': {'dtype': 'float32', 'shape': [1], 'names': None},
                         'frame_index': {'dtype': 'int64', 'shape': [1], 'names': None},
                         'episode_index': {'dtype': 'int64', 'shape': [1], 'names': None},
                         'index': {'dtype': 'int64', 'shape': [1], 'names': None},
                         'task_index': {'dtype': 'int64', 'shape': [1], 'names': None}}}
    if not fps:
        del info['fps']
    (root / 'meta' / 'info.json').write_text(json.dumps(info, indent=4))
    pq.write_table(pa.table({'episode_index': [0, 1], 'tasks': [['put the cup on the plate']] * 2,
                             'length': [30, 20], 'dataset_from_index': [0, 30], 'dataset_to_index': [30, 50]}),
                   root / 'meta' / 'episodes' / 'chunk-000' / 'file-000.parquet')
    ts = (frame_times(30) if ep0_times is None else ep0_times) + frame_times(20)
    eps = [0] * 30 + [1] * 20
    pq.write_table(pa.table({
        'observation.state': pa.array([[0.] * 6] * 50, type=pa.list_(pa.float32(), 6)),
        'action': pa.array([[0.] * 6] * 50, type=pa.list_(pa.float32(), 6)),
        'timestamp': pa.array(ts, type=pa.float32()),
        'frame_index': pa.array(list(range(30)) + list(range(20)), type=pa.int64()),
        'episode_index': pa.array(eps, type=pa.int64()),
        'index': pa.array(range(50), type=pa.int64()),
        'task_index': pa.array([0] * 50, type=pa.int64()),
        # columns LeRobot's annotation writer drops when it writes the language columns
        'subtask_index': pa.array([0] * 50, type=pa.int64()),
        'tools': pa.array(['[]'] * 50, type=pa.string())}), root / 'data' / 'chunk-000' / 'file-000.parquet')
    return root


def step(t0, t1, arm, verb, dest=''):
    return {'t_s': t0, 'end_s': t1, 'arm': arm, 'verb_class': verb, 'object': '',
            'carry_phase': f'-> {dest}' if dest else '', 'contribution': 'advancing'}


EP0 = {  # the board's episode file (board/to_board.py), named by the board
    'event_labels': [step(.5, 1.5, 'left', 'pick up the cup'), step(1., 2., 'right', 'hold the plate steady'),
                     step(2.4, 2.9, 'both', 'set the cup down', 'onto the plate')],
    'key_events': [{'t_s': 1.23, 'label': 'cup lifted', 'kind': 'grasp', 'outcome': 'success', 'note': ''},
                   {'t_s': None, 'label': 'untimed', 'kind': '', 'outcome': '', 'note': ''},
                   {'t_s': 5.0, 'label': 'after the end', 'kind': '', 'outcome': '', 'note': ''}],
    'operator_mistakes': [{'issue': 'cup slips in the grip', 'category': 'failed_grasp', 'severity': 'low',
                           't_s': 1.7, 'end_s': 1.9, 'recovered': True}],
    'recovery': [{'failure_t_s': 1.7, 'failure': 'cup slips', 'recovered': True, 'recovered_at_s': 2.5,
                  'correction': 'regrips', 'how_to_fix': 'grip lower'}],
    'state_changes': [{'t_s': 2.6, 'object': 'cup', 'from': 'in hand', 'to': 'on the plate'}],
    'completion': {'task_completed': 'success', 'completed_at_s': 2.8, 'goal_reached_at_s': 2.8},
    'tasks': [], '_meta': {'episode_id': 'so100__episode_000000', 'run_episode': 'episode_000000'}}
EP1 = {'event_labels': [step(0., 1.9, 'right', 'wave')], 'completion': {'task_completed': 'failure'},
       '_meta': {'episode_id': 'episode_1'}}


def labels(folder: Path) -> Path:
    folder.mkdir()
    (folder / 'so100__episode_000000.json').write_text(json.dumps(EP0))
    (folder / 'b.json').write_text(json.dumps(EP1))
    (folder / 'notes.json').write_text(json.dumps({'_meta': {'episode_id': 'summary'}}))
    return folder


def digest(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file()}


@pytest.fixture
def exported(tmp_path):
    src = dataset(tmp_path / 'src')
    before = digest(src)
    report = lx.export(src, labels(tmp_path / 'labels'), tmp_path / 'out')
    assert digest(src) == before            # the source dataset is only read
    table = pq.read_table(tmp_path / 'out' / 'data' / 'chunk-000' / 'file-000.parquet')
    return src, tmp_path / 'out', report, table


def per_episode(table, e):
    rows = [r for r in table.to_pylist() if r['episode_index'] == e]
    return rows


def test_the_columns_have_lerobots_types_and_pass_its_checks(exported):
    src, out, report, table = exported
    assert report['validation_errors'] == [] and lx.validate(out) == []
    assert table.schema.field('language_persistent').type == lx.column_types()['language_persistent']
    assert table.schema.field('language_events').type == lx.column_types()['language_events']
    # every source column is kept, row for row
    source = pq.read_table(src / 'data' / 'chunk-000' / 'file-000.parquet')
    kept = [c for c in source.column_names if c not in ('subtask_index', 'tools')]
    assert table.select(kept).equals(source.select(kept))
    info = json.loads((out / 'meta' / 'info.json').read_text())
    assert info['features']['language_events'] == {'dtype': 'language', 'shape': [1], 'names': None}
    assert (out / 'videos' / 'observation.images.front' / 'chunk-000' / 'file-000.mp4').read_bytes() == b'\0' * 64
    assert report['labels'] == 2 and report['unmatched_labels'] == ['notes.json'] and report['episodes_written'] == 2


def test_subtasks_cover_every_frame_with_one_text_each(exported):
    _, _, _, table = exported
    rows = per_episode(table, 0)
    lists = [r['language_persistent'] for r in rows]
    assert all(x == lists[0] for x in lists)            # the same list on every frame of the episode
    ts = [r['timestamp'] for r in rows]
    got = [(r['timestamp'], r['content']) for r in lists[0]]
    near = lambda t: min(ts, key=lambda f: abs(f - t))
    assert got == [(ts[0], 'idle'), (near(.5), 'pick up the cup'),
                   (near(1.), 'left: pick up the cup; right: hold the plate steady'),
                   (near(1.5), 'hold the plate steady'), (near(2.), 'idle'),
                   (near(2.4), 'set the cup down -> onto the plate'), (near(2.9), 'idle')]
    assert all(r['style'] == 'subtask' and r['camera'] is None for r in lists[0])
    # episode 1 carries its own list, matched by _meta.episode_id
    assert [r['content'] for r in per_episode(table, 1)[0]['language_persistent']] == ['wave', 'idle']


def test_each_finding_fires_on_the_frame_read_from_the_data_file(exported):
    _, _, report, table = exported
    rows = per_episode(table, 0)
    fired = {(r['timestamp'], e['style']): json.loads(e['content']) for r in rows for e in r['language_events']}
    ts = [r['timestamp'] for r in rows]
    near = lambda t: min(ts, key=lambda f: abs(f - t))
    # the key event at 1.23s lands on frame 12, which arrived 7ms late: the frame's own timestamp, not 12 / fps
    assert near(1.23) == ts[12] != np.float32(12 / FPS).item()
    assert fired[(ts[12], 'key_event')]['label'] == 'cup lifted'
    assert fired[(near(1.7), 'mistake')]['category'] == 'failed_grasp'
    assert fired[(near(2.5), 'recovery')]['correction'] == 'regrips'
    assert fired[(near(2.6), 'state_change')]['to'] == 'on the plate'
    assert fired[(near(2.8), 'outcome')] == {'scope': 'episode', 'task_completed': 'success',
                                             'completed_at_s': 2.8, 'goal_reached_at_s': 2.8}
    assert sum(len(r['language_events']) for r in rows) == 5
    # the untimed key event is counted, the one after the episode's end is refused with its distance
    assert report['untimed'] == {'key_event': 1, 'outcome': 1}          # episode 1's verdict has no time
    assert [(r['episode'], r['what'], r['t_s']) for r in report['refused']] == \
        [('so100__episode_000000.json', 'key_event', 5.0)]
    assert report['refused'][0]['off_s'] > .1 and report['max_snap_s'] <= .1


def test_the_checks_catch_a_column_lerobot_would_reject(exported, tmp_path):
    _, out, _, table = exported
    path = out / 'data' / 'chunk-000' / 'file-000.parquet'
    rows = table.column('language_persistent').to_pylist()
    rows[1] = rows[1][1:]                                           # one frame of episode 0 loses its first subtask
    events = table.column('language_events').to_pylist()
    events[0] = [{'role': 'assistant', 'content': '{}', 'style': 'subtask', 'camera': None, 'tool_calls': None}]
    broken = table.drop_columns(['language_persistent', 'language_events'])
    broken = broken.append_column('language_persistent', lx._array(rows, 'language_persistent'))
    broken = broken.append_column('language_events', lx._array(events, 'language_events'))
    path.unlink()
    pq.write_table(broken, path)
    errors = lx.validate(out)
    assert 'ep=0: the persistent list differs between frames' in errors
    assert "ep=0: style 'subtask' in language_events" in errors


def test_it_refuses_a_folder_that_is_not_empty_and_two_files_for_one_episode(tmp_path):
    src = dataset(tmp_path / 'src')
    (tmp_path / 'out').mkdir()
    (tmp_path / 'out' / 'x').write_text('')
    with pytest.raises(SystemExit, match='not empty'):
        lx.export(src, labels(tmp_path / 'labels'), tmp_path / 'out')
    (tmp_path / 'labels' / 'again.json').write_text(json.dumps(EP1))
    with pytest.raises(SystemExit, match='two episode files for LeRobot episode 1'):
        lx.export(src, tmp_path / 'labels', tmp_path / 'out2')


def export_one(tmp_path, ep0, **kw):
    """Export episode 0's label ep0 (and episode 1's usual one) from a dataset built with kw. (report, table)"""
    src = dataset(tmp_path / 'src', **kw)
    folder = tmp_path / 'labels'
    folder.mkdir()
    (folder / 'episode_000000.json').write_text(json.dumps(ep0))
    (folder / 'b.json').write_text(json.dumps(EP1))
    report = lx.export(src, folder, tmp_path / 'out')
    return report, pq.read_table(tmp_path / 'out' / 'data' / 'chunk-000' / 'file-000.parquet')


def test_a_step_that_starts_on_the_frame_another_ends_on_keeps_its_own_subtask(tmp_path):
    # the pour starts 10ms after the grasp ends, so both boundaries round to the frame at 2.0s: the later one wins
    ep0 = {'event_labels': [step(.5, 2., 'right', 'grasp the cup'), step(2.01, 2.8, 'right', 'pour the water')]}
    report, table = export_one(tmp_path, ep0)
    rows = per_episode(table, 0)
    ts = [r['timestamp'] for r in rows]
    got = [(r['timestamp'], r['content']) for r in rows[0]['language_persistent']]
    assert got == [(ts[0], 'idle'), (ts[5], 'grasp the cup'), (ts[20], 'pour the water'), (ts[28], 'idle')]
    assert report['validation_errors'] == []


@pytest.mark.parametrize('fps', [True, False], ids=['rate from meta/info.json', 'median frame gap'])
def test_one_frame_close_to_the_next_does_not_narrow_how_far_a_time_may_snap(tmp_path, fps):
    # frame 5 arrives 1ms before frame 6; a time 40ms from a frame is still well inside one frame period (100ms)
    times = frame_times(30)
    times[5] = np.float32(.599).item()
    ep0 = {'event_labels': [], 'key_events': [{'t_s': 2.04, 'label': 'cup lifted', 'kind': '', 'outcome': '',
                                                'note': ''}]}
    report, table = export_one(tmp_path, ep0, ep0_times=times, fps=fps)
    assert report['refused'] == []
    rows = per_episode(table, 0)
    assert [r['timestamp'] for r in rows if r['language_events']] == [rows[20]['timestamp']]


def test_each_episode_is_one_row_group_and_the_old_annotation_columns_are_dropped(exported):
    _, out, _, table = exported
    f = pq.ParquetFile(out / 'data' / 'chunk-000' / 'file-000.parquet')
    groups = [set(f.read_row_group(i, columns=['episode_index']).column(0).to_pylist())
              for i in range(f.num_row_groups)]
    assert groups == [{0}, {1}]
    assert 'subtask_index' not in table.column_names and 'tools' not in table.column_names


def test_the_checks_say_which_rules_are_the_exporters_own():
    doc = ' '.join(lx.validate.__doc__.split())
    assert "the exporter's own rules, not LeRobot's" in doc


def test_a_declared_rate_lower_than_the_frames_does_not_widen_how_far_a_time_may_snap(tmp_path):
    # meta/info.json says 10 fps, episode 0's frames run at 30 fps: one frame period is 33ms, not 100ms, so a time
    # 2.5 real frames past the last frame is refused instead of landing on it (issue 196)
    times = [np.float32(k / 30).item() for k in range(30)]
    late = round(times[-1] + 2.5 / 30, 4)
    ep0 = {'event_labels': [], 'key_events': [{'t_s': late, 'label': 'after the end', 'kind': '', 'outcome': '',
                                                'note': ''}]}
    report, table = export_one(tmp_path, ep0, ep0_times=times)
    assert [(r['what'], r['t_s']) for r in report['refused']] == [('key_event', late)]
    assert not any(r['language_events'] for r in per_episode(table, 0))


def test_dropped_frames_do_not_widen_how_far_a_time_may_snap(tmp_path):
    # 30 fps declared and true, but 3 frames in every 10 dropped: the frames' span rate is about 21 fps (48ms), while
    # one frame period is still 33ms, so a time 1.2 real frames past the last frame is refused
    kept = [k for k in range(50) if k % 10 not in (2, 5, 8)][:30]
    times = [np.float32(k / 30).item() for k in kept]
    late = round(times[-1] + 1.2 / 30, 4)
    ep0 = {'event_labels': [], 'key_events': [{'t_s': late, 'label': 'after the end', 'kind': '', 'outcome': '',
                                                'note': ''}]}
    report, table = export_one(tmp_path, ep0, ep0_times=times, rate=30)
    assert [(r['what'], r['t_s']) for r in report['refused']] == [('key_event', late)]
    assert not any(r['language_events'] for r in per_episode(table, 0))


def test_a_declared_rate_higher_than_the_frames_still_places_a_time_on_the_nearer_frame(tmp_path):
    # 10 fps frames under a 30 fps declaration: a time 20ms after a frame lands on that frame
    ep0 = {'event_labels': [], 'key_events': [{'t_s': 1.42, 'label': 'between frames', 'kind': '', 'outcome': '',
                                                'note': ''}]}
    report, table = export_one(tmp_path, ep0, rate=30)
    assert report['refused'] == []
    rows = per_episode(table, 0)
    assert [r['timestamp'] for r in rows if r['language_events']] == [rows[14]['timestamp']]


def test_a_declared_rate_higher_than_the_frames_places_a_time_midway_between_two_frames(tmp_path):
    # 10 fps frames under a 30 fps declaration: a time 50ms from the frames on both sides lies inside the episode and
    # lands on the nearer frame, as one 20ms from a frame does; only a time past the episode's ends is refused
    ep0 = {'event_labels': [step(1.45, 2.95, 'right', 'pour the water')],
           'key_events': [{'t_s': 1.45, 'label': 'between frames', 'kind': '', 'outcome': '', 'note': ''}]}
    report, table = export_one(tmp_path, ep0, rate=30)
    assert report['refused'] == []
    rows = per_episode(table, 0)
    assert len([r for r in rows if r['language_events']]) == 1
    assert [r['content'] for r in rows[0]['language_persistent']] == ['idle', 'pour the water']


def test_a_step_that_starts_in_a_run_of_dropped_frames_still_starts_its_subtask(tmp_path):
    # 30 fps declared and true, frames 20 to 24 dropped: a step starting at 0.75s, 83ms from the nearer frame of the
    # gap, starts its subtask on that frame instead of leaving every frame after it with the step before
    times = [np.float32(k / 30).item() for k in range(35) if not 20 <= k <= 24]
    ep0 = {'event_labels': [step(0., .75, 'right', 'grasp the cup'), step(.75, 1.1, 'right', 'pour the water')]}
    report, table = export_one(tmp_path, ep0, ep0_times=times, rate=30)
    rows = per_episode(table, 0)
    assert [(r['timestamp'], r['content']) for r in rows[0]['language_persistent']] == \
        [(times[0], 'grasp the cup'), (times[20], 'pour the water'), (times[28], 'idle')]
    assert report['refused'] == [] and report['validation_errors'] == []


# episode 0 at 30 fps with frames 21 to 29 dropped: a 333ms gap from 0.667s to 1.0s
GAP_TIMES = [np.float32(k / 30).item() for k in range(39) if not 21 <= k <= 29]


def test_a_time_written_more_than_one_frame_period_from_its_frame_is_listed_with_its_distance(tmp_path):
    # a step and a key event at 0.82s land on the 0.667s frame, 153ms away: each is written and listed by row, as a
    # refused time is, never shown only in the dataset-wide max_snap_s
    ep0 = {'event_labels': [step(.82, 1.2, 'right', 'pour the water')],
           'key_events': [{'t_s': .82, 'label': 'cup tipped', 'kind': '', 'outcome': '', 'note': ''}]}
    report, table = export_one(tmp_path, ep0, ep0_times=GAP_TIMES, rate=30)
    assert report['refused'] == []
    far = sorted((r['episode'], r['what'], r['text'], r['t_s'], r['nearest_frame_s'], round(r['off_s'], 3))
                 for r in report['far_snaps'])
    assert [f[:4] for f in far] == [('episode_000000.json', 'key_event', 'cup tipped', .82),
                                    ('episode_000000.json', 'subtask', 'pour the water', .82)]
    assert all(abs(f[4] - GAP_TIMES[20]) < 1e-6 and f[5] == .153 for f in far)


def test_a_subtask_that_loses_its_frame_to_a_later_boundary_is_listed(tmp_path):
    # tap runs 0.71s to 0.74s inside the gap: its start and the pour's both land on the 0.667s frame, the pour is kept
    # there, and the tap, which then has no frame of its own, is listed instead of dropped without a word
    ep0 = {'event_labels': [step(0., .71, 'right', 'grasp the cup'), step(.71, .74, 'right', 'tap the rim'),
                            step(.74, 1.2, 'right', 'pour the water')]}
    report, table = export_one(tmp_path, ep0, ep0_times=GAP_TIMES, rate=30)
    rows = per_episode(table, 0)
    assert [r['content'] for r in rows[0]['language_persistent']] == ['grasp the cup', 'pour the water', 'idle']
    lost = [(r['episode'], r['text'], r['t_s'], r['replaced_by']) for r in report['subtasks_lost']]
    assert lost == [('episode_000000.json', 'tap the rim', .71, 'pour the water')]
    assert abs(report['subtasks_lost'][0]['frame_s'] - GAP_TIMES[20]) < 1e-6
