"""LeRobot's own metadata never opts a recording into evidence inspection (label/evidence_access.py needs_inspection).

The fixture is episode 0's recorded metadata from two public Unitree Go2 LeRobot v3 datasets, as the reader kept it:
an RGB-only upload (dancher00/go2_object_approach_v1) and one with a depth video (nullPointerExcept1on/
realworld_go2_task0). Before the fix the reader marked only part of info.json as represented, so the per-episode
rows (task text, row and video spans, stats of every column) and the video encoder settings opted every LeRobot
recording into inspection by default, RGB-only ones included.
"""
import copy
import json
from pathlib import Path

import pytest

from label import evidence_access as ea
from label.source_eligibility import _covered, _leaves, needs_extra_evidence
from prepare.formats import lerobot_metadata

FIXTURE = json.loads((Path(__file__).parent / 'fixtures' / 'lerobot_v3_go2_metadata.json').read_text())
DEPTH_CALIBRATION = {'depth_unit', 'video.depth_min', 'video.depth_max', 'video.shift', 'video.use_log'}


def prepared(tmp_path, name, change=None):
    case = copy.deepcopy(FIXTURE[name])
    root = tmp_path / case['dataset']
    path = str(root / case['episode_metadata'])
    rows = [{**row, 'source': path} for row in case['rows']]
    if change:
        change(rows[0]['fields'])
    ctx = {'episode_index': case['episode_index']}
    lerobot_metadata(ctx, {'dir': str(root), 'info': case['info'],
                           'episode_metadata': {case['episode_index']: {path: rows}}})
    return ctx


def uncovered(ctx):
    covered = ctx.get('represented_source_members', {}).get('json', {})
    return {source: [p for p in _leaves(value) if not _covered(p, covered.get(source, []))]
            for source, value in ctx['recorded_metadata'].items()}


def test_an_rgb_only_lerobot_recording_is_not_inspected_by_default(tmp_path):
    ctx = prepared(tmp_path, 'rgb_only')
    assert not any(uncovered(ctx).values()), uncovered(ctx)
    assert not needs_extra_evidence(ctx)
    ep = {'dir': tmp_path, 'context': ctx, 'signals': {}, 'sources': {}}
    assert not ea.needs_inspection(ep, {}, 'auto')
    # nothing left over for the prompt's metadata note, which the stats alone had pushed past its limit
    assert 'uploader_notes' not in ctx
    assert not [i for i in ctx.get('reader_issues', []) if i.get('kind') == 'metadata_limit']


def test_depth_calibration_stays_evidence_the_prompt_does_not_carry(tmp_path):
    left = uncovered(prepared(tmp_path, 'rgb_and_depth'))
    info = next(source for source in left if source.endswith('info.json'))
    assert {p.rsplit('/', 1)[-1] for p in left[info]} == DEPTH_CALIBRATION
    assert all('observation.images.front_depth' in p for p in left[info])
    assert not any(left[source] for source in left if source != info)


@pytest.mark.parametrize('key, value', [('success', True),                          # a publisher's own column
                                        ('stats/observation.unread/max', [1.0]),     # stats of an undeclared column
                                        ('videos/observation.images.front/note', 'lens swapped')])
def test_a_column_beyond_lerobot_bookkeeping_still_opts_the_recording_in(tmp_path, key, value):
    ctx = prepared(tmp_path, 'rgb_only', lambda fields: fields.__setitem__(key, value))
    assert needs_extra_evidence(ctx)
    left = [p for paths in uncovered(ctx).values() for p in paths]
    assert len(left) == 1 and _covered(left[0], ['/0/fields/' + key.replace('/', '~1')])
