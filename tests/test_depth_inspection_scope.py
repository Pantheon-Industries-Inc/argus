"""Depth alone does not buy an evidence selection call under --inspection auto (label/evidence_access.py
inspection_reason); its pictures still ride in the request, and --inspection all still inspects it. The dry run
records why each unit will or will not be inspected, and review counts it before spending."""
import argparse
import json

import h5py
import numpy as np

from label import episode, evidence_access as ea, policy
from prepare import formats


def depth_episode(tmp_path):
    """An HDF5 demo with one RGB camera and a millimetre depth stream, prepared as an upload would be."""
    root = tmp_path / 'upload'
    root.mkdir()
    with h5py.File(root / 'capture.hdf5', 'w') as source:
        demo = source.create_group('data/demo_0')
        demo.create_dataset('images_top', data=np.full((6, 64, 80, 3), [70, 110, 130], dtype=np.uint8))
        demo.create_dataset('timestamps', data=10**12 + np.arange(6, dtype=np.int64) * 100_000_000)
        demo.create_dataset('depth_top', data=np.full((6, 64, 80), 500, dtype=np.uint16)).attrs['units'] = 'mm'
    report = formats.convert(root, 'ego_head', tmp_path / 'episodes', 'depth', 900)
    assert not report['failed']
    return tmp_path / 'episodes' / report['episodes'][0]['episode_id']


def test_a_depth_recording_is_not_inspected_by_default_and_is_with_all(tmp_path):
    home = depth_episode(tmp_path)
    assert episode.load(home)['depth']
    auto = episode.build_request(home, inspect_evidence=True)
    assert auto['evidence_access'] is None and auto['inspection_reason'] is None
    assert auto['depth_s'] and 'depth' in auto['blocks']    # its pictures still ride in the request
    every = episode.build_request(home, inspect_evidence=True, inspection='all')
    assert every['evidence_access'] is not None and every['inspection_reason'] == 'all'


def test_the_help_text_says_what_each_mode_inspects():
    parser = argparse.ArgumentParser()
    policy.add_arguments(parser)
    text = next(a.help for a in parser._actions if '--inspection' in a.option_strings)
    auto, _, every = text.partition(', all ')
    assert 'touch sensor' in auto and 'fields the prompt does not carry' in auto
    assert 'depth' not in auto and 'depth' in every


def test_the_dry_run_plan_counts_units_by_reason(tmp_path):
    for i, reason in enumerate(['touch', None, 'unrepresented', 'touch', 'all']):
        (tmp_path / f'episode_{i:06d}.json').write_text(json.dumps({'dry_run': True, 'inspection': reason}))
    plan = ea.inspection_plan(tmp_path)
    assert plan == {'touch sensor': 2, 'none': 1, 'recorded fields the prompt does not carry': 1,
                    '--inspection all': 1}
    lines = ea.inspection_plan_lines(plan)
    assert lines[0].endswith('4 of 5 units') and '  touch sensor: 2' in lines and len(lines) == 4
