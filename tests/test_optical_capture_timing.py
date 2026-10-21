"""Mapped tactile captures retain their own event-time bounds."""
import json

import numpy as np

from label import episode as me, evidence_access as ea, sensor_evidence as se
from prepare.formats import nearest, probe
from tests.test_label import _packed_mp4
from tests.test_optical_motion import pad
from tests.test_tactile_contract import episode


def mapped_episode():
    ep = episode({}, [], cameras={'exo': {'name': 'scene'}, 'extra1': {'name': 'gelsight'}})
    anchor = np.array([0., 1., 2., 3.])
    tactile = np.array([0., 3.])
    ep.update(state=np.zeros((4, 0)), times={'exo': anchor, 'extra1': tactile},
              sources={'exo': {'n_frames': 4}, 'extra1': {'n_frames': 2}},
              kmap={'extra1': nearest(tactile, anchor)})
    ep['context'].update(state_kind='none', profile='teleop_arms')
    return ep


def test_marker_pairs_use_distinct_capture_times_across_a_camera_gap():
    ep = mapped_episode()
    images = {k: pad(2 if source else 0) for k, source in enumerate(ep['kmap']['extra1'])}
    doc = se.build(ep, {'n': 4, 'ks': [0, 1, 2, 3]}, shown={'extra1': [0, 1, 2, 3]},
                   images={'extra1': images})
    sensor = doc['sensors'][0]
    assert sensor['times'] == [0., 3.]
    assert next(row for row in doc['coverage'] if row['id'] == 'camera:extra1')['samples'] == 2
    pairs = sensor['marker_motion']['pairs']
    assert [(pair['from_s'], pair['to_s']) for pair in pairs] == [(0., 3.)]
    invented = {'start_s': 1., 'end_s': 2., 'headline': 'The contact shifts',
                'claim_type': 'marker_motion', 'adds_beyond_video': True,
                'evidence': [{'sensor_id': 'camera:extra1', 'time_s': [1., 2.]}]}
    assert not se.bind(doc, [invented])['findings']


def test_identity_mapping_omission_does_not_replace_camera_capture_times():
    ep = episode({}, [], cameras={'exo': {'name': 'scene'}, 'extra1': {'name': 'gelsight'}})
    ep['times']['extra1'] = np.array([.05, .15, .25, .35, .45])
    doc = se.build(ep, {'n': 5, 'ks': [0, 4]}, shown={'extra1': [0, 4]},
                   images={'extra1': {0: pad(), 4: pad(2)}})
    assert doc['sensors'][0]['times'] == [.05, .45]


def test_missing_side_capture_clock_cannot_support_synchronized_findings():
    ep = mapped_episode()
    ep['times'].pop('extra1')
    doc = se.build(ep, {'n': 4, 'ks': [0, 3]}, shown={'extra1': [0, 3]},
                   images={'extra1': {0: pad(), 3: pad(2)}})
    assert doc['sensors'][0]['episode_alignment_verified'] is False
    assert doc['sensors'][0]['times'] == []


def test_boundary_context_images_do_not_authorize_events_outside_the_scene():
    ep = mapped_episode()
    ep['times'] = {'exo': np.array([0., 1., 2., 3.9]),
                   'extra1': np.array([-.04, .2, 3.7, 4.01])}
    ep['sources']['exo']['last_frame_interval_s'] = .1
    ep['sources']['extra1']['n_frames'] = 4
    ep['kmap']['extra1'] = np.arange(4)
    doc = se.build(ep, {'n': 4, 'ks': [0, 1, 2, 3]}, shown={'extra1': [0, 1, 2, 3]},
                   images={'extra1': {k: pad(k) for k in range(4)}})
    assert doc['sensors'][0]['times'] == [-.04, .2, 3.7, 4.01]
    for start, end, cited, valid in ((-.04, -.04, [-.04], False),
                                    (4.01, 4.01, [4.01], False),
                                    (3.7, 4.0000005, [3.7, 4.01], True),
                                    (3.7, 4.000002, [3.7, 4.01], False),
                                    (-.0000005, .2, [-.04, .2], True)):
        finding = {'start_s': start, 'end_s': end, 'headline': 'The pad response changes',
                   'claim_type': 'image_deformation',
                   'evidence': [{'sensor_id': 'camera:extra1', 'time_s': cited}]}
        assert bool(se.bind(doc, [finding])['findings']) is valid
    assert doc['episode_window_s'] == [0., 4.]
    assert 'episode_window_s' not in se.prompt(doc)
    legacy = dict(doc)
    legacy.pop('episode_window_s')
    assert se.bind(legacy, [{'start_s': -.04, 'end_s': -.04, 'headline': 'Legacy boundary context',
        'claim_type': 'image_deformation',
        'evidence': [{'sensor_id': 'camera:extra1', 'time_s': [-.04]}]}])['findings']


def test_inspected_boundary_context_does_not_authorize_outside_events(tmp_path):
    movie = tmp_path / 'packed.mp4'
    _packed_mp4(movie, 4)
    pts = probe(movie)['pts']
    ep = mapped_episode()
    ep['dir'] = tmp_path
    ep['times'] = {'exo': np.array([0., 1., 2., 3.9]), 'exo_pts': pts,
                   'extra1': np.array([-.04, .2, 3.7, 4.01]), 'extra1_pts': pts}
    ep['sources'] = {'exo': {'packed': str(movie), 'base_s': 0., 'n_frames': 4,
                             'last_frame_interval_s': .1},
                     'extra1': {'packed': str(movie), 'base_s': 0., 'n_frames': 4}}
    ep['kmap']['extra1'] = np.arange(4)
    for plan in (None, {'n': 3}):
        access = ea.Access(ep, plan=plan)
        field = next(field for field in access.inventory() if field['kind'] == 'image' and field['name'] == 'extra1')
        receipt = access.inspect({'field_id': field['id'], 'mode': 'images', 'rows': [0, 1, 2, 3]})
        assert receipt['times_s'] == [-.04, .2, 3.7, 4.01]
        for start, end, cited, valid in ((-.04, -.04, [-.04], False),
                                        (4.01, 4.01, [4.01], False),
                                        (3.7, 3.8, [3.7, 4.01], True),
                                        (3.7, 3.95, [3.7, 4.01], plan is None)):
            finding = {'start_s': start, 'end_s': end, 'headline': 'Inspected pad response',
                'adds_beyond_video': True, 'evidence': [{'evidence_id': receipt['id'], 'time_s': cited}]}
            assert bool(access.bind([finding])['findings']) is valid


def test_declared_fixed_grid_keeps_its_episode_times_without_capture_arrays():
    ep = episode({}, [], cameras={'exo': {'name': 'scene'}, 'extra1': {'name': 'gelsight'}})
    ep['times'] = None
    doc = se.build(ep, {'n': 5, 'ks': [0, 4]}, shown={'extra1': [0, 4]},
                   images={'extra1': {0: pad(), 4: pad(2)}})
    assert doc['sensors'][0]['times'] == [0., .4]
    assert doc['sensors'][0]['episode_alignment_verified'] is True


def test_source_capture_mapping_is_supplied_beside_anchor_grid_times():
    ep = mapped_episode()
    text = me._camera_capture_text(ep, {'ks': [0, 1, 2, 3]})
    mapping = json.loads(text.splitlines()[-1])
    assert mapping['extra1'] == [[1., 0, 0., -1.], [2., 1, 3., 1.]]
    heading = me._frames_head(ep, 240, 160, (240, 160), capture_mapped=True)
    assert 'The times are exact: use them, do not invent your own.' not in heading


def test_inspection_reports_camera_capture_times_instead_of_anchor_targets(tmp_path):
    movie = tmp_path / 'packed.mp4'
    _packed_mp4(movie, 4)
    pts = probe(movie)['pts']
    ep = mapped_episode()
    ep['dir'] = tmp_path
    ep['times'].update(exo_pts=pts, extra1_pts=pts[:2])
    ep['sources'] = {'exo': {'packed': str(movie), 'base_s': 0., 'n_frames': 4},
                     'extra1': {'packed': str(movie), 'base_s': 0., 'n_frames': 2}}
    access = ea.Access(ep)
    field = next(field for field in access.inventory() if field['kind'] == 'image' and field['name'] == 'extra1')
    receipt = access.inspect({'field_id': field['id'], 'mode': 'images', 'rows': [1, 2]})
    assert receipt['times_s'] == [0., 3.]
    assert [frame['time_s'] for frame in receipt['frames']] == [0., 3.]
    assert [frame['source_frame'] for frame in receipt['source_frames']] == [0, 1]
    captions = [part['text'] for part in access.images[ea.packed([field['id'], {'field_id': field['id'], 'mode': 'images', 'rows': [1, 2]}])]
                if part['type'] == 'text']
    assert '0.000000s (source capture; anchor display 1.000000s)' in captions[0]


def test_inspection_deduplicates_reused_capture_rows(tmp_path):
    movie = tmp_path / 'packed.mp4'
    _packed_mp4(movie, 4)
    pts = probe(movie)['pts']
    ep = mapped_episode()
    ep['dir'] = tmp_path
    ep['times'].update(exo_pts=pts, extra1_pts=pts[:2])
    ep['sources'] = {'exo': {'packed': str(movie), 'base_s': 0., 'n_frames': 4},
                     'extra1': {'packed': str(movie), 'base_s': 0., 'n_frames': 2}}
    access = ea.Access(ep)
    field = next(field for field in access.inventory() if field['kind'] == 'image' and field['name'] == 'extra1')
    receipt = access.inspect({'field_id': field['id'], 'mode': 'images', 'rows': [0, 1, 2, 3]})
    assert receipt['times_s'] == [0., 3.]
    assert receipt['reused_rows'] == [{'row': 1, 'same_capture_as_row': 0}, {'row': 3, 'same_capture_as_row': 2}]
    assert not receipt['unavailable_rows']


def test_normal_request_includes_capture_mapping_without_claiming_exact_side_times(tmp_path):
    movie = tmp_path / 'packed.mp4'
    _packed_mp4(movie, 4)
    pts = probe(movie)['pts']
    ctx = {'profile': 'teleop_arms', 'state_kind': 'none', 'fps': 1, 'real_times': 'times.npz',
           'cameras': {'exo': {'name': 'scene'}, 'extra1': {'name': 'gelsight'}}}
    (tmp_path / 'context.json').write_text(json.dumps(ctx))
    (tmp_path / 'sources.json').write_text(json.dumps({
        'exo': {'packed': str(movie), 'base_s': 0., 'n_frames': 4},
        'extra1': {'packed': str(movie), 'base_s': 0., 'n_frames': 2, 'kmap': 'map.npy'}}))
    np.save(tmp_path / 'map.npy', np.array([0, 0, 1, 1]))
    np.savez(tmp_path / 'times.npz', exo=np.array([0., 1., 2., 3.]), extra1=np.array([0., 3.]),
             exo_pts=pts, extra1_pts=pts[:2])
    request = me.build_request(tmp_path, cell_w=64)
    assert 'CAMERA CAPTURE MAPPING' in request['prompt']
    assert 'The times are exact: use them, do not invent your own.' not in request['prompt']
    assert request['sensor_evidence']['sensors'][0]['times'] == [0., 3.]
