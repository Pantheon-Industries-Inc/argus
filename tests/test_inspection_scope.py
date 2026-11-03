"""Which recordings the default inspection reaches (label/evidence_access.py needs_inspection): touch, never a
wrist force/torque sensor alone, whatever the field dictionary did; and an array's regional analysis always keeps the
inspection record it is owned by."""
import numpy as np
import pytest

from label import episode, evidence_access as ea, sensor_evidence as se
from prepare import formats
from test_camera_timing import video
from test_semantic_evidence import descriptor
from test_sensor_analysis import request
from test_tactile_contract import episode as tactile_episode


@pytest.mark.parametrize('role', ['pressure', 'normal_force', 'force_torque'])
def test_an_array_with_regional_analysis_keeps_its_inspection_owner_in_every_mode_whatever_its_quantity(role):
    values = np.full((5, 16), 100.)
    values[3, :8] = 80
    semantics = descriptor(regions=[{'name': 'region A', 'indices': list(range(8))},
                                    {'name': 'region B', 'indices': list(range(8, 16))}])
    ep = tactile_episode({'pad': values}, [{'name': 'pad', 'shape': [4, 4], 'unit': 'ADC',
                                            'sensor_semantics': semantics}], {'pad': {'role': role}})
    doc = se.build(ep, {'n': 5, 'ks': [0, 4]})
    assert request(doc) is not None                                  # the regional analysis needs an owner
    assert all(ea.needs_inspection(ep, doc, mode) for mode in ('auto', 'all', 'off'))


@pytest.mark.parametrize('name, inspected', [('force_torque', False), ('right_wrist_force', False),
                                             ('wrist_wrench', False), ('right_pressure', True),
                                             ('finger_force', True), ('wrist_pressure', True)])
def test_a_wrist_wrench_is_not_inspected_by_default_without_a_dictionary_and_touch_still_is(tmp_path, name, inspected):
    path = tmp_path / 'exo.mp4'
    video(path, 90)
    ep = tmp_path / 'episode_a'
    readings = np.cumsum(np.random.default_rng(0).normal(size=(90, 6)), axis=0)
    formats.video_views_episode(ep, {'exo': ('exo', path)}, 'teleop_arms', 'probe', {},
                                real={'exo': np.arange(90) / 30}, signals={name: readings})
    request_ = episode.build_request(ep, inspect_evidence=True)
    assert (request_['evidence_access'] is not None) == inspected, request_['sensor_evidence'].get('sensors')
