"""Recorded camera semantics survive generic interpretations without inventing markers."""
import pytest
from PIL import Image

from label import sensor_evidence as se
from tests.test_optical_motion import pad
from tests.test_tactile_contract import episode


def camera_evidence(meta, interpretation=None):
    meta = {'name': 'channel0', **meta}
    ep = episode({}, [], cameras={'exo': {'name': 'scene'}, 'extra1': meta})
    if interpretation is not None:
        ep['context']['data_dictionary'] = {
            'episode_id': 'episode_test',
            'fields': [{'id': 'pad', 'kind': 'camera', 'name': meta['name'],
                        'bindings': [{'episode': 'episode_test', 'context_path': 'cameras/extra1'}]}],
            'entries': {'pad': interpretation}}
    doc = se.build(ep, {'n': 5, 'ks': [0, 4]}, shown={'extra1': [0, 4]},
                   images={'extra1': {0: pad(), 4: pad(2)}})
    return ep, doc


@pytest.mark.parametrize('declaration', [
    {'content_role': 'optical_tactile'},
    {'recorded_role': 'optical_tactile'},
    {'role': 'optical_tactile'},
    {'sensor_type': 'optical_tactile'},
])
def test_recorded_optical_cameras_supply_deformation_and_motion(declaration):
    _, doc = camera_evidence(declaration)
    assert len(doc['sensors']) == 1
    sensor = doc['sensors'][0]
    assert sensor['quantity'] == 'optical_tactile'
    assert {'image_deformation', 'contact_shape_change', 'marker_motion'} <= set(sensor['capabilities'])
    assert doc['coverage'] == [{'id': 'camera:extra1', 'name': 'channel0', 'status': 'sampled', 'samples': 2}]


def test_recorded_tactile_role_survives_generic_machine_rgb_interpretation():
    _, doc = camera_evidence({'content_role': 'tactile_image'},
                            {'role': 'rgb_image', 'provenance': 'machine'})
    assert len(doc['sensors']) == 1
    # still a tactile sensor; its fixed marker view makes it optical (label/camera_class.py)
    assert doc['sensors'][0]['quantity'] == 'optical_tactile'
    assert doc['sensors'][0]['class_basis'].startswith('recorded content role tactile')


@pytest.mark.parametrize('provenance', ['machine', 'human'])
def test_generic_tactile_interpretation_retains_optical_brand_capabilities(provenance):
    _, doc = camera_evidence({'name': 'gelsight'}, {'role': 'tactile_image', 'provenance': provenance})
    sensor = doc['sensors'][0]
    assert sensor['quantity'] == 'optical_tactile'
    finding = {'start_s': 0., 'end_s': .4, 'headline': 'The contact patch changes shape',
               'claim_type': 'image_deformation', 'adds_beyond_video': True,
               'evidence': [{'sensor_id': 'camera:extra1', 'time_s': [0., .4]}]}
    assert se.bind(doc, [finding])['findings']


@pytest.mark.parametrize('role', ['scene', 'rgb_image', 'joint_state', None])
def test_explicit_human_exclusions_override_recorded_optical_metadata(role):
    _, doc = camera_evidence({'content_role': 'optical_tactile', 'sensor_type': 'optical_tactile'},
                            {'role': role, 'provenance': 'human'})
    assert not doc['sensors']


@pytest.mark.parametrize('declaration', [
    {'description': 'Rendered pressure heatmap; pixels are not physical markers'},
    {'sensor_type': 'heatmap'},
])
def test_rendered_source_descriptors_do_not_supply_optical_deformation_or_markers(declaration):
    _, doc = camera_evidence({'name': 'gelsight', **declaration}, {'role': 'tactile_image'})
    sensor = doc['sensors'][0]
    assert sensor['quantity'] == 'tactile_image'
    assert sensor['capabilities'] == ['image_response_change']
    assert 'marker_motion' not in sensor


def test_markerless_optical_images_keep_image_capabilities_without_tracking():
    ep, _ = camera_evidence({'sensor_type': 'optical_tactile'})
    doc = se.build(ep, {'n': 5, 'ks': [0, 4]}, shown={'extra1': [0, 4]},
                   images={'extra1': {0: Image.new('RGB', (240, 160), 'blue'),
                                      4: Image.new('RGB', (240, 160), 'red')}})
    assert len(doc['sensors']) == 1
    assert doc['sensors'][0]['capabilities'] == ['image_deformation', 'contact_shape_change', 'slip_candidate']
