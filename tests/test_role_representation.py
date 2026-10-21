"""A dictionary role is read against the field it describes: a numeric array is never dropped for an image role.

The paid ActionSense dictionary called two numeric 32 x 32 glove arrays tactile_image. The numeric evidence selector
read any role it could not consume as "not touch" before its name fallback, so both gloves left the default request.
These tests give numeric glove arrays of that shape the same image role.
"""
import numpy as np
import pytest

from label import sensor_evidence as se
from label import signals as sg
from test_dictionary_roles import bind, fixture


def glove(n=60):
    """A 32 x 32 glove pressure array: a resting pad, then one region pressed."""
    a = np.full((n, 32, 32), 600.0)
    a[n // 2:, 4:9, 10:14] += np.linspace(0, 90, n - n // 2)[:, None, None]
    return a.reshape(n, -1)


@pytest.mark.parametrize('name', ['tactile-glove-left/tactile_data/data', 'left_glove/data'])
@pytest.mark.parametrize('role', ['tactile_image', 'optical_tactile'])
def test_a_numeric_glove_called_an_image_by_the_dictionary_stays_numeric_touch_evidence(name, role):
    ep, record = fixture(name=name, array=glove(), role=role, shape=[32, 32])
    ep = bind(ep, record)
    doc = se.build(ep, {'n': 60, 'ks': [0, 20, 40, 59]})
    sensor = next((s for s in doc['sensors'] if s['id'] == 'signal:' + name), None)
    assert sensor is not None, doc['coverage']
    assert sensor['quantity'] == 'sensor_response' and sensor.get('array_evidence'), sensor.keys()
    assert [c['status'] for c in doc['coverage'] if c['name'] == name] != ['unsupported interpretation']
    assert any(note.startswith(name + ': the field dictionary called this signal ' + role)
               for note in doc['limitations']), doc['limitations']


def test_a_numeric_field_called_a_scene_image_is_read_by_its_own_name():
    touch, plain = glove(), glove()
    ep, record = fixture(name='glove_pressure', array=touch, role='scene_image', shape=[32, 32])
    assert any(s['id'] == 'signal:glove_pressure' for s in se.build(bind(ep, record), {'n': 60, 'ks': [0, 59]})['sensors'])
    ep, record = fixture(name='channel0', array=plain, role='scene_image', shape=[32, 32])
    assert not any(s['id'] == 'signal:channel0' for s in se.build(bind(ep, record), {'n': 60, 'ks': [0, 59]})['sensors'])


def test_roles_keep_their_meaning_and_only_a_representation_mismatch_is_read_again():
    from label.dictionary_context import image_role
    for role in ('touch', 'tactile', 'pressure', 'force_torque', 'event_flag', 'joint_state', 'temperature', ''):
        assert sg.signal_role(role) == role
    assert sg.signal_role('tactile_image') == sg.signal_role('Optical_Tactile ') == 'touch'
    assert sg.signal_role('scene_image') == sg.signal_role('sensor_view') == ''
    for role in ('tactile_image', 'optical_tactile', 'scene', 'touch', 'tactile', 'event_flag', ''):
        assert image_role(role) == role
    assert image_role('pressure') == image_role('normal_force') == 'tactile' and image_role('force_torque') == ''
