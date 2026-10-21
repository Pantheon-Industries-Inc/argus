"""A measurement the recording holds reaches the model in a form that shows it.

Each case is built from the shape of a real audited input: the UMI tactile board (384 unnamed components on a 30 fps
LeRobot table, one declared group), the Sparsh DIGIT 360 bag (ROS topics /d360_0/image_raw/compressed beside
/d360_0/pressure_topic, no dictionary role for the cameras), and the Inspire index-nail pad (a 12 x 8 x 3 array the
model asked for at four exact rows with no columns).
"""
import numpy as np
import pytest
from pytest import approx
from PIL import Image, ImageDraw

from label import evidence_access as ea
from label import sensor_evidence as se
from label import signals as sg
from label import tactile_capture
from tests.test_tactile_contract import episode


# ---- pooled arrays -------------------------------------------------------------------------------------------------

def board(n=605, width=384, swap=336, scale=1.0):
    """Cells 0..191 pressed and 192..383 free until row swap, then the reverse: the mean stays 0.5 throughout the swap,
    while every cell changes. An early ramp gives the mean its own extrema elsewhere."""
    rng = np.random.default_rng(0)
    a = np.zeros((n, width))
    a[:swap, :width // 2] = 1.0
    a[swap:, width // 2:] = 1.0
    a[:60] *= np.linspace(0, 1, 60)[:, None]
    a += rng.uniform(0, 0.02, a.shape)
    return np.clip(a, 0, 1) * scale


def board_episode(a):
    meta = {'name': 'observation.tactile_left_raw', 'shape': [a.shape[1]]}
    entry = {'role': 'touch', 'provenance': 'machine',
             'meaning': '384-column raw left-tactile observation; individual channel meanings are unspecified.',
             'layout': [{'start': 0, 'count': a.shape[1], 'name': 'raw left-tactile channels'}]}
    ep = episode({meta['name']: a}, [meta], {meta['name']: entry})
    ep['context']['fps'] = 30
    ep['times'] = {view: np.arange(len(a)) / 30 for view in ep['context']['cameras']}
    return ep


def pooled(doc):
    return next(row for row in doc['series'] if row['operation'] == 'mean of declared group')


def test_pooled_board_mean_carries_the_component_change_it_hides():
    a = board()
    doc = se.build(board_episode(a), {'n': len(a), 'ks': list(range(0, len(a), 30))})
    row = pooled(doc)
    change = row['component_change']
    assert change['components'] == 384
    entries = [dict(zip(change['columns'], entry)) for entry in change['rows']]
    # the swap lies between two consecutive supplied times, and the table shows it where the mean does not
    swap = [e for e in entries if e['from_s'] < 336 / 30 <= e['to_s']]
    assert len(swap) == 1 and swap[0]['to_s'] - swap[0]['from_s'] <= 1 / 30 + 1e-9
    assert swap[0]['changed'] == 384 and swap[0]['rising'] == 192 and swap[0]['falling'] == 192
    assert abs(swap[0]['mean_change']) < 0.01
    assert {column for column, _ in swap[0]['largest']} <= set(range(384)) and len(swap[0]['largest']) == 5
    # the threshold is the array's own: a tenth of its typical range here, scaling with it
    assert change['threshold'] == pytest.approx(0.1, rel=0.05)
    scaled = pooled(se.build(board_episode(a * 1000), {'n': len(a), 'ks': [0, 300, 604]}))['component_change']
    assert scaled['threshold'] == pytest.approx(1000 * change['threshold'], rel=1e-6)
    assert 'component_change' in se.prompt(doc)


def test_evidence_selection_sees_the_largest_component_change(tmp_path):
    a = board()
    ep = board_episode(a)
    supplied = se.build(ep, {'n': len(a), 'ks': list(range(0, len(a), 30))})
    access = ea.Access({**ep, 'dir': tmp_path, 'sources': {}, 'action': None, 'depth': {}}, supplied)
    access.state_usable = False
    text = access.selection_content([])[0]['text']
    assert '"largest_component_change"' in text
    assert '"changed":384' in text and '"rising":192' in text and '"falling":192' in text


def test_change_threshold_comes_from_range_and_step_noise():
    flat = np.zeros((50, 8))
    assert sg.change_threshold(flat)[0] == 0
    noisy = np.random.default_rng(1).normal(0, 5, (400, 8)) + np.linspace(0, 20, 400)[:, None]
    threshold, basis = sg.change_threshold(noisy)
    assert basis['step_noise'] > 0
    expected = max(sg.REST_FRACTION * basis['typical_range'], basis['step_noise'])
    assert threshold == approx(expected, rel=1e-5)
    got = sg.component_change(np.zeros(6), np.array([1, -1, 1, -1, 0.01, 0]), 0.5, names=list('abcdef'))
    assert (got['changed'], got['rising'], got['falling'], got['mean_change']) == (4, 2, 2, 0.00166667)
    assert got['largest_names'][:4] == ['a', 'b', 'c', 'd']


def test_readout_summary_counts_values_that_changed():
    a = np.random.default_rng(4).uniform(0, 0.02, (40, 100))
    a[:20, :50] += 1
    a[20:, 50:] += 1
    rows = dict(sg.summary_rows('hand landmarks', a, [0, 19, 20, 39], role='landmarks'))
    label = next(lb for lb in rows if 'values changed by more than' in lb)
    assert rows[label][2] == '100 (50 up, 50 down)'


# ---- optical tactile cameras ---------------------------------------------------------------------------------------

def gel(seed=0, lit=(90, 160, 120)):
    """A DIGIT 360 style frame: a lit circular gel in a dark field, the same in every frame up to sensor noise."""
    im = Image.new('RGB', (320, 180), (2, 2, 2))
    ImageDraw.Draw(im).ellipse((100, 30, 220, 150), fill=lit)
    a = np.asarray(im, dtype=float) + np.random.default_rng(seed).normal(0, 0.5, (180, 320, 3))
    return Image.fromarray(np.clip(a, 0, 255).astype('uint8'))


def scene(seed):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 255, (180, 320, 3), dtype=np.uint8))


def sparsh_context(**dictionary):
    cameras = {'right': {'key': '/d360_0/image_raw/compressed', 'name': 'gripper', 'role_provenance': 'presentation',
                         'mounting': 'unspecified'},
               'extra1': {'key': '/d360_1/image_raw/compressed', 'name': 'd360_1_compressed',
                          'role_provenance': 'presentation', 'mounting': 'unspecified'},
               'extra2': {'key': '/realsense/color/image_raw', 'name': 'realsense',
                          'role_provenance': 'presentation', 'mounting': 'unspecified'}}
    signals = [{'name': '/d360_0/pressure_topic', 'shape': [2], 'names': ['pressure', 'temperature']},
               {'name': '/d360_1/pressure_topic', 'shape': [2], 'names': ['pressure', 'temperature']},
               {'name': '/d360_0/imu_raw_topic (type 1)', 'shape': [3]},
               {'name': '/realsense/color/camera_info k', 'shape': [9]}]
    ep = episode({m['name']: np.ones((6, m['shape'][0])) for m in signals}, signals, dictionary, cameras=cameras)
    ep['context']['profile'] = 'handheld_gripper'
    return ep


FRAMES = {'right': {k: gel(k) for k in range(6)}, 'extra1': {k: gel(10 + k, (60, 140, 150)) for k in range(6)},
          'extra2': {k: scene(k) for k in range(6)}}


def test_d360_cameras_are_optical_tactile_from_their_device_and_fixed_view():
    from label.camera_class import camera_class, touch_device
    ep = sparsh_context(**{'/d360_0/pressure_topic': {'role': 'pressure', 'provenance': 'machine'}})
    ctx = ep['context']
    assert touch_device(ctx, 'right')['namespace'] == '/d360_0'
    for view in ('right', 'extra1'):
        got = camera_class(ctx, view, FRAMES[view])
        assert got['class'] == 'optical_tactile' and got['known'], got
        assert tactile_capture.eligible(ctx, view, FRAMES[view])
    realsense = camera_class(ctx, 'extra2', FRAMES['extra2'])
    assert realsense['class'] == 'camera' and not realsense['known']
    # without its frames the device alone decides nothing: the camera stays a camera, class unknown
    blind = camera_class(ctx, 'right')
    assert blind['class'] == 'camera' and not blind['known']


def test_d360_reaches_sensor_evidence_as_optical_and_unknown_cameras_are_recorded():
    ep = sparsh_context()
    shown = {view: list(range(6)) for view in FRAMES}
    doc = se.build(ep, {'n': 6, 'ks': list(range(6))}, shown=shown, images=FRAMES)
    optical = {s['view']: s for s in doc['sensors'] if s.get('kind') == 'image'}
    assert set(optical) == {'right', 'extra1'}
    assert all(s['quantity'] == 'optical_tactile' and 'image_deformation' in s['capabilities']
               for s in optical.values())
    classes = doc['camera_classes']
    assert [classes[view]['name'] for view in classes if not classes[view]['known']] == ['realsense']
    assert classes['extra2']['class'] == 'camera' and classes['right']['class'] == 'optical_tactile'


def test_evidence_beats_images_and_names_never_decide_alone():
    from label.camera_class import camera_class, touch_device
    # a dictionary role keeps a fixed device camera a camera
    ep = sparsh_context()
    ctx = ep['context']
    ctx['data_dictionary']['fields'].append({'id': 'cam', 'kind': 'camera', 'name': 'gripper', 'bindings': [
        {'episode': 'episode_test', 'context_path': 'cameras/right'}]})
    ctx['data_dictionary']['entries']['cam'] = {'role': 'camera_image', 'provenance': 'machine'}
    assert camera_class(ctx, 'right', FRAMES['right'])['class'] == 'camera'
    # a fixed view without a touch device is a still scene, not a sensor
    still = {'exo': {'key': '/head/image_raw', 'name': 'head'}}
    assert camera_class({'cameras': still, 'signals': []}, 'exo', FRAMES['right'])['tactile'] is False
    # a shared root holding several cameras is not one device
    root = {'cameras': {'a': {'key': 'observations/images/left'}, 'b': {'key': 'observations/images/right'}},
            'signals': [{'name': 'observations/tactile/left'}]}
    assert touch_device(root, 'a') is None
    # a tactile camera whose frames change is a rendering, and one described as a heatmap stays one when still
    moving = sparsh_context()['context']
    moving['cameras']['right']['name'] = 'tactile_left_heatmap'
    assert camera_class(moving, 'right', FRAMES['extra2'])['class'] == 'tactile_image'
    assert camera_class(moving, 'right', FRAMES['right'])['rendered']


# ---- inspection requests -------------------------------------------------------------------------------------------

def pad_access(tmp_path, rows=371):
    """The Inspire index-nail pad: 12 x 8 cells in three equal channels, 371 rows."""
    access = ea.Access({'dir': tmp_path, 'context': {'episode_id': 'episode_a', 'fps': 30, 'state_kind': 'none'},
                        'sources': {}, 'state': np.zeros((rows, 0)), 'action': None, 'times': None,
                        'signals': {}, 'signal_meta': {}, 'depth': {}})
    cells = np.random.default_rng(2).integers(0, 40, (rows, 12, 8, 1))
    access.ep['signals']['observation.tactiles.left_tactile_index_finger_nail'] = np.repeat(cells, 3, axis=3)
    access.refresh()
    field = next(f for f in access.inventory() if f['name'].endswith('index_finger_nail'))
    return access, field['id']


def test_inspection_accepts_exact_rows_over_every_component_that_fits(tmp_path):
    access, ident = pad_access(tmp_path)
    got = access.inspect({'field_id': ident, 'mode': 'samples', 'reason': 'compare the whole pad',
                          'rows': [0, 139, 286, 370]})
    assert got['rows'] == [0, 139, 286, 370]
    assert got['columns'] == list(range(288)) and got['omitted_columns'] == 0
    assert len(ea.packed(got).encode()) <= ea.MAX_RESPONSE_BYTES


def test_column_limit_is_the_response_budget_and_a_refusal_says_what_fits(tmp_path):
    access, ident = pad_access(tmp_path, rows=371)
    wide = np.random.default_rng(3).normal(size=(371, 2000))
    access.ep['signals']['wide'] = wide
    access.refresh()
    field = next(f['id'] for f in access.inventory() if f['name'] == 'wide')
    default = access.inspect({'field_id': field})
    assert 8 < len(default['columns']) < 2000
    assert len(ea.packed(default).encode()) <= ea.MAX_RESPONSE_BYTES
    assert any(f'{len(default["columns"])} of 2000 components fit' in note for note in default['limitations'])
    with pytest.raises(ValueError, match=r'about \d+ components fit at 64 rows, or about \d+ rows'):
        access.inspect({'field_id': field, 'columns': list(range(2000))})
    # fewer rows leave room for more components
    few = access.inspect({'field_id': field, 'rows': [0, 100], 'columns': list(range(500))})
    assert len(few['columns']) == 500


@pytest.mark.parametrize('request_extra, message', [
    ({'row': [0, 5]}, r"unsupported request fields \['row'\] for this \w+ field; it accepts .*rows"),
    ({'rows': [0, 5], 'start_row': 0}, r'cannot be combined with start_row'),
    ({'rows': [5, 0]}, r'increasing and distinct'),
    ({'rows': [0, 999]}, r'rows \[999\] are outside this field, whose source rows are 0 to 370'),
    ({'columns': [0, 400]}, r'columns \[400\] are outside this field, whose components are 0 to 287'),
    ({'mode': 'images'}, r"mode 'images' is unavailable for this \w+ field; use one of samples"),
])
def test_refusals_name_what_to_change(tmp_path, request_extra, message):
    access, ident = pad_access(tmp_path)
    with pytest.raises(ValueError, match=message):
        access.inspect({'field_id': ident, **request_extra})
