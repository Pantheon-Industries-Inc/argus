"""A tactile sensor's own image never takes a scene or gripper slot, so it never times the episode.

The recordings copy the shape of the audited Sparsh bag: a DIGIT 360 publishes /d360_0/image_raw/compressed beside
/d360_0/pressure_topic, and a RealSense publishes the scene. By name the D360 image sorted first and became the
anchor gripper camera whose clock timed everything else.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from prepare import formats as f
from test_reader_cameras import _json_mcap

D360 = '/d360_0/image_raw/compressed'
PRESSURE = '/d360_0/pressure_topic'
SCENE = '/realsense/color/image_raw/compressed'


def _jpeg_of(image) -> dict:
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=95)
    return {'format': 'jpeg', 'data': base64.b64encode(buffer.getvalue()).decode()}


def gel(k: int) -> dict:
    """The sensor's lit gel in a dark field, the same picture in every frame up to sensor noise."""
    image = Image.new('RGB', (160, 96), (3, 3, 3))
    ImageDraw.Draw(image).ellipse((50, 12, 110, 84), fill=(90, 160, 120))
    noisy = np.asarray(image, dtype=float) + np.random.default_rng(k).normal(0, 0.5, (96, 160, 3))
    return _jpeg_of(Image.fromarray(np.clip(noisy, 0, 255).astype('uint8')))


def scene(k: int) -> dict:
    """A moving scene: a bright block sweeping across a textured table."""
    rng = np.random.default_rng(100)
    table = rng.integers(40, 200, (96, 160, 3)).astype('uint8')
    image = Image.fromarray(table)
    ImageDraw.Draw(image).rectangle((4 * k % 140, 20, 4 * k % 140 + 30, 60), fill=(250, 250, 250))
    return _jpeg_of(image)


def pressure(k: int) -> dict:
    return {'pressure': 106800.0 + 5 * k, 'temperature': 30.0 + 0.01 * k}


def prepare(tmp_path: Path, chans: dict, rig: str = 'handheld_gripper') -> tuple[Path, dict, dict]:
    root = tmp_path / 'upload'
    root.mkdir()
    _json_mcap(root / 'lego_tap.mcap', chans, n=60)
    report = f.convert(root, rig, tmp_path / 'episodes', 'sparsh_shape', 900)
    assert report['episodes'], report
    ep = next((tmp_path / 'episodes').iterdir())
    return ep, json.loads((ep / 'context.json').read_text()), json.loads((ep / 'sources.json').read_text())


def test_d360_image_never_becomes_the_anchor_and_the_scene_camera_times_the_episode(tmp_path):
    from label import episode as me
    ep, ctx, src = prepare(tmp_path, {D360: ('sensor_msgs/msg/CompressedImage', gel),
                                      PRESSURE: ('d360_msgs/msg/PressureD360', pressure),
                                      SCENE: ('sensor_msgs/msg/CompressedImage', scene)})
    order = me.order_views(src)
    assert src[order[0]]['camera_key'] == SCENE
    tactile_view = next(v for v in order if src[v]['camera_key'] == D360)
    assert tactile_view.startswith('extra')
    classes = ctx['source']['camera_classes']
    assert classes[D360]['class'] == 'optical_tactile' and classes[SCENE]['class'] == 'camera'
    # nothing is lost: both cameras are slotted, the pressure is a signal on the scene camera's clock
    assert {s['camera_key'] for s in src.values()} == {D360, SCENE}
    assert any(m['name'].startswith(PRESSURE) for m in ctx['signals'])
    times = np.load(ep / 'times.npz')
    assert times[order[0]][0] == 0 and len(times[order[0]]) == src[order[0]]['n_frames']
    # the labeller reads the same classes from the prepared episode
    from label.camera_class import camera_class
    from prepare.camera_slots import sample_frames
    assert camera_class(ctx, tactile_view, sample_frames(src[tactile_view]['packed']))['optical']


def test_recording_with_only_tactile_cameras_says_so(tmp_path):
    ep, ctx, src = prepare(tmp_path, {D360: ('sensor_msgs/msg/CompressedImage', gel),
                                      PRESSURE: ('d360_msgs/msg/PressureD360', pressure)})
    issues = [i for i in ctx.get('reader_issues') or [] if i['kind'] == 'only_tactile_cameras']
    assert len(issues) == 1 and D360 in issues[0]['what']
    assert [s['camera_key'] for s in src.values()] == [D360]


def test_a_still_scene_beside_a_robot_state_stays_the_scene_camera(tmp_path):
    from label import episode as me
    still = {'/robot/camera/image/compressed': ('sensor_msgs/msg/CompressedImage', lambda k: scene(0)),
             '/robot/gripper_pressure': ('std_msgs/Float64', lambda k: {'data': 1.0 + 0.1 * k}),
             '/robot/joint_states': ('sensor_msgs/JointState', lambda k: {'position': [0.01 * k] * 6})}
    _, ctx, src = prepare(tmp_path, still)
    assert src[me.order_views(src)[0]]['camera_key'] == '/robot/camera/image/compressed'
    assert ctx['source']['camera_classes']['/robot/camera/image/compressed']['class'] == 'camera'
    assert not [i for i in ctx.get('reader_issues') or [] if i['kind'] == 'only_tactile_cameras']


def test_slots_rank_scene_cameras_by_clock_then_frames_and_keep_tactile_names_out():
    names = ['/d360_0/image_raw/compressed', '/realsense/color/image_raw', '/realsense/color/image_raw/compressed']
    out, unused = f.assign_views(names, 'teleop_arms', tactile={names[0]},
                                 prefer=lambda name: (0, -124 if name.endswith('compressed') else -122))
    assert out['exo'] == names[2] and names[0] in {v for k, v in out.items() if k.startswith('extra')}
    assert not unused
