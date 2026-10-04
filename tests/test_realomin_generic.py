"""Controlled indexed pose recordings bind native fields and camera clocks."""
import hashlib
import json

import numpy as np
import pytest
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from mcap_protobuf.writer import Writer

from label import episode
from label.frames import extract_frames
from prepare import formats, mcap_pose, realomin
from test_prepare import _h264_frames


def native_types():
    fd = descriptor_pb2.FileDescriptorProto(name='indexed_pose_parity.proto',
                                           package='indexed_pose', syntax='proto3')
    def message(name, fields):
        msg = fd.message_type.add(name=name)
        for number, (field, kind, repeated, ref) in enumerate(fields, 1):
            item = msg.field.add(name=field, number=number, type=kind, label=3 if repeated else 1)
            if ref:
                item.type_name = '.indexed_pose.' + ref
    message('Vector3', [('x', 1, False, None), ('y', 1, False, None), ('z', 1, False, None)])
    message('Quaternion', [('x', 1, False, None), ('y', 1, False, None),
                           ('z', 1, False, None), ('w', 1, False, None)])
    message('Pose', [('position', 11, False, 'Vector3'), ('orientation', 11, False, 'Quaternion')])
    message('Header', [('frame_id', 9, False, None)])
    message('PoseStamped', [('pose', 11, False, 'Pose'), ('frame_id', 9, False, None), ('header', 11, False, 'Header')])
    message('CompressedVideo', [('format', 9, False, None), ('data', 12, False, None),
                                ('frame_id', 9, False, None)])
    message('Scalar', [('value', 1, False, None)])
    message('Imu', [('angular_velocity', 11, False, 'Vector3'),
                    ('linear_acceleration', 11, False, 'Vector3')])
    pool = descriptor_pool.DescriptorPool()
    pool.Add(fd)
    return {name: message_factory.GetMessageClass(pool.FindMessageTypeByName('indexed_pose.' + name))
            for name in ['PoseStamped', 'CompressedVideo', 'Scalar', 'Imu']}


def indexed_recording(path, *, omit_pose=None, invalid_quaternion=False, actor_map=True, changing_frame=None):
    types = native_types()
    packets = _h264_frames(6)
    base = 1_700_000_000_000_000_000
    stamps = base + np.asarray([0, 32, 69, 101, 142, 177], dtype=np.int64) * 1_000_000
    with Writer(str(path)) as writer:
        metadata = {
            'position_unit': 'metres', 'orientation_order': 'xyzw',
            'magnetic_encoder_unit': 'metres'}
        if actor_map:
            metadata['actor_map'] = json.dumps({'robot0': 'left', 'robot1': 'right'})
        writer._writer.add_metadata('source', metadata)
        for robot, side in [(0, 'left'), (1, 'right')]:
            for index, stamp in enumerate(stamps):
                now = int(stamp) + robot * 2_000_000
                writer.write_message(f'/robot{robot}/sensor/camera0/compressed',
                    types['CompressedVideo'](format='h264', data=packets[index], frame_id=side + '_gripper'),
                    log_time=now, publish_time=now - 700_000_000)
                if robot != omit_pose:
                    pose = types['PoseStamped'](frame_id=side + '_gripper')
                    if changing_frame == 'header':
                        pose.ClearField('frame_id')
                        pose.header.frame_id = side + '_gripper'
                    if changing_frame and robot == 0 and index == 2:
                        if changing_frame == 'header':
                            pose.header.frame_id = 'different_coordinate_frame'
                        else:
                            pose.frame_id = 'different_coordinate_frame'
                    pose.pose.position.x = 0.1 + robot + index * 0.01
                    pose.pose.position.y, pose.pose.position.z = 0.2, 0.3
                    angle = index * 0.025
                    pose.pose.orientation.z, pose.pose.orientation.w = np.sin(angle), np.cos(angle)
                    if invalid_quaternion and index == 2:
                        pose.pose.orientation.w = 3.0
                    writer.write_message(f'/robot{robot}/vio/eef_pose', pose,
                                         log_time=now, publish_time=now - 700_000_000)
                writer.write_message(f'/robot{robot}/sensor/magnetic_encoder',
                    types['Scalar'](value=0.01 + index * 0.002 + robot * 0.001), log_time=now)
                imu = types['Imu']()
                imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = index * 0.01, 0.2, 0.3
                imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z = 0.4, 0.5, 9.8
                writer.write_message(f'/robot{robot}/sensor/imu', imu, log_time=now)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def converted(root, out, monkeypatch, *, legacy):
    original = formats.upload_adapters
    with monkeypatch.context() as patch:
        patch.setattr(formats, 'upload_adapters', lambda kind: [realomin] if kind == 'mcap' and legacy
                      else [] if kind == 'mcap' else original(kind))
        report = formats.convert(root, 'handheld_gripper', out, 'indexed native fixture', 900)
    assert not report['failed'] and len(report['episodes']) == 1, report
    path = out / report['episodes'][0]['episode_id']
    return path, json.loads((path / 'context.json').read_text())


def test_recorded_pose_gripper_join_matches_original_native_reader(tmp_path, monkeypatch):
    root = tmp_path / 'native'
    root.mkdir()
    raw = root / 'episode.mcap'
    digest = indexed_recording(raw)
    old_ep, old = converted(root, tmp_path / 'old', monkeypatch, legacy=True)
    new_ep, new = converted(root, tmp_path / 'generic', monkeypatch, legacy=False)
    assert new['state_kind'] == old['state_kind'] == 'ee_pose'
    assert new['n_state_frames'] == old['n_state_frames'] == 6
    with np.load(old_ep / 'state.npz') as a, np.load(new_ep / 'state.npz') as b:
        assert a['state'].shape == b['state'].shape == (6, 14)
        assert a['state'].dtype == b['state'].dtype
        assert a['state'].tobytes() == b['state'].tobytes()
    clock_facts = {}
    with np.load(old_ep / 'times.npz') as a, np.load(new_ep / 'times.npz') as b:
        for key in a.files:
            assert key in b.files and a[key].dtype == b[key].dtype
            if key.endswith('_pts'):
                # Original wrapper encoded each camera from zero; generic clips keep the common clock offset.
                assert np.array_equal(a[key] - a[key][0], b[key] - b[key][0]), key
            else:
                assert np.array_equal(a[key], b[key]), key
            clock_facts[key] = {'old': a[key].tolist(), 'generic': b[key].tolist(), 'dtype': str(a[key].dtype)}
    with np.load(old_ep / 'signals.npz') as a, np.load(new_ep / 'signals.npz') as b:
        for old_entry in old['signals']:
            new_entry = next(row for row in new['signals'] if row['name'] == old_entry['name'])
            assert a[old_entry['key']].dtype == b[new_entry['key']].dtype
            assert a[old_entry['key']].tobytes() == b[new_entry['key']].tobytes()
    before, after = episode.build_request(old_ep), episode.build_request(new_ep)
    assert before['plan']['ks'] == after['plan']['ks']
    for view in ['left', 'right']:
        decoded = []
        for path in [old_ep, new_ep]:
            source = json.loads((path / 'sources.json').read_text())[view]
            with np.load(path / 'times.npz') as times:
                frames = extract_frames(source['packed'], 0, 6, after['plan']['ks'], pts=times[view + '_pts'])
            decoded.append({key: image.tobytes() for key, image in frames.items()})
        assert decoded[0] == decoded[1]
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == digest
    print(json.dumps({'original_sha256': digest, 'state_dtype': 'float32', 'state_shape': [6, 14],
                      'native_capture_arrays_exact': True, 'encoded_clock_facts': clock_facts,
                      'selected_indices': after['plan']['ks'],
                      'qualified_difference': 'Old wrapper starts each encoded clip at zero. Generic encoded PTS retain the shared camera offset. Capture arrays and decoded selected frames remain exact.'}))


def test_structural_pose_join_matches_literal_old_arrays(tmp_path):
    raw = tmp_path / 'episode.mcap'
    indexed_recording(raw)
    old_ep = tmp_path / 'old'
    realomin.convert(raw, old_ep, 'upload/episode.mcap')
    q = (1_700_000_000_000_000_000 + np.asarray([0, 32, 69, 101, 142, 177], dtype=np.int64) * 1_000_000) / 1e9
    state, used, ctx, limitation = mcap_pose.pose_state([raw], q, 'arrival')
    assert limitation is None
    with np.load(old_ep / 'state.npz') as original:
        assert state.dtype == original['state'].dtype
        assert state.tobytes() == original['state'].tobytes()
    assert len(used) == 4
    assert formats.state_contract_actors(ctx, 2) == ['left', 'right']
    assert [row['side'] for row in ctx['state_identities']] == [None, None]
    assert ctx['pose_field_bindings'][0]['actor'] == 'robot0'


@pytest.mark.parametrize('kwargs', [{'omit_pose': 1}, {'invalid_quaternion': True},
                                    {'changing_frame': 'direct'}, {'changing_frame': 'header'}])
def test_partial_or_invalid_pose_never_creates_precise_state(tmp_path, kwargs):
    raw = tmp_path / 'episode.mcap'
    indexed_recording(raw, **kwargs)
    q = (1_700_000_000_000_000_000 + np.arange(6, dtype=np.int64) * 32_000_000) / 1e9
    state, used, ctx, limitation = mcap_pose.pose_state([raw], q, 'arrival')
    assert state is None and not used and not ctx and limitation


def test_index_numbers_alone_never_establish_camera_sides(tmp_path):
    raw = tmp_path / 'episode.mcap'
    indexed_recording(raw, actor_map=False)
    topics = list(realomin.CAMERA_TOPICS)
    assert mcap_pose.camera_roles(raw, topics) == ({}, {})
    indexed_recording(raw, actor_map=True)
    aliases, provenance = mcap_pose.camera_roles(raw, topics)
    assert aliases[topics[0]].startswith('left ')
    assert aliases[topics[1]].startswith('right ')
    assert provenance['source'] == 'MCAP actor_map metadata'


def test_coordinate_frame_names_do_not_supply_actor_sides(tmp_path):
    raw = tmp_path / 'episode.mcap'
    indexed_recording(raw, actor_map=False)
    q = (1_700_000_000_000_000_000 + np.asarray([0, 32, 69, 101, 142, 177], dtype=np.int64) * 1_000_000) / 1e9
    state, _, ctx, limitation = mcap_pose.pose_state([raw], q, 'arrival')
    assert state is not None and limitation is None
    assert ctx['pose_field_bindings'][0]['frame_ids'] == ['left_gripper']
    assert formats.state_contract_actors(ctx, 2) is None
    assert all(row['side'] is None for row in ctx['state_identities'])
