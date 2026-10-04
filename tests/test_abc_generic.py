"""Controlled protobuf MCAP parity keeps capture clocks distinct from message arrival."""
import hashlib
import json

import numpy as np
import pytest
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from mcap_protobuf.writer import Writer

from label import episode
from label.frames import extract_frames
from prepare import abc130k, formats
from test_prepare import _h264_frames


def schema():
    fd = descriptor_pb2.FileDescriptorProto(name='argus_capture_parity.proto', package='capture_parity', syntax='proto3')
    def message(name, fields):
        msg = fd.message_type.add(name=name)
        for i, (name, kind, repeated, ref) in enumerate(fields, 1):
            f = msg.field.add(name=name, number=i, type=kind, label=3 if repeated else 1)
            if ref:
                f.type_name = '.capture_parity.' + ref
    message('Timestamp', [('seconds', 3, False, None), ('nanos', 5, False, None)])
    message('CompressedVideo', [('timestamp', 11, False, 'Timestamp'), ('format', 9, False, None), ('data', 12, False, None)])
    message('Robot', [('timestamp', 11, False, 'Timestamp'), ('position', 1, True, None), ('velocity', 1, True, None)])
    message('Text', [('data', 9, False, None)])
    pool = descriptor_pool.DescriptorPool()
    pool.Add(fd)
    result = {name: message_factory.GetMessageClass(pool.FindMessageTypeByName('capture_parity.' + name))
              for name in ['CompressedVideo', 'Robot', 'Text']}
    result['Video'] = result['CompressedVideo']
    return result


def capture_mcap(path, *, n=6, omit=(), bad_header=None, top='/top-camera', tied=None, instruction=True):
    types = schema()
    frames = _h264_frames(n)
    base = 1_700_000_000_000_000_000
    capture = base + np.arange(n, dtype=np.int64) * 33_000_000
    with Writer(str(path)) as writer:
        writer._writer.add_metadata('source', {'task_name': 'align fabric', 'operator_id': 'fixture operator',
            'station_id': 'fixture station', 'top_camera_width': '64', 'top_camera_height': '48'})
        for camera, shift in [(top, 0), ('/left-wrist-camera', 2_000_000), ('/right-wrist-camera', 4_000_000)]:
            for i, packet in enumerate(frames):
                stamp = int(capture[i]) + shift
                msg = types['Video'](format='h264', data=packet)
                msg.timestamp.seconds, msg.timestamp.nanos = divmod(stamp, 1_000_000_000)
                if camera == bad_header and i == 2:
                    msg.timestamp.nanos = 1_000_000_000
                writer.write_message(camera, msg, log_time=base + 700_000_000 + i * 61_000_000 + shift,
                                     publish_time=stamp)
        for j, topic in enumerate(abc130k.ARM + abc130k.ARM_ACT):
            if topic in omit:
                continue
            width = 1 if '-ee-' in topic else 6
            for i, stamp in enumerate(capture):
                msg = types['Robot'](position=[0.1 * j + 0.01 * i + k * 0.001 for k in range(width)],
                                     velocity=[0.003 * i + k * 0.001 for k in range(width)])
                msg.timestamp.seconds, msg.timestamp.nanos = divmod(int(stamp), 1_000_000_000)
                if topic == bad_header and i == 2:
                    msg.timestamp.nanos = 1_000_000_000
                if topic == tied and i == 2:
                    msg.timestamp.seconds, msg.timestamp.nanos = divmod(int(capture[i - 1]), 1_000_000_000)
                writer.write_message(topic, msg, log_time=base + 710_000_000 + i * 61_000_000,
                                     publish_time=int(stamp))
        if instruction:
            writer.write_message('/instruction', types['Text'](data='align fabric'), log_time=base, publish_time=base)
    return {'n': n, 'capture_ns': capture.tolist(), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def converted(root, out, monkeypatch, *, legacy):
    original = formats.upload_adapters
    with monkeypatch.context() as patch:
        patch.setattr(formats, 'upload_adapters', lambda kind: [abc130k] if kind == 'mcap' and legacy
                      else [] if kind == 'mcap' else original(kind))
        report = formats.convert(root, 'teleop_arms', out, 'native fixture', 900)
    assert not report['failed'] and len(report['episodes']) == 1, report
    path = out / report['episodes'][0]['episode_id']
    return path, json.loads((path / 'context.json').read_text())


@pytest.mark.parametrize('top', ['/top-camera', '/top-left-camera'])
def test_protobuf_capture_clock_and_split_gripper_state_match_public_converter(tmp_path, monkeypatch, top):
    root = tmp_path / 'native'
    root.mkdir()
    raw = root / 'episode.mcap'
    proof = capture_mcap(raw, top=top)
    old_ep, old = converted(root, tmp_path / 'old', monkeypatch, legacy=True)
    new_ep, new = converted(root, tmp_path / 'generic', monkeypatch, legacy=False)
    assert new['state_kind'] == old['state_kind'] == 'joints'
    assert new['n_state_frames'] == old['n_state_frames'] == proof['n']
    arrays = {}
    with np.load(old_ep / 'state.npz') as a, np.load(new_ep / 'state.npz') as b:
        for key in a.files:
            assert b[key].shape == a[key].shape == (proof['n'], 14), key
            assert a[key].dtype == b[key].dtype and a[key].tobytes() == b[key].tobytes(), key
            arrays[key] = {'dtype': str(b[key].dtype), 'shape': list(b[key].shape),
                           'sha256': hashlib.sha256(b[key].tobytes()).hexdigest()}
    clocks = {}
    with np.load(old_ep / 'times.npz') as a, np.load(new_ep / 'times.npz') as b:
        for key in a.files:
            assert key in b.files and a[key].dtype == b[key].dtype
            assert np.array_equal(a[key], b[key]), key
            clocks[key] = {'dtype': str(b[key].dtype), 'values': b[key].tolist()}
    assert new['instruction'] == old['instruction'] == 'align fabric'
    for topic, facts in old['stream_checks']['streams'].items():
        assert {key: new['stream_checks']['streams'][topic][key] for key in facts} == facts
    for key, value in [('operator_id', 'fixture operator'), ('station_id', 'fixture station')]:
        assert value in json.dumps(new), key
    before = episode.build_request(old_ep)
    after = episode.build_request(new_ep)
    assert before['plan']['ks'] == after['plan']['ks']
    images = lambda req: [p for p in req['content'] if p['type'] == 'image_url']
    assert images(before) and len(images(before)) == len(images(after))
    decoded_hashes = {}
    for view in ['exo', 'left', 'right']:
        pixels = []
        for ep in [old_ep, new_ep]:
            with np.load(ep / 'times.npz') as stamps:
                source = json.loads((ep / 'sources.json').read_text())[view]
                frames = extract_frames(source['packed'], 0, proof['n'], after['plan']['ks'], pts=stamps[view + '_pts'])
                pixels.append({k: image.tobytes() for k, image in frames.items()})
        assert pixels[0] == pixels[1]
        decoded_hashes[view] = {k: hashlib.sha256(value).hexdigest() for k, value in pixels[1].items()}
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == proof['source_sha256']
    print(json.dumps({'source_sha256': proof['source_sha256'], 'capture_ns': proof['capture_ns'],
                      'arrays': arrays, 'times': clocks, 'selected_indices': after['plan']['ks'], 'selected_pixels_sha256': decoded_hashes,
                      'legacy_camera_caption': old['cameras']['exo']['name'],
                      'generic_camera_caption': new['cameras']['exo']['name'],
                      'legacy_gripper_range': old.get('gripper_range'), 'generic_gripper_range': new.get('gripper_range'),
                      'generic_gripper_qualification': new.get('gripper_range_note'),
                      'legacy_robot_type': old.get('robot_type'), 'generic_robot_type': new.get('robot_type'),
                      'differences': ['recorded topic camera caption changes composite image header',
                                      'numeric signal placement uses declared capture clock instead of arrival clock',
                                      'source robot and gripper calibration claims are not inferred',
                                      'raw numeric samples and distinct integer clocks are additionally retained']}))


def test_native_capture_and_arrival_clocks_are_retained_separately(tmp_path, monkeypatch):
    root = tmp_path / 'clock'
    root.mkdir()
    proof = capture_mcap(root / 'episode.mcap')
    ep, ctx = converted(root, tmp_path / 'generic', monkeypatch, legacy=False)
    with np.load(ep / ctx['recorded_mcap_fields']) as fields:
        camera = next(row for row in ctx['mcap_field_inventory'] if row['topic'] == '/top-camera')
        assert fields[camera['capture_ns']].dtype == np.int64
        assert fields[camera['capture_ns']].tolist() == proof['capture_ns']
        assert (fields[camera['log_ns']] - fields[camera['capture_ns']]).tolist() == [
            700_000_000 + i * 28_000_000 for i in range(proof['n'])]
        stream = next(row for row in ctx['mcap_field_inventory'] if row['topic'] == '/left-ee-state')
        assert fields[stream['fields']['position']].shape == (proof['n'], 1)
        assert fields[stream['fields']['velocity']].shape == (proof['n'], 1)
        for j, topic in enumerate(abc130k.ARM + abc130k.ARM_ACT):
            width = 1 if '-ee-' in topic else 6
            row = next(row for row in ctx['mcap_field_inventory'] if row['topic'] == topic)
            expected = np.array([[0.1 * j + 0.01 * i + k * .001 for k in range(width)] for i in range(proof['n'])])
            assert fields[row['fields']['position']].dtype == np.float64
            assert fields[row['fields']['position']].tobytes() == expected.tobytes()
            expected = np.array([[.003 * i + k * .001 for k in range(width)] for i in range(proof['n'])])
            assert fields[row['fields']['velocity']].tobytes() == expected.tobytes()
            assert row['capture_fields'][0] == {'seconds': 1_700_000_000, 'nanos': 0}
    assert {topic + ' velocity' for topic in abc130k.ARM + abc130k.ARM_ACT} <= {s['name'] for s in ctx['signals']}
    assert ctx['recorded_metadata']['mcap_metadata'][0]['metadata']['station_id'] == 'fixture station'


@pytest.mark.parametrize('kwargs', [
    {'omit': ['/right-ee-state']}, {'bad_header': '/left-arm-state'}, {'tied': '/left-ee-state'},
    {'bad_header': '/top-camera'}])
def test_incomplete_or_ambiguous_split_state_retains_every_native_field(tmp_path, monkeypatch, kwargs):
    root = tmp_path / 'fallback'
    root.mkdir()
    capture_mcap(root / 'episode.mcap', **kwargs)
    ep, ctx = converted(root, tmp_path / 'generic', monkeypatch, legacy=False)
    assert ctx['state_kind'] == 'none'
    assert not (ep / 'state.npz').exists()
    assert ctx.get('state_note')
    assert ctx.get('recorded_mcap_fields')
    if kwargs.get('bad_header'):
        row = next(row for row in ctx['mcap_field_inventory'] if row['topic'] == kwargs['bad_header'])
        assert row['capture_fields'][2]['nanos'] == 1_000_000_000
        assert row['capture_errors']


def test_capture_timestamp_validation_never_substitutes_arrival_silently():
    assert formats.mcap_capture_stamp({'timestamp': {'seconds': 10, 'nanos': 23}}) == (10_000_000_023, None)
    assert formats.mcap_capture_stamp({'header': {'stamp': {'sec': 10, 'nanosec': 23}}}) == (10_000_000_023, None)
    assert formats.mcap_capture_stamp({}) == (None, None)
    for stamp in [{'seconds': 10, 'nanos': -1}, {'seconds': 10, 'nanos': 1_000_000_000},
                  {'seconds': 10.5, 'nanos': 0}, {'seconds': True, 'nanos': 0}]:
        assert formats.mcap_capture_stamp({'timestamp': stamp})[1]


def test_absent_action_never_discards_complete_capture_state(tmp_path, monkeypatch):
    root = tmp_path / 'no_action'
    root.mkdir()
    capture_mcap(root / 'episode.mcap', omit=abc130k.ARM_ACT)
    ep, ctx = converted(root, tmp_path / 'generic', monkeypatch, legacy=False)
    assert ctx['state_kind'] == 'joints'
    with np.load(ep / 'state.npz') as arrays:
        assert arrays.files == ['state']
        assert arrays['state'].shape == (6, 14)


def test_split_gripper_gaps_and_competing_sources_are_not_interpolated_as_state():
    q = np.arange(151) / 30
    times = np.array([0, .033, .066, 4.9, 4.95, 5])
    def stream(topic, width, t=q):
        return {'topic': topic, 't': t, 'pos': np.zeros((len(t), width)), 'field': 'position',
                'names': None, 'clocks': {'capture'}, 'clock_errors': []}
    arm, grip = '/left-arm-state', '/left-ee-state'
    streams = {arm: stream(arm, 6), grip: stream(grip, 1, times)}
    composed = formats.compose_mcap_arms(streams, q, 'capture')
    assert formats.joint_state(composed, q, clock='capture')[0] is None
    streams[grip] = stream(grip, 1)
    streams['/left-gripper-state'] = stream('/left-gripper-state', 1)
    assert formats.joint_state(formats.compose_mcap_arms(streams, q, 'capture'), q, clock='capture')[0] is None


def test_metadata_goal_and_owned_same_clock_sensor_samples_remain_complete(tmp_path, monkeypatch):
    from test_formats import _json_mcap
    root = tmp_path / 'owned'
    root.mkdir()
    capture_mcap(root / 'episode.mcap', instruction=False)
    values = np.arange(6, dtype=np.float64)
    _json_mcap(root / 'sensor.mcap', {'/force': [(i * .033, {'force': [float(i), float(i + 1)]}) for i in range(6)]},
               t0=1_700_000_000)
    originals = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
    ep, ctx = converted(root, tmp_path / 'generic', monkeypatch, legacy=False)
    assert ctx['instruction'] == 'align fabric'
    signal = next(s for s in ctx['signals'] if s['name'] == '/force force')
    with np.load(ep / 'signals.npz') as arrays:
        assert arrays[signal['key']].tobytes() == np.stack([values, values + 1], axis=1).astype(np.float32).tobytes()
    retained = ctx['recorded_sensor_fields'][0]
    row = next(row for row in retained['mcap_field_inventory'] if row['topic'] == '/force')
    with np.load(ep / retained['recorded_mcap_fields']) as fields:
        assert fields[row['fields']['force']].tobytes() == np.stack([values, values + 1], axis=1).tobytes()
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()} == originals


def test_retired_upload_discovery_does_not_import_the_published_adapter(monkeypatch):
    import importlib
    load = importlib.import_module
    def declared_only(name, *args, **kwargs):
        if name == 'prepare.abc130k':
            pytest.fail('retired adapter imported on the generic path')
        return load(name, *args, **kwargs)
    monkeypatch.setattr(importlib, 'import_module', declared_only)
    assert [module.__name__ for module in formats.upload_adapters('mcap')] == ['prepare.genhumanego', 'prepare.realomin']
