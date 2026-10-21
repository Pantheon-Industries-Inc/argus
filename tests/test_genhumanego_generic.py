"""Six frame protobuf source claims and original camera parity without a model."""
import hashlib
import json
from unittest.mock import patch

import av
import numpy as np
import pytest
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from mcap_protobuf.writer import Writer

from prepare import formats, genhumanego
from test_prepare import _h264_frames


# The field types Gen-HumanEgo's own recordings declare (read from its published MCAPs on 2026-10-21): segment ids
# and invalid codes are text. INTEGER_IDS is the integer variant a recorder could write for the same fields.
TEXT_IDS = {'id': 9, 'code': 9}
INTEGER_IDS = {'id': 5, 'code': 5}


def native_types(ids=TEXT_IDS):
    fd = descriptor_pb2.FileDescriptorProto(name='structured_ego_parity.proto', package='structured_ego', syntax='proto3')
    def message(name, fields):
        msg = fd.message_type.add(name=name)
        for i, (field, kind, repeated, ref) in enumerate(fields, 1):
            item = msg.field.add(name=field, number=i, type=kind, label=3 if repeated else 1)
            if ref:
                item.type_name = '.structured_ego.' + ref
    message('CompressedVideo', [('format', 9, False, None), ('data', 12, False, None)])
    message('CameraInfo', [('width', 5, False, None), ('height', 5, False, None),
        ('distortion_model', 9, False, None), ('D', 1, True, None), ('T_b_c', 1, True, None), ('frame_id', 9, False, None)])
    message('Subsegment', [('segment_id', ids['id'], False, None), ('start_time_s', 1, False, None),
        ('end_time_s', 1, False, None), ('fine_label', 9, False, None), ('fine_label_detail', 9, False, None), ('is_success', 8, False, None)])
    message('Segment', [('segment_id', ids['id'], False, None), ('start_time_s', 1, False, None),
        ('end_time_s', 1, False, None), ('fine_label', 9, False, None), ('sub_segments_info', 11, True, 'Subsegment')])
    message('Annotation', [('video_id', 9, False, None), ('bold_mark', 9, False, None), ('sst', 9, False, None),
        ('environment_description', 9, False, None), ('segments_info', 11, True, 'Segment')])
    message('InvalidRange', [('start_time_s', 1, False, None), ('end_time_s', 1, False, None),
        ('invalid_code', ids['code'], False, None), ('invalid_message', 9, False, None)])
    message('TimeRangeValidity', [('invalid_ranges', 11, True, 'InvalidRange')])
    message('FrameValidity', [('is_valid', 8, False, None)])
    message('TextClaim', [('data', 9, False, None)])
    pool = descriptor_pool.DescriptorPool()
    pool.Add(fd)
    return {name: message_factory.GetMessageClass(pool.FindMessageTypeByName('structured_ego.' + name))
            for name in ['CompressedVideo', 'CameraInfo', 'Annotation', 'TimeRangeValidity', 'FrameValidity', 'TextClaim']}


def six_packets(offset):
    original = np.full
    def pixel(shape, value, *args, **kwargs):
        return original(shape, value + offset if shape == (48, 64, 3) else value, *args, **kwargs)
    with patch('test_prepare.np.full', side_effect=pixel):
        return _h264_frames(6)


def recording(path, *, other_camera=False, explicit_primary=False, both_calibrated=False, six_cameras=False, early_other_ns=0, other_topic='/robot0/sensor/camera0/compressed', step_text=False, ids=TEXT_IDS):
    types = native_types(ids)
    as_id = str if ids['id'] == 9 else int
    as_code = str if ids['code'] == 9 else int
    base = 1_700_000_000_000_000_000
    offsets = np.asarray([0, 33, 68, 101, 143, 179], dtype=np.int64) * 1_000_000
    stamps = base + offsets
    topics = [genhumanego.CAMERA_TOPIC]
    if six_cameras:
        topics = [f'/robot0/sensor/camera{i}/compressed' for i in range(6)]
    elif other_camera:
        topics.append(other_topic)
    with Writer(str(path)) as writer:
        if explicit_primary:
            writer._writer.add_metadata('recording', {'primary_camera': genhumanego.CAMERA_TOPIC})
        for topic in topics:
            packets = six_packets(40 if topic == genhumanego.CAMERA_TOPIC else 0)
            topic_stamps = stamps - early_other_ns if topic != genhumanego.CAMERA_TOPIC else stamps
            for i, stamp in enumerate(topic_stamps):
                writer.write_message(topic, types['CompressedVideo'](format='h264', data=packets[i]),
                                     log_time=int(stamp), publish_time=int(stamp) - 1_000_000)
        calibrated = topics if both_calibrated else [genhumanego.CAMERA_TOPIC]
        for topic in calibrated:
            calibration = types['CameraInfo'](width=64, height=48, distortion_model='double_sphere',
                D=[32, 32, 32, 24, .2, .6], T_b_c=np.eye(4).ravel().tolist(), frame_id=topic.rsplit('/', 1)[0])
            writer.write_message(topic.rsplit('/', 1)[0] + '/camera_info', calibration, log_time=base)
        annotation = types['Annotation'](video_id='source episode', bold_mark='fold the cloth', sst='folding',
                                       environment_description='original workspace')
        segment = annotation.segments_info.add(segment_id=as_id(4), start_time_s=0, end_time_s=.179, fine_label='fold cloth')
        segment.sub_segments_info.add(segment_id=as_id(5), start_time_s=.033, end_time_s=.068,
            fine_label='grasp corner', fine_label_detail='original nested detail', is_success=False)
        segment.sub_segments_info.add(segment_id=as_id(6), start_time_s=.101, end_time_s=.179,
            fine_label='fold once', fine_label_detail='original successful detail', is_success=True)
        writer.write_message(genhumanego.ANNOTATION_TOPIC, annotation, log_time=base)
        if step_text:
            writer.write_message('/task/subtask', types['TextClaim'](data='recorded final step'), log_time=base + 33_000_000)
        validity = types['TimeRangeValidity']()
        validity.invalid_ranges.add(start_time_s=.04, end_time_s=.06, invalid_code=as_code(3), invalid_message='source validity claim')
        writer.write_message('/robot0/time_range_validity', validity, log_time=base)
        for i, stamp in enumerate(stamps):
            writer.write_message('/robot0/frame_validity', types['FrameValidity'](is_valid=i not in (2, 4)), log_time=int(stamp))
    return {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'offsets_ns': offsets.tolist(), 'topics': topics}


def converted(root, out):
    """The upload folder root read as Data Review reads it, by the generic reader."""
    report = formats.convert(root, 'ego_head', out, 'native structured fixture', 900)
    assert not report['failed'] and len(report['episodes']) == 1, report
    ep = out / report['episodes'][0]['episode_id']
    return ep, json.loads((ep / 'context.json').read_text())


def published(root, out):
    """The episode the published Gen-HumanEgo preparation (python -m prepare genhumanego) writes for the one
    recording in root, extracted beside it as prepare_one extracts a download."""
    raw = root / 'episode.mcap'
    ep = out / 'episode'
    genhumanego.extract(raw, ep / 'source', raw.name)
    genhumanego.write_sidecar(ep / 'source', ep)
    return ep, json.loads((ep / 'context.json').read_text())


def rgb(path):
    with av.open(str(path)) as container:
        container.streams.video[0].codec_context.thread_count = 1
        return [hashlib.sha256(frame.to_ndarray(format='rgb24').tobytes()).hexdigest() for frame in container.decode(video=0)]


@pytest.mark.parametrize('ids', [TEXT_IDS, INTEGER_IDS], ids=['text ids, as published', 'integer ids'])
def test_structured_source_goal_nested_steps_and_false_success_match_legacy(tmp_path, ids):
    root = tmp_path / 'native'
    root.mkdir()
    raw = root / 'episode.mcap'
    proof = recording(raw, ids=ids)
    old_ep, old = published(root, tmp_path / 'old')
    new_ep, new = converted(root, tmp_path / 'generic')
    # the annotation and validity channels are read whichever type the recorder gave their ids and codes
    meta = json.loads((old_ep / 'source/meta.json').read_text())
    assert not meta['source_failures'], meta['source_failures']
    assert [r['code'] for r in meta['invalid_ranges']] == [ids is TEXT_IDS and '3' or 3]
    assert new['instruction'] == old['instruction'] == 'fold the cloth'
    assert new['task_label'] == old['task_label'] == ['folding']
    assert new['annotation_subtasks'] == old['annotation_subtasks']
    assert new['annotation_subtasks'][0]['ok'] is False
    assert new['n_state_frames'] == old['n_state_frames'] == 6
    with np.load(old_ep / 'times.npz') as before, np.load(new_ep / 'times.npz') as after:
        assert np.array_equal(before['exo'], after['exo'])
    old_source = json.loads((old_ep / 'sources.json').read_text())['exo']
    new_source = json.loads((new_ep / 'sources.json').read_text())['exo']
    assert rgb(old_source['packed']) == rgb(new_source['packed'])
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == proof['sha256']


def test_structured_calibration_validity_and_nested_original_fields_are_retained(tmp_path):
    root = tmp_path / 'native'
    root.mkdir()
    raw = root / 'episode.mcap'
    proof = recording(raw)
    old_ep, _ = published(root, tmp_path / 'old')
    _, new = converted(root, tmp_path / 'generic')
    meta = json.loads((old_ep / 'source/meta.json').read_text())
    records = {r['topic']: r['fields'] for r in new['recorded_metadata']['mcap_records']}
    calibration = records[genhumanego.CAMERA_TOPIC.rsplit('/', 1)[0] + '/camera_info']
    assert calibration['D'] == meta['calib2']['D'] and calibration['T_b_c'] == meta['calib2']['T']
    assert calibration['distortion_model'] == meta['calib2']['model']
    assert (calibration['width'], calibration['height'], calibration['frame_id']) == (
        meta['calib2']['w'], meta['calib2']['h'], meta['calib2']['frame'])
    ranges = records['/robot0/time_range_validity']['invalid_ranges']
    assert ranges[0]['start_time_s'] == meta['invalid_ranges'][0]['t0']
    assert ranges[0]['invalid_message'] == meta['invalid_ranges'][0]['msg']
    annotation = records[genhumanego.ANNOTATION_TOPIC]
    assert annotation['segments_info'][0]['sub_segments_info'][0]['fine_label_detail'] == 'original nested detail'
    assert annotation['segments_info'][0]['sub_segments_info'][0]['is_success'] is False
    frame_claims = [r for r in new['recorded_metadata']['mcap_records'] if r['topic'] == '/robot0/frame_validity']
    assert sum(r['fields']['is_valid'] is False for r in frame_claims) == meta['fv_invalid'] == 2
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == proof['sha256']


def test_recorded_primary_metadata_keeps_original_camera_images_and_other_views(tmp_path):
    root = tmp_path / 'native'
    root.mkdir()
    raw = root / 'episode.mcap'
    proof = recording(raw, other_camera=True, explicit_primary=True, both_calibrated=True)
    old_ep, old = published(root, tmp_path / 'old')
    new_ep, new = converted(root, tmp_path / 'generic')
    assert new['cameras']['exo']['key'] == genhumanego.CAMERA_TOPIC
    assert 'primary' in new['source']['camera_selection'].lower()
    before = json.loads((old_ep / 'sources.json').read_text())['exo']['packed']
    after = json.loads((new_ep / 'sources.json').read_text())['exo']['packed']
    assert rgb(before) == rgb(after)
    assert new['unshown_cameras'][0]['name'] == '/robot0/sensor/camera0/compressed'
    assert len(rgb(new['unshown_cameras'][0]['packed'])) == 6
    assert 'DAS' not in new['cameras']['exo']['desc'] and 'forward' not in new['cameras']['exo']['desc']
    assert old['n_state_frames'] == new['n_state_frames'] == 6
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == proof['sha256']


def test_unique_topic_calibration_selects_original_camera_without_physical_claim(tmp_path):
    root = tmp_path / 'native'
    root.mkdir()
    raw = root / 'episode.mcap'
    recording(raw, other_camera=True)
    _, new = converted(root, tmp_path / 'generic')
    assert new['cameras']['exo']['key'] == genhumanego.CAMERA_TOPIC
    assert 'calibration' in new['source']['camera_selection'].lower()


def test_multiple_calibrated_cameras_without_primary_keep_policy_and_every_stream(tmp_path):
    root = tmp_path / 'native'
    root.mkdir()
    raw = root / 'episode.mcap'
    proof = recording(raw, six_cameras=True, both_calibrated=True)
    old_ep, _ = published(root, tmp_path / 'old')
    new_ep, ctx = converted(root, tmp_path / 'generic')
    assert ctx['cameras']['exo']['key'] == genhumanego.CAMERA_TOPIC
    assert 'compatibility' in ctx['source']['camera_selection']
    assert 'does not establish' in ctx['source']['camera_selection']
    assert len(ctx['unshown_cameras']) == 5
    before = json.loads((old_ep / 'sources.json').read_text())['exo']['packed']
    after = json.loads((new_ep / 'sources.json').read_text())['exo']['packed']
    assert rgb(before) == rgb(after)
    with np.load(old_ep / 'times.npz') as old_times, np.load(new_ep / 'times.npz') as new_times:
        assert np.array_equal(old_times['exo'], new_times['exo'])
    assert all(len(rgb(camera['packed'])) == 6 for camera in ctx['unshown_cameras'])
    inventory = {row['topic']: row for row in ctx['mcap_field_inventory']}
    with np.load(new_ep / ctx['recorded_mcap_fields']) as clocks:
        for topic in proof['topics']:
            stamps = clocks[inventory[topic]['log_ns']]
            assert (stamps - stamps[0]).tolist() == proof['offsets_ns']
            assert (stamps - clocks[inventory[topic]['publish_ns']]).tolist() == [1_000_000] * 6
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == proof['sha256']


def test_claim_policy_keeps_priority_false_unknown_fields_and_invalid_times():
    from prepare.mcap_claims import apply_claims, snapshot, select_primary, valid_calibration
    types = native_types()
    annotation = types['Annotation'](bold_mark='lower priority')
    segment = annotation.segments_info.add()
    segment.sub_segments_info.add(start_time_s=.02, end_time_s=.05, fine_label='source claim', is_success=False,
                                 fine_label_detail='original detail')
    segment.sub_segments_info.add(start_time_s=-1, end_time_s=2, fine_label='invalid range')
    records = [{'topic': '/recording/annotation', 'fields': snapshot(annotation)}]
    ctx = {'instruction': 'direct original instruction', 'unknown': {'original': True}}
    apply_claims(ctx, records)
    assert ctx['instruction'] == 'direct original instruction'
    assert ctx['annotation_subtasks'] == [{'t0': .02, 't1': .05, 'label': 'source claim', 'ok': False}]
    assert ctx['recorded_metadata']['mcap_records'] == records
    assert ctx['unknown'] == {'original': True}
    assert not valid_calibration({'width': 64, 'height': 48, 'K': [float('nan')] * 9})
    calib = {'width': 64, 'height': 48, 'K': [32, 0, 32, 0, 32, 24, 0, 0, 1]}
    cameras = ['/camera_a/video', '/camera_b/video']
    assert select_primary([{'topic': '/different/camera_info', 'fields': calib}], [], cameras)[0] is None
    assert select_primary([], [{'metadata': {'primary_camera': '/missing/video'}}], cameras)[0] is None
    assert select_primary([], [{'metadata': {'primary_camera': cameras[1]}}], cameras)[0] == cameras[1]


def test_camera_authority_does_not_promote_arbitrary_indices_or_invalid_calibration():
    from prepare.mcap_claims import select_primary
    six = [f'/actor/sensor/camera{i}/compressed' for i in range(6)]
    assert select_primary([], [], six)[0] == six[2]
    assert select_primary([], [], six[:5])[0] is None
    assert select_primary([], [], six[:-1] + ['/different/camera5/compressed'])[0] is None
    assert select_primary([], [], six[:-1] + ['/actor/sensor/camera5/other'])[0] is None
    assert select_primary([], [{'metadata': {'primary_camera': six[1]}}], six)[0] == six[1]
    assert select_primary([], [{'metadata': {'primary_camera': six[1]}},
                                {'metadata': {'primary_camera': six[2]}}], six)[0] is None
    invalid = {'width': 64, 'height': 48, 'K': [float('inf')] * 9}
    assert select_primary([{'topic': '/actor/sensor/camera1/info', 'fields': invalid}], [], six)[0] == six[2]


def test_typed_unknown_bytes_and_unusable_large_time_remain_recorded():
    from prepare.mcap_claims import snapshot, finite
    assert snapshot({'unknown': b'\x00\xff', 'nested': {'decision': False}}) == {
        'unknown': {'bytes_hex': '00ff'}, 'nested': {'decision': False}}
    assert finite(10 ** 1000) is False


def test_native_inspection_uses_typed_source_fields_and_original_integer_message_clocks(tmp_path):
    from prepare.mcap_claims import inspect, select_primary
    raw = tmp_path / 'episode.mcap'
    proof = recording(raw, six_cameras=True, both_calibrated=True, explicit_primary=True)
    records, metadata = inspect(raw, proof['topics'])
    assert select_primary(records, metadata, proof['topics'])[0] == genhumanego.CAMERA_TOPIC
    annotation = next(row for row in records if row['topic'] == genhumanego.ANNOTATION_TOPIC)
    assert annotation['fields']['bold_mark'] == 'fold the cloth'
    assert annotation['fields']['segments_info'][0]['sub_segments_info'][0]['is_success'] is False
    assert annotation['log_ns'] == 1_700_000_000_000_000_000
    validity = [row for row in records if row['topic'] == '/robot0/frame_validity']
    assert [row['log_ns'] - validity[0]['log_ns'] for row in validity] == proof['offsets_ns']
    assert sum(not row['fields']['is_valid'] for row in validity) == 2
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == proof['sha256']


def test_one_message_that_cannot_be_copied_costs_that_message_only(tmp_path, monkeypatch):
    from prepare import mcap_claims
    raw = tmp_path / 'episode.mcap'
    proof = recording(raw, six_cameras=True, both_calibrated=True, explicit_primary=True)
    whole, _ = mcap_claims.inspect(raw, proof['topics'])
    copy = mcap_claims.snapshot

    def deep(value):
        if getattr(value, 'bold_mark', None):                         # the annotation message, nested too deep to copy
            raise RecursionError('maximum recursion depth exceeded')
        return copy(value)
    monkeypatch.setattr(mcap_claims, 'snapshot', deep)
    records, metadata = mcap_claims.inspect(raw, proof['topics'])
    assert [r['topic'] for r in records] == [r['topic'] for r in whole if r['topic'] != genhumanego.ANNOTATION_TOPIC]
    assert mcap_claims.select_primary(records, metadata, proof['topics'])[0] == genhumanego.CAMERA_TOPIC


def test_unshown_frames_before_recorded_primary_are_preserved(tmp_path):
    root = tmp_path / 'native'
    root.mkdir()
    raw = root / 'episode.mcap'
    proof = recording(raw, other_camera=True, explicit_primary=True, early_other_ns=100_000_000,
                      other_topic='/robot0/sensor/thermal0/compressed')
    ep, ctx = converted(root, tmp_path / 'generic')
    camera = next(row for row in ctx['unshown_cameras'] if row['name'] == '/robot0/sensor/thermal0/compressed')
    assert len(rgb(camera['packed'])) == camera['n_frames'] == 6
    assert camera['start_s'] == -.1
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory
    from prepare.remux import remux
    with raw.open('rb') as stream:
        messages = list(make_reader(stream, decoder_factories=[DecoderFactory()]).iter_decoded_messages(
            topics=['/robot0/sensor/thermal0/compressed']))
    stamps = np.asarray([message.log_time for _, _, message, _ in messages], dtype=np.int64)
    original_packets = [bytes(decoded.data) for _, _, _, decoded in messages]
    expected = tmp_path / 'original_secondary.mp4'
    remux(original_packets, (stamps - stamps[0]) / 1e9, 'h264', expected)
    assert rgb(expected) == rgb(camera['packed'])
    with np.load(ep / camera['camera_times']) as times:
        assert times['capture_ns'].dtype == np.dtype('int64')
        assert np.array_equal(times['capture_ns'], stamps)
        expected_times = (stamps - 1_700_000_000_000_000_000) / 1e9
        assert np.array_equal(times['capture'], expected_times)
        assert np.array_equal(times['presentation'], expected_times)
        # These are exact storage addresses; native message clocks above own presentation timing.
        np.testing.assert_array_equal(times['pts'], formats.probe(camera['packed'])['pts'])
        assert np.all(np.diff(times['pts']) > 0)
    old_ep, _ = published(root, tmp_path / 'old')
    with np.load(old_ep / 'times.npz') as original_times, np.load(ep / 'times.npz') as times:
        assert original_times['exo'].dtype == times['exo'].dtype
        assert original_times['exo'].tobytes() == times['exo'].tobytes()
        main_source = json.loads((ep / 'sources.json').read_text())['exo']
        np.testing.assert_array_equal(times['exo_pts'], formats.probe(main_source['packed'])['pts'])
        assert np.all(np.diff(times['exo_pts']) > 0)
    assert 'camera_clock' not in camera
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == proof['sha256']


def test_upload_hand_pose_spec_has_unknown_intrinsics_with_original_claims_retained(tmp_path):
    from board.hand_pose.specs import spec
    root = tmp_path / 'native'
    root.mkdir()
    recording(root / 'episode.mcap')
    new_ep, _ = converted(root, tmp_path / 'generic')
    assert spec(new_ep)['camera'] == {'model': 'unknown_pinhole'}


def test_calibration_dimensions_must_describe_integral_image_size():
    from prepare.mcap_claims import valid_calibration
    fields = {'width': 64.5, 'height': 48, 'K': [32, 0, 32, 0, 32, 24, 0, 0, 1]}
    assert valid_calibration(fields) is False


def test_recorded_double_sphere_model_short_name_is_structural_calibration():
    from prepare.mcap_claims import valid_calibration
    fields = {'width': 64, 'height': 48, 'distortion_model': 'ds', 'D': [32, 32, 32, 24, .2, .6]}
    assert valid_calibration(fields) is True


def test_malformed_nested_claims_stay_recorded_without_failing_usable_recording():
    from prepare.mcap_claims import apply_claims
    records = [{'topic': '/annotation', 'fields': {'segments_info': [None, 'bad',
               {'sub_segments_info': [None, 'bad']}, {'sub_segments_info': None}]}},
               {'topic': '/other_annotation', 'fields': {'segments_info': None}}]
    ctx = {'unknown': {'original': True}}
    apply_claims(ctx, records)
    assert ctx['recorded_metadata']['mcap_records'] == records
    assert 'annotation_subtasks' not in ctx
    assert ctx['unknown'] == {'original': True}


def test_early_unshown_camera_does_not_extend_selected_step_claim_end(tmp_path):
    from mcap.reader import make_reader
    controls = []
    source_ends = []
    for name, options in [('selected_only', {}), ('early_secondary', {
            'other_camera': True, 'early_other_ns': 100_000_000,
            'other_topic': '/robot0/sensor/thermal0/compressed'})]:
        root = tmp_path / name
        root.mkdir()
        raw = root / 'episode.mcap'
        recording(raw, step_text=True, **options)
        with raw.open('rb') as stream:
            selected = np.asarray([message.log_time for _, _, message in make_reader(stream).iter_messages(
                topics=[genhumanego.CAMERA_TOPIC])], dtype=np.int64)
        source_ends.append(round((selected[-1] - selected[0] + np.median(np.diff(selected))) / 1e9, 3))
        ep, ctx = converted(root, tmp_path / (name + '_prepared'))
        with np.load(ep / ctx['recorded_camera_clocks']) as native:
            np.testing.assert_array_equal(native['exo_log_ns'], selected)
        assert ctx['source']['presentation_clock']['basis'] == 'mcap_log_time'
        assert ctx['source']['presentation_clock']['physical_synchronization_verified'] is False
        controls.append(ctx['annotation_subtasks'])
    assert source_ends == [.214, .214]
    assert controls[0] == [{'t0': .033, 't1': source_ends[0], 'label': 'recorded final step'}]
    assert controls[1] == controls[0]


def test_recorded_task_class_replaces_only_filename_placeholder():
    from prepare.mcap_claims import apply_claims
    records = [{'topic': '/annotation', 'fields': {'segments_info': [], 'sst': 'source class'}}]
    placeholder = {'task_label': ['recording'], 'source': {'file': 'recording'}}
    apply_claims(placeholder, records)
    assert placeholder['task_label'] == ['source class']
    explicit = {'task_label': ['direct task class'], 'source': {'file': 'recording'}}
    apply_claims(explicit, records)
    assert explicit['task_label'] == ['direct task class']
    conflicting = {'task_label': ['recording'], 'source': {'file': 'recording'}}
    apply_claims(conflicting, records + [{'topic': '/other', 'fields': {'segments_info': [], 'sst': 'other class'}}])
    assert conflicting['task_label'] == ['recording']
