"""Original numeric types and bookkeeping survive native field retention."""
import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

from prepare.mcap_fields import native_fields


def recorded_type():
    fd = descriptor_pb2.FileDescriptorProto(name='numeric_leaf_parity.proto', package='numeric_leaf', syntax='proto3')
    header = fd.message_type.add(name='Header')
    header.field.add(name='seq', number=1, type=3, label=1)
    record = fd.message_type.add(name='Record')
    record.field.add(name='count', number=1, type=3, label=1)
    record.field.add(name='gain', number=2, type=2, label=3)
    record.field.add(name='valid', number=3, type=8, label=1)
    record.field.add(name='unsigned', number=4, type=4, label=1)
    record.field.add(name='header', number=5, type=11, type_name='.numeric_leaf.Header', label=1)
    pool = descriptor_pool.DescriptorPool()
    pool.Add(fd)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName('numeric_leaf.Record'))


def test_native_int64_float32_bool_and_bookkeeping_values_are_exact():
    message = recorded_type()(count=2**53 + 7, gain=[0.1, 0.3], valid=True, unsigned=2**63 + 5)
    message.header.seq = 2**53 + 9
    result = native_fields(message)
    assert result['count']['values'].dtype == np.dtype('int64')
    assert result['count']['values'].item() == 2**53 + 7
    assert result['header.seq']['values'].item() == 2**53 + 9
    assert result['unsigned']['values'].dtype == np.dtype('uint64')
    assert result['unsigned']['values'].item() == 2**63 + 5
    assert result['valid']['values'].dtype == np.dtype('bool')
    assert result['valid']['values'].item() is True
    assert result['gain']['values'].dtype == np.dtype('float32')
    assert result['gain']['values'].tobytes() == np.asarray(message.gain, dtype=np.float32).tobytes()


def test_declared_zero_and_empty_fields_keep_type_and_shape():
    result = native_fields(recorded_type()())
    assert result['count']['values'].item() == 0
    assert result['valid']['values'].item() is False
    assert result['gain']['values'].dtype == np.dtype('float32')
    assert result['gain']['shape'] == [0]


def test_json_mixed_numeric_types_do_not_round_large_integer():
    result = native_fields({'header': {'seq': 2**53 + 3}, 'values': [2**53 + 7, 0.25, True], 'text': 'note'})
    assert result['header.seq']['values'].item() == 2**53 + 3
    assert result['values[0]']['values'].item() == 2**53 + 7
    assert result['values[1]']['values'].item() == 0.25
    assert result['values[2]']['values'].item() is True
    assert result['text']['dtype'] == 'string'
    assert result['text']['values'] is None
    assert result['text']['original'] == 'note'
    assert result['text']['dtype_source'] == 'decoded Python type'


def test_protobuf_maps_keep_declared_numeric_and_nested_values():
    fd = descriptor_pb2.FileDescriptorProto(name='numeric_maps.proto', package='numeric_maps', syntax='proto3')
    detail = fd.message_type.add(name='Detail')
    detail.field.add(name='gain', number=1, type=2, label=1)
    detail.field.add(name='valid', number=2, type=8, label=1)
    record = fd.message_type.add(name='Record')
    for number, name, value_type, type_name in (
        (1, 'counters', 3, ''), (2, 'details', 11, '.numeric_maps.Detail'), (3, 'notes', 9, ''),
    ):
        entry = record.nested_type.add(name=name.title() + 'Entry')
        entry.options.map_entry = True
        entry.field.add(name='key', number=1, type=9, label=1)
        value = entry.field.add(name='value', number=2, type=value_type, label=1)
        if type_name:
            value.type_name = type_name
        record.field.add(name=name, number=number, type=11,
                         type_name='.numeric_maps.Record.' + entry.name, label=3)
    pool = descriptor_pool.DescriptorPool()
    pool.Add(fd)
    message = message_factory.GetMessageClass(pool.FindMessageTypeByName('numeric_maps.Record'))()
    message.counters['motor'] = 2**53 + 17
    message.counters['motor"]'] = 0
    message.details['sensor'].gain = 0.1
    message.notes['text'] = 'not numeric'
    result = native_fields(message)
    assert result['counters["motor"]']['values'].item() == 2**53 + 17
    assert result['counters["motor"]']['dtype'] == 'int64'
    assert result['counters["motor"]']['dtype_source'] == 'protobuf declaration'
    assert result['counters["motor\\\"]"]']['values'].item() == 0
    assert result['details["sensor"].gain']['values'].tobytes() == np.float32(message.details['sensor'].gain).tobytes()
    assert result['details["sensor"].valid']['values'].item() is False
    assert result['details["sensor"].valid']['present'] is True
    assert result['notes["text"]']['dtype'] == 'string'
    assert result['notes["text"]']['values'] is None
    assert result['notes["text"]']['original'] == message.notes['text']
    assert result['notes["text"]']['dtype_source'] == 'protobuf declaration'


def test_one_frame_secondary_camera_remains_a_usable_recording(tmp_path, monkeypatch):
    from mcap_protobuf.writer import Writer
    from prepare import formats
    from test_abc_generic import capture_mcap
    root = tmp_path / 'native'
    root.mkdir()
    write = Writer.write_message
    seen = set()
    def one_frame(writer, topic, message, **kwargs):
        if topic == '/left-wrist-camera':
            if topic in seen:
                return
            seen.add(topic)
        return write(writer, topic, message, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Writer, 'write_message', one_frame)
        capture_mcap(root / 'episode.mcap')
    original = formats.upload_adapters
    monkeypatch.setattr(formats, 'upload_adapters', lambda kind: [] if kind == 'mcap' else original(kind))
    report = formats.convert(root, 'teleop_arms', tmp_path / 'generic', 'short secondary camera', 900)
    assert not report['failed'] and len(report['episodes']) == 1, report
