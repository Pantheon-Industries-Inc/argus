"""Typed recorded claims and structural camera authority without dataset aliases."""
import math
import re
from pathlib import PurePosixPath
from collections.abc import Mapping

UPLOAD = None


def snapshot(value):
    """Copy protobuf defaults too, so an explicit source false remains false."""
    if hasattr(value, 'DESCRIPTOR'):
        return {field.name: (None if field.message_type is not None and not field.is_repeated
                            and not value.HasField(field.name) else snapshot(getattr(value, field.name)))
                for field in value.DESCRIPTOR.fields}
    if isinstance(value, Mapping):
        return {str(key): snapshot(item) for key, item in value.items()}
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, bytes):
        return {'bytes_hex': value.hex()}
    try:
        return [snapshot(item) for item in value]
    except TypeError:
        return str(value)


CLAIM_KEYS = {'bold_mark', 'segments_info', 'invalid_ranges', 'is_valid',
              'distortion_model', 'intrinsics', 'K', 'T_b_c'}


def is_claim(fields):
    descriptor = getattr(fields, 'DESCRIPTOR', None)
    keys = {field.name for field in descriptor.fields} if descriptor else set(fields) if isinstance(fields, Mapping) else set()
    return bool(keys & CLAIM_KEYS)


def inspect(path, camera_topics):
    """Read only typed noncamera claims and container metadata using canonical decoders."""
    from mcap.reader import make_reader
    from . import formats
    metadata, records, decoders = [], [], {}
    with open(path, 'rb') as stream:
        try:
            metadata = [{'name': item.name, 'metadata': dict(item.metadata)}
                        for item in make_reader(stream).iter_metadata()]
        except Exception:
            pass
    topics = [topic for topic, _ in formats.mcap_channels(path) if topic not in camera_topics]
    factories = formats._decoders()
    with open(path, 'rb') as stream:
        for schema, channel, message in formats.mcap_messages(stream, path, topics):
            if channel.id not in decoders:
                decoders[channel.id] = formats._decoder_for(channel.message_encoding, schema, factories)
            decoder = decoders[channel.id]
            if decoder is None:
                continue
            try:
                decoded = decoder(message.data)
                if not is_claim(decoded):
                    continue
                fields = snapshot(decoded)
            except Exception:
                continue
            if is_claim(fields):
                records.append({'topic': channel.topic, 'schema': schema.name if schema else None,
                                'log_ns': int(message.log_time), 'publish_ns': int(message.publish_time),
                                'fields': fields})
    return records, metadata


def finite(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def valid_calibration(fields):
    """Dimensions and finite recorded intrinsics establish association, never view direction."""
    if not all(finite(fields.get(key)) and fields[key] > 0 and int(fields[key]) == fields[key] for key in ('width', 'height')):
        return False
    for key, minimum in (('K', 9), ('D', 4)):
        values = fields.get(key)
        if isinstance(values, list) and len(values) >= minimum and all(finite(value) for value in values):
            if key == 'K' and values[0] > 0 and values[4] > 0:
                return True
            if key == 'D' and fields.get('distortion_model') in ('double_sphere', 'ds') and len(values) == 6 and values[0] > 0 and values[1] > 0:
                return True
    return False


def select_primary(records, metadata, camera_topics):
    declared = {entry['metadata'][key] for entry in metadata
                for key in ('primary_camera', 'primary_camera_topic', 'anchor_camera')
                if entry.get('metadata', {}).get(key)}
    if declared:
        if len(declared) == 1 and next(iter(declared)) in camera_topics:
            return next(iter(declared)), 'The recording declares this exact topic as its primary camera.'
        return None, 'The recorded primary camera declarations conflict or do not name a colour camera topic. The existing deterministic selection policy applies.'
    calibrated = {topic for topic in camera_topics for record in records
                  if PurePosixPath(topic).parent == PurePosixPath(record['topic']).parent
                  and valid_calibration(record['fields'])}
    if len(calibrated) == 1:
        return next(iter(calibrated)), 'This is the only colour camera topic with associated valid recorded calibration. Its physical view direction is not inferred.'
    pattern = re.compile(r'^(.*)/camera([0-5])(/.*)$')
    matches = [pattern.fullmatch(topic) for topic in camera_topics]
    if len(matches) == 6 and all(matches) and {match[2] for match in matches} == set('012345') and len({(match[1], match[3]) for match in matches}) == 1:
        topic = next(topic for topic, match in zip(camera_topics, matches) if match[2] == '2')
        return topic, 'The six numbered colour camera channels use camera2 as a compatibility selection policy. The index does not establish a physical view direction.'
    return None, 'No unique recorded primary camera authority is available. The existing deterministic selection policy applies.'


def apply_claims(ctx, records):
    ctx.setdefault('recorded_metadata', {})['mcap_records'] = records
    annotations = [record for record in records if 'segments_info' in record['fields']]
    goals = {record['fields'].get('bold_mark', '').strip() for record in annotations
             if isinstance(record['fields'].get('bold_mark'), str) and record['fields']['bold_mark'].strip()}
    if not ctx.get('instruction') and len(goals) == 1:
        ctx.update(instruction=goals.pop(), instruction_note='The recorded structured annotation supplies this task claim.')
    labels = {record['fields']['sst'].strip() for record in annotations
              if isinstance(record['fields'].get('sst'), str) and record['fields']['sst'].strip()}
    placeholder = [(ctx.get('source') or {}).get('file')]
    if len(labels) == 1 and (not ctx.get('task_label') or ctx['task_label'] == placeholder):
        ctx['task_label'] = [labels.pop()]
    tasks = []
    for record in annotations:
        segments = record['fields'].get('segments_info')
        for segment in segments if isinstance(segments, list) else []:
            if not isinstance(segment, dict):
                continue
            subs = segment.get('sub_segments_info')
            for sub in subs if isinstance(subs, list) else []:
                if not isinstance(sub, dict):
                    continue
                start, end, label = sub.get('start_time_s'), sub.get('end_time_s'), sub.get('fine_label')
                if finite(start) and finite(end) and 0 <= start <= end and isinstance(label, str) and label.strip():
                    task = {'t0': start, 't1': end, 'label': label}
                    if isinstance(sub.get('is_success'), bool):
                        task['ok'] = sub['is_success']
                    tasks.append(task)
    if tasks and not ctx.get('annotation_subtasks'):
        ctx['annotation_subtasks'] = tasks
