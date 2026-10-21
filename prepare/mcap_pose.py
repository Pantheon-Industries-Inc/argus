"""Join recorded Cartesian pose and scalar gripper channels without dataset dispatch."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


def camera_roles(path, topics):
    """Explicit actor metadata can assign camera sides while keeping native topics."""
    from mcap.reader import make_reader
    mappings = []
    with open(path, 'rb') as stream:
        metadata = []
        try:
            for record in make_reader(stream).iter_metadata():
                metadata.append(record)
        except Exception:  # noqa: BLE001, S110 - the mcap reader on upload bytes; named by formats.retain_mcap_fields
            pass  # the records before the damage are kept; the same failure is named on the episode there
    for record in metadata:
        raw = record.metadata.get('actor_map')
        if raw:
            try:
                parsed = json.loads(raw)
            except (TypeError, ValueError, RecursionError):  # a map nested past the recursion limit is unparsed too
                return {}, {'limitation': 'Recorded actor map could not be parsed.'}
            if not isinstance(parsed, dict) or any(value not in ('left', 'right') for value in parsed.values()) \
                    or len(set(parsed.values())) != len(parsed):
                return {}, {'limitation': 'Recorded actor map does not establish distinct sides.'}
            mappings.append(parsed)
    if not mappings:
        return {}, {}
    if any(mapping != mappings[0] for mapping in mappings):
        return {}, {'limitation': 'Recorded actor maps conflict; camera sides were not inferred.'}
    mapping = mappings[0]
    aliases = {}
    for topic in topics:
        actor = topic.split('/')[1] if topic.startswith('/') else topic.split('/')[0]
        if actor in mapping:
            aliases[topic] = mapping[actor] + ' ' + topic
    return aliases, {'source': 'MCAP actor_map metadata', 'actor_map': mapping}


def pose_contract(path):
    """Read explicit source units and quaternion order without treating topic names as calibration."""
    from mcap.reader import make_reader
    keys = ("position_unit", "orientation_order")
    claims = {key: set() for key in keys}
    with open(path, 'rb') as stream:
        records = []
        try:
            for record in make_reader(stream).iter_metadata():
                records.append(record)
        except Exception:  # noqa: BLE001, S110 - the mcap reader on upload bytes; named by formats.retain_mcap_fields
            pass  # the records before the damage are kept; the same failure is named on the episode there
    for record in records:
        for key in keys:
            if key in record.metadata:
                claims[key].add(record.metadata[key].strip().lower())
    metres = {'m', 'meter', 'meters', 'metre', 'metres'}
    position = claims['position_unit'] <= metres if claims['position_unit'] else None
    orientation = claims['orientation_order'] == {'xyzw'} if claims['orientation_order'] else None
    return position, orientation


def join_pose_pieces(held, more):
    """Two time pieces of one pose or gripper stream as one, in time order."""
    if np.shape(held['values'])[1:] != np.shape(more['values'])[1:]:
        return None
    t = np.concatenate([held['t'], more['t']])
    order = np.argsort(t, kind='stable')
    return {**held, 't': t[order], 'values': np.concatenate([held['values'], more['values']])[order],
            'frames': set(held['frames']) | set(more['frames'])}


def pose_state(paths, q, clock):
    """Return state, consumed fields, identity metadata and a qualified limitation. Each file is read on its own and
    the files are gathered by formats.merge_streams, the rule the signals and joint streams follow (formats.relate):
    pieces of one recording are joined, and one actor's stream recorded by two files at overlapping times is two
    recordings, so no pose state is read and the limitation says why."""
    from prepare import formats as f
    per_file, failures = [], []
    for path in paths:
        poses, grips = {}, {}
        per_file.append((Path(path).name, poses, grips))
        channels = f.mcap_channels(path)
        wanted = [topic for topic, _ in channels if re.search(
            r'/(?:eef_pose|ee_pose|tcp_pose|magnetic_encoder|gripper_opening)$', topic)]
        if not wanted:
            continue
        position_m, orientation_xyzw = pose_contract(path)
        decoders, factories = {}, f._decoders()
        with open(path, 'rb') as stream:
            for schema, channel, message in f.mcap_messages(stream, path, wanted):
                if channel.id not in decoders:
                    decoders[channel.id] = f._decoder_for(channel.message_encoding, schema, factories)
                decoder = decoders[channel.id]
                try:
                    decoded = decoder(message.data) if decoder else None
                except Exception:  # noqa: BLE001 - a decoder on upload bytes; named in failures just below
                    decoded = None
                if decoded is None:
                    failures.append(channel.topic + ' cannot be decoded')
                    continue
                stamp, error = (f.mcap_capture_stamp(decoded, schema=schema, message_encoding=channel.message_encoding)
                                if clock == 'capture' else (None, None))
                kind = 'capture' if stamp is not None else 'arrival'
                if error or kind != clock:
                    failures.append(channel.topic + ' has invalid or incompatible pose timing')
                    continue
                stamp = stamp if stamp is not None else message.log_time
                # Actor namespace is structural ownership, not a claim of left or right.
                actor = channel.topic.split('/')[1]
                pose = f._field(decoded, 'pose')
                if pose is not None:
                    # ROS geometry poses define metres and named xyzw quaternion components.
                    ros_pose = (schema is not None and schema.encoding in ('ros1msg', 'ros2msg')
                                and schema.name in ('geometry_msgs/PoseStamped', 'geometry_msgs/msg/PoseStamped',
                                                    'geometry_msgs/Pose', 'geometry_msgs/msg/Pose'))
                    proven = ((ros_pose and position_m is not False and orientation_xyzw is not False)
                              or (position_m and orientation_xyzw))
                    if not proven:
                        failures.append(channel.topic + ' lacks a metre position and xyzw orientation contract')
                        continue
                    position, orientation = f._field(pose, 'position'), f._field(pose, 'orientation')
                    row = [f._field(position, name) for name in ('x', 'y', 'z')]
                    row += [f._field(orientation, name) for name in ('x', 'y', 'z', 'w')]
                    target = poses
                else:
                    row, target = [f._field(decoded, 'value')], grips
                try:
                    values = np.asarray(row, dtype=np.float64)
                except (TypeError, ValueError):
                    failures.append(channel.topic + ' has unreadable pose or gripper values')
                    continue
                if not np.isfinite(values).all():
                    failures.append(channel.topic + ' has nonfinite pose or gripper values')
                    continue
                key = (actor, channel.topic)
                group = target.setdefault(key, {'t': [], 'values': [], 'frames': set()})
                group['t'].append(stamp / 1e9)
                group['values'].append(values)
                for frame in [f._field(decoded, 'frame_id'), f._field(f._field(decoded, 'header'), 'frame_id')]:
                    if frame:
                        group['frames'].add(str(frame))
    def gathered(index):
        files = [(file, {key: {**group, 't': np.asarray(group['t']), 'values': np.asarray(group['values'])}
                         for key, group in streams[index].items()}) for file, *streams in per_file]
        return f.merge_streams(files, values='values', combine=join_pose_pieces,
                               rename=lambda key, name: (key[0], name))
    poses, pose_notes = gathered(0)
    grips, grip_notes = gathered(1)
    if pose_notes or grip_notes:
        return None, {}, {}, '; '.join(n['what'] for n in pose_notes + grip_notes) + ', so which one is the pose is unknown'
    if not poses:
        return None, {}, {}, '; '.join(dict.fromkeys(failures)) if failures else None
    actors = sorted({actor for actor, _ in poses} | {actor for actor, _ in grips})
    if failures or len(actors) not in (1, 2):
        return None, {}, {}, '; '.join(dict.fromkeys(failures)) or 'The pose actor count is unsupported.'
    matrices, used, groups = [], {}, []
    identities = []
    for actor in actors:
        p = [(topic, group) for (owner, topic), group in poses.items() if owner == actor]
        g = [(topic, group) for (owner, topic), group in grips.items() if owner == actor]
        if len(p) != 1 or len(g) != 1:
            return None, {}, {}, 'Each recorded pose actor needs exactly one pose and one scalar gripper stream.'
        pose_topic, pose = p[0]
        grip_topic, grip = g[0]
        if len(pose['frames']) > 1:
            return None, {}, {}, 'Recorded pose frame ids change within the stream.'
        pt, gt = np.asarray(pose['t']), np.asarray(grip['t'])
        pv, gv = np.asarray(pose['values']), np.asarray(grip['values'])
        if pv.shape[1:] != (7,) or gv.shape[1:] != (1,) or len(pt) < 2 or len(gt) < 2:
            return None, {}, {}, 'Recorded pose or scalar gripper widths are unsupported.'
        if (np.diff(pt) <= 0).any() or (np.diff(gt) <= 0).any():
            return None, {}, {}, 'Pose and gripper timestamps must be strictly increasing.'
        if not np.allclose(np.linalg.norm(pv[:, 3:], axis=1), 1.0, atol=1e-3, rtol=0):
            return None, {}, {}, 'Recorded orientation is not a finite unit xyzw quaternion.'
        for times, values in [(pt, pv), (gt, gv)]:
            _, gap = f.fill_rows(q, times, values)
            if gap:
                return None, {}, {}, 'Recorded pose or gripper timing does not cover the camera frames.'
        values = pv[f.nearest(pt, q)]
        x, y, z, w = values[:, 3:].T
        rpy = np.column_stack([
            np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)),
            np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))])
        matrices.append(np.concatenate([values[:, :3], rpy, gv[f.nearest(gt, q)]], axis=1))
        used[pose_topic], used[grip_topic] = {'pose.position', 'pose.orientation'}, {''}
        # A coordinate frame names the reference system, not the actor that moved.
        groups.append((pose_topic, None, 7))
        identities.append({'actor': actor, 'pose': pose_topic, 'gripper': grip_topic,
                           'frame_ids': sorted(pose['frames'])})
    extra = {'source': {'state': 'recorded Cartesian pose and scalar gripper channels'},
             'pose_field_bindings': identities,
             'pose_transform': 'Recorded xyzw quaternions converted to roll pitch yaw radians; nearest native pose and gripper sample per camera frame.',
             'pose_unit_note': 'Position units are declared metres; quaternion components are xyzw. Gripper values retain their recorded scale and full opening range is not inferred.',
             'state_contract': None}
    from prepare.state_contract import declared_contract
    names = [actor + '_' + name for actor in actors for name in ('x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper')]
    extra['state_contract'] = declared_contract('prepare.mcap_pose:pose', 'ee_pose', names,
                                                (['m'] * 3 + ['rad'] * 3 + [None]) * len(actors))
    contracts = [camera_roles(path, [topic for _, topic in poses])[1] for path in paths]
    mappings = [contract['actor_map'] for contract in contracts if contract.get('actor_map')]
    if mappings and all(mapping == mappings[0] for mapping in mappings) \
            and all(actor in mappings[0] for actor in actors):
        extra['state_actor_contract'] = {'actors': [mappings[0][actor] for actor in actors],
                                         'source': 'MCAP actor_map metadata'}
    f.record_state_groups(extra, groups)
    return np.concatenate(matrices, axis=1).astype(np.float32), used, extra, None
