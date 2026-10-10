"""Join recorded Cartesian pose and scalar gripper channels without dataset dispatch."""
from __future__ import annotations

import json
import re

import numpy as np


def camera_roles(path, topics):
    """Explicit actor metadata can assign camera sides while keeping native topics."""
    from mcap.reader import make_reader
    mappings = []
    with open(path, 'rb') as stream:
        try:
            metadata = list(make_reader(stream).iter_metadata())
        except Exception:
            metadata = []
    for record in metadata:
        raw = record.metadata.get('actor_map')
        if raw:
            try:
                parsed = json.loads(raw)
            except (TypeError, ValueError):
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


def pose_state(paths, q, clock):
    """Return state, consumed fields, identity metadata and a qualified limitation."""
    from prepare import formats as f
    poses, grips, failures = {}, {}, []
    for path in paths:
        channels = f.mcap_channels(path)
        wanted = [topic for topic, _ in channels if re.search(
            r'/(?:eef_pose|ee_pose|tcp_pose|magnetic_encoder|gripper_opening)$', topic)]
        if not wanted:
            continue
        decoders, factories = {}, f._decoders()
        with open(path, 'rb') as stream:
            for schema, channel, message in f.mcap_messages(stream, path, wanted):
                if channel.id not in decoders:
                    decoders[channel.id] = f._decoder_for(channel.message_encoding, schema, factories)
                decoder = decoders[channel.id]
                try:
                    decoded = decoder(message.data) if decoder else None
                except Exception:
                    decoded = None
                if decoded is None:
                    failures.append(channel.topic + ' cannot be decoded')
                    continue
                stamp, error = f.mcap_capture_stamp(decoded)
                kind = 'capture' if stamp is not None else 'arrival'
                if error or kind != clock:
                    failures.append(channel.topic + ' has invalid or incompatible pose timing')
                    continue
                stamp = stamp if stamp is not None else message.log_time
                # Actor namespace is structural ownership, not a claim of left or right.
                actor = channel.topic.split('/')[1]
                pose = f._field(decoded, 'pose')
                if pose is not None:
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
             'pose_unit_note': 'Position and gripper values retain their recorded scale; physical units and full opening range are not inferred.'}
    contracts = [camera_roles(path, [topic for _, topic in poses])[1] for path in paths]
    mappings = [contract['actor_map'] for contract in contracts if contract.get('actor_map')]
    if mappings and all(mapping == mappings[0] for mapping in mappings) \
            and all(actor in mappings[0] for actor in actors):
        extra['state_actor_contract'] = {'actors': [mappings[0][actor] for actor in actors],
                                         'source': 'MCAP actor_map metadata'}
    f.record_state_groups(extra, groups)
    return np.concatenate(matrices, axis=1).astype(np.float32), used, extra, None
