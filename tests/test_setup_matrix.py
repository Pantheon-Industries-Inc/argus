"""Every supported setup through the whole pipeline, offline, with nothing lost on the way.

Round after round of QA found data dropped silently: a camera or signal stream, a whole episode, findings written in
the other findings list, streams that share a name. Each was fixed where it was found. This test catches the class:
a tiny recording of every setup the README lists is generated here, never downloaded, and run through `python -m
review` (prepare, checks, clips, the request build, a reply, binding, the board build) and the JSON Lines export. One
recording is labelled in parts and stitched, and one glove gets the separate regional pass. Every reply lists its
tasks, and each task's outcome and the instruction comparison must also reach the board and the export. The reviews
run side by side in a few spawned workers (review_case). The model is offline
(OfflineModel): each reply, for every part and every pass, cites every camera, signal and inspected evidence id its
request offered, in every list the schema allows.

Each fixture lists what went in (its inventory: recordings, cameras, signals, recorded fields). Every item must come
out at every stage, the prepared episode, the request's evidence, the bound annotations, the board's episode record
and the export, or be named in a reason the pipeline recorded (a reader issue, a left-out note, a limitation, a
set-aside episode). Every citation a reply made must be bound or named unbound with its reason, and everything a part
kept must reach the stitched episode, the board and the export. The checks are the same for every setup: nothing here knows what a
setup is expected to lose, only what its fixture put in."""
from __future__ import annotations

import io
import json
import os
import re
import sys
import traceback
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

# ---- the fixture corpus ----------------------------------------------------------------------------------------------

@dataclass
class Item:
    kind: str           # recording, camera, signal, depth or field
    name: str           # the name the source gives it: a topic, a dataset path, a column or a file stem
    file: str | None = None  # the file it is recorded in, when two items share a name

    def __str__(self):
        return f'{self.kind} {self.name}' + (f' ({self.file})' if self.file else '')


@dataclass
class Setup:
    name: str
    rig: str
    data: Path
    items: list = field(default_factory=list)
    part_s: float | None = None     # a request part this long, so a short recording is labelled in parts
    fault: str | None = None        # the offline model fails the second part's annotation: error, cutoff or rejected
    tasks: int = 2                  # how many tasks the offline model finds in the recording (one, or two)


def teleop_lerobot(root: Path) -> Setup:
    """A LeRobot v2.1 teleop dataset: two cameras, observation.state and action, and an episode the metadata lists
    whose files were never downloaded."""
    from test_prepare import _lerobot_v21
    data = root / 'teleop_lerobot'
    _lerobot_v21(data, n=60)
    return Setup('teleop_lerobot', 'teleop_arms', data, [
        Item('recording', '000000'), Item('recording', '000001'),
        Item('camera', 'observation.images.cam_high'), Item('camera', 'observation.images.cam_left_wrist'),
        Item('signal', 'observation.state'), Item('signal', 'action')])


def mcap_arm(root: Path) -> Setup:
    """One MCAP: two compressed cameras, an arm's joint state and an IMU topic."""
    from test_reader_cameras import _json_mcap, _jpg_msg
    data = root / 'mcap_arm'
    data.mkdir(parents=True)
    imu = lambda k: {'linear_acceleration': {'x': 0.01 * k, 'y': 0.0, 'z': 9.8},
                     'angular_velocity': {'x': 0.0, 'y': 0.02 * (k % 7), 'z': 0.0}}
    _json_mcap(data / 'session.mcap', {'/scene_cam/image/compressed': ('foxglove.CompressedImage', _jpg_msg),
                                       '/wrist_cam/image/compressed': ('foxglove.CompressedImage', _jpg_msg),
                                       '/imu/data': ('sensor_msgs/Imu', imu)}, n=90, joints=True)
    return Setup('mcap_arm', 'teleop_arms', data, [
        Item('recording', 'session'),
        Item('camera', '/scene_cam/image/compressed'), Item('camera', '/wrist_cam/image/compressed'),
        Item('signal', '/right_arm/joint_state'), Item('signal', '/imu/data')])


def umi_gripper(root: Path) -> Setup:
    """A two-gripper UMI recording: each gripper's wrist camera, its tracked pose, its jaw encoder and its IMU."""
    from test_realomin_generic import indexed_recording
    data = root / 'umi_gripper'
    data.mkdir(parents=True)
    indexed_recording(data / 'grippers.mcap')
    return Setup('umi_gripper', 'handheld_gripper', data, [Item('recording', 'grippers'), Item('field', 'actor_map'),
                                                           Item('field', 'orientation_order')] + [
        Item(kind, f'/robot{robot}/{topic}') for robot in (0, 1) for kind, topic in (
            ('camera', 'sensor/camera0/compressed'), ('signal', 'vio/eef_pose'),
            ('signal', 'sensor/magnetic_encoder'), ('signal', 'sensor/imu'))])


def ego_head(root: Path) -> Setup:
    """A human ego recording: a head camera with its calibration, the publisher's annotation and validity claims."""
    from prepare import genhumanego
    from test_genhumanego_generic import recording
    data = root / 'ego_head'
    data.mkdir(parents=True)
    recording(data / 'ego.mcap')
    return Setup('ego_head', 'ego_head', data, [
        Item('recording', 'ego'), Item('camera', genhumanego.CAMERA_TOPIC),
        Item('field', genhumanego.ANNOTATION_TOPIC), Item('field', '/robot0/time_range_validity'),
        Item('field', '/robot0/frame_validity')])


def hdf5_gloves(root: Path) -> Setup:
    """A log split over files: the scene video beside one HDF5 file per glove, each glove's array and clock under the
    same names, with different readings, and a backup file holding the right glove's readings again."""
    import h5py
    from test_formats import _clip, _glove_h5
    data = root / 'hdf5_gloves'
    data.mkdir(parents=True)
    _clip(data / 'scene.mp4', 90)
    for side in ('left', 'right', 'right_backup'):
        _glove_h5(data / f'{side}_glove.h5', 90)
        with h5py.File(data / f'{side}_glove.h5', 'a') as h:
            h.attrs[f'{side}_glove_serial'] = f'GLV-{side}-0042'
            if side.startswith('right'):
                h['pressure'][...] = h['pressure'][()][::-1]     # the other hand presses at other times
    return Setup('hdf5_gloves', 'teleop_arms', data, [
        Item('recording', 'scene'), Item('camera', 'scene'),
        Item('signal', 'pressure', 'left_glove.h5'), Item('signal', 'pressure', 'right_glove.h5'),
        Item('signal', 'pressure', 'right_backup_glove.h5'),
        Item('field', 'left_glove_serial'), Item('field', 'right_glove_serial')])


def depth_mcap(root: Path) -> Setup:
    """An MCAP with metric depth: raw front and wrist cameras, the front camera's 16-bit depth and the arm's named
    joint states."""
    from test_ingestion_v5 import raw_camera_mcap
    data = root / 'depth_mcap'
    data.mkdir(parents=True)
    raw_camera_mcap(data / 'depth_take.mcap')
    return Setup('depth_mcap', 'teleop_arms', data, [
        Item('recording', 'depth_take'), Item('camera', '/camera/front/image_raw'),
        Item('camera', '/camera/wrist/image_raw'), Item('depth', '/camera/front/depth'),
        Item('signal', '/joint_states')])


def tactile_device_mcap(root: Path) -> Setup:
    """A handheld MCAP shaped like the Sparsh DIGIT 360 bag: the sensor's own gel image beside its pressure topic in
    one device namespace, and a RealSense scene camera. The gel image is a tactile camera, never the anchor."""
    from test_camera_slots import D360, PRESSURE, SCENE, gel, pressure, scene
    from test_reader_cameras import _json_mcap
    data = root / 'tactile_device_mcap'
    data.mkdir(parents=True)
    _json_mcap(data / 'lego_tap.mcap', {D360: ('sensor_msgs/msg/CompressedImage', gel),
                                       PRESSURE: ('d360_msgs/msg/PressureD360', pressure),
                                       SCENE: ('sensor_msgs/msg/CompressedImage', scene)}, n=90)
    return Setup('tactile_device_mcap', 'handheld_gripper', data, [
        Item('recording', 'lego_tap'), Item('camera', D360), Item('camera', SCENE), Item('signal', PRESSURE)])


def tactile_lerobot(root: Path) -> Setup:
    """A LeRobot dataset with touch: a numeric glove array, an 8 x 8 pad stored as grey PNG images and a fingertip
    pad stored as RGB PNG images, beside the scene camera and the arm state, in two episodes."""
    import pandas  # noqa: F401 - the LeRobot writer needs it
    from test_formats import _lerobot, _png
    from test_prepare import joint_names
    rng = np.random.default_rng(5)
    n = 60
    data = root / 'tactile_lerobot'
    state = np.cumsum(rng.normal(0, 0.01, (n, 14)), axis=0)
    press = np.clip(np.sin(np.arange(n) / 9.0), 0, None)
    pad = [{'bytes': _png((rng.random((8, 8)) * 60 + 150 * p).astype(np.uint8))} for p in press]
    tip = [{'bytes': _png(np.stack([(rng.random((6, 6)) * 80 + 120 * p).astype(np.uint8)] * 3, -1))} for p in press]
    glove = [rng.random(16) * 0.2 + p for p in press]
    episode = {'observation.state': list(state), 'glove': glove, 'tactile.pad': pad, 'tactile.fingertip': tip}
    _lerobot(data, {0: episode, 1: {**episode, 'glove': glove[::-1]}},
             feats={'observation.state': {'dtype': 'float32', 'shape': [14], 'names': joint_names()},
                    'tactile.pad': {'dtype': 'image', 'shape': [8, 8, 1]},
                    'tactile.fingertip': {'dtype': 'image', 'shape': [6, 6, 3]}}, n_video=n)
    return Setup('tactile_lerobot', 'teleop_arms', data, [
        Item('recording', '000000'), Item('recording', '000001'), Item('camera', 'observation.images.cam_high'),
        Item('signal', 'observation.state'), Item('signal', 'glove'), Item('signal', 'tactile.pad'),
        Item('signal', 'tactile.fingertip')])


def plain_video(root: Path) -> Setup:
    """A single video file and its sidecar of recorded notes."""
    from test_formats import _clip
    data = root / 'plain_video'
    data.mkdir(parents=True)
    _clip(data / 'take_one.mp4', 90)
    (data / 'take_one.json').write_text(json.dumps({'operator_id': 'op-7', 'lighting_note': 'dim lamp'}))
    return Setup('plain_video', 'ego_head', data, [Item('recording', 'take_one'), Item('camera', 'take_one'),
                                                   Item('field', 'operator_id'), Item('field', 'lighting_note')])


def zipped_mcap(root: Path) -> Setup:
    """The MCAP setup uploaded as a ZIP archive of its folder."""
    inner = mcap_arm(root / 'unzipped')
    archive = root / 'zipped_mcap.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(inner.data.rglob('*')):
            if path.is_file():
                z.write(path, path.relative_to(inner.data.parent).as_posix())
    return Setup('zipped_mcap', inner.rig, archive, inner.items)


def long_hdf5(root: Path) -> Setup:
    """A recording longer than its request part: two cameras, a force channel and a pressure pad in one HDF5 file,
    20s at 10 fps. The test lowers the part length (Setup.part_s) instead of recording minutes, so it is labelled in
    two parts and stitched."""
    import h5py
    data = root / 'long_hdf5'
    data.mkdir(parents=True)
    n, rng = 200, np.random.default_rng(7)
    press = np.clip(np.sin(np.arange(n) / 15.0), 0, None)
    with h5py.File(data / 'long_take.h5', 'w') as h:
        h.attrs['fps'] = 10
        h['images/top'] = (rng.random((n, 48, 64, 3)) * 255).astype(np.uint8)
        h['images/wrist'] = (rng.random((n, 48, 64, 3)) * 255).astype(np.uint8)
        h['force'] = np.cos(np.arange(n) / 7.0)
        h['left_pressure'] = rng.random((n, 4, 4)) * 0.2 + press[:, None, None]
    return Setup('long_hdf5', 'teleop_arms', data, [
        Item('recording', 'long_take'), Item('camera', 'images/top'), Item('camera', 'images/wrist'),
        Item('signal', 'force'), Item('signal', 'left_pressure')], part_s=12.0)


def part_failure(fault: str):
    """long_hdf5 with its second part's annotation failed by the offline model: a provider error (error), a reply cut
    off at the output limit (cutoff), or a reply the completion rule rejects for holding no annotation (rejected)."""
    def build(root: Path) -> Setup:
        setup = long_hdf5(root)
        return Setup(f'part_{fault}', setup.rig, setup.data, setup.items, part_s=setup.part_s, fault=fault)
    build.__name__ = f'part_{fault}'
    build.__doc__ = part_failure.__doc__
    return build


part_error, part_cutoff, part_rejected = (part_failure(fault) for fault in ('error', 'cutoff', 'rejected'))


def regional_glove(root: Path) -> Setup:
    """A pressure glove in the OpenTouch layout, whose bundled publisher contract declares the hand regions, so the
    pad gets its separate regional pass: the publisher-schema builder's file, lengthened to 4s of presses."""
    import h5py
    from PIL import Image
    from test_publisher_schemas import recording
    data = root / 'regional_glove'
    data.mkdir(parents=True)
    path = data / 'glove_take.h5'
    recording(path)
    n = 40
    press = np.clip(np.sin(np.arange(n) / 4.0), 0, None)
    with h5py.File(path, 'a') as h:
        group = h['data/demo_00']
        for name in list(group):
            shape, dtype = group[name].shape[1:], group[name].dtype
            del group[name]
            if name == 'timestamps':
                group[name] = np.arange(n) / 10
            elif name == 'rgb':
                frames = group.create_dataset(name, (n,), dtype=h5py.vlen_dtype(np.uint8))
                for row in range(n):
                    buffer = io.BytesIO()
                    Image.new('RGB', (64, 36), (20 + 4 * row, 80, 130)).save(buffer, format='JPEG')
                    frames[row] = np.frombuffer(buffer.getvalue(), np.uint8)
            elif name == 'right_pressure':
                base = np.full((n, *shape), 3000.0, dtype)
                base[:, :8, :] -= 900 * press[:, None, None]
                base[:, 8:, :] -= 300 * press[::-1, None, None]
                group[name] = base
            elif name == 'right_hand_landmarks':
                hand = np.random.default_rng(3).random(shape)               # 21 distinct points, curling a little
                group[name] = (hand[None] * (1 + 0.05 * press[:, None, None])).astype(dtype)
            else:
                group[name] = (np.ones((n, *shape)) + np.arange(n).reshape(-1, *[1] * len(shape)) * 0.01).astype(dtype)
    return Setup('regional_glove', 'ego_head', data, [
        Item('recording', 'glove_take'), Item('camera', 'rgb'), Item('signal', 'right_pressure'),
        Item('signal', 'right_hand_landmarks'), Item('signal', 'right_palm_pos'), Item('signal', 'right_wrist_pos')])


def text_mcap(path: Path, task: list, steps: list) -> None:
    """A 4s robot MCAP: a front camera, the task topic's texts and the step topic's, each a list of (seconds, text)
    on the recorder's clock."""
    import base64
    from mcap.writer import Writer
    from PIL import Image
    wall = 1_790_000_000
    with path.open('wb') as target:
        writer = Writer(target)
        writer.start()
        image = writer.register_schema('sensor_msgs/CompressedImage', 'jsonschema', b'{}')
        text = writer.register_schema('TaskText', 'jsonschema', b'{}')
        camera = writer.register_channel('/camera/front/compressed', 'json', image)
        for k in range(121):
            png = io.BytesIO()
            Image.new('RGB', (32, 24), ((7 * k) % 256, 80, 130)).save(png, 'PNG')
            body = json.dumps({'format': 'png', 'data': base64.b64encode(png.getvalue()).decode()}).encode()
            writer.add_message(camera, round((wall + k / 30) * 1e9), body, 0)
        for topic, rows in (('/task', task), ('/task/subtask', steps)):
            channel = writer.register_channel(topic, 'json', text)
            for at, label in rows:
                writer.add_message(channel, round((wall + at) * 1e9), json.dumps({'data': label}).encode(), 0)
        writer.finish()


def steps_mcap(root: Path) -> Setup:
    """Per-task prompts only: a robot MCAP whose task topic changes as the work does, beside its step topic, so the
    dataset times its own prompts and gives the recording no instruction."""
    data = root / 'steps_mcap'
    data.mkdir(parents=True)
    text_mcap(data / 'steps_take.mcap', [(0.0, 'stack the cups'), (2.0, 'wipe the table')],
              [(0.0, 'pick up a cup'), (1.0, 'place it on the stack'), (2.0, 'wipe')])
    return Setup('steps_mcap', 'teleop_arms', data, [Item('recording', 'steps_take'),
                                                     Item('camera', '/camera/front/compressed')])


def both_mcap(root: Path) -> Setup:
    """An instruction and per-task prompts: one task text for the recording, and its timed steps."""
    data = root / 'both_mcap'
    data.mkdir(parents=True)
    text_mcap(data / 'both_take.mcap', [(0.0, 'stack the cups')], [(0.0, 'pick up a cup'), (2.0, 'place it on the stack')])
    return Setup('both_mcap', 'teleop_arms', data, [Item('recording', 'both_take'),
                                                    Item('camera', '/camera/front/compressed')])


def one_task_mcap(root: Path) -> Setup:
    """Neither an instruction nor timed steps, and one task: the task's verdict is the recording's."""
    setup = mcap_arm(root)
    return Setup('one_task_mcap', setup.rig, setup.data, setup.items, tasks=1)


def long_one_task(root: Path) -> Setup:
    """Neither, one task over a recording labelled in parts: the task is joined across the cuts and its verdict is the
    recording's."""
    setup = long_hdf5(root)
    return Setup('long_one_task', setup.rig, setup.data, setup.items, part_s=setup.part_s, tasks=1)


SETUPS = [teleop_lerobot, mcap_arm, umi_gripper, ego_head, hdf5_gloves, depth_mcap, tactile_lerobot,
           plain_video, zipped_mcap, long_hdf5, regional_glove, tactile_device_mcap, part_error, part_cutoff,
           part_rejected, steps_mcap, both_mcap, one_task_mcap, long_one_task]


# ---- the offline model -------------------------------------------------------------------------------------------------

FINDING_LISTS = ('sensor_findings', 'evidence_findings')


def _texts(body: dict) -> list[str]:
    out = []
    for message in body.get('messages') or []:
        content = message.get('content')
        for part in content if isinstance(content, list) else [{'type': 'text', 'text': content}]:
            if isinstance(part, dict) and part.get('type') == 'text' and isinstance(part.get('text'), str):
                out.append(part['text'])
    return out


def _json_lines(text: str):
    """Every line of a request text that is a JSON object."""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('{'):
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict):
                yield value


def _expand_clock(row: dict, clocks: dict, key: str = 'times'):
    if key in row:
        return row[key]
    return clocks.get(row.get('time_clock')) or []


class OfflineModel:
    """The provider, answered here: urllib.request.urlopen as the harness calls it. Each reply cites every camera,
    sensor, series and inspection its request offered, in every list the schema allows (data issues and camera
    observations name the cameras, sensor_findings the supplied sensors, evidence_findings the inspections), and each
    finding is written in both finding lists. Its evidence selection asks for every measured field, three a round. The
    reply is kept in the run record, so what it cited is read back from there (unit_losses)."""

    def __init__(self):
        self.calls = []
        self.selected = {}       # an inventory's field ids -> the ids already requested from it
        self.written = 0         # findings and issues written so far, so every tag is unique across the parts
        self.fault = None        # Setup.fault
        self.tasks = 2           # Setup.tasks

    def tag(self, kind: str) -> str:
        self.written += 1
        return f'offline {kind} {self.written}'

    def __call__(self, request, timeout=None):
        body = json.loads(request.data)
        texts = _texts(body)
        whole = '\n'.join(texts)
        if 'Interpret recorded field descriptors' in whole:
            kind, reply = 'dictionary', {'entries': []}
        elif 'You choose the image resolution' in whole:
            kind, reply = 'route', {'fine_detail': False, 'why': 'whole objects'}
        elif 'FIELD INVENTORY' in whole:
            kind, reply = 'selection', self.selection(whole)
        elif 'A person will watch this recording with short cards' in whole:
            kind, reply = 'cards', self.cards(whole)
        elif 'Analyze the supplied regional sensor responses' in whole:
            kind, reply = 'regional', {'sensor_findings': [
                {**f, 'observation': self.tag('regional')} for f in self.sensor_findings(texts)]}
        else:
            kind, reply = 'final', self.final(texts)
        self.calls.append(kind)
        content, finish = json.dumps(reply), 'stop'
        if kind == 'final' and self.fault and 'This request is part 2 of' in whole:
            self.calls[-1] = f'final {self.fault}'
            if self.fault == 'error':
                import email.message
                import urllib.error
                raise urllib.error.HTTPError(request.full_url, 400, 'Bad Request', email.message.Message(),
                                             io.BytesIO(b'{"error":{"message":"offline provider failure"}}'))
            if self.fault == 'cutoff':
                content, finish = content[:len(content) // 2], 'length'
            if self.fault == 'rejected':
                content = json.dumps({'timeline_columns': [], 'timeline': []})
        payload = {'id': f'gen-offline-{len(self.calls)}', 'model': body.get('model'),
                   'usage': {'cost': 0.001, 'prompt_tokens': 10, 'completion_tokens': 10},
                   'choices': [{'finish_reason': finish, 'message': {'content': content}}]}
        return io.BytesIO(json.dumps(payload).encode())

    # -- evidence selection: request every measured field once, three a round
    def selection(self, text: str) -> dict:
        inventory = text[text.index('FIELD INVENTORY'):]
        fields = []
        for doc in _json_lines(inventory):
            fields += [f for f in doc.get('fields') or [] if isinstance(f, dict)]
        duration = next((d.get('duration_s') for d in _json_lines(text) if isinstance(d.get('duration_s'), (int, float))),
                        None)
        key = tuple(sorted(f.get('id') for f in fields if isinstance(f.get('id'), str)))
        if 'INSPECTED EVIDENCE' not in text and '"error"' not in text:
            self.selected.pop(key, None)      # a first round: another episode can share every field id
        done = self.selected.setdefault(key, set())
        requests = []
        for doc in _json_lines(text):
            # a field whose clock is not the episode's is asked again by its source rows, as the refusal says
            asked = doc.get('request') if isinstance(doc.get('request'), dict) else {}
            if 'source rows' in str(doc.get('error')) and 'start_s' in asked and len(requests) < 3:
                requests.append({'field_id': asked.get('field_id'), 'mode': asked.get('mode'),
                                 'reason': asked.get('reason'), 'start_row': 0, 'end_row': 16})
        for f in fields:
            modes = f.get('available_modes') or []
            if f.get('id') in done or f.get('kind') in ('context',) or not modes:
                continue
            mode = next((m for m in ('samples', 'images', 'metadata', 'extrema', 'regions') if m in modes), modes[0])
            request = {'field_id': f['id'], 'mode': mode, 'reason': 'what this field recorded over the episode'}
            rows = (f.get('shape') or [None])[0]
            if mode != 'metadata' and f.get('kind') == 'native' and isinstance(rows, int):
                request.update(start_row=0, end_row=min(rows, 16))       # a native clock is not the episode's
            elif mode != 'metadata' and duration:
                request.update(start_s=0.0, end_s=float(duration))
            requests.append(request)
            done.add(f['id'])
            if len(requests) == 3:
                break
        return {'requests': requests}

    # -- sensor_findings: one finding per supplied sensor, citing its own supplied samples
    def sensor_findings(self, texts: list[str]) -> list:
        findings = []
        for text in texts:
            for doc in _json_lines(text):
                if not isinstance(doc.get('sensors'), list) or 'series' not in doc and 'coverage' not in doc:
                    continue
                clocks = doc.get('sample_clocks') or {}
                for sensor in doc['sensors']:
                    finding = self._sensor_finding(sensor, doc.get('series') or [], clocks)
                    if finding:
                        findings.append(finding)
        return findings

    @staticmethod
    def _sensor_finding(sensor, series, clocks):
        ident = sensor.get('id')
        capabilities = sensor.get('capabilities') or []
        if not isinstance(ident, str) or not capabilities:
            return None
        ref = None
        # its own traces first, then a derived one (a hand-shape distance is the only trace a landmark set supplies)
        for row in sorted(series, key=lambda row: bool(row.get('derivation_id'))):
            if row.get('sensor_id') != ident:
                continue
            times = _expand_clock(row, clocks)
            values = row.get('values') or []
            valid = [t for t, v in zip(times, values) if v is not None]
            if valid:
                ref = {'sensor_id': ident, 'series_id': row['id'], 'time_s': sorted({valid[0], valid[-1]})}
                break
        array = sensor.get('array_evidence') or {}
        if ref is None and array.get('frames'):
            valid = [f['time_s'] for f in array['frames'] if f.get('valid_cells')]
            if valid:
                ref = {'sensor_id': ident, 'time_s': sorted({valid[0], valid[-1]})}
        if ref is None and sensor.get('times'):
            ref = {'sensor_id': ident, 'time_s': sorted({sensor['times'][0], sensor['times'][-1]})}
        if ref is None:
            return None
        return {'start_s': ref['time_s'][0], 'end_s': ref['time_s'][-1], 'inspect_s': ref['time_s'][0],
                'headline': f'{sensor.get("name") or ident} readings change', 'detail': 'The readings rise and fall.',
                'observation': 'Recorded readings at the cited times.', 'claim': 'The readings vary.',
                'alternative': 'Noise.', 'confidence': 'low', 'adds_beyond_video': True,
                'claim_type': capabilities[0], 'evidence': [ref]}

    # -- evidence_findings: one finding per inspection, citing its own inspected rows or times
    @staticmethod
    def evidence_findings(texts: list[str]) -> list:
        findings = []
        for text in texts:
            if not text.startswith('INSPECTED EVIDENCE'):
                continue
            for receipt in _json_lines(text):
                ident = receipt.get('id')
                if not isinstance(ident, str) or not ident.startswith('inspection:'):
                    continue
                base = {'headline': 'Inspected readings', 'detail': 'The inspected readings vary.',
                        'observation': 'Inspected values.', 'claim': 'The field varies.', 'alternative': 'Noise.',
                        'confidence': 'low', 'adds_beyond_video': True}
                times = receipt.get('times_s') or [f.get('time_s') for f in receipt.get('frames') or []
                                                   if isinstance(f, dict) and f.get('time_s') is not None]
                columns = receipt.get('columns')
                values = receipt.get('values')
                if receipt.get('mode') == 'metadata':
                    findings.append({**base, 'start_s': None, 'end_s': None,
                                     'evidence': [{'evidence_id': ident, 'pointer': receipt.get('pointer')}]})
                    continue
                if times:
                    if isinstance(values, list) and columns:
                        times = [t for t, row in zip(times, values) if all(v is not None for v in row)]
                    if not times:
                        continue
                    cite = sorted({times[0], times[-1]})
                    ref = {'evidence_id': ident, 'time_s': cite, **({'columns': columns} if columns else {})}
                    findings.append({**base, 'start_s': cite[0], 'end_s': cite[-1], 'evidence': [ref]})
                elif receipt.get('rows'):
                    rows = receipt['rows']
                    if isinstance(values, list) and columns:
                        rows = [r for r, row in zip(rows, values) if all(v is not None for v in row)]
                    if rows:
                        findings.append({**base, 'start_s': None, 'end_s': None, 'evidence': [
                            {'evidence_id': ident, 'rows': [rows[0]], **({'columns': columns} if columns else {})}]})
        return findings

    def final(self, texts: list[str]) -> dict:
        """The episode annotation. Every citable finding goes in both finding lists, each copy tagged in its
        observation (offline <list> <n>) so it can be found again after binding; a citation in the list that does
        not own it must be routed or set aside with its reason, never dropped."""
        whole = '\n'.join(texts)
        cameras, begin, end = [], None, 0.0
        for start, stop, caption in re.findall(r'=== grid t=([0-9.]+)-([0-9.]+)s \| rows ([^|]*) \|', whole):
            cameras += [c.strip() for c in caption.split(',') if c.strip() and c.strip() not in cameras]
            begin = float(start) if begin is None else min(begin, float(start))
            end = max(end, float(stop))
        observed = re.search(r'The supplied view IDs and image-row labels are: (.*?)\. Include', whole)
        views = [pair.split(':', 1)[0].strip() for pair in observed.group(1).split(';')] if observed else []
        cited = self.sensor_findings(texts) + self.evidence_findings(texts)
        # the work every request asks for (label/completeness.py): a timeline over the footage it sent, and the
        # outcome in each field its instructions name (the tasks below, the one task's completion)
        reply = {'task_summary': 'Move the object.',
                 'timeline': [{'start_s': begin or 0.0, 'end_s': end, 'arm': 'right', 'action': 'reach',
                               'contribution': 'advancing', 'progress': 1.0}] if end else [],
                 'data_issues': [{'issue': f'Camera {c} is dim.', 'category': 'camera_fault', 'severity': 'low',
                                  't_s': None, 'end_s': None, 'evidence': f'{self.tag("data_issues")}: camera {c}'}
                                 for c in cameras]}
        if cameras:
            # an issue that only says the clip ends mid-task, at the request's last instant: at an internal cut the
            # stitcher sets it aside with its reason (label/pieces.py is_cut_artifact), elsewhere it stands
            reply['data_issues'].append({'issue': 'The clip ends mid-task.', 'category': 'truncated_episode',
                                         'severity': 'medium', 't_s': end, 'end_s': end,
                                         'evidence': f'{self.tag("data_issues")}: camera {cameras[0]} at {end}s'})
        for name in FINDING_LISTS:
            reply[name] = [{**json.loads(json.dumps(f)), 'observation': self.tag(name)} for f in cited]
        if views:
            reply['camera_observations'] = [{'view': v, 'content': 'scene', 'evidence': f'offline view {v}'}
                                            for v in views]
        if '"tasks": [' in whole:
            # the first task is judged against the most specific claim the request offers (a timed step, else the
            # instruction, else its own end state); a second, unasked one against its own end state, so a recording
            # with an instruction keeps its completion and one without has no single verdict
            claim = ('subtask' if "ABOUT THE DATASET'S TIMED STEPS" in whole else
                     'instruction' if "ABOUT THE EPISODE'S INSTRUCTION" in whole else 'own_end_state')
            half = round(end / 2, 3)
            rows = [(0.0, half, 'success', claim, None), (half, end, 'partial', 'own_end_state', None)]
            if self.tasks == 1:
                # one task over the whole footage; in a part of a longer recording it continues across each cut
                part = re.search(r'This request is part (\d+) of (\d+)', whole)
                first, last = (part.group(1) == '1', part.group(1) == part.group(2)) if part else (True, True)
                note = ' '.join([*([] if first else ['Continues the task from the previous part.']),
                                 *([] if last else ['The clip ends while the task is still under way.'])]) or None
                rows = [(begin or 0.0, end, 'success' if last else 'partial', claim, note)]
            reply['tasks'] = [{'start_s': a, 'end_s': b, 'task': f'{self.tag("tasks")}: move the object',
                               'objects': ['object'], 'judged_against': against,
                               'subtask': 1 if against == 'subtask' else None, 'outcome': outcome,
                               'success_predicate': 'the object is moved',
                               'completed_at_s': b if outcome == 'success' else None, 'note': note}
                              for a, b, outcome, against, note in rows]
        if '"task_completed"' in whole:
            reply['completion'] = {'task_completed': 'success'}
        if '"goal_alignment"' in whole:
            reply['goal_alignment'] = {'matches_given': True, 'relation': 'aligned',
                                       'note': f'{self.tag("goal_alignment")}: the work matches the instruction'}
        return reply

    def cards(self, text: str) -> dict:
        recording = text[text.index('\nRECORDING\n') + len('\nRECORDING\n'):]
        rows = json.loads(recording).get('findings') or []
        return {'cards': [{'finding': row['index'], 'headline': 'The readings change.',
                           'detail': 'They rise and then fall.'} for row in rows[:1]],
                'list': [{'finding': row['index'], 'headline': 'The readings change.',
                          'detail': 'They rise and then fall.'} for row in rows[1:]]}


# ---- running review ----------------------------------------------------------------------------------------------------

def run_review(setup: Setup, job: Path, model: OfflineModel, monkeypatch) -> int:
    """python -m review on the fixture, as a person runs it, with the model answered offline. The checks and clips run
    as the child processes review starts; the dry run and the label run are the harness's own main, in this process,
    so they reach the offline model."""
    from label import harness
    from review import __main__ as review
    monkeypatch.setenv('OPENROUTER_API_KEYS', 'sk-or-offline-fixture')
    monkeypatch.setattr(harness.urllib.request, 'urlopen', model)
    real_step = review.run_step

    def step(job, kind, command, env, **kwargs):
        if kind not in ('dry_run', 'label'):
            return real_step(job, kind, command, env, **kwargs)
        with monkeypatch.context() as local:
            local.setattr(sys, 'argv', command[2:])
            for name in ('ARGUS_REVIEW_OUTPUT_LOCK_FD', 'ARGUS_REVIEW_OUTPUT_LOCK_PATH'):
                if name in env:
                    local.setenv(name, env[name])
            return harness.main()
    monkeypatch.setattr(review, 'run_step', step)
    if setup.part_s:
        # the part length is the one saved in each episode's annotation policy, which resolve keeps when it is
        # within the default; a shorter one here cuts a 20s recording as a long one is cut
        from label import policy
        persist = policy.persist

        def shorter_parts(episodes, **overrides):
            saved = persist(episodes, **overrides)
            for name in saved:
                path = Path(episodes) / name / 'context.json'
                ctx = json.loads(path.read_text())
                ctx['annotation_policy']['part_max_s'] = setup.part_s
                path.write_text(json.dumps(ctx))
                saved[name] = ctx['annotation_policy']
            return saved
        monkeypatch.setattr(policy, 'persist', shorter_parts)
    monkeypatch.setattr(sys, 'argv', ['review', '--data', str(setup.data), '--rig', setup.rig, '--out', str(job),
                                      '--concurrency', '1', '--jobs', '1', '--cap', '20', '--max-tokens', '4000'])
    return review.main()


# ---- what each stage holds, and why something is not there -----------------------------------------------------------

WORD = r'[A-Za-z0-9_]'
# A recorded reason says that something is not kept. Notes that only qualify a kept stream (its clock, its alignment)
# name it too, so a reason outside the pipeline's own left-out records must say so in these words.
LEFT_OUT = re.compile(r'left out|not read|could not|cannot be read|unreadable|not kept|not decod|does not decode|'
                      r'no reading|not placed|set aside|withheld|not supplied|skipped|dropped|not labelled|no uploaded|'
                      r'not uploaded|not shown|excluded|unbound|not bound', re.I)


def mentions(text, name: str) -> bool:
    return isinstance(text, str) and re.search(rf'(?<!{WORD}){re.escape(name)}(?!{WORD})', text) is not None


def strings(value):
    """Every string in a JSON value, keys included."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from strings(item)


@dataclass
class Carrier:
    """One stream a stage holds: its kind, the name later stages know it by, and every name it records for itself."""
    kind: str
    ident: str
    texts: list


def _read(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def prepared_carriers(ep: Path, ctx: dict) -> list:
    sources = _read(ep / 'sources.json') or {}
    out = [Carrier('camera', view, [view, *strings(rec), *strings(sources.get(view))])
           for view, rec in (ctx.get('cameras') or {}).items()]
    out += [Carrier('depth', view, [view, *strings(rec)]) for view, rec in (ctx.get('depth') or {}).items()]
    out += [Carrier('signal', s['name'], [s['name'], *strings(s)])
            for s in ctx.get('signals') or [] if isinstance(s, dict) and isinstance(s.get('name'), str)]
    if (ep / 'state.npz').exists():
        held = set(np.load(ep / 'state.npz').files)
        described = [*strings(ctx.get('state_identities')), *strings(ctx.get('pose_field_bindings')),
                     *strings((ctx.get('source') or {}).get('state'))]
        for name in ('state', 'action'):
            if name in held:
                out.append(Carrier('signal', name, [name, *described]))
    out += [Carrier('field', key, [key, *strings(value)]) for key, value in ctx.items()
            if key not in ('reader_issues', 'cameras', 'signals', 'depth')]
    return out


def _inventory_carriers(inventory, cameras=True) -> list:
    out = []
    for f in inventory or []:
        if not isinstance(f, dict):
            continue
        kind = {'depth': 'depth', 'metadata': 'field', 'context': 'field', 'image': 'camera'}.get(f.get('kind'), 'signal')
        if kind != 'camera' or cameras:
            out.append(Carrier(kind, f.get('name'), [f.get('name'), *strings(f.get('descriptor'))]))
    return out


def request_carriers(record: dict, prompt: str) -> list:
    """What the request offered the model: its cameras, its inspection inventory, its sensor evidence, and the
    readout of each recorded field in the prompt itself."""
    config = record.get('config') or {}
    labels = dict(zip(config.get('views') or [], config.get('cam_labels') or []))
    out = [Carrier('camera', view, [view, labels.get(view)]) for view in config.get('views') or []]
    out += _inventory_carriers((record.get('evidence_inspection') or {}).get('inventory'))
    out += [Carrier('signal', s.get('name'), [s.get('name'), s.get('source')])
            for s in (record.get('sensor_evidence') or {}).get('sensors') or [] if isinstance(s, dict)]
    out += [Carrier(kind, '(prompt)', [prompt]) for kind in ('signal', 'field', 'depth')]
    return out


def record_carriers(d: dict) -> list:
    """What an episode record (the board's file, or its export) names itself."""
    out = [Carrier('camera', view, [view]) for view in d.get('camera_views') or []]
    out += _inventory_carriers((d.get('evidence_inspection') or {}).get('inventory'), cameras=False)
    out += [Carrier('signal', s.get('name'), [s.get('name'), s.get('source')])
            for s in (d.get('sensor_evidence') or {}).get('sensors') or [] if isinstance(s, dict)]
    out.append(Carrier('field', '(record)', list(strings(d))))
    return out


def board_carriers(job: Path, eid: str, d: dict) -> list:
    """What the board shows: each camera whose clip it has, each depth stream with its clip, each signal of its sensor
    lanes, and what its episode record names. A camera named in the record with no clip is not on the board."""
    from board.clips import DEPTH, clip_path
    clips = job / 'clips'
    out = [Carrier('camera', view, [view]) for view in d.get('camera_views') or []
           if clip_path(clips, eid, view).exists()]
    lanes = _read(job / 'sensors' / f'{eid}.json') or {}
    out += [Carrier('depth', view, [view]) for view in (lanes.get('depth') or {})
            if clip_path(clips, eid, DEPTH + view).exists()]
    out += [Carrier('signal', s.get('name'), [s.get('name'), s.get('source')])
            for s in lanes.get('signals') or [] if isinstance(s, dict)]
    return out + [c for c in record_carriers(d) if c.kind != 'camera']


def present(carriers: list, kind: str, ident: str) -> bool:
    """Whether a stage holds the stream an earlier stage knew by this name."""
    kinds = {kind, 'camera'} if kind == 'depth' else {kind}
    return any(c.kind in kinds and any(mentions(t, ident) for t in c.texts) for c in carriers)


def resolve(items: list, carriers: list, files: set) -> dict:
    """The prepared streams each fixture item became, {id(item): [names]}. A stream that names an item's name holds
    it; when items share a name, a stream holds those whose files it names, or, naming no file, just one of them. So one
    stream never passes for two streams of the same name unless it names both their files."""
    held, by_name = {}, {}
    for item in items:
        by_name.setdefault((item.kind, item.name), []).append(item)
    for (kind, name), group in by_name.items():
        kinds = {'camera', 'depth'} if kind == 'depth' else {kind}
        candidates = [c for c in carriers if c.kind in kinds and any(mentions(t, name) for t in c.texts)]
        if len(group) == 1:
            held[id(group[0])] = [c.ident for c in candidates]
            continue
        free = []
        for c in candidates:
            named = {f for f in files if any(mentions(t, f) for t in c.texts)}
            owners = [item for item in group if item.file in named]
            for owner in owners:            # a kept copy that names every file it holds serves each of them
                held.setdefault(id(owner), []).append(c.ident)
            if not named:
                free.append(c)
        for item in group:
            if not held.get(id(item)) and free:
                held[id(item)] = [free.pop(0).ident]
    return held


def explained(reasons: list, name: str, file: str | None = None) -> str | None:
    return next((text for text in reasons if mentions(text, name) and (file is None or mentions(text, file))), None)


def left_out_texts(values) -> list:
    """The pipeline's own records of what it left out: every string in them is a reason."""
    return [t for value in values for t in strings(value)]


def flagged_texts(values) -> list:
    """Notes and limitations: a reason only where the sentence says something is not kept."""
    return [t for value in values for t in strings(value) if LEFT_OUT.search(t)]


# ---- the export -----------------------------------------------------------------------------------------------------------

def export_line(d: dict, name: str) -> dict:
    """The episode as POST /api/export writes it: the board's file after the public projection, as one JSON line."""
    from board import serve
    label = serve.finite(serve.public_label(serve.current_label(d, serve.DICTIONARY_JOB, name), export=True))
    return json.loads(json.dumps(label, ensure_ascii=False, separators=(',', ':'), allow_nan=False))


# ---- the citations ----------------------------------------------------------------------------------------------------------

def finding_tags(rows) -> list:
    return [r['observation'] for r in rows or [] if isinstance(r, dict)
            and isinstance(r.get('observation'), str) and r['observation'].startswith('offline ')]


def unbound_tags(rows) -> dict:
    """{tag: reason} of the findings set aside with their reason."""
    out = {}
    for row in rows or []:
        if isinstance(row, dict) and isinstance(row.get('reason'), str) and row['reason'].strip():
            out.update({tag: row['reason'] for tag in finding_tags([row.get('finding')])})
    return out


def placed_findings(d: dict) -> tuple:
    """(tags of bound findings, {tag: reason} of findings set aside) in an episode record."""
    evidence = d.get('sensor_evidence') if isinstance(d.get('sensor_evidence'), dict) else {}
    inspection = d.get('evidence_inspection') if isinstance(d.get('evidence_inspection'), dict) else {}
    bound = {*finding_tags(evidence.get('findings')), *finding_tags(evidence.get('withheld_findings')),
             *finding_tags(inspection.get('untimed_findings'))}
    aside = {**unbound_tags(evidence.get('unbound_findings')), **unbound_tags(inspection.get('unbound_findings'))}
    return bound, aside


def issue_tags(d: dict) -> set:
    """Tags of the reply's data issues an episode record holds, among its issues or excluded by a rule with its reason
    (a run record holds them in its labels)."""
    labels = d.get('labels') if isinstance(d.get('labels'), dict) else {}
    excluded = d.get('_excluded') or labels.get('_excluded') or []
    excluded = [row for rows in excluded.values() for row in rows] if isinstance(excluded, dict) else excluded
    rows = [*(d.get('data_issues') or labels.get('data_issues') or []),
            *(row for row in excluded if isinstance(row, dict) and any(isinstance(row.get(k), str) and row[k].strip()
                                                                      for k in ('reason', 'excluded_reason', 'why')))]
    return {r['evidence'].split(':', 1)[0] for r in rows if isinstance(r, dict) and isinstance(r.get('evidence'), str)
            and r['evidence'].startswith('offline ')}


def observation_views(d: dict) -> tuple:
    co = d.get('camera_observations') if isinstance(d.get('camera_observations'), dict) else {}
    kept = {o.get('view') for o in co.get('observations') or [] if isinstance(o, dict)}
    withheld = [o for o in co.get('withheld_observations') or []
                if isinstance(o, dict) and isinstance(o.get('why'), str) and o['why'].strip()]
    # a withheld row names its view, the observation it holds back, or the views it covers (label/harness.py
    # bind_camera_observations, label/pieces.py merge_camera_observations)
    aside = ({o.get('view') for o in withheld} | {(o.get('observation') or {}).get('view') for o in withheld}
             | {v for o in withheld for v in o.get('views') or []})
    return kept, aside


def reply_of(record: dict) -> dict:
    labels = record.get('labels')
    return labels if isinstance(labels, dict) and not labels.get('_parse_error') else {}


def reply_citations(record: dict) -> dict:
    """{tag: evidence} of every finding a unit's reply wrote, its regional pass included."""
    reply = reply_of(record)
    cited = {t: f.get('evidence') for name in FINDING_LISTS for f in reply.get(name) or [] if isinstance(f, dict)
             for t in finding_tags([f])}
    analysis = record.get('sensor_analysis') if isinstance(record.get('sensor_analysis'), dict) else {}
    try:
        regional = json.loads(analysis.get('response_text') or '{}').get('sensor_findings')
    except (ValueError, AttributeError):
        regional = None
    cited.update({t: f.get('evidence') for f in regional or [] if isinstance(f, dict) for t in finding_tags([f])})
    return cited


def reply_issues(record: dict) -> set:
    return issue_tags(reply_of(record))


def tagged_tasks(d: dict) -> dict:
    """{tag: outcome} of the reply's tasks an episode record holds, a task joined across a cut by each task it joined,
    and a task a cut qualified by the outcome its part gave it."""
    labels = d.get('labels') if isinstance(d.get('labels'), dict) else {}
    out = {}
    for task in d.get('tasks') or labels.get('tasks') or []:
        for row in [task, *(task.get('joined_from') or [])] if isinstance(task, dict) else []:
            text = row.get('task') if isinstance(row, dict) else None
            if isinstance(text, str) and text.startswith('offline '):
                outcome = row.get('outcome_in_part') or row.get('outcome')
                # the board stores partial as a failure of the kind partial (board/to_board.py, board/scores.py)
                out[text.split(':', 1)[0]] = 'partial' if outcome == 'failure' and row.get('failure_kind') == 'partial' \
                    else outcome
    return out


def alignment_tags(d: dict) -> set:
    """Tags of the reply's goal alignments an episode record holds: its own, or each part's (part_assessments)."""
    labels = d.get('labels') if isinstance(d.get('labels'), dict) else {}
    rows = [d.get('goal_alignment') or labels.get('goal_alignment'),
            *(a.get('goal_alignment') for a in d.get('part_assessments') or labels.get('part_assessments') or []
              if isinstance(a, dict))]
    return {row['note'].split(':', 1)[0] for row in rows if isinstance(row, dict)
            and isinstance(row.get('note'), str) and row['note'].startswith('offline ')}


def reply_views(record: dict) -> set:
    return {r.get('view') for r in reply_of(record).get('camera_observations') or [] if isinstance(r, dict)}


def unit_losses(tag: str, record: dict) -> list:
    """At binding, in one request: every citation its reply made is bound or set aside with its reason, every camera
    observation is kept or withheld with its reason, and every sensor the request offered was cited."""
    if not reply_of(record):
        return [f'bound: {tag}the reply did not parse: {(record.get("labels") or {}).get("_parse_error")}']
    lost = []
    cited = reply_citations(record)
    bound, aside = placed_findings(record)
    lost += [f'bound: {tag}{t} citing {json.dumps(refs)[:160]} is neither bound nor set aside with a reason'
             for t, refs in cited.items() if t not in bound and t not in aside]
    kept, withheld = observation_views(record)
    lost += [f'bound: {tag}the camera observation of {v} is neither kept nor withheld with a reason'
             for v in sorted(reply_views(record) - kept - withheld, key=str)]
    # the reply cites every sensor the request offered, so a sensor no finding cites is one binding never saw
    refs = {r.get('sensor_id') or r.get('evidence_id') for f in cited.values() if isinstance(f, list)
            for r in f if isinstance(r, dict)}
    lost += [f'bound: {tag}offered sensor {s.get("id")} ({s.get("name")}) was never cited'
             for s in (record.get('sensor_evidence') or {}).get('sensors') or []
             if isinstance(s, dict) and s.get('id') not in refs and not str(s.get('id')).startswith('mechanics:')]
    return lost


def carried_losses(tag: str, units: list, stages: list) -> list:
    """What binding kept in every request (bound findings, findings set aside with their reason, data issues, camera
    observations) reaches each later stage: the stitched episode, the board and the export."""
    lost, bound, aside, issues, views, tasks, aligned = [], set(), {}, set(), set(), {}, set()
    for record in units:
        tasks.update(tagged_tasks(reply_of(record)))
        aligned |= alignment_tags(reply_of(record))
        got, got_aside = placed_findings(record)
        bound |= got
        aside.update(got_aside)
        issues |= reply_issues(record)
        kept, withheld = observation_views(record)
        views |= kept | withheld
    for stage, doc in stages:
        got, got_aside = placed_findings(doc)
        lost += [f'{stage}: {tag}bound finding {t} is missing' for t in sorted(bound - got)]
        lost += [f'{stage}: {tag}{t}, set aside at binding, is missing with its reason'
                 for t in sorted(set(aside) - set(got_aside) - got)]
        lost += [f'{stage}: {tag}{t} (a data issue naming a camera) is missing' for t in sorted(issues - issue_tags(doc))]
        kept, withheld = observation_views(doc)
        lost += [f'{stage}: {tag}the camera observation of {v} is neither kept nor withheld with a reason'
                 for v in sorted(views - kept - withheld, key=str)]
        held = tagged_tasks(doc)
        lost += [f'{stage}: {tag}task {t} is missing' for t in sorted(set(tasks) - set(held))]
        lost += [f'{stage}: {tag}task {t} has outcome {held[t]!r}, not its reply\'s {tasks[t]!r}'
                 for t in sorted(set(tasks) & set(held)) if held[t] != tasks[t]]
        lost += [f'{stage}: {tag}goal alignment {t} is missing' for t in sorted(aligned - alignment_tags(doc))]
    return lost


# ---- the inventory check ------------------------------------------------------------------------------------------------------

def unit_names(job: Path, eid: str) -> list:
    """The requests a recording was labelled in: its parts when it was cut (label/pieces.py), else itself."""
    return sorted(p.name for p in (job / 'units').glob(f'{eid}__p*') if p.is_dir()) or [eid]


def unit_record(job: Path, name: str) -> dict | None:
    """A request's record as the harness left it: its reply, or the reply cut off at the output limit (failed_), or
    the claim of a request that got none (noreply_)."""
    for prefix in ('', 'failed_', 'noreply_'):
        record = _read(job / 'run' / 'out' / f'{prefix}{name}.json')
        if record is not None:
            return record
    return None


def gap_losses(tag: str, part: int, record: dict | None, stages: list) -> list:
    """A part that gave no labels is named at every later stage as a gap with its span and reason, and the recording
    is never presented as whole: the stitched episode, the board and the export each list the part under the stitched
    gaps, the board and the export flag it at its span, and the recording's outcome is unclear and says why, or absent
    (a session's outcome is its tasks'). The stitched gap keeps the cause the harness recorded for the part: the
    parser's error, or why no reply came."""
    lost = []
    labels = (record or {}).get('labels') if isinstance((record or {}).get('labels'), dict) else {}
    cause = labels.get('_parse_error') or (record or {}).get('no_reply')
    for stage, doc in stages:
        stitched = doc.get('stitched') if stage == 'stitched' else doc.get('_stitched')
        gaps = [g for g in (stitched or {}).get('missing') or [] if isinstance(g, dict) and g.get('part') == part]
        if not gaps or not all(isinstance(g.get(k), (int, float)) for g in gaps for k in ('t0_s', 't1_s')) \
                or not all(isinstance(g.get('why'), str) and g['why'].strip() for g in gaps):
            lost.append(f'{stage}: {tag}part {part} gave no labels and is not named as a gap with its span and reason')
        elif stage == 'stitched' and isinstance(cause, str) and not any(cause in g['why'] for g in gaps):
            lost.append(f'{stage}: {tag}part {part}\'s gap says {gaps[0]["why"]!r}, not the recorded cause {cause!r}')
        if stage != 'stitched':
            flagged = [i for i in (doc.get('dataset_checks') or {}).get('reader_issues') or [] if isinstance(i, dict)
                       and i.get('kind') == 'part_not_labelled' and isinstance(i.get('t0_s'), (int, float))]
            if not any(gaps and abs(i['t0_s'] - gaps[0].get('t0_s', -1)) < 1e-6 for i in flagged):
                lost.append(f'{stage}: {tag}part {part} gave no labels and is not flagged at its span')
        completion = ((doc.get('labels') if stage == 'stitched' else doc) or {}).get('completion') or {}
        verdict = completion.get('task_completed')
        if verdict not in (None, 'unclear'):
            lost.append(f'{stage}: {tag}part {part} gave no labels, yet the recording has the outcome {verdict!r}')
        elif not mentions(str(completion.get('reason') or ''), f'part {part} has no labels') and verdict == 'unclear':
            lost.append(f'{stage}: {tag}the recording\'s outcome does not say part {part} has no labels')
    return lost


def losses(setup: Setup, job: Path) -> list:
    """Every item of the fixture's inventory, and every citation the replies made, that a stage holds neither itself
    nor in a recorded reason. A recording labelled in parts is checked in each part's request and binding, then in the
    stitched episode. Empty means nothing was lost."""
    lost = []
    report = _read(job / 'report.json') or {}
    set_aside = _read(job / 'clips' / 'failed.json') or {}
    report_reasons = (left_out_texts([report.get(k) for k in ('failed', 'skipped', 'missing')] + [set_aside])
                      + flagged_texts([report.get('notes')]))
    files = {p.name for p in (setup.data.rglob('*') if setup.data.is_dir() else [setup.data]) if p.is_file()}
    episodes = []
    for item in (i for i in setup.items if i.kind == 'recording'):
        found = [e['episode_id'] for e in report.get('episodes') or [] if mentions(e.get('name'), item.name)]
        episodes += found[:1]
        if not found and not explained(report_reasons, item.name):
            lost.append(f'prepared: {item} has no episode and no reason names it')
    streams = [i for i in setup.items if i.kind != 'recording']
    for eid in episodes:
        tag = f'{eid}: '
        ep = job / 'episodes' / eid
        ctx = _read(ep / 'context.json')
        stitched = _read(job / 'run_final' / 'out' / f'{eid}.json')
        board = _read(job / 'qa' / f'{eid}.json')
        names = unit_names(job, eid)
        records = {name: unit_record(job, name) for name in names}
        from label.harness import annotation_complete
        # a part that gave no labels is a gap: it is checked for being named as one, never as a part that was labelled
        gaps = [name for name, record in records.items() if record is None or not annotation_complete(record)]
        units = {name: record for name, record in records.items() if name not in gaps}
        missing = [stage for stage, doc in (('prepared', ctx), ('stitched', stitched), ('board', board)) if doc is None]
        if len(names) == 1 and gaps:
            lost.append(f'bound: {tag}the reply gave no labels: {(records[eid] or {}).get("labels")}')
        if missing:
            if not explained(report_reasons, eid):
                lost += [f'{stage}: {tag}the episode is missing with no reason' for stage in missing]
            continue
        exported = export_line(board, f'{eid}.json')
        source = ctx.get('source') if isinstance(ctx.get('source'), dict) else {}
        reasons = {'prepared': report_reasons + left_out_texts([v for k, v in source.items() if k.startswith('unused_')])
                   + flagged_texts([ctx.get('reader_issues')]),
                   'board': left_out_texts([(board.get('reader_notes') or {}).get('left_out'), set_aside])
                   + flagged_texts([(board.get('dataset_checks') or {}).get('reader_issues')])}
        stitched_evidence = stitched.get('sensor_evidence') or {}
        reasons['stitched'] = flagged_texts([stitched_evidence.get('limitations'),
                                             (stitched.get('evidence_inspection') or {}).get('limitations')])
        reasons['export'] = reasons['board']
        carriers = {'stitched': request_carriers(stitched, ''), 'board': board_carriers(job, eid, board),
                    'export': record_carriers(exported)}
        for name, record in records.items():
            if record is None:
                continue
            evidence, inspection = record.get('sensor_evidence') or {}, record.get('evidence_inspection') or {}
            carriers[name] = request_carriers(record, (_read(job / 'dry' / f'{name}.json') or {}).get('prompt_text') or '')
            reasons[name] = flagged_texts([evidence.get('limitations'), inspection.get('limitations')])
        held = resolve(streams, prepared_carriers(ep, ctx), files)
        for item in streams:
            # a recorded field is metadata text, so it is found by its own name at every stage
            idents = [item.name] if item.kind == 'field' and held.get(id(item)) else held.get(id(item)) or []
            if not idents:
                if not explained(reasons['prepared'], item.name, item.file):
                    lost.append(f'prepared: {tag}{item} is not in the prepared episode and no reason names it')
                continue
            # the stitched record joins the parts' labels and evidence; recorded fields stay in the episode's context,
            # which the board reads again, so they are checked in each part's request, the board and the export
            for stage in [*(n for n in names if records[n] is not None),
                          *(['stitched'] if item.kind != 'field' else []), 'board', 'export']:
                label = f'request ({stage})' if stage in records else stage
                for ident in idents:
                    if not (present(carriers[stage], item.kind, ident) or explained(reasons[stage], ident)
                            or explained(reasons[stage], item.name)):
                        lost.append(f'{label}: {tag}{item}, prepared as {ident!r}, is missing and no reason names it')
        for name, record in units.items():
            lost += unit_losses(f'{name}: ', record)
        lost += carried_losses(tag, list(units.values()),
                               [('stitched', stitched), ('board', board), ('export', exported)])
        for name in gaps if len(names) > 1 else []:
            lost += gap_losses(tag, names.index(name) + 1, records[name],
                               [('stitched', stitched), ('board', board), ('export', exported)])
    return lost


# ---- the test -------------------------------------------------------------------------------------------------------------

# Losses that open pull requests fix, by (setup, pattern of the loss): the reason names the pull request. Each must still
# occur, so when its pull request has merged and the loss is gone the test fails until its entry is removed.
EXPECTED = {}


CASES = [build.__name__ for build in SETUPS]
WORKERS = 3     # reviews run side by side; each also starts its checks and clips as child processes


def review_case(case) -> dict:
    """One setup reviewed, in a worker process: the setup's losses, or the error that stopped it."""
    name, root = case
    os.environ['ARGUS_READER_ISOLATION'] = '0'       # as conftest.py sets for a test in this process
    from label import harness
    harness.AUTO_RESEND_DELAYS_S = (0.0,) * len(harness.AUTO_RESEND_DELAYS_S)
    patch = pytest.MonkeyPatch()
    try:
        setup = globals()[name](root / 'source')
        model = OfflineModel()
        model.fault, model.tasks = setup.fault, setup.tasks
        rc = run_review(setup, root / 'job', model, patch)
        return {'rc': rc, 'calls': model.calls, 'losses': losses(setup, root / 'job'),
                'paths': sorted(paths_reached(root / 'job', model)), 'untruths': request_untruths(root / 'job'),
                'structure': structure_facts(root / 'job')}
    except BaseException:  # noqa: BLE001 - reported by the case's own test
        return {'error': traceback.format_exc()}
    finally:
        patch.undo()


def request_untruths(job: Path) -> list:
    """What each request of the review states about its recording that the recording does not establish
    (tests/test_request_hygiene.py untruths), over every prepared episode and every request the dry run built."""
    from test_request_hygiene import untruths
    out = []
    for dry in sorted((job / 'dry').glob('*.json')):
        record = _read(dry) or {}
        ep_dir = Path(record.get('episode_dir') or '')
        ctx = _read(ep_dir / 'context.json') if ep_dir.name else None
        if ctx is not None:
            out += [f'{dry.stem}: {x}' for x in untruths(ctx, record.get('prompt_text') or '')]
    return out


def structure_facts(job: Path) -> dict:
    """{episode: what its board record, and its export, say about the recording's verdict and its tasks}."""
    out = {}
    for path in sorted((job / 'qa').glob('episode_*.json')):
        board = _read(path) or {}
        exported = export_line(board, path.name)
        meta = board.get('_meta') or {}
        tasks = board.get('tasks') or []
        out[path.stem] = {
            'instruction': bool(meta.get('given_prompt')), 'steps': len(board.get('dataset_labels') or []),
            'schema': meta.get('label_schema'),
            'verdict': [(c.get('task_completed'), c.get('basis')) for c in (board.get('completion') or {},
                                                                             exported.get('completion') or {})],
            'judged': [[(t.get('judged_against'), t.get('subtask')) for t in rows]
                       for rows in (tasks, exported.get('tasks') or [])],
            'joined': [len(t.get('joined_from') or []) for t in tasks],
            'relation': (board.get('goal_alignment') or {}).get('relation')}
    return out


def paths_reached(job: Path, model: OfflineModel) -> set:
    """The paths of the pipeline a review went through, so the corpus can show it reaches every one it checks."""
    replies = [reply_of(_read(p) or {}) for p in (job / 'run' / 'out').glob('episode_*.json')]
    return ({'parts'} if any('__p' in p.stem for p in (job / 'run' / 'out').glob('episode_*.json')) else set()) \
        | ({'regional pass'} if 'regional' in model.calls else set()) \
        | ({'tasks'} if any(tagged_tasks(r) for r in replies) else set()) \
        | ({'goal alignment'} if any(alignment_tags(r) for r in replies) else set()) \
        | ({'sensor findings'} if any(finding_tags(r.get('sensor_findings')) for r in replies) else set()) \
        | ({'camera observations'} if any(r.get('camera_observations') for r in replies) else set()) \
        | {f'part without labels ({kind.split()[-1]})' for kind in model.calls if kind.startswith('final ')} \
        | ({'issue set aside at a cut'} if any(row.get('excluded_by') == 'piece_cut' and issue_tags({'data_issues': [row]})
                                               for p in (job / 'run_final' / 'out').glob('episode_*.json')
                                               for row in ((_read(p) or {}).get('labels') or {}).get('_excluded') or []
                                               if isinstance(row, dict)) else set())


@pytest.fixture(scope='module')
def reviewed(tmp_path_factory):
    import multiprocessing
    root = tmp_path_factory.mktemp('setups')
    # spawned, not forked: a forked macOS process aborts when it first imports pyarrow
    with multiprocessing.get_context('spawn').Pool(WORKERS) as pool:
        results = pool.map(review_case, [(name, root / name) for name in CASES], chunksize=1)
    return dict(zip(CASES, results))


@pytest.mark.parametrize('name', CASES)
def test_every_setup_keeps_every_stream_and_citation_through_to_the_export(name, reviewed):
    result = reviewed[name]
    assert 'error' not in result, result.get('error')
    assert 'final' in result['calls'], result['calls']
    found = result['losses']
    expected = {pattern: why for (setup, pattern), why in EXPECTED.items() if setup == name}
    unknown = [loss for loss in found if not any(re.search(p, loss) for p in expected)]
    stale = [p for p in expected if not any(re.search(p, loss) for loss in found)]
    assert not unknown, f'{name} lost data with no recorded reason (review exited {result["rc"]}):\n' + \
        '\n'.join(unknown)
    assert not stale, f'{name}: these expected losses no longer occur, so remove them from EXPECTED: {stale}'
    if found:
        pytest.xfail('; '.join(sorted(set(expected.values()))) + ': ' + ' | '.join(found))


@pytest.mark.parametrize('name', CASES)
def test_every_setups_request_states_only_what_its_recording_establishes(name, reviewed):
    """No request gives a value of a signal that shares no clock with the cameras at a frame instant, times a contact
    by one, counts a signal on a part that never grasps as a hand's, or names a contact's hand or side without saying
    what establishes it."""
    result = reviewed[name]
    assert 'error' not in result, result.get('error')
    assert result['untruths'] == [], '\n'.join(result['untruths'])


# What each instruction structure gives, on the board and in the export: (instruction, timed steps, the recording's
# verdict and its basis, what each task is judged against, how many parts each task joins)
STRUCTURES = {
    # global only: the completion is the reply's own, against the instruction; the unasked task never changes it
    'teleop_lerobot': (True, 0, ('success', 'instruction'), [('instruction', None), ('own_end_state', None)], [0, 0]),
    # per-task only: no verdict; each task against its step, the steps compared with the tasks
    'steps_mcap': (False, 5, (None, 'task_prompts'), [('subtask', 1), ('own_end_state', None)], [0, 0]),
    # both: the completion against the instruction, and a task against its step
    'both_mcap': (True, 2, ('success', 'instruction'), [('subtask', 1), ('own_end_state', None)], [0, 0]),
    # neither, two tasks: no verdict
    'mcap_arm': (False, 0, (None, 'tasks'), [('own_end_state', None), ('own_end_state', None)], [0, 0]),
    # neither, one task: its verdict is the recording's, in one request and across the cuts of a long one
    'one_task_mcap': (False, 0, ('success', 'one_task'), [('own_end_state', None)], [0]),
    'long_one_task': (False, 0, ('success', 'one_task'), [('own_end_state', None)], [2]),
}


@pytest.mark.parametrize('name', sorted(STRUCTURES))
def test_each_instruction_structure_reaches_the_board_and_the_export_with_its_verdict(name, reviewed):
    result = reviewed[name]
    assert 'error' not in result, result.get('error')
    (facts,) = result['structure'].values()
    instruction, steps, verdict, judged, joined = STRUCTURES[name]
    assert (facts['instruction'], facts['steps'], facts['schema']) == (instruction, steps, 'tasks'), facts
    assert facts['verdict'] == [verdict, verdict], facts
    assert facts['judged'] == [judged, judged], facts
    assert facts['joined'] == joined, facts
    # the instruction, or with none the timed steps, is compared with what was observed
    assert (facts['relation'] == 'aligned') is (instruction or bool(steps)), facts


def test_the_corpus_reaches_every_path_it_checks(reviewed):
    """A check that never runs proves nothing: some setup is cut into parts, gets the regional pass, writes tasks, a
    goal alignment, sensor findings and camera observations, has an issue set aside at a cut, and has a part that gave
    no labels for each kind of failure."""
    reached = {path for result in reviewed.values() for path in result.get('paths') or []}
    assert reached >= {'parts', 'regional pass', 'tasks', 'goal alignment', 'sensor findings',
                       'camera observations', 'issue set aside at a cut', 'part without labels (error)',
                       'part without labels (cutoff)', 'part without labels (rejected)'}, sorted(reached)


# ---- the check itself catches each kind of loss ----------------------------------------------------------------------

def test_one_stream_never_passes_for_two_streams_of_the_same_name():
    left, right = Item('signal', 'pressure', 'left_glove.h5'), Item('signal', 'pressure', 'right_glove.h5')
    files = {'left_glove.h5', 'right_glove.h5'}
    one = [Carrier('signal', 'pressure', ['pressure', 'HDF5 dataset pressure'])]
    held = resolve([left, right], one, files)
    assert sorted(len(held.get(id(i)) or []) for i in (left, right)) == [0, 1]
    both = [Carrier('signal', 'left_glove pressure', ['left_glove pressure', 'left_glove.h5']),
            Carrier('signal', 'right_glove pressure', ['right_glove pressure', 'right_glove.h5'])]
    held = resolve([left, right], both, files)
    assert held[id(left)] == ['left_glove pressure'] and held[id(right)] == ['right_glove pressure']


def test_a_reason_counts_only_where_the_pipeline_records_something_left_out():
    assert flagged_texts([['pressure (from right_glove.h5) was placed on the footage from both starts']]) == []
    assert flagged_texts([['pressure could not be read']]) == ['pressure could not be read']
    assert mentions('/imu/data linear_acceleration', '/imu/data') and not mentions('episode_000001', '000001')


def _record(findings=(), unbound=(), reply=()):
    return {'labels': {'sensor_findings': list(reply), 'evidence_findings': []},
            'sensor_evidence': {'sensors': [], 'findings': list(findings), 'unbound_findings': list(unbound)}}


def test_a_citation_dropped_at_binding_or_after_it_is_a_loss():
    cite = {'observation': 'offline sensor_findings 0', 'evidence': [{'sensor_id': 'signal:glove'}]}
    assert unit_losses('ep: ', _record(reply=[cite])) == [
        'bound: ep: offline sensor_findings 0 citing [{"sensor_id": "signal:glove"}] is neither bound nor set aside '
        'with a reason']
    aside = _record(unbound=[{'finding': cite, 'reason': 'no verified alignment'}], reply=[cite])
    assert unit_losses('ep: ', aside) == [] and carried_losses('ep: ', [aside], [('board', aside)]) == []
    bound = _record(findings=[cite], reply=[cite])
    assert carried_losses('ep: ', [bound, _record()], [('stitched', bound), ('export', _record())]) == [
        'export: ep: bound finding offline sensor_findings 0 is missing']
