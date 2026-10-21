"""Every supported setup through the whole pipeline, offline, with nothing lost on the way.

Round after round of QA found data dropped silently: a camera or signal stream, a whole episode, findings written in
the other findings list, streams that share a name. Each was fixed where it was found. This test catches the class:
a tiny recording of every setup the README lists is generated here, never downloaded, and run through `python -m
review` (prepare, checks, clips, the request build, a reply, binding, the board build) and the JSON Lines export. The
model is offline (OfflineModel): its reply cites every camera, signal and inspected evidence id its request offered,
in every list the schema allows.

Each fixture lists what went in (its inventory: recordings, cameras, signals, recorded fields). Every item must come
out at every stage, the prepared episode, the request's evidence, the bound annotations, the board's episode record
and the export, or be named in a reason the pipeline recorded (a reader issue, a left-out note, a limitation, a
set-aside episode). Every citation the reply made must be bound or named unbound with its reason, and every bound
finding must reach the board and the export. The checks are the same for every setup: nothing here knows what a
setup is expected to lose, only what its fixture put in."""
from __future__ import annotations

import io
import json
import re
import sys
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
    same names, with different readings."""
    import h5py
    from test_formats import _clip, _glove_h5
    data = root / 'hdf5_gloves'
    data.mkdir(parents=True)
    _clip(data / 'scene.mp4', 90)
    for side in ('left', 'right'):
        _glove_h5(data / f'{side}_glove.h5', 90)
        with h5py.File(data / f'{side}_glove.h5', 'a') as h:
            h.attrs[f'{side}_glove_serial'] = f'GLV-{side}-0042'
            if side == 'right':
                h['pressure'][...] = h['pressure'][()][::-1]     # the other hand presses at other times
    return Setup('hdf5_gloves', 'teleop_arms', data, [
        Item('recording', 'scene'), Item('camera', 'scene'),
        Item('signal', 'pressure', 'left_glove.h5'), Item('signal', 'pressure', 'right_glove.h5'),
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


SETUPS = [teleop_lerobot, mcap_arm, umi_gripper, ego_head, hdf5_gloves, depth_mcap, tactile_lerobot,
          plain_video, zipped_mcap]


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
    reply is kept in the run record, so what it cited is read back from there (citation_losses)."""

    def __init__(self):
        self.calls = []
        self.selected = {}       # an inventory's field ids -> the ids already requested from it

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
                {**f, 'observation': f'offline regional {i}'} for i, f in enumerate(self.sensor_findings(texts))]}
        else:
            kind, reply = 'final', self.final(texts)
        self.calls.append(kind)
        payload = {'id': f'gen-offline-{len(self.calls)}', 'model': body.get('model'),
                   'usage': {'cost': 0.001, 'prompt_tokens': 10, 'completion_tokens': 10},
                   'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(reply)}}]}
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
        for row in series:
            if row.get('sensor_id') != ident or row.get('derivation_id'):
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
        cameras = []
        for caption in re.findall(r'=== grid t=[^|]*\| rows ([^|]*) \|', whole):
            cameras += [c.strip() for c in caption.split(',') if c.strip() and c.strip() not in cameras]
        observed = re.search(r'The supplied view IDs and image-row labels are: (.*?)\. Include', whole)
        views = [pair.split(':', 1)[0].strip() for pair in observed.group(1).split(';')] if observed else []
        cited = self.sensor_findings(texts) + self.evidence_findings(texts)
        reply = {'task_summary': 'Move the object.', 'timeline_columns': [], 'timeline': [],
                 'data_issues': [{'issue': f'offline data_issues {i}: camera {c} is dim', 'category': 'camera_fault',
                                  'severity': 'low', 't_s': 0.0, 'end_s': None, 'evidence': f'camera {c} at 0.0s'}
                                 for i, c in enumerate(cameras)]}
        for name in FINDING_LISTS:
            reply[name] = [{**json.loads(json.dumps(f)), 'observation': f'offline {name} {i}'}
                           for i, f in enumerate(cited)]
        if views:
            reply['camera_observations'] = [{'view': v, 'content': 'scene', 'evidence': f'offline view {v}'}
                                            for v in views]
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
    it; when items share a name, a stream holds one of them only by also naming that item's file, or else it holds just
    one of them. So one stream never passes for two streams of the same name."""
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
            if len(owners) == 1:
                held.setdefault(id(owners[0]), []).append(c.ident)
            elif not named:
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
    rows = list(d.get('data_issues') or []) + list((d.get('_excluded') or {}).get('data_issues') or [])
    return {r['issue'].split(':', 1)[0] for r in rows if isinstance(r, dict) and isinstance(r.get('issue'), str)
            and r['issue'].startswith('offline ')}


def observation_views(d: dict) -> tuple:
    co = d.get('camera_observations') if isinstance(d.get('camera_observations'), dict) else {}
    kept = {o.get('view') for o in co.get('observations') or [] if isinstance(o, dict)}
    withheld = [o for o in co.get('withheld_observations') or []
                if isinstance(o, dict) and isinstance(o.get('why'), str) and o['why'].strip()]
    aside = {(o.get('observation') or {}).get('view') for o in withheld} | {v for o in withheld for v in o.get('views') or []}
    return kept, aside


def reply_of(record: dict) -> dict:
    labels = record.get('labels')
    return labels if isinstance(labels, dict) and not labels.get('_parse_error') else {}


def citation_losses(tag: str, record: dict, board: dict, exported: dict) -> list:
    """Every citation the reply made is bound or set aside with its reason, then reaches the board and the export."""
    lost = []
    reply = reply_of(record)
    if not reply:
        return [f'bound: {tag}the reply did not parse: {(record.get("labels") or {}).get("_parse_error")}']
    cited = {t: f.get('evidence') for name in FINDING_LISTS for f in reply.get(name) or [] if isinstance(f, dict)
             for t in finding_tags([f])}
    analysis = record.get('sensor_analysis') if isinstance(record.get('sensor_analysis'), dict) else {}
    try:
        regional = json.loads(analysis.get('response_text') or '{}').get('sensor_findings')
    except (ValueError, AttributeError):
        regional = None
    cited.update({t: f.get('evidence') for f in regional or [] if isinstance(f, dict) for t in finding_tags([f])})
    bound, aside = placed_findings(record)
    for t, refs in cited.items():
        if t not in bound and t not in aside:
            lost.append(f'bound: {tag}{t} citing {json.dumps(refs)[:160]} is neither bound nor set aside with a reason')
    for stage, doc in (('board', board), ('export', exported)):
        got, got_aside = placed_findings(doc)
        lost += [f'{stage}: {tag}bound finding {t} is missing' for t in sorted(bound - got)]
        lost += [f'{stage}: {tag}{t}, set aside at binding, is missing with its reason'
                 for t in sorted(set(aside) - set(got_aside) - got)]
    issues = {r['issue'].split(':', 1)[0] for r in reply.get('data_issues') or []
              if isinstance(r, dict) and isinstance(r.get('issue'), str) and r['issue'].startswith('offline ')}
    for stage, doc in (('bound', reply), ('board', board), ('export', exported)):
        lost += [f'{stage}: {tag}{t} (a data issue naming a camera) is missing' for t in sorted(issues - issue_tags(doc))]
    views = {r.get('view') for r in reply.get('camera_observations') or [] if isinstance(r, dict)}
    for stage, doc in (('bound', record), ('board', board), ('export', exported)):
        kept, withheld = observation_views(doc)
        lost += [f'{stage}: {tag}the camera observation of {v} is neither kept nor withheld with a reason'
                 for v in sorted(views - kept - withheld, key=str)]
    # the reply cites every sensor the request offered, so a sensor no finding cites is one binding never saw
    refs = {r.get('sensor_id') or r.get('evidence_id') for f in cited.values() if isinstance(f, list)
            for r in f if isinstance(r, dict)}
    for s in (record.get('sensor_evidence') or {}).get('sensors') or []:
        if isinstance(s, dict) and s.get('id') not in refs and not str(s.get('id')).startswith('mechanics:'):
            lost.append(f'bound: {tag}offered sensor {s.get("id")} ({s.get("name")}) was never cited')
    return lost


# ---- the inventory check ------------------------------------------------------------------------------------------------------

def losses(setup: Setup, job: Path) -> list:
    """Every item of the fixture's inventory, and every citation the reply made, that a stage holds neither itself nor
    in a recorded reason. Empty means nothing was lost."""
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
        record = _read(job / 'run_final' / 'out' / f'{eid}.json')
        board = _read(job / 'qa' / f'{eid}.json')
        missing = [stage for stage, doc in (('prepared', ctx), ('bound', record), ('board', board)) if doc is None]
        if missing:
            if not explained(report_reasons, eid):
                lost += [f'{stage}: {tag}the episode is missing with no reason' for stage in missing]
            continue
        exported = export_line(board, f'{eid}.json')
        dry = _read(job / 'dry' / f'{eid}.json') or {}
        evidence = record.get('sensor_evidence') or {}
        inspection = record.get('evidence_inspection') or {}
        source = ctx.get('source') if isinstance(ctx.get('source'), dict) else {}
        reasons = {'prepared': report_reasons + left_out_texts([v for k, v in source.items() if k.startswith('unused_')])
                   + flagged_texts([ctx.get('reader_issues')]),
                   'request': flagged_texts([evidence.get('limitations'), inspection.get('limitations')]),
                   'board': left_out_texts([(board.get('reader_notes') or {}).get('left_out'), set_aside])
                   + flagged_texts([(board.get('dataset_checks') or {}).get('reader_issues')])}
        reasons['export'] = reasons['board']
        carriers = {'request': request_carriers(record, dry.get('prompt_text') or ''),
                    'board': board_carriers(job, eid, board), 'export': record_carriers(exported)}
        held = resolve(streams, prepared_carriers(ep, ctx), files)
        for item in streams:
            # a recorded field is metadata text, so it is found by its own name at every stage
            idents = [item.name] if item.kind == 'field' and held.get(id(item)) else held.get(id(item)) or []
            if not idents:
                if not explained(reasons['prepared'], item.name, item.file):
                    lost.append(f'prepared: {tag}{item} is not in the prepared episode and no reason names it')
                continue
            for stage in ('request', 'board', 'export'):
                for ident in idents:
                    if not (present(carriers[stage], item.kind, ident) or explained(reasons[stage], ident)
                            or explained(reasons[stage], item.name)):
                        lost.append(f'{stage}: {tag}{item}, prepared as {ident!r}, is missing and no reason names it')
        lost += citation_losses(tag, record, board, exported)
    return lost


# ---- the test -------------------------------------------------------------------------------------------------------------

# Losses that open pull requests fix, by (setup, pattern of the loss): the reason names the pull request. Each must still
# occur, so when its pull request has merged and the loss is gone the test fails until its entry is removed.
PR75_ROUTING = ('PR 75 (eric/data-loss-four) sends a finding written in the other findings list to the binder its '
                'citations name; before it, a request with no supplied sensor evidence never binds sensor_findings')
PR75_SAME_NAME = ('PR 75 (eric/data-loss-four) keeps signals of the same name from different files; before it, the '
                  'file read last replaces the other')
EXPECTED = {
    **{(name, r'^bound: .*offline sensor_findings \d+ citing \[\{"evidence_id"'): PR75_ROUTING
       for name in ('teleop_lerobot', 'mcap_arm', 'umi_gripper', 'ego_head', 'depth_mcap', 'zipped_mcap')},
    ('hdf5_gloves', r'^prepared: .*signal pressure \((left|right)_glove\.h5\) is not in the prepared episode'):
        PR75_SAME_NAME,
}


@pytest.mark.parametrize('build', SETUPS, ids=[f.__name__ for f in SETUPS])
def test_every_setup_keeps_every_stream_and_citation_through_to_the_export(build, tmp_path, monkeypatch):
    setup = build(tmp_path / 'source')
    model = OfflineModel()
    rc = run_review(setup, tmp_path / 'job', model, monkeypatch)
    assert 'final' in model.calls, model.calls
    found = losses(setup, tmp_path / 'job')
    expected = {pattern: why for (name, pattern), why in EXPECTED.items() if name == setup.name}
    unknown = [loss for loss in found if not any(re.search(p, loss) for p in expected)]
    stale = [p for p in expected if not any(re.search(p, loss) for loss in found)]
    assert not unknown, f'{setup.name} lost data with no recorded reason (review exited {rc}):\n' + '\n'.join(unknown)
    assert not stale, f'{setup.name}: these expected losses no longer occur, so remove them from EXPECTED: {stale}'
    if found:
        pytest.xfail('; '.join(sorted(set(expected.values()))) + ': ' + ' | '.join(found))


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
    dropped = citation_losses('ep: ', _record(reply=[cite]), _record(), _record())
    assert dropped == ['bound: ep: offline sensor_findings 0 citing [{"sensor_id": "signal:glove"}] is neither bound '
                       'nor set aside with a reason']
    aside = _record(unbound=[{'finding': cite, 'reason': 'no verified alignment'}], reply=[cite])
    assert citation_losses('ep: ', aside, aside, aside) == []
    bound = _record(findings=[cite], reply=[cite])
    assert citation_losses('ep: ', bound, bound, _record()) == ['export: ep: bound finding offline sensor_findings 0 '
                                                                'is missing']
