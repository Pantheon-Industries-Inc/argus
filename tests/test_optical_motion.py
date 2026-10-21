"""Protect optical evidence against erased motion, image artifacts and stale clocks."""
import copy

import numpy as np
import pytest
from PIL import Image, ImageDraw

from label import sensor_evidence as se
from tests.test_tactile_contract import episode


def pad(dx=0, local=False):
    im = Image.new('RGB', (480, 320), (150, 180, 200))
    draw = ImageDraw.Draw(im)
    for y in range(40, 300, 40):
        for x in range(40, 460, 40):
            shift = 2 * dx if not local or y < 160 else 0
            draw.ellipse((x+shift-5, y-5, x+shift+5, y+5), fill=(30, 40, 50))
    return im.resize((240, 160), Image.Resampling.LANCZOS)


def evidence(images, shown=(0, 4), entry=None):
    ep = episode({}, [], cameras={'exo': {'name': 'scene'},
                                  'extra1': {'name': 'unfamiliar_tactile_pad'}})
    if entry:
        ep['context']['data_dictionary'] = {
            'episode_id': 'episode_test',
            'fields': [{'id': 'pad', 'kind': 'camera', 'name': 'unfamiliar_tactile_pad',
                        'bindings': [{'episode': 'episode_test', 'context_path': 'cameras/extra1'}]}],
            'entries': {'pad': entry}}
    doc = se.build(ep, {'n': 5, 'ks': [0, 2, 4]},
                   shown={'extra1': list(shown)}, images={'extra1': images})
    return ep, doc


def test_coherent_and_local_marker_motion_survive_without_force_claims():
    for local in (False, True):
        ep, doc = evidence({0: pad(), 4: pad(2, local)})
        sensor = doc['sensors'][0]
        assert 'marker_motion' in sensor['capabilities']
        motion = sensor['marker_motion']
        pair = motion['pairs'][0]
        assert pair['from_s'] == 0 and pair['to_s'] == .4
        assert pair['p95_px'] == pytest.approx(2, abs=.25)
        assert pair['mean_dx_px'] > .6
        assert pair['matched'] >= 30
        if not local:
            assert pair['local_p95_px'] < .2
        assert 'force_change' not in sensor['capabilities']
        # a fixed marker-pad view is an optical sensor's own camera (label/camera_class.py)
        assert sensor['quantity'] == 'optical_tactile'
        assert se.compatible(doc, ep['context'])


def test_brightness_changes_do_not_become_marker_movement_and_blank_images_decline():
    im = pad()
    bright = Image.fromarray(np.clip(np.asarray(im, dtype=float)*.8+20, 0, 255).astype('uint8'))
    _, doc = evidence({0: im, 4: bright})
    assert doc['sensors'][0]['marker_motion']['pairs'][0]['p95_px'] < .15
    _, blank = evidence({0: Image.new('RGB', (240, 160), 'blue'), 4: im})
    assert 'marker_motion' not in blank['sensors'][0]['capabilities']
    assert 'marker_motion' not in blank['sensors'][0]


def test_unshown_frames_and_human_exclusions_cannot_supply_motion():
    _, doc = evidence({0: pad(), 2: pad(2), 4: pad()}, shown=(0, 4))
    assert [p['to_s'] for p in doc['sensors'][0]['marker_motion']['pairs']] == [.4]
    assert doc['sensors'][0]['marker_motion']['pairs'][0]['p95_px'] < .01
    _, excluded = evidence({0: pad(), 4: pad(2)}, entry={'role': 'scene', 'provenance': 'human'})
    assert not excluded['sensors']
    _, heatmap = evidence({0: pad(), 4: pad(2)}, entry={'role': 'tactile_image',
                         'meaning': 'Rendered pressure heatmap', 'provenance': 'human'})
    assert 'marker_motion' not in heatmap['sensors'][0]['capabilities']


def test_piece_merge_shifts_motion_pairs_and_keeps_quality_gaps_unusable():
    ep, doc = evidence({0: pad(), 4: pad(2)})
    finding = {'start_s': 0, 'end_s': .4, 'headline': 'The object shifts against the grip',
               'claim_type': 'marker_motion', 'adds_beyond_video': True,
               'evidence': [{'sensor_id': 'camera:extra1', 'time_s': [0, .4]}]}
    bound = se.bind(doc, [finding])
    merged = se.merge([({'piece': {'index': 0, 't0_s': 20}}, {'sensor_evidence': bound})], ep['context'])
    motion = merged['sensors'][0]['marker_motion']
    assert motion['reference_s'] == 20
    assert motion['samples'][-1]['time_s'] == 20.4
    assert motion['pairs'][0]['from_s'] == 20 and motion['pairs'][0]['to_s'] == 20.4
    assert merged['findings'][0]['quoted_time_offset_s'] == 20
    missing = copy.deepcopy(doc)
    missing['sensors'][0]['marker_motion']['pairs'][0]['p95_px'] = None
    assert not se.bind(missing, [finding])['findings']


@pytest.mark.parametrize("cell_w", [192, 320])
def test_normal_request_derives_motion_from_decoded_tactile_images(tmp_path, cell_w, monkeypatch):
    import av
    import json
    from label import episode as me
    from label import optical_motion as om
    from label.frames import Shrunk

    folder = tmp_path / 'episode_dots'
    folder.mkdir()
    movie = folder / 'pad.mp4'
    with av.open(str(movie), 'w') as container:
        stream = container.add_stream('libx264', rate=30)
        stream.width, stream.height, stream.pix_fmt = 720, 480, 'yuv420p'
        for k in range(120):
            frame = av.VideoFrame.from_image(pad(2 if 30 <= k < 90 else 0).resize((720, 480)))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    ctx = {'dataset': 'unfamiliar_pad', 'profile': 'ego_head', 'state_kind': 'none', 'fps': 30,
           'cameras': {'exo': {'name': 'tactile_camera', 'width': 720, 'height': 480}}}
    (folder / 'context.json').write_text(json.dumps(ctx))
    (folder / 'sources.json').write_text(json.dumps({'exo': {'packed': str(movie), 'base_s': 0, 'n_frames': 120}}))
    observed = {}
    measure = om.measure
    def capture(frames, times, **kwargs):
        observed.update(frames=frames, times=times)
        return measure(frames, times, **kwargs)
    monkeypatch.setattr(om, 'measure', capture)
    request = me.build_request(folder, cell_w=cell_w, sample_fps=1)
    motion = request['sensor_evidence']['sensors'][0]['marker_motion']
    assert motion['analysis_size'] == [cell_w, int(cell_w * 2 / 3 / 2 + .5) * 2]
    pairs = motion['pairs']
    assert any(p.get('p95_px', 0) > 1 for p in pairs if p.get('p95_px') is not None)
    sent = request['sensor_evidence']['sensors'][0]['times']
    assert all(p['from_s'] in sent and p['to_s'] in sent for p in pairs)
    assert 'common analysis_size' in motion['coordinates']
    ks = sorted(observed['frames'])
    assert isinstance(observed['frames'][ks[0]], Image.Image)
    assert isinstance(observed['frames'][ks[1]], Shrunk)
    assert isinstance(observed['frames'][ks[-1]], Image.Image)
    # A common analysis size must not conceal actual capture-geometry changes.
    first = observed['frames'][ks[0]]
    for changed_size in ((720, 400), (360, 240)):
        changed = Shrunk(first.resize(changed_size), [cell_w])
        assert measure({0: first, 1: changed, 2: first}, {0: 0., 1: 1., 2: 2.}) is None


@pytest.mark.parametrize("module_name", ["optical_motion", "sensor_evidence"])
def test_optical_algorithm_changes_invalidate_receipts_without_changing_rgb_payload(tmp_path, monkeypatch, module_name):
    import importlib
    import json
    from pathlib import Path
    from label import episode as me, evidence_access as ea, harness, optical_motion as om
    from tests.test_label import _packed_episode

    folder, _ = _packed_episode(tmp_path)
    (folder / 'state.npz').unlink()
    path = folder / 'context.json'
    ctx = json.loads(path.read_text())
    ctx['state_kind'] = 'none'
    ctx.pop('n_state_frames')
    path.write_text(json.dumps(ctx))
    kwargs = dict(model='openai/gpt-6.1-sol', reasoning='high', max_tokens=2000,
                  cell_w=128, example_dir=None)
    before = harness.input_identity(folder, **kwargs)
    request = me.build_request(folder, cell_w=128)
    assert not request['sensor_evidence']['sensors']
    access = ea.Access(me.load(folder)).signature()
    implementation = ea.IMPLEMENTATION_SHA256
    read_bytes = Path.read_bytes
    algorithm = Path(om.__file__).with_name(module_name + ".py").resolve()
    def changed_bytes(path):
        data = read_bytes(path)
        return data + b'\n# changed optical algorithm\n' if path.resolve() == algorithm else data
    try:
        with monkeypatch.context() as patch:
            patch.setattr(Path, 'read_bytes', changed_bytes)
            importlib.reload(ea)
            assert not harness.same_input_identity(before, harness.input_identity(folder, **kwargs))
            assert ea.IMPLEMENTATION_SHA256 != implementation
            assert ea.Access(me.load(folder)).signature() != access
            assert me.build_request(folder, cell_w=128)['content'] == request['content']
    finally:
        importlib.reload(ea)


def test_marker_comparison_can_use_other_supplied_pad_images_as_context():
    _, doc = evidence({0: pad(), 4: pad(2)})
    context = copy.deepcopy(doc['sensors'][0])
    context['id'] = 'camera:extra2'
    context['marker_motion']['pairs'][0]['p95_px'] = None
    doc['sensors'].append(context)
    finding = {'start_s': 0, 'end_s': .4, 'headline': 'The pads respond differently during the grasp',
               'claim_type': 'marker_motion', 'adds_beyond_video': True,
               'evidence': [{'sensor_id': 'camera:extra1', 'time_s': [0, .4]},
                            {'sensor_id': 'camera:extra2', 'time_s': [0, .4]}]}
    assert se.bind(doc, [finding])['findings'] == [finding]
    missing = copy.deepcopy(doc)
    missing['sensors'][0]['marker_motion']['pairs'][0]['p95_px'] = None
    assert not se.bind(missing, [finding])['findings']
    for change in ('unshown', 'unaligned'):
        invalid = copy.deepcopy(doc)
        if change == 'unshown':
            invalid['sensors'][1]['times'] = [0]
        else:
            invalid['sensors'][1]['episode_alignment_verified'] = False
        assert not se.bind(invalid, [finding])['findings']


def test_prompt_budget_preserves_small_optical_comparisons_and_quality_limits(monkeypatch):
    import json

    _, doc = evidence({0: pad(), 4: pad(2)})
    template = doc['sensors'][0]
    doc['sensors'] = []
    doc['episode_window_s'] = [0., 64.]
    for channel in range(4):
        sensor = copy.deepcopy(template)
        sensor['id'] = f'camera:extra{channel + 1}'
        sensor['times'] = list(range(64))
        motion = sensor['marker_motion']
        pair = motion['pairs'][0]
        sample = motion['samples'][-1]
        # A full-episode request includes both large transitions and smaller
        # changes during a hold. Magnitude alone does not rank their usefulness.
        motion['pairs'] = [{**copy.deepcopy(pair), 'from_s': i, 'to_s': i+1,
                            'p95_px': .4 if i == 35 else 3.,
                            'local_p95_px': .2 if i == 35 else 1.5,
                            'correspondence': {'global_translation_px': [0., 0.],
                                             'alternative_peak_ratio': .8,
                                             'coherent_local_field': False, 'ambiguous': False}}
                           for i in range(63)]
        motion['samples'] = [{**copy.deepcopy(sample), 'time_s': t,
                              'p95_px': None if t == 35 else 2.,
                              'correspondence': {'global_translation_px': [0., 0.],
                                               'alternative_peak_ratio': 1. if t == 35 else .8,
                                               'coherent_local_field': True, 'ambiguous': t == 35}}
                             for t in sensor['times']]
        motion['comparisons_total'] = 64
        doc['sensors'].append(sensor)
    original = copy.deepcopy(doc)
    finding = {'start_s': 35, 'end_s': 36, 'headline': 'Contact changes during the hold',
               'claim_type': 'marker_motion', 'adds_beyond_video': True,
               'evidence': [{'sensor_id': 'camera:extra1', 'time_s': [35, 36]}]}
    assert se.bind(doc, [finding])['findings'] == [finding]
    sent = se.prompt(doc)
    assert len(sent.encode()) <= se.MAX_PROMPT_BYTES
    projected = json.loads(sent.split('\n')[-1])
    for before, after in zip(original['sensors'], projected['sensors']):
        for key in ('pairs', 'samples'):
            table = after['marker_motion'][key]
            restored = [{k: v for k, v in zip(table['columns'], row) if v is not None}
                        for row in table['rows']]
            expected = [{**{k: v for k, v in row.items() if v is not None and k != 'correspondence'},
                         **{'correspondence.' + k: v for k, v in row.get('correspondence', {}).items()}}
                        for row in before['marker_motion'][key]]
            assert restored == expected
    assert doc == original
    # Extra native detail cannot evict the full sequence of scene-anchor comparisons.
    monkeypatch.setattr(se, 'MAX_PROMPT_BYTES', len(sent.encode()) + 512)
    for sensor in doc['sensors']:
        motion = sensor['marker_motion']
        for i in range(4):
            motion['pairs'].append({**copy.deepcopy(motion['pairs'][0]), 'from_s': i + .1,
                                    'to_s': i + .2, 'selection': 'native_detail'})
    projected = json.loads(se.prompt(doc).split('\n')[-1])
    for sensor in projected['sensors']:
        table = sensor['marker_motion']['pairs']
        rows = [dict(zip(table['columns'], row)) for row in table['rows']]
        assert {(row['from_s'], row['to_s']) for row in rows if row.get('selection') != 'native_detail'} == {
            (i, i + 1) for i in range(63)}
    # A reference-relative quality gap cannot erase a valid adjacent pair, but
    # that pair must not become usable when its own correspondence fails.
    doc['sensors'][0]['marker_motion']['pairs'][35]['p95_px'] = None
    assert not se.bind(doc, [finding])['findings']
