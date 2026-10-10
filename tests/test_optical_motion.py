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
        assert sensor['quantity'] == 'tactile_image'
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


def test_normal_request_derives_motion_from_decoded_tactile_images(tmp_path):
    import av
    import json
    from label import episode as me

    folder = tmp_path / 'episode_dots'
    folder.mkdir()
    movie = folder / 'pad.mp4'
    with av.open(str(movie), 'w') as container:
        stream = container.add_stream('libx264', rate=30)
        stream.width, stream.height, stream.pix_fmt = 240, 160, 'yuv420p'
        for k in range(15):
            frame = av.VideoFrame.from_image(pad(2 if k >= 7 else 0))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    ctx = {'dataset': 'unfamiliar_pad', 'profile': 'ego_head', 'state_kind': 'none', 'fps': 30,
           'cameras': {'exo': {'name': 'tactile_camera', 'width': 240, 'height': 160}}}
    (folder / 'context.json').write_text(json.dumps(ctx))
    (folder / 'sources.json').write_text(json.dumps({'exo': {'packed': str(movie), 'base_s': 0, 'n_frames': 15}}))
    request = me.build_request(folder, cell_w=192)
    pairs = request['sensor_evidence']['sensors'][0]['marker_motion']['pairs']
    assert any(p.get('p95_px', 0) > 1 for p in pairs if p.get('p95_px') is not None)
    sent = request['sensor_evidence']['sensors'][0]['times']
    assert all(p['from_s'] in sent and p['to_s'] in sent for p in pairs)
