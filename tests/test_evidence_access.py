"""Unknown measurements remain inspectable, with exact evidence and bounded model work."""
import json
from pathlib import Path

import numpy as np
import pytest

from label import evidence_access as ea


def access(tmp_path, **extra):
    ctx = {'episode_id': 'episode_a', 'fps': 10, 'state_kind': 'none', **extra}
    return ea.Access({'dir': tmp_path, 'context': ctx, 'sources': {}, 'state': np.zeros((4, 0)),
                      'action': None, 'times': None, 'signals': {}, 'signal_meta': {}, 'depth': {}})


def test_saved_findings_follow_stored_values_and_external_media(tmp_path):
    import copy
    from board import build
    ep_dir = tmp_path / 'episodes' / 'episode_a'
    ep_dir.mkdir(parents=True)
    media = tmp_path / 'private_media'
    media.mkdir()
    rgb, depth = media / 'rgb.bin', media / 'depth.bin'
    rgb.write_bytes(b'encoded rgb pixels')
    depth.write_bytes(b'encoded depth pixels')
    ctx = {'episode_id': 'episode_a', 'fps': 10, 'state_kind': 'joints',
           'cameras': {'exo': {'name': 'scene'}},
           'depth': {'exo': {'kind': 'depth', 'scale_m': .001}},
           'signals': [{'name': 'pressure', 'key': 'p', 'shape': [1], 'unit': 'N'}]}
    (ep_dir / 'context.json').write_text(json.dumps(ctx))
    (ep_dir / 'sources.json').write_text(json.dumps({'exo': {'packed': str(rgb), 'n_frames': 4}}))
    (ep_dir / 'depth.json').write_text(json.dumps({'exo': {'packed': str(depth), 'kmap': 'depth_map.npy'}}))
    np.save(ep_dir / 'depth_map.npy', np.arange(4))
    np.savez(ep_dir / 'state.npz', state=np.arange(4).reshape(4, 1), action=np.arange(4).reshape(4, 1))
    np.savez(ep_dir / 'signals.npz', p=np.arange(4).reshape(4, 1))
    ep = {'dir': ep_dir, 'context': ctx, 'sources': {'exo': {'packed': str(rgb), 'n_frames': 4}},
          'state': np.arange(4).reshape(4, 1), 'action': np.arange(4).reshape(4, 1),
          'times': None, 'kmap': {}, 'signals': {'pressure': np.arange(4).reshape(4, 1)},
          'signal_meta': {'pressure': {'unit': 'N'}}, 'depth': {}}
    a = ea.Access(ep, plan={'state_usable': True})
    field = next(f for f in a.inventory() if f['name'] == 'pressure')
    receipt = a.inspect({'field_id': field['id']})
    timed = {'headline': 'Pressure rises', 'start_s': 0, 'end_s': .3,
             'adds_beyond_video': True, 'evidence': [{'evidence_id': receipt['id'], 'time_s': [0, .3]}]}
    unaligned = ea.Access(ep, plan={'state_usable': False})
    row_receipt = unaligned.inspect({'field_id': field['id'], 'start_row': 0, 'end_row': 4})
    untimed = {'headline': 'Stored pressure has four samples', 'start_s': None, 'end_s': None,
               'adds_beyond_video': True, 'evidence': [{'evidence_id': row_receipt['id'], 'rows': [0, 3]}]}
    evidence = a.bind([timed])
    inspection = {**unaligned.record(), 'untimed_findings': unaligned.bind([untimed])['untimed_findings']}
    original = {'sensor_evidence': evidence, 'evidence_inspection': inspection}
    assert len(evidence['findings']) == 1 and inspection['untimed_findings'] == [untimed]
    assert str(media) not in json.dumps(original)
    for path, update in ((ep_dir / 'signals.npz', lambda p: np.savez(p, p=np.arange(4, 8).reshape(4, 1))),
                         (ep_dir / 'state.npz', lambda p: np.savez(p, state=np.arange(4, 8).reshape(4, 1),
                                                                action=np.arange(4, 8).reshape(4, 1))),
                         (rgb, lambda p: p.write_bytes(b'encoded rgb changed')),
                         (depth, lambda p: p.write_bytes(b'encoded depth changed')),
                         (ep_dir / 'depth_map.npy', lambda p: np.save(p, np.arange(4)[::-1])),
                         (rgb, lambda p: p.unlink())):
        saved = copy.deepcopy(original)
        update(path)
        build.add_context(saved, ctx, ep_dir)
        assert saved['sensor_evidence']['findings'] == []
        assert saved['sensor_evidence']['withheld_findings'] == evidence['findings']
        assert saved['evidence_inspection']['untimed_findings'] == []
        assert saved['evidence_inspection']['withheld_untimed_findings'] == [untimed]
    legacy = copy.deepcopy(original)
    legacy['sensor_evidence'].pop('source_proof')
    legacy['evidence_inspection'].pop('source_proof')
    build.add_context(legacy, ctx, ep_dir)
    assert not legacy['sensor_evidence']['findings']
    assert not legacy['evidence_inspection']['untimed_findings']


def test_production_request_keeps_mismatched_state_rows_untimed(tmp_path):
    from tests.test_label import _packed_mp4
    from label import episode
    ep_dir = tmp_path / 'episode_mismatch'
    ep_dir.mkdir()
    movie = ep_dir / 'scene.mp4'
    _packed_mp4(movie, 4)
    ctx = {'dataset': 'unfamiliar_sensor', 'profile': 'teleop_arms', 'state_kind': 'joints',
           'fps': 30, 'cameras': {'exo': {'name': 'scene', 'width': 64, 'height': 36}},
           'signals': [{'name': 'pressure', 'key': 'p', 'shape': [1]}]}
    (ep_dir / 'context.json').write_text(json.dumps(ctx))
    (ep_dir / 'sources.json').write_text(json.dumps({'exo': {'packed': str(movie), 'base_s': 0, 'n_frames': 4}}))
    np.savez(ep_dir / 'state.npz', state=np.zeros((5, 1)), action=np.zeros((5, 1)))
    np.savez(ep_dir / 'signals.npz', p=np.arange(5).reshape(5, 1))
    req = episode.build_request(ep_dir, inspect_evidence=True)
    assert not req['plan']['state_usable']
    access = req['evidence_access']
    state_field = next(f for f in access.inventory() if f['name'] == 'state')
    receipt = access.inspect({'field_id': state_field['id'], 'start_row': 0, 'end_row': 5})
    assert 'times_s' not in receipt and receipt['rows'] == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match='alignment'):
        access.inspect({'field_id': state_field['id'], 'start_s': 0, 'end_s': .1})


def test_parent_rebuild_checks_each_piece_untimed_source(tmp_path, monkeypatch):
    import copy
    from board import build
    from label import episode, pieces
    from prepare import formats
    from test_formats import recorder_folder
    monkeypatch.setattr(pieces, 'piece_max', lambda ctx: 1.5)
    upload = tmp_path / 'upload'
    recorder_folder(upload, n=90, duplicate_stamps=False)
    prepared = tmp_path / 'job' / 'episodes'
    report = formats.convert(upload, 'teleop_arms', prepared, 'test', 900)
    parent = prepared / report['episodes'][0]['episode_id']
    pieces.write_units(tmp_path / 'job', prepared)
    part_dirs = sorted((tmp_path / 'job' / 'pieces').glob('episode_*__p*'))
    assert len(part_dirs) == 2
    records = []
    for index, directory in enumerate(part_dirs):
        pc = json.loads((directory / 'context.json').read_text())
        pc['recorded_metadata'] = {'piece_fact': index}
        (directory / 'context.json').write_text(json.dumps(pc))
        ep = episode.load(directory)
        a = ea.Access(ep, plan=episode.plan(ep))
        field = next(f for f in a.inventory() if f['name'] == 'recorded_metadata')
        receipt = a.inspect({'field_id': field['id'], 'mode': 'metadata', 'pointer': '/piece_fact'})
        finding = {'headline': f'Piece {index + 1} metadata', 'start_s': None, 'end_s': None,
                   'adds_beyond_video': True, 'evidence': [{'evidence_id': receipt['id'], 'pointer': '/piece_fact'}]}
        record = {**a.record(), 'untimed_findings': a.bind([finding])['untimed_findings']}
        assert record['untimed_findings'] == [finding]
        assert ea.piece_reference(parent, directory, pc['piece']) is not None, (parent, directory, pc['piece'])
        records.append((pc, {'episode_dir': str(directory), 'evidence_inspection': record}))
    merged = ea.merge(records, parent)
    assert [p.get('source_location') for p in merged['parts']] == [
        f'pieces/{p.name}' for p in part_dirs]
    ctx = json.loads((parent / 'context.json').read_text())
    saved = {'evidence_inspection': copy.deepcopy(merged)}
    build.add_context(saved, ctx, parent)
    assert [part['record']['untimed_findings'] for part in saved['evidence_inspection']['parts']] == [
        records[0][1]['evidence_inspection']['untimed_findings'],
        records[1][1]['evidence_inspection']['untimed_findings']]
    pc = json.loads((part_dirs[1] / 'context.json').read_text())
    pc['recorded_metadata']['piece_fact'] = 99
    (part_dirs[1] / 'context.json').write_text(json.dumps(pc))
    stale = {'evidence_inspection': copy.deepcopy(merged)}
    build.add_context(stale, ctx, parent)
    first, second = [part['record'] for part in stale['evidence_inspection']['parts']]
    assert first['untimed_findings']
    assert not second['untimed_findings'] and second['withheld_untimed_findings']
    (part_dirs[1] / 'context.json').write_text(json.dumps(records[1][0]))
    with np.load(parent / 'state.npz') as stored:
        changed = {key: stored[key] for key in stored.files}
    changed['state'] = changed['state'].copy()
    changed['state'].flat[0] += 1
    np.savez(parent / 'state.npz', **changed)
    parent_stale = {'evidence_inspection': copy.deepcopy(merged)}
    build.add_context(parent_stale, ctx, parent)
    for part in parent_stale['evidence_inspection']['parts']:
        assert not part['record']['untimed_findings']
        assert part['record']['withheld_untimed_findings']
    (parent / 'state.npz').unlink()
    parent_missing = {'evidence_inspection': copy.deepcopy(merged)}
    build.add_context(parent_missing, ctx, parent)
    assert all(not part['record']['untimed_findings'] for part in parent_missing['evidence_inspection']['parts'])


def test_unknown_vector_keeps_each_component_and_exact_uint64(tmp_path):
    a = access(tmp_path)
    a.ep['signals']['unknown vector'] = np.array([[1, -1, 2], [2, -2, 3], [3, -3, 4], [4, -4, 5]])
    a.ep['signal_meta']['unknown vector'] = {'names': ['x', 'y', 'z'], 'units': 'N'}
    a.refresh()
    field = next(f for f in a.inventory() if f['name'] == 'unknown vector')
    r = a.inspect({'field_id': field['id'], 'mode': 'samples', 'start_s': 0, 'end_s': .3})
    assert r['values'] == [[1, -1, 2], [2, -2, 3], [3, -3, 4], [4, -4, 5]]
    assert r['times_s'] == pytest.approx([0, .1, .2, .3])
    assert r['columns'] == [0, 1, 2]
    assert r['component_units'] == ['N'] * 3
    assert [s['unit'] for s in a.bind([])['series']] == ['N'] * 3
    selected = a.inspect({'field_id': field['id'], 'mode': 'samples', 'columns': [2, 0]})
    selected_series = [s for s in a.bind([])['series'] if s['sensor_id'] == selected['id']]
    assert [s['label'] for s in selected_series] == ['z', 'x']
    assert [s['values'] for s in selected_series] == [[2, 3, 4, 5], [1, 2, 3, 4]]
    a.ep['signals']['counter'] = np.array([[2**63 + i] for i in range(4)], dtype=np.uint64)
    a.refresh()
    counter = next(f for f in a.inventory() if f['name'] == 'counter')
    assert a.inspect({'field_id': counter['id']})['values'][0][0] == 2**63
    peak = a.inspect({'field_id': counter['id'], 'mode': 'extrema', 'start_s': .1, 'end_s': .2})
    assert peak['extrema'] == [{'column': 0, 'valid_rows': 2, 'missing_rows': 0,
                              'minimum': {'row': 1, 'value': 2**63+1},
                              'maximum': {'row': 2, 'value': 2**63+2}}]
    assert peak['values'] == [[2**63+1], [2**63+2]]
    assert peak['exact_integer_values'] == [{'row': 1, 'column': 0, 'decimal': str(2**63+1)},
                                            {'row': 2, 'column': 0, 'decimal': str(2**63+2)}]
    assert peak['times_s'] == pytest.approx([.1, .2])
    # A side camera's display fallback does not invalidate the anchor, but the anchor's does.
    a.ep['context']['camera_clock'] = {'left': {'clock_problem': 'repeated timestamps'}}
    assert a.inspect({'field_id': field['id']})['times_s'] == pytest.approx([0, .1, .2, .3])
    a.ep['context']['camera_clock']['exo'] = {'clock_problem': 'repeated timestamps'}
    uncertain = a.inspect({'field_id': field['id'], 'start_row': 0, 'end_row': 4})
    assert uncertain['values'] == r['values'] and 'times_s' not in uncertain
    finding = {'headline': 'Response changes', 'start_s': 0, 'end_s': .3, 'adds_beyond_video': True,
               'evidence': [{'evidence_id': uncertain['id'], 'columns': [0], 'time_s': [0, .1]}]}
    assert not a.bind([finding])['findings']
    with pytest.raises(ValueError, match='alignment'):
        a.inspect({'field_id': field['id'], 'start_s': 0, 'end_s': .3})


def test_metadata_is_discoverable_and_nested_inspection_is_bounded(tmp_path):
    a = access(tmp_path, recorded_metadata={'levels': [{'material': 'steel', 'secret_path': '/Users/x/private/a.h5'}]})
    assert ea.needs_inspection(a.ep, {})
    f = next(f for f in a.inventory() if f['name'] == 'recorded_metadata')
    r = a.inspect({'field_id': f['id'], 'mode': 'metadata', 'pointer': '/levels/0/material'})
    assert r['value'] == 'steel'
    assert '/Users/' not in json.dumps(a.inspect({'field_id': f['id'], 'mode': 'metadata'}))
    finding = {'headline': 'Recorded material constrains interpretation', 'start_s': None, 'end_s': None,
               'adds_beyond_video': True, 'evidence': [{'evidence_id': r['id'], 'pointer': r['pointer']}]}
    bound = a.bind([finding, {**finding, 'evidence': [{'evidence_id': r['id'], 'pointer': '/invented'}]}])
    assert bound['untimed_findings'] == [finding]
    assert not bound['findings'] and len(bound['unbound_findings']) == 1
    # Rebuilding must retain valid recording claims and respect later human interpretation edits.
    import copy
    from board import build
    record = {**a.record(), 'untimed_findings': bound['untimed_findings']}
    original = copy.deepcopy(record)
    unchanged = {'evidence_inspection': copy.deepcopy(record)}
    build.add_context(unchanged, a.ep['context'], tmp_path)
    assert unchanged['evidence_inspection']['untimed_findings'] == [finding]
    edited = copy.deepcopy(a.ep['context'])
    edited['data_dictionary'] = {'schema': 1, 'status': 'success', 'fields': [],
        'entries': {'material': {'meaning': 'Coating only, not the object material', 'provenance': 'human'}}}
    nested = {'evidence_inspection': ea.merge([({'piece': {'index': 1}}, {'evidence_inspection': record})])}
    build.add_context(nested, edited, tmp_path)
    stale = nested['evidence_inspection']['parts'][0]['record']
    assert stale['untimed_findings'] == []
    assert stale['withheld_untimed_findings'] == [finding]
    assert stale['inspections'] == original['inspections'] and record == original
    changed = copy.deepcopy(a.ep['context'])
    changed['recorded_metadata']['levels'][0]['material'] = 'rubber'
    rebuilt = {'evidence_inspection': copy.deepcopy(record)}
    build.add_context(rebuilt, changed, tmp_path)
    assert rebuilt['evidence_inspection']['untimed_findings'] == []
    ordinary = access(tmp_path, timebase_neighbours_in_upload=1, annotation_subtasks=['already supplied'])
    assert not ea.needs_inspection(ordinary.ep, {})
    assert not ea.needs_inspection(access(tmp_path, recorded_metadata={}).ep, {})
    with pytest.raises(ValueError):
        a.inspect({'field_id': '../../etc/passwd'})

    # Reader-owned source maps need distinct executable public owners, not private file keys.
    from prepare.formats import lerobot_metadata
    owned = {'episode_index': 0, 'profile': 'ego_head', 'state_kind': 'none'}
    sources = ['/Users/synthetic/owner_a/meta/info.json', '/Users/synthetic/owner_b/meta/info.json']
    native = {'fields': {'/native/channel': 2**60 + 1, 'path': '/native/member'}, 'calibration': 'A'}
    lerobot_metadata(owned, {'dir': '/Users/synthetic/owner_b', 'info': {**native, 'calibration': 'B'},
                            'episode_metadata': {0: {sources[0]: native}}})
    before = copy.deepcopy(owned)
    (tmp_path / 'context.json').write_text(json.dumps(owned))
    source_before = (tmp_path / 'context.json').read_bytes()
    a = access(tmp_path, **owned)
    proof = ea.source_proof(a.ep['context'], tmp_path)
    field = next(f for f in a.inventory() if f['name'] == 'recorded_metadata')
    whole = a.inspect({'field_id': field['id'], 'mode': 'metadata'})
    assert '/Users/synthetic/' not in ea.packed(whole)
    assert 'uploader_annotation' not in {f['name'] for f in a.inventory()}
    notes_field = next(f for f in a.inventory() if f['name'] == 'uploader_notes')
    notes = a.inspect({'field_id': notes_field['id'], 'mode': 'metadata'})
    assert '/Users/synthetic/' not in ea.packed(notes)
    independent = access(tmp_path, uploader_notes={'annotation': 'structured claim'},
                         uploader_annotation='Independent publisher prose.')
    assert 'uploader_annotation' in {f['name'] for f in independent.inventory()}
    from board.public_export import reader_annotation_projection
    assert reader_annotation_projection('Publisher prose.', 'Publisher prose.') is None
    withheld = access(tmp_path, uploader_notes='https://example.org/object?token=synthetic')
    withheld_field = next(f for f in withheld.inventory() if f['name'] == 'uploader_notes')
    assert withheld.inspect({'field_id': withheld_field['id'], 'mode': 'metadata'})['value'] is None
    owners = field['descriptor']['children']
    assert len(set(owners)) == 2 and all(owner.startswith('info.json') for owner in owners)
    assert set(whole['value']) == set(owners)
    assert {value['calibration'] for value in whole['value'].values()} == {'A', 'B'}
    reversed_context = {**owned, 'recorded_metadata': dict(reversed(list(owned['recorded_metadata'].items())))}
    reordered = access(tmp_path, **reversed_context)
    reordered_field = next(f for f in reordered.inventory() if f['name'] == 'recorded_metadata')
    reordered_value = reordered.inspect({'field_id': reordered_field['id'], 'mode': 'metadata'})['value']
    assert {value['calibration']: owner for owner, value in reordered_value.items()} == {
        value['calibration']: owner for owner, value in whole['value'].items()}
    for owner in owners:
        pointer = '/' + owner.replace('~', '~0').replace('/', '~1') + '/fields/~1native~1channel'
        selected = a.inspect({'field_id': field['id'], 'mode': 'metadata', 'pointer': pointer})
        assert selected['value'] == 2**60 + 1 and selected['pointer'] == pointer
        assert selected['exact_metadata_values'] == [{'pointer': '', 'decimal': str(2**60 + 1)}]
        assert whole['value'][owner]['fields']['path'] == '/native/member'
    assert {item['pointer'] for item in whole['exact_metadata_values']} == {
        '/' + owner + '/fields/~1native~1channel' for owner in owners}
    assert ea.source_proof(a.ep['context'], tmp_path) == proof
    assert owned == before and (tmp_path / 'context.json').read_bytes() == source_before


def test_native_mcap_roundtrip_keeps_frequency_types_and_shape_variants(tmp_path):
    from mcap.writer import Writer
    from prepare.formats import retain_mcap_fields
    rate = 48000
    path = tmp_path / 'sensor.mcap'
    with path.open('wb') as stream:
        writer = Writer(stream)
        writer.start()
        schema = writer.register_schema('UnknownSensorPacket', 'jsonschema', b'{}')
        channel = writer.register_channel('/sensor', 'json', schema)
        for i, width in enumerate((1024, 512)):
            message = {'sample_rate': rate, 'counter': 2**63+1+i,
                       'coordinate_frame': 'tool_optical', 'units': {'data': 'ADC'},
                       'names': ['left', 'right'], 'counters': [2**63+1+i, 2**63+3+i],
                       'big_counters': [2**80+1+i, 2**80+3+i],
                       'cells': [{'counter': 2**63+1+i}, {'counter': 2**63+3+i}],
                       'data': np.sin(2*np.pi*3000*np.arange(width)/rate).tolist()}
            writer.add_message(channel, 1790000000000000000+i*10_000_000,
                               json.dumps(message).encode(), 1790000000000000000+i*10_000_000)
        writer.finish()
    ctx = {}
    retain_mcap_fields(path, tmp_path, ctx)
    a = access(tmp_path, **ctx)
    waves = [f for f in a.inventory() if f['name'] == '/sensor data']
    assert len(waves) == 2
    wave = next(f for f in waves if f['shape'][1] == 1024)
    r = a.inspect({'field_id': wave['id'], 'mode': 'spectrum'})
    assert r['dominant_frequency_hz'] == pytest.approx(3000, abs=47)
    assert r['frequency_resolution_hz'] == pytest.approx(rate / 1024)
    assert r['packet_duration_s'] == pytest.approx(1024 / rate)
    assert r['sample_rate_hz'] == rate and 'times_s' not in r
    counter = next(f for f in a.inventory() if f['name'] == '/sensor counter (native dtype)')
    assert a.inspect({'field_id': counter['id']})['values'] == [[2**63+1], [2**63+2]]
    identity = next(f for f in a.inventory() if f['name'] == '/sensor coordinate_frame (native dtype)')
    assert a.inspect({'field_id': identity['id'], 'mode': 'metadata', 'pointer': '/0'})['value'] == 'tool_optical'
    unit = next(f for f in a.inventory() if f['name'] == '/sensor units.data (native dtype)')
    assert a.inspect({'field_id': unit['id'], 'mode': 'metadata'})['value'] == ['ADC', 'ADC']
    cells = next(f for f in a.inventory() if f['name'] == '/sensor cells')
    assert a.inspect({'field_id': cells['id']})['values'] == [[2**63+1, 2**63+3], [2**63+2, 2**63+4]]
    counters = next(f for f in a.inventory() if f['name'] == '/sensor counters')
    assert a.inspect({'field_id': counters['id']})['values'] == [[2**63+1, 2**63+3], [2**63+2, 2**63+4]]
    big = next(f for f in a.inventory() if f['name'] == '/sensor big_counters')
    assert a.inspect({'field_id': big['id']})['values'] == [[2**80+1, 2**80+3], [2**80+2, 2**80+4]]
    b = access(tmp_path, recorded_sensor_fields=[ctx])
    assert any(f['kind'] == 'native' for f in b.inventory())
    search = a.inspect({'mode': 'inventory', 'query': 'unknownsensorpacket'})
    assert search['total_fields'] == sum(f.get('descriptor', {}).get('schema') == 'UnknownSensorPacket'
                                         for f in a.inventory())
    assert all(f['descriptor']['schema'] == 'UnknownSensorPacket' for f in search['fields'])
    assert {'schema': 'UnknownSensorPacket', 'fields': search['total_fields']} in a.page()['schemas']
    exact = a.inspect({'mode': 'inventory', 'query': '/sensor counter (native dtype)'})
    assert [f['id'] for f in exact['fields']] == [counter['id']]
    assert a.inspect({'field_id': exact['fields'][0]['id']})['values'][1][0] == 2**63+2
    assert a.inspect({'mode': 'inventory', 'query': 'unrecorded-sensor'})['fields'] == []
    assert len(a.record()['coverage']) == len(a.inventory())
    # The same recorded schema cannot keep an old source-row claim after values are replaced.
    import copy
    from board import build
    counter_receipt = a.inspect({'field_id': counter['id']})
    claim = {'headline': 'Recorded counter advances', 'start_s': None, 'end_s': None,
        'adds_beyond_video': True, 'evidence': [{'evidence_id': counter_receipt['id'], 'rows': [0, 1]}]}
    saved = {**a.record(), 'untimed_findings': a.bind([claim])['untimed_findings']}
    assert saved['untimed_findings'] == [claim]
    valid = {'evidence_inspection': copy.deepcopy(saved)}
    build.add_context(valid, a.ep['context'], tmp_path)
    assert valid['evidence_inspection']['untimed_findings'] == [claim]
    archive = a.local(ctx['recorded_mcap_fields'])
    with np.load(archive) as source:
        replaced = {k: source[k] for k in source.files}
    key = a.fields[counter['id']]['key']
    replaced[key] = replaced[key].copy()
    replaced[key].flat[0] += np.uint64(9)
    np.savez_compressed(archive, **replaced)
    stale = {'evidence_inspection': copy.deepcopy(saved)}
    build.add_context(stale, a.ep['context'], tmp_path)
    assert stale['evidence_inspection']['untimed_findings'] == []
    assert stale['evidence_inspection']['withheld_untimed_findings'] == [claim]
    assert stale['evidence_inspection']['inspections'] == saved['inspections']


def test_sampling_and_invalid_requests_cannot_exceed_access_budget(tmp_path):
    a = access(tmp_path)
    a.ep['signals']['wide'] = np.arange(400000).reshape(4, 100000)
    a.refresh()
    f = next(f for f in a.inventory() if f['name'] == 'wide')
    r = a.inspect({'field_id': f['id']})
    assert len(r['columns']) <= ea.MAX_COLUMNS
    assert r['omitted_columns'] == 100000 - len(r['columns'])
    for bad in ({'columns': [-1]}, {'columns': [True]}, {'start_s': float('nan')}, {'mode': 'python'},
                {'rows': [2]}, {'start_row': 1, 'start_s': 0}):
        with pytest.raises(ValueError):
            a.inspect({'field_id': f['id'], **bad})


def test_citations_bind_only_to_available_evidence_and_stay_exportable(tmp_path):
    from board.to_board import convert
    a = access(tmp_path)
    a.ep['signals']['motor current'] = np.array([[1.], [3.], [2.], [1.]])
    a.refresh()
    f = next(f for f in a.inventory() if f['name'] == 'motor current')
    r = a.inspect({'field_id': f['id']})
    finding = {'start_s': 0, 'end_s': .3, 'headline': 'Current rises during the action',
               'observation': 'The reading rises from 1 to 3.', 'claim': 'Possible increased motor demand.',
               'confidence': 'medium', 'adds_beyond_video': True,
               'evidence': [{'evidence_id': r['id'], 'time_s': [0, .1]}]}
    doc = a.bind([finding, {**finding, 'evidence': [{'evidence_id': 'invented', 'time_s': [0]}]}])
    assert len(doc['findings']) == 1 and len(doc['unbound_findings']) == 1
    out = convert({'labels': {'timeline': []}, 'evidence_inspection': a.record(), 'sensor_evidence': doc})
    assert out['evidence_inspection']['inspections'][0]['values'][1] == [3.]
    assert out['sensor_evidence']['findings'][0]['evidence'][0]['time_s'] == [0, .1]


def test_inspection_rounds_are_cached_and_reused_without_rebilling(tmp_path):
    a = access(tmp_path, new_fact={'value': 7})
    f = next(f for f in a.inventory() if f['name'] == 'new_fact')
    def reply(content):
        return {'id': 'generation_1', 'usage': {'cost': .01}, 'choices': [{'finish_reason': 'stop',
                'message': {'content': json.dumps({'requests': [{'field_id': f['id'], 'mode': 'metadata'}]})}}]}
    result, _ = ea.discover(a, [], reply, tmp_path / 'receipt.json', rounds=1)
    assert result['cost_usd'] == .01
    def forbidden(content):
        pytest.fail('A saved inspection selection must not be billed again')
    b = access(tmp_path, new_fact={'value': 7})
    repeated, _ = ea.discover(b, [], forbidden, tmp_path / 'receipt.json', rounds=1)
    assert repeated['inspections'][0]['value'] == {'value': 7}


def test_interrupted_dispatch_is_not_silently_repeated(tmp_path):
    a = access(tmp_path, new_fact=7)
    def failure(content):
        raise TimeoutError('unknown dispatch outcome')
    doc, _ = ea.discover(a, [], failure, tmp_path / 'receipt.json', rounds=1)
    assert doc['status'] == 'incomplete'
    assert doc['cost_usd'] == ea.MAX_SELECTION_COST
    def forbidden(content):
        pytest.fail('An uncertain dispatch needs explicit retry authorization')
    doc, _ = ea.discover(access(tmp_path, new_fact=7), [], forbidden, tmp_path / 'receipt.json', rounds=1)
    assert doc['status'] == 'incomplete'
    assert doc['cost_usd'] == ea.MAX_SELECTION_COST


def test_depth_inspection_measures_requested_region_on_recorded_scale(tmp_path, monkeypatch):
    import av
    from fractions import Fraction
    path = tmp_path / 'depth.mkv'
    with av.open(str(path), 'w') as dst:
        stream = dst.add_stream('ffv1', rate=10)
        stream.width = stream.height = 16
        stream.pix_fmt = 'gray16le'
        stream.time_base = Fraction(1, 10)
        for k in range(4):
            pixels = np.full((16, 16), (k + 1) * 1000, np.uint16)
            pixels[:8, :8] = 500
            frame = av.VideoFrame.from_ndarray(pixels, format='gray16le')
            frame.pts, frame.time_base = k, Fraction(1, 10)
            for packet in stream.encode(frame):
                dst.mux(packet)
        for packet in stream.encode():
            dst.mux(packet)
    with av.open(str(path)) as src:
        pts = np.array([f.pts for f in src.decode(video=0)])
    a = access(tmp_path)
    a.ep['depth'] = {'exo': {'packed': str(path), 'pts': pts, 'km': np.arange(4), 'scale_m': .001}}
    a.ep['context']['depth'] = {'exo': {'scale_m': .001, 'units': 'm', 'kind': 'depth'}}
    a.refresh()
    field = next(f for f in a.inventory() if f['kind'] == 'depth')
    r = a.inspect({'field_id': field['id'], 'mode': 'images', 'roi': [0, 0, .5, .5]})
    assert [f['median_m'] for f in r['frames']] == [.5] * 4
    assert r['frames'][0]['spatial_grid']['medians'][0][0] == .5
    assert r['frames'][-1]['spatial_grid']['medians'][-1][-1] == 4.
    assert r['frames'][0]['spatial_grid']['unit'] == 'm'
    measured = a.inspect({'field_id': field['id'], 'mode': 'regions',
                         'regions': [[0, 0, .5, .5], [.5, .5, 1, 1]]})
    assert measured['values'] == [[.5, 1.], [.5, 2.], [.5, 3.], [.5, 4.]]
    assert measured['component_units'] == ['m', 'm']
    assert [s['unit'] for s in a.bind([])['series'] if s['sensor_id'] == measured['id']] == ['m', 'm']
    selected = a.inspect({'field_id': field['id'], 'mode': 'regions', 'rows': [0, 3],
                          'regions_by_row': [[[.5, .5, 1, 1]], [[0, 0, .5, .5]]]})
    assert selected['values'] == [[1.], [.5]] and selected['rows'] == [0, 3]
    assert selected['frames'][0]['regions'][0]['roi'] == [.5, .5, 1, 1]
    assert selected['frames'][1]['regions'][0]['roi'] == [0, 0, .5, .5]
    assert selected['times_s'] == pytest.approx([0, .3])
    for bad in ({'rows': [0, 4]}, {'rows': [0, 0]},
                {'rows': [0, 3], 'regions_by_row': [[[0, 0, 1, 1]]]}):
        with pytest.raises(ValueError):
            a.inspect({'field_id': field['id'], 'mode': 'regions', **bad})
    assert r['times_s'] == pytest.approx([0, .1, .2, .3])
    image_media = a.images[ea.packed([field['id'], {'field_id': field['id'], 'mode': 'images',
                                                 'roi': [0, 0, .5, .5]}])]
    region_media = a.images[ea.packed([field['id'], {'field_id': field['id'], 'mode': 'regions',
                            'rows': [0, 3], 'regions_by_row': [[[.5, .5, 1, 1]], [[0, 0, .5, .5]]]}])]
    assert sum(p['type'] == 'image_url' for p in image_media) == 4
    assert sum(p['type'] == 'image_url' for p in region_media) == 2
    assert selected['preview_rows'] == [0, 3]
    assert [f['pixel_sha256'] for f in selected['frames']] == [r['frames'][i]['pixel_sha256'] for i in [0, 3]]
    assert all('R1' in p['text'] and 'depth-image' in p['text'] for p in region_media if p['type'] == 'text')
    finding = {'headline': 'Depth in the inspected region stays steady', 'start_s': 0, 'end_s': .3,
               'adds_beyond_video': True, 'evidence': [{'evidence_id': r['id'], 'time_s': [0, .1]}]}
    from label.sensor_evidence import merge
    doc = a.bind([finding])
    merged = merge([({'piece': {'index': 1, 't0_s': 10}}, {'sensor_evidence': doc})], a.ep['context'])
    assert merged['findings'][0]['start_s'] == 10
    assert merged['series'][0]['unit'] == 'm'
    # An unaligned robot state does not invalidate independently timed depth readings.
    a.ep['context']['state_unaligned'] = True
    independent_depth = a.bind([finding])
    from label.sensor_evidence import compatible
    assert compatible(independent_depth, a.ep['context'])
    assert merge([({'piece': {'index': 1, 't0_s': 10}}, {'sensor_evidence': independent_depth})],
                 a.ep['context'])['findings'][0]['start_s'] == 10
    a.ep['context'].pop('state_unaligned')
    # Actual marked depth previews may be removed at the final payload boundary.
    # Neither timed nor recording-level surface claims may survive that removal.
    from label import harness
    region_request = {'field_id': field['id'], 'mode': 'regions', 'regions': [[0, 0, .5, .5]]}
    region_receipt = a.inspect(region_request)
    marked = a.images[ea.packed([field['id'], region_request])]
    monkeypatch.setattr(ea, 'discover', lambda *args, **kwargs: ({**a.record(), 'limitations': []},
        [{'type': 'text', 'text': 'INSPECTED EVIDENCE\n' + ea.packed(region_receipt)}] + marked))
    monkeypatch.setattr(harness, 'MAX_IMAGES', 1)
    surface = {'headline': 'Inspected surface depth is stable', 'start_s': 0, 'end_s': .3,
        'adds_beyond_video': True, 'evidence': [{'evidence_id': region_receipt['id'], 'time_s': [0, .3]}]}
    untimed_surface = {**surface, 'start_s': None, 'end_s': None,
        'evidence': [{'evidence_id': region_receipt['id'], 'rows': [0, 3]}]}
    def final_provider(content, *args, **kwargs):
        assert sum(p['type'] == 'image_url' for p in content) == 1
        return {'id': 'payload_boundary', 'usage': {'cost': .01}, 'choices': [{'finish_reason': 'stop',
            'message': {'content': json.dumps({'timeline': [], 'evidence_findings': [surface, untimed_surface]})}}]}
    monkeypatch.setattr(harness, 'call_model', final_provider)
    withheld = harness._call_and_record(tmp_path, tmp_path / 'capped.json',
        [next(p for p in marked if p['type'] == 'image_url')], 0, {}, model='fixture',
        reasoning='medium', api_key='fixture', max_tokens=2000, timeout=10, evidence_access=a)
    assert not withheld['sensor_evidence']['findings']
    assert not withheld['evidence_inspection']['untimed_findings']
    saved_region = next(r for r in withheld['evidence_inspection']['inspections'] if r['id'] == region_receipt['id'])
    assert saved_region['withheld_from_final'] is True
    assert saved_region['values'] == region_receipt['values']
    import copy
    changed_clock = copy.deepcopy(a.ep['context'])
    changed_clock['depth_camera_clock'] = {'exo': {'what': 'Presentation timing assumed'}}
    from label.sensor_evidence import compatible
    assert not compatible(doc, changed_clock)
    # A scale alone does not convert inverse depth into a distance.
    a.ep['depth']['exo']['kind'] = 'disparity'
    a.ep['context']['depth']['exo']['kind'] = 'disparity'
    a.refresh()
    inverse = a.inspect({'field_id': field['id'], 'mode': 'images'})
    assert all('median_m' not in frame for frame in inverse['frames'])
    assert inverse['frames'][0]['spatial_grid']['unit'] is None
    inverse_media = a.images[ea.packed([field['id'], {'field_id': field['id'], 'mode': 'images'}])]
    assert all('metric:' not in p['text'] for p in inverse_media if p['type'] == 'text')
    # Presentation clocks may permit browsing, but cannot support a timed sensor finding.
    a.ep['context']['depth_camera_clock'] = {'exo': {'what': 'Presentation timing assumed'}}
    a.refresh()
    uncertain = a.inspect({'field_id': field['id'], 'mode': 'regions', 'start_row': 0, 'end_row': 4})
    assert 'times_s' not in uncertain
    assert uncertain['rows'] == [0, 1, 2, 3]
    assert uncertain['source_frames'][0]['pts'] == int(pts[0])
    from board import build
    source_claim = {'headline': 'Recorded region depth changes', 'start_s': None, 'end_s': None,
        'adds_beyond_video': True, 'evidence': [{'evidence_id': uncertain['id'], 'rows': [0, 3]}]}
    source_record = {**a.record(), 'untimed_findings': a.bind([source_claim])['untimed_findings']}
    unchanged = {'evidence_inspection': copy.deepcopy(source_record)}
    build.add_context(unchanged, a.ep['context'], tmp_path)
    assert unchanged['evidence_inspection']['untimed_findings'] == [source_claim]
    changed_scale = copy.deepcopy(a.ep['context'])
    changed_scale['depth']['exo']['scale_m'] = .002
    rebuilt = {'evidence_inspection': copy.deepcopy(source_record)}
    build.add_context(rebuilt, changed_scale, tmp_path)
    assert rebuilt['evidence_inspection']['untimed_findings'] == []
    with pytest.raises(ValueError, match='alignment'):
        a.inspect({'field_id': field['id'], 'mode': 'regions', 'start_s': 0, 'end_s': .3})


def test_harness_uses_requested_evidence_in_final_annotation_and_accounts_for_cost(tmp_path, monkeypatch):
    from label import harness, episode
    from test_camera_timing import recording
    from board.to_board import convert
    ep = recording(tmp_path, np.arange(12) / 30)
    # Exercise the production decoder's memory-saving frame wrapper on a real wide video.
    import av
    with av.open(str(tmp_path / 'exo.mp4'), 'w') as dst:
        stream = dst.add_stream('libx264', rate=30)
        stream.width, stream.height, stream.pix_fmt = 640, 360, 'yuv420p'
        for k in range(12):
            frame = av.VideoFrame.from_ndarray(np.full((360, 640, 3), k * 2, np.uint8), format='rgb24')
            for packet in stream.encode(frame):
                dst.mux(packet)
        for packet in stream.encode():
            dst.mux(packet)
    def provider(content, model, reasoning, api_key, max_tokens, timeout, **options):
        selections = [p['text'] for p in content if p.get('type') == 'text' and '\nFIELD INVENTORY\n' in p['text']]
        receipts = [json.loads(p['text'].split('\n', 1)[1]) for p in content
                    if p.get('type') == 'text' and p.get('text', '').startswith('INSPECTED EVIDENCE\n')]
        if selections:
            page = json.loads(selections[-1].split('\nFIELD INVENTORY\n')[1])
            field = next(f for f in page['fields'] if f['name'] == 'force')
            camera = next(f for f in page['fields'] if f['kind'] == 'image')
            text = {'requests': [] if receipts else [{'field_id': field['id'], 'mode': 'samples'},
                    {'field_id': camera['id'], 'mode': 'images'}]}
        else:
            r = next(r for r in receipts if r['mode'] == 'samples')
            camera = next(r for r in receipts if r['mode'] == 'images')
            assert len(camera['times_s']) == 4 and not camera['unavailable_rows']
            assert r['values'][0] == [0.] and r['values'][-1] == [11.]
            text = {'timeline': [], 'evidence_findings': [{'start_s': r['times_s'][0], 'end_s': r['times_s'][-1],
                    'headline': 'Recorded response increases', 'observation': '0 to 11',
                    'claim': 'An uncalibrated sensor response increases.', 'adds_beyond_video': True,
                    'confidence': 'medium', 'evidence': [{'evidence_id': r['id'], 'time_s': r['times_s'][::5]}]}]}
        return {'id': 'actual_inspection_fixture', 'usage': {'cost': .01}, 'choices': [{'finish_reason': 'stop',
                'message': {'content': json.dumps(text)}}]}
    monkeypatch.setattr(harness, 'call_model', provider)
    result = harness.label_episode(ep, tmp_path / 'out.json', model='openai/gpt-6-astra', reasoning='medium',
                                   api_key='fixture', max_tokens=2000, timeout=10, cell_w=128)
    assert len(result['sensor_evidence']['findings']) == 1
    assert harness.episode_cost(result) == pytest.approx(.03)
    exported = convert(result)
    assert exported['evidence_inspection']['cost_usd'] == pytest.approx(.02)
    assert exported['sensor_evidence']['series'][0]['values'] == list(np.arange(12, dtype=float))
    assert episode.load(ep)['signals']['force'].shape == (12, 1)
    def unavailable(*args, **kwargs):
        raise TimeoutError('Final dispatch outcome is unknown')
    monkeypatch.setattr(harness, 'call_model', unavailable)
    with pytest.raises(TimeoutError):
        harness.label_episode(ep, tmp_path / 'out.json', model='openai/gpt-6-astra', reasoning='medium',
                              api_key='fixture', max_tokens=2000, timeout=10, cell_w=128)
    failed = json.loads((tmp_path / 'noreply_out.json').read_text())
    assert failed['final_dispatch_outcome'] == 'unverified'
    assert harness.episode_cost(failed) == pytest.approx(failed['final_reserved_usd'] + .03)
    assert failed['final_cost_history'][0]['generation_id'] == 'actual_inspection_fixture'
    assert failed['evidence_inspection']['inspections'][0]['values'][-1] == [11.]

    missing_calls = []
    def missing_usage(content, *args, **kwargs):
        reply = provider(content, *args, **kwargs)
        if any('FIELD INVENTORY' in p.get('text', '') for p in content):
            missing_calls.append(1)
            reply.pop('usage')
        return reply
    monkeypatch.setattr(harness, 'call_model', missing_usage)
    missing = harness.label_episode(ep, tmp_path / 'missing_usage.json', model='openai/gpt-6-astra',
                                    reasoning='medium', api_key='fixture', max_tokens=2000, timeout=10, cell_w=128)
    assert len(missing_calls) == 1
    assert missing['evidence_inspection']['status'] == 'incomplete'
    assert missing['evidence_inspection']['cost_is_conservative_estimate'] is True
    assert missing['evidence_inspection']['cost_usd'] > 0
    def broken_decoder(*args, **kwargs):
        raise AttributeError('Unexpected decoded frame interface')
    prior_cost = harness.episode_cost(failed)
    monkeypatch.setattr(ea.Access, 'pictures', broken_decoder)
    with pytest.raises(AttributeError):
        harness.label_episode(ep, tmp_path / 'out.json', model='openai/gpt-6-astra', reasoning='medium',
                              api_key='fixture', max_tokens=2000, timeout=10, cell_w=128)
    failed = json.loads((tmp_path / 'noreply_out.json').read_text())
    assert harness.episode_cost(failed) == pytest.approx(prior_cost)
    assert failed['final_dispatch_outcome'] == 'not dispatched'


def test_run_batch_caps_selection_before_dispatch_and_keeps_paid_inspection(tmp_path, monkeypatch):
    from label import harness
    from label.dictionary_stage import label_spend
    from test_camera_timing import recording
    ep = recording(tmp_path, np.arange(12) / 30)
    calls = []
    def provider(content, *args, **kwargs):
        calls.append(1)
        return {'usage': {'cost': .02}, 'choices': [{'finish_reason': 'stop',
                'message': {'content': '{"requests": []}'}}]}
    monkeypatch.setattr(harness, 'call_model', provider)
    args = dict(keys=['sk-or-fixture'], concurrency=1, force=False, model='openai/gpt-6-astra',
                reasoning='medium', max_tokens=64000, timeout=1, cell_w=128)
    small = tmp_path / 'small'
    assert harness.run_batch([ep], small, max_spend=.01, **args) == 0
    assert calls == []
    small_receipt = json.loads((small / f'noreply_{ep.name}.json').read_text())
    assert small_receipt['final_dispatch_outcome'] == 'not dispatched'
    assert small_receipt['evidence_inspection']['status'] == 'incomplete'
    paid = tmp_path / 'job' / 'run' / 'out'
    assert harness.run_batch([ep], paid, max_spend=1.5, **args) == 0
    assert calls == [1]
    paid_receipt = json.loads((paid / f'noreply_{ep.name}.json').read_text())
    assert paid_receipt['evidence_inspection']['cost_usd'] == pytest.approx(.02)
    assert harness.episode_cost(paid_receipt) == pytest.approx(.02)
    assert label_spend(tmp_path / 'job') == pytest.approx(.02)
    assert paid_receipt['evidence_inspection']['rounds'][0]['state'] == 'received'


def test_crashed_selection_reservation_survives_resume_and_service_spend(tmp_path):
    from label.dictionary_stage import label_spend, label_spend_complete
    inspected = access(tmp_path)
    job = tmp_path / 'job'
    cache = job / 'run' / 'out' / '.evidence' / 'episode_a.json.inspection.json'
    def interrupted(_):
        raise SystemExit('worker stopped after selection dispatch')
    with pytest.raises(SystemExit):
        ea.discover(inspected, [{'type': 'text', 'text': 'test'}], interrupted, cache,
                    rounds=1, reserve_cost=.2)
    saved = json.loads(cache.read_text())
    assert saved['rounds'][0]['state'] == 'dispatched'
    assert saved['rounds'][0]['reserved_cost_usd'] == .2
    assert saved['cost_usd'] == 0
    assert label_spend(job) == pytest.approx(.2)
    assert label_spend_complete(job) is False
    called = []
    doc, _ = ea.discover(inspected, [{'type': 'text', 'text': 'test'}], lambda _: called.append(1), cache,
                         rounds=1, reserve_cost=.2)
    assert called == []
    assert doc['status'] == 'incomplete'
    assert doc['cost_is_conservative_estimate'] is True
    assert doc['cost_usd'] == pytest.approx(.2)
    assert json.loads(cache.read_text())['rounds'][0]['reservation_accounted'] is True
    assert label_spend(job) == pytest.approx(.2)


def test_rejected_selection_proof_survives_settlement_crash_and_retains_claim_history(tmp_path, monkeypatch):
    import io
    import urllib.error
    from label import harness
    inspected = access(tmp_path)
    cache = tmp_path / 'job' / 'run' / 'out' / '.evidence' / 'episode_a.json.inspection.json'
    def rejected(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 401, 'Unauthorized', {}, io.BytesIO(b'{}'))
    monkeypatch.setattr(harness.urllib.request, 'urlopen', rejected)
    def call(content):
        return harness.call_model_once(content, 'openai/gpt-6-astra', 'medium', 'sk-or-fixture', 100, 1)
    def settlement(reserved, cost):
        assert reserved == .2 and cost == 0
        saved = json.loads(cache.read_text())
        assert saved['rounds'][0]['state'] == 'rejected unpaid' and saved['cost_usd'] == 0
        raise SystemExit('stopped after durable refusal')
    with pytest.raises(SystemExit):
        ea.discover(inspected, [], call, cache, rounds=1, reserve_cost=.2, settle_dispatch=settlement)
    assert cache.with_name(cache.name + '.round0.claim').exists()
    monkeypatch.setattr(harness.urllib.request, 'urlopen', lambda *a, **kw: io.BytesIO(json.dumps({
        'usage': {'cost': .01}, 'choices': [{'message': {'content': '{"requests":[]}'}}]}).encode()))
    doc, _ = ea.discover(inspected, [], call, cache, rounds=1, reserve_cost=.2)
    assert doc['status'] == 'complete' and doc['cost_usd'] == pytest.approx(.01)
    assert doc['rounds'][0]['state'] == 'received'
    assert doc['rounds'][0]['rejected_attempts'][0]['http_status'] == 401
    assert cache.with_name(cache.name + '.round0.retry1.claim').exists()


def test_unproven_old_key_error_keeps_conservative_hold_without_replay(tmp_path):
    from label import harness
    inspected = access(tmp_path)
    cache = tmp_path / 'inspection.json'
    def unproven(_):
        raise harness.KeyExhausted('HTTP 402: legacy text without a saved response')
    doc, _ = ea.discover(inspected, [], unproven, cache, rounds=1, reserve_cost=.2)
    before = json.loads(cache.read_text())
    assert doc['status'] == 'incomplete' and doc['cost_usd'] == .2
    resumed, _ = ea.discover(inspected, [], lambda _: pytest.fail('unknown dispatch replayed'), cache,
                             rounds=1, reserve_cost=.2)
    assert resumed['cost_usd'] == .2 and resumed['rounds'] == before['rounds']
    assert resumed['cost_is_conservative_estimate'] is True


def test_malformed_rejection_proof_becomes_unknown_without_retry(tmp_path):
    from label import harness
    inspected = access(tmp_path)
    cache = tmp_path / 'inspection.json'
    def rejected(_):
        raise harness.KeyExhausted('explicit refusal', http_status=402, first_attempt=True)
    with pytest.raises(harness.KeyExhausted):
        ea.discover(inspected, [], rejected, cache, rounds=1, reserve_cost=.2)
    saved = json.loads(cache.read_text())
    saved['rounds'][0].update(http_attempt=True, cost_usd=False)
    cache.write_text(json.dumps(saved))
    doc, _ = ea.discover(inspected, [], lambda _: pytest.fail('malformed rejection replayed'), cache,
                         rounds=1, reserve_cost=.2)
    assert doc['status'] == 'incomplete' and doc['cost_usd'] == .2
    assert doc['rounds'][0]['state'] == 'dispatched'
    assert doc['rounds'][0]['reservation_accounted'] is True


def test_legacy_resume_diagnostic_does_not_change_settled_spend(tmp_path):
    from label import harness
    from label.dictionary_stage import label_spend, label_spend_complete
    ep = tmp_path / 'episode_a'
    ep.mkdir()
    job = tmp_path / 'job'
    out = job / 'run' / 'out'
    out.mkdir(parents=True)
    (out / 'episode_a.json').write_text(json.dumps({
        'parse_ok': True, 'labels': {}, 'usage': {'est_cost_usd': .2}}))
    assert harness.run_batch([ep], out, keys=['sk-or-fixture'], concurrency=1, force=False) == 1
    diagnostic = json.loads((out / 'stale_episode_a.json').read_text())
    assert diagnostic['status'] == 'stale'
    assert label_spend(job) == pytest.approx(.2)
    assert label_spend_complete(job) is True


@pytest.mark.parametrize('prefix', ['', 'failed_', 'noreply_'])
@pytest.mark.parametrize('record, expected, complete', [
    ({'usage': {'est_cost_usd': None}}, 1e300, False),
    ({'final_dispatch_outcome': 'unverified', 'final_reserved_usd': .3}, .3, False),
    ({'final_dispatch_outcome': 'claimed', 'final_reserved_usd': .4}, .4, False),
])
def test_spend_retains_unknown_paid_receipts(tmp_path, prefix, record, expected, complete):
    from label.dictionary_stage import label_spend, label_spend_complete
    out = tmp_path / 'run' / 'out'
    out.mkdir(parents=True)
    (out / f'{prefix}episode_a.json').write_text(json.dumps(record))
    assert label_spend(tmp_path) == pytest.approx(expected)
    assert label_spend_complete(tmp_path) is complete


def test_rgb_request_and_model_call_are_unchanged(tmp_path, monkeypatch):
    from label import harness, episode
    from test_camera_timing import recording
    ep = recording(tmp_path, np.arange(12) / 30)
    ctx = json.loads((ep / 'context.json').read_text())
    original_signals = ctx.pop('signals')
    (ep / 'context.json').write_text(json.dumps(ctx))
    baseline = episode.build_request(ep, cell_w=128)
    routed = episode.build_request(ep, cell_w=128, inspect_evidence=True)
    assert routed.pop('evidence_access') is None
    assert routed['content'] == baseline['content'] and routed['prompt'] == baseline['prompt']
    calls = []
    def provider(content, *args, **kwargs):
        calls.append(content)
        assert content == baseline['content']
        return {'usage': {'cost': .01}, 'choices': [{'finish_reason': 'stop',
                'message': {'content': '{"timeline": [], "task_summary": "Wait beside the visible workspace."}'}}]}
    monkeypatch.setattr(harness, 'call_model', provider)
    result = harness.label_episode(ep, tmp_path / 'rgb.json', model='openai/gpt-6-astra', reasoning='medium',
                                   api_key='fixture', max_tokens=2000, timeout=10, cell_w=128)
    assert len(calls) == 1 and 'evidence_inspection' not in result
    assert harness.episode_cost(result) == pytest.approx(.01)
    ctx.update(state_unaligned=True, signals=original_signals)
    np.savez(ep / 'signals.npz', **{original_signals[0]['key']: np.array([2**63+1, 2**63+3], np.uint64)})
    (ep / 'context.json').write_text(json.dumps(ctx))
    unaligned = episode.build_request(ep, cell_w=128, inspect_evidence=True)['evidence_access']
    field = next(f for f in unaligned.inventory() if f['name'] == 'force')
    receipt = unaligned.inspect({'field_id': field['id']})
    assert receipt['values'] == [[2**63+1], [2**63+3]] and 'times_s' not in receipt
    ctx['state_unaligned'] = False
    ctx['signals'][0]['aligned_by'] = 'assumed start'
    (ep / 'context.json').write_text(json.dumps(ctx))
    assumed = episode.build_request(ep, cell_w=128, inspect_evidence=True)['evidence_access']
    field = next(f for f in assumed.inventory() if f['name'] == 'force')
    receipt = assumed.inspect({'field_id': field['id']})
    assert receipt['values'][:2] == [[2**63+1], [2**63+3]] and 'times_s' not in receipt


def test_inventory_pages_keep_unknown_fields_discoverable(tmp_path):
    a = access(tmp_path, **{'unknown_' + str(i): {'x': 'v'} for i in range(500)})
    a.ep['signals'] = {'measurement_' + str(i): np.arange(4)[:, None] for i in range(100)}
    a.refresh()
    rows, page = [], a.page()
    measurements = {f['id']: f['name'] for f in a.inventory() if f['kind'] == 'numeric'}
    assert page['measurement_index'] == measurements
    assert page['measurement_index_omitted'] == 0
    while True:
        rows.extend(page['fields'])
        if page['next_offset'] is None:
            break
        page = a.inspect({'mode': 'inventory', 'offset': page['next_offset']})
    assert len(rows) == len(a.inventory())
    assert len({f['id'] for f in rows}) == len(rows)
    def page_request(content):
        return {'usage': {'cost': .01}, 'choices': [{'message': {'content': json.dumps({
            'requests': [{'mode': 'inventory', 'offset': a.page()['next_offset']}]})}}]}
    doc, content = ea.discover(a, [], page_request, tmp_path / 'pages.json', rounds=1)
    assert not doc['limitations']
    assert json.loads(content[0]['text'].split('\n', 1)[1])['fields']


def test_unaligned_state_is_inspectable_without_invented_timestamps(tmp_path):
    a = access(tmp_path, state_unaligned=True)
    a.ep['state'] = np.arange(12).reshape(4, 3)
    a.refresh()
    f = next(f for f in a.inventory() if f['name'] == 'state')
    r = a.inspect({'field_id': f['id']})
    assert r['values'][0] == [0, 1, 2]
    assert 'times_s' not in r


@pytest.mark.parametrize('rate_field', ['sample_rate', 'fs', 'sample_frequency', 'sample_rate_hz'])
def test_spectrum_joins_rate_to_source_message_not_packet_row(tmp_path, rate_field):
    from mcap.writer import Writer
    from prepare import formats
    rate, n = 48000, 1024
    samples = np.sin(2*np.pi*3000*np.arange(n)/rate).tolist()
    source = tmp_path / 'waveforms.mcap'
    with source.open('wb') as stream:
        writer = Writer(stream)
        writer.start()
        schema = writer.register_schema('recorded.waveform', 'jsonschema', b'{}')
        for topic, additional in [('/sensor', {}), ('/ambiguous', {'sampling_rate': rate})]:
            channel = writer.register_channel(topic, 'json', schema)
            for row, record in enumerate([{rate_field: 1000}, {rate_field: rate, 'data': samples, **additional}]):
                stamp = 10**18 + row * 10**8
                writer.add_message(channel, stamp, json.dumps(record).encode(), stamp)
        writer.finish()
    ctx = {'state_kind': 'none', 'fps': 10}
    formats.retain_mcap_fields(source, tmp_path, ctx)
    a = access(tmp_path, **ctx)
    f = next(f for f in a.inventory() if f['name'] == '/sensor data')
    assert 'spectrum' in f['available_modes']
    receipt = a.inspect({'field_id': f['id'], 'mode': 'spectrum'})
    assert receipt['sample_rate_hz'] == rate and receipt['dominant_frequency_hz'] == 3000
    ambiguous = next(f for f in a.inventory() if f['name'] == '/ambiguous data')
    with pytest.raises(ValueError, match='unique recorded sample rate'):
        a.inspect({'field_id': ambiguous['id'], 'mode': 'spectrum'})


@pytest.mark.parametrize('changing_names', [False, True])
def test_native_mcap_inspection_keeps_selected_row_component_identities(tmp_path, changing_names):
    import hashlib
    from mcap.writer import Writer
    from prepare import formats
    names = [['index_finger', 'middle_finger']] * 4
    if changing_names:
        names = [names[0], ['middle_finger', 'index_finger'], ['ring_finger', 'little_finger'], []]
    records = [{'name': labels, 'pressure': values} for labels, values in
               zip(names, [[10, 20], [21, 11], [30, 40], [50, 60]])]
    source = tmp_path / 'glove.mcap'
    with source.open('wb') as stream:
        writer = Writer(stream)
        writer.start()
        schema = writer.register_schema('named.readings', 'jsonschema', b'{}')
        channel = writer.register_channel('/glove', 'json', schema)
        for row, record in enumerate(records):
            writer.add_message(channel, 10**18 + row * 10**8, json.dumps(record).encode(),
                               10**18 + row * 10**8)
        writer.finish()
    ctx = {'episode_id': 'episode_glove', 'state_kind': 'none', 'fps': 10}
    formats.retain_mcap_fields(source, tmp_path, ctx)
    archive = tmp_path / ctx['recorded_mcap_fields']
    before = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (source, archive)]
    a = access(tmp_path, **ctx)
    f = next(f for f in a.inventory() if f['name'] == '/glove pressure')
    r = a.inspect({'field_id': f['id'], 'start_row': 1, 'end_row': 4, 'columns': [1, 0]})
    assert r['rows'] == [1, 2, 3]
    assert r['values'] == [[11, 21], [40, 30], [60, 50]]
    expected = [[labels[1], labels[0]] if labels else [None, None] for labels in names[1:]]
    assert r['component_names_by_row'] == expected
    if changing_names:
        assert not f['descriptor'].get('names') and f['descriptor']['names_vary_by_row']
        assert 'component_names' not in r
        assert any('Component identities vary' in note for note in r['limitations'])
    else:
        assert f['descriptor']['names'] == names[0]
        assert r['component_names'] == expected[0]
    assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in (source, archive)] == before


@pytest.mark.parametrize('declared_role', [None, 'command', 'target', 'invalid'])
def test_canonical_hdf_commands_reach_inspection_without_extra_sensors(tmp_path, declared_role):
    import h5py
    from prepare import formats
    from label import episode
    root = tmp_path / 'upload'
    root.mkdir()
    names = [f'joint{i}_rad' for i in range(1, 7)] + ['gripper_fraction']
    with h5py.File(root / 'canonical.hdf5', 'w') as recording:
        demo = recording.create_group('data/demo_0')
        demo.create_dataset('images_top', data=np.full((6, 64, 80, 3), [70, 110, 130], dtype=np.uint8))
        demo.create_dataset('timestamps', data=10**12 + np.arange(6, dtype=np.int64) * 100_000_000)
        for kind, values in [('qpos', np.zeros((6, 7))), ('action', np.ones((6, 7)))]:
            field = demo.create_dataset(kind, data=values)
            field.attrs['names'] = names
            field.attrs['units'] = ['rad'] * 6 + ['fraction']
            if kind == 'action' and declared_role is not None:
                field.attrs['action_semantics'] = json.dumps({'version': 1, 'role': declared_role})
    report = formats.convert(root, 'teleop_arms', tmp_path / 'episodes', 'recorded sensors', 900)
    assert not report['failed']
    directory = tmp_path / 'episodes' / report['episodes'][0]['episode_id']
    ep = episode.load(directory)
    assert not ep['signals'] and ep['action'] is not None
    request = episode.build_request(directory, inspect_evidence=True)
    access = request['evidence_access']
    assert access is not None
    receipts = {}
    expected_role = declared_role if declared_role in ('command', 'target') else 'action/target_unknown'
    for kind, role in [('state', 'observation'), ('action', expected_role)]:
        field = next(row for row in access.inventory() if row['name'] == kind and row['kind'] == 'numeric')
        assert field['descriptor']['recorded_role'] == role
        if kind == 'action':
            assert field['descriptor']['action_semantics']['role'] == expected_role
            assert 'RECORDED ACTION SEMANTICS' in request['prompt']
        receipts[kind] = access.inspect({'field_id': field['id'], 'columns': [0, 6]})
        assert receipts[kind]['component_names'] == [names[0], names[6]]
        assert receipts[kind]['component_units'] == ['rad', 'fraction']
    assert receipts['state']['values'] == [[0, 0]] * 6
    assert receipts['action']['values'] == [[1, 1]] * 6
    assert receipts['action']['times_s'] == receipts['state']['times_s']
    if declared_role == 'command':
        from prepare.state_contract import action_semantics
        ep['context']['action_semantics'] = {'version': 1, 'producer': 'another exact source', 'role': 'command'}
        assert action_semantics(ep['context'])['role'] == 'command'
        ep['context']['action_semantics']['role'] = 'target'
        assert action_semantics(ep['context'])['role'] == 'action/target_unknown'
        ep['context']['action_semantics'] = {'version': 1, 'producer': 'invalid source', 'role': 'command',
            'relationship': {'kind': 'next_observation', 'field': 'state', 'row_offset': 1, 'terminal': 'repeat_last'}}
        assert action_semantics(ep['context'])['role'] == 'action/target_unknown'
        assert ea.needs_inspection(ep, {})
        ep['context'].pop('action_semantics')
    ep['action'] = None
    assert not ea.needs_inspection(ep, {})
    ep['state'] = np.zeros((6, 0))
    assert not ea.needs_inspection(ep, {})


def test_each_selection_respects_media_caps_before_dispatch(tmp_path):
    a = access(tmp_path, new_fact=7)
    calls = []
    content = [{'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,AA=='}}] * 4
    def forbidden(parts):
        calls.append(parts)
        pytest.fail('Oversized content must not be dispatched')
    doc, _ = ea.discover(a, content, forbidden, tmp_path / 'receipt.json', image_limit=3)
    assert doc['status'] == 'incomplete'
    assert not calls


def test_piece_merge_keeps_generic_findings_and_inspection_cost(tmp_path):
    from label.sensor_evidence import merge
    meta = {'name': 'pressure array', 'key': 's0', 'names': [f'taxel_{i}' for i in range(256)], 'units': 'Pa'}
    ctx = {'episode_id': 'parent', 'fps': 10, 'signals': [meta]}
    a = access(tmp_path, **ctx)
    a.ep['signals']['pressure array'] = np.repeat(np.array([[1.], [3.], [2.], [1.]]), 256, axis=1)
    a.ep['signal_meta']['pressure array'] = meta
    a.refresh()
    f = next(f for f in a.inventory() if f['name'] == 'pressure array')
    r = a.inspect({'field_id': f['id']})
    finding = {'start_s': 0, 'end_s': .3, 'headline': 'Response increases', 'adds_beyond_video': True,
               'evidence': [{'evidence_id': r['id'], 'time_s': [0, .1]}]}
    pc = {'piece': {'index': 2, 't0_s': 10, 't1_s': 10.3}}
    record = {'sensor_evidence': a.bind([finding]), 'evidence_inspection': {**a.record(), 'cost_usd': .1}}
    doc = merge([(pc, record)], ctx)
    assert doc['findings'][0]['start_s'] == 10
    assert doc['findings'][0]['evidence'][0]['time_s'] == [10, 10.1]
    assert doc['series'][0]['unit'] == 'Pa'
    stitched = ea.merge([(pc, record)])
    assert stitched['cost_usd'] == .1
    assert stitched['parts'][0]['time_origin_s'] == 10
    assert stitched['parts'][0]['record']['inspections'][0]['times_s'][0] == 0


@pytest.mark.parametrize('changed_kind', [None, 'state', 'action', 'camera'])
def test_piece_merge_preserves_native_meanings_and_withholds_only_changed_citations(tmp_path, changed_kind):
    import copy
    from label import episode, sensor_evidence
    from tests.test_label import _packed_mp4
    directory = tmp_path / 'episode_native__p01'
    directory.mkdir()
    movie = directory / 'scene.mp4'
    _packed_mp4(movie, 3)
    parent = {'episode_id': 'episode_native', 'profile': 'teleop_arms', 'state_kind': 'joints',
              'fps': 30, 'source': {'state': '/observations/qpos', 'action': '/action'},
              'cameras': {'exo': {'name': 'recorded_head'}},
              'data_dictionary': {'episode_id': 'episode_native', 'fields': [
                  {'id': 'state', 'kind': 'state', 'name': '/observations/qpos', 'bindings': [
                      {'episode': 'episode_native', 'context_path': 'state', 'file': 'state.npz', 'key': 'state'}]},
                  {'id': 'action', 'kind': 'action', 'name': '/action', 'bindings': [
                      {'episode': 'episode_native', 'context_path': 'action', 'file': 'state.npz', 'key': 'action'}]},
                  {'id': 'camera', 'kind': 'camera', 'name': 'recorded_head', 'bindings': [
                      {'episode': 'episode_native', 'context_path': 'cameras/exo'}]}],
                  'entries': {kind: {'meaning': 'Reviewed ' + kind, 'role': 'unknown', 'provenance': 'human'}
                              for kind in ('state', 'action', 'camera')}}}
    piece = {**parent, 'episode_id': directory.name,
             'piece': {'of': 'episode_native', 'index': 1, 't0_s': 10, 't1_s': 10.1}}
    (directory / 'context.json').write_text(json.dumps(piece))
    (directory / 'sources.json').write_text(json.dumps({'exo': {
        'packed': str(movie), 'base_s': 0, 'n_frames': 3}}))
    np.savez(directory / 'state.npz', state=np.array([[1], [2], [3]]), action=np.array([[2], [3], [4]]))
    access = ea.Access(episode.load(directory))
    references, findings = {}, []
    for field in access.inventory():
        if field['kind'] not in ('numeric', 'image'):
            continue
        kind = 'camera' if field['kind'] == 'image' else field['name']
        receipt = access.inspect({'field_id': field['id']})
        references[kind] = {'evidence_id': receipt['id'], 'time_s': [0, 2 / 30]}
        findings.append({'headline': kind, 'start_s': 0, 'end_s': 2 / 30, 'adds_beyond_video': True,
                         'evidence': [references[kind]]})
    findings.append({'headline': 'combined', 'start_s': 0, 'end_s': 2 / 30, 'adds_beyond_video': True,
                     'evidence': [references['state'], references['camera']]})
    original = access.bind(findings)
    assert len(original['findings']) == 4
    current = copy.deepcopy(parent)
    if changed_kind:
        current['data_dictionary']['entries'][changed_kind]['meaning'] = 'Corrected reviewed meaning'
    merged = sensor_evidence.merge([(piece, {'sensor_evidence': original})], current)
    expected = {None: ['state', 'action', 'camera', 'combined'],
                'state': ['action', 'camera'], 'action': ['state', 'camera', 'combined'],
                'camera': ['state', 'action']}[changed_kind]
    assert sorted(finding['headline'] for finding in merged['findings']) == sorted(expected)
    assert len(merged['unbound_findings']) == 4 - len(expected)
    assert all(finding['start_s'] == 10 for finding in merged['findings'])
    assert original['findings'] == access.bind(findings)['findings']


def test_source_row_findings_are_saved_without_becoming_timed_overlays(tmp_path):
    a = access(tmp_path, state_unaligned=True)
    a.ep['signals']['unknown force'] = np.arange(12).reshape(4, 3)
    a.refresh()
    f = next(f for f in a.inventory() if f['name'] == 'unknown force')
    r = a.inspect({'field_id': f['id']})
    doc = a.bind([{'headline': 'Source readings increase', 'start_s': None, 'end_s': None,
                   'adds_beyond_video': True, 'evidence': [{'evidence_id': r['id'], 'rows': [0, 3]}]}])
    assert not doc['findings'] and not doc['unbound_findings']
    assert doc['untimed_findings'][0]['evidence'][0]['rows'] == [0, 3]


def test_source_proof_covers_depth_presentation_and_retained_camera_assets(tmp_path):
    ctx = {'state_kind': 'none', 'depth_presentation_times': 'depth_display.npz',
           'unshown_cameras': [{'name': 'side', 'packed': 'side.mp4',
                                'camera_times': 'side_times.npz', 'kmap': 'side_kmap.npy'}]}
    assets = ['depth_display.npz', 'side.mp4', 'side_times.npz', 'side_kmap.npy']
    for name in assets:
        (tmp_path / name).write_bytes(b'original')
    saved = ea.source_proof(ctx, tmp_path)
    assert ea.same_source_proof(saved, ctx, tmp_path)
    for name in assets:
        (tmp_path / name).write_bytes(b'changed')
        assert not ea.same_source_proof(saved, ctx, tmp_path), name
        (tmp_path / name).write_bytes(b'original')
        assert ea.same_source_proof(saved, ctx, tmp_path)
    (tmp_path / assets[0]).unlink()
    assert not ea.same_source_proof(saved, ctx, tmp_path)


def test_resolved_annotation_policy_does_not_opt_rgb_into_inspection(tmp_path):
    from label.source_eligibility import needs_extra_evidence
    ctx = {'state_kind': 'none', 'annotation_policy': {'sample_fps': 1, 'task_mode': 'session'},
           'task_mode': 'session', 'check_versions': {'stream_pairing': 1}}
    ep = {'dir': tmp_path, 'context': ctx, 'signals': {}, 'sources': {}}
    assert not needs_extra_evidence(ctx)
    assert not ea.needs_inspection(ep, {})
    inspected = access(tmp_path, **ctx, recorded_metadata={'material': 'steel'})
    fields = {field['name']: field for field in inspected.inventory()}
    generated = inspected.inspect({'field_id': fields['check_versions']['id'], 'mode': 'metadata'})
    original = inspected.inspect({'field_id': fields['recorded_metadata']['id'], 'mode': 'metadata',
                                  'pointer': '/material'})
    def finding(receipt, headline):
        return {'headline': headline, 'start_s': None, 'end_s': None, 'adds_beyond_video': True,
                'evidence': [{'evidence_id': receipt['id'], 'pointer': receipt['pointer']}]}
    version_finding = finding(generated, 'Generated algorithm version appears in metadata')
    material_finding = finding(original, 'Recorded material constrains interpretation')
    bound = inspected.bind([version_finding, material_finding])
    assert bound['untimed_findings'] == [material_finding]
    assert [row['finding'] for row in bound['unbound_findings']] == [version_finding]
    assert fields['check_versions']['kind'] == 'context'
    assert fields['recorded_metadata']['kind'] == 'metadata'
    assert {'annotation_policy', 'task_mode', 'check_versions'} <= ea.PREPARED_CONTEXT
