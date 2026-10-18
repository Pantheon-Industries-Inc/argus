"""Exercise real public serializers using synthetic private access descriptors."""
import io
import json
from email.message import Message
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from board import serve, static


PRIVATE_PATH = '/srv/private-upload/source.h5'
PRIVATE_URL = 'https://example.org/object?X-Amz-Signature=synthetic-secret'
CITATION = 'https://example.org/public-dataset'


class Request(serve.Handler):
    def __init__(self, path, body=None, headers=None):
        self.path = path
        encoded = b'' if body is None else json.dumps(body).encode()
        self.headers = Message()
        for key, value in {'Content-Length': str(len(encoded)), **(headers or {})}.items():
            self.headers[key] = value
        self.rfile, self.wfile = io.BytesIO(encoded), io.BytesIO()
        self.status = None
        self.close_connection = False

    def send_response(self, status):
        self.status = status

    def send_header(self, *args):
        pass

    def end_headers(self):
        pass


@pytest.fixture
def public_sources(tmp_path, monkeypatch):
    paths = {name: tmp_path / name for name in ('qa', 'compare', 'sensors', 'hands', 'keypoints', 'clips')}
    for path in paths.values():
        path.mkdir()
    descriptor = {'source_path': PRIVATE_PATH, 'url': PRIVATE_URL,
                  'source': {'key': '/observations/force'}, 'public_citation': CITATION}
    label = {'dataset': 'synthetic', 'episode_id': 'ep', 'episode_prompt': 'Place the cup.',
             'duration_s': 1, 'publisher_labels': {'label': 'public label'},
             'uploader_notes': [{'title': 'Notes in the files', 'kind': 'notes', 'items': [
                 {'name': 'recorded episode metadata / ' + PRIVATE_PATH + ' / source', 'value': PRIVATE_PATH},
                 {'name': 'recorded episode metadata / ' + PRIVATE_PATH + ' / fields / path',
                  'value': '/observations/force'},
                 {'name': 'recorded annotation data source', 'value': PRIVATE_PATH}]}],
             'dataset_checks': {'reader_issues': [{'kind': 'metadata_limit', 'what': PRIVATE_PATH
                 + ' and its recorded metadata exceed the prompt note limit; complete original rows '
                 + 'and source claims remain in the episode context and uploader notes.'}]},
             'data_dictionary': {'fields': [descriptor]},
             'evidence_inspection': {'parts': [{'source_location': PRIVATE_PATH, 'record': descriptor}]},
             'sensor_evidence': {'sensors': [descriptor]},
             '_withheld_checks': {'sensor_check': {'reason': 'unavailable', 'result': {'error': 'unavailable',
                                                                                      'source_path': PRIVATE_PATH}}}}
    owners = [PRIVATE_PATH + '/owner_a/meta/info.json', PRIVATE_PATH + '/owner_b/meta/info.json']
    pointer = lambda owner: '/' + owner.replace('~', '~0').replace('/', '~1') + '/fields/~1native~1channel'
    field = {'id': 'metadata:owned', 'kind': 'metadata', 'name': 'recorded_metadata',
             'descriptor': {'children': [owners[1]]}}
    whole = {'id': 'inspection:whole', 'mode': 'metadata', 'descriptor': field, 'pointer': '',
             'value': {owner: {'fields': {'/native/channel': 2**60 + 1, 'path': '/native/member'}}
                       for owner in owners},
             'exact_metadata_values': [{'pointer': pointer(owner), 'decimal': str(2**60 + 1)} for owner in owners]}
    selected = {'id': 'inspection:selected', 'mode': 'metadata', 'descriptor': field,
                'pointer': pointer(owners[1]), 'value': 2**60 + 1,
                'exact_metadata_values': [{'pointer': '', 'decimal': str(2**60 + 1)}]}
    orphan = {'id': 'inspection:orphan', 'mode': 'metadata',
              'descriptor': {**field, 'id': 'metadata:orphan'},
              'pointer': pointer(PRIVATE_PATH + '/unknown/meta/info.json').rsplit('/', 1)[0],
              'value': {'/native/channel': 7, 'path': '/native/member',
                        'recorded_metadata': {'/native/channel': 2**60 + 1}},
              'exact_metadata_values': [{'pointer': '/recorded_metadata/~1native~1channel',
                                         'decimal': str(2**60 + 1)}]}
    native = {'id': 'inspection:native', 'mode': 'metadata',
              'descriptor': {'kind': 'metadata', 'name': 'custom_metadata'},
              'pointer': '/~1native~1channel', 'value': {'/native/channel': '/native/member'}}
    label['evidence_inspection'].update(
        source_current=True, source_proof=[{'name': 'prepared:context.json', 'sha256': 'a' * 64}],
        inspections=[whole, selected, orphan, native],
        untimed_findings=[{'evidence': [{'evidence_id': selected['id'], 'pointer': selected['pointer']},
                                      {'evidence_id': orphan['id'], 'pointer': orphan['pointer']}]}])
    notes = {**whole, 'id': 'inspection:notes',
             'descriptor': {**field, 'id': 'metadata:notes', 'name': 'uploader_notes',
                            'descriptor': {'children': ['recorded episode metadata']}},
             'value': {'recorded episode metadata': whole['value']},
             'exact_metadata_values': [{'pointer': '/recorded episode metadata' + pointer(owner),
                                        'decimal': str(2**60 + 1)} for owner in owners]}
    selected_notes = {**selected, 'id': 'inspection:selected_notes', 'descriptor': notes['descriptor'],
                      'pointer': '/recorded episode metadata', 'value': whole['value'],
                      'exact_metadata_values': whole['exact_metadata_values']}
    label['evidence_inspection']['inspections'].extend([notes, selected_notes])
    label['evidence_inspection']['parts'][0]['record'] = {
        **label['evidence_inspection']['parts'][0]['record'], 'inspections': [
        {'id': 'inspection:bounded', 'mode': 'metadata',
         'descriptor': {**field, 'descriptor': {'preview': json.dumps(field['descriptor'])[:180],
                                               'truncated': True, 'characters': 2000}}, 'pointer': '',
         'value': {'preview': json.dumps(whole['value'])[:180], 'truncated': True, 'characters': 20000}},
        notes,
        {'id': 'inspection:serialized', 'mode': 'metadata', 'pointer': '',
         'descriptor': {'kind': 'metadata', 'name': 'uploader_annotation'},
         'value': json.dumps(notes['value'], ensure_ascii=False)}]}
    (paths['qa'] / 'ep.json').write_text(json.dumps(label))
    for kind in ('sensors', 'hands', 'keypoints'):
        (paths[kind] / 'ep.json').write_text(json.dumps({'series': [], **descriptor}))
        (paths[kind] / 'index.json').write_text(json.dumps({'files': {'ep.json': {'bytes': 17, **descriptor}}}))
    for name in ('index.json', 'metrics.json'):
        (paths['compare'] / name).write_text(json.dumps({'records': [descriptor]}))
    (paths['compare'] / 'model').mkdir()
    (paths['compare'] / 'model' / 'ep.json').write_text(json.dumps(label))
    for name, key in (('HERE', 'qa'), ('COMPARE_DIR', 'compare'), ('SENSORS_DIR', 'sensors'),
                      ('HANDS_DIR', 'hands'), ('KEYPOINTS_DIR', 'keypoints')):
        monkeypatch.setattr(serve, name, paths[key])
    monkeypatch.setattr(serve, 'DICTIONARY_JOB', None)
    return paths


@pytest.mark.parametrize('route,body', [
    ('/api/episode?file=ep.json', None), ('/api/episode?file=ep.json&download=1', None),
    ('/api/export', {'files': ['ep.json']}), ('/api/dictionary?file=ep.json', None),
    ('/api/sensors?file=ep.json', None), ('/api/sensors?file=index.json', None),
    ('/api/hands?file=ep.json', None), ('/api/keypoints?file=ep.json', None),
    ('/api/keypoints?file=ep.json&download=1', None),
    ('/api/compare/index', None), ('/api/compare/metrics', None),
    ('/api/compare/episode?key=model&file=ep.json', None),
])
def test_live_public_json_boundaries(public_sources, route, body):
    before = {p: p.read_bytes() for directory in public_sources.values() for p in directory.rglob('*.json')}
    request = Request(route, body)
    request.do_POST() if body else request.do_GET()
    text = request.wfile.getvalue().decode()
    assert request.status == 200
    assert PRIVATE_PATH not in text and PRIVATE_URL not in text
    assert CITATION in text
    assert '/observations/force' in text
    if route.startswith(('/api/episode', '/api/export', '/api/compare/episode')):
        assert 'public label' in text
    assert all(p.read_bytes() == content for p, content in before.items())


@pytest.mark.parametrize('route', ['/api/episode?file=../outside.json',
                                  '/api/dictionary?file=../outside.json',
                                  '/api/sensors?file=../outside.json'])
def test_projection_does_not_replace_containment(public_sources, route):
    (public_sources['qa'].parent / 'outside.json').write_text('{"private": true}')
    request = Request(route)
    request.do_GET()
    assert request.status == 404


def test_projection_does_not_enable_remote_dictionary_writes(public_sources, monkeypatch):
    monkeypatch.setattr(serve, 'DICTIONARY_JOB', public_sources['qa'].parent)
    request = Request('/api/dictionary', {'revision': 0},
                      {'Host': 'example.org', 'Origin': 'http://example.org', 'Content-Type': 'application/json'})
    request.do_POST()
    assert request.status == 403
    assert not (public_sources['qa'].parent / 'dictionary_overrides.json').exists()


def test_static_json_boundaries(public_sources, tmp_path, monkeypatch):
    monkeypatch.setattr(static, 'plan', lambda *a, **k: [
        {'rec': {'file': 'ep.json', 'episode_id': 'ep', 'duration_s': 1},
         'ds': 'synthetic', 'media': {}, 'frames': {}}])
    args = SimpleNamespace(board=tmp_path, **public_sources, out=tmp_path / 'site',
                           build_id='synthetic', force=False, public_base=None, title='Synthetic', header=None)
    before = {p: p.read_bytes() for directory in public_sources.values() for p in directory.rglob('*.json')}
    assert static.cmd_site(args) == 0
    exported = {p.relative_to(args.out).as_posix(): p.read_text() for p in args.out.rglob('*.json')}
    assert 'synthetic/data/ep/ep.json' in exported
    assert 'synthetic/data/sensors/ep.json' in exported
    assert 'synthetic/data/compare/index.json' in exported
    assert 'synthetic/data/keypoints/index.json' in exported
    assert any(name.startswith('media/k/') for name in exported)
    for name, text in exported.items():
        assert PRIVATE_PATH not in text and PRIVATE_URL not in text, name
        if name.endswith('BUILD.json'):
            assert str(tmp_path) not in text
    assert CITATION in exported['synthetic/data/ep/ep.json']
    assert 'public label' in exported['synthetic/data/ep/ep.json']
    live = Request('/api/episode?file=ep.json&download=1')
    live.do_GET()
    for document in (json.loads(live.wfile.getvalue()), json.loads(exported['synthetic/data/ep/ep.json'])):
        evidence = document['evidence_inspection']
        whole, selected, orphan, native, notes, selected_notes = evidence['inspections']
        assert [r['id'] for r in evidence['inspections']] == [
            'inspection:whole', 'inspection:selected', 'inspection:orphan', 'inspection:native',
            'inspection:notes', 'inspection:selected_notes']
        assert evidence['source_current'] and evidence['source_proof'][0]['sha256'] == 'a' * 64
        owners = list(whole['value'])
        assert len(set(owners)) == 2 and all(owner.startswith('info.json') for owner in owners)
        assert whole['descriptor']['descriptor']['children'] == [owners[1]]
        assert selected['pointer'] == '/' + owners[1] + '/fields/~1native~1channel'
        assert selected['value'] == whole['value'][owners[1]]['fields']['/native/channel'] == 2**60 + 1
        assert whole['value'][owners[0]]['fields']['path'] == '/native/member'
        assert whole['exact_metadata_values'][1]['pointer'] == selected['pointer']
        refs = evidence['untimed_findings'][0]['evidence']
        assert refs[0]['pointer'] == selected['pointer']
        assert 'pointer' not in orphan and orphan['pointer_executable'] is False and orphan['limitations']
        assert orphan['value'] == {'/native/channel': 7, 'path': '/native/member',
                                   'recorded_metadata': {'/native/channel': 2**60 + 1}}
        assert orphan['exact_metadata_values'] == [{'pointer': '/recorded_metadata/~1native~1channel',
                                                   'decimal': str(2**60 + 1)}]
        assert 'pointer' not in refs[1] and refs[1]['pointer_executable'] is False
        assert native['pointer'] == '/~1native~1channel' and native['value'] == {'/native/channel': '/native/member'}
        assert notes['value']['recorded episode metadata'] == whole['value']
        assert selected_notes['pointer'] == '/recorded episode metadata'
        assert selected_notes['value'] == whole['value']
        assert selected_notes['exact_metadata_values'] == whole['exact_metadata_values']
        assert notes['exact_metadata_values'][1]['pointer'] == '/recorded episode metadata' + selected['pointer']
        bounded = evidence['parts'][0]['record']['inspections'][0]
        assert bounded['id'] == 'inspection:bounded' and bounded['value']['truncated']
        assert bounded['value']['characters'] == 20000 and bounded['limitations']
        serialized = evidence['parts'][0]['record']['inspections'][2]
        assert serialized['id'] == 'inspection:serialized'
        assert json.loads(serialized['value']) == notes['value']
    assert all(p.read_bytes() == content for p, content in before.items())


@pytest.mark.skipif(not shutil.which('node'), reason='node is required for the browser export path')
def test_native_integer_evidence_is_exact_in_static_json_lines(public_sources, tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    from label.evidence_access import Access
    from prepare import formats

    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    originals = np.array([2**60 + 1, 2**64 - 1], dtype=np.uint64)
    signals = formats.recorded_signals(pd.DataFrame({'flags': originals}), set(), 2)
    context = {'episode_id': 'ep', 'profile': 'ego_head', 'state_kind': 'none',
               'n_state_frames': 2, 'fps': 30, 'source': {}, 'cameras': {}}
    formats.write_signals(prepared, context, signals)
    (prepared / 'context.json').write_text(json.dumps(context))
    access = Access({'context': context, 'dir': prepared, 'signals': dict(signals),
                     'state': None, 'action': None})
    field = next(row for row in access.inventory() if row['kind'] == 'array' and row['name'] == 'flags')
    receipt = access.inspect({'field_id': field['id'], 'mode': 'samples',
                              'start_row': 0, 'end_row': 2, 'columns': [0]})
    label_path = public_sources['qa'] / 'ep.json'
    label = json.loads(label_path.read_text())
    label['ordinary_readings'] = [None, 0, -1, 1.25]
    label['evidence_inspection'] = {
        'version': 1, 'inspections': [receipt], 'untimed_findings': [
            {'headline': 'Recorded flags differ', 'adds_beyond_video': True,
             'evidence': [{'evidence_id': receipt['id'], 'rows': [0, 1], 'columns': [0]}]}]}
    label_path.write_text(json.dumps(label))

    download = Request('/api/episode?file=ep.json&download=1')
    download.do_GET()
    assert download.status == 200
    live = json.loads(download.wfile.getvalue())
    assert live['evidence_inspection']['inspections'][0]['values'] == [[int(value)] for value in originals]
    monkeypatch.setattr(static, 'plan', lambda *a, **k: [
        {'rec': {'file': 'ep.json', 'episode_id': 'ep', 'duration_s': 1},
         'ds': 'synthetic', 'media': {}, 'frames': {}}])
    args = SimpleNamespace(board=tmp_path, **public_sources, out=tmp_path / 'site',
                           build_id='precision', force=False, public_base=None, title='Synthetic', header=None)
    assert static.cmd_site(args) == 0
    episode = (args.out / 'precision/data/ep/ep.json').read_text()
    script = r"""
const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const {src, piece} = require('./tests/page_functions')();
const render = new Function('esc', 'genericSensorHeadlineHtml',
  piece('recordingFindings') + piece('recordingFindingsHtml')
  + 'return recordingFindingsHtml(recordingFindings(JSON.parse(arguments[2])));');
const html = render(String, s => s, input.episode);
const start = src.indexOf('async function exportBlob(');
const code = src.slice(start, src.indexOf('function saveBlob(', start));
const exportBlob = new Function('STATIC', 'fetch', 'episodeUrl', code + 'return exportBlob;')(
  true, async () => new Response(input.episode), file => file);
exportBlob(['ep.json']).then(async blob => {
  process.stdout.write(JSON.stringify({html, exported: await blob.text()}));
}).catch(error => { console.error(error); process.exitCode = 1; });
"""
    result = subprocess.run([shutil.which('node'), '-e', script], cwd=Path(__file__).resolve().parents[1],
                            input=json.dumps({'episode': episode}), text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    browser = json.loads(result.stdout)
    for value in originals:
        assert f'<td>{int(value)}</td>' in browser['html']
    lines = browser['exported'].splitlines()
    assert len(lines) == 1
    exported = json.loads(lines[0])
    inspected = exported['evidence_inspection']['inspections'][0]
    assert inspected['values'] == [[int(value)] for value in originals]
    assert inspected['native_dtype'] == 'uint64'
    assert inspected['exact_integer_values'] == receipt['exact_integer_values']
    assert exported['ordinary_readings'] == label['ordinary_readings']
    assert PRIVATE_PATH not in browser['exported'] and PRIVATE_URL not in browser['exported']
    assert json.loads(label_path.read_text()) == label
