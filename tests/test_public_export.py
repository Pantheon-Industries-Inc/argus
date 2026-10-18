"""Privacy belongs at every JSON boundary, after source/freshness interpretation."""
import copy
import json
import base64

import pytest

from board.public_export import project, public_build_manifest


def test_credential_values_and_access_attributes_do_not_escape_under_native_names():
    def segment(value):
        return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip('=')
    jwt = '.'.join((segment({'alg': 'HS256', 'typ': 'JWT'}), segment({'sub': 'fixture'}), 'signature'))
    secrets = [jwt, '-----BEGIN PRIVATE KEY-----\nfixture\n-----END PRIVATE KEY-----',
               'AKIA' + 'A' * 16, 'ghp_' + 'a' * 36, 'sk-ant-' + 'a' * 40,
               'xoxb-' + '1' * 12 + '-' + '2' * 12 + '-' + 'a' * 24,
               'Bearer unusable-fixture', 'Authorization: Token unusable-fixture',
               'Connect with Authorization: Bearer unusable-fixture, then inspect the image.',
               'https://discord.com/api/webhooks/123456789/' + 'a' * 60,
               'camera.example-tailnet.ts.net/path', 'operator@100.101.7.9']
    secrets.append('Basic ' + base64.b64encode(b'fixture:unusable').decode())
    names = ('jwt', 'csrf', 'sentry_dsn', 'connection_string', 'otp_seed', 'pass', 'pwd',
             'privateKeyPem', 'secret_key_base', 'api_keys', 'secrets', 'tokens')
    native = {'session_id': 'session-17', 'session': 4, 'signature': [0.1, 0.2], 'sig': 3,
              'eos_token': '</s>', 'pad_token': '<pad>', 'stop_token': '<end>',
              **{key + '_key': '/observations/' + key for key in
                 ('task', 'image', 'video', 'language', 'episode', 'primary')},
              'firmware_version': '10.3.0.17', 'token_ids': [1, 7],
              'key': 'a' * 32,
              'task_name': 'Basic calibration', 'episode_prompt': 'Basic grasp',
              'citation': 'https://example.org/paper',
              'text': 'The gripper does not close. Type api_key on the panel.'}
    raw = {'native': native, 'unknown': secrets,
           'credentials': {name: 'unusable-fixture' for name in names},
           'attributes': {name: 'unusable-fixture' for name in names},
           'service': {'kaggle': {'key': 'a' * 32}},
           'metadata': {'fields': {'eos_token': '</s>', 'api_key': 'unusable-fixture'}},
           'sentence': 'Read https://example.org/paper. Authorization: Token unusable-fixture'}
    before = copy.deepcopy(raw)
    public = project(raw)
    assert public['native'] == native
    assert public['unknown'] == [None] * len(secrets)
    assert public['attributes'] == {} and public['service'] == {'kaggle': {}}
    assert public['metadata'] == {'fields': {'eos_token': '</s>'}}
    assert 'sentence' not in public
    assert raw == before and project(public) == public
    from board.build import uploader_groups
    note = {'kaggle': {'key': 'a' * 32}}
    flattened = project({'uploader_notes': uploader_groups(note, None, 10)})
    assert 'a' * 32 not in json.dumps(flattened)
    field = {'id': 'service-key', 'bindings': [{'context_path': 'uploader_notes/kaggle/key'}],
             'summary': {'preview': 'a' * 32}}
    dictionary = project({'fields': [field], 'entries': {'service-key': {'meaning': 'a' * 32}}})
    assert 'a' * 32 not in json.dumps(dictionary)


@pytest.mark.parametrize('outcome,kind,heading,reason', [
    ('not dispatched', 'spend_cap', 'Annotation not sent', 'remaining spend cap'),
    ('not dispatched', None, 'Annotation not sent', None),
    ('claimed', None, 'Response outcome unknown', None),
    ('unverified', None, 'Response outcome unknown', None),
    (None, None, 'No response from Astra', None),
    ('private /srv/request?token=secret', 'private /srv/request?token=secret', 'No response from Astra', None),
])
def test_billing_refusal_reaches_viewer_without_exposing_private_error(outcome, kind, heading, reason):
    import shutil
    import subprocess
    from pathlib import Path
    from board.to_board import convert

    node = shutil.which('node')
    if not node:
        pytest.skip('no node')
    record = {'episode_dir': '/srv/private/episode_demo_00', 'model': 'openai/gpt-6-astra',
              'reasoning_effort': 'medium', 'parse_ok': False,
              'no_reply': 'final request reserve $6.44 exceeds remaining spend cap /srv/private?token=secret',
              'final_reserved_usd': 0.0, 'final_dispatch_outcome': outcome, 'no_reply_kind': kind,
              'evidence_inspection': {'version': 1, 'cost_usd': 0.021, 'status': 'complete'}}
    before = copy.deepcopy(record)
    shown = project(convert(record, 'numeric'))
    assert record == before
    assert shown['_meta']['model'] == 'openai/gpt-6-astra'
    assert shown['evidence_inspection']['cost_usd'] == 0.021
    failure = shown['_label_failed']
    assert failure['status'] == 'no_reply'
    assert failure['why'] == 'No verified model reply was recorded.'
    if outcome in ('claimed', 'unverified', 'not dispatched'):
        assert failure['final_dispatch_outcome'] == outcome
    else:
        assert 'final_dispatch_outcome' not in failure
    assert ('no_reply_kind' in failure) == (kind == 'spend_cap')
    # Projection also validates typed status fields on older directly written board files.
    assert project({'_label_failed': {'status': 'no_reply', 'why': record['no_reply'],
                    'final_dispatch_outcome': outcome, 'no_reply_kind': kind}})['_label_failed'] == failure
    part = {'part': 1, 't0_s': 0, 't1_s': 4.199, 'why': record['no_reply'],
            'final_dispatch_outcome': outcome, 'no_reply_kind': kind}
    split = project({'_label_failed': {'status': 'no_part', 'parts': [part]},
                     '_stitched': {'parts': 1, 'missing': [part]}})
    public_part = split['_label_failed']['parts'][0]
    assert public_part == split['_stitched']['missing'][0]
    assert public_part['why'] == 'No verified labels were recorded for this part.'
    assert ('final_dispatch_outcome' in public_part) == (outcome in ('claimed', 'unverified', 'not dispatched'))
    assert ('no_reply_kind' in public_part) == (kind == 'spend_cap' and outcome == 'not dispatched')
    assert project(split) == split
    assert 'token=secret' not in json.dumps(split)
    program = """
const {piece} = require('./tests/page_functions')('./board/serve.py');
const render = new Function(piece('esc') + piece('fmtDur') + piece('cmpFailHtml')
  + piece('noLabelsHtml') + 'return {cmpFailHtml, noLabelsHtml};')();
const d = JSON.parse(require('fs').readFileSync(0, 'utf8'));
console.log(JSON.stringify({card:render.cmpFailHtml(d._label_failed,'Astra',d._usage,true),
                           sections:render.noLabelsHtml(d._label_failed)}));
"""
    repo = Path(__file__).resolve().parent.parent
    completed = subprocess.run([node, '-e', program], input=json.dumps(shown), cwd=repo,
                               capture_output=True, text=True, check=True)
    rendered = json.loads(completed.stdout)
    assert heading in rendered['card']
    if reason:
        assert reason in rendered['card']
    if outcome == 'not dispatched':
        assert 'No response' not in rendered['card'] and 'got no reply' not in rendered['sections']
    if outcome in ('claimed', 'unverified'):
        assert 'unknown' in rendered['sections'] and 'Annotation not sent' not in rendered['card']
    assert 'footage, checks and sensors' in rendered['card']
    assert 'token=secret' not in completed.stdout and '/srv/private' not in completed.stdout
    assert 'not called' not in completed.stdout


def test_nested_access_is_private_without_changing_source_or_annotation():
    label = {
        'publisher_labels': {'text': 'Open /data/example, then type api_key.',
                             'path': 'left to right'},
        'dataset_source': {'url': 'https://example.org/data?version=2', 'publisher': 'Example'},
        'data_dictionary': {'fields': [{'source_path': '/srv/uploads/episode/signals.npz',
                                       'source': 'signals.npz#force', 'shape': [3, 2]}]},
        'evidence_inspection': {'parts': [{'source_location': 'pieces/episode__p00',
                                          'record': {'source_current': True},
                                          'descriptor': {'url': 'https://example.org/o?X-Amz-Signature=secret',
                                                         'unit': 'mm'}}]},
        'sensor_evidence': {'sensors': [{'source': {'path': '/srv/source.h5',
                                                  'key': '/observations/force',
                                                  'url': 'https://example.org/source'}}]},
        'nested': [{'_run': {'run_id': 'private'}, 'credentials': {'password': 'private'},
                    'access_descriptor_digest': 'a' * 64, 'source_proof': [{'sha256': 'b' * 64}]}],
        '_compare': {'run_id': 'private', 'code': 'private', 'name': 'Other model'},
    }
    before = copy.deepcopy(label)
    public = project(label)
    assert public['publisher_labels'] == label['publisher_labels']
    assert public['dataset_source'] == label['dataset_source']
    assert public['data_dictionary']['fields'] == [{'source': 'signals.npz#force', 'shape': [3, 2]}]
    assert public['evidence_inspection']['parts'] == [
        {'record': {'source_current': True}, 'descriptor': {'unit': 'mm'}}]
    assert public['sensor_evidence']['sensors'][0]['source'] == {
        'key': '/observations/force', 'url': 'https://example.org/source'}
    assert public['nested'] == [{'access_descriptor_digest': 'a' * 64,
                                'source_proof': [{'sha256': 'b' * 64}]}]
    assert public['_compare'] == {'name': 'Other model'}
    assert label == before
    public['publisher_labels']['text'] = 'changed'
    assert label == before
    assert project(project(before)) == project(before)


@pytest.mark.parametrize('location', [
    '/mnt/private/source.h5', r'C:\Users\owner\source.h5', r'\\server\share\source.h5',
    '../uploads/source.h5', '~/uploads/source.h5', 'file:///srv/source.h5',
    's3://private-bucket/source.h5', 'gs://private-bucket/source.h5',
    'file:/srv/private/source.h5',
])
def test_typed_locations_are_removed(location):
    assert project({'provenance': [{'source_path': location}], 'source': location}) == {
        'provenance': [{}]}


@pytest.mark.parametrize('url', [
    'https://example.org/object?X-Amz-Signature=secret',
    'https://example.org/object?x-goog-credential=secret',
    'https://example.org/object?sig=secret&sv=2025',
    'https://example.org/object?access_token=secret',
    'https://example.org/object#token=secret',
    'https://user:secret@example.org/object',
    'http://127.0.0.1/object', 'http://10.0.0.1/object',
    'http://[::1]/object', 'http://storage.internal/object',
])
def test_unsafe_urls_in_typed_fields_and_arrays(url):
    public = project({'source_url': url, 'descriptor': {'urls': [url, 'https://example.org/paper']}})
    assert public == {'descriptor': {'urls': [None, 'https://example.org/paper']}}


def test_preserves_array_positions_and_schema_values():
    data = {'values': [None, 0, False, [1, 2]], 'source': {'context_path': '/signals/0',
            'key': '/hdf/channel', 'file': 'data/chunk-000.parquet'},
            'provenance': 'human', 'access_kind': 'depth', 'source_current': False}
    assert project(data) == data
    assert project({'url': ['https://example.org/a?token=secret', 'https://example.org/b']}) == {
        'url': [None, 'https://example.org/b']}


def test_secret_key_spelling_does_not_bypass_nested_projection():
    assert project({'descriptor': [{'apiKey': 'secret', 'Authorization': 'Bearer secret',
                                   'access-token': 'secret', 'Cookie': 'secret', 'secret_key': 'secret',
                                   'shape': [2], 'dtype': 'float32'}]}) == {
        'descriptor': [{'shape': [2], 'dtype': 'float32'}]}


def test_standalone_access_url_is_typed_even_under_an_extension_field():
    assert project({'extension': ['https://example.org/object?token=secret'],
                    'description': 'Read https://example.org/paper for the task.'}) == {
        'extension': [None], 'description': 'Read https://example.org/paper for the task.'}


def test_public_build_manifest_contains_counts_and_no_operational_locations():
    build = {'build_id': 'demo', 'built_at': '2026-10-11', 'code': 'a' * 40,
             'qa': '/srv/private/qa', 'clips': '/srv/private/clips',
             'enc': 'h264', 'public_base': 'https://example.org/board/', 'episodes': 3,
             'datasets': {'example': 3}, 'compare': {'dir': '/srv/private/compare', 'files': 2},
             'hands': {'dir': '/srv/private/hands', 'files': 1,
                       'skipped': {'ep.json': 'cannot read /srv/private/keypoints.npz'}},
             'sensors': {'dir': '/srv/private/sensors', 'files': 3},
             'keypoints': {'dir': '/srv/private/keypoints', 'files': 3, 'bytes': 123},
             'media': {'videos': 3, 'videos_present': 2}, 'missing_media': ['v/example.mp4'],
             'future_private_field': '/srv/private/future'}
    before = copy.deepcopy(build)
    public = public_build_manifest(build)
    assert public['compare'] == {'files': 2}
    assert public['hands'] == {'files': 1, 'skipped': {'ep.json': 'unavailable'}}
    assert public['keypoints'] == {'files': 3, 'bytes': 123}
    assert public['media'] == build['media']
    assert public['missing_media'] == build['missing_media']
    assert public['code'] == build['code']
    assert '/srv/private' not in json.dumps(public)
    assert 'future_private_field' not in public and 'qa' not in public and 'clips' not in public
    assert build == before


def test_flattened_source_note_owners_are_private_but_native_fields_and_prose_survive(tmp_path):
    from board.build import uploader_groups
    from prepare.formats import h5_text
    import h5py
    import numpy as np
    private = '/srv/private-upload/metadata/info.json'
    data = '/srv/private-upload/data/chunk-000/episode.parquet'
    native = '/observations/right/force'
    prose = 'Open /data/example, then type api_key.'
    workstation = '/srv/publisher-workstation/recordings'
    signed = 'https://example.org/asset?X-Amz-Signature=unusable-url-test'
    citation = 'https://example.org/public-paper'
    source = tmp_path / 'native.h5'
    with h5py.File(source, 'w') as f:
        f.attrs['source_directory'] = workstation
        f.attrs['source_location'] = 'SSH:user:unusable-uri-credential@private-host'
        f.attrs['api_key'] = 'unusable-attribute-value'
        f.attrs['source'] = native
        f.attrs['source_url'] = 'file:///srv/private-env/source.h5'
        f.attrs['description'] = prose
        f.attrs['token_count'] = 17
        f.attrs['asset_urls'] = [signed, citation]
        f.attrs['env_args'] = np.array(('/srv/private-env/config.json',
            'file:///srv/private-env/source.h5', native, prose), dtype=[
            ('path', 'S64'), ('source_url', 'S64'), ('native_key', 'S64'), ('description', 'S80')])[()]
        episode = f.create_group('episode')
        episode.attrs['path'] = native
        episode.attrs['source_location'] = np.array(([workstation],),
            dtype=[('recording', 'S64', (1,))])[()]
        episode.attrs['credentials'] = np.array(('unusable-attribute-access',),
            dtype=[('access_id', 'S64')])[()]
    source_before = source.read_bytes()
    with h5py.File(source, 'r') as f:
        _, attributes = h5_text(f, 'episode', [])
    notes = {'recorded episode metadata': {private: {'source': private,
              'fields': {native: 12, 'instruction': prose, 'path': native,
                         'source': '/publisher/source_identifier',
                         'unknown_extension_url': '/publisher/native_identity',
                         'api_key': 'unusable-test-value',
                         'credentials': {'access_id': 'unusable-other-value'},
                         'paths': [native, '/observations/left/force'],
                         'token_count': 17, 'unknown_extension_token': 'native token identity'}}},
             'recorded annotation data source': data,
             'recorded annotation tables': {private: {'native_step_index': {'claims': {'0': [
                 {'row': 0, 'source': private, 'fields': {'path': native,
                  'source': '/publisher/source_identifier'}}]}}}},
             **attributes}
    groups = uploader_groups(notes, None, 10)
    before = copy.deepcopy(groups)
    # Existing saved boards have only flattened name/value entries.
    legacy = [{**g, 'items': [{k: v for k, v in item.items() if k != '_uploader_field_path'}
                            for item in g['items']]} for g in groups]
    for input_groups in (groups, legacy):
        shown = project({'uploader_notes': input_groups})['uploader_notes']
        assert private not in json.dumps(shown) and data not in json.dumps(shown)
        assert native in json.dumps(shown) and prose in json.dumps(shown)
        assert '/publisher/source_identifier' in json.dumps(shown)
        assert '/publisher/native_identity' in json.dumps(shown)
        assert 'unusable-test-value' not in json.dumps(shown)
        assert 'unusable-other-value' not in json.dumps(shown)
        assert 'native token identity' in json.dumps(shown)
        assert '/observations/left/force' in json.dumps(shown)
        assert workstation not in json.dumps(shown)
        assert 'unusable-uri-credential' not in json.dumps(shown)
        assert 'unusable-attribute-value' not in json.dumps(shown)
        assert 'unusable-attribute-access' not in json.dumps(shown)
        assert signed not in json.dumps(shown) and citation in json.dumps(shown)
        assert '/srv/private-env' not in json.dumps(shown)
        assert any(row['name'] == 'file attribute token_count' and row['value'] == '17'
                   for group in shown for row in group['items'])
        assert any('info.json' in row['name'] for group in shown for row in group['items'])
        assert project({'uploader_notes': shown}) == {'uploader_notes': shown}
    raw = project({'recorded_metadata': {'native.h5': attributes}})['recorded_metadata']['native.h5']
    assert workstation not in json.dumps(raw)
    assert 'unusable-uri-credential' not in json.dumps(raw)
    assert '/srv/private-env' not in json.dumps(raw)
    assert raw['file attribute env_args']['native_key'] == native
    assert raw['file attribute env_args']['description'] == prose
    assert raw['file attribute source'] == native and raw['episode attribute path'] == native
    assert raw['file attribute description'] == prose and raw['file attribute token_count'] == 17
    assert 'file attribute api_key' not in raw and 'episode attribute credentials' not in raw
    native_map = project({'recorded_metadata': {private: {'fields': {'path': native,
        'source': '/publisher/source_identifier', 'instruction': prose}}}})['recorded_metadata']
    assert private not in json.dumps(native_map)
    assert native_map['info.json']['fields']['path'] == native
    assert native_map['info.json']['fields']['source'] == '/publisher/source_identifier'
    assert native_map['info.json']['fields']['instruction'] == prose
    import hashlib
    collision = 'info.json#' + hashlib.sha256(private.encode()).hexdigest()[:12]
    owners = [(private, {'value': 1}), ('info.json', {'value': 2}),
              (collision, {'value': 3}), (data.rsplit('/', 1)[0] + '/info.json', {'value': 4})]
    for order in (owners, list(reversed(owners))):
        value = {'recorded_metadata': dict(order)}
        public = project(value)
        assert len(public['recorded_metadata']) == 4
        assert {row['value'] for row in public['recorded_metadata'].values()} == {1, 2, 3, 4}
        assert project(public) == public
    assert source.read_bytes() == source_before
    large = uploader_groups({'asset_urls': [signed, citation] * 20}, None, 1)
    receipt = large[0]['items'][0]['_uploader_location_values']
    assert receipt['count'] == 40 and len(receipt['values']) == 4
    assert signed not in json.dumps(project({'uploader_notes': large}))
    large[0]['items'][0].pop('_uploader_location_values')
    large[0]['items'][0].pop('_uploader_field_path')
    assert signed not in json.dumps(project({'uploader_notes': large}))
    assert '40 values' in project({'uploader_notes': large})['uploader_notes'][0]['items'][0]['value']
    locations = uploader_groups({'source_paths': [private, data]}, None, 1)
    displayed = project({'uploader_notes': locations})
    assert private not in json.dumps(displayed) and 'info.json' in json.dumps(displayed)
    assert groups == before and notes['recorded episode metadata'][private]['fields'][native] == 12
    plain = project({'uploader_notes': uploader_groups(prose, None, 10)})
    assert plain['uploader_notes'][0]['items'] == [{'name': 'note', 'value': prose}]
    upper_signed = signed.replace('https://', 'HTTPS://')
    uri_notes = {'native': {'fields': {'references': [upper_signed, citation],
                                     'links': [upper_signed, citation],
                                     'asset_urls': [upper_signed, native, citation],
                                     'primitive_values': [upper_signed, 0, True, native, citation],
                                     'protocol_relative': [upper_signed.removeprefix('HTTPS:'), native, citation]}},
                 'asset_urls': [upper_signed, native, citation],
                 'publisher_note': 'Read ' + upper_signed + ' for the task.'}
    original_uri_notes = copy.deepcopy(uri_notes)
    uri_groups = uploader_groups(uri_notes, None, 1)
    original_uri_groups = copy.deepcopy(uri_groups)
    uri_public = project({'uploader_notes': uri_groups})
    for group in uri_public['uploader_notes']:
        for item in group['items']:
            if item['name'] == 'publisher_note':
                assert item['value'] == 'withheld'
            else:
                assert upper_signed not in item['value'] and citation in item['value']
                if item['name'].endswith(('asset_urls', 'primitive_values', 'protocol_relative')):
                    assert native in item['value']
                if item['name'].endswith('primitive_values'):
                    assert '0, True' in item['value']
                assert 'X-Amz-Signature' not in item['value']
    assert uri_notes == original_uri_notes and uri_groups == original_uri_groups
    legacy_uri = uploader_groups({'native': {'fields': {'references': [upper_signed, citation],
                              'mixed': [upper_signed, native, citation]}}}, None, 1)
    for group in legacy_uri:
        for item in group['items']:
            item.pop('_uploader_location_values', None)
            item.pop('_uploader_field_path', None)
    legacy_public = project({'uploader_notes': legacy_uri})
    legacy_items = {item['name']: item['value'] for group in legacy_public['uploader_notes'] for item in group['items']}
    assert upper_signed not in legacy_items['native / fields / references']
    assert upper_signed not in legacy_items['native / fields / mixed']
    assert native in legacy_items['native / fields / mixed']
    uri_large = uploader_groups({'native': {'fields': {'references': [upper_signed, native, citation] * 20}}}, None, 1)
    uri_receipt = uri_large[0]['items'][0]['_uploader_location_values']
    assert uri_receipt['count'] == 60 and len(uri_receipt['values']) == 4
    assert upper_signed not in json.dumps(project({'uploader_notes': uri_large}))
    assert native in json.dumps(project({'uploader_notes': uri_large}))
    for group in uri_large:
        for item in group['items']:
            item.pop('_uploader_location_values', None)
            item.pop('_uploader_field_path', None)
    legacy_large = project({'uploader_notes': uri_large})
    assert upper_signed not in json.dumps(legacy_large) and native in json.dumps(legacy_large)
    assert '60 values' in json.dumps(legacy_large)
    access_uris = ['  ' + upper_signed + '  ', '//example.org/asset?X-Amz-Signature=unusable-url-test',
                   'FILE:/srv/private/source.h5', 'S3://private-bucket/source.h5',
                   'GS://private-bucket/source.h5', 'AZ://private-container/source.h5',
                   'ABFS://private-container/source.h5', 'SSH://private-host/source.h5',
                   'SFTP://private-host/source.h5', 'FTP://private-host/source.h5']
    assert project({'extension': access_uris}) == {'extension': [None] * len(access_uris)}
    native_identifiers = [native, 'urn:publisher:source', 'publisher:source_identity', citation,
                          'Open FILE:/srv/private/source.h5 for the task.']
    assert project({'extension': native_identifiers}) == {'extension': native_identifiers}
    opaque_uris = ['SSH:user:unusable-uri-credential@private-host',
                   'FTP:user:unusable-uri-credential@private-host',
                   'FILE:unusable-uri-credential', 'S3:unusable-uri-credential',
                   'GS:unusable-uri-credential', 'AZ:unusable-uri-credential',
                   'ABFS:unusable-uri-credential', 'SFTP:unusable-uri-credential',
                   'SSH%3Auser%3Aunusable-uri-credential%40private-host',
                   'SSH%3Auser%3Aunusable-uri-credential%20encoded%40private-host',
                   'HTTPS%3A%2F%2Fexample.org%2Fasset%3FX-Amz-Signature%3Dunusable-uri-credential']
    opaque_groups = uploader_groups({'native': {'fields': {'references': [*opaque_uris, native, citation]}}}, None, 1)
    for historical in (False, True):
        value = copy.deepcopy(opaque_groups)
        if historical:
            for group in value:
                for item in group['items']:
                    item.pop('_uploader_location_values', None)
                    item.pop('_uploader_field_path', None)
        public = project({'uploader_notes': value})
        assert 'unusable-uri-credential' not in json.dumps(public)
        assert native in json.dumps(public) and citation in json.dumps(public)
    for uri in [*opaque_uris, 'HTTP://[', 'SSH://[']:
        raw_owner = {'recorded_metadata': {uri: {'source': uri, 'fields': {'path': native}}},
                     'file attribute source_directory': uri}
        public = project(raw_owner)
        assert 'unusable-uri-credential' not in json.dumps(public)
        assert native in json.dumps(public)
        assert project(public) == public


def test_reader_metadata_limit_paths_use_typed_source_identity_without_rewriting_publisher_text(tmp_path, monkeypatch):
    path = '/srv/private-upload/data/chunk-000/episode.parquet'
    message = (path + ' and its recorded metadata exceed the prompt note limit; complete original rows '
               'and source claims remain in the episode context and uploader notes.')
    label = {'dataset_checks': {'reader_issues': [{'kind': 'metadata_limit', 'what': message,
             'family': 'metadata-unreadable'}]}, 'publisher_labels': {'text': message}}
    unresolved = (f'task_index in {path} has unresolved recorded annotation codes; '
                  'all original rows and table claims are retained as uploader notes.')
    label['dataset_checks']['reader_issues'].append({'kind': 'metadata_unreadable',
        'what': unresolved, 'family': 'metadata-unreadable'})
    label['publisher_labels']['unresolved_native_text'] = unresolved
    before = copy.deepcopy(label)
    shown = project(label)
    issue = shown['dataset_checks']['reader_issues'][0]
    assert path not in issue['what'] and 'episode.parquet' in issue['what']
    assert issue['kind'] == 'metadata_limit' and issue['family'] == 'metadata-unreadable'
    assert shown['publisher_labels']['text'] == message and label == before
    annotation_issue = shown['dataset_checks']['reader_issues'][1]
    assert path not in annotation_issue['what'] and 'task_index in episode.parquet' in annotation_issue['what']
    assert annotation_issue['kind'] == 'metadata_unreadable'
    assert shown['publisher_labels']['unresolved_native_text'] == unresolved
    for head, detail in [('meta/codes.jsonl could not be fully read: ', 'table read failed ('),
                         (path + ' retains an unassigned metadata claim: ', 'metadata table read failed (')]:
        native = head + detail + "[Errno 2] '/srv/other-cache/input.json')."
        issue = {'kind': 'metadata_unreadable', 'what': native, 'source': path}
        public = project({'dataset_checks': {'reader_issues': [issue]}, 'publisher_labels': {'text': native}})
        assert '/srv/other-cache' not in public['dataset_checks']['reader_issues'][0]['what']
        assert public['publisher_labels']['text'] == native
    from prepare import formats
    context = {}
    for _ in range(2):
        formats.add_issue(context, 'metadata_limit', 'episode.parquet exceeds the note limit.', source=path)
    assert len(context['reader_issues']) == 1
    assert context['reader_issues'][0]['source'] == path
    formats.add_issue(context, 'metadata_limit', 'episode.parquet exceeds the note limit.',
                      source='/srv/other-upload/episode.parquet')
    assert len(context['reader_issues']) == 2
    public = project({'dataset_checks': context})
    assert path not in json.dumps(public)
    # Exercise the actual annotation reader twice, not just its issue constructor.
    import pandas as pd
    (tmp_path / 'meta').mkdir()
    data = tmp_path / 'episode.parquet'
    frame = pd.DataFrame({'timestamp': [0., 1., 2.], 'frame_index': [0, 1, 2],
                          'native_step': ['one', 'one', 'one'], 'task_index': [91, 91, 92]})
    frame.to_parquet(data)
    index = {'dir': tmp_path, 'fps': 1, 'metadata_read': set()}
    context = {'episode_index': 0, 'fps': 1, 'n_state_frames': 3}
    monkeypatch.setattr(formats, 'ANNOTATION_MAX_CHARS', 1)
    for _ in range(2):
        formats.lerobot_annotations(context, index, frame, data)
    issues = [issue for issue in context['reader_issues']
              if issue['kind'] == 'metadata_limit' and issue.get('source') == str(data)]
    assert len(issues) == 2 and all(str(tmp_path) not in issue['what'] or
                                   'has unresolved recorded annotation codes' in issue['what'] for issue in issues)
    note_limits = [issue for issue in issues if 'exceed the prompt note limit' in issue['what']]
    assert len(note_limits) == 1 and str(tmp_path) not in note_limits[0]['what']
    unresolved_issues = [issue for issue in context['reader_issues']
        if 'has unresolved recorded annotation codes' in issue['what']]
    assert len(unresolved_issues) == 1 and unresolved_issues[0].get('source') == str(data)
    assert unresolved_issues[0]['kind'] == 'metadata_limit'
    assert str(data) in unresolved_issues[0]['what']
    assert [claim['value'] for claim in context['annotation_unresolved']] == [91, 91, 92]
    assert [claim['raw_time'] for claim in context['annotation_unresolved']] == [0., 1., 2.]
    assert all(claim['data_source'] == str(data) for claim in context['annotation_unresolved'])
    assert any(step['label'] == 'one' for step in context['annotation_subtasks'])
    from board.families import Families
    assert not Families().classify({'dataset_checks': {'reader_issues': issues}})['counted']
    assert str(data) not in json.dumps(project({'dataset_checks': {'reader_issues': context['reader_issues']}}))


def test_actual_reader_and_check_errors_keep_private_details_and_public_identity(tmp_path):
    from board import build, clips, serve
    from checks import capture_qc, sensors, stream_pairing
    from prepare import annotations, formats
    episode = tmp_path / 'episode'
    episode.mkdir()
    (episode / 'context.json').mkdir()
    _, why = build.read_context(episode)
    context_issue = build.context_unreadable(why)
    try:
        (tmp_path / 'private-source' / 'missing_depth.mkv').read_bytes()
    except OSError as error:
        depth_issue = clips.depth_failed(tmp_path, 'exo', error)
    meta = tmp_path / 'metadata'
    meta.mkdir()
    (meta / 'unreadable.jsonl').mkdir()
    _, claims = annotations.episode_metadata(meta, formats.read_jsonl)
    context = {'episode_index': 0}
    formats.lerobot_metadata(context, {'dir': tmp_path, 'unassigned_metadata': claims})
    groups = build.uploader_groups(context['uploader_notes'], None, 1)
    missing = tmp_path / 'missing_episode'
    qc = capture_qc.run_episode(missing)
    _, sensor, _ = sensors._safe(str(missing))
    _, pairing, _ = stream_pairing._safe('pairing', str(missing))
    label = {'dataset_checks': {'reader_issues': [context_issue, depth_issue],
        'capture_qc': qc, 'sensor_checks': sensor, 'stream_pairing': pairing,
        'fired_with_supplement': {'status': 'fired', 'why': 'Three gaps were measured.', 'error': why, 'rows': [1, 2]},
        'fired_partial': {'status': 'fired', 'why': 'Native check reason. It was not run on every camera, as the check stopped with an error (' + why + ').',
                          'unchecked': 'it was not run on every camera, as the check stopped with an error (' + why + ')',
                          'rows': [3, 4]},
        'decode_flag': {'check': 'video_decode_failure', 'camera': 'left', 't_s': 2.0,
                        'evidence': 'camera left fails to decode: ' + why},
        'camera_metrics': {'exo': {'decoded': 30, 'mean': 0}, 'left': {'decode_error': why}}},
        'set_aside_checks': [{'check': 'stream_pairing', **serve.withheld_status(pairing)}],
        'uploader_notes': groups, 'native_context': context['uploader_notes'],
        '_label_failed': {'status': 'not_shown', 'error': why, 'raw_head': 'Native model prose.'},
        'data_dictionary': {'fields': [{'id': 'original-field-id', 'source': 'native source',
            'bindings': [{'episode': 'episode', 'context_path': 'recorded_metadata/~1srv~1private~1info.json/fields/key',
                          'file': 'signals.npz', 'key': '/observations/right/force'},
                         {'episode': 'episode', 'context_path': 'native/fields/~1observations~1right~1force'}]}]},
        'publisher_labels': {'text': why, 'status': 'errored', 'error': why,
            'fields': {'dataset_checks': {'status': 'errored', 'error': why,
                'why': 'it was not run on every camera, as the check stopped with an error (' + why + ')'},
                'reader_issues': [{'kind': 'depth_clip_failed', 'what': why}],
                'uploader_notes': [{'kind': 'notes', 'items': [{'name': 'file attribute source_directory', 'value': why}]}]}}}
    before = copy.deepcopy(label)
    for historical in (False, True):
        value = copy.deepcopy(label)
        if historical:
            for issue in value['dataset_checks']['reader_issues']:
                for key in ('_reader_public_what', '_reader_public_error'):
                    issue.pop(key, None)
            for group in value['uploader_notes']:
                for item in group['items']:
                    item.pop('_uploader_field_path', None)
        shown = project(value)
        assert str(tmp_path) not in json.dumps({key: item for key, item in shown.items() if key != 'publisher_labels'})
        assert shown['publisher_labels'] == label['publisher_labels']
        assert shown['dataset_checks']['reader_issues'][1]['camera'] == 'exo'
        assert shown['set_aside_checks'][0]['status'] == 'errored'
        assert shown['dataset_checks']['fired_with_supplement']['why'] == 'Three gaps were measured.'
        assert shown['dataset_checks']['fired_with_supplement']['rows'] == [1, 2]
        assert shown['dataset_checks']['fired_partial']['status'] == 'fired'
        assert shown['dataset_checks']['fired_partial']['rows'] == [3, 4]
        assert shown['dataset_checks']['fired_partial']['why'].startswith('Native check reason.')
        assert shown['dataset_checks']['decode_flag']['camera'] == 'left'
        assert shown['dataset_checks']['decode_flag']['t_s'] == 2.0
        assert shown['dataset_checks']['decode_flag']['evidence'].startswith('camera left fails to decode;')
        assert shown['dataset_checks']['camera_metrics']['exo'] == {'decoded': 30, 'mean': 0}
        assert shown['_label_failed']['raw_head'] == 'Native model prose.'
        field = shown['data_dictionary']['fields'][0]
        assert field['id'] == 'original-field-id' and field['source'] == 'native source'
        assert field['bindings'][0] == {'episode': 'episode', 'file': 'signals.npz', 'key': '/observations/right/force'}
        assert field['bindings'][1]['context_path'] == 'native/fields/~1observations~1right~1force'
    assert label == before and str(tmp_path) in context_issue['error']


def test_actual_hdf_dictionary_credentials_keep_descriptors_and_native_signal_values(tmp_path):
    import hashlib
    import io
    import h5py
    import numpy as np
    from PIL import Image
    from prepare import formats
    from label import dictionary, dictionary_context, dictionary_editor
    upload = tmp_path / 'upload'
    upload.mkdir()
    source = upload / 'native.h5'
    image = io.BytesIO()
    Image.new('RGB', (64, 64), (40, 90, 160)).save(image, format='JPEG')
    pixels = np.frombuffer(image.getvalue(), np.uint8)
    with h5py.File(source, 'w') as f:
        f.attrs['api_key'] = 94281376
        f.attrs['token'] = [77, 81, 89]
        f.attrs['token_count'] = 17
        f.attrs['sampling_frequency'] = 30
        group = f.create_group('data/demo_0')
        group.attrs['credentials'] = np.array((94281376,), dtype=[('access_id', 'i8')])[()]
        camera = group.create_dataset('rgb_images_jpeg', (3,), dtype=h5py.vlen_dtype(np.dtype('uint8')))
        for row in range(3):
            camera[row] = pixels
        group.create_dataset('timestamps', data=np.arange(3) / 30)
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    report = formats.convert(upload, 'ego_head', tmp_path / 'episodes', 'native_attributes', 900)
    assert not report['failed'] and len(report['episodes']) == 1
    owner = report['episodes'][0]['episode_id']
    episode = tmp_path / 'episodes' / owner
    context = json.loads((episode / 'context.json').read_text())
    # A scientific signal named token has a signal binding, not credential ownership.
    np.savez(episode / 'signals.npz', token=np.array([[2.], [3.], [4.]]),
             credentials=np.array([[8.], [9.], [10.]]))
    context['signals'] = [{'name': 'token', 'file': 'signals.npz', 'key': 'token'}]
    (episode / 'context.json').write_text(json.dumps(context))
    inventory = dictionary.inventory([episode])
    secrets = {field['id'] for field in inventory['fields'] if any(
        binding.get('context_path', '').startswith(('uploader_notes/file attribute api_key',
            'uploader_notes/file attribute token/', 'uploader_notes/file attribute token',
            'uploader_notes/episode attribute credentials/'))
        and 'token_count' not in binding.get('context_path', '')
        for binding in field['bindings'])}
    assert len(secrets) >= 3
    entries = {field['id']: {'meaning': 'Recorded numeric value ' + str(field.get('summary', {}).get('mean', '')),
                            'role': 'bookkeeping'} for field in inventory['fields']}
    record = {'schema': 1, 'status': 'complete', 'inventory': inventory,
              'entries': entries, 'inventory_digest': inventory['digest']}
    (tmp_path / 'dictionary.json').write_text(json.dumps(record))
    overlay = dictionary_context.apply_context(context, record, episode_id=owner)['data_dictionary']
    private_views = [dictionary_editor.context_dictionary(overlay), dictionary_editor.public_dictionary(tmp_path, owner)]
    before = copy.deepcopy(private_views)
    for private in private_views:
        public = project(private)
        public_fields = {field['id']: field for field in public['fields']}
        for field in private['fields']:
            shown = public_fields[field['id']]
            if field['id'] in secrets:
                assert 'summary' in field and 'summary' not in shown
                for key in ('id', 'name', 'source', 'kind', 'shape', 'dtype'):
                    if key in field:
                        assert shown[key] == field[key]
                assert field['id'] not in public['entries']
                assert field['id'] not in public.get('machine_entries', {})
            else:
                assert shown.get('summary') == field.get('summary')
                assert public['entries'][field['id']] == private['entries'][field['id']]
        assert any(field['name'] == 'token' and field['summary']['mean'] == 3 for field in public['fields'])
        if any(field['name'] == 'credentials' for field in private['fields']):
            assert any(field['name'] == 'credentials' and field['summary']['mean'] == 9 for field in public['fields'])
        assert any('token_count' in field['name'] and field['summary']['mean'] == 17 for field in public['fields'])
        assert '94281376' not in json.dumps(public)
        assert project(public) == public
    assert private_views == before
    assert any(field['name'] == 'credentials' and field['summary']['mean'] == 9
               for field in project(private_views[1])['fields'])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original
