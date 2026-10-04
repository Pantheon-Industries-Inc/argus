"""Tiny dictionary provenance and editing controls with no models or media."""
from __future__ import annotations

import concurrent.futures
import contextlib
import copy
import importlib
import importlib.util
import json
import socketserver
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from board import build, serve


def editor():
    assert importlib.util.find_spec('label.dictionary_editor') is not None, 'dictionary editor is not implemented'
    return importlib.import_module('label.dictionary_editor')


def make_job(tmp_path):
    job = tmp_path / 'job'
    (job / 'units' / 'episode_a').mkdir(parents=True)
    field = {'id': 'field_a', 'name': '<img src=x onerror=window.bad=1>', 'kind': 'signal', 'shape': [2],
             'dtype': 'float32', 'names': ['left', 'right'], 'rate_hz': 30, 'source': 'signals.npz#s0',
             'episodes': ['episode_a'], 'bindings': [{'episode': 'episode_a', 'context_path': 'signals/0',
                                                       'file': 'signals.npz', 'key': 's0'}]}
    record = {'schema': 1, 'inventory_digest': 'bound_inventory', 'status': 'success', 'inventory': {'fields': [field]},
              'entries': {'field_a': {'meaning': '<script>window.bad=2</script>', 'role': 'touch',
                                       'provenance': 'machine'}}, 'raw_text': 'SECRET raw provider text',
              'owner': {'api_key': 'SECRET', 'path': '/private/context.json'}, 'limitations': []}
    (job / 'dictionary.json').write_text(json.dumps(record))
    (job / 'run' / 'out').mkdir(parents=True)
    (job / 'run' / 'out' / 'episode_a.json').write_text('{"labels":{"human":"rejected"},"raw":"untouched"}')
    (job / 'qa').mkdir()
    (job / 'qa' / 'episode_a.json').write_text(json.dumps({'episode_prompt': 'test', 'dataset': 'tiny',
          '_meta': {'episode_id': 'episode_a'}, 'completion': {'task_completed': 'failure'}, 'unknown': {'human': 'reject'}}))
    return job, record


def test_build_retains_dictionary_and_reader_left_out(tmp_path):
    job, record = make_job(tmp_path)
    d = {'completion': {'task_completed': 'failure'}, 'unknown': {'human': 'reject'}}
    before = copy.deepcopy(d)
    ctx = {'source': {'unused_arrays': ['recorded_extra.npy'], 'unused_signals': ['unaligned']}}
    build.add_context(d, ctx, job / 'units' / 'episode_a')
    assert d['data_dictionary']['entries'] == record['entries']
    assert d['data_dictionary']['machine_entries'] == record['entries']
    assert d['data_dictionary']['fields'][0]['name'] == record['inventory']['fields'][0]['name']
    assert d['reader_notes']['left_out'] == {'signals': ['unaligned'], 'arrays': ['recorded_extra.npy']}
    assert d['completion'] == before['completion'] and d['unknown'] == before['unknown']


def test_edit_cleared_role_history_and_originals(tmp_path):
    job, _ = make_job(tmp_path)
    protected = {p: p.read_bytes() for p in [job / 'dictionary.json', job / 'run/out/episode_a.json', job / 'qa/episode_a.json']}
    e = editor()
    result = e.save_override(job, {'revision': 0, 'field_id': 'field_a', 'meaning': 'Human meaning', 'role': ''}, 'owner')
    assert result['revision'] == 1 and result['entries']['field_a']['role'] == ''
    assert result['entries']['field_a']['provenance'] == 'human'
    assert result['machine_entries']['field_a']['role'] == 'touch'
    assert result['after_labelling'] is True and len(result['history']) == 1
    for p, raw in protected.items():
        assert p.read_bytes() == raw
    assert e.public_dictionary(job)['entries']['field_a']['role'] == ''


@pytest.mark.parametrize('change', [
    {'field_id': 'unknown'}, {'meaning': 'x' * 501}, {'role': 'x' * 81}, {'role': 'two\nlines'},
    {'layout': [{'start': 0, 'count': 3, 'name': 'too wide'}]},
    {'layout': [{'start': 0, 'count': 1, 'name': 'gap'}]},
    {'layout': [{'start': 0, 'count': 2, 'name': 'one'}, {'start': 1, 'count': 1, 'name': 'overlap'}]},
])
def test_invalid_edit_does_not_write(tmp_path, change):
    job, _ = make_job(tmp_path)
    e = editor()
    with pytest.raises(ValueError):
        e.save_override(job, {'revision': 0, 'field_id': 'field_a', 'role': 'state', **change}, 'owner')
    assert not (job / 'dictionary_overrides.json').exists()


def test_concurrent_revision_has_one_winner(tmp_path):
    job, _ = make_job(tmp_path)
    e = editor()
    def edit(role):
        try:
            e.save_override(job, {'revision': 0, 'field_id': 'field_a', 'role': role}, 'owner')
            return 'saved'
        except e.RevisionConflict:
            return 'conflict'
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(edit, ['state', 'event_flag'])) == ['conflict', 'saved']
    saved = json.loads((job / 'dictionary_overrides.json').read_text())
    assert saved['revision'] == 1 and len(saved['history']) == 1


def test_valid_layout_and_public_projection(tmp_path):
    job, record = make_job(tmp_path)
    record['inventory']['fields'][0]['source'] = '/private/upload/signals.npz'
    (job / 'dictionary.json').write_text(json.dumps(record))
    e = editor()
    result = e.save_override(job, {'revision': 0, 'field_id': 'field_a', 'layout': [
        {'start': 0, 'count': 1, 'name': 'left'}, {'start': 1, 'count': 1, 'name': 'right'}]}, 'private-owner')
    assert len(result['entries']['field_a']['layout']) == 2
    text = json.dumps(result)
    assert 'SECRET' not in text and '/private/' not in text and 'private-owner' not in text
    assert result['fields'][0]['name'] == record['inventory']['fields'][0]['name']
    assert e.public_dictionary(job, 'episode_other')['fields'] == []


@contextlib.contextmanager
def local_server(monkeypatch, job, enabled):
    monkeypatch.setattr(serve, 'HERE', job / 'qa')
    monkeypatch.setattr(serve, 'DICTIONARY_JOB', job if enabled else None, raising=False)
    with socketserver.ThreadingTCPServer(('127.0.0.1', 0), serve.Handler) as httpd:
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            yield 'http://127.0.0.1:' + str(httpd.server_address[1])
        finally:
            httpd.shutdown()
            thread.join()


def send(url, body=None, origin=None):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json', **({'Origin': origin} if origin else {})})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)


def test_local_edit_is_opt_in_and_same_origin(tmp_path, monkeypatch):
    job, _ = make_job(tmp_path)
    body = {'revision': 0, 'field_id': 'field_a', 'role': ''}
    with local_server(monkeypatch, job, False) as origin:
        assert send(origin + '/api/dictionary?file=episode_a.json')[1]['editable'] is False
        assert send(origin + '/api/dictionary', body, origin)[0] == 403
    with local_server(monkeypatch, job, True) as origin:
        assert send(origin + '/api/dictionary?file=episode_a.json')[1]['editable'] is True
        assert send(origin + '/api/dictionary', body, 'http://elsewhere')[0] == 403
        assert send(origin + '/api/dictionary', body, origin)[0] == 200
        assert send(origin + '/api/dictionary', body, origin)[0] == 409


def test_static_dictionary_has_safe_fold_and_no_edit_route():
    page = serve.render_index('Tiny', {'mode': 'static', 'data': 'data/'})
    assert 'Data dictionary' in page and 'dictionaryRows' in page
    assert 'dictionary_url' not in page.split('const BOARD = ', 1)[1].split(';', 1)[0]
    assert "name.textContent" in page and "meaning.textContent" in page


def test_missing_upload_receipt_uses_bound_context_overlay(tmp_path):
    ep = tmp_path / 'units' / 'episode_a'
    ep.mkdir(parents=True)
    overlay = {'schema': 1, 'status': 'success', 'override_revision': 2, 'fields': [{'id': 'field_a', 'name': 'native',
        'kind': 'signal', 'bindings': [{'episode': 'episode_a', 'key': 's0'}]}],
        'entries': {'field_a': {'meaning': 'Human interpretation', 'role': '', 'provenance': 'human'}},
        'limitations': [], 'override_limitations': []}
    before = copy.deepcopy(overlay)
    label = {'completion': {'task_completed': 'failure'}}
    build.add_context(label, {'data_dictionary': overlay}, ep)
    assert label['data_dictionary']['entries'] == before['entries']
    assert label['data_dictionary']['fields'] == before['fields']
    assert label['data_dictionary']['revision'] == 2
    assert overlay == before


def test_damaged_dictionary_does_not_stop_other_board_metadata(tmp_path):
    job, _ = make_job(tmp_path)
    (job / 'dictionary.json').write_text('{')
    label = {'completion': {'task_completed': 'failure'}}
    build.add_context(label, {'source': {'unused_arrays': ['extra.npy']}}, job / 'units/episode_a')
    assert label['data_dictionary']['status'] == 'unreadable'
    assert label['data_dictionary']['limitations']
    assert label['completion'] == {'task_completed': 'failure'}
    assert label['reader_notes']['left_out']['arrays'] == ['extra.npy']


def test_rebuilding_dictionary_display_retains_human_clear(tmp_path):
    job, _ = make_job(tmp_path)
    editor().save_override(job, {'revision': 0, 'field_id': 'field_a', 'role': ''}, 'owner')
    label = {'completion': {'task_completed': 'failure'}}
    build.add_context(label, {}, job / 'units/episode_a')
    assert label['data_dictionary']['entries']['field_a']['role'] == ''
    assert label['data_dictionary']['machine_entries']['field_a']['role'] == 'touch'
    assert len(label['data_dictionary']['history']) == 1


def test_free_request_is_not_a_saved_label(tmp_path):
    job, _ = make_job(tmp_path)
    (job / 'run/out/episode_a.json').write_text('{"dry_run":true,"prompt_text":"free"}')
    (job / 'qa/episode_a.json').unlink()
    result = editor().save_override(job, {'revision': 0, 'field_id': 'field_a', 'role': ''}, 'owner')
    assert result['after_labelling'] is False


def test_free_request_with_placeholder_board_is_not_a_saved_label(tmp_path):
    job, _ = make_job(tmp_path)
    (job / 'run/out/episode_a.json').write_text('{"dry_run":true,"prompt_text":"free"}')
    result = editor().save_override(job, {'revision': 0, 'field_id': 'field_a', 'role': ''}, 'owner')
    assert result['after_labelling'] is False


def test_local_json_content_type_is_required(tmp_path, monkeypatch):
    job, _ = make_job(tmp_path)
    with local_server(monkeypatch, job, True) as origin:
        req = urllib.request.Request(origin + '/api/dictionary', data=json.dumps({
            'revision': 0, 'field_id': 'field_a', 'role': ''}).encode(),
            headers={'Origin': origin, 'Content-Type': 'text/plain'})
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(req)
        assert error.value.code == 400
        error.value.close()
    assert not (job / 'dictionary_overrides.json').exists()


def test_piece_uses_original_dictionary_owner(tmp_path, monkeypatch):
    job, record = make_job(tmp_path)
    editor().save_override(job, {'revision': 0, 'field_id': 'field_a', 'role': ''}, 'private-human')
    piece = job / 'pieces' / 'episode_a__p01'
    piece.mkdir(parents=True)
    d = {'_meta': {'episode_id': piece.name}}
    build.add_context(d, {'piece': {'of': 'episode_a'}}, piece)
    assert d['data_dictionary']['fields'] == editor().public_dictionary(job, 'episode_a')['fields']
    assert d['data_dictionary']['episode_id'] == 'episode_a'
    assert d['data_dictionary']['entries']['field_a']['role'] == ''
    (job / 'qa' / (piece.name + '.json')).write_text(json.dumps(d))
    with local_server(monkeypatch, job, True) as origin:
        response = send(origin + '/api/dictionary?file=' + piece.name + '.json')[1]
        assert response['fields'] == d['data_dictionary']['fields']
        assert response['episode_id'] == 'episode_a'


def test_context_overlay_preserves_machine_and_public_history(tmp_path):
    ep = tmp_path / 'units' / 'episode_a'
    ep.mkdir(parents=True)
    overlay = {'schema': 1, 'episode_id': 'episode_a', 'fields': [], 'entries': {},
        'machine_entries': {'a': {'meaning': 'Original', 'role': 'state'}}, 'override_revision': 1,
        'override_limitations': ['Unknown role was not applied.'],
        'override_history': [{'revision': 1, 'field_id': 'a', 'actor': 'private-human',
            'after_labelling': True, 'after': {'role': ''}}], 'missing_fields': ['b']}
    before = copy.deepcopy(overlay)
    d = {}
    build.add_context(d, {'data_dictionary': overlay}, ep)
    assert d['data_dictionary']['machine_entries'] == overlay['machine_entries']
    assert d['data_dictionary']['history'][0]['after_labelling'] is True
    assert d['data_dictionary']['missing_fields'] == ['b']
    assert 'Unknown role was not applied.' in d['data_dictionary']['limitations']
    assert d['data_dictionary']['episode_id'] == 'episode_a'
    assert 'private-human' not in json.dumps(d)
    assert overlay == before


def test_public_source_hides_other_absolute_directories(tmp_path):
    job, record = make_job(tmp_path)
    record['inventory']['fields'][0]['source'] = '/srv/storage/upload/signals.npz'
    record['inventory']['fields'][0]['bindings'][0]['file'] = '/mnt/native/signals.npz'
    (job / 'dictionary.json').write_text(json.dumps(record))
    public = editor().public_dictionary(job)
    assert public['fields'][0]['source'] == 'signals.npz'
    assert public['fields'][0]['bindings'][0]['file'] == 'signals.npz'


def test_prefetch_uses_piece_media_identity_independently_of_dictionary(tmp_path):
    import shutil
    import subprocess
    node = shutil.which('node')
    if not node:
        pytest.skip('node is not installed')
    page = serve.render_index('Tiny', {'mode': 'static', 'data': 'data/'})
    start = page.index('async function prefetchDataset(ds)')
    end = page.index('\nfunction prefetchWhenIdle()', start)
    script = '''const _prefetched=new Set(), _prefetchImgs=[], BY=null, seen=[];
    const ensureDataset=async()=>{},searchTerms=()=>[],railEps=()=>[{file:'piece.json'}],
      datasetOf=()=> 'tiny',matchesSearch=()=>true,fetchEpisode=async()=>({
        _meta:{episode_id:'episode_a__p01'}, data_dictionary:{episode_id:'episode_a'}}),
      episodeCams=()=>({main:'exo',side:['left']}),posterSrc=(file,id,cam)=>{seen.push([file,id,cam]);return 'tiny';};
    globalThis.Image=class{};
    ''' + page[start:end] + "\nprefetchDataset('tiny').then(()=>console.log(JSON.stringify(seen)));"
    path = tmp_path / 'prefetch.js'
    path.write_text(script)
    result = subprocess.run([node, str(path)], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == [['piece.json', 'episode_a__p01', 'exo'],
                                         ['piece.json', 'episode_a__p01', 'left']]
