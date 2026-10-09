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


def test_depth_receipt_regions_render_only_on_matching_depth_view(tmp_path):
    import copy
    import re
    import shutil
    import subprocess
    node = shutil.which('node')
    if not node:
        pytest.skip('node is not installed')
    source = re.search(r'function depthRoiFrames\(.+?// each camera with depth:', serve.INDEX_HTML, re.S)
    assert source
    frame = {'time_s': 21, 'source_shape': [480, 640],
             'regions': [{'pixels': [396, 156, 422, 170]}, {'pixels': [352, 19, 416, 48]}]}
    record = {'mode': 'regions', 'descriptor': {'kind': 'depth', 'name': 'exo',
              'descriptor': {'paired_camera': 'exo'}}, 'frames': [frame]}
    from board.dictionary_projection import apply
    from label.evidence_access import source_proof
    episode = tmp_path / 'episode_a'
    episode.mkdir()
    context = {'state_kind': 'none'}
    (episode / 'context.json').write_text(json.dumps(context))
    saved = {'evidence_inspection': {'version': 1, 'source_proof': source_proof(context, episode),
                                     'inspections': [record]}}
    assert apply(copy.deepcopy(saved), context, episode)['evidence_inspection'].get('source_current') is not False
    (episode / 'context.json').write_text(json.dumps({'state_kind': 'none', 'changed': True}))
    projected = apply(copy.deepcopy(saved), context, episode)['evidence_inspection']
    assert projected['source_current'] is False and projected['inspections'] == [record]
    script = source.group().split('// each camera with depth:')[0] + '''
const assert = require('assert');
const d = {evidence_inspection: {inspections: [JSON.parse(process.argv[1])]}};
assert.equal(depthRoiFrames(d, 'exo').length, 1);
assert.equal(depthRoiFrames(d, 'left').length, 0);
const svg = depthRoiMarkup(depthRoiFrames(d, 'exo')[0]);
assert(svg.includes('viewBox="0 0 640 480"'));
assert(svg.includes('x="396" y="156" width="26" height="14"'));
assert(svg.includes('x="352" y="19" width="64" height="29"'));
assert(svg.includes('R1') && svg.includes('R2') && svg.includes('21.0s'));
d.evidence_inspection.source_current = false;
assert.equal(depthRoiFrames(d, 'exo').length, 0);
'''
    subprocess.run([node, '-e', script, json.dumps(record)], check=True, capture_output=True, text=True)


def test_stitched_depth_regions_use_piece_time_and_current_piece_proof(tmp_path):
    import re
    import shutil
    import subprocess
    from board.dictionary_projection import apply
    from label import evidence_access as ea
    node = shutil.which('node')
    if not node:
        pytest.skip('node is not installed')
    parent = tmp_path / 'eps' / 'episode_a'
    piece = tmp_path / 'pieces' / 'episode_a__p01'
    parent.mkdir(parents=True)
    piece.mkdir(parents=True)
    parent_ctx = {'state_kind': 'none', 'pieces': {'parts': [piece.name]}}
    piece_ctx = {'state_kind': 'none', 'piece': {'index': 1, 'of': parent.name, 't0_s': 5.0}}
    (parent / 'context.json').write_text(json.dumps(parent_ctx))
    (piece / 'context.json').write_text(json.dumps(piece_ctx))
    receipt = {'id': 'inspection:actual_regions', 'mode': 'regions',
               'descriptor': {'kind': 'depth', 'descriptor': {'paired_camera': 'exo'}},
               'frames': [{'time_s': 21, 'source_shape': [480, 640],
                           'regions': [{'pixels': [396, 156, 422, 170]}]}]}
    result = {'episode_dir': str(piece), 'evidence_inspection': {
        'version': 1, 'source_proof': ea.source_proof(piece_ctx, piece), 'inspections': [receipt]}}
    merged = ea.merge([(piece_ctx, result)], parent)
    assert merged['parts'][0]['source_location'] == 'pieces/' + piece.name
    source = re.search(r'function depthRoiFrames\(.+?// each camera with depth:', serve.INDEX_HTML, re.S)
    assert source
    script = source.group().split('// each camera with depth:')[0] + '''
const assert = require('assert');
const label = JSON.parse(process.argv[1]);
const frames = depthRoiFrames(label, 'exo');
assert.equal(frames.length, Number(process.argv[2]));
if (frames.length) {
  assert.equal(frames[0].time_s, 26);
  assert(depthRoiMarkup(frames[0]).includes('x="396" y="156" width="26" height="14"'));
}
'''
    fresh = apply({'evidence_inspection': copy.deepcopy(merged)}, parent_ctx, parent)
    subprocess.run([node, '-e', script, json.dumps(fresh), '1'], check=True, capture_output=True, text=True)
    changed_parent_ctx = {**parent_ctx, 'changed': True}
    (parent / 'context.json').write_text(json.dumps(changed_parent_ctx))
    changed_parent = apply({'evidence_inspection': copy.deepcopy(merged)}, changed_parent_ctx, parent)
    assert changed_parent['evidence_inspection']['source_current'] is False
    subprocess.run([node, '-e', script, json.dumps(changed_parent), '0'], check=True,
                   capture_output=True, text=True)
    (parent / 'context.json').write_text(json.dumps(parent_ctx))
    (piece / 'context.json').write_text(json.dumps({**piece_ctx, 'changed': True}))
    stale = apply({'evidence_inspection': copy.deepcopy(merged)}, parent_ctx, parent)
    assert stale['evidence_inspection']['parts'][0]['record']['source_current'] is False
    subprocess.run([node, '-e', script, json.dumps(stale), '0'], check=True, capture_output=True, text=True)
    (piece / 'context.json').unlink()
    missing = apply({'evidence_inspection': copy.deepcopy(merged)}, parent_ctx, parent)
    subprocess.run([node, '-e', script, json.dumps(missing), '0'], check=True, capture_output=True, text=True)


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
    deferred = copy.deepcopy(record['inventory']['fields'][0])
    deferred.update(id='field_b', name='recorded outcome', kind='metadata')
    record['inventory']['fields'].append(deferred)
    record.update(status='partial', request_field_ids=['field_a'], deferred_fields=['field_b'],
                  missing_fields=['field_b'])
    (job / 'dictionary.json').write_text(json.dumps(record))
    d = {'completion': {'task_completed': 'failure'}, 'unknown': {'human': 'reject'}}
    before = copy.deepcopy(d)
    ctx = {'source': {'unused_arrays': ['recorded_extra.npy'], 'unused_signals': ['unaligned']}}
    build.add_context(d, ctx, job / 'units' / 'episode_a')
    assert d['data_dictionary']['entries'] == record['entries']
    assert d['data_dictionary']['machine_entries'] == record['entries']
    assert d['data_dictionary']['fields'][0]['name'] == record['inventory']['fields'][0]['name']
    assert d['data_dictionary']['request_field_ids'] == ['field_a']
    assert d['data_dictionary']['deferred_fields'] == ['field_b']
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


def test_local_edit_projects_episode_and_jsonl_without_changing_source(tmp_path, monkeypatch):
    from label.evidence_access import source_proof
    from label.sensor_evidence import signature
    import numpy as np

    job, _ = make_job(tmp_path)
    ep = job / 'units/episode_a'
    ctx = {'episode_id': 'episode_a', 'state_kind': 'none',
           'signals': [{'name': '<img src=x onerror=window.bad=1>', 'key': 's0', 'file': 'signals.npz'}]}
    (ep / 'context.json').write_text(json.dumps(ctx))
    np.savez(ep / 'signals.npz', s0=np.array([0.0, 1.0]))
    old = editor().public_dictionary(job, 'episode_a')
    effective = {**ctx, 'data_dictionary': old}
    qa = job / 'qa/episode_a.json'
    label = json.loads(qa.read_text())
    label['data_dictionary'] = old
    label['sensor_evidence'] = {'version': 1, 'interpretation_digest': signature(effective),
                                'source_proof': source_proof(effective, ep),
                                'findings': [{'claim': 'touch finding', 'evidence': []}], 'limitations': []}
    label['contacts'] = [{'id': 'touch', 'start_s': 0, 'end_s': 1}]
    label['dataset_checks'] = {'contact_checks': {'checked': 1, 'contacts': 1,
                                                  'notes': [{'check': 'touch_not_seen', 'evidence': 'old touch'}]}}
    qa.write_text(json.dumps(label))
    original = qa.read_bytes()
    with local_server(monkeypatch, job, True) as origin:
        url = origin + '/api/episode?file=episode_a.json'
        assert send(url)[1]['sensor_evidence']['findings'][0]['claim'] == 'touch finding'
        assert send(origin + '/api/dictionary', {'revision': 0, 'field_id': 'field_a', 'role': ''}, origin)[0] == 200
        current = send(url)[1]
        downloaded = send(url + '&download=1')[1]
        exported = send(origin + '/api/export', {'files': ['episode_a.json']})[1]
        (ep / 'context.json').unlink()
        missing_source = send(url)[1]
    for row in (current, downloaded, exported):
        assert row['data_dictionary']['entries']['field_a']['role'] == ''
        assert row['sensor_evidence']['findings'] == []
        assert row['sensor_evidence']['withheld_findings'][0]['claim'] == 'touch finding'
        assert row['withheld_contacts'][0]['id'] == 'touch'
        assert 'contact_checks' not in row['dataset_checks']
        assert row['withheld_contact_checks']['checked'] == 1
    import shutil
    import subprocess
    node = shutil.which('node')
    if node:
        page = serve.render_index('Tiny', {'mode': 'api'})
        start = page.index('function checksSection(d)')
        end = page.index('\nfunction matchesIssueFilter(', start)
        script = ('const OUR_CHECKS=[], CHECKS_OPEN=false, esc=x=>String(x), '
                  'sentences=x=>String(x), placementText=x=>String(x);' + page[start:end] +
                  '\nconsole.log(JSON.stringify([checksSection(JSON.parse(process.argv[1])), '
                  'checksSection(JSON.parse(process.argv[2]))]));')
        original_html, current_html = json.loads(subprocess.check_output(
            [node, '-e', script, json.dumps(label), json.dumps(current)], text=True))
        assert 'Contact checks' in original_html
        assert 'Contact checks' not in current_html
    assert qa.read_bytes() == original
    assert missing_source['sensor_evidence']['findings'] == []


def test_build_projects_after_contacts_are_added(tmp_path, monkeypatch):
    job, _ = make_job(tmp_path)
    old = editor().public_dictionary(job, 'episode_a')
    editor().save_override(job, {'revision': 0, 'field_id': 'field_a', 'role': ''}, 'owner')
    ep = job / 'units/episode_a'

    def add_saved_contacts(label, _ctx, _result):
        label['contacts'] = [{'id': 'touch', 'start_s': 0, 'end_s': 1}]
        label.setdefault('dataset_checks', {})['contact_checks'] = {'checked': 1, 'contacts': 1}

    monkeypatch.setattr(build, 'add_contacts', add_saved_contacts)
    label = {'completion': {'task_completed': 'failure'}}
    build.add_context(label, {'data_dictionary': old}, ep)
    assert 'contacts' not in label
    assert label['withheld_contacts'][0]['id'] == 'touch'
    assert 'contact_checks' not in label['dataset_checks']
    assert label['withheld_contact_checks']['checked'] == 1


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
            'after_labelling': True, 'after': {'role': ''}}], 'missing_fields': ['b'],
        'request_field_ids': ['a'], 'deferred_fields': ['b']}
    before = copy.deepcopy(overlay)
    d = {}
    build.add_context(d, {'data_dictionary': overlay}, ep)
    assert d['data_dictionary']['machine_entries'] == overlay['machine_entries']
    assert d['data_dictionary']['history'][0]['after_labelling'] is True
    assert d['data_dictionary']['missing_fields'] == ['b']
    assert d['data_dictionary']['request_field_ids'] == ['a']
    assert d['data_dictionary']['deferred_fields'] == ['b']
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


def test_dictionary_save_invalidates_cached_episode_across_navigation(tmp_path):
    import shutil
    import subprocess
    node = shutil.which('node')
    if not node:
        pytest.skip('node is not installed')
    page = serve.render_index('Tiny', {'mode': 'api'})
    cache_start = page.index('const _epCache = new Map()')
    cache_end = page.index('// other models', cache_start)
    form_start = page.index('function dictionaryRows(data, endpoint)')
    form_end = page.index('\nlet _dictionaryLoadToken = 0;', form_start)
    load_end = page.index('\nfunction readerNotesHtml(', form_end)
    script = r'''
    class Element {
      constructor(tag) { this.tag=tag; this.children=[]; this.listeners={}; this.open=false; }
      append(...children) { this.children.push(...children); }
      replaceChildren() { this.children=[]; }
      contains(element) { return this===element || this.children.some(child=>child.contains(element)); }
      setAttribute() {}
      addEventListener(name, callback) { this.listeners[name]=callback; }
      querySelector(tag) {
        for (const child of this.children) {
          if (child.tag===tag) return child;
          const found=child.querySelector(tag); if (found) return found;
        }
        return null;
      }
      names() { return (this.className==='dictionary-name' ? [this.textContent] : [])
        .concat(...this.children.map(child=>child.names())); }
      texts() { return [this.textContent].concat(...this.children.map(child=>child.texts())); }
    }
    const target=new Element('div'), location={protocol:'http:'};
    const window={__DICTIONARY_URL:'/api/dictionary'}, BOARD={};
    const document={getElementById:()=>target,createElement:tag=>new Element(tag),
      createTextNode:text=>Object.assign(new Element('text'),{textContent:text})};
    const STATIC=false, episodeUrl=file=>'api/episode?file='+file, loadSensors=()=>{};
    let _activeFile='a.json', navToken=0, revision=0, pauseReload=false;
    const data=name=>({revision,editable:true,fields:[{id:'shared',name,kind:'text'}],
      entries:{shared:{meaning:'shared field',role:revision>=3 ? 'annotation' : 'touch'}}});
    const deferred=()=>{let resolve;const promise=new Promise(r=>{resolve=r});return {promise,resolve};};
    const postGate=deferred(), reloadStarted=deferred(), reloadGate=deferred();
    const thirdStarted=deferred(), thirdGate=deferred();
    const dictionaryGates=[deferred(),deferred()]; let dictionaryCalls=0;
    const network=[];
    async function fetch(url, options={}) {
      network.push(url);
      if (options.method==='POST') {
        if (revision===0) await postGate.promise;
        if (revision===2) { thirdStarted.resolve(); await thirdGate.promise; }
        revision++;
        return {ok:true,json:async()=>({})};
      }
      if (url.startsWith('/api/dictionary?')) {
        const call=++dictionaryCalls;
        await dictionaryGates[call-1].promise;
        return {ok:true,json:async()=>data(call===1 ? 'stale response' : 'fresh response')};
      }
      if (url==='api/episode?file=a.json' && revision>0 && pauseReload) {
        reloadStarted.resolve(); await reloadGate.promise;
      }
      const name=url.endsWith('a.json') ? (revision ? 'fresh A' : 'stale A') : (revision>=3 ? 'fresh B' : 'B');
      return {ok:true,json:async()=>data(name)};
    }
    ''' + page[cache_start:cache_end] + page[form_start:load_end] + r'''
    async function selectEp(file) {
      const token=++navToken;
      _activeFile=file;
      const result=await fetchEpisode(file);
      if (token===navToken && _activeFile===file)
        dictionaryRows(result,'/api/dictionary?file='+file);
    }
    (async()=>{
      _epCache.set('a.json',Promise.resolve(data('stale A')));
      dictionaryRows(data('stale A'),'/api/dictionary?file=a.json');
      const saving=target.querySelector('form').listeners.submit({preventDefault(){}});
      await selectEp('b.json');
      await selectEp('a.json');
      postGate.resolve();
      await saving;
      const returned={names:target.names(),aFetches:network.filter(x=>x==='api/episode?file=a.json').length};
      pauseReload=true;
      const savingAgain=target.querySelector('form').listeners.submit({preventDefault(){}});
      await reloadStarted.promise;
      await selectEp('b.json');
      reloadGate.resolve();
      await savingAgain;
      const afterNavigation=target.names(), active=_activeFile;
      _activeFile='a.json';
      const oldLoad=loadDictionary(data('old response'),'a.json');
      const newLoad=loadDictionary(data('fresh response'),'a.json');
      dictionaryGates[1].resolve(); await newLoad;
      dictionaryGates[0].resolve(); await oldLoad;
      const dictionary=target.names();
      await selectEp('a.json');
      const thirdSave=target.querySelector('form').listeners.submit({preventDefault(){}});
      await thirdStarted.promise;
      await selectEp('b.json');
      const beforeThirdSave=target.names();
      const beforeThirdRole=target.texts().includes('touch');
      thirdGate.resolve(); await thirdSave;
      console.log(JSON.stringify({returned,afterNavigation,active,dictionary,
        beforeThirdSave,beforeThirdRole,afterThirdSave:target.names(),
        afterThirdRole:target.texts().includes('annotation'),activeAfterThirdSave:_activeFile}));
    })().catch(error=>{console.error(error);process.exitCode=1;});
    '''
    path = tmp_path / 'dictionary_cache.js'
    path.write_text(script)
    result = subprocess.run([node, str(path)], capture_output=True, text=True, check=True, timeout=10)
    observed = json.loads(result.stdout)
    assert observed['returned'] == {'names': ['fresh A'], 'aFetches': 1}
    assert observed['afterNavigation'] == ['B']
    assert observed['active'] == 'b.json'
    assert observed['dictionary'] == ['fresh response']
    assert observed['beforeThirdSave'] == ['B']
    assert observed['beforeThirdRole'] is True
    assert observed['afterThirdSave'] == ['fresh B']
    assert observed['afterThirdRole'] is True
    assert observed['activeAfterThirdSave'] == 'b.json'
