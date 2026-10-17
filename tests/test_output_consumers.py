"""Partial model replies retain evidence without supplying invented field semantics."""
import copy
import json
import shutil
import subprocess
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from board import build, serve, to_board
from compare import metrics
from label.harness import parse_response
from test_board import _episodes, _output, _run


def malformed_reply(value):
    labels = _output('episode_000000')['labels']
    labels.update(completion={'task_completed': value, 'reason': value},
        tasks=[{'task': 'place cup', 'objects': value, 'outcome': value}],
        key_events=[{'t_s': 2, 'label': value, 'kind': value, 'outcome': value}],
        scene={'objects': [{'name': 'cup', 'attributes': value}]},
        scene_graph=[{'t_s': 2, 'relations': value}],
        recovery=[{'failure_t_s': 1, 'failure': 'slip', 'recovered': value}],
        goal_alignment={'relation': value, 'matches_given': value},
        data_issues=[{'issue': 'visible issue', 'category': value, 'severity': value}])
    for row in labels['timeline']:
        row.update(arm=value, object=value, contribution=value, progress=value, hands_visible=value)
    return labels


@pytest.mark.parametrize('value', [True, 7, ['left'], {'outcome': 'success'}, 'not_an_enum'])
def test_partial_replies_project_invalid_nested_values_without_semantics(value):
    raw = json.dumps(malformed_reply(value))
    labels, ok = parse_response(raw)
    assert ok
    result = {'parse_ok': ok, 'labels': labels, 'raw': raw, 'episode_dir': '/x/episode_000000'}
    before = copy.deepcopy(result)
    d = to_board.convert(result, 'demo')
    assert d['completion']['task_completed'] is None
    assert d['event_labels'][0]['arm'] == (value if isinstance(value, str) else '')
    assert d['event_labels'][0]['object'] == (value if isinstance(value, str) else '')
    assert d['event_labels'][0]['contribution'] == ''
    assert d['event_labels'][0]['progress'] is None
    assert d['tasks'][0]['objects'] == (['left'] if value == ['left'] else [])
    assert d['tasks'][0]['outcome'] == ''
    assert d['key_events'][0]['outcome'] == ''
    assert d['key_events'][0]['label'] == (value if isinstance(value, str) else '')
    assert d['scene_graph'][0]['relations'] == (['left'] if value == ['left'] else [])
    assert d['recovery'][0].get('recovered') in (None, True)
    assert d['_off_schema'] and d.get('_schema_violations')
    measured = metrics.measure(labels, 'demo')
    assert measured['outcome'] is None
    assert measured['families'] == (['d:Not an enum'] if isinstance(value, str) else ['d:Untagged'])
    assert measured['data_issues'] == 0 and measured['tasks'] == [1, 0]
    assert result == before
    without_tasks = {key: item for key, item in labels.items() if key != 'tasks'}
    assert metrics.measure(without_tasks, 'demo')['outcome'] is None


@pytest.mark.skipif(not shutil.which('node'), reason='no node')
@pytest.mark.parametrize('profile,value', [('teleop_arms', True), ('handheld_gripper', ['left']),
    ('ego_head', {'outcome': 'success'}), ('teleop_arms', 7)])
def test_wrong_type_reply_survives_real_comparison_publication_and_render(tmp_path, monkeypatch, profile, value):
    run = _run(tmp_path / 'runs/demo', '20260102-0000_full_abc1234', 'success', 'aligned')
    eps = _episodes(tmp_path / 'episodes/demo', profile=profile)
    other = tmp_path / 'runs/other'
    (other / 'out').mkdir(parents=True)
    (other / 'run.json').write_text(json.dumps({'run_id': 'other', 'code': 'abc', 'kind': 'full',
        'status': 'done', 'slice': str(eps), 'command': ['--model', 'some/model']}))
    reply = malformed_reply(value)
    if profile != 'ego_head':
        reply.pop('tasks')
    if isinstance(value, list):
        reply['timeline'] = [[0, 2, value, 'reach', None, None, None, 'advancing', .2, None]]
    raw = json.dumps(reply)
    labels, ok = parse_response(raw)
    assert ok
    result = {**_output('episode_000000'), 'labels': labels, 'raw': raw}
    (run / 'out/episode_000000.json').write_text(json.dumps(result))
    (other / 'out/episode_000000.json').write_text(json.dumps(result))
    board = tmp_path / 'board'
    board.mkdir()
    (board / 'manifest.json').write_text(json.dumps({'board': 'demo', 'datasets': [
        {'dataset': 'demo', 'run': str(run), 'episodes': str(eps)}], 'comparisons': [
        {'key': 'other', 'name': 'Other', 'run': str(other)}]}))
    build.build(board)
    built = json.loads((board / 'qa/episode_000000.json').read_text())
    assert '_label_failed' not in built and built['_off_schema']
    assert (board / 'compare/metrics.json').exists()
    monkeypatch.setattr(serve, 'HERE', board / 'qa')
    monkeypatch.setattr(serve, 'COMPARE_DIR', board / 'compare')
    monkeypatch.setattr(serve, 'MP4_DIR', tmp_path / 'clips')
    monkeypatch.setattr(serve, 'HANDS_DIR', tmp_path / 'hands')
    monkeypatch.setattr(serve, 'SENSORS_DIR', tmp_path / 'sensors')
    server = ThreadingHTTPServer(('127.0.0.1', 0), serve.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    page = tmp_path / 'page.html'
    page.write_text(serve.render_index('Partial replies', {'mode': 'api', 'compare': True}))
    try:
        p = subprocess.run(['node', str(Path(__file__).parent / 'render_smoke.js'), str(page),
            f'http://127.0.0.1:{server.server_port}/'], text=True, capture_output=True)
        assert p.returncode == 0, p.stderr
        report = json.loads(p.stdout)
        assert report['episodes'] == 2 and report['compare_view'] and report['rendered_comparisons']
    finally:
        server.shutdown()
        thread.join()


def test_model_supplied_diagnostic_shapes_remain_diagnostic_not_executable_metadata():
    for field in ('_schema_violations', '_dropped'):
        for value in (True, 'not a diagnostic list', [7], {'why': 'not a list'}):
            labels, ok = parse_response(json.dumps({'task_summary': 'The hand reaches.', field: value}))
            assert ok
            original = copy.deepcopy(labels)
            d = to_board.convert({'labels': labels})
            measured = metrics.measure(labels, 'demo')
            assert d['_off_schema'][field]
            assert any(field in message for message in measured['violations'])
            assert labels == original
    labels, ok = parse_response(json.dumps({'task_summary': 'The hand reaches.',
        '_schema_violations': ['retained source violation'],
        '_dropped': [{'field': 'timeline', 'row': 0, 'why': 'retained source omission'}]}))
    assert ok
    d = to_board.convert({'labels': labels})
    assert 'retained source violation' in d['_schema_violations']
    assert d['_off_schema']['timeline'] == 1
    measured = metrics.measure(labels, 'demo')
    assert 'retained source violation' in measured['violations']
    assert any('retained source omission' in message for message in measured['violations'])


def test_nested_assessments_use_the_same_structural_contract_as_episode_labels():
    for field, value in (('scene', True), ('timeline', 'not rows'), ('completion', ['success'])):
        raw = {'task_summary': 'The hand reaches.', 'part_assessments': [
            {'task_summary': 'Local reach.', field: value}]}
        labels, ok = parse_response(json.dumps(raw))
        assert ok
        original = copy.deepcopy(labels)
        d = to_board.convert({'labels': labels})
        measured = metrics.measure(labels, 'demo')
        assert d['part_assessments'][0]['task_summary'] == 'Local reach.'
        assert field not in d['part_assessments'][0]
        assert any('part_assessments' in key and field in key for key in d['_off_schema'])
        assert any('part_assessments' in message and field in message for message in measured['violations'])
        assert labels == original


@pytest.mark.skipif(not shutil.which('node'), reason='no node')
def test_invalid_supplied_booleans_cannot_trigger_missing_field_outcome_inference():
    cases = []
    for fields, recovered, match in (({}, True, True),
            ({'recovered': True, 'matches_given': True}, True, True),
            ({'recovered': False, 'matches_given': False}, False, False),
            ({'recovered': None, 'matches_given': None}, None, None),
            ({'recovered': 'false', 'matches_given': 'false'}, None, None)):
        labels, ok = parse_response(json.dumps({'task_summary': 'The hand reaches.',
            'recovery': [{'failure_t_s': 1, 'failure': 'slip', 'recovered_at_s': 2,
                          **{k: v for k, v in fields.items() if k == 'recovered'}}],
            'goal_alignment': {'relation': 'aligned',
                              **{k: v for k, v in fields.items() if k == 'matches_given'}}}))
        assert ok
        cases.append({'d': to_board.convert({'labels': labels}), 'recovered': recovered, 'match': match})
    script = r'''
const fs=require('fs'),assert=require('assert');
const s=fs.readFileSync('board/index.html','utf8'),cases=JSON.parse(process.argv[1]);
const start=s.indexOf('  const recoveries ='),end=s.indexOf('  const ivHtml =',start);
const recovery=new Function('d','esc','fmtT',s.slice(start,end)+'return {recoveries,rcHtml};');
const gaStart=s.indexOf('  const ga = d.goal_alignment'),gaEnd=s.indexOf('  const bannerBody =',gaStart);
const alignment=new Function('d','givenMode','esc',s.slice(gaStart,gaEnd)+'return alignHtml;');
for(const c of cases){
 const r=recovery(c.d,String,String),a=alignment(c.d,true,String);
 assert.strictEqual(r.recoveries[0].recovered,c.recovered);
 assert.strictEqual(r.rcHtml.includes('class="rec-ok"'),c.recovered===true);
 assert.strictEqual(a.includes('align-chip match'),c.match===true);
 assert.strictEqual(a.includes('align-chip nomatch'),c.match===false);
 if(c.match===null)assert(a.includes('unclear'));
}
'''
    subprocess.run(['node', '-e', script, json.dumps(cases)], check=True)


def test_legitimate_unclear_ego_task_retains_neighbor_and_is_not_a_schema_violation():
    labels = _output('episode_ego', tasks=[{'task': 'fold cloth', 'outcome': 'unclear'},
        {'task': 'place cloth', 'outcome': 'success'}])['labels']
    parsed, ok = parse_response(json.dumps(labels))
    assert ok
    projected = to_board.convert({'labels': parsed}, 'demo')
    assert [row['outcome'] for row in projected['tasks']] == ['unclear', 'success']
    measured = metrics.measure(parsed, 'demo')
    assert measured['tasks'] == [2, 1]
    assert not any('task 0: outcome' in item for item in measured['violations'])
    parsed['tasks'][0]['outcome'] = 'mostly'
    assert any('task 0: outcome' in item for item in metrics.measure(parsed, 'demo')['violations'])
