"""One reply must retain the same meaning across annotation consumers."""
import copy
import json
from types import SimpleNamespace

import pytest

from board import build, static, to_board
from board.label_projection import project
from board.families import Families
from checks import label_consistency
from compare import metrics
from gate.score import judge
from label.harness import parse_response, TIMELINE_COLUMNS
from test_output_consumers import complete_schema_reply
from test_public_export_boundaries import public_sources, Request  # noqa: F401 (public_sources is a fixture)


def test_public_credentials_are_withheld_by_live_and_static_exports(public_sources, tmp_path, monkeypatch):
    source = public_sources['qa'] / 'ep.json'
    label = json.loads(source.read_text())
    names = ['hf_token', 'auth_token', 'aws_session_token', 'github_token', 'bearer_token',
             'openrouter_key', 'api_token', 'openai_key', 'anthropic_key',
             'access_key', 'db_password', 'api_secret', 'stripe_secret_key', 'slack_bot_token',
             'replicate_api_token', 'auth', 'passphrase', 'bearer', 'slackbottoken']
    label['publisher_metadata'] = {name: 'fixture-access-' + name for name in names}
    label['publisher_metadata'].update(token_count=17, key='/observations/force',
        session_id='episode-session', session=4, signature=[0.1, 0.2], sig=3,
        eos_token='</s>', task_key='/tasks/0', firmware_version='10.3.0.17',
        nested_access=['ghp_' + 'a' * 36, 'Bearer fixture-access-value'],
        citation='https://example.org/paper?version=2',
        access='https://example.org/video?key=fixture-query-access',
        host='https://camera.example-tailnet.ts.net/video',
        access_examples=['camera.example-tailnet.ts.net', '100.101.7.9', 'nas.x.local', 'api.x.internal',
                         'https://oauth.example.org/callback?code=fixture-oauth-access',
                         'https://hooks.slack.com/services/fixture-webhook-access',
                         'Authorization: Bearer fixture-bearer-access', '100.101.7.9:8896',
                         '127.0.0.1:9000', '192.168.1.2:80', 'example.org:443'],
        fields={'signature': [0.1, 0.2], 'token_ids': [3, 7], 'field_key': '/observations/force'})
    source.write_text(json.dumps(label))
    request = Request('/api/episode?file=ep.json&download=1')
    from email.message import Message
    request.headers = Message()
    request.headers['Content-Length'] = '0'
    request.do_GET()
    assert request.status == 200
    monkeypatch.setattr(static, 'plan', lambda *a, **k: [
        {'rec': {'file': 'ep.json', 'episode_id': 'ep', 'duration_s': 1},
         'ds': 'synthetic', 'media': {}, 'frames': {}}])
    args = SimpleNamespace(board=tmp_path, **public_sources, out=tmp_path / 'site',
        build_id='credential-boundary', force=False, public_base=None, title='Synthetic', header=None)
    assert static.cmd_site(args) == 0
    outputs = [request.wfile.getvalue().decode(),
               (args.out / 'credential-boundary/data/ep/ep.json').read_text()]
    for output in outputs:
        assert 'fixture-access-' not in output and 'fixture-query-access' not in output
        assert '.ts.net' not in output and '100.101.7.9' not in output
        assert 'fixture-oauth-access' not in output and 'fixture-bearer-access' not in output
        assert 'fixture-webhook-access' not in output and 'nas.x.local' not in output and 'api.x.internal' not in output
        metadata = json.loads(output)['publisher_metadata']
        assert metadata == {'token_count': 17, 'key': '/observations/force',
                            'session_id': 'episode-session', 'session': 4, 'signature': [0.1, 0.2], 'sig': 3,
                            'eos_token': '</s>', 'task_key': '/tasks/0', 'firmware_version': '10.3.0.17',
                            'nested_access': [None, None],
                            'citation': 'https://example.org/paper?version=2', 'access_examples': [None] * 10 + ['example.org:443'],
                            'fields': {'signature': [0.1, 0.2], 'token_ids': [3, 7],
                                       'field_key': '/observations/force'}}
    assert json.loads(source.read_text()) == label


@pytest.mark.parametrize('key,value', [('_dropped', 5), ('_schema_violations', {}),
    ('_schema_row_indices', {'timeline': [4, 9]})])
def test_model_diagnostic_keys_cannot_break_parsing_or_relabel_source_rows(key, value):
    raw = complete_schema_reply()
    raw['timeline'] = [dict(raw['timeline'][0]), dict(raw['timeline'][0], contribution='unsupported')]
    raw['timeline'][0]['progress'] = True
    raw['key_events'] = [7, {'t_s': 1, 'label': 'reach', 'kind': 'milestone', 'outcome': 'success'}]
    raw[key] = value
    parsed, ok = parse_response(json.dumps(raw))
    assert ok and len(parsed['timeline']) == 2
    shown = to_board.convert({'labels': parsed})
    measured = metrics.measure(parsed, None)
    assert any('timeline[1].contribution' in text for text in shown['_schema_violations'])
    assert not any('timeline[9]' in text for text in shown['_schema_violations'] + measured['violations'])
    # Directly supplied legacy structured records must also remain safe.
    direct = metrics.measure(raw, None)
    assert direct['segments'] == 2
    assert not any('timeline[9]' in text for text in direct['violations'])


def test_case_and_whitespace_meanings_agree_across_board_compare_and_gate():
    labels = complete_schema_reply()
    labels.update(completion={'task_completed': ' Partial '},
        goal_alignment={'relation': ' Aligned '},
        data_issues=[{'category': 'camera_fault', 'issue': 'Frozen camera', 'severity': ' High '}])
    labels['timeline'][0]['contribution'] = ' Advancing '
    parsed, ok = parse_response(json.dumps(labels))
    assert ok
    shown = to_board.convert({'labels': parsed})
    measured = metrics.measure(parsed, None)
    assert shown['completion']['task_completed'] == measured['outcome'] == 'failure'
    assert shown['data_issues'][0]['severity'] == 'high' and measured['data_issues'] == 1
    assert shown.get('_schema_violations', []) == measured['violations'] == []
    assert judge({'outcome': ['failure'], 'align': ['aligned'], 'issue': ['camera_fault'], 'sev': 'medium'}, parsed)[0]
    assert not judge({'max_issue_sev': 'low'}, parsed)[0]
    tasks = {'tasks': [{'task': 'fold', 'outcome': ' Success '}]}
    assert judge({'outcome': ['tasks:success']}, tasks)[0]


@pytest.mark.parametrize('severity', [['high'], {'level': 'high'}, None])
def test_gate_does_not_turn_invalid_or_unknown_severity_into_a_low_severity_claim(severity):
    labels = {'data_issues': [{'category': 'camera_fault', 'issue': 'Frozen camera', 'severity': severity}]}
    assert not judge({'issue': ['camera_fault'], 'sev': 'medium'}, labels)[0]
    # an unrated issue has no tier: the bound holds over the rated ones, as the board counts them, and the unrated
    # issue is reported for review rather than read as low or as high
    met, why = judge({'max_issue_sev': 'low'}, labels)
    assert met and why == ["needs review: 1 unrated issues have no severity to bound: ['camera_fault']"]
    assert not judge({'issue': ['camera_fault']}, {'data_issues': [
        {'category': 'camera_fault', 'severity': 'high', 'issue': ['not text']} ]})[0]


@pytest.mark.parametrize('progress', [True, 7, 61, float('nan'), float('inf')])
def test_consistency_does_not_infer_progress_from_rejected_values(progress):
    labels = {'completion': {'task_completed': 'failure'},
              'timeline': [{'start_s': 0, 'end_s': 1, 'action': 'reach', 'progress': progress}]}
    shown = to_board.convert({'labels': labels})
    assert not any(item['rule'] == 'progress_vs_outcome' for item in label_consistency.check(labels, 2))
    assert label_consistency.check(labels, 2) == label_consistency.check(shown, 2)


def test_all_displayed_issue_and_outcome_times_are_checked_against_episode_bounds():
    labels = {'duration_s': 4, 'event_labels': [], 'key_events': [],
              'data_issues': [{'issue': 'Frozen camera', 't_s': 999}],
              'operator_mistakes': [{'issue': 'Drop', 't_s': 2, 'end_s': 1}],
              'tasks': [{'task': 'reach', 'completed_at_s': 999}],
              'recovery': [{'failure_t_s': -5}], 'completion': {'goal_reached_at_s': 999}}
    before = copy.deepcopy(labels)
    notes = build.steps_outside(labels)
    assert notes and 'data issue' in notes[0]['what'] and 'operator mistake' in notes[0]['what']
    assert all(name in notes[0]['what'] for name in ('task', 'recovery', 'completion'))
    assert label_consistency.check({'completion': {'task_completed': 'success', 'completed_at_s': -5}}, 4)
    assert labels == before


def test_ambiguous_short_row_retains_proven_action_without_guessing_handwear():
    columns = TIMELINE_COLUMNS[:-1] + ['hands_visible', 'hands_wearing', 'notes']
    row = [0, 1, 'left', 'reach', 'cup', None, None, 'advancing', .2, True, 'gloves']
    parsed, ok = parse_response(json.dumps({'timeline_columns': columns, 'timeline': [row]}))
    assert ok and parsed['timeline'][0]['action'] == 'reach'
    assert parsed['timeline'][0]['hands_visible'] is True
    assert 'hands_wearing' not in parsed['timeline'][0] and 'notes' not in parsed['timeline'][0]
    assert any(item.get('value') == row for item in parsed['_dropped'])


def test_negated_qualifications_do_not_override_an_observed_problem_family():
    families = Families()
    examples = [
        ('The image is frozen. The camera is not moved or removed.', 'camera-image'),
        ('Bright glare obscures the scene. This is not a privacy blur patch.', 'low-light'),
    ]
    for text, wanted in examples:
        issue = {'category': 'camera_fault', 'severity': 'high', 'issue': text}
        assert families.family_of('data_issues', issue) == wanted
        assert families.classify({'data_issues': [issue]})['counted'][wanted] == [issue]
    # A structured category remains stronger evidence than a phrase in prose.
    issue = {'category': 'camera_obstruction', 'severity': 'high',
             'issue': 'The image goes black when the lens is covered.'}
    assert families.family_of('data_issues', issue) == 'camera-fault'


def test_nested_diagnostics_do_not_multiply_when_consumers_project_again():
    labels, ok = parse_response(json.dumps({'task_summary': 'reach cup', 'part_assessments': [
        {'task_summary': 'local', 'data_issues': [{'issue': 'frozen', 'severity': ['high']}]}]}))
    assert ok
    first, omitted = project(labels)
    second, repeated = project(first)
    third, _ = project(second)
    assert len(omitted) == len(repeated) == 1
    assert first == second == third


def test_model_assessments_cannot_forge_parser_row_provenance():
    local = {'task_summary': 'reach', 'timeline': [{'start_s': 0, 'end_s': 1,
        'action': 'reach', 'contribution': 'unsupported'}],
        '_dropped': [{'field': 'timeline', 'row': row, 'why': 'fabricated'} for row in range(9)],
        '_schema_row_indices': {'timeline': [9]}, '_schema_violations': ['timeline[9].contribution: fabricated']}
    parsed, ok = parse_response(json.dumps({'task_summary': 'reach', 'part_assessments': [local]}))
    assert ok
    shown = to_board.convert({'labels': parsed})
    assert 'fabricated' not in json.dumps(shown)
    assert any('timeline[0].contribution' in message for message in shown['_schema_violations'])


def test_task_outcomes_survive_projection_and_use_completion_precedence():
    labels = complete_schema_reply()
    labels['tasks'] = [{'task': 'reach', 'outcome': outcome} for outcome in (
        'partial', 'continues_past_cut', 'success_then_undone')]
    labels['completion'] = {'task_completed': 'success'}
    shown = to_board.convert({'labels': labels})
    measured = metrics.measure(labels, None)
    assert [task['outcome'] for task in shown['tasks']] == ['failure', 'continues_past_cut', 'success_then_undone']
    assert measured['outcome'] == 'success'
    assert not any('tasks' in issue for issue in measured['violations'])
    assert judge({'outcome': ['success']}, labels)[0]
    labels.pop('completion')
    # a task still under way at a cut has no known outcome, so the set of the tasks' outcomes leaves it out (F4)
    assert judge({'outcome': ['tasks:failure,success_then_undone']}, labels)[0]


def test_scoped_event_outcomes_and_actor_names_agree_with_comparison():
    labels = complete_schema_reply()
    labels['timeline'][0]['arm'] = ' LEFT '
    labels['key_events'] = [{'t_s': 0, 'label': 'reach', 'kind': 'phase', 'outcome': 'Partial'}]
    shown = to_board.convert({'labels': labels})
    measured = metrics.measure(labels, None)
    assert shown['event_labels'][0]['arm'] == 'left'
    assert shown['key_events'][0]['outcome'] in ('', None)
    assert any('key_events[0].outcome' in issue for issue in shown['_schema_violations'])
    assert measured['violations'] == shown['_schema_violations']
    labels['timeline'][0]['arm'] = ' '
    shown = to_board.convert({'labels': labels})
    assert any('timeline[0].arm' in issue for issue in shown['_schema_violations'])
    labels['timeline'][0]['arm'] = 'front camera'
    config = {'views': ['exo'], 'cam_labels': ['front camera']}
    assert not any('timeline[0].arm' in issue for issue in
                   metrics.measure(labels, None, config)['violations'])
    assert to_board.convert({'labels': labels, 'config': config})['event_labels'][0]['arm'] == 'front camera'


def test_positive_problem_predicates_are_not_cancelled_by_later_negation():
    families = Families()
    cases = [
        ('The camera is frozen and not recovered', '', 'camera-image'),
        ('Frozen view', 'no motion for 10 s', 'camera-image'),
        ('Camera frozen, not moved', '', 'camera-image'),
        ("Removed from the mount and doesn't capture the task", '', 'camera-fault'),
        ("Glare washes out the view and the cup can't be seen", '', 'low-light'),
        ('The image is frozen. The camera is not moved or removed.', '', 'camera-image'),
        ('This is not a privacy blur patch. Bright glare obscures the scene.', '', 'low-light'),
        ('The grasp is not visible because the exo view is frozen.', '', 'camera-image'),
        ('The image is frozen; no glare, clipping, or overexposure is visible.', '', 'camera-image'),
        ('No glare, clipping or overexposure. The frame is frozen.', '', 'camera-image'),
        ('Glare is not present in any frame. The image is frozen.', '', 'camera-image'),
        ('Initially looks like occlusion but it is not. The image is frozen.', '', 'camera-image'),
        ('No privacy blur patch is present. The view is frozen.', '', 'camera-image'),
        ('The image is frozen but it is not dark.', '', 'camera-image'),
        ('Glare obscures the cup but it is not an occlusion.', '', 'low-light'),
        ('The view is occluded but it is not a privacy blur patch.', '', 'camera-fault'),
    ]
    for text, evidence, expected in cases:
        issue = {'category': 'camera_fault', 'severity': 'high', 'issue': text, 'evidence': evidence}
        assert families.family_of('data_issues', issue) == expected
        assert families.classify({'data_issues': [issue]})['counted'][expected] == [issue]


@pytest.mark.parametrize('name', ['Gripper', 'Left'])
def test_single_camera_actor_case_preserves_the_publishers_label(name):
    config = {'views': ['exo'], 'cam_labels': [name]}
    for actor in (name, name.lower(), name.upper(), ' ' + name.swapcase() + ' '):
        labels = complete_schema_reply()
        labels['timeline'][0]['arm'] = actor
        projected, dropped = project(labels, config)
        assert projected['timeline'][0]['arm'] == name
        assert not any(row['field'] == 'timeline.arm' for row in dropped)
        assert to_board.convert({'labels': labels, 'config': config})['event_labels'][0]['arm'] == name
        assert not any('timeline[0].arm' in text for text in metrics.measure(labels, None, config)['violations'])


def test_episode_family_status_is_exclusive_without_losing_unrated_issue_counts():
    issue = {'issue': 'The image is frozen.', 'category': 'camera_fault'}
    labels = {'data_issues': [dict(issue, severity='high'), dict(issue, severity=None)]}
    families = Families().classify(labels)
    assert 'camera-image' in families['counted'] and 'camera-image' not in families['unrated']
    assert metrics.measure(labels, None)['data_issues_unrated'] == 1


def test_comparison_bars_include_unrated_issues_without_inventing_rated_ones():
    import shutil
    import subprocess
    from pathlib import Path
    node = shutil.which('node')
    if not node:
        pytest.skip('no node')
    measured = metrics.measure({'data_issues': [{'issue': 'Unknown severity', 'severity': None}],
                                'operator_mistakes': [{'issue': 'Unknown severity', 'severity': None}]}, None)
    program = r"""
const {src,piece}=require('./tests/page_functions')('./board/index.html');
const fNum=String, fPct=String, fUsd=String, fSec=String, fMin=String;
const rows = {};
const cmpView = {querySelector(selector) {
  if (!selector.startsWith('[data-chart=')) return {querySelectorAll:()=>[],querySelector:()=>({})};
  const id = selector.match(/"([^"]+)"/)[1];
  const segs = [{style:{}},{style:{}},{style:{}}];
  const bar = {style:{},querySelectorAll:()=>segs};
  const small = {}, bv = {};
  const row = {dataset:{k:'model'},querySelector:s=>s==='.br-bar'?bar:s==='small'?small:bv};
  rows[id] = {segs,bar,small,bv,row};
  return {querySelectorAll:()=>[row],querySelector:()=>({})};
}};
const CMP_RIG='all', PAIR_CHARTS=[];
const CMP_METRICS={main:['model'],models:[{key:'model',name:'Model'}],
  summary:{all:{responses:{},metrics:{model:JSON.parse(require('fs').readFileSync(0,'utf8'))}}}};
const document={getElementById:()=>({})};
const cmpOrder=()=>[], shownName=x=>x, renderEpisodeTable=()=>{};
const tween=(target,value)=>{target.value=value};
eval(piece('esc') + src.slice(src.indexOf('const MAIN_CHARTS ='),src.indexOf('const ptsDelta ='))
  + piece('updateCompare') + ';updateCompare(true);');
console.log(JSON.stringify(rows));
"""
    output = subprocess.run([node, '-e', program], input=json.dumps(measured), text=True,
                            capture_output=True, check=True, cwd=Path(__file__).resolve().parent.parent)
    for key in ('data_issues', 'operator_mistakes'):
        bar = json.loads(output.stdout)[key]
        assert bar['bar']['style']['width'] == '100.00%'
        assert [segment['style']['flexGrow'] for segment in bar['segs']] == ['0', '0', '1']
        assert '1 unrated' in bar['small']['textContent'] and 'unrated' in bar['row']['dataset']['tip']


def test_unrated_issue_stays_visible_without_becoming_low_severity():
    import shutil
    import subprocess
    from pathlib import Path
    node = shutil.which('node')
    if not node:
        pytest.skip('no node')
    raw = {'data_issues': [{'issue': 'The camera image is frozen.', 'severity': None}]}
    shown = to_board.convert({'labels': raw})
    item = dict(shown['data_issues'][0], counted=False)
    assert not Families().counts('data_issues', item)
    measured = metrics.measure(raw, None)
    assert measured['data_issues_minor'] == 0 and measured['data_issues_unrated'] == 1
    # the gate bounds the highest rated severity, as the board counts it, and asks for review of the unrated one
    met, reasons = judge({'max_issue_sev': 'high'}, raw)
    assert met and reasons == ['needs review: 1 unrated issues have no severity to bound: [None]']
    assert judge({'max_issue_sev': 'low'}, raw)[0]
    program = r"""
const {src,piece}=require('./tests/page_functions')('./board/index.html');
const start=src.indexOf('  const byCount =');
const end=src.indexOf('  const sp =',start);
const render=new Function(piece('esc')+src.match(/^const countsIssue = [^\n]+/m)[0]
  +src.slice(start,end)+'return issueRows;')();
console.log(render(JSON.parse(require('fs').readFileSync(0,'utf8')),'data_issues'));
"""
    rendered = subprocess.run([node, '-e', program], input=json.dumps([item]), text=True,
                              capture_output=True, check=True, cwd=Path(__file__).resolve().parent.parent).stdout
    assert 'The camera image is frozen.' in rendered
    assert '>unknown</span>' in rendered and '>unrated</span>' in rendered
    assert 'low severity' not in rendered and '>minor</span>' not in rendered


@pytest.mark.parametrize('key', ['data_issues', 'operator_mistakes'])
def test_unknown_severity_is_not_minor_in_family_or_live_and_static_rail(tmp_path, key):
    from board import serve
    for severity in (None, 'unsupported', ['high']):
        labels = {key: [{'issue': 'The camera image is frozen.', 'severity': severity},
                        {'issue': 'A small visible defect.', 'severity': 'low'}]}
        shown = to_board.convert({'labels': labels})
        classified = Families().classify(shown)
        assert sum(map(len, classified['minor'].values())) == 1
        assert sum(map(len, classified['unrated'].values())) == 1
        shown['episode_prompt'] = 'Move the cup.'
        (tmp_path / 'episode_x.json').write_text(json.dumps(shown))
        row, = serve.rail_records(tmp_path)
        suffix = 'issues' if key == 'data_issues' else 'mistakes'
        assert row['n_minor_' + suffix] == 1
        assert row['n_unrated_' + suffix] == 1
        assert not row['families']
        assert row['unrated_families']


@pytest.mark.parametrize('text,wanted', [
    ('The exo view appears to be frozen.', 'camera-image'),
    ('The wrist feed seems to be corrupted.', 'camera-image'),
    ('Glare never clears during the placement.', 'low-light'),
    ('Frozen frame is not present in any view.', 'd:Camera fault'),
    ('The footage is free of glare.', 'd:Camera fault'),
    ('No task progress while the image is frozen.', 'camera-image'),
    ('Initially looks like occlusion but it is not. The image is frozen.', 'camera-image'),
])
def test_legacy_camera_family_uses_asserted_problem_not_hedge_or_remedy(text, wanted):
    issue = {'category': 'camera_fault', 'severity': 'high', 'issue': text}
    assert Families().family_of('data_issues', issue) == wanted


@pytest.mark.parametrize('problem', ['Anonymization', 'Overexposure', 'Occlusion', 'Corruption', 'Streaking'])
@pytest.mark.parametrize('predicate', ['is not present', 'was not observed', 'absent throughout'])
def test_partial_word_family_patterns_still_respect_denied_predicates(problem, predicate):
    issue = {'category': 'camera_fault', 'severity': 'high',
             'issue': 'The camera is slightly out of focus.', 'evidence': f'{problem} {predicate}'}
    assert Families().family_of('data_issues', issue) == 'd:Camera fault'


@pytest.mark.parametrize('text,wanted', [
    ('Glare? No.', 'd:Camera fault'),
    ('The view is not without glare.', 'low-light'),
    ('Washed-out highlights are not present.', 'd:Camera fault'),
    ('Rotated camera is not present.', 'd:Camera fault'),
])
def test_legacy_family_keeps_question_answers_and_presence_predicates_together(text, wanted):
    assert Families().family_of('data_issues', {'category': 'camera_fault', 'issue': text}) == wanted
