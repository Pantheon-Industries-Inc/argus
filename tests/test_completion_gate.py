"""A reply is finished work only when it holds what its request asked for (label/completeness.py).

The audit's reply {"task_summary": "A demonstration was recorded."} parsed, and a normally stopped saved record of it
counted as a finished annotation, so a resume would keep it and never label the episode again. Completeness is now
judged against the request: a timeline covering the footage it sent, and the outcome its instructions ask for in its
mode. These tests hold each site to that one rule.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from label import harness, pieces, prompts

# a 60 s request sampled every 1.5 s
INSTANTS = [round(1.5 * k, 3) for k in range(41)]
AUDIT_REPLY = '{"task_summary":"A demonstration was recorded."}'


def request(rig='teleop_arms', task_mode='episode', instruction='Put the cup in the bin.', instants=INSTANTS):
    return {'annotation_policy': {'version': 1, 'rig': rig, 'task_mode': task_mode},
            'given_prompt': instruction, 'config': {'timesteps_s': list(instants)}}


def saved(text, **kw):
    """A normally stopped saved record of this reply, with parse_ok as the parser alone gives it."""
    labels, ok = harness.parse_response(text)
    return {**request(**kw), 'finish_reason': 'stop', 'parse_ok': ok, 'labels': labels, 'response_text': text}


def steps(*spans):
    return [[a, b, 'right', 'reach', 'cup', None, None, 'advancing', 0.5, None] for a, b in spans]


def reply(timeline=((0.0, 30.0), (30.0, 60.0)), **fields):
    return json.dumps({'task_summary': 'Put the cup in the bin.', 'timeline': steps(*timeline), **fields})


SUCCESS = {'completion': {'task_completed': 'success'}}
# the outcome a robot request with an instruction asks for now: its completion against the instruction, and its tasks
TASKS = {**SUCCESS, 'tasks': [{'start_s': 0.0, 'end_s': 600.0, 'task': 'put the cup in the bin', 'outcome': 'success'}]}


def test_the_audit_minimal_reply_is_not_a_finished_annotation():
    record = saved(AUDIT_REPLY)
    assert record['parse_ok'] is True              # it is valid JSON with a summary
    assert not harness.annotation_complete(record)
    assert harness.annotation_problems(record) == [
        'the reply has no timeline', 'the request asks the outcome of its task, and completion.task_completed is missing']
    labels, ok = harness.read_reply(AUDIT_REPLY, request())
    assert ok is False and labels['task_summary'] == 'A demonstration was recorded.'
    assert labels['_incomplete'] == harness.annotation_problems(record)


def test_a_timeline_of_the_first_seconds_of_a_long_request_is_incomplete():
    record = saved(reply(timeline=((0.0, 1.5), (1.5, 4.5)), **SUCCESS))
    assert record['parse_ok'] is True
    assert not harness.annotation_complete(record)
    (why,) = harness.annotation_problems(record)
    assert why == 'the timeline leaves 37 of the 41 sampled instants (0 to 60 s) uncovered: 6 to 60 s'


def test_a_gap_is_measured_by_the_requests_own_sampling():
    """A step boundary between two sampled instants is within what the model saw; a skipped instant is not."""
    assert harness.annotation_complete(saved(reply(timeline=((0.0, 30.0), (31.5, 60.0)), **SUCCESS)))
    record = saved(reply(timeline=((0.0, 28.5), (31.5, 60.0)), **SUCCESS))
    assert harness.annotation_problems(record) == [
        'the timeline leaves 1 of the 41 sampled instants (0 to 60 s) uncovered: 30 s']
    # the same reply covers a request sampled every 3 s, whose instants it was given
    assert harness.annotation_complete(saved(reply(timeline=((0.0, 28.5), (31.5, 60.0)), **SUCCESS),
                                             instants=[3.0 * k for k in range(21)]))
    # a step outside the sent span covers none of it
    assert not harness.annotation_complete(saved(reply(timeline=((70.0, 90.0),), **SUCCESS)))


def test_a_single_task_reply_without_its_outcome_is_incomplete():
    for missing in ({}, {'completion': {'task_completed': None, 'reason': 'not sure'}}):
        record = saved(reply(**missing))
        assert record['parse_ok'] is True and not harness.annotation_complete(record)
        assert harness.annotation_problems(record) == [
            'the request asks the outcome of its task, and completion.task_completed is missing']


def test_a_proper_reply_is_still_a_finished_annotation():
    assert harness.annotation_complete(saved(reply(**SUCCESS)))
    assert harness.annotation_complete(saved(reply(**SUCCESS), instruction=None))
    labels, ok = harness.read_reply(reply(**SUCCESS), request())
    assert ok and '_incomplete' not in labels


def test_the_outcome_asked_for_follows_the_requests_mode_and_instructions():
    tasks = {'tasks': [{'start_s': 0.0, 'end_s': 60.0, 'task': 'put the cup in the bin', 'outcome': 'success'}]}
    # a session or a head camera asks each activity's outcome under tasks, never one completion
    for kw in ({'task_mode': 'session'}, {'rig': 'ego_head'}, {'rig': 'ego_head', 'instruction': None}):
        assert harness.annotation_complete(saved(reply(**tasks), **kw))
        assert harness.annotation_problems(saved(reply(**SUCCESS), **kw)) == [
            'the request asks each activity in tasks with its outcome, and the reply lists none']
        no_outcome = {'tasks': [{**tasks['tasks'][0], 'outcome': None}]}
        assert harness.annotation_problems(saved(reply(**no_outcome), **kw)) == ['task 0 of 1 has no outcome']
    # the field named is the one the request's own instructions declare: every request asks for its tasks, and a
    # robot episode with an instruction also for its completion against it, which holds its outcome
    for rig in ('teleop_arms', 'handheld_gripper', 'ego_head'):
        for instructed in (True, False):
            field = prompts.requested_outcome(rig, has_instruction=instructed)
            text = prompts.fixed_instructions(rig, has_instruction=instructed)
            assert field == ('completion.task_completed' if instructed and rig != 'ego_head' else 'tasks')
            assert '"tasks": [' in text and ('"task_completed"' in text) is (field != 'tasks')


def test_a_request_in_the_tasks_format_is_held_to_what_it_asks():
    tasks = {'tasks': [{'start_s': 0.0, 'end_s': 60.0, 'task': 'put the cup in the bin', 'outcome': 'success'}]}
    # a robot recording with an instruction: its completion, judged against it, and its tasks beside it, which idle
    # footage may leave empty
    assert harness.annotation_complete(saved(reply(**SUCCESS, **tasks), task_mode=None))
    assert harness.annotation_complete(saved(reply(**SUCCESS, tasks=[]), task_mode=None))
    assert harness.annotation_problems(saved(reply(**tasks), task_mode=None)) == [
        'the request asks the outcome of its task, and completion.task_completed is missing']
    # a missing tasks list, or a task with no outcome, beside a valid completion is a schema issue, never a gap
    assert harness.annotation_problems(saved(reply(**SUCCESS), task_mode=None)) == []
    no_outcome = {'tasks': [{**tasks['tasks'][0], 'outcome': None}]}
    assert harness.annotation_problems(saved(reply(**SUCCESS, **no_outcome), task_mode=None)) == []
    # with no instruction only the tasks: a robot recording may hold none (idle footage)
    assert harness.annotation_complete(saved(reply(**tasks), task_mode=None, instruction=None))
    assert harness.annotation_complete(saved(reply(tasks=[]), task_mode=None, instruction=None))
    # a head-camera clip always holds one, with or without an annotated goal, as it always did
    for instruction in ('Put the cup in the bin.', None):
        assert harness.annotation_complete(saved(reply(**tasks), rig='ego_head', task_mode=None,
                                                 instruction=instruction))
        assert not harness.annotation_complete(saved(reply(tasks=[]), rig='ego_head', task_mode=None,
                                                     instruction=instruction))
    # the record says so itself, whatever its policy names: with no instruction a completion alone no longer suffices
    record = saved(reply(**SUCCESS), task_mode='episode', instruction=None)
    assert harness.annotation_complete(record) and not harness.annotation_complete({**record, 'label_schema': 'tasks'})


def test_each_part_of_a_recording_labelled_in_parts_is_judged_against_its_own_request(tmp_path, monkeypatch):
    from test_pieces import _part_result, _three_parts
    job, src, parts = _three_parts(tmp_path, monkeypatch)
    out = job / 'run' / 'out'
    hollow = _part_result(parts[1], 'carry')
    # part 2 was sent 1 s of its own footage, and its reply covers its first 0.2 s
    hollow['config']['timesteps_s'] = [0.0, 0.25, 0.5, 0.75, 1.0]
    hollow['labels']['timeline'] = [{'start_s': 0.0, 'end_s': 0.2, 'action': 'carry'}]
    for part, record in zip(parts, (_part_result(parts[0], 'reach'), hollow, _part_result(parts[2], 'place'))):
        (out / f'{part.name}.json').write_text(json.dumps(record))
    for part in (parts[0], parts[2]):
        record, why = pieces._part_reply(out, part.name)
        assert why is None
    record, why = pieces._part_reply(out, parts[1].name)
    assert why == ("the model's reply is incomplete: the timeline leaves 3 of the 5 sampled instants (0 to 1 s) "
                   'uncovered: 0.5 to 1 s')
    res = pieces.stitch_run(job, src.parent, {src.name: [p.name for p in parts]}, tmp_path / 'final')
    assert res['incomplete'] == [src.name]
    stitched = json.loads((tmp_path / 'final' / f'{src.name}.json').read_text())
    assert [g['part'] for g in stitched['stitched']['missing']] == [2]
    assert [s['action'] for s in stitched['labels']['timeline']] == ['reach', 'place']
    assert harness.annotation_problems(stitched) == [f'part 2 has no labels: {why}']


def run_options():
    from test_dispatch_integrity_v5 import options
    return options()


def provider(monkeypatch, finals):
    """A provider that answers every final request with the next text of finals, counting what it was sent."""
    from test_dispatch_integrity_v5 import provider_kind, response
    sent = []

    def transport(request, timeout):
        kind = provider_kind(request)
        sent.append(kind)
        if kind == 'selection':
            return response(json.dumps({'requests': []}))
        if kind == 'cards':
            return response(json.dumps({'cards': []}))
        return response(finals[min(sent.count('final'), len(finals)) - 1])
    monkeypatch.setattr(harness.urllib.request, 'urlopen', transport)
    return sent


@pytest.fixture(autouse=True)
def _reader_in_process(monkeypatch):
    monkeypatch.setenv('ARGUS_READER_ISOLATION', '0')


def test_an_incomplete_reply_is_recorded_as_such_and_never_resent_by_itself(tmp_path, monkeypatch):
    from test_label import _packed_episode
    ep, _ = _packed_episode(tmp_path)
    out = tmp_path / 'out'
    proper = reply(timeline=((0.0, 600.0),), **TASKS)
    sent = provider(monkeypatch, [AUDIT_REPLY, proper])
    assert harness.run_batch([ep], out, **run_options()) == 1
    record = json.loads((out / f'{ep.name}.json').read_text())
    assert record['parse_ok'] is False and record['finish_reason'] == 'stop'
    assert record['labels']['task_summary'] == 'A demonstration was recorded.'
    assert record['labels']['_incomplete'][0] == 'the reply has no timeline'
    assert not harness.annotation_complete(record)
    # a resume keeps the paid reply and its reason, and sends nothing on its own
    assert harness.run_batch([ep], out, **run_options()) == 1
    assert sent.count('final') == 1
    marker = json.loads((out / f'stale_{ep.name}.json').read_text())
    assert marker['refresh'] == f'--retry-received {ep.name}'
    from label import run
    assert run.parsed(tmp_path, [ep.name]) == 0
    # naming it sends it again once, and a proper reply is a finished annotation
    assert harness.run_batch([ep], out, **run_options(), retry_received=[ep.name]) == 0
    assert sent.count('final') == 2
    record = json.loads((out / f'{ep.name}.json').read_text())
    assert record['parse_ok'] is True and harness.annotation_complete(record)


def test_the_board_and_the_comparison_never_count_an_incomplete_reply_as_labelled():
    from board import build, to_board
    from compare import metrics
    for record in (saved(AUDIT_REPLY), {**saved(AUDIT_REPLY), 'parse_ok': False,
                                        'labels': harness.read_reply(AUDIT_REPLY, request())[0]}):
        failed = to_board.label_failed(record)
        assert failed['status'] == 'incomplete' and failed['why'][0] == 'the reply has no timeline'
        assert failed['raw_head'] == AUDIT_REPLY
        (issue,) = build.label_failure(record)
        assert issue['kind'] == 'model_reply_incomplete'
        assert metrics.read_output(Path('episode_a.json'), record)['status'] == 'incomplete'
    proper = saved(reply(**SUCCESS))
    assert to_board.label_failed(proper) is None
    assert metrics.read_output(Path('episode_a.json'), proper)['status'] == 'parsed'


def test_reparse_reads_a_stored_incomplete_reply_with_the_same_rule():
    from label import reparse
    record = {**saved(AUDIT_REPLY), 'parse_ok': False, 'labels': harness.read_reply(AUDIT_REPLY, request())[0]}
    assert reparse.refresh_received(record)['parse_ok'] is False
    whole = {**record, 'response_text': reply(**SUCCESS)}
    assert reparse.refresh_received(whole)['parse_ok'] is True


def test_completeness_is_decided_in_one_place():
    """Every site that calls a unit done reads annotation_problems, never parse_ok alone."""
    from label import completeness
    root = Path(__file__).resolve().parents[1]
    for path in ('label/run.py', 'review/__main__.py'):
        assert 'annotation_complete' in (root / path).read_text()
    for path in ('label/pieces.py', 'board/to_board.py', 'compare/metrics.py'):
        assert 'annotation_problems' in (root / path).read_text()
    assert 'read_reply' in (root / 'label/reparse.py').read_text()
    assert completeness.annotation_gaps({}, request()) == [
        'the reply has no timeline', 'the request asks the outcome of its task, and completion.task_completed is missing']


@pytest.mark.skipif(not shutil.which('node'), reason='no node')
def test_an_unbound_finding_and_its_reason_reach_the_episode_json_the_export_and_the_board_list(tmp_path, monkeypatch):
    """README: every finding stays in the list and the JSON export. A finding no binder accepts is kept unbound with
    its reason (PR 75); it must reach the board's episode JSON, its public export and the page's finding list."""
    from test_findings_routing import gripper_finding, install
    from test_label import _packed_episode
    from board import public_export, to_board
    ep, _ = _packed_episode(tmp_path)
    inspected = {'evidence_id': 'inspection:0123456789abcdef0123', 'time_s': [1.0], 'columns': [0]}
    install(monkeypatch, lambda receipts: {
        'task_summary': 'A hand reaches.', 'timeline': steps((0.0, 600.0)), **TASKS,
        'sensor_findings': [gripper_finding(1.0, 1.0, inspected)]})
    out = tmp_path / 'out'
    assert harness.run_batch([ep], out, **run_options()) == 0
    saved_record = json.loads((out / (ep.name + '.json')).read_text())
    (row,) = saved_record['sensor_evidence']['unbound_findings']
    assert row['reason'] and row['finding']['headline'] == 'The recorded gripper value falls during pickup.'
    episode = to_board.convert(saved_record, 'demo')
    assert episode['sensor_evidence']['unbound_findings'] == [row]
    exported = public_export.project(json.loads(to_board.dumps(episode)))
    assert [(x['finding']['headline'], x['reason']) for x in exported['sensor_evidence']['unbound_findings']] == [
        (row['finding']['headline'], row['reason'])]
    source = (Path(__file__).resolve().parents[1] / 'board/index.html').read_text()
    begin = source.index('// ================= sensor evidence:')
    end = source.index('// ================= touch:', begin)
    script = ('const esc=s=>String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;");'
              'const api=new Function("esc","fmtT",process.argv[1]+"return {sensorEvidence,sensorEvidenceHtml};")'
              '(esc,String);const d=JSON.parse(process.argv[2]);'
              'console.log(api.sensorEvidenceHtml(api.sensorEvidence({...d,contacts:[]}),[]));')
    html = subprocess.check_output([shutil.which('node'), '-e', script, source[begin:end], json.dumps(exported)],
                                   text=True)
    assert 'The recorded gripper value falls during pickup.' in html
    esc = lambda s: s.replace('&', '&amp;').replace('<', '&lt;')
    assert esc(row['reason']) in html
