"""Every row of a reply's findings lists is bound or kept with the reason it was not.

Two paid FastUMI replies put inspection-backed gripper findings in sensor_findings instead of evidence_findings.
The request had no structured sensor evidence, so the sensor binder never ran, the inspection binder read only
evidence_findings, and nine valid findings left no trace. These tests send the same reply shape through the real
harness and the stored-reply rebinder.
"""
import copy
import json

import pytest

from label import harness, reparse
from label import sensor_evidence as se
from test_dispatch_integrity_v5 import options, provider_kind, response
from test_label import _packed_episode
import complete_reply


@pytest.fixture(autouse=True)
def _reader_in_process(monkeypatch):
    monkeypatch.setenv('ARGUS_READER_ISOLATION', '0')


def gripper_finding(start, end, ref):
    """The FastUMI reply row: sensor_findings copy fields, an inspection citation with time_s and columns."""
    return {'start_s': start, 'end_s': end, 'headline': 'The recorded gripper value falls during pickup.',
            'detail': 'One side falls much farther than the other.',
            'observation': 'The value falls from 0.96 to 0.01 across the cited samples.',
            'claim': 'The recorded gripper coordinate changes during pickup.',
            'alternative': 'It may be a normalized estimate rather than jaw separation.',
            'confidence': 'high', 'adds_beyond_video': True, 'evidence': [ref]}


def install(monkeypatch, final):
    """A provider that asks to inspect two numeric fields, then answers the final request with final(receipts)."""
    calls = []

    def provider(request, timeout):
        kind = provider_kind(request)
        calls.append(kind)
        content = json.loads(request.data)['messages'][0]['content']
        if kind == 'selection':
            first = next(p['text'] for p in content if p.get('text', '').startswith('Inspect additional recorded evidence'))
            page = json.loads(first.split('FIELD INVENTORY\n', 1)[1])
            fields = [f for f in page['fields'] if f['kind'] == 'numeric']
            answer = {'requests': [] if calls.count('selection') > 1 else
                      [{'field_id': f['id'], 'mode': 'samples'} for f in fields[:2]]}
        elif kind == 'cards':
            answer = {'cards': []}
        else:
            receipts = [json.loads(part['text'].split('\n', 1)[1]) for part in content
                        if part.get('text', '').startswith('INSPECTED EVIDENCE\n')]
            answer = final(receipts)
        return response(json.dumps(answer))
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    return calls


@pytest.mark.usefixtures('inspect_every_field')
def test_an_inspection_backed_finding_written_in_sensor_findings_is_bound(tmp_path, monkeypatch):
    ep, _ = _packed_episode(tmp_path)

    def final(receipts):
        selected = next(row for row in receipts if row.get('times_s') and row.get('columns'))
        ref = {'evidence_id': selected['id'], 'time_s': selected['times_s'][::5], 'columns': selected['columns'][:2]}
        return {**complete_reply.labels('A hand reaches.'), 'evidence_findings': [],
                'sensor_findings': [gripper_finding(selected['times_s'][0], selected['times_s'][-1], ref)]}
    install(monkeypatch, final)
    out = tmp_path / 'out'
    assert harness.run_batch([ep], out, **options()) == 0
    saved = json.loads((out / (ep.name + '.json')).read_text())
    evidence = saved['sensor_evidence']
    assert [f['headline'] for f in evidence['findings']] == ['The recorded gripper value falls during pickup.']
    assert evidence['unbound_findings'] == []
    assert evidence['routed_findings'] == [{'field': 'sensor_findings', 'row': 0, 'bound_as': 'evidence_findings'}]

    # A record saved before the routing existed is repaired by rebinding its stored reply, with no model call.
    old = copy.deepcopy(saved)
    old['sensor_evidence'] = {k: v for k, v in old['sensor_evidence'].items() if k != 'routed_findings'}
    old['sensor_evidence']['findings'] = []
    reparse.rebind(old)
    assert [f['headline'] for f in old['sensor_evidence']['findings']] == ['The recorded gripper value falls during pickup.']
    assert old['sensor_evidence']['unbound_findings'] == []


def test_findings_with_no_binder_in_the_request_are_kept_with_the_reason(tmp_path, monkeypatch):
    ep, _ = _packed_episode(tmp_path)
    inspected = {'evidence_id': 'inspection:0123456789abcdef0123', 'time_s': [1.0], 'columns': [0]}
    supplied = {'sensor_id': 'touch', 'series_id': 'touch:0', 'time_s': [1.0]}
    install(monkeypatch, lambda receipts: {
        **complete_reply.labels('A hand reaches.'), 'sensor_findings': [gripper_finding(1.0, 1.0, inspected)],
        'evidence_findings': [gripper_finding(1.0, 1.0, supplied)]})
    out = tmp_path / 'out'
    assert harness.run_batch([ep], out, **options()) == 0
    saved = json.loads((out / (ep.name + '.json')).read_text())
    assert 'evidence_inspection' not in saved or not saved['evidence_inspection'].get('inspections')
    unbound = saved['sensor_evidence']['unbound_findings']
    assert [(row['field'], row['row']) for row in unbound] == [('sensor_findings', 0), ('evidence_findings', 0)]
    assert all(row['reason'] and row['finding']['headline'] for row in unbound)
    assert saved['sensor_evidence']['findings'] == []


def test_routing_follows_the_citations_and_never_loses_a_row():
    inspected = {'evidence_id': 'inspection:a', 'time_s': [1.0]}
    supplied = {'sensor_id': 's', 'time_s': [1.0]}
    labels = {'sensor_findings': [{'evidence': [inspected]}, {'evidence': [supplied]}, {'evidence': []}, 'text'],
              'evidence_findings': [{'evidence': [supplied]}, {'evidence': [inspected, supplied]}]}
    for sensor, inspection in ((True, True), (True, False), (False, True), (False, False)):
        routes = se.route_findings(labels, sensor_evidence=sensor, inspection=inspection)
        assert (len(routes['sensor_findings']) + len(routes['evidence_findings']) + len(routes['unbound'])
                == len(labels['sensor_findings']) + len(labels['evidence_findings']))
    routes = se.route_findings(labels, sensor_evidence=True, inspection=True)
    assert routes['evidence_findings'] == [{'evidence': [inspected]}, {'evidence': [inspected, supplied]}]
    assert routes['sensor_findings'] == [{'evidence': [supplied]}, {'evidence': []}, 'text', {'evidence': [supplied]}]
