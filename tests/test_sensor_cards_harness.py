"""The video card call comes after a paid, bound annotation: nothing it does may cost that annotation."""
import json

import pytest

from label import harness, policy, sensor_cards
from label.dictionary_stage import output_spend
from test_dispatch_integrity_v5 import options, provider_kind, response
from test_label import _packed_episode
from test_request_attestation import source_edit
import complete_reply

# every request here inspects a recorded signal of the packed fixture, which only inspection all reaches
pytestmark = pytest.mark.usefixtures('inspect_every_field')


@pytest.fixture(autouse=True)
def _reader_in_process(monkeypatch):
    monkeypatch.setenv('ARGUS_READER_ISOLATION', '0')


def install(monkeypatch, calls, cards):
    def provider(request, timeout):
        kind = provider_kind(request)
        calls.append(kind)
        if kind == 'cards':
            return cards()
        content = json.loads(request.data)['messages'][0]['content']
        if kind == 'selection':
            first = next(p['text'] for p in content if p.get('text', '').startswith('Inspect additional recorded evidence'))
            page = json.loads(first.split('FIELD INVENTORY\n', 1)[1])
            fields = [f for f in page['fields'] if f['kind'] == 'numeric']
            answer = {'requests': [] if calls.count('selection') > 1 else
                      [{'field_id': f['id'], 'mode': 'samples'} for f in fields[:2]]}
        else:
            receipts = [json.loads(part['text'].split('\n', 1)[1]) for part in content
                        if part.get('text', '').startswith('INSPECTED EVIDENCE\n')]
            selected = next(row for row in receipts if row.get('times_s'))
            answer = {**complete_reply.labels('A hand reaches.'), 'evidence_findings': [{
                'start_s': selected['times_s'][0], 'end_s': selected['times_s'][-1],
                'headline': 'Recorded response changes', 'observation': 'Recorded samples across the request.',
                'claim': 'An uncalibrated recorded response changes.', 'adds_beyond_video': True,
                'confidence': 'medium', 'evidence': [{'evidence_id': selected['id'], 'time_s': selected['times_s'][::5]}]}]}
        return response(json.dumps(answer))
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)


def lost():
    raise TimeoutError('card response lost')


@pytest.mark.parametrize('failure', ['unknown outcome', 'spend cap'])
def test_a_failed_card_call_leaves_the_paid_annotation_current_after_a_code_change(tmp_path, monkeypatch, failure):
    ep, _ = _packed_episode(tmp_path)
    calls = []
    install(monkeypatch, calls, lost if failure == 'unknown outcome' else (lambda: response('{"cards": []}')))
    if failure == 'spend cap':
        monkeypatch.setattr(sensor_cards, 'ANSWER_TOKENS', 10 ** 6)   # the card reservation exceeds the cap; the final fits
    out = tmp_path / 'out'
    assert harness.run_batch([ep], out, **options()) == 0
    saved = json.loads((out / (ep.name + '.json')).read_text())
    assert saved['parse_ok'] is True and saved['evidence_inspection']['status'] == 'complete'
    assert saved['evidence_inspection']['limitations'] == []
    paid, before = output_spend(out), list(calls)
    monkeypatch.setattr(sensor_cards, 'ANSWER_TOKENS', 16000)
    source_edit(monkeypatch, 'label/run.py')
    assert harness.run_batch([ep], out, **options()) == 0
    assert not (out / ('stale_' + ep.name + '.json')).exists()
    assert calls == before and output_spend(out) == paid


def test_the_card_call_is_checked_against_what_the_final_call_billed_not_its_bound(tmp_path, monkeypatch):
    # At this cap the final call's bound ($1.36) plus the card bound ($1.05) exceeds the cap, while the billed final
    # ($0.003) plus the card bound fits easily: an unsettled final bound refused these cards. The cap also holds each
    # selection reservation beside the final call's bound ($0.66 + $1.01).
    bounds = []
    real = harness.final_cost_bound
    def spy(content, max_tokens, **kw):
        value = real(content, max_tokens, **kw)
        bounds.append((max_tokens, value))
        return value
    monkeypatch.setattr(harness, 'final_cost_bound', spy)
    ep, _ = _packed_episode(tmp_path)
    calls = []
    install(monkeypatch, calls, lambda: response('{"cards": []}'))
    out = tmp_path / 'out'
    assert harness.run_batch([ep], out, **{**options(), 'max_tokens': 12000, 'max_spend': 1.9}) == 0
    final_bound = max(v for t, v in bounds if t == 12000)
    card_bound = next(v for t, v in bounds if t == policy.output_limit(sensor_cards.ANSWER_TOKENS, 'medium'))
    assert final_bound + card_bound > 1.9 > output_spend(out) + card_bound
    assert 'cards' in calls


class Killed(BaseException):
    """Stands in for the process dying while the card request is in flight."""


def test_a_process_death_during_the_card_call_keeps_the_bound_annotation(tmp_path, monkeypatch):
    ep, _ = _packed_episode(tmp_path)
    calls = []
    def die():
        raise Killed()
    install(monkeypatch, calls, die)
    out = tmp_path / 'out'
    with pytest.raises(Killed):
        harness.run_batch([ep], out, **options())
    saved = json.loads((out / (ep.name + '.json')).read_text())
    assert saved['parse_ok'] is True and saved['sensor_evidence']['findings']
    before = list(calls)
    install(monkeypatch, calls, lambda: response('{"cards": []}'))
    assert harness.run_batch([ep], out, **options()) == 0
    assert 'final' not in calls[len(before):], 'the paid annotation is never sent again'
