import json

import numpy as np

from label import sensor_evidence as se
from tests.test_tactile_contract import episode


def test_many_sensor_pads_each_receive_evidence_within_the_prompt_budget():
    n = 120
    signals = {f'tactile_pad_{i}': np.c_[np.zeros(n), np.arange(n) * (i + 1)] for i in range(34)}
    ep = episode(signals, [{'name': name, 'shape': [2]} for name in signals])
    doc = se.build(ep, {'n': n})
    sent = json.loads(se.prompt(doc).split('\n')[-1])
    assert {s['name'] for s in sent['sensors']} == set(signals)
    assert {r['sensor_id'] for r in sent['series']} == {s['id'] for s in sent['sensors']}
    assert len(se.prompt(doc).encode()) <= se.MAX_PROMPT_BYTES


def test_coverage_reports_missing_and_unsupported_tactile_channels():
    signals = {'tactile_pressure': np.arange(5).reshape(5, 1), 'tactile_vibration': np.ones((5, 1))}
    ep = episode(signals, [{'name': name, 'shape': [1]} for name in [*signals, 'tactile_unread']],
                 {'tactile_vibration': {'role': 'vibration'}})
    doc = se.build(ep, {'n': 5})
    coverage = {r['name']: r for r in doc['coverage']}
    assert coverage['tactile_pressure']['status'] == 'sampled'
    assert coverage['tactile_vibration']['status'] == 'unsupported interpretation'
    assert coverage['tactile_unread']['status'] == 'readings unavailable'


def test_small_mixed_measurements_keep_pressure_and_temperature_columns():
    n = 120
    signals = {'pressure_topic': np.c_[np.arange(n) * 1000, np.linspace(20, 20.1, n)]}
    signals.update({f'tactile_pad_{i}': np.tile(np.arange(n)[:, None], (1, 25)) for i in range(6)})
    ep = episode(signals, [{'name': name, 'shape': [a.shape[1]],
                          **({'names': ['pressure', 'temperature']} if name == 'pressure_topic' else {})}
                         for name, a in signals.items()])
    doc = se.build(ep, {'n': n})
    assert [r['label'] for r in doc['series'] if r['sensor_id'] == 'signal:pressure_topic'] == ['pressure', 'temperature']


def test_lens_check_uses_original_aspect_ratio_for_full_and_shrunk_frames():
    from PIL import Image
    from label import frames, lens
    image = Image.new('RGB', (432, 240), 'white')
    small = frames.Shrunk(image, [384])
    assert lens.thumb(image).shape == lens.thumb(small).shape
    assert lens.circular_image({0: image, 1: small})['frames'] == 2


def test_dictionary_truncation_keeps_only_complete_entries_and_original_reply():
    from label import dictionary as dd
    inv = {'fields': [{'id': 'first'}, {'id': 'second'}], 'digest': 'test'}
    record = dd._base(inv, 'pending')
    raw = '{"entries":[{"id":"first","meaning":"Recorded pressure","role":"pressure"},{"id":"second","meaning":"Unfinished'
    record.update(raw_text=raw, finish_reason='length')
    dd._parse(record)
    assert set(record['entries']) == {'first'}
    assert record['missing_fields'] == ['second']
    assert record['raw_text'] == raw
    assert record['response_truncated'] is True


def test_dictionary_damaged_json_without_truncation_is_not_recovered():
    import pytest
    from label import dictionary as dd
    record = dd._base({'fields': [{'id': 'first'}], 'digest': 'test'}, 'pending')
    record['raw_text'] = '{"entries":[{"id":"first","meaning":"Pressure","role":"pressure"},INVALID'
    with pytest.raises(ValueError):
        dd._parse(record)


def test_truncated_dictionary_receipt_is_partial_and_never_resends(tmp_path):
    from label import dictionary as dd
    from tests.test_dictionary import episode as dictionary_episode
    ep = dictionary_episode(tmp_path, 'episode', [('pressure', np.ones((2, 1)), {'shape': [1]})])
    inv = dd.inventory([ep])
    field = next(f for f in inv['fields'] if f['name'] == 'pressure')
    raw = '{"entries":[' + json.dumps({'id': field['id'], 'meaning': 'Recorded pressure', 'role': 'pressure'}) + ',{"id":"unfinished'
    calls = []
    def fake(*args, **kwargs):
        calls.append(kwargs)
        return {'choices': [{'message': {'content': raw}, 'finish_reason': 'length'}], 'usage': {'cost': .01}}
    job = tmp_path / 'job'
    first = dd.prepare_upload(job, [ep], 'fake', fake)
    assert first['status'] == 'partial' and first['raw_text'] == raw
    assert first['entries'][field['id']]['role'] == 'pressure'
    assert dd.prepare_upload(job, [ep], 'fake', fake) == first
    assert len(calls) == 1
