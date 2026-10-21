"""Resume paid replies by their actual ordered model inputs, without another call."""
import base64
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from label import harness
from label.dictionary_stage import output_spend
from test_dispatch_integrity_v5 import options, provider_kind, response
from test_label import _packed_episode
import complete_reply

# every request here inspects a recorded signal of the packed fixture, which only inspection all reaches
pytestmark = pytest.mark.usefixtures('inspect_every_field')


def paid_episode(tmp_path, monkeypatch, saved_inspection=None, **extra):
    ep, _ = _packed_episode(tmp_path)
    if saved_inspection is not None:
        from label import policy
        context = json.loads((ep / 'context.json').read_text())
        context['annotation_policy'] = policy.resolve(context, inspection=saved_inspection)
        (ep / 'context.json').write_text(json.dumps(context))
    calls = []
    def provider(request, timeout):
        kind = provider_kind(request); calls.append(kind)
        if kind == 'cards':
            return response('{"cards": []}')
        if kind == 'selection':
            content = json.loads(request.data)['messages'][0]['content']
            first = next(p['text'] for p in content if p.get('text', '').startswith('Inspect additional recorded evidence'))
            page = json.loads(first.split('FIELD INVENTORY\n', 1)[1])
            fields = [f for f in page['fields'] if f['kind'] == 'numeric']
            answer = {'requests': [{'field_id': f['id'], 'mode': 'samples'} for f in fields[:2]]}
            if calls.count('selection') > 1:
                answer = {'requests': []}
        else:
            content = json.loads(request.data)['messages'][0]['content']
            receipts = [json.loads(part['text'].split('\n', 1)[1]) for part in content
                        if part.get('text', '').startswith('INSPECTED EVIDENCE\n')]
            selected = next(row for row in receipts if row.get('times_s'))
            answer = {**complete_reply.labels('A hand reaches.'), 'evidence_findings': [{
                'start_s': selected['times_s'][0], 'end_s': selected['times_s'][-1],
                'headline': 'Recorded response changes', 'observation': 'Recorded samples across the request.',
                'claim': 'An uncalibrated recorded response changes.', 'adds_beyond_video': True,
                'confidence': 'medium', 'evidence': [{'evidence_id': selected['id'],
                    'time_s': selected['times_s'][::5]}]}]}
        return response(json.dumps(answer))
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    out = tmp_path / 'out'
    assert harness.run_batch([ep], out, **options(**extra)) == 0
    saved = json.loads((out / (ep.name + '.json')).read_text())
    assert len(saved['evidence_inspection']['inspections']) == 2
    return ep, out, calls


def test_a_run_whose_inspection_differs_from_the_saved_policy_resumes_after_a_code_change_without_payment(
        tmp_path, monkeypatch):
    # the episode's saved policy says auto, the run says all: the offline rebuild must use the run's mode, or the paid
    # request no longer matches and resuming it would pay again
    ep, out, calls = paid_episode(tmp_path, monkeypatch, saved_inspection='auto', inspection='all')
    path = out / (ep.name + '.json'); before = path.read_bytes()
    paid = output_spend(out)
    source_edit(monkeypatch, 'label/harness.py')
    assert harness.run_batch([ep], out, **options(inspection='all')) == 0
    assert calls == ['selection', 'selection', 'final', 'cards']
    assert path.read_bytes() == before and output_spend(out) == paid


def source_edit(monkeypatch, module):
    original = Path.read_bytes
    target = Path(harness.__file__).resolve().parents[1] / module
    monkeypatch.setattr(Path, 'read_bytes', lambda path:
                        original(path) + (b'\n# invoice or status presentation repair\n' if path == target else b''))


def fixture_tree(root):
    return {str(path.relative_to(root)): (path.is_dir(), path.stat().st_size, path.stat().st_mtime_ns,
                                          path.read_bytes() if path.is_file() else None)
            for path in root.rglob('*')}


@pytest.mark.parametrize('module', ['label/harness.py', 'label/run.py', 'label/evidence_access.py'])
def test_bookkeeping_source_edit_resumes_actual_paid_request_without_payment(tmp_path, monkeypatch, module):
    ep, out, calls = paid_episode(tmp_path, monkeypatch)
    path = out / (ep.name + '.json'); before = path.read_bytes()
    cache = out / '.evidence' / (path.name + '.inspection.json'); cached = cache.read_bytes()
    paid = output_spend(out)
    tree = fixture_tree(tmp_path)
    source_edit(monkeypatch, module)
    assert harness.run_batch([ep], out, **options()) == 0
    assert calls == ['selection', 'selection', 'final', 'cards']
    assert path.read_bytes() == before and cache.read_bytes() == cached
    assert output_spend(out) == paid
    assert fixture_tree(tmp_path) == tree


@pytest.mark.parametrize('change', ['prompt', 'image', 'image_order', 'meaning'])
def test_actual_prompt_image_or_field_meaning_change_still_requires_explicit_refresh(tmp_path, monkeypatch, change):
    ep, out, calls = paid_episode(tmp_path, monkeypatch)
    source_edit(monkeypatch, 'label/harness.py')
    if change == 'meaning':
        from label import evidence_access as ea
        interpret = ea.prepared_interpretation
        monkeypatch.setattr(ea, 'prepared_interpretation', lambda *args, **kwargs:
                            {**interpret(*args, **kwargs), 'meaning': 'A changed calibrated field meaning.'})
    else:
        build = harness.me.build_request
        def changed(*args, **kwargs):
            request = build(*args, **kwargs)
            if change == 'prompt':
                request['content'][0]['text'] += '\nNew instruction that changes the annotation request.'
            elif change == 'image':
                jpeg = io.BytesIO(); Image.new('RGB', (64, 36), 'red').save(jpeg, 'JPEG')
                part = next(p for p in request['content'] if p.get('type') == 'image_url')
                part['image_url']['url'] = 'data:image/jpeg;base64,' + base64.b64encode(jpeg.getvalue()).decode()
            else:
                indices = [index for index, part in enumerate(request['content']) if part.get('type') == 'image_url']
                first, last = indices[0], indices[-1]
                request['content'][first], request['content'][last] = request['content'][last], request['content'][first]
            return request
        monkeypatch.setattr(harness.me, 'build_request', changed)
    assert harness.run_batch([ep], out, **options()) == 1
    assert calls == ['selection', 'selection', 'final', 'cards']
    assert json.loads((out / ('stale_' + ep.name + '.json')).read_text())['status'] == 'stale'


@pytest.mark.parametrize('damage', ['missing_inspection', 'reordered_evidence', 'truncated_selection', 'missing_attestation'])
def test_incomplete_or_changed_saved_request_proof_fails_closed(tmp_path, monkeypatch, damage):
    ep, out, calls = paid_episode(tmp_path, monkeypatch)
    path = out / (ep.name + '.json'); saved = json.loads(path.read_text())
    if damage == 'missing_inspection':
        saved.pop('evidence_inspection')
    elif damage == 'missing_attestation':
        saved['config'].pop('request_images')
    else:
        choice = saved['evidence_inspection']['rounds'][0]['response']['choices'][0]
        if damage == 'truncated_selection':
            choice['finish_reason'] = 'length'
        else:
            answer = json.loads(choice['message']['content'])
            answer['requests'].reverse()
            choice['message']['content'] = json.dumps(answer)
    path.write_text(json.dumps(saved)); before = path.read_bytes()
    source_edit(monkeypatch, 'label/run.py')
    assert harness.run_batch([ep], out, **options()) == 1
    assert path.read_bytes() == before
    assert calls == ['selection', 'selection', 'final', 'cards']


def test_compatible_raw_reply_repairs_binding_and_retains_human_rejection_offline(tmp_path, monkeypatch):
    ep, out, calls = paid_episode(tmp_path, monkeypatch)
    path = out / (ep.name + '.json'); saved = json.loads(path.read_text())
    assert len(saved['sensor_evidence']['findings']) == 1
    finding = saved['sensor_evidence']['findings'][0]
    finding.update(review_status='rejected', time_origin_s=999, start_label='invented old binding')
    path.write_text(json.dumps(saved))
    paid = output_spend(out)
    raw, proof = saved['response_text'], saved['config']['request_images']
    source_edit(monkeypatch, 'label/run.py')
    assert harness.run_batch([ep], out, **options()) == 0
    refreshed = json.loads(path.read_text())
    finding = refreshed['sensor_evidence']['findings'][0]
    assert finding['review_status'] == 'rejected'
    assert 'time_origin_s' not in finding and 'start_label' not in finding
    assert refreshed['response_text'] == raw and refreshed['config']['request_images'] == proof
    assert output_spend(out) == paid and calls == ['selection', 'selection', 'final', 'cards']


def test_rgb_only_paid_request_uses_actual_attestation_after_bookkeeping_edit(tmp_path, monkeypatch):
    from test_route_accounting import observed_packed_episode
    ep = observed_packed_episode(tmp_path)
    calls = []
    monkeypatch.setattr(harness.urllib.request, 'urlopen', lambda request, timeout:
                        calls.append(provider_kind(request)) or response(complete_reply.text('A hand reaches.')))
    out = tmp_path / 'out'
    assert harness.run_batch([ep], out, **options()) == 0
    assert 'evidence_inspection' not in json.loads((out / (ep.name + '.json')).read_text())
    source_edit(monkeypatch, 'label/run.py')
    assert harness.run_batch([ep], out, **options()) == 0
    assert calls == ['final']


def test_history_only_paid_attestation_reuses_reply_without_changing_ledger(tmp_path, monkeypatch):
    ep, out, calls = paid_episode(tmp_path, monkeypatch)
    paid = output_spend(out)
    refused = options(); refused.update(force=True, max_spend=.000001)
    assert harness.run_batch([ep], out, **refused) == 1
    (out / (ep.name + '.json')).unlink()
    assert harness.annotation_record(out, ep.name + '.json')[1]['_authority_history_index'] == 0
    source_edit(monkeypatch, 'label/run.py')
    assert harness.run_batch([ep], out, **options()) == 0
    assert output_spend(out) == paid and calls == ['selection', 'selection', 'final', 'cards']
