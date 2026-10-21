"""Request policy must reach actual frames and receipt identity, not just a recommendation table."""
import json

import pytest

from label import episode, harness, policy
from test_label import _packed_episode


@pytest.mark.parametrize('rig', list(policy.PERIODS))
def test_model_policy_controls_real_request_cadence_and_resume_identity(tmp_path, rig):
    ep, _ = _packed_episode(tmp_path)
    path = ep / 'context.json'
    ctx = json.loads(path.read_text())
    ctx['profile'] = rig
    ctx['annotation_policy'] = policy.resolve(ctx, model=policy.SOL61)
    path.write_text(json.dumps(ctx))
    request = episode.build_request(ep, cell_w=128)
    assert request['sampling'] == f'{rig}-every-1s'
    assert 'one every 1 s' in request['prompt']
    default = harness.input_identity(ep, model=policy.SOL61, reasoning='high', max_tokens=100,
                                     cell_w=128, example_dir=None)
    ctx['annotation_policy'] = policy.resolve(ctx, sample_fps=2)
    path.write_text(json.dumps(ctx))
    dense = episode.build_request(ep, cell_w=128)
    assert dense['sampling'] == f'{rig}-every-0.5s'
    assert len(dense['timesteps']) > len(request['timesteps'])
    assert ctx['annotation_policy']['part_max_s'] == 330
    changed = harness.input_identity(ep, model=policy.SOL61, reasoning='high', max_tokens=100,
                                     cell_w=128, example_dir=None)
    assert not harness.same_input_identity(default, changed)


def test_policy_preserves_saved_context_and_explicit_model_precedence():
    ctx = {'profile': 'handheld_gripper', 'instruction': 'Several assembly tasks at this station.', 'sessions': True}
    ctx['annotation_policy'] = policy.resolve(ctx, dataset_note='An open-ended assembly session.')
    resolved = policy.resolve(ctx, model=policy.SOL61)
    assert resolved['reasoning'] == 'high' and resolved['dataset_note'] == 'An open-ended assembly session.'


@pytest.mark.parametrize('declared', [{'sessions': True}, {'task_mode': 'session'}, {'task_mode': 'episode'}, {}])
def test_a_new_policy_names_no_task_mode_whatever_the_source_declares(declared):
    """Every request asks for a list of tasks, so a source's declaration of episodes or sessions is provenance only and
    a new policy carries no task mode."""
    assert 'task_mode' not in policy.resolve({'profile': 'teleop_arms', **declared})


@pytest.mark.parametrize('mode', policy.TASK_MODES)
def test_a_saved_or_replayed_legacy_task_mode_comes_back_unchanged(mode):
    """A policy saved in the episode or session mode resolves to itself, so a Data Review job prepared then resumes
    (review/__main__.py verified_preparation compares them) and a saved request replays with its own identity."""
    saved = {**policy.resolve({'profile': 'teleop_arms'}), 'task_mode': mode}
    assert policy.resolve({'profile': 'teleop_arms', 'annotation_policy': saved}) == saved
    assert policy.resolve({'profile': 'teleop_arms'}, task_mode=mode)['task_mode'] == mode
    with pytest.raises(ValueError, match='task mode'):
        policy.resolve({'profile': 'teleop_arms'}, task_mode='sessions')


def test_the_command_line_has_no_task_mode():
    import argparse
    parser = argparse.ArgumentParser()
    policy.add_arguments(parser)
    assert not any('--task-mode' in action.option_strings for action in parser._actions)


def test_inherited_provider_policy_is_checked_before_routing(tmp_path, monkeypatch):
    ep, _ = _packed_episode(tmp_path)
    path = ep / 'context.json'
    ctx = json.loads(path.read_text())
    ctx['annotation_policy'] = policy.resolve(ctx, model='other/vendor-model')
    path.write_text(json.dumps(ctx))
    monkeypatch.setattr(harness, 'route_width', lambda *a, **k: pytest.fail('incompatible provider was routed'))
    with pytest.raises(ValueError, match='requires an OpenRouter key'):
        harness.label_episode(ep, tmp_path / 'label.json', model=None, reasoning=None,
                              api_key='sk-proj-test', max_tokens=100, timeout=10)
    assert not (tmp_path / 'label.json').exists()


def test_only_astra_defaults_to_medium_reasoning_and_every_other_model_to_high():
    ctx = {'profile': 'teleop_arms'}
    assert policy.resolve(ctx)['reasoning'] == 'medium'                         # Astra, the default model
    for model in ('google/gemini-4-argon', policy.SOL61, 'custom/model'):
        assert policy.resolve(ctx, model=model)['reasoning'] == 'high', model
        assert policy.resolve(ctx, model=model, reasoning='low')['reasoning'] == 'low'
    saved = {**ctx, 'annotation_policy': policy.resolve(ctx, model='google/gemini-4-argon', reasoning='medium')}
    assert policy.resolve(saved)['reasoning'] == 'medium'                       # a saved choice is kept on resume


def test_the_paid_calls_beside_the_annotation_are_on_by_default_and_written_only_when_turned_off(tmp_path):
    ctx = {'profile': 'teleop_arms'}
    default = policy.resolve(ctx)
    assert default['inspection'] == 'auto' and 'field_dictionary' not in default     # a new policy writes its mode
    assert policy.resolve(ctx, inspection='auto', field_dictionary=True) == default
    off = policy.resolve(ctx, inspection='off', field_dictionary=False)
    assert off == {**default, 'inspection': 'off', 'field_dictionary': False}
    assert policy.resolve({**ctx, 'annotation_policy': off}) == off             # a saved choice is kept on resume
    assert policy.resolve({**ctx, 'annotation_policy': off}, inspection='auto', field_dictionary=True) == default
    # a policy saved before the setting existed inspected every recorded field, and keeps doing so on resume, with
    # the saved policy unchanged, so a job labelled then rebinds its paid replies instead of refusing them as stale
    legacy = {key: value for key, value in default.items() if key != 'inspection'}
    assert policy.resolve({**ctx, 'annotation_policy': legacy}) == legacy
    assert policy.inspection_mode(legacy) == 'all' and policy.inspection_mode(default) == 'auto'
    assert 'inspection' not in policy.resolve(ctx, inspection='all')
    for bad in ('none', 0, True, 'Off'):
        with pytest.raises(ValueError):
            policy.resolve(ctx, inspection=bad)
    with pytest.raises(ValueError):
        policy.resolve(ctx, field_dictionary='off')
    ep, _ = _packed_episode(tmp_path)
    kw = dict(model=policy.SOL61, reasoning='high', max_tokens=100, cell_w=128, example_dir=None)
    same = harness.input_identity(ep, **kw, inspection='auto')
    assert harness.same_input_identity(harness.input_identity(ep, **kw), same)
    assert not harness.same_input_identity(same, harness.input_identity(ep, **kw, inspection='off'))


def test_every_reader_of_a_saved_policy_takes_its_task_mode_as_optional():
    """A policy saved without a task mode (every policy from the tasks format on, label/policy.py resolve) is read
    by request replay (grids, request attestation, resume's input check), parts and the request record alike; a
    subscript read would raise KeyError, which resume swallows into "inputs changed" and refuses every dispatch."""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    reads = [f'{path.relative_to(root)}:{n}' for path in sorted((root / 'label').glob('*.py'))
             for n, line in enumerate(path.read_text().splitlines(), 1)
             if re.search(r"""\[\s*['"]task_mode['"]\s*\]""", line) and not re.search(r"""\[\s*['"]task_mode['"]\s*\]\s*=""", line)]
    assert reads == []
