import pytest

from label import policy


@pytest.mark.parametrize('model', [policy.ASTRA, policy.SOL61, 'custom/model'])
def test_new_ego_policy_defaults_to_two_fps_for_every_model(model):
    resolved = policy.resolve({'profile': 'ego_head'}, model=model)
    assert resolved['sample_every_s'] == 0.5


def test_changing_to_sol_keeps_the_ego_default_at_two_fps():
    context = {'profile': 'ego_head'}
    context['annotation_policy'] = policy.resolve(context, model=policy.ASTRA)
    assert policy.resolve(context, model=policy.SOL61)['sample_every_s'] == 0.5


def test_saved_one_fps_ego_run_retains_its_paid_request_cadence():
    saved = {**policy.resolve({'profile': 'ego_head'}, model=policy.SOL61),
             'sample_every_s': 1.0, 'part_max_s': 450.0}
    context = {'profile': 'ego_head', 'annotation_policy': saved}
    assert policy.resolve(context) == saved
    assert policy.resolve(context, sample_fps=2)['sample_every_s'] == 0.5


@pytest.mark.parametrize('rig', ['teleop_arms', 'handheld_gripper'])
def test_sol_other_rigs_keep_their_one_fps_default(rig):
    assert policy.resolve({'profile': rig}, model=policy.SOL61)['sample_every_s'] == 1.0
