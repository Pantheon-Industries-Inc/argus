"""The hand pose 2D readout (board/hand_pose/core.py mode_soft_argmax): one hand per slot, never a point between two."""
from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from board.hand_pose import core  # noqa: E402

H, W, J = 21, 26, 21


def plain_soft_argmax(a, h, w):
    """ACE-Ego-Hand's own readout (MemoryAlternatingEncoder._soft_argmax), for comparison."""
    hm = a.reshape(a.shape[0], a.shape[1], h, w)
    us, vs = torch.linspace(0.0, 1.0, w), torch.linspace(0.0, 1.0, h)
    s = hm.sum(dim=(-1, -2)).clamp(min=1e-6)
    return torch.stack([(hm.sum(-2) * us).sum(-1) / s, (hm.sum(-1) * vs).sum(-1) / s], -1)


def blob(cy, cx, sigma=1.0):
    ii, jj = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    return torch.exp(-((ii - cy) ** 2 + (jj - cx) ** 2) / (2 * sigma ** 2))


def maps(slot0, slot1):
    a = torch.stack([slot0] * J + [slot1] * J).reshape(1, 2 * J, H * W)
    return a / a.sum(-1, keepdim=True)


def cells(uv):
    return uv * torch.tensor([W - 1, H - 1], dtype=uv.dtype)


def test_one_hand_per_slot_reads_as_the_model_does():
    a = maps(blob(5, 6), blob(12, 18))
    assert torch.allclose(core.mode_soft_argmax(a, H, W, 2, radius=3), plain_soft_argmax(a, H, W), atol=1e-4)


def test_attention_split_between_two_hands_reads_the_stronger_hand():
    # slot 1 attends 60/40 to a hand at column 8 and another hand at column 20, both on row 12
    a = maps(blob(5, 6), 0.6 * blob(12, 8) + 0.4 * blob(12, 20))
    plain = cells(plain_soft_argmax(a, H, W))[0, J]
    mode = cells(core.mode_soft_argmax(a, H, W, 2, radius=3))[0, J]
    assert 12 < plain[0] < 16                       # the model's own mean lands between the hands, on neither
    assert abs(mode[0] - 8) < 0.1 and abs(mode[1] - 12) < 0.1
    # slot 0 is untouched by slot 1's second hand
    assert torch.allclose(core.mode_soft_argmax(a, H, W, 2, 3)[0, :J], plain_soft_argmax(a, H, W)[0, :J], atol=1e-4)


def test_set_readout_installs_and_restores():
    class Enc:
        num_slots = 2

        @staticmethod
        def _soft_argmax(a, h, w):
            return plain_soft_argmax(a, h, w)

    enc = Enc()
    model = types.SimpleNamespace(net=types.SimpleNamespace(projectors={"l15": types.SimpleNamespace(
        alternating_encoder=enc)}))
    a = maps(blob(5, 6), 0.6 * blob(12, 8) + 0.4 * blob(12, 20))
    assert core.set_readout(model, 3) == [enc]
    assert torch.allclose(enc._soft_argmax(a, H, W), core.mode_soft_argmax(a, H, W, 2, 3))
    core.set_readout(model, None)
    assert torch.allclose(enc._soft_argmax(a, H, W), plain_soft_argmax(a, H, W))


def _hand(cx, cy, size, n=1):
    """n frames of one slot: a fixed 21-point hand shape centred on (cx, cy) in pixels of a 100 x 100 pinhole."""
    import numpy as np
    rng = np.random.default_rng(0)
    shape = rng.uniform(-0.5, 0.5, (21, 2))
    shape -= shape.mean(0)
    return np.repeat((np.array([cx, cy]) + size * shape)[None] / 100.0, n, 0)


def test_combine_keeps_the_model_where_the_readouts_agree_and_replaces_a_blend():
    import numpy as np
    pin = {"image_width": 100, "image_height": 100}
    n = 60
    # slot 0: both readouts on the same hand, the peak one at half size (as mode_soft_argmax draws it)
    m0, p0 = _hand(30, 30, 20, n), _hand(30, 30, 10, n)
    # slot 1: the same for 59 frames, then one frame where the model's own reading sits between two hands
    m1, p1 = _hand(70, 70, 20, n), _hand(70, 70, 10, n)
    m1[-1] = _hand(50, 50, 20)[0]
    jm = np.stack([m0, m1], 1); jp = np.stack([p0, p1], 1)
    j, info = core.combine_readouts(jm, jp, np.ones((n, 2), bool), pin)
    assert np.allclose(j[:-1], jm[:-1]) and np.allclose(j[-1, 0], jm[-1, 0])
    assert info["k"] == [2.0, 2.0]
    assert np.allclose(j[-1, 1], _hand(70, 70, 20)[0])        # the peak reading, scaled back to the hand's size
    assert info["replaced"] == round(1 / (2 * n), 4)
