"""Signal descriptions must not invent clock facts for an unknown placement."""
import numpy as np
import pytest

from label import signals
from prepare.signal_alignment import ALIGNED_ASSUMED, ALIGNED_CAMERA, ALIGNED_ROWS, COARSE_CLOCK, placement_text


@pytest.mark.parametrize('marker', ['future placement', '__proto__', 'constructor'])
def test_unknown_signal_placement_is_qualified_without_clock_claims(marker):
    values = np.ones((40, 1), dtype=np.float32)
    original = values.tobytes()
    text = signals.describe('right_force', values, aligned_by=marker)
    assert text == f'  right_force (1 value, {placement_text(marker)}): 1 throughout'
    assert 'both starts' not in text and 'no clock is shared' not in text
    assert values.tobytes() == original


@pytest.mark.parametrize(('marker', 'words'), [
    (None, ''),
    ('', ''),
    (ALIGNED_ASSUMED, ', placed from both starts as no clock is shared'),
    (ALIGNED_ROWS, ', placed one row per frame as it has as many rows as the video has frames'),
    (COARSE_CLOCK, ', tied readings placed within each stamp interval as an assumption'),
    (ALIGNED_CAMERA, ', placed on footage through its assumed camera presentation clock'),
])
def test_known_signal_placement_keeps_existing_description(marker, words):
    text = signals.describe('right_force', np.ones((40, 1)), aligned_by=marker)
    assert text == f'  right_force (1 value{words}): 1 throughout'
