"""Invalid saved times retain their labels without requesting an impossible frame."""
import copy
import json

import pytest

from board import static
from board.to_board import _time


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('field', ['completion', 'task', 'reached'])
def test_nonfinite_saved_goal_times_do_not_request_frames_or_change_labels(value, field):
    if field == 'task':
        saved = {'tasks': [{'task': 'lift', 'completed_at_s': value}]}
    elif field == 'reached':
        saved = {'completion': {'task_completed': 'success_then_undone', 'goal_reached_at_s': value}}
    else:
        saved = {'completion': {'completed_at_s': value}}
    original = json.dumps(saved)
    assert _time(value) is None
    assert static.goal_times(saved) == []
    assert json.dumps(saved) == original


def test_finite_saved_goal_requests_keep_their_exact_times_and_order():
    saved = {'tasks': [{'task': 'first', 'completed_at_s': 0.123456789},
                       {'task': 'second', 'completed_at_s': '2.34567891'},
                       {'task': 'invalid', 'completed_at_s': float('nan')}]}
    original = copy.deepcopy(saved)
    assert static.goal_times(saved) == [0.123456789, 2.34567891]
    assert static.ms_key(static.goal_times(saved)[0]) == 123
    assert static.ms_key(static.goal_times(saved)[1]) == 2346
    assert json.dumps(saved) == json.dumps(original)


def test_invalid_numeric_strings_do_not_request_goal_frames():
    saved = {'tasks': [{'task': 'lift', 'completed_at_s': value}
                       for value in ('nan', 'inf', '-inf', 'unknown', None)]}
    assert static.goal_times(saved) == []


def test_integer_outside_float_range_does_not_request_a_goal_frame():
    value = 10 ** 1000
    saved = {'completion': {'completed_at_s': value}}
    assert static.goal_times(saved) == []
    assert saved['completion']['completed_at_s'] == value
