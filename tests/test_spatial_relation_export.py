"""A timeline step's spatial_relation survives the board export next to its destination."""
from board import label_projection, lerobot_export, to_board


def _events(*steps):
    projected, dropped = label_projection.project({'timeline': list(steps)})
    assert not dropped
    return to_board.convert({'labels': projected})['event_labels']


def test_relation_kept_beside_destination_and_carry_phase_unchanged():
    both, only_relation, neither = _events(
        {'start_s': 0, 'end_s': 1, 'arm': 'right', 'action': 'drop the lemon', 'object': 'lemon',
         'destination': 'wire basket', 'spatial_relation': 'inside basket'},
        {'start_s': 1, 'end_s': 2, 'arm': 'left', 'action': 'hold the cloth', 'object': 'cloth',
         'spatial_relation': 'around pineapple'},
        {'start_s': 2, 'end_s': 3, 'arm': 'left', 'action': 'wave'})
    assert both['spatial_relation'] == 'inside basket' and both['carry_phase'] == '-> wire basket'
    # a step with only a relation keeps today's fallback: the relation stands in for the destination
    assert only_relation['spatial_relation'] == 'around pineapple' and only_relation['carry_phase'] == '-> around pineapple'
    assert neither['spatial_relation'] == '' and neither['carry_phase'] == ''
    # the LeRobot export reads the destination out of carry_phase, so its subtask text is unchanged
    assert lerobot_export.step_text(both) == 'drop the lemon -> wire basket'
    assert lerobot_export.step_text(only_relation) == 'hold the cloth -> around pineapple'
