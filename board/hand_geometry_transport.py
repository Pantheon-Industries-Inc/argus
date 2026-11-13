"""Optional display transport with at most 0.0005 pixel XY and 0.0000005 meter Z error.

Only joint and vertex coordinates change. Native files, downloads, identity, presence,
frame times, topology and missing hands stay unchanged. The canonical JSON decoder applies.
"""
from copy import deepcopy
import math

XY_MAX_ERROR = .0005
Z_MAX_ERROR = .0000005


def compact_geometry(doc: dict) -> dict:
    """Copy a canonical header or chunk for display; never mutate the source document."""
    if not isinstance(doc, dict) or doc.get('format') not in (
            'board-hand-geometry/1', 'board-hand-geometry-chunk/1'):
        raise ValueError('Unsupported hand geometry transport')
    result = deepcopy(doc)
    if doc['format'] == 'board-hand-geometry/1':
        return result
    if not isinstance(result.get('frames'), list):
        raise ValueError('Invalid hand geometry frames')
    for frame in result['frames']:
        if not isinstance(frame, dict):
            raise ValueError('Invalid hand geometry frame')
        for side in ('left', 'right'):
            hand = frame.get(side)
            if hand is None:
                continue
            if not isinstance(hand, dict):
                raise ValueError('Invalid hand geometry hand')
            for field in ('joints', 'vertices'):
                points = hand.get(field)
                if points is None:
                    continue
                if not isinstance(points, list):
                    raise ValueError('Invalid source-camera coordinates')
                for point in points:
                    if (not isinstance(point, list) or len(point) != 3
                            or any(not isinstance(v, (int, float)) or isinstance(v, bool)
                                   or not math.isfinite(v) for v in point) or point[2] <= 0):
                        raise ValueError('Invalid source-camera coordinates')
                    for axis, value in enumerate(point):
                        limit = Z_MAX_ERROR if axis == 2 else XY_MAX_ERROR
                        rounded = round(value, 6 if axis == 2 else 3)
                        # Preserve tiny positive depths and exact values at floating point rounding boundaries.
                        if abs(rounded - value) <= limit and (axis != 2 or rounded > 0):
                            point[axis] = rounded
    return result
