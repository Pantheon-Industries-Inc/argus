"""Retain typed native numeric leaves separately from judgment signal projections."""
from __future__ import annotations

import numpy as np


PROTO_DTYPES = {1: 'float64', 2: 'float32', 3: 'int64', 4: 'uint64', 5: 'int32',
                6: 'uint64', 7: 'uint32', 8: 'bool', 13: 'uint32', 14: 'int32',
                15: 'int32', 16: 'int64', 17: 'int32', 18: 'int64'}


def native_fields(message):
    """Exact native path to numeric array, source dtype, shape and presence descriptor."""
    out = {}

    def add(path, value, dtype=None, present=True, dtype_source='decoded Python type'):
        array = np.asarray(value, dtype=dtype)
        out[path] = {'values': array, 'dtype': str(array.dtype), 'shape': list(array.shape),
                     'present': present, 'dtype_source': dtype_source}

    def walk(value, path=''):
        descriptor = getattr(value, 'DESCRIPTOR', None)
        if descriptor is not None and hasattr(descriptor, 'fields'):
            present = {field.name for field, _ in value.ListFields()}
            for field in descriptor.fields:
                key = path + '.' + field.name if path else field.name
                cell = getattr(value, field.name)
                if field.type in PROTO_DTYPES:
                    add(key, list(cell) if field.is_repeated else cell,
                        PROTO_DTYPES[field.type], field.name in present, 'protobuf declaration')
                elif field.message_type is not None and field.name in present:
                    if field.is_repeated:
                        for index, item in enumerate(cell):
                            walk(item, f'{key}[{index}]')
                    else:
                        walk(cell, key)
            return
        if isinstance(value, (bool, np.bool_)):
            add(path, value, 'bool')
        elif isinstance(value, (int, np.integer)):
            dtype = 'int64' if -(2**63) <= value < 2**63 else 'uint64' if 0 <= value < 2**64 else None
            if dtype:
                add(path, value, dtype)
            else:
                out[path] = {'values': None, 'dtype': 'integer outside uint64 and int64',
                             'shape': [], 'original': str(value), 'present': True}
        elif isinstance(value, (float, np.floating)):
            add(path, value, value.dtype if isinstance(value, np.floating) else 'float64')
        elif isinstance(value, dict):
            for name, item in value.items():
                walk(item, path + '.' + str(name) if path else str(name))
        elif isinstance(value, (list, tuple, np.ndarray)):
            # Mixed lists retain separate leaves instead of rounding integers into a float array.
            for index, item in enumerate(value):
                walk(item, f'{path}[{index}]')
        elif value is not None and not isinstance(value, (str, bytes, bytearray, memoryview)):
            from prepare.formats import _msg_items
            for name, item, _ in _msg_items(value):
                walk(item, path + '.' + str(name) if path else str(name))

    walk(message)
    return out
