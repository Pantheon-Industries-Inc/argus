"""Bounded reads of original HDF5 metadata, without duplicating source arrays."""
from pathlib import Path

import numpy as np

MAX_VALUES = 4096


def value(x):
    if isinstance(x, bytes):
        return x.decode('utf-8', 'replace')
    if isinstance(x, np.ndarray):
        if x.size > MAX_VALUES:
            return {'shape': list(x.shape), 'dtype': str(x.dtype), 'truncated': True,
                    'preview': [value(y) for y in x.ravel()[:50]],
                    'note': 'The complete metadata array remains in the original HDF5 file.'}
        return value(x[()]) if x.ndim == 0 else [value(y) for y in x]
    if isinstance(x, np.void) and x.dtype.names:
        return {name: value(x[name]) for name in x.dtype.names}
    if isinstance(x, np.generic):
        return value(x.item())
    if isinstance(x, (list, tuple)):
        return [value(y) for y in x]
    if x is None or isinstance(x, (str, bool, int, float)):
        return x
    return {'unsupported_type': type(x).__name__}


def inventory(f, group, recording_paths, shared_paths, episode_groups):
    path = Path(f.filename).resolve()
    stat = path.stat()
    fields = []
    datasets = {p: 'recording' for p in recording_paths}
    datasets.update({p: 'file' for p in shared_paths})
    for name, scope in sorted(datasets.items()):
        ds = f[name]
        fields.append({'name': f'HDF5 {scope} {name}', 'object': name, 'scope': scope,
                       'shape': list(ds.shape) if ds.shape is not None else None, 'dtype': str(ds.dtype)})

    def attributes(name, obj):
        local = not group or name == group or name.startswith(group + '/')
        other_episode = any(e and (name == e or name.startswith(e + '/')) for e in episode_groups)
        if not local and other_episode:
            return
        scope = 'recording' if local else 'file'
        for key in obj.attrs:
            raw = obj.attrs.get_id(key)
            fields.append({'name': f'HDF5 {scope} {name or "/"} attribute {key}',
                           'object': name, 'attribute': key, 'scope': scope,
                           'shape': list(raw.shape) if raw.shape is not None else None, 'dtype': str(raw.dtype)})

    attributes('', f)
    f.visititems(attributes)
    return {'source_path': str(path), 'source_size': stat.st_size, 'source_mtime_ns': stat.st_mtime_ns,
            'source_required_for_inspection': True, 'fields': fields}


def check_source(source):
    path = Path(source['source_path'])
    try:
        stat = path.stat()
    except OSError as error:
        raise ValueError('Original HDF5 metadata source is unavailable') from error
    if stat.st_size != source['source_size'] or stat.st_mtime_ns != source['source_mtime_ns']:
        raise ValueError('Original HDF5 metadata source changed after preparation')
    return path


def read(source, field, pointer):
    import h5py
    path = check_source(source)
    tokens = [t.replace('~1', '/').replace('~0', '~') for t in pointer.split('/')[1:]]
    with h5py.File(path, 'r') as f:
        obj = f[field['object'] or '/']
        if Path(obj.file.filename).resolve() != path.resolve():
            raise ValueError('External HDF5 metadata requires its own source inventory')
        if 'attribute' in field:
            info = obj.attrs.get_id(field['attribute'])
            if info.shape is None:
                raise ValueError('Original HDF5 attribute has a null dataspace and contains no values')
            if np.prod(info.shape, dtype=object) * info.dtype.itemsize > 256 * 1024 * 1024:
                raise ValueError('Original HDF5 attribute exceeds the metadata read budget')
            selected = obj.attrs[field['attribute']]
        else:
            if obj.shape is None:
                raise ValueError('Original HDF5 field has a null dataspace and contains no values')
            if obj.is_virtual or obj.external:
                raise ValueError('External HDF5 storage requires its own source inventory')
            indices = []
            while tokens and len(indices) < obj.ndim and tokens[0].isdecimal():
                index = int(tokens.pop(0))
                if index >= obj.shape[len(indices)]:
                    raise ValueError('metadata pointer is not present')
                indices.append(index)
            if np.prod(obj.shape[len(indices):], dtype=object) > MAX_VALUES:
                raise ValueError('Metadata selection exceeds 4096 values; request a deeper JSON pointer')
            selected = obj[tuple(indices) if indices else ()]
        try:
            for token in tokens:
                if isinstance(selected, (dict, np.void)):
                    selected = selected[token]
                else:
                    if not token.isdecimal():
                        raise ValueError('metadata array pointer requires a nonnegative index')
                    selected = selected[int(token)]
        except (IndexError, KeyError, TypeError, ValueError) as error:
            raise ValueError('metadata pointer is not present') from error
        if isinstance(selected, np.ndarray) and selected.size > MAX_VALUES:
            raise ValueError('Metadata selection exceeds 4096 values; request a deeper JSON pointer')
        return value(selected)
