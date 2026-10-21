"""Bounded reads of original HDF5 metadata, without duplicating source arrays."""
from pathlib import Path
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np

MAX_VALUES = 4096
_SOURCES = ContextVar('hdf_sources', default=None)



def open_hdf5(path):
    """Open an uploaded HDF5 file for reading. What its links and storage may reach is decided per object, as the
    walker reaches it (children, storage_admitted), so an object naming storage outside the upload costs that
    object and never the file's cameras."""
    import h5py
    return h5py.File(path, 'r')


ATTRIBUTE_BYTES = 1 << 20    # an attribute holding more than this is named, not copied into the episode


def attribute_values(obj, keys=None, issues=None):
    """[(name, value)] of obj's attributes (only those named in keys, in that order, when keys is given), each read
    on its own. A damaged attribute is recorded on issues and skipped, and one holding more than ATTRIBUTE_BYTES is
    recorded and skipped unread where its size is known, so one attribute never costs the others or the episode.
    Iterating attrs.items() read every value before any was chosen, so one bad attribute failed the whole read."""
    from prepare.isolation import cameras_only
    if cameras_only():
        from prepare.formats import LEFT_UNREAD
        if "the HDF5 attributes" not in LEFT_UNREAD:
            LEFT_UNREAD.append("the HDF5 attributes")    # named by the retry's issue (prepare/formats.py read_episode)
        return []           # a second attempt after a read that stopped responding reads no attribute (isolation)
    where = getattr(obj, 'name', '') or '/'

    def note(what, error=None):
        if issues is not None:
            issues.append({'kind': 'attribute_unreadable', 'what': f'HDF5 attribute {what} was not read'
                           + (f' ({type(error).__name__})' if error is not None else '') + '.'})
    try:
        names = list(obj.attrs.keys())
    except Exception as error:  # noqa: BLE001 - h5py on upload bytes raises many types; recorded by note
        note(f'list of {where}', error)
        return []
    if keys is not None:
        present = set(names)
        names = [key for key in keys if key in present]
    out = []
    for name in names:
        try:
            size = obj.attrs.get_id(name).get_storage_size()
            if size > ATTRIBUTE_BYTES:
                note(f'{name} of {where}, {size:,} bytes,')
                continue
            value = obj.attrs[name]
        except Exception as error:  # noqa: BLE001 - h5py on upload bytes raises many types; recorded by note
            note(f'{name} of {where}', error)
            continue
        held = value.nbytes if isinstance(value, np.ndarray) and value.dtype.kind != 'O' else \
            sum(len(x) for x in np.asarray(value, dtype=object).ravel() if isinstance(x, (str, bytes))) \
            if isinstance(value, np.ndarray) else len(value) if isinstance(value, (str, bytes)) else 0
        if held > ATTRIBUTE_BYTES:
            note(f'{name} of {where}, {held:,} bytes,')
            continue
        out.append((name, value))
    return out


def current_sources():
    """The files authorized_sources admits here, or None outside it."""
    return _SOURCES.get()


@contextmanager
def authorized_sources(paths):
    """Only files selected by the upload may supply linked HDF objects or storage."""
    token = _SOURCES.set(frozenset(Path(p).resolve() for p in paths))
    try:
        yield
    finally:
        _SOURCES.reset(token)


def object_id(obj):
    import h5py
    return (str(Path(obj.file.filename).resolve()), h5py.h5o.get_info(obj.id).addr)


def raw_storage_admitted(path, external, allowed):
    """Whether every file a dataset's external raw storage names is an upload file. HDF5 resolves such a name
    against the working directory, or HDF5_EXTFILE_PREFIX when set, and an absolute name as it stands, never
    against the HDF5 file's folder. A name is admitted only when it lands on an upload file both ways, so neither
    a decoy beside the file nor a host file of the same name can supply values."""
    import os
    if os.environ.get('HDF5_EXTFILE_PREFIX'):
        return False
    for name, _, _ in external:
        name = Path(os.fsdecode(name))
        places = [name] if name.is_absolute() else [Path.cwd() / name, path.parent / name]
        if any(place.resolve() not in allowed for place in places):
            return False
    return True


def storage_admitted(obj, allowed, seen=None):
    """Resolve virtual source datasets too, since their paths can hide external links."""
    import h5py
    path = Path(obj.file.filename).resolve()
    if path not in allowed:
        return False
    if not isinstance(obj, h5py.Dataset):
        return True
    if obj.external and not raw_storage_admitted(path, obj.external, allowed):
        return False
    if not obj.is_virtual:
        return True
    seen = set() if seen is None else seen
    identity = object_id(obj)
    if identity in seen:
        return False
    seen.add(identity)
    try:
        for source in obj.virtual_sources():
            filename = path if source.file_name == '.' else (path.parent / source.file_name).resolve()
            if filename not in allowed:
                return False
            with h5py.File(filename, 'r') as linked:
                if not storage_admitted(linked[source.dset_name], allowed, seen):
                    return False
        return True
    finally:
        seen.remove(identity)


def children(group, issues=None):
    import h5py
    allowed = _SOURCES.get()
    if allowed is None:
        allowed = {Path(group.file.filename).resolve()}

    def admitted(path):
        return Path(path).resolve() in allowed

    def excluded(name):
        if issues is not None:
            issue = {'kind': 'external_source_unread',
                     'what': f'HDF5 object {name} references storage outside the selected upload; '
                             'its external values were not read.'}
            if issue not in issues:
                issues.append(issue)

    for key in group:
        name = group.name.rstrip('/') + '/' + key
        try:
            link = group.get(key, getlink=True)
            if isinstance(link, h5py.ExternalLink):
                target = Path(group.file.filename).resolve().parent / link.filename
                if not admitted(target):
                    excluded(name)
                    continue
            obj = group[key]
            if not storage_admitted(obj, allowed):
                excluded(name)
                continue
            yield key, obj
        except (KeyError, OSError):
            continue


def objects(group, prefix='', issues=None):
    """Walk each owned group once, including legal hard-link and soft-link cycles."""
    import h5py
    seen = {object_id(group)}
    pending = [(prefix, iter(children(group, issues)))]
    while pending:
        base, entries = pending[-1]
        try:
            key, obj = next(entries)
        except StopIteration:
            pending.pop()
            continue
        name = f'{base}/{key}' if base else key
        yield name, obj
        if isinstance(obj, h5py.Group) and object_id(obj) not in seen:
            seen.add(object_id(obj))
            pending.append((name, iter(children(obj, issues))))


class NumericRows:
    """Read healthy numeric rows when an individual HDF chunk fails its checksum."""
    def __init__(self, dataset, issues):
        self.dataset, self.issues = dataset, issues
        self.shape = dataset.shape

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        try:
            return self.dataset[key]
        except OSError:
            rows = range(len(self))[key if isinstance(key, slice) else slice(None)]
            result = np.full((len(rows), *self.shape[1:]), np.nan)
            missing = []
            for destination, row in enumerate(rows):
                try:
                    result[destination] = self.dataset[row]
                except OSError:
                    missing.append(row)
            if missing:
                self.issues.append({'kind': 'signal_bad_cells', 'signal': self.dataset.name,
                                    'what': f'{self.dataset.name}: {len(missing)} numeric rows could not '
                                            'be read from damaged HDF5 chunks and remain missing.'})
            return result


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
    # h5o.visit can traverse a damaged dataset B-tree even when its useful peers and attributes are readable.
    for name, obj in objects(f):
        attributes(name, obj)
    from label import content_identity
    recorded = content_identity.identity(path, read_once=True)
    return {'source_path': str(path), 'source_size': recorded['size'], 'source_sha256': recorded['sha256'],
            'source_mtime_ns': recorded['mtime_ns'], 'source_required_for_inspection': True, 'fields': fields}


def check_source(source):
    """The original file, while it holds the bytes it was prepared from (by content, label/content_identity.py; a source
    prepared before content identity is checked by its size and modification time)."""
    from label import content_identity
    path = Path(source['source_path'])
    if not path.is_file():
        raise ValueError('Original HDF5 metadata source is unavailable')
    if not content_identity.same_file(source, path, size='source_size', sha256='source_sha256',
                                      mtime_ns='source_mtime_ns'):
        raise ValueError('Original HDF5 metadata source changed after preparation')
    return path


def read(source, field, pointer):
    path = check_source(source)
    tokens = [t.replace('~1', '/').replace('~0', '~') for t in pointer.split('/')[1:]]
    with open_hdf5(path) as f:
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
