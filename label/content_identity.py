"""Whether a file still holds what was recorded about it, decided by its bytes and never by its timestamps.

A copy, an untar, a re-download or a sync gives a file new modification and change times and keeps its bytes, and an
edit can keep a file's size and put its modification time back. So every identity, resume and cache decision compares
the SHA-256 of the content (digest). The stat of a file (device, inode, size, modification and change times) is only a
fast path: while it is unchanged on a filesystem that keeps change times (durable_metadata), the digest read for that
stat before is used again, and once any part of it differs the content is read again. The same bytes then pass under
their new stat, and different bytes do not.

An identity written before content identity existed holds a size and a modification time and no digest. It is checked
the way it was written (same_file, same_recorded), since nothing else is known about the bytes it stood for.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import sys
from functools import lru_cache
from pathlib import Path

CHUNK = 1024 * 1024


def stat_version(path) -> tuple:
    """(device, inode, size, modification ns, change ns) of the file at path: the stat a digest is kept for."""
    st = os.stat(path)
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def durable_metadata(path: str, version: tuple) -> bool:
    """Only reuse durable hashes on local filesystems with change-time semantics."""
    if os.name != 'posix' or version[-1] % 1_000_000_000 == 0:
        return False
    import ctypes
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        buffer = ctypes.create_string_buffer(4096)
        if sys.platform == 'darwin':
            # statfs is the legacy ABI on Intel. statfs64 has the fixed layout
            # on both supported Mac architectures, including f_fstypename at 72.
            statfs = libc.statfs64
            statfs.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
            statfs.restype = ctypes.c_int
            if statfs(os.fsencode(path), ctypes.byref(buffer)):
                return False
            return buffer.raw[72:88].split(b'\0', 1)[0] in (b'apfs', b'hfs')
        if libc.statfs(os.fsencode(path), ctypes.byref(buffer)):
            return False
        if sys.platform.startswith('linux'):
            kind = ctypes.c_ulong.from_buffer(buffer).value
            return kind in {0xEF53, 0x58465342, 0x9123683E, 0x2FC12FC1, 0x01021994, 0x794C7630}
    except (AttributeError, OSError):
        pass
    return False


def _cache_path() -> Path:
    return Path(os.environ.get('BOARD_SOURCE_DIGEST_CACHE') or
                Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'argus/source-digests.sqlite3')


def _read(path: str, version: tuple, restat) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(CHUNK), b''):
            h.update(chunk)
    if tuple(restat(path)) != tuple(version):
        raise ValueError('Media changed while its identity was being read')
    return h.hexdigest()


@lru_cache(maxsize=8192)
def _durable_digest(path: str, version: tuple, restat=stat_version) -> str:
    # ctime also changes on in-place edits whose size and restored mtime match.
    cache, key = _cache_path(), json.dumps([path, version])
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(cache, timeout=1) as db:
            db.execute('CREATE TABLE IF NOT EXISTS digests (path TEXT PRIMARY KEY, version TEXT, digest TEXT)')
            row = db.execute('SELECT digest FROM digests WHERE path=? AND version=?', (path, key)).fetchone()
        if row and re.fullmatch(r'[0-9a-f]{64}', row[0]) and tuple(restat(path)) == tuple(version):
            return row[0]
    except (OSError, sqlite3.Error):
        pass  # A missing, busy, or damaged cache costs a hash, never an identity.
    digest_ = _read(path, version, restat)
    try:
        with sqlite3.connect(cache, timeout=1) as db:
            db.execute('INSERT OR REPLACE INTO digests VALUES (?, ?, ?)', (path, key, digest_))
    except (OSError, sqlite3.Error):
        pass
    return digest_


def digest(path, version: tuple | None = None, *, durable=None, restat=None) -> str:
    """SHA-256 of the file's content. A digest read before is used again only while the file keeps the whole stat it
    was read at, on a filesystem whose change times can be trusted (durable_metadata); anything else reads the bytes.
    A file that changes while it is read raises ValueError. durable and restat replace the policy and the stat reader
    (board/static.py passes its own, which its tests replace)."""
    path = str(Path(path).resolve())
    restat = restat or stat_version
    version = tuple(version) if version is not None else tuple(restat(path))
    if (durable or durable_metadata)(path, version):
        return _durable_digest(path, version, restat)
    return _read(path, version, restat)


digest.cache_clear = _durable_digest.cache_clear


_READ_ONCE: dict = {}


def identity(path, *, read_once=False) -> dict:
    """What is recorded about a file's content: its size and digest, plus its modification time, which only lets a
    record written before content identity be compared with it (same_recorded). read_once keeps each stat's digest for
    the rest of the process on every filesystem, for a reader that records one input file many times while nothing
    writes it (an HDF5 file that holds many episodes)."""
    for _ in range(3):
        version = stat_version(path)
        key = (str(Path(path).resolve()), version)
        try:
            sha = (_READ_ONCE.get(key) if read_once else None) or digest(path, version)
        except ValueError:
            continue       # written while it was read: read the version it settled on
        if read_once:
            _READ_ONCE[key] = sha
        return {'size': version[2], 'sha256': sha, 'mtime_ns': version[3]}
    raise OSError(f'{path} kept changing while its content was read')


def content(entry: dict) -> dict:
    """An identity without the times that only bridge to older records, for keys built from identities."""
    return {k: v for k, v in entry.items() if k != 'mtime_ns'} if 'sha256' in entry else dict(entry)


def same_recorded(saved: dict, current: dict) -> bool:
    """Whether two identities stand for the same file content: by digest when both have one (their times are ignored,
    so a byte-identical copy passes), and as an older record was written, by size and modification time, when either
    has none. Every other recorded field must be equal."""
    if not isinstance(saved, dict) or not isinstance(current, dict):
        return False
    if 'sha256' in saved and 'sha256' in current:
        return content(saved) == content(current)
    def written(entry):
        return {k: v for k, v in entry.items() if k != 'sha256'}
    return written(saved) == written(current)


def same_file(saved: dict, path, *, size='size', sha256='sha256', mtime_ns='mtime_ns') -> bool:
    """Whether the file at path still holds the content a saved identity recorded (the key names say where a record
    keeps each fact). A record with a digest is compared by size and digest, and one without by size and modification
    time, as it was written. A missing file is not the same."""
    try:
        version = stat_version(path)
        if not isinstance(saved, dict) or saved.get(size) != version[2]:
            return False
        if saved.get(sha256):
            return digest(path, version) == saved[sha256]
        return mtime_ns in saved and saved[mtime_ns] == version[3]
    except (OSError, ValueError):
        return False
