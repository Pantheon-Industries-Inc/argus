"""A file is the one recorded about it when it holds the same bytes (label/content_identity.py): a copy with new times
passes, an edit that keeps the size and puts the modification time back does not, and the stat only spares a read."""
from __future__ import annotations

import hashlib
import os
import shutil

import pytest

from label import content_identity as ci


def _copy_with_new_times(path):
    copy = path.with_name(path.name + '.copy')
    shutil.copyfile(path, copy)
    os.replace(copy, path)
    os.utime(path, ns=(path.stat().st_mtime_ns + 10**10,) * 2)


def _same_size_and_time(path, data):
    seen = path.stat()
    path.write_bytes(data)
    os.utime(path, ns=(seen.st_atime_ns, seen.st_mtime_ns))
    assert (path.stat().st_size, path.stat().st_mtime_ns) == (seen.st_size, seen.st_mtime_ns)


def test_the_stat_only_spares_a_read_and_content_decides(tmp_path, monkeypatch):
    monkeypatch.setenv('BOARD_SOURCE_DIGEST_CACHE', str(tmp_path / 'digests.sqlite3'))
    monkeypatch.setattr(ci, 'durable_metadata', lambda *args: True)
    ci.digest.cache_clear()
    reads = []
    real = ci._read
    monkeypatch.setattr(ci, '_read', lambda *args: reads.append(args[0]) or real(*args))
    f = tmp_path / 'take.mp4'
    f.write_bytes(b'abcd')
    first = ci.identity(f)
    assert first['sha256'] == hashlib.sha256(b'abcd').hexdigest() and len(reads) == 1
    assert ci.same_file(first, f) and len(reads) == 1          # an unchanged stat reuses the digest
    _copy_with_new_times(f)
    assert ci.same_file(first, f) and len(reads) == 2          # a new stat reads the bytes, which are the same
    _same_size_and_time(f, b'abce')
    assert not ci.same_file(first, f) and len(reads) == 3      # the change time moved, so the edit is read and seen


def test_untrusted_filesystems_always_read_the_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(ci, 'durable_metadata', lambda *args: False)
    f = tmp_path / 'take.mp4'
    f.write_bytes(b'abcd')
    first = ci.identity(f)
    _same_size_and_time(f, b'abce')
    assert not ci.same_file(first, f)


def test_records_compare_by_content_and_older_records_as_they_were_written():
    new = {'name': 'a', 'size': 4, 'sha256': 'x' * 64, 'mtime_ns': 1}
    assert ci.same_recorded(new, {**new, 'mtime_ns': 2})
    assert not ci.same_recorded(new, {**new, 'sha256': 'y' * 64})
    assert not ci.same_recorded(new, {**new, 'name': 'b'})
    legacy = {'name': 'a', 'size': 4, 'mtime_ns': 1}
    assert ci.same_recorded(legacy, new) and not ci.same_recorded(legacy, {**new, 'mtime_ns': 2})
    assert ci.content(new) == {'name': 'a', 'size': 4, 'sha256': 'x' * 64}


def test_an_hdf5_source_is_checked_by_content(tmp_path):
    from prepare import hdf_metadata
    f = tmp_path / 'recording.h5'
    f.write_bytes(b'hdf5 bytes')
    recorded = ci.identity(f)
    source = {'source_path': str(f), 'source_size': recorded['size'], 'source_sha256': recorded['sha256'],
              'source_mtime_ns': recorded['mtime_ns']}
    _copy_with_new_times(f)
    assert hdf_metadata.check_source(source) == f
    _same_size_and_time(f, b'hdf5 bytez')
    with pytest.raises(ValueError, match='changed'):
        hdf_metadata.check_source(source)
    # a source prepared before content identity is held to its size and modification time
    legacy = {'source_path': str(f), 'source_size': f.stat().st_size, 'source_mtime_ns': f.stat().st_mtime_ns}
    assert hdf_metadata.check_source(legacy) == f
    os.utime(f, ns=(f.stat().st_mtime_ns + 1,) * 2)
    with pytest.raises(ValueError, match='changed'):
        hdf_metadata.check_source(legacy)
