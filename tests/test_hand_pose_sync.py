"""Pulling the hand pose outputs from the Modal volume copies a file by its content (board/hand_pose/sync.py): a remote
file rewritten under the same size is copied, and a local file that holds the remote bytes is left alone."""
from __future__ import annotations

import hashlib
import json

import pytest

from board.hand_pose import sync


class Volume:
    """The output volume as pull sees it: a listing of (path, size) and a reader that yields a file's bytes."""
    def __init__(self, files):
        self.files = files
        self.reads = []

    def entries(self):
        return [(f'/{path}', len(data)) for path, data in self.files.items()]

    def read_file(self, path):
        self.reads.append(path.lstrip('/'))
        data = self.files[path.lstrip('/')]
        yield data[:3]
        yield data[3:]


def _volume(outputs, with_digests):
    files = {f'ds/ep/{name}': data for name, data in outputs.items()}
    if with_digests:
        files['ds/ep/digests.json'] = json.dumps(
            {name: hashlib.sha256(data).hexdigest() for name, data in outputs.items()}).encode()
    return Volume(files)


@pytest.mark.parametrize('with_digests', [True, False])
def test_a_same_size_file_with_other_bytes_is_copied_and_an_identical_one_is_kept(tmp_path, with_digests):
    outputs = {'hands2d.json': b'{"hands": 1}', 'raw.npz': b'raw bytes', 'run.json': b'{"run": 1}'}
    local = tmp_path / 'ds' / 'ep'
    local.mkdir(parents=True)
    (local / 'hands2d.json').write_bytes(b'{"hands": 0}')        # same size, other bytes: a rerun on the volume
    (local / 'raw.npz').write_bytes(outputs['raw.npz'])           # already the remote bytes
    volume = _volume(outputs, with_digests)
    done = sync.pull_outputs(tmp_path, volume.entries(), volume.read_file)
    assert sorted(done['copied']) == ['ds/ep/hands2d.json', 'ds/ep/run.json']
    assert done['kept'] == ['ds/ep/raw.npz']
    assert {p.name: p.read_bytes() for p in local.iterdir()} == outputs
    assert json.loads((tmp_path / 'index.json').read_text()) == {
        'ds/ep': {'hands2d': 'ds/ep/hands2d.json', 'raw': 'ds/ep/raw.npz'}}
    if with_digests:
        # the remote digest spares reading a file the machine already holds
        assert 'ds/ep/raw.npz' not in volume.reads
    # a second pull finds every file already holding the remote bytes and leaves them as they are
    times = {p.name: p.stat().st_mtime_ns for p in local.iterdir()}
    again = sync.pull_outputs(tmp_path, volume.entries(), volume.read_file)
    assert again['copied'] == [] and len(again['kept']) == 3
    assert {p.name: p.stat().st_mtime_ns for p in local.iterdir()} == times


def test_a_download_that_does_not_match_its_remote_digest_is_refused(tmp_path):
    volume = _volume({'hands2d.json': b'{"hands": 1}'}, True)
    volume.files['ds/ep/hands2d.json'] = b'{"hands": 2}'          # changed after its digest was written
    with pytest.raises(ValueError, match='digest'):
        sync.pull_outputs(tmp_path, volume.entries(), volume.read_file)
    assert not (tmp_path / 'ds' / 'ep' / 'hands2d.json').exists()
    assert [p.name for p in (tmp_path / 'ds' / 'ep').iterdir()] == []


def test_the_remote_digests_cover_every_output(tmp_path):
    for name, data in (('hands2d.json', b'a'), ('raw.npz', b'bb'), ('run.json', b'ccc'), ('other.txt', b'x')):
        (tmp_path / name).write_bytes(data)
    assert sync.digests(tmp_path) == {name: hashlib.sha256(data).hexdigest()
                                      for name, data in (('hands2d.json', b'a'), ('raw.npz', b'bb'), ('run.json', b'ccc'))}
