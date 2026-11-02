import os

import pytest

from sharding import assign, parse


@pytest.fixture(autouse=True)
def _uploads_read_in_this_process(monkeypatch):
    """Uploads are read in a child process (prepare.isolation), which never sees a function or limit a test replaced
    in this one, so tests read them here. tests/test_reader_isolation.py tests the child process itself, and the
    corrupted-input and timing harnesses run the command line, child process included."""
    monkeypatch.setenv('ARGUS_READER_ISOLATION', '0')


@pytest.fixture(autouse=True)
def _automatic_resends_without_waiting(monkeypatch):
    """A unit the provider failed is sent again after a backoff (label/harness.py AUTO_RESEND_DELAYS_S); tests send it
    again at once."""
    from label import harness
    monkeypatch.setattr(harness, 'AUTO_RESEND_DELAYS_S', (0.0,) * len(harness.AUTO_RESEND_DELAYS_S))


@pytest.fixture
def resends_used_up(monkeypatch):
    """A unit past its automatic resends (label/harness.py AUTO_RESENDS): what Argus does once none is left, which is
    to keep every hold and send nothing more without --retry-received or --force."""
    from label import harness
    monkeypatch.setattr(harness, 'AUTO_RESENDS', 0)



@pytest.fixture
def routing_on(monkeypatch):
    """Resolution routing turned on (label/route.py enabled), for the tests of routing itself: it is off by default."""
    monkeypatch.setenv('ARGUS_RESOLUTION_ROUTE', 'on')


_SHARD = {}


def _shard_of(config) -> tuple[int, dict] | None:
    """(this shard's index, {test file: shard index}) under ARGUS_TEST_SHARD=i/n, else None. Every test file under
    tests/ is placed, so each shard computes the same split from the same files (tests/sharding.py)."""
    value = os.environ.get('ARGUS_TEST_SHARD')
    if not value:
        return None
    if value not in _SHARD:
        index, count = parse(value)
        root = config.rootpath
        files = [f.relative_to(root).as_posix() for f in sorted((root / 'tests').rglob('test_*.py'))]
        _SHARD[value] = (index, assign(files, count))
    return _SHARD[value]


def pytest_ignore_collect(collection_path, config):
    """With ARGUS_TEST_SHARD=i/n, skip the test files of the other shards before they are imported. Without it, run
    all."""
    shard = _shard_of(config)
    if shard is None or not (collection_path.name.startswith('test_') and collection_path.suffix == '.py'):
        return None
    index, placed = shard
    try:
        rel = collection_path.relative_to(config.rootpath).as_posix()
    except ValueError:
        return None
    return True if placed.get(rel, index) != index else None


def pytest_collection_modifyitems(config, items):
    """Files named on the command line are collected whatever the shard (pytest does not ask pytest_ignore_collect
    about them), so their tests are deselected here when they belong to another shard."""
    shard = _shard_of(config)
    if shard is None:
        return
    index, placed = shard
    root = config.rootpath
    keep, dropped = [], []
    for item in items:
        (keep if placed.get(item.path.relative_to(root).as_posix(), index) == index else dropped).append(item)
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = keep


@pytest.fixture
def inspect_every_field(monkeypatch):
    """Evidence inspection of every recorded field beyond the cameras (policy inspection all), which a test of the
    inspection machinery on arm signals, commands or metadata exercises; by default only depth and touch sensors are
    inspected (label/policy.py INSPECTION)."""
    from label import policy
    monkeypatch.setattr(policy, 'INSPECTION', 'all')
