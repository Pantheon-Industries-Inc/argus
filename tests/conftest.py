import os

import pytest

from sharding import assign, parse


@pytest.fixture(autouse=True)
def _uploads_read_in_this_process(monkeypatch):
    """Uploads are read in a child process (prepare.isolation), which never sees a function or limit a test replaced
    in this one, so tests read them here. tests/test_reader_isolation.py tests the child process itself, and the
    corrupted-input and timing harnesses run the command line, child process included."""
    monkeypatch.setenv('ARGUS_READER_ISOLATION', '0')


def pytest_collection_modifyitems(config, items):
    """With ARGUS_TEST_SHARD=i/n, keep only the test files of shard i of n (tests/sharding.py). Without it, run all."""
    value = os.environ.get('ARGUS_TEST_SHARD')
    if not value:
        return
    index, count = parse(value)
    root = config.rootpath
    path = {item: item.path.relative_to(root).as_posix() for item in items}
    shard = assign(path.values(), count)
    keep = [item for item in items if shard[path[item]] == index]
    dropped = [item for item in items if shard[path[item]] != index]
    if dropped:
        config.hook.pytest_deselected(items=dropped)
    items[:] = keep
