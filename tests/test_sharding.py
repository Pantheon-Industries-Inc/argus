"""The CI shards together run every test file exactly once (tests/sharding.py)."""
from pathlib import Path

import pytest

from sharding import TIMES, assign, parse


def test_every_file_lands_in_exactly_one_shard_whatever_the_count():
    files = sorted(p.as_posix() for p in Path('tests').glob('test_*.py'))
    for count in (1, 2, 3, 4, 7):
        shard = assign(files, count)
        assert sorted(shard) == files and set(shard.values()) <= set(range(count))


def test_a_new_file_without_a_recorded_time_is_still_run():
    files = ['tests/test_known.py', 'tests/test_new_one.py']
    shard = assign(files, 4, {'tests/test_known.py': 10.0})
    assert set(shard) == set(files)


def test_the_recorded_times_name_only_files_that_exist_and_split_evenly():
    import json
    times = json.loads(TIMES.read_text())
    assert all(Path(f).exists() for f in times), [f for f in times if not Path(f).exists()]
    loads = [0.0] * 4
    for f, i in assign(times, 4).items():
        loads[i] += times[f]
    assert max(loads) <= 1.2 * sum(loads) / 4, loads


@pytest.mark.parametrize('value', ['0/4', '5/4', '1/0', 'x/4', '2'])
def test_a_shard_outside_its_count_is_refused(value):
    with pytest.raises(ValueError):
        parse(value)
