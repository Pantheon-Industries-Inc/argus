"""Split the suite into shards that CI runs side by side (ARGUS_TEST_SHARD=1/4 ... 4/4).

A shard is a set of whole test files, so every file runs complete and in its usual order, as it does in one run.
Files are placed largest first, each on the shard with the least time so far, by the seconds recorded in
tests/shard_times.json. A file missing from it (a new one) gets the median time, so it is still run, by exactly one
shard. Every shard computes the same split from the same files, so together they run each file once."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

TIMES = Path(__file__).with_name('shard_times.json')


def parse(value: str) -> tuple[int, int]:
    """(index from 0, count) of "i/n", where 1 <= i <= n."""
    index, _, count = value.partition('/')
    i, n = int(index), int(count)
    if not 1 <= i <= n:
        raise ValueError(f'ARGUS_TEST_SHARD must be i/n with 1 <= i <= n, not {value!r}')
    return i - 1, n


def assign(files, count: int, times: dict[str, float] | None = None) -> dict[str, int]:
    """{file: shard index} for every file, balancing the recorded seconds across count shards."""
    times = json.loads(TIMES.read_text()) if times is None else times
    default = statistics.median(times.values()) if times else 1.0
    loads = [0.0] * count
    shard = {}
    for f in sorted(set(files), key=lambda f: (-times.get(f, default), f)):
        i = min(range(count), key=lambda k: (loads[k], k))
        shard[f] = i
        loads[i] += times.get(f, default)
    return shard
