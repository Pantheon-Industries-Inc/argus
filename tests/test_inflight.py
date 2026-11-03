"""Every labelling entry point sends all of its requests at once by default, and --concurrency still caps them
(label/inflight.py). The old defaults kept 4 per key, 8 or 16 in flight, which made a 338-episode run of short clips
take an hour instead of 15 minutes."""
import ast
import json
import sys
import threading
from pathlib import Path

import pytest

from label import catalog, harness, inflight

REPO = Path(__file__).resolve().parent.parent
N = 24          # more than any old default (4 per key, 6, 8, 16)


class Peak:
    """Counts calls in flight. Each call waits until all `expect` calls are in flight at once (or a short timeout),
    so a pool smaller than `expect` shows up as a lower peak instead of a slow test."""

    def __init__(self, expect):
        self.expect, self.now, self.peak = expect, 0, 0
        self.lock, self.all_in = threading.Lock(), threading.Event()

    def __call__(self):
        with self.lock:
            self.now += 1
            self.peak = max(self.peak, self.now)
            if self.now >= self.expect:
                self.all_in.set()
        self.all_in.wait(2)
        with self.lock:
            self.now -= 1


def _episodes(tmp_path, n):
    eps = []
    for i in range(n):
        d = tmp_path / 'slice' / f'episode_{i:06d}'
        d.mkdir(parents=True)
        eps.append(d)
    return eps


def _harness_main(tmp_path, monkeypatch, expect, *flags):
    # the fixture episodes share one prompt variant; this measures the pool, not the cache warm-up (#169)
    monkeypatch.setenv('ARGUS_CACHE_WARMUP', '0')
    eps = _episodes(tmp_path, N)
    peak = Peak(expect)

    def fake_label(ep, out, *, api_key, **kw):
        peak()
        out.write_text(json.dumps({'parse_ok': True, 'usage': {'est_cost_usd': 0.0}}))
        return {'usage': {'est_cost_usd': 0.0}}

    monkeypatch.setattr(harness, 'label_episode', fake_label)
    monkeypatch.setattr(harness, 'discover_episodes', lambda root: eps)
    monkeypatch.setenv('OPENROUTER_API_KEYS', 'sk-or-a')
    monkeypatch.setattr(sys, 'argv', ['label.harness', '--episodes-root', str(tmp_path / 'slice'),
                                      '--out-dir', str(tmp_path / 'out'), *flags])
    assert harness.main() == 0
    assert len(list((tmp_path / 'out').glob('episode_*.json'))) == N
    return peak.peak


def test_harness_default_puts_every_episode_in_flight(tmp_path, monkeypatch):
    assert _harness_main(tmp_path, monkeypatch, N) == N


def test_harness_concurrency_still_caps(tmp_path, monkeypatch):
    assert _harness_main(tmp_path, monkeypatch, 3, '--concurrency', '3') == 3


def test_label_run_passes_no_cap_by_default(tmp_path, monkeypatch):
    from label import run
    _episodes(tmp_path, 1)
    sent = {}
    monkeypatch.setattr(run, 'commit', lambda: ('abc1234', False))
    monkeypatch.setattr(run, 'finish', lambda run_dir, info, cmd, slice_dir, mode: sent.setdefault('cmd', cmd) and 0)
    monkeypatch.setattr(sys, 'argv', ['label', '--dataset', 'd', '--episodes', str(tmp_path / 'slice'),
                                      '--kind', 'dry', '--runs', str(tmp_path / 'runs')])
    run.main()
    cmd = sent['cmd']
    assert cmd[cmd.index('--concurrency') + 1] == '0'


def test_catalog_sends_every_text_at_once():
    peak = Peak(N)

    class Asker(catalog.Asker):
        def __init__(self):
            pass

        def __call__(self, text, tokens, model=None):
            peak()
            return text

    texts = [str(i) for i in range(N)]
    assert Asker().many(texts, 100, 0) == texts and peak.peak == N
    peak = Peak(5)
    assert Asker().many(texts, 100, 5) == texts and peak.peak == 5


def test_no_entry_point_sets_its_own_default():
    """Every --concurrency comes from label/inflight.py add_argument, so no entry point brings back a small default.
    The old ones were add_argument("--concurrency", type=int, default=8) and the like in seven modules."""
    found = []
    for path in REPO.glob('*/**/*.py'):
        if 'tests' in path.parts or '.venv' in path.parts or path == REPO / 'label' / 'inflight.py':
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Call) and getattr(node.func, 'attr', None) == 'add_argument' and node.args
                    and isinstance(node.args[0], ast.Constant) and node.args[0].value == '--concurrency'):
                found.append(str(path.relative_to(REPO)))
    assert not found, found


@pytest.mark.parametrize('module', ['review.__main__', 'label.run', 'label.harness', 'label.catalog',
                                    'label.issue_groups', 'label.seam', 'compare.__main__', 'gate.__main__'])
def test_every_entry_point_uses_the_shared_flag(module):
    source = (REPO / (module.replace('.', '/') + '.py')).read_text()
    assert 'inflight.add_argument(' in source


def test_open_file_limit_bounds_the_pool(monkeypatch, capsys):
    monkeypatch.setattr(inflight, 'open_file_limit', lambda: 200)
    monkeypatch.setattr(inflight, 'open_files', lambda: 0)
    room = (200 - inflight.FILES_KEPT_FREE) // inflight.FILES_PER_REQUEST
    assert inflight.in_flight(0, 5000) == room
    assert 'open files' in capsys.readouterr().err
    assert inflight.in_flight(0, 10) == 10
    assert inflight.in_flight(4, 10) == 4
    assert inflight.in_flight(0, 0) == 1
