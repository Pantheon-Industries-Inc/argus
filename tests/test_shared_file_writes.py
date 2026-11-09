"""Files several processes write, and per-process memos of files another process can replace.

Several writer processes replace one JSON file through write_atomic while a reader process reads it; every read is
one whole file one writer wrote. A memo of what was read from a file is never served for a version it did not read."""
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import textwrap
import time

from label.atomic import cached_while_unchanged
from prepare import display

REPO = Path(__file__).resolve().parents[1]

WRITE = textwrap.dedent('''
    import json, sys, time
    sys.path.insert(0, sys.argv[1])
    from label.atomic import write_atomic
    path, me, until = sys.argv[2], sys.argv[3], float(sys.argv[4])
    n = 0
    while time.time() < until:
        n += 1
        # a large body of one writer's own letter, so a torn or mixed file cannot pass for a whole one
        write_atomic(path, {'writer': me, 'n': n, 'body': me * (50_000 + n % 7)}, indent=None)
    print(n)
''')

READ = textwrap.dedent('''
    import json, sys, time
    path, until = sys.argv[1], float(sys.argv[2])
    reads = bad = 0
    while time.time() < until:
        try:
            with open(path) as source:
                record = json.load(source)
        except FileNotFoundError:
            continue
        except ValueError:
            bad += 1
            continue
        reads += 1
        if record['body'] != record['writer'] * (50_000 + record['n'] % 7):
            bad += 1
    print(reads, bad)
''')


def test_several_processes_writing_one_file_never_publish_a_torn_or_foreign_file(tmp_path):
    path, until = tmp_path / 'shared.json', time.time() + 2
    writers = [subprocess.Popen([sys.executable, '-c', WRITE, str(REPO), str(path), letter, str(until)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for letter in 'abcd']
    reader = subprocess.Popen([sys.executable, '-c', READ, str(path), str(until)], stdout=subprocess.PIPE, text=True)
    results = [writer.communicate(timeout=60) for writer in writers]
    assert all(writer.returncode == 0 for writer in writers), [err[-500:] for _, err in results]
    assert sum(int(out) for out, _ in results) > 40
    reads, bad = map(int, reader.communicate(timeout=60)[0].split())
    assert reads > 10 and bad == 0
    assert sorted(os.listdir(tmp_path)) == ['shared.json']


def test_a_memo_keeps_a_result_only_for_the_version_it_read(tmp_path):
    path, calls = tmp_path / 'value.txt', []
    path.write_text('one')
    version = lambda: (path.stat().st_size, path.stat().st_mtime_ns)

    @cached_while_unchanged(8, lambda p, v: (Path(p).stat().st_size, Path(p).stat().st_mtime_ns) == v)
    def read(p, v):
        calls.append(v)
        return Path(p).read_text()
    first = version()
    assert read(str(path), first) == 'one' and read(str(path), first) == 'one'
    assert len(calls) == 1
    time.sleep(0.05)  # a fast runner can write twice inside one kernel clock tick, leaving the stat unchanged
    path.write_text('two')
    assert read(str(path), version()) == 'two'


def test_geometry_never_serves_a_probe_of_a_version_it_did_not_read(tmp_path):
    """A file is replaced between the stat that names its version and the probe that reads it; the probe's answer
    (for the replacement) must not be kept for the version the stat named, which the file can have again."""
    video, marker = tmp_path / 'clip.mp4', tmp_path / 'replace-before-reading'
    video.write_text(json.dumps({'width': 10, 'height': 10}))
    probe = tmp_path / 'ffprobe'
    probe.write_text(f'#!{sys.executable}\n' + textwrap.dedent(f'''
        import json, os, sys
        target, marker = sys.argv[-1], {str(marker)!r}
        if os.path.exists(marker):              # another process replaces the file just before this read
            os.remove(marker)
            with open(target + '.new', 'w') as out:
                out.write(json.dumps({{'width': 2000, 'height': 2000}}))
            os.replace(target + '.new', target)
        print(json.dumps({{'streams': [json.load(open(target))]}}))
    '''))
    probe.chmod(probe.stat().st_mode | stat.S_IEXEC)
    original = video.read_text()
    seen = video.stat()
    marker.write_text('')
    display._geometry.cache_clear()
    assert display.geometry(str(video), str(probe))['stored'] == (2000, 2000)    # what this read actually found
    # the original comes back with the size and modification time the first stat saw
    video.write_text(original)
    os.utime(video, ns=(seen.st_atime_ns, seen.st_mtime_ns))
    assert display.geometry(str(video), str(probe))['stored'] == (10, 10)
