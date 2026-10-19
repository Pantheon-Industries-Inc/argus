"""Run one native read of an uploaded file in its own process.

A native reader (HDF5's C library, a video or decompression codec) can loop forever or crash the process on damaged
bytes, and no Python boundary can stop either: the call never returns, or the process is gone. So the read of one
episode, and the planning of an upload's HDF5 files, run in a child process. The parent stops a child that makes no
progress for READER_STALL_S, and a child that dies or is stopped costs that read alone, never the upload or the
server that runs it. The caller decides what to try next (read_episode reads the episode again with its cameras
alone, cameras_only).

    ctx, item = run_isolated(convert_item, item, rig, out, dataset, scope, watch=out)
"""
from __future__ import annotations

import importlib
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

UPLOAD = None                     # reads no dataset of its own (formats.upload_adapters)
READER_STALL_S = float(os.environ.get('ARGUS_READER_STALL_S') or 300)
_INLINE = 'ARGUS_READER_INLINE'           # set in a child, whose own reads run in place
_MODE = 'ARGUS_READER_MODE'
_HEARTBEAT = 'ARGUS_READER_HEARTBEAT'
# sh -c _GUARD guard PARENT CHILD: wait while both live, then make sure the child is gone
_GUARD = ('while kill -0 "$1" 2>/dev/null && kill -0 "$2" 2>/dev/null; do sleep 1; done; '
          'kill -9 "$2" 2>/dev/null')


class ReaderStopped(ValueError):
    """A child reader stopped responding or died. Its words complete "the reader ...", for the uploader."""


def isolating() -> bool:
    """Whether reads are moved into child processes here: not inside a child, and not where ARGUS_READER_ISOLATION
    is 0 (a test that replaces a reader's functions in this process)."""
    return not os.environ.get(_INLINE) and os.environ.get('ARGUS_READER_ISOLATION', '1') != '0'


def cameras_only() -> bool:
    """Whether this read is the second attempt of one that stopped responding, which reads cameras and nothing else."""
    return os.environ.get(_MODE) == 'cameras'


def progress() -> None:
    """Tell the parent this child is still making progress (a long loop that writes nothing calls this)."""
    beat = os.environ.get(_HEARTBEAT)
    if beat:
        try:
            os.utime(beat)
        except OSError:
            pass


def _entries(root):
    try:
        return set(os.listdir(root))
    except OSError:
        return set()


def _activity(watch, beat):
    """A number that changes while the child works: the heartbeat's time, and the sizes and times of what it writes
    into each watched folder (watch: {folder: the names it held before}, whose earlier entries are not its work)."""
    total = 0
    try:
        total += os.stat(beat).st_mtime_ns
    except OSError:
        pass
    for root, before in watch.items():
        for name in _entries(root) - before:
            for path in [root / name, *((root / name).rglob('*') if (root / name).is_dir() else [])]:
                try:
                    stat = path.stat()
                    total += stat.st_size + stat.st_mtime_ns
                except OSError:
                    continue
    return total


def run_isolated(fn, *args, watch=(), mode=None, stall_s=None, **kwargs):
    """fn(*args, **kwargs), run in a child process when isolating() and inline otherwise. Raises what fn raised, or
    ReaderStopped when the child made no progress for stall_s seconds (it is stopped) or died without a result.
    fn, its arguments and its result must pickle; fn must be a module-level function."""
    if not isolating():
        return fn(*args, **kwargs)
    stall_s = READER_STALL_S if stall_s is None else stall_s
    with tempfile.TemporaryDirectory(prefix='argus-reader-') as scratch:
        scratch = Path(scratch)
        request, result, beat = scratch / 'request.pkl', scratch / 'result.pkl', scratch / 'heartbeat'
        request.write_bytes(pickle.dumps((fn.__module__, fn.__qualname__, args, kwargs)))
        beat.touch()
        env = {**os.environ, _INLINE: '1', _HEARTBEAT: str(beat),
               'PYTHONPATH': os.pathsep.join(p or os.getcwd() for p in sys.path)}
        if mode:
            env[_MODE] = mode
        else:
            env.pop(_MODE, None)
        child = subprocess.Popen([sys.executable, '-m', 'prepare.isolation', str(request), str(result)], env=env,
                                 stdin=subprocess.DEVNULL)
        # A reader stuck inside native code holds the interpreter lock, so it cannot notice its parent die. A shell
        # outside it does, and stops the child when this process is killed before it could.
        guard = subprocess.Popen(['/bin/sh', '-c', _GUARD, 'guard', str(os.getpid()), str(child.pid)],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 start_new_session=True)
        watched = {Path(w): _entries(w) for w in (watch if isinstance(watch, (list, tuple)) else [watch])}
        try:
            seen, since = _activity(watched, beat), time.monotonic()
            while True:
                try:
                    child.wait(timeout=1)       # returns the moment the child exits
                    break
                except subprocess.TimeoutExpired:
                    pass
                now = _activity(watched, beat)
                if now != seen:
                    seen, since = now, time.monotonic()
                elif time.monotonic() - since > stall_s:
                    raise ReaderStopped(f'stopped responding (no progress for {stall_s:g}s) and was stopped')
        finally:
            if child.poll() is None:            # stopped, or the parent was interrupted: never left running
                child.kill()
                child.wait()
            guard.kill()
            guard.wait()
        if not result.exists():
            how = (f'was ended by signal {-child.returncode}' if child.returncode < 0
                   else f'exited with status {child.returncode}')
            raise ReaderStopped(f'{how} before finishing')
        status, value = pickle.loads(result.read_bytes())
    if status == 'ok':
        return value
    raise value


def run_while_progressing(cmd, *, watch=(), stall_s=None, lead_s=0.0, check=False, text=False):
    """subprocess.run for a native tool whose work grows with its input (an encode, a decode of every frame). Its
    bound is the reader's: it is stopped only when it makes no progress for stall_s seconds, never after a fixed time,
    so a long recording is never cut short for its length while a tool that hangs is still stopped. Progress is any
    byte it writes to stdout or stderr, or a change to a watched path (the file it writes). Before its first progress
    it is also allowed lead_s, the time a known stretch of work that shows none takes (frames decoded before the first
    one kept). A stopped tool raises subprocess.TimeoutExpired, as subprocess.run's timeout does."""
    import threading
    stall_s = READER_STALL_S if stall_s is None else stall_s
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    chunks = {'out': [], 'err': []}
    heard = []                                  # times a pipe brought bytes

    def pump(stream, key):
        for block in iter(lambda: stream.read1(1 << 16), b''):
            chunks[key].append(block)
            heard.append(time.monotonic())
    pumps = [threading.Thread(target=pump, args=(proc.stdout, 'out'), daemon=True),
             threading.Thread(target=pump, args=(proc.stderr, 'err'), daemon=True)]
    for t in pumps:
        t.start()

    def files():
        state = []
        for path in watch:
            try:
                stat = os.stat(path)
                state.append((stat.st_size, stat.st_mtime_ns))
            except OSError:
                state.append(None)
        return state
    try:
        seen, started = files(), time.monotonic()
        since, progressed = started, False
        while True:
            try:
                proc.wait(timeout=1)
                break
            except subprocess.TimeoutExpired:
                pass
            now = files()
            if now != seen or (heard and heard[-1] > since):
                seen, since, progressed = now, time.monotonic(), True
            elif time.monotonic() - since > stall_s + (0.0 if progressed else lead_s):
                raise subprocess.TimeoutExpired(cmd, time.monotonic() - started)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        for t in pumps:
            t.join()
    out, err = b''.join(chunks['out']), b''.join(chunks['err'])
    if text:
        out, err = out.decode(errors='replace'), err.decode(errors='replace')
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd, out, err)
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


def _child(request, result):
    module, name, args, kwargs = pickle.loads(Path(request).read_bytes())
    fn = importlib.import_module(module)
    for part in name.split('.'):
        fn = getattr(fn, part)
    try:
        outcome = ('ok', fn(*args, **kwargs))
    except Exception as error:
        outcome = ('error', error)
    try:
        data = pickle.dumps(outcome)
        pickle.loads(data)
    except Exception as error:
        if outcome[0] == 'ok':
            data = pickle.dumps(('error', ReaderStopped(f'returned what could not be passed back '
                                                        f'({type(error).__name__})')))
        else:
            original = outcome[1]
            kind = ValueError if isinstance(original, ValueError) else OSError if isinstance(original, OSError) \
                else RuntimeError
            data = pickle.dumps(('error', kind(f'{type(original).__name__}: {original}')))
    temporary = Path(str(result) + '.partial')
    temporary.write_bytes(data)
    os.replace(temporary, result)


if __name__ == '__main__':
    _child(sys.argv[1], sys.argv[2])
