"""How many model requests a batch keeps in flight.

Every labelling entry point sends all of its requests at once unless --concurrency sets a ceiling. Each request is one
model call that takes minutes, so a batch's wall clock is the number of requests divided by the number in flight,
times those minutes, and a small fixed pool only makes a run slower. The real limits are elsewhere: the provider's
rate limit, which the harness meets by retrying 408/409/425/429/5xx with Retry-After (label/harness.py _post), the
spend cap, which holds a request until its reservation fits, and the CPU spent decoding frames, which
label/harness.py DECODE_GATE bounds whatever the number in flight.

The one limit of the machine itself is open files: every request in flight holds a socket. in_flight raises the
process's soft open-file limit to its hard limit and keeps the number in flight within it.
"""
from __future__ import annotations

import os
import sys

FILES_PER_REQUEST = 2      # the request's socket, and a file it reads or writes while it waits
FILES_KEPT_FREE = 64       # for the process itself: logs, outputs, decoders


def open_file_limit() -> int:
    """The soft open-file limit, raised first to the hard limit where the system allows it."""
    try:
        import resource
    except ImportError:          # not POSIX: no limit we can read
        return 1 << 20
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    for target in (hard, 1 << 20, 10240):    # macOS reports an unlimited hard limit but refuses more than OPEN_MAX
        if target == resource.RLIM_INFINITY or target <= soft:
            continue
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            soft = target
            break
        except (ValueError, OSError):
            continue
    return (1 << 20) if soft == resource.RLIM_INFINITY else soft


def open_files() -> int:
    for folder in ('/proc/self/fd', '/dev/fd'):
        try:
            return len(os.listdir(folder))
        except OSError:
            continue
    return 0


def in_flight(requested: int | None, items: int) -> int:
    """Requests to keep in flight for a batch of items: every item when requested is 0 or None, otherwise at most
    requested; never more than the open-file limit can hold, and always at least one."""
    want = max(1, items if not requested or requested <= 0 else min(requested, items))
    room = max(1, (open_file_limit() - open_files() - FILES_KEPT_FREE) // FILES_PER_REQUEST)
    if want > room:
        print(f'{want} requests in flight would need more open files than this process may hold; sending {room} at '
              f'a time (raise the hard limit, ulimit -Hn, to send more)', file=sys.stderr, flush=True)
        return room
    return want


def add_argument(ap, what: str = 'requests') -> None:
    """--concurrency for an entry point: off by default, so every request goes out at once."""
    ap.add_argument('--concurrency', type=int, default=0, metavar='N',
                    help=f'most {what} in flight at once (default 0: all of them)')
