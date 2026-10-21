"""Every moment flagged over the video (a key event, the goal reached, a task done) keeps one clock: it shows from 0.3s
before its moment until the next later moment, for at most 2.5s (board/index.html captionFlags, flagWindows)."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).parent
pytestmark = pytest.mark.skipif(not shutil.which('node'), reason='no node')

# The page's own functions: what the caption flags at each time t, for the given key events and goal moments.
SCRIPT = r'''
const {src, piece} = require(process.argv[1])();
const line = name => src.slice(src.indexOf(name), src.indexOf('\n', src.indexOf(name)) + 1);
const [keys, goal, tasks, times] = JSON.parse(process.argv[2]);
const code = line('const FLAG_LEAD_S') + ['flagWindows', 'flagsAt', 'captionFlags'].map(piece).join('')
  + `const w = captionFlags(${JSON.stringify(keys)}, ${JSON.stringify(goal)}, ${JSON.stringify(tasks)});
     ${JSON.stringify(times)}.map(t => flagsAt(w, t).map(x => x.goal ? 'goal:' + x.goal : x.label))`;
console.log(JSON.stringify(eval(code)));
'''


def flags(keys, goal, tasks, times):
    p = subprocess.run(['node', '-e', SCRIPT, str(HERE / 'page_functions.js'), json.dumps([keys, goal, tasks, times])],
                       text=True, capture_output=True)
    assert p.returncode == 0, p.stderr
    return dict(zip(times, json.loads(p.stdout)))


def test_a_task_done_ends_with_the_key_event_beside_it():
    # the actionsense case: a phase and a task done at the same instant leave together, at most 2.5s later
    got = flags([{'t_s': 2.0, 'label': 'first display ends'}, {'t_s': 9.0, 'label': 'later'}], None, [2.0],
                [1.8, 2.0, 4.4, 4.6])
    assert got[1.8] == got[2.0] == got[4.4] == ['first display ends', 'goal:task']
    assert got[4.6] == []


def test_the_next_moment_replaces_a_goal_and_a_goal_replaces_a_key_event():
    got = flags([{'t_s': 1.0, 'label': 'grasp'}, {'t_s': 3.0, 'label': 'lift'}], 2.0, [], [1.5, 2.0, 2.6, 2.8, 5.0])
    assert got[1.5] == ['grasp']
    assert got[2.0] == ['goal:episode']             # the goal replaces the grasp
    assert got[2.6] == ['goal:episode']
    assert got[2.8] == ['lift']                     # the next key event replaces the goal 0.3s before its moment
    assert got[5.0] == ['lift']                     # 2.5s after the lift
    assert flags([{'t_s': 3.0, 'label': 'lift'}], None, [], [5.6])[5.6] == []


def test_a_task_done_at_the_episode_goal_is_flagged_once_as_the_goal():
    assert flags([], 4.0, [4.0, 9.0], [4.0])[4.0] == ['goal:episode']


def test_two_key_events_at_one_instant_both_show():
    got = flags([{'t_s': 1.0, 'label': 'a'}, {'t_s': 1.0, 'label': 'b'}], None, [], [1.0, 3.4, 3.6])
    assert sorted(got[1.0]) == sorted(got[3.4]) == ['a', 'b'] and got[3.6] == []
