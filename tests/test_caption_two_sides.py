"""The caption over the video (board/index.html captionSteps, captionHtml) shows the step that started last among
those still running (the one that started last when none runs), and while a left and a right step both run (the two
grippers or hands doing different things at once) one row for each side, left first. Before, it showed only the step
that started last, so the other side's overlapping step never showed."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).parent
SOURCE = HERE.parent / 'board' / 'index.html'
pytestmark = pytest.mark.skipif(not shutil.which('node'), reason='no node')

# The page's own functions: the caption's markup at each time t for the given steps, built as renderOverlay builds its
# windows (t0 the start, t1 the end or a second after the start).
SCRIPT = r'''
const {piece} = require(process.argv[1])();
const [steps, times] = JSON.parse(process.argv[2]);
const code = "let ARM_NOUN = 'grippers';\n" + ['esc', 'fmtT', 'armLabel', 'contribText', 'contribClass', 'buildPhrase',
  'captionSteps', 'captionRowHtml', 'captionHtml'].map(piece).join('')
  + `const w = ${JSON.stringify(steps)}.map((e, i) => ({t0: e.t_s, t1: e.end_s != null ? e.end_s : e.t_s + 1.0,
       ev: {...e, event_idx: i}})).sort((a, b) => a.t0 - b.t0);
     ${JSON.stringify(times)}.map(t => captionHtml(captionSteps(w, t)))`;
console.log(JSON.stringify(eval(code)));
'''


def captions(steps, times):
    p = subprocess.run(['node', '-e', SCRIPT, str(HERE / 'page_functions.js'), json.dumps([steps, times])],
                       text=True, capture_output=True)
    assert p.returncode == 0, p.stderr
    return dict(zip(times, json.loads(p.stdout)))


def step(t0, t1, arm, verb, contribution='advancing'):
    return {'t_s': t0, 'end_s': t1, 'arm': arm, 'verb_class': verb, 'contribution': contribution}


def test_overlapping_left_and_right_steps_each_get_a_caption_row():
    # the shape of a real UMI moment: the left gripper grasps one object while the right places another
    got = captions([step(2.0, 6.0, 'left', 'grasp the red cup'), step(3.0, 5.0, 'right', 'place the lid on the board')],
                   [2.5, 4.0, 5.5])
    assert got[2.5].count('vo-row') == 0 and 'grasp the red cup' in got[2.5]
    both = got[4.0]
    assert both.count('class="vo-row"') == 2
    assert 'grasp the red cup' in both and 'place the lid on the board' in both
    assert both.index('vo-arm left') < both.index('vo-arm right')          # left first, whichever started later
    # once the right step ends, the left step still running is the caption, though the right one started later
    assert 'place the lid' not in got[5.5] and 'grasp the red cup' in got[5.5]


def test_one_side_at_a_time_and_both_steps_keep_one_caption():
    got = captions([step(0.0, 2.0, 'left', 'reach for the cup'), step(2.0, 4.0, 'right', 'grasp the lid'),
                    step(4.0, 8.0, 'both', 'carry the tray together'), step(5.0, 6.0, 'left', 'tilt the tray')],
                   [1.0, 3.0, 4.5, 5.5, 9.0])
    assert 'reach for the cup' in got[1.0] and got[1.0].count('vo-row') == 0
    assert 'grasp the lid' in got[3.0] and 'reach for the cup' not in got[3.0]
    assert 'carry the tray together' in got[4.5] and 'both grippers' in got[4.5]
    # a side step inside a both step is the latest step, shown alone: only a left and a right step make two rows
    assert 'tilt the tray' in got[5.5] and got[5.5].count('vo-row') == 0
    # after every step ends, the step that started last stays, as before
    assert 'tilt the tray' in got[9.0] and 'carry the tray' not in got[9.0]


def test_a_side_change_without_overlap_keeps_one_caption_row():
    # the caption looks 0.05s ahead for the next step's start; the step ending there has ended by the same moment.
    # A real UMI label (left lifts a valve until 13.01s, right approaches it from 13.01s) showed both rows at 13.0s,
    # a flash of two rows at every change of side, though the two steps never overlap.
    got = captions([step(12.5, 13.01, 'left', 'lift valve'), step(13.01, 14.01, 'right', 'approach held valve'),
                    step(14.01, 15.0, 'left', 'turn valve')], [12.9, 12.97, 13.0, 13.96, 13.99, 14.5])
    for t in (12.97, 13.0):
        assert got[t].count('vo-row') == 0 and 'approach held valve' in got[t] and 'lift valve' not in got[t], got[t]
    for t in (13.96, 13.99):
        assert got[t].count('vo-row') == 0 and 'turn valve' in got[t] and 'approach held valve' not in got[t], got[t]
    assert 'lift valve' in got[12.9] and 'turn valve' in got[14.5]
    # a real overlap still gets its two rows up to the moment one side ends
    got = captions([step(2.0, 6.0, 'left', 'grasp the red cup'), step(3.0, 5.0, 'right', 'place the lid')], [4.9])
    assert got[4.9].count('class="vo-row"') == 2


def test_the_page_draws_the_caption_from_these_functions():
    src = SOURCE.read_text()
    body = src[src.index('  function renderOverlay(t) {'):src.index('  // the score chip shows a subgoal')]
    assert 'captionHtml(rows)' in body and 'captionSteps(segWindows, t)' in body
    assert '.video-overlay .vo-row' in src
