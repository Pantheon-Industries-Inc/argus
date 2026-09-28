// The board's progress readout (progressPoints and progressAt in board/serve.py), run on the cases it got wrong:
// 100% shown while the finishing step was still under way, and a parked arm's whole-episode step drawing a drop to 0
// at the end. Data Review's QA viewer carries the same two functions; tests/test_board.py checks they are identical.
//
//   node tests/progress_points.js [PAGE_SOURCE]     (default board/serve.py)
//
// Prints nothing and exits 0 when every case holds; prints each failure and exits 1 otherwise.
'use strict';
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(process.argv[2] || path.join(__dirname, '..', 'board', 'serve.py'), 'utf8');
const grab = name => {
  const m = src.match(new RegExp(`\\nfunction ${name}\\([\\s\\S]*?\\n}\\n`));
  if (!m) { console.log(`no function ${name} in the page`); process.exit(1); }
  return m[0];
};
const {progressPoints, progressAt} = new Function(grab('progressPoints') + grab('progressAt') + 'return {progressPoints, progressAt};')();

let bad = 0;
const check = (ok, what) => { if (!ok) { bad++; console.log('FAIL: ' + what); } };
const step = (t_s, end_s, progress, contribution = 'advancing', arm = 'right') => ({t_s, end_s, progress, contribution, arm});

// a bimanual pick-and-place with the left arm parked the whole episode (a Data Review upload, as labelled)
const cube = [step(0, 13.9, 0, 'idle', 'left'), step(0, 3, 0.1), step(3, 4, 0.15), step(4, 4.54, 0.3), step(4.54, 6.04, 0.55),
  step(6.04, 7.1, 0.15, 'wasteful'), step(7.1, 7.54, 0.15, 'wasteful'), step(7.54, 9.04, 0.2), step(9.04, 9.6, 0.35),
  step(9.6, 10.57, 0.8), step(10.57, 12.07, 1), step(12.07, 13.57, 1, 'idle'), step(13.57, 13.9, 1, 'idle')];
const pts = progressPoints(cube, 13.938, []);
check(pts.every((pt, i) => i === 0 || pt.t >= pts[i - 1].t), 'points are in time order');
check(Math.abs(progressAt(pts, 13.938) - 1) < 1e-9, `the end reads 100%, not the parked arm's 0 (got ${progressAt(pts, 13.938)})`);
check(!pts.some((pt, i) => i > 0 && pt.p < 1 && pts.slice(0, i).some(q => q.p === 1)), 'no drop after the goal is reached');
check(progressAt(pts, 11.0) < 1, `100% only when the finishing step ends, not during it (11.0 s reads ${progressAt(pts, 11.0)})`);
check(Math.abs(progressAt(pts, 12.07) - 1) < 1e-9, 'the finishing step ends at 100%');
check(pts[pts.length - 1].t === 13.938, 'the last level holds to the end of the episode');
check(progressAt(pts, 7.1) < progressAt(pts, 6.04), 'a release short of the target lowers the level');

// a goal reached and then undone keeps its drop
const undone = [step(0, 5, 0.5), step(5, 8, 1), step(8, 10, 0.3, 'wasteful')];
check(Math.abs(progressAt(progressPoints(undone, 10, []), 10) - 0.3) < 1e-9, 'a goal that is undone ends at the lower level');

// a session of two tasks: each task's own progress folded into the tasks done
const two = [step(0, 4, 0.5), step(4, 6, 1), step(6, 9, 0.5), step(9, 12, 1)];
const tp = progressPoints(two, 12, [6, 12]);
check(Math.abs(progressAt(tp, 6) - 0.5) < 1e-9 && Math.abs(progressAt(tp, 12) - 1) < 1e-9,
      `two tasks read 50% after the first and 100% after the second (got ${progressAt(tp, 6)}, ${progressAt(tp, 12)})`);
check(tp.every((pt, i) => i === 0 || pt.p >= tp[i - 1].p), `a session that only advances never falls back (${JSON.stringify(tp)})`);

process.exit(bad ? 1 : 0);
