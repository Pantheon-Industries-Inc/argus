// The board's progress chart (progressPoints and progressAt in board/serve.py), held to one rule: it reaches 100% exactly
// at the goal frame the board shows and nowhere else, so the chart, the goal frame and the outcome never disagree. Run on
// real episodes where they used to disagree (tests/fixtures/progress_cases.json, cut from the published board) and on
// the synthetic cases the chart once got wrong. Data Review's QA viewer carries the same two functions; its tests check
// they are identical.
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
const {progressPoints, progressAt, progressPct} = new Function(grab('progressPoints') + grab('progressAt') + grab('progressPct')
  + 'return {progressPoints, progressAt, progressPct};')();

let bad = 0;
const check = (ok, what) => { if (!ok) { bad++; console.log('FAIL: ' + what); } };
const near = (a, b) => Math.abs(a - b) < 1e-9;
const shown = p => progressPct(p);   // what the readout prints
const step = (t_s, end_s, progress, contribution = 'advancing', arm = 'right') => ({t_s, end_s, progress, contribution, arm});
const goalOf = c => ({completedAt: c.completed_at_s, reachedAt: c.goal_reached_at_s, undoneAt: c.undone_at_s});
const sample = (pts, t0, t1, n = 400) => Array.from({length: n + 1}, (_, k) => t0 + (t1 - t0) * k / n).map(t => [t, progressAt(pts, t)]);

// ------------------------------------------------------------------ real episodes from the published board
const cases = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures', 'progress_cases.json'), 'utf8'));
for (const [name, c] of Object.entries(cases)) {
  const dur = c.duration_s;
  const tasks = c.tasks || [];
  const pts = progressPoints(c.event_labels, dur, tasks.length ? {tasks} : goalOf(c.completion));
  check(pts.every((pt, i) => i === 0 || pt.t >= pts[i - 1].t), `${name}: points are in time order`);
  if (tasks.length) {
    // a session: each task's share is full from its own goal frame, and 100% only when every task is done
    const goals = tasks.map(t => t.completed_at_s).filter(t => t != null).sort((a, b) => a - b);
    const all = goals.length === tasks.length;
    goals.forEach((g, k) => check(progressAt(pts, g) >= (k + 1) / tasks.length - 1e-9,
      `${name}: task goal frame ${g} carries ${k + 1} of ${tasks.length} shares (reads ${progressAt(pts, g)})`));
    const last = all ? goals[goals.length - 1] : dur;
    check(sample(pts, 0, last - 0.06).every(([, p]) => shown(p) < 100), `${name}: under 100% before every task is done`);
    if (all) check(near(progressAt(pts, last), 1) && near(progressAt(pts, dur), 1), `${name}: 100% at the last task's goal frame`);
    else check(sample(pts, 0, dur).every(([, p]) => shown(p) < 100), `${name}: a session with an unfinished task never reads 100%`);
    continue;
  }
  const comp = c.completion;
  const g = comp.completed_at_s != null ? comp.completed_at_s : comp.goal_reached_at_s;   // the frame the board shows
  if (g == null) {
    check(sample(pts, 0, dur).every(([, p]) => shown(p) < 100), `${name}: no goal frame, so never 100%`);
    continue;
  }
  check(near(progressAt(pts, g), 1), `${name}: 100% at the goal frame ${g} (reads ${progressAt(pts, g)})`);
  check(sample(pts, 0, g - 0.06).every(([, p]) => shown(p) < 100), `${name}: under 100% before the goal frame`);
  const u = comp.undone_at_s;
  if (u != null) {
    check(sample(pts, g, u).every(([, p]) => near(p, 1)), `${name}: holds 100% until the goal is undone at ${u}`);
    check(shown(progressAt(pts, dur)) < 100, `${name}: ends under 100% once undone`);
  } else {
    check(sample(pts, g, dur).every(([, p]) => near(p, 1)), `${name}: holds 100% from the goal frame to the end`);
  }
}
// the cases named in the plan, by their real times
const at = (name, t) => {
  const c = cases[name];
  return progressAt(progressPoints(c.event_labels, c.duration_s, c.tasks.length ? {tasks: c.tasks} : goalOf(c.completion)), t);
};
check(near(at('rice_cooker_lid_beyond_instruction', 16.0), 1) && shown(at('rice_cooker_lid_beyond_instruction', 15.9)) < 100,
      'rice cooker: 100% when the spoon is back (16.0 s), not when the lid closes (19 s)');
check(near(at('goal_finished_while_robot_waits', 37.1), 1),
      'a goal finished while the robot waits (the collaborator closes the lid) still reaches 100%');
check(near(at('second_filter_after_goal', 48.53), 1), 'coffee filter: 100% when the first filter is seated, as the goal frame says');
check(near(at('session_with_an_unfinished_task', 14.5), 0.5), 'one of two tasks done reads 50%');

// ------------------------------------------------------------------ synthetic cases the chart once got wrong
// a bimanual pick-and-place with the left arm parked the whole episode (a Data Review upload, as labelled)
const cube = [step(0, 13.9, 0, 'idle', 'left'), step(0, 3, 0.1), step(3, 4, 0.15), step(4, 4.54, 0.3), step(4.54, 6.04, 0.55),
  step(6.04, 7.1, 0.15, 'wasteful'), step(7.1, 7.54, 0.15, 'wasteful'), step(7.54, 9.04, 0.2), step(9.04, 9.6, 0.35),
  step(9.6, 10.57, 0.8), step(10.57, 12.07, 1), step(12.07, 13.57, 1, 'idle'), step(13.57, 13.9, 1, 'idle')];
const pts = progressPoints(cube, 13.938, {completedAt: 12.07});
check(pts.every((pt, i) => i === 0 || pt.t >= pts[i - 1].t), 'points are in time order');
check(near(progressAt(pts, 13.938), 1), `the end reads 100%, not the parked arm's 0 (got ${progressAt(pts, 13.938)})`);
check(shown(progressAt(pts, 11.0)) < 100, `100% only at the goal frame, not during the finishing step (11.0 s reads ${progressAt(pts, 11.0)})`);
check(near(progressAt(pts, 12.07), 1), 'the goal frame reads 100%');
check(pts[pts.length - 1].t === 13.938, 'the last level holds to the end of the episode');
check(progressAt(pts, 7.1) < progressAt(pts, 6.04), 'a release short of the target lowers the level');
// the same labels with no goal frame never read 100%
check(sample(progressPoints(cube, 13.938, {}), 0, 13.938).every(([, p]) => shown(p) < 100), 'labels at 1.0 with no goal frame never read 100%');

// a goal reached and then undone keeps its drop
const undone = [step(0, 5, 0.5), step(5, 8, 1), step(8, 10, 0.3, 'wasteful')];
const up = progressPoints(undone, 10, {reachedAt: 8, undoneAt: 9});
check(near(progressAt(up, 8), 1) && near(progressAt(up, 9), 1), 'an undone goal holds 100% until it is undone');
check(shown(progressAt(up, 10)) < 100, 'a goal that is undone ends under 100%');

// a session of two tasks: each task's own progress folded into the tasks done
const two = [step(0, 4, 0.5), step(4, 6, 1), step(6, 9, 0.5), step(9, 12, 1)];
const tp = progressPoints(two, 12, {tasks: [{start_s: 0, completed_at_s: 6}, {start_s: 6, completed_at_s: 12}]});
check(near(progressAt(tp, 6), 0.5) && near(progressAt(tp, 12), 1),
      `two tasks read 50% after the first and 100% after the second (got ${progressAt(tp, 6)}, ${progressAt(tp, 12)})`);
check(tp.every((pt, i) => i === 0 || pt.p >= tp[i - 1].p), `a session that only advances never falls back (${JSON.stringify(tp)})`);
// the second task never finished: the session stays under 100% however far its own progress got
const tf = progressPoints(two, 12, {tasks: [{start_s: 0, completed_at_s: 6}, {start_s: 6, completed_at_s: null}]});
check(sample(tf, 0, 12).every(([, p]) => shown(p) < 100), 'a session with an unfinished task never reads 100%');

process.exit(bad ? 1 : 0);
