"""The viewer's scores, its Subgoals and key events table, score chip and key-event names, run from the page's own source
in node."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from board import scores

PAGE = Path(__file__).resolve().parents[1] / 'board' / 'index.html'

SCRIPT = r'''
const fs = require('fs'), assert = require('assert');
const s = fs.readFileSync(process.argv[1], 'utf8');
const cut = (a, b) => { const i = s.indexOf(a), j = s.indexOf(b, i); assert(i >= 0 && j > i, a); return s.slice(i, j); };
const esc = x => String(x ?? '').replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');
const fmtT = t => `${Number(t).toFixed(1)}s`;
const kindName = k => String(k || '').replace(/_/g, ' ').trim();

// the scores from one record: the Execution and Outcome boxes, and the score cells of a subgoal's row
const qualitySrc = cut("  // when the episode's goal was met", '  // lanes under the timeline');
const quality = new Function('d', 'comp', 'esc', 'fmtT',
  qualitySrc + 'return {scoresHtml, episodeGoalT, subs, Q, qNum, momentScoreCells, weighted};');
const sub = (o) => ({start_s: 0, end_s: 5, label: 'cup lifted', goal: false, task: null, reached: true, duration_s: 5,
  failed_attempts: [], execution: 9, cleanliness: 9.5, efficiency: 8, steady: 7, wasteful_share: 0.05, penalties: [],
  peers: 12, peer_median_s: 6, peer_basis: 'task', ...o});
const scores = (o) => ({version: Number(process.argv[2]), execution: 8, outcome: 10, cleanliness: 9, efficiency: 7, fluency: 8, peers: 12,
  peer_basis: 'task', relation: 'aligned', weights: JSON.parse(process.argv[3]),
  subgoals: [sub({}), sub({start_s: 5, end_s: 9, peer_basis: 'similar', peers: 200}),
  sub({start_s: 9, end_s: 12, peer_basis: null, peers: 0, peer_median_s: null}), sub({start_s: 12, end_s: 14})],
  inputs: {outcome: 'success', wasteful_share: 0.1, penalties: [], time_to_goal_s: 14, peer_median_time_to_goal_s: 16,
           hesitation_share: 0.05, progress_reached: 1}, ...o});
const speedTip = html => html.match(/class="mo-d num speed" data-k="Speed" title="([^"]*)"/)[1];

let q = quality({scores: scores({})}, {completed_at_s: 14}, esc, fmtT);
assert.deepStrictEqual(q.Q.subgoals.map(st => speedTip(q.momentScoreCells(st))), [
  'Took 5s, median 6s. Median of 12 demos of this task', 'Took 5s, median 6s. Median of 200 similar subgoals in this dataset',
  'Took 5s. Nothing comparable on this board', 'Took 5s, median 6s. Median of 12 demos of this task']);
assert(q.scoresHtml.includes('title="Time from the first action to the goal, against the median of 12 demos of this task"'));
// each part names its weight in execution, as the scorer sets it
const w = JSON.parse(process.argv[3]);
for (const [name, k] of [['Cleanliness', 'cleanliness'], ['Speed', 'efficiency'], ['Steadiness', 'fluency']])
  assert(q.scoresHtml.includes(`${name}<span class="qc-w" title="Its weight in execution">${Math.round(100 * w[k])}%</span>`), name);
// a row's cells are named, so the narrow layout can label them; a row with no stretch keeps the columns, empty
const cells = q.momentScoreCells(q.Q.subgoals[0]);
for (const k of ['Mistakes', 'Cleanliness', 'Speed', 'Steadiness', 'Execution']) assert(cells.includes(`data-k="${k}"`), k);
assert.strictEqual(q.momentScoreCells(null), '<span></span>'.repeat(5));
// older scores with no weights name none
q = quality({scores: scores({weights: undefined})}, {completed_at_s: 14}, esc, fmtT);
assert(!q.scoresHtml.includes('class="qc-w"'));

// a basis that is missing is nothing comparable
const old = scores({peer_basis: undefined});
old.subgoals.forEach(st => delete st.peer_basis);
q = quality({scores: old}, {completed_at_s: 14}, esc, fmtT);
assert(q.Q.subgoals.every(st => speedTip(q.momentScoreCells(st)).endsWith('Nothing comparable on this board')));

// nothing measured: a dash, no "/10", no bar and no invented 0%
q = quality({scores: scores({execution: null, cleanliness: null, efficiency: null, fluency: null, peer_basis: null,
  subgoals: [sub({execution: null, cleanliness: null, efficiency: null, steady: null, duration_s: null, peer_basis: null,
    peer_median_s: null})],
  inputs: {outcome: 'success', penalties: [], time_to_goal_s: null}})}, {completed_at_s: 14}, esc, fmtT);
const exBox = q.scoresHtml.slice(q.scoresHtml.indexOf('Execution</span>'), q.scoresHtml.indexOf('Outcome</span>'));
assert(exBox.includes('<span class="qual-score">-</span>') && !exBox.includes('/10'), exBox.slice(0, 200));
assert(exBox.includes('qual-bar none') && !exBox.includes('% wasteful') && !exBox.includes('% waiting'));
// shown on two cells; the Steadiness tooltip also names busyness as not measured
assert.strictEqual((exBox.replace(/title="[^"]*"/g, '').match(/not measured/g) || []).length, 2);
assert(exBox.includes('busyness (actions per unit of progress against other demos of this task, not measured)'));
const row = q.momentScoreCells(q.Q.subgoals[0]);
assert(row.includes('<b>-</b></span>') && !row.includes('-s<') && !row.includes('qual-bar') && !row.includes('NaN'), row);

// progress that is unknown (missing or null) is "not measured", never an invented 0% or NaN%; a measured share is its
// percent
const progressOf = p => {
  const h = quality({scores: scores({inputs: {outcome: 'failure', penalties: [], progress_reached: p}})},
    {completed_at_s: null}, esc, fmtT).scoresHtml;
  return h.slice(h.indexOf('<div class="qc-k">Progress</div>'), h.indexOf('<div class="qc-k">Label</div>'));
};
for (const p of [undefined, null]) {
  const cellHtml = progressOf(p);
  assert(cellHtml.startsWith('<div class="qc-k">Progress</div><div class="qc-v qc-word">not measured</div>')
    && !cellHtml.includes('%'), cellHtml);
}
assert(progressOf(0.5).startsWith('<div class="qc-k">Progress</div><div class="qc-v">50%</div>'), progressOf(0.5));
assert(progressOf(0).startsWith('<div class="qc-k">Progress</div><div class="qc-v">0%</div>'), progressOf(0));

// a stretch's mistakes read by severity, most severe first, with the failed tries no mistake explains; each is listed
// with its time in the tooltip; none reads none
q = quality({scores: scores({})}, {completed_at_s: 14}, esc, fmtT);
const mis = q.momentScoreCells(sub({penalties: [{mistake: 'dropped the cup', severity: 'medium', recovered: true, t_s: 2},
  {mistake: 'knocked the stand', severity: 'high', recovered: false, t_s: 3},
  {mistake: 'paused', severity: 'medium', recovered: false, t_s: 4}],
  failed_attempts: [{t_s: 2, label: 'slipped'}, {t_s: 3}]}));
assert(mis.includes('<span class="mo-sev high">high</span><span class="mo-sev medium">2 medium</span>'
  + '<span class="mo-sev try">2 failed tries</span>'), mis);
assert(mis.includes('title="2.0s  dropped the cup (medium, recovered)\n3.0s  knocked the stand (high)'), mis);
assert(q.momentScoreCells(sub({})).includes('<span class="mo-none">none</span>'));
// scores written before stretches carried penalties still count their mistakes, unrated
assert(q.momentScoreCells(sub({penalties: undefined, mistakes: ['a', 'b']})).includes('<span class="mo-sev">2 unrated</span>'));

// a goal reached and later undone has only goal_reached_at_s: the Outcome box, caption and chip all read it
q = quality({scores: scores({})}, {completed_at_s: null, goal_reached_at_s: 44}, esc, fmtT);
assert.strictEqual(q.episodeGoalT, 44);
assert(q.scoresHtml.includes('reached 44.0s'));

// no goal time reads "not reached" only for a failed or partial verdict; an unclear one says so, unflagged; a session
// with no labelled task says that, with the completion's reason (a part with no labels) as its tooltip
const outcomeBox = html => html.slice(html.indexOf('Outcome</span>'));
q = quality({scores: scores({inputs: {outcome: 'unclear', penalties: [], progress_reached: null}})}, {}, esc, fmtT);
assert(outcomeBox(q.scoresHtml).includes('unclear') && !outcomeBox(q.scoresHtml).includes('not reached'));
q = quality({scores: scores({inputs: {outcome: 'failure', penalties: [], progress_reached: .4}})}, {}, esc, fmtT);
assert(outcomeBox(q.scoresHtml).includes('not reached'));
const why = "a session of several tasks; each task's outcome is under tasks; part 1 has no labels, as reply cut off";
q = quality({prompt_mode: 'session', scores: scores({inputs: {outcome: null, penalties: [], progress_reached: null}})},
            {reason: why}, esc, fmtT);
const sessionBox = outcomeBox(q.scoresHtml);
assert(sessionBox.includes('none labelled') && sessionBox.includes(esc(why)) && !sessionBox.includes('not reached'), sessionBox);

// the Subgoals and key events table: every key event in time order, a subgoal's scores on the row of the key event that
// ends it, the goal on its own row when no key event marks it, a subgoal not reached says so, and a key event with no
// time comes last with nothing to seek to
const momentsSrc = cut('  // Subgoals and key events, one table in time order', '  // the state changes; the scene graph');
const momentsOf = (keyEvents, d, sc) => {
  const Qm = sc ? quality({scores: sc}, {completed_at_s: 14}, esc, fmtT) : null;
  return new Function('keyEvents', 'subs', 'Q', 'qIn', 'd', 'esc', 'fmtT', 'keyKindName', 'momentScoreCells', 'weighted',
    'foldRows', momentsSrc + 'return momentsHtml;')(keyEvents, Qm ? Qm.subs : [], Qm ? Qm.Q : null,
    Qm ? Qm.Q.inputs : {}, d, esc, fmtT, k => k.kind === 'subgoal_complete' ? 'Subgoal' : kindName(k.kind),
    Qm ? Qm.momentScoreCells : null, Qm ? Qm.weighted : x => x, rows => rows.join(''));
};
const mk = [{t_s: 3, kind: 'contact', label: 'touches cup'}, {t_s: 5, kind: 'subgoal_complete', label: 'cup lifted'},
  {t_s: 9, kind: 'subgoal_complete', label: 'cup moved', outcome: 'failure'}];
const goalSub = sub({start_s: 9, end_s: 14, goal: true, label: 'cup on the shelf'});
let html = momentsOf(mk, {key_events: [...mk, {label: 'checks the shelf'}]},
  scores({subgoals: [sub({}), sub({start_s: 5, end_s: 9, reached: false, outcome: 'failure', execution: null}), goalSub]}));
const rowsOf = h => h.split('<div class="mo-row ').slice(2);
let rs = rowsOf(html);
assert(html.includes('Subgoals and key events <span class="count">5</span>'), html.slice(0, 200));
assert.deepStrictEqual(rs.map(r => r.match(/data-t="([^"]*)"/)?.[1] ?? null), ['3', '5', '9', '14', null]);
assert(rs[0].includes('touches cup') && rs[0].includes('<span></span>'.repeat(5)), rs[0]);
assert(rs[1].includes('cup lifted') && rs[1].includes('<b>9</b>'), rs[1]);
assert(rs[2].includes('<span class="outcome failure">not reached</span>'), rs[2]);
assert(rs[3].startsWith('success goal"') && rs[3].includes('>Goal</span>') && rs[3].includes('cup on the shelf'), rs[3]);
assert(rs[4].startsWith('untimed') && rs[4].includes('no time') && rs[4].includes('checks the shelf'), rs[4]);
// unscored: the key events alone, with no score columns
html = momentsOf(mk, {key_events: mk}, null);
assert(html.includes('Key events <span class="count">3</span>') && !html.includes('Mistakes') && !html.includes('mo-exec'), html);
assert(!html.includes('class="moments scored"'));

// key-event kinds: a subgoal_complete is "Subgoal N" everywhere, numbered like the Quality table when scored and
// among subgoals only when not
const nameSrc = cut("  // a key event's kind in words", '  let keyMarkersHtml');
const names = new Function('keyEvents', 'subs', 'Q', 'kindName', nameSrc + 'return keyEvents.map(keyKindName);');
const kes = [{t_s: 3, kind: 'contact'}, {t_s: 5, kind: 'subgoal_complete'}, {t_s: 7, kind: 'grasp'},
  {t_s: 9.0000001, kind: 'subgoal_complete'}];
const qs = scores({});
assert.deepStrictEqual(names(kes, qs.subgoals, qs, kindName), ['contact', 'Subgoal 1', 'grasp', 'Subgoal 2']);
assert.deepStrictEqual(names(kes, [], null, kindName), ['contact', 'Subgoal 1', 'grasp', 'Subgoal 2']);
for (const src of [cut('const keyHtml = k ?', '</span>'), cut('class="tip">${esc(fmtT(k.t_s))}', '}${esc(k.label')])
  assert(src.includes('keyKindName(k)') && !src.includes('kindName(k.kind)'), src);

// the score chip: only beside a caption whose key event is a subgoal, or the goal reached
// the caption's moments keep one clock (captionFlags), loaded from the page with the chip
const chipSrc = cut('const FLAG_LEAD_S', '// a key event') + cut('  const subNo = new Map();', '  let subN = 0;')
  + cut('  const captionWindows =', '  const stateChanges =')
  + cut('  let _scSig = null', "  document.querySelectorAll('.moments .mo-row");
const chipEl = {innerHTML: '', on: false, classList: {add() { chipEl.on = true; }, remove() { chipEl.on = false; }}};
const chip = new Function('keyEvents', 'subs', 'Q', 'qNum', 'episodeGoalT', 'taskGoalTimes', 'document',
  chipSrc + 'return renderScoreChip;');
const chipKes = [{t_s: 4.5, kind: 'contact'}, {t_s: 5, kind: 'subgoal_complete'}];
const render = chip(chipKes, qs.subgoals, qs, x => String(x), 44, [], {getElementById: () => chipEl});
render(4.5);   // the caption is on the contact, so no score beside it
assert.strictEqual(chipEl.on, false);
render(5.2);   // the subgoal came after the contact, so it replaced it at once: the caption is on Subgoal 1
assert(chipEl.on && chipEl.innerHTML === '<span class="sc-k">Subgoal 1</span><span class="sc-v of10"><span class="sc-num">9</span><span class="sc-den">10</span></span>', chipEl.innerHTML);
render(6.4);   // still within the subgoal's 1.5s
assert(chipEl.on && chipEl.innerHTML === '<span class="sc-k">Subgoal 1</span><span class="sc-v of10"><span class="sc-num">9</span><span class="sc-den">10</span></span>', chipEl.innerHTML);
render(20);
assert.strictEqual(chipEl.on, false);
render(44.5);  // the goal from goal_reached_at_s
assert(chipEl.on && chipEl.innerHTML === '<span class="sc-k">Execution</span><span class="sc-v of10"><span class="sc-num">8</span><span class="sc-den">10</span></span>'
  + '<span class="sc-k">Outcome</span><span class="sc-v of10"><span class="sc-num">10</span><span class="sc-den">10</span></span>', chipEl.innerHTML);

// a closed fold names the row the video is on; an open one does not
const foldLabel = new Function(cut("function foldLabel(btn)", 'function foldToggle') + 'return foldLabel;')();
const FOLD = (open, now) => ({dataset: {n: '23', noun: 'key events', now}, closest: () => ({classList: {contains: () => open}})});
globalThis.FOLD_FIRST = 5;
assert.strictEqual(foldLabel(FOLD(false, '14')), 'Show all 23 key events<span class="fold-now">, now at 14</span>');
assert.strictEqual(foldLabel(FOLD(false, '')), 'Show all 23 key events');
assert.strictEqual(foldLabel(FOLD(true, '14')), 'Show the first 5');
'''


@pytest.mark.skipif(not shutil.which('node'), reason='no node')
def test_quality_section_chip_and_key_event_names_run_from_the_page():
    # the version the scorer writes, so a bump the page does not follow fails here
    p = subprocess.run(['node', '-e', SCRIPT, str(PAGE), str(scores.VERSION), json.dumps(scores.EXECUTION_WEIGHTS)],
                       text=True, capture_output=True)
    assert p.returncode == 0, p.stderr


def test_quality_css_uses_only_tokens_the_page_defines():
    page = PAGE.read_text()
    root = page[page.index(':root {'):page.index('}', page.index(':root {'))]
    defined = set(re.findall(r'(--[\w-]+)\s*:', root))
    start = page.index('/* the Quality section:')
    css = page[start:page.index('/* top-right transient toast', start)]
    used = set(re.findall(r'var\((--[\w-]+)\)', css))   # a var() with a fallback is allowed to be undefined
    local = set(re.findall(r'(--[\w-]+)\s*:', css))       # a property the section sets for its own boxes
    assert used and used <= defined | local, used - defined - local
    # the narrow layout puts each row's scores on lines of their own, where a cell no wider than its name goes on to the
    # next line instead of running into the next cell (tests/test_layout_fits.py checks it at a phone's width)
    narrow = css[css.index('@container (max-width: 620px)'):]
    assert 'flex-wrap: wrap' in narrow and 'min-width: max-content' in narrow
