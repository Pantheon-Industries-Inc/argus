"""The viewer's Quality section, score chip and key-event names, run from the page's own source in node."""
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

// the Quality section from one record
const qualitySrc = cut("  // when the episode's goal was met", '  // lanes under the timeline');
const quality = new Function('d', 'comp', 'esc', 'fmtT', 'SUBS_OPEN', 'SUBS_FIRST',
  qualitySrc + 'return {qualityHtml, episodeGoalT, subs, Q, qNum};');
const sub = (o) => ({start_s: 0, end_s: 5, label: 'cup lifted', goal: false, task: null, reached: true, duration_s: 5,
  failed_attempts: [], execution: 9, peers: 12, peer_median_s: 6, peer_basis: 'task', ...o});
const scores = (o) => ({version: Number(process.argv[2]), execution: 8, outcome: 10, cleanliness: 9, efficiency: 7, fluency: 8, peers: 12,
  peer_basis: 'task', relation: 'aligned', subgoals: [sub({}), sub({start_s: 5, end_s: 9, peer_basis: 'similar', peers: 200}),
  sub({start_s: 9, end_s: 12, peer_basis: null, peers: 0, peer_median_s: null}), sub({start_s: 12, end_s: 14})],
  inputs: {outcome: 'success', wasteful_share: 0.1, penalties: [], time_to_goal_s: 14, peer_median_time_to_goal_s: 16,
           hesitation_share: 0.05, progress_reached: 1}, ...o});
const titles = html => [...html.matchAll(/class="num q-med" title="([^"]*)"/g)].map(m => m[1]);

let q = quality({scores: scores({})}, {completed_at_s: 14}, esc, fmtT, false, 5);
assert.deepStrictEqual(titles(q.qualityHtml), ['Median of 12 demos of this task',
  'Median of 200 similar subgoals in this dataset', 'Nothing comparable on this board', 'Median of 12 demos of this task']);
assert(q.qualityHtml.includes('title="Time from the first action to the goal, against the median of 12 demos of this task"'));
// every column is named, so the narrow layout can place it; the folded table carries the same columns
assert((q.qualityHtml.match(/<col class="q-exec">/g) || []).length === 1);
for (const c of ['q-no', 'q-at', 'qt-name', 'q-took', 'q-med', 'q-fail', 'q-exec'])
  assert(q.qualityHtml.includes(`<td class="${c}`) || q.qualityHtml.includes(`<td class="num ${c}`), c);

// a basis that is missing is nothing comparable
const old = scores({peer_basis: undefined});
old.subgoals.forEach(st => delete st.peer_basis);
q = quality({scores: old}, {completed_at_s: 14}, esc, fmtT, false, 5);
assert(titles(q.qualityHtml).every(t => t === 'Nothing comparable on this board'));

// nothing measured: a dash, no "/10", no bar and no invented 0%
q = quality({scores: scores({execution: null, cleanliness: null, efficiency: null, fluency: null, peer_basis: null,
  subgoals: [sub({execution: null, duration_s: null, peer_basis: null, peer_median_s: null})],
  inputs: {outcome: 'success', penalties: [], time_to_goal_s: null}})}, {completed_at_s: 14}, esc, fmtT, false, 5);
const exBox = q.qualityHtml.slice(q.qualityHtml.indexOf('Execution</span>'), q.qualityHtml.indexOf('Outcome</span>'));
assert(exBox.includes('<span class="qual-score">-</span>') && !exBox.includes('/10'), exBox.slice(0, 200));
assert(exBox.includes('qual-bar none') && !exBox.includes('% wasteful') && !exBox.includes('% waiting'));
// shown on two cells; the Steadiness tooltip also names busyness as not measured
assert.strictEqual((exBox.replace(/title="[^"]*"/g, '').match(/not measured/g) || []).length, 2);
assert(exBox.includes('busyness (actions per unit of progress against other demos of this task, not measured)'));
const row = q.qualityHtml.slice(q.qualityHtml.indexOf('<tr data-t'));
assert(row.includes('<b>-</b></span>') && !row.includes('-s<') && !row.includes('qual-bar'));

// progress that is unknown (missing or null) is "not measured", never an invented 0% or NaN%; a measured share is its
// percent
const progressOf = p => {
  const h = quality({scores: scores({inputs: {outcome: 'failure', penalties: [], progress_reached: p}})},
    {completed_at_s: null}, esc, fmtT, false, 5).qualityHtml;
  return h.slice(h.indexOf('<div class="qc-k">Progress</div>'), h.indexOf('<div class="qc-k">Label</div>'));
};
for (const p of [undefined, null]) {
  const cellHtml = progressOf(p);
  assert(cellHtml.startsWith('<div class="qc-k">Progress</div><div class="qc-v qc-word">not measured</div>')
    && !cellHtml.includes('%'), cellHtml);
}
assert(progressOf(0.5).startsWith('<div class="qc-k">Progress</div><div class="qc-v">50%</div>'), progressOf(0.5));
assert(progressOf(0).startsWith('<div class="qc-k">Progress</div><div class="qc-v">0%</div>'), progressOf(0));

// a subgoal not reached is flagged in Done at, one the footage leaves unclear is not; its failed tries are counted,
// flagged and listed in the tooltip
q = quality({scores: scores({subgoals: [
  sub({reached: false, outcome: 'failure', failed_attempts: [{t_s: 2, label: 'slipped'}, {t_s: 3}]}),
  sub({start_s: 5, end_s: 9, reached: false, outcome: 'unclear'}), sub({start_s: 9, end_s: 12})]})},
  {completed_at_s: 14}, esc, fmtT, false, 5);
const subRows = q.qualityHtml.split('<tr data-t').slice(1);
assert(subRows[0].includes('<td class="num q-at qc-bad">not reached</td>'), subRows[0]);
assert(subRows[1].includes('<td class="num q-at">unclear</td>'), subRows[1]);
assert(subRows[2].includes('<td class="num q-at">12.0s</td>'), subRows[2]);
assert(subRows[0].includes('<td class="num q-fail qc-bad" title="2.0s  slipped\n3.0s  ">2</td>'), subRows[0]);
assert(subRows[2].includes('<td class="num q-fail" title="">0</td>'), subRows[2]);

// a goal reached and later undone has only goal_reached_at_s: the Outcome box, caption and chip all read it
q = quality({scores: scores({})}, {completed_at_s: null, goal_reached_at_s: 44}, esc, fmtT, false, 5);
assert.strictEqual(q.episodeGoalT, 44);
assert(q.qualityHtml.includes('reached 44.0s'));

// no goal time reads "not reached" only for a failed or partial verdict; an unclear one says so, unflagged; a session
// with no labelled task says that, with the completion's reason (a part with no labels) as its tooltip
const outcomeBox = html => html.slice(html.indexOf('Outcome</span>'));
q = quality({scores: scores({inputs: {outcome: 'unclear', penalties: [], progress_reached: null}})}, {}, esc, fmtT, false, 5);
assert(outcomeBox(q.qualityHtml).includes('unclear') && !outcomeBox(q.qualityHtml).includes('not reached'));
q = quality({scores: scores({inputs: {outcome: 'failure', penalties: [], progress_reached: .4}})}, {}, esc, fmtT, false, 5);
assert(outcomeBox(q.qualityHtml).includes('not reached'));
const why = "a session of several tasks; each task's outcome is under tasks; part 1 has no labels, as reply cut off";
q = quality({prompt_mode: 'session', scores: scores({inputs: {outcome: null, penalties: [], progress_reached: null}})},
            {reason: why}, esc, fmtT, false, 5);
const sessionBox = outcomeBox(q.qualityHtml);
assert(sessionBox.includes('none labelled') && sessionBox.includes(esc(why)) && !sessionBox.includes('not reached'), sessionBox);

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
const chipSrc = cut('  const subNo = new Map();', '  let subN = 0;') + cut('  const keyWindows =', '  const stateChanges =')
  + cut('  let _scSig = null', "  document.querySelectorAll('.qual-table");
const chipEl = {innerHTML: '', on: false, classList: {add() { chipEl.on = true; }, remove() { chipEl.on = false; }}};
const chip = new Function('keyEvents', 'subs', 'Q', 'qNum', 'episodeGoalT', 'document',
  chipSrc + 'return renderScoreChip;');
const chipKes = [{t_s: 4.5, kind: 'contact'}, {t_s: 5, kind: 'subgoal_complete'}];
const render = chip(chipKes, qs.subgoals, qs, x => String(x), 44, {getElementById: () => chipEl});
render(4.5);   // the caption is on the contact, so no score beside it
assert.strictEqual(chipEl.on, false);
render(5.2);   // the subgoal came after the contact, so it replaced it at once: the caption is on Subgoal 1
assert(chipEl.on && chipEl.innerHTML === '<span class="sc-k">Subgoal 1</span><span class="sc-v">9</span>', chipEl.innerHTML);
render(7.2);   // still within the subgoal's 2.5s
assert(chipEl.on && chipEl.innerHTML === '<span class="sc-k">Subgoal 1</span><span class="sc-v">9</span>', chipEl.innerHTML);
render(20);
assert.strictEqual(chipEl.on, false);
render(44.5);  // the goal from goal_reached_at_s
assert(chipEl.on && chipEl.innerHTML === '<span class="sc-k">Episode</span><span class="sc-v">8</span>'
  + '<span class="sc-k">Outcome</span><span class="sc-v">10</span>', chipEl.innerHTML);

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
    p = subprocess.run(['node', '-e', SCRIPT, str(PAGE), str(scores.VERSION)], text=True, capture_output=True)
    assert p.returncode == 0, p.stderr


def test_quality_css_uses_only_tokens_the_page_defines():
    page = PAGE.read_text()
    root = page[page.index(':root {'):page.index('}', page.index(':root {'))]
    defined = set(re.findall(r'(--[\w-]+)\s*:', root))
    start = page.index('/* the Quality section:')
    css = page[start:page.index('/* top-right transient toast', start)]
    used = set(re.findall(r'var\((--[\w-]+)\)', css))   # a var() with a fallback is allowed to be undefined
    assert used and used <= defined, used - defined
    # the narrow layout keeps the header, the shown rows and the folded rows on one grid
    assert '@container (max-width: 620px)' in css and 'grid-template-areas' in css
