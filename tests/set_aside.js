// What the episode page shows that has no place on the timeline or in the counts (board/serve.py): the steps and key
// events the model gave with no time the board can read (untimedRows), listed after the timed ones with "no time"
// where the time goes, and the issues and checks this dataset's rules set aside (setAsideHtml): each excluded issue
// with the rule's reason, and each check a rule withheld with the reason, in a fold that is closed until opened.
//
//   node tests/set_aside.js [PAGE_SOURCE]     (default board/serve.py)
//
// Prints nothing and exits 0 when every case holds; prints each failure and exits 1 otherwise.
'use strict';
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(process.argv[2] || path.join(__dirname, '..', 'board', 'serve.py'), 'utf8');
const piece = (name) => {
  const a = src.indexOf('\nfunction ' + name + '(');
  if (a < 0) { console.log('no ' + name + ' in the page'); process.exit(1); }
  let i = src.indexOf('{', a), depth = 0;
  for (; i < src.length; i++) {
    if (src[i] === '{') depth++;
    else if (src[i] === '}' && --depth === 0) break;
  }
  return src.slice(a, i + 1) + '\n';
};
const stubs = 'const buildPhrase = e => esc(e.verb_class || ""); const armLabel = a => a || "-";'
  + 'const contribClass = c => c || ""; const tagName = (c) => String(c || "").replace(/_/g, " ");'
  + 'const famName = s => ({"gripper-flat": "Recorded gripper opening never changes"})[s] || s;'
  + 'const OUR_CHECKS = [["gripper_channels", "flagged", "gripper-flat"]];';
const T = new Function(stubs + piece('esc') + piece('fmtT') + piece('untimedRows') + piece('foundWords')
  + piece('setAsideHtml')
  + 'return {untimedRows, setAsideHtml};')();

let bad = 0;
const check = (ok, what) => { if (!ok) { bad++; console.log('FAIL: ' + what); } };

const u = T.untimedRows({event_labels: [{t_s: 1, verb_class: 'reach'}, {t_s: null, verb_class: 'wipe <b>',
                                         contribution: 'advancing', arm: 'left'}],
                         key_events: [{t_s: 2, label: 'grasp'}, {t_s: null, label: 'lid open', outcome: 'success'}]});
check(u.steps.length === 1 && u.steps[0].includes('wipe &lt;b&gt;') && u.steps[0].includes('no time')
  && !u.steps[0].includes('data-t'), 'an untimed step is a row with no time and nothing to seek to');
check(u.keys.length === 1 && u.keys[0].includes('lid open') && u.keys[0].includes('no time')
  && !u.keys[0].includes('data-t'), 'an untimed key event is a row with no time');
check(T.untimedRows({}).steps.length === 0 && T.untimedRows({}).keys.length === 0, 'nothing untimed, no rows');

check(T.setAsideHtml({}) === '' && T.setAsideHtml({_excluded: [], set_aside_checks: []}) === '',
  'nothing set aside draws nothing');
const h = T.setAsideHtml({
  _excluded: [{category: 'missing_instruction', issue: 'No instruction', severity: 'medium', t_s: 3.25,
               excluded_by: 'no_task_text', reason: 'the model inferred the task'},
              {category: 'truncated_episode', issue: 'Ends mid task', list: 'operator_mistakes',
               excluded_by: 'piece_cut', reason: 'our cut'}],
  set_aside_checks: [{check: 'gripper_channels', reason: 'one-armed tasks', status: 'fired'}]});
check(h.includes('No instruction') && h.includes('The model inferred the task.') && h.includes('missing instruction')
  && h.includes('data-t="3.25"'), 'an excluded issue shows its text, tag, time and the rule\'s reason');
check(h.includes('Ends mid task') && h.includes('Our cut.'), 'every excluded issue is listed');
check(h.includes('Recorded gripper opening never changes') && h.includes('One-armed tasks.') && h.includes('fired'),
  'a withheld check shows its name, whether it fired and the reason');
check(h.includes('pub-fold') && h.includes('aria-expanded="false"') && h.includes('3 set aside'),
  'a fold, closed, that says how many are inside');
const e = T.setAsideHtml({set_aside_checks: [{check: 'gripper_channels', reason: 'r', status: 'errored',
                                               error: 'ValueError: boom'}]});
check(e.includes('stopped with an error') && e.includes('ValueError: boom') && !e.includes('clear'),
  'a withheld check that crashed reads errored, never clear');
// a withheld check reads as what it found (board/serve.py withheld_status), never clear when it was not assessed or
// when checks of its own set fired or crashed
const words = c => { const x = T.setAsideHtml({set_aside_checks: [{check: 'capture_qc', reason: 'r', ...c}]});
  return x.slice(x.indexOf('di-issue'), x.indexOf('not counted')); };
check(words({status: 'not_assessed', why: 'the right arm reads at too few frames'})
  .includes('not assessed (the right arm reads at too few frames)'), 'a withheld check not assessed says why');
check(words({status: 'fired', fired: 2, errored: 0, of: 38}).includes('2 of 38 fired'),
  'a withheld set of checks says how many fired');
check(words({status: 'errored', fired: 1, errored: 3, of: 38}).includes('3 of 38 stopped with an error, 1 fired'),
  'a withheld set of checks says how many stopped with an error, and how many fired');
check(words({status: 'clear', fired: 0, errored: 0, of: 30}).includes(', clear'), 'a withheld check that ran clear');
check(!words({status: 'fired', fired: 0, errored: 0, of: 0}).includes('0 of 0')
  && words({status: 'fired', fired: 0, errored: 0, of: 0}).includes('flagged'),
  'a withheld result that fired only because it says it flagged says so, never "0 of 0 fired"');
process.exit(bad ? 1 : 0);
