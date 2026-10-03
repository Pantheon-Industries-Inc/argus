// The outcome a list card shows (board/serve.py cardOutcomeHtml): a reply that gave no labels says so, a session of
// tasks gives its share done, and a long recording with a part not labelled never reads complete.
//
//   node tests/card_outcome.js [PAGE_SOURCE]     (default board/serve.py)
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
const stubs = 'const ST_WORDS = {unparsed: "did not parse"}; const outcomeWords = (o) => o;';
const T = new Function(stubs + piece('esc') + piece('cardOutcomeHtml') + 'return {cardOutcomeHtml};')();

let bad = 0;
const check = (ok, what) => { if (!ok) { bad++; console.log('FAIL: ' + what); } };

check(T.cardOutcomeHtml({label_failed: 'unparsed'}).includes('>no labels<'), 'no labels');
check(T.cardOutcomeHtml({n_tasks: 2, n_task_success: 2}).includes('success">2/2 tasks<'), 'a session done');
const gap = T.cardOutcomeHtml({n_tasks: 2, n_task_success: 2, parts_missing: 1, parts: 3});
check(!gap.includes('success') && gap.includes('2/2 tasks') && gap.includes('1 of 3 parts not labelled'),
  'a long recording with a part not labelled never reads complete');
check(T.cardOutcomeHtml({task_completed: 'success'}).includes('>success<'), 'an outcome');
check(T.cardOutcomeHtml({}).includes('>unrated<'), 'no outcome');
process.exit(bad ? 1 : 0);
