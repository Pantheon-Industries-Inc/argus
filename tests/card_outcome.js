// The outcome a list card shows (board/serve.py cardOutcomeHtml): a reply that gave no labels says so, a session of
// tasks gives its share done, and a long recording with a part not labelled never reads complete, on its card or in
// its tasks panel (partsGapHtml).
//
//   node tests/card_outcome.js [PAGE_SOURCE]     (default board/serve.py)
//
// Prints nothing and exits 0 when every case holds; prints each failure and exits 1 otherwise.
'use strict';
const {src, piece} = require('./page_functions')(process.argv[2]);
const stubs = 'const ST_WORDS = {unparsed: "did not parse"}; const outcomeWords = (o) => o;';
const T = new Function(stubs + piece('esc') + piece('cardOutcomeHtml') + piece('partsGapHtml')
  + 'return {cardOutcomeHtml, partsGapHtml};')();

let bad = 0;
const check = (ok, what) => { if (!ok) { bad++; console.log('FAIL: ' + what); } };

check(T.cardOutcomeHtml({label_failed: 'unparsed'}).includes('>no labels<'), 'no labels');
check(T.cardOutcomeHtml({n_tasks: 2, n_task_success: 2}).includes('success">2/2 tasks<'), 'a session done');
const gap = T.cardOutcomeHtml({n_tasks: 2, n_task_success: 2, parts_missing: 1, parts: 3});
check(!gap.includes('success') && gap.includes('2/2 tasks') && gap.includes('1 of 3 parts not labelled'),
  'a long recording with a part not labelled never reads complete');
check(T.cardOutcomeHtml({task_completed: 'success'}).includes('>success<'), 'an outcome');
// a recording judged as a whole that lists its tasks (board/label_projection.py shown_tasks): its verdict, then the
// count; a record with no verdict_first mark keeps the count alone, as a session always did
const both = T.cardOutcomeHtml({task_completed: 'failure', verdict_first: true, n_tasks: 2, n_task_success: 1});
check(both.includes('failure">failure, 1/2 tasks<'), 'a verdict beside its tasks');
check(T.cardOutcomeHtml({task_completed: 'unclear', n_tasks: 4, n_task_success: 3}).includes('>3/4 tasks<'),
  'an older record lists its tasks as before');
check(T.cardOutcomeHtml({}).includes('>unrated<'), 'no outcome');
const malformed = T.cardOutcomeHtml({task_completed: 'success"><img src=x onerror="throw Error(1)">'});
check(!malformed.includes('<img') && malformed.includes('&lt;img'),
  'malformed saved completion remains text inside the outcome card');
// the tasks panel of the episode says it too, next to its success count
check(T.partsGapHtml({}) === '' && T.partsGapHtml({_stitched: {parts: 3, missing: []}}) === '', 'no part missing');
check(T.partsGapHtml({_stitched: {parts: 3, missing: [{part: 2}]}}).includes('1 of 3 parts not labelled'),
  'the tasks panel names the parts not labelled');
process.exit(bad ? 1 : 0);
