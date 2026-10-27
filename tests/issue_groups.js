// A grouped issue's card on the board (board/index.html issueOccurrencesHtml, from label/issue_groups.py): every report
// it holds is listed as its part wrote it, escaped, with its time, part and evidence; an issue with fewer than two
// occurrences draws nothing extra.
//
//   node tests/issue_groups.js [PAGE_SOURCE]     (default board/index.html)
//
// Prints nothing and exits 0 when every case holds; prints each failure and exits 1 otherwise.
'use strict';
const {piece} = require('./page_functions')(process.argv[2]);
const T = new Function(piece('esc') + piece('fmtT') + piece('issueOccurrences') + piece('issueOccurrencesHtml')
  + 'return {issueOccurrences, issueOccurrencesHtml};')();

let bad = 0;
const check = (ok, what) => { if (!ok) { bad++; console.log('FAIL: ' + what); } };

check(T.issueOccurrencesHtml({issue: 'one'}) === '' && T.issueOccurrences({occurrences: [{issue: 'a'}]}) === null,
  'an issue that is not grouped draws nothing extra');
const html = T.issueOccurrencesHtml({issue: 'The wrist cameras are swapped.', occurrences: [
  {issue: 'Left and right <b> appear reversed.', t_s: 0, end_s: 438.27, part: 1, severity: 'high',
   evidence: 'At 0.00 s, left sees the other gripper.'},
  {issue: 'The side claims appear reversed.', t_s: null, end_s: null, part: 2, severity: 'high',
   evidence: 'At 423.36 s the left stream sees the opposite gripper.'}]});
check(html.includes('2 reports, as each part wrote them'), 'the fold says how many reports it holds');
check(html.includes('Left and right &lt;b&gt; appear reversed.'), 'each report is escaped');
check(html.includes('data-t="0"') && html.includes('0.0s to 438.3s') && html.includes('part 1'),
  'a timed report seeks to its own time and names its part');
check(html.includes('untimed') && (html.match(/data-t=/g) || []).length === 1, 'an untimed report does not seek');
check(html.includes('At 423.36 s the left stream sees the opposite gripper.'), 'every report keeps its evidence');
process.exit(bad ? 1 : 0);
