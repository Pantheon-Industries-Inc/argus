// The recording checks card's rows for the problems an episode was kept and flagged with (board/serve.py
// readerIssueRows, from board/build.py reader_issues): one row per entry in the sentence it carries, escaped, under the
// name of the family it raises, with its time and signal when it has them; an entry with no sentence draws nothing.
// One whose family is not a fault in the recording (board/families.py COUNTED_LISTS) is marked as not counted.
//
//   node tests/reader_issues.js [PAGE_SOURCE]     (default board/serve.py)
//
// Prints nothing and exits 0 when every case holds; prints each failure and exits 1 otherwise.
'use strict';
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(process.argv[2] || path.join(__dirname, '..', 'board', 'serve.py'), 'utf8');
const e0 = src.indexOf('\nfunction esc('), e1 = src.indexOf('\nfunction isNullish(');
const t0 = src.indexOf('\nfunction fmtT('), t1 = src.indexOf('\n', t0 + 1);
const r0 = src.indexOf('\nfunction readerIssueRows(');
if (e0 < 0 || e1 < e0 || t0 < 0 || r0 < 0) {
  console.log('no esc, fmtT or readerIssueRows in the page');
  process.exit(1);
}
const r1 = src.indexOf('\n}\n', r0) + 3;
const NAMES = {'clip-frames': 'Camera video has fewer frames than the episode',
               'camera-undecodable': 'Camera video does not decode', 'label-failed': 'Model reply gave no labels'};
const LISTS = {'label-failed': 'labelling'};
const T = new Function('famName', 'famList', src.slice(e0, e1) + src.slice(t0, t1) + src.slice(r0, r1)
  + 'return {readerIssueRows};')(s => NAMES[s] || String(s).slice(2), s => LISTS[s] || 'data');

let bad = 0;
const check = (ok, what) => { if (!ok) { bad++; console.log('FAIL: ' + what); } };

check(T.readerIssueRows({}).length === 0 && T.readerIssueRows({dataset_checks: {reader_issues: 'x'}}).length === 0,
  'no entries draw nothing');
const rows = T.readerIssueRows({dataset_checks: {reader_issues: [
  {kind: 'camera_not_decodable', camera: 'exo', family: 'camera-undecodable',
   what: 'The main camera video could not be decoded, so this episode is shown and labelled without it.'},
  {kind: 'clip_frame_count', camera: 'left', family: 'clip-frames', what: 'The left <b> video has 29 frames.'},
  {kind: 'signal_gap', signal: 'force', what: 'The force signal stops for 2 s.', t0_s: 3.25},
  {kind: 'empty', what: '  '}, null]}});
check(rows.length === 3, 'one row per entry with a sentence');
check(rows[0].includes('<div class="di-issue">The main camera video could not be decoded, so this episode is shown '
  + 'and labelled without it.</div>') && rows[0].includes('<span class="di-cat">Camera video does not decode</span>'),
  'the sentence, then the family name');
check(rows[1].includes('The left &lt;b&gt; video has 29 frames.') && rows[1].includes('Camera video has fewer frames'),
  'the sentence escaped');
check(rows[2].includes('data-t="3.25"') && rows[2].includes('@ 3.3s') && rows[2].includes('>force</span>')
  && rows[2].includes('<span class="di-cat">signal gap</span>'), 'a kind with no family, its time and its signal');
check(!rows[0].includes('data-t='), 'no time, no seek');
// a problem of a family that is not a fault in the recording (a reply that gave no labels) is shown, marked as not
// counted, never in the style of a counted fault
const [lf] = T.readerIssueRows({dataset_checks: {reader_issues: [{kind: 'model_reply_unparsed', family: 'label-failed',
  what: 'The model\'s reply did not parse as JSON.'}]}});
check(lf && lf.includes('di-row low minor') && lf.includes('>not counted<') && lf.includes('Model reply gave no labels'),
  'a problem that is not a fault in the recording is shown as not counted');
check(rows[0].includes('di-row high') && rows[0].includes('>check<'), 'a fault in the recording is shown as before');
process.exit(bad ? 1 : 0);
