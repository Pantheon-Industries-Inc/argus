// The board's provenance line (board/serve.py readerNotesHtml): the reader's note on the recorded state and what the
// model was not shown of the upload by kind, each item escaped; nothing for an episode the reader read whole.
//
//   node tests/reader_notes.js [PAGE_SOURCE]     (default board/serve.py)
//
// Prints nothing and exits 0 when every case holds; prints each failure and exits 1 otherwise.
'use strict';
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(process.argv[2] || path.join(__dirname, '..', 'board', 'serve.py'), 'utf8');
const e0 = src.indexOf('\nfunction esc('), e1 = src.indexOf('\nfunction isNullish(');
const r0 = src.indexOf('\nfunction readerNotesHtml(');
if (e0 < 0 || e1 < e0 || r0 < 0) { console.log('no esc or readerNotesHtml in the page'); process.exit(1); }
const r1 = src.indexOf('\n}\n', r0) + 3;
const T = new Function(src.slice(e0, e1) + src.slice(r0, r1) + 'return {readerNotesHtml};')();

let bad = 0;
const check = (ok, what) => { if (!ok) { bad++; console.log('FAIL: ' + what); } };
check(T.readerNotesHtml(null) === '' && T.readerNotesHtml(undefined) === '', 'no notes draw nothing');
const h = T.readerNotesHtml({state_note: 'Labelled from the video: the recorded state has 16 values per frame.',
  left_out: {cameras: ['observation.images.cam_high_mask'], signals: ['recorder_time_ns (a clock) <x>']}});
check(h.startsWith('Labelled from the video: the recorded state has 16 values per frame.'), 'the note first, as written');
check(h.includes('The model was not shown the cameras observation.images.cam_high_mask; the signals '),
  'each kind named, cameras first');
check(h.includes('recorder_time_ns (a clock) &lt;x&gt;'), 'each item escaped');
check(!h.includes('The reader left out'), 'the line never says the reader left anything out');
check(T.readerNotesHtml({left_out: {arrays: ['a (1 x 2)', 'b']}}) === 'The model was not shown the arrays a (1 x 2), b.',
  'a list alone, without a note');
process.exit(bad ? 1 : 0);
