// The board's provenance line (board/serve.py readerNotesHtml): the reader's note on the recorded state as text, then
// what the model was not shown of the upload as a closed fold (counts by kind, items one per line, each escaped);
// nothing for an episode the reader read whole.
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
const count = (h, what) => h.split(what).length - 1;
const note = 'Labelled from the video: the recorded state has 16 values per frame.';
const many = (n, p) => Array.from({length: n}, (_, i) => p + i);

check(T.readerNotesHtml(null) === '' && T.readerNotesHtml(undefined) === '', 'no notes draw nothing');
check(T.readerNotesHtml({}) === '' && T.readerNotesHtml({left_out: {}}) === '', 'an empty record draws nothing');

// the note as text, then the lists folded away, closed by default
const h = T.readerNotesHtml({state_note: note + ' <b>',
  left_out: {cameras: ['observation.images.cam_high_mask'], signals: ['recorder_time_ns (a clock) <x>']}});
check(h.startsWith('<div class="rn-note">Labelled from the video: the recorded state has 16 values per frame. &lt;b&gt;'
  + '</div>'), 'the note first, as text, escaped');
check(count(h, '<details class="rn-fold">') === 1 && !h.includes('<details class="rn-fold" open'), 'one fold, closed');
check(h.includes('<summary class="ck-more">The model was not shown 1 camera and 1 signal</summary>'),
  'the summary counts each kind, singular for one');
check(h.includes('<div class="rn-k">Cameras</div><div class="rn-i">observation.images.cam_high_mask</div>'),
  'a kind under its heading, cameras first');
check(h.includes('<div class="rn-k">Signals</div><div class="rn-i">recorder_time_ns (a clock) &lt;x&gt;</div>'),
  'each item escaped');
check(h.indexOf('Cameras</div>') < h.indexOf('Signals</div>'), 'cameras before signals');
check(!h.includes('The reader left out') && !h.includes('Arrays</div>'), 'no wording from before, no empty kind');

// plural rules and joining
const p = T.readerNotesHtml({left_out: {cameras: many(7, 'c'), signals: many(38, 's')}});
check(p.includes('>The model was not shown 7 cameras and 38 signals</summary>'), 'two kinds joined with and');
check(count(p, '<div class="rn-i">') === 45, 'one item per line, one element each');
const q = T.readerNotesHtml({left_out: {cameras: ['a'], signals: ['s', 't'], arrays: ['x', 'y', 'z'], depth: ['d']}});
check(q.includes('>The model was not shown 1 camera, 2 signals, 3 arrays and 1 depth stream</summary>'),
  'three or more kinds joined with commas and a final and');
check(q.includes('<div class="rn-k">Arrays</div>') && q.includes('<div class="rn-k">Depth streams</div>'),
  'arrays and depth streams have their headings');
check(T.readerNotesHtml({left_out: {depth: ['d', 'e']}}).includes('>The model was not shown 2 depth streams</summary>'),
  'depth streams plural');

// a note alone has no fold, a list alone has no note line
const n = T.readerNotesHtml({state_note: note});
check(n === '<div class="rn-note">' + note + '</div>', 'a note alone, no fold');
const l = T.readerNotesHtml({left_out: {arrays: ['a (1 x 2)', 'b']}});
check(!l.includes('rn-note') && l.includes('>The model was not shown 2 arrays</summary>')
  && l.includes('<div class="rn-i">a (1 x 2)</div><div class="rn-i">b</div>'), 'a list alone, without a note');
process.exit(bad ? 1 : 0);
