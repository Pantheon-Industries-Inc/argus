// The sensors panel's words for a signal (board/serve.py snWhat, snErrorsHtml): a signal the reader placed on the
// video from both starts, because no clock was shared ("aligned_by", prepare/formats.py mark_assumed), says so in its
// lane, and a signal the board could not draw is named with the reason, under the lanes it did draw.
//
//   node tests/sensor_lanes.js [PAGE_SOURCE]     (default board/serve.py)
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
const T = new Function(piece('esc') + piece('snWhat') + piece('snErrorsHtml') + 'return {snWhat, snErrorsHtml};')();

let bad = 0;
const check = (ok, what) => { if (!ok) { bad++; console.log('FAIL: ' + what); } };

check(T.snWhat({dims: 1, aligned_by: 'assumed start'}).includes('placed from both starts, as no clock is shared'),
  'a signal placed by an assumed start says so');
check(!T.snWhat({dims: 1}).includes('both starts'), 'any other signal says nothing of it');
check(T.snWhat({dims: 1, aligned_by: 'row per frame'}).includes('placed one row per frame, as it has as many rows as the video has frames')
  && !T.snWhat({dims: 1, aligned_by: 'row per frame'}).includes('both starts'), 'a table placed row by row says so');
check(T.snErrorsHtml([]) === '' && T.snErrorsHtml(undefined) === '', 'no error, nothing drawn');
const h = T.snErrorsHtml([{name: 'pressure <map>', error: 'ValueError: boom'}, {name: 'glove', error: 'KeyError: x'}]);
check(h.includes('pressure &lt;map&gt;') && h.includes('ValueError: boom') && h.includes('glove')
  && h.includes('could not be drawn'), 'each signal the board could not draw is named with the reason');
process.exit(bad ? 1 : 0);
