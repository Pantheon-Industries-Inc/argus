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
const T = new Function('const snNum = v => String(v);' + piece('esc') + piece('snWhat') + piece('snErrorsHtml')
  + piece('snStillHtml') + 'return {snWhat, snErrorsHtml, snStillHtml};')();

let bad = 0;
const check = (ok, what) => { if (!ok) { bad++; console.log('FAIL: ' + what); } };

check(T.snWhat({dims: 1, aligned_by: 'assumed start'}).includes('placed from both starts, as no clock is shared'),
  'a signal placed by an assumed start says so');
check(!T.snWhat({dims: 1}).includes('both starts'), 'any other signal says nothing of it');
check(T.snErrorsHtml([]) === '' && T.snErrorsHtml(undefined) === '', 'no error, nothing drawn');
const h = T.snErrorsHtml([{name: 'pressure <map>', error: 'ValueError: boom'}, {name: 'glove', error: 'KeyError: x'}]);
check(h.includes('pressure &lt;map&gt;') && h.includes('ValueError: boom') && h.includes('glove')
  && h.includes('could not be drawn'), 'each signal the board could not draw is named with the reason');
// a signal that never changes is listed as constant, and one with no reading at any frame as having no reading, never
// as constant (board/sensors.py signal_doc)
const st = T.snStillHtml({constant: [{name: 'health', dims: 1, value: [1]}], none: [{name: 'glove <l>', dims: 3}]});
check(st.includes('Constant through this episode: health (1).') && st.includes('No reading at any frame: glove &lt;l&gt; '
  + '(3 values).') && !/Constant[^<]*glove/.test(st), 'a signal with no reading is named as such, not as constant');
check(T.snStillHtml({constant: [], none: []}) === '', 'nothing still, nothing drawn');
process.exit(bad ? 1 : 0);
