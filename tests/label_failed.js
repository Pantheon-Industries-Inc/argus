// The episode page's block for a reply that gave no labels (board/serve.py cmpFailHtml): another model's, under
// Labels by, and the board's own (d._label_failed, board/to_board.py convert), which keeps the episode's footage,
// checks and sensors on the page and shows the reply as it came.
//
//   node tests/label_failed.js [PAGE_SOURCE]     (default board/serve.py)
//
// Prints nothing and exits 0 when every case holds; prints each failure and exits 1 otherwise.
'use strict';
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(process.argv[2] || path.join(__dirname, '..', 'board', 'serve.py'), 'utf8');
const piece = (name) => {
  const a = src.indexOf('\nfunction ' + name + '(');
  if (a < 0) { console.log('no ' + name + ' in the page'); process.exit(1); }
  // the function's body, by its braces (the page's template literals keep theirs balanced)
  let i = src.indexOf('{', a), depth = 0;
  for (; i < src.length; i++) {
    if (src[i] === '{') depth++;
    else if (src[i] === '}' && --depth === 0) break;
  }
  return src.slice(a, i + 1) + '\n';
};
const T = new Function(piece('esc') + piece('fmtDur') + piece('cmpFailHtml') + 'return {cmpFailHtml};')();

let bad = 0;
const check = (ok, what) => { if (!ok) { bad++; console.log('FAIL: ' + what); } };

const un = T.cmpFailHtml({status: 'unparsed', parse_error: 'JSONDecodeError', raw_head: '{<b>', raw_chars: 4}, 'Astra',
                         {est_cost_usd: 0.2}, true);
check(un.includes('did not parse') && un.includes('{&lt;b&gt;') && un.includes('JSONDecodeError'),
  'the board\'s own unparsed reply shows its error and its start, escaped');
check(un.includes('footage, checks and sensors') && !un.includes('comparison'),
  'the board\'s own failure says the rest of the episode is shown, not what a comparison counts');
const cut = T.cmpFailHtml({status: 'cut_off', out_tokens: 64000, tail: 'end'}, 'Astra', {}, true);
check(cut.includes('cut off') && cut.includes('64,000') && cut.includes('>end</pre>') && !cut.includes('comparison'),
  'the board\'s own cut-off reply shows its end');
const cmp = T.cmpFailHtml({status: 'unparsed', raw_head: 'x'}, 'Other', {}, false);
check(cmp.includes('The comparison counts it'), 'another model\'s failure keeps its comparison wording');
// an episode that got no reply, one whose reply the board could not read, and a part's own reply of a long recording
const nr = T.cmpFailHtml({status: 'no_reply', why: 'spend cap $20.00 reached'}, 'Astra', {}, true);
check(nr.includes('No response from Astra') && nr.includes('spend cap $20.00 reached')
  && nr.includes('footage, checks and sensors'), 'an episode with no reply says why and that the rest is shown');
const ns = T.cmpFailHtml({status: 'not_shown', error: 'KeyError: x', raw_head: '{"a": <1>}', raw_chars: 10}, 'Astra',
                         {}, true);
check(ns.includes('could not be shown') && ns.includes('KeyError: x') && ns.includes('{&quot;a&quot;: &lt;1&gt;}'),
  'a reply the board could not read shows the error and the reply, escaped');
const np = T.cmpFailHtml({status: 'no_part', parts: [{part: 1, t0_s: 0, t1_s: 400, why: 'did not parse',
                                                       raw_head: 'oops'}]}, 'Astra', {}, true);
check(np.includes('Start of part 1') && np.includes('>oops</pre>'), 'a failed part shows the start of its own reply');
process.exit(bad ? 1 : 0);
