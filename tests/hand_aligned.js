// The line the page shows when the hand keypoints were laid on the clip a frame or two off (board/hands.py
// align_frames, the drawing's "aligned"): board/serve.py hpAlignedText, and hpDecode carrying it.
//
//   node tests/hand_aligned.js [PAGE_SOURCE]     (default board/serve.py)
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
const T = new Function(piece('hpAlignedText') + 'return {hpAlignedText};')();

let bad = 0;
const check = (ok, what) => { if (!ok) { bad++; console.log('FAIL: ' + what); } };

check(T.hpAlignedText(null) === '' && T.hpAlignedText(undefined) === '', 'nothing when they line up');
const short = T.hpAlignedText({keypoint_frames: 21, clip_frames: 23});
check(short.includes('21 frames') && short.includes('23') && short.includes('last 2 frames have no hand'),
  'fewer keypoint frames: the clip\'s last frames have none');
const long = T.hpAlignedText({keypoint_frames: 24, clip_frames: 23});
check(long.includes('24 frames') && long.includes('last keypoint frame is left out'),
  'more keypoint frames: the extra one is left out');
check(/hpDecode[\s\S]*aligned: doc\.aligned/.test(piece('hpDecode')), 'the decoded drawing carries aligned');
process.exit(bad ? 1 : 0);
