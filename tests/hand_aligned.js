// The line the page shows when the hand keypoints were laid on the clip a frame or two off (board/hands.py
// align_frames, the drawing's "aligned"): board/serve.py hpAlignedText, and hpDecode carrying it.
//
//   node tests/hand_aligned.js [PAGE_SOURCE]     (default board/serve.py)
//
// Prints nothing and exits 0 when every case holds; prints each failure and exits 1 otherwise.
'use strict';
const {src, piece} = require('./page_functions')(process.argv[2]);
const alphabet = src.slice(src.indexOf('const HP_B64 = '), src.indexOf('const HP_COLOR = '));
const T = new Function(alphabet + piece('hpReader') + piece('hpDecode') + piece('hpAlignedText')
  + 'return {hpDecode, hpAlignedText};')();

let bad = 0;
const check = (ok, what) => { if (!ok) { bad++; console.log('FAIL: ' + what); } };

check(T.hpAlignedText(null) === '' && T.hpAlignedText(undefined) === '', 'nothing when they line up');
const short = T.hpAlignedText({keypoint_frames: 21, clip_frames: 23});
check(short.includes('21 frames') && short.includes('23') && short.includes('last 2 frames have no hand'),
  'fewer keypoint frames: the clip\'s last frames have none');
const long = T.hpAlignedText({keypoint_frames: 24, clip_frames: 23});
check(long.includes('24 frames') && long.includes('last keypoint frame is left out'),
  'more keypoint frames: the extra one is left out');
const alignment = {keypoint_frames: 21, clip_frames: 23};
const drawing = {clip: {frames: 1, w: 640, h: 480, pts: 'A', tb: [1, 30], dur: 1}, step: 1, aligned: alignment};
check(T.hpDecode(drawing).aligned === alignment, 'the decoded drawing carries alignment to the player');
check(T.hpDecode({...drawing, aligned: undefined}).aligned === null, 'a matched drawing needs no alignment warning');
check(T.hpDecode(drawing).times[0] === 0, 'alignment metadata does not move the clip clock');
process.exit(bad ? 1 : 0);
