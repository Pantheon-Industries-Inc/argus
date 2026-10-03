// The episode page's checks section (board/serve.py checksSection) for a check that stopped with an error: our own
// checks (an "error" in the stored result), the capture checks (status "errored") and the sensor checks (status
// "errored") each show the check as an error with the reason, among the checks shown before the full list, never as
// clear and never as not applicable.
//
//   node tests/errored_checks.js [PAGE_SOURCE]     (default board/serve.py)
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
const stubs = 'let CHECKS_OPEN = false; const famName = s => s;'
  + 'const OUR_CHECKS = [["stream_pairing", "crossed", "streams-crossed"],'
  + '["recorded_jumps", "flagged", "recorded-jump"]];'
  + 'const sentences = t => String(t); const asSentence = t => String(t);';
const T = new Function(stubs + piece('esc') + piece('checksSection') + 'return {checksSection};')();

let bad = 0;
const check = (ok, what) => { if (!ok) { bad++; console.log('FAIL: ' + what); } };

const h = T.checksSection({_rig: 'teleop_arms', dataset_checks: {
  stream_pairing: {crossed: false},
  recorded_jumps: {error: 'ValueError: boom', flagged: false},
  capture_qc: {checks: [{check: 'video_frozen_run', name: 'Frozen picture', group: 'Video', status: 'clear'},
                        {check: 'gripper_sensor_bug', name: 'Gripper sensor bug', group: 'Grippers',
                         status: 'errored', why: 'the check stopped with an error (ValueError: boom)'}],
                flags: [], notes: []},
  sensor_checks: {checks: [{check: 'constant', name: 'Signal never changes', status: 'clear'},
                           {check: 'clock_offset', name: 'Sensor clocks apart', status: 'errored',
                            error: 'KeyError: offset_ms'}], notes: []}}});
const rowOf = name => {
  const i = h.indexOf('>' + name + '<');
  return i < 0 ? '' : h.slice(h.lastIndexOf('<div class="ck-row', i), h.indexOf('</div>', i));
};
check(rowOf('recorded-jump').includes('ck-row err') && rowOf('recorded-jump').includes('>error<')
  && h.includes('ValueError: boom'), 'our check that crashed is an error with its reason');
check(rowOf('streams-crossed').includes('ck-row clear'), 'the other check stays as it came out');
check(/0 of 2 fired, 1 error/.test(h), 'our checks\' line counts the check that stopped with an error');
const theirs = h.slice(h.indexOf('ck-theirs'), h.indexOf('ck-all'));
check(theirs.includes('Gripper sensor bug') && theirs.includes('>error<') && theirs.includes('stopped with an error'),
  'a capture check that crashed is shown before the full list, as an error with its reason');
check(/1 error/.test(theirs), 'the capture checks\' summary counts the errors');
check(rowOf('Sensor clocks apart').includes('ck-row err') && h.includes('KeyError: offset_ms'),
  'a sensor check that crashed is an error with its reason');
process.exit(bad ? 1 : 0);
