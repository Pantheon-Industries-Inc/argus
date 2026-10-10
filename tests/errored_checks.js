// The episode page's checks section (board/serve.py checksSection) for a check that stopped with an error: our own
// checks (an "error" in the stored result), the capture checks (status "errored") and the sensor checks (status
// "errored") each show the check as an error with the reason, among the checks shown before the full list, never as
// clear and never as not applicable. Contacts placed from both starts are named as not judged for timing.
//
//   node tests/errored_checks.js [PAGE_SOURCE]     (default board/serve.py)
//
// Prints nothing and exits 0 when every case holds; prints each failure and exits 1 otherwise.
'use strict';
const {src, piece} = require('./page_functions')(process.argv[2]);
const stubs = 'let CHECKS_OPEN = false; const famName = s => s;'
  + 'const OUR_CHECKS = [["stream_pairing", "crossed", "streams-crossed"],'
  + '["recorded_jumps", "flagged", "recorded-jump"]];'
  + 'const sentences = t => String(t);';
// the page's own asSentence, the helper every reason is worded through
const s0 = src.indexOf('const asSentence = '), s1 = src.indexOf('const sentences = ');
if (s0 < 0 || s1 < s0) { console.log('no asSentence in the page'); process.exit(1); }
const T = new Function(stubs + src.slice(s0, s1) + piece('esc') + piece('placementText') + piece('checksSection') + 'return {checksSection};')();

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

// contacts placed from both starts are named as not judged for timing, never as a check that ran clear or fired
const hp = T.checksSection({_rig: 'teleop_arms', dataset_checks: {contact_checks: {contacts: 2, checked: 2, notes: [],
  placed_from_both_starts: ['c1', 'c2']}}});
const placedRow = hp.slice(hp.indexOf('Contacts placed from both starts'));
check(placedRow.includes('>not applicable<') && hp.includes('Contacts c1 and c2 are placed from both starts')
  && hp.includes('not recorded times'), 'contacts placed from both starts are named as not judged for timing');
check(/0 of 3 noted/.test(hp), 'the contact checks\' line counts only the checks that ran: '
  + (hp.match(/\d+ of \d+ noted/) || [''])[0]);
// a note's reason is worded through the same sentence helper as an issue's, so a reason that came lowercase and with no
// full stop reads as a sentence
const hn = T.checksSection({_rig: 'teleop_arms', dataset_checks: {capture_qc: {checks: [
  {check: 'video_low_contrast', name: 'Low contrast', group: 'Video', status: 'fired', shown_as: 'note', events: 1,
   why: 'it was not run on every camera, as the check stopped with an error (camera right: TypeError: x)'}],
  flags: [], notes: [{check: 'video_low_contrast', evidence: 'Camera left is nearly uniform.'}]}}});
check(hn.includes('Camera left is nearly uniform. It was not run on every camera, as the check stopped with an error '
  + '(camera right: TypeError: x).'), 'a note\'s reason reads as a sentence');
// a sensor check not run on a signal names it; one run on no signal is shown as not applicable with why, never clear
// and never left out, and only the checks that ran are counted
const hs = T.checksSection({_rig: 'teleop_arms', dataset_checks: {sensor_checks: {notes: [], checks: [
  {check: 'pinned', name: 'Values pinned at the end of their range', status: 'clear',
   not_run_on: 'probe empty (it has no rows)'},
  {check: 'constant', name: 'Signal never changes', status: 'na', why: 'no signal could be checked: probe empty'},
  {check: 'depth_invalid', name: 'Depth pictures mostly without readings', status: 'na'}]}}});
check(hs.includes('Not run on probe empty (it has no rows).'), 'a sensor check names the signal it was not run on');
check(hs.slice(hs.indexOf('Signal never changes')).includes('>not applicable<')
  && hs.includes('no signal could be checked'), 'a sensor check run on no signal says why, as not applicable');
check(!hs.includes('Depth pictures mostly'), 'a sensor check with nothing to run on is left out');
check(/0 of 1 noted/.test(hs), 'the sensor checks\' line counts only the checks that ran');
for (const [field, words] of [['placed_within_stamp_intervals', 'stamp interval'],
  ['placed_row_per_frame', 'row per frame'], ['placed_with_unspecified_alignment', 'unspecified alignment assumption']]) {
  const html = T.checksSection({_rig: 'teleop_arms', dataset_checks: {contact_checks: {
    contacts: 1, checked: 1, notes: [], [field]: ['qualified']}}});
  check(html.includes(words) && html.includes('qualified') && html.includes('not applicable'),
    field + ' retains its qualification in the checks card');
  check(!html.includes('both starts'), field + ' does not invent a common start');
}
process.exit(bad ? 1 : 0);
