// The board's Touch lane and contact card (the touch block of board/serve.py), held to what they must say: one bar per
// hand the contacts name, each contact styled by what the model found at it, the lane's counts, the grasps the model saw
// with no contact on their hand's row, the card's times and plain sentences, and the strength curve inside each bar.
//
//   node tests/touch_lane.js [PAGE_SOURCE]     (default board/serve.py)
//
// Prints nothing and exits 0 when every case holds; prints each failure and exits 1 otherwise.
'use strict';
const fs = require('fs');
const path = require('path');

const src = fs.readFileSync(process.argv[2] || path.join(__dirname, '..', 'board', 'serve.py'), 'utf8');
const a = src.indexOf('// ================= touch:'), b = src.indexOf('// each camera with depth: its switch');
if (a < 0 || b < a) { console.log('no touch block in the page'); process.exit(1); }
// esc and fmtT, the page's own (each a line or two, up to the end of fmtT's line)
const e0 = src.indexOf('\nfunction esc('), e1 = src.indexOf('\n', src.indexOf('\nfunction fmtT(') + 1);
if (e0 < 0 || e1 < e0) { console.log('no esc or fmtT in the page'); process.exit(1); }
const T = new Function(src.slice(e0, e1) + src.slice(a, b) + 'return {tcState, tcData, touchLaneHtml, '
  + 'tcCardsHtml, tcCardHtml, tcRegions, tcCurve};')();

let bad = 0;
const check = (ok, what) => { if (!ok) { bad++; console.log('FAIL: ' + what); } };
const lanePct = t => 10 * t;                    // a 10 s timeline
const chev = () => '<svg></svg>';
const count = (s, re) => (s.match(re) || []).length;
const contact = (id, hand, s0, s1, extra = {}) => ({id, hand, signals: ['right_pressure'], start_s: s0, peak_s: (s0 + s1) / 2,
  end_s: s1, from_start: false, to_end: false, peak_strength: 2, regions: {}, dips_s: [], shown: true, ...extra});
const seen = (touch, more = {}) => ({seen: {touch_seen: touch, hand: 'right', object: 'mug', grip: 'pinch',
  action: 'lifts it', slip: 'no', notes: null, ...more}});

// an episode with no contacts draws nothing new
check(T.touchLaneHtml(T.tcData({}), lanePct, chev) === '' && T.tcCardsHtml(T.tcData({})) === '',
  'an episode with no contacts has no lane and no card');

// what the model found, as a style
check(T.tcState(contact('c1', 'right', 0, 1, seen('yes'))) === 'yes', 'seen yes');
check(T.tcState(contact('c1', 'right', 0, 1, seen('NO'))) === 'no', 'seen no, in any case');
check(T.tcState(contact('c1', 'right', 0, 1, seen('unclear'))) === 'unclear', 'unclear');
check(T.tcState(contact('c1', 'right', 0, 1, {shown: false})) === 'unshown', 'not shown');
check(T.tcState(contact('c1', 'right', 0, 1)) === 'unanswered', 'shown with no answer');

// two hands, a contact of each kind, and a grasp the model saw on the left hand
const d = {contacts: [contact('c2', 'left', 3, 4, seen('no')), contact('c1', 'right', 1, 2, seen('yes')),
                      contact('c3', 'right', 5, 6, {shown: false}), contact('c4', 'right', 7, 8, seen('unclear'))],
           contacts_missing: [{t_s: '6.5', hand: 'left', object: 'napkin'}, {t_s: null, hand: 'left'}]};
const D = T.tcData(d);
check(D.contacts.map(c => c.id).join() === 'c1,c2,c3,c4', 'contacts in time order');
check(D.rows.join() === 'left,right', 'one row per hand, left first');
check(D.missing.length === 1 && D.missing[0].t_s === 6.5, 'a grasp with no time is left out');
const lane = T.touchLaneHtml(D, lanePct, chev);
check(lane.includes('4 contacts, 3 checked, 1 not seen, 1 grasp with no contact'), 'the lane counts: ' + lane.match(
  /lane-sum">([^<]*)/)[1]);
check(lane.includes('>Left hand<') && lane.includes('>Right hand<'), 'each row is named by its hand');
check(count(lane, /class="tc-seg /g) === 4 && count(lane, /class="tc-peak /g) === 4, 'a bar and a peak tick per contact');
check(lane.includes('tc-seg st-yes') && lane.includes('tc-seg st-no') && lane.includes('tc-seg st-unshown')
  && lane.includes('tc-seg st-unclear'), 'each contact styled by what the model found');
check(lane.includes('left:10%;width:max(3px, 10%)'), 'a bar runs from its begin to its end on the timeline');
const rows = lane.split('class="tc-row"').slice(1);
check(rows.length === 2 && rows[0].includes('tc-miss') && !rows[1].includes('tc-miss'), 'the grasp is on its hand\'s row');
check(rows[0].includes('data-c="1"') && !rows[0].includes('data-c="0"'), 'a contact is on its own hand\'s row');
check(lane.includes('id="lane-touch-pos">4 contacts<'), 'the stepper starts with the count');

// a grasp on a hand the contacts do not name gets that hand's row; a recording that names no hand has one row
check(T.tcData({contacts: [contact('c1', 'right', 0, 1)], contacts_missing: [{t_s: 2, hand: 'left'}]}).rows.join()
  === 'left,right', 'a hand only a grasp names gets a row');
const one = T.tcData({contacts: [contact('c1', null, 0, 1, seen('yes'))], contacts_missing: [{t_s: 2, hand: 'left'}]});
const oneLane = T.touchLaneHtml(one, lanePct, chev);
check(one.rows.join() === '' && !oneLane.includes('class="tc-hand"') && oneLane.includes('tc-miss'),
  'one unnamed row, without a hand name, carrying every grasp');

// the card
const cards = T.tcCardsHtml(D);
check(count(cards, /class="tc-card info-block/g) === 5 && cards.includes('tc-empty on'),
  'a card per contact and the one for no contact, which shows first');
const c0 = T.tcCardHtml(contact('c1', 'right', 0, 4.25, {...seen('yes', {hand: 'left', notes: 'null'}), from_start: true,
  dips_s: [1.24, 3]}), 0, 'yes');
check(c0.includes('Right hand, contact c1') && c0.includes('Already touching at the start')
  && c0.includes('ends <span data-t="4.25">4.3s</span>'), 'the card\'s hand and times');
check(c0.includes('The frames show the left hand, and the signal\'s name says right'), 'a hand the frames disagree on');
check(!c0.includes('>Notes<'), 'a null note is left out');
check(c0.includes('1.2s and 3.0s: its strength falls under half its peak and comes back'), 'the dips');
const c3 = T.tcCardHtml(contact('c3', 'right', 0, 1, {shown: false, to_end: true}), 2, 'unshown');
check(c3.includes('The model was not shown this contact') && c3.includes('still touching at the end')
  && !c3.includes('>Object<'), 'a contact the model was not shown says so');
check(T.tcRegions({regions: {p: {cells: 5, rows: [2, 2], columns: [0, 3], of: [16, 16]}, active_signals: ['p', 'q']}})
  .join('|') === 'p: 5 of its 256 cells, row 3 and columns 1 to 4 of 16 x 16|Active at its strongest: p and q',
  'where a contact bears, rows and columns counted from 1');
// a contact placed from both starts says on its card and its bar that its times are not recorded times; one on a
// recorded clock says nothing of it
const cp = T.tcCardHtml(contact('c5', 'right', 1, 2, {aligned_by: 'assumed start'}), 4, 'unanswered');
check(cp.includes('placed from both starts') && cp.includes('not recorded times'), 'a placed contact\'s card says so');
check(!c0.includes('placed from both starts'), 'a recorded contact\'s card says nothing of placing');
const pl = T.touchLaneHtml(T.tcData({contacts: [contact('c5', 'right', 1, 2, {aligned_by: 'assumed start'})]}), lanePct,
  chev);
check(pl.includes('2.0s placed from both starts:'), 'a placed contact\'s bar says so');
check(!lane.includes('placed from both starts'), 'a recorded contact\'s bar says nothing of placing');
check(!/[\u2013\u2014]/.test(lane + cards + c0 + c3 + cp + pl), 'no long dashes');

// the strength curve: the contact's signals summed per sample, on one scale for the episode
const Dsn = {t: Float64Array.from([0, 1, 2, 3, 4]), signals: [{name: 'right_pressure', str: Float32Array.from([0, 1, 2, 1, 0])},
  {name: 'other', str: Float32Array.from([5, 5, 5, 5, 5])}]};
const path_ = T.tcCurve(contact('c1', 'right', 1, 3), Dsn, 4);
check(path_ === 'M0.0 100L0.0 75.0L50.0 50.0L100.0 75.0L100.0 100Z', 'the curve: ' + path_);
check(T.tcCurve({...contact('c1', 'right', 1, 3), signals: ['missing']}, Dsn, 4) === '', 'no curve without its signals');

process.exit(bad ? 1 : 0);
