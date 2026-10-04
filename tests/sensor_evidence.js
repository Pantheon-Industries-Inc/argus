'use strict';
const fs = require('fs');
const assert = require('assert');
const source = fs.readFileSync(process.argv[2], 'utf8');
const begin = source.indexOf('// ================= sensor evidence:');
const end = source.indexOf('// ================= touch:', begin);
assert(begin >= 0 && end > begin, 'sensor evidence presentation is missing');
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
const fmtT = t => t.toFixed(1) + 's';
const E = new Function('esc', 'fmtT', source.slice(begin, end)
  + 'return {sensorEvidence, activeSensorEvidence, sensorEvidenceHtml, sensorEvidenceOverlayHtml, setupSensorEvidence};')(esc, fmtT);
const c = {id:'c1', hand:'right', start_s:0, end_s:4, from_start:true, to_end:true, dips_s:[], shown:true,
  signals:['pressure'], seen:{touch_seen:'yes', hand:'right', object:'cup', action:'lift and lower cup', slip:'no'}};
let d = E.sensorEvidence({contacts:[c]});
assert.equal(d.moments.length, 0, 'clip boundaries must not become contact onsets or releases');
let active = E.activeSensorEvidence(d, 2);
assert.equal(active.visual, 'Touch seen in sampled frames');
assert.equal(active.detail, 'lift and lower cup');
assert(!/stable|no slip|pressure applied|kilogram/i.test(JSON.stringify(active)), 'do not invent physical conclusions');
assert.equal(E.activeSensorEvidence(d, 5), null, 'contact evidence ends with its recorded span');
d = E.sensorEvidence({contacts:[{...c, start_s:1, from_start:false, end_s:4, to_end:false, dips_s:[2]}]});
assert.deepEqual(d.moments.map(x => x.t), [1,2,4]);
active = E.activeSensorEvidence(d, 2);
assert.equal(active.headline, 'Contact signal weakens and returns');
assert(!/slip|regrasp/i.test(active.headline), 'a signal dip cannot identify a slip or regrasp');
assert.equal(E.activeSensorEvidence(d, 4.2).headline, 'Contact signal ends');
d = E.sensorEvidence({contacts:[{...c, aligned_by:'assumed start', seen:{...c.seen, touch_seen:'no'}}]});
active = E.activeSensorEvidence(d, 2);
assert.equal(active.reading, 'Assumed timing');
assert(!/disagree|without visible|mismatch/i.test(active.headline), 'assumed alignment does not prove disagreement');
d = E.sensorEvidence({contacts:[{...c, seen:{...c.seen, hand:'left'}}]});
assert.equal(E.activeSensorEvidence(d, 2).headline, 'Hand attribution disagrees');
d = E.sensorEvidence({contacts:[{...c, shown:false}]});
assert.equal(E.activeSensorEvidence(d, 2).visual, 'Not visually checked');
assert(!E.activeSensorEvidence(d,2).detail, 'unshown contacts cannot reuse model interpretation');
d = E.sensorEvidence({contacts_missing:[{t_s:3, hand:'left', object:'cup'},{t_s:null}]});
assert.equal(d.moments.length, 1);
assert.equal(E.activeSensorEvidence(d,3).headline, 'Visible grasp has no recorded contact');
const depth = E.sensorEvidenceHtml(E.sensorEvidence({}), ['exo']);
assert(depth.includes('No depth-specific finding in the saved annotation'));
assert(depth.includes('Compare RGB and depth'));
assert.equal(E.sensorEvidenceHtml(E.sensorEvidence({}), []), '', 'ordinary video-only pages remain unchanged');
d = E.sensorEvidence({contacts:[{...c, seen:{...c.seen, object:'<img src=x>', action:'<script>alert(1)</script>'}}]});
const html = E.sensorEvidenceOverlayHtml(E.activeSensorEvidence(d,2));
assert(html.includes('&lt;img') && !html.includes('<script>'), 'model text is escaped');
assert(!/[\u2013\u2014]/.test(html + depth));
const buttons = {inspect:{}, overlay:{classList:{toggle(){}}}, now:{}, details:{scrollIntoView(){}}};
global.document = {
  getElementById(id) { return ({'sensor-overlay':buttons.overlay, 'sensor-evidence-now':buttons.now,
    'sensor-evidence-details':buttons.details})[id] || null; },
  querySelectorAll(selector) { return selector.includes('sensor-evidence-inspect') ? [buttons.inspect, buttons.overlay] : []; }
};
const listeners = new Map();
let inspected = -1;
d = E.sensorEvidence({contacts:[{...c, seen:{...c.seen, slip:'yes'}}, {...c, id:'c2', start_s:1}]});
const wire = E.setupSensorEvidence(d, ()=>{}, (el, type, fn)=>listeners.set(el,fn), i=>{inspected=i;});
wire.sync(2); listeners.get(buttons.inspect)();
assert.equal(inspected,0, 'Inspect selects the displayed high-priority contact, not the latest overlapping contact');
assert(buttons.details.open, 'Inspect opens the supporting evidence');
assert.equal(E.activeSensorEvidence(d,2).headline, 'Slip reported in this contact');
delete global.document;
console.log('Sensor evidence controls passed');
