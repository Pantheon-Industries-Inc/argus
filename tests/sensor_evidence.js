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
  + 'return {sensorEvidence, activeSensorEvidence, sensorEvidenceHtml, sensorEvidenceOverlayHtml, setupSensorEvidence, '
  + 'sensorProfiles, sensorProfileAt, sensorProfileHtml, sensorPhasesHtml, sensorMedianTrace, sensorTraceHtml, sensorDistributionHtml};')(esc, fmtT);
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
const raw = {t:Float64Array.from([0,.1,.2,.3,.4,.5]), signals:[{name:'pressure', str:Float32Array.from([1,2,4,8,6,4]),act:Float32Array.from([1,2,4,8,6,4])}]};
const episode = {contacts:[{...c, peak_strength:8}], event_labels:[
  {t_s:0,end_s:.2,arm:'right',verb_class:'close grip'},
  {t_s:.2,end_s:.4,arm:'right',verb_class:'lift'},
  {t_s:.4,end_s:.6,arm:'left',verb_class:'other hand'}]};
const profiles = E.sensorProfiles(episode,raw);
assert.equal(profiles.length,1);
assert.equal(profiles[0].phases.length,2,'another hand cannot be attributed to this signal');
assert.equal(profiles[0].phases[0].percent,19);
assert.equal(profiles[0].phases[1].percent,75);
assert.equal(E.sensorProfileAt(profiles[0],.3).percent,100);
assert.equal(E.sensorProfileAt(profiles[0],-.1),null,'do not extrapolate before a recorded sample');
const quantitative = E.sensorProfileHtml(profiles[0],.3);
assert(quantitative.includes('100%') && quantitative.includes('episode peak'));
assert(!quantitative.includes('Right hand contact') && !quantitative.includes('with cup'),'do not repeat the action caption');
assert(!/force|newton|kg|stable grasp|slip recovered/i.test(quantitative),'relative intensity is not calibrated force');
const missingRaw = {...raw,signals:[{name:'pressure',str:Float32Array.from([1,NaN,4,8,6,4]),act:Float32Array.from([1,NaN,4,8,6,4])}]};
assert.equal(E.sensorProfileAt(E.sensorProfiles(episode,missingRaw)[0],.1),null,'missing readings cannot become zero intensity');
for (const availability of [
  {act:Float32Array.from([1,NaN,4,8,6,4])},
  {vals:{dims:1,v:Float32Array.from([1,NaN,4,8,6,4])}},
  {dims:1,map:Float32Array.from([1,NaN,4,8,6,4])}]) {
  const absent = {...raw,signals:[{name:'pressure',str:Float32Array.from([1,0,4,8,6,4]),...availability}]};
  assert.equal(E.sensorProfileAt(E.sensorProfiles(episode,absent)[0],.1),null,'the writer encodes missing strength as finite zero');
}
const unknown = {...raw,signals:[{name:'pressure',str:raw.signals[0].str}]};
assert.equal(E.sensorProfiles(episode,unknown).length,0,'strength alone does not establish reading availability');
const singleton = {t:[0],values:[8],peak:8};
assert.equal(E.sensorProfileAt(singleton,999),null,'a single reading cannot persist indefinitely');
const assumed = E.sensorProfiles({...episode,contacts:[{...c,aligned_by:'assumed start'}]},raw)[0];
assert.equal(assumed.phases.length,0,'assumed sensor placement cannot substantiate action medians');
assert(E.sensorProfileHtml(assumed,.3).includes('Assumed timing'));
for (const aligned_by of ['row per frame','coarse clock','unknown placement']) {
  assert.equal(E.sensorProfiles({...episode,contacts:[{...c,aligned_by}]},raw)[0].phases.length,0);
  assert.equal(E.sensorProfiles(episode,{...raw,signals:[{...raw.signals[0],aligned_by} ]})[0].phases.length,0);
}
assert.equal(E.sensorProfiles(episode,{...raw,signals:[{...raw.signals[0],camera_aligned_by:'assumed camera clock'}]})[0].phases.length,0);
const multiple = E.sensorProfiles({...episode,contacts:[...episode.contacts,{...c,signals:['other']}]},
  {...raw,signals:[...raw.signals,{...raw.signals[0],name:'other'}]});
assert.notEqual(E.sensorProfileHtml(multiple[0],.3),E.sensorProfileHtml(multiple[1],.3),'different sensor groups must be identifiable');
assert.notEqual(E.sensorPhasesHtml([multiple[0]]),E.sensorPhasesHtml([multiple[1]]));
const fineT = Array.from({length:12},(_,i)=>i/30);
const fine = {n:12,t:fineT,intensity:[1,1,1,4,1,1,1,NaN,2,2,2,2],active:Array(12).fill(4),
 focus:Array(12).fill(.5),row:Array(12).fill(.5),col:Array(12).fill(.5),shape:[2,2],top_cells:1,map:Array(48).fill(.25)};
const full = E.sensorProfiles(episode,{...raw,signals:[{...raw.signals[0],tactile:fine}]})[0];
assert.equal(full.t.length,12,'quantitative profiles must use full-rate samples');
assert.equal(full.values[3],4,'full-rate raw intensity retains a one-frame peak');
assert.equal(full.sustained[3],1,'a median distinguishes a transient from sustained load');
assert(Number.isNaN(full.sustained[7]),'smoothing cannot fill a gap');
assert(Number.isNaN(full.sustained[8]),'a new segment needs its own window');
assert(E.sensorProfileHtml(full,.1).includes('0.2s median'));
assert(E.sensorPhasesHtml([full]).includes('samples'));
assert(E.sensorTraceHtml(full).includes('Raw') && E.sensorTraceHtml(full).includes('0.2s median'));
assert(E.sensorDistributionHtml(full,.1).includes('Sensor grid'));
const unaligned = {...full,assumed:true,phases:[]};
assert(E.sensorPhasesHtml([unaligned]).includes('Full-rate raw tactile'),'the raw trace survives unavailable action timing');
assert(!E.sensorPhasesHtml([unaligned]).includes('data-phase-t'));
const regular = E.sensorMedianTrace([0,.1,.2,.3,.4,.5],[1,1,1,1,1,1]);
assert.equal(regular[4],1,'rounding must not create a gap on a regular 10Hz clock');
const gaps = E.sensorTraceHtml({...full,t:[0,.03,.5,.53],values:[1,1,1,1],sustained:[1,1,1,1]});
assert((gaps.match(/M/g)||[]).length >= 4,'both trace paths break across clock gaps');

assert(!/finger|newton|kilogram|stable grasp/.test(E.sensorDistributionHtml(full,.1)));
(async () => {
  fine.active[3] = 1;
  const panel = {now:{},overlay:{classList:{toggle(){}}},details:{}};
  global.document = {body:{contains(){return true;}},querySelectorAll(){return [];},
    getElementById(id){return ({'sensor-overlay':panel.overlay,'sensor-evidence-now':panel.now,'sensor-evidence-details':panel.details})[id] || null;}};
  global._activeFile = 'fine';
  global.loadSensors = () => Promise.resolve({...raw,signals:[{...raw.signals[0],tactile:fine}]});
  const live = E.setupSensorEvidence(E.sensorEvidence(episode),()=>{},()=>{},()=>{},episode,'fine');
  await Promise.resolve();
  live.sync(2/30); const before = panel.now.innerHTML;
  live.sync(3/30); const after = panel.now.innerHTML;
  assert(before.includes('4 / 4') && after.includes('1 / 4'),'distribution must refresh when rounded intensity is unchanged');
  delete global.document; delete global._activeFile; delete global.loadSensors;
  console.log('Sensor evidence controls passed');
})().catch(e => {console.error(e);process.exitCode=1;});
