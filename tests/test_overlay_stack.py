"""Protect sensor placement and timing without moving existing overlays."""
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).parent
SOURCE = HERE.parent / 'board/serve.py'
pytestmark = pytest.mark.skipif(not shutil.which('node'), reason='no node')


def test_overlapping_findings_keep_active_claim_and_bound_retained_event():
    script = r'''
const assert = require('assert');
const {src} = require(process.argv[1])();
const begin = src.indexOf('// ================= sensor evidence:');
const end = src.indexOf('// ================= touch:', begin);
const api = new Function('esc', 'fmtT', src.slice(begin,end)
  + 'return {sensorEvidence,displayGripFindings,gripFindingStripHtml};')(String,String);
const findings = JSON.parse(process.argv[2]).findings;
const E = api.sensorEvidence({sensor_evidence:{version:1,findings}});
assert.deepEqual(api.displayGripFindings(E,.9),[], 'do not announce future observations');
let shown = api.displayGripFindings(E,2.46244);
assert.deepEqual(shown.map(f=>[f.findingIndex,f.phase]),[[1,'Now'],[0,'Now']]);
shown = api.displayGripFindings(E,2.76247);
assert.deepEqual(shown.map(f=>[f.findingIndex,f.phase]),[[0,'Now'],[1,'Earlier']],
  'ended sound transient cannot displace continuing hold evidence');
assert.equal(shown[1].end,2.595846,'retention must not extend the claimed observation interval');
assert(api.gripFindingStripHtml(shown[1]).includes('2.595846'));
assert.deepEqual(api.displayGripFindings(E,6),[], 'ended findings do not persist indefinitely');
assert.deepEqual(api.displayGripFindings(E,.9),[], 'seeking backwards clears retained findings');
const rejected = api.sensorEvidence({sensor_evidence:{version:1,findings:
  findings.map(f=>({...f,review_status:'rejected'}))}});
assert.deepEqual(api.displayGripFindings(rejected,2.46244),[], 'human rejection stays authoritative');
// Equal inspection times still refer to different findings and must remain selectable.
const strip={innerHTML:'',hidden:true,querySelectorAll:()=>[]};
const buttons=[0,1].map(i=>({dataset:{gripFinding:String(i),evidenceT:'2.46244',evidencePause:'true'},
  setAttribute(k,v){this[k]=v;}}));
const callbacks=new Map();
const doc={getElementById:id=>id==='grip-finding-overlay'?strip:null,
  querySelectorAll:q=>['[data-evidence-t]','[data-grip-finding]'].includes(q)?buttons:[]};
const interactive=new Function('esc','fmtT','document',src.slice(begin,end)
  +'return {sensorEvidence,setupSensorEvidence};')(String,String,doc);
const pair=interactive.sensorEvidence({sensor_evidence:{version:1,findings:
  findings.map(f=>({...f,inspect_s:2.46244}))}});
let placements=0,seeks=0;
const wire=interactive.setupSensorEvidence(pair,()=>seeks++,(el,event,fn)=>callbacks.set(el,fn),()=>{},
  null,null,()=>placements++);
wire.sync(2.46244);assert.deepEqual(buttons.map(b=>b['aria-pressed']),['false','true']);
callbacks.get(buttons[0])();assert.deepEqual(buttons.map(b=>b['aria-pressed']),['true','false']);
callbacks.get(buttons[1])();assert.deepEqual(buttons.map(b=>b['aria-pressed']),['false','true']);
const previousSeeks=seeks,previousPlacements=placements;
callbacks.get(strip)({target:{closest:()=>({dataset:{overlayFinding:'0'}})}});
assert.deepEqual(buttons.map(b=>b['aria-pressed']),['true','false']);
assert.equal(seeks,previousSeeks,'inline selection keeps the paused playhead');
assert.equal(placements,previousPlacements+1,'changed paused finding schedules its new card geometry');
wire.sync(2.46244);assert.equal(placements,previousPlacements+1,'unchanged finding does not reschedule');
'''
    subprocess.run(['node', '-e', script, str(HERE / 'page_functions.js'),
                    (HERE / 'fixtures/overlapping_sensor_findings.json').read_text()], check=True)


def test_sensor_dock_clears_existing_overlays_without_changing_their_layout():
    script = r'''
const fs=require('fs'),assert=require('assert');
const s=fs.readFileSync(process.argv[1],'utf8');
const start=s.indexOf('  function placeSensors() {'),end=s.indexOf('  let topPlacementFrame',start);
const rect=(left,top,width,height)=>({left,top,right:left+width,bottom:top+height,width,height});
const readonly=()=>new Proxy({}, {set(){throw Error('sensor placement changed an existing overlay');}});
const element=(bounds,active=true)=>({style:readonly(),classList:{contains:()=>active},
  closest:()=>null,getBoundingClientRect:()=>bounds});
const cr=rect(30,50,700,600),frame=rect(130,150,500,400);
const anchor=element(rect(142,158,140,28));
const recovery=element(rect(142,220,270,80));
const state=element(rect(480,230,210,110));
const caption=element(rect(160,470,380,36));
const depth=element(rect(650,190,40,26));depth.style={};
const host={style:readonly(),getBoundingClientRect:()=>cr,
  querySelector:q=>q==='#video-overlay'?caption:q==='.cam-dp:not([hidden])'?depth:null};
const stack={style:{},scrollHeight:100,hidden:false,
  classList:{remove(){},toggle(){}},getBoundingClientRect:()=>rect(frame.left+12,cr.top+(parseFloat(stack.style.top)||0),210,100)};
const page={getElementById:id=>id==='sensor-overlay-stack'?stack:null};
const computed=el=>({position:'absolute',getPropertyValue:k=>({
  '--fx-left':'100px','--fx-right':'100px','--fx-top':'100px','--fx-bottom':'100px'}[k]||'')});
const place=new Function('exoCell','topHud','progOverlay','stateToast','recOverlay','document',
  'getComputedStyle','let evidenceWire=null;'+s.slice(start,end)+'return placeSensors;')(
    host,null,anchor,state,recovery,page,computed);
const before=[anchor,recovery,state,caption].map(el=>({...el.getBoundingClientRect()}));
place();
assert.equal(parseFloat(stack.style.top),258,'dock clears recovery in the same horizontal lane');
assert.equal(parseFloat(stack.style.maxHeight),154,'dock stops before the existing caption');
assert.deepEqual([anchor,recovery,state,caption].map(el=>el.getBoundingClientRect()),before);
assert.equal(stack.style.minHeight,undefined);assert.equal(stack.style.height,undefined);
// A right-hand state card does not waste the left gap. Moving it into that lane requires clearance.
state.getBoundingClientRect=()=>rect(150,230,210,110);place();
assert.equal(parseFloat(stack.style.top),298,'dock also clears a state card that intersects its lane');
assert.equal(parseFloat(stack.style.maxHeight),114);
assert(parseFloat(stack.style.top)+Math.min(stack.scrollHeight,parseFloat(stack.style.maxHeight))
  <=caption.getBoundingClientRect().top-cr.top-8);
// If existing cards occupy the entire lane, a fictitious full-height gap would cover them.
recovery.getBoundingClientRect=()=>rect(142,130,270,500);place();
assert.equal(parseFloat(stack.style.maxHeight),0,'no free lane cannot display over the existing recovery');
'''
    subprocess.run(['node', '-e', script, str(SOURCE)], check=True)


def test_paused_sensor_dock_remeasures_letterbox_bounds_without_resizing_player():
    script = r'''
const fs=require('fs'),assert=require('assert');
const s=fs.readFileSync(process.argv[1],'utf8');
const layoutStart=s.indexOf('  function layoutFrame() {'),layoutEnd=s.indexOf('  if (vid) {',layoutStart);
const placementStart=s.indexOf('  function placeTop() {'),placementEnd=s.indexOf('  let _recSig',placementStart);
const properties={},pending=new Map();let frameId=0;
const host={style:new Proxy({setProperty(k,v){properties[k]=v;}},
    {set(){throw Error('sensor placement resized or moved the player');}}),
  getBoundingClientRect:()=>({top:0,left:0,right:700,bottom:600,height:600,width:700}),
  querySelector:()=>null};
const vid={paused:true,videoWidth:700,videoHeight:400,clientWidth:700,clientHeight:400,
  getBoundingClientRect:()=>({left:0,top:100})};
const stack={style:{},scrollHeight:100,hidden:false,classList:{remove(){},toggle(){}},
  getBoundingClientRect:()=>({left:12,right:222,width:210})};
const page={getElementById:id=>id==='sensor-overlay-stack'?stack:null};
const computed=el=>({position:'absolute',getPropertyValue:k=>properties[k]||'0px'});
const api=new Function('vid','exoCell','topHud','progOverlay','stateToast','recOverlay','fsBtn',
  'document','getComputedStyle','requestAnimationFrame','cancelAnimationFrame',
  'let evidenceWire=null;'+s.slice(layoutStart,layoutEnd)+s.slice(placementStart,placementEnd)
    +'return {layoutFrame,placeSensors};')(
    vid,host,null,null,null,null,null,page,computed,
    fn=>{pending.set(++frameId,fn);return frameId;},id=>pending.delete(id));
function settle(){let count=0;while(pending.size){assert(++count<5,'placement must settle');
  const [id,fn]=pending.entries().next().value;pending.delete(id);fn();}}
api.placeSensors();
// Late metadata changes image bounds while paused, with neither a progress chip nor a caption.
api.layoutFrame();settle();
assert.equal(properties['--fx-top'],'100.0px');assert.equal(properties['--fx-bottom'],'100.0px');
assert.equal(parseFloat(stack.style.top),108,'headline stays inside the actual image without a progress chip');
assert.equal(parseFloat(stack.style.maxHeight),348,'dock stops inside the image above native controls');
assert.equal(stack.style.minHeight,undefined);assert.equal(stack.style.height,undefined);
assert.equal(vid.paused,true);
const original={...host.getBoundingClientRect()};api.placeSensors();settle();
assert.deepEqual(host.getBoundingClientRect(),original,'sensor text never changes player geometry');
'''
    subprocess.run(['node', '-e', script, str(SOURCE)], check=True)


def test_anomalous_contact_updates_without_changing_numeric_profiles():
    script = r'''
const fs=require('fs'),assert=require('assert');
const s=fs.readFileSync(process.argv[1],'utf8');
const begin=s.indexOf('// ================= sensor evidence:'),end=s.indexOf('// ================= touch:',begin);
const overlay={innerHTML:'',classList:{active:false,toggle(k,v){this.active=v;}}};
const now={innerHTML:''};
const doc={getElementById(id){return id==='sensor-overlay'?overlay:id==='sensor-evidence-now'?now:null;},
  querySelectorAll(){return [];}};
const api=new Function('esc','fmtT','document',s.slice(begin,end)
  +'return {sensorEvidence,setupSensorEvidence};')(String,String,doc);
const E=api.sensorEvidence({contacts:[{id:'slip',start_s:1,end_s:2,shown:true,
  seen:{slip:'yes',touch_seen:'yes'}}]});
const wire=api.setupSensorEvidence(E,()=>{},()=>{},()=>{});
wire.sync(0);assert.equal(overlay.classList.active,false);
wire.sync(1.2);assert.equal(overlay.classList.active,true,
  'slip entry changes overlay despite empty stable profile HTML');
assert(overlay.innerHTML.includes('Slip reported'));
wire.sync(0);assert.equal(overlay.classList.active,false,'backward seek removes stale slip');
wire.sync(1.2);wire.sync(3);assert.equal(overlay.classList.active,false,'event exit removes slip');
const ordinary=api.sensorEvidence({contacts:[{id:'ordinary',start_s:1,end_s:2,shown:true,
  seen:{touch_seen:'yes'}}]});
api.setupSensorEvidence(ordinary,()=>{},()=>{},()=>{}).sync(1.2);
assert.equal(overlay.classList.active,false,'ordinary contact does not add a redundant warning');
'''
    subprocess.run(['node', '-e', script, str(SOURCE)], check=True)


def test_blocked_sensor_findings_wait_for_visible_media_time_without_collecting_seek_history():
    script = r'''
const fs=require('fs'),assert=require('assert');
const s=fs.readFileSync(process.argv[1],'utf8');
const begin=s.indexOf('// ================= sensor evidence:'),end=s.indexOf('// ================= touch:',begin);
const strip={innerHTML:'',hidden:true,querySelectorAll:()=>[]};
const doc={getElementById:id=>id==='grip-finding-overlay'?strip:null,querySelectorAll:()=>[]};
const api=new Function('esc','fmtT','document',s.slice(begin,end)
  +'return {sensorEvidence,setupSensorEvidence};')(String,t=>t.toFixed(1)+'s',doc);
const episode={sensor_evidence:{version:1,findings:[
  {start_s:1,end_s:1.13,headline:'Brief first signal',adds_beyond_video:true},
  {start_s:2,end_s:2.1,headline:'Brief second signal',adds_beyond_video:true},
  {start_s:3,end_s:3.1,headline:'Skipped on jump',adds_beyond_video:true}
]}};
const E=api.sensorEvidence(episode),original=JSON.stringify(E.insights);
const wire=api.setupSensorEvidence(E,()=>{},()=>{},()=>{});
wire.setCapacity(0);wire.sync(.9);wire.sync(1.2);wire.sync(2.2);wire.sync(7);
wire.setCapacity(200);
assert(!strip.hidden);assert(strip.innerHTML.includes('Brief second signal'));
assert(strip.innerHTML.includes('Earlier'));assert(strip.innerHTML.includes('2.0s to 2.1s'));
assert(strip.innerHTML.includes('aria-label="1 of 2 findings"'),'both encountered signals remain available');
assert(!strip.innerHTML.includes('Skipped on jump'),'a large seek does not queue intervening findings');
wire.sync(7.5);wire.sync(8);wire.setCapacity(0);
for(let t=8.5;t<=11;t+=.5)wire.sync(t);
wire.setCapacity(200);
assert(strip.innerHTML.includes('Brief second signal'),'blocked media time does not consume presentation');
for(let t=11.5;t<=13.5;t+=.5)wire.sync(t);
assert(strip.innerHTML.includes('Brief first signal'),'unselected deferred signal has not expired');
assert(!strip.innerHTML.includes('Brief second signal'),'presented signal expires after 3.5 visible media seconds');
wire.sync(0);assert(strip.hidden,'backward seek clears deferred findings');
wire.sync(7);assert(strip.hidden,'cleared queue does not collect old findings on a jump');
assert.equal(JSON.stringify(E.insights),original,'presentation never rewrites the recorded findings');
'''
    subprocess.run(['node', '-e', script, str(SOURCE)], check=True)


@pytest.mark.parametrize('first_sample', [1.05, 1.2], ids=['during-event', 'just-after-event'])
def test_paused_blocked_pending_card_keeps_measurement_until_readable_space_returns(first_sample):
    script = r'''
const fs=require('fs'),assert=require('assert');
const s=fs.readFileSync(process.argv[1],'utf8');
const begin=s.indexOf('// ================= sensor evidence:'),end=s.indexOf('// ================= touch:',begin);
const placementStart=s.indexOf('  function placeSensors() {'),placementEnd=s.indexOf('  let _recSig',placementStart);
const pending=new Map();let frameId=0,active=true,wire;
const readonly=()=>new Proxy({}, {set(){throw Error('sensor changed existing layout');}});
const rect=(left,top,width,height)=>({left,top,right:left+width,bottom:top+height,width,height});
const strip={innerHTML:'',hidden:true,querySelectorAll:()=>[]};
const stack={style:{},get scrollHeight(){return strip.hidden?0:100;},
  classList:{remove(){},toggle(){}},getBoundingClientRect:()=>rect(12,0,210,100)};
const recovery={style:readonly(),classList:{contains:()=>active},closest:()=>null,
  getBoundingClientRect:()=>rect(12,46,270,534)};
const host={style:readonly(),getBoundingClientRect:()=>rect(0,0,700,600),querySelector:()=>null};
const doc={getElementById:id=>id==='grip-finding-overlay'?strip:id==='sensor-overlay-stack'?stack:null,
  querySelectorAll:()=>[]};
const api=new Function('esc','fmtT','document',s.slice(begin,end)
  +'return {sensorEvidence,setupSensorEvidence};')(String,t=>t.toFixed(1)+'s',doc);
const placement=new Function('exoCell','topHud','progOverlay','stateToast','recOverlay','document',
  'getComputedStyle','evidenceWire','requestAnimationFrame','cancelAnimationFrame',
  s.slice(placementStart,placementEnd)+'return {placeSensors,scheduleTopPlacement};')(
    host,null,null,null,recovery,doc,()=>({position:'absolute',opacity:'0',getPropertyValue:()=> '0px'}),
    {setCapacity:height=>wire.setCapacity(height)},
    fn=>{pending.set(++frameId,fn);return frameId;},id=>pending.delete(id));
const E=api.sensorEvidence({sensor_evidence:{version:1,findings:[
  {start_s:1,end_s:1.13,headline:'Brief blocked signal',adds_beyond_video:true}
]}});
wire=api.setupSensorEvidence(E,()=>{},()=>{},()=>{},null,null,placement.scheduleTopPlacement);
function settle(){let count=0;while(pending.size){assert(++count<=8,'paused card measurement must settle');
  const [id,fn]=pending.entries().next().value;pending.delete(id);fn();}}
wire.sync(.9);settle();wire.sync(Number(process.argv[2]));settle();
assert.equal(stack.style.visibility,'hidden','a 30px gap cannot present the 100px claim');
wire.sync(7);settle();
assert.equal(stack.style.visibility,'hidden');
assert.equal(stack.scrollHeight,100,'pending Earlier card keeps its measured height while hidden');
assert(strip.innerHTML.includes('Brief blocked signal'));assert(strip.innerHTML.includes('Earlier'));
assert(strip.innerHTML.includes('1.0s to 1.1s'),'deferred claim preserves the original interval');
active=false;placement.scheduleTopPlacement();settle();
assert.equal(stack.style.visibility,'','claim appears only after the existing card clears');
assert(!strip.hidden);assert(strip.innerHTML.includes('Earlier'));
for(let t=7.5;t<=10.5;t+=.5){wire.sync(t);settle();}
assert(strip.hidden,'claim expires after 3.5 actual visible media seconds');
assert.equal(stack.style.minHeight,undefined);assert.equal(stack.style.height,undefined);
'''
    subprocess.run(['node', '-e', script, str(SOURCE), str(first_sample)], check=True)
