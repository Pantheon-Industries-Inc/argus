"""Protect sensor placement and timing without moving existing overlays."""
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).parent
SOURCE = HERE.parent / 'board/index.html'
pytestmark = pytest.mark.skipif(not shutil.which('node'), reason='no node')


# A small DOM boundary for exercising the real timeline with retained node identities.
CARD_DOM = r'''
const cardDocument = () => {
 const children=[];
 const strip={hidden:true,append(card){children.push(card);},
  get innerHTML(){return children.map(c=>`<div data-sensor-finding="${c.dataset.sensorFinding}">${c.innerHTML}</div>`).join('');},
  querySelectorAll(){return [...children];},
  querySelector(selector){const id=selector.match(/="(\d+)"/)[1];return children.find(c=>c.dataset.sensorFinding===id)||null;}};
 const createElement=()=>{
  const phase={textContent:'',classList:{toggle(){}}};
  return {dataset:{},innerHTML:'',querySelector:()=>phase,querySelectorAll:()=>[],
   remove(){children.splice(children.indexOf(this),1);}};
 };
 return {strip,createElement};
};
'''


def test_new_onsets_replace_older_findings_without_changing_source_intervals():
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
assert.deepEqual(shown.map(f=>[f.findingIndex,f.phase]),[[1,'Now'],[0,'Now']],
  'the hold remains until its two-second minimum while the transient is introduced');
assert.equal(shown[0].end,2.595846,'presentation does not change the source interval');
assert(api.gripFindingStripHtml(shown[0]).includes('2.595846'));
shown = api.displayGripFindings(E,2.76247);
assert.deepEqual(shown.map(f=>[f.findingIndex,f.phase]),[[1,'Earlier'],[0,'Now']]);
assert.deepEqual(api.displayGripFindings(E,3).map(f=>f.findingIndex),[1],
  'the superseded hold cannot return after its minimum');
assert.deepEqual(api.displayGripFindings(E,4.4),[], 'the transient expires two seconds after its onset');
assert.deepEqual(api.displayGripFindings(E,6),[], 'ended findings do not persist indefinitely');
assert.deepEqual(api.displayGripFindings(E,.9),[], 'seeking backwards clears retained findings');
const rejected = api.sensorEvidence({sensor_evidence:{version:1,findings:
  findings.map(f=>({...f,review_status:'rejected'}))}});
assert.deepEqual(api.displayGripFindings(rejected,2.46244),[], 'human rejection stays authoritative');
// Equal inspection times still refer to different findings and must remain selectable.
const {strip,createElement}=cardDocument();
const buttons=[0,1].map(i=>({dataset:{gripFinding:String(i),evidenceT:'2.46244',evidencePause:'true'},
  setAttribute(k,v){this[k]=v;}}));
const callbacks=new Map();
const doc={createElement,getElementById:id=>id==='grip-finding-overlay'?strip:null,
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
callbacks.get(strip)({target:{closest:()=>({dataset:{sensorFinding:'1'}})}});
assert.deepEqual(buttons.map(b=>b['aria-pressed']),['false','true']);
assert.equal(seeks,previousSeeks+1,'each visible card replays its own finding');
assert.equal(placements,previousPlacements,'replaying an already visible card does not rebuild it');
wire.sync(2.46244);assert.equal(placements,previousPlacements,'unchanged findings do not reschedule');
// Replaying the earlier finding from later playback must preserve explicit selection.
wire.sync(7);
callbacks.get(buttons[0])();assert.deepEqual(buttons.map(b=>b['aria-pressed']),['true','false']);
assert(strip.innerHTML.includes(findings[0].headline),'backward replay selects the clicked finding');
'''
    subprocess.run(['node', '-e', CARD_DOM + script, str(HERE / 'page_functions.js'),
                    (HERE / 'fixtures/overlapping_sensor_findings.json').read_text()], check=True)


def test_sensor_cards_use_clear_image_gaps_and_attach_when_existing_overlays_fill_them():
    script = r"""
const fs=require('fs'),assert=require('assert');
const s=fs.readFileSync(process.argv[1],'utf8');
const start=s.indexOf('  function placeSensors() {'),end=s.indexOf('  let topPlacementFrame',start);
const rect=(left,top,width,height)=>({left,top,right:left+width,bottom:top+height,width,height});
const readonly=()=>new Proxy({}, {set(){throw Error('sensor placement changed an existing overlay');}});
const element=(bounds)=>({style:readonly(),classList:{contains:()=>true},closest:()=>null,getBoundingClientRect:()=>bounds});
const cr=rect(30,50,700,600),anchor=element(rect(142,158,140,28));
const recovery=element(rect(142,220,270,80)),state=element(rect(480,230,210,110));
const caption=element(rect(160,470,380,36));
const host={style:readonly(),getBoundingClientRect:()=>cr,querySelector:q=>q==='#video-overlay'?caption:null};
const cards=[0,1].map(()=>({style:{},getBoundingClientRect:()=>rect(0,0,210,100)}));
const stack={style:{},dataset:{},classList:{toggle(){}},querySelectorAll:()=>cards};
const band={style:{},hidden:true,getBoundingClientRect:()=>rect(cr.left,cr.bottom,cr.width,0)};
const page={getElementById:id=>id==='sensor-overlay-stack'?stack:id==='sensor-evidence-band'?band:null};
const computed=()=>({position:'absolute',opacity:'1',getPropertyValue:k=>({
 '--fx-left':'100px','--fx-right':'100px','--fx-top':'100px','--fx-bottom':'100px'}[k]||'')});
const presentation={};
const place=new Function('d','exoCell','topHud','progOverlay','stateToast','recOverlay','document','getComputedStyle',
 s.slice(start,end)+'return placeSensors;')(presentation,host,null,anchor,state,recovery,page,computed);
const before=[anchor,recovery,state,caption].map(el=>({...el.getBoundingClientRect()}));
place();
assert.equal(stack.dataset.layout,'overlay');assert(band.hidden);
const positions=cards.map(c=>rect(cr.left+parseFloat(c.style.left),cr.top+parseFloat(c.style.top),210,100));
assert.deepEqual(positions.map(p=>[p.left,p.top]),[[142,308],[408,348]],'both findings receive separate unobstructed regions');
for(const p of positions) for(const r of before) assert(p.right<=r.left||p.left>=r.right||p.bottom<=r.top||p.top>=r.bottom);
presentation.viewer_presentation={sensor_overlay_side:'right'};
for(const card of cards) card.style.left='';
place();assert(parseFloat(cards[0].style.left)>parseFloat(cards[1].style.left),'a right-side preference changes sensor placement without moving existing overlays');
recovery.getBoundingClientRect=()=>rect(142,130,500,500);
place();
assert.equal(stack.dataset.layout,'attached');assert.equal(band.hidden,false);
assert(cards.every(c=>parseFloat(c.style.top)>=cr.height),'a full recovery cannot hide a finding or put it over the footage');
assert(parseFloat(band.style.height)>=100,'the attached strip reserves real space for its cards');
assert.deepEqual([anchor,state,caption].map(el=>el.getBoundingClientRect()),[before[0],before[2],before[3]]);
assert.deepEqual(host.getBoundingClientRect(),cr,'findings do not resize the camera');
"""
    subprocess.run(['node', '-e', CARD_DOM + script, str(SOURCE)], check=True)


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
    subprocess.run(['node', '-e', CARD_DOM + script, str(SOURCE)], check=True)


def test_concurrent_recorded_findings_and_quality_warning_are_all_rendered_without_a_carousel():
    script = r"""
const fs=require('fs'),assert=require('assert');
const src=fs.readFileSync(process.argv[1],'utf8');
const begin=src.indexOf('// ================= sensor evidence:'),end=src.indexOf('// ================= touch:',begin);
const {strip,createElement}=cardDocument(),detail={innerHTML:''};
const doc={createElement,getElementById:id=>id==='grip-finding-overlay'?strip:id==='sensor-evidence-now'?detail:null,querySelectorAll:()=>[]};
const api=new Function('esc','fmtT','document',src.slice(begin,end)
 +'return {sensorEvidence,setupSensorEvidence,displayGripFindings,gripFindingDetailHtml,gripFindingStripHtml};')(String,t=>t.toFixed(1)+'s',doc);
const findings=JSON.parse(fs.readFileSync(process.argv[2],'utf8')).findings;
const verbose={...findings[0],claim:'The repeated excursions are consistent with changing contact or loading during the held-object motion. Unknown response polarity and occluded contact geometry prevent identifying tighter grip, looser grip, or a surface tap.'};
const projected=api.sensorEvidence({sensor_evidence:{version:1,findings:[verbose]}}).insights[0];
assert(!api.gripFindingStripHtml(projected).includes(verbose.claim),'physical interpretation is not an automatic video caption');
assert(api.gripFindingDetailHtml(projected).includes(verbose.claim),'the complete interpretation remains inspectable');
const concise='Two drops and rebounds occur while the block remains held.';
assert(api.gripFindingStripHtml({...projected,overlayDetail:concise}).includes(concise),'explicit overlay copy remains intact');
assert(!api.gripFindingStripHtml({...projected,overlayDetail:''}).includes('gf-detail'),'an explicitly empty overlay caption does not fall back to prose');

const E=api.sensorEvidence({sensor_evidence:{version:1,findings},tactile_qc:{version:1,warnings:[
 {start_s:2.4,end_s:2.6,headline:'Touch and video clocks disagree',detail:'Measured offset exceeds the review threshold.'},
 {start_s:2.4,end_s:2.6,headline:'Rejected warning',review_status:'rejected'}
]}});
const original=JSON.stringify(E),wire=api.setupSensorEvidence(E,()=>{},()=>{},()=>{},null,null,()=>{
 if(strip.innerHTML.includes('data-sensor-finding')) assert(!strip.hidden,'new cards must be measurable when placement runs');
});
wire.sync(1.1);assert(strip.innerHTML.includes(findings[0].headline));
const heldCard=strip.querySelector('[data-sensor-finding="0"]');
const heldMarkup=heldCard.innerHTML;
wire.sync(2.46244);
assert.equal(strip.querySelector('[data-sensor-finding="0"]'),heldCard,'introducing a new finding must retain the earlier DOM node');
assert.equal(heldCard.innerHTML,heldMarkup,'an accompanying card cannot restart the retained card or its plots');
assert(strip.innerHTML.includes(findings[0].headline),'the long hold gets its full two-second minimum');
assert(!strip.innerHTML.includes(findings[0].detail),'full reasoning stays in the evidence panel');
assert(strip.innerHTML.includes(findings[1].headline));
assert(strip.innerHTML.includes('Touch and video clocks disagree')&&strip.innerHTML.includes('Measured offset exceeds the review threshold.'));
assert.equal((strip.innerHTML.match(/data-sensor-finding=/g)||[]).length,3,'nearby findings each receive their minimum presentation time');
assert(!strip.innerHTML.includes('data-overlay-finding')&&!strip.innerHTML.includes('gf-pager'),'no unseen finding is behind a selector');
assert(!strip.innerHTML.includes('Rejected warning'));
const paused=strip.innerHTML;wire.sync(2.46244);assert.equal(strip.innerHTML,paused,'paused playback has no independent presentation clock');
wire.sync(2.76247);assert(strip.querySelectorAll().some(c=>c.querySelector('.gf-phase').textContent==='Earlier'),'retained findings are distinguished from source-active findings');
wire.sync(3.1);assert(!strip.innerHTML.includes(findings[0].headline),'the superseded hold stays retired');
wire.sync(4.1);assert.equal((strip.innerHTML.match(/data-sensor-finding=/g)||[]).length,2);
wire.sync(4.5);assert(strip.hidden,'minimum presentation time does not create unbounded persistence');
const stopped=strip.innerHTML;wire.sync(4.5);assert.equal(strip.innerHTML,stopped);
wire.reset(0);assert(strip.hidden,'replay clears previously displayed findings');
wire.reset(4.5);assert(strip.hidden,'a seek does not revive expired findings');
wire.sync(2.46244);const lastDetail=detail.innerHTML;wire.sync(20);assert(strip.hidden);
assert.equal(detail.innerHTML,lastDetail,'retiring an overlay does not reset its evidence');
assert.equal(JSON.stringify(E),original,'rendering never changes annotations');
// Real OpenTouch subsecond intervals each receive two seconds, with source timestamps intact.
const short=api.sensorEvidence({grip_evidence:{version:1,findings:[
 {start_s:1.79971,end_s:2.2663,headline:'Pressure shifts across the thumb',detail:'Compared with the start, one part of the thumb presses harder and a nearby part presses less.'},
 {start_s:2.13298,end_s:3.23281,headline:'Pressure shifts toward the index fingertip',detail:'Pressure shifts along the index finger toward the fingertip area.'}
]}});
const fast=api.setupSensorEvidence(short,()=>{},()=>{},()=>{});
fast.sync(1.9);assert(strip.innerHTML.includes(short.insights[0].detail));
fast.sync(2.2);assert.equal((strip.innerHTML.match(/data-sensor-finding=/g)||[]).length,2);
fast.sync(2.4);assert.equal((strip.innerHTML.match(/data-sensor-finding=/g)||[]).length,2);
assert(strip.querySelectorAll().some(c=>c.querySelector('.gf-phase').textContent==='Earlier'));
fast.sync(3.8);assert.equal((strip.innerHTML.match(/data-sensor-finding=/g)||[]).length,1);
assert(!strip.innerHTML.includes(short.insights[0].detail),'the earlier card retires at onset plus two seconds');
fast.sync(4.14);assert(strip.hidden);
// Long intervals yield to later onsets after their two-second minimum.
const timeline=api.sensorEvidence({grip_evidence:{version:1,findings:[
 {start_s:1.5,end_s:10,headline:'Hold continues',detail:'First long observation.'},
 {start_s:2.7,end_s:10,headline:'Pressure increases',detail:'Second long observation.'},
 {start_s:3,end_s:10,headline:'Contact shifts',detail:'Third long observation.'},
 {start_s:3,end_s:10,headline:'State changes',detail:'Same onset, separate evidence.'},
 {start_s:3.8,end_s:4.1,headline:'Brief release',detail:'A genuine subsecond observation.'}
]}});
const shownAt=t=>api.displayGripFindings(timeline,t).map(f=>f.headline);
assert.deepEqual(shownAt(2.69),['Hold continues']);
assert.deepEqual(shownAt(2.7),['Pressure increases','Hold continues']);
assert.deepEqual(shownAt(3.1),['Contact shifts','State changes','Pressure increases','Hold continues']);
assert.deepEqual(shownAt(3.5),['Contact shifts','State changes','Pressure increases']);
assert.deepEqual(shownAt(3.7),['Contact shifts','State changes','Pressure increases']);
assert.deepEqual(shownAt(3.85),['Brief release','Contact shifts','State changes','Pressure increases']);
assert.deepEqual(shownAt(4.05),['Brief release','Contact shifts','State changes','Pressure increases']);
assert.deepEqual(shownAt(4.7),['Brief release','Contact shifts','State changes']);
assert.deepEqual(shownAt(5.1),['Brief release']);
assert.deepEqual(shownAt(5.8),[],'the subsecond finding also receives two seconds');
assert.deepEqual(shownAt(7),[],'superseded long intervals never return');
fast.dispose();
wire.dispose();
"""
    subprocess.run(['node', '-e', CARD_DOM + script, str(SOURCE), str(HERE / 'fixtures/overlapping_sensor_findings.json')], check=True)


def test_static_camera_order_matches_retained_scene_presentation(tmp_path):
    from board import static
    import json
    qa = tmp_path / 'qa'
    qa.mkdir()
    label = {'_meta': {'episode_id': 'retained'}, 'dataset': 'controlled', 'episode_prompt': 'Controlled camera check',
             'camera_views': ['exo', 'left'], 'camera_labels': ['Touch', 'Wrist'],
             'camera_presentation': {'exo': {'role': 'tactile'}, 'unshown1': {'role': 'scene'}},
             'unshown_cameras': [{'view': 'unshown1', 'name': 'Scene', 'why': 'not selected'}],
             'completion': {'completed_at_s': 1, 'task_completed': 'success'}}
    (qa / 'retained.json').write_text(json.dumps(label))
    assert static.shown_cams(label) == ['unshown1', 'left', 'exo']
    (episode,) = static.plan(qa, tmp_path / 'clips')
    assert list(episode['media']) == ['unshown1', 'left', 'exo']
    assert episode['media']['unshown1']['main']
    assert 'unshown1|1000' in episode['frames']
    assert label['camera_views'] == ['exo', 'left']


def test_retained_scene_clip_size_keeps_original_model_query_clock(tmp_path):
    from board import clips
    import json
    import numpy as np
    episode = tmp_path / 'episode_a'
    episode.mkdir()
    sources = {'exo': {'packed': str(tmp_path / 'touch.mp4'), 'base_s': 0, 'n_frames': 3}}
    context = {'episode_id': 'episode_a', 'fps': 10, 'state_kind': 'none', 'real_times': 'times.npz',
        'cameras': {'exo': {'name': 'pad_camera', 'role': 'tactile_image'}},
        'unshown_cameras': [{'name': 'retained_camera', 'packed': str(tmp_path / 'scene.mp4'),
                            'n_frames': 4, 'fps': 10, 'start_s': .2}]}
    (episode / 'sources.json').write_text(json.dumps(sources))
    np.savez(episode / 'times.npz', exo=np.array([0, .11, .24]))
    (episode / 'context.json').write_text(json.dumps(context))
    unknown = {job[-1]: job for job in clips.episode_jobs(episode, tmp_path / 'clips', True)}
    assert unknown['exo'][5] is True and unknown['unshown1'][5] is False
    context['unshown_cameras'][0]['role'] = 'scene_camera'
    (episode / 'context.json').write_text(json.dumps(context))
    before = {name: (episode / name).read_bytes() for name in ('context.json', 'sources.json', 'times.npz')}
    qualified = {job[-1]: job for job in clips.episode_jobs(episode, tmp_path / 'clips', True)}
    assert qualified['exo'][5] is False and qualified['unshown1'][5] is True
    np.testing.assert_array_equal(qualified['exo'][9], unknown['exo'][9])
    np.testing.assert_array_equal(qualified['exo'][9], [0, .11, .24])
    assert qualified['unshown1'][6] == .2
    assert clips.main_cam(sources) == 'exo'
    assert {name: (episode / name).read_bytes() for name in before} == before


def test_dataset_and_license_urls_cannot_turn_into_executable_links():
    script = r'''
const assert=require('assert'),{piece}=require(process.argv[1])();
const api=new Function(piece('esc')+piece('safeExternalUrl')+piece('datasetSourceHtml')
 +'return {esc,safeExternalUrl,datasetSourceHtml};')();
assert.equal(api.esc("' onclick='x"),'&#39; onclick=&#39;x');
for(const url of ['javascript:alert(1)','JaVaScRiPt:alert(1)','java\nscript:alert(1)','data:text/html,<script>x</script>']) {
 assert.equal(api.safeExternalUrl(url),'');
 assert(!api.datasetSourceHtml({name:'Dataset',license:'Public',hub:url,license_url:url}).includes('href='));
}
const html=api.datasetSourceHtml({name:'Dataset',license:'Public',hub:'https://example.com/data?q=a&b=c'});
assert(html.includes('href="https://example.com/data?q=a&amp;b=c"'));
'''
    subprocess.run(['node', '-e', script, str(HERE / 'page_functions.js')], check=True)


def test_temporary_frame_load_failure_retries_only_live_local_frames():
    script = r'''
const assert=require('assert'),{piece}=require(process.argv[1])();
(async()=>{
 let attempts=0,revoked=0,clock=0;
 const now={now:()=>clock},wait=(fn,ms)=>{clock+=ms;fn();};
 const urlApi=class extends URL{};
 urlApi.createObjectURL=()=> 'blob:ready';urlApi.revokeObjectURL=()=>revoked++;
 const image={tagName:'IMG',isConnected:true,currentSrc:'https://board.example/api/frame?id=x&t=1',
  src:'https://board.example/api/frame?id=x&t=1',addEventListener(event,fn){this.loaded=fn;}};
 const responses=[...Array.from({length:5},()=>({status:503,ok:false,headers:{get:()=> '1'}})),
  {status:200,ok:true,blob:async()=>({})}];
 const api=new Function('fetch','URL','location','setTimeout','Date',piece('retryFrameImage')+'return retryFrameImage;')(
  async()=>{attempts++;return responses.shift();},urlApi,{href:'https://board.example/',origin:'https://board.example'},wait,now);
 await api(image);assert.equal(attempts,6);assert.equal(image.src,'blob:ready');image.loaded();assert.equal(revoked,1);
 await api({...image,currentSrc:'https://external.example/api/frame?id=x'});assert.equal(attempts,6);
 await api({...image,isConnected:false});assert.equal(attempts,6);
 await api({...image,currentSrc:'https://board.example/media/missing.jpg'});assert.equal(attempts,6);
 assert.equal(clock,5000,'a normally busy renderer can recover after five seconds');
 let failures=0;clock=0;
 const down=new Function('fetch','URL','location','setTimeout','Date',piece('retryFrameImage')+'return retryFrameImage;')(
  async()=>{failures++;return {status:503,ok:false,headers:{get:()=> '100000'}};},urlApi,
  {href:'https://board.example/',origin:'https://board.example'},wait,now);
 await down({...image,src:'https://board.example/api/frame?id=other',currentSrc:'https://board.example/api/frame?id=other'});assert.equal(clock,30000,'retry window is bounded');assert(failures<=16,'busy responses do not cause a retry storm');
})().catch(error=>{console.error(error);process.exit(1);});
'''
    subprocess.run(['node', '-e', script, str(HERE / 'page_functions.js')], check=True)
