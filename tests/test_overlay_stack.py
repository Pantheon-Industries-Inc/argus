import subprocess
from pathlib import Path


def test_recovery_moves_pressure_below_banner_without_a_positioning_cycle():
    source = Path(__file__).resolve().parents[1] / 'board/serve.py'
    script = r'''
const fs=require('fs'),assert=require('assert'),s=fs.readFileSync(process.argv[1],'utf8');
const start=s.indexOf('  function placeTop() {'),end=s.indexOf('  let _recSig',start);
function element(top,height,left=10,width=300){return {style:{top:top+'px'},disabled:false,static:false,
 classList:{active:false,contains(){return this.active}},closest(){return null},
 getBoundingClientRect(){return {top:+parseFloat(this.style.top),bottom:+parseFloat(this.style.top)+height,height,left,right:left+width}}};}
const cell=element(0,480);cell.querySelector=()=>null;
const hud=element(0,60),rec=element(0,110),grip=element(80,170),state=element(0,45,10);
const place=new Function('exoCell','topHud','progOverlay','stateToast','fsBtn','recOverlay','sensorOverlay','gripFindingOverlay','getComputedStyle',s.slice(start,end)+'return placeTop;')(cell,hud,null,state,null,rec,null,grip,e=>({position:e.static?'static':'absolute',opacity:e.entering?'0':e.leaving?'1':e.classList.active?'1':'0'}));
place();assert.equal(parseFloat(grip.style.top),68,'normal position below HUD');
rec.classList.active=true;rec.entering=true;state.leaving=true;place();
assert.equal(grip.style.transitionDuration,'0ms','entering recovery moves the sensor immediately');
assert.equal(grip.style.transitionDelay,'0ms','departing state toast cannot delay the sensor');
assert(parseFloat(grip.style.top)>=rec.getBoundingClientRect().bottom+8,'recovery and pressure must not overlap');
rec.entering=false;state.leaving=false;
state.classList.active=true;place();const top=parseFloat(grip.style.top);place();place();
assert.equal(parseFloat(grip.style.top),top,'no state/recovery/pressure positioning cycle');
assert(rec.getBoundingClientRect().top>=state.getBoundingClientRect().bottom+8);
assert(grip.getBoundingClientRect().top>=rec.getBoundingClientRect().bottom+8);
rec.classList.active=false;state.classList.active=false;place();assert.equal(parseFloat(grip.style.top),68,'return after recovery');
rec.classList.active=true;rec.static=true;rec.style.top='500px';place();
assert.equal(parseFloat(grip.style.top),68,'mobile recovery outside the video does not push pressure out of frame');
'''
    subprocess.run(['node', '-e', script, str(source)], check=True)


def test_tall_sensor_card_reserves_space_for_action_caption():
    source = Path(__file__).resolve().parents[1] / 'board/serve.py'
    script = r'''
const fs=require('fs'),assert=require('assert'),s=fs.readFileSync(process.argv[1],'utf8');
const start=s.indexOf('  function placeTop() {'),end=s.indexOf('  let _recSig',start);
function element(top,height,left,width){return {style:{top:top+'px'},disabled:false,
 classList:{active:true,contains(){return this.active}},closest(){return null},
 getBoundingClientRect(){return {top:parseFloat(this.style.top),bottom:parseFloat(this.style.top)+height,height,left,right:left+width,width}}};}
const cell=element(0,512,0,684),rec=element(0,84,8,668),grip=element(0,220,12,300),caption=element(428,40,222,240);
cell.querySelector=q=>q==='#video-overlay'?caption:null;
const place=new Function('exoCell','topHud','progOverlay','stateToast','fsBtn','recOverlay','sensorOverlay','gripFindingOverlay','getComputedStyle',s.slice(start,end)+'return placeTop;')(cell,element(0,120,0,0),null,null,null,rec,null,grip,e=>({position:'absolute',opacity:'1'}));
place();assert.equal(parseFloat(grip.style.top),220);
assert(parseFloat(caption.style.left)>=grip.getBoundingClientRect().right+8,'action caption must clear the sensor card');
assert.equal(caption.style.transform,'none');
rec.classList.active=false;place();
assert.equal(caption.style.left,'','normal action placement returns when the card clears it');
'''
    subprocess.run(['node', '-e', script, str(source)], check=True)
