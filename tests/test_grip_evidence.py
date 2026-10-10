import copy
import json
import shutil
import socketserver
import subprocess
import threading
import urllib.request
from pathlib import Path

import pytest

from board import to_board


@pytest.mark.skipif(not shutil.which('node'), reason='no node')
def test_saved_legacy_grip_is_withheld_from_current_view_and_downloads(tmp_path, monkeypatch):
    from board import serve
    from label.dictionary_editor import public_dictionary, save_override
    import numpy as np

    job = tmp_path / 'job'
    episode = job / 'episodes/episode_a'
    episode.mkdir(parents=True)
    (episode / 'context.json').write_text(json.dumps({'episode_id': 'episode_a', 'state_kind': 'none',
        'signals': [{'name': 'pressure_pad', 'key': 's0', 'file': 'signals.npz'}]}))
    np.savez(episode / 'signals.npz', s0=np.array([0.0, 1.0]))
    (job / 'dictionary.json').write_text(json.dumps({'schema': 1, 'inventory_digest': 'tiny', 'status': 'success',
        'inventory': {'fields': [{'id': 'pad', 'name': 'pressure_pad', 'kind': 'signal',
                                  'episodes': ['episode_a'], 'shape': [1],
                                  'bindings': [{'episode': 'episode_a', 'context_path': 'signals/0',
                                                'file': 'signals.npz', 'key': 's0'}]}]},
        'entries': {'pad': {'meaning': 'pressure pad', 'role': 'touch', 'provenance': 'machine'}}}))
    legacy = {'version': 1, 'findings': [{'start_s': .2, 'end_s': .7, 'headline': 'Legacy grip claim'}],
              'model': 'saved model', 'regions': {'thumb': [0, 1]}}
    saved = {'dataset': 'tiny', 'episode_prompt': 'lift', 'data_dictionary': public_dictionary(job, 'episode_a'),
             'grip_evidence': legacy, 'sensor_evidence': {'version': 1, 'findings': []},
             'completion': {'task_completed': 'failure'}}
    qa = job / 'qa/episode_a.json'
    qa.parent.mkdir()
    qa.write_text(json.dumps(saved))
    original = qa.read_bytes()
    save_override(job, {'revision': 0, 'field_id': 'pad', 'role': 'annotation'}, 'human')
    monkeypatch.setattr(serve, 'HERE', qa.parent)
    monkeypatch.setattr(serve, 'DICTIONARY_JOB', job)
    with socketserver.ThreadingTCPServer(('127.0.0.1', 0), serve.Handler) as httpd:
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            base = 'http://127.0.0.1:' + str(httpd.server_address[1])
            with urllib.request.urlopen(base + '/api/episode?file=episode_a.json') as response:
                current = json.load(response)
            with urllib.request.urlopen(base + '/api/episode?file=episode_a.json&download=1') as response:
                download = json.load(response)
            request = urllib.request.Request(base + '/api/export', data=b'{"files":["episode_a.json"]}',
                                             headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request) as response:
                jsonl = json.loads(response.read())
        finally:
            httpd.shutdown()
            thread.join()
    for row in (current, download, jsonl):
        assert row['data_dictionary']['entries']['pad']['role'] == 'annotation'
        assert 'grip_evidence' not in row
        assert row['withheld_grip_evidence'] == legacy
        assert row['sensor_evidence']['findings'] == []
    source = (Path(__file__).resolve().parents[1] / 'board/serve.py').read_text()
    begin = source.index('// ================= sensor evidence:')
    end = source.index('// ================= touch:', begin)
    script = ('const api=new Function("esc","fmtT",process.argv[1]+"return {sensorEvidence};")'
              '(String,String);console.log(JSON.stringify(['
              'api.sensorEvidence(JSON.parse(process.argv[2])).insights.length,'
              'api.sensorEvidence(JSON.parse(process.argv[3])).insights.length]));')
    counts = json.loads(subprocess.check_output([shutil.which('node'), '-e', script,
        source[begin:end], json.dumps(saved), json.dumps(current)], text=True))
    assert counts == [1, 0]
    (episode / 'context.json').unlink()
    assert 'grip_evidence' not in serve.current_label(saved, job, qa.name)
    from board.dictionary_projection import project
    rgb = {'episode_prompt': 'plain RGB'}
    assert project(rgb, tmp_path / 'plain') == rgb
    assert qa.read_bytes() == original


def test_saved_grip_evidence_survives_board_conversion():
    evidence = {"version": 1, "findings": [{"start_s": .6, "end_s": .87,
        "headline": "Likely grip tightening", "confidence": "medium"}], "model": "openai/gpt-6-astra"}
    result = {"episode_dir": "/x/episode_000001", "labels": {"task_summary": "Lift cup"},
              "grip_evidence": evidence, "usage": {"cost": .1}}
    before = copy.deepcopy(result)
    converted = to_board.convert(result, "demo")
    assert converted.get("grip_evidence") == evidence
    assert result == before
    assert converted["_usage"] == {"cost": .1}
    converted["grip_evidence"]["findings"].clear()
    assert result == before


@pytest.mark.skipif(not shutil.which('node'), reason='no node')
def test_grip_findings_are_timed_and_do_not_repeat_contact():
    root = Path(__file__).resolve().parents[1]
    script = r'''
const fs=require('fs'),assert=require('assert');
const source=fs.readFileSync(process.argv[1],'utf8');
const begin=source.indexOf('// ================= sensor evidence:');
const end=source.indexOf('// ================= touch:',begin);
const esc=s=>String(s).replace(/</g,'&lt;'),fmtT=t=>t+' s';
const api=new Function('esc','fmtT',source.slice(begin,end)+'return {sensorEvidence,activeSensorEvidence,sensorEvidenceHtml,displayGripFinding,gripFindingDetailHtml,gripFindingStripHtml,gripEvidencePanelHtml};')(esc,fmtT);
const grip={version:1,findings:[{start_s:.6,end_s:.87,headline:'Likely grip tightening',confidence:'medium',detail:'Response rises after closure',observation:'Several cells rise together',claim:'Additional relative loading',alternative:'Glove bending may contribute',inspect_s:.733},
 {start_s:2.73,end_s:3.03,headline:'Thumb response eases',confidence:'medium',detail:'Index response persists',inspect_s:3},
 {start_s:0,end_s:4,headline:'Rejected claim',review_status:'rejected'}]};
const evidence=api.sensorEvidence({grip_evidence:grip});
assert.equal(api.activeSensorEvidence(evidence,0),null,'no percentage or force at noncontact start');
assert.equal(api.activeSensorEvidence(evidence,.7)?.headline,'Likely grip tightening');
assert.equal(api.activeSensorEvidence(evidence,1),null,'do not extend an inferred change outside its span');
assert.equal(api.displayGripFinding(evidence,0),null);
assert.equal(api.displayGripFinding(evidence,.7).phase,'Now');
assert.equal(api.displayGripFinding(evidence,1).phase,'Earlier');
assert.equal(api.displayGripFinding(evidence,1).headline,'Likely grip tightening');
assert.equal(api.displayGripFinding(evidence,2.8).headline,'Thumb response eases');
assert.equal(api.displayGripFinding(evidence,3.8).phase,'Earlier');
assert.equal(api.displayGripFinding(evidence,.2),null,'backward seeking clears later findings');
assert.equal(api.displayGripFinding(evidence,.7).end,.87,'display persistence does not change the evidence interval');
const detail=api.gripFindingDetailHtml(evidence.insights[0]);
assert(detail.includes('Several cells rise together'));
assert(detail.includes('Additional relative loading'));
assert(detail.includes('Other explanations'));
assert(api.gripFindingDetailHtml(evidence.insights[0]).includes('Response rises after closure'),'supporting explanation remains available below the video');
const signed=api.gripEvidencePanelHtml({times:[0,1,2],map:{positions:[],signed:true},regions:{thumb:[-250,50,100],index:[-40,0,50]}});
assert(signed.includes('-300'),'signed response changes retain their negative axis');
assert(!signed.includes('NaN'));
const chartPaths=[...signed.matchAll(/<path d="([^"]+)"/g)].map(m=>m[1]);
assert(chartPaths.length===2);
for(const p of chartPaths) for(const match of p.matchAll(/[ML][^,]+,([0-9.]+)/g)) assert(+match[1]>=0 && +match[1]<=150,'negative samples remain inside the plot');

assert.equal(api.activeSensorEvidence(evidence,3)?.headline,'Thumb response eases');
assert.equal(api.activeSensorEvidence(evidence,.7)?.visual,'Medium confidence');
const panel=api.sensorEvidenceHtml(evidence,[]);
assert(panel.includes('Tactile findings')&&panel.replace(/<[^>]*>/g,'').includes('Thumb response eases'));
assert(panel.includes('Astra annotation'));
assert(panel.includes('aria-pressed="true"'),'the initial finding is selected and explained');
assert(!panel.includes('sensor-phase-profile'),'whole-glove percentages are not the evidence for mapped findings');
assert(!panel.includes('Rejected claim'),'a human rejection overrides the saved machine finding');
assert(panel.includes('data-evidence-t="0.733"'),'jump to the supporting interval');
assert.equal(api.sensorEvidenceHtml(api.sensorEvidence({}),[]),'');

const reviewed=api.sensorEvidence({contacts:[
 {id:'discarded',start_s:1,end_s:2,hand:'right',shown:true,review_status:'rejected',seen:{touch_seen:'no',hand:'left'}},
 {id:'raw',start_s:3,end_s:4,hand:'right',shown:true,seen:{review_status:'rejected',touch_seen:'no',hand:'left',slip:'yes',action:'Rejected action'}},
 {id:'accepted',start_s:5,end_s:6,hand:'right',shown:true,seen:{touch_seen:'yes',hand:'right',object:'mug'}}],
 contacts_missing:[{t_s:7,hand:'left',object:'cup',review_status:'rejected'},{t_s:8,hand:'right',object:'lid'}]});
assert.deepEqual(reviewed.contacts.map(c=>c.id),['raw','accepted']);
assert.equal(api.activeSensorEvidence(reviewed,3.8).headline,'Right hand contact');
assert.equal(api.activeSensorEvidence(reviewed,3.8).visual,'No visual verdict');
assert.equal(api.activeSensorEvidence(reviewed,3.8).detail,'');
assert.equal(api.activeSensorEvidence(reviewed,5.8).headline,'Right hand contact with mug');
assert.deepEqual(reviewed.moments.filter(m=>m.kind==='missing').map(m=>m.t),[8]);

'''
    run = subprocess.run([shutil.which('node'), '-e', script, str(root/'board/serve.py')],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr


@pytest.mark.skipif(not shutil.which('node'), reason='no node')
def test_grip_finding_inspection_pauses_at_supporting_sample():
    root = Path(__file__).resolve().parents[1]
    script = r'''
const fs=require('fs'),assert=require('assert');
const source=fs.readFileSync(process.argv[1],'utf8');
const begin=source.indexOf('  const seek = (t');
const end=source.indexOf("  document.querySelectorAll('.timeline .marker')",begin);
let paused=false,plays=0;
const vid={currentTime:0,parentElement:null,play(){plays++;paused=false;return Promise.resolve()},pause(){paused=true},scrollIntoView(){}};
const seek=new Function('vid','document',source.slice(begin,end)+'return seek;')(vid,{body:{}});
seek(.733,false);
assert.equal(vid.currentTime,.733);
assert(paused,'inspection keeps the short evidence interval visible');
assert.equal(plays,0);
seek(2);
assert.equal(plays,1,'ordinary timeline jumps still play');
'''
    run = subprocess.run([shutil.which('node'), '-e', script, str(root/'board/serve.py')], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr


@pytest.mark.skipif(not shutil.which('node'), reason='no node')
def test_local_patch_evidence_preserves_opposing_changes_and_censored_gaps():
    root = Path(__file__).resolve().parents[1]
    script = r'''
const fs=require('fs'),assert=require('assert');
const source=fs.readFileSync(process.argv[1],'utf8');
const begin=source.indexOf('// ================= sensor evidence:');
const end=source.indexOf('// ================= touch:',begin);
const api=new Function('esc','fmtT',source.slice(begin,end)+'return {sensorEvidence,gripEvidencePanelHtml};')(String,String);
const G={version:1,times:[0,1,2,3],regions:{thumb:[0,0,0,0]},map:{signed:true,positions:[{cell:207,region:'thumb',x:500,y:240},{cell:223,region:'thumb',x:510,y:240},{cell:253,region:'thumb',x:510,y:260},{cell:254,region:'thumb',x:520,y:260}],response:[[0,0,0,0],[-600,-400,500,700],[-800,-600,800,1000],[-700,-500,700,900]],zero:[[false,false,false,false],[false,false,false,false],[false,false,true,false],[false,false,false,false]]},findings:[{start_s:1,end_s:3,headline:'Opposing thumb loading',cell_groups:[{label:'Thumb patch A',cells:[207,223]},{label:'Thumb patch B',cells:[253,254]}]}]};
const f=api.sensorEvidence({grip_evidence:G}).insights[0];
const html=api.gripEvidencePanelHtml(G,f);
assert(html.includes('Local patches over time'),'local claims must expose local evidence instead of flat regional means');
assert(html.replace(/<[^>]*>/g,'').includes('Thumb patch A')&&html.replace(/<[^>]*>/g,'').includes('Thumb patch B'));
assert(html.includes('Censored samples remain blank'));
assert(!html.includes('NaN'));
const paths=[...html.matchAll(/<path d="([^"]+)"/g)].map(x=>x[1]);
assert.equal(paths.length,2);
assert(paths[1].includes('M600.00'),'a zero endpoint creates a gap, not a false pressure peak');
assert(paths[0]!==paths[1],'opposing local changes survive a zero whole-finger mean');
assert(html.includes('data-grip-patch-color'),'patch locations and trace colors are linked');
const fallback=api.gripEvidencePanelHtml(G,{cellGroups:[{label:'Invalid',cells:[999]}]});
assert(fallback.includes('Response over time'),'unknown cell groups fall back to regional evidence');
'''
    run = subprocess.run([shutil.which('node'), '-e', script, str(root/'board/serve.py')], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr



@pytest.mark.skipif(not shutil.which('node'), reason='no node')
def test_anatomical_copy_is_complete_emphasized_and_escaped():
    root = Path(__file__).resolve().parents[1]
    script = r"""
const fs=require('fs'),assert=require('assert');
const source=fs.readFileSync(process.argv[1],'utf8');
const begin=source.indexOf('// ================= sensor evidence:');
const end=source.indexOf('// ================= touch:',begin);
const esc=s=>String(s).replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[c]));
const api=new Function('esc','fmtT',source.slice(begin,end)+'return {sensorEvidence,gripFingerCopy,gripFingerHtml,gripFindingStripHtml};')(esc,String);
assert.equal(api.gripFingerCopy('Thumb and index pressure increase'),'Thumb and index finger pressure increase');
assert.equal(api.gripFingerCopy('index finger, ring finger, little finger, middle finger'),'index finger, ring finger, little finger, middle finger');
assert.equal(api.gripFingerCopy('index-finger pressure at the index fingertip'),'index finger pressure at the index finger tip');
assert.equal(api.gripFingerCopy('middle of the thumb'),'middle of the thumb');
const raw={version:1,findings:[{start_s:1,end_s:2,headline:'Thumb and index pressure increase',detail:'Index pressure rises',confidence:'medium'}]};
const before=JSON.stringify(raw),E=api.sensorEvidence({grip_evidence:raw});
assert.equal(JSON.stringify(raw),before,'display formatting preserves original saved evidence');
assert.equal(E.insights[0].headline,'Thumb and index finger pressure increase');
const html=api.gripFindingStripHtml({...E.insights[0],phase:'Now'});
assert(html.includes('<span class="grip-finger-name" data-finger="thumb">Thumb</span> and <span class="grip-finger-name" data-finger="index">index finger</span> pressure increase'));
assert(!html.includes('gf-detail')&&!html.includes('gf-confidence'),'the overlay keeps only the finding and pressure visual');
assert.equal(api.gripFingerHtml('<img src=x onerror=alert(1)> thumb').includes('<img'),false);
assert(!source.includes('id="grip-hand-overlay"'));
assert(!source.includes('title="${gripFingerHtml('),'rich text is never interpolated into HTML attributes');
assert(source.includes('id="grip-finding-overlay"'),'the left pressure overlay is preserved');
"""
    run = subprocess.run([shutil.which('node'), '-e', script, str(root/'board/serve.py')], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr


@pytest.mark.skipif(not shutil.which('node'), reason='no node')
def test_overlay_pressure_traces_keep_scale_sign_and_missing_readings():
    root = Path(__file__).resolve().parents[1]
    script = r"""
const fs=require('fs'),assert=require('assert');
const source=fs.readFileSync(process.argv[1],'utf8');
const begin=source.indexOf('// ================= sensor evidence:');
const end=source.indexOf('// ================= touch:',begin);
const node=dataset=>({dataset,attributes:{},setAttribute(k,v){this.attributes[k]=v}});
const dots=[node({gripMiniDot:'0'}),node({gripMiniDot:'1'})],gaps=[node({gripMiniGap:'0'}),node({gripMiniGap:'1'})],time={textContent:''};
const document={querySelectorAll(q){return q==='[data-grip-mini-dot]'?dots:q==='[data-grip-mini-gap]'?gaps:q==='[data-grip-mini-time]'?[time]:[]}};
const api=new Function('esc','fmtT','document',source.slice(begin,end)+'return {gripOverlaySeries,gripOverlayPressureHtml,syncGripOverlayPressure};')(String,t=>t+'s',document);
const G={times:[0,.03,.06,.09],regions:{thumb:[0,100,-50,200],index:[0,50,null,100],middle:[0,500,800,900]},map:{positions:[],response:[]}};
const f={headline:'Thumb and index finger pressure increase',detail:'Pressure rises',start:.03,end:.06};
const traces=api.gripOverlaySeries(G,f),original=JSON.stringify(G);
assert.deepEqual(traces.map(v=>v.label),['Thumb','Index finger'],'unmentioned fingers do not clutter the overlay');
const html=api.gripOverlayPressureHtml(G,f);
assert(html.includes('stroke-dasharray="2 3"'),'the initial level remains a visible reference');
assert(!html.includes('%')&&!html.includes('NaN'));
api.syncGripOverlayPressure(G,f,.03);const firstY=dots[0].attributes.cy;
api.syncGripOverlayPressure(G,f,.09);assert(dots[0].attributes.cy<firstY,'larger response is higher on the same fixed scale');
api.syncGripOverlayPressure(G,f,.06);assert.equal(dots[1].attributes.visibility,'hidden');assert.equal(gaps[1].attributes.visibility,'visible');
api.syncGripOverlayPressure(G,f,1);assert(dots.every(d=>d.attributes.visibility==='hidden'),'old samples are not shown as current through gaps');
api.syncGripOverlayPressure(G,f,.03);assert.equal(dots[0].attributes.cy,firstY,'seeking preserves the scale');
assert.equal(JSON.stringify(G),original,'display does not change evidence');
const local={times:[0,.03,.06],regions:{thumb:[0,0,0]},map:{positions:[{cell:1},{cell:2}],response:[[0,0],[100,-100],[200,-200]],zero:[[false,false],[false,true],[false,false]]}};
const patchFinding={headline:'Pressure shifts across the thumb',start:.03,end:.06,cellGroups:[{label:'Thumb patch A',cells:[1]},{label:'Thumb patch B',cells:[2]}]};
const patches=api.gripOverlaySeries(local,patchFinding);
assert.deepEqual(patches.map(v=>v.label),['Thumb area 1','Thumb area 2']);
assert.equal(patches[0].values[2],200);assert.equal(patches[1].values[2],-200,'opposing patches are not collapsed into a flat finger average');
assert(!Number.isFinite(patches[1].values[1]),'censored endpoints do not become false pressure values');
"""
    run = subprocess.run([shutil.which('node'), '-e', script, str(root/'board/serve.py')], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
