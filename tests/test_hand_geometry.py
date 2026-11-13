"""Canonical geometry contracts reject stale chunks and invalid native topology."""
import subprocess
from pathlib import Path
import pytest


def manifest(prediction="p1", chunk="part.json"):
    return {"format":"board-hand-geometry/1", "prediction_id":prediction,
            "coordinates":{"xy":"source_pixels","z":"camera_meters"},
            "clip":{"w":100,"h":50,"times":[0]}, "edges":[[0,1]],
            "topology":{"left":[[0,1,2]],"right":[]},
            "chunks":[{"file":chunk,"start":0,"end":1}]}


def test_hand_asset_routing_defaults_and_explicit_base():
    from board import serve
    import json
    import re
    config = {'mode':'static', 'data':'metadata/', 'hands_data':'https://assets.example/shared/'}
    page = serve.render_index('Example', config)
    board = json.loads(re.search(r'const BOARD = (.*);', page).group(1))
    script = r"""
const assert = require('node:assert/strict'), {src,piece} = require('./tests/page_functions')();
let STATIC=true, BOARD=JSON.parse(process.argv[1]);
const EP_SITE={'private/episode.json':'public episode.json'};
eval(src.match(/const siteFile = [^;]+;/)[0].replace('const siteFile', 'var siteFile'));
eval(piece('handsUrl'));
assert.equal(handsUrl('private/episode.json'),'https://assets.example/shared/hands/public%20episode.json');
assert.equal(handsUrl('part.json'),'https://assets.example/shared/hands/part.json');
delete BOARD.hands_data;
assert.equal(handsUrl('private/episode.json'),'metadata/hands/public%20episode.json');
BOARD.data='';
assert.equal(handsUrl('part.json'),'hands/part.json');
STATIC=false; BOARD.hands_data='https://assets.example/shared/';
assert.equal(handsUrl('private/episode.json'),'api/hands?file=private%2Fepisode.json');
"""
    subprocess.run(['node', '-e', script, json.dumps(board)], check=True)


def test_hand_header_priority_preserves_legacy_api_fetch_options():
    subprocess.run(['node', '-e', r"""
const assert=require('node:assert/strict'),{piece}=require('./tests/page_functions')();
const STATIC=false,BOARD={},_hpCache=new Map(),_hpData=new Map();
const hpDecode=doc=>doc,remember=(cache,key,promise)=>(cache.set(key,promise),promise);
const requests=[];
const fetch=async(url,options)=>{requests.push({url,options});return {ok:true,json:async()=>({n:1})};};
eval(piece('handsUrl')+piece('fetchJson')+piece('loadHands'));
(async()=>{assert.deepEqual(await loadHands('private/episode.json'),{n:1});
assert.equal(requests.length,1);
assert.equal(requests[0].url,'api/hands?file=private%2Fepisode.json');
assert.equal(requests[0].options.cache,'no-store');
assert.equal(requests[0].options.priority,'high');
assert.deepEqual(await loadHands('private/episode.json'),{n:1});
assert.equal(requests.length,1);
})().catch(error=>{console.error(error);process.exit(1);});
"""], check=True)


def test_geometry_contract():
    html = Path('board/index.html').read_text()
    source = html.split('// BEGIN canonical hand geometry')[1].split('// END canonical hand geometry')[0] if '// BEGIN canonical hand geometry' in html else ''
    script = source + r'''
const assert = require('node:assert/strict');
const doc = {format:'board-hand-geometry/1', prediction_id:'p1', coordinates:{xy:'source_pixels',z:'camera_meters'},
 clip:{w:100,h:50,times:[0,.04,.09]},edges:[[0,1]],topology:{left:[[0,1,2]],right:[]},
 chunks:[{start:0,end:2,file:'part-0.json'},{start:2,end:3,file:'part-1.json'}]};
const d = hpGeometry(doc);
assert.equal(d.n,3);
assert.equal(hpChunkAt(d,1).file,'part-0.json');
assert.equal(hpChunkAt(d,2).file,'part-1.json');
assert.throws(()=>hpGeometry({...doc,clip:{...doc.clip,times:[0,.09,.04]}}));
assert.throws(()=>hpGeometry({...doc,chunks:[{start:0,end:3,file:'../private.json'}]}));
const hand={joints:[[5,6,1],[8,9,1]],vertices:[[0,0,1],[10,0,1],[0,10,1]]};
const chunk={format:'board-hand-geometry-chunk/1',prediction_id:'p1',start:0,frames:[{left:hand},{left:null}]};
assert.equal(hpGeometryChunk(d,doc.chunks[0],chunk).frames[0].left.vertices.length,3);
assert.throws(()=>hpGeometryChunk(d,doc.chunks[0],{...chunk,prediction_id:'old'}));
assert.throws(()=>hpGeometryChunk(d,doc.chunks[0],{...chunk,frames:[{left:{...hand,vertices:[[0,0,1]]}},{}]}));
assert.throws(()=>hpGeometryChunk(d,doc.chunks[0],{...chunk,frames:[{left:{...hand,joints:[[NaN,0,1]]}},{}]}));
const tris=hpTriangles({left:hand},doc.topology);
assert.equal(tris.length,1);
assert.deepEqual(tris[0].points,hand.vertices);
'''
    result = subprocess.run(['node', '-e', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_geometry_web_copy_requires_identical_timing(monkeypatch):
    import pytest
    from board import hands
    doc = manifest()
    doc['clip']['times']=[0,.04,.09]
    doc['chunks'][0]['end']=3
    monkeypatch.setattr(hands, 'probe_clip', lambda _: {'w': 100, 'h': 50, 'frames': 3, 'tb': [1,100], 'dur': 4, 'pts': 'AAAC'})
    # Supply decoded timestamps independently: web encodes must retain every source frame.
    monkeypatch.setattr(hands, 'frame_times', lambda _: [0, .04, .09])
    assert hands.for_copy(doc, Path('clip.mp4'))['clip']['times'] == [0, .04, .09]
    monkeypatch.setattr(hands, 'frame_times', lambda _: [0, .04, .08])
    with pytest.raises(ValueError):
        hands.for_copy(doc, Path('clip.mp4'))


def test_chunk_assets_reject_traversal_and_wrong_prediction(tmp_path):
    import json
    import pytest
    from board.hands import geometry_assets
    doc = manifest()
    (tmp_path/'part.json').write_text(json.dumps({'format':'board-hand-geometry-chunk/1','prediction_id':'old','start':0,'frames':[{}]}))
    with pytest.raises(ValueError):
        list(geometry_assets(doc,tmp_path))
    doc['chunks'][0]['file']='../private.json'
    with pytest.raises(ValueError):
        list(geometry_assets(doc,tmp_path))


@pytest.mark.parametrize('hands_data', [None, 'shared/'])
def test_overlay_playback_seek_modes_and_chunk_loading(tmp_path, hands_data):
    """Use the production overlay on actual decoded video in headless Chrome."""
    import contextlib
    import http.server
    import json
    import tempfile
    import threading
    import time
    import pytest
    from test_layout_fits import chrome, _devtools
    import shutil
    if not chrome() or not shutil.which('ffmpeg'):
        pytest.skip('Chrome and ffmpeg required')
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=black:s=160x120:r=10:d=3',
                    '-c:v','libx264','-pix_fmt','yuv420p',str(tmp_path/'clip.mp4')],check=True)
    html=Path('board/index.html').read_text()
    script=html.split('let HP_MODE =')[1].split('// ================= the recording')[0]
    script='let HP_MODE ='+script
    doc={'format':'board-hand-geometry/1','prediction_id':'fixture','coordinates':{'xy':'source_pixels','z':'camera_meters'},
         'clip':{'w':160,'h':120,'times':[i/10 for i in range(30)]},'edges':[[0,1]],
         'topology':{'left':[[0,1,2],[0,2,3]],'right':[]},
         'chunks':[{'start':0,'end':15,'file':'chunk-a.json'},{'start':15,'end':30,'file':'chunk-b.json'}]}
    asset_base = hands_data or 'metadata/'
    asset_dir = tmp_path / asset_base / 'hands'
    asset_dir.mkdir(parents=True)
    (asset_dir/'safe-episode.json').write_text(json.dumps(doc))
    missing={**doc, 'chunks':[{'start':0,'end':30,'file':'absent-part.json'}]}
    (asset_dir/'missing-episode.json').write_text(json.dumps(missing))
    for name,start,x in [('chunk-a.json',0,10),('chunk-b.json',15,80)]:
        frame={'left':{'joints':[[x,20,1],[x+20,40,1]],'vertices':[[x,20,1],[x+20,20,1],[x+20,40,1],[x,40,1]]}}
        (asset_dir/name).write_text(json.dumps({'format':'board-hand-geometry-chunk/1','prediction_id':'fixture',
                                              'start':start,'frames':[frame]*15}))
    import base64
    video_url='data:video/mp4;base64,'+base64.b64encode((tmp_path/'clip.mp4').read_bytes()).decode()
    prelude="const STATIC=true, BOARD="+json.dumps({'hands':True,'data':'metadata/',**({'hands_data':hands_data} if hands_data else {})})+""", KP_INDEX=null; let _activeFile='episode.json';
window._epCleanup=[]; const EP_SITE={'episode.json':'safe-episode.json'};
const siteFile=file=>EP_SITE[file] || file;
const handFetches=[],nativeFetch=window.fetch.bind(window);
window.fetch=(url,options={})=>{handFetches.push({url,priority:options.priority,abortable:options.signal instanceof AbortSignal});return nativeFetch(url,options);};
PRODUCTION_FETCH_JSON
function remember(c,k,p){c.set(k,p);return p;}
let testAbort=new AbortController(); const on=(e,k,f)=>e.addEventListener(k,f,{signal:testAbort.signal});
function selectMode(mode){document.getElementById('hp-mode').dispatchEvent(new KeyboardEvent('keydown',{key:{off:'Home',skeleton:'ArrowRight',mesh:'End'}[mode]}));}
"""
    prelude=prelude.replace('PRODUCTION_FETCH_JSON', 'async function fetchJson('+html.split('async function fetchJson(',1)[1].split('function episodeDownloadUrl',1)[0])
    import re
    mode = re.search(r'<button id="hp-mode"[\s\S]*?</button>', html).group()
    tail="""
const vid=document.querySelector('video');
vid.addEventListener('loadedmetadata',()=>setupHandPose(vid,document.getElementById('cell'),true,on,_activeFile));
"""
    (tmp_path/'index.html').write_text('''<style>#cell{position:relative;width:320px;height:280px}video{position:absolute;top:40px;width:320px;height:240px}.hp-canvas{position:absolute;pointer-events:none}</style>
<button id="hp-btn" class="ep-head-dl hp-btn" aria-pressed="true"><span class="hp-sw"></span>Hand pose</button>
<span id="hp-mode-home" hidden></span>MODE_MARKUP<div id="hp-aligned"><div class="kp-note-in"></div></div>
<div id="cell"><video src="VIDEO_URL" muted playsinline></video><canvas id="hp-canvas"></canvas></div><script>'''+prelude+script+tail+'</script>')
    (tmp_path/'index.html').write_text((tmp_path/'index.html').read_text().replace('MODE_MARKUP',mode))
    (tmp_path/'index.html').write_text((tmp_path/'index.html').read_text().replace('VIDEO_URL',video_url))
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def __init__(self,*args,**kwargs): super().__init__(*args,directory=str(tmp_path),**kwargs)
        def log_message(self,*args): pass
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Quiet)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    with tempfile.TemporaryDirectory() as profile:
        send,close=_devtools(chrome(),['--headless=new','--no-sandbox','--autoplay-policy=no-user-gesture-required',f'--user-data-dir={profile}','about:blank'])
        try:
            target=send('Target.createTarget',{'url':f'http://127.0.0.1:{server.server_port}/index.html'})['targetId']
            sid=send('Target.attachToTarget',{'targetId':target,'flatten':True})['sessionId']
            def js(code):
                result=send('Runtime.evaluate',{'expression':code,'returnByValue':True,'awaitPromise':True,'userGesture':True},sid,deadline=time.monotonic()+20)
                assert 'exceptionDetails' not in result,result
                return result.get('result',{}).get('value')
            for _ in range(100):
                if js("!!window._hp && window._hp.state.data"):break
                time.sleep(.05)
            assert js('window._hp.state.data')
            assert js("performance.getEntriesByType('resource').filter(e=>e.name.endsWith('/safe-episode.json')).map(e=>new URL(e.name).pathname)")==[f'/{asset_base}hands/safe-episode.json']
            assert js("_hpData.get(_activeFile).pending.has('chunk-b.json') || _hpData.get(_activeFile).loaded.has('chunk-b.json')") is True
            assert js("handFetches.map(r=>({file:r.url.split('/').pop(),priority:r.priority,abortable:r.abortable}))")==[
                {'file':'safe-episode.json','priority':'high','abortable':False},
                {'file':'chunk-a.json','priority':'high','abortable':True},
                {'file':'chunk-b.json','priority':'high','abortable':True}]
            js("selectMode('mesh')")
            for _ in range(100):
                if js("document.querySelector('.hp-status').textContent === ''"):break
                time.sleep(.02)
            assert js("document.getElementById('hp-mode').getAttribute('aria-valuetext')")=='Mesh'
            assert js("[...document.querySelectorAll('#hp-mode [data-mode]')].map(e=>e.textContent)")==['Off','Skeleton','Mesh']
            for mode, expected in (('off','Off'),('skeleton','Skeleton'),('mesh','Mesh')):
                js(f"document.querySelector('#hp-mode [data-mode={mode}]').click()")
                assert js("document.getElementById('hp-mode').getAttribute('aria-valuetext')")==expected
            track=js("""(() => {const r=document.querySelector('#hp-mode .hp-track').getBoundingClientRect();
return {left:r.left+1,right:r.right-1,y:r.top+r.height/2};})()""")
            send('Input.dispatchMouseEvent',{'type':'mousePressed','x':track['left'],'y':track['y'],'button':'left','clickCount':1},sid)
            send('Input.dispatchMouseEvent',{'type':'mouseMoved','x':track['right'],'y':track['y'],'buttons':1},sid)
            send('Input.dispatchMouseEvent',{'type':'mouseReleased','x':track['right'],'y':track['y'],'button':'left','clickCount':1},sid)
            assert js("document.getElementById('hp-mode').getAttribute('aria-valuenow')")=='2'
            assert js("document.querySelector('canvas').getContext('2d').getImageData(30,60,1,1).data[3]")>0
            assert js("""(() => { const surface=hpSurface(); const image=surface.draw([
              {h:'right',points:[[0,0,1],[100,0,10],[0,100,10]]},
              {h:'left',points:[[0,0,2],[100,0,2],[0,100,2]]}],100,100,100,100);
              const c=document.createElement('canvas');c.width=100;c.height=100;
              const ctx=c.getContext('2d');ctx.drawImage(image,0,0);
              const near=ctx.getImageData(5,5,1,1).data[0],far=ctx.getImageData(80,5,1,1).data[0];
              surface.dispose();return near>120 && far<100;})()""")
            js("vid.currentTime=2")
            for _ in range(100):
                if js('window._hp.state.frame>=20') and js("document.querySelector('.hp-status').textContent === ''"):break
                time.sleep(.02)
            assert js('window._hp.state.frame')==20, js('JSON.stringify({state:window._hp.state,time:vid.currentTime,seeking:vid.seeking,ready:vid.readyState,error:vid.error,status:document.querySelector(".hp-status").textContent})')
            assert js("performance.getEntriesByType('resource').filter(e=>e.name.endsWith('chunk-b.json')).map(e=>new URL(e.name).pathname)")==[f'/{asset_base}hands/chunk-b.json']
            assert js("document.querySelector('canvas').getContext('2d').getImageData(30,60,1,1).data[3]")==0
            js("vid.play(); selectMode('off')")
            assert js('window._hp.state.active') is False
            assert js('vid.paused') is False
            assert js("document.querySelector('canvas').getContext('2d').getImageData(170,60,1,1).data[3]")==0
            js("selectMode('off');selectMode('skeleton')")
            assert js('window._hp.state.active') is True
            js("document.getElementById('cell').style.width='195px';vid.style.width='195px'")
            assert js('window._hp.state.data') is True
            # The same pill follows the fullscreen camera, then returns to its header slot.
            js("window.savedPill=document.getElementById('hp-mode');document.getElementById('cell').requestFullscreen()")
            js("new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))")
            assert js("document.fullscreenElement===document.getElementById('cell') && savedPill.parentElement===document.fullscreenElement") is True
            js("selectMode('mesh')")
            assert js("savedPill.getAttribute('aria-valuetext')")=='Mesh'
            js("document.exitFullscreen()")
            js("new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))")
            assert js("document.getElementById('hp-mode-home').nextElementSibling===savedPill") is True
            js("selectMode('off');selectMode('skeleton')")
            # Second episode setup must discard the first episode's frame and canvas state.
            js("window._epCleanup.forEach(f=>f()); window._epCleanup=[];testAbort.abort();testAbort=new AbortController(); setupHandPose(vid,document.getElementById('cell'),true,on,'episode.json')")
            assert js('window._hp.state.active') is True
            js("window._epCleanup.forEach(f=>f()); window._epCleanup=[];testAbort.abort();testAbort=new AbortController(); const nativeContext=HTMLCanvasElement.prototype.getContext; HTMLCanvasElement.prototype.getContext=function(kind,...args){return kind==='webgl'?null:nativeContext.call(this,kind,...args)}; setupHandPose(vid,document.getElementById('cell'),true,on,'episode.json'); selectMode('mesh')")
            assert js("document.querySelector('.hp-status').textContent")=='Unavailable'
            js("HTMLCanvasElement.prototype.getContext=nativeContext; selectMode('off');selectMode('skeleton')")
            assert js('window._hp.state.active') is True
            js("window._epCleanup.forEach(f=>f()); window._epCleanup=[];testAbort.abort();testAbort=new AbortController(); _activeFile='missing-episode.json'; setupHandPose(vid,document.getElementById('cell'),true,on,_activeFile); vid.currentTime=0;vid.play()")
            for _ in range(100):
                if js("document.querySelector('.hp-status').textContent === 'Unavailable'"):break
                time.sleep(.02)
            time.sleep(.4)
            assert js('vid.currentTime') > .2
            assert js('vid.paused') is False
            assert js("document.querySelector('.hp-status').textContent")=='Unavailable'
            assert js("performance.getEntriesByType('resource').filter(e=>e.name.endsWith('absent-part.json')).length")==1
            # An optional mesh preference must never replace or disable legacy controls.
            js("window._epCleanup.forEach(f=>f());window._epCleanup=[];testAbort.abort();testAbort=new AbortController();_activeFile='legacy.json';HP_MODE='mesh';HP_ON=true;_hpData.set(_activeFile,{n:30,w:160,h:120,times:Array.from({length:30},(_,i)=>i/10),edges:[[0,1]],hands:{left:{xy:new Float32Array(30*42).fill(20),spans:[[0,30]]}}});setupHandPose(vid,document.getElementById('cell'),true,on,_activeFile)")
            assert js("document.getElementById('hp-btn').hidden") is False
            assert js("document.getElementById('hp-mode').hidden") is True
            assert js('window._hp.state.active') is True
            js("document.getElementById('hp-btn').click()")
            assert js('window._hp.state.active') is False
            assert js("localStorage.getItem('board.handPose')")=='0'
            js("window._epCleanup.forEach(f=>f());window._epCleanup=[];testAbort.abort();testAbort=new AbortController();_activeFile='legacy-unfetched.json';setupHandPose(vid,document.getElementById('cell'),true,on,_activeFile)")
            assert js("document.getElementById('hp-btn').hidden") is False
            assert js("document.getElementById('hp-mode').hidden") is True
            assert js("performance.getEntriesByType('resource').filter(e=>e.name.endsWith('legacy-unfetched.json')).length")==0
        finally:
            close()
            server.shutdown();server.server_close()


def test_board_build_copies_canonical_geometry_without_inference(tmp_path):
    import json
    from board.build import build_hands
    board=tmp_path/'board';board.mkdir()
    qa=board/'qa.new';qa.mkdir()
    (qa/'episode_000000.json').write_text('{}')
    src=tmp_path/'geometry';src.mkdir()
    doc=manifest('p')
    (src/'episode_000000.json').write_text(json.dumps(doc))
    (src/'part.json').write_text(json.dumps({'format':'board-hand-geometry-chunk/1','prediction_id':'p','start':0,'frames':[{}]}))
    result=build_hands(board,{'geometry':str(src)},qa,{})
    assert result['written']==1
    assert json.loads((board/'hands.new'/'part.json').read_text())['prediction_id']=='p'
    assert not list((board/'hand_keypoints.new').iterdir())


def test_public_geometry_validation_rejects_bad_units_timing_bounds_and_values(tmp_path):
    import copy
    import json
    import pytest
    from board.hands import geometry_assets, validate_geometry
    valid=manifest()
    variants=[]
    d=copy.deepcopy(valid);d["private_checkpoint_path"]="private";variants.append(d)
    d=copy.deepcopy(valid);d['coordinates']['z']='millimeters';variants.append(d)
    d=copy.deepcopy(valid);d['clip']['times']=[0,float('nan')];variants.append(d)
    d=copy.deepcopy(valid);d['clip']['times']=[.1,0];variants.append(d)
    d=copy.deepcopy(valid);d['chunks']=[{'start':0,'end':1,'file':'part.json'},{'start':0,'end':1,'file':'part.json'}];variants.append(d)
    d=copy.deepcopy(valid);d['chunks'][0]['end']=2;variants.append(d)
    for doc in variants:
        with pytest.raises(ValueError): validate_geometry(doc)
    hand={'joints':[[0,0,1],[1,1,1]],'vertices':[[0,0,1],[1,0,1],[0,1,1]]}
    chunk={'format':'board-hand-geometry-chunk/1','prediction_id':'p1','start':0,'frames':[{'left':hand}]}
    (tmp_path/'part.json').write_text(json.dumps(chunk))
    assert len(list(geometry_assets(valid,tmp_path)))==1
    for field,value in [('vertices',[[0,0,1]]),('vertices',[[0,0,float('nan')],[1,0,1],[0,1,1]]),('joints',[[0,0,-1],[1,0,1]])]:
        bad=copy.deepcopy(chunk);bad['frames'][0]['left'][field]=value
        (tmp_path/'part.json').write_text(json.dumps(bad))
        with pytest.raises(ValueError):list(geometry_assets(valid,tmp_path))
    chunk['frames']=[];(tmp_path/'part.json').write_text(json.dumps(chunk))
    with pytest.raises(ValueError):list(geometry_assets(valid,tmp_path))


def test_two_episode_chunk_collision_refuses_mixed_prediction(tmp_path):
    import json
    import pytest
    from board.hands import geometry_bundle
    for name,pred in [('episode_000001.json','p1'),('episode_000002.json','p2')]:
        (tmp_path/name).write_text(json.dumps(manifest(pred)))
    (tmp_path/'part.json').write_text(json.dumps({'format':'board-hand-geometry-chunk/1','prediction_id':'p1','start':0,'frames':[{}]}))
    with pytest.raises(ValueError):geometry_bundle(tmp_path,['episode_000001.json','episode_000002.json'])


def test_identical_shared_chunks_reuse_public_filename(tmp_path):
    import json
    from board.hands import geometry_bundle
    for name in ['episode_000001.json','episode_000002.json']:
        (tmp_path/name).write_text(json.dumps(manifest()))
    (tmp_path/'part.json').write_text(json.dumps({'format':'board-hand-geometry-chunk/1','prediction_id':'p1','start':0,'frames':[{}]}))
    assert set(geometry_bundle(tmp_path,['episode_000001.json','episode_000002.json']))=={'episode_000001.json','episode_000002.json','part.json'}


@pytest.mark.parametrize("display_mode", ["skeleton", "mesh"])
def test_geometry_prefetch_continuous_delayed_chunks_and_seek_abort(tmp_path, display_mode):
    """Real video crosses six boundaries with 280ms chunks and one 1.2s response."""
    import base64
    import http.server
    import json
    import re
    import shutil
    import tempfile
    import threading
    import time
    import pytest
    from test_layout_fits import chrome, _devtools
    if not chrome() or not shutil.which('ffmpeg'):
        pytest.skip('Chrome and ffmpeg required')
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=black:s=160x120:r=30:d=8',
                    '-c:v','libx264','-pix_fmt','yuv420p',str(tmp_path/'clip.mp4')],check=True)
    header={'format':'board-hand-geometry/1','prediction_id':'buffer-fixture',
            'coordinates':{'xy':'source_pixels','z':'camera_meters'},
            'clip':{'w':160,'h':120,'times':[i/30 for i in range(240)]},'edges':[[0,1]],
            'topology':{'left':[[0,1,2]],'right':[]},
            'chunks':[{'start':i*30,'end':(i+1)*30,'file':f'chunk-{i}.json'} for i in range(8)]}
    (tmp_path/'episode.json').write_text(json.dumps(header))
    for spec in header['chunks']:
        frames=[]
        for frame in range(spec['start'],spec['end']):
            x=20+frame%100
            vertices=[[x+(i%3)*.123456789012345,20+(i%5)*.123456789012345,1.123456789012345] for i in range(718)]
            vertices[:3]=[[x-3,17,1],[x+3,17,1],[x,25,1]]
            hand={'joints':[[x,20,1],[x+10,30,1]],'vertices':vertices}
            frames.append({'left':hand,'right':hand})
        (tmp_path/spec['file']).write_text(json.dumps({'format':'board-hand-geometry-chunk/1',
            'prediction_id':'buffer-fixture','start':spec['start'],'frames':frames}))
    source=Path('board/index.html').read_text()
    script='let HP_MODE ='+source.split('let HP_MODE =',1)[1].split('// ================= the recording',1)[0]
    mode=re.search(r'<button id="hp-mode"[\s\S]*?</button>',source).group()
    css=source.split('<style>',1)[1].split('</style>',1)[0]
    video='data:video/mp4;base64,'+base64.b64encode((tmp_path/'clip.mp4').read_bytes()).decode()
    prelude="""const STATIC=true,BOARD={hands:true,data:''},KP_INDEX=null;let _activeFile='episode.json';
window._epCleanup=[];const siteFile=x=>x;const remember=(c,k,p)=>(c.set(k,p),p);
function fetchJson(url,options={}){return fetch(url.replace('hands/',''),options).then(r=>r.ok?r.json():null);}
const testAbort=new AbortController(),on=(e,k,f)=>e.addEventListener(k,f,{signal:testAbort.signal});
"""
    page='<style>'+css+'#cell{position:relative;width:320px;height:240px}video{width:320px;height:240px}</style>'
    page+='<button id="hp-btn"></button><span id="hp-mode-home" hidden></span>'+mode
    page+='<div id="hp-aligned"><div class="kp-note-in"></div></div><div id="cell"><video muted src="'+video+'"></video><canvas id="hp-canvas"></canvas></div>'
    page+='<script>'+prelude+script+"const vid=document.querySelector('video');vid.addEventListener('loadedmetadata',()=>setupHandPose(vid,document.getElementById('cell'),true,on,_activeFile));</script>"
    (tmp_path/'index.html').write_text(page)
    class Delayed(http.server.SimpleHTTPRequestHandler):
        def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(tmp_path),**kwargs)
        def do_GET(self):
            if self.path.startswith('/chunk-'):time.sleep(1.2 if self.path=='/chunk-4.json' else .28)
            try:super().do_GET()
            except (BrokenPipeError,ConnectionResetError):pass
        def log_message(self,*args):pass
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Delayed)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    with tempfile.TemporaryDirectory() as profile:
        send,close=_devtools(chrome(),['--headless=new','--no-sandbox','--autoplay-policy=no-user-gesture-required',f'--user-data-dir={profile}','about:blank'])
        try:
            target=send('Target.createTarget',{'url':f'http://127.0.0.1:{server.server_port}/index.html'})['targetId']
            sid=send('Target.attachToTarget',{'targetId':target,'flatten':True})['sessionId']
            def js(code):
                result=send('Runtime.evaluate',{'expression':code,'returnByValue':True,'awaitPromise':True,'userGesture':True},sid,deadline=time.monotonic()+20)
                assert 'exceptionDetails' not in result,result
                return result.get('result',{}).get('value')
            for _ in range(200):
                if js("window._hp?.state.data && document.getElementById('hp-status').textContent==='' "):break
                time.sleep(.02)
            js(f"document.querySelector('[data-mode={display_mode}]').click()")
            result=js("""(async()=>{const samples=[],pill=document.getElementById('hp-mode'),width=pill.getBoundingClientRect().width;
await vid.play();while(vid.currentTime<6.4){await new Promise(r=>setTimeout(r,15));const s=window._hp.state,d=_hpData.get(_activeFile);
if(s.frame<1)continue;const x=(20+s.frame%100)*2,alpha=document.getElementById('hp-canvas').getContext('2d').getImageData(x,40,1,1).data[3];
samples.push({frame:s.frame,status:document.getElementById('hp-status').textContent,alpha,loaded:d.loaded.size,pending:d.pending.size,width:pill.getBoundingClientRect().width});}
vid.pause();return {samples,width};})()""")
            samples=result['samples']
            assert len({s['frame']//30 for s in samples})>=7
            assert all(s['status']=='' and s['alpha']>0 for s in samples),[s for s in samples if s['status'] or not s['alpha']][:15]
            assert max(s['loaded'] for s in samples)<=5 and max(s['pending'] for s in samples)<=3
            assert all(s['width']==result['width'] for s in samples)
            # A far seek discards old requests and never draws a stale source frame.
            js("vid.currentTime=.1")
            for _ in range(100):
                if js("!vid.seeking && window._hp.state.frame===3"):break
                time.sleep(.02)
            assert js('window._hp.state.frame')==3
            assert js('_hpData.get(_activeFile).pending.size')>0
            assert js("document.getElementById('hp-canvas').getContext('2d').getImageData(46,40,1,1).data[3]")==0
            js("document.querySelector('[data-mode=off]').click()")
            assert js('_hpData.get(_activeFile).pending.size')==0
            assert js('window._hp.state.active') is False
            js("document.querySelector('[data-mode=skeleton]').click();window._epCleanup.forEach(f=>f());testAbort.abort()")
            assert js('_hpData.get(_activeFile).pending.size')==0
        finally:
            close();server.shutdown();server.server_close()
