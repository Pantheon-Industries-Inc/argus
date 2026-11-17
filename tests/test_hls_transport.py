"""Optional HLS uses bounded buffers and the episode's existing video timeline."""
import subprocess


def test_hls_attachment_limits_transfer_and_abort():
    subprocess.run(['node','-e',r'''
const assert=require('node:assert/strict'),{piece}=require('./tests/page_functions')();
const scripts=[],instances=[];let _hlsLibrary=null;
const document={createElement:()=>({remove(){}}),head:{appendChild(s){scripts.push(s);}}},window={};
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('"','&quot;');
class FakeHls{static isSupported(){return true;}constructor(config){this.config=config;instances.push(this);}attachMedia(v){this.media=v;}loadSource(url){this.url=url;}destroy(){this.destroyed=(this.destroyed||0)+1;}}
function video(native=false){const attrs=new Map();return {currentTime:23.5,paused:false,dataset:{},canPlayType:()=>native?'probably':'',getAttribute:k=>attrs.get(k),set src(v){attrs.set('src',v)},get src(){return attrs.get('src')},removeAttribute:k=>attrs.delete(k),pause(){this.paused=true;},load(){},dispatchEvent(){}};}
eval(piece('isHlsSource')+piece('videoSourceAttr')+piece('loadHlsLibrary')+piece('attachVideoSource')+piece('retainVideoSource'));
(async()=>{assert.equal(videoSourceAttr('clip.mp4'),'src="clip.mp4"');assert.ok(videoSourceAttr('clip.M3U8?x=1&y=2').startsWith('data-hls-src='));assert.equal(isHlsSource('clip.mp4?next=x.m3u8'),false);
const mp4=video();await attachVideoSource(mp4,'clip.mp4',new AbortController().signal);assert.equal(mp4.src,'clip.mp4');assert.equal(scripts.length,0);
const v=video(),old=new AbortController(),ready=attachVideoSource(v,'clip.m3u8',old.signal);assert.equal(scripts.length,1);assert.equal(scripts[0].crossOrigin,'anonymous');assert.ok(scripts[0].integrity.startsWith('sha384-'));assert.ok(scripts[0].src.includes('hls.js@1.6.19/'));
window.Hls=FakeHls;scripts[0].onload();await ready;assert.equal(instances.length,1);assert.equal(instances[0].url,'clip.m3u8');assert.equal(instances[0].media,v);
assert.deepEqual(instances[0].config,{maxBufferLength:8,maxMaxBufferLength:12,backBufferLength:4,maxBufferSize:6*1024*1024});
retainVideoSource(v);old.abort();assert.equal(instances[0].destroyed,undefined);
const current=new AbortController();await attachVideoSource(v,'clip.m3u8',current.signal);assert.equal(instances.length,1);assert.equal(v.currentTime,23.5);assert.equal(v.paused,false);current.abort();assert.equal(instances[0].destroyed,1);
const pending=video(),aborted=new AbortController();aborted.abort();await attachVideoSource(pending,'other.m3u8',aborted.signal);assert.equal(instances.length,1);
})().catch(e=>{console.error(e);process.exit(1)});
'''],check=True)


def test_hls_native_fallback_and_pending_cleanup():
    subprocess.run(['node','-e',r'''
const assert=require('node:assert/strict'),{piece}=require('./tests/page_functions')();let _hlsLibrary=null;
const scripts=[],document={createElement:()=>({remove(){}}),head:{appendChild(s){scripts.push(s);}}},window={};
function video(){return {src:'',canPlayType:()=> 'probably',removeAttribute(){this.src=''},pause(){},load(){},dispatchEvent(){}};}
eval(piece('isHlsSource')+piece('loadHlsLibrary')+piece('attachVideoSource'));
(async()=>{const first=video(),stop=new AbortController(),ready=attachVideoSource(first,'first.m3u8',stop.signal);stop.abort();window.Hls={isSupported:()=>false};scripts[0].onload();await ready;assert.equal(first.src,'');
const second=video(),owner=new AbortController();await attachVideoSource(second,'second.m3u8',owner.signal);assert.equal(second.src,'second.m3u8');owner.abort();assert.equal(second.src,'');
})().catch(e=>{console.error(e);process.exit(1)});
'''],check=True)


def test_hls_browser_preserves_passthrough_pts_seek_and_bounded_buffer(tmp_path):
    import base64
    import functools
    import hashlib
    import http.server
    import json
    from pathlib import Path
    import shutil
    import tempfile
    import threading
    import time
    import urllib.error
    import urllib.request
    import pytest
    from test_layout_fits import chrome, _devtools
    if not chrome() or not shutil.which('ffmpeg'):
        pytest.skip('Chrome and ffmpeg required')
    try:
        sdk=urllib.request.urlopen('https://cdn.jsdelivr.net/npm/hls.js@1.6.19/dist/hls.min.js',timeout=15).read()
    except (OSError,urllib.error.URLError):
        pytest.skip('Pinned HLS SDK unavailable')
    assert base64.b64encode(hashlib.sha384(sdk).digest()).decode()=='eIepVstlIv9sSqP0mAsB37yZCJzcs0Y9qhxne7qDj9mO6EneOgR3uvZy28Xn6XqH'
    (tmp_path/'sdk.js').write_bytes(sdk)
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=s=64x48:r=10:d=24',
                    '-c:v','libx264','-g','20','-bf','0','-pix_fmt','yuv420p','-video_track_timescale','1000000',str(tmp_path/'clip.mp4')],check=True)
    subprocess.run(['ffmpeg','-v','error','-copyts','-i',str(tmp_path/'clip.mp4'),'-c','copy','-hls_time','1.9',
                    '-hls_segment_type','fmp4','-hls_flags','independent_segments','-hls_list_size','0',
                    '-hls_segment_options','video_track_timescale=1000000','-hls_segment_filename',str(tmp_path/'part-%03d.m4s'),str(tmp_path/'index.m3u8')],check=True)
    frames=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','json',str(tmp_path/'clip.mp4')]))['frames']
    expected=[float(f['best_effort_timestamp_time'])for f in frames]
    source=Path('board/index.html').read_text()
    helpers=source[source.index('function isHlsSource('):source.index('function posterAttr(')]
    page='''<video id="mp4" src="clip.mp4" muted controls></video><video id="hls" data-hls-src="index.m3u8" muted controls></video><script>
const append=document.head.appendChild.bind(document.head);document.head.appendChild=s=>{if(s.src.includes('hls.js@1.6.19/'))s.src='sdk.js';return append(s);};
const owner=new AbortController();
'''+helpers+"attachVideoSources(document,owner.signal);</script>"
    (tmp_path/'page.html').write_text(page)
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self,*args):pass
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Quiet,directory=str(tmp_path)))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    with tempfile.TemporaryDirectory()as profile:
        send,close=_devtools(chrome(),['--headless=new','--no-sandbox','--autoplay-policy=no-user-gesture-required',f'--user-data-dir={profile}','about:blank'])
        try:
            tid=send('Target.createTarget',{'url':f'http://127.0.0.1:{server.server_port}/page.html'})['targetId']
            sid=send('Target.attachToTarget',{'targetId':tid,'flatten':True})['sessionId']
            def js(code):
                result=send('Runtime.evaluate',{'expression':code,'returnByValue':True,'awaitPromise':True},sid,deadline=time.monotonic()+20)
                assert 'exceptionDetails'not in result,result
                return result['result'].get('value')
            deadline=time.monotonic()+15
            while not js("[...document.querySelectorAll('video')].every(v=>v.readyState>=2)"):
                assert time.monotonic()<deadline,'HLS did not load'
                time.sleep(.05)
            result=js("""(async()=>{const videos=[...document.querySelectorAll('video')],times={mp4:[],hls:[]};let measure=true;
for(const v of videos){const tick=(now,md)=>{if(measure){times[v.id].push(md.mediaTime);v.requestVideoFrameCallback(tick)}};v.requestVideoFrameCallback(tick);}
await Promise.all(videos.map(v=>v.play()));await new Promise(r=>setTimeout(r,2100));measure=false;videos.forEach(v=>v.pause());
const h=document.getElementById('hls'),ahead=h.buffered.end(h.buffered.length-1)-h.currentTime;
const sought=new Promise(r=>h.requestVideoFrameCallback((now,md)=>r(md.mediaTime)));h.currentTime=18.4;
const seekTime=await sought;owner.abort();return {times,ahead,seekTime,disposed:!h._hlsTransport,source:h.getAttribute('src')};})()""")
            for mode,times in result['times'].items():
                assert len(times)>=8 and times[-1]-times[0]>=1.5,(mode,times)
                assert all(min(abs(t-want)for want in expected)<1e-6 for t in times),(mode,times)
            assert result['ahead']<=14,'HLS buffered beyond its cap plus one fragment'
            assert abs(result['seekTime']-18.4)<1e-6
            assert result['disposed'] and result['source'] is None
        finally:
            close();server.shutdown();server.server_close()
