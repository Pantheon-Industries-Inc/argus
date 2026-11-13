"""Default headings and hand controls use the viewer's shared presentation."""
import subprocess
import tempfile
import time

import pytest

from board import serve


def test_default_title_does_not_repeat_the_board_name():
    page = serve.render_index('Example', {'mode': 'api'}, 'Example')
    assert '<h1>Example</h1>' in page
    assert '<span class="ph-board">Example</span>' not in page
    distinct = serve.render_index('Data Dashboard', {'mode': 'api'}, 'Example')
    assert '<span class="ph-board">Example</span>' in distinct


def test_legacy_attribution_remains_default_and_supplied_geometry_is_generic():
    legacy = serve.render_index('Example', {'mode': 'api', 'hands': True})
    assert 'ACE-Ego-Hand' in legacy and 'Non-commercial use only.' in legacy
    geometry = serve.render_index('Example', {'mode': 'static', 'hands': True, 'hand_geometry': True})
    assert 'ACE-Ego-Hand' not in geometry
    assert 'Non-commercial use only.' not in geometry.split('<script>', 1)[0]
    assert 'id="kp-size"' in geometry


def test_single_dataset_heading_is_not_repeated_and_multi_dataset_tabs_remain():
    script = r'''
const assert = require('node:assert/strict');
const {piece} = require('./tests/page_functions')();
const elements = {body:{classList:{toggle(k,v){this[k]=v;}}},
  coverage:{classList:{toggle(k,v){this[k]=v;}}}};
const document = {body:elements.body, querySelectorAll:()=>[{textContent:'Example'}]};
const coverageEl=elements.coverage;
const DS_SHORT={}, dsLabel=x=>x;
eval(piece('renderDatasetHeadings'));
renderDatasetHeadings([{d:'Example'}]);
assert.equal(elements.body.classList['single-dataset'],true);
assert.equal(elements.body.classList['duplicate-dataset-heading'],true);
renderDatasetHeadings([{d:'Example'},{d:'Another'}]);
assert.equal(elements.body.classList['single-dataset'],false);
assert.equal(elements.body.classList['duplicate-dataset-heading'],false);
renderDatasetHeadings([{d:'Different dataset'}]);
assert.equal(elements.body.classList['duplicate-dataset-heading'],false);
'''
    subprocess.run(['node', '-e', script], check=True)


def test_keypoint_download_terms_follow_the_supplied_format():
    subprocess.run(['node', '-e', r'''
const assert=require('node:assert/strict'),{piece}=require('./tests/page_functions')();
const BOARD={hand_geometry:false},_hpData=new Map(),_activeFile='episode.json',fmtBytes=x=>x+'B';
eval(piece('hpKeypointsTitle'));
const kp={frames:10,bytes:100};
assert.ok(hpKeypointsTitle(kp).includes('Non-commercial use only.'));
assert.ok(!hpKeypointsTitle(kp,true).includes('Non-commercial'));
BOARD.hand_geometry=true;
assert.ok(!hpKeypointsTitle(kp).includes('Non-commercial'));
BOARD.hand_geometry=false;_hpData.set(_activeFile,{geometry:true});
assert.ok(!hpKeypointsTitle(kp).includes('Non-commercial'));
'''], check=True)


def test_optional_mode_pill_matches_canonical_font_and_layout():
    from pathlib import Path
    import re
    from test_layout_fits import chrome, _devtools
    if not chrome():
        pytest.skip('Chrome required')
    source = Path('board/index.html').read_text()
    match = re.search(r'<button id="hp-mode"[\s\S]*?</button>', source)
    assert match, 'Supplied geometry needs one canonical three-state hand pill'
    css = source.split('<style>', 1)[1].split('</style>', 1)[0]
    headings = source.split('function renderDatasetHeadings', 1)[1].split('function renderCoverage', 1)[0]
    html = '<style>'+css+'</style><header class="page-head"><h1>Example</h1></header>'
    html += '<nav class="coverage" id="coverage"><span class="cv-name">Example</span></nav><span id="rd-name">Example</span>'
    html += '<aside class="left"><div class="ep-head"><div class="ep-head-side"><div class="ep-head-acts">'+match[0]+'<a class="ep-head-dl" id="reference">Episode JSON</a></div></div></div>'
    html += '<div class="cam-row"><div class="cam-cell cam-exo" id="cell"><video width="160" height="90"></video><div class="video-overlay active">Current action</div></div>'
    html += '<div class="cam-cell cam-wrist"><video width="160" height="90"></video></div><div class="cam-cell cam-wrist"><video width="160" height="90"></video></div></div></aside>'
    html += '<script>document.getElementById("hp-mode").hidden=false;const DS_SHORT={},dsLabel=x=>x,coverageEl=document.getElementById("coverage");function renderDatasetHeadings'+headings
    html += 'renderDatasetHeadings([{d:"Example"}]);</script>'
    with tempfile.TemporaryDirectory() as profile:
        send, close = _devtools(chrome(), ['--headless=new','--no-sandbox',f'--user-data-dir={profile}','about:blank'])
        try:
            target=send('Target.createTarget',{'url':'about:blank'})['targetId']
            sid=send('Target.attachToTarget',{'targetId':target,'flatten':True})['sessionId']
            send('Page.enable',{},sid)
            frame=send('Page.getFrameTree',{},sid)['frameTree']['frame']['id']
            send('Page.setDocumentContent',{'frameId':frame,'html':html},sid)
            def js(code):
                return send('Runtime.evaluate',{'expression':code,'returnByValue':True},sid,deadline=time.monotonic()+15)['result']['value']
            result=js("""(() => {const b=document.getElementById('hp-mode'),s=getComputedStyle(b);
const visible=x=>!!document.querySelector(x).getClientRects().length;
const single=[visible('#coverage'),visible('.cv-name'),visible('#rd-name')];
renderDatasetHeadings([{d:'Example'},{d:'Another'}]);
return {font:s.font,reference:getComputedStyle(document.getElementById('reference')).font,
background:s.backgroundColor,label:document.getElementById(b.getAttribute('aria-labelledby')).textContent,
role:b.getAttribute('role'),single,multi:[visible('#coverage'),visible('.cv-name'),visible('#rd-name')]};})()""")
            assert result['font']==result['reference']
            assert result['background']=='rgba(0, 0, 0, 0)'
            assert result['label']=='Hand pose' and result['role']=='slider'
            assert js("[...document.querySelectorAll('#hp-mode [data-mode]')].map(e=>e.textContent)")==['Off','Skeleton','Mesh']
            thumb=[]
            for value in range(3):
                thumb.append(js("""(() => {const b=document.getElementById('hp-mode');b.setAttribute('aria-valuenow','VALUE');
const style=document.createElement('style');style.textContent='.hp-track::before{transition:none}';document.head.append(style);
const t=document.querySelector('.hp-track'),s=getComputedStyle(t,'::before');
return {x:new DOMMatrix(s.transform).m41,width:parseFloat(s.width),track:t.getBoundingClientRect().width,
labels:[...t.children].map(e=>({width:e.getBoundingClientRect().width,color:getComputedStyle(e).color}))};})()""".replace('VALUE',str(value))))
            assert thumb[0]['x']==0 and thumb[0]['x']<thumb[1]['x']<thumb[2]['x']
            assert abs(thumb[2]['x']-2*thumb[0]['width'])<1
            assert all(abs(t['width']*3-(t['track']-4))<1 for t in thumb)
            assert all(max(e['width'] for e in t['labels'])-min(e['width'] for e in t['labels'])<1 for t in thumb), thumb
            assert result['single']==[False,False,False] and result['multi']==[True,True,True]
            for width in (390,280):
                send('Emulation.setDeviceMetricsOverride',{'width':width,'height':900,'deviceScaleFactor':1,'mobile':False},sid)
                layout=js("""(() => {const p=document.getElementById('hp-mode').getBoundingClientRect(),v=document.querySelector('#cell video').getBoundingClientRect(),c=document.querySelector('.video-overlay').getBoundingClientRect();return {p:p.toJSON(),v:v.toJSON(),c:c.toJSON(),width:innerWidth,scroll:document.documentElement.scrollWidth};})()""")
                assert layout['scroll']==width
                assert layout['p']['left']>=0 and layout['p']['right']<=width
                assert layout['v']['top']>=layout['p']['bottom']-1
                assert layout['c']['height']<=90
        finally:
            close()
