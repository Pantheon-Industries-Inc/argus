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
    assert 'ACE-Ego-Hand' not in geometry and 'Non-commercial use only.' not in geometry
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
assert.equal(coverageEl.classList['duplicate-name'],true);
renderDatasetHeadings([{d:'Example'},{d:'Another'}]);
assert.equal(elements.body.classList['single-dataset'],false);
assert.equal(coverageEl.classList['duplicate-name'],false);
renderDatasetHeadings([{d:'Different dataset'}]);
assert.equal(coverageEl.classList['duplicate-name'],false);
'''
    subprocess.run(['node', '-e', script], check=True)


def test_hand_modes_share_canonical_button_font_without_a_camera_padding_bar():
    from pathlib import Path
    from test_layout_fits import chrome, _devtools
    if not chrome():
        pytest.skip('Chrome required')
    source = Path('board/index.html').read_text()
    css = source.split('<style>', 1)[1].split('</style>', 1)[0]
    modes = source.split('function hpModes()', 1)[1].split('\nfunction hpStatus', 1)[0]
    headings = source.split('function renderDatasetHeadings', 1)[1].split('function renderCoverage', 1)[0]
    html = '<style>'+css+'</style><header class="page-head"><h1>Example</h1></header>'
    html += '<nav class="coverage" id="coverage"><span class="cv-name">Example</span></nav><span id="rd-name">Example</span>'
    html += '<aside class="left"><div class="cam-cell cam-exo" id="cell"></div></aside>'
    html += '<a class="ep-head-dl" id="reference">Video</a><script>const HP_MODE="skeleton";function hpModes()'+modes
    html += 'document.getElementById("cell").innerHTML=hpModes()+"<video></video>";document.getElementById("hp-modes").hidden=false;'
    html += 'const DS_SHORT={},dsLabel=x=>x,coverageEl=document.getElementById("coverage");function renderDatasetHeadings'+headings
    html += 'renderDatasetHeadings([{d:"Example"}]);</script>'
    with tempfile.TemporaryDirectory() as profile:
        send, close = _devtools(chrome(), ['--headless=new', '--no-sandbox', f'--user-data-dir={profile}', 'about:blank'])
        try:
            target = send('Target.createTarget', {'url':'about:blank'})['targetId']
            sid = send('Target.attachToTarget', {'targetId':target, 'flatten':True})['sessionId']
            send('Page.enable', {}, sid)
            frame = send('Page.getFrameTree', {}, sid)['frameTree']['frame']['id']
            send('Page.setDocumentContent', {'frameId':frame, 'html':html}, sid)
            result = send('Runtime.evaluate', {'expression':'''(() => {
const buttons=[...document.querySelectorAll('.hp-modes button')];
const reference=getComputedStyle(document.getElementById('reference'));
const visible=selector=>!!document.querySelector(selector).getClientRects().length;
const single=[visible('.cv-name'),visible('#rd-name')];
renderDatasetHeadings([{d:'Example'},{d:'Another'}]);
return {fonts:buttons.map(b=>getComputedStyle(b).font),reference:reference.font,
 padding:getComputedStyle(document.getElementById('cell')).paddingTop,
 background:getComputedStyle(buttons[1]).backgroundColor,
 selected:buttons.map(b=>b.getAttribute('aria-pressed')),single,multi:[visible('.cv-name'),visible('#rd-name')]};})()''', 'returnByValue':True}, sid, deadline=time.monotonic()+15)['result']['value']
            assert result['fonts'] == [result['reference']] * 3
            assert result['padding'] == '0px'
            assert result['background'] == 'rgba(0, 0, 0, 0)'
            assert result['selected'] == ['false', 'true', 'false']
            assert result['single'] == [False, False]
            assert result['multi'] == [True, True]
        finally:
            close()
