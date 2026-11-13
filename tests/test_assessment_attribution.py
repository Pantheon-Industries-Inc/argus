"""Public assessment copy is configurable without changing internal attribution."""
import json
import re
import subprocess
from board.serve import render_index


def config(page):
    return json.loads(re.search(r'const BOARD = (.*?);', page).group(1))


def test_public_attribution_omits_model_configuration():
    cfg = config(render_index('Sample', {'mode':'static', 'model_attribution':False, 'compare':True}))
    assert cfg['models'] == {}
    assert cfg.get('reasoning') is None
    assert cfg['compare'] is False


def test_internal_attribution_keeps_model_configuration():
    cfg = config(render_index('Sample', {'mode':'static', 'models':{'internal/test':'Test assessor'}}))
    assert cfg['models']['internal/test'] == 'Test assessor'


def test_public_copy_and_internal_copy_are_distinct():
    subprocess.run(['node','-e',r'''
const assert=require('node:assert/strict'),{piece,src}=require('./tests/page_functions')();
let BOARD={model_attribution:false,models:{'vendor/id':'Named model'}};
const esc=s=>String(s).replaceAll('&','&amp;');
eval(piece('modelName')+piece('nameParts')+piece('whoOwn')+piece('cmpFailHtml'));
assert.equal(modelName('vendor/id'),'we');
assert.equal(whoOwn('Named model','independent assessment'),'our independent assessment');
assert.ok(cmpFailHtml({status:'no_reply'},'Named model',{est_cost_usd:1}).includes('Assessment unavailable'));
assert.ok(!cmpFailHtml({status:'no_reply'},'Named model',{est_cost_usd:1}).includes('Named model'));
BOARD.model_attribution=true;
assert.equal(modelName('vendor/id'),'Named model');
assert.equal(whoOwn('Named model','tasks'),'Named model&rsquo;s tasks');
assert.ok(src.includes('>Success '));
assert.ok(!src.includes("${esc(who)}'s success criterion"));
'''],check=True)
