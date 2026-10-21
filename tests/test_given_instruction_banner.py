"""The banner over the dense timeline, run from the page's own source in node: a recording that carries an instruction
shows it above the model's reading in either mode (a session's summary or an episode's assessment), and one with none
shows the model's reading alone."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[1] / 'board' / 'index.html'

SCRIPT = r'''
const fs = require('fs');
const s = fs.readFileSync(process.argv[1], 'utf8');
const cut = (a, b) => { const i = s.indexOf(a), j = s.indexOf(b, i); if (i < 0 || j < i) throw new Error(a); return s.slice(i, j); };
const fn = name => cut(`function ${name}(`, '\n}\n') + '\n}\n';
const esc = x => String(x ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;');
const src = fn('nameParts') + fn('whoOwn') + cut('  // Given or inferred. When the dataset ships', '  // Key events (milestone layer), sorted.');
const banner = new Function('d', 'meta', 'hasTasks', 'who', 'esc', src + 'return {bannerLabel, bannerBody};');
const cases = JSON.parse(process.argv[2]);
console.log(JSON.stringify(cases.map(c => banner(c.d, c.meta, c.hasTasks, 'Astra', esc))));
'''

CASES = [
    # a session labelled from a recording with an instruction: the instruction, then the summary
    {'d': {'episode_prompt': 'packs four toys', 'goal_alignment': {'relation': 'aligned', 'matches_given': True}},
     'meta': {'prompt_mode': 'session', 'given_prompt': 'pack the toys into the box'}, 'hasTasks': True},
    # an episode labelled against its instruction: the instruction, then the model's own assessment
    {'d': {'episode_prompt': 'stacks the cups'}, 'meta': {'prompt_mode': 'given', 'given_prompt': 'stack the cups'},
     'hasTasks': False},
    # a session with no instruction: the summary alone
    {'d': {'episode_prompt': 'wipes the table'}, 'meta': {'prompt_mode': 'session'}, 'hasTasks': True},
]


@pytest.mark.skipif(not shutil.which('node'), reason='no node')
def test_a_recorded_instruction_shows_above_the_models_reading_in_either_mode():
    p = subprocess.run(['node', '-e', SCRIPT, str(PAGE), json.dumps(CASES)], text=True, capture_output=True)
    assert p.returncode == 0, p.stderr
    session, episode, bare = json.loads(p.stdout)
    assert session['bannerLabel'] == 'Given task and session summary'
    assert session['bannerBody'].index('pack the toys into the box') < session['bannerBody'].index('packs four toys')
    assert 'matches given goal' in session['bannerBody']
    assert episode['bannerBody'].index('stack the cups') < episode['bannerBody'].index('stacks the cups')
    assert 'given' not in bare['bannerBody'] and 'wipes the table' in bare['bannerBody']
    assert bare['bannerLabel'] == 'Session summary'
