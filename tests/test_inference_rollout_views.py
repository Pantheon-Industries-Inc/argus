"""An inference rollout's episode file carries two things the board shows beside the model's reading (issue 183): the
rollout's ground truth in `_meta.rollout` (the instruction the policy was given, whether it succeeded, how long it ran,
how many action chunks it sent, the policy model and its effort) and the policy's latency in `idle` (how many action
chunks, the total time the robot held still while the policy computed them, and the first gaps). The episode as the
board serves it keeps both fields, and the page draws each from its own function, run here from the page's source in
node on one inference episode."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from board.serve import episode_view

PAGE = Path(__file__).resolve().parents[1] / 'board' / 'index.html'

ROLLOUT = {'instruction': 'put the red block in the bowl', 'model': 'policy-model-x', 'reasoning_effort': 'low',
           'status': 'succeeded', 'elapsed_s': 94.2, 'actions': 17}
IDLE = {'n': 17, 'total_idle_s': 61.5, 'thinking_gaps_s': [3.1, 4.4, 2.9, 5.0, 3.3]}
EPISODE = {'episode_prompt': 'places the red block in the bowl', 'event_labels': [], 'idle': IDLE,
           '_meta': {'episode_id': 'rollout_0001', 'rollout': ROLLOUT}}

SCRIPT = r'''
const fs = require('fs');
const s = fs.readFileSync(process.argv[1], 'utf8');
const cut = (a, b) => { const i = s.indexOf(a), j = s.indexOf(b, i); if (i < 0 || j < i) throw new Error(a); return s.slice(i, j); };
const fn = name => cut(`function ${name}(`, '\n}\n') + '\n}\n';
const esc = x => String(x ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;');
const views = new Function('esc', cut('function fmtDur(', '\n') + '\n' + fn('rolloutHtml') + fn('policyLatencyHtml')
  + 'return {rolloutHtml, policyLatencyHtml};')(esc);
const d = JSON.parse(process.argv[2]);
const meta = d._meta || {};
console.log(JSON.stringify({
  rollout: views.rolloutHtml(meta.rollout), latency: views.policyLatencyHtml(d.idle),
  plain_rollout: views.rolloutHtml(undefined), plain_latency: views.policyLatencyHtml(undefined),
  failed: views.rolloutHtml({...meta.rollout, status: 'failed'}), no_chunks: views.policyLatencyHtml({n: 0}),
  // the right column's template draws both, right under the banner with the model's reading
  in_column: /\$\{rolloutHtml\(meta\.rollout\)\}\s*\$\{policyLatencyHtml\(d\.idle\)\}/.test(cut('rightCol.innerHTML = failed', '<h3 class="section dense-head">')),
}));
'''


def test_the_served_episode_keeps_the_rollout_and_the_policy_latency():
    served = episode_view(json.loads(json.dumps(EPISODE)))
    assert served['_meta']['rollout'] == ROLLOUT
    assert served['idle'] == IDLE


@pytest.mark.skipif(not shutil.which('node'), reason='no node')
def test_an_inference_episode_shows_its_rollout_ground_truth_and_policy_latency():
    p = subprocess.run(['node', '-e', SCRIPT, str(PAGE), json.dumps(episode_view(json.loads(json.dumps(EPISODE))))],
                       text=True, capture_output=True)
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout)
    rollout, latency = out['rollout'], out['latency']
    # the instruction the policy was given, its result, time, chunks, model and effort
    assert 'put the red block in the bowl' in rollout and 'policy-model-x' in rollout
    assert 'succeeded' in rollout and 'pm-green' in rollout
    assert '1.6 min' in rollout and '>17<' in rollout and '>low<' in rollout
    assert 'pm-green' not in out['failed'] and 'failed' in out['failed']
    # the chunk count, the total held still, and the first four gaps only
    assert '>17<' in latency and '1.0 min' in latency
    assert [g in latency for g in ('>3s<', '>4s<', '>5s<')] == [True, True, True] and latency.count('gap') == 4
    # an episode that is not a rollout draws neither
    assert out['plain_rollout'] == '' and out['plain_latency'] == '' and out['no_chunks'] == ''
    assert out['in_column']
