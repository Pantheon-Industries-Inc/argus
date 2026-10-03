import json,subprocess
import pytest
from board import static

@pytest.mark.parametrize('t',[(2**52+1)/1000,(2**52+3)/1000, (2**52)/1000,0.123456789123])
def test_every_admitted_goal_has_the_same_browser_lookup_key(t):
    saved={'completion':{'completed_at_s':t}}
    planned=static.goal_times(saved)
    browser=json.loads(subprocess.check_output(['node','-e',
        'console.log(JSON.stringify(JSON.parse(process.argv[1]).map(t=>String(Math.round(t*1000)))))',
        json.dumps(planned)],text=True))
    assert [str(static.ms_key(value)) for value in planned]==browser

@pytest.mark.parametrize('field', ['completion', 'tasks', 'reached'])
@pytest.mark.parametrize('t', [(2**52+1)/1000, (2**52+3)/1000])
def test_saved_goal_with_a_different_browser_key_is_withheld(t, field):
    if field == 'completion':
        saved = {'completion': {'completed_at_s': t}}
    elif field == 'tasks':
        saved = {'tasks': [{'task': 'lift', 'start_s': 0, 'completed_at_s': t}]}
    else:
        saved = {'completion': {'task_completed': 'success_then_undone', 'goal_reached_at_s': t}}
    before = json.dumps(saved)
    assert static.goal_times(saved) == []
    assert json.dumps(saved) == before

@pytest.mark.parametrize('t', [(2**52)/1000, (2**52+2)/1000, 0.123456789123, 2.34567891])
def test_usable_existing_keys_remain_admitted(t):
    assert static.goal_times({'completion': {'completed_at_s': t}}) == [t]
