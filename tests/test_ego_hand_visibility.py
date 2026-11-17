"""Future Ego visibility is per hand; historical labels keep their existing cue."""
import copy
import subprocess

from board import label_projection, to_board
from board.families import Families
from label import harness, pieces, prompts


def test_ego_prompt_requests_both_hands_independently():
    text = prompts.fixed_instructions('ego_head', has_instruction=False, recorded=False)
    assert 'left_hand_visible' in text and 'right_hand_visible' in text
    assert 'regardless of which hand is acting' in text
    assert 'partly visible' in text
    assert 'visibility changes' in text
    for rig in ('teleop_arms', 'handheld_gripper'):
        other = prompts.fixed_instructions(rig, has_instruction=False, recorded=False)
        assert 'left_hand_visible' not in other and 'right_hand_visible' not in other


def test_visibility_survives_parsing_projection_and_board_export():
    labels = {'timeline_columns':['start_s','end_s','arm','action','hands_visible',
                                 'left_hand_visible','right_hand_visible','notes'],
              'timeline':[[0,1,'left','release remote',True,True,False,None]]}
    parsed = harness.normalize_timeline(copy.deepcopy(labels))
    projected, dropped = label_projection.project(parsed)
    assert not dropped
    event = to_board.convert({'labels':projected})['event_labels'][0]
    assert event['left_hand_visible'] is True and event['right_hand_visible'] is False
    assert {'left_hand_visible','right_hand_visible'} <= set(pieces.STEP_TEXT_FIELDS)


def test_invalid_visibility_stays_unknown_and_legacy_export_stays_legacy():
    labels = {'timeline':[{'start_s':0,'end_s':1,'arm':'both','action':'wait',
                           'left_hand_visible':'false','right_hand_visible':0}]}
    projected, dropped = label_projection.project(labels)
    assert projected['timeline'][0]['left_hand_visible'] is None
    assert projected['timeline'][0]['right_hand_visible'] is None
    assert len(dropped) >= 2
    old = {'labels':{'timeline':[{'start_s':0,'end_s':1,'hands_visible':False}]}}
    event = to_board.convert(old)['event_labels'][0]
    assert event['hands_visible'] is False
    assert 'left_hand_visible' not in event and 'right_hand_visible' not in event


def test_canonical_badge_labels_and_legacy_fallback():
    subprocess.run(['node','-e',r'''
const assert=require('node:assert/strict'),{piece}=require('./tests/page_functions')();
eval(piece('handVisibilityLabel'));
for(const [event,expected] of [
 [{left_hand_visible:false,right_hand_visible:true},'Left hand out of view'],
 [{left_hand_visible:true,right_hand_visible:false},'Right hand out of view'],
 [{left_hand_visible:false,right_hand_visible:false},'Both hands out of view'],
 [{left_hand_visible:true,right_hand_visible:true},''],
 [{left_hand_visible:null,right_hand_visible:null,hands_visible:false},''],
 [{left_hand_visible:false,right_hand_visible:null},'Left hand out of view'],
 [{hands_visible:false},'Both hands out of view'],[{hands_visible:true},''],[null,''],
 [{arm:'left'},''],[{hands_visible:true,pose_detected:false},'']
])assert.equal(handVisibilityLabel(event),expected);
'''],check=True)


def test_badge_updates_between_sides_and_clears_without_changing_style():
    subprocess.run(['node', '-e', r'''
const assert = require('node:assert/strict'), {src, piece} = require('./tests/page_functions')();
const classes = new Set();
const esHand = {textContent:'', classList:{
  add(...names) { names.forEach(n => classes.add(n)); },
  remove(...names) { names.forEach(n => classes.delete(n)); }
}};
const exoCell = {classList:{toggle(name, on) { this[name] = on; }}};
const isEgo = true, segWindows = [], isBare = w => !w || w === 'bare';
const events = [
 {left_hand_visible:false, right_hand_visible:true},
 {left_hand_visible:true, right_hand_visible:false},
 {left_hand_visible:false, right_hand_visible:false},
 {left_hand_visible:true, right_hand_visible:true},
 {hands_visible:false},
 {left_hand_visible:null, right_hand_visible:null, hands_visible:false},
 {left_hand_visible:true, right_hand_visible:true, hands_wearing:'work glove'}
];
const captionSteps = (_, t) => [{t0:t, ev:events[t]}];
let _handSig = null;
eval(piece('handVisibilityLabel'));
const start = src.indexOf('  function renderHands(t) {');
const end = src.indexOf('\n  // ----- a session of tasks:', start);
assert.ok(start > 0 && end > start);
eval(src.slice(start, end));
for (const [t, label] of ['Left hand out of view','Right hand out of view','Both hands out of view',
                         '', 'Both hands out of view', '', 'Wearing work glove'].entries()) {
  renderHands(t);
  assert.equal(classes.has('show'), !!label);
  assert.equal(classes.has('hidden-hands'), label.endsWith('out of view'));
  assert.equal(exoCell.classList['hands-hidden'], label.endsWith('out of view'));
  if (label) assert.equal(esHand.textContent, label);
}
'''], check=True)


def test_stitch_preserves_visibility_changes_at_piece_boundaries():
    before = {'start_s':0, 'end_s':5, 'arm':'left', 'action':'carry',
              'left_hand_visible':True, 'right_hand_visible':False}
    after = {**before, 'start_s':5, 'end_s':10, 'right_hand_visible':True}
    origin = {id(before):{**before, 'end_s':6}, id(after):{**after, 'start_s':4}}
    windows = {5:{'t0_s':4, 't1_s':6}}
    assert len(pieces.join_steps_across_cuts([before, after], windows, origin)) == 2
    same = {**after, 'right_hand_visible':False}
    origin[id(same)] = {**same, 'start_s':4}
    joined = pieces.join_steps_across_cuts([before, same], windows, origin)
    assert len(joined) == 1 and joined[0]['right_hand_visible'] is False
