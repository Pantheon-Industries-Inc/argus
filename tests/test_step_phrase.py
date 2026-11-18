"""A step shown on the board names its object: the object is added after the model's phrase when the phrase does not
is only verbs. Fable 5.2 wrote bare verbs ("carry", "release") with the object only in its own column, and the board
showed "carry" alone (2026-11-18). The object goes where English puts it, so "carry to centre" with a playing card
reads "carry the playing card to centre", not "carry to centre playing card"."""
import subprocess

SCRIPT = r'''
const assert = require('node:assert/strict');
const {piece} = require('./tests/page_functions')();
const code = ['esc', 'stepPhrase', 'buildPhrase'].map(piece).join('\n') + '\nmodule.exports = {stepPhrase, buildPhrase};';
const m = {exports: {}};
new Function('module', code)(m);
const {stepPhrase, buildPhrase} = m.exports;
// Fable's bare verbs from the 10K UMI take fob_47b204-56c6708d-1680-139-lr gain their object where English puts it:
// before the first preposition, or last when there is none
assert.equal(stepPhrase({verb_class: 'carry to centre', object: 'playing card'}), 'carry the playing card to centre');
assert.equal(stepPhrase({verb_class: 'release', object: 'chess piece'}), 'release the chess piece');
assert.equal(stepPhrase({verb_class: 'reach and grasp', object: 'playing card'}), 'reach and grasp the playing card');
assert.equal(stepPhrase({verb_class: 'lay flat and release', object: 'playing card'}), 'lay flat and release the playing card');
// an object with its own determiner keeps it
assert.equal(stepPhrase({verb_class: 'stack', object: 'two cups'}), 'stack two cups');
// a phrase with a noun of its own is shown as written, whatever the object column says (real Sol steps)
assert.equal(stepPhrase({verb_class: 'grasp the two handles', object: 'rolling pin'}), 'grasp the two handles');
assert.equal(stepPhrase({verb_class: 'tip carton onto broad face', object: 'SP//dr Suit model-kit box'}), 'tip carton onto broad face');
assert.equal(stepPhrase({verb_class: 'final pulls on knot', object: 'black wire'}), 'final pulls on knot');
assert.equal(stepPhrase({verb_class: 'Grasp next piece', object: 'spare miniature brick'}), 'Grasp next piece');
// the gripper's own motion, and a particle, are shown as written
assert.equal(stepPhrase({verb_class: 'return to cup', object: 'paper cup'}), 'return to cup');
assert.equal(stepPhrase({verb_class: 'reach for red connector', object: 'small red connector'}), 'reach for red connector');
assert.equal(stepPhrase({verb_class: 'release and pull back', object: 'right-side truck'}), 'release and pull back');
assert.equal(stepPhrase({verb_class: 'close on small connector', object: 'small toy connector'}), 'close on small connector');
// a phrase that already names the object, as Sol writes them, is shown unchanged
assert.equal(stepPhrase({verb_class: 'release chess piece to right', object: 'wooden chess piece'}),
             'release chess piece to right');
assert.equal(stepPhrase({verb_class: 'Throw ball toward jar', object: 'orange table-tennis ball'}), 'Throw ball toward jar');
assert.equal(stepPhrase({verb_class: 'release and withdraw', object: 'loose piece'}), 'release and withdraw');
// no object, or no phrase
assert.equal(stepPhrase({verb_class: 'withdraw', object: null}), 'withdraw');
assert.equal(stepPhrase({verb_class: '', object: 'rolling pin'}), 'handle the rolling pin');
assert.equal(stepPhrase({}), 'move');
// the caption uses the same text, and the destination stays its own chip
const html = buildPhrase({verb_class: 'carry', object: 'soccer ball', carry_phase: '-> to back right'});
assert.match(html, /<span class="verb">carry the soccer ball<\/span>/);
assert.match(html, /dest-chip"><span class="prep">to<\/span> back right/);
'''


def test_a_step_names_its_object_when_the_phrase_does_not():
    subprocess.run(['node', '-e', SCRIPT], check=True)
