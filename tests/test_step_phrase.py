"""A step shown on the board names its object: the object is added after the model's phrase when the phrase does not
already name it. Fable 5.2 wrote bare verbs ("carry", "release") with the object only in its own column, and the board
showed "carry" alone (2026-11-18)."""
import subprocess

SCRIPT = r'''
const assert = require('node:assert/strict');
const {piece} = require('./tests/page_functions')();
const code = ['esc', 'stepPhrase', 'buildPhrase'].map(piece).join('\n') + '\nmodule.exports = {stepPhrase, buildPhrase};';
const m = {exports: {}};
new Function('module', code)(m);
const {stepPhrase, buildPhrase} = m.exports;
// Fable's bare verbs from the 10K UMI take fob_47b204-56c6708d-1680-139-lr gain their object
assert.equal(stepPhrase({verb_class: 'carry to centre', object: 'soccer ball'}), 'carry to centre soccer ball');
assert.equal(stepPhrase({verb_class: 'release', object: 'chess piece'}), 'release chess piece');
// a phrase that already names the object, as Sol writes them, is shown unchanged
assert.equal(stepPhrase({verb_class: 'release chess piece to right', object: 'wooden chess piece'}),
             'release chess piece to right');
assert.equal(stepPhrase({verb_class: 'Throw ball toward jar', object: 'orange table-tennis ball'}), 'Throw ball toward jar');
// a word inside another word does not count as naming it ("pen" in "open")
assert.equal(stepPhrase({verb_class: 'open', object: 'pen'}), 'open pen');
// no object, or no phrase
assert.equal(stepPhrase({verb_class: 'withdraw', object: null}), 'withdraw');
assert.equal(stepPhrase({verb_class: '', object: 'rolling pin'}), 'handle the rolling pin');
assert.equal(stepPhrase({}), 'move');
// the caption uses the same text, and the destination stays its own chip
const html = buildPhrase({verb_class: 'carry', object: 'soccer ball', carry_phase: '-> to back right'});
assert.match(html, /<span class="verb">carry soccer ball<\/span>/);
assert.match(html, /dest-chip"><span class="prep">to<\/span> back right/);
'''


def test_a_step_names_its_object_when_the_phrase_does_not():
    subprocess.run(['node', '-e', SCRIPT], check=True)
