// Load actual page functions for focused behavior checks.
'use strict';
const fs = require('fs');
const path = require('path');
const vm = require('vm');

module.exports = function loadPageFunctions(filename) {
  let sourcePath = filename || path.join(__dirname, '..', 'board', 'index.html');
  let src = fs.readFileSync(sourcePath, 'utf8');
  if (path.basename(sourcePath) === 'serve.py' && !src.includes('\nfunction episodeCams('))
    src = fs.readFileSync(path.join(path.dirname(sourcePath), 'index.html'), 'utf8');
  const piece = name => {
    let start = src.indexOf('\nfunction ' + name + '(');
    if (start < 0) start = src.indexOf('\nasync function ' + name + '(');
    if (start < 0) throw new Error('No function ' + name + ' in the page');
    // Let the JavaScript parser handle braces in strings, comments and templates.
    for (let end = src.indexOf('}', start); end >= 0; end = src.indexOf('}', end + 1)) {
      const code = src.slice(start, end + 1) + '\n';
      try {
        new vm.Script(code);
        return code;
      } catch (error) {
        if (!(error instanceof SyntaxError)) throw error;
      }
    }
    throw new Error('Unclosed function ' + name + ' in the page');
  };
  return {src, piece};
};
