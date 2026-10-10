// Load actual page functions for focused behavior checks.
'use strict';
const fs = require('fs');
const path = require('path');

module.exports = function loadPageFunctions(filename) {
  const src = fs.readFileSync(filename || path.join(__dirname, '..', 'board', 'serve.py'), 'utf8');
  const piece = name => {
    const start = src.indexOf('\nfunction ' + name + '(');
    if (start < 0) throw new Error('No function ' + name + ' in the page');
    let end = src.indexOf('{', start), depth = 0;
    for (; end < src.length; end++) {
      if (src[end] === '{') depth++;
      else if (src[end] === '}' && --depth === 0) break;
    }
    if (end === src.length) throw new Error('Unclosed function ' + name + ' in the page');
    return src.slice(start, end + 1) + '\n';
  };
  return {src, piece};
};
