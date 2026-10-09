// The cameras the model is not shown, on the episode page (board/serve.py unshownCellsHtml, unshownNote): each plays in
// a cell of its own beside the other cameras, named as not shown to the model, and one line under the cameras says why
// for each. A board file without any draws nothing.
//
//   node tests/unshown_cameras.js [PAGE_SOURCE]     (default board/serve.py)
//
// Prints nothing and exits 0 when every case holds; prints each failure and exits 1 otherwise.
'use strict';
const {src, piece} = require('./page_functions')(process.argv[2]);
const stubs = 'const videoSrc = (e, c) => "api/video?id=" + e + "&cam=" + c; const posterAttr = () => "";';
const T = new Function(stubs + piece('esc') + piece('unshownCams') + piece('unshownCellsHtml') + piece('unshownNote')
  + 'return {unshownCams, unshownCellsHtml, unshownNote};')();

let bad = 0;
const check = (ok, what) => { if (!ok) { bad++; console.log('FAIL: ' + what); } };

const d = {unshown_cameras: [{view: 'unshown1', name: 'cam_ir', why: 'an infrared video'},
                             {view: 'unshown2', name: 'cam <low>', why: 'more extra cameras than the model is shown'},
                             {name: 'no view'}]};
check(T.unshownCams({}).length === 0 && T.unshownCellsHtml({}, 'e') === '' && T.unshownNote({}) === '',
  'no camera left out, nothing drawn');
check(T.unshownCams(d).length === 2, 'only entries with a view');
const cells = T.unshownCellsHtml(d, 'ep1');
check((cells.match(/class="cam-cell cam-wrist cam-unshown"/g) || []).length === 2, 'one cell per camera');
check(['unshown1', 'unshown2'].every(view => cells.includes(`id="video-${view}"`)
  && cells.includes(`src="api/video?id=ep1&amp;cam=${view}"`)),
  'each side camera retains its own source with the query escaped for HTML');
check(cells.includes('cam_ir, not shown to the model') && cells.includes('cam &lt;low&gt;, not shown to the model'),
  'each is named as not shown to the model');
const note = T.unshownNote(d);
check(note.includes('cam_ir') && note.includes('an infrared video') && note.includes('cam &lt;low&gt;')
  && note.includes('more extra cameras than the model is shown'), 'one line says why for each');
process.exit(bad ? 1 : 0);
