// Runs the board page's script as served, against a live board, under a stub DOM: every episode, every model under
// "Labels by", and the comparison view, through the page's own functions and the server's (or the static build's)
// real responses. The Python tests never execute the page's script, and `node --check` only parses it, so an error
// that happens while the page runs (a name used before it is defined, a field of a response read the wrong way)
// would otherwise reach a reader as a blank page.
//
//   node tests/render_smoke.js PAGE.html BASE_URL
//
// PAGE.html is the page as the server (or the static build) returns it, BASE_URL where its relative requests go.
// Prints a JSON summary (failed_requests lists every request that did not succeed) and exits 1 on the first uncaught
// error, unhandled rejection or console error.
'use strict';
const [pagePath, base] = process.argv.slice(2);
let P;
try {
  P = require('./page_sandbox')(pagePath, base, {expose: `
globalThis.__smoke = {eps: () => ALL_EPS, cmp: () => CMP, by: () => BY_EPS, kp: () => KP_INDEX,
                      sn: () => SN_INDEX, loadSensors, active: () => _activeFile};
`});
} catch (e) { console.error(e.message); process.exit(2); }
const {ctx, doc, created, failedRequests, fail, settle} = P;

(async () => {
  P.run();
  const S = ctx.__smoke;
  const rendered = [];
  if (!P.failure()) {
    const renderEp = ctx.renderEp;
    ctx.renderEp = function (d, opts) {
      try { return renderEp.call(this, d, opts); } catch (e) { fail(`renderEp(${S.active()})`, e); }
      finally { rendered.push({file: S.active(), cmp: !!d._compare, src: doc.getElementById('current-ep-src').innerHTML}); }
    };
  }
  const out = {episodes: 0, rendered: 0, labellers: [], compare_view: false, footage_lines: 0, keypoint_links: 0,
               sensors_panels: 0, touch_lanes: 0, contact_cards: 0, contact_checks: 0, no_label_pages: 0,
               answered_on_failed: 0, no_label_cards: 0};
  try {
    if (!P.failure()) { await ctx.loadEpisodes(); await settle(); }
    const eps = P.failure() ? [] : S.eps();
    out.episodes = eps.length;
    for (const e of eps) {
      if (P.failure()) break;
      doc.getElementById('sn-slot').innerHTML = '';
      await ctx.setDataset(e.dataset, e.file); await settle();
      // an episode with a sensors file: its panel, once the file is in (the render fills it when it arrives)
      if ((S.sn() || {})[e.file]) { await S.loadSensors(e.file); await settle(); }
      if (doc.getElementById('sn-slot').innerHTML.includes('class="subsection sn-h"')) out.sensors_panels++;
      // an episode with contacts: its Touch lane, its contact cards and its contact checks
      const left = doc.getElementById('left-col').innerHTML;
      if (left.includes('id="lane-touch"')) out.touch_lanes++;
      if (left.includes('id="tc-cards"')) out.contact_cards++;
      if (left.includes('Contact checks')) out.contact_checks++;
      // an episode whose reply gave no labels: one line in place of the label sections, and none of them drawn
      const right = doc.getElementById('right-col').innerHTML;
      if (left.includes('class="no-labels"')) out.no_label_pages++;
      if (e.label_failed && ['Key events', 'Was the task completed', 'Object relationships', 'Workspace inventory']
        .some(w => left.includes(w)) || e.label_failed && right.includes('Dense timeline')) out.answered_on_failed++;
      if (doc.getElementById('current-ep-src').innerHTML.startsWith('Footage: ')) out.footage_lines++;
      if (!doc.getElementById('kp-dl').hidden) out.keypoint_links++;
    }
    // the list's cards, each built with createElement: the episode whose reply gave no labels says so
    out.no_label_cards = new Set(created.filter(el => el.className === 'ep-card' && /&gt;no labels&lt;|>no labels</
      .test(el.innerHTML)).map(el => el.dataset.file)).size;
    const cmp = P.failure() ? null : S.cmp();
    for (const mdl of (cmp && cmp.models) || []) {
      if (P.failure()) break;
      await ctx.setLabeller(mdl.key); await settle();
      const list = S.by() || [];
      for (const e of list) { if (P.failure()) break; await ctx.setDataset(e.dataset, e.file); await settle(); }
      out.labellers.push({key: mdl.key, episodes: list.length,
        note: doc.getElementById('lb-note-in').innerHTML, band: doc.getElementById('sb-note').innerHTML});
    }
    if (cmp && !P.failure()) {
      await ctx.setLabeller(null); await settle();
      await ctx.showCompare(); await settle();
      out.compare_view = doc.getElementById('cmp-view').innerHTML.includes('How the models compare');
      await ctx.leaveCompare(); await settle();
    }
  } catch (e) { fail('drive', e); }
  out.rendered = rendered.length;
  out.rendered_comparisons = rendered.filter(r => r.cmp).length;
  out.failed_requests = failedRequests;
  if (P.failure()) { console.error('RENDER SMOKE FAIL\n' + P.failure()); process.exit(1); }
  process.stdout.write(JSON.stringify(out) + '\n');
  process.exit(0);
})();
