// The board page with review tags: the filter lists each tag with its count, a tag narrows the rail, and the open
// episode says why it is on the board.
//
//   node tests/review_tags.js PAGE.html BASE_URL
'use strict';
const [pagePath, base] = process.argv.slice(2);
const sandbox = require('./page_sandbox');
const expose = `
globalThis.__t = {eps: () => ALL_EPS, rows: ds => reviewTagCounts(railEps().filter(e => datasetOf(e) === ds)),
  match: (e, tag) => { ISSUE_FILTER = TAG_FILTER + tag; const m = matchesIssueFilter(e); ISSUE_FILTER = null; return m; },
  why: d => reviewWhyHtml(d), card: e => reviewChipsHtml(e)};
`;
(async () => {
  const P = sandbox(pagePath, base, {expose});
  P.run();
  await P.ctx.loadEpisodes();
  await P.settle();
  const t = P.ctx.__t, eps = t.eps();
  const d = await fetch(new URL('api/episode?file=episode_000000.json', base)).then(r => r.json());
  const ds = eps.find(e => e.file === 'episode_000000.json').dataset;
  console.log(JSON.stringify({
    rows: Object.fromEntries(t.rows(ds)),
    filtered: eps.filter(e => t.match(e, 'tracking/low valid')).map(e => e.file),
    other_tag: eps.filter(e => t.match(e, 'nothing')).map(e => e.file),
    why: t.why(d), why_untagged: t.why({}),
    card: t.card(eps.find(e => e.file === 'episode_000000.json')),
  }));
  process.exit(0);
})().catch(e => { console.error(e); process.exit(1); });
