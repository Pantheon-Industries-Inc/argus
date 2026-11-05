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
  why: d => reviewWhyHtml(d), card: e => reviewChipsHtml(e), end: () => endRender(), build: ds => buildIssueFilter(ds),
  rail: ds => renderRail(ds, _activeFile), filter: v => { ISSUE_FILTER = v; },
  clearIssues: () => { for (const e of railEps()) { e.families = []; e.minor_families = []; } }};
`;
(async () => {
  const P = sandbox(pagePath, base, {expose});
  P.run();
  await P.ctx.loadEpisodes();
  await P.settle();
  const t = P.ctx.__t, eps = t.eps();
  const d = await fetch(new URL('api/episode?file=episode_000000.json', base)).then(r => r.json());
  const ds = eps.find(e => e.file === 'episode_000000.json').dataset;
  // what the page itself left: the open episode's reasons, its card in the rail, and the filter's rows and button
  const openWhy = P.state()['current-ep-why'], made = () => P.created.length;
  const htmlOf = (from, cls) => P.created.slice(from).filter(e => e.className === cls).map(e => e.innerHTML);
  let at = made();
  t.rail(ds);
  const cards = P.created.slice(at).map(e => e.innerHTML).filter(h => h.includes('class="review-tag"'));
  at = made();
  t.build(ds);
  const rowsShown = htmlOf(at, 'if-row'), button = htmlOf(at, 'if-btn')[0];
  at = made();
  t.filter('tag:tracking/low valid');
  t.build(ds);
  const buttonFiltered = htmlOf(at, 'if-btn')[0];
  t.filter(null);
  // a board with tags and no issue at all still has its filter
  t.clearIssues();
  at = made();
  t.build(ds);
  const rowsNoIssues = htmlOf(at, 'if-row');
  t.end();
  const whyAfterEnd = P.state()['current-ep-why'];
  const hostile = '<img src=x>';
  console.log(JSON.stringify({
    failure: P.failure(), open_why: openWhy && openWhy.html, why_after_end: whyAfterEnd && whyAfterEnd.html,
    card_chips: cards, rows_shown: rowsShown, button, button_filtered: buttonFiltered, rows_no_issues: rowsNoIssues,
    hostile_card: t.card({review_tags: [hostile]}), hostile_why: t.why({review_tags: [{tag: hostile, why: hostile}]}),
    rows: Object.fromEntries(t.rows(ds)),
    filtered: eps.filter(e => t.match(e, 'tracking/low valid')).map(e => e.file),
    other_tag: eps.filter(e => t.match(e, 'nothing')).map(e => e.file),
    why: t.why(d), why_untagged: t.why({}),
    card: t.card(eps.find(e => e.file === 'episode_000000.json')),
  }));
  process.exit(0);
})().catch(e => { console.error(e); process.exit(1); });
