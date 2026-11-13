// The board page under navigation that outruns its requests, against a live board: a read that fails once must not
// stay failed for the rest of the visit, a response that arrives after the reader moved on must not undo the move, and
// what fills in a drawn episode afterwards must land while that episode stays drawn, whatever navigation kept it.
//
//   node tests/viewer_nav.js PAGE.html BASE_URL
//
// Each case runs the page afresh (page_sandbox.js) and reports what the reader would see. Prints a JSON summary and
// exits 1 on the first uncaught error, unhandled rejection or console error.
'use strict';
const [pagePath, base] = process.argv.slice(2);
const sandbox = require('./page_sandbox');
const wait = ms => new Promise(r => setTimeout(r, ms));
const expose = `
globalThis.__nav = {eps: () => ALL_EPS, cmp: () => CMP, active: () => _activeFile, by: () => BY, cmpHas,
                    dataset: () => currentDataset, rail: () => railEps().filter(e => datasetOf(e) === currentDataset),
                    drawn: () => { try { return _drawnFile; } catch (_) { return 'unknown'; } },
                    cached: f => _epCache.has(f), grids: () => _msView, shown: () => SHOWN};
`;

// a page whose requests matching a rule can fail once or be held until released, with setup code run before it starts
function page(rules, setup) {
  const P = sandbox(pagePath, base, {expose: expose + (setup || ''), fetch: async (url, real) => {
    for (const rule of rules) if (rule.match(String(url))) return rule.answer(real, String(url));
    return real();
  }});
  P.run();
  return P;
}
// every episode's frame grids and dictionary rows, each marked with the episode's file so what is shown says whose it
// is (every episode asks for grids under a fingerprint of its own); late(kind, file) delays each response
const fileOf = u => decodeURIComponent((String(u).match(/[?&]file=([^&]*)/) || [])[1] || '');
const FILLED_IN = 'BOARD.grids = true; window.__DICTIONARY_URL = "api/probe-dictionary";';
const filledIn = late => [
  {match: u => /api\/episode\?file=/.test(u), answer: async (real, u) => {
    await late('episode', fileOf(u));
    const d = await (await real()).json();
    d.request_images = {fingerprint: 'fp:' + fileOf(u), grid_images: [{}]};
    return Response.json(d); }},
  {match: u => /api\/grids\?|probe-dictionary/.test(u), answer: async (real, u) => {
    const dictionary = /probe-dictionary/.test(u), f = fileOf(u);
    await late(dictionary ? 'dictionary' : 'grids', f);
    return Response.json(dictionary ? {fields: [{id: 'probe', name: f, kind: 'text'}]}
      : {fingerprint: 'fp:' + f, grids: [{dispatch_status: 'response_received', src: 'grid.jpg'}]}); }},
];
// whose fetched dictionary rows are on the pane: each draw first shows the episode's own rows (no endpoint), which
// have none
function dictionaryShown(P) {
  let of = null;
  const rows = P.ctx.dictionaryRows;
  P.ctx.dictionaryRows = (data, endpoint) => {
    of = endpoint ? ((data && data.fields) || [{}])[0].name : null;
    return rows(data, endpoint);
  };
  return () => of;
}
const shown = P => ({active: P.ctx.__nav.active(), drawn: P.ctx.__nav.drawn(), search: P.ctx.location.search,
  compare_view: P.doc.body.classList.contains('view-cmp'), fading: P.doc.body.classList.contains('view-fade'),
  error: P.doc.getElementById('right-col').innerHTML.includes('could not be loaded'),
  title: P.doc.getElementById('current-ep').textContent});

(async () => {
  const out = {};
  const failures = [];
  const done = P => { if (P.failure()) failures.push(P.failure()); };

  // 1. An episode whose read fails once (the server's 500 with a JSON error body, a document caught mid-write), then
  // is repaired: opening it again shows it.
  {
    let failNext = null;
    const P = page([{match: u => failNext && /api\/episode\?file=|\/ep\//.test(u) && u.includes(encodeURIComponent(failNext)),
      answer: async () => { failNext = null; return new Response(JSON.stringify({error: 'The saved document could not be read.'}),
        {status: 500, headers: {'Content-Type': 'application/json'}}); }}]);
    await P.ctx.loadEpisodes(); await P.settle();
    const [a, b] = P.ctx.__nav.eps();
    await P.ctx.setDataset(a.dataset, a.file); await P.settle();
    out.first = shown(P);
    failNext = b.file;
    await P.ctx.setDataset(b.dataset, b.file); await P.settle();
    out.failed_read = {...shown(P), cached: P.ctx.__nav.cached(b.file)};
    await P.ctx.setDataset(a.dataset, a.file); await P.settle();
    await P.ctx.setDataset(b.dataset, b.file); await P.settle();
    out.after_repair = {...shown(P), cached: P.ctx.__nav.cached(b.file), file: b.file};
    done(P);
  }

  // 2. The comparison opened, then Back pressed while its metrics are still on their way: the episode stays.
  for (const [name, holdMetrics, backAfter] of [['back_while_metrics_load', true, 50],
                                                 ['back_while_view_fades', false, 60],
                                                 ['no_back', true, null]]) {
    let release;
    const held = new Promise(r => { release = r; });
    const P = page([{match: u => holdMetrics && /compare\/metrics/.test(u),
      answer: async real => { await held; return real(); }}]);
    await P.ctx.loadEpisodes(); await P.settle();
    const [a] = P.ctx.__nav.eps();
    await P.ctx.setDataset(a.dataset, a.file); await P.settle();
    const before = P.ctx.location.search;
    const opening = P.ctx.showCompare();
    if (backAfter != null) { await wait(backAfter); await P.popstate(before); }
    await wait(20); release();
    await opening; await wait(400); await P.settle();
    out[name] = {...shown(P), back_to: before};
    done(P);
  }

  // 3. What fills in a drawn episode after it is drawn (its frame grids, its data dictionary), held until released:
  // it lands while that draw is on the pane, whatever navigations kept it there (the comparison opened, then Back
  // before it appeared), and is dropped once another episode is drawn in its place. Each episode's grids carry its own
  // fingerprint and its dictionary its own name, so what is shown says whose it is.
  for (const kind of ['grids', 'dictionary']) {
    for (const how of ['back', 'no_back', 'replaced']) {
      let releaseAux, releaseMetrics, holdFile = null;
      const auxHeld = new Promise(r => { releaseAux = r; }), metricsHeld = new Promise(r => { releaseMetrics = r; });
      const P = page([
        {match: u => /compare\/metrics/.test(u), answer: async real => { await metricsHeld; return real(); }},
        ...filledIn(async (k, f) => { if (k === kind && f === holdFile) await auxHeld; }),
      ], FILLED_IN);
      const dictionaryOf = dictionaryShown(P);
      await P.ctx.loadEpisodes(); await P.settle();
      const [a, b] = P.ctx.__nav.eps();
      holdFile = a.file;
      P.ctx.setDataset(a.dataset, a.file); await wait(150);
      if (how === 'back') {
        const before = P.ctx.location.search;
        const opening = P.ctx.showCompare();
        await wait(20); await P.popstate(before);
        releaseMetrics(); await opening;
      } else if (how === 'replaced') {
        // the other episode drawn, and its own grids and dictionary landed, while the first one's are still held
        await P.ctx.setDataset(b.dataset, b.file);
        for (let i = 0; i < 100 && P.ctx.__nav.drawn() !== b.file; i++) await wait(20);
        await wait(150);
      }
      releaseAux(); releaseMetrics(); await wait(50); await P.settle();
      const grids = P.ctx.__nav.grids();
      out[kind + '_' + how] = {active: P.ctx.__nav.active(), drawn: P.ctx.__nav.drawn(),
        compare_view: P.doc.body.classList.contains('view-cmp'), a: a.file, b: b.file,
        grids: grids && grids.fingerprint, grids_hidden: P.doc.getElementById('ms-btn').hidden, dictionary: dictionaryOf()};
      done(P);
    }
  }

  // 4. An episode drawn in full (its header, downloads, buttons, grids and dictionary), then another whose read fails:
  // nothing of the first is left on any element of the page's markup. The reference is a page that opened only the
  // failing episode, never the first.
  {
    const failing = (bad, setup) => {
      const P = page([
        {match: u => /api\/episode\?file=/.test(u) && fileOf(u) === bad, answer: async () => Response.json(
          {error: 'The saved document could not be read.'}, {status: 500})},
        ...filledIn(async () => {}),
      ], FILLED_IN + 'BOARD.footage = true; BOARD.hands = true;');
      // the first episode carries everything a draw puts beside it: a raw id unlike its name, its dataset's source,
      // reader notes, a head camera (the hand pose switch) and frame grids
      const ep = P.ctx.fetchEpisode;
      P.ctx.fetchEpisode = file => ep(file).then(d => d && Object.assign(d, {_rig: 'ego_head',
        dataset_source: {name: 'Probe source', license: 'CC0', hub: 'https://example.org/' + file},
        reader_notes: {state_note: 'Probe note for ' + file, left_out: {cameras: ['probe_cam']}},
        _meta: {...d._meta, episode_id: 'episode_dual_arm__probe__' + file.replace(/\D/g, '')}}));
      return P;
    };
    const P0 = page([]);
    await P0.ctx.loadEpisodes(); await P0.settle();
    const [a, b] = P0.ctx.__nav.eps();
    done(P0);
    const diff = (x, y) => Object.keys(x).filter(id => JSON.stringify(x[id]) !== JSON.stringify(y[id]))
      .map(id => ({id, left: x[id], reference: y[id]}));
    // the reference: only b opened, and its read fails
    const R = failing(b.file);
    R.ctx.history.replaceState(null, '', '?ep=' + encodeURIComponent(b.file));
    await R.ctx.loadEpisodes(); await R.settle();
    // a opened and drawn in full, then b, whose read fails
    const P = failing(b.file);
    P.ctx.history.replaceState(null, '', '?ep=' + encodeURIComponent(a.file));
    await P.ctx.loadEpisodes(); await P.settle();
    const drawnA = {drawn: P.ctx.__nav.drawn(),
      overlay_canvas: P.doc.getElementById('left-col').innerHTML.includes('id="hp-canvas"'), differs: diff(P.state(), R.state()).map(x => x.id)};
    await P.ctx.setDataset(b.dataset, b.file); await P.settle();
    out.failed_after_drawn = {a: a.file, b: b.file, drawn_a: drawnA, reference_error: shown(R).error,
      error: shown(P).error, drawn: P.ctx.__nav.drawn(),
      overlay_canvas: P.doc.getElementById('left-col').innerHTML.includes('id="hp-canvas"'), left: diff(P.state(), R.state())};
    done(P); done(R);
  }

  // 5. Random clicks, tabs, labellers, the comparison and Back, each made before the last one's responses arrive, and
  // every response late by a random time: once all have arrived, the page shows what its URL says (the comparison
  // exactly when the URL asks for it, otherwise the URL's episode drawn under the URL's labeller, and no fade left on),
  // and the frame grids and dictionary rows on the pane are the drawn episode's (no grids under another labeller's).
  const rng = seed => () => {
    seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
  out.random = {runs: 0, wrong: []};
  for (let seed = 1; seed <= Number(process.env.VIEWER_NAV_RUNS || 25); seed++) {
    const rand = rng(seed), pick = xs => xs[Math.floor(rand() * xs.length)];
    // the grids' delays come from a stream of their own, so the actions and other delays of a seed stay as they were
    const randGrids = rng(seed + 1000);
    const P = page([...filledIn(async k => { await wait((k === 'grids' ? randGrids : rand)() * 120); }),
                    {match: () => true, answer: async real => { await wait(rand() * 120); return real(); }}], FILLED_IN);
    const dictionaryOf = dictionaryShown(P);
    const N = () => P.ctx.__nav, inCompare = () => P.doc.body.classList.contains('view-cmp');
    let last = null;
    const renderEp = P.ctx.renderEp;
    P.ctx.renderEp = function (d, o) {
      last = {id: (d._meta || {}).episode_id, by: (d._compare || {}).key || null};
      return renderEp.call(this, d, o);
    };
    await P.ctx.loadEpisodes(); await P.settle(); await wait(300);
    const eps = N().eps(), keys = [null, ...((N().cmp() || {}).models || []).map(m => m.key)];
    const steps = [];
    // what a reader can do on the screen in front of them: the list's cards, the dataset tabs, the Labels by menu,
    // the comparison and its episode links, and the browser's Back
    const act = () => {
      const r = rand();
      if (r < 0.25 && !inCompare()) {
        const rail = N().rail();
        if (rail.length) { const e = pick(rail); steps.push('card ' + e.file); return P.ctx.selectEp(e.file); }
      }
      if (r < 0.4) {
        const ds = pick(eps).dataset;
        steps.push('tab ' + ds);
        return inCompare() ? P.ctx.leaveCompare(ds) : (ds !== N().dataset() ? P.ctx.setDataset(ds) : null);
      }
      if (r < 0.55) { const k = pick(keys); steps.push('labeller ' + k); return P.ctx.setLabeller(k); }
      if (r < 0.7) { steps.push('compare'); return P.ctx.showCompare(); }
      if (r < 0.8 && inCompare()) {
        const e = pick(eps), k = pick(keys);
        steps.push('table link ' + e.file + ' ' + k);
        return P.ctx.leaveCompare(null, e.file, k && N().cmpHas(k, e.file) ? k : null);
      }
      steps.push('back');
      return P.back();
    };
    for (let i = 0; i < 6; i++) { act(); await wait(rand() * 100); }
    await P.settle(); await wait(600); await P.settle();
    const q = new URLSearchParams(P.ctx.location.search), active = N().active();
    const want = {compare: q.get('view') === 'compare', ep: q.get('ep'), by: q.get('by') || null};
    const rec = eps.find(e => e.file === active) || {};
    const shows = {compare: inCompare(), fading: P.doc.body.classList.contains('view-fade')
      || P.doc.body.classList.contains('lb-swap'), active, by: N().by(), drawn: last};
    const drawn = N().drawn(), grids = N().grids();
    const filled = {grids: grids && grids.fingerprint, dictionary: dictionaryOf()};
    const filledRight = !drawn || (filled.dictionary === drawn
      && filled.grids === (N().shown() ? null : 'fp:' + drawn));
    const right = shows.compare === want.compare && !shows.fading && (want.compare || (
      (!want.ep || want.ep === active) && shows.by === want.by && last && last.id === rec.episode_id
      && last.by === (shows.by && N().cmpHas(shows.by, active) ? shows.by : null)));
    out.random.runs++;
    if (!right || !filledRight) out.random.wrong.push({seed, steps, url: P.ctx.location.search, shows, drawn, filled});
    done(P);
  }

  process.stdout.write(JSON.stringify(out) + '\n');
  if (failures.length) { console.error('VIEWER NAV FAIL\n' + failures.join('\n')); process.exit(1); }
  process.exit(0);
})();
