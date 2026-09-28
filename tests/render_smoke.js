// Runs the board page's script as served, against a live board, under a stub DOM: every episode, every model under
// "Labels by", and the comparison view, through the page's own functions and the server's (or the static build's)
// real responses. The Python tests never execute the page's script, and `node --check` only parses it, so an error
// that happens while the page runs (a name used before it is defined, a field of a response read the wrong way)
// would otherwise reach a reader as a blank page.
//
//   node tests/render_smoke.js PAGE.html BASE_URL
//
// PAGE.html is the page as the server (or the static build) returns it, BASE_URL where its relative requests go.
// Prints a JSON summary and exits 1 on the first uncaught error, unhandled rejection or console error.
'use strict';
const fs = require('fs');
const vm = require('vm');

const [pagePath, base] = process.argv.slice(2);
const html = fs.readFileSync(pagePath, 'utf8');
const m = html.match(/<script>([\s\S]*?)<\/script>/);
if (!m) { console.error('no <script> block in ' + pagePath); process.exit(2); }
// the page starts itself with its last statement; the smoke starts it instead, once the run is instrumented
const boot = '\nloadEpisodes();\n';
if (!m[1].endsWith(boot)) { console.error('the page does not end with loadEpisodes()'); process.exit(2); }
const pageJs = m[1].slice(0, -boot.length) + `
globalThis.__smoke = {eps: () => ALL_EPS, cmp: () => CMP, by: () => BY_EPS, kp: () => KP_INDEX,
                      active: () => _activeFile};
`;

let failure = null;
const fail = (what, e) => { if (!failure) failure = what + ': ' + (e && e.stack || e); };
process.on('unhandledRejection', e => fail('unhandled rejection', e));
process.on('uncaughtException', e => fail('uncaught exception', e));

// ---- a stub DOM: every element takes any property and call, and keeps what the page writes into it ----
const noop = () => {};
function makeEl(id) {
  const t = {id: id || '', _html: '', _text: '', _val: '', dataset: {}, children: [], hidden: false,
    style: {setProperty: noop, removeProperty: noop},
    classList: {_s: new Set(), add(...c) { c.forEach(x => this._s.add(x)); }, remove(...c) { c.forEach(x => this._s.delete(x)); },
      toggle(c, on) { const v = on === undefined ? !this._s.has(c) : !!on; if (v) this._s.add(c); else this._s.delete(c); return v; },
      contains(c) { return this._s.has(c); }}};
  return new Proxy(t, {
    get(o, p) {
      if (p === 'innerHTML') return o._html;
      if (p === 'textContent') return o._text;
      if (p === 'value') return o._val;
      if (p in o) return o[p];
      switch (p) {
        case 'querySelectorAll': return () => [];
        case 'querySelector': return () => makeEl();
        case 'appendChild': case 'removeChild': case 'insertBefore': case 'append': return (c) => c;
        case 'getAttribute': return () => null;
        case 'closest': return () => null;
        case 'contains': return () => false;
        case 'cloneNode': return () => makeEl();
        case 'getContext': return () => new Proxy({}, {get: () => noop});
        case 'getBoundingClientRect': return () => ({top: 0, left: 0, width: 100, height: 100, right: 100, bottom: 100});
        case 'parentElement': case 'parentNode': case 'firstChild': case 'nextElementSibling': return null;
        case 'offsetTop': case 'offsetLeft': case 'offsetHeight': case 'offsetWidth': case 'scrollTop':
        case 'scrollLeft': case 'scrollWidth': case 'clientWidth': case 'clientHeight': case 'scrollHeight': return 0;
        case 'then': return undefined;
        default: return typeof p === 'string' ? noop : undefined;
      }
    },
    set(o, p, v) {
      if (p === 'innerHTML') o._html = String(v);
      else if (p === 'textContent') o._text = String(v);
      else if (p === 'value') o._val = v;
      else o[p] = v;
      return true;
    },
  });
}
const byId = new Map();
const doc = {
  getElementById: (id) => { if (!byId.has(id)) byId.set(id, makeEl(id)); return byId.get(id); },
  createElement: () => makeEl(), createDocumentFragment: () => makeEl(), querySelector: () => makeEl(), querySelectorAll: () => [],
  addEventListener: noop, removeEventListener: noop, body: makeEl('body'), documentElement: makeEl(),
  exitFullscreen: noop, fullscreenElement: null, activeElement: null,
};
const store = new Map();
let inflight = 0, rafs = 0;
const ctx = {
  document: doc, console: {log: noop, warn: noop, info: noop, error: (...a) => fail('console.error', a.join(' '))},
  location: {search: '', href: base, origin: new URL(base).origin, pathname: new URL(base).pathname, reload: noop},
  history: {replaceState: noop, pushState: noop},
  localStorage: {getItem: k => store.has(k) ? store.get(k) : null, setItem: (k, v) => store.set(k, String(v)),
    removeItem: k => store.delete(k)},
  fetch: async (url, opts) => {
    inflight++;
    try { return await fetch(new URL(String(url), base), {method: (opts || {}).method, headers: (opts || {}).headers,
      body: (opts || {}).body}); } finally { inflight--; }
  },
  setTimeout, clearTimeout, setInterval: () => 0, clearInterval: noop, queueMicrotask, structuredClone,
  requestAnimationFrame: (cb) => { if (++rafs < 5000) return setTimeout(() => cb(performance.now()), 4); return 0; },
  cancelAnimationFrame: (h) => clearTimeout(h),
  getComputedStyle: () => ({overflowY: 'auto', getPropertyValue: () => ''}),
  matchMedia: () => ({matches: false, addEventListener: noop, addListener: noop}),
  addEventListener: noop, removeEventListener: noop, alert: (m) => fail('alert', m), performance,
  innerWidth: 1440, innerHeight: 900, devicePixelRatio: 1,
  AbortController, URL, URLSearchParams, TextEncoder, TextDecoder, Blob, JSON, Math, Date, Promise,
  IntersectionObserver: function () { return {observe: noop, disconnect: noop, unobserve: noop}; },
  ResizeObserver: function () { return {observe: noop, disconnect: noop, unobserve: noop}; },
  CSS: {supports: () => true, escape: s => s},
};
ctx.window = ctx;
ctx.globalThis = ctx;
vm.createContext(ctx);

// wait until the page has no request in flight and its timers have had a turn
async function settle() {
  for (let quiet = 0; quiet < 3;) {
    await new Promise(r => setTimeout(r, 30));
    quiet = inflight ? 0 : quiet + 1;
  }
}

(async () => {
  try { vm.runInContext(pageJs, ctx, {filename: 'board-page.js'}); } catch (e) { fail('top level', e); }
  const S = ctx.__smoke;
  const rendered = [];
  if (!failure) {
    const renderEp = ctx.renderEp;
    ctx.renderEp = function (d, opts) {
      try { return renderEp.call(this, d, opts); } catch (e) { fail(`renderEp(${S.active()})`, e); }
      finally { rendered.push({file: S.active(), cmp: !!d._compare, src: doc.getElementById('current-ep-src').innerHTML}); }
    };
  }
  const out = {episodes: 0, rendered: 0, labellers: [], compare_view: false, footage_lines: 0, keypoint_links: 0};
  try {
    if (!failure) { await ctx.loadEpisodes(); await settle(); }
    const eps = failure ? [] : S.eps();
    out.episodes = eps.length;
    for (const e of eps) {
      if (failure) break;
      await ctx.setDataset(e.dataset, e.file); await settle();
      if (doc.getElementById('current-ep-src').innerHTML.startsWith('Footage: ')) out.footage_lines++;
      if (!doc.getElementById('kp-dl').hidden) out.keypoint_links++;
    }
    const cmp = failure ? null : S.cmp();
    for (const mdl of (cmp && cmp.models) || []) {
      if (failure) break;
      await ctx.setLabeller(mdl.key); await settle();
      const list = S.by() || [];
      for (const e of list) { if (failure) break; await ctx.setDataset(e.dataset, e.file); await settle(); }
      out.labellers.push({key: mdl.key, episodes: list.length,
        note: doc.getElementById('lb-note-in').innerHTML, band: doc.getElementById('sb-note').innerHTML});
    }
    if (cmp && !failure) {
      await ctx.setLabeller(null); await settle();
      await ctx.showCompare(); await settle();
      out.compare_view = doc.getElementById('cmp-view').innerHTML.includes('How the models compare');
      await ctx.leaveCompare(); await settle();
    }
  } catch (e) { fail('drive', e); }
  out.rendered = rendered.length;
  out.rendered_comparisons = rendered.filter(r => r.cmp).length;
  if (failure) { console.error('RENDER SMOKE FAIL\n' + failure); process.exit(1); }
  process.stdout.write(JSON.stringify(out) + '\n');
  process.exit(0);
})();
