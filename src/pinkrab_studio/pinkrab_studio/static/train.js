// The training page: the runs the server has read, and their numbers drawn.
//
// The run list comes down the page's socket (live.js) as it changes - it is
// small, and it is what says a run has moved on. The points are asked for
// only for the ticked runs, and only when one of them has logged something
// new since the last ask, so a finished run is fetched once.
//
// Charts are uPlot, one per metric, one line per run - the smoothed line in
// the run's colour over the raw one, faint. Every chart shares the cursor, so
// hovering one step reads it on all of them.

import { onConnection, watch } from './live.js';
import { sourceForm } from './source-form.js';

const uPlot = window.uPlot;

const PALETTE = ['#e2488e', '#1668c9', '#17a673', '#f5b301', '#7a4fd6', '#e8743b', '#0aa5b8', '#8f7283'];
// The order charts appear in: the loss and what it is made of first, then
// the optimiser, then the machine. Anything else a policy logs comes after.
const ORDER = ['loss', 'eval/eval_loss', 'l1_loss', 'kld_loss', 'grad_norm', 'lr',
  'samples_per_s', 'update_s', 'dataloading_s', 'gpu_mem_gb'];
const BOOKKEEPING = new Set(['steps', 'samples', 'episodes', 'epochs']);
const LOSSY = (key) => /loss/.test(key) || key === 'grad_norm';
const TITLES = {
  loss: 'loss', 'eval/eval_loss': 'eval loss', l1_loss: 'L1 loss', kld_loss: 'KL loss',
  grad_norm: 'gradient norm', lr: 'learning rate', samples_per_s: 'samples / s',
  update_s: 'update s / step', dataloading_s: 'data s / step', gpu_mem_gb: 'GPU memory, GB',
};
// How often the ages on the page - "2 min ago", time left, a finished job's
// line - are written again between updates. The page's own clock: nothing
// is asked.
const AGES_EVERY_MS = 5000;
// The server's clock less this page's, as of the last update: ages are
// counted on the clock the runs were stamped by.
let skew = 0;
const serverNow = () => Date.now() / 1000 + skew;

const runsEl = document.getElementById('runs');
const chartsEl = document.getElementById('charts');
const chipsEl = document.getElementById('source-chips');
const smoothingEl = document.getElementById('smoothing');
const smoothingValueEl = document.getElementById('smoothing-value');
const xAxisEl = document.getElementById('x-axis');
const logScaleEl = document.getElementById('log-scale');
const hoverStepEl = document.getElementById('hover-step');

const stored = (key, fallback) => {
  try { const v = localStorage.getItem(`train.${key}`); return v === null ? fallback : JSON.parse(v); } catch { return fallback; }
};
const store = (key, value) => { try { localStorage.setItem(`train.${key}`, JSON.stringify(value)); } catch { /* per-viewer only */ } };

let runs = [];
let sources = [];
let selected = new Set(stored('selected', []));
let everSeen = new Set(stored('seen', []));
let series = {};            // run id -> metric -> [steps, values, times]
let fetchedAt = {};         // run id -> last_step when its points were fetched
const charts = new Map();   // metric -> { u, el, readout, signature }
const cursorSync = uPlot.sync('train');

smoothingEl.value = stored('smoothing', 0.6);
xAxisEl.value = stored('xAxis', 'step');
logScaleEl.checked = stored('logScale', true);
smoothingValueEl.textContent = Number(smoothingEl.value).toFixed(2);

const colour = (id) => PALETTE[id % PALETTE.length];

// --- words for numbers ---------------------------------------------------------

function steps(n) {
  if (n == null) return '—';
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(n >= 1e4 ? 0 : 1)}k`;
  return String(n);
}
function duration(s) {
  if (s == null || !isFinite(s)) return '';
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  if (h) return `${h}h${String(m).padStart(2, '0')}m`;
  if (m) return `${m}m`;
  return `${s}s`;
}
function ago(t, now) {
  if (!t) return 'never';
  return `${duration(now - t) || '0s'} ago`;
}
function num(v) {
  if (v == null || !isFinite(v)) return '—';
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  return v.toPrecision(4).replace(/\.?0+$/, '');
}
function clock(t) {
  return t ? new Date(t * 1000).toLocaleString() : '—';
}

// --- the run list ------------------------------------------------------------------

function renderRuns(now) {
  if (!runs.length) {
    runsEl.innerHTML = '<li class="empty">no runs yet: New run starts one</li>';
    return;
  }
  runsEl.replaceChildren(...runs.map((r) => {
    const li = document.createElement('li');
    li.className = 'run';
    li.title = 'draw this run alone';
    li.style.setProperty('--swatch', colour(r.id));
    const frac = r.steps ? Math.min(1, (r.last_step || 0) / r.steps) : 0;
    const bits = [];
    if (r.loss != null) bits.push(`loss <b>${num(r.loss)}</b>`);
    if (r.rate) bits.push(`${r.rate.toFixed(1)} step/s`);
    if (r.eta_s) bits.push(`${duration(r.eta_s)} left`);
    if (r.checkpoints) bits.push(`${r.checkpoints} ckpt`);
    bits.push(r.status === 'running' ? `updated ${ago(r.updated, now)}` : `last ${ago(r.updated, now)}`);
    li.innerHTML = `
      <input type="checkbox" title="draw this run beside the others" ${selected.has(r.id) ? 'checked' : ''}>
      <div class="title-line"><span class="name"></span><button class="details" title="checkpoints and the run's facts">details</button></div>
      <div class="what"><span class="chip ${r.status}">${r.status}</span> ${r.policy || '?'} · ${r.dataset || '?'} · ${r.source}</div>
      <div class="progress-line"><span class="meter"><span style="width:${(frac * 100).toFixed(1)}%"></span></span>${steps(r.last_step)} / ${steps(r.steps)}</div>
      <div class="numbers">${bits.map((b) => `<span>${b}</span>`).join('')}</div>
      ${r.error && r.status === 'failed' ? '<div class="error"></div>' : ''}`;
    li.querySelector('.name').textContent = r.name || r.output_dir;
    li.querySelector('.name').title = r.output_dir;
    if (r.error && r.status === 'failed') li.querySelector('.error').textContent = r.error;
    li.querySelector('input').addEventListener('change', (e) => {
      if (e.target.checked) selected.add(r.id); else selected.delete(r.id);
      store('selected', [...selected]);
      refreshSeries(true);
    });
    // The card itself draws this run and no other; the tick adds it beside them.
    li.addEventListener('click', (e) => {
      // A drag that selected text - an error to copy - is not a click.
      if (e.target.closest('input, button') || !getSelection().isCollapsed) return;
      selected = new Set([r.id]);
      store('selected', [...selected]);
      renderRuns(serverNow());
      refreshSeries(true);
    });
    li.querySelector('.details').addEventListener('click', () => openRun(r.id));
    return li;
  }));
}

function renderSourcesLine(now) {
  const chips = sources.map((s) => {
    const chip = document.createElement('button');
    chip.className = `chip-source${s.error ? ' bad' : ''}`;
    const name = document.createElement('b');
    name.textContent = s.name;
    const busy = s.unrecorded?.length ? ` · ${s.unrecorded.length} not recorded` : '';
    chip.append(name, ` · ${s.error ? 'cannot read' : `${s.runs} run${s.runs === 1 ? '' : 's'}${busy} · ${ago(s.synced, now)}`}`);
    chip.title = s.error ? `${s.name}: ${s.error}` : s.kind === 'local' ? s.root : `${s.user || ''}@${s.host}:${s.port} ${s.root}`;
    chip.addEventListener('click', openSources);
    return chip;
  });
  const add = document.createElement('button');
  add.className = 'chip-source add';
  add.textContent = '+ source';
  add.title = 'add a machine to train on';
  add.addEventListener('click', openSources);
  chipsEl.replaceChildren(...chips, add);
  renderUnrecorded();
}

// What is on a source's GPUs and no run here accounts for.
function renderUnrecorded() {
  const box = document.getElementById('unrecorded');
  const rows = sources.flatMap((s) => (s.unrecorded || []).map((u) => ({ ...u, source: s.name })));
  box.hidden = !rows.length;
  box.querySelector('ul').replaceChildren(...rows.map((u) => {
    const li = document.createElement('li');
    const name = Object.assign(document.createElement('b'), { textContent: u.name });
    const where = Object.assign(document.createElement('span'), {
      textContent: `${u.source}${u.memory_mb ? `, ${(u.memory_mb / 1024).toFixed(1)} GB of GPU` : ''}` });
    li.append(name, where);
    return li;
  }));
}

function showRuns(answer) {
  if (!answer || answer.error) return;
  skew = answer.now - Date.now() / 1000;
  runs = answer.runs;
  sources = answer.sources;
  // A run seen for the first time while it is training is drawn without
  // being asked for: that is what the page is usually opened to watch.
  for (const r of runs) {
    if (!everSeen.has(r.id)) {
      everSeen.add(r.id);
      if (r.status === 'running') selected.add(r.id);
    }
  }
  // Nothing ticked among the runs listed is an empty page: the newest runs
  // are drawn instead.
  if (runs.length && !runs.some((r) => selected.has(r.id))) {
    for (const r of runs.slice(0, 3)) selected.add(r.id);
  }
  store('seen', [...everSeen]);
  store('selected', [...selected]);
  renderRuns(answer.now);
  renderSourcesLine(answer.now);
  if (sourcesDialog.open) renderSources(answer.now);
  refreshSeries(false).catch(() => { /* drawn again with the next update */ });
}

// --- the points ----------------------------------------------------------------------

async function refreshSeries(force) {
  const ids = runs.filter((r) => selected.has(r.id)).map((r) => r.id);
  const stale = ids.filter((id) => {
    const r = runs.find((x) => x.id === id);
    return force || !(id in series) || fetchedAt[id] !== r.last_step;
  });
  if (stale.length) {
    const answer = await (await fetch(`/train/series?ids=${stale.join(',')}`)).json();
    for (const [id, metrics] of Object.entries(answer.series)) {
      series[id] = metrics;
      fetchedAt[id] = runs.find((x) => x.id === Number(id))?.last_step;
    }
  }
  // Drawn for the runs ticked now: a card clicked while this was fetching
  // has changed them.
  drawCharts(runs.filter((r) => selected.has(r.id)).map((r) => r.id));
}

// TensorBoard's smoothing: an exponential moving average, corrected for
// having started at zero so the first points are not dragged down.
function smooth(values, weight) {
  if (!weight) return values.slice();
  const out = new Array(values.length);
  let last = 0, n = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (v == null || !isFinite(v)) { out[i] = null; continue; }
    last = last * weight + (1 - weight) * v;
    n += 1;
    out[i] = last / (1 - Math.pow(weight, n));
  }
  return out;
}

function xs(id, metric) {
  const s = series[id][metric];
  const mode = xAxisEl.value;
  if (mode === 'hours') {
    const run = runs.find((r) => r.id === id);
    const t0 = run?.started ?? s[2].find((t) => t != null);
    return s[2].map((t) => (t == null || t0 == null ? null : (t - t0) / 3600));
  }
  if (mode === 'epochs') {
    const epochs = series[id].epochs;
    if (!epochs) return s[0];
    const at = new Map(epochs[0].map((step, i) => [step, epochs[1][i]]));
    return s[0].map((step) => at.get(step) ?? null);
  }
  return s[0];
}

function metricsOf(ids) {
  const found = new Set();
  for (const id of ids) for (const m of Object.keys(series[id] || {})) if (!BOOKKEEPING.has(m)) found.add(m);
  return [...found].sort((a, b) => {
    const ia = ORDER.indexOf(a), ib = ORDER.indexOf(b);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b);
  });
}

function drawCharts(ids) {
  const withData = ids.filter((id) => series[id]);
  const metrics = metricsOf(withData);
  if (!metrics.length) {
    for (const c of charts.values()) c.u.destroy();
    charts.clear();
    chartsEl.innerHTML = `<p class="empty">${ids.length ? 'no points yet for the ticked runs' : 'tick a run to draw it'}</p>`;
    return;
  }
  chartsEl.querySelector('p.empty')?.remove();
  const weight = Number(smoothingEl.value);
  for (const metric of [...charts.keys()]) {
    if (!metrics.includes(metric)) { charts.get(metric).u.destroy(); charts.get(metric).el.remove(); charts.delete(metric); }
  }
  metrics.forEach((metric, order) => {
    const having = withData.filter((id) => series[id][metric]);
    // One table per run - x, raw, smoothed - then joined on x, the way uPlot
    // wants every series to share one x array.
    const tables = having.map((id) => {
      const x = xs(id, metric);
      const raw = series[id][metric][1];
      const keep = x.map((v, i) => v != null && raw[i] != null ? i : -1).filter((i) => i >= 0);
      const kx = keep.map((i) => x[i]);
      const kv = keep.map((i) => raw[i]);
      const order2 = kx.map((_, i) => i).sort((a, b) => kx[a] - kx[b]);
      const sx = order2.map((i) => kx[i]);
      const sv = order2.map((i) => kv[i]);
      return [sx, sv, smooth(sv, weight)];
    });
    const data = tables.length ? uPlot.join(tables) : [[]];
    const log = logScaleEl.checked && LOSSY(metric) && having.every((id) => series[id][metric][1].every((v) => v == null || v > 0));
    const signature = `${having.join(',')}|${log}|${xAxisEl.value}`;
    let chart = charts.get(metric);
    if (!chart || chart.signature !== signature) {
      if (chart) { chart.u.destroy(); chart.el.remove(); }
      const el = document.createElement('section');
      el.className = 'chart';
      el.innerHTML = '<header><h3></h3><span class="readout"></span></header><div class="plot"></div>';
      el.querySelector('h3').textContent = TITLES[metric] || metric;
      const readout = el.querySelector('.readout');
      const seriesOpts = [{}];
      for (const id of having) {
        seriesOpts.push({ label: `${id} raw`, stroke: `${colour(id)}40`, width: 1, spanGaps: true, points: { show: false } });
        seriesOpts.push({ label: `${id}`, stroke: colour(id), width: 2, spanGaps: true, points: { show: false } });
      }
      const u = new uPlot({
        width: 400, height: 220,
        cursor: { sync: { key: cursorSync.key }, drag: { x: true, y: false } },
        legend: { show: false },
        scales: { x: { time: false }, y: log ? { distr: 3 } : {} },
        axes: [
          { stroke: '#8f7283', grid: { stroke: '#ecd9e2' }, ticks: { stroke: '#ecd9e2' },
            values: (_, vals) => vals.map((v) => (xAxisEl.value === 'step' ? steps(v) : num(v))) },
          { stroke: '#8f7283', size: 56,
            // A log axis marks every 1..9 of a decade; only 1, 2 and 5 are kept,
            // for the labels, the ticks and the grid alike.
            grid: { stroke: '#ecd9e2', ...(log ? { filter: keepOnesTwosFives } : {}) },
            ticks: { stroke: '#ecd9e2', ...(log ? { filter: keepOnesTwosFives } : {}) },
            ...(log ? { filter: keepOnesTwosFives } : {}),
            values: (_, vals) => vals.map((v) => (v == null ? '' : num(v))) },
        ],
        series: seriesOpts,
        hooks: { setCursor: [(uu) => readCursor(uu, having, readout)] },
      }, data, el.querySelector('.plot'));
      chart = { u, el, readout, signature };
      charts.set(metric, chart);
    } else {
      chart.u.setData(data);
    }
    // Keep the grid in the metric order even as charts come and go.
    if (chartsEl.children[order] !== chart.el) chartsEl.insertBefore(chart.el, chartsEl.children[order] || null);
  });
  sizeCharts();
}

function onesTwosFives(v) {
  if (!(v > 0)) return false;
  const m = v / Math.pow(10, Math.floor(Math.log10(v) + 1e-9));
  return [1, 2, 5].some((k) => Math.abs(m - k) < 1e-6);
}
const keepOnesTwosFives = (_, splits) => splits.map((v) => (onesTwosFives(v) ? v : null));

function readCursor(u, ids, readout) {
  const i = u.cursor.idx;
  if (i == null) { readout.textContent = ''; hoverStepEl.textContent = ''; return; }
  const x = u.data[0][i];
  hoverStepEl.textContent = xAxisEl.value === 'step' ? `step ${Math.round(x).toLocaleString()}` : `${xAxisEl.value} ${num(x)}`;
  readout.innerHTML = ids.map((id, k) => {
    const v = u.data[2 + k * 2]?.[i];
    return v == null ? '' : `<span><i style="background:${colour(id)}"></i>${num(v)}</span>`;
  }).join('');
}

function sizeCharts() {
  for (const c of charts.values()) {
    const w = c.el.querySelector('.plot').clientWidth;
    if (w && Math.abs(w - c.u.width) > 2) c.u.setSize({ width: w, height: 220 });
  }
}
window.addEventListener('resize', sizeCharts);

// --- controls ----------------------------------------------------------------------------

smoothingEl.addEventListener('input', () => {
  smoothingValueEl.textContent = Number(smoothingEl.value).toFixed(2);
  store('smoothing', Number(smoothingEl.value));
  drawCharts(runs.filter((r) => selected.has(r.id)).map((r) => r.id));
});
xAxisEl.addEventListener('change', () => { store('xAxis', xAxisEl.value); refreshSeries(false); });
logScaleEl.addEventListener('change', () => { store('logScale', logScaleEl.checked); refreshSeries(false); });
document.getElementById('sync').addEventListener('click', async (e) => {
  const button = e.currentTarget;
  button.classList.add('spinning');
  await fetch('/train/sync', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
  // What the sync finds comes down the socket; the spin is only to say it was asked.
  setTimeout(() => button.classList.remove('spinning'), 2500);
});

// --- asking the server to do things -----------------------------------------------------

async function post(url, body) {
  const answer = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });
  let json = {};
  try { json = await answer.json(); } catch { /* an empty answer */ }
  if (!answer.ok || json.error) throw new Error(json.error || `${answer.status}`);
  return json;
}

function probeLine(p) {
  if (!p) return '';
  const gpus = (p.gpus || []).map((g) => `${g.name} ${(g.mem_used_mb / 1024).toFixed(0)}/${(g.mem_total_mb / 1024).toFixed(0)} GB used`
    + (g.driver ? `, driver ${g.driver}` : '') + (g.compute ? `, sm ${g.compute}` : '')).join(', ') || 'no GPU';
  if ((p.short || []).length) return `${p.host}: cannot train here - ${p.short.join('; ')}`;
  const ready = p.venv && p.lock_matches ? 'set up' : p.venv ? 'set up, lock changed - will update' : 'not set up yet - the first run sets it up';
  const busy = (p.running || []).length ? `, ${(p.running || []).length} running` : '';
  return `${p.host}: ${gpus} · ${p.cores} cores · ${p.free_gb} GB free · ${ready}${busy}`;
}

// --- one run -------------------------------------------------------------------------------

const runDialog = document.getElementById('run');
const runErrorEl = document.getElementById('run-error');
let openRunId = null;

async function openRun(id) {
  const r = await (await fetch(`/train/run?id=${id}`)).json();
  if (r.error) return;
  openRunId = id;
  runErrorEl.textContent = '';
  const listed = runs.find((x) => x.id === id);
  const status = listed?.status || r.status;
  const local = r.source === 'this machine';
  document.getElementById('run-title').textContent = r.name || r.output_dir;
  const facts = [
    ['status', status + (r.error ? ` - ${r.error}` : '')], ['source', r.source], ['host', r.host],
    ['output', r.output_dir], ['policy', r.policy], ['dataset', r.dataset],
    ['steps', `${(r.last_step || 0).toLocaleString()} of ${(r.steps || 0).toLocaleString()}`],
    ['batch', r.batch_size], ['log every', r.log_freq ? `${r.log_freq} steps` : null],
    ['started', clock(r.started)], ['last news', clock(r.updated)],
    ...(r.launch?.log ? [['terminal', r.launch.log]] : []),
  ];
  const dl = document.getElementById('run-facts');
  // A fact the run did not leave is left out, not shown as a dash.
  dl.replaceChildren(...facts.filter(([, v]) => v !== null && v !== undefined && v !== '').flatMap(([k, v]) => {
    const dt = document.createElement('dt'); dt.textContent = k;
    const dd = document.createElement('dd'); dd.textContent = v ?? '—';
    return [dt, dd];
  }));
  const autofetch = document.getElementById('run-autofetch');
  autofetch.parentElement.hidden = local;
  autofetch.checked = !!r.launch?.auto_fetch;
  const body = document.getElementById('run-checkpoints');
  body.replaceChildren(...(r.checkpoints.length ? r.checkpoints : [null]).map((c) => {
    const tr = document.createElement('tr');
    if (!c) { tr.innerHTML = '<td colspan="3">none saved yet</td>'; return tr; }
    tr.innerHTML = '<td></td><td></td><td></td>';
    tr.children[0].textContent = steps(c.step);
    const here = local ? `${c.path}/pretrained_model` : c.local;
    tr.children[1].textContent = here || c.path;
    const cell = tr.children[2];
    if (here) {
      const mark = document.createElement('span'); mark.className = 'here'; mark.textContent = 'here ';
      const copy = document.createElement('button'); copy.textContent = 'copy path';
      copy.addEventListener('click', () => navigator.clipboard?.writeText(here));
      cell.append(mark, copy);
    } else {
      const get = document.createElement('button'); get.textContent = 'copy here';
      get.addEventListener('click', async () => {
        get.disabled = true;
        try { await post('/train/fetch', { id, step: c.step }); get.textContent = 'queued'; } catch (exc) { runErrorEl.textContent = exc.message; get.disabled = false; }
      });
      cell.append(get);
    }
    return tr;
  }));
  document.getElementById('run-config').textContent = r.config ? JSON.stringify(r.config, null, 2) : '';
  document.getElementById('run-config').closest('details').hidden = !r.config;
  document.getElementById('run-stop').hidden = !['running', 'quiet'].includes(status);
  document.getElementById('run-resume').hidden = ['running', 'done'].includes(status) || !r.checkpoints.length;
  runDialog.showModal();
}
document.getElementById('run-close').addEventListener('click', () => runDialog.close());
// A click outside the dialog's box - on the backdrop - closes it too, when
// the press began there as well: a text selection dragged out of the box ends
// on the backdrop and is not a click on it.
function onBackdrop(e) {
  const box = runDialog.getBoundingClientRect();
  return e.target === runDialog
    && (e.clientX < box.left || e.clientX > box.right || e.clientY < box.top || e.clientY > box.bottom);
}
let pressedOnBackdrop = false;
// However the dialog closes, a Stop pressed once is let go: the next run
// opened must not stop on a single press.
runDialog.addEventListener('close', () => {
  const stop = document.getElementById('run-stop');
  stop.dataset.sure = '';
  stop.textContent = 'Stop';
});
runDialog.addEventListener('mousedown', (e) => { pressedOnBackdrop = onBackdrop(e); });
runDialog.addEventListener('click', (e) => {
  if (pressedOnBackdrop && onBackdrop(e)) runDialog.close();
  pressedOnBackdrop = false;
});
document.getElementById('run-autofetch').addEventListener('change', async (e) => {
  try { await post('/train/autofetch', { id: openRunId, on: e.target.checked }); } catch (exc) { runErrorEl.textContent = exc.message; }
});
for (const [button, url] of [['run-stop', '/train/stop'], ['run-resume', '/train/resume']]) {
  const el = document.getElementById(button);
  el.addEventListener('click', async () => {
    if (button === 'run-stop' && el.dataset.sure !== '1') { el.dataset.sure = '1'; el.textContent = 'Stop - sure?'; return; }
    try { await post(url, { id: openRunId }); runDialog.close(); } catch (exc) { runErrorEl.textContent = exc.message; }
    el.dataset.sure = ''; el.textContent = button === 'run-stop' ? 'Stop' : 'Resume';
  });
}

// --- what the server is doing for the page -----------------------------------------------

const jobsEl = document.getElementById('jobs');
let lastJobs = null;

function showJobs(state) {
  if (!state || state.error) return;
  lastJobs = state;
  for (const kind of ['launch', 'fetch']) {
    const j = state[kind];
    const el = document.getElementById(`job-${kind}`);
    const recent = j.ended && Date.now() / 1000 - j.ended < (j.error ? 600 : 30);
    el.hidden = !(j.busy || recent);
    el.className = `job${j.busy ? '' : j.error ? ' failed' : ' done'}`;
    el.querySelector('b').textContent = j.what || '';
    el.querySelector('.line').textContent = j.error ? j.error : j.busy ? (j.lines.at(-1) || 'starting') : (j.lines.at(-1) || 'done');
  }
  jobsEl.hidden = document.getElementById('job-launch').hidden && document.getElementById('job-fetch').hidden;
}

// --- starting a run ------------------------------------------------------------------------------

const launchDialog = document.getElementById('launch');
const launchForm = document.getElementById('launch-form');
const launchErrorEl = document.getElementById('launch-error');
const launchProbeEl = document.getElementById('launch-probe');
let recipes = null;

async function probeInto(name, el) {
  el.textContent = `asking ${name}…`;
  try { el.textContent = probeLine(await post('/train/sources/test', { name })); } catch (exc) { el.textContent = exc.message; }
}

document.getElementById('open-launch').addEventListener('click', async () => {
  launchErrorEl.textContent = '';
  recipes = await (await fetch('/train/recipes')).json();
  const sourceEl = document.getElementById('launch-source');
  sourceEl.replaceChildren(...sources.map((s) => new Option(s.kind === 'local' ? 'this machine' : `${s.name} (${s.host})`, s.name)));
  const remote = sources.find((s) => s.kind !== 'local');
  if (remote) sourceEl.value = remote.name;
  const datasetEl = document.getElementById('launch-dataset');
  datasetEl.replaceChildren(...recipes.datasets.map((d) => new Option(
    `${d.name} · ${d.episodes ?? '?'} episodes · ${d.frames ?? '?'} frames${d.image ? ` · ${d.image[1]}x${d.image[0]}` : ''} · ${(d.bytes / 1e6).toFixed(0)} MB`, d.name)));
  const policyEl = document.getElementById('launch-policy');
  policyEl.replaceChildren(...Object.entries(recipes.recipes).map(([k, r]) => new Option(r.label, k)));
  launchForm.steps.value = recipes.defaults.steps;
  launchForm.log_freq.value = recipes.defaults.log_freq;
  launchForm.save_freq.value = recipes.defaults.save_freq;
  launchForm.batch_size.value = '';
  launchForm.batch_size.placeholder = recipes.recipes[policyEl.value].batch_size;
  launchDialog.showModal();
  probeInto(sourceEl.value, launchProbeEl);
});
document.getElementById('launch-source').addEventListener('change', (e) => probeInto(e.target.value, launchProbeEl));
document.getElementById('launch-policy').addEventListener('change', (e) => {
  launchForm.batch_size.placeholder = recipes.recipes[e.target.value].batch_size;
});
document.getElementById('launch-close').addEventListener('click', () => launchDialog.close());
launchForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  launchErrorEl.textContent = '';
  const payload = Object.fromEntries(new FormData(launchForm).entries());
  payload.auto_fetch = launchForm.auto_fetch.checked;
  for (const k of ['steps', 'batch_size', 'log_freq', 'save_freq']) payload[k] = payload[k] ? Number(payload[k]) : null;
  try {
    await post('/train/launch', payload);
    launchDialog.close();
  } catch (exc) {
    launchErrorEl.textContent = exc.message;
  }
});

// --- sources -------------------------------------------------------------------------------

const sourcesDialog = document.getElementById('sources');
const form = sourceForm(document.getElementById('source-form'), async () => form.say('saved - reading it now'));

const TRASH = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13M10 11v6M14 11v6"/></svg>';

function renderSources(now) {
  const list = document.getElementById('sources-list');
  list.replaceChildren(...sources.map((s) => {
    const li = document.createElement('li');
    li.className = 'source-row';
    li.innerHTML = `<div class="who"><b></b><span class="where"></span></div>
      <div class="state-line"></div><div class="actions"></div><p class="probe"></p>`;
    li.querySelector('b').textContent = s.name;
    li.querySelector('.where').textContent = s.kind === 'local' ? s.root
      : `${s.user || ''}@${s.host}:${s.port}  ${s.root}${s.key ? '' : '  · no key'}`;
    const state = li.querySelector('.state-line');
    state.textContent = s.error ? s.error : `${s.runs} run${s.runs === 1 ? '' : 's'} · read ${ago(s.synced, now)}`;
    state.classList.toggle('bad', !!s.error);
    const probeEl = li.querySelector('.probe');
    const group = document.createElement('div');
    group.className = 'segmented';
    const test = document.createElement('button');
    test.textContent = 'Test';
    test.title = 'GPUs, driver, cores, room, what is running';
    test.addEventListener('click', () => probeInto(s.name, probeEl));
    group.append(test);
    const actions = li.querySelector('.actions');
    actions.append(group);
    if (s.kind !== 'local') {
      const edit = document.createElement('button');
      edit.textContent = 'Edit';
      edit.title = 'fill the form below with this source';
      edit.addEventListener('click', () => form.fill(s));
      group.append(edit);
      const del = document.createElement('button');
      del.className = 'remove';
      del.title = 'remove this source and forget its runs here (nothing on the machine is touched)';
      del.innerHTML = TRASH;
      del.addEventListener('click', async () => {
        if (!del.classList.contains('sure')) { del.classList.add('sure'); del.textContent = 'Remove?'; return; }
        await post('/train/sources/delete', { name: s.name });
      });
      del.addEventListener('mouseleave', () => { if (del.classList.contains('sure')) { del.classList.remove('sure'); del.innerHTML = TRASH; } });
      actions.append(del);
    }
    return li;
  }));
}

function openSources() {
  renderSources(serverNow());
  form.say('');
  sourcesDialog.showModal();
}
document.getElementById('sources-close').addEventListener('click', () => sourcesDialog.close());

watch('train', showRuns);
watch('train_jobs', showJobs);
// A studio that stops answering says so where the sources are; the next
// update, once it is back, draws them again.
onConnection((up) => {
  if (up) return;
  const chip = document.createElement('span');
  chip.className = 'chip-source bad';
  chip.textContent = 'the studio is not answering';
  chipsEl.replaceChildren(chip);
});
setInterval(() => {
  if (runs.length) { renderRuns(serverNow()); renderSourcesLine(serverNow()); }
  if (lastJobs) showJobs(lastJobs);
}, AGES_EVERY_MS);
