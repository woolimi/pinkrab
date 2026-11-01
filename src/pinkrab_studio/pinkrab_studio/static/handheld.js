// The handheld page: the live server's three pictures and its numbers.
// One switch starts and stops the live server through the studio; the
// pictures are that server's own MJPEG streams, and the cards are its stats
// as the studio relays them. Everything the page shows comes down its socket
// (live.js) as it changes.

import { metric, showStream, sourceSelect } from './cams.js';
import { setPage, watch } from './live.js';
import { Connection } from './connection.js';
import { topLight } from './toplight.js';
import { openReplay } from './replay.js';

const toggleEl = document.getElementById('live-toggle');
const stateEl = document.getElementById('live-state');
const dotEl = document.getElementById('live-dot');
const lineEl = document.getElementById('live-line');
const noteEl = document.getElementById('note');
const retryEl = document.getElementById('head-retry');
retryEl.addEventListener('click', async () => {
  retryEl.disabled = true;
  retryEl.textContent = 'opening ...';
  try {
    const answer = await ask('/rig/live/head/retry', {});
    if (answer.error) { noteEl.textContent = answer.error; retryEl.disabled = false; retryEl.textContent = 'Retry top camera'; }
  } catch (err) { noteEl.textContent = String(err); retryEl.disabled = false; }
});
const cams = {
  left: document.getElementById('cam-left'),
  right: document.getElementById('cam-right'),
  head: document.getElementById('cam-head'),
};
const showLight = topLight(document.getElementById('head-light'));

const cards = {
  left: document.getElementById('card-left'),
  right: document.getElementById('card-right'),
  desk: document.getElementById('card-desk'),
};

let liveUrl = null;
let running = false;
// Set while a Connect or a Disconnect is on its way: the button says which and
// takes no second click, and the rig's updates leave it alone.
let switching = null;

async function ask(path, body) {
  const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify(body ?? {}) });
  return r.json();
}

function showRig(rig) {
  if (!rig?.live) return;
  if (rig.live_url) liveUrl = rig.live_url;
  const live = rig.live;
  // A live view on the follower arms' cameras - the policy page's - is not
  // this page's: Connect starts it again on the handhelds'.
  const theirs = live.running && (live.options?.cameras ?? 'handheld') !== 'handheld';
  running = live.running && !theirs;
  dotEl.className = 'dot' + (running ? ' on' : live.exit ? ' bad' : '');
  drawToggle();
  lineEl.textContent = running ? 'connected - cameras, jaws, tracking'
    : theirs ? "the live view has the follower arms' cameras - Connect opens the handhelds'"
    : (live.note || 'not connected');
  stateEl.classList.toggle('bad', !running && !!live.exit);
  for (const [which, fig] of Object.entries(cams)) {
    const img = fig.querySelector('img');
    if (which !== 'head') sourceSelect(fig, which, () => img.removeAttribute('src'));
    // Up, the pictures are asked for once the view's numbers come (showStats).
    if (!running) {
      img.removeAttribute('src');
      img.hidden = true; fig.querySelector('.off').hidden = false; fig.classList.remove('on');
      fig.querySelector('.stat').textContent = '';
    }
  }
  if (!running) {
    for (const c of Object.values(cards)) metric(c, '—', '', '');
    noteEl.textContent = '';
    retryEl.hidden = true;
    if (typeof showEpisode === 'function') showEpisode(null);
  }
  // The view's numbers may have come before the rig said it was up; they
  // come again only when they change. Down, they were the last view's, and a
  // picture asked of them would be asked of a server not yet listening.
  if (!running) lastStats = null;
  else if (lastStats) showStats(lastStats);
}

function drawToggle() {
  toggleEl.textContent = switching || (running ? 'Disconnect' : 'Connect');
  toggleEl.className = switching ? 'busy' : running ? 'stop' : 'primary';
  toggleEl.disabled = !!switching;
}

// The live view's numbers, as they change while it runs. None until its
// server answers - it opens the top camera before it listens, a few seconds
// - and that is also when its pictures can be asked for.
let lastStats = null;
function showStats(s) {
  lastStats = s;
  // A view on the arms' cameras, going down as this page's comes up, is not
  // this page's: its numbers would point the pictures at a server not yet
  // listening.
  showLight(running && s?.head !== false ? s?.head_light : null);
  if (!running || !s || !Object.keys(s).length || (s.cameras ?? 'handheld') !== 'handheld') return;
  for (const side of ['left', 'right']) {
    const img = cams[side].querySelector('img');
    if (liveUrl && !img.getAttribute('src')) showStream(img, liveUrl + `gripper/${side}`);
  }
  if (s.head === false) {
    // The view is up without a top camera: the picture stays dark, says
    // why, and offers to open it again - after a replug, one click.
    const img = cams.head.querySelector('img');
    img.removeAttribute('src'); img.hidden = true;
    cams.head.classList.remove('on');
    const off = cams.head.querySelector('.off');
    off.hidden = false;
    off.classList.add('why');
    // The reason's last clause is the one that says it: "No device connected".
    const why = (s.head_reason || 'not connected').split(':').pop().trim();
    off.textContent = s.head_opening ? 'opening the top camera ...'
      : `no top camera - ${why}. Plug it in, then retry.`;
    retryEl.hidden = false;
    retryEl.disabled = !!s.head_opening;
    retryEl.textContent = s.head_opening ? 'opening ...' : 'Retry top camera';
    cams.head.querySelector('.stat').textContent = '';
  } else {
    retryEl.hidden = true;
    cams.head.querySelector('.off').textContent = 'top camera';
    cams.head.querySelector('.off').classList.remove('why');
    cams.head.querySelector('.stat').textContent = s.fps == null ? 'opening' : `${Math.round(s.fps)} fps`;
    const img = cams.head.querySelector('img');
    if (liveUrl && !img.getAttribute('src')) showStream(img, liveUrl + 'stream');
  }
  for (const side of ['left', 'right']) {
    const g = s.gripper?.[side];
    cams[side].querySelector('.stat').textContent = g?.running
      ? `${Math.round(g.fps)} fps · brightness ${Math.round(g.mean)}` + (g.lens ? ` · ${g.lens}` : '')
      : (g?.note || 'no camera');
    const c = s.solids?.[`handheld_${side}`];
    if (!c) metric(cards[side], 'not in frame', '', '');
    else if (!c.solved) metric(cards[side], 'seen, not solved', '', 'bad');
    else metric(cards[side], `${(c.distance_mm / 10).toFixed(0)} cm`,
                `${c.faces} ${c.faces === 1 ? 'face' : 'faces'} · ${c.reproj_px.toFixed(1)} px` + (c.world ? '' : ' · no world'),
                c.world ? 'ok' : 'bad');
  }
  const w = s.world;
  showEpisode(s.episode);
  if (s.head === false) { for (const c of Object.values(cards)) metric(c, '—', 'no top camera', ''); feedStartup(s, null); return; }
  if (!w) metric(cards.desk, '—', '', '');
  else if (w.tracking) metric(cards.desk, `${w.faces} faces`, `${w.reproj_px.toFixed(1)} px`, 'ok');
  else metric(cards.desk, 'not in view', w.reason || '', 'bad');
  noteEl.textContent = s.trouble ? `frame failed: ${s.trouble}` : '';
  feedStartup(s, null);
}
watch('stats', showStats);

// --- the connection ------------------------------------------------------------
// A line per thing the live view opens and the pedal, each with what it said
// last and a Refresh that opens that one thing again - see connection.js.
// A Refresh whose answer carries the rig draws it at once.
const withRig = (asked) => async () => {
  const answer = await asked();
  if (answer?.rig) feedStartup(null, answer.rig);
  return answer;
};
let liveFailed = false;
const conn = new Connection(document.getElementById('startup'), [
  ['live', 'live view', () => reconnect()],
  ['head', 'top camera', withRig(() => ask('/rig/live/head/retry', {}))],
  ['left', 'left gripper camera', withRig(() => ask('/rig/live/gripper/left/retry', {}))],
  ['right', 'right gripper camera', withRig(() => ask('/rig/live/gripper/right/retry', {}))],
  ['jaw_left', 'left jaw', withRig(() => ask('/rig/live/jaw/left/retry', {}))],
  ['jaw_right', 'right jaw', withRig(() => ask('/rig/live/jaw/right/retry', {}))],
  ['pedal', 'pedal', withRig(() => ask('/rig/footswitch/start', {}))],
], {
  storageKey: 'collect.connection.folded',
  shown: () => running || liveFailed,
  canRefresh: (key) => key === 'live' || key === 'pedal' || running,
});

// Fed from the view's numbers and the rig's state, as they change.
function feedStartup(s, rig) {
  if (rig) {
    const pedal = rig.footswitch;
    if (pedal?.failed) conn.set('pedal', 'failed', pedal.failed);
    else conn.set('pedal', pedal?.running ? (/cannot|no foot switch|went away/.test(pedal.note || '') ? 'failed' : 'running')
                              : pedal?.exit != null ? 'failed' : 'starting',
                     pedal?.running ? (pedal.note || 'reading')
                       : pedal?.exit != null ? 'did not start - see the terminal' : 'starting ...');
    liveFailed = !rig.live?.running && !!rig.live?.exit;
    if (liveFailed) conn.set('live', 'failed', rig.live.note || 'did not start');
  }
  if (s) {
    conn.set('live', 'running', 'answering');
    if (s.head === false) conn.set('head', s.head_opening ? 'starting' : 'failed', s.head_opening ? 'opening ...' : (s.head_reason || 'not connected').split(':').pop().trim());
    else conn.set('head', s.fps == null ? 'starting' : 'running', s.fps == null ? 'opening ...' : `${Math.round(s.fps)} fps`);
    for (const side of ['left', 'right']) {
      const g = s.gripper?.[side];
      conn.set(side, g?.running ? 'running' : g?.note ? 'failed' : 'starting',
                  g?.running ? `${Math.round(g.fps)} fps` + (g.lens ? ` · ${g.lens}` : '') : (g?.note || 'opening ...'));
      const j = s.jaw?.[side];
      conn.set(`jaw_${side}`, j && j.aperture != null ? 'running' : j?.note ? 'failed' : 'starting',
                  j && j.aperture != null ? `open ${(j.aperture * 100).toFixed(0)} %` : (j?.note || 'reading ...'));
    }
  }
  conn.draw();
}

async function connect(stopping) {
  if (switching) return;
  switching = stopping ? 'Disconnecting ...' : 'Connecting ...';
  drawToggle();
  let rig = null;
  try {
    if (!stopping) {
      conn.begin();
      conn.set('live', 'starting', 'starting ...');
    } else conn.end();
    conn.draw();
    lastStats = null;   // the view about to start, or stop, is not the one they came from
    const answer = await ask(`/rig/live/${stopping ? 'stop' : 'start'}`, {});
    if (answer.error) { lineEl.textContent = answer.error; conn.end(); }
    rig = answer.rig;
  } catch (err) { lineEl.textContent = String(err); }
  switching = null;
  if (rig) { showRig(rig); feedStartup(null, rig); } else { drawToggle(); conn.draw(); }
}

// The live view's Refresh: down and up again, which is also what finds a
// camera plugged in after it started on macOS.
// The live row's Refresh: a running view is restarted by the server, which
// refuses while an episode records; one that is not up is connected.
async function reconnect() {
  if (running) return ask('/rig/live/restart', {});
  await connect(false);
  return {};
}

toggleEl.addEventListener('click', () => connect(running));

// The rig: the live server's address and what each process is doing.
watch('rig', (rig) => { showRig(rig); feedStartup(null, rig); });


// --- recording -------------------------------------------------------------------
// While this page is in front, pedal 2 starts an episode and pedal 3 stops
// it; the page's own pedals do the same by click. The live view records;
// this page asks, shows, and beeps.
const recDot = document.getElementById('rec-dot');
const recLine = document.getElementById('rec-line');
const recNote = document.getElementById('rec-note');
const pedalRecord = document.getElementById('pedal-record');
const pedalStop = document.getElementById('pedal-stop');
const PEDALS = { handheld: pedalRecord, stop: pedalStop, jog: document.getElementById('pedal-jog') };
const shelfBody = document.querySelector('#episodes tbody');
const shelfEmpty = document.getElementById('episodes-empty');
let recording = null;      // the episode name while one records
let audio = null;
let pedalSeen = 0;

// Two tones: up for start, down for stop. Made here, not from a file, so the
// page carries them; the first click on the page is what lets audio play.
function beep(kind) {
  try {
    audio = audio || new (window.AudioContext || window.webkitAudioContext)();
    const notes = kind === 'start' ? [[660, 0], [880, 0.12]] : [[880, 0], [660, 0.12]];
    for (const [hz, at] of notes) {
      const osc = audio.createOscillator(); const gain = audio.createGain();
      osc.frequency.value = hz; osc.connect(gain); gain.connect(audio.destination);
      const t = audio.currentTime + at;
      gain.gain.setValueAtTime(0.0001, t); gain.gain.exponentialRampToValueAtTime(0.2, t + 0.01);
      gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.11);
      osc.start(t); osc.stop(t + 0.12);
    }
  } catch { /* no audio here */ }
}

function fmtBytes(n) { return n > 1e9 ? `${(n / 1e9).toFixed(1)} GB` : n > 1e6 ? `${(n / 1e6).toFixed(0)} MB` : `${Math.round(n / 1e3)} kB`; }

function showEpisode(ep) {
  const was = recording;
  recording = ep?.active ? ep.name : null;
  recDot.className = 'rec-dot' + (recording ? ' on' : '');
  // Lit from the state, as on the teleop page: 2 while recording, 3 while not.
  pedalRecord.classList.toggle('on', !!recording);
  pedalStop.classList.toggle('on', running && !recording);
  pedalRecord.disabled = !running || !!recording;
  pedalStop.disabled = !running || !recording;
  if (recording) {
    const f = ep.frames || {};
    recLine.textContent = `recording ${ep.seconds}s · head ${f.head ?? 0} · left ${f.gripper_left ?? 0} · right ${f.gripper_right ?? 0}`;
    recLine.title = ep.name;
  } else {
    recLine.textContent = running ? 'not recording' : 'not connected';
    recLine.title = '';
  }
  if (was && !recording) beep('stop');
  if (!was && recording) beep('start');
}

// --- the shelf --------------------------------------------------------------------
// Every take on disk, a page of them at a time, newest first. Ticks are for
// deleting; the dataset below is made from every complete take.
const ticked = new Set();
const tickAll = document.getElementById('tick-all');
const deleteTicked = document.getElementById('delete-ticked');
const countEl = document.getElementById('episodes-count');
const tickFailing = document.getElementById('tick-failing');
const sortFollow = document.getElementById('sort-follow');
const pagePrev = document.getElementById('page-prev');
const pageNext = document.getElementById('page-next');
const pageWhere = document.getElementById('page-where');
const pageSizeEl = document.getElementById('page-size');
const pagerEl = document.getElementById('episodes-pager');
const PAGE_SIZES = [20, 50, 100];
let shelf = [];
// Each take's verdict by name (verdict.py), and the takes still being replayed.
let verdicts = {};
let verdictsPending = [];
let sortByFollow = false;
let pageSize = PAGE_SIZES[0];
try {
  const kept = Number(localStorage.getItem('collect.shelf.pageSize'));
  if (PAGE_SIZES.includes(kept)) pageSize = kept;
} catch { /* no storage here */ }
pageSizeEl.value = String(pageSize);
let pageAt = 0;              // which page of the shelf is shown, from the newest

function fmtSeconds(s) { return s >= 60 ? `${Math.floor(s / 60)} min ${Math.round(s % 60)} s` : `${Math.round(s)} s`; }

// The takes on the page shown, in the shelf's order.
function shownTakes() {
  return sortedShelf().slice(pageAt * pageSize, (pageAt + 1) * pageSize);
}

function showTicks() {
  for (const tr of shelfBody.querySelectorAll('tr')) {
    tr.classList.toggle('ticked', ticked.has(tr.dataset.name));
    const box = tr.querySelector('td.tick input');
    if (box) box.checked = ticked.has(tr.dataset.name);
  }
  deleteTicked.disabled = ticked.size === 0;
  // A delete waiting for its second click keeps saying so, whatever redraws.
  if (deleteTicked.classList.contains('sure')) deleteTicked.textContent = `sure? delete ${ticked.size}`;
  else deleteTicked.textContent = ticked.size ? `delete ${ticked.size} ticked` : 'delete ticked';
  // The box in the head ticks the page shown: the takes it sits above.
  const complete = shownTakes().filter((e) => e.complete);
  tickAll.checked = complete.length > 0 && complete.every((e) => ticked.has(e.name));
  const failing = shelf.filter((e) => e.complete && verdicts[e.name]?.fails.length).length;
  tickFailing.disabled = failing === 0;
  tickFailing.textContent = verdictsPending.length ? `tick failing (${failing}, judging ${verdictsPending.length})`
    : `tick failing (${failing})`;
}

tickFailing.addEventListener('click', () => {
  for (const ep of shelf) if (ep.complete && verdicts[ep.name]?.fails.length) ticked.add(ep.name);
  showTicks();
});

sortFollow.addEventListener('click', () => {
  sortByFollow = !sortByFollow;
  sortFollow.classList.toggle('on', sortByFollow);
  pageAt = 0;
  drawShelf();
});

pagePrev.addEventListener('click', () => { pageAt -= 1; drawShelf(); });
pageNext.addEventListener('click', () => { pageAt += 1; drawShelf(); });
pageSizeEl.addEventListener('change', () => {
  // The take at the top of the page shown stays on the page shown.
  const first = pageAt * pageSize;
  pageSize = Number(pageSizeEl.value);
  try { localStorage.setItem('collect.shelf.pageSize', String(pageSize)); } catch { /* no storage here */ }
  pageAt = Math.floor(first / pageSize);
  drawShelf();
});

// The verdict cells of the rows on show, from what the server has judged.
function fillVerdicts() {
  for (const tr of shelfBody.querySelectorAll('tr')) {
    const v = verdicts[tr.dataset.name];
    const follow = tr.querySelector('td.follow');
    const cell = tr.querySelector('td.verdict');
    if (!follow || !cell) continue;
    if (!v) {
      follow.textContent = '';
      cell.className = 'verdict';
      cell.textContent = verdictsPending.includes(tr.dataset.name) ? 'judging ...' : '';
      cell.title = '';
      continue;
    }
    if (v.error) {
      // Not judged: the take is not at fault, and tick failing leaves it be.
      follow.textContent = '';
      cell.className = 'verdict warn';
      cell.textContent = `not judged - ${v.error}`;
      cell.title = '';
      continue;
    }
    follow.textContent = v.feasible == null ? '' : `${Math.round(v.feasible * 100)}%`;
    // What fails it and what is only worth knowing, apart: a warning beside
    // a failure is not why the take fails.
    cell.className = `verdict ${v.fails.length ? 'fail' : v.warns.length ? 'warn' : ''}`;
    cell.replaceChildren(...[[v.fails, 'fails'], [v.warns, 'warns']].filter(([list]) => list.length)
      .map(([list, kind]) => Object.assign(document.createElement('span'), { className: kind, textContent: list.join(' · ') })));
    if (!v.fails.length && !v.warns.length) cell.textContent = 'ok';
    const flags = Object.entries(v.flags || {}).map(([k, n]) => `${k} ${n}`).join(', ') || 'none';
    // Each hand's jaw, as far as it moved across its travel.
    const jaws = Object.entries(v.jaw_spread || {}).map(([hand, s]) => `${hand} ${Math.round(s * 100)}%`).join(', ');
    cell.title = v.kind === 'rollout'
      ? `rollout · corrected ${v.held_s} s · robot unread ${v.lost_s} s · jaws moved ${jaws} of their travel`
        + ` · frames flagged: ${flags}`
      : `held ${v.held_s} s · lost ${v.lost_s} s (the worse hand) · jaws moved ${jaws} of their travel, `
        + `read ${Math.round(v.jaw_read * 100)}% · frames the replay flagged: ${flags}`;
  }
  showTicks();
}

// Each take's verdict as the server judges it; takes still being replayed
// come in one by one.
watch('verdicts', (answer) => {
  verdicts = answer?.verdicts || {};
  verdictsPending = answer?.pending || [];
  // Drawn again only when the rows on the page have changed: a redraw puts
  // back every row, and a delete half confirmed with them.
  const shown = [...shelfBody.querySelectorAll('tr')].map((tr) => tr.dataset.name).join(',');
  if (sortByFollow && shown !== shownTakes().map((e) => e.name).join(',')) drawShelf(); else fillVerdicts();
});

tickAll.addEventListener('change', () => {
  for (const ep of shownTakes()) if (ep.complete) { if (tickAll.checked) ticked.add(ep.name); else ticked.delete(ep.name); }
  showTicks();
});

deleteTicked.addEventListener('click', async () => {
  // Twice: the second click, within a few seconds, is the yes.
  if (!deleteTicked.classList.contains('sure')) {
    deleteTicked.classList.add('sure'); deleteTicked.textContent = `sure? delete ${ticked.size}`;
    setTimeout(() => { deleteTicked.classList.remove('sure'); showTicks(); }, 4000);
    return;
  }
  deleteTicked.disabled = true;
  await ask('/rig/live/episode/delete', { names: [...ticked] }).catch(() => {});
  ticked.clear();
  deleteTicked.classList.remove('sure');
  showTicks();
});

// The shelf, as it changes on disk: a take begun, finished, deleted.
watch('episodes', (answer) => {
  const list = answer?.episodes || [];
  shelf = list;
  for (const name of [...ticked]) if (!list.some((e) => e.name === name && e.complete)) ticked.delete(name);
  const seconds = list.reduce((a, e) => a + (e.seconds || 0), 0);
  const bytes = list.reduce((a, e) => a + (e.bytes || 0), 0);
  const frames = list.reduce((a, e) => a + ((e.frames || {}).head || 0), 0);
  countEl.textContent = list.length ? `${list.length} episodes · ${fmtSeconds(seconds)} · ${frames} frames · ${fmtBytes(bytes)}` : '';
  drawShelf();
});

function sortedShelf() {
  return sortByFollow
    ? [...shelf].sort((a, b) => (verdicts[a.name]?.feasible ?? 2) - (verdicts[b.name]?.feasible ?? 2))   // unjudged last
    : shelf;
}

function drawShelf() {
  const total = shelf.length;
  const pages = Math.max(1, Math.ceil(total / pageSize));
  pageAt = Math.min(Math.max(0, pageAt), pages - 1);
  const list = shownTakes();
  shelfBody.replaceChildren(...list.map((ep) => {
    const tr = document.createElement('tr');
    tr.dataset.name = ep.name;
    if (!ep.complete) tr.className = 'partial';
    const tickTd = document.createElement('td'); tickTd.className = 'tick';
    const tick = document.createElement('input'); tick.type = 'checkbox'; tick.checked = ticked.has(ep.name); tick.disabled = !ep.complete;
    tick.addEventListener('change', () => { if (tick.checked) ticked.add(ep.name); else ticked.delete(ep.name); showTicks(); });
    tickTd.append(tick); tr.append(tickTd);
    const f = ep.frames || {};
    // Play first: the whole point of the shelf is to see what an episode does.
    const playTd = document.createElement('td');
    const play = document.createElement('button');
    play.textContent = '▶'; play.className = 'play'; play.title = 'play it back on the arms';
    play.disabled = !ep.complete;
    play.addEventListener('click', () => openReplay(ep.name));
    playTd.append(play); tr.append(playTd);
    const cells = [ep.name, ep.seconds != null ? ep.seconds.toFixed(1) : (ep.complete ? '' : 'recording / partial'), f.head ?? '', f.gripper_left ?? '', f.gripper_right ?? '', ep.dropped ?? '', fmtBytes(ep.bytes || 0)];
    for (const [i, text] of cells.entries()) {
      const td = document.createElement('td'); td.textContent = text; if (i === 0) td.className = 'name'; tr.append(td);
      // A rollout - a policy's run, its arms put right by the hands where it
      // went wrong, recorded on the robot - is kept beside the demonstrations
      // and said so.
      if (i === 0 && ep.kind === 'rollout') {
        const tag = document.createElement('span');
        const corrected = ep.controlled?.human;
        tag.className = 'kind';
        tag.textContent = 'rollout' + (corrected != null ? ` · corrected ${corrected.toFixed(1)} s` : '');
        td.append(' ', tag);
      }
    }
    for (const name of ['follow', 'verdict']) {
      const td = document.createElement('td'); td.className = name; tr.append(td);
    }
    const td = document.createElement('td');
    const del = document.createElement('button'); del.textContent = 'delete';
    // Twice to delete: the second click, within a few seconds, is the yes.
    del.addEventListener('click', async () => {
      if (!del.classList.contains('sure')) { del.classList.add('sure'); del.textContent = 'sure?'; setTimeout(() => { del.classList.remove('sure'); del.textContent = 'delete'; }, 4000); return; }
      del.disabled = true;
      await ask('/rig/live/episode/delete', { name: ep.name }).catch(() => {});
    });
    td.append(del); tr.append(td);
    return tr;
  }));
  shelfEmpty.hidden = total > 0;
  pagerEl.hidden = total <= PAGE_SIZES[0];
  pageWhere.textContent = total ? `${pageAt * pageSize + 1}–${pageAt * pageSize + list.length} of ${total}` : '';
  pagePrev.disabled = pageAt === 0;
  pageNext.disabled = pageAt >= pages - 1;
  fillVerdicts();
}

async function askEpisode(verb) {
  try {
    const answer = await ask(`/rig/live/episode/${verb}`, {});
    recNote.textContent = answer.error || '';
  } catch (err) { recNote.textContent = String(err); }
}
pedalRecord.addEventListener('click', () => askEpisode('start'));
pedalStop.addEventListener('click', () => askEpisode('stop'));

// A press on the floor flashes the pedal here, as on the teleop page.
function showPedal(pedal) {
  if (!pedal || pedal.count === pedalSeen) return;
  const first = pedalSeen === 0;
  pedalSeen = pedal.count;
  if (first && pedal.age > 2) return;
  const button = PEDALS[pedal.name];
  if (!button) return;
  button.classList.remove('pressed');
  void button.offsetWidth;
  button.classList.add('pressed');
  setTimeout(() => button.classList.remove('pressed'), 350);
}
watch('pedal', showPedal);

// This page in front, while it is visible: that is what makes the floor's
// pedals record rather than drive. The clutch says what it made of a press.
setPage('collect');
watch('clutch', (clutch) => { if (clutch?.note) recNote.textContent = clutch.note; });
