// The Curate page: the shelf's takes with their verdicts and keep labels,
// the job that makes the labels, a look at each take's labels, and the
// conversion into a dataset with randomised backgrounds.
// Everything it shows comes down its socket (live.js) as it changes.

import { setPage, watch } from './live.js';
import { sourceForm } from './source-form.js';

const SIDES = ['left', 'right'];

let takes = [];          // complete takes, newest first
let verdicts = {};
let labels = {};         // take -> its masks.json account
let job = null;          // the labels job, as the server has it
let machine = null;      // where SAM 3 runs, and whether it is ready there
let converting = false;
const got = new Set();   // episodes, verdicts and masks once each has come: until then, loading
let judging = 0;         // takes whose replay is still being solved for their verdict
const posting = { labels: false, convert: false };   // a start asked for, its answer not back yet
let stopping = false;    // a Stop asked for, the job not ended yet

async function ask(path, body) {
  const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify(body ?? {}) });
  return r.json();
}
function fmtBytes(n) { return n > 1e9 ? `${(n / 1e9).toFixed(1)} GB` : n > 1e6 ? `${(n / 1e6).toFixed(0)} MB` : `${Math.round(n / 1e3)} kB`; }
function cell(text, className) {
  const td = document.createElement('td');
  td.textContent = text;
  if (className) td.className = className;
  return td;
}
// Something on its way, said where it will show: the turning ring and the words.
function waiting(el, text) {
  const ring = document.createElement('span');
  ring.className = 'spin';
  ring.setAttribute('aria-hidden', 'true');
  el.replaceChildren(ring, text);
  el.classList.add('loading');
}
function say(el, text) {
  el.textContent = text;
  el.classList.remove('loading');
}
const loaded = () => got.has('episodes') && got.has('verdicts');

// A video on its way: its overlay says so - after a moment, so one already
// at hand does not flash it - until `until` fires on it, or it fails.
// Answers whether it came.
const waits = new WeakMap();
function awaitVideo(video, text, until) {
  const wait = video.nextElementSibling;
  const gen = (waits.get(video) || 0) + 1;
  waits.set(video, gen);
  const timer = setTimeout(() => {
    if (waits.get(video) !== gen) return;
    waiting(wait, text);
    wait.hidden = false;
  }, 200);
  return new Promise((resolve) => {
    const done = (came) => {
      if (waits.get(video) !== gen) return;
      waits.set(video, gen + 1);
      clearTimeout(timer);
      if (came) wait.hidden = true;
      else { say(wait, 'this video could not be loaded'); wait.hidden = false; }
      resolve(came);
    };
    video.addEventListener(until, () => done(true), { once: true });
    video.addEventListener('error', () => done(false), { once: true });
  });
}
// A player put away: whatever it was waiting for no longer says anything.
function forgetVideo(video) {
  waits.set(video, (waits.get(video) || 0) + 1);
  video.nextElementSibling.hidden = true;
}

// --- the task's things ---------------------------------------------------------------
// Named in words, one at a time, kept in this browser until labels are made
// with them; then the labels' own are the starting point.
let named = null;
try { named = JSON.parse(localStorage.getItem('pinkrab.things') || 'null'); } catch { /* no storage, or old */ }
if (!Array.isArray(named)) named = null;
function things() { return named || []; }
function setThings(list) {
  named = [...new Set(list.map((t) => t.trim()).filter(Boolean))];
  try { localStorage.setItem('pinkrab.things', JSON.stringify(named)); } catch { /* no storage */ }
  drawThings();
  drawTakes();
  look();
}
function sameThings(a, b) { return (a || []).join('\n') === (b || []).join('\n'); }
const passes = (name) => { const v = verdicts[name]; return !!v && !v.error && !(v.fails || []).length; };
// The takes a Make labels would do: those that pass, without labels for these things.
function wanting() {
  const asked = things();
  return takes.filter((e) => passes(e.name) && !sameThings(labels[e.name]?.things, asked)).map((e) => e.name);
}

// --- the takes ------------------------------------------------------------------------
const takesBody = document.querySelector('#takes tbody');
function flagCount(account) {
  return Object.values(account?.flags || {}).reduce((n, spans) => n + spans.length, 0);
}
// A take worth a look: labels not made, made for other things, being made,
// or with spans flagged.
const PAGE = 20;
let showAll = false;
let takesPage = 0;
function needsLook(e, making) {
  const account = labels[e.name];
  return !account || !sameThings(account.things, things()) || e.name === making || flagCount(account) > 0;
}

function drawTakes() {
  const count = document.getElementById('takes-count');
  const empty = document.getElementById('takes-empty');
  if (!loaded()) {
    // Until the shelf and its verdicts have come: a row that says so, not
    // an empty list that reads as no takes.
    const note = document.createElement('span');
    waiting(note, 'loading the takes ...');
    const td = Object.assign(document.createElement('td'), { colSpan: 5 });
    td.append(note);
    const tr = Object.assign(document.createElement('tr'), { className: 'loading-row' });
    tr.append(td);
    takesBody.replaceChildren(tr);
    say(count, '');
    empty.hidden = true;
    document.getElementById('takes-pager').hidden = true;
    drawLabels();
    drawConvert();
    drawTakeChoice();
    return;
  }
  const making = job?.running ? job.at?.take : null;
  const passing = takes.filter((e) => passes(e.name));
  const look = passing.filter((e) => needsLook(e, making))
    .sort((a, b) => flagCount(labels[b.name]) - flagCount(labels[a.name]));
  const counted = passing.length
    ? `${passing.length} takes that pass · ${look.length ? `${look.length} to look at` : 'nothing to look at'}` : '';
  say(count, counted);
  if (judging) {
    // The ring by the words for what is still coming, not by the whole count.
    const still = document.createElement('span');
    waiting(still, `judging ${judging} take${judging === 1 ? '' : 's'}`);
    count.replaceChildren(counted ? `${counted} · ` : '', still);
  }
  const showLook = document.getElementById('show-look');
  const showAllEl = document.getElementById('show-all');
  showLook.textContent = `To look at (${look.length})`;
  showAllEl.textContent = `All (${passing.length})`;
  showLook.classList.toggle('on', !showAll);
  showAllEl.classList.toggle('on', showAll);
  showLook.setAttribute('aria-pressed', String(!showAll));
  showAllEl.setAttribute('aria-pressed', String(showAll));
  const list = showAll ? passing : look;
  const pages = Math.max(1, Math.ceil(list.length / PAGE));
  takesPage = Math.min(takesPage, pages - 1);
  const page = list.slice(takesPage * PAGE, (takesPage + 1) * PAGE);
  empty.hidden = page.length > 0;
  if (takes.length && !passing.length && judging) waiting(empty, `judging the takes - ${judging} to go`);
  else {
    say(empty, !takes.length ? 'no takes on the shelf - record them on the Collect page'
      : !passing.length ? 'no take passes yet - see the verdicts on the Collect page'
        : 'every take has its labels and nothing flagged');
  }
  takesBody.replaceChildren(...page.map((e) => {
    const tr = document.createElement('tr');
    tr.dataset.name = e.name;
    const account = labels[e.name];
    const made = account ? account.things.join(', ') + (sameThings(account.things, things()) ? '' : ' (other things)') : 'none';
    const labelCell = cell(e.name === making ? 'making ...' : made, account ? 'labels made' : 'labels');
    const n = flagCount(account);
    const flagCell = cell(account ? (n ? `${n} span${n === 1 ? '' : 's'}` : 'none') : '', `look${n ? ' some' : ''}`);
    const td = document.createElement('td');
    const open = document.createElement('button');
    open.className = 'play';
    open.textContent = 'Review';
    open.disabled = !account;
    open.addEventListener('click', () => openReview(e.name));
    td.append(open);
    tr.append(cell(e.name, 'name'), cell(e.seconds == null ? '' : e.seconds.toFixed(1)), labelCell, flagCell, td);
    return tr;
  }));
  const pager = document.getElementById('takes-pager');
  pager.hidden = list.length <= PAGE;
  document.getElementById('takes-where').textContent = list.length
    ? `${takesPage * PAGE + 1}–${takesPage * PAGE + page.length} of ${list.length}` : '';
  document.getElementById('takes-prev').disabled = takesPage === 0;
  document.getElementById('takes-next').disabled = takesPage >= pages - 1;
  drawLabels();
  drawConvert();
  drawTakeChoice();
}
document.getElementById('show-look').addEventListener('click', () => { showAll = false; takesPage = 0; drawTakes(); });
document.getElementById('show-all').addEventListener('click', () => { showAll = true; takesPage = 0; drawTakes(); });
document.getElementById('takes-prev').addEventListener('click', () => { takesPage -= 1; drawTakes(); });
document.getElementById('takes-next').addEventListener('click', () => { takesPage += 1; drawTakes(); });

watch('episodes', (answer) => {
  got.add('episodes');
  takes = (answer?.episodes || []).filter((e) => e.complete).sort((a, b) => b.name.localeCompare(a.name));
  drawTakes();
});
watch('verdicts', (answer) => {
  got.add('verdicts');
  verdicts = answer?.verdicts || {};
  judging = (answer?.pending || []).length;
  drawTakes();
});

// --- making the labels -----------------------------------------------------------------
const labelsGo = document.getElementById('labels-go');
const labelsStop = document.getElementById('labels-stop');
const labelsProgress = document.getElementById('labels-progress');
const labelsLines = document.getElementById('labels-lines');
const labelsNote = document.getElementById('labels-note');

function drawLabels() {
  const running = !!job?.running;
  const n = wanting().length;
  labelsGo.disabled = running || posting.labels || !things().length || n === 0 || !machine?.ready;
  labelsGo.classList.toggle('busy', running || posting.labels);
  labelsGo.textContent = posting.labels ? 'starting ...' : running ? 'making ...'
    : n ? `Make labels (${n} take${n === 1 ? '' : 's'})` : 'Make labels';
  if (!running) stopping = false;
  labelsStop.hidden = !running;
  labelsStop.disabled = stopping;
  labelsStop.classList.toggle('busy', stopping);
  labelsStop.textContent = stopping ? 'stopping ...' : 'Stop';
  if (!job?.asked) { labelsProgress.hidden = true; return; }
  labelsProgress.hidden = false;
  const total = job.asked.takes.length;
  const at = job.at || {};
  let within = 0;
  // Each take's two videos, each followed forward and then, for a thing
  // found late, back: the forward pass is most of a video's time.
  if (at.frames) {
    const video = at.video === 'gripper_right' ? 1 : 0;
    within = (video + (at.stage === 'back' ? 1 : at.frame / at.frames)) / 2;
  }
  const finished = job.done.length + Object.keys(job.failed).length;
  const share = running ? ((at.index ?? finished) + within) / total : 1;
  labelsProgress.querySelector('.bar > span').style.width = `${Math.min(100, share * 100)}%`;
  const on = job.asked.machine && job.asked.machine !== 'this machine' ? ` on ${job.asked.machine}` : '';
  const head = running
    ? `${finished} / ${total} takes${on} · ${job.seconds ?? 0} s` + (at.take ? ` · ${at.take}` : '')
      + (at.stage === 'forward' ? ` · ${at.video?.replace('gripper_', '')} · frame ${at.frame} / ${at.frames}`
        : at.stage === 'back' ? ` · ${at.video?.replace('gripper_', '')} · ${job.asked.things[at.thing] ?? ''} back to the start`
          : ' · grippers')
    : job.stopped ? `stopped: ${job.done.length} of ${total} takes made` : `done: ${job.done.length} of ${total} takes made in ${job.seconds} s`;
  labelsLines.textContent = [head, ...Object.entries(job.failed).map(([name, why]) => `  ${name}: ${why}`)].join('\n');
}

labelsGo.addEventListener('click', async () => {
  labelsNote.textContent = '';
  posting.labels = true;
  drawLabels();
  try {
    const answer = await ask('/masks', { takes: wanting(), things: things() });
    if (answer.error) labelsNote.textContent = answer.error;
    else job = answer;
  } catch (err) { labelsNote.textContent = String(err); }
  posting.labels = false;
  drawTakes();
});
// Stopped at the next frame SAM 3 reports, not at once: the button says so
// until the job has ended.
labelsStop.addEventListener('click', async () => {
  stopping = true;
  drawLabels();
  await ask('/masks/stop', {}).catch(() => { stopping = false; });
  drawLabels();
});

// --- where SAM 3 runs ----------------------------------------------------------------
const samMachine = document.getElementById('sam-machine');
const samState = document.getElementById('sam-state');
const samRetry = document.getElementById('sam-retry');
const samAdd = document.getElementById('sam-add');
const samSource = document.getElementById('sam-source');
function drawMachine() {
  const names = machine?.machines || [];
  if ([...samMachine.options].map((o) => o.value).join('\n') !== names.join('\n')) {
    samMachine.replaceChildren(...names.map((name) => Object.assign(document.createElement('option'), { value: name, textContent: name })));
  }
  samMachine.value = machine?.name || '';
  samMachine.disabled = !machine || machine.busy || !!job?.running;
  samAdd.disabled = samMachine.disabled;
  const lines = machine?.lines || [];
  if (!machine) {
    if (got.has('masks')) say(samState, '');
    else waiting(samState, 'finding where SAM 3 runs ...');
  } else if (machine.busy) waiting(samState, lines[lines.length - 1] || 'getting it ready ...');
  else say(samState, machine.error ? machine.error : 'ready');
  samState.classList.toggle('bad', !!machine?.error);
  samRetry.hidden = !machine?.error;
}
samMachine.addEventListener('change', async () => {
  samMachine.disabled = true;
  samAdd.disabled = true;
  waiting(samState, `switching to ${samMachine.value} ...`);
  samState.classList.remove('bad');
  const answer = await ask('/masks/machine', { name: samMachine.value }).catch((err) => ({ error: String(err) }));
  if (answer.error) {
    say(samState, answer.error);
    samState.classList.add('bad');
    samMachine.disabled = samAdd.disabled = false;
    return;
  }
  machine = answer;
  drawMachine();
  drawLabels();
});
// Saved, the new machine is the one SAM 3 runs on.
const samForm = sourceForm(document.getElementById('sam-source-form'), async (name) => {
  const answer = await ask('/masks/machine', { name });
  if (answer.error) throw new Error(answer.error);
  machine = answer;
  drawMachine();
  drawLabels();
  samSource.close();
});
samAdd.addEventListener('click', () => { samForm.say(''); samSource.showModal(); });
document.getElementById('sam-source-close').addEventListener('click', () => samSource.close());
samRetry.addEventListener('click', async () => {
  samRetry.disabled = true;
  samRetry.classList.add('busy');
  const answer = await ask('/masks/machine', {}).catch((err) => ({ error: String(err) }));
  samRetry.disabled = false;
  samRetry.classList.remove('busy');
  if (!answer.error) { machine = answer; drawMachine(); }
});

watch('masks', (answer) => {
  got.add('masks');
  labels = answer?.takes || {};
  const wasMaking = !!job?.running;
  job = answer?.job || null;
  const wasReady = !!machine?.ready && machine?.name === answer?.machine?.name;
  machine = answer?.machine || null;
  drawMachine();
  // Ready now, on a machine the preview has not asked yet, or free again
  // after making labels: the things looked for again.
  if (machine?.ready && things().length && (!wasReady || (wasMaking && !answer?.job?.running))) look();
  // Until some are named here: the things last asked for, else the latest labels'.
  if (named === null) {
    const latest = Object.values(labels).sort((a, b) => (b.made || 0) - (a.made || 0))[0];
    const start = job?.asked?.things || latest?.things;
    if (start) { named = [...start]; drawThings(); look(); }
  }
  drawTakes();
});

// --- the labeler: one frame, the grippers drawn, the things looked for ----------------
const lookTake = document.getElementById('look-take');
const lookLeft = document.getElementById('look-left');
const lookRight = document.getElementById('look-right');
const lookFrame = document.getElementById('look-frame');
const lookAt = document.getElementById('look-at');
const lookBox = document.getElementById('look-box');
const lookWait = document.getElementById('look-wait');
const lookCanvas = document.getElementById('look-canvas');
const thingsList = document.getElementById('things-list');
const thingInput = document.getElementById('thing-input');
let lookSide = 'left';
let seen = null;          // the last answer: picture, grippers, things found
let picture = null;       // its picture, decoded
let lit = null;           // the thing whose row the pointer is on
let asking = false;
let again = false;        // asked for something else while a look was out
let failed = false;       // the last look came back without a picture

const css = (name, fallback) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;

function trace(g, flat, sx, sy) {
  g.moveTo(flat[0] * sx, flat[1] * sy);
  for (let i = 2; i < flat.length; i += 2) g.lineTo(flat[i] * sx, flat[i + 1] * sy);
  g.closePath();
}

function drawLook() {
  const box = lookCanvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(box.width * devicePixelRatio));
  const h = Math.max(1, Math.round(box.height * devicePixelRatio));
  lookCanvas.width = w;
  lookCanvas.height = h;
  const g = lookCanvas.getContext('2d');
  g.fillStyle = '#141416';
  g.fillRect(0, 0, w, h);
  if (!seen || !picture) return;
  const [ph, pw] = seen.size;
  const sx = w / pw;
  const sy = h / ph;
  g.drawImage(picture, 0, 0, w, h);
  // What is not kept, dimmed: a veil with the kept shapes cut out of it.
  const veil = document.createElement('canvas');
  veil.width = w;
  veil.height = h;
  const v = veil.getContext('2d');
  v.fillStyle = 'rgba(20, 12, 18, 0.62)';
  v.fillRect(0, 0, w, h);
  v.globalCompositeOperation = 'destination-out';
  v.beginPath();
  for (const flat of seen.grippers) trace(v, flat, sx, sy);
  for (const t of seen.things) for (const flat of t.outlines) trace(v, flat, sx, sy);
  v.fill();
  g.drawImage(veil, 0, 0);
  const lw = 2 * devicePixelRatio;
  g.lineWidth = lw;
  g.setLineDash([6 * devicePixelRatio, 4 * devicePixelRatio]);
  g.strokeStyle = '#ffffff';
  g.beginPath();
  for (const flat of seen.grippers) trace(g, flat, sx, sy);
  g.stroke();
  g.setLineDash([]);
  const pink = css('--shell', '#e2488e');
  g.font = `600 ${12 * devicePixelRatio}px Pretendard, system-ui, sans-serif`;
  g.textBaseline = 'top';
  for (const t of seen.things) {
    if (!t.outlines.length) continue;
    g.beginPath();
    for (const flat of t.outlines) trace(g, flat, sx, sy);
    if (t.text === lit) { g.fillStyle = 'rgba(226, 72, 142, 0.38)'; g.fill(); }
    g.strokeStyle = pink;
    g.stroke();
    // Its name by its top-left corner, on the studio's pink.
    let x = Infinity;
    let y = Infinity;
    for (const flat of t.outlines) for (let i = 0; i < flat.length; i += 2) { x = Math.min(x, flat[i] * sx); y = Math.min(y, flat[i + 1] * sy); }
    const pad = 4 * devicePixelRatio;
    const tw = g.measureText(t.text).width + 2 * pad;
    const th = 18 * devicePixelRatio;
    const tx = Math.min(Math.max(0, x), w - tw);
    const ty = Math.max(0, y - th - 2 * devicePixelRatio);
    g.fillStyle = css('--shell-deep', '#b8306f');
    g.fillRect(tx, ty, tw, th);
    g.fillStyle = '#ffffff';
    g.fillText(t.text, tx + pad, ty + 3 * devicePixelRatio);
  }
}

function drawThings() {
  const found = Object.fromEntries((seen?.things || []).map((t) => [t.text, t]));
  thingsList.replaceChildren(...things().map((text) => {
    const li = document.createElement('li');
    const t = found[text];
    const name = Object.assign(document.createElement('span'), { className: 'thing-name', textContent: text });
    const n = t ? t.scores.length : null;
    const here = Object.assign(document.createElement('span'), {
      className: `thing-here${n === 0 ? ' none' : ''}`,
      textContent: n == null ? '' : n ? `${n} here` : 'not in this frame',
      title: t?.scores.length ? `scores ${t.scores.join(', ')}` : '',
    });
    const drop = document.createElement('button');
    drop.className = 'icon';
    drop.title = `remove ${text}`;
    drop.setAttribute('aria-label', `remove ${text}`);
    drop.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18"/></svg>';
    drop.addEventListener('click', () => setThings(things().filter((x) => x !== text)));
    li.append(name, here, drop);
    li.addEventListener('pointerenter', () => { lit = text; drawLook(); });
    li.addEventListener('pointerleave', () => { lit = null; drawLook(); });
    return li;
  }));
}

document.getElementById('thing-add').addEventListener('submit', (event) => {
  event.preventDefault();
  const text = thingInput.value.trim();
  if (!text) return;
  thingInput.value = '';
  setThings([...things(), text]);
});

// The takes the labeler can show: those that pass, newest first.
function drawTakeChoice() {
  const passing = takes.filter((e) => passes(e.name)).map((e) => e.name);
  const was = lookTake.value;
  const same = [...lookTake.options].map((o) => o.value).join() === passing.join();
  if (!same) {
    lookTake.replaceChildren(...passing.map((name) => Object.assign(document.createElement('option'), { value: name, textContent: name })));
    lookTake.value = passing.includes(was) ? was : passing[0] || '';
    if (lookTake.value !== was) { lookFrame.value = '0'; look(); }
  }
  const ep = takes.find((e) => e.name === lookTake.value);
  const n = ep?.frames?.[`gripper_${lookSide}`] || 1;
  lookFrame.max = String(n - 1);
  lookAt.textContent = `frame ${lookFrame.value} / ${n - 1}`;
  // No take to show yet: the empty frame says whether one is coming.
  if (!lookTake.value && !asking) {
    lookWait.hidden = loaded();
    lookWait.classList.add('first');
    if (!loaded()) waiting(lookWait, 'loading the takes ...');
  }
}

// Until SAM 3 is ready where it runs, the frame and its grippers alone.
async function look() {
  if (!lookTake.value) return;
  // Said at once, even when this waits for the look already out: the frame
  // never shows the last answer while the next is on its way. The first
  // picture in the middle of the empty frame, a new look in its corner.
  lookBox.setAttribute('aria-busy', 'true');
  lookWait.hidden = false;
  lookWait.classList.toggle('first', !seen);
  const sam = machine?.ready && things().length;
  waiting(lookWait, seen ? (sam ? 'looking for the things ...' : 'loading the frame ...')
    : sam ? 'starting SAM 3 - the first look takes up to a minute' : 'loading the frame ...');
  if (asking) { again = true; return; }
  asking = true;
  const note = document.getElementById('labels-note');
  try {
    const answer = await ask('/masks/look', { take: lookTake.value, side: lookSide, frame: Number(lookFrame.value), things: machine?.ready ? things() : [] });
    if (answer.error) {
      // No picture of this frame: the frame is emptied rather than left on
      // another take's, and says why beside the note.
      note.textContent = answer.error;
      seen = null;
      picture = null;
      failed = true;
    } else {
      failed = false;
      note.textContent = answer.note || '';
      const img = new Image();
      img.src = answer.picture;
      await img.decode();
      seen = answer;
      picture = img;
    }
  } catch (err) { note.textContent = String(err); seen = null; picture = null; failed = true; }
  asking = false;
  drawThings();
  drawLook();
  // Asked again meanwhile, for a take still there: that look carries on the waiting.
  const next = again && !!lookTake.value;
  again = false;
  if (next) { look(); return; }
  lookBox.setAttribute('aria-busy', 'false');
  lookWait.hidden = !failed;
  lookWait.classList.toggle('first', failed);
  if (failed) say(lookWait, 'no picture of this frame - see the note beside it');
}

function pickSide(side) {
  lookSide = side;
  lookLeft.classList.toggle('on', side === 'left');
  lookRight.classList.toggle('on', side === 'right');
  lookLeft.setAttribute('aria-pressed', String(side === 'left'));
  lookRight.setAttribute('aria-pressed', String(side === 'right'));
  drawTakeChoice();
  look();
}
lookLeft.addEventListener('click', () => pickSide('left'));
lookRight.addEventListener('click', () => pickSide('right'));
lookTake.addEventListener('change', () => { lookFrame.value = '0'; drawTakeChoice(); look(); });
lookFrame.addEventListener('input', drawTakeChoice);
lookFrame.addEventListener('change', look);
addEventListener('resize', drawLook);
drawThings();
drawLook();

// --- converting -----------------------------------------------------------------------
const convertOut = document.getElementById('convert-out');
const convertTask = document.getElementById('convert-task');
const convertGo = document.getElementById('convert-go');
const convertProgress = document.getElementById('convert-progress');
const convertLines = document.getElementById('convert-lines');
const convertNote = document.getElementById('convert-note');
const convertView = document.getElementById('convert-view');
const bgNote = document.getElementById('bg-note');
try { convertTask.value = localStorage.getItem('pinkrab.task') || ''; } catch { /* no storage */ }

function drawConvert() {
  const passing = takes.filter((e) => passes(e.name));
  const labelled = passing.filter((e) => labels[e.name]).length;
  document.getElementById('labels-state').textContent = passing.length ? `${labelled} of ${passing.length} takes that pass labelled` : '';
  document.getElementById('convert-state').textContent = `${passing.length} take${passing.length === 1 ? '' : 's'} pass`;
  // As the server decides it: randomised once any take has labels, a
  // take without them then left out; as recorded while none has.
  const bare = passing.length - labelled;
  bgNote.textContent = !passing.length ? ''
    : !labelled ? `${passing.length} takes, the pictures as recorded - no take has keep labels yet (step 1)`
      : `${labelled} takes, each with a background of its own outside step 1's labels`
        + (bare ? ` - ${bare} without labels left out; make them in step 1 to put them in` : '');
  if (!loaded()) {
    say(document.getElementById('labels-state'), '');
    say(document.getElementById('convert-state'), '');
    bgNote.textContent = '';
  }
  convertGo.disabled = converting || posting.convert || passing.length === 0;
  convertGo.classList.toggle('busy', converting || posting.convert);
  convertGo.textContent = posting.convert ? 'starting ...' : converting ? 'converting ...' : 'Convert';
}

convertGo.addEventListener('click', async () => {
  convertNote.textContent = '';
  try { localStorage.setItem('pinkrab.task', convertTask.value); } catch { /* no storage */ }
  posting.convert = true;
  drawConvert();
  const answer = await ask('/convert', {
    out: convertOut.value.trim(), task: convertTask.value.trim(),
  }).catch((err) => ({ error: String(err) }));
  posting.convert = false;
  if (answer.error) { convertNote.textContent = answer.error; drawConvert(); return; }
  showConvert(answer);
});

function showConvert(c) {
  const was = converting;
  converting = !!c?.running;
  drawConvert();
  if (c?.asked && (c.running || c.result || c.error)) {
    convertProgress.hidden = false;
    convertProgress.querySelector('.bar > span').style.width = `${c.total ? (c.done / c.total) * 100 : 0}%`;
    const head = c.running ? `${c.done} / ${c.total} episodes · ${c.seconds ?? 0} s`
      : c.error ? `failed: ${c.error}` : `done: ${c.result?.dataset} in ${c.result?.seconds} s`;
    convertLines.textContent = [head, ...(c.lines || [])].join('\n');
  }
  // Made: the dataset a click away, as training will read it.
  const made = !c?.running && c?.result ? c.asked?.out : null;
  convertView.hidden = !made;
  convertView.onclick = made ? () => openDataset(made, convertView) : null;
  // Done well, the name is cleared for the next one; failed, it stays to try again.
  if (was && !converting && !c?.error) convertOut.value = '';
}
watch('convert', showConvert);

const datasetsEmpty = document.getElementById('datasets-empty');
waiting(datasetsEmpty, 'loading the datasets ...');
watch('datasets', (answer) => {
  const list = answer?.datasets || [];
  document.getElementById('datasets-count').textContent = list.length ? `${list.length} · ${answer.dir}` : answer?.dir || '';
  document.querySelector('#datasets tbody').replaceChildren(...list.map((d) => {
    const tr = document.createElement('tr');
    tr.append(cell(d.name, 'name'), cell(d.episodes ?? ''), cell(d.frames ?? ''),
      cell(d.randomised ? 'randomised' : 'as recorded'), cell(d.task ?? ''),
      cell(fmtBytes(d.bytes || 0)), cell(`${answer.dir}/${d.name}`, 'path'));
    const td = document.createElement('td');
    td.className = 'actions';
    const view = document.createElement('button');
    view.className = 'play';
    view.textContent = 'View';
    view.addEventListener('click', () => openDataset(d.name, view));
    td.append(view);
    const del = document.createElement('button');
    del.textContent = 'delete';
    del.addEventListener('click', async () => {
      if (!del.classList.contains('sure')) {
        del.classList.add('sure'); del.textContent = 'sure?';
        // Unasked within four seconds, the ask lapses - unless the delete is on its way.
        setTimeout(() => { if (!del.disabled) { del.classList.remove('sure'); del.textContent = 'delete'; } }, 4000);
        return;
      }
      del.disabled = true;
      del.classList.add('busy');
      del.textContent = 'deleting ...';
      const answer = await ask('/datasets/delete', { name: d.name }).catch((err) => ({ error: String(err) }));
      // Gone, the row goes with the next list; not gone, the button is back and says why.
      if (answer.error) {
        document.getElementById('datasets-note').textContent = answer.error;
        del.disabled = false;
        del.classList.remove('busy', 'sure');
        del.textContent = 'delete';
      }
    });
    td.append(del);
    tr.append(td);
    return tr;
  }));
  say(datasetsEmpty, 'no datasets yet');
  datasetsEmpty.hidden = list.length > 0;
});

// --- the review ------------------------------------------------------------------------
const dialog = document.getElementById('review');
const videos = { left: document.getElementById('review-left'), right: document.getElementById('review-right') };
const scrub = document.getElementById('review-scrub');
const frameEl = document.getElementById('review-frame');
const playEl = document.getElementById('review-play');
const flagsCanvas = document.getElementById('review-flags');
const listEl = document.getElementById('review-list');
const noteEl = document.getElementById('review-note');
let reviewing = null;    // the take open

// The left video is the clock; the right follows it. Frames are counted on
// the left video's.
function frames() {
  return labels[reviewing]?.frames?.gripper_left || Math.max(1, Math.round((videos.left.duration || 0) * 30));
}
function frameAt(t) { const d = videos.left.duration; return d ? Math.min(frames() - 1, Math.round((t / d) * frames())) : 0; }
function seek(frame) {
  const d = videos.left.duration;
  if (!d) return;
  const t = (frame / frames()) * d;
  for (const side of SIDES) {
    const v = videos[side];
    if (v.duration) v.currentTime = Math.min(v.duration, t);
  }
  showFrame();
}
function showFrame() {
  const f = frameAt(videos.left.currentTime);
  scrub.value = String(f);
  frameEl.textContent = `frame ${f} / ${frames() - 1}`;
}

function drawFlags() {
  const account = labels[reviewing];
  const rect = flagsCanvas.getBoundingClientRect();
  flagsCanvas.width = Math.max(1, Math.round(rect.width * devicePixelRatio));
  flagsCanvas.height = Math.max(1, Math.round(rect.height * devicePixelRatio));
  const g = flagsCanvas.getContext('2d');
  const css = getComputedStyle(document.documentElement);
  g.fillStyle = css.getPropertyValue('--line').trim() || '#ecd9e2';
  g.fillRect(0, 0, flagsCanvas.width, flagsCanvas.height);
  g.fillStyle = css.getPropertyValue('--shell-deep').trim() || '#b8306f';
  const half = flagsCanvas.height / 2;
  SIDES.forEach((side, row) => {
    const n = account?.frames?.[`gripper_${side}`] || 1;
    for (const span of account?.flags?.[`gripper_${side}`] || []) {
      const x0 = (span.start / n) * flagsCanvas.width;
      const x1 = Math.max(x0 + 2, (span.end / n) * flagsCanvas.width);
      g.fillRect(x0, row * half + 1, x1 - x0, half - 2);
    }
  });
}

function drawList() {
  const account = labels[reviewing];
  const rows = SIDES.flatMap((side) => (account?.flags?.[`gripper_${side}`] || []).map((span) => ({ side, ...span })));
  rows.sort((a, b) => a.start - b.start);
  const found = SIDES.map((side) => {
    const n = account?.frames?.[`gripper_${side}`] || 0;
    const each = Object.entries(account?.found?.[`gripper_${side}`] || {})
      .map(([thing, k]) => `${thing} ${n ? Math.round((k / n) * 100) : 0}%`).join(', ');
    return `${side}: ${each || 'no things'}`;
  }).join(' · ');
  noteEl.textContent = `found in ${found}`;
  if (!rows.length) {
    listEl.replaceChildren(Object.assign(document.createElement('li'), { className: 'quiet', textContent: 'no span worth a look' }));
    return;
  }
  listEl.replaceChildren(...rows.map((r) => {
    const li = document.createElement('li');
    const go = document.createElement('button');
    go.textContent = `${r.start}`;
    go.title = 'go to this frame';
    go.addEventListener('click', () => { pause(); seek(r.start); });
    li.append(go, `${r.side} · ${r.thing} ${r.why === 'lost' ? 'lost' : 'jumped'} · frames ${r.start}–${r.end - 1}`);
    return li;
  }));
}

function load() {
  const at = videos.left.currentTime || 0;
  // A take's review videos are made on their first ask, a few seconds each:
  // the players say so, and Play waits for both.
  const take = reviewing;
  playEl.disabled = scrub.disabled = true;
  Promise.all(SIDES.map((side) => awaitVideo(videos[side], 'making the review video ...', 'loadeddata')))
    .then((came) => { if (reviewing === take && came.every(Boolean)) playEl.disabled = scrub.disabled = false; });
  for (const side of SIDES) {
    videos[side].src = `/curate/review/${reviewing}/gripper_${side}.mp4?made=${labels[reviewing]?.made || 0}`;
  }
  videos.left.addEventListener('loadedmetadata', () => {
    videos.left.currentTime = at; videos.right.currentTime = at; scrub.max = String(frames() - 1); showFrame();
  },
    { once: true });
}

function openReview(name) {
  reviewing = name;
  document.getElementById('review-title').textContent = name;
  scrub.max = String(frames() - 1);
  scrub.value = '0';
  for (const side of SIDES) videos[side].removeAttribute('src');
  load();
  dialog.showModal();
  drawFlags();
  drawList();
  showFrame();
}

function pause() { for (const side of SIDES) videos[side].pause(); playEl.textContent = 'Play'; }
playEl.addEventListener('click', () => {
  if (videos.left.paused) {
    // Played to the end, both start over: a video that has ended starts
    // over by itself, and the other, set to that end, would stay there.
    const t = videos.left.ended || videos.left.currentTime >= videos.left.duration - 0.05 ? 0 : videos.left.currentTime;
    for (const side of SIDES) videos[side].currentTime = t;
    for (const side of SIDES) videos[side].play().catch(() => {});
    playEl.textContent = 'Pause';
  } else pause();
});
videos.left.addEventListener('timeupdate', showFrame);
videos.left.addEventListener('ended', pause);
scrub.addEventListener('input', () => { pause(); seek(Number(scrub.value)); });
document.getElementById('review-close').addEventListener('click', () => dialog.close());
dialog.addEventListener('close', () => {
  pause();
  reviewing = null;
  for (const side of SIDES) { forgetVideo(videos[side]); videos[side].removeAttribute('src'); }
});
addEventListener('resize', () => { if (reviewing) drawFlags(); });

// --- a dataset as made ----------------------------------------------------------------
const dv = document.getElementById('dataset-view');
const dvVideos = { left: document.getElementById('dv-left'), right: document.getElementById('dv-right') };
const dvScrub = document.getElementById('dv-scrub');
const dvFrame = document.getElementById('dv-frame');
const dvPlay = document.getElementById('dv-play');
const dvList = document.getElementById('dv-episodes');
let shown = null;        // the dataset open: its episodes, from the server
let ep = null;           // the episode playing

const camera = (side) => `observation.images.gripper_${side}`;
// Each camera's span of its file for the episode, the left one the clock.
const span = (side) => ep.videos[camera(side)];

function dvAt() {
  const { from } = span('left');
  return Math.max(0, Math.min(ep.length - 1, Math.floor((dvVideos.left.currentTime - from) * shown.fps + 1e-3)));
}
function dvShowFrame() {
  if (!ep) return;
  const k = dvAt();
  dvScrub.value = String(k);
  dvFrame.textContent = `frame ${k} / ${ep.length - 1}`;
}
function dvSeek(k) {
  // To the middle of the frame, so the picture shown is that frame's.
  for (const side of SIDES) dvVideos[side].currentTime = span(side).from + (k + 0.5) / shown.fps;
  dvShowFrame();
}
function dvPause() { for (const side of SIDES) dvVideos[side].pause(); dvPlay.textContent = 'Play'; }

function pickEpisode(i) {
  dvPause();
  ep = shown.episodes[i];
  for (const li of dvList.children) li.classList.toggle('on', Number(li.dataset.i) === i);
  dvScrub.max = String(ep.length - 1);
  for (const side of SIDES) {
    const url = `/datasets/${shown.name}/${span(side).path}`;
    const v = dvVideos[side];
    // Shown once its first frame is: a new file to load, or a seek within one.
    awaitVideo(v, 'loading the episode ...', 'seeked');
    if (v.dataset.url !== url) {
      v.dataset.url = url;
      v.src = url;
    }
  }
  const go = () => dvSeek(0);
  if (dvVideos.left.readyState >= 1) go();
  else dvVideos.left.addEventListener('loadedmetadata', go, { once: true });
}

async function openDataset(name, button) {
  const note = document.getElementById('datasets-note');
  const label = button?.textContent;
  if (button) { button.disabled = true; button.classList.add('busy'); button.textContent = 'opening ...'; }
  let answer;
  try {
    const r = await fetch(`/datasets/${name}/episodes`, { redirect: 'error' });
    answer = await r.json();
  } catch (err) {
    // A studio started before this page was can answer with no such path.
    note.textContent = `could not open ${name} - restart the studio if it was started before this page changed (${err.message})`;
    return;
  } finally {
    if (button) { button.disabled = false; button.classList.remove('busy'); button.textContent = label; }
  }
  if (answer.error) { note.textContent = answer.error; return; }
  note.textContent = '';
  shown = answer;
  document.getElementById('dv-title').textContent = name;
  document.getElementById('dv-note').textContent = `${answer.episodes.length} episodes · ${answer.fps} Hz · `
    + `${answer.randomised ? 'backgrounds randomised' : 'as recorded'}${answer.task ? ` · ${answer.task}` : ''}`;
  dvList.replaceChildren(...answer.episodes.map((e, i) => {
    const li = document.createElement('li');
    li.dataset.i = String(i);
    li.innerHTML = '<button><span class="dv-i"></span><span class="dv-take"></span><span class="dv-s"></span></button>';
    li.querySelector('.dv-i').textContent = String(e.index);
    li.querySelector('.dv-take').textContent = e.source || '';
    li.querySelector('.dv-s').textContent = `${e.seconds.toFixed(1)} s`;
    li.querySelector('button').addEventListener('click', () => pickEpisode(i));
    return li;
  }));
  dv.showModal();
  pickEpisode(0);
}

dvPlay.addEventListener('click', () => {
  if (!ep) return;
  if (dvVideos.left.paused) {
    if (dvAt() >= ep.length - 1) dvSeek(0);
    for (const side of SIDES) dvVideos[side].play().catch(() => {});
    dvPlay.textContent = 'Pause';
  } else dvPause();
});
// An episode is a span of a file that holds others after it: playing stops at its end.
dvVideos.left.addEventListener('timeupdate', () => {
  if (!ep) return;
  if (dvVideos.left.currentTime >= span('left').to - 0.5 / shown.fps) {
    dvPause();
    dvSeek(ep.length - 1);
  }
  dvShowFrame();
});
dvScrub.addEventListener('input', () => { dvPause(); dvSeek(Number(dvScrub.value)); });
document.getElementById('dv-close').addEventListener('click', () => dv.close());
dv.addEventListener('close', () => {
  dvPause();
  ep = null;
  for (const side of SIDES) {
    forgetVideo(dvVideos[side]);
    dvVideos[side].removeAttribute('src');
    delete dvVideos[side].dataset.url;
    dvVideos[side].load();
  }
});

// Before the socket has said anything: each part says it is loading.
drawTakes();
drawMachine();
setPage('curate');
