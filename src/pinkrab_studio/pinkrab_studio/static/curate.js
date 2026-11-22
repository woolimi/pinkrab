// The Curate page: a take's frame with the robot drawn in as Convert will
// keep it, and the conversion of the takes that pass into a dataset.
// Everything it shows comes down its socket (live.js) as it changes.

import { setPage, watch } from './live.js';

const SIDES = ['left', 'right'];

let takes = [];          // complete takes, newest first
let verdicts = {};
let converting = false;
const got = new Set();   // episodes and verdicts once each has come: until then, loading
let judging = 0;         // takes whose replay is still being solved for their verdict
const posting = { convert: false };   // a start asked for, its answer not back yet

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

const passes = (name) => { const v = verdicts[name]; return !!v && !v.error && !(v.fails || []).length; };

function drawTakes() {
  const passing = takes.filter((e) => passes(e.name));
  const demos = passing.filter((e) => e.kind !== 'rollout');
  const state = document.getElementById('drawn-state');
  if (!loaded()) waiting(state, 'loading the takes ...');
  else if (judging) waiting(state, `judging ${judging} take${judging === 1 ? '' : 's'}`);
  else say(state, demos.length ? `${demos.length} demonstration${demos.length === 1 ? '' : 's'} pass` : '');
  drawConvert();
  drawTakeChoice();
}

watch('episodes', (answer) => {
  got.add('episodes');
  // A rollout is here for its corrections - what a dataset learns from it -
  // its seconds theirs; one with none has nothing to convert.
  takes = (answer?.episodes || [])
    .filter((e) => e.complete && (e.kind !== 'rollout' || (e.corrections || []).length))
    .map((e) => (e.kind === 'rollout'
      ? { ...e, seconds: e.corrections.reduce((s, c) => s + (c.seconds || 0), 0) } : e))
    .sort((a, b) => b.name.localeCompare(a.name));
  drawTakes();
});
watch('verdicts', (answer) => {
  got.add('verdicts');
  verdicts = answer?.verdicts || {};
  judging = (answer?.pending || []).length;
  drawTakes();
});

// --- a frame with the robot drawn in --------------------------------------------------
// Demonstrations only: a rollout's pictures show the robot already.
const lookTake = document.getElementById('look-take');
const lookLeft = document.getElementById('look-left');
const lookRight = document.getElementById('look-right');
const lookFrame = document.getElementById('look-frame');
const lookAt = document.getElementById('look-at');
const lookBox = document.getElementById('look-box');
const lookWait = document.getElementById('look-wait');
const lookPicture = document.getElementById('look-picture');
let lookSide = 'left';
let asking = false;
let again = false;        // asked for something else while a look was out
let drawnTake = null;     // the take the server has followed, which answers in a moment

function drawTakeChoice() {
  const passing = takes.filter((e) => passes(e.name) && e.kind !== 'rollout').map((e) => e.name);
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
  lookTake.disabled = lookFrame.disabled = lookLeft.disabled = lookRight.disabled = !lookTake.value;
  // No take to show yet: the empty frame says whether one is coming.
  if (!lookTake.value && !asking) {
    lookPicture.hidden = true;
    lookWait.hidden = false;
    lookWait.classList.add('first');
    if (!loaded() || judging) waiting(lookWait, 'loading the takes ...');
    else say(lookWait, 'no demonstration passes yet - see the verdicts on the Collect page');
  }
}

async function look() {
  if (!lookTake.value) return;
  // Said at once, even when this waits for the look already out. A take not
  // yet followed takes some seconds; a frame of the one followed, a moment.
  lookBox.setAttribute('aria-busy', 'true');
  lookWait.hidden = false;
  lookWait.classList.toggle('first', lookPicture.hidden);
  waiting(lookWait, drawnTake === lookTake.value ? 'drawing the frame ...'
    : 'following the take with the robot - some seconds ...');
  if (asking) { again = true; return; }
  asking = true;
  const take = lookTake.value;
  let failed = '';
  try {
    const answer = await ask('/curate/drawn', { take, side: lookSide, frame: Number(lookFrame.value) });
    if (answer.error) failed = answer.error;
    else {
      const img = new Image();
      img.src = answer.picture;
      await img.decode();
      lookPicture.src = img.src;
      lookPicture.hidden = false;
      drawnTake = take;
    }
  } catch (err) { failed = String(err); }
  asking = false;
  // Asked again meanwhile, for a take still there: that look carries on the waiting.
  const next = again && !!lookTake.value;
  again = false;
  if (next) { look(); return; }
  lookBox.setAttribute('aria-busy', 'false');
  if (failed) {
    // No picture of this frame: the frame is emptied rather than left on
    // another take's, and says why.
    lookPicture.hidden = true;
    lookWait.classList.add('first');
    say(lookWait, failed);
  } else lookWait.hidden = true;
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

// --- converting -----------------------------------------------------------------------
const convertOut = document.getElementById('convert-out');
const convertTask = document.getElementById('convert-task');
const convertGo = document.getElementById('convert-go');
const convertProgress = document.getElementById('convert-progress');
const convertLines = document.getElementById('convert-lines');
const convertNote = document.getElementById('convert-note');
const convertView = document.getElementById('convert-view');
try { convertTask.value = localStorage.getItem('pinkrab.task') || ''; } catch { /* no storage */ }

function drawConvert() {
  const passing = takes.filter((e) => passes(e.name));
  const state = document.getElementById('convert-state');
  if (!loaded()) say(state, '');
  else say(state, `${passing.length} take${passing.length === 1 ? '' : 's'} pass`);
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
    const counted = c.running && c.measuring != null ? c.measuring : c.done;
    convertProgress.querySelector('.bar > span').style.width = `${c.total ? (counted / c.total) * 100 : 0}%`;
    const head = c.running && c.measuring != null
      ? `measuring the gripper cameras' latency: ${c.measuring} / ${c.total} takes · ${c.seconds ?? 0} s`
      : c.running ? `${c.done} / ${c.total} episodes · ${c.seconds ?? 0} s`
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
      cell(d.robot_drawn ? 'robot drawn' : 'as recorded'), cell(d.task ?? ''),
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
    + `${answer.robot_drawn ? 'robot drawn' : 'as recorded'}${answer.task ? ` · ${answer.task}` : ''}`;
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
setPage('curate');
