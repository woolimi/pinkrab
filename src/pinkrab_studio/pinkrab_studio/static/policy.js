// The policy page: a checkpoint, the follower arms' gripper cameras, and what
// the policy asks for given them.
//
// Two loops. The cameras are the live server's own streams, started through
// the studio the way the collect page starts them - on the arms' cameras,
// named by `pinkrab cameras`, not the handhelds'. The answer comes down the
// page's socket (live.js) - state for the cards, and the steps since the last
// one for the traces, so a page opened late fills in rather than starting blank.

import { hideCam, metric, showStream, sourceSelect } from './cams.js';
import { say, setPage, until, watch } from './live.js';
import { Connection } from './connection.js';

const connectEl = document.getElementById('connect');
const dotEl = document.getElementById('live-dot');
const lineEl = document.getElementById('live-line');
const noteEl = document.getElementById('note');
const wantsEl = document.getElementById('wants');
const askEl = document.getElementById('ask');
const driveEl = document.getElementById('drive');
const haltEl = document.getElementById('halt');
const readyEl = document.getElementById('ready');
const homeEl = document.getElementById('home');
const statusEl = document.getElementById('status');
const checkpointEl = document.getElementById('checkpoint');
const commandField = document.getElementById('command-field');
const commandEl = document.getElementById('command');
const commandTasks = document.getElementById('command-tasks');
const ARMS = ['left', 'right'];

const cams = {
  left: document.getElementById('cam-gripper-left'),
  right: document.getElementById('cam-gripper-right'),
};
// Which camera is which arm's: chosen in the settings drawer, set once.
const sources = {
  left: document.getElementById('source-left'),
  right: document.getElementById('source-right'),
};
const posesEl = document.getElementById('poses');
// The settings drawer covers the page while open, Stop with it: it opens
// only while nothing runs, and a run that starts closes it.
const settingsOpen = document.getElementById('settings-open');
const drawer = document.getElementById('settings-drawer');
const checkpointState = document.getElementById('checkpoint-state');
// Shown beside them only while the hands have the arms - see showHead.
const headFig = document.getElementById('cam-head');
const cards = {
  left: document.getElementById('m-left'),
  right: document.getElementById('m-right'),
  rate: document.getElementById('m-rate'),
};

let liveUrl = null;
let camsUp = false;
// The robot, as the studio last reported it. The driver and the gate are the
// rig's own launch file, started in a terminal, not from here: this page looks
// for them and says so. What Connect starts is the bridge that carries what
// they say into this page, and the handheld loop a correction hands the arms to.
let robotUp = false;
let teleopUp = false;
// The correction as the studio last said (dagger.py), and an order of it on its way.
let dagger = null;
let corrBusy = false;
let robotFound = { driver: null, gate: null };
let asking = false;
// What the Connect button last showed, and a Connect or a Disconnect on its
// way: while one is, the button says which and takes no second click, and the
// rig's updates leave it alone.
let connected = false;
let starting = false;
let switching = null;
const PARTS = ['bridge', 'teleop'];
const ROBOT_LINE = 'ros2 launch pinkrab_bringup robot.launch.py';
// A policy is shown the grippers alone; the top camera is open, and drawn
// beside them, only while the hands have the arms (showHead) - a correction
// needs it: it follows the handhelds' solids.
// A line per thing Connect opens, with a Refresh for each this page can open
// again - see connection.js. The robot's own launch has none: it is started in
// a terminal, and looked for here.
const withRig = (asked) => async () => {
  const answer = await asked();
  if (answer?.rig) showRig(answer.rig);
  return answer;
};
const conn = new Connection(document.getElementById('startup'), [
  ['live', 'live view', withRig(() => ask(camsUp ? '/rig/live/restart' : '/rig/live/start', { cameras: 'follower' }))],
  ['head', 'top camera - for corrections', withRig(() => ask('/rig/live/head/retry', {}))],
  ['left', 'left gripper camera', withRig(() => ask('/rig/live/gripper/left/retry', {}))],
  ['right', 'right gripper camera', withRig(() => ask('/rig/live/gripper/right/retry', {}))],
  ['robot', 'robot - driver and gate'],
  ['bridge', 'bridge', withRig(() => ask('/mode', { real: true, parts: ['bridge'] }))],
  ['teleop', 'handheld follower', withRig(() => ask('/mode', { real: true, parts: ['teleop'] }))],
  ['pedal', 'pedal', withRig(() => ask('/rig/footswitch/start', {}))],
], {
  storageKey: 'policy.connection.folded',
  shown: () => connected,
  // Nothing while a Connect or a Disconnect is on its way: a Refresh then
  // would be a second start or stop on top of it.
  // The top camera opens for a correction and closes after it: opened again
  // only while the hands have the arms.
  canRefresh: (key) => !switching
    && (key === 'live' || key === 'pedal' || (['bridge', 'teleop'].includes(key) ? connected
      : key === 'head' ? camsUp && !!dagger?.hand_has : camsUp)),
});
const WORDS = { pending: 'waiting', starting: 'starting ...', running: 'running', already: 'already running',
                failed: 'did not start - see the terminal', exited: 'exited - see the terminal' };
// Which camera keys each checkpoint was trained on, and the arms it drives, by path.
let wantsByPath = {};
let handsByPath = {};
// The commands each checkpoint that reads one was trained on, by path.
let commandsByPath = {};
// The live view's last word on its cameras: per gripper side, and the head.
let liveStats = null;
// The view's numbers as last sent, whether or not the cameras were up then.
let lastStats = null;
// Every step the page has been told about, oldest first. The server keeps its
// own window; this is what the traces are drawn from.
let steps = [];

// A new trace: the server sends steps from the first one again.
function startTrace() {
  steps = [];
  say({ since: 0 });
}

async function ask(path, body) {
  const r = await fetch(path, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {}),
  });
  return r.json();
}

// --- the cameras ------------------------------------------------------------

// The rig as last drawn, to draw again when the robot's launch is found.
let lastRig = null;

function showRig(rig) {
  if (!rig?.live) return;
  lastRig = rig;
  if (rig.live_url) liveUrl = rig.live_url;
  // A live view on the handhelds' cameras - the collect page's - is not this
  // page's: Connect starts it again on the arms'.
  const theirs = rig.live.running && rig.live.options?.cameras !== 'follower';
  camsUp = rig.live.running && !theirs;
  const parts = rig.robot?.parts || {};
  const up = (k) => !!rig.robot?.asked && ['running', 'already'].includes(parts[k]?.step);
  const bridgeUp = up('bridge');
  robotUp = bridgeUp && !!robotFound.driver && !!robotFound.gate;
  // Only a correction needs the handheld loop: Ask and Drive go without it.
  teleopUp = up('teleop');
  if (dagger) showDagger(dagger);
  connected = camsUp || !!rig.robot?.asked;
  starting = !!rig.robot?.starting;
  const bad = (!camsUp && rig.live.exit) || Object.values(parts).some((p) => ['failed', 'exited'].includes(p.step));
  dotEl.className = 'dot' + (camsUp && robotUp ? ' on' : bad ? ' bad' : '');
  lineEl.textContent = rig.robot?.starting ? 'connecting ...'
    : camsUp && robotUp ? 'cameras, robot and bridge up'
    : camsUp && bridgeUp ? `cameras and bridge up - the robot is not: ${ROBOT_LINE}`
    : camsUp ? (rig.robot?.asked ? 'cameras up, bridge incomplete' : 'cameras up, bridge off')
    : rig.robot?.asked ? (theirs ? "bridge up - the live view has the handhelds' cameras" : 'bridge up, cameras off')
    : theirs ? "the live view has the handhelds' cameras - Connect opens the arms'"
    : rig.live.note || 'not connected';
  drawConnect();
  feedStartup(rig, null);

  // The cameras the chosen checkpoint is shown are marked.
  const hands = handsByPath[checkpointEl.value] || [];
  for (const [side, fig] of Object.entries(cams)) {
    const img = fig.querySelector('img');
    sourceSelect(sources[side], side, () => hideCam(img), 'follower');
    fig.classList.toggle('fed', hands.includes(side));
    // Up, the pictures are asked for once the view's numbers come (showCameras).
    if (!camsUp && img.getAttribute('src')) hideCam(img);
  }
  // The view's numbers may have come before the rig said the cameras were
  // up; they come again only when they change. Down, they were the last
  // view's, and a picture asked of them would be asked of a server not yet
  // listening.
  if (!camsUp) lastStats = null;
  showCameras(lastStats);
}

function drawConnect() {
  const busy = switching || (starting ? 'Connecting ...' : null);
  connectEl.textContent = busy || (connected ? 'Disconnect' : 'Connect');
  connectEl.className = busy ? 'busy' : connected ? 'stop' : 'primary';
  connectEl.disabled = !!busy;
  // The poses walk the arms: once there is a rig to walk them through.
  posesEl.hidden = !connected;
  // The list's Refreshes are off while a Connect or a Disconnect is on its way.
  conn.draw();
}

// --- what the rig is missing, before Connect ----------------------------------
// The port assignment, a calibration per arm, the description: files the
// driver needs and this page can only ask for. Listed with the command that
// writes each, and asked again from the dialog once they have been run.
const dialog = document.getElementById('bringup');
const needsWhat = document.getElementById('bringup-what');
const needsList = document.getElementById('bringup-list');

async function rigNeeds() {
  try { return (await (await fetch('/rig/needs', { cache: 'no-store' })).json()).needs || []; }
  catch { return []; }
}

function showNeeds(needs, again) {
  needsWhat.textContent = `${again ? 'Still missing: ' : 'Missing on this machine: '}`
    + needs.map((n) => n.what).join(', ') + '.';
  needsList.replaceChildren(...needs.map((n) => {
    const li = document.createElement('li');
    const what = document.createElement('b'); what.textContent = n.what;
    const why = document.createElement('span'); why.className = 'why'; why.textContent = ` - ${n.why}`;
    const pre = document.createElement('pre'); const code = document.createElement('code');
    code.textContent = n.run; pre.append(code);
    li.append(what, why, pre);
    return li;
  }));
}

document.getElementById('bringup-cancel').addEventListener('click', () => {
  dialog.close();
  switching = null;
  drawConnect();
});
// Esc closes the dialog without the Cancel button.
dialog.addEventListener('cancel', () => { switching = null; drawConnect(); });
document.getElementById('bringup-retry').addEventListener('click', async () => {
  const needs = await rigNeeds();
  if (!needs.length) { dialog.close(); return connect(); }
  showNeeds(needs, true);
});
document.getElementById('bringup-anyway').addEventListener('click', () => { dialog.close(); connect(); });

connectEl.addEventListener('click', async () => {
  if (switching) return;
  noteEl.textContent = '';
  if (connected) return disconnect();
  switching = 'Connecting ...';
  drawConnect();
  const needs = await rigNeeds();
  if (needs.length) { showNeeds(needs, false); dialog.showModal(); return; }
  return connect();
});

async function disconnect() {
  switching = 'Disconnecting ...';
  drawConnect();
  let rig = null;
  try {
    conn.end();
    // A policy holding the arm is halted first - idle at the gate, its node
    // gone - while the bridge is still there to carry the idle; the server
    // refuses the simulation under a policy. So is a hand correcting one,
    // which keeps what it recorded.
    if (policyState?.running || policyState?.driving || dagger?.hand_has) {
      await ask('/policy/halt', {});
      await new Promise((r) => setTimeout(r, 600));
    }
    // The robot first: on the way down the driver walks any arm holding
    // torque home, and that should be over before anything else goes.
    const answer = await ask('/mode', { real: false });
    if (answer.error) throw new Error(answer.error);
    if (camsUp) rig = (await ask('/rig/live/stop', {})).rig;
  } catch (err) {
    noteEl.textContent = String(err);
  }
  switching = null;
  if (rig) showRig(rig); else drawConnect();
}

async function connect() {
  switching = 'Connecting ...';
  drawConnect();
  try {
    conn.begin();
    conn.set('live', 'starting', 'starting ...');
    conn.draw();
    if (!camsUp) lastStats = null;   // the view about to start is not the one they came from
    const live = camsUp ? { rig: null } : await ask('/rig/live/start', { cameras: 'follower' });
    if (live.error) { noteEl.textContent = live.error; conn.set('live', 'failed', live.error); }
    if (live.rig) showRig(live.rig);
    // The bridge. The robot's own launch is looked for, not started: its
    // terminal is where its output belongs, and its speed limit is set there.
    for (const k of PARTS) conn.set(k, 'starting', 'starting ...');
    conn.draw();
    const answer = await ask('/mode', { real: true, parts: PARTS });
    if (answer.error) noteEl.textContent = answer.error;
    if (answer.rig) showRig(answer.rig);
  } catch (err) {
    noteEl.textContent = String(err);
  }
  switching = null;
  drawConnect();
}

// The corner panel, fed from whatever the page just learnt: the rig (the
// robot's launch, the bridge, the pedal, the live view's process) and the
// live view's stats (each camera).
function feedStartup(rig, stats) {
  if (rig) {
    const live = rig.live;
    if (live?.running && live.options?.cameras === 'follower') conn.set('live', 'running', 'answering');
    else if (live?.exit) conn.set('live', 'failed', live.note || 'did not start');
    for (const k of PARTS) {
      const step = rig.robot?.parts?.[k]?.step;
      if (step && step !== 'pending') conn.set(k, step, WORDS[step] || step);
    }
    if (robotFound.driver && robotFound.gate) conn.set('robot', 'already', `driver pid ${robotFound.driver}, gate pid ${robotFound.gate}`);
    else conn.set('robot', 'failed', `not running here - in a terminal: ${ROBOT_LINE}`);
    const pedal = rig.footswitch;
    if (pedal) conn.set('pedal', pedal.running ? 'running' : pedal.exit ? 'failed' : 'starting',
                           pedal.running ? 'reading' : pedal.exit ? 'did not start' : 'starting ...');
  }
  if (stats) {
    conn.set('live', 'running', 'answering');
    if (stats.head === false && stats.head_off) conn.set('head', 'off', 'off');
    else if (stats.head === false) conn.set('head', stats.head_opening ? 'starting' : 'failed',
      stats.head_opening ? 'opening ...' : (stats.head_reason || 'not connected').split(':').pop().trim());
    else conn.set('head', stats.fps == null ? 'starting' : 'running', stats.fps == null ? 'opening ...' : `${Math.round(stats.fps)} fps`);
    for (const side of ['left', 'right']) {
      const g = stats.gripper?.[side];
      // No word yet is a camera still opening; a word is why it did not.
      if (g?.running) conn.set(side, 'running', `${g.fps ?? '—'} fps`);
      else if (g?.note) conn.set(side, 'failed', g.note);
      else conn.set(side, 'starting', 'opening ...');
    }
  }
  conn.draw();
}

checkpointEl.addEventListener('change', () => { showWants(); showCommand(); loadSettings(); showTally(); if (lastRig) showRig(lastRig); });

// The per-camera line on each tile, and whether the checkpoint can be fed.
// The view's numbers come once its server answers, which is also when its
// pictures can be asked for.
function showCameras(s) {
  liveStats = s;
  if (s) feedStartup(null, s);
  for (const [side, fig] of Object.entries(cams)) {
    const img = fig.querySelector('img');
    if (s && liveUrl && !img.getAttribute('src')) showStream(img, liveUrl + `gripper/${side}`);
    const g = s?.gripper?.[side];
    const stat = fig.querySelector('.stat');
    if (g?.running) stat.textContent = `${g.fps ?? '—'} fps` + (g.lens ? ` · ${g.lens}` : '');
    else stat.textContent = camsUp ? (g?.note || 'not found') : '';
  }
  showHead();
  showWants();
}

// The top camera, while the hands have the arms: it is what follows the
// handhelds' solids, so a correction is watched through it - its caption says
// which solids it has now. Streamed only then.
function showHead() {
  const on = !!dagger?.hand_has && camsUp && !!liveUrl;
  document.querySelector('.cam-grid').classList.toggle('with-head', on);
  headFig.hidden = !on;
  const img = headFig.querySelector('img');
  if (!on) {
    if (img.getAttribute('src')) hideCam(img);
    return;
  }
  const s = lastStats;
  const stat = headFig.querySelector('.stat');
  // Asked for once the view's numbers come - which is when its pictures can
  // be, as for the gripper cameras.
  if (!s) { stat.textContent = 'opening ...'; return; }
  if (!img.getAttribute('src')) showStream(img, liveUrl + 'stream');
  if (s.head === false) { stat.textContent = (s.head_reason || 'not connected').split(':').pop().trim(); return; }
  // An arm whose solid is not in view holds where it is.
  const solids = (dagger.arms || []).map((arm) => `${arm} ${s.solids?.[`handheld_${arm}`]?.solved ? '✓' : '✗ holds'}`);
  stat.textContent = `solids: ${solids.join(' · ')} · ${s.fps != null ? Math.round(s.fps) : '—'} fps`;
}

// The arm whose gripper camera a picture the checkpoint wants comes from:
// observation.images.gripper_<arm>.
const sideOf = (key) => key.split('.').pop().replace('gripper_', '');

// What the chosen checkpoint wants to see, and which of those is answering.
// A missing one is why the run button stays off.
function fed() {
  const wants = wantsByPath[checkpointEl.value] || [];
  const answers = Object.fromEntries(wants.map((key) => [key, !!liveStats?.gripper?.[sideOf(key)]?.running]));
  return { wants, answers, all: wants.length > 0 && wants.every((k) => answers[k]) };
}

function showWants() {
  const { wants, answers } = fed();
  if (!checkpointEl.value || !wants.length) { wantsEl.replaceChildren(); return; }
  const parts = wants.map((key) => {
    const name = `${sideOf(key)} gripper camera`;
    const el = document.createElement('span');
    el.className = answers[key] ? '' : 'missing';
    el.textContent = `${name} ${answers[key] ? '✓' : (camsUp ? '✗ not answering' : '— cameras off')}`;
    return el;
  });
  const lead = document.createElement('b');
  lead.textContent = 'this checkpoint looks at: ';
  wantsEl.replaceChildren(lead, ...parts.flatMap((p, i) => (i ? [document.createTextNode('  ·  '), p] : [p])));
  updateButtons();
}

// What may be pressed: Ask and Drive want the cameras and the robot's side up
// and the checkpoint fed; Stop wants something to stop. Loading is neither.
let policyState = { running: false, driving: false, loading: false };
function updateButtons() {
  const idle = !policyState.running && !policyState.driving && !policyState.loading;
  const may = idle && camsUp && robotUp && fed().all && !commandMissing();
  askEl.disabled = !may;
  driveEl.disabled = !may;
  // The poses want the robot's side, not the checkpoint: nothing is asked.
  readyEl.disabled = !(idle && robotUp);
  homeEl.disabled = !(idle && robotUp);
  haltEl.disabled = idle;
  checkpointEl.disabled = !idle;
  commandEl.disabled = !idle;
  const handsHave = !!dagger?.hand_has;
  settingsOpen.disabled = !idle || handsHave;
  if (settingsOpen.disabled && drawer.open) drawer.close();
  // The order under way stays lit and says what it is doing; idle, none is.
  const doing = policyState.driving ? driveEl : !idle ? askEl : null;
  for (const [el, word, busy] of [[askEl, 'Ask', policyState.loading ? 'Loading' : 'Asking'],
                                  [driveEl, 'Drive', 'Driving']]) {
    el.classList.toggle('active', el === doing);
    el.setAttribute('aria-pressed', String(el === doing));
    el.querySelector('b').textContent = el === doing ? busy : word;
  }
}

// --- the policy -------------------------------------------------------------

async function loadCheckpoints() {
  const { checkpoints } = await (await fetch('/policy/checkpoints', { cache: 'no-store' })).json();
  if (!checkpoints.length) {
    checkpointEl.replaceChildren(new Option('no checkpoint under ~/pinkrab/train', ''));
    checkpointState.textContent = 'train one on the Train page';
    updateButtons();
    return;
  }
  checkpointState.textContent = `${checkpoints.length} on this machine`;
  wantsByPath = Object.fromEntries(checkpoints.map((c) => [c.path, c.images || []]));
  handsByPath = Object.fromEntries(checkpoints.map((c) => [c.path, c.hands || []]));
  commandsByPath = Object.fromEntries(checkpoints.filter((c) => c.command).map((c) => [c.path, c.tasks || []]));
  checkpointEl.replaceChildren(...checkpoints.map((c) => {
    const when = new Date(c.at * 1000).toLocaleString();
    return new Option(`${c.run} · step ${c.step} · ${when}`, c.path);
  }));
  showWants();
  showCommand();
  loadSettings();
  if (lastRig) showRig(lastRig);
}

// The command, for a checkpoint that reads one: its dataset's offered, and
// filled in when it had only one. Asked and driven with what the field holds;
// left empty, the checkpoint's own - which one that is it cannot say when its
// dataset held several, so then a command must be typed.
function showCommand() {
  const tasks = commandsByPath[checkpointEl.value];
  commandField.hidden = !tasks;
  commandTasks.replaceChildren(...(tasks || []).map((t) => new Option(t)));
  commandEl.value = tasks && tasks.length === 1 ? tasks[0] : '';
  commandEl.placeholder = !tasks ? '' : tasks.length > 1 ? `one of its dataset's ${tasks.length}, or another` : 'what to do';
  updateButtons();
}
function command() { return commandField.hidden ? '' : commandEl.value.trim(); }
function commandMissing() {
  const tasks = commandsByPath[checkpointEl.value];
  return !!tasks && tasks.length !== 1 && !command();
}
commandEl.addEventListener('input', () => updateButtons());

// --- how the checkpoint is asked ---------------------------------------------
// Its policy type's settings in the rig's inference.yaml (pinkrab_core.inference),
// beside what it was trained with, and under them the ones every policy
// shares. Saved, they are what the next Ask or Drive runs with - the gate's
// speed when the robot's launch next starts it; a run keeps what it started
// with, said under the tables.
const settingsEl = document.getElementById('settings');
const settingsTitle = document.getElementById('settings-title');
const settingsFile = document.getElementById('settings-file');
const kindTable = document.getElementById('settings-kind');
const commonBody = document.querySelector('#settings-common tbody');
const settingsShown = document.getElementById('settings-trained');
const settingsSave = document.getElementById('settings-save');
const settingsNote = document.getElementById('settings-note');
const settingsInUse = document.getElementById('settings-inuse');
// name -> { set, value }, per section: set ones are saved, the others left to the default.
let kindForm = {};
let commonForm = {};

const shownValue = (v) => (v === null || v === undefined ? 'off' : String(v));

settingsOpen.addEventListener('click', () => drawer.showModal());
document.getElementById('settings-close').addEventListener('click', () => drawer.close());

async function loadSettings() {
  const path = checkpointEl.value;
  settingsNote.textContent = '';
  if (!path) { settingsEl.hidden = true; return; }
  try {
    const answer = await (await fetch(`/policy/settings?checkpoint=${encodeURIComponent(path)}`, { cache: 'no-store' })).json();
    if (answer.error) { settingsEl.hidden = false; settingsNote.textContent = answer.error; return; }
    drawSettings(answer);
  } catch (err) { settingsNote.textContent = String(err); }
}

function drawSettings(d) {
  settingsEl.hidden = false;
  settingsFile.textContent = d.file;
  kindForm = Object.fromEntries(d.knobs.map((k) => [k.name, { set: k.saved, value: k.using }]));
  commonForm = Object.fromEntries(d.common.map((k) => [k.name, { set: k.saved, value: k.using }]));
  kindTable.hidden = !d.known;
  kindTable.querySelector('tbody').replaceChildren(...d.knobs.map((k) => knobRow(k, kindForm, k.trained)));
  commonBody.replaceChildren(...d.common.map((k) => knobRow(k, commonForm, k.default)));
  const fallback = Object.entries(d.using).map(([k, v]) => `${k} ${shownValue(v)}`).join('  ·  ');
  settingsTitle.textContent = d.known ? `${d.label} - how it is asked`
    : `${d.type} - no settings known, it runs as trained${fallback ? ` but for ${fallback}` : ''}`;
  const shown = Object.entries(d.shown).map(([k, v]) => `${k} ${shownValue(v)}`).join('  ·  ');
  settingsShown.textContent = shown ? `trained with: ${shown}` : '';
  if (d.problems.length) settingsNote.textContent = `the saved settings do not fit this checkpoint: ${d.problems.join('; ')}`;
  settingsSave.disabled = true;
}

// One row: the setting - what it does behind its ?, on hover or focus - its
// value here, what it is beside (trained, or Pinkrab's default), and a way
// back to the default.
function knobRow(k, form, beside) {
  const tr = document.createElement('tr');
  const name = document.createElement('td');
  name.className = 'name';
  const label = document.createElement('b');
  label.textContent = k.name;
  const help = document.createElement('span');
  help.className = 'help';
  help.tabIndex = 0;
  help.textContent = '?';
  help.setAttribute('aria-label', `what ${k.name} does`);
  const tip = document.createElement('span');
  tip.className = 'tip';
  tip.id = `tip-${form === commonForm ? 'common' : 'kind'}-${k.name}`;
  tip.setAttribute('role', 'tooltip');
  tip.textContent = k.help;
  help.setAttribute('aria-describedby', tip.id);
  help.append(tip);
  name.append(label, help);
  const cell = document.createElement('td');
  cell.className = 'value';
  let input;
  if (k.kind === 'choice') {
    input = document.createElement('select');
    input.append(...k.choices.map((c) => new Option(c, c)));
  } else if (k.kind === 'bool') {
    input = document.createElement('input');
    input.type = 'checkbox';
  } else {
    input = document.createElement('input');
    input.type = 'number';
    input.step = k.kind === 'int' ? '1' : 'any';
    if (k.minimum !== null) input.min = String(k.minimum);
    if (k.maximum !== null) input.max = String(k.maximum);
  }
  // An optional one is a number or nothing: off, or left to the policy.
  const off = document.createElement('label');
  off.className = 'off';
  off.innerHTML = '<input type="checkbox"> off';
  const offBox = off.querySelector('input');
  off.hidden = !k.optional;
  const reset = document.createElement('button');
  reset.className = 'reset';
  reset.title = 'leave it to the default';
  const show = () => {
    const f = form[k.name];
    offBox.checked = f.value === null;
    input.disabled = f.value === null;
    if (k.kind === 'bool') input.checked = !!f.value;
    else input.value = f.value === null ? '' : String(f.value);
    reset.textContent = f.set ? 'default' : 'default ✓';
    reset.disabled = !f.set;
    tr.classList.toggle('set', f.set);
  };
  const changed = () => {
    let value;
    if (k.optional && offBox.checked) value = null;
    else if (k.kind === 'choice') value = input.value;
    else if (k.kind === 'bool') value = input.checked;
    else value = k.kind === 'int' ? Number.parseInt(input.value, 10) : Number.parseFloat(input.value);
    form[k.name] = { set: true, value: Number.isNaN(value) ? input.value : value };
    settingsSave.disabled = false;
    settingsNote.textContent = '';
    show();
  };
  input.addEventListener('change', changed);
  offBox.addEventListener('change', () => {
    // Back on, it starts from what it was trained with, or the knob's floor.
    if (!offBox.checked) input.value = String(k.trained ?? k.default ?? Math.max(k.minimum ?? 0, 0));
    changed();
  });
  reset.addEventListener('click', () => {
    const top = k.maximum ?? Infinity;
    const fallback = typeof k.default === 'number' ? Math.min(k.default, top) : k.default;
    form[k.name] = { set: false, value: k.default !== null ? fallback : k.trained };
    settingsSave.disabled = false;
    show();
  });
  cell.append(input, off);
  const ref = document.createElement('td');
  ref.className = 'trained';
  ref.textContent = shownValue(beside);
  const tail = document.createElement('td');
  tail.append(reset);
  tr.append(name, cell, ref, tail);
  show();
  return tr;
}

const setOnes = (form) => Object.fromEntries(Object.entries(form).filter(([, f]) => f.set).map(([n, f]) => [n, f.value]));

settingsSave.addEventListener('click', async () => {
  settingsSave.disabled = true;
  try {
    const answer = await ask('/policy/settings', { checkpoint: checkpointEl.value,
                                                   values: setOnes(kindForm), common: setOnes(commonForm) });
    if (answer.error) { settingsNote.textContent = answer.error; settingsSave.disabled = false; return; }
    drawSettings(answer);
    settingsNote.textContent = 'saved - the next Ask or Drive runs with these, the gate its speed when the robot next starts';
  } catch (err) { settingsNote.textContent = String(err); settingsSave.disabled = false; }
});

// What the run going now was started with, which a save does not change.
function showSettingsInUse(s) {
  const on = (s.running || s.driving) && s.settings;
  settingsInUse.textContent = on
    ? `this run: ${Object.entries(s.settings).map(([k, v]) => `${k} ${shownValue(v)}`).join('  ·  ') || 'as trained'}`
    : '';
}

// --- the three orders --------------------------------------------------------

// A pose is a place, not a joint vector: the arms are there when every joint
// it names is within the tolerance of it - the one the rollout node uses
// before it lets the policy take over, as the Ask answer says it.
function nearReady(angles, ready, tolDeg) {
  const names = Object.keys(ready).filter((n) => !n.endsWith('_gripper'));
  if (!names.length) return false;
  const tol = (tolDeg * Math.PI) / 180;
  return names.every((n) => n in angles && Math.abs(angles[n] - ready[n]) <= tol);
}

// The arms to the ready pose or home, through the gate, nothing else driving.
// What follows is drawn from the readings; the button only orders the walk.
async function poseButton(which) {
  noteEl.textContent = '';
  statusEl.textContent = `walking to the ${which} pose ...`;
  const answer = await ask('/policy/goto', { pose: which });
  if (answer.error) { noteEl.textContent = answer.error; statusEl.textContent = ''; }
  updateButtons();
}
readyEl.addEventListener('click', () => poseButton('ready'));
homeEl.addEventListener('click', () => poseButton('home'));

// Walk to ready and on to the start, watch the readings until the arms are
// there, then ask.
askEl.addEventListener('click', async () => {
  askEl.disabled = true; driveEl.disabled = true;
  noteEl.textContent = '';
  startTrace();
  try {
    const answer = await ask('/policy/ask', { checkpoint: checkpointEl.value });
    if (answer.error) { noteEl.textContent = answer.error; updateButtons(); return; }
    const start = answer.ready || {};
    if (answer.note) noteEl.textContent = answer.note;
    statusEl.textContent = 'walking to the start ...';
    // The readings as they come, until the arms are there or 45 s have gone.
    const there = await until('servo', (st) => {
      statusEl.textContent = `walking to the start ...${st?.fresh ? '' : ' (no readings yet)'}`;
      return !!st?.fresh && nearReady(st.angles_rad || {}, start, answer.ready_tol_deg);
    }, 45000, { fresh: true });
    if (!there) { noteEl.textContent = 'the arms did not reach the start - are they armed, and is the driver holding them?'; statusEl.textContent = ''; updateButtons(); return; }
    statusEl.textContent = 'at the start - asking';
    const started = await ask('/policy/start', { checkpoint: checkpointEl.value, task: command() });
    if (started.error) { noteEl.textContent = started.error; statusEl.textContent = ''; }
  } catch (err) {
    noteEl.textContent = String(err);
  }
  updateButtons();
});

// Hand the arms to the rollout node. It walks them to ready and the start
// itself, then drives; what it asked for arrives as steps and is drawn like
// the asking loop's.
driveEl.addEventListener('click', async () => {
  askEl.disabled = true; driveEl.disabled = true;
  noteEl.textContent = '';
  startTrace();
  const answer = await ask('/policy/drive', { checkpoint: checkpointEl.value, task: command() });
  if (answer.error) noteEl.textContent = answer.error;
  updateButtons();
});

// Idle at the gate: nothing more passes, the servos hold where they are.
haltEl.addEventListener('click', async () => {
  haltEl.disabled = true;
  const answer = await ask('/policy/halt', {});
  if (answer.error) noteEl.textContent = answer.error;
  statusEl.textContent = 'stopped - the arms hold where they are';
  updateButtons();
});

function showPolicy(s) {
  asking = s.running;
  policyState = s;
  showSettingsInUse(s);
  if (s.driving) {
    statusEl.textContent = s.phase === 'walking' ? `driving: walking to the start, ${s.to_go_deg ?? '?'} deg to go`
      : s.phase === 'driving' ? 'driving: the policy has the arms'
      : s.phase === 'refused' ? `not driving - ${s.refused}`
      : 'driving: the rollout node is starting ...';
  } else if (s.loading) {
    statusEl.textContent = 'loading the checkpoint ...';
  } else if (s.running) {
    statusEl.textContent = 'at the start - asking; nothing moves';
  }
  updateButtons();
  if (s.note) noteEl.textContent = s.note;

  const last = s.last;
  if (!last) {
    for (const arm of ARMS) metric(cards[arm], '—', 'mm per tick', '');
    metric(cards.rate, '—', 'steps, Hz, latency', '');
    return;
  }
  // Each arm: how far its tool steps and turns, along x y z of its own frame,
  // and its jaw - an arm the checkpoint does not drive is a dash.
  for (const arm of ARMS) {
    const a = last.arms?.[arm];
    if (!a) { metric(cards[arm], '—', 'not driven', ''); continue; }
    metric(cards[arm], `${a.step_mm.toFixed(1)} mm · ${a.step_deg.toFixed(1)}°`,
           a.action.slice(0, 3).map((v) => (v * 1000).toFixed(1)).join('  ')
           + `  (x y z) · gripper ${a.action[9].toFixed(2)}`, a.wild ? 'bad' : 'ok');
  }
  metric(cards.rate, `${s.steps}`, `${s.hz.toFixed(1)}${s.rate ? ` of ${s.rate}` : ''} Hz · ${s.latency_ms.toFixed(0)} ms · ${s.wild} refused`,
         s.wild > 0 ? 'bad' : '');
}

// --- the traces -------------------------------------------------------------

const traceEls = [...document.querySelectorAll('canvas.trace')];
const pathEl = document.getElementById('path');
const COLOURS = { dx: '#e2488e', dy: '#1668c9', dz: '#17a673', rot: '#f5b301', grip: '#7a4fd6' };
// The two arms' paths: a colour and a line each.
const PATHS = { left: { colour: '#e2488e', dash: [] }, right: { colour: '#1668c9', dash: [6, 4] } };

function series(arm) {
  // Millimetres and degrees, so the traces are read in the units the gate
  // refuses on rather than in metres nobody can see.
  const of = steps.map((s) => s.arms?.[arm]).filter(Boolean);
  return {
    dx: of.map((a) => a.action[0] * 1000),
    dy: of.map((a) => a.action[1] * 1000),
    dz: of.map((a) => a.action[2] * 1000),
    rot: of.map((a) => a.step_deg),
    grip: of.map((a) => a.action[9] * 10),
  };
}

function drawTraces() {
  document.getElementById('traces').hidden = steps.length < 2;
  // One scale for both arms, so a bigger step on one canvas is a bigger step.
  const lines = Object.fromEntries(ARMS.map((arm) => [arm, series(arm)]));
  const all = ARMS.flatMap((arm) => Object.values(lines[arm]).flat());
  const span = Math.max(Math.abs(Math.min(...all, 0)), Math.abs(Math.max(...all, 0)), 1);
  for (const el of traceEls) drawTrace(el, el.dataset.arm, lines[el.dataset.arm], span);
}

function drawTrace(traceEl, arm, lines, span) {
  const ctx = traceEl.getContext('2d');
  const { width: w, height: h } = traceEl;
  ctx.clearRect(0, 0, w, h);
  if (lines.dx.length < 2) {
    ctx.fillStyle = '#8f7283';
    ctx.font = '14px ui-sans-serif, system-ui, sans-serif';
    ctx.fillText(steps.length < 2 ? 'nothing asked yet' : `nothing asked of the ${arm} arm`, 14, 26);
    return;
  }
  const mid = h / 2;
  const y = (v) => mid - (v / span) * (mid - 12);
  const x = (i) => (i / Math.max(steps.length - 1, 1)) * (w - 1);

  ctx.strokeStyle = '#ecd9e2';
  ctx.lineWidth = 1;
  for (const v of [-span, 0, span]) {
    ctx.beginPath(); ctx.moveTo(0, y(v)); ctx.lineTo(w, y(v)); ctx.stroke();
  }
  ctx.fillStyle = '#8f7283';
  ctx.font = '11px ui-sans-serif, system-ui, sans-serif';
  ctx.fillText(`${span.toFixed(1)}`, 4, y(span) + 12);
  ctx.fillText(`-${span.toFixed(1)}`, 4, y(-span) - 4);

  // A tick the gate would have refused, behind the traces.
  ctx.fillStyle = 'rgba(184, 48, 111, 0.18)';
  steps.forEach((s, i) => { if (s.arms?.[arm]?.wild) ctx.fillRect(x(i) - 1, 0, 2, h); });

  ctx.lineWidth = 1.5;
  for (const [name, values] of Object.entries(lines)) {
    ctx.strokeStyle = COLOURS[name];
    ctx.beginPath();
    values.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))));
    ctx.stroke();
  }
}

function drawPath() {
  const ctx = pathEl.getContext('2d');
  const { width: w, height: h } = pathEl;
  ctx.clearRect(0, 0, w, h);
  if (steps.length < 2) return;
  const wheres = Object.fromEntries(ARMS.map((arm) => [arm, steps.map((s) => s.arms?.[arm]?.where).filter(Boolean)]));
  const all = Object.values(wheres).flat();
  // One scale for both axes and both arms, so a path is a shape and not a stretched one.
  const half = Math.max(...all.map((p) => Math.max(Math.abs(p[0]), Math.abs(p[1]))), 0.01) * 1.15;
  const px = (v) => w / 2 + (v / half) * (w / 2 - 14);
  const py = (v) => h / 2 - (v / half) * (h / 2 - 14);

  ctx.strokeStyle = '#ecd9e2';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(px(0), 0); ctx.lineTo(px(0), h); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, py(0)); ctx.lineTo(w, py(0)); ctx.stroke();

  ctx.lineWidth = 1.5;
  for (const [arm, path] of Object.entries(wheres)) {
    if (path.length < 2) continue;
    ctx.strokeStyle = PATHS[arm].colour;
    ctx.setLineDash(PATHS[arm].dash);
    ctx.beginPath();
    path.forEach((p, i) => (i ? ctx.lineTo(px(p[0]), py(p[1])) : ctx.moveTo(px(p[0]), py(p[1]))));
    ctx.stroke();
    // Where it is now; each started at the middle.
    ctx.setLineDash([]);
    ctx.fillStyle = PATHS[arm].colour;
    ctx.beginPath(); ctx.arc(px(path.at(-1)[0]), py(path.at(-1)[1]), 4, 0, Math.PI * 2); ctx.fill();
  }
  ctx.setLineDash([]);
  ctx.fillStyle = '#8f7283';
  ctx.beginPath(); ctx.arc(px(0), py(0), 3, 0, Math.PI * 2); ctx.fill();

  document.getElementById('path-hint').textContent =
    `${(half * 2 * 1000).toFixed(0)} mm across · ${steps.length} steps · each from where its tool started`;
}

// --- what the studio says, as it changes ---------------------------------------

watch('policy', showPolicy);
watch('policy_history', ({ steps: fresh, reset }) => {
  // Drawn afresh on a new socket, a trace started over here, or a run
  // started elsewhere that numbers its steps from one again.
  steps = (reset ? [] : steps).concat(fresh).slice(-600);
  drawTraces();
  drawPath();
});
// The robot's launch, looked for in the process table; the rig the bridge and
// the live view; the view's numbers while it shows the arms' cameras.
watch('robot', (found) => { robotFound = found || robotFound; if (lastRig) showRig(lastRig); });
watch('rig', showRig);
watch('stats', (stats) => {
  // Numbers from a view on the handhelds' cameras - going down as this
  // page's comes up - are not this page's.
  lastStats = stats && Object.keys(stats).length && (stats.cameras ?? 'follower') === 'follower' ? stats : null;
  showCameras(camsUp ? lastStats : null);
  // The run's time recorded and the hand's opening are the view's.
  if (dagger) showDagger(dagger);
});

loadCheckpoints().catch((err) => { noteEl.textContent = String(err); });
drawTraces();

// --- the correction (DAgger) ----------------------------------------------------
// Every run is recorded whole, Drive to stop, as a rollout episode; while it
// drives, the handhelds can take its arms over and put them right, and the
// stretches the hands drove are the corrections convert learns from. A
// correction gone wrong is redone: thrown away, the arms walked back to where
// they were taken over, and taken over again. The pedals do it while this
// page is in front - see dagger.py - and these are the same three.

const corr = {
  dot: document.getElementById('corr-dot'),
  line: document.getElementById('corr-line'),
  follow: document.getElementById('corr-follow'),
  toggle: document.getElementById('corr-toggle'),
  stop: document.getElementById('corr-stop'),
  jaws: Object.fromEntries([...document.querySelectorAll('.correct .jaws')].map((el) => [el.dataset.arm, el])),
  note: document.getElementById('corr-note'),
  count: document.getElementById('corr-count'),
  list: document.getElementById('corr-list'),
};
let shownSaved = null;
// The shelf's rollouts by name, as the episodes topic lists them: each one's
// outcome and checkpoint, which the rows and the tally are drawn from.
let rollouts = {};
function showDagger(d) {
  dagger = d;
  if (!d) return;
  updateButtons();
  // The run's time recorded and the hand's share of it, as the live view counts them.
  const episode = lastStats?.episode?.active ? lastStats.episode : null;
  const recorded = episode
    ? `${(episode.seconds ?? 0).toFixed(1)} s, corrected ${(episode.controlled?.human ?? 0).toFixed(1)} s - `
    : d.driving ? 'not recording - ' : '';
  // The hands may have the arms though not taken over from here - a studio
  // started again mid-correction; 2 still hands them back.
  const handHas = d.hand_has;
  // 3 throws away a correction taken over from here, whose pose was kept;
  // a correction thrown away, a redo's walk back or none at all, it stops.
  // Either way the arms stop where they are. 2 after a throw redoes it.
  const throws = d.correcting && !d.thrown && !d.redoing;
  corr.dot.className = 'rec-dot' + (d.recording ? ' on' : '');
  const kept = d.redone ? ` (${d.redone} redone)` : '';
  corr.line.textContent = d.redoing
    ? `${recorded}redoing - the arms walk back to where they were taken over, held there; 3 stops`
    : d.thrown
    ? `${recorded}correction thrown away, the arms hold: put the desk back as it was, then 2 walks them back to where they were taken over; 3 stops the run`
    : handHas
    ? (d.following
      ? `${recorded}correction ${d.corrections}${kept}: the arms follow the hands; 1 holds them, 2 hands them back, ${throws ? '3 throws it away' : '3 stops'}`
      : `${recorded}correction ${d.corrections}${kept}: the arms hold; set each hand as its arm holds its tool, then 1 to follow`)
    : d.driving ? `${recorded}the policy drives the arms; 2 takes them over, 3 stops and keeps the run`
    : 'not driving';
  corr.line.title = corr.line.textContent;
  corr.toggle.querySelector('.word').textContent = d.thrown || d.redoing ? 'redo' : handHas ? 'hand back' : 'take over';
  corr.toggle.classList.toggle('on', handHas);
  // Taking over wants the hands followed: the loop up. The top camera that
  // sees their solids is opened by the take-over itself, while the arms are held.
  corr.toggle.disabled = corrBusy || !(handHas || (d.driving && teleopUp));
  corr.follow.querySelector('.word').textContent = handHas && d.following ? 'hold' : 'follow';
  corr.follow.classList.toggle('on', handHas && d.following);
  corr.follow.disabled = corrBusy || !handHas || d.thrown || d.redoing;
  corr.stop.querySelector('.word').textContent = throws ? 'throw away' : 'stop';
  // The stop stays at hand while another order waits: the server never makes it wait.
  corr.stop.disabled = !(handHas || d.driving);
  const headFailed = lastStats?.head === false && !lastStats.head_opening && !lastStats.head_off;
  if (!teleopUp && d.driving && !handHas) corr.note.textContent = 'the handheld follower is not running - Connect again to correct';
  else if (handHas && headFailed) corr.note.textContent = `the top camera did not open - the arms hold until it does: ${(lastStats.head_reason || '').split(':').pop().trim()}`;
  else if (d.note && !corr.note.textContent) corr.note.textContent = d.note;
  showJaws(d);
  showHead();
  showSaved();
}

// Rows made again only when the list or an outcome on it changes: made afresh
// with every update - the counter while recording, the grip while a policy
// drives - a row took the click from under the pointer.
function showSaved() {
  const saved = dagger?.saved || [];
  const savedKey = JSON.stringify(saved.map((c) => [c.name, c.seconds, rollouts[c.name]?.outcome]));
  if (savedKey === shownSaved) return;
  shownSaved = savedKey;
  corr.list.replaceChildren(...saved.map(savedRow));
}

// A run the hands took over is not the policy's own: it counts as failed,
// whatever was said of it.
function takenOver(c) { return (c.controlled?.human ?? 0) > 0; }

// The chosen checkpoint's runs on the shelf, as the person watching judged
// them: the ones taken over counted as failed, the ones not judged left out.
function showTally() {
  const runs = Object.values(rollouts).filter((e) => e.policy && e.policy === checkpointEl.value);
  const all = Object.values(rollouts).length;
  const shelf = all ? `${all} rollout${all === 1 ? '' : 's'} on the shelf` : '';
  if (!runs.length) { corr.count.textContent = shelf; return; }
  const over = runs.filter(takenOver).length;
  const succeeded = runs.filter((e) => !takenOver(e) && e.outcome === 'succeeded').length;
  const failed = runs.filter((e) => !takenOver(e) && e.outcome === 'failed').length;
  const unjudged = runs.length - over - succeeded - failed;
  corr.count.textContent = `this checkpoint: ${succeeded} of ${succeeded + failed + over} succeeded`
    + (over ? ` (${over} taken over, counted as failed)` : '')
    + (unjudged ? ` · ${unjudged} not judged` : '') + (shelf ? ` · ${shelf}` : '');
}

// Each hand's opening against its arm's grip - what that jaw was last sent:
// drawn while the hands have the arms and the two are apart, which is when the
// arm's jaw still holds what the policy left it at.
function showJaws(d) {
  for (const [arm, el] of Object.entries(corr.jaws)) {
    const hand = lastStats?.jaw?.[arm]?.aperture;
    const grip = d.grips?.[arm];
    const apart = d.hand_has && hand != null && grip != null && Math.abs(hand - grip) > d.jaw_catch;
    el.hidden = !apart;
    if (!apart) continue;
    el.querySelector('.hand').style.left = `${(hand * 100).toFixed(1)}%`;
    el.querySelector('.arm').style.left = `${(grip * 100).toFixed(1)}%`;
    el.querySelector('.words').textContent = `hand ${hand.toFixed(2)} · arm grips at ${grip.toFixed(2)} - `
      + (hand > grip ? 'close' : 'open') + ' the hand to it to take the jaw';
  }
}

// A kept rollout played back: the left arm's camera with the controls, the
// right arm's and the top camera following it - played, paused and sought
// together.
const clip = {
  dialog: document.getElementById('corr-play'),
  title: document.getElementById('corr-play-title'),
  lead: document.getElementById('corr-play-left'),
  others: [document.getElementById('corr-play-right'), document.getElementById('corr-play-head')],
  figures: [document.getElementById('corr-play-right-figure'), document.getElementById('corr-play-head-figure')],
  note: document.getElementById('corr-play-note'),
};
// A kept rollout in a few words: its run's length and the hand's share of it.
function rolloutWords(c) {
  const corrected = c.controlled?.human;
  return `${c.seconds != null ? c.seconds.toFixed(1) : '?'} s`
    + (corrected != null ? `, corrected ${corrected.toFixed(1)} s` : '');
}
function openClip(c) {
  const at = (stream) => `/episodes/${encodeURIComponent(c.name)}/${stream}.mp4`;
  clip.title.textContent = `${c.name} · ${rolloutWords(c)}`;
  clip.note.textContent = '';
  for (const f of clip.figures) f.hidden = false;
  clip.lead.src = at('gripper_left');
  clip.others[0].src = at('gripper_right');
  clip.others[1].src = at('head');
  clip.dialog.showModal();
  clip.lead.play().catch(() => { /* the controls are there */ });
}
function closeClip() {
  for (const v of [clip.lead, ...clip.others]) { v.pause(); v.removeAttribute('src'); v.load(); }
}
clip.lead.addEventListener('play', () => {
  for (const v of clip.others) { v.currentTime = clip.lead.currentTime; v.play().catch(() => {}); }
});
clip.lead.addEventListener('pause', () => { for (const v of clip.others) v.pause(); });
clip.lead.addEventListener('seeked', () => { for (const v of clip.others) v.currentTime = clip.lead.currentTime; });
clip.lead.addEventListener('error', () => { if (clip.lead.getAttribute('src')) clip.note.textContent = 'no video from the left arm\'s camera in this rollout'; });
clip.others.forEach((v, i) => v.addEventListener('error', () => { if (v.getAttribute('src')) clip.figures[i].hidden = true; }));
clip.dialog.addEventListener('close', closeClip);
document.getElementById('corr-play-close').addEventListener('click', () => clip.dialog.close());

function savedRow(c) {
  const li = document.createElement('li');
  const name = document.createElement('button');
  name.className = 'play';
  name.title = 'play this rollout';
  name.textContent = `▶ ${c.name} · ${rolloutWords(c)}`;
  name.addEventListener('click', () => openClip(c));
  const drop = document.createElement('button');
  drop.className = 'drop';
  drop.textContent = '×';
  drop.title = 'delete this rollout';
  drop.addEventListener('click', async () => {
    drop.disabled = true;
    // The row stays when the delete does not go through; it may be tried again.
    try {
      const answer = await ask('/dagger/delete', { name: c.name });
      corr.note.textContent = answer.error || '';
      if (answer.error) drop.disabled = false;
    } catch (err) {
      corr.note.textContent = String(err);
      drop.disabled = false;
    }
  });
  li.append(name, judgeOf(c), drop);
  return li;
}

// Succeeded or failed, said once the run is watched; the one already said,
// pressed again, is taken back. A run taken over is failed already.
function judgeOf(c) {
  if (takenOver(c)) {
    const over = document.createElement('span');
    over.className = 'judged';
    over.textContent = 'taken over';
    return over;
  }
  const outcome = rollouts[c.name]?.outcome ?? null;
  const seg = document.createElement('span');
  seg.className = 'segmented judge';
  for (const [value, word] of [['succeeded', 'succeeded'], ['failed', 'failed']]) {
    const b = document.createElement('button');
    b.textContent = word;
    b.classList.toggle('on', outcome === value);
    // Judged once the run is on the shelf, its meta.json written as it closed.
    b.disabled = !rollouts[c.name];
    b.addEventListener('click', async () => {
      for (const x of seg.children) x.disabled = true;
      try {
        const answer = await ask('/dagger/outcome', { name: c.name, outcome: outcome === value ? null : value });
        corr.note.textContent = answer.error || '';
      } catch (err) {
        corr.note.textContent = String(err);
      }
      // Drawn again from the shelf the topic sends next, outcome and all.
      shownSaved = null;
      showSaved();
    });
    seg.append(b);
  }
  return seg;
}

async function order(path) {
  corrBusy = true;
  corr.note.textContent = '';
  showDagger(dagger);
  try {
    const answer = await ask(path, {});
    corr.note.textContent = answer.error || (answer.handed_back ? 'handed back - the policy drives again'
      : answer.redone ? 'redone - the arms are back where they were taken over, held; 1 to follow'
      : answer.thrown ? 'thrown away - the arms hold; put the desk back, then 2 walks them back' : '');
  } catch (err) {
    corr.note.textContent = String(err);
  }
  corrBusy = false;
  showDagger(dagger);
}
corr.toggle.addEventListener('click', () => order(
  dagger?.thrown ? '/dagger/redo' : dagger?.hand_has ? '/dagger/handback' : '/dagger/takeover'));
corr.follow.addEventListener('click', () => order('/dagger/follow'));
// The stop, sent itself rather than through the Stop button: that one is off
// once no run is going, and the hands can still have the arms then.
corr.stop.addEventListener('click', async () => {
  if (dagger?.correcting && !dagger?.thrown && !dagger?.redoing) {
    order('/dagger/throw');
    return;
  }
  const answer = await ask('/policy/halt', {});
  corr.note.textContent = answer.error || 'stopped - the arms hold where they are; the run\'s recording is kept';
});

// A press on the floor flashes its pedal here, as on the collect page.
const CORR_PEDALS = { jog: corr.follow, handheld: corr.toggle, stop: corr.stop };
let pedalSeen = 0;
watch('pedal', (pedal) => {
  if (!pedal || pedal.count === pedalSeen) return;
  const first = pedalSeen === 0;
  pedalSeen = pedal.count;
  if (first && pedal.age > 2) return;
  const button = CORR_PEDALS[pedal.name];
  if (!button) return;
  button.classList.remove('pressed');
  void button.offsetWidth;
  button.classList.add('pressed');
  setTimeout(() => button.classList.remove('pressed'), 350);
});
watch('dagger', showDagger);
watch('episodes', ({ episodes }) => {
  rollouts = Object.fromEntries((episodes || []).filter((e) => e.kind === 'rollout' && e.complete).map((e) => [e.name, e]));
  showTally();
  showSaved();
});
// This page in front, while it is visible: what makes the floor's pedals
// correct rather than drive.
setPage('policy');
