// The front page: choose the robot, then go to the teleop page.
// The choice is made on the server before the page opens, so the teleop page
// builds itself for the right robot from the first frame. Real is checked
// first: with no driver and no gate on this machine there is nothing to
// drive, and the line to type is shown rather than a page that waits.
import { until, watch } from './live.js';

const hint = document.getElementById('teleop-hint');
// Where a page sent us from, to go on to once sim or real is chosen.
const next = (() => {
  const asked = new URLSearchParams(location.search).get('next');
  return asked && asked.startsWith('/') && !asked.startsWith('//') ? asked : null;
})();
if (next) hint.textContent = `Choose the real robot or the simulation first - ${next} opens after.`;
const buttons = [...document.querySelectorAll('button[data-real]')];
const dialog = document.getElementById('bringup');
const what = document.getElementById('bringup-what');
const cmd = document.getElementById('bringup-cmd');

const progress = document.getElementById('progress');
const NAMES = { bridge: 'bridge', teleop: 'handheld loop' };
const WORDS = { pending: 'waiting', starting: 'starting ...', running: 'running', already: 'already running',
                failed: 'did not start - see the terminal', exited: 'exited - see the terminal' };

// The two starts, drawn as they happen: the server starts them one after
// another and says which it is on; this draws each as it comes, then goes.
function drawProgress(robot) {
  progress.hidden = false;
  const parts = robot?.parts || {};
  const done = Object.values(parts).filter((p) => ['running', 'already', 'failed', 'exited'].includes(p.step)).length;
  progress.querySelector('.bar > span').style.width = `${(done / Object.keys(NAMES).length) * 100}%`;
  progress.querySelector('ul').replaceChildren(...Object.entries(NAMES).map(([key, label]) => {
    const li = document.createElement('li');
    const step = parts[key]?.step || 'pending';
    li.className = step;
    li.textContent = `${label} - ${WORDS[step] || step}`;
    return li;
  }));
}

async function go(real) {
  if (!real) {
    hint.textContent = 'stopping the robot side ...';
    try {
      const answer = await (await fetch('/mode', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                                   body: JSON.stringify({ real }) })).json();
      // Refused while a policy has the arm: the robot side stays, and so does this page.
      if (answer.error) {
        hint.textContent = answer.error;
        for (const b of buttons) b.disabled = false;
        return;
      }
      location.href = next ?? '/teleop';
    } catch (err) {
      hint.textContent = `could not reach the studio: ${err}`;
      for (const b of buttons) b.disabled = false;
    }
    return;
  }
  hint.textContent = '';
  try {
    const answer = await (await fetch('/mode', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                                 body: JSON.stringify({ real }) })).json();
    drawProgress(answer.rig?.robot);
    // Until the server has been through both, then open the page: a rig
    // sent since this choice, asked and no longer starting.
    if (answer.rig?.robot?.starting) {
      await until('rig', (rig) => {
        drawProgress(rig?.robot);
        return !!rig?.robot?.asked && !rig.robot.starting;
      }, 50000, { fresh: true });
    }
    await new Promise((r) => setTimeout(r, 600));
    location.href = next ?? '/teleop';
  } catch (err) {
    hint.textContent = `could not reach the studio: ${err}`;
    for (const b of buttons) b.disabled = false;
  }
}

// Whether the driver and the gate are running here, as the studio last
// looked; null if it has not said within a few seconds.
function robotUp() {
  return until('robot', () => true, 3000);
}

// Say which half is missing, and the command that starts it.
function describe(up, again) {
  const missing = !up ? 'the studio could not look'
    : (!up.driver && !up.gate) ? 'neither the driver nor the gate is running'
    : !up.driver ? 'the gate is running but the driver is not'
    : 'the driver is running but the gate is not';
  what.textContent = `${again ? 'Still not up. ' : ''}Nothing is driving the arms on this machine: ${missing}.`;
  cmd.replaceChildren(terminal('ros2 launch pinkrab_bringup robot.launch.py'));
}

async function choose(real) {
  for (const b of buttons) b.disabled = true;
  if (!real) return go(false);
  const up = await robotUp();
  if (up && up.driver && up.gate) return go(true);
  describe(up, false);
  hint.textContent = '';
  dialog.showModal();
}

document.getElementById('bringup-cancel').addEventListener('click', () => {
  dialog.close();
  for (const b of buttons) b.disabled = false;
});
// Escape closes it too, without the button: the choice comes back the same way.
dialog.addEventListener('cancel', () => { for (const b of buttons) b.disabled = false; });
document.getElementById('bringup-retry').addEventListener('click', async () => {
  const up = await robotUp();
  if (up && up.driver && up.gate) { dialog.close(); return go(true); }
  describe(up, true);
});

for (const b of buttons) b.addEventListener('click', () => choose(b.dataset.real === '1'));

// Sent here by Teleop (the pages in the server's ARM_PAGES): ask, in front of
// everything. The line under the buttons stays for after a Cancel.
const PAGE_NAMES = { '/teleop': 'Teleop' };
const ask = document.getElementById('choose');
const why = new URLSearchParams(location.search).get('why');
const askChoice = () => {
  document.getElementById('choose-what').textContent =
    `${PAGE_NAMES[next] ?? next} drives an arm, so it opens once you have chosen which arms: `
    + 'the real ones on the wall, or the simulation.';
  ask.showModal();
};
if (next && why === 'down') {
  // Real was chosen and the robot's side had gone: say what is down - unless
  // it is back, as after a reload once it was started, when it is a choice.
  robotUp().then((up) => {
    if (up?.driver && up?.gate) return askChoice();
    describe(up, false);
    for (const b of buttons) b.disabled = true;
    dialog.showModal();
  });
} else if (next) {
  askChoice();
}
document.getElementById('choose-cancel').addEventListener('click', () => ask.close());
document.getElementById('choose-sim').addEventListener('click', () => { ask.close(); choose(false); });
document.getElementById('choose-real').addEventListener('click', () => { ask.close(); choose(true); });

// A command as a terminal line: a prompt, the command, and a button that puts
// the command on the clipboard. Each line of a multi-line command gets its
// own prompt; the button copies all of them.
function terminal(command) {
  const box = Object.assign(document.createElement('div'), { className: 'term' });
  const pre = document.createElement('pre');
  pre.append(...command.split('\n').map((line) => {
    const row = Object.assign(document.createElement('span'), { className: 'line' });
    row.append(Object.assign(document.createElement('span'), { className: 'prompt', textContent: '$' }), line);
    return row;
  }));
  const copy = Object.assign(document.createElement('button'), { type: 'button', className: 'copy', textContent: 'Copy' });
  copy.setAttribute('aria-label', `Copy: ${command}`);
  copy.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(command);
      copy.textContent = 'Copied';
    } catch {
      getSelection().selectAllChildren(pre);
      copy.textContent = 'Press \u2318C';
    }
    copy.classList.add('copied');
    setTimeout(() => { copy.textContent = 'Copy'; copy.classList.remove('copied'); }, 1500);
  });
  box.append(pre, copy);
  return box;
}

// Which one the server is on now, marked so a page reopened mid-session says so.
// Before starting: the server's list in its groups, a group with something
// left first and, inside a group, what is left first - each with the command
// under it - then what is done, as a line. Optional items (a second handheld)
// count only when done. Folded into the pill at the top once every
// required item is done; open by itself while one is not.
const beforeList = document.getElementById('before-list');
const beforeCount = document.getElementById('before-count');
const beforeCard = document.getElementById('before');
const beforeBar = document.getElementById('before-bar');
const pill = document.getElementById('setup-pill');
pill.classList.add('checking');
pill.addEventListener('click', () => {
  beforeCard.hidden = !beforeCard.hidden;
  pill.setAttribute('aria-expanded', String(!beforeCard.hidden));
  if (!beforeCard.hidden) beforeCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
});
async function drawBefore() {
  let items = [], groups = {};
  try {
    ({ items = [], groups = {} } = await (await fetch('/checklist', { cache: 'no-store' })).json());
  } catch (err) {
    beforeList.replaceChildren(Object.assign(document.createElement('p'), { className: 'pending', textContent: `could not ask the server: ${err}` }));
    return;
  }
  const required = items.filter((i) => !i.optional);
  const done = required.filter((i) => i.ok).length;
  const optional = items.filter((i) => i.optional && !i.ok).length;
  beforeCount.textContent = done === required.length ? 'all done' : `${done} of ${required.length} done`;
  const missing = required.length - done;
  pill.classList.remove('checking');
  pill.classList.toggle('missing', missing > 0);
  pill.textContent = missing ? `${missing} setup step${missing === 1 ? '' : 's'} left`
    : optional ? `Set up, ${optional} optional left` : 'Set up';
  if (missing) { beforeCard.hidden = false; pill.setAttribute('aria-expanded', 'true'); }
  beforeBar.style.width = `${required.length ? (100 * done) / required.length : 100}%`;
  beforeCard.classList.toggle('complete', missing === 0);
  // Groups with something left first, in their own order; finished ones after.
  const order = Object.keys(groups);
  for (const item of items) if (!order.includes(item.group)) order.push(item.group);
  const parts = order.map((key) => items.filter((i) => i.group === key)).filter((g) => g.length);
  const left = (g) => g.some((i) => !i.ok && !i.optional);
  beforeList.replaceChildren(...[...parts.filter(left), ...parts.filter((g) => !left(g))]
    .map((g) => group(groups[g[0].group] || g[0].group, g)));
}
// One group (the arms, the handheld, this machine): a heading with how much
// of it is done, what is left as cards carrying their command as a terminal
// line, then what is done as quiet ticked lines.
function group(title, items) {
  const required = items.filter((i) => !i.optional);
  const done = required.filter((i) => i.ok).length;
  const complete = done === required.length;
  const box = document.createElement('section');
  box.className = `todo-group${complete ? ' complete' : ''}`;
  const head = document.createElement('h3');
  head.append(title, Object.assign(document.createElement('span'), {
    className: 'n', textContent: complete ? '\u2713 done' : `${done} / ${required.length}`,
  }));
  const ul = document.createElement('ul');
  ul.className = 'todo';
  const rank = (i) => (i.ok ? 2 : i.optional ? 1 : 0);
  ul.append(...[...items].sort((x, y) => rank(x) - rank(y)).map((item) => {
    const kind = item.ok ? 'ok' : item.optional ? 'optional' : 'missing';
    const li = document.createElement('li');
    li.className = kind;
    const mark = Object.assign(document.createElement('span'), { className: 'mark', textContent: item.ok ? '\u2713' : '' });
    mark.setAttribute('aria-hidden', 'true');
    const text = Object.assign(document.createElement('div'), { className: 'text' });
    text.append(Object.assign(document.createElement('span'), {
      className: 'label', textContent: item.label + (kind === 'optional' ? ' (optional)' : ''),
    }));
    if (item.detail) text.append(Object.assign(document.createElement('span'), { className: 'detail', textContent: item.detail }));
    if (!item.ok && item.run) text.append(terminal(item.run));
    li.append(mark, text);
    return li;
  }));
  box.append(head, ul);
  return box;
}
document.getElementById('before-again').addEventListener('click', drawBefore);
drawBefore();

fetch('/physics').then((r) => r.json()).then((info) => {
  for (const b of buttons) b.classList.toggle('current', (b.dataset.real === '1') === !!info.real);
}).catch(() => {});

// Where each stage stands, as it changes: a run going on somewhere keeps its
// line moving.
function hours(seconds) {
  if (!seconds) return '';
  const h = Math.floor(seconds / 3600), m = Math.round((seconds % 3600) / 60);
  return h ? `${h} h ${m} min left` : `${m} min left`;
}
function kilo(n) { return n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : String(n ?? 0); }
function drawNow(home) {
  if (!home || home.error) return;
  const collect = document.getElementById('now-collect');
  const minutes = Math.round(home.episode_seconds / 60);
  collect.innerHTML = home.episodes
    ? `<b>${home.episodes}</b> episodes, ${minutes} min recorded`
    : 'Nothing recorded yet.';
  document.getElementById('now-curate').innerHTML = home.datasets || home.labelled
    ? `<b>${home.labelled}</b> take${home.labelled === 1 ? '' : 's'} labelled; <b>${home.datasets}</b> dataset${home.datasets === 1 ? '' : 's'}`
    : 'No dataset yet.';
  const train = document.getElementById('now-train');
  // Recorded runs with their progress; beside them whatever is on a GPU
  // that no recorded run accounts for - seen to be training, with no numbers.
  const unrecorded = home.unrecorded || [];
  if (home.training.length || unrecorded.length) {
    train.replaceChildren(Object.assign(document.createElement('span'), {
      innerHTML: `<b>${home.training.length + unrecorded.length}</b> training now` }));
    for (const r of home.training.slice(0, 3)) {
      const row = document.createElement('div');
      row.className = 'run';
      row.innerHTML = '<span class="name"></span><span class="left"></span><span class="meter"><span></span></span>';
      row.querySelector('.name').textContent = r.name;
      row.querySelector('.left').textContent = `${kilo(r.step)} / ${kilo(r.steps)}${r.eta_s ? `, ${hours(r.eta_s)}` : ''}`;
      row.querySelector('.meter span').style.width = `${Math.min(100, (100 * (r.step || 0)) / (r.steps || 1))}%`;
      train.append(row);
    }
    for (const u of unrecorded.slice(0, 3)) {
      const row = document.createElement('div');
      row.className = 'run unrecorded';
      row.innerHTML = '<span class="name"></span><span class="left"></span>';
      row.querySelector('.name').textContent = u.name;
      row.querySelector('.name').title = u.name;
      row.querySelector('.left').textContent = `on ${u.source}, not recorded`;
      row.title = 'on the GPU, started without pinkrab-train: no progress or loss to show';
      train.append(row);
    }
  } else {
    train.innerHTML = home.runs ? `<b>${home.runs}</b> run${home.runs === 1 ? '' : 's'}, none training now` : 'No run yet.';
  }
  document.getElementById('now-policy').innerHTML = home.checkpoints
    ? `<b>${home.checkpoints}</b> checkpoint${home.checkpoints === 1 ? '' : 's'} on this machine`
    : 'No checkpoint on this machine yet.';
}
watch('home', drawNow);

// The name, made solid: its two layers part while the pointer moves and
// close into one word when it rests.
import('./wordmark.js').then(({ extrude, tilt }) => {
  const wordmark = document.getElementById('wordmark');
  extrude(wordmark);
  tilt(wordmark);
});

// The robot, once its meshes are in. A studio without a description shows the
// page without the picture rather than an empty frame.
import('./hero.js').then(({ showRobot }) => showRobot(document.getElementById('hero-robot')))
  .catch(() => {
    document.querySelector('.hero .scene').remove();
    document.querySelector('.hero').classList.add('no-robot');
  });
