// The dashboard: the robot, the hands, the top camera and the handheld, each in a pane,
// the panes split by bars you drag.
//
// A layout is a small tree. A leaf is a pane's name; a node is a split - which
// way it runs, its two halves, and how the room is shared between them, as two
// percentages that add to a hundred. Dragging the bar between two halves moves
// room from one to the other and nowhere else, which is what a bar between two
// things should do; double-clicking it turns the split the other way round.
// What you drag is remembered in the browser, and one button puts it back.
//
// The panes themselves are written in index.html. This module builds the
// splits around them and moves each section into its slot, so the page's
// markup stays readable as three things rather than as a tree of divs.
//
// The two switches - the camera's and the loop's - are here too. Their state
// rides down the physics topic with everything else, so a process that dies
// on its own shows as dead within a frame, and the camera's numbers come down
// the same socket (live.js) while it runs.

import { streamUrl } from './cams.js';
import { watch } from './live.js';
import { topLight } from './toplight.js';

const STORE = 'pinkrab.dash.v3';
const MIN_PCT = 18;   // no half of a split thinner than this - a pane you cannot read is not a pane

const DEFAULT = {
  dir: 'row', size: [72, 28],
  a: { dir: 'row', size: [50, 50], a: 'robot', b: 'hands' },
  b: { dir: 'column', size: [42, 58], a: 'teleop', b: 'camera' },
};

async function ask(path, body) {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {}),
  });
  const answer = await response.json();
  if (!response.ok) throw new Error(answer.error ?? response.statusText);
  return answer;
}

function load() {
  try {
    const kept = JSON.parse(localStorage.getItem(STORE) ?? 'null');
    if (kept && kept.dir && kept.a && kept.b) return kept;
  } catch { /* a layout that does not parse is no layout */ }
  return structuredClone(DEFAULT);
}

export class Dash {
  constructor() {
    this.root = document.getElementById('dash');
    this.panes = {
      robot: document.getElementById('pane-robot'),
      hands: document.getElementById('pane-hands'),
      camera: document.getElementById('pane-camera'),
      teleop: document.getElementById('pane-teleop'),
    };
    this.tree = load();
    this.#render();
    this.#wireRig();
  }

  // --- the splits ----------------------------------------------------------

  #render() {
    // Panes out first, so building the tree does not move a section into a
    // slot that is about to be thrown away with the old tree.
    for (const pane of Object.values(this.panes)) document.body.appendChild(pane);
    this.root.replaceChildren(this.#build(this.tree));
  }

  #build(node) {
    if (typeof node === 'string') return this.panes[node];
    const split = document.createElement('div');
    split.className = `split ${node.dir}`;
    const a = this.#build(node.a);
    const bar = document.createElement('div');
    bar.className = 'bar';
    bar.title = 'drag to resize · double-click to turn the split';
    const b = this.#build(node.b);
    a.style.flexBasis = `${node.size[0]}%`;
    b.style.flexBasis = `${node.size[1]}%`;
    split.append(a, bar, b);
    this.#drag(bar, split, node, a, b);
    bar.addEventListener('dblclick', () => {
      node.dir = node.dir === 'row' ? 'column' : 'row';
      this.#save();
      this.#render();
    });
    return split;
  }

  #drag(bar, split, node, a, b) {
    bar.addEventListener('pointerdown', (down) => {
      down.preventDefault();
      bar.setPointerCapture(down.pointerId);
      document.body.classList.add('dragging');
      const rect = split.getBoundingClientRect();
      const move = (event) => {
        const along = node.dir === 'row'
          ? (event.clientX - rect.left) / rect.width
          : (event.clientY - rect.top) / rect.height;
        const first = Math.min(100 - MIN_PCT, Math.max(MIN_PCT, along * 100));
        node.size = [first, 100 - first];
        a.style.flexBasis = `${node.size[0]}%`;
        b.style.flexBasis = `${node.size[1]}%`;
      };
      const up = () => {
        bar.removeEventListener('pointermove', move);
        bar.removeEventListener('pointerup', up);
        bar.removeEventListener('pointercancel', up);
        document.body.classList.remove('dragging');
        this.#save();
      };
      bar.addEventListener('pointermove', move);
      bar.addEventListener('pointerup', up);
      bar.addEventListener('pointercancel', up);
    });
  }

  #save() {
    try { localStorage.setItem(STORE, JSON.stringify(this.tree)); } catch { /* fine */ }
  }

  /** The layout as it was before anybody dragged anything. */
  reset() {
    this.tree = structuredClone(DEFAULT);
    this.#save();
    this.#render();
  }

  // --- the two processes ---------------------------------------------------

  #wireRig() {
    this.rows = {
      live: {
        dot: document.getElementById('live-dot'),
        toggle: document.getElementById('live-toggle'),
        line: document.getElementById('live-line'),
        options: () => ({}),
      },
    };
    this.pictureEl = document.getElementById('camera-frame');
    this.statusEl = document.getElementById('camera-status');
    this.offEl = document.getElementById('camera-off');
    this.showLight = topLight(document.getElementById('camera-light'));
    this.liveUrl = null;
    this.rig = null;
    this.stats = null;
    for (const [which, row] of Object.entries(this.rows)) {
      row.toggle.addEventListener('click', () => this.#toggle(which));
    }
    watch('stats', (stats) => { this.stats = stats; this.#showStats(stats); });
  }

  async #toggle(which) {
    const row = this.rows[which];
    const running = this.rig?.[which]?.running;
    row.toggle.disabled = true;
    if (which === 'live') this.stats = null;   // the camera about to start or stop is not the one they came from
    try {
      const answer = await ask(`/rig/${which}/${running ? 'stop' : 'start'}`,
                               running ? {} : row.options());
      this.frame(answer.rig);
    } catch (error) {
      row.line.textContent = String(error.message ?? error);
      row.line.classList.add('bad');
    } finally {
      row.toggle.disabled = false;
    }
  }

  /** Called with every frame's `rig` field, and with a switch's answer. */
  frame(rig) {
    if (!rig || !rig.live) return;
    const wasUp = !!this.rig?.live?.running;
    this.rig = rig;
    // Where the camera's page is: another server on another port, and only
    // the studio's server knows which.
    if (rig.live_url) this.liveUrl = rig.live_url;
    const names = { live: 'camera' };
    for (const [which, row] of Object.entries(this.rows)) {
      const one = rig[which];
      row.dot.className = 'dot' + (one.running ? ' on' : one.exit ? ' bad' : '');
      this.panes.camera.classList.toggle('running', one.running);
      row.toggle.textContent = `${one.running ? 'Stop' : 'Start'} ${names[which]}`;
      // What it said, when it is not running. Running, the loop has its log
      // and the camera its numbers, both in the pane under this line.
      row.line.textContent = !one.running && one.note ? one.note : '';
      row.line.classList.toggle('bad', !one.running && !!one.exit);
    }
    // The picture is asked for once the camera's server answers (#showStats):
    // one pointed at a port nobody answers is a broken-image glyph in the pane.
    const cameraUp = rig.live.running && !!this.liveUrl;
    if (!cameraUp) this.showLight(null);
    if (!cameraUp && this.pictureEl.src) this.pictureEl.removeAttribute('src');
    // Numbers that came before the rig said the camera was up come again
    // only when they change: drawn once as it comes up, not on every frame,
    // which would make the retry button anew under the pointer. Down, they
    // were the last camera's.
    if (!rig.live.running) this.stats = null;
    if (cameraUp && !wasUp) this.#showStats(this.stats);
    this.pictureEl.hidden = !cameraUp;
    this.statusEl.hidden = !cameraUp;
    this.offEl.hidden = cameraUp;
    // Off, the pane says why - a camera that would not start is not a camera
    // that is off, and the reason belongs where the eye already is.
    if (!cameraUp) {
      const p = this.offEl.querySelector('p');
      const note = rig.live.exit ? rig.live.note : '';
      if (note) {
        // Everything it said on the way out, in order; the last line is
        // usually the reason, and the lines before it say where it got to.
        const said = document.createElement('code');
        said.className = 'said';
        said.textContent = (rig.live.tail ?? [note]).join('\n');
        p.replaceChildren('The camera did not start. ', document.createElement('br'), said);
        this.offEl.classList.add('bad');
      } else {
        p.textContent = 'The camera is off. Start it to see what the top camera sees and how the marker solves.';
        this.offEl.classList.remove('bad');
      }
    }
  }

  // The camera's numbers, as they change while it runs: how far the solid is,
  // how fast the pictures come, how many faces the solve stood on. None while
  // it is not running - in the simulation it never is - and none until its
  // server answers, which is also when its picture can be asked for.
  #showStats(stats) {
    const line = this.statusEl;
    this.showLight(this.rig?.live?.running ? stats?.head_light : null);
    if (!stats || !Object.keys(stats).length || !this.rig?.live?.running) return;
    if (this.liveUrl && !this.pictureEl.src) this.pictureEl.src = streamUrl(this.liveUrl + 'stream');
    if (stats.head === false) {
      // No top camera behind the view: say so, and offer to open it again.
      line.replaceChildren(stats.head_opening ? 'opening the top camera ...'
        : `no top camera - ${(stats.head_reason || 'not connected').split(':').pop().trim()}. `);
      if (!stats.head_opening) {
        const retry = document.createElement('button');
        retry.textContent = 'retry';
        retry.className = 'inline';
        retry.addEventListener('click', () => ask('/rig/live/head/retry', {}).catch(() => {}));
        line.append(retry);
      }
      return;
    }
    if (stats.fps == null) { line.textContent = 'opening the camera'; return; }
    const parts = [`${Math.round(stats.fps)} fps`];
    if (stats.solved && stats.distance_mm != null) {
      parts.unshift(`${(stats.distance_mm / 10).toFixed(0)} cm`);
      parts.push(`${stats.faces} ${stats.faces === 1 ? 'face' : 'faces'}`);
    } else {
      parts.push(stats.reason ? `no solid, ${stats.reason}` : 'no solid');
    }
    // Teleop follows each hand's motion against the desk solid - the camera
    // turns with the head - so a desk solid out of view holds the arms.
    if (stats.world && !stats.world.tracking) parts.push('desk solid not in view');
    // A frame that would not solve. The picture keeps coming; this says why
    // the numbers beside it may be old.
    if (stats.trouble) parts.push(`frame failed: ${stats.trouble}`);
    line.textContent = parts.join('   ');
  }
}
