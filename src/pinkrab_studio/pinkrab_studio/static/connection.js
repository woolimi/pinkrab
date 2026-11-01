// The connection list: a line per thing a page's Connect opens, each with what
// it said last and a Refresh that opens that one thing again, and a bar while
// they come up. Shown from Connect for as long as the page is connected,
// folded to its title by a click on it, which the page remembers.
//
// Fed, not polled: the page is already told what state the server is in, and
// hands that here (`set`). The page says when it is connected and which rows
// may be refreshed now; this draws.

// `off` is a part closed on purpose until it is wanted (the policy page's top camera).
const SETTLED = new Set(['running', 'failed', 'already', 'exited', 'off']);
// How long the bar stays after every line has settled, and the longest it
// waits for lines that never do.
const LINGER_MS = 2500;
const GIVE_UP_MS = 30000;
// A Refresh the server refused keeps its reason on the row this long: the
// updates would otherwise write the part's state over it within a second.
const HOLD_MS = 6000;
// The answer to a Refresh is only that it was asked - what it found comes with
// the next update - so the button rests a moment rather than invite a second click.
const REST_MS = 2000;

export class Connection {
  // `parts` is [[key, label, refresh], ...] in the order they are drawn; a
  // part without `refresh` has no button. `shown()` says whether the page is
  // connected; `canRefresh(key)` whether that row's button may be pressed now.
  constructor(el, parts, { storageKey, shown, canRefresh = () => true }) {
    this.el = el;
    this.parts = parts;
    this.shown = shown;
    this.canRefresh = canRefresh;
    this.storageKey = storageKey;
    this.steps = {};         // key -> { state, text }
    this.held = {};          // key -> until when a refused Refresh's reason stays
    this.refreshing = new Set();
    this.connecting = null;  // { since, settledAt, timer } from Connect until everything answered
    this.title = el.querySelector('h3');
    this.folded = false;
    try { this.folded = localStorage.getItem(storageKey) === '1'; } catch { /* no storage here */ }
    const fold = () => {
      this.folded = !this.folded;
      try { localStorage.setItem(storageKey, this.folded ? '1' : '0'); } catch { /* no storage here */ }
      this.draw();
    };
    this.title.addEventListener('click', fold);
    this.title.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fold(); }
    });
    // One row per part, made once: redrawn on every update, a row made
    // afresh each time took the click from under the pointer.
    this.rows = {};
    el.querySelector('ul').replaceChildren(...parts.map(([key, label, refresh]) => {
      const li = document.createElement('li');
      const what = document.createElement('span');
      li.append(what);
      let button = null;
      if (refresh) {
        button = document.createElement('button');
        button.className = 'refresh';
        button.title = `open the ${label} again`;
        button.addEventListener('click', () => this.refresh(key, refresh));
        li.append(button);
      }
      this.rows[key] = { li, what, button };
      return li;
    }));
  }

  // Connect pressed: every line waiting again, and the bar up.
  begin() {
    this.steps = {};
    this.connecting = { since: Date.now() };
    this.draw();
  }

  // Disconnect pressed, or a Connect that came to nothing.
  end() {
    this.connecting = null;
    this.draw();
  }

  set(key, state, text) {
    if ((this.held[key] || 0) > Date.now()) return;
    this.steps[key] = { state, text };
  }

  async refresh(key, refresh) {
    this.refreshing.add(key);
    this.set(key, 'starting', 'opening again ...');
    this.draw();
    const refused = (text) => { this.steps[key] = { state: 'failed', text }; this.held[key] = Date.now() + HOLD_MS; };
    try {
      const answer = await refresh();
      if (answer?.error) refused(answer.error);
    } catch (err) {
      refused(String(err));
    }
    setTimeout(() => { this.refreshing.delete(key); this.draw(); }, REST_MS);
    this.draw();
  }

  draw() {
    const c = this.connecting;
    if (c?.timer) { clearTimeout(c.timer); c.timer = null; }
    const shown = !!c || this.shown();
    this.el.hidden = !shown;
    if (!shown) return;
    const done = this.parts.filter(([k]) => SETTLED.has(this.steps[k]?.state)).length;
    const trouble = this.parts.filter(([k]) => ['failed', 'exited'].includes(this.steps[k]?.state)).length;
    if (c) {
      if (done === this.parts.length && !c.settledAt) c.settledAt = Date.now();
      const settledFor = c.settledAt ? Date.now() - c.settledAt : 0;
      if (settledFor > LINGER_MS || Date.now() - c.since > GIVE_UP_MS) this.connecting = null;
    }
    const connecting = this.connecting;
    this.el.classList.toggle('folded', this.folded);
    this.el.querySelector('.bar').hidden = !connecting;
    this.el.querySelector('.bar > span').style.width = `${(done / this.parts.length) * 100}%`;
    this.title.textContent = connecting ? 'connecting'
      : `connection - ${trouble ? `${trouble} not working` : `${done}/${this.parts.length} up`}`;
    this.title.title = this.folded ? 'show the list' : 'fold the list';
    for (const [key, label] of this.parts) {
      const row = this.rows[key];
      const step = this.steps[key] || { state: 'pending', text: 'waiting' };
      row.li.className = step.state;
      row.what.textContent = `${label} - ${step.text}`;
      if (row.button) {
        row.button.textContent = this.refreshing.has(key) ? '...' : '↻';
        row.button.disabled = this.refreshing.has(key) || !this.canRefresh(key);
      }
    }
    // Come back to take the bar down once it has lingered, or once it has
    // waited long enough, even if nothing else feeds this in the meantime.
    if (connecting) connecting.timer = setTimeout(() => this.draw(), connecting.settledAt ? LINGER_MS + 100 : 1000);
  }
}
