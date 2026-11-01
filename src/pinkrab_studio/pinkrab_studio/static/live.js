// The page's one socket to the studio. The studio says what changed as it
// changes: each topic the page watches, once when it starts watching and again
// on every change. The page says back which topics, which page it is and
// whether it is in front, and the poses its sliders ask of the simulation.
// Actions that have an answer - start this, delete that - stay ordinary POSTs.
//
// The socket comes back by itself after the studio restarts, and every topic
// is sent again then, so a handler always sees the studio as it is now.

const handlers = new Map();      // topic -> Set of functions
const latest = new Map();        // topic -> the last value sent
const connectionHandlers = new Set();
let socket = null;
let open = false;
let waiting = null;              // the timer that opens the socket again
let retryMs = 500;
let page = '';
let asking = false;              // a watch list is due on the next microtask

/** Call `fn` with every value of `topic`, starting with the last one if any. */
export function watch(topic, fn) {
  if (!handlers.has(topic)) handlers.set(topic, new Set());
  handlers.get(topic).add(fn);
  if (latest.has(topic)) fn(latest.get(topic));
  askSoon();
}

function forget(topic, fn) {
  const set = handlers.get(topic);
  if (!set) return;
  set.delete(fn);
  if (!set.size) { handlers.delete(topic); latest.delete(topic); askSoon(); }
}

/** The first value of `topic` that passes `test` - null if none has in `ms`.
 *  `fresh` leaves out the value already here: only what is sent from now on
 *  counts, for a wait on something just asked for. */
export function until(topic, test, ms, { fresh = false } = {}) {
  return new Promise((resolve) => {
    let settled = false;
    let listening = !fresh;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      queueMicrotask(() => forget(topic, check));
      resolve(value);
    };
    const check = (value) => { if (listening && test(value)) finish(value); };
    const timer = setTimeout(() => finish(null), ms);
    watch(topic, check);
    listening = true;
  });
}

/** Send a message up the socket; dropped while it is closed. */
export function say(message) {
  if (open) socket.send(JSON.stringify(message));
}

/** Which page this is: the studio arms the pedals while the collect page is in
 *  front. A named page says it is visible once a second for as long as it is -
 *  what the studio hears, not what it last heard, keeps them armed. */
export function setPage(name) {
  page = name;
  say({ page });
  setInterval(() => { if (document.visibilityState === 'visible') say({ visible: true }); }, 1000);
}

/** Call `fn` with true when the socket opens and false when it closes. */
export function onConnection(fn) {
  connectionHandlers.add(fn);
  fn(open);
}

function askSoon() {
  if (asking) return;
  asking = true;
  queueMicrotask(() => {
    asking = false;
    say({ watch: [...handlers.keys()] });
    if (!socket && !waiting) connect();
  });
}

function connect() {
  socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  socket.addEventListener('open', () => {
    open = true;
    retryMs = 500;
    say({ watch: [...handlers.keys()], page, visible: document.visibilityState === 'visible' });
    for (const fn of connectionHandlers) fn(true);
  });
  socket.addEventListener('message', (event) => {
    let message;
    try { message = JSON.parse(event.data); } catch { return; }
    // A topic no longer watched may still arrive, sent before the server had
    // the new list: kept, it would be the stale first value of the next watch.
    if (!handlers.has(message.topic)) return;
    latest.set(message.topic, message.data);
    for (const fn of [...(handlers.get(message.topic) ?? [])]) {
      // One handler failing must not keep the value from the others.
      try { fn(message.data); } catch (err) { console.error(`${message.topic}:`, err); }
    }
  });
  socket.addEventListener('close', () => {
    const was = open;
    open = false;
    socket = null;
    if (was) for (const fn of connectionHandlers) fn(false);
    waiting = setTimeout(() => { waiting = null; connect(); }, retryMs);
    retryMs = Math.min(retryMs * 2, 5000);
  });
}

document.addEventListener('visibilitychange', () => say({ visible: document.visibilityState === 'visible' }));

// --- what went wrong -------------------------------------------------------------
// The studio's alerts (alerts.py) - a process that died, a checkpoint that
// would not load, a conversion or a job that failed - stand in a corner of
// whichever page is open until they are closed. On a socket's first word the
// recent ones are shown too: an error a moment before the page opened is news.
const RECENT_S = 120;
const CLOSED_KEY = 'studio.alerts.closed';
const ALERT_STYLE = `
.studio-alerts { position: fixed; top: 14px; right: 14px; z-index: 1000; display: flex; flex-direction: column;
  gap: 8px; width: min(420px, calc(100vw - 28px)); pointer-events: none; }
.studio-alert { pointer-events: auto; background: #fff; color: #2c1a25; border: 1px solid #ebb4b4;
  border-left: 4px solid #d03b3b; border-radius: 8px; padding: 10px 12px; box-shadow: 0 8px 24px rgba(0,0,0,.14);
  font: 13px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; animation: studio-alert-in .18s ease-out; }
.studio-alert header { display: flex; gap: 8px; align-items: baseline; }
.studio-alert header b { color: #b02a2a; font-weight: 700; }
.studio-alert header .when { margin-left: auto; color: #8f7283; font-size: 11px; font-variant-numeric: tabular-nums; }
.studio-alert header button { flex: 0 0 auto; margin: 0; padding: 0 2px; border: 0; background: none; color: #8f7283;
  font-size: 17px; line-height: 1; cursor: pointer; }
.studio-alert header button:hover { color: #2c1a25; background: none; }
.studio-alert p { margin: 4px 0 0; overflow-wrap: anywhere; }
.studio-alert details { margin-top: 6px; }
.studio-alert summary { cursor: pointer; color: #8f7283; font-size: 12px; }
.studio-alert pre { margin: 6px 0 0; max-height: 180px; overflow: auto; padding: 6px 8px; border-radius: 6px;
  background: #f9f1f4; font: 11px/1.4 ui-monospace, monospace; white-space: pre-wrap; overflow-wrap: anywhere; }
@keyframes studio-alert-in { from { opacity: 0; transform: translateY(-6px); } to { opacity: 1; transform: none; } }
@media (prefers-reduced-motion: reduce) { .studio-alert { animation: none; } }`;
const shownAlerts = new Set();
let alertsFirst = true;
let alertStack = null;

const alertKey = (a) => `${a.at}:${a.id}`;

function closedAlerts() {
  try { return new Set(JSON.parse(localStorage.getItem(CLOSED_KEY) || '[]')); } catch { return new Set(); }
}

function alertBox() {
  if (!alertStack) {
    const style = document.createElement('style');
    style.textContent = ALERT_STYLE;
    document.head.append(style);
    alertStack = document.createElement('div');
    alertStack.className = 'studio-alerts';
    alertStack.setAttribute('aria-live', 'assertive');
    document.body.append(alertStack);
  }
  return alertStack;
}

function showAlert(a) {
  const card = document.createElement('div');
  card.className = 'studio-alert';
  const head = document.createElement('header');
  const source = Object.assign(document.createElement('b'), { textContent: a.source });
  const when = Object.assign(document.createElement('span'), {
    className: 'when', textContent: new Date(a.at * 1000).toLocaleTimeString() });
  const close = Object.assign(document.createElement('button'), { textContent: '×', title: 'close' });
  close.setAttribute('aria-label', 'close this alert');
  close.addEventListener('click', () => {
    card.remove();
    // Closed here is closed on every page, and stays so across a reload.
    try { localStorage.setItem(CLOSED_KEY, JSON.stringify([...closedAlerts(), alertKey(a)].slice(-100))); } catch { /* no storage */ }
  });
  head.append(source, when, close);
  card.append(head, Object.assign(document.createElement('p'), { textContent: a.message }));
  if (a.detail?.length) {
    const said = document.createElement('details');
    said.append(Object.assign(document.createElement('summary'), { textContent: 'what it said' }),
                Object.assign(document.createElement('pre'), { textContent: a.detail.join('\n') }));
    card.append(said);
  }
  alertBox().prepend(card);   // the newest on top
}

watch('alerts', (data) => {
  if (!data?.alerts) return;
  const closed = closedAlerts();
  for (const a of data.alerts) {
    const key = alertKey(a);
    if (shownAlerts.has(key) || closed.has(key)) continue;
    shownAlerts.add(key);
    if (alertsFirst && data.now - a.at > RECENT_S) continue;   // old news
    showAlert(a);
  }
  alertsFirst = false;
});
// A studio come back is a first word again: its alerts are judged by age.
onConnection((up) => { if (up) alertsFirst = true; });
