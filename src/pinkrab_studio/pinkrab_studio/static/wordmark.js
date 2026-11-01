// The name on the front page: two solid layers.
//
// "pink" is one layer, "crab" another in front of it (home.css); the k's claw
// and the c are the same shape, so straight on the front one hides the back
// one. Each is given thickness here: copies of its face stacked behind it,
// darker the further back. Straight on the faces make one k. While the
// pointer moves the pair turns, the front layer slides off the back one and
// the two words come apart; when it stops they slowly close into one again.

const DEPTH = 0.2;         // how far the front layer stands out, in widths of the name
const THICK = 0.04;        // how thick each layer is, in widths of the name
const SLICES = 12;         // copies that make up that thickness
const SHADE = [18, 48];    // % of black mixed into the nearest and the furthest copy
const PERSPECTIVE = 1.8;   // the eye's distance, in widths of the name
// The turn, from how fast the pointer moves - not where it is.
const TURN_DEG = 22;       // about the vertical, at full speed: always to the right,
                           // the way the front layer moves off the back one
const NOD_DEG = 9;         // about the horizontal, the way the pointer moves up or down
const FULL_SPEED = 1600;   // px/s that turn it all the way
const STILL_S = 0.1;       // no move for this long is the pointer at rest
const DROP_S = 0.25;       // how fast the pointer's speed is let go of at rest
const OPEN_S = 0.35;       // how fast the words part
const CLOSE_S = 0.8;       // and how slowly they close again

// Lengths follow the name's width, so it looks the same at every size.
function size(el) {
  const w = el.clientWidth;
  const depth = w * DEPTH;
  const perspective = w * PERSPECTIVE;
  el.style.setProperty('--depth', `${depth}px`);
  el.style.setProperty('--step', `${(w * THICK) / SLICES}px`);
  // On the h1: the perspective is its property, not the tilt's.
  el.parentElement.style.setProperty('--perspective', `${perspective}px`);
  // A layer pushed toward the eye by d is drawn p / (p - d) larger: scaled
  // back by the inverse about the eye's point, it lands on the one behind.
  el.style.setProperty('--undo', String((perspective - depth) / perspective));
}

/** The two layers made solid: SLICES copies of each face behind it. */
export function extrude(el) {
  if (!el) return;
  for (const [face, ink, front] of [[el.querySelector('.layer-pin'), '--wm-pin', false],
                                    [el.querySelector('.layer-crab'), '--wm-crab', true]]) {
    for (let i = 1; i <= SLICES; i += 1) {
      const side = face.cloneNode(true);
      side.classList.add('side');
      const z = front ? `calc(var(--depth) - var(--step) * ${i})` : `calc(var(--step) * ${-i})`;
      side.style.transform = `translateZ(${z})${front ? ' scale(var(--undo))' : ''}`;
      const black = SHADE[0] + ((SHADE[1] - SHADE[0]) * (i - 1)) / (SLICES - 1);
      side.style.color = `color-mix(in oklab, var(${ink}), black ${black.toFixed(1)}%)`;
      el.append(side);
    }
  }
  new ResizeObserver(() => size(el)).observe(el);
  size(el);
}


/**
 * The pair turned by the pointer's movement: the faster it moves, the further
 * apart the words; at rest they close into one k.
 */
export function tilt(el) {
  if (!el || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  let drive = 0;      // 0..1, how fast the pointer is moving now
  let rise = 0;       // -1..1, how much of that is upward
  let open = 0;       // 0..1, how far apart the words are shown
  let nod = 0;
  let lastMove = 0;
  let px = null;
  let py = null;
  let pt = 0;
  addEventListener('pointermove', (event) => {
    const now = performance.now();
    if (px !== null && now > pt) {
      const dt = (now - pt) / 1000;
      const vx = (event.clientX - px) / dt;
      const vy = (event.clientY - py) / dt;
      const speed = clamp(Math.hypot(vx, vy) / FULL_SPEED, 0, 1);
      drive += (Math.max(speed, drive) - drive) * 0.5;
      rise += (clamp(-vy / FULL_SPEED, -1, 1) - rise) * 0.3;
    }
    px = event.clientX;
    py = event.clientY;
    pt = now;
    lastMove = now;
    wake();
  }, { passive: true });

  let running = false;
  let last = 0;
  function wake() {
    if (running) return;
    running = true;
    last = performance.now();
    requestAnimationFrame(frame);
  }
  function frame(now) {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    if ((now - lastMove) / 1000 > STILL_S) {
      const letGo = Math.exp(-dt / DROP_S);
      drive *= letGo;
      rise *= letGo;
    }
    const ease = 1 - Math.exp(-dt / (drive > open ? OPEN_S : CLOSE_S));
    open += (drive - open) * ease;
    nod += (rise * open - nod) * ease;
    el.style.setProperty('--ry', `${(open * TURN_DEG).toFixed(3)}deg`);
    el.style.setProperty('--rx', `${(nod * NOD_DEG).toFixed(3)}deg`);
    if (open > 1e-3 || drive > 1e-3) requestAnimationFrame(frame);
    else {
      el.style.setProperty('--ry', '0deg');
      el.style.setProperty('--rx', '0deg');
      open = 0;
      nod = 0;
      running = false;
    }
  }
}
