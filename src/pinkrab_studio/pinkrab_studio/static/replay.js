// An episode played back: its videos in step, and the arms doing what
// the recorded hand would have made them do. The joints come from the studio
// (/episodes/<name>/replay.json, solved once and cached), one moment each on
// the take's own clock (`t`, seconds) - a demonstration's at the rate a
// dataset of it runs at. The page keeps that clock itself and sets each video
// to the frame the moment shows: a camera that skipped frames left its video
// shorter than the take, so no video can be the clock. Moments the solver
// flagged are red on the timeline.
import * as THREE from 'three';
import { OrbitControls } from './OrbitControls.js';
import { loadURDF } from './urdf.js';
import { HULL, OUTLINE, materialFor } from './look.js';

const dialog = document.getElementById('replay');
const titleEl = document.getElementById('replay-title');
const noteEl = document.getElementById('replay-note');
const canvas = document.getElementById('replay-view');
const videos = {
  head: document.getElementById('replay-head'),
  gripper_left: document.getElementById('replay-left'),
  gripper_right: document.getElementById('replay-right'),
};
const scrub = document.getElementById('replay-scrub');
const flagsCanvas = document.getElementById('replay-flags');
const playEl = document.getElementById('replay-play');
const frameEl = document.getElementById('replay-frame');
const whyEl = document.getElementById('replay-why');
const summaryEl = document.getElementById('replay-summary');

// --- the picture ---------------------------------------------------------------
let renderer = null, scene = null, camera = null, controls = null, world = null, robot = null;
let raf = 0;

async function ensureScene() {
  if (renderer) return;
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf9f1f4);
  // URDF is z-up; three.js is y-up. The world group turns one into the other.
  world = new THREE.Group();
  world.rotation.x = -Math.PI / 2;
  scene.add(world);
  camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
  camera.position.set(0.9, 0.7, 0.9);
  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  scene.add(new THREE.HemisphereLight(0xfffaf0, 0xcbbfa8, 1.5));
  const key = new THREE.DirectionalLight(0xfff6e8, 1.9);
  key.position.set(1.35, 2.1, 0.95);
  key.castShadow = true;
  Object.assign(key.shadow.camera, { left: -1, right: 1, top: 1, bottom: -1, near: 0.2, far: 6 });
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.bias = -0.0008;
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xeef2ff, 0.55);
  fill.position.set(-0.9, 0.7, -1.3);
  scene.add(fill);
  const desk = new THREE.Mesh(new THREE.PlaneGeometry(2.4, 1.6),
                              new THREE.MeshStandardMaterial({ color: 0xefe9dd, roughness: 0.95 }));
  desk.receiveShadow = true;
  world.add(desk);
  const grid = new THREE.GridHelper(2.4, 24, 0xc9bda6, 0xdbd2c1);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = 0.001;
  world.add(grid);
  // The board and the desk height, as the simulation states them.
  try {
    const physics = await (await fetch('/physics')).json();
    const slab = physics.scenery?.pegboard;
    if (slab) {
      const board = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1),
                                   new THREE.MeshStandardMaterial({ color: 0x1d1d20, roughness: 0.85 }));
      const [hx, hy, hz] = slab.half;
      board.scale.set(hx * 2, hy * 2, hz * 2);
      board.position.set(...slab.pos);
      board.castShadow = true; board.receiveShadow = true;
      world.add(board);
    }
    const floor = physics.scenery?.desk;
    if (floor) { desk.position.z = floor.pos[2]; grid.position.z = floor.pos[2] + 0.001; }
  } catch { /* the scenery is decoration */ }
  const urdf = await (await fetch('/model')).text();
  robot = await loadURDF(urdf, '/meshes', { materialFor, outlineMaterial: OUTLINE, hullMaterial: HULL });
  robot.root.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
  world.add(robot.root);
  fit();
}

function fit() {
  world.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(robot.root);
  const centre = box.getCenter(new THREE.Vector3());
  controls.target.copy(centre);
  camera.position.copy(centre).add(new THREE.Vector3(0.95, 0.55, 0.95));
  controls.update();
}

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  if (canvas.width !== w * renderer.getPixelRatio() || canvas.height !== h * renderer.getPixelRatio()) {
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
}

// --- the episode --------------------------------------------------------------------
let replay = null;     // replay.json
let current = -1;
// Playing: the take's time at `from`, and the page's clock (ms) then.
let playing = null;

// The last moment at or before `t` seconds into the take.
function frameAt(t) {
  if (!replay) return 0;
  const frames = replay.trajectory;
  let lo = 0, hi = frames.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (frames[mid].t <= t) lo = mid; else hi = mid - 1;
  }
  return lo;
}

// Each video at the frame moment `f` shows: left alone while it plays within
// a frame or two of it, set there otherwise.
function syncVideos(f) {
  for (const [name, v] of Object.entries(videos)) {
    const idx = f[name];
    if (idx == null || !v.duration) continue;
    const want = (idx + 0.5) / (replay.videos?.[name] || 30);
    if (Math.abs(v.currentTime - want) > (playing ? 0.1 : 0.02)) v.currentTime = want;
  }
}

function setPlaying(on) {
  if (!replay) return;
  if (on) {
    const last = replay.trajectory.length - 1;
    const from = current >= last ? 0 : replay.trajectory[Math.max(0, current)].t;
    playing = { from, at: performance.now() };
    for (const v of Object.values(videos)) v.play().catch(() => {});
  } else {
    playing = null;
    for (const v of Object.values(videos)) v.pause();
  }
  playEl.textContent = on ? 'pause' : 'play';
}

function showFrame(i) {
  if (!replay || i === current) return;
  current = i;
  const f = replay.trajectory[i];
  robot.setPose(f.joints);
  scrub.value = i;
  frameEl.textContent = `${i + 1} / ${replay.trajectory.length}   ${f.t.toFixed(2)} s`;
  const why = [];
  for (const arm of ['left', 'right']) {
    const flags = f.flags?.[arm] || [];
    if (!flags.length) continue;
    const words = flags.map((k) => k === 'short' ? `short by ${f.short?.[arm]?.mm ?? '?'} mm / ${f.short?.[arm]?.deg ?? '?'}°`
      : k === 'limit' ? `at the band: ${(f.at_limit?.[arm] || []).map((n) => n.replace(`${arm}_`, '')).join(', ')}`
      : k === 'lost' ? (replay.kind === 'rollout' ? 'no reading of the arm' : 'hand not in the world (arm holds)')
      : k === 'held' ? 'the clutch held the arm (not learned from)'
      : k === 'policy' ? 'the policy drove the arm (not learned from)'
      : k === 'idle' ? 'neither the policy nor the hand drove the arm (not learned from)'
      : k === 'jump' ? 'joint jump' : k === 'touch' ? `touching: ${(f.touching || []).join('; ')}` : k);
    why.push(`${arm}: ${words.join(' · ')}`);
  }
  whyEl.textContent = why.join('    ');
  whyEl.classList.toggle('bad', why.length > 0);
  syncVideos(f);
}

function drawFlags() {
  const ctx = flagsCanvas.getContext('2d');
  const w = flagsCanvas.width = flagsCanvas.clientWidth * devicePixelRatio;
  const h = flagsCanvas.height = flagsCanvas.clientHeight * devicePixelRatio;
  ctx.clearRect(0, 0, w, h);
  if (!replay) return;
  const n = replay.trajectory.length;
  for (let i = 0; i < n; i++) {
    const f = replay.trajectory[i];
    const l = f.flags?.left || [], r = f.flags?.right || [];
    const x0 = Math.floor(i / n * w), x1 = Math.max(x0 + 1, Math.floor((i + 1) / n * w));
    // Grey where the arm was not followed - the hand lost, a rollout's arm
    // not the hand's (the policy's, held by the clutch, or nobody's), or a
    // take with no place for the hand in the robot's world - red where it
    // would have been held back.
    const NOT_FOLLOWED = ['lost', 'held', 'policy', 'idle'];
    const grey = (flags) => replay.unjudged || flags.some((k) => NOT_FOLLOWED.includes(k));
    ctx.fillStyle = grey(l) ? '#c9bda6' : l.length ? '#e2488e' : '#5fd38a';
    ctx.fillRect(x0, 0, x1 - x0, h / 2 - 1);
    ctx.fillStyle = grey(r) ? '#c9bda6' : r.length ? '#e2488e' : '#5fd38a';
    ctx.fillRect(x0, h / 2 + 1, x1 - x0, h / 2 - 1);
  }
}

function tick() {
  raf = requestAnimationFrame(tick);
  resize();
  controls.update();
  if (replay && playing) {
    const t = playing.from + (performance.now() - playing.at) / 1000;
    const last = replay.trajectory.length - 1;
    showFrame(frameAt(t));
    if (t >= replay.trajectory[last].t) setPlaying(false);
  }
  renderer.render(scene, camera);
}

// A take keeps no top-camera video (its poses are what it keeps of that
// camera), nor an older one some stream it lost: that video's tile goes.
for (const v of Object.values(videos)) {
  v.addEventListener('error', () => { if (v.getAttribute('src')) v.closest('figure').hidden = true; });
}

export async function openReplay(name) {
  dialog.showModal();
  titleEl.textContent = name;
  noteEl.textContent = 'solving the arms ...';
  whyEl.textContent = '';
  summaryEl.textContent = '';
  replay = null; current = -1; playing = null; playEl.textContent = 'play';
  for (const [key, v] of Object.entries(videos)) {
    v.pause();
    v.closest('figure').hidden = false;
    v.src = `/episodes/${name}/${key}.mp4`;
    v.load();
  }
  try {
    await ensureScene();
    const answer = await (await fetch(`/episodes/${name}/replay.json`, { cache: 'no-store' })).json();
    if (answer.error) { noteEl.textContent = answer.error; return; }
    replay = answer;
    scrub.max = replay.trajectory.length - 1;
    const s = replay.summary;
    const words = (arm) => Object.entries(s.flags[arm] || {}).map(([k, v]) => `${k} ${v}`).join(', ') || 'clean';
    // Per hand in the take: what went red, whether the arm could take the
    // take's start, and how far that start lies from the mean of the takes
    // that worked that hand - near where a policy run begins.
    const hands = (replay.hands || []).map((arm) => {
      const st = replay.start?.[arm];
      const vs = replay.start_vs_others?.[arm];
      const start = !st ? '' : st.reached ? ' · start reached'
        : ` · start NOT reached (${st.reach_mm} mm / ${st.reach_deg}°)`;
      const apart = vs?.deg == null ? '' : ` · start ${vs.deg}° / ${vs.mm > 0 ? '+' : ''}${Math.round(vs.mm)} mm from the mean`
        + (vs.far ? ' - FAR' : '');
      return `${arm}: ${words(arm)}${start}${apart}`;
    });
    const unplaced = Object.entries(replay.unplaced || {}).map(([arm, why]) => `${arm} not on the desk: ${why}`);
    const unjudged = replay.unjudged ? [`arms not judged: ${replay.unjudged}`] : [];
    summaryEl.textContent = `${replay.frames} frames · ${s.clean} clean · `
      + `${[...hands, ...unplaced, ...unjudged].join(' · ') || 'no hand in this take'}`;
    noteEl.textContent = '';
    drawFlags();
    showFrame(0);
    if (!raf) tick();
  } catch (err) {
    noteEl.textContent = `could not load the replay: ${err}`;
  }
}

function closeReplay() {
  for (const v of Object.values(videos)) { v.pause(); v.removeAttribute('src'); v.load(); }
  if (raf) { cancelAnimationFrame(raf); raf = 0; }
  replay = null; playing = null;
  dialog.close();
}

document.getElementById('replay-close').addEventListener('click', closeReplay);
dialog.addEventListener('cancel', (e) => { e.preventDefault(); closeReplay(); });
playEl.addEventListener('click', () => setPlaying(!playing));
scrub.addEventListener('input', () => {
  if (!replay) return;
  setPlaying(false);
  showFrame(Number(scrub.value));
});
window.addEventListener('resize', () => { if (replay) drawFlags(); });
