// The front page's picture: the two arms on their board, in the ready pose,
// reaching for the pointer when it comes near them.
//
// Built from the same description, meshes, finish and board as the teleop
// view, so the robot on the front page is the robot that page drives. Nothing
// here talks to the simulation: the reach is solved here, on the page, by the
// same solver the teleop view's gripper handle uses.

import * as THREE from 'three';
import { loadURDF } from './urdf.js';
import { hermit } from './hermit.js';
import { HULL, OUTLINE, materialFor, pegboard } from './look.js';
import { Chain, chainAbove } from './ik.js';

// Where the camera stands: a little to the right of straight on, above the hands.
const VIEW_AZ = 0.35;
// A hand notices the pointer inside NEAR of its place and follows it at most
// REACH out of it, easing there over EASE_S; closer in, the jaw opens.
const NEAR = 0.34;         // m
const REACH = 0.15;        // m
const EASE_S = 0.3;
const JAW_OPEN = 0.9;      // rad
const JAW_EASE_S = 0.12;
// Each frame the arm is drawn a little back to the ready pose before it
// reaches, so the joints the reach leaves free stay where the pose has them.
const POSTURE = 0.08;
const DESK_CLEAR = 0.04;   // m: the lowest a hand goes above the desk
// The two Hermits on the desk, each in front of its arm: where, lying which
// way, and the jaw a little open. Lying with the finger tunnel down: the
// right claw's tunnel is on its gripper link's -y, the left's - a mirror - on
// its +y (`up`, the link's y that points up). The jaws point along the desk,
// `jaws` (desk x, y), each the other's mirror.
const HERMITS = [
  { side: 'right', at: [0.58, -0.13], jaws: [0.33, 0.94], up: 1 },
  { side: 'left', at: [0.58, 0.13], jaws: [0.33, -0.94], up: -1 },
];
const HERMIT_JAW = 0.2;                       // rad
// The camera's distance (m from what it looks at): close for a wide picture,
// farther back for a tall one. If the board, the arms, the Hermits and the disc
// under them do not fit there - each within this much of the picture's half-width and
// half-height - the camera puts the board and arms in the middle and stands
// back until they do.
const VIEW_R = { wide: 1.4, tall: 2.0 };
const VIEW_FILL = 0.94;
// The disc on the desk: its colour, and its radius as a share of the farthest part's distance from its centre.
const DISC_COLOR = 0xfde6f0;
const DISC_GROW = 0.8;
// A label given a time to appear draws its line out from the robot, then shows its words.
const LINE_S = 0.4;
const WORDS_S = 0.25;

// The labels over the picture: each pinned to a point on the robot and drawn
// out to its words, re-pinned every frame the picture changes. A label whose
// data-draw-at holds a time on the page clock (ms) appears then, line first;
// "wait" keeps it out of sight; without the attribute it is drawn whole.
// Returns whether a label is still appearing, so the caller keeps drawing.
function callouts(canvas, camera, points) {
  const layer = canvas.parentElement;
  const svg = layer.querySelector('svg.leaders');
  const items = [...layer.querySelectorAll('.callouts [data-at]')];
  const at = new THREE.Vector3();
  return function place(now = performance.now()) {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
    const box = canvas.getBoundingClientRect();
    const lo = 16 - box.left;
    const hi = document.documentElement.clientWidth - 16 - box.left;
    const lines = [];
    let appearing = false;
    for (const item of items) {
      const point = points[item.dataset.at];
      if (!point || !point(at)) { item.hidden = true; continue; }
      at.project(camera);
      const x = (at.x + 1) / 2 * w;
      const y = (1 - at.y) / 2 * h;
      const dx = Number(item.dataset.dx) * w;
      const dy = Number(item.dataset.dy) * h;
      const ex = x + dx;
      const ey = y + dy;
      const west = dx < 0;
      item.hidden = false;
      item.classList.toggle('west', west);
      // The words stay on the page: an edge that would leave it is held a
      // gutter inside, and the line runs on to meet it.
      const width = item.offsetWidth;
      const edge = west ? Math.max(ex - 16, width + lo) : Math.min(ex + 16, hi - width);
      item.style.left = `${edge}px`;
      item.style.top = `${ey}px`;
      const from = item.dataset.drawAt;
      const t = from === undefined ? Infinity : (now - Number(from)) / 1000;
      const line = Number.isNaN(t) ? 0 : THREE.MathUtils.clamp(t / LINE_S, 0, 1);
      const words = Number.isNaN(t) ? 0 : THREE.MathUtils.clamp((t - LINE_S) / WORDS_S, 0, 1);
      if (!Number.isNaN(t) && words < 1) appearing = true;
      item.style.opacity = String(words);
      if (line === 0) continue;
      const d = `M${x.toFixed(1)} ${y.toFixed(1)}L${ex.toFixed(1)} ${ey.toFixed(1)}H${edge.toFixed(1)}`;
      const dash = line < 1 ? ` pathLength="1" stroke-dasharray="1" stroke-dashoffset="${(1 - line).toFixed(3)}"` : '';
      // Drawn twice, paper under ink, so the line reads over the black board too.
      lines.push(`<path class="halo"${dash} d="${d}"/><path${dash} d="${d}"/><circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="3"/>`);
    }
    svg.innerHTML = lines.join('');
    return appearing;
  };
}

// Where the picture's parts are asked for: the studio's server. A static page
// (the documentation) passes the same answers as files.
const STUDIO = {
  physics: '/physics', model: '/model', ready: '/ready', hermit: '/hermit',
  meshes: '/meshes', hermitMeshes: '/hermit/meshes',
};

export async function showRobot(canvas, sources = STUDIO) {
  // The drawing kept between frames, so a screenshot of the page has the robot in it.
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  // The page's own paper, so the robot stands on the page rather than in a box
  // on it. Opaque, not a transparent canvas: premultiplied alpha washed the
  // whole picture out against the page.
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(getComputedStyle(document.body).backgroundColor);
  const world = new THREE.Group();
  world.rotation.x = -Math.PI / 2;   // the description is z-up
  scene.add(world);

  scene.add(new THREE.HemisphereLight(0xfffaf0, 0xcbbfa8, 1.6));
  const key = new THREE.DirectionalLight(0xfff6e8, 2.1);
  key.position.set(1.6, 2.2, 1.2);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  Object.assign(key.shadow.camera, { left: -1, right: 1, top: 1, bottom: -1, near: 0.2, far: 6 });
  key.shadow.bias = -0.0008;
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xffeef6, 0.6);
  fill.position.set(-1.0, 0.8, -1.2);
  scene.add(fill);

  // The floor is only its shadow: the paper is the floor.
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), new THREE.ShadowMaterial({ opacity: 0.14 }));
  floor.receiveShadow = true;
  world.add(floor);

  const [physics, model, ready, hands] = await Promise.all([
    fetch(sources.physics).then((r) => r.json()),
    fetch(sources.model).then((r) => { if (!r.ok) throw new Error('no model'); return r.text(); }),
    fetch(sources.ready).then((r) => r.json()),
    fetch(sources.hermit).then((r) => r.json()).catch(() => ({})),
  ]);

  const slab = physics?.scenery?.pegboard;
  let target = new THREE.Vector3(0.2, 0.28, 0);
  if (slab) {
    world.add(pegboard(slab, renderer.capabilities.getMaxAnisotropy()));
    // Look at the space in front of the board, where the hands are.
    target = new THREE.Vector3(slab.pos[0] + 0.2, slab.pos[2] + 0.02, -slab.pos[1]);
  }
  const desk = physics?.scenery?.desk;
  if (desk) floor.position.z = desk.pos[2];

  const robot = await loadURDF(model, sources.meshes, { materialFor, outlineMaterial: OUTLINE, hullMaterial: HULL });
  robot.root.traverse((o) => { if (o.isMesh && o.material !== HULL) o.castShadow = true; });
  robot.setPose(ready);
  world.add(robot.root);
  world.updateMatrixWorld(true);

  // The Hermits, lying on the desk in front of the arms: each set down on its
  // lowest point.
  const lying = [];
  async function layHermit({ side, at, jaws, up }) {
    const built = await hermit(robot, side, hands[side], sources.hermitMeshes).catch(() => null);
    if (!built) return;
    // A little open, whichever way this hand's jaw opens: the left one's is the mirror.
    const band = hands[side].jaw_band;
    built.setJaw(band ? HERMIT_JAW * Math.sign(band[1] - band[0]) : HERMIT_JAW);
    const hand = built.hand;
    // Columns: where the link's x, y and z go on the desk. Its y up by `up`,
    // its -z (the jaws' way, the claw's +z) along `jaws`.
    const [hx, hy] = jaws;
    hand.position.set(0, 0, 0);
    hand.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(
      new THREE.Vector3(up * hy, -up * hx, 0), new THREE.Vector3(0, 0, up), new THREE.Vector3(-hx, -hy, 0)));
    world.add(hand);
    world.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(hand);
    const low = new THREE.Vector3(0, box.min.y, 0);
    world.worldToLocal(low);
    hand.position.set(at[0], at[1], (desk ? desk.pos[2] : 0) - low.z);
    world.updateMatrixWorld(true);
    hand.userData.lyingSide = side;
    lying.push(hand);
  }
  await Promise.all(HERMITS.filter(({ side }) => hands[side]).map(layHermit));
  // Everything the picture has to hold, as the corners of each part's box: all
  // but the floor, which is only its shadow. The board and the arms, the heavy
  // part of the picture, apart: they are what is put in the middle.
  const corners = [];
  const heavy = [];
  const part = new THREE.Box3();
  const held = new Set();
  for (const hand of lying) hand.traverse((o) => held.add(o));
  world.traverse((o) => {
    if (!o.isMesh || o === floor || o.material === HULL) return;
    part.setFromObject(o);
    for (let i = 0; i < 8; i += 1) {
      const c = new THREE.Vector3(i & 1 ? part.max.x : part.min.x, i & 2 ? part.max.y : part.min.y, i & 4 ? part.max.z : part.min.z);
      corners.push(c);
      if (!held.has(o)) heavy.push(c);
    }
  });

  // A pale pink disc on the desk under the board, the arms and the Hermits,
  // just below the floor so their shadows fall on it.
  // Centred from the board out to the Hermits, and across on the board.
  const box = (points) => {
    const local = points.map((c) => world.worldToLocal(c.clone()));
    return [local.reduce((m, c) => m.min(c), new THREE.Vector3(Infinity, Infinity, Infinity)),
            local.reduce((m, c) => m.max(c), new THREE.Vector3(-Infinity, -Infinity, -Infinity))];
  };
  const [lo, hi] = box(corners);
  const [blo, bhi] = box(heavy);
  const ground = corners.map((c) => world.worldToLocal(c.clone()));
  const centre = new THREE.Vector2((lo.x + hi.x) / 2, (blo.y + bhi.y) / 2);
  const reach = Math.max(...ground.map((c) => centre.distanceTo(new THREE.Vector2(c.x, c.y))));
  const disc = new THREE.Mesh(new THREE.CircleGeometry(reach * DISC_GROW, 96), new THREE.MeshBasicMaterial({ color: DISC_COLOR }));
  disc.position.set(centre.x, centre.y, floor.position.z - 0.001);
  world.add(disc);
  world.updateMatrixWorld(true);
  // Its rim is in the picture too.
  for (let i = 0; i < 24; i += 1) {
    const a = (i / 24) * Math.PI * 2;
    corners.push(world.localToWorld(new THREE.Vector3(
      centre.x + Math.cos(a) * reach * DISC_GROW, centre.y + Math.sin(a) * reach * DISC_GROW, disc.position.z)));
  }

  // One reach per arm, from the point between its fingers. The pointer is
  // read where its ray crosses the plane through the hands, square to the
  // board: the hands move along the board's face, the way the arms can.
  const arms = ['left', 'right'].map((side) => {
    const tip = robot.links.get(`${side}_gripper_frame_link`);
    if (!tip) return null;
    const place = tip.getWorldPosition(new THREE.Vector3());
    // The jaw is shut at zero and opens toward its limit farther from zero: positive
    // on the left, negative on the right, whose servo is turned round.
    const band = robot.joints.get(`${side}_gripper`);
    const opens = band && Math.abs(band.lower) > Math.abs(band.upper) ? -1 : 1;
    return { side, chain: new Chain(robot, chainAbove(robot, tip), tip), place, opens,
             aim: place.clone(), goal: place.clone(), jaw: 0, jawGoal: 0 };
  }).filter(Boolean);
  const facing = new THREE.Vector3(1, 0, 0).applyQuaternion(world.quaternion);
  const middle = arms.reduce((sum, a) => sum.add(a.place), new THREE.Vector3()).divideScalar(arms.length || 1);
  const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(facing, middle);
  const lowest = floor.getWorldPosition(new THREE.Vector3()).y + DESK_CLEAR;

  const camera = new THREE.PerspectiveCamera(30, 1, 0.01, 50);
  let dirty = true;
  function size() {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    const at = new THREE.Vector3();
    const look = new THREE.Vector3();
    const right = new THREE.Vector3();
    const up = new THREE.Vector3();
    const extent = (points = corners) => {
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (const c of points) {
        at.copy(c).project(camera);
        x0 = Math.min(x0, at.x); x1 = Math.max(x1, at.x); y0 = Math.min(y0, at.y); y1 = Math.max(y1, at.y);
      }
      return { x0, x1, y0, y1 };
    };
    const fits = ({ x0, x1, y0, y1 }) => Math.max(-x0, x1) <= VIEW_FILL && Math.max(-y0, y1) <= VIEW_FILL;
    const near = camera.aspect < 1 ? VIEW_R.tall : VIEW_R.wide;
    camera.position.set(target.x + near * Math.cos(VIEW_AZ), target.y + 0.3, target.z + near * Math.sin(VIEW_AZ));
    camera.lookAt(target);
    camera.updateMatrixWorld();
    if (!fits(extent())) for (let r = near; r < near * 3; r += 0.05) {
      camera.position.set(target.x + r * Math.cos(VIEW_AZ), target.y + 0.3, target.z + r * Math.sin(VIEW_AZ));
      camera.lookAt(target);
      camera.updateMatrixWorld();
      const e = extent();
      const mid = extent(heavy);
      const halfH = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * camera.position.distanceTo(target);
      right.setFromMatrixColumn(camera.matrixWorld, 0).multiplyScalar(((mid.x0 + mid.x1) / 2) * halfH * camera.aspect);
      up.setFromMatrixColumn(camera.matrixWorld, 1).multiplyScalar(((e.y0 + e.y1) / 2) * halfH);
      camera.position.add(right).add(up);
      look.copy(target).add(right).add(up);
      camera.lookAt(look);
      camera.updateMatrixWorld();
      if (fits(extent())) break;
    }
    dirty = true;
  }
  new ResizeObserver(size).observe(canvas);
  size();

  // The pointer, anywhere on the page: over the words laid across the
  // picture as much as over the picture itself. Off it, the hands go home.
  const caster = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  const hit = new THREE.Vector3();
  let pressed = false;
  function aimAt(event) {
    const r = canvas.getBoundingClientRect();
    const inside = event && event.clientX >= r.left && event.clientX <= r.right
      && event.clientY >= r.top && event.clientY <= r.bottom;
    let point = null;
    if (inside) {
      ndc.set(((event.clientX - r.left) / r.width) * 2 - 1, -((event.clientY - r.top) / r.height) * 2 + 1);
      caster.setFromCamera(ndc, camera);
      point = caster.ray.intersectPlane(plane, hit);
    }
    for (const arm of arms) {
      const away = point ? point.clone().sub(arm.place) : null;
      // How much the hand cares: fully inside 60 % of NEAR, not at all past it.
      const care = away ? 1 - THREE.MathUtils.smoothstep(away.length(), NEAR * 0.6, NEAR) : 0;
      arm.goal.copy(arm.place);
      if (care > 0) arm.goal.addScaledVector(away.clampLength(0, REACH), care);
      arm.goal.y = Math.max(arm.goal.y, lowest);
      arm.jawGoal = pressed ? 0 : arm.opens * JAW_OPEN * care;
    }
  }
  addEventListener('pointermove', aimAt, { passive: true });
  // Out of the window altogether: a mouseout with nowhere it went to.
  document.addEventListener('mouseout', (event) => { if (!event.relatedTarget) aimAt(null); });
  // Pressing closes the jaws on whatever they were reaching for.
  canvas.addEventListener('pointerdown', (event) => { pressed = true; aimAt(event); });
  addEventListener('pointerup', (event) => { pressed = false; aimAt(event); });

  // What each label is pinned to.
  const onBoard = slab && new THREE.Vector3(slab.pos[0] + slab.half[0], slab.half[1] * 0.6, slab.pos[2] + slab.half[2] * 0.75);
  const points = {
    board: onBoard && ((v) => world.localToWorld(v.copy(onBoard))),
    joints: robot.joints.has('left_wrist_yaw') && ((v) => robot.joints.get('left_wrist_yaw').group.getWorldPosition(v)),
    pink: robot.links.has('right_upper_arm_link') && ((v) => robot.links.get('right_upper_arm_link').getWorldPosition(v)),
    claw: robot.links.has('right_jaw_link') && ((v) => robot.links.get('right_jaw_link').getWorldPosition(v)),
    // The right hand's Hermit, in front of the right arm; none while there is none to point at.
    hermit: (v) => {
      const held = lying.find((h) => h.userData.lyingSide === 'right');
      return held ? new THREE.Box3().setFromObject(held).getCenter(v) : null;
    },
  };
  const label = callouts(canvas, camera, points);

  // Exposed for the console, as the teleop view exposes its own.
  window.pinkrabHero = { renderer, scene, camera, world, robot, target, arms, plane, lying, label };
  let last = performance.now();
  let appearing = false;
  function frame(now) {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    const ease = 1 - Math.exp(-dt / EASE_S);
    const jawEase = 1 - Math.exp(-dt / JAW_EASE_S);
    for (const arm of arms) {
      const still = arm.aim.distanceTo(arm.goal) < 1e-4 && arm.aim.distanceTo(arm.chain.tip()) < 1e-3
        && Math.abs(arm.jaw - arm.jawGoal) < 1e-3;
      if (still) continue;
      arm.aim.lerp(arm.goal, ease);
      for (const name of arm.chain.names) {
        const joint = robot.joints.get(name);
        robot.setJoint(name, joint.value + ((ready[name] ?? joint.value) - joint.value) * POSTURE);
      }
      arm.chain.solve(arm.aim);
      arm.jaw += (arm.jawGoal - arm.jaw) * jawEase;
      robot.setJoint(`${arm.side}_gripper`, arm.jaw);
      dirty = true;
    }
    if (dirty) renderer.render(scene, camera);
    if (dirty || appearing) appearing = label(now);
    dirty = false;
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  // For a page that sets its labels' times after the robot is up: draw again.
  return { redraw() { dirty = true; } };
}
