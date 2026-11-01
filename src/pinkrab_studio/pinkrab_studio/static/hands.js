// The hands view: the operator's side of the rig, drawn in the robot's frame.
//
// The operator sits where the robot reckons them - their chest on the board's
// face, midway between the two arms - with the desk solid against it, where
// it is reckoned (desk_solid.py), and each handheld floats where the top
// camera last placed its solid against the solid, carried
// `desk_reckoned_back_m` further out as a take is placed, held by the
// operator's arm - where the shelf judges a take and a policy run starts the
// arms. Teleop follows the hands' motion, not their place, so a hand followed
// need not be drawn over its arm's tool; a hand the camera has not placed
// lately is drawn on the arm's tool, where the arm holds it. The frame and the viewpoint are the robot view's: orbiting either
// view turns both.

import * as THREE from 'three';
import { OrbitControls } from './OrbitControls.js';
import { STLLoader } from './STLLoader.js';
import { watch } from './live.js';
import { drawPart } from './urdf.js';
import { HULL, OUTLINE } from './look.js';
import { PAPER, hermit } from './hermit.js';
import { addMarkers } from './markers.js';

const ARMS = ['left', 'right'];
// The solid's own axes, drawn on the desk from under it.
const AXIS_M = 0.09;
// A hand not placed for this long is drawn on its arm's tool instead: a
// moment's miss holds it where it was, rather than throwing it at the arm.
const STALE_MS = 2000;
// The operator, seated at the desk where the board stands: a seated adult's
// rough proportions, metres. Only drawn - nothing is solved against them.
const BODY = {
  chestDepth: 0.22, chestWidth: 0.34,
  hips: -0.06, shoulderZ: 0.33,          // above the desk
  shoulderY: 0.19, shoulderBack: 0.08,   // across, and behind the chest's front
  upperArm: 0.29, forearm: 0.27, neck: 0.15, head: 0.095,
};
// Elbows hang down and out, a little back: the way a held arm bends.
const ELBOW_POLE = [-0.2, 0.4, -1.0];
const SKIN = new THREE.MeshStandardMaterial({
  color: 0xe7c2a9, roughness: 0.8, transparent: true, opacity: 0.3, depthWrite: false,
});
// The top camera on its head strap, metres: a band round the forehead and one
// over the crown, the camera module on the band's front, its lens aimed down
// at the desk `aimAhead` in front of the chest, where the hands work.
const STRAP = { band: 0.026, thick: 0.004, lift: 0.02, plate: 0.03, aimAhead: 0.25 };
const MODULE = { board: 0.038, depth: 0.016, lensR: 0.008, lensL: 0.018 };
const STRAP_MAT = new THREE.MeshStandardMaterial({ color: 0x34323d, roughness: 0.9, side: THREE.DoubleSide });
const CAMERA_MAT = new THREE.MeshStandardMaterial({ color: 0x1d1b2e, roughness: 0.5, metalness: 0.2 });
const LENS_MAT = new THREE.MeshStandardMaterial({ color: 0x0c0c10, roughness: 0.25, metalness: 0.4 });

// The solid with its paper and markers, in its own frame (origin at its
// centre, z up), metres; its axes `drop` below that, on the desk.
async function deskSolid(spec, drop) {
  const geometry = await new STLLoader().loadAsync('/desk_solid.stl');
  const solid = new THREE.Group();
  drawPart(solid, geometry, PAPER, { outlineMaterial: OUTLINE, hullMaterial: HULL });
  addMarkers(solid, spec.markers);
  // +X and +Y, along the desk from under the solid: which way it was turned.
  const axes = new THREE.Group();
  axes.position.z = -drop + 0.002;
  for (const [dir, colour] of [[new THREE.Vector3(1, 0, 0), 0xd64545], [new THREE.Vector3(0, 1, 0), 0x2f9e5a]]) {
    axes.add(new THREE.ArrowHelper(dir, new THREE.Vector3(), AXIS_M, colour, 0.018, 0.012));
  }
  solid.add(axes);
  solid.traverse((o) => { if (o.isMesh && o.material !== HULL) o.castShadow = true; });
  return solid;
}

// The operator's torso and head, the top camera strapped to it: the chest's
// front at `chestX`, the board's face, centred across the arms.
function operator(chestX, acrossY) {
  const body = new THREE.Group();
  const r = BODY.chestDepth / 2;
  const tall = BODY.shoulderZ + 0.03 - BODY.hips;
  const torso = new THREE.Mesh(new THREE.CapsuleGeometry(r, Math.max(0.01, tall - 2 * r), 6, 24), SKIN);
  torso.rotation.x = Math.PI / 2;                // the capsule's length along z
  torso.scale.set(1, 1, BODY.chestWidth / BODY.chestDepth);
  torso.position.set(chestX - r, acrossY, BODY.hips + tall / 2);
  const head = new THREE.Mesh(new THREE.SphereGeometry(BODY.head, 24, 16), SKIN);
  head.position.set(chestX - r, acrossY, BODY.shoulderZ + BODY.neck);
  body.add(torso, head);
  body.traverse((o) => { o.renderOrder = 4; });
  body.add(topCamera(head.position, chestX + STRAP.aimAhead));
  return body;
}

// The head strap and the top camera on it, about the head's centre; the lens
// looks at the desk at `aimX`.
function topCamera(centre, aimX) {
  const rig = new THREE.Group();
  rig.position.copy(centre);
  const R = BODY.head;
  const bandR = Math.sqrt(R * R - STRAP.lift * STRAP.lift) + STRAP.thick;
  const band = new THREE.Mesh(new THREE.CylinderGeometry(bandR, bandR, STRAP.band, 40, 1, true), STRAP_MAT);
  band.rotation.x = Math.PI / 2;                 // the cylinder's axis along z
  band.position.z = STRAP.lift;
  // Over the crown, ear to ear, from the band up: a ring's arc turned into
  // the y-z plane.
  const rise = Math.asin(STRAP.lift / R);
  const arc = new THREE.Mesh(new THREE.TorusGeometry(R + STRAP.thick, STRAP.thick, 8, 32, Math.PI - 2 * rise), STRAP_MAT);
  arc.rotation.z = rise;
  const crown = new THREE.Group();
  crown.add(arc);
  crown.rotation.set(Math.PI / 2, 0, Math.PI / 2, 'ZYX');
  const plate = new THREE.Mesh(new THREE.BoxGeometry(0.006, STRAP.plate, STRAP.plate), CAMERA_MAT);
  plate.position.set(bandR + 0.003, 0, STRAP.lift);
  // The module: its boards square to the lens, the lens along +x before it is
  // pitched down onto the desk.
  const camera = new THREE.Group();
  const front = bandR + 0.006 + MODULE.depth / 2;
  camera.position.set(front, 0, STRAP.lift);
  const pitch = Math.atan2(centre.z + STRAP.lift, aimX - (centre.x + front));
  camera.rotation.y = pitch;
  const box = new THREE.Mesh(new THREE.BoxGeometry(MODULE.depth, MODULE.board, MODULE.board), CAMERA_MAT);
  const lens = new THREE.Mesh(new THREE.CylinderGeometry(MODULE.lensR, MODULE.lensR, MODULE.lensL, 24), LENS_MAT);
  lens.rotation.z = -Math.PI / 2;                // the cylinder's axis along +x
  lens.position.x = (MODULE.depth + MODULE.lensL) / 2;
  camera.add(box, lens);
  rig.add(band, crown, plate, camera);
  rig.traverse((o) => { if (o.isMesh) o.castShadow = true; });
  return rig;
}

// One of the operator's arms, shoulder to wrist: two segments and three
// joints, the elbow solved in the plane the pole leans it into.
function operatorArm() {
  const group = new THREE.Group();
  const bone = (radius) => new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, 1, 16), SKIN);
  const joint = (radius) => new THREE.Mesh(new THREE.SphereGeometry(radius, 16, 12), SKIN);
  const parts = { upper: bone(0.042), fore: bone(0.034), shoulder: joint(0.046), elbow: joint(0.04), wrist: joint(0.032) };
  group.add(...Object.values(parts));
  group.traverse((o) => { o.renderOrder = 4; });
  const up = new THREE.Vector3(0, 1, 0);
  const lay = (mesh, from, to) => {
    const along = to.clone().sub(from);
    mesh.position.copy(from).addScaledVector(along, 0.5);
    mesh.scale.set(1, Math.max(1e-4, along.length()), 1);
    mesh.quaternion.setFromUnitVectors(up, along.normalize());
  };
  const elbow = new THREE.Vector3();
  const pose = (shoulder, wrist, side) => {
    const l1 = BODY.upperArm;
    const l2 = BODY.forearm;
    const span = wrist.clone().sub(shoulder);
    const d = Math.min(Math.max(span.length(), Math.abs(l1 - l2) + 1e-3), l1 + l2 - 1e-3);
    const u = span.normalize();
    const pole = new THREE.Vector3(ELBOW_POLE[0], side * ELBOW_POLE[1], ELBOW_POLE[2]);
    const v = pole.addScaledVector(u, -pole.dot(u)).normalize();
    const a = (l1 * l1 - l2 * l2 + d * d) / (2 * d);
    elbow.copy(shoulder).addScaledVector(u, a).addScaledVector(v, Math.sqrt(Math.max(0, l1 * l1 - a * a)));
    lay(parts.upper, shoulder, elbow);
    lay(parts.fore, elbow, wrist);
    parts.shoulder.position.copy(shoulder);
    parts.elbow.position.copy(elbow);
    parts.wrist.position.copy(wrist);
  };
  return { group, pose };
}

/**
 * @param canvas     the hands view's canvas
 * @param lineEl     the pane's status line
 * @param robot      the robot view's arms: the handhelds' claws, and the tools
 *                   an unplaced hand is drawn on
 * @param robotWorld the robot view's z-up group, the frame the arms stand in
 * @param scenery    the scene's desk, as the simulation compiled it
 * @param view       the robot view's { camera, controls, background }
 */
export async function showHands({ canvas, lineEl, robot, robotWorld, scenery, view }) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const scene = new THREE.Scene();
  scene.background = view.background.clone();
  const world = new THREE.Group();
  world.rotation.x = -Math.PI / 2;   // the description is z-up
  scene.add(world);

  scene.add(new THREE.HemisphereLight(0xfffaf0, 0xcbbfa8, 1.5));
  const key = new THREE.DirectionalLight(0xfff6e8, 1.9);
  key.position.set(1.35, 2.1, 0.95);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  Object.assign(key.shadow.camera, { left: -1, right: 1, top: 1, bottom: -1, near: 0.2, far: 6 });
  key.shadow.bias = -0.0008;
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xeef2ff, 0.55);
  fill.position.set(-0.9, 0.7, -1.3);
  scene.add(fill);

  const deskZ = scenery?.desk?.pos[2] ?? 0;
  const desk = new THREE.Mesh(new THREE.PlaneGeometry(2.4, 1.6),
    new THREE.MeshStandardMaterial({ color: 0xefe9dd, roughness: 0.95 }));
  desk.receiveShadow = true;
  desk.position.z = deskZ;
  world.add(desk);
  const grid = new THREE.GridHelper(2.4, 24, 0xc9bda6, 0xdbd2c1);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = deskZ + 0.001;
  world.add(grid);

  // One viewpoint for both views: dragging here turns the robot view's own
  // camera, about its own target, and this view's camera copies that one
  // every frame - with its own aspect, its canvas being its own shape. Two
  // cameras each steering the other drift: each one's damping is fed back.
  const camera = new THREE.PerspectiveCamera(view.camera.fov, 1, view.camera.near, view.camera.far);
  const controls = new OrbitControls(view.camera, canvas);
  controls.target = view.controls.target;
  controls.enableDamping = true;

  const [spec, parts] = await Promise.all([
    fetch('/desk_solid').then((r) => r.json()),
    fetch('/hermit').then((r) => r.json()),
  ]);
  if (spec.error) throw new Error(spec.error);
  // The solid is drawn where it is reckoned; a hand against it is carried
  // into the world by world_T_desk, further out, as a take is placed (the
  // shelf's replay, a policy run's start).
  const deskInWorld = new THREE.Matrix4().set(...spec.world_T_desk.flat());
  const solid = await deskSolid(spec, spec.solid_in_world[2][3] - deskZ);
  solid.matrixAutoUpdate = false;
  solid.matrix.set(...spec.solid_in_world.flat());
  world.add(solid);
  // The operator where the robot reckons them, the solid against their chest.
  const chestX = spec.chest_x;
  const acrossY = spec.solid_in_world[1][3];
  world.add(operator(chestX, acrossY));

  // Each handheld, placed by its solid: the link's pose is the solid's, less
  // where the solid hangs in the link.
  const hands = {};
  for (const arm of ARMS) {
    if (!parts[arm]) continue;
    const built = await hermit(robot, arm, parts[arm]);
    built.hand.position.set(0, 0, 0);
    built.hand.quaternion.identity();
    built.hand.updateMatrixWorld(true);
    const solidInverse = built.solid.matrixWorld.clone().invert();
    // Where the operator's wrist meets the handheld: the back of its body,
    // away from the jaws (the link's -z), across its middle.
    const shell = built.body?.children.find((c) => c.isMesh) ?? built.hand;   // the body alone, not its solid
    const box = new THREE.Box3().setFromObject(shell);
    const grip = new THREE.Vector3((box.min.x + box.max.x) / 2, (box.min.y + box.max.y) / 2, box.max.z);
    const side = arm === 'left' ? 1 : -1;
    const shoulder = new THREE.Vector3(chestX - BODY.shoulderBack, acrossY + side * BODY.shoulderY,
      deskZ + BODY.shoulderZ);
    const limb = operatorArm();
    world.add(limb.group);
    built.hand.matrixAutoUpdate = false;
    world.add(built.hand);
    hands[arm] = { ...built, solidInverse, band: parts[arm].jaw_band, placedAt: -Infinity, grip, shoulder, side, limb };
  }

  const OFF = 'top camera off - the hands are drawn on the arms';
  lineEl.textContent = spec.note || OFF;
  // Each hand's pose low-passed as teleop low-passes it: the solved pose
  // shakes by a millimetre and half a degree from frame to frame.
  const tau = (spec.smoothing_s ?? 0) * 1000;
  const solidInWorld = new THREE.Matrix4();
  const wanted = { position: new THREE.Vector3(), quaternion: new THREE.Quaternion(), scale: new THREE.Vector3() };
  const smooth = (hand, matrix, now) => {
    matrix.decompose(wanted.position, wanted.quaternion, wanted.scale);
    const fresh = now - hand.placedAt > STALE_MS || !hand.held;
    const share = fresh || tau <= 0 ? 1 : (now - hand.placedAt) / (tau + now - hand.placedAt);
    hand.held ??= { position: new THREE.Vector3(), quaternion: new THREE.Quaternion() };
    hand.held.position.lerp(wanted.position, share);
    hand.held.quaternion.slerp(wanted.quaternion, share);
    if (fresh) { hand.held.position.copy(wanted.position); hand.held.quaternion.copy(wanted.quaternion); }
    hand.hand.matrix.compose(hand.held.position, hand.held.quaternion, wanted.scale.set(1, 1, 1));
  };
  const place = (live) => {
    if (!live || !live.solids) {
      lineEl.textContent = live ? 'the live view is not answering' : OFF;
      return;
    }
    const now = performance.now();
    const words = [];
    for (const [arm, hand] of Object.entries(hands)) {
      const pose = live.solids[`handheld_${arm}`];
      if (pose) {
        solidInWorld.set(...pose.flat());
        smooth(hand, solidInWorld.premultiply(deskInWorld).multiply(hand.solidInverse), now);
        hand.hand.matrixWorldNeedsUpdate = true;
        hand.placedAt = now;
        const opening = live.jaw?.[arm];
        if (opening != null && hand.band) hand.setJaw(hand.band[0] + opening * (hand.band[1] - hand.band[0]));
      }
      words.push(`${arm} ${pose ? 'seen' : 'on the arm'}`);
    }
    lineEl.textContent = words.join(' · ');
  };
  watch('hands', place);
  // Exposed, as the robot view is: a recorded take's poses can be laid here
  // from the console, a frame at a time, without the top camera.
  window.pinkrabHands = { place, hands, solid, deskInWorld, camera, controls };

  // A hand not placed lately sits on its arm's tool: the gripper link as the
  // robot view has it, carried into this view's frame, its jaw as the arm's.
  const toHere = new THREE.Matrix4();
  function onTheArms(now) {
    robotWorld.updateMatrixWorld(true);
    toHere.copy(robotWorld.matrixWorld).invert();
    for (const [arm, hand] of Object.entries(hands)) {
      if (now - hand.placedAt < STALE_MS) continue;
      hand.held = null;
      hand.hand.matrix.multiplyMatrices(toHere, robot.links.get(`${arm}_gripper_link`).matrixWorld);
      hand.hand.matrixWorldNeedsUpdate = true;
      const jaw = robot.joints.get(`${arm}_gripper`);
      if (jaw) hand.setJaw(jaw.value);
    }
  }

  // Each of the operator's arms reaching its handheld, wherever that is drawn.
  const wrist = new THREE.Vector3();
  function reach() {
    for (const hand of Object.values(hands)) {
      wrist.copy(hand.grip).applyMatrix4(hand.hand.matrix);
      hand.limb.pose(hand.shoulder, wrist, hand.side);
    }
  }

  function tick(now) {
    const { clientWidth: w, clientHeight: h } = canvas;
    if (w && h && (canvas.width !== Math.round(w * renderer.getPixelRatio()) || camera.aspect !== w / h)) {
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    onTheArms(now);
    reach();
    controls.update();
    camera.position.copy(view.camera.position);
    camera.quaternion.copy(view.camera.quaternion);
    renderer.render(scene, camera);
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}
