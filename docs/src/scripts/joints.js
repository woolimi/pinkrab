// The joint ranges in the introduction: one arm in the middle of its board,
// reaching out, moving the joint picked beside it from one limit to the other
// over a fan that marks the range.

import * as THREE from 'three';
import { fitView, orbit } from './canvases.js';
import { SIDE, solids, stage } from './stage.js';
import { pick } from './lang.js';

// Where the camera looks from, in the description's z-up frame: in front of
// the board, to the arm's outer side and a little above it. Each joint turns
// this toward its fan's face, so no fan is seen edge on.
const VIEW = new THREE.Vector3(1.0, 0.75, 0.55);
const TOWARD_FAN = 0.9;
const FRONT = 0.3;          // the least the camera stays in front of the board
const ABOVE = 0.15;         // and above the hands
const FOV = 30;
const MARGIN = 1.1;
// The pose the other joints hold, in degrees on the left arm: the ready pose
// with the arm reaching out instead of folded, so the parts a joint turns are
// clear of the rest and the fan under them.
const REACHING = { shoulder_pan: 0, shoulder_lift: -175, elbow_flex: 120 };
// The joint moves at this speed on average, slowing into each limit.
const SWEEP_DPS = 80;
// Picking another joint: the last one goes back to that pose, and the
// camera moves to frame the new one, both over these.
const RETURN_S = 0.35;
const FRAME_S = 0.7;
// The fan: a little shorter than the part it turns, so the part's end is seen
// passing over its rim.
const FAN_REACH = 0.75;
const FAN_MIN = 0.035;     // m
const FAN_MAX = 0.3;       // m
const FAN_FILL = new THREE.MeshBasicMaterial({
  color: 0xe2488e, transparent: true, opacity: 0.16, side: THREE.DoubleSide, depthWrite: false,
});
const FAN_RIM = new THREE.LineBasicMaterial({ color: 0xb8306f });
const FAN_ZERO = new THREE.LineDashedMaterial({ color: 0x765c6a, dashSize: 0.006, gapSize: 0.005 });
const NEEDLE = new THREE.LineBasicMaterial({ color: 0x5e1638 });
const NEEDLE_TIP = new THREE.MeshBasicMaterial({ color: 0x5e1638 });

const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);
const deg = (rad) => (rad * 180) / Math.PI;

export async function showJoints(figure, sources) {
  const canvas = figure.querySelector('canvas');
  const chips = [...figure.querySelectorAll('[data-joint]')];
  const slider = figure.querySelector('input[type="range"]');
  const play = figure.querySelector('.play');
  const angle = figure.querySelector('.angle');

  const { robot, world, board, ready, addView } = await stage(sources);
  const rest = { ...ready };
  for (const [key, d] of Object.entries(REACHING)) rest[`${SIDE}_${key}`] = THREE.MathUtils.degToRad(d);
  robot.setPose(rest);
  world.updateMatrixWorld(true);

  // One fan per joint, in its anchor: the frame the joint turns its child
  // about the z of. Pointing where the child points, so the part sweeps the fan.
  const corner = new THREE.Vector3();
  function fanFor(name) {
    const joint = robot.joints.get(name);
    const toAnchor = joint.anchor.matrixWorld.clone().invert();
    let sx = 0, sy = 0, far = 0;
    for (const mesh of solids(robot.links.get(joint.child))) {
      const box = mesh.geometry.boundingBox ?? (mesh.geometry.computeBoundingBox(), mesh.geometry.boundingBox);
      const toHere = toAnchor.clone().multiply(mesh.matrixWorld);
      for (let i = 0; i < 8; i++) {
        corner.set(i & 1 ? box.max.x : box.min.x, i & 2 ? box.max.y : box.min.y, i & 4 ? box.max.z : box.min.z)
          .applyMatrix4(toHere);
        sx += corner.x;
        sy += corner.y;
        far = Math.max(far, Math.hypot(corner.x, corner.y));
      }
    }
    const turn = Math.sign(joint.axis.z) || 1;
    const zero = Math.atan2(sy, sx) - turn * joint.value;   // where the child points at zero
    const r = THREE.MathUtils.clamp(far * FAN_REACH, FAN_MIN, FAN_MAX);
    const from = zero + Math.min(turn * joint.lower, turn * joint.upper);
    const span = joint.upper - joint.lower;

    const group = new THREE.Group();
    group.add(new THREE.Mesh(new THREE.CircleGeometry(r, 128, from, span), FAN_FILL));
    const rim = [new THREE.Vector3()];
    for (let i = 0; i <= 128; i++) {
      const a = from + (span * i) / 128;
      rim.push(new THREE.Vector3(r * Math.cos(a), r * Math.sin(a), 0));
    }
    rim.push(new THREE.Vector3());
    group.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(rim), FAN_RIM));
    const zeroLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(), new THREE.Vector3(r * Math.cos(zero), r * Math.sin(zero), 0)]), FAN_ZERO);
    zeroLine.computeLineDistances();
    group.add(zeroLine);
    const needle = new THREE.Group();
    needle.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(), new THREE.Vector3(r * 1.06, 0, 0)]), NEEDLE));
    const tip = new THREE.Mesh(new THREE.SphereGeometry(r * 0.035, 12, 8), NEEDLE_TIP);
    tip.position.x = r * 1.06;
    needle.add(tip);
    group.add(needle);
    group.visible = false;
    group.renderOrder = 2;
    joint.anchor.add(group);
    group.updateMatrixWorld(true);
    // The fan's face, toward the side the camera is on.
    const face = new THREE.Vector3(0, 0, 1).transformDirection(joint.anchor.matrixWorld);
    return { group, needle, zero, turn, face };
  }

  // What to frame for a joint: the corners of every part it carries, at every
  // angle it reaches, and the fan's rim.
  function framing(name) {
    const joint = robot.joints.get(name);
    const keep = joint.value;
    const points = [];
    const parts = solids(robot.links.get(joint.child));
    for (let i = 0; i <= 12; i++) {
      robot.setJoint(name, joint.lower + ((joint.upper - joint.lower) * i) / 12);
      world.updateMatrixWorld(true);
      for (const mesh of parts) {
        const box = mesh.geometry.boundingBox ?? (mesh.geometry.computeBoundingBox(), mesh.geometry.boundingBox);
        for (let c = 0; c < 8; c++) {
          points.push(new THREE.Vector3(c & 1 ? box.max.x : box.min.x, c & 2 ? box.max.y : box.min.y,
            c & 4 ? box.max.z : box.min.z).applyMatrix4(mesh.matrixWorld));
        }
      }
    }
    robot.setJoint(name, keep);
    world.updateMatrixWorld(true);
    fans.get(name).group.traverse((o) => {
      if (!o.isLine) return;
      const at = o.geometry.attributes.position;
      for (let i = 0; i < at.count; i++) points.push(new THREE.Vector3().fromBufferAttribute(at, i).applyMatrix4(o.matrixWorld));
    });
    return points;
  }

  const names = chips.map((chip) => chip.dataset.joint);
  const fans = new Map(names.map((name) => [name, fanFor(name)]));
  const frames = new Map(names.map((name) => [name, framing(name)]));

  const camera = new THREE.PerspectiveCamera(FOV, 1, 0.01, 20);
  const view = VIEW.clone().normalize().applyQuaternion(world.quaternion);
  // Each joint's camera: the overall view turned toward the fan's face, kept
  // in front of the board and above the arm.
  function lookFor(name) {
    const face = fans.get(name).face.clone();
    if (face.dot(view) < 0) face.negate();
    const d = view.clone().addScaledVector(face, TOWARD_FAN).normalize();
    d.x = Math.max(d.x, FRONT);
    d.y = Math.max(d.y, ABOVE);
    return d.normalize();
  }

  const fit = (points, direction) => fitView(points, direction, camera, MARGIN);

  // The sweep: from the lower limit to the upper and back, a cosine so it
  // slows into each end. `phase` is in sweeps, 0..1.
  const pose = { ...rest };
  let selected = null;
  let phase = 0;
  let period = 1;
  let back = null;   // the joint going back to the reaching pose
  let shot = null;   // the camera on its way to the next joint
  const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let playing = !still;

  const limits = (name) => robot.joints.get(name);
  const sweep = ({ lower, upper }, p) => lower + ((upper - lower) * (1 - Math.cos(2 * Math.PI * p))) / 2;
  function phaseAt({ lower, upper }, value) {
    const u = THREE.MathUtils.clamp((value - lower) / (upper - lower), 0, 1);
    return Math.acos(1 - 2 * u) / (2 * Math.PI);
  }

  const { controls, moving } = orbit(camera, canvas, () => picture.redraw());

  function step(now, dt) {
    if (playing) {
      phase = (phase + dt / period) % 1;
      pose[selected] = sweep(limits(selected), phase);
    }
    if (back) {
      const t = Math.min((now - back.start) / 1000 / RETURN_S, 1);
      pose[back.name] = back.from + (back.to - back.from) * ease(t);
      if (t >= 1) back = null;
    }
    if (shot) {
      const t = Math.min((now - shot.start) / 1000 / FRAME_S, 1);
      const k = ease(t);
      controls.target.lerpVectors(shot.fromTarget, shot.toTarget, k);
      const direction = shot.fromDirection.clone().lerp(shot.toDirection, k).normalize();
      const distance = shot.fromDistance + (shot.toDistance - shot.fromDistance) * k;
      camera.position.copy(controls.target).addScaledVector(direction, distance);
      if (t >= 1) shot = null;
    }
    const turning = moving(now);
    show();
    return Boolean(playing || back || shot || turning);
  }

  function show() {
    const fan = fans.get(selected);
    const value = pose[selected];
    fan.needle.rotation.z = fan.zero + fan.turn * value;
    const d = deg(value);
    slider.value = String(d);
    angle.textContent = `${d >= 0.5 ? '+' : d <= -0.5 ? '−' : ''}${Math.abs(Math.round(d))}°`;
    play.setAttribute('aria-pressed', String(playing));
    play.setAttribute('aria-label', playing ? pick('일시정지', 'Pause') : pick('재생', 'Play'));
  }

  const picture = addView(canvas, { camera, pose, step });
  for (const fan of fans.values()) picture.own(fan.group);
  // The board with the one arm in the middle of it, along it.
  if (board) {
    const middle = board.clone();
    const pan = robot.joints.get(`${SIDE}_shoulder_pan`).anchor.getWorldPosition(new THREE.Vector3());
    middle.position.y = world.worldToLocal(pan).y;
    world.add(middle);
    picture.own(middle);
  }

  function aimAt(name, now) {
    const toDirection = lookFor(name);
    const { target, distance } = fit(frames.get(name), toDirection);
    const look = camera.position.clone().sub(controls.target);
    shot = {
      start: now,
      fromTarget: controls.target.clone(), toTarget: target,
      fromDirection: look.lengthSq() > 0 ? look.normalize() : toDirection, toDirection,
      fromDistance: look.length() || distance, toDistance: distance,
    };
  }
  function place(direction) {
    const { target, distance } = fit(frames.get(selected), direction);
    controls.target.copy(target);
    camera.position.copy(target).addScaledVector(direction, distance);
  }
  picture.onResize = () => {
    if (selected && !shot) place(camera.position.clone().sub(controls.target).normalize());
  };

  function select(name, now) {
    if (name === selected) return;
    if (selected) {
      fans.get(selected).group.visible = false;
      back = { name: selected, from: pose[selected], to: rest[selected], start: now };
    }
    selected = name;
    if (back?.name === name) back = null;
    pose[name] = rest[name];
    phase = phaseAt(limits(name), pose[name]);
    period = (2 * deg(limits(name).upper - limits(name).lower)) / SWEEP_DPS;
    fans.get(name).group.visible = true;
    for (const chip of chips) chip.setAttribute('aria-pressed', String(chip.dataset.joint === name));
    slider.min = String(deg(limits(name).lower));
    slider.max = String(deg(limits(name).upper));
    show();
  }

  chips.forEach((chip) => chip.addEventListener('click', () => {
    const now = performance.now();
    select(chip.dataset.joint, now);
    aimAt(chip.dataset.joint, now);
    picture.redraw();
  }));
  slider.addEventListener('input', () => {
    playing = false;
    pose[selected] = THREE.MathUtils.degToRad(Number(slider.value));
    show();
    picture.redraw();
  });
  play.addEventListener('click', () => {
    playing = !playing;
    if (playing) phase = phaseAt(limits(selected), pose[selected]);
    show();
    picture.redraw();
  });

  select(names[0], performance.now());
  place(lookFor(names[0]));
  picture.redraw();
  figure.classList.add('ready');
}
