// An assembly, step by step: every part where it ends up, the ones a step adds
// in the shell's pink and coming in along the way they can go on, the ones
// before pale, the ones after not yet there; then the step's screws turn in.
// The parts are the description's own meshes where they sit on the arm - the
// Hermit's are the studio's Hermit with the parts it leaves out - so the
// picture is the robot as it is built, not a drawing of it. Which way each
// step's parts come on and where its screws go is worked out from the same
// meshes (docs/scripts/assembly.py, public/robot/assembly.json).

import * as THREE from 'three';
import { STLLoader } from '../../../src/pinkrab_studio/pinkrab_studio/static/STLLoader.js';
import { drawPart, loadURDF } from '../../../src/pinkrab_studio/pinkrab_studio/static/urdf.js';
import { HULL, OUTLINE, materialFor, pegboard as drawPegboard } from '../../../src/pinkrab_studio/pinkrab_studio/static/look.js';
import { PAPER, hermit as drawHermit } from '../../../src/pinkrab_studio/pinkrab_studio/static/hermit.js';
import { addView, fitView, orbit } from './canvases.js';

const FOV = 30;
const MARGIN = 1.15;
// In the description's z-up frame: in front, a little above, to the outer
// side. The Hermit lies with its jaws along +x and its marker solid up.
const ARM_VIEW = new THREE.Vector3(1.0, 0.9, 0.55);
const HERMIT_VIEW = new THREE.Vector3(0.45, -1.0, 0.6);
// How far a step's parts come in from, as a share of the whole assembly's
// size but no further than the way was found clear for, and how long they
// take. A first part, with nothing to clear, comes out of the board (the arm)
// or down from above (the Hermit).
const APPROACH = 0.18;
const CLEAR = 0.05;
const ARRIVE_S = 0.9;
// A part a board is screwed onto is first seen whole, held off, from the
// board's side, this long.
const HOLD_S = 1.2;
const HOLD_CONTEXT = 0.05;
// Then each screw turns in from this far out, one after another.
const LIFT = 0.012;
const TURNS = 4;
const DRIVE_S = 0.7;
const STAGGER_S = 0.09;
// Each step's picture is framed on what it adds, with this share of the
// whole assembly's size round it at least, so a servo is seen where it goes.
// The arm is long and its parts small against it; the Hermit is all one handful.
const CONTEXT = { arm: 0.22, hermit: 0.5 };
// Once the parts are on, the camera closes in on the screws: these few
// centimetres round them, so a 2 mm screw is seen turning.
const SCREW_CONTEXT = 0.05;
const REFRAME_S = 0.6;
// How near the camera may come to what it looks at (m), and how far back it
// may go, in the whole assembly's size; and how much a zoom button moves it.
const MIN_DISTANCE = 0.03;
const FARTHEST = 3;
const ZOOM_STEP = 1.25;

const DONE = {
  shell: new THREE.MeshStandardMaterial({ color: 0xf2d6e2, roughness: 0.6, metalness: 0.02 }),
  hardware: new THREE.MeshStandardMaterial({ color: 0x8a8290, roughness: 0.6, metalness: 0.1 }),
  screw: new THREE.MeshStandardMaterial({ color: 0x6b707a, roughness: 0.4, metalness: 0.6 }),
};
const NOW = {
  shell: new THREE.MeshStandardMaterial({ color: 0xe2488e, roughness: 0.54, metalness: 0.02 }),
  // A servo added is the shell's deep pink: dark enough to read as the servo,
  // pink enough to read as this step's.
  hardware: new THREE.MeshStandardMaterial({ color: 0xb8306f, roughness: 0.5, metalness: 0.1 }),
  // This step's screws in the handles' yellow: they have to be found in a
  // hole of a pink part, a few millimetres across.
  screw: new THREE.MeshStandardMaterial({ color: 0xf5b301, roughness: 0.35, metalness: 0.35,
    emissive: 0x6b4a00, emissiveIntensity: 0.35 }),
};
// A part a step's screws are behind while they turn in - the inside of a
// bracket, a nut's pocket - is seen through, without its outline: a part this
// far or more in front of a screw's head, not just the rim of its hole.
const IN_FRONT = 0.004;
const GHOSTS = new Map();
function ghostOf(material) {
  if (!GHOSTS.has(material)) {
    const ghost = material.clone();
    ghost.transparent = true;
    ghost.opacity = 0.35;
    ghost.depthWrite = false;
    GHOSTS.set(material, ghost);
  }
  return GHOSTS.get(material);
}
// The cross of a screw's drive.
const RECESS = new THREE.MeshStandardMaterial({ color: 0x1d1b2e, roughness: 0.8 });
const DRESS = { outlineMaterial: OUTLINE, hullMaterial: HULL };
// A screw's look by where it goes: shank diameter, head diameter and height (m).
const SCREW = {
  mount: { d: 0.002, head: 0.0037, h: 0.0014 },
  horn: { d: 0.0025, head: 0.0045, h: 0.0018 },
  centre: { d: 0.003, head: 0.0055, h: 0.002 },
  // the base's countersunk bolts, their heads flush in the plate
  bolt: { d: 0.004, head: 0.008, h: 0.0022, sunk: true },
  // a screw through one printed part into another, by its hole
  'M2×6': { d: 0.002, head: 0.0037, h: 0.0014 },
  'M2.5': { d: 0.0025, head: 0.0045, h: 0.0017 },
  'M3×10': { d: 0.003, head: 0.0055, h: 0.003 },
};
// An M3 nut: across its flats, thick.
const NUT = { flats: 0.0055, h: 0.0024 };

const stem = (file, side) => file.replace(/\.stl$/, '').replace(new RegExp(`_${side}$`), '');

// A servo board, roughly, in its own colours whatever the step: across it x,
// up it y away from its servo sockets along the bottom, out of its free face
// z. Its USB-C is at the end `usb` (+1 or -1) of x with its power at the
// other, or with `usb` 0 both along the top, the power a DC jack; with `xiao`
// the USB-C is on a XIAO module that stands on headers.
const PCB = new THREE.MeshStandardMaterial({ color: 0x3a3f47, roughness: 0.7, metalness: 0.05 });
const CHIP = new THREE.MeshStandardMaterial({ color: 0x1d1f24, roughness: 0.5 });
const CONNECTOR = new THREE.MeshStandardMaterial({ color: 0xe9e6e2, roughness: 0.6 });
const TERMINAL = new THREE.MeshStandardMaterial({ color: 0x5b616b, roughness: 0.6 });
const METAL = new THREE.MeshStandardMaterial({ color: 0xb4b8bf, roughness: 0.3, metalness: 0.8 });
// m: a XIAO and its headers, a USB-C, a servo socket, a power terminal (x, y, z)
const XIAO = [0.021, 0.0175, 0.0012];
const HEADER = 0.0025;
const USB_C = [0.0074, 0.009, 0.0032];
const SOCKET = [0.008, 0.006, 0.006];
const TERMINAL_SIZE = [0.0075, 0.01, 0.0085];
const JACK = [0.009, 0.014, 0.011];
const EDGE = 0.001;
function boardShapes({ size: [length, width, t], usb, xiao }) {
  const box = ([x, y, z], [sx, sy, sz], material) => [new THREE.BoxGeometry(sx, sy, sz).translate(x, y, z), material];
  const face = t / 2;
  const shapes = [box([0, 0, 0], [length, width, t], PCB)];
  // the servo sockets along the bottom, open downward
  const low = -width / 2 + SOCKET[1] / 2;
  for (const x of [-0.2, 0.2]) shapes.push(box([x * length, low, face + SOCKET[2] / 2], SOCKET, CONNECTOR));
  if (!usb) {
    // both out of the top edge, a little past it for their plugs
    const top = width / 2 + EDGE;
    shapes.push(box([-0.2 * length, top - JACK[1] / 2, face + JACK[2] / 2], JACK, TERMINAL));
    shapes.push(box([0.2 * length, top - USB_C[0] / 2, face + USB_C[2] / 2], [USB_C[1], USB_C[0], USB_C[2]], METAL));
    shapes.push(box([0.05 * length, -0.05 * width, face + 0.0005], [0.007, 0.007, 0.001], CHIP));
    return shapes;
  }
  shapes.push(box([-usb * (length / 2 - TERMINAL_SIZE[0] / 2 - EDGE), 0, face + TERMINAL_SIZE[2] / 2], TERMINAL_SIZE, TERMINAL));
  const end = usb * (length / 2 + EDGE);     // the USB-C a little past the edge, for its plug
  if (xiao) {
    const x = usb * (length / 2 - XIAO[0] / 2 - EDGE), y = width * 0.12;
    for (const side of [-1, 1]) {
      shapes.push(box([x, y + side * (XIAO[1] / 2 - HEADER / 2), face + HEADER / 2], [XIAO[0], HEADER, HEADER], CHIP));
    }
    const top = face + HEADER + XIAO[2];
    shapes.push(box([x, y, top - XIAO[2] / 2], XIAO, PCB));
    shapes.push(box([end - usb * USB_C[0] / 2, y, top + USB_C[2] / 2], USB_C, METAL));
  } else {
    shapes.push(box([0, width * 0.1, face + 0.0005], [0.007, 0.007, 0.001], CHIP));
    shapes.push(box([end - usb * USB_C[0] / 2, width * 0.1, face + USB_C[2] / 2], USB_C, METAL));
  }
  return shapes;
}

/** A nut about the origin, its faces square to y. */
function nutOf() {
  const nut = new THREE.Mesh(new THREE.CylinderGeometry(NUT.flats / Math.sqrt(3), NUT.flats / Math.sqrt(3), NUT.h, 6),
    NOW.screw);
  const group = new THREE.Group();
  group.add(nut);
  return group;
}

/** A screw along +y: its head on the seat at the origin - above it, or
 * sunk below it to lie flush - and its shank down -y. */
function screwOf(kind, length) {
  const { d, head, h, sunk } = SCREW[kind];
  const group = new THREE.Group();
  const shank = new THREE.Mesh(new THREE.CylinderGeometry(d / 2, d / 2, Math.max(length, d), 12), NOW.screw);
  shank.position.y = -Math.max(length, d) / 2;
  const cap = new THREE.Mesh(new THREE.CylinderGeometry(head / 2, sunk ? d / 2 : head / 2, h, 20), NOW.screw);
  cap.position.y = sunk ? -h / 2 : h / 2;
  const top = sunk ? 0 : h;
  // The drive's cross, so a turn reads as a turn.
  const across = new THREE.Mesh(new THREE.BoxGeometry(head * 0.62, h * 0.3, head * 0.16), RECESS);
  across.position.y = top;
  const along = across.clone();
  along.rotation.y = Math.PI / 2;
  group.add(shank, cap, across, along);
  return group;
}

/**
 * The assembly in `figure`: `sources` the robot's files (arm-figure.ts),
 * `kind` 'arm' or 'hermit'. The steps are the figure's `<li>`s, each with its
 * parts in `data-parts`. Returns once the first side is drawn; the side
 * buttons and step buttons in the figure drive it from there.
 */
export async function showAssembly(figure, sources, kind) {
  const canvas = figure.querySelector('canvas');
  const items = [...figure.querySelectorAll('ol.steps > li')];
  const steps = items.map((li) => (li.dataset.parts || '').split(/\s+/).filter(Boolean));
  const count = figure.querySelector('.count');
  const [prev, next] = ['.prev', '.next'].map((s) => figure.querySelector(s));
  const [closer, farther] = ['.zoom .in', '.zoom .out'].map((s) => figure.querySelector(s));
  const sideButtons = [...figure.querySelectorAll('.sides button')];

  const scene = new THREE.Scene();
  const world = new THREE.Group();
  world.rotation.x = -Math.PI / 2;   // the description is z-up
  scene.add(world);
  scene.add(new THREE.HemisphereLight(0xfffaf0, 0xcbbfa8, 1.7));
  const key = new THREE.DirectionalLight(0xfff6e8, 2.0);
  key.position.set(1.6, 2.2, 1.2);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xffeef6, 0.7);
  fill.position.set(-1.0, 0.8, -1.2);
  scene.add(fill);

  const [physics, model, hands, laid] = await Promise.all([
    fetch(sources.physics).then((r) => r.json()),
    fetch(sources.model).then((r) => { if (!r.ok) throw new Error('no model'); return r.text(); }),
    kind === 'hermit' ? fetch(sources.hermit).then((r) => r.json()) : null,
    fetch(sources.assembly).then((r) => r.json()),
  ]);
  const robot = await loadURDF(model, sources.meshes, { materialFor, ...DRESS });
  const pose = {};
  // The right arm's lift and elbow turn the other way round (its bands are
  // the left's negated): the same reach is the angles' mirror.
  for (const [side, sign] of [['left', 1], ['right', -1]]) {
    for (const [joint, d] of Object.entries(laid.pose)) pose[`${side}_${joint}`] = sign * THREE.MathUtils.degToRad(d);
  }
  robot.setPose(pose);

  // Each side's assembly, as it is being built (a promise) and once built:
  // the root its ways in are said in, its parts by name - each a holder that
  // the step's arrival moves and whose solids it recolours - and its screws.
  const builds = {};
  const built = {};
  async function build(side) {
    const made = kind === 'arm' ? arm(side) : await hermit(side);
    made.screws = screws(side, made.linkOf);
    for (const [name, holder, i] of boards(side, made.linkOf)) {
      made.parts.set(name, holder);
      if (!steps[i].includes(name)) steps[i].push(name);
    }
    return made;
  }

  function arm(side) {
    const parts = new Map();
    for (const [name, link] of robot.links) {
      if (!name.startsWith(`${side}_`)) continue;
      for (const holder of link.children) {
        if (holder.userData.file) parts.set(`${name.slice(side.length + 1)}:${stem(holder.userData.file, side)}`, holder);
      }
    }
    return { root: robot.root, parts, view: ARM_VIEW, board: true, first: new THREE.Vector3(1, 0, 0),
      linkOf: (name) => robot.links.get(name) };
  }

  // The Hermit as the studio draws it (hermit.js): the arm's gripper link
  // with its servo, the handheld's body and jaw, the marker solid on its
  // peg. Added here, the parts it leaves out: the handheld's own camera
  // bracket and its peg in the claw's frame, the board mount and the driver
  // plate in the solid's. Laid down with its jaws level and its solid up.
  async function hermit(side) {
    const loader = new STLLoader();
    const load = (name) => loader.loadAsync(`${sources.hermitMeshes}/${name}`);
    const extra = { camera_bracket: `camera_bracket_${side}.stl`, peg: `handheld_peg_${side}.stl`,
      board_mount: `board_mount_${side}.stl`, driver_plate: `driver_plate_${side}.stl` };
    const [made, ...meshes] = await Promise.all([
      drawHermit(robot, side, hands[side], sources.hermitMeshes),
      ...Object.values(extra).map(load),
    ]);
    const geometry = Object.fromEntries(Object.keys(extra).map((k, i) => [k, meshes[i]]));
    const { hand, body, solid } = made;
    hand.position.set(0, 0, 0);
    hand.quaternion.identity();
    const parts = new Map([['body', body], ['solid', solid]]);
    hand.traverse((o) => {
      const file = o.userData.file;
      if (file === `jaw_${side}.stl`) parts.set('jaw', o);
      else if (file === 'servo.stl') parts.set('servo', o);
      else if (file === `camera_bracket_${side}.stl`) o.visible = false;   // the handheld's own stands in its place
    });
    const shell = body.children.find((c) => c.isMesh && c.material !== HULL).material;
    const beside = (holder, frame, name) => {
      holder.position.copy(frame.position);
      holder.quaternion.copy(frame.quaternion);
      frame.parent.add(holder);
      drawPart(holder, geometry[name], shell, DRESS);
      parts.set(name, holder);
    };
    for (const name of ['camera_bracket', 'peg']) beside(new THREE.Group(), body, name);
    for (const name of ['board_mount', 'driver_plate']) beside(new THREE.Group(), solid, name);
    // Jaws along +x, the solid up: the claw's +z (its fingertips) level, the
    // way from the body to the solid, square to it, straight up.
    hand.updateMatrixWorld(true);
    const along = new THREE.Vector3(0, 0, 1).applyQuaternion(body.quaternion).normalize();
    const middle = (o) => new THREE.Box3().setFromObject(o).getCenter(new THREE.Vector3());
    const up = middle(solid).sub(middle(body));
    up.addScaledVector(along, -up.dot(along)).normalize();
    const across = new THREE.Vector3().crossVectors(up, along);
    hand.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(along, across, up).invert());
    world.add(hand);
    hand.visible = false;
    return { root: hand, parts, view: HERMIT_VIEW, board: false, first: null,
      linkOf: (name) => (name === hand.name ? hand : hand.getObjectByName(name)) };
  }

  // The side's screws, each on the link it goes with, seated; hidden until its step.
  function screws(side, linkOf) {
    const out = [];
    for (const s of laid[kind][side].screws) {
      const link = linkOf(s.link);
      if (!link) continue;
      const screw = s.kind === 'nut' ? nutOf() : screwOf(s.kind === 'join' ? s.size : s.kind, s.length);
      const seat = new THREE.Vector3(...s.seat);
      const axis = new THREE.Vector3(...s.axis).normalize();
      screw.position.copy(seat);
      screw.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), axis);
      screw.visible = false;
      link.add(screw);
      rest.set(screw, seat);
      // A nut is pushed into its pocket, not turned.
      out.push({ screw, step: s.step, seat, axis, turned: screw.quaternion.clone(), turns: s.kind === 'nut' ? 0 : TURNS });
    }
    return out;
  }

  // The servo boards a step screws onto its part, each a plate on the link
  // the part is on: [name, holder, step].
  function boards(side, linkOf) {
    return laid[kind][side].boards.flatMap((b) => {
      const link = linkOf(b.link);
      if (!link) return [];
      const holder = new THREE.Group();
      holder.userData.hardware = true;
      for (const [geometry, material] of boardShapes(b)) drawPart(holder, geometry, material, DRESS);
      for (const o of holder.children) if (o.isMesh && o.material !== HULL) o.userData.kept = true;
      const u = new THREE.Vector3(...b.u), v = new THREE.Vector3(...b.v);
      holder.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(u, v, new THREE.Vector3().crossVectors(u, v)));
      holder.position.set(...b.centre);
      holder.visible = false;
      link.add(holder);
      return [[`board:${b.step}`, holder, b.step]];
    });
  }

  world.add(robot.root);
  const slab = physics?.scenery?.pegboard ?? null;
  const pegboard = slab ? drawPegboard(slab, 1) : null;
  if (pegboard) world.add(pegboard);

  // Where each part sits once it is on, in its holder's parent's frame.
  const rest = new Map();
  const kindOf = (holder) => (holder.userData.hardware || holder.userData.file?.startsWith('servo') ? 'hardware' : 'shell');

  let current = null;
  let side = 'left';
  let step = 0;
  let arriving = null;       // { from: Map(holder -> offset), t }
  let driving = null;        // { screws: [...], t }
  let reframing = null;      // { from: {target, position}, to: {target, position}, t }

  const camera = new THREE.PerspectiveCamera(FOV, 1, 0.005, 20);
  const { controls, moving, zoomBy } = orbit(camera, canvas, () => picture.redraw(), { zoom: true });
  controls.minDistance = MIN_DISTANCE;
  const picture = addView(canvas, {
    scene, camera,
    step(now, dt) {
      let busy = moving(now);
      const distance = camera.position.distanceTo(controls.target);
      closer.disabled = distance <= controls.minDistance * 1.001;
      farther.disabled = distance >= controls.maxDistance * 0.999;
      if (arriving) {
        arriving.t = Math.min(1, arriving.t + dt / ARRIVE_S);
        const k = 1 - (1 - arriving.t) ** 3;
        for (const [holder, offset] of arriving.from) {
          holder.position.copy(rest.get(holder)).addScaledVector(offset, 1 - k);
        }
        if (arriving.t >= 1) arriving = null;
        busy = true;
      }
      if (driving) {
        driving.t += dt;
        if (!driving.closed && driving.t >= driving.wait) {
          driving.closed = true;
          focus(driving.screws.map((s) => s.screw), true, SCREW_CONTEXT, facing(driving.screws));
          // where the heads sit once in, with their part if it is held off
          const heads = driving.screws.map((s) => s.screw.parent.localToWorld(
            s.seat.clone().add(driving.off?.get(s.screw) ?? new THREE.Vector3())));
          seeThrough(heads, reframing?.to.position ?? camera.position);
        }
        let left = false;
        driving.screws.forEach((s, i) => {
          const t = (driving.t - driving.wait - i * STAGGER_S) / DRIVE_S;
          s.screw.visible = t > 0;
          const k = Math.min(1, Math.max(0, t));
          const out = 1 - (1 - k) ** 2;
          s.screw.position.copy(s.seat).addScaledVector(s.axis, LIFT * (1 - out));
          if (driving.off) s.screw.position.add(driving.off.get(s.screw));
          s.screw.quaternion.copy(s.turned).multiply(
            new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), -s.turns * 2 * Math.PI * (1 - out)));
          left ||= k < 1;
        });
        if (!left) {
          const then = driving.then;
          driving = null;
          then?.();
        }
        busy = true;
      }
      if (reframing) {
        reframing.t = Math.min(1, reframing.t + dt / REFRAME_S);
        const k = 1 - (1 - reframing.t) ** 3;
        controls.target.lerpVectors(reframing.from.target, reframing.to.target, k);
        camera.position.lerpVectors(reframing.from.position, reframing.to.position, k);
        if (reframing.t >= 1) reframing = null;
        busy = true;
      }
      return busy;
    },
  });

  // The marker solid stays white, as it is printed, and a board as it is
  // made: their solids are not repainted.
  function paint(holder, look) {
    for (const o of holder.children) {
      if (o.isMesh && o.material !== HULL && o.material !== PAPER && !o.userData.kept) o.material = look;
    }
  }
  function paintScrew(s, look) {
    for (const o of s.screw.children) if (o.material !== RECESS) o.material = look;
  }

  // The parts between `eye` and `points`, made see-through until the next
  // step: each mesh with what it had, to put back (or, with no material, an
  // object hidden for a while).
  let ghosted = [];
  const solid = (o) => o.isMesh && o.material !== HULL && !o.material.transparent;
  function seeThrough(points, eye) {
    world.updateMatrixWorld(true);
    const holders = new Set([...current.parts.values()].filter((h) => h.visible));
    const ray = new THREE.Raycaster();
    const front = new Set();
    for (const point of points) {
      const to = point.clone().sub(eye);
      const far = to.length();
      ray.set(eye, to.normalize());
      ray.far = far - IN_FRONT;
      for (const hit of ray.intersectObjects([...holders], true)) {
        if (!solid(hit.object)) continue;
        let holder = hit.object;
        while (holder && !holders.has(holder)) holder = holder.parent;
        if (holder) front.add(holder);
      }
    }
    for (const holder of front) {
      holder.traverse((o) => {
        if (!o.isMesh) return;
        ghosted.push([o, o.material, o.visible]);
        if (o.material === HULL) o.visible = false;
        else o.material = ghostOf(o.material);
      });
    }
  }
  function solidAgain() {
    for (const [o, material, visible] of ghosted) { if (material) o.material = material; o.visible = visible; }
    ghosted = [];
  }

  // The way a step's parts come on, in the scene's frame.
  function wayIn(index) {
    const given = laid[kind][side].approach[index];
    const frame = current.root.getWorldQuaternion(new THREE.Quaternion());
    if (given) return new THREE.Vector3(...given).applyQuaternion(frame).normalize();
    if (current.first) return current.first.clone().applyQuaternion(frame).normalize();
    return new THREE.Vector3(0, 1, 0);
  }

  function show(index, animate = true) {
    if (!current) return;     // nothing drawn yet, or never
    step = Math.max(0, Math.min(steps.length - 1, index));
    solidAgain();
    if (pegboard) pegboard.visible = current.board;
    const { parts } = current;
    const added = new Set();
    steps.forEach((names, i) => {
      for (const name of names) {
        const holder = parts.get(name);
        if (!holder) continue;
        holder.visible = i <= step;
        if (i < step) paint(holder, DONE[kindOf(holder)]);
        if (i === step) { paint(holder, NOW[kindOf(holder)]); added.add(holder); }
        if (rest.has(holder)) holder.position.copy(rest.get(holder));
      }
    });
    // The screws: those of steps done in, seated; this step's turned in after its parts.
    const these = [];
    for (const s of current.screws) {
      s.screw.position.copy(s.seat);
      s.screw.quaternion.copy(s.turned);
      s.screw.visible = s.step <= step;
      paintScrew(s, s.step < step ? DONE.screw : NOW.screw);
      if (s.step === step) these.push(s);
    }
    // In along the way the step's parts can come on.
    driving = null;
    arriving = null;
    const from = new Map();
    if (animate && added.size) {
      world.updateMatrixWorld(true);
      const away = wayIn(step).multiplyScalar(Math.min(APPROACH * size, CLEAR));
      const off = (o) => away.clone().applyQuaternion(o.parent.getWorldQuaternion(new THREE.Quaternion()).invert());
      for (const holder of added) from.set(holder, off(holder));
      // A board is screwed onto its part first, off the robot, and the two
      // go on together with the screws in.
      if (parts.has(`board:${step}`)) for (const s of these) from.set(s.screw, off(s.screw));
    }
    shown = added.size ? [...added] : these.length ? these.map((s) => s.screw)
      : [...parts.values()].filter((h) => h.visible);
    if (animate && these.length) for (const s of these) s.screw.visible = false;
    if (animate && these.length && from.has(these[0].screw)) {
      for (const [o, offset] of from) o.position.copy(rest.get(o)).add(offset);
      driving = { screws: these, t: 0, wait: HOLD_S, closed: false, off: from,
        then: () => {
          solidAgain();
          if (pegboard) pegboard.visible = current.board;
          arriving = { from, t: 0 };
          focus(shown, true);
        } };
      // Seen first from the board's side, the screws' heads on the other: in
      // the hand, off the robot - the robot so far, its screws and the
      // pegboard out of the picture until the part goes on.
      if (pegboard) pegboard.visible = false;
      for (const o of [...parts.values(), ...current.screws.map((s) => s.screw)]) {
        if (!o.visible || added.has(o) || these.some((s) => s.screw === o)) continue;
        ghosted.push([o, null, true]);
        o.visible = false;
      }
      focus(shown, true, HOLD_CONTEXT, facing(these).negate());
    } else {
      if (animate && these.length) driving = { screws: these, t: 0, wait: added.size ? ARRIVE_S : 0, closed: false };
      if (from.size) arriving = { from, t: 0 };
      focus(shown, animate);
    }
    items.forEach((li, i) => {
      li.classList.toggle('on', i === step);
      li.setAttribute('aria-current', i === step ? 'step' : 'false');
    });
    count.textContent = `${step + 1} / ${steps.length}`;
    // A button about to be turned off while it has the focus hands it over,
    // so the arrow keys still reach the figure.
    const first = step === 0, last = step === steps.length - 1;
    if (first && document.activeElement === prev) next.focus();
    if (last && document.activeElement === next) prev.focus();
    prev.disabled = first;
    next.disabled = last;
    picture.redraw();
  }

  let size = 1;
  let shown = [];
  const corners = (box) => Array.from({ length: 8 }, (_, c) => new THREE.Vector3(
    c & 1 ? box.max.x : box.min.x, c & 2 ? box.max.y : box.min.y, c & 4 ? box.max.z : box.min.z));

  // The whole assembly's size, and a first look along `direction` at all of it.
  function frame(direction) {
    world.updateMatrixWorld(true);
    const box = new THREE.Box3();
    for (const holder of current.parts.values()) box.expandByObject(holder);
    size = box.getSize(new THREE.Vector3()).length();
    controls.maxDistance = FARTHEST * size;
    const { target, distance } = fitView(corners(box), direction, camera, MARGIN);
    controls.target.copy(target);
    camera.position.copy(target).addScaledVector(direction, distance);
  }

  // The camera brought round to `objects` - as they stand once on - from the
  // way it looks now, with room round them for what they go onto.
  // A way to look at screws from where they go in - a screw's head, a nut's
  // pocket - turned from the way the camera looks now.
  function facing(list) {
    const now = camera.position.clone().sub(controls.target).normalize();
    const sum = new THREE.Vector3();
    for (const s of list) sum.add(s.axis.clone().transformDirection(s.screw.parent.matrixWorld));
    if (sum.lengthSq() < 1e-12) return now;
    return now.multiplyScalar(0.5).add(sum.normalize()).normalize();
  }

  function focus(objects, animate, least = CONTEXT[kind] * size, look = null) {
    if (!objects.length) return;
    for (const [holder] of arriving?.from ?? []) holder.position.copy(rest.get(holder));
    const hidden = objects.filter((o) => !o.visible);
    for (const o of hidden) o.visible = true;
    world.updateMatrixWorld(true);
    const box = new THREE.Box3();
    for (const o of objects) box.expandByObject(o);
    for (const o of hidden) o.visible = false;
    if (arriving) for (const [holder, offset] of arriving.from) holder.position.copy(rest.get(holder)).add(offset);
    if (box.isEmpty()) return;
    const middle = box.getCenter(new THREE.Vector3());
    const span = box.getSize(new THREE.Vector3());
    box.setFromCenterAndSize(middle, new THREE.Vector3(Math.max(span.x, least), Math.max(span.y, least),
      Math.max(span.z, least)));
    const direction = look ?? camera.position.clone().sub(controls.target).normalize();
    const { target, distance } = fitView(corners(box), direction, camera, MARGIN);
    const to = { target, position: target.clone().addScaledVector(direction, distance) };
    if (!animate) {
      controls.target.copy(to.target);
      camera.position.copy(to.position);
      reframing = null;
      return;
    }
    reframing = { from: { target: controls.target.clone(), position: camera.position.clone() }, to, t: 0 };
  }

  async function choose(which) {
    side = which;
    sideButtons.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.side === side)));
    // Built once, however often it is asked for meanwhile; drawn only if it
    // is still the side asked for when it is ready.
    builds[which] ??= build(which).then((b) => (built[which] = b));
    let chosen;
    try {
      chosen = await builds[which];
    } catch (error) {
      // Asked again, it is built again; meanwhile the side drawn is the one pressed.
      delete builds[which];
      if (side === which && current) {
        side = current === built.right ? 'right' : 'left';
        sideButtons.forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.side === side)));
      }
      throw error;
    }
    if (side !== which) return;
    current = chosen;
    // Only this side's parts and root: the other arm, the other hand, out of
    // the way - with what the step being shown had hidden or seen through put
    // back first.
    solidAgain();
    for (const b of Object.values(built)) {
      b.root.visible = false;
      for (const holder of b.parts.values()) holder.visible = false;
      for (const s of b.screws) s.screw.visible = false;
    }
    current.root.visible = true;
    if (kind === 'arm') {
      robot.root.traverse((o) => {
        if (o.userData.file) o.visible = false;
      });
      if (pegboard) pegboard.visible = current.board;
    } else {
      if (pegboard) pegboard.visible = false;
      robot.root.visible = false;
    }
    for (const holder of current.parts.values()) if (!rest.has(holder)) rest.set(holder, holder.position.clone());
    const view = current.view.clone();
    // The right is the left's mirror across the description's y: seen from the mirrored side.
    if (side === 'right') view.y *= -1;
    frame(view.normalize().applyQuaternion(world.quaternion));
    show(step, false);
  }

  // A zoom from where the camera is: a step's move to its parts stops there.
  for (const [button, factor] of [[closer, 1 / ZOOM_STEP], [farther, ZOOM_STEP]]) {
    button.addEventListener('click', () => { reframing = null; zoomBy(factor); });
  }
  prev.addEventListener('click', () => show(step - 1));
  next.addEventListener('click', () => show(step + 1));
  items.forEach((li, i) => li.querySelector('button')?.addEventListener('click', () => show(i)));
  figure.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight') { show(step + 1); e.preventDefault(); }
    if (e.key === 'ArrowLeft') { show(step - 1); e.preventDefault(); }
    if (e.key === '+' || e.key === '=') { reframing = null; zoomBy(1 / ZOOM_STEP); e.preventDefault(); }
    if (e.key === '-') { reframing = null; zoomBy(ZOOM_STEP); e.preventDefault(); }
  });
  sideButtons.forEach((b) => b.addEventListener('click', () => choose(b.dataset.side).catch(() => {})));
  picture.onResize = () => {
    if (!current) return;
    frame(camera.position.clone().sub(controls.target).normalize());
    focus(shown, false);
  };

  await choose('left');
  // The step titles take a click only now there is a picture to move.
  for (const li of items) li.querySelector('button').disabled = false;
  figure.classList.add('ready');
}
