// Parse a URDF into a three.js scene graph, and pose it.
//
// No library: a URDF is a tree of links joined by transforms, which is exactly what
// three.js Object3D already is. Every joint becomes a Group whose local rotation is
// the joint value about its axis, so posing the robot is setting numbers on those
// groups and letting three.js compose the rest.

import * as THREE from 'three';
import { STLLoader } from './STLLoader.js';

const loader = new STLLoader();

// Where the same point appears in two triangles, to within 0.01 mm. Far below any
// feature on a printed part and far above float32 noise at these coordinates, so it
// welds what the exporter meant to be one corner and nothing else.
const WELD = 1e5;

/**
 * Add a `smoothNormal` attribute: each vertex given the area-weighted average of
 * the faces meeting at it.
 *
 * The silhouette hull needs this and the shaded mesh must not have it. An STL
 * carries one normal per triangle, which is exactly right for a hard-surface part -
 * a printed bracket should read as facets, not as a blob. But pushing a hull out
 * along per-face normals moves each triangle in its own direction, and every edge
 * where two faces disagree opens a gap: the outline arrives as a torn fringe rather
 * than a line. Averaged normals move a shared corner one way, and the shell holds.
 *
 * A hash over quantised positions rather than a sort or a Map - 2.5 million vertices
 * arrive across the whole robot, and typed arrays are what keeps that in the tens of
 * milliseconds instead of seconds.
 */
function addSmoothNormals(geometry) {
  const pos = geometry.attributes.position.array;
  const n = pos.length / 3;

  // Open addressing, load factor 1/2, so probes stay short.
  let cap = 4;
  while (cap < n * 2) cap *= 2;
  const mask = cap - 1;
  const slotAt = new Int32Array(cap).fill(-1);
  const keyX = new Int32Array(n);
  const keyY = new Int32Array(n);
  const keyZ = new Int32Array(n);
  const sum = new Float32Array(n * 3);
  const slotOfVertex = new Int32Array(n);
  let slots = 0;

  for (let i = 0; i < n; i += 1) {
    const x = Math.round(pos[i * 3] * WELD);
    const y = Math.round(pos[i * 3 + 1] * WELD);
    const z = Math.round(pos[i * 3 + 2] * WELD);
    let h = (Math.imul(x, 73856093) ^ Math.imul(y, 19349663) ^ Math.imul(z, 83492791)) & mask;
    for (;;) {
      const slot = slotAt[h];
      if (slot < 0) {
        slotAt[h] = slots;
        keyX[slots] = x; keyY[slots] = y; keyZ[slots] = z;
        slotOfVertex[i] = slots;
        slots += 1;
        break;
      }
      if (keyX[slot] === x && keyY[slot] === y && keyZ[slot] === z) {
        slotOfVertex[i] = slot;
        break;
      }
      h = (h + 1) & mask;
    }
  }

  // Cross products are left unnormalised on purpose: their length is twice the
  // triangle's area, so a big face pulls the shared corner around more than a sliver
  // does. Normalising first would let a fan of slivers outvote the face they trim.
  for (let t = 0; t < n; t += 3) {
    const ax = pos[t * 3], ay = pos[t * 3 + 1], az = pos[t * 3 + 2];
    const ux = pos[t * 3 + 3] - ax, uy = pos[t * 3 + 4] - ay, uz = pos[t * 3 + 5] - az;
    const vx = pos[t * 3 + 6] - ax, vy = pos[t * 3 + 7] - ay, vz = pos[t * 3 + 8] - az;
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;
    for (let k = 0; k < 3; k += 1) {
      const slot = slotOfVertex[t + k] * 3;
      sum[slot] += nx; sum[slot + 1] += ny; sum[slot + 2] += nz;
    }
  }

  const smooth = new Float32Array(n * 3);
  for (let i = 0; i < n; i += 1) {
    const slot = slotOfVertex[i] * 3;
    const x = sum[slot], y = sum[slot + 1], z = sum[slot + 2];
    const length = Math.sqrt(x * x + y * y + z * z) || 1;   // hypot is careful, and slow at this count
    smooth[i * 3] = x / length;
    smooth[i * 3 + 1] = y / length;
    smooth[i * 3 + 2] = z / length;
  }
  geometry.setAttribute('smoothNormal', new THREE.BufferAttribute(smooth, 3));
}

function vec(text, fallback = [0, 0, 0]) {
  if (!text) return fallback;
  const parts = text.trim().split(/\s+/).map(Number);
  return parts.length === 3 && parts.every((n) => Number.isFinite(n)) ? parts : fallback;
}

// URDF rpy is fixed-axis roll-pitch-yaw, which is intrinsic ZYX in three.js terms.
function setOrigin(object, originEl) {
  const xyz = vec(originEl?.getAttribute('xyz'));
  const rpy = vec(originEl?.getAttribute('rpy'));
  object.position.set(xyz[0], xyz[1], xyz[2]);
  object.rotation.set(rpy[0], rpy[1], rpy[2], 'ZYX');
}

export class Robot {
  constructor() {
    this.root = new THREE.Group();
    this.links = new Map();   // name -> Group
    this.joints = new Map();  // name -> {group, axis, type, lower, upper, value}
    this.order = [];          // moving joint names, each base outward to its tip
    this.pending = 0;
  }

  /** Set one joint, in radians. Silently ignores unknown names so a partial pose is fine. */
  setJoint(name, value) {
    const joint = this.joints.get(name);
    if (!joint || joint.type === 'fixed') return;
    joint.value = value;
    joint.group.quaternion.setFromAxisAngle(joint.axis, value);
  }

  setPose(pose) {
    for (const [name, value] of Object.entries(pose)) this.setJoint(name, value);
  }

  /** The current pose, as a plain object setPose takes back. */
  pose() {
    const out = {};
    for (const [name, joint] of this.joints) {
      if (joint.type !== 'fixed') out[name] = joint.value;
    }
    return out;
  }

  /**
   * Joint names from each base outward to its tip, fixed ones excluded.
   *
   * Walked off the scene graph, not read out of the joints map. The map is in
   * document order, and the description is not written base-first: the yaw joint
   * was grafted in by a later step and sits at the end of the file, so a panel
   * built from the map put wrist_yaw below shoulder_pan - past the shoulder, on an
   * arm where it is three joints from the hand.
   */
  movingJoints() {
    return this.order.map((name) => {
      const j = this.joints.get(name);
      return { name, lower: j.lower, upper: j.upper, value: j.value };
    });
  }

  /** A fixed joint's placement, as the description states it. */
  anchorOf(name) {
    return this.joints.get(name)?.anchor ?? null;
  }

  worldPosition(linkName) {
    const link = this.links.get(linkName);
    if (!link) return null;
    this.root.updateMatrixWorld(true);
    return link.getWorldPosition(new THREE.Vector3());
  }
}

/**
 * One part as the robot's parts are drawn, into `holder`: the shaded solid,
 * its crease edges and its silhouette. Returns the solid.
 */
export function drawPart(holder, geometry, material, { outlineMaterial, hullMaterial }) {
  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  holder.add(mesh);
  // 32 degrees keeps the edges that describe the part and drops the
  // ones that only describe the mesh: a printed fillet arrives as a
  // fan of narrow triangles, and every seam in it would draw.
  if (!outlineMaterial) return mesh;     // a plain copy - the ghost arm - has no drawing
  const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geometry, 32), outlineMaterial);
  edges.renderOrder = 1;
  holder.add(edges);
  // The same geometry inside out. Untagged and casting nothing: it is a
  // drawing of the part, not a second copy of it, and the collision
  // guard and the shadow pass both have to keep ignoring it.
  addSmoothNormals(geometry);
  const hull = new THREE.Mesh(geometry, hullMaterial);
  hull.castShadow = false;
  hull.receiveShadow = false;
  holder.add(hull);
  return mesh;
}

/**
 * @param materialFor   given a mesh filename and the colour the description states
 *                      for it, the material to wear. The printed
 *                      parts and the bought servos are different objects and the
 *                      picture says so, which is why this is a function and not one
 *                      material for the whole robot.
 * @param outlineMaterial  drawn over every mesh as its crease edges. Shaded solids
 *                      alone lose their form against a light background - a fillet
 *                      and a flat face are the same value from most angles - so the
 *                      edges are always on rather than a thing to switch on.
 * @param hullMaterial  the silhouette, drawn as a back-facing copy pushed outward.
 *                      Crease edges cannot draw one: a watertight STL has no edge
 *                      that belongs to a single triangle, so the one line that
 *                      separates the robot from the room behind it is the line
 *                      EdgesGeometry never emits.
 */
export async function loadURDF(urdfText, meshBase, { materialFor, outlineMaterial, hullMaterial }) {
  const doc = new DOMParser().parseFromString(urdfText, 'application/xml');
  const error = doc.querySelector('parsererror');
  if (error) throw new Error('URDF is not valid XML');

  const robot = new Robot();
  const meshJobs = [];

  // Colours, as the description states them. Read here rather than decided by
  // whatever is drawing: RViz reads these same numbers, and a palette kept beside
  // the renderer is a second robot that slowly stops matching the first.
  const colours = new Map();
  for (const materialEl of doc.querySelectorAll('robot > material')) {
    const name = materialEl.getAttribute('name');
    const rgba = materialEl.querySelector('color')?.getAttribute('rgba');
    if (!name || !rgba) continue;
    const [r, g, b] = rgba.trim().split(/\s+/).map(Number);
    // Read as sRGB, which is how these numbers are meant and how three.js takes a
    // hex colour. Handing them to setRGB as working-space values would darken
    // every part on the robot.
    colours.set(name, new THREE.Color().setRGB(r, g, b, THREE.SRGBColorSpace));
  }

  for (const linkEl of doc.querySelectorAll('robot > link')) {
    const name = linkEl.getAttribute('name');
    const group = new THREE.Group();
    group.name = name;
    robot.links.set(name, group);

    for (const visual of linkEl.querySelectorAll('visual')) {
      const meshEl = visual.querySelector('geometry > mesh');
      if (!meshEl) continue;
      const holder = new THREE.Group();
      setOrigin(holder, visual.querySelector('origin'));
      group.add(holder);

      const file = meshEl.getAttribute('filename').split('/').pop();
      holder.userData.file = file;   // which part this is, for a picture that swaps one
      // A visual either names a material declared at robot scope or carries its own
      // colour inline. Both are URDF; the first is what this description uses.
      const materialEl = visual.querySelector('material');
      const inline = materialEl?.querySelector('color')?.getAttribute('rgba');
      let colour = colours.get(materialEl?.getAttribute('name'));
      if (!colour && inline) {
        const [r, g, b] = inline.trim().split(/\s+/).map(Number);
        colour = new THREE.Color().setRGB(r, g, b, THREE.SRGBColorSpace);
      }
      meshJobs.push(
        new Promise((resolve) => {
          loader.load(
            `${meshBase}/${file}`,
            (geometry) => {
              const mesh = drawPart(holder, geometry, materialFor(file, colour), { outlineMaterial, hullMaterial });
              // Which link a mesh belongs to, so the collision guard can pick out
              // the robot's own geometry and leave the drag handles - which are
              // parented to links too - out of it.
              mesh.userData.urdfLink = name;
              resolve();
            },
            undefined,
            // A missing mesh should leave a gap in the picture, not an empty page.
            () => { console.warn(`mesh failed to load: ${file}`); resolve(); },
          );
        }),
      );
    }
  }

  const children = new Set();
  for (const jointEl of doc.querySelectorAll('robot > joint')) {
    const name = jointEl.getAttribute('name');
    const type = jointEl.getAttribute('type');
    const parentName = jointEl.querySelector('parent').getAttribute('link');
    const childName = jointEl.querySelector('child').getAttribute('link');
    const parent = robot.links.get(parentName);
    const child = robot.links.get(childName);
    if (!parent || !child) continue;
    children.add(childName);

    // Two groups per joint: one holding the fixed offset from the parent, one
    // carrying the joint value. Keeping them apart means posing never disturbs the
    // geometry the URDF fixed.
    const anchor = new THREE.Group();
    setOrigin(anchor, jointEl.querySelector('origin'));
    const articulation = new THREE.Group();
    anchor.add(articulation);
    articulation.add(child);
    parent.add(anchor);

    const limit = jointEl.querySelector('limit');
    robot.joints.set(name, {
      anchor,
      group: articulation,
      child: childName,   // the link this joint carries, so a handle can be put on it
      axis: new THREE.Vector3(...vec(jointEl.querySelector('axis')?.getAttribute('xyz'), [1, 0, 0])).normalize(),
      type,
      lower: limit ? Number(limit.getAttribute('lower')) : -Math.PI,
      upper: limit ? Number(limit.getAttribute('upper')) : Math.PI,
      value: 0,
    });
  }

  for (const [name, group] of robot.links) {
    if (!children.has(name)) robot.root.add(group);
  }

  // Depth-first from the root is base-to-tip along every chain. Settled once, here,
  // because movingJoints() is asked for it on every solver step.
  const nameOfGroup = new Map();
  for (const [name, joint] of robot.joints) {
    if (joint.type !== 'fixed') nameOfGroup.set(joint.group, name);
  }
  robot.order = [];
  robot.root.traverse((object) => {
    const name = nameOfGroup.get(object);
    if (name) robot.order.push(name);
  });

  await Promise.all(meshJobs);
  return robot;
}
