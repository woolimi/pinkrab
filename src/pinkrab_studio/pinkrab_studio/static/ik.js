// Drag the arm by its hand.
//
// Two handles per arm, because the arm splits naturally in two: the shoulder places
// the arm, everything past it aims the hand. The seam is the ELBOW's own origin -
// the forearm is a yaw bracket now, so the elbow turns the hand rather than
// reaching with it. Dragging the ELBOW sphere swings the arm - shoulder_pan and
// shoulder_lift solved for a point that lives on a sphere about the shoulder, which
// is why two angles are the right number for it. Dragging the TIP sphere aims the
// hand - elbow_flex, wrist_flex and wrist_yaw solved for the fingertip, the
// shoulder staying where it is. wrist_roll moves neither point (the tip rides on its axis), so no
// drag can reach it and its slider stays the only way to turn it - which is right,
// a roll is not a direction you can point at with a mouse.
//
// The solver is damped least squares on an analytic Jacobian. Nothing here knows
// the arm's dimensions: axes and origins are read off the posed scene graph, so
// the same code solves both arms, survives every geometry change upstream of it,
// and clamps to whatever limits the description declares.

import * as THREE from 'three';

const ITERATIONS = 12;     // per pointer move - the target moves a little each frame,
                           // so convergence is spread across frames anyway
const DAMPING = 0.05;      // metres; keeps the step finite near singularities
const DONE = 0.0005;       // m - close enough to stop early

/** dq = J^T (J J^T + damping^2 I)^-1 err, for a 3xN Jacobian. */
function dampedStep(cols, err) {
  const d2 = DAMPING * DAMPING;
  // A = J J^T + d2 I  (3x3, symmetric)
  const a = [[d2, 0, 0], [0, d2, 0], [0, 0, d2]];
  for (const c of cols) {
    a[0][0] += c.x * c.x; a[0][1] += c.x * c.y; a[0][2] += c.x * c.z;
    a[1][1] += c.y * c.y; a[1][2] += c.y * c.z;
    a[2][2] += c.z * c.z;
  }
  a[1][0] = a[0][1]; a[2][0] = a[0][2]; a[2][1] = a[1][2];
  const m = new THREE.Matrix3().set(
    a[0][0], a[0][1], a[0][2],
    a[1][0], a[1][1], a[1][2],
    a[2][0], a[2][1], a[2][2],
  ).invert();
  const w = err.clone().applyMatrix3(m);
  return cols.map((c) => c.dot(w));
}

export class Chain {
  constructor(robot, jointNames, frame) {
    this.robot = robot;
    this.names = jointNames.filter((n) => robot.joints.has(n));
    this.frame = frame;                       // the link Group whose origin is solved
  }

  tip() {
    return this.frame.getWorldPosition(new THREE.Vector3());
  }

  /**
   * Reach for `target`, one damped step at a time.
   *
   * Nothing here is refused. What comes out is a request, and whether the arm can
   * take that shape is MuJoCo's answer, arriving a frame later - so this solves
   * freely and the simulation stops the hand on whatever it meets.
   */
  solve(target) {
    const { robot } = this;
    for (let i = 0; i < ITERATIONS; i += 1) {
      robot.root.updateMatrixWorld(true);
      const p = this.tip();
      const err = target.clone().sub(p);
      if (err.length() < DONE) break;

      const cols = this.names.map((n) => {
        const j = robot.joints.get(n);
        const axis = j.axis.clone()
          .applyQuaternion(j.group.getWorldQuaternion(new THREE.Quaternion()));
        const origin = j.group.getWorldPosition(new THREE.Vector3());
        return axis.cross(p.clone().sub(origin));   // d(tip)/d(q), one column
      });

      const dq = dampedStep(cols, err);
      this.names.forEach((n, k) => {
        const j = robot.joints.get(n);
        const value = Math.min(j.upper, Math.max(j.lower, j.value + dq[k]));
        robot.setJoint(n, value);
      });
    }
  }
}

// Saturated and nearly opaque, because the page is light now. The old handles
// leaned on emissive to glow out of a dark scene; against paper that only drains
// the colour, so the contrast has to come from the pigment instead.
//
// A handle sits at its joint's origin, which on this arm is the servo's output
// shaft - inside the servo body. That is where it belongs and it is also where it
// cannot be seen, and the two ways out of that each broke something: standing it off
// along the axis moved the handle off the joint, and switching depth off floated the
// far arm's handles in front of the near arm's.
//
// So it is drawn twice. The solid pass is depth-tested and shows wherever the ball
// actually clears the geometry; the ghost pass ignores depth and paints last at low
// opacity, so the buried part still reads - dimmer, which is exactly the cue that
// says it is behind something. Perspective survives and the handle stays findable.
//
// Only the solid mesh goes in the raycast list, and it does not matter that it may
// be hidden: a ray does not care what is in front of what.
const GHOST_OPACITY = 0.26;

function handleMesh(radius, color) {
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 20, 14),
    new THREE.MeshStandardMaterial({
      color, roughness: 0.32, metalness: 0.0, emissive: color, emissiveIntensity: 0.12,
    }),
  );
  const ghost = new THREE.Mesh(
    mesh.geometry,
    new THREE.MeshBasicMaterial({
      color, transparent: true, opacity: GHOST_OPACITY,
      depthTest: false, depthWrite: false,
    }),
  );
  ghost.renderOrder = 10;
  ghost.raycast = () => {};        // the solid one answers for both
  mesh.add(ghost);
  return mesh;
}

/**
 * Wire the drag handles into the page. `synced` is called whenever the solver has
 * changed joint values, so the sliders can follow the picture; `onBlocked` is called
 * with 'desk' or 'board' when a surface is what stopped the reach, and with null
 * when it did not, so the page can say why the hand will not go further; `onHover`
 * is called with a handle's label and the pointer event, or null when the pointer
 * leaves one. What each sphere does is written on the sphere, which is where you
 * are looking when you want to know.
 */
/**
 * Two kinds of handle, and the difference is the point of them.
 *
 * BLUE is a jog: one ball per joint, sitting on that joint, and dragging it turns
 * THAT JOINT AND NOTHING ELSE. No solver is involved - the angle your pointer sweeps
 * around the joint's own axis is the angle the joint turns. It is the direct control,
 * for when you know which joint you want and by how much.
 *
 * ORANGE is inverse kinematics: one ball at the end of the arm, and dragging it asks
 * every joint between it and the base to co-operate in putting that point where the
 * pointer is. It is the indirect control, for when you know where the hand should be
 * and do not care how the arm gets there.
 *
 * Mixing the two was the mistake worth naming: handles that all solved, each with its
 * own arbitrary slice of the chain, meant no handle moved one joint and no handle
 * moved them all. shoulder_pan gets no blue ball because its origin is bolted to the
 * base, and the gripper gets none because its jaw pivot sits where the orange one is.
 */
const HANDLE_COLOUR = 0x1668c9;      // blue - a joint
const TIP_COLOUR = 0xf5b301;         // orange - the gripper, the end of the arm


/** The moving joints above `object` in the tree, base first. */
export function chainAbove(robot, object) {
  const nameOfGroup = new Map();
  for (const [name, joint] of robot.joints) {
    if (joint.type !== 'fixed') nameOfGroup.set(joint.group, name);
  }
  const out = [];
  for (let o = object.parent; o; o = o.parent) {
    const name = nameOfGroup.get(o);
    if (name) out.push(name);
  }
  return out.reverse();
}

/**
 * Turning one joint by the angle the pointer sweeps about its axis.
 *
 * The handle sits ON the axis, so it cannot be dragged the way a solved handle is -
 * there is no direction to push it. What is read instead is where the pointer's ray
 * crosses the joint's own plane of rotation, as an angle about the joint. Grab
 * anywhere and swing: the joint follows the swing, not the ball.
 */
class Jog {
  constructor(robot, name) {
    this.robot = robot;
    this.name = name;
  }

  /** Fix the rotation plane and the starting angle. Both stay put for the drag. */
  begin(ray) {
    const joint = this.robot.joints.get(this.name);
    this.robot.root.updateMatrixWorld(true);
    this.origin = joint.group.getWorldPosition(new THREE.Vector3());
    this.axis = joint.axis.clone()
      .applyQuaternion(joint.group.getWorldQuaternion(new THREE.Quaternion())).normalize();
    // Any two directions across the axis will do; they only have to stay the same
    // two for the length of the drag.
    const seed = Math.abs(this.axis.x) > 0.9
      ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
    this.u = new THREE.Vector3().crossVectors(this.axis, seed).normalize();
    this.v = new THREE.Vector3().crossVectors(this.axis, this.u);
    this.plane = new THREE.Plane().setFromNormalAndCoplanarPoint(this.axis, this.origin);
    this.from = this._angle(ray);
    this.was = joint.value;
    return this.from !== null;
  }

  _angle(ray) {
    const hit = ray.intersectPlane(this.plane, new THREE.Vector3());
    if (!hit) return null;      // sighting along the plane - no angle to read
    const d = hit.sub(this.origin);
    return Math.atan2(d.dot(this.v), d.dot(this.u));
  }

  /** Turn the joint to wherever the pointer has swung. */
  update(ray) {
    const now = this._angle(ray);
    if (now === null || this.from === null) return null;
    // Wrapped into (-pi, pi], so a drag that crosses the cut does not spin the joint
    // a whole turn the other way.
    let step = now - this.from;
    while (step > Math.PI) step -= 2 * Math.PI;
    while (step < -Math.PI) step += 2 * Math.PI;

    const joint = this.robot.joints.get(this.name);
    // Clamped to what the description declares. A joint limit is not contact - it
    // is a fact about the servo's travel, and the simulation clamps the command to
    // the same range, so asking past it would only show a number nothing can honour.
    const wanted = Math.min(joint.upper, Math.max(joint.lower, this.was + step));
    this.robot.setJoint(this.name, wanted);
  }
}

/**
 * Wire the drag handles into the page.
 *
 * `synced` is called with the joint names the gesture just moved - one for a jog,
 * the chain for a solved handle. Naming them matters: the page may be driving a
 * simulation, and a caller that had to guess would have to read the whole pose
 * back off the robot, which after a frame of physics is where the arm HAS got to
 * rather than what anyone asked for.
 */
export function enableDrag({
  robot, camera, canvas, controls, synced, onHover, locked = () => false,
  onGrab = () => {}, onRelease = () => {},
}) {
  const handles = [];

  for (const arm of ['left', 'right']) {
    // One jog per joint, on the joint. shoulder_pan included this time: a jog turns
    // its own joint, and shoulder_pan can certainly turn - it was only ever excluded
    // because nothing upstream could move a ball sitting on its axis.
    for (const name of robot.order) {
      if (!name.startsWith(`${arm}_`) || name.endsWith('_gripper')) continue;
      const frame = robot.links.get(robot.joints.get(name).child);
      if (!frame) continue;
      const mesh = handleMesh(0.010, HANDLE_COLOUR);
      frame.add(mesh);
      handles.push({ mesh, label: name.slice(arm.length + 1), jog: new Jog(robot, name) });
    }

    // ..and one solver at the end of the arm. Its frame hangs off the gripper link
    // by a fixed joint, so walking up from it collects the six arm joints and not
    // the gripper - which is right, opening the jaw does not move the point between
    // the fingers.
    const tip = robot.links.get(`${arm}_gripper_frame_link`);
    if (!tip) continue;
    const mesh = handleMesh(0.012, TIP_COLOUR);
    tip.add(mesh);
    handles.push({
      mesh, label: 'gripper (IK)', chain: new Chain(robot, chainAbove(robot, tip), tip),
    });
  }

  const caster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const plane = new THREE.Plane();
  const target = new THREE.Vector3();
  let active = null;

  function cast(event) {
    const r = canvas.getBoundingClientRect();
    pointer.set(
      ((event.clientX - r.left) / r.width) * 2 - 1,
      -((event.clientY - r.top) / r.height) * 2 + 1,
    );
    caster.setFromCamera(pointer, camera);
  }

  canvas.addEventListener('pointerdown', (event) => {
    if (locked()) return;
    cast(event);
    const hit = caster.intersectObjects(handles.map((h) => h.mesh), false)[0];
    if (!hit) return;
    active = handles.find((h) => h.mesh === hit.object);
    if (active.jog) {
      // The jog reads an angle about the joint's own axis, so its plane is the
      // joint's plane of rotation, not the screen's.
      if (!active.jog.begin(caster.ray)) { active = null; return; }
    } else {
      // The solve goes on from what it last asked for, not from where the arm is
      // drawn: the picture follows the arm, the arm trails its command, and a
      // solve restarted from a trailing pose each move pulls the request back
      // and forth - the arm then twitches after it. The arm's pose seeds it once.
      active.asked = Object.fromEntries(
        active.chain.names.map((n) => [n, robot.joints.get(n).value]));
      // Drag in the camera-facing plane through the handle: the one mapping of a 2D
      // mouse onto 3D that never surprises, because it moves what you see sideways.
      plane.setFromNormalAndCoplanarPoint(
        camera.getWorldDirection(new THREE.Vector3()),
        active.mesh.getWorldPosition(new THREE.Vector3()),
      );
    }
    controls.enabled = false;
    onGrab(active.jog ? [active.jog.name] : active.chain.names);
    onHover?.(null, event);
    canvas.setPointerCapture(event.pointerId);
    event.preventDefault();
  });

  canvas.addEventListener('pointermove', (event) => {
    if (!active) {
      cast(event);
      const hit = caster.intersectObjects(handles.map((h) => h.mesh), false)[0];
      const over = hit ? handles.find((h) => h.mesh === hit.object) : null;
      canvas.style.cursor = over ? (locked() ? 'not-allowed' : 'grab') : '';
      onHover?.(over?.label ?? null, event);
      return;
    }
    cast(event);
    if (active.jog) {
      active.jog.update(caster.ray);
    } else {
      if (!caster.ray.intersectPlane(plane, target)) return;
      for (const [n, v] of Object.entries(active.asked)) robot.setJoint(n, v);
      active.chain.solve(target);
      for (const n of active.chain.names) active.asked[n] = robot.joints.get(n).value;
    }
    synced(active.jog ? [active.jog.name] : active.chain.names);
  });

  const drop = (event) => {
    if (!active) return;
    active = null;
    controls.enabled = true;
    onRelease();
    if (event.pointerId !== undefined) {
      try { canvas.releasePointerCapture(event.pointerId); } catch { /* already gone */ }
    }
  };
  canvas.addEventListener('pointerup', drop);
  canvas.addEventListener('pointercancel', drop);
  canvas.addEventListener('pointerleave', (event) => onHover?.(null, event));

  return handles;
}

const LOCKED_COLOUR = 0x9a9a9a;

/** Grey the handles out while they cannot be used, and give them their colour back after. */
export function greyHandles(handles, grey) {
  for (const { mesh } of handles) {
    const ghost = mesh.children[0];
    mesh.userData.colour ??= mesh.material.color.getHex();
    const colour = grey ? LOCKED_COLOUR : mesh.userData.colour;
    mesh.material.color.setHex(colour);
    mesh.material.emissive.setHex(colour);
    ghost?.material.color.setHex(colour);
  }
}
