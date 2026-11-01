// Which joint is which: one arm in the middle of its board, reaching out, with
// each joint's name and servo id drawn out to the side of the picture on a line
// from the joint itself. The labels follow the joints as the view is turned.

import * as THREE from 'three';
import { fitView, orbit } from './canvases.js';
import { SIDE, solids, stage } from './stage.js';

// In the description's z-up frame: in front of the board and to the arm's outer side.
const VIEW = new THREE.Vector3(1.0, 0.9, 0.55);
const FOV = 30;
const MARGIN = 1.6;         // room at the sides for the labels
// The arm spread out, as the joint ranges show it, so no joint hides another.
const REACHING = { shoulder_pan: 0, shoulder_lift: -175, elbow_flex: 120 };
// The servos, base outward, which is also their order on the bus: id 1 to 7.
const JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_yaw', 'wrist_roll', 'gripper'];
const ROW = 30;             // px between two labels' centres
const INSET = 12;           // px from the picture's edge to a label

export async function showNames(figure, sources) {
  const canvas = figure.querySelector('canvas');
  const layer = figure.querySelector('.labels');
  const leaders = figure.querySelector('svg.leaders');
  const { robot, world, board, ready, addView } = await stage(sources);

  const pose = { ...ready };
  for (const [key, d] of Object.entries(REACHING)) pose[`${SIDE}_${key}`] = THREE.MathUtils.degToRad(d);
  robot.setPose(pose);
  world.updateMatrixWorld(true);
  // Where each joint turns: its anchor, on the axis, at the servo's horn.
  const marks = JOINTS.map((key, i) => ({
    key, id: i + 1,
    at: robot.joints.get(`${SIDE}_${key}`).anchor.getWorldPosition(new THREE.Vector3()),
  }));
  const points = [];
  for (const mesh of solids(robot.links.get(`${SIDE}_base_link`))) {
    const box = mesh.geometry.boundingBox ?? (mesh.geometry.computeBoundingBox(), mesh.geometry.boundingBox);
    for (let c = 0; c < 8; c++) {
      points.push(new THREE.Vector3(c & 1 ? box.max.x : box.min.x, c & 2 ? box.max.y : box.min.y,
        c & 4 ? box.max.z : box.min.z).applyMatrix4(mesh.matrixWorld));
    }
  }

  const labels = marks.map(({ key, id }) => {
    const el = document.createElement('span');
    el.className = 'label';
    el.innerHTML = `<b>${id}</b><code>${key}</code>`;
    layer.append(el);
    return el;
  });

  const camera = new THREE.PerspectiveCamera(FOV, 1, 0.01, 20);
  const { controls, moving } = orbit(camera, canvas, () => picture.redraw());
  const picture = addView(canvas, {
    camera, pose,
    step(now) {
      const turning = moving(now);
      lay();
      return turning;
    },
  });
  if (board) {
    const middle = board.clone();
    const pan = robot.joints.get(`${SIDE}_shoulder_pan`).anchor.getWorldPosition(new THREE.Vector3());
    middle.position.y = world.worldToLocal(pan).y;
    world.add(middle);
    picture.own(middle);
  }

  // Each label to the side its joint is on, in the order the joints stand,
  // ROW apart; a line from the label to the joint.
  const seen = new THREE.Vector3();
  function lay() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    const spots = marks.map(({ at }, i) => {
      seen.copy(at).project(camera);
      return { i, x: (seen.x + 1) / 2 * w, y: (1 - seen.y) / 2 * h };
    });
    const middle = spots.reduce((sum, s) => sum + s.x, 0) / spots.length;
    const lines = [];
    for (const side of ['left', 'right']) {
      const these = spots.filter((s) => (side === 'left') === (s.x < middle)).sort((a, b) => a.y - b.y);
      // as near each joint's height as the rows allow, kept inside the picture
      let y = -Infinity;
      const rows = these.map((s) => (y = Math.max(s.y, y + ROW)));
      const over = rows.length ? Math.max(0, rows[rows.length - 1] - (h - ROW / 2)) : 0;
      these.forEach((s, k) => {
        const el = labels[s.i];
        const top = Math.max(ROW / 2, rows[k] - over);
        el.style.top = `${top}px`;
        el.style.left = side === 'left' ? `${INSET}px` : '';
        el.style.right = side === 'right' ? `${INSET}px` : '';
        el.dataset.side = side;
        const edge = side === 'left' ? INSET + el.offsetWidth : w - INSET - el.offsetWidth;
        const path = `${edge},${top} ${edge + (side === 'left' ? 10 : -10)},${top} ${s.x},${s.y}`;
        // a light halo under a dark line, so it reads on the pink arm and the black board alike
        lines.push(`<polyline class="halo" points="${path}"/><polyline points="${path}"/>`,
          `<circle cx="${s.x}" cy="${s.y}" r="4"/>`);
      });
    }
    leaders.setAttribute('viewBox', `0 0 ${w} ${h}`);
    leaders.innerHTML = lines.join('');
  }

  function place(direction) {
    const { target, distance } = fitView(points, direction, camera, MARGIN);
    controls.target.copy(target);
    camera.position.copy(target).addScaledVector(direction, distance);
  }
  picture.onResize = () => place(camera.position.clone().sub(controls.target).normalize());
  place(VIEW.clone().normalize().applyQuaternion(world.quaternion));
  picture.redraw();
  figure.classList.add('ready');
}
