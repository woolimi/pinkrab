// The two measurements `pinkrab mount` asks for, drawn over the arms at home:
// from in front of the board, the spacing between the two shoulder_pan axes and
// the height of the bases' top faces above the desk; from the side, the face
// that height is measured to. The camera stands still - a dimension drawn on a
// picture that turns would point at nothing.
//
// The numbers are read off the model the page draws (its mount.yaml as the
// site was built), so the figure says what the description says.

import * as THREE from 'three';
import { LAYERS, SIDE, solids, stage } from './stage.js';
import { pick } from './lang.js';

const SVG = 'http://www.w3.org/2000/svg';
// Room round the arms, as a fraction of the picture, for the dimensions.
const MARGIN = { front: { top: 0.2, bottom: 0.08, left: 0.2, right: 0.06 },
  side: { top: 0.1, bottom: 0.08, left: 0.06, right: 0.28 } };

export async function showMount(figure, sources, which) {
  const canvas = figure.querySelector('canvas');
  const overlay = figure.querySelector('svg');
  const { robot, world, addView } = await stage(sources);
  const pose = Object.fromEntries(robot.movingJoints().map(({ name }) => [name, 0]));
  robot.setPose(pose);
  world.updateMatrixWorld(true);

  // Points of the description (metres, z up) in the scene.
  const at = (x, y, z) => world.localToWorld(new THREE.Vector3(x, y, z));
  const origin = (link) => {
    const p = new THREE.Vector3();
    robot.links.get(link).getWorldPosition(p);
    return world.worldToLocal(p);
  };
  const other = SIDE === 'left' ? 'right' : 'left';
  const base = { [SIDE]: origin(`${SIDE}_base_link`), [other]: origin(`${other}_base_link`) };
  const pan = { [SIDE]: origin(`${SIDE}_shoulder_link`), [other]: origin(`${other}_shoulder_link`) };
  const top = base[SIDE].z;
  const spacing = Math.abs(base[SIDE].y - base[other].y);

  // Looking at the board from in front (+x), or at the arm from across the
  // other arm's side, the board on the left of the picture.
  const front = which === 'front';
  const direction = front ? at(1, 0, 0).sub(at(0, 0, 0)) : at(0, SIDE === 'left' ? -1 : 1, 0).sub(at(0, 0, 0));
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10);
  camera.up.copy(at(0, 0, 1).sub(at(0, 0, 0)));
  camera.layers.enable(LAYERS.board);
  if (front) camera.layers.enable(LAYERS.otherArm);
  const picture = addView(canvas, { camera, pose });

  // What has to be in the picture: the arms shown, and the desk under them.
  const shown = front ? robot.root : robot.links.get(`${SIDE}_base_link`);
  const points = [];
  for (const mesh of solids(shown)) {
    const box = mesh.geometry.boundingBox ?? (mesh.geometry.computeBoundingBox(), mesh.geometry.boundingBox);
    for (let c = 0; c < 8; c++) {
      points.push(new THREE.Vector3(c & 1 ? box.max.x : box.min.x, c & 2 ? box.max.y : box.min.y,
        c & 4 ? box.max.z : box.min.z).applyMatrix4(mesh.matrixWorld));
    }
  }
  for (const b of Object.values(base)) points.push(at(b.x, b.y, 0));
  if (!front) points.push(at(0, base[SIDE].y, 0));

  function frame() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    const right = new THREE.Vector3().crossVectors(camera.up, direction).normalize();
    const up = camera.up.clone().normalize();
    const lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
    let middle = new THREE.Vector3();
    for (const p of points) {
      const u = p.dot(right), v = p.dot(up);
      lo[0] = Math.min(lo[0], u); hi[0] = Math.max(hi[0], u);
      lo[1] = Math.min(lo[1], v); hi[1] = Math.max(hi[1], v);
      middle.add(p);
    }
    middle.divideScalar(points.length);
    const m = MARGIN[which];
    // The arms' extent, widened for the room the dimensions take, then to the
    // picture's proportions.
    let span = [(hi[0] - lo[0]) / (1 - m.left - m.right), (hi[1] - lo[1]) / (1 - m.top - m.bottom)];
    if (span[0] / span[1] < w / h) span[0] = span[1] * w / h; else span[1] = span[0] * h / w;
    const cu = (lo[0] + hi[0]) / 2 + (m.right - m.left) / 2 * span[0];
    const cv = (lo[1] + hi[1]) / 2 + (m.top - m.bottom) / 2 * span[1];
    const depth = middle.dot(direction);
    const centre = right.clone().multiplyScalar(cu).addScaledVector(up, cv).addScaledVector(direction, depth);
    camera.position.copy(centre).addScaledVector(direction, 2);
    camera.lookAt(centre);
    camera.left = -span[0] / 2; camera.right = span[0] / 2;
    camera.top = span[1] / 2; camera.bottom = -span[1] / 2;
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld(true);
    draw(w, h);
  }

  // A description point where it lands on the canvas, in CSS pixels.
  function screen(x, y, z) {
    const p = at(x, y, z).project(camera);
    return [(p.x + 1) / 2 * canvas.clientWidth, (1 - p.y) / 2 * canvas.clientHeight];
  }

  function draw(w, h) {
    overlay.setAttribute('viewBox', `0 0 ${w} ${h}`);
    overlay.replaceChildren();
    const add = (name, attrs, text) => {
      const el = document.createElementNS(SVG, name);
      for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
      if (text) el.textContent = text;
      overlay.append(el);
      return el;
    };
    const line = (a, b, cls) => add('line', { x1: a[0], y1: a[1], x2: b[0], y2: b[1], class: cls });
    const dimension = (a, b) => add('line', { x1: a[0], y1: a[1], x2: b[0], y2: b[1], class: 'dim',
      'marker-start': 'url(#arrow)', 'marker-end': 'url(#arrow)' });
    const tag = (p, text, cls = '') => {
      const label = add('text', { x: p[0], y: p[1], class: `value ${cls}`, 'text-anchor': 'middle', 'dominant-baseline': 'central' }, text);
      const box = label.getBBox();
      const back = document.createElementNS(SVG, 'rect');
      for (const [k, v] of Object.entries({ x: box.x - 6, y: box.y - 3, width: box.width + 12, height: box.height + 6, rx: 4, class: 'tag' })) back.setAttribute(k, String(v));
      overlay.insertBefore(back, label);
    };
    const defs = add('defs', {});
    defs.innerHTML = '<marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="currentColor"/></marker>';

    // The desk, the length of the picture.
    const deskY = screen(base[SIDE].x, base[SIDE].y, 0)[1];
    add('rect', { x: 0, y: deskY, width: w, height: h - deskY, class: 'desk' });
    line([0, deskY], [w, deskY], 'desk-edge');

    const mm = (m) => `${Math.round(m * 1000)} mm`;
    if (front) {
      // Both shoulder_pan axes, the spacing between them above the arms.
      const axes = Object.keys(pan).map((s) => screen(pan[s].x, pan[s].y, 0)[0]).sort((a, b) => a - b);
      const above = h * MARGIN.front.top * 0.5;
      for (const x of axes) line([x, above - 10], [x, deskY], 'axis');
      dimension([axes[0], above], [axes[1], above]);
      tag([(axes[0] + axes[1]) / 2, above], mm(spacing));
      // The height, beside the arm on the picture's left, up to its base's top face.
      const leftSide = Object.keys(pan).find((s) => screen(pan[s].x, pan[s].y, 0)[0] === axes[0]);
      const faceY = screen(base[leftSide].x, base[leftSide].y, top)[1];
      const x = w * MARGIN.front.left * 0.45;
      line([x - 12, faceY], [axes[0], faceY], 'ext');
      dimension([x, faceY], [x, deskY]);
      tag([x, (faceY + deskY) / 2], mm(top));
      const names = { left: pick('왼팔', 'Left arm'), right: pick('오른팔', 'Right arm') };
      for (const s of Object.keys(pan)) {
        const [px] = screen(pan[s].x, pan[s].y, 0);
        add('text', { x: px, y: deskY + 18, class: 'label', 'text-anchor': 'middle' }, names[s]);
      }
    } else {
      // The face the height goes up to, and the height beside it.
      const b = base[SIDE];
      const face = [screen(0.018, b.y, top), screen(pan[SIDE].x + 0.03, b.y, top)];
      const x = w * (1 - MARGIN.side.right * 0.45);
      line(face[0], [x + 12, face[0][1]], 'ext');
      line(face[0], face[1], 'face-halo');
      line(face[0], face[1], 'face');
      dimension([x, face[0][1]], [x, deskY]);
      tag([x, (face[0][1] + deskY) / 2], mm(top));
      add('text', { x: x - 10, y: face[0][1] - 8, class: 'label hot', 'text-anchor': 'end' }, pick('베이스 윗면', 'Base top'));
      const axis = screen(pan[SIDE].x, b.y, 0)[0];
      line([axis, h * MARGIN.side.top * 0.6], [axis, deskY], 'axis');
      add('text', { x: axis, y: h * MARGIN.side.top * 0.45, class: 'label dimmed', 'text-anchor': 'middle' }, pick('shoulder_pan 축', 'shoulder_pan axis'));
      add('text', { x: screen(0.009, b.y, 0)[0], y: deskY + 18, class: 'label', 'text-anchor': 'middle' }, pick('보드', 'Board'));
    }
  }

  picture.onResize = frame;
  frame();
  picture.redraw();
  figure.classList.add('ready');
}
