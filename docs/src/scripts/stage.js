// One scene and one arm for every picture of the arm on a page. Each picture is
// a view (canvases.js) with its own camera, its own pose for the arm and its
// own marks, so the arm's meshes are loaded and held on the GPU once.

import * as THREE from 'three';
import { loadURDF } from '../../../src/pinkrab_studio/pinkrab_studio/static/urdf.js';
import { HULL, OUTLINE, materialFor, pegboard } from '../../../src/pinkrab_studio/pinkrab_studio/static/look.js';
import { addView, context } from './canvases.js';

// The arm a one-arm picture shows; the other is its mirror.
export const SIDE = 'left';
// What a view's camera turns on to see more than that arm: the other arm,
// and the board where the scene has it. Each view's own marks are on a layer
// of their own above these.
export const LAYERS = { otherArm: 1, board: 2 };

let made = null;

/** The page's stage, made the first time a view asks for it. */
export function stage(sources) {
  made ??= build(sources);
  return made;
}

async function build(sources) {
  const renderer = context();
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

  const [physics, model, ready] = await Promise.all([
    fetch(sources.physics).then((r) => r.json()),
    fetch(sources.model).then((r) => { if (!r.ok) throw new Error('no model'); return r.text(); }),
    fetch(sources.ready).then((r) => r.json()),
  ]);
  const robot = await loadURDF(model, sources.meshes, { materialFor, outlineMaterial: OUTLINE, hullMaterial: HULL });
  robot.setPose(ready);
  world.add(robot.root);

  // Everything not on this arm is on the other arm's layer, for the views that ask for it.
  const base = robot.links.get(`${SIDE}_base_link`);
  if (!base) throw new Error(`no ${SIDE} arm`);
  const onArm = (object) => { for (let o = object; o; o = o.parent) if (o === base) return true; return false; };
  robot.root.traverse((o) => { if ((o.isMesh || o.isLine) && !onArm(o)) o.layers.set(LAYERS.otherArm); });
  // The board, once: a view that wants it elsewhere draws a copy of this one,
  // which shares its geometry and its texture.
  const slab = physics?.scenery?.pegboard ?? null;
  const board = slab ? pegboard(slab, renderer.capabilities.getMaxAnisotropy()) : null;
  if (board) {
    board.layers.set(LAYERS.board);
    world.add(board);
  }
  world.updateMatrixWorld(true);

  /** A picture of the arm on `canvas`, its pose `pose` (changed in place); see canvases.addView. */
  const addArmView = (canvas, { camera, pose, step }) => addView(canvas, {
    scene, camera, step,
    before() { robot.setPose(pose); world.updateMatrixWorld(true); },
  });

  return { robot, world, slab, board, desk: physics?.scenery?.desk ?? null, ready: robot.pose(), addView: addArmView };
}

/** The parts of the arm under `link`, the solids only: no outline, no silhouette. */
export function solids(link) {
  const out = [];
  link.traverse((o) => { if (o.isMesh && o.visible && o.userData.urdfLink) out.push(o); });
  return out;
}
