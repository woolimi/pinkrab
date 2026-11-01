// The home pose: every joint at the description's zero, the pose an arm is held
// in when it is calibrated. Both arms on their board, as they sit at home.

import * as THREE from 'three';
import { fitView, orbit } from './canvases.js';
import { LAYERS, solids, stage } from './stage.js';

// From in front of the board, a little to the side and above (three.js frame:
// +x out of the board, +y up, +z toward the right arm).
const VIEW = new THREE.Vector3(1.4, 0.7, 0.9);
const FOV = 30;
const MARGIN = 1.08;

export async function showHome(figure, sources) {
  const canvas = figure.querySelector('canvas');
  const { robot, world, addView } = await stage(sources);
  const pose = Object.fromEntries(robot.movingJoints().map(({ name }) => [name, 0]));

  const camera = new THREE.PerspectiveCamera(FOV, 1, 0.01, 20);
  camera.layers.enable(LAYERS.otherArm);
  camera.layers.enable(LAYERS.board);
  const { controls, moving } = orbit(camera, canvas, () => picture.redraw());
  const picture = addView(canvas, { camera, pose, step: moving });

  // Framed on the two arms at home.
  robot.setPose(pose);
  world.updateMatrixWorld(true);
  const points = [];
  for (const mesh of solids(robot.root)) {
    const box = mesh.geometry.boundingBox ?? (mesh.geometry.computeBoundingBox(), mesh.geometry.boundingBox);
    for (let c = 0; c < 8; c++) {
      points.push(new THREE.Vector3(c & 1 ? box.max.x : box.min.x, c & 2 ? box.max.y : box.min.y,
        c & 4 ? box.max.z : box.min.z).applyMatrix4(mesh.matrixWorld));
    }
  }
  function place(direction) {
    const { target, distance } = fitView(points, direction, camera, MARGIN);
    controls.target.copy(target);
    camera.position.copy(target).addScaledVector(direction, distance);
  }
  picture.onResize = () => place(camera.position.clone().sub(controls.target).normalize());
  place(VIEW.clone().normalize());
  picture.redraw();
  figure.classList.add('ready');
}
