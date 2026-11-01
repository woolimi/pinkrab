// The reach in the introduction, as the studio's reach switch shows it: both
// arms on their board in the ready pose, and on the desk each arm's patch -
// every cell the tool reaches at grasping height in a grasping attitude, solved
// by the studio and exported beside the description.

import * as THREE from 'three';
import { fitView, orbit } from './canvases.js';
import { LAYERS, solids, stage } from './stage.js';

// The studio's: its camera's direction (three.js frame: +x out of the board,
// +y up, +z toward the right arm), its desk and grid, and each arm's colour.
const VIEW = new THREE.Vector3(1.5, 0.85, 1.35);
const FOV = 30;
const MARGIN = 1.05;
const DESK = { size: [2.4, 1.6], colour: 0xefe9dd };
const GRID = { size: 2.4, cells: 24, colours: [0xc9bda6, 0xdbd2c1] };
const REACH_COLOUR = { left: 0x1668c9, right: 0xe2488e };
const REACH_OPACITY = 0.45;

// A layer of cells on the desk, one tile each, as the studio lays them.
function tilesOf(cells, size, lift, colour) {
  const material = new THREE.MeshStandardMaterial({
    color: colour, transparent: true, opacity: REACH_OPACITY, roughness: 0.9, depthWrite: false,
  });
  const tiles = new THREE.InstancedMesh(new THREE.BoxGeometry(size, size, 0.002), material, cells.length);
  const place = new THREE.Object3D();
  cells.forEach(([x, y], i) => {
    place.position.set(x, y, lift);
    place.updateMatrix();
    tiles.setMatrixAt(i, place.matrix);
  });
  tiles.instanceMatrix.needsUpdate = true;
  tiles.renderOrder = 1;
  return tiles;
}

export async function showReach(figure, sources) {
  const canvas = figure.querySelector('canvas');
  const [{ robot, world, ready, addView }, reach] = await Promise.all([
    stage(sources),
    fetch(sources.workspace).then((r) => { if (!r.ok) throw new Error('no workspace'); return r.json(); }),
  ]);

  const marks = new THREE.Group();
  const desk = new THREE.Mesh(
    new THREE.PlaneGeometry(...DESK.size),
    new THREE.MeshStandardMaterial({ color: DESK.colour, roughness: 0.95 }),
  );
  desk.position.z = reach.desk_z;
  marks.add(desk);
  const grid = new THREE.GridHelper(GRID.size, GRID.cells, ...GRID.colours);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = reach.desk_z + 0.001;   // just off the desk, so it is not z-fighting with it
  marks.add(grid);
  const lift = reach.desk_z + 0.0015;       // just off the desk, over the grid lines
  for (const [arm, patch] of Object.entries(reach.arms)) {
    if (patch.cells.length) marks.add(tilesOf(patch.cells, reach.cell_m, lift, REACH_COLOUR[arm] ?? 0x888888));
    const area = figure.querySelector(`[data-area="${arm}"]`);
    if (area) area.textContent = `${Math.round(patch.area_cm2).toLocaleString('ko-KR')} cm²`;
  }
  world.add(marks);

  const pose = { ...ready };
  const camera = new THREE.PerspectiveCamera(FOV, 1, 0.01, 20);
  camera.layers.enable(LAYERS.otherArm);
  camera.layers.enable(LAYERS.board);
  const { controls, moving } = orbit(camera, canvas, () => picture.redraw());

  // Framed on the two arms and their patches.
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
  for (const patch of Object.values(reach.arms)) {
    for (const [x, y] of patch.cells) points.push(new THREE.Vector3(x, y, lift).applyMatrix4(world.matrixWorld));
  }
  function place(direction) {
    const { target, distance } = fitView(points, direction, camera, MARGIN);
    controls.target.copy(target);
    camera.position.copy(target).addScaledVector(direction, distance);
  }

  const picture = addView(canvas, { camera, pose, step: moving });
  picture.own(marks);
  picture.onResize = () => place(camera.position.clone().sub(controls.target).normalize());
  place(VIEW.clone().normalize());
  picture.redraw();
  figure.classList.add('ready');
}
