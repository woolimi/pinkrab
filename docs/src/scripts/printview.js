// The printing page's pictures: a project's plates on the bed, a tab a plate,
// and a tab a project where a figure offers several.

import * as THREE from 'three';
import { addView, fitView, orbit } from './canvases.js';
import { bedOf, partsOf, readProject } from './plates.js';
import { pick } from './lang.js';

// In the bed's frame (z up, y toward the back): from in front, a little to the
// right and above, the way one stands at the printer.
const VIEW = new THREE.Vector3(0.35, -1.0, 0.95);
const FOV = 30;
const MARGIN = 1.06;

export async function showPrint(figure) {
  const canvas = figure.querySelector('canvas');
  const fileTabs = [...figure.querySelectorAll('[data-file]')];
  const plateBar = figure.querySelector('.plates');
  const download = figure.querySelector('a.download');

  const scene = new THREE.Scene();
  scene.add(new THREE.HemisphereLight(0xfffaf0, 0xcbbfa8, 1.6));
  const key = new THREE.DirectionalLight(0xfff6e8, 2.0);
  key.position.set(1.2, 2.4, 1.6);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xffeef6, 0.6);
  fill.position.set(-1.2, 0.8, -1.0);
  scene.add(fill);
  // Millimetres and z up, as the project is written.
  const world = new THREE.Group();
  world.rotation.x = -Math.PI / 2;
  world.scale.setScalar(0.001);
  scene.add(world);

  const camera = new THREE.PerspectiveCamera(FOV, 1, 0.01, 20);
  const { controls, moving } = orbit(camera, canvas, () => picture.redraw());
  const picture = addView(canvas, { scene, camera, step: moving });

  const projects = new Map();   // file -> the project read, once
  let bed = null;
  let shown = null;
  let current = null;

  function frame(project, plate, direction) {
    world.updateMatrixWorld(true);
    const [x, y] = project.bed;
    let top = 0;
    for (const { geometry, matrix } of project.plates[plate].parts) {
      geometry.boundingBox ?? geometry.computeBoundingBox();
      top = Math.max(top, geometry.boundingBox.clone().applyMatrix4(matrix).max.z);
    }
    const points = [];
    for (const z of [0, top]) for (const [px, py] of [[0, 0], [x, 0], [0, y], [x, y]]) {
      points.push(new THREE.Vector3(px, py, z).applyMatrix4(world.matrixWorld));
    }
    const { target, distance } = fitView(points, direction, camera, MARGIN);
    controls.target.copy(target);
    camera.position.copy(target).addScaledVector(direction, distance);
  }
  const looking = () => camera.position.clone().sub(controls.target).normalize();

  function showPlate(project, plate) {
    if (shown) world.remove(shown);
    project.drawn[plate] ??= partsOf(project.plates[plate]);
    shown = project.drawn[plate];
    world.add(shown);
    for (const [i, tab] of [...plateBar.children].entries()) tab.setAttribute('aria-pressed', String(i === plate));
    frame(project, plate, current ? looking() : VIEW.clone().normalize().applyQuaternion(world.quaternion));
    current = { project, plate };
    picture.redraw();
  }

  async function showFile(file) {
    figure.classList.add('loading');
    for (const tab of fileTabs) tab.setAttribute('aria-pressed', String(tab.dataset.file === file));
    download.href = file;
    download.querySelector('span').textContent = pick(`${file.split('/').pop()} 내려받기`, `Download ${file.split('/').pop()}`);
    if (!projects.has(file)) {
      const answer = await fetch(file);
      if (!answer.ok) throw new Error(`no ${file}`);
      projects.set(file, { ...readProject(await answer.arrayBuffer()), drawn: [] });
    }
    const project = projects.get(file);
    if (!bed) {
      bed = bedOf(project.bed);
      world.add(bed);
    }
    plateBar.replaceChildren(...project.plates.map((plate, i) => {
      const tab = document.createElement('button');
      tab.type = 'button';
      tab.textContent = `plate ${i + 1}`;
      tab.addEventListener('click', () => showPlate(project, i));
      return tab;
    }));
    showPlate(project, 0);
    figure.classList.remove('loading');
  }

  picture.onResize = () => { if (current) frame(current.project, current.plate, looking()); };
  for (const tab of fileTabs) tab.addEventListener('click', () => showFile(tab.dataset.file));
  await showFile(fileTabs[0].dataset.file);
  figure.classList.add('ready');
}
