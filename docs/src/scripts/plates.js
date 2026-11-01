// A Bambu Studio project as it lies on the printer's bed: each plate's parts,
// placed and turned as the project has them, read straight out of the 3MF the
// page offers for download - so the picture is the file.
//
// A 3MF is a zip. Its 3D/3dmodel.model places each object on the build space
// (build items) and names the mesh files its parts are in (components, the
// production extension); Metadata/model_settings.config says which plate an
// object is on and which filament each part prints in; project_settings.config
// gives the bed's size and the filaments' colours. Bambu Studio lays its plates
// out on the build space in a grid, a fifth of a bed apart.

import * as THREE from 'three';
import { strFromU8, unzipSync } from 'three/examples/jsm/libs/fflate.module.js';
import { drawPart } from '../../../src/pinkrab_studio/pinkrab_studio/static/urdf.js';
import { FINISH, HULL, OUTLINE } from '../../../src/pinkrab_studio/pinkrab_studio/static/look.js';

const PLATE_GAP = 1 / 5;
const BED = { colour: 0x2b2b30, line: 0x45454c, thickness: 3, grid: 50 };   // mm

const xml = (text) => new DOMParser().parseFromString(text, 'application/xml');
const meta = (el, key) => el.querySelector(`:scope > metadata[key="${key}"]`)?.getAttribute('value') ?? null;

// A 3MF transform is a 4x3 matrix written row by row, points as row vectors.
function matrix(text) {
  if (!text) return new THREE.Matrix4();
  const m = text.trim().split(/\s+/).map(Number);
  return new THREE.Matrix4().set(m[0], m[3], m[6], m[9], m[1], m[4], m[7], m[10], m[2], m[5], m[8], m[11], 0, 0, 0, 1);
}

// The meshes in one model file, by object id. Read with expressions rather than
// a DOM: a part is tens of thousands of elements.
function meshes(text) {
  const out = new Map();
  for (const [, id, body] of text.matchAll(/<object id="(\d+)"[^>]*>([\s\S]*?)<\/object>/g)) {
    if (!body.includes('<mesh>')) continue;
    const vertices = [];
    for (const [, x, y, z] of body.matchAll(/<vertex x="([^"]+)" y="([^"]+)" z="([^"]+)"/g)) vertices.push(+x, +y, +z);
    const index = [];
    for (const [, a, b, c] of body.matchAll(/<triangle v1="(\d+)" v2="(\d+)" v3="(\d+)"/g)) index.push(+a, +b, +c);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    geometry.setIndex(index);
    // Flat faces, as a slicer shows a part: the welded mesh would shade its
    // creases round. Edges and the silhouette are drawn over it (drawPart).
    out.set(id, geometry.toNonIndexed());
    out.get(id).computeVertexNormals();
  }
  return out;
}

/** The project in `buffer`: its bed size and, plate by plate, its parts (geometry, matrix, filament colour). */
export function readProject(buffer) {
  const files = unzipSync(new Uint8Array(buffer));
  const text = (path) => strFromU8(files[path.replace(/^\//, '')]);
  const settings = JSON.parse(text('Metadata/project_settings.config'));
  const [bedX, bedY] = settings.printable_area[2].split('x').map(Number);
  const colours = settings.filament_colour ?? [];

  const config = xml(text('Metadata/model_settings.config'));
  const filamentOf = new Map();   // "<object>/<part>" -> filament number
  for (const object of config.querySelectorAll('config > object')) {
    const own = meta(object, 'extruder') ?? '1';
    for (const part of object.querySelectorAll(':scope > part')) {
      filamentOf.set(`${object.getAttribute('id')}/${part.getAttribute('id')}`, Number(meta(part, 'extruder') ?? own));
    }
  }
  const plateOf = new Map();
  const plates = [...config.querySelectorAll('config > plate')];
  plates.forEach((plate, i) => {
    for (const instance of plate.querySelectorAll('model_instance')) plateOf.set(meta(instance, 'object_id'), i);
  });

  const model = xml(text('3D/3dmodel.model'));
  const files3d = new Map();
  const meshesIn = (path) => {
    if (!files3d.has(path)) files3d.set(path, meshes(text(path)));
    return files3d.get(path);
  };
  const objects = new Map([...model.querySelectorAll('resources > object')].map((o) => [o.getAttribute('id'), o]));

  const columns = Math.ceil(Math.sqrt(plates.length));
  const placed = plates.map(() => []);
  for (const item of model.querySelectorAll('build > item')) {
    const id = item.getAttribute('objectid');
    const plate = plateOf.get(id);
    if (plate === undefined) continue;
    // From the build space to this plate's own corner.
    const corner = new THREE.Matrix4().makeTranslation(
      -(plate % columns) * bedX * (1 + PLATE_GAP), Math.floor(plate / columns) * bedY * (1 + PLATE_GAP), 0);
    const onPlate = corner.multiply(matrix(item.getAttribute('transform')));
    for (const component of objects.get(id).querySelectorAll('components > component')) {
      const part = component.getAttribute('objectid');
      const geometry = meshesIn(component.getAttribute('p:path')).get(part);
      if (!geometry) continue;
      const filament = filamentOf.get(`${id}/${part}`) ?? 1;
      placed[plate].push({
        geometry,
        matrix: onPlate.clone().multiply(matrix(component.getAttribute('transform'))),
        colour: colours[filament - 1] ?? '#888888',   // the filament it prints in
      });
    }
  }
  return { bed: [bedX, bedY], plates: placed.map((parts, i) => ({ name: meta(plates[i], 'plater_name') || '', parts })) };
}

/** The bed: the plate itself, a grid on it, millimetres, its top at z = 0. */
export function bedOf([x, y]) {
  const bed = new THREE.Group();
  const slab = new THREE.Mesh(new THREE.BoxGeometry(x, y, BED.thickness),
    new THREE.MeshStandardMaterial({ color: BED.colour, roughness: 0.9 }));
  slab.position.set(x / 2, y / 2, -BED.thickness / 2);
  bed.add(slab);
  const lines = [];
  for (let at = BED.grid; at < x; at += BED.grid) lines.push(at, 0, 0.1, at, y, 0.1);
  for (let at = BED.grid; at < y; at += BED.grid) lines.push(0, at, 0.1, x, at, 0.1);
  const grid = new THREE.BufferGeometry();
  grid.setAttribute('position', new THREE.Float32BufferAttribute(lines, 3));
  bed.add(new THREE.LineSegments(grid, new THREE.LineBasicMaterial({ color: BED.line })));
  return bed;
}

const materials = new Map();
/** A plate's parts as drawn objects, in the robot's finish, outline and silhouette. */
export function partsOf(plate) {
  const group = new THREE.Group();
  for (const { geometry, matrix: m, colour } of plate.parts) {
    if (!materials.has(colour)) {
      materials.set(colour, new THREE.MeshStandardMaterial({ color: new THREE.Color(colour), ...FINISH.shell }));
    }
    const holder = new THREE.Group();
    holder.matrixAutoUpdate = false;
    holder.matrix.copy(m);
    drawPart(holder, geometry, materials.get(colour), { outlineMaterial: OUTLINE, hullMaterial: HULL });
    group.add(holder);
  }
  return group;
}
