// A marker solid's paper and markers, drawn on its faces: the handhelds'
// solids and the desk's. What /hermit and /desk_solid send as `markers` - each
// marked face's pose in the solid's frame and its marker's modules - read the
// way the detector reads them: the module grid's first row along the face's
// +y edge, its first column along -x.

import * as THREE from 'three';

// Just off the face, over its pocket: the paper, and the marker on it.
const PAPER_LIFT = 0.0004;
const MARKER_LIFT = 0.0006;
const PAPER = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.95 });

// One marker as a texture: a block of pixels a module, sampled without
// smoothing so the squares stay squares.
function markerTexture(modules) {
  const px = 12;
  const side = modules.length;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = side * px;
  const g = canvas.getContext('2d');
  g.fillStyle = '#fff';
  g.fillRect(0, 0, canvas.width, canvas.height);
  g.fillStyle = '#000';
  modules.forEach((row, r) => row.forEach((black, c) => { if (black) g.fillRect(c * px, r * px, px, px); }));
  const texture = new THREE.CanvasTexture(canvas);
  texture.magFilter = THREE.NearestFilter;
  texture.minFilter = THREE.NearestFilter;
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

/** Lay `markers` on `solid`, a group in the solid's own frame. */
export function addMarkers(solid, markers) {
  const sheet = new THREE.PlaneGeometry(markers.patch_m, markers.patch_m);
  const square = new THREE.PlaneGeometry(markers.marker_m, markers.marker_m);
  for (const face of markers.faces) {
    const holder = new THREE.Group();
    new THREE.Matrix4().set(...face.face_T.flat()).decompose(holder.position, holder.quaternion, holder.scale);
    const paper = new THREE.Mesh(sheet, PAPER);
    paper.position.z = PAPER_LIFT;
    const marker = new THREE.Mesh(square,
      new THREE.MeshStandardMaterial({ map: markerTexture(face.modules), roughness: 0.9 }));
    marker.position.z = MARKER_LIFT;
    holder.add(paper, marker);
    solid.add(holder);
  }
}
