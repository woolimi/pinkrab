// How the robot looks: the finish of its parts, its outline, its silhouette.
// Shared by the teleop page and the replay, so the arm played back is drawn
// the way the arm being driven is.
import * as THREE from 'three';

// Colour comes from the description - it is a fact about the robot, and RViz reads
// the same numbers. What is decided here is the finish, which URDF has no way of
// saying: printed plastic scatters, a servo case is smoother and slightly metallic.
export const FINISH = {
  shell: { roughness: 0.54, metalness: 0.02 },
  hardware: { roughness: 0.6, metalness: 0.12 },
};
// Stood in when a description states no colour for a part. This one always does;
// a page that drew nothing rather than guess would be harder to debug than one
// that draws the part in the shell colour and lets you see it is unpainted.
export const UNPAINTED = { shell: 0xe04b8e, hardware: 0x1a1a1a };
export const OUTLINE = new THREE.LineBasicMaterial({ color: 0x4a1030 });

// The silhouette. Each link is drawn a second time inside out, with every vertex
// pushed along its normal by a distance that grows with how far away it is - so the
// line keeps the same weight on screen whether the arm is under your nose or across
// the desk. WebGL will not draw a line thicker than one pixel, which is why the
// outline has to be geometry rather than a line at all.
export const OUTLINE_PX = 0.0018;   // radians of view angle, ~2 px at this field of view
export const HULL = new THREE.ShaderMaterial({
  uniforms: { outlineColor: { value: new THREE.Color(0x2b0a03) }, width: { value: OUTLINE_PX } },
  vertexShader: `
    uniform float width;
    // Welded in urdf.js. The mesh's own per-face normals would tear this shell open
    // along every hard edge; see addSmoothNormals.
    attribute vec3 smoothNormal;
    void main() {
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      vec3 n = normalize(normalMatrix * smoothNormal);
      // -mv.z is the distance in front of the camera, so this is a constant angle
      // rather than a constant length.
      mv.xyz += n * width * max(-mv.z, 0.05);
      gl_Position = projectionMatrix * mv;
    }`,
  fragmentShader: `
    uniform vec3 outlineColor;
    void main() { gl_FragColor = vec4(outlineColor, 1.0); }`,
  side: THREE.BackSide,
});

// One material per colour and finish, not per mesh: twenty-two parts share two
// materials, and a fresh one for each would be twenty-two draw calls that cannot
// be batched.
const materials = new Map();

export function materialFor(file, colour) {
  const servo = file.startsWith('servo');
  const kind = servo ? 'hardware' : 'shell';
  const key = `${kind}:${colour ? colour.getHexString() : 'none'}`;
  let material = materials.get(key);
  if (!material) {
    material = new THREE.MeshStandardMaterial({
      color: colour ?? new THREE.Color(UNPAINTED[kind]),
      ...FINISH[kind],
    });
    materials.set(key, material);
  }
  return material;
}

// A SKADIS-style pegboard: rounded vertical slots on a DIAMOND lattice, not a
// square one. Every other column drops half a pitch, so each slot sits between the
// two beside it and the eye reads diagonals rather than rows and columns. That
// stagger is what the real board looks like, and it is also what lets a hook span
// two holes at an angle.
//
// Drawn rather than cut, because the holes are 5 mm on a board seen from half a
// metre away and punching a thousand of them through a box would cost real geometry
// to say what a texture says for nothing. The pattern is the whole board in one
// canvas, not a tile repeated - there is no seam to line up and the lattice lands
// where the board's own edges are.
const PEG_PITCH = 0.020;    // m, centre to centre both ways
const PEG_SLOT_LONG = 0.010;   // leaves as much board between two slots as slot
const PEG_SLOT_WIDE = 0.005;
const PX_PER_M = 2600;      // ~2.6 px/mm: a 5 mm slot lands on 13 px

export function pegboardTexture(width, height, anisotropy = 1) {
  // BoxGeometry's +x face runs u along the box's z and v along its y, so the canvas
  // is turned a quarter relative to the board as you look at it: canvas x is world
  // z (up), canvas y is world y (across). The slots are long in world z, which is
  // long in canvas x.
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(height * PX_PER_M);
  canvas.height = Math.round(width * PX_PER_M);
  const g = canvas.getContext('2d');

  g.fillStyle = '#1c1c1f';
  g.fillRect(0, 0, canvas.width, canvas.height);

  const pitch = PEG_PITCH * PX_PER_M;
  const long = PEG_SLOT_LONG * PX_PER_M;
  const wide = PEG_SLOT_WIDE * PX_PER_M;
  const acrossZ = Math.floor(height / PEG_PITCH);
  const acrossY = Math.floor(width / PEG_PITCH);
  // Centre the lattice on the board so the margin is equal at both edges, the way a
  // cut sheet of pegboard actually reads.
  const originX = (canvas.width - (acrossZ - 1) * pitch) / 2;
  const originY = (canvas.height - (acrossY - 1) * pitch) / 2;

  for (let i = 0; i < acrossZ; i += 1) {
    for (let j = 0; j < acrossY; j += 1) {
      // The stagger. A dropped column runs one slot past the bottom edge, so the
      // row count stays put and the overhanging slot is simply not drawn.
      const cx = originX + i * pitch + (j % 2 ? pitch / 2 : 0);
      const cy = originY + j * pitch;
      if (cx + long / 2 > canvas.width) continue;
      // The hole, then a hairline of the board's own colour along one side of it:
      // a 5 mm slot in a flat black face is invisible without something to say it
      // has a depth, and a lit edge is the cheapest thing that says it.
      g.fillStyle = '#000000';
      g.beginPath();
      g.roundRect(cx - long / 2, cy - wide / 2, long, wide, wide / 2);
      g.fill();
      g.strokeStyle = 'rgba(255, 255, 255, 0.13)';
      g.lineWidth = Math.max(1, wide * 0.11);
      g.beginPath();
      g.roundRect(cx - long / 2, cy - wide / 2 + wide * 0.16, long, wide, wide / 2);
      g.stroke();
    }
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.anisotropy = anisotropy;
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

/** The board as the scene states it - `slab` is its centre and half extents,
 *  the way MuJoCo has them - perforated on its two large faces. */
export function pegboard(slab, anisotropy = 1) {
  const [hx, hy, hz] = slab.half;
  const face = new THREE.MeshStandardMaterial({ roughness: 0.82, map: pegboardTexture(hy * 2, hz * 2, anisotropy) });
  const edge = new THREE.MeshStandardMaterial({ color: 0x141416, roughness: 0.85 });
  // BoxGeometry's faces in order: +x, -x, +y, -y, +z, -z.
  const board = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), [face, face, edge, edge, edge, edge]);
  board.scale.set(hx * 2, hy * 2, hz * 2);
  board.position.set(...slab.pos);
  board.castShadow = true;
  board.receiveShadow = true;
  return board;
}
