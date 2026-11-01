// One WebGL context for every 3D picture on a page.
//
// Each picture is a view: a canvas of its own, a scene and a camera. Views are
// drawn into the one context and copied onto their canvases, so a page with
// several pictures keeps one context and holds a mesh two pictures share on
// the GPU once. A view is drawn only while it is on screen and only when
// something in it has changed.

import * as THREE from 'three';
import { OrbitControls } from '../../../src/pinkrab_studio/pinkrab_studio/static/OrbitControls.js';

const MAX_PIXEL_RATIO = 2;
// Layers below this are a scene's own to give out (stage.js does); each view
// takes one above it for the marks only it draws.
const FIRST_VIEW_LAYER = 8;

let renderer = null;
const views = new Set();
let nextLayer = FIRST_VIEW_LAYER;
let running = false;
let last = 0;
let width = 0, height = 0;

/** The page's renderer, off the page: the views' canvases are what is seen. */
export function context() {
  if (!renderer) {
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(1);   // sizes below are in device pixels already
    renderer.setScissorTest(true);
  }
  return renderer;
}

function draw(view) {
  const w = view.canvas.width, h = view.canvas.height;
  if (!w || !h) return;
  if (w > width || h > height) {
    width = Math.max(width, w);
    height = Math.max(height, h);
    renderer.setSize(width, height, false);
  }
  view.before?.();
  view.scene.background = view.background;
  // WebGL counts rows from the bottom; the copy below takes the same corner.
  renderer.setViewport(0, 0, w, h);
  renderer.setScissor(0, 0, w, h);
  renderer.render(view.scene, view.camera);
  view.context.drawImage(renderer.domElement, 0, height - h, w, h, 0, 0, w, h);
}

function frame(now) {
  const dt = Math.min((now - last) / 1000, 0.05);
  last = now;
  let moving = false;
  for (const view of views) {
    if (!view.visible) continue;
    const still = view.step ? !view.step(now, dt) : true;
    if (view.dirty || !still) draw(view);
    view.dirty = false;
    moving ||= !still;
  }
  running = moving;
  if (running) requestAnimationFrame(frame);
}

function wake() {
  if (running) return;
  running = true;
  last = performance.now();
  requestAnimationFrame(frame);
}

/**
 * A picture of `scene` through `camera` on `canvas`. `before()` sets the scene
 * up for this view just before it is drawn; `step(now, dt)` moves what moves
 * and says whether anything still is. Returns the view: `view.redraw()` asks
 * for one more drawing, `view.own(object)` puts an object in this view only.
 */
export function addView(canvas, { scene, camera, step, before }) {
  context();
  const layer = nextLayer++;
  camera.layers.enable(layer);
  const view = {
    canvas, scene, camera, step, before, layer, visible: false, dirty: true,
    context: canvas.getContext('2d'),
    background: new THREE.Color(getComputedStyle(canvas.parentElement).backgroundColor),
    redraw() { view.dirty = true; if (view.visible) wake(); },
    own(object) { object.traverse((o) => o.layers.set(layer)); return object; },
  };
  const resize = () => {
    const ratio = Math.min(devicePixelRatio, MAX_PIXEL_RATIO);
    const w = Math.round(canvas.clientWidth * ratio), h = Math.round(canvas.clientHeight * ratio);
    if (!w || !h) return;
    canvas.width = w;
    canvas.height = h;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    view.onResize?.();
    view.redraw();
  };
  new ResizeObserver(resize).observe(canvas);
  new IntersectionObserver(([entry]) => {
    view.visible = entry.isIntersecting;
    if (view.visible) view.redraw();
  }).observe(canvas);
  views.add(view);
  resize();
  return view;
}

// The most one wheel notch zooms by, in its deltaY: a mouse's notch is about
// 100, a pinch's many small events a few each.
const WHEEL_STEP = 60;

/**
 * The view's camera turned round its target by dragging, on a mouse only: on a
 * phone a drag over a picture scrolls the page, as it does everywhere else.
 * `moving(now)` says whether the drag, or the damping after it, still moves it.
 * With `zoom`, the wheel with Ctrl or ⌘ brings it closer or takes it back, as
 * `zoomBy(factor)` does for buttons, between the controls' min and max distance.
 */
export function orbit(camera, canvas, redraw, { zoom = false } = {}) {
  const controls = new OrbitControls(camera, canvas);
  controls.enableZoom = false;
  controls.enablePan = false;
  controls.enableDamping = true;
  controls.enabled = matchMedia('(pointer: fine)').matches;
  canvas.style.touchAction = controls.enabled ? 'none' : 'auto';
  let dragging = false;
  let settling = 0;
  controls.addEventListener('start', () => { dragging = true; redraw(); });
  controls.addEventListener('end', () => { dragging = false; settling = performance.now() + 1000; });
  const moving = (now) => Boolean(controls.update() || dragging || now < settling);
  // Closer to the target, or farther, by `factor` of the way it is now,
  // within the controls' distances.
  const zoomBy = (factor) => {
    const offset = camera.position.clone().sub(controls.target);
    const distance = THREE.MathUtils.clamp(offset.length() * factor, controls.minDistance, controls.maxDistance);
    camera.position.copy(controls.target).addScaledVector(offset.normalize(), distance);
    settling = performance.now() + 300;
    redraw();
  };
  // A plain wheel scrolls the page past the picture; with Ctrl (a trackpad's
  // pinch is one) or ⌘ it zooms.
  if (zoom) {
    canvas.addEventListener('wheel', (event) => {
      if (!event.ctrlKey && !event.metaKey) return;
      event.preventDefault();
      zoomBy(Math.exp(THREE.MathUtils.clamp(event.deltaY, -WHEEL_STEP, WHEEL_STEP) * 0.005));
    }, { passive: false });
  }
  return { controls, moving, zoomBy };
}

/**
 * Where `camera` looks and how far back along `direction` (a unit vector,
 * from the target toward the camera) it stands for every one of `points` to
 * be in its picture: the middle of the points as seen from there, then the
 * distance the farthest of them needs. `margin` above 1 leaves room round them.
 */
export function fitView(points, direction, camera, margin = 1) {
  const tanV = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) / margin;
  const tanH = tanV * camera.aspect;
  const right = new THREE.Vector3().crossVectors(camera.up, direction).normalize();
  const up = new THREE.Vector3().crossVectors(direction, right);
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  const seen = points.map((p) => [p.dot(right), p.dot(up), p.dot(direction)]);
  for (const q of seen) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], q[k]); hi[k] = Math.max(hi[k], q[k]); }
  const mid = lo.map((l, k) => (l + hi[k]) / 2);
  let distance = 0;
  for (const [x, y, z] of seen) {
    distance = Math.max(distance, z - mid[2] + Math.abs(x - mid[0]) / tanH, z - mid[2] + Math.abs(y - mid[1]) / tanV);
  }
  const target = new THREE.Vector3()
    .addScaledVector(right, mid[0]).addScaledVector(up, mid[1]).addScaledVector(direction, mid[2]);
  return { target, distance };
}
