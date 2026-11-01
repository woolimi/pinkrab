// The Pinkrab studio page: show the robot, pose it, compare it against the servos.

import { MODE_HANDHELD, MODE_IDLE, MODE_STUDIO } from './modes.js';
import * as THREE from 'three';
import { OrbitControls } from './OrbitControls.js';
import { loadURDF } from './urdf.js';
import { enableDrag, greyHandles } from './ik.js';
import { connect as connectPhysics } from './physics.js';
import { Dash } from './dash.js';
import { showHands } from './hands.js';
import { HULL, OUTLINE, materialFor, pegboardTexture } from './look.js';

// The panes and the bars between them, laid out before anything is drawn in
// them: the canvas takes its size from its pane, and the pane has to be in the
// tree before it has one.
const dash = new Dash();
window.pinkrabDash = dash;   // dash.reset() from the console puts the layout back

const canvas = document.getElementById('view');
const statusEl = document.getElementById('status');
const whichEl = document.getElementById('which');
const jointsEl = document.getElementById('joints');

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
// A light scene needs cast shadows in a way a dark one does not. Without them
// nothing sits ON anything: the arms and the desk are both pale, and the one
// question this view exists to answer - how high is the hand above the desk - has
// no visual answer left.
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

// Paper, not a void. Everything below is picked to sit under a red robot: warm
// neutrals, nothing saturated enough to argue with the shell.
const PAPER = 0xf9f1f4;
const scene = new THREE.Scene();
scene.background = new THREE.Color(PAPER);

// URDF is z-up; three.js defaults to y-up. Rotating the world once here means every
// coordinate on screen still reads the way the description writes it.
const world = new THREE.Group();
world.rotation.x = -Math.PI / 2;
scene.add(world);

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
camera.position.set(0.9, 0.7, 0.9);
// Leave camera.up alone. The world group already turns the description's z-up into
// three.js's y-up, so overriding up to z put OrbitControls at odds with the scene
// and the view would not rotate at all.

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;

// Lights are placed in scene coordinates, where the world group has already turned
// the description's z-up into three.js's y-up: +y is up, +x is out of the board.
scene.add(new THREE.HemisphereLight(0xfffaf0, 0xcbbfa8, 1.5));

const key = new THREE.DirectionalLight(0xfff6e8, 1.9);
key.position.set(1.35, 2.1, 0.95);
key.castShadow = true;
// The whole rig fits inside a metre or so. Sizing the shadow camera to it rather
// than to some generous default is what keeps a 2048 map sharp enough to read the
// gap between a claw and the desk.
key.shadow.mapSize.set(2048, 2048);
key.shadow.camera.left = -1.0;
key.shadow.camera.right = 1.0;
key.shadow.camera.top = 1.0;
key.shadow.camera.bottom = -1.0;
key.shadow.camera.near = 0.2;
key.shadow.camera.far = 6;
key.shadow.bias = -0.0008;
scene.add(key);

// Fill from the far side so the shadowed faces do not go flat black against a
// background this bright. No shadow of its own - a second shadow would read as a
// second desk.
const fill = new THREE.DirectionalLight(0xeef2ff, 0.55);
fill.position.set(-0.9, 0.7, -1.3);
scene.add(fill);

// The desk the board stands on, so heights have something to be measured from.
const desk = new THREE.Mesh(
  new THREE.PlaneGeometry(2.4, 1.6),
  new THREE.MeshStandardMaterial({ color: 0xefe9dd, roughness: 0.95 }),
);
desk.receiveShadow = true;
world.add(desk);
const grid = new THREE.GridHelper(2.4, 24, 0xc9bda6, 0xdbd2c1);
grid.rotation.x = Math.PI / 2;
grid.position.z = 0.001;   // just off the desk, so it is not z-fighting with it
world.add(grid);

// The board the arms are bolted to. Not in the URDF - it is scenery - so the page
// draws it, and the simulation says how big it is: `scene.xml` carries the geom
// that stops an arm, and /physics hands its size over. A board drawn from its own
// numbers would sooner or later be a different size from that one, and the hand
// would appear to halt in mid air near the edge.


// The face texture is drawn to the board's own size, so it is made once the scene
// has said what that is. Until then the material carries no map.
const boardFace = new THREE.MeshStandardMaterial({ roughness: 0.82, metalness: 0.0 });
const boardEdge = new THREE.MeshStandardMaterial({ color: 0x141416, roughness: 0.85 });
// Material per face, in BoxGeometry's order: +x, -x, +y, -y, +z, -z. The perforated
// faces are the two large ones; the four edges are the sawn thickness and have no
// holes in them.
//
// A unit box, scaled into place. The scene states half extents, the way MuJoCo
// does, so the scale is twice those and the position is the geom's own.
const board = new THREE.Mesh(
  new THREE.BoxGeometry(1, 1, 1),
  [boardFace, boardFace, boardEdge, boardEdge, boardEdge, boardEdge],
);
board.castShadow = true;
board.receiveShadow = true;
world.add(board);

/** Size and place the board and the desk from the scene the simulation compiled. */
function placeScenery(scenery) {
  const slab = scenery?.pegboard;
  if (slab) {
    const [hx, hy, hz] = slab.half;
    board.scale.set(hx * 2, hy * 2, hz * 2);
    board.position.set(...slab.pos);
    boardFace.map = pegboardTexture(hy * 2, hz * 2, renderer.capabilities.getMaxAnisotropy());
    boardFace.needsUpdate = true;
  }
  // The desk is a plane, and a MuJoCo plane is a half space: its size is what the
  // simulator draws, not what it collides. So only the height is a fact to honour -
  // how far the desk is drawn is this page's own business, and it is drawn well
  // past the board so the workspace has a floor rather than a tabletop with an edge.
  const floor = scenery?.desk;
  if (floor) {
    desk.position.z = floor.pos[2];
    grid.position.z = floor.pos[2] + 0.001;
  }
}

let robot = null;
// The arm as asked for, drawn see-through over the arm as it is. The handles hang
// on it, so a drag moves it at once while the real arm - a fifth of a second
// behind its command - follows; without it the only arm on the screen is the
// one lagging behind the hand. Hidden whenever the two agree.
let ghost = null;
let ghostParts = [];
let dragging = null;          // the joint names a drag is moving, or null
// Wider than the servos fall short when they settle (a degree or two), or the
// ghost would sit over an arm standing still.
const GHOST_SHOWN_DEG = 3.0;
const ghostMaterial = new THREE.MeshStandardMaterial({
  color: 0xe04b8e, transparent: true, opacity: 0.3, depthWrite: false,
  roughness: 0.6, metalness: 0,
});
function showGhost(on) {
  for (const mesh of ghostParts) mesh.visible = on;
}
// The simulation. There is no version of this page without one: contact is decided
// by MuJoCo, against the real collision meshes, and it sees the arms meeting each
// other as well as the desk and the board. Testing the two surfaces here instead
// would give a move two meanings - refused by the page, pushed back by the
// simulation - and would know nothing about the arms meeting.
let sim = null;

function fit() {
  if (!robot) return;
  // Box3.setFromObject already reports world coordinates. Transforming the centre
  // by the world group's matrix on top of that moved the target somewhere else
  // entirely, which is why recentring kept leaving the camera inside the robot.
  world.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(robot.root);
  const centre = box.getCenter(new THREE.Vector3());
  const radius = box.getSize(new THREE.Vector3()).length() / 2 || 0.5;
  controls.target.copy(centre);
  // Look down enough that the desk reads as a surface. Edge-on, the arms appear
  // to float and there is nothing to judge height against - which is the one thing
  // this view exists to help with.
  // In three.js coordinates, after the world rotation: +y is up, +x is out of the
  // board, +z is toward the right arm.
  const dir = new THREE.Vector3(1.5, 0.85, 1.35).normalize();
  camera.position.copy(controls.target).add(dir.multiplyScalar(radius * 2.9));
  controls.update();
}

// --- surfaces ------------------------------------------------------------
// Say what is touching, and say it on the surface itself. A status line alone
// leaves you hunting for the part that is against something; a board that lights up
// tells you where to look before you have finished reading the words.
const SURFACES = { desk: desk.material, board: boardFace };

function light(surface) {
  const material = SURFACES[surface];
  if (!material) return;
  material.emissive.setHex(0xc42d18);
  material.emissiveIntensity = 0.5;
}

// MuJoCo names what is touching what: `desk` and `pegboard` are the scene's own
// geoms, everything else is a link of the robot. The two surfaces the page draws
// light up where they are touched; the rest - an arm on itself, or on the other
// arm - has nowhere to light up and is said in words.
const SCENERY = { desk: 'desk', pegboard: 'board' };

function partName(name) {
  return name.replace(/_link$/, '').replace(/_/g, ' ');
}

// Under this a contact is two parts resting against each other and is said as
// "touching"; over it, something is being driven into something else and the depth
// is worth a number. The line is drawn clear of the contact solver's own give -
// two wrists pressed together settle around half a millimetre in and would
// otherwise flicker between the two readings - and well under what a servo pushing
// at its force limit manages, which is a claw about ten millimetres into the desk.
const RESTING_M = 0.002;

// Which surfaces are being touched right now. A refused move is an event and
// fades; resting on the desk is a state and does not, so this is re-lit every
// frame rather than once when the contact was reported.
const touching = new Set();

function reportContacts(contacts) {
  touching.clear();
  if (!contacts?.length) return null;
  const worst = contacts[0];
  const surfaces = [SCENERY[worst.a], SCENERY[worst.b]].filter(Boolean);
  for (const contact of contacts) {
    for (const geom of [contact.a, contact.b]) {
      if (SCENERY[geom]) touching.add(SCENERY[geom]);
    }
  }
  const mm = Math.round(worst.depth * 1000);
  const how = worst.depth > RESTING_M ? `${mm} mm in` : 'touching';
  if (surfaces.length) {
    const part = SCENERY[worst.a] ? worst.b : worst.a;
    return `${partName(part)} on the ${surfaces[0]} - ${how}`;
  }
  return `${partName(worst.a)} against ${partName(worst.b)} - ${how}`;
}

function armOf(jointName) {
  const cut = jointName.indexOf('_');
  return cut > 0 ? jointName.slice(0, cut) : '';
}

const sliderFor = new Map();
// Where each joint actually got to, when that is not where it was asked to go.
// With physics under the page those two are different things, and the difference
// is the one thing a kinematic view could never show.
const reachedFor = new Map();
// Where the servo says it is, when there is a servo. A third number, and the reason
// the panel is worth looking at with hardware attached: asked, simulated, measured.
const measuredFor = new Map();

const degrees = (radians) => (radians * 180 / Math.PI).toFixed(0);
// The slider's track is filled up to its thumb by style.css, from this.
function paint(slider) {
  const lo = Number(slider.min), hi = Number(slider.max);
  const pct = hi > lo ? ((Number(slider.value) - lo) / (hi - lo)) * 100 : 0;
  slider.style.setProperty('--fill', `${pct}%`);
}
// Below this the arm has arrived as far as anyone looking at the page cares. A
// position servo settles with a little droop under gravity, and reporting half a
// degree of it as a miss would mark every joint on the robot.
const REACHED_SLACK_DEG = 1.5;
// How far the metal may sit from the picture before the number is worth marking.
// Wider than the simulated slack: this gap carries the servo's own droop, the play
// in a printed joint and whatever the tick-to-angle pairing did not capture.
const MEASURED_SLACK_DEG = 4;

function buildSliders() {
  jointsEl.textContent = '';
  sliderFor.clear();
  reachedFor.clear();
  measuredFor.clear();
  const byArm = new Map();
  for (const joint of robot.movingJoints()) {
    const arm = armOf(joint.name);
    if (!byArm.has(arm)) byArm.set(arm, []);
    byArm.get(arm).push(joint);
  }

  for (const [arm, joints] of byArm) {
    // Hand at the top, base at the bottom. movingJoints() hands them over base
    // first, which is the order the arm is built in but the opposite of the order
    // it is read in: you look at the claw, then follow it back down to the board.
    joints.reverse();

    const title = document.createElement('div');
    title.className = 'arm-title';
    title.textContent = arm || 'joints';
    jointsEl.append(title);

    for (const joint of joints) {
      const wrap = document.createElement('div');
      wrap.className = 'joint';
      const label = document.createElement('label');
      const name = document.createElement('span');
      name.className = 'name';
      name.textContent = joint.name.slice(arm.length + 1);
      const value = document.createElement('span');
      value.className = 'value';
      const reached = document.createElement('span');
      reached.className = 'reached';
      // Only with arms attached. Without them there is nothing to measure, and an
      // empty column would read as a servo answering zero.
      const measured = document.createElement('span');
      measured.className = 'measured';
      const slider = document.createElement('input');
      slider.type = 'range';
      slider.min = joint.lower;
      slider.max = joint.upper;
      slider.step = 0.005;
      slider.value = joint.value;
      // Locked until the gate (or the simulation) has said the page drives.
      slider.disabled = jogLocked !== false;

      const show = () => { value.textContent = `${degrees(slider.value)}°`; paint(slider); };
      show();
      slider.addEventListener('input', () => {
        // Ask the servo, do not place the arm. Whether it arrives is the
        // simulation's answer and it comes back on the stream, which is why the
        // slider is left where you put it and nothing is refused here. The slider
        // is the command; the picture is what came of it.
        ghost?.setJoint(joint.name, Number(slider.value));
        askedSince?.add(joint.name);
        sim.command({ [joint.name]: Number(slider.value) });
        show();
      });

      label.append(name, value, reached);
      if (sim.real) label.append(measured);
      wrap.append(label, slider);
      jointsEl.append(wrap);
      sliderFor.set(joint.name, slider);
      reachedFor.set(joint.name, reached);
      measuredFor.set(joint.name, measured);
    }
  }
  return byArm;
}

/**
 * Push the model's joint values back onto the sliders.
 *
 * With `names`, only those joints - a drag moves the joints it owns and no others,
 * and the sliders are what the arm was ASKED for. Sweeping all fourteen after a
 * gesture would overwrite thirteen commands with wherever those joints have
 * settled, which is not the same number.
 */
function syncSliders(names) {
  const wanted = names ? new Set(names) : null;
  for (const joint of (ghost ?? robot).movingJoints()) {
    if (wanted && !wanted.has(joint.name)) continue;
    const slider = sliderFor.get(joint.name);
    if (!slider) continue;
    slider.value = joint.value;
    paint(slider);
    slider.dispatchEvent(new Event('change'));
    const label = slider.previousElementSibling?.querySelector('.value');
    if (label) label.textContent = `${degrees(joint.value)}°`;
    // The pose the panel now shows is the pose just asked for, so nothing has
    // fallen short of it yet. The next frame off the stream says whether it did.
    const reached = reachedFor.get(joint.name);
    if (reached) reached.textContent = '';
  }
}

// What the status line says when nothing is touching. The stream overwrites that
// line on every frame it sends, so the last thing worth saying has to be kept.
let idle = '';

/**
 * Show a pose that was not driven from this page - zero, the homed offsets, the
 * servos. The rule is about where a pose comes from, not about which of them the
 * panel happens to offer.
 *
 * Placed, not driven. Asking the simulation to travel to a stated pose would show
 * the trip instead of the pose, and would never arrive at all if the pose is one
 * the desk is standing in - which the servos' own home pose is, on this rig. A
 * stated pose is a fact about the arm, so it is set outright and whatever it turns
 * out to be touching is named rather than refused.
 */
function applyReadPose(pose, what) {
  frames.length = 0;   // what was buffered led up to the old pose, not this one
  playAt = null;
  robot.setPose(pose);
  syncSliders();
  idle = what;
  sim.place(pose);
  statusEl.textContent = what;
}

/**
 * The joints a gesture moved, as a pose to ask the servos for.
 *
 * Only those joints, and this is the whole of it: `robot.pose()` would read back
 * all fourteen, and with physics under the page those fourteen are where the arm
 * HAS got to, not what it was asked for. A position servo settles a little under
 * its target, so commanding a joint to its own settled value re-aims it lower -
 * and a gesture that does that every frame walks the joints nobody touched
 * downwards, a step at a time. The other arm drifts with nothing near it.
 */
function commandFor(names) {
  const pose = {};
  for (const name of names) {
    const joint = (ghost ?? robot).joints.get(name);
    if (joint) pose[name] = joint.value;
  }
  return pose;
}

/**
 * A frame off the simulation: where the robot is, and what it is against.
 *
 * Two vectors arrive, and keeping them apart is the whole point of the panel:
 * `qpos` is where the arm IS and drives the picture, `ctrl` is what it was asked
 * for and is what the angles are measured against.
 */
// The stream is sixty frames a second, but they do not arrive sixty times a second:
// a frame lands whenever the server's thread and the network let it, a few
// milliseconds either side. Drawn on arrival, a screen frame sometimes gets no
// new pose and the next gets two, and an arm moving smoothly in the simulation
// judders on the page - most at the gripper, which moves furthest. So poses are
// buffered by their simulation time and drawn a little behind the newest one,
// interpolated between the two frames either side.
const PLAYBACK_DELAY_S = 0.02;
// The readings of the real arms, a frame of the stream apart at best and uneven:
// drawn a little further behind so there are two to interpolate between.
const REAL_PLAYBACK_DELAY_S = 0.05;
const frames = [];     // { t, qpos }, oldest first
let playAt = null;     // the simulation time being drawn
let drawnAt = null;    // performance.now() of the last draw

function bufferPose(frame) {
  // With the real arms the simulation is placed, not stepped, and its clock stands
  // still: the readings are timed by when they arrive instead.
  const t = sim?.real ? performance.now() / 1000 : frame.time;
  const last = frames[frames.length - 1];
  if (!last || !(t > last.t)) {
    // The first frame, or time did not move - the arms are placed from the servos
    // rather than stepped, or the simulation was reset. There is nothing to
    // interpolate: show it as is. A settled simulation sends its pose once, so
    // the first frame may be the only one until something moves.
    frames.length = 0;
    playAt = null;
    robot.setPose(frame.qpos);
  }
  frames.push({ t, qpos: frame.qpos });
  while (frames.length > 2 && frames[0].t < t - 1) frames.shift();
}

function drawPose(now) {
  if (frames.length < 2) return;
  const latest = frames[frames.length - 1].t;
  const delay = sim?.real ? REAL_PLAYBACK_DELAY_S : PLAYBACK_DELAY_S;
  playAt = playAt === null ? latest - delay : playAt + (now - drawnAt) / 1000;
  drawnAt = now;
  // Held within reach of the newest frame: a stall on the stream slows the picture
  // for a moment rather than letting it fall behind for good.
  playAt = Math.min(latest, Math.max(latest - 4 * delay, playAt));
  let k = frames.length - 2;
  while (k > 0 && frames[k].t > playAt) k -= 1;
  const a = frames[k], b = frames[k + 1];
  const u = b.t > a.t ? Math.min(1, Math.max(0, (playAt - a.t) / (b.t - a.t))) : 1;
  const pose = {};
  for (const [name, angle] of Object.entries(b.qpos)) {
    const from = a.qpos[name] ?? angle;
    pose[name] = from + (angle - from) * u;
  }
  robot.setPose(pose);
}

function applyFrame(frame) {
  bufferPose(frame);
  // The see-through arm is where the arm is headed; the gate's shaping carries
  // the arm there smoothly. In the simulation, and on the metal while the
  // page asks - a drag, "jog the arms", a pose walk - that is the page's own
  // target. On the metal under the handheld loop it is what the loop last
  // asked the gate for. Stopped, the page asks nothing and the arm has moved
  // on from its target: the ghost stays on the arm, out of sight, so a jog
  // taken up after starts its drags from where the arm is.
  const wasOnMetal = onMetal;
  onMetal = !!frame.measured?.expected;
  // Onto the metal or off it, the lock is not known until the gate (or the
  // simulation) says, as on a fresh page.
  if (onMetal !== wasOnMetal) jogLocked = null;
  const teleop = frame.measured?.teleop;
  const following = onMetal && !!teleop?.armed && teleop.mode === MODE_HANDHELD;
  const pageAsks = !onMetal || dragging !== null || jogLocked === false || moving;
  // A pose walk, once the bridge is on it, asks for every joint; off the metal
  // the target is the page's alone.
  if (frame.measured?.walking || !onMetal) askedSince = null;
  // A joint not asked lately - a hand out of sight under the loop, one the
  // page has not moved since it took over - is drawn where it stands.
  const headed = !pageAsks ? (following ? { ...frame.qpos, ...frame.measured?.asked_rad } : null)
    : askedSince ? { ...frame.qpos, ...pick(frame.ctrl, askedSince) } : frame.ctrl;
  if (ghost && headed) {
    let apart = dragging !== null;
    for (const [name, asked] of Object.entries(headed)) {
      const held = dragging?.includes(name) || sliderFor.get(name) === document.activeElement;
      if (!held) ghost.setJoint(name, asked);
      const at = frame.qpos?.[name];
      if (at !== undefined && Math.abs(at - asked) * 180 / Math.PI > GHOST_SHOWN_DEG) apart = true;
    }
    showGhost(apart);
  } else if (ghost && frame.qpos) {
    for (const [name, at] of Object.entries(frame.qpos)) ghost.setJoint(name, at);
    showGhost(false);
  }

  for (const [name, angle] of Object.entries(frame.qpos)) {
    const reached = reachedFor.get(name);
    if (!reached) continue;
    // What was asked for comes off the wire, not out of the slider. The servos hold
    // the command and report it back; reading it out of the panel instead would
    // measure the picture against a number the DOM happens to be showing, which is
    // the same mistake as taking a command from the pose the arm has settled into.
    const asked = (askedSince && !askedSince.has(name) ? undefined : frame.ctrl?.[name]) ?? angle;
    const gap = Math.abs(angle - asked) * 180 / Math.PI;
    reached.textContent = gap > REACHED_SLACK_DEG ? `→ ${degrees(angle)}°` : '';

    // And the slider follows it, because not every command comes from this page: a
    // take being replayed, or another laptop on the same simulation, moves the arm
    // without touching the panel. The one being dragged is left alone - it is
    // already the command, and a frame of lag would shove it back under the hand.
    const slider = sliderFor.get(name);
    if (!slider || slider === document.activeElement) continue;
    if (Math.abs(Number(slider.value) - asked) > 1e-4) {
      slider.value = asked;
      paint(slider);
      const shown = slider.previousElementSibling?.querySelector('.value');
      if (shown) shown.textContent = `${degrees(asked)}°`;
    }
  }

  showMeasured(frame.measured);
  cameraUp = !!frame.rig?.live?.running;
  showGate(frame.measured);
  showSimFollow(frame.sim_hands);
  showPedal(frame.measured?.pedal);
  // The two processes the dashboard switches ride on the same frame: a camera or a
  // loop that died shows as dead on the frame it died.
  dash.frame(frame.rig);
  showSidecars(frame.rig);
  statusEl.textContent = reportContacts(frame.contacts) ?? idle;
}

/**
 * Where the arms actually are, when the server is reading any.
 *
 * Shown against the simulation rather than against the command: the question a
 * person has with hardware attached is whether the picture matches the desk, and
 * that is qpos against the servo, not the servo against what was asked for.
 */
function showMeasured(reading) {
  if (!reading) return;
  const angles = reading.angles_rad ?? {};
  for (const [name, span] of measuredFor) {
    const angle = angles[name];
    if (angle === undefined) { span.textContent = ''; continue; }
    span.textContent = `${degrees(angle)}°`;
    const off = Math.abs(angle - (robot.joints.get(name)?.value ?? angle)) * 180 / Math.PI;
    span.classList.toggle('adrift', off > MEASURED_SLACK_DEG);
  }
}

// What the safety gate says about the robot, and the two ways to drive it.
//
// Armed, the mode and the reason are the gate's to own: a page that lit its own
// button on the click that asked would go on showing a driving arm after
// somebody stopped it from another window. So every one of these comes off the
// frame, and a frame that has gone stale is drawn as stale rather than as the
// robot.
//
// Torque is not a button here. An operator wants the arms live or not live, and
// making that two controls in the right order was a way to end up with a mode
// set on limp arms and no clue why nothing moved. Driving turns it on; stopping
// leaves it on, because these arms hang on a wall and a limp one falls.
const gateEl = document.getElementById('gate');
const gateStateEl = document.getElementById('gate-state');
const gateReasonEl = document.getElementById('gate-reason');
const jogEl = document.getElementById('drive-jog');
const handheldEl = document.getElementById('drive-handheld');
const stopEl = document.getElementById('gate-stop');
// The handheld's own pane says the same thing this row does, because it is the
// same fact. It has no button of its own: the row above owns the asking.
const teleopDotEl = document.getElementById('teleop-dot');
const teleopLineEl = document.getElementById('teleop-line');
const MODE_NAMES = ['stopped', 'following the handheld', 'driven from the page', 'running the policy'];
const torqueEl = document.getElementById('torque');

let cameraUp = false;

// A pose walk - the ready or home button with the real arms. While it runs only
// the pose buttons and stop are live: the pedals, the torque switch, the sliders
// and the drag handles would each take the arms off the walk halfway. The page
// knows the walk has begun the moment it asks, before the bridge reports it, and
// that it has ended when the bridge says so.
const MOVING_GRACE_MS = 2000;
let movingAsked = 0;
let movingTo = null;   // 'ready' or 'home' when this page asked; null when it did not
let walkSeen = false;
let moving = false;
let handles = [];

function setMoving(on) {
  if (on === moving) return;
  moving = on;
  jogEl.disabled = on;
  handheldEl.disabled = on || !cameraUp;
  torqueEl.disabled = on;
}

// The sliders and the drag handles drive the arms only once "jog the arms" is
// on: the page asks the gate for the studio mode, and until the gate says it
// has it - or while a pose walk runs - they are locked and grey. The
// simulation has no gate, and keeps the same mode itself (showSimFollow).
// null until the first word from the gate (or the simulation), so that word
// always sets the sliders and handles - and, on the metal, starts askedSince.
let jogLocked = null;
// Whether the page draws the metal, and where the arm last was (applyFrame).
let onMetal = false;
// Set when the page takes the metal over for "jog the arms": the joints the
// page has asked for since. The rest stand, for the ghost, the handles and the
// sliders, where the arm reads - not at whatever the page last asked before.
let askedSince = null;
const pick = (pose, names) => Object.fromEntries(Object.entries(pose ?? {}).filter(([k]) => names.has(k)));
function setJogLocked(on) {
  if (on === jogLocked) return;
  jogLocked = on;
  // Unlocked on the metal, the page becomes the arm's driver. Its target is
  // what it asked before the handheld loop or a stop moved the arm on; a
  // drag worked out from that pose would send the arm back toward it. So the
  // ghost and the sliders start from where the arm reads (`askedSince`), and
  // nothing is sent until the operator moves something: a drag then asks
  // only for the joints it moves, a jaw holding an object is not touched.
  if (!on && onMetal) askedSince = new Set();
  for (const slider of sliderFor.values()) slider.disabled = on;
  greyHandles(handles, on);
}

function showTorque(torque) {
  const arms = Object.entries(torque ?? {});
  torqueEl.checked = arms.some(([, on]) => on);
  document.getElementById('torque-switch').title = arms.length
    ? `${arms.map(([arm, on]) => `${arm} ${on ? 'on' : 'off'}`).join(', ')} - torque off lets the arms fall, hold them first`
    : 'no word from the driver yet';
}

function showGate(reading) {
  if (!reading?.expected) return;
  gateEl.hidden = false;
  const teleop = reading.teleop;
  if (!reading.fresh || !teleop) {
    gateStateEl.textContent = reading.heard ? 'stale' : 'waiting';
    gateStateEl.className = 'chip stale';
    gateReasonEl.textContent = reading.heard
      ? `nothing from ROS for ${(reading.age ?? 0).toFixed(1)}s - is the bridge running?`
      : (reading.quiet ?? 'waiting for ROS: ros2 launch pinkrab_bringup robot.launch.py');
    setDriving(null);
    setJogLocked(true);
    return;
  }
  const driving = teleop.armed ? teleop.mode : MODE_IDLE;
  if (reading.walking) walkSeen = true;
  const pending = movingAsked && !walkSeen && performance.now() - movingAsked < MOVING_GRACE_MS;
  const walking = !!reading.walking || pending;
  if (!walking) { movingAsked = 0; movingTo = null; }
  gateStateEl.textContent = walking ? `going to ${movingTo ?? 'a pose'}` : (MODE_NAMES[driving] ?? `mode ${teleop.mode}`);
  gateStateEl.className = `chip${walking ? ' moving' : driving === MODE_IDLE ? '' : ' driving'}`;
  gateReasonEl.textContent = teleop.reason ?? '';
  showTorque(reading.torque);
  setDriving(driving);
  setMoving(walking);
  setJogLocked(walking || driving !== MODE_STUDIO);
}

// The simulation has no gate: the studio keeps its mode and follows the hands
// itself, and the same three pedals switch it as on the metal - 1 the page
// jogs the arms, 2 the hands are followed, 3 stops. Torque is the metal's
// and is not shown.
function showSimFollow(sim) {
  if (!sim) return;
  gateEl.hidden = false;
  document.getElementById('torque-switch').hidden = true;
  const driving = sim.mode ?? MODE_IDLE;
  const arms = Object.entries(sim.arms ?? {}).map(([arm, how]) => `${arm} ${how}`).join(' · ');
  gateStateEl.textContent = MODE_NAMES[driving];
  gateStateEl.className = `chip${driving === MODE_IDLE ? '' : ' driving'}`;
  gateReasonEl.textContent = driving === MODE_HANDHELD ? [sim.note, arms].filter(Boolean).join(' · ') : '';
  setDriving(driving);
  setJogLocked(driving !== MODE_STUDIO);
}

// A press on the floor, shown on the page: the matching pedal flashes for a
// moment the instant the server hears it, before the gate has answered, so a
// foot that pressed knows it did. Whether the press took is still the frame's
// to say - the flash is the press, the lit edge is the state.
const PEDAL_BUTTON = { jog: () => jogEl, handheld: () => handheldEl, stop: () => stopEl };
const PEDAL_FLASH_MS = 350;
let pedalSeen = 0;
function showPedal(pedal) {
  if (!pedal || pedal.count === pedalSeen) return;
  const firstSight = pedalSeen === 0;
  pedalSeen = pedal.count;
  // A press from before this page was opened is history, not news.
  if (firstSight && pedal.age > 2) return;
  const button = PEDAL_BUTTON[pedal.name]?.();
  if (!button) return;
  button.classList.remove('pressed');
  void button.offsetWidth;  // restart the animation on a repeated press
  button.classList.add('pressed');
  setTimeout(() => button.classList.remove('pressed'), PEDAL_FLASH_MS);
}

/** Which button is lit, and whether the handheld can be asked for at all. */
function setDriving(mode) {
  jogEl.classList.toggle('on', mode === MODE_STUDIO);
  handheldEl.classList.toggle('on', mode === MODE_HANDHELD);
  // The stop pedal sits pressed while the gate is closed: idle is a state the
  // frame reports, not the absence of one.
  stopEl.classList.toggle('on', mode === MODE_IDLE);
  // The handheld is solved off the top camera. Without it there is no solid and
  // nothing to follow, so the button says what is missing rather than arming an
  // arm that will then sit still.
  handheldEl.disabled = !cameraUp || moving;
  handheldEl.title = cameraUp ? 'the solid in frame picks the arm'
                              : 'start the top camera first';
  const following = mode === MODE_HANDHELD;
  document.getElementById('pane-teleop')?.classList.toggle('running', following);
  teleopDotEl.className = `dot${following ? ' on' : ''}`;
  teleopLineEl.textContent = following
    ? 'the solid in frame picks the arm'
    : (cameraUp ? '' : 'the top camera is off');
}

// Simulation or the real robot: chosen on the front page, shown here.
function showWhich(real) {
  whichEl.textContent = real ? 'real' : 'sim';
  whichEl.className = real ? 'real' : 'sim';
}

// The robot's side, as the server reports it: bridge, handheld loop, pedal.
const sidecarsEl = document.getElementById('sidecars');
function showSidecars(rig) {
  const robot = rig?.robot;
  if (!robot || !sim?.real) { sidecarsEl.hidden = true; return; }
  sidecarsEl.hidden = false;
  const names = { bridge: 'bridge', teleop: 'handheld loop', footswitch: 'pedal' };
  const parts = { ...robot.parts, footswitch: rig.footswitch || {} };
  sidecarsEl.replaceChildren(...Object.entries(parts).map(([name, part]) => {
    const span = document.createElement('span');
    span.textContent = names[name] ?? name;
    span.className = part.running ? 'on' : (part.exit ? 'bad' : '');
    span.title = part.running ? `pid ${part.pid}`
      : part.exit ? `exited with ${part.exit} - see the terminal`
      : 'not started here - already running, or refused (see the terminal)';
    return span;
  }));
}

// A button asks ROS for something by leaving the request with the server; the
// bridge picks it up and makes the service call. Nothing is drawn from the
// answer - what the page shows is the next state the gate publishes.
async function askRobot(body) {
  try {
    await fetch('/real/request', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch { /* the next frame says whether anything happened */ }
}

/**
 * Make the arms live and hand them to one driver. The server turns this into
 * idle, torque on, then the mode - the same three the foot switch asks for,
 * from the same place, so a pedal on the floor and a pedal on the page cannot
 * drift apart.
 */
async function drive(mode) {
  await askRobot({ kind: 'drive', mode });
}

jogEl.addEventListener('click', () => drive(MODE_STUDIO));
handheldEl.addEventListener('click', () => drive(MODE_HANDHELD));
// Stop, not limp: the mode goes to idle and nothing more passes the gate, while
// the servos keep holding the pose they are in.
stopEl.addEventListener('click', () => {
  movingAsked = 0;
  movingTo = null;
  askRobot({ kind: 'mode', mode: MODE_IDLE });
});
// Torque, one switch. Off stops whatever was driving and lets the arms go limp;
// on holds them where they are with nothing driving. The switch is put back to
// what the driver reports on every frame, so a click that did not take does
// not stay drawn as if it had.
torqueEl.addEventListener('change', () => {
  if (torqueEl.checked) {
    askRobot({ kind: 'torque', arm: 'both', enable: true });
  } else {
    askRobot({ kind: 'mode', mode: MODE_IDLE });
    askRobot({ kind: 'torque', arm: 'both', enable: false });
  }
});

// Two poses, and they are not the same thing. Home is the description's zero:
// every joint at 0, the arms parked folded on themselves, where the bringup takes
// them on the way down. Ready is where the tool is picked up - bent, the wrists
// turned - and where the page opens. The server says which joints are not zero
// in the ready pose and by how much, so this button and a policy run's walk
// cannot end up being two different quarter turns; if it cannot say, every
// joint goes to zero and the pose is still a pose.
function homePose() {
  const pose = {};
  for (const joint of robot.movingJoints()) pose[joint.name] = 0;
  return pose;
}

async function readyPose() {
  const pose = homePose();
  try {
    const answer = await fetch('/ready');
    if (answer.ok) {
      for (const [name, value] of Object.entries(await answer.json())) {
        if (name in pose) pose[name] = value;
      }
    }
  } catch (err) { /* zero is the fallback, and it is a real pose */ }
  return pose;
}

// With the metal, a pose button is not a placement - the picture follows the
// readings, and placing it would be overwritten by the next one. It is an
// order: take the arms over from whatever was driving them (the handheld loop
// lets go when the mode leaves it), and walk them there slowly, through the
// gate like anything else. Useful before a session and in the middle of one,
// when the arm has wandered somewhere the hand cannot follow from.
async function poseButton(pose, what) {
  if (!sim.real) {
    // The page jogs the arms there, as on the metal.
    await drive(MODE_STUDIO);
    applyReadPose(pose, what);
    return;
  }
  statusEl.textContent = `walking to the ${what}`;
  movingAsked = performance.now();
  movingTo = what.replace(/ pose$/, '');
  walkSeen = false;
  setMoving(true);
  await drive(MODE_STUDIO);
  await askRobot({ kind: 'goto', pose });
}
document.getElementById('ready').addEventListener('click', async () => {
  poseButton(await readyPose(), 'ready pose');
});
document.getElementById('home').addEventListener('click', () => {
  poseButton(homePose(), 'home pose');
});

// Where the tool can grasp on the desk, in any arm shape: one patch an arm,
// solved by the server once for the description and laid on the desk the
// first time the switch is turned on. It does not follow the arms: it is the
// robot's reach, not the pose's.
const REACH_COLOUR = { left: 0x1668c9, right: 0xe2488e };
const REACH_OPACITY = 0.45;
const reach = new THREE.Group();
reach.visible = false;
world.add(reach);
const reachSwitch = document.getElementById('workspace');
let reachDrawn = false;
let reachBusy = false;

// A layer of cells on the desk, one tile each.
function tilesOf(cells, size, lift, colour, opacity, order) {
  const tile = new THREE.BoxGeometry(size, size, 0.002);
  const material = new THREE.MeshStandardMaterial({
    color: colour, transparent: true, opacity, roughness: 0.9, depthWrite: false,
  });
  const tiles = new THREE.InstancedMesh(tile, material, cells.length);
  const place = new THREE.Object3D();
  cells.forEach(([x, y], i) => {
    place.position.set(x, y, lift);
    place.updateMatrix();
    tiles.setMatrixAt(i, place.matrix);
  });
  tiles.instanceMatrix.needsUpdate = true;
  tiles.renderOrder = order;
  return tiles;
}

async function loadReach() {
  reachBusy = true;
  try {
    // Answered at once once solved; in the minute after the server starts,
    // when the solve it started then is done.
    const data = await (await fetch('/workspace')).json();
    if (data.error) throw new Error(data.error);
    const lift = data.desk_z + 0.0015;   // just off the desk, over the grid lines
    for (const [arm, patch] of Object.entries(data.arms)) {
      if (!patch.cells.length) continue;
      reach.add(tilesOf(patch.cells, data.cell_m, lift, REACH_COLOUR[arm] ?? 0x888888, REACH_OPACITY, 1));
    }
    reachDrawn = true;
  } finally {
    reachBusy = false;
  }
}

reachSwitch.addEventListener('change', async () => {
  reach.visible = reachSwitch.checked;
  // Drawn once; turned on again, or while the answer is on its way, it only shows.
  if (!reachSwitch.checked || reachDrawn || reachBusy) return;
  try {
    await loadReach();
  } catch (err) {
    statusEl.textContent = `desk reach: ${err.message}`;
    reachSwitch.checked = false;
    reach.visible = false;
  }
});

// The two drag spheres explain themselves on hover: a label that appears under the
// pointer, on the thing the pointer is on, is read at the moment the question is
// asked rather than in a paragraph somewhere before it.
const tipEl = document.getElementById('tip');
function showTip(text, event) {
  if (!text) { tipEl.hidden = true; return; }
  tipEl.textContent = text;
  tipEl.hidden = false;
  // Flip to the other side of the cursor rather than run off the edge. The panel
  // sits on the right now, so a tip near it has somewhere to go.
  const gap = 14;
  const right = event.clientX + gap + tipEl.offsetWidth;
  tipEl.style.left = `${right > innerWidth ? event.clientX - gap - tipEl.offsetWidth : event.clientX + gap}px`;
  tipEl.style.top = `${event.clientY + gap}px`;
}

function resize() {
  const { clientWidth: w, clientHeight: h } = canvas;
  if (canvas.width !== w || canvas.height !== h) {
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
}

function tick(now) {
  resize();
  controls.update();
  drawPose(now ?? performance.now());
  for (const surface of touching) light(surface);
  // Fade the surface that refused the last move. Held on, it would read as part of
  // the scene rather than as an answer to something you just did. A surface that
  // is still being touched was just re-lit above, so this does not reach it.
  for (const material of Object.values(SURFACES)) {
    const lit = material.emissiveIntensity;
    if (lit > 0) material.emissiveIntensity = Math.max(0, lit - 0.02);
  }
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}

(async function main() {
  try {
    const response = await fetch('/model');
    if (!response.ok) throw new Error((await response.json()).error ?? response.statusText);
    robot = await loadURDF(await response.text(), '/meshes', {
      materialFor, outlineMaterial: OUTLINE, hullMaterial: HULL,
    });
    world.add(robot.root);
    ghost = await loadURDF(await (await fetch('/model')).text(), '/meshes', {
      materialFor: () => ghostMaterial, outlineMaterial: null, hullMaterial: null,
    });
    ghost.root.traverse((o) => {
      if (o.isMesh && o.userData.urdfLink) { o.castShadow = false; o.receiveShadow = false; ghostParts.push(o); }
    });
    showGhost(false);
    world.add(ghost.root);

    // Before the sliders and the handles, both of which command it. connect throws
    // if the server has no simulation - there is nothing for this page to be
    // without one, and a viewer that quietly degraded to posing the description
    // would look identical while meaning something else entirely.
    sim = await connectPhysics();
    placeScenery(sim.scenery);
    // The hands beside the arms: the same frame, the same viewpoint. Not
    // waited for - the arms are this page's first business.
    const handsLine = document.getElementById('hands-line');
    showHands({
      canvas: document.getElementById('hands-view'), lineEl: handsLine, robot, robotWorld: world,
      scenery: sim.scenery, view: { camera, controls, background: scene.background },
    }).catch((err) => { handsLine.textContent = err.message; });
    // Exposed on purpose. A viewer you cannot poke at from the console is harder
    // to debug than the thing it is meant to help you debug.
    window.pinkrab = { robot, ghost, sim, scene, world, camera, controls, THREE };
    buildSliders();
    // Drag the spheres: blue at the elbow places the arm (pan, lift solved),
    // yellow at the tip aims the hand (elbow, flex, yaw solved). See ik.js.
    //
    // The solver runs in the browser - a round trip per pointer move would put the
    // network in the middle of a gesture - but its answer is a request rather than
    // a placement, and the next frame off the stream replaces it with wherever the
    // arm got to. Each step is then solved from where the arm IS toward where the
    // handle is, which is a servo law, and it converges.
    handles = enableDrag({
      robot: ghost, camera, canvas, controls, onHover: showTip, locked: () => jogLocked !== false,
      onGrab: (names) => { dragging = names; showGhost(true); },
      onRelease: () => { dragging = null; },
      synced: (names) => {
        syncSliders(names);
        for (const name of names) askedSince?.add(name);
        sim.command(commandFor(names));
      },
    });
    fit();
    showWhich(sim.real);
    whichEl.hidden = false;
    // The sliders stay live with real arms. They are the command path now: what
    // the page asks its simulation for is what the bridge publishes on
    // /joint_commands, so a slider crosses the gate and reaches the servos. The
    // picture under it is the metal, placed from each reading.
    idle = `${robot.movingJoints().length} joints - ${sim.scene}`;
    statusEl.textContent = idle;
    sim.onFrame(applyFrame);
  } catch (error) {
    statusEl.textContent = 'failed';
    jointsEl.textContent = String(error.message ?? error);
  }
  tick();
})();
