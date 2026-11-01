// Hermit, the handheld: the arm's claw, taken off the robot with its servo and
// camera bracket, and given the handheld's own body and jaw and the marker
// solid on its peg, its markers on it. Drawn by the front page lying on the
// desk and by the teleop page's hands view where the top camera sees it.

import * as THREE from 'three';
import { STLLoader } from './STLLoader.js';
import { drawPart } from './urdf.js';
import { HULL, OUTLINE } from './look.js';
import { addMarkers } from './markers.js';

export const PAPER = new THREE.MeshStandardMaterial({ color: 0xf6f2f3, roughness: 0.9 });

/**
 * One hand's Hermit, in its gripper link's frame.
 *
 * @param robot  the loaded description, whose `<side>_gripper_link` is cloned
 * @param parts  that hand's entry of /hermit: the mesh names, tool_T_solid and
 *               the marker solid's markers
 * @param meshes where the handheld's meshes are fetched from
 * @returns { hand, body, solid, setJaw } - the link's group, the handheld's
 *          body and the marker solid's group inside it, and a function that
 *          turns the jaw to an angle (rad)
 */
export async function hermit(robot, side, parts, meshes = '/hermit/meshes') {
  const loader = new STLLoader();
  const load = (name) => loader.loadAsync(`${meshes}/${name}`);
  const [body, jaw, solidMesh] = await Promise.all([load(parts.body), load(parts.jaw), load(parts.solid)]);
  const hand = robot.links.get(`${side}_gripper_link`).clone(true);
  const dress = { outlineMaterial: OUTLINE, hullMaterial: HULL };
  const axis = robot.joints.get(`${side}_gripper`).axis;
  let claw = null;
  let turning = null;
  hand.traverse((o) => {
    const swap = { [`claw_${side}.stl`]: body, [`jaw_${side}.stl`]: jaw }[o.userData.file];
    if (!swap) return;
    const material = o.children.find((c) => c.isMesh && c.material !== HULL)?.material;
    o.clear();
    drawPart(o, swap, material, dress);
    if (swap === body) claw = o;
    // The jaw's link hangs from the joint's turning group: it opens there.
    else turning = o.parent.parent;
  });
  // The solid where it hangs in the arm's tool frame (the gripper frame
  // link), carried into the gripper link's: the tool frame is fixed to it.
  robot.root.updateMatrixWorld(true);
  const linkToTool = robot.links.get(`${side}_gripper_link`).matrixWorld.clone().invert()
    .multiply(robot.links.get(`${side}_gripper_frame_link`).matrixWorld);
  const solid = new THREE.Group();
  linkToTool.clone().multiply(new THREE.Matrix4().set(...parts.tool_T_solid.flat()))
    .decompose(solid.position, solid.quaternion, solid.scale);
  drawPart(solid, solidMesh, PAPER, dress);
  if (parts.markers) addMarkers(solid, parts.markers);
  hand.add(solid);
  hand.traverse((o) => { if (o.isMesh && o.material !== HULL) o.castShadow = true; });
  const setJaw = (angle) => { turning?.quaternion.setFromAxisAngle(axis, angle); };
  return { hand, body: claw, solid, setJaw };
}
