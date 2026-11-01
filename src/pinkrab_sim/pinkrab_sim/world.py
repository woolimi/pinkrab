"""The task scene: the rig's own scene.xml, and what a take needs on top of it.

Built in memory with MjSpec from scene.xml, never written back: that file is
what the safety gate checks a pose against, and blocks or a hand in it would
change what the gate says. On top of it go

  - the gripper cameras, on each arm and on the Hermit, placed from camera.yaml
    by the construction the bracket is cut from, as five cube-map faces each;
  - what is moved, free bodies on the desk: the two blocks, or a block and a
    bowl for each hand, those a layout does not use parked out of sight behind
    the board;
  - a Hermit for each hand that makes takes: the claw link of the arm that
    follows it, with that hand's body, jaw and marker solid on it, carried by a
    mocap body and touching nothing;
  - the desk solid, where every recorded pose is measured from - for both
    hands where the rig reckons it, at the operator's chest - and the top
    camera looking down at the work;
  - a cutting mat for a desk, so a camera has something to see move;
  - the servos as the description fits them (kp, kv, armature, force range),
    the gripper's squeeze capped where Torque_Limit caps it, and contact
    friction fit for grasping.

Frames: the world is the robot's (scene.xml), cameras hand out OpenCV frames.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from pinkrab_core.config import SETTINGS as CORE
from pinkrab_core.follow import TOOL_TURN
from pinkrab_core.gripper_camera import mirrored as mirrored_camera
from pinkrab_core.jaw import shut_wide
from pinkrab_mujoco.scene import load_spec
from pinkrab_umi.config import SETTINGS as UMI

from .config import SETTINGS, BlockSpec, BowlSpec
from .lens import CV_TO_MJ, Fisheye, camera_spec, tcp_T_camera

# Geoms put in this group are seen by the top camera and not by a gripper
# camera: the camera's own bracket, which in the room sits behind the lens.
BEHIND_LENS_GROUP = 5
# The collision bit a carried cap keeps: everything but the arms has it.
CARRIED_BIT = 2
# blocks: one hand, the cap onto the green piece; bowls: both hands, each its
# block into its bowl.
SCENES = ("blocks", "bowls")
TWO_HANDS = ("left", "right")


def hermit(hand: str) -> str:
    """The name the Hermit in `hand` goes by: its body, and the prefix of all it carries."""
    return f"hermit_{hand}"


def hands_of(scene: str) -> tuple[str, ...]:
    """The hands a scene's takes are made with, and the arms that run them."""
    return TWO_HANDS if scene == "bowls" else (SETTINGS.arm,)


def mirrored(span: tuple[float, float], hand: str) -> tuple[float, float]:
    """A range across the desk written for the left hand, for `hand`."""
    return span if hand == "left" else (-span[1], -span[0])
# Behind the board, where nothing on the desk side sees: the blocks and bowls
# a layout leaves out stand in a row here.
PARK_X = -0.25


def quat_of(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).reshape(-1))
    return q


def pose_of(pos, quat) -> np.ndarray:
    T = np.eye(4)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat, dtype=float))
    T[:3, :3] = R.reshape(3, 3)
    T[:3, 3] = pos
    return T


def look_at(pos, target) -> np.ndarray:
    """An OpenCV camera frame at `pos` looking at `target`, image up toward +z."""
    forward = np.asarray(target, float) - np.asarray(pos, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([right, down, forward])
    T[:3, 3] = pos
    return T


def block_mesh(block: BlockSpec) -> list[float]:
    """A square frustum's corners, its bottom face on z = 0; a pyramid's apex."""
    b, t, h = block.bottom / 2, block.top / 2, block.height
    corners = [(-b, -b, 0), (b, -b, 0), (b, b, 0), (-b, b, 0)]
    corners += [(0.0, 0.0, h)] if t == 0 else [(-t, -t, h), (t, -t, h), (t, t, h), (-t, t, h)]
    return [c for corner in corners for c in corner]


def mat_texture(cell_px: int = 24, cells: int = 10) -> tuple[int, bytes]:
    """A self-healing cutting mat's grid, 10 cells of 10 mm, a bolder line each 50."""
    n = cell_px * cells
    img = np.empty((n, n, 3), dtype=np.uint8)
    img[:] = (46, 58, 60)
    for k in range(cells):
        width = 2 if k % 5 == 0 else 1
        shade = (150, 165, 165) if k % 5 == 0 else (92, 108, 110)
        img[k * cell_px:k * cell_px + width, :] = shade
        img[:, k * cell_px:k * cell_px + width] = shade
    return n, img.tobytes()


def gripper_camera_in_link(spec: mujoco.MjSpec, arm: str, pitch: float | None, yaw: float | None) -> np.ndarray:
    """The gripper camera in an arm's gripper link frame, OpenCV axes.

    The spec is the right hand's; the left arm is the right mirrored across its
    link's y, and the image with it - the fingers still enter from the bottom.
    """
    c = camera_spec()
    tcp = spec.site("right_tcp")
    link_T_tcp = pose_of(tcp.pos, tcp.quat)
    T = link_T_tcp @ tcp_T_camera(c["offset_out"], c["offset_back"],
                                  c["pitch"] if pitch is None else pitch,
                                  c["yaw"] if yaw is None else yaw)
    return mirrored_camera(T) if arm == "left" else T


def add_view_cameras(body, prefix: str, link_T_cam: np.ndarray) -> None:
    for name, R_face in Fisheye.face_rotations():
        R = link_T_cam[:3, :3] @ R_face @ CV_TO_MJ
        body.add_camera(name=f"{prefix}_{name}", pos=link_T_cam[:3, 3], quat=quat_of(R), fovy=90.0)


@dataclass
class Layout:
    """Where each thing starts, by name: (x, y, yaw) on the desk. `carried`
    goes onto or into `target`. A layout for both hands says each hand's
    (carried, target) in `hands`."""

    things: dict[str, tuple[float, float, float]]
    carried: str = SETTINGS.carried_block
    target: str = SETTINGS.target_block
    hands: dict[str, tuple[str, str]] = field(default_factory=dict)

    def of(self, hand: str) -> Layout:
        """The same desk, as the `hand` hand's take sees it: its own thing
        and target, everything else in the way."""
        carried, target = self.hands[hand]
        return Layout(self.things, carried, target)


@dataclass
class Thing:
    """What a plan and a judgement need of something moved: its height, half
    its width across the jaw, how high its frame stands over the desk when it
    rests, and its collision geoms (filled once compiled)."""

    name: str
    kind: str
    height: float
    half_width: float
    rest_z: float = 0.0
    geoms: list[int] = field(default_factory=list)


def add_bowl(spec: mujoco.MjSpec, bowl: BowlSpec) -> Thing:
    """A bowl of plain shapes, a free body whose frame is under its floor: a
    floor disc and a ring of wall slabs, each slab long enough to meet the
    next. A carried block keeps touching it."""
    body = spec.worldbody.add_body(name=bowl.name, pos=[0.0, 0.0, 0.0])
    body.add_freejoint(name=f"{bowl.name}_free")
    bits = CARRIED_BIT | 1
    radius = bowl.diameter / 2.0
    walls = bowl.height - bowl.wall
    floor_mass = bowl.mass * 0.3
    body.add_geom(name=f"{bowl.name}_floor", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[radius, bowl.wall / 2, 0],
                  pos=[0.0, 0.0, bowl.wall / 2], mass=floor_mass, rgba=list(bowl.rgba), condim=4,
                  contype=bits, conaffinity=bits)
    middle = radius - bowl.wall / 2
    half_long = middle * math.tan(math.pi / bowl.segments) * 1.08
    for k in range(bowl.segments):
        a = 2 * math.pi * k / bowl.segments
        R = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
        body.add_geom(name=f"{bowl.name}_wall{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[bowl.wall / 2, half_long, walls / 2],
                      pos=[middle * math.cos(a), middle * math.sin(a), bowl.wall + walls / 2], quat=quat_of(R),
                      mass=(bowl.mass - floor_mass) / bowl.segments, rgba=list(bowl.rgba), condim=4,
                      contype=bits, conaffinity=bits)
    return Thing(bowl.name, "bowl", bowl.height, radius)


def build_spec(arms: tuple[str, ...] = ("left",), hermits: tuple[str, ...] = ("left",),
               pitch: float | None = None, yaw: float | None = None,
               scene: str = "blocks") -> tuple[mujoco.MjSpec, dict[str, Thing]]:
    spec = load_spec(CORE.scene_file)
    spec.visual.map.znear = 0.002 / max(spec.stat.extent, 1e-6)
    # A headlight moves with the camera, and a fisheye made of five cameras
    # would be lit five ways, the seams between them drawn in light. The room's
    # own lights and an even ambient instead.
    spec.visual.headlight.diffuse = [0.0, 0.0, 0.0]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.visual.headlight.ambient = [0.42, 0.42, 0.42]
    spec.worldbody.add_light(name="room", pos=[0.4, 0.3, 1.6], dir=[0.0, -0.2, -1.0], diffuse=[0.5, 0.5, 0.5],
                             castshadow=False)

    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = SETTINGS.impratio
    grip = SETTINGS.servo_stall_nm * CORE.gripper_torque_limit / 1000.0
    for arm in ("left", "right"):
        bracket = spec.geom(f"camera_bracket_{arm}")
        if bracket is not None:
            bracket.group = BEHIND_LENS_GROUP
        actuator = spec.actuator(f"{arm}_gripper")
        actuator.forcerange = [-grip, grip]
        spec.joint(f"{arm}_gripper").actfrcrange = [-grip, grip]
    for arm in arms:
        add_view_cameras(spec.body(f"{arm}_gripper_link"), f"{arm}_gripcam",
                          gripper_camera_in_link(spec, arm, pitch, yaw))

    # The desk as the mat it is.
    size, data = mat_texture()
    texture = spec.add_texture(name="mat", type=mujoco.mjtTexture.mjTEXTURE_2D, width=size, height=size,
                               nchannel=3)
    texture.data = data
    desk_material = spec.material("desk")
    desk_material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "mat"
    desk_material.texrepeat = [10.0, 10.0]
    desk_material.texuniform = True

    for scenery in ("desk", "pegboard"):
        spec.geom(scenery).contype = CARRIED_BIT | 1
        spec.geom(scenery).conaffinity = CARRIED_BIT | 1
    things: dict[str, Thing] = {}
    if scene == "blocks":
        for block in SETTINGS.blocks:
            spec.add_mesh(name=f"block_{block.name}", uservert=block_mesh(block))
            body = spec.worldbody.add_body(name=block.name, pos=[0.0, 0.0, 0.0])
            body.add_freejoint(name=f"{block.name}_free")
            bits = 1 if block.name == SETTINGS.carried_block else CARRIED_BIT | 1
            body.add_geom(name=block.name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"block_{block.name}",
                          mass=block.mass, rgba=list(block.rgba), friction=[block.friction, 0.005, 0.0001],
                          condim=4, contype=bits, conaffinity=bits)
            things[block.name] = Thing(block.name, "block", block.height, block.bottom / 2.0)
        carriable = [SETTINGS.carried_block]
    elif scene == "bowls":
        for block in SETTINGS.bowl_blocks:
            spec.add_mesh(name=f"block_{block.name}", uservert=block_mesh(block))
            body = spec.worldbody.add_body(name=block.name, pos=[0.0, 0.0, 0.0])
            body.add_freejoint(name=f"{block.name}_free")
            body.add_geom(name=block.name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"block_{block.name}",
                          mass=block.mass, rgba=list(block.rgba), friction=[block.friction, 0.005, 0.0001],
                          condim=4, contype=1, conaffinity=1)
            things[block.name] = Thing(block.name, "block", block.height, block.bottom / 2.0)
        for bowl in SETTINGS.bowls:
            things[bowl.name] = add_bowl(spec, bowl)
        carriable = [block.name for block in SETTINGS.bowl_blocks]
    else:
        raise ValueError(f"scene is one of {', '.join(SCENES)}, not {scene!r}")

    # A weld per arm from its claw to each thing it may carry, off until a
    # grasp is judged made (evaluate.Carry): the thing then goes where the
    # claw goes. While it does it is taken out of the claw's and the jaw's
    # collisions - a jaw still squeezing a welded thing only fights the weld -
    # and keeps touching the desk, the board and what it is put on or in,
    # which carry the second collision bit for it.
    for arm in arms:
        for name in carriable:
            spec.add_equality(name=f"carry_{arm}_{name}", type=mujoco.mjtEq.mjEQ_WELD,
                              name1=f"{arm}_gripper_link", name2=name, objtype=mujoco.mjtObj.mjOBJ_BODY,
                              active=False)

    head_pos, head_target, head_fovy = head_view(scene)
    head = look_at(head_pos, head_target)
    spec.worldbody.add_camera(name="head", pos=head[:3, 3], quat=quat_of(head[:3, :3] @ CV_TO_MJ), fovy=head_fovy)

    for place in hermits:
        add_hermit(spec, place, pitch, yaw)
    return spec, things


def head_view(scene: str) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """Where the top camera stands, what it looks at and its vertical field
    of view: beside the working arm for one hand, wide and in the middle for two."""
    if scene == "bowls":
        return SETTINGS.two_hand_head_pos, SETTINGS.two_hand_head_target, SETTINGS.two_hand_head_fovy_deg
    side = 1.0 if SETTINGS.arm == "left" else -1.0
    (px, py, pz), (tx, ty, tz) = SETTINGS.head_pos, SETTINGS.head_target
    return (px, side * py, pz), (tx, side * ty, tz), SETTINGS.head_fovy_deg


def add_hermit(spec: mujoco.MjSpec, place: str, pitch: float | None, yaw: float | None) -> None:
    """The handheld of the `place` hand, laid out as that arm's gripper link is.

    Its frame IS a gripper link frame, so the camera and the jaw's hinge sit
    where they sit on the arm that follows it, and its claw and jaw are on the
    sides that arm's are. What it wears is the handheld's own - the printed
    body around the claw, its jaw, and the marker solid on its peg. Its TCP is
    in the handheld's convention, which the arm's tool frame is TOOL_TURN of.
    """
    claw = spec.geom(f"claw_{place}")
    link_T_claw = pose_of(claw.pos, claw.quat)
    meshes = UMI.mesh_dir
    name = hermit(place)
    for part in (f"handheld_body_{place}", f"handheld_jaw_{place}", f"marker_solid_handheld_{place}"):
        spec.add_mesh(name=f"hermit_{part}", file=str(meshes / f"{part}.stl"))

    body = spec.worldbody.add_body(name=name, mocap=True, pos=[0.3, 0.6 if place == "left" else -0.6, 0.3])
    pink = [0.8784, 0.2941, 0.5569, 1.0]
    visual = dict(contype=0, conaffinity=0, density=0)
    body.add_geom(name=f"{name}_body", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"hermit_handheld_body_{place}",
                  pos=claw.pos, quat=claw.quat, rgba=pink, group=2, **visual)
    body.add_geom(name=f"{name}_bracket", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"camera_bracket_{place}",
                  pos=claw.pos, quat=claw.quat, rgba=pink, group=BEHIND_LENS_GROUP, **visual)
    servo = next(g for g in spec.body(f"{place}_gripper_link").geoms
                 if g.meshname == "servo" and g.contype == 0)
    body.add_geom(name=f"{name}_servo", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="servo",
                  pos=servo.pos, quat=servo.quat, rgba=[0.1, 0.1, 0.1, 1.0], group=2, **visual)
    solid = link_T_claw @ UMI.for_solid(f"handheld_{place}").claw_T_solid
    body.add_geom(name=f"{name}_solid", type=mujoco.mjtGeom.mjGEOM_MESH,
                  meshname=f"hermit_marker_solid_handheld_{place}", pos=solid[:3, 3], quat=quat_of(solid[:3, :3]),
                  rgba=[0.93, 0.93, 0.93, 1.0], group=2, **visual)
    tcp = spec.site(f"{place}_tcp")
    link_T_tcp = pose_of(tcp.pos, tcp.quat)
    body.add_site(name=f"{name}_tcp", pos=tcp.pos, quat=quat_of(link_T_tcp[:3, :3] @ TOOL_TURN[place].T),
                  size=[0.004, 0, 0], group=4)

    arm_jaw = spec.body(f"{place}_jaw_link")
    jaw = body.add_body(name=f"{name}_jaw", pos=arm_jaw.pos, quat=arm_jaw.quat)
    jaw.explicitinertial = True
    jaw.mass = 0.012
    jaw.inertia = [6.6e-6, 5.3e-6, 1.9e-6]
    hinge = spec.joint(f"{place}_gripper")
    jaw.add_joint(name=f"{name}_gripper", axis=hinge.axis, range=hinge.range)
    arm_jaw_geom = spec.geom(f"jaw_{place}")
    jaw.add_geom(name=f"{name}_jaw", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"hermit_handheld_jaw_{place}",
                 pos=arm_jaw_geom.pos, quat=arm_jaw_geom.quat, rgba=pink, group=2, **visual)
    actuator = spec.add_actuator(name=f"{name}_gripper", target=f"{name}_gripper",
                                 trntype=mujoco.mjtTrn.mjTRN_JOINT)
    actuator.set_to_position(kp=2.0, kv=0.05)
    actuator.ctrlrange = hinge.range
    actuator.ctrllimited = True

    add_view_cameras(body, f"{name}_gripcam", gripper_camera_in_link(spec, place, pitch, yaw))


def desk_pose(scene: str = "blocks") -> np.ndarray:
    """The desk solid's frame in the robot's: z up, turned by desk.yaml's yaw.
    For both hands it stands where the rig reckons a solid set at the
    operator's chest (desk_solid.world_T_desk), so the takes go through the
    shelf, convert and the start of a run as the rig's do; for one, off to
    the side of where the blocks go."""
    desk = UMI.for_solid(UMI.desk_solid)
    if scene == "bowls":
        from pinkrab_umi.desk_solid import world_T_desk

        return world_T_desk((SETTINGS.desk_yaw_deg, desk.cut_mm), desk.face_size_mm)
    c, s = math.cos(math.radians(SETTINGS.desk_yaw_deg)), math.sin(math.radians(SETTINGS.desk_yaw_deg))
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    x, y, z = SETTINGS.desk_origin
    T[:3, 3] = [x, y if SETTINGS.arm == "left" else -y, z]
    return T


class World:
    """The task scene compiled, with its cameras ready to be drawn."""

    def __init__(self, arms: tuple[str, ...] = ("left",), hermits: tuple[str, ...] = ("left",),
                 pitch: float | None = None, yaw: float | None = None, scene: str = "blocks") -> None:
        self.scene = scene
        self.spec, self.things = build_spec(arms, hermits, pitch, yaw, scene)
        self.fisheye = Fisheye(SETTINGS.lens_spec, SETTINGS.video_scale, SETTINGS.face_oversample)
        n = self.fisheye.face
        hw, hh = SETTINGS.head_size
        self.spec.visual.global_.offwidth = max(n, hw, 1280)
        self.spec.visual.global_.offheight = max(n, hh, 960)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.arms = arms
        self.hermits = hermits
        self._face_renderer = mujoco.Renderer(self.model, n, n)
        self._head_renderer = mujoco.Renderer(self.model, hh, hw)
        self._views: dict[tuple[int, int], mujoco.Renderer] = {}
        self._gripper_option = mujoco.MjvOption()
        self._gripper_option.geomgroup[BEHIND_LENS_GROUP] = 0
        self._head_option = mujoco.MjvOption()
        self._head_option.geomgroup[BEHIND_LENS_GROUP] = 1
        self.desk = desk_pose(scene)
        self.head = look_at(*head_view(scene)[:2])
        m = self.model
        for thing in self.things.values():
            body = m.body(thing.name).id
            thing.geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == body and m.geom_contype[g]]
        self._free = {n: m.jnt_qposadr[m.joint(f"{n}_free").id] for n in self.things}
        self._free_v = {n: m.jnt_dofadr[m.joint(f"{n}_free").id] for n in self.things}
        # Each Hermit: its mocap slot, its jaw's place in qpos and qvel, its
        # jaw's actuator, and its TCP in its own frame.
        self._hermit = {}
        for hand in hermits:
            name = hermit(hand)
            self._hermit[hand] = (int(m.body_mocapid[m.body(name).id]),
                                  m.jnt_qposadr[m.joint(f"{name}_gripper").id],
                                  m.jnt_dofadr[m.joint(f"{name}_gripper").id],
                                  m.actuator(f"{name}_gripper").id,
                                  self._site_in_body(f"{name}_tcp"))
        # Each arm's jaw band, shut to wide - which a Hermit's jaw shares.
        self.bands = {arm: shut_wide(*(float(v) for v in m.jnt_range[m.joint(f"{arm}_gripper").id]))
                      for arm in TWO_HANDS}

    def close(self) -> None:
        for r in (self._face_renderer, self._head_renderer, *self._views.values()):
            r.close()
        self._views.clear()

    def _site_in_body(self, site: str) -> np.ndarray:
        s = self.model.site(site)
        return pose_of(s.pos, s.quat)

    # --- poses -------------------------------------------------------------------------
    def site_pose(self, site: str) -> np.ndarray:
        s = self.model.site(site).id
        T = np.eye(4)
        T[:3, :3] = self.data.site_xmat[s].reshape(3, 3)
        T[:3, 3] = self.data.site_xpos[s]
        return T

    def body_pose(self, body: str) -> np.ndarray:
        b = self.model.body(body).id
        T = np.eye(4)
        T[:3, :3] = self.data.xmat[b].reshape(3, 3)
        T[:3, 3] = self.data.xpos[b]
        return T

    def tcp(self, arm: str) -> np.ndarray:
        return self.site_pose(f"{arm}_tcp")

    # --- the Hermits -----------------------------------------------------------------------
    def hold_hermit(self, hand: str, world_T_tcp: np.ndarray, aperture: float) -> None:
        """Put the `hand` Hermit's TCP at a pose, its jaw at an opening (0 shut .. 1 wide)."""
        mocap, jaw, jaw_v, act, link_T_tcp = self._hermit[hand]
        body = world_T_tcp @ np.linalg.inv(link_T_tcp)
        self.data.mocap_pos[mocap] = body[:3, 3]
        self.data.mocap_quat[mocap] = quat_of(body[:3, :3])
        band = self.bands[hand]
        angle = band[0] + float(aperture) * (band[1] - band[0])
        self.data.ctrl[act] = angle
        self.data.qpos[jaw] = angle
        self.data.qvel[jaw_v] = 0.0

    # --- the blocks -------------------------------------------------------------------------
    def place_block(self, name: str, world_T: np.ndarray) -> None:
        a, v = self._free[name], self._free_v[name]
        self.data.qpos[a:a + 3] = world_T[:3, 3]
        self.data.qpos[a + 3:a + 7] = quat_of(world_T[:3, :3])
        self.data.qvel[v:v + 6] = 0.0

    def block_pose(self, name: str) -> np.ndarray:
        return self.body_pose(name)

    def speed(self, name: str) -> float:
        """How fast a thing moves, metres a second."""
        v = self._free_v[name]
        return float(np.linalg.norm(self.data.qvel[v:v + 3]))

    def lay_out(self, layout: Layout) -> None:
        parked = [n for n in self.things if n not in layout.things]
        spots = {n: (PARK_X, -0.3 + 0.6 * k / max(1, len(parked) - 1), 0.0) for k, n in enumerate(parked)}
        for name, (x, y, yaw) in {**layout.things, **spots}.items():
            T = np.eye(4)
            c, s = math.cos(yaw), math.sin(yaw)
            T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
            T[:3, 3] = [x, y, self.things[name].rest_z]
            self.place_block(name, T)

    # --- the arms -----------------------------------------------------------------------
    def set_joints(self, pose: dict[str, float]) -> None:
        """Joints outright, radians by name; the actuators pointed at the same."""
        for name, value in pose.items():
            self.data.qpos[self.model.jnt_qposadr[self.model.joint(name).id]] = value
            a = self.model.actuator(name).id
            self.data.ctrl[a] = value
        mujoco.mj_forward(self.model, self.data)

    def command(self, pose: dict[str, float]) -> None:
        for name, value in pose.items():
            self.data.ctrl[self.model.actuator(name).id] = value

    def joints(self, names) -> np.ndarray:
        return np.array([self.data.qpos[self.model.jnt_qposadr[self.model.joint(n).id]] for n in names])

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    def step(self, seconds: float) -> None:
        for _ in range(max(1, int(round(seconds / self.model.opt.timestep)))):
            mujoco.mj_step(self.model, self.data)

    # --- pictures --------------------------------------------------------------------------
    def gripper_image(self, which: str) -> np.ndarray:
        """The fisheye frame from an arm's camera (`left`, `right`) or a
        Hermit's (`hermit_left`, `hermit_right`), RGB."""
        faces = []
        for name, _ in Fisheye.face_rotations():
            self._face_renderer.update_scene(self.data, camera=f"{which}_gripcam_{name}",
                                             scene_option=self._gripper_option)
            faces.append(self._face_renderer.render())
        return self.fisheye.compose(faces)

    def head_image(self) -> np.ndarray:
        self._head_renderer.update_scene(self.data, camera="head", scene_option=self._head_option)
        return self._head_renderer.render()

    def view(self, camera: str, size: tuple[int, int] = (640, 480)) -> np.ndarray:
        """One of scene.xml's own cameras (front, side, top), for looking at the take."""
        renderer = self._views.get(size)
        if renderer is None:
            renderer = self._views[size] = mujoco.Renderer(self.model, size[1], size[0])
        renderer.update_scene(self.data, camera=camera, scene_option=self._head_option)
        return renderer.render()
