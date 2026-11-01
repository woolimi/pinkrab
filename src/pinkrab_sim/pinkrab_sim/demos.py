"""Demonstrations by a Hermit with no arm behind it, written as the recorder writes them.

A take is planned the way a hand does one: from wherever the hand is over the
desk, over to the cap and down along the tool's own axis to it, shut, up,
across, down onto the green piece, open, away. With both hands (`bowls`) each
hand does the same at once with its own block and its own bowl, and the take
is kept only if the two arms would not have run into each other.
The plan is kept only if the ARM could have followed it - the tool pose
of every frame reached by `track()` from the frame before, as the rollout
would, with no joint asked to jump - and if the arm's own claw, standing
where the Hermit stands, would not have gone into the desk. A take the arm
could not have made is never recorded.

The Hermit touches nothing. What it holds rides with it from the moment the
jaw shuts to the moment it opens, and physics has it from there: the cap
drops the last few millimetres onto the green piece and settles, a block
drops into its bowl, or does not, and such a take is not kept either.

What is written is what `pinkrab-umi live` writes - head.mp4 and each hand's
gripper video, frames.jsonl, poses.jsonl, events.jsonl, meta.json - so
`pinkrab-umi convert --from` reads it unchanged. The poses carry the
tracker's noise; the clocks are exact, each reading stamped so that convert,
subtracting the latencies it knows, lands it on its own frame.
"""
from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from pinkrab_core.config import SETTINGS as CORE
from pinkrab_core.follow import TOOL_TURN
from pinkrab_core.kinematics import ArmKinematics
from pinkrab_core.origin import ready_joints
from pinkrab_umi.config import SETTINGS as UMI
from pinkrab_umi.episode import Encoder

from .config import SETTINGS
from .world import TWO_HANDS, Layout, World, hands_of, hermit, mirrored

# Where the hand comes from: the heading it takes to a block is roughly the
# line from there to it. From the robot's side it is the arm's shoulder, read
# off the scene; from across the table, a person sitting this far out from the
# board's back, facing the robot - a take no arm on this board can make.
OPPOSITE_X = 0.95
# The rollout's own limits on a tick, and what the gate refuses.
ORIENTATION_WEIGHT = 0.1
REACH_M = 0.005
REACH_DEG = 5.0
JUMP_DEG = 15.0
# How far the arm's claw may go into the desk or the green piece before the
# grasp is raised.
DESK_PENETRATION_M = 0.0005
# A cap put down is on the green piece when its centre is this near the
# piece's axis, its bottom this near the piece's top, and it stands this
# straight.
PLACED_MM = 10.0
PLACED_HEIGHT_MM = 4.0
PLACED_TILT_DEG = 12.0
# A thing is in a bowl when its centre is within this share of the bowl's
# half width of the bowl's axis, its bottom this far under the rim.
IN_BOWL = 0.75
IN_BOWL_DEPTH = 0.005
# Half the height of the slice of a thing the claw's face meets it in.
GRASP_SLAB_M = 0.008


class DemoError(RuntimeError):
    pass


# --- geometry -----------------------------------------------------------------------------
def tool_rotation(heading: float, pitch: float) -> np.ndarray:
    """The tool heading `heading` round the vertical, its axis `pitch` below level.

    The jaw's opening axis stays level and the tool's y points up and away,
    which is where the camera is: above the fingers, looking down past them.
    """
    z = np.array([math.cos(pitch) * math.cos(heading), math.cos(pitch) * math.sin(heading), -math.sin(pitch)])
    x = np.array([-math.sin(heading), math.cos(heading), 0.0])
    return np.column_stack([x, np.cross(z, x), z])


def yaw_pose(x: float, y: float, z: float, yaw: float) -> np.ndarray:
    T = np.eye(4)
    c, s = math.cos(yaw), math.sin(yaw)
    T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    T[:3, 3] = [x, y, z]
    return T


def faces(block_yaw: float, x: float, y: float, origin: tuple[float, float]) -> tuple[int, int]:
    """The block's two faces nearest the line from `origin`, where the hand
    comes from, as quarter turns from its yaw: the nearest, then the next,
    round the side of the block."""
    ox, oy = origin
    turns = (math.atan2(y - oy, x - ox) - block_yaw) / (math.pi / 2)
    k = round(turns)
    return k, k + 1 if turns >= k else k - 1


def face_heading(block_yaw: float, x: float, y: float, rng: np.random.Generator,
                 origin: tuple[float, float], second: bool = False) -> float:
    """The heading square to one of the block's `faces`, the nearest unless
    `second`: the hand turns its wrist to meet the block."""
    k = faces(block_yaw, x, y, origin)[second]
    return block_yaw + k * math.pi / 2 + rng.normal(0.0, math.radians(2.0))


def minimum_jerk(s: np.ndarray) -> np.ndarray:
    return s ** 3 * (10 - 15 * s + 6 * s ** 2)


@dataclass
class Key:
    t: float
    T: np.ndarray
    aperture: float


@dataclass
class Plan:
    layout: Layout
    keys: list[Key]
    grasp_t: float
    release_t: float
    tcp_T_thing: np.ndarray
    info: dict = field(default_factory=dict)

    @property
    def seconds(self) -> float:
        return self.keys[-1].t


class Tremor:
    """A held hand's slow wander: three sines per axis, random in rate and phase."""

    def __init__(self, rng: np.random.Generator, metres: float, radians: float) -> None:
        self.freq = rng.uniform(0.3, 1.6, size=(6, 3))
        self.phase = rng.uniform(0.0, 2 * math.pi, size=(6, 3))
        self.amp = np.array([metres] * 3 + [radians] * 3)[:, None] / math.sqrt(3.0)

    def __call__(self, t: float) -> np.ndarray:
        wave = (self.amp * np.sin(2 * math.pi * self.freq * t + self.phase)).sum(axis=1)
        T = np.eye(4)
        T[:3, :3] = Rotation.from_rotvec(wave[3:]).as_matrix()
        T[:3, 3] = wave[:3]
        return T


class Trajectory:
    """The plan as a tool pose and an opening at any time."""

    def __init__(self, plan: Plan, tremor: Tremor | None) -> None:
        self.plan = plan
        self.tremor = tremor
        self.times = np.array([k.t for k in plan.keys])

    def at(self, t: float) -> tuple[np.ndarray, float]:
        keys = self.plan.keys
        i = int(np.clip(np.searchsorted(self.times, t, side="right") - 1, 0, len(keys) - 2))
        a, b = keys[i], keys[i + 1]
        s = 0.0 if b.t <= a.t else float(np.clip(minimum_jerk(np.clip((t - a.t) / (b.t - a.t), 0.0, 1.0)), 0.0, 1.0))
        T = np.eye(4)
        T[:3, 3] = (1 - s) * a.T[:3, 3] + s * b.T[:3, 3]
        T[:3, :3] = Slerp([0.0, 1.0], Rotation.from_matrix([a.T[:3, :3], b.T[:3, :3]]))(s).as_matrix()
        aperture = (1 - s) * a.aperture + s * b.aperture
        if self.tremor is not None:
            T = T @ self.tremor(t)
        return T, aperture


# --- planning -----------------------------------------------------------------------------
class Planner:
    """Takes the arm could have made, and what its claw would have touched."""

    def __init__(self, world: World, viewpoint: str = "robot", two_faces: bool = False,
                 arm: str | None = None) -> None:
        self.world = world
        self.arm = arm or SETTINGS.arm
        # The opening the hand waits at before it shuts.
        self.open = SETTINGS.two_hand_open_aperture if world.scene == "bowls" else SETTINGS.open_aperture
        # A block is met square to a face. One bound for a bowl is held by its
        # side against the claw and let go over the bowl; the cap is held where
        # the takes hold it and set down.
        self.side_hold = self.into = world.scene == "bowls"
        self.chain = ArmKinematics(self.arm)
        self.ready = ready_joints(self.arm, self.chain.joint_names)
        m = world.model
        self._claw = {i for i in range(m.ngeom) if m.geom_contype[i]
                      and m.geom_bodyid[i] in (m.body(f"{self.arm}_gripper_link").id,
                                                m.body(f"{self.arm}_jaw_link").id)}
        self._jaw = {i for i in self._claw if m.geom_bodyid[i] == m.body(f"{self.arm}_jaw_link").id}
        self._desk = m.geom("desk").id
        at_zero = mujoco.MjData(m)
        mujoco.mj_kinematics(m, at_zero)
        sx, sy = (float(v) for v in at_zero.xanchor[m.joint(f"{self.arm}_shoulder_pan").id][:2])
        self.origin = {"robot": (sx, sy), "opposite": (OPPOSITE_X, sy)}[viewpoint]
        # Blocks taken by either of the two faces nearest the hand, half the
        # takes each: two right ways from one picture.
        self.two_faces = two_faces

    def layout(self, rng: np.random.Generator) -> Layout:
        """The blocks somewhere on the desk."""
        s = SETTINGS
        for _ in range(200):
            spots = [(rng.uniform(*s.block_x), rng.uniform(*s.block_y)) for _ in range(2)]
            if math.dist(*spots) >= s.block_gap:
                yaw = [rng.uniform(-1, 1) * math.radians(s.block_yaw_deg) for _ in range(2)]
                return Layout({s.carried_block: (*spots[0], yaw[0]), s.target_block: (*spots[1], yaw[1])})
        raise DemoError("no layout with the blocks far enough apart")

    # The arm's claw, standing where the Hermit would -------------------------------------------
    def _pose_arm(self, joints_deg: np.ndarray, aperture: float) -> None:
        w = self.world
        pose = {n: math.radians(v) for n, v in zip(self.chain.joint_names, joints_deg)}
        band = w.bands[self.arm]
        pose[f"{self.arm}_gripper"] = band[0] + aperture * (band[1] - band[0])
        w.set_joints(pose)

    def _touching(self, geoms: set[int], others: set[int]) -> float:
        """The deepest press between the two sets, metres; 0 when apart."""
        d = self.world.data
        deepest = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist < 0 and ((c.geom1 in geoms and c.geom2 in others) or (c.geom2 in geoms and c.geom1 in others)):
                deepest = max(deepest, -float(c.dist))
        return deepest

    def tool(self, T: np.ndarray) -> np.ndarray:
        """The arm's tool pose for a hand's: the handheld's convention turned
        into this arm's, as the rollout turns a policy's."""
        out = T.copy()
        out[:3, :3] = T[:3, :3] @ TOOL_TURN[self.arm]
        return out

    def reach(self, T: np.ndarray) -> np.ndarray | None:
        target = self.tool(T)
        q = self.chain.solve(target, seed=self.ready, restarts=8, orientation_weight=ORIENTATION_WEIGHT)
        err, ang = self.chain.reach_error(target, q)
        return q if err < REACH_M and ang < REACH_DEG else None

    def track(self, T: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, float, float]:
        """One tick toward a hand's tool pose from `q`: the joints, and how far
        off they leave the tool (metres, degrees)."""
        target = self.tool(T)
        q = self.chain.track(target, q, orientation_weight=ORIENTATION_WEIGHT)
        return (q, *self.chain.reach_error(target, q))

    def reach_toward_claw(self, layout: Layout, axis: np.ndarray, height: float) -> float:
        """How far the carried thing reaches from its axis toward the claw
        (along `axis`, the arm tool's x in the world), in a slab of it about
        `height` over the desk - its own shape, turned as it lies, and not
        its widest side."""
        w = self.world
        m, d = w.model, w.data
        w.lay_out(layout)
        mujoco.mj_kinematics(m, d)
        centre = w.block_pose(layout.carried)[:3, 3]
        far = 0.0
        for g in w.things[layout.carried].geoms:
            mesh = m.geom_dataid[g]
            if mesh < 0:
                continue
            a, n = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
            v = m.mesh_vert[a:a + n] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            slab = v[np.abs(v[:, 2] - height) < GRASP_SLAB_M]
            # A block's sides stand straight, its corners top and bottom only:
            # no corner in the slab, and the whole block is its slab.
            v = slab if len(slab) or self.world.things[layout.carried].kind != "block" else v
            if len(v):
                far = max(far, float(np.max((v - centre) @ axis)))
        return far

    def holds(self, layout: Layout, R: np.ndarray, height: float) -> list[np.ndarray]:
        """Where the carried thing's grasp point may sit in the arm's tool
        frame, tried in turn: the cap where the takes hold it; a block bound for
        a bowl with its side a millimetre off the claw's inner face, and then
        further while the claw is still in it."""
        if not self.side_hold:
            return [np.array(SETTINGS.grasp_in_tcp)]
        axis = R @ (TOOL_TURN[self.arm] @ np.array([1.0, 0.0, 0.0]))
        side = self.reach_toward_claw(layout, axis, height)
        z = SETTINGS.grasp_in_tcp[2]
        return [np.array([SETTINGS.claw_face_in_tcp - side - gap, 0.0, z]) for gap in np.arange(0.001, 0.0101, 0.002)]

    def grasp(self, layout: Layout, heading: float, pitch: float) -> tuple[np.ndarray, float, np.ndarray]:
        """The TCP pose that holds the carried thing, the opening its sides
        are met at, and the arm's joints there. Raised until the claw clears
        the desk; the claw and the jaw may touch nothing else on it."""
        thing = self.world.things[layout.carried]
        mine = set(thing.geoms)
        others = {g for n in layout.things if n != layout.carried for g in self.world.things[n].geoms}
        x, y, yaw = layout.things[layout.carried]
        R = tool_rotation(heading, pitch)
        start = SETTINGS.thing_grasp_height if self.side_hold else SETTINGS.grasp_height
        for fraction in np.arange(start, 0.81, 0.05):
            point = np.array([x, y, fraction * thing.height])
            into_desk = False
            for hold in self.holds(layout, R, point[2]):
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = point - R @ (TOOL_TURN[self.arm] @ hold)
                q = self.reach(T)
                if q is None:
                    raise DemoError(f"the arm cannot reach the {layout.carried}")
                self.world.lay_out(layout)
                self._pose_arm(q, self.open)
                if self._touching(self._claw, {self._desk}) > DESK_PENETRATION_M:
                    into_desk = True
                    break
                if self._touching(self._claw, others) > DESK_PENETRATION_M:
                    raise DemoError("the claw is in something else on the desk")
                if self._touching(self._claw, mine) > DESK_PENETRATION_M:
                    continue
                for aperture in np.arange(self.open, -0.001, -0.01):
                    self._pose_arm(q, aperture)
                    if self._touching(self._jaw, others) > 0:
                        raise DemoError("the jaw shuts on something else on the desk")
                    if self._touching(self._jaw, mine) > 0:
                        return T, float(aperture), q
                raise DemoError(f"the jaw shuts without meeting the {layout.carried}")
            if not into_desk:
                raise DemoError(f"the claw is in the {layout.carried} with the jaw open")
        raise DemoError("no grasp height keeps the claw out of the desk")

    def heading_to(self, x: float, y: float, yaw: float, rng: np.random.Generator,
                   came_from: tuple[float, float], second: bool = False, square: bool = True) -> float:
        """The tool's heading to something at (x, y): square to a block's face
        nearest the line from the shoulder; not `square` - for a bowl - the way
        from where the hand `came_from`, give or take a hand's wander.
        `second` takes a block by its next nearest face."""
        if square:
            return face_heading(yaw, x, y, rng, self.origin, second)
        ox, oy = came_from
        return math.atan2(y - oy, x - ox) + rng.normal(0.0, math.radians(SETTINGS.heading_deg))

    def blocks_start(self, rng: np.random.Generator, layout: Layout) -> tuple[float, float, float, float, float]:
        """Where a block take starts, (x, y, height, heading, pitch): somewhere
        clear of the blocks, facing out from the shoulder."""
        s = SETTINGS
        if self.world.scene == "bowls":
            xs, ys = s.two_hand_start_x, mirrored(s.two_hand_start_y, self.arm)
        else:
            xs, ys = s.start_x, s.start_y
        for _ in range(50):
            sx, sy = rng.uniform(*xs), rng.uniform(*ys)
            if all(math.hypot(x - sx, y - sy) >= s.start_clear for x, y, _ in layout.things.values()):
                break
        else:
            raise DemoError("no start clear of what is on the desk")
        ox, oy = self.origin
        toward = math.atan2(sy - oy, sx - ox) + rng.normal(0, math.radians(s.heading_deg))
        start_pitch = math.radians(rng.uniform(*s.start_pitch_deg))
        return sx, sy, rng.uniform(*s.start_height), toward, start_pitch

    def plan(self, rng: np.random.Generator, layout: Layout | None = None, start: tuple | None = None,
             second: bool | None = None) -> Plan:
        """A take on `layout` from `start`, each drawn when not given. A block
        is taken by its `second` nearest face when that is true; left None,
        by either half the time if the planner takes two faces."""
        s = SETTINGS
        layout = layout or self.layout(rng)
        thing = self.world.things[layout.carried]
        target = self.world.things[layout.target]
        cx, cy, cyaw = layout.things[layout.carried]
        bx, by, byaw = layout.things[layout.target]
        pitch = math.radians(rng.uniform(*s.pitch_deg))
        if second is None:
            second = self.two_faces and bool(rng.random() < 0.5)
        heading = self.heading_to(cx, cy, cyaw, rng, start[:2] if start else self.origin, second)
        T_grasp, contact, _ = self.grasp(layout, heading, pitch)
        # The hand shuts a little past where the jaw meets the thing; on a
        # block bound for a bowl further, so an arm a few millimetres off the
        # hand's pose still squeezes it against the claw.
        hold = max(0.0, contact - s.thing_squeeze) if self.side_hold else min(s.grasp_aperture, contact - 0.03)
        T_thing = yaw_pose(cx, cy, thing.rest_z, cyaw)
        tcp_T_thing = np.linalg.inv(T_grasp) @ T_thing

        # Onto the green piece, a few millimetres over it; into a bowl, its
        # bottom just over the rim.
        place_heading = self.heading_to(bx, by, byaw, rng, (cx, cy), square=not self.into)
        R_place = tool_rotation(place_heading, pitch)
        thing_target = np.eye(4)
        thing_target[:3, :3] = R_place @ tcp_T_thing[:3, :3]
        thing_target[:3, 3] = [bx, by, (target.height + s.bowl_drop + thing.rest_z) if self.into
                               else target.height + s.drop]
        T_place = thing_target @ np.linalg.inv(tcp_T_thing)
        q_place = self.reach(T_place)
        if q_place is None:
            raise DemoError(f"the arm cannot reach the {layout.target}")
        if self.into:
            self.world.lay_out(layout)
            self._pose_arm(q_place, hold)
            if self._touching(self._claw, set(target.geoms)) > DESK_PENETRATION_M:
                raise DemoError("the claw is in the bowl where it lets go")

        up = np.eye(4)
        up[2, 3] = s.thing_carry_height if self.into else s.carry_height

        def back(T, distance):
            out = T.copy()
            out[:3, 3] -= T[:3, 2] * distance
            return out

        if start is None:
            start = self.blocks_start(rng, layout)
        sx, sy, sz, toward, start_pitch = start
        T_start = np.eye(4)
        T_start[:3, :3] = tool_rotation(toward, start_pitch)
        T_start[:3, 3] = [sx, sy, sz]
        T_retreat = T_place.copy()
        level = np.array([T_place[0, 2], T_place[1, 2], 0.0])
        level /= max(np.linalg.norm(level), 1e-9)
        T_retreat[:3, 3] += -level * s.retreat_back + np.array([0.0, 0.0, s.retreat_up])

        keys, t = [], 0.0

        def key(T, aperture, seconds):
            nonlocal t
            t += seconds
            keys.append(Key(t, T, aperture))

        opened = self.open
        keys.append(Key(0.0, T_start, opened))
        key(T_start, opened, rng.uniform(0.2, 0.4))
        key(back(T_grasp, s.approach), opened, rng.uniform(*s.reach_s))
        key(T_grasp, opened, rng.uniform(*s.descend_s))
        key(T_grasp, hold, rng.uniform(*s.close_s))
        grasp_t = t
        key(up @ T_grasp, hold, rng.uniform(*s.descend_s))
        key(up @ T_place, hold, rng.uniform(*s.carry_s))
        key(T_place, hold, rng.uniform(*s.lower_s))
        release_t = t
        key(T_place, opened, rng.uniform(*s.open_s))
        key(T_retreat, opened, rng.uniform(*s.retreat_s))
        key(T_retreat, opened, rng.uniform(0.2, 0.4))
        return Plan(layout, keys, grasp_t, release_t, tcp_T_thing, {
            "pitch_deg": math.degrees(pitch), "heading_deg": math.degrees(heading), "second_face": second,
            "place_heading_deg": math.degrees(place_heading), "contact_aperture": contact, "hold_aperture": hold,
            "carried": layout.carried, "target": layout.target,
        })

    def walk(self, trajectory: Trajectory, fps: int, seconds: float | None = None,
             joints: list | None = None) -> dict:
        """The arm following the take frame by frame, as the rollout drives it,
        for `seconds` (the take's own by default; past its end the hand holds
        still). The joints of each frame are put in `joints` when given."""
        n = int(math.floor((seconds or trajectory.plan.seconds) * fps)) + 1
        T0, _ = trajectory.at(0.0)
        q = self.reach(T0)
        if q is None:
            raise DemoError("the arm cannot reach where the take starts")
        if joints is not None:
            joints.append(q)
        worst_m = worst_deg = worst_jump = 0.0
        for k in range(1, n):
            T, _ = trajectory.at(k / fps)
            nq, err, ang = self.track(T, q)
            jump = float(np.max(np.abs(nq - q)))
            worst_m, worst_deg, worst_jump = max(worst_m, err), max(worst_deg, ang), max(worst_jump, jump)
            if err > REACH_M or ang > REACH_DEG or jump > JUMP_DEG:
                raise DemoError(f"the arm falls behind at {k / fps:.2f} s ({err * 1000:.1f} mm, {ang:.1f} deg, "
                                f"jump {jump:.1f} deg)")
            q = nq
            if joints is not None:
                joints.append(q)
        return {"frames": n, "worst_mm": worst_m * 1000, "worst_deg": worst_deg, "worst_jump_deg": worst_jump}


# --- both hands ---------------------------------------------------------------------------------
def two_hand_layout(rng: np.random.Generator) -> Layout:
    """Each working hand's block and bowl on its own side of the desk: a
    hand's two far enough apart for its block to go down beside its bowl, and
    nothing of one hand near anything of the other. A hand at rest has
    nothing on the desk."""
    s = SETTINGS
    working = s.two_hand_working
    blocks = {b.name.removeprefix("block_"): b.name for b in s.bowl_blocks}
    bowls = {b.name.removeprefix("bowl_"): b.name for b in s.bowls}
    for _ in range(500):
        spots: dict[str, tuple[float, float, float]] = {}
        for hand in working:
            spots[blocks[hand]] = (rng.uniform(*s.two_hand_block_x), rng.uniform(*mirrored(s.two_hand_block_y, hand)),
                                   rng.uniform(-1, 1) * math.radians(s.block_yaw_deg))
            spots[bowls[hand]] = (rng.uniform(*s.two_hand_bowl_x), rng.uniform(*mirrored(s.two_hand_bowl_y, hand)),
                                  rng.uniform(-math.pi, math.pi))
        mine = all(math.dist(spots[blocks[h]][:2], spots[bowls[h]][:2]) >= s.two_hand_block_bowl_gap
                   for h in working)
        apart = len(working) < 2 or all(
            math.dist(spots[a][:2], spots[b][:2]) >= s.two_hand_apart
            for a in (blocks["left"], bowls["left"]) for b in (blocks["right"], bowls["right"]))
        if mine and apart:
            return Layout(spots, hands={h: (blocks[h], bowls[h]) for h in working})
    raise DemoError("no layout with each hand's block and bowl apart and the two hands' clear of each other")


def rest_plan(planner: Planner, layout: Layout, seconds: float) -> Plan:
    """A hand at rest for a take: held where its arm stands ready - a pose
    the arm can take, which a Hermit laid on the desk is not - the jaw open,
    for `seconds`. It carries nothing."""
    T = planner.chain.forward(np.array(planner.ready, dtype=float))
    T[:3, :3] = T[:3, :3] @ TOOL_TURN[planner.arm].T
    opened = planner.open
    return Plan(layout, [Key(0.0, T, opened), Key(seconds, T, opened)], math.inf, math.inf, np.eye(4),
                {"rest": True})


def two_hand_take(world: World, planners: dict[str, Planner], rng: np.random.Generator,
                  tremor: bool = True) -> tuple[Layout, dict[str, Plan], dict[str, Trajectory], dict[str, dict]]:
    """One take of both hands - the working ones planned, the rest held
    ready - that both arms could follow together: its layout, each hand's
    plan and trajectory, and how each arm followed. DemoError otherwise."""
    layout = two_hand_layout(rng)
    plans = {h: planners[h].plan(rng, layout=layout.of(h)) for h in layout.hands}
    seconds = max(p.seconds for p in plans.values())
    for h in planners:
        if h not in plans:
            plans[h] = rest_plan(planners[h], layout, seconds)
    trajectories = {h: Trajectory(plans[h], Tremor(rng, SETTINGS.tremor_m, SETTINGS.tremor_rad) if tremor else None)
                    for h in planners}
    joints: dict[str, list] = {h: [] for h in planners}
    walked = {h: planners[h].walk(trajectories[h], SETTINGS.fps, seconds, joints[h]) for h in planners}
    frames = min(len(v) for v in joints.values())
    apertures = {h: [trajectories[h].at(k / SETTINGS.fps)[1] for k in range(frames)] for h in planners}
    arms_apart(world, joints, apertures, {h: planners[h].chain.joint_names for h in planners})
    return layout, plans, trajectories, walked


def arm_geoms(world: World, arm: str) -> set[int]:
    """The arm's collision pieces: every body of it, from the base out."""
    m = world.model
    return {g for g in range(m.ngeom) if m.geom_contype[g] and m.body(int(m.geom_bodyid[g])).name.startswith(f"{arm}_")}


def arms_apart(world: World, joints: dict[str, list[np.ndarray]], apertures: dict[str, list[float]],
               names: dict[str, list[str]]) -> None:
    """Both arms, frame by frame as they would follow the hands, kept out of
    each other: a pair of takes the two arms could not make together is not
    a two-handed take."""
    mine = {arm: arm_geoms(world, arm) for arm in joints}
    left, right = (mine[a] for a in TWO_HANDS)
    d = world.data
    for k in range(min(len(v) for v in joints.values())):
        pose = {}
        for arm, qs in joints.items():
            pose.update({n: math.radians(v) for n, v in zip(names[arm], qs[k])})
            band = world.bands[arm]
            pose[f"{arm}_gripper"] = band[0] + apertures[arm][k] * (band[1] - band[0])
        world.set_joints(pose)
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist < 0 and ((c.geom1 in left and c.geom2 in right) or (c.geom1 in right and c.geom2 in left)):
                raise DemoError(f"the two arms meet at {k / SETTINGS.fps:.2f} s")


# --- writing -----------------------------------------------------------------------------------
def head_intrinsics(scene: str = "blocks") -> dict:
    from .world import head_view

    w, h = SETTINGS.head_size
    f = (h / 2) / math.tan(math.radians(head_view(scene)[2]) / 2)
    return {"stream": {"kind": "color", "width": w, "height": h, "fps": SETTINGS.fps, "format": "bgr8"},
            "camera": {"width": w, "height": h, "fx": f, "fy": f, "cx": w / 2, "cy": h / 2,
                       "coeffs": [0.0] * 5}}


def noisy(T: np.ndarray, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    out = T.copy()
    out[:3, :3] = T[:3, :3] @ Rotation.from_rotvec(rng.normal(0.0, SETTINGS.pose_noise_rad * scale, 3)).as_matrix()
    out[:3, 3] += rng.normal(0.0, SETTINGS.pose_noise_m * scale, 3)
    return out


def record(world: World, plans: dict[str, Plan], trajectories: dict[str, Trajectory], root: Path,
           rng: np.random.Generator, sim_info: dict, noise: float = 1.0) -> dict:
    """One take into `root`, in the recorder's format: each hand of `plans`
    making its own at once, the take as long as the longer of them, the other
    hand holding still at its end. Returns what happened to what each carried.

    `noise` scales the tracker's noise on the recorded poses; 0 records the truth."""
    s = SETTINGS
    fps = s.fps
    hands = list(plans)
    seconds = max(plan.seconds for plan in plans.values())
    root.mkdir(parents=True, exist_ok=False)
    world.data.qvel[:] = 0
    world.lay_out(plans[hands[0]].layout)
    world.set_joints({n: 0.0 for n in (f"{a}_{j}" for a in TWO_HANDS for j in CORE.joint_names)})
    tcp_T_solid = {h: np.linalg.inv(UMI.for_solid(f"handheld_{h}").solid_T_tcp) for h in hands}
    head_T_world = np.linalg.inv(world.head)
    desk_T_world = np.linalg.inv(world.desk)
    head_video = None
    gripper_videos: dict[str, Encoder] = {}
    # truth.jsonl is the simulation's own: the TCP as it was, in the robot's
    # frame, beside what the recorder would have seen of it.
    lines = {"frames": open(root / "frames.jsonl", "w"), "poses": open(root / "poses.jsonl", "w"),
             "events": open(root / "events.jsonl", "w"), "truth": open(root / "truth.jsonl", "w")}
    started_wall = time.time()
    t0 = started_wall
    lines["events"].write(json.dumps({"host_ts": t0, "monotonic": 0.0, "event": "start"}) + "\n")
    n = int(math.floor(seconds * fps)) + 1
    substeps = max(1, int(round(1.0 / (fps * world.model.opt.timestep))))
    dt = 1.0 / (fps * substeps)
    sizes = {}
    try:
        for k in range(n):
            t = k / fps
            now = {h: trajectories[h].at(t) for h in hands}
            for h, (T, aperture) in now.items():
                world.hold_hermit(h, T, aperture)
            world.forward()
            stamp = t0 + t
            head = cv2.cvtColor(world.head_image(), cv2.COLOR_RGB2BGR)
            if head_video is None:
                head_video = Encoder(root / "head.mp4", fps, head.shape[1], head.shape[0])
                sizes["head"] = [head.shape[1], head.shape[0]]
            head_video.write(head)
            lines["frames"].write(json.dumps({"stream": "head", "index": k, "host_ts": stamp + 0.03,
                                              "sensor_ts": stamp, "frame_number": k}) + "\n")
            for h in hands:
                gripper = cv2.cvtColor(world.gripper_image(hermit(h)), cv2.COLOR_RGB2BGR)
                if h not in gripper_videos:
                    gripper_videos[h] = Encoder(root / f"gripper_{h}.mp4", fps, gripper.shape[1], gripper.shape[0])
                    sizes[h] = [gripper.shape[1], gripper.shape[0]]
                gripper_videos[h].write(gripper)
                lines["frames"].write(json.dumps({"stream": f"gripper_{h}", "index": k,
                                                  "host_ts": stamp + UMI.gripper_camera_latency_s}) + "\n")
            lines["truth"].write(json.dumps({"stamp": stamp, **{
                (f"tcp_{h}" if len(hands) > 1 else "tcp"): T.tolist() for h, (T, _) in now.items()}, **{
                (f"aperture_{h}" if len(hands) > 1 else "aperture"): float(a) for h, (_, a) in now.items()}}) + "\n")
            solids, jaws = {}, {}
            for h in TWO_HANDS:
                if h in now:
                    seen = noisy(now[h][0] @ tcp_T_solid[h], rng, noise)
                    solids[f"handheld_{h}"] = {"solved": True, "faces": [], "reproj_px": 0.0,
                                               "pose": (head_T_world @ seen).tolist(),
                                               "world_pose": (desk_T_world @ seen).tolist()}
                    jaws[h] = float(now[h][1])
                else:
                    solids[f"handheld_{h}"] = {"solved": False, "faces": [], "reproj_px": None, "pose": None,
                                               "world_pose": None}
                    jaws[h] = None
            lines["poses"].write(json.dumps({
                "host_ts": stamp + UMI.jaw_latency_s, "stamp": stamp, "primary": f"handheld_{hands[0]}",
                "solids": solids, "desk": {"tracking": True, "faces": 3, "reproj_px": 0.0, "reason": ""},
                "jaw": jaws}) + "\n")
            # Physics until the next frame, the Hermits moving on between.
            for j in range(substeps):
                tj = t + (j + 1) * dt
                for h in hands:
                    Tj, aj = trajectories[h].at(tj)
                    world.hold_hermit(h, Tj, aj)
                    plan = plans[h]
                    if plan.grasp_t <= tj < plan.release_t:
                        world.place_block(plan.layout.carried, Tj @ plan.tcp_T_thing)
                mujoco.mj_step(world.model, world.data)
        # Let what was put down come to rest.
        world.step(0.5)
    finally:
        for f in lines.values():
            f.close()
        for video in (head_video, *gripper_videos.values()):
            if video is not None:
                video.close()
    ended = t0 + seconds
    with open(root / "events.jsonl", "a") as f:
        f.write(json.dumps({"host_ts": ended, "monotonic": seconds, "event": "stop"}) + "\n")
    each = {h: placed(world, plans[h].layout) for h in hands if not plans[h].info.get("rest")}
    outcome = (each[hands[0]] if len(hands) == 1 else
               {"placed": all(o["placed"] for o in each.values()),
                "off_mm": max(o["off_mm"] for o in each.values()), "hands": each,
                "resting": [h for h in hands if plans[h].info.get("rest")]})
    desk = UMI.for_solid(UMI.desk_solid)
    lens = s.lens_spec
    layout = plans[hands[0]].layout
    meta = {
        "name": root.name, "started_wall": started_wall, "started_monotonic": 0.0,
        "calibration": head_intrinsics(world.scene),
        "desk": {"solid": UMI.desk_solid, "yaw_deg": s.desk_yaw_deg,
                 "origin_above_table_mm": (desk.cut_mm if world.scene == "bowls" else s.desk_origin[2] * 1000),
                 "face_mm": desk.face_size_mm,
                 # Both hands' takes are made against a solid where the rig
                 # reckons one at the chest, and are judged and started there.
                 **({"placement": "chest"} if world.scene == "bowls" else {})},
        "hands": {f"handheld_{h}": UMI.for_solid(f"handheld_{h}").face_size_mm for h in TWO_HANDS},
        "solid_T_tcp": {f"handheld_{h}": UMI.for_solid(f"handheld_{h}").solid_T_tcp.tolist() for h in TWO_HANDS},
        "gripper_cameras": {},
        "streams": {
            "head": {"file": "head.mp4", "fps": fps, "codec": "libx264", "size": sizes.get("head"),
                     "frames": n, "dropped": 0},
            **{f"gripper_{h}": ({"file": f"gripper_{h}.mp4", "fps": fps, "codec": "libx264", "size": sizes.get(h),
                                 "frames": n, "dropped": 0, "centre": [lens.cx / lens.width, lens.cy / lens.height]}
                                if h in hands else
                                {"file": f"gripper_{h}.mp4", "fps": fps, "codec": "", "size": None, "frames": 0,
                                 "dropped": 0})
               for h in TWO_HANDS},
        },
        "ended_wall": ended, "seconds": round(seconds, 3),
        "sim": {**sim_info, "scene": world.scene, "layout": {k: list(v) for k, v in layout.things.items()},
                "plan": (plans[hands[0]].info if len(hands) == 1 else {h: plans[h].info for h in hands}),
                "outcome": outcome},
    }
    (root / "meta.json").write_text(json.dumps(meta, indent=2))
    return outcome


def in_bowl(world: World, thing: str, bowl: str) -> bool:
    """A thing is in a bowl when its centre is inside the bowl's rim, well in
    from its edge, and its bottom is below the rim - a tall one stands out of
    the bowl it is in."""
    T, B = world.block_pose(thing), world.block_pose(bowl)
    b = world.things[bowl]
    off = float(np.linalg.norm(T[:2, 3] - B[:2, 3]))
    bottom = T[2, 3] - world.things[thing].rest_z
    return bool(off < IN_BOWL * b.half_width and bottom < B[2, 3] - b.rest_z + b.height - IN_BOWL_DEPTH)


def placed(world: World, layout: Layout) -> dict:
    """Whether the carried thing ended where its layout asked: a hand's block
    in its bowl, the cap on the green piece."""
    thing = world.block_pose(layout.carried)
    target = world.block_pose(layout.target)
    off = float(np.linalg.norm(thing[:2, 3] - target[:2, 3])) * 1000
    if world.scene == "bowls":
        return {"placed": in_bowl(world, layout.carried, layout.target), "off_mm": round(off, 1)}
    height = world.things[layout.target].height
    rise = float(thing[2, 3] - target[2, 3] - height) * 1000
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, thing[2, 2]))))
    ok = off < PLACED_MM and abs(rise) < PLACED_HEIGHT_MM and tilt < PLACED_TILT_DEG
    return {"placed": bool(ok), "off_mm": round(off, 1), "rise_mm": round(rise, 1), "tilt_deg": round(tilt, 1)}


def generate_parallel(count: int, batch: str, seed: int, workers: int, say=print, **kwargs) -> dict:
    """The same takes from several processes, each with its own renderer and
    seed: worker k writes `<batch>-k<k>-NNN`. Rendering is what a take costs,
    and one process leaves most of the machine idle."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    shares = [count // workers + (1 if k < count % workers else 0) for k in range(workers)]
    started = time.monotonic()
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(generate, n, f"{batch}-k{k}", seed * 100 + k, _quiet, **kwargs)
                   for k, n in enumerate(shares) if n]
        reports = [f.result() for f in futures]
    rejected: dict[str, int] = {}
    for r in reports:
        for reason, n in r["rejected"].items():
            rejected[reason] = rejected.get(reason, 0) + n
    kept = [name for r in reports for name in r["kept"]]
    say(f"  {len(kept)} takes from {len(reports)} workers")
    return {"kept": kept, "tried": sum(r["tried"] for r in reports), "rejected": rejected,
            "seconds": round(time.monotonic() - started, 1), "episodes_dir": reports[0]["episodes_dir"]}


def _quiet(*_args) -> None:
    pass


def generate(count: int, batch: str, seed: int = 0, say=print, noise: float = 1.0,
             viewpoint: str = "robot", tries: int | None = None, scene: str = "blocks",
             per_layout: int = 1, two_faces: bool = False) -> dict:
    """`count` kept takes under the sim episodes directory, named `<batch>-NNN`.

    With `per_layout` above one - the blocks with `two_faces` - a layout and a
    start are drawn once and the cap taken by each of its two faces. A layout that keeps fewer than two is dropped whole, its takes
    teaching nothing apart. With both hands (`bowls`) each take is the two
    hands' at once on a desk of their own. Gives up after `tries` plans
    (default: twenty per take asked for)."""
    if per_layout < 1:
        raise DemoError(f"--per-layout is {per_layout}: at least one take a layout")
    if per_layout > 1 and (scene != "blocks" or not two_faces):
        raise DemoError("several takes a layout are the blocks' two faces")
    if scene == "bowls" and (two_faces or viewpoint != "robot"):
        raise DemoError("both hands' takes are made from the robot's side, each block by its nearest face")
    rng = np.random.default_rng(seed)
    hands = hands_of(scene)
    world = World(arms=hands, hermits=hands, scene=scene)
    planner = Planner(world, viewpoint, two_faces)
    planners = {h: Planner(world, viewpoint, two_faces, arm=h) for h in hands}
    out_dir = SETTINGS.episodes_dir
    tries = tries or 20 * count
    out_dir.mkdir(parents=True, exist_ok=True)
    kept, tried, why = [], 0, {}
    started = time.monotonic()
    layouts = made = 0

    def rejected(reason: str) -> None:
        why[reason] = why.get(reason, 0) + 1

    def planned(layout: Layout | None = None, start: tuple | None = None, second: bool | None = None):
        """A take the arm could follow, or None."""
        nonlocal tried
        tried += 1
        try:
            plan = planner.plan(rng, layout=layout, start=start, second=second)
            trajectory = Trajectory(plan, Tremor(rng, SETTINGS.tremor_m, SETTINGS.tremor_rad))
            return plan, trajectory, planner.walk(trajectory, SETTINGS.fps)
        except DemoError as exc:
            rejected(str(exc).split(" (")[0].split(" at ")[0])
            return None

    def take(plan: Plan, trajectory: Trajectory, walked: dict, group: str | None = None) -> bool:
        return written({planner.arm: plan}, {planner.arm: trajectory}, walked, group)

    def written(plans: dict[str, Plan], trajectories: dict[str, Trajectory], walked: dict,
                group: str | None = None) -> bool:
        nonlocal made
        # Numbered by takes written, not kept: a layout dropped whole leaves its numbers unused.
        name = f"{batch}-{made:03d}"
        root = out_dir / name
        if root.exists():
            raise DemoError(f"{root} exists - pick another batch name")
        info = {"seed": seed, "try": tried, "walk": walked, "noise": noise, "viewpoint": viewpoint,
                "two_faces": two_faces}
        if group is not None:
            info["layout_group"] = group
        outcome = record(world, plans, trajectories, root, rng, info, noise)
        made += 1
        if not outcome["placed"]:
            shutil.rmtree(root)
            missed = [h for h, o in outcome.get("hands", {"": outcome}).items() if not o["placed"]]
            carried = " and ".join(plans[h].layout.carried for h in missed if h in plans) or \
                next(iter(plans.values())).layout.carried
            rejected(f"the {carried} did not stay {'in' if world.scene != 'blocks' else 'on'} the target")
            return False
        kept.append(name)
        if len(plans) == 1:
            plan = next(iter(plans.values()))
            say(f"  {name}: {plan.seconds:.1f} s, pitch {plan.info['pitch_deg']:.0f} deg, "
                f"hold {plan.info['hold_aperture']:.2f}, worst {walked['worst_mm']:.1f} mm / "
                f"{walked['worst_deg']:.1f} deg, {plan.layout.carried} {outcome['off_mm']:.0f} mm off")
        else:
            say(f"  {name}: {max(p.seconds for p in plans.values()):.1f} s, "
                + ", ".join(f"{h} pitch {p.info['pitch_deg']:.0f} deg {outcome['hands'][h]['off_mm']:.0f} mm off"
                            for h, p in plans.items() if not p.info.get("rest"))
                + "".join(f", {h} at rest" for h in outcome.get("resting", [])))
        return True

    def both() -> None:
        """One take of both hands, written if the arms could have made it together."""
        nonlocal tried
        tried += 1
        try:
            _, plans, trajectories, walked = two_hand_take(world, planners, rng)
        except DemoError as exc:
            rejected(str(exc).split(" (")[0].split(" at ")[0])
            return
        written(plans, trajectories, {"hands": walked, "worst_mm": max(w["worst_mm"] for w in walked.values()),
                                      "worst_deg": max(w["worst_deg"] for w in walked.values())})

    try:
        while len(kept) < count and tried < tries:
            if len(hands) > 1:
                both()
                continue
            if per_layout == 1:
                if (one := planned()) is not None:
                    take(*one)
                continue
            group = f"{batch}-g{layouts:03d}"
            layouts += 1
            # The desk and where the hand starts, once; then the takes it allows,
            # planned and walked first, all of them: rendering is what a take
            # costs, and a layout with one followable take is dropped anyway.
            try:
                base = planner.layout(rng)
                start = planner.blocks_start(rng, base)
            except DemoError as exc:
                tried += 1
                rejected(str(exc).split(" (")[0].split(" at ")[0])
                continue
            # The blocks by both faces, from the same start: one first
            # picture, two right ways on from it.
            plans = [p for p in (planned(base, start, second) for second in (False, True)) if p is not None]
            if len(plans) < 2:
                rejected("a layout kept fewer than two of its takes")
                continue
            mine = []
            # The last layout may run past `count`: cut short, it would teach nothing.
            for one in plans:
                if take(*one, group):
                    mine.append(kept[-1])
            if len(mine) < 2:
                for name in mine:
                    shutil.rmtree(out_dir / name)
                    kept.remove(name)
                rejected("a layout kept fewer than two of its takes")
    finally:
        world.close()
    return {"kept": kept, "tried": tried, "rejected": why, "seconds": round(time.monotonic() - started, 1),
            "episodes_dir": str(out_dir)}
