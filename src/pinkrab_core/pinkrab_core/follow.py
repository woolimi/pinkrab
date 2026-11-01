"""Where the tool should be, given where the hand is.

Teleop and a correction (DAgger) - `tool_step_target`, `HandFollower`: the
arm's tool moves as the hand's tool has moved since the origin, said in the
hand tool's OWN frame - forward along where it points, turned about its own
axes - which is how a policy's step is said too. The motion between two
poses of the hand in one frame does not depend on that frame, as long as the
frame holds still: the top camera, worn on the operator's head, does not, so
the hand is read against the desk solid, which may lie anywhere on the desk.
The ROS teleop node follows with it on the metal, the studio on the
simulation.

A take placed in the robot's world (`tool_absolute_target`): the hand's tool
pose against the desk solid, carried in by where the solid is reckoned to
stand there - what the shelf's replay judges a take by and where a policy
run starts its arms (pinkrab_core.demo_start).
"""
from __future__ import annotations

import math
import time

import numpy as np

from .config import SETTINGS as CORE
from .kinematics import ArmKinematics
from .origin import ready_joints

# How much the solver is asked to care about the tool's ATTITUDE beside its
# position. This wrist cannot hold both exactly across much of a hand's motion,
# so the weight is where the trade is made. Measured from the ready pose on an
# 80 mm wander with a 15 degree wobble: at 1.0 the attitude is held to a degree
# and the position falls 45 mm behind; at 0.1 the attitude gives up to 5 degrees
# and the position is within 14 mm. A tool being carried is about where it is
# first and how it is held second, and five degrees is under what the hand
# itself wobbles by.
ORIENTATION_WEIGHT = 0.1

# The left arm's tool frame is the right's turned half a turn about the
# approach axis: the description mirrors the arm, and a mirrored gripper has
# its jaw axis the other way round (measured with forward kinematics at the
# ready pose: the right tool's y leans to world +z, 0.42, the left's to -z,
# -0.42). The
# handheld's tool frame is the right's convention, so an absolute attitude
# asked of the left arm is the hand's attitude turned by this first.
TOOL_TURN = {
    "left": np.diag([-1.0, -1.0, 1.0]),
    "right": np.eye(3),
}


def tool_absolute_target(arm: str, world_T_desk: np.ndarray, desk_T_tcp: np.ndarray) -> np.ndarray:
    """The tool pose to ask the arm for, base frame, 4x4 metres: the hand's
    tool centre against the desk solid (`desk_T_tcp`) where the solid stands
    in the robot's world. The hand's tool is in the handheld's convention,
    which this arm's tool frame is TOOL_TURN of."""
    target = world_T_desk @ desk_T_tcp
    target[:3, :3] = target[:3, :3] @ TOOL_TURN[arm]
    return target


def desk_tool(arm: str, world_T_desk: np.ndarray, tool: np.ndarray) -> np.ndarray:
    """The hand's tool centre against the desk solid, 4x4 metres, that asks
    this arm for `tool` (base frame): tool_absolute_target undone."""
    out = np.linalg.inv(world_T_desk) @ tool
    out[:3, :3] = out[:3, :3] @ TOOL_TURN[arm].T
    return out


def tool_step_target(arm: str, start_tcp: np.ndarray, origin_tcp: np.ndarray,
                     now_tcp: np.ndarray) -> np.ndarray:
    """The tool pose to ask the arm for, base frame, 4x4 metres: where the arm's
    tool was when the origin was taken (`start_tcp`), moved as the hand's tool
    has moved from `origin_tcp` to `now_tcp` - any one frame for both, held
    still between them - in its own frame. The hand's tool is in the
    handheld's convention, which this arm's tool frame is TOOL_TURN of, so the
    step is turned into it, as rollout turns a policy's."""
    turn = np.eye(4)
    turn[:3, :3] = TOOL_TURN[arm]
    step = np.linalg.inv(origin_tcp) @ now_tcp
    return start_tcp @ turn.T @ step @ turn


def blend_pose(start: np.ndarray, end: np.ndarray, weight: float) -> np.ndarray:
    """`weight` of the way from `start` to `end`: the position straight, the
    attitude along the shortest turn between the two."""
    from scipy.spatial.transform import Rotation

    out = np.eye(4)
    out[:3, 3] = start[:3, 3] + weight * (end[:3, 3] - start[:3, 3])
    step = Rotation.from_matrix(start[:3, :3].T @ end[:3, :3]).as_rotvec()
    out[:3, :3] = start[:3, :3] @ Rotation.from_rotvec(weight * step).as_matrix()
    return out


def smooth_pose(previous: np.ndarray | None, current: np.ndarray) -> np.ndarray:
    """The pose to follow: `current` pulled part of the way from `previous`.

    A first-order low-pass with the time constant the settings give, run at the
    solver's rate. The translation is blended straight; the rotation is blended
    along the geodesic between the two, so a pose that turned is followed by a
    pose that turned less, not by one that stretched.
    """
    tau = CORE.teleop_pose_smoothing_s
    if previous is None or tau <= 0.0:
        return current
    dt = 1.0 / CORE.ik_rate_hz
    return blend_pose(previous, current, dt / (tau + dt))


class HandFollower:
    """One arm following one hand, relatively: the solid is against the
    desk solid, the first one seen is the origin, and the arm's tool moves
    from where it stood as the hand's has moved since - the origin taken again
    where the hand comes back after being out of sight longer than
    `teleop_regrab_s`, so a hand moved while unseen does not throw the arm
    after it."""

    def __init__(self, arm: str, chain: ArmKinematics, start: dict[str, float] | None,
                 solid_T_tcp: np.ndarray, orientation_weight: float = ORIENTATION_WEIGHT) -> None:
        self.arm = arm
        self.chain = chain
        self.solid_T_tcp = solid_T_tcp
        self.orientation_weight = orientation_weight
        if start:
            self.joints = np.array(
                [math.degrees(start.get(name, 0.0)) for name in self.chain.joint_names],
                dtype=float)
        else:
            # Nothing has said where the arm is. The ready pose is where a
            # session is meant to begin, and it is a pose the arm can leave in
            # every direction - the description's zero is not.
            self.joints = np.array(ready_joints(arm, self.chain.joint_names), dtype=float)
        # Where the tool stood and where the solid was when the origin was
        # taken; None until the hand is first seen, and again after it was lost.
        self.start_tcp: np.ndarray | None = None
        self.seen_at: float | None = None
        self.origin: np.ndarray | None = None
        # The solid's pose after the low-pass, carried from tick to tick.
        self.smoothed: np.ndarray | None = None

    def follow(self, solid: np.ndarray, measured: dict[str, float] | None = None,
               now: float | None = None) -> tuple[dict[str, float], bool]:
        """Where the arm should be, given where the solid is now against the
        desk solid: radians by joint, and whether the origin was taken
        with this call.

        Solved from where the arm IS, when `measured` says: the solver steps a
        bounded distance from its seed, and seeded from its own last answer it
        runs ahead of an arm that cannot keep up - measured seventy degrees
        ahead in a fast sweep - and the arm then spends the next second catching
        up with a hand that has already stopped. Seeded from the reading, the
        command is never more than a step ahead of the metal: the arm goes as
        fast as it can toward where the hand is NOW, and an excess the servos
        could not deliver is dropped rather than queued. The target is the
        origin's tool moved as the hand has moved, not the last target moved
        on, so nothing is lost by dropping it - the next frame asks for the
        right place again.
        """
        now = time.monotonic() if now is None else now
        if measured:
            self.joints = np.array(
                [math.degrees(measured[name]) if name in measured else value
                 for name, value in zip(self.chain.joint_names, self.joints)], dtype=float)
        began = self.seen_at is None or now - self.seen_at > CORE.teleop_regrab_s
        if began:
            self.start_tcp = self.chain.forward(self.joints)
            self.origin = solid
            self.smoothed = None
        self.seen_at = now
        solid = self.smoothed = smooth_pose(self.smoothed, solid)
        # The solid's transform to the tool centre is what MOVES: a hand rolling
        # the solid about its peg has to come out as the arm rolling about the
        # tool, not about the marker 120 mm away. So the target is the tool
        # centre's - and the step is taken about it.
        target = tool_step_target(self.arm, self.start_tcp, self.origin @ self.solid_T_tcp, solid @ self.solid_T_tcp)
        self.joints = self.chain.track(target, self.joints, orientation_weight=self.orientation_weight)
        return {name: float(np.deg2rad(value))
                for name, value in zip(self.chain.joint_names, self.joints)}, began
