"""Offline check of following the hand: each arm at the ready pose, its hand
somewhere in the top camera's picture, the loop run at its rate with the arm
landing where it is sent. A still hand moves nothing wherever it is, a hand
carried 50 mm along its own approach carries the arm's tool 50 mm along its
own - either arm, the left's tool frame being the right's turned half a turn
- and a hand back after a gap longer than the regrab time takes the origin
again where it is, the arm still. An arm not yet read in full is not taken
up. A correction's jaws are caught each at its own hand's opening. No servo,
no ROS graph.

    python src/pinkrab_teleop/test/follow_check.py
"""
from __future__ import annotations

import math
import sys

import numpy as np

from pinkrab_core.config import SETTINGS as CORE  # noqa: E402
from pinkrab_core.follow import HandFollower  # noqa: E402
from pinkrab_core.kinematics import ArmKinematics  # noqa: E402
from pinkrab_core.origin import ready_joints  # noqa: E402
from pinkrab_umi.config import SETTINGS as UMI  # noqa: E402

FAILURES: list[str] = []
DT = 1.0 / CORE.ik_rate_hz


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def run(follower: HandFollower, solid: np.ndarray, seconds: float, t0: float, joints: dict) -> tuple[list, float]:
    """The loop for `seconds`, the arm landing where it was sent each tick."""
    tools, t = [], t0
    while t < t0 + seconds - 1e-9:
        out, _ = follower.follow(solid, joints, now=t)
        joints = out
        tools.append(follower.chain.forward([math.degrees(out[n]) for n in follower.chain.joint_names]))
        t += DT
    return tools, t


def waits_for_readings() -> None:
    """The handheld mode takes an arm up only once all of it has been read:
    an origin taken from a pose the arm is not in would send it there."""
    from types import SimpleNamespace

    from pinkrab_teleop.teleop import TeleopNode

    node = TeleopNode.__new__(TeleopNode)   # the handlers alone: no ROS graph
    quiet = SimpleNamespace(info=lambda *a, **k: None, warn=lambda *a, **k: None)
    node.get_logger = lambda: quiet
    node.chains, node._chains_lock = {}, __import__("threading").Lock()
    node.driving, node.following, node.correcting = True, True, False
    node.followers, node.bands, node._jaw_openings, node._quiet = {}, {}, {}, 0
    node.orientation_weight = 0.1
    sent = []
    node._publish = sent.append
    node._note_timing = lambda *a: None
    node._say_quietly = lambda *a: None
    chain = ArmKinematics("right")
    solid = np.eye(4)
    solid[2, 3] = 0.8
    node._stats = lambda: {"solids": {"handheld_right": {"solved": True, "pose": solid.tolist()}}}
    node.at = {chain.joint_names[0]: 0.0}
    node._tick()
    check("an arm read only in part is not taken up", "right" not in node.followers and not sent,
          f"followers {list(node.followers)}")
    node.at = {n: math.radians(v) for n, v in zip(chain.joint_names, ready_joints("right", chain.joint_names))}
    node._tick()
    check("and is once every joint has been read", "right" in node.followers and bool(sent))


def jaws_caught_each_on_their_own() -> None:
    """A correction's jaws, through the node's own handlers: each arm's jaw
    holds what was last sent until its own hand's opening reaches it, and a
    follow toggle lets go of both catches."""
    from types import SimpleNamespace

    from pinkrab_teleop.teleop import TeleopNode

    node = TeleopNode.__new__(TeleopNode)   # the handlers alone: no ROS graph
    quiet = SimpleNamespace(info=lambda *a, **k: None)
    node.get_logger = lambda: quiet
    node._say_following = lambda: None
    band = (0.0, 1.0)
    node.sent = {"left_gripper": 0.3, "right_gripper": 0.7}
    node.at = {}
    node.following, node.followers = False, {}
    node.jaw_caught, node.jaw_side = {}, {}
    node._on_follow(None, SimpleNamespace())
    left = [node._catch_jaw("left", opening, band) for opening in (0.6, 0.45, 0.31)]
    right = node._catch_jaw("right", 0.2, band)
    check("after a follow, each jaw is caught at its own hand's opening, not the other's",
          left == [False, False, True] and not right, f"left {left}, right {right}")
    node._on_follow(None, SimpleNamespace())
    check("and a toggle lets go of both catches", not node._catch_jaw("left", 0.9, band))


def follows(arm: str) -> None:
    chain = ArmKinematics(arm)
    ready = ready_joints(arm, chain.joint_names)
    start = {n: math.radians(v) for n, v in zip(chain.joint_names, ready)}
    tool = chain.forward(ready)
    tcp = UMI.for_solid(f"handheld_{arm}").solid_T_tcp
    # The solid somewhere in the top camera's picture, turned some way: where
    # it is has nothing to do with where the arm stands.
    solid = np.eye(4)
    solid[:3, :3] = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    solid[:3, 3] = [0.12, -0.05, 0.85]
    follower = HandFollower(arm, chain, start, tcp)
    tools, t = run(follower, solid, 1.0, 0.0, start)
    drift = max(np.linalg.norm(T[:3, 3] - tool[:3, 3]) for T in tools) * 1000
    check(f"{arm}: a still hand moves nothing, wherever it is", drift < 2.0, f"{drift:.2f} mm at most")
    along = np.eye(4)
    along[2, 3] = 0.05                     # 50 mm along the hand tool's own approach
    moved = solid @ tcp @ along @ np.linalg.inv(tcp)
    joints = {n: math.radians(v) for n, v in zip(chain.joint_names, follower.joints)}
    tools, _ = run(follower, moved, 2.0, t, joints)
    went = tools[-1][:3, 3] - tool[:3, 3]
    ahead = float(went @ tool[:3, 2]) * 1000
    check(f"{arm}: the tool moves as the hand moved, in its own frame", abs(ahead - 50.0) < 3.0,
          f"{ahead:.1f} mm along the approach of 50")
    # Out of sight past the regrab time, and moved another 50 mm meanwhile:
    # the origin is taken again there, and the arm does not follow the move.
    at = chain.forward(follower.joints)
    joints = {n: math.radians(v) for n, v in zip(chain.joint_names, follower.joints)}
    later = moved @ tcp @ along @ np.linalg.inv(tcp)
    out, began = follower.follow(later, joints, now=t + 2.0 + CORE.teleop_regrab_s + 0.5)
    back = chain.forward([math.degrees(out[n]) for n in chain.joint_names])
    jump = np.linalg.norm(back[:3, 3] - at[:3, 3]) * 1000
    check(f"{arm}: a hand back after a gap takes the origin again, the arm still",
          began and jump < 2.0, f"moved {jump:.2f} mm")


if __name__ == "__main__":
    for arm in ("right", "left"):
        follows(arm)
    waits_for_readings()
    jaws_caught_each_on_their_own()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s): {FAILURES}")
        sys.exit(1)
    print("all ok")
