#!/usr/bin/env python3
"""Pick where the ready pose waits: the tool held as a resting hand holds it,
at the place its camera sees most of the desk.

The attitude and height come from an arm's joints - the pose an operator
finds comfortable, followed into the robot and read off it - tipped further
down about the jaw's axis by --steeper, if asked. Every place on a
grid at that height is then solved for: the arm reaching it with the tool at
that attitude, every joint clear of its stops; room to carry the tool 60 mm
each way along the desk and 40 mm up and down from it; and the share of the
desk the arm can work on (its desk reach) inside the picture a policy is
shown - the follower camera's window, through its measured lens. The fingers
in the bottom of that picture are not taken off it. Of the ways the arm can
hold the tool there, the one bending its wrist least (wrist_flex and
wrist_yaw together) is kept, and --max-bend drops a place whose wrist bends
further than that. Prints the places that pass, the best seen first, and the
comfortable place itself.

origin.py's ready pose is the operator's own hold, not a place this picks:
what this weighs is a ready that shows the camera more than a hand at rest
does. For a comfortable pose given as this mount's left arm holds it, tipped
30 degrees steeper, the wrist bent no more than 30:

    python scripts/pick_ready.py --arm left --joints -24.4 -126.8 78.6 45.1 -68.8 -34.5 --steeper 30 --max-bend 30
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from pinkrab_core.config import SETTINGS as CORE
from pinkrab_core.dataset import gripper_window
from pinkrab_core.kinematics import ArmKinematics
from pinkrab_core.origin import ready_joints
from pinkrab_sim.config import measured_lenses
from pinkrab_core.workspace import DeskReach
from pinkrab_mujoco.scene import Scene, load_spec
from pinkrab_sim.lens import project
from pinkrab_sim.world import gripper_camera_in_link, pose_of

LAND_MM, LAND_DEG = 2.0, 2.0
STOP_DEG = 10.0          # the least a ready pose keeps from any stop
SEEDS = 12               # random arm shapes tried at each place, beside the given ones
ROOM_STOP_DEG = 2.0      # ...and the least a pose carried from it keeps
ROOM_ALONG_M, ROOM_UP_M = 0.06, 0.04


def tcp_T_cam(arm: str) -> np.ndarray:
    """The follower camera in the arm's tool frame, OpenCV axes, as the
    simulation mounts it (pinkrab_sim.world.gripper_camera_in_link)."""
    spec = load_spec(CORE.scene_file)
    tcp = spec.site(f"{arm}_tcp")
    return np.linalg.inv(pose_of(tcp.pos, tcp.quat)) @ gripper_camera_in_link(spec, arm, None, None)


def covered(tool: np.ndarray, cam: np.ndarray, lens, window, cells: np.ndarray) -> float:
    """The share of `cells` (n x 3, world) inside the policy's window of the
    camera on a tool at `tool`."""
    world_T_cam = tool @ cam
    p = (np.linalg.inv(world_T_cam) @ np.column_stack([cells, np.ones(len(cells))]).T).T[:, :3]
    theta = np.arctan2(np.linalg.norm(p[:, :2], axis=1), p[:, 2])
    inside_lens = theta < lens.fov / 2.0
    uv = project(lens, p)
    x, y, w, h = window
    inside = inside_lens & (uv[:, 0] >= x) & (uv[:, 0] < x + w) & (uv[:, 1] >= y) & (uv[:, 1] < y + h)
    return float(inside.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--joints", type=float, nargs=6, required=True, metavar="DEG",
                        help="the comfortable pose: pan, lift, elbow, wrist_flex, wrist_yaw, wrist_roll")
    parser.add_argument("--step", type=float, default=0.01, help="grid step, m")
    parser.add_argument("--steeper", type=float, default=0.0, metavar="DEG",
                        help="look this much further down than the given pose, turned about the jaw's axis")
    parser.add_argument("--max-bend", type=float, default=None, metavar="DEG",
                        help="drop places whose wrist bends further than this (wrist_flex and wrist_yaw together)")
    parser.add_argument("--x", type=float, nargs=2, default=(0.15, 0.50), metavar=("FROM", "TO"),
                        help="out from the board, m")
    parser.add_argument("--y", type=float, nargs=2, default=(0.0, 0.40), metavar=("FROM", "TO"),
                        help="out to the arm's own side, m")
    args = parser.parse_args()

    chain = ArmKinematics(args.arm)
    lo, hi = chain.limits
    comfort = np.array(args.joints)
    held = chain.forward(comfort)
    if args.steeper:
        # About the tool's own x, the jaw's axis - level, or near it, in a
        # grasping hand - the way that tips the approach further down.
        for sign in (1.0, -1.0):
            a = math.radians(sign * args.steeper)
            turned = held[:3, :3] @ np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
            if turned[2, 2] < held[2, 2]:
                held[:3, :3] = turned
                break
    lens = measured_lenses()[f"ar0234_follower_{args.arm}"]
    window = gripper_window(lens.width, lens.height, (lens.cx / lens.width, lens.cy / lens.height))
    cam = tcp_T_cam(args.arm)
    # The desk reach the studio draws, for the desk and board the scene has -
    # read back if it stands for this description and ready pose, solved
    # again (a minute or two) if it does not.
    scenery = Scene(CORE.scene_file).scenery()
    desk_z = scenery["desk"]["pos"][2] if "desk" in scenery else 0.0
    board = scenery.get("pegboard")
    board_x = board["pos"][0] + board["half"][0] if board else 0.0
    reach = DeskReach().get(desk_z, board_x)[args.arm]["cells"]
    cells = np.array([[x, y, desk_z] for x, y, _ in reach])

    names = [n.removeprefix(f"{args.arm}_") for n in chain.joint_names]

    def margin(j):
        return float(np.min(np.minimum(j - lo, hi - j)))

    def bend(j):
        return abs(j[names.index("wrist_flex")]) + abs(j[names.index("wrist_yaw")])

    def land(target, seeds, stop):
        best = None
        for seed in seeds:
            j = chain.solve(target, seed=seed, restarts=1, orientation_weight=1.0)
            mm, deg = chain.reach_error(target, j)
            if mm * 1000 <= LAND_MM and deg <= LAND_DEG and margin(j) >= stop:
                if best is None or bend(j) < bend(best):
                    best = j
        return best

    def room(target, j):
        for d in ((ROOM_ALONG_M, 0, 0), (-ROOM_ALONG_M, 0, 0), (0, ROOM_ALONG_M, 0), (0, -ROOM_ALONG_M, 0),
                  (0, 0, ROOM_UP_M), (0, 0, -ROOM_UP_M)):
            moved = target.copy()
            moved[:3, 3] += d
            if land(moved, [j], ROOM_STOP_DEG) is None:
                return False
        return True

    ready = np.array(ready_joints(args.arm, chain.joint_names), dtype=float)
    rng = np.random.default_rng(0)
    shapes = [comfort, ready] + [rng.uniform(lo, hi) for _ in range(SEEDS)]
    side = 1.0 if args.arm == "left" else -1.0
    found = []
    for x in np.arange(args.x[0], args.x[1] + 1e-9, args.step):
        for y in np.arange(args.y[0], args.y[1] + 1e-9, args.step) * side:
            target = held.copy()
            target[:3, 3] = [x, y, held[2, 3]]
            j = land(target, shapes, STOP_DEG)
            if j is None or (args.max_bend is not None and bend(j) > args.max_bend) or not room(target, j):
                continue
            found.append((covered(target, cam, lens, window, cells), x, y, j))
    found.sort(key=lambda f: -f[0])
    print(f"held at {np.round(held[:3, 3] * 1000, 1)} mm, {math.degrees(math.asin(-held[2, 2])):.1f} deg down; "
          f"that place covers "
          f"{covered(held, cam, lens, window, cells):.1%} of {len(cells)} desk cells")
    print(f"{len(found)} places pass; the best:")
    for share, x, y, j in found[:12]:
        print(f"  {share:6.1%}  x {x * 1000:5.0f}  y {y * 1000:5.0f} mm  stops >= {margin(j):4.1f}  "
              f"wrist bend {bend(j):5.1f} deg  "
              + "  ".join(f"{n} {v:6.1f}" for n, v in zip(names, j)))


if __name__ == "__main__":
    main()
