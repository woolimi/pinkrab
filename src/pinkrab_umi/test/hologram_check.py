"""Offline check of the robot's arms drawn into the top camera's picture: an
arm stands at the ready pose until its hand is seen; a hand held at the ready
pose leaves its arm there, drawn green, short of nothing; a hand held out of
reach leaves its arm short and drawn red; the shared follow rule lands where
the replay's did. No camera, no robot.

    python src/pinkrab_umi/test/hologram_check.py
"""
from __future__ import annotations

import sys

import numpy as np

from pinkrab_core.config import ARMS
from pinkrab_core.follow import SHORT_MM, Following, tool_absolute_target
from pinkrab_core.kinematics import ArmKinematics
from pinkrab_core.origin import ready_joints
from pinkrab_umi import hologram, ready_guide
from pinkrab_umi.config import SETTINGS
from pinkrab_umi.desk_solid import world_T_desk
from pinkrab_umi.pnp import Intrinsics

FAILURES: list[str] = []
CAMERA = Intrinsics(width=1920, height=1200, fx=700.0, fy=700.0, cx=960.0, cy=600.0)


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def worn(desk: np.ndarray) -> np.ndarray:
    """cam_T_desk of a camera worn on a head over the chest: 0.45 m above the
    solid's origin and 0.25 m behind it, looking down the desk at 40 degrees."""
    tilt = np.radians(40.0)
    desk_T_cam = np.eye(4)
    # z ahead and down along the desk, x to the right, y down the picture.
    forward = np.array([0.0, np.cos(tilt), -np.sin(tilt)])
    right = np.array([1.0, 0.0, 0.0])
    down = np.cross(forward, right)
    desk_T_cam[:3, :3] = np.column_stack([right, down, forward])
    desk_T_cam[:3, 3] = [0.0, -0.25, 0.45]
    return np.linalg.inv(desk_T_cam)


def count(canvas: np.ndarray, colour) -> int:
    return int(np.all(canvas == np.array(colour, np.uint8), axis=2).sum())


def main() -> int:
    desk = world_T_desk()
    guide = ready_guide.load()

    print("the follow rule the replay, the drawing and the picture share")
    for arm in ARMS:
        chain = ArmKinematics(arm)
        ready = np.array(ready_joints(arm, chain.joint_names), dtype=float)
        follow = Following(chain, ready)
        at = tool_absolute_target(arm, desk, guide.tools[arm])
        follow.step(at, 0.0, 30.0)
        check(f"{arm}: the ready tool is reached from ready", not follow.short and follow.reach[0] < 1.0,
              f"{follow.reach[0]:.2f} mm, {follow.reach[1]:.2f} deg")
        far = at.copy()
        far[:3, 3] += [0.0, 0.0, 0.8]
        follow.step(far, 1.0, 30.0)
        check(f"{arm}: a tool 0.8 m above it is short", follow.short and follow.reach[0] > SHORT_MM,
              f"{follow.reach[0]:.0f} mm")

    print("the arms in the picture")
    cam_T_desk = worn(desk)
    solid = {arm: np.linalg.inv(SETTINGS.for_solid(f"handheld_{arm}").solid_T_tcp) for arm in ARMS}
    arms = hologram.Hologram()
    canvas = np.zeros((CAMERA.height, CAMERA.width, 3), np.uint8)
    arms.draw(canvas, CAMERA, cam_T_desk, {})
    grey = count(canvas, hologram.ARM_COLOUR)
    check("unseen hands: both arms drawn at ready, grey", grey > 1000 and not any(f.started for f in arms.follow.values()),
          f"{grey} outline px")

    at_ready = {arm: cam_T_desk @ guide.tools[arm] @ solid[arm] for arm in ARMS}
    arms.follow_hands(cam_T_desk, at_ready, 0.0)
    canvas = np.zeros_like(canvas)
    arms.draw(canvas, CAMERA, cam_T_desk, {arm: True for arm in ARMS})
    check("hands at the ready pose: arms followed, not short, drawn green",
          all(f.started and not f.short for f in arms.follow.values())
          and count(canvas, ready_guide.MATCHED_COLOUR) > 1000 and count(canvas, hologram.SHORT_COLOUR) == 0,
          f"{count(canvas, ready_guide.MATCHED_COLOUR)} green px")

    away = {arm: T.copy() for arm, T in at_ready.items()}
    away["left"] = cam_T_desk @ np.array([[1, 0, 0, 0.0], [0, 1, 0, 0.0], [0, 0, 1, 0.9], [0, 0, 0, 1.0]]) \
        @ guide.tools["left"] @ solid["left"]
    for k in range(1, 40):   # long enough for a fresh solve from ready to be tried
        arms.follow_hands(cam_T_desk, away, k / 30.0)
    canvas = np.zeros_like(canvas)
    arms.draw(canvas, CAMERA, cam_T_desk, {"right": True})
    check("a hand out of reach: its arm short and drawn red, the other still green",
          arms.follow["left"].short and not arms.follow["right"].short
          and count(canvas, hologram.SHORT_COLOUR) > 500 and count(canvas, ready_guide.MATCHED_COLOUR) > 500,
          f"left {arms.follow['left'].reach[0]:.0f} mm short")

    behind = np.linalg.inv(np.linalg.inv(cam_T_desk) @ np.diag([1.0, -1.0, -1.0, 1.0]))
    canvas = np.zeros_like(canvas)
    arms.draw(canvas, CAMERA, behind, {})
    check("a camera turned away from the arms draws nothing", not canvas.any())

    print(f"\n{'all checks passed' if not FAILURES else f'{len(FAILURES)} failed: ' + ', '.join(FAILURES)}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
