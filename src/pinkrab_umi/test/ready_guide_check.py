"""Offline check of the ready guide drawn into the top camera's picture: the
hand's tool it shows, carried out to the arm as the replay carries a hand,
asks the arm for its ready pose; a hand held there is drawn green and one
held 150 mm off is not; a picture whose camera has the guide behind it is
left alone. No camera, no robot.

    python src/pinkrab_umi/test/ready_guide_check.py
"""
from __future__ import annotations

import sys

import numpy as np

from pinkrab_core.config import ARMS
from pinkrab_core.follow import desk_tool, tool_absolute_target
from pinkrab_core.kinematics import ArmKinematics
from pinkrab_core.origin import ready_joints
from pinkrab_umi import ready_guide
from pinkrab_umi.config import SETTINGS
from pinkrab_umi.desk_solid import world_T_desk
from pinkrab_umi.pnp import Intrinsics

FAILURES: list[str] = []
CAMERA = Intrinsics(width=1920, height=1200, fx=900.0, fy=900.0, cx=960.0, cy=600.0)


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def looking_down(height: float) -> np.ndarray:
    """cam_T_desk of a camera `height` above the solid's origin, looking straight down."""
    desk_T_cam = np.eye(4)
    desk_T_cam[:3, :3] = np.diag([1.0, -1.0, -1.0])
    desk_T_cam[:3, 3] = [0.0, 0.15, height]
    return np.linalg.inv(desk_T_cam)


def drawn(cam_T_desk: np.ndarray, hands: dict, guide) -> np.ndarray:
    canvas = np.zeros((CAMERA.height, CAMERA.width, 3), np.uint8)
    ready_guide.draw(canvas, CAMERA, cam_T_desk, hands, guide)
    return canvas


def count(canvas: np.ndarray, colour) -> int:
    return int(np.all(canvas == np.array(colour, np.uint8), axis=2).sum())


if __name__ == "__main__":
    desk = world_T_desk()
    guide = ready_guide.load()
    tools = guide.tools
    for arm in ARMS:
        chain = ArmKinematics(arm)
        ready = np.array(ready_joints(arm, chain.joint_names), dtype=float)
        tool = chain.forward(ready)
        back = tool_absolute_target(arm, desk, desk_tool(arm, desk, tool))
        check(f"{arm}: desk_tool undoes tool_absolute_target", np.allclose(back, tool, atol=1e-9))
        mm, deg = chain.reach_error(tool_absolute_target(arm, desk, tools[arm]), ready)
        check(f"{arm}: the guide's tool asks the arm for its ready pose", mm < 1e-6 and deg < 1e-3,
              f"{mm * 1000:.4f} mm, {deg:.4f} deg")
    for arm in ARMS:
        span = np.ptp(guide.hermits[arm].verts, axis=0)
        check(f"{arm}: the Hermit's surface is a handheld's size", 0.05 < span.max() < 0.25,
              f"{np.round(span * 1000)} mm")

    cam_T_desk = looking_down(0.7)
    at_ready = {arm: cam_T_desk @ tools[arm] @ np.linalg.inv(SETTINGS.for_solid(f"handheld_{arm}").solid_T_tcp)
                for arm in ARMS}
    canvas = drawn(cam_T_desk, at_ready, guide)
    check("hands held at the ready pose are drawn green",
          count(canvas, ready_guide.MATCHED_COLOUR) > 500 and count(canvas, ready_guide.GUIDE_COLOUR) == 0,
          f"{count(canvas, ready_guide.MATCHED_COLOUR)} green px")
    off = {arm: T.copy() for arm, T in at_ready.items()}
    for T in off.values():
        T[0, 3] += 0.15
    canvas = drawn(cam_T_desk, off, guide)
    check("hands held 150 mm off are not", count(canvas, ready_guide.MATCHED_COLOUR) == 0
          and count(canvas, ready_guide.GUIDE_COLOUR) > 500, f"{count(canvas, ready_guide.GUIDE_COLOUR)} px")
    canvas = drawn(cam_T_desk, {arm: None for arm in ARMS}, guide)
    check("unseen hands still get the guide", count(canvas, ready_guide.GUIDE_COLOUR) > 500)
    canvas = drawn(np.linalg.inv(np.linalg.inv(cam_T_desk) @ np.diag([1.0, -1.0, -1.0, 1.0])), off, guide)
    check("a camera with the guide behind it draws nothing", not canvas.any())

    print("\n" + ("all checks passed" if not FAILURES else f"{len(FAILURES)} failed: {', '.join(FAILURES)}"))
    sys.exit(1 if FAILURES else 0)
