"""Offline check of each handheld's frames against the description and the
printed parts: the solid's pose times solid_T_tcp, turned by TOOL_TURN as teleop
turns it, lands on that arm's gripper frame; and the solid sits on the
handheld body as far from it as on the other hand - on its peg, not beside
it. The left handheld is the right one's mirror image, and both have to hold.

    python src/pinkrab_umi/test/hand_frames_check.py
"""
from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from pinkrab_core.follow import TOOL_TURN
from pinkrab_core.kinematics import fresh_urdf
from pinkrab_umi.config import SETTINGS as UMI

FAILURES: list[str] = []
TOL_MM, TOL_DEG = 0.05, 0.05


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def placed(element: ET.Element) -> np.ndarray:
    """An URDF origin as a 4x4: fixed-axis roll, pitch, yaw."""
    origin = element.find("origin")
    r, p, y = (float(v) for v in (origin.get("rpy") or "0 0 0").split())
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    out = np.eye(4)
    out[:3, :3] = (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
                   @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
                   @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))
    out[:3, 3] = [float(v) for v in origin.get("xyz").split()]
    return out


def apart(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Millimetres and degrees between two poses."""
    turn = a[:3, :3].T @ b[:3, :3]
    deg = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(turn) - 1) / 2))))
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1000), deg


if __name__ == "__main__":
    root = ET.fromstring(fresh_urdf().read_text())
    gaps = {}
    for side in ("right", "left"):
        hand = UMI.for_solid(f"handheld_{side}")
        tool = placed(next(j for j in root.iter("joint") if j.get("name") == f"{side}_gripper_frame_joint"))
        claw = placed(next(v for link in root.iter("link") if link.get("name") == f"{side}_gripper_link"
                           for v in link.findall("visual")
                           if v.find("geometry/mesh").get("filename").endswith(f"claw_{side}.stl")))
        turn = np.eye(4)
        turn[:3, :3] = TOOL_TURN[side]
        solid = claw @ hand.claw_T_solid
        mm, deg = apart(solid @ hand.solid_T_tcp @ turn, tool)
        check(f"{side}: the solid times solid_T_tcp, turned, is the arm's gripper frame",
              mm < TOL_MM and deg < TOL_DEG, f"{mm:.3f} mm, {deg:.3f} deg off")
        body = trimesh.load(UMI.mesh_dir / f"handheld_body_{side}.stl")
        body.apply_transform(claw)
        mesh = trimesh.load(UMI.mesh_dir / f"marker_solid_handheld_{side}.stl")
        mesh.apply_transform(solid)
        inside = int((trimesh.proximity.signed_distance(body, mesh.vertices) > 0.0005).sum())
        gaps[side] = float(trimesh.proximity.closest_point(body, [solid[:3, 3]])[1][0] * 1000)
        check(f"{side}: the solid is clear of the body", inside == 0, f"{inside} vertices inside")
    check("the solid sits as close to its body on either hand - on the peg",
          abs(gaps["left"] - gaps["right"]) < 0.2, f"right {gaps['right']:.2f} mm, left {gaps['left']:.2f} mm")
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s): {FAILURES}")
        sys.exit(1)
    print("all ok")
