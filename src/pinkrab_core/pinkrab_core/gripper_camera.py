"""Where the gripper camera sits on a gripper - the robot's and the Hermit's alike.

The same bracket bolts to the robot's claw and to the Hermit with the same two
screws, so one pose serves both: camera.yaml's pose block, the camera in the
right hand's TCP frame, the left the right mirrored across its gripper link's
y. Frames here are OpenCV's: x right, y down, z along the optical axis.
"""
from __future__ import annotations

import math

import numpy as np
import yaml

from .config import SETTINGS

# The TCP in the claw mesh's frame (the description's gripper_frame_joint and
# the claw's visual origin, composed): the camera spec is written in the TCP
# frame, the bracket is cut in the claw's.
CLAW_R_TCP = np.diag([-1.0, -1.0, 1.0])
CLAW_P_TCP = np.array([-0.0079, 0.0, 0.10908])


def camera_spec() -> dict:
    """camera.yaml's pose block: offset_out, offset_back, pitch, yaw."""
    return yaml.safe_load(SETTINGS.camera_file.read_text())["pose"]


def claw_T_camera(out: float, back: float, pitch: float, yaw: float) -> np.ndarray:
    """The camera in the claw mesh's frame, OpenCV axes, the right hand's side.

    The same construction the bracket (meshes/camera_bracket_*.stl) is cut
    from: the lens out of the opening plane and
    back along the approach axis, the optical axis pitched toward the fingers
    and yawed toward the jaw, the image's horizontal along the opening axis and
    its bottom toward the claw body.
    """
    lens = CLAW_R_TCP @ np.array([0.0, out, -back]) + CLAW_P_TCP
    cp, sp, cy, sy = math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    n = np.array([cp * sy, sp, cp * cy])
    u = np.array([1.0, 0.0, 0.0]) - n[0] * n
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([u, v, n])
    T[:3, 3] = lens
    return T


def claw_T_tcp() -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = CLAW_R_TCP
    T[:3, 3] = CLAW_P_TCP
    return T


def tcp_T_camera(out: float, back: float, pitch: float, yaw: float) -> np.ndarray:
    return np.linalg.inv(claw_T_tcp()) @ claw_T_camera(out, back, pitch, yaw)


def mirrored(link_T_camera: np.ndarray) -> np.ndarray:
    """The right hand's camera in its gripper link's frame as the left hand
    has it: mirrored across the link's y, the image with it - the fingers
    still enter from the bottom."""
    M = np.diag([1.0, -1.0, 1.0])
    out = link_T_camera.copy()
    out[:3, :3] = M @ link_T_camera[:3, :3] @ np.diag([-1.0, 1.0, 1.0])
    out[:3, 3] = M @ link_T_camera[:3, 3]
    return out
