"""Rotations, poses and the one composition the pipeline is.

world_T_solid = world_T_cam @ cam_T_solid, where world_T_cam is the inverse of the
camera's pose against the desk solid in the same frame. Both halves come from one
picture, so there is nothing to align in time; slerp is here for the stream
filter, which smooths one pose toward the next.
"""

from __future__ import annotations

import numpy as np


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """(x, y, z, w) -> 3x3. TUM's order, which is not the order most maths books use."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(rot: np.ndarray) -> np.ndarray:
    """3x3 -> (x, y, z, w), by the largest-diagonal branch.

    The single-branch formula divides by something near zero for half of all
    rotations, and the half it fails on includes the 180 degree turns a head makes.
    """
    trace = np.trace(rot)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w, x, y, z = 0.25 * s, (rot[2, 1] - rot[1, 2]) / s, (rot[0, 2] - rot[2, 0]) / s, (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2
        w, x, y, z = (rot[2, 1] - rot[1, 2]) / s, 0.25 * s, (rot[0, 1] + rot[1, 0]) / s, (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2
        w, x, y, z = (rot[0, 2] - rot[2, 0]) / s, (rot[0, 1] + rot[1, 0]) / s, 0.25 * s, (rot[1, 2] + rot[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2
        w, x, y, z = (rot[1, 0] - rot[0, 1]) / s, (rot[0, 2] + rot[2, 0]) / s, (rot[1, 2] + rot[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Shortest-arc interpolation between two (x, y, z, w) quaternions."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    # q and -q are the same rotation but opposite ends of the sphere, so without
    # this the interpolation takes the long way round for half of all pairs.
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        # The arc is shorter than the numerical noise in acos; lerp is exact enough
        # and does not divide by a vanishing sine.
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    return (np.sin((1 - t) * theta) / sin_theta) * q0 + (np.sin(t * theta) / sin_theta) * q1


def pose_matrix(position: np.ndarray, quat: np.ndarray) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = quat_to_matrix(np.asarray(quat, float))
    pose[:3, 3] = np.asarray(position, float)
    return pose


def rotation_angle_deg(rot: np.ndarray) -> float:
    """The angle of a rotation matrix, in degrees - how far apart two frames point."""
    cos = (np.trace(rot[:3, :3]) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def compose(world_T_cam: np.ndarray, cam_T_solid: np.ndarray) -> np.ndarray:
    """world_T_solid. The whole point of the pipeline, in one line."""
    return world_T_cam @ cam_T_solid
