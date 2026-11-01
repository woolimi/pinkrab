"""The gripper camera: where it sits on the hand, and a fisheye drawn from five views.

MuJoCo renders pinhole cameras, and a pinhole cannot see 90 degrees off its
axis, let alone the hundred a 200 degree lens takes in. So each fisheye frame is
drawn from five pinhole views with a common centre - forward, left, right, up,
down, 90 degrees square each - and every fisheye pixel is looked up in
whichever view its ray passes through. The lookup is a fixed table per lens,
made once, and one `cv2.remap` per frame.

Frames here are OpenCV's: x right, y down, z along the optical axis. MuJoCo
cameras look down their -z with +y up, which is this frame turned half a turn
about x.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from pinkrab_core.gripper_camera import (  # noqa: F401 - the sim's names
    camera_spec,
    claw_T_camera,
    tcp_T_camera,
)

from .config import LensSpec

# Where the fingertips meet with the jaw shut, in the claw mesh's frame - the
# point camera.yaml's pitch and yaw are solved to put on the image's centre
# line, a quarter of the way up from the bottom.
FINGERTIPS_CLAW = np.array([-0.0004, 0.0, 0.1119])

# The five faces, as (right, down, forward) in the fisheye's own frame.
FACES = (
    ("front", (1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ("right", (0, 0, -1), (0, 1, 0), (1, 0, 0)),
    ("left", (0, 0, 1), (0, 1, 0), (-1, 0, 0)),
    ("up", (1, 0, 0), (0, 0, 1), (0, -1, 0)),
    ("down", (1, 0, 0), (0, 0, -1), (0, 1, 0)),
)
# A camera's OpenCV axes into MuJoCo's: x kept, y and z reversed.
CV_TO_MJ = np.diag([1.0, -1.0, -1.0])


def project(lens: LensSpec, points_cam: np.ndarray) -> np.ndarray:
    """Points in the camera's OpenCV frame to fisheye pixels, (n, 2)."""
    p = np.atleast_2d(points_cam)
    theta = np.arctan2(np.linalg.norm(p[:, :2], axis=1), p[:, 2])
    phi = np.arctan2(p[:, 1], p[:, 0])
    r = lens.radius(theta)
    return np.column_stack([lens.cx + r * np.cos(phi), lens.cy + r * np.sin(phi)])


def solve_pitch_yaw(lens: LensSpec, out: float, back: float, rows_from_bottom: float = 0.25,
                    tol: float = 1e-9) -> tuple[float, float]:
    """Pitch and yaw that put the shut fingertips where camera.yaml wants them.

    On the image's vertical centre line, `rows_from_bottom` of the height up
    from its bottom edge - for the lens given, at the lens position given.
    Two conditions, two angles; Newton on both from level.
    """
    want = np.array([lens.cx, lens.height * (1.0 - rows_from_bottom)])
    x = np.array([0.0, 0.0])

    def at(angles):
        T = claw_T_camera(out, back, angles[0], angles[1])
        tip = np.linalg.inv(T) @ np.append(FINGERTIPS_CLAW, 1.0)
        return project(lens, tip[:3])[0]

    for _ in range(60):
        got = at(x)
        err = got - want
        if np.max(np.abs(err)) < tol:
            break
        J = np.empty((2, 2))
        for k in range(2):
            d = np.zeros(2)
            d[k] = 1e-6
            J[:, k] = (at(x + d) - got) / 1e-6
        x = x - np.linalg.solve(J, err)
    return float(x[0]), float(x[1])


class Fisheye:
    """The lookup from five square views to one fisheye frame, for one lens."""

    def __init__(self, lens: LensSpec, scale: float = 1.0, oversample: float = 1.2) -> None:
        self.width = int(round(lens.width * scale))
        self.height = int(round(lens.height * scale))
        f = lens.f * scale
        cx, cy = lens.cx * scale, lens.cy * scale
        # A pinhole face of N pixels has N/2 pixels per unit of tan at its
        # centre, and the fisheye f per radian: match them, then some.
        self.face = int(math.ceil(2.0 * f * oversample / 2.0) * 2)
        n = self.face
        u, v = np.meshgrid(np.arange(self.width) + 0.5, np.arange(self.height) + 0.5)
        dx, dy = u - cx, v - cy
        theta = lens.theta_at(np.hypot(dx, dy) / scale)
        phi = np.arctan2(dy, dx)
        # A pixel past what the lens model answers for has no ray; it is
        # masked below, and drawn here along the axis so the arithmetic holds.
        ray = np.where(np.isfinite(theta), theta, 0.0)
        d = np.stack([np.sin(ray) * np.cos(phi), np.sin(ray) * np.sin(phi), np.cos(ray)], axis=-1)
        axes = np.array([[r, dn, fw] for _, r, dn, fw in FACES], dtype=float)   # (5, 3, 3)
        facing = np.einsum("hwc,kc->hwk", d, axes[:, 2])
        which = np.argmax(facing, axis=-1)
        a = axes[which]                                                          # (h, w, 3, 3)
        z = np.einsum("hwc,hwc->hw", d, a[..., 2, :])
        x = np.einsum("hwc,hwc->hw", d, a[..., 0, :]) / z
        y = np.einsum("hwc,hwc->hw", d, a[..., 1, :]) / z
        col = np.clip((x + 1.0) * n / 2.0 - 0.5, 0.0, n - 1.0) + which * n
        row = np.clip((y + 1.0) * n / 2.0 - 0.5, 0.0, n - 1.0)
        outside = theta > lens.fov / 2.0
        col[outside] = -1.0
        row[outside] = -1.0
        self.map_x = col.astype(np.float32)
        self.map_y = row.astype(np.float32)

    @staticmethod
    def face_rotations() -> list[tuple[str, np.ndarray]]:
        """Each face's OpenCV axes in the fisheye's frame, by name."""
        return [(name, np.column_stack([r, dn, fw]).astype(float)) for name, r, dn, fw in FACES]

    def compose(self, faces: list[np.ndarray]) -> np.ndarray:
        """Five renders, in FACES order, into one fisheye frame."""
        atlas = np.concatenate(faces, axis=1)
        return cv2.remap(atlas, self.map_x, self.map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
