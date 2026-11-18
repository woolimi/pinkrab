"""The robot's arms drawn into the top camera's picture, following the hands.

Each hand's tool, as its solid is solved this frame, is carried into the
robot's world by where the desk solid stands there (follow.tool_absolute_target)
and its arm follows it as the shelf's replay will follow the take
(follow.Following): solved from the ready pose the first time, tracked after,
held where the hand is not solved. Until a hand is first seen its arm stands
at the ready pose, where the ready guide's Hermit hangs from it.

Each arm is drawn as its silhouette - every link but the gripper's, the real
Hermit being the gripper - outlined and faintly filled, so the picture stays
readable through it. An arm is grey while it follows, the guide's green while
its hand is at the ready pose, and red while it falls short of the hand
(SHORT_MM, SHORT_DEG): a stretch the replay will flag and convert cut, seen
while it is being recorded.
"""
from __future__ import annotations

import functools

import cv2
import numpy as np

from pinkrab_core.config import ARMS
from pinkrab_core.follow import Following, tool_absolute_target
from pinkrab_core.origin import ready_joints

from .config import SETTINGS
from .desk_solid import world_T_desk
from .gripper_scene import decimated
from .pnp import Intrinsics, _project
from .ready_guide import MATCHED_COLOUR, _label

ARM_COLOUR = (185, 185, 185)   # BGR
SHORT_COLOUR = (60, 60, 235)
# The arms drawn every frame are their meshes with corners this near each
# other made one, filled at this fraction of the picture's size: a silhouette
# needs no more, and every triangle more is time out of the camera's frame.
ARM_CELL_M = 0.012
MASK_SCALE = 2
FILL_ALPHA = 0.25
# A corner nearer the camera than this, or further off its axis than the
# picture's corners by this factor, is not drawn to: a lens model asked for a
# point far off its axis lands it anywhere.
NEAREST_M = 0.02
FIELD_MARGIN = 1.1


class Hologram:
    """Both arms, following the hands frame by frame. Built once a process
    (the arms' model is read for it); used from the one thread that draws."""

    def __init__(self) -> None:
        import placo

        from pinkrab_core.kinematics import ArmKinematics, fresh_urdf

        from .robot_drawn import _arm_parts

        self.world_T_desk = world_T_desk()
        self.robot = placo.RobotWrapper(str(fresh_urdf()))
        self.follow: dict[str, Following] = {}
        self.names: dict[str, tuple[str, ...]] = {}
        for arm in ARMS:
            chain = ArmKinematics(arm)
            self.follow[arm] = Following(chain, np.array(ready_joints(arm, chain.joint_names), dtype=float))
            self.names[arm] = tuple(chain.joint_names)
        # Each link's corners and the triangles between them, in the link's frame.
        self.links: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}
        for name, parts in _arm_parts().items():
            tris = decimated(np.concatenate([part.tris for part in parts]), ARM_CELL_M)
            verts, faces = np.unique(tris.reshape(-1, 3), axis=0, return_inverse=True)
            self.links[name] = (next(arm for arm in ARMS if name.startswith(f"{arm}_")), verts,
                                faces.reshape(-1, 3))
        self.seen_at: dict[str, float] = {}

    def follow_hands(self, cam_T_desk: np.ndarray, hands: dict[str, np.ndarray | None], now: float) -> None:
        """Step each arm after its hand: `hands` is each hand's solid as solved
        at `now` (cam_T_solid, seconds on any one clock), or None where it was
        not - that arm holds."""
        desk_T_cam = np.linalg.inv(cam_T_desk)
        for arm, cam_T_solid in hands.items():
            if cam_T_solid is None or arm not in self.follow:
                continue
            solid_T_tcp = SETTINGS.for_solid(f"handheld_{arm}").solid_T_tcp
            target = tool_absolute_target(arm, self.world_T_desk, desk_T_cam @ cam_T_solid @ solid_T_tcp)
            # The arm may close what the time since its hand was last seen
            # lets it: a hand lost for a moment is caught up with at the
            # arm's own pace.
            rate = 1.0 / max(now - self.seen_at.get(arm, now - 1.0 / 30.0), 1e-3)
            self.follow[arm].step(target, now, rate)
            self.seen_at[arm] = now

    def draw(self, canvas: np.ndarray, intrinsics: Intrinsics, cam_T_desk: np.ndarray,
             ready: dict[str, bool]) -> None:
        """Both arms where they follow, drawn on `canvas`, the desk solid at
        `cam_T_desk` in the picture; `ready` says which hands are at the
        ready pose."""
        for arm, follow in self.follow.items():
            for name, value in zip(self.names[arm], follow.joints):
                self.robot.set_joint(name, float(np.radians(value)))
        self.robot.update_kinematics()
        cam_T_world = cam_T_desk @ np.linalg.inv(self.world_T_desk)
        height, width = canvas.shape[:2]
        masks = {arm: np.zeros((height // MASK_SCALE, width // MASK_SCALE), np.uint8) for arm in self.follow}
        field = _field(intrinsics)
        for name, (arm, verts, faces) in self.links.items():
            cam_T = cam_T_world @ np.asarray(self.robot.get_T_world_frame(name))
            points = verts @ cam_T[:3, :3].T + cam_T[:3, 3]
            ahead = points[:, 2] > NEAREST_M
            ray = points[:, :2] / np.where(ahead, points[:, 2], 1.0)[:, None]
            seen = ahead & (np.linalg.norm(ray, axis=1) <= field)
            shown = faces[seen[faces].all(axis=1)]
            if len(shown):
                px = np.round(_project(cam_T, verts, intrinsics) / MASK_SCALE).astype(np.int32)
                # One triangle at a time: fillPoly given them all at once
                # takes four times as long.
                for triangle in px[shown]:
                    cv2.fillConvexPoly(masks[arm], triangle, 255)
        for arm, follow in self.follow.items():
            colour = (SHORT_COLOUR if follow.started and follow.short
                      else MATCHED_COLOUR if ready.get(arm) else ARM_COLOUR)
            _silhouette(canvas, masks[arm], colour)
            # How far an arm falls short, beside its wrist.
            wrist = cam_T_world @ np.asarray(self.robot.get_T_world_frame(f"{arm}_wrist_link"))
            if follow.started and follow.short and wrist[2, 3] > NEAREST_M:
                mm, deg = follow.reach
                at = _project(wrist, np.zeros((1, 3)), intrinsics)[0]
                _label(canvas, f"{arm} arm short  {mm:.0f} mm  {deg:.0f} deg", at, SHORT_COLOUR)


def _silhouette(canvas: np.ndarray, mask: np.ndarray, colour) -> None:
    """`mask` (the picture at 1/MASK_SCALE) laid on `canvas`: faintly filled
    in `colour` and outlined in it."""
    # The pinholes thinning the meshes leaves, closed.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return
    x, y, w, h = cv2.boundingRect(np.concatenate(contours))
    x0, y0 = x * MASK_SCALE, y * MASK_SCALE
    x1, y1 = min((x + w) * MASK_SCALE, canvas.shape[1]), min((y + h) * MASK_SCALE, canvas.shape[0])
    inside = cv2.resize(mask[y:y + h, x:x + w], (w * MASK_SCALE, h * MASK_SCALE),
                        interpolation=cv2.INTER_NEAREST)[:y1 - y0, :x1 - x0]
    roi = canvas[y0:y1, x0:x1]
    tinted = cv2.addWeighted(roi, 1.0 - FILL_ALPHA, np.full_like(roi, colour), FILL_ALPHA, 0.0)
    cv2.copyTo(tinted, inside, roi)
    cv2.polylines(canvas, [c * MASK_SCALE for c in contours], True, colour, 2, cv2.LINE_AA)


@functools.lru_cache(maxsize=4)
def _field(intrinsics: Intrinsics) -> float:
    """How far off its axis the camera sees, as the tangent of the angle: the
    picture's corners carried back through the lens, a little past."""
    corners = np.array([[0, 0], [intrinsics.width, 0], [0, intrinsics.height],
                        [intrinsics.width, intrinsics.height]], dtype=np.float64).reshape(-1, 1, 2)
    rays = cv2.undistortPoints(corners, intrinsics.matrix, intrinsics.distortion).reshape(-1, 2)
    return float(np.linalg.norm(rays, axis=1).max()) * FIELD_MARGIN
