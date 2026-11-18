"""Where the grippers are in a demonstration's gripper pictures, frame by frame.

The grippers are drawn from geometry, not looked for: the robot's gripper
off the description (gripper_model), the camera where camera.yaml puts it on
the gripper (gripper_camera), the lens as the take recorded it (a
Kannala-Brandt fit), each jaw turned to its hand's opening at that moment,
and the other hand where the top camera placed it against this one - each
hand's tool off its solid, as convert places it. A camera sees its own
gripper the same way in every frame but for its jaw, so its own is drawn
from the jaw alone; the other is drawn only where both hands were placed.

The labels say where each gripper is, at the size the dataset keeps the
picture (gripper_view's window about the lens axis, scaled): OWN for this
camera's own gripper - its claw, jaw, servo and camera bracket - and OTHER
for the other hand's, as the robot's other arm will show it. robot_drawn
draws the robot in around them.
"""
from __future__ import annotations

import functools
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from pinkrab_core.config import ARMS
from pinkrab_core.dataset import GRIPPER_WIDTH, gripper_window
from pinkrab_core.follow import TOOL_TURN
from pinkrab_core.gripper_camera import camera_spec, mirrored, tcp_T_camera
from pinkrab_core.jaw import gripper_band, shut_wide

from .config import SETTINGS
from .convert import cameras_on_desk, exposures, fill_holes, jaw_readings, placed
from .episode import _lines
from .gripper_model import Gripper, gripper

OWN, OTHER = 1, 2
# A triangle with a corner nearer the lens than this, or behind it, is not
# drawn: the fisheye model folds what is behind back into the picture.
NEAREST_M = 0.002
# The meshes are drawn with corners this near each other made one: the
# servo and the claw are tens of thousands of triangles, the picture needs
# a few thousand.
CELL_M = 0.0015
# A gripper frame this far outside the head camera's frames takes the hands
# from the nearest one: about a frame at 30 fps.
EDGE_S = 0.04


def decimated(tris: np.ndarray, cell: float = CELL_M) -> np.ndarray:
    """Triangles (n,3,3) with corners within `cell` of each other made one,
    the triangles that leaves flat dropped."""
    corners = tris.reshape(-1, 3)
    keys, which = np.unique(np.floor(corners / cell).astype(np.int64), axis=0, return_inverse=True)
    which = which.ravel()
    verts = np.zeros((len(keys), 3))
    np.add.at(verts, which, corners)
    verts /= np.bincount(which)[:, None]
    faces = which.reshape(-1, 3)
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
    faces = np.unique(np.sort(faces, axis=1), axis=0)
    return verts[faces]


class Lens:
    """A gripper camera as the take recorded it, and the dataset's window
    of its pictures: points in the camera's frame to the window's pixels."""

    def __init__(self, stream: dict) -> None:
        lens = stream["lens"]
        if lens.get("model") != "kannala_brandt":
            raise ValueError(f"a {lens.get('model')} lens is not drawn here")
        self.K = np.array([[lens["fx"], 0.0, lens["cx"]], [0.0, lens["fy"], lens["cy"]], [0.0, 0.0, 1.0]])
        self.D = np.array(lens["k"], dtype=float).reshape(4, 1)
        cols, rows = lens["size"]
        self.x, self.y, w, h = gripper_window(cols, rows, tuple(stream["centre"]))
        self.scale = GRIPPER_WIDTH / w
        self.size = (round(h * self.scale), GRIPPER_WIDTH)   # (height, width), as gripper_view

    def draw(self, canvas: np.ndarray, tris_cam: np.ndarray, label: int) -> None:
        """Fill `tris_cam` (n,3,3, the camera's frame) on `canvas` as `label`."""
        tris = tris_cam[(tris_cam[:, :, 2] > NEAREST_M).all(axis=1)]
        if not len(tris):
            return
        px, _ = cv2.fisheye.projectPoints(tris.reshape(-1, 1, 3), np.zeros((3, 1)), np.zeros((3, 1)), self.K, self.D)
        px = (px.reshape(-1, 3, 2) - [self.x, self.y]) * self.scale
        cv2.fillPoly(canvas, list(np.round(px).astype(np.int32)), label)


def _moved(tris: np.ndarray, T: np.ndarray) -> np.ndarray:
    return tris @ T[:3, :3].T + T[:3, 3]


def jaw_angle(side: str, opening: float | None) -> float:
    shut, wide = shut_wide(*gripper_band(side))
    return shut + (0.0 if opening is None else float(opening)) * (wide - shut)


@functools.lru_cache(maxsize=1)
def grippers() -> tuple[dict[str, Gripper], dict[str, np.ndarray]]:
    """Each side's robot gripper, its meshes thinned, and each side's camera
    in its gripper link's frame - read once a process: the description does
    not change under a running studio."""
    from pinkrab_core.kinematics import fresh_urdf

    urdf = ET.parse(fresh_urdf()).getroot()
    out = {}
    for side in ARMS:
        g = gripper(urdf, side)
        g.fixed = {name: decimated(tris) for name, tris in g.fixed.items()}
        g.jaw = decimated(g.jaw)
        out[side] = g
    spec = camera_spec()
    right = out["right"].link_T_tcp @ tcp_T_camera(spec["offset_out"], spec["offset_back"], spec["pitch"], spec["yaw"])
    return out, {"right": right, "left": mirrored(right)}


def _lerp_pose(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    out = np.eye(4)
    out[:3, 3] = a[:3, 3] + w * (b[:3, 3] - a[:3, 3])
    turn = Rotation.from_matrix([a[:3, :3], b[:3, :3]])
    out[:3, :3] = (turn[0] * Rotation.from_rotvec(w * (turn[0].inv() * turn[1]).as_rotvec())).as_matrix()
    return out


def _at(tools: list[np.ndarray | None], times: np.ndarray, t: float) -> np.ndarray | None:
    """A hand's tool at moment `t`, between the head frames either side of
    it, or None where either has no place for the hand. A moment up to
    EDGE_S before the first head frame or after the last - a gripper frame
    taken as the head camera started or stopped - has the hand where that
    frame put it."""
    i = int(np.searchsorted(times, t))
    if i == 0:
        return tools[0] if len(times) and times[0] - t <= EDGE_S else None
    if i >= len(times):
        return tools[-1] if t - times[-1] <= EDGE_S else None
    a, b = tools[i - 1], tools[i]
    if a is None or b is None or times[i] <= times[i - 1]:
        return None
    return _lerp_pose(a, b, (t - times[i - 1]) / (times[i] - times[i - 1]))



class Scene:
    """A demonstration's grippers, ready to be drawn into any frame of its
    gripper videos: each hand's tool, the robot's tool frame, where the top
    camera placed it in the desk's frame, as convert places it, at each
    moment it was placed, and each jaw's angle at any moment."""

    def __init__(self, root: Path) -> None:
        meta = json.loads((root / "meta.json").read_text())
        if meta.get("kind", "demo") != "demo":
            raise ValueError(f"a {meta.get('kind')} is not a demonstration")
        frames = _lines(root / "frames.jsonl")
        streams = meta.get("streams") or {}
        self.sides = [side for side in ARMS if f"gripper_{side}" in streams]
        self.lenses = {side: Lens(streams[f"gripper_{side}"]) for side in self.sides}
        # Each gripper frame's moment, on the host clock the hands' are on,
        # in the video's order.
        self.stamps = {}
        for side in self.sides:
            shots = sorted((f for f in frames if f["stream"] == f"gripper_{side}"), key=lambda f: f["index"])
            if [f["index"] for f in shots] != list(range(len(shots))):
                raise ValueError(f"the gripper_{side} frame list skips frames of its video")
            self.stamps[side] = [f["host_ts"] - SETTINGS.gripper_camera_latency_s for f in shots]
        # Each hand's tool at self.times, None where it has no place, and its
        # jaw's angles (times, radians) - None where it was never read.
        self.tools: dict[str, list[np.ndarray | None]] = {}
        self.jaws: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
        poses = _lines(root / "poses.jsonl")
        self.times = exposures(poses, meta)
        world_T_cams = cameras_on_desk(poses, ARMS)
        for side in ARMS:
            solid_T_tcp = np.array(SETTINGS.recorded_solid_T_tcp(f"handheld_{side}", meta), dtype=float)
            hand = [W @ np.array(p["solids"][f"handheld_{side}"]["pose"]) @ solid_T_tcp
                    if W is not None and placed(p, side) else None for p, W in zip(poses, world_T_cams)]
            turn = np.eye(4)
            turn[:3, :3] = TOOL_TURN[side]
            self.tools[side] = [None if T is None else T @ turn
                                for T in fill_holes(hand, self.times, SETTINGS.convert_max_hole_s)]
            t, opening = jaw_readings(poses, side)
            self.jaws[side] = (t, np.array([jaw_angle(side, o) for o in opening])) if len(t) else None

    def _angle(self, side: str, t: float) -> float:
        jaw = self.jaws[side]
        return jaw_angle(side, None) if jaw is None else float(np.interp(t, *jaw))

    def draw(self, side: str, k: int) -> np.ndarray:
        """Frame `k` of `side`'s gripper video's labels, (height, width)
        uint8: OWN and OTHER where the grippers are, 0 elsewhere. A camera
        sees its own gripper the same way in every frame but for its jaw; the
        other is drawn only where both hands were placed."""
        t = self.stamps[side][k]
        return drawn(side, self.lenses[side], {s: _at(self.tools[s], self.times, t) for s in ARMS},
                     {s: self._angle(s, t) for s in ARMS})


def drawn(side: str, lens: Lens, tools: dict[str, np.ndarray | None], angles: dict[str, float]) -> np.ndarray:
    """`side`'s gripper camera's labels through `lens`, (height, width) uint8:
    OWN and OTHER where the grippers are, 0 elsewhere - each hand's tool
    (the robot's tool frame, in any frame both share) and jaw angle as given.
    A camera sees its own gripper the same way at any pose but for its jaw;
    the other is drawn only where both tools are known."""
    models, link_T_cam = grippers()
    other = next(s for s in ARMS if s != side)
    labels = np.zeros(lens.size, dtype=np.uint8)
    mine, theirs = tools.get(side), tools.get(other)
    if mine is not None and theirs is not None:
        world_T_cam = mine @ np.linalg.inv(models[side].link_T_tcp) @ link_T_cam[side]
        cam_T_other = np.linalg.inv(world_T_cam) @ theirs @ np.linalg.inv(models[other].link_T_tcp)
        lens.draw(labels, _moved(models[other].triangles(angles[other]), cam_T_other), OTHER)
    lens.draw(labels, _moved(models[side].triangles(angles[side]), np.linalg.inv(link_T_cam[side])), OWN)
    return labels

