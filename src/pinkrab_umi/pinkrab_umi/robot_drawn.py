"""The robot drawn into a demonstration's gripper pictures, as a rollout's cameras will see it.

A take made with the Hermits shows the hands that held them; on the robot the
same cameras see the arms and the pegboard they hang on. Convert can draw
those in. Each hand's arm stands where the shelf's replay puts it under that
hand: the hand's tool carried into the robot's world from the desk solid at
the chest (desk_solid.carried), the first placed moment solved from the ready
pose and the arm following from there. Its links are drawn from the wrist
down - the gripper itself, claw, jaw, servo and camera bracket, is the
Hermit's, which the picture already shows - and the board (scene.xml's
`pegboard`, the face the bases bolt to) behind them. The camera drawing is
where the hand held it: the picture was taken from there.

Nothing is taken out of the picture. A drawn part lies over whatever the
picture shows where it falls - the hand that held the Hermit, its marker
solid, the room - but for the grippers, which gripper_scene places in every frame
and which stay as recorded. The printed parts take the pink the other hand's
Hermit shows in the same picture, so a drawn arm is lit as the room lit the
Hermit; the servos, the board and its slots take the Look's colours.
"""
from __future__ import annotations

import functools
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pinkrab_core.config import ARMS
from pinkrab_core.config import SETTINGS as CORE

from .config import SETTINGS
from .desk_solid import carried
from .gripper_model import origin, stl_triangles
from .gripper_scene import NEAREST_M, OTHER, OWN, Scene, _at, _moved, decimated, grippers

# The links a hand's own gripper is made of: the Hermit's are in the picture.
GRIPPER_LINKS = ("gripper_link", "gripper_servo_link", "camera_bracket_link", "gripper_frame_link", "jaw_link")
# What counts as the Hermit's pink in a picture (OpenCV HSV): hue either side
# of red's wrap, well saturated, not in shadow.
PINK_HUE = (150, 5)
PINK_MIN_S, PINK_MIN_V = 90, 60
# The gripper cameras' field, the vendor's, across the image circle; a
# triangle is drawn with corners this far past its edge.
FIELD_DEG = 200.0
RAY_MARGIN_DEG = 10.0
# Fewer pink pixels of the other Hermit than this and the Look's pink is used.
PINK_MIN_PIXELS = 100
# A face's shade: this much of its colour edge on, all of it facing the
# camera; the measured pink is what a face at the middle shade shows.
SHADE_FLOOR = 0.6


class DrawError(ValueError):
    """A take the robot cannot be drawn into."""


@dataclass(frozen=True)
class Look:
    """The colours (BGR) of what a picture does not show itself: the servos,
    the board and its slots (None: a plain board), and the pink for a
    picture with no other Hermit in it."""
    pink: tuple[int, int, int]
    servo: tuple[int, int, int]
    board: tuple[int, int, int]
    slot: tuple[int, int, int] | None

    @classmethod
    def rig(cls) -> "Look":
        """The rig's: its black pegboard and its slots (the settings')."""
        return cls(SETTINGS.robot_drawn_pink_bgr, SETTINGS.robot_drawn_servo_bgr,
                   SETTINGS.robot_drawn_board_bgr, SETTINGS.robot_drawn_slot_bgr)

    @classmethod
    def described(cls) -> "Look":
        """The description's own colours - the printed parts', the servos'
        and scene.xml's plain board - which the simulation renders."""
        from pinkrab_core.kinematics import fresh_urdf

        def bgr(rgba: str) -> tuple[int, int, int]:
            r, g, b = (round(255 * float(v)) for v in rgba.split()[:3])
            return (b, g, r)

        urdf = ET.parse(fresh_urdf()).getroot()
        colours = {m.get("name"): bgr(m.find("color").get("rgba")) for m in urdf.findall("material")
                   if m.find("color") is not None}
        board = ET.parse(CORE.scene_file).getroot().find(".//material[@name='board']")
        if board is None or "3d_printed" not in colours or "sts3215" not in colours:
            raise DrawError("the description does not colour the printed parts, the servos and the board")
        return cls(colours["3d_printed"], colours["sts3215"], bgr(board.get("rgba")), None)


@dataclass(frozen=True)
class _Part:
    tris: np.ndarray    # (n,3,3) in its link's frame
    printed: bool       # a printed part (pink), else a servo


@functools.lru_cache(maxsize=1)
def _arm_parts() -> dict[str, list[_Part]]:
    """Each arm link's meshes but the grippers', thinned - read once a process."""
    from pinkrab_core.kinematics import fresh_urdf

    urdf = ET.parse(fresh_urdf()).getroot()
    out: dict[str, list[_Part]] = {}
    for link in urdf.findall("link"):
        name = link.get("name")
        if not name.startswith(tuple(f"{arm}_" for arm in ARMS)) or name.endswith(GRIPPER_LINKS):
            continue
        parts = []
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None or not Path(mesh.get("filename")).is_file():
                continue
            T = origin(visual.find("origin"))
            tris = stl_triangles(Path(mesh.get("filename"))) * float(mesh.get("scale", "1 1 1").split()[0])
            tris = decimated(tris, SETTINGS.robot_drawn_cell_m)
            material = visual.find("material")
            parts.append(_Part(tris @ T[:3, :3].T + T[:3, 3],
                               material is not None and material.get("name") != "sts3215"))
        if parts:
            out[name] = parts
    if not out:
        raise DrawError("the description has no arm meshes to draw")
    return out


@functools.lru_cache(maxsize=1)
def _board() -> tuple[np.ndarray, np.ndarray]:
    """The board's face in the robot's world, as triangles - its face cut in
    tiles, so a tile behind the lens leaves the rest drawn - and its slots,
    a hair in front of it."""
    root = ET.parse(CORE.scene_file).getroot()
    geom = root.find(".//geom[@name='pegboard']")
    if geom is None:
        raise DrawError(f"{CORE.scene_file} has no pegboard")
    half = np.array([float(v) for v in geom.get("size").split()])
    centre = np.array([float(v) for v in geom.get("pos", "0 0 0").split()])
    face = centre[0] + half[0]
    y0, y1 = centre[1] - half[1], centre[1] + half[1]
    z0, z1 = centre[2] - half[2], centre[2] + half[2]

    def quads(ys, zs, x):
        out = []
        for ya, yb in ys:
            for za, zb in zs:
                a, b, c, d = [x, ya, za], [x, yb, za], [x, yb, zb], [x, ya, zb]
                out += [[a, b, c], [a, c, d]]
        return out

    tile = 0.04
    ys = [(y, min(y + tile, y1)) for y in np.arange(y0, y1, tile)]
    zs = [(z, min(z + tile, z1)) for z in np.arange(z0, z1, tile)]
    board = np.array(quads(ys, zs, face))
    w, h = (v / 1000.0 for v in SETTINGS.robot_drawn_slot_mm)
    pitch, row = SETTINGS.robot_drawn_slot_pitch_mm / 1000.0, SETTINGS.robot_drawn_row_pitch_mm / 1000.0
    slots = []
    for r, z in enumerate(np.arange(z0 + row / 2 + h / 2, z1 - h / 2, row)):
        for y in np.arange(y0 + pitch / 2 + (r % 2) * pitch / 2, y1 - w / 2, pitch):
            slots += quads([(y - w / 2, y + w / 2)], [(z - h / 2, z + h / 2)], face + 0.0005)
    return board, np.array(slots).reshape(-1, 3, 3)


def _followed(chain, ready: np.ndarray, targets: list[np.ndarray | None], times: np.ndarray,
              rate_hz: float) -> np.ndarray:
    """An arm's joints (n, dof) under each target in turn, degrees: at ready
    until the first, then followed as the shelf's replay follows a take
    (follow.Following), held where a target is missing."""
    from pinkrab_core.follow import Following

    out = np.empty((len(targets), len(ready)))
    follow = Following(chain, ready)
    for i, target in enumerate(targets):
        if target is not None:
            follow.step(target, float(times[i]), rate_hz)
        out[i] = follow.joints
    return out


class RobotDrawn:
    """The robot, ready to be drawn into any frame of a demonstration's
    gripper videos, as convert keeps them (gripper_view) at `size`."""

    def __init__(self, src: Path, look: Look, size: tuple[int, int]) -> None:
        import placo

        from pinkrab_core.kinematics import ArmKinematics, fresh_urdf
        from pinkrab_core.origin import ready_joints

        meta = json.loads((src / "meta.json").read_text())
        world_T_desk, why = carried(meta.get("desk") or {})
        if world_T_desk is None:
            raise DrawError(f"no place in the robot's world for its hands: {why}")
        try:
            self.scene = Scene(src)
        except ValueError as exc:
            raise DrawError(str(exc)) from None
        if sorted(self.scene.sides) != sorted(ARMS):
            raise DrawError("the robot is drawn into a take of both hands")
        self.optics = {side: Optics(lens, meta["streams"][f"gripper_{side}"])
                       for side, lens in self.scene.lenses.items()}
        for side, lens in self.scene.lenses.items():
            if tuple(lens.size) != tuple(size):
                raise DrawError(f"the {side} lens draws {lens.size[1]}x{lens.size[0]}, the pictures are "
                                f"{size[1]}x{size[0]}")
        self.look = look
        self.world_T_desk = world_T_desk
        self.models, self.link_T_cam = grippers()
        self.parts = _arm_parts()
        self.board, self.slots = _board()
        self.robot = placo.RobotWrapper(str(fresh_urdf()))
        times = np.asarray(self.scene.times, dtype=float)
        rate = (len(times) - 1) / max(float(times[-1] - times[0]), 1e-6) if len(times) > 1 else 30.0
        self.times = times
        self.joints: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        for arm in ARMS:
            chain = ArmKinematics(arm)
            ready = np.array(ready_joints(arm, chain.joint_names), dtype=float)
            targets = [None if T is None else world_T_desk @ T for T in self.scene.tools[arm]]
            if not any(T is not None for T in targets):
                raise DrawError(f"the {arm} hand is never placed")
            self.joints[arm] = (chain.joint_names, _followed(chain, ready, targets, times, rate))

    def _pose(self, t: float) -> None:
        for names, track in self.joints.values():
            for j, name in enumerate(names):
                self.robot.set_joint(name, float(np.radians(np.interp(t, self.times, track[:, j]))))
        self.robot.update_kinematics()

    def draw(self, view: np.ndarray, side: str, k: int) -> np.ndarray:
        """Frame `k` of `side`'s gripper video, `view` as convert keeps it
        (BGR), with the robot drawn in: unchanged where the hand holding this
        camera has no place at that moment."""
        labels = self.scene.draw(side, k)
        layers = self.layers(side, k, self._pink(view, labels))
        if layers is None:
            return view
        layer, alpha = _raster(self.optics[side], layers)
        keep = cv2.dilate(((labels == OWN) | (labels == OTHER)).astype(np.uint8), np.ones((3, 3), np.uint8))
        alpha *= 1.0 - cv2.GaussianBlur(keep.astype(float), (3, 3), 0)[..., None]
        return (view * (1.0 - alpha) + layer * alpha).astype(np.uint8)

    def layers(self, side: str, k: int, pink: np.ndarray) -> list | None:
        """What frame `k` of `side`'s camera sees of the robot, for _raster:
        the board, its slots, then the arms - each in the camera's frame - or
        None where the hand holding it has no place at that moment."""
        t = self.scene.stamps[side][k]
        tool = _at(self.scene.tools[side], self.scene.times, t)
        if tool is None:
            return None
        self._pose(t)
        world_T_cam = self.world_T_desk @ tool @ np.linalg.inv(self.models[side].link_T_tcp) @ self.link_T_cam[side]
        cam_T_world = np.linalg.inv(world_T_cam)
        out = [(_moved(self.board, cam_T_world), np.array(self.look.board, float), False)]
        if self.look.slot is not None:
            out.append((_moved(self.slots, cam_T_world), np.array(self.look.slot, float), False))
        arms, colours = [], []
        for name, parts in self.parts.items():
            T = cam_T_world @ np.asarray(self.robot.get_T_world_frame(name))
            for part in parts:
                arms.append(_moved(part.tris, T))
                colours.append(np.repeat([pink if part.printed else self.look.servo], len(part.tris), 0))
        out.append((np.concatenate(arms), np.concatenate(colours).astype(float), True))
        return out

    def _pink(self, view: np.ndarray, labels: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(view, cv2.COLOR_BGR2HSV)
        hue = hsv[..., 0].astype(int)
        other = cv2.erode((labels == OTHER).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        pink = other & ((hue >= PINK_HUE[0]) | (hue <= PINK_HUE[1])) & (hsv[..., 1] > PINK_MIN_S) & (hsv[..., 2] > PINK_MIN_V)
        if pink.sum() < PINK_MIN_PIXELS:
            return np.array(self.look.pink, float)
        return np.median(view[pink], axis=0).astype(float)


class Optics:
    """A gripper camera as the robot is drawn through it: gripper_scene's lens
    (Kannala-Brandt, and the dataset's window of the picture) carried to
    rays off its axis by their own angle - a fisheye of 200 degrees sees
    what is behind the plane of its lens, which a triangle with a corner
    there is drawn through as well - and the image circle, which nothing is
    drawn outside of."""

    def __init__(self, lens, stream: dict) -> None:
        self.size, self.x, self.y, self.scale = lens.size, lens.x, lens.y, lens.scale
        self.f = np.array([lens.K[0, 0], lens.K[1, 1]])
        self.c = np.array([lens.K[0, 2], lens.K[1, 2]])
        self.k = lens.D.reshape(-1)
        # Rays out to the vendor's field and a little past, for a triangle
        # across its edge - no further than the polynomial keeps growing.
        grid = np.radians(np.arange(0.0, FIELD_DEG / 2 + RAY_MARGIN_DEG, 0.25))
        grows = np.diff(self._bent(grid)) > 0
        self.widest = float(grid[np.argmin(grows)] if not grows.all() else grid[-1])
        circle = stream["lens"].get("circle")
        if circle:
            centre, radius = np.array([circle["cx"], circle["cy"]]), circle["diameter"] / 2
        else:
            centre, radius = self.c, float(self.f.mean() * self._bent(np.radians(FIELD_DEG / 2)))
        self.circle = ((centre - [self.x, self.y]) * self.scale, radius * self.scale)

    def _bent(self, theta: np.ndarray) -> np.ndarray:
        t2 = theta * theta
        return theta * (1 + t2 * (self.k[0] + t2 * (self.k[1] + t2 * (self.k[2] + t2 * self.k[3]))))

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Points (..., 3) in the camera's frame to the window's pixels, and
        whether each is a ray the lens takes."""
        across = np.linalg.norm(points[..., :2], axis=-1)
        theta = np.arctan2(across, points[..., 2])
        out = self._bent(theta) / np.maximum(across, 1e-12)
        px = self.c + self.f * points[..., :2] * out[..., None]
        ok = (theta <= self.widest) & (np.linalg.norm(points, axis=-1) > NEAREST_M)
        return (px - [self.x, self.y]) * self.scale, ok


def _raster(optics: Optics, layers: list[tuple[np.ndarray, np.ndarray, bool]]) -> tuple[np.ndarray, np.ndarray]:
    """`layers` drawn through `optics` in order - each (triangles in the
    camera's frame, a colour for all or one each, shaded?) and the shaded
    ones far to near among themselves - as a picture and its coverage
    (0..1), each the window's size."""
    ss = SETTINGS.robot_drawn_supersample
    h, w = optics.size
    picture = np.zeros((h * ss, w * ss, 3), np.uint8)
    for tris, colour, lit in layers:
        colours = np.broadcast_to(colour, (len(tris), 3))
        px, ok = optics.project(tris)
        px = np.round(px * ss).astype(np.int32)
        lo, hi = px.min(axis=1), px.max(axis=1)
        seen = ok.all(axis=1) & (hi[:, 0] >= 0) & (hi[:, 1] >= 0) & (lo[:, 0] < w * ss) & (lo[:, 1] < h * ss)
        order = np.arange(len(tris))
        shade = np.ones(len(tris))
        if lit:
            normal = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
            normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
            centre = tris.mean(axis=1)
            distance = np.linalg.norm(centre, axis=1)
            facing = np.abs((normal * centre).sum(axis=1)) / np.maximum(distance, 1e-9)
            shade = (SHADE_FLOOR + (1.0 - SHADE_FLOOR) * facing) / ((1.0 + SHADE_FLOOR) / 2)
            order = np.argsort(-distance)
        # Every colour drawn is brighter than black, so what is not black is
        # what was drawn.
        fill = np.clip(colours * shade[:, None], 1, 255)
        for i in order[seen[order]]:
            cv2.fillPoly(picture, [px[i]], fill[i].tolist())
    cover = (picture.max(axis=2) > 0).astype(np.uint8) * 255
    (cx, cy), radius = optics.circle
    inside = np.zeros_like(cover)
    cv2.circle(inside, (round(cx * ss), round(cy * ss)), round(radius * ss), 255, -1)
    cover &= inside
    picture = cv2.resize(picture, (w, h), interpolation=cv2.INTER_AREA).astype(float)
    alpha = cv2.resize(cover, (w, h), interpolation=cv2.INTER_AREA).astype(float)[..., None] / 255.0
    # Colour averaged with the black of what is not drawn, at an edge: undone.
    picture = np.where(alpha > 0, picture / np.maximum(alpha, 1e-6), 0.0)
    return np.clip(picture, 0, 255), alpha
