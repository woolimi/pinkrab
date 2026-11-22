"""Where each hand is held to start a take, drawn into the top camera's picture.

A policy run starts its arms where a take's hands began (demo_start), and a
hand that only waits through the take keeps its arm where it waits. A hand
held out of the arm's reach there - nearer the board, lower or flatter than
the arm can hold its tool - leaves the arm short of the take from its first
moment. The robot's ready pose is one the arm holds: its tool, carried back
onto the desk solid as the replay carries a hand out to the arm
(follow.desk_tool), is where the hand's tool goes. The picture shows each
hand its Hermit standing there, drawn whole - the handheld's body and jaw,
the servo, the camera bracket and the marker solid - in lines: its outline
and the sharpest creases of the faces the camera would see, to lay the real
one over (a fill would cost the picture a frame in three), a line from the
real jaw's tip to the drawn one's, and
how far the hand is from it, in green once within ready_guide_mm and
ready_guide_deg. Its size tells near from far: a Hermit too near the camera
looks bigger than it.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import NamedTuple

import cv2
import numpy as np

from pinkrab_core.config import ARMS
from pinkrab_core.follow import TOOL_TURN, desk_tool
from pinkrab_core.origin import ready_joints

from .config import SETTINGS
from .desk_solid import world_T_desk
from .gripper_model import gripper, merged, stl_triangles
from .pnp import Intrinsics, _project

GUIDE_COLOUR = (235, 235, 235)   # BGR: the ready pose's Hermit, not yet matched
MATCHED_COLOUR = (60, 230, 60)
SHADOW = (30, 30, 30)
# The Hermit drawn every frame is its meshes with corners this near each other
# made one - thousands of faces rather than tens of thousands - and an edge
# between two faces drawn where they meet at more than this.
HERMIT_CELL_M = 0.005
HERMIT_CREASE_DEG = 60.0


class Shape(NamedTuple):
    """A Hermit as a mesh coarse enough to draw every frame, in its tool frame
    (the handheld's convention), metres: its corners, its faces (outward by
    the order of their corners) with their normals and centres, and its edges -
    each with its one or two faces (-1 for none) and whether it is a crease."""
    verts: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    centres: np.ndarray
    edges: np.ndarray
    edge_faces: np.ndarray
    crease: np.ndarray


class Guide(NamedTuple):
    tools: dict[str, np.ndarray]     # each hand's tool at the ready pose, against the desk solid, 4x4 m
    hermits: dict[str, Shape]        # each Hermit, in its tool frame


def load() -> Guide:
    """The guide for the desk solid as desk.yaml stands it today: each hand's
    tool centre against the solid that asks its arm for the tool at the ready
    pose, and each Hermit."""
    from pinkrab_core.kinematics import ArmKinematics, fresh_urdf

    desk = world_T_desk()
    urdf = ET.parse(fresh_urdf()).getroot()
    tools, hermits = {}, {}
    for arm in ARMS:
        chain = ArmKinematics(arm)
        tool = chain.forward(np.array(ready_joints(arm, chain.joint_names), dtype=float))
        tools[arm] = desk_tool(arm, desk, tool)
        hermits[arm] = hermit_shape(urdf, arm, SETTINGS.for_solid(f"handheld_{arm}").solid_T_tcp)
    return Guide(tools, hermits)


def _shape(tris: np.ndarray) -> Shape:
    """Triangles (n,3,3) as a Shape: corners within HERMIT_CELL_M of each other
    made one, the faces that leaves flat dropped, and an edge a crease where
    its faces meet at more than HERMIT_CREASE_DEG or it has only one."""
    verts, faces = merged(tris, HERMIT_CELL_M)
    a, b, c = (verts[faces[:, k]] for k in range(3))
    normals = np.cross(b - a, c - a)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    # Each edge with the faces either side of it.
    pairs = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(len(faces)), 3)
    edges, slot, counts = np.unique(pairs, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(slot.ravel(), kind="stable")
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    edge_faces = np.full((len(edges), 2), -1)
    edge_faces[:, 0] = owner[order[starts]]
    two = counts >= 2
    edge_faces[two, 1] = owner[order[starts[two] + 1]]
    cos = np.einsum("ij,ij->i", normals[edge_faces[:, 0]], normals[np.maximum(edge_faces[:, 1], 0)])
    crease = (edge_faces[:, 1] < 0) | (cos < np.cos(np.radians(HERMIT_CREASE_DEG)))
    return Shape(verts, faces, normals, (a + b + c) / 3.0, edges, edge_faces, crease)


def hermit_shape(urdf: ET.Element, arm: str, solid_T_tcp: np.ndarray) -> Shape:
    """A Hermit in its tool frame (the handheld's convention): the arm's
    gripper link and what hangs from it - the servo, the camera bracket, the
    jaw shut - with the claw and the jaw the handheld's own, as the hands view
    puts a Hermit together (hermit.js), and its marker solid where the
    handheld holds it."""
    own = {f"claw_{arm}.stl": SETTINGS.mesh_dir / f"handheld_body_{arm}.stl",
           f"jaw_{arm}.stl": SETTINGS.mesh_dir / f"handheld_jaw_{arm}.stl"}
    model = gripper(urdf, arm, own)
    tool = model.link_T_tcp
    # Into the tool frame the arm is asked for, then the handheld's: TOOL_TURN of it.
    turn = TOOL_TURN[arm] @ tool[:3, :3].T
    hermit = (model.triangles(0.0) - tool[:3, 3]) @ turn.T
    solid = np.linalg.inv(solid_T_tcp)
    marker = stl_triangles(SETTINGS.mesh_dir / f"marker_solid_handheld_{arm}.stl") @ solid[:3, :3].T + solid[:3, 3]
    return _shape(np.concatenate([hermit, marker]))


def apart(now: np.ndarray, want: np.ndarray) -> tuple[float, float]:
    """How far tool pose `now` is from `want`: millimetres, and degrees."""
    mm = float(np.linalg.norm(now[:3, 3] - want[:3, 3])) * 1000.0
    turn = now[:3, :3].T @ want[:3, :3]
    deg = float(np.degrees(np.arccos(np.clip((np.trace(turn) - 1.0) / 2.0, -1.0, 1.0))))
    return mm, deg


def at_ready(cam_T_desk: np.ndarray, arm: str, cam_T_solid: np.ndarray, guide: Guide) -> tuple[float, float, bool]:
    """How far hand `arm`'s tool, its solid solved at `cam_T_solid`, is from
    where the ready pose holds it - mm and degrees - and whether that is
    within ready_guide_mm and ready_guide_deg, where the guide turns green."""
    solid_T_tcp = SETTINGS.for_solid(f"handheld_{arm}").solid_T_tcp
    mm, deg = apart(np.linalg.inv(cam_T_desk) @ cam_T_solid @ solid_T_tcp, guide.tools[arm])
    return mm, deg, mm <= SETTINGS.ready_guide_mm and deg <= SETTINGS.ready_guide_deg


def _label(canvas: np.ndarray, text: str, at: np.ndarray, colour) -> None:
    x, y = int(at[0]) - 40, int(at[1]) - 14
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, SHADOW, 5, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)


def draw(canvas: np.ndarray, intrinsics: Intrinsics, cam_T_desk: np.ndarray,
         hands: dict[str, np.ndarray | None], guide: Guide) -> None:
    """Each hand's Hermit where the ready pose would hold it, drawn on
    `canvas`, the desk solid at `cam_T_desk` in the picture: `hands` is each
    hand's solid as solved now (cam_T_solid), or None where it was not, and
    `guide` load()'s answer."""
    for arm, want in guide.tools.items():
        solid_T_tcp = SETTINGS.for_solid(f"handheld_{arm}").solid_T_tcp
        colour, words = GUIDE_COLOUR, ""
        now = hands.get(arm)
        if now is not None:
            mm, deg, matched = at_ready(cam_T_desk, arm, now, guide)
            if matched:
                colour = MATCHED_COLOUR
            words = f"  {mm:.0f} mm  {deg:.0f} deg"
        shape = guide.hermits[arm]
        cam_T = cam_T_desk @ want
        R, t = cam_T[:3, :3], cam_T[:3, 3]
        if ((shape.verts @ R.T + t)[:, 2] <= 0.0).any():
            continue   # behind the camera, or through it
        px = np.round(_project(cam_T, shape.verts, intrinsics)).astype(np.int32)
        # The outline - an edge between a face turned to the camera and one
        # turned away - and the creases between the faces turned to it.
        front = np.einsum("ij,ij->i", shape.normals @ R.T, shape.centres @ R.T + t) < 0.0
        near = front[shape.edge_faces[:, 0]]
        far = np.where(shape.edge_faces[:, 1] >= 0, front[np.maximum(shape.edge_faces[:, 1], 0)], False)
        outline = near != far
        cv2.polylines(canvas, list(px[shape.edges[shape.crease & near & far]]), False, colour, 1, cv2.LINE_AA)
        cv2.polylines(canvas, list(px[shape.edges[outline]]), False, colour, 2, cv2.LINE_AA)
        # A line from the real jaw's tip to the drawn one's: the way to move.
        tip = _project(cam_T, np.zeros((1, 3)), intrinsics)[0].astype(np.int32)
        cv2.circle(canvas, tuple(tip), 8, colour, 2, cv2.LINE_AA)
        if now is not None and (now @ solid_T_tcp)[2, 3] > 0.0:
            real = _project(now @ solid_T_tcp, np.zeros((1, 3)), intrinsics)[0].astype(np.int32)
            cv2.line(canvas, tuple(real), tuple(tip), colour, 2, cv2.LINE_AA)
            cv2.circle(canvas, tuple(real), 6, colour, -1, cv2.LINE_AA)
        _label(canvas, f"{arm} ready{words}", px[np.argmin(px[:, 1])], colour)
