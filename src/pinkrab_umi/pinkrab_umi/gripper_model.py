"""A gripper as the description draws it, for pictures: its parts' triangles.

The robot's gripper link carries the claw, the jaw servo and the camera
bracket with its camera, and the jaw hangs from it on its joint; the Hermit
is the same gripper with the claw and jaw swapped for the handheld's own
(hermit.js puts it together the same way). Both are read here off the
description, each part in the gripper link's frame, the jaw turned to an
angle, with the tool frame (gripper_frame_link) the arm is asked for.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def origin(element) -> np.ndarray:
    """A URDF <origin> as a 4x4, identity when there is none."""
    out = np.eye(4)
    if element is not None:
        out[:3, :3] = Rotation.from_euler("xyz", [float(v) for v in element.get("rpy", "0 0 0").split()]).as_matrix()
        out[:3, 3] = [float(v) for v in element.get("xyz", "0 0 0").split()]
    return out


def stl_triangles(path: Path) -> np.ndarray:
    """A binary STL's triangles, (n,3,3) in its own units."""
    data = path.read_bytes()
    count = int(np.frombuffer(data, "<u4", 1, 80)[0])
    tris = np.frombuffer(data, np.dtype([("normal", "<3f4"), ("v", "<9f4"), ("attr", "<u2")]), count, 84)
    return tris["v"].reshape(-1, 3, 3).astype(float)


@dataclass
class Gripper:
    """One side's gripper: each fixed part's triangles in the gripper link's
    frame by link name, the jaw's in its own link's frame and the joint it
    turns on, and the tool frame in the gripper link's."""
    fixed: dict[str, np.ndarray]
    jaw: np.ndarray
    jaw_origin: np.ndarray
    jaw_axis: np.ndarray
    link_T_tcp: np.ndarray

    def triangles(self, jaw_angle: float) -> np.ndarray:
        """Every part's triangles in the gripper link's frame, the jaw turned
        `jaw_angle` radians on its joint."""
        T = self.jaw_origin.copy()
        T[:3, :3] = T[:3, :3] @ Rotation.from_rotvec(jaw_angle * self.jaw_axis).as_matrix()
        return np.concatenate([*self.fixed.values(), self.jaw @ T[:3, :3].T + T[:3, 3]])


def gripper(urdf: ET.Element, side: str, own: dict[str, Path] | None = None) -> Gripper:
    """`side`'s gripper off the description `urdf`: its gripper link's parts
    and what hangs from it, a mesh named in `own` swapped for that file (the
    handheld's claw and jaw, say)."""
    links = {link.get("name"): link for link in urdf.findall("link")}
    joints: dict[str, list] = {}
    for joint in urdf.findall("joint"):
        joints.setdefault(joint.find("parent").get("link"), []).append(joint)

    def visuals(name: str, link_T: np.ndarray) -> list[np.ndarray]:
        out = []
        for visual in links[name].findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                continue
            path = Path(mesh.get("filename"))
            path = (own or {}).get(path.name, path)
            if not path.is_file():
                continue
            T = link_T @ origin(visual.find("origin"))
            tris = stl_triangles(path) * float(mesh.get("scale", "1 1 1").split()[0])
            out.append(tris @ T[:3, :3].T + T[:3, 3])
        return out

    fixed: dict[str, np.ndarray] = {}
    jaw, jaw_origin, jaw_axis, tool = None, None, None, None
    hanging = [(f"{side}_gripper_link", np.eye(4))]
    while hanging:
        name, link_T = hanging.pop()
        if name == f"{side}_gripper_frame_link":
            tool = link_T
        if name == f"{side}_jaw_link":
            jaw = np.concatenate(visuals(name, np.eye(4)) or [np.empty((0, 3, 3))])
        else:
            found = visuals(name, link_T)
            if found:
                fixed[name] = np.concatenate(found)
        for joint in joints.get(name, []):
            child = joint.find("child").get("link")
            if joint.get("type") == "revolute" and child == f"{side}_jaw_link":
                jaw_origin = link_T @ origin(joint.find("origin"))
                axis = np.array([float(v) for v in joint.find("axis").get("xyz").split()])
                jaw_axis = axis / np.linalg.norm(axis)
                hanging.append((child, jaw_origin))
            else:
                hanging.append((child, link_T @ origin(joint.find("origin"))))
    if tool is None or jaw is None or not fixed:
        raise FileNotFoundError(f"the {side} gripper's tool frame, jaw or meshes are not in the description")
    return Gripper(fixed, jaw, jaw_origin, jaw_axis, tool)
