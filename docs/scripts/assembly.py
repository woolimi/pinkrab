"""How each step of the assembly pages goes on, worked out from the meshes.

The pages' steps (src/data/assembly.json) say which parts each step adds. This
finds, for every step and either side, the way those parts come on - the
direction they can travel the last stretch in without passing through anything
already on - and every screw a servo takes: into its holder through the
holder's holes, and through a part into its horns, found where a part has a
hole over the servo's own. What it finds goes to public/robot/assembly.json,
which the pages draw from. Run it again when the description, the Hermit's
parts or the steps change:

    python docs/scripts/assembly.py

The parts sit as the pages draw them: the arm in the pose the data names, the
Hermit in its gripper link's frame (the studio's hermit.js lays it out the
same way).
"""
from __future__ import annotations

import json
import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

DOCS = Path(__file__).resolve().parents[1]
REPO = DOCS.parent
STEPS = DOCS / "src" / "data" / "assembly.json"
URDF = DOCS / "public" / "robot" / "model.urdf"
HERMIT = DOCS / "public" / "robot" / "hermit.json"
ARM_MESHES = REPO / "src" / "pinkrab_description" / "meshes"
HERMIT_MESHES = REPO / "src" / "pinkrab_umi" / "meshes"
OUT = DOCS / "public" / "robot" / "assembly.json"

# The STS3215 in its own mesh's frame (metres): the horn turns about z
# through HORN_XY; four holes in each horn on a square about it, the top
# horn's centre screw seat and the hole bottoms under both horns; four tapped holes on the top
# face of the case and four on the bottom, of which a holder uses those its
# own holes stand over. Measured off servo.stl.
HORN_XY = (0.0125, 0.0)
HORN_HOLES = [(0.0125 + sx * 0.00495, sy * 0.00495) for sx in (-1, 1) for sy in (-1, 1)]
HORN_CENTRE_TOP = 0.0202
HOLE_DEPTH = 0.0144                    # every tapped hole's bottom, either face
MOUNT_TOP = [(0.0042, s * 0.01025) for s in (-1, 1)] + [(-0.0165, s * 0.01025) for s in (-1, 1)]
MOUNT_BOTTOM = [(0.0042, s * 0.01025) for s in (-1, 1)] + [(-0.0203, s * 0.01025) for s in (-1, 1)]
# A part holds a screw where the ring round the hole is the part's and the
# hole's middle is not: rays as far out as the screw's head reaches, so many
# round. The head rests where the first of them meets the part - on the face,
# or on a counterbore's floor when the counterbore takes the whole head.
RING = {"horn": 0.00225, "mount": 0.00185}
RAYS = 12
# How far out of a hole's bottom a part's face is looked for: past a plate
# and a screw head, short of the far wall of a bracket round the servo.
REACH = 0.014

# A step's parts come on along a line they can travel this far along
# without crossing anything already on; contact at the start is the fit.
TRAVEL = 0.05
CONTACT = 0.0015
VOXEL = 0.0008
SAMPLE = 0.0015


# --- the meshes --------------------------------------------------------------

def read_stl(path: Path) -> np.ndarray:
    """Triangles (n, 3, 3), metres."""
    data = path.read_bytes()
    count = struct.unpack("<I", data[80:84])[0]
    if 84 + count * 50 == len(data):
        rows = np.frombuffer(data[84:84 + count * 50],
                             dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
        return rows["v"].reshape(-1, 3, 3).astype(np.float64)
    text = data.decode()
    points = [list(map(float, line.split()[1:])) for line in text.splitlines() if line.strip().startswith("vertex")]
    return np.array(points, dtype=np.float64).reshape(-1, 3, 3)


def placed(triangles: np.ndarray, T: np.ndarray) -> np.ndarray:
    return triangles @ T[:3, :3].T + T[:3, 3]


def sample(triangles: np.ndarray, spacing: float) -> np.ndarray:
    """Points over the triangles' surfaces, about `spacing` apart."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    longest = np.maximum.reduce([np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1),
                                 np.linalg.norm(a - c, axis=1)])
    out = [triangles.reshape(-1, 3)]
    for k in np.unique(np.ceil(longest / spacing).astype(int)):
        if k < 2:
            continue
        pick = np.ceil(longest / spacing).astype(int) == k
        u, v = np.meshgrid(np.linspace(0, 1, k + 1), np.linspace(0, 1, k + 1))
        keep = (u + v) <= 1
        u, v = u[keep], v[keep]
        A, B, C = a[pick], b[pick], c[pick]
        out.append((A[:, None] + (B - A)[:, None] * u[None, :, None] + (C - A)[:, None] * v[None, :, None]).reshape(-1, 3))
    return np.concatenate(out)


# --- the description ----------------------------------------------------------

def transform(xyz, rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    T = np.eye(4)
    T[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                 [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                 [-sp, cp * sr, cp * cr]]
    T[:3, 3] = xyz
    return T


def origin_of(element) -> np.ndarray:
    o = element.find("origin") if element is not None else None
    if o is None:
        return np.eye(4)
    xyz = [float(v) for v in o.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in o.get("rpy", "0 0 0").split()]
    return transform(xyz, rpy)


def axis_angle(axis, angle) -> np.ndarray:
    x, y, z = np.asarray(axis, float) / np.linalg.norm(axis)
    c, s, t = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    T = np.eye(4)
    T[:3, :3] = [[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                 [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                 [t * x * z - s * y, t * y * z + s * x, t * z * z + c]]
    return T


def links_at(root, pose: dict[str, float]) -> tuple[dict[str, np.ndarray], dict[str, list]]:
    """Every link's pose in the description's frame, and its visuals: (file, link_T_visual)."""
    joints = {}
    for j in root.findall("joint"):
        joints[j.find("child").get("link")] = j
    visuals = {}
    for link in root.findall("link"):
        visuals[link.get("name")] = [
            (v.find("geometry/mesh").get("filename").split("/")[-1], origin_of(v))
            for v in link.findall("visual") if v.find("geometry/mesh") is not None]
    poses = {}

    def pose_of(name):
        if name in poses:
            return poses[name]
        j = joints.get(name)
        if j is None:
            poses[name] = np.eye(4)
            return poses[name]
        T = pose_of(j.find("parent").get("link")) @ origin_of(j)
        if j.get("type") in ("revolute", "continuous") and j.get("name") in pose:
            axis = [float(v) for v in j.find("axis").get("xyz").split()]
            T = T @ axis_angle(axis, pose[j.get("name")])
        poses[name] = T
        return T

    for name in visuals:
        pose_of(name)
    return poses, visuals


# --- one assembly, laid out -------------------------------------------------------

class Part:
    """One mesh where it sits: in the assembly's frame (`T`, its mesh frame),
    and on which link (`link`, that link's pose `link_T`)."""
    def __init__(self, key: str, mesh: np.ndarray, T: np.ndarray, link: str, link_T: np.ndarray,
                 servo: bool = False):
        self.key, self.servo, self.T, self.link, self.link_T = key, servo, T, link, link_T
        self.triangles = placed(mesh, T)
        self.lo, self.hi = self.triangles.reshape(-1, 3).min(0), self.triangles.reshape(-1, 3).max(0)


def arm_parts(root, side: str, pose: dict[str, float]) -> dict[str, Part]:
    poses, visuals = links_at(root, pose)
    parts = {}
    for name, items in visuals.items():
        if not name.startswith(f"{side}_"):
            continue
        for file, T in items:
            stem = file[:-4].removesuffix(f"_{side}")
            key = f"{name[len(side) + 1:]}:{stem}"
            parts[key] = Part(key, read_stl(ARM_MESHES / file), poses[name] @ T, name, poses[name],
                              servo=file.startswith("servo"))
    return parts


def hermit_parts(root, side: str, hermit: dict) -> dict[str, Part]:
    """In the gripper link's frame, as hermit.js lays the Hermit out."""
    poses, visuals = links_at(root, {})
    link = np.linalg.inv(poses[f"{side}_gripper_link"])
    def visual(link_name, file):
        T = dict(visuals[link_name])[file]
        return link @ poses[link_name] @ T
    hand, jaw_link = f"{side}_gripper_link", f"{side}_jaw_link"
    on = {"jaw": (jaw_link, link @ poses[jaw_link])}
    claw = visual(f"{side}_gripper_link", f"claw_{side}.stl")
    jaw = visual(f"{side}_jaw_link", f"jaw_{side}.stl")
    servo = visual(f"{side}_gripper_servo_link", "servo.stl")
    tool = link @ poses[f"{side}_gripper_frame_link"]
    solid = tool @ np.array(hermit[side]["tool_T_solid"])
    layout = {
        "body": (HERMIT_MESHES / f"handheld_body_{side}.stl", claw),
        "servo": (ARM_MESHES / "servo.stl", servo),
        "jaw": (HERMIT_MESHES / f"handheld_jaw_{side}.stl", jaw),
        "camera_bracket": (HERMIT_MESHES / f"camera_bracket_{side}.stl", claw),
        "peg": (HERMIT_MESHES / f"handheld_peg_{side}.stl", claw),
        "solid": (HERMIT_MESHES / f"marker_solid_handheld_{side}.stl", solid),
        "board_mount": (HERMIT_MESHES / f"board_mount_{side}.stl", solid),
        "driver_plate": (HERMIT_MESHES / f"driver_plate_{side}.stl", solid),
    }
    parts = {}
    for key, (path, T) in layout.items():
        name, link_T = on.get(key, (hand, np.eye(4)))
        parts[key] = Part(key, read_stl(path), T, name, link_T, servo=key == "servo")
    return parts


# --- rays ---------------------------------------------------------------------------

def first_hit(origin, direction, parts: list[Part], limit: float):
    """The nearest part along a ray within `limit`, and how far: (part, t) or (None, None)."""
    best = (None, None)
    o, d = np.asarray(origin), np.asarray(direction)
    end = o + d * limit
    lo, hi = np.minimum(o, end) - 0.003, np.maximum(o, end) + 0.003
    for part in parts:
        if np.any(part.hi < lo) or np.any(part.lo > hi):
            continue
        tri = part.triangles
        near = np.all((tri.max(1) >= lo) & (tri.min(1) <= hi), axis=1)
        tri = tri[near]
        if not len(tri):
            continue
        v0, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        p = np.cross(d, e2)
        det = np.einsum("ij,ij->i", e1, p)
        ok = np.abs(det) > 1e-14
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = o - v0
        u = np.einsum("ij,ij->i", s, p) * inv
        q = np.cross(s, e1)
        v = (q @ d) * inv
        t = np.einsum("ij,ij->i", e2, q) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-7) & (t <= limit)
        if hit.any():
            tmin = t[hit].min()
            if best[1] is None or tmin < best[1]:
                best = (part, float(tmin))
    return best


def neighbours(root) -> dict[str, set[str]]:
    """For each servo's link, the links a screw of it may go through: the
    holder it is fixed to and the link its horns turn."""
    parent, driven = {}, {}
    for j in root.findall("joint"):
        p, c = j.find("parent").get("link"), j.find("child").get("link")
        parent[c] = p
        if j.get("type") in ("revolute", "continuous"):
            driven[p] = c
    return {c: {p, driven.get(p, p)} for c, p in parent.items() if c.endswith("servo_link")}


def screws_of(servo: Part, parts: dict[str, Part], near: set[str] | None = None) -> list[dict]:
    """Every screw this servo takes: the top horn's own, and one through a part
    wherever the part has a hole over a horn's or the case's."""
    R, t = servo.T[:3, :3], servo.T[:3, 3]
    at = lambda x, y, z: R @ np.array([x, y, z]) + t
    up = R @ np.array([0.0, 0.0, 1.0])

    def local(part: Part, seat, axis):
        """The seat and axis in the frame of the link the screw goes with."""
        inv = np.linalg.inv(part.link_T)
        return {"link": part.link, "seat": (inv[:3, :3] @ seat + inv[:3, 3]).tolist(),
                "axis": (inv[:3, :3] @ axis).tolist(), "seat_world": np.asarray(seat).tolist()}

    out = [{"servo": servo.key, "part": servo.key, "kind": "centre", "length": 0.006,
            **local(servo, at(*HORN_XY, HORN_CENTRE_TOP), up)}]
    has_bottom_horn = servo.key.endswith(":servo") or servo.key == "servo"
    # Servos hold no screws of another's and stand in the way of none that
    # go in before them: a part's own faces decide.
    others = [p for p in parts.values() if not p.servo and (near is None or p.link in near)]

    def through(xy, depth, face, kind, centre_only=False):
        """The part a screw at this hole goes through, its seat and length - or None."""
        axis = up * face
        bottom = at(xy[0], xy[1], depth)
        start = bottom + axis * REACH
        middle, _ = first_hit(start, -axis, others, REACH - 0.001)
        if middle is not None:
            return None
        if centre_only:
            return True
        side = np.cross(axis, [1.0, 0, 0] if abs(axis[0]) < 0.9 else [0, 1.0, 0])
        side /= np.linalg.norm(side)
        other = np.cross(axis, side)
        hits = []
        for k in range(RAYS):
            a = 2 * math.pi * k / RAYS
            hits.append(first_hit(start + RING[kind] * (math.cos(a) * side + math.sin(a) * other), -axis,
                                  others, REACH))
        found = [p for p, _ in hits if p is not None]
        if not found:
            return None
        holder = max(set(found), key=found.count)
        if found.count(holder) < RAYS - 4:
            return None
        seat = start - axis * float(min(d for p, d in hits if p is holder))
        return holder, seat, axis, float(np.dot(seat - bottom, axis))

    # A horn turns: the part on it may stand at any angle of it, so its holes
    # are looked for round the horn, the angle that finds most of them taken.
    for face, depth in [(1, HOLE_DEPTH)] + ([(-1, -HOLE_DEPTH)] if has_bottom_horn else []):
        best = []
        for degrees in range(0, 90, 2):
            a = math.radians(degrees)
            ring = [(HORN_XY[0] + math.cos(a) * dx - math.sin(a) * dy, HORN_XY[1] + math.sin(a) * dx + math.cos(a) * dy)
                    for dx, dy in ((x - HORN_XY[0], y - HORN_XY[1]) for x, y in HORN_HOLES)]
            if sum(bool(through(xy, depth, face, "horn", centre_only=True)) for xy in ring) <= len(best):
                continue
            found = [f for f in (through(xy, depth, face, "horn") for xy in ring) if f]
            if len(found) > len(best):
                best = found
        for holder, seat, axis, length in best:
            # A horn's screw goes with the part the horn turns.
            out.append({"servo": servo.key, "part": holder.key, "kind": "horn", "length": length,
                        **local(holder, seat, axis)})
    for xy, depth, face in [(xy, HOLE_DEPTH, 1) for xy in MOUNT_TOP] + [(xy, -HOLE_DEPTH, -1) for xy in MOUNT_BOTTOM]:
        found = through(xy, depth, face, "mount")
        if found:
            holder, seat, axis, length = found
            out.append({"servo": servo.key, "part": holder.key, "kind": "mount", "length": length,
                        **local(servo, seat, axis)})
    return out


# The base's bolts: through-holes this wide across the base's plate, square to
# the board, found by slicing the plate; the bolt's head sits flush in the
# countersink of the face away from the board.
BOLT_HOLE = (0.0019, 0.0026)
BOLT_LENGTH = 0.02


def base_bolts(base: Part) -> list[dict]:
    """The base's bolts to the board, in the base link's frame."""
    import manifold3d as mf

    tri = np.linalg.inv(base.T) @ np.c_[base.triangles.reshape(-1, 3), np.ones(len(base.triangles) * 3)].T
    local = tri[:3].T.reshape(-1, 3, 3)               # back in the mesh's own frame
    verts, index = np.unique(np.round(local.reshape(-1, 3), 7), axis=0, return_inverse=True)
    solid = mf.Manifold(mf.Mesh(vert_properties=verts.astype(np.float32),
                                tri_verts=index.reshape(-1, 3).astype(np.uint32)))
    lo, hi = solid.bounding_box()[2], solid.bounding_box()[5]
    holes = []
    for poly in solid.slice(lo + 0.001).to_polygons():
        q = np.array(poly)
        x, y = q[:, 0], q[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        perimeter = np.sum(np.linalg.norm(q - np.roll(q, 1, axis=0), axis=1))
        r = math.sqrt(area / math.pi) if area > 0 else 0
        if area > 0 and 4 * math.pi * area / perimeter ** 2 > 0.9 and BOLT_HOLE[0] < r < BOLT_HOLE[1]:
            holes.append((float(x.mean()), float(y.mean())))
    out = []
    R, t = base.T[:3, :3], base.T[:3, 3]
    out_of_board = R @ np.array([0.0, 0.0, 1.0])
    for x, y in holes:
        # the face the head sits in: the first of the plate a ray down the hole's side meets
        start = R @ np.array([x + 0.0045, y, hi + 0.01]) + t
        _, d = first_hit(start, -out_of_board, [base], hi - lo + 0.02)
        if d is None:
            continue
        seat = R @ np.array([x, y, hi + 0.01]) + t - out_of_board * d
        inv = np.linalg.inv(base.link_T)
        out.append({"servo": base.key, "part": base.key, "kind": "bolt", "length": BOLT_LENGTH,
                    "link": base.link, "seat": (inv[:3, :3] @ seat + inv[:3, 3]).tolist(),
                    "axis": (inv[:3, :3] @ out_of_board).tolist(), "seat_world": seat.tolist()})
    return out


# A screw that joins two printed parts shows as a counterbored hole: a
# clearance hole, and a wider bore at its end for the head. Found by slicing
# each part across each of its mesh's axes; sized by the clearance hole.
JOIN_HOLE = (0.0009, 0.0019)
JOIN_BORE = 0.0005                    # a bore is at least this much wider
SLICE = 0.0008
JOIN_SIZES = [(0.0012, "M2×6", 0.006), (0.00155, "M2.5", 0.006), (0.0019, "M3×10", 0.010)]
# A step that names a part's nuts (`nuts`) puts in the nuts its M3 bolts take,
# in the pocket behind the wall the bolt comes through: found off the bolt by
# this much, inside the nut's flats and outside the bolt's hole.
NUT_BESIDE = 0.0023
NUT_H = 0.0024
# A step that names a board (`board`) screws it onto its part before the part
# goes on: a board on the part's far face from the screws' heads, found off
# each screw as a nut's pocket is, as big as the part less this much round
# it, this thick; its servo sockets on the edge `servo` points to (an axis
# of the link's frame, or of the mesh frame of the part it names `of`), its
# USB-C as the step names it for this side - on the side toward the middle of
# the assembly (`in`) or away from it (`out`), its power on the other, or with
# its power on the edge across from the sockets (`back`); with `xiao`, the
# USB-C is a XIAO module's, on headers.
BOARD_BESIDE = 0.0019
BOARD_INSET = 0.0005
BOARD_T = 0.0016
NEAR_SCREW = 0.0015                   # a hole this close to a screw found already is that screw's
# A part's holes on a horn's bolt circle let a driver through to the horn's
# screws; they hold none of their own.
HORN_CIRCLE = (0.0055, 0.0085)
# A screw joins its part to another: something else stands this close to its head.
JOINS_TO = 0.015


def solid_of(part: Part):
    import manifold3d as mf

    inv = np.linalg.inv(part.T)
    local = part.triangles @ inv[:3, :3].T + inv[:3, 3]
    verts, index = np.unique(np.round(local.reshape(-1, 3), 7), axis=0, return_inverse=True)
    return mf.Manifold(mf.Mesh(vert_properties=verts.astype(np.float32),
                               tri_verts=index.reshape(-1, 3).astype(np.uint32)))


# Each mesh axis turned onto z for slicing, and the way back: the slice's
# (u, v, height) to the mesh's (x, y, z).
SLICED = {0: ([0, 90, 0], lambda u, v, h: (-h, v, u), (-1.0, 0.0, 0.0)),
          1: ([-90, 0, 0], lambda u, v, h: (u, -h, v), (0.0, -1.0, 0.0)),
          2: ([0, 0, 0], lambda u, v, h: (u, v, h), (0.0, 0.0, 1.0))}


def runs(part: Part) -> list[tuple]:
    """Round holes through the part, each as a run of slices: (axis, u, v, r, h0, h1)."""
    solid = solid_of(part)
    out = []
    for axis, (turn, _, _) in SLICED.items():
        s = solid.rotate(turn)
        bb = s.bounding_box()
        seen = {}
        for h in np.arange(bb[2] + SLICE / 2, bb[5], SLICE):
            for poly in s.slice(h).to_polygons():
                q = np.array(poly)
                if len(q) < 6:
                    continue
                x, y = q[:, 0], q[:, 1]
                area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                perimeter = np.sum(np.linalg.norm(q - np.roll(q, 1, axis=0), axis=1))
                if area <= 0 or 4 * math.pi * area / perimeter ** 2 < 0.9:
                    continue
                r = math.sqrt(area / math.pi)
                if JOIN_HOLE[0] < r < 0.0035:
                    key = (round(float(x.mean()), 4), round(float(y.mean()), 4), round(r, 4))
                    seen.setdefault(key, []).append(float(h))
        for (u, v, r), hs in seen.items():
            hs.sort()
            start = hs[0]
            for a, b in zip(hs, hs[1:] + [None]):
                if b is None or b - a > SLICE * 1.6:
                    out.append((axis, u, v, r, start - SLICE / 2, a + SLICE / 2))
                    if b is not None:
                        start = b
    return out


def joins(part: Part, found: list[dict], servos: list[Part], beside) -> list[dict]:
    """The screws through this part's counterbored holes that no screw found
    already stands in, off any horn's bolt circle, and with something else
    near to join the part to (`beside(link, seat)`, the page's pose): the
    head on the bore's floor, the axis out of it."""
    holes = runs(part)
    horns = [(s.T[:3, :3] @ np.array([*HORN_XY, 0.0]) + s.T[:3, 3], s.T[:3, 2]) for s in servos]
    R, t = part.T[:3, :3], part.T[:3, 3]
    inv = np.linalg.inv(part.link_T)
    seats = [np.array(f["seat_world"]) for f in found]
    out = []
    for axis, u, v, r, h0, h1 in holes:
        if not JOIN_HOLE[0] < r < JOIN_HOLE[1]:
            continue
        _, back, along = SLICED[axis]
        for bore in holes:
            if (bore[0] != axis or abs(bore[1] - u) > 0.0004 or abs(bore[2] - v) > 0.0004
                    or bore[3] < r + JOIN_BORE):
                continue
            # the bore sits at one end of the hole: the floor is where they meet
            if abs(bore[4] - h1) < SLICE * 1.6:
                floor, out_h = h1, 1.0
            elif abs(bore[5] - h0) < SLICE * 1.6:
                floor, out_h = h0, -1.0
            else:
                continue
            seat = R @ np.array(back(u, v, floor)) + t
            out_axis = R @ (np.array(along) * out_h)
            if any(np.linalg.norm(np.cross(q - seat, out_axis)) < NEAR_SCREW
                   and abs(np.dot(q - seat, out_axis)) < 0.03 for q in seats):
                continue
            if any(abs(np.dot(out_axis, a)) > 0.99
                   and HORN_CIRCLE[0] < np.linalg.norm(np.cross(seat - c, a)) < HORN_CIRCLE[1] for c, a in horns):
                continue
            local_seat = inv[:3, :3] @ seat + inv[:3, 3]
            if not beside(part, local_seat):
                continue
            size = next(name for top, name, _ in JOIN_SIZES if r < top)
            length = next(length for top, _, length in JOIN_SIZES if r < top)
            out.append({"servo": part.key, "part": part.key, "kind": "join", "size": size, "length": length,
                        "link": part.link, "seat": (inv[:3, :3] @ seat + inv[:3, 3]).tolist(),
                        "axis": (inv[:3, :3] @ out_axis).tolist(), "seat_world": seat.tolist()})
            seats.append(seat)
    return out


# --- how a step's parts come on ------------------------------------------------

def approach(moving: list[Part], fixed: list[Part], frames: list[np.ndarray]) -> list[float] | None:
    """The way the parts come on: of each frame's six axes, the one they can
    travel TRAVEL along crossing least of what is on, and of those the one
    leading most away from it."""
    lo = np.min([p.lo for p in moving], 0) - TRAVEL
    hi = np.max([p.hi for p in moving], 0) + TRAVEL
    near = []
    for p in fixed:
        if np.any(p.hi < lo) or np.any(p.lo > hi):
            continue
        tri = p.triangles[np.all((p.triangles.max(1) >= lo) & (p.triangles.min(1) <= hi), axis=1)]
        if len(tri):
            near.append(tri)
    points = sample(np.concatenate([p.triangles for p in moving]), SAMPLE)
    middle = points.mean(0)
    if not near:
        return None                    # nothing on yet: the page brings it in its own way
    obstacle = np.concatenate(near)
    cells = np.unique(np.floor(sample(obstacle, VOXEL * 0.8) / VOXEL).astype(np.int64), axis=0)
    span = cells.max(0) - cells.min(0) + 3
    base = cells.min(0) - 1
    def key(c):
        c = c - base
        return (c[:, 0] * span[1] + c[:, 1]) * span[2] + c[:, 2]
    solid = np.sort(key(cells))
    away = middle - sample(obstacle, 0.004).mean(0)
    candidates = []
    for F in frames:
        for k in range(3):
            for s in (1, -1):
                d = F[:3, k] * s
                if all(np.dot(d, c) < 0.999 for c in candidates):
                    candidates.append(d)
    best = None
    for d in candidates:
        blocked = np.zeros(len(points), dtype=bool)
        for travel in np.arange(CONTACT, TRAVEL, VOXEL):
            c = np.floor((points + d * travel) / VOXEL).astype(np.int64)
            inside = np.all((c >= base) & (c < base + span), axis=1)
            k = key(c[inside])
            hit = np.zeros(len(points), dtype=bool)
            pos = np.searchsorted(solid, k)
            pos = np.minimum(pos, len(solid) - 1)
            hit[np.flatnonzero(inside)] = solid[pos] == k
            blocked |= hit
        score = (int(blocked.sum()), -float(np.dot(d, away)))
        if best is None or score < best[0]:
            best = (score, d)
    return best[1].tolist()


def lay_out(steps: list[dict], parts: dict[str, Part], frame_of, zero: dict[str, Part], which: str,
            near: dict[str, set[str]] | None = None) -> dict:
    """Each step's way in, with `parts` where the page draws them, and the
    screws it drives, found with `zero` (every joint at zero, the horns'
    holes where the servo's mesh has them), for one side (`which`)."""
    step_of = {}
    for i, step in enumerate(steps):
        for key in step["parts"]:
            step_of[key] = i
    ways = []
    for i, step in enumerate(steps):
        moving = [parts[k] for k in step["parts"] if k in parts]
        fixed = [parts[k] for k, j in step_of.items() if j < i and k in parts]
        if not moving:
            ways.append(None)
            continue
        ways.append(approach(moving, fixed, frame_of(moving)))
    screws = []
    if "base_link:base" in zero and "base_link:base" in step_of:
        for bolt in base_bolts(zero["base_link:base"]):
            bolt["step"] = step_of["base_link:base"]
            screws.append(bolt)
    for part in zero.values():
        if not part.servo or part.key not in step_of:
            continue
        for screw in screws_of(part, {k: p for k, p in zero.items() if k in step_of},
                               near.get(part.link) if near else None):
            screw["step"] = max(step_of[screw["servo"]], step_of.get(screw["part"], 0))
            screws.append(screw)
    servos = [p for p in zero.values() if p.servo and p.key in step_of]
    others = {k: p for k, p in parts.items() if k in step_of}

    def beside(part, local_seat):
        """Whether, as the page draws them, another part stands near this seat."""
        here = parts[part.key].link_T
        seat = here[:3, :3] @ local_seat + here[:3, 3]
        for key, other in others.items():
            if key == part.key or np.any(other.lo - JOINS_TO > seat) or np.any(other.hi + JOINS_TO < seat):
                continue
            if np.min(np.linalg.norm(other.triangles.reshape(-1, 3) - seat, axis=1)) < JOINS_TO:
                return True
        return False

    for part in zero.values():
        if part.servo or part.key not in step_of:
            continue
        for screw in joins(part, screws, servos, beside):
            # A screw that goes on into a servo - a case's front screws -
            # goes in with that servo.
            here = parts[part.key].link_T
            seat = here[:3, :3] @ np.array(screw["seat"]) + here[:3, 3]
            axis = here[:3, :3] @ np.array(screw["axis"])
            into, _ = first_hit(seat - axis * 0.0003, -axis, [parts[s.key] for s in servos], screw["length"] + 0.003)
            screw["step"] = max(step_of[part.key], step_of[into.key] if into else 0)
            screws.append(screw)
    for i, step in enumerate(steps):
        bolted = step.get("nuts")
        if not bolted:
            continue
        for bolt in [s for s in screws if s["kind"] == "join" and s["part"] == bolted and s["size"].startswith("M3")]:
            T = zero[bolted].link_T
            way = -(T[:3, :3] @ np.array(bolt["axis"]))
            seat = T[:3, :3] @ np.array(bolt["seat"]) + T[:3, 3]
            side = np.cross(way, [1.0, 0, 0] if abs(way[0]) < 0.9 else [0, 1.0, 0])
            side /= np.linalg.norm(side)
            other = np.cross(way, side)
            candidates = [p for k, p in zero.items() if not p.servo and k != bolted and k in step_of]
            # beside the bolt: into the wall it comes through, and out of
            # that wall into the pocket the nut fills
            met = []
            for k in range(RAYS):
                a = 2 * math.pi * k / RAYS
                o = seat + NUT_BESIDE * (math.cos(a) * side + math.sin(a) * other)
                wall, t0 = first_hit(o, way, candidates, 0.015)
                if wall is None:
                    continue
                _, t1 = first_hit(o + way * (t0 + 0.0002), way, [wall], 0.015)
                if t1 is not None:
                    met.append((wall, t0 + 0.0002 + t1))
            if not met:
                continue
            holder = max({w for w, _ in met}, key=lambda w: sum(1 for m, _ in met if m is w))
            floor = float(np.median([t for w, t in met if w is holder]))
            centre = seat + way * (floor + NUT_H / 2)
            inv = np.linalg.inv(holder.link_T)
            # it goes in from inside the part, against the bolt's way
            screws.append({"servo": holder.key, "part": holder.key, "kind": "nut", "size": "M3 육각 너트",
                           "length": NUT_H, "link": holder.link, "step": i,
                           "seat": (inv[:3, :3] @ centre + inv[:3, 3]).tolist(),
                           "axis": (inv[:3, :3] @ way).tolist()})
    boards = []
    for i, step in enumerate(steps):
        held = [s for s in screws if s["kind"] == "join" and s["step"] == i]
        if not step.get("board") or not held:
            continue
        part = zero[held[0]["part"]]
        T = part.link_T
        way = -(T[:3, :3] @ np.array(held[0]["axis"]))
        seats = [T[:3, :3] @ np.array(s["seat"]) + T[:3, 3] for s in held]
        side = np.cross(way, [1.0, 0, 0] if abs(way[0]) < 0.9 else [0, 1.0, 0])
        side /= np.linalg.norm(side)
        other = np.cross(way, side)
        # beside each screw, from its head through the part to the far face
        through = []
        for seat in seats:
            for k in range(RAYS):
                a = 2 * math.pi * k / RAYS
                o = seat - way * 0.0003 + BOARD_BESIDE * (math.cos(a) * side + math.sin(a) * other)
                _, t0 = first_hit(o, way, [part], 0.01)
                if t0 is None:
                    continue
                _, t1 = first_hit(o + way * (t0 + 0.0002), way, [part], 0.03)
                if t1 is not None:
                    through.append(np.dot(seat, way) + t0 + 0.0002 + t1 - 0.0003)
        if not through:
            continue
        # on the face farthest out where screws come through: a mount's face
        # may step, and a board is flat
        plane = max(through)
        # up the board: away from its servo sockets, square to it; across it, the third way
        sockets = step["board"]["servo"]
        frame = zero[sockets["of"]].T if "of" in sockets else T
        v = -(frame[:3, :3] @ np.array(sockets["axis"], dtype=float))
        v -= way * np.dot(v, way)
        v /= np.linalg.norm(v)
        u = np.cross(v, way)
        corners = part.triangles.reshape(-1, 3)
        pu, pv = corners @ u, corners @ v
        middle = (u * (pu.min() + pu.max()) / 2 + v * (pv.min() + pv.max()) / 2 + way * (plane + BOARD_T / 2))
        # toward the middle of the assembly, across the board
        inward = 1.0 if np.dot(-middle, u) > 0 else -1.0
        usb = step["board"][which]["usb"]
        inv = np.linalg.inv(part.link_T)
        boards.append({"step": i, "link": part.link, "centre": (inv[:3, :3] @ middle + inv[:3, 3]).tolist(),
                       "u": (inv[:3, :3] @ u).tolist(), "v": (inv[:3, :3] @ v).tolist(),
                       "size": [float(pu.max() - pu.min()) - 2 * BOARD_INSET,
                                float(pv.max() - pv.min()) - 2 * BOARD_INSET, BOARD_T],
                       "usb": {"in": inward, "out": -inward, "back": 0.0}[usb],
                       "xiao": bool(step["board"].get("xiao"))})
    for screw in screws:
        screw.pop("seat_world", None)
    return {"approach": ways, "screws": screws, "boards": boards}


def main() -> None:
    data = json.loads(STEPS.read_text())
    root = ET.parse(URDF).getroot()
    hermit = json.loads(HERMIT.read_text())
    # The pose the arm is drawn in, for the page to draw it in.
    out = {"pose": data["pose"], "arm": {}, "hermit": {}}
    for side, sign in (("left", 1), ("right", -1)):
        pose = {f"{side}_{j}": sign * math.radians(d) for j, d in data["pose"].items()}
        frames = lambda moving: [m.T for m in moving[:1]]
        out["arm"][side] = lay_out(data["arm"], arm_parts(root, side, pose), frames, arm_parts(root, side, {}),
                                   side, neighbours(root))
        hand = hermit_parts(root, side, hermit)
        out["hermit"][side] = lay_out(data["hermit"], hand, frames, hand, side)
    for kind in ("arm", "hermit"):
        for side, laid in out[kind].items():
            counts = {}
            for s in laid["screws"]:
                counts[(s["step"], s["kind"])] = counts.get((s["step"], s["kind"]), 0) + 1
            print(kind, side, len(laid["screws"]), "screws",
                  {f"{data[kind][st]['title']}/{k}": n for (st, k), n in sorted(counts.items())})
    rounded = json.loads(json.dumps(out), parse_float=lambda v: round(float(v), 6))
    OUT.write_text(json.dumps(rounded, separators=(",", ":")) + "\n")
    print("wrote", OUT.relative_to(REPO))


if __name__ == "__main__":
    main()
