"""The marker solid: where its faces are, and how to print and print it.

One file makes both halves of the solid because they have to agree. The sheet says
where the ink goes, the STL says where the paper sits, and `solid_object_points`
says where the solver should expect to find a corner - three statements of one
geometry. Split across files they drift, and a drifted geometry does not fail: it
returns a pose that is wrong by however far they drifted.

The solid frame has its origin at the centre of the solid, +Z out of the face
opposite the stalk. Each marker face gets a frame with X right, Y up and Z along
the outward normal, which is the frame ArUco reports a marker in, so a face's
transform into the solid is the whole of what pnp needs to fuse five faces into
one pose.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import cv2
import numpy as np

from .config import SETTINGS, SHAPE_INRADIUS_PER_HALF_FACE, Settings

MM = 1e-3  # the sheet and the STL are in millimetres, everything solved is in metres
PT_PER_MM = 72.0 / 25.4


def dictionary() -> cv2.aruco.Dictionary:
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, SETTINGS.aruco_dict))


# --- geometry ------------------------------------------------------------

# Each face's frame, as the three axes (X right, Y up, Z outward normal) it puts
# on the solid. The four side faces all take Y up so that a solid held the right way
# up has every side marker the right way up, which is what makes the printed
# corner mark a check somebody can actually do.
def face_label(face: str) -> str:
    """'z_plus' -> '+Z', 'xp_ym' -> '+X-Y' - the face as the sheet writes it."""
    if face.endswith(("_plus", "_minus")):
        axis, sign = face.split("_")
        return ("+" if sign == "plus" else "-") + axis.upper()
    return "".join(("+" if part[1] == "p" else "-") + part[0].upper() for part in face.split("_"))


def face_normal(name: str) -> np.ndarray:
    """Outward unit normal of a face from its name: 'z_plus', 'xp_ym', ..."""
    if name.endswith(("_plus", "_minus")):
        axis, sign = name.split("_")
        n = np.zeros(3)
        n["xyz".index(axis)] = 1.0 if sign == "plus" else -1.0
        return n
    n = np.zeros(3)
    for part in name.split("_"):
        n["xyz".index(part[0])] = 1.0 if part[1] == "p" else -1.0
    return n / np.linalg.norm(n)


def face_axes(cfg: Settings = SETTINGS) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """face name -> (x, y, z) unit axes of the face frame, in the solid's frame.

    z is the outward normal. y - "up", where the printed bar points - is the
    solid's +Z laid flat onto the face, so every side face reads the same way up
    and a glued patch has one right way round; the two faces whose normal IS
    +-Z have no such direction and take +-Y instead. x completes a right-handed
    frame.
    """
    out = {}
    for name in tuple(cfg.marker_faces) + tuple(cfg.socket_faces):
        z = face_normal(name)
        up = np.array([0.0, 0.0, 1.0]) - z * z[2]
        if np.linalg.norm(up) < 1e-9:
            up = np.array([0.0, np.sign(z[2]), 0.0])   # +Y on top, -Y underneath
        y = up / np.linalg.norm(up)
        x = np.cross(y, z)
        out[name] = (x, y, z)
    return out


def face_poses(cfg: Settings = SETTINGS) -> dict[str, np.ndarray]:
    """face name -> 4x4 solid_T_face, metres.

    The translation is the solid's inradius along the outward normal, so the
    face frame sits on the printed paper rather than at the centre of the block.
    """
    out = cfg.inradius_mm * MM
    poses = {}
    for name, (ax, ay, az) in face_axes(cfg).items():
        pose = np.eye(4)
        pose[:3, :3] = np.column_stack([ax, ay, az])
        pose[:3, 3] = az * out
        poses[name] = pose
    return poses


def marker_corners_face(cfg: Settings = SETTINGS) -> np.ndarray:
    """The four black-square corners in a face's own frame, metres.

    ArUco's order: top-left, top-right, bottom-right, bottom-left, seen from in
    front of the marker. The detector returns image corners in this order, so the
    object points have to be listed in it too or the pose comes back rotated.
    """
    h = cfg.marker_mm * MM / 2.0
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]], dtype=np.float64)


def solid_object_points(cfg: Settings = SETTINGS) -> dict[int, np.ndarray]:
    """marker id -> (4,3) corner positions in the solid frame, metres."""
    corners = marker_corners_face(cfg)
    poses = face_poses(cfg)
    out = {}
    for name, marker_id in cfg.face_ids.items():
        pose = poses[name]
        out[marker_id] = (corners @ pose[:3, :3].T) + pose[:3, 3]
    return out


def marker_modules(marker_id: int) -> np.ndarray:
    """The marker as a boolean module grid, black True, border included.

    Read back out of OpenCV rather than reimplemented: the bit layout of a
    dictionary is OpenCV's business, and a second implementation of it is a second
    thing that can disagree with the detector.
    """
    side = 6  # 4x4 payload plus the one-module black border
    img = cv2.aruco.generateImageMarker(dictionary(), marker_id, side * 10)
    return img.reshape(side, 10, side, 10)[:, 5, :, 5] < 128


# --- the printed sheet ---------------------------------------------------

PER_PAGE_ROWS = 4          # four rows of 34 mm patches clear the ruler on A4; five would not


def _patch_origins(cfg: Settings = SETTINGS) -> list[tuple[str, float, float, int]]:
    """(face name, left mm, bottom mm, page) for each patch.

    Four rows a page and as many columns as fit inside a 15 mm margin, the ruler
    under them: a hand solid's sixteen 28 mm patches go four by four on one A4
    page, the desk solid's thirteen the same way. Nothing here is measured by
    the pipeline - the patch outline is a cutting guide - so the layout only
    has to fit and be readable.
    """
    page_w, page_h = cfg.page_mm
    patch = cfg.patch_mm
    gap = 14.0
    cols = max(n for n in (1, 2, 3, 4) if n * patch + (n - 1) * gap <= page_w - 30.0)
    col_pitch = patch + gap
    row_pitch = patch + gap
    left = (page_w - ((cols - 1) * col_pitch + patch)) / 2.0
    top = page_h - 44.0
    per_page = cols * PER_PAGE_ROWS
    out = []
    for i, name in enumerate(cfg.marker_faces):
        page, slot = divmod(i, per_page)
        row, col = divmod(slot, cols)
        out.append((name, left + col * col_pitch, top - row * row_pitch - patch, page))
    return out


def page_count(cfg: Settings = SETTINGS) -> int:
    return max(o[3] for o in _patch_origins(cfg)) + 1


def _patch_shapes(name: str, x0: float, y0: float, cfg: Settings = SETTINGS) -> tuple[list, list, list]:
    """Everything on one patch as (filled rects, stroked rects, texts), mm.

    Shared by the PDF and the PNG so the two cannot disagree about where a module
    lands. Rects are (x, y, w, h) with the origin at the bottom-left of the page.
    """
    marker_id = cfg.face_ids[name]
    modules = marker_modules(marker_id)
    n = modules.shape[0]
    step = cfg.marker_mm / n
    mx = x0 + (cfg.patch_mm - cfg.marker_mm) / 2.0
    my = y0 + (cfg.patch_mm - cfg.marker_mm) / 2.0

    filled = []
    for row in range(n):
        for col in range(n):
            if modules[row, col]:
                # Module rows run top-down in the image and the page runs
                # bottom-up, so the row index is flipped here and nowhere else.
                filled.append((mx + col * step, my + (n - 1 - row) * step, step, step))

    # Which way up this face goes, as a bar at the top edge of the patch. The
    # marker's own frame is what comes back from PnP, so a patch glued a quarter
    # turn round is a pose wrong by a right angle that nothing downstream notices.
    # It stays in the outermost up_mark_mm so the marker keeps clear white round it.
    m = cfg.up_mark_mm
    filled.append((x0 + cfg.patch_mm / 4, y0 + cfg.patch_mm - m, cfg.patch_mm / 2, m / 2))

    stroked = [(x0, y0, cfg.patch_mm, cfg.patch_mm)]
    # Which way "up" points on the solid: +Z for a side face, +Y for the top one -
    # the top face has no side to be up.
    up = "+Y" if name == "z_plus" else "+Z"
    # Two short lines rather than one long one. A caption wider than the column
    # runs into the patch beside it, and what it ran into was the id of the face
    # somebody was about to glue on. The part that is the same on all five - cut
    # on the line, glue it bar up - is said once, in the heading.
    # The solid's name is in the heading; under a patch there is room for the id
    # and the face, and a second line for which way the bar goes.
    texts = [(x0, y0 - 4.6, 3.0, f"id {marker_id}   {name}"),
             (x0, y0 - 8.8, 3.0, f"bar toward {up}")]
    return filled, stroked, texts


def _sheet_shapes(cfg: Settings = SETTINGS, page: int = 0) -> tuple[list, list, list]:
    """One page: its patches, the ruler, and the heading."""
    page_w, page_h = cfg.page_mm
    filled: list = []
    stroked: list = []
    texts: list = []
    for name, x0, y0, on_page in _patch_origins(cfg):
        if on_page != page:
            continue
        f, s, t = _patch_shapes(name, x0, y0, cfg)
        filled += f
        stroked += s
        texts += t

    ids = sorted(cfg.face_ids[n] for n in cfg.marker_faces)
    pages = page_count(cfg)
    paged = f" - page {page + 1} of {pages}" if pages > 1 else ""
    texts.append((20.0, page_h - 20.0, 4.2,
                  f"Pinkrab UMI marker {cfg.shape} - {cfg.solid} - ids {ids[0]}-{ids[-1]}{paged}"))
    texts.append((20.0, page_h - 26.5, 3.2,
                  (f"{cfg.aruco_dict}, {cfg.marker_mm:g} mm markers on "
                   f"{cfg.face_size_mm:g} mm faces.")))
    texts.append((20.0, page_h - 31.0, 3.2,
                  "Print at 100% - no scaling, no fit-to-page. Check the ruler below first."))
    texts.append((20.0, page_h - 35.5, 3.2,
                  ("Cut each patch on its line and glue it bar up, with the bar "
                   "toward the face named under it.")))

    # The ruler: a bar with a tick every 10 mm. If it does not measure what it
    # says, the print scaled, and every marker on the page scaled with it.
    ruler_y = 30.0
    ruler_x = (page_w - cfg.ruler_mm) / 2.0
    filled.append((ruler_x, ruler_y, cfg.ruler_mm, 0.4))
    for i in range(int(cfg.ruler_mm // 10) + 1):
        tall = 5.0 if i % 5 == 0 else 3.0
        filled.append((ruler_x + i * 10.0 - 0.2, ruler_y, 0.4, tall))
    texts.append((ruler_x, ruler_y - 5.5, 3.2,
                  f"This bar is exactly {cfg.ruler_mm:g} mm. Measure it."))
    return filled, stroked, texts


def write_sheet_pdf(path: Path, cfg: Settings = SETTINGS) -> Path:
    """The solid's sheet as vector PDF - a page or two, exact by construction."""
    return write_pdf(path, cfg.page_mm, [_sheet_shapes(cfg, page) for page in range(page_count(cfg))])


def write_pdf(path: Path, page_mm: tuple[float, float], pages: list[tuple[list, list, list]]) -> Path:
    """Pages of (filled rects, stroked rects, texts) in mm, as a vector PDF.

    Written by hand rather than through a PDF library because a rectangle and a
    line of Helvetica is the whole of what these pages are, and because a vector
    page has no resolution to be printed at wrong: a module is 6 mm because the
    file says 6 mm, not because 600 dpi rounded to it.
    """
    page_w, page_h = page_mm

    def content(shapes: tuple[list, list, list]) -> bytes:
        filled, stroked, texts = shapes
        ops = ["0 g", "0.2 w"]
        # One path, filled once: rectangles that touch are one shape to the
        # rasteriser, so no hairline shows where they meet.
        for x, y, w, h in filled:
            ops.append(f"{x * PT_PER_MM:.4f} {y * PT_PER_MM:.4f} "
                       f"{w * PT_PER_MM:.4f} {h * PT_PER_MM:.4f} re")
        if filled:
            ops.append("f")
        for x, y, w, h in stroked:
            ops.append(f"{x * PT_PER_MM:.4f} {y * PT_PER_MM:.4f} "
                       f"{w * PT_PER_MM:.4f} {h * PT_PER_MM:.4f} re S")
        for x, y, size, text in texts:
            escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
            ops.append(f"BT /F1 {size * PT_PER_MM:.2f} Tf "
                       f"{x * PT_PER_MM:.4f} {y * PT_PER_MM:.4f} Td ({escaped}) Tj ET")
        return "\n".join(ops).encode("ascii")

    # Objects: 1 catalog, 2 pages, 3 font, then a (page, content) pair per page.
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"",   # the page tree, filled in once the kids' numbers are known
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    kids = []
    for shapes in pages:
        page_no, content_no = len(objects) + 1, len(objects) + 2
        kids.append(f"{page_no} 0 R")
        objects.append(
            (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_w * PT_PER_MM:.4f} "
             f"{page_h * PT_PER_MM:.4f}] /Resources << /Font << /F1 3 0 R >> >> "
             f"/Contents {content_no} 0 R >>").encode("ascii"))
        stream = content(shapes)
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"
                       + stream + b"\nendstream")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(pages)} >>".encode("ascii")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n").encode()
    path.write_bytes(bytes(out))
    return path


def write_svg(path: Path, page_mm: tuple[float, float], shapes: tuple[list, list, list]) -> Path:
    """One page of write_pdf's (filled rects, stroked rects, texts), as SVG in mm:
    the same page for a screen, where a PDF cannot be shown as a picture."""
    page_w, page_h = page_mm
    filled, stroked, texts = shapes
    # PDF y runs up from the page's bottom, SVG y down from its top.
    path_d = "".join(f"M{x:.3f} {page_h - y - h:.3f}h{w:.3f}v{h:.3f}h{-w:.3f}z" for x, y, w, h in filled)
    out = [(f'<svg xmlns="http://www.w3.org/2000/svg" width="{page_w:g}mm" height="{page_h:g}mm" '
            f'viewBox="0 0 {page_w:g} {page_h:g}">'),
           f'<rect width="{page_w:g}" height="{page_h:g}" fill="#fff"/>']
    if filled:
        out.append(f'<path d="{path_d}" fill="#000"/>')
    for x, y, w, h in stroked:
        out.append(f'<rect x="{x:.3f}" y="{page_h - y - h:.3f}" width="{w:.3f}" height="{h:.3f}" '
                   f'fill="none" stroke="#000" stroke-width="0.07"/>')
    for x, y, size, text in texts:
        escaped = text.replace("&", "&amp;").replace("<", "&lt;")
        out.append(f'<text x="{x:.3f}" y="{page_h - y:.3f}" font-family="Helvetica, Arial, sans-serif" '
                   f'font-size="{size:g}">{escaped}</text>')
    out.append("</svg>\n")
    path.write_text("\n".join(out))
    return path


def solid_mesh(cfg: Settings = SETTINGS) -> np.ndarray:
    """The solid as an (n, 3, 3) array of triangles, millimetres.

    Every marker face gets a shallow pocket the paper drops into, with its name
    cut into the floor; the blank face gets a deep square socket for the stalk,
    and the board's mount gets the same socket - in the face that carries it,
    or in the flat cut of a shape that has one. Cut with manifold3d: the hull is a convex shape with triangles
    between the squares, and writing that by hand is how a triangle ends up
    wound the wrong way and a slicer fills the pocket it thinks is a hole.
    """
    return _polyhedron_mesh(cfg)


def _polyhedron_mesh(cfg: Settings) -> np.ndarray:
    """The rhombicuboctahedron, whole or cut to a dome, with its pockets and socket."""
    import itertools

    import manifold3d as m

    if cfg.shape not in SHAPE_INRADIUS_PER_HALF_FACE:
        raise ValueError(f"no mesh for shape {cfg.shape!r}")
    half = cfg.face_size_mm / 2.0
    k = 1.0 + 2.0 ** 0.5
    # Its 24 vertices are every signed permutation of (1, 1, 1+sqrt2), scaled so
    # a square face is face_size across.
    points = set()
    for s1, s2, s3 in itertools.product((-1.0, 1.0), repeat=3):
        for perm in set(itertools.permutations((s1, s2, s3 * k))):
            points.add(tuple(half * c for c in perm))
    solid = m.Manifold.hull_points([list(pt) for pt in points])
    if cfg.cut_mm is not None:
        # everything under the side squares goes: the cut is flat, as wide as
        # the solid, and carries the driver board's mount - a separate print
        # (board_mount_*.stl) that plugs into a second stalk
        # socket in the middle of the cut, the same socket the roof's peg gets
        solid = solid ^ m.Manifold.cube([4 * half * 3, 4 * half * 3, 4 * half * 3], True) \
            .translate([0.0, 0.0, 6 * half - cfg.cut_mm])
        if cfg.solid_spec.sockets:
            socket = m.Manifold.cube([cfg.stalk_mm, cfg.stalk_mm, cfg.stalk_depth_mm + 0.2], True) \
                .translate([0.0, 0.0, -cfg.cut_mm + cfg.stalk_depth_mm / 2.0 - 0.1])
            solid = solid - socket

    from .glyphs import text_section

    for name, (ex, ey, n) in face_axes(cfg).items():
        if name not in cfg.marker_faces and name not in cfg.socket_faces:
            continue   # a plain face: no marker, no socket - the desk's blank side
        blank = name in cfg.socket_faces
        side = cfg.stalk_mm if blank else cfg.pocket_mm
        depth = cfg.stalk_depth_mm if blank else cfg.recess_mm
        out = cfg.mount_out_mm if blank else cfg.inradius_mm
        # a square prism standing on the face, sunk `depth` in and poking a
        # touch out, in the face's own frame
        prism = m.Manifold.cube([side, side, depth + 0.2], True)
        frame = np.column_stack([ex, ey, n])
        origin = n * (out - depth / 2.0 + 0.1)
        prism = prism.transform(np.hstack([frame, origin.reshape(3, 1)]))
        solid = solid - prism
        if blank:
            continue
        # The face's name, cut into the middle of its pocket's floor. Read while
        # gluing - which patch goes here, and which way is up, since the letters
        # stand upright when the face does - and hidden by the paper afterwards,
        # so the marker's white border stays white.
        letters = text_section(face_label(name), 0.0, 0.0,
                               cfg.name_mm, cfg.name_mm * 0.7, cfg.name_stroke_mm,
                               cfg.name_mm * 0.25)
        cut = letters.extrude(cfg.name_depth_mm + 0.2) \
            .translate([0.0, 0.0, -(cfg.name_depth_mm + 0.2)])
        cut = cut.transform(np.hstack([frame, (n * (out - depth + 0.1)).reshape(3, 1)]))
        solid = solid - cut
        if cfg.tile_key_mm > 0 or cfg.tile_rib_mm > 0:
            solid = solid + _tile_seat(m, cfg, frame, n * out, depth)

    # The hand's letter in every triangle, standing upright toward the solid's
    # +z. A triangle's plane is n . p = the largest n . p over the vertices,
    # and its centre sits on n there.
    letter = cfg.solid_spec.hand_letter
    if letter and cfg.cut_mm is None:
        glyph = text_section(letter, 0.0, 0.0, cfg.hand_letter_mm, cfg.hand_letter_mm * 0.7,
                             cfg.hand_letter_stroke_mm, 0.0)
        cut = glyph.extrude(cfg.name_depth_mm + 0.2).translate([0.0, 0.0, -cfg.name_depth_mm])
        for sx, sy, sz in itertools.product((-1.0, 1.0), repeat=3):
            n = np.array([sx, sy, sz]) / math.sqrt(3.0)
            reach = max(float(n @ np.array(pt)) for pt in points)
            ey = np.array([0.0, 0.0, 1.0]) - sz / math.sqrt(3.0) * n
            ey = ey / np.linalg.norm(ey)
            ex = np.cross(ey, n)
            frame = np.column_stack([ex, ey, n])
            solid = solid - cut.transform(np.hstack([frame, (n * reach).reshape(3, 1)]))

    mesh = solid.to_mesh()
    verts = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
    faces = np.asarray(mesh.tri_verts, dtype=np.int64)
    return verts[faces]


def _tile_seat(m, cfg: Settings, frame: np.ndarray, face: np.ndarray, depth: float):
    """What a pocket keeps of itself for a printed tile, in the solid's frame: its
    (-x, -y) corner, filled a touch clear of the tile's cut one, and the crush
    ribs, two on each wall at a quarter of the way in from either end, standing
    from the floor to tile_rib_lead_mm under the face. `face` is the face's
    centre, `frame` its (x, y, normal) axes as columns."""
    half = cfg.pocket_mm / 2.0
    fit = cfg.recess_clearance_mm
    place = np.hstack([frame, face.reshape(3, 1)])
    parts = []
    if cfg.tile_key_mm > 0:
        legs = cfg.tile_key_mm + fit - 2.0 * math.sqrt(2.0) * fit
        corner = m.CrossSection([[(-half, -half), (-half + legs, -half), (-half, -half + legs)]])
        parts.append(m.Manifold.extrude(corner, depth + 0.02).translate([0.0, 0.0, -depth - 0.01]))
    if cfg.tile_rib_mm > 0:
        w, into = cfg.tile_rib_width_mm / 2.0, 0.2
        # one rib on the +x wall at y = 0, apex inward, its base on the wall and
        # carried `into` it; turned to each wall and slid along it
        rib = m.CrossSection([[(half + into, -w), (half + into, w), (half, w),
                               (half - cfg.tile_rib_mm, 0.0), (half, -w)]])
        rib = m.Manifold.extrude(rib, depth - cfg.tile_rib_lead_mm + 0.01).translate([0.0, 0.0, -depth - 0.01])
        for turn in (0.0, 90.0, 180.0, 270.0):
            for along in (-half / 2.0, half / 2.0):
                parts.append(rib.translate([0.0, along, 0.0]).rotate([0.0, 0.0, turn]))
    return m.Manifold.batch_boolean(parts, m.OpType.Add).transform(place)


def solid_stl(cfg: Settings = SETTINGS, scale: float = 1.0) -> bytes:
    """The solid as a binary STL, millimetres times `scale`."""
    tris = solid_mesh(cfg) * scale
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(lengths == 0, 1.0, lengths)

    out = bytearray(b"pinkrab_umi marker solid".ljust(80, b" "))
    out += struct.pack("<I", len(tris))
    for normal, tri in zip(normals, tris):
        out += struct.pack("<12fH", *normal, *tri[0], *tri[1], *tri[2], 0)
    return bytes(out)


def write_solid_stl(path: Path, cfg: Settings = SETTINGS) -> Path:
    """Binary STL, millimetres - what every slicer assumes and this is."""
    path.write_bytes(solid_stl(cfg))
    return path


def read_stl(path: Path) -> np.ndarray:
    """Binary STL back to (n, 3, 3) millimetres - for checking what was written."""
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 80)[0]
    tris = np.empty((count, 3, 3))
    for i in range(count):
        values = struct.unpack_from("<12fH", data, 84 + i * 50)
        tris[i] = np.array(values[3:12]).reshape(3, 3)
    return tris


def is_manifold(tris: np.ndarray, tol: int = 6) -> tuple[bool, dict[str, int]]:
    """Every edge used exactly twice, once each way - the printable definition.

    Vertices are matched on rounded coordinates rather than identity: the mesh is
    built from exact axis arithmetic, but an STL has been through float32 and a
    slicer will weld on tolerance anyway.
    """
    from collections import Counter

    keys = [tuple(np.round(v, tol)) for tri in tris for v in tri]
    index = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    directed: Counter = Counter()
    for t in range(len(tris)):
        ids = [index[keys[t * 3 + k]] for k in range(3)]
        for k in range(3):
            directed[(ids[k], ids[(k + 1) % 3])] += 1

    undirected: Counter = Counter()
    for (i, j), n in directed.items():
        undirected[(min(i, j), max(i, j))] += n
    stats = {
        "triangles": len(tris),
        "vertices": len(index),
        "edges": len(undirected),
        "boundary_edges": sum(1 for n in undirected.values() if n != 2),
        "reversed_edges": sum(1 for e, n in directed.items() if n != 1),
    }
    return stats["boundary_edges"] == 0 and stats["reversed_edges"] == 0, stats


def write_all(out_dir: Path, cfg: Settings = SETTINGS) -> list[Path]:
    """The sheet and the solid, into one directory, named for the solid."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"marker_solid_{cfg.solid}"
    return [
        write_sheet_pdf(out_dir / f"{stem}_sheet.pdf", cfg),
        write_solid_stl(out_dir / f"{stem}.stl", cfg),
    ]
