#!/usr/bin/env python3
"""Verify the whole pipeline without a camera, a solid or a room.

Every stage here fails silently. A face transform with a sign the wrong way round
still solves; the pose is just wrong, by exactly the amount that looks like a hand
moving. A quaternion convention mismatch in the composition gives a smooth
trajectory of the wrong thing. Reprojection error catches none of
it, because the fit is fine - it is the geometry behind the fit that is wrong.

So the solid is rendered into synthetic frames from a known pose, put through the
same detector, solver and composition the real thing uses, and compared against
the pose it was rendered from. That closes the loop the residuals cannot.

    python src/pinkrab_umi/test/offline_check.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC / "pinkrab_umi"))
sys.path.insert(0, str(SRC / "pinkrab_core"))

from pinkrab_umi import bench, filtering, markers, pnp, trajectory  # noqa: E402
from pinkrab_umi.config import SETTINGS  # noqa: E402

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


# --- the synthetic camera -------------------------------------------------

# A plausible camera. The numbers are nominal rather than a device's, which is
# the point: the test is about the geometry, and the geometry has to hold
# whatever the focal length turns out to be.
CAMERA = pnp.Intrinsics(width=848, height=480, fx=425.0, fy=425.0, cx=424.0, cy=240.0)

MM = markers.MM


def patch_image(marker_id: int, pixels: int = 960) -> np.ndarray:
    """One face as it is printed: the marker centred on a white patch."""
    patch = np.full((pixels, pixels), 255, np.uint8)
    span = round(pixels * SETTINGS.marker_mm / SETTINGS.patch_mm)
    marker = cv2.aruco.generateImageMarker(markers.dictionary(), marker_id, span)
    off = (pixels - span) // 2
    patch[off:off + span, off:off + span] = marker
    return patch


def patch_corners_solid() -> dict[int, np.ndarray]:
    """The patch's own four corners in the solid frame - the render's quad, not PnP's."""
    h = SETTINGS.patch_mm * MM / 2.0
    local = np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])
    poses = markers.face_poses()
    return {
        SETTINGS.face_ids[name]: (local @ poses[name][:3, :3].T) + poses[name][:3, 3]
        for name in SETTINGS.marker_faces
    }


# The render is done this many times larger and boxed back down. Warping a printed
# patch straight into a 50-pixel square aliases its edges, and an aliased edge
# moves the refined corner by a fifth of a pixel - which at this range is
# millimetres, and would set the tolerance below rather than the geometry doing it.
# A real sensor integrates over its pixel; this is the same thing.
SUPERSAMPLE = 4


def render(cam_T_solid: np.ndarray, intrinsics: pnp.Intrinsics = CAMERA,
           only: tuple[str, ...] | None = None) -> np.ndarray:
    """Draw the solid as a real camera would see it, back faces omitted.

    A painter over the front-facing quads only. The solid is convex, so no two
    front-facing faces overlap and there is nothing for a depth test to decide.
    `only` draws just those faces - the solid has no direction it shows one face
    from, so a single-face frame has to be made rather than found.
    """
    big = pnp.Intrinsics(
        width=intrinsics.width * SUPERSAMPLE, height=intrinsics.height * SUPERSAMPLE,
        fx=intrinsics.fx * SUPERSAMPLE, fy=intrinsics.fy * SUPERSAMPLE,
        # A pixel centre at c maps to SUPERSAMPLE*c + (SUPERSAMPLE-1)/2 in the
        # finer grid; getting this offset wrong shifts the whole image half a pixel
        # and biases every pose in the same direction.
        cx=intrinsics.cx * SUPERSAMPLE + (SUPERSAMPLE - 1) / 2,
        cy=intrinsics.cy * SUPERSAMPLE + (SUPERSAMPLE - 1) / 2,
        coeffs=intrinsics.coeffs,
    )
    fine = _render_at(cam_T_solid, big, only)
    return cv2.resize(fine, (intrinsics.width, intrinsics.height), interpolation=cv2.INTER_AREA)


def _render_at(cam_T_solid: np.ndarray, intrinsics: pnp.Intrinsics,
               only: tuple[str, ...] | None = None) -> np.ndarray:
    image = np.full((intrinsics.height, intrinsics.width), 210, np.uint8)
    face_poses = markers.face_poses()
    for name in (only or SETTINGS.marker_faces):
        marker_id = SETTINGS.face_ids[name]
        solid_T_face = face_poses[name]
        normal_cam = cam_T_solid[:3, :3] @ solid_T_face[:3, 2]
        centre_cam = cam_T_solid[:3, :3] @ solid_T_face[:3, 3] + cam_T_solid[:3, 3]
        # Facing the camera means the outward normal points back along the ray to
        # it. The margin keeps grazing faces out of the render entirely, so a
        # dropped grazing face is the solver's decision rather than the renderer's.
        if float(normal_cam @ (centre_cam / np.linalg.norm(centre_cam))) > -0.25:
            continue

        corners_solid = patch_corners_solid()[marker_id]
        corners_cam = corners_solid @ cam_T_solid[:3, :3].T + cam_T_solid[:3, 3]
        projected, _ = cv2.projectPoints(
            corners_cam.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
            intrinsics.matrix, intrinsics.distortion,
        )
        dst = projected.reshape(4, 2).astype(np.float32)

        source = patch_image(marker_id)
        n = source.shape[0] - 1
        src = np.array([[0, 0], [n, 0], [n, n], [0, n]], np.float32)
        warp = cv2.getPerspectiveTransform(src, dst)
        drawn = cv2.warpPerspective(source, warp, (intrinsics.width, intrinsics.height),
                                    flags=cv2.INTER_AREA, borderValue=0)
        mask = cv2.warpPerspective(np.full_like(source, 255), warp,
                                   (intrinsics.width, intrinsics.height),
                                   flags=cv2.INTER_NEAREST, borderValue=0)
        image[mask > 127] = drawn[mask > 127]
    return image


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """world_T_cam for a camera at `eye` pointing at `target`, +Z out of the lens."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    # World up, unless the camera is looking along it - straight down at the solid
    # is a shot this test takes, and there the cross product is zero.
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(forward @ up)) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([right, down, forward])
    pose[:3, 3] = eye
    return pose


def rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return cv2.Rodrigues(axis * angle)[0]


def pose_error(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(millimetres, degrees) between two 4x4 poses."""
    relative = np.linalg.inv(a) @ b
    return (float(np.linalg.norm(relative[:3, 3]) * 1000.0),
            trajectory.rotation_angle_deg(relative))


# --- the checks -----------------------------------------------------------

def check_geometry() -> None:
    print("\nsolid geometry")
    poses = markers.face_poses()
    marked = SETTINGS.marker_faces
    check("eighteen squares: sixteen marked, two with sockets",
          len(poses) == 18 and len(SETTINGS.face_ids) == 16
          and len(SETTINGS.socket_faces) == 2)
    check("every face frame is a right-handed rotation",
          all(np.isclose(np.linalg.det(p[:3, :3]), 1.0) for p in poses.values()))
    check("every face sits the inradius out along its own normal",
          all(np.allclose(p[:3, 3], p[:3, 2] * SETTINGS.inradius_mm * MM)
              for p in poses.values()))
    # The socket faces are where the stalk and the board's mount go. A marker on
    # either would be one the solid cannot be held without hiding.
    check("the socket faces carry no marker",
          all(name not in SETTINGS.face_ids for name in SETTINGS.socket_faces))
    check("it hangs by the forward face and carries the board on the hand's face",
          SETTINGS.blank_face == "y_plus" and SETTINGS.board_face == "z_minus")
    # There is a solid on the desk with this paper glued to it. The sheet and the
    # object points are both generated from this map so they cannot drift from
    # each other - but they can both drift away from the printed solid, and a
    # face solved under the wrong id is a pose confidently wrong by a right angle.
    check("the face assignment is the one that got printed",
          SETTINGS.face_ids == {
              "z_plus": 0, "x_plus": 1, "x_minus": 2, "y_minus": 3,
              "xp_yp": 4, "xp_ym": 5, "xm_yp": 6, "xm_ym": 7,
              "xp_zp": 8, "xm_zp": 9, "yp_zp": 10, "ym_zp": 11,
              "xp_zm": 12, "xm_zm": 13, "yp_zm": 14, "ym_zm": 15},
          str(SETTINGS.face_ids))
    check("the right hand's ids follow the left's without overlap",
          min(SETTINGS.for_solid("handheld_right").face_ids.values())
          > max(SETTINGS.face_ids.values()))
    # Which way up each marker is printed, as the sheet states it: image-up is
    # the solid's +Z laid flat onto the face, and +Y on the face that is +Z.
    check("every marker's up lies in its face and leans toward +Z",
          all(abs(poses[n][:3, 1] @ poses[n][:3, 2]) < 1e-9
              and (poses[n][:3, 1][2] > 1e-9 or n == "z_plus")
              for n in marked))
    check("the top marker is printed up toward +Y",
          np.allclose(poses["z_plus"][:3, 1], [0, 1, 0]))

    corners = markers.solid_object_points()
    check("sixteen markers, four corners each",
          len(corners) == 16 and all(c.shape == (4, 3) for c in corners.values()))
    sides = [np.linalg.norm(c[1] - c[0]) for c in corners.values()]
    check("corner spacing is marker_mm, not face_size_mm",
          np.allclose(sides, SETTINGS.marker_mm * MM),
          f"{np.mean(sides) * 1000:.3f} mm vs {SETTINGS.marker_mm:g} mm")
    # ArUco reports corners top-left, top-right, bottom-right, bottom-left seen
    # from in front. Listed any other way the pose comes back turned by a right
    # angle, which is a perfectly good fit to a perfectly wrong solid.
    # Top-left to top-right, crossed with top-left to bottom-left, points back
    # into the marker - so a face whose corners were listed the other way round
    # (or whose axes are left-handed) shows up here as a positive dot.
    check("corners wind the way the detector reports them",
          all(np.dot(np.cross(c[1] - c[0], c[3] - c[0]), poses[name][:3, 2]) < 0
              for name, i in SETTINGS.face_ids.items() for c in [corners[i]]))
    check("every corner is on its face's plane",
          all(np.allclose(corners[i] @ poses[name][:3, 2], SETTINGS.inradius_mm * MM)
              for name, i in SETTINGS.face_ids.items()))
    module = SETTINGS.marker_mm / 6  # 4x4 payload plus the one-module black border
    check("the marker's white border is at least one module wide",
          (SETTINGS.patch_mm - SETTINGS.marker_mm) / 2 >= module,
          f"{(SETTINGS.patch_mm - SETTINGS.marker_mm) / 2:g} mm vs {module:g} mm")
    # The up mark is printed inside the patch, so it eats into that border.
    check("the up mark leaves the marker some white to itself",
          (SETTINGS.patch_mm - SETTINGS.marker_mm) / 2 - SETTINGS.up_mark_mm > 0.5 * module,
          f"{(SETTINGS.patch_mm - SETTINGS.marker_mm) / 2 - SETTINGS.up_mark_mm:g} mm left")


def check_stl(out_dir: Path) -> None:
    print("\nthe printed solid")
    written = markers.write_all(out_dir)
    for path in written:
        check(f"{path.name} written", path.exists() and path.stat().st_size > 0,
              f"{path.stat().st_size:,} bytes" if path.exists() else "missing")

    stl = out_dir / f"marker_solid_{SETTINGS.solid}.stl"
    tris = markers.read_stl(stl)
    watertight, stats = markers.is_manifold(tris)
    # A slicer given a mesh with a hole in it repairs it, and what it repairs to
    # is its own guess. That guess would be the pocket the paper sits in.
    check("the mesh is watertight - every edge used exactly twice", watertight,
          f"{stats['boundary_edges']} boundary, {stats['reversed_edges']} mis-wound")
    check("it is one closed surface (V - E + F = 2)",
          stats["vertices"] - stats["edges"] + stats["triangles"] == 2,
          f"V{stats['vertices']} E{stats['edges']} F{stats['triangles']}")
    verts = np.unique(np.round(tris.reshape(-1, 3), 6), axis=0)
    check("the solid spans twice the inradius on every axis",
          np.allclose(verts.max(axis=0), SETTINGS.inradius_mm, atol=1e-3)
          and np.allclose(verts.min(axis=0), -SETTINGS.inradius_mm, atol=1e-3))
    # Every face's floor: the vertices that sit exactly one recess under a marker
    # face, or one socket under a socket face. Measured along each face's own
    # normal, since the edge faces lie at forty-five degrees to every axis.
    poses = markers.face_poses()

    def floor_at(name: str, depth: float) -> bool:
        along = verts @ poses[name][:3, 2]
        return bool(np.any(np.isclose(along, SETTINGS.inradius_mm - depth, atol=1e-3)))

    check("every marker face has a paper pocket one recess deep",
          all(floor_at(n, SETTINGS.recess_mm) for n in SETTINGS.marker_faces))
    check("both socket faces have a stalk socket one socket deep",
          all(floor_at(n, SETTINGS.stalk_depth_mm) for n in SETTINGS.socket_faces))
    check("the sockets do not break out of the solid",
          SETTINGS.stalk_depth_mm < SETTINGS.inradius_mm)

    pdf = (out_dir / f"marker_solid_{SETTINGS.solid}_sheet.pdf").read_bytes()
    check("the sheet is a one-page PDF",
          pdf.startswith(b"%PDF-") and pdf.rstrip().endswith(b"%%EOF")
          and b"/Count 1" in pdf and b"startxref" in pdf)
    # Sixteen patches, and the ruler bar is drawn to the millimetre it claims.
    check("the page carries every face and a ruler",
          all(f"id {i}".encode() in pdf for i in SETTINGS.face_ids.values())
          and f"exactly {SETTINGS.ruler_mm:g} mm".encode() in pdf)
    lefts = [x for _, x, _, _ in markers._patch_origins(SETTINGS)]
    check("every patch stays inside a 15 mm margin",
          min(lefts) >= 15.0 and max(lefts) + SETTINGS.patch_mm <= SETTINGS.page_mm[0] - 15.0,
          f"{min(lefts):.0f}..{max(lefts) + SETTINGS.patch_mm:.0f} mm across")


def check_detection() -> None:
    print("\ndetection and pose, from rendered frames")
    solid_T = np.eye(4)
    # Off a corner, so three faces are well presented rather than one grazing.
    cam_T_solid = np.linalg.inv(look_at(np.array([0.22, -0.22, 0.22]), np.zeros(3))) @ solid_T
    result = pnp.solve_frame(render(cam_T_solid), CAMERA)
    check("the solid is found in a rendered frame", result.ok, result.reason)
    check("more than one face contributes", len(result.faces) >= 2, str(result.faces))
    check("reprojection is inside the threshold",
          result.reproj_px <= SETTINGS.max_reproj_px, f"{result.reproj_px:.3f} px")
    mm, deg = pose_error(cam_T_solid, result.pose)
    check("the recovered pose is the rendered pose", mm < 3.0 and deg < 2.0,
          f"{mm:.2f} mm, {deg:.2f} deg")

    # Nothing in the frame is a real case - a hand turns the solid away - and it has
    # to come back as a hole rather than as an exception or a stale pose.
    blank = np.full((480, 848), 200, np.uint8)
    empty = pnp.solve_frame(blank, CAMERA)
    check("an empty frame is a hole, not an error", not empty.ok and empty.pose is None,
          empty.reason)

    # Solved again from grey, which is what the colour path does underneath.
    mono = pnp.solve_frame(render(cam_T_solid), CAMERA)
    check("the grey path agrees", mono.ok
          and pose_error(cam_T_solid, mono.pose)[0] < 5.0,
          f"{pose_error(cam_T_solid, mono.pose)[0]:.2f} mm" if mono.ok else mono.reason)

    # The colour path is the same solve with a colour picture in: rendered grey
    # widened to three channels.
    in_colour = pnp.solve_colour(cv2.cvtColor(render(cam_T_solid), cv2.COLOR_GRAY2BGR), CAMERA)
    check("a colour picture solves, in the same frame", in_colour.ok
          and pose_error(cam_T_solid, in_colour.pose)[0] < 5.0,
          f"{pose_error(cam_T_solid, in_colour.pose)[0]:.2f} mm"
          if in_colour.ok else in_colour.reason)

    # Solving from one face is the degraded case, not the broken one. The solid
    # has no direction it shows a single face from, so one is drawn alone.
    # Tilted rather than square on, because a planar square seen head-on has two
    # poses that fit equally well and the solver picks between them on noise -
    # which is the whole reason the solid has sixteen faces and not one.
    one_face = np.linalg.inv(look_at(np.array([0.075, -0.39, 0.075]), np.zeros(3))) @ solid_T
    result = pnp.solve_frame(render(one_face, only=("y_minus",)), CAMERA)
    check("a single face still solves", result.ok and len(result.faces) == 1,
          f"{result.faces}")
    # One face is a plane and a plane fits two poses, so the other solution is
    # carried out for the stream filter to rank against the frame before.
    check("a single face carries its alternate", result.alternate is not None)
    # The face is the inradius from the centre, so a degree of face is most of
    # a millimetre of centre. Which of the two the solver puts first turns on a
    # pixel or so of the marker's outline - from here a hand solid's 21 mm face
    # puts the other one first - so the right place has to be one of the two,
    # for the stream filter to keep against the frame before; what this rules
    # out is a face solved to neither, or not carrying the other at all.
    errors = [pose_error(one_face, T) for T in (result.pose, result.alternate) if T is not None] \
        if result.ok else [(float("inf"),) * 2]
    mm, deg = min(errors, key=lambda e: e[1])
    check("and one of its two solutions is the right place", mm < 10.0 and deg < 3.0,
          f"{mm:.2f} mm, {deg:.2f} deg")


def check_end_to_end() -> None:
    print("\nend to end - a moving camera and a moving solid")
    # The solid travels and turns; the camera orbits and looks at it. Neither is
    # still, so a composition that got world_T_cam and cam_T_solid the wrong way
    # round comes out visibly wrong rather than accidentally right.
    stamps = np.linspace(0.0, 1.0, 25)
    truth: list[np.ndarray] = []
    cam_track_poses: list[np.ndarray] = []
    for t in stamps:
        world_T_solid = np.eye(4)
        world_T_solid[:3, :3] = rotation(np.array([0.3, -0.4, 1.0]), 1.2 * t)
        world_T_solid[:3, 3] = [0.05 * np.sin(3 * t), 0.04 * t, 0.03 * np.cos(2 * t)]
        truth.append(world_T_solid)

        # A full turn round the solid, so every marked face gets
        # its turn in frame and a face transform with a sign wrong somewhere is
        # exercised rather than skipped.
        angle = 2 * np.pi * t
        # ...swinging from above the solid to below it on the way round, so the
        # faces that lean down get their turn as well as the ones that lean up.
        eye = np.array([0.34 * np.sin(angle), -0.34 * np.cos(angle),
                        0.03 + 0.20 * np.cos(2 * angle)])
        cam_track_poses.append(look_at(eye, world_T_solid[:3, 3]))

    results: list[pnp.SolidPose] = []
    cam_poses: list[np.ndarray | None] = []
    for world_T_cam, world_T_solid in zip(cam_track_poses, truth):
        result = pnp.solve_frame(render(np.linalg.inv(world_T_cam) @ world_T_solid), CAMERA)
        results.append(result)
        cam_poses.append(result.pose if result.ok else None)

    cover = bench.coverage(results)
    check("every rendered frame solved", cover.solved == len(stamps),
          f"{cover.solved}/{len(stamps)}")
    check("every marked face is seen across the take",
          len(cover.face_counts) == len(SETTINGS.marker_faces),
          str(sorted(cover.face_counts)))
    check("coverage passes its threshold", cover.passes, f"{cover.fraction:.1%}")

    # Composed frame by frame with the camera's own pose, as the sheet gives it.
    times = [t for t, c in zip(stamps, cam_poses) if c is not None]
    poses = [trajectory.compose(world_T_cam, c)
             for world_T_cam, c in zip(cam_track_poses, cam_poses) if c is not None]
    check("every solved frame composed to the world", len(poses) == cover.solved)
    errors = [pose_error(a, b) for a, b in zip(truth, poses)]
    worst_mm = max(e[0] for e in errors)
    worst_deg = max(e[1] for e in errors)
    # 5 mm is the renderer's floor, not the pipeline's: warping a printed patch
    # into a 50-pixel square leaves a fraction of a pixel on each corner even
    # supersampled, and at this range a fraction of a pixel is millimetres. What
    # this bound is for is geometry - a face transform with a sign wrong, or an id
    # on the wrong face, puts the solid out by a good part of its own 50 mm, and no
    # amount of rendering care would bring that under 5.
    check("the recovered world trajectory is the true one",
          worst_mm < 5.0 and worst_deg < 2.0,
          f"worst {worst_mm:.2f} mm, {worst_deg:.2f} deg  "
          f"(median {np.median([e[0] for e in errors]):.2f} mm)")

    # Composition is not commutative and both orders type-check. This is the one
    # test that would notice.
    swapped = [trajectory.compose(c, np.linalg.inv(t)) for c, t in zip(cam_poses, truth)]
    check("composing the other way round does not accidentally work",
          max(pose_error(a, b)[0] for a, b in zip(truth, swapped)) > 10.0)
    check("timestamps survive composition", np.allclose(times, stamps))


def check_closure() -> None:
    print("\nclosure metric")
    def loop(closed: bool) -> list[np.ndarray]:
        poses = []
        for t in np.linspace(0.0, 1.0, 30):
            pose = np.eye(4)
            pose[:3, :3] = rotation(np.array([0.0, 0.0, 1.0]),
                                    2 * np.pi * t if closed else 0.6 * t)
            pose[:3, 3] = [0.1 * np.sin(2 * np.pi * t), 0.1 * (1 - np.cos(2 * np.pi * t)),
                           0.0 if closed else 0.05 * t]
            poses.append(pose)
        return poses

    shut = bench.closure_error(loop(True))
    check("a closed loop closes to nothing",
          shut.translation_mm < 1e-6 and shut.rotation_deg < 1e-6,
          f"{shut.translation_mm:.6f} mm, {shut.rotation_deg:.6f} deg")
    check("and passes the thresholds", shut.passes)

    open_ = bench.closure_error(loop(False))
    check("an open take does not", open_.translation_mm > 10.0 and open_.rotation_deg > 3.0,
          f"{open_.translation_mm:.2f} mm, {open_.rotation_deg:.2f} deg")
    check("and fails the thresholds", not open_.passes)
    # Two frames cannot be told apart from a take that never moved, so the metric
    # says it has nothing rather than reporting a confident zero.
    check("one pose is not a loop", np.isnan(bench.closure_error([np.eye(4)]).translation_mm))

    # The end-to-end path, closed: render a take that comes home and check the
    # measured closure is the pipeline's own error rather than the loop's.
    world_T_cam = look_at(np.array([0.0, -0.36, 0.10]), np.zeros(3))
    poses = []
    for t in np.linspace(0.0, 1.0, 12):
        world_T_solid = np.eye(4)
        world_T_solid[:3, :3] = rotation(np.array([0.2, 0.3, 1.0]), 2 * np.pi * t)
        world_T_solid[:3, 3] = [0.03 * np.sin(2 * np.pi * t), 0.0,
                                0.03 * (np.cos(2 * np.pi * t) - 1)]
        result = pnp.solve_frame(render(np.linalg.inv(world_T_cam) @ world_T_solid), CAMERA)
        if result.ok:
            poses.append(trajectory.compose(np.eye(4), result.pose))
    measured = bench.closure_error(poses)
    check("a rendered take that comes home measures near zero", measured.passes,
          f"{measured.translation_mm:.2f} mm, {measured.rotation_deg:.2f} deg")


def check_filter() -> None:
    """The stream filter, on constructed streams where the right answer is known."""
    print("\nstream filter")

    def pose_at(angle_deg: float) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = rotation(np.array([0.0, 1.0, 0.0]), np.radians(angle_deg))
        T[:3, 3] = (0.0, 0.0, 0.5)
        return T

    def single(pose, alternate=None, reproj=0.3, alt_reproj=0.35, reason=""):
        return pnp.SolidPose(ok=True, pose=pose, faces=(0,), reproj_px=reproj,
                             alternate=alternate, alternate_reproj_px=alt_reproj,
                             reason=reason)

    def multi(pose):
        return pnp.SolidPose(ok=True, pose=pose, faces=(0, 1), reproj_px=0.3)

    dt = 1 / 30

    # The choices below are judged on which answer the filter picked, so the
    # smoothing that follows the choice is switched off for them and checked
    # on its own afterwards.
    from dataclasses import replace as _replace
    unsmoothed = _replace(SETTINGS, filter_ema_alpha=1.0)
    filtering.SETTINGS = unsmoothed

    # An ambiguous single-face frame whose solver-preferred pose is the flip: the
    # anchor from the multi-face frame before it must pull the choice back.
    f = filtering.PoseFilter()
    f.apply(multi(pose_at(0.0)), 0.0)
    out = f.apply(single(pose_at(80.0), alternate=pose_at(2.0)), dt)
    check("ambiguous flip is pulled back to the continuous solution",
          out.ok and trajectory.rotation_angle_deg(out.pose[:3, :3]) < 10.0)

    # Residuals that rank clearly let the frame answer for itself.
    out = f.apply(single(pose_at(4.0), alternate=pose_at(60.0), alt_reproj=2.0), 2 * dt)
    check("a clearly ranked single face keeps its own answer",
          out.ok and abs(trajectory.rotation_angle_deg(out.pose[:3, :3]) - 4.0) < 1.0)

    # Both solutions a teleport away: refused, not chosen between.
    out = f.apply(single(pose_at(90.0), alternate=pose_at(-70.0)), 3 * dt)
    check("a frame both of whose answers teleport is refused",
          not out.ok and "jump" in out.reason)

    # The refusal must not have moved the anchor.
    out = f.apply(single(pose_at(80.0), alternate=pose_at(5.0)), 4 * dt)
    check("a refused frame does not become anybody's anchor",
          out.ok and abs(trajectory.rotation_angle_deg(out.pose[:3, :3]) - 5.0) < 1.0)

    # An anchor a second stale cannot vouch for an ambiguous frame.
    out = f.apply(single(pose_at(6.0), alternate=pose_at(70.0)),
                  4 * dt + SETTINGS.filter_anchor_max_age_s + 0.5)
    check("an ambiguous frame with a stale anchor is refused",
          not out.ok and "anchor" in out.reason)

    # A frame the reprojection gate already flagged passes through untouched.
    flagged = single(pose_at(0.0), reason="reprojection 3.00 px over 2")
    check("a gate-flagged frame is not this filter's case",
          f.apply(flagged, 10.0) is flagged)

    # Multi-face frames pass untouched and re-arm the anchor.
    f2 = filtering.PoseFilter()
    kept = multi(pose_at(10.0))
    check("a multi-face frame passes as itself", f2.apply(kept, 0.0) is kept)
    out = f2.apply(single(pose_at(75.0), alternate=pose_at(12.0)), dt)
    check("...and its anchor decides the demoted frame after it",
          out.ok and abs(trajectory.rotation_angle_deg(out.pose[:3, :3]) - 12.0) < 1.0)
    filtering.SETTINGS = SETTINGS

    # The smoothing: a still solid read with the noise the rig actually has comes
    # out steadier, and a moving one is not left far behind.
    rng = np.random.default_rng(0)

    def noisy(pose):
        axis = rng.normal(size=3)
        out = pose.copy()
        out[:3, :3] = rotation(axis, np.radians(rng.normal(0, 0.55))) @ pose[:3, :3]
        out[:3, 3] = out[:3, 3] + rng.normal(0, 0.0012, 3)
        return out

    still = np.eye(4)
    still[:3, 3] = [0.0, 0.0, 0.4]
    f3 = filtering.PoseFilter()
    raw = [noisy(still) for _ in range(300)]
    smoothed = [f3.apply(multi(p), i * dt).pose for i, p in enumerate(raw)]
    raw_std = np.array([p[:3, 3] for p in raw]).std(axis=0).mean()
    out_std = np.array([p[:3, 3] for p in smoothed]).std(axis=0).mean()
    check("a still solid's noise comes out under half of itself",
          out_std < 0.5 * raw_std, f"{raw_std * 1000:.2f} -> {out_std * 1000:.2f} mm")

    f4 = filtering.PoseFilter()
    lag = []
    for i in range(90):
        moving = still.copy()
        moving[0, 3] = 0.2 * i * dt            # a hand at 200 mm/s
        lag.append(moving[0, 3] - f4.apply(multi(noisy(moving)), i * dt).pose[0, 3])
    behind = float(np.mean(lag[30:]))
    check("a hand at 200 mm/s is followed under 100 ms behind",
          behind < 0.2 * 0.1, f"{behind * 1000:.1f} mm, {behind / 0.2 * 1000:.0f} ms")


def check_gripper_view() -> None:
    """The gripper picture: a window about the lens axis, the same picture
    whether it is cut from the sensor's frame (convert) or the live view's
    half-size JPEG (rollout), and two units' axes brought together."""
    from pinkrab_core.dataset import gripper_shape, gripper_view, gripper_window

    print("\nthe gripper picture")
    size = gripper_shape(1920, 1200)
    check("an AR0234 frame goes in at 780 x 570", size == (570, 780), str(size))

    def frame_at(cx: float, cy: float) -> np.ndarray:
        # A lit disc for the image circle and a dot where the lens axis lands.
        frame = np.zeros((1200, 1920, 3), np.uint8)
        cv2.circle(frame, (round(cx), round(cy)), 777, (255, 255, 255), -1)
        cv2.circle(frame, (round(cx), round(cy)), 12, (0, 0, 255), -1)
        return frame

    def dot(picture: np.ndarray) -> np.ndarray:
        ys, xs = np.nonzero((picture[..., 2] > 128) & (picture[..., 1] < 100))
        return np.array([xs.mean(), ys.mean()])

    handheld, follower = (961.4, 605.4), (947.5, 578.9)   # the left pair as measured
    a = gripper_view(frame_at(*handheld), size, (handheld[0] / 1920, handheld[1] / 1200))
    b = gripper_view(frame_at(*follower), size, (follower[0] / 1920, follower[1] / 1200))
    apart = float(np.linalg.norm(dot(a) - dot(b)))
    check("two units 30 px apart land on the same pixel", apart < 1.0, f"{apart:.2f} px apart")
    check("the window stays inside the frame",
          all(0 <= x and 0 <= y and x + w <= 1920 and y + h <= 1200
              for x, y, w, h in (gripper_window(1920, 1200, c) for c in ((0, 0), (1, 1), (0.5, 0.5)))))
    live = gripper_view(cv2.resize(frame_at(*follower), (960, 600), interpolation=cv2.INTER_AREA), size,
                        (follower[0] / 1920, follower[1] / 1200))
    diff = np.abs(b.astype(int) - live.astype(int))
    check("cut from the live view it is the same picture", (diff > 64).mean() < 0.005,
          f"{(diff > 64).mean():.2%} of pixels apart")
    # The cameras opened at the profile's region of the sensor, scaled: the
    # same picture, from the frame and from the live view's half of it.
    from pinkrab_core.config import SETTINGS as CORE
    profile = CORE.camera_profile("gripper")
    rx, ry, rw, rh = profile.sensor_region
    sensor_frame = frame_at(*follower)[ry:ry + rh, rx:rx + rw]
    for width in (profile.width, profile.width * 3 // 4):
        part = cv2.resize(sensor_frame, (width, round(width * rh / rw)), interpolation=cv2.INTER_AREA)
        cut = gripper_view(part, size, (follower[0] / 1920, follower[1] / 1200))
        diff = np.abs(b.astype(int) - cut.astype(int))
        check(f"cut from the {part.shape[1]}x{part.shape[0]} region frame it is the same picture",
              (diff > 64).mean() < 0.005, f"{(diff > 64).mean():.2%} of pixels apart")
    frame = frame_at(960, 600)
    whole = gripper_view(frame, (600, 960), (0.5, 0.5))
    check("a checkpoint trained on whole frames is shown whole frames",
          np.array_equal(whole, cv2.resize(frame, (960, 600), interpolation=cv2.INTER_AREA)))
    try:
        gripper_view(frame, (600, 780), (0.5, 0.5))
        refused = False
    except ValueError:
        refused = True
    check("a checkpoint cut some other way is refused, not stretched", refused)


def check_redone_corrections() -> None:
    """A rollout's corrections in its dataset: what the hands drove, but for a
    correction thrown away and made again."""
    from pinkrab_umi.convert import learned_from

    print("\ncorrections redone")
    readings = ([{"controller": "policy", "correction": None}] * 2
                + [{"controller": "hold", "correction": 1}] + [{"controller": "human", "correction": 1}] * 3
                + [{"controller": "none", "correction": None}] * 2
                + [{"controller": "human", "correction": 2}] * 4
                + [{"controller": "human"}] * 2)       # a rollout recorded before corrections were numbered
    kept, redone = learned_from(readings, [{"event": "start"}, {"event": "redo", "correction": 1}])
    check("a redone correction is left out", not kept[3:6].any())
    check("the correction made again is kept", kept[8:12].all())
    check("the policy's and the clutch's stretches are left out", not kept[:3].any() and not kept[6:8].any())
    check("an older rollout's corrections are kept", kept[12:].all())
    check("the redo is counted", redone == 1, str(redone))


def check_outcome(root: Path) -> None:
    """A rollout's outcome, said by the person watching: written into its
    meta.json, listed on the shelf, taken back; refused for a demonstration
    and for a rollout still recording."""
    import json

    from pinkrab_umi.episode import EpisodeError, list_episodes, set_outcome

    print("\nrollout outcome")
    (root / "run").mkdir()
    (root / "run" / "meta.json").write_text(json.dumps({"kind": "rollout", "policy": "/ckpt/a", "seconds": 3.0}))
    (root / "take").mkdir()
    (root / "take" / "meta.json").write_text(json.dumps({"kind": "demo"}))
    (root / "open").mkdir()

    def refused(*args) -> bool:
        try:
            set_outcome(*args, root=root)
        except EpisodeError:
            return True
        return False

    set_outcome("run", "succeeded", root=root)
    listed = {e["name"]: e for e in list_episodes(root)}
    check("the outcome is listed with the checkpoint", listed["run"].get("outcome") == "succeeded"
          and listed["run"].get("policy") == "/ckpt/a", str(listed["run"]))
    check("the rest of meta.json is kept", json.loads((root / "run" / "meta.json").read_text())["seconds"] == 3.0)
    set_outcome("run", None, root=root)
    check("an outcome taken back is gone", "outcome" not in json.loads((root / "run" / "meta.json").read_text()))
    check("a demonstration has no outcome", refused("take", "failed"))
    check("a rollout still recording has none yet", refused("open", "failed"))
    check("only succeeded or failed", refused("run", "maybe"))
    check("no path outside the shelf", refused("../run", "failed"))


def check_marker_meta() -> None:
    """A take's meta.json carries every marker it looked for, corner by
    corner on its solid, so its poses can be solved again without this code."""
    import json

    from pinkrab_umi.episode import _marker_meta

    print("\nmarkers in meta.json")
    hands = {name: SETTINGS.for_solid(name).solid_spec.face_mm for name in ("handheld_left", "handheld_right")}
    meta = json.loads(json.dumps(_marker_meta(hands)))
    check("the dictionary is named", meta["dictionary"] == SETTINGS.aruco_dict, meta["dictionary"])
    check("both hands and the desk are there", set(meta["solids"]) == {*hands, SETTINGS.desk_solid},
          str(sorted(meta["solids"])))
    for name, solid in meta["solids"].items():
        want = markers.solid_object_points(SETTINGS.for_solid(name))
        got = {int(i): np.array(c) for i, c in solid["corners"].items()}
        same = set(got) == set(want) and all(np.allclose(got[i], want[i], atol=1e-6) for i in want)
        check(f"{name}: every marker's corners as the solver has them", same, f"{len(got)} markers")
    bigger = _marker_meta({"handheld_right": hands["handheld_right"] * 2})["solids"]["handheld_right"]
    some = next(iter(bigger["corners"]))
    check("a solid recorded at another size is written at that size",
          not np.allclose(bigger["corners"][some], meta["solids"]["handheld_right"]["corners"][some]))


def check_top_light() -> None:
    """The top camera's light on a camera of another make: one with a gain
    has it fitted, one without is left at the exposure, and one whose
    exposure cannot be held is refused, said in words."""
    from pinkrab_umi import head_cam, record
    from pinkrab_umi.uvc import ControlError

    print("\ntop camera light, other cameras")

    class Cap:
        def read(self):
            return True, np.full((12, 16, 3), 100, np.uint8)

    class Full:
        def __init__(self):
            self.value = 4

        def manual_exposure(self, units):
            return units

        def gain_range(self):
            return 0, 64

        def gain(self):
            return self.value

        def set_gain(self, value):
            self.value = value

    class NoGain(Full):
        def gain_range(self):
            raise ControlError("the camera refused control 0x4 of entity 2")

    class NoExposure(Full):
        def manual_exposure(self, units):
            raise ControlError("the camera refused control 0x4 of entity 1")

    held, has_gain, said = head_cam.pin_exposure(Cap(), Full(), 10000, 160)
    check("a camera with a gain has it fitted", has_gain and "gain" in said, said)
    check("the exposure is held in the camera's own count",
          held == round(10000 / SETTINGS.top_exposure_unit_us), str(held))
    held, has_gain, said = head_cam.pin_exposure(Cap(), NoGain(), 10000, 160)
    check("a camera without a gain still opens, at its exposure", not has_gain and "no gain" in said, said)
    try:
        head_cam.pin_exposure(Cap(), NoExposure(), 10000, 160)
        refused = ""
    except record.RecorderError as exc:
        refused = str(exc)
    check("a camera whose exposure cannot be held is refused, in words", "no manual exposure" in refused,
          refused[:80])



def main() -> int:
    check_geometry()
    with tempfile.TemporaryDirectory() as tmp:
        check_stl(Path(tmp))
    check_detection()
    check_end_to_end()
    check_closure()
    check_filter()
    check_gripper_view()
    check_redone_corrections()
    with tempfile.TemporaryDirectory() as tmp:
        check_outcome(Path(tmp))
    check_marker_meta()
    check_top_light()

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)}")
        for name in FAILURES:
            print(f"  - {name}")
        print()
        return 1
    print("All offline checks passed.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
