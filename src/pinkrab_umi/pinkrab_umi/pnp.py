"""Where the solid is, relative to the camera, in one colour frame.

Every detected face is solved together rather than one at a time and averaged.
Two reasons. A single square is planar, so its pose has a twofold ambiguity that
noise flips between - the classic marker jitter, where a face viewed near
face-on suddenly reports itself tilted the other way. And the faces of a solid are
not coplanar, so two of them pin the pose outright: the ambiguity is gone, not
averaged down.

Every face that decodes is kept, however obliquely it is seen. A face at seventy
degrees off square places the pose worse than one seen head-on, but it places it
better than nothing: measured on rendered sweeps, dropping faces past any angle
raised the error, and the frames that swing furthest are the ones down to a
single face. What is dropped is disagreement. After the joint solve each face's
own residual is compared with the rest, and a face at odds with them - a patch
glued a hair off, a corner refined onto the pocket's edge - is left out and the
others solved again.

What comes back is cam_T_solid - the solid's frame expressed in the camera's, which
is the direction trajectory.py composes in. The camera is the top camera and
nothing else: every pose in this package is in its frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import markers
from .config import SETTINGS, Settings


@dataclass(frozen=True)
class Intrinsics:
    """A pinhole camera. Metres in, pixels out."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    coeffs: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @property
    def distortion(self) -> np.ndarray:
        return np.array(self.coeffs, dtype=np.float64)

    @classmethod
    def from_dict(cls, data: dict) -> Intrinsics:
        return cls(
            width=int(data["width"]), height=int(data["height"]),
            fx=float(data["fx"]), fy=float(data["fy"]),
            cx=float(data["cx"]), cy=float(data["cy"]),
            coeffs=tuple(float(c) for c in data.get("coeffs", (0.0,) * 5)),
        )

    def to_dict(self) -> dict:
        return {"width": self.width, "height": self.height, "fx": self.fx, "fy": self.fy,
                "cx": self.cx, "cy": self.cy, "coeffs": list(self.coeffs)}


@dataclass
class SolidPose:
    """One frame's answer, with enough beside it to tell a good frame from a bad one.

    `ok` is False for a frame with nothing usable in it, and such a frame is a
    hole in the demonstration rather than an error - a hand turns the solid away
    sometimes. bench counts them.
    """

    ok: bool
    pose: np.ndarray | None = None            # 4x4 cam_T_solid, metres
    # The pose before the stream filter blended it toward the frames before: the
    # solution it chose, un-averaged. None where nothing was blended.
    unblended: np.ndarray | None = None
    faces: tuple[int, ...] = ()               # marker ids that contributed
    reproj_px: float = float("nan")           # RMS over the contributing corners
    detected: tuple[int, ...] = ()            # ids seen at all, including rejected ones
    # Where those ids were, four image corners each, in the order `detected` lists
    # them. The solve has no use for them - it is carrying them for whoever wants
    # to draw the detection back over the frame it came from, which is the only way
    # to see that a face is being found and then thrown away.
    corners: tuple[np.ndarray, ...] = ()
    # A single face is a plane, and a plane fits two poses. This is the one the
    # solver did not pick, with its own residual beside `reproj_px` - carried out
    # because only somebody watching the stream over time can tell which of the
    # two the solid is actually in. Multi-face solves have no second answer.
    alternate: np.ndarray | None = None
    alternate_reproj_px: float = float("nan")
    reason: str = ""

    @property
    def held(self) -> np.ndarray | None:
        """The pose as the stream filter chose it, not averaged with the frames
        before: what a recording keeps, for a later pass to smooth without lag."""
        return self.pose if self.unblended is None else self.unblended


# The AprilTag library names its families for the payload and the code distance:
# tag25h9 is 25 data bits, five by five, and no two codes closer than nine bit
# flips. OpenCV names the same families the other way round, so this is the map.
APRILTAG_FAMILY = {
    "DICT_APRILTAG_16h5": "tag16h5",
    "DICT_APRILTAG_25h9": "tag25h9",
    "DICT_APRILTAG_36h10": "tag36h10",
    "DICT_APRILTAG_36h11": "tag36h11",
}


def _detector():
    """The reader named in the settings, built for the dictionary named there."""
    if SETTINGS.detector == "aruco":
        params = cv2.aruco.DetectorParameters()
        # Corners refined against the image gradient rather than left at the
        # quadrilateral fit. This is most of the pose accuracy at these marker
        # sizes, and it is off by default.
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        return cv2.aruco.ArucoDetector(markers.dictionary(), params)

    if SETTINGS.detector != "apriltag":
        raise ValueError(f"no such detector: {SETTINGS.detector!r} - "
                         "settings.detector is 'aruco' or 'apriltag'")
    family = APRILTAG_FAMILY.get(SETTINGS.aruco_dict)
    if family is None:
        raise ValueError(
            f"the apriltag reader cannot read {SETTINGS.aruco_dict} - it reads the "
            f"AprilTag families only ({', '.join(sorted(APRILTAG_FAMILY))})"
        )
    try:
        import pupil_apriltags
    except ImportError as exc:  # pragma: no cover - environment, not logic
        raise ImportError(
            "settings.detector is 'apriltag' but pupil-apriltags is not installed - "
            "pip install pupil-apriltags, or set detector to 'aruco'"
        ) from exc
    return pupil_apriltags.Detector(families=family, nthreads=4,
                                    quad_decimate=1.0, refine_edges=True)


_DETECTOR = None
_DETECTOR_FOR: tuple[str, str] | None = None


def detector():
    """One detector for the process, rebuilt only when the settings name another.

    Building it per frame costs more than the solve, and the settings can change
    under a test that is comparing the two readers on the same frames.
    """
    global _DETECTOR, _DETECTOR_FOR
    want = (SETTINGS.detector, SETTINGS.aruco_dict)
    if _DETECTOR is None or _DETECTOR_FOR != want:
        _DETECTOR, _DETECTOR_FOR = _detector(), want
    return _DETECTOR


def _read(gray: np.ndarray):
    """(ids, quads, rejected quads) from whichever reader is configured, in OpenCV's corner order.

    The rejected quads are squares the reader found but could not decode - a
    face at the edge of reading, which is the face that comes and goes between
    frames. They are kept for `_recover`, which asks the solid where such a
    face ought to be. The AprilTag reader keeps no such list.

    ArUco reports a marker's corners clockwise from its top-left as printed;
    AprilTag reports the same four points the other way round and starting two
    along. The object points are listed in ArUco's order, so AprilTag's are put
    into it here rather than anywhere the geometry would have to branch.
    """
    reader = detector()
    if SETTINGS.detector == "aruco":
        corners, ids, rejected = reader.detectMarkers(gray)
        rejected = tuple(q.reshape(4, 2).astype(np.float64) for q in (rejected or ()))
        if ids is None or len(ids) == 0:
            return (), (), rejected
        return (tuple(int(i) for i in ids.flatten()),
                tuple(q.reshape(4, 2).astype(np.float64) for q in corners), rejected)
    found = reader.detect(gray)
    if not found:
        return (), (), ()
    return (tuple(int(d.tag_id) for d in found),
            tuple(np.roll(np.asarray(d.corners, dtype=np.float64)[::-1], 2, axis=0)
                  for d in found), ())


def _quad_area(corners: np.ndarray) -> float:
    x, y = corners[:, 0], corners[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


@dataclass(frozen=True)
class _Solve:
    """One pass of the solver over a set of faces."""

    pose: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    reproj_px: float
    per_face_px: tuple[float, ...]
    alternate: np.ndarray | None = None
    alternate_reproj_px: float = float("nan")


def _solve_faces(used: list[int], obj: list[np.ndarray], img: list[np.ndarray],
                 intrinsics: Intrinsics, cfg: Settings = SETTINGS) -> _Solve | None:
    """One solve over these faces. `cfg` is the solid they belong to: a lone face
    is solved in its own plane and walked back into the solid by that solid's
    face table, and another solid's ids are not in the default one."""
    obj_arr = np.concatenate(obj).reshape(-1, 1, 3)
    img_arr = np.concatenate(img).reshape(-1, 1, 2)

    alternate = None
    alternate_reproj = float("nan")
    if len(used) == 1:
        # One face is a planar square, and IPPE_SQUARE is the solver written for
        # exactly that - but it wants the canonical square in its own plane, not
        # the same square carried out to a solid face. So it is solved in the
        # face's frame and the answer walked back into the solid's. Generic rather
        # than plain solvePnP because a plane fits two poses, and the loser is
        # kept: near face-on the two residuals converge and the winner is decided
        # by noise, so the choice belongs to whoever can see the frames around
        # this one, not to this function.
        count, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            markers.marker_corners_face(cfg).reshape(-1, 1, 3), img_arr,
            intrinsics.matrix, intrinsics.distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if count == 0:
            return None

        face = cfg.id_faces[used[0]]
        face_T_solid = np.linalg.inv(markers.face_poses(cfg)[face])

        def solid_pose_of(rv, tv) -> np.ndarray:
            cam_T_face = np.eye(4)
            cam_T_face[:3, :3] = cv2.Rodrigues(rv)[0]
            cam_T_face[:3, 3] = tv.ravel()
            return cam_T_face @ face_T_solid

        # Solutions come back best-residual first.
        cam_T_solid = solid_pose_of(rvecs[0], tvecs[0])
        if count > 1:
            alternate = solid_pose_of(rvecs[1], tvecs[1])
            # errs is (N, 1); numpy will not make a scalar of a one-element row.
            alternate_reproj = float(np.ravel(errs)[1])
        rvec = cv2.Rodrigues(cam_T_solid[:3, :3])[0]
        tvec = cam_T_solid[:3, 3].reshape(3, 1)
    else:
        # Two or more faces are not coplanar, so the planar ambiguity is gone.
        # SQPnP takes them without an initial guess.
        ok, rvec, tvec = cv2.solvePnP(
            obj_arr, img_arr, intrinsics.matrix, intrinsics.distortion,
            flags=cv2.SOLVEPNP_SQPNP,
        )
        if not ok:
            return None

    # SQPnP and IPPE both minimise an algebraic cost; this is the geometric one.
    rvec, tvec = cv2.solvePnPRefineLM(
        obj_arr, img_arr, intrinsics.matrix, intrinsics.distortion, rvec, tvec
    )

    projected, _ = cv2.projectPoints(
        obj_arr, rvec, tvec, intrinsics.matrix, intrinsics.distortion
    )
    residual = np.sum((projected.reshape(-1, 2) - img_arr.reshape(-1, 2)) ** 2, axis=1)
    reproj = float(np.sqrt(np.mean(residual)))
    per_face = tuple(float(np.sqrt(np.mean(residual[i:i + 4])))
                     for i in range(0, len(residual), 4))

    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(rvec)[0]
    pose[:3, 3] = tvec.ravel()
    return _Solve(pose=pose, rvec=rvec, tvec=tvec, reproj_px=reproj, per_face_px=per_face,
                  alternate=alternate, alternate_reproj_px=alternate_reproj)


def _without_worst(solve: _Solve, used: list[int], resolve) -> tuple[list[int], _Solve] | None:
    """The solve with its worst-fitting face left out, if that face was the problem.

    A patch glued a hair off, a corner refinement that latched onto the pocket's
    edge, a detection of a neighbouring face's id: each is one face wrong while
    the rest agree. Its own residual does not say so - the joint solve spreads
    the disagreement over every face, and the worst one's share stays about
    twice the others' however far off it is. Leaving it out does say so: the
    residual of the rest collapses, where dropping a good face barely moves it.
    Needs three faces - with two, the disagreement has no side to take - and a
    residual worth explaining, so a clean frame never pays for the second solve.
    """
    if len(used) < 3 or solve.reproj_px <= SETTINGS.face_outlier_min_px:
        return None
    worst = int(np.argmax(solve.per_face_px))
    rest = [m for i, m in enumerate(used) if i != worst]
    again = resolve(rest)
    if again is None or again.reproj_px > SETTINGS.face_outlier_gain * solve.reproj_px:
        return None
    return rest, again


def read_markers(gray: np.ndarray):
    """Every marker id in one picture and its quad - the detector, and nothing else.

    Separate from `faces_in` because the detector is the expensive half and it
    does not care which solid is being asked about. Two handhelds in one room are
    two sets of ids in the same picture: decoding once and asking each solid which
    of them are its own costs one detection rather than two.
    """
    if gray.ndim != 2:
        raise ValueError(f"expected a grayscale frame, got shape {gray.shape}")
    return _read(gray)


def faces_in(gray: np.ndarray, cfg: Settings = SETTINGS, detections=None):
    """One image, decoded: (every id seen, their quads, and the solid's own faces).

    The last two are keyed by marker id and hold only faces this solid owns and
    presents big enough to fix a pose - a marker from some other board in the
    room is dropped, and so is one so nearly edge-on that its four corners are
    almost collinear. `cfg` is which solid is being asked about: the ids belong
    to one solid's face table, and another solid's ids are simply not in it.

    `detections` is a decoded picture from `read_markers`, when the caller has
    one already.
    """
    detected, quads, *_ = read_markers(gray) if detections is None else detections
    if not detected:
        return (), (), {}, {}
    object_points = markers.solid_object_points(cfg)
    obj: dict[int, np.ndarray] = {}
    img: dict[int, np.ndarray] = {}
    for marker_id, quad in zip(detected, quads):
        if marker_id not in object_points:
            continue
        if _quad_area(quad) < SETTINGS.min_face_area_px:
            continue
        obj[marker_id] = object_points[marker_id]
        img[marker_id] = quad
    return detected, quads, obj, img


_BOARDS: dict[str, object] = {}


def _board(cfg: Settings):
    """This solid as an OpenCV board: its faces' corners in its own frame, by id."""
    key = f"{cfg.solid}:{cfg.aruco_dict}"
    if key not in _BOARDS:
        points = markers.solid_object_points(cfg)
        ids = sorted(points)
        _BOARDS[key] = cv2.aruco.Board(
            np.array([points[i] for i in ids], dtype=np.float32),
            markers.dictionary(), np.array(ids, dtype=np.int32).reshape(-1, 1))
    return _BOARDS[key]


def _recover(gray: np.ndarray, cfg: Settings, obj: dict, img: dict, rejected,
             intrinsics: Intrinsics):
    """Read the faces that would not decode, from where the decoded ones put them.

    A face at the edge of reading - small, oblique, in shadow - is found as a
    square and thrown away as a marker on one frame and kept on the next, so
    the solid's face count flickers and the pose with it. OpenCV's board
    refinement takes the faces that did decode, poses the whole solid from
    them, projects the faces that did not, and matches each projection to a
    rejected square close enough to be it; the square's corners are then read
    with the id the geometry says. What comes back is the fuller detection,
    (ids, quads, what is still rejected), or None when nothing was recovered.
    """
    ids = sorted(obj)
    corners = [img[i].reshape(1, 4, 2).astype(np.float32) for i in ids]
    rej = [q.reshape(1, 4, 2).astype(np.float32) for q in rejected]
    try:
        out = detector().refineDetectedMarkers(
            gray, _board(cfg), corners, np.array(ids, dtype=np.int32).reshape(-1, 1), rej,
            intrinsics.matrix, intrinsics.distortion)
    except cv2.error:
        return None
    new_corners, new_ids, still_rejected, recovered = out
    if new_ids is None or recovered is None or len(recovered) == 0:
        return None
    return (tuple(int(i) for i in new_ids.flatten()),
            tuple(q.reshape(4, 2).astype(np.float64) for q in new_corners),
            tuple(q.reshape(4, 2).astype(np.float64) for q in (still_rejected or ())))


def solve_frame(gray: np.ndarray, intrinsics: Intrinsics,
                cfg: Settings = SETTINGS, detections=None) -> SolidPose:
    """Detect the solid's faces in one grayscale frame and solve for its pose.

    One camera. A monocular solve reads depth off how big the marker looks and
    carries the whole error of the printed size on the axis pointing away from
    the camera; the solid's faces seen together give that back in part, since
    faces fix depth by their angle to each other.

    The detector wants a single channel; a colour frame is converted by the
    caller rather than here, so nothing silently accepts the wrong stream.
    """
    detections = read_markers(gray) if detections is None else detections
    detected, quads, obj, img = faces_in(gray, cfg, detections)
    if not detected:
        return SolidPose(ok=False, reason="no markers detected")
    if obj and SETTINGS.recover_faces and len(detections) > 2 and detections[2]:
        # Faces the reader saw as squares but could not read are asked for
        # again where the solid's own pose says they must be.
        recovered = _recover(gray, cfg, obj, img, detections[2], intrinsics)
        if recovered is not None:
            detected, quads, obj, img = faces_in(gray, cfg, recovered)

    used = list(obj)
    if not used:
        return SolidPose(ok=False, detected=detected, corners=quads,
                         reason="no solid face big enough to solve")

    def solve(faces: list[int]) -> _Solve | None:
        return _solve_faces(faces, [obj[m] for m in faces], [img[m] for m in faces],
                            intrinsics, cfg)

    result = solve(used)
    if result is None:
        return SolidPose(ok=False, detected=detected, corners=quads,
                         reason="solvePnP failed")

    # One face at odds with the rest goes, and the rest are asked again, until
    # leaving a face out stops paying.
    while (better := _without_worst(result, used, solve)) is not None:
        used, result = better

    reproj = result.reproj_px
    reason = "" if reproj <= SETTINGS.max_reproj_px else (
        f"reprojection {reproj:.2f} px over {SETTINGS.max_reproj_px:g} - "
        "check the patches are square to the faces and face_size_mm is the solid"
    )
    return SolidPose(ok=True, pose=result.pose, faces=tuple(used), reproj_px=reproj,
                     detected=detected, corners=quads, alternate=result.alternate,
                     alternate_reproj_px=result.alternate_reproj_px, reason=reason)


def solve_colour(bgr: np.ndarray, intrinsics: Intrinsics,
                 cfg: Settings = SETTINGS, detections=None) -> SolidPose:
    """The solid's pose in the camera's frame, read off a colour picture.

    The corners stay in image pixels: they are for drawing back over the
    picture they were found in.
    """
    if bgr.ndim != 3:
        raise ValueError(f"expected a colour frame, got shape {bgr.shape}")
    return solve_frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), intrinsics, cfg, detections)


def _project(pose: np.ndarray, obj: np.ndarray, intrinsics: Intrinsics) -> np.ndarray:
    """Object points through a pose and a camera, as (n, 2) pixels."""
    rvec = cv2.Rodrigues(pose[:3, :3])[0]
    out, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), rvec, pose[:3, 3],
                               intrinsics.matrix, intrinsics.distortion)
    return out.reshape(-1, 2)
