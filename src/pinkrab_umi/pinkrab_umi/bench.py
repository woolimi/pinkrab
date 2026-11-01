"""Is the pipeline good enough to record demonstrations with?

One number answers that, and it is not any stage's own residual. Reprojection
error says the marker fit the corners; it says nothing about a focal length that
is a percent out, or about the desk solve leaning a degree. Take the
gripper off a fixture, move it around for thirty seconds, put it back on the same
fixture: whatever pose difference comes out is the whole error budget, measured
end to end, with no ground truth needed beyond a lump of printed plastic that does
not move.

Beside it, coverage - how much of the take actually solved. A closure of two
millimetres over a take where the solid was visible in a third of the frames is not
a good take, it is a short one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import pnp, record
from .config import SETTINGS
from .filtering import PoseFilter
from .trajectory import compose, rotation_angle_deg


@dataclass
class Closure:
    """How far a take that came home did not come home."""

    translation_mm: float
    rotation_deg: float
    frames: int

    @property
    def passes(self) -> bool:
        return (self.translation_mm <= SETTINGS.closure_translation_mm
                and self.rotation_deg <= SETTINGS.closure_rotation_deg)


@dataclass
class Coverage:
    """How much of the take the solid was solvable in."""

    frames: int
    solved: int
    composed: int
    mean_faces: float
    mean_reproj_px: float
    over_reproj: int
    face_counts: dict[int, int] = field(default_factory=dict)

    @property
    def fraction(self) -> float:
        return self.composed / self.frames if self.frames else 0.0

    @property
    def passes(self) -> bool:
        return self.fraction >= SETTINGS.min_coverage


def closure_error(poses: list[np.ndarray]) -> Closure:
    """Start pose against end pose, as millimetres and degrees.

    Taken as a relative transform rather than by differencing the two translations,
    because a pose is a rotation as well and the two errors are not separable
    afterwards: an orientation error at the start puts the solid's origin somewhere
    else even when the fixture held it perfectly.
    """
    if len(poses) < 2:
        return Closure(translation_mm=float("nan"), rotation_deg=float("nan"),
                       frames=len(poses))
    relative = np.linalg.inv(poses[0]) @ poses[-1]
    return Closure(
        translation_mm=float(np.linalg.norm(relative[:3, 3]) * 1000.0),
        rotation_deg=rotation_angle_deg(relative),
        frames=len(poses),
    )


def solve_session(root: Path) -> tuple[list[float], list[np.ndarray], Coverage]:
    """Run every frame of a recorded session through detection, the desk solid, and composition."""
    root = Path(root)
    frames = record.load_frames(root)
    calibration = json.loads(record.SessionPaths(root).calibration_file.read_text())
    intrinsics = record.camera_of(calibration)

    def picture(entry: dict, key: str, flag: int) -> np.ndarray:
        if not entry.get(key):
            raise record.RecorderError(f"{root}: a frame has no {key} image, and the "
                                       "session was recorded with one")
        image = cv2.imread(str(root / entry[key]), flag)
        if image is None:
            raise record.RecorderError(f"{root / entry[key]} could not be read")
        return image

    stamps: list[float] = []
    cam_poses: list[np.ndarray | None] = []
    results: list[pnp.SolidPose] = []
    posefilter = PoseFilter()
    deskfilter = PoseFilter()
    desk_cfg = SETTINGS.for_solid(SETTINGS.desk_solid)
    for entry in frames:
        # The desk solid is read off the same picture as the hand's, so the
        # camera's place on the desk and the hand's place are one instant.
        stamp = float(entry.get("host_ts", entry["sensor_ts"]))
        image = picture(entry, "picture" if "picture" in entry else "color", cv2.IMREAD_UNCHANGED)
        grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        seen = pnp.read_markers(grey)
        raw = pnp.solve_frame(grey, intrinsics, detections=seen)
        desk = deskfilter.apply(pnp.solve_frame(grey, intrinsics, desk_cfg, seen), stamp)
        result = posefilter.apply(raw, stamp)
        results.append(result)
        stamps.append(stamp)
        placed = desk.ok and not desk.reason and desk.pose is not None
        cam_poses.append(compose(np.linalg.inv(desk.pose), result.pose)
                         if result.ok and placed else None)

    # Frames the desk solid could not place are holes, not guesses.
    times = [t for t, p in zip(stamps, cam_poses) if p is not None]
    poses = [p for p in cam_poses if p is not None]
    return times, poses, coverage(results, composed=len(poses))


def coverage(results: list[pnp.SolidPose], composed: int | None = None) -> Coverage:
    solved = [r for r in results if r.ok]
    counts: dict[int, int] = {}
    for result in solved:
        for marker_id in result.faces:
            counts[marker_id] = counts.get(marker_id, 0) + 1
    return Coverage(
        frames=len(results),
        solved=len(solved),
        composed=len(solved) if composed is None else composed,
        mean_faces=float(np.mean([len(r.faces) for r in solved])) if solved else 0.0,
        mean_reproj_px=float(np.mean([r.reproj_px for r in solved])) if solved else float("nan"),
        over_reproj=sum(1 for r in solved if r.reproj_px > SETTINGS.max_reproj_px),
        face_counts=dict(sorted(counts.items())),
    )


def report(closure: Closure, cover: Coverage, root: Path) -> int:
    """Print the verdict. Returns a process exit code, so a bad take fails a script."""
    faces = SETTINGS.id_faces
    print(f"\nsession   {root}")
    ids = sorted(SETTINGS.for_solid(SETTINGS.desk_solid).face_ids.values())
    print(f"world     the desk solid, ids {ids[0]}-{ids[-1]}")

    solved_fraction = cover.solved / cover.frames if cover.frames else 0.0
    print("\ncoverage")
    print(f"  frames                 {cover.frames}")
    print(f"  solid solved            {cover.solved}  ({solved_fraction:.1%})")
    print(f"  composed to world      {cover.composed}  ({cover.fraction:.1%})"
          f"  {'ok' if cover.passes else 'FAIL'}"
          f"  threshold {SETTINGS.min_coverage:.0%}")
    print(f"  faces per solve        {cover.mean_faces:.2f}")
    print(f"  reprojection           {cover.mean_reproj_px:.3f} px mean, "
          f"{cover.over_reproj} frames over {SETTINGS.max_reproj_px:g}")
    for marker_id, count in cover.face_counts.items():
        print(f"    id {marker_id} {faces.get(marker_id, '?'):<8} {count}")

    print("\nclosure  (start pose against end pose)")
    if np.isnan(closure.translation_mm):
        print("  not enough composed frames to close a loop")
        return 1
    print(f"  translation            {closure.translation_mm:8.2f} mm"
          f"  {'ok' if closure.translation_mm <= SETTINGS.closure_translation_mm else 'FAIL'}"
          f"  threshold {SETTINGS.closure_translation_mm:g} mm")
    print(f"  rotation               {closure.rotation_deg:8.2f} deg"
          f"  {'ok' if closure.rotation_deg <= SETTINGS.closure_rotation_deg else 'FAIL'}"
          f"  threshold {SETTINGS.closure_rotation_deg:g} deg")

    passed = closure.passes and cover.passes
    # Only meaningful if the take was actually recorded as a closed loop; the
    # verdict says so rather than pretending to know.
    print(f"\n{'PASS' if passed else 'FAIL'} - assuming this take began and ended "
          "on the same fixture\n")
    return 0 if passed else 1


def evaluate(root: Path) -> int:
    _, poses, cover = solve_session(Path(root))
    return report(closure_error(poses), cover, Path(root))
