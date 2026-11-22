"""Offline check of what convert draws into a demonstration's gripper
pictures: the robot's grippers read off the description, each camera's own
gripper drawn where the lens puts it, the lens carried past the plane of its
own glass as a 200 degree fisheye sees, and the robot's arms and board read
off the description and the scene. No camera, no take.

    python src/pinkrab_umi/test/robot_drawn_check.py
"""
from __future__ import annotations

import sys

import cv2
import numpy as np

from pinkrab_core.config import ARMS
from pinkrab_umi import convert, gripper_scene, robot_drawn

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


LENS = {"model": "kannala_brandt", "fx": 560.0, "fy": 560.0, "cx": 960.0, "cy": 600.0,
        "k": [0.02, -0.01, 0.003, -0.0004], "size": [1920, 1200]}


def main() -> int:
    print("grippers off the description")
    models, link_T_cam = gripper_scene.grippers()
    for side in ARMS:
        g = models[side]
        shut, wide = (gripper_scene.jaw_angle(side, o) for o in (0.0, 1.0))
        moved = np.abs(g.triangles(wide) - g.triangles(shut)).max()
        check(f"{side}: parts and a jaw that turns", len(g.fixed) > 0 and len(g.jaw) > 0 and moved > 0.005,
              f"{len(g.fixed)} parts, jaw tip moves {moved * 1000:.0f} mm")
    flip = np.diag([1.0, -1.0, 1.0])
    check("left camera is the right one across the link's y",
          np.allclose(flip @ link_T_cam["right"][:3, 3], link_T_cam["left"][:3, 3], atol=1e-9))

    print("own gripper drawn where the camera sees it")
    stream = {"lens": {**LENS, "k": [0.0] * 4}, "centre": [960.0, 600.0]}
    view = gripper_scene.Lens(stream)
    for side in ARMS:
        labels = np.zeros(view.size, dtype=np.uint8)
        tris = gripper_scene._moved(models[side].triangles(gripper_scene.jaw_angle(side, 0.5)),
                                    np.linalg.inv(link_T_cam[side]))
        view.draw(labels, tris, gripper_scene.OWN)
        own = labels == gripper_scene.OWN
        share = own.mean()
        lower = own[labels.shape[0] // 2:].sum() / max(1, own.sum())
        check(f"{side}: the claw fills part of the picture, mostly its lower half",
              0.02 < share < 0.6 and lower > 0.7, f"{share:.0%} of the picture, {lower:.0%} of it below the middle")

    print("a take recorded before the lens was measured")
    for bare in ({}, {"lens": LENS}, {"centre": [960.0, 600.0]}):
        try:
            gripper_scene.Lens(bare)
            refused = False
        except ValueError:
            refused = True
        check(f"a lens that is not recorded is refused, not a KeyError - {sorted(bare)}", refused)

    print("the tools as convert smooths them")
    times = np.arange(30) / 30.0
    rng = np.random.default_rng(0)
    track = []
    for i in range(30):
        T = np.eye(4)
        T[0, 3] = 0.01 * i + rng.normal(0.0, 0.002)
        track.append(T)
    track[12] = None
    smooth = convert.smoothed(track, times)
    runs = [r for r in (smooth[:12], smooth[13:]) if len(r)]

    def rough(xs):
        return float(np.std(np.diff(np.diff([T[0, 3] for T in xs]))))

    check("a hole stays a hole and each run is smoother than its solves",
          smooth[12] is None and all(rough(r) < rough(track[a:b]) for r, (a, b) in zip(runs, [(0, 12), (13, 30)])))

    print("the lens past the plane of its glass")
    stream = {"lens": LENS, "centre": [960.0, 600.0]}
    optics = robot_drawn.Optics(gripper_scene.Lens(stream), stream)
    lens = gripper_scene.Lens(stream)
    rng = np.random.default_rng(0)
    ahead = rng.uniform([-1, -1, 0.3], [1, 1, 2], (200, 3))
    ours, ok = optics.project(ahead)
    theirs, _ = cv2.fisheye.projectPoints(ahead.reshape(-1, 1, 3), np.zeros((3, 1)), np.zeros((3, 1)), lens.K, lens.D)
    theirs = (theirs.reshape(-1, 2) - [lens.x, lens.y]) * lens.scale
    off = float(np.abs(ours - theirs).max())
    check("in front of the lens, rays land where OpenCV's fisheye puts them", ok.all() and off < 1e-6,
          f"{off:.2g} px apart")
    beside = np.array([[np.sin(a), 0.0, np.cos(a)] for a in np.radians([85.0, 95.0])])
    px, ok = optics.project(beside)
    centre, radius = optics.circle
    reach = np.linalg.norm(px - centre, axis=1)
    check("a ray just past 90 degrees lands further out, inside the 200 degree circle",
          ok.all() and reach[1] > reach[0] and reach[1] < radius, f"{reach.round(1)} px, circle {radius:.1f}")
    behind = np.array([[0.0, 0.0, -1.0]])
    check("a ray straight back is not one the lens takes", not optics.project(behind)[1].any())

    print("the robot off the description and the scene")
    parts = robot_drawn._arm_parts()
    check("each arm's links but its gripper's", all(any(name.startswith(f"{arm}_") for name in parts) for arm in ARMS)
          and not any(name.endswith(robot_drawn.GRIPPER_LINKS) for name in parts), f"{len(parts)} links")
    board, slots = robot_drawn._board()
    face = board[:, :, 0]
    check("the board is a face, its slots a hair in front of it",
          np.ptp(face) < 1e-9 and np.all(slots[:, :, 0] > face.max()) and len(slots) > 0,
          f"{len(board)} triangles, {len(slots) // 2} slots")
    look = robot_drawn.Look.described()
    check("the description colours the printed parts, the servos and the board",
          look.pink != look.servo != look.board and look.slot is None, str(look))

    print(f"\n{'all passed' if not FAILURES else f'{len(FAILURES)} failed: ' + ', '.join(FAILURES)}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
