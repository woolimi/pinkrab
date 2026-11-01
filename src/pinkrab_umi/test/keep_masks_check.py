"""Offline check of the keep labels and the backgrounds painted from them:
the robot's grippers read off the description, each camera's own gripper
drawn into its picture where the lens puts it, the review flags, and a
picture painted over everywhere but where its labels keep it. No camera, no
SAM 3.

    python src/pinkrab_umi/test/keep_masks_check.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from pinkrab_core.config import ARMS
from pinkrab_umi import backgrounds, keep_masks

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def lens() -> keep_masks.Lens:
    """A fisheye like the gripper cameras', its axis in the picture's middle."""
    return keep_masks.Lens({"lens": {"model": "kannala_brandt", "fx": 560.0, "fy": 560.0, "cx": 960.0, "cy": 600.0,
                                     "k": [0.0, 0.0, 0.0, 0.0], "size": [1920, 1200]},
                            "centre": [960.0, 600.0]})


def main() -> int:
    print("grippers off the description")
    models, link_T_cam = keep_masks.grippers()
    for side in ARMS:
        g = models[side]
        shut, wide = (keep_masks._jaw_angle(side, o) for o in (0.0, 1.0))
        moved = np.abs(g.triangles(wide) - g.triangles(shut)).max()
        check(f"{side}: parts and a jaw that turns", len(g.fixed) > 0 and len(g.jaw) > 0 and moved > 0.005,
              f"{len(g.fixed)} parts, jaw tip moves {moved * 1000:.0f} mm")
    flip = np.diag([1.0, -1.0, 1.0])
    mirrored = flip @ link_T_cam["right"][:3, 3]
    check("left camera is the right one across the link's y",
          np.allclose(mirrored, link_T_cam["left"][:3, 3], atol=1e-9))

    print("own gripper drawn where the camera sees it")
    view = lens()
    for side in ARMS:
        labels = np.zeros(view.size, dtype=np.uint8)
        tris = keep_masks._moved(models[side].triangles(keep_masks._jaw_angle(side, 0.5)),
                                 np.linalg.inv(link_T_cam[side]))
        view.draw(labels, tris, keep_masks.OWN)
        share = (labels == keep_masks.OWN).mean()
        lower = (labels[labels.shape[0] // 2:] == keep_masks.OWN).sum() / max(1, (labels == keep_masks.OWN).sum())
        check(f"{side}: the claw fills part of the picture, mostly its lower half",
              0.02 < share < 0.6 and lower > 0.7, f"{share:.0%} of the picture, {lower:.0%} of it below the middle")

    print("flags")
    frames = np.zeros((10, 60, 80), dtype=np.uint8)
    frames[:4, 20:30, 30:40] = 3          # found, in the middle ...
    frames[7:, 20:30, 30:40] = 3          # ... lost for 4..6, found again
    frames[:, 0:10, 0:10] = 4             # at the edge throughout
    frames[5:, 0:10, 0:10] = 0            # gone at the edge: left the view, not lost
    spans = keep_masks.flags(frames, ["cube", "bowl"])
    check("a thing gone in the middle of the picture is lost until found again",
          {"thing": "cube", "label": 3, "start": 4, "end": 7, "why": "lost"} in spans, str(spans))
    check("a thing gone at the picture's edge is not flagged", not any(s["thing"] == "bowl" for s in spans), str(spans))
    claw = np.zeros_like(frames)
    claw[4:, 28:45, 28:45] = keep_masks.OWN   # the claw comes over the cube as it goes
    check("a thing gone behind a gripper is not flagged", not keep_masks.flags(frames, ["cube"], claw))
    grown = np.zeros((3, 60, 80), dtype=np.uint8)
    grown[0, 20:24, 20:24] = 3
    grown[1:, 10:40, 10:40] = 3
    cover = np.zeros_like(grown)
    cover[0, 10:40, 25:40] = keep_masks.OWN   # the claw over most of it, then gone
    check("a thing grown out from behind a gripper is not flagged", not keep_masks.flags(grown, ["cube"], cover))
    check("a thing grown many times over between frames is flagged",
          any(s["why"] == "jumped" and s["start"] == 1 for s in keep_masks.flags(grown, ["cube"])))

    print("painted backgrounds")
    a, b = backgrounds.pattern(1, (60, 80)), backgrounds.pattern(2, (60, 80))
    check("a seed is the same pattern every time, another seed another",
          np.array_equal(a, backgrounds.pattern(1, (60, 80))) and not np.array_equal(a, b))
    check("each take has its own seed, the same on every conversion",
          backgrounds.seed("t1") == backgrounds.seed("t1") and len({backgrounds.seed(t) for t in ("t1", "t2", "t3")}) == 3)
    picture = np.full((60, 80, 3), 7, dtype=np.uint8)
    labels = np.zeros((60, 80), dtype=np.uint8)
    labels[20:40, 30:50] = keep_masks.OWN
    out = backgrounds.painted(picture, labels, a)
    check("what is kept stays as recorded", (out[20:40, 30:50] == 7).all())
    check("the rest is the backdrop", np.array_equal(out[:5, :5], a[:5, :5]))
    small = labels[::2, ::2]
    check("labels at another size are scaled to the picture",
          (backgrounds.painted(picture, small, a)[22:38, 32:48] == 7).all())

    print("lens border")
    root = Path(tempfile.mkdtemp())
    size = (60, 80)
    (root / "meta.json").write_text(json.dumps({"streams": {"gripper_left": {"centre": [40.0, 30.0]}}}))
    writer = cv2.VideoWriter(str(root / "gripper_left.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (80, 60))
    ys, xs = np.mgrid[0:60, 0:80]
    circle = (xs - 40) ** 2 + (ys - 30) ** 2 < 28 ** 2
    for k in range(10):
        frame = np.zeros((60, 80, 3), dtype=np.uint8)
        frame[circle] = 120 + 10 * k
        frame[26:34, 36:44] = 0              # a black thing in view, always there
        frame[24:36, 8:22] = 0               # and one in view at the circle's edge, joined to the black
        writer.write(frame)
    writer.release()
    with mock.patch("pinkrab_core.dataset.gripper_view", lambda picture, size, centre: picture):
        edge = keep_masks.border(root, "gripper_left", size)
    check("the dark outside the circle is the border", edge[0, 0] and edge[59, 79] and not edge[30, 20],
          f"{edge.mean():.0%} of the picture")
    check("a dark thing inside the picture is not", not edge[30, 40])
    check("nor is a dark thing at the circle's edge", not edge[30, 16] and not edge[26, 20])

    print(f"\n{'all passed' if not FAILURES else f'{len(FAILURES)} failed: ' + ', '.join(FAILURES)}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
