#!/usr/bin/env python3
"""Offline check of painting a policy's pictures: which masker a checkpoint
is painted with - none for a dataset that kept its backgrounds or with off
chosen, the one chosen, else the newest that finds its takes' things, an
error where none does, the one chosen is gone or its weights are not here -
and one painted picture: the masker's things and the gripper kept,
the lens border kept, the rest the pattern. No masker process, no camera,
no policy.

    python src/pinkrab_policy/test/painting_check.py
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import threading
from pathlib import Path
from unittest import mock

import numpy as np

from pinkrab_core.dataset import lens_centre, lens_fit
from pinkrab_core.kinematics import ArmKinematics
from pinkrab_core.origin import ready_joints
from pinkrab_policy import painting
from pinkrab_umi import keep_masks
from pinkrab_umi.config import SETTINGS as UMI

FAILURES: list[str] = []
THINGS = ["gray cutting mat", "black cube"]


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def masker_for(root: Path, randomised: bool, things: list[str] | None, weights: bool,
               choice: str = "auto") -> tuple[dict | None, str]:
    """masker_for over a dataset of one take in `root`, saying `things`, two
    masker records for THINGS - m1 the older - with or without m2's weights
    here, and `choice` the masker chosen for them."""
    dataset, episodes, maskers = root / "dataset", root / "episodes", root / "maskers"
    (dataset / "meta").mkdir(parents=True, exist_ok=True)
    (dataset / "meta" / "pinkrab_convert.json").write_text(json.dumps(
        {"randomised": randomised, "episodes": [{"episode": "take"}]}))
    (episodes / "take" / "masks").mkdir(parents=True, exist_ok=True)
    (episodes / "take" / "masks" / "masks.json").write_text(json.dumps({"things": things} if things else {}))
    for made, name in enumerate(("m1", "m2")):
        (maskers / name).mkdir(parents=True, exist_ok=True)
        (maskers / name / "masker.json").write_text(json.dumps(
            {"id": name, "things": THINGS, "machine": "kt", "iou": {text: 0.95 for text in THINGS}, "made": float(made), "resolution": 432,
             "model": "RFDETRSegMedium", "weights": "/elsewhere/checkpoint_best_total.pth"}))
        (maskers / name / "checkpoint_best_total.pth").touch()
    if not weights:
        (maskers / "m2" / "checkpoint_best_total.pth").unlink()
    with mock.patch("pinkrab_core.demo_start.dataset_of_checkpoint", return_value=dataset), \
         mock.patch("pinkrab_core.inference.masker_choice", return_value=choice), \
         mock.patch.object(type(UMI), "episodes_dir", new_callable=mock.PropertyMock, return_value=episodes), \
         mock.patch.object(type(UMI), "maskers_dir", new_callable=mock.PropertyMock, return_value=maskers):
        try:
            return painting.masker_for("checkpoint"), ""
        except painting.PaintingError as exc:
            return None, str(exc)


class Reader:
    def __init__(self, picture: np.ndarray) -> None:
        self.picture = picture

    def newest(self):
        return self.picture, 1, 0.0

    def read(self):
        return self.picture


class Writer:
    def __init__(self, stop: threading.Event) -> None:
        self.stop, self.out = stop, None

    def write(self, picture, stamp) -> None:
        self.out = picture
        self.stop.set()


def painted_once(side: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One picture painted by a Painting with no process behind it: the
    camera's frame, the masker's labels, the pattern, and what came out."""
    centre = lens_centre(f"follower_{side}")
    lens = keep_masks.Lens({"lens": lens_fit(f"follower_{side}"), "centre": list(centre)})
    h, w = lens.size
    # The camera's frame at the lens fit's size, so the cut takes the window.
    cols, rows = lens_fit(f"follower_{side}")["size"]
    frame = np.full((rows, cols, 3), 200, dtype=np.uint8)
    labels = np.zeros((h, w), dtype=np.uint8)
    labels[h // 2 - 20:h // 2 + 20, w // 2 - 20:w // 2 + 20] = keep_masks.FIRST_THING + 1
    p = object.__new__(painting.Painting)
    p.hands, p.centres, p.lenses = [side], {side: centre}, {side: lens}
    p.cameras, p.masks = {side: Reader(frame)}, {side: Reader(labels[:, :, None])}
    p._stop, p._lock, p._newest, p._trouble = threading.Event(), threading.Lock(), {}, ""
    p.shown = {side: Writer(p._stop)}
    p.chains = {s: ArmKinematics(s) for s in ("left", "right")}
    p.backdrops = {side: np.zeros((h, w, 3), dtype=np.uint8)}
    border = np.zeros((h, w), dtype=bool)
    border[:, :3] = True
    p.borders = {side: border}
    p._joints = {n: math.radians(v) for s, c in p.chains.items() for n, v in zip(c.joint_names, ready_joints(s, c.joint_names))}
    p._paint()
    return frame, labels, border, p.shown[side].out


def main() -> int:
    print("\nwhich masker")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        record, why = masker_for(root, False, THINGS, True)
        check("a dataset that kept its backgrounds is shown its pictures as they are", record is None and not why)
        record, why = masker_for(root, True, THINGS, True)
        check("painted takes, nothing chosen: the newest masker that finds their things",
              record is not None and record["id"] == "m2", why)
        record, why = masker_for(root, True, THINGS, True, "m1")
        check("one chosen: that one", record is not None and record["id"] == "m1", why)
        record, why = masker_for(root, True, THINGS, True, "off")
        check("off chosen: the pictures as they are", record is None and not why, why)
        record, why = masker_for(root, True, THINGS, True, "m9")
        check("the one chosen is gone: said", record is None and "m9, is gone" in why, why)
        record, why = masker_for(root, True, ["white bowl"], True)
        check("no masker finds them: said", record is None and "no masker finds white bowl" in why, why)
        record, why = masker_for(root, True, THINGS, False)
        check("its weights not here: said", record is None and "on kt" in why, why)
        record, why = masker_for(root, True, None, True)
        check("takes that say nothing: said", record is None and "not here" in why, why)

    print("\none painted picture")
    frame, labels, border, out = painted_once("right")
    h, w = labels.shape
    check("as big as the lens's window", out.shape == (h, w, 3), str(out.shape))
    check("the masker's thing is kept", (out[h // 2, w // 2] == 200).all())
    check("the lens border is kept", (out[h // 2, 0] == 200).all())
    chains = {s: ArmKinematics(s) for s in ("left", "right")}
    tools = {s: c.forward(ready_joints(s, c.joint_names)) for s, c in chains.items()}
    own = keep_masks.drawn("right", keep_masks.Lens({"lens": lens_fit("follower_right"),
                                                     "centre": list(lens_centre("follower_right"))}),
                           tools, {s: keep_masks.jaw_angle(s, None) for s in chains})
    grown = keep_masks.SETTINGS.keep_grow_px + 2
    far = np.ones_like(labels, dtype=bool)
    far[:, :3 + grown] = False
    far[h // 2 - 20 - grown:h // 2 + 20 + grown, w // 2 - 20 - grown:w // 2 + 20 + grown] = False
    import cv2
    near_own = cv2.dilate((own != 0).astype(np.uint8), np.ones((2 * grown + 1, 2 * grown + 1), np.uint8)) != 0
    check("its own gripper is drawn", (own == keep_masks.OWN).any())
    check("the grippers are kept", (out[own != 0] == 200).all())
    check("the rest is the pattern", (out[far & ~near_own] == 0).all())
    print(f"\n{'all passed' if not FAILURES else f'{len(FAILURES)} failed: ' + ', '.join(FAILURES)}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
