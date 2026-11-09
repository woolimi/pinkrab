"""The task's masker on the arms' cameras, live: each camera's newest frame
labelled as fast as the model goes, for a policy shown painted pictures.

Started by pinkrab_policy.painting with the masker environment's Python, this
checkout's pinkrab_core and pinkrab_umi on its path (PYTHONPATH):

    python -P -m pinkrab_umi.live_mask '<json>'

{"cameras": {"follower_left": {"side": "left", "centre": [x, y], "size": [h, w]}, ...},
 "weights": path, "model": "RFDETRSegMedium", "resolution": 432,
 "labels": [3, 4, 5], "score": 0.5}

Each camera's newest frame comes from its shared block (frame_share), cut as
the dataset cuts it (gripper_view) at `size`, and the cameras go to the model
together, one batch; each one's labels - one channel, labels[k] where the
model's class k is found, as masker_run labels a video - go to a block of
their own, live_masks_camera of its side, stamped with the frame's arrival.
A frame already labelled is not labelled again. It says {"ready": true} on
stdout once the model is loaded, and ends when its stdin closes: the process
that started it gone.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time

import cv2

from pinkrab_core.dataset import gripper_view

from .config import SETTINGS
from .frame_share import FrameReader, FrameWriter
from .masker_run import found_in, labels_from, ready

# How long it waits for a camera that has nothing new, in seconds.
IDLE_S = 0.01


def end_with_stdin() -> None:
    """The process ended once stdin closes, whatever it is doing."""
    def read() -> None:
        for _ in sys.stdin:
            pass
        os._exit(0)
    threading.Thread(target=read, daemon=True).start()


def main(ask: dict) -> None:
    end_with_stdin()
    model = ready(ask["weights"], ask["model"], int(ask["resolution"]))
    labels_of = [int(v) for v in ask["labels"]]
    score = float(ask["score"])
    cameras = ask["cameras"]
    readers = {camera: FrameReader(camera) for camera in cameras}
    writers = {camera: FrameWriter(SETTINGS.live_masks_camera.format(side=spec["side"]))
               for camera, spec in cameras.items()}
    print(json.dumps({"ready": True}), flush=True)
    seen: dict[str, int] = {}
    while True:
        batch, taken = [], []
        for camera, spec in cameras.items():
            try:
                picture, count, stamp = readers[camera].newest()
            except RuntimeError:
                continue    # no picture now: the camera's block says why to whoever reads it
            if seen.get(camera) == count:
                continue
            seen[camera] = count
            view = gripper_view(picture, tuple(spec["size"]), tuple(spec["centre"]))
            batch.append(cv2.cvtColor(view, cv2.COLOR_BGR2RGB))
            taken.append((camera, stamp))
        if not batch:
            time.sleep(IDLE_S)
            continue
        for (camera, stamp), detections in zip(taken, found_in(model, batch, score)):
            labels = labels_from(detections, labels_of, cameras[camera]["size"])
            writers[camera].write(labels[:, :, None], stamp)


if __name__ == "__main__":
    main(json.loads(sys.argv[1]))
