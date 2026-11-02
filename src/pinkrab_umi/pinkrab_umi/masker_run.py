"""Train a task's masker and label gripper videos with it, in the masker's own environment.

A masker is an RF-DETR segmentation model (Apache-2.0) taught a task's
things from takes SAM 3 has already labelled: once it has seen them in a
few dozen takes, it finds them in a new take's every frame in seconds,
where SAM 3 tracking takes minutes. Run by keep_masks with the Python of the
environment RF-DETR is installed in (umi config masker_env); this file
imports nothing of pinkrab's. Kept running, the model loaded once, it
answers one JSON line after another on stdin:

    python masker_run.py --serve

{"train": dataset, "into": folder, "model": "RFDETRSegMedium", "epochs": 12,
 "resolution": 432} trains on a COCO dataset folder (train/ and valid/, each
with _annotations.coco.json), saying {"epoch": i, "epochs": n, "map50": ...}
after each epoch and {"done": true, "weights": path, ...} at the end;

{"video": mp4, "window": [x, y, w, h], "size": [height, width], "weights":
path, "model": ..., "resolution": ..., "labels": [3, 4, 5], "score": 0.5,
"out": npz} labels every frame of the video, cropped to the dataset's window
and size as sam_track crops it: each pixel the label of the thing found
there (labels[k] for the model's class k, a later thing over an earlier),
0 elsewhere. It says {"frame": i, "frames": n} on the way and {"done": true}
at the end. A request that fails says {"error": "..."}.

Asked over ssh, a video comes in its line as "video_data" (base64), and with
no "out" the npz goes back as "data" in the last line. The service ends when
its asker goes - its stdin closed.
"""
from __future__ import annotations

import base64
import contextlib
import csv
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# How often a masker being taught says it is still going, in seconds.
HEARTBEAT_S = 60.0
# A masker being taught, in its own process: stopped with this one.
TEACHING: subprocess.Popen | None = None
# Where this process's answers go: the real stdout. RF-DETR prints while it
# trains, and that goes to stderr, never into the answers.
OUT = sys.stdout


def say(**fields) -> None:
    print(json.dumps(fields), file=OUT, flush=True)


def model_for(name: str, resolution: int, weights: str | None = None):
    import rfdetr

    kind = getattr(rfdetr, name)
    # A relative path RF-DETR would look for in its own download folder.
    return kind(resolution=resolution, **({"pretrain_weights": str(Path(weights).resolve())} if weights else {}))


def train(ask: dict, tell=say) -> None:
    """A masker trained on ask's dataset, into ask's folder: its best weights,
    by the held-out takes' mask mAP."""
    out = Path(ask["into"]).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    epochs = int(ask["epochs"])
    metrics = out / "metrics.csv"
    done = threading.Event()

    def watch() -> None:
        """Each epoch's held-out scores, said as the training writes them,
        and that it is still going every HEARTBEAT_S: an epoch can take
        longer than the asker waits for a line."""
        said, last = -1, time.monotonic()
        while not done.wait(5.0):
            if time.monotonic() - last > HEARTBEAT_S:
                tell(training=True)
                last = time.monotonic()
            try:
                rows = list(csv.DictReader(metrics.open()))
            except (OSError, csv.Error):
                continue
            for row in rows:
                epoch = int(row.get("epoch") or -1)
                if epoch > said and row.get("val/segm_mAP_50"):
                    said = epoch
                    tell(epoch=epoch + 1, epochs=epochs, map50=round(float(row["val/segm_mAP_50"]), 4),
                         map=round(float(row["val/segm_mAP_50_95"]), 4))
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    started = time.monotonic()
    # Taught in a process of its own: its data loader forks workers, and a
    # fork of this one - its stdin reader and the watcher running - hangs.
    # Its output goes to stderr, never into the answers.
    global TEACHING
    TEACHING = subprocess.Popen([sys.executable, "-I", __file__, "--teach", json.dumps(ask)],
                                stdout=sys.stderr, stderr=sys.stderr)
    try:
        if TEACHING.wait() != 0:
            raise ValueError(f"the teaching stopped (exit {TEACHING.returncode}) - its log is on stderr")
    finally:
        done.set()
        watcher.join()
    # The best weights, by the held-out takes: the EMA's where only it was kept.
    weights = next((out / f for f in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth")
                    if (out / f).is_file()), None)
    if weights is None:
        raise ValueError("the training left no best weights")
    best = {"map50": 0.0, "map": 0.0}
    with contextlib.suppress(OSError, csv.Error, ValueError):
        for row in csv.DictReader(metrics.open()):
            if row.get("val/segm_mAP_50_95") and float(row["val/segm_mAP_50_95"]) >= best["map"]:
                best = {"map50": round(float(row["val/segm_mAP_50"]), 4),
                        "map": round(float(row["val/segm_mAP_50_95"]), 4)}
    tell(done=True, weights=str(weights), seconds=round(time.monotonic() - started), **best)


def teach(ask: dict) -> None:
    """The teaching itself, in the child train() starts."""
    import torch

    model = model_for(ask["model"], int(ask["resolution"]))
    model.train(dataset_dir=str(Path(ask["train"]).expanduser()), output_dir=str(Path(ask["into"]).expanduser()),
                epochs=int(ask["epochs"]), batch_size="auto" if torch.cuda.is_available() else 4,
                num_workers=int(ask.get("workers", 8)), run_test=False, progress_bar=None)


def frames(video: str, window, size):
    """The video's frames as the dataset keeps them: cropped to the window
    and scaled to size, as sam_track's frames_into makes them."""
    x, y, w, h = window
    height, width = size
    cap = cv2.VideoCapture(video)
    try:
        while True:
            ok, picture = cap.read()
            if not ok:
                return
            yield cv2.resize(picture[y:y + h, x:x + w], (width, height), interpolation=cv2.INTER_AREA)
    finally:
        cap.release()


# Frames given to the model at once: a batch keeps the GPU busy.
BATCH = 16


def label(models: dict, ask: dict, tell=say) -> None:
    """Every frame of ask's video labelled with its masker, into ask's out."""
    key = (str(Path(ask["weights"]).expanduser()), ask["model"], int(ask["resolution"]))
    if key not in models:
        with contextlib.redirect_stdout(sys.stderr):
            models.clear()
            models[key] = model_for(ask["model"], int(ask["resolution"]), key[0])
            models[key].inference(compile=False)
    model = models[key]
    labels_of = [int(v) for v in ask["labels"]]
    score = float(ask.get("score", 0.5))
    height, width = ask["size"]
    cap = cv2.VideoCapture(ask["video"])
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    out: list[np.ndarray] = []
    batch: list[np.ndarray] = []

    def flush() -> None:
        with contextlib.redirect_stdout(sys.stderr):
            found = model.predict(batch, threshold=score)
        for detections in found if isinstance(found, list) else [found]:
            labels = np.zeros((height, width), dtype=np.uint8)
            if detections.mask is not None and len(detections):
                # A later thing over an earlier, as SAM 3's labels are laid.
                for k in np.argsort(detections.class_id, kind="stable"):
                    cls = int(detections.class_id[k])
                    if 0 <= cls < len(labels_of):
                        labels[detections.mask[k]] = labels_of[cls]
            out.append(labels)
        batch.clear()
        tell(frame=len(out) - 1, frames=max(count, len(out)))
    for picture in frames(ask["video"], ask["window"], ask["size"]):
        batch.append(cv2.cvtColor(picture, cv2.COLOR_BGR2RGB))
        if len(batch) == BATCH:
            flush()
    if batch:
        flush()
    if not out:
        raise ValueError("the video has no frames")
    np.savez_compressed(ask["out"], labels=np.stack(out))
    tell(done=True)


def answered(models: dict, ask: dict) -> None:
    """One request done, its video taken from the request and its npz sent
    back in the last line where the asker is on another machine."""
    if "train" in ask:
        train(ask)
        return
    with tempfile.TemporaryDirectory() as tmp:
        if "video_data" in ask:
            path = Path(tmp) / "video.mp4"
            path.write_bytes(base64.b64decode(ask.pop("video_data")))
            ask["video"] = str(path)
        inline = "out" not in ask
        if inline:
            ask["out"] = str(Path(tmp) / "out.npz")
        last: dict = {}

        def hold(**fields) -> None:
            # The closing line waits for the npz to go with it.
            nonlocal last
            if inline and fields.get("done"):
                last = fields
            else:
                say(**fields)
        label(models, ask, hold)
        if last:
            say(**last, data=base64.b64encode(Path(ask["out"]).read_bytes()).decode())


def serve() -> int:
    """Requests one JSON line at a time off stdin. stdin is read on a thread
    of its own, so its end - the asker gone - ends the process at once, not
    after the request in hand."""
    lines: queue.Queue = queue.Queue()

    def read() -> None:
        for line in sys.stdin:
            lines.put(line)
        if TEACHING is not None and TEACHING.poll() is None:
            TEACHING.kill()
        os._exit(0)
    threading.Thread(target=read, daemon=True).start()
    models: dict = {}
    while True:
        line = lines.get()
        try:
            answered(models, json.loads(line))
        except Exception as exc:  # noqa: BLE001 - one bad request is an answer, not the end of the service
            say(error=f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--teach"]:
        teach(json.loads(sys.argv[2]))
        sys.exit(0)
    sys.exit(serve())
