"""A task's masker: a segmentation model taught the task's things from the
takes SAM 3 tracked, labelling the rest of the takes in seconds each.

SAM 3 tracks a take's things through every frame of its gripper videos,
minutes a take; a job over dozens of takes is hours. Once SAM 3 has tracked
some (masker_from_takes), their frames teach an RF-DETR segmentation model
(masker_run, in its own environment) the same things in the same room, and
it finds them in each frame of the remaining takes at once. It is taught on
the machine SAM 3 runs on and its weights stay there; here is its record,
in maskers_dir/<id>/masker.json: the things it finds, in order, where its
weights are, the takes it learned from and how it scored on those kept out.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from pinkrab_core.dataset import gripper_window

from .backgrounds import read
from .config import SETTINGS
from .keep_masks import FIRST_THING, Lens, MaskerService, Remote, account

# What the trainer leaves beside the best weights, only for teaching on: its
# last states and the checkpoint it keeps every few epochs, 1.3 GB a masker.
LEFT = ("*.ckpt", "last_ema.pth")
# The best weights the trainer keeps: of the model as trained and of its EMA.
BEST = ("checkpoint_best_total.pth", "checkpoint_best_ema.pth")
# How long the teaching set is given to reach the machine, in seconds.
SEND_TIMEOUT_S = 30 * 60
# The smallest piece of a thing's label taught as part of it, in pixels:
# smaller ones are the label's ragged edge.
MIN_PIECE_PX = 20


def teaching_set(takes: list[Path], things: list[str], out: Path, say=lambda **_: None) -> dict:
    """A COCO segmentation set in `out` (train/ and valid/, each with its
    pictures and _annotations.coco.json) from `takes`' SAM 3 labels: every
    masker_frame_step-th frame of each gripper video, of the frames its
    things were tracked over, cropped as the dataset keeps it; one take in
    masker_held_out goes to valid/. Returns how many pictures each part has."""
    from pinkrab_core.dataset import gripper_view

    # Ids from 1, as RF-DETR reads a COCO set: its class k is the k-th of them.
    categories = [{"id": k + 1, "name": text} for k, text in enumerate(things)]
    sets = {part: {"images": [], "annotations": [], "categories": categories} for part in ("train", "valid")}
    for part in sets:
        (out / part).mkdir(parents=True, exist_ok=True)
    for i, root in enumerate(takes):
        part = "valid" if i % SETTINGS.masker_held_out == 0 else "train"
        said = account(root) or {}
        meta = json.loads((root / "meta.json").read_text())
        for name, spans in (said.get("tracked") or {}).items():
            labels = read(root, name)
            if labels is None:
                continue
            taught = np.zeros(len(labels), dtype=bool)
            for a, b in spans or [(0, len(labels))]:
                taught[a:b] = True
            stream = meta["streams"][name]
            cap = cv2.VideoCapture(str(root / f"{name}.mp4"))
            try:
                for k in range(len(labels)):
                    ok, picture = cap.read()
                    if not ok:
                        break
                    if k % SETTINGS.masker_frame_step or not taught[k]:
                        continue
                    _add(sets[part], out / part, f"{root.name}_{name}_{k:05d}.jpg",
                         gripper_view(picture, labels.shape[1:], tuple(stream["centre"])), labels[k], len(things))
            finally:
                cap.release()
        say(take=root.name, index=i, takes=len(takes))
    for part, coco in sets.items():
        (out / part / "_annotations.coco.json").write_text(json.dumps(coco))
    return {part: len(coco["images"]) for part, coco in sets.items()}


def _add(coco: dict, folder: Path, file: str, picture: np.ndarray, labels: np.ndarray, things: int) -> None:
    """One picture and its things' outlines added to a COCO set."""
    cv2.imwrite(str(folder / file), picture, [cv2.IMWRITE_JPEG_QUALITY, 92])
    height, width = labels.shape
    image = len(coco["images"]) + 1
    coco["images"].append({"id": image, "file_name": file, "height": height, "width": width})
    for k in range(things):
        mask = (labels == FIRST_THING + k).astype(np.uint8)
        rings, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rings = [r for r in rings if len(r) >= 3 and cv2.contourArea(r) >= MIN_PIECE_PX]
        if not rings:
            continue
        x, y, w, h = cv2.boundingRect(np.concatenate(rings))
        coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": image, "category_id": k + 1,
                                    "bbox": [x, y, w, h], "area": float(mask.sum()), "iscrowd": 0,
                                    "segmentation": [r.reshape(-1).astype(float).tolist() for r in rings]})


def find(things: list[str], machine: str) -> dict | None:
    """The latest masker taught on `machine` that finds exactly `things` and
    scored well enough to be used, or None."""
    best = None
    for path in SETTINGS.maskers_dir.glob("*/masker.json"):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if (record.get("things") == things and record.get("machine") == machine
                and record.get("map50", 0) >= SETTINGS.masker_min_map50
                and (best is None or record["made"] > best["made"])):
            best = record
    return best


def teach(takes: list[Path], things: list[str], masker: MaskerService, machine: str,
          say=lambda **_: None) -> dict:
    """A masker taught `things` from `takes`' SAM 3 labels on the machine
    `masker` runs on (named `machine`), recorded in maskers_dir - whatever
    it scored; find() leaves out one that scored too low. `say` is told how
    far the set and the teaching have got."""
    made = time.time()
    record = {"id": time.strftime("%Y%m%d-%H%M%S", time.localtime(made)), "things": things, "machine": machine,
              "model": SETTINGS.masker_model, "resolution": SETTINGS.masker_resolution,
              "taught_from": [t.name for t in takes], "made": made}
    remote = masker.remote
    here = SETTINGS.maskers_dir / record["id"]
    there = f"maskers/{record['id']}"
    with tempfile.TemporaryDirectory() as tmp:
        sets = Path(tmp) / "set"
        count = teaching_set(takes, things, sets, lambda **line: say(stage="set", **line))
        if not count["train"] or not count["valid"]:
            raise ValueError("too few frames to teach a masker")
        record["pictures"] = count
        if remote is None:
            folder = here
            shutil.copytree(sets, folder / "set")
            data, out = str(folder / "set"), str(folder / "train")
        else:
            say(stage="send")
            _send(remote, sets, there)
            data, out = _remote_path(remote, there, "set"), _remote_path(remote, there, "train")
    try:
        answer = masker.ask({"train": data, "into": out, "model": record["model"],
                             "resolution": record["resolution"], "epochs": SETTINGS.masker_epochs},
                            lambda **line: say(stage="teach", **line))
    finally:
        # The set and what the trainer leaves besides its best weights are
        # only for teaching, and go however it ended.
        _tidy(remote, here, there, ["set", *(f"train/{name}" for name in LEFT)])
    if "error" in answer:
        raise RuntimeError(f"the masker could not be taught: {answer['error']}")
    # One set of best weights kept: the one it is used with.
    _tidy(remote, here, there, [f"train/{name}" for name in BEST if name != Path(answer["weights"]).name])
    record.update(weights=answer["weights"], map50=answer.get("map50", 0.0), map=answer.get("map", 0.0),
                  seconds=answer.get("seconds"))
    here.mkdir(parents=True, exist_ok=True)
    (here / "masker.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def _tidy(remote: Remote | None, here: Path, there: str, paths: list[str]) -> None:
    """`paths` (globs, under the masker's folder) removed, here or on the
    machine it was taught on."""
    if remote is None:
        for pattern in paths:
            for path in here.glob(pattern):
                shutil.rmtree(path) if path.is_dir() else path.unlink()
    else:
        # The folder quoted, a name's glob left for the remote shell to expand.
        remote.run("rm -rf " + " ".join(remote.path(there, *folder) + "/" + name
                                        for folder, name in ((p.split("/")[:-1], p.split("/")[-1]) for p in paths)))


def _remote_path(remote: Remote, *parts: str) -> str:
    """A path under the remote's root as the remote process reads it: ~ left
    for that machine's Python to expand."""
    return "/".join((remote.root, *parts))


def _send(remote: Remote, folder: Path, there: str) -> None:
    """`folder` put under the remote's root at `there`, as a tar over ssh."""
    pack = subprocess.Popen(["tar", "-C", str(folder.parent), "-cf", "-", folder.name],
                            stdout=subprocess.PIPE, env={"COPYFILE_DISABLE": "1", "PATH": "/usr/bin:/bin"})
    try:
        done = subprocess.run([*remote.ssh, f"mkdir -p {remote.path(there)} && tar -C {remote.path(there)} -xf -"],
                              stdin=pack.stdout, capture_output=True, timeout=SEND_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{remote.name}: the masker's pictures were not sent in {SEND_TIMEOUT_S // 60} "
                           "minutes") from exc
    finally:
        pack.stdout.close()
        pack.wait()
    if done.returncode != 0 or pack.returncode != 0:
        said = done.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError(f"{remote.name}: the masker's pictures could not be sent "
                           f"({said[-1] if said else f'ssh exited {done.returncode}'})")


def labelled(root: Path, name: str, things: list[str], record: dict, masker: MaskerService, say=lambda **_: None,
             spans: list[tuple[int, int]] | None = None) -> np.ndarray:
    """A gripper video's labels for the task's `things` by its masker - each
    the label FIRST_THING + its place in the list where it was found - over
    every frame, or with `spans` over those [start, end) frames only, the
    rest left 0, as tracked() gives SAM 3's."""
    if record["things"] != things:
        raise ValueError("the masker finds other things")
    meta = json.loads((root / "meta.json").read_text())
    stream = meta["streams"][name]
    lens = Lens(stream)
    window = list(gripper_window(*stream["size"], tuple(stream["centre"])))
    with tempfile.TemporaryDirectory() as tmp:
        answer = masker.ask({"video": str(root / f"{name}.mp4"), "window": window, "size": list(lens.size),
                             "weights": record["weights"], "model": record["model"],
                             "resolution": record["resolution"], "score": SETTINGS.masker_score,
                             "labels": [FIRST_THING + k for k in range(len(things))],
                             "out": str(Path(tmp) / "labels.npz")}, say)
        if "error" in answer:
            raise RuntimeError(f"the masker stopped on {name}: {answer['error']}")
        labels = np.load(Path(tmp) / "labels.npz")["labels"]
    if spans is not None:
        kept = np.zeros(len(labels), dtype=bool)
        for a, b in spans:
            kept[a:b] = True
        labels[~kept] = 0
    return labels
