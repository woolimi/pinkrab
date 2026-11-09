"""A task's masker: a segmentation model taught the task's things from
pictures SAM 3 labelled, labelling every take in seconds.

SAM 3 tracks a take's things through every frame of its gripper videos,
minutes a take; a job over dozens of takes is hours. Instead SAM 3 looks at
a few frames of every take (masker_frames_per_video, each picture on its
own), and those pictures - every take's room, light and moves, not a few
takes' - teach an RF-DETR segmentation model (masker_run, in its own
environment) the same things; it then finds them in every frame of every
take. It is scored against SAM 3 on the pictures of the takes kept out of
its teaching (masker_held_out): each thing's IoU with SAM 3's, which must
reach masker_min_iou for it to be used. It is taught on the machine SAM 3
runs on and its weights stay there; here is its record, in
maskers_dir/<id>/masker.json: the things it finds, in order, where its
weights are, the takes it learned from and how it scored.
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
from .keep_masks import FIRST_THING, Lens, MaskerService, Remote, Sam, Scene, picture, tracked_spans

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
# In a held-out picture's truth (TeachingSet), where nothing is scored: the
# grippers drawn over the things, which no masker is asked to find.
UNSCORED = 255


def picks(frames: int, spans: list[tuple[int, int]] | None) -> list[int]:
    """masker_frames_per_video frames of a gripper video of `frames`, spread
    evenly over those its things are looked for in (`spans`; every frame for
    None), neither end."""
    looked = np.arange(frames) if spans is None else np.concatenate(
        [np.arange(a, b) for a, b in spans] or [np.zeros(0, dtype=int)])
    n = SETTINGS.masker_frames_per_video
    if len(looked) <= n:
        return [int(k) for k in looked]
    return sorted({int(looked[round(i)]) for i in np.linspace(0, len(looked) - 1, n + 2)[1:-1]})


class TeachingSet:
    """A masker's teaching pictures, as RF-DETR reads a COCO segmentation set:
    train/ and valid/ in `folder`, each with its pictures and
    _annotations.coco.json - and truth/, each valid picture's labels as SAM 3
    gave them, for scoring the masker against SAM 3 (masker_run). A take's
    pictures all go to one part: one take in masker_held_out to valid/."""

    def __init__(self, folder: Path, things: list[str]) -> None:
        self.folder, self.things = folder, things
        # Ids from 1, as RF-DETR reads a COCO set: its class k is the k-th of them.
        categories = [{"id": k + 1, "name": text} for k, text in enumerate(things)]
        self.sets = {part: {"images": [], "annotations": [], "categories": categories} for part in ("train", "valid")}
        for part in (*self.sets, "truth"):
            (folder / part).mkdir(parents=True, exist_ok=True)
        self.takes: list[str] = []

    def take(self, root: Path, pictures) -> None:
        """A take's pictures - (gripper video, frame, picture, labels) as the
        dataset keeps them, the labels SAM 3's with the grippers drawn over -
        added to the part its place among the takes gives it."""
        part = "valid" if len(self.takes) % SETTINGS.masker_held_out == 0 else "train"
        self.takes.append(root.name)
        for name, k, shot, labels in pictures:
            file = f"{root.name}_{name}_{k:05d}"
            self._add(self.sets[part], self.folder / part, f"{file}.jpg", shot, labels)
            if part == "valid":
                truth = np.where((labels >= FIRST_THING) & (labels < FIRST_THING + len(self.things)), labels, 0)
                truth[(labels != 0) & (truth == 0)] = UNSCORED
                cv2.imwrite(str(self.folder / "truth" / f"{file}.png"), truth.astype(np.uint8))

    def close(self) -> dict[str, int]:
        """The set written out; how many pictures each part has."""
        for part, coco in self.sets.items():
            (self.folder / part / "_annotations.coco.json").write_text(json.dumps(coco))
        return {part: len(coco["images"]) for part, coco in self.sets.items()}

    def _add(self, coco: dict, folder: Path, file: str, shot: np.ndarray, labels: np.ndarray) -> None:
        """One picture and its things' outlines added to a COCO set."""
        cv2.imwrite(str(folder / file), shot, [cv2.IMWRITE_JPEG_QUALITY, 92])
        height, width = labels.shape
        image = len(coco["images"]) + 1
        coco["images"].append({"id": image, "file_name": file, "height": height, "width": width})
        for k in range(len(self.things)):
            mask = (labels == FIRST_THING + k).astype(np.uint8)
            rings, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rings = [r for r in rings if len(r) >= 3 and cv2.contourArea(r) >= MIN_PIECE_PX]
            if not rings:
                continue
            x, y, w, h = cv2.boundingRect(np.concatenate(rings))
            coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": image, "category_id": k + 1,
                                        "bbox": [x, y, w, h], "area": float(mask.sum()), "iscrowd": 0,
                                        "segmentation": [r.reshape(-1).astype(float).tolist() for r in rings]})


def looked(root: Path, things: list[str], sam: Sam, say=lambda **_: None):
    """A take's teaching pictures by SAM 3: the picks of each gripper video,
    each looked at on its own - every instance of each thing, a later thing
    over an earlier - with the grippers drawn over, as a take's labels are
    made (keep_masks.make). Yields (gripper video, frame, picture, labels)."""
    scene = Scene(root)
    for side in scene.sides:
        name = f"gripper_{side}"
        lens = scene.lenses[side]
        for k in picks(len(scene.stamps[side]), tracked_spans(root, name)):
            shot = picture(root, name, k, lens.size)
            labels = np.zeros(lens.size, dtype=np.uint8)
            with tempfile.TemporaryDirectory() as tmp:
                cv2.imwrite(str(Path(tmp) / "picture.jpg"), shot, [cv2.IMWRITE_JPEG_QUALITY, 95])
                said = sam.ask({"image": str(Path(tmp) / "picture.jpg"), "texts": things,
                                "out": str(Path(tmp) / "found.npz"), "resolution": SETTINGS.sam3_resolution,
                                "new_thing_score": SETTINGS.sam3_new_thing_score,
                                "thing_score": SETTINGS.sam3_thing_score})
                if "error" in said:
                    raise RuntimeError(f"SAM 3 stopped on {name} frame {k}: {said['error']}")
                found = np.load(Path(tmp) / "found.npz")
                for t in range(len(things)):
                    masks = found[f"t{t}"]
                    if len(masks):
                        labels[masks.any(axis=0)] = FIRST_THING + t
            drawn = scene.draw(side, k)
            labels[drawn != 0] = drawn[drawn != 0]
            say(video=name, frame=k)
            yield name, k, shot, labels


def kept(root: Path):
    """A take's teaching pictures from labels SAM 3 tracked it with before:
    the picks of each gripper video, as looked() gives them."""
    from pinkrab_core.dataset import gripper_view

    meta = json.loads((root / "meta.json").read_text())
    for name, stream in meta["streams"].items():
        labels = read(root, name) if name.startswith("gripper_") else None
        if labels is None:
            continue
        want = set(picks(len(labels), tracked_spans(root, name)))
        cap = cv2.VideoCapture(str(root / f"{name}.mp4"))
        try:
            for k in range(max(want, default=-1) + 1):
                ok, frame = cap.read()
                if not ok:
                    break
                if k in want:
                    yield name, k, gripper_view(frame, labels.shape[1:], tuple(stream["centre"])), labels[k]
        finally:
            cap.release()


def maskers(things: list[str]) -> list[dict]:
    """Every masker that finds exactly `things` and scored well enough to be
    used, on any machine, the newest first."""
    found = []
    for path in SETTINGS.maskers_dir.glob("*/masker.json"):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if record.get("things") == things and usable(record):
            found.append(record)
    return sorted(found, key=lambda record: record["made"], reverse=True)


def usable(record: dict) -> bool:
    """Whether a masker found each of its things as SAM 3 did, near enough:
    an IoU with SAM 3's of masker_min_iou or more on the pictures kept out."""
    iou = record.get("iou") or {}
    return all(iou.get(text, 0.0) >= SETTINGS.masker_min_iou for text in record.get("things") or [None])


def find(things: list[str], machine: str | None = None) -> dict | None:
    """The latest masker taught on `machine` - on any, for None - that finds
    exactly `things` and scored well enough to be used, or None."""
    return next((r for r in maskers(things) if machine in (None, r.get("machine"))), None)


def weights_here(record: dict) -> Path:
    """Where a masker's weights are on this machine: where it was taught, if
    that was here, else beside its record, where a copy is fetched to."""
    taught = Path(record["weights"]).expanduser()
    return taught if taught.is_file() else SETTINGS.maskers_dir / record["id"] / taught.name


def teach(teaching: TeachingSet, masker: MaskerService, machine: str, say=lambda **_: None) -> dict:
    """A masker taught `teaching`'s things from its pictures, written out, on
    the machine `masker` runs on (named `machine`), and scored against SAM 3;
    recorded in maskers_dir whatever it scored - find() leaves out one that
    is not usable(). `say` is told how far the teaching has got."""
    count = teaching.close()
    if not count["train"] or not count["valid"]:
        raise ValueError("too few pictures to teach a masker")
    made = time.time()
    record = {"id": time.strftime("%Y%m%d-%H%M%S", time.localtime(made)), "things": teaching.things,
              "machine": machine, "model": SETTINGS.masker_model, "resolution": SETTINGS.masker_resolution,
              "taught_from": teaching.takes, "pictures": count, "made": made}
    remote = masker.remote
    here = SETTINGS.maskers_dir / record["id"]
    there = f"maskers/{record['id']}"
    if remote is None:
        shutil.copytree(teaching.folder, here / "set")
        data, out = str(here / "set"), str(here / "train")
    else:
        say(stage="send")
        _send(remote, teaching.folder, there)
        data, out = _remote_path(remote, there, "set"), _remote_path(remote, there, "train")
    try:
        answer = masker.ask({"train": data, "into": out, "model": record["model"],
                             "resolution": record["resolution"], "epochs": SETTINGS.masker_epochs,
                             "labels": [FIRST_THING + k for k in range(len(teaching.things))],
                             "score": SETTINGS.masker_score},
                            lambda **line: say(stage="teach", **line))
    finally:
        # The set and what the trainer leaves besides its best weights are
        # only for teaching, and go however it ended.
        _tidy(remote, here, there, ["set", *(f"train/{name}" for name in LEFT)])
    if "error" in answer:
        raise RuntimeError(f"the masker could not be taught: {answer['error']}")
    # One set of best weights kept: the one it is used with.
    _tidy(remote, here, there, [f"train/{name}" for name in BEST if name != Path(answer["weights"]).name])
    record.update(weights=answer["weights"], seconds=answer.get("seconds"),
                  iou=dict(zip(teaching.things, answer["iou"])))
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
