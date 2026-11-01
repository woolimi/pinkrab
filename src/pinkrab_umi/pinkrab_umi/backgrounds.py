"""Gripper pictures with everything but what is kept painted over.

A take's keep labels (keep_masks) say, frame by frame, where its grippers
and its task's things are; the rest of the picture - the room, the person,
the handheld's body - is painted over with a pattern, so a policy cannot lean
on it. The pattern is made from a seed and stays the same through a take,
as a room does; another seed is another room. Each is a smooth field of
colour with stripes, checks or speckle laid over it at random.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import cv2
import numpy as np

from .config import SETTINGS

# Where a take's keep labels are, beside it: one compressed array per gripper
# video, (frames, height, width) uint8, and masks.json saying what each
# label is.
MASKS = "masks"


def write(root: Path, labels: dict[str, np.ndarray]) -> Path:
    """`labels` beside the take, in masks/: one compressed array per video."""
    folder = root / MASKS
    folder.mkdir(exist_ok=True)
    for name, array in labels.items():
        np.savez_compressed(folder / f"{name}.npz", labels=array)
    return folder


def read(root: Path, name: str) -> np.ndarray | None:
    """A gripper video's labels, or None where it has none."""
    path = root / MASKS / f"{name}.npz"
    return np.load(path)["labels"] if path.is_file() else None


def seed(take: str) -> int:
    """The pattern seed of a take: the same on every conversion, another for
    every take."""
    return zlib.crc32(take.encode())


def pattern(seed: int, size: tuple[int, int]) -> np.ndarray:
    """A random backdrop, (height, width, 3) uint8 BGR, the same for the same seed."""
    rng = np.random.default_rng(seed)
    height, width = size
    # A smooth field: a few random colours stretched over the picture.
    cells = rng.integers(2, 7, size=2)
    field = cv2.resize(rng.uniform(0, 255, (cells[0], cells[1], 3)).astype(np.float32), (width, height),
                       interpolation=cv2.INTER_CUBIC)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    kind = rng.integers(0, 4)
    if kind == 0:   # stripes, at an angle
        angle, period = rng.uniform(0, np.pi), rng.uniform(12, 90)
        wave = np.sin((xs * np.cos(angle) + ys * np.sin(angle)) * 2 * np.pi / period) > 0
    elif kind == 1:   # checks
        cell = rng.uniform(10, 70)
        wave = ((xs // cell + ys // cell) % 2).astype(bool)
    elif kind == 2:   # speckle
        small = rng.uniform(0, 1, (max(1, height // 8), max(1, width // 8))).astype(np.float32)
        wave = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST) > 0.5
    else:   # the field alone
        wave = np.zeros((height, width), dtype=bool)
    other = rng.uniform(0, 255, 3).astype(np.float32)
    mix = rng.uniform(0.3, 0.8)
    field[wave] = (1 - mix) * field[wave] + mix * other
    return np.clip(field, 0, 255).astype(np.uint8)


def kept(labels: np.ndarray) -> np.ndarray:
    """Where a picture is kept, from its labels: every labelled pixel, grown
    by keep_grow_px so an edge drawn or tracked a little short is not eaten."""
    grow = SETTINGS.keep_grow_px
    keep = (labels != 0).astype(np.uint8)
    if grow > 0:
        keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    return keep.astype(bool)


def painted(picture: np.ndarray, labels: np.ndarray, backdrop: np.ndarray) -> np.ndarray:
    """`picture` with what `labels` does not keep replaced by `backdrop`, the
    labels scaled to the picture where it is kept at another size."""
    if labels.shape != picture.shape[:2]:
        labels = cv2.resize(labels, (picture.shape[1], picture.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = backdrop.copy()
    keep = kept(labels)
    out[keep] = picture[keep]
    return out
