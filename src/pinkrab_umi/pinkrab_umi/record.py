"""The camera's sessions: disk format, read-back, and what a picture is solved with.

The live view is what drives this - it feeds the writer during a take, and
bench reads the result back through load_frames.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from .config import SETTINGS
from .pnp import Intrinsics


class RecorderError(RuntimeError):
    """Raised when the camera cannot be opened, or is not there at all."""


@dataclass
class SessionPaths:
    """Where one recording lives. One directory, so a session moves by copying it."""

    root: Path

    @property
    def pictures(self) -> Path:
        return self.root / "frames"

    @property
    def frames_file(self) -> Path:
        return self.root / "frames.jsonl"

    @property
    def calibration_file(self) -> Path:
        return self.root / "calibration.json"

    def create(self) -> SessionPaths:
        self.pictures.mkdir(parents=True, exist_ok=True)
        return self


def session_paths(session: str) -> SessionPaths:
    return SessionPaths(SETTINGS.session_dir(session))


def default_session_name() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def calibration_dict(camera: Intrinsics, source: str, unit: str | None) -> dict:
    """The picture's numbers, and where they came from.

    The focal length is what turns a marker's size in pixels into metres, so it
    is the scale of every pose. The top camera carries no numbers of its own:
    they are its lens as fitted against a board (`pinkrab-umi lens fit top`),
    and `camera_unit` names the lens the fit was made of - which is what a take
    is matched to a later fit by when it is solved again (`pinkrab-umi resolve`).
    """
    return {
        "stream": {"kind": "top", "width": SETTINGS.top_camera_width, "height": SETTINGS.top_camera_height,
                   "fps": SETTINGS.fps, "format": "bgr8"},
        "camera": camera.to_dict(),
        "camera_source": source,
        "camera_unit": unit,
    }


def camera_of(calibration: dict) -> Intrinsics:
    """The imager's numbers out of a calibration, as written today or before."""
    data = calibration.get("camera") or calibration.get("color")
    if data is None:
        raise RecorderError("this calibration has no imager in it")
    return Intrinsics.from_dict(data)


class SessionWriter:
    """One recording on disk, fed a frame at a time by whoever has the camera.

    The disk format lives here and only here: the live view is the one place a
    session gets written, and bench the one place it gets read, and this class is
    the contract between them. A frame's stamp is its exposure on this host's
    clock already (its arrival less the camera's latency), so it is `host_ts`
    as it comes.
    """

    def __init__(self, paths: SessionPaths, calibration: dict):
        self.paths = paths.create()
        self.paths.calibration_file.write_text(json.dumps(calibration, indent=2) + "\n")
        self._records: list[dict] = []
        self._closed = False

    @property
    def count(self) -> int:
        return len(self._records)

    def add(self, picture, sensor_ts: float, frame_number: int, arrival: float) -> bool:
        """One frame down, as PNG. False once closed, for the frame that was
        already in flight when somebody ended the take.
        """
        if self._closed:
            return False
        index = len(self._records)
        path = self.paths.pictures / f"{index:06d}.png"
        cv2.imwrite(str(path), picture)
        self._records.append({
            "index": index,
            "sensor_ts": sensor_ts,
            "host_ts": sensor_ts,
            "arrival_ts": arrival,
            "frame_number": frame_number,
            "picture": f"frames/{path.name}",
        })
        return True

    def close(self) -> Path:
        """Write frames.jsonl, and hand back the session."""
        if not self._closed:
            self._closed = True
            with self.paths.frames_file.open("w") as handle:
                for record_ in self._records:
                    handle.write(json.dumps(record_) + "\n")
        return self.paths.root


def load_frames(root: Path) -> list[dict]:
    """Read a session's frame index back, whoever recorded it."""
    paths = SessionPaths(Path(root))
    if not paths.frames_file.exists():
        raise RecorderError(f"{paths.frames_file} is not there - not a session directory")
    return [json.loads(line) for line in paths.frames_file.read_text().splitlines() if line.strip()]


def load_intrinsics(root: Path) -> Intrinsics:
    """The intrinsics the session was solved with, or a stated failure.

    Deliberately no nominal fallback. A session solved against another unit's
    focal length comes out scaled, and a scaled trajectory looks exactly like a
    correct one.
    """
    paths = SessionPaths(Path(root))
    if not paths.calibration_file.exists():
        raise RecorderError(
            f"{paths.calibration_file} is not there. It is written at record time - the "
            "lens fit it was solved with; without it there is no "
            "honest scale for this session."
        )
    return camera_of(json.loads(paths.calibration_file.read_text()))
