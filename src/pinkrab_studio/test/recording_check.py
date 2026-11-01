"""Offline check of the shelf's camera test: a take whose videos ran whole
passes it; one with a video missing, one kept at half its rate, one with a
hole in it and one whose picture never changed each fail it, on that video.
Small videos made up here. No camera.

    python src/pinkrab_studio/test/recording_check.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from pinkrab_studio.verdict import recording

FAILURES: list[str] = []
FPS = 30
STREAMS = ("head", "gripper_left", "gripper_right")


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def write_take(root: Path, frozen: str = "", missing: str = "", every: dict | None = None,
               hole: str = "") -> None:
    """Three seconds of three videos, a square crossing each; `frozen` holds
    one picture throughout, `missing` has no video, `every` keeps one frame
    in n of a stream, `hole` stops one for half a second."""
    root.mkdir(parents=True)
    frames, streams = [], {}
    for name in STREAMS:
        if name == missing:
            continue
        keep = (every or {}).get(name, 1)
        out = cv2.VideoWriter(str(root / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (160, 100))
        written = 0
        for i in range(0, 3 * FPS, keep):
            t = 100.0 + i / FPS
            if name == hole and 1.0 <= i / FPS < 1.5:
                continue
            picture = np.full((100, 160, 3), 40, np.uint8)
            x = 0 if name == frozen else (i * 4) % 140
            cv2.rectangle(picture, (x, 30), (x + 20, 70), (230, 230, 230), -1)
            out.write(picture)
            frames.append({"stream": name, "index": written, "host_ts": t})
            written += 1
        out.release()
        streams[name] = {"file": f"{name}.mp4", "fps": FPS, "frames": written}
    (root / "meta.json").write_text(json.dumps({"kind": "demo", "seconds": 3.0, "streams": streams}))
    (root / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in frames))


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_take(tmp / "whole")
        check("a take whose videos ran whole passes", recording(tmp / "whole") == [], str(recording(tmp / "whole")))
        for label, kwargs, want in (
                ("a video missing fails on it", {"missing": "gripper_left"}, "gripper_left: no video"),
                ("one kept at half its rate fails on it", {"every": {"head": 2}}, "head: 15 fps of 30"),
                ("one with a hole in it fails on it", {"hole": "gripper_right"}, "gripper_right: no frame for"),
                ("one whose picture never changed fails on it", {"frozen": "gripper_left"},
                 "gripper_left: picture never changed")):
            name = label.split()[1] + "-" + want.split(":")[0]
            write_take(tmp / name, **kwargs)
            fails = recording(tmp / name)
            check(label, len(fails) == 1 and fails[0].startswith(want), str(fails))
    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
