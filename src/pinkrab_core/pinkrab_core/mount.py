"""The two numbers a tape measure gives about this rig, into mount.yaml.

mount.yaml says where the arms are; most of it is derived and stays put. Two
values are measured, and only these two change when the jig is moved or built
again:

  * how far apart the two arm bases are, centre to centre - each base sits at
    half that, either side of pinkrab_base_link;
  * how high the arms are above the desk - measured to the top of a base, the
    flat face its mounting plate, its beam and the cap over the shoulder_pan
    servo share, which a tape can be stood against. Each arm's base_link sits
    level with that face, so this is wall.height as it stands.

The file is edited in place, line by line, rather than loaded and dumped: it
is mostly comments explaining the frames, and a dump would throw them away.
"""
from __future__ import annotations

import re
from pathlib import Path

from .config import SETTINGS

_HEIGHT = re.compile(r"^(\s*height:\s*)([0-9.]+)(.*)$", re.MULTILINE)
_LEFT_XYZ = re.compile(r"^(\s*left:\s*\n\s*xyz:\s*\[\s*[0-9.\-]+,\s*)([0-9.\-]+)(\s*,\s*[0-9.\-]+\s*\].*)$", re.MULTILINE)
_RIGHT_XYZ = re.compile(r"^(\s*right:\s*\n\s*xyz:\s*\[\s*[0-9.\-]+,\s*)([0-9.\-]+)(\s*,\s*[0-9.\-]+\s*\].*)$", re.MULTILINE)


class MountError(RuntimeError):
    """mount.yaml is not shaped the way this expects to edit it."""


def read(path: Path | None = None) -> dict[str, float]:
    """The two measurements as they stand: base spacing and base-top height, metres."""
    text = (path or SETTINGS.mount_file).read_text()
    height = _HEIGHT.search(text)
    left = _LEFT_XYZ.search(text)
    right = _RIGHT_XYZ.search(text)
    if not (height and left and right):
        raise MountError(f"{path or SETTINGS.mount_file} does not have the wall.height and "
                         "arms.left/right.xyz lines this edits")
    return {
        "spacing": float(left.group(2)) - float(right.group(2)),
        "top_height": float(height.group(2)),
    }


def write(spacing: float, top_height: float, path: Path | None = None) -> Path:
    """Put the two measurements into mount.yaml, touching nothing else.

    `spacing` and `top_height` in metres. The arms go at plus and minus half
    the spacing along y; the height is wall.height as it is.
    """
    if spacing <= 0 or top_height <= 0:
        raise MountError("both measurements have to be positive")
    target = path or SETTINGS.mount_file
    text = target.read_text()
    half = spacing / 2.0
    edited, n_h = _HEIGHT.subn(lambda m: f"{m.group(1)}{top_height:.4f}{m.group(3)}", text, count=1)
    edited, n_l = _LEFT_XYZ.subn(lambda m: f"{m.group(1)}{half:.4f}{m.group(3)}", edited, count=1)
    edited, n_r = _RIGHT_XYZ.subn(lambda m: f"{m.group(1)}{-half:.4f}{m.group(3)}", edited, count=1)
    if (n_h, n_l, n_r) != (1, 1, 1):
        raise MountError(f"{target} does not have the wall.height and arms.left/right.xyz "
                         "lines this edits")
    target.write_text(edited)
    # The workspace's URDF is an expansion of this file, made once and kept.
    # Left there it would keep describing the old mount.
    if SETTINGS.urdf_file.is_file():
        SETTINGS.urdf_file.unlink()
    return target


def measure_interactive(ask=input) -> dict[str, float]:
    """One measurement at a time, in millimetres, the current value as the default.

    Enter on its own keeps what is there. Raises EOFError or KeyboardInterrupt
    the way input does, so the caller can say that nothing was written.
    """
    current = read()
    prompts = [
        ("spacing", "distance between the two arm bases, centre to centre"),
        ("top_height", "height of a base's top face above the desk"),
    ]
    answers: dict[str, float] = {}
    for key, what in prompts:
        default_mm = current[key] * 1000.0
        while True:
            raw = ask(f"  {what}, in mm [{default_mm:.0f}]: ").strip()
            if not raw:
                answers[key] = current[key]
                break
            try:
                value = float(raw)
            except ValueError:
                print("    a number, in millimetres")
                continue
            if value <= 0:
                print("    it has to be positive")
                continue
            answers[key] = value / 1000.0
            break
    return answers
