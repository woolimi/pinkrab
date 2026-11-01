"""Where the desk solid stands, seen from the robot.

The top camera reports poses in the desk solid's frame, z up. The robot's base
frame is z up too, so the one number between them is a yaw, recorded once per
placing of the solid in `desk.yaml` in the workspace and kept with each take
(its meta.json). Teleop and the studio's hands view place the solid in the
robot's world with it (pinkrab_umi.desk_solid).

The standard placing is the solid's +Y facing the way the arms reach (robot +x)
and its +X on the operator's right, cut face down: that is a yaw of -90.

The same file carries `height_mm`: how far the solid's origin - its centre,
where its frame's z = 0 is - sits above the desk surface. The solid is cut
flat one half-face below its centre and stands on that cut, so as printed
that is half a face (15 mm for the 30 mm solid); it is recorded here rather
than assumed because a print sanded down or a mat under the solid changes it,
and every pose the head reports is this much higher than the desk. Each take
keeps the height it was recorded at (its meta.json). Nothing the policy
learns needs it - a delta between two poses does not care where zero is -
but anything that puts a recorded pose beside the desk itself does.
"""
from __future__ import annotations

from pathlib import Path

from .config import SETTINGS


# The standard placing's yaw: the solid's +Y the way the arms reach.
STANDARD_YAW_DEG = -90.0


class DeskMissing(RuntimeError):
    """Raised when the desk solid's yaw has not been recorded for this rig."""


def load_desk(path: Path | None = None) -> dict:
    import yaml

    fpath = path or SETTINGS.desk_file
    if not fpath.is_file():
        raise DeskMissing(
            f"No desk placement at {fpath}.\n"
            "  The desk solid's frame is turned from the robot's by a yaw that has\n"
            "  to be measured where the solid was put down: write `yaw_deg: <deg>`\n"
            "  there (-90 for the standard placing, the solid's +Y facing the way\n"
            "  the arms reach)."
        )
    data = yaml.safe_load(fpath.read_text())
    if not isinstance(data, dict) or "yaw_deg" not in data:
        raise DeskMissing(f"{fpath} has no yaw_deg.")
    height = data.get("height_mm")
    if height is not None and not isinstance(height, (int, float)):
        raise DeskMissing(f"{fpath}: height_mm has to be a number of millimetres, not {height!r}.")
    return data

