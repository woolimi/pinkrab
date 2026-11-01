"""The front page's robot as static files, for a page that has no server under it.

The studio asks the server for the description, the ready pose, the desk and
the board, Hermit's parts and each arm's reach on the desk. A static site - the
documentation - draws the same pictures from the same answers, written down
here once. The meshes are not copied: they are in the repository already, and a
static site takes them from there when it is built.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pinkrab_core import origin, workspace
from pinkrab_core.config import SETTINGS
from pinkrab_core.urdf import expand_for_meshes
from pinkrab_mujoco.scene import Scene

from .server import hermit_parts

FILES = ("model.urdf", "ready.json", "physics.json", "hermit.json", "workspace.json")
XACRO_BANNER = re.compile(r"<!-- =+ -->\n(<!-- \|.*\| -->\n)+<!-- =+ -->\n")


def export(out: Path, mount: Path | None = None, scene: Path | None = None) -> list[Path]:
    """Write what the front page fetches into `out`; returns the files written."""
    out.mkdir(parents=True, exist_ok=True)
    scenery = Scene(scene or SETTINGS.scene_file).scenery()
    # The desk's top and the board's face, as the studio's /workspace takes them.
    desk, board = scenery.get("desk"), scenery.get("pegboard")
    desk_z = desk["pos"][2] if desk else 0.0
    board_x = board["pos"][0] + board["half"][0] if board else 0.0
    answers = {
        # Without xacro's banner, which names the temporary file it expanded
        # and so changes on every export.
        "model.urdf": XACRO_BANNER.sub("", expand_for_meshes("meshes", mount), count=1),
        "ready.json": origin.ready_pose(),
        "physics.json": {"scenery": scenery},
        "hermit.json": hermit_parts(),
        # Each arm's patch of the desk, as the studio's reach switch lays it down.
        "workspace.json": {"cell_m": workspace.CELL_M, "desk_z": desk_z,
                           "arms": workspace.DeskReach().get(desk_z=desk_z, board_x=board_x)},
    }
    written = []
    for name in FILES:
        answer = answers[name]
        path = out / name
        path.write_text(answer if isinstance(answer, str) else json.dumps(answer, indent=1) + "\n")
        written.append(path)
    return written
