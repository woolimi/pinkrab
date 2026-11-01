"""Where the desk solid stands in the robot's world.

The operator sets the solid on the desk in front of their torso, centred on
it, where their hands work, turned by desk.yaml's yaw - the standard placing
has its +Y away from the operator, the way the arms reach, and its +X toward
their right hand and the right arm. The robot reckons it against its board,
midway between the two arms (`solid_in_world`) - the operator's chest stands
where the board does - and reckons it `desk_reckoned_back_m` nearer the
operator than the hands working level with it: a hand's pose against the
solid is carried into the robot's world that much further out
(`world_T_desk`), where the arms work. The studio's hands view draws the
handhelds there, the shelf's replay judges a take there and a policy run
starts its arms where a take's hands began there - on the solid as the take
recorded it standing (`placing`) and at the size it recorded: a smaller solid
with its side against the operator's chest has its centre nearer them.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import yaml

from pinkrab_core.config import ARMS
from pinkrab_core.config import SETTINGS as CORE
from pinkrab_core.desk import STANDARD_YAW_DEG

from .config import SETTINGS
from .markers import solid_mesh


class PlacementError(RuntimeError):
    """mount.yaml or desk.yaml could not be read into a place for the solid."""


@lru_cache(maxsize=4)
def _corners(face_mm: float) -> np.ndarray:
    """The solid's vertices in its own frame, metres, at `face_mm` faces."""
    return solid_mesh(SETTINGS.at_face(SETTINGS.desk_solid, face_mm)).reshape(-1, 3) / 1000.0


def _desk_yaml() -> tuple[float, float | None]:
    """desk.yaml's yaw (degrees) and origin height (mm): the standard placing's
    yaw where none has been written, no height where none has. One written
    but not a number is a fault, not a blank: it is raised, so nothing is
    placed on a yaw or a height somebody did not write."""
    path = CORE.desk_file
    try:
        data = yaml.safe_load(path.read_text()) if path.is_file() else None
    except (OSError, yaml.YAMLError) as exc:
        raise PlacementError(f"{path}: {type(exc).__name__}: {exc}") from exc
    data = data if isinstance(data, dict) else {}
    try:
        yaw = float(data["yaw_deg"]) if "yaw_deg" in data else STANDARD_YAW_DEG
        height = float(data["height_mm"]) if data.get("height_mm") is not None else None
    except (TypeError, ValueError) as exc:
        raise PlacementError(f"{path}: yaw_deg and height_mm have to be numbers ({exc})") from exc
    return yaw, height


def _mount() -> tuple[float, float]:
    """The board's face along x and the middle between the arms along y, in
    the robot's world: its z = 0 is the desk and its x = 0 the back of the
    board (mount.yaml), so the face stands the board's thickness out."""
    try:
        mount = yaml.safe_load(CORE.mount_file.read_text())
        face_x = float(mount["wall"]["thickness"])
        across = sum(float(mount["arms"][arm]["xyz"][1]) for arm in ARMS) / len(ARMS)
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise PlacementError(f"{CORE.mount_file}: {type(exc).__name__}: {exc}") from exc
    return face_x, across


def chest_x() -> float:
    """Where the operator's chest is reckoned to stand along x in the robot's
    world: on the board's face."""
    return _mount()[0]


def placing() -> tuple[float, float]:
    """How the solid stands by desk.yaml: its yaw (degrees) and its origin's
    height above the desk (mm) - the cut plane it stands on as printed where
    desk.yaml gives none."""
    yaw, height = _desk_yaml()
    return yaw, SETTINGS.for_solid(SETTINGS.desk_solid).cut_mm if height is None else height


def solid_in_world(placed: tuple[float, float] | None = None, face_mm: float | None = None) -> np.ndarray:
    """Where the solid is reckoned to stand in the robot's world, 4x4 metres:
    its back on the board's face, midway between the arms, on the desk -
    stood as `placed` says (yaw degrees, origin height mm), or as desk.yaml
    does, and `face_mm` across a face (a take's recorded size), or as it is
    printed now."""
    _, across = _mount()
    yaw_deg, height_mm = placing() if placed is None else placed
    desk = SETTINGS.for_solid(SETTINGS.desk_solid)
    face = desk.face_size_mm if face_mm is None else float(face_mm)
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    turn = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    # As far as the solid reaches back toward the board, turned as it stands.
    back = float((_corners(face) @ turn.T)[:, 0].min())
    out = np.eye(4)
    out[:3, :3] = turn
    out[:3, 3] = [chest_x() - back, across, height_mm / 1000.0]
    return out


def world_T_desk(placed: tuple[float, float] | None = None, face_mm: float | None = None) -> np.ndarray:
    """The frame a pose against the solid is carried into the robot's world
    by, 4x4 metres: the reckoned solid (stood as `placed` says, or as
    desk.yaml does, at `face_mm` or today's size) moved `desk_reckoned_back_m`
    out along its own +Y, away from the operator."""
    out = solid_in_world(placed, face_mm)
    out[:3, 3] += SETTINGS.desk_reckoned_back_m * out[:3, 1]
    return out


def carried(desk: dict) -> tuple[np.ndarray | None, str]:
    """world_T_desk for a take whose solid stood as `desk` says - its
    meta.json's desk, or a dataset's account of it: the solid at the chest,
    turned and raised as recorded, of the size recorded. None, and why, for
    a take with no place in the robot's world."""
    if desk.get("placement") != "chest":
        return None, "recorded before the desk solid stood at the chest"
    if desk.get("yaw_deg") is None:
        return None, "no desk yaw recorded"
    if desk.get("face_mm") is None:
        return None, "no desk solid size recorded"
    return world_T_desk((float(desk["yaw_deg"]), float(desk["origin_above_table_mm"])), float(desk["face_mm"])), ""
