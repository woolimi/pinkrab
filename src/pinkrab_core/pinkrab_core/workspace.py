"""Where the tool can grasp on the desk.

An arm's reach is not one volume. Where the tool can be AT is a wide shape;
where it can be while looking down at the desk the way a hand looks when it
picks something up is a narrow, lopsided patch of it. This is that patch:
every cell of the desk in front of the board that the tool reaches at grasping
height in some attitude of the window a hand grasps in (GRASP_PITCH_DEG,
GRASP_HEADING_DEG, the jaw level), in any arm shape, each cell solved from
scratch. The board is the only edge: the cells tried are every one the arm's
lengths could reach (`ArmKinematics.reach_ball`).

It belongs to the description and the desk, not to where the arm is: solved
once, about a minute, and kept on disk until the description, the desk or a
setting it is solved with changes. The desk alone: what the tool can do up in
the air is not drawn.
"""
from __future__ import annotations

import json
import math
import threading
from pathlib import Path

import numpy as np

from .config import ARMS, SETTINGS
from .follow import TOOL_TURN
from .kinematics import ArmKinematics, fresh_urdf
from .origin import READY_DEG, ready_joints

# What a hand holds while grasping on the desk, as the takes are recorded and
# so as a run may ask: this far below horizontal, and the approach turned this
# far from straight out from the board, the jaw's axis level. Rolling the jaw
# as well was measured to add a third of a percent to the patch at three times
# the time.
GRASP_PITCH_DEG = (30.0, 45.0, 60.0)
GRASP_HEADING_DEG = (-45.0, 0.0, 45.0)
# The tool centre's height above the desk while grasping: the takes' lowest.
GRASP_ABOVE_DESK_M = 0.025
# The grid, in every direction: 20 mm draws as a surface at the page's scale.
# Its lines are the board's face and y = 0, so both arms' cells fall on one lattice.
CELL_M = 0.02
# A cell is reached when the solver lands within this of it, no joint at its stop.
LAND_M, LAND_DEG, STOP_DEG = 0.005, 5.0, 0.5
# Each cell is tried from the neighbours' answers (the cell before in the row,
# the same y a row back), then the ready pose, then this many random shapes.
ANY_RANDOM_SEEDS = 3


def _attitude(heading: float, pitch: float) -> np.ndarray:
    """A tool rotation: the approach (tool z) this far round from +x and this far
    below horizontal, the jaw's axis (tool x) level."""
    z = np.array([math.cos(pitch) * math.cos(heading), math.cos(pitch) * math.sin(heading), -math.sin(pitch)])
    x = np.array([-math.sin(heading), math.cos(heading), 0.0])
    return np.column_stack([x, np.cross(z, x), z])


def _rotation(index: tuple[int, int], arm: str) -> np.ndarray:
    """A window attitude, as (pitch, heading) indices, as the arm's tool frame
    holds it: the left tool's frame is the right's turned half round about the
    approach (TOOL_TURN), and the same grasp is the same window attitude on either."""
    i, j = index
    return _attitude(math.radians(GRASP_HEADING_DEG[j]), math.radians(GRASP_PITCH_DEG[i])) @ TOOL_TURN[arm]


def _landed(chain: ArmKinematics, P: np.ndarray, j: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> bool:
    """Whether joints `j` hold the tool at `P`, clear of every stop."""
    mm, deg = chain.reach_error(P, j)
    return mm <= LAND_M and deg <= LAND_DEG and bool(np.all(np.minimum(j - lo, hi - j) > STOP_DEG))


def desk_reach(arm: str, desk_z: float = 0.0, board_x: float = 0.0) -> dict:
    """Every cell of the desk the tool can grasp on in some arm shape, with how
    many attitudes of the window reach it: each cell solved from scratch at the
    window's pitches and headings, the jaw level, from the neighbours' answers
    first, then the ready pose, then a few random shapes."""
    chain = ArmKinematics(arm)
    lo, hi = chain.limits
    rng = np.random.default_rng(0)
    grasp_z = desk_z + GRASP_ABOVE_DESK_M
    # Every cell in front of the board inside the circle where the arm's ball
    # meets the grasping height: none outside it can be reached.
    centre, radius = chain.reach_ball()
    drop = centre[2] - grasp_z
    across = math.sqrt(max(0.0, radius ** 2 - drop ** 2))
    xs = [round(board_x + (i + 0.5) * CELL_M, 4)
          for i in range(max(0, math.ceil((centre[0] + across - board_x) / CELL_M)))]
    ys = [round((k + 0.5) * CELL_M, 4)
          for k in range(math.floor((centre[1] - across) / CELL_M), math.ceil((centre[1] + across) / CELL_M))]
    ready = np.array(ready_joints(arm, chain.joint_names), dtype=float)

    counts: dict[tuple[float, float], int] = {}
    for pitch in range(len(GRASP_PITCH_DEG)):
        for heading in range(len(GRASP_HEADING_DEG)):
            R = _rotation((pitch, heading), arm)
            last_row: dict[float, np.ndarray] = {}
            for x in xs:
                previous = None
                for y in ys:
                    if (x - centre[0]) ** 2 + (y - centre[1]) ** 2 > across ** 2:
                        continue
                    P = np.eye(4)
                    P[:3, :3] = R
                    P[:3, 3] = [x, y, grasp_z]
                    found = None
                    seeds = [s for s in (previous, last_row.get(y)) if s is not None]
                    seeds += [ready] + [rng.uniform(lo, hi) for _ in range(ANY_RANDOM_SEEDS)]
                    for seed in seeds:
                        j = chain.solve(P, seed=seed, restarts=1, orientation_weight=1.0)
                        if _landed(chain, P, j, lo, hi):
                            found = j
                            break
                    previous = found
                    if found is not None:
                        last_row[y] = found
                        counts[(x, y)] = counts.get((x, y), 0) + 1
                    else:
                        last_row.pop(y, None)
    cells = [[x, y, n] for (x, y), n in sorted(counts.items())]
    return {"cells": cells, "area_cm2": round(len(cells) * (CELL_M * 100) ** 2, 1),
            "attitudes": len(GRASP_PITCH_DEG) * len(GRASP_HEADING_DEG)}


class DeskReach:
    """Each arm's `desk_reach`, solved once for a description and desk and kept
    in memory and on disk."""

    def __init__(self, path: Path | None = None) -> None:
        # One solve at a time: a second asker waits for the first and is
        # answered from what it kept, rather than solving the minute again.
        self._lock = threading.Lock()
        self._path = path or SETTINGS.workspace / "desk_reach.json"
        self._kept: dict | None = None

    @staticmethod
    def _key(desk_z: float, board_x: float) -> dict:
        """What the patch was solved for: the description, the desk and every setting it is solved with."""
        return {"urdf_mtime": fresh_urdf().stat().st_mtime, "cell_m": CELL_M,
                "seeds": ANY_RANDOM_SEEDS,
                # Lists, not tuples: the key is compared with itself read back from JSON.
                "window": [list(GRASP_PITCH_DEG), list(GRASP_HEADING_DEG)],
                # Everything else the patch is solved with: the height, what
                # counts as reached, the ready pose every cell is tried from,
                # and the turn the left tool frame carries. The cells tried
                # follow from the description, in the key already.
                "grasp_above_m": GRASP_ABOVE_DESK_M, "cells_tried": "reach_ball",
                "landed": [LAND_M, LAND_DEG, STOP_DEG],
                # The solver's own settings: what a step costs, the tick it
                # is spent over, and the speed it is limited to.
                "ik": [SETTINGS.ik_regularization, SETTINGS.ik_rate_hz, SETTINGS.max_joint_speed_dps],
                "ready": {arm: dict(READY_DEG[arm]) for arm in ARMS},
                "tool_turn": {arm: np.asarray(TOOL_TURN[arm]).tolist() for arm in ARMS},
                "desk_z": round(desk_z, 4), "board_x": round(board_x, 4)}

    def _kept_for(self, key: dict) -> dict[str, dict] | None:
        """The patches for `key` from memory or disk, else None. Under the lock."""
        if self._kept is not None and self._kept.get("key") == key:
            return self._kept["arms"]
        if self._path.is_file():
            try:
                data = json.loads(self._path.read_text())
            except (OSError, ValueError):
                data = None
            if data and data.get("key") == key:
                self._kept = data
                return data["arms"]
        return None

    def warm(self, desk_z: float = 0.0, board_x: float = 0.0) -> None:
        """Solved now on a thread, so the page finds it ready."""
        def run() -> None:
            try:
                self.get(desk_z, board_x)
            except Exception as exc:   # noqa: BLE001 - the studio starts all the same
                # The page that asks is told in its own words; a thread's
                # failure is only worth a line on the terminal.
                print(f"  desk reach: not solved - {type(exc).__name__}: {exc}")

        threading.Thread(target=run, name="desk-reach-warm", daemon=True).start()

    def get(self, desk_z: float = 0.0, board_x: float = 0.0) -> dict[str, dict]:
        """Each arm's patch for this description and desk: kept, or solved now."""
        with self._lock:
            key = self._key(desk_z, board_x)
            kept = self._kept_for(key)
            if kept is not None:
                return kept
            data = {"key": key, "arms": {arm: desk_reach(arm, desk_z, board_x) for arm in ARMS}}
            # Kept in memory first: a disk that will not take it should not
            # cost the next request the whole minute again.
            self._kept = data
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._path.write_text(json.dumps(data))
            except OSError:
                pass
            return data["arms"]
