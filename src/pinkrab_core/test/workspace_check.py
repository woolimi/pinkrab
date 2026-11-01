"""Offline check of the desk reach: each arm has a patch of the desk it can
grasp on, in front of the board and lopsided the way the arm is, holding the
desk under the ready pose's tool and cut nowhere but at the board - the arm's
ball holds every pose the arm can take, and the patch stops short of its edge;
solved once, it is answered from memory and from disk after, for that desk only.
No servo, no page. A couple of minutes: the patch is solved cell by cell.

    python src/pinkrab_core/test/workspace_check.py
"""
from __future__ import annotations

import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pinkrab_core import workspace  # noqa: E402
from pinkrab_core.config import ARMS  # noqa: E402
from pinkrab_core.kinematics import ArmKinematics  # noqa: E402
from pinkrab_core.origin import ready_joints  # noqa: E402
from pinkrab_core.workspace import CELL_M, DeskReach  # noqa: E402

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


if __name__ == "__main__":
    workspace.ANY_RANDOM_SEEDS = 1   # the check's own trade: a coarser patch, half the wait
    path = Path(tempfile.mkdtemp()) / "desk_reach.json"
    keeper = DeskReach(path)
    keeper.warm()
    began = time.monotonic()
    arms = keeper.get()   # waits for the solve warm() started, rather than starting another
    solved_s = time.monotonic() - began
    check("both arms answered by name", set(arms) == set(ARMS))
    for arm in ARMS:
        patch = arms[arm]
        cells = {(x, y) for x, y, _ in patch["cells"]}
        xs = [x for x, _ in cells]
        ys = [y for _, y in cells]
        check(f"{arm}: a patch", patch["area_cm2"] > 500, f"{patch['area_cm2']} cm2")
        check(f"{arm}: it lies in front of the board", bool(xs) and min(xs) > 0)
        chain = ArmKinematics(arm)
        centre, radius = chain.reach_ball()
        lo, hi = chain.limits
        rng = np.random.default_rng(1)
        furthest = max(float(np.linalg.norm(chain.forward(rng.uniform(lo, hi))[:3, 3] - centre))
                       for _ in range(300))
        check(f"{arm}: the ball holds every pose tried", furthest <= radius,
              f"furthest {furthest * 1000:.0f} mm of {radius * 1000:.0f} mm")
        across = math.sqrt(radius ** 2 - (centre[2] - workspace.GRASP_ABOVE_DESK_M) ** 2)
        edge = max((math.hypot(x - centre[0], y - centre[1]) for x, y in cells), default=across)
        check(f"{arm}: the patch stops short of the cells tried, so nothing was cut off",
              edge + CELL_M < across, f"{edge * 1000:.0f} mm out of {across * 1000:.0f} mm")
        tool = chain.forward(ready_joints(arm, chain.joint_names))[:3, 3]
        under = min(cells, key=lambda c: (c[0] - tool[0]) ** 2 + (c[1] - tool[1]) ** 2, default=(9.0, 9.0))
        check(f"{arm}: it holds the desk under the ready pose's tool",
              abs(under[0] - tool[0]) <= CELL_M and abs(under[1] - tool[1]) <= CELL_M,
              f"tool at ({tool[0]:+.3f}, {tool[1]:+.3f}) m")
        check(f"{arm}: more than one attitude of the window reaches most of it",
              sum(1 for _, _, n in patch["cells"] if n >= 2) > len(cells) / 2)
        outer = (min(ys) if arm == "right" else max(ys)) if ys else 0.0
        inner = (max(ys) if arm == "right" else min(ys)) if ys else 0.0
        check(f"{arm}: it reaches further to its outer side than across the middle",
              abs(outer) > abs(inner), f"outer {outer:+.2f} m, inner {inner:+.2f} m")

    began = time.monotonic()
    again = keeper.get()
    check("asked again, the same patches from memory", all(again[a] is arms[a] for a in ARMS),
          f"{time.monotonic() - began:.3f} s")
    began = time.monotonic()
    read = DeskReach(path).get()
    took = time.monotonic() - began
    check("a new keeper reads them from disk instead of solving", read == arms and took < solved_s / 10,
          f"{took:.3f} s against {solved_s:.0f} s solved")
    check("kept for this desk only", DeskReach._key(-0.05, 0.0) != DeskReach._key(0.0, 0.0))
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s): {FAILURES}")
        sys.exit(1)
    print("all ok")
