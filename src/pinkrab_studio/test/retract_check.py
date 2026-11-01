"""Offline check of the shelf's retract rule: after a hand lets go - its jaw's
last opening - it has to rise and draw back along its approach before the
take stops. Hand poses made up in the desk's frame, the top camera standing
at the desk's origin so the recorded poses are the desk's. And of its
shoulder_pan rule: a take whose replay swings an arm far round from the
ready pose is pointed out on that arm, and of its joint speed rule: one
turned faster than the servos turn fails. And of where a hand ended against the ready
pose, which excuses a short retract. No camera.

    python src/pinkrab_studio/test/retract_check.py
"""
from __future__ import annotations

import math
import sys

import numpy as np

from pinkrab_core.origin import READY_DEG
from pinkrab_core.follow import desk_tool
from pinkrab_core.kinematics import ArmKinematics
from pinkrab_studio.verdict import _hand, end_from_ready, fastest_joint, retract
from pinkrab_umi.config import SETTINGS as UMI
from pinkrab_umi.desk_solid import world_T_desk

FAILURES: list[str] = []
# The tool centre where the solid is: what the hand moves is what is measured.
META = {"solid_T_tcp": {f"handheld_{h}": np.eye(4).tolist() for h in ("left", "right")}}


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def tool(x: float, y: float, z: float) -> list[list[float]]:
    """A tool at (x, y, z) m pointing out along +x and 45 degrees down."""
    c = np.cos(np.radians(45))
    T = np.eye(4)
    approach = np.array([c, 0.0, -c])
    side = np.array([0.0, 1.0, 0.0])
    T[:3, :3] = np.column_stack([np.cross(side, approach), side, approach])
    T[:3, 3] = (x, y, z)
    return T.tolist()


def take(path: list[tuple[float, float, float]], jaws: list[float], hand: str = "right") -> list[dict]:
    lines = []
    for (x, y, z), jaw in zip(path, jaws):
        pose = tool(x, y, z)
        lines.append({"solids": {f"handheld_{hand}": {"solved": True, "pose": pose, "world_pose": pose}},
                      "jaw": {hand: jaw}})
    return lines


def main() -> int:
    print("\nretract after letting go")
    # Reach out holding the block (jaw half shut), open at x = 0.30, then ...
    reach = [(0.20 + 0.01 * i, 0.0, 0.05) for i in range(10)]
    held = [0.3] * 10
    stay = take(reach + [(0.30, 0.0, 0.05)] * 6, held + [0.3, 0.9, 0.9, 0.9, 0.9, 0.9])
    moved = retract(stay, META, "right")
    check("a hand that stops at the release has not retracted", moved is not None
          and moved["lift_mm"] < 1 and moved["back_mm"] < 1, str(moved))

    out = [(0.30 - 0.006 * i, 0.0, 0.05 + 0.008 * i) for i in range(1, 6)]
    away = take(reach + [(0.30, 0.0, 0.05)] + out, held + [0.3] + [0.9] * 5)
    moved = retract(away, META, "right")
    check("pulled back and up: both measured", moved is not None and abs(moved["lift_mm"] - 40) < 1
          and abs(moved["back_mm"] - 30) < 1, str(moved))

    forward = [(0.30 + 0.006 * i, 0.0, 0.05 + 0.008 * i) for i in range(1, 6)]
    on = take(reach + [(0.30, 0.0, 0.05)] + forward, held + [0.3] + [0.9] * 5)
    moved = retract(on, META, "right")
    check("lifted while pushing on is no drawing back", moved is not None and moved["back_mm"] < 1, str(moved))

    # The jaw shut again before the stop does not hide the release.
    shut_again = take(reach + [(0.30, 0.0, 0.05)] + out + [out[-1]] * 2, held + [0.3] + [0.9] * 5 + [0.2, 0.2])
    moved = retract(shut_again, META, "right")
    check("a jaw shut again at the end still shows the release", moved is not None
          and abs(moved["lift_mm"] - 40) < 1, str(moved))

    # Opened wide, shut on the block, carried, and let go of by shutting
    # further - the block dropped - then drawn back and up: the release is
    # where the jaw left the width it carried the block at.
    # Half a second's carry at least, as a hand carries a thing: twenty
    # frames out to the place.
    carried_out = [(0.20 + 0.005 * i, 0.0, 0.05) for i in range(20)]
    wide = [0.9] * 3 + [0.4] * 17
    dropped = take(carried_out + [(0.30, 0.0, 0.05)] + out, wide + [0.4] + [0.1] * 5)
    moved = retract(dropped, META, "right")
    check("a jaw shut further to let go shows the release", moved is not None
          and abs(moved["lift_mm"] - 40) < 1 and abs(moved["back_mm"] - 30) < 1, str(moved))
    check("and that it let go without opening", moved is not None and moved["opened"] is False, str(moved))
    # ...and one opened wider to let go than it opened to take the block.
    wider = take(carried_out + [(0.30, 0.0, 0.05)] + out, [0.6] * 3 + [0.4] * 17 + [0.4] + [0.9, 0.9, 0.2, 0.1, 0.1])
    moved = retract(wider, META, "right")
    check("a jaw opened wider to let go than to take shows the release", moved is not None
          and abs(moved["lift_mm"] - 40) < 1 and abs(moved["back_mm"] - 30) < 1, str(moved))
    check("and that it opened to let go", moved is not None and moved["opened"] is True, str(moved))
    held_on = take(carried_out + [(0.30, 0.0, 0.05)] * 6, wide + [0.4] * 6)
    moved = retract(held_on, META, "right")
    check("a jaw holding on at the stop let nothing go and is not judged", moved is None, str(moved))

    still_jaw = take(reach + out, [0.5] * 15)
    check("a hand whose jaw hardly moved let go of nothing", retract(still_jaw, META, "right") is None)
    check("a hand never seen is not judged", retract(away, META, "left") is None)

    print("\nended at ready")
    desk = world_T_desk()
    chain = ArmKinematics("right")
    ready = desk_tool("right", desk, chain.forward(np.array([READY_DEG["right"][n.removeprefix("right_")]
                                                             for n in chain.joint_names])))
    answer = {"desk": desk.tolist()}
    back_home = stay + [{"solids": {"handheld_right": {"solved": True, "pose": ready.tolist(),
                                                      "world_pose": ready.tolist()}}, "jaw": {"right": 0.9}}]
    home = end_from_ready(back_home, META, "right", answer)
    check("a hand back on the ready pose's tool ends at ready", home is not None and home["mm"] < 0.1
          and home["deg"] < 0.1, str(home))
    home = end_from_ready(stay, META, "right", answer)
    check("one left over the place does not", home is not None and home["mm"] > UMI.ready_guide_mm, str(home))
    check("a take with no place in the robot's world is not judged so",
          end_from_ready(stay, META, "right", {"desk": None}) is None)

    print("\njoint speed")
    turns = [{"t": i / 50, "flags": {"right": []},
              "joints": {"right_elbow_flex": math.radians(2.0 * i), "right_gripper": math.radians(30.0 * i)}}
             for i in range(10)]
    name, speed = fastest_joint(turns, "right")
    check("the fastest joint is the arm's, not the gripper's", name == "right_elbow_flex" and abs(speed - 100) < 1e-6,
          f"{name} {speed:.1f} deg/s")
    turns[5]["joints"]["right_elbow_flex"] += math.radians(40.0)
    check("a flip between two frames is past the arm's speed",
          fastest_joint(turns, "right")[1] > 300, f"{fastest_joint(turns, 'right')[1]:.0f} deg/s")
    turns[5]["flags"]["right"] = ["lost"]
    check("not across a moment the arm did not follow", abs(fastest_joint(turns, "right")[1] - 100) < 1e-6,
          f"{fastest_joint(turns, 'right')[1]:.0f} deg/s")

    print("\nshoulder_pan from ready")
    seen = take(reach + out, [0.3] * 10 + [0.9] * 5)
    ready = READY_DEG["right"]["shoulder_pan"]
    limit = UMI.keep_warn_pan_from_ready_deg

    def swung(by: float) -> list[str]:
        frames = [{"t": i / 5, "flags": {"right": []},
                   "joints": {"right_shoulder_pan": math.radians(ready + by * i / 14)}} for i in range(15)]
        return _hand(seen, {"trajectory": frames, "hands": ["right"]}, "right", 30.0)[2]

    check("an arm swung past the limit is pointed out", any(w.startswith("shoulder_pan") for w in swung(limit + 10)),
          str(swung(limit + 10)))
    check("one swung short of it is not", not any(w.startswith("shoulder_pan") for w in swung(limit - 10)),
          str(swung(limit - 10)))

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
