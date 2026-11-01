#!/usr/bin/env python3
"""Verify the write path with no servos on the desk.

This is the first code in the project that can move metal, and the ways it can be
wrong are the ways that do not raise: a pose written to an arm whose torque is
off and obeyed later, torque enabled against a goal left over from a previous
session, an angle turned into a tick with the sign of the other arm. So the bus
is replaced with one that records what it was told, and what the driver says is
compared against what it should have said.

    python src/pinkrab_driver/test/offline_check.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC / "pinkrab_core"))

from pinkrab_core import arms as arms_module  # noqa: E402
from pinkrab_core.config import SETTINGS  # noqa: E402
from pinkrab_core.angles import degrees_to_ticks  # noqa: E402

FAILURES: list[str] = []

# A rig's calibration, made up here rather than read from the workspace: this test
# is about the arithmetic and the order of operations, and it should say the same
# thing on a machine that has never had an arm plugged into it.
OFFSETS = {
    "neutral_ticks": 2048,
    "resolution": 4095,
    "offsets_deg": {
        "left": {"shoulder_pan": -3.0, "shoulder_lift": 103.4, "elbow_flex": 94.2,
                 "wrist_flex": -3.7, "wrist_yaw": 96.2, "wrist_roll": 91.1,
                 "gripper": 50.3},
        "right": {"shoulder_pan": 5.9, "shoulder_lift": 104.2, "elbow_flex": 92.7,
                  "wrist_flex": -1.8, "wrist_yaw": 95.6, "wrist_roll": -91.0,
                  "gripper": 50.0},
    },
}


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


class FakeBus:
    """A servo bus that answers and remembers, on a desk with no servos."""

    def __init__(self, port) -> None:
        self.port = port
        self.calls: list[tuple] = []
        self.torque = False
        self.present = dict.fromkeys(
            ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
             "wrist_yaw", "wrist_roll", "gripper"), 2048)

    @property
    def motors(self): return dict.fromkeys(self.present)

    def connect(self, handshake=True): self.calls.append(("connect",))
    def configure_motors(self): self.calls.append(("configure_motors",))
    def set_baudrate(self, rate): self.calls.append(("baudrate", rate))
    def disconnect(self, disable_torque=True): self.calls.append(("disconnect", disable_torque))

    def read(self, register, motor, normalize=True, num_retry=0):
        if register == "Torque_Enable":
            return int(self.torque)
        return self.present[motor]

    def sync_read(self, register, motors=None, normalize=True, num_retry=0):
        return dict(self.present)

    def write(self, register, motor, value, normalize=True, num_retry=0):
        self.calls.append(("write", register, motor, value))

    def sync_write(self, register, values, normalize=True, num_retry=0):
        self.calls.append(("sync_write", register, dict(values)))
        # A servo goes where it is sent, at once: enough for the walk home to
        # be watched arriving without a clock in the test.
        if register == "Goal_Position":
            self.present.update(values)

    def enable_torque(self, motors=None, num_retry=0):
        self.torque = True
        self.calls.append(("enable_torque",))

    def disable_torque(self, motors=None, num_retry=0):
        self.torque = False
        self.calls.append(("disable_torque",))

    def goals(self) -> list[dict]:
        return [c[2] for c in self.calls if c[0] == "sync_write" and c[1] == "Goal_Position"]


class FakePorts:
    def assigned(self): return ("left", "right")
    def __getitem__(self, arm): return Path(f"/dev/fake-{arm}")


def main() -> int:
    buses: dict[str, FakeBus] = {}

    def make_bus(port):
        bus = FakeBus(port)
        buses[str(port).rsplit("-", 1)[-1]] = bus
        return bus

    arms_module.make_bus = make_bus
    arms_module.ports.load = lambda *a, **k: FakePorts()
    arms_module.load_offsets = lambda *a, **k: OFFSETS

    arms = arms_module.Arms()

    print("\nopening a port changes nothing")
    check("both arms are on the bus", arms.arms == ("left", "right"), str(arms.arms))
    check("torque starts off", arms.torque_held() == {"left": False, "right": False})
    check("and nothing has been asked of a servo",
          not any(b.goals() for b in buses.values()))
    writes = {(c[1], c[2]): c[3] for c in buses["left"].calls if c[0] == "write"}
    check("the servo registers are written as LeRobot writes them",
          writes.get(("P_Coefficient", "elbow_flex")) == 16
          and writes.get(("Max_Torque_Limit", "gripper")) == 500
          and writes.get(("Max_Torque_Limit", "wrist_yaw")) == 1000,
          "gains 16/0/32, the gripper's protection, the other joints' factory values")
    check("with torque left off", not any(b.torque for b in buses.values()))

    print("\na pose for a limp arm is dropped, not queued")
    sent = arms.write({"left_elbow_flex": 0.3})
    check("write returns nothing", sent == [], str(sent))
    check("and the servo was never given a goal",
          not buses["left"].goals(),
          "a goal stored now is a goal obeyed the moment torque comes on")

    print("\nturning torque on holds the arm where it already is")
    buses["left"].present["elbow_flex"] = 1500
    arms.torque(True, "left")
    calls = [c[0] for c in buses["left"].calls]
    goals = buses["left"].goals()
    check("the goal is written before the torque is enabled",
          "sync_write" in calls and "enable_torque" in calls
          and calls.index("sync_write") < calls.index("enable_torque"), str(calls[-4:]))
    check("and the goal is where the arm is",
          bool(goals) and goals[-1]["elbow_flex"] == 1500, str(goals[-1] if goals else None))
    limit = ("write", "Torque_Limit", "gripper", SETTINGS.gripper_torque_limit)
    check("the gripper's Torque_Limit is written before the torque is enabled",
          limit in buses["left"].calls
          and buses["left"].calls.index(limit) < calls.index("enable_torque"),
          str([c for c in buses["left"].calls if c[0] == "write"][-2:]))
    check("the other arm was left alone", buses["right"].calls[-1][0] != "enable_torque")
    check("only the left is holding", arms.torque_held() == {"left": True, "right": False})

    print("\nwriting a pose")
    wanted = math.radians(30.0)
    sent = arms.write({"left_elbow_flex": wanted, "right_elbow_flex": wanted})
    expected = degrees_to_ticks("left", {"elbow_flex": 30.0}, OFFSETS)["elbow_flex"]
    check("only the arm with torque is written to", sent == ["left_elbow_flex"], str(sent))
    check("the angle becomes the tick the calibration says",
          buses["left"].goals()[-1] == {"elbow_flex": expected},
          f"{buses['left'].goals()[-1]} vs {expected}")

    arms.torque(True, "right")
    before = len(buses["right"].goals())
    arms.write({"right_shoulder_pan": math.radians(20.0)})
    got = buses["right"].goals()[-1]["shoulder_pan"]
    own = degrees_to_ticks("right", {"shoulder_pan": 20.0}, OFFSETS)["shoulder_pan"]
    check("the right arm is written through its own offsets",
          len(buses["right"].goals()) == before + 1 and got == own,
          f"{got} vs {own}")
    check("and that is not the left arm's answer for the same angle",
          got != degrees_to_ticks("left", {"shoulder_pan": 20.0}, OFFSETS)["shoulder_pan"])

    print("\nwhat is refused")
    before = len(buses["left"].goals())
    arms.write({"left_wrist_of_nothing": 0.0})
    check("a joint of no robot is not written", len(buses["left"].goals()) == before)
    arms.write({"left_elbow_flex": math.radians(2000.0)})
    check("an angle off the end of the encoder is not written",
          len(buses["left"].goals()) == before,
          "a tick outside 0..4095 means some other pose entirely")

    print("\nthe stop")
    arms.torque(False)
    check("both arms go limp", arms.torque_held() == {"left": False, "right": False})
    check("by asking the servos, not by forgetting",
          buses["left"].calls[-1][0] == "disable_torque"
          and buses["right"].calls[-1][0] == "disable_torque")
    before = len(buses["left"].goals())
    arms.write({"left_elbow_flex": 0.1})
    check("and nothing is written afterwards", len(buses["left"].goals()) == before)

    print("\nthe walk home on the way down")
    from itertools import pairwise

    from pinkrab_core.angles import ticks_to_degrees
    arms.torque(True, "left")
    buses["left"].present["elbow_flex"] = degrees_to_ticks(
        "left", {"elbow_flex": 90.0}, OFFSETS)["elbow_flex"]
    arms.start()
    import time as _time
    _time.sleep(0.1)  # a reading or two
    left_home = {f"left_{j}": 0.0 for j in buses["left"].present}
    right_home = {f"right_{j}": 0.0 for j in buses["right"].present}
    before_right = len(buses["right"].goals())
    started = len(buses["left"].goals())
    late = arms.park({**left_home, **right_home}, speed_dps=600.0, timeout_s=3.0)
    arms.stop()
    walked = buses["left"].goals()[started:]
    check("the arm that holds torque gets there", late == [], str(late))
    check("the limp arm is not sent anywhere", len(buses["right"].goals()) == before_right)
    check("it is walked, not sent", len(walked) > 3, f"{len(walked)} goals")
    degrees = [ticks_to_degrees("left", g, OFFSETS)["elbow_flex"] for g in walked if "elbow_flex" in g]
    jumps = [abs(b - a) for a, b in pairwise(degrees)]
    check("and no goal is more than a tick of the ramp from the last",
          bool(jumps) and max(jumps) <= 600.0 / 60 * 1.5,
          f"largest {max(jumps) if jumps else 0:.2f} deg")
    check("ending at home", abs(degrees[-1]) < 3.0 if degrees else False,
          f"{degrees[-1] if degrees else None}")
    check("and the port is only closed afterwards",
          buses["left"].calls[-1][0] == "disconnect")


    if FAILURES:
        print(f"\nFAILED: {len(FAILURES)}")
        for name in FAILURES:
            print(f"  - {name}")
        print()
        return 1
    print("\nThe write path does what it says.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
