#!/usr/bin/env python3
"""Offline check of a rollout waiting for the arms: no step is taken while a joint
of the last pose sent reads far from it or has not been read, the wait ends the
run once it has lasted, and an arm that catches up goes on. No ROS graph, no
policy.

    python src/pinkrab_policy/test/tracking_check.py
"""
from __future__ import annotations

import math
import sys

from pinkrab_policy.rollout import DriveStopped, Tracking

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def main() -> int:
    names = ["left_shoulder_pan", "left_elbow_flex"]
    sent = {"left_shoulder_pan": 0.5, "left_elbow_flex": -0.3, "left_gripper": 0.9}

    print("\nthe arm follows")
    t = Tracking(stale_s=0.25, lag_deg=15.0, hold_s=2.0)
    t.read(names, 0.0)
    check("before anything is sent there is nothing to wait for", t.wait({}, 0.0) is None)
    t.put(sent)
    at = {"left_shoulder_pan": 0.5 - math.radians(3), "left_elbow_flex": -0.3, "left_gripper": 0.2}
    check("a joint a few degrees behind is no reason to wait", t.wait(at, 0.1) is None)
    check("a jaw closed on a block, far from where it was sent, is none either", "left_gripper" not in t.sent)

    print("\nthe arm does not")
    t.read(names, 1.0)
    at = {"left_shoulder_pan": 0.5 - math.radians(40), "left_elbow_flex": -0.3}
    why = t.wait(at, 1.0)
    check("a joint far from where it was sent makes the step wait, and says which",
          why is not None and "left_shoulder_pan" in why, str(why))
    check("the wait goes on", t.wait(at, 2.5) is not None)
    try:
        t.wait(at, 3.2)
        stopped = False
    except DriveStopped as exc:
        stopped = "do not follow" in str(exc)
    check("and ends the run once it has lasted", stopped)

    print("\nthe arm catches up")
    t = Tracking(stale_s=0.25, lag_deg=15.0, hold_s=2.0)
    t.read(names, 0.0)
    t.put(sent)
    t.wait(at, 0.0)
    t.read(names, 1.0)
    caught = {"left_shoulder_pan": 0.5, "left_elbow_flex": -0.3}
    check("steps go on", t.wait(caught, 1.0) is None)
    t.read(names, 2.5)
    check("and the wait starts over the next time", t.wait(at, 2.5) is not None and t.waiting_since == 2.5)

    print("\nthe readings stop")
    t = Tracking(stale_s=0.25, lag_deg=15.0, hold_s=2.0)
    t.read(names, 0.0)
    t.put(sent)
    why = t.wait(caught, 0.1)
    check("fresh readings are enough", why is None)
    why = t.wait(caught, 0.6)
    check("a joint not read lately makes the step wait", why is not None and "no reading" in why, str(why))
    check("a joint never read does too", Tracking(0.25, 15.0, 2.0).unread(names, 0.0) == sorted(names))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s): {FAILURES}")
        return 1
    print("all ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
