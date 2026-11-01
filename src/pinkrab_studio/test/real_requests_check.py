#!/usr/bin/env python3
"""Offline check of what the page asks of the robot on its way to the bridge: a
request is handed over once and goes stale, a stop is handed over again at every
poll until the robot's own state shows it done, and none of them is dropped for
being old before the bridge has been handed it.

    python src/pinkrab_studio/test/real_requests_check.py
"""
from __future__ import annotations

import sys

import pinkrab_studio.real as real

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def kinds(requests: list[dict]) -> list[str]:
    return [r["kind"] for r in requests]


def main() -> int:
    robot = real.RealRobot(expected=True)

    print("\nan ordinary request")
    robot.request("mode", mode=real.MODE_STUDIO)
    first = robot.take_requests()
    check("is handed over once", kinds(first) == ["mode"] and not robot.take_requests())
    check("with a token and nothing of the studio's bookkeeping",
          "token" in first[0] and "at" not in first[0] and "delivered" not in first[0], str(first[0]))
    robot.request("mode", mode=real.MODE_STUDIO)
    robot._requests[0]["at"] -= real.REQUEST_TTL_S + 1.0
    check("and is dropped once the bridge has been away too long", not robot.take_requests())

    print("\na stop")
    robot.request("estop")
    robot._requests[0]["at"] -= real.STOP_KEEP_S + 100.0
    first = robot.take_requests()
    check("is handed over however old it is when the bridge first comes for it", kinds(first) == ["estop"])
    again = robot.take_requests()
    check("and again at the next poll, the response may have been lost", kinds(again) == ["estop"]
          and again[0]["token"] == first[0]["token"])
    robot.push({}, teleop={"armed": False, "mode": 0, "reason": "stopped"}, torque={"left": False, "right": True})
    check("still, while an arm holds torque", kinds(robot.take_requests()) == ["estop"])
    robot.push({}, teleop={"armed": False, "mode": 0, "reason": "stopped"}, torque={"left": False, "right": False})
    check("until the robot shows it carried out", not robot.take_requests())

    robot.request("mode", mode=real.MODE_IDLE)
    robot.take_requests()
    check("idle is handed over again while the gate has not left its stop",
          kinds(robot.take_requests()) == ["mode"])
    robot.push({}, teleop={"armed": False, "mode": 0, "reason": "no mode chosen"})
    check("and done once it has", not robot.take_requests())

    robot.request("torque", arm="left", enable=False)
    robot.push({}, torque={"left": True, "right": False})
    robot.take_requests()
    check("torque off for an arm is handed over again while that arm holds", kinds(robot.take_requests()) == ["torque"])
    robot.push({}, torque={"left": False, "right": False})
    check("and done when it does not", not robot.take_requests())

    robot = real.RealRobot(expected=True)
    robot.request("estop")
    robot.take_requests()
    robot._requests[0]["delivered"] -= real.STOP_KEEP_S + 1.0
    check("a stop the robot never shows done is let go after a while", not robot.take_requests())

    print("\na studio and a bridge that outlive each other")
    other = real.RealRobot(expected=True)
    robot.request("estop")
    other.request("estop")
    check("a studio started again does not repeat a token",
          robot.take_requests()[0]["token"] != other.take_requests()[0]["token"])

    print("\nback to simulation")
    robot = real.RealRobot(expected=True)
    robot.request("estop")
    robot.request("mode", mode=real.MODE_HANDHELD)
    robot.forget()
    check("keeps the stops", kinds(robot.take_requests()) == ["estop"])

    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s): {FAILURES}")
        return 1
    print("all ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
