#!/usr/bin/env python3
"""Offline check of the bridge's calls to the gate and the driver: a stop is made
until it is answered yes and holds back what was asked after it, a request the
studio says twice is one call, a request the service is not there for is dropped
unless it is a stop, and a page target for an arm without torque is not driven
toward. No ROS graph: the bridge's handlers are called on their own, with
clients that remember.

    python src/pinkrab_bridge/test/outbox_check.py
"""
from __future__ import annotations

import collections
import sys
import threading
import time
from types import SimpleNamespace

import rclpy.time
from pinkrab_interfaces.msg import TeleopState

import pinkrab_bridge.bridge as bridge

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


class Future:
    def __init__(self) -> None:
        self.answer = None

    def done(self) -> bool:
        return self.answer is not None

    def exception(self):
        return None

    def result(self):
        return self.answer


class Client:
    def __init__(self, ready: bool = True) -> None:
        self.ready = ready
        self.calls: list[tuple] = []
        self.removed: list[Future] = []

    def service_is_ready(self) -> bool:
        return self.ready

    def call_async(self, call):
        future = Future()
        self.calls.append((call, future))
        return future

    def remove_pending_request(self, future) -> None:
        self.removed.append(future)


def node() -> bridge.Bridge:
    quiet = SimpleNamespace(info=lambda *a, **k: None, warn=lambda *a, **k: None)
    b = bridge.Bridge.__new__(bridge.Bridge)
    b.get_logger = lambda: quiet
    b._lock = threading.Lock()
    b._outbox = collections.deque()
    b._requests = collections.deque()
    b._seen = {}
    b.mode, b.estop, b.torque, b.follow = Client(), Client(), Client(), Client()
    b._teleop, b._own_disarm = None, False
    b._goal, b._walk, b._legs, b._walk_sent, b._landed = {}, None, [], {}, set()
    b._settle_since = b._blind_since = None
    b._wanted = {}
    b._goal_live = {}
    b._torque = None
    b._angles, b._read_at = {}, {}
    b._jaws_wide, b._jaws_opened, b._walk_span = {}, set(), None
    b._still_at, b._still_angles = None, {}
    return b


def main() -> int:
    bridge.RESEND_S = 0.05

    print("\na stop is made until it is answered yes")
    b = node()
    b.estop.ready = False
    b._ask({"kind": "estop"})
    b._flush()
    check("with no service to hear it, nothing is sent and the stop stays", not b.estop.calls and len(b._outbox) == 1)
    b.estop.ready = True
    b._flush()
    check("when the service is there it is sent", len(b.estop.calls) == 1)
    time.sleep(0.06)
    b._flush()
    check("unanswered, it is sent again and the old request let go",
          len(b.estop.calls) == 2 and b.estop.removed == [b.estop.calls[0][1]])
    b.estop.calls[1][1].answer = SimpleNamespace(success=False, message="no")
    time.sleep(0.06)
    b._flush()
    check("answered no, it is sent again", len(b.estop.calls) == 3)
    b.estop.calls[2][1].answer = SimpleNamespace(success=True, message="stopped")
    b._flush()
    check("answered yes, it is done", not b._outbox)

    print("\nwhat is asked after a stop waits for it")
    b = node()
    b._ask({"kind": "estop"})
    b._flush()
    b._ask({"kind": "mode", "mode": TeleopState.MODE_STUDIO})
    b._flush()
    check("a mode asked behind an unanswered stop is not sent", not b.mode.calls and len(b._outbox) == 2)
    b.estop.calls[0][1].answer = SimpleNamespace(success=True, message="")
    b._flush()
    check("and is sent once the stop is answered", len(b.mode.calls) == 1 and not b._outbox)

    print("\na stop nobody is there for does not hold back another service's stop")
    b = node()
    b.estop.ready = False
    b._ask({"kind": "estop"})
    b._ask({"kind": "torque", "arm": "left", "enable": False})
    b._ask({"kind": "mode", "mode": TeleopState.MODE_STUDIO})
    b._flush()
    check("the torque off is sent", len(b.torque.calls) == 1)
    check("the mode behind the unanswered estop is not", not b.mode.calls)

    print("\nidle and torque off are stops, torque on and the other modes are not")
    b = node()
    b._ask({"kind": "mode", "mode": TeleopState.MODE_IDLE})
    b._ask({"kind": "torque", "arm": "left", "enable": False})
    b._ask({"kind": "torque", "arm": "left", "enable": True})
    b._ask({"kind": "follow"})
    check("as the outbox has them", [e["stop"] for e in b._outbox] == [True, True, False, False])

    print("\na request nobody is there for")
    b = node()
    b.mode.ready = False
    b._ask({"kind": "mode", "mode": TeleopState.MODE_HANDHELD})
    b._flush()
    check("is dropped, not kept for later", not b._outbox and not b.mode.calls)
    b._ask({"kind": "mode", "mode": TeleopState.MODE_IDLE})
    b._flush()
    check("unless it is a stop", len(b._outbox) == 1)
    b = node()
    b.estop.ready = False
    b._ask({"kind": "estop"})
    b._ask({"kind": "mode", "mode": TeleopState.MODE_STUDIO})
    b._outbox[1]["at"] -= bridge.BEHIND_STOP_TTL_S + 1.0
    b._flush()
    b.estop.ready = True
    b._flush()
    b.estop.calls[0][1].answer = SimpleNamespace(success=True, message="")
    b._flush()
    check("and what waited behind a stop too long is not done when the stop is heard",
          not b._outbox and not b.mode.calls)

    print("\na request the studio says again")
    b = node()
    said = [{"requests": [{"kind": "estop", "token": "1:1"}]}] * 3 + [{"requests": [{"kind": "estop", "token": "2:1"}]}]
    b._get = lambda path: said.pop(0)
    for _ in range(4):
        b._poll()
    check("is one call per token", [r["token"] for r in b._requests] == ["1:1", "2:1"], str(list(b._requests)))

    print("\na page target for an arm that holds no torque")
    b = node()
    published = []
    b.commands = SimpleNamespace(publish=published.append)
    b.get_clock = lambda: SimpleNamespace(now=lambda: rclpy.time.Time())
    b._goal = {"left_elbow_flex": 0.5, "right_elbow_flex": 0.5}
    b._torque = {"left": False, "right": True}
    b._drive()
    check("is not driven toward", len(published) == 1 and list(published[0].name) == ["right_elbow_flex"],
          str([list(m.name) for m in published]))
    check("and is dropped from the goal", list(b._goal) == ["right_elbow_flex"])

    print("\na page target the gate passes none of")
    b = node()
    b.commands = SimpleNamespace(publish=lambda m: None)
    b.get_clock = lambda: SimpleNamespace(now=lambda: rclpy.time.Time())
    b._goal = {"right_elbow_flex": 0.5}
    b._goal_live = {"right_elbow_flex": time.monotonic() - bridge.GOAL_REFUSED_S - 1.0}
    b._drive()
    check("is let go after a while", not b._goal)
    b._goal = {"right_elbow_flex": 0.5}
    b._goal_live = {"right_elbow_flex": time.monotonic()}
    b._drive()
    check("but not while it is new", bool(b._goal))

    print("\na walk to ready that opens the jaws on the way")
    clock = [100.0]
    real_time = bridge.time
    bridge.time = SimpleNamespace(monotonic=lambda: clock[0])
    try:
        b = node()
        published = []
        b.commands = SimpleNamespace(publish=published.append)
        b.get_clock = lambda: SimpleNamespace(now=lambda: rclpy.time.Time())
        b._ask = lambda request: b._start_walk(request) if request.get("kind") == "goto" else None
        b._angles = {"right_shoulder_lift": 2.0, "right_elbow_flex": -1.0, "right_gripper": 0.3}
        ready = {"right_shoulder_lift": 1.6, "right_elbow_flex": -0.66, "right_gripper": 0.0}
        b._start_walk({"poses": [ready], "speed_dps": 30.0, "jaws_wide": {"right_gripper": 1.4}})
        trace = []
        for _ in range(3000):
            clock[0] += 1 / 60
            b._read_at = dict.fromkeys(b._angles, clock[0])
            b._drive()
            if published:
                last = published[-1]
                b._angles.update(zip(last.name, last.position))
            trace.append((dict(b._angles), b._walk is not None))
            if b._walk is None:
                break
        lift = [a["right_shoulder_lift"] for a, _ in trace]
        jaw = [a["right_gripper"] for a, _ in trace]
        widest = max(range(len(jaw)), key=jaw.__getitem__)
        check("the walk ends at ready", b._walk is None and abs(lift[-1] - 1.6) < 0.01, f"{lift[-1]:.3f}")
        check("the jaw opens wide on the way", jaw[widest] > 1.35, f"{jaw[widest]:.3f}")
        check("while the arm is already moving", 1.6 < lift[widest] < 2.0 - 0.01, f"{lift[widest]:.3f}")
        check("and is closed again when the arm arrives", abs(jaw[-1]) < 0.01, f"{jaw[-1]:.3f}")
        check("the arm does not wait for the jaw before it sets off",
              lift[len(lift) // 10] < 2.0 - 0.01, f"{lift[len(lift) // 10]:.3f}")
    finally:
        bridge.time = real_time

    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s): {FAILURES}")
        return 1
    print("all ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
