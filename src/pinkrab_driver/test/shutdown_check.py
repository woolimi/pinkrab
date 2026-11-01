#!/usr/bin/env python3
"""Check that the walk home can be stopped from the graph after Ctrl-C.

The driver walks every arm holding torque home when it is interrupted, and a
`/set_torque` that lets the arms go ends that walk. That only holds while the
node can still serve a request after the interrupt, so this runs the driver's
own `main` on fake buses, interrupts it as a terminal would, and asks a client
on a context of its own to let both arms go while they walk.

    ROS_DOMAIN_ID=231 python src/pinkrab_driver/test/shutdown_check.py

Run from a shell that has sourced install/setup.bash, so that the built
pinkrab_interfaces is found.
"""
from __future__ import annotations

import os
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rclpy  # noqa: E402
from offline_check import OFFSETS, FakeBus, FakePorts  # noqa: E402
from pinkrab_interfaces.srv import SetTorque  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402

from pinkrab_core import arms as arms_module  # noqa: E402
from pinkrab_core.angles import degrees_to_ticks  # noqa: E402
from pinkrab_core.config import SETTINGS  # noqa: E402
from pinkrab_driver import driver as driver_module  # noqa: E402

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def main() -> int:
    buses: dict[str, FakeBus] = {}

    def make_bus(port):
        bus = FakeBus(port)
        # Held and away from home, so that Ctrl-C has a walk to start.
        bus.torque = True
        bus.present["elbow_flex"] = degrees_to_ticks(
            "left", {"elbow_flex": 90.0}, OFFSETS)["elbow_flex"]
        buses[str(port).rsplit("-", 1)[-1]] = bus
        return bus

    arms_module.make_bus = make_bus
    arms_module.ports.load = lambda *a, **k: FakePorts()
    arms_module.load_offsets = lambda *a, **k: OFFSETS
    driver_module.Arms = arms_module.Arms

    answers: list[bool] = []

    def operator() -> None:
        time.sleep(1.0)
        os.kill(os.getpid(), signal.SIGINT)            # Ctrl-C: the walk home starts
        while not any(len(bus.goals()) for bus in buses.values()):
            time.sleep(0.02)
        context = rclpy.Context()
        rclpy.init(context=context)
        node = rclpy.create_node("shutdown_check_client", context=context)
        client = node.create_client(SetTorque, "set_torque")
        if client.wait_for_service(timeout_sec=3.0):
            future = client.call_async(SetTorque.Request(arm=SetTorque.Request.ARM_BOTH, enable=False))
            executor = SingleThreadedExecutor(context=context)
            executor.add_node(node)
            executor.spin_until_future_complete(future, timeout_sec=3.0)
            answers.append(bool(future.done() and future.result().success))
        node.destroy_node()
        rclpy.shutdown(context=context)

    print("\nthe walk home stops when the arms are let go after Ctrl-C")
    thread = threading.Thread(target=operator, daemon=True)
    thread.start()
    started = time.monotonic()
    driver_module.main()
    took = time.monotonic() - started
    thread.join(timeout=5.0)

    check("the request to let go was answered", answers == [True], str(answers))
    check("both arms are let go", not any(bus.torque for bus in buses.values()),
          str({arm: bus.torque for arm, bus in buses.items()}))
    check("well before the walk would have ended", took < 1.0 + 90.0 / SETTINGS.park_speed_dps,
          f"{took:.1f} s")

    if FAILURES:
        print(f"\nFAILED: {len(FAILURES)}")
        return 1
    print("\nall ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
