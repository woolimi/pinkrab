#!/usr/bin/env python3
"""Check the gate node's emergency stop on a graph of its own: the gate, a stand-in
for the driver and a client, in one process on whatever ROS_DOMAIN_ID the shell
names (use one nothing else is on). The stop idles the gate and the gate asks the
driver to let both arms go; a driver that does not answer is asked again, one that
answers no is asked again, and a return to idle ends the asking. A stand-in
driver also tells the gate which arms hold torque.

    ROS_DOMAIN_ID=231 python src/pinkrab_safety/test/node_check.py

Run from a shell that has sourced install/setup.bash, so that the built
pinkrab_interfaces is found.
"""
from __future__ import annotations

import sys
import threading
import time

import rclpy
from pinkrab_interfaces.msg import TeleopState, TorqueState
from pinkrab_interfaces.srv import Estop, SetTeleopMode, SetTorque
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile

from pinkrab_core.config import SETTINGS
from pinkrab_safety import gate as gate_module

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


class StandInDriver(Node):
    def __init__(self) -> None:
        super().__init__("stand_in_driver")
        self.asked: list[tuple[str, bool]] = []
        self.answers: list[str] = []      # what to do with the next calls: "ignore", "no" or "yes"
        self.create_service(SetTorque, "set_torque", self._on_torque)
        self.state = self.create_publisher(
            TorqueState, "torque_state", QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def say(self, held: dict[str, bool]) -> None:
        message = TorqueState()
        message.arms, message.enabled = sorted(held), [held[a] for a in sorted(held)]
        self.state.publish(message)

    def _on_torque(self, request, response):
        self.asked.append((request.arm, request.enable))
        answer = self.answers.pop(0) if self.answers else "yes"
        if answer == "ignore":
            time.sleep(0.8)   # longer than the gate waits
        response.success = answer != "no"
        response.message = "stand-in"
        return response


def until(condition, seconds: float = 4.0) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


def main() -> int:
    SETTINGS.pin_ros_middleware()
    rclpy.init()
    gate = gate_module.GateNode()
    driver = StandInDriver()
    client = Node("stand_in_client")
    executor = MultiThreadedExecutor(num_threads=4)
    for node in (gate, driver, client):
        executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    mode = client.create_client(SetTeleopMode, "set_teleop_mode")
    estop = client.create_client(Estop, "estop")
    seen: list[TeleopState] = []
    client.create_subscription(TeleopState, "teleop_state", seen.append, 10)
    assert mode.wait_for_service(5.0) and estop.wait_for_service(5.0)

    def call(service, request):
        future = service.call_async(request)
        assert until(future.done), "no answer"
        return future.result()

    print("\nthe gate hears which arms hold torque")
    driver.say({"left": False, "right": True})
    check("and keeps it", until(lambda: gate.gate.torque == {"left": False, "right": True}), str(gate.gate.torque))

    print("\nthe stop")
    call(mode, SetTeleopMode.Request(mode=TeleopState.MODE_STUDIO))
    driver.answers = ["ignore", "no", "yes"]
    check("the gate is armed", gate.gate.armed)
    answer = call(estop, Estop.Request())
    check("the stop is answered at once, whatever the driver does", answer.success)
    check("and idles the gate", not gate.gate.armed and gate.gate.stopped)
    check("the driver is asked to let both arms go",
          until(lambda: driver.asked[:1] == [("both", False)]), str(driver.asked))
    check("asked again when it did not answer, and again when it said no",
          until(lambda: len(driver.asked) >= 3), str(driver.asked))
    check("and the asking ends once it says yes", until(lambda: not gate._limp_due))
    n = len(driver.asked)
    time.sleep(1.2)
    check("nothing is asked after that", len(driver.asked) == n, f"{n} -> {len(driver.asked)}")

    print("\nback to idle")
    call(mode, SetTeleopMode.Request(mode=TeleopState.MODE_IDLE))
    driver.answers = ["no"] * 50
    n = len(driver.asked)
    call(estop, Estop.Request())
    check("a stop the driver keeps refusing is asked again and again",
          until(lambda: len(driver.asked) >= n + 3), str(len(driver.asked) - n))
    call(mode, SetTeleopMode.Request(mode=TeleopState.MODE_IDLE))
    n = len(driver.asked)
    time.sleep(1.2)
    check("until a deliberate return to idle", len(driver.asked) <= n + 1, f"{len(driver.asked) - n} more")

    executor.shutdown()
    for node in (gate, driver, client):
        node.destroy_node()
    rclpy.try_shutdown()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s): {FAILURES}")
        return 1
    print("all ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
