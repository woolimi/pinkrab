"""The servo bus on the graph, and nothing else.

Readings out on `/joint_states`, poses in on `/joint_commands_checked`, torque on
request. What this node does not have is an opinion: joint bands, contact and the
ramp were applied by `pinkrab_safety` before a pose got here, and a driver that
also made those decisions would be the hardware deciding policy - the thing that
cannot be undone later, because by then the bands live next to the serial port.

The checked topic is the only subscription. There is no path from
`/joint_commands` to a servo that does not cross the gate, so a gate that is not
running is a robot that takes no new commands rather than one taking them
unchecked. A pose whose stamp is older than `command_timeout_s` is dropped for
the same reason: a command that queued up somewhere is a statement about a
moment that has passed.

Torque is left as the port found it, which on a rig that was just powered up is
off, and a limp arm stays limp until somebody asks for torque. Opening a port
should not change what the metal is doing, and the arms hang on a wall. An arm
that was already holding is reported as holding, so that it is parked and let go
like any other.

Torque goes off again when this node is taken down - after the arms have been
walked home. A servo holds its last goal with nobody watching it otherwise, and
an arm left rigid after the bringup is gone is an arm holding that pose into
whatever moves in front of it; an arm let go of mid-air is an arm on the desk.
So Ctrl-C on the launch walks every arm that is holding torque to the home pose,
at `park_speed_dps`, and lets go once it is there. An arm that is not home when
`park_timeout_s` is up keeps its torque and says so. The node keeps answering
services during the walk, and commands from the graph are not written: a
`/set_torque` that lets an arm go ends the walk for it, which is the way to stop
one, and no request to hold torque is taken meanwhile. The walk is straight in
joint space, ramped here rather than through the gate - which may already be
gone - and checked against nothing but the encoder's range. Home is the
description's zero, every joint at 0, where the arms park against themselves -
not the ready pose the page opens at. From ready the way home was walked
against the scene and is clear of the robot itself; it is not clear of a cup
left under an arm. A process that is killed rather than shut down never gets
here, and those servos keep holding.

    ros2 run pinkrab_driver driver
    ros2 service call /set_torque pinkrab_interfaces/srv/SetTorque "{arm: left, enable: true}"

The emergency stop is the gate's `/estop`, which idles the gate and asks this node
to let both arms go through `/set_torque`.
"""

from __future__ import annotations

import signal

import rclpy
from pinkrab_interfaces.msg import TorqueState
from pinkrab_interfaces.srv import SetTorque
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState

from pinkrab_core.arms import Arms, ArmsError
from pinkrab_core.config import SETTINGS


class DriverNode(Node):
    def __init__(self, arms: Arms) -> None:
        super().__init__("pinkrab_driver")
        self.arms = arms
        self.declare_parameter("command_timeout_s", 0.5)
        self.timeout = float(self.get_parameter("command_timeout_s").value)

        self.states = self.create_publisher(JointState, "joint_states", 10)
        self.torque_state = self.create_publisher(
            TorqueState, "torque_state",
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.create_subscription(
            JointState, "joint_commands_checked", self._on_checked, 10
        )
        self.create_service(SetTorque, "set_torque", self._on_torque)
        # True while the walk home runs on the way down: nothing from the graph
        # is written then, and no torque is taken.
        self._parking = False

        # Published as the bus answers rather than on a timer of its own: the
        # poll rate is what the readings cost, and republishing between two of
        # them would be the same numbers dressed as news.
        self.arms.listener = self._publish
        self.arms.start()
        self._publish_torque()
        # A failed read publishes nothing, and the bus still takes writes: an
        # arm can be driven on from a reading that stopped coming. Said once a
        # second while it happens, which arm and why.
        self.create_timer(1.0, self._report_reads)
        held = [arm for arm, on in self.arms.torque_held().items() if on]
        self.get_logger().info(
            f"{', '.join(self.arms.arms)} on the bus at {SETTINGS.arm_poll_hz} Hz, "
            + (f"torque already on: {', '.join(held)}" if held else "torque off")
        )

    # --- out ----------------------------------------------------------------
    def _report_reads(self) -> None:
        for arm, (count, why) in self.arms.failed_reads().items():
            self.get_logger().warn(f"{arm} arm: {count} read(s) of the bus failed in the last second - {why}")

    def _publish(self, angles: dict[str, float]) -> None:
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = sorted(angles)
        message.position = [float(angles[n]) for n in message.name]
        self.states.publish(message)

    def _publish_torque(self) -> None:
        held = self.arms.torque_held()
        message = TorqueState()
        message.stamp = self.get_clock().now().to_msg()
        message.arms = sorted(held)
        message.enabled = [bool(held[a]) for a in message.arms]
        self.torque_state.publish(message)

    # --- in -----------------------------------------------------------------
    def _on_checked(self, message: JointState) -> None:
        if self._parking:
            return
        if not message.name or len(message.name) != len(message.position):
            self.get_logger().warn("a checked command needs matching name and position")
            return
        stamp = message.header.stamp
        if stamp.sec or stamp.nanosec:
            age = (self.get_clock().now().nanoseconds
                   - (stamp.sec * 1_000_000_000 + stamp.nanosec)) / 1e9
            if age > self.timeout:
                self.get_logger().warn(
                    f"dropped a command {age:.2f}s old - is the gate keeping up?",
                    throttle_duration_sec=1.0,
                )
                return
        pose = dict(zip(message.name, (float(v) for v in message.position)))
        try:
            self.arms.write(pose)
        except ArmsError as exc:
            self.get_logger().error(str(exc))

    # --- torque ---------------------------------------------------------------
    def _on_torque(self, request, response):
        if self._parking and request.enable:
            response.success = False
            response.message = "the arms are being parked - torque is not taken now"
            return response
        arm = (request.arm or SetTorque.Request.ARM_BOTH).strip().lower()
        which = None if arm == SetTorque.Request.ARM_BOTH else arm
        if which is not None and which not in self.arms.arms:
            response.success = False
            response.message = f"no arm called {arm!r} is on the bus"
            return response
        try:
            changed = self.arms.torque(bool(request.enable), which)
        except ArmsError as exc:
            response.success, response.message = False, str(exc)
            self.get_logger().error(str(exc))
            # An arm that did let go is still said to have.
            self._publish_torque()
            return response
        response.success = bool(changed)
        self._publish_torque()
        state = "on" if request.enable else "off"
        response.message = f"torque {state}: {', '.join(changed) or 'nothing'}"
        self.get_logger().info(response.message)
        return response

    def destroy_node(self) -> bool:
        # Home, then let go, then close the port: see the module docstring.
        holding = [arm for arm, on in self.arms.torque_held().items() if on]
        if holding:
            self.get_logger().warn(
                f"walking {', '.join(holding)} home at {SETTINGS.park_speed_dps:g} deg/s"
            )
            self._parking = True
            late = self.arms.park(
                self._home(),
                SETTINGS.park_speed_dps, SETTINGS.park_timeout_s,
                progress=lambda deg: self.get_logger().info(f"  {deg:.0f} deg to go"),
                poll=self._serve,
            )
            for arm in late:
                self.get_logger().error(
                    f"{arm} arm is not home after {SETTINGS.park_timeout_s:g}s - "
                    "keeping its torque: it is still holding when this node is gone, and a driver "
                    "started again takes it as holding, to be let go through /set_torque"
                )
            still = self.arms.torque_held()
            for arm in holding:
                if arm in late:
                    continue
                if not still.get(arm):
                    self.get_logger().warn(f"{arm} arm let go during the walk home, where it stood")
                    continue
                try:
                    self.arms.torque(False, arm)
                except ArmsError as exc:
                    self.get_logger().error(f"could not let go of {arm}: {exc}")
                else:
                    self.get_logger().warn(f"{arm} arm home, torque off")
        self.arms.stop()
        return super().destroy_node()

    def _serve(self) -> None:
        """The requests that come in during the walk home, answered between its
        steps: the torque services, which let an arm go and so end its walk."""
        try:
            rclpy.spin_once(self, timeout_sec=0)
        except Exception as exc:  # noqa: BLE001 - a request that fails is that request's
            # failure; the walk home goes on
            self.get_logger().error(f"during the walk home: {type(exc).__name__}: {exc}")

    def _home(self) -> dict[str, float]:
        """Every joint at zero: the description's origin, where the arms park."""
        return {
            f"{arm}_{joint}": 0.0
            for arm in self.arms.arms for joint in SETTINGS.joint_names
        }


def main(argv: list[str] | None = None) -> None:
    SETTINGS.pin_ros_middleware()
    # rclpy's own signal handlers shut the context down on the first SIGINT,
    # and a context that is down serves nothing - not the /set_torque that is
    # the way to stop the walk home. So the signals are this function's: either
    # one ends the spin as KeyboardInterrupt, with the context still up.
    rclpy.init(args=argv, signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        arms = Arms()
    except ArmsError as exc:
        print(f"no arms to drive: {exc}")
        rclpy.try_shutdown()
        return
    node = DriverNode(arms)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # One interrupt is the order to go home; a second must not cut the
        # walk short. Under `ros2 launch` in a terminal there is always a
        # second: ctrl-c reaches the whole foreground group, and launch then
        # sends its own SIGINT to each process on top of it. Measured, the
        # second landed inside the walk and left an arm holding torque halfway
        # home. SIGTERM still ends it - launch sends that after the timeout
        # the launch file gives the walk.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
