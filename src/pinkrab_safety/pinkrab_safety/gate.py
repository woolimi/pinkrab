"""Every command to the servos crosses this, or does not arrive.

`/joint_commands` comes in, `/joint_commands_checked` goes out, and in between a
pose is held against three things: the joint bands, the scene the studio draws,
and how far the arm is allowed to move in one step. What survives is what the
driver is allowed to see - the driver subscribes to the checked topic alone, so
a gate that is not running is a robot that takes no new commands rather than one
taking them unchecked.

Three checks, and each is here for a different failure.

The joint bands come from the model, and they are the only thing standing
between the arm and the pairs MuJoCo will not report: contact between a body and
its own parent is skipped, so the forearm bracket meeting the truss is a limit's
job and not a contact's. The jaw closing onto the camera bracket is the same
case and the tightest of them - 2.1 mm at the closed end, held by the gripper's
band alone, because the bracket is a geom of the link the jaw hangs off.

The scene is `pinkrab_mujoco`'s, which is the studio's - one `scene.xml`. An
operator watching the page stop at the desk and the arm stop somewhere else is
an operator with two sets of facts. What the scene does not hold - a cable, a
tripod, a cup, a hand - it cannot check, so a pose that passes here is clear of
the ROBOT, not of the room.

The ramp is the same one the simulation walks its actuators along - the one
joint speed in the config - for the same reason: the gap between where an arm is
and where it has just been asked to go is most of a joint's travel the moment a
slider is thrown, and an arm closing that gap as fast as its servos allow is an
arm swinging at the wall it hangs on. It is a speed, not a step per command, so
it means the same thing at whatever rate commands arrive.

A policy's commands are shaped as well, unless inference.yaml's smoothing is
off: each one is only where the joints are headed, and a timer moves the
setpoint on toward it at policy_shaping_hz - its speed under the policy's,
its acceleration and jerk under the file's limits - holding every pose it
sends to the same scene. A policy answers in chunks, and where one meets the
next the command can turn at once; shaped, the arm does not jolt there. The
file is read again each time a policy takes the arms. A hand's commands - the
handheld, the studio's jog and walks, a correction - are shaped the same way
under the config's teleop limits, unless teleop_smoothing is off; their jaws
are only ramped.

What an operator is allowed to believe lives here too. Armed, the mode, and why
the last command did not go through are published as `TeleopState` and nowhere
else: a page that kept its own copy would go on showing ARM after the stop was
hit from another window.

    ros2 run pinkrab_safety gate
    ros2 run pinkrab_safety gate --ros-args -p max_speed_dps:=120.0
    ros2 run pinkrab_safety gate --ros-args -p max_speed_dps:=0.0   # no ramp
"""

from __future__ import annotations

import math
from pathlib import Path

import rclpy
from pinkrab_interfaces.msg import TeleopState
from pinkrab_interfaces.srv import Estop, SetTeleopMode
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState

from pinkrab_core import inference
from pinkrab_core.config import SETTINGS
from pinkrab_mujoco.scene import Scene, SceneError

from .policy import MODE_CORRECT, MODE_HANDHELD, MODE_IDLE, MODE_POLICY, MODE_STUDIO, Gate


class GateNode(Node):
    def __init__(self) -> None:
        super().__init__("pinkrab_safety")
        assert (MODE_IDLE, MODE_HANDHELD, MODE_STUDIO, MODE_POLICY, MODE_CORRECT) == (
            TeleopState.MODE_IDLE, TeleopState.MODE_HANDHELD, TeleopState.MODE_STUDIO, TeleopState.MODE_POLICY,
            TeleopState.MODE_CORRECT,
        ), "the gate's modes have drifted from TeleopState's"

        self.declare_parameter("scene", str(SETTINGS.scene_file))
        self.declare_parameter("max_speed_dps", float(SETTINGS.max_joint_speed_dps))
        self.declare_parameter("state_hz", 10.0)

        scene_path = Path(self.get_parameter("scene").value)
        speed = float(self.get_parameter("max_speed_dps").value)
        try:
            self.gate = Gate(Scene(scene_path), speed)
        except inference.InferenceError as exc:
            # The policy speed is read here: a file that does not hold is a
            # gate that does not start, not one that guesses.
            raise SystemExit(f"gate: {exc}") from exc
        self.get_logger().info(
            f"{scene_path.name}: {len(self.gate.scene.joints)} joints, "
            + (f"ramped at {speed:g} deg/s" if speed > 0 else "no ramp")
            + f", {math.degrees(self.gate.policy_rate):g} deg/s under a policy"
            + (f", shaped at {math.degrees(self.gate.accel):g} deg/s^2 and {math.degrees(self.gate.jerk):g} deg/s^3"
               if self.gate.smooth else ", not shaped")
            + (f"; a hand's shaped at {math.degrees(self.gate.hand_accel):g} deg/s^2 and "
               f"{math.degrees(self.gate.hand_jerk):g} deg/s^3" if self.gate.hand_smooth else "; a hand's not shaped")
        )

        self.checked = self.create_publisher(JointState, "joint_commands_checked", 10)
        self.state = self.create_publisher(TeleopState, "teleop_state", 10)
        self.create_subscription(JointState, "joint_commands", self._on_command, 10)
        self.create_subscription(JointState, "joint_states", self._on_state, 10)
        self.create_service(SetTeleopMode, "set_teleop_mode", self._on_mode)
        self.create_service(Estop, "estop", self._on_stop)
        # Published on a timer rather than on change: a client that has not had a
        # frame for a second has lost the gate, and that is only tellable if the
        # gate speaks when nothing is happening.
        hz = max(1.0, float(self.get_parameter("state_hz").value))
        self.create_timer(1.0 / hz, self._publish_state)
        # The shaping's tick: does nothing unless a shaped mode has a command.
        self.create_timer(1.0 / SETTINGS.policy_shaping_hz, self._on_tick)

    # --- subscriptions -----------------------------------------------------
    def _on_state(self, message: JointState) -> None:
        # A reading that is not a number says nothing about where that joint is.
        pose = {k: v for k, v in _pose(message).items() if math.isfinite(v)}
        if pose:
            self.gate.observe(pose)

    def _on_command(self, message: JointState) -> None:
        # Passed whole: a command with a joint that is not a number is
        # refused whole by the check, not sent on without that joint.
        pose = _pose(message)
        if not pose:
            self.get_logger().warn("joint_commands needs matching name and position")
            return
        if self.gate.shaping:
            if not self.gate.aim(pose):
                self.get_logger().warn(f"refused: {self.gate.reason}", throttle_duration_sec=1.0)
            return
        self._send(self.gate.check(pose))

    def _on_tick(self) -> None:
        if self.gate.aiming:
            self._send(self.gate.tick())

    def _send(self, allowed: dict[str, float] | None) -> None:
        if allowed is None:
            if self.gate.reason:
                self.get_logger().warn(f"refused: {self.gate.reason}", throttle_duration_sec=1.0)
            return
        if self.gate.reason:
            self.get_logger().info(self.gate.reason, throttle_duration_sec=1.0)
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = list(allowed)
        out.position = [allowed[n] for n in out.name]
        self.checked.publish(out)

    # --- services ----------------------------------------------------------
    def _on_mode(self, request, response):
        response.success, response.message = self.gate.set_mode(int(request.mode))
        self.get_logger().info(f"mode: {response.message}")
        self._publish_state()
        return response

    def _on_stop(self, _request, response):
        # The request carries nothing: there is one thing to do.
        response.success, response.message = self.gate.stop()
        self.get_logger().warn("emergency stop - nothing passes until idle is asked for")
        self._publish_state()
        return response

    # --- what the page draws ------------------------------------------------
    def _publish_state(self) -> None:
        message = TeleopState()
        message.stamp = self.get_clock().now().to_msg()
        message.armed = self.gate.armed
        message.mode = self.gate.mode
        message.gate_reason = self.gate.reason
        self.state.publish(message)


def _pose(message: JointState) -> dict[str, float]:
    if not message.name or len(message.name) != len(message.position):
        return {}
    return {name: float(value) for name, value in zip(message.name, message.position)}


def main(argv: list[str] | None = None) -> None:
    SETTINGS.pin_ros_middleware()
    rclpy.init(args=argv)
    try:
        node = GateNode()
    except SceneError as exc:
        print(f"the gate has no scene to check against: {exc}")
        rclpy.try_shutdown()
        return
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
