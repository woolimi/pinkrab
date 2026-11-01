"""The handheld, driving an arm.

The top camera solves the marker solid on the handheld; this turns how far the
hand has carried it into where the tool should be, solves the arm for that, and
publishes `/joint_commands`. What happens next is not this node's business: the
gate holds the pose against the joint bands, the scene and the ramp, and the
driver writes what survives.

Nobody says which arm. The left handheld's solid carries marker ids 0-15 and the
right one 16-31, so the live server solves whichever is in frame and names it,
and picking up the other handheld drives the other arm. Both in frame is both
arms following, each its own solid. There is no flag to get wrong and no flag to
forget, and nothing switches between the hands.

An arm follows its hand RELATIVELY, in the tool's own frame: the first
solved frame is the origin, and from then on the arm's tool moves as the
hand's tool has moved since - forward along where the handheld points, turned
about its axes. The top camera is worn on the operator's head and turns
with it, so the solid's pose is taken against the desk solid, which holds
still: a hand is followed only while the desk solid is in view beside it, the
desk solid may lie anywhere on the desk, and where the hand starts has
nothing to do with where the arm stands. A hand out of sight longer than
`teleop_regrab_s` takes the origin again where it comes back.

The jaw comes with it, always. `pinkrab-umi calibrate` paired the handheld
servo's ticks with the two ends of its travel; the live view reads the servo
beside the cameras and reports the opening in /stats, and the same fraction
drives the arm's gripper across its own range. A handheld that is not plugged
in, or never calibrated, costs the jaw and not the run.

Driving only happens in one mode. The gate publishes `TeleopState`, and unless
that says armed and MODE_HANDHELD this node solves and publishes nothing - so
the node stays up across a session and the switch is a service call rather than
a process being started and killed.

The mode takes the arms where they stand - wherever they were jogged or
walked to - and follows the hands from there: following is relative, so
there is no place the arms have to be first. An arm is taken up once every
joint of it has been read.

Correcting a policy (MODE_CORRECT) differs, all so the hands can take the
arms over in the middle of a task: it begins held, and each jaw holds what
the policy last sent it until its hand's opening reaches that - from either
side, or within `correct_jaw_catch` - so taking over an arm that holds
something does not open it.

Following can be held and resumed (`toggle_follow`, std_srvs/Trigger, each
call the other one - decided here, in the order the calls come): held, the
arms stay where they are and nothing is published; resumed, each origin is
taken again where its hand is then - a clutch, to set the hands down and
pick them up held another way. A correction begins held: each hand is set
the way its arm holds its tool before anything follows it.

    ros2 run pinkrab_teleop teleop
    ros2 run pinkrab_teleop teleop --ros-args -p live:=http://192.168.1.20:8802
"""

from __future__ import annotations

import json
import math
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import rclpy
from pinkrab_interfaces.msg import TeleopState
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState

from pinkrab_core.config import ARMS
from pinkrab_core.config import SETTINGS as CORE
from pinkrab_core.follow import ORIENTATION_WEIGHT, HandFollower
from pinkrab_core.jaw import aperture_of, gripper_band
from pinkrab_core.kinematics import ArmKinematics
from pinkrab_core.parent import watch_parent
from pinkrab_umi.config import SETTINGS as UMI

# Which arm a solid belongs to. The solid's name says it, and the ids under that
# name are what the live server told apart to name it.
ARM_OF_SOLID = {"handheld_left": "left", "handheld_right": "right",
                "dome_left": "left", "dome_right": "right"}


class TeleopNode(Node):
    def __init__(self) -> None:
        super().__init__("pinkrab_teleop")
        self.declare_parameter("live", f"http://127.0.0.1:{UMI.live_port}")
        self.declare_parameter("rate", 30.0)
        self.declare_parameter("orientation_weight", ORIENTATION_WEIGHT)
        # The studio that started this node, by pid: gone, and so is this.
        self.declare_parameter("parent", 0)
        watch_parent(int(self.get_parameter("parent").value))
        self.live = self.get_parameter("live").value.rstrip("/")
        self.orientation_weight = float(self.get_parameter("orientation_weight").value)

        self.commands = self.create_publisher(JointState, "joint_commands", 10)
        self.create_subscription(TeleopState, "teleop_state", self._on_teleop, 10)
        self.create_service(Trigger, "toggle_follow", self._on_follow)
        # Whether the hand is followed, said by this node alone - latched, so
        # the bridge (and through it the page and a correction's readings)
        # has it from the moment it listens, and never a guess of its own.
        self.following_pub = self.create_publisher(
            Bool, "teleop_following", QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        # True outside a correction; False is held - the arms where they are
        # and nothing followed - which a correction begins as (_on_teleop).
        self.following = True
        self._say_following()
        self.create_subscription(JointState, "joint_states", self._on_state, 10)
        # What the gate last let through to each joint - the servo's goal -
        # which a corrected jaw is caught at (see `_catch_jaw`).
        self.create_subscription(JointState, "joint_commands_checked", self._on_sent, 10)
        self.sent: dict[str, float] = {}

        self.at: dict[str, float] = {}
        self.driving = False
        # Whether the arms are being corrected (MODE_CORRECT): followed from
        # where they stand, begun held, each jaw caught (`_catch_jaw`) -
        # where the handheld mode follows at once and the jaws with it.
        self.correcting = False
        # Each corrected arm's jaw: whether its hand has caught it, and which
        # side of it the hand's opening started on.
        self.jaw_caught: dict[str, bool] = {}
        self.jaw_side: dict[str, float] = {}
        # One follower per arm, for as long as the mode lasts; one solver per
        # arm, for as long as the node does.
        self.followers: dict[str, HandFollower] = {}
        self.chains: dict[str, ArmKinematics] = {}
        # Built now, on a thread, rather than when a solid first shows: loading
        # the description takes seconds, which a mode taking the arms over
        # where they stand would otherwise spend still.
        self._chains_lock = threading.Lock()
        threading.Thread(target=self._warm, name="teleop-chains", daemon=True).start()
        # Where the time goes, said every couple of seconds while following:
        # how old the picture behind the pose is, how long a tick took, and how
        # far the arm is behind what it was last asked for. Which of the three
        # is large says which side of the loop is late.
        self._timing: list[tuple[float, float, float]] = []
        self._timing_said = 0.0
        # Each arm's jaw band, read from the description once and kept; the
        # opening itself arrives with the stats, read by the live view.
        self.bands: dict[str, tuple[float, float] | None] = {}
        self._jaw_openings: dict[str, float | None] = {}
        self._quiet = 0

        rate = max(1.0, float(self.get_parameter("rate").value))
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f"following the handheld at {self.live}, {rate:.0f} Hz - "
            "the solid in frame picks the arm, its motion read against the desk solid"
        )

    # --- what the rest of the graph says -----------------------------------
    def _on_state(self, message: JointState) -> None:
        if message.name and len(message.name) == len(message.position):
            self.at.update(dict(zip(message.name, (float(v) for v in message.position))))

    def _on_follow(self, _request, response):
        """Held to following, or following to held. Either way the followers
        go: resumed, each origin is taken again where its hand is then, and
        each jaw is caught again."""
        self.following = not self.following
        self.followers = {}
        self.jaw_caught = {}
        self.jaw_side = {}
        self._say_following()
        self.get_logger().info("following the hands" if self.following else "holding - nothing followed")
        response.success = True
        response.message = "following" if self.following else "holding"
        return response

    def _say_following(self) -> None:
        self.following_pub.publish(Bool(data=self.following))

    def _on_sent(self, message: JointState) -> None:
        if message.name and len(message.name) == len(message.position):
            self.sent.update(dict(zip(message.name, (float(v) for v in message.position))))

    def _on_teleop(self, message: TeleopState) -> None:
        armed = bool(message.armed)
        correcting = armed and message.mode == TeleopState.MODE_CORRECT
        driving = armed and (message.mode == TeleopState.MODE_HANDHELD or correcting)
        if driving == self.driving and correcting == self.correcting:
            return
        self.driving = driving
        self.correcting = correcting
        # The followers go with the mode: coming back, an arm takes its origin
        # again where its hand is, rather than being sent after wherever the
        # hand wandered while nobody was following it.
        self.followers = {}
        self.jaw_caught = {}
        self.jaw_side = {}
        self.following = not correcting
        self._say_following()
        if correcting:
            self.get_logger().info("correcting the arms - held until following is asked for")
        elif driving:
            self.get_logger().info("handheld mode: each arm moves as its hand moves, from where it stands")
        else:
            self.get_logger().info("not driving - the arms hold where they are")

    # --- the loop ------------------------------------------------------------
    def _tick(self) -> None:
        if not self.driving:
            return
        if not self.following:
            self.get_logger().info("holding - nothing followed until following is asked for",
                                   throttle_duration_sec=2.0)
            return
        last = self._stats()
        if last is None:
            return
        tick_started = time.monotonic()
        # The pose to follow, as the live view reports it: each solid against
        # the desk solid, which holds still while the head turns, so the
        # solid's motion in it is the hand's.
        wanted: dict[str, float] = {}
        unfollowed: list[str] = []
        for solid_name, entry in (last.get("solids") or {}).items():
            arm = ARM_OF_SOLID.get(solid_name)
            if arm is None or not entry.get("solved"):
                continue
            pose = entry.get("world_pose")
            if pose is None:
                unfollowed.append(f"{solid_name}: desk solid not in view")
                continue
            follower = self.followers.get(arm)
            if follower is None:
                # The origin is where the arm IS: taken only once every joint
                # of it has been read, or the first command would be solved
                # from a pose the arm is not in.
                chain = self._chain(arm)
                if not all(name in self.at for name in chain.joint_names):
                    unfollowed.append(f"{solid_name}: the {arm} arm is not read yet")
                    continue
                follower = self.followers[arm] = HandFollower(
                    arm, self._chain(arm), self._pose_of(arm),
                    UMI.for_solid(solid_name).solid_T_tcp, self.orientation_weight)
                self.get_logger().info(f"{solid_name} is in frame - the {arm} arm follows it")
            out, began = follower.follow(np.asarray(pose, dtype=float), self._pose_of(arm))
            if began:
                self.get_logger().info(f"origin taken; the {arm} arm now moves as the hand does")
            wanted.update(out)

        # The jaw follows the hand whether or not the solid is in the picture: a
        # hand that turned its marker away is still squeezing, and a gripper
        # that only listened while the arm was being steered would drop the cup
        # in the moment the picture went.
        arms_followed = bool(wanted)
        for arm, opening in self._jaw_openings.items():
            band = self._band(arm)
            if opening is None or band is None:
                continue
            if self.correcting and not self._catch_jaw(arm, opening, band):
                continue
            wanted[f"{arm}_gripper"] = band[0] + opening * (band[1] - band[0])

        if not arms_followed:
            if unfollowed:
                self._say_quietly("; ".join(unfollowed))
            else:
                self._say_quietly(f"no solid ({last.get('reason') or 'not in frame'})")
            if wanted:
                self._publish(wanted)
            return
        self._quiet = 0
        self._publish(wanted)
        self._note_timing(last, tick_started, wanted)

    def _note_timing(self, last: dict, tick_started: float, wanted: dict[str, float]) -> None:
        now = time.monotonic()
        solved_at = last.get("solved_at")
        picture_age = None if solved_at is None else (time.time() - float(solved_at)) * 1000.0
        behind = max((abs(self.at[n] - v) for n, v in wanted.items() if n in self.at), default=0.0)
        self._timing.append((picture_age or 0.0, (now - tick_started) * 1000.0, math.degrees(behind)))
        if now - self._timing_said < 2.0:
            return
        self._timing_said = now
        ages, ticks, gaps = zip(*self._timing)
        self._timing = []
        self.get_logger().info(
            f"timing: picture {sum(ages) / len(ages):.0f} ms old, tick {max(ticks):.0f} ms, "
            f"arm {sum(gaps) / len(gaps):.1f} deg behind the command (worst {max(gaps):.1f})"
        )

    def _publish(self, wanted: dict[str, float]) -> None:
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = sorted(wanted)
        message.position = [wanted[n] for n in message.name]
        self.commands.publish(message)

    # --- the pieces ----------------------------------------------------------
    def _pose_of(self, arm: str) -> dict[str, float]:
        return {name: value for name, value in self.at.items() if name.startswith(f"{arm}_")}

    def _warm(self) -> None:
        """The solvers, built before a hand is first taken up."""
        for arm in ARMS:
            self._chain(arm)

    def _chain(self, arm: str) -> ArmKinematics:
        """The arm's solver, built once and kept for as long as the node runs."""
        with self._chains_lock:
            if arm not in self.chains:
                self.chains[arm] = ArmKinematics(arm)
            return self.chains[arm]

    def _band(self, arm: str) -> tuple[float, float] | None:
        """The band the arm's jaw opening is spent across, read once from the
        description; None when it cannot say. Optional: a description that
        has not been expanded costs the jaw and not the run. The opening itself
        is the live view's to read; a handheld that is not plugged in, or never
        calibrated, shows there as no reading."""
        if arm not in self.bands:
            try:
                self.bands[arm] = gripper_band(arm)
            except (OSError, KeyError, TypeError, ValueError) as exc:
                self.bands[arm] = None    # tried, and not tried again every tick
                self.get_logger().warn(f"no {arm} jaw: {type(exc).__name__} reading the description")
        return self.bands[arm]

    def _catch_jaw(self, arm: str, opening: float, band: tuple[float, float]) -> bool:
        """Whether a corrected arm's jaw follows its hand yet: once the hand's
        opening has reached what the jaw was last sent - crossed it from the
        side it started on, or come within `correct_jaw_catch` - and from then
        on for the rest of the correction. Sent, not read: a jaw on a block
        reads the block's width, and a hand taken there would let go of it."""
        if self.jaw_caught.get(arm):
            return True
        name = f"{arm}_gripper"
        goal = self.sent.get(name, self.at.get(name))
        if goal is None:
            return False
        grip = aperture_of(arm, goal, band)
        side = math.copysign(1.0, opening - grip)
        started = self.jaw_side.setdefault(arm, side)
        if abs(opening - grip) <= CORE.correct_jaw_catch or side != started:
            self.jaw_caught[arm] = True
            self.get_logger().info(f"the hand has the {arm} jaw")
            return True
        self.get_logger().info(f"the {arm} jaw holds at {grip:.2f} until the hand's ({opening:.2f}) reaches it",
                               throttle_duration_sec=2.0)
        return False

    def _stats(self) -> dict | None:
        try:
            with urllib.request.urlopen(f"{self.live}/stats", timeout=1.0) as answer:
                stats = json.load(answer)
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            self._say_quietly(f"{self.live} is not answering ({type(exc).__name__})")
            return None
        # The jaws ride along: the same poll, the same instant.
        self._jaw_openings = {arm: (stats.get("jaw") or {}).get(arm, {}).get("aperture")
                              for arm in ARMS}
        return stats.get("last") or {}

    def _say_quietly(self, line: str) -> None:
        # Once a second at most. A hand turns the solid away for whole seconds at
        # a time and that is normal, not news.
        self._quiet += 1
        self.get_logger().info(line, throttle_duration_sec=1.0)

    def destroy_node(self) -> bool:
        return super().destroy_node()


def main(argv: list[str] | None = None) -> None:
    CORE.pin_ros_middleware()
    rclpy.init(args=argv)
    node = TeleopNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
