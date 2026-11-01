"""The studio's side of ROS, in both directions.

The robot lives on the ROS graph now: the driver holds the servo bus, the gate
decides what reaches it. The studio is a browser page with a MuJoCo scene under
it and no `rclpy` anywhere - it runs on a laptop that need not have ROS at all -
so something has to carry between them. That is this node, and it owns nothing.

Three carries, each one way:

    /joint_states, /teleop_state  ->  POST /real/state       where the arms are
    GET /real/requests            ->  service calls          what a button asked
    studio's own target pose      ->  /joint_commands        what to drive toward

The last of those is the one worth reading twice. What the page asks its
simulation for is what gets published, and it crosses the gate like anything
else - joint bands, contact, ramp - before the driver writes a tick. Dragging a
handle in the browser and driving the metal are therefore the same act, which is
the point: there is no second path to the servos with a second set of rules.

Nothing is published until the page asks for something. The studio always has a
target - it opens at the home pose - and publishing that at startup would ask
the arms to go there before anyone had touched anything. So the first target
seen is taken as a baseline, and only a change to it becomes a command.

After that the same command goes out every tick until the arms arrive. The gate
walks a command in at the project's joint speed and no faster, which is a bound
on how fast an arm may move rather than a place to stop: a pose asked for once
and published once would move the arm one tick of the way there and leave it.

`drive_hz` is that tick, and it is the teleop frame rate: how finely the ramp is
sampled, not how fast it goes - the gate spends its speed over the time between
commands whatever that is. `push_hz` is the other direction - readings to the
page - and it is sixty too: at twenty the arm on the page moved in visible
50 ms steps, which read as the arm jumping while a hand dragged it.

One request is a walk rather than a target. The page's pose buttons ask, with
--real, for the arms to be taken to the ready pose or home from wherever they
are, and an arm that may be anywhere is walked at the parking speed rather than
the gate's ceiling. So a `goto` carries its own speed, and the bridge steps its
command toward that pose at that speed from where the arm was last read - the
gate's ramp still applies on top, but never binds. A `goto` may carry more
than one pose - a policy is taken to the ready pose and on to where its takes
began - and the legs are walked in turn, each from where the arm was read at
its start.
A walk is not taken over by the page's target: while one runs, the target is
noted and not driven, and the walk ends only by arriving, by a stop, by the
next walk a pose button asks for, or stopped where it is: an arm that stands
short of the pose for STUCK_S, or readings that stop coming for
reading_stale_s and WALK_BLIND_S more.

    ros2 run pinkrab_bridge bridge
    ros2 run pinkrab_bridge bridge --ros-args -p studio:=http://192.168.1.20:8730
"""

from __future__ import annotations

import collections
import json
import math
import threading
import time
import urllib.error
import urllib.request

import rclpy
from pinkrab_interfaces.msg import TeleopState, TorqueState
from pinkrab_interfaces.srv import Estop, SetTeleopMode, SetTorque
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState

from pinkrab_core.config import SETTINGS
from pinkrab_core.parent import watch_parent

# How long to wait before trying the studio again. Long enough not to hammer a
# machine that is off, short enough that starting the studio second still works.
RETRY_S = 2.0

# A joint has to move by more than this before what the page wants counts as a
# new command. Rounding in the stream is five decimals; this is a hundredth of a
# degree, well above it and far below anything a hand means.
COMMAND_EPS = math.radians(0.01)

# How close the arm has to get before there is nothing left to ask for. Much
# wider than COMMAND_EPS, and it has to be: a servo's encoder is 4095 counts to
# the turn, so a tick is 0.088 degrees and a reading can never sit closer than
# half of one to a commanded angle. Asked for perfection, the bridge would
# republish the same pose forever at an arm already standing in it. Real metal
# is looser still - a position servo settles short of where it was sent, under
# its own weight and in its own deadband, by up to a degree on this arm's step
# responses and 0.6 on a jaw closed on its stop - so this is set above that
# rather than at the encoder's own resolution.
ARRIVED_EPS = math.radians(2.0)

# A leg of a walk whose arm has stopped moving is done even if a joint is
# further off than ARRIVED_EPS (or LEG_EPS on the way) - held by a cable, or
# resting on something - as long as every joint is within SETTLED_NEAR of the
# pose. An arm that stops further off than that for STUCK_S - a command the
# gate refuses, a joint held against something - ends the walk where it is,
# stopped, and the log says which joints were short. Either way the page does
# not say "going to" forever. On a walk's last leg a settled arm does not end
# the walk but starts its settle (WALK_SETTLE_*).
SETTLED_S = 1.0
SETTLED_STILL = math.radians(0.2)
SETTLED_NEAR = math.radians(8.0)
STUCK_S = 3.0

# A walk's last pose is then held as it is - sent whole, no ramp, no lead - for
# up to WALK_SETTLE_S, until every joint reads within WALK_SETTLE_EPS of it, and
# only then is the walk over. The walk's own commands step from readings that
# lag a little under the arm's weight, and a joint dropped from the goal the
# moment it came within ARRIVED_EPS kept the last of them - a degree or two
# short - as its servo's goal: the arms stopped about 2 deg short of home. The
# same settle the driver's park ends with, from the same settings.
WALK_SETTLE_EPS = math.radians(SETTINGS.park_settle_deg)
WALK_SETTLE_S = SETTINGS.park_settle_s

# How close the arm has to get to a pose on the WAY before the next leg starts.
# Wider than ARRIVED_EPS on purpose: a pose passed through is not a pose to be
# held, and a servo settling a degree or two under its own weight would
# otherwise never be found at the waypoint - the walk stalled there until the
# button was pressed again.
LEG_EPS = math.radians(3.0)

# How far a walk's command may run ahead of the arm's last reading. A servo
# needs its setpoint a little in front of it to keep moving; a setpoint that
# walks off from an arm that is stuck is an arm that sprints when freed.
WALK_LEAD = math.radians(10.0)
# The walk is eased - a fraction of its speed over the first park_ease_s, and
# again over the last park_slow_deg of the furthest joint - the same shape as
# the walk pinkrab_core takes the arms home with, from the same settings.
WALK_EASE_S = SETTINGS.park_ease_s
WALK_SLOW = math.radians(SETTINGS.park_slow_deg)

# A walk sends nothing while a joint of it has not been read for
# SETTINGS.reading_stale_s, and ends, stopped, once that has lasted
# WALK_BLIND_S. A bus whose reads fail still takes writes: stepped on from the
# last reading, the command runs WALK_LEAD ahead of an arm that goes there
# unseen - a jump - and stands still in the readings until the walk calls
# itself stuck.
WALK_BLIND_S = 1.0
# A joint the gate has not been asked about for this long is not headed
# anywhere: its last ask - a hand since lost, a jog, a stopped policy - is left
# out of what the page draws as the pose the arm is headed to.
ASKED_FRESH_S = 0.3


class Bridge(Node):
    def __init__(self) -> None:
        super().__init__("pinkrab_bridge")
        self.declare_parameter("studio", "http://127.0.0.1:8730")
        self.declare_parameter("push_hz", 60.0)
        self.declare_parameter("drive_hz", 60.0)
        self.declare_parameter("poll_hz", 5.0)
        # The studio that started this node, by pid: when it is gone, so is
        # this. Started by hand, 0, and nothing is watched.
        self.declare_parameter("parent", 0)
        self.studio = self.get_parameter("studio").value.rstrip("/")
        watch_parent(int(self.get_parameter("parent").value))

        self.commands = self.create_publisher(JointState, "joint_commands", 10)
        self.create_subscription(JointState, "joint_states", self._on_state, 10)
        self.create_subscription(TeleopState, "teleop_state", self._on_teleop, 10)
        # What the gate last let through to each joint, for the page: a
        # correction's jaw is caught at what was sent, not where it reads.
        self.create_subscription(JointState, "joint_commands_checked", self._on_sent, 10)
        # What the gate was last asked for, whoever asked - the handheld loop,
        # this bridge for the page: the pose the arm is headed to, which the
        # page draws see-through while the gate's shaping carries the arm there.
        self.create_subscription(JointState, "joint_commands", self._on_asked, 10)
        # Whether the handheld loop follows the hand (its clutch), as it says.
        self.create_subscription(Bool, "teleop_following", self._on_following,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(
            TorqueState, "torque_state", self._on_torque,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.mode = self.create_client(SetTeleopMode, "set_teleop_mode")
        self.estop = self.create_client(Estop, "estop")
        self.torque = self.create_client(SetTorque, "set_torque")
        # The handheld loop's clutch: held to following, or back.
        self.follow = self.create_client(Trigger, "toggle_follow")

        self._lock = threading.Lock()
        self._angles: dict[str, float] = {}
        self._sent: dict[str, float] = {}
        # What the gate was last asked per joint, and when (monotonic).
        self._asked: dict[str, tuple[float, float]] = {}
        self._following: bool | None = None
        self._teleop: dict | None = None
        self._torque: dict[str, bool] | None = None

        push_hz = max(1.0, float(self.get_parameter("push_hz").value))
        drive_hz = max(1.0, float(self.get_parameter("drive_hz").value))
        poll_hz = max(0.5, float(self.get_parameter("poll_hz").value))
        # The studio's HTTP on threads of its own, one each for pushing and
        # polling: a POST that takes its full timeout must not hold back the
        # readings, the drive loop that judges how fresh they are, or the
        # buttons - sharing one thread, a push always due starves the poll.
        # What a poll fetches is queued for the drive loop, which alone moves
        # the goal and the walk.
        self._requests: collections.deque[dict] = collections.deque()
        self.create_timer(1.0 / push_hz, self._push, callback_group=MutuallyExclusiveCallbackGroup())
        self.create_timer(1.0 / drive_hz, self._drive)
        self.create_timer(1.0 / poll_hz, self._poll, callback_group=MutuallyExclusiveCallbackGroup())

        # What the page last asked its own simulation for, and what of that is
        # still to be reached. The stream is a blocking read, so it gets a
        # thread; everything it does with the node is publishing, which rclpy
        # allows from any thread.
        self._wanted: dict[str, float] = {}
        self._goal: dict[str, float] = {}
        # A walk in progress: the pose being walked to, the speed, and where the
        # walked command has got to per joint. None between walks, and then the
        # goal is published as it is - the gate's ramp is the only one.
        self._walk: dict[str, float] | None = None
        self._legs: list[dict[str, float]] = []
        self._walk_rate = 0.0
        self._walk_sent: dict[str, float] = {}
        self._walk_at: float | None = None
        self._walk_started: float | None = None
        # Joints of the last leg that have come within ARRIVED_EPS - sent their
        # pose whole from then on - and when the whole leg had, for the settle.
        self._landed: set[str] = set()
        self._settle_since: float | None = None
        # When each joint was last read, and since when a walk has had a
        # joint not read recently enough to step it (WALK_BLIND_S).
        self._read_at: dict[str, float] = {}
        self._blind_since: float | None = None
        # A disarm this bridge asked for itself, still on its way back as a
        # TeleopState. The goal was dropped when it was asked; what the page
        # ordered after it - the walk behind a ready button, which asks for
        # idle, then the mode, then the walk - must not be dropped again when
        # the gate's answer arrives late.
        self._own_disarm = False
        # The request id of the walk being walked, reported so the studio knows
        # which of its walks is under way.
        self._walk_id = None
        # When the arm last moved, for a walk that settles short of its pose.
        self._still_at: float | None = None
        self._still_angles: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._follow, daemon=True)
        self._thread.start()
        self.get_logger().info(f"carrying between the graph and {self.studio}")

    # --- ROS -> studio -----------------------------------------------------
    def _on_following(self, message: Bool) -> None:
        with self._lock:
            self._following = bool(message.data)

    def _on_sent(self, message: JointState) -> None:
        if not message.name or len(message.name) != len(message.position):
            return
        with self._lock:
            self._sent.update({name: float(value) for name, value in zip(message.name, message.position)
                               if math.isfinite(value)})

    def _on_asked(self, message: JointState) -> None:
        if not message.name or len(message.name) != len(message.position):
            return
        with self._lock:
            now = time.monotonic()
            self._asked.update({name: (float(value), now) for name, value in zip(message.name, message.position)
                                if math.isfinite(value)})

    def _on_state(self, message: JointState) -> None:
        if not message.name or len(message.name) != len(message.position):
            return
        now = time.monotonic()
        with self._lock:
            fresh = {
                name: float(value)
                for name, value in zip(message.name, message.position)
                if math.isfinite(value)
            }
            self._angles.update(fresh)
            self._read_at.update(dict.fromkeys(fresh, now))

    def _on_teleop(self, message: TeleopState) -> None:
        with self._lock:
            was_armed = bool(self._teleop and self._teleop["armed"])
            self._teleop = {
                "armed": bool(message.armed),
                "mode": int(message.mode),
                "reason": str(message.gate_reason),
            }
            # Disarmed - a stop, an estop, a gate that went down and came
            # back idle - and whatever was being asked for is no longer
            # asked. Kept, it would be refused sixty times a second until
            # the next arming, and then taken at the gate's full speed by an
            # arm nobody had just told to go there.
            if was_armed and not message.armed and not self._own_disarm:
                self._drop_goal()
            if not message.armed:
                self._own_disarm = False

    def _on_torque(self, message: TorqueState) -> None:
        with self._lock:
            self._torque = dict(zip(message.arms, (bool(e) for e in message.enabled)))

    def _drop_goal(self) -> None:
        """Nothing more to drive toward; the next page target is a baseline again."""
        self._goal = {}
        self._walk = None
        self._legs = []
        self._walk_sent = {}
        self._landed = set()
        self._settle_since = None
        self._blind_since = None
        self._wanted = {}

    def _push(self) -> None:
        with self._lock:
            angles, teleop, torque, sent = dict(self._angles), self._teleop, self._torque, dict(self._sent)
            fresh = time.monotonic() - ASKED_FRESH_S
            asked = {name: value for name, (value, at) in self._asked.items() if at >= fresh}
            following = self._following
            walking = self._walk is not None
            walk_id = self._walk_id if walking else None
        if not angles and teleop is None:
            return
        self._post("/real/state", {"angles_rad": angles, "teleop": teleop, "sent_rad": sent, "asked_rad": asked,
                                   "following": following,
                                   "torque": torque, "walking": walking, "walk_id": walk_id})

    # --- studio -> ROS -----------------------------------------------------
    def _poll(self) -> None:
        answer = self._get("/real/requests")
        self._requests.extend((answer or {}).get("requests", []))

    def _ask(self, request: dict) -> None:
        kind = request.get("kind")
        if kind == "mode":
            client, call = self.mode, SetTeleopMode.Request(mode=int(request.get("mode", 0)))
            # A new mode is a new driver; what the old one wanted goes with it.
            with self._lock:
                self._drop_goal()
                if call.mode == TeleopState.MODE_IDLE and self._teleop and self._teleop["armed"]:
                    self._own_disarm = True
        elif kind == "estop":
            client, call = self.estop, Estop.Request()
            with self._lock:
                self._drop_goal()
        elif kind == "torque":
            client, call = self.torque, SetTorque.Request(
                arm=str(request.get("arm", SetTorque.Request.ARM_BOTH)),
                enable=bool(request.get("enable")),
            )
        elif kind == "follow":
            client, call = self.follow, Trigger.Request()
        elif kind == "goto":
            self._start_walk(request)
            return
        else:
            return
        if not client.service_is_ready():
            self.get_logger().warn(f"{kind}: nothing is offering that service")
            return
        # Asked and left. What came of it arrives as the next TeleopState, which
        # is the only thing that knows - a reply held here would be a second
        # answer for the page to believe.
        client.call_async(call)
        self.get_logger().info(f"asked for {kind}")

    def _start_walk(self, request: dict) -> None:
        # A walk is a list of poses, the last of them where the arms end up and
        # any before it passed on the way: the studio says the way and this
        # only walks it, leg by leg.
        poses = request.get("poses")
        if not poses:
            poses = [request.get("pose") or {}]
        legs = [
            {str(k): float(v) for k, v in dict(pose).items() if math.isfinite(float(v))}
            for pose in poses
        ]
        legs = [leg for leg in legs if leg]
        if not legs:
            return
        with self._lock:
            self._legs = legs[1:]
            self._goal = dict(legs[0])
            self._walk = dict(legs[-1])
            self._walk_rate = math.radians(max(1.0, float(request.get("speed_dps", 30.0))))
            self._walk_sent = {}
            self._walk_at = None
            self._walk_started = None
            self._still_angles = {}
            self._landed = set()
            self._settle_since = None
            self._blind_since = None
            self._walk_id = request.get("id")
        self.get_logger().info(
            f"walking {len(legs[-1])} joints to a pose at {math.degrees(self._walk_rate):.0f} deg/s"
            + (f", through {len(legs) - 1} pose(s) on the way" if len(legs) > 1 else "")
        )

    def _follow(self) -> None:
        while not self._stop.is_set():
            try:
                self._read_stream()
            except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
                self.get_logger().warn(
                    f"studio at {self.studio} unreachable ({type(exc).__name__}); "
                    f"retrying in {RETRY_S:.0f}s"
                )
                self._wanted = {}
                self._stop.wait(RETRY_S)

    def _read_stream(self) -> None:
        request = urllib.request.Request(
            f"{self.studio}/physics/stream", headers={"Accept": "text/event-stream"}
        )
        with urllib.request.urlopen(request) as response:
            for line in response:
                if self._stop.is_set():
                    return
                if line.startswith(b"data: "):
                    self._commanded(json.loads(line[6:]))

    def _commanded(self, frame: dict) -> None:
        """Publish what the page has newly asked for, and only that.

        A reconnect starts from whatever the page wants at that moment, taken as
        the baseline rather than as a command: the studio has a target at all
        times, and a bridge coming back up must not read it as an instruction
        issued while it was away.
        """
        wanted = frame.get("ctrl") or {}
        if not wanted:
            return
        if not self._wanted:
            self._wanted = {k: float(v) for k, v in wanted.items()}
            return
        moved = {
            name: float(value) for name, value in wanted.items()
            if abs(float(value) - self._wanted.get(name, float(value))) > COMMAND_EPS
        }
        self._wanted.update({k: float(v) for k, v in wanted.items()})
        if not moved:
            return
        with self._lock:
            if self._walk is not None:
                # A walk is not taken over by the page's target. Changed while
                # a walk runs, the target would be driven straight there at the
                # gate's full speed, past the pose between that the walk goes
                # through - so it is only noted. A walk ends by arriving, by a
                # stop, by another walk (the pose buttons) or stopped short
                # (stuck, or unread), never by this.
                self.get_logger().info(
                    "the page moved its target during a walk - left to the walk",
                    throttle_duration_sec=2.0)
                return
            self._goal.update(moved)

    def _drive(self) -> None:
        """Keep asking for the pose the page wants, until the arms are in it.

        The gate lets a command through as a step toward it rather than as the
        whole distance, so this repeats. A joint drops out of the goal when the
        reading says it has arrived - which is also what stops this republishing
        a pose forever at an arm that has torque off and cannot take it, since
        nothing arrives and nothing is left to ask for once the page's target
        stops changing.

        During a walk the command published is not the goal but a step toward
        it, at the walk's speed, from where the arm was last read - and
        nothing while a joint of it has not been read for reading_stale_s (and
        the walk over WALK_BLIND_S after that). On the walk's last leg nothing
        drops out: a joint within ARRIVED_EPS is sent
        its pose whole from then on, and once every joint is, the pose is held
        for the settle (WALK_SETTLE_*) and the walk ends there.
        """
        while self._requests:
            self._ask(self._requests.popleft())
        now = time.monotonic()
        with self._lock:
            goal, angles = dict(self._goal), dict(self._angles)
            walking = self._walk is not None
            on_the_way = walking and bool(self._legs)
            blind = sorted(k for k in goal if now - self._read_at.get(k, -math.inf) > SETTINGS.reading_stale_s)
            if walking and blind:
                if self._blind_since is None:
                    self._blind_since = now
                blind_for = now - self._blind_since
                ages = [now - self._read_at[k] for k in blind if k in self._read_at]
                # The walk's speed is not spent over the gap. The arm's
                # stillness is: nothing is sent meanwhile, so an arm that reads
                # the same after it stood still, and one that moved resets it.
                self._walk_at = None
            else:
                self._blind_since = None
        if not goal:
            return
        if walking and blind:
            if blind_for < WALK_BLIND_S:
                return
            arms = sorted({k.split("_", 1)[0] for k in blind})
            self.get_logger().warn(
                f"no reading of {len(blind)} joint(s) of the {' and '.join(arms)} arm"
                + (f" for {max(ages):.1f} s" if ages else " yet")
                + " - the walk ends here; is the driver reading the bus?")
            with self._lock:
                self._drop_goal()
            self._ask({"kind": "mode", "mode": TeleopState.MODE_IDLE})
            return
        near = LEG_EPS if on_the_way else ARRIVED_EPS
        finished = False
        arrived = [
            name for name, value in goal.items()
            if name in angles and abs(angles[name] - value) <= near
        ]
        stuck = False
        if walking:
            still = self._still_for(goal, angles)
            near_all = all(k in angles and abs(angles[k] - v) <= SETTLED_NEAR for k, v in goal.items())
            if still >= SETTLED_S and near_all:
                arrived = list(goal)
            elif still >= STUCK_S and self._settle_since is None:   # settling is still on purpose
                stuck = True
        if stuck:
            short = ", ".join(f"{k} {math.degrees(angles[k] - v):+.1f} deg"
                              for k, v in goal.items() if k in angles)
            self.get_logger().warn(f"the walk is stuck and ends here - still to go: {short}")
            with self._lock:
                self._drop_goal()
            self._ask({"kind": "mode", "mode": TeleopState.MODE_IDLE})
            return
        if walking and not on_the_way:
            # The last leg: a joint that has come within ARRIVED_EPS is sent its
            # pose whole from then on, and once every joint has, the pose is held
            # until the readings settle onto it (WALK_SETTLE_*) - then the walk
            # is over. Dropped at ARRIVED_EPS instead, a joint kept the walk's
            # last step, short of the pose, as its servo's goal.
            now = time.monotonic()
            with self._lock:
                self._landed.update(arrived)
                settling = self._settle_since
                if settling is None and self._landed >= set(goal):
                    # Judged from the next reading on: the joint that landed
                    # just now has not yet been sent its pose whole.
                    self._settle_since = now
            if settling is not None:
                on_it = all(k in angles and abs(angles[k] - v) <= WALK_SETTLE_EPS for k, v in goal.items())
                if on_it or now - settling >= WALK_SETTLE_S:
                    with self._lock:
                        self._goal = {}
                        self._walk_sent = {}
                        self._walk = None
                        self._landed = set()
                        self._settle_since = None
                    finished = True
                    self.get_logger().info("the walk is over - the arms are there")
            arrived = []
        if arrived:
            with self._lock:
                for name in arrived:
                    self._goal.pop(name, None)
                    self._walk_sent.pop(name, None)
                if walking and not self._goal and self._legs:
                    # This leg is done; the next starts from where the arm is.
                    self._goal = dict(self._legs.pop(0))
                    self._walk_sent = {}
                    self._walk_at = None
                    self._walk_started = None
                    self._still_angles = {}
                    self.get_logger().info(
                        f"on the way - {len(self._legs) + 1} pose(s) left to walk")
                    goal = dict(self._goal)
                    arrived = []
            goal = {k: v for k, v in goal.items() if k not in arrived}
        if finished:
            # A pose walk ends stopped: the arms hold where they arrived and
            # nothing drives them until the page, a pedal or the handheld asks.
            self._ask({"kind": "mode", "mode": TeleopState.MODE_IDLE})
            return
        if not goal:
            return
        if walking:
            with self._lock:
                landed = set(self._landed)
            stepping = {k: v for k, v in goal.items() if k not in landed}
            command = {**(self._walked(stepping, angles) if stepping else {}),
                       **{k: v for k, v in goal.items() if k in landed}}
        else:
            command = goal
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.name = sorted(command)
        message.position = [command[n] for n in message.name]
        self.commands.publish(message)

    def _still_for(self, goal: dict[str, float], angles: dict[str, float]) -> float:
        """How long the joints of `goal` have stood within SETTLED_STILL, seconds."""
        now = time.monotonic()
        moved = any(abs(angles.get(k, 0.0) - v) > SETTLED_STILL for k, v in self._still_angles.items())
        if moved or set(self._still_angles) != {k for k in goal if k in angles} or self._still_at is None:
            self._still_angles = {k: angles[k] for k in goal if k in angles}
            self._still_at = now
            return 0.0
        return now - self._still_at

    def _walked(self, goal: dict[str, float], angles: dict[str, float]) -> dict[str, float]:
        """One step of the walk toward `goal`, from where the arm was last read."""
        now = time.monotonic()
        if self._walk_started is None:
            self._walk_started = now
        dt = 0.0 if self._walk_at is None else min(0.1, now - self._walk_at)
        self._walk_at = now
        anchors = {}
        for name, target in goal.items():
            here = angles.get(name)
            anchor = self._walk_sent.get(name, here if here is not None else target)
            if here is not None:
                anchor = here + max(-WALK_LEAD, min(WALK_LEAD, anchor - here))
            anchors[name] = anchor
        furthest = max(abs(goal[n] - a) for n, a in anchors.items())
        scale = min(1.0, (now - self._walk_started) / WALK_EASE_S, max(0.1, furthest / WALK_SLOW))
        step = self._walk_rate * scale * dt
        out = {}
        # In proportion, so every joint arrives together: a straight line in
        # joint space is the path that was checked, and joints each stepping at
        # the full speed take a different one.
        for name, anchor in anchors.items():
            gap = goal[name] - anchor
            share = abs(gap) / furthest if furthest > 0 else 0.0
            sent = anchor + max(-step * share, min(step * share, gap))
            self._walk_sent[name] = sent
            out[name] = sent
        return out

    # --- plumbing ----------------------------------------------------------
    def _post(self, path: str, body: dict) -> None:
        request = urllib.request.Request(
            f"{self.studio}{path}", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            urllib.request.urlopen(request, timeout=1.0).close()
        except (urllib.error.URLError, OSError) as exc:
            self.get_logger().warn(
                f"{path} not delivered: {type(exc).__name__}", throttle_duration_sec=5.0
            )

    def _get(self, path: str) -> dict | None:
        try:
            with urllib.request.urlopen(f"{self.studio}{path}", timeout=1.0) as answer:
                return json.load(answer)
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return None

    def destroy_node(self) -> bool:
        self._stop.set()
        return super().destroy_node()


def main(argv: list[str] | None = None) -> None:
    SETTINGS.pin_ros_middleware()
    rclpy.init(args=argv)
    node = Bridge()
    # A thread for each HTTP timer's group, and one for everything else in
    # the node's default group, one callback at a time as before.
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
