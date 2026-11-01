"""The metal, as the studio hears about it - second hand, and saying so.

The studio does not open the servo bus. One process can hold a serial port, and
that process is the driver on the ROS side; a page that also opened it would be
a second owner of the same wire and a second set of rules about what the arms
may do. So what the studio knows about the real robot arrives here, pushed in by
`pinkrab_bridge`: where the joints are, and what the safety gate believes.

Everything in here carries an age for that reason. A reading is a fact about a
moment, and the moment a bridge stops pushing - its process gone, the network
away, the driver down - the last frame it sent is still on the page looking
exactly like a fresh one. So the page is told how old what it is drawing is, and
stops calling it real when it goes stale.

Requests go the other way for the same reason. The page cannot call a ROS
service, so a button leaves a request here and the bridge picks it up and makes
the call. What comes back is not the button's own optimism but the next state
the gate publishes.
"""

from __future__ import annotations

import threading
import time

# How long a pushed frame stands for before the page stops treating it as the
# robot. Two frames at the driver's own poll rate plus room for a slow hop: long
# enough not to blink on a busy machine, short enough that nobody watches a
# still arm on a dead connection and believes it.
STALE_S = 1.5

# Requests wait for the bridge's next poll, not forever. A mode asked for while
# the bridge was down is not worth acting on when it comes back a minute later -
# whoever pressed it has moved on, and the robot would arm behind them.
# The requests that only make the arms safer (`_stops`) are the exception: one
# is kept until the bridge has been handed it, however long that takes, and is
# handed over again at every poll after that until the robot's own state shows it
# done or STOP_KEEP_S has passed since it was first handed over, so that a response lost on the way does not lose
# the stop. Every request carries a token, which the bridge makes one call of.
REQUEST_TTL_S = 5.0
STOP_KEEP_S = 30.0


# The gate's modes, as TeleopState.msg numbers them. Copied rather than imported:
# this side of the studio has no rclpy and no generated messages.
MODE_IDLE = 0
MODE_HANDHELD = 1
MODE_STUDIO = 2
MODE_POLICY = 3
MODE_CORRECT = 4


class RealRobot:
    """What the bridge last said, and what the page wants asked of ROS."""

    def __init__(self, expected: bool = False) -> None:
        # Whether the page has been switched to the real robot. The difference between
        # "no robot here" and "a robot that has gone quiet" is the whole reason
        # this is a flag rather than an inference from having heard nothing.
        self.expected = expected
        self._lock = threading.Lock()
        self._angles: dict[str, float] = {}
        self._sent: dict[str, float] = {}
        # What the gate is being asked for per joint - where the arm is headed
        # under its shaping - the joints asked lately only, as the bridge last
        # said: replaced whole, so a joint no longer asked drops out.
        self._asked: dict[str, float] = {}
        # Whether the handheld loop follows the hand, as it says; None unheard.
        self._following: bool | None = None
        self._teleop: dict | None = None
        # Which arms hold torque, as the driver last said; None until it has.
        self._torque: dict[str, bool] | None = None
        # A pose walk the bridge is carrying out: the page's pose buttons.
        self._walking = False
        self._trouble: str | None = None
        self._at: float | None = None
        self._requests: list[dict] = []
        self._next_id = 1
        # Tokens are not repeated by a studio started again, whose ids begin at
        # one: the bridge outlives it and remembers the tokens it has taken.
        self._epoch = time.time_ns()
        # The last pedal pressed on the floor, numbered so the page can tell a
        # new press from the one it already flashed.
        self._pedal: dict | None = None
        self._held: dict[str, bool] = {}
        # Called with each pushed reading, outside the lock: the studio stands
        # its simulated arms where the metal is.
        self.listener = None

    # --- from the bridge ---------------------------------------------------
    def push(self, angles: dict[str, float], teleop: dict | None = None,
             trouble: str | None = None, torque: dict | None = None,
             walking: bool = False, sent: dict[str, float] | None = None,
             following: bool | None = None, asked: dict[str, float] | None = None) -> None:
        with self._lock:
            if angles:
                self._angles.update(angles)
            if sent:
                self._sent.update(sent)
            if asked is not None:
                self._asked = dict(asked)
            if following is not None:
                self._following = following
            if teleop is not None:
                self._teleop = teleop
            if torque is not None:
                self._torque = {str(k): bool(v) for k, v in torque.items()}
            self._walking = walking
            self._trouble = trouble
            self._at = time.monotonic()
        listener = self.listener
        if angles and listener is not None:
            listener(dict(angles))

    def take_requests(self) -> list[dict]:
        """Hand over what the page has asked for: once, except for the stops,
        which come again until the robot shows them done."""
        now = time.monotonic()
        with self._lock:
            waiting, kept = [], []
            for r in self._requests:
                if self._stops(r["kind"], r):
                    if r["delivered"] is not None and (now - r["delivered"] > STOP_KEEP_S or self._done(r)):
                        continue
                    if r["delivered"] is None:
                        r["delivered"] = now
                    kept.append(r)
                    waiting.append(r)
                elif now - r["at"] <= REQUEST_TTL_S:
                    waiting.append(r)
            self._requests = kept
        return [{k: v for k, v in r.items() if k not in ("at", "delivered")} for r in waiting]

    def _done(self, request: dict) -> bool:
        """Whether the robot's own state, as last pushed, shows a stop carried
        out. Called with the lock held."""
        teleop, torque = self._teleop, self._torque
        kind = request["kind"]
        if kind == "estop":
            return (teleop is not None and not teleop.get("armed") and teleop.get("reason") == "stopped"
                    and bool(torque) and not any(torque.values()))
        if kind == "mode":
            return teleop is not None and not teleop.get("armed") and teleop.get("reason") != "stopped"
        if torque is None:
            return False
        arm = str(request.get("arm", "both"))
        return not any(torque.values()) if arm == "both" else not torque.get(arm, False)

    # --- from the page -----------------------------------------------------
    @staticmethod
    def _stops(kind: str, fields: dict) -> bool:
        """A request that only ever makes the arms safer: stop, idle, torque off."""
        return (kind == "estop"
                or (kind == "mode" and fields.get("mode") == MODE_IDLE)
                or (kind == "torque" and fields.get("enable") is False))

    def request(self, kind: str, **fields) -> int | None:
        """Queue one request for the bridge. While the page is on the
        simulation only those that stop get through: a pedal pressed then -
        the foot switch outlives a Disconnect - must not reach a bridge that
        happens to be up, or one started in the next few seconds."""
        with self._lock:
            if not self.expected and not self._stops(kind, fields):
                return None
            entry = {"id": self._next_id, "kind": kind, "at": time.monotonic(), "delivered": None,
                     "token": f"{self._epoch}:{self._next_id}", **fields}
            self._next_id += 1
            self._requests.append(entry)
            return entry["id"]

    def angles(self) -> dict[str, float]:
        """The arms as last heard, radians by joint name - empty before anything came."""
        with self._lock:
            return dict(self._angles)

    def sent(self) -> dict[str, float]:
        """What the gate last let through to each joint - the servos' goals -
        radians by joint name, as last heard."""
        with self._lock:
            return dict(self._sent)

    def forget(self) -> None:
        """Back to simulation: nothing heard, and nothing asked but the stops
        not yet taken."""
        with self._lock:
            self._angles = {}
            self._sent = {}
            self._asked = {}
            self._following = None
            self._teleop = None
            self._torque = None
            self._walking = False
            self._trouble = None
            self._at = None
            # The stops stay: one not yet taken still has to reach the bridge.
            self._requests = [r for r in self._requests if self._stops(r["kind"], r)]

    def pedal(self, name: str, down: bool = True) -> None:
        """A foot switch pedal went down (or came up): remembered for the page."""
        with self._lock:
            if down:
                count = 1 if self._pedal is None else self._pedal["count"] + 1
                self._pedal = {"name": name, "count": count, "at": time.monotonic()}
            self._held[name] = down

    def drive(self, mode: int) -> int | None:
        """Make the arms live and hand them to one driver.

        Idle first, because the stop latches: a gate that has been stopped
        refuses every mode until it has been put back to idle, and an operator
        asking for "jog" after a stop means "go", not "explain the latch to
        me". Then torque, then the mode. The page's pedals and the foot switch
        both come through here, so the sequence exists once.
        """
        self.request("mode", mode=MODE_IDLE)
        self.request("torque", arm="both", enable=True)
        return self.request("mode", mode=mode)

    def why_quiet(self, domain: str | None, bridge_pid: int | None) -> str:
        """Why nothing has arrived, in the words of what is actually missing.

        A page that only says "waiting" sends somebody to read three terminals.
        The two things that are wrong in practice are a bridge that is not
        running and a bridge that is running on another ROS domain than the
        robot, and both are knowable from here.
        """
        if bridge_pid is None:
            return ("no bridge is running. Switch the page to real, or run it "
                    "yourself: ros2 run pinkrab_bridge bridge")
        return (f"a bridge is running (pid {bridge_pid}) but nothing has arrived. "
                f"It is on ROS_DOMAIN_ID {domain or '0'}; the robot has to be on "
                "the same one - check with: ros2 node list")

    def snapshot(self) -> dict:
        """What the page draws, with its age attached."""
        with self._lock:
            age = None if self._at is None else round(time.monotonic() - self._at, 3)
            return {
                "expected": self.expected,
                "heard": self._at is not None,
                "fresh": age is not None and age <= STALE_S,
                "age": age,
                "angles_rad": dict(self._angles),
                "asked_rad": dict(self._asked),
                "teleop": self._teleop,
                "torque": self._torque,
                "following": self._following,
                "walking": self._walking,
                "trouble": self._trouble,
                "held": {k: v for k, v in self._held.items() if v},
                "pedal": None if self._pedal is None else {
                    "name": self._pedal["name"], "count": self._pedal["count"],
                    "age": round(time.monotonic() - self._pedal["at"], 3)},
            }
