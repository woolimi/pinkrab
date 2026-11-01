"""What may pass, and what the operator is told. No ROS in here.

The node in `gate.py` is a few dozen lines of plumbing around this class, and the
plumbing is not where the danger is. Every way this can be wrong is quiet: a ramp
that does not cap, a band read off the wrong model, a contact test that refuses
everything because the arm was already resting on the desk. None of them raise,
and all of them are a person's hand on a slider away from two arms on a wall.

So the policy is importable and testable with no ROS environment at all - see
`test/offline_check.py`, which runs it against the real scene.xml.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

from ruckig import ControlInterface, InputParameter, Result, Ruckig, RuckigError, Synchronization, Trajectory

from pinkrab_core import inference
from pinkrab_core.config import SETTINGS
from pinkrab_mujoco.scene import Scene, pressing

# The same numbers TeleopState carries. Repeated so the policy below can be
# imported and tested without a ROS environment, and checked against the message
# at startup so the two cannot drift apart unnoticed.
MODE_IDLE = 0
MODE_HANDHELD = 1
MODE_STUDIO = 2
MODE_POLICY = 3
MODE_CORRECT = 4
MODE_NAMES = {MODE_IDLE: "idle", MODE_HANDHELD: "handheld", MODE_STUDIO: "studio", MODE_POLICY: "policy",
              MODE_CORRECT: "correct"}
# The modes a hand drives in - the handheld followed, the page's jog and its
# pose walks, a correction - shaped under the teleop limits, the jaws only ramped.
HAND_MODES = (MODE_HANDHELD, MODE_STUDIO, MODE_CORRECT)

# A gap between commands longer than this is a new gesture, not a long tick. The
# ramp spends speed over the time since the last command, and a command arriving
# after a second of silence would otherwise be granted a second's worth of travel
# at once - which is the jump the ramp exists to prevent. Instead it re-anchors:
# the first command after silence sets the ramp at the arm and moves it nowhere,
# and the next one moves it. Sources publish every frame, so nobody notices.
SILENCE_S = 0.25
# Shaped commands that stopped this long ago - a few ticks - have stopped
# asking: the arm is braked to a standstill where it gets to, not carried on
# to the last place it was sent.
SHAPE_SILENCE_S = 0.1
# A shape taken up again after silence starts from where it stood rather
# than from the readings, for a joint read within this of it: a servo at rest
# sits a hair short of its goal, and a setpoint put back at the reading would
# step the arm by that hair.
RESUME_RAD = 0.035
# The most time one tick carries the shape along, in ticks: a timer late by
# more leaves the arm behind its trajectory rather than jumping it ahead.
MAX_TICKS = 3
# A shaped joint slower than this, and turning its speed no faster, stands.
STILL_RAD = 1e-6
# How far a target is moved for Ruckig's second try at an input it fails on.
NUDGE_RAD = 1e-9


class Gate:
    """The policy, with no ROS in it: what may pass, and what the operator is told."""

    def __init__(
        self,
        scene: Scene,
        max_speed_dps: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.scene = scene
        # The ramp, as a speed: it is the same limit whatever rate the commands
        # arrive at, and the same one the simulation walks its actuators along.
        # Zero is no ramp at all - the pose goes through whole, as the config says.
        self.rate = math.radians(
            SETTINGS.max_joint_speed_dps if max_speed_dps is None else max_speed_dps
        )
        self._clock = clock
        # Slower for a policy, whatever the ramp above is - none included. The
        # rollout does not check speed (the takes themselves turn a joint this
        # fast); under a policy this ramp is the only bound on it. The rig's
        # choice of it, and of the shaping, is in inference.yaml, read as the
        # gate starts and each time a policy takes the arms.
        self._read_policy()
        # A hand's commands, shaped the same way under their own limits.
        self.hand_smooth = bool(SETTINGS.teleop_smoothing) and self.rate > 0
        self.hand_accel = math.radians(SETTINGS.teleop_max_joint_accel_dps2)
        self.hand_jerk = math.radians(SETTINGS.teleop_max_joint_jerk_dps3)
        # Where the ramp has got to, per joint, and when it last moved. This is
        # the gate's own setpoint - what it last allowed - and the arm follows it.
        # It is not held to the arm's last reading: the servo's own top speed is
        # about the ramp's, so an arm that falls behind and catches up again does
        # it no faster than the ramp would have let it.
        self._ramp: dict[str, float] = {}
        self._ramp_at: float | None = None
        # Commands, shaped (`shaping`): the latest command per joint
        # (`aim`), and each joint's shaped setpoint, speed and acceleration,
        # moved on every tick (`tick`) - the arm follows the setpoint.
        self._target: dict[str, float] = {}
        # When each joint was last named in a command: a joint whose commands
        # stopped - one hand lost to the camera while the other is followed -
        # is braked on its own (`tick`).
        self._asked_at: dict[str, float] = {}
        self._braking = False
        self._target_notes: tuple[list[str], list[str]] = ([], [])
        self._shape: dict[str, tuple[float, float, float]] = {}
        self._shape_at: float | None = None
        # Where the arms actually are, as last reported. Empty until something
        # says: a gate that guessed would ramp from a pose the arm is not in, and
        # the first command would be exactly the jump the ramp exists to prevent.
        self.at: dict[str, float] = {}
        # When each joint was last read, by the gate's clock: a joint not read
        # for SETTINGS.reading_stale_s is not commanded (see `check`).
        self._read_at: dict[str, float] = {}
        # What the robot is already pressed against - per pair of parts, and
        # how deep. An arm resting on the desk must not have every further
        # command refused for the contact it is already making, so only
        # contacts that are NEW at the target count - and a hand's fingers on
        # the desk, which may press no deeper than fingertip_desk_mm or than
        # they rest (see `observe`). `_now` is the depth as last read, `_off_tip`
        # the finger-on-desk pairs read touching away from the fingertips.
        self._resting: dict[tuple[str, str], float] = {}
        self._now: dict[tuple[str, str], float] = {}
        self._off_tip: set[tuple[str, str]] = set()
        self.mode = MODE_IDLE
        # Set by the stop and cleared only by a deliberate return to idle, so a
        # second press of the same button can never be what re-arms the robot.
        self.stopped = False
        self.reason = "no mode chosen"

    @property
    def armed(self) -> bool:
        return self.mode != MODE_IDLE and not self.stopped

    @property
    def shaping(self) -> bool:
        """Whether commands are aimed and ticked (`aim`, `tick`) rather than checked one by one."""
        if self.mode == MODE_POLICY:
            return self.smooth
        return self.mode in HAND_MODES and self.hand_smooth

    def _limits(self) -> tuple[float, float, float]:
        """The shaping's speed, acceleration and jerk for the mode driving."""
        if self.mode == MODE_POLICY:
            return self.policy_rate, self.accel, self.jerk
        return self.rate, self.hand_accel, self.hand_jerk

    @property
    def aiming(self) -> bool:
        """Whether a shaped command is in hand for `tick` to move the arm toward."""
        return self.shaping and bool(self._target)

    def _read_policy(self) -> None:
        """How a policy's commands go to the arm, from inference.yaml's common
        section: read as the gate starts and each time a policy takes the
        arms, so what the policy page saves holds from the next run."""
        common = inference.common()
        self.policy_rate = math.radians(inference.policy_dps(math.degrees(self.rate), common))
        self.smooth = bool(common["smoothing"])
        self.accel = math.radians(float(common["max_joint_accel_dps2"]))
        self.jerk = math.radians(float(common["max_joint_jerk_dps3"]))

    # --- what the operator asks for ---------------------------------------
    def set_mode(self, mode: int) -> tuple[bool, str]:
        if mode not in MODE_NAMES:
            return False, f"no such mode: {mode}"
        if self.stopped and mode != MODE_IDLE:
            return False, "stopped - go to idle first"
        if mode == MODE_POLICY:
            try:
                self._read_policy()
            except inference.InferenceError as exc:
                return False, f"inference settings: {exc}"
        self.mode = mode
        # A new mode is a new driver; the ramp starts again from the arm.
        self._ramp, self._ramp_at = {}, None
        self._let_go()
        if mode == MODE_IDLE:
            self.stopped = False
            self.reason = "no mode chosen"
        else:
            self.reason = ""
        return True, MODE_NAMES[mode]

    @property
    def arms(self) -> list[str]:
        """The arms the scene has joints of, by the prefix of their names."""
        return sorted({name.split("_", 1)[0] for name in self.scene.joints})

    def stop(self) -> tuple[bool, str]:
        self.mode = MODE_IDLE
        self.stopped = True
        self._ramp, self._ramp_at = {}, None
        self._let_go()
        self.reason = "stopped"
        return True, "stopped"

    # --- what the robot says -----------------------------------------------
    def observe(self, pose: dict[str, float]) -> None:
        """Where the arms are now, and what they are already touching there."""
        self.at.update(pose)
        now = self._clock()
        self._read_at.update(dict.fromkeys(pose, now))
        previous, self._resting, self._off_tip = self._resting, {}, set()
        fingers = set()
        for c in self.scene.touching(self.at):
            key = (c["a"], c["b"])
            self._resting[key] = max(self._resting.get(key, 0.0), c["depth"])
            if c.get("hand"):
                fingers.add(key)
                if not c.get("fingertip"):
                    self._off_tip.add(key)
        self._now = dict(self._resting)
        # Fingers on the desk are held to the shallowest they have been read
        # at since they touched it. A reading is not a command: a servo a tick
        # past its goal, or a link bent by the press, reads a little deeper
        # than the gate let it go, and a depth taken from each reading would
        # let each next command that little further in. Kept per pair of
        # parts, whatever the reading says of the fingertips.
        for key in fingers & previous.keys():
            self._resting[key] = min(self._resting[key], previous[key])

    # --- the gate ----------------------------------------------------------
    def check(self, pose: dict[str, float]) -> dict[str, float] | None:
        """The pose that may be sent, or None. `reason` says what happened either way."""
        accepted = self._accept(pose)
        if accepted is None:
            return None
        wanted, clamped, unknown = accepted

        # How much travel this command is granted: the speed, spent over the time
        # since the last one. A first command, or one after silence, is granted
        # none and only anchors the ramp at the arm.
        now = self._clock()
        if self._ramp_at is None or now - self._ramp_at > SILENCE_S:
            self._ramp = {}
            dt = 0.0
        else:
            dt = max(0.0, now - self._ramp_at)
        self._ramp_at = now
        rate = self.policy_rate if self.mode == MODE_POLICY else self.rate
        step = rate * dt

        stepped: dict[str, float] = {}
        unseen: list[str] = []
        for name, value in wanted.items():
            here = self.at.get(name)
            if here is None:
                unseen.append(name)
                continue
            if rate <= 0:
                stepped[name] = value
                continue
            # Walk the setpoint on from where it was - kept only if this
            # command passes, so a refused one does not carry the ramp ahead
            # of what the arm was sent.
            anchor = self._ramp.get(name, here)
            stepped[name] = anchor + _clamp(value - anchor, step)
        allowed = self._passes(stepped, clamped, unseen, unknown)
        if allowed is not None and rate > 0:
            self._ramp.update(allowed)
        return allowed

    def aim(self, pose: dict[str, float]) -> bool:
        """A command, shaped (`shaping`): held to the same tests as one
        checked, then kept as where each joint is headed - `tick` moves the
        arm toward it. False, with `reason`, for one refused."""
        accepted = self._accept(pose)
        if accepted is None:
            # A refused command is no reason to go on to the last one passed.
            self._braking = True
            return False
        wanted, clamped, unknown = accepted
        now = self._clock()
        if not self._target and self._shape_at is not None and now - self._shape_at <= SILENCE_S:
            # Asked again soon after every joint came to rest: the shape goes
            # on from rest now, not from the last tick before it stopped.
            self._shape_at = now
        self._target.update(wanted)
        self._braking = False
        self._asked_at.update(dict.fromkeys(wanted, now))
        self._target_notes = (clamped, unknown)
        return True

    def tick(self) -> dict[str, float] | None:
        """While shaping, each joint's setpoint moved on toward its command -
        its speed, its acceleration and the change of that under the mode's
        limits (`_limits`), a jerk-limited trajectory (Ruckig) worked
        out afresh each tick from where the setpoint is; in a hand's modes the
        jaws only ramped at the joint speed, a braked jaw standing where it
        is - and the pose held
        to the bands and the scene: the pose to send, or None. A tick after
        silence, or the first, anchors every joint at the arm - or where its
        shape came to rest, for a joint read within RESUME_RAD of it - and
        sends it there. Commands that stopped coming (`SHAPE_SILENCE_S`), or one
        refused for what it asks (`aim`), brake the arm to a standstill. A
        tick refused - a pose the scene stops, no trajectory - holds the arm
        at the last pose sent, and the next command starts from it; a joint
        no longer read lets the shape go, and the next starts from the
        readings once they come back."""
        if not self.aiming:
            return None
        now = self._clock()
        stale = self._stale(self._target, now)
        if stale:
            self.reason = stale
            self._let_go()
            return None
        if self._shape_at is None or now - self._shape_at > SILENCE_S:
            # After silence: from the arm, or from where the shape came to
            # rest for a joint the arm stands at (RESUME_RAD).
            self._shape = {k: (q, 0.0, 0.0) for k, (q, _, _) in self._shape.items()
                           if k in self.at and abs(q - self.at[k]) <= RESUME_RAD}
            dt = 0.0
        else:
            dt = min(max(0.0, now - self._shape_at), MAX_TICKS / SETTINGS.policy_shaping_hz)
        self._shape_at = now

        here: dict[str, tuple[float, float, float]] = {}
        unseen: list[str] = []
        for name in self._target:
            if name not in self.at:
                unseen.append(name)
                continue
            here[name] = self._shape.get(name, (self.at[name], 0.0, 0.0))
        # Braked: every joint, after a refused command; otherwise each joint
        # whose own commands stopped coming.
        braked = {k for k in here if self._braking or now - self._asked_at.get(k, -math.inf) > SHAPE_SILENCE_S}
        # A hand's jaws are ramped, not shaped: a jaw taken over on a block
        # starts from the reading - the block's width - and must be back at
        # its squeeze within a tick or two, not the fifth of a second the
        # shaping takes; and a grasp or a release is the hand's to time.
        jaws = {k for k in here if self.mode in HAND_MODES and k.endswith("_gripper")}
        shaped = here
        if dt > 0 and here:
            shaped = {}
            for k in jaws:
                q = here[k][0]
                if k not in braked:
                    q += _clamp(self._target[k] - q, self.rate * dt)
                shaped[k] = (q, 0.0, 0.0)
            for group, brake in ((braked - jaws, True), (here.keys() - braked - jaws, False)):
                if not group:
                    continue
                step = shape_step({k: here[k] for k in group}, self._target, dt, *self._limits(), brake=brake)
                if step is None:
                    self.reason = "the shaping found no trajectory to the command"
                    self._hold()
                    return None
                shaped.update(step)
        # Held in the bands: a joint the shaping carries past one stops there.
        pose, held = self.scene.clamp({k: q for k, (q, _, _) in shaped.items()})
        shaped = {k: (pose[k], 0.0, 0.0) if k in held else shaped[k] for k in pose}
        clamped, unknown = self._target_notes
        allowed = self._passes(pose, sorted(set(clamped) | set(held)), unseen, unknown)
        if allowed is None:
            self._hold()
            return None
        self._shape.update(shaped)
        # A joint braked to a standstill drops out: nothing more to send it
        # until a command names it again.
        for k in [k for k in braked if k in shaped
                  and abs(shaped[k][1]) < STILL_RAD and abs(shaped[k][2]) < STILL_RAD]:
            self._target.pop(k, None)
        return allowed

    def _stale(self, names, now: float) -> str | None:
        """Why joints of `names` may not be commanded - not read for
        SETTINGS.reading_stale_s - or None."""
        stale = sorted(k for k in names if k in self.at
                       and now - self._read_at.get(k, now) > SETTINGS.reading_stale_s)
        if not stale:
            return None
        age = max(now - self._read_at[k] for k in stale)
        arms = sorted({k.split("_", 1)[0] for k in stale})
        return (f"no reading of {len(stale)} joint(s) of the {' and '.join(arms)} arm"
                f" for {age:.1f} s - is the driver reading the bus?")

    def _hold(self) -> None:
        """The arm left standing at the last pose sent, the command dropped:
        the next one starts from that pose, not from a reading that may still
        be catching up with it - which would step the arm back."""
        self._shape = {k: (q, 0.0, 0.0) for k, (q, _, _) in self._shape.items()}
        self._target, self._target_notes = {}, ([], [])
        self._asked_at, self._braking = {}, False

    def _let_go(self) -> None:
        self._shape, self._shape_at = {}, None
        self._target, self._target_notes = {}, ([], [])
        self._asked_at, self._braking = {}, False

    def _accept(self, pose: dict[str, float]) -> tuple[dict[str, float], list[str], list[str]] | None:
        """A command held to everything but the step and the scene: the joints
        it names within their bands, with those clamped there and the names
        that are not joints - or None, with `reason`."""
        if self.stopped:
            self.reason = "stopped"
            return None
        if not self.armed:
            self.reason = "not armed"
            return None
        if not self.at:
            self.reason = "no joint states yet - is the driver running?"
            return None

        # A joint asked for NaN or infinity is not a place: refused whole. It
        # would pass every comparison below and walk the joint a full step.
        bad = sorted(k for k, v in pose.items() if not math.isfinite(v))
        if bad:
            self.reason = f"not a number for {', '.join(bad)}"
            return None
        wanted, clamped = self.scene.clamp(pose)
        unknown = sorted(set(pose) - set(wanted))
        if not wanted:
            self.reason = f"no joint of the robot is named in this command: {unknown}"
            return None

        # A joint whose readings stopped coming is not where the gate last read
        # it: a bus that fails its reads still takes writes, and an arm sent
        # on from a stale reading moves while every check below is made
        # against where it was. Refused whole, and the ramp let go, so the
        # first command after the readings come back anchors it at the arm
        # however short the gap - not a step of the time since the last pass.
        stale = self._stale(wanted, self._clock())
        if stale:
            self.reason = stale
            self._ramp, self._ramp_at = {}, None
            self._let_go()
            return None
        return wanted, clamped, unknown

    def _passes(self, stepped: dict[str, float], clamped: list[str], unseen: list[str],
                unknown: list[str]) -> dict[str, float] | None:
        """The stepped pose held to the scene: itself, or None - `reason` says why, or what was held."""
        if not stepped:
            self.reason = f"no reading yet for {sorted(unseen)}"
            return None

        # The whole robot at the target, not just the joints being commanded: the
        # other arm is where it is, and whether these two meet is a question about
        # both of them.
        candidate = dict(self.at)
        candidate.update(stepped)
        moving = {name.split("_", 1)[0] for name in stepped}
        new = [c for c in self.scene.touching(candidate) if not self._stands(c, moving)]
        if new:
            first = new[0]
            if first.get("fingertip"):
                self.reason = (f"the {first['fingertip']} fingers would press {first['depth'] * 1000:.1f} mm "
                               f"into the desk, past {SETTINGS.fingertip_desk_mm:g}")
            else:
                self.reason = f"{first['a']} would meet {first['b']}"
            return None

        notes = []
        if clamped:
            notes.append(f"held at the limit: {', '.join(sorted(clamped))}")
        if unseen:
            notes.append(f"no reading yet for {', '.join(sorted(unseen))}")
        if unknown:
            notes.append(f"not joints of this robot: {', '.join(unknown)}")
        self.reason = "; ".join(notes)
        return stepped


    def _stands(self, contact: dict, moving: set[str]) -> bool:
        """Whether a contact at the target may stand: fingertips on the desk
        within the allowance, or a contact the arm already makes.

        A hand's fingers on the desk past the allowance only as they rest, no
        deeper - not a hair: `observe` holds the depth to the shallowest
        reading since they touched, and any slack granted on it would be
        granted again at the next reading. A command that leaves that hand
        where it is (the other arm's, say) meets them as last read. And a part
        of the fingers away from their tips - the camera bracket shares the
        claw's link - may not come onto the desk because the tips already
        rest there."""
        if not pressing(contact):
            return True
        key = (contact["a"], contact["b"])
        rest = self._resting.get(key)
        if rest is None:
            return False
        arm = contact.get("hand")
        if arm is None:
            return True
        if arm not in moving:
            return contact["depth"] <= self._now.get(key, 0.0)
        if not contact.get("fingertip") and key not in self._off_tip:
            return False
        return contact["depth"] <= rest



def _clamp(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def shape_step(state: dict[str, tuple[float, float, float]], target: dict[str, float], dt: float,
               vmax: float, amax: float, jmax: float, brake: bool = False) -> dict[str, tuple[float, float, float]] | None:
    """Each joint's setpoint, speed and acceleration (`state`), `dt` seconds
    along the quickest trajectory to `target` that keeps the speed under
    `vmax`, the acceleration under `amax` and the jerk under `jmax` and stops
    there - or, with `brake`, to a standstill wherever the joint gets to.
    Each joint on its own (Ruckig, no synchronisation); radians and seconds.
    None when no trajectory is found."""
    names = list(state)
    n = len(names)
    given = InputParameter(n)
    given.current_position = [state[k][0] for k in names]
    given.current_velocity = [state[k][1] for k in names]
    given.current_acceleration = [state[k][2] for k in names]
    given.target_position = [target[k] for k in names]
    given.target_velocity = [0.0] * n
    given.target_acceleration = [0.0] * n
    given.max_velocity = [vmax] * n
    given.max_acceleration = [amax] * n
    given.max_jerk = [jmax] * n
    given.synchronization = Synchronization.No
    if brake:
        given.control_interface = ControlInterface.Velocity
    path = Trajectory(n)
    try:
        found = Ruckig(n).calculate(given, path)
    except RuckigError:
        # Ruckig fails on some inputs that sit exactly on the edge between two
        # of its profiles - a 2 deg step from rest at the defaults is one - and
        # finds the trajectory a nanoradian to either side. Asked once more there.
        given.target_position = [p + NUDGE_RAD for p in given.target_position]
        try:
            found = Ruckig(n).calculate(given, path)
        except RuckigError:
            return None
    if found not in (Result.Working, Result.Finished):
        return None
    q, v, a = path.at_time(min(dt, path.duration))
    return {k: (q[i], v[i], a[i]) for i, k in enumerate(names)}
