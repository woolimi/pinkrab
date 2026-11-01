"""The robot as MuJoCo has it, stepped beside the studio server.

There is no separate window. This steps `pinkrab_description/mjcf/scene.xml`
headless so the browser can be the window.

Headless is what makes that possible at all: `mj_step` wants no main thread and no
display, so the simulation runs in a plain thread inside the HTTP server and the
studio can serve a simulated robot from a rig with no screen on it.

The joint names in the MJCF are the joint names in the URDF - the MJCF is generated
from it - so the browser poses its own scene graph straight from `qpos` with no
mapping in between. Fourteen hinges, fourteen position actuators: the whole state
the page needs is fourteen numbers, and the whole command is fourteen more.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from pathlib import Path

from pinkrab_core.config import SETTINGS
from pinkrab_core.origin import ready_pose

from .scene import Scene, SceneError

# How often a snapshot is worth taking. The simulation steps at its own timestep
# (500 Hz for this model); nothing on a screen needs to be told about all of them.
STREAM_HZ = 60

# A command reaches the real servo's motion this long after it is sent. Fitted with
# the model's servo values by scripts/fit_servo.py to step responses recorded on the
# arm; MJCF has no way to state it, so the loop holds each command back instead.
SERVO_DELAY_S = 0.04

PhysicsError = SceneError


class Physics:
    """One MuJoCo world, stepped in real time by a thread of its own.

    Every method that touches `data` takes the lock, including the step itself. A
    snapshot read mid-step would mix joints from two instants - a small error, but
    one that shows up as a limb that flickers, and the contention here is nil: the
    step of a 14 degree-of-freedom model is microseconds and the loop sleeps
    between steps with the lock released.
    """

    def __init__(self, scene: Path) -> None:
        self.world = Scene(scene)
        self._mujoco = self.world.mujoco
        self.scene = scene
        self.model = self.world.model
        self.data = self._mujoco.MjData(self.model)

        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        # A command is a place to end up, not a place to be. Between the two is a
        # ramp: the actuators are walked toward the target at a bounded rate rather
        # than pointed at it outright.
        #
        # This is not decoration on the simulation. The gap between what is asked
        # for and where the arm is can be most of the joint's travel - the moment a
        # target is set, or a slider is thrown across its range - and an arm that
        # closes that gap as fast as its servos allow is an arm swinging at the wall
        # it is bolted to. Bounding it here means the picture moves the way the
        # metal will, and it is the same limiter the gate applies on the way to the
        # servos.
        #
        # The project's one joint speed, in radians a second. Zero means no ramp -
        # the target is taken whole - and not a robot that never moves.
        self.rate = math.radians(SETTINGS.max_joint_speed_dps)
        self._target = self.data.ctrl.copy()
        # What has been sent toward the target so far, and the commands still on
        # their way to the servos (SERVO_DELAY_S of them).
        self._sent = self.data.ctrl.copy()
        self._in_flight: deque = deque()

        # Set when the robot on screen is a picture of metal rather than a
        # simulation of it: the arms are read from the bus and placed here, and
        # integrating between two readings only lets the model sag away from the
        # pose it was just put in and snap back when the next one lands. A tenth of
        # a degree of that reads as a tremor on a still arm.
        self.placed_only = False
        # Whether a placement has ever set the target. The first one must: with
        # the metal driving the picture, the simulation opens at the home pose
        # while the arm is wherever it was left, and a target left at home is a
        # standing request to walk the arm there - which the gate would grant,
        # five degrees at a time, the moment anybody armed. Nobody asked for it.
        # After that first reading the target is the operator's alone; see place().
        self._target_seeded = False

        self._joints = self.world.joints
        self._actuators = self.world.actuators
        self.limits = self.world.limits
        self._geom_label = self.world.geom_label

        # Opened at the ready pose rather than at home, the description's zero.
        # Ready is where the tool is picked up, and a simulation that opens
        # somewhere else is one the operator has to put right before anything it
        # shows means what it looks like.
        start = {name: 0.0 for name in self._joints}
        start.update({n: v for n, v in ready_pose().items() if n in start})
        self.place(start)

    # --- what the page asks for ------------------------------------------
    @property
    def joint_names(self) -> list[str]:
        return list(self._joints)

    def describe(self) -> dict:
        """Everything the page needs once, before the stream starts."""
        return {
            "available": True,
            "scene": self.scene.name,
            "timestep": float(self.model.opt.timestep),
            "rate": STREAM_HZ,
            "joints": self.joint_names,
            "limits": {name: list(limit) for name, limit in self.limits.items()},
            "scenery": self.scenery(),
        }

    def scenery(self) -> dict:
        return self.world.scenery()

    def snapshot(self) -> dict:
        """Where the robot is, what it was asked for, and what it is touching.

        Rounded on the way out. Five decimals is 0.0006 degrees - far below anything
        the page can draw - and rounding is what lets the stream notice that nothing
        has changed instead of sending the last digit of a settled joint at 60 Hz.
        """
        with self._lock:
            qpos = {n: round(float(self.data.qpos[a]), 5) for n, a in self._joints.items()}
            # The target, not the ramped command. What was asked for is a thing
            # somebody chose; watching it crawl toward itself would read as the
            # panel changing the number under you.
            ctrl = {n: round(float(self._target[a]), 5) for n, a in self._actuators.items()}
            contacts = self._contacts()
            t = round(float(self.data.time), 3)
        return {"time": t, "qpos": qpos, "ctrl": ctrl, "contacts": contacts}

    def _contacts(self) -> list[dict]:
        """Deepest first, and only what is actually pressed together.

        Called with the lock held. MuJoCo reports a contact before the surfaces meet
        - the pair is within the solver's margin - and a page that lit up the desk
        for those would be lit up most of the time. Only negative distance is touch.
        """
        found: list[dict] = []
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            depth = -float(contact.dist)
            if depth <= 0:
                continue
            found.append({
                "a": self._geom_label[int(contact.geom1)],
                "b": self._geom_label[int(contact.geom2)],
                "depth": round(depth, 5),
            })
        found.sort(key=lambda c: -c["depth"])
        # Two hands and a desk make plenty of contacts at once, and a page can only
        # say so much. The deepest few are the ones being asked about.
        return found[:8]

    # --- what the page does ----------------------------------------------
    def command(self, pose: dict[str, float]) -> None:
        """Ask the servos for a pose. Physics decides whether they get there.

        This is the honest half of putting MuJoCo under the studio: a command is not
        a placement. The desk, the board, the other arm and a 10 N·m force limit all
        get a say between this call and where the arm ends up - which is exactly
        what happens on the bench, and what a kinematic viewer cannot show.
        """
        with self._lock:
            for name, value in pose.items():
                a = self._actuators.get(name)
                if a is None:
                    continue
                lo, hi = self.limits[name]
                self._target[a] = min(hi, max(lo, float(value)))

    def place(self, pose: dict[str, float]) -> None:
        """Put the robot at a pose outright, without driving it there.

        For poses that come from somewhere else - zero, the homed offsets, the
        servos. Those are facts about the rig rather than requests, and asking the
        simulation to reach one would show the trip instead of the pose, or fail to
        arrive at all if the pose is inside the desk. So the state is set and only
        the kinematics are run.

        Velocity is cleared and the actuators are pointed at the same pose, or the
        robot would set off for wherever it was last asked to go the moment the
        stepping resumes.
        """
        with self._lock:
            for name, value in pose.items():
                adr = self._joints.get(name)
                if adr is None:
                    continue
                lo, hi = self.limits[name]
                clamped = min(hi, max(lo, float(value)))
                self.data.qpos[adr] = clamped
                a = self._actuators.get(name)
                if a is not None:
                    # Both, and no ramp between them: a stated pose is where the
                    # robot IS, so there is no distance for the actuators to cover.
                    self.data.ctrl[a] = clamped
                    self._sent[a] = clamped
                    # While the metal is the one being drawn, a placement arrives
                    # with every reading, and a target reset by each of them would
                    # swallow what the operator just asked for a frame after they
                    # asked for it. The FIRST one still has to land, or the target
                    # stays where this process opened - at home, metres from the
                    # arm - and asks for a journey nobody requested.
                    if not self.placed_only or not self._target_seeded:
                        self._target[a] = clamped
            self.data.qvel[:] = 0
            self._in_flight.clear()
            if self.placed_only:
                self._target_seeded = True
            self._mujoco.mj_forward(self.model, self.data)

    # --- the loop ---------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="physics", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self) -> None:
        step = self.model.opt.timestep
        clock = time.perf_counter()
        ceiling = self.rate * step
        lag = max(0, round(SERVO_DELAY_S / step))
        while not self._stop.is_set():
            if not self.placed_only:
                with self._lock:
                    # Walk the command toward the target, never further than the
                    # ceiling allows in one step's worth of time, and hand the
                    # servos the one sent SERVO_DELAY_S ago.
                    gap = self._target - self._sent
                    self._sent += gap.clip(-ceiling, ceiling) if ceiling > 0 else gap
                    self._in_flight.append(self._sent.copy())
                    while len(self._in_flight) > lag:
                        self.data.ctrl[:] = self._in_flight.popleft()
                    self._mujoco.mj_step(self.model, self.data)
            # Simulated time tracks wall time. Without this the robot moves at
            # whatever speed the machine manages, which is not a robot - and the
            # page would show a rehearsal running at some other tempo than the arm.
            clock += step
            ahead = clock - time.perf_counter()
            if ahead > 0:
                time.sleep(ahead)
            else:
                # Behind: give up the debt rather than sprint to catch it. Catching
                # up means stepping without sleeping, which is how a slow moment
                # turns into a burst of motion nobody asked for.
                clock = time.perf_counter()
