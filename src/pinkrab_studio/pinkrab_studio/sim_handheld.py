"""The handheld driving the simulation.

On the metal the ROS teleop node follows the hand, through the gate. With the
simulation there is no gate and no node, so the studio follows it here the
same way (pinkrab_core.follow.HandFollower): each hand the live view sees
moves its arm as it moves, from where it was first seen, and its jaw's
opening is spent across the arm's jaw band. With no gate, this also keeps
what the gate's mode would say, so the page reads the simulation as it reads
the metal: stopped until asked - nothing drives, the jog locked - 1 the page
jogs the arms, 2 the hands are followed, 3 stopped again.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from pinkrab_core.config import ARMS
from pinkrab_core.config import SETTINGS as CORE
from pinkrab_core.follow import HandFollower
from pinkrab_core.jaw import gripper_band
from pinkrab_core.kinematics import ArmKinematics
from pinkrab_umi.config import SETTINGS as UMI

from .real import MODE_HANDHELD, MODE_IDLE, MODE_STUDIO
from .rig import live_hands


class SimHandheld:
    """What drives the simulated arms, and a thread that follows the hands
    while they do."""

    def __init__(self, physics, stopping: threading.Event) -> None:
        self._physics = physics
        self._stopping = stopping
        self._lock = threading.Lock()
        # What drives the simulated arms, as the gate's mode on the metal.
        self._mode = MODE_IDLE
        self._arms: dict[str, str] = {}
        self._note = ""
        self._wake = threading.Event()
        self._chains: dict[str, ArmKinematics] = {}
        threading.Thread(target=self._run, name="sim-handheld", daemon=True).start()

    def set(self, mode: int) -> dict:
        """MODE_IDLE, MODE_STUDIO or MODE_HANDHELD, the gate's: the arms hold
        where they are, the page jogs them, or they follow the hands."""
        if mode not in (MODE_IDLE, MODE_STUDIO, MODE_HANDHELD):
            raise ValueError(f"no such mode on the simulation: {mode!r}")
        with self._lock:
            self._mode = mode
            self._arms = {}
            self._note = ""
        self._wake.set()
        return self.state()

    def state(self) -> dict:
        with self._lock:
            return {"mode": self._mode, "arms": dict(self._arms), "note": self._note}

    def _say(self, arms: dict[str, str], note: str) -> None:
        with self._lock:
            if self._mode == MODE_HANDHELD:
                self._arms, self._note = arms, note

    def _chain(self, arm: str) -> ArmKinematics:
        if arm not in self._chains:
            self._chains[arm] = ArmKinematics(arm)
        return self._chains[arm]

    def _run(self) -> None:
        period = 1.0 / CORE.ik_rate_hz
        followers: dict[str, HandFollower] = {}
        bands: dict[str, tuple[float, float]] = {}
        while not self._stopping.is_set():
            with self._lock:
                on = self._mode == MODE_HANDHELD
            if not on:
                # Off: the next time on takes every arm's origin again.
                followers = {}
                self._wake.wait(0.5)
                self._wake.clear()
                continue
            began = time.monotonic()
            try:
                self._tick(followers, bands)
            except Exception as exc:  # noqa: BLE001 - said on the page, and tried again next tick
                self._say({}, f"{type(exc).__name__}: {exc}")
            self._stopping.wait(max(0.0, period - (time.monotonic() - began)))

    def _tick(self, followers: dict[str, HandFollower], bands: dict) -> None:
        hands = live_hands()
        if not hands:
            self._say({}, "the live view is not answering - start the top camera")
            return
        qpos = self._physics.snapshot()["qpos"]
        now = time.monotonic()
        wanted: dict[str, float] = {}
        arms: dict[str, str] = {}
        for arm in ARMS:
            # Against the desk solid, as the teleop node follows it: the top
            # camera turns with the operator's head, the desk solid holds still.
            pose = (hands.get("solids") or {}).get(f"handheld_{arm}")
            if pose is None:
                arms[arm] = "not seen with the desk solid"
                continue
            measured = {n: v for n, v in qpos.items() if n.startswith(f"{arm}_")}
            follower = followers.get(arm)
            if follower is None:
                follower = followers[arm] = HandFollower(
                    arm, self._chain(arm), measured, UMI.for_solid(f"handheld_{arm}").solid_T_tcp)
            out, _ = follower.follow(np.asarray(pose, dtype=float), measured, now)
            wanted.update(out)
            arms[arm] = "following"
        # The jaw follows the hand whether or not its solid is seen, as on the metal.
        for arm, opening in (hands.get("jaw") or {}).items():
            if arm not in ARMS or opening is None:
                continue
            if arm not in bands:
                bands[arm] = gripper_band(arm)
            shut, wide = bands[arm]
            wanted[f"{arm}_gripper"] = shut + opening * (wide - shut)
        # Sent only while the hands still drive: a mode changed while this
        # tick was solving - a stop, a pose button placing the arms - is not
        # undone by the tick's last command.
        with self._lock:
            if wanted and self._mode == MODE_HANDHELD:
                self._physics.command(wanted)
        self._say(arms, "")
