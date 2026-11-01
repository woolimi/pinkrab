"""A pose walked to, not jumped to: one step at a time at a bounded speed.

The arms are taken somewhere with nobody steering more often than it looks -
home when the bringup comes down, the ready pose when the handheld mode
begins or the page asks for it - and each of those wants the same motion: slow,
eased off the mark and eased onto it. Written once here so the
three of them cannot drift into three different walks.

No ROS and no bus in it. Whoever owns the loop calls `step` each tick with the
latest reading and sends what comes back.
"""

from __future__ import annotations

import math

from .config import SETTINGS


class Walk:
    """The command, walked from where the arm was last read toward a goal."""

    def __init__(self, speed_dps: float | None = None, ease_s: float | None = None,
                 slow_deg: float | None = None) -> None:
        self.rate = math.radians(SETTINGS.park_speed_dps if speed_dps is None else speed_dps)
        self.ease = SETTINGS.park_ease_s if ease_s is None else ease_s
        self.slow = math.radians(SETTINGS.park_slow_deg if slow_deg is None else slow_deg)
        self._sent: dict[str, float] = {}
        self._started: float | None = None
        self._at: float | None = None

    def step(self, goal: dict[str, float], measured: dict[str, float], now: float) -> dict[str, float]:
        """The next command: every joint of `goal` the arm has a reading for.

        Eased: a fraction of the speed while starting, and again while the
        furthest joint is inside its last stretch - never below a tenth, or the
        last degrees would take forever. A joint with no reading yet is left
        out rather than guessed at.

        A straight line in joint space: the furthest joint moves at the speed
        and every other joint in proportion, so they all arrive together. Each
        joint stepping at the full speed is a different path - the short ones
        finish first and the arm passes through poses nobody checked. The
        straight line between home and ready is the one route_check walks
        against the scene.
        """
        if self._started is None:
            self._started = now
        dt = 0.0 if self._at is None else min(0.1, max(0.0, now - self._at))
        self._at = now
        out: dict[str, float] = {}
        anchors: dict[str, float] = {}
        for name in goal:
            here = measured.get(name)
            if here is None:
                continue
            anchors[name] = self._sent.get(name, here)
        if not anchors:
            return out
        furthest = max(abs(goal[name] - anchor) for name, anchor in anchors.items())
        scale = min(1.0,
                    (now - self._started) / self.ease if self.ease > 0 else 1.0,
                    max(0.1, furthest / self.slow) if self.slow > 0 else 1.0)
        step = self.rate * scale * dt
        for name, anchor in anchors.items():
            gap = goal[name] - anchor
            share = abs(gap) / furthest if furthest > 0 else 0.0
            sent = anchor + _clamp(gap, step * share)
            self._sent[name] = sent
            out[name] = sent
        return out

    @staticmethod
    def arrived(goal: dict[str, float], measured: dict[str, float], tolerance_deg: float) -> bool:
        """Every joint of `goal` read, and within the tolerance of it."""
        tolerance = math.radians(tolerance_deg)
        return all(name in measured and abs(measured[name] - target) <= tolerance
                   for name, target in goal.items())

    def furthest_deg(self, goal: dict[str, float], measured: dict[str, float]) -> float:
        return max((math.degrees(abs(target - measured[name]))
                    for name, target in goal.items() if name in measured), default=0.0)


def _clamp(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


class Approach:
    """One pose walked to, until the reading says it is reached.

    The tolerance is the caller's for calling the walk arrived: the walk is
    left the moment every joint of the pose is inside it, so the arm does not
    have to settle on the pose before whoever asked moves on.
    """

    def __init__(self, goal: dict[str, float], tolerance_deg: float,
                 speed_dps: float | None = None, ease_s: float | None = None,
                 slow_deg: float | None = None) -> None:
        self.goal = dict(goal)
        self.tolerance_deg = tolerance_deg
        self.walk = Walk(speed_dps, ease_s, slow_deg)
        self.arrived = False

    def step(self, measured: dict[str, float], now: float) -> dict[str, float]:
        """The next command, or nothing once the pose is reached."""
        if not self.arrived and Walk.arrived(self.goal, measured, self.tolerance_deg):
            self.arrived = True
        return {} if self.arrived else self.walk.step(self.goal, measured, now)

    def to_go_deg(self, measured: dict[str, float]) -> float:
        """How far the furthest joint still has to go."""
        return 0.0 if self.arrived else self.walk.furthest_deg(self.goal, measured)
