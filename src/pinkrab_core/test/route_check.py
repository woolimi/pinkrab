"""Offline check of the walk between home and ready: straight in joint space it
is clear of the robot itself in the scene, both arms at once and one at a time,
both ways - and `walk.Approach` walks to its pose and says when it is there.

    python src/pinkrab_core/test/route_check.py

Needs mujoco and pinkrab_mujoco for the scene; no servo, no ROS.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2] / "pinkrab_mujoco"))

from pinkrab_core.config import SETTINGS  # noqa: E402
from pinkrab_core.origin import ready_pose  # noqa: E402
from pinkrab_core.walk import Approach  # noqa: E402

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def walks_in_scene() -> None:
    from pinkrab_mujoco.scene import Scene

    print("home <-> ready against the scene")
    scene = Scene(SETTINGS.scene_file)
    names = list(scene.joint_names)
    home = {n: 0.0 for n in names}
    ready = {n: ready_pose().get(n, 0.0) for n in names}
    left_ready = {n: (ready[n] if n.startswith("left") else 0.0) for n in names}
    right_ready = {n: (ready[n] if n.startswith("right") else 0.0) for n in names}

    def first_contact(start, goal, samples=200):
        for i in range(0, samples + 1):
            t = i / samples
            hit = scene.touching({n: start[n] + (goal[n] - start[n]) * t for n in names})
            if hit:
                return f"{hit[0]['a']} ~ {hit[0]['b']} at {t:.2f}"
        return None

    import mujoco

    desk = [i for i in range(scene.model.ngeom) if scene.geom_label[i] == "desk"]
    arm_geoms = [i for i in range(scene.model.ngeom)
                 if scene.geom_label[i].startswith(("left_", "right_"))]
    probe = mujoco.MjData(scene.model)

    def desk_clearance(start, goal, samples=80):
        """The nearest any part of an arm comes to the desk on the way, metres."""
        nearest = 9.0
        for i in range(0, samples + 1):
            t = i / samples
            probe.qpos[:] = 0
            for n, adr in scene.joints.items():
                probe.qpos[adr] = start[n] + (goal[n] - start[n]) * t
            mujoco.mj_forward(scene.model, probe)
            for g in arm_geoms:
                for s in desk:
                    nearest = min(nearest, mujoco.mj_geomDistance(scene.model, probe, g, s, 0.5, None))
        return nearest

    # Home is the arm resting on itself - the hand a hair off the shoulder, the
    # upper arm a hair over the base - and still touches nothing.
    walks = {
        "home -> ready": (home, ready),
        "ready -> home": (ready, home),
        "left at ready, right home -> both ready": (left_ready, ready),
        "right alone home -> ready": (home, right_ready),
        "both ready -> right alone home": (ready, left_ready),
    }
    for label, (start, goal) in walks.items():
        hit = first_contact(start, goal)
        check(f"{label}: clear the whole way", hit is None, hit or "")
    ends = min(desk_clearance(home, home, samples=1), desk_clearance(ready, ready, samples=1))
    for label in ("home -> ready", "ready -> home"):
        near = desk_clearance(*walks[label])
        check(f"{label}: never nearer the desk than its two ends are ({ends * 1000:.0f} mm)",
              near >= ends - 1e-3, f"{near * 1000:.0f} mm")


def approach_walks_to_its_pose() -> None:
    print("Approach")
    goal = {"a": math.radians(60.0), "b": math.radians(10.0)}
    walked = Approach(goal, tolerance_deg=1.0, speed_dps=600.0, ease_s=0.0, slow_deg=0.0)
    at = {"a": 0.0, "b": 0.0}
    now = 0.0
    steps = 0
    check("it has the whole way to go at the start", abs(walked.to_go_deg(at) - 60.0) < 1e-6)
    for _ in range(400):
        now += 0.01
        command = walked.step(at, now)
        if walked.arrived:
            break
        # A perfect arm: it is where it was sent.
        at.update(command)
        steps += 1
    check("it is walked there, not sent", steps > 3, f"{steps} steps")
    check("it arrives within its tolerance of the pose", walked.arrived
          and all(abs(at[j] - goal[j]) <= math.radians(1.0) + 1e-9 for j in goal),
          f"a={math.degrees(at['a']):.1f} b={math.degrees(at['b']):.1f}")
    check("nothing more is asked once there", walked.step(at, now + 1) == {} and walked.to_go_deg(at) == 0.0)
    check("it stays arrived when the arm drifts off again", walked.step({"a": 0.0, "b": 0.0}, now + 2) == {})


if __name__ == "__main__":
    walks_in_scene()
    approach_walks_to_its_pose()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} failure(s): {FAILURES}")
        sys.exit(1)
    print("all ok")
