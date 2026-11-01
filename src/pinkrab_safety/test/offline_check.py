#!/usr/bin/env python3
"""Verify the gate without ROS, a robot, or anything moving.

What the gate decides is the last thing between a person's hand on a slider and
two arms on a wall, and every way it can be wrong is quiet: a ramp that does not
cap, a limit that is read off the wrong model, a contact test that refuses
everything because the arm was already resting on the desk. None of those raise.
So the policy is written with no ROS in it and exercised here against the real
`scene.xml` - the same file the studio draws and the gate checks against.

    python src/pinkrab_safety/test/offline_check.py
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC / "pinkrab_safety"))
sys.path.insert(0, str(SRC / "pinkrab_mujoco"))
sys.path.insert(0, str(SRC / "pinkrab_core"))

import numpy as np  # noqa: E402

from pinkrab_core import inference  # noqa: E402
from pinkrab_core.config import SETTINGS  # noqa: E402
from pinkrab_core.kinematics import ArmKinematics  # noqa: E402
from pinkrab_core.origin import ready_pose  # noqa: E402
from pinkrab_safety.policy import MODE_CORRECT, MODE_HANDHELD, MODE_IDLE, MODE_NAMES, MODE_POLICY, MODE_STUDIO, Gate  # noqa: E402
from pinkrab_mujoco.scene import Scene  # noqa: E402

FAILURES: list[str] = []

# A clock that stands still, for the checks that are not about time: on the
# wall clock, work done between a reading and a command - IK solved some
# hundreds of times - would age the reading past reading_stale_s and the gate
# would refuse for that.
STILL = lambda: 0.0  # noqa: E731


def rates(trace: list[float], hz: float) -> tuple[list[float], list[float], list[float]]:
    """A joint's speeds, accelerations and jerks, told from its positions a tick apart."""
    speeds = [(b - a) * hz for a, b in itertools.pairwise(trace)]
    accels = [(b - a) * hz for a, b in itertools.pairwise(speeds)]
    return speeds, accels, [(b - a) * hz for a, b in itertools.pairwise(accels)]


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def main() -> int:
    scene = Scene(SETTINGS.scene_file)
    ready = {name: 0.0 for name in scene.joint_names}
    ready.update({n: v for n, v in ready_pose().items() if n in ready})

    print("\nthe scene the gate checks against")
    check("every joint of both arms is in it", len(scene.joint_names) == 14,
          f"{len(scene.joint_names)}")
    check("the ready pose is clear of everything", not scene.touching(ready),
          f"{scene.touching(ready)[:1]}")

    print("\nnothing passes until someone says so")
    gate = Gate(scene, clock=STILL)
    check("a fresh gate is not armed", not gate.armed)
    check("a command is refused before a mode is chosen",
          gate.check({"left_elbow_flex": 1.0}) is None, gate.reason)
    gate.set_mode(MODE_HANDHELD)
    check("armed once a mode is asked for", gate.armed)
    check("still refused with no reading of where the arms are",
          gate.check({"left_elbow_flex": 1.0}) is None, gate.reason)
    check("and it says why", "joint states" in gate.reason, gate.reason)

    print("\nthe ramp")
    # The gate is handed a clock so the test can say how much time a command is
    # granted, and so ten commands in no time at all are ten commands in no time.
    tick = 1.0 / SETTINGS.teleop_fps
    per_tick = SETTINGS.max_joint_speed_dps * tick
    clock = [0.0]
    gate = Gate(scene, clock=lambda: clock[0])
    gate.set_mode(MODE_HANDHELD)
    gate.observe(ready)
    far = ready["left_elbow_flex"] + math.radians(60)
    allowed = gate.check({"left_elbow_flex": far})
    check("a command 60 degrees away is not refused, it is walked toward",
          allowed is not None, gate.reason)
    check("the first command anchors the ramp at the arm and moves it nowhere",
          allowed is not None and abs(allowed["left_elbow_flex"] - ready["left_elbow_flex"]) < 1e-9,
          "a first command granted a jump is the jump the ramp exists to prevent")
    clock[0] += tick
    allowed = gate.check({"left_elbow_flex": far})
    step = math.degrees(allowed["left_elbow_flex"] - ready["left_elbow_flex"]) if allowed else -1
    check(f"one tick later it has moved one tick's worth ({per_tick:g} deg), not the gap",
          abs(step - per_tick) < 1e-6, f"{step:.3f} deg")
    clock[0] += tick
    near = allowed["left_elbow_flex"] + math.radians(1)
    allowed = gate.check({"left_elbow_flex": near})
    check("a small command arrives whole",
          allowed is not None and abs(allowed["left_elbow_flex"] - near) < 1e-9)

    # The speed is spent over time, not per command. Commands arriving faster
    # than the clock moves are granted nothing extra.
    clock[0] += tick
    before = gate.check({"left_elbow_flex": far})["left_elbow_flex"]
    for _ in range(10):
        after = gate.check({"left_elbow_flex": far})["left_elbow_flex"]
    check("ten commands in no time move the arm no further than one",
          abs(after - before) < 1e-9, f"{math.degrees(after - before):.3f} deg")

    # With the reading standing still - a servo behind the ramp, or an arm held,
    # read every tick at the same angle - the setpoint goes on at the speed
    # and stops at the target, no faster.
    start = gate.check({"left_elbow_flex": far})["left_elbow_flex"]
    for _ in range(5):
        clock[0] += tick
        gate.observe(ready)
        stepped = gate.check({"left_elbow_flex": far})
    walked = math.degrees(stepped["left_elbow_flex"] - start)
    check(f"with the reading standing still the setpoint still walks at the speed ({5 * per_tick:g} deg in five ticks)",
          abs(walked - 5 * per_tick) < 1e-6, f"{walked:.3f} deg")
    for _ in range(60):
        clock[0] += tick
        gate.observe(ready)
        stepped = gate.check({"left_elbow_flex": far})
    end, _ = gate.scene.clamp({"left_elbow_flex": far})
    check("and stops at the target, held at the joint's limit",
          abs(stepped["left_elbow_flex"] - end["left_elbow_flex"]) < 1e-9,
          f"{math.degrees(stepped['left_elbow_flex']):.2f} vs {math.degrees(end['left_elbow_flex']):.2f}")

    # Silence re-anchors. A command a second after the last - the arm read
    # all along - is not granted a second's worth of travel.
    clock[0] += 1.0
    gate.observe(ready)
    stepped = gate.check({"left_elbow_flex": far})
    ahead = math.degrees(stepped["left_elbow_flex"] - ready["left_elbow_flex"])
    check("a command after silence starts again from the arm",
          abs(ahead) < 1e-9, f"{ahead:.3f} deg ahead")

    # With the arm following, every tick is bounded the same way and the arm
    # gets there. Thirty degrees, inside the elbow's band from ready, so the
    # gate does not clamp it at a stop - which would be a test of the limit,
    # not of the ramp.
    walk_to = ready["left_elbow_flex"] + math.radians(30)
    walking = Gate(scene, clock=lambda: clock[0])
    walking.set_mode(MODE_HANDHELD)
    at = dict(ready)
    walking.observe(at)
    biggest = 0.0
    steps = 0
    while steps < 60 and abs(at["left_elbow_flex"] - walk_to) > 1e-9:
        stepped = walking.check({"left_elbow_flex": walk_to})
        if stepped is None:
            break
        biggest = max(biggest, abs(math.degrees(stepped["left_elbow_flex"] - at["left_elbow_flex"])))
        at.update(stepped)
        walking.observe(at)
        clock[0] += tick
        steps += 1
    check("no tick is larger when the readings follow along",
          biggest <= per_tick + 1e-6, f"largest {biggest:.3f} deg")
    # 30 degrees at 5 a tick is six of them after the anchoring one, and the ramp
    # is what makes it seven rather than one.
    check("and the arm gets all the way there, a tick at a time",
          abs(at["left_elbow_flex"] - walk_to) < 1e-9 and steps == 7,
          f"{math.degrees(at['left_elbow_flex'] - ready['left_elbow_flex']):.2f} deg in {steps} ticks")

    # Within the joint's band from ready, so what comes back is the ramp's doing
    # alone, not a stop's.
    unramped = Gate(scene, max_speed_dps=0.0, clock=STILL)
    unramped.set_mode(MODE_STUDIO)
    unramped.observe(ready)
    swing = ready["left_elbow_flex"] + math.radians(45)
    allowed = unramped.check({"left_elbow_flex": swing})
    check("max_speed_dps 0 is no ramp: a 45 degree command arrives whole",
          allowed is not None and abs(allowed["left_elbow_flex"] - swing) < 1e-9,
          "not an arm held where it is by a step of nothing")

    # A policy is ramped at its own, slower speed - with the ramp the gate was
    # given, and with none at all.
    policy_dps = inference.common()["max_joint_dps"]
    policy_tick = math.radians(policy_dps) * tick
    for label, speed in (("the ramp", None), ("no ramp", 0.0)):
        clock[0] = 0.0
        driven = Gate(scene, max_speed_dps=speed, clock=lambda: clock[0])
        driven.set_mode(MODE_POLICY)
        driven.smooth = False     # sent on as they come: the ramp alone
        driven.observe(ready)
        driven.check({"left_elbow_flex": swing})
        clock[0] += tick
        allowed = driven.check({"left_elbow_flex": swing})
        moved = None if allowed is None else abs(allowed["left_elbow_flex"] - ready["left_elbow_flex"])
        check(f"under a policy, not shaped, with {label}, a 45 degree command moves "
              f"{policy_dps:g} deg/s's worth a tick",
              moved is not None and abs(moved - policy_tick) < 1e-9,
              f"{math.degrees(moved or 0):.2f} deg against {math.degrees(policy_tick):.2f}")

    # A joint asked for NaN is not a place: refused whole, not walked a step.
    clock[0] = 0.0
    driven = Gate(scene, clock=lambda: clock[0])
    driven.set_mode(MODE_POLICY)
    driven.observe(ready)
    driven.check({"left_elbow_flex": swing})
    clock[0] += tick
    check("a joint asked for NaN is refused whole",
          driven.check({"left_elbow_flex": math.nan, "right_elbow_flex": swing}) is None, driven.reason)

    print("\nreadings that stopped coming")
    # A bus whose reads fail still takes writes: an arm commanded from its last
    # reading moves while the gate checks against where it was.
    clock[0] = 0.0
    blind = Gate(scene, clock=lambda: clock[0])
    blind.set_mode(MODE_STUDIO)
    blind.observe(ready)
    blind.check({"left_elbow_flex": swing})
    clock[0] += tick
    check("read a tick ago, a joint is commanded", blind.check({"left_elbow_flex": swing}) is not None, blind.reason)
    clock[0] += SETTINGS.reading_stale_s + tick
    check(f"not read for more than {SETTINGS.reading_stale_s:g} s, it is refused",
          blind.check({"left_elbow_flex": swing}) is None, blind.reason)
    check("and it says so", "no reading of 1 joint(s) of the left arm" in blind.reason, blind.reason)
    blind.observe({k: v for k, v in ready.items() if k.startswith("right_")})
    check("a joint of the other arm, still read, is refused with it - the command is refused whole",
          blind.check({"left_elbow_flex": swing, "right_elbow_flex": ready["right_elbow_flex"]}) is None
          and "right" not in blind.reason, blind.reason)
    clock[0] += tick
    blind.observe(ready)
    allowed = blind.check({"left_elbow_flex": swing})
    check("read again, it is commanded again - from the arm, not from where the ramp had got to",
          allowed is not None and abs(allowed["left_elbow_flex"] - ready["left_elbow_flex"]) < 1e-9,
          blind.reason)
    # A gap only just past the threshold - shorter than SILENCE_S counted from
    # the last command that passed - anchors at the arm all the same.
    clock[0] = 0.0
    brief = Gate(scene, clock=lambda: clock[0])
    brief.set_mode(MODE_STUDIO)
    brief.observe(ready)
    for _ in range(3):
        brief.check({"left_elbow_flex": swing})
        clock[0] += tick
        brief.observe(ready)
    clock[0] += SETTINGS.reading_stale_s - tick / 2
    brief.check({"left_elbow_flex": swing})      # the last to pass, just before the reading goes stale
    clock[0] += tick
    refused = brief.check({"left_elbow_flex": swing}) is None
    brief.observe(ready)
    allowed = brief.check({"left_elbow_flex": swing})
    check("after a brief gap the first command is the arm itself, not a step of the gap",
          refused and allowed is not None
          and abs(allowed["left_elbow_flex"] - ready["left_elbow_flex"]) < 1e-9,
          f"refused {refused}, {brief.reason}")

    print("\nthe joint bands")
    # The stop at the bottom of wrist_flex's band, which the wrist reaches
    # from ready without meeting anything: the band alone answers.
    far = scene.limits["left_wrist_flex"][0]
    wide = Gate(scene, max_speed_dps=0.0, clock=STILL)
    wide.set_mode(MODE_HANDHELD)
    wide.observe(ready)
    allowed = wide.check({"left_wrist_flex": far + math.copysign(math.radians(30), far)})
    check("a pose past a stop is held at the stop",
          allowed is not None and abs(allowed["left_wrist_flex"] - far) < 1e-9,
          f"{allowed}")
    check("and the operator is told", "limit" in wide.reason, wide.reason)
    check("a joint of no robot is ignored rather than passed on",
          "wrist_of_nothing" not in (wide.check({"wrist_of_nothing": 0.0}) or {}))

    print("\nthe scene")
    # Measured: from ready, the left arm swung inward, across the middle and
    # round to the end of shoulder_pan's band - on the left arm its positive
    # end - lays its hand into the board. Contact, not a joint limit - the
    # band allows it.
    into_board = scene.limits["left_shoulder_pan"][1]
    wide.observe(ready)
    check("a pose inside the board is refused",
          wide.check({"left_shoulder_pan": into_board}) is None, wide.reason)

    # Walked toward the board until it refuses, and asked a few times more:
    # the ramp stays at what was last sent, so asking for that pose again
    # gets exactly it - not a setpoint the refused ticks carried further on.
    clock[0] = 0.0
    driven = Gate(scene, clock=lambda: clock[0])
    driven.set_mode(MODE_POLICY)
    driven.observe(ready)
    sent = None
    refused = 0
    for _ in range(400):
        allowed = driven.check({"left_shoulder_pan": into_board})
        clock[0] += tick
        driven.observe(ready)
        if allowed is None:
            refused += 1
            if refused == 5:
                break
        else:
            sent = allowed["left_shoulder_pan"]
    again = driven.check({"left_shoulder_pan": sent}) if sent is not None else None
    check("after refusals, the last pose sent is granted as it was",
          refused == 5 and again is not None and abs(again["left_shoulder_pan"] - sent) < 1e-9,
          f"{refused} refused, {driven.reason or 'granted'}")
    check("named as the two things that would meet",
          " would meet " in wide.reason and "pegboard" in wide.reason and "left_" in wide.reason, wide.reason)

    resting = dict(ready)
    resting["left_shoulder_pan"] = into_board
    wide.observe(resting)
    # The jaw opening: nothing of it touches, so only the contact the arm
    # already makes could refuse it.
    check("an arm already touching is not frozen by the contact it already makes",
          wide.check({"left_gripper": ready["left_gripper"] - 0.05}) is not None,
          wide.reason)

    print("\nthe fingertips on the desk")
    # The right tool held 45 degrees down and turned 32 degrees in, 66 mm from
    # the middle - stated by the joints that hold it there - walked straight
    # down from 70 mm above the desk (the description's z = 0) half a
    # millimetre at a time, until its fingers are well into the desk. Solved
    # from scratch at the top: the shape it is stated by cannot carry that
    # attitude so low.
    chain = ArmKinematics("right")
    held = [-25.0, 66.2, -56.6, 11.7, 35.8, 11.8]
    top = chain.forward(held)
    q = None
    steps = []
    for height_mm in np.arange(70.0, -20.0, -0.5):
        target = top.copy()
        target[2, 3] = height_mm / 1000.0
        if q is None:
            q = chain.solve(target, seed=held, orientation_weight=1.0)
        for _ in range(40):
            q = chain.track(target, q, orientation_weight=1.0)
        pose = dict(ready)
        pose.update({n: math.radians(float(v)) for n, v in zip(chain.joint_names, q)})
        contacts = scene.touching(pose)
        steps.append((pose, contacts))
        if contacts and max(c["depth"] for c in contacts) > 0.008:
            break
    allow = SETTINGS.fingertip_desk_mm / 1000.0

    def deepest(contacts):
        return max((c["depth"] for c in contacts), default=0.0)

    clear = next((pose for pose, c in reversed(steps) if not c), None)
    shallow = next((pose for pose, c in steps
                    if c and all(x.get("fingertip") == "right" for x in c) and deepest(c) < allow * 0.8), None)
    deep = next((pose for pose, c in steps if deepest(c) > allow * 1.5), None)
    first = next((c for _, c in steps if c), [])
    check("the first thing the lowered tool meets is the desk, at its fingertips",
          bool(first) and all(c.get("fingertip") == "right" for c in first), f"{first[:2]}")
    check("found a pose clear of the desk, one just on it and one pressed in",
          None not in (clear, shallow, deep))
    if None not in (clear, shallow, deep):
        flat = Gate(scene, max_speed_dps=0.0, clock=STILL)
        flat.set_mode(MODE_HANDHELD)
        flat.observe(clear)
        check(f"fingertips onto the desk, within {SETTINGS.fingertip_desk_mm:g} mm, pass",
              flat.check(shallow) is not None, flat.reason)
        check("fingertips pressed past it are refused",
              flat.check(deep) is None and "fingers would press" in flat.reason, flat.reason)
        flat.observe(shallow)
        check("resting on the desk does not let them press further in",
              flat.check(deep) is None, flat.reason)
        check("and they can lift off again", flat.check(clear) is not None, flat.reason)

        # The same fingers read just past the tips' reach, so their contact
        # rests as not the fingertips': it may not press further in either.
        import pinkrab_mujoco.scene as scene_module
        tips_reach = scene_module.SETTINGS
        scene_module.SETTINGS = dataclasses.replace(tips_reach, fingertip_reach_mm=0.0)
        try:
            beyond = Gate(scene, max_speed_dps=0.0, clock=STILL)
            beyond.set_mode(MODE_HANDHELD)
            beyond.observe(shallow)
            check("fingers resting on the desk past the tips' reach do not press further in either",
                  beyond.check(deep) is None, beyond.reason)
            check("and may stay where they rest", beyond.check(shallow) is not None, beyond.reason)
        finally:
            scene_module.SETTINGS = tips_reach

        # Walked down a third of a millimetre at a time, the gate reading the
        # arm at every pose it lets through: resting at each step must not let
        # the next one in a little further.
        walked = Gate(scene, max_speed_dps=0.0, clock=STILL)
        walked.set_mode(MODE_HANDHELD)
        walked.observe(clear)
        q = np.array([math.degrees(clear[n]) for n in chain.joint_names])
        start = chain.forward(q)
        let_in = 0.0
        for drop_mm in np.arange(0.0, 12.0, 0.3):
            target = start.copy()
            target[2, 3] -= drop_mm / 1000.0
            for _ in range(20):
                q = chain.track(target, q, orientation_weight=1.0)
            pose = dict(clear)
            pose.update({n: math.radians(float(v)) for n, v in zip(chain.joint_names, q)})
            if walked.check(pose) is not None:
                walked.observe(pose)
                let_in = max(let_in, max((c["depth"] for c in scene.touching(pose)
                                          if c.get("fingertip")), default=0.0))
        check(f"walked down a step at a time, never let past {SETTINGS.fingertip_desk_mm:g} mm",
              0.0 < let_in <= allow, f"{let_in * 1000:.2f} mm")

        # The same walk with every reading a step deeper than the command the
        # gate let through, as a servo past its goal or a bent link reads: the
        # readings must not carry the commands in with them.
        sagging = Gate(scene, max_speed_dps=0.0, clock=STILL)
        sagging.set_mode(MODE_HANDHELD)
        sagging.observe(clear)
        q = np.array([math.degrees(clear[n]) for n in chain.joint_names])
        poses = []
        for drop_mm in np.arange(0.0, 12.0, 0.3):
            target = start.copy()
            target[2, 3] -= drop_mm / 1000.0
            for _ in range(20):
                q = chain.track(target, q, orientation_weight=1.0)
            pose = dict(clear)
            pose.update({n: math.radians(float(v)) for n, v in zip(chain.joint_names, q)})
            poses.append(pose)
        let_in = 0.0
        for i, pose in enumerate(poses[:-1]):
            if sagging.check(pose) is not None:
                sagging.observe(poses[i + 1])
                let_in = max(let_in, max((c["depth"] for c in scene.touching(pose)
                                          if c.get("fingertip")), default=0.0))
        check(f"readings a step deeper than each command do not carry it past {SETTINGS.fingertip_desk_mm:g} mm",
              0.0 < let_in <= allow, f"{let_in * 1000:.2f} mm")

        # Again, with every other reading taking the contact as off the tips:
        # what the reading says of the fingertips must not reset the depth.
        flipping = Gate(scene, max_speed_dps=0.0, clock=STILL)
        flipping.set_mode(MODE_HANDHELD)
        flipping.observe(clear)
        let_in = 0.0
        for i, pose in enumerate(poses[:-1]):
            if flipping.check(pose) is not None:
                scene_module.SETTINGS = (dataclasses.replace(tips_reach, fingertip_reach_mm=0.0)
                                         if i % 2 else tips_reach)
                try:
                    flipping.observe(poses[i + 1])
                finally:
                    scene_module.SETTINGS = tips_reach
                let_in = max(let_in, max((c["depth"] for c in scene.touching(pose)
                                          if c.get("fingertip")), default=0.0))
        check("nor when the reading takes them as the fingertips one time and not the next",
              0.0 < let_in <= allow, f"{let_in * 1000:.2f} mm")

        # A part of the fingers away from the tips - the camera bracket is on
        # the claw's link - coming onto the desk is refused, tips resting or
        # not: asked at the depth the tips rest at, so only that rule refuses.
        tips = Gate(scene, max_speed_dps=0.0, clock=STILL)
        tips.set_mode(MODE_HANDHELD)
        tips.observe(shallow)
        scene_module.SETTINGS = dataclasses.replace(tips_reach, fingertip_reach_mm=0.0)
        try:
            check("the fingers' link away from the tips may not join them on the desk",
                  tips.check(shallow) is None, tips.reason)
        finally:
            scene_module.SETTINGS = tips_reach

        # Fingers read deeper than the shallowest they rested at - the reading
        # wanders - do not freeze the other arm, which a command leaves them as read.
        found = Gate(scene, max_speed_dps=0.0, clock=STILL)
        found.set_mode(MODE_HANDHELD)
        for read in (shallow, deep):
            found.observe(read)
        check("fingers read deeper than they rested do not stop a command to the other arm",
              found.check({"left_elbow_flex": ready["left_elbow_flex"] + 0.05}) is not None, found.reason)
        check("while their own arm may not take them there",
              found.check(deep) is None, found.reason)

    print("\nboth arms, corrected by the handhelds")
    both = Gate(scene, clock=lambda: clock[0])
    both.observe(ready)
    ok, said = both.set_mode(MODE_CORRECT)
    check("correcting takes the arms", ok and both.armed, said)
    check("a command for both arms passes",
          both.check({"right_elbow_flex": ready["right_elbow_flex"],
                      "left_elbow_flex": ready["left_elbow_flex"]}) is not None, both.reason)

    print("\na policy's commands, shaped")
    hz = SETTINGS.policy_shaping_hz
    slack = 1.0 + 1e-6

    def shaped_gate(rate_dps=None, accel_dps2=None, jerk_dps3=None) -> Gate:
        clock[0] = 0.0
        g = Gate(scene, clock=lambda: clock[0])
        g.set_mode(MODE_POLICY)
        g.smooth = True
        if rate_dps is not None:
            g.policy_rate, g.accel, g.jerk = map(math.radians, (rate_dps, accel_dps2, jerk_dps3))
        g.observe(ready)
        return g

    def drive(g: Gate, joint: str, target, seconds: float, every: int = 2) -> list[float]:
        """Commands at hz / every (target(t), None for none), the gate ticked at
        hz, the arm going where it is sent: the joint's positions sent."""
        sent = []
        for i in range(int(seconds * hz)):
            t = i / hz
            clock[0] = t
            if i % every == 0 and (goal := target(t)) is not None:
                g.aim({joint: goal})
            g.observe({**ready, joint: sent[-1] if sent else ready[joint]})
            allowed = g.tick()
            if allowed is not None:
                sent.append(allowed[joint])
        return sent

    def within(g: Gate, trace: list[float]) -> tuple[bool, str]:
        speeds, accels, jerks = rates(trace, hz)
        top = [max(map(abs, x), default=0.0) for x in (speeds, accels, jerks)]
        ok = top[0] <= g.policy_rate * slack and top[1] <= g.accel * slack and top[2] <= g.jerk * slack
        return ok, f"{math.degrees(top[0]):.1f} deg/s, {math.degrees(top[1]):.0f} deg/s^2, {math.degrees(top[2]):.0f} deg/s^3"

    g = shaped_gate()
    check("a policy is shaped by default", g.shaping and inference.common()["smoothing"] is True)
    base = ready["left_elbow_flex"]
    g.aim({"left_elbow_flex": base + math.radians(30)})
    first = g.tick()
    check("the first tick holds the arm where it is", first is not None
          and abs(first["left_elbow_flex"] - base) < 1e-12, str(first))

    # Steps of every size, under the defaults and under the knobs' far ends:
    # within the three limits all the way, at the command at the end, and
    # never past it.
    worst = []
    for knobs, seconds in ((None, 3.0), ((300, 10000, 100), 30.0), ((300, 10000, 200000), 3.0),
                           ((1, 10, 100), 40.0), ((60, 10, 100), 30.0)):
        # 2 deg is where Ruckig's profiles meet at the defaults - 2 a^3 / j^2.
        for degrees in (1, 2, 3, 9, 30):
            g = shaped_gate(*(knobs or ()))
            goal = base + math.radians(degrees)
            trace = drive(g, "left_elbow_flex", lambda t, goal=goal: goal, seconds)
            ok, said = within(g, trace)
            arrived = abs(trace[-1] - goal) < 1e-6
            past = max(trace) - goal
            if not (ok and arrived and past <= 1e-6):
                worst.append(f"{knobs or 'defaults'} {degrees} deg: {said}, off by "
                             f"{math.degrees(trace[-1] - goal):.3f}, past by {math.degrees(past):.3f} deg")
    check("steps of 1 to 30 degrees, at the defaults and the knobs' far ends, keep the three limits, "
          "arrive and go no further", not worst, "; ".join(worst[:3]))

    # A policy's commands as they come: thirty a second, a chunk's jump every
    # third of a second, jitter on top. Whatever they do, what goes to the
    # arm keeps to the same three limits.
    g = shaped_gate()
    trace = drive(g, "left_elbow_flex", lambda t: base + math.radians(15.0) * t
                  + (math.radians(4.0) if int(t * 3) % 2 else 0.0) + math.radians(0.4) * math.sin(37.0 * t), 4.0)
    ok, said = within(g, trace)
    check("jumpy commands go to the arm within all three limits", len(trace) == int(4 * hz) and ok, said)

    # Commands that stop coming: the arm is braked where it gets to, not
    # carried on to the last one, and then nothing more is sent.
    g = shaped_gate()
    far = base + math.radians(40)
    trace = drive(g, "left_elbow_flex", lambda t: far if t < 0.5 else None, 3.0)
    ok, said = within(g, trace)
    at_stop = trace[int(0.5 * hz)]
    check("commands that stop: braked short of the last, within the limits, then nothing sent",
          ok and trace[-1] < far - math.radians(5) and trace[-1] - at_stop < math.radians(10)
          and len(trace) < int(3 * hz) and not g.aiming,
          f"{said}; {math.degrees(trace[-1] - base):.1f} deg of 40, {len(trace)} sent")

    # A command refused (not a number) brakes the arm the same way.
    g = shaped_gate()
    trace = drive(g, "left_elbow_flex", lambda t: far if t < 0.5 else math.nan, 3.0)
    check("a refused command brakes the arm short of the last one passed",
          trace[-1] < far - math.radians(5) and not g.aiming, f"{math.degrees(trace[-1] - base):.1f} deg of 40")

    # A pose the scene stops - a wall at 10 degrees here - with the readings
    # three ticks behind what was sent, as the servos are: the arm stands at
    # the last pose sent and pushes on from it, never stepping back to a
    # reading that has not caught up.
    g = shaped_gate()
    wall = base + math.radians(10)
    passes = g._passes
    g._passes = lambda pose, *rest: None if pose.get("left_elbow_flex", base) > wall else passes(pose, *rest)
    sent = []
    for i in range(int(3 * hz)):
        clock[0] = i / hz
        if i % 2 == 0:
            g.aim({"left_elbow_flex": base + math.radians(40)})
        g.observe({**ready, "left_elbow_flex": sent[-3] if len(sent) >= 3 else base})
        allowed = g.tick()
        if allowed is not None:
            sent.append(allowed["left_elbow_flex"])
    back = max((a - b for a, b in itertools.pairwise(sent)), default=0.0)
    check("a pose the scene stops holds the arm where it was sent, no step back",
          sent and max(sent) <= wall and back <= 1e-9,
          f"{math.degrees(max(sent) - base):.2f} deg of the wall's 10, back by {math.degrees(back):.3f} deg")

    # The bands hold every pose sent, the shaping's included.
    joint = "left_gripper"
    _, hi = scene.limits[joint]
    g = shaped_gate(300, 10000, 200000)
    start_at = ready[joint]
    trace = drive(g, joint, lambda t: hi + 1.0, 2.0)
    check("a command past a band's end is shaped to the end and no further",
          trace and max(trace) <= hi + 1e-12 and abs(trace[-1] - hi) < 1e-6, f"{max(trace) - hi:.2e} past, from {start_at:.3f}")

    # A reading that stops coming lets the shape go: nothing is sent, and the
    # next command starts again from the arm.
    shaped = shaped_gate()
    shaped.aim({"left_elbow_flex": base})
    clock[0] += SETTINGS.reading_stale_s + 0.1
    check("a stale reading sends nothing while shaped", shaped.tick() is None and not shaped.aiming, shaped.reason)

    # The file is read as a policy takes the arms: what the page saves holds
    # from the next run.
    real_common = inference.common
    try:
        inference.common = lambda: {**real_common(), "smoothing": False}
        shaped.set_mode(MODE_IDLE)
        shaped.set_mode(MODE_POLICY)
        check("smoothing turned off holds from the next take-over", not shaped.shaping)
    finally:
        inference.common = real_common
    shaped.set_mode(MODE_IDLE)
    shaped.set_mode(MODE_POLICY)
    check("and on again", shaped.shaping)
    shaped.set_mode(MODE_IDLE)
    check("idle is not shaped", not shaped.shaping)

    print("\na hand's commands, shaped")
    for mode in (MODE_HANDHELD, MODE_STUDIO, MODE_CORRECT):
        shaped.set_mode(MODE_IDLE)
        shaped.set_mode(mode)
        check(f"{MODE_NAMES[mode]} is shaped", shaped.shaping)
    shaped.set_mode(MODE_IDLE)

    def hand_gate() -> Gate:
        clock[0] = 0.0
        g = Gate(scene, clock=lambda: clock[0])
        g.set_mode(MODE_HANDHELD)
        g.observe(ready)
        return g

    def hand_within(g: Gate, trace: list[float]) -> tuple[bool, str]:
        speeds, accels, jerks = rates(trace, hz)
        top = [max(map(abs, x), default=0.0) for x in (speeds, accels, jerks)]
        ok = top[0] <= g.rate * slack and top[1] <= g.hand_accel * slack and top[2] <= g.hand_jerk * slack
        return ok, f"{math.degrees(top[0]):.1f} deg/s, {math.degrees(top[1]):.0f} deg/s^2, {math.degrees(top[2]):.0f} deg/s^3"

    # The top camera's poses: a step every other tick, as a hand at 30 Hz.
    hg = hand_gate()
    joint = "left_elbow_flex"
    base = ready[joint]
    goal = base + math.radians(30)
    trace = drive(hg, joint, lambda t: goal, 1.5)
    ok, said = hand_within(hg, trace)
    check("a 30 deg step is taken under the hand's speed, acceleration and jerk", ok, said)
    check("it sets off gently - the first moves are small",
          len(trace) > 3 and abs(trace[3] - base) < math.radians(1.0),
          f"{math.degrees(trace[3] - base):.2f} deg after three ticks" if len(trace) > 3 else "too few ticks")
    check("and comes to rest on the goal",
          abs(trace[-1] - goal) < math.radians(0.05) and abs(trace[-1] - trace[-2]) < 1e-6,
          f"{math.degrees(trace[-1] - goal):.3f} deg off")

    # A hand that moves on and then is lost to the camera for a while: once
    # its commands have stopped for SHAPE_SILENCE_S the arm is braked to a
    # standstill where it gets to, and it sets off again when the hand is back.
    hg = hand_gate()
    ramp = lambda t: base + math.radians(60) * min(t, 0.5)  # noqa: E731
    trace = drive(hg, joint, lambda t: ramp(t) if t < 0.5 or t > 1.2 else None, 2.0)
    ok, said = hand_within(hg, trace)
    check("a marker lost and found: no lurch, under the same limits", ok, said)

    # Lost long enough to brake to rest, and back within the silence: the arm
    # sets off again from rest, a tick's worth first, not the time it stood.
    hg = hand_gate()
    clock[0] = 0.0
    hg.aim({joint: base + math.radians(20)})
    rest = base
    for i in range(int(2.0 * hz)):
        clock[0] = i / hz
        hg.observe({**ready, joint: rest})
        allowed = hg.tick()
        if allowed is not None:
            rest = allowed[joint]
        if not hg.aiming:
            break
    check("the step is braked to rest once its commands stop", not hg.aiming, f"at {clock[0]:.2f} s")
    clock[0] += 6 / hz
    hg.aim({joint: base + math.radians(40)})
    hg.observe({**ready, joint: rest})
    clock[0] += 1 / hz
    again = hg.tick()
    first = abs(again[joint] - rest) if again else 0.0
    check("back soon after coming to rest: the first step is one tick's worth",
          again is not None and first < math.radians(0.05), f"{math.degrees(first):.4f} deg")

    # Two hands, one lost: its arm brakes while the other goes on following.
    hg = hand_gate()
    other = "right_elbow_flex"
    sent_left, sent_right = [], []
    for i in range(int(1.2 * hz)):
        t = i / hz
        clock[0] = t
        if i % 2 == 0:
            asked = {other: ready[other] + math.radians(40) * min(t, 1.0)}
            if t < 0.3:
                asked[joint] = base + math.radians(40) * t
            hg.aim(asked)
        hg.observe({**ready, joint: sent_left[-1] if sent_left else base,
                    other: sent_right[-1] if sent_right else ready[other]})
        allowed = hg.tick()
        if allowed is not None:
            if joint in allowed:
                sent_left.append(allowed[joint])
            sent_right.append(allowed[other])
    lost_at = base + math.radians(40) * 0.3
    speeds, _, _ = rates(sent_left, hz)
    check("one hand lost: its arm brakes to rest short of where it was last sent",
          len(sent_left) > 2 and sent_left[-1] < lost_at and abs(speeds[-1]) < math.radians(5),
          f"stopped {math.degrees(lost_at - sent_left[-1]):.2f} deg short" if sent_left else "never moved")
    check("and the other goes on following",
          len(sent_right) > 2 and sent_right[-1] - ready[other] > math.radians(25),
          f"{math.degrees(sent_right[-1] - ready[other]):.1f} deg" if sent_right else "never moved")

    # A correction takes the jaw over on a block: the reading is the block's
    # width, short of the squeeze it was sent. Back at the squeeze in a tick
    # or two, at the joint speed - not over the shaping's fifth of a second.
    jg = Gate(scene, clock=lambda: clock[0])
    clock[0] = 0.0
    jg.set_mode(MODE_CORRECT)
    jaw = "left_gripper"
    on_block = ready[jaw]
    squeeze = on_block + math.radians(8)
    jg.observe({**ready, jaw: on_block})
    sent = []
    for i in range(6):
        clock[0] = i / hz
        jg.aim({jaw: squeeze})
        jg.observe({**ready, jaw: on_block})
        allowed = jg.tick()
        if allowed is not None:
            sent.append(allowed[jaw])
    per_tick = jg.rate / hz
    check("a correction's jaw on a block is back at its squeeze within a few ticks, ramped",
          len(sent) >= 3 and abs(sent[-1] - squeeze) < 1e-9
          and max(abs(b - a) for a, b in itertools.pairwise(sent)) <= per_tick * slack,
          f"{[round(math.degrees(q - on_block), 2) for q in sent]} deg")

    real_hand = SETTINGS.teleop_smoothing
    try:
        object.__setattr__(SETTINGS, "teleop_smoothing", False)
        off = Gate(scene, clock=STILL)
        off.set_mode(MODE_HANDHELD)
        check("teleop_smoothing off: a hand's commands are ramped, not shaped", not off.shaping)
    finally:
        object.__setattr__(SETTINGS, "teleop_smoothing", real_hand)

    print("\na source that commands slowly")
    # A policy at a dataset rate of a few Hz asks for a place every half
    # second, and the place is further than the arm can get in that time: the
    # arm must go on toward it between commands, not brake after a tenth of a
    # second of each gap.
    joint = "left_elbow_flex"
    lo, hi = scene.limits[joint]
    slow = shaped_gate(rate_dps=15.0, accel_dps2=400.0, jerk_dps3=8000.0)
    reach = lambda t: min(hi, ready[joint] + math.radians(30.0) * t)  # noqa: E731
    trace = drive(slow, joint, reach, 3.0, every=hz // 2)
    speeds, _, _ = rates(trace, hz)
    settled = speeds[int(1.5 * hz):]
    check("an arm asked every half second goes on between the commands",
          bool(settled) and min(settled) > math.radians(10.0),
          f"slowest {math.degrees(min(settled)):.1f} deg/s of {math.degrees(slow.policy_rate):g}" if settled else "no trace")
    # The same source gone: brought to rest, after a few of its gaps and not
    # at once.
    last = len(trace)
    for i in range(int(4.0 * hz)):
        clock[0] = 3.0 + i / hz
        slow.observe({**ready, joint: trace[-1]})
        allowed = slow.tick()
        if allowed is not None:
            trace.append(allowed[joint])
    speeds, _, _ = rates(trace, hz)
    check("and once the source is gone it is braked to rest",
          abs(speeds[-1]) < math.radians(1.0), f"{math.degrees(speeds[-1]):.2f} deg/s")
    check("not the instant a command is a tenth of a second late",
          abs(speeds[last - 1 + int(0.15 * hz)]) > math.radians(10.0),
          f"{math.degrees(speeds[last - 1 + int(0.15 * hz)]):.1f} deg/s a tenth and a half after the last")
    # One command a second, the dataset rate of 1 Hz.
    one = shaped_gate(rate_dps=15.0, accel_dps2=400.0, jerk_dps3=8000.0)
    trace = drive(one, joint, lambda t: min(hi, ready[joint] + math.radians(30.0) * t), 4.0, every=hz)
    speeds, _, _ = rates(trace, hz)
    settled = speeds[int(2.0 * hz):]
    check("one command a second is followed without stopping between them",
          bool(settled) and min(settled) > math.radians(10.0),
          f"slowest {math.degrees(min(settled)):.1f} deg/s" if settled else "no trace")

    print("\nan arm that holds no torque")
    limp = Gate(scene, clock=lambda: clock[0])
    clock[0] = 0.0
    limp.set_mode(MODE_HANDHELD)
    limp.observe(ready)
    limp.hold({"left": False, "right": True})
    far_left = ready["left_elbow_flex"] + math.radians(60)
    check("a command for the limp arm alone is refused, and says why",
          not limp.aim({"left_elbow_flex": far_left}) and "no torque" in limp.reason, limp.reason)
    both = {"left_elbow_flex": far_left, "right_elbow_flex": ready["right_elbow_flex"] + math.radians(20)}
    check("a command for both goes on for the arm that holds", limp.aim(both), limp.reason)
    sent = {}
    for i in range(int(1.0 * hz)):
        clock[0] = i / hz
        limp.observe(ready)
        allowed = limp.tick()
        if allowed is not None:
            sent = allowed
    check("only the holding arm is sent anything", sent and all(k.startswith("right_") for k in sent), str(sorted(sent)))
    check("and the limp arm's setpoint was not walked meanwhile",
          not any(k.startswith("left_") for k in limp._shape) and not any(k.startswith("left_") for k in limp._target))
    limp.hold({"left": True, "right": True})
    clock[0] += 1 / hz
    limp.observe(ready)
    check("torque back: the arm's next command starts at the reading",
          limp.aim({"left_elbow_flex": far_left}), limp.reason)
    first = limp.tick()
    check("and its first tick holds it there", first is not None
          and abs(first["left_elbow_flex"] - ready["left_elbow_flex"]) < 1e-9, str(first))
    unshaped = Gate(scene, max_speed_dps=0.0, clock=STILL)
    unshaped.set_mode(MODE_STUDIO)
    unshaped.observe(ready)
    unshaped.hold({"left": False, "right": True})
    check("an unshaped command for a limp arm is refused too",
          unshaped.check({"left_elbow_flex": far_left}) is None and "no torque" in unshaped.reason,
          unshaped.reason)

    print("\ncontact that is already there")
    import types
    held = Gate(scene, clock=STILL)
    key = ("left_forearm", "board")
    contact = lambda depth: {"a": key[0], "b": key[1], "depth": depth}  # noqa: E731
    reading = [[contact(0.0010)]]
    real_scene = held.scene
    held.scene = types.SimpleNamespace(touching=lambda pose: reading[0])
    held.observe(ready)
    check("a part resting against another may not press deeper",
          not held._stands(contact(0.0030), {"left"}))
    check("may stay as it is", held._stands(contact(0.0010), {"left"}))
    check("and may come back out", held._stands(contact(0.0004), {"left"}))
    check("a contact that is new is still refused", not held._stands({"a": "x", "b": "y", "depth": 0.0001}, {"left"}))
    reading[0] = [contact(0.0015)]
    held.observe(ready)
    check("read deeper than the gate let it go, it is still held to the shallowest",
          not held._stands(contact(0.0015), {"left"}) and held._stands(contact(0.0012), {"left"}))
    held.scene = real_scene

    print("\nthe way between two poses")
    swing_from = {"right_shoulder_pan": -46.2, "right_shoulder_lift": 183.3, "right_elbow_flex": -75.6,
                  "right_wrist_flex": 49.8, "right_wrist_yaw": 166.5, "right_wrist_roll": 1.9}
    swing_to = {"right_shoulder_pan": 64.9, "right_shoulder_lift": 9.3, "right_elbow_flex": -187.2,
                "right_wrist_flex": -38.5, "right_wrist_yaw": 185.3, "right_wrist_roll": 174.5}
    start = {**ready, **{k: math.radians(v) for k, v in swing_from.items()}}
    end = {**ready, **{k: math.radians(v) for k, v in swing_to.items()}}
    check("both ends of the swing are clear", not scene.touching(start) and not scene.touching(end))
    flat = Gate(scene, max_speed_dps=0.0, clock=STILL)
    flat.set_mode(MODE_STUDIO)
    flat.observe(start)
    flat.hold({})
    check("a step whose ends are clear but whose way is not is refused",
          flat.check({k: end[k] for k in swing_to}) is None and "would meet" in flat.reason, flat.reason)
    check("a short step along it passes",
          flat.check({k: start[k] + 0.2 * (end[k] - start[k]) * 0.01 for k in swing_to}) is not None, flat.reason)

    print("\nthe stop")
    gate.stop()
    check("nothing passes after the stop", gate.check({"left_elbow_flex": near}) is None,
          gate.reason)
    check("not armed", not gate.armed)
    ok, _ = gate.set_mode(MODE_HANDHELD)
    check("and a mode cannot re-arm it on its own", not ok and not gate.armed)
    gate.set_mode(MODE_IDLE)
    ok, _ = gate.set_mode(MODE_HANDHELD)
    check("coming back is deliberate: idle, then a mode", ok and gate.armed)

    if FAILURES:
        print(f"\nFAILED: {len(FAILURES)}")
        for name in FAILURES:
            print(f"  - {name}")
        print()
        return 1
    print("\nThe gate holds.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
