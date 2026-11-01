"""The pose the arms wait in - walked to by the page's ready button and on the way into a policy run.

The description's zero is where the ARMS park: the upper arm lying back over its
base, the forearm folded forward on it, the hand against the shoulder - a pose
the arm rests in on itself, and a thing you can photograph. It is also the worst
place to start following a hand from. shoulder_lift, elbow_flex and wrist_yaw all
sit on an end of their band there, and the body is in the hand's way: half the
ways the tool can go from zero are through a stop or through the arm.

So the ready pose is opened out, and it is stated as three things about ONE arm,
the right, which is the description's own frame.

Each gripper is held the way an operator holds a handheld at rest, the arms
out in front of them: well out over the desk, a hand's height above it, the
jaw looking down at a slant and turned in toward the middle. The board
stands vertical and its face looks along world +x; shoulder_pan turns about
the vertical, 105.5 mm out from the board. shoulder_pan stands 6 degrees
out, shoulder_lift at 92, the elbow is opened 38 degrees from its fold,
wrist_flex at -32 and wrist_yaw at 82, and the tool looks 33 degrees off +x
toward the middle and 38 degrees below level, meeting the desk 211 mm ahead
of it. It stands at (389, -141, 165) mm in world, and the two tips are about
270 mm apart.

The attitude, height and place are the operator's own, from a take held as
they found comfortable - the median of each hand over the take's first
second and a half, the left's mirrored onto the right and the two met half
way - so a take starts, and a hand that only waits waits, where the arm
already is. Held so, the arm can carry the tool 60 mm each way along the
desk and 40 mm up and down clear of every stop, and its gripper camera,
through the window a policy is shown, takes in 87 percent of the task mat
and three quarters of where its block and bowl are put. A hand held steeper
than this to show the camera more is one the arm meets only with
shoulder_pan swung round toward its stop.

The nearest stop is the elbow's fold, 38 degrees away; wrist_flex and
wrist_yaw have 82, shoulder_lift 92 and shoulder_pan 113.

wrist_roll turns about the tool's own approach and moves neither the tool
centre nor any of the distances above - it only says which way round the jaw
opens - and it is stated at 34 degrees, mirrored, the way round the operator's
jaw opened.

Both the page's ready button and a policy run's walk want these numbers, and
numbers kept in two places end up different in two places. They are here, and
both ask. Home is a different pose and needs no numbers: it is the description's
zero, every joint at 0, where the arms park. The walk between the two is
straight in joint space - the elbow opens as the shoulder lifts, which takes
the hand off the shoulder and keeps it off - and route_check walks it against
the scene, both arms at once, both ways.

The left is the mirror. Every joint axis follows its servo and a servo is turned
round rather than reflected to fit the mirrored arm, so the mirrored pose changes
sign on every joint. Its tool stands at (384, 127, 167) mm, 37 degrees off +x
and 38 down.
"""
from __future__ import annotations

import math

def mirrored(pose: dict[str, float]) -> dict[str, float]:
    """A right pose by bare joint name, as the left arm's mirror of it: every
    joint changes sign, gripper included - the left jaw opens toward negative."""
    return {name: -value for name, value in pose.items()}


READY_DEG = {
    "right": {
        "shoulder_pan": 5.8,
        "shoulder_lift": 92.4,
        "elbow_flex": -37.8,
        "wrist_flex": -32.1,
        "wrist_yaw": 82.1,
        "wrist_roll": 34.0,
    },
}
READY_DEG["left"] = mirrored(READY_DEG["right"])


def ready_joints(arm: str, joint_names: tuple[str, ...]) -> list[float]:
    """The ready pose as a vector in the order `joint_names` gives, degrees."""
    pose = READY_DEG[arm]
    return [pose[name.removeprefix(f"{arm}_")] for name in joint_names]


def ready_pose() -> dict[str, float]:
    """What is NOT zero in the ready pose, in radians, by joint name - both arms."""
    return {
        f"{arm}_{name}": math.radians(value)
        for arm, pose in READY_DEG.items()
        for name, value in pose.items()
        if value != 0.0
    }
