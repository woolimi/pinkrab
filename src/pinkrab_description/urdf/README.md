# urdf

`pinkrab.urdf.xacro` is the whole description: one SO-ARM 101 arm as a macro -
derived from [TheRobotStudio/SO-ARM100][up] (Apache-2.0) - and a Pinkrab as a wall,
a jig and two of those arms.

The two arms are one mirrored pair, so the arm is written once and a `mirror` factor
flips every y offset and every roll and yaw. Four joint origins are given per hand
at the top of the macro: `wrist_flex` and `shoulder_pan`, which are not a
reflection of their opposite number, and `shoulder_lift` and `elbow_flex`, which
are.

Every joint axis points the way its servo counts up, so a joint angle is the servo's
reading plus an offset. A servo is turned round, not mirrored, to fit the mirrored
part, so on the left arm `shoulder_lift`, `elbow_flex` and `gripper` count the other
way from the right and a mirrored pose reads the opposite sign on those joints. The
joint limits are given per hand for the same reason.

[up]: https://github.com/TheRobotStudio/SO-ARM100

Where *your* Pinkrab is bolted is not here - it is in
[`../config/mount.yaml`](../config/mount.yaml), because it is a measurement of one
installation rather than a property of the robot.

```
world
 +- wall_link                 the wall surface, x pointing out of it
     +- pinkrab_base_link     the jig both arms share
         +- left_base_link    -> ... -> left_gripper_frame_link
         +- right_base_link   -> ... -> right_gripper_frame_link
```

The stock SO-ARM description assumes a desk the base is clamped to. Here each base
is a plate bolted flat to a vertical board, carrying the arm out from it with
`shoulder_pan` upright: an arm's `base_link` lies on the board's face, straight
behind that axis and level with the base's top, with the world's axes. The joints
load as upstream's do; the reachable volume and the collision envelope are the
board's and the desk's.

## Build it

```bash
xacro urdf/pinkrab.urdf.xacro > /tmp/pinkrab.urdf
xacro urdf/pinkrab.urdf.xacro mount:=/path/to/another_rig.yaml
xacro urdf/pinkrab.urdf.xacro arm:=left     # one bare arm, rooted at `mount`
./scripts/check_description.py              # expand and sanity-check it
```

`arm:=` is for kinematics asking about the shape of an arm rather than about a rig:
it drops the wall, the jig and the prefixes.

## Meshes

`../meshes/` holds the STLs the description points at, committed rather than fetched
at build time: a rig without internet still has to come up. They are in metres, as a
URDF reads them; `../print/` holds the printed ones again in millimetres, with the
Bambu Studio project for each arm.
