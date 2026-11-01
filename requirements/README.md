# requirements

Two layers, on purpose.

- **`*.in`** - hand written, loose. What Pinkrab actually needs and why. Read these
  to understand the dependencies.
- **`lock/<platform>.txt`** - generated, every version pinned including transitive
  ones. Install these to get *the same* environment rather than merely a working
  one.

`scripts/setup.sh` installs from the lock when one exists for the platform and
falls back to the `.in` files when it does not.

## Slices

A machine installs only the roles it plays. There is no reason for a laptop that
just drives the arms to pull a training stack.

| Slice | For | Pulls |
|---|---|---|
| `arm.in` | talking to the servos | lerobot[feetech] |
| `dev.in` | regenerating locks, linting, checking the description | pip-tools, ruff, xacro (non-Linux only) |
| `sim.in` | the model in MuJoCo - `pinkrab_mujoco` steps it for the studio and the gate, and the description check compares it against the URDF | mujoco |
| `umi.in` | the handheld collection pipeline - marker solids, camera capture, gripper pose | opencv, manifold3d |

The studio has no slice of its own. Its server is the standard library and its
three.js is vendored beside it, so a machine that wants the page installs `sim.in`
and nothing more.

Planned:

| Slice | For | Note |
|---|---|---|
| `train.in` | policy training | torch already arrives with lerobot, this adds the rest |

ROS dependencies do **not** live here. They are declared in each package's
`package.xml` and installed with `rosdep`, which is what a ROS user expects and
what colcon reads.

## Regenerating a lock

Only on a machine of that platform - the resolution is platform specific.

```bash
pip install -r requirements/dev.in
./scripts/lock.sh
```

Commit the result. A lock that is not committed is not a lock.
