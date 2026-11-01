# lerobot_robot_pinkrab

Registers Pinkrab's two arms as a LeRobot robot type, so LeRobot's own commands can
drive them:

```bash
pip install -e lerobot-plugin        # scripts/setup.sh does this
lerobot-calibrate --robot.type=bi_pinkrab_follower
```

| type | what |
| --- | --- |
| `bi_pinkrab_follower` | both arms, seven servos each |

With nothing more on the command line it is this rig's pair: each arm on the board
`pinkrab ports detect` stored for it, and calibrated under the ids and in the
directory `pinkrab calibrate` uses (`pinkrab_left`, `pinkrab_right` in
`~/pinkrab/calibration/robots/pinkrab_follower/`). Each of these can still be given:
`--robot.left_arm_config.port=...`, `--robot.id=...`, `--robot.calibration_dir=...`.

A single arm is not a type of its own. The arms are mirror images, and an arm alone
cannot tell which one it is: given the left arm's id with the right arm's port, it
drives the right arm by the left's calibration.

The arm classes live in `pinkrab_core.robot`; this package only makes LeRobot find
them. It sits outside `src/` so colcon does not build it.
