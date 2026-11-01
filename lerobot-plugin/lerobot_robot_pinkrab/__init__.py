"""Pinkrab's arms for LeRobot's own commands.

LeRobot imports this package by its distribution name, and importing it registers
`bi_pinkrab_follower` - so `--robot.type=bi_pinkrab_follower` works in
lerobot-calibrate, lerobot-record, lerobot-teleoperate and the rest.

Kept to one import on purpose: when a plugin fails to import, LeRobot logs it and
carries on, and the only sign left is a robot type it has never heard of.
"""

from pinkrab_core.robot import *  # noqa: F401,F403
from pinkrab_core.robot import __all__  # noqa: F401
