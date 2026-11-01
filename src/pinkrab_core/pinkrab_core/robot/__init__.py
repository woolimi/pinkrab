"""The Pinkrab arms, as LeRobot robots.

Importing this package registers `bi_pinkrab_follower` - the pair - with LeRobot's
config registry, which is what makes `--robot.type=` accept it and what lets its
factory build one from a config alone. A single arm is not a type of its own: see
config_pinkrab_follower.
"""

from .config_pinkrab_follower import BiPinkrabFollowerConfig, PinkrabFollowerConfig
from .pinkrab_follower import BiPinkrabFollower, PinkrabFollower

__all__ = [
    "BiPinkrabFollower",
    "BiPinkrabFollowerConfig",
    "PinkrabFollower",
    "PinkrabFollowerConfig",
]
