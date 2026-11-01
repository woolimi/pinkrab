"""Configuration for the Pinkrab arms.

Kept beside the robot rather than inside it, the way LeRobot lays out its own, so
that its registry can find the class from the config's module.

Only the pair is a LeRobot robot type. The two arms are mirror images, and an arm
on its own has no way of knowing which one it is: told the left arm's id with the
right arm's port, it drives the right arm by the left's calibration, and on a wall
that is a swing into the wall. The pair names its arms itself, from the port
assignment `pinkrab ports detect` stored and the ids `pinkrab calibrate` writes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig
from lerobot.robots.so_follower import SOFollowerConfig

from .. import ports
from ..config import SETTINGS

# The gripper is the one joint LeRobot normalises 0..100 rather than in degrees:
# what matters about it is how far open it is, not the angle of a servo horn.
GRIPPER = "gripper"


@dataclass
class PinkrabArmConfig(SOFollowerConfig):
    """One arm's knobs: the stock follower's, with the position gains taken from
    this project's config (its I is not the SO-101's 0) and one more default: a
    step ceiling.

    The commanded pose and the arm's pose are never identical when a run starts,
    and without a ceiling the arm closes that gap in a single frame. On a desk that
    is untidy; on a wall it is a swing into the wall.
    """

    max_relative_target: float | dict[str, float] | None = (
        SETTINGS.max_joint_speed_dps / SETTINGS.teleop_fps
    )
    position_p_coefficient: int = SETTINGS.servo_p_coefficient
    position_i_coefficient: int = SETTINGS.servo_i_coefficient
    position_d_coefficient: int = SETTINGS.servo_d_coefficient


@dataclass
class PinkrabFollowerConfig(RobotConfig, PinkrabArmConfig):
    """One arm as a robot: what the pair builds each of its arms from, and what
    `pinkrab calibrate` sweeps one at a time. Not registered with LeRobot."""


def _stored_port(arm: str) -> str:
    """The arm's board as `pinkrab ports detect` stored it, or empty if it never ran."""
    port = ports.stored().get(arm)
    return str(port) if port else ""


@RobotConfig.register_subclass("bi_pinkrab_follower")
@dataclass
class BiPinkrabFollowerConfig(RobotConfig):
    """The pair. Cameras named here are not attached to either arm.

    With nothing given it is this rig's pair: each arm on the board stored for it,
    calibrated under the ids `pinkrab calibrate` writes, in its directory.
    """

    left_arm_config: PinkrabArmConfig = field(default_factory=lambda: PinkrabArmConfig(port=_stored_port("left")))
    right_arm_config: PinkrabArmConfig = field(default_factory=lambda: PinkrabArmConfig(port=_stored_port("right")))
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # Each arm's id is this with _left or _right after it.
    id: str | None = SETTINGS.pair_id
    calibration_dir: Path | None = field(default_factory=lambda: SETTINGS.calibration_dir)
