"""Pinkrab as a LeRobot robot: one arm, and the pair.

LeRobot's `SOFollower` describes the stock SO-ARM: six servos, of which five
articulate. This arm has seven. A printed clevis carries a wrist yaw between
`wrist_flex` and the wrist, so the arm reaches six articulated joints and can be
handed a full end-effector pose; the yaw sits fifth on the bus, which moves
`wrist_roll` and `gripper` up one id.

Two things follow from that, and they are the whole reason this module exists:

* the bus is built from `SETTINGS.motor_ids`, so the ids live in one place rather
  than being written out again here;
* `wrist_roll` has stops like every other joint: the gripper carries a printed
  lug that stops it at +/-175 degrees, so its range is swept rather than declared
  as a full turn.
"""

from __future__ import annotations

import logging
import time
from dataclasses import fields

from lerobot.cameras import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.robots.bi_so_follower import BiSOFollower
from lerobot.robots.so_follower import SOFollower
from lerobot.utils.decorators import check_if_already_connected
from lerobot.utils.utils import enter_pressed, move_cursor_up

from ..angles import joint_bands, record_zero_pose, sweep_direction_disagreements, unjudged
from .. import ports, servos
from ..config import ARMS, SETTINGS
from .config_pinkrab_follower import GRIPPER, BiPinkrabFollowerConfig, PinkrabArmConfig, PinkrabFollowerConfig

logger = logging.getLogger(__name__)


class PinkrabFollower(SOFollower):
    """One wall-mounted Pinkrab arm: six articulated joints and a gripper."""

    config_class = PinkrabFollowerConfig
    name = "pinkrab_follower"

    def __init__(self, config: PinkrabFollowerConfig):
        # Deliberately not SOFollower.__init__: that one writes out the stock six
        # motors. Everything else it does is here, and the rest of the class - the
        # feature lists, configure, get_observation, send_action - reads
        # `self.bus.motors` and needs no telling how many there are.
        super(SOFollower, self).__init__(config)
        self.config = config
        body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                name: Motor(
                    motor_id,
                    SETTINGS.motor_model,
                    MotorNormMode.RANGE_0_100 if name == GRIPPER else body,
                )
                for name, motor_id in SETTINGS.motor_ids.items()
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """SOFollower.connect, leaving the arm limp when its calibration is not checked.

        `calibrate=False` is how `pinkrab calibrate` and lerobot-calibrate open the
        arm, and the first thing either does next is ask for a hand on it. The stock
        connect configures the servos with torque on regardless, so the arm stiffened
        under that hand - through the redo prompt as well - until calibrate() let go.
        """
        self.bus.connect()
        if calibrate and not self.is_calibrated:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file "
                "or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure(torque=calibrate)
        logger.info(f"{self} connected.")

    def configure(self, torque: bool = True) -> None:
        """The servo registers, as every Pinkrab tool writes them.

        LeRobot's SO follower writes its gripper protection by name and switches
        torque on at the end; this writes the same values through servos.configure,
        which also resets the joints' protection to factory and, unless `torque` is
        False, switches torque on against the present position rather than whatever
        goal a servo still holds.
        """
        servos.configure(
            self.bus,
            p=self.config.position_p_coefficient,
            i=self.config.position_i_coefficient,
            d=self.config.position_d_coefficient,
            torque_after=torque,
        )

    def calibrate(self) -> None:
        """Take the held pose as zero, sweep every joint, centre each on its travel.

        The pose held when ENTER is pressed is the description's zero - the home pose
        in config/home.yaml - and it is written to joint_offsets.yaml as such, so a
        joint angle read off the bus needs no pairing by eye afterwards.

        That pose is often at one end of a joint's travel: several zeros are chosen
        against a stop. The stock routine makes the held pose the encoder's half
        turn, and a joint held near an end then crosses the 0/4095 seam during the
        sweep and records its range as the whole circle. Here the sweep is unwrapped
        as it is read, and the half turn goes to the midpoint of the two ends found,
        so the servo's limits bound the real travel wherever zero sits inside it.

        The stock arm exempts `wrist_roll` and declares it a full turn. Here the
        gripper carries a printed lug that meets the wrist's at +/-175 degrees, so
        the sweep finds a real end of travel and that is what belongs in the servo's
        EEPROM - which, on the write path, is the only thing between a command and
        the wall.
        """
        if self.calibration:
            answer = input(
                f"Press ENTER to write the existing calibration for id {self.id} to the "
                "motors, or type 'c' and press ENTER to measure it again: "
            )
            if answer.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        arm = str(self.id).rsplit("_", 1)[-1]
        if arm not in ARMS:
            raise ValueError(f"cannot tell which arm {self.id!r} is, expected a name ending in {ARMS}")

        motors = list(self.bus.motors)
        # What the servos held before, put back if this attempt does not finish: the
        # sweep starts by clearing their homing and limits, and a run that stops
        # half way would otherwise leave an arm reading raw ticks against offsets
        # measured for the old homing.
        previous = dict(self.calibration) if self.calibration else None
        self.bus.reset_calibration(motors)
        try:
            input(f"Hold {self} at its home pose, gripper closed, and press ENTER...")
            print(
                "Move every joint sequentially through its entire range of motion.\n"
                "Recording positions. Press ENTER to stop..."
            )
            sweeps = self._record_unwrapped_sweeps(motors)

            calibration = {}
            zero_ticks = {}
            for motor, m in self.bus.motors.items():
                try:
                    homing, lo, hi, zero = centred_calibration(*sweeps[motor])
                except ValueError as exc:
                    raise ValueError(f"{motor}: {exc}") from exc
                calibration[motor] = MotorCalibration(
                    id=m.id, drive_mode=0, homing_offset=homing, range_min=lo, range_max=hi
                )
                zero_ticks[motor] = zero
            # The sweep can tell, on a joint whose travel leans to one side of
            # home, whether the servo counts the way the description's axis points.
            # A disagreement is a servo fitted the other way round, and it would
            # write angles that move metal backwards, so nothing is saved until it
            # is settled.
            try:
                bands = joint_bands(arm)
            except Exception as exc:  # noqa: BLE001 - the description, not the arm
                print(f"WARNING: the description's joint bands could not be read ({exc});")
                print("         the sweep was not checked against the description's axes.")
            else:
                resolution = self.bus.model_resolution_table[SETTINGS.motor_model]
                wrong = sweep_direction_disagreements(
                    arm, {m: (0, low, high) for m, (_, low, high) in sweeps.items()},
                    bands, resolution=resolution,
                )
                if wrong:
                    raise ValueError(
                        "the sweep disagrees with the description's axes:\n  "
                        + "\n  ".join(wrong)
                    )
                blind = [m for m in unjudged(bands) if m in sweeps]
                if blind:
                    print(f"Not checked by the sweep - their travel straddles home: {', '.join(blind)}.")
                    print("  Turn each in the studio once this is saved, and see it moves the way it is drawn.")
        except BaseException:
            if previous:
                self.bus.write_calibration(previous)
                print("Calibration not changed: the servos hold what they held before.")
            raise
        self.calibration = calibration

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)
        print("Zero pose saved to", record_zero_pose(arm, zero_ticks))


    def _record_unwrapped_sweeps(self, motors: list[str]) -> dict[str, tuple[int, int, int]]:
        """Stream raw positions until ENTER, following each joint across the seam.

        Returns, per motor, the raw position the sweep started at and the lowest and
        highest positions reached relative to it, unwrapped. No hand turns a joint
        half a revolution between two reads, so the shorter way round between them
        is always the way it went.

        A servo now and then answers one read with a position it is not at. Folded
        into the extremes, that single number stretches a range - enough to take a
        joint with a printed stop past a whole turn. So a jump larger than a hand
        makes between reads is held back until the next read agrees with it, and
        counted; a real fast move costs one read of delay, a glitch costs nothing.
        """
        resolution = self.bus.model_resolution_table[SETTINGS.motor_model]
        jump = sweep_jump_ticks(resolution)
        start = self.bus.sync_read("Present_Position", motors, normalize=False, num_retry=5)
        tracker = {m: SweepTracker(start[m], resolution, jump) for m in motors}
        while True:
            now = self.bus.sync_read("Present_Position", motors, normalize=False, num_retry=5)
            for m in motors:
                tracker[m].feed(now[m])
            print("\n-------------------------------------------")
            print(f"{'NAME':<15} | {'LOW':>6} | {'NOW':>6} | {'HIGH':>6} | {'SPAN deg':>8} | {'IGNORED':>7}")
            for m in motors:
                t = tracker[m]
                span = (t.high - t.low) * 360.0 / resolution
                print(f"{m:<15} | {t.low:>6} | {t.travel:>6} | {t.high:>6} | {span:>8.1f} | {t.ignored:>7}")
            if enter_pressed():
                break
            move_cursor_up(len(motors) + 3)
            time.sleep(0.02)

        # A joint that moved less than this was not swept - left at home, most often
        # the gripper, closed - and limits written from it would all but lock it.
        least = round(SWEEP_MIN_DEG * resolution / 360.0)
        still = [m for m in motors if tracker[m].high - tracker[m].low < least]
        if still:
            raise ValueError(
                f"not swept through {SWEEP_MIN_DEG:.0f} degrees, so no range to record: {', '.join(still)}"
            )
        return {m: (start[m], tracker[m].low, tracker[m].high) for m in motors}

# The least travel a sweep has to show for a joint to count as swept. Every joint on
# this arm turns well past it; one left at home does not reach it.
SWEEP_MIN_DEG = 20.0


def sweep_jump_ticks(resolution: int = 4096) -> int:
    """The most a hand turns a joint between two reads: 30 degrees."""
    return resolution // 12


class SweepTracker:
    """One joint's unwrapped travel through a sweep, deaf to single-read glitches."""

    def __init__(self, start: int, resolution: int = 4096, jump: int | None = None) -> None:
        self.resolution = resolution
        self.half = resolution // 2
        self.jump = sweep_jump_ticks(resolution) if jump is None else jump
        self.last = start
        self.pending: int | None = None
        self.travel = self.low = self.high = 0
        self.ignored = 0

    def _step(self, a: int, b: int) -> int:
        return (b - a + self.half) % self.resolution - self.half

    def feed(self, reading: int) -> None:
        step = self._step(self.last, reading)
        if abs(step) > self.jump:
            if self.pending is None or abs(self._step(self.pending, reading)) > self.jump:
                # A jump nothing has confirmed yet. Hold it and wait for the next read.
                if self.pending is not None:
                    self.ignored += 1
                self.pending = reading
                return
        elif self.pending is not None:
            # Back where it was: the held jump was a glitch.
            self.ignored += 1
        self.pending = None
        self.travel += step
        self.low = min(self.low, self.travel)
        self.high = max(self.high, self.travel)
        self.last = reading

def centred_calibration(
    start: int, low: int, high: int, resolution: int = 4096
) -> tuple[int, int, int, int]:
    """Homing offset, position limits and zero tick for a joint centred on its travel.

    `start` is the raw encoder position at the held pose; `low` and `high` are the
    furthest the joint went either way from it, unwrapped. Feetech reports
    Present_Position = raw - Homing_Offset, and the offset register holds a sign and
    eleven bits, so it is kept within +/-(resolution/2 - 1). The last value is where
    the held pose reads once that offset is in place.
    """
    span = high - low
    half = resolution // 2
    if span >= resolution - 1:
        raise ValueError(
            f"a sweep of {span} ticks covers the whole turn - the joint has no end of "
            "travel to calibrate against"
        )
    centre = start + (low + high) // 2
    homing = (centre - (half - 1) + half) % resolution - half
    if homing == -half:
        homing += 1
    below = (low + high) // 2 - low
    lo = (half - 1) - below
    hi = lo + span
    return homing, lo, hi, lo - low

class BiPinkrabFollower(BiSOFollower):
    """Both arms as one robot, so a dataset carries them in one frame.

    Observations and actions are the single-arm keys with `left_` and `right_`
    in front, which is what the stock bimanual wrapper does and what any policy
    trained against one expects.
    """

    config_class = BiPinkrabFollowerConfig
    name = "bi_pinkrab_follower"

    def __init__(self, config: BiPinkrabFollowerConfig):
        # Not BiSOFollower.__init__ either, for the same reason: it builds stock
        # arms. The per-arm config is copied field by field rather than listed, so
        # a knob LeRobot adds to SOFollowerConfig arrives here without an edit.
        super(BiSOFollower, self).__init__(config)
        self.config = config

        self._top_level_cam_keys = set(config.cameras)
        per_arm = set(config.left_arm_config.cameras) | set(config.right_arm_config.cameras)
        clash = self._top_level_cam_keys & per_arm
        if clash:
            raise ValueError(f"Camera named twice, once per arm and once shared: {sorted(clash)}")

        arms = {}
        for side in ARMS:
            shared = getattr(config, f"{side}_arm_config")
            if not shared.port:
                raise ports.PortError(
                    f"The {side} arm has no port: none given and none stored.\n"
                    "  Run: pinkrab ports detect"
                )
            values = {f.name: getattr(shared, f.name) for f in fields(PinkrabArmConfig)}
            # The shared cameras ride with the left arm, and keep their bare names.
            if side == "left":
                values["cameras"] = {**shared.cameras, **config.cameras}
            arms[side] = PinkrabFollower(
                PinkrabFollowerConfig(
                    id=f"{config.id}_{side}" if config.id else None,
                    calibration_dir=config.calibration_dir,
                    **values,
                )
            )

        self.left_arm = arms["left"]
        self.right_arm = arms["right"]
        # Only for the parts of LeRobot that expect a `robot.cameras` attribute.
        self.cameras = {**self.left_arm.cameras, **self.right_arm.cameras}
