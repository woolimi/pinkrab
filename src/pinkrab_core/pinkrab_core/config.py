"""Single source of truth for every path, id and constant Pinkrab uses.

Nothing else in the package reads the environment or hardcodes a path — it all goes
through ``SETTINGS``. LeRobot is configured through one environment variable
(``HF_LEROBOT_HOME``); that is a process-boundary requirement of the ``lerobot-*``
CLIs, so it is exported from here and nowhere else. The same for ROS's
``RMW_IMPLEMENTATION``, which every node sets from here before it starts.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

HOME = Path.home()

# Pinkrab is two arms bolted to a wall — the claws. There is no leader arm: a leader
# has to be clamped down to be usable, and the wall is taken. Input comes from the
# browser-based virtual leader (pinkrab_studio) and later from the handheld UMI
# gripper, so both arms here are followers.
ARMS: tuple[str, ...] = ("left", "right")

# The handheld UMI grippers. Each carries ONE servo - the gripper's, torque off,
# read for the angle the hand is pinching at - on a board of its own. They are
# not arms: nothing plans for them, nothing commands them, and a rig with no
# handheld plugged in is a rig working normally. They are here because they sit
# on the same kind of bus, take the same kind of assignment, and are numbered by
# the same tool; keeping them in a second place would mean a second copy of all
# three.
HANDHELDS: tuple[str, ...] = ("umi_left", "umi_right")
DEVICES: tuple[str, ...] = ARMS + HANDHELDS

# The rig's two camera roles: the gripper cameras (the handhelds' and the arms',
# one model) and the top camera over the desk.
CAMERA_ROLES: tuple[str, ...] = ("gripper", "top")


@dataclass(frozen=True)
class CameraProfile:
    """One camera role's model as camera.yaml states it - what it is opened as
    and how its pictures are read. See that file for each field."""

    role: str
    model: str
    usb: tuple[int, int] | None
    width: int
    height: int
    fps: int
    fourcc: str
    latency_s: float
    crop: tuple[float, float] = (1.0, 1.0)
    exposure_unit_us: float = 100.0
    hfov_deg: float | None = None
    # The sensor's whole frame, and the part of it (x, y, width, height) the
    # mode opened shows, scaled to the resolution; the whole frame by default.
    sensor: tuple[int, int] | None = None
    region: tuple[int, int, int, int] | None = None

    @property
    def sensor_size(self) -> tuple[int, int]:
        return self.sensor or (self.width, self.height)

    @property
    def sensor_region(self) -> tuple[int, int, int, int]:
        return self.region or (0, 0, *self.sensor_size)

    @classmethod
    def from_yaml(cls, role: str, block: dict) -> CameraProfile:
        try:
            usb = block.get("usb")
            width, height = (int(v) for v in block["resolution"])
            profile = cls(role=role, model=str(block["model"]),
                          usb=None if usb is None else (int(usb[0]), int(usb[1])),
                          width=width, height=height, fps=int(block["fps"]),
                          fourcc=str(block.get("fourcc") or "MJPG"), latency_s=float(block["latency_s"]),
                          crop=tuple(float(v) for v in block.get("crop") or (1.0, 1.0)),
                          exposure_unit_us=float(block.get("exposure_unit_us") or 100.0),
                          hfov_deg=None if block.get("hfov_deg") is None else float(block["hfov_deg"]),
                          sensor=None if block.get("sensor") is None else tuple(int(v) for v in block["sensor"]),
                          region=None if block.get("region") is None else tuple(int(v) for v in block["region"]))
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ValueError(f"camera.yaml's {role!r} profile: {type(exc).__name__}: {exc}") from exc
        if len(profile.fourcc) != 4:
            raise ValueError(f"camera.yaml's {role!r} profile: fourcc {profile.fourcc!r} is not four letters")
        sw, sh = profile.sensor_size
        x, y, w, h = profile.sensor_region
        if not (0 <= x and 0 <= y and x + w <= sw and y + h <= sh):
            raise ValueError(f"camera.yaml's {role!r} profile: region {list(profile.sensor_region)} is not inside "
                             f"the {sw}x{sh} sensor")
        if len(profile.crop) != 2 or not all(0.0 < v <= 1.0 for v in profile.crop):
            raise ValueError(f"camera.yaml's {role!r} profile: crop {list(profile.crop)} is not two "
                             "fractions in (0, 1]")
        return profile

    def as_dict(self) -> dict:
        from dataclasses import asdict

        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}


# camera.yaml read once per change: (path, mtime) -> profiles by role.
_PROFILES: dict[tuple[str, float], dict[str, CameraProfile]] = {}


@dataclass(frozen=True)
class Settings:
    # --- workspace -------------------------------------------------------
    # Everything LeRobot writes (calibration files, recorded datasets) lands here.
    # Deliberately not ~/.cache/huggingface: that directory already holds calibration
    # from earlier SO-ARM and OpenMANIPULATOR experiments under ids like
    # `my_follower_arm`, and a stray id match there would be read silently.
    workspace: Path = HOME / "pinkrab"

    # --- arms ------------------------------------------------------------
    # This arm has a joint the stock SO classes do not, so it is its own robot class
    # rather than an so101_* alias; calibration lands in the directory LeRobot names
    # after it. Only the pair is registered as a LeRobot type, by pinkrab_core.robot.
    follower_type: str = "pinkrab_follower"
    bi_follower_type: str = "bi_pinkrab_follower"

    # Calibration ids are derived from this, never typed by hand.
    id_prefix: str = "pinkrab"

    # Feetech STS3215, 7 per arm: six articulated joints and the gripper. Listed base
    # outward, which is also the order they are daisy-chained in, so the servo ids
    # follow from this list rather than being typed twice.
    #
    # wrist_yaw is not upstream's. Stock SO-ARM 101 leaves three axes parallel and
    # reaches five dimensions of the six in SE(3); the yaw - a printed clevis between
    # wrist_flex and the wrist - is what lets a full end-effector pose be commanded.
    # It sits FIFTH on the bus, so wrist_roll and the gripper are ids 6 and 7 rather
    # than 5 and 6.
    joint_names: tuple[str, ...] = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_yaw",
        "wrist_roll",
        "gripper",
    )
    # Which of the joints above the handheld's one servo stands for. It reads the
    # same thing the arm's gripper servo does - how far the jaw is open - so it
    # carries the same name and the same id, and nothing downstream has to learn
    # a second name for one number.
    handheld_joint: str = "gripper"
    motor_model: str = "sts3215"
    baudrate: int = 1_000_000
    @property
    def arm_dof(self) -> int:
        """Articulated joints, excluding the gripper."""
        return len(self.joint_names) - 1

    # --- teleoperation ---------------------------------------------------
    teleop_fps: int = 60
    # --- the gripper's grip ---------------------------------------------------
    # Torque_Limit written to each arm's gripper servo whenever its torque is
    # switched on, in the servo's 0..1000 scale. A RAM register: it goes back to
    # Max_Torque_Limit at every power-up, so it is written per session and the
    # EEPROM value is never touched. It caps the current, and with it the speed.
    # 250 is a quarter of the stall torque; 150 and 500 were tried either side
    # of it on 2026-09-20 and 250 kept.
    gripper_torque_limit: int = 250

    # --- servo registers ----------------------------------------------------
    # What LeRobot's SO-101 writes into every STS3215 on connecting
    # (SOFollower.configure), and what servos.configure writes here, so the arms
    # hold the same values whichever tool opened them: the position loop's
    # gains, and the gripper's protection - half the torque and current, a
    # quarter when overloaded - so a jaw closed on something does not burn out.
    # The other joints get the STS3215's factory protection back, read off a
    # servo nothing had configured: an SO-101's gripper keeps its protection when
    # it is renumbered into another joint.
    #
    # I is the one value that is not LeRobot's (0). With P alone a servo stops
    # where its pull no longer beats the arm's weight: stepped on this arm,
    # shoulder_lift stopped 3 deg short going down with gravity and elbow_flex
    # 1.4. At I 8 the same steps end within 0.5 deg and settle in 0.25-0.4 s;
    # 16 overshot by 2 deg (step_response.py --i, 2026-09-26).
    servo_p_coefficient: int = 16
    servo_i_coefficient: int = 8
    servo_d_coefficient: int = 32
    gripper_max_torque_limit: int = 500
    gripper_protection_current: int = 250
    gripper_overload_torque: int = 25
    joint_max_torque_limit: int = 1000
    joint_protection_current: int = 310
    joint_overload_torque: int = 80
    # --- the ramp ---------------------------------------------------------
    # The most any joint may be driven per second, everywhere a command is walked
    # toward rather than jumped to: the simulation's actuators, the gate on the
    # way to the servos, and the IK solver's joint velocity limit. One number so
    # the picture, the metal and the solver agree on how fast an arm moves -
    # under a policy, both the gate and the simulation walk at the common
    # max_joint_dps of inference.yaml instead (policy_max_joint_dps when it
    # names none - see inference.policy_dps). It is
    # also about what the servos can do, so the ramp is a ceiling rather than a
    # brake. The commanded pose and the arm's pose are never identical when a run
    # starts, and without this the arm closes that gap in a single frame - Pinkrab
    # hangs on a wall, and a frame that large swings into it. 0 disables.
    max_joint_speed_dps: float = 300.0
    # How long after a command is sent the servo starts to move, seconds: the
    # fit of scripts/fit_servo.py to step responses recorded on the arm. The
    # MJCF has no way to state it, so the simulation holds each command back by
    # this much, and fit_servo scores the model with it.
    servo_delay_s: float = 0.04

    # --- the desk under the fingers ---------------------------------------
    # The one contact that is wanted: a thin object - a card, a coin - lies flat
    # on the desk, and only fingers that reach the surface close under it. So
    # the fingers may press the desk this deep, in the model, and nothing else
    # may touch it: the top camera places the hand to about 2 mm against the
    # desk (measured 2026-10-04 with its board-fitted lens), and the rest is the
    # convex pieces the model's fingers are made of. A contact counts as the
    # fingers' when it lands this close to the tool centre - the camera bracket
    # on the same link is ten centimetres away.
    fingertip_desk_mm: float = 3.0
    fingertip_reach_mm: float = 30.0

    # --- parking ----------------------------------------------------------
    # Ctrl-C on the bringup takes the arms home before it lets go of them, at
    # this speed and within this time. Slower than the ramp: nobody is steering.
    # The speed is eased in over park_ease_s and eased out over the last
    # park_slow_deg of the longest joint's travel, so the arms neither jerk off
    # the mark nor arrive on it. An arm that is not home when the time is up
    # keeps its torque - dropping it mid-air is the one thing worse than leaving
    # it stiff.
    park_speed_dps: float = 30.0
    park_ease_s: float = 1.0
    park_slow_deg: float = 20.0
    park_timeout_s: float = 12.0
    # The walk is over when every joint reads within park_tolerance_deg of home
    # - wide enough that a servo settling under its own weight still counts.
    # Left there, a joint carrying the arm's weight stops short of home by
    # about that much. So once the walk is over the exact home pose is held for
    # up to park_settle_s more, until the readings are within park_settle_deg,
    # and only then is torque let go.
    park_tolerance_deg: float = 3.0
    park_settle_deg: float = 1.0
    park_settle_s: float = 2.0

    # --- the differential IK ------------------------------------------------
    # What a joint step costs the solver beside the task error. Zero is a plain
    # least squares, which leaps wherever the Jacobian is nearly singular; this
    # wrist is nearly singular over most of its workspace. Measured on a 3.7 mm
    # target step: 1e-4 and 1e-3 both close it to about a millimetre in one
    # solve, 1e-2 takes five. Bigger is calmer and later.
    ik_regularization: float = 1e-3
    # The tick the speed above is spent over: the rate the teleop loop and the
    # take player call the solver at.
    ik_rate_hz: float = 30.0
    # Time constant of the low-pass the teleop node puts on the handheld's pose
    # before the arm follows it, in seconds. The tracker's own noise is under a
    # millimetre and half a degree, but the solver turns half a degree at the
    # tool into several degrees across the wrist and elbow, and a still hand
    # came out as an arm shaking by 6 to 8 degrees peak to peak. At 30 Hz a
    # tenth of a second keeps about a third of that noise and puts the arm a
    # tenth of a second behind the hand. Zero switches the filter off.
    teleop_pose_smoothing_s: float = 0.1
    # A hand out of sight longer than this is taken up again when it comes
    # back - it has moved meanwhile, perhaps far: the arm follows the hand's
    # motion, so its origin is taken again where the hand is, and the arm
    # does not jump after the move it made unseen.
    teleop_regrab_s: float = 0.5
    # The live view's newest solve counts as a picture only while it keeps
    # changing: a solve that stays the same this long is a camera that has
    # stopped, however alive the server still is, and the hand it shows is
    # not followed. Compared on this side's clock, so a live view on another
    # machine needs no synchronised one.
    teleop_picture_stale_s: float = 0.3
    # Correcting a policy (MODE_CORRECT) the handheld takes the arm over where
    # it stands. The jaw is not taken at once - an arm holding a block would
    # drop it to an open hand. It is taken when the hand's opening reaches what
    # the jaw was last SENT (0..1 of each band) - not where it reads, which on
    # a block is the block's width, short of the squeeze - crossing it from
    # whichever side the hand started, or coming within this of it; and it
    # follows the hand from then on.
    correct_jaw_catch: float = 0.02

    # How often the servo bus is read. The gate ramps from the last reading and
    # checks contact against it, so readings are kept fresher than the teleop
    # frame rate. One sync read of an
    # arm's seven servos is a few milliseconds at 1 Mbaud; two arms fit in a
    # 60 Hz tick with room for the writes.
    arm_poll_hz: int = 60
    # How old a joint's last reading may be before nothing is sent to it: the
    # gate refuses a command for a joint not read this recently, and the
    # bridge's pose walk holds where it is. A bus whose reads fail still takes writes - a
    # sync write waits for no answer - so an arm commanded from a reading
    # that stopped coming moves while every check is made against where it
    # was. Fifteen polls at arm_poll_hz.
    reading_stale_s: float = 0.25
    # How many times turning an arm's torque off is tried again after a bus
    # error before it is reported failed: a dropped status packet on the stop
    # must not leave an arm rigid.
    torque_off_retries: int = 5

    # --- a trained policy driving ------------------------------------------
    # Each of these is Pinkrab's own value, run unless the workspace's
    # inference.yaml names another (see pinkrab_core.inference): this block is
    # what an empty file means.
    #
    # How long each predicted chunk runs before the policy looks again, in
    # seconds - n_action_steps is this at the checkpoint's dataset's rate - for
    # every kind of policy with a chunk, listed in inference.KINDS or not. A chunk is trained long
    # (ACT's is 2 s), and run whole it is 2 s with the camera ignored; a
    # tenth of a second is a new look ten times a second, the rate UMI runs its
    # policies at - three steps at 30 fps.
    policy_action_s: float = 0.1
    # How fast the gate lets a joint turn while a policy drives, degrees per
    # second - see `inference.policy_dps`. A solver chasing a target out of reach swings
    # the arm through another configuration, and at the gate's own 300 deg/s
    # that swing meets the wall before the contact check says anything; at this
    # speed the check holds the arm short of it. Positive: a policy is never
    # unramped.
    policy_max_joint_dps: float = 60.0
    # The gate's shaping of a policy's commands (inference.yaml's smoothing,
    # on unless turned off): each joint moved on toward the latest command at
    # policy_shaping_hz, its speed under the policy's max_joint_dps, its
    # acceleration under policy_max_joint_accel_dps2 and the change of that
    # under policy_max_joint_jerk_dps3. A policy answers in chunks, and where
    # one chunk meets the next the command can turn at once; shaped, the arm
    # turns over a tenth of a second. 400 deg/s^2 takes a joint to the 60
    # deg/s ceiling in 0.15 s, the jerk builds that acceleration in 0.05 s.
    # Each tick heads for the latest command as a place to stop at, so a
    # joint following a moving one runs a little behind it (about 2 deg at
    # 30 deg/s) and never past a still one.
    policy_shaping_hz: float = 60.0
    # The rate a dataset is converted at unless another is asked for: one of
    # the policy's steps every 1/dataset_fps s, which is also how often a
    # policy trained on it is asked. Any rate up to policy_shaping_hz goes;
    # past it, a step between two of the gate's ticks would never reach a
    # servo. Pictures and poses are laid on that clock whatever the cameras'
    # own rates (pinkrab_umi.convert). 50, as ALOHA's takes and policies run.
    dataset_fps: int = 50
    policy_max_joint_accel_dps2: float = 400.0
    policy_max_joint_jerk_dps3: float = 8000.0
    # The same shaping of a hand's commands - the handheld followed, the page's
    # jog, a correction, the pose walks - on unless this is False. Without it
    # each new pose the top camera solves, 30 a second, is a step the servo
    # lurches to and stops at, and a marker lost and found is a stop and a
    # start at full speed; shaped, the arm sets off, follows and comes to rest
    # with its acceleration ramped. The speed stays max_joint_speed_dps. Higher
    # limits follow a fast hand closer and smooth less: 900 deg/s^2 takes a
    # joint to 300 deg/s in a third of a second, 15000 deg/s^3 builds that
    # acceleration in 0.06 s - raise both for a throw. The jaws are ramped at
    # that speed and not shaped: a grasp and a release are the hand's to time.
    teleop_smoothing: bool = True
    teleop_max_joint_accel_dps2: float = 900.0
    teleop_max_joint_jerk_dps3: float = 15000.0
    # A policy's solve keeps every joint this far inside its band, and a joint
    # held there is at its stop. Being there ends nothing - the takes hold
    # wrist_yaw at its stop for long stretches - but while it is, what the
    # policy asks for and the arm does not make is summed, and the run ends
    # once that passes the slip below: the policy pushing past the reach the
    # takes were made in, the solver bending the other joints after it.
    policy_limit_margin_deg: float = 1.0
    # A policy's next step waits while an arm's joints read further than this
    # from the pose it was last sent, or are not being read: the gate is
    # refusing, or holding the arm at a contact, and a step composed onto a
    # pose the arm never reached walks the command away from it. The run ends
    # when the wait lasts policy_follow_hold_s.
    policy_follow_lag_deg: float = 15.0
    policy_follow_hold_s: float = 2.0
    # How far one tick's solve may fall short of the pose the policy asked for
    # before the run ends: further is a step the arm could not take. A replay
    # of the takes misses by 8 mm at most. Also how far the misses may add up
    # while a joint is at its stop.
    policy_slip_mm: float = 10.0
    policy_slip_deg: float = 8.0
    # A step the policy asks for faster than these is a bad answer, not a
    # motion: held. Speeds, so they hold at any dataset rate - 30 mm and 10
    # degrees a step at 30 fps.
    policy_max_tool_speed_mm_s: float = 900.0
    policy_max_tool_turn_dps: float = 300.0
    # How much the arm's solve weighs the tool's attitude against its place: a
    # wrist that cannot reach both is let off the turn first.
    policy_orientation_weight: float = 0.1
    # Ready is a place, not a joint vector: the arm is there when every joint
    # the ready pose names is this close to it.
    policy_ready_tol_deg: float = 3.0
    # A checkpoint looks at its pictures (pinkrab-picture-swap) when, on every
    # hand, its answer with another frame's pictures points no closer than
    # this to its answer with its own. Simulated checkpoints that went for the
    # block measure +0.19 to +0.50, real ones that answered from their state
    # and drove off +0.63 and +0.95 - few runs, a line between them.
    swap_max_picture_cos: float = 0.55

    # --- a policy held at a joint's stop --------------------------------------
    # Not an inference.yaml setting. What a policy asks past a stop is summed
    # while any joint is within this of where the solve holds it, and let go
    # only once every joint is further off: a push that lifts a joint a hair
    # off its stop for a tick is the same push.
    stop_release_deg: float = 0.5

    # --- the training page ----------------------------------------------------
    # Runs are read from the train directory here every train_poll_s, and from
    # each ssh source every train_remote_poll_s: a login to a cloud box takes a
    # second or two, and a run logs a line every log_freq steps, a few times a
    # minute at most. A run whose files have not grown for train_stale_s and
    # that has not said it ended is shown as gone quiet - killed, or its
    # machine taken away.
    train_poll_s: float = 10.0
    train_remote_poll_s: float = 30.0
    train_stale_s: float = 300.0

    # --- ROS ---------------------------------------------------------------
    # The middleware every node of this robot runs on, whatever the shell that
    # started it says. Two on one graph pass topics but not services: a Fast
    # DDS client's request reaches a Cyclone service with its fields zeroed -
    # a mode asked for arrives as idle, a torque on as off - and neither side
    # reports anything. See `pin_ros_middleware`.
    ros_middleware: str = "rmw_cyclonedds_cpp"

    # --- serial ports ----------------------------------------------------
    # macOS names the device after the board's USB serial number
    # (/dev/tty.usbmodem5B7B0139551) and Linux exposes the same stability through
    # /dev/serial/by-id. Both survive a replug; /dev/ttyACMx does not, which is the
    # whole reason the assignment is pinned to a file.
    ports_file: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ports_file", self.workspace / "ports.json")
        if self.policy_max_joint_dps <= 0:
            raise ValueError("policy_max_joint_dps must be positive - a policy is never unramped")

    # --- derived ---------------------------------------------------------
    @property
    def is_macos(self) -> bool:
        return sys.platform == "darwin"

    @property
    def serial_dir(self) -> Path:
        return Path("/dev") if self.is_macos else Path("/dev/serial/by-id")

    @property
    def port_glob(self) -> str:
        """Glob matching a servo board, stable across replug on both platforms."""
        return "tty.usbmodem*" if self.is_macos else "*"

    def arm_id(self, arm: str) -> str:
        """'left' -> 'pinkrab_left'."""
        return f"{self.id_prefix}_{arm}"

    @property
    def pair_id(self) -> str:
        """Calibration id for the pair.

        The bimanual wrapper derives each arm's id by appending _left and _right to
        this, so a pair driven through it reads exactly the files written when the
        arms were calibrated one at a time.
        """
        return self.id_prefix

    @property
    def calibration_dir(self) -> Path:
        """Where LeRobot stores calibration, keyed on the robot class name."""
        return self.workspace / "calibration" / "robots" / self.follower_type

    def calibration_file(self, arm: str) -> Path:
        return self.calibration_dir / f"{self.arm_id(arm)}.json"

    @property
    def description_pkg(self) -> Path:
        """The pinkrab_description package, however this checkout is laid out.

        Walked up from this file rather than looked up in the ament index, because
        the tooling has to work the same on a machine with no ROS. A colcon install
        and a pip checkout put this package at different depths, so search instead
        of counting parents - counting is what silently pointed at the wrong
        directory the first time.
        """
        here = Path(__file__).resolve()
        for parent in here.parents:
            for candidate in (parent / "pinkrab_description", parent / "share" / "pinkrab_description"):
                if (candidate / "urdf").is_dir():
                    return candidate
            for candidate in (parent / "src" / "pinkrab_description",):
                if (candidate / "urdf").is_dir():
                    return candidate
        # Nothing found: return the most likely path so the error names a place.
        return here.parents[2] / "pinkrab_description"

    @property
    def xacro_file(self) -> Path:
        return self.description_pkg / "urdf" / "pinkrab.urdf.xacro"

    @property
    def mesh_dir(self) -> Path:
        return self.description_pkg / "meshes"

    @property
    def mount_file(self) -> Path:
        return self.description_pkg / "config" / "mount.yaml"

    @property
    def camera_file(self) -> Path:
        """The rig's camera models and the gripper camera's pose."""
        return self.description_pkg / "config" / "camera.yaml"

    def camera_profile(self, role: str) -> CameraProfile:
        """`role`'s camera (CAMERA_ROLES) as camera.yaml states it."""
        import yaml

        path = self.camera_file
        key = (str(path), path.stat().st_mtime)
        if key not in _PROFILES:
            data = yaml.safe_load(path.read_text()) or {}
            _PROFILES.clear()
            _PROFILES[key] = {r: CameraProfile.from_yaml(r, data.get(r) or {}) for r in CAMERA_ROLES}
        return _PROFILES[key][role]

    @property
    def offsets_file(self) -> Path:
        """What a servo tick means in URDF joint angles, for this calibration.

        Lives in the workspace beside the calibration it depends on, not in the
        repository: re-homing an arm invalidates it, and it describes one rig.
        """
        return self.workspace / "joint_offsets.yaml"

    @property
    def inference_file(self) -> Path:
        """How trained policies are run on this rig, per policy type: the
        settings that change inference without retraining. In the workspace,
        as the studio's policy page writes it; see pinkrab_core.inference."""
        return self.workspace / "inference.yaml"

    @property
    def handheld_file(self) -> Path:
        """What the handheld's servo ticks mean, as a fraction of its jaw's travel.

        Beside the arms' offsets and for the same reason: it describes one
        printed handheld with one servo pressed into it, not the design. A jaw
        reprinted or a servo swapped invalidates it.
        """
        return self.workspace / "umi_gripper.yaml"

    # --- the foot switch ------------------------------------------------------
    # A PCsensor three-pedal USB foot switch, seen by the host as a keyboard.
    # Its pedals are programmed (once, with the `footswitch` tool) to F16, F17
    # and F19 - keys macOS binds to nothing, where F14 and F15 dim and brighten
    # the screen and F18 is a common choice for the Korean/English switch - and
    # the studio reads the keyboard interface itself, so a
    # press reaches the robot whichever window has focus. Left to right, as the
    # unit sits on the floor and as the page draws them: jog, follow, stop.
    footswitch_usb: tuple[int, int] = (0x3553, 0xB001)
    footswitch_pedals: dict[int, str] = field(default_factory=lambda: {
        0x6B: "jog",       # F16
        0x6C: "handheld",  # F17
        0x6E: "stop",      # F19
    })

    @property
    def desk_file(self) -> Path:
        """Which way the desk solid faces, seen from the robot: a yaw in degrees.

        The handheld's pose comes in the desk solid's frame and the arm's in the
        robot's. Both stand level on the same desk, so the two frames differ by
        a turn about the vertical, and that turn is `yaw_deg`: the world's axes
        turned by it are the robot's - kept with each take; teleop does not use
        it. Beside the offsets, for the same reason: it describes where one
        solid was put down, and moving it invalidates it.

        Also `height_mm`: the solid's origin above the desk surface, half a
        face as printed (it stands on a cut one half-face below its centre).
        See desk.py.
        """
        return self.workspace / "desk.yaml"

    @property
    def cameras_file(self) -> Path:
        """Which USB socket each gripper camera is in.

        The two modules are the same part and report the same serial number,
        so the only thing that tells them apart is where they are plugged in.
        Recorded by `pinkrab-umi cameras`, which has you plug them in one at a
        time; beside the offsets because it describes one rig's cabling, and a
        plug moved to another socket invalidates it.
        """
        return self.workspace / "cameras.yaml"

    @property
    def follower_cameras_file(self) -> Path:
        """Which USB socket each follower arm's gripper camera is in.

        The same module as the handhelds', named the same way by `pinkrab
        cameras`. The two files may name the same sockets: the rig has two
        camera ports, the handhelds' cameras are plugged in to record and the
        arms' to run a policy, so which file is read is what decides whose
        camera a socket holds.
        """
        return self.workspace / "follower_cameras.yaml"

    @property
    def lens_file(self) -> Path:
        """Each camera's lens - the gripper cameras' and the top camera's - as
        `pinkrab-umi lens fit` and `pinkrab lens fit` measured it.

        A lens is a unit, not a design, so the fit lives with this rig's other
        calibrations and is never committed.
        """
        return self.workspace / "lens.yaml"

    @property
    def episodes_dir(self) -> Path:
        """Where recorded episodes land: raw, one directory each, outside the repo.

        An episode is three videos and a few jsonl files of one hand on one
        desk; the LeRobot dataset is made from these later and can be remade.
        """
        return self.workspace / "episodes"

    @property
    def train_dir(self) -> Path:
        """Where a training run keeps its checkpoints.

        Under the workspace with the datasets a run reads, because a checkpoint
        is about this rig too: it was fit to these cameras at these angles, and
        it means nothing on another.
        """
        return self.workspace / "train"

    @property
    def keys_dir(self) -> Path:
        """ssh keys handed to the training page for its sources, one per source."""
        return self.workspace / "keys"

    @property
    def remote_lock_file(self) -> Path:
        """The Linux lock, which a training machine is set up from.

        Searched for from this file, as the description is: a checkout keeps it
        under requirements/lock/, and a machine trained on must end up with the
        versions this one runs, or its checkpoints will not load here.
        """
        here = Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / "requirements" / "lock" / "linux.txt"
            if candidate.is_file():
                return candidate
        return here.parents[3] / "requirements" / "lock" / "linux.txt"

    @property
    def train_db(self) -> Path:
        """The training page's store: every run it has read, from any machine."""
        return self.workspace / "train.sqlite"

    @property
    def studio_port(self) -> int:
        return 8730

    @property
    def scene_file(self) -> Path:
        """The MuJoCo scene: the robot, plus the desk and board it stands on."""
        return self.description_pkg / "mjcf" / "scene.xml"

    @property
    def urdf_file(self) -> Path:
        """The description everything geometric derives from.

        Inverse kinematics - the only way to drive these arms without a leader -
        cannot run without it.
        """
        return self.workspace / "urdf" / "pinkrab.urdf"

    @property
    def motor_ids(self) -> dict[str, int]:
        """shoulder_pan..gripper -> 1..7, base outward."""
        return {name: i for i, name in enumerate(self.joint_names, start=1)}

    def subprocess_env(self) -> dict[str, str]:
        """Environment for the ``lerobot-*`` child processes."""
        env = dict(os.environ)
        env["HF_LEROBOT_HOME"] = str(self.workspace)
        return env

    def pin_ros_middleware(self) -> None:
        """Run this process's ROS on `ros_middleware`. Called by every node
        before ``rclpy.init``, which loads the middleware the environment names
        then - so a node started by hand from a shell set up for something else
        still joins the robot's graph on the robot's terms."""
        os.environ["RMW_IMPLEMENTATION"] = self.ros_middleware

    def shell_ros_middleware(self) -> str | None:
        """The middleware this process's environment names - what a `ros2`
        command typed in the same shell would run on. None when unset, which
        is the distribution's default and not this robot's."""
        return os.environ.get("RMW_IMPLEMENTATION")

    def shell_ros_domain(self) -> str | None:
        """The ROS domain this process's environment names, which the bridge it
        starts runs on. None when unset, which is domain 0."""
        return os.environ.get("ROS_DOMAIN_ID")


SETTINGS = Settings()
