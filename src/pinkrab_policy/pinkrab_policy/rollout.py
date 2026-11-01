"""A trained policy, driving the arms.

The policy was trained on what the handhelds' cameras saw and how each
handheld moved from there - a delta in its tool's own frame - and, with two,
where one tool stood from the other. So this shows it what the arms' gripper
cameras see now and where the arms' tools stand from each other, takes the
deltas it answers with, composes each onto where that tool was last put, and
solves its arm for that pose the way teleop does: `track()`, one step,
through the gate. Each gripper follows the policy's opening across its arm's
own band. The arms are the checkpoint's: those whose gripper pictures it
was trained on.

The pictures come from the live view - the same process that records - so
the pictures the policy sees at rollout are the pictures that were recorded,
cut and resized the same way (`gripper_view`, about each arm camera's own lens
axis - a run does not start on an arm whose lens is not measured): each
camera's newest frame, whole, from shared memory (pinkrab_umi.frame_share),
which the tick copies without waiting for anything. The state is each arm's
own jaw as a fraction of its band and the tools' relative pose, which is what
the dataset carried.

Driving happens in one mode: the gate publishes `TeleopState`, and unless it
says armed and MODE_POLICY nothing is solved or published. The mode begins at
the ready pose, walked to slowly, as the handheld mode does, and then each
arm walks to where its hand began the takes (pinkrab_core.demo_start).

The solve keeps every joint limit_margin_deg inside its band: that is the
stop a policy drives against. The run ends - nothing more published, the gate
holding the last pose - when an arm cannot follow: a step the solve falls
well short of, a policy that goes on asking past a joint's stop, or an arm that
does not get to where it was sent (`Tracking`). A joint at
its stop is no end by itself: the takes, replayed through the arms, hold
wrist_yaw at its stop for long stretches. What ends it is the motion asked
for there and not made, summed while a joint stays at its stop. How fast the
joints turn is the gate's to hold (inference.policy_dps).

A correction (DAgger) is the handhelds taking the arms over mid-run
(MODE_CORRECT) and handing them back: the mode returns to policy straight
from the correction, and the policy goes on from where the hands left the
arms - no walk to ready, its own memory of the last step cleared - which
also lifts a run that had stopped at an arm's reach.

    ros2 run pinkrab_policy rollout --ros-args -p checkpoint:=/path/to/pretrained_model

    python -m pinkrab_policy.rollout --dry-run --checkpoint ...
                                     # no ROS: watch what it would ask for
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from pinkrab_core import inference
from pinkrab_core.config import SETTINGS as CORE
from pinkrab_core.dataset import (LensMissing, delta_to_matrix, gripper_cut, gripper_view,
                                  hands_of, image_key, relative_pose, require_lens, state_names, state_vector)
from pinkrab_core.follow import TOOL_TURN
from pinkrab_core.jaw import aperture_of, gripper_band
from pinkrab_core.kinematics import ArmKinematics
from pinkrab_core.native import quiet_ffmpeg_twins
from pinkrab_core.origin import ready_joints
from pinkrab_umi.config import SETTINGS as UMI
from pinkrab_umi.frame_share import FrameReader


# How the steps' pace is judged: over this long, and slow below this share of
# the dataset's rate.
PACE_WINDOW_S = 2.0
PACE_SLOW = 0.95


class DriveStopped(RuntimeError):
    """The run ends here: an arm cannot follow the policy without leaving the
    reach the takes were made in. The arms hold where they were last put."""


class Tracking:
    """Whether the arms are where the policy last put them.

    Each step is composed onto the pose the one before it asked for, so a step
    taken while the arm is not following - the gate refusing the pose, holding
    the arm at a contact, the readings gone - carries the command further from
    the arm with every tick, and the arm then sets off after it. So no step is
    taken while a joint of the last pose sent reads further from it than
    `lag_deg` or has not been read for `stale_s`; the gate is sent nothing and
    brakes the arm. A wait that lasts `hold_s` ends the run.
    """

    def __init__(self, stale_s: float, lag_deg: float, hold_s: float) -> None:
        self.stale_s, self.lag, self.hold_s = stale_s, math.radians(lag_deg), hold_s
        self.read_at: dict[str, float] = {}
        # The joints of the last pose sent, radians; the jaws are left out, a
        # jaw closed on a block never reads where it was sent.
        self.sent: dict[str, float] = {}
        self.waiting_since: float | None = None

    def read(self, names, now: float) -> None:
        self.read_at.update(dict.fromkeys(names, now))

    def unread(self, names, now: float) -> list[str]:
        """The joints among `names` not read within `stale_s`."""
        return sorted(n for n in names if now - self.read_at.get(n, -math.inf) > self.stale_s)

    def start(self) -> None:
        self.sent, self.waiting_since = {}, None

    def put(self, wanted: dict[str, float]) -> None:
        self.sent = {n: v for n, v in wanted.items() if not n.endswith("_gripper")}

    def wait(self, at: dict[str, float], now: float) -> str | None:
        """Why the next step has to wait, or None. DriveStopped once it has
        waited `hold_s`."""
        why = None
        if quiet := self.unread(self.sent, now):
            why = f"no reading of {', '.join(quiet)}"
        else:
            worst = max(self.sent, key=lambda n: abs(at[n] - self.sent[n]), default=None)
            if worst is not None and abs(at[worst] - self.sent[worst]) > self.lag:
                why = f"{worst} reads {math.degrees(abs(at[worst] - self.sent[worst])):.0f} deg from where it was sent"
        if why is None:
            self.waiting_since = None
            return None
        if self.waiting_since is None:
            self.waiting_since = now
        if now - self.waiting_since > self.hold_s:
            raise DriveStopped(f"the arms do not follow the policy: {why}")
        return why


def checkpoint_state_names(checkpoint: str, dim: int, hands) -> list[str]:
    """What the checkpoint's observation.state holds, by name, from the dataset
    it was trained on. When that dataset is not on this machine, what convert
    writes for these hands, if it is that many numbers; otherwise refused."""
    from pinkrab_core.demo_start import DemoStartError, dataset_of_checkpoint

    try:
        info = json.loads((dataset_of_checkpoint(Path(checkpoint)) / "meta" / "info.json").read_text())
        names = list(info["features"]["observation.state"]["names"])
    except (DemoStartError, OSError, KeyError, ValueError) as exc:
        names = state_names(hands)
        if len(names) != dim:
            raise RuntimeError(f"{checkpoint}: a state of {dim} numbers, and no dataset here that names them") from exc
    if len(names) != dim:
        raise RuntimeError(f"{checkpoint}: a state of {dim} numbers, and its dataset names {len(names)}")
    return names


def carry(desk: dict) -> np.ndarray | None:
    """Where the desk solid of a take stood in the robot's world - the start's
    `carry` on the rig (pinkrab_core.demo_start)."""
    from pinkrab_umi.desk_solid import carried
    return carried(desk)[0]


class Observer:
    """The live view's pictures, cut and resized the way the dataset had them."""

    def __init__(self, sizes: dict[str, tuple[int, int]]) -> None:
        self.sizes = sizes   # key -> (height, width)
        self.hands = hands_of(sizes)
        unknown = sorted(set(sizes) - {image_key(hand) for hand in self.hands})
        if unknown:
            raise ValueError(f"the checkpoint asks for pictures no arm has: {', '.join(unknown)}")
        # Each arm's camera is its own unit, not the handheld's: its picture is
        # cut about its own lens axis, as `pinkrab lens fit` measured it, or the
        # run does not start (LensMissing) - nor for a checkpoint whose
        # pictures were cut some other way (ValueError).
        self.centres = {hand: require_lens(f"follower_{hand}") for hand in self.hands}
        for size in sizes.values():
            gripper_cut(size, UMI.gripper_camera_width, UMI.gripper_camera_height)
        self.frames = {hand: FrameReader(f"follower_{hand}") for hand in self.hands}

    def observe(self) -> dict[str, np.ndarray]:
        """Each arm's picture as the checkpoint learned from it, or RuntimeError: no picture now."""
        import cv2
        out = {}
        for hand in self.hands:
            key = image_key(hand)
            picture = self.frames[hand].read()
            out[key] = cv2.cvtColor(gripper_view(picture, self.sizes[key], self.centres[hand]), cv2.COLOR_BGR2RGB)
        return out


class Brain:
    """The checkpoint alone: pictures and a state in, the deltas out.

    Separate from the arms below it because asking a policy what it wants needs
    no arm - no description, no kinematics, no servo band. The studio's policy
    page asks exactly this and drives nothing, and it would otherwise need a
    URDF to see a number.
    """

    def __init__(self, checkpoint: str, device: str | None = None,
                 task: str | None = None) -> None:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        from lerobot.processor import TokenizerProcessorStep
        from lerobot.utils.device_utils import auto_select_torch_device, is_amp_available
        from lerobot.utils.import_utils import register_third_party_plugins

        # Policy types from other packages are
        # known only once imported, and the config cannot be read before.
        register_third_party_plugins()
        # The rate it is asked at, one step a tick: its dataset's. A step is one
        # of that dataset's frames apart, so any other rate moves the arms
        # slower or faster than the takes did.
        from pinkrab_core.demo_start import DemoStartError, checkpoint_fps
        try:
            self.fps = checkpoint_fps(Path(checkpoint))
        except (DemoStartError, OSError, ValueError, KeyError) as exc:
            raise RuntimeError(f"{checkpoint}: no rate to ask it at - its dataset says it, and {exc}") from exc
        self.period = 1.0 / self.fps
        # How Pinkrab drives with any policy here - the device, when a step is
        # a bad answer, when the run ends - read once, for this run. (The
        # gate's speed is the gate's: it read it when it started.)
        self.common = inference.common()
        # The device: the one asked for, else the rig's choice, else the best
        # here - CUDA, Apple's Metal, the CPU - named before the config is
        # read: a checkpoint trained on CUDA says cuda, and the config would
        # otherwise warn that it is switching.
        if device is None and self.common["device"] != "auto":
            device = self.common["device"]
        device = device or auto_select_torch_device().type
        cfg = PreTrainedConfig.from_pretrained(checkpoint, cli_overrides=[f"--device={device}"])
        self.use_amp = bool(self.common["use_amp"]) and is_amp_available(cfg.device)
        # How it is asked - the rig's inference.yaml for this policy type -
        # set before the policy is built from the config, which is where it
        # reads them: ACT sizes its action queue and its ensembler there.
        # Kept with the common ones this run uses, for the page - not the
        # gate's speed, which this process does not set.
        used = {k: v for k, v in self.common.items() if k != "max_joint_dps"}
        asked = inference.apply(cfg, self.fps)
        # Not with an ensembler: it sums every chunk in the answer's own dtype,
        # and half precision overflows long before the weights' float32 does.
        if getattr(cfg, "temporal_ensemble_coeff", None) is not None:
            self.use_amp = False
        self.settings = {**asked, **used, "device": cfg.device, "use_amp": self.use_amp, "fps": self.fps}
        self.policy = get_policy_class(cfg.type).from_pretrained(checkpoint, config=cfg)
        self.policy.eval()
        self.policy.reset()
        self.device = next(self.policy.parameters()).device
        self.torch = torch
        # Normalisation is not in the policy. LeRobot keeps it in two pipelines
        # saved beside the weights - the statistics of the very dataset this was
        # fit to - and `select_action` is the middle of the sandwich: it takes
        # normalised input and answers in normalised units. Calling it alone
        # returns numbers that look like metres and are not, which is a tool
        # asked for half a metre a tick rather than four millimetres.
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        # The images the policy wants, by key: (height, width) - one per hand.
        self.sizes = {k: (f.shape[1], f.shape[2]) for k, f in self.policy.config.input_features.items()
                      if f.type.name == "VISUAL"}
        self.hands = hands_of(self.sizes)
        if not self.hands:
            raise RuntimeError(f"{checkpoint}: shown no gripper picture of either hand ({', '.join(self.sizes)})")
        self.state_dim = next((f.shape[0] for k, f in self.policy.config.input_features.items()
                               if k == "observation.state"), 10 * len(self.hands))
        self.state_names = checkpoint_state_names(checkpoint, self.state_dim, self.hands)
        # The command, for a policy that reads one - one whose pipeline
        # tokenises the task. Unsaid, it is its dataset's, when that dataset
        # held one; a policy trained on several is told which. A policy with
        # no language input is shown none.
        from pinkrab_core.demo_start import checkpoint_tasks
        self.tasks = checkpoint_tasks(Path(checkpoint))
        self.language = any(isinstance(step, TokenizerProcessorStep) for step in self.preprocessor.steps)
        if not self.language:
            task = None
        elif task is None and len(self.tasks) == 1:
            task = self.tasks[0]
        elif task is None:
            raise RuntimeError(f"{checkpoint} reads a command, and its dataset "
                               + (f"holds {len(self.tasks)} - say which" if self.tasks else "is not here to say it"))
        self.task = task

    def reset(self) -> None:
        self.policy.reset()

    def ask(self, images: dict[str, np.ndarray], state: np.ndarray, task: str | None = None) -> np.ndarray:
        """One answer - ten numbers a hand - from what the cameras see and the
        state (`state_vector` of `state_names`): each jaw as a fraction of its
        band, and where one tool stands from the other. Nothing about where a
        tool is - the dataset carried none, so the policy was never shown any.
        `task` stands in for the command this once, for a policy that reads one.
        """
        from lerobot.policies.utils import prepare_observation_for_inference

        # The pictures go in as the recorder had them - (H, W, C), 0..255 - and
        # come out of here batched, channel first and in 0..1. Done by lerobot's
        # own helper so this and `lerobot-record` cannot drift apart on what
        # "an observation" means.
        observation = prepare_observation_for_inference(
            {"observation.state": state, **images}, self.device,
            task=(task if task is not None else self.task) if self.language else None,
        )
        with self.torch.inference_mode(), self._precision():
            action = self.postprocessor(self.policy.select_action(self.preprocessor(observation)))
        # Back to float32 first: under mixed precision it may come out half or
        # bfloat16, and numpy has no bfloat16.
        return action[0].float().cpu().numpy()

    def _precision(self):
        return self.torch.autocast(device_type=self.device.type) if self.use_amp else contextlib.nullcontext()


class _Arm:
    """One arm under the policy: its solve, its stops, where its tool was put.
    `fps` is how often it is stepped - the policy's dataset's rate."""

    def __init__(self, arm: str, common: dict, fps: float) -> None:
        self.arm = arm
        self.common = common
        self.fps = fps
        self.chain = ArmKinematics(arm)
        self.lower, self.upper = self.chain.limits
        self.chain.keep_off_stops(common["limit_margin_deg"])
        self.band = gripper_band(arm)
        self.turn = np.eye(4)
        self.turn[:3, :3] = TOOL_TURN[arm]
        self.joints: np.ndarray | None = None
        self.T_cmd: np.ndarray | None = None
        self.held = 0
        # What the policy asked for and the arm did not make, summed while a
        # joint is at its stop: mm in x, y, z, then degrees about them.
        self.pushed = np.zeros(6)

    def start(self, joints_deg) -> None:
        self.joints = np.asarray(joints_deg, dtype=float)
        self.T_cmd = self.chain.forward(self.joints)
        self.pushed[:] = 0.0

    def tool(self) -> np.ndarray:
        """Where the tool was put, in the handheld's convention."""
        return self.T_cmd @ self.turn

    def _room(self, joints: np.ndarray) -> dict[str, float]:
        """How far each joint is from the nearer of its stops, degrees."""
        names = [n.removeprefix(f"{self.arm}_") for n in self.chain.joint_names]
        return {n: float(min(v - lo, hi - v)) for n, v, lo, hi in zip(names, joints, self.lower, self.upper)}

    @staticmethod
    def _shortfall(target: np.ndarray, reached: np.ndarray) -> np.ndarray:
        """What `target` asks beyond `reached`, in the base frame: mm along x,
        y, z, then degrees about them."""
        turn = Rotation.from_matrix(reached[:3, :3].T @ target[:3, :3]).as_rotvec()
        return np.concatenate([(target[:3, 3] - reached[:3, 3]) * 1000.0,
                               np.degrees(reached[:3, :3] @ turn)])

    def _check(self, joints: np.ndarray, target: np.ndarray) -> None:
        """Raise DriveStopped when this tick's solve is not a policy being followed.

        How fast the joints turn is not asked here: the takes themselves,
        replayed through the arm, turn a joint as fast as one that has
        flipped. The gate holds the speed (inference.policy_dps). The slip
        catches a single step the solve could not close. A policy pushing past
        a joint's stop falls short a little each tick instead - the next
        target is composed onto where the arm went - so while a joint is at
        its stop what each tick did not make is summed, as vectors: a policy
        asking past the stop adds up, a solve's jitter cancels out. Past the
        slip's own bounds, the run ends; the sum is let go once every joint is
        stop_release_deg clear of its stop - not the tick one lifts off it.
        """
        short = self._shortfall(target, self.chain.forward(joints))
        slip_mm, slip_deg = float(np.linalg.norm(short[:3])), float(np.linalg.norm(short[3:]))
        limit_mm, limit_deg = self.common["slip_mm"], self.common["slip_deg"]
        if slip_mm > limit_mm or slip_deg > limit_deg:
            raise DriveStopped(f"the {self.arm} arm falls {slip_mm:.0f} mm / {slip_deg:.0f} deg short of "
                               "the pose asked for")
        margin = self.common["limit_margin_deg"]
        at_stop = [name for name, room in self._room(joints).items() if room <= margin + CORE.stop_release_deg]
        if not at_stop:
            self.pushed[:] = 0.0
            return
        self.pushed += short
        pushed_mm, pushed_deg = float(np.linalg.norm(self.pushed[:3])), float(np.linalg.norm(self.pushed[3:]))
        if pushed_mm > limit_mm or pushed_deg > limit_deg:
            raise DriveStopped(f"{self.arm} {', '.join(at_stop)} at its stop, and the policy asks past it: "
                               f"{pushed_mm:.0f} mm / {pushed_deg:.0f} deg not made")

    def step(self, action: np.ndarray) -> tuple[dict[str, float], dict]:
        """This arm's ten numbers of an answer into its joints and its gripper."""
        info: dict = {"action": action.tolist()}
        dp_mm = float(np.linalg.norm(action[:3]) * 1000.0)
        delta = delta_to_matrix(action)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(delta[:3, :3]) - 1.0) / 2.0))))
        if (dp_mm * self.fps > self.common["max_tool_speed_mm_s"]
                or angle * self.fps > self.common["max_tool_turn_dps"]):
            self.held += 1
            info["held"] = f"step of {dp_mm:.0f} mm / {angle:.0f} deg refused"
            delta = np.eye(4)
        # The delta is in the handheld's tool convention; this arm's tool frame
        # may be that turned half a turn, and a motion said in a turned frame
        # is conjugated into it.
        delta_here = self.turn @ delta @ self.turn
        target = self.T_cmd @ delta_here
        try:
            joints = self.chain.track(target, self.joints, orientation_weight=self.common["orientation_weight"],
                                      rate_hz=self.fps)
        except Exception as exc:  # noqa: BLE001 - a solve that cannot be had is a stop, not a dead node
            # Seeded from a joint further outside the band the solve keeps than
            # one solve can bring it back - a limit_margin_deg past what a solve
            # turns, an arm handed back deep in its stop - the QP has no answer.
            raise DriveStopped(f"the solve found no pose for the {self.arm} arm to go to ({exc})") from exc
        self._check(joints, target)
        # The next delta goes on from where the arm was put, not from where it
        # was asked to be: a target composed onto targets it never reached
        # walks off out of reach, and the solver swings the arm after it.
        self.T_cmd = self.chain.forward(joints)
        self.joints = joints
        wanted = {name: float(np.deg2rad(v)) for name, v in zip(self.chain.joint_names, joints)}
        opening = float(min(1.0, max(0.0, action[9])))
        wanted[f"{self.arm}_gripper"] = self.band[0] + opening * (self.band[1] - self.band[0])
        info["dp_mm"] = dp_mm
        info["opening"] = opening
        return wanted, info


class Driver:
    """The policy and its arms, without ROS: pictures in, joint angles out."""

    def __init__(self, checkpoint: str, device: str | None = None, task: str | None = None) -> None:
        self.brain = Brain(checkpoint, device, task)
        self.hands = self.brain.hands
        self.arms = {hand: _Arm(hand, self.brain.common, self.brain.fps) for hand in self.hands}
        self.steps = 0

    # What the arm half needs from the policy half, named where it was before.
    @property
    def policy(self):
        return self.brain.policy

    @property
    def device(self):
        return self.brain.device

    @property
    def sizes(self) -> dict[str, tuple[int, int]]:
        return self.brain.sizes

    @property
    def state_dim(self) -> int:
        return self.brain.state_dim

    @property
    def chains(self) -> dict[str, ArmKinematics]:
        return {hand: arm.chain for hand, arm in self.arms.items()}

    def start(self, joints_deg: dict[str, np.ndarray]) -> None:
        """Where the arms are now is where the policy's first deltas are from."""
        for hand, arm in self.arms.items():
            arm.start(joints_deg[hand])
        self.brain.reset()
        self.steps = 0

    def state(self, grippers_rad: dict[str, float]) -> tuple[np.ndarray, dict[str, float]]:
        """observation.state as the dataset had it, from each jaw's reading
        and where each tool was put; and the openings it holds."""
        apertures = {hand: aperture_of(hand, grippers_rad[hand], arm.band) for hand, arm in self.arms.items()}
        relative = (relative_pose(self.arms[self.hands[0]].tool(), self.arms[self.hands[1]].tool())
                    if len(self.hands) == 2 else None)
        return state_vector(self.brain.state_names, apertures, relative), apertures

    def step(self, images: dict[str, np.ndarray], grippers_rad: dict[str, float]) -> tuple[dict[str, float], dict]:
        state, apertures = self.state(grippers_rad)
        action = self.brain.ask(images, state)
        # An answer that is not a number ends the run: NaN passes every limit
        # below - each comparison with it is false - and a solve fed one is
        # joints of NaN.
        if not np.isfinite(action).all():
            raise DriveStopped("the policy answered with a number that is not one (NaN or infinite)")
        if len(action) != 10 * len(self.hands):
            raise DriveStopped(f"the policy answered {len(action)} numbers for {len(self.hands)} hand(s)")
        wanted, info = {}, {"action": action.tolist(), "arms": {}}
        for i, (hand, arm) in enumerate(self.arms.items()):
            joints, said = arm.step(action[10 * i:10 * i + 10])
            wanted.update(joints)
            info["arms"][hand] = {**said, "aperture": apertures[hand]}
        self.steps += 1
        return wanted, info


def dry_run(args) -> int:
    """No ROS, no robot: the policy against the live view, joint angles printed."""
    quiet_ffmpeg_twins()
    try:
        driver = Driver(args.checkpoint, args.device, args.task)
        observer = Observer(driver.sizes)
    except (LensMissing, ValueError, RuntimeError) as exc:
        print(f"rollout: {exc}", file=sys.stderr)
        return 1
    driver.start({hand: ready_joints(hand, chain.joint_names) for hand, chain in driver.chains.items()})
    print(f"  policy {driver.policy.config.type} on {driver.device}, arms {', '.join(driver.hands)}, "
          f"images {driver.sizes}, state {driver.state_dim}")
    print(f"  from the ready pose, {args.seconds:.0f} s at {driver.brain.fps} Hz, against the live view")
    started = time.monotonic()
    period = driver.brain.period
    while time.monotonic() - started < args.seconds:
        tick = time.monotonic()
        try:
            images = observer.observe()
        except RuntimeError as exc:
            print(f"  no picture: {exc}")
            time.sleep(1.0)
            continue
        try:
            wanted, info = driver.step(images, {hand: arm.band[0] for hand, arm in driver.arms.items()})
        except DriveStopped as exc:
            print(f"  stopped: {exc}")
            return 1
        line = []
        for hand, arm in driver.arms.items():
            said = info["arms"][hand]
            tool = np.round(arm.T_cmd[:3, 3] * 1000).astype(int)
            line.append(f"{hand} dp {said['dp_mm']:5.1f} mm opening {said['opening']:.2f} tool {tool} mm "
                        f"{said.get('held', '')}")
        print(f"  {time.monotonic() - started:5.2f}s  " + "  |  ".join(line))
        time.sleep(max(0.0, period - (time.monotonic() - tick)))
    return 0


def node_main() -> None:
    quiet_ffmpeg_twins()    # one thread yet: see pinkrab_core.native
    import rclpy
    from pinkrab_interfaces.msg import TeleopState
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    from pinkrab_core.origin import ready_pose
    from pinkrab_core.parent import watch_parent
    from pinkrab_core.walk import Approach

    class RolloutNode(Node):
        def __init__(self) -> None:
            super().__init__("pinkrab_policy")
            self.declare_parameter("checkpoint", "")
            self.declare_parameter("device", "")
            # The command, for a policy that reads one. Empty is its dataset's.
            self.declare_parameter("task", "")
            # Where each step is reported, so the page that asked for the drive
            # can draw what the policy asked for. Empty means nobody is told;
            # the log line every two seconds is then all there is.
            self.declare_parameter("studio", "")
            # The studio that started this node, by pid: gone, and so is this.
            self.declare_parameter("parent", 0)
            watch_parent(int(self.get_parameter("parent").value))
            checkpoint = self.get_parameter("checkpoint").value
            self.checkpoint = checkpoint
            if not checkpoint:
                raise SystemExit("rollout: -p checkpoint:=<dir with config.json and model.safetensors> is required")
            try:
                # Ticking at the checkpoint's dataset's rate, one step a tick.
                self.driver = Driver(checkpoint, self.get_parameter("device").value or None,
                                     self.get_parameter("task").value or None)
            except (inference.InferenceError, RuntimeError) as exc:
                raise SystemExit(f"rollout: {exc}") from exc
            # Set when the checkpoint's dataset has changed since it learned:
            # no start can be trusted, so nothing is driven - known here, so
            # the arms are not even walked to ready first.
            self.refused = ""
            from pinkrab_core.demo_start import (
                DatasetChanged,
                DemoStartError,
                dataset_of_checkpoint,
            )
            try:
                dataset_of_checkpoint(Path(checkpoint))
            except DatasetChanged as exc:
                self.refused = str(exc)
                self.get_logger().error(f"not driving: {exc}")
            except (DemoStartError, OSError, ValueError):
                pass    # no dataset to check here; the demo start says so when it is asked
            try:
                self.observer = Observer(self.driver.sizes)
            except (LensMissing, ValueError) as exc:
                raise SystemExit(f"rollout: {exc}") from exc
            self.studio = str(self.get_parameter("studio").value).rstrip("/")
            self.commands = self.create_publisher(JointState, "joint_commands", 10)
            self.create_subscription(TeleopState, "teleop_state", self._on_teleop, 10)
            self.create_subscription(JointState, "joint_states", self._on_state, 10)
            self.at: dict[str, float] = {}
            self.driving = False
            # The mode before this one: policy straight after a correction is
            # a hand-back, not a new start.
            self.last_mode: int | None = None
            # Why the run stopped at an arm's reach, until a correction lifts it.
            self.halted = ""
            self.tracking = Tracking(CORE.reading_stale_s, CORE.policy_follow_lag_deg, CORE.policy_follow_hold_s)
            self.approach: Approach | None = None
            self.at_start = False      # ready reached, the demo start is the leg after
            self.started = False
            self._said = 0.0
            self.period = self.driver.brain.period
            # How fast the steps have gone since the last look: (monotonic, steps).
            self._pace = (0.0, 0)
            # Reports go out from a thread of their own: a POST in the tick
            # would be paid for by the arms, every tick.
            self._reports: queue.Queue = queue.Queue(maxsize=8)
            threading.Thread(target=self._send_reports, name="policy-reports", daemon=True).start()
            self.create_timer(self.period, self._tick)
            self.get_logger().info(
                f"policy {self.driver.policy.config.type} on {self.driver.device} drives the "
                f"{' and '.join(self.driver.hands)} arm(s) in policy mode, {self.driver.brain.fps} Hz, "
                "pictures from the live view's shared frames"
                + (f", told {self.driver.brain.task!r}" if self.driver.brain.task else ""))

        def _demo_start_leg(self) -> dict[str, float] | None:
            """The joints that put each tool where its hand began the takes, or None."""
            from pinkrab_core.demo_start import DatasetChanged, dataset_of_checkpoint, start_joints
            try:
                joints, info = start_joints(dataset_of_checkpoint(Path(self.checkpoint)), self.driver.chains, carry)
            except DatasetChanged as exc:
                self.refused = str(exc)
                self.get_logger().error(f"not driving: {exc}")
                self._report({"phase": "refused", "reason": self.refused})
                return None
            except (RuntimeError, OSError, ValueError) as exc:   # the solver's own failures too
                self.get_logger().warn(f"no demo start - driving from ready: {exc}")
                return None
            for hand, found in info["arms"].items():
                self.get_logger().info(
                    f"walking the {hand} arm to the demo start: tool {found['pitch_deg']:.0f} deg down, "
                    f"{found['height_m'] * 1000:.0f} mm above the desk, {found['margin_deg']:.0f} deg from "
                    f"the nearest stop ({info['episodes']} episodes)")
            return {name: math.radians(float(v)) for hand, chain in self.driver.chains.items()
                    for name, v in zip(chain.joint_names, joints[hand])}

        def _on_state(self, message: JointState) -> None:
            if message.name and len(message.name) == len(message.position):
                self.at.update(dict(zip(message.name, (float(v) for v in message.position))))
                self.tracking.read(message.name, time.monotonic())

        def _on_teleop(self, message: TeleopState) -> None:
            mode = int(message.mode) if message.armed else TeleopState.MODE_IDLE
            before, self.last_mode = self.last_mode, mode
            driving = mode == TeleopState.MODE_POLICY
            if driving == self.driving:
                return
            self.driving = driving
            if driving and before == TeleopState.MODE_CORRECT:
                # Handed back: on from where the hands left the arms.
                self.approach = None
                self.at_start = True
                self.started = False
                self.halted = ""
                self.get_logger().info("handed back - the policy goes on from where the arms are")
            elif driving:
                self.approach = Approach(ready_pose(), self.driver.brain.common["ready_tol_deg"])
                self.at_start = False
                self.started = False
                self.halted = ""   # a new run is not the one that stopped
                self.get_logger().info(f"policy mode: walking to the ready pose first, {CORE.park_speed_dps:g} deg/s")
            else:
                self.approach = None
                self.started = False
                self.get_logger().info("not driving - the arms hold where they are")

        def _tick(self) -> None:
            if not self.driving:
                return
            if self.refused or self.halted:
                now = time.monotonic()
                if now - self._said > 2.0:
                    # Again every two seconds: a report can be dropped, and the
                    # page should not go on saying the node is starting.
                    self._said = now
                    self.get_logger().error(f"not driving: {self.refused or self.halted}")
                    self._report({"phase": "refused", "reason": self.refused or self.halted})
                return
            if self.approach is not None:
                command = self.approach.step(self.at, time.monotonic())
                if self.approach.arrived:
                    self.approach = None
                    if not self.at_start:
                        # Ready reached; now where the takes began, so the
                        # first picture the policy sees is one it was shown.
                        self.at_start = True
                        leg = self._demo_start_leg()
                        if self.refused:
                            return
                        if leg is not None:
                            self.approach = Approach(leg, self.driver.brain.common["ready_tol_deg"])
                            return
                    self.get_logger().info("at the start - the policy takes over")
                    return
                to_go = self.approach.to_go_deg(self.at)
                self.get_logger().info(f"walking to the {'start' if self.at_start else 'ready pose'}: {to_go:.0f} deg to go",
                                       throttle_duration_sec=1.0)
                self._report({"phase": "walking", "to_go_deg": round(float(to_go), 1)})
                if command:
                    self._publish(command)
                return
            now = time.monotonic()
            if not self.started:
                # From the arms as they read, every joint of them: a joint
                # that has not been read is not at zero.
                names = [n for chain in self.driver.chains.values() for n in chain.joint_names]
                if quiet := self.tracking.unread(names, now):
                    self.get_logger().info(f"waiting for a reading of {', '.join(quiet)}",
                                           throttle_duration_sec=1.0)
                    return
                self.driver.start({hand: [math.degrees(self.at[n]) for n in chain.joint_names]
                                   for hand, chain in self.driver.chains.items()})
                self.tracking.start()
                self.started = True
                self._pace = (now, self.driver.steps)
            try:
                why = self.tracking.wait(self.at, now)
            except DriveStopped as exc:
                self._halt(exc)
                return
            if why is not None:
                # Nothing is published and no step is taken: the gate brakes the
                # arm, and the next step is composed onto the pose last put.
                self.get_logger().info(f"holding - {why}", throttle_duration_sec=1.0)
                return
            began = time.monotonic()
            try:
                images = self.observer.observe()
            except RuntimeError as exc:
                self.get_logger().info(f"{exc} - holding", throttle_duration_sec=1.0)
                return
            grippers = {hand: self.at.get(f"{hand}_gripper", arm.band[0]) for hand, arm in self.driver.arms.items()}
            try:
                wanted, info = self.driver.step(images, grippers)
            except DriveStopped as exc:
                self._halt(exc)
                return
            self.tracking.put(wanted)
            self._publish(wanted)
            # Each tick applies one of the dataset's frame steps, so steps
            # slower than its rate are arms slower than the takes. Said from
            # the rate over a couple of seconds: one tick that waits on its
            # picture or a chunk is made up by the ticks after it.
            now = time.monotonic()
            since, steps = self._pace
            if now - since >= PACE_WINDOW_S:
                rate = (self.driver.steps - steps) / (now - since)
                if rate < PACE_SLOW * self.driver.brain.fps:
                    self.get_logger().warn(
                        f"stepping at {rate:.0f} Hz of the takes' {self.driver.brain.fps} - "
                        f"the arms move slower than the takes did (this step took {(now - began) * 1000:.0f} ms)")
                self._pace = (now, self.driver.steps)
            held = {hand: said["held"] for hand, said in info["arms"].items() if "held" in said}
            self._report({"phase": "driving", "step": self.driver.steps, "arms": info["arms"]})
            now = time.monotonic()
            if now - self._said > 2.0 or held:
                self._said = now
                self.get_logger().info(
                    f"step {self.driver.steps}: " + ", ".join(
                        f"{hand} dp {said['dp_mm']:.1f} mm opening {said['opening']:.2f}"
                        + (f" - {said['held']}" if "held" in said else "") for hand, said in info["arms"].items()))

        def _halt(self, exc: DriveStopped) -> None:
            # Nothing more is published: the gate holds the last pose it
            # passed - shaped, it starts braking once the commands have
            # stopped for a tenth of a second, and the arm stands short of
            # the last one passed. Said again every two seconds by the
            # branch above, until a correction hands the arm back.
            self.halted = f"stopped - {exc}"
            self.get_logger().error(f"not driving: {self.halted}")
            self._report({"phase": "refused", "reason": self.halted})

        def _report(self, body: dict) -> None:
            """One line to the studio, handed to the reporting thread.

            The page draws it; nothing here depends on it arriving. When the
            thread is behind, the line is dropped rather than waited for."""
            if not self.studio:
                return
            try:
                # With how the policy is being asked, so the page says what
                # this run uses rather than what the file says now.
                self._reports.put_nowait({"hands": list(self.driver.hands), "settings": self.driver.brain.settings,
                                          **body})
            except queue.Full:
                pass

        def _send_reports(self) -> None:
            while True:
                body = self._reports.get()
                request = urllib.request.Request(
                    f"{self.studio}/policy/step", data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"})
                try:
                    urllib.request.urlopen(request, timeout=0.2).close()
                except (urllib.error.URLError, OSError) as exc:
                    self.get_logger().info(f"studio not answering ({exc})", throttle_duration_sec=5.0)

        def _publish(self, wanted: dict[str, float]) -> None:
            message = JointState()
            message.header.stamp = self.get_clock().now().to_msg()
            message.name = sorted(wanted)
            message.position = [wanted[n] for n in message.name]
            self.commands.publish(message)

    CORE.pin_ros_middleware()
    rclpy.init()
    node = RolloutNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--dry-run" in argv:
        parser = argparse.ArgumentParser(description="the policy against the live view, without ROS")
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--checkpoint", required=True)
        parser.add_argument("--device", default=None)
        parser.add_argument("--task", default=None, help="the command, for a policy that reads one")
        parser.add_argument("--seconds", type=float, default=10.0)
        return dry_run(parser.parse_args(argv))
    node_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
