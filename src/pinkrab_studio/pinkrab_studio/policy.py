"""A trained policy, answering about what the cameras see right now.

Between a finished training run and an arm that moves there is one question
worth asking on its own: given this picture, what does the policy want to do?
Nothing here drives anything. The cameras are the rig's, the pictures are the
ones the recorder would have kept, and the answer is a row of numbers per
arm on a page - so a checkpoint can be judged before a servo is ever armed.

The observation is built the way `pinkrab_policy.rollout` builds it, from the
same live view and through the same resize, because a policy shown a picture
made some other way is being asked a different question than the one it was
trained on. What differs is the jaws and the tools. On the arm the aperture
in the state is the gripper's own, measured, and the tools stand where the
arms put them; nothing is driven here, so the policy's last commanded
opening is fed back in its place, and each tool is where the policy's deltas
would have taken it from where its arm stood when the asking began - which is
also where the tools stand from each other. That makes this a free-running
answer: after the first step the state is the policy's own, and a checkpoint
that drifts drifts here too rather than being carried by the arms.

The deltas are composed as rollout composes them, so the page can draw where
each tool would have gone. It is where the policy would have taken it from
wherever it started, not a claim about the room.
"""
from __future__ import annotations

import collections
import json
import math
import threading
import time
from pathlib import Path

import numpy as np

from pinkrab_core.config import SETTINGS as CORE

from .alerts import ALERTS

# The page draws this many steps of history; the rest is dropped as it ages.
HISTORY = 600


def checkpoint_hands(checkpoint: str | Path) -> tuple[str, ...]:
    """The arms a checkpoint drives - those whose gripper pictures it was
    trained on - read off its config, not its weights; PolicyError for one
    shown neither."""
    from pinkrab_core.dataset import hands_of

    try:
        features = json.loads((Path(checkpoint) / "config.json").read_text()).get("input_features") or {}
    except (OSError, ValueError) as exc:
        raise PolicyError(f"{checkpoint} is not a checkpoint - {exc}") from exc
    hands = hands_of(features)
    if not hands:
        raise PolicyError(f"{checkpoint} is shown no gripper picture of either arm")
    return hands


def _need_lens(hands) -> None:
    """PolicyError unless each arm's gripper camera has its lens measured: a
    policy is shown that camera's picture cut about its own lens axis."""
    from pinkrab_core.dataset import LensMissing, require_lens

    try:
        for arm in hands:
            require_lens(f"follower_{arm}")
    except LensMissing as exc:
        raise PolicyError(str(exc)) from exc


def checkpoints() -> list[dict]:
    """Every checkpoint under the workspace's train directory, newest first.

    A run keeps checkpoints/<step>/pretrained_model; that inner directory is
    what `from_pretrained` takes, so it is what is listed, named for the run
    and the step it came from.
    """
    from pinkrab_core.dataset import hands_of
    from pinkrab_core.demo_start import checkpoint_tasks

    root = CORE.train_dir
    found: list[dict] = []
    if not root.is_dir():
        return found
    for run in sorted(p for p in root.iterdir() if p.is_dir()):
        for step in sorted((run / "checkpoints").glob("*/pretrained_model")):
            if not (step / "config.json").is_file():
                continue
            # `last` is a symlink to one of the others; listing both would
            # offer the same checkpoint twice under two names.
            if step.parent.is_symlink():
                continue
            # What it was trained to look at, so the page can say which camera
            # a run cannot start without. Read from the config, not the
            # weights: the weights take seconds and this is a listing.
            try:
                features = json.loads((step / "config.json").read_text()).get("input_features") or {}
            except (OSError, ValueError):
                features = {}
            command = reads_command(step)
            found.append({
                "path": str(step),
                "run": run.name,
                "step": step.parent.name,
                "at": step.stat().st_mtime,
                "images": [k for k in features if k.startswith("observation.images.")],
                "hands": list(hands_of(features)),
                # Whether it is told what to do, and what its dataset told it.
                "command": command,
                "tasks": checkpoint_tasks(step) if command else [],
            })
    found.sort(key=lambda c: c["at"], reverse=True)
    return found


def reads_command(checkpoint: Path) -> bool:
    """Whether a checkpoint is told what to do: its input pipeline tokenises
    the task, as a VLA's does. Read from the saved pipeline, not the weights."""
    try:
        steps = json.loads((checkpoint / "policy_preprocessor.json").read_text()).get("steps") or []
    except (OSError, ValueError):
        return False
    return any(step.get("registry_name") == "tokenizer_processor" for step in steps)


class PolicyError(RuntimeError):
    """Something the page asked for that cannot be done."""


class PolicyRun:
    """One checkpoint, asked about the live pictures until it is stopped.

    Held here rather than in a process of its own: the weights take seconds to
    reach the GPU and a page that restarted them per question would spend its
    time loading. One at a time - a second checkpoint would want the same
    cameras and the same GPU.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._brain = None
        self._history: collections.deque = collections.deque(maxlen=HISTORY)
        self._checkpoint = ""
        self._task = ""
        self._hands: tuple[str, ...] = ()
        self._steps = 0
        self._wild = 0
        self._note = ""
        self._loading = False
        self._hz = 0.0
        self._latency_ms = 0.0
        # Each tool where the policy's deltas have taken it, from its start.
        self._poses: dict[str, np.ndarray] = {}
        # How the policy is being asked in this run (pinkrab_core.inference):
        # the asking loop's own, or what the rollout node reports.
        self._settings: dict | None = None
        # Driving: the rollout node asks the policy and moves the arm, and
        # reports each step here. Nothing is asked in this process then; the
        # history is what the node said.
        self._driving = False
        self._phase = ""
        self._to_go_deg: float | None = None
        self._refused = ""
        self._beat = 0.0

    # --- what the page sees ----------------------------------------------

    def state(self) -> dict:
        with self._lock:
            running = self._thread is not None and self._thread.is_alive()
            last = self._history[-1] if self._history else None
            return {
                "running": running or self._driving,
                "driving": self._driving,
                "phase": self._phase,
                "to_go_deg": self._to_go_deg,
                "refused": self._refused,
                "loading": self._loading,
                "checkpoint": self._checkpoint,
                "task": self._task,
                "hands": list(self._hands),
                # How often it is asked: its dataset's rate (Brain.fps), as
                # the run says it - none before it has loaded.
                "rate": (self._settings or {}).get("fps"),
                "steps": self._steps,
                "wild": self._wild,
                "note": self._note,
                "hz": round(self._hz, 2),
                "latency_ms": round(self._latency_ms, 1),
                "last": last,
                "settings": self._settings,
            }

    @property
    def phase(self) -> str:
        """The rollout node's phase as it last reported, while it drives."""
        with self._lock:
            return self._phase if self._driving else ""

    def history(self, since: int = 0) -> dict:
        """The steps the page has not drawn yet, by step number."""
        with self._lock:
            rows = [row for row in self._history if row["step"] > since]
            return {"steps": rows, "latest": self._steps}

    # --- start and stop ---------------------------------------------------

    def start(self, checkpoint: str, joints_now, task: str = "") -> None:
        """Ask `checkpoint` about the live pictures until stopped. `joints_now()`
        is where the arms stand, radians by joint name - the tools' start.
        `task` is the command, for a checkpoint told one; empty is its dataset's."""
        path = Path(checkpoint)
        if not (path / "config.json").is_file():
            raise PolicyError(f"{path} is not a checkpoint - no config.json in it")
        hands = checkpoint_hands(path)
        _need_lens(hands)
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise PolicyError("a policy is already answering - stop it first")
            if self._driving:
                raise PolicyError("the policy is driving the arm - stop that first")
            self._stop.clear()
            self._history.clear()
            self._checkpoint = str(path)
            self._task = task
            self._hands = hands
            self._steps = 0
            self._wild = 0
            self._note = ""
            self._settings = None
            self._loading = True
            self._hz = 0.0
            self._poses = {}
            self._thread = threading.Thread(target=self._run, args=(joints_now,), name="policy", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            thread = self._thread
        if thread is None or not thread.is_alive():
            raise PolicyError("no policy is running")
        self._stop.set()
        thread.join(timeout=5.0)

    def stop_quietly(self) -> None:
        """For the server coming down: no complaint when nothing is running."""
        self._stop.set()
        with self._lock:
            thread = self._thread
            self._driving = False
        if thread is not None:
            thread.join(timeout=5.0)

    # --- driven by the node -------------------------------------------------

    def begin_driven(self, checkpoint: str, task: str = "") -> tuple[str, ...]:
        """The rollout node has the policy and the arms; this keeps the record.
        Returns the arms it drives."""
        hands = checkpoint_hands(checkpoint)
        _need_lens(hands)
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise PolicyError("a policy is answering here - stop it before driving")
            self._history.clear()
            self._checkpoint = str(checkpoint)
            self._task = task
            self._hands = hands
            self._steps = 0
            self._wild = 0
            self._note = ""
            self._hz = 0.0
            self._latency_ms = 0.0
            self._poses = {hand: np.eye(4) for hand in hands}
            self._driving = True
            self._settings = None
            self._phase = "starting"
            self._to_go_deg = None
            self._refused = ""
            self._beat = 0.0
        return hands

    def end_driven(self) -> None:
        with self._lock:
            self._driving = False
            self._phase = ""
            self._to_go_deg = None
            self._refused = ""

    def record(self, report: dict) -> None:
        """One step as the node reported it, into the same history the page draws.

        The node's numbers are the truth of what was asked of the arms; the
        composed poses here are the same bookkeeping the asking loop does, so
        the path panel means the same thing either way.
        """
        with self._lock:
            if not self._driving:
                return
            if isinstance(report.get("settings"), dict):
                self._settings = report["settings"]
            phase = str(report.get("phase", ""))
            self._phase = phase
            if phase == "refused":
                reason = str(report.get("reason", ""))
                if reason != self._refused:    # the node says it each tick; once is news
                    ALERTS.say("policy", f"not driving - {reason}")
                self._refused = reason
                return
            # Not refused now: a refusal after this is news again.
            self._refused = ""
            if phase != "driving":
                to_go = report.get("to_go_deg")
                self._to_go_deg = float(to_go) if isinstance(to_go, (int, float)) else None
                return
            self._to_go_deg = None
            said = report.get("arms") or {}
            if set(said) != set(self._hands):
                return
            limits = self._settings or {}
            arms = {}
            for hand in self._hands:
                action = np.asarray(said[hand].get("action") or [], dtype=np.float32)
                if action.shape != (10,):
                    return
                held = str(said[hand].get("held") or "")
                arms[hand] = self._arm_row(hand, action, float(said[hand].get("aperture") or 0.0), held, limits)
            self._steps += 1
            wild = any(row["wild"] for row in arms.values())
            if wild:
                self._wild += 1
            now = time.monotonic()
            if self._beat:
                gap = now - self._beat
                rate = 1.0 / gap if gap > 1e-6 else 0.0
                self._hz = rate if self._hz == 0.0 else self._hz * 0.8 + rate * 0.2
            self._beat = now
            self._note = "; ".join(f"{hand}: {said[hand]['held']}" for hand in self._hands if said[hand].get("held"))
            self._history.append({"step": self._steps, "at": time.time(), "arms": arms, "wild": wild,
                                  "latency_ms": 0.0})

    def _arm_row(self, hand: str, action: np.ndarray, aperture: float, held: str, limits: dict) -> dict:
        """One arm's part of a step, for the page - its tool composed on, held
        steps not - under the lock."""
        from pinkrab_core.dataset import delta_to_matrix

        delta = delta_to_matrix(action)
        step_mm = float(np.linalg.norm(action[:3]) * 1000.0)
        turn = (np.trace(delta[:3, :3]) - 1.0) / 2.0
        step_deg = math.degrees(math.acos(max(-1.0, min(1.0, turn))))
        # Past the run's own limits - the ones the node held on, speeds over
        # one of its steps - is a bad answer, not the tool moving.
        fps = limits.get("fps") or CORE.dataset_fps
        wild = (bool(held) or step_mm * fps > limits.get("max_tool_speed_mm_s", CORE.policy_max_tool_speed_mm_s)
                or step_deg * fps > limits.get("max_tool_turn_dps", CORE.policy_max_tool_turn_dps))
        if not held:
            self._poses[hand] = self._poses[hand] @ delta
        return {"action": [round(float(v), 6) for v in action], "aperture": round(aperture, 4),
                "step_mm": round(step_mm, 3), "step_deg": round(step_deg, 3), "wild": wild,
                "where": [round(float(v), 5) for v in self._poses[hand][:3, 3]]}

    # --- the loop ---------------------------------------------------------

    def _run(self, joints_now) -> None:
        # Imported here, not at the top: torch and the checkpoint cost seconds
        # to load, and a studio that never opens this page should not pay them.
        from pinkrab_core.dataset import delta_to_matrix, relative_pose, state_vector
        from pinkrab_core.follow import TOOL_TURN
        from pinkrab_core.jaw import aperture_of, gripper_band
        from pinkrab_core.kinematics import ArmKinematics
        from pinkrab_core.origin import ready_joints
        from pinkrab_policy.rollout import Brain, Observer

        try:
            brain = Brain(self._checkpoint, task=self._task or None)
            observer = Observer(brain.sizes)
            # Where each tool starts, in the handheld's convention: where its
            # arm stands now, or at ready when it is not read; and each jaw's
            # opening as it reads now, as a run starts from - shut when unread.
            now = joints_now() or {}
            tools, apertures = {}, {}
            for hand in brain.hands:
                chain = ArmKinematics(hand)
                ready = ready_joints(hand, chain.joint_names)
                joints = [math.degrees(now[n]) if n in now else r for n, r in zip(chain.joint_names, ready)]
                turn = np.eye(4)
                turn[:3, :3] = TOOL_TURN[hand]
                tools[hand] = chain.forward(joints) @ turn
                band = gripper_band(hand)
                apertures[hand] = aperture_of(hand, now.get(f"{hand}_gripper", band[0]), band)
        except Exception as exc:                      # noqa: BLE001 - said on the page
            ALERTS.say("policy", f"could not load the checkpoint: {exc}")
            with self._lock:
                self._note = f"could not load the checkpoint: {exc}"
                self._loading = False
            return

        with self._lock:
            self._brain = brain
            self._task = brain.task or ""
            self._settings = brain.settings
            self._loading = False
            self._poses = {hand: np.eye(4) for hand in brain.hands}

        # The jaws the first observation carries are the arms' as read when
        # the asking began; from the next step on they are the policy's own
        # openings, as nothing here moves a jaw.
        # One step a tick of its dataset's rate.
        period = brain.period
        beat = time.monotonic()
        limits = brain.settings

        while not self._stop.is_set():
            began = time.monotonic()
            try:
                images = observer.observe()
            except Exception as exc:                  # noqa: BLE001
                self._trouble(f"no picture: {exc}")
                if self._stop.wait(0.5):
                    break
                continue

            hands = brain.hands
            relative = relative_pose(tools[hands[0]], tools[hands[1]]) if len(hands) == 2 else None
            try:
                action = brain.ask(images, state_vector(brain.state_names, apertures, relative))
            except Exception as exc:                  # noqa: BLE001
                self._trouble(f"the policy did not answer: {exc}")
                if self._stop.wait(0.5):
                    break
                continue

            latency = (time.monotonic() - began) * 1000.0
            with self._lock:
                # A step past the limits the rollout would hold on is the
                # policy answering badly, not the tool moving. Marked rather
                # than refused: nothing is being driven, and the whole point
                # of looking is to see how often it happens.
                arms = {hand: self._arm_row(hand, action[10 * i:10 * i + 10], apertures[hand], "", limits)
                        for i, hand in enumerate(hands)}
                wild = any(row["wild"] for row in arms.values())
                self._steps += 1
                if wild:
                    self._wild += 1
                now_s = time.monotonic()
                gap = now_s - beat
                beat = now_s
                # Smoothed, so the number on the page does not flicker with
                # one slow decode.
                rate = 1.0 / gap if gap > 1e-6 else 0.0
                self._hz = rate if self._hz == 0.0 else self._hz * 0.8 + rate * 0.2
                self._latency_ms = latency
                self._note = ""
                self._history.append({"step": self._steps, "at": time.time(), "arms": arms, "wild": wild,
                                      "latency_ms": round(latency, 1)})

            # What the next observation carries: the openings asked for, the
            # tools moved by the deltas just answered. This is the
            # free-running part.
            for i, hand in enumerate(hands):
                step = action[10 * i:10 * i + 10]
                apertures[hand] = float(step[9])
                tools[hand] = tools[hand] @ delta_to_matrix(step)

            rest = period - (time.monotonic() - began)
            if rest > 0 and self._stop.wait(rest):
                break

        with self._lock:
            self._brain = None

    def _trouble(self, note: str) -> None:
        with self._lock:
            self._note = note
