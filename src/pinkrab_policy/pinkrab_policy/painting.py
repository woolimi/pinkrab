"""The arms' pictures painted the way a policy's dataset painted its takes.

A checkpoint whose dataset was converted with randomised backgrounds learned
from pictures in which only the grippers and the task's things are the
room's, everything else a pattern (pinkrab_umi.backgrounds). Shown the arms'
pictures as the cameras take them, it is asked about a room it never saw.
Painting paints them the same way, live:

- the task's things where its masker - taught on the Curate page - finds
  them in each arm camera's newest frame, in a process of its own on this
  machine (pinkrab_umi.live_mask, the masker's environment), its labels
  handed over in shared memory;
- the grippers where the arms' joints put them, drawn as keep_masks draws a
  rollout's;
- the lens border found from the first frames, as keep_masks finds a take's;
- and everything else one pattern for the run, as a take had one.

It paints on a thread of its own, each camera's frame as it comes - drawing
and painting two cameras take some 40 ms, two ticks of a 50 Hz run - and the
policy is shown the newest painted picture. The same pictures go to shared
memory as the camera policy_view_camera of each side, for the policy page.
"""
from __future__ import annotations

import json
import os
import queue
import random
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

from pinkrab_core.config import ARMS
from pinkrab_core.dataset import gripper_view, lens_fit
from pinkrab_core.kinematics import ArmKinematics
from pinkrab_umi import keep_masks, masker
from pinkrab_umi.backgrounds import painted, pattern
from pinkrab_umi.config import SETTINGS as UMI
from pinkrab_umi.frame_share import FrameReader, FrameWriter

# The lens border is found from this many frames, this far apart, in seconds.
BORDER_FRAMES = 20
BORDER_GAP_S = 0.05


class PaintingError(RuntimeError):
    """A policy that wants its pictures painted, and nothing to paint them with."""


def masker_for(checkpoint: str | Path) -> dict | None:
    """The masker a checkpoint's pictures are painted with: None for one
    whose dataset kept its takes' backgrounds; PaintingError where it wants
    one and none is here to use."""
    record = masker_wanted(checkpoint)
    if record is not None and not masker.weights_here(record).is_file():
        raise PaintingError(f"the masker that finds {', '.join(record['things'])} is on {record['machine']}, "
                            "not here")
    return record


def masker_wanted(checkpoint: str | Path) -> dict | None:
    """The masker a checkpoint's pictures are to be painted with, wherever its
    weights are - the one chosen for its takes' things (inference.yaml's
    maskers), else the newest that finds them: None for one whose dataset
    kept its takes' backgrounds, or with OFF chosen; PaintingError where the
    one chosen is gone or none finds them."""
    from pinkrab_core import inference

    things = painted_things(checkpoint)
    choice = None if things is None else inference.masker_choice(things)
    if choice in (None, inference.OFF):
        return None
    if choice != inference.AUTO:
        record = next((r for r in masker.maskers(things) if r["id"] == choice), None)
        if record is None:
            raise PaintingError(f"the masker chosen for {', '.join(things)}, {choice}, is gone - "
                                "choose another in the settings")
        return record
    record = masker.find(things)
    if record is None:
        raise PaintingError(f"no masker finds {', '.join(things)} - Make labels on the Curate page teaches one; "
                            "or choose off in the settings to show it the pictures as they are")
    return record


def painted_things(checkpoint: str | Path) -> list[str] | None:
    """The things a checkpoint's dataset kept of its takes' pictures, the rest
    painted over: None for one that kept the pictures whole; PaintingError
    where it cannot tell."""
    from pinkrab_core.demo_start import DemoStartError, dataset_of_checkpoint

    try:
        report = json.loads((dataset_of_checkpoint(Path(checkpoint)) / "meta" / "pinkrab_convert.json").read_text())
    except (DemoStartError, OSError, ValueError) as exc:
        raise PaintingError(f"cannot tell whether its pictures were painted: {exc}") from exc
    if not report.get("randomised"):
        return None
    told = {tuple(said["things"]) for e in report.get("episodes") or []
            if (said := keep_masks.account(UMI.episodes_dir / e["episode"])) and said.get("things")}
    if len(told) != 1:
        raise PaintingError("its takes are not here to say what was kept of them" if not told
                            else "its takes kept different things")
    return list(told.pop())


class Painting:
    """The arms' pictures, painted, for a run: started with the masker's
    process, ended with close()."""

    def __init__(self, record: dict, centres: dict[str, tuple[float, float]]) -> None:
        self.record = record
        self.hands = list(centres)
        self.centres = centres
        self.lenses = {}
        for hand in self.hands:
            fit = lens_fit(f"follower_{hand}")
            if fit is None:
                raise PaintingError(f"the {hand} arm camera's lens is not measured")
            self.lenses[hand] = keep_masks.Lens({"lens": fit, "centre": list(centres[hand])})
        self.cameras = {hand: FrameReader(f"follower_{hand}") for hand in self.hands}
        self.masks = {hand: FrameReader(UMI.live_masks_camera.format(side=hand)) for hand in self.hands}
        self.shown = {hand: FrameWriter(UMI.policy_view_camera.format(side=hand)) for hand in self.hands}
        self.chains = {side: ArmKinematics(side) for side in ARMS}
        # One pattern for the run, as a take had one.
        seed = random.getrandbits(32)
        self.backdrops = {hand: pattern(seed, lens.size) for hand, lens in self.lenses.items()}
        self._joints: dict[str, float] = {}
        self._lock = threading.Lock()
        self._newest: dict[str, tuple[np.ndarray, float]] = {}
        self._trouble = ""
        self._stop = threading.Event()
        try:
            self._process = self._start()
            self.borders = self._borders()
        except BaseException:
            self.close()
            raise
        self._thread = threading.Thread(target=self._paint, name="painting", daemon=True)
        self._thread.start()

    # --- what the policy is shown --------------------------------------------

    def set_joints(self, joints: dict[str, float]) -> None:
        """Where the arms stand now, radians by joint name: where the grippers are drawn."""
        with self._lock:
            self._joints = dict(joints)

    def view(self, hand: str) -> np.ndarray:
        """`hand`'s newest painted picture, BGR at its lens's size, or
        RuntimeError: none fresh now, and why."""
        with self._lock:
            newest, trouble = self._newest.get(hand), self._trouble
        if newest is None or time.monotonic() - newest[1] > UMI.camera_share_stale_s:
            raise RuntimeError(trouble or f"no painted picture of the {hand} arm camera yet")
        return newest[0]

    def close(self) -> None:
        self._stop.set()
        thread = getattr(self, "_thread", None)
        if thread is not None:
            thread.join(timeout=2.0)
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            # Its stdin closed, it ends; killed if it has not soon after.
            process.stdin.close()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for writer in self.shown.values():
            writer.close()
        err = getattr(self, "_err", None)
        if err is not None:
            err.close()

    # --- the masker's process -------------------------------------------------

    def _start(self) -> subprocess.Popen:
        python = UMI.masker_python
        if not python.is_file():
            raise PaintingError(f"no masker environment at {python} - see the docs' install page")
        import pinkrab_core
        import pinkrab_umi

        ask = {"cameras": {f"follower_{hand}": {"side": hand, "centre": list(self.centres[hand]),
                                                "size": list(self.lenses[hand].size)} for hand in self.hands},
               "weights": str(masker.weights_here(self.record)), "model": self.record["model"],
               "resolution": self.record["resolution"],
               "labels": [keep_masks.FIRST_THING + k for k in range(len(self.record["things"]))],
               "score": UMI.masker_score}
        # This checkout's packages, for the camera blocks and the dataset's cut.
        path = ":".join(str(Path(p.__file__).resolve().parent.parent) for p in (pinkrab_core, pinkrab_umi))
        self._err = tempfile.TemporaryFile("w+")  # noqa: SIM115 - it lives as long as the process
        process = subprocess.Popen([str(python), "-P", "-m", "pinkrab_umi.live_mask", json.dumps(ask)],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err, text=True,
                                   env={**os.environ, "PYTHONPATH": path})
        lines: queue.Queue = queue.Queue()
        threading.Thread(target=lambda: [lines.put(line) for line in process.stdout], daemon=True).start()
        try:
            line = lines.get(timeout=UMI.live_mask_start_s)
        except queue.Empty:
            line = ""
        if '"ready"' not in line:
            if process.poll() is None:
                process.kill()
            process.wait()
            self._err.seek(0)
            said = self._err.read().strip().splitlines()
            raise PaintingError("the masker did not start: " + (said[-1] if said else
                                f"no answer in {UMI.live_mask_start_s:.0f} s"))
        return process

    # --- painting ---------------------------------------------------------------

    def _borders(self) -> dict[str, np.ndarray]:
        """Each camera's lens border, from the brightest each pixel gets over
        BORDER_FRAMES frames."""
        import cv2

        brightest = {hand: np.zeros(lens.size, dtype=np.uint8) for hand, lens in self.lenses.items()}
        got = dict.fromkeys(self.hands, 0)
        for _ in range(BORDER_FRAMES):
            for hand in self.hands:
                try:
                    picture = self.cameras[hand].read()
                except RuntimeError:
                    continue
                view = gripper_view(picture, self.lenses[hand].size, self.centres[hand])
                np.maximum(brightest[hand], cv2.cvtColor(view, cv2.COLOR_BGR2GRAY), out=brightest[hand])
                got[hand] += 1
            time.sleep(BORDER_GAP_S)
        if missing := [hand for hand, n in got.items() if not n]:
            raise PaintingError(f"no picture from the {' and '.join(missing)} arm camera(s)")
        return {hand: keep_masks.circle_border(b) for hand, b in brightest.items()}

    def _paint(self) -> None:
        seen: dict[str, int] = {}
        while not self._stop.is_set():
            with self._lock:
                joints = self._joints
            tools, angles = {}, {}
            for side, chain in self.chains.items():
                names = chain.joint_names
                tools[side] = (chain.forward(np.degrees([joints[n] for n in names]))
                               if all(n in joints for n in names) else None)
                angles[side] = joints.get(f"{side}_gripper", keep_masks.jaw_angle(side, None))
            painted_any = False
            for hand in self.hands:
                try:
                    picture, count, stamp = self.cameras[hand].newest()
                    if seen.get(hand) == count:
                        continue
                    labels = self.masks[hand].read()[:, :, 0].copy()
                except RuntimeError as exc:
                    with self._lock:
                        self._trouble = str(exc)
                    continue
                seen[hand] = count
                lens = self.lenses[hand]
                labels[self.borders[hand] & (labels == 0)] = keep_masks.BORDER
                grippers = keep_masks.drawn(hand, lens, tools, angles)
                labels[grippers != 0] = grippers[grippers != 0]
                out = painted(gripper_view(picture, lens.size, self.centres[hand]), labels, self.backdrops[hand])
                self.shown[hand].write(out, stamp)
                with self._lock:
                    self._newest[hand] = (out, stamp)
                    self._trouble = ""
                painted_any = True
            if not painted_any:
                self._stop.wait(0.005)
