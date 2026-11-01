"""DAgger: a policy's run recorded whole, and the handhelds taking its arms over.

Every run a policy drives (the policy page's Drive) is recorded from its start
to its stop as a `rollout` episode, from the robot's side: the arms' cameras,
the driven arms' joints, and the top camera while a correction has it open. Each reading of the joints says
who had the arms then (`controller`):

    policy   the gate in MODE_POLICY, the rollout node driving
    human    the gate in MODE_CORRECT, the arms following the handhelds
    hold     the gate in MODE_CORRECT, the clutch holding the arms
    none     anything else - the walk to ready, a run stopped at an arm's reach

Each take-over is a correction, numbered in the run, and every reading in
it carries its number. Convert learns from the `human` stretches, one
dataset episode each, beside the demonstrations. The rest stays in the
episode: what the policy did before each correction, and the runs it
finished alone.

While a policy drives the arms, the operator can take them over with the
handhelds - the top camera tracking each one's solid, as in teleop, each
hand its own arm - and put them right. Handing back gives the arms to the
policy again from where the hands left them. The top camera is open only
for that: the arms' live view starts it closed, a take-over opens it - it
comes up while the arms are held - and it is closed again once the hands
let go of the arms for good: handed back, a redo given up, the run stopped.
A correction thrown away keeps it open for the redo.

The floor's pedals do it while the policy page is in front - it says so over
its socket once a second: 2 takes over and hands back, 1 holds the arms and
lets them follow again - a clutch, the hands set down and picked up held
another way - and 3 stops the arms where they are, as on every page. With
no correction in hand it is the stop and keeps the run's recording; in a
correction it throws the correction away (below) and the run goes on. A
take-over begins held, so each hand is set the way its arm holds its tool
before anything follows it. The arms follow the hands' motion against the
desk solid, not their place, so the desk solid may lie anywhere in view.
The page's buttons are the same calls.

A correction gone wrong is made again rather than kept. The arms' pose at
the take-over is kept with it. Pedal 3 throws the correction away: the
arms leave the hands and hold where they are, and a `redo` event goes in
the episode, which convert reads to leave its readings out. What was moved
on the desk is put back by hand; then 2 walks the arms back to the
take-over pose in the studio mode, at the parking speed, and takes them
over again, held, as the next correction (redo) - or 3 stops the run.
Pressing 3 never sets the arms moving.

What the gate is told:

    take over   MODE_CORRECT        teleop holds the arms where they stand, then follows on pedal 1
    hand back   MODE_POLICY         the rollout goes on from there
    throw away  MODE_IDLE           the arms hold where they are
    redo        MODE_STUDIO         the arms walked back to the take-over pose (idle as the walk ends),
                then MODE_CORRECT   held again, a new correction

The recording starts with Drive, before the node is started - a run that
cannot be recorded is not driven - and is closed and kept at the stop: the
page's Stop, pedal 3 with no correction in hand, the page's Disconnect.
Taking over, following, holding and handing back change nothing in it but
the readings' `controller` and `correction` (the take-over's number, while
the gate is in MODE_CORRECT), and a correction thrown away adds a `redo`
event. `controller` is what teleop's word on following decides. The joints come from the bridge's
pushes, sixty a second, and go to the live view in batches.
"""
from __future__ import annotations

import collections
import math
import threading
import time
from collections.abc import Callable

from pinkrab_core.config import SETTINGS as CORE
from pinkrab_core.jaw import aperture_of, gripper_band
from pinkrab_core.kinematics import arm_fingerprint
from pinkrab_umi.config import SETTINGS as UMI

from .alerts import ALERTS
from .real import MODE_CORRECT, MODE_IDLE, MODE_POLICY, MODE_STUDIO
from .rig import live_request

# How long the policy page's last word keeps the pedals; it speaks every second.
IN_FRONT_S = 3.0
# How long the gate is given to say it took a mode.
MODE_WAIT_S = 3.0
# How often the joints gathered go to the live view.
SEND_EVERY_S = 0.2
# Closing three videos takes the live view a few seconds.
STOP_TIMEOUT_S = 15.0
# The rollouts of this sitting the page lists, newest first.
KEPT = 20
# A redo's walk back to the take-over pose: done once every joint is this
# close, and given the time the farthest joint takes at the parking speed and
# this much more before it is given up.
REDO_TOL_DEG = 2.0
REDO_SLACK_S = 10.0

# The floor's pedals, by the names the foot switch posts.
FOLLOW_PEDAL = "jog"
TOGGLE_PEDAL = "handheld"
STOP_PEDAL = "stop"


def controller(mode: int, following: bool, phase: str) -> str:
    """Who had the arm at a reading: the gate's mode, whether teleop says the
    arm follows the hand, and the rollout node's phase as it last reported."""
    if mode == MODE_POLICY:
        return "policy" if phase == "driving" else "none"
    if mode == MODE_CORRECT:
        return "human" if following else "hold"
    return "none"


class Dagger:
    def __init__(self, real, policy, teleop_up: Callable[[], bool], halt: Callable[[], None],
                 walk: Callable[[list[dict[str, float]]], int | None]) -> None:
        self._real = real
        self._policy = policy
        self._teleop_up = teleop_up
        # The policy page's Stop, for the stop pedal: the gate idle, the run ended.
        self._halt = halt
        # The studio's pose walk, through the bridge: a redo's way back.
        self._walk = walk
        self._lock = threading.Lock()     # what is below
        self._act = threading.Lock()      # one order at a time
        self._flushing = threading.Lock() # one batch of readings on its way at a time
        # How many stops there have been: an order that began before the
        # latest one does not ask the gate for anything after it, and nor does
        # one for a take-over that began before it.
        self._halts = 0
        # The take-over in hand - the stop count it began at, its number in
        # the run, the arms' pose when it began and whether it was thrown
        # away (awaiting its redo) - until it is handed back or stopped.
        self._taken: dict | None = None
        # The corrections of the run being recorded, and of them how many redone.
        self._corrections = 0
        self._redone = 0
        # A redo on its way: the walk back, then the take-over again.
        self._redoing = False
        # A throw on its way: a second 3 meanwhile is the stop.
        self._throwing = False
        self._seen = 0.0
        self._recording: str | None = None
        self._arms: list[str] = []
        self._samples: list[dict] = []
        self._saved: collections.deque = collections.deque(maxlen=KEPT)
        self._note = ""
        self._bands: dict[str, tuple[float, float]] = {}
        threading.Thread(target=self._send_samples, name="dagger-samples", daemon=True).start()

    # --- the page in front --------------------------------------------------

    def seen(self) -> None:
        """The policy page, visible, once a second."""
        with self._lock:
            self._seen = time.monotonic()

    @property
    def armed(self) -> bool:
        with self._lock:
            return time.monotonic() - self._seen < IN_FRONT_S

    @property
    def correcting(self) -> bool:
        with self._lock:
            return self._taken is not None

    def pedal(self, name: str, down: bool) -> None:
        """A press (not a release) on the floor, while the policy page is in front.
        Off the request thread: the stop closes a recording, which takes seconds."""
        if not down or not self.armed:
            return
        order = {TOGGLE_PEDAL: self.toggle, FOLLOW_PEDAL: self.follow_toggle, STOP_PEDAL: self.stop_or_redo}.get(name)
        if order is not None:
            threading.Thread(target=order, name="dagger-pedal", daemon=True).start()

    # --- the run's recording ------------------------------------------------

    def begin(self, arms: list[str], checkpoint: str) -> dict:
        """The run Drive is about to start recorded from here - asked under the
        cameras lock, before the node is started. Refused, not waited for,
        while another order holds the run - a stop closing the last run's
        recording takes seconds, and a Stop pressed while Drive waited
        would find nothing yet to stop."""
        if self._policy.state()["driving"]:
            return self._fail("a policy is driving - stop it first")
        if not self._act.acquire(blocking=False):
            return self._fail("the last run's recording is still closing - Drive again in a moment")
        try:
            with self._lock:
                closing = self._recording is not None
            if closing:
                return self._fail("the last run's recording is still closing - Drive again in a moment")
            answer, code = live_request("/episode/start", {
                "kind": "rollout", "arms": arms, "policy": checkpoint,
                "description": {arm: arm_fingerprint(arm) for arm in arms}})
            if code != 200:
                return self._fail(answer.get("error") or "the live view did not start recording")
            with self._lock:
                self._recording = answer.get("name")
                self._arms = list(arms)
                self._samples = []
                self._note = ""
                self._corrections = 0
                self._redone = 0
            return {"ok": True, "name": answer.get("name")}
        finally:
            self._act.release()

    def abandon(self) -> None:
        """The recording begun for a run that did not start, thrown away."""
        with self._act:
            if self._recording is not None:
                self._end(keep=False)

    # --- the orders ---------------------------------------------------------

    def halt_begun(self) -> tuple[int, str | None]:
        """A stop: any order still on its way asks the gate for nothing more.
        Returns the stop's count and the recording it closes (`stop`) - not
        one a Drive after it begins while the close waits its turn."""
        with self._lock:
            self._halts += 1
            return self._halts, self._recording

    def _mode_unless_stopped(self, halts: int, **fields) -> bool:
        """Ask the gate for a mode, unless a stop has come since `halts` was read.
        The check and the asking under the lock a stop counts itself under, so
        the stop's idle is always queued after - never before - what this asked."""
        with self._lock:
            if self._halts != halts:
                return False
            self._real.request("mode", **fields)
            return True

    def _hand_has(self) -> bool:
        """Whether the gate has an arm with the handheld - which is what hand
        back answers, taken over from here or not: a studio started again
        mid-correction still leaves the arm with the hand until it is handed back."""
        teleop = self._real.snapshot().get("teleop") or {}
        return bool(teleop.get("armed")) and teleop.get("mode") == MODE_CORRECT

    @property
    def thrown(self) -> bool:
        with self._lock:
            return bool(self._taken and self._taken.get("thrown"))

    def toggle(self) -> dict:
        if self.thrown:
            return self.redo()
        return self.hand_back() if self.correcting or self._hand_has() else self.take_over()

    def _busy_redoing(self) -> dict | None:
        """An order refused while a redo walks the arms back, rather than left
        waiting for it and done once the arms are taken over again: a press
        made during the walk is not one for after it. The stop alone goes
        through (stop_or_redo)."""
        with self._lock:
            redoing = self._redoing
        return self._fail("redoing - wait for the arms to be taken over again") if redoing else None

    def take_over(self) -> dict:
        """The policy's arms to the handhelds, held until the first follow (follow_toggle)."""
        if busy := self._busy_redoing():
            return busy
        with self._act:
            with self._lock:
                halts = self._halts
            if self.correcting:
                return self._fail("already correcting")
            run = self._policy.state()
            if not run["driving"]:
                return self._fail("no policy is driving - Drive first")
            teleop = self._real.snapshot().get("teleop") or {}
            if not (teleop.get("armed") and teleop.get("mode") == MODE_POLICY):
                return self._fail("the gate is not in policy mode - nothing to take over")
            if not self._teleop_up():
                return self._fail("the handheld follower (teleop) is not running - Connect again")
            # Opened now, to come up while the gate hands the arms over and
            # they are held: nothing follows the hands before it sees them.
            self._top_camera(True)
            # Where the arms stand as they are taken over: what a redo walks back to.
            prefixes = tuple(f"{arm}_" for arm in run["hands"])
            pose = {k: v for k, v in self._real.angles().items() if k.startswith(prefixes)}
            if not self._mode_unless_stopped(halts, mode=MODE_CORRECT):
                self._top_camera(False)
                return self._fail("stopped before the arms were handed over")
            if not self._wait_mode(MODE_CORRECT):
                reason = (self._real.snapshot().get("teleop") or {}).get("reason") or "no answer"
                # Behind the correction asked for, in the same queue: a gate
                # that takes it late is given straight back to the policy.
                self._mode_unless_stopped(halts, mode=MODE_POLICY)
                self._top_camera(False)
                return self._fail(f"the gate did not hand the arms over ({reason})")
            with self._lock:
                self._corrections += 1
                self._taken = {"halts": halts, "number": self._corrections, "pose": pose, "thrown": False}
            return {"ok": True}

    def hand_back(self) -> dict:
        """The arms to the policy again; the recording goes on."""
        if busy := self._busy_redoing():
            return busy
        with self._act:
            with self._lock:
                # The take-over's own count: a stop since it began is a stop
                # this does not undo.
                taken = self._taken
                halts = taken["halts"] if taken else self._halts
            if not (taken or self._hand_has()):
                return self._fail("nothing is being corrected")
            if taken and taken["thrown"]:
                return self._fail("the correction was thrown away - 2 walks the arms back, 3 stops the run")
            # Not after a stop: the arm stays where it is.
            self._mode_unless_stopped(halts, mode=MODE_POLICY)
            with self._lock:
                self._taken = None
            self._top_camera(False)
            return {"ok": True, "handed_back": True}

    def _following(self) -> bool:
        """Whether the arms follow the hands - teleop's clutch, as teleop says
        it (through the bridge); held when it has said nothing."""
        return bool(self._real.snapshot().get("following"))

    def follow_toggle(self) -> dict:
        """Hold the arms, or let them follow the hands again - a clutch:
        following again takes each origin where its hand is then. Teleop flips it, one
        flip a press in the order they come, and what it then says it does is
        what the page shows and the readings are marked with."""
        if busy := self._busy_redoing():
            return busy
        with self._act:
            with self._lock:
                taken = self._taken
            if taken is None:
                if self._hand_has():
                    return self._fail("the arms were not taken over from here - hand them back first")
                return self._fail("nothing is being corrected")
            if taken["thrown"]:
                return self._fail("the correction was thrown away - 2 walks the arms back, 3 stops the run")
            # Under the lock a stop counts itself under, as a mode is asked
            # for: the stop's idle always goes to the gate after this, never before.
            with self._lock:
                stopped = self._halts != taken["halts"]
                if not stopped:
                    self._real.request("follow")
            if stopped:
                return self._fail("stopped - the arms follow nothing more")
            return {"ok": True}

    def stop(self, halt: tuple[int, str | None], settle: Callable[[], None] | None = None) -> dict:
        """The run's recording closed and kept - for every stop of the run
        (halt_policy): the page's Stop, the stop pedal, the page's
        Disconnect, which stops the run first. `halt` is what halt_begun
        said: a take-over and a recording begun after this stop are left
        alone. `settle` runs first, as soon as no order of a take-over is on
        its way - what the stop asks of the gate after any of them - and
        before the videos close."""
        count, name = halt
        with self._act:
            if settle is not None:
                settle()
            with self._lock:
                if self._taken is not None and self._taken["halts"] < count:
                    self._taken = None
                recording = name is not None and self._recording == name
                taken = self._taken is not None
            # Nothing follows the hands after a stop: unless a take-over begun
            # since holds the arms, the top camera goes too.
            if not taken:
                self._top_camera(False)
            if not recording:
                return {"ok": True}
            return self._end(keep=True)

    def stop_or_redo(self) -> dict:
        """Pedal 3 on the policy page, and its button: in a correction, the
        correction thrown away and the arms held (throw_away); otherwise - a
        correction already thrown away or being thrown, a redo walking the
        arms back, no correction at all - the stop it is everywhere: the gate
        idle, the run ended, its recording kept. Either way the arms stop
        where they are."""
        with self._lock:
            throw = (self._taken is not None and not self._taken["thrown"]
                     and not self._redoing and not self._throwing)
            self._throwing = self._throwing or throw
        if not throw:
            return self._halt()
        try:
            return self.throw_away()
        finally:
            with self._lock:
                self._throwing = False

    def throw_away(self) -> dict:
        """The correction in hand thrown away: the arms taken from the hands
        and held where they are, then the correction marked in the recording
        for convert to leave out. The arms go back to where it began only on
        the redo asked after it (pedal 2). Asked only by stop_or_redo."""
        with self._act:
            with self._lock:
                taken = self._taken
                if taken is None:
                    return self._fail("no correction to throw away")
                halts = taken["halts"]
                recording = self._recording
            # The arms first: nothing follows the hands from here.
            if not self._mode_unless_stopped(halts, mode=MODE_IDLE):
                return self._fail("stopped - the arms hold where they stand")
            if not self._wait_gate(lambda t: not (t.get("armed") and t.get("mode") == MODE_CORRECT)):
                # Not a correction to go on with: the next 3 is the stop.
                with self._lock:
                    self._taken = None
                return self._fail("the gate did not go idle - the arms may still follow the hands; Stop")
            with self._lock:
                self._taken = {**taken, "thrown": True}
            if recording is not None:
                answer, code = live_request("/episode/redo", {"correction": taken["number"]})
                if code != 200:
                    return self._fail((answer.get("error") or "the recording did not take the redo")
                                      + " - this correction stays in the recording; the arms hold")
            with self._lock:
                self._redone += 1
            return {"ok": True, "thrown": True}

    def redo(self) -> dict:
        """A correction thrown away, begun again: the arms walked back to
        where they stood when it began, then taken over again - held, the
        next correction. A redo that does not finish ends the correction:
        the gate idle, the arms holding where they stand."""
        if busy := self._busy_redoing():
            return busy
        with self._act:
            with self._lock:
                taken = self._taken
                if taken is None or not taken["thrown"]:
                    return self._fail("no correction thrown away - 3 throws one away first")
                halts = taken["halts"]
                self._redoing = True
            try:
                # Open still from the correction thrown away, unless something closed it.
                self._top_camera(True)
                if not self._walk_back(taken["pose"], halts):
                    return self._end_correction(halts, "the walk back to the take-over pose did not finish")
                if not self._mode_unless_stopped(halts, mode=MODE_CORRECT) or not self._wait_mode(MODE_CORRECT):
                    return self._end_correction(halts, "the arms were walked back but not taken over again")
                with self._lock:
                    self._corrections += 1
                    self._taken = {**taken, "number": self._corrections, "thrown": False}
                return {"ok": True, "redone": True}
            finally:
                with self._lock:
                    self._redoing = False

    def _end_correction(self, halts: int, why: str) -> dict:
        """A redo given up: the correction over, the gate idle unless a stop
        came first - which idles it itself."""
        stopped = not self._mode_unless_stopped(halts, mode=MODE_IDLE)
        with self._lock:
            self._taken = None
        self._top_camera(False)
        if stopped:
            return self._fail("stopped during the redo - the arms hold where they stand")
        return self._fail(f"{why} - the arms hold where they stand; Stop to end the run")

    def _walk_back(self, pose: dict[str, float], halts: int) -> bool:
        """The arms walked to `pose` through the gate in the studio mode - the
        teleop page's pose walk - and arrived, the gate left idle as a walk
        ends, unless a stop came first. Arms already there are not walked:
        the gate is idle already (throw_away)."""
        if not pose:
            return False
        if self._near(pose):
            with self._lock:
                return self._halts == halts
        with self._lock:
            if self._halts != halts:
                return False
            self._real.drive(MODE_STUDIO)
        if self._walk([pose]) is None:
            return False
        here = self._real.angles()
        farthest = max((abs(math.degrees(v - here[k])) for k, v in pose.items() if k in here), default=0.0)
        asked = time.monotonic()
        end = asked + farthest / max(1.0, CORE.park_speed_dps) + REDO_SLACK_S
        # Arrived only once this walk was under way - seen walking, or given
        # the time the bridge takes to read the request: the arms near the
        # pose and nothing walking before it starts is not the end of it.
        started = False
        while time.monotonic() < end:
            with self._lock:
                if self._halts != halts:
                    return False
            walking = bool(self._real.snapshot().get("walking"))
            started = started or walking or time.monotonic() - asked > MODE_WAIT_S
            if started and not walking and self._near(pose):
                return True
            time.sleep(0.05)
        return False

    def _near(self, pose: dict[str, float]) -> bool:
        """Whether every joint of `pose` reads within REDO_TOL_DEG of it."""
        here = self._real.angles()
        return all(k in here and abs(math.degrees(v - here[k])) <= REDO_TOL_DEG for k, v in pose.items())

    def delete(self, name: str) -> dict:
        """A rollout of this sitting, thrown away from the shelf."""
        answer, code = live_request("/episode/delete", {"name": name})
        if code != 200:
            return self._fail(answer.get("error") or f"could not delete {name}")
        with self._lock:
            self._saved = collections.deque((c for c in self._saved if c["name"] != name), maxlen=KEPT)
        return {"ok": True}

    def outcome(self, name: str, outcome: str | None) -> dict:
        """What the person watching said of a rollout - succeeded, failed, or
        taken back - kept in its meta.json by the live view."""
        answer, code = live_request("/episode/outcome", {"name": name, "outcome": outcome})
        if code != 200:
            return self._fail(answer.get("error") or f"could not mark {name}")
        return {"ok": True}

    # --- the robot's readings -----------------------------------------------

    def robot(self, angles: dict[str, float], teleop: dict | None, following: bool | None) -> None:
        """One push from the bridge: kept, while a run records, for its arms -
        marked with who had them."""
        with self._lock:
            if self._recording is None or not angles:
                return
        phase = self._policy.phase
        mode = teleop.get("mode") if teleop and teleop.get("armed") else 0
        with self._lock:
            if self._recording is None:
                return
            prefixes = tuple(f"{arm}_" for arm in self._arms)
            number = self._taken["number"] if self._taken and mode == MODE_CORRECT else None
            self._samples.append({"host_ts": time.time(), "mode": mode, "following": bool(following),
                                  "controller": controller(mode, bool(following), phase),
                                  "correction": number,
                                  "joints": {k: v for k, v in angles.items() if k.startswith(prefixes)}})

    def _send_samples(self) -> None:
        while True:
            time.sleep(SEND_EVERY_S)
            self._flush()

    def _flush(self) -> None:
        # Taken and sent as one: a batch taken here must reach the live view
        # before a stop that follows it, or it lands on no episode - or on the
        # next one.
        with self._flushing:
            with self._lock:
                samples, self._samples = self._samples, []
            if not samples:
                return
            answer, code = live_request("/episode/robot", {"samples": samples})
        if code != 200:
            self._say(f"the live view kept no robot readings: {answer.get('error')}")

    def _end(self, keep: bool) -> dict:
        """The recording closed - its last readings sent first - and kept or thrown away."""
        with self._lock:
            name, arms = self._recording, self._arms
            self._recording = None    # no reading is taken for it from here
        self._flush()                 # and what was taken goes before the stop
        answer, code = live_request("/episode/stop", {}, timeout=STOP_TIMEOUT_S)
        if code != 200:
            return self._fail(answer.get("error") or f"could not close {name}")
        if not keep:
            gone, code = live_request("/episode/delete", {"name": name})
            if code != 200:
                return self._fail(gone.get("error") or f"could not throw {name} away")
            return {"ok": True, "discarded": name}
        episode = answer.get("episode") or {}
        with self._lock:
            self._saved.appendleft({"name": name, "arms": arms, "seconds": episode.get("seconds"),
                                    "controlled": episode.get("controlled"), "at": time.time()})
        return {"ok": True, "saved": name}

    def _top_camera(self, on: bool) -> None:
        """The live view's top camera opened (for a correction) or closed. The
        view answers at once - opening takes a few seconds, and its stats say
        how it went; a view that does not answer is said."""
        answer, code = live_request("/head/retry" if on else "/head/off", {})
        if code != 200:
            self._say(f"the top camera was not {'opened' if on else 'closed'}: "
                      f"{answer.get('error') or 'no answer'}")

    def _wait_mode(self, mode: int) -> bool:
        return self._wait_gate(lambda t: bool(t.get("armed")) and t.get("mode") == mode)

    def _wait_gate(self, test: Callable[[dict], bool]) -> bool:
        """Whether what the gate says (its teleop state) passes `test` within MODE_WAIT_S."""
        end = time.monotonic() + MODE_WAIT_S
        while time.monotonic() < end:
            if test(self._real.snapshot().get("teleop") or {}):
                return True
            time.sleep(0.05)
        return False

    # --- what the page sees -----------------------------------------------------

    def _fail(self, note: str) -> dict:
        self._say(note)
        return {"error": note}

    def _say(self, note: str) -> None:
        with self._lock:
            self._note = note
        ALERTS.say("dagger", note)

    def _grip(self, arm: str) -> float | None:
        """What the arm's jaw was last sent, as a fraction of its band - where
        the hand's opening has to come to before it takes the jaw (teleop
        catches it there, not where it reads: on a block that is the block's
        width, short of the squeeze). Its reading until anything was sent."""
        name = f"{arm}_gripper"
        here = self._real.sent().get(name, self._real.angles().get(name))
        if here is None:
            return None
        if arm not in self._bands:
            try:
                self._bands[arm] = gripper_band(arm)
            except (OSError, KeyError, TypeError, ValueError):
                return None
        return round(aperture_of(arm, here, self._bands[arm]), 3)

    def state(self) -> dict:
        armed = self.armed
        run = self._policy.state()
        with self._lock:
            taken = self._taken
            # A rollout deleted elsewhere - the collect page's shelf - is
            # gone from this list too.
            self._saved = collections.deque((c for c in self._saved if (UMI.episodes_dir / c["name"]).is_dir()),
                                            maxlen=KEPT)
            arms = self._arms if self._recording else run["hands"] if run["driving"] else []
            answer = {
                "armed": armed,
                "driving": bool(run["driving"]),
                "correcting": taken is not None,
                "redoing": self._redoing,
                "thrown": bool(taken and taken["thrown"]),
                "corrections": self._corrections - self._redone,
                "redone": self._redone,
                "recording": self._recording,
                "arms": arms,
                "saved": list(self._saved),
                "note": self._note,
                "jaw_catch": CORE.correct_jaw_catch,
            }
        answer["following"] = self._following()
        answer["grips"] = {arm: self._grip(arm) for arm in arms}
        # The arm with the hand though not taken over from here - a studio
        # started again mid-correction: hand back is still what pedal 2 does.
        answer["hand_has"] = answer["correcting"] or self._hand_has()
        return answer
