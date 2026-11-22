"""Offline check of a correction redone: a policy driving, the arms taken over
with their pose kept, the readings numbered by correction, pedal 3 while the
hands have the arms throwing it away - the arms held where they are, the
redo marked in the recording - and pedal 2 then redoing it: the arms walked
back to the take-over pose in the studio mode and taken over again, held, as
the next correction (no walk for arms already there). Pedal 3 with no
correction, one thrown away, or during a redo is the stop. A stop during
the walk back leaves the arms where they are, and a walk back that fails
ends the correction. The studio stops a run whose policy page has been gone
for longer than the configured time, or whose rollout node has exited, by the
same halt as the page's Stop, and tells the page why. No robot, no live view:
the gate, the bridge and the recording are stand-ins.

    python src/pinkrab_studio/test/dagger_check.py
"""
from __future__ import annotations

import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from pinkrab_core.config import SETTINGS
from pinkrab_studio import dagger as dagger_module
from pinkrab_studio.dagger import Dagger
from pinkrab_studio.feeds import Feeds
from pinkrab_studio.policy import PolicyRun
from pinkrab_studio.real import MODE_CORRECT, MODE_IDLE, MODE_POLICY, MODE_STUDIO
from pinkrab_studio.server import StudioState

FAILURES: list[str] = []
TAKEN = {"left_shoulder_pan": 0.10, "left_gripper": 0.30, "right_shoulder_pan": -0.10, "right_gripper": -0.30}


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


class Gate:
    """The robot as the studio hears it: the gate's mode, the arms' angles,
    a walk that lands where it was sent."""

    def __init__(self) -> None:
        self.mode = MODE_POLICY
        self.angles_now = dict(TAKEN)
        self.asked: list[tuple[str, dict]] = []
        self.walking = False
        self.lock = threading.Lock()

    def request(self, kind: str, **fields) -> int:
        with self.lock:
            self.asked.append((kind, fields))
            if kind == "mode":
                self.mode = fields["mode"]
        return len(self.asked)

    def drive(self, mode: int) -> int:
        self.request("mode", mode=MODE_IDLE)
        self.request("torque", arm="both", enable=True)
        return self.request("mode", mode=mode)

    def snapshot(self) -> dict:
        with self.lock:
            return {"teleop": {"armed": True, "mode": self.mode}, "following": False, "walking": self.walking}

    def angles(self) -> dict[str, float]:
        with self.lock:
            return dict(self.angles_now)

    def sent(self) -> dict[str, float]:
        return {}

    def walk(self, poses: list[dict[str, float]]) -> int:
        def land() -> None:
            with self.lock:
                self.walking = True
            time.sleep(0.1)
            with self.lock:
                self.angles_now.update(poses[-1])
                self.walking = False
        threading.Thread(target=land, daemon=True).start()
        self.walks.append(poses)
        return 1


class Run:
    """The policy's run: not driving yet when Drive begins the recording, then driving."""
    phase = "driving"
    driving = False

    def state(self) -> dict:
        return {"driving": self.driving, "hands": ["left", "right"], "running": self.driving}


def studio_over(run: PolicyRun, dagger, node_exit, page_gone):
    """The parts of the studio the watch over a run reads, around the real
    policy run: the rollout node's exit and how long no policy page has had a
    socket open are set by the caller."""
    halts: list[str] = []
    studio = SimpleNamespace(
        policy=run, dagger=dagger, feeds=SimpleNamespace(policy_page_gone_s=lambda: page_gone[0]),
        rig=SimpleNamespace(state=lambda: {"policy": {"exit": node_exit[0]}}, stop_policy=lambda: {}),
        real=SimpleNamespace(request=lambda kind, **fields: halts.append(f"{kind} {fields}") or 1),
        _walk_lock=threading.Lock(), _walk_target=(7, {"left_shoulder_pan": 0.0}),
        _after_halt=lambda halt: None)
    studio.halted = halts
    return studio


def watch_over_a_policy_run() -> None:
    print("\nthe page or the node of a driving policy gone")
    lost = SETTINGS.policy_page_lost_s
    run = PolicyRun()
    dagger = SimpleNamespace(correcting=False, halt_begun=lambda: (1, None))
    node_exit, page_gone = [None], [0.0]
    studio = studio_over(run, dagger, node_exit, page_gone)

    def reason() -> str | None:
        return StudioState.policy_halt_reason(studio)

    check("nothing is watched while no policy drives", reason() is None)
    page_gone[0] = lost + 5
    check("not even a page that is long gone", reason() is None)
    run._driving = True
    page_gone[0] = 0.0
    check("a page that has its socket open keeps it driving, in front or not", reason() is None)
    page_gone[0] = lost - 0.5
    check("a page back within the time keeps it driving", reason() is None)
    page_gone[0] = lost + 0.5
    check("no page for longer than the time stops it", reason() is not None, str(reason()))
    page_gone[0] = 0.0
    node_exit[0] = 1
    check("a rollout node that exited stops it", "exited" in (reason() or ""), str(reason()))
    node_exit[0] = None
    dagger.correcting, run._driving = True, False
    page_gone[0] = lost + 0.5
    check("and the page gone during a correction", reason() is not None, str(reason()))
    node_exit[0] = 1
    page_gone[0] = 0.0
    check("while a node that exited is no news once the run is over", reason() is None, str(reason()))
    dagger.correcting = False

    print("\nthe stop the watch makes")
    run._driving = True
    why = "no policy page has been connected"
    StudioState.halt_policy(studio, why)
    check("idles the gate", studio.halted and studio.halted[0].startswith("mode") and "0" in studio.halted[0],
          str(studio.halted))
    check("ends the run", not run.state()["driving"])
    check("tells the page why", run.state()["stopped"] == why, run.state()["stopped"])
    check("and a walk waited for is let go", studio._walk_target is None)
    StudioState.halt_policy(studio)
    check("the page's own stop says no reason", run.state()["stopped"] == "")

    print("\nthe policy pages' sockets")
    feeds = Feeds(MagicMock())
    now = time.monotonic()
    feeds.policy_heard(now - 5.0)
    check("counted from when the studio started, not moved back by an older answer", feeds.policy_page_gone_s() < 1.0)
    feeds._policy_heard = now - 20.0     # a studio up for a while
    feeds.policy_heard(now - 3.0)
    feeds.policy_heard(now - 9.0)
    gone = feeds.policy_page_gone_s()
    check("gone is counted from the freshest answer", 2.9 < gone < 4.0, str(gone))
    feeds.policy_heard(time.monotonic())
    check("an answer brings it back", feeds.policy_page_gone_s() < 0.5)

    print("\nwhat the clutch believes is recording")
    from pinkrab_studio.clutch import Clutch
    clutch = Clutch(threading.Lock())
    clutch.sync("take-3")
    check("is what the live view says", clutch.state()["recording"] == "take-3")
    clutch.sync(None)
    check("and nothing once it ended", clutch.state()["recording"] is None)


def main() -> int:
    posted: list[tuple[str, dict]] = []

    def live_request(path: str, body: dict, timeout: float = 5.0):
        posted.append((path, body))
        return ({"ok": True, "name": "rollout-1"} if path == "/episode/start" else {"ok": True}), 200

    dagger_module.live_request = live_request
    gate = Gate()
    gate.walks = []
    halts = []
    run = Run()
    d = Dagger(gate, run, teleop_up=lambda: True, halt=lambda: halts.append(1), walk=gate.walk)

    print("\ntaking over")
    check("Drive begins the recording", d.begin(["left", "right"], "checkpoint").get("ok") is True)
    run.driving = True
    answer = d.take_over()
    check("the arms are taken over", answer.get("ok") is True, str(answer))
    check("the take-over keeps the arms' pose", d._taken["pose"] == TAKEN, str(d._taken["pose"]))
    d.robot(gate.angles(), {"armed": True, "mode": MODE_CORRECT}, True)
    d._flush()
    sent = [x for path, body in posted if path == "/episode/robot" for x in body["samples"]]
    check("a reading in the correction carries its number", sent and sent[-1]["correction"] == 1,
          str(sent[-1:]))

    print("\nthrowing it away")
    gate.angles_now.update({"left_shoulder_pan": 0.6, "right_shoulder_pan": -0.6})   # the hands moved the arms
    gate.asked.clear()
    d.stop_or_redo()
    check("pedal 3 with the hands on the arms does not stop the run", not halts)
    check("the arms leave the hands first and hold", gate.asked == [("mode", {"mode": MODE_IDLE})], str(gate.asked))
    check("nothing walks", not gate.walks, str(gate.walks))
    check("the redo is marked in the recording",
          ("/episode/redo", {"correction": 1}) in posted, str(posted[-3:]))
    check("the correction is held as thrown away", d._taken and d._taken["thrown"], str(d._taken))
    pressed = [d.hand_back(), d.follow_toggle()]
    check("hand back and follow are refused", all("error" in x for x in pressed), str(pressed))

    print("\nredoing it (pedal 2)")
    gate.asked.clear()
    d.toggle()
    check("the arms are walked back to the take-over pose", gate.walks and gate.walks[-1] == [TAKEN],
          str(gate.walks[-1:]))
    check("the walk goes in the studio mode", ("mode", {"mode": MODE_STUDIO}) in gate.asked, str(gate.asked))
    check("and the arms are taken over again", gate.asked[-1] == ("mode", {"mode": MODE_CORRECT}), str(gate.asked[-1]))
    check("as the next correction", d._taken and d._taken["number"] == 2 and not d._taken["thrown"], str(d._taken))
    state = d.state()
    check("the page is told one correction kept so far and one redone",
          state["corrections"] == 1 and state["redone"] == 1, f"{state['corrections']}, {state['redone']}")

    print("\nredoing it with the arms still where they were taken over")
    d.throw_away()
    gate.asked.clear()
    walks = len(gate.walks)
    d.redo()
    check("no walk for arms already there", len(gate.walks) == walks, str(gate.walks[walks:]))
    check("taken over again from the hold", gate.asked == [("mode", {"mode": MODE_CORRECT})], str(gate.asked))
    check("as the next correction again", d._taken and d._taken["number"] == 3, str(d._taken))

    print("\npedal 3 during a redo, and after a throw")
    d._redoing = True
    d.stop_or_redo()
    pressed = [d.hand_back(), d.follow_toggle(), d.redo()]
    d._redoing = False
    check("is the stop", halts == [1], str(halts))
    check("and other orders are refused, not kept for after", all("error" in x for x in pressed), str(pressed))
    check("the correction still in hand", d._taken and d._taken["number"] == 3, str(d._taken))
    d.throw_away()
    d.stop_or_redo()
    check("pedal 3 with a correction thrown away is the stop", halts == [1, 1], str(halts))
    d._taken = {**d._taken, "thrown": False}
    halts.clear()

    print("\na second 3 while the first throws")
    d._throwing = True
    d.stop_or_redo()
    d._throwing = False
    check("is the stop", halts == [1], str(halts))
    check("and leaves the correction to the first", d._taken and not d._taken["thrown"], str(d._taken))
    halts.clear()

    print("\nhanding back, then pedal 3")
    d.hand_back()
    d.stop_or_redo()
    check("pedal 3 with no correction is the stop", halts == [1], str(halts))

    print("\na stop during the walk back")
    gate2 = Gate()
    gate2.walks = []
    stopped = []
    run2 = Run()
    d2 = Dagger(gate2, run2, teleop_up=lambda: True, halt=lambda: stopped.append(1), walk=gate2.walk)
    d2.begin(["left", "right"], "checkpoint")
    run2.driving = True
    d2.take_over()
    gate2.angles_now.update({"left_shoulder_pan": 1.0})
    d2.throw_away()

    def slow_walk(poses):
        gate2.walks.append(poses)
        with gate2.lock:
            gate2.walking = True
        return 1
    d2._walk = slow_walk
    worker = threading.Thread(target=d2.redo)
    worker.start()
    time.sleep(0.2)
    d2.halt_begun()
    worker.join(timeout=5)
    check("the redo gives up when stopped", not worker.is_alive())
    check("and asks for no correction after the stop",
          gate2.asked[-1][1].get("mode") != MODE_CORRECT, str(gate2.asked[-1]))
    check("and the correction is over", d2._taken is None, str(d2._taken))

    print("\na walk back that does not arrive")
    gate3 = Gate()
    gate3.walks = []
    run3 = Run()
    d3 = Dagger(gate3, run3, teleop_up=lambda: True, halt=lambda: None, walk=slow_walk)
    d3.begin(["left", "right"], "checkpoint")
    run3.driving = True
    d3.take_over()
    gate3.angles_now.update({"left_shoulder_pan": 1.0})
    d3.throw_away()
    dagger_module.REDO_SLACK_S = 0.3
    answer = d3.redo()
    check("the redo fails", "error" in answer, str(answer))
    check("the correction is over", d3._taken is None, str(d3._taken))
    check("the gate idled, not taken over", gate3.asked[-1] == ("mode", {"mode": MODE_IDLE}), str(gate3.asked[-1]))

    print("\na throw the recording refuses")
    gate4 = Gate()
    d4 = Dagger(gate4, Run(), teleop_up=lambda: True, halt=lambda: None, walk=gate4.walk)
    d4._policy.driving = False
    d4.begin(["left", "right"], "checkpoint")
    d4._policy.driving = True
    d4.take_over()
    dagger_module.live_request = lambda path, body, timeout=5.0: ({"error": "no episode"}, 409)
    answer = d4.throw_away()
    check("says so", "error" in answer and "stays in the recording" in answer["error"], str(answer))
    check("and the arms hold all the same", gate4.mode == MODE_IDLE and d4._taken["thrown"], f"{gate4.mode} {d4._taken}")

    print("\nthe hands have the arms")
    gate5 = Gate()
    d5 = Dagger(gate5, Run(), teleop_up=lambda: True, halt=lambda: None, walk=gate5.walk)
    check("not before a take-over", not d5.hand_has())
    d5._policy.driving = True
    d5.take_over()
    check("from the take-over", d5.hand_has())
    d5._taken = None
    gate5.mode = MODE_CORRECT
    check("and for a studio started again mid-correction", d5.hand_has())

    watch_over_a_policy_run()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
