"""The policy's chunks a step at a time (rollout.Plan), with no checkpoint.

    python src/pinkrab_policy/test/plan_check.py
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.spatial.transform import Rotation

from pinkrab_core import chunks
from pinkrab_policy.rollout import Plan

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def step(k: int) -> np.ndarray:
    """One hand's k-th step: a few millimetres and a degree or two."""
    R = Rotation.from_euler("zx", [1.0 + 0.1 * k, 0.5], degrees=True).as_matrix()
    return np.concatenate([[0.002, 0.001 * k, -0.001], R[:, 0], R[:, 1], [0.1 * k]])


class Brain:
    """Two hands, each chunk the relative trajectory of the steps from `at` on."""

    hands = ("left", "right")

    def __init__(self, settings: dict) -> None:
        self.settings = settings
        self.at = 0
        self.asked = 0

    def chunk(self, images, state, task=None) -> np.ndarray:
        self.asked += 1
        steps = np.array([np.concatenate([step(self.at + k), step(self.at + k)]) for k in range(20)])
        return chunks.relative_chunk(steps).astype(np.float32)


def main() -> int:
    print("\none chunk at a time")
    brain = Brain({"n_action_steps": 5, "temporal_ensemble_coeff": None})
    plan = Plan(brain)
    tools = {hand: np.eye(4) for hand in brain.hands}
    got = [plan.next({}, np.zeros(11), tools) for _ in range(5)]
    worst = max(np.abs(g[:10] - step(k)).max() for k, g in enumerate(got))
    check("its entries come back as the steps that composed them", worst < 1e-5 and brain.asked == 1, f"{worst:.2e}")
    plan.next({}, np.zeros(11), tools)
    check("a fresh chunk is asked once the first n_action_steps are spent", brain.asked == 2)

    print("\nblended")
    brain = Brain({"n_action_steps": 1, "temporal_ensemble_coeff": 0.01})
    plan = Plan(brain)
    tools = {hand: np.eye(4) for hand in brain.hands}
    worst = 0.0
    for tick in range(8):
        answer = plan.next({}, np.zeros(11), tools)
        # The arm makes each step; the next chunk is asked from where it went,
        # its first entry the step after.
        for i, hand in enumerate(brain.hands):
            tools[hand] = tools[hand] @ chunks.as_matrix(answer[10 * i:10 * i + 9])
        worst = max(worst, float(np.abs(answer[:9] - step(tick)[:9]).max()))
        brain.at += 1
    check("chunks that agree blend into the steps they share", worst < 1e-5, f"{worst:.2e}")
    check("the oldest chunk is let go once its entries are spent", len(plan.plans) <= 20)

    print("\n" + ("all checks passed" if not FAILURES else f"{len(FAILURES)} failed: {', '.join(FAILURES)}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
