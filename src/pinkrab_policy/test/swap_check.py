"""Offline check of pinkrab-picture-swap on two hands: a policy that moves
each hand with its own camera is said to look at its pictures, one that
answers from the state is not, and each hand is read from its own
place in the action. No checkpoint, no dataset: both are stand-ins.

    python src/pinkrab_policy/test/swap_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pinkrab_core import demo_start  # noqa: E402
from pinkrab_core.dataset import action_names, image_key, state_names  # noqa: E402
from pinkrab_policy import rollout, swap  # noqa: E402

FAILURES: list[str] = []
HANDS = ("left", "right")
NAMES = action_names(HANDS)


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def direction(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=3)


class Dataset:
    """Frames whose pictures each carry a direction for their hand (in the
    first pixel's brightness), and whose action is that direction."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.n = 60

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        action = np.zeros(len(NAMES), dtype=np.float32)
        x = {"task": "put the blocks in the bowls"}
        for k, hand in enumerate(HANDS):
            picture = torch.zeros(3, 4, 4)
            picture[0, 0, 0] = (i * 2 + k) / (2 * self.n)
            x[image_key(hand)] = picture
            at = [NAMES.index(f"{hand}_{n}") for n in ("dx", "dy", "dz")]
            action[at] = direction(i * 2 + k)
        x["action"] = torch.from_numpy(action)
        x["observation.state"] = torch.from_numpy(np.random.default_rng(1000 + i).normal(size=11).astype(np.float32))
        return x


class Brain:
    """A policy of two hands: `looks` answers each hand from its own picture,
    otherwise from the state."""

    looks = True

    def __init__(self, *_args, **_kwargs) -> None:
        self.sizes = {image_key(h): (4, 4) for h in HANDS}
        self.hands = HANDS
        self.state_dim = 11
        self.state_names = state_names(HANDS)
        self.language = False

    def reset(self) -> None:
        pass

    def ask(self, pictures: dict, state: np.ndarray, task=None) -> np.ndarray:
        out = np.zeros(len(NAMES), dtype=np.float32)
        for k, hand in enumerate(HANDS):
            at = [NAMES.index(f"{hand}_{n}") for n in ("dx", "dy", "dz")]
            if self.looks:
                index = round(pictures[image_key(hand)][0, 0, 0] / 255 * 2 * Dataset().n)
                out[at] = direction(index)
            else:
                out[at] = state[3 * k:3 * k + 3]
        return out


def main() -> int:
    import lerobot.datasets.lerobot_dataset as lerobot_dataset

    lerobot_dataset.LeRobotDataset = Dataset
    rollout.Brain = Brain
    demo_start.checkpoint_tasks = lambda _path: []
    demo_start.dataset_of_checkpoint = lambda _path: Path("/nowhere/blocks-two-hands")

    print("\npicture swap, two hands")
    Brain.looks = True
    looker = swap.picture_swap(Path("/nowhere/ckpt"), frames=40)
    for hand in HANDS:
        h = looker["hands"][hand]
        check(f"{hand}: a policy that reads its camera answers as recorded", h["cos_to_recorded"] > 0.99,
              f"{h['cos_to_recorded']:+.2f}")
        check(f"{hand}: and changes its answer with the pictures", h["cos_picture_swapped"] < 0.3,
              f"{h['cos_picture_swapped']:+.2f}")
        check(f"{hand}: and not with the state", h["cos_state_swapped"] > 0.99, f"{h['cos_state_swapped']:+.2f}")
    check("it is said to look at its pictures", looker["looks_at_pictures"])

    Brain.looks = False
    copycat = swap.picture_swap(Path("/nowhere/ckpt"), frames=40)
    for hand in HANDS:
        h = copycat["hands"][hand]
        check(f"{hand}: a policy that reads the state does not change with the pictures",
              h["cos_picture_swapped"] > 0.99, f"{h['cos_picture_swapped']:+.2f}")
    check("it is said not to look at its pictures", not copycat["looks_at_pictures"])

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
