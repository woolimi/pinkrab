"""What pinkrab-train teaches (run_log.Relative), on a dataset made here.

    python src/pinkrab_policy/test/relative_check.py
"""
from __future__ import annotations

import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import torch

from pinkrab_core import chunks
from pinkrab_policy.run_log import Relative

FAILURES: list[str] = []
CHUNK = 4


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}" + (f" - {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


def forward(k: int) -> np.ndarray:
    """One hand's step: 2 mm along x, no turn, the jaw opening k/10."""
    return np.array([0.002, 0, 0, 1, 0, 0, 0, 1, 0, k / 10])


class Dataset:
    """Two episodes of six steps, one hand; an item is a chunk from its frame,
    padded past its episode's end with the last step, as LeRobot pads."""

    def __init__(self) -> None:
        self.steps = np.array([forward(k) for k in range(6)] * 2, dtype=np.float32)
        self.episodes = np.repeat([0, 1], 6)
        self.hf_dataset = SimpleNamespace(data=pa.table({"action": self.steps.tolist(),
                                                         "episode_index": self.episodes}))
        like = {"mean": np.zeros(10), "std": np.zeros(10), "min": np.zeros(10), "max": np.zeros(10),
                "count": np.array([12]), "q50": np.zeros(10)}
        self.meta = SimpleNamespace(stats={"action": like})
        self.num_frames = 12

    def __len__(self) -> int:
        return 12

    def __getitem__(self, index: int) -> dict:
        end = (index // 6 + 1) * 6
        rows = [min(index + k, end - 1) for k in range(CHUNK)]
        return {"action": torch.as_tensor(self.steps[rows]),
                "action_is_pad": torch.as_tensor([index + k >= end for k in range(CHUNK)])}


def main() -> int:
    dataset = Dataset()
    relative = Relative(dataset, CHUNK)
    item = relative[1]
    want = [0.002, 0.004, 0.006, 0.008]
    check("a chunk is taught as where the tool is k+1 steps on", np.allclose(item["action"][:, 0], want, atol=1e-6),
          str(item["action"][:, 0].tolist()))
    item = relative[4]
    check("past its episode's end the tool holds where it last was",
          np.allclose(item["action"][:, 0], [0.002, 0.004, 0.004, 0.004], atol=1e-6), str(item["action"][:, 0].tolist()))
    stats = dataset.meta.stats["action"]
    # Six steps, chunks of four: 6 + 5 + 4 + 3 entries an episode.
    check("the action's statistics are those of every entry of every chunk",
          stats["count"].tolist() == [36] and abs(stats["max"][0] - 0.008) < 1e-9 and abs(stats["min"][0] - 0.002) < 1e-9,
          f"count {stats['count'].tolist()}, x {stats['min'][0]:.3f}..{stats['max'][0]:.3f}")
    check("the rest is the dataset's", relative.num_frames == 12 and len(relative) == 12)
    again = pickle.loads(pickle.dumps(relative))
    check("it goes to a loader's worker and back", np.allclose(again[1]["action"], relative[1]["action"]))
    held_out = Dataset()
    Relative(held_out, CHUNK, like=relative)
    check("held-out episodes are scored in the training set's units",
          held_out.meta.stats["action"] is dataset.meta.stats["action"])
    check("the chunk's first entry is its first step", np.allclose(chunks.steps_of(relative[0]["action"].numpy())[0],
                                                                   dataset.steps[0], atol=1e-6))

    print("\n" + ("all checks passed" if not FAILURES else f"{len(FAILURES)} failed: {', '.join(FAILURES)}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
