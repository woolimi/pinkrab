"""What pinkrab-train teaches (run_log.Relative), on a dataset made here.

    python src/pinkrab_policy/test/relative_check.py
"""
from __future__ import annotations

import json
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
    entries = relative.entries
    taught = lambda i: chunks.unstandardise(relative[i]["action"].numpy(), entries)  # noqa: E731
    check("each entry's statistics are its own", np.allclose(entries["mean"][:, 0], [0.002, 0.004, 0.006, 0.008]),
          str(entries["mean"][:, 0].tolist()))
    check("a spread that never moves is standardised by the floor", np.allclose(entries["std"][:, 0], chunks.FLOOR[0]),
          str(entries["std"][:, 0].tolist()))
    item = taught(1)
    want = [0.002, 0.004, 0.006, 0.008]
    check("a chunk is taught as where the tool is k+1 steps on", np.allclose(item[:, 0], want, atol=1e-6),
          str(item[:, 0].tolist()))
    check("each entry is taught standardised", np.allclose(relative[1]["action"][:, 0], 0, atol=1e-3),
          str(relative[1]["action"][:, 0].tolist()))
    item = taught(4)
    check("past its episode's end the tool holds where it last was",
          np.allclose(item[:, 0], [0.002, 0.004, 0.004, 0.004], atol=1e-6), str(item[:, 0].tolist()))
    stats = dataset.meta.stats["action"]
    # Six steps, chunks of four: 6 + 5 + 4 + 3 entries an episode.
    check("the action's statistics are those of every entry of every chunk, standardised",
          stats["count"].tolist() == [36] and abs(stats["max"][0]) < 1e-6 and abs(stats["min"][0]) < 1e-6,
          f"count {stats['count'].tolist()}, x {stats['min'][0]:.3f}..{stats['max'][0]:.3f}")
    check("the rest is the dataset's", relative.num_frames == 12 and len(relative) == 12)
    again = pickle.loads(pickle.dumps(relative))
    check("it goes to a loader's worker and back", np.allclose(again[1]["action"], relative[1]["action"]))
    held_out = Dataset()
    other = Relative(held_out, CHUNK, like=relative)
    check("held-out episodes are scored in the training set's units",
          held_out.meta.stats["action"] is dataset.meta.stats["action"] and other.entries is entries)
    check("the chunk's first entry is its first step", np.allclose(chunks.steps_of(taught(0))[0],
                                                                   dataset.steps[0], atol=1e-6))
    read = chunks.read_mark(json.dumps(chunks.mark(Relative.entries)))
    check("a checkpoint's mark carries the entries' statistics",
          read is not None and np.allclose(read["mean"], entries["mean"]) and np.allclose(read["std"], entries["std"]))
    check("a mark of another rule is refused", chunks.read_mark(json.dumps({"action": "relative trajectory"})) is None)

    print("\n" + ("all checks passed" if not FAILURES else f"{len(FAILURES)} failed: {', '.join(FAILURES)}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
