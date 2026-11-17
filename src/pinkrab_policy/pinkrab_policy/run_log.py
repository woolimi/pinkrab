"""`pinkrab-train` - lerobot-train, with every logged number kept in a file.

lerobot-train prints its metrics rounded for a terminal (`loss:0.081`, a
step shown as `1K`) and sends the exact ones only to Weights & Biases. This
stands in the place lerobot-train keeps for its wandb logger and writes what
it would have sent, unrounded, one JSON object per line, to
`<output_dir>/metrics.jsonl`:

    {"kind": "run", ...}          once, first: what is being trained on what
    {"kind": "metrics", ...}      every log_freq steps: loss, grad_norm, lr, ...
    {"kind": "checkpoint", ...}   every checkpoint saved
    {"kind": "end", ...}          once, last: done or failed, and why

The studio's training page reads that file - from this machine, or over ssh
from a cloud box - and draws it. The arguments are lerobot-train's own,
passed through.

One thing it changes: what the policy is taught to answer. A Pinkrab dataset
keeps a step a frame, and each chunk the policy learns is given to it as the
relative trajectory those steps compose (pinkrab_core.chunks) - each
statistic the action is normalised by taken over such chunks - and every
checkpoint saved is marked so (chunks.MARK_FILE): a rollout refuses one
without the mark, a policy that answers steps.

Standard library only until `main()` imports lerobot, so this file and
chunks.py can be copied to a machine that has lerobot and nothing of this
project:

    pinkrab-train --policy.type=act --dataset.repo_id=... --output_dir=...
    python run_log.py --policy.type=act ...          # the same, as a copied file
"""
from __future__ import annotations

import json
import math
import signal
import socket
import sys
import time
from pathlib import Path

FILE_NAME = "metrics.jsonl"


def _plain(value):
    """A JSON-safe copy: numbers stay numbers, NaN and inf become null, the
    rest (paths, enums) becomes its string."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    try:
        return float(value) if math.isfinite(float(value)) else None  # a 0-d tensor, a numpy scalar
    except (TypeError, ValueError):
        return str(value)


class RunLog:
    """What lerobot-train calls its WandBLogger, writing to a file instead.

    The last one made is kept on the class, so `main()` can write the closing
    line whether training returned or raised.
    """

    current: RunLog | None = None

    def __init__(self, cfg) -> None:
        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.path = out / FILE_NAME
        self._file = open(self.path, "a", buffering=1)
        policy = getattr(cfg, "policy", None)
        dataset = getattr(cfg, "dataset", None)
        try:
            config = cfg.to_dict()
        except Exception:  # noqa: BLE001 - a config that will not serialise still trains
            config = None
        self._write({
            "kind": "run",
            "name": cfg.job_name or out.name,
            "output_dir": str(out.resolve()),
            "policy": getattr(policy, "type", None),
            "dataset": getattr(dataset, "repo_id", None),
            "steps": cfg.steps,
            "batch_size": cfg.batch_size,
            "log_freq": cfg.log_freq,
            "save_freq": cfg.save_freq,
            "resume": bool(getattr(cfg, "resume", False)),
            "host": socket.gethostname(),
            "config": config,
        })
        RunLog.current = self

    def _write(self, line: dict) -> None:
        line = {"t": time.time(), **line}
        self._file.write(json.dumps(_plain(line), separators=(",", ":")) + "\n")

    # --- the logger lerobot-train calls -------------------------------------
    def log_dict(self, d: dict, step: int | None = None, mode: str = "train",
                 custom_step_key: str | None = None) -> None:
        values = {k: v for k, v in _plain(d).items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        self._write({"kind": "metrics", "mode": mode, "step": step, "values": values})

    def log_policy(self, checkpoint_dir) -> None:
        path = Path(checkpoint_dir)
        chunks = _chunks()
        (path / "pretrained_model" / chunks.MARK_FILE).write_text(json.dumps(chunks.MARK))
        step = int(path.name) if path.name.isdigit() else None
        self._write({"kind": "checkpoint", "step": step, "path": str(path.resolve())})

    def log_video(self, video_path, step: int | None = None, mode: str = "train") -> None:
        pass

    # --- the end ------------------------------------------------------------
    def close(self, status: str, error: str | None = None) -> None:
        self._write({"kind": "end", "status": status, "error": error})
        self._file.close()


def _interrupt(signum, _frame):
    raise KeyboardInterrupt(signal.Signals(signum).name)


def _chunks():
    """pinkrab_core.chunks: the installed one here, the copy sent beside
    this file on a training machine."""
    try:
        from pinkrab_core import chunks
    except ImportError:
        import chunks
    return chunks


class Relative:
    """A LeRobotDataset whose items carry their action chunk as the relative
    trajectory its steps compose. Its action statistics become those of such
    chunks of `chunk` entries over its own episodes - or `like`'s, another
    Relative's, so held-out episodes are scored in the training set's units.
    Everything else is the dataset's."""

    def __init__(self, dataset, chunk: int, like: Relative | None = None) -> None:
        import numpy as np

        self.dataset = dataset
        stats = dataset.meta.stats
        if like is not None:
            stats["action"] = like.dataset.meta.stats["action"]
            return
        table = dataset.hf_dataset.data
        steps = np.asarray(table.column("action").to_pylist(), dtype=float)
        episodes = np.asarray(table.column("episode_index").to_numpy())
        stats["action"] = _chunks().relative_stats(steps, episodes, chunk, stats["action"])

    def __getattr__(self, name):
        if name == "dataset":    # unpickled in a worker, before __init__'s fields
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        import torch

        item = self.dataset[index]
        pad = item.get("action_is_pad")
        relative = _chunks().relative_chunk(item["action"].numpy(), None if pad is None else pad.numpy())
        item["action"] = torch.as_tensor(relative, dtype=item["action"].dtype)
        return item


def _teach_relative(trainer) -> None:
    """lerobot-train's datasets made Relative, at the chunk its policy has."""
    make = trainer.make_train_eval_datasets

    def made(cfg):
        dataset, eval_dataset = make(cfg)
        chunk = int(cfg.policy.chunk_size)
        train = Relative(dataset, chunk)
        return train, None if eval_dataset is None else Relative(eval_dataset, chunk, like=train)

    trainer.make_train_eval_datasets = made


def main() -> None:
    import lerobot.scripts.lerobot_train as trainer

    # A kill is a Ctrl-C here, so a run stopped from outside still says so on
    # its last line instead of going quiet like one that crashed the machine.
    signal.signal(signal.SIGTERM, _interrupt)

    # lerobot-train makes its wandb logger only when asked for one with a
    # project name; this is asked for, and made in its place.
    trainer.WandBLogger = RunLog
    _teach_relative(trainer)
    args = [a for a in sys.argv[1:] if not a.startswith(("--wandb.enable", "--wandb.project"))]
    sys.argv = [sys.argv[0], *args, "--wandb.enable=true", "--wandb.project=pinkrab"]
    try:
        trainer.main()
    except BaseException as exc:
        if RunLog.current is not None:
            stopped = isinstance(exc, KeyboardInterrupt)
            RunLog.current.close("stopped" if stopped else "failed", f"{type(exc).__name__}: {exc}".strip(": "))
        raise
    if RunLog.current is not None:
        RunLog.current.close("done")


if __name__ == "__main__":
    main()
