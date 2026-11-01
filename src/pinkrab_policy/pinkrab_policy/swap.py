"""`pinkrab-picture-swap` - whether a checkpoint looks at its pictures.

Frames are drawn from the dataset the checkpoint was trained on, and each is
asked about fresh - no queued chunk - several ways: as recorded, with both
hands' pictures taken from another frame, with the state of another frame,
and with nothing swapped against the recorded action. Each hand's answer is
compared by the cosine between the directions its tool is asked to move.

A policy that learnt the task moves with its pictures; one that answers
from its state moves with the state and hardly at all with the pictures,
and at rollout drives itself off in whatever direction it began.
The pictures are swapped together, so a hand that leans on the other hand's
camera is caught as well.

A policy that reads a command, trained on a dataset of several, is asked
once more: with another of the dataset's commands. Near +1 is a policy that
does what the scene suggests whatever it is told.

    pinkrab-picture-swap ~/pinkrab/train/<run>/checkpoints/last/pretrained_model
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from pinkrab_core.config import SETTINGS as CORE
from pinkrab_core.dataset import action_names


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def picture_swap(checkpoint: Path, frames: int = 150, seed: int = 0, device: str | None = None) -> dict:
    """Each hand's answers with its pictures, another frame's pictures, another
    frame's state and another command, compared - see the module's doc."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from pinkrab_core.demo_start import checkpoint_tasks, dataset_of_checkpoint

    from .rollout import Brain

    tasks = checkpoint_tasks(Path(checkpoint))
    # Each frame is asked with its own command; this one only lets a policy
    # trained on several load.
    brain = Brain(str(checkpoint), device, task=tasks[0] if tasks else None)
    root = dataset_of_checkpoint(Path(checkpoint))
    dataset = LeRobotDataset(f"pinkrab/{root.name}", root=root)
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(dataset), size=min(frames, len(dataset)), replace=False)
    others = rng.permutation(picks)
    keys = list(brain.sizes)
    names = action_names(brain.hands)
    # Where each hand's step sits in the action.
    moves = {hand: [names.index(f"{hand}_{n}") for n in ("dx", "dy", "dz")] for hand in brain.hands}

    def item(i):
        x = dataset[int(i)]
        pictures = {k: (x[k].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8) for k in keys}
        return pictures, x["observation.state"].numpy(), x["action"].numpy(), x["task"]

    def ask(pictures, state, task):
        brain.reset()
        return brain.ask(pictures, state, task)

    told = brain.language and len(tasks) > 1
    rows = {hand: {"recorded": [], "picture": [], "state": [], "command": [], "mm": [],
                   "recorded_mm": []} for hand in brain.hands}
    for i, j in zip(picks, others):
        pictures, state, action, task = item(i)
        other_pictures, other_state, _, _ = item(j)
        base = ask(pictures, state, task)
        swapped = ask(other_pictures, state, task)
        restated = ask(pictures, other_state, task)
        retold = ask(pictures, state, str(rng.choice([t for t in tasks if t != task]))) if told else None
        for hand, at in moves.items():
            step = base[at]
            rows[hand]["recorded"].append(_cos(step, action[at]))
            rows[hand]["picture"].append(_cos(step, swapped[at]))
            rows[hand]["state"].append(_cos(step, restated[at]))
            if retold is not None:
                rows[hand]["command"].append(_cos(step, retold[at]))
            rows[hand]["mm"].append(float(np.linalg.norm(step) * 1000))
            rows[hand]["recorded_mm"].append(float(np.linalg.norm(action[at]) * 1000))

    def mean(values):
        return float(np.mean(values)) if values else None

    hands = {hand: {"cos_to_recorded": mean(r["recorded"]), "cos_picture_swapped": mean(r["picture"]),
                    "cos_state_swapped": mean(r["state"]), "cos_command_swapped": mean(r["command"]),
                    "step_mm": mean(r["mm"]), "recorded_step_mm": mean(r["recorded_mm"])}
             for hand, r in rows.items()}
    looks = all(h["cos_picture_swapped"] <= CORE.swap_max_picture_cos for h in hands.values())
    return {"checkpoint": str(checkpoint), "dataset": str(root), "frames": len(picks),
            "state_dim": brain.state_dim, "hands": hands, "looks_at_pictures": looks,
            "max_picture_cos": CORE.swap_max_picture_cos}


def report(result: dict, say=print) -> None:
    say(f"  {result['frames']} frames of {result['dataset']}, state {result['state_dim']}")
    for hand, h in result["hands"].items():
        say(f"  {hand} hand")
        say(f"    answer vs the recorded step       cos {h['cos_to_recorded']:+.2f}")
        say(f"    answer vs with other pictures     cos {h['cos_picture_swapped']:+.2f}   "
            "(near +1: the pictures are not looked at)")
        say(f"    answer vs with another state      cos {h['cos_state_swapped']:+.2f}")
        if h["cos_command_swapped"] is not None:
            say(f"    answer vs with another command    cos {h['cos_command_swapped']:+.2f}   "
                "(near +1: the command is not listened to)")
        say(f"    step {h['step_mm']:.1f} mm asked, {h['recorded_step_mm']:.1f} mm recorded")
    say(f"  {'looks at its pictures' if result['looks_at_pictures'] else 'does NOT look at its pictures'}"
        f" (pictures swapped at most cos {result['max_picture_cos']:+.2f} on every hand)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pinkrab-picture-swap", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="a pretrained_model directory")
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--json", action="store_true", help="the numbers as JSON")
    args = parser.parse_args(argv)
    result = picture_swap(Path(args.checkpoint).expanduser(), args.frames, args.seed, args.device)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        report(result)
    return 0 if result["looks_at_pictures"] else 1


if __name__ == "__main__":
    sys.exit(main())
