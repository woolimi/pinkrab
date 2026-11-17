"""What a policy is taught to answer: each hand's chunk as a relative trajectory.

A dataset keeps one step a frame - the tool's motion from t to t+1 in the
tool's own frame, and the jaw's next opening (pinkrab_core.dataset). A policy
answers a chunk of them at once, and taught them as they are, every number of
the chunk is a single step: a millimetre or less on most frames, the size of
the tracker's noise. So a chunk is taught as UMI teaches it - its k-th entry
is where the tool is k+1 steps on, seen from where it was when the chunk
began: T_t^-1 T_{t+k+1}, the steps from t composed. Two seconds on the tool
has gone some 35 mm, and the noise is what it was. The first entry is the
step itself.

At rollout each entry is put on where the tool stood when the chunk was
asked for, so an entry the arm did not quite make is not carried into the
next: the next is set from the same place.

The entries are of very different sizes - the first a step, a millimetre;
the last two seconds' motion, tens of them - and a rollout runs only the
first few before it asks again. Normalised by statistics pooled over every
entry, those first few are taught in units of the last ones, and an error
the policy makes there of a few hundredths of such a unit is a drift of a
fraction of a millimetre every tenth of a second that nothing pulls back: a
hand at rest walks off by centimetres. So each entry is taught standardised
by its own statistics - those of the k-th entry of every chunk - and a
checkpoint carries them (MARK_FILE) for a rollout to put its answers back.

numpy only: the training machines run this file beside run_log.py with
nothing else of Pinkrab installed.
"""
from __future__ import annotations

import json

import numpy as np

# The numbers of a hand in an action: dp (3), the turn's first two columns
# (6), the jaw's opening (1).
PER_HAND = 10
# What a checkpoint trained so carries beside its weights, with each entry's
# statistics - its absence is a policy that answers steps, another mark one
# taught by another rule.
MARK_FILE = "pinkrab_action.json"
MARK = {"action": "relative trajectory, each entry standardised"}
# The least spread an entry's number is standardised by: a tenth of a
# millimetre of place, a thousandth of a rotation column's entry (some 0.06
# degrees), a hundredth of the jaw's band. Finer than these is noise, and a
# number that hardly moves - a turn's diagonal, a jaw that never opens -
# would otherwise be taught magnified a thousandfold.
FLOOR = np.array([1e-4] * 3 + [1e-3] * 6 + [1e-2])


def as_matrix(nine) -> np.ndarray:
    """A place and two rotation columns - nine numbers - as a 4x4."""
    a = np.asarray(nine, dtype=float)
    c1 = a[3:6] / max(np.linalg.norm(a[3:6]), 1e-9)
    c2 = a[6:9] - np.dot(c1, a[6:9]) * c1
    c2 = c2 / max(np.linalg.norm(c2), 1e-9)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([c1, c2, np.cross(c1, c2)])
    T[:3, 3] = a[:3]
    return T


def as_nine(T: np.ndarray) -> np.ndarray:
    """A 4x4's place and its rotation's first two columns."""
    return np.concatenate([T[:3, 3], T[:3, 0], T[:3, 1]])


def relative_chunk(steps: np.ndarray, pad: np.ndarray | None = None) -> np.ndarray:
    """A chunk of a dataset's actions (K, 10 per hand) as the relative
    trajectory it is taught as: each hand's steps composed from the chunk's
    start, the jaws as they are. An entry past the episode's end (`pad`) holds
    where the tool last was."""
    steps = np.asarray(steps, dtype=float)
    out = steps.copy()
    for at in range(0, steps.shape[1], PER_HAND):
        T = np.eye(4)
        for k in range(len(steps)):
            if pad is None or not pad[k]:
                T = T @ as_matrix(steps[k, at:at + 9])
            out[k, at:at + 9] = as_nine(T)
    return out


def _relative_episode(steps: np.ndarray, chunk: int) -> list[np.ndarray]:
    """Every entry of every chunk an episode's actions (N, 10 per hand) are
    taught as, chunks running past the end left out: the k-th entries,
    (N - k, 10 per hand), for each k."""
    n = len(steps)
    out = [np.empty((n - k, steps.shape[1])) for k in range(min(chunk, n))]
    for at in range(0, steps.shape[1], PER_HAND):
        # Where the tool is after each step, from the episode's start.
        T = np.empty((n + 1, 4, 4))
        T[0] = np.eye(4)
        for j in range(n):
            T[j + 1] = T[j] @ as_matrix(steps[j, at:at + 9])
        inverse = np.linalg.inv(T)
        for k in range(len(out)):
            starts = np.arange(n - k)
            R = inverse[starts] @ T[starts + k + 1]
            out[k][:, at:at + PER_HAND] = np.concatenate(
                [R[:, :3, 3], R[:, :3, 0], R[:, :3, 1], steps[starts + k, at + 9:at + 10]], axis=1)
    return out


def standardise(relative: np.ndarray, entries: dict) -> np.ndarray:
    """A relative trajectory (K, 10 per hand) as it is taught: each entry
    less its mean, over its spread (`entries`, the mean and spread relative_stats gives)."""
    k = len(relative)
    return (np.asarray(relative) - entries["mean"][:k]) / entries["std"][:k]


def unstandardise(taught: np.ndarray, entries: dict) -> np.ndarray:
    """A policy's answer (K, 10 per hand) back into the relative trajectory
    it stands for."""
    k = len(taught)
    return np.asarray(taught) * entries["std"][:k] + entries["mean"][:k]


def relative_stats(steps: np.ndarray, episodes: np.ndarray, chunk: int, like: dict) -> tuple[dict, dict]:
    """What a policy taught relative chunks is normalised by, from a
    dataset's actions `steps` (N, 10 per hand), each row's episode in
    `episodes`: each entry's own mean and spread (`chunk`, 10 per hand) -
    the spread no finer than FLOOR - and the statistics of every entry of
    every chunk so standardised, under the names and in the types the
    dataset's own action statistics (`like`) have them. An entry past every
    episode's end is given the last entry's."""
    steps = np.asarray(steps, dtype=float)
    episodes = np.asarray(episodes)
    per: list[list[np.ndarray]] = [[] for _ in range(chunk)]
    for e in np.unique(episodes):
        for k, rows in enumerate(_relative_episode(steps[episodes == e], chunk)):
            per[k].append(rows)
    floor = np.tile(FLOOR, steps.shape[1] // PER_HAND)
    mean = np.zeros((chunk, steps.shape[1]))
    std = np.zeros((chunk, steps.shape[1]))
    pooled = []
    for k in range(chunk):
        if not per[k]:
            mean[k], std[k] = mean[k - 1], std[k - 1]
            continue
        rows = np.concatenate(per[k])
        per[k] = []
        mean[k] = rows.mean(axis=0)
        std[k] = np.maximum(rows.std(axis=0), floor)
        pooled.append((rows - mean[k]) / std[k])
    pooled = np.concatenate(pooled)
    out = {}
    for name, value in like.items():
        if name == "mean":
            v = pooled.mean(axis=0)
        elif name == "std":
            v = pooled.std(axis=0)
        elif name == "min":
            v = pooled.min(axis=0)
        elif name == "max":
            v = pooled.max(axis=0)
        elif name == "count":
            v = np.array([len(pooled)])
        elif name.startswith("q") and name[1:].isdigit():
            v = np.quantile(pooled, int(name[1:]) / 100.0, axis=0)
        else:
            raise ValueError(f"an action statistic {name!r} that relative chunks have no rule for")
        out[name] = np.asarray(v).astype(np.asarray(value).dtype)
    return {"mean": mean, "std": std}, out


def mark(entries: dict) -> dict:
    """What MARK_FILE holds: the mark, and each entry's statistics."""
    return {**MARK, "mean": entries["mean"].tolist(), "std": entries["std"].tolist()}


def read_mark(text: str) -> dict | None:
    """Each entry's statistics from MARK_FILE's `text` - None for a file
    that is not this mark."""
    try:
        held = json.loads(text)
        if any(held.get(k) != v for k, v in MARK.items()):
            return None
        return {"mean": np.asarray(held["mean"], dtype=float), "std": np.asarray(held["std"], dtype=float)}
    except (ValueError, KeyError, TypeError, AttributeError):
        return None


def steps_of(relative: np.ndarray) -> np.ndarray:
    """A relative trajectory (K, 10 per hand) back into the steps that compose
    it - the first entry is the first step - the jaws as they are."""
    relative = np.asarray(relative, dtype=float)
    out = relative.copy()
    for at in range(0, relative.shape[1], PER_HAND):
        before = np.eye(4)
        for k in range(len(relative)):
            here = as_matrix(relative[k, at:at + 9])
            out[k, at:at + 9] = as_nine(np.linalg.inv(before) @ here)
            before = here
    return out


def mean_pose(poses: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """The weighted mean of 4x4 poses: the places averaged, the rotations'
    weighted sum taken to the nearest rotation."""
    w = np.asarray(weights, dtype=float) / float(np.sum(weights))
    P = np.asarray(poses, dtype=float)
    U, _, Vt = np.linalg.svd(np.einsum("i,ijk->jk", w, P[:, :3, :3]))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = w @ P[:, :3, 3]
    return T
