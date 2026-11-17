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

numpy only: the training machines run this file beside run_log.py with
nothing else of Pinkrab installed.
"""
from __future__ import annotations

import numpy as np

# The numbers of a hand in an action: dp (3), the turn's first two columns
# (6), the jaw's opening (1).
PER_HAND = 10
# What a checkpoint trained so carries beside its weights - its absence is a
# policy that answers steps.
MARK_FILE = "pinkrab_action.json"
MARK = {"action": "relative trajectory"}


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


def _relative_episode(steps: np.ndarray, chunk: int) -> np.ndarray:
    """Every entry of every chunk an episode's actions (N, 10 per hand) are
    taught as, chunks running past the end left out: (n, 10 per hand)."""
    n = len(steps)
    rows = []
    for at in range(0, steps.shape[1], PER_HAND):
        # Where the tool is after each step, from the episode's start.
        T = np.empty((n + 1, 4, 4))
        T[0] = np.eye(4)
        for j in range(n):
            T[j + 1] = T[j] @ as_matrix(steps[j, at:at + 9])
        inverse = np.linalg.inv(T)
        hand = []
        for k in range(min(chunk, n)):
            starts = np.arange(n - k)
            R = inverse[starts] @ T[starts + k + 1]
            hand.append(np.concatenate([R[:, :3, 3], R[:, :3, 0], R[:, :3, 1],
                                        steps[starts + k, at + 9:at + 10]], axis=1))
        rows.append(np.concatenate(hand))
    return np.concatenate(rows, axis=1)


def relative_stats(steps: np.ndarray, episodes: np.ndarray, chunk: int, like: dict) -> dict:
    """The statistics a policy taught relative chunks is normalised by: those
    of every entry of every chunk of `chunk` entries in the dataset - its
    actions `steps` (N, 10 per hand), each row's episode in `episodes` -
    under the names and in the types the dataset's own action statistics
    (`like`) have them."""
    steps = np.asarray(steps, dtype=float)
    episodes = np.asarray(episodes)
    pooled = np.concatenate([_relative_episode(steps[episodes == e], chunk) for e in np.unique(episodes)])
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
    return out


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
