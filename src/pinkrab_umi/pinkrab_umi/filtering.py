"""The stream-level judge of single-face poses: which of the two, or neither.

A single visible face is a plane, and a plane in one image fits two poses -
mirrored about the image plane, tens of degrees apart, with residuals that
converge as the face comes square to the camera. Nothing in that frame can tell
them apart; the frames around it can, because a solid does not turn eighty degrees
in a thirtieth of a second. Unfiltered, a still solid reads rotation spikes of
85 degrees whenever a face sits near face-on, and a two-face view that sheds one
face at the area gate flips the same way.

So the filter runs where time exists - over a stream - and does three things. A
frame with two or more faces passes untouched: non-coplanar faces pin the pose
outright, and each one refreshes the anchor. A single-face frame whose two
residuals clearly rank picks its own winner. A single-face frame whose residuals
do not rank is decided by the anchor - the accepted pose moments ago - and when
the anchor is too old to vouch for anything, the frame is refused rather than
guessed at. Refused frames become holes, and holes are visible downstream;
solid_track already drops them rather than interpolating across them.

Then the accepted pose is smoothed. A solved pose is a measurement, and at arm's
length the measurement is about a millimetre and half a degree of noise frame to
frame - and half a degree at the marker is more than a millimetre at the tool
centre, 139 mm away on its lever. An exponential average takes a fixed fraction
of each new reading: at 0.3 and 30 Hz it settles in three frames, under the
time it takes to see, and the standing noise comes out at less than half. The
average is restarted whenever the stream has been dark longer than the anchor
lives - a pose from before a gap is not something to be pulled toward.

One instance is one stream. The live view runs one over what it shows, and the
verdict of a take - and bench over a session - runs a fresh one from the first
frame, so the same frames in the same order give the same answer everywhere.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .config import SETTINGS
from .pnp import SolidPose
from .trajectory import matrix_to_quat, quat_to_matrix, rotation_angle_deg, slerp


def _apart_deg(a: np.ndarray, b: np.ndarray) -> float:
    """How far two poses' rotations sit apart, in degrees."""
    return rotation_angle_deg(a[:3, :3].T @ b[:3, :3])


def _blend(previous: np.ndarray, fresh: np.ndarray, alpha: float) -> np.ndarray:
    """`previous` moved `alpha` of the way to `fresh`: lerp on position, slerp on rotation."""
    out = np.eye(4)
    out[:3, :3] = quat_to_matrix(slerp(matrix_to_quat(previous[:3, :3]),
                                       matrix_to_quat(fresh[:3, :3]), alpha))
    out[:3, 3] = previous[:3, 3] + alpha * (fresh[:3, 3] - previous[:3, 3])
    return out


class PoseFilter:
    """Causal and stateful: feed it the stream in order, once."""

    def __init__(self) -> None:
        self._anchor: np.ndarray | None = None
        self._anchor_t: float = float("-inf")

    def _accept(self, result: SolidPose, t: float, pose: np.ndarray) -> SolidPose:
        # The anchor is the smoothed pose: it is the better estimate of where
        # the solid is, which is what the next frame's judgement wants.
        if self._anchor is not None and t - self._anchor_t <= SETTINGS.filter_anchor_max_age_s:
            pose = _blend(self._anchor, pose, SETTINGS.filter_ema_alpha)
        self._anchor = pose
        self._anchor_t = t
        if pose is result.pose:
            return result
        return replace(result, pose=pose)

    def apply(self, result: SolidPose, t: float) -> SolidPose:
        """One frame in, one frame out - possibly the other solution, possibly refused."""
        if not result.ok or result.reason:
            # Nothing solved, or the reprojection gate already flagged it. Not this
            # filter's case to judge, and not fit to anchor anything either.
            return result

        if len(result.faces) >= 2:
            return self._accept(result, t, result.pose)

        anchor_fresh = t - self._anchor_t <= SETTINGS.filter_anchor_max_age_s

        if result.alternate is None or (
            result.alternate_reproj_px
            > SETTINGS.filter_ambiguity_ratio * max(result.reproj_px, 0.05)
        ):
            # One solution, or two that rank clearly: the frame answers for itself.
            pose = result.pose
        elif anchor_fresh:
            # Two answers the residuals cannot rank; the recent past can.
            pose = min((result.pose, result.alternate),
                       key=lambda p: _apart_deg(self._anchor, p))
        else:
            return replace(result, ok=False, reason="single face, ambiguous, no recent anchor")

        if anchor_fresh:
            # Even the chosen answer has to be reachable from where the solid just
            # was. A hand can flick, so the allowance is generous - what it refuses
            # is teleportation, which is what a flip is.
            step = _apart_deg(self._anchor, pose)
            allowed = SETTINGS.filter_max_rot_speed_dps * max(t - self._anchor_t, 1 / SETTINGS.fps)
            if step > allowed:
                return replace(result, ok=False,
                               reason=f"rotation jump {step:.0f} deg - both solutions implausible")

        return self._accept(result, t, pose)
