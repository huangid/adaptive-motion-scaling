"""SpaceMouse motion scaling: fixed, or adapted from a policy's predicted action chunk.

The operator's input sets the direction and how hard they push; the scaler sets how far
that push moves the robot. A gain of g means a full-deflection push asks the fingertip
to go g metres from where it is now (the SpaceMouse pilot's ``pos_scale``).

Adaptive scaling reads how fast the policy *plans* to move: the mean distance per step
over the start of its predicted chunk. Fast plans (carrying across the table) give a
high gain, slow plans (grasping, inserting) a low one. Only the magnitude changes; the
direction always comes from the operator.
"""

from dataclasses import dataclass

import numpy as np


def chunk_speed(chunk_pos, current_pos, k: int) -> float:
    """Mean distance between consecutive targets over the first ``k`` steps of a chunk.

    Only step-to-step changes *within* the plan count. Teleop actions, and so a policy
    trained on them, are targets set some distance ahead of the fingertip (the scaled
    SpaceMouse offset, up to 15 cm), so the jump from the current position to the first
    target measures that lead, not speed. ``current_pos`` is used only when the chunk has
    a single target.

    Args:
        chunk_pos: (H, 3) absolute fingertip targets, in metres.
        current_pos: (3,) fingertip position now.
        k: how many steps of the chunk to look at.
    """
    pts = np.asarray(chunk_pos, dtype=float)[:k + 1, :3]
    if len(pts) < 2:
        pts = np.vstack([np.asarray(current_pos, dtype=float).reshape(1, 3), pts])
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).mean())


@dataclass
class FixedScaler:
    """The baseline: one gain for the whole episode."""

    gain: float = 0.15

    def reset(self):
        pass

    def update(self, chunk_pos=None, current_pos=None) -> float:
        return self.gain


@dataclass
class AdaptiveScaler:
    """Gain from the policy's planned speed, linear between a slow and a fast speed.

    planned speed <= slow_speed -> gain_min (precise)
    planned speed >= fast_speed -> gain_max (coarse)
    in between                  -> linear interpolation

    An exponential moving average (``smoothing`` = weight of the old value) keeps the
    gain from jumping when the plan is replanned. Speeds are metres per control step
    (15 Hz); the defaults are starting points to tune on real demos.
    """

    gain_min: float = 0.04
    gain_max: float = 0.20
    slow_speed: float = 0.001
    fast_speed: float = 0.006
    lookahead: int = 10
    smoothing: float = 0.7
    _gain: float | None = None
    last_speed: float = 0.0

    def __post_init__(self):
        if not 0.0 < self.gain_min <= self.gain_max:
            raise ValueError("need 0 < gain_min <= gain_max")
        if not 0.0 <= self.slow_speed < self.fast_speed:
            raise ValueError("need 0 <= slow_speed < fast_speed")
        if not 0.0 <= self.smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")

    def reset(self):
        self._gain = None
        self.last_speed = 0.0

    def target_gain(self, speed: float) -> float:
        frac = (speed - self.slow_speed) / (self.fast_speed - self.slow_speed)
        return self.gain_min + float(np.clip(frac, 0.0, 1.0)) * (self.gain_max - self.gain_min)

    def update(self, chunk_pos, current_pos) -> float:
        self.last_speed = chunk_speed(chunk_pos, current_pos, self.lookahead)
        target = self.target_gain(self.last_speed)
        if self._gain is None:
            self._gain = target
        else:
            self._gain = self.smoothing * self._gain + (1.0 - self.smoothing) * target
        return self._gain
