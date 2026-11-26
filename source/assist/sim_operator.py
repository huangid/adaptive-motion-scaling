"""TEST ONLY: a scripted stand-in for the human at the SpaceMouse.

Produces SpaceMouse-like raw input (each axis in [-1, 1]) that points at the next goal of
the tube task (from ``predictors.tube_phase``), pushes harder the further away it is, adds
a little hand jitter, and presses the buttons to grasp and release. The robot motion then
goes through the same gain as a real operator's input, so the assistance loop can be
exercised without anyone at the controls.

It mirrors the parts of ``SpaceMousePilot`` the assisted-teleop runner uses:
``get_actions(..., pos_scale=)``, ``last_raw_pos``, ``is_idle``, ``grip_events``, ``reset``.
"""

import numpy as np
import torch

from .predictors import tube_phase


class SimulatedOperator:
    def __init__(self, env_u, push_gain: float = 5.0, jitter: float = 0.03, seed: int = 0):
        self.env_u = env_u
        self.push_gain = push_gain       # raw input per metre of error, before clipping to [-1, 1]
        self.jitter = jitter             # std of raw-input noise
        self.rng = np.random.default_rng(seed)
        self.pos_scale = 0.15
        self.reset()

    def reset(self):
        self._grip = -1.0
        self.grip_events = 0
        self.last_raw_pos = np.zeros(3)
        self.is_idle = True
        self.phase = ""
        self._cmd_quat = None       # held orientation, like SpaceMousePilot(hold_orientation=True)

    def _press(self, value):
        if value != self._grip:
            self._grip = value
            self.grip_events += 1

    def get_actions(self, eidx, pos, quat=None, grip=None, verbose=False, pos_scale=None):
        gain = self.pos_scale if pos_scale is None else float(pos_scale)
        target, _precise, self.phase, seated = tube_phase(self.env_u)
        tip = pos[0].detach().cpu().numpy()
        err = target - tip

        raw = np.clip(self.push_gain * err, -1.0, 1.0) + self.rng.normal(0.0, self.jitter, 3)
        raw = np.clip(raw, -1.0, 1.0)
        raw[np.abs(raw) < 0.005 / self.pos_scale] = 0.0
        if self.phase == "grasp" and np.linalg.norm(err) < 0.006:
            self._press(1.0)                       # fingers around the tube: close
        if self.phase == "insert" and seated:
            self._press(-1.0)                      # tube down in the hole: release
        self.last_raw_pos = raw
        self.is_idle = bool((raw == 0.0).all())

        out = torch.zeros((pos.shape[0], 8), device=pos.device)
        out[:, 0:3] = pos + torch.tensor(raw * gain, dtype=torch.float32, device=pos.device)
        if self._cmd_quat is None:
            self._cmd_quat = (quat.detach().clone() if quat is not None
                              else torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=pos.device))
        out[:, 3:7] = self._cmd_quat
        out[:, 7] = self._grip
        return out
