import os
import sys
import threading
import time
import torch
import numpy as np
from scipy.spatial.transform import Rotation
import hid

# Per-step axis readout. Off by default — at 15 Hz it prints ~1400 lines per
# episode, which scrolls the task instruction off screen during demo collection.
# Re-enable with SM_DEBUG=1.
_SM_DEBUG = bool(os.environ.get("SM_DEBUG"))



def _to_int16(lo, hi):
    v = lo | (hi << 8)
    return v - 65536 if v >= 32768 else v



class SpaceMousePilot:
    def __init__(self, cfg_path=None, data_path=None, num_envs=1,
                 device="cpu", replay_mode=False,
                 pos_scale=0.15, rot_scale=0.3, vendor=0x256f, product=0xc635,
                 hold_orientation=False):
        """
        Args:
            hold_orientation: integrate rotation input onto the pilot's own commanded
                orientation instead of the measured one. With False (the original
                behaviour) the target orientation is re-anchored to wherever the gripper
                is each step, so any disturbance (a contact, controller compliance) is
                never undone and the gripper drifts: on the tube task it tilted ~2.5 deg
                every 4 s while held. True keeps the commanded orientation fixed unless
                the operator rotates, and the arm is pulled back to it.
        """
        self._device = device
        self.hold_orientation = hold_orientation
        self._cmd_quat = None
        self._num_envs = num_envs
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale
        self._pos = np.zeros(3)
        self._rot = np.zeros(3)
        self._grip = -1.0
        # Number of gripper button presses seen. Monotonic; read it to detect that the
        # operator touched the gripper at all (see _reader).
        self.grip_events = 0
        self._prev_button_mask = 0
        self._lock = threading.Lock()
        self._dev = hid.device()
        self._dev.open(vendor, product)
        self._dev.set_nonblocking(True)
        print("SpaceMousePilot opened:", self._dev.get_product_string(),
              file=sys.stderr, flush=True)
        self._run = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while self._run:
            data = self._dev.read(64)
            if data:
                rid = data[0]
                if rid == 1 and len(data) >= 7:
                    x = -_to_int16(data[1], data[2]) / 350.0
                    y = _to_int16(data[3], data[4]) / 350.0
                    z = -_to_int16(data[5], data[6]) / 350.0
                    with self._lock:
                        self._pos = np.array([x, y, z])
                elif rid == 2 and len(data) >= 7:
                    rx = -_to_int16(data[1], data[2]) / 350.0
                    ry = _to_int16(data[3], data[4]) / 350.0
                    rz = -_to_int16(data[5], data[6]) / 350.0
                    with self._lock:
                        self._rot = np.array([rx, ry, rz])
                elif rid == 3:
                    mask = data[1]
                    with self._lock:
                        if mask & 0x01:         # left button -> close (big grip closed on object)
                            self._grip = 1.0
                        elif mask & 0x02:       # right button -> open (release)
                            self._grip = -1.0
                        # Count button EVENTS, not changes of the resulting state. A caller
                        # that watches only `_grip` cannot see the operator press "open"
                        # while the gripper is already open — which is exactly what a human
                        # does first when the robot has closed on nothing. Shared autonomy
                        # uses this counter to decide when the human has taken the gripper.
                        if mask != self._prev_button_mask:
                            if mask:
                                self.grip_events += 1
                            self._prev_button_mask = mask
            else:
                time.sleep(0.002)

    def reset(self):
        with self._lock:
            self._pos = np.zeros(3)
            self._rot = np.zeros(3)
        self._cmd_quat = None       # re-anchored to the measured orientation on the next call

    def clear(self, env_ids=None):
        self.reset()

    def get_actions(self, eidx, pos, quat=None, grip=None, verbose=False, pos_scale=None):
        """Target pose = current pose + scaled SpaceMouse input.

        ``pos_scale`` overrides the translation gain for this call (motion scaling);
        None keeps ``self.pos_scale``. The dead zone is applied to the raw input, so it
        drops the same small pushes whatever the gain (at the default gain this is
        identical to applying 0.005 m to the scaled offset). The raw input is kept in
        ``last_raw_pos`` for logging.
        """
        N = pos.shape[0]
        gain = self.pos_scale if pos_scale is None else float(pos_scale)
        with self._lock:
            raw = self._pos.copy()
            drot = self._rot * self.rot_scale
            gval = self._grip

        dz = 0.005
        raw[np.abs(raw) < dz / self.pos_scale] = 0.0
        self.last_raw_pos = raw
        dpos = torch.tensor(raw * gain, dtype=torch.float32, device=self._device)

        if _SM_DEBUG:
            print("SM cmd:", dpos.cpu().numpy(), drot, file=sys.stderr, flush=True)
        drot_arr = np.where(np.abs(drot) < dz, 0.0, drot)
        # Idle = no translation input past the dead zone (judged on the raw input, so a low
        # gain cannot make a real push look idle) and no rotation.
        self.is_idle = bool((raw == 0.0).all()) and bool((np.abs(drot_arr) < dz).all())

        new_pos = pos + dpos.unsqueeze(0).expand(N, -1)

        if quat is None:
            quat = torch.zeros((N, 4), device=self._device)
            quat[:, 0] = 1.0

        if self.hold_orientation:
            if self._cmd_quat is None or self._cmd_quat.shape[0] != N:
                self._cmd_quat = quat.detach().clone().to(self._device)
            base_quat = self._cmd_quat
        else:
            base_quat = quat

        dq = Rotation.from_rotvec(drot_arr).as_quat()
        dq = torch.tensor([dq[3], dq[0], dq[1], dq[2]],
                          dtype=torch.float32, device=self._device)
        new_quat = self._quat_mul(dq.unsqueeze(0).expand(N, -1), base_quat)
        new_quat = new_quat / new_quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        if self.hold_orientation:
            self._cmd_quat = new_quat.detach().clone()

        new_grip = torch.full((N, 1), gval, dtype=torch.float32, device=self._device)
        return torch.cat([new_pos, new_quat, new_grip], dim=-1)

    @staticmethod
    def _quat_mul(a, b):
        aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
        return torch.stack([
            aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
        ], dim=-1)
