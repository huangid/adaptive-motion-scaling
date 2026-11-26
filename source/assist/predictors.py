"""Where the predicted action chunk comes from: the trained SmolVLA, or a test stand-in.

Both expose ``chunk(env_u) -> (H, 8)`` absolute fingertip targets (pos3 + quat4 +
gripper1) for env 0, replanning every ``replan_every`` steps and advancing through the
current plan in between, as a deployed chunking policy does.

``OraclePredictor`` is for testing only. It plans from the simulator's true object
positions (which the real policy has to infer from images), so the assistance pipeline
can be exercised before a policy is trained. It must never be used in the study.
"""

import numpy as np
import torch

_DOWN = np.array([0.0, 1.0, 0.0, 0.0])   # gripper pointing straight down (wxyz)


class _ChunkCache:
    """Replan every ``replan_every`` steps; in between, step through the last plan."""

    def __init__(self, replan_every: int):
        self.replan_every = max(1, int(replan_every))
        self.reset()

    def reset(self):
        self._plan = None
        self._age = 0

    def chunk(self, env_u):
        if self._plan is None or self._age >= self.replan_every or self._age >= len(self._plan) - 1:
            self._plan = self._predict(env_u)
            self._age = 0
        out = self._plan[self._age:]
        self._age += 1
        return out

    def _predict(self, env_u):
        raise NotImplementedError


# ---------------------------------------------------------------------------------------
# Task phases from the simulator's true state (shared by the oracle and the test operator)
# ---------------------------------------------------------------------------------------

def tube_phase(env_u, near_radius: float = 0.05):
    """(target fingertip pos (3,), precise: bool, phase name, seated: bool) for env 0.

    Not held: go above the tube, then descend to grasp it.
    Held:     pull it straight up out of the rack, carry it high, then line the tube's
              bottom (not the fingertip: the tube can sit off-centre in the fingers) up
              over the destination hole and lower it in. Phases within ``near_radius``
              (xy) of the tube or hole, and the pull-out, are precise; the rest coarse.
    ``seated`` is True once the tube's bottom is down in the hole (time to let go).
    """
    t = env_u.cfg_task
    tip = env_u.fingertip_midpoint_pos[0].detach().cpu().numpy()
    tube = env_u.held_pos[0].detach().cpu().numpy()
    bottom, hole_floor = (v[0].detach().cpu().numpy() for v in env_u._get_held_target_pos())
    closed = float(env_u.gripper[0, 0]) > 0.5
    z_table = t.table_top_z
    grasp_z = z_table + t.tube_height - 0.035
    travel_z = z_table + 0.27
    rack_top = z_table + env_u.dest_geom["height"]
    hole = np.array([float(env_u.dest_rack_xy[0, 0] + env_u.dest_hole_x[0]), float(env_u.dest_rack_xy[0, 1])])

    lifted = tube[2] > z_table + t.tube_height / 2.0 + 0.01
    held = closed and (lifted or np.linalg.norm(tip[:2] - tube[:2]) < 0.02)
    if not held:
        d = np.linalg.norm(tip[:2] - tube[:2])
        z = travel_z if d > 0.015 else grasp_z
        return np.array([tube[0], tube[1], z]), d < near_radius, ("reach" if d > 0.015 else "grasp"), False

    offset = hole - bottom[:2]            # where the tube's bottom must still move, in xy
    d = float(np.linalg.norm(offset))
    low = bottom[2] < rack_top + 0.02
    xy = tip[:2] + offset
    seated = d < 0.01 and bottom[2] - hole_floor[2] < 0.012
    if d < 0.004 or (d < 0.01 and low):
        return np.array([xy[0], xy[1], grasp_z + 0.003]), True, "insert", seated
    if low and d > near_radius:
        return np.array([tip[0], tip[1], travel_z]), True, "pull_out", False
    return np.array([xy[0], xy[1], travel_z]), d < near_radius, ("carry" if d >= near_radius else "align"), False


class OraclePredictor(_ChunkCache):
    """TEST ONLY: expert plan from true state; fast when coarse, slow when precise."""

    def __init__(self, horizon: int = 20, fast_speed: float = 0.008, slow_speed: float = 0.0008,
                 replan_every: int = 1):
        self.horizon = horizon
        self.fast_speed = fast_speed
        self.slow_speed = slow_speed
        self.last_phase = ""
        super().__init__(replan_every)

    def _predict(self, env_u):
        target, precise, self.last_phase, _ = tube_phase(env_u)
        tip = env_u.fingertip_midpoint_pos[0].detach().cpu().numpy()
        step = self.slow_speed if precise else self.fast_speed
        delta = target - tip
        dist = float(np.linalg.norm(delta))
        unit = delta / dist if dist > 1e-9 else np.zeros(3)
        travel = np.minimum(np.arange(1, self.horizon + 1) * step, dist)
        pos = tip + travel[:, None] * unit
        out = np.zeros((self.horizon, 8))
        out[:, 0:3] = pos
        out[:, 3:7] = _DOWN
        out[:, 7] = float(env_u.gripper[0, 0])
        return out


# ---------------------------------------------------------------------------------------
# The trained policy
# ---------------------------------------------------------------------------------------

class SmolVLAPredictor(_ChunkCache):
    """Predicted chunks from a fine-tuned SmolVLA, fed the wrist and side camera views.

    Image keys are matched to cameras by name ("wrist" / "side") when the checkpoint uses
    them, otherwise in order (first key = wrist, second = side), which is how the
    two-camera datasets map views onto smolvla_base's camera slots.
    """

    def __init__(self, checkpoint: str, device: str = "cuda", replan_every: int = 5):
        from residual_copilot.pilot_models.smolvla_pilot import SmolVLA_Pilot

        self.pilot = SmolVLA_Pilot(checkpoint, device=device)
        keys = sorted(self.pilot.policy.config.image_features)
        if len(keys) < 2:
            raise RuntimeError(f"checkpoint declares {len(keys)} camera(s); the tube task uses wrist + side")
        named = {k: ("wrist" if "wrist" in k else "side" if "side" in k else None) for k in keys}
        if all(named.values()):
            self.key_to_cam = {k: f"{v}_camera" if v == "wrist" else "upper_right_camera" for k, v in named.items()}
        else:
            ordered = keys[:2]
            self.key_to_cam = {ordered[0]: "wrist_camera", ordered[1]: "upper_right_camera"}
        print("[SmolVLAPredictor] camera mapping: "
              + ", ".join(f"{k} <- {c}" for k, c in self.key_to_cam.items()), flush=True)
        super().__init__(replan_every)

    def reset(self):
        super().reset()
        if hasattr(self, "pilot"):
            self.pilot.reset()

    def frame(self, env_u):
        state = torch.cat([
            env_u.fingertip_midpoint_pos[0], env_u.fingertip_midpoint_quat[0], env_u.gripper[0],
            env_u.ee_linvel_fd[0], env_u.ee_angvel_fd[0],
        ], dim=-1)
        frame = {"observation.state": state, "task": env_u.instructions[0]}
        for key, cam in self.key_to_cam.items():
            frame[key] = getattr(env_u, cam).data.output["rgb"][0]
        return frame

    def _predict(self, env_u):
        return self.pilot.predict_chunk(self.frame(env_u)).numpy()
