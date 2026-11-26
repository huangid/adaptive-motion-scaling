"""Assisted SpaceMouse teleoperation on the tube transfer task (the 2 x 2 study runner).

Each control step:

    raw SpaceMouse input (direction + how hard)            <- operator
    gain = scaler(predicted chunk)                          <- fixed, or adapted to the plan
    a_H  = current fingertip pose + gain * raw input        <- the scaled human command
    a_exec = a_H                                            <- (action blending: next step)
    env.step(a_exec)

Conditions:
    --scaling fixed|adaptive     fixed gain (--gain) vs gain from the predicted chunk's speed
    --blending off|on            partial-denoising action blending (not implemented yet)

The predicted chunk comes from ``--predictor smolvla --checkpoint ...`` (the trained policy)
or ``--predictor oracle``, a TEST-ONLY stand-in that plans from the simulator's true state.
``--operator sim`` likewise replaces the human with a scripted test driver.

Every step is logged (and, with --record, written in play.py's rollout format so
convert_demos.py reads it): observation, raw input, gain, planned speed, the human command
and the executed action. Episode results go to meta/stats.json.

Examples:
    # real study condition: adaptive scaling, trained policy, human at the SpaceMouse
    scripts/isaac_python.sh scripts/assisted_teleop.py --scaling adaptive \\
        --predictor smolvla --checkpoint outputs/train/<run>/checkpoints/last/pretrained_model --record

    # pipeline test without a policy or a human
    scripts/isaac_python.sh scripts/assisted_teleop.py --scaling adaptive --predictor oracle \\
        --operator sim --num_episodes 3 --headless
"""
import argparse
import json
import os
import random
import shutil
import sys

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Assisted SpaceMouse teleoperation (tube transfer).")
parser.add_argument("--scaling", choices=["fixed", "adaptive"], required=True)
parser.add_argument("--blending", choices=["off", "on"], default="off")
parser.add_argument("--predictor", choices=["smolvla", "oracle"], default="smolvla",
                    help="Source of the predicted chunk. 'oracle' is a test-only stand-in.")
parser.add_argument("--checkpoint", type=str, default=None, help="SmolVLA checkpoint (for --predictor smolvla).")
parser.add_argument("--policy_device", type=str, default="cuda")
parser.add_argument("--replan_every", type=int, default=5,
                    help="Steps between policy predictions; the plan is stepped through in between.")
parser.add_argument("--operator", choices=["spacemouse", "sim"], default="spacemouse",
                    help="'sim' is a scripted test driver instead of a person.")
# scaling
parser.add_argument("--gain", type=float, default=0.15, help="Fixed gain (SpaceMouse pos_scale).")
parser.add_argument("--gain_min", type=float, default=0.04)
parser.add_argument("--gain_max", type=float, default=0.20)
parser.add_argument("--slow_speed", type=float, default=0.001, help="Planned m/step that maps to gain_min.")
parser.add_argument("--fast_speed", type=float, default=0.006, help="Planned m/step that maps to gain_max.")
parser.add_argument("--lookahead", type=int, default=10, help="Chunk steps used to measure planned speed.")
parser.add_argument("--smoothing", type=float, default=0.7, help="EMA weight on the previous gain.")
# episodes / recording
parser.add_argument("--num_episodes", type=int, default=1)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--layout_seed", type=int, default=None, help="Repeat one layout every episode.")
parser.add_argument("--record", action="store_true", help="Write rollouts (robot JSON + camera frames).")
parser.add_argument("--rollout_dir", type=str, default=None)
parser.add_argument("--yes", "-y", action="store_true", help="Overwrite an existing rollout dir.")
parser.add_argument("--trace", type=int, default=0, help="Print the state every N steps (debugging).")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(kit_args="--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error")
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

if args.blending == "on":
    raise SystemExit("--blending on (partial denoising) is not implemented yet.")
if args.predictor == "smolvla" and not args.checkpoint:
    raise SystemExit("--predictor smolvla needs --checkpoint")
need_cameras = args.record or args.predictor == "smolvla"
if need_cameras:
    args.enable_cameras = True
if args.seed is None:
    args.seed = random.randint(0, 2**31 - 1)
print(f"[INFO] seed {args.seed}")

app = AppLauncher(args).app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import residual_copilot  # noqa: E402,F401
from residual_copilot.assist import AdaptiveScaler, FixedScaler  # noqa: E402
from residual_copilot.assist.predictors import OraclePredictor, SmolVLAPredictor  # noqa: E402
from residual_copilot.xarm_assembly_env.xarm_env_cfg import XArmTubeTransferGuidedDiffusionCfg  # noqa: E402

TASK_ID = "XArm-TubeTransfer-GuidedDiffusion"
CAMERA_DIRS = (("wrist_camera", "camera_wrist"), ("upper_right_camera", "camera_side"))


def gripper_tilt_deg(env_u):
    """Angle between the gripper's pointing axis and straight down, in degrees."""
    q = env_u.fingertip_midpoint_quat[0]
    w, x, y, z = (float(v) for v in q)
    zz = 1 - 2 * (x * x + y * y)          # world z component of the gripper's z axis
    return float(np.degrees(np.arccos(np.clip(-zz, -1.0, 1.0))))


def make_env():
    cfg = XArmTubeTransferGuidedDiffusionCfg()
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.pilot_model = "knn"          # env-internal pilot unused; the operator is ours
    cfg.pilot_type = "none"
    cfg.vis.print_rew = False
    cfg.vis.store_rgb = need_cameras
    cfg.use_front_camera = False
    cfg.use_wrist_camera = need_cameras
    cfg.use_upper_right_camera = need_cameras
    cfg.num_rerenders_on_reset = 2 if need_cameras else 0
    if args.layout_seed is not None:
        cfg.task.layout_seed = args.layout_seed
    return gym.make(TASK_ID, cfg=cfg)


def make_scaler():
    if args.scaling == "fixed":
        return FixedScaler(gain=args.gain)
    return AdaptiveScaler(gain_min=args.gain_min, gain_max=args.gain_max, slow_speed=args.slow_speed,
                          fast_speed=args.fast_speed, lookahead=args.lookahead, smoothing=args.smoothing)


def make_operator(env_u):
    if args.operator == "sim":
        from residual_copilot.assist.sim_operator import SimulatedOperator
        return SimulatedOperator(env_u, seed=args.seed)
    from residual_copilot.pilot_models.spacemouse_pilot import SpaceMousePilot
    op = SpaceMousePilot(num_envs=1, device=str(env_u.device), hold_orientation=True)
    op.pos_scale = args.gain        # also sets the dead zone reference
    return op


def episode_meta(env_u):
    """Layout at episode START (the env resets inside the step that ends an episode)."""
    return {
        "instruction": env_u.instructions[0],
        "source_rack_xy": env_u.source_rack_xy[0].tolist(),
        "source_slot": int(env_u.source_slot[0]),
        "dest_rack_xy": env_u.dest_rack_xy[0].tolist(),
    }


def step_record(obs, a_h, a_exec, raw, gain, speed, phase):
    """play.py's keys (obs.* / base_action.* / action.*) plus the assistance signals."""
    return {
        "obs.fingertip_pos": obs[0:3].tolist(),
        "obs.fingertip_quat": obs[3:7].tolist(),
        "obs.gripper": obs[7:8].tolist(),
        "obs.fingertip_pos_rel_fixed": obs[8:11].tolist(),
        "obs.fingertip_pos_rel_held": obs[11:14].tolist(),
        "obs.ee_linvel_fd": obs[14:17].tolist(),
        "obs.ee_angvel_fd": obs[17:20].tolist(),
        "base_action.fingertip_pos": a_h[0:3].tolist(),      # scaled human command
        "base_action.fingertip_quat": a_h[3:7].tolist(),
        "base_action.gripper": a_h[7:8].tolist(),
        "action.fingertip_pos": a_exec[0:3].tolist(),   # what was executed
        "action.fingertip_quat": a_exec[3:7].tolist(),
        "action.gripper": a_exec[7:8].tolist(),
        "raw_input": [float(v) for v in raw],
        "gain": float(gain),
        "planned_speed": float(speed),
        "phase": phase,                  # oracle / sim operator only
    }


def make_rollout_dir():
    path = os.path.abspath(args.rollout_dir or os.path.join(
        "logs", "rollouts", f"assisted_{args.scaling}_blend-{args.blending}_{args.predictor}"))
    if os.path.exists(path):
        if not args.yes and input(f"{path} exists. Overwrite? [y/N] ").strip().lower() != "y":
            raise SystemExit("Aborting.")
        shutil.rmtree(path)
    os.makedirs(path)
    return path


def main():
    env = make_env()
    env_u = env.unwrapped
    scaler = make_scaler()
    predictor = (OraclePredictor() if args.predictor == "oracle"
                 else SmolVLAPredictor(args.checkpoint, device=args.policy_device, replan_every=args.replan_every))
    operator = make_operator(env_u)
    rollout_path = make_rollout_dir() if args.record else None
    print(f"[INFO] scaling={args.scaling} blending={args.blending} predictor={args.predictor} "
          f"operator={args.operator}" + (f" -> {rollout_path}" if rollout_path else ""))
    if args.predictor == "oracle" or args.operator == "sim":
        print("[INFO] TEST MODE: oracle predictor / simulated operator are not for the study.")

    obs_dict, _ = env.reset()
    stats = {}
    for ep in range(args.num_episodes):
        name = f"episode_{ep:04d}"
        meta = episode_meta(env_u)
        scaler.reset()
        predictor.reset()
        operator.reset()
        records, frames = [], []
        steps, success, ended = 0, False, False
        gains = {}
        max_steps = env_u.max_episode_length

        while steps < max_steps:
            obs = (obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict)[0].detach().cpu().numpy()
            tip = env_u.fingertip_midpoint_pos[0].detach().cpu().numpy()
            with torch.inference_mode():
                chunk = predictor.chunk(env_u)
                gain = scaler.update(chunk[:, 0:3], tip)
                a_h = operator.get_actions(None, env_u.fingertip_midpoint_pos, env_u.fingertip_midpoint_quat,
                                           env_u.gripper, pos_scale=gain)[0].to(env_u.device).float().clone()
                a_h[7] = 1.0 if float(a_h[7]) > 0 else 0.0   # SpaceMouse grip is -1/+1; env wants [0, 1]
                a_exec = a_h.clone()                          # blending goes here
            speed = getattr(scaler, "last_speed", float("nan"))
            phase = getattr(operator, "phase", "") or getattr(predictor, "last_phase", "")
            gains.setdefault(phase or "-", []).append(gain)
            if args.trace and steps % args.trace == 0:
                tube = env_u.held_pos[0].tolist()
                print(f"  t={steps:4d} {phase:8s} tip={[round(v, 3) for v in tip.tolist()]} "
                      f"tube={[round(v, 3) for v in tube]} grip={float(env_u.gripper[0, 0]):.2f} "
                      f"raw={[round(float(v), 2) for v in operator.last_raw_pos]} gain={gain:.3f} "
                      f"tilt={float(env_u._tube_tilt_deg()[0]):.1f}deg "
                      f"gripper_tilt={gripper_tilt_deg(env_u):.1f}deg", flush=True)

            records.append(step_record(obs, a_h.cpu().numpy(), a_exec.cpu().numpy(),
                                       operator.last_raw_pos, gain, speed, phase))
            if rollout_path is not None:
                frames.append({sub: getattr(env_u, cam).data.output["rgb"][0].cpu().numpy()
                               for cam, sub in CAMERA_DIRS})

            with torch.inference_mode():
                obs_dict, _r, terminated, truncated, _i = env.step(a_exec.view(1, -1))
            steps += 1
            if not args.headless:
                print(f"\r[{name}] step {steps:4d}  gain {gain:.3f}  planned {speed * 1000:5.2f} mm/step  "
                      f"{phase:9s}", end="", flush=True)
            if bool((terminated | truncated)[0]):
                success = bool(env_u.ep_succeeded[0])
                ended = True
                break

        seconds = steps / 15.0
        phase_gains = {p: round(float(np.mean(g)), 3) for p, g in gains.items()}
        print(f"\n[{name}] {'SUCCESS' if success else 'fail'} in {steps} steps ({seconds:.1f} s) | "
              f"mean gain by phase: {phase_gains}", flush=True)
        stats[name] = {**meta, "success": success, "steps": steps, "seconds": round(seconds, 2),
                       "scaling": args.scaling, "blending": args.blending, "mean_gain_by_phase": phase_gains}

        if rollout_path is not None:
            ep_dir = os.path.join(rollout_path, name)
            os.makedirs(os.path.join(ep_dir, "robot"))
            for _cam, sub in CAMERA_DIRS:
                os.makedirs(os.path.join(ep_dir, sub, "rgb"))
            for t, rec in enumerate(records):
                with open(os.path.join(ep_dir, "robot", f"{t:06d}.json"), "w") as f:
                    json.dump(rec, f)
            for t, views in enumerate(frames):
                for sub, img in views.items():
                    cv2.imwrite(os.path.join(ep_dir, sub, "rgb", f"{t:06d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        if not ended:
            with torch.inference_mode():
                obs_dict, _ = env.reset()

    if rollout_path is not None:
        os.makedirs(os.path.join(rollout_path, "meta"))
        infos = {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))}
        with open(os.path.join(rollout_path, "meta", "infos.json"), "w") as f:
            json.dump(infos, f, indent=2)
        with open(os.path.join(rollout_path, "meta", "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
    n_ok = sum(s["success"] for s in stats.values())
    print(f"[INFO] {n_ok}/{len(stats)} successful"
          + (f"; recorded to {rollout_path}" if rollout_path else ""), flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
