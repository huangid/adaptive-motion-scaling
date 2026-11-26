# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

This repo studies **policy-informed adaptive motion scaling** for SpaceMouse teleoperation:
one VLA provides both action blending (To the Noise and Back-style partial denoising) and
adaptive scaling of the operator's input gain, compared in a 2 × 2 study. See `README.md`.

Most of the code is inherited from Residual Copilot via `huangid/vla-shared-autonomy`
(branch `two-camera`); the sections after "Tube transfer task" describe that inherited code.
Its full command reference is in `docs/BASE_README.md`.

## Tube transfer task (this project)

The study's task: move a test tube from a 3-hole source rack to a single-hole holder, both
placed at random each episode. Full design, settings and rationale: `docs/TUBE_TRANSFER.md`.

- `source/xarm_assembly_env/tube_transfer_env.py`: `XArmTubeTransferEnv`, a self-contained
  subclass of `XArmEnv`. Add tube-transfer logic here, not as new `name ==` branches in
  `xarm_env.py`.
- `source/xarm_assembly_env/lab_assets.py`: downloads LabUtopia's rack and tube (CC BY-NC 4.0,
  pinned revision) and prepares them (re-triangulates the rack, shortens it, cuts one tube
  out of a set). Outputs are cached next to the download; bump `_RACK_VERSION` /
  `_EXTRACT_VERSION` when changing that processing.
- Config: `TubeTransfer` in `assembly_tasks_cfg.py`, `XArmTubeTransferCfg` in `xarm_env_cfg.py`.
  Gym ids `XArm-TubeTransfer-Residual` and `XArm-TubeTransfer-GuidedDiffusion`.
- Physics and appearance are separate: invisible solid colliders (a cylinder, thick boxes)
  with the LabUtopia models drawn over them. Do not use the thin-shelled models as colliders.
- Tested values: tube radius 0.012 m with `gripper_ctrl_clamp = 1.3` (thinner tubes and
  full squeeze both failed to hold); `table_top_z = -0.0045` (measured).
- `eval_smolvla.py` and `shared_autonomy.py` still assume RandomBlock.
- The SpaceMouse pilot runs with `hold_orientation=True` for this task: re-anchoring the
  target orientation to the measured one made the gripper (and held tube) drift and tilt.

## Assistance (this project)

`source/assist/` (package `residual_copilot.assist`) and `scripts/assisted_teleop.py`: the
2 × 2 study runner. `scaling.py` has `FixedScaler` / `AdaptiveScaler` (gain from the
predicted chunk's planned speed); `predictors.py` has `SmolVLAPredictor` and the
**test-only** `OraclePredictor`; `sim_operator.py` is a **test-only** scripted operator.
Action blending (partial denoising) is not implemented yet. See `docs/ASSISTANCE.md`.

## Setup & Installation

```bash
git clone --recurse-submodules <repo>
uv venv --python 3.11 --seed .venv
source .venv/bin/activate
uv sync
```

The package manager is **uv** (not pip). The Python package is named `residual_copilot` and maps to the `source/` directory (see `pyproject.toml`).

On this machine the repo has no `.venv`; run everything through `scripts/isaac_python.sh`,
which uses the Isaac Sim venv in `~/Residual_Copilot` (override with `ISAAC_VENV`) but
imports this repo's `source/` via a `.pypath/` symlink, and sets the `libgomp` `LD_PRELOAD`.
Running `python` from `~/Residual_Copilot` runs that checkout's code instead.

Running any Isaac Lab environment requires an active **Isaac Sim 5.1.0** / Isaac Lab 2.3.2 runtime — `import residual_copilot` outside that runtime will fail on `pxr`/USD imports.

## Linting

```bash
flake8 source/ scripts/
```

Config in `.flake8`: max line length 120, max complexity 30, Google-style docstrings.

## Validation Scripts (no formal test suite)

```bash
python scripts/list_envs.py          # confirm environment registration
scripts/isaac_python.sh scripts/play.py --task TubeTransfer --pilot kNNPilot --two_cam --headless  # tube transfer smoke run
```

## Training

```bash
python scripts/train.py \
  --task XArm-GearMesh-Residual \
  --num_envs 128 \
  --headless
```

Key CLI args: `--task`, `--num_envs`, `--checkpoint` (resume), `--wandb_project`, `--distributed` (multi-GPU).

Evaluation/rendering:

```bash
python scripts/play.py --task GearMesh --pilot kNNPilot --copilot ResidualCopilot
```

## Architecture

### Control Stack

```
obs (11D) → Pilot (KNN or BC/DiffusionPolicy) → base_action (8D)
         ↓
[+noise or +lag augmentation if pilot_type != "none"]
         ↓
obs (35D policy / more for critic) → RL Residual Policy → residual_action (7D)
         ↓
_apply_residual() → admittance control (control.py) → IK (Pinocchio/SAPIEN) → joint targets
         ↓
Isaac Sim physics
```

### Observation Space
- **Policy obs (35D):** fingertip pos, quat, gripper, relative held/base positions, velocities, base actions — noisy (matches data distribution)
- **Critic state:** adds ground-truth joint positions, held object pose, base object pose — ground truth for critic-only

### Three Task Environments

Registered in `source/xarm_assembly_env/__init__.py`:
- `XArm-GearMesh-Residual`
- `XArm-NutThread-Residual`
- `XArm-PegInsert-Residual`

Each task defines its own asset configs, reward scales, and data/model HuggingFace paths in `assembly_tasks_cfg.py`.

### Configuration Hierarchy

1. **`XArmEnvCfg`** (`xarm_env_cfg.py`): base environment — pilot type, observation order, domain randomization (`DomainRandCfg`), control thresholds (`CtrlCfg`), 128 parallel envs
2. **Task cfg** (`assembly_tasks_cfg.py`): task-specific assets, reward scales, success thresholds, data paths
3. **RL-Games PPO** (`agents/rl_games_ppo_cfg.yaml`): γ=0.99, τ=0.95, lr=5e-4 (adaptive), 32-step horizon, 2-layer MLP [32, 32]

### Pilot Models

- **KNN** (`pilot_models/knn_pilot.py`): finds K=10 nearest neighbors in training data, interpolates actions with variable-horizon queuing (1–15 steps). Config: `config/knn_cfg.json`.
- **BC/DiffusionPolicy** (`pilot_models/bc_pilot.py`): wraps LeRobot's DiffusionPolicy. Two variants: `bc_teleop` (teleop data) and `bc_expert` (expert data). Config: `config/state_dp_cfg.json` or `vision_dp_cfg.json`.

Set via `XArmEnvCfg.pilot_model` ∈ `{"knn", "bc_teleop", "bc_expert"}` and `XArmEnvCfg.pilot_type` ∈ `{"noisy", "laggy"}`.

### Key Files

| File | Role |
|------|------|
| `source/xarm_assembly_env/xarm_env.py` | Main `DirectRLEnv` — scene setup, step loop, rewards, reset |
| `source/xarm_assembly_env/xarm_env_cfg.py` | All config dataclasses (`XArmEnvCfg`, `DomainRandCfg`, `CtrlCfg`, obs/state dim maps) |
| `source/xarm_assembly_env/assembly_tasks_cfg.py` | Task-specific asset + reward configs |
| `source/utils/control.py` | `adm_ctrl_task_space()` (admittance law) + `IK_Controller` (Pinocchio) |
| `source/utils/utils.py` | `resolve_hf()` (HuggingFace asset downloads), math helpers |
| `source/pilot_models/knn_pilot.py` | KNN pilot |
| `source/pilot_models/bc_pilot.py` | DiffusionPolicy BC pilot |
| `scripts/train.py` | RL training entry point |

### Data & Assets

All training data, model checkpoints, and USD/URDF assets are fetched from HuggingFace repo `shashuo0104/residual_copilot_data` via `resolve_hf()` in `utils.py`. Paths are specified in task configs.

### Domain Randomization

Admittance control parameters (`Kx`, `Kr`, `mx`, `mr`) are randomized per episode when `dmr.use_dmr=True`. This is the primary source of sim-to-real robustness.
