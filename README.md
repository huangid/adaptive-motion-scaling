# Policy-Informed Adaptive Motion Scaling for Assisted Teleoperation

Undergraduate research project.

**Research question:** when a single generative robot policy provides both action-level
assistance and adaptive motion scaling, how do these mechanisms interact to affect speed,
precision, and the operator's sense of control?

## Idea

A vision-language-action model (VLA) trained on teleoperation demonstrations offers two
distinct ways to assist a human operator:

- **Action blending** changes *what the robot does*: the human's action is partially
  noised and then denoised toward the learned policy, in the style of
  *To the Noise and Back* (Yoneda et al., RSS 2023).
- **Adaptive motion scaling** changes *how much human input moves the robot*: the
  policy's predicted action chunks (for example, their magnitudes) set the SpaceMouse
  gain, high for coarse transport and low for precise insertion. Scaling keeps the
  operator's commanded direction and must work with or without blending.

## Plan

1. **Task.** An Isaac Lab manipulation task that needs both large, coarse motions and
   small, precise ones, such as carrying objects across a table and inserting them into
   tight-fitting slots.
2. **Data and VLA training.** Collect demonstrations with fast transport and slow
   insertion, then train an action-chunking VLA.
3. **Two assistance channels from the same VLA.** (a) Partial noising and denoising of
   the human action; (b) policy-informed SpaceMouse scaling.
4. **Evaluation: a 2 × 2 study.**

   |                      | No action blending           | With action blending          |
   |----------------------|------------------------------|-------------------------------|
   | **Fixed scaling**    | Standard teleoperation       | Partial denoising only        |
   | **Adaptive scaling** | Policy-informed scaling only | Scaling + partial denoising   |

   Measures: completion time, insertion accuracy, task success, perceived control, and
   how strongly denoising changes the scaled human command. The question is whether the
   two channels are complementary or redundant, and whether scaling reduces the
   corrections blending has to make.
5. **Related work review.** An annotated list of papers on adaptive motion scaling,
   learned teleoperation interfaces, and generative-policy shared autonomy.

## Status

Just started. The codebase is inherited (see below) and has not yet been adapted to
this project.

## Starting point

This repo begins from the code of an earlier project,
[`huangid/vla-shared-autonomy`](https://github.com/huangid/vla-shared-autonomy)
(branch `two-camera`), which itself builds on
[Residual Copilot](https://github.com/shuosha/Residual_Copilot)
([paper](https://arxiv.org/abs/2603.22787)). Pieces expected to be reused:

| Need | Existing code |
|------|---------------|
| Isaac Lab xArm environment and tasks (incl. `PegInsert`) | `source/xarm_assembly_env/` |
| SpaceMouse teleoperation, with fixed `pos_scale` / `rot_scale` gains | `source/pilot_models/spacemouse_pilot.py` |
| Demo collection and conversion to LeRobot format | `scripts/collect_data.py`, `scripts/convert_demos.py`, `scripts/npy_to_lerobot.py` |
| SmolVLA policy, training and evaluation | `source/pilot_models/smolvla_pilot.py`, `scripts/eval_smolvla.py`, `third_party/lerobot` |
| Diffusion-based shared-autonomy copilot (starting point for blending) | `source/pilot_models/disco_pilot.py` |

Setup, data-collection, training and evaluation commands from the earlier project are in
[docs/BASE_README.md](docs/BASE_README.md).

## Installation

Requires Python 3.11, [uv](https://docs.astral.sh/uv/), and CUDA 12.8 (Isaac Sim 5.1.0,
PyTorch 2.7.0 cu128).

```bash
git clone --recurse-submodules https://github.com/huangid/adaptive-motion-scaling.git
cd adaptive-motion-scaling
uv venv --python 3.11 --seed .venv
source .venv/bin/activate
UV_HTTP_TIMEOUT=300 uv sync
python scripts/list_envs.py   # confirm environment registration
```

## References

1. Yoon et al. (2025). *SIMS: Surgeon-Intention-driven Motion Scaling.*
2. Wang, Ji & Sun (2023). *An Auto-tuned Motion Scaling Method Based on Visual Recognition.* RCAR.
3. Tao et al. (2025). *LAMS: LLM-Driven Automatic Mode Switching for Assistive Teleoperation.* HRI.
4. Yoneda et al. (2023). *To the Noise and Back: Diffusion for Shared Autonomy.* RSS.
5. Conti & Khatib (2005). *Spanning Large Workspaces Using Small Haptic Devices.*
6. Shi et al. (2025). *Exploring the Synergistic Effects of Teleoperation Scaling Ratio and Learning From Demonstration.* IEEE TASE.
7. Song et al. (2026). *Assistron: Bayesian Shared Autonomy with Off-the-shelf Vision-Language-Action Models.* arXiv.

## License and attribution

Apache License 2.0 (see [LICENSE](LICENSE)). Based on the Residual Copilot codebase,
also released under Apache-2.0; see [NOTICE](NOTICE).
