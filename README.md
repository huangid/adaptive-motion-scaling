# Policy-Informed Adaptive Motion Scaling for Assisted Teleoperation

Undergraduate research project. Full proposal:
[docs/proposal.pdf](docs/proposal.pdf).

**Research question:** when a single generative robot policy provides both action-level
assistance and adaptive motion scaling, how do these mechanisms interact to affect speed,
precision, and the operator's sense of control?

## Background

Shared autonomy combines human control inputs with autonomous robot assistance to
improve task performance. One common technique in teleoperation is **motion scaling**,
which adjusts how much the robot moves in response to human input: higher gains support
fast, coarse motions, while lower gains support precise manipulation. Most approaches use
fixed or manually adjusted scaling, though some adapt scaling using task information,
operator behavior, or learned demonstrations [1, 2, 6].

Generative robot policies such as vision-language-action models (VLAs) predict future
actions (*action chunks*) and may offer a more flexible way to inform the appropriate
movement scale at different task stages. The same policy can also provide action-level
assistance through partial denoising [4]. These are distinct assistance channels:

- **Action blending changes what the robot does.** The human's action is partially
  noised and then denoised toward the learned policy.
- **Motion scaling changes how much human input moves the robot**, without changing its
  direction.

Unlike systems that alternate between autonomous movement and human intervention [7],
the operator remains actively involved throughout.

## Plan

1. **Task design.** Develop an Isaac Lab manipulation task requiring both large, coarse
   movements and small, precise movements, such as transporting objects across a table
   before carefully inserting them into tight-fitting slots.
2. **Data collection and VLA training.** Collect demonstrations including fast transport
   and slow insertion motions, and train an action-chunking VLA to predict future robot
   actions.
3. **Two assistance channels from the same trained VLA.**
   - (a) *To the Noise and Back*-style partial noising and denoising, to adjust human
     actions toward the learned policy [4].
   - (b) Predicted action chunks (for example, their magnitudes) adapt the SpaceMouse
     motion scaling. Scaling preserves the operator's directional command and must also
     work without action blending.
4. **Evaluation: a 2 × 2 comparison**, enabling action blending and policy-informed
   adaptive scaling independently.

   |                      | No action blending           | With action blending          |
   |----------------------|------------------------------|-------------------------------|
   | **Fixed scaling**    | Standard teleoperation       | Partial denoising only        |
   | **Adaptive scaling** | Policy-informed scaling only | Scaling + partial denoising   |

   Measures: completion time, insertion accuracy, task success, and **perceived
   control**, plus how strongly denoising changes the scaled human command. The analysis
   tests whether the two methods are complementary or redundant, and whether scaling
   reduces the corrections needed from blending.
5. **Related work review.** After the initial experiments, search for additional
   related research on adaptive motion scaling, learned teleoperation interfaces, and
   generative-policy shared autonomy, and compile a short annotated list of relevant
   papers and how they relate to the project.

## Expected outcome

A proof of concept showing whether the two policy-informed assistance channels work best
separately or together, and whether input-level adaptation can preserve a greater sense
of human control.

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
5. Conti & Khatib (2005). *Spanning Large Workspaces Using Small Haptic Devices.* Historical context: a different way to deal with motion scaling in teleoperation.
6. Shi et al. (2025). *Exploring the Synergistic Effects of Teleoperation Scaling Ratio and Learning From Demonstration.* IEEE TASE.
7. Song et al. (2026). *Assistron: Bayesian Shared Autonomy with Off-the-shelf Vision-Language-Action Models.* arXiv.

## License and attribution

Apache License 2.0 (see [LICENSE](LICENSE)). Based on the Residual Copilot codebase,
also released under Apache-2.0; see [NOTICE](NOTICE).
