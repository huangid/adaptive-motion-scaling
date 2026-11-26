# Assistance: policy-informed motion scaling (and, next, action blending)

The study's two assistance channels (proposal step 3), run on the tube transfer task by
`scripts/assisted_teleop.py`. Status: **motion scaling is implemented; action blending
(partial denoising) is not yet.**

## Control loop

Each control step (15 Hz):

```
raw SpaceMouse input   (direction + how hard the operator pushes)
gain = scaler(predicted chunk)          fixed, or adapted to the policy's planned speed
a_H  = fingertip pose + gain * raw      the scaled human command
a_exec = a_H                            (blending will modify this)
env.step(a_exec)
```

Only the size of the motion is scaled; its direction always comes from the operator.

## Conditions

```bash
# fixed gain (baseline)
scripts/isaac_python.sh scripts/assisted_teleop.py --scaling fixed --gain 0.15 \
    --predictor smolvla --checkpoint <ckpt> --record
# policy-informed adaptive gain
scripts/isaac_python.sh scripts/assisted_teleop.py --scaling adaptive \
    --predictor smolvla --checkpoint <ckpt> --record
```

`--blending on` is reserved for partial denoising and currently exits with a message.

## Adaptive scaling (`source/assist/scaling.py`)

The planned speed is the mean fingertip travel per step over the first `--lookahead`
steps of the predicted chunk. It maps linearly to a gain:

| Planned speed | Gain |
|---|---|
| ≤ `--slow_speed` (1 mm/step) | `--gain_min` (0.04) |
| ≥ `--fast_speed` (6 mm/step) | `--gain_max` (0.20) |
| in between | linear |

An exponential moving average (`--smoothing 0.7`, the weight on the previous gain) takes
about half a second to settle after a change. All values are starting points to tune on
the trained policy's real predictions and in pilot sessions.

## Predictors (`source/assist/predictors.py`)

- `SmolVLAPredictor`: the trained policy, fed the wrist and side views. It replans every
  `--replan_every` steps (default 5) and steps through the plan in between.
- `OraclePredictor`: **test only.** It plans from the simulator's true object positions,
  fast when coarse and slow when precise, so the pipeline can be exercised before a policy
  exists. Never use it in the study.

`--operator sim` (`source/assist/sim_operator.py`) is likewise a **test-only** scripted
stand-in for the person at the SpaceMouse. It is a crude model (it does not push harder
when motion is slow, as people do), so its timings say nothing about which condition is
better.

```bash
# pipeline test without a policy or a person
scripts/isaac_python.sh scripts/assisted_teleop.py --scaling adaptive \
    --predictor oracle --operator sim --num_episodes 3 --headless --trace 60
```

## Logging

With `--record`, each step is written in `play.py`'s rollout format (so `convert_demos.py`
reads it) plus `raw_input`, `gain`, `planned_speed` and `phase` (test stand-ins only).
`meta/stats.json` holds per-episode success, steps, seconds, layout and mean gain by phase;
`meta/infos.json` holds the run's settings. `--trace N` prints the state every N steps.

## Still to do

- Action blending: partial noising of the operator's command and SmolVLA denoising from
  there ("To the Noise and Back"), with one knob for how much of the plan comes through.
- Logging how strongly blending changes the scaled command (needs blending).
- Tuning the gain range on real predictions; adapting `eval_smolvla.py` to this task.
