# Tube transfer task

The manipulation task for the adaptive motion scaling study (proposal step 1). The operator
moves a test tube from a source rack to an empty single-hole holder:

```
        (robot base)
             |
   SOURCE RACK  (random spot, +y side)          DESTINATION  (random spot, -y side)
   [o ● o]   -- grasp, pull out, carry -->       [o]
   tube in a random hole                         one hole
```

Each trip alternates precise and coarse motion: grasp the tube, pull it straight up out of
its hole (precise), carry it ~0.3–0.45 m across the table (fast, coarse), line it up over a
the destination hole and lower it in (precise). That change in the right input gain within an episode is
what policy-informed adaptive scaling has to track.

## Scene

What the cameras see and what the physics uses are separate:

| | Cameras see | Physics uses (invisible) |
|---|---|---|
| Tube | LabUtopia glass test tube with orange liquid and a stopper | Solid cylinder, 2.4 cm × 15 cm |
| Source rack | LabUtopia two-tier rack, shortened to 3 holes in a row | Thick boxes around 3 square holes |
| Destination | The same model shortened to 1 hole | Thick boxes around 1 square hole |

The LabUtopia models have thin glass and plastic shells, which make poor colliders (thin
walls caused unstable contact in the earlier project), so they are display-only. LabUtopia's
rack stores its plate surfaces as large polygons with curved outlines, which the renderer
splits into triangles that cut across the holes (they looked like irregular diamonds); they
are re-triangulated by ear clipping. The rack has 6 holes; it is shortened by collapsing the
band holding the extra holes and sliding the far end panel inward, so its plates stay
continuous. It is then stretched sideways until
its round holes match the collider holes, `tube diameter + rack_clearance` wide. The tube is
cut out of LabUtopia's 5-tube set and fitted to the cylinder. Both racks
lie lengthwise along x, pointing away from the robot, so the side camera sees every hole.

Settings live in `TubeTransfer` (`source/xarm_assembly_env/assembly_tasks_cfg.py`):

| Setting | Default | Meaning |
|---|---|---|
| `tube_radius`, `tube_height` | 0.012, 0.15 m | Tube size (25 × 150 mm test tube) |
| `source_rack_holes`, `dest_rack_holes` | 3, 1 | Holes per rack (1–6) |
| `rack_clearance` | 6 mm | Hole width minus tube diameter; smaller = harder insertion |
| `source_x_range`, `source_y_range` | x 0.32–0.42, y 0.14–0.22 | Where the source rack is resampled each episode |
| `dest_x_range`, `dest_y_range` | x 0.34–0.40, y −0.21 to −0.15 | Where the destination is resampled (±3 cm, under the side camera) |
| `layout_seed` | None | Set to repeat one layout every episode (practice, debugging, fixed evaluation layouts) |
| `use_lab_visuals` | True | False shows the plain collider shapes and needs no download |
| `lab_tube_index` | 1 | Which tube of LabUtopia's TestTube_02 set (0–4) |
| `gripper_ctrl_clamp` | 1.3 | Gripper squeeze cap, tuned for the 2.4 cm tube |
| `instruction` | "move the test tube to the empty rack" | Language instruction for the VLA |

**Success:** the tube's bottom is inside the destination hole and seated (within
`seat_tolerance` of the table), it is upright (`upright_tolerance_deg`), and the gripper has
let go (`release_gripper_thresh`). **Failure:** the tube tips past `tipped_deg` or falls off
the table. Episodes last up to 90 s (`episode_length_s` in `XArmTubeTransferCfg`); they end
early on success or failure.

Debug markers are off by default for this task: they render into camera frames, so they
must never appear in recorded demos.

Recordings made with `play.py --record` store `source_rack_xy`, `source_slot` (the hole the
tube started in) and `dest_rack_xy` per episode in `meta/stats.json`.

## Design choices

- **One destination hole.** With several target holes the policy could head for a different
  one than the operator, and action blending would then pull against them, which would
  distort the sense-of-control measure. One hole also makes every trial's final precision
  demand identical, so the four conditions are comparable.
- **Both racks move.** The source varies (position and starting hole) so the policy cannot
  memorise where the tube starts. The destination varies by ±3 cm so the target has to be
  seen: with a fixed target, a hand-written "slow down near this point" rule could stand in
  for policy-informed scaling.
- **Evaluation.** Collect demos with fresh layouts. For the 2 × 2 comparison, consider a
  fixed set of layouts (via `layout_seed`) shared by all conditions.

## Running it

Run from this repo. `scripts/isaac_python.sh` borrows the Isaac Sim venv from
`~/Residual_Copilot` (override with `ISAAC_VENV`) but loads this repo's code. `--two_cam`
gives the wrist + side camera setup used for SmolVLA:

```bash
scripts/isaac_python.sh scripts/play.py --task TubeTransfer --pilot SpaceMousePilot --num_envs 1 --two_cam
# record demos
scripts/isaac_python.sh scripts/play.py --task TubeTransfer --pilot SpaceMousePilot --num_envs 1 --two_cam --record
```

Running `python scripts/play.py` from `~/Residual_Copilot` instead runs that checkout's code,
which has no TubeTransfer task.

The first launch downloads the two LabUtopia files (~6 MB) into the Hugging Face cache,
builds the shortened rack and cuts a single tube out of the TestTube_02 set; later launches
reuse all of them.

Gym ids: `XArm-TubeTransfer-Residual` (pilot + residual stack, used by `play.py`) and
`XArm-TubeTransfer-GuidedDiffusion` (8D absolute actions, for policy and blended control).

## Assets and licence

The rack and tube models come from LabUtopia (Li et al., 2025,
[arXiv:2505.22634](https://arxiv.org/abs/2505.22634)),
[dataset](https://huggingface.co/datasets/Ruinwalker/LabUtopia-Dataset) licensed
**CC BY-NC 4.0**: non-commercial use, with attribution. Cite the paper in any publication
that shows the task. The files are downloaded at runtime and never committed (the repo's
`.gitignore` excludes USD files). See `NOTICE`.

## Not done yet

- `eval_smolvla.py` and `shared_autonomy.py` still assume RandomBlock (block colours and
  layouts) and need adapting once demos and a trained policy exist.
- Without a demo file, non-SpaceMouse pilots get a placeholder that holds the arm still.
