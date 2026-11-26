import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from ..utils.utils import resolve_hf

ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"
HF_ASSETS_REPO = "shashuo0104/residual_copilot_assets"   # robot URDF/USD, object USD
HF_MODELS_REPO = "shashuo0104/residual_copilot_models"   # BC + RL checkpoints
HF_DATA_REPO   = "shashuo0104/residual_copilot_data"     # .npy training data

def _default_rigid_props(disable_gravity: bool = False) -> sim_utils.RigidBodyPropertiesCfg:
    return sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=disable_gravity,
        max_depenetration_velocity=5.0,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=3666.0,
        enable_gyroscopic_forces=True,
        solver_position_iteration_count=192,
        solver_velocity_iteration_count=1,
        max_contact_impulse=1e32,
    )


def _default_collision_props() -> sim_utils.CollisionPropertiesCfg:
    return sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0)


@configclass
class FixedAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0
    height: float = 0.0
    base_height: float = 0.0
    friction: float = 0.75
    mass: float = 0.05


@configclass
class HeldAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0
    height: float = 0.0
    friction: float = 0.75
    mass: float = 0.05


@configclass
class RobotCfg:
    robot_file: str = "robot/xarm7_gripper.usd"
    friction: float = 0.75

    sim_fingertip2eef = [0.0, 0.0, 0.165]
    real_fingertip2eef = [0.0, 0.0, 0.23]


@configclass
class AssemblyTask:
    robot_cfg: RobotCfg = RobotCfg()
    name: str = ""
    duration_s = 5.0

    fixed_asset_cfg: FixedAssetCfg = FixedAssetCfg()
    held_asset_cfg: HeldAssetCfg = HeldAssetCfg()

    # Per-episode object-pose randomization. When False, tasks place their assets
    # at fixed poses on reset. Subclasses that set this True must also provide the
    # spawn-range fields their reset path reads (see RandomBlock).
    randomize_positions: bool = False

    success_threshold: float = 0.5

    # gripper clamp values
    gripper_obs_clamp: float = 1.0
    gripper_ctrl_clamp: float = 1.6

    # rewards
    task_success_reward_scale = 30.0
    termination_reward_scale = 50.0
    action_smoothing_reward_scale = 0.1
    xy_aligned_reward_scale = 0.05
    action_norm_reward_scale = 0.3
    tilt_penalty_reward_scale = 1.0
    force_penalty_reward_scale = 0.2


@configclass
class Peg8mm(HeldAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_peg_8mm.usd"
    diameter = 0.007986
    height = 0.050
    mass = 0.019
    grasp_offset = [0.0, 0.0, 0.02]


@configclass
class Hole8mm(FixedAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_hole_8mm.usd"
    diameter = 0.0081
    height = 0.025
    base_height = 0.0


@configclass
class PegInsert(AssemblyTask):
    # data
    train_data_path: str = "teleop/peginsert_train_data.npy"

    # checkpoints
    dp_teleop_path: str = "shared_autonomy_policies/bc_teleop/PegInsert_bc_teleop"
    dp_expert_path: str = "shared_autonomy_policies/bc_expert/PegInsert_bc_expert"

    name = "peg_insert"
    fixed_asset_cfg = Hole8mm()
    held_asset_cfg = Peg8mm()
    duration_s = 20.0
    close_gripper: float = 0.93
    success_threshold: float = 0.04
    gripper_obs_clamp: float = 0.94

    fixed_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=fixed_asset_cfg.usd_path,
            activate_contact_sensors=True,
            rigid_props=_default_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=_default_collision_props(),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )
    held_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=held_asset_cfg.usd_path,
            activate_contact_sensors=True,
            rigid_props=_default_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=held_asset_cfg.mass),
            collision_props=_default_collision_props(),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )

    held_asset_contact_sensor_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/HeldAsset/forge_round_peg_8mm",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
    )


@configclass
class GearBase(FixedAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_gear_base.usd"
    height = 0.02
    base_height = 0.005

    small_gear_base_offset = [5.075e-2, 0.0, 0.0]
    medium_gear_base_offset = [2.025e-2, 0.0, 0.0]
    large_gear_base_offset = [-3.025e-2, 0.0, 0.0]


@configclass
class MediumGear(HeldAssetCfg):
    usd_path = f"{ASSET_DIR}/factory_gear_medium.usd"
    diameter = 0.03  # Used for gripper width.
    height: float = 0.03
    mass = 0.012
    grasp_offset = [0.0, 0.0, 0.025]


@configclass
class GearMesh(AssemblyTask):
    # data
    train_data_path: str = "teleop/gearmesh_train_data.npy"

    # checkpoints
    dp_teleop_path: str = "shared_autonomy_policies/bc_teleop/GearMesh_bc_teleop"
    dp_expert_path: str = "shared_autonomy_policies/bc_expert/GearMesh_bc_expert"

    name = "gear_mesh"
    fixed_asset_cfg = GearBase()
    held_asset_cfg = MediumGear()
    duration_s = 20.0
    close_gripper: float = 0.69
    success_threshold: float = 0.05
    gripper_obs_clamp: float = 0.695
    gripper_ctrl_clamp: float = 1.2

    small_gear_usd = f"{ASSET_DIR}/factory_gear_small.usd"
    large_gear_usd = f"{ASSET_DIR}/factory_gear_large.usd"

    small_gear_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/SmallGearAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=small_gear_usd,
            activate_contact_sensors=True,
            rigid_props=_default_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
            collision_props=_default_collision_props(),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )

    large_gear_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/LargeGearAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=large_gear_usd,
            activate_contact_sensors=True,
            rigid_props=_default_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
            collision_props=_default_collision_props(),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )

    fixed_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=fixed_asset_cfg.usd_path,
            activate_contact_sensors=True,
            rigid_props=_default_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=_default_collision_props(),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )
    held_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=held_asset_cfg.usd_path,
            activate_contact_sensors=True,
            rigid_props=_default_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=held_asset_cfg.mass),
            collision_props=_default_collision_props(),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )

    held_asset_contact_sensor_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/HeldAsset/factory_gear_medium",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
    )

@configclass
class ThreeBlocks(AssemblyTask):
    train_data_path = "logs/data/threeblocks_seq_demos.npy"
    # train_data_path: str = "teleop/nutthread_train_data.npy"
    dp_expert_path: str = (
        "/home/wisc-rt2-trimanual/xarm-shared-autonomy/outputs/train/"
        "threeblocks_expert_50_bc_expert/checkpoints/060000/pretrained_model"
    )
    name = "three_blocks"
    duration_s = 20.0
    close_gripper: float = 0.7
    success_threshold: float = 0.05
    gripper_obs_clamp: float = 0.7
    gripper_ctrl_clamp: float = 1.2

    fixed_asset_cfg = FixedAssetCfg()
    held_asset_cfg = HeldAssetCfg()

    block_a_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/BlockA",
        spawn=sim_utils.CuboidCfg(
            size=(0.03, 0.03, 0.03),
            rigid_props=_default_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
            collision_props=_default_collision_props(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            activate_contact_sensors = True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, -0.15, 0.05), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    block_b_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/BlockB",
        spawn=sim_utils.CuboidCfg(
            size=(0.03, 0.03, 0.03),
            rigid_props=_default_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
            collision_props=_default_collision_props(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    block_c_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/BlockC",
        spawn=sim_utils.CuboidCfg(
            size=(0.03, 0.03, 0.03),
            rigid_props=_default_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
            collision_props=_default_collision_props(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.15, 0.05), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # Reverted to the original solid bin (matching the geometry threeblocks_seq_demos.npy
    # was actually recorded against) — the hollow walled version kept causing erratic
    # arm motion on wall contact, even after tuning the admittance velocity clamp
    # (raised ceiling for free motion + force-gated tight clamp on contact spikes) —
    # the instability is deeper than a velocity cap, likely the contact response itself
    # against thin kinematic walls. Re-introduce walls only alongside a real fix to that
    # (e.g. softened wall collision stiffness) plus a re-recorded demo set.
    #
    # The bin is kinematic (kinematic_enabled=True): it is a placement target, not an
    # object to manipulate, so it must never be pushed by the arm or by dropped blocks.
    fixed_asset: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Bin",
        spawn=sim_utils.CuboidCfg(
            size=(0.12, 0.12, 0.03),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=_default_collision_props(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.65, 0.0, 0.0025), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    held_asset = block_a_cfg

    held_asset_contact_sensor_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/BlockA",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
    )


@configclass
class RandomBlock(ThreeBlocks):
    """ThreeBlocks with per-episode randomized block positions and a fixed bin.

    This is the base task for the VLA shared-autonomy study. It reuses the
    ThreeBlocks scene, success check, and termination logic (``name`` stays
    ``"three_blocks"``), but the three block xy positions are resampled
    uniformly over a rectangular region on every reset by
    ``XArmEnv._sample_block_layout`` instead of being hard-coded to a fixed row.
    A policy therefore cannot memorize one trajectory and has to localize the
    blocks from observation.

    The bin is **not** randomized: it spawns at ``bin_pos`` every episode and is
    kinematic (immovable — it never drifts under contact). To build the
    "challenging task" held-out variation, widen ``block_x_range`` /
    ``block_y_range`` or point the train and eval configs at disjoint
    sub-regions.
    """

    name = "three_blocks"  # reuse three_blocks env branching
    randomize_positions: bool = True

    # Single-target pick: each episode names one block colour ("pick up the <colour>
    # block and put it in the bin"); the other two are distractors that must stay
    # out of the bin. Colour index maps to block A / B / C = red / green / blue.
    single_target: bool = True
    target_colors: tuple = ("red", "green", "blue")
    # Which colour indices episodes may target. Restrict this (e.g. (0, 1)) to hold a
    # colour out for the "challenging task" eval set.
    allowed_target_idx: tuple = (0, 1, 2)
    instruction_template: str = "pick up the {color} block and put it in the bin"

    # Block spawn region: xy sampled uniformly in this rectangle, rejection-resampled
    # so no two blocks are within `block_min_separation` and no block lands within
    # `bin_clearance` of the bin centre. Kept close to the arm base for comfortable
    # SpaceMouse teleop (blocks in front, bin just behind them).
    # NOTE (two-camera branch): this layout differs from main's. The bin has moved to
    # the robot's right (-y) so it sits under the square-on side camera, and the block
    # region has shifted to +y to keep clear of it. Data recorded here is NOT compatible
    # with main's datasets or checkpoints — different scene AND different cameras.
    # Square 16 x 16 cm. The original region was 14 x 30 cm, nearly twice as wide as
    # deep, which biases the task toward left-right placement; a square region spreads
    # the difficulty evenly over both axes.
    block_x_range: tuple = (0.29, 0.45)
    block_y_range: tuple = (-0.06, 0.10)
    block_min_separation: float = 0.06
    # Blocks are rejected within this radius of the bin centre. Sized from the geometry
    # rather than guessed: the bin is 12 cm square (half-diagonal 0.085) and a block is
    # 3 cm (half-diagonal 0.021), so 0.106 is the contact limit. 0.12 leaves ~4.5 cm of
    # clear table between bin edge and block edge at the closest, which is what lets
    # blocks spawn visibly near the bin instead of being pushed to the far side.
    bin_clearance: float = 0.12
    # Blocks are also rejected within this radius of the gripper's start xy, so no
    # episode begins with a block under the fingers. Sized against usable area, not
    # just safety: the start pose sits inside the spawn square, so the exclusion disc
    # comes straight out of it — 0.08 would leave only 29% of the region usable and
    # bank every block into the corners. 0.05 clears the gripper footprint plus a
    # block half-diagonal and keeps 70%. 0 disables the check.
    start_clearance: float = 0.05
    # When set, every reset draws the SAME block layout (the sampler runs off a
    # generator re-seeded with this value each call). Used to collect "same scene,
    # different instruction" demos: replay one layout across all three colours so
    # the instruction, not the layout, is the only thing that predicts the action.
    # Leave None for the usual fresh-layout-per-episode behaviour.
    layout_seed: int | None = None
    # Resting centre height of a 3 cm block on the table. Measured from recorded
    # demos: blocks settle at z ~= 0.0107, so the inherited ThreeBlocks value of
    # 0.05 spawned them ~4 cm in the air and they fell during reset (leaving the
    # first recorded frame of every episode with the block still mid-drop).
    block_spawn_z: float = 0.012

    # Fixed bin pose, pinned every reset. The bin geometry (kinematic, immovable)
    # is inherited from ThreeBlocks.fixed_asset.
    # To the robot's right (-y, since the base faces +x) and under the side camera at
    # (0.37, -0.55, 0.62). Its x matches the block region's centre x (0.37), so the bin
    # centre and the block-region centre lie on a single line parallel to y — the same
    # line the side camera looks along. Reach radius 0.41 m, well inside the 0.54 m of
    # the original straight-ahead bin, so teleop stays comfortable.
    bin_pos: tuple = (0.37, -0.18, 0.0025)

    # Starting fingertip pose: pulled in over the block region and lifted well
    # clear of the blocks so the arm starts fairly upright, not over-extended.
    start_eef_xy: tuple = (0.36, 0.0)
    start_eef_z: float = 0.18


@configclass
class GearMeshIntent(GearMesh):
    name = "gear_mesh_intent"

    # Local override while training behaviors are being finalized; resolve_hf
    # passes absolute paths through unchanged.
    train_data_path: str = (
        "/home/shuo/projects/Residual_Copilot_Deployment/logs/"
        "shared_autonomy_online/260507_gearmesh_intent_processed/data/teleop_data.npy"
    )

    # Per-episode intent: list of ints in {0, 1, 2} = {small, medium, large}.
    # Length must equal the number of episodes in train_data_path.
    # Empty -> defaults to all-medium (matches gear_mesh).
    eps_intent: list = [0] * 5 + [1] * 5 + [2] * 5


@configclass
class NutM32(HeldAssetCfg):
    usd_hf_file = "objects/m36_nut.usd"
    diameter = 0.024
    height = 0.01
    mass = 0.03
    friction = 0.01  # Additive with the nut means friction is (-0.25 + 0.75)/2 = 0.25

@configclass
class BoltM32(FixedAssetCfg):
    usd_hf_file = "objects/m36_bolt.usd"
    diameter = 0.024
    height = 0.025
    base_height = 0.01
    thread_pitch = 0.002
    nut_offset = [0.0, 0.0, 0.04]

@configclass
class NutThread(AssemblyTask):
    name = "nut_thread"

    train_data_path: str = "teleop/nutthread_train_data.npy"
    dp_teleop_path: str = "shared_autonomy_policies/bc_teleop/NutThread_bc_teleop"
    dp_expert_path: str = "shared_autonomy_policies/bc_expert/NutThread_bc_expert"

    fixed_asset_cfg = BoltM32()
    held_asset_cfg = NutM32()
    asset_size = 16.0
    duration_s = 40.0
    close_gripper: float = 0.39
    success_threshold: float = 0.375
    success_rotation_threshold_deg: float = 90.0
    gripper_obs_clamp: float = 0.40
    gripper_ctrl_clamp: float = 0.75

    fixed_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=resolve_hf(HF_ASSETS_REPO, fixed_asset_cfg.usd_hf_file),
            activate_contact_sensors=True,
            rigid_props=_default_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=_default_collision_props(),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )
    held_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=resolve_hf(HF_ASSETS_REPO, held_asset_cfg.usd_hf_file),
            activate_contact_sensors=True,
            rigid_props=_default_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=held_asset_cfg.mass),
            collision_props=_default_collision_props(),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )

    held_asset_contact_sensor_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/HeldAsset/m36_nut",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
    )


@configclass
class TubeTransfer(AssemblyTask):
    """Move a test tube from a source rack to an empty destination rack.

    The task for the adaptive motion scaling study. Each trip has precise phases (grasp
    the tube, pull it straight out, line it up over a hole, lower it in) separated by a
    fast, coarse carry across the table, so the right SpaceMouse gain changes several
    times per episode.

    The cameras see LabUtopia's test tube and racks, shortened to the configured number of
    holes (see ``lab_assets.py``). Physics uses invisible solid stand-ins: a cylinder for the tube and
    thick boxes around a row of square holes for each rack. The visual rack is stretched sideways so its round holes
    match the collider holes, ``tube diameter + rack_clearance`` wide.

    Both racks lie lengthwise along x (pointing away from the robot) with their holes in a
    row. Coordinates are in the robot base frame (x forward, y left, z up), metres.
    """

    name = "tube_transfer"
    duration_s = 90.0   # informational; the env reads XArmTubeTransferCfg.episode_length_s
    instruction: str = "move the test tube to the empty rack"

    # --- Tube (collider: solid cylinder) ---
    # 2.4 cm diameter. A 1.5 cm tube is too thin for this gripper in sim: the fingers
    # only reach it near full close (1 of 2 scripted transfers held; with the command
    # capped at 1.1 it was never gripped).
    tube_radius: float = 0.012
    tube_height: float = 0.15         # a 25 x 150 mm test tube
    tube_mass: float = 0.03

    # --- Racks ---
    # Holes per rack, in a row (1-6; LabUtopia's model has 6 and is shortened to fit).
    # The source has three, so where the tube starts varies. The destination has one, so
    # the operator and the policy always share the same goal: with several target holes
    # the policy could head for a different one than the operator, and blending would then
    # pull against them.
    source_rack_holes: int = 3
    dest_rack_holes: int = 1
    # Hole width = tube diameter + clearance. Larger is easier to insert into.
    rack_clearance: float = 0.006
    # Height scale for the rack model (its upper plate sits 9.5 cm above the table at 1.0).
    rack_scale_z: float = 1.0

    # --- Appearance ---
    # False shows the plain collider shapes instead (no download needed).
    use_lab_visuals: bool = True
    # Which tube of LabUtopia's TestTube_02 set to use, 0-4 (see lab_assets.TUBE_LIQUID).
    lab_tube_index: int = 1           # orange liquid, about half full
    # Colours of the collider shapes, used only when use_lab_visuals is False.
    tube_color: tuple = (0.15, 0.45, 0.85)
    source_rack_color: tuple = (0.55, 0.55, 0.55)
    dest_rack_color: tuple = (0.85, 0.85, 0.85)

    # --- Layout ---
    # Rack positions are the middle of the hole row, resampled uniformly in these regions
    # every episode; the tube starts in a random source hole. Every carry is roughly
    # 0.30-0.45 m. The destination moves too (+-3 cm around the spot under the side
    # camera) so the target must be seen: with a fixed target, a hand-written "slow down
    # near this point" rule could stand in for policy-informed scaling.
    source_x_range: tuple = (0.32, 0.42)
    source_y_range: tuple = (0.14, 0.22)
    dest_x_range: tuple = (0.34, 0.40)
    dest_y_range: tuple = (-0.21, -0.15)
    # When set, every reset draws the SAME layout (rack positions and starting hole), for
    # practice, debugging, or a fixed set of evaluation layouts. None = fresh each episode.
    layout_seed: int | None = None
    # Height of the table surface in the robot base frame. Measured: a free tube settles
    # with its bottom at z = -0.0045 (RandomBlock's blocks give the same, -0.0043).
    table_top_z: float = -0.0045

    # --- Robot start: fingertip between the racks, clear above the tube ---
    start_eef_xy: tuple = (0.37, 0.0)
    start_eef_z: float = 0.27

    # --- Success: tube seated in the destination hole, upright, and let go ---
    seat_tolerance: float = 0.01      # tube bottom within this of the table
    upright_tolerance_deg: float = 15.0
    release_gripper_thresh: float = 0.2   # gripper obs below this counts as open
    # Failure: tube knocked over, or dropped off the table.
    tipped_deg: float = 45.0

    # gripper clamp values
    gripper_obs_clamp: float = 1.0
    # Cap on the closing command (joint target, full close = 1.6). Measured with the
    # 2.4 cm tube: 1.1 never touches it (0/4 scripted transfers), 1.6 squeezes it out of
    # the fingers (0/6), 1.3 holds it (4/4). Retune if tube_radius changes.
    gripper_ctrl_clamp: float = 1.3

    # Only the friction values are read here; geometry comes from the fields above.
    held_asset_cfg = HeldAssetCfg(friction=1.0)
    fixed_asset_cfg = FixedAssetCfg(friction=0.5)

    held_asset_contact_sensor_cfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Tube",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
    )
