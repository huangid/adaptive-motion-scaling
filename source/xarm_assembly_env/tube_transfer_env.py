"""Tube-transfer task: move a test tube from a source rack to an empty destination rack.

Kept out of ``XArmEnv`` on purpose: the base class branches on task names in dozens of
places, and this subclass replaces only the pieces the task needs (scene, reset, success)
so the inherited tasks are untouched.

Scene (robot base frame, x forward, y left; racks lie lengthwise along x):

    source rack (random xy, +y side)  --- carry ~0.3-0.45 m --->  destination rack (random xy, -y side)
    tube in a random one of 3 holes                                  1 empty hole

Physics and appearance are separate. Each rack is one kinematic rigid body of thick boxes
around a row of square holes, and the tube is a solid cylinder; these colliders are invisible.
The cameras see LabUtopia's rack and test tube (``lab_assets.py``) laid over them.
"""

import torch

import isaacsim.core.utils.torch as torch_utils  # type: ignore
from pxr import Usd, UsdGeom  # type: ignore

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sensors import ContactSensor, TiledCamera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers import VisualizationMarkers

from ..utils.utils import set_friction
from residual_copilot.pilot_models.spacemouse_pilot import SpaceMousePilot
from . import lab_assets
from .xarm_env import XArmEnv
from .xarm_env_guided_diffusion import XArmEnvGuidedDiffusion

_ENVS = "/World/envs/env_.*"
# Turns the rack model's hole row (its local y axis) onto the world x axis.
_YAW_MINUS_90 = (0.70710678, 0.0, 0.0, -0.70710678)


def rack_geometry(task_cfg, n_holes: int) -> dict:
    """Collider rack dimensions, derived from the LabUtopia rack scaled to fit the tube.

    The model's round holes are stretched sideways (x and y by ``scale``) until they are
    ``2 * tube_radius + rack_clearance`` wide; the square collider holes have that side.
    Rack frame: origin at the middle of the hole row on the table, holes along +x.
    """
    r = lab_assets.RACK
    hole = 2.0 * task_cfg.tube_radius + task_cfg.rack_clearance
    scale = hole / (2.0 * r["hole_radius"])
    pitch = r["hole_pitch"] * scale
    n = n_holes
    return dict(
        n_holes=n,
        hole=hole,
        scale=scale,
        hole_x=[(i - (n - 1) / 2.0) * pitch for i in range(n)],
        length=lab_assets.rack_length(n) * scale,
        depth=r["depth"] * scale,
        height=r["top_z"] * task_cfg.rack_scale_z,
    )


def rack_bar_layout(geom: dict):
    """Boxes that make a solid rack with a row of square holes along x.

    Returns ``[(size_xyz, center_xyz), ...]`` in the rack frame: two long side bars, plus
    end caps and the separators between holes.
    """
    a, length, depth, height = geom["hole"], geom["length"], geom["depth"], geom["height"]
    side = (depth - a) / 2.0
    if side <= 0.0:
        raise ValueError(f"hole {a:.4f} is not narrower than the rack depth {depth:.4f}")
    bars = [
        ((length, side, height), (0.0, +(a / 2.0 + side / 2.0), height / 2.0)),
        ((length, side, height), (0.0, -(a / 2.0 + side / 2.0), height / 2.0)),
    ]
    edges = [-length / 2.0]
    for x in geom["hole_x"]:
        edges += [x - a / 2.0, x + a / 2.0]
    edges.append(length / 2.0)
    for lo, hi in zip(edges[0::2], edges[1::2]):
        if hi - lo <= 1e-4:
            raise ValueError("rack holes overlap; reduce rack_clearance")
        bars.append(((hi - lo, a, height), ((lo + hi) / 2.0, 0.0, height / 2.0)))
    return bars


def _hide(stage, root):
    """Hide a spawned collider shape. Its ``geometry`` prim is untyped, so hide what is inside."""
    for prim in Usd.PrimRange(stage.GetPrimAtPath(f"{root}/geometry")):
        if prim.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(prim).MakeInvisible()


class _HoldPilot:
    """Placeholder base pilot: commands the current pose, so the arm holds still.

    Scripts such as ``eval_smolvla.py`` set ``pilot_model="knn"`` only because the env
    requires some pilot; the base action is unused there. This task has no demo file for
    kNN to load yet, so non-SpaceMouse runs get this instead.
    """

    def get_actions(self, eidx, pos, quat=None, grip=None, verbose=False):
        return torch.cat([pos, quat, grip], dim=-1)

    def clear(self, env_ids=None):
        pass


class XArmTubeTransferEnv(XArmEnv):
    """XArm env for the tube-transfer task (see module docstring)."""

    # -------------------------------------------------------------------------
    # Scene
    # -------------------------------------------------------------------------

    def _setup_scene(self):
        t = self.cfg_task
        self.source_geom = rack_geometry(t, t.source_rack_holes)
        self.dest_geom = rack_geometry(t, t.dest_rack_holes)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))
        table_cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        table_cfg.func(
            f"{_ENVS}/Table", table_cfg, translation=(0.55, 0.0, -0.0015), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        self._robot = Articulation(self.cfg.robot)

        self._build_rack("SourceRack", self.source_geom, t.source_rack_color)
        self._build_rack("DestRack", self.dest_geom, t.dest_rack_color)
        self._source_rack = RigidObject(RigidObjectCfg(prim_path=f"{_ENVS}/SourceRack", spawn=None))
        # The destination rack is the base class's "fixed asset" (the target) and the
        # tube its "held asset", so observations and rewards work unchanged.
        self._fixed_asset = RigidObject(RigidObjectCfg(prim_path=f"{_ENVS}/DestRack", spawn=None))
        self._held_asset = RigidObject(RigidObjectCfg(
            prim_path=f"{_ENVS}/Tube",
            spawn=sim_utils.CylinderCfg(
                radius=t.tube_radius,
                height=t.tube_height,
                axis="Z",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=5.0,
                    solver_position_iteration_count=192,
                    solver_velocity_iteration_count=1,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=t.tube_mass),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=t.tube_color),
                activate_contact_sensors=True,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.37, 0.17, 0.08)),
        ))
        if t.use_lab_visuals:
            self._dress_tube()

        self.eef_contact_sensor = ContactSensor(self.cfg.eef_contact_sensor_cfg)
        self.scene.sensors["eef_contact_sensor"] = self.eef_contact_sensor
        self.held_asset_contact_sensor = ContactSensor(t.held_asset_contact_sensor_cfg)
        self.scene.sensors["held_asset_contact_sensor"] = self.held_asset_contact_sensor

        if self.cfg.vis.store_rgb:
            if getattr(self.cfg, "use_front_camera", True):
                self.front_camera = TiledCamera(self.cfg.front_camera_cfg)
                self.scene.sensors["front_camera"] = self.front_camera
            if getattr(self.cfg, "use_wrist_camera", False):
                self.wrist_camera = TiledCamera(self.cfg.wrist_camera_cfg)
                self.scene.sensors["wrist_camera"] = self.wrist_camera
            if getattr(self.cfg, "use_upper_right_camera", False):
                self.upper_right_camera = TiledCamera(self.cfg.upper_right_camera_cfg)
                self.scene.sensors["upper_right_camera"] = self.upper_right_camera

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["fixed_asset"] = self._fixed_asset
        self.scene.rigid_objects["held_asset"] = self._held_asset
        self.scene.rigid_objects["source_rack"] = self._source_rack

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        if self.cfg.vis.vis_obs:
            self.held_asset_marker = VisualizationMarkers(self.cfg.vis.frame_marker_cfg)
            self.fixed_asset_marker = VisualizationMarkers(self.cfg.vis.frame_marker_cfg)

    def _build_rack(self, name, g, color):
        """One rack per env: a kinematic Xform body with box colliders, plus the visual model."""
        t = self.cfg_task
        env_paths = sim_utils.find_matching_prim_paths(_ENVS)
        for env in env_paths:
            sim_utils.create_prim(f"{env}/{name}", prim_type="Xform")
        bars = rack_bar_layout(g)
        for i, (size, center) in enumerate(bars):
            bar_cfg = sim_utils.CuboidCfg(
                size=size,
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
            bar_cfg.func(f"{_ENVS}/{name}/bar_{i}", bar_cfg, translation=center)

        stage = sim_utils.get_current_stage()
        rack_usd = str(lab_assets.rack_usd(g["n_holes"])) if t.use_lab_visuals else None
        s, sz = g["scale"], t.rack_scale_z
        r = lab_assets.RACK
        row_y = lab_assets.rack_row_center_y(g["n_holes"])
        for env in env_paths:
            root = f"{env}/{name}"
            sim_utils.define_rigid_body_properties(
                root, sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            )
            if rack_usd is None:
                continue
            for i in range(len(bars)):
                _hide(stage, f"{root}/bar_{i}")
            sim_utils.create_prim(
                f"{root}/visual", prim_type="Xform", usd_path=rack_usd,
                translation=(-row_y * s, 0.0, -r["base_z"] * sz),
                orientation=_YAW_MINUS_90,
                scale=(s, s, sz),
            )

    def _dress_tube(self):
        """Hide the collider cylinder and attach one LabUtopia test tube in its place."""
        t = self.cfg_task
        tube_usd = lab_assets.single_tube_usd(t.lab_tube_index)
        # Fit the model's measured bounds (it is in centimetres) exactly onto the collider.
        lo, hi = lab_assets.usd_bounds(tube_usd)
        s_xy = 2.0 * t.tube_radius / (hi[0] - lo[0])
        s_z = t.tube_height / (hi[2] - lo[2])
        cx, cy = (lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0
        stage = sim_utils.get_current_stage()
        for root in sim_utils.find_matching_prim_paths(f"{_ENVS}/Tube"):
            _hide(stage, root)
            sim_utils.create_prim(
                f"{root}/visual", prim_type="Xform", usd_path=str(tube_usd),
                translation=(-cx * s_xy, -cy * s_xy, -t.tube_height / 2.0 - lo[2] * s_z),
                scale=(s_xy, s_xy, s_z),
            )

    def _set_default_dynamics_parameters(self):
        super()._set_default_dynamics_parameters()
        set_friction(self._source_rack, self.cfg_task.fixed_asset_cfg.friction, self.scene.num_envs)

    # -------------------------------------------------------------------------
    # Init
    # -------------------------------------------------------------------------

    def _init_tensors(self):
        super()._init_tensors()
        self.source_rack_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.source_slot = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.dest_rack_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.source_hole_x = torch.tensor(self.source_geom["hole_x"], device=self.device)
        self.dest_hole_x = torch.tensor(self.dest_geom["hole_x"], device=self.device)
        self.instructions = [self.cfg_task.instruction for _ in range(self.num_envs)]
        self._layout_gen = None

    def _init_pilot(self):
        """Pilots without a demo file: the SpaceMouse, or a pose-holding placeholder."""
        if self.cfg.pilot_model == "spacemouse":
            self.pilot = SpaceMousePilot(num_envs=self.num_envs, device=self.device)
        elif self.cfg.pilot_model in ("knn", "replay"):
            print(f"[TubeTransfer] no demo data yet: pilot_model={self.cfg.pilot_model!r} "
                  "runs a placeholder that holds the current pose", flush=True)
            self.pilot = _HoldPilot()
        else:
            raise ValueError(f"TubeTransfer does not support pilot_model={self.cfg.pilot_model!r} yet")

        self.add_lag_to_base = self.cfg.pilot_type == "laggy"
        self.add_noise_to_base = self.cfg.pilot_type == "noisy"

        self.base_actions = torch.zeros((self.num_envs, 8), device=self.device)
        self.last_base_actions = torch.zeros((self.num_envs, 8), device=self.device)
        self.has_last_base = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        # No recorded start states: layouts are sampled and the start pose is fixed.
        self.initial_poses = None
        self.total_episodes = 1
        self.episode_idx = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

    # -------------------------------------------------------------------------
    # Success / termination
    # -------------------------------------------------------------------------

    def _tube_tilt_deg(self):
        """Angle between the tube's axis and world up, in degrees (ignores spin about its axis)."""
        up = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3)
        axis = torch_utils.quat_rotate(self.held_quat, up)
        return torch.rad2deg(torch.acos(axis[:, 2].clamp(-1.0, 1.0)))

    def _get_held_target_pos(self):
        """(tube bottom centre, floor centre of the nearest destination hole)."""
        half = torch.tensor([0.0, 0.0, -self.cfg_task.tube_height / 2.0], device=self.device).expand(self.num_envs, 3)
        tube_bottom = self.held_pos + torch_utils.quat_rotate(self.held_quat, half)
        dx = tube_bottom[:, 0:1] - (self.fixed_pos[:, 0:1] + self.dest_hole_x)   # (N, n_holes)
        nearest = dx.abs().argmin(dim=1)
        target = self.fixed_pos.clone()
        target[:, 0] += self.dest_hole_x[nearest]
        return tube_bottom, target

    def _check_success(self):
        t = self.cfg_task
        tube_bottom, hole_floor = self._get_held_target_pos()
        half = self.dest_geom["hole"] / 2.0
        in_hole = ((tube_bottom[:, :2] - hole_floor[:, :2]).abs() < half).all(dim=1)
        seated = (tube_bottom[:, 2] - hole_floor[:, 2]) < t.seat_tolerance
        upright = self._tube_tilt_deg() < t.upright_tolerance_deg
        released = self.gripper.squeeze(-1) < t.release_gripper_thresh
        success = in_hole & seated & upright & released
        assembly_error = torch.linalg.vector_norm(tube_bottom - hole_floor, dim=1)
        return success, assembly_error

    def _get_dones(self):
        self._compute_intermediate_values(dt=self.physics_dt)
        t = self.cfg_task

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        task_success, assembly_error = self._check_success()
        self.assembly_error = assembly_error
        self.first_success = task_success & (~self.ep_succeeded.to(torch.bool))
        self.ep_succeeded[task_success] = 1

        diverged = (~torch.isfinite(self.fingertip_midpoint_pos).all(dim=1)) | (
            torch.norm(self.fingertip_midpoint_pos, dim=1) > 2.0
        )
        tipped = self._tube_tilt_deg() > t.tipped_deg
        off_table = self.held_pos[:, 2] < t.table_top_z - 0.05
        self.tube_failed = tipped | off_table
        terminated = diverged | self.tube_failed | self.ep_succeeded.bool()

        done = time_out | terminated
        s = self.ep_succeeded[done].float()
        n = s.numel()
        if n > 0:
            alpha = self.ema_alpha
            exponents = torch.arange(n - 1, -1, -1, device=s.device, dtype=torch.float32)
            weights = (1.0 - alpha) ** exponents
            self.rolling_success_rate = (
                (1.0 - alpha) ** n * self.rolling_success_rate + alpha * (weights * s).sum()
            )
        self.extras["rolling_avg_succ_rate"] = float(self.rolling_success_rate)
        self.extras["assembly_error"] = float(assembly_error.mean())

        return terminated, time_out

    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------

    def _layout_generator(self):
        """Seeded like RandomBlock: fixed layout when layout_seed is set, else one stream per run."""
        t = self.cfg_task
        if t.layout_seed is not None:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(int(t.layout_seed))
            return gen
        if self._layout_gen is None:
            self._layout_gen = torch.Generator(device=self.device)
            seed = self.cfg.seed
            if seed is None:
                seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
            self._layout_gen.manual_seed(int(seed) + 7919)
        return self._layout_gen

    def _reset_asset_poses(self, env_ids, translation_noise, yaw_delta_quat, fixed_height_noise, identity_quat):
        """Place both racks at sampled positions and stand the tube in a random source hole."""
        t = self.cfg_task
        n = len(env_ids)
        origins = self.scene.env_origins[env_ids]

        gen = self._layout_generator()

        def uniform(x_range, y_range):
            lo = torch.tensor([x_range[0], y_range[0]], device=self.device)
            span = torch.tensor([x_range[1] - x_range[0], y_range[1] - y_range[0]], device=self.device)
            return lo + span * torch.rand(n, 2, device=self.device, generator=gen)

        src_xy = uniform(t.source_x_range, t.source_y_range)
        slot = torch.randint(0, len(self.source_hole_x), (n,), device=self.device, generator=gen)
        dst_xy = uniform(t.dest_x_range, t.dest_y_range)
        self.source_rack_xy[env_ids] = src_xy
        self.source_slot[env_ids] = slot
        self.dest_rack_xy[env_ids] = dst_xy

        src_state = torch.zeros((n, 13), device=self.device)
        src_state[:, 0:2] = src_xy
        src_state[:, 2] = t.table_top_z
        src_state[:, 3] = 1.0
        src_state[:, 0:3] += origins
        self._source_rack.write_root_pose_to_sim(src_state[:, 0:7], env_ids=env_ids)
        self._source_rack.write_root_velocity_to_sim(src_state[:, 7:], env_ids=env_ids)
        self._source_rack.reset(env_ids=env_ids)

        dest_pos = torch.zeros((n, 3), device=self.device)
        dest_pos[:, 0:2] = dst_xy
        dest_pos[:, 2] = t.table_top_z

        # Tube upright in the chosen source hole, its bottom 1 mm above the table.
        held_pos = torch.zeros((n, 3), device=self.device)
        held_pos[:, 0] = src_xy[:, 0] + self.source_hole_x[slot]
        held_pos[:, 1] = src_xy[:, 1]
        held_pos[:, 2] = t.table_top_z + t.tube_height / 2.0 + 0.001

        self._set_assets_state(
            held_pos=held_pos, held_quat=identity_quat,
            fixed_pos=dest_pos, fixed_quat=identity_quat,
            env_ids=env_ids,
        )

        # The base class turns this into fixed_pos_obs_frame: the top middle of the destination rack.
        fixed_tip_pos_local = torch.zeros((n, 3), device=self.device)
        fixed_tip_pos_local[:, 2] = self.dest_geom["height"]
        return fixed_tip_pos_local

    def _reset_robot_pose(self, env_ids, translation_noise, yaw_delta_quat, identity_quat):
        """Fixed start: fingertip between the racks, gripper pointing straight down."""
        t = self.cfg_task
        n = len(env_ids)
        sim_eef = torch.zeros((n, 7), device=self.device)
        sim_eef[:, 0] = t.start_eef_xy[0]
        sim_eef[:, 1] = t.start_eef_xy[1]
        sim_eef[:, 2] = t.start_eef_z
        sim_eef[:, 3:7] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device)  # 180 deg about x
        sim_eef[:, 0:3] = torch_utils.tf_combine(
            sim_eef[:, 3:7], sim_eef[:, 0:3], identity_quat, -self.sim_fingertip2eef[env_ids],
        )[1]
        init_qpos = self._robot.data.default_joint_pos[env_ids, :7]
        qpos = self.ik_controller.compute_ik(init_qpos=init_qpos, cartesian_target=sim_eef)
        self._set_default_robot_pose(joints=qpos, env_ids=env_ids)


class XArmTubeTransferGuidedDiffusionEnv(XArmTubeTransferEnv, XArmEnvGuidedDiffusion):
    """Tube transfer with 8D absolute actions (for policy-driven and blended control)."""
