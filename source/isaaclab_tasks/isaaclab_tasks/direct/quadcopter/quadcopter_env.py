# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sensors import Imu, ImuCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 2
    action_space = 4
    observation_space = 12  # lin_acc_b(3) + ang_vel_b(3) + euler_angles(3) + desired_pos_b(3)
    state_space = 0
    debug_vis = True

    ui_window_class_type = QuadcopterEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True, clone_in_fabric=True
    )

    # robot
    # Override the Crazyflie mass to 36.9 grams (0.0369 kg) via spawn.mass_props
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=CRAZYFLIE_CFG.spawn.replace(
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0369)
        ),
    )
    imu_sensor: ImuCfg = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/body",
        update_period=0.02,
        debug_vis=False,
    )

    # IMU noise parameters (realistic values for MEMS IMU like MPU6050/BMI160)
    @configclass
    class ImuNoiseCfg:
        # Accelerometer noise (m/s^2)
        lin_acc_noise_std: tuple[float, float, float] = (0.4, 0.4, 0.4)  # White noise
        lin_acc_bias_std: tuple[float, float, float] = (0.3, 0.3, 0.3)    # Bias drift
        # Gyroscope noise (rad/s)
        ang_vel_noise_std: tuple[float, float, float] = (0.02, 0.02, 0.02)  # White noise
        ang_vel_bias_std: tuple[float, float, float] = (0.01, 0.01, 0.01)    # Bias drift

    imu_noise: ImuNoiseCfg = ImuNoiseCfg()

    thrust_to_weight = 1.9
    moment_scale = 0.01

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0


class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Initialize IMU sensor AFTER simulation starts and environments are cloned
        # This ensures the sensor sees all environment instances
        self._imu = Imu(self.cfg.imu_sensor)
        # Trigger initialization if simulation is already playing (timeline callback won't fire)
        if not self._imu.is_initialized:
            self._imu._initialize_callback(None)
        
        # CRITICAL FIX: The IMU initialized before cloning, so its internal buffers are sized for 1 env
        # We must manually resize these buffers to match the actual number of environments
        actual_env_count = self.num_envs
        imu_env_count = getattr(self._imu, '_num_envs', None)
        if imu_env_count is not None and imu_env_count < actual_env_count:
            print(f"[IMU FIX] Resizing IMU buffers from {imu_env_count} to {actual_env_count} environments")
            # Resize sensor base buffers
            self._imu._num_envs = actual_env_count
            self._imu._is_outdated = torch.ones(actual_env_count, dtype=torch.bool, device=self.device)
            self._imu._timestamp = torch.zeros(actual_env_count, device=self.device)
            self._imu._timestamp_last_update = torch.zeros_like(self._imu._timestamp)
            # Reinitialize IMU-specific buffers with correct size
            self._imu._initialize_buffers_impl()
            # Initialize _dt attribute (normally set in update() method)
            self._imu._dt = self.cfg.sim.dt * self.cfg.decimation
            print(f"[IMU FIX] Buffers resized successfully")
        
        self.scene.sensors["imu"] = self._imu

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        
        # IMU noise and bias simulation
        imu_noise_cfg = self.cfg.imu_noise
        self._imu_lin_acc_noise_std = torch.tensor(imu_noise_cfg.lin_acc_noise_std, device=self.device).view(1, 3)
        self._imu_ang_vel_noise_std = torch.tensor(imu_noise_cfg.ang_vel_noise_std, device=self.device).view(1, 3)
        self._imu_lin_acc_bias_std = torch.tensor(imu_noise_cfg.lin_acc_bias_std, device=self.device).view(1, 3)
        self._imu_ang_vel_bias_std = torch.tensor(imu_noise_cfg.ang_vel_bias_std, device=self.device).view(1, 3)
        
        # Initialize bias and noise buffers
        self._imu_lin_acc_bias = torch.zeros(self.num_envs, 3, device=self.device)
        self._imu_ang_vel_bias = torch.zeros(self.num_envs, 3, device=self.device)
        self._imu_lin_acc_noise = torch.zeros_like(self._imu_lin_acc_bias)
        self._imu_ang_vel_noise = torch.zeros_like(self._imu_ang_vel_bias)
        
        # Check if noise/bias is enabled
        self._imu_has_lin_acc_noise = torch.any(self._imu_lin_acc_noise_std != 0.0).item()
        self._imu_has_ang_vel_noise = torch.any(self._imu_ang_vel_noise_std != 0.0).item()
        self._imu_has_lin_acc_bias = torch.any(self._imu_lin_acc_bias_std != 0.0).item()
        self._imu_has_ang_vel_bias = torch.any(self._imu_ang_vel_bias_std != 0.0).item()

        self._imu_log_interval = 100
        self._last_imu_log_step = -1

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
            ]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # NOTE: IMU initialization is deferred until after sim.reset() to ensure
        # all cloned environments are present when the sensor initializes its buffers.
        # The IMU will be created in __init__ after simulation starts.
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _resample_imu_bias(self, env_ids: torch.Tensor):
        """Resample IMU bias for the given environments (simulates bias drift on reset)."""
        if self._imu_has_lin_acc_bias:
            self._imu_lin_acc_bias[env_ids] = torch.randn_like(
                self._imu_lin_acc_bias[env_ids]
            ) * self._imu_lin_acc_bias_std
        if self._imu_has_ang_vel_bias:
            self._imu_ang_vel_bias[env_ids] = torch.randn_like(
                self._imu_ang_vel_bias[env_ids]
            ) * self._imu_ang_vel_bias_std

    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._desired_pos_w
        )
        self._log_imu_debug()

        # Use IMU sensor data matching real Crazyflie hardware
        # Crazyflie provides: acc.x/y/z, gyro.x/y/z, stabilizer.roll/pitch/yaw
        imu_data = self._imu.data
        
        # Apply realistic IMU noise and bias
        # Linear acceleration: raw + bias + white_noise (matches Crazyflie acc.x/y/z)
        lin_acc_b = imu_data.lin_acc_b.clone()
        if self._imu_has_lin_acc_bias:
            lin_acc_b += self._imu_lin_acc_bias
        if self._imu_has_lin_acc_noise:
            self._imu_lin_acc_noise = torch.randn_like(self._imu_lin_acc_noise) * self._imu_lin_acc_noise_std
            lin_acc_b += self._imu_lin_acc_noise
        
        # Angular velocity: raw + bias + white_noise (matches Crazyflie gyro.x/y/z)
        ang_vel_b = imu_data.ang_vel_b.clone()
        if self._imu_has_ang_vel_bias:
            ang_vel_b += self._imu_ang_vel_bias
        if self._imu_has_ang_vel_noise:
            self._imu_ang_vel_noise = torch.randn_like(self._imu_ang_vel_noise) * self._imu_ang_vel_noise_std
            ang_vel_b += self._imu_ang_vel_noise
        
        # Extract Euler angles from quaternion (matches Crazyflie stabilizer.roll/pitch/yaw)
        # This replaces projected_gravity_b which is not available on real hardware
        quat = self._robot.data.root_quat_w  # (num_envs, 4) in [w, x, y, z] format
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        euler_angles = self._quat_to_euler(quat)
        
        obs = torch.cat(
            [
                lin_acc_b,      # Noisy linear acceleration from IMU (3) - matches acc.x/y/z
                ang_vel_b,      # Noisy angular velocity from IMU (3) - matches gyro.x/y/z
                euler_angles,   # Euler angles (roll, pitch, yaw) (3) - matches stabilizer.roll/pitch/yaw
                desired_pos_b,  # Desired position in body frame (3)
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 2.0)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        # Resolve env ids into tensor/list forms for downstream use
        if env_ids is None:
            env_ids_tensor = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            reset_all = True
        else:
            if isinstance(env_ids, torch.Tensor):
                env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
            else:
                env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            reset_all = env_ids_tensor.numel() == self.num_envs
        env_ids_sequence: Sequence[int] = env_ids_tensor.tolist()

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids_tensor] - self._robot.data.root_pos_w[env_ids_tensor], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids_tensor])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids_tensor] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids_tensor]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids_tensor]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids_tensor)
        super()._reset_idx(env_ids_tensor)
        if reset_all:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # Resample IMU bias on environment reset (simulates sensor drift)
        self._resample_imu_bias(env_ids_tensor)
        
        self._actions[env_ids_tensor] = 0.0
        # Sample new commands
        self._desired_pos_w[env_ids_tensor, :2] = torch.zeros_like(
            self._desired_pos_w[env_ids_tensor, :2]
        ).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids_tensor, :2] += self._terrain.env_origins[env_ids_tensor, :2]
        self._desired_pos_w[env_ids_tensor, 2] = torch.zeros_like(
            self._desired_pos_w[env_ids_tensor, 2]
        ).uniform_(0.5, 1.5)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids_tensor]
        joint_vel = self._robot.data.default_joint_vel[env_ids_tensor]
        default_root_state = self._robot.data.default_root_state[env_ids_tensor]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids_tensor]
        # Robot write functions need tensor indices (they convert to PhysX indices internally)
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids_tensor)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids_tensor)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids_tensor)

        if reset_all:
            self._imu.reset(None)
        else:
            self._imu.reset(env_ids_tensor.tolist())

    def _quat_to_euler(self, quat: torch.Tensor) -> torch.Tensor:
        """Convert quaternion to Euler angles (roll, pitch, yaw) in radians.
        
        Args:
            quat: Quaternion tensor in [w, x, y, z] format, shape (num_envs, 4)
        
        Returns:
            Euler angles tensor (roll, pitch, yaw) in radians, shape (num_envs, 3)
        """
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = torch.where(
            torch.abs(sinp) >= 1,
            torch.copysign(torch.tensor(torch.pi / 2, device=quat.device), sinp),
            torch.asin(sinp)
        )
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        
        return torch.stack([roll, pitch, yaw], dim=-1)

    def _log_imu_debug(self):
        progress_buf = getattr(self, "progress_buf", None)
        if self.num_envs == 0 or progress_buf is None or progress_buf.numel() == 0:
            return
        step_count = int(progress_buf[0].item())
        if step_count % self._imu_log_interval != 0:
            return
        if step_count == self._last_imu_log_step:
            return
        imu_data = self._imu.data
        lin_acc = imu_data.lin_acc_b[0].detach().cpu().tolist()
        ang_vel = imu_data.ang_vel_b[0].detach().cpu().tolist()
        print(f"[IMU DEBUG] step={step_count} lin_acc={lin_acc} ang_vel={ang_vel}", flush=True)
        self._last_imu_log_step = step_count

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
