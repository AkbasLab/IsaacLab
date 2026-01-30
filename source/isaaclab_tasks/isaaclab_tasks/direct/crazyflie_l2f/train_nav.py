#!/usr/bin/env python3
"""
Crazyflie L2F Navigation Training Script - Single-Phase Approach

This script trains a Crazyflie 2.1 navigation policy that flies to random
goal positions. Key improvements over the two-phase approach:

1. SINGLE PHASE: No hover→navigate transition - learns unified navigation
2. RANDOM GOALS: Goals sampled on reset, larger range (±1m XY, 0.5-1.5m Z)
3. UNIFIED REWARD: Tanh-based distance reward (proven in quadcopter_env.py)
4. L2F COMPATIBLE: Same 146-dim observation for firmware export

The observation space matches what's available on the real Crazyflie:
- Position error (from Flow Deck + state estimator)
- Rotation matrix (from IMU)
- Velocity (from Flow Deck + state estimator)
- Angular velocity (from gyroscope)
- Action history (maintained in firmware)

Usage:
    # Training mode
    python train_nav_v2.py --num_envs 4096 --max_iterations 2000 --headless
    
    # Play mode with trained checkpoint
    python train_nav_v2.py --play --checkpoint checkpoints_nav_v2/best_model.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import math
from collections.abc import Sequence
from typing import Tuple

import torch
import torch.nn as nn
import gymnasium as gym

# Isaac Sim setup - must happen before other imports
from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description="Crazyflie L2F Navigation Training V2")
    
    # Mode selection
    parser.add_argument("--play", action="store_true", help="Run in play mode with trained model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for play mode")
    parser.add_argument("--resume_from", type=str, default=None, 
                        help="Resume training from checkpoint (e.g., hover checkpoint)")
    
    # Training parameters  
    parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments")
    parser.add_argument("--max_iterations", type=int, default=2000, help="Maximum training iterations")
    parser.add_argument("--save_interval", type=int, default=100, help="Save checkpoint every N iterations")
    
    # Navigation parameters
    parser.add_argument("--goal_xy_range", type=float, default=1.0, help="Max XY goal distance from origin (m)")
    parser.add_argument("--goal_z_min", type=float, default=0.5, help="Min goal height (m)")
    parser.add_argument("--goal_z_max", type=float, default=1.5, help="Max goal height (m)")
    
    # Hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    
    # AppLauncher adds its own args (including --headless)
    AppLauncher.add_app_launcher_args(parser)
    
    args = parser.parse_args()
    return args


# Parse args and launch Isaac Sim
args = parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import Isaac Lab modules
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import CUBOID_MARKER_CFG

# Import our custom Crazyflie configuration
from crazyflie_21_cfg import CRAZYFLIE_21_CFG, CrazyflieL2FParams


# ==============================================================================
# L2F Physics Constants
# ==============================================================================

class L2FConstants:
    """Physical parameters matching learning-to-fly exactly."""
    
    MASS = 0.027  # kg (27g)
    ARM_LENGTH = 0.028  # m (28mm)
    GRAVITY = 9.81  # m/s²
    
    IXX = 3.85e-6
    IYY = 3.85e-6
    IZZ = 5.9675e-6
    
    THRUST_COEFFICIENT = 3.16e-10  # N/RPM²
    TORQUE_COEFFICIENT = 0.005964552  # Nm/N
    RPM_MIN = 0.0
    RPM_MAX = 21702.0
    MOTOR_TIME_CONSTANT = 0.15  # seconds
    
    ROTOR_POSITIONS = [
        (0.028, -0.028, 0.0),
        (-0.028, -0.028, 0.0),
        (-0.028, 0.028, 0.0),
        (0.028, 0.028, 0.0),
    ]
    
    ROTOR_YAW_DIRS = [-1.0, 1.0, -1.0, 1.0]
    
    @classmethod
    def hover_rpm(cls) -> float:
        thrust_per_motor = cls.MASS * cls.GRAVITY / 4.0
        return math.sqrt(thrust_per_motor / cls.THRUST_COEFFICIENT)
    
    @classmethod
    def hover_action(cls) -> float:
        return 2.0 * cls.hover_rpm() / cls.RPM_MAX - 1.0


# ==============================================================================
# Environment Configuration
# ==============================================================================

@configclass
class CrazyflieNavV2EnvCfg(DirectRLEnvCfg):
    """Configuration for single-phase Crazyflie navigation environment."""
    
    # Episode settings
    episode_length_s = 10.0
    decimation = 1  # Control at physics rate (100 Hz)
    
    # Spaces - MUST match L2F exactly
    observation_space = 146
    action_space = 4
    state_space = 0
    debug_vis = True
    
    # Simulation - 100 Hz physics
    sim: SimulationCfg = SimulationCfg(
        dt=1/100,
        render_interval=2,
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
    
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True
    )
    
    robot: ArticulationCfg = CRAZYFLIE_21_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )
    
    # === GOAL SAMPLING ===
    goal_xy_range = 1.0  # Goals within ±1m of spawn in XY
    goal_z_min = 0.5     # Min goal height
    goal_z_max = 1.5     # Max goal height
    
    # === SPAWN CONFIGURATION ===
    # Spawn near center, let goal be anywhere in range
    spawn_xy_range = 0.3  # Spawn within ±0.3m of env origin
    spawn_z = 1.0         # Spawn at 1m height
    spawn_vel_range = 0.1  # Small initial velocity perturbation
    spawn_ang_range = 0.05  # Small angular perturbation
    
    # === REWARD WEIGHTS (inspired by quadcopter_env.py) ===
    # Distance to goal (primary objective)
    distance_reward_scale = 15.0  # Positive reward for being close
    distance_tanh_scale = 0.8     # Tanh falloff parameter
    
    # Reaching bonus
    reaching_threshold = 0.1  # m - considered "reached" if within 10cm
    reaching_bonus = 5.0      # One-time bonus for reaching goal
    
    # Stability penalties (keep drone stable while navigating)
    lin_vel_penalty = 0.05    # Penalize excessive linear velocity
    ang_vel_penalty = 0.5     # Penalize excessive angular velocity (INCREASED from 0.01)
    yaw_rate_penalty = 2.0    # Extra penalty for yaw rate specifically (NEW - prevents spinning)
    action_rate_penalty = 0.1  # Penalize rapid action changes (smooth control)
    
    # === TERMINATION THRESHOLDS ===
    term_xy_threshold = 2.0   # m - terminate if too far from origin
    term_z_min = 0.1          # m - terminate if too low
    term_z_max = 2.5          # m - terminate if too high
    term_tilt_threshold = 1.0  # rad (~57 deg) - terminate if too tilted
    term_lin_vel_threshold = 3.0  # m/s - terminate if moving too fast
    term_ang_vel_threshold = 5.0  # rad/s - terminate if spinning too fast (DECREASED from 10.0)
    
    # === DOMAIN RANDOMIZATION ===
    enable_disturbance = True
    disturbance_force_std = 0.0132  # N (mass * g / 20)
    disturbance_torque_std = 2.65e-5  # Nm
    
    # Action history
    action_history_length = 32


# ==============================================================================
# Environment Implementation
# ==============================================================================

class CrazyflieNavV2Env(DirectRLEnv):
    """Crazyflie navigation environment - single phase, random goals."""
    
    cfg: CrazyflieNavV2EnvCfg
    
    def __init__(self, cfg: CrazyflieNavV2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Cache physics parameters
        self._mass = L2FConstants.MASS
        self._thrust_coef = L2FConstants.THRUST_COEFFICIENT
        self._torque_coef = L2FConstants.TORQUE_COEFFICIENT
        self._motor_tau = L2FConstants.MOTOR_TIME_CONSTANT
        self._min_rpm = L2FConstants.RPM_MIN
        self._max_rpm = L2FConstants.RPM_MAX
        self._gravity = L2FConstants.GRAVITY
        self._hover_rpm = L2FConstants.hover_rpm()
        self._hover_action = L2FConstants.hover_action()
        self._dt = cfg.sim.dt
        
        self._motor_alpha = min(self._dt / self._motor_tau, 1.0)
        
        self._rotor_positions = torch.tensor(
            L2FConstants.ROTOR_POSITIONS, device=self.device, dtype=torch.float32
        )
        self._rotor_yaw_dirs = torch.tensor(
            L2FConstants.ROTOR_YAW_DIRS, device=self.device, dtype=torch.float32
        )
        
        # State tensors
        self._actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._rpm_state = torch.zeros(self.num_envs, 4, device=self.device)
        
        # Goal position (world frame)
        self._goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Track if goal was reached this episode (for one-time bonus)
        self._goal_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Force/torque buffers
        self._thrust_body = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._torque_body = torch.zeros(self.num_envs, 1, 3, device=self.device)
        
        # Action history buffer
        self._action_history = torch.zeros(
            self.num_envs, cfg.action_history_length, 4, device=self.device
        )
        
        # Disturbance forces
        self._disturbance_force = torch.zeros(self.num_envs, 3, device=self.device)
        self._disturbance_torque = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Episode statistics
        self._episode_sums = {
            "distance_reward": torch.zeros(self.num_envs, device=self.device),
            "reaching_bonus": torch.zeros(self.num_envs, device=self.device),
            "lin_vel_penalty": torch.zeros(self.num_envs, device=self.device),
            "ang_vel_penalty": torch.zeros(self.num_envs, device=self.device),
            "yaw_rate_penalty": torch.zeros(self.num_envs, device=self.device),
            "action_rate_penalty": torch.zeros(self.num_envs, device=self.device),
            "total_reward": torch.zeros(self.num_envs, device=self.device),
        }
        
        self._body_id = self._robot.find_bodies("body")[0]
        
        self.set_debug_vis(self.cfg.debug_vis)
        
        self._print_env_info()
    
    def _print_env_info(self):
        print("\n" + "="*60)
        print("Crazyflie L2F Navigation V2 (Single-Phase)")
        print("="*60)
        print(f"  Physics dt:        {self._dt*1000:.1f} ms ({1/self._dt:.0f} Hz)")
        print(f"  Episode length:    {self.cfg.episode_length_s:.1f} s")
        print(f"  Num envs:          {self.num_envs}")
        print(f"  Observation dim:   {self.cfg.observation_space}")
        print(f"  Action dim:        {self.cfg.action_space}")
        print(f"  Goal XY range:     ±{self.cfg.goal_xy_range:.1f} m")
        print(f"  Goal Z range:      {self.cfg.goal_z_min:.1f} - {self.cfg.goal_z_max:.1f} m")
        print(f"  Reaching threshold:{self.cfg.reaching_threshold:.2f} m")
        print(f"  Hover action:      {self._hover_action:.4f}")
        print("="*60 + "\n")
    
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        self.scene.clone_environments(copy_from_source=False)
        
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    
    def _quat_to_rotation_matrix(self, quat: torch.Tensor) -> torch.Tensor:
        """Convert quaternion [w,x,y,z] to flattened rotation matrix (9 elements)."""
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        
        r00 = 1 - 2*y*y - 2*z*z
        r01 = 2*x*y - 2*w*z
        r02 = 2*x*z + 2*w*y
        r10 = 2*x*y + 2*w*z
        r11 = 1 - 2*x*x - 2*z*z
        r12 = 2*y*z - 2*w*x
        r20 = 2*x*z - 2*w*y
        r21 = 2*y*z + 2*w*x
        r22 = 1 - 2*x*x - 2*y*y
        
        return torch.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=-1)
    
    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions using L2F motor model."""
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)
        
        target_rpm = (self._actions + 1.0) / 2.0 * self._max_rpm
        
        self._rpm_state = self._rpm_state + self._motor_alpha * (target_rpm - self._rpm_state)
        self._rpm_state = self._rpm_state.clamp(self._min_rpm, self._max_rpm)
        
        thrust_per_motor = self._thrust_coef * self._rpm_state ** 2
        
        total_thrust = thrust_per_motor.sum(dim=-1)
        
        thrust_body = torch.zeros(self.num_envs, 3, device=self.device)
        thrust_body[:, 2] = total_thrust
        
        roll_torque = (
            thrust_per_motor[:, 0] * self._rotor_positions[0, 1] +
            thrust_per_motor[:, 1] * self._rotor_positions[1, 1] +
            thrust_per_motor[:, 2] * self._rotor_positions[2, 1] +
            thrust_per_motor[:, 3] * self._rotor_positions[3, 1]
        )
        
        pitch_torque = -(
            thrust_per_motor[:, 0] * self._rotor_positions[0, 0] +
            thrust_per_motor[:, 1] * self._rotor_positions[1, 0] +
            thrust_per_motor[:, 2] * self._rotor_positions[2, 0] +
            thrust_per_motor[:, 3] * self._rotor_positions[3, 0]
        )
        
        yaw_torque = self._torque_coef * (
            self._rotor_yaw_dirs[0] * thrust_per_motor[:, 0] +
            self._rotor_yaw_dirs[1] * thrust_per_motor[:, 1] +
            self._rotor_yaw_dirs[2] * thrust_per_motor[:, 2] +
            self._rotor_yaw_dirs[3] * thrust_per_motor[:, 3]
        )
        
        torque_body = torch.stack([roll_torque, pitch_torque, yaw_torque], dim=-1)
        
        if self.cfg.enable_disturbance:
            thrust_body = thrust_body + self._disturbance_force
            torque_body = torque_body + self._disturbance_torque
        
        self._thrust_body[:, 0, :] = thrust_body
        self._torque_body[:, 0, :] = torque_body
        
        # Update action history
        self._action_history[:, :-1] = self._action_history[:, 1:].clone()
        self._action_history[:, -1] = self._actions
    
    def _apply_action(self):
        self._robot.set_external_force_and_torque(
            forces=self._thrust_body,
            torques=self._torque_body,
            body_ids=self._body_id,
        )
        self._robot.write_data_to_sim()
    
    def _get_observations(self) -> dict:
        """Construct observations matching L2F firmware format (146 dims).
        
        Observation = [pos_error(3), rot_matrix(9), lin_vel(3), ang_vel(3), action_history(128)]
        """
        pos_w = self._robot.data.root_pos_w
        quat_w = self._robot.data.root_quat_w
        lin_vel_w = self._robot.data.root_lin_vel_w
        ang_vel_b = self._robot.data.root_ang_vel_b
        
        # Position error relative to goal
        pos_error = pos_w - self._goal_pos_w
        pos_error_clipped = pos_error.clamp(-0.5, 0.5)
        
        lin_vel_clipped = lin_vel_w.clamp(-2.0, 2.0)
        
        rot_matrix = self._quat_to_rotation_matrix(quat_w)
        
        action_history_flat = self._action_history.view(self.num_envs, -1)
        
        obs = torch.cat([
            pos_error_clipped,   # 3
            rot_matrix,          # 9
            lin_vel_clipped,     # 3
            ang_vel_b,           # 3
            action_history_flat, # 128
        ], dim=-1)
        
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        """Compute reward using tanh-based distance metric (proven approach)."""
        cfg = self.cfg
        
        pos_w = self._robot.data.root_pos_w
        lin_vel = self._robot.data.root_lin_vel_w
        ang_vel = self._robot.data.root_ang_vel_b
        
        # Distance to goal
        distance_to_goal = torch.linalg.norm(self._goal_pos_w - pos_w, dim=1)
        
        # Tanh-based distance reward (1 at goal, 0 far away)
        distance_mapped = 1.0 - torch.tanh(distance_to_goal / cfg.distance_tanh_scale)
        distance_reward = distance_mapped * cfg.distance_reward_scale * self._dt
        
        # Reaching bonus (one-time when first reaching goal)
        just_reached = (distance_to_goal < cfg.reaching_threshold) & ~self._goal_reached
        reaching_bonus = just_reached.float() * cfg.reaching_bonus
        self._goal_reached = self._goal_reached | just_reached
        
        # Stability penalties
        lin_vel_sq = (lin_vel ** 2).sum(dim=-1)
        ang_vel_sq = (ang_vel ** 2).sum(dim=-1)
        yaw_rate_sq = ang_vel[:, 2] ** 2  # Z-axis angular velocity (yaw rate)
        
        lin_vel_penalty = lin_vel_sq * cfg.lin_vel_penalty * self._dt
        ang_vel_penalty = ang_vel_sq * cfg.ang_vel_penalty * self._dt
        yaw_rate_penalty = yaw_rate_sq * cfg.yaw_rate_penalty * self._dt
        
        # Action rate penalty (smooth control)
        action_rate = self._actions - self._prev_actions
        action_rate_sq = (action_rate ** 2).sum(dim=-1)
        action_rate_penalty = action_rate_sq * cfg.action_rate_penalty * self._dt
        
        # Total reward
        reward = distance_reward + reaching_bonus - lin_vel_penalty - ang_vel_penalty - yaw_rate_penalty - action_rate_penalty
        
        # Track stats
        self._episode_sums["distance_reward"] += distance_reward
        self._episode_sums["reaching_bonus"] += reaching_bonus
        self._episode_sums["lin_vel_penalty"] += lin_vel_penalty
        self._episode_sums["ang_vel_penalty"] += ang_vel_penalty
        self._episode_sums["yaw_rate_penalty"] += yaw_rate_penalty
        self._episode_sums["action_rate_penalty"] += action_rate_penalty
        self._episode_sums["total_reward"] += reward
        
        return reward
    
    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        cfg = self.cfg
        
        pos_w = self._robot.data.root_pos_w
        quat = self._robot.data.root_quat_w
        lin_vel = self._robot.data.root_lin_vel_w
        ang_vel = self._robot.data.root_ang_vel_b
        
        # XY position relative to env origin
        xy_offset = pos_w[:, :2] - self._terrain.env_origins[:, :2]
        xy_exceeded = torch.norm(xy_offset, dim=-1) > cfg.term_xy_threshold
        
        # Height check
        height = pos_w[:, 2] - self._terrain.env_origins[:, 2]
        too_low = height < cfg.term_z_min
        too_high = height > cfg.term_z_max
        
        # Tilt check
        qw = quat[:, 0]
        tilt_angle = 2.0 * torch.acos(torch.clamp(torch.abs(qw), 0.0, 1.0))
        too_tilted = tilt_angle > cfg.term_tilt_threshold
        
        # Velocity checks
        lin_vel_exceeded = torch.norm(lin_vel, dim=-1) > cfg.term_lin_vel_threshold
        ang_vel_exceeded = torch.norm(ang_vel, dim=-1) > cfg.term_ang_vel_threshold
        
        terminated = xy_exceeded | too_low | too_high | too_tilted | lin_vel_exceeded | ang_vel_exceeded
        
        return terminated, time_out
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specified environments with new random goals."""
        if env_ids is None or len(env_ids) == 0:
            return
        
        # Log stats before reset
        if len(env_ids) > 0 and hasattr(self, '_episode_sums'):
            extras = {}
            for key, values in self._episode_sums.items():
                avg = torch.mean(values[env_ids]).item()
                steps = self.episode_length_buf[env_ids].float().mean().item()
                if steps > 0:
                    extras[f"Episode/{key}"] = avg / max(steps, 1)
            self.extras["log"] = extras
        
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        
        n = len(env_ids)
        cfg = self.cfg
        
        # Reset goal reached flag
        self._goal_reached[env_ids] = False
        
        # Sample random spawn position (near center)
        spawn_pos = torch.zeros(n, 3, device=self.device)
        spawn_pos[:, 0] = (torch.rand(n, device=self.device) * 2 - 1) * cfg.spawn_xy_range
        spawn_pos[:, 1] = (torch.rand(n, device=self.device) * 2 - 1) * cfg.spawn_xy_range
        spawn_pos[:, 2] = cfg.spawn_z
        spawn_pos = spawn_pos + self._terrain.env_origins[env_ids]
        
        # Sample random goal position
        self._goal_pos_w[env_ids, 0] = self._terrain.env_origins[env_ids, 0] + \
            (torch.rand(n, device=self.device) * 2 - 1) * cfg.goal_xy_range
        self._goal_pos_w[env_ids, 1] = self._terrain.env_origins[env_ids, 1] + \
            (torch.rand(n, device=self.device) * 2 - 1) * cfg.goal_xy_range
        self._goal_pos_w[env_ids, 2] = self._terrain.env_origins[env_ids, 2] + \
            cfg.goal_z_min + torch.rand(n, device=self.device) * (cfg.goal_z_max - cfg.goal_z_min)
        
        # Identity quaternion with small perturbation
        quat = torch.zeros(n, 4, device=self.device)
        quat[:, 0] = 1.0
        
        # Small initial velocities
        lin_vel = (torch.rand(n, 3, device=self.device) * 2 - 1) * cfg.spawn_vel_range
        ang_vel = (torch.rand(n, 3, device=self.device) * 2 - 1) * cfg.spawn_ang_range
        
        # Write to sim
        root_pose = torch.cat([spawn_pos, quat], dim=-1)
        root_vel = torch.cat([lin_vel, ang_vel], dim=-1)
        
        self._robot.write_root_pose_to_sim(root_pose, env_ids)
        self._robot.write_root_velocity_to_sim(root_vel, env_ids)
        
        # Initialize motor state to hover RPM
        self._rpm_state[env_ids] = self._hover_rpm
        
        # Initialize action history to hover action
        self._action_history[env_ids] = self._hover_action
        self._actions[env_ids] = self._hover_action
        self._prev_actions[env_ids] = self._hover_action
        
        # Sample disturbances
        if cfg.enable_disturbance:
            self._disturbance_force[env_ids] = torch.randn(n, 3, device=self.device) * cfg.disturbance_force_std
            self._disturbance_torque[env_ids] = torch.randn(n, 3, device=self.device) * cfg.disturbance_torque_std
        
        # Reset stats
        for key in self._episode_sums:
            self._episode_sums[key][env_ids] = 0.0
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            marker_cfg = CUBOID_MARKER_CFG.copy()
            marker_cfg.markers["cuboid"].size = (0.1, 0.1, 0.1)
            marker_cfg.prim_path = "/Visuals/Command/goal_position"
            self._goal_markers = VisualizationMarkers(marker_cfg)
        else:
            if hasattr(self, "_goal_markers"):
                self._goal_markers.set_visibility(False)
    
    def _debug_vis_callback(self, event):
        if hasattr(self, "_goal_markers"):
            self._goal_markers.visualize(self._goal_pos_w)


# ==============================================================================
# L2F-Compatible Networks
# ==============================================================================

class L2FActorNetwork(nn.Module):
    """Actor network matching L2F architecture exactly."""
    
    HOVER_ACTION = L2FConstants.hover_action()
    
    def __init__(self, obs_dim: int = 146, hidden_dim: int = 64, action_dim: int = 4, init_std: float = 0.3):
        super().__init__()
        
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        self.log_std = nn.Parameter(torch.ones(action_dim) * math.log(init_std))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.fc1, self.fc2]:
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)
        hover_bias = math.atanh(max(-0.99, min(0.99, self.HOVER_ACTION)))
        nn.init.constant_(self.fc3.bias, hover_bias)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.fc1(obs))
        x = torch.tanh(self.fc2(x))
        mean = torch.tanh(self.fc3(x))
        return mean
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        mean = self.forward(obs)
        if deterministic:
            return mean
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        return action.clamp(-1.0, 1.0)
    
    def get_action_and_log_prob(self, obs: torch.Tensor):
        mean = self.forward(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.clamp(-1.0, 1.0), log_prob


class L2FCriticNetwork(nn.Module):
    """Critic network matching L2F architecture."""
    
    def __init__(self, obs_dim: int = 146, hidden_dim: int = 64):
        super().__init__()
        
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.fc1, self.fc2, self.fc3]:
            nn.init.orthogonal_(m.weight, gain=1.0)
            nn.init.zeros_(m.bias)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.fc1(obs))
        x = torch.tanh(self.fc2(x))
        value = self.fc3(x)
        return value.squeeze(-1)


# ==============================================================================
# PPO Agent
# ==============================================================================

class RunningMeanStd:
    """Running mean and std for observation normalization."""
    
    def __init__(self, shape: tuple, epsilon: float = 1e-8, device: torch.device = None):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon
        self.epsilon = epsilon
        self.device = device
    
    def update(self, x: torch.Tensor):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + self.epsilon)
    
    def to(self, device):
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)
        self.device = device
        return self


class L2FPPOAgent:
    """PPO Agent with L2F-compatible architecture."""
    
    def __init__(
        self,
        obs_dim: int = 146,
        action_dim: int = 4,
        device: torch.device = None,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        epochs: int = 10,
        entropy_coef: float = 0.005,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        self.actor = L2FActorNetwork(obs_dim, 64, action_dim).to(device)
        self.critic = L2FCriticNetwork(obs_dim, 64).to(device)
        
        self.obs_normalizer = RunningMeanStd((obs_dim,), device=device)
        self.normalize_observations = True
        
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr
        )
    
    def normalize_obs(self, obs: torch.Tensor, update: bool = True) -> torch.Tensor:
        if not self.normalize_observations:
            return obs
        if update:
            self.obs_normalizer.update(obs)
        return self.obs_normalizer.normalize(obs)
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        with torch.no_grad():
            obs_norm = self.normalize_obs(obs, update=False)
            return self.actor.get_action(obs_norm, deterministic)
    
    def get_action_and_value(self, obs: torch.Tensor):
        obs_norm = self.normalize_obs(obs, update=True)
        action, log_prob = self.actor.get_action_and_log_prob(obs_norm)
        value = self.critic(obs_norm)
        return action, log_prob, value
    
    def get_value(self, obs: torch.Tensor):
        with torch.no_grad():
            obs_norm = self.normalize_obs(obs, update=False)
            return self.critic(obs_norm)
    
    def update(self, obs: torch.Tensor, actions: torch.Tensor,
               log_probs: torch.Tensor, returns: torch.Tensor, advantages: torch.Tensor):
        obs = obs.detach()
        actions = actions.detach()
        log_probs = log_probs.detach()
        returns = returns.detach()
        advantages = advantages.detach()
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        obs_norm = self.normalize_obs(obs, update=False)
        
        total_loss = 0.0
        for _ in range(self.epochs):
            mean = self.actor(obs_norm)
            std = torch.exp(self.actor.log_std)
            dist = torch.distributions.Normal(mean, std)
            
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()
            
            ratio = (new_log_probs - log_probs).exp()
            clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(ratio * advantages, clip_adv).mean()
            
            values = self.critic(obs_norm)
            value_loss = ((values - returns) ** 2).mean()
            
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / self.epochs
    
    def save(self, path: str, iteration: int, best_reward: float):
        torch.save({
            "iteration": iteration,
            "best_reward": best_reward,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "log_std": self.actor.log_std.data,
            "optimizer": self.optimizer.state_dict(),
            "obs_mean": self.obs_normalizer.mean,
            "obs_var": self.obs_normalizer.var,
            "obs_count": self.obs_normalizer.count,
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor.log_std.data = checkpoint["log_std"]
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "obs_mean" in checkpoint:
            self.obs_normalizer.mean = checkpoint["obs_mean"].to(self.device)
            self.obs_normalizer.var = checkpoint["obs_var"].to(self.device)
            self.obs_normalizer.count = checkpoint["obs_count"]
        return checkpoint.get("iteration", 0), checkpoint.get("best_reward", 0.0)


# ==============================================================================
# Training Loop
# ==============================================================================

def compute_gae(rewards, values, dones, next_value, gamma=0.99, gae_lambda=0.95):
    """Compute Generalized Advantage Estimation."""
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]
        
        delta = rewards[t] + gamma * next_val * (1 - dones[t].float()) - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t].float()) * last_gae
    
    returns = advantages + values
    return returns, advantages


def train(env: CrazyflieNavV2Env, agent: L2FPPOAgent, args):
    """Main training loop."""
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints_nav_v2")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    steps_per_rollout = 128
    num_envs = env.num_envs
    
    best_reward = float("-inf")
    start_iteration = 0
    
    # Load checkpoint if resuming
    if args.resume_from is not None:
        if os.path.exists(args.resume_from):
            print(f"\n{'='*60}")
            print("Loading pre-trained checkpoint (e.g., from hover training)")
            print(f"{'='*60}")
            print(f"  Checkpoint: {args.resume_from}")
            start_iteration, _ = agent.load(args.resume_from)
            print(f"  Loaded from iteration: {start_iteration}")
            print(f"  (Resetting counters for navigation training)")
            start_iteration = 0
            best_reward = float("-inf")
            print(f"{'='*60}\n")
        else:
            print(f"WARNING: Checkpoint not found: {args.resume_from}")
    
    print(f"\n{'='*60}")
    print("Starting L2F Navigation V2 PPO Training (Single-Phase)")
    print(f"{'='*60}")
    print(f"  Environments:       {num_envs}")
    print(f"  Max iterations:     {args.max_iterations}")
    print(f"  Steps per rollout:  {steps_per_rollout}")
    print(f"  Total batch size:   {steps_per_rollout * num_envs}")
    print(f"  Goal XY range:      ±{args.goal_xy_range}m")
    print(f"  Goal Z range:       {args.goal_z_min}-{args.goal_z_max}m")
    print(f"{'='*60}\n")
    
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    
    for iteration in range(args.max_iterations):
        obs_buffer = []
        action_buffer = []
        log_prob_buffer = []
        value_buffer = []
        reward_buffer = []
        done_buffer = []
        
        episode_rewards = torch.zeros(num_envs, device=env.device)
        goals_reached = 0
        
        for step in range(steps_per_rollout):
            action, log_prob, value = agent.get_action_and_value(obs)
            
            obs_buffer.append(obs)
            action_buffer.append(action)
            log_prob_buffer.append(log_prob)
            value_buffer.append(value)
            
            obs_dict, reward, terminated, truncated, info = env.step(action)
            next_obs = obs_dict["policy"]
            done = terminated | truncated
            
            reward_buffer.append(reward)
            done_buffer.append(done)
            episode_rewards += reward
            
            # Count goals reached
            goals_reached += env._goal_reached.sum().item()
            
            obs = next_obs
        
        obs_t = torch.stack(obs_buffer)
        actions_t = torch.stack(action_buffer)
        log_probs_t = torch.stack(log_prob_buffer)
        values_t = torch.stack(value_buffer)
        rewards_t = torch.stack(reward_buffer)
        dones_t = torch.stack(done_buffer)
        
        with torch.no_grad():
            next_value = agent.get_value(obs)
        
        returns_t, advantages_t = compute_gae(
            rewards_t, values_t, dones_t, next_value,
            gamma=agent.gamma, gae_lambda=agent.gae_lambda
        )
        
        obs_flat = obs_t.reshape(-1, obs_t.shape[-1])
        actions_flat = actions_t.reshape(-1, actions_t.shape[-1])
        log_probs_flat = log_probs_t.reshape(-1)
        returns_flat = returns_t.reshape(-1)
        advantages_flat = advantages_t.reshape(-1)
        
        loss = agent.update(obs_flat, actions_flat, log_probs_flat, returns_flat, advantages_flat)
        
        mean_reward = episode_rewards.mean().item() / steps_per_rollout
        mean_return = returns_flat.mean().item()
        reach_rate = goals_reached / (steps_per_rollout * num_envs) * 100
        
        is_best = mean_reward > best_reward
        if is_best:
            best_reward = mean_reward
            agent.save(os.path.join(checkpoint_dir, "best_model.pt"), iteration, best_reward)
        
        if iteration % 10 == 0 or is_best:
            star = " *BEST*" if is_best else ""
            print(f"[Iter {iteration:4d}] Reward: {mean_reward:8.3f} | Return: {mean_return:8.2f} | Reach: {reach_rate:5.1f}% | Loss: {loss:.4f}{star}")
        
        if iteration > 0 and iteration % args.save_interval == 0:
            agent.save(os.path.join(checkpoint_dir, f"checkpoint_{iteration}.pt"), iteration, best_reward)
    
    agent.save(os.path.join(checkpoint_dir, "final_model.pt"), args.max_iterations, best_reward)
    print(f"\nTraining complete! Best reward: {best_reward:.3f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


def play(env: CrazyflieNavV2Env, agent: L2FPPOAgent, checkpoint_path: str):
    """Run trained policy with visualization."""
    iteration, best_reward = agent.load(checkpoint_path)
    print(f"\n[Play Mode] Loaded checkpoint from iteration {iteration}")
    print(f"[Play Mode] Best training reward: {best_reward:.3f}")
    print("[Play Mode] Press Ctrl+C to stop\n")
    
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    
    step_count = 0
    episode_reward = 0.0
    goals_reached = 0
    
    try:
        while simulation_app.is_running():
            action = agent.get_action(obs, deterministic=True)
            
            obs_dict, reward, _, _, _ = env.step(action)
            obs = obs_dict["policy"]
            
            episode_reward += reward.mean().item()
            goals_reached += env._goal_reached.sum().item()
            step_count += 1
            
            if step_count % 100 == 0:
                avg_dist = torch.linalg.norm(
                    env._goal_pos_w - env._robot.data.root_pos_w, dim=1
                ).mean().item()
                print(f"[Step {step_count:5d}] Reward: {episode_reward:.2f} | Avg Dist: {avg_dist:.3f}m | Goals: {goals_reached}")
    
    except KeyboardInterrupt:
        print("\n[Play Mode] Stopped by user")


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    cfg = CrazyflieNavV2EnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.goal_xy_range = args.goal_xy_range
    cfg.goal_z_min = args.goal_z_min
    cfg.goal_z_max = args.goal_z_max
    
    env = CrazyflieNavV2Env(cfg)
    
    agent = L2FPPOAgent(
        obs_dim=cfg.observation_space,
        action_dim=cfg.action_space,
        device=env.device,
        lr=args.lr,
        gamma=args.gamma,
    )
    
    if args.play:
        if args.checkpoint is None:
            checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints_nav_v2")
            args.checkpoint = os.path.join(checkpoint_dir, "best_model.pt")
        
        if not os.path.exists(args.checkpoint):
            print(f"Error: Checkpoint not found: {args.checkpoint}")
            sys.exit(1)
        
        play(env, agent, args.checkpoint)
    else:
        train(env, agent, args)
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
