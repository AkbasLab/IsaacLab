#!/usr/bin/env python3
"""
Crazyflie L2F Navigation Training Script - Two-Phase Approach

This script trains a Crazyflie 2.1 navigation policy that:
1. PHASE 1 (Hover): First stabilizes at target height for 3 seconds
2. PHASE 2 (Navigate): Then navigates horizontally to target XY position

The policy builds on hover skills by requiring stable flight before rewarding
horizontal movement. This curriculum helps the drone learn to fly safely.

KEY DESIGN DECISIONS:
1. Same observation space as hover (146 dims) for L2F compatibility
2. Two-phase reward: hover first, then navigate
3. Target position changes during episode (after hover phase)
4. Builds on hover policy architecture

Usage:
    # Training mode
    python train_nav.py --num_envs 4096 --max_iterations 1000 --headless
    
    # Play mode with trained checkpoint
    python train_nav.py --play --checkpoint checkpoints_nav/best_model.pt
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
    parser = argparse.ArgumentParser(description="Crazyflie L2F Navigation Training")
    
    # Mode selection
    parser.add_argument("--play", action="store_true", help="Run in play mode with trained model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for play mode")
    parser.add_argument("--resume_from", type=str, default=None, 
                        help="Resume training from checkpoint (e.g., hover checkpoint for curriculum learning)")
    
    # Training parameters  
    parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments")
    parser.add_argument("--max_iterations", type=int, default=1000, help="Maximum training iterations")
    parser.add_argument("--save_interval", type=int, default=50, help="Save checkpoint every N iterations")
    
    # Navigation parameters
    parser.add_argument("--hover_time", type=float, default=3.0, help="Seconds to hover before navigation")
    parser.add_argument("--nav_distance", type=float, default=0.5, help="Max XY navigation distance (m)")
    
    # Hyperparameters (tuned for quadrotor)
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
    
    # Mass and geometry
    MASS = 0.027  # kg (27g)
    ARM_LENGTH = 0.028  # m (28mm)
    GRAVITY = 9.81  # m/s²
    
    # Inertia (diagonal)
    IXX = 3.85e-6    # kg·m²
    IYY = 3.85e-6    # kg·m²
    IZZ = 5.9675e-6  # kg·m²
    
    # Motor model
    THRUST_COEFFICIENT = 3.16e-10  # N/RPM²
    TORQUE_COEFFICIENT = 0.005964552  # Nm/N
    RPM_MIN = 0.0
    RPM_MAX = 21702.0
    MOTOR_TIME_CONSTANT = 0.15  # seconds
    
    # Rotor positions (X-config)
    ROTOR_POSITIONS = [
        (0.028, -0.028, 0.0),   # M1
        (-0.028, -0.028, 0.0),  # M2
        (-0.028, 0.028, 0.0),   # M3
        (0.028, 0.028, 0.0),    # M4
    ]
    
    # Rotor yaw directions: -1=CW, +1=CCW
    ROTOR_YAW_DIRS = [-1.0, 1.0, -1.0, 1.0]
    
    # Computed hover RPM
    @classmethod
    def hover_rpm(cls) -> float:
        thrust_per_motor = cls.MASS * cls.GRAVITY / 4.0
        return math.sqrt(thrust_per_motor / cls.THRUST_COEFFICIENT)
    
    # Hover action in normalized space [-1, 1]
    @classmethod
    def hover_action(cls) -> float:
        return 2.0 * cls.hover_rpm() / cls.RPM_MAX - 1.0


# ==============================================================================
# Environment Configuration
# ==============================================================================

@configclass
class CrazyflieNavEnvCfg(DirectRLEnvCfg):
    """Configuration for Crazyflie navigation environment with two-phase learning."""
    
    # Episode settings - longer for navigation
    episode_length_s = 10.0  # 3s hover + 7s navigation
    decimation = 1  # Control at physics rate (100 Hz)
    
    # Spaces - CRITICAL: Must match L2F exactly (same as hover)
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
        num_envs=4096, env_spacing=2.5, replicate_physics=True
    )
    
    # Robot - use custom Crazyflie 2.1 with L2F parameters
    robot: ArticulationCfg = CRAZYFLIE_21_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )
    
    # === PHASE 1: HOVER PARAMETERS ===
    hover_time = 3.0  # Seconds to hover before navigation phase
    init_target_height = 1.0  # m - target hover height
    
    # Hover reward weights (tuned for smooth sim-to-real transfer)
    hover_position_weight = 50.0
    hover_height_weight = 20.0
    hover_orientation_weight = 30.0
    hover_velocity_weight = 30.0      # INCREASED from 10 - penalize fast movements
    hover_angular_velocity_weight = 15.0  # INCREASED from 2 - penalize wobbling
    hover_action_weight = 0.01
    hover_action_rate_weight = 20.0   # NEW - penalize rapid action changes
    
    # Hover stability thresholds (must meet these to advance to phase 2)
    hover_xy_threshold = 0.15  # m - must be within 15cm of target XY
    hover_z_threshold = 0.15   # m - must be within 15cm of target height
    hover_velocity_threshold = 0.3  # m/s - must be relatively stationary
    hover_stable_steps_required = 20  # Consecutive stable steps needed (was 50)
    
    # === PHASE 2: NAVIGATION PARAMETERS ===
    nav_distance_max = 0.5  # m - max XY distance for navigation targets
    nav_height_variation = 0.0  # m - keep same height during navigation (for now)
    
    # Navigation reward weights (tuned for smooth sim-to-real transfer)
    nav_progress_weight = 10.0  # Reward for moving toward target
    nav_position_weight = 30.0  # Reward for reaching target
    nav_orientation_weight = 20.0
    nav_velocity_weight = 15.0          # INCREASED from 5 - don't rush
    nav_angular_velocity_weight = 10.0  # INCREASED from 2 - stay level
    nav_action_weight = 0.01
    nav_action_rate_weight = 15.0       # NEW - smooth action changes
    nav_reaching_bonus = 5.0  # Bonus for reaching target
    nav_reaching_threshold = 0.1  # m - distance to consider "reached"
    
    # === SHARED PARAMETERS ===
    reward_scale = 1.0
    reward_constant = 2.0
    reward_action_baseline = 0.334
    
    # Initialization (spawn at hover position)
    init_height_offset_min = 0.0
    init_height_offset_max = 0.0
    init_max_xy_offset = 0.0
    init_max_angle = 0.0
    init_max_linear_velocity = 0.0
    init_max_angular_velocity = 0.0
    init_guidance_probability = 1.0
    
    # Termination thresholds
    term_xy_threshold = 1.0  # m - wider for navigation
    term_z_min = 0.3
    term_z_max = 2.0
    term_tilt_threshold = 0.7  # rad (~40 deg)
    term_linear_velocity_threshold = 3.0
    term_angular_velocity_threshold = 8.0
    
    # Domain randomization
    enable_disturbance = True
    disturbance_force_std = 0.0132
    disturbance_torque_std = 2.65e-5
    
    # Action history
    action_history_length = 32


# ==============================================================================
# Environment Implementation
# ==============================================================================

class CrazyflieNavEnv(DirectRLEnv):
    """Crazyflie navigation environment with two-phase learning."""
    
    cfg: CrazyflieNavEnvCfg
    
    def __init__(self, cfg: CrazyflieNavEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Cache physics parameters
        self._mass = L2FConstants.MASS
        self._arm_length = L2FConstants.ARM_LENGTH
        self._thrust_coef = L2FConstants.THRUST_COEFFICIENT
        self._torque_coef = L2FConstants.TORQUE_COEFFICIENT
        self._motor_tau = L2FConstants.MOTOR_TIME_CONSTANT
        self._min_rpm = L2FConstants.RPM_MIN
        self._max_rpm = L2FConstants.RPM_MAX
        self._gravity = L2FConstants.GRAVITY
        self._hover_rpm = L2FConstants.hover_rpm()
        self._hover_action = L2FConstants.hover_action()
        self._dt = cfg.sim.dt
        
        # Motor dynamics alpha
        self._motor_alpha = min(self._dt / self._motor_tau, 1.0)
        
        # Rotor geometry tensors
        self._rotor_positions = torch.tensor(
            L2FConstants.ROTOR_POSITIONS, device=self.device, dtype=torch.float32
        )
        self._rotor_yaw_dirs = torch.tensor(
            L2FConstants.ROTOR_YAW_DIRS, device=self.device, dtype=torch.float32
        )
        
        # Hover phase tracking
        self._hover_steps = int(cfg.hover_time / cfg.sim.dt)  # Steps in hover phase
        self._phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # 0=hover, 1=nav
        self._hover_stable_count = torch.zeros(self.num_envs, device=self.device)  # Consecutive stable steps
        
        # Navigation target (XY offset from spawn, set after hover phase)
        self._nav_target = torch.zeros(self.num_envs, 3, device=self.device)
        
        # State tensors
        self._actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, 4, device=self.device)  # For action rate penalty
        self._rpm_state = torch.zeros(self.num_envs, 4, device=self.device)
        
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
            "hover_reward": torch.zeros(self.num_envs, device=self.device),
            "nav_reward": torch.zeros(self.num_envs, device=self.device),
            "total_reward": torch.zeros(self.num_envs, device=self.device),
        }
        
        # Get body ID for force application
        self._body_id = self._robot.find_bodies("body")[0]
        
        # Debug visualization
        self.set_debug_vis(self.cfg.debug_vis)
        
        # Print info
        self._print_env_info()
    
    def _print_env_info(self):
        """Print environment configuration."""
        print("\n" + "="*60)
        print("Crazyflie L2F Navigation Environment (Two-Phase)")
        print("="*60)
        print(f"  Physics dt:        {self._dt*1000:.1f} ms ({1/self._dt:.0f} Hz)")
        print(f"  Episode length:    {self.cfg.episode_length_s:.1f} s")
        print(f"  Num envs:          {self.num_envs}")
        print(f"  Observation dim:   {self.cfg.observation_space}")
        print(f"  Action dim:        {self.cfg.action_space}")
        print(f"  Mass:              {self._mass*1000:.1f} g")
        print(f"  Hover RPM:         {self._hover_rpm:.0f}")
        print(f"  Hover action:      {self._hover_action:.4f}")
        print("-"*60)
        print("  PHASE 1 (Hover):")
        print(f"    Duration:        {self.cfg.hover_time:.1f} s ({self._hover_steps} steps)")
        print(f"    Target height:   {self.cfg.init_target_height:.2f} m")
        print("  PHASE 2 (Navigate):")
        print(f"    Max distance:    {self.cfg.nav_distance_max:.2f} m")
        print(f"    Reach threshold: {self.cfg.nav_reaching_threshold:.2f} m")
        print("="*60 + "\n")
    
    def _setup_scene(self):
        """Set up the simulation scene."""
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        self.scene.clone_environments(copy_from_source=False)
        
        # Add lighting
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
        # Store previous action for action rate penalty
        self._prev_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)
        
        # Map to target RPM
        target_rpm = (self._actions + 1.0) / 2.0 * self._max_rpm
        
        # Apply first-order motor dynamics
        self._rpm_state = self._rpm_state + self._motor_alpha * (target_rpm - self._rpm_state)
        self._rpm_state = self._rpm_state.clamp(self._min_rpm, self._max_rpm)
        
        # Compute thrust per motor
        thrust_per_motor = self._thrust_coef * self._rpm_state ** 2
        
        # Total thrust (body z-axis)
        total_thrust = thrust_per_motor.sum(dim=-1)
        
        thrust_body = torch.zeros(self.num_envs, 3, device=self.device)
        thrust_body[:, 2] = total_thrust
        
        # Roll torque
        roll_torque = (
            thrust_per_motor[:, 0] * self._rotor_positions[0, 1] +
            thrust_per_motor[:, 1] * self._rotor_positions[1, 1] +
            thrust_per_motor[:, 2] * self._rotor_positions[2, 1] +
            thrust_per_motor[:, 3] * self._rotor_positions[3, 1]
        )
        
        # Pitch torque
        pitch_torque = -(
            thrust_per_motor[:, 0] * self._rotor_positions[0, 0] +
            thrust_per_motor[:, 1] * self._rotor_positions[1, 0] +
            thrust_per_motor[:, 2] * self._rotor_positions[2, 0] +
            thrust_per_motor[:, 3] * self._rotor_positions[3, 0]
        )
        
        # Yaw torque
        yaw_torque = self._torque_coef * (
            self._rotor_yaw_dirs[0] * thrust_per_motor[:, 0] +
            self._rotor_yaw_dirs[1] * thrust_per_motor[:, 1] +
            self._rotor_yaw_dirs[2] * thrust_per_motor[:, 2] +
            self._rotor_yaw_dirs[3] * thrust_per_motor[:, 3]
        )
        
        torque_body = torch.stack([roll_torque, pitch_torque, yaw_torque], dim=-1)
        
        # Add disturbances
        if self.cfg.enable_disturbance:
            thrust_body = thrust_body + self._disturbance_force
            torque_body = torque_body + self._disturbance_torque
        
        self._thrust_body[:, 0, :] = thrust_body
        self._torque_body[:, 0, :] = torque_body
        
        # Update action history
        self._action_history[:, :-1] = self._action_history[:, 1:].clone()
        self._action_history[:, -1] = self._actions
    
    def _apply_action(self):
        """Apply forces and torques to the robot."""
        self._robot.set_external_force_and_torque(
            forces=self._thrust_body,
            torques=self._torque_body,
            body_ids=self._body_id,
        )
        self._robot.write_data_to_sim()
    
    def _get_current_target(self) -> torch.Tensor:
        """Get current target position based on phase."""
        # Hover target: env origin + target height
        hover_target = self._terrain.env_origins.clone()
        hover_target[:, 2] += self.cfg.init_target_height
        
        # Navigation phase: use nav target (set after hover stabilizes)
        # Return hover_target for phase 0, nav_target for phase 1
        target = torch.where(
            self._phase.unsqueeze(-1) == 0,
            hover_target,
            self._nav_target
        )
        return target
    
    def _get_observations(self) -> dict:
        """Construct observations matching L2F firmware format (146 dims).
        
        The observation is relative to the CURRENT target (hover or nav).
        """
        pos_w = self._robot.data.root_pos_w
        quat_w = self._robot.data.root_quat_w
        lin_vel_w = self._robot.data.root_lin_vel_w
        ang_vel_b = self._robot.data.root_ang_vel_b
        
        # Position relative to current target
        target_pos = self._get_current_target()
        pos_error = pos_w - target_pos
        
        # Clip position error
        pos_error_clipped = pos_error.clamp(-0.5, 0.5)
        
        # Clip velocity
        lin_vel_clipped = lin_vel_w.clamp(-2.0, 2.0)
        
        # Rotation matrix
        rot_matrix = self._quat_to_rotation_matrix(quat_w)
        
        # Action history (flatten)
        action_history_flat = self._action_history.view(self.num_envs, -1)
        
        # Concatenate (146 dims total)
        obs = torch.cat([
            pos_error_clipped,   # 3
            rot_matrix,          # 9
            lin_vel_clipped,     # 3
            ang_vel_b,           # 3
            action_history_flat, # 128
        ], dim=-1)
        
        return {"policy": obs}
    
    def _check_hover_stable(self) -> torch.Tensor:
        """Check if drone is stable enough to transition to navigation phase."""
        cfg = self.cfg
        
        pos_w = self._robot.data.root_pos_w
        lin_vel = self._robot.data.root_lin_vel_w
        
        # Hover target
        hover_target = self._terrain.env_origins.clone()
        hover_target[:, 2] += cfg.init_target_height
        
        pos_error = pos_w - hover_target
        
        # Check XY position
        xy_ok = torch.norm(pos_error[:, :2], dim=-1) < cfg.hover_xy_threshold
        
        # Check height
        z_ok = torch.abs(pos_error[:, 2]) < cfg.hover_z_threshold
        
        # Check velocity
        vel_ok = torch.norm(lin_vel, dim=-1) < cfg.hover_velocity_threshold
        
        return xy_ok & z_ok & vel_ok
    
    def _transition_to_navigation(self, env_ids: torch.Tensor):
        """Transition specified environments from hover to navigation phase."""
        if len(env_ids) == 0:
            return
        
        cfg = self.cfg
        n = len(env_ids)
        
        # Set phase to navigation
        self._phase[env_ids] = 1
        
        # Generate random navigation target (XY offset from current position)
        # Keep same height during navigation
        current_pos = self._robot.data.root_pos_w[env_ids].clone()
        
        # Random XY offset
        angle = torch.rand(n, device=self.device) * 2 * math.pi
        distance = torch.rand(n, device=self.device) * cfg.nav_distance_max
        
        xy_offset = torch.zeros(n, 2, device=self.device)
        xy_offset[:, 0] = distance * torch.cos(angle)
        xy_offset[:, 1] = distance * torch.sin(angle)
        
        # Set navigation target
        self._nav_target[env_ids, 0] = current_pos[:, 0] + xy_offset[:, 0]
        self._nav_target[env_ids, 1] = current_pos[:, 1] + xy_offset[:, 1]
        self._nav_target[env_ids, 2] = self._terrain.env_origins[env_ids, 2] + cfg.init_target_height
    
    def _get_rewards(self) -> torch.Tensor:
        """Compute reward based on current phase."""
        cfg = self.cfg
        
        pos_w = self._robot.data.root_pos_w
        quat = self._robot.data.root_quat_w
        lin_vel = self._robot.data.root_lin_vel_w
        ang_vel = self._robot.data.root_ang_vel_b
        
        # Get current target
        target_pos = self._get_current_target()
        pos_error = pos_w - target_pos
        
        # === PHASE 1: HOVER REWARD ===
        # XY position cost
        xy_cost = (pos_error[:, :2] ** 2).sum(dim=-1)
        
        # Height cost
        height_cost = pos_error[:, 2] ** 2
        
        # Orientation cost
        orientation_cost = 1.0 - quat[:, 0] ** 2
        
        # Velocity cost
        velocity_cost = (lin_vel ** 2).sum(dim=-1)
        
        # Angular velocity cost
        angular_velocity_cost = (ang_vel ** 2).sum(dim=-1)
        
        # Action cost
        action_deviation = self._actions - self._hover_action
        action_cost = (action_deviation ** 2).sum(dim=-1)
        
        # Action rate cost (penalize rapid changes for smooth control)
        action_rate = self._actions - self._prev_actions
        action_rate_cost = (action_rate ** 2).sum(dim=-1)
        
        # Hover weighted cost
        hover_cost = (
            cfg.hover_position_weight * xy_cost +
            cfg.hover_height_weight * height_cost +
            cfg.hover_orientation_weight * orientation_cost +
            cfg.hover_velocity_weight * velocity_cost +
            cfg.hover_angular_velocity_weight * angular_velocity_cost +
            cfg.hover_action_weight * action_cost +
            cfg.hover_action_rate_weight * action_rate_cost  # NEW
        )
        
        hover_reward = -cfg.reward_scale * hover_cost + cfg.reward_constant
        hover_reward = hover_reward.clamp(0.0, cfg.reward_constant)
        
        # === PHASE 2: NAVIGATION REWARD ===
        # Distance to target
        distance_to_target = torch.norm(pos_error, dim=-1)
        
        # Progress reward (closer = better)
        nav_position_reward = torch.exp(-distance_to_target / 0.2)  # Exponential falloff
        
        # Reaching bonus
        reached = distance_to_target < cfg.nav_reaching_threshold
        reaching_bonus = reached.float() * cfg.nav_reaching_bonus
        
        # Navigation weighted cost (penalize instability during navigation)
        nav_cost = (
            cfg.nav_orientation_weight * orientation_cost +
            cfg.nav_velocity_weight * velocity_cost +           # Added velocity penalty
            cfg.nav_angular_velocity_weight * angular_velocity_cost +
            cfg.nav_action_weight * action_cost +
            cfg.nav_action_rate_weight * action_rate_cost       # NEW - smooth actions
        )
        
        nav_reward = cfg.nav_position_weight * nav_position_reward - nav_cost + reaching_bonus
        nav_reward = nav_reward.clamp(0.0, cfg.nav_position_weight + cfg.nav_reaching_bonus)
        
        # === COMBINE BASED ON PHASE ===
        is_hover_phase = self._phase == 0
        reward = torch.where(is_hover_phase, hover_reward, nav_reward)
        
        # === PHASE TRANSITION CHECK ===
        # Check if drone is stable enough to transition
        is_stable = self._check_hover_stable()
        
        # Increment stable counter for hover phase drones (decay instead of reset)
        self._hover_stable_count = torch.where(
            is_hover_phase & is_stable,
            self._hover_stable_count + 1,
            (self._hover_stable_count * 0.9).clamp(min=0)  # Decay instead of reset
        )
        
        # Transition after hover_steps (3 seconds at 100Hz = 300 steps)
        # Option 1: Stable for enough steps, OR
        # Option 2: Been hovering long enough even if slightly wobbly (time-based fallback)
        stable_enough = self._hover_stable_count >= cfg.hover_stable_steps_required
        time_fallback = self.episode_length_buf >= (self._hover_steps * 1.5)  # 4.5 seconds fallback
        
        should_transition = (
            is_hover_phase & 
            (self.episode_length_buf >= self._hover_steps) &
            (stable_enough | time_fallback)
        )
        
        transition_ids = torch.where(should_transition)[0]
        if len(transition_ids) > 0:
            self._transition_to_navigation(transition_ids)
        
        # Track stats
        self._episode_sums["hover_reward"] += torch.where(is_hover_phase, hover_reward, torch.zeros_like(hover_reward))
        self._episode_sums["nav_reward"] += torch.where(~is_hover_phase, nav_reward, torch.zeros_like(nav_reward))
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
        
        # XY position relative to env origin (not target)
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
        lin_vel_exceeded = torch.norm(lin_vel, dim=-1) > cfg.term_linear_velocity_threshold
        ang_vel_exceeded = torch.norm(ang_vel, dim=-1) > cfg.term_angular_velocity_threshold
        
        terminated = xy_exceeded | too_low | too_high | too_tilted | lin_vel_exceeded | ang_vel_exceeded
        
        return terminated, time_out
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specified environments."""
        if env_ids is None or len(env_ids) == 0:
            return
        
        # Log stats before reset
        if len(env_ids) > 0 and hasattr(self, '_episode_sums'):
            extras = {}
            for key, values in self._episode_sums.items():
                avg = torch.mean(values[env_ids]).item()
                steps = self.episode_length_buf[env_ids].float().mean().item()
                if steps > 0:
                    extras[f"Episode/{key}"] = avg / steps
            self.extras["log"] = extras
        
        # Reset robot
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        
        n = len(env_ids)
        cfg = self.cfg
        
        # Reset phase to hover (phase 0)
        self._phase[env_ids] = 0
        self._hover_stable_count[env_ids] = 0
        
        # Initialize position at hover target
        pos = torch.zeros(n, 3, device=self.device)
        pos[:, 2] = cfg.init_target_height
        pos = pos + self._terrain.env_origins[env_ids]
        
        # Initialize nav target to hover target (will be updated at phase transition)
        self._nav_target[env_ids] = pos.clone()
        
        # Identity quaternion
        quat = torch.zeros(n, 4, device=self.device)
        quat[:, 0] = 1.0
        
        # Zero velocities
        lin_vel = torch.zeros(n, 3, device=self.device)
        ang_vel = torch.zeros(n, 3, device=self.device)
        
        # Write to sim
        root_pose = torch.cat([pos, quat], dim=-1)
        root_vel = torch.cat([lin_vel, ang_vel], dim=-1)
        
        self._robot.write_root_pose_to_sim(root_pose, env_ids)
        self._robot.write_root_velocity_to_sim(root_vel, env_ids)
        
        # Initialize motor state to hover RPM
        self._rpm_state[env_ids] = self._hover_rpm
        
        # Initialize action history to hover action
        self._action_history[env_ids] = self._hover_action
        self._actions[env_ids] = self._hover_action
        self._prev_actions[env_ids] = self._hover_action  # Initialize prev actions
        
        # Sample disturbances
        if cfg.enable_disturbance:
            self._disturbance_force[env_ids] = torch.randn(n, 3, device=self.device) * cfg.disturbance_force_std
            self._disturbance_torque[env_ids] = torch.randn(n, 3, device=self.device) * cfg.disturbance_torque_std
        
        # Reset stats
        for key in self._episode_sums:
            self._episode_sums[key][env_ids] = 0.0
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set up debug visualization."""
        if debug_vis:
            marker_cfg = CUBOID_MARKER_CFG.copy()
            marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
            marker_cfg.prim_path = "/Visuals/Command/goal_position"
            self._goal_markers = VisualizationMarkers(marker_cfg)
        else:
            if hasattr(self, "_goal_markers"):
                self._goal_markers.set_visibility(False)
    
    def _debug_vis_callback(self, event):
        """Update debug visualization."""
        if hasattr(self, "_goal_markers"):
            goal_pos = self._get_current_target()
            self._goal_markers.visualize(goal_pos)


# ==============================================================================
# L2F-Compatible Actor Network (same as hover)
# ==============================================================================

class L2FActorNetwork(nn.Module):
    """Actor network matching L2F architecture exactly."""
    
    HOVER_ACTION = 2.0 * math.sqrt(0.027 * 9.81 / (4 * 3.16e-10)) / 21702.0 - 1.0
    
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
# PPO Agent with Observation Normalization
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


def train(env: CrazyflieNavEnv, agent: L2FPPOAgent, args):
    """Main training loop."""
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints_nav")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    steps_per_rollout = 128
    num_envs = env.num_envs
    
    best_reward = float("-inf")
    start_iteration = 0
    
    # Load checkpoint if resuming (curriculum learning from hover)
    if args.resume_from is not None:
        if os.path.exists(args.resume_from):
            print(f"\n{'='*60}")
            print("CURRICULUM LEARNING: Loading pre-trained checkpoint")
            print(f"{'='*60}")
            print(f"  Checkpoint: {args.resume_from}")
            start_iteration, best_reward = agent.load(args.resume_from)
            print(f"  Loaded from iteration: {start_iteration}")
            print(f"  Previous best reward:  {best_reward:.3f}")
            print(f"  (Resetting iteration counter for nav training)")
            start_iteration = 0  # Start fresh iteration count for nav
            best_reward = float("-inf")  # Reset best reward for nav phase
            print(f"{'='*60}\n")
        else:
            print(f"WARNING: Checkpoint not found: {args.resume_from}")
            print("Starting training from scratch...")
    
    print(f"\n{'='*60}")
    print("Starting L2F Navigation PPO Training (Two-Phase)")
    print(f"{'='*60}")
    print(f"  Environments:       {num_envs}")
    print(f"  Max iterations:     {args.max_iterations}")
    print(f"  Steps per rollout:  {steps_per_rollout}")
    print(f"  Total batch size:   {steps_per_rollout * num_envs}")
    print(f"  Hover time:         {args.hover_time}s")
    print(f"  Nav distance:       {args.nav_distance}m")
    if args.resume_from:
        print(f"  Resumed from:       {os.path.basename(args.resume_from)}")
    print(f"{'='*60}\n")
    
    # Initial reset ONCE - episodes persist across iterations
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    
    for iteration in range(args.max_iterations):
        obs_buffer = []
        action_buffer = []
        log_prob_buffer = []
        value_buffer = []
        reward_buffer = []
        done_buffer = []
        
        # Don't reset here! Let episodes continue across rollouts.
        # env.step() handles individual env resets when they terminate.
        
        episode_rewards = torch.zeros(num_envs, device=env.device)
        
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
        
        # Count phase transitions
        nav_count = (env._phase == 1).sum().item()
        nav_pct = nav_count / num_envs * 100
        
        is_best = mean_reward > best_reward
        if is_best:
            best_reward = mean_reward
            agent.save(os.path.join(checkpoint_dir, "best_model.pt"), iteration, best_reward)
        
        if iteration % 10 == 0 or is_best:
            star = " *BEST*" if is_best else ""
            print(f"[Iter {iteration:4d}] Reward: {mean_reward:8.3f} | Return: {mean_return:8.2f} | Nav: {nav_pct:5.1f}% | Loss: {loss:.4f}{star}")
        
        if iteration > 0 and iteration % args.save_interval == 0:
            agent.save(os.path.join(checkpoint_dir, f"checkpoint_{iteration}.pt"), iteration, best_reward)
    
    agent.save(os.path.join(checkpoint_dir, "final_model.pt"), args.max_iterations, best_reward)
    print(f"\nTraining complete! Best reward: {best_reward:.3f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


def play(env: CrazyflieNavEnv, agent: L2FPPOAgent, checkpoint_path: str):
    """Run trained policy with visualization."""
    iteration, best_reward = agent.load(checkpoint_path)
    print(f"\n[Play Mode] Loaded checkpoint from iteration {iteration}")
    print(f"[Play Mode] Best training reward: {best_reward:.3f}")
    print("[Play Mode] Policy uses trained smooth control (action rate penalty)")
    print("[Play Mode] Press Ctrl+C to stop\n")
    
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    
    step_count = 0
    episode_reward = 0.0
    
    try:
        while simulation_app.is_running():
            # Get action directly from policy - it should be smooth from training
            action = agent.get_action(obs, deterministic=True)
            
            obs_dict, reward, _, _, _ = env.step(action)
            obs = obs_dict["policy"]
            
            episode_reward += reward.mean().item()
            step_count += 1
            
            if step_count % 100 == 0:
                nav_count = (env._phase == 1).sum().item()
                phase_str = f"Nav: {nav_count}/{env.num_envs}"
                print(f"[Step {step_count:5d}] Reward: {episode_reward:.2f} | {phase_str}")
    
    except KeyboardInterrupt:
        print("\n[Play Mode] Stopped by user")


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    cfg = CrazyflieNavEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.hover_time = args.hover_time
    cfg.nav_distance_max = args.nav_distance
    
    env = CrazyflieNavEnv(cfg)
    
    agent = L2FPPOAgent(
        obs_dim=cfg.observation_space,
        action_dim=cfg.action_space,
        device=env.device,
        lr=args.lr,
        gamma=args.gamma,
    )
    
    if args.play:
        if args.checkpoint is None:
            checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints_nav")
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
