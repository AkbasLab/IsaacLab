#!/usr/bin/env python3
"""
Crazyflie L2F Point Navigation + Obstacle Avoidance Training Script

This script extends the point navigation environment with:
1. Random cylindrical obstacles placed between drone and goal
2. Proximity "sector bins" observation (sim-to-real compatible)
3. Collision penalty and proximity-based avoidance rewards

SIM-TO-REAL STRATEGY (AI-deck 1.1 compatibility):
    The AI-deck has a Himax HM01B0 320x320 grayscale camera and a GAP8
    ultra-low-power RISC-V processor. Full image processing is too expensive
    for onboard inference.

    APPROACH: Sector-binned proximity observation
    - In SIMULATION: We analytically compute distances from the drone to
      each obstacle, project them into N angular sectors around the drone's
      forward axis, and output the minimum distance per sector. This gives
      an N-dimensional proximity vector (e.g., 8 sectors = 8 floats).
    - On REAL HARDWARE: A tiny CNN (e.g., MobileNet-v1 with 0.25x width)
      runs on GAP8, processing the 320x320 grayscale image to produce the
      SAME 8-float proximity vector. The CNN is trained separately using
      simulated camera images paired with ground-truth proximity labels.
    - The RL policy only sees the 8-float proximity vector, NOT raw images.
      This makes the policy architecture identical in sim and real.

    This "perception bottleneck" design ensures:
    a) The RL policy is small (MLP, runs on STM32 at 100Hz)
    b) The perception model is small (tiny CNN, runs on GAP8 at ~10-30Hz)
    c) The two are decoupled â€” can improve perception without retraining RL
    d) Sim-to-real gap is isolated to the perception model

OBSERVATION LAYOUT (157 dims):
    [0:3]     Position error from spawn (clipped)
    [3:12]    Rotation matrix (9 elements)
    [12:15]   Linear velocity (clipped)
    [15:18]   Angular velocity (body frame)
    [18:146]  Action history (32 * 4)
    [146:149] Goal relative position (clipped)
    [149:157] Proximity sector bins (8 sectors, normalized [0,1])

    The first 149 dims are IDENTICAL to train_pointnav.py.
    The last 8 dims are the proximity sensor output.

OBSTACLE CONFIGURATION:
    - 1 to 3 cylindrical pillars per environment (randomized)
    - Placed randomly between drone spawn and goal
    - Radius: 3-8 cm (Crazyflie-scale obstacles)
    - Height: ground to 2m (taller than flight altitude)
    - Repositioned each episode reset

Usage (from IsaacLab directory):
    # Sanity test first
    .\\isaaclab.bat -p source\\isaaclab_tasks\\isaaclab_tasks\\direct\\crazyflie_l2f\\train_pointnav_obs_avoidance.py --sanity_test --num_envs 16

    # Training mode
    .\\isaaclab.bat -p source\\isaaclab_tasks\\isaaclab_tasks\\direct\\crazyflie_l2f\\train_pointnav_obs_avoidance.py --num_envs 4096 --max_iterations 1000 --headless

    # Play mode
    .\\isaaclab.bat -p source\\isaaclab_tasks\\isaaclab_tasks\\direct\\crazyflie_l2f\\train_pointnav_obs_avoidance.py --play --checkpoint source\\isaaclab_tasks\\isaaclab_tasks\\direct\\crazyflie_l2f\\checkpoints_pointnav_obs\\best_model.pt --num_envs 64
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
import time
import numpy as np

# Isaac Sim setup - must happen before other imports
from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description="Crazyflie L2F Point Navigation + Obstacle Avoidance")

    # Mode selection
    parser.add_argument("--play", action="store_true", help="Run in play mode with trained model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--sanity_test", action="store_true", help="Run sanity test")

    # Training parameters
    parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments")
    parser.add_argument("--max_iterations", type=int, default=1000, help="Maximum training iterations")
    parser.add_argument("--save_interval", type=int, default=100, help="Save checkpoint every N iterations")

    # Hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")

    # =========================================================================
    # CURRICULUM / DIFFICULTY SETTINGS
    # =========================================================================
    parser.add_argument("--term_on_collision", action="store_true",
                        help="Terminate episodes on collision (curriculum stage 2)")
    parser.add_argument("--max_obstacles", type=int, default=3,
                        help="Maximum obstacles per environment (default: 3, try 5-6)")
    parser.add_argument("--min_obstacles", type=int, default=1,
                        help="Minimum obstacles per episode (default: 1)")
    parser.add_argument("--obstacle_radius_min", type=float, default=0.03,
                        help="Min obstacle radius in meters (default: 0.03)")
    parser.add_argument("--obstacle_radius_max", type=float, default=0.08,
                        help="Max obstacle radius in meters (default: 0.08)")
    parser.add_argument("--goal_min_distance", type=float, default=0.5,
                        help="Min goal distance in meters (default: 0.5)")
    parser.add_argument("--goal_max_distance", type=float, default=1.2,
                        help="Max goal distance in meters (default: 1.2)")
    parser.add_argument("--collision_penalty", type=float, default=-20.0,
                        help="Collision penalty per step (default: -20.0)")
    parser.add_argument("--lateral_spread", type=float, default=0.4,
                        help="Max lateral offset for obstacle placement (default: 0.4)")

    # Curriculum
    parser.add_argument("--curriculum", action="store_true", default=True,
                        help="Enable automatic curriculum (default: True)")
    parser.add_argument("--no_curriculum", action="store_true",
                        help="Disable curriculum, use all settings as-is")
    parser.add_argument("--curriculum_phase", type=int, default=None,
                        help="Force start at specific curriculum phase (1-4)")

    # Performance tuning
    parser.add_argument("--minibatch_size", type=int, default=8192,
                        help="Minibatch size for PPO updates (default: 8192)")
    parser.add_argument("--ppo_epochs", type=int, default=5,
                        help="PPO update epochs (default: 5)")

    # AppLauncher adds its own args (including --headless)
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()
    return args


# Parse args and launch Isaac Sim (only when run directly, not when imported)
if __name__ == "__main__":
    args = parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

# Now import Isaac Lab modules (safe after AppLauncher init, whether from this script or importer)
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import CUBOID_MARKER_CFG

# Import our custom Crazyflie configuration
from crazyflie_21_cfg import CRAZYFLIE_21_CFG, CrazyflieL2FParams

# Import flight evaluation utilities
from flight_eval_utils import FlightDataLogger


# ==============================================================================
# L2F Physics Constants (IDENTICAL to train_hover.py / train_pointnav.py)
# ==============================================================================

class L2FConstants:
    """Physical parameters matching learning-to-fly exactly."""

    MASS = 0.027  # kg (27g)
    ARM_LENGTH = 0.028  # m (28mm)
    GRAVITY = 9.81  # m/sÂ²

    IXX = 3.85e-6
    IYY = 3.85e-6
    IZZ = 5.9675e-6

    THRUST_COEFFICIENT = 3.16e-10
    TORQUE_COEFFICIENT = 0.005964552
    RPM_MIN = 0.0
    RPM_MAX = 21702.0
    MOTOR_TIME_CONSTANT = 0.15

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
# Proximity Sensor Model (Sim-to-Real Compatible)
# ==============================================================================

class ProximitySensorModel:
    """Analytical proximity sensor that produces sector-binned distances.

    Simulates what a lightweight CNN on the AI-deck GAP8 processor would
    output: minimum obstacle distance in each angular sector around the drone.

    The sensor operates in the drone's body-frame XY plane (yaw-aligned).
    It divides the horizontal plane into N sectors (default 8 = 45deg each)
    and reports the minimum distance to any obstacle in each sector.

    On real hardware:
        - HM01B0 camera (320x320 grayscale, forward-facing, ~60deg FOV)
        - Tiny CNN on GAP8 â†’ 8 float distances
        - Rear sectors get max_range (no camera coverage) or use flow deck

    In simulation:
        - We know exact obstacle positions
        - Project obstacle-to-drone vector into body frame
        - Compute angular sector and distance analytically
        - Account for obstacle radius (distance to surface, not center)
    """

    def __init__(
        self,
        num_sectors: int = 8,
        max_range: float = 2.0,
        device: torch.device = None,
    ):
        self.num_sectors = num_sectors
        self.max_range = max_range
        self.device = device

        # Precompute sector boundaries (in radians, body frame)
        # Sector 0: centered at 0deg (forward), sector 1: centered at 45deg, etc.
        self.sector_width = 2.0 * math.pi / num_sectors
        # Sector boundaries: [-Ï€/N, Ï€/N), [Ï€/N, 3Ï€/N), ...
        self.sector_centers = torch.tensor(
            [i * self.sector_width for i in range(num_sectors)],
            device=device, dtype=torch.float32
        )  # (num_sectors,)

    def compute_proximity(
        self,
        drone_pos: torch.Tensor,     # (N, 3)
        drone_quat: torch.Tensor,    # (N, 4) [w,x,y,z]
        obstacle_pos: torch.Tensor,  # (N, max_obstacles, 3)
        obstacle_radii: torch.Tensor,  # (N, max_obstacles)
        obstacle_active: torch.Tensor,  # (N, max_obstacles) bool
    ) -> torch.Tensor:
        """Compute sector-binned proximity distances.

        Args:
            drone_pos: Drone world position (N, 3)
            drone_quat: Drone world quaternion [w,x,y,z] (N, 4)
            obstacle_pos: Obstacle world positions (N, max_obstacles, 3)
            obstacle_radii: Obstacle radii in meters (N, max_obstacles)
            obstacle_active: Which obstacles are active (N, max_obstacles)

        Returns:
            proximity: Normalized distances per sector (N, num_sectors)
                       0.0 = touching obstacle, 1.0 = max range or no obstacle
        """
        N = drone_pos.shape[0]
        max_obs = obstacle_pos.shape[1]

        # Initialize all sectors to max range
        proximity = torch.ones(N, self.num_sectors, device=self.device)

        if max_obs == 0:
            return proximity

        # Compute vectors from drone to each obstacle (world frame, XY only)
        delta = obstacle_pos[:, :, :2] - drone_pos[:, :2].unsqueeze(1)  # (N, max_obs, 2)

        # Compute distances to obstacle surfaces (subtract radius)
        dist_centers = torch.norm(delta, dim=-1)  # (N, max_obs)
        dist_surfaces = (dist_centers - obstacle_radii).clamp(min=0.0)  # (N, max_obs)

        # Extract drone yaw from quaternion for body-frame projection
        # yaw = atan2(2(wz + xy), 1 - 2(yÂ² + zÂ²))
        w, x, y, z = drone_quat[:, 0], drone_quat[:, 1], drone_quat[:, 2], drone_quat[:, 3]
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))  # (N,)

        # Compute bearing angle to each obstacle in world frame
        bearing_world = torch.atan2(delta[:, :, 1], delta[:, :, 0])  # (N, max_obs)

        # Convert to body frame (subtract drone yaw)
        bearing_body = bearing_world - yaw.unsqueeze(1)  # (N, max_obs)

        # Normalize bearing to [0, 2Ï€)
        bearing_body = bearing_body % (2 * math.pi)

        # Determine which sector each obstacle falls into
        sector_idx = (bearing_body / self.sector_width).long() % self.num_sectors  # (N, max_obs)

        # Normalize distances to [0, 1]
        dist_normalized = (dist_surfaces / self.max_range).clamp(0.0, 1.0)  # (N, max_obs)

        # For inactive obstacles, set distance to max (1.0)
        dist_normalized = torch.where(obstacle_active, dist_normalized, torch.ones_like(dist_normalized))

        # For each sector, take minimum distance across all obstacles
        # Use scatter_reduce to find minimum per sector
        for obs_i in range(max_obs):
            sector = sector_idx[:, obs_i]  # (N,)
            dist = dist_normalized[:, obs_i]  # (N,)

            # Update proximity: take element-wise minimum
            # proximity[n, sector[n]] = min(proximity[n, sector[n]], dist[n])
            existing = proximity.gather(1, sector.unsqueeze(1)).squeeze(1)  # (N,)
            new_min = torch.minimum(existing, dist)
            proximity.scatter_(1, sector.unsqueeze(1), new_min.unsqueeze(1))

        return proximity


# ==============================================================================
# Environment Configuration
# ==============================================================================

@configclass
class CrazyfliePointNavObsAvoidEnvCfg(DirectRLEnvCfg):
    """Configuration for point navigation with obstacle avoidance.

    Extends CrazyfliePointNavEnvCfg with:
    1. Obstacle spawning configuration
    2. Proximity sensor configuration
    3. Collision avoidance rewards
    4. Extended observation space (149 + 8 = 157)
    """

    # Episode settings
    episode_length_s = 10.0
    decimation = 1

    # =========================================================================
    # OBSERVATION SPACE: 149 (pointnav) + 8 (proximity sectors) = 157
    # =========================================================================
    num_proximity_sectors: int = 8  # 8 angular sectors (45deg each)
    observation_space = 157  # 149 + 8
    action_space = 4
    state_space = 0
    debug_vis = True

    # Simulation - 100 Hz physics (L2F standard)
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

    robot: ArticulationCfg = CRAZYFLIE_21_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    # =========================================================================
    # OBSTACLE CONFIGURATION
    # =========================================================================
    # We spawn max_obstacles per env, but only activate 1-3 randomly per episode.
    # Obstacles are static kinematic cylinders (pillars).
    max_obstacles: int = 3  # Maximum obstacles per environment
    min_obstacles: int = 1  # Minimum obstacles per episode
    obstacle_radius_min: float = 0.03  # 3cm minimum radius
    obstacle_radius_max: float = 0.08  # 8cm maximum radius
    obstacle_height: float = 2.0  # 2m tall pillars (taller than flight altitude)

    # Obstacle placement: between drone and goal
    obstacle_min_dist_from_drone: float = 0.15  # Don't spawn right on top of drone
    obstacle_min_dist_from_goal: float = 0.15  # Don't block the goal completely
    obstacle_lateral_spread: float = 0.4  # Max lateral offset from drone-goal line

    # Collision detection
    collision_radius_margin: float = 0.03  # 3cm margin around drone for collision detection
    drone_collision_radius: float = 0.05  # Effective drone radius (arm length + margin)

    # Obstacle RigidObject configs - spawn 3 cylinders per env
    # We use a moderate fixed radius for the USD prim; actual collision detection
    # uses the randomized radii stored in obstacle_radii tensor
    obstacle_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Obstacle_1",
        spawn=sim_utils.CylinderCfg(
            radius=0.05,
            height=2.0,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(100.0, 100.0, 1.0)),
    )
    obstacle_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Obstacle_2",
        spawn=sim_utils.CylinderCfg(
            radius=0.05,
            height=2.0,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.4, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(100.0, 100.0, 1.0)),
    )
    obstacle_3: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Obstacle_3",
        spawn=sim_utils.CylinderCfg(
            radius=0.05,
            height=2.0,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.6, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(100.0, 100.0, 1.0)),
    )
    # Extra obstacle slots for curriculum (initially unused if max_obstacles <= 3)
    obstacle_4: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Obstacle_4",
        spawn=sim_utils.CylinderCfg(
            radius=0.05,
            height=2.0,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.2, 0.7)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(100.0, 100.0, 1.0)),
    )
    obstacle_5: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Obstacle_5",
        spawn=sim_utils.CylinderCfg(
            radius=0.05,
            height=2.0,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.7, 0.7)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(100.0, 100.0, 1.0)),
    )
    obstacle_6: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Obstacle_6",
        spawn=sim_utils.CylinderCfg(
            radius=0.05,
            height=2.0,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.9)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(100.0, 100.0, 1.0)),
    )

    # Maximum number of obstacle USD prims spawned in scene
    # (must be >= max_obstacles, kept at 6 to allow curriculum scaling)
    _max_obstacle_prims: int = 6

    # =========================================================================
    # PROXIMITY SENSOR CONFIGURATION (Sim-to-Real compatible)
    # =========================================================================
    proximity_max_range: float = 2.0  # Maximum sensing range (meters)
    # num_proximity_sectors defined above (8)

    # =========================================================================
    # HOVER STABILITY REWARDS (from train_pointnav.py)
    # =========================================================================
    hover_reward_scale = 0.2
    hover_reward_constant = 0.5
    hover_position_weight = 0.0
    hover_height_weight = 6.0
    hover_orientation_weight = 15.0
    hover_xy_velocity_weight = 0.5
    hover_z_velocity_weight = 1.0
    hover_angular_velocity_weight = 1.0
    hover_action_weight = 0.01

    hover_gate_radius = 0.5
    hover_gate_min = 0.2

    # =========================================================================
    # NAVIGATION REWARDS (from train_pointnav.py)
    # =========================================================================
    nav_progress_weight = 8.0       # Boosted from 5.0 to prioritize goal-seeking
    nav_reach_bonus = 200.0          # Boosted from 100.0 to strongly incentivize reaching
    nav_timeout_penalty = 0.0
    nav_braking_weight = 1.0
    nav_braking_radius = 0.3

    nav_height_recovery_weight = 1.0
    nav_height_track_weight = 0.5

    nav_speed_penalty_weight = 0.1
    nav_speed_penalty_threshold = 4.0

    nav_low_height_penalty_floor = 0.7
    nav_low_height_penalty_weight = 3.0

    # =========================================================================
    # OBSTACLE AVOIDANCE REWARDS (NEW)
    # =========================================================================
    # Collision penalty: negative reward for hitting an obstacle
    # NOTE: Kept moderate to avoid overwhelming navigation rewards.
    # Curriculum system scales this up over phases.
    obs_collision_penalty: float = -20.0  # Penalty per step while in collision

    # Proximity warning: small penalty when getting too close (within warning zone)
    obs_proximity_warning_radius: float = 0.15  # Start warning at 15cm from surface
    obs_proximity_warning_weight: float = 2.0  # Quadratic penalty coefficient

    # Clearance reward: small positive reward for maintaining safe distance
    obs_clearance_reward_weight: float = 0.1  # Per-step bonus for being > warning radius

    # =========================================================================
    # GOAL SAMPLING CONFIGURATION (from train_pointnav.py)
    # =========================================================================
    goal_min_distance = 0.5
    goal_max_distance = 1.2
    goal_height = 1.0
    goal_reach_threshold = 0.1

    obs_position_clip = 1.5
    obs_velocity_clip = 2.0

    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    init_target_height = 1.0
    init_height_offset_min = -0.05
    init_height_offset_max = 0.05
    init_max_xy_offset = 0.0
    init_max_angle = 0.1
    init_max_linear_velocity = 0.2
    init_max_angular_velocity = 0.2
    init_guidance_probability = 0.2

    # =========================================================================
    # TERMINATION THRESHOLDS (from train_pointnav.py)
    # =========================================================================
    term_xy_threshold = 3.0

    term_z_soft_min: float = 0.25
    term_z_hard_min: float = 0.10
    term_z_soft_max: float = 2.50
    term_z_hard_max: float = 3.00
    term_z_persistence_steps: int = 50

    term_tilt_soft_threshold = 1.22
    term_tilt_hard_threshold = 2.62
    term_tilt_persistence_steps = 50

    term_linear_velocity_soft_threshold: float = 4.0
    term_linear_velocity_hard_threshold: float = 6.0
    term_linear_velocity_persistence_steps: int = 50

    term_angular_velocity_soft_threshold = 30.0
    term_angular_velocity_hard_threshold = 50.0
    term_angular_velocity_persistence_steps = 10

    # NEW: Collision termination (optional â€” start without it for learning)
    term_on_collision: bool = False  # Don't terminate on collision initially
    # Can enable later in curriculum: collision â†’ termination teaches hard avoidance

    # Domain randomization
    enable_disturbance = True
    disturbance_force_std = 0.0132
    disturbance_torque_std = 2.65e-5

    # Action history
    action_history_length = 32

    # Hover-centered actions
    use_hover_centered_actions: bool = True
    action_scale: float = 0.3


# ==============================================================================
# Environment Implementation
# ==============================================================================

class CrazyfliePointNavObsAvoidEnv(DirectRLEnv):
    """Crazyflie environment for point navigation with obstacle avoidance."""

    cfg: CrazyfliePointNavObsAvoidEnvCfg

    def __init__(self, cfg: CrazyfliePointNavObsAvoidEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Cache physics parameters (IDENTICAL to train_pointnav.py)
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

        self._motor_alpha = min(self._dt / self._motor_tau, 1.0)

        self._rotor_positions = torch.tensor(
            L2FConstants.ROTOR_POSITIONS, device=self.device, dtype=torch.float32
        )
        self._rotor_yaw_dirs = torch.tensor(
            L2FConstants.ROTOR_YAW_DIRS, device=self.device, dtype=torch.float32
        )

        # State tensors
        self._actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._rpm_state = torch.zeros(self.num_envs, 4, device=self.device)

        self._thrust_body = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._torque_body = torch.zeros(self.num_envs, 1, 3, device=self.device)

        self._action_history = torch.zeros(
            self.num_envs, cfg.action_history_length, 4, device=self.device
        )

        self._disturbance_force = torch.zeros(self.num_envs, 3, device=self.device)
        self._disturbance_torque = torch.zeros(self.num_envs, 3, device=self.device)

        # =====================================================================
        # NAVIGATION STATE (from train_pointnav.py)
        # =====================================================================
        self._goal_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_dist_xy = torch.zeros(self.num_envs, device=self.device)
        self._prev_speed = torch.zeros(self.num_envs, device=self.device)
        self._prev_height_below_target = torch.zeros(self.num_envs, device=self.device)
        self._goal_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Persistent copy of goal_reached that survives auto-reset (for eval scripts)
        self._last_done_goal_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_done_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # =====================================================================
        # OBSTACLE AVOIDANCE STATE (NEW)
        # =====================================================================
        max_obs = cfg._max_obstacle_prims  # Size for max possible (curriculum can scale)

        # Obstacle positions (world frame) â€” (N, max_obstacles, 3)
        self._obstacle_pos = torch.zeros(self.num_envs, max_obs, 3, device=self.device)

        # Obstacle radii â€” (N, max_obstacles) â€” randomized per episode
        self._obstacle_radii = torch.ones(self.num_envs, max_obs, device=self.device) * 0.05

        # Which obstacles are active this episode â€” (N, max_obstacles)
        self._obstacle_active = torch.zeros(self.num_envs, max_obs, dtype=torch.bool, device=self.device)

        # Collision flag (per env, per step)
        self._in_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Minimum distance to any obstacle surface (for reward computation)
        self._min_obstacle_dist = torch.ones(self.num_envs, device=self.device) * cfg.proximity_max_range

        # Proximity sensor model
        self._proximity_sensor = ProximitySensorModel(
            num_sectors=cfg.num_proximity_sectors,
            max_range=cfg.proximity_max_range,
            device=self.device,
        )

        # Current proximity readings (cached for obs and reward)
        self._proximity_bins = torch.ones(self.num_envs, cfg.num_proximity_sectors, device=self.device)

        # =====================================================================
        # EPISODE STATISTICS
        # =====================================================================
        self._episode_sums = {
            "height_cost": torch.zeros(self.num_envs, device=self.device),
            "orientation_cost": torch.zeros(self.num_envs, device=self.device),
            "xy_velocity_cost": torch.zeros(self.num_envs, device=self.device),
            "z_velocity_cost": torch.zeros(self.num_envs, device=self.device),
            "angular_velocity_cost": torch.zeros(self.num_envs, device=self.device),
            "action_cost": torch.zeros(self.num_envs, device=self.device),
            "hover_reward": torch.zeros(self.num_envs, device=self.device),
            "progress_reward": torch.zeros(self.num_envs, device=self.device),
            "braking_reward": torch.zeros(self.num_envs, device=self.device),
            "speed_penalty": torch.zeros(self.num_envs, device=self.device),
            "reach_bonus": torch.zeros(self.num_envs, device=self.device),
            "collision_penalty": torch.zeros(self.num_envs, device=self.device),
            "proximity_penalty": torch.zeros(self.num_envs, device=self.device),
            "clearance_reward": torch.zeros(self.num_envs, device=self.device),
            "total_reward": torch.zeros(self.num_envs, device=self.device),
            "goal_reached": torch.zeros(self.num_envs, device=self.device),
            "final_distance": torch.zeros(self.num_envs, device=self.device),
            "collision_count": torch.zeros(self.num_envs, device=self.device),
        }

        # Termination counters
        self._term_counters = {
            "xy_exceeded": 0,
            "too_low": 0,
            "too_high": 0,
            "too_tilted": 0,
            "lin_vel_exceeded": 0,
            "ang_vel_exceeded": 0,
            "collision": 0,
            "goal_reached": 0,
            "timeout": 0,
            "total": 0,
        }

        self._episode_lengths = []
        self._max_episode_buffer = 10000

        # Persistence counters
        self._tilt_violation_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._angvel_violation_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._linvel_violation_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._height_low_violation_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._height_high_violation_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # Body ID for force application
        self._body_id = self._robot.find_bodies("body")[0]

        # Debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

        self._print_env_info()
        self._verify_invariants()

    def _verify_invariants(self):
        """Verify critical invariants are satisfied."""
        cfg = self.cfg

        assert cfg.obs_position_clip >= cfg.goal_max_distance
        assert cfg.goal_min_distance > cfg.goal_reach_threshold
        assert cfg.term_z_hard_min < cfg.init_target_height < cfg.term_z_hard_max
        assert cfg.term_z_hard_min < cfg.term_z_soft_min < cfg.init_target_height
        assert cfg.init_target_height < cfg.term_z_soft_max < cfg.term_z_hard_max
        assert cfg.term_tilt_soft_threshold < cfg.term_tilt_hard_threshold
        assert cfg.term_tilt_persistence_steps > 0
        assert cfg.term_angular_velocity_soft_threshold < cfg.term_angular_velocity_hard_threshold
        assert cfg.term_angular_velocity_persistence_steps > 0
        assert cfg.term_linear_velocity_soft_threshold < cfg.term_linear_velocity_hard_threshold
        assert cfg.term_linear_velocity_persistence_steps > 0
        assert cfg.term_z_persistence_steps > 0
        assert cfg.num_proximity_sectors > 0
        assert cfg.max_obstacles >= cfg.min_obstacles >= 0, \
            f"max_obstacles ({cfg.max_obstacles}) >= min_obstacles ({cfg.min_obstacles}) >= 0"
        assert cfg.observation_space == 149 + cfg.num_proximity_sectors, \
            f"observation_space ({cfg.observation_space}) must be 149 + {cfg.num_proximity_sectors}"

        print("[INVARIANTS] All invariants verified âœ“")

    def _print_env_info(self):
        """Print environment configuration."""
        cfg = self.cfg
        print("\n" + "="*60)
        print("Crazyflie L2F Point Nav + Obstacle Avoidance Environment")
        print("="*60)
        print(f"  Physics dt:        {self._dt*1000:.1f} ms ({1/self._dt:.0f} Hz)")
        print(f"  Episode length:    {cfg.episode_length_s:.1f} s")
        print(f"  Num envs:          {self.num_envs}")
        print(f"  Observation dim:   {cfg.observation_space} (149 nav + {cfg.num_proximity_sectors} proximity)")
        print(f"  Action dim:        {cfg.action_space}")
        print(f"  Mass:              {self._mass*1000:.1f} g")
        print(f"  Hover RPM:         {self._hover_rpm:.0f}")
        print("--- Obstacles ---")
        print(f"  Max per env:       {cfg.max_obstacles}")
        print(f"  Min per episode:   {cfg.min_obstacles}")
        print(f"  Radius range:      [{cfg.obstacle_radius_min*100:.0f}, {cfg.obstacle_radius_max*100:.0f}] cm")
        print(f"  Height:            {cfg.obstacle_height:.1f} m")
        print("--- Proximity Sensor ---")
        print(f"  Sectors:           {cfg.num_proximity_sectors}")
        print(f"  Max range:         {cfg.proximity_max_range:.1f} m")
        print(f"  Sector width:      {360/cfg.num_proximity_sectors:.0f}deg")
        print("--- Avoidance Rewards ---")
        print(f"  Collision penalty: {cfg.obs_collision_penalty}")
        print(f"  Warning radius:    {cfg.obs_proximity_warning_radius*100:.0f} cm")
        print(f"  Warning weight:    {cfg.obs_proximity_warning_weight}")
        print(f"  Terminate on col:  {cfg.term_on_collision}")
        print("="*60 + "\n")

    def _setup_scene(self):
        """Set up the simulation scene with robot, obstacles, and terrain."""
        # Robot
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # Obstacles (up to _max_obstacle_prims per env, only max_obstacles used)
        self._obstacles = []
        obstacle_cfgs = [
            self.cfg.obstacle_1, self.cfg.obstacle_2, self.cfg.obstacle_3,
            self.cfg.obstacle_4, self.cfg.obstacle_5, self.cfg.obstacle_6,
        ]
        num_prims_to_spawn = max(self.cfg.max_obstacles, self.cfg._max_obstacle_prims)
        for i in range(num_prims_to_spawn):
            obs_obj = RigidObject(obstacle_cfgs[i])
            self.scene.rigid_objects[f"obstacle_{i}"] = obs_obj
            self._obstacles.append(obs_obj)

        # Terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        # Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _quat_to_rotation_matrix(self, quat: torch.Tensor) -> torch.Tensor:
        """Convert quaternion to flattened rotation matrix (9 elements)."""
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

    def _get_tilt_angle(self, quat: torch.Tensor) -> torch.Tensor:
        """Compute tilt angle from rotation matrix."""
        rot_matrix = self._quat_to_rotation_matrix(quat)
        r22 = rot_matrix[:, 8]
        cos_tilt = torch.clamp(r22, -1.0, 1.0)
        return torch.acos(cos_tilt)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions using L2F motor model (identical to train_pointnav.py)."""
        self._actions = actions.clone().clamp(-1.0, 1.0)

        if self.cfg.use_hover_centered_actions:
            hover_rpm = self._hover_rpm
            pos_range = (self._max_rpm - hover_rpm) * self.cfg.action_scale
            neg_range = (hover_rpm - self._min_rpm) * self.cfg.action_scale
            target_rpm = torch.where(
                self._actions >= 0,
                hover_rpm + self._actions * pos_range,
                hover_rpm + self._actions * neg_range
            )
        else:
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

        self._action_history[:, :-1] = self._action_history[:, 1:].clone()
        self._action_history[:, -1] = self._actions

        # =====================================================================
        # UPDATE OBSTACLE PROXIMITY (every physics step)
        # =====================================================================
        self._update_obstacle_proximity()

    def _apply_action(self):
        """Apply forces and torques to the robot."""
        self._robot.set_external_force_and_torque(
            forces=self._thrust_body,
            torques=self._torque_body,
            body_ids=self._body_id,
        )
        self._robot.write_data_to_sim()

    def _update_obstacle_proximity(self):
        """Update proximity sensor readings and collision detection.

        This is the core of the obstacle avoidance system:
        1. Compute sector-binned proximity (for observations)
        2. Compute minimum distance to any obstacle (for rewards)
        3. Detect collisions (for penalties/termination)
        """
        drone_pos = self._robot.data.root_pos_w  # (N, 3)
        drone_quat = self._robot.data.root_quat_w  # (N, 4)

        # Update proximity bins using the analytical sensor model
        self._proximity_bins = self._proximity_sensor.compute_proximity(
            drone_pos=drone_pos,
            drone_quat=drone_quat,
            obstacle_pos=self._obstacle_pos,
            obstacle_radii=self._obstacle_radii,
            obstacle_active=self._obstacle_active,
        )

        # Compute minimum distance to any active obstacle surface (for rewards)
        # delta_xy: (N, max_obs, 2)
        delta_xy = self._obstacle_pos[:, :, :2] - drone_pos[:, :2].unsqueeze(1)
        dist_centers = torch.norm(delta_xy, dim=-1)  # (N, max_obs)
        dist_surfaces = (dist_centers - self._obstacle_radii).clamp(min=0.0)

        # Mask inactive obstacles to max range
        dist_surfaces = torch.where(
            self._obstacle_active,
            dist_surfaces,
            torch.ones_like(dist_surfaces) * self.cfg.proximity_max_range
        )

        self._min_obstacle_dist = dist_surfaces.min(dim=-1).values  # (N,)

        # Collision detection: drone center within (obstacle_radius + drone_radius) of obstacle center
        collision_threshold = self._obstacle_radii + self.cfg.drone_collision_radius
        in_collision_per_obs = (dist_centers < collision_threshold) & self._obstacle_active  # (N, max_obs)
        self._in_collision = in_collision_per_obs.any(dim=-1)  # (N,)

    def _get_observations(self) -> dict:
        """Construct observations: 149 (pointnav) + 8 (proximity) = 157 dims.

        Layout:
        - [0:3]     Position error from spawn (clipped)
        - [3:12]    Rotation matrix (9 elements)
        - [12:15]   Linear velocity (clipped)
        - [15:18]   Angular velocity (body frame)
        - [18:146]  Action history (32 * 4 = 128)
        - [146:149] Goal relative position (clipped)
        - [149:157] Proximity sector bins (8 sectors, normalized [0,1])
        """
        cfg = self.cfg

        pos_w = self._robot.data.root_pos_w
        quat_w = self._robot.data.root_quat_w
        lin_vel_w = self._robot.data.root_lin_vel_w
        ang_vel_b = self._robot.data.root_ang_vel_b

        spawn_pos = self._terrain.env_origins.clone()
        spawn_pos[:, 2] += cfg.init_target_height
        pos_error = pos_w - spawn_pos
        pos_error_clipped = pos_error.clamp(-cfg.obs_position_clip, cfg.obs_position_clip)

        lin_vel_clipped = lin_vel_w.clamp(-cfg.obs_velocity_clip, cfg.obs_velocity_clip)
        rot_matrix = self._quat_to_rotation_matrix(quat_w)
        action_history_flat = self._action_history.view(self.num_envs, -1)

        goal_relative = self._goal_pos - pos_w
        goal_relative_clipped = goal_relative.clamp(-cfg.obs_position_clip, cfg.obs_position_clip)

        # Proximity bins (already computed in _pre_physics_step)
        proximity = self._proximity_bins  # (N, num_sectors)

        obs = torch.cat([
            pos_error_clipped,       # 3
            rot_matrix,              # 9
            lin_vel_clipped,         # 3
            ang_vel_b,               # 3
            action_history_flat,     # 128
            goal_relative_clipped,   # 3
            proximity,               # 8 (proximity sectors)
        ], dim=-1)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute reward: hover stability + navigation + obstacle avoidance."""
        cfg = self.cfg

        pos_w = self._robot.data.root_pos_w
        quat = self._robot.data.root_quat_w
        lin_vel = self._robot.data.root_lin_vel_w
        ang_vel = self._robot.data.root_ang_vel_b

        # =====================================================================
        # HOVER STABILITY COSTS (from train_pointnav.py)
        # =====================================================================
        target_height = self._terrain.env_origins[:, 2] + cfg.init_target_height
        height_error = pos_w[:, 2] - target_height
        height_cost = height_error ** 2

        orientation_cost = 1.0 - quat[:, 0] ** 2
        xy_velocity_cost = (lin_vel[:, :2] ** 2).sum(dim=-1)
        z_velocity_cost = lin_vel[:, 2] ** 2
        angular_velocity_cost = (ang_vel ** 2).sum(dim=-1)
        action_deviation = self._actions - self._hover_action
        action_cost = (action_deviation ** 2).sum(dim=-1)

        hover_cost = (
            cfg.hover_height_weight * height_cost +
            cfg.hover_orientation_weight * orientation_cost +
            cfg.hover_xy_velocity_weight * xy_velocity_cost +
            cfg.hover_z_velocity_weight * z_velocity_cost +
            cfg.hover_angular_velocity_weight * angular_velocity_cost +
            cfg.hover_action_weight * action_cost
        )

        hover_reward = -cfg.hover_reward_scale * hover_cost + cfg.hover_reward_constant
        hover_reward = hover_reward.clamp(0.0, cfg.hover_reward_constant)

        # =====================================================================
        # NAVIGATION REWARDS (from train_pointnav.py)
        # =====================================================================
        delta_xy = pos_w[:, :2] - self._goal_pos[:, :2]
        dist_xy = torch.norm(delta_xy, dim=-1)

        gate = torch.clamp(1.0 - (dist_xy / cfg.hover_gate_radius), 0.0, 1.0)
        hover_gate = cfg.hover_gate_min + (1.0 - cfg.hover_gate_min) * gate
        hover_reward = hover_reward * hover_gate

        progress = self._prev_dist_xy - dist_xy
        progress_reward = cfg.nav_progress_weight * progress
        self._prev_dist_xy = dist_xy.clone()

        just_reached = (dist_xy < cfg.goal_reach_threshold) & (~self._goal_reached)
        reach_bonus = just_reached.float() * cfg.nav_reach_bonus
        self._goal_reached = self._goal_reached | (dist_xy < cfg.goal_reach_threshold)

        v_xy = torch.norm(lin_vel[:, :2], dim=-1)
        near_goal = dist_xy < cfg.nav_braking_radius
        speed_reduction = self._prev_speed - v_xy
        braking_reward = torch.where(
            near_goal,
            cfg.nav_braking_weight * speed_reduction,
            torch.zeros_like(speed_reduction)
        )
        self._prev_speed = v_xy.clone()

        height_tracking_reward = cfg.nav_height_track_weight * torch.exp(-5.0 * height_error ** 2)

        height_below = torch.relu(target_height - pos_w[:, 2])
        height_recovery = self._prev_height_below_target - height_below
        height_recovery_reward = cfg.nav_height_recovery_weight * torch.clamp(height_recovery, -0.5, 0.5)
        self._prev_height_below_target = height_below.clone()

        speed_excess = torch.relu(v_xy - cfg.nav_speed_penalty_threshold)
        speed_penalty = -cfg.nav_speed_penalty_weight * speed_excess ** 2

        height_above_ground = pos_w[:, 2] - self._terrain.env_origins[:, 2]
        low_margin = torch.relu(cfg.nav_low_height_penalty_floor - height_above_ground)
        low_height_penalty = -cfg.nav_low_height_penalty_weight * (low_margin ** 2)

        # =====================================================================
        # OBSTACLE AVOIDANCE REWARDS (NEW)
        # =====================================================================

        # 1. Collision penalty: large negative reward when touching obstacle
        collision_penalty = self._in_collision.float() * cfg.obs_collision_penalty

        # 2. Proximity warning: quadratic penalty when within warning radius
        # Uses minimum distance to any active obstacle surface
        proximity_margin = torch.relu(
            cfg.obs_proximity_warning_radius - self._min_obstacle_dist
        )
        proximity_penalty = -cfg.obs_proximity_warning_weight * (proximity_margin ** 2)

        # 3. Clearance reward: small bonus for maintaining safe distance
        safe_from_obstacles = (self._min_obstacle_dist > cfg.obs_proximity_warning_radius).float()
        clearance_reward = cfg.obs_clearance_reward_weight * safe_from_obstacles

        # =====================================================================
        # TOTAL REWARD
        # =====================================================================
        reward = (
            hover_reward
            + progress_reward
            + braking_reward
            + height_tracking_reward
            + height_recovery_reward
            + speed_penalty
            + low_height_penalty
            + reach_bonus
            + collision_penalty
            + proximity_penalty
            + clearance_reward
        )

        # Track stats
        self._episode_sums["height_cost"] += height_cost
        self._episode_sums["orientation_cost"] += orientation_cost
        self._episode_sums["xy_velocity_cost"] += xy_velocity_cost
        self._episode_sums["z_velocity_cost"] += z_velocity_cost
        self._episode_sums["angular_velocity_cost"] += angular_velocity_cost
        self._episode_sums["action_cost"] += action_cost
        self._episode_sums["hover_reward"] += hover_reward
        self._episode_sums["progress_reward"] += progress_reward
        self._episode_sums["braking_reward"] += braking_reward
        self._episode_sums["speed_penalty"] += speed_penalty
        self._episode_sums["reach_bonus"] += reach_bonus
        self._episode_sums["collision_penalty"] += collision_penalty
        self._episode_sums["proximity_penalty"] += proximity_penalty
        self._episode_sums["clearance_reward"] += clearance_reward
        self._episode_sums["total_reward"] += reward
        self._episode_sums["goal_reached"] += just_reached.float()
        self._episode_sums["final_distance"] = dist_xy
        self._episode_sums["collision_count"] += self._in_collision.float()

        return reward

    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions (same as train_pointnav.py + collision)."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        cfg = self.cfg
        pos_w = self._robot.data.root_pos_w
        quat = self._robot.data.root_quat_w
        lin_vel = self._robot.data.root_lin_vel_w
        ang_vel = self._robot.data.root_ang_vel_b

        # XY position check
        xy_offset = pos_w[:, :2] - self._terrain.env_origins[:, :2]
        xy_exceeded = torch.norm(xy_offset, dim=-1) > cfg.term_xy_threshold

        # Height check (persistence-based)
        height = pos_w[:, 2] - self._terrain.env_origins[:, 2]

        too_low_hard = height < cfg.term_z_hard_min
        too_high_hard = height > cfg.term_z_hard_max

        too_low_soft = height < cfg.term_z_soft_min
        too_high_soft = height > cfg.term_z_soft_max

        self._height_low_violation_counter = torch.where(
            too_low_soft,
            self._height_low_violation_counter + 1,
            torch.zeros_like(self._height_low_violation_counter)
        )
        self._height_high_violation_counter = torch.where(
            too_high_soft,
            self._height_high_violation_counter + 1,
            torch.zeros_like(self._height_high_violation_counter)
        )

        too_low_persistence = self._height_low_violation_counter >= cfg.term_z_persistence_steps
        too_high_persistence = self._height_high_violation_counter >= cfg.term_z_persistence_steps

        too_low = too_low_hard | too_low_persistence
        too_high = too_high_hard | too_high_persistence

        # Tilt check (persistence-based)
        tilt_angle = self._get_tilt_angle(quat)
        hard_tilted = tilt_angle > cfg.term_tilt_hard_threshold
        soft_tilted = tilt_angle > cfg.term_tilt_soft_threshold
        self._tilt_violation_counter = torch.where(
            soft_tilted,
            self._tilt_violation_counter + 1,
            torch.zeros_like(self._tilt_violation_counter)
        )
        persistence_tilted = self._tilt_violation_counter >= cfg.term_tilt_persistence_steps
        too_tilted = hard_tilted | persistence_tilted

        # Linear velocity check (persistence-based, XY only)
        v_xy = torch.norm(lin_vel[:, :2], dim=-1)
        hard_fast = v_xy > cfg.term_linear_velocity_hard_threshold
        soft_fast = v_xy > cfg.term_linear_velocity_soft_threshold
        self._linvel_violation_counter = torch.where(
            soft_fast,
            self._linvel_violation_counter + 1,
            torch.zeros_like(self._linvel_violation_counter)
        )
        persistence_fast = self._linvel_violation_counter >= cfg.term_linear_velocity_persistence_steps
        lin_vel_exceeded = hard_fast | persistence_fast

        # Angular velocity check (persistence-based)
        ang_vel_mag = torch.norm(ang_vel, dim=-1)
        hard_spin = ang_vel_mag > cfg.term_angular_velocity_hard_threshold
        soft_spin = ang_vel_mag > cfg.term_angular_velocity_soft_threshold
        self._angvel_violation_counter = torch.where(
            soft_spin,
            self._angvel_violation_counter + 1,
            torch.zeros_like(self._angvel_violation_counter)
        )
        persistence_spin = self._angvel_violation_counter >= cfg.term_angular_velocity_persistence_steps
        ang_vel_exceeded = hard_spin | persistence_spin

        # Collision termination (optional)
        collision_terminated = self._in_collision & cfg.term_on_collision

        # Safety terminations
        safety_terminated = (
            xy_exceeded | too_low | too_high | too_tilted
            | lin_vel_exceeded | ang_vel_exceeded | collision_terminated
        )

        # Goal reached
        goal_terminated = self._goal_reached
        terminated = safety_terminated | goal_terminated

        # Store termination reasons for done envs (survives auto-reset for eval use)
        done = terminated | time_out
        if done.any():
            self._last_done_goal_reached[done] = goal_terminated[done]
            self._last_done_collision[done] = collision_terminated[done]

        # Update counters
        self._term_counters["xy_exceeded"] += xy_exceeded.sum().item()
        self._term_counters["too_low"] += too_low.sum().item()
        self._term_counters["too_high"] += too_high.sum().item()
        self._term_counters["too_tilted"] += too_tilted.sum().item()
        self._term_counters["lin_vel_exceeded"] += lin_vel_exceeded.sum().item()
        self._term_counters["ang_vel_exceeded"] += ang_vel_exceeded.sum().item()
        self._term_counters["collision"] += collision_terminated.sum().item()
        self._term_counters["goal_reached"] += goal_terminated.sum().item()
        self._term_counters["timeout"] += (time_out & ~terminated).sum().item()
        self._term_counters["total"] += (terminated | time_out).sum().item()

        return terminated, time_out

    def _sample_goals(self, env_ids: torch.Tensor):
        """Sample goal positions (same as train_pointnav.py)."""
        n = len(env_ids)
        cfg = self.cfg

        distance = torch.empty(n, device=self.device).uniform_(
            cfg.goal_min_distance, cfg.goal_max_distance
        )
        angle = torch.empty(n, device=self.device).uniform_(0, 2 * math.pi)

        x_offset = distance * torch.cos(angle)
        y_offset = distance * torch.sin(angle)

        goal = self._terrain.env_origins[env_ids].clone()
        goal[:, 0] += x_offset
        goal[:, 1] += y_offset
        goal[:, 2] = goal[:, 2] + cfg.goal_height

        self._goal_pos[env_ids] = goal

    def _sample_obstacles(self, env_ids: torch.Tensor):
        """Sample obstacle positions between drone spawn and goal.

        For each environment, randomly place 1-3 cylinders along the path
        from drone to goal with some lateral spread. This creates scenarios
        where the drone must navigate around obstacles to reach the goal.

        Obstacle placement strategy:
        1. Sample a point along the drone-to-goal line (parametric t âˆˆ [0.2, 0.8])
        2. Add lateral offset perpendicular to the line
        3. Randomize radius within configured range
        4. Deactivate unused obstacle slots by moving them far away
        """
        n = len(env_ids)
        cfg = self.cfg

        # Drone spawn position (center of env, at target height)
        spawn_pos = self._terrain.env_origins[env_ids].clone()
        spawn_pos[:, 2] += cfg.init_target_height

        # Goal positions (already sampled)
        goal_pos = self._goal_pos[env_ids]

        # Direction from spawn to goal (XY only)
        direction = goal_pos[:, :2] - spawn_pos[:, :2]  # (n, 2)
        dir_norm = torch.norm(direction, dim=-1, keepdim=True).clamp(min=1e-6)
        dir_unit = direction / dir_norm  # (n, 2)

        # Perpendicular direction (rotate 90deg)
        perp_unit = torch.stack([-dir_unit[:, 1], dir_unit[:, 0]], dim=-1)  # (n, 2)

        # Random number of active obstacles per env
        # Handle curriculum phase 1 where max_obstacles=0
        if cfg.max_obstacles == 0:
            num_active = torch.zeros(n, dtype=torch.long, device=self.device)
        else:
            num_active = torch.randint(
                cfg.min_obstacles, cfg.max_obstacles + 1, (n,), device=self.device
            )

        # Deactivate ALL obstacle slots first (handles curriculum transitions
        # where max_obstacles decreases — previously active prims get moved away)
        far_away_base = self._terrain.env_origins[env_ids].clone()
        far_away_base[:, 0] += 100.0
        far_away_base[:, 1] += 100.0
        far_away_base[:, 2] += 1.0
        for obs_i in range(self.cfg._max_obstacle_prims):
            self._obstacle_active[env_ids, obs_i] = False
            if obs_i >= cfg.max_obstacles:
                # Move unused prims far away
                self._obstacle_pos[env_ids, obs_i] = far_away_base
                deactivate_pose = torch.zeros(n, 7, device=self.device)
                deactivate_pose[:, :3] = far_away_base
                deactivate_pose[:, 3] = 1.0
                if obs_i < len(self._obstacles):
                    self._obstacles[obs_i].write_root_link_pose_to_sim(deactivate_pose, env_ids=env_ids)

        for obs_i in range(cfg.max_obstacles):
            # Is this obstacle active for each env?
            active = obs_i < num_active  # (n,) bool

            # Sample parametric position along spawn-to-goal line
            # t limits computed from min-distance constraints relative to drone-goal distance
            t_min = (cfg.obstacle_min_dist_from_drone / dir_norm.squeeze(-1)).clamp(0.1, 0.4)
            t_max = (1.0 - cfg.obstacle_min_dist_from_goal / dir_norm.squeeze(-1)).clamp(0.6, 0.9)
            t = torch.empty(n, device=self.device).uniform_(0.0, 1.0) * (t_max - t_min) + t_min

            # Base position along the line
            base_pos = spawn_pos[:, :2] + t.unsqueeze(1) * direction  # (n, 2)

            # Lateral offset
            lateral = torch.empty(n, device=self.device).uniform_(
                -cfg.obstacle_lateral_spread, cfg.obstacle_lateral_spread
            )
            offset_pos = base_pos + lateral.unsqueeze(1) * perp_unit  # (n, 2)

            # Random radius
            radii = torch.empty(n, device=self.device).uniform_(
                cfg.obstacle_radius_min, cfg.obstacle_radius_max
            )

            # Set obstacle position (XY from sampling, Z centered at flight height)
            obs_pos = torch.zeros(n, 3, device=self.device)
            obs_pos[:, 0] = offset_pos[:, 0]
            obs_pos[:, 1] = offset_pos[:, 1]
            obs_pos[:, 2] = self._terrain.env_origins[env_ids, 2] + cfg.obstacle_height / 2

            # For inactive obstacles, move them far away (100m offset)
            far_away = self._terrain.env_origins[env_ids].clone()
            far_away[:, 0] += 100.0
            far_away[:, 1] += 100.0
            far_away[:, 2] += 1.0

            obs_pos = torch.where(active.unsqueeze(1), obs_pos, far_away)

            # Store positions and radii
            self._obstacle_pos[env_ids, obs_i] = obs_pos
            self._obstacle_radii[env_ids, obs_i] = radii
            self._obstacle_active[env_ids, obs_i] = active

            # Write obstacle pose to simulation
            obstacle_pose = torch.zeros(n, 7, device=self.device)
            obstacle_pose[:, :3] = obs_pos
            obstacle_pose[:, 3] = 1.0  # w=1 identity quaternion

            self._obstacles[obs_i].write_root_link_pose_to_sim(obstacle_pose, env_ids=env_ids)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specified environments."""
        if env_ids is None or len(env_ids) == 0:
            return

        # Record episode lengths
        if len(env_ids) > 0:
            ep_lengths = self.episode_length_buf[env_ids].cpu().tolist()
            self._episode_lengths.extend(ep_lengths)
            if len(self._episode_lengths) > self._max_episode_buffer:
                self._episode_lengths = self._episode_lengths[-self._max_episode_buffer:]

        # Log stats before reset
        if len(env_ids) > 0 and hasattr(self, '_episode_sums'):
            extras = {}
            for key, values in self._episode_sums.items():
                if key in ["goal_reached", "final_distance", "collision_count"]:
                    extras[f"Episode/{key}"] = torch.mean(values[env_ids]).item()
                else:
                    avg = torch.mean(values[env_ids]).item()
                    steps = self.episode_length_buf[env_ids].float().mean().item()
                    if steps > 0:
                        extras[f"Episode/{key}"] = avg / steps

            reach_count = self._goal_reached[env_ids].float().sum().item()
            extras["Episode/reach_rate"] = reach_count / len(env_ids)

            self.extras["log"] = extras

        # Reset robot
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        n = len(env_ids)
        cfg = self.cfg

        # Sample goals FIRST
        self._sample_goals(env_ids)

        # Sample obstacles between drone and goal
        self._sample_obstacles(env_ids)

        # Reset goal-reached flag
        self._goal_reached[env_ids] = False

        # Initialize position
        guidance_mask = torch.rand(n, device=self.device) < cfg.init_guidance_probability

        pos = torch.zeros(n, 3, device=self.device)
        pos[~guidance_mask, 0] = torch.empty((~guidance_mask).sum(), device=self.device).uniform_(
            -cfg.init_max_xy_offset, cfg.init_max_xy_offset
        )
        pos[~guidance_mask, 1] = torch.empty((~guidance_mask).sum(), device=self.device).uniform_(
            -cfg.init_max_xy_offset, cfg.init_max_xy_offset
        )

        height_offset = torch.empty(n, device=self.device).uniform_(
            cfg.init_height_offset_min, cfg.init_height_offset_max
        )
        height_offset[guidance_mask] = 0
        pos[:, 2] = cfg.init_target_height + height_offset
        pos = pos + self._terrain.env_origins[env_ids]

        # Sample orientation
        quat = torch.zeros(n, 4, device=self.device)
        quat[:, 0] = 1.0
        if cfg.init_max_angle > 0:
            axis = torch.randn(n, 3, device=self.device)
            axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-8)
            angle = torch.empty(n, device=self.device).uniform_(0, cfg.init_max_angle)
            angle[guidance_mask] = 0
            half_angle = angle / 2
            quat[:, 0] = torch.cos(half_angle)
            quat[:, 1:] = axis * torch.sin(half_angle).unsqueeze(-1)
            quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)

        # Sample velocities
        lin_vel = torch.empty(n, 3, device=self.device).uniform_(
            -cfg.init_max_linear_velocity, cfg.init_max_linear_velocity
        )
        lin_vel[guidance_mask] = 0

        ang_vel = torch.empty(n, 3, device=self.device).uniform_(
            -cfg.init_max_angular_velocity, cfg.init_max_angular_velocity
        )
        ang_vel[guidance_mask] = 0

        # Write to sim
        root_pose = torch.cat([pos, quat], dim=-1)
        root_vel = torch.cat([lin_vel, ang_vel], dim=-1)
        self._robot.write_root_pose_to_sim(root_pose, env_ids)
        self._robot.write_root_velocity_to_sim(root_vel, env_ids)

        # Initialize motor state
        self._rpm_state[env_ids] = self._hover_rpm
        self._action_history[env_ids] = self._hover_action
        self._actions[env_ids] = self._hover_action

        # Initialize navigation state
        delta_xy = self._goal_pos[env_ids, :2] - pos[:, :2]
        self._prev_dist_xy[env_ids] = torch.norm(delta_xy, dim=-1)
        self._prev_speed[env_ids] = 0.0
        self._prev_height_below_target[env_ids] = 0.0

        # Initialize obstacle avoidance state
        self._in_collision[env_ids] = False
        self._min_obstacle_dist[env_ids] = self.cfg.proximity_max_range
        self._proximity_bins[env_ids] = 1.0

        # Disturbances
        if cfg.enable_disturbance:
            self._disturbance_force[env_ids] = torch.randn(n, 3, device=self.device) * cfg.disturbance_force_std
            self._disturbance_torque[env_ids] = torch.randn(n, 3, device=self.device) * cfg.disturbance_torque_std

        # Reset persistence counters
        self._tilt_violation_counter[env_ids] = 0
        self._angvel_violation_counter[env_ids] = 0
        self._linvel_violation_counter[env_ids] = 0
        self._height_low_violation_counter[env_ids] = 0
        self._height_high_violation_counter[env_ids] = 0

        # Reset stats
        for key in self._episode_sums:
            self._episode_sums[key][env_ids] = 0.0

    def get_episode_length_stats(self) -> dict:
        if len(self._episode_lengths) < 10:
            return {"mean": 0, "p50": 0, "p90": 0, "count": len(self._episode_lengths)}
        lengths = torch.tensor(self._episode_lengths, dtype=torch.float32)
        return {
            "mean": lengths.mean().item(),
            "p50": lengths.median().item(),
            "p90": lengths.quantile(0.9).item(),
            "count": len(self._episode_lengths),
        }

    def clear_episode_stats(self):
        self._episode_lengths = []
        for k in self._term_counters:
            self._term_counters[k] = 0

    def get_termination_diagnostics(self, max_episode_length: int) -> dict:
        ep_stats = self.get_episode_length_stats()
        p50 = ep_stats["p50"]
        p90 = ep_stats["p90"]
        H = max_episode_length

        total = max(self._term_counters["total"], 1)
        term_pcts = {
            reason: 100.0 * count / total
            for reason, count in self._term_counters.items()
            if reason != "total"
        }

        return {
            "episode_lengths": ep_stats,
            "termination_pcts": term_pcts,
        }

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set up debug visualization for goals and obstacles."""
        if debug_vis:
            # Goal markers (green cubes)
            marker_cfg = CUBOID_MARKER_CFG.copy()
            marker_cfg.markers["cuboid"].size = (0.1, 0.1, 0.1)
            marker_cfg.prim_path = "/Visuals/Command/goal_position"
            self._goal_markers = VisualizationMarkers(marker_cfg)
        else:
            if hasattr(self, "_goal_markers"):
                self._goal_markers.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update debug visualization."""
        if hasattr(self, "_goal_markers"):
            self._goal_markers.visualize(self._goal_pos)


# ==============================================================================
# L2F-Compatible Actor Network (Extended for 157-dim observations)
# ==============================================================================

class L2FActorNetwork(nn.Module):
    """Actor network with extended observation space for obstacle avoidance.

    Architecture: 157 -> 64 (tanh) -> 64 (tanh) -> 4 (tanh)

    The additional 8 proximity inputs are naturally handled by the first
    linear layer. The tanh activations and small hidden size ensure the
    network stays deployable on STM32 firmware.
    """

    HOVER_ACTION = 2.0 * math.sqrt(0.027 * 9.81 / (4 * 3.16e-10)) / 21702.0 - 1.0

    def __init__(self, obs_dim: int = 157, hidden_dim: int = 64, action_dim: int = 4, init_std: float = 0.5):
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
        return torch.tanh(self.fc3(x))

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        mean = self.forward(obs)
        if deterministic:
            return mean
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        return dist.sample().clamp(-1.0, 1.0)

    def get_action_and_log_prob(self, obs: torch.Tensor):
        mean = self.forward(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.clamp(-1.0, 1.0), log_prob


class L2FCriticNetwork(nn.Module):
    """Critic network for obstacle avoidance."""

    def __init__(self, obs_dim: int = 157, hidden_dim: int = 64):
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
        return self.fc3(x).squeeze(-1)


# ==============================================================================
# PPO Agent (same as train_pointnav.py, different obs_dim default)
# ==============================================================================

class RunningMeanStd:
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
    """PPO Agent for obstacle avoidance training.

    Performance optimizations vs naive implementation:
    - Minibatch SGD (saturates GPU with many small batches per epoch)
    - torch.no_grad() / inference_mode during rollout collection
    - Separate obs normalizer update (no autograd overhead)
    - Fused optimizer step with set_to_none=True
    - Pre-computed std outside minibatch loop
    """

    def __init__(
        self,
        obs_dim: int = 157,
        action_dim: int = 4,
        device: torch.device = None,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        epochs: int = 5,
        entropy_coef: float = 0.005,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        minibatch_size: int = 8192,
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.minibatch_size = minibatch_size

        self.actor = L2FActorNetwork(obs_dim, 64, action_dim).to(device)
        self.critic = L2FCriticNetwork(obs_dim, 64).to(device)
        self.obs_normalizer = RunningMeanStd((obs_dim,), device=device)
        self.normalize_observations = True

        # Fused Adam is faster on CUDA
        use_fused = (device is not None and
                     (device == 'cuda' or (hasattr(device, 'type') and device.type == 'cuda')
                      or (isinstance(device, str) and device.startswith('cuda'))))
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr, fused=use_fused,
        )

    @torch.no_grad()
    def normalize_obs(self, obs: torch.Tensor, update: bool = True) -> torch.Tensor:
        if not self.normalize_observations:
            return obs
        if update:
            self.obs_normalizer.update(obs)
        return self.obs_normalizer.normalize(obs)

    @torch.inference_mode()
    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        obs_norm = self.normalize_obs(obs, update=False)
        return self.actor.get_action(obs_norm, deterministic)

    @torch.inference_mode()
    def get_action_and_value(self, obs: torch.Tensor):
        """Collect action, log_prob, value for rollout (no gradients needed)."""
        self.obs_normalizer.update(obs)  # update stats separately
        obs_norm = self.obs_normalizer.normalize(obs)
        action, log_prob = self.actor.get_action_and_log_prob(obs_norm)
        value = self.critic(obs_norm)
        return action, log_prob, value

    @torch.inference_mode()
    def get_value(self, obs: torch.Tensor):
        obs_norm = self.normalize_obs(obs, update=False)
        return self.critic(obs_norm)

    def update(self, obs, actions, log_probs, returns, advantages):
        """Minibatch PPO update — shuffles data and processes in small batches.

        This keeps the GPU fully saturated by processing many small forward+backward
        passes per epoch, rather than one huge full-batch pass.
        """
        N = obs.shape[0]
        obs = obs.detach()
        actions = actions.detach()
        log_probs = log_probs.detach()
        returns = returns.detach()
        advantages = advantages.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Pre-normalize observations ONCE (shared across all minibatches)
        obs_norm = self.obs_normalizer.normalize(obs)

        total_loss = 0.0
        num_updates = 0
        mb = self.minibatch_size

        for epoch in range(self.epochs):
            # Shuffle indices each epoch
            perm = torch.randperm(N, device=self.device)

            for start in range(0, N, mb):
                end = min(start + mb, N)
                idx = perm[start:end]

                mb_obs = obs_norm[idx]
                mb_actions = actions[idx]
                mb_log_probs = log_probs[idx]
                mb_returns = returns[idx]
                mb_advantages = advantages[idx]

                # Forward pass
                mean = self.actor(mb_obs)
                std = torch.exp(self.actor.log_std)
                dist = torch.distributions.Normal(mean, std)
                new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = (new_log_probs - mb_log_probs).exp()
                clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * mb_advantages
                policy_loss = -torch.min(ratio * mb_advantages, clip_adv).mean()

                values = self.critic(mb_obs)
                value_loss = ((values - mb_returns) ** 2).mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                num_updates += 1

        return total_loss / max(num_updates, 1)

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

@torch.no_grad()
def compute_gae(rewards, values, dones, next_value, gamma=0.99, gae_lambda=0.95):
    """Vectorized GAE computation — runs entirely on GPU, no Python loops."""
    T = rewards.shape[0]
    not_dones = 1.0 - dones.float()  # (T, N)

    # Build next_values: shift values by 1, last entry is bootstrap
    next_values = torch.cat([values[1:], next_value.unsqueeze(0)], dim=0)  # (T, N)

    # TD residuals
    deltas = rewards + gamma * next_values * not_dones - values  # (T, N)

    # Reverse-scan GAE (must be sequential, but minimize Python overhead)
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(rewards[0])  # (N,) — on GPU
    for t in range(T - 1, -1, -1):
        last_gae = deltas[t] + gamma * gae_lambda * not_dones[t] * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return returns, advantages


def sanity_test(env: CrazyfliePointNavObsAvoidEnv, num_steps: int = 100):
    """Run sanity tests for obstacle avoidance environment.

    Tests:
    1. Environment can reset
    2. Observation shape is correct (157 dims)
    3. Goals are within expected bounds
    4. Obstacles are spawned and active
    5. Proximity sensor produces valid output
    6. Random actions don't crash
    7. Rewards are finite
    8. Obstacle placement is between drone and goal
    """
    print("\n" + "="*60)
    print("SANITY TEST MODE - Point Nav + Obstacle Avoidance")
    print("="*60)

    # Test 1: Reset
    print("[Test 1] Reset environment...")
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    print(f"  âœ“ Reset successful")

    # Test 2: Observation shape
    print("[Test 2] Check observation shape...")
    expected_dim = env.cfg.observation_space
    actual_dim = obs.shape[-1]
    assert actual_dim == expected_dim, f"Expected {expected_dim} dims, got {actual_dim}"
    print(f"  âœ“ Observation shape correct: {obs.shape} ({expected_dim} dims)")

    # Test 3: Goal distances
    print("[Test 3] Check goal distances...")
    spawn_pos = env._terrain.env_origins.clone()
    spawn_pos[:, 2] += env.cfg.init_target_height
    goal_dist = torch.norm(env._goal_pos - spawn_pos, dim=-1)
    min_dist = goal_dist.min().item()
    max_dist = goal_dist.max().item()
    assert min_dist >= env.cfg.goal_min_distance - 0.01, f"Goals too close: {min_dist}"
    assert max_dist <= env.cfg.goal_max_distance + 0.01, f"Goals too far: {max_dist}"
    print(f"  âœ“ Goal distances in range: [{min_dist:.3f}, {max_dist:.3f}] m")

    # Test 4: Obstacle spawning
    print("[Test 4] Check obstacle spawning...")
    active_count = env._obstacle_active.sum(dim=-1)  # (N,)
    min_active = active_count.min().item()
    max_active = active_count.max().item()
    mean_active = active_count.float().mean().item()
    assert min_active >= env.cfg.min_obstacles, f"Too few obstacles: {min_active}"
    assert max_active <= env.cfg.max_obstacles, f"Too many obstacles: {max_active}"
    print(f"  âœ“ Active obstacles: min={min_active}, max={max_active}, mean={mean_active:.1f}")

    # Test 5: Proximity sensor
    print("[Test 5] Check proximity sensor output...")
    proximity = env._proximity_bins  # (N, 8)
    assert proximity.shape == (env.num_envs, env.cfg.num_proximity_sectors), \
        f"Wrong proximity shape: {proximity.shape}"
    assert (proximity >= 0.0).all() and (proximity <= 1.0).all(), \
        f"Proximity out of range: [{proximity.min():.3f}, {proximity.max():.3f}]"
    # At least some sectors should detect obstacles (not all max range)
    sectors_with_obstacles = (proximity < 1.0).any(dim=0).sum().item()
    print(f"  âœ“ Proximity shape: {proximity.shape}")
    print(f"  âœ“ Proximity range: [{proximity.min():.3f}, {proximity.max():.3f}]")
    print(f"  âœ“ Sectors with detections: {sectors_with_obstacles}/{env.cfg.num_proximity_sectors}")

    # Test 6: Obstacle positions are between drone and goal
    print("[Test 6] Check obstacle placement...")
    for obs_i in range(env.cfg.max_obstacles):
        active_mask = env._obstacle_active[:, obs_i]
        if active_mask.any():
            obs_pos_xy = env._obstacle_pos[active_mask, obs_i, :2]
            drone_pos_xy = spawn_pos[active_mask, :2]
            goal_pos_xy = env._goal_pos[active_mask, :2]

            # Check obstacles are within reasonable range of the drone-goal corridor
            obs_to_drone = torch.norm(obs_pos_xy - drone_pos_xy, dim=-1)
            obs_to_goal = torch.norm(obs_pos_xy - goal_pos_xy, dim=-1)

            print(f"  Obstacle {obs_i}: active in {active_mask.sum().item()} envs, "
                  f"dist_drone=[{obs_to_drone.min():.3f}, {obs_to_drone.max():.3f}] m, "
                  f"dist_goal=[{obs_to_goal.min():.3f}, {obs_to_goal.max():.3f}] m")
    print(f"  âœ“ Obstacle placement verified")

    # Test 7: Random steps
    print(f"[Test 7] Run {num_steps} random steps...")
    total_reward = 0.0
    collision_steps = 0
    for step in range(num_steps):
        action = torch.rand(env.num_envs, 4, device=env.device) * 2 - 1
        obs_dict, reward, terminated, truncated, info = env.step(action)
        assert torch.isfinite(reward).all(), f"Non-finite reward at step {step}"
        total_reward += reward.mean().item()
        collision_steps += env._in_collision.sum().item()

    avg_reward = total_reward / num_steps
    print(f"  âœ“ {num_steps} steps completed, avg reward: {avg_reward:.3f}")
    print(f"  âœ“ Total collision steps: {collision_steps} across all envs")

    # Test 8: Proximity observation dimensions
    print("[Test 8] Check observation components...")
    obs = obs_dict["policy"]
    pos_error = obs[:, :3]
    assert (pos_error.abs() <= env.cfg.obs_position_clip + 0.1).all(), "Position error not clipped"
    goal_rel = obs[:, 146:149]
    proximity_obs = obs[:, 149:157]
    assert (proximity_obs >= -0.1).all() and (proximity_obs <= 1.1).all(), \
        f"Proximity obs out of range: [{proximity_obs.min():.3f}, {proximity_obs.max():.3f}]"
    print(f"  âœ“ Observation components verified")
    print(f"    Proximity obs range: [{proximity_obs.min():.3f}, {proximity_obs.max():.3f}]")

    print("\n" + "="*60)
    print("ALL SANITY TESTS PASSED âœ“")
    print("="*60 + "\n")


# ==============================================================================
# Curriculum Manager
# ==============================================================================

class CurriculumManager:
    """Adaptive plateau-based curriculum that phases in obstacle difficulty.

    Phase 1 ("Navigate"): No obstacles. Agent learns stable flight + goal reaching.
        - All obstacles deactivated (min_obstacles=0, max_obstacles=0)
        - Collision penalty = 0
        - Focus: hover stability, progress toward goal, reach bonus
        - Advance when: plateau detected OR emergency exit

    Phase 2 ("Easy Avoidance"): 1 obstacle, reduced penalty.
        - min_obstacles=1, max_obstacles=1
        - Collision penalty = -10 (gentle)
        - Proximity warning active
        - Advance when: plateau detected OR emergency exit

    Phase 3 ("Full Avoidance"): 1-3 obstacles, moderate penalty.
        - min_obstacles=1, max_obstacles=3 (or user-specified)
        - Collision penalty = -20 (moderate)
        - Advance when: plateau detected OR emergency exit

    Phase 4 ("Hardened"): Full difficulty + collision termination.
        - Same obstacles as phase 3
        - Collision penalty = -30
        - term_on_collision = True
        - Runs until end of training
        
    Plateau Detection: Monitors reach rate & reward over sliding window. Advances when 
    improvement drops below threshold for sustained period, ensuring efficient progression.
    """

    def __init__(self, total_iterations: int, start_phase: int = 1,
                 user_max_obstacles: int = 3, user_collision_penalty: float = -20.0):
        self.total_iterations = total_iterations
        self.current_phase = start_phase
        self.user_max_obstacles = user_max_obstacles
        self.user_collision_penalty = user_collision_penalty

        # Plateau detection for phase advancement
        self.plateau_window = 15  # Check plateau over last 15 iterations
        self.plateau_threshold = 0.02  # 2% improvement threshold
        self.min_phase_iterations = 25  # Minimum iterations before considering plateau
        self.phase_start_iter = 0  # Track when current phase started
        
        # Performance history for plateau detection
        self.reach_rate_history = []
        self.reward_history = []
        
        # Emergency fallback: absolute maximum iterations per phase
        self._max_phase1_iters = int(total_iterations * 0.30)  # Max 30% in Phase 1
        self._max_phase2_iters = int(total_iterations * 0.40)  # Max 40% in Phase 2  
        self._max_phase3_iters = int(total_iterations * 0.50)  # Max 50% in Phase 3

    def get_phase_config(self, phase: int) -> dict:
        """Return config overrides for each phase."""
        if phase == 1:
            return {
                "min_obstacles": 0,
                "max_obstacles": 0,
                "obs_collision_penalty": 0.0,
                "obs_proximity_warning_weight": 0.0,
                "obs_clearance_reward_weight": 0.0,
                "term_on_collision": False,
                "label": "Phase 1: Navigate (no obstacles)",
            }
        elif phase == 2:
            return {
                "min_obstacles": 1,
                "max_obstacles": 1,
                "obs_collision_penalty": -10.0,
                "obs_proximity_warning_weight": 1.5,
                "obs_clearance_reward_weight": 0.1,
                "term_on_collision": False,
                "label": "Phase 2: Easy Avoidance (1 obstacle, gentle penalty)",
            }
        elif phase == 3:
            return {
                "min_obstacles": 1,
                "max_obstacles": min(self.user_max_obstacles, 3),
                "obs_collision_penalty": self.user_collision_penalty,
                "obs_proximity_warning_weight": 2.0,
                "obs_clearance_reward_weight": 0.1,
                "term_on_collision": False,
                "label": f"Phase 3: Full Avoidance (1-{min(self.user_max_obstacles, 3)} obstacles)",
            }
        else:  # phase 4
            return {
                "min_obstacles": 1,
                "max_obstacles": self.user_max_obstacles,
                "obs_collision_penalty": self.user_collision_penalty * 1.5,
                "obs_proximity_warning_weight": 2.5,
                "obs_clearance_reward_weight": 0.1,
                "term_on_collision": True,
                "label": f"Phase 4: Hardened (1-{self.user_max_obstacles} obstacles, collision terminates)",
            }

    def apply_phase(self, env: CrazyfliePointNavObsAvoidEnv, phase: int = None):
        """Apply curriculum phase config to the environment."""
        if phase is not None:
            self.current_phase = phase
        config = self.get_phase_config(self.current_phase)
        cfg = env.cfg
        cfg.min_obstacles = config["min_obstacles"]
        cfg.max_obstacles = config["max_obstacles"]
        cfg.obs_collision_penalty = config["obs_collision_penalty"]
        cfg.obs_proximity_warning_weight = config["obs_proximity_warning_weight"]
        cfg.obs_clearance_reward_weight = config["obs_clearance_reward_weight"]
        cfg.term_on_collision = config["term_on_collision"]
        return config

    def _detect_plateau(self, current_reach_rate: float, current_reward: float) -> bool:
        """Detect if performance has plateaued based on recent history."""
        # Add current metrics to history
        self.reach_rate_history.append(current_reach_rate)
        self.reward_history.append(current_reward)
        
        # Keep only recent history
        if len(self.reach_rate_history) > self.plateau_window:
            self.reach_rate_history.pop(0)
            self.reward_history.pop(0)
        
        # Need sufficient data and minimum phase time
        if (len(self.reach_rate_history) < self.plateau_window or 
            len(self.reach_rate_history) < self.min_phase_iterations):
            return False
            
        # Check if improvement has stagnated over recent window
        recent_window = self.plateau_window // 2
        if recent_window < 3:
            recent_window = 3
            
        early_period = self.reach_rate_history[-self.plateau_window:-recent_window]
        recent_period = self.reach_rate_history[-recent_window:]
        
        early_reward = self.reward_history[-self.plateau_window:-recent_window]
        recent_reward = self.reward_history[-recent_window:]
        
        if not early_period or not recent_period:
            return False
            
        early_reach_avg = sum(early_period) / len(early_period)
        recent_reach_avg = sum(recent_period) / len(recent_period)
        
        early_reward_avg = sum(early_reward) / len(early_reward)
        recent_reward_avg = sum(recent_reward) / len(recent_reward)
        
        # Check if improvement is below threshold
        reach_improvement = recent_reach_avg - early_reach_avg
        reward_improvement = recent_reward_avg - early_reward_avg
        
        # Plateau detected if improvement is negligible on either metric
        # Use lower threshold for already-high performance (reach > 95%)
        if recent_reach_avg > 0.95:
            # High performance phase - advance even with tiny improvement
            reach_plateaued = reach_improvement < 0.005  # Less than 0.5% improvement
        else:
            # Lower performance phase - require more improvement
            reach_plateaued = reach_improvement < self.plateau_threshold
        
        # Reward plateaued if change is less than 0.1
        reward_plateaued = abs(reward_improvement) < 0.1
        
        return reach_plateaued and reward_plateaued

    def check_advance(self, iteration: int, reach_rate: float, collision_rate: float, reward: float) -> bool:
        """Check if we should advance to the next phase based on plateaus and emergency exits only."""
        if self.current_phase >= 4:
            return False

        iterations_in_phase = iteration - self.phase_start_iter
        plateau_detected = self._detect_plateau(reach_rate, reward)
        
        advance = False
        phase_ready = iterations_in_phase >= self.min_phase_iterations
        
        if self.current_phase == 1:
            # Phase 1: Navigate without obstacles - advance when plateaued or emergency
            emergency_exit = iterations_in_phase >= self._max_phase1_iters
            
            if emergency_exit:
                print(f"[CURRICULUM] Emergency exit from Phase 1 (iter {iterations_in_phase})")
                advance = True
            elif phase_ready and plateau_detected:
                print(f"[CURRICULUM] Phase 1 plateau detected (reach: {reach_rate:.1%}, reward: {reward:.1f})")
                advance = True
                
        elif self.current_phase == 2:
            # Phase 2: Easy avoidance - advance when plateaued or emergency
            emergency_exit = iterations_in_phase >= self._max_phase2_iters
            
            if emergency_exit:
                print(f"[CURRICULUM] Emergency exit from Phase 2 (iter {iterations_in_phase})")
                advance = True
            elif phase_ready and plateau_detected:
                print(f"[CURRICULUM] Phase 2 plateau detected (reach: {reach_rate:.1%}, col: {collision_rate:.1%}, reward: {reward:.1f})")
                advance = True
                
        elif self.current_phase == 3:
            # Phase 3: Full avoidance - advance when plateaued or emergency
            emergency_exit = iterations_in_phase >= self._max_phase3_iters
            
            if emergency_exit:
                print(f"[CURRICULUM] Emergency exit from Phase 3 (iter {iterations_in_phase})")
                advance = True
            elif phase_ready and plateau_detected:
                print(f"[CURRICULUM] Phase 3 plateau detected (reach: {reach_rate:.1%}, col: {collision_rate:.1%}, reward: {reward:.1f})")
                advance = True

        if advance:
            self.current_phase += 1
            self.phase_start_iter = iteration
            # Clear history for new phase
            self.reach_rate_history.clear()
            self.reward_history.clear()
            return True
        return False


def train(env: CrazyfliePointNavObsAvoidEnv, agent: L2FPPOAgent, args):
    """High-performance training loop with pre-allocated buffers and timing."""
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints_pointnav_obs")
    os.makedirs(checkpoint_dir, exist_ok=True)

    steps_per_rollout = 128  # Shorter rollout = lower latency per iteration
    num_envs = env.num_envs
    obs_dim = env.cfg.observation_space
    act_dim = env.cfg.action_space
    total_samples = steps_per_rollout * num_envs

    best_reward = float("-inf")
    best_reach_rate = 0.0

    print(f"\n{'='*60}")
    print("Starting L2F Point Nav + Obstacle Avoidance PPO Training")
    print(f"{'='*60}")
    print(f"  Environments:       {num_envs}")
    print(f"  Max iterations:     {args.max_iterations}")
    print(f"  Steps per rollout:  {steps_per_rollout}")
    print(f"  Total batch size:   {total_samples}")
    print(f"  Minibatch size:     {agent.minibatch_size}")
    print(f"  PPO epochs:         {agent.epochs}")
    print(f"  Observation dim:    {obs_dim}")
    print(f"  Action dim:         {act_dim}")
    print(f"  Goal distance:      [{env.cfg.goal_min_distance}, {env.cfg.goal_max_distance}] m")
    print(f"  Obstacles:          {env.cfg.min_obstacles}-{env.cfg.max_obstacles} per env")
    print(f"  Proximity sectors:  {env.cfg.num_proximity_sectors}")
    use_curriculum = args.curriculum and not args.no_curriculum
    print(f"  Curriculum:         {'ENABLED' if use_curriculum else 'DISABLED'}")
    print(f"{'='*60}\n")

    # =========================================================================
    # PRE-ALLOCATE ROLLOUT BUFFERS (avoid repeated allocation)
    # =========================================================================
    device = env.device
    obs_buf    = torch.zeros(steps_per_rollout, num_envs, obs_dim, device=device)
    act_buf    = torch.zeros(steps_per_rollout, num_envs, act_dim, device=device)
    logp_buf   = torch.zeros(steps_per_rollout, num_envs, device=device)
    val_buf    = torch.zeros(steps_per_rollout, num_envs, device=device)
    rew_buf    = torch.zeros(steps_per_rollout, num_envs, device=device)
    done_buf   = torch.zeros(steps_per_rollout, num_envs, dtype=torch.bool, device=device)

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    # Resume from checkpoint
    start_iteration = 0
    if args.resume:
        if args.checkpoint:
            ckpt_path = args.checkpoint
        else:
            ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(checkpoint_dir, "final_model.pt")
        if os.path.exists(ckpt_path):
            print(f"\n[Resume] Loading checkpoint: {ckpt_path}")
            start_iteration, best_reward = agent.load(ckpt_path)
            print(f"[Resume] Starting from iteration {start_iteration}")

    # =========================================================================
    # CURRICULUM SETUP
    # =========================================================================
    use_curriculum = args.curriculum and not args.no_curriculum
    curriculum = None
    if use_curriculum:
        start_phase = args.curriculum_phase if args.curriculum_phase else 1
        curriculum = CurriculumManager(
            total_iterations=args.max_iterations,
            start_phase=start_phase,
            user_max_obstacles=args.max_obstacles,
            user_collision_penalty=args.collision_penalty,
        )
        config = curriculum.apply_phase(env)
        print(f"\n[CURRICULUM] {config['label']}")
        print(f"  Obstacles: {env.cfg.min_obstacles}-{env.cfg.max_obstacles}")
        print(f"  Collision penalty: {env.cfg.obs_collision_penalty}")
        print(f"  Term on collision: {env.cfg.term_on_collision}")
        print(f"  Advancement: Plateau-based (window={curriculum.plateau_window}, threshold={curriculum.plateau_threshold:.1%})")
        print(f"  Min iterations per phase: {curriculum.min_phase_iterations}")
        print(f"  Emergency exits: P1@{curriculum._max_phase1_iters}, P2@{curriculum._max_phase2_iters}, P3@{curriculum._max_phase3_iters}\n")

    # Timing
    iter_times = []
    train_start = time.time()

    for iteration in range(start_iteration, start_iteration + args.max_iterations):
        iter_start = time.time()

        # =====================================================================
        # ROLLOUT COLLECTION (inference_mode, no gradients, no autograd overhead)
        # =====================================================================
        episode_rewards = torch.zeros(num_envs, device=device)
        reach_count = 0
        episode_count = 0
        collision_count_total = 0

        rollout_start = time.time()
        for step in range(steps_per_rollout):
            # Inference-mode action + value (no autograd graph)
            action, log_prob, value = agent.get_action_and_value(obs)

            # Store in pre-allocated buffers
            obs_buf[step] = obs
            act_buf[step] = action
            logp_buf[step] = log_prob
            val_buf[step] = value

            obs_dict, reward, terminated, truncated, info = env.step(action)
            next_obs = obs_dict["policy"]
            done = terminated | truncated

            rew_buf[step] = reward
            done_buf[step] = done
            episode_rewards += reward

            collision_count_total += env._in_collision.sum().item()

            if "log" in env.extras and "Episode/reach_rate" in env.extras["log"]:
                reach_count += env.extras["log"]["Episode/reach_rate"] * done.sum().item()
                episode_count += done.sum().item()

            obs = next_obs
        rollout_time = time.time() - rollout_start

        # =====================================================================
        # GAE COMPUTATION (vectorized, no_grad)
        # =====================================================================
        gae_start = time.time()
        next_value = agent.get_value(obs)

        returns_t, advantages_t = compute_gae(
            rew_buf, val_buf, done_buf, next_value,
            gamma=agent.gamma, gae_lambda=agent.gae_lambda
        )
        gae_time = time.time() - gae_start

        # Flatten: (T, N, ...) -> (T*N, ...)
        obs_flat = obs_buf.reshape(-1, obs_dim)
        actions_flat = act_buf.reshape(-1, act_dim)
        log_probs_flat = logp_buf.reshape(-1)
        returns_flat = returns_t.reshape(-1)
        advantages_flat = advantages_t.reshape(-1)

        # =====================================================================
        # PPO UPDATE (minibatch SGD)
        # =====================================================================
        update_start = time.time()
        loss = agent.update(obs_flat, actions_flat, log_probs_flat, returns_flat, advantages_flat)
        update_time = time.time() - update_start

        iter_time = time.time() - iter_start
        iter_times.append(iter_time)

        # =====================================================================
        # LOGGING
        # =====================================================================
        mean_reward = episode_rewards.mean().item() / steps_per_rollout
        reach_rate = reach_count / max(episode_count, 1) if episode_count > 0 else 0.0
        collision_rate = collision_count_total / (steps_per_rollout * num_envs)

        is_best = mean_reward > best_reward
        if is_best:
            best_reward = mean_reward
            agent.save(os.path.join(checkpoint_dir, "best_model.pt"), iteration, best_reward)

        if reach_rate > best_reach_rate:
            best_reach_rate = reach_rate
            agent.save(os.path.join(checkpoint_dir, "best_reach_model.pt"), iteration, best_reward)

        if iteration % 10 == 0 or is_best:
            std = torch.exp(agent.actor.log_std).mean().item()
            star = " *BEST*" if is_best else ""
            fps = total_samples / iter_time
            print(f"[Iter {iteration:4d}] Rew: {mean_reward:8.3f} | Reach: {reach_rate*100:5.1f}% "
                  f"| Col: {collision_rate*100:4.1f}% | Std: {std:.3f} | Loss: {loss:.4f} "
                  f"| {iter_time:.1f}s (sim:{rollout_time:.1f} gae:{gae_time:.2f} ppo:{update_time:.1f}) "
                  f"| {fps/1000:.0f}k fps{star}")

        # Diagnostics every 50 iterations
        if iteration > 0 and iteration % 50 == 0:
            ep_stats = env.get_episode_length_stats()
            avg_iter = sum(iter_times[-50:]) / len(iter_times[-50:])
            avg_fps = total_samples / avg_iter
            elapsed = time.time() - train_start
            remaining_iters = args.max_iterations - (iteration - start_iteration)
            eta_s = avg_iter * remaining_iters

            print(f"\n  === DIAGNOSTICS ===")
            if curriculum:
                print(f"  Curriculum Phase: {curriculum.current_phase}/4")
            print(f"  Elapsed: {elapsed/60:.1f} min | ETA: {eta_s/60:.1f} min | Avg: {avg_iter:.1f}s/iter | {avg_fps/1000:.0f}k fps")
            print(f"  Episode Length: mean={ep_stats['mean']:.1f} p50={ep_stats['p50']:.1f} "
                  f"p90={ep_stats['p90']:.1f} (n={ep_stats['count']})")

            tc = env._term_counters
            total = max(tc["total"], 1)
            print(f"  Terminations: xy:{tc['xy_exceeded']/total*100:.1f}% "
                  f"low:{tc['too_low']/total*100:.1f}% high:{tc['too_high']/total*100:.1f}% "
                  f"tilt:{tc['too_tilted']/total*100:.1f}% linvel:{tc['lin_vel_exceeded']/total*100:.1f}% "
                  f"angvel:{tc['ang_vel_exceeded']/total*100:.1f}% col:{tc['collision']/total*100:.1f}% "
                  f"goal:{tc['goal_reached']/total*100:.1f}% timeout:{tc['timeout']/total*100:.1f}%")
            print(f"  ==========================\n")

            env.clear_episode_stats()

        # Curriculum phase advancement check (every 10 iterations)
        if curriculum and iteration > 0 and iteration % 10 == 0:
            advanced = curriculum.check_advance(iteration, reach_rate, collision_rate, mean_reward)
            if advanced:
                # Save checkpoint of previous curriculum phase before transitioning
                prev_phase = curriculum.current_phase - 1
                curriculum_checkpoint_path = os.path.join(checkpoint_dir, f"curriculum_phase{prev_phase}_final.pt")
                agent.save(curriculum_checkpoint_path, iteration, mean_reward)
                print(f"[CURRICULUM] Saved Phase {prev_phase} final checkpoint: curriculum_phase{prev_phase}_final.pt")
                
                config = curriculum.apply_phase(env)
                iterations_in_phase = iteration - curriculum.phase_start_iter
                print(f"\n{'*'*60}")
                print(f"[CURRICULUM] Advanced to {config['label']}")
                print(f"  Previous phase duration: {iterations_in_phase} iterations")
                print(f"  Current iteration: {iteration}")
                print(f"  Obstacles: {env.cfg.min_obstacles}-{env.cfg.max_obstacles}")
                print(f"  Collision penalty: {env.cfg.obs_collision_penalty}")
                print(f"  Term on collision: {env.cfg.term_on_collision}")
                print(f"{'*'*60}\n")

        if iteration > 0 and iteration % args.save_interval == 0:
            agent.save(os.path.join(checkpoint_dir, f"checkpoint_{iteration}.pt"), iteration, best_reward)

    total_time = time.time() - train_start
    agent.save(os.path.join(checkpoint_dir, "final_model.pt"), args.max_iterations, best_reward)
    
    # Save final curriculum phase checkpoint
    if curriculum:
        final_curriculum_checkpoint = os.path.join(checkpoint_dir, f"curriculum_phase{curriculum.current_phase}_final.pt")
        agent.save(final_curriculum_checkpoint, args.max_iterations, best_reward)
        print(f"Saved final curriculum checkpoint: curriculum_phase{curriculum.current_phase}_final.pt")
    
    print(f"\nTraining complete in {total_time/60:.1f} minutes!")
    print(f"  Best reward: {best_reward:.3f}, Best reach rate: {best_reach_rate*100:.1f}%")
    print(f"  Avg iteration: {sum(iter_times)/len(iter_times):.1f}s")
    print(f"  Checkpoints saved to: {checkpoint_dir}")


def play(env: CrazyfliePointNavObsAvoidEnv, agent: L2FPPOAgent, checkpoint_path: str):
    """Run trained policy with visualization."""
    iteration, best_reward = agent.load(checkpoint_path)
    print(f"\n[Play Mode] Loaded checkpoint from iteration {iteration}")
    print(f"[Play Mode] Best training reward: {best_reward:.3f}")
    print("[Play Mode] Press Ctrl+C to stop\n")

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    step_count = 0
    episode_reward = 0.0
    reach_count = 0
    episode_count = 0
    collision_steps = 0

    logger = FlightDataLogger()
    run_tag = int(time.time())
    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_dir = os.path.join(script_dir, "eval", "pointnav_obs", f"pointnav_obs_{run_tag}")
    os.makedirs(eval_dir, exist_ok=True)
    csv_filename = os.path.join(eval_dir, "pointnav_obs_eval_latest.csv")

    try:
        while simulation_app.is_running():
            action = agent.get_action(obs, deterministic=True)
            obs_dict, reward, terminated, truncated, info = env.step(action)
            obs = obs_dict["policy"]

            done = terminated | truncated
            episode_reward += reward.mean().item()
            step_count += 1
            collision_steps += env._in_collision.sum().item()

            logger.log_step(env, env_idx=0)

            if done.any():
                reaches = env._goal_reached[done].sum().item()
                reach_count += reaches
                episode_count += done.sum().item()

            if step_count % 500 == 0:
                reach_rate = reach_count / max(episode_count, 1) * 100
                col_rate = collision_steps / (step_count * env.num_envs) * 100
                print(f"[Step {step_count:5d}] Reward: {episode_reward:.2f} | "
                      f"Reach: {reach_rate:.1f}% | Col: {col_rate:.1f}% | Saving...")
                logger.save_and_plot(csv_filename, title_prefix="Point Nav + Obs Avoid", output_dir=eval_dir)
            elif step_count % 100 == 0:
                reach_rate = reach_count / max(episode_count, 1) * 100
                print(f"[Step {step_count:5d}] Reward: {episode_reward:.2f} | Reach: {reach_rate:.1f}%")

    except KeyboardInterrupt:
        print("\n[Play Mode] Stopped by user")
        reach_rate = reach_count / max(episode_count, 1) * 100
        col_rate = collision_steps / max(step_count * env.num_envs, 1) * 100
        print(f"Final reach rate: {reach_rate:.1f}% ({reach_count}/{episode_count})")
        print(f"Final collision rate: {col_rate:.1f}%")


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    cfg = CrazyfliePointNavObsAvoidEnvCfg()
    cfg.scene.num_envs = args.num_envs

    # =========================================================================
    # Apply CLI curriculum/difficulty overrides
    # =========================================================================
    if hasattr(args, 'term_on_collision') and args.term_on_collision:
        cfg.term_on_collision = True
    if hasattr(args, 'max_obstacles'):
        cfg.max_obstacles = args.max_obstacles
    if hasattr(args, 'min_obstacles'):
        cfg.min_obstacles = args.min_obstacles
    if hasattr(args, 'obstacle_radius_min'):
        cfg.obstacle_radius_min = args.obstacle_radius_min
    if hasattr(args, 'obstacle_radius_max'):
        cfg.obstacle_radius_max = args.obstacle_radius_max
    if hasattr(args, 'goal_min_distance'):
        cfg.goal_min_distance = args.goal_min_distance
    if hasattr(args, 'goal_max_distance'):
        cfg.goal_max_distance = args.goal_max_distance
        # obs_position_clip must be >= goal_max_distance
        cfg.obs_position_clip = max(cfg.obs_position_clip, args.goal_max_distance + 0.3)
    if hasattr(args, 'collision_penalty'):
        cfg.obs_collision_penalty = args.collision_penalty
    if hasattr(args, 'lateral_spread'):
        cfg.obstacle_lateral_spread = args.lateral_spread

    env = CrazyfliePointNavObsAvoidEnv(cfg)

    if args.sanity_test:
        sanity_test(env)
        env.close()
        simulation_app.close()
        return

    agent = L2FPPOAgent(
        obs_dim=cfg.observation_space,
        action_dim=cfg.action_space,
        device=env.device,
        lr=args.lr,
        gamma=args.gamma,
        minibatch_size=args.minibatch_size,
        epochs=args.ppo_epochs,
    )

    if args.play:
        if args.checkpoint is None:
            checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints_pointnav_obs")
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

