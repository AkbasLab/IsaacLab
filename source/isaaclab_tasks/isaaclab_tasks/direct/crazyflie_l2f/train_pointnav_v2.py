#!/usr/bin/env python3
"""
Crazyflie L2F Point Navigation Training Script - V2 (Ground-Start Training)

This script extends train_pointnav.py with ground-start training support,
enabling the simulated drone to learn takeoff from the ground just like
the physical Crazyflie.

AUTOMATED MULTI-STAGE PIPELINE:
  When staged_training=True (default), training proceeds through three stages
  automatically, with no manual intervention required:

  Stage 1 — Mid-Air Navigation (GS prob = 0%)
    Trains pure mid-air navigation from scratch until mid-air reach rate
    exceeds stage1_transition_reach. This replaces the need for a separate
    v1 training run.

  Stage 2 — Ground-Start Introduction (GS prob = 25%)
    Loads Stage 1's best checkpoint via load_pretrained() (resets obs
    normalizer for new obs distribution). Introduces ground-start episodes
    with 2x reward multiplier. Transitions to Stage 3 when GS reach rate
    exceeds stage2_transition_gs_reach.

  Stage 3 — Anti-Forgetting with Auto-Reload
    Monitors GS reach rate. If it drops below peak by more than
    stage3_forgetting_tolerance, automatically reloads best_gs_model.pt
    to recover lost takeoff capability.

  Use --no_staged to disable staging and control GS probability manually.
  Use --stage N to start from a specific stage (useful for experiments).

KEY CHANGES FROM V1 (train_pointnav.py):
1. GROUND-START SPAWNING: Configurable fraction of episodes start with the
   drone on the ground (z~0.03m, motors OFF, zero velocity) matching
   real-world Crazyflie test conditions.

2. TAKEOFF GRACE PERIOD: Ground-start episodes get a grace period (default 3s)
   during which low-height termination is suppressed and takeoff-specific
   rewards replace hover stability penalties.

3. TAKEOFF REWARDS: Dense reward for altitude gain and orientation maintenance
   during the grace period. After the grace period ends, normal hover+nav
   rewards resume seamlessly.

4. MOTOR RESPONSE: Per-environment motor alpha — ground-start episodes use
   faster motor response (tau=0.05s) matching real coreless DC spin-up.

5. RELAXED BOUNDS: Height termination thresholds lowered to accommodate
   ground-level operation; observation clips widened for the larger
   position range.

6. WIDER ACTION SCALE: Increased from 0.3 to 0.5 for more control authority
   during takeoff (T/W ratio ~1.56 at max action vs ~1.32 in v1).

PRESERVED FROM V1:
- Same 149-dim observation space (146 hover + 3 goal) for firmware compatibility
- Same 4-dim hover-centered action space
- Same L2F physics (100Hz, 27g, thrust coefficients)
- Same network architecture (64->64 tanh)
- Same PPO training algorithm
- All navigation rewards (progress, braking, reach bonus)

Usage (from IsaacLab directory):
    # Sanity test
    .\\isaaclab.bat -p source\\isaaclab_tasks\\isaaclab_tasks\\direct\\crazyflie_l2f\\train_pointnav_v2.py --sanity_test --num_envs 16

    # Train from scratch — fully automated (recommended)
    .\\isaaclab.bat -p source\\isaaclab_tasks\\isaaclab_tasks\\direct\\crazyflie_l2f\\train_pointnav_v2.py --num_envs 4096 --max_iterations 2000 --headless

    # Skip Stage 1 using an existing v1 checkpoint
    .\\isaaclab.bat -p source\\isaaclab_tasks\\isaaclab_tasks\\direct\\crazyflie_l2f\\train_pointnav_v2.py --pretrained source\\isaaclab_tasks\\isaaclab_tasks\\direct\\crazyflie_l2f\\checkpoints_pointnav\\best_model.pt --num_envs 4096 --max_iterations 1000 --headless

    # Manual mode (no staging, fixed 25% GS)
    .\\isaaclab.bat -p source\\isaaclab_tasks\\isaaclab_tasks\\direct\\crazyflie_l2f\\train_pointnav_v2.py --no_staged --ground_start_prob 0.25 --num_envs 4096 --headless

    # Play mode with trained v2 checkpoint
    .\\isaaclab.bat -p source\\isaaclab_tasks\\isaaclab_tasks\\direct\\crazyflie_l2f\\train_pointnav_v2.py --play --checkpoint source\\isaaclab_tasks\\isaaclab_tasks\\direct\\crazyflie_l2f\\checkpoints_pointnav_v2\\best_model.pt --num_envs 64
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

# Isaac Sim setup - must happen before other imports
from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description="Crazyflie L2F PointNav V2 (Ground-Start)")

    # Mode selection
    parser.add_argument("--play", action="store_true", help="Run in play mode with trained model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for play mode or resume training")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint (uses --checkpoint or latest)")
    parser.add_argument("--sanity_test", action="store_true", help="Run sanity test (few steps, verify no crashes)")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to v1 checkpoint for fine-tuning (loads weights, resets obs normalizer)")

    # Training parameters
    parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments")
    parser.add_argument("--max_iterations", type=int, default=1500, help="Maximum training iterations")
    parser.add_argument("--save_interval", type=int, default=100, help="Save checkpoint every N iterations")

    # Hyperparameters (tuned for quadrotor)
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")

    # V2: Ground-start overrides
    parser.add_argument("--ground_start_prob", type=float, default=None,
                        help="Override ground-start probability (0.0=all mid-air, 1.0=all ground)")
    parser.add_argument("--no_curriculum", action="store_true",
                        help="Disable ground-start curriculum (use fixed probability)")
    parser.add_argument("--stage", type=int, default=None, choices=[1, 2, 3],
                        help="Override starting stage (1=mid-air, 2=GS intro, 3=anti-forgetting)")
    parser.add_argument("--no_staged", action="store_true",
                        help="Disable automatic staged training (use manual control)")

    # AppLauncher adds its own args (including --headless)
    AppLauncher.add_app_launcher_args(parser)

    args, _ = parser.parse_known_args()
    return args


# Check if Isaac Sim is already running (i.e., we're being imported by another script)
def _is_isaac_sim_running():
    """Check if Isaac Sim/Omniverse is already initialized."""
    try:
        import omni.kit.app
        app = omni.kit.app.get_app()
        return app is not None and app.is_running()
    except (ImportError, AttributeError, Exception):
        return False


# Only initialize AppLauncher if Isaac Sim isn't already running
if _is_isaac_sim_running():
    args = parse_args()
    try:
        import omni.kit.app
        simulation_app = omni.kit.app.get_app()
    except:
        simulation_app = None
else:
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

# Import flight evaluation utilities
from flight_eval_utils import FlightDataLogger


# ==============================================================================
# L2F Physics Constants (IDENTICAL to train_pointnav.py / train_hover.py)
# ==============================================================================

class L2FConstants:
    """Physical parameters matching learning-to-fly exactly."""

    MASS = 0.027          # kg (27g)
    ARM_LENGTH = 0.028    # m (28mm)
    GRAVITY = 9.81        # m/s²

    IXX = 3.85e-6         # kg·m²
    IYY = 3.85e-6         # kg·m²
    IZZ = 5.9675e-6       # kg·m²

    THRUST_COEFFICIENT = 3.16e-10  # N/RPM²
    TORQUE_COEFFICIENT = 0.005964552  # Nm/N
    RPM_MIN = 0.0
    RPM_MAX = 21702.0
    MOTOR_TIME_CONSTANT = 0.15  # seconds (mid-air)
    MOTOR_TIME_CONSTANT_GROUND = 0.05  # seconds (ground-start, faster to match real motor inertia)

    ROTOR_POSITIONS = [
        (0.028, -0.028, 0.0),   # M1: front-right
        (-0.028, -0.028, 0.0),  # M2: back-right
        (-0.028, 0.028, 0.0),   # M3: back-left
        (0.028, 0.028, 0.0),    # M4: front-left
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
# Environment Configuration (V2 - Ground-Start Support)
# ==============================================================================

@configclass
class CrazyfliePointNavV2EnvCfg(DirectRLEnvCfg):
    """Configuration for V2 environment with ground-start training.

    All v1 parameters are preserved. New parameters are in the
    GROUND-START TRAINING section below.
    """

    # --- Episode settings ---
    episode_length_s = 10.0
    decimation = 1

    # --- Spaces (UNCHANGED: 149-dim obs, 4-dim action) ---
    observation_space = 149
    action_space = 4
    state_space = 0
    debug_vis = True

    # --- Simulation (100 Hz) ---
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
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
            restitution=0.0,  # No bounce (important for ground-start)
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
    # HOVER STABILITY REWARDS (from v1, unchanged)
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
    # NAVIGATION REWARDS (from v1, unchanged)
    # =========================================================================
    nav_progress_weight = 5.0
    nav_reach_bonus = 100.0
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
    # GOAL SAMPLING (from v1, unchanged)
    # =========================================================================
    goal_min_distance = 0.2
    goal_max_distance = 0.5
    goal_height = 1.0
    goal_reach_threshold = 0.1

    # =========================================================================
    # V2 CHANGE: WIDER OBSERVATION CLIPS
    # =========================================================================
    # Increased from v1 (1.0, 2.0) to handle ground-start observations.
    # At ground start, pos_error_z ≈ -0.97m. With clip=1.0 this is barely
    # representable. clip=2.0 gives comfortable headroom.
    # Velocity clip increased for takeoff dynamics.
    obs_position_clip = 2.0   # v1: 1.0
    obs_velocity_clip = 3.0   # v1: 2.0

    # =========================================================================
    # INITIALIZATION (mid-air spawns, same as v1)
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
    # V2 NEW: GROUND-START TRAINING CONFIGURATION
    # =========================================================================
    # These parameters control the ground-start curriculum that teaches the
    # policy takeoff behavior matching the real-world Crazyflie.
    #
    # DESIGN RATIONALE:
    # - The real Crazyflie starts on the ground with motors off
    # - v1 only trained mid-air spawns, so the policy never learned takeoff
    # - v2 mixes ground-start episodes into training with curriculum scheduling
    # - A grace period protects ground-start episodes from premature termination
    # - Dedicated takeoff rewards encourage altitude gain during the grace period
    # =========================================================================

    # Ground-start spawn parameters
    ground_start_probability: float = 0.25   # Fixed 25% ground-start (raised from 15% to prevent catastrophic forgetting)
    ground_start_height: float = 0.03        # m above ground plane (~half drone thickness)

    # Grace period: suppresses low-height termination and applies takeoff
    # rewards for the first N steps of ground-start episodes.
    # 300 steps = 3.0 seconds at 100Hz — enough time for motor spin-up
    # and initial climb.
    ground_start_grace_steps: int = 300

    # Curriculum: DISABLED (Option A — fixed probability gives consistent
    # gradient signal instead of starving the policy with a gate)
    ground_start_curriculum: bool = False
    ground_start_curriculum_start: float = 0.25
    ground_start_curriculum_end: float = 0.25
    ground_start_curriculum_iters: int = 1000
    ground_start_curriculum_gate: float = 0.0  # unused when curriculum=False

    # GS reward weight multiplier: Amplifies rewards for ground-start envs.
    # Since GS envs are a minority (25%), their gradient is proportionally
    # weaker. Multiplying their reward by this factor makes GS learning
    # signals compete on equal footing with mid-air signals.
    # Run 3 showed 47% GS reach collapsing to 3% (catastrophic forgetting).
    # A 2.0x multiplier effectively gives GS 50% of gradient weight despite
    # only 25% of episodes, preventing mid-air optimization from overwriting
    # takeoff behavior.
    ground_start_reward_multiplier: float = 2.0

    # Option B: Faster motor response for ground-start episodes.
    # Real coreless DC motors spin up much faster than our τ=0.15s model.
    # Using τ_gs=0.05s (α≈0.2) reduces dead time from ~300ms to ~100ms,
    # letting the policy learn takeoff without fighting excessive sim lag.
    ground_start_motor_tau: float = 0.05  # seconds (τ for ground-start episodes)

    # Takeoff rewards (applied during grace period only)
    takeoff_altitude_reward_weight: float = 10.0   # Per-meter altitude gain per step (was 5)
    takeoff_orientation_bonus_weight: float = 2.0  # Bonus for staying upright (was 1)
    takeoff_progress_weight: float = 2.0           # XY progress toward goal during takeoff

    # =========================================================================
    # V2 NEW: AUTOMATED MULTI-STAGE TRAINING PIPELINE
    # =========================================================================
    # When staged_training=True, the training loop automatically progresses
    # through three stages without manual intervention:
    #
    # STAGE 1 — Mid-Air Navigation (from scratch)
    #   ground_start_probability = 0.0
    #   Trains pure mid-air pointnav. Transitions to Stage 2 when mid-air
    #   reach rate exceeds stage1_transition_reach for stage1_transition_window
    #   consecutive diagnostic checks.
    #
    # STAGE 2 — Ground-Start Introduction
    #   ground_start_probability = ground_start_probability (default 0.25)
    #   Loads Stage 1 best checkpoint via load_pretrained() (resets obs
    #   normalizer + optimizer for new observation distribution).
    #   Transitions to Stage 3 when GS reach > stage2_transition_gs_reach.
    #
    # STAGE 3 — Anti-Forgetting with Auto-Reload
    #   Continues with same GS probability. If GS reach rate drops below
    #   (peak - stage3_forgetting_tolerance), automatically reloads
    #   best_gs_model.pt to recover lost takeoff capability.
    #   Respects a cooldown period between reloads.
    # =========================================================================
    staged_training: bool = True  # Enable automated staging (disable for manual control)

    # Stage 1 → 2 transition
    stage1_transition_reach: float = 0.80     # Mid-air reach rate to trigger transition
    stage1_transition_window: int = 2         # Consecutive diagnostic checks above threshold
    stage1_min_iterations: int = 100          # Minimum iterations in Stage 1

    # Stage 2 → 3 transition
    stage2_transition_gs_reach: float = 0.25  # GS reach rate to trigger transition
    stage2_transition_window: int = 2         # Consecutive checks above threshold
    stage2_min_iterations: int = 50           # Minimum iterations in Stage 2

    # Stage 3 anti-forgetting
    stage3_forgetting_tolerance: float = 0.15 # Reload if GS drops this much below peak
    stage3_reload_cooldown: int = 50          # Minimum iterations between reloads
    stage3_reload_max: int = 5                # Maximum number of auto-reloads

    # =========================================================================
    # V2 CHANGE: RELAXED TERMINATION THRESHOLDS
    # =========================================================================
    term_xy_threshold = 2.0  # unchanged

    # Height: Hard min lowered well below ground plane. The sim ground
    # collision prevents the drone from actually reaching negative heights,
    # but this removes artificial termination at low altitude.
    # Soft min lowered so persistence only triggers near ground after
    # staying there for a long time.
    term_z_soft_min: float = 0.01    # v1: 0.25 — almost at ground
    term_z_hard_min: float = -0.5    # v1: 0.10 — below ground = impossible in sim
    term_z_soft_max: float = 2.50    # unchanged
    term_z_hard_max: float = 3.00    # unchanged
    term_z_persistence_steps: int = 100  # v1: 50 — more patience at low altitude

    # Tilt (unchanged from v1)
    term_tilt_soft_threshold = 1.22
    term_tilt_hard_threshold = 2.62
    term_tilt_persistence_steps = 50

    # Linear velocity (unchanged from v1)
    term_linear_velocity_soft_threshold: float = 4.0
    term_linear_velocity_hard_threshold: float = 6.0
    term_linear_velocity_persistence_steps: int = 50

    # Angular velocity (unchanged from v1)
    term_angular_velocity_soft_threshold = 30.0
    term_angular_velocity_hard_threshold = 50.0
    term_angular_velocity_persistence_steps = 10

    # Domain randomization (unchanged)
    enable_disturbance = True
    disturbance_force_std = 0.0132
    disturbance_torque_std = 2.65e-5

    # Action history (unchanged)
    action_history_length = 32

    # =========================================================================
    # V2 CHANGE: WIDER ACTION SCALE
    # =========================================================================
    # Increased from 0.3 to 0.5 for more control authority during takeoff.
    #
    # With scale=0.3: max thrust   = 1.32× weight (slow climb)
    #                 min thrust   = 0.49× weight
    # With scale=0.5: max thrust   = 1.56× weight (healthy climb rate)
    #                 min thrust   = 0.25× weight (gentler descent)
    #
    # The wider range helps the policy modulate thrust for both:
    # - Aggressive spin-up during takeoff (needs >1× weight to lift off)
    # - Gentle touchdown/hover hold (needs fine thrust control)
    use_hover_centered_actions: bool = True
    action_scale: float = 0.5  # v1: 0.3


# ==============================================================================
# Environment Implementation (V2)
# ==============================================================================

class CrazyfliePointNavV2Env(DirectRLEnv):
    """Crazyflie environment for point navigation with ground-start training."""

    cfg: CrazyfliePointNavV2EnvCfg

    def __init__(self, cfg: CrazyfliePointNavV2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Cache physics parameters (IDENTICAL to v1)
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

        # Motor dynamics (per-env alpha for Option B)
        self._motor_alpha_default = min(self._dt / self._motor_tau, 1.0)
        self._motor_alpha_ground = min(self._dt / cfg.ground_start_motor_tau, 1.0)
        # Per-env motor alpha tensor (allows different dynamics for GS vs mid-air)
        # Shape (num_envs, 1) for broadcasting with (num_envs, 4) rpm_state
        self._motor_alpha = torch.full((cfg.scene.num_envs, 1), self._motor_alpha_default,
                                       device=self.device, dtype=torch.float32)

        # Rotor geometry tensors
        self._rotor_positions = torch.tensor(
            L2FConstants.ROTOR_POSITIONS, device=self.device, dtype=torch.float32
        )
        self._rotor_yaw_dirs = torch.tensor(
            L2FConstants.ROTOR_YAW_DIRS, device=self.device, dtype=torch.float32
        )

        # State tensors
        self._actions = torch.zeros(self.num_envs, 4, device=self.device)
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

        # Navigation state
        self._goal_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_dist_xy = torch.zeros(self.num_envs, device=self.device)
        self._prev_speed = torch.zeros(self.num_envs, device=self.device)
        self._prev_height_below_target = torch.zeros(self.num_envs, device=self.device)
        self._goal_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # =====================================================================
        # V2 NEW: Ground-start state tracking
        # =====================================================================
        # Per-env flag: True if this episode started on the ground
        self._is_ground_start = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Per-env grace period countdown (decremented each step in _get_dones)
        self._grace_period_remaining = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # Previous height above ground plane (for takeoff reward delta)
        self._prev_height_above_ground = torch.zeros(self.num_envs, device=self.device)

        # Runtime ground-start probability (updated by curriculum)
        self._current_ground_start_prob = cfg.ground_start_probability

        # Episode statistics (extended from v1)
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
            "takeoff_reward": torch.zeros(self.num_envs, device=self.device),  # V2 NEW
            "total_reward": torch.zeros(self.num_envs, device=self.device),
            "goal_reached": torch.zeros(self.num_envs, device=self.device),
            "final_distance": torch.zeros(self.num_envs, device=self.device),
        }

        # Termination counters (extended from v1)
        self._term_counters = {
            "xy_exceeded": 0,
            "too_low": 0,
            "too_high": 0,
            "too_tilted": 0,
            "lin_vel_exceeded": 0,
            "ang_vel_exceeded": 0,
            "goal_reached": 0,
            "timeout": 0,
            "total": 0,
            # V2 NEW: ground-start specific counters
            "ground_start_episodes": 0,
            "ground_start_goal_reached": 0,
            "midair_goal_reached": 0,
        }

        # Episode length tracking
        self._episode_lengths = []
        self._max_episode_buffer = 10000

        # Persistence counters (same as v1)
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

    # -----------------------------------------------------------------
    # Curriculum control (called from training loop)
    # -----------------------------------------------------------------

    def set_ground_start_probability(self, prob: float):
        """Update the ground-start probability (for curriculum scheduling)."""
        self._current_ground_start_prob = max(0.0, min(1.0, prob))

    def get_ground_start_probability(self) -> float:
        return self._current_ground_start_prob

    def get_ground_start_reach_rate(self) -> float:
        """Return current ground-start goal reach rate (for performance-gated curriculum)."""
        gs_eps = max(self._term_counters.get("ground_start_episodes", 0), 1)
        gs_reach = self._term_counters.get("ground_start_goal_reached", 0)
        return gs_reach / gs_eps

    # -----------------------------------------------------------------
    # Invariant verification
    # -----------------------------------------------------------------

    def _verify_invariants(self):
        """Verify critical invariants are satisfied."""
        cfg = self.cfg

        # Obs clip must cover goal range
        assert cfg.obs_position_clip >= cfg.goal_max_distance, \
            f"obs_position_clip ({cfg.obs_position_clip}) must be >= goal_max_distance ({cfg.goal_max_distance})"

        # Min goal distance > reach threshold
        assert cfg.goal_min_distance > cfg.goal_reach_threshold, \
            f"goal_min_distance ({cfg.goal_min_distance}) must be > goal_reach_threshold ({cfg.goal_reach_threshold})"

        # Mid-air spawn height within termination bounds
        assert cfg.term_z_hard_min < cfg.init_target_height < cfg.term_z_hard_max, \
            f"init_target_height ({cfg.init_target_height}) must be within [{cfg.term_z_hard_min}, {cfg.term_z_hard_max}]"

        # V2: Soft min can be below ground-start height (grace period protects)
        # Only require hard_min < soft_min (order check)
        assert cfg.term_z_hard_min < cfg.term_z_soft_min, \
            f"term_z_hard_min ({cfg.term_z_hard_min}) must be < term_z_soft_min ({cfg.term_z_soft_min})"

        # Height thresholds properly ordered on the high side
        assert cfg.init_target_height < cfg.term_z_soft_max < cfg.term_z_hard_max, \
            f"Height thresholds must be ordered: init_height < soft_max < hard_max"

        # Hover action sanity
        expected_hover = L2FConstants.hover_action()
        assert abs(expected_hover - 0.334) < 0.01, \
            f"hover_action ({expected_hover}) should be ~0.334"

        # Motor alpha in valid range (both default and ground-start)
        assert 0 < self._motor_alpha_default < 1, \
            f"motor_alpha_default ({self._motor_alpha_default}) must be in (0, 1)"
        assert 0 < self._motor_alpha_ground < 1, \
            f"motor_alpha_ground ({self._motor_alpha_ground}) must be in (0, 1)"
        assert self._motor_alpha_ground >= self._motor_alpha_default, \
            f"motor_alpha_ground ({self._motor_alpha_ground}) must be >= motor_alpha_default ({self._motor_alpha_default})"

        # Tilt thresholds ordered
        assert cfg.term_tilt_soft_threshold < cfg.term_tilt_hard_threshold
        assert cfg.term_tilt_persistence_steps > 0

        # Angular velocity thresholds ordered
        assert cfg.term_angular_velocity_soft_threshold < cfg.term_angular_velocity_hard_threshold
        assert cfg.term_angular_velocity_persistence_steps > 0

        # Linear velocity thresholds ordered
        assert cfg.term_linear_velocity_soft_threshold < cfg.term_linear_velocity_hard_threshold
        assert cfg.term_linear_velocity_persistence_steps > 0

        # Height persistence > 0
        assert cfg.term_z_persistence_steps > 0

        # V2: Ground-start height must be positive and below init height
        assert 0 < cfg.ground_start_height < cfg.init_target_height, \
            f"ground_start_height ({cfg.ground_start_height}) must be in (0, {cfg.init_target_height})"

        # V2: Obs clip must cover ground-start position error
        ground_pos_error = cfg.init_target_height - cfg.ground_start_height
        assert cfg.obs_position_clip >= ground_pos_error, \
            f"obs_position_clip ({cfg.obs_position_clip}) must be >= ground position error ({ground_pos_error:.2f})"

        print("[INVARIANTS] All invariants verified [OK]")

    # -----------------------------------------------------------------
    # Info printing
    # -----------------------------------------------------------------

    def _print_env_info(self):
        """Print environment configuration."""
        cfg = self.cfg
        print("\n" + "=" * 60)
        print("Crazyflie L2F Point Navigation V2 (Ground-Start)")
        print("=" * 60)
        print(f"  Physics dt:        {self._dt*1000:.1f} ms ({1/self._dt:.0f} Hz)")
        print(f"  Episode length:    {cfg.episode_length_s:.1f} s")
        print(f"  Num envs:          {self.num_envs}")
        print(f"  Observation dim:   {cfg.observation_space} (146 hover + 3 goal)")
        print(f"  Action dim:        {cfg.action_space}")
        print(f"  Mass:              {self._mass*1000:.1f} g")
        print(f"  Hover RPM:         {self._hover_rpm:.0f}")
        print(f"  Hover action:      {self._hover_action:.4f}")
        print(f"  Motor alpha:       {self._motor_alpha_default:.4f} (mid-air, τ={L2FConstants.MOTOR_TIME_CONSTANT}s)")
        print(f"  Motor alpha (GS):  {self._motor_alpha_ground:.4f} (ground-start, τ={cfg.ground_start_motor_tau}s)")
        print("--- Navigation ---")
        print(f"  Goal distance:     [{cfg.goal_min_distance:.2f}, {cfg.goal_max_distance:.2f}] m")
        print(f"  Reach threshold:   {cfg.goal_reach_threshold:.2f} m")
        print(f"  Position clip:     ±{cfg.obs_position_clip:.1f} m")
        print(f"  Velocity clip:     ±{cfg.obs_velocity_clip:.1f} m/s")
        print("--- Ground-Start (V2) ---")
        print(f"  Probability:       {cfg.ground_start_probability:.0%}")
        print(f"  Spawn height:      {cfg.ground_start_height:.3f} m")
        print(f"  Grace period:      {cfg.ground_start_grace_steps} steps ({cfg.ground_start_grace_steps*self._dt:.1f}s)")
        print(f"  Curriculum:        {'ON' if cfg.ground_start_curriculum else 'OFF'}")
        if cfg.ground_start_curriculum:
            print(f"    Start prob:      {cfg.ground_start_curriculum_start:.0%}")
            print(f"    End prob:        {cfg.ground_start_curriculum_end:.0%}")
            print(f"    Over iters:      {cfg.ground_start_curriculum_iters}")
            print(f"    Perf gate:       {cfg.ground_start_curriculum_gate:.0%} GS reach required")
        print(f"  Takeoff alt wt:    {cfg.takeoff_altitude_reward_weight:.1f}")
        print(f"  Takeoff orient wt: {cfg.takeoff_orientation_bonus_weight:.1f}")
        print(f"  GS motor τ:        {cfg.ground_start_motor_tau:.3f}s (α={self._motor_alpha_ground:.4f})")
        print(f"  GS reward mult:   {cfg.ground_start_reward_multiplier:.1f}x")
        print("--- Termination ---")
        print(f"  Height hard min:   {cfg.term_z_hard_min:.2f} m")
        print(f"  Height soft min:   {cfg.term_z_soft_min:.2f} m @ {cfg.term_z_persistence_steps} steps")
        print(f"  Height soft max:   {cfg.term_z_soft_max:.2f} m")
        print(f"  Height hard max:   {cfg.term_z_hard_max:.2f} m")
        print(f"  Tilt soft:         {cfg.term_tilt_soft_threshold:.2f} rad @ {cfg.term_tilt_persistence_steps} steps")
        print(f"  Tilt hard:         {cfg.term_tilt_hard_threshold:.2f} rad")
        print("--- Action Parameterization ---")
        if cfg.use_hover_centered_actions:
            print(f"  Mode:              HOVER-CENTERED (action=0 -> hover thrust)")
            print(f"  Action scale:      {cfg.action_scale:.2f}")
            max_rpm = self._hover_rpm + cfg.action_scale * (self._max_rpm - self._hover_rpm)
            min_rpm = self._hover_rpm * (1.0 - cfg.action_scale)
            max_tw = 4 * self._thrust_coef * max_rpm ** 2 / (self._mass * self._gravity)
            min_tw = 4 * self._thrust_coef * min_rpm ** 2 / (self._mass * self._gravity)
            print(f"  Thrust/Weight:     [{min_tw:.2f}, {max_tw:.2f}]")
        else:
            print(f"  Mode:              RAW (action=0 -> 50% thrust)")
        print("=" * 60 + "\n")

    # -----------------------------------------------------------------
    # Scene setup (IDENTICAL to v1)
    # -----------------------------------------------------------------

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # -----------------------------------------------------------------
    # Quaternion / tilt utilities (IDENTICAL to v1)
    # -----------------------------------------------------------------

    def _quat_to_rotation_matrix(self, quat: torch.Tensor) -> torch.Tensor:
        """Convert quaternion [w,x,y,z] to flattened rotation matrix (9 elements)."""
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        r00 = 1 - 2 * y * y - 2 * z * z
        r01 = 2 * x * y - 2 * w * z
        r02 = 2 * x * z + 2 * w * y
        r10 = 2 * x * y + 2 * w * z
        r11 = 1 - 2 * x * x - 2 * z * z
        r12 = 2 * y * z - 2 * w * x
        r20 = 2 * x * z - 2 * w * y
        r21 = 2 * y * z + 2 * w * x
        r22 = 1 - 2 * x * x - 2 * y * y
        return torch.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=-1)

    def _get_tilt_angle(self, quat: torch.Tensor) -> torch.Tensor:
        """Compute tilt angle from quaternion via rotation matrix R[2,2]."""
        rot_matrix = self._quat_to_rotation_matrix(quat)
        r22 = rot_matrix[:, 8]
        cos_tilt = torch.clamp(r22, -1.0, 1.0)
        return torch.acos(cos_tilt)

    # -----------------------------------------------------------------
    # Action processing (IDENTICAL to v1)
    # -----------------------------------------------------------------

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions using L2F motor model with hover-centered parameterization."""
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

        # First-order motor dynamics (per-env alpha: Option B)
        self._rpm_state = self._rpm_state + self._motor_alpha * (target_rpm - self._rpm_state)
        self._rpm_state = self._rpm_state.clamp(self._min_rpm, self._max_rpm)

        # Thrust per motor
        thrust_per_motor = self._thrust_coef * self._rpm_state ** 2
        total_thrust = thrust_per_motor.sum(dim=-1)

        thrust_body = torch.zeros(self.num_envs, 3, device=self.device)
        thrust_body[:, 2] = total_thrust

        # Torques
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
        """Apply forces and torques to the robot."""
        self._robot.set_external_force_and_torque(
            forces=self._thrust_body,
            torques=self._torque_body,
            body_ids=self._body_id,
        )
        self._robot.write_data_to_sim()

    # -----------------------------------------------------------------
    # Observations (IDENTICAL to v1 — obs_position_clip handles wider range)
    # -----------------------------------------------------------------

    def _get_observations(self) -> dict:
        """Construct 149-dim observations: 146 hover + 3 goal-relative.

        The reference point for position error is always init_target_height
        (1.0m). For ground-start episodes this means pos_error_z ≈ -0.97
        at spawn, giving the policy a clear signal that it's below target.
        """
        cfg = self.cfg

        pos_w = self._robot.data.root_pos_w
        quat_w = self._robot.data.root_quat_w
        lin_vel_w = self._robot.data.root_lin_vel_w
        ang_vel_b = self._robot.data.root_ang_vel_b

        # Position relative to spawn reference (fixed at target height)
        spawn_pos = self._terrain.env_origins.clone()
        spawn_pos[:, 2] += cfg.init_target_height
        pos_error = pos_w - spawn_pos
        pos_error_clipped = pos_error.clamp(-cfg.obs_position_clip, cfg.obs_position_clip)

        lin_vel_clipped = lin_vel_w.clamp(-cfg.obs_velocity_clip, cfg.obs_velocity_clip)
        rot_matrix = self._quat_to_rotation_matrix(quat_w)
        action_history_flat = self._action_history.view(self.num_envs, -1)

        goal_relative = self._goal_pos - pos_w
        goal_relative_clipped = goal_relative.clamp(-cfg.obs_position_clip, cfg.obs_position_clip)

        obs = torch.cat([
            pos_error_clipped,       # 3
            rot_matrix,              # 9
            lin_vel_clipped,         # 3
            ang_vel_b,               # 3
            action_history_flat,     # 128
            goal_relative_clipped,   # 3
        ], dim=-1)

        return {"policy": obs}

    # -----------------------------------------------------------------
    # Rewards (V2: adds takeoff reward blending)
    # -----------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        """Compute reward with takeoff/hover blending for ground-start episodes.

        During the grace period (ground-start episodes), the reward is:
            takeoff_altitude + takeoff_orientation + nav_progress

        After the grace period (or for mid-air episodes from the start):
            Standard v1 hover + navigation rewards
        """
        cfg = self.cfg

        # Get state
        pos_w = self._robot.data.root_pos_w
        quat = self._robot.data.root_quat_w
        lin_vel = self._robot.data.root_lin_vel_w
        ang_vel = self._robot.data.root_ang_vel_b

        # Current height above ground plane
        height_above_ground = pos_w[:, 2] - self._terrain.env_origins[:, 2]

        # =================================================================
        # HOVER STABILITY COSTS (same as v1)
        # =================================================================
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

        # =================================================================
        # NAVIGATION REWARDS (same as v1)
        # =================================================================
        delta_xy = pos_w[:, :2] - self._goal_pos[:, :2]
        dist_xy = torch.norm(delta_xy, dim=-1)

        # Gate hover reward by distance to goal
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

        low_margin = torch.relu(cfg.nav_low_height_penalty_floor - height_above_ground)
        low_height_penalty = -cfg.nav_low_height_penalty_weight * (low_margin ** 2)

        # Standard reward (used for mid-air episodes and post-grace-period)
        standard_reward = (
            hover_reward + progress_reward + braking_reward +
            height_tracking_reward + height_recovery_reward +
            speed_penalty + low_height_penalty + reach_bonus
        )

        # =================================================================
        # V2 NEW: TAKEOFF REWARD (during grace period only)
        # =================================================================
        # Dense reward for:
        # 1. Altitude gain (positive height change) — encourages climbing
        # 2. Staying upright — prevents flipping during motor spin-up
        # 3. XY progress toward goal — gentle nudge even during takeoff

        in_grace = (self._grace_period_remaining > 0).float()

        # Altitude gain reward: reward per-step height gain, capped
        height_gain = height_above_ground - self._prev_height_above_ground
        takeoff_altitude_r = cfg.takeoff_altitude_reward_weight * height_gain.clamp(-0.02, 0.05)

        # Orientation bonus: quat[:, 0]² ≈ 1.0 when perfectly upright
        takeoff_orientation_r = cfg.takeoff_orientation_bonus_weight * (quat[:, 0] ** 2)

        # XY progress toward goal (same direction signal, possibly different weight)
        takeoff_progress_r = cfg.takeoff_progress_weight * progress.clamp(-0.02, 0.05)

        takeoff_reward = takeoff_altitude_r + takeoff_orientation_r + takeoff_progress_r

        # =================================================================
        # BLEND: grace period → takeoff reward | otherwise → standard reward
        # =================================================================
        reward = in_grace * takeoff_reward + (1.0 - in_grace) * standard_reward

        # =================================================================
        # V2 ANTI-FORGETTING: Amplify GS reward to prevent catastrophic
        # forgetting. Without this, mid-air gradients overwhelm GS signal
        # and the policy loses takeoff ability after ~60 iterations.
        # =================================================================
        if cfg.ground_start_reward_multiplier != 1.0:
            gs_mask = self._is_ground_start.float()  # (num_envs,)
            # GS envs get multiplied reward, mid-air envs are unchanged
            reward = reward * (1.0 + (cfg.ground_start_reward_multiplier - 1.0) * gs_mask)

        # Update previous height for next step's delta
        self._prev_height_above_ground = height_above_ground.clone()

        # =================================================================
        # Track statistics
        # =================================================================
        self._episode_sums["height_cost"] += height_cost
        self._episode_sums["orientation_cost"] += orientation_cost
        self._episode_sums["xy_velocity_cost"] += xy_velocity_cost
        self._episode_sums["z_velocity_cost"] += z_velocity_cost
        self._episode_sums["angular_velocity_cost"] += angular_velocity_cost
        self._episode_sums["action_cost"] += action_cost
        self._episode_sums["hover_reward"] += hover_reward * (1.0 - in_grace)
        self._episode_sums["progress_reward"] += progress_reward
        self._episode_sums["braking_reward"] += braking_reward
        self._episode_sums["speed_penalty"] += speed_penalty * (1.0 - in_grace)
        self._episode_sums["reach_bonus"] += reach_bonus
        self._episode_sums["takeoff_reward"] += takeoff_reward * in_grace
        self._episode_sums["total_reward"] += reward
        self._episode_sums["goal_reached"] += just_reached.float()
        self._episode_sums["final_distance"] = dist_xy

        return reward

    # -----------------------------------------------------------------
    # Termination (V2: grace period exemption)
    # -----------------------------------------------------------------

    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Check termination with grace period for ground-start episodes.

        During the grace period:
        - Low-height termination is SUPPRESSED (soft and hard)
        - Low-height violation counter is held at 0
        - All other termination conditions remain active
        """
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        cfg = self.cfg

        pos_w = self._robot.data.root_pos_w
        quat = self._robot.data.root_quat_w
        lin_vel = self._robot.data.root_lin_vel_w
        ang_vel = self._robot.data.root_ang_vel_b

        # Grace period mask (before decrementing)
        in_grace = self._grace_period_remaining > 0

        # XY exceeded
        xy_offset = pos_w[:, :2] - self._terrain.env_origins[:, :2]
        xy_exceeded = torch.norm(xy_offset, dim=-1) > cfg.term_xy_threshold

        # =================================================================
        # HEIGHT CHECK (persistence-based, V2: grace period exemption)
        # =================================================================
        height = pos_w[:, 2] - self._terrain.env_origins[:, 2]

        too_low_hard = height < cfg.term_z_hard_min
        too_high_hard = height > cfg.term_z_hard_max

        too_low_soft = height < cfg.term_z_soft_min
        too_high_soft = height > cfg.term_z_soft_max

        # V2: During grace period, suppress low-height violations entirely
        # This prevents the drone from being terminated before it has time
        # to spin up motors and take off.
        self._height_low_violation_counter = torch.where(
            in_grace,
            torch.zeros_like(self._height_low_violation_counter),  # Reset during grace
            torch.where(
                too_low_soft,
                self._height_low_violation_counter + 1,
                torch.zeros_like(self._height_low_violation_counter)
            )
        )

        self._height_high_violation_counter = torch.where(
            too_high_soft,
            self._height_high_violation_counter + 1,
            torch.zeros_like(self._height_high_violation_counter)
        )

        too_low_persistence = self._height_low_violation_counter >= cfg.term_z_persistence_steps
        too_high_persistence = self._height_high_violation_counter >= cfg.term_z_persistence_steps

        # V2: Suppress hard low-height termination during grace period
        too_low = (too_low_hard & ~in_grace) | too_low_persistence
        too_high = too_high_hard | too_high_persistence

        # =================================================================
        # TILT CHECK (persistence-based, same as v1)
        # =================================================================
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

        # =================================================================
        # LINEAR VELOCITY CHECK (persistence-based, same as v1)
        # =================================================================
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

        # =================================================================
        # ANGULAR VELOCITY CHECK (persistence-based, same as v1)
        # =================================================================
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

        # =================================================================
        # COMBINE
        # =================================================================
        safety_terminated = xy_exceeded | too_low | too_high | too_tilted | lin_vel_exceeded | ang_vel_exceeded
        goal_terminated = self._goal_reached
        terminated = safety_terminated | goal_terminated

        # Update termination counters
        self._term_counters["xy_exceeded"] += xy_exceeded.sum().item()
        self._term_counters["too_low"] += too_low.sum().item()
        self._term_counters["too_high"] += too_high.sum().item()
        self._term_counters["too_tilted"] += too_tilted.sum().item()
        self._term_counters["lin_vel_exceeded"] += lin_vel_exceeded.sum().item()
        self._term_counters["ang_vel_exceeded"] += ang_vel_exceeded.sum().item()
        self._term_counters["goal_reached"] += goal_terminated.sum().item()
        self._term_counters["timeout"] += (time_out & ~terminated).sum().item()
        self._term_counters["total"] += (terminated | time_out).sum().item()

        # V2: Track ground-start vs mid-air goal reaches
        done_mask = terminated | time_out
        if done_mask.any():
            gs_done = done_mask & self._is_ground_start
            ma_done = done_mask & ~self._is_ground_start
            self._term_counters["ground_start_goal_reached"] += (self._goal_reached & gs_done).sum().item()
            self._term_counters["midair_goal_reached"] += (self._goal_reached & ma_done).sum().item()

        # V2: Decrement grace period AFTER termination check
        self._grace_period_remaining = (self._grace_period_remaining - 1).clamp(min=0)

        return terminated, time_out

    # -----------------------------------------------------------------
    # Goal sampling (IDENTICAL to v1)
    # -----------------------------------------------------------------

    def _sample_goals(self, env_ids: torch.Tensor):
        """Sample goal positions with min/max distance enforcement."""
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

    # -----------------------------------------------------------------
    # Reset (V2: ground-start spawning)
    # -----------------------------------------------------------------

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specified environments with ground-start support.

        For each resetting environment, independently decides whether to
        apply a ground-start (motors off, on the ground) or a standard
        mid-air spawn (same as v1).
        """
        if env_ids is None or len(env_ids) == 0:
            return

        # Record episode lengths before reset
        if len(env_ids) > 0:
            ep_lengths = self.episode_length_buf[env_ids].cpu().tolist()
            self._episode_lengths.extend(ep_lengths)
            if len(self._episode_lengths) > self._max_episode_buffer:
                self._episode_lengths = self._episode_lengths[-self._max_episode_buffer:]

        # Log stats before reset
        if len(env_ids) > 0 and hasattr(self, '_episode_sums'):
            extras = {}
            for key, values in self._episode_sums.items():
                if key in ["goal_reached", "final_distance"]:
                    extras[f"Episode/{key}"] = torch.mean(values[env_ids]).item()
                else:
                    avg = torch.mean(values[env_ids]).item()
                    steps = self.episode_length_buf[env_ids].float().mean().item()
                    if steps > 0:
                        extras[f"Episode/{key}"] = avg / steps

            reach_count = self._goal_reached[env_ids].float().sum().item()
            extras["Episode/reach_rate"] = reach_count / len(env_ids)

            # V2: Ground-start episode fraction
            gs_count = self._is_ground_start[env_ids].float().sum().item()
            extras["Episode/ground_start_frac"] = gs_count / len(env_ids)

            self.extras["log"] = extras

        # Reset robot
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        n = len(env_ids)
        cfg = self.cfg

        # Sample new goals FIRST
        self._sample_goals(env_ids)
        self._goal_reached[env_ids] = False

        # =================================================================
        # V2: Decide ground-start vs mid-air for each env
        # =================================================================
        ground_start_mask = torch.rand(n, device=self.device) < self._current_ground_start_prob
        self._is_ground_start[env_ids] = ground_start_mask
        self._term_counters["ground_start_episodes"] += ground_start_mask.sum().item()

        # Guidance: spawn perfectly at target (mid-air only)
        guidance_mask = torch.rand(n, device=self.device) < cfg.init_guidance_probability
        # Ground-start envs never get guidance (they start on ground)
        guidance_mask = guidance_mask & ~ground_start_mask

        # =================================================================
        # Standard mid-air position initialization (same as v1)
        # =================================================================
        pos = torch.zeros(n, 3, device=self.device)

        pos[~guidance_mask & ~ground_start_mask, 0] = torch.empty(
            (~guidance_mask & ~ground_start_mask).sum(), device=self.device
        ).uniform_(-cfg.init_max_xy_offset, cfg.init_max_xy_offset)
        pos[~guidance_mask & ~ground_start_mask, 1] = torch.empty(
            (~guidance_mask & ~ground_start_mask).sum(), device=self.device
        ).uniform_(-cfg.init_max_xy_offset, cfg.init_max_xy_offset)

        height_offset = torch.empty(n, device=self.device).uniform_(
            cfg.init_height_offset_min, cfg.init_height_offset_max
        )
        height_offset[guidance_mask] = 0
        height_offset[ground_start_mask] = 0  # Will be overridden below

        pos[:, 2] = cfg.init_target_height + height_offset
        pos = pos + self._terrain.env_origins[env_ids]

        # =================================================================
        # V2: Override position for ground-start envs
        # =================================================================
        if ground_start_mask.any():
            gs_idx = ground_start_mask.nonzero(as_tuple=True)[0]
            gs_origins = self._terrain.env_origins[env_ids[gs_idx]]
            pos[gs_idx, 0] = gs_origins[:, 0]  # Centered XY
            pos[gs_idx, 1] = gs_origins[:, 1]
            pos[gs_idx, 2] = gs_origins[:, 2] + cfg.ground_start_height

        # =================================================================
        # Orientation
        # =================================================================
        quat = torch.zeros(n, 4, device=self.device)
        quat[:, 0] = 1.0  # Identity quaternion

        if cfg.init_max_angle > 0:
            axis = torch.randn(n, 3, device=self.device)
            axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-8)
            angle = torch.empty(n, device=self.device).uniform_(0, cfg.init_max_angle)
            angle[guidance_mask] = 0
            # V2: Ground-start envs get zero initial angle (upright on ground)
            angle[ground_start_mask] = 0

            half_angle = angle / 2
            quat[:, 0] = torch.cos(half_angle)
            quat[:, 1:] = axis * torch.sin(half_angle).unsqueeze(-1)
            quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)

        # =================================================================
        # Velocities
        # =================================================================
        lin_vel = torch.empty(n, 3, device=self.device).uniform_(
            -cfg.init_max_linear_velocity, cfg.init_max_linear_velocity
        )
        lin_vel[guidance_mask] = 0
        # V2: Ground-start envs have zero velocity
        lin_vel[ground_start_mask] = 0

        ang_vel = torch.empty(n, 3, device=self.device).uniform_(
            -cfg.init_max_angular_velocity, cfg.init_max_angular_velocity
        )
        ang_vel[guidance_mask] = 0
        # V2: Ground-start envs have zero angular velocity
        ang_vel[ground_start_mask] = 0

        # Write to sim
        root_pose = torch.cat([pos, quat], dim=-1)
        root_vel = torch.cat([lin_vel, ang_vel], dim=-1)

        self._robot.write_root_pose_to_sim(root_pose, env_ids)
        self._robot.write_root_velocity_to_sim(root_vel, env_ids)

        # =================================================================
        # Motor state initialization
        # =================================================================
        # Mid-air: hover RPM (matches v1)
        self._rpm_state[env_ids] = self._hover_rpm

        # V2: Ground-start: motors OFF (0 RPM)
        # Motor dynamics will naturally ramp up from 0 as the policy
        # commands thrust, matching real-world motor spin-up behavior.
        if ground_start_mask.any():
            gs_env_ids = env_ids[ground_start_mask]
            self._rpm_state[gs_env_ids] = 0.0

        # Option B: Set per-env motor alpha
        # Ground-start envs get faster motor response (τ_gs=0.05s)
        # Mid-air envs keep default (τ=0.15s)
        self._motor_alpha[env_ids] = self._motor_alpha_default
        if ground_start_mask.any():
            gs_env_ids = env_ids[ground_start_mask]
            self._motor_alpha[gs_env_ids] = self._motor_alpha_ground

        # =================================================================
        # Action history initialization
        # =================================================================
        # Mid-air: hover action (same as v1)
        self._action_history[env_ids] = self._hover_action
        self._actions[env_ids] = self._hover_action

        # V2: Ground-start: set action history to 0.0 (matching firmware)
        # Real firmware (rl_tools_init) zeros the action history buffer.
        # With hover-centered actions, 0.0 = hover thrust command (but
        # RPM state is still 0 — the action history records what was
        # *commanded*, not what the motors actually did).
        if ground_start_mask.any():
            gs_env_ids = env_ids[ground_start_mask]
            self._action_history[gs_env_ids] = 0.0
            self._actions[gs_env_ids] = 0.0

        # =================================================================
        # Navigation state initialization
        # =================================================================
        # Previous XY distance to goal (computed from actual position)
        delta_xy = self._goal_pos[env_ids, :2] - pos[:, :2]
        self._prev_dist_xy[env_ids] = torch.norm(delta_xy, dim=-1)

        self._prev_speed[env_ids] = 0.0

        # Height below target (for height recovery reward)
        target_h = self._terrain.env_origins[env_ids, 2] + cfg.init_target_height
        height_below = torch.relu(target_h - pos[:, 2])
        self._prev_height_below_target[env_ids] = height_below

        # V2: Previous height above ground (for takeoff altitude reward)
        self._prev_height_above_ground[env_ids] = pos[:, 2] - self._terrain.env_origins[env_ids, 2]

        # =================================================================
        # V2: Grace period initialization
        # =================================================================
        self._grace_period_remaining[env_ids] = torch.where(
            ground_start_mask,
            torch.full((n,), cfg.ground_start_grace_steps, dtype=torch.int32, device=self.device),
            torch.zeros(n, dtype=torch.int32, device=self.device)
        )

        # =================================================================
        # Disturbances
        # =================================================================
        if cfg.enable_disturbance:
            self._disturbance_force[env_ids] = torch.randn(n, 3, device=self.device) * cfg.disturbance_force_std
            self._disturbance_torque[env_ids] = torch.randn(n, 3, device=self.device) * cfg.disturbance_torque_std

        # =================================================================
        # Reset persistence counters
        # =================================================================
        self._tilt_violation_counter[env_ids] = 0
        self._angvel_violation_counter[env_ids] = 0
        self._linvel_violation_counter[env_ids] = 0
        self._height_low_violation_counter[env_ids] = 0
        self._height_high_violation_counter[env_ids] = 0

        # Reset episode stats
        for key in self._episode_sums:
            self._episode_sums[key][env_ids] = 0.0

    # -----------------------------------------------------------------
    # Diagnostics helpers
    # -----------------------------------------------------------------

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
            if reason not in ("total", "ground_start_episodes", "ground_start_goal_reached", "midair_goal_reached")
        }

        feasibility = {
            "p50_ratio": p50 / H if H > 0 else 0,
            "p90_ratio": p90 / H if H > 0 else 0,
            "too_tilted_pct": term_pcts.get("too_tilted", 0),
            "timeout_pct": term_pcts.get("timeout", 0),
            "is_feasible": (
                (p50 / H > 0.2 if H > 0 else False) and
                (p90 / H > 0.6 if H > 0 else False) and
                term_pcts.get("too_tilted", 100) < 40
            ),
        }

        return {
            "episode_lengths": ep_stats,
            "termination_pcts": term_pcts,
            "feasibility": feasibility,
        }

    # -----------------------------------------------------------------
    # Debug visualization (same as v1)
    # -----------------------------------------------------------------

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
            self._goal_markers.visualize(self._goal_pos)


# ==============================================================================
# L2F-Compatible Actor Network (IDENTICAL to v1)
# ==============================================================================

class L2FActorNetwork(nn.Module):
    """Actor network: 149 -> 64 (tanh) -> 64 (tanh) -> 4 (tanh)."""

    HOVER_ACTION = 2.0 * math.sqrt(0.027 * 9.81 / (4 * 3.16e-10)) / 21702.0 - 1.0

    def __init__(self, obs_dim: int = 149, hidden_dim: int = 64, action_dim: int = 4, init_std: float = 0.5):
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
    """Critic network: 149 -> 64 (tanh) -> 64 (tanh) -> 1."""

    def __init__(self, obs_dim: int = 149, hidden_dim: int = 64):
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
# PPO Agent (IDENTICAL to v1, with added load_pretrained method)
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
    """PPO Agent with L2F-compatible architecture and observation normalization."""

    def __init__(
        self,
        obs_dim: int = 149,
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

    def save(self, path: str, iteration: int, best_reward: float, stage_meta: dict | None = None):
        data = {
            "iteration": iteration,
            "best_reward": best_reward,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "log_std": self.actor.log_std.data,
            "optimizer": self.optimizer.state_dict(),
            "obs_mean": self.obs_normalizer.mean,
            "obs_var": self.obs_normalizer.var,
            "obs_count": self.obs_normalizer.count,
        }
        if stage_meta is not None:
            data["stage_meta"] = stage_meta
        torch.save(data, path)

    def load(self, path: str):
        """Load full checkpoint (weights + obs normalizer + optimizer).

        Returns (iteration, best_reward, stage_meta_or_None).
        """
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
        stage_meta = checkpoint.get("stage_meta", None)
        return checkpoint.get("iteration", 0), checkpoint.get("best_reward", 0.0), stage_meta

    def load_pretrained(self, path: str):
        """Load actor/critic weights from v1 checkpoint WITHOUT loading obs normalizer.

        This is for fine-tuning: the policy weights provide a warm start,
        but the obs normalizer is reset because v2 has a different observation
        distribution (ground-start observations were never seen during v1 training).
        The optimizer state is also reset.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor.log_std.data = checkpoint["log_std"]
        # Deliberately NOT loading obs normalizer or optimizer
        print(f"[Pretrained] Loaded actor/critic weights from: {path}")
        print(f"[Pretrained] Obs normalizer RESET (will re-learn for v2 distribution)")
        print(f"[Pretrained] Optimizer RESET (fresh Adam state)")
        return checkpoint.get("iteration", 0), checkpoint.get("best_reward", 0.0)


# ==============================================================================
# Training Utilities
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


# ==============================================================================
# Sanity Test (V2: extended for ground-start)
# ==============================================================================

def sanity_test(env: CrazyfliePointNavV2Env, num_steps: int = 200):
    """Run sanity tests including ground-start behavior verification."""
    print("\n" + "=" * 60)
    print("SANITY TEST MODE (V2 - Ground-Start)")
    print("=" * 60)

    # Test 1: Reset
    print("[Test 1] Reset environment...")
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    print(f"  OK: Reset successful, obs shape: {obs.shape}")

    # Test 2: Observation shape
    print("[Test 2] Check observation shape...")
    assert obs.shape[-1] == env.cfg.observation_space, \
        f"Expected {env.cfg.observation_space} dims, got {obs.shape[-1]}"
    print(f"  OK: {obs.shape[-1]} dims")

    # Test 3: Goal distances
    print("[Test 3] Check goal distances...")
    spawn_pos = env._terrain.env_origins.clone()
    spawn_pos[:, 2] += env.cfg.init_target_height
    goal_dist = torch.norm(env._goal_pos - spawn_pos, dim=-1)
    print(f"  OK: Goal distances in [{goal_dist.min().item():.3f}, {goal_dist.max().item():.3f}] m")

    # Test 4: Ground-start fraction
    print("[Test 4] Check ground-start fraction...")
    gs_count = env._is_ground_start.float().sum().item()
    gs_frac = gs_count / env.num_envs
    print(f"  Ground-start envs: {int(gs_count)}/{env.num_envs} ({gs_frac:.0%})")
    print(f"  Expected probability: {env._current_ground_start_prob:.0%}")

    # Test 5: Ground-start initial conditions
    print("[Test 5] Check ground-start initial conditions...")
    if gs_count > 0:
        gs_mask = env._is_ground_start
        gs_heights = env._robot.data.root_pos_w[gs_mask, 2] - env._terrain.env_origins[gs_mask, 2]
        gs_rpm = env._rpm_state[gs_mask]
        gs_vel = env._robot.data.root_lin_vel_w[gs_mask]
        gs_grace = env._grace_period_remaining[gs_mask]

        print(f"  Height: mean={gs_heights.mean().item():.4f} m (expect ~{env.cfg.ground_start_height})")
        print(f"  RPM: mean={gs_rpm.mean().item():.1f} (expect 0)")
        print(f"  Velocity: mean_mag={torch.norm(gs_vel, dim=-1).mean().item():.4f} m/s (expect ~0)")
        print(f"  Grace remaining: mean={gs_grace.float().mean().item():.0f} steps (expect {env.cfg.ground_start_grace_steps})")

        # Verify initial condition constraints
        assert gs_heights.mean().item() < 0.1, "Ground-start height too high"
        assert gs_rpm.mean().item() < 1.0, "Ground-start RPM not zero"
        print(f"  OK: Ground-start initial conditions verified")
    else:
        print(f"  SKIP: No ground-start envs (probability may be 0)")

    # Test 6: Mid-air initial conditions
    print("[Test 6] Check mid-air initial conditions...")
    ma_mask = ~env._is_ground_start
    if ma_mask.any():
        ma_heights = env._robot.data.root_pos_w[ma_mask, 2] - env._terrain.env_origins[ma_mask, 2]
        ma_rpm = env._rpm_state[ma_mask]
        print(f"  Height: mean={ma_heights.mean().item():.3f} m (expect ~{env.cfg.init_target_height})")
        print(f"  RPM: mean={ma_rpm.mean().item():.0f} (expect ~{env._hover_rpm:.0f})")
        print(f"  OK: Mid-air initial conditions verified")

    # Test 7: Run random steps
    print(f"[Test 7] Run {num_steps} random steps...")
    total_reward = 0.0
    grace_active_steps = 0
    for step in range(num_steps):
        action = torch.rand(env.num_envs, 4, device=env.device) * 2 - 1
        obs_dict, reward, terminated, truncated, info = env.step(action)
        assert torch.isfinite(reward).all(), f"Non-finite reward at step {step}"
        total_reward += reward.mean().item()
        grace_active_steps += (env._grace_period_remaining > 0).sum().item()
    avg_reward = total_reward / num_steps
    print(f"  OK: {num_steps} steps, avg reward: {avg_reward:.3f}")
    print(f"  Grace period active in {grace_active_steps} env-steps")

    # Test 8: Verify pos error clip handles ground-start
    print("[Test 8] Check observation clips...")
    obs = obs_dict["policy"]
    pos_error = obs[:, :3]
    goal_rel = obs[:, -3:]
    assert (pos_error.abs() <= env.cfg.obs_position_clip + 0.01).all(), "Position error not clipped"
    assert (goal_rel.abs() <= env.cfg.obs_position_clip + 0.01).all(), "Goal relative not clipped"
    print(f"  OK: All observations within clip bounds (±{env.cfg.obs_position_clip})")

    print("\n" + "=" * 60)
    print("ALL SANITY TESTS PASSED (V2)")
    print("=" * 60 + "\n")


# ==============================================================================
# Training Loop (V2: multi-stage automated pipeline)
# ==============================================================================

def train(env: CrazyfliePointNavV2Env, agent: L2FPPOAgent, args):
    """Main training loop with automated multi-stage ground-start pipeline.

    Stages (when staged_training=True):
      Stage 1 — Mid-air only (GS prob=0%). Builds navigation skill.
      Stage 2 — Ground-start intro (GS prob=25%). Loads Stage 1 best as
                pretrained (resets obs normalizer). Builds takeoff skill.
      Stage 3 — Anti-forgetting. Same as Stage 2 but auto-reloads
                best_gs_model.pt if GS reach regresses.

    When staged_training=False, behaves like the original manual training.
    """
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints_pointnav_v2")
    os.makedirs(checkpoint_dir, exist_ok=True)

    steps_per_rollout = 256
    num_envs = env.num_envs

    best_reward = float("-inf")
    best_reach_rate = 0.0
    best_gs_reach_rate = 0.0  # Track best ground-start reach rate separately

    cfg = env.cfg

    # =====================================================================
    # STAGED TRAINING STATE
    # =====================================================================
    use_staged = cfg.staged_training and not getattr(args, "no_staged", False)
    current_stage = args.stage if args.stage else (1 if use_staged else 2)
    stage_iters_in_current = 0        # Iterations spent in current stage
    stage_transitions = {}            # {stage_num: iteration when entered}
    peak_gs_reach = 0.0               # Peak GS reach rate in Stage 3
    reload_count = 0                  # Number of anti-forgetting reloads
    last_reload_iter = -1000          # Last iteration with a reload
    transition_check_passes = 0       # Consecutive diagnostic checks above threshold

    print(f"\n{'='*60}")
    print("Starting L2F PointNav V2 PPO Training (Ground-Start)")
    print(f"{'='*60}")
    print(f"  Environments:       {num_envs}")
    print(f"  Max iterations:     {args.max_iterations}")
    print(f"  Steps per rollout:  {steps_per_rollout}")
    print(f"  Total batch size:   {steps_per_rollout * num_envs}")
    print(f"  Observation dim:    {cfg.observation_space}")
    print(f"  Action dim:         {cfg.action_space}")
    print(f"  Goal distance:      [{cfg.goal_min_distance}, {cfg.goal_max_distance}] m")
    if use_staged:
        print(f"  Staged training:    ON (starting at Stage {current_stage})")
        print(f"  Stage 1→2:          mid-air reach > {cfg.stage1_transition_reach:.0%} for {cfg.stage1_transition_window} checks")
        print(f"  Stage 2→3:          GS reach > {cfg.stage2_transition_gs_reach:.0%} for {cfg.stage2_transition_window} checks")
        print(f"  Anti-forgetting:    tolerance={cfg.stage3_forgetting_tolerance:.0%}, max reloads={cfg.stage3_reload_max}")
    else:
        print(f"  Staged training:    OFF (manual mode)")
        print(f"  Ground-start prob:  {env.get_ground_start_probability():.0%}")
        print(f"  Curriculum:         {'ON' if cfg.ground_start_curriculum and not args.no_curriculum else 'OFF'}")
    if args.pretrained:
        print(f"  Pretrained from:    {args.pretrained}")
    print(f"{'='*60}\n")

    # Verify quaternion ordering at first reset
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    quat = env._robot.data.root_quat_w
    print(f"[Quaternion Check] At reset: quat[:,0].mean={quat[:,0].mean():.4f}")

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
            start_iteration, best_reward, _stage_meta = agent.load(ckpt_path)
            print(f"[Resume] Starting from iteration {start_iteration}, best_reward={best_reward:.2f}")

            # Restore stage metadata if available
            if _stage_meta and use_staged:
                current_stage = _stage_meta.get("stage", current_stage)
                stage_transitions = _stage_meta.get("transitions", {})
                peak_gs_reach = _stage_meta.get("peak_gs_reach", 0.0)
                reload_count = _stage_meta.get("reload_count", 0)
                best_gs_reach_rate = _stage_meta.get("best_gs_reach_rate", 0.0)
                print(f"[Resume] Restored stage={current_stage}, peak_gs={peak_gs_reach:.1%}, "
                      f"reloads={reload_count}")
        else:
            print(f"\n[Resume] No checkpoint found at {ckpt_path}, starting fresh")

    # Apply initial stage settings
    if use_staged:
        if current_stage == 1:
            env.set_ground_start_probability(0.0)
            print(f"[Stage 1] Mid-air only training (GS prob = 0%)")
        elif current_stage >= 2:
            env.set_ground_start_probability(cfg.ground_start_probability)
            print(f"[Stage {current_stage}] GS probability = {cfg.ground_start_probability:.0%}")

    def _make_stage_meta():
        """Build stage metadata dict for checkpoint saving."""
        return {
            "stage": current_stage,
            "transitions": stage_transitions,
            "peak_gs_reach": peak_gs_reach,
            "reload_count": reload_count,
            "best_gs_reach_rate": best_gs_reach_rate,
        }

    for iteration in range(start_iteration, start_iteration + args.max_iterations):
        # =================================================================
        # V2: Stage-aware ground-start probability
        # =================================================================
        if use_staged:
            # Staged training sets GS probability based on current stage
            if current_stage == 1:
                env.set_ground_start_probability(0.0)
            elif current_stage >= 2:
                env.set_ground_start_probability(cfg.ground_start_probability)
        elif cfg.ground_start_curriculum and not args.no_curriculum:
            # Legacy curriculum mode (when staged_training=False)
            gs_reach_rate = env.get_ground_start_reach_rate()
            linear_progress = min(iteration / cfg.ground_start_curriculum_iters, 1.0)
            linear_prob = cfg.ground_start_curriculum_start + \
                   (cfg.ground_start_curriculum_end - cfg.ground_start_curriculum_start) * linear_progress
            current_prob = env.get_ground_start_probability()
            if gs_reach_rate >= cfg.ground_start_curriculum_gate or iteration < 20:
                env.set_ground_start_probability(linear_prob)
            else:
                env.set_ground_start_probability(min(current_prob, linear_prob))

        # Override from CLI if specified
        if args.ground_start_prob is not None:
            env.set_ground_start_probability(args.ground_start_prob)

        # Collect rollout
        obs_buffer = []
        action_buffer = []
        log_prob_buffer = []
        value_buffer = []
        reward_buffer = []
        done_buffer = []

        episode_rewards = torch.zeros(num_envs, device=env.device)
        reach_count = 0
        episode_count = 0

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

            if "log" in env.extras and "Episode/reach_rate" in env.extras["log"]:
                reach_count += env.extras["log"]["Episode/reach_rate"] * done.sum().item()
                episode_count += done.sum().item()

            obs = next_obs

        # Stack and compute GAE
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

        # Stats
        mean_reward = episode_rewards.mean().item() / steps_per_rollout
        mean_return = returns_flat.mean().item()
        reach_rate = reach_count / max(episode_count, 1) if episode_count > 0 else 0.0

        is_best = mean_reward > best_reward
        if is_best:
            best_reward = mean_reward
            agent.save(os.path.join(checkpoint_dir, "best_model.pt"), iteration, best_reward,
                       stage_meta=_make_stage_meta())

        if reach_rate > best_reach_rate:
            best_reach_rate = reach_rate
            agent.save(os.path.join(checkpoint_dir, "best_reach_model.pt"), iteration, best_reward,
                       stage_meta=_make_stage_meta())

        # Save best GS-specific reach rate checkpoint
        gs_rr_current = env.get_ground_start_reach_rate()
        if gs_rr_current > best_gs_reach_rate and iteration >= 10:
            best_gs_reach_rate = gs_rr_current
            agent.save(os.path.join(checkpoint_dir, "best_gs_model.pt"), iteration, best_reward,
                       stage_meta=_make_stage_meta())

        # Log
        if iteration % 10 == 0 or is_best:
            std = torch.exp(agent.actor.log_std).mean().item()
            gs_prob = env.get_ground_start_probability()
            gs_rr = env.get_ground_start_reach_rate()
            star = " *BEST*" if is_best else ""
            stage_tag = f" S{current_stage}" if use_staged else ""
            gate_status = ""
            if not use_staged and cfg.ground_start_curriculum and not args.no_curriculum and iteration >= 20:
                if gs_rr < cfg.ground_start_curriculum_gate:
                    gate_status = " [GATED]"
            print(f"[Iter {iteration:4d}{stage_tag}] Reward: {mean_reward:8.3f} | Reach: {reach_rate*100:5.1f}% | "
                  f"Std: {std:.3f} | GS: {gs_prob:.0%} (rr:{gs_rr:.0%}){gate_status} | Loss: {loss:.4f}{star}")

        # Comprehensive diagnostics every 50 iterations
        if iteration > 0 and iteration % 50 == 0:
            ep_stats = env.get_episode_length_stats()
            H = steps_per_rollout
            p50 = ep_stats["p50"]
            p90 = ep_stats["p90"]

            # V2: Phase classification accounts for ground-start (short mid-air
            # episodes are GOOD, they mean fast goal reaching)
            tc = env._term_counters
            total = max(tc["total"], 1)
            gs_reach_rate = env.get_ground_start_reach_rate()
            ma_reach_pct = tc["midair_goal_reached"] / max(tc["total"] - tc["ground_start_episodes"], 1)
            if p50 < 10 and ma_reach_pct < 0.3:
                phase = "Phase 0: NOT LEARNABLE"
            elif ma_reach_pct < 0.5:
                phase = "Phase 1: Learning mid-air"
            elif gs_reach_rate < 0.15:
                phase = "Phase 2: Mid-air solid, learning takeoff"
            elif gs_reach_rate < 0.40:
                phase = "Phase 3: Takeoff improving"
            else:
                phase = "Phase 4: Healthy training"

            print(f"\n  === DIAGNOSTICS (H={H}) ===")
            if use_staged:
                print(f"  Training Stage: {current_stage} (iters in stage: {stage_iters_in_current})")
            print(f"  Episode Length: mean={ep_stats['mean']:.1f} p50={p50:.1f} p90={p90:.1f} (n={ep_stats['count']})")
            print(f"  Phase: {phase}")

            print(f"  Terminations: xy:{tc['xy_exceeded']/total*100:.1f}% low:{tc['too_low']/total*100:.1f}% "
                  f"high:{tc['too_high']/total*100:.1f}% tilt:{tc['too_tilted']/total*100:.1f}% "
                  f"linvel:{tc['lin_vel_exceeded']/total*100:.1f}% angvel:{tc['ang_vel_exceeded']/total*100:.1f}% "
                  f"goal:{tc['goal_reached']/total*100:.1f}% timeout:{tc['timeout']/total*100:.1f}%")

            # V2: Ground-start specific stats
            gs_episodes = max(tc["ground_start_episodes"], 1)
            gs_reach = tc["ground_start_goal_reached"]
            ma_reach = tc["midair_goal_reached"]
            print(f"  Ground-Start: {tc['ground_start_episodes']} episodes, "
                  f"reach rate: {gs_reach/max(gs_episodes,1)*100:.1f}% | "
                  f"Mid-air reach rate: {ma_reach/max(total-gs_episodes,1)*100:.1f}%")
            print(f"  Current GS probability: {env.get_ground_start_probability():.0%}")

            # Actionable guidance
            max_term = max(tc["xy_exceeded"], tc["too_low"], tc["too_high"],
                          tc["too_tilted"], tc["lin_vel_exceeded"], tc["ang_vel_exceeded"])
            if max_term / total > 0.4:
                dominant = max(
                    [(k, v) for k, v in tc.items()
                     if k not in ("total", "ground_start_episodes", "ground_start_goal_reached", "midair_goal_reached")],
                    key=lambda x: x[1]
                )[0]
                print(f"  WARNING: '{dominant}' dominates ({max_term/total*100:.0f}%). Consider relaxing threshold.")

            print(f"  ==========================\n")

            # =============================================================
            # STAGE TRANSITION CHECKS (every 50 iterations)
            # =============================================================
            if use_staged:
                if current_stage == 1:
                    # Stage 1 → 2: Mid-air reach above threshold?
                    if (ma_reach_pct >= cfg.stage1_transition_reach
                            and stage_iters_in_current >= cfg.stage1_min_iterations):
                        transition_check_passes += 1
                        print(f"  [Stage 1] Transition check {transition_check_passes}/"
                              f"{cfg.stage1_transition_window} "
                              f"(mid-air reach {ma_reach_pct:.0%} >= {cfg.stage1_transition_reach:.0%})")
                        if transition_check_passes >= cfg.stage1_transition_window:
                            # === TRANSITION: Stage 1 → 2 ===
                            stage1_ckpt = os.path.join(checkpoint_dir, "stage1_best.pt")
                            agent.save(stage1_ckpt, iteration, best_reward,
                                       stage_meta=_make_stage_meta())
                            print(f"\n{'='*60}")
                            print(f">>> STAGE TRANSITION: 1 → 2  (iteration {iteration})")
                            print(f">>> Saved Stage 1 best: {stage1_ckpt}")
                            print(f">>> Loading as pretrained (resetting obs normalizer + optimizer)")
                            print(f">>> GS probability: 0% → {cfg.ground_start_probability:.0%}")
                            print(f"{'='*60}\n")
                            agent.load_pretrained(stage1_ckpt)
                            current_stage = 2
                            stage_transitions[2] = iteration
                            transition_check_passes = 0
                            stage_iters_in_current = 0
                            best_gs_reach_rate = 0.0
                            best_reward = float("-inf")  # Reset best since obs distribution changes
                            env.set_ground_start_probability(cfg.ground_start_probability)
                            obs_dict, _ = env.reset()
                            obs = obs_dict["policy"]
                    else:
                        transition_check_passes = 0

                elif current_stage == 2:
                    # Stage 2 → 3: GS reach above threshold?
                    if (gs_reach_rate >= cfg.stage2_transition_gs_reach
                            and stage_iters_in_current >= cfg.stage2_min_iterations):
                        transition_check_passes += 1
                        print(f"  [Stage 2] Transition check {transition_check_passes}/"
                              f"{cfg.stage2_transition_window} "
                              f"(GS reach {gs_reach_rate:.0%} >= {cfg.stage2_transition_gs_reach:.0%})")
                        if transition_check_passes >= cfg.stage2_transition_window:
                            # === TRANSITION: Stage 2 → 3 ===
                            stage2_ckpt = os.path.join(checkpoint_dir, "stage2_best.pt")
                            agent.save(stage2_ckpt, iteration, best_reward,
                                       stage_meta=_make_stage_meta())
                            print(f"\n{'='*60}")
                            print(f">>> STAGE TRANSITION: 2 → 3  (iteration {iteration})")
                            print(f">>> Anti-forgetting mode active")
                            print(f">>> Peak GS reach: {gs_reach_rate:.0%}")
                            print(f">>> Will reload best_gs_model.pt if GS drops by "
                                  f"{cfg.stage3_forgetting_tolerance:.0%}")
                            print(f"{'='*60}\n")
                            current_stage = 3
                            stage_transitions[3] = iteration
                            transition_check_passes = 0
                            stage_iters_in_current = 0
                            peak_gs_reach = gs_reach_rate
                    else:
                        transition_check_passes = 0

                elif current_stage == 3:
                    # Stage 3: Anti-forgetting monitoring
                    if gs_reach_rate > peak_gs_reach:
                        peak_gs_reach = gs_reach_rate
                        print(f"  [Stage 3] New peak GS reach: {peak_gs_reach:.0%}")
                    elif gs_reach_rate < peak_gs_reach - cfg.stage3_forgetting_tolerance:
                        can_reload = (
                            iteration - last_reload_iter >= cfg.stage3_reload_cooldown
                            and reload_count < cfg.stage3_reload_max
                        )
                        gs_ckpt = os.path.join(checkpoint_dir, "best_gs_model.pt")
                        if can_reload and os.path.exists(gs_ckpt):
                            reload_count += 1
                            last_reload_iter = iteration
                            print(f"\n{'='*60}")
                            print(f">>> ANTI-FORGETTING RELOAD #{reload_count} (iteration {iteration})")
                            print(f">>> GS reach dropped: {gs_reach_rate:.0%} < "
                                  f"peak {peak_gs_reach:.0%} - {cfg.stage3_forgetting_tolerance:.0%}")
                            print(f">>> Reloading: {gs_ckpt}")
                            print(f"{'='*60}\n")
                            _, _, _ = agent.load(gs_ckpt)
                            obs_dict, _ = env.reset()
                            obs = obs_dict["policy"]
                            # Don't reset peak — we want to track absolute peak
                        elif not can_reload:
                            if reload_count >= cfg.stage3_reload_max:
                                print(f"  [Stage 3] GS declining ({gs_reach_rate:.0%}) "
                                      f"but max reloads ({cfg.stage3_reload_max}) exhausted")
                            else:
                                remaining = cfg.stage3_reload_cooldown - (iteration - last_reload_iter)
                                print(f"  [Stage 3] GS declining ({gs_reach_rate:.0%}) "
                                      f"— cooldown: {remaining} iters remaining")

                stage_iters_in_current += 50  # Diagnostic interval

            env.clear_episode_stats()

        # Save periodically
        if iteration > 0 and iteration % args.save_interval == 0:
            agent.save(os.path.join(checkpoint_dir, f"checkpoint_{iteration}.pt"), iteration, best_reward,
                       stage_meta=_make_stage_meta())

    # Final save
    agent.save(os.path.join(checkpoint_dir, "final_model.pt"), start_iteration + args.max_iterations, best_reward,
               stage_meta=_make_stage_meta())
    print(f"\nTraining complete! Best reward: {best_reward:.3f}, Best reach rate: {best_reach_rate*100:.1f}%")
    print(f"Best GS reach rate: {best_gs_reach_rate*100:.1f}%")
    if use_staged:
        print(f"Final stage: {current_stage}, Transitions: {stage_transitions}")
        print(f"Anti-forgetting reloads: {reload_count}/{cfg.stage3_reload_max}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


# ==============================================================================
# Play Mode
# ==============================================================================

def play(env: CrazyfliePointNavV2Env, agent: L2FPPOAgent, checkpoint_path: str):
    """Run trained policy with visualization and data logging."""
    iteration, best_reward, _stage_meta = agent.load(checkpoint_path)
    print(f"\n[Play Mode] Loaded checkpoint from iteration {iteration}")
    print(f"[Play Mode] Best training reward: {best_reward:.3f}")
    print(f"[Play Mode] Ground-start probability: {env.get_ground_start_probability():.0%}")
    print("[Play Mode] Press Ctrl+C to stop\n")

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    step_count = 0
    episode_reward = 0.0
    reach_count = 0
    episode_count = 0
    gs_reach_count = 0
    gs_episode_count = 0

    # Initialize flight data logger
    logger = FlightDataLogger()

    # Create eval directory
    run_tag = int(time.time())
    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_dir = os.path.join(script_dir, "eval", "pointnav_v2", f"pointnav_v2_{run_tag}")
    os.makedirs(eval_dir, exist_ok=True)

    csv_filename = os.path.join(eval_dir, "pointnav_v2_eval_latest.csv")
    title_prefix = "Point Navigation V2 Evaluation"

    try:
        while simulation_app.is_running():
            action = agent.get_action(obs, deterministic=True)
            obs_dict, reward, terminated, truncated, info = env.step(action)
            obs = obs_dict["policy"]

            done = terminated | truncated
            episode_reward += reward.mean().item()
            step_count += 1

            logger.log_step(env, env_idx=0)

            if done.any():
                reaches = env._goal_reached[done].sum().item()
                reach_count += reaches
                episode_count += done.sum().item()

                # Track ground-start reaches separately
                gs_done = done & env._is_ground_start
                if gs_done.any():
                    gs_reach_count += env._goal_reached[gs_done].sum().item()
                    gs_episode_count += gs_done.sum().item()

            if step_count % 500 == 0:
                reach_rate = reach_count / max(episode_count, 1) * 100
                gs_rate = gs_reach_count / max(gs_episode_count, 1) * 100
                print(f"[Step {step_count:5d}] Reward: {episode_reward:.2f} | "
                      f"Reach: {reach_rate:.1f}% | GS Reach: {gs_rate:.1f}% | Saving...")
                logger.save_and_plot(csv_filename, title_prefix=title_prefix, output_dir=eval_dir)
            elif step_count % 100 == 0:
                reach_rate = reach_count / max(episode_count, 1) * 100
                print(f"[Step {step_count:5d}] Reward: {episode_reward:.2f} | Reach: {reach_rate:.1f}%")

    except KeyboardInterrupt:
        print("\n[Play Mode] Stopped by user")
        reach_rate = reach_count / max(episode_count, 1) * 100
        gs_rate = gs_reach_count / max(gs_episode_count, 1) * 100
        print(f"Final reach rate: {reach_rate:.1f}% ({reach_count}/{episode_count})")
        print(f"Ground-start reach rate: {gs_rate:.1f}% ({gs_reach_count}/{gs_episode_count})")


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    # Create config
    cfg = CrazyfliePointNavV2EnvCfg()
    cfg.scene.num_envs = args.num_envs

    # Apply CLI overrides
    if args.ground_start_prob is not None:
        cfg.ground_start_probability = args.ground_start_prob
    if args.no_curriculum:
        cfg.ground_start_curriculum = False
    if getattr(args, "no_staged", False):
        cfg.staged_training = False

    # Create environment
    env = CrazyfliePointNavV2Env(cfg)

    if args.sanity_test:
        sanity_test(env)
        env.close()
        simulation_app.close()
        return

    # Create agent
    agent = L2FPPOAgent(
        obs_dim=cfg.observation_space,
        action_dim=cfg.action_space,
        device=env.device,
        lr=args.lr,
        gamma=args.gamma,
    )

    # V2: Load pretrained weights if specified (skips Stage 1 when using staged training)
    if args.pretrained:
        if not os.path.exists(args.pretrained):
            print(f"Error: Pretrained checkpoint not found: {args.pretrained}")
            sys.exit(1)
        agent.load_pretrained(args.pretrained)
        # If staged and no explicit --stage, pretrained implies starting at Stage 2
        if cfg.staged_training and args.stage is None:
            args.stage = 2
            print(f"[Staged] Pretrained checkpoint provided — starting at Stage 2")

    if args.play:
        if args.checkpoint is None:
            checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints_pointnav_v2")
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
