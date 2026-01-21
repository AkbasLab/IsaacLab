# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""
Crazyflie Learning-to-Fly Compatible Environment

This environment implements the learning-to-fly (L2F) physics and control contract
within Isaac Lab. The key design principle is that:
- Isaac Lab handles rigid body dynamics (position, velocity integration)
- We compute thrust/torque externally to match L2F's motor model exactly
- Motor RPM is tracked as explicit state with first-order dynamics

CRITICAL IMPLEMENTATION DETAILS:
1. Actions are 4 normalized motor RPM commands in [-1, 1]
2. Actions map to RPM: rpm = (action + 1) / 2 * max_rpm
3. Motor state has first-order lag: d(rpm)/dt = (target - rpm) / tau
4. Thrust per motor: F = k_f * rpm^2
5. Forces/torques are computed from individual motor thrusts via mixer

References:
- learning-to-fly/include/learning_to_fly/simulator/parameters/dynamics/crazy_flie.h
- learning-to-fly/include/learning_to_fly/simulator/operations_generic.h
- learning-to-fly/src/config/ppo_config.h
"""

from __future__ import annotations

import torch
import math
from typing import Tuple
from dataclasses import dataclass

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers, CUBOID_MARKER_CFG
from isaaclab.envs.ui import BaseEnvWindow

from isaaclab_assets import CRAZYFLIE_CFG


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

class ConfigurationError(Exception):
    """Raised when environment configuration is invalid."""
    pass


def validate_positive(value: float, name: str) -> None:
    """Validate that a value is positive."""
    if value <= 0:
        raise ConfigurationError(f"{name} must be positive, got {value}")


def validate_non_negative(value: float, name: str) -> None:
    """Validate that a value is non-negative."""
    if value < 0:
        raise ConfigurationError(f"{name} must be non-negative, got {value}")


def validate_range(value: float, min_val: float, max_val: float, name: str) -> None:
    """Validate that a value is within range."""
    if not (min_val <= value <= max_val):
        raise ConfigurationError(f"{name} must be in [{min_val}, {max_val}], got {value}")


# =============================================================================
# L2F PHYSICS PARAMETERS (from learning-to-fly/crazy_flie.h)
# =============================================================================

@configclass
class L2FPhysicsCfg:
    """
    Physical parameters matching learning-to-fly Crazyflie model exactly.
    
    These values are taken directly from:
    learning-to-fly/include/learning_to_fly/simulator/parameters/dynamics/crazy_flie.h
    
    DO NOT modify without updating the reference documentation.
    """
    # Mass (kg) - total vehicle mass
    mass: float = 0.027
    
    # Rotor arm length (m) - distance from center to each rotor
    arm_length: float = 0.028
    
    # Moments of inertia (kg*m^2) - diagonal elements of inertia tensor
    Ixx: float = 3.85e-6
    Iyy: float = 3.85e-6
    Izz: float = 5.9675e-6
    
    # Thrust coefficient: F = k_f * rpm^2 (N/RPM^2)
    # From L2F: thrust_constants = {0, 0, 3.16e-10}
    thrust_coefficient: float = 3.16e-10
    
    # Torque coefficient ratio (k_t / k_f) for yaw moment
    torque_coefficient: float = 0.005964552
    
    # RPM limits
    min_rpm: float = 0.0
    max_rpm: float = 21702.0
    
    # Motor time constant (s) - first-order lag: tau in d(rpm)/dt = (cmd - rpm)/tau
    motor_time_constant: float = 0.15
    
    # Gravity (m/s^2)
    gravity: float = 9.81
    
    # Rotor positions relative to body center (m) - X configuration
    # Looking from above: M1=front-right, M2=back-right, M3=back-left, M4=front-left
    # From L2F crazy_flie.h
    rotor_positions: tuple = (
        (0.028, -0.028, 0.0),   # M1: front-right
        (-0.028, -0.028, 0.0),  # M2: back-right  
        (-0.028, 0.028, 0.0),   # M3: back-left
        (0.028, 0.028, 0.0),    # M4: front-left
    )
    
    # Rotor spin directions for yaw torque: -1=CW, +1=CCW
    # From L2F: rotor_torque_directions = {-1, +1, -1, +1}
    rotor_yaw_directions: tuple = (-1.0, 1.0, -1.0, 1.0)
    
    def validate(self) -> None:
        """Validate all physics parameters."""
        validate_positive(self.mass, "mass")
        validate_positive(self.arm_length, "arm_length")
        validate_positive(self.Ixx, "Ixx")
        validate_positive(self.Iyy, "Iyy")
        validate_positive(self.Izz, "Izz")
        validate_positive(self.thrust_coefficient, "thrust_coefficient")
        validate_positive(self.torque_coefficient, "torque_coefficient")
        validate_non_negative(self.min_rpm, "min_rpm")
        validate_positive(self.max_rpm, "max_rpm")
        validate_positive(self.motor_time_constant, "motor_time_constant")
        validate_positive(self.gravity, "gravity")
        
        if self.min_rpm >= self.max_rpm:
            raise ConfigurationError(f"min_rpm ({self.min_rpm}) must be < max_rpm ({self.max_rpm})")
        
        if len(self.rotor_positions) != 4:
            raise ConfigurationError("rotor_positions must have exactly 4 entries")
        if len(self.rotor_yaw_directions) != 4:
            raise ConfigurationError("rotor_yaw_directions must have exactly 4 entries")
    
    @property
    def hover_rpm(self) -> float:
        """RPM required for each motor to hover."""
        # Total thrust needed = weight = m * g
        # 4 motors: F_per_motor = m * g / 4
        # F = k_f * rpm^2 => rpm = sqrt(F / k_f)
        thrust_per_motor = self.mass * self.gravity / 4.0
        return math.sqrt(thrust_per_motor / self.thrust_coefficient)
    
    @property
    def hover_action(self) -> float:
        """Normalized action [-1, 1] that produces hover."""
        # rpm = (action + 1) / 2 * max_rpm
        # action = 2 * rpm / max_rpm - 1
        return 2.0 * self.hover_rpm / self.max_rpm - 1.0


# =============================================================================
# OBSERVATION CONFIGURATION
# =============================================================================

@configclass
class L2FObservationCfg:
    """
    Observation space matching learning-to-fly firmware adapter.
    
    Layout (146 dimensions total):
    - [0:3]   Position (x, y, z) in world frame
    - [3:12]  Rotation matrix (9 elements, row-major, from quaternion)
    - [12:15] Linear velocity (vx, vy, vz) in world frame  
    - [15:18] Angular velocity (wx, wy, wz) in BODY frame
    - [18:146] Action history (32 timesteps * 4 actions = 128)
    
    Reference: learning-to-fly/firmware_patches/rl_tools_adapter_ppo.cpp
    """
    # Core state dimensions
    position_dim: int = 3
    rotation_matrix_dim: int = 9
    linear_velocity_dim: int = 3
    angular_velocity_dim: int = 3
    
    # Action history configuration
    action_history_length: int = 32
    action_dim: int = 4
    use_action_history: bool = True
    
    # Observation noise standard deviations (from L2F default.h)
    position_noise_std: float = 0.001
    orientation_noise_std: float = 0.001
    linear_velocity_noise_std: float = 0.002
    angular_velocity_noise_std: float = 0.002
    
    @property
    def core_dim(self) -> int:
        """Core observation dimensions (without action history)."""
        return (self.position_dim + self.rotation_matrix_dim + 
                self.linear_velocity_dim + self.angular_velocity_dim)
    
    @property
    def history_dim(self) -> int:
        """Action history dimensions."""
        return self.action_history_length * self.action_dim if self.use_action_history else 0
    
    @property
    def total_dim(self) -> int:
        """Total observation dimensions."""
        return self.core_dim + self.history_dim
    
    def validate(self) -> None:
        """Validate observation configuration."""
        if self.action_history_length < 0:
            raise ConfigurationError("action_history_length must be non-negative")
        if self.action_dim != 4:
            raise ConfigurationError("action_dim must be 4 for quadrotor")


# =============================================================================
# REWARD CONFIGURATION (from learning-to-fly squared.h)
# =============================================================================

@configclass
class L2FRewardCfg:
    """
    Reward function matching learning-to-fly reward_squared_position_only_torque.
    
    Formula:
        weighted_cost = pos_w * ||pos||^2 + ori_w * (1-qw^2) + vel_w * ||vel||^2 + act_w * ||act-baseline||^2
        reward = -scale * weighted_cost + constant
    
    Reference: learning-to-fly/include/learning_to_fly/simulator/parameters/reward_functions/default.h
    """
    # Reward shaping parameters
    scale: float = 0.5
    constant: float = 2.0
    non_negative: bool = False
    termination_penalty: float = 0.0
    
    # Component weights (from reward_squared_position_only_torque)
    position_weight: float = 5.0
    orientation_weight: float = 5.0
    linear_velocity_weight: float = 0.01
    angular_velocity_weight: float = 0.0  # Not used in L2F default
    
    # Action regularization
    action_weight: float = 0.01
    action_baseline: float = 0.0  # Normalized hover action ≈ 0
    
    def validate(self) -> None:
        """Validate reward configuration."""
        validate_positive(self.scale, "scale")
        validate_non_negative(self.position_weight, "position_weight")
        validate_non_negative(self.orientation_weight, "orientation_weight")
        validate_non_negative(self.linear_velocity_weight, "linear_velocity_weight")
        validate_non_negative(self.angular_velocity_weight, "angular_velocity_weight")
        validate_non_negative(self.action_weight, "action_weight")


# =============================================================================
# INITIALIZATION CONFIGURATION
# =============================================================================

@configclass
class L2FInitializationCfg:
    """
    Initial state distribution matching learning-to-fly all_around_2 config.
    
    Reference: learning-to-fly/include/learning_to_fly/simulator/parameters/init/default.h
    """
    # Position randomization (m) - uniform in [-max, +max]
    max_position: float = 0.2
    
    # Orientation randomization (rad) - max angle from upright
    max_angle: float = 3.14  # Full rotation
    
    # Velocity randomization (m/s, rad/s)
    max_linear_velocity: float = 1.0
    max_angular_velocity: float = 1.0
    
    # Initial height above ground (m)
    initial_height: float = 1.0
    
    # Guidance probability (spawn at origin with some probability)
    guidance_probability: float = 0.1
    
    def validate(self) -> None:
        """Validate initialization configuration."""
        validate_non_negative(self.max_position, "max_position")
        validate_non_negative(self.max_angle, "max_angle")
        validate_non_negative(self.max_linear_velocity, "max_linear_velocity")
        validate_non_negative(self.max_angular_velocity, "max_angular_velocity")
        validate_positive(self.initial_height, "initial_height")
        validate_range(self.guidance_probability, 0.0, 1.0, "guidance_probability")


# =============================================================================
# TERMINATION CONFIGURATION
# =============================================================================

@configclass
class L2FTerminationCfg:
    """
    Termination conditions matching learning-to-fly.
    
    Reference: learning-to-fly/include/learning_to_fly/simulator/operations_generic.h
    """
    enabled: bool = True
    
    # Position threshold (m) - terminate if |pos_i| > threshold for any axis
    position_threshold: float = 10.0
    
    # Velocity thresholds
    linear_velocity_threshold: float = 10.0
    angular_velocity_threshold: float = 30.0
    
    # Height bounds
    min_height: float = 0.05
    max_height: float = 3.0
    
    def validate(self) -> None:
        """Validate termination configuration."""
        validate_positive(self.position_threshold, "position_threshold")
        validate_positive(self.linear_velocity_threshold, "linear_velocity_threshold")
        validate_positive(self.angular_velocity_threshold, "angular_velocity_threshold")
        validate_non_negative(self.min_height, "min_height")
        validate_positive(self.max_height, "max_height")


# =============================================================================
# DOMAIN RANDOMIZATION CONFIGURATION
# =============================================================================

@configclass
class L2FDomainRandomizationCfg:
    """
    Domain randomization for sim-to-real transfer.
    
    Reference: learning-to-fly ablation study (DefaultAblationSpec)
    """
    # External disturbances (constant per episode)
    enable_disturbance: bool = True
    disturbance_force_std: float = 0.0132  # mass * g / 20
    disturbance_torque_std: float = 2.65e-5  # mass * g / 10000
    
    # Motor delay (first-order lag) - always enabled via physics config
    # Observation noise - always enabled via observation config
    
    def validate(self) -> None:
        """Validate domain randomization configuration."""
        validate_non_negative(self.disturbance_force_std, "disturbance_force_std")
        validate_non_negative(self.disturbance_torque_std, "disturbance_torque_std")


# =============================================================================
# CALIBRATION CONFIGURATION
# =============================================================================

@configclass
class L2FCalibrationCfg:
    """
    Calibration thresholds for physics parity validation.
    
    Training will refuse to start unless calibration passes or is explicitly skipped.
    """
    # Whether calibration is required before training
    require_calibration: bool = True
    
    # Hover calibration
    hover_thrust_tolerance: float = 0.02  # ±2% of body weight
    hover_altitude_drift_max: float = 0.5  # m over 5 seconds (increased for sim tolerances)
    
    # Motor dynamics
    motor_time_constant_tolerance: float = 0.1  # ±10%
    
    # Step response
    roll_rate_peak_tolerance: float = 0.15  # ±15%
    pitch_rate_peak_tolerance: float = 0.15
    yaw_rate_peak_tolerance: float = 0.20  # ±20% (harder to match)
    
    # Trajectory comparison
    trajectory_nrmse_max: float = 0.10  # 10% normalized RMSE


# =============================================================================
# MAIN ENVIRONMENT CONFIGURATION
# =============================================================================

@configclass
class CrazyflieL2FEnvCfg(DirectRLEnvCfg):
    """
    Configuration for Crazyflie Learning-to-Fly compatible environment.
    
    This is the single source of truth for L2F-compatible quadrotor training.
    """
    # Episode configuration
    episode_length_s: float = 4.0  # 4 seconds per episode
    decimation: int = 1  # Control at physics rate (no decimation)
    
    # Spaces
    observation_space: int = 146  # 18 core + 128 action history
    action_space: int = 4  # 4 motor RPM commands
    state_space: int = 0
    
    # UI
    ui_window_class_type = None  # Set below after class definition
    debug_vis: bool = True
    
    # Simulation configuration - 100Hz physics
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 100.0,  # 100Hz physics timestep
        render_interval=2,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    # Terrain
    terrain: TerrainImporterCfg = TerrainImporterCfg(
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
    
    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=True,
    )
    
    # Robot configuration
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )
    
    # L2F-specific configurations
    physics: L2FPhysicsCfg = L2FPhysicsCfg()
    observation: L2FObservationCfg = L2FObservationCfg()
    reward: L2FRewardCfg = L2FRewardCfg()
    initialization: L2FInitializationCfg = L2FInitializationCfg()
    termination: L2FTerminationCfg = L2FTerminationCfg()
    domain_randomization: L2FDomainRandomizationCfg = L2FDomainRandomizationCfg()
    calibration: L2FCalibrationCfg = L2FCalibrationCfg()
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate all sub-configurations
        self.physics.validate()
        self.observation.validate()
        self.reward.validate()
        self.initialization.validate()
        self.termination.validate()
        self.domain_randomization.validate()
        
        # Verify observation space matches configuration
        expected_obs_dim = self.observation.total_dim
        if self.observation_space != expected_obs_dim:
            raise ConfigurationError(
                f"observation_space ({self.observation_space}) does not match "
                f"observation config total_dim ({expected_obs_dim})"
            )
        
        # Verify physics dt is reasonable
        if self.sim.dt > 0.02:
            raise ConfigurationError(
                f"Physics dt ({self.sim.dt}) is too large for quadrotor simulation. "
                f"Use dt <= 0.02s (50Hz minimum)."
            )


# =============================================================================
# UI WINDOW
# =============================================================================

class CrazyflieL2FEnvWindow(BaseEnvWindow):
    """Window manager for the Crazyflie L2F environment."""

    def __init__(self, env: "CrazyflieL2FEnv", window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


# Set UI class after definition
CrazyflieL2FEnvCfg.ui_window_class_type = CrazyflieL2FEnvWindow


# =============================================================================
# MAIN ENVIRONMENT CLASS
# =============================================================================

class CrazyflieL2FEnv(DirectRLEnv):
    """
    Crazyflie environment implementing L2F physics and control contract.
    
    This environment is the single source of truth for L2F-compatible training.
    It implements:
    - L2F motor model: actions are 4 normalized RPM commands [-1, 1]
    - First-order motor dynamics with configurable time constant
    - Per-motor thrust computation: F = k_f * rpm^2
    - Proper mixer for roll/pitch/yaw from differential thrust
    - L2F observation format with action history
    - L2F reward function (squared costs)
    
    CRITICAL: Motor RPM is tracked as explicit state, not derived from actions.
    """
    
    cfg: CrazyflieL2FEnvCfg
    
    def __init__(self, cfg: CrazyflieL2FEnvCfg, render_mode: str | None = None, **kwargs):
        """
        Initialize the Crazyflie L2F environment.
        
        Args:
            cfg: Environment configuration
            render_mode: Rendering mode (None, "human", "rgb_array")
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(cfg, render_mode, **kwargs)
        
        # Validate configuration was checked
        if not hasattr(cfg.physics, 'mass'):
            raise ConfigurationError("Physics configuration not properly initialized")
        
        # Cache physics parameters for fast access
        self._mass = cfg.physics.mass
        self._arm_length = cfg.physics.arm_length
        self._thrust_coef = cfg.physics.thrust_coefficient
        self._torque_coef = cfg.physics.torque_coefficient
        self._motor_tau = cfg.physics.motor_time_constant
        self._min_rpm = cfg.physics.min_rpm
        self._max_rpm = cfg.physics.max_rpm
        self._gravity = cfg.physics.gravity
        self._hover_rpm = cfg.physics.hover_rpm
        self._hover_action = cfg.physics.hover_action
        self._dt = cfg.sim.dt
        
        # Pre-compute motor dynamics alpha
        self._motor_alpha = self._dt / self._motor_tau
        if self._motor_alpha > 1.0:
            print(f"[WARNING] Motor alpha={self._motor_alpha:.3f} > 1.0, clamping. "
                  f"Consider reducing dt or increasing motor_time_constant.")
            self._motor_alpha = 1.0
        
        # Build rotor position tensor
        self._rotor_positions = torch.tensor(
            cfg.physics.rotor_positions, 
            device=self.device, 
            dtype=torch.float32
        )  # (4, 3)
        
        # Build yaw direction tensor
        self._rotor_yaw_dirs = torch.tensor(
            cfg.physics.rotor_yaw_directions,
            device=self.device,
            dtype=torch.float32
        )  # (4,)
        
        # Initialize state tensors
        self._actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._rpm_state = torch.zeros(self.num_envs, 4, device=self.device)
        
        # Force/torque buffers
        self._thrust_body = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._torque_body = torch.zeros(self.num_envs, 1, 3, device=self.device)
        
        # Action history buffer
        if cfg.observation.use_action_history:
            self._action_history = torch.zeros(
                self.num_envs,
                cfg.observation.action_history_length,
                cfg.observation.action_dim,
                device=self.device,
            )
        else:
            self._action_history = None
        
        # Disturbance forces (constant per episode)
        self._disturbance_force = torch.zeros(self.num_envs, 3, device=self.device)
        self._disturbance_torque = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Episode statistics
        self._episode_sums = {
            "position_cost": torch.zeros(self.num_envs, device=self.device),
            "orientation_cost": torch.zeros(self.num_envs, device=self.device),
            "linear_velocity_cost": torch.zeros(self.num_envs, device=self.device),
            "angular_velocity_cost": torch.zeros(self.num_envs, device=self.device),
            "action_cost": torch.zeros(self.num_envs, device=self.device),
            "total_reward": torch.zeros(self.num_envs, device=self.device),
        }
        
        # Get body ID for force application
        self._body_id = self._robot.find_bodies("body")[0]
        
        # Debug visualization
        self.set_debug_vis(self.cfg.debug_vis)
        
        # Print configuration summary
        self._print_config_summary()
    
    def _print_config_summary(self) -> None:
        """Print configuration summary for verification."""
        print("\n" + "="*60)
        print("Crazyflie L2F Environment Initialized")
        print("="*60)
        print(f"  Physics dt:         {self._dt*1000:.1f} ms ({1/self._dt:.0f} Hz)")
        print(f"  Control dt:         {self._dt*1000:.1f} ms (decimation=1)")
        print(f"  Episode length:     {self.cfg.episode_length_s:.1f} s ({self.max_episode_length} steps)")
        print(f"  Num environments:   {self.num_envs}")
        print(f"  Observation dim:    {self.cfg.observation_space}")
        print(f"  Action dim:         {self.cfg.action_space}")
        print(f"  Mass:               {self._mass*1000:.1f} g")
        print(f"  Hover RPM:          {self._hover_rpm:.0f}")
        print(f"  Hover action:       {self._hover_action:.4f}")
        print(f"  Motor tau:          {self._motor_tau*1000:.0f} ms")
        print(f"  Motor alpha:        {self._motor_alpha:.4f}")
        print("="*60 + "\n")
    
    # =========================================================================
    # SCENE SETUP
    # =========================================================================
    
    def _setup_scene(self) -> None:
        """Set up the simulation scene."""
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        
        # Filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    
    # =========================================================================
    # QUATERNION UTILITIES
    # =========================================================================
    
    def _quat_to_rotation_matrix(self, quat: torch.Tensor) -> torch.Tensor:
        """
        Convert quaternion to flattened rotation matrix.
        
        MUST match L2F observe_rotation_matrix() exactly.
        
        Args:
            quat: Quaternion [w, x, y, z], shape (N, 4)
            
        Returns:
            Flattened rotation matrix, shape (N, 9), row-major order
        """
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        
        # Rotation matrix elements (row-major as in L2F)
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
    
    def _quat_rotate(self, quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """
        Rotate vector by quaternion.
        
        Args:
            quat: Quaternion [w, x, y, z], shape (N, 4)
            vec: Vector to rotate, shape (N, 3)
            
        Returns:
            Rotated vector, shape (N, 3)
        """
        # Extract components
        w, x, y, z = quat[:, 0:1], quat[:, 1:2], quat[:, 2:3], quat[:, 3:4]
        
        # Quaternion rotation: v' = q * v * q^-1
        # Using Rodrigues formula for efficiency
        qvec = torch.cat([x, y, z], dim=-1)  # (N, 3)
        
        uv = torch.cross(qvec, vec, dim=-1)
        uuv = torch.cross(qvec, uv, dim=-1)
        
        return vec + 2 * (w * uv + uuv)
    
    # =========================================================================
    # OBSERVATION CONSTRUCTION
    # =========================================================================
    
    def _get_observations(self) -> dict:
        """
        Construct observations matching L2F firmware format.
        
        Layout (146 dims):
        - [0:3]   Position error (relative to target)
        - [3:12]  Rotation matrix (9 elements, row-major)
        - [12:15] Linear velocity in world frame
        - [15:18] Angular velocity in BODY frame
        - [18:146] Action history (32 * 4 = 128)
        
        Returns:
            Dictionary with "policy" key containing observation tensor
        """
        # Get state from simulator
        pos_w = self._robot.data.root_pos_w  # (N, 3)
        quat_w = self._robot.data.root_quat_w  # (N, 4) [w,x,y,z]
        lin_vel_w = self._robot.data.root_lin_vel_w  # (N, 3)
        ang_vel_b = self._robot.data.root_ang_vel_b  # (N, 3) body frame
        
        # Compute position error relative to target (env origin + initial height)
        target_pos = self._terrain.env_origins.clone()
        target_pos[:, 2] += self.cfg.initialization.initial_height
        pos_error = pos_w - target_pos
        
        # Convert quaternion to rotation matrix
        rot_matrix = self._quat_to_rotation_matrix(quat_w)  # (N, 9)
        
        # Apply observation noise
        obs_cfg = self.cfg.observation
        if obs_cfg.position_noise_std > 0:
            pos_error = pos_error + torch.randn_like(pos_error) * obs_cfg.position_noise_std
        if obs_cfg.orientation_noise_std > 0:
            rot_matrix = rot_matrix + torch.randn_like(rot_matrix) * obs_cfg.orientation_noise_std
        if obs_cfg.linear_velocity_noise_std > 0:
            lin_vel_w = lin_vel_w + torch.randn_like(lin_vel_w) * obs_cfg.linear_velocity_noise_std
        if obs_cfg.angular_velocity_noise_std > 0:
            ang_vel_b = ang_vel_b + torch.randn_like(ang_vel_b) * obs_cfg.angular_velocity_noise_std
        
        # Concatenate core observation (18 dims)
        core_obs = torch.cat([pos_error, rot_matrix, lin_vel_w, ang_vel_b], dim=-1)
        
        # Append action history if enabled (128 dims)
        if self._action_history is not None:
            action_history_flat = self._action_history.view(self.num_envs, -1)
            obs = torch.cat([core_obs, action_history_flat], dim=-1)
        else:
            obs = core_obs
        
        # Validate observation shape
        expected_dim = self.cfg.observation_space
        if obs.shape[-1] != expected_dim:
            raise RuntimeError(
                f"Observation dimension mismatch: got {obs.shape[-1]}, expected {expected_dim}"
            )
        
        return {"policy": obs}
    
    # =========================================================================
    # ACTION PROCESSING (L2F MOTOR MODEL)
    # =========================================================================
    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        Process actions and compute forces/torques.
        
        L2F Motor Model:
        1. Actions are normalized RPM commands in [-1, 1]
        2. Map to target RPM: target_rpm = (action + 1) / 2 * max_rpm
        3. First-order dynamics: rpm += alpha * (target_rpm - rpm)
        4. Thrust per motor: F = k_f * rpm^2
        5. Compute body forces/torques via mixer
        
        Args:
            actions: Normalized motor commands, shape (N, 4)
        """
        # Validate actions
        if actions.shape != (self.num_envs, 4):
            raise RuntimeError(f"Invalid action shape: {actions.shape}, expected ({self.num_envs}, 4)")
        
        # Store and clamp actions
        self._actions = actions.clone().clamp(-1.0, 1.0)
        
        # Map to target RPM
        target_rpm = (self._actions + 1.0) / 2.0 * self._max_rpm
        
        # Apply first-order motor dynamics
        # d(rpm)/dt = (target - rpm) / tau
        # rpm_new = rpm + dt/tau * (target - rpm) = rpm + alpha * (target - rpm)
        self._rpm_state = self._rpm_state + self._motor_alpha * (target_rpm - self._rpm_state)
        self._rpm_state = self._rpm_state.clamp(self._min_rpm, self._max_rpm)
        
        # Compute thrust per motor: F = k_f * rpm^2
        thrust_per_motor = self._thrust_coef * self._rpm_state ** 2  # (N, 4)
        
        # Total thrust (body z-axis, pointing up)
        total_thrust = thrust_per_motor.sum(dim=-1, keepdim=True)  # (N, 1)
        
        # Thrust vector in body frame (pointing up = +Z in body frame)
        # Note: Isaac Lab's set_external_force_and_torque expects forces in BODY/LOCAL frame
        thrust_body = torch.zeros(self.num_envs, 3, device=self.device)
        thrust_body[:, 2] = total_thrust.squeeze()
        
        # Compute torques from differential thrust
        # Roll (about body x-axis): differential y-position thrust
        # M1(+y) and M4(+y) vs M2(-y) and M3(-y) -- wait, check positions
        # Positions: M1=(+x,-y), M2=(-x,-y), M3=(-x,+y), M4=(+x,+y)
        # Roll moment = sum(F_i * y_i)
        roll_torque = (
            thrust_per_motor[:, 0] * self._rotor_positions[0, 1] +  # M1
            thrust_per_motor[:, 1] * self._rotor_positions[1, 1] +  # M2
            thrust_per_motor[:, 2] * self._rotor_positions[2, 1] +  # M3
            thrust_per_motor[:, 3] * self._rotor_positions[3, 1]    # M4
        )
        
        # Pitch moment = -sum(F_i * x_i) (negative because of coordinate convention)
        pitch_torque = -(
            thrust_per_motor[:, 0] * self._rotor_positions[0, 0] +
            thrust_per_motor[:, 1] * self._rotor_positions[1, 0] +
            thrust_per_motor[:, 2] * self._rotor_positions[2, 0] +
            thrust_per_motor[:, 3] * self._rotor_positions[3, 0]
        )
        
        # Yaw moment = reaction torque from motor spin
        # tau_yaw = k_t * F = k_t * k_f * rpm^2 = torque_coef * thrust
        yaw_torque = self._torque_coef * (
            self._rotor_yaw_dirs[0] * thrust_per_motor[:, 0] +
            self._rotor_yaw_dirs[1] * thrust_per_motor[:, 1] +
            self._rotor_yaw_dirs[2] * thrust_per_motor[:, 2] +
            self._rotor_yaw_dirs[3] * thrust_per_motor[:, 3]
        )
        
        # Combine torques (body frame)
        torque_body_vec = torch.stack([roll_torque, pitch_torque, yaw_torque], dim=-1)
        
        # Add disturbances if enabled (disturbances are also in body frame)
        if self.cfg.domain_randomization.enable_disturbance:
            thrust_body = thrust_body + self._disturbance_force
            torque_body_vec = torque_body_vec + self._disturbance_torque
        
        # Store for application (in body frame for Isaac Lab)
        self._thrust_body[:, 0, :] = thrust_body
        self._torque_body[:, 0, :] = torque_body_vec
        
        # Update action history
        self._update_action_history()
    
    def _update_action_history(self) -> None:
        """Update action history buffer with current action."""
        if self._action_history is None:
            return
        
        # Shift history left (discard oldest)
        self._action_history[:, :-1] = self._action_history[:, 1:].clone()
        
        # Add current action at the end
        self._action_history[:, -1] = self._actions
    
    def _apply_action(self) -> None:
        """Apply computed forces and torques to the robot.
        
        Note: Isaac Lab's set_external_force_and_torque expects forces/torques in the
        body/local frame of the link, NOT the world frame. The physics engine handles
        the transformation to world frame internally.
        """
        self._robot.set_external_force_and_torque(
            forces=self._thrust_body,
            torques=self._torque_body,
            body_ids=self._body_id,
        )
    
    # =========================================================================
    # REWARD FUNCTION
    # =========================================================================
    
    def _get_rewards(self) -> torch.Tensor:
        """
        Compute reward matching L2F reward_squared_position_only_torque.
        
        Formula:
            weighted_cost = pos_w * ||pos||^2 + ori_w * (1-qw^2) + vel_w * ||vel||^2 + act_w * ||act-baseline||^2
            reward = -scale * weighted_cost + constant
        
        Returns:
            Reward tensor, shape (N,)
        """
        reward_cfg = self.cfg.reward
        
        # Get state
        pos_world = self._robot.data.root_pos_w
        quat = self._robot.data.root_quat_w
        lin_vel = self._robot.data.root_lin_vel_w
        ang_vel = self._robot.data.root_ang_vel_b
        
        # Position relative to target (environment origin at initial height)
        # Goal is to hover at env_origin + (0, 0, initial_height)
        target_pos = self._terrain.env_origins.clone()
        target_pos[:, 2] += self.cfg.initialization.initial_height
        pos_error = pos_world - target_pos
        
        # Position cost: ||pos_error||^2
        position_cost = (pos_error ** 2).sum(dim=-1)
        
        # Orientation cost: 1 - qw^2 (deviation from upright)
        orientation_cost = 1.0 - quat[:, 0] ** 2
        
        # Linear velocity cost
        linear_velocity_cost = (lin_vel ** 2).sum(dim=-1)
        
        # Angular velocity cost
        angular_velocity_cost = (ang_vel ** 2).sum(dim=-1)
        
        # Action cost: ||action - baseline||^2
        action_diff = self._actions - reward_cfg.action_baseline
        action_cost = (action_diff ** 2).sum(dim=-1)
        
        # Weighted sum
        weighted_cost = (
            reward_cfg.position_weight * position_cost +
            reward_cfg.orientation_weight * orientation_cost +
            reward_cfg.linear_velocity_weight * linear_velocity_cost +
            reward_cfg.angular_velocity_weight * angular_velocity_cost +
            reward_cfg.action_weight * action_cost
        )
        
        # Compute reward
        reward = -reward_cfg.scale * weighted_cost + reward_cfg.constant
        
        # Clamp to non-negative if configured
        if reward_cfg.non_negative:
            reward = torch.clamp(reward, min=0.0)
        
        # Update episode statistics
        self._episode_sums["position_cost"] += position_cost
        self._episode_sums["orientation_cost"] += orientation_cost
        self._episode_sums["linear_velocity_cost"] += linear_velocity_cost
        self._episode_sums["angular_velocity_cost"] += angular_velocity_cost
        self._episode_sums["action_cost"] += action_cost
        self._episode_sums["total_reward"] += reward
        
        return reward
    
    # =========================================================================
    # TERMINATION
    # =========================================================================
    
    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Check termination conditions.
        
        Returns:
            Tuple of (terminated, time_out) tensors
        """
        # Time-based termination
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        term_cfg = self.cfg.termination
        if not term_cfg.enabled:
            return torch.zeros_like(time_out), time_out
        
        # Get state (position relative to target)
        pos_world = self._robot.data.root_pos_w
        target_pos = self._terrain.env_origins.clone()
        target_pos[:, 2] += self.cfg.initialization.initial_height
        pos_error = pos_world - target_pos
        
        lin_vel = self._robot.data.root_lin_vel_w
        ang_vel = self._robot.data.root_ang_vel_b
        
        # Position exceeded (any axis) - use relative position
        pos_exceeded = (torch.abs(pos_error) > term_cfg.position_threshold).any(dim=-1)
        
        # Height bounds (use relative height from target)
        # target height is initial_height, so pos_error[:, 2] is height relative to target
        height_above_ground = pos_world[:, 2] - self._terrain.env_origins[:, 2]
        too_low = height_above_ground < term_cfg.min_height
        too_high = height_above_ground > term_cfg.max_height
        
        # Velocity thresholds
        lin_vel_exceeded = torch.norm(lin_vel, dim=-1) > term_cfg.linear_velocity_threshold
        ang_vel_exceeded = torch.norm(ang_vel, dim=-1) > term_cfg.angular_velocity_threshold
        
        # Combined termination
        terminated = pos_exceeded | too_low | too_high | lin_vel_exceeded | ang_vel_exceeded
        
        return terminated, time_out
    
    # =========================================================================
    # RESET
    # =========================================================================
    
    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        """Reset specified environments."""
        if env_ids is None or len(env_ids) == 0:
            return
        
        # Log episode statistics before reset
        self._log_episode_stats(env_ids)
        
        # Parent reset
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        
        n = len(env_ids)
        init_cfg = self.cfg.initialization
        
        # Determine which envs get "guided" initialization (spawn at origin)
        guidance_mask = torch.rand(n, device=self.device) < init_cfg.guidance_probability
        
        # Sample positions
        pos = torch.zeros(n, 3, device=self.device)
        
        # Random position for non-guided
        pos[~guidance_mask, 0] = torch.empty(
            (~guidance_mask).sum(), device=self.device
        ).uniform_(-init_cfg.max_position, init_cfg.max_position)
        pos[~guidance_mask, 1] = torch.empty(
            (~guidance_mask).sum(), device=self.device
        ).uniform_(-init_cfg.max_position, init_cfg.max_position)
        pos[~guidance_mask, 2] = init_cfg.initial_height + torch.empty(
            (~guidance_mask).sum(), device=self.device
        ).uniform_(-init_cfg.max_position, init_cfg.max_position)
        
        # Guided envs at origin
        pos[guidance_mask, 2] = init_cfg.initial_height
        
        # Add terrain offset
        pos = pos + self._terrain.env_origins[env_ids]
        
        # Sample orientation (random quaternion within angle limit)
        if init_cfg.max_angle > 0 and not guidance_mask.all():
            quat = self._sample_random_quaternion(n, init_cfg.max_angle, guidance_mask)
        else:
            # Identity quaternion for all
            quat = torch.zeros(n, 4, device=self.device)
            quat[:, 0] = 1.0
        
        # Sample velocities
        lin_vel = torch.empty(n, 3, device=self.device).uniform_(
            -init_cfg.max_linear_velocity, init_cfg.max_linear_velocity
        )
        lin_vel[guidance_mask] = 0.0
        
        ang_vel = torch.empty(n, 3, device=self.device).uniform_(
            -init_cfg.max_angular_velocity, init_cfg.max_angular_velocity
        )
        ang_vel[guidance_mask] = 0.0
        
        # Build root state
        root_pose = torch.cat([pos, quat], dim=-1)
        root_vel = torch.cat([lin_vel, ang_vel], dim=-1)
        
        # Write to simulation
        self._robot.write_root_pose_to_sim(root_pose, env_ids)
        self._robot.write_root_velocity_to_sim(root_vel, env_ids)
        
        # CRITICAL: Initialize motor state to HOVER RPM
        self._rpm_state[env_ids] = self._hover_rpm
        
        # Initialize action history to hover action
        if self._action_history is not None:
            self._action_history[env_ids] = self._hover_action
        
        # Reset stored actions to hover
        self._actions[env_ids] = self._hover_action
        
        # Sample new disturbances
        if self.cfg.domain_randomization.enable_disturbance:
            dr_cfg = self.cfg.domain_randomization
            self._disturbance_force[env_ids] = torch.randn(n, 3, device=self.device) * dr_cfg.disturbance_force_std
            self._disturbance_torque[env_ids] = torch.randn(n, 3, device=self.device) * dr_cfg.disturbance_torque_std
        
        # Reset episode statistics
        for key in self._episode_sums:
            self._episode_sums[key][env_ids] = 0.0
    
    def _sample_random_quaternion(
        self, 
        n: int, 
        max_angle: float, 
        identity_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample random quaternions within angle limit.
        
        Uses rejection sampling to ensure angle < max_angle.
        
        Args:
            n: Number of quaternions to sample
            max_angle: Maximum rotation angle (radians)
            identity_mask: Mask for envs that should get identity quaternion
            
        Returns:
            Quaternions [w, x, y, z], shape (n, 4)
        """
        quat = torch.zeros(n, 4, device=self.device)
        quat[:, 0] = 1.0  # Default to identity
        
        # Sample for non-identity envs
        to_sample = ~identity_mask
        if not to_sample.any():
            return quat
        
        n_sample = to_sample.sum().item()
        
        # Use uniform random quaternion (Shoemake's method)
        u = torch.rand(n_sample, 3, device=self.device)
        
        sqrt_u0 = torch.sqrt(1 - u[:, 0])
        sqrt_u0_prime = torch.sqrt(u[:, 0])
        
        theta1 = 2 * math.pi * u[:, 1]
        theta2 = 2 * math.pi * u[:, 2]
        
        w = sqrt_u0_prime * torch.cos(theta2)
        x = sqrt_u0 * torch.sin(theta1)
        y = sqrt_u0 * torch.cos(theta1)
        z = sqrt_u0_prime * torch.sin(theta2)
        
        q = torch.stack([w, x, y, z], dim=-1)
        
        # Compute angle: angle = 2 * acos(|w|)
        angle = 2 * torch.acos(torch.clamp(torch.abs(q[:, 0]), max=1.0))
        
        # For angles exceeding max_angle, scale them down
        scale_needed = angle > max_angle
        if scale_needed.any():
            # Scale the rotation to max_angle
            target_w = math.cos(max_angle / 2)
            scale_factor = torch.sqrt((1 - target_w**2) / (1 - q[scale_needed, 0]**2 + 1e-8))
            q[scale_needed, 1:] *= scale_factor.unsqueeze(-1)
            q[scale_needed, 0] = target_w
        
        # Normalize
        q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)
        
        quat[to_sample] = q
        return quat
    
    def _log_episode_stats(self, env_ids: torch.Tensor) -> None:
        """Log episode statistics."""
        if len(env_ids) == 0:
            return
        
        extras = {}
        for key, values in self._episode_sums.items():
            avg = torch.mean(values[env_ids]).item()
            # Normalize by episode length
            steps = self.episode_length_buf[env_ids].float().mean().item()
            if steps > 0:
                extras[f"Episode/{key}"] = avg / steps
            else:
                extras[f"Episode/{key}"] = avg
        
        self.extras["log"] = extras
    
    # =========================================================================
    # DEBUG VISUALIZATION
    # =========================================================================
    
    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """Set up debug visualization."""
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
    
    def _debug_vis_callback(self, event) -> None:
        """Debug visualization callback."""
        # Visualize origin as goal (hover position)
        origins = self._terrain.env_origins.clone()
        origins[:, 2] = self.cfg.initialization.initial_height
        self.goal_pos_visualizer.visualize(origins)
    
    # =========================================================================
    # CALIBRATION INTERFACE
    # =========================================================================
    
    def get_rpm_state(self) -> torch.Tensor:
        """Get current motor RPM state for calibration."""
        return self._rpm_state.clone()
    
    def set_rpm_state(self, rpm: torch.Tensor) -> None:
        """Set motor RPM state directly (for calibration)."""
        if rpm.shape != self._rpm_state.shape:
            raise ValueError(f"RPM shape mismatch: {rpm.shape} vs {self._rpm_state.shape}")
        self._rpm_state = rpm.clone()
    
    def get_thrust_per_motor(self) -> torch.Tensor:
        """Get thrust per motor for calibration."""
        return self._thrust_coef * self._rpm_state ** 2
    
    def get_total_thrust(self) -> torch.Tensor:
        """Get total thrust magnitude for calibration."""
        return self.get_thrust_per_motor().sum(dim=-1)
    
    def step_calibration(self, actions: torch.Tensor) -> dict:
        """
        Step environment and return detailed state for calibration.
        
        Args:
            actions: Motor commands in [-1, 1], shape (N, 4)
            
        Returns:
            Dictionary with full state information
        """
        obs, reward, terminated, truncated, info = self.step(actions)
        
        return {
            "observation": obs,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "position": self._robot.data.root_pos_w.clone(),
            "quaternion": self._robot.data.root_quat_w.clone(),
            "linear_velocity": self._robot.data.root_lin_vel_w.clone(),
            "angular_velocity": self._robot.data.root_ang_vel_b.clone(),
            "rpm_state": self._rpm_state.clone(),
            "thrust_per_motor": self.get_thrust_per_motor(),
            "total_thrust": self.get_total_thrust(),
            "actions": self._actions.clone(),
        }
