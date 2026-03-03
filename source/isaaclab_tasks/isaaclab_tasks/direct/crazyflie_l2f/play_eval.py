#!/usr/bin/env python3
"""
Crazyflie Simulation Evaluation Script

Evaluates trained RL policies in Isaac Lab simulation and outputs IMU values,
motor RPM and thrust data for real-world comparison. Runs 100 parallel agents
and computes averaged statistics across all environments.

Supports two modes:
  - hover:    Evaluate hovering stability at fixed height
  - pointnav: Evaluate point navigation to goals (random or fixed)

Output includes:
  - IMU data (accelerometer, gyroscope) from simulation
  - Motor RPM and thrust per motor
  - Position, velocity, and attitude data
  - Averaged statistics across all 100 agents
  - Plots generated using flight_eval_utils.py

Usage (from IsaacLab directory):
    # Hover evaluation at default 1.0m height
    .\\isaaclab.bat -p source\\isaaclab_tasks\\isaaclab_tasks\\direct\\crazyflie_l2f\\play_eval.py --mode hover
    
    # Hover at custom height (e.g., 0.5m)
    .\\isaaclab.bat -p source\\isaaclab_tasks\\isaaclab_tasks\\direct\\crazyflie_l2f\\play_eval.py --mode hover --target_z 0.5
    
    # Point navigation with random goals
    .\\isaaclab.bat -p source\\isaaclab_tasks\\isaaclab_tasks\\direct\\crazyflie_l2f\\play_eval.py --mode pointnav
    
    # Point navigation with fixed goal (e.g., go to x=0.3, y=0.2, z=1.0)
    .\\isaaclab.bat -p source\\isaaclab_tasks\\isaaclab_tasks\\direct\\crazyflie_l2f\\play_eval.py --mode pointnav --target_x 0.3 --target_y 0.2 --goal_z 1.0
    
    # Custom checkpoint
    .\\isaaclab.bat -p source\\isaaclab_tasks\\isaaclab_tasks\\direct\\crazyflie_l2f\\play_eval.py --mode hover --checkpoint path/to/model.pt
    
    # With visualization (fewer envs)
    .\\isaaclab.bat -p source\\isaaclab_tasks\\isaaclab_tasks\\direct\\crazyflie_l2f\\play_eval.py --mode hover --num_envs 16
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Isaac Sim setup - must happen before other imports
from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description="Crazyflie Simulation Evaluation")
    
    # Mode selection
    parser.add_argument("--mode", type=str, choices=["hover", "pointnav"], default="hover",
                        help="Evaluation mode: hover or pointnav (default: hover)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (auto-detected if not specified)")
    
    # Simulation parameters
    parser.add_argument("--num_envs", type=int, default=100,
                        help="Number of parallel environments (default: 100)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Evaluation duration in seconds (default: 10.0)")
    parser.add_argument("--save_interval", type=int, default=100,
                        help="Save data every N steps (default: 100)")
    
    # Target position parameters (for standardization)
    parser.add_argument("--target_z", type=float, default=1.0,
                        help="[Hover] Target hover height in meters (default: 1.0)")
    parser.add_argument("--target_x", type=float, default=None,
                        help="[PointNav] Fixed goal X position in meters (default: random)")
    parser.add_argument("--target_y", type=float, default=None,
                        help="[PointNav] Fixed goal Y position in meters (default: random)")
    parser.add_argument("--goal_z", type=float, default=None,
                        help="[PointNav] Fixed goal Z height in meters (default: same as spawn)")
    parser.add_argument("--hold_time", type=float, default=5.0,
                        help="Time to hold at target after reaching it (default: 5.0s)")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (auto-generated if not specified)")
    parser.add_argument("--no_plot", action="store_true",
                        help="Disable plot generation")
    
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

# Import our custom Crazyflie configuration
from crazyflie_21_cfg import CRAZYFLIE_21_CFG, CrazyflieL2FParams

# Import utilities
from flight_eval_utils import quat_to_euler, plot_flight_data, plot_motor_data


# ==============================================================================
# L2F Physics Constants
# ==============================================================================

class L2FConstants:
    """Physical parameters matching learning-to-fly exactly."""
    
    MASS = 0.027  # kg (27g)
    ARM_LENGTH = 0.028  # m (28mm)
    GRAVITY = 9.81  # m/s^2
    
    IXX = 3.85e-6    # kg·m²
    IYY = 3.85e-6    # kg·m²
    IZZ = 5.9675e-6  # kg·m²
    
    THRUST_COEFFICIENT = 3.16e-10  # N/RPM²
    TORQUE_COEFFICIENT = 0.005964552  # Nm/N
    RPM_MIN = 0.0
    RPM_MAX = 21702.0
    MOTOR_TIME_CONSTANT = 0.15  # seconds
    
    ROTOR_POSITIONS = [
        (0.028, -0.028, 0.0),   # M1
        (-0.028, -0.028, 0.0),  # M2
        (-0.028, 0.028, 0.0),   # M3
        (0.028, 0.028, 0.0),    # M4
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
# Extended Flight Data Logger with Motor Data
# ==============================================================================

# Shared CSV column order — identical to WebUI ui.html FLIGHT_LOG_FIELDS + sim extras.
# Any comparison tool can rely on these columns existing in both sim and real CSVs.
SHARED_CSV_FIELDS = [
    # --- Base required (matches flight_logger.py / WebUI order) ---
    "t",
    "stateEstimate.x", "stateEstimate.y", "stateEstimate.z",
    "acc.x", "acc.y", "acc.z",
    "gyro.x", "gyro.y", "gyro.z",
    "stabilizer.roll", "stabilizer.pitch", "stabilizer.yaw",
    # --- Target ---
    "target.x", "target.y", "target.z", "target.yaw",
    # --- Battery (real-only; sim writes simulated nominal) ---
    "pm.vbat",
    # --- Motors ---
    "motor.m1", "motor.m2", "motor.m3", "motor.m4",
    # --- Raw IMU (real best-effort; sim mirrors filtered) ---
    "imu.acc_x", "imu.acc_y", "imu.acc_z",
    "imu.gyro_x", "imu.gyro_y", "imu.gyro_z",
    "imu.mag_x", "imu.mag_y", "imu.mag_z",
    # --- Sim-only extras (real CSVs will have NaN) ---
    "velocity.x", "velocity.y", "velocity.z",
    "motor.thrust.m1", "motor.thrust.m2", "motor.thrust.m3", "motor.thrust.m4",
    "motor.thrust.total",
]


class ExtendedFlightDataLogger:
    """Logger for collecting comprehensive flight evaluation data.
    
    Outputs CSV columns in the same order as the WebUI CSV export
    (SHARED_CSV_FIELDS) so that sim and real data are directly comparable.
    
    Includes:
    - IMU data (accelerometer, gyroscope)
    - Motor RPM per motor
    - Thrust per motor
    - Averaged statistics across multiple environments
    """
    
    def __init__(self, num_envs: int):
        """Initialize the flight data logger.
        
        Args:
            num_envs: Number of environments to track
        """
        self.num_envs = num_envs
        self.log_data = []
        self.t0 = time.perf_counter()
        self.prev_vel = None
        self.prev_time = None
        self.sim_time = 0.0
        
    def reset(self):
        """Reset the logger for a new flight session."""
        self.log_data = []
        self.t0 = time.perf_counter()
        self.prev_vel = None
        self.prev_time = None
        self.sim_time = 0.0
        
    def log_step(self, env, dt: float = 0.01, target: Optional[Tuple[float, float, float]] = None,
                  target_yaw: float = 0.0, env_idx: int = 0):
        """Log data for a single environment (like real drone flight).
        
        Output columns match SHARED_CSV_FIELDS so sim/real CSVs are directly comparable.
        
        Args:
            env: The environment instance
            dt: Simulation timestep in seconds
            target: Optional target position (x, y, z) to log
            target_yaw: Target yaw in degrees (default: 0)
            env_idx: Index of environment to log (default: 0)
        """
        self.sim_time += dt
        current_time = self.sim_time
        
        # Get state data from single environment (matches real drone)
        pos = env._robot.data.root_pos_w[env_idx].cpu().numpy()  # [3]
        vel = env._robot.data.root_lin_vel_w[env_idx].cpu().numpy()  # [3]
        ang_vel = env._robot.data.root_ang_vel_w[env_idx].cpu().numpy()  # [3]
        quat = env._robot.data.root_quat_w[env_idx].cpu()  # [4]
        
        # Convert quaternion to euler
        euler = quat_to_euler(quat.unsqueeze(0)).squeeze(0).numpy()  # [3]
        
        # Compute acceleration from velocity (like real IMU)
        if self.prev_vel is not None and self.prev_time is not None:
            delta_t = current_time - self.prev_time
            if delta_t > 0:
                acc = (vel - self.prev_vel) / delta_t
            else:
                acc = np.zeros(3)
        else:
            acc = np.zeros(3)
        
        # Get motor data from single environment
        rpm = env._rpm_state[env_idx].cpu().numpy()  # [4]
        thrust_per_motor = L2FConstants.THRUST_COEFFICIENT * rpm ** 2  # [4]
        total_thrust = thrust_per_motor.sum()
        
        # Acceleration with gravity offset (IMU-like)
        acc_x = float(acc[0])
        acc_y = float(acc[1])
        acc_z = float(acc[2] + L2FConstants.GRAVITY)
        
        # Gyro in deg/s
        gyro_x = float(ang_vel[0] * 180.0 / np.pi)
        gyro_y = float(ang_vel[1] * 180.0 / np.pi)
        gyro_z = float(ang_vel[2] * 180.0 / np.pi)
        
        # Build entry with SHARED_CSV_FIELDS column order
        entry = {
            # --- Base required (flight_logger / WebUI order) ---
            "t": current_time,
            "stateEstimate.x": float(pos[0]),
            "stateEstimate.y": float(pos[1]),
            "stateEstimate.z": float(pos[2]),
            "acc.x": acc_x,
            "acc.y": acc_y,
            "acc.z": acc_z,
            "gyro.x": gyro_x,
            "gyro.y": gyro_y,
            "gyro.z": gyro_z,
            "stabilizer.roll": float(euler[0]),
            "stabilizer.pitch": float(euler[1]),
            "stabilizer.yaw": float(euler[2]),
            # --- Target ---
            "target.x": target[0] if target else float('nan'),
            "target.y": target[1] if target else float('nan'),
            "target.z": target[2] if target else float('nan'),
            "target.yaw": target_yaw,
            # --- Battery (simulated nominal 4.2 V) ---
            "pm.vbat": 4.2,
            # --- Motors ---
            "motor.m1": float(rpm[0]),
            "motor.m2": float(rpm[1]),
            "motor.m3": float(rpm[2]),
            "motor.m4": float(rpm[3]),
            # --- Raw IMU (sim has no separate raw; mirror filtered) ---
            "imu.acc_x": acc_x,
            "imu.acc_y": acc_y,
            "imu.acc_z": acc_z,
            "imu.gyro_x": gyro_x,
            "imu.gyro_y": gyro_y,
            "imu.gyro_z": gyro_z,
            "imu.mag_x": float('nan'),  # no magnetometer in sim
            "imu.mag_y": float('nan'),
            "imu.mag_z": float('nan'),
            # --- Sim-only extras ---
            "velocity.x": float(vel[0]),
            "velocity.y": float(vel[1]),
            "velocity.z": float(vel[2]),
            "motor.thrust.m1": float(thrust_per_motor[0]),
            "motor.thrust.m2": float(thrust_per_motor[1]),
            "motor.thrust.m3": float(thrust_per_motor[2]),
            "motor.thrust.m4": float(thrust_per_motor[3]),
            "motor.thrust.total": float(total_thrust),
        }
        
        self.log_data.append(entry)
        
        # Update previous values for next acceleration computation
        self.prev_vel = vel.copy()
        self.prev_time = current_time
        
    def save_to_csv(self, filename: str) -> Optional[str]:
        """Save logged data to CSV file.
        
        Args:
            filename: Path to save the CSV file
            
        Returns:
            The filename where data was saved, or None if no data
        """
        if not self.log_data:
            print("Warning: No flight data to save.")
            return None
        
        # Use SHARED_CSV_FIELDS order so sim/real CSVs are directly comparable
        fieldnames = SHARED_CSV_FIELDS
        
        with open(filename, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.log_data)
            
        return filename
    
    def get_summary_stats(self) -> Dict[str, float]:
        """Compute summary statistics from logged data.
        
        Returns:
            Dictionary of summary statistics
        """
        if not self.log_data:
            return {}
        
        # Extract columns as arrays (no pandas dependency)
        def get_col(key: str) -> np.ndarray:
            return np.array([row[key] for row in self.log_data])
        
        z_vals = get_col("stateEstimate.z")
        x_vals = get_col("stateEstimate.x")
        y_vals = get_col("stateEstimate.y")
        roll_vals = get_col("stabilizer.roll")
        pitch_vals = get_col("stabilizer.pitch")
        yaw_vals = get_col("stabilizer.yaw")
        gyro_x = get_col("gyro.x")
        gyro_y = get_col("gyro.y")
        gyro_z = get_col("gyro.z")
        rpm_m1 = get_col("motor.m1")
        rpm_m2 = get_col("motor.m2")
        rpm_m3 = get_col("motor.m3")
        rpm_m4 = get_col("motor.m4")
        thrust_total = get_col("motor.thrust.total")
        
        stats = {
            # Position stats
            "avg_height": float(np.mean(z_vals)),
            "height_std": float(np.std(z_vals)),
            "max_xy_drift": float(np.max(np.sqrt(x_vals**2 + y_vals**2))),
            # Attitude stats
            "avg_roll": float(np.mean(roll_vals)),
            "avg_pitch": float(np.mean(pitch_vals)),
            "avg_yaw": float(np.mean(yaw_vals)),
            "max_roll": float(np.max(np.abs(roll_vals))),
            "max_pitch": float(np.max(np.abs(pitch_vals))),
            # Angular velocity stats
            "avg_gyro_x": float(np.mean(gyro_x)),
            "avg_gyro_y": float(np.mean(gyro_y)),
            "avg_gyro_z": float(np.mean(gyro_z)),
            "max_gyro": float(max(np.max(np.abs(gyro_x)), np.max(np.abs(gyro_y)), np.max(np.abs(gyro_z)))),
            # Motor stats
            "avg_rpm": float(np.mean((rpm_m1 + rpm_m2 + rpm_m3 + rpm_m4) / 4)),
            "avg_total_thrust": float(np.mean(thrust_total)),
            "rpm_std": float(np.std(np.stack([rpm_m1, rpm_m2, rpm_m3, rpm_m4]))),
        }
        
        return stats


# ==============================================================================
# Hover Mode
# ==============================================================================

def run_hover_eval(checkpoint_path: str, num_envs: int, duration: float, 
                   output_dir: str, no_plot: bool, target_z: float = 1.0,
                   hold_time: float = 5.0):
    """Run hover evaluation with specified checkpoint.
    
    Uses the same PointNav environment as training, with a fixed goal at (0, 0, target_z)
    to evaluate hovering stability. The drone will fly to the target height and hover
    there for hold_time seconds after reaching it.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        num_envs: Number of parallel environments
        duration: Max evaluation duration in seconds
        output_dir: Directory to save output files
        no_plot: If True, skip plot generation
        target_z: Target hover height in meters
        hold_time: Time to hold at target after reaching it (default: 5.0s)
    """
    
    # Use same environment as training (pointnav) for checkpoint compatibility
    from train_pointnav import CrazyfliePointNavEnvCfg, CrazyfliePointNavEnv, L2FActorNetwork
    
    # Configure environment
    cfg = CrazyfliePointNavEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = duration + 1.0  # Ensure episode is long enough
    
    # Fixed goal at (0, 0, target_z) for hover mode
    fixed_goal = (0.0, 0.0, target_z)
    
    print(f"\n{'='*60}")
    print(f"HOVER EVALUATION (using PointNav environment)")
    print(f"{'='*60}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Environments: {num_envs}")
    print(f"  Duration: {duration}s")
    print(f"  Target Position: ({fixed_goal[0]:.2f}, {fixed_goal[1]:.2f}, {fixed_goal[2]:.2f})m")
    print(f"  Hold Time: {hold_time}s after reaching target")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Create environment
    env = CrazyfliePointNavEnv(cfg)
    
    # Create and load actor network
    obs_dim = cfg.observation_space
    action_dim = cfg.action_space
    
    actor = L2FActorNetwork(obs_dim, 64, action_dim).to(env.device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=env.device)
    actor.load_state_dict(checkpoint["actor"])
    
    # Load observation normalization stats if both are available
    if "obs_mean" in checkpoint and "obs_std" in checkpoint:
        obs_mean = checkpoint["obs_mean"].to(env.device)
        obs_std = checkpoint["obs_std"].to(env.device)
    else:
        # No normalization - use identity transform
        obs_mean = torch.zeros(obs_dim, device=env.device)
        obs_std = torch.ones(obs_dim, device=env.device)
    
    actor.eval()
    
    # Initialize logger
    logger = ExtendedFlightDataLogger(num_envs)
    
    # Reset environment
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    
    # Set fixed hover goal for all environments
    env._goal_pos[:, 0] = fixed_goal[0]
    env._goal_pos[:, 1] = fixed_goal[1]
    env._goal_pos[:, 2] = fixed_goal[2]
    print(f"  Fixed goal set to: ({fixed_goal[0]:.2f}, {fixed_goal[1]:.2f}, {fixed_goal[2]:.2f})m")
    
    # Compute number of steps
    dt = cfg.sim.dt
    num_steps = int(duration / dt)
    hold_steps = int(hold_time / dt)
    
    # Target reaching criteria (within 10cm of 3D goal position)
    goal_threshold = 0.10  # 10cm tolerance
    
    # Track target reaching and hold state for each environment
    target_reached = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    hold_counter = torch.zeros(num_envs, dtype=torch.int32, device=env.device)
    hold_complete = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    time_to_reach = torch.full((num_envs,), float('nan'), device=env.device)
    
    print(f"Running evaluation for up to {num_steps} steps (or until {hold_time}s hold complete)...")
    
    step_count = 0
    try:
        while simulation_app.is_running() and step_count < num_steps:
            # Normalize observation
            obs_norm = (obs - obs_mean) / (obs_std + 1e-8)
            
            # Get action
            with torch.no_grad():
                action = actor.get_action(obs_norm, deterministic=True)
            
            # Step environment
            obs_dict, reward, terminated, truncated, info = env.step(action)
            obs = obs_dict["policy"]
            
            # Log data
            logger.log_step(env, dt, target=fixed_goal)
            
            # Check target reaching (3D distance to fixed goal)
            drone_pos = env._robot.data.root_pos_w
            goal_pos = env._goal_pos
            dist_3d = torch.sqrt(
                (drone_pos[:, 0] - goal_pos[:, 0])**2 + 
                (drone_pos[:, 1] - goal_pos[:, 1])**2 +
                (drone_pos[:, 2] - goal_pos[:, 2])**2
            )
            at_target = dist_3d < goal_threshold
            
            # Track first time reaching target
            newly_reached = at_target & ~target_reached
            if newly_reached.any():
                time_to_reach[newly_reached] = step_count * dt
                target_reached = target_reached | at_target
            
            # Increment hold counter for envs at target
            hold_counter[at_target] += 1
            hold_counter[~at_target] = 0  # Reset if drone drifts away
            
            # Check if hold is complete
            newly_complete = (hold_counter >= hold_steps) & ~hold_complete
            hold_complete = hold_complete | (hold_counter >= hold_steps)
            
            step_count += 1
            
            # Progress update
            if step_count % 100 == 0:
                elapsed = step_count * dt
                pct_reached = target_reached.float().mean().item() * 100
                pct_holding = hold_complete.float().mean().item() * 100
                print(f"  Step {step_count:5d}/{num_steps} | Time: {elapsed:.2f}s | At target: {pct_reached:.0f}% | Hold complete: {pct_holding:.0f}%")
            
            # End early if all envs have completed hold
            if hold_complete.all():
                print(f"\n  All {num_envs} environments completed {hold_time}s hold at target!")
                break
                
    except KeyboardInterrupt:
        print("\nEvaluation stopped by user")
    
    # Report reaching statistics
    reached_count = target_reached.sum().item()
    complete_count = hold_complete.sum().item()
    avg_reach_time = time_to_reach[target_reached].mean().item() if reached_count > 0 else float('nan')
    
    print(f"\n  Targets reached: {reached_count}/{num_envs} ({reached_count/num_envs*100:.1f}%)")
    print(f"  Hold completed: {complete_count}/{num_envs} ({complete_count/num_envs*100:.1f}%)")
    if not math.isnan(avg_reach_time):
        print(f"  Avg time to reach: {avg_reach_time:.2f}s")
    
    # Save data
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "hover_eval_data.csv")
    logger.save_to_csv(csv_path)
    print(f"\nData saved to: {csv_path}")
    
    # Print summary stats
    stats = logger.get_summary_stats()
    print(f"\n{'='*40}")
    print(f"SUMMARY STATISTICS (averaged over {num_envs} envs)")
    print(f"{'='*40}")
    print(f"  Height: {stats.get('avg_height', 0):.4f}m (std: {stats.get('height_std', 0):.4f})")
    print(f"  Max XY drift: {stats.get('max_xy_drift', 0):.4f}m")
    print(f"  Max roll: {stats.get('max_roll', 0):.2f} deg")
    print(f"  Max pitch: {stats.get('max_pitch', 0):.2f} deg")
    print(f"  Max gyro: {stats.get('max_gyro', 0):.2f} deg/s")
    print(f"  Avg motor RPM: {stats.get('avg_rpm', 0):.0f}")
    print(f"  Avg total thrust: {stats.get('avg_total_thrust', 0)*1000:.2f}mN")
    print(f"{'='*40}\n")
    
    # Generate plots using flight_eval_utils
    if not no_plot:
        plot_flight_data(csv_path, title_prefix="Hover Evaluation", output_dir=output_dir)
        plot_motor_data(csv_path, title_prefix="Hover Evaluation", output_dir=output_dir,
                       hover_rpm=L2FConstants.hover_rpm(), mass=L2FConstants.MASS, gravity=L2FConstants.GRAVITY)
    
    # Cleanup
    env.close()
    
    return stats


# ==============================================================================
# Point Navigation Mode
# ==============================================================================

def run_pointnav_eval(checkpoint_path: str, num_envs: int, duration: float,
                      output_dir: str, no_plot: bool,
                      target_x: Optional[float] = None,
                      target_y: Optional[float] = None,
                      goal_z: Optional[float] = None,
                      hold_time: float = 5.0):
    """Run point navigation evaluation with specified checkpoint.
    
    The drone will navigate to the goal and hover there for hold_time seconds
    after reaching it to demonstrate stability.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        num_envs: Number of parallel environments
        duration: Max evaluation duration in seconds
        output_dir: Directory to save output files
        no_plot: If True, skip plot generation
        target_x: Fixed goal X position in meters (None for random)
        target_y: Fixed goal Y position in meters (None for random)
        goal_z: Fixed goal Z height in meters (None for default)
        hold_time: Time to hold at goal after reaching it (default: 5.0s)
    """
    
    # Import pointnav environment
    from train_pointnav import CrazyfliePointNavEnvCfg, CrazyfliePointNavEnv, L2FActorNetwork
    
    # Configure environment
    cfg = CrazyfliePointNavEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = duration + 1.0
    
    # Use fixed goal if specified
    use_fixed_goal = target_x is not None and target_y is not None
    fixed_goal_pos = None
    if use_fixed_goal:
        fixed_z = goal_z if goal_z is not None else cfg.goal_height
        fixed_goal_pos = (target_x, target_y, fixed_z)
    
    print(f"\n{'='*60}")
    print(f"POINT NAVIGATION EVALUATION")
    print(f"{'='*60}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Environments: {num_envs}")
    print(f"  Duration: {duration}s")
    if use_fixed_goal:
        print(f"  Fixed Goal: ({target_x:.2f}, {target_y:.2f}, {fixed_goal_pos[2]:.2f})m")
    else:
        print(f"  Goal Mode: Random (within {cfg.goal_max_distance}m)")
    print(f"  Hold Time: {hold_time}s after reaching goal")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Create environment
    env = CrazyfliePointNavEnv(cfg)
    
    # Store fixed goal for later use
    env._eval_fixed_goal = fixed_goal_pos
    
    # Create and load actor network
    obs_dim = cfg.observation_space
    action_dim = cfg.action_space
    
    actor = L2FActorNetwork(obs_dim, 64, action_dim).to(env.device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=env.device)
    actor.load_state_dict(checkpoint["actor"])
    
    # Load observation normalization stats if both are available
    if "obs_mean" in checkpoint and "obs_std" in checkpoint:
        obs_mean = checkpoint["obs_mean"].to(env.device)
        obs_std = checkpoint["obs_std"].to(env.device)
    else:
        # No normalization - use identity transform
        obs_mean = torch.zeros(obs_dim, device=env.device)
        obs_std = torch.ones(obs_dim, device=env.device)
    
    actor.eval()
    
    # Initialize logger
    logger = ExtendedFlightDataLogger(num_envs)
    
    # Reset environment
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    
    # Override goal positions if fixed goal specified
    if fixed_goal_pos is not None:
        env._goal_pos[:, 0] = fixed_goal_pos[0]
        env._goal_pos[:, 1] = fixed_goal_pos[1]
        env._goal_pos[:, 2] = fixed_goal_pos[2]
        # Update previous distance for progress reward
        drone_pos = env._robot.data.root_pos_w
        dist_xy = torch.sqrt((drone_pos[:, 0] - env._goal_pos[:, 0])**2 + 
                             (drone_pos[:, 1] - env._goal_pos[:, 1])**2)
        env._prev_dist_xy = dist_xy
        print(f"  Fixed goal set to: ({fixed_goal_pos[0]:.2f}, {fixed_goal_pos[1]:.2f}, {fixed_goal_pos[2]:.2f})m")
    
    # Compute number of steps
    dt = cfg.sim.dt
    num_steps = int(duration / dt)
    hold_steps = int(hold_time / dt)
    
    # Goal reaching criteria (within 10cm of goal)
    goal_threshold = 0.10  # 10cm tolerance
    
    # Track goal reaching and hold state for each environment
    goal_reached = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    hold_counter = torch.zeros(num_envs, dtype=torch.int32, device=env.device)
    hold_complete = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    time_to_reach = torch.full((num_envs,), float('nan'), device=env.device)
    
    print(f"Running evaluation for up to {num_steps} steps (or until {hold_time}s hold complete)...")
    
    step_count = 0
    try:
        while simulation_app.is_running() and step_count < num_steps:
            # Normalize observation
            obs_norm = (obs - obs_mean) / (obs_std + 1e-8)
            
            # Get action
            with torch.no_grad():
                action = actor.get_action(obs_norm, deterministic=True)
            
            # Step environment
            obs_dict, reward, terminated, truncated, info = env.step(action)
            obs = obs_dict["policy"]
            
            # Log data - get current goal position from environment
            avg_goal = env._goal_pos.mean(dim=0).cpu().numpy()
            logger.log_step(env, dt, target=(float(avg_goal[0]), float(avg_goal[1]), float(avg_goal[2])))
            
            # Check goal reaching (3D distance)
            drone_pos = env._robot.data.root_pos_w
            goal_pos = env._goal_pos
            dist_3d = torch.sqrt(
                (drone_pos[:, 0] - goal_pos[:, 0])**2 + 
                (drone_pos[:, 1] - goal_pos[:, 1])**2 +
                (drone_pos[:, 2] - goal_pos[:, 2])**2
            )
            at_goal = dist_3d < goal_threshold
            
            # Track first time reaching goal
            newly_reached = at_goal & ~goal_reached
            if newly_reached.any():
                time_to_reach[newly_reached] = step_count * dt
                goal_reached = goal_reached | at_goal
            
            # Increment hold counter for envs at goal
            hold_counter[at_goal] += 1
            hold_counter[~at_goal] = 0  # Reset if drone drifts away from goal
            
            # Check if hold is complete
            hold_complete = hold_complete | (hold_counter >= hold_steps)
            
            step_count += 1
            
            # Progress update
            if step_count % 100 == 0:
                elapsed = step_count * dt
                pct_reached = goal_reached.float().mean().item() * 100
                pct_holding = hold_complete.float().mean().item() * 100
                print(f"  Step {step_count:5d}/{num_steps} | Time: {elapsed:.2f}s | At goal: {pct_reached:.0f}% | Hold complete: {pct_holding:.0f}%")
            
            # End early if all envs have completed hold
            if hold_complete.all():
                print(f"\n  All {num_envs} environments completed {hold_time}s hold at goal!")
                break
                
    except KeyboardInterrupt:
        print("\nEvaluation stopped by user")
    
    # Report reaching statistics
    reached_count = goal_reached.sum().item()
    complete_count = hold_complete.sum().item()
    avg_reach_time = time_to_reach[goal_reached].mean().item() if reached_count > 0 else float('nan')
    
    print(f"\n  Goals reached: {reached_count}/{num_envs} ({reached_count/num_envs*100:.1f}%)")
    print(f"  Hold completed: {complete_count}/{num_envs} ({complete_count/num_envs*100:.1f}%)")
    if not math.isnan(avg_reach_time):
        print(f"  Avg time to reach: {avg_reach_time:.2f}s")
    
    # Save data
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "pointnav_eval_data.csv")
    logger.save_to_csv(csv_path)
    print(f"\nData saved to: {csv_path}")
    
    # Print summary stats
    stats = logger.get_summary_stats()

    print(f"\n{'='*40}")
    print(f"SUMMARY STATISTICS (averaged over {num_envs} envs)")
    print(f"{'='*40}")
    print(f"  Goal reach rate: {reached_count/num_envs*100:.1f}% ({reached_count}/{num_envs})")
    print(f"  Hold success rate: {complete_count/num_envs*100:.1f}% ({complete_count}/{num_envs})")
    print(f"  Height: {stats.get('avg_height', 0):.4f}m (std: {stats.get('height_std', 0):.4f})")
    print(f"  Max XY drift: {stats.get('max_xy_drift', 0):.4f}m")
    print(f"  Max roll: {stats.get('max_roll', 0):.2f} deg")
    print(f"  Max pitch: {stats.get('max_pitch', 0):.2f} deg")
    print(f"  Max gyro: {stats.get('max_gyro', 0):.2f} deg/s")
    print(f"  Avg motor RPM: {stats.get('avg_rpm', 0):.0f}")
    print(f"  Avg total thrust: {stats.get('avg_total_thrust', 0)*1000:.2f}mN")
    print(f"{'='*40}\n")
    
    # Generate plots using flight_eval_utils
    if not no_plot:
        plot_flight_data(csv_path, title_prefix="Point Navigation Evaluation", output_dir=output_dir)
        plot_motor_data(csv_path, title_prefix="Point Navigation Evaluation", output_dir=output_dir,
                       hover_rpm=L2FConstants.hover_rpm(), mass=L2FConstants.MASS, gravity=L2FConstants.GRAVITY)
    
    # Cleanup
    env.close()
    
    return stats


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Use pointnav checkpoint for both modes (it handles both hover and navigation)
        checkpoint_path = os.path.join(script_dir, "checkpoints_pointnav", "best_model.pt")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Determine output directory (default: play_eval_results/{mode}_{timestamp} in script directory)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(script_dir, "play_eval_results", f"{args.mode}_{timestamp}")
    
    # Run evaluation
    if args.mode == "hover":
        stats = run_hover_eval(
            checkpoint_path=checkpoint_path,
            num_envs=args.num_envs,
            duration=args.duration,
            output_dir=output_dir,
            no_plot=args.no_plot,
            target_z=args.target_z,
            hold_time=args.hold_time
        )
    else:  # pointnav
        stats = run_pointnav_eval(
            checkpoint_path=checkpoint_path,
            num_envs=args.num_envs,
            duration=args.duration,
            output_dir=output_dir,
            no_plot=args.no_plot,
            target_x=args.target_x,
            target_y=args.target_y,
            goal_z=args.goal_z,
            hold_time=args.hold_time
        )
    
    print(f"\nEvaluation complete!")
    print(f"Results saved to: {output_dir}")
    
    # Cleanup
    simulation_app.close()


if __name__ == "__main__":
    main()
