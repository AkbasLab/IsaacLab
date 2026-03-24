"""
flight_eval_utils.py

Utility functions and classes for logging and plotting flight evaluation data.
Matches the format used in real-world Crazyflie deployment for easy comparison.
"""

import csv
import os
import time
from typing import Optional
import numpy as np
import torch
import matplotlib.pyplot as plt


def quat_to_euler(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion [w,x,y,z] to Euler angles [roll,pitch,yaw] in degrees.
    
    Args:
        quat: Quaternion tensor of shape [..., 4] with ordering [w, x, y, z]
        
    Returns:
        Euler angles tensor of shape [..., 3] with [roll, pitch, yaw] in degrees
    """
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = torch.where(torch.abs(sinp) >= 1,
                       torch.sign(sinp) * torch.pi / 2,
                       torch.asin(sinp))
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    
    # Convert to degrees
    return torch.stack([roll, pitch, yaw], dim=-1) * 180.0 / torch.pi


def load_csv_as_dict(filename: str) -> dict:
    """Load CSV data into a dictionary of numpy arrays (no pandas dependency).
    
    Args:
        filename: Path to CSV file
        
    Returns:
        Dictionary mapping column names to numpy arrays
    """
    data = {}
    with open(filename, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if key not in data:
                    data[key] = []
                data[key].append(float(value))
    return {key: np.array(values) for key, values in data.items()}


def plot_flight_data(filename: str, title_prefix: str = "Flight", output_dir: Optional[str] = None):
    """Plot flight data from CSV file and save as JPG images.
    
    Creates four plots matching the real-world deployment visualization:
    1. Position (x, y, z) over time
    2. Attitude (roll, pitch, yaw) over time
    3. Accelerometer data
    4. Gyroscope data
    
    Args:
        filename: Path to CSV file containing flight data
        title_prefix: Prefix for plot titles (e.g., "Hover", "Point Navigation")
        output_dir: Directory to save plots as JPG files. If None, plots are displayed.
    """
    df = load_csv_as_dict(filename)
    
    t = df["t"]
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # --- Plot position (x, y, z) ---
    plt.figure(figsize=(10, 6))
    plt.plot(t, df["stateEstimate.x"], label="x (m)")
    plt.plot(t, df["stateEstimate.y"], label="y (m)")
    plt.plot(t, df["stateEstimate.z"], label="z (m)")
    plt.title(f"{title_prefix} - Drone Position over Time")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.legend()
    plt.grid(True)
    if output_dir:
        plot_path = os.path.join(output_dir, f"{title_prefix.lower().replace(' ', '_')}_position.jpg")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # --- Plot attitude (roll, pitch, yaw) ---
    plt.figure(figsize=(10, 6))
    plt.plot(t, df["stabilizer.roll"], label="Roll (°)")
    plt.plot(t, df["stabilizer.pitch"], label="Pitch (°)")
    plt.plot(t, df["stabilizer.yaw"], label="Yaw (°)")
    plt.title(f"{title_prefix} - Attitude (Orientation) over Time")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [°]")
    plt.legend()
    plt.grid(True)
    if output_dir:
        plot_path = os.path.join(output_dir, f"{title_prefix.lower().replace(' ', '_')}_attitude.jpg")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # --- Plot accelerations ---
    plt.figure(figsize=(10, 6))
    plt.plot(t, df["acc.x"], label="ax")
    plt.plot(t, df["acc.y"], label="ay")
    plt.plot(t, df["acc.z"], label="az")
    plt.title(f"{title_prefix} - Accelerometer Data")
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [g]")
    plt.legend()
    plt.grid(True)
    if output_dir:
        plot_path = os.path.join(output_dir, f"{title_prefix.lower().replace(' ', '_')}_accelerometer.jpg")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # --- Plot gyro ---
    plt.figure(figsize=(10, 6))
    plt.plot(t, df["gyro.x"], label="wx")
    plt.plot(t, df["gyro.y"], label="wy")
    plt.plot(t, df["gyro.z"], label="wz")
    plt.title(f"{title_prefix} - Gyroscope Data")
    plt.xlabel("Time [s]")
    plt.ylabel("Angular velocity [°/s]")
    plt.legend()
    plt.grid(True)
    if output_dir:
        plot_path = os.path.join(output_dir, f"{title_prefix.lower().replace(' ', '_')}_gyroscope.jpg")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # --- Combined plot with all four subplots ---
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Position subplot
    ax1.plot(t, df["stateEstimate.x"], label="x (m)")
    ax1.plot(t, df["stateEstimate.y"], label="y (m)")
    ax1.plot(t, df["stateEstimate.z"], label="z (m)")
    ax1.set_title(f"{title_prefix} - Position")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Position [m]")
    ax1.legend()
    ax1.grid(True)
    
    # Attitude subplot
    ax2.plot(t, df["stabilizer.roll"], label="Roll (°)")
    ax2.plot(t, df["stabilizer.pitch"], label="Pitch (°)")
    ax2.plot(t, df["stabilizer.yaw"], label="Yaw (°)")
    ax2.set_title(f"{title_prefix} - Attitude")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Angle [°]")
    ax2.legend()
    ax2.grid(True)
    
    # Accelerometer subplot
    ax3.plot(t, df["acc.x"], label="ax")
    ax3.plot(t, df["acc.y"], label="ay")
    ax3.plot(t, df["acc.z"], label="az")
    ax3.set_title(f"{title_prefix} - Accelerometer")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Acceleration [g]")
    ax3.legend()
    ax3.grid(True)
    
    # Gyroscope subplot
    ax4.plot(t, df["gyro.x"], label="wx")
    ax4.plot(t, df["gyro.y"], label="wy")
    ax4.plot(t, df["gyro.z"], label="wz")
    ax4.set_title(f"{title_prefix} - Gyroscope")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Angular velocity [°/s]")
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    if output_dir:
        plot_path = os.path.join(output_dir, f"{title_prefix.lower().replace(' ', '_')}_combined.jpg")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


class FlightDataLogger:
    """Logger for collecting flight evaluation data at ~100Hz.
    
    Collects position, velocity, acceleration, attitude, and angular velocity data
    in a format compatible with real-world Crazyflie deployment for easy comparison.
    """
    
    def __init__(self):
        """Initialize the flight data logger."""
        self.log_data = []
        self.t0 = time.perf_counter()
        self.prev_vel = None
        self.prev_time = None
        self.gravity = 9.81
        
    def reset(self):
        """Reset the logger for a new flight session."""
        self.log_data = []
        self.t0 = time.perf_counter()
        self.prev_vel = None
        self.prev_time = None
        
    def log_step(self, env, env_idx: int = 0, target: Optional[tuple[float, float, float, float]] = None):
        """Log data for a single timestep.
        
        Args:
            env: The environment instance (must have _robot attribute)
            env_idx: Index of the environment to log (default: 0, first environment)
            target: Optional target tuple (x, y, z, yaw_deg). If not provided,
                tries to read env._goal_pos and uses NaN yaw.
        """
        current_time = time.perf_counter() - self.t0
        
        # Get state data from environment
        pos = env._robot.data.root_pos_w[env_idx].cpu().numpy()
        vel = env._robot.data.root_lin_vel_w[env_idx].cpu().numpy()
        ang_vel = env._robot.data.root_ang_vel_w[env_idx].cpu().numpy()
        quat = env._robot.data.root_quat_w[env_idx].cpu()
        euler = quat_to_euler(quat.unsqueeze(0)).squeeze(0).numpy()
        
        # Compute acceleration (derivative of velocity)
        if self.prev_vel is not None and self.prev_time is not None:
            dt = current_time - self.prev_time
            acc = (vel - self.prev_vel) / dt if dt > 0 else np.zeros(3)
        else:
            acc = np.zeros(3)

        # Convert world-frame acceleration estimate to g to match the real CSV.
        acc_g = acc / self.gravity

        if target is not None:
            target_x, target_y, target_z, target_yaw = target
        elif hasattr(env, "_goal_pos"):
            goal = env._goal_pos[env_idx].cpu().numpy()
            target_x, target_y, target_z = float(goal[0]), float(goal[1]), float(goal[2])
            target_yaw = float("nan")
        else:
            target_x = target_y = target_z = target_yaw = float("nan")

        if hasattr(env, "_rpm_state"):
            rpm = env._rpm_state[env_idx].detach().cpu().numpy()
        else:
            rpm = np.full(4, np.nan, dtype=np.float32)

        # Real Crazyflie logs motor.m* as PWM-like values in [0, 65535].
        rpm_max = getattr(env, "_max_rpm", 21702.0)
        pwm = rpm / rpm_max * 65535.0 if np.all(np.isfinite(rpm)) else np.full(4, np.nan, dtype=np.float32)
        
        # Store data in format matching real-world deployment
        self.log_data.append({
            "t": current_time,
            "stateEstimate.x": pos[0],
            "stateEstimate.y": pos[1],
            "stateEstimate.z": pos[2],
            "acc.x": acc_g[0],
            "acc.y": acc_g[1],
            "acc.z": acc_g[2],
            "gyro.x": ang_vel[0] * 180.0 / np.pi,  # Convert to deg/s
            "gyro.y": ang_vel[1] * 180.0 / np.pi,
            "gyro.z": ang_vel[2] * 180.0 / np.pi,
            "stabilizer.roll": euler[0],
            "stabilizer.pitch": euler[1],
            "stabilizer.yaw": euler[2],
            "target.x": target_x,
            "target.y": target_y,
            "target.z": target_z,
            "target.yaw": target_yaw,
            "pm.vbat": float("nan"),
            "motor.m1": pwm[0],
            "motor.m2": pwm[1],
            "motor.m3": pwm[2],
            "motor.m4": pwm[3],
            "motor.rpm.m1": rpm[0],
            "motor.rpm.m2": rpm[1],
            "motor.rpm.m3": rpm[2],
            "motor.rpm.m4": rpm[3],
            "imu.acc_x": acc_g[0],
            "imu.acc_y": acc_g[1],
            "imu.acc_z": acc_g[2],
            "imu.gyro_x": ang_vel[0] * 180.0 / np.pi,
            "imu.gyro_y": ang_vel[1] * 180.0 / np.pi,
            "imu.gyro_z": ang_vel[2] * 180.0 / np.pi,
            "imu.mag_x": 0.0,
            "imu.mag_y": 0.0,
            "imu.mag_z": 0.0,
        })
        
        # Store for next acceleration computation
        self.prev_vel = vel
        self.prev_time = current_time
        
    def save_to_csv(self, filename: str) -> Optional[str]:
        """Save logged data to CSV file.
        
        Args:
            filename: Path to save the CSV file
            
        Returns:
            The filename where data was saved, or None if no data
        """
        if not self.log_data:
            print("⚠️ No flight data to save.")
            return None
            
        fieldnames = [
            "t",
            "stateEstimate.x", "stateEstimate.y", "stateEstimate.z",
            "acc.x", "acc.y", "acc.z",
            "gyro.x", "gyro.y", "gyro.z",
            "stabilizer.roll", "stabilizer.pitch", "stabilizer.yaw",
            "target.x", "target.y", "target.z", "target.yaw",
            "pm.vbat",
            "motor.m1", "motor.m2", "motor.m3", "motor.m4",
            "motor.rpm.m1", "motor.rpm.m2", "motor.rpm.m3", "motor.rpm.m4",
            "imu.acc_x", "imu.acc_y", "imu.acc_z",
            "imu.gyro_x", "imu.gyro_y", "imu.gyro_z",
            "imu.mag_x", "imu.mag_y", "imu.mag_z",
        ]
        
        with open(filename, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(self.log_data)
            
        return filename
        
    def save_and_plot(self, filename: str, title_prefix: str = "Flight", output_dir: Optional[str] = None) -> Optional[str]:
        """Save data to CSV and generate plots as JPG files.
        
        Args:
            filename: Path to save the CSV file
            title_prefix: Prefix for plot titles
            output_dir: Directory to save plot JPG files. If None, plots are not displayed or saved.
            
        Returns:
            The filename where data was saved, or None if no data
        """
        saved_file = self.save_to_csv(filename)
        if saved_file and output_dir:
            try:
                plot_flight_data(saved_file, title_prefix=title_prefix, output_dir=output_dir)
            except Exception as e:
                print(f"Failed to generate plots: {e}")
        return saved_file


def plot_motor_data(filename: str, title_prefix: str = "Flight", output_dir: Optional[str] = None,
                    hover_rpm: float = 7249.0, mass: float = 0.027, gravity: float = 9.81,
                    rpm_max: float = 21702.0):
    """Plot motor PWM and thrust data from CSV file.
    
    Motor columns (motor.m1..m4) are in PWM (0-65535), matching Crazyflie
    firmware output. A hover-PWM reference line is computed from hover_rpm.
    
    Args:
        filename: Path to CSV file containing motor data
        title_prefix: Prefix for plot titles
        output_dir: Directory to save plots. If None, plots are displayed.
        hover_rpm: Expected hover RPM (used to compute hover PWM reference)
        mass: Drone mass in kg for thrust reference
        gravity: Gravitational acceleration in m/s^2
        rpm_max: Maximum RPM (used for RPM → PWM conversion)
    """
    df = load_csv_as_dict(filename)
    t = df["t"]
    
    # Check if motor columns exist (supports both formats: motor.m1 and motor.rpm.m1)
    motor_cols_new = ["motor.m1", "motor.m2", "motor.m3", "motor.m4"]
    motor_cols_old = ["motor.rpm.m1", "motor.rpm.m2", "motor.rpm.m3", "motor.rpm.m4"]
    
    if all(col in df for col in motor_cols_new):
        motor_cols = motor_cols_new
    elif all(col in df for col in motor_cols_old):
        motor_cols = motor_cols_old
    else:
        print("Warning: Motor data columns not found in CSV. Skipping motor plots.")
        return
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # --- Plot Motor PWM ---
    hover_pwm = hover_rpm / rpm_max * 65535.0
    plt.figure(figsize=(12, 6))
    plt.plot(t, df[motor_cols[0]], label="M1 (FR)", linewidth=1)
    plt.plot(t, df[motor_cols[1]], label="M2 (BR)", linewidth=1)
    plt.plot(t, df[motor_cols[2]], label="M3 (BL)", linewidth=1)
    plt.plot(t, df[motor_cols[3]], label="M4 (FL)", linewidth=1)
    plt.axhline(y=hover_pwm, color='r', linestyle='--', label=f'Hover PWM ({hover_pwm:.0f})', alpha=0.5)
    plt.title(f"{title_prefix} - Motor PWM")
    plt.xlabel("Time [s]")
    plt.ylabel("PWM (0–65535)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if output_dir:
        plt.savefig(os.path.join(output_dir, f"{title_prefix.lower().replace(' ', '_')}_motor_pwm.jpg"), 
                    dpi=150, bbox_inches='tight')
        plt.close()
    
    # --- Plot Motor Thrust ---
    thrust_cols = ["motor.thrust.m1", "motor.thrust.m2", "motor.thrust.m3", "motor.thrust.m4"]
    if all(col in df for col in thrust_cols):
        plt.figure(figsize=(12, 6))
        plt.plot(t, df["motor.thrust.m1"] * 1000, label="M1 (FR)", linewidth=1)
        plt.plot(t, df["motor.thrust.m2"] * 1000, label="M2 (BR)", linewidth=1)
        plt.plot(t, df["motor.thrust.m3"] * 1000, label="M3 (BL)", linewidth=1)
        plt.plot(t, df["motor.thrust.m4"] * 1000, label="M4 (FL)", linewidth=1)
        hover_thrust = mass * gravity / 4.0 * 1000  # mN per motor
        plt.axhline(y=hover_thrust, color='r', linestyle='--', 
                    label=f'Hover thrust ({hover_thrust:.1f} mN)', alpha=0.5)
        plt.title(f"{title_prefix} - Motor Thrust")
        plt.xlabel("Time [s]")
        plt.ylabel("Thrust [mN]")
        plt.legend()
        plt.grid(True, alpha=0.3)
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"{title_prefix.lower().replace(' ', '_')}_motor_thrust.jpg"),
                        dpi=150, bbox_inches='tight')
            plt.close()
    
    # --- Plot Total Thrust ---
    if "motor.thrust.total" in df:
        plt.figure(figsize=(12, 6))
        plt.plot(t, df["motor.thrust.total"] * 1000, label="Total Thrust", 
                 color="purple", linewidth=1.5)
        weight = mass * gravity * 1000  # mN
        plt.axhline(y=weight, color='r', linestyle='--', 
                    label=f'Weight ({weight:.1f} mN)', alpha=0.5)
        plt.title(f"{title_prefix} - Total Thrust vs Weight")
        plt.xlabel("Time [s]")
        plt.ylabel("Force [mN]")
        plt.legend()
        plt.grid(True, alpha=0.3)
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"{title_prefix.lower().replace(' ', '_')}_total_thrust.jpg"),
                        dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
