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
    import pandas as pd
    
    df = pd.read_csv(filename)
    
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
    plt.ylabel("Acceleration [m/s²]")
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
    ax3.set_ylabel("Acceleration [m/s²]")
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
        
    def reset(self):
        """Reset the logger for a new flight session."""
        self.log_data = []
        self.t0 = time.perf_counter()
        self.prev_vel = None
        self.prev_time = None
        
    def log_step(self, env, env_idx: int = 0):
        """Log data for a single timestep.
        
        Args:
            env: The environment instance (must have _robot attribute)
            env_idx: Index of the environment to log (default: 0, first environment)
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
        
        # Store data in format matching real-world deployment
        self.log_data.append({
            "t": current_time,
            "stateEstimate.x": pos[0],
            "stateEstimate.y": pos[1],
            "stateEstimate.z": pos[2],
            "acc.x": acc[0],
            "acc.y": acc[1],
            "acc.z": acc[2],
            "gyro.x": ang_vel[0] * 180.0 / np.pi,  # Convert to deg/s
            "gyro.y": ang_vel[1] * 180.0 / np.pi,
            "gyro.z": ang_vel[2] * 180.0 / np.pi,
            "stabilizer.roll": euler[0],
            "stabilizer.pitch": euler[1],
            "stabilizer.yaw": euler[2],
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
            "stabilizer.roll", "stabilizer.pitch", "stabilizer.yaw"
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
