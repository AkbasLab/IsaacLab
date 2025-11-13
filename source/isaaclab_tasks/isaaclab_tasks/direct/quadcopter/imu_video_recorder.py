# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IMU data video recorder for quadcopter training visualization."""

from __future__ import annotations

import os
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch

# Use imageio for video creation (already available in env)
try:
    import imageio
    VIDEO_AVAILABLE = True
except ImportError:
    VIDEO_AVAILABLE = False
    print("[WARNING] imageio not installed. Video recording disabled.")


class IMUVideoRecorder:
    """Records IMU data and creates a 4-panel video visualization matching real Crazyflie telemetry format.
    
    Creates a single MP4 video with 4 subplots:
    1. Position (x, y, z)
    2. Attitude (roll, pitch, yaw)
    3. Accelerometer (ax, ay, az)
    4. Gyroscope (wx, wy, wz)
    """
    
    def __init__(
        self,
        output_dir: str,
        env_index: int = 0,
        max_samples: int = 15000,
        fps: int = 50,
        video_name: str = "imu_data_visualization.mp4"
    ):
        """Initialize the IMU video recorder.
        
        Args:
            output_dir: Directory to save the video
            env_index: Which environment to record (default: 0)
            max_samples: Maximum number of timesteps to record
            fps: Frames per second for the output video
            video_name: Name of the output video file
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.env_index = env_index
        self.max_samples = max_samples
        self.fps = fps
        self.video_path = self.output_dir / video_name
        
        # Data buffers (using deque for efficient append/pop)
        self.time_data = deque(maxlen=max_samples)
        self.pos_x = deque(maxlen=max_samples)
        self.pos_y = deque(maxlen=max_samples)
        self.pos_z = deque(maxlen=max_samples)
        self.roll = deque(maxlen=max_samples)
        self.pitch = deque(maxlen=max_samples)
        self.yaw = deque(maxlen=max_samples)
        self.acc_x = deque(maxlen=max_samples)
        self.acc_y = deque(maxlen=max_samples)
        self.acc_z = deque(maxlen=max_samples)
        self.gyro_x = deque(maxlen=max_samples)
        self.gyro_y = deque(maxlen=max_samples)
        self.gyro_z = deque(maxlen=max_samples)
        
        self.current_time = 0.0
        self.frame_buffer = []  # Store matplotlib figures as images
        
        # Create figure with 4 subplots
        self.fig, self.axes = plt.subplots(2, 2, figsize=(16, 12))
        self.fig.suptitle('IMU Data Visualization (Simulation)', fontsize=16, fontweight='bold')
        
        print(f"[IMU Video Recorder] Initialized. Will save to: {self.video_path}")
        print(f"[IMU Video Recorder] Recording environment {env_index} for up to {max_samples} timesteps")
        if not VIDEO_AVAILABLE:
            print("[IMU Video Recorder] WARNING: imageio not available, video will not be saved!")
    
    def record_step(
        self,
        dt: float,
        pos_w: torch.Tensor,
        euler_angles: torch.Tensor,
        lin_acc_b: torch.Tensor,
        ang_vel_b: torch.Tensor
    ):
        """Record one timestep of IMU data.
        
        Args:
            dt: Time step size in seconds
            pos_w: World position (num_envs, 3)
            euler_angles: Euler angles in radians (num_envs, 3) - roll, pitch, yaw
            lin_acc_b: Linear acceleration in body frame (num_envs, 3)
            ang_vel_b: Angular velocity in body frame (num_envs, 3)
        """
        # Extract data for the selected environment
        idx = self.env_index
        
        # Update time
        self.current_time += dt
        self.time_data.append(self.current_time)
        
        # Position (m)
        self.pos_x.append(pos_w[idx, 0].item())
        self.pos_y.append(pos_w[idx, 1].item())
        self.pos_z.append(pos_w[idx, 2].item())
        
        # Attitude (convert radians to degrees for visualization)
        self.roll.append(np.degrees(euler_angles[idx, 0].item()))
        self.pitch.append(np.degrees(euler_angles[idx, 1].item()))
        self.yaw.append(np.degrees(euler_angles[idx, 2].item()))
        
        # Accelerometer (m/s²)
        self.acc_x.append(lin_acc_b[idx, 0].item())
        self.acc_y.append(lin_acc_b[idx, 1].item())
        self.acc_z.append(lin_acc_b[idx, 2].item())
        
        # Gyroscope (convert rad/s to deg/s for visualization)
        self.gyro_x.append(np.degrees(ang_vel_b[idx, 0].item()))
        self.gyro_y.append(np.degrees(ang_vel_b[idx, 1].item()))
        self.gyro_z.append(np.degrees(ang_vel_b[idx, 2].item()))
    
    def generate_frame(self):
        """Generate a single frame image from current data."""
        if len(self.time_data) < 2:
            return  # Need at least 2 samples to plot
        
        # Convert deques to numpy arrays
        t = np.array(self.time_data)
        
        # Clear all subplots
        for ax in self.axes.flat:
            ax.clear()
        
        # Plot 1: Position
        ax1 = self.axes[0, 0]
        ax1.plot(t, self.pos_x, 'r-', label='x (m)', linewidth=2)
        ax1.plot(t, self.pos_y, 'g-', label='y (m)', linewidth=2)
        ax1.plot(t, self.pos_z, 'b-', label='z (m)', linewidth=2)
        ax1.set_title('Drone Position over Time', fontweight='bold')
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Position [m]')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Attitude (roll, pitch, yaw in degrees)
        ax2 = self.axes[0, 1]
        ax2.plot(t, self.roll, 'r-', label='Roll (°)', linewidth=2)
        ax2.plot(t, self.pitch, 'g-', label='Pitch (°)', linewidth=2)
        ax2.plot(t, self.yaw, 'b-', label='Yaw (°)', linewidth=2)
        ax2.set_title('Attitude (Orientation) over Time', fontweight='bold')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Angle [°]')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Accelerometer
        ax3 = self.axes[1, 0]
        ax3.plot(t, self.acc_x, 'r-', label='ax', linewidth=2)
        ax3.plot(t, self.acc_y, 'g-', label='ay', linewidth=2)
        ax3.plot(t, self.acc_z, 'b-', label='az', linewidth=2)
        ax3.set_title('Accelerometer Data', fontweight='bold')
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Acceleration [m/s²]')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Gyroscope (in degrees/s for easy comparison with Crazyflie)
        ax4 = self.axes[1, 1]
        ax4.plot(t, self.gyro_x, 'r-', label='wx', linewidth=2)
        ax4.plot(t, self.gyro_y, 'g-', label='wy', linewidth=2)
        ax4.plot(t, self.gyro_z, 'b-', label='wz', linewidth=2)
        ax4.set_title('Gyroscope Data', fontweight='bold')
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('Angular velocity [°/s]')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        
        # Adjust layout
        self.fig.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Convert matplotlib figure to numpy array
        self.fig.canvas.draw()
        # Use buffer_rgba() for newer matplotlib versions instead of deprecated tostring_rgb()
        img = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (4,))
        # Convert RGBA to RGB (drop alpha channel)
        img = img[:, :, :3]
        
        self.frame_buffer.append(img)
    
    def save_video(self):
        """Save all recorded frames as an MP4 video using imageio."""
        if not VIDEO_AVAILABLE:
            print("[IMU Video Recorder] Cannot save video - imageio not installed")
            return
        
        if len(self.frame_buffer) == 0:
            print("[IMU Video Recorder] No frames to save")
            return
        
        print(f"[IMU Video Recorder] Saving {len(self.frame_buffer)} frames to video...")
        
        try:
            # Write video using imageio with ffmpeg backend
            with imageio.get_writer(
                str(self.video_path),
                fps=self.fps,
                codec='libx264',
                pixelformat='yuv420p',
                macro_block_size=1
            ) as writer:
                for frame in self.frame_buffer:
                    writer.append_data(frame)
            
            print(f"[IMU Video Recorder] ✅ Video saved successfully: {self.video_path}")
            print(f"[IMU Video Recorder] Duration: {len(self.frame_buffer) / self.fps:.2f} seconds")
            print(f"[IMU Video Recorder] Total frames: {len(self.frame_buffer)}")
            
        except Exception as e:
            print(f"[IMU Video Recorder] ❌ Error saving video: {e}")
            print(f"[IMU Video Recorder] Attempting fallback to GIF format...")
            try:
                # Fallback: save as GIF if MP4 fails
                gif_path = self.video_path.with_suffix('.gif')
                imageio.mimsave(str(gif_path), self.frame_buffer, fps=self.fps)
                print(f"[IMU Video Recorder] ✅ Saved as GIF instead: {gif_path}")
            except Exception as e2:
                print(f"[IMU Video Recorder] ❌ GIF fallback also failed: {e2}")
        
        finally:
            plt.close(self.fig)
    
    def __del__(self):
        """Cleanup when recorder is destroyed."""
        if hasattr(self, 'fig'):
            plt.close(self.fig)
