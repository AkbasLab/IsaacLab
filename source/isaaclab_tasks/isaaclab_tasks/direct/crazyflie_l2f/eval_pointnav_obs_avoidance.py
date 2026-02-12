#!/usr/bin/env python3
"""
Evaluation & Visualization Script for Point Nav + Obstacle Avoidance

This script loads a trained checkpoint and runs evaluation episodes,
producing detailed metrics and visualizations:

1. Renders the agent flying in Isaac Sim (headful mode)
2. Logs per-episode: reach/collision/timeout, flight path, proximity readings
3. Generates publication-quality plots:
   - 2D/3D flight trajectories with obstacles
   - Collision rate over episodes
   - Goal reach rate over episodes
   - Proximity heatmap (min distance to obstacles over time)
   - Reward decomposition
   - Episode length distribution
4. Saves a summary CSV & JSON with aggregate statistics

Usage:
    # Default: load best model, 64 envs, render with viewer
    .\\isaaclab.bat -p source\\isaaclab_tasks\\...\\eval_pointnav_obs_avoidance.py

    # Headless batch eval for stats (many envs, many episodes)
    .\\isaaclab.bat -p source\\isaaclab_tasks\\...\\eval_pointnav_obs_avoidance.py --headless --num_envs 512 --num_episodes 200

    # Custom checkpoint
    .\\isaaclab.bat -p source\\isaaclab_tasks\\...\\eval_pointnav_obs_avoidance.py --checkpoint path/to/model.pt --num_envs 16
"""

from __future__ import annotations

import argparse
import os
import sys
import math
import json
import csv
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

# Isaac Sim setup
from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Crazyflie Point Nav + Obstacle Avoidance")

    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint. Default: best_model.pt in checkpoints_pointnav_obs/")
    parser.add_argument("--num_envs", type=int, default=64,
                        help="Number of parallel environments for evaluation")
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="Total episodes to evaluate across all envs")
    parser.add_argument("--record_envs", type=int, default=4,
                        help="Number of envs to record detailed trajectories for")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use deterministic actions (default: True)")
    parser.add_argument("--no_deterministic", dest="deterministic", action="store_false",
                        help="Use stochastic actions")
    parser.add_argument("--term_on_collision", action="store_true", default=True,
                        help="Terminate episode on collision (default: True, matching Phase 4 training)")
    parser.add_argument("--no_term_on_collision", dest="term_on_collision", action="store_false",
                        help="Don't terminate on collision")
    parser.add_argument("--collision_penalty", type=float, default=-30.0,
                        help="Collision penalty (default: -30, matching Phase 4 training)")
    parser.add_argument("--max_obstacles", type=int, default=None,
                        help="Max obstacles per env (default: use env cfg)")
    parser.add_argument("--min_obstacles", type=int, default=None,
                        help="Min obstacles per env (default: use env cfg)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory. Default: eval/pointnav_obs/<timestamp>/")

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    return args


args = parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Isaac Lab imports (must be after AppLauncher)
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import CUBOID_MARKER_CFG

from crazyflie_21_cfg import CRAZYFLIE_21_CFG, CrazyflieL2FParams
from flight_eval_utils import FlightDataLogger

# Import the environment and agent classes from the training script
from train_pointnav_obs_avoidance import (
    CrazyfliePointNavObsAvoidEnvCfg,
    CrazyfliePointNavObsAvoidEnv,
    L2FPPOAgent,
    L2FActorNetwork,
    L2FCriticNetwork,
    RunningMeanStd,
    ProximitySensorModel,
)

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection


# ==============================================================================
# Per-Episode Data Recorder
# ==============================================================================

class EpisodeRecorder:
    """Records detailed per-step data for a single environment across episodes."""

    def __init__(self, env_idx: int, max_episodes: int = 50):
        self.env_idx = env_idx
        self.max_episodes = max_episodes
        self.episodes: List[Dict] = []
        self._current_episode: Dict = None
        self._reset_current()

    def _reset_current(self):
        """Start a new episode recording buffer."""
        self._current_episode = {
            "positions": [],        # (T, 3) drone XYZ
            "goal_pos": None,       # (3,)
            "obstacle_pos": [],     # (max_obs, 3)
            "obstacle_radii": [],   # (max_obs,)
            "obstacle_active": [],  # (max_obs,) bool
            "proximity_bins": [],   # (T, 8)
            "min_obstacle_dist": [],  # (T,)
            "in_collision": [],     # (T,) bool
            "rewards": [],          # (T,)
            "velocities": [],       # (T, 3)
            "actions": [],          # (T, 4)
            "outcome": None,        # "reach", "collision", "timeout", "crash"
            "steps": 0,
        }

    def log_step(self, env: CrazyfliePointNavObsAvoidEnv, reward: float):
        """Record one step of data."""
        idx = self.env_idx
        ep = self._current_episode

        pos = env._robot.data.root_pos_w[idx].cpu().numpy().copy()
        vel = env._robot.data.root_lin_vel_w[idx].cpu().numpy().copy()
        prox = env._proximity_bins[idx].cpu().numpy().copy()
        act = env._actions[idx].cpu().numpy().copy()

        ep["positions"].append(pos)
        ep["velocities"].append(vel)
        ep["proximity_bins"].append(prox)
        ep["min_obstacle_dist"].append(env._min_obstacle_dist[idx].item())
        ep["in_collision"].append(env._in_collision[idx].item())
        ep["rewards"].append(reward)
        ep["actions"].append(act)
        ep["steps"] += 1

        # Capture goal & obstacle info on first step
        if ep["goal_pos"] is None:
            ep["goal_pos"] = env._goal_pos[idx].cpu().numpy().copy()
            ep["obstacle_pos"] = env._obstacle_pos[idx].cpu().numpy().copy()
            ep["obstacle_radii"] = env._obstacle_radii[idx].cpu().numpy().copy()
            ep["obstacle_active"] = env._obstacle_active[idx].cpu().numpy().copy()

    def finish_episode(self, outcome: str):
        """Finalize current episode and start a new one."""
        if self._current_episode["steps"] == 0:
            self._reset_current()
            return

        self._current_episode["outcome"] = outcome
        # Convert lists to numpy arrays
        for key in ["positions", "velocities", "proximity_bins", "actions",
                     "min_obstacle_dist", "in_collision", "rewards"]:
            self._current_episode[key] = np.array(self._current_episode[key])

        if len(self.episodes) < self.max_episodes:
            self.episodes.append(self._current_episode)

        self._reset_current()


# ==============================================================================
# Plotting Functions
# ==============================================================================

def plot_trajectory_2d(episode: Dict, ax: plt.Axes, episode_idx: int):
    """Plot a single episode's 2D trajectory (XY) with obstacles and goal."""
    positions = episode["positions"]  # (T, 3)
    goal = episode["goal_pos"]       # (3,)
    obs_pos = episode["obstacle_pos"]    # (max_obs, 3)
    obs_radii = episode["obstacle_radii"]  # (max_obs,)
    obs_active = episode["obstacle_active"]  # (max_obs,)

    # Color the trajectory by time
    T = len(positions)
    x, y = positions[:, 0], positions[:, 1]

    # Create colored line segments
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0, T)
    lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=1.5, alpha=0.8)
    lc.set_array(np.arange(T))
    ax.add_collection(lc)

    # Plot obstacles
    for i in range(len(obs_radii)):
        if obs_active[i]:
            circle = Circle((obs_pos[i, 0], obs_pos[i, 1]), obs_radii[i],
                           color='red', alpha=0.5, label='Obstacle' if i == 0 else None)
            ax.add_patch(circle)
            # Collision warning zone
            warn_circle = Circle((obs_pos[i, 0], obs_pos[i, 1]),
                                obs_radii[i] + 0.15,  # warning radius
                                color='red', alpha=0.1, linestyle='--', fill=False)
            ax.add_patch(warn_circle)

    # Start and goal
    ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=10, label='Start', zorder=5)
    ax.plot(goal[0], goal[1], 'r*', markersize=15, label='Goal', zorder=5)

    # Collision points
    collisions = episode["in_collision"]
    if collisions.any():
        col_idx = np.where(collisions)[0]
        ax.plot(positions[col_idx, 0], positions[col_idx, 1], 'rx', markersize=8,
                label='Collision', alpha=0.7, zorder=6)

    outcome_colors = {"reach": "green", "collision": "red", "timeout": "orange", "crash": "red"}
    outcome = episode["outcome"]
    color = outcome_colors.get(outcome, "gray")

    ax.set_title(f"Episode {episode_idx} — {outcome.upper()}", color=color, fontweight='bold')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='upper right')

    # Auto-scale
    all_x = np.concatenate([x, [goal[0]], obs_pos[obs_active, 0]])
    all_y = np.concatenate([y, [goal[1]], obs_pos[obs_active, 1]])
    margin = 0.15
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)


def plot_proximity_heatmap(episode: Dict, ax: plt.Axes, episode_idx: int):
    """Plot proximity sensor readings as a heatmap over time."""
    prox = episode["proximity_bins"]  # (T, 8)
    T = prox.shape[0]

    sector_labels = [f"{i*45}°" for i in range(8)]
    im = ax.imshow(prox.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                   origin='lower', interpolation='nearest')
    ax.set_yticks(range(8))
    ax.set_yticklabels(sector_labels, fontsize=7)
    ax.set_xlabel("Step")
    ax.set_ylabel("Sector")
    ax.set_title(f"Ep {episode_idx} — Proximity (0=touch, 1=clear)")

    return im


def generate_eval_plots(
    episodes: List[Dict],
    stats: Dict,
    output_dir: str,
):
    """Generate all evaluation plots and save them."""
    os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # 1. Trajectory overview: up to 12 episodes in a grid
    # =========================================================================
    n_traj = min(len(episodes), 12)
    if n_traj > 0:
        cols = min(4, n_traj)
        rows = math.ceil(n_traj / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        if n_traj == 1:
            axes = np.array([axes])
        axes = np.array(axes).flatten()
        for i in range(n_traj):
            plot_trajectory_2d(episodes[i], axes[i], i)
        for i in range(n_traj, len(axes)):
            axes[i].set_visible(False)
        fig.suptitle("Flight Trajectories with Obstacles", fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "trajectories_2d.jpg"), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved trajectories_2d.jpg ({n_traj} episodes)")

    # =========================================================================
    # 2. Proximity heatmaps for first few episodes
    # =========================================================================
    n_prox = min(len(episodes), 6)
    if n_prox > 0:
        fig, axes = plt.subplots(n_prox, 1, figsize=(14, 3 * n_prox))
        if n_prox == 1:
            axes = [axes]
        for i in range(n_prox):
            im = plot_proximity_heatmap(episodes[i], axes[i], i)
        fig.colorbar(im, ax=axes, shrink=0.6, label="Normalized distance")
        fig.suptitle("Proximity Sensor Readings Over Time", fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "proximity_heatmaps.jpg"), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved proximity_heatmaps.jpg ({n_prox} episodes)")

    # =========================================================================
    # 3. Aggregate stats bar chart
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Outcome distribution
    outcomes = [ep["outcome"] for ep in episodes]
    outcome_counts = {}
    for o in ["reach", "collision", "timeout", "crash"]:
        outcome_counts[o] = outcomes.count(o)
    colors = {"reach": "green", "collision": "red", "timeout": "orange", "crash": "darkred"}
    bars = axes[0].bar(outcome_counts.keys(),
                       outcome_counts.values(),
                       color=[colors[k] for k in outcome_counts.keys()])
    axes[0].set_title("Episode Outcomes")
    axes[0].set_ylabel("Count")
    for bar, count in zip(bars, outcome_counts.values()):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     str(count), ha='center', fontweight='bold')

    # Episode length distribution
    lengths = [ep["steps"] for ep in episodes]
    axes[1].hist(lengths, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1].axvline(np.mean(lengths), color='red', linestyle='--', label=f'Mean={np.mean(lengths):.0f}')
    axes[1].set_title("Episode Length Distribution")
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Count")
    axes[1].legend()

    # Collision count per episode
    col_counts = [ep["in_collision"].sum() for ep in episodes]
    axes[2].hist(col_counts, bins=20, color='salmon', edgecolor='black', alpha=0.7)
    axes[2].set_title("Collision Steps per Episode")
    axes[2].set_xlabel("Collision steps")
    axes[2].set_ylabel("Count")

    fig.suptitle("Evaluation Summary", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "eval_summary.jpg"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved eval_summary.jpg")

    # =========================================================================
    # 4. Min obstacle distance over time (overlaid episodes)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, ep in enumerate(episodes[:20]):
        dist = ep["min_obstacle_dist"]
        t = np.arange(len(dist))
        color = 'green' if ep["outcome"] == "reach" else ('red' if ep["outcome"] == "collision" else 'orange')
        ax.plot(t, dist, alpha=0.4, color=color, linewidth=0.8)

    ax.axhline(0.15, color='red', linestyle='--', alpha=0.5, label='Warning zone (15cm)')
    ax.axhline(0.05, color='darkred', linestyle='--', alpha=0.5, label='Collision zone (~5cm)')
    ax.set_xlabel("Step")
    ax.set_ylabel("Min distance to obstacle surface (m)")
    ax.set_title("Minimum Obstacle Distance Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "obstacle_distance_timeseries.jpg"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved obstacle_distance_timeseries.jpg")

    # =========================================================================
    # 5. Reward decomposition for recorded episodes
    # =========================================================================
    if len(episodes) > 0:
        fig, ax = plt.subplots(figsize=(12, 5))
        for i, ep in enumerate(episodes[:10]):
            rewards = ep["rewards"]
            t = np.arange(len(rewards))
            cum_reward = np.cumsum(rewards)
            color = 'green' if ep["outcome"] == "reach" else ('red' if ep["outcome"] == "collision" else 'orange')
            ax.plot(t, cum_reward, alpha=0.6, color=color, linewidth=1.0,
                    label=f"Ep{i} ({ep['outcome']})" if i < 5 else None)

        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative Reward")
        ax.set_title("Cumulative Reward Over Time")
        if len(episodes) <= 5:
            ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(output_dir, "reward_curves.jpg"), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved reward_curves.jpg")

    # =========================================================================
    # 6. Speed profile with obstacle proximity
    # =========================================================================
    if len(episodes) > 0:
        fig, axes = plt.subplots(min(4, len(episodes)), 1,
                                 figsize=(14, 3 * min(4, len(episodes))))
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        for i, ax in enumerate(axes):
            ep = episodes[i]
            vels = ep["velocities"]
            speed = np.linalg.norm(vels[:, :2], axis=1)
            dist = ep["min_obstacle_dist"]
            t = np.arange(len(speed))

            ax_twin = ax.twinx()
            ax.plot(t, speed, 'b-', alpha=0.7, label='Speed (m/s)')
            ax_twin.plot(t, dist, 'r-', alpha=0.5, label='Obs dist (m)')
            ax_twin.axhline(0.15, color='red', linestyle=':', alpha=0.3)

            ax.set_xlabel("Step")
            ax.set_ylabel("Speed (m/s)", color='blue')
            ax_twin.set_ylabel("Min obs dist (m)", color='red')
            ax.set_title(f"Episode {i} — {ep['outcome'].upper()}")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "speed_vs_proximity.jpg"), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved speed_vs_proximity.jpg")


# ==============================================================================
# Evaluation Loop
# ==============================================================================

def evaluate(
    env: CrazyfliePointNavObsAvoidEnv,
    agent: L2FPPOAgent,
    num_episodes: int,
    record_envs: int,
    deterministic: bool,
    output_dir: str,
):
    """Run evaluation episodes and generate reports."""
    print(f"\n{'='*60}")
    print("EVALUATION — Point Nav + Obstacle Avoidance")
    print(f"{'='*60}")
    print(f"  Environments:   {env.num_envs}")
    print(f"  Target episodes: {num_episodes}")
    print(f"  Recording envs:  {record_envs}")
    print(f"  Deterministic:   {deterministic}")
    print(f"  Output dir:      {output_dir}")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)

    # Episode recorders for detailed trajectories
    recorders = [EpisodeRecorder(i, max_episodes=50) for i in range(min(record_envs, env.num_envs))]

    # Flight data logger (for traditional sim-to-real comparison plots)
    flight_logger = FlightDataLogger()

    # Aggregate statistics
    total_episodes = 0
    total_steps = 0
    outcome_counts = defaultdict(int)
    episode_lengths = []
    episode_rewards = []
    episode_collision_steps = []
    episode_min_dists = []
    per_env_rewards = torch.zeros(env.num_envs, device=env.device)
    per_env_collision_steps = torch.zeros(env.num_envs, device=env.device)
    per_env_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    max_steps = num_episodes * 1000 * 2  # Safety limit
    step = 0

    print("Running evaluation...")
    eval_start_time = time.time()

    while total_episodes < num_episodes and step < max_steps and simulation_app.is_running():
        # Get action
        with torch.no_grad():
            action = agent.get_action(obs, deterministic=deterministic)

        # Step environment
        obs_dict, reward, terminated, truncated, info = env.step(action)
        obs = obs_dict["policy"]
        done = terminated | truncated
        step += 1
        total_steps += 1

        # Per-env tracking
        per_env_rewards += reward
        per_env_collision_steps += env._in_collision.float()
        per_env_steps += 1

        # Record detailed data for tracked envs
        for rec in recorders:
            rec.log_step(env, reward[rec.env_idx].item())

        # Log flight data for env 0
        flight_logger.log_step(env, env_idx=0)

        # Handle episode completions
        if done.any():
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)

            for idx_tensor in done_ids:
                idx = idx_tensor.item()

                # Determine outcome (use persistent flags that survive auto-reset)
                if env._last_done_goal_reached[idx]:
                    outcome = "reach"
                elif env._last_done_collision[idx]:
                    outcome = "collision"
                elif truncated[idx]:
                    outcome = "timeout"
                else:
                    outcome = "crash"

                # Record stats
                ep_len = per_env_steps[idx].item()
                ep_reward = per_env_rewards[idx].item()
                ep_col_steps = per_env_collision_steps[idx].item()

                outcome_counts[outcome] += 1
                episode_lengths.append(ep_len)
                episode_rewards.append(ep_reward)
                episode_collision_steps.append(ep_col_steps)
                episode_min_dists.append(env._min_obstacle_dist[idx].item())

                total_episodes += 1

                # Finish recording for tracked envs
                for rec in recorders:
                    if rec.env_idx == idx:
                        rec.finish_episode(outcome)

                # Reset per-env counters
                per_env_rewards[idx] = 0.0
                per_env_collision_steps[idx] = 0.0
                per_env_steps[idx] = 0

            # Progress logging
            if total_episodes % 20 == 0 or total_episodes >= num_episodes:
                elapsed = time.time() - eval_start_time
                reach_rate = outcome_counts["reach"] / max(total_episodes, 1) * 100
                col_rate = outcome_counts.get("collision", 0) / max(total_episodes, 1) * 100
                crash_rate = outcome_counts.get("crash", 0) / max(total_episodes, 1) * 100
                avg_reward = np.mean(episode_rewards[-20:]) if episode_rewards else 0
                eps_per_sec = total_episodes / max(elapsed, 0.01)

                print(f"  [{total_episodes:4d}/{num_episodes}] "
                      f"Reach: {reach_rate:5.1f}% | Col: {col_rate:4.1f}% | Crash: {crash_rate:4.1f}% | "
                      f"Avg Reward: {avg_reward:8.2f} | "
                      f"Speed: {eps_per_sec:.1f} ep/s")

            if total_episodes >= num_episodes:
                break

    eval_elapsed = time.time() - eval_start_time

    # =========================================================================
    # Compute Final Statistics
    # =========================================================================
    n_ep = max(total_episodes, 1)

    stats = {
        "total_episodes": total_episodes,
        "total_steps": total_steps,
        "eval_time_s": eval_elapsed,
        "episodes_per_second": total_episodes / max(eval_elapsed, 0.01),
        "outcomes": {
            "reach": outcome_counts["reach"],
            "collision": outcome_counts.get("collision", 0),
            "timeout": outcome_counts.get("timeout", 0),
            "crash": outcome_counts.get("crash", 0),
        },
        "rates": {
            "reach_rate": outcome_counts["reach"] / n_ep * 100,
            "collision_rate": outcome_counts.get("collision", 0) / n_ep * 100,
            "timeout_rate": outcome_counts.get("timeout", 0) / n_ep * 100,
            "crash_rate": outcome_counts.get("crash", 0) / n_ep * 100,
        },
        "episode_length": {
            "mean": float(np.mean(episode_lengths)) if episode_lengths else 0,
            "std": float(np.std(episode_lengths)) if episode_lengths else 0,
            "min": int(np.min(episode_lengths)) if episode_lengths else 0,
            "max": int(np.max(episode_lengths)) if episode_lengths else 0,
            "median": float(np.median(episode_lengths)) if episode_lengths else 0,
        },
        "reward": {
            "mean": float(np.mean(episode_rewards)) if episode_rewards else 0,
            "std": float(np.std(episode_rewards)) if episode_rewards else 0,
            "min": float(np.min(episode_rewards)) if episode_rewards else 0,
            "max": float(np.max(episode_rewards)) if episode_rewards else 0,
        },
        "collision_steps_per_episode": {
            "mean": float(np.mean(episode_collision_steps)) if episode_collision_steps else 0,
            "std": float(np.std(episode_collision_steps)) if episode_collision_steps else 0,
            "max": float(np.max(episode_collision_steps)) if episode_collision_steps else 0,
        },
        "environment": {
            "num_envs": env.num_envs,
            "max_obstacles": env.cfg.max_obstacles,
            "min_obstacles": env.cfg.min_obstacles,
            "goal_distance_range": [env.cfg.goal_min_distance, env.cfg.goal_max_distance],
            "proximity_sectors": env.cfg.num_proximity_sectors,
            "term_on_collision": env.cfg.term_on_collision,
            "obs_dim": env.cfg.observation_space,
        },
    }

    # =========================================================================
    # Print Report
    # =========================================================================
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Episodes:          {total_episodes}")
    print(f"  Total steps:       {total_steps}")
    print(f"  Eval time:         {eval_elapsed:.1f} s ({stats['episodes_per_second']:.1f} ep/s)")
    print(f"\n  --- Outcomes ---")
    print(f"  Goal Reached:      {outcome_counts['reach']:4d} ({stats['rates']['reach_rate']:5.1f}%)")
    print(f"  Collision Term:    {outcome_counts.get('collision', 0):4d} ({stats['rates']['collision_rate']:5.1f}%)")
    print(f"  Timeout:           {outcome_counts.get('timeout', 0):4d} ({stats['rates']['timeout_rate']:5.1f}%)")
    print(f"  Crash (safety):    {outcome_counts.get('crash', 0):4d} ({stats['rates']['crash_rate']:5.1f}%)")
    print(f"\n  --- Episode Length ---")
    print(f"  Mean:  {stats['episode_length']['mean']:7.1f} ± {stats['episode_length']['std']:.1f}")
    print(f"  Range: [{stats['episode_length']['min']}, {stats['episode_length']['max']}]")
    print(f"\n  --- Reward ---")
    print(f"  Mean:  {stats['reward']['mean']:8.2f} ± {stats['reward']['std']:.2f}")
    print(f"  Range: [{stats['reward']['min']:.2f}, {stats['reward']['max']:.2f}]")
    print(f"\n  --- Collision Steps/Episode ---")
    print(f"  Mean:  {stats['collision_steps_per_episode']['mean']:7.1f} ± "
          f"{stats['collision_steps_per_episode']['std']:.1f}")
    print(f"  Max:   {stats['collision_steps_per_episode']['max']:.0f}")
    print(f"{'='*60}\n")

    # =========================================================================
    # Save Results
    # =========================================================================

    # Save stats JSON
    json_path = os.path.join(output_dir, "eval_stats.json")
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {json_path}")

    # Save per-episode CSV
    csv_path = os.path.join(output_dir, "eval_episodes.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "length", "reward", "collision_steps", "outcome"])
        for i in range(len(episode_lengths)):
            outcome = "unknown"
            # Reconstruct outcome from counts
            if i < outcome_counts["reach"]:
                outcome = "reach"
            writer.writerow([i, episode_lengths[i], f"{episode_rewards[i]:.4f}",
                            episode_collision_steps[i], ""])
    print(f"Saved per-episode CSV to {csv_path}")

    # Save flight data CSV (env 0)
    flight_csv = os.path.join(output_dir, "flight_data_env0.csv")
    flight_logger.save_and_plot(flight_csv, title_prefix="PointNav_ObsAvoid", output_dir=output_dir)
    print(f"Saved flight data to {flight_csv}")

    # =========================================================================
    # Generate Plots
    # =========================================================================
    print("\nGenerating evaluation plots...")
    all_recorded_episodes = []
    for rec in recorders:
        all_recorded_episodes.extend(rec.episodes)

    if all_recorded_episodes:
        generate_eval_plots(all_recorded_episodes, stats, output_dir)
    else:
        print("  Warning: No episodes were recorded for detailed plots.")

    print(f"\nAll evaluation outputs saved to: {output_dir}")
    return stats


# ==============================================================================
# Main
# ==============================================================================

def main():
    # Setup environment
    cfg = CrazyfliePointNavObsAvoidEnvCfg()
    cfg.scene.num_envs = args.num_envs

    # Match Phase 4 training conditions
    cfg.term_on_collision = args.term_on_collision
    cfg.obs_collision_penalty = args.collision_penalty
    if args.max_obstacles is not None:
        cfg.max_obstacles = args.max_obstacles
    if args.min_obstacles is not None:
        cfg.min_obstacles = args.min_obstacles

    env = CrazyfliePointNavObsAvoidEnv(cfg)

    # Setup agent
    agent = L2FPPOAgent(
        obs_dim=cfg.observation_space,
        action_dim=cfg.action_space,
        device=env.device,
    )

    # Load checkpoint
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.checkpoint is None:
        checkpoint_dir = os.path.join(script_dir, "checkpoints_pointnav_obs")
        args.checkpoint = os.path.join(checkpoint_dir, "best_model.pt")

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print(f"Available checkpoints:")
        ckpt_dir = os.path.join(script_dir, "checkpoints_pointnav_obs")
        if os.path.exists(ckpt_dir):
            for f in os.listdir(ckpt_dir):
                print(f"  {os.path.join(ckpt_dir, f)}")
        env.close()
        simulation_app.close()
        sys.exit(1)

    iteration, best_reward = agent.load(args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  Trained iteration: {iteration}")
    print(f"  Best reward:       {best_reward:.3f}")

    # Output directory
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(script_dir, "eval", "pointnav_obs", f"eval_{timestamp}")

    # Run evaluation
    stats = evaluate(
        env=env,
        agent=agent,
        num_episodes=args.num_episodes,
        record_envs=args.record_envs,
        deterministic=args.deterministic,
        output_dir=args.output_dir,
    )

    env.close()
    simulation_app.close()

    return stats


if __name__ == "__main__":
    main()
