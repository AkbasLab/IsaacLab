#!/usr/bin/env python3
"""
Training-style point navigation evaluation for Crazyflie 2.1.

This script intentionally avoids the legacy play_eval flow and instead reuses
the same environment and actor classes as train_pointnav.py. It loads a .pt
checkpoint, injects a fixed local-frame goal, and runs a rollout with the same
Crazyflie 2.1 model/configuration used during training.

Key properties:
- Uses CrazyfliePointNavEnvCfg and CrazyfliePointNavEnv from train_pointnav.py
- Uses the exact L2FActorNetwork class from train_pointnav.py
- Starts at the training init height for the provided goal_z
- Keeps training-style reset/randomization unless explicitly overridden
- Defaults to stochastic actions so the rollout behaves like training

Examples:
    .\\isaaclab.bat -p source\\isaaclab_tasks\\isaaclab_tasks\\direct\\crazyflie_l2f\\eval_pointnav_goal.py --target_x 0.0 --target_y 0.0 --target_z 0.3 --headless

    .\\isaaclab.bat -p source\\isaaclab_tasks\\isaaclab_tasks\\direct\\crazyflie_l2f\\eval_pointnav_goal.py --target_x 0.3 --target_y 0.2 --target_z 1.0 --checkpoint source\\isaaclab_tasks\\isaaclab_tasks\\direct\\crazyflie_l2f\\checkpoints_pointnav\\best_model.pt --num_envs 64 --duration 10 --headless
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
import torch

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description="Training-style fixed-goal Crazyflie evaluation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint")
    parser.add_argument("--target_x", type=float, required=True, help="Local-frame goal x in meters")
    parser.add_argument("--target_y", type=float, required=True, help="Local-frame goal y in meters")
    parser.add_argument("--target_z", type=float, required=True, help="Local-frame goal z in meters")
    parser.add_argument("--z_reference_offset", type=float, default=0.0,
                        help="Virtual height offset in meters. Example: 1.0 means user z=0.3 is evaluated at world z=1.3")
    parser.add_argument("--num_envs", type=int, default=32, help="Number of parallel environments")
    parser.add_argument("--duration", type=float, default=10.0, help="Rollout duration in seconds")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save CSV/summary")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actor output instead of stochastic sampling")
    parser.add_argument("--ground_start", action="store_true", help="Start from the ground instead of the training-height reset")
    parser.add_argument("--show_goal_marker", action="store_true",
                        help="Render a visible goal marker for env 0 in non-headless mode")
    parser.add_argument("--seed", type=int, default=None, help="Optional torch seed")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args = parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from train_pointnav import CrazyfliePointNavEnvCfg, CrazyfliePointNavEnv, L2FActorNetwork  # noqa: E402
from isaaclab.markers import VisualizationMarkers  # noqa: E402
from isaaclab.markers.config import POSITION_GOAL_MARKER_CFG  # noqa: E402


def quat_to_euler_wxyz(quat: torch.Tensor) -> tuple[float, float, float]:
    """Convert quaternion [w, x, y, z] to Euler angles in degrees."""
    w, x, y, z = [float(v) for v in quat]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return (
        math.degrees(roll),
        math.degrees(pitch),
        math.degrees(yaw),
    )


def default_checkpoint_path() -> str:
    script_dir = Path(__file__).resolve().parent
    return str(script_dir / "checkpoints_pointnav" / "best_model.pt")


def output_directory() -> Path:
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_dir = Path(__file__).resolve().parent
        out_dir = script_dir / "play_eval_results" / f"goal_eval_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def build_env() -> CrazyfliePointNavEnv:
    world_target_z = args.target_z + args.z_reference_offset
    cfg = CrazyfliePointNavEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.debug_vis = False
    cfg.episode_length_s = max(cfg.episode_length_s, args.duration)

    # Make the provided fixed goal look like a training-time task specification.
    cfg.goal_height = world_target_z
    cfg.init_target_height = world_target_z
    # Evaluation should record one continuous benchmark rollout, not a training
    # episode with resets from success or aggressive safety terminations.
    cfg.goal_hold_steps = int(math.ceil(args.duration / cfg.sim.dt)) + 100
    cfg.enable_disturbance = False
    cfg.init_guidance_probability = 1.0
    cfg.init_max_xy_offset = 0.0
    cfg.init_max_angle = 0.0
    cfg.init_max_linear_velocity = 0.0
    cfg.init_max_angular_velocity = 0.0
    cfg.init_height_offset_min = 0.0
    cfg.init_height_offset_max = 0.0

    # Relax termination bounds so the drone is not reset mid-log. We still
    # preserve the physics and controller; this only removes training-style
    # episode endings from the evaluation harness.
    cfg.term_xy_threshold = 10.0
    cfg.term_z_soft_min = -1.0
    cfg.term_z_hard_min = -2.0
    cfg.term_z_soft_max = max(args.target_z + 3.0, 4.0)
    cfg.term_z_hard_max = max(args.target_z + 4.0, 5.0)
    cfg.term_tilt_soft_threshold = math.pi
    cfg.term_tilt_hard_threshold = math.pi + 0.1
    cfg.term_tilt_persistence_steps = int(math.ceil(args.duration / cfg.sim.dt)) + 100
    cfg.term_linear_velocity_soft_threshold = 100.0
    cfg.term_linear_velocity_hard_threshold = 200.0
    cfg.term_linear_velocity_persistence_steps = int(math.ceil(args.duration / cfg.sim.dt)) + 100
    cfg.term_angular_velocity_soft_threshold = 1000.0
    cfg.term_angular_velocity_hard_threshold = 2000.0
    cfg.term_angular_velocity_persistence_steps = int(math.ceil(args.duration / cfg.sim.dt)) + 100

    env = CrazyfliePointNavEnv(cfg)
    env._eval_fixed_goal = (args.target_x, args.target_y, world_target_z)
    return env


def apply_ground_start(env: CrazyfliePointNavEnv):
    """Place all drones on the ground with motors idle and level attitude."""
    num_envs = env.num_envs
    device = env.device

    ground_pos = env._terrain.env_origins.clone()
    # The evaluation convention is that "ground level" lives at the reference
    # z offset, not world z=0. Keep the body center slightly above that plane.
    ground_pos[:, 2] += 0.03

    quat = torch.zeros(num_envs, 4, device=device)
    quat[:, 0] = 1.0
    vel = torch.zeros(num_envs, 6, device=device)

    env_ids = torch.arange(num_envs, device=device)
    env._robot.write_root_pose_to_sim(torch.cat([ground_pos, quat], dim=-1), env_ids)
    env._robot.write_root_velocity_to_sim(vel, env_ids)

    if hasattr(env, "_rpm_state"):
        env._rpm_state[:] = 0.0
    if hasattr(env, "_actions"):
        env._actions[:] = -1.0
    if hasattr(env, "_action_history"):
        env._action_history[:] = -1.0
    if hasattr(env, "_prev_speed"):
        env._prev_speed[:] = 0.0
    if hasattr(env, "_prev_height_below_target"):
        env._prev_height_below_target[:] = 0.0
    if hasattr(env, "_prev_dist_xy"):
        dist_xy = torch.norm(env._goal_pos[:, :2] - ground_pos[:, :2], dim=-1)
        env._prev_dist_xy[:] = dist_xy

    env.scene.write_data_to_sim()
    env.sim.step()
    env.scene.update(env.cfg.sim.dt)


def apply_goal_height_start(env: CrazyfliePointNavEnv):
    """Place all drones at the commanded goal height with hover-ready state."""
    num_envs = env.num_envs
    device = env.device

    start_pos = env._terrain.env_origins.clone()
    start_pos[:, 2] += args.target_z + args.z_reference_offset

    quat = torch.zeros(num_envs, 4, device=device)
    quat[:, 0] = 1.0
    vel = torch.zeros(num_envs, 6, device=device)

    env_ids = torch.arange(num_envs, device=device)
    env._robot.write_root_pose_to_sim(torch.cat([start_pos, quat], dim=-1), env_ids)
    env._robot.write_root_velocity_to_sim(vel, env_ids)

    if hasattr(env, "_rpm_state"):
        env._rpm_state[:] = env._hover_rpm
    if hasattr(env, "_actions"):
        env._actions[:] = env._hover_action
    if hasattr(env, "_action_history"):
        env._action_history[:] = env._hover_action
    if hasattr(env, "_prev_speed"):
        env._prev_speed[:] = 0.0
    if hasattr(env, "_prev_height_below_target"):
        env._prev_height_below_target[:] = 0.0
    if hasattr(env, "_prev_dist_xy"):
        env._prev_dist_xy[:] = 0.0

    env.scene.write_data_to_sim()
    env.sim.step()
    env.scene.update(env.cfg.sim.dt)


def clear_eval_runtime_state(env: CrazyfliePointNavEnv):
    """Clear stale done/reset bookkeeping before starting the recorded rollout."""
    for attr in ("reset_terminated", "reset_time_out", "terminated", "truncated"):
        if hasattr(env, attr):
            value = getattr(env, attr)
            if torch.is_tensor(value):
                value.zero_()
    if hasattr(env, "episode_length_buf") and torch.is_tensor(env.episode_length_buf):
        env.episode_length_buf.zero_()


def load_actor(device: torch.device, obs_dim: int, action_dim: int):
    checkpoint_path = args.checkpoint or default_checkpoint_path()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    actor = L2FActorNetwork(obs_dim=obs_dim, hidden_dim=64, action_dim=action_dim).to(device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    if "obs_mean" in checkpoint and "obs_std" in checkpoint:
        obs_mean = checkpoint["obs_mean"].to(device)
        obs_std = checkpoint["obs_std"].to(device)
    elif "obs_mean" in checkpoint and "obs_var" in checkpoint:
        obs_mean = checkpoint["obs_mean"].to(device)
        obs_std = torch.sqrt(checkpoint["obs_var"].to(device) + 1e-8)
    else:
        obs_mean = torch.zeros(obs_dim, device=device)
        obs_std = torch.ones(obs_dim, device=device)

    return checkpoint_path, actor, obs_mean, obs_std


def create_goal_marker() -> VisualizationMarkers | None:
    """Create a visible goal marker for env 0 when running with a GUI."""
    if args.headless or not args.show_goal_marker:
        return None

    marker_cfg = POSITION_GOAL_MARKER_CFG.copy()
    marker_cfg.prim_path = "/Visuals/GoalMarker"
    marker_cfg.markers["target_far"].radius = 0.04
    marker_cfg.markers["target_near"].radius = 0.04
    marker_cfg.markers["target_invisible"].radius = 0.04
    return VisualizationMarkers(marker_cfg)


def update_goal_marker(goal_marker: VisualizationMarkers | None, env: CrazyfliePointNavEnv, goal_threshold: float):
    """Update the goal marker position and color for env 0."""
    if goal_marker is None:
        return

    goal_pos = env._goal_pos[0:1].detach()
    goal_dist = torch.norm(env._goal_pos[0] - env._robot.data.root_pos_w[0]).item()
    marker_index = 1 if goal_dist < goal_threshold else 0
    marker_indices = torch.tensor([marker_index], dtype=torch.int64, device=env.device)
    goal_marker.visualize(translations=goal_pos, marker_indices=marker_indices)


def write_csv_header(writer: csv.writer):
    writer.writerow([
        "time_s",
        "env_id",
        "episode_step",
        "goal_x",
        "goal_y",
        "goal_z",
        "goal_z_virtual",
        "goal_local_x",
        "goal_local_y",
        "goal_local_z",
        "goal_local_z_virtual",
        "pos_x",
        "pos_y",
        "pos_z",
        "pos_z_virtual",
        "pos_local_x",
        "pos_local_y",
        "pos_local_z",
        "pos_local_z_virtual",
        "vel_x",
        "vel_y",
        "vel_z",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "ang_vel_x",
        "ang_vel_y",
        "ang_vel_z",
        "action_m1",
        "action_m2",
        "action_m3",
        "action_m4",
        "reward",
        "goal_distance",
        "terminated",
        "truncated",
    ])


def append_env0_row(
    writer: csv.writer,
    env: CrazyfliePointNavEnv,
    elapsed_s: float,
    episode_step: int,
    action: torch.Tensor,
    reward: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
):
    env_id = 0
    pos = env._robot.data.root_pos_w[env_id].detach().cpu()
    vel = env._robot.data.root_lin_vel_w[env_id].detach().cpu()
    quat = env._robot.data.root_quat_w[env_id].detach().cpu()
    ang_vel = env._robot.data.root_ang_vel_b[env_id].detach().cpu()
    goal = env._goal_pos[env_id].detach().cpu()
    env_origin = env._terrain.env_origins[env_id].detach().cpu()
    roll, pitch, yaw = quat_to_euler_wxyz(quat)
    goal_dist = torch.norm(env._goal_pos[env_id] - env._robot.data.root_pos_w[env_id]).item()
    goal_z_virtual = goal[2].item() - args.z_reference_offset
    pos_z_virtual = pos[2].item() - args.z_reference_offset
    goal_local = goal - env_origin
    pos_local = pos - env_origin
    goal_local_z_virtual = goal_local[2].item() - args.z_reference_offset
    pos_local_z_virtual = pos_local[2].item() - args.z_reference_offset

    writer.writerow([
        f"{elapsed_s:.4f}",
        env_id,
        episode_step,
        f"{goal[0].item():.6f}",
        f"{goal[1].item():.6f}",
        f"{goal[2].item():.6f}",
        f"{goal_z_virtual:.6f}",
        f"{goal_local[0].item():.6f}",
        f"{goal_local[1].item():.6f}",
        f"{goal_local[2].item():.6f}",
        f"{goal_local_z_virtual:.6f}",
        f"{pos[0].item():.6f}",
        f"{pos[1].item():.6f}",
        f"{pos[2].item():.6f}",
        f"{pos_z_virtual:.6f}",
        f"{pos_local[0].item():.6f}",
        f"{pos_local[1].item():.6f}",
        f"{pos_local[2].item():.6f}",
        f"{pos_local_z_virtual:.6f}",
        f"{vel[0].item():.6f}",
        f"{vel[1].item():.6f}",
        f"{vel[2].item():.6f}",
        f"{roll:.6f}",
        f"{pitch:.6f}",
        f"{yaw:.6f}",
        f"{ang_vel[0].item():.6f}",
        f"{ang_vel[1].item():.6f}",
        f"{ang_vel[2].item():.6f}",
        f"{action[env_id, 0].item():.6f}",
        f"{action[env_id, 1].item():.6f}",
        f"{action[env_id, 2].item():.6f}",
        f"{action[env_id, 3].item():.6f}",
        f"{reward[env_id].item():.6f}",
        f"{goal_dist:.6f}",
        int(terminated[env_id].item()),
        int(truncated[env_id].item()),
    ])


def main():
    if args.seed is not None:
        torch.manual_seed(args.seed)

    env = build_env()
    checkpoint_path, actor, obs_mean, obs_std = load_actor(
        env.device, env.cfg.observation_space, env.cfg.action_space
    )

    out_dir = output_directory()
    csv_path = out_dir / "goal_eval_data.csv"
    summary_path = out_dir / "summary.json"

    print("\n" + "=" * 60)
    print("TRAINING-STYLE FIXED-GOAL EVALUATION")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Goal (virtual frame): ({args.target_x:.2f}, {args.target_y:.2f}, {args.target_z:.2f}) m")
    print(f"Z reference offset: {args.z_reference_offset:.2f} m")
    print(f"Goal (world frame): ({args.target_x:.2f}, {args.target_y:.2f}, {args.target_z + args.z_reference_offset:.2f}) m")
    print(f"Environments: {args.num_envs}")
    print(f"Duration: {args.duration:.2f} s")
    print(f"Policy mode: {'deterministic' if args.deterministic else 'stochastic'}")
    print(f"Start mode: {'ground' if args.ground_start else 'training-height'}")
    print(f"Goal marker: {'shown' if args.show_goal_marker and not args.headless else 'hidden'}")
    print(f"Output: {out_dir}")
    print("=" * 60 + "\n")

    obs_dict, _ = env.reset()
    if abs(args.target_z) > 1e-6:
        apply_goal_height_start(env)
        obs_dict = env._get_observations()
    elif args.ground_start:
        apply_ground_start(env)
        obs_dict = env._get_observations()
    clear_eval_runtime_state(env)
    obs = obs_dict["policy"]

    dt = env.cfg.sim.dt
    num_steps = int(args.duration / dt)
    goal_threshold = env.cfg.goal_reach_threshold
    goal_marker = create_goal_marker()
    update_goal_marker(goal_marker, env, goal_threshold)
    reached_once = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)
    first_reach_time = torch.full((args.num_envs,), float("nan"), device=env.device)
    episode_step = 0

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        write_csv_header(writer)

        for step in range(num_steps):
            if not simulation_app.is_running():
                break

            obs_norm = (obs - obs_mean) / (obs_std + 1e-8)
            with torch.no_grad():
                action = actor.get_action(obs_norm, deterministic=args.deterministic)

            obs_dict, reward, terminated, truncated, _ = env.step(action)
            terminated = terminated.clone()
            truncated = truncated.clone()
            obs = obs_dict["policy"]
            episode_step += 1

            elapsed_s = step * dt
            goal_dist = torch.norm(env._goal_pos - env._robot.data.root_pos_w, dim=-1)
            newly_reached = (goal_dist < goal_threshold) & (~reached_once)
            if newly_reached.any():
                first_reach_time[newly_reached] = elapsed_s
                reached_once |= newly_reached
            update_goal_marker(goal_marker, env, goal_threshold)

            # Ignore one-step startup done artifacts immediately after reset/setup.
            if step == 0 and bool((terminated[0] | truncated[0]).item()):
                terminated[0] = False
                truncated[0] = False

            append_env0_row(writer, env, elapsed_s, episode_step, action, reward, terminated, truncated)

            done = terminated | truncated
            if bool(done[0].item()):
                print(f"Stopping rollout after env0 termination at t={elapsed_s:.2f}s")
                break
            if done.any():
                episode_step = 0

            if (step + 1) % 100 == 0:
                reach_rate = reached_once.float().mean().item() * 100.0
                env0_dist = goal_dist[0].item()
                print(
                    f"Step {step + 1:5d}/{num_steps} | "
                    f"time={elapsed_s:6.2f}s | "
                    f"env0_dist={env0_dist:6.3f}m | "
                    f"reached_once={reach_rate:5.1f}%"
                )

    summary = {
        "checkpoint": checkpoint_path,
        "goal_local": {
            "x": args.target_x,
            "y": args.target_y,
            "z": args.target_z,
        },
        "z_reference_offset_m": args.z_reference_offset,
        "goal_world": {
            "x": args.target_x,
            "y": args.target_y,
            "z": args.target_z + args.z_reference_offset,
        },
        "num_envs": args.num_envs,
        "duration_s": args.duration,
        "deterministic": args.deterministic,
        "goal_threshold_m": goal_threshold,
        "reach_rate_percent": reached_once.float().mean().item() * 100.0,
        "mean_first_reach_time_s": (
            torch.nanmean(first_reach_time).item()
            if torch.isfinite(first_reach_time).any()
            else None
        ),
        "csv_path": str(csv_path),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved:")
    print(f"  CSV: {csv_path}")
    print(f"  Summary: {summary_path}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
