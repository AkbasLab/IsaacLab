#!/usr/bin/env python3
"""
Fine-tune a pointnav checkpoint for sustained fixed-position hold.

This script warm-starts from an existing 149-dim pointnav checkpoint and
continues PPO training on a single fixed local-frame goal so the policy learns
to hold that position for 10 seconds.

It reuses the same Crazyflie 2.1 environment, actor/critic, and observation
normalization pipeline from train_pointnav.py.

Example:
    .\\isaaclab.bat -p source\\isaaclab_tasks\\isaaclab_tasks\\direct\\crazyflie_l2f\\train_hold_finetune.py --target_x 0.0 --target_y 0.0 --target_z 0.3 --z_reference_offset 1.0 --headless
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tempfile
from collections import deque
from datetime import datetime

import torch

from isaaclab.app import AppLauncher


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_TMP_DIR = os.path.join(SCRIPT_DIR, ".isaac_tmp")
os.makedirs(LOCAL_TMP_DIR, exist_ok=True)
os.environ["TMP"] = LOCAL_TMP_DIR
os.environ["TEMP"] = LOCAL_TMP_DIR
os.environ.setdefault("TMPDIR", LOCAL_TMP_DIR)
tempfile.tempdir = LOCAL_TMP_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune pointnav checkpoint for fixed-position hold")
    parser.add_argument("--checkpoint", type=str, default=None, help="Warm-start checkpoint path")
    parser.add_argument("--task_mode", type=str, default="mixed_nav_hold", choices=("precision_hold", "mixed_nav_hold", "fast_nav_hold", "trajectory_track", "trajectory_generalized"),
                        help="Training task mode. 'precision_hold' preserves tight hold specialization. 'mixed_nav_hold' broadens into full 3D navigation plus hold. 'fast_nav_hold' pushes harder on travel speed. 'trajectory_track' trains directly on a timed waypoint CSV. 'trajectory_generalized' mixes timed trajectory envs with random 3D navigation envs.")
    parser.add_argument("--resume_best", action="store_true",
                        help="Warm-start from checkpoints_hold_finetune/best_model.pt by default")
    parser.add_argument("--checkpoint_dir_name", type=str, default="checkpoints_hold_finetune_unified",
                        help="Output checkpoint directory name under the script folder")
    parser.add_argument("--timestamp_checkpoint_dir", action="store_true",
                        help="Append a YYYYMMDD_HHMMSS suffix to the checkpoint directory name")
    parser.add_argument("--target_x", type=float, default=None, help="Optional fixed local-frame goal x in meters")
    parser.add_argument("--target_y", type=float, default=None, help="Optional fixed local-frame goal y in meters")
    parser.add_argument("--target_z", type=float, default=None, help="Optional fixed goal z in meters")
    parser.add_argument("--goal_z_min", type=float, default=0.3, help="Minimum random goal z in meters")
    parser.add_argument("--goal_z_max", type=float, default=0.3, help="Maximum random goal z in meters")
    parser.add_argument("--trajectory_csv", type=str, default=None,
                        help="CSV with t,target.x,target.y,target.z columns for trajectory_track mode.")
    parser.add_argument("--trajectory_skip_initial", type=int, default=0,
                        help="Skip this many initial finite target rows from the trajectory CSV.")
    parser.add_argument("--trajectory_min_spacing", type=float, default=0.0,
                        help="Minimum 3D spacing in meters between kept trajectory targets.")
    parser.add_argument("--trajectory_stride", type=int, default=1,
                        help="Keep every Nth trajectory target after spacing.")
    parser.add_argument("--trajectory_fraction", type=float, default=0.5,
                        help="For trajectory_generalized, fraction of environments that use timed trajectory goals.")
    parser.add_argument("--spawn_z_min", type=float, default=0.0,
                        help="Minimum spawn z offset above the reference height in meters")
    parser.add_argument("--spawn_z_max", type=float, default=0.3,
                        help="Maximum spawn z offset above the reference height in meters")
    parser.add_argument("--z_reference_offset", type=float, default=0.0,
                        help="Virtual height offset. Example: 1.0 means user z=0.3 trains at world z=1.3")
    parser.add_argument("--hold_time", type=float, default=10.0, help="Required hold time in seconds")
    parser.add_argument("--goal_radius", type=float, default=0.03,
                        help="Goal radius in meters for precision hold fine-tuning")
    parser.add_argument("--episode_length_s", type=float, default=15.0, help="Episode length in seconds")
    parser.add_argument("--curriculum", action="store_true",
                        help="Use automatic hold-time curriculum during training")
    parser.add_argument("--curriculum_threshold", type=float, default=0.02,
                        help="Advance curriculum when hold events/env reaches this threshold")
    parser.add_argument("--curriculum_window", type=int, default=5,
                        help="Rolling window size for autonomous curriculum advancement")
    parser.add_argument("--num_envs", type=int, default=2048, help="Number of parallel environments")
    parser.add_argument("--max_iterations", type=int, default=300, help="Number of PPO iterations")
    parser.add_argument("--save_interval", type=int, default=25, help="Save checkpoint every N iterations")
    parser.add_argument("--proxy_hold_topk", type=int, default=64,
                        help="Use the mean hold streak of the top-K envs as the in-training selection metric")
    parser.add_argument("--steps_per_rollout", type=int, default=256, help="Steps per PPO rollout")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor")
    parser.add_argument("--seed", type=int, default=None, help="Optional torch seed")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args = parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from train_pointnav import (  # noqa: E402
    CrazyfliePointNavEnvCfg,
    CrazyfliePointNavEnv,
    L2FPPOAgent,
    compute_gae,
)


def default_checkpoint_path() -> str:
    return os.path.join(SCRIPT_DIR, "checkpoints_hold_finetune_phase3_20260323_161844", "best_model.pt")


def default_resume_checkpoint_path() -> str:
    return os.path.join(SCRIPT_DIR, "checkpoints_hold_finetune", "best_model.pt")


def resolve_warm_start_checkpoint() -> str:
    if args.checkpoint:
        return args.checkpoint
    if args.resume_best:
        return default_resume_checkpoint_path()
    return default_checkpoint_path()


def resolve_checkpoint_dir() -> str:
    dir_name = args.checkpoint_dir_name
    if args.timestamp_checkpoint_dir:
        dir_name = f"{dir_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return os.path.join(SCRIPT_DIR, dir_name)


def build_hold_curriculum(final_hold_time: float) -> list[float]:
    """Construct monotonic curriculum stages ending at final_hold_time."""
    base = [1.0, 3.0, 5.0, 10.0]
    stages = []
    for value in base:
        if value < final_hold_time - 1e-6:
            stages.append(value)
    stages.append(final_hold_time)

    deduped = []
    for value in stages:
        if not deduped or abs(deduped[-1] - value) > 1e-6:
            deduped.append(value)
    return deduped


def set_hold_time(env: CrazyfliePointNavEnv, hold_time_s: float):
    """Apply hold duration to the live environment config."""
    hold_steps = max(1, int(round(hold_time_s / env.cfg.sim.dt)))
    env.cfg.goal_hold_steps = hold_steps
    return hold_steps


def format_hold_tag(hold_time_s: float) -> str:
    """Create a filesystem-friendly label for curriculum checkpoints."""
    return f"{hold_time_s:.1f}s".replace(".", "p")


def load_timed_trajectory_from_csv(csv_path: str) -> list[tuple[float, float, float, float]]:
    """Load a timed local-frame trajectory as (t_rel, x, y, z)."""
    rows_out: list[tuple[float, float, float, float]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = ("t", "target.x", "target.y", "target.z")
        missing = [name for name in required if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Trajectory CSV missing columns: {', '.join(missing)}")

        for row in reader:
            try:
                t = float(row["t"])
                x = float(row["target.x"])
                y = float(row["target.y"])
                z = float(row["target.z"])
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(v) for v in (t, x, y, z)):
                continue
            rows_out.append((t, x, y, z))

    if args.trajectory_skip_initial > 0:
        rows_out = rows_out[args.trajectory_skip_initial:]

    if args.trajectory_min_spacing > 0.0:
        spaced_rows: list[tuple[float, float, float, float]] = []
        for row in rows_out:
            if not spaced_rows:
                spaced_rows.append(row)
                continue
            _, x, y, z = row
            _, px, py, pz = spaced_rows[-1]
            dist = math.sqrt((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2)
            if dist >= args.trajectory_min_spacing:
                spaced_rows.append(row)
        rows_out = spaced_rows

    stride = max(1, args.trajectory_stride)
    rows_out = rows_out[::stride]

    if not rows_out:
        raise ValueError(f"No finite trajectory targets found in {csv_path}")

    t0 = rows_out[0][0]
    return [(t - t0, x, y, z) for t, x, y, z in rows_out]


def apply_timed_trajectory_goals(
    env: CrazyfliePointNavEnv,
    schedule_t: torch.Tensor,
    schedule_xyz: torch.Tensor,
    env_mask: torch.Tensor | None = None,
):
    """Update each environment's goal from the timed trajectory using its episode clock."""
    if env_mask is None:
        env_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    if not torch.any(env_mask):
        return

    env_ids = env_mask.nonzero(as_tuple=False).squeeze(-1)
    episode_time = env.episode_length_buf[env_ids].to(torch.float32) * env.cfg.sim.dt
    waypoint_idx = torch.searchsorted(schedule_t, episode_time, right=True) - 1
    waypoint_idx = waypoint_idx.clamp(min=0, max=schedule_t.numel() - 1)

    current_goals_local = schedule_xyz[waypoint_idx]
    current_goals_world = env._terrain.env_origins[env_ids].clone()
    current_goals_world[:, :2] += current_goals_local[:, :2]
    current_goals_world[:, 2] = current_goals_local[:, 2] + args.z_reference_offset

    target_changed = None
    if hasattr(env, "_trajectory_last_idx"):
        prev_idx = env._trajectory_last_idx[env_ids]
        target_changed = waypoint_idx != prev_idx
        env._trajectory_last_idx[env_ids] = waypoint_idx.clone()
    else:
        env._trajectory_last_idx = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        env._trajectory_last_idx[env_ids] = waypoint_idx.clone()
        target_changed = torch.ones_like(waypoint_idx, dtype=torch.bool)

    env._goal_pos[env_ids] = current_goals_world

    if target_changed.any():
        changed_env_ids = env_ids[target_changed]
        pos_w = env._robot.data.root_pos_w
        env._prev_dist_xy[changed_env_ids] = env._goal_distance(pos_w[changed_env_ids], changed_env_ids)
        env._prev_height_below_target[changed_env_ids] = torch.clamp(
            env._goal_pos[changed_env_ids, 2] - pos_w[changed_env_ids, 2],
            min=0.0,
        )
        env._goal_reached[changed_env_ids] = False
        env._goal_held[changed_env_ids] = False
        env._goal_hold_counter[changed_env_ids] = 0

def build_env() -> CrazyfliePointNavEnv:
    target_z = args.target_z if args.target_z is not None else args.goal_z_min
    world_target_z = target_z + args.z_reference_offset
    use_fixed_goal = args.target_x is not None and args.target_y is not None
    spawn_height_min = args.z_reference_offset + args.spawn_z_min
    spawn_height_max = args.z_reference_offset + args.spawn_z_max
    uses_trajectory_schedule = args.task_mode in ("trajectory_track", "trajectory_generalized")
    trajectory_schedule = load_timed_trajectory_from_csv(args.trajectory_csv) if uses_trajectory_schedule else None
    if trajectory_schedule:
        _, first_x, first_y, first_z = trajectory_schedule[0]
        target_z = first_z
        world_target_z = target_z + args.z_reference_offset
        use_fixed_goal = False

    cfg = CrazyfliePointNavEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.debug_vis = False

    # One long episode for long-hold specialization.
    cfg.episode_length_s = args.episode_length_s
    cfg.goal_hold_steps = int(round(args.hold_time / cfg.sim.dt))
    cfg.goal_reach_threshold = args.goal_radius
    cfg.goal_height = world_target_z
    cfg.init_target_height = world_target_z
    cfg.use_3d_goal_distance = True

    if args.task_mode == "precision_hold":
        # Precision-lock task: start near the target, but not exactly on it, and
        # force the policy to stay inside a very tight 3D radius for a long time.
        cfg.init_guidance_probability = 0.0
        cfg.init_max_xy_offset = 0.08
        cfg.init_max_angle = 0.035
        cfg.init_max_linear_velocity = 0.03
        cfg.init_max_angular_velocity = 0.03
        cfg.init_height_offset_min = -0.02
        cfg.init_height_offset_max = 0.02

        # Reward shaping: touch matters a little, exact sustained hold matters a lot.
        cfg.hover_gate_radius = max(0.08, args.goal_radius * 3.0)
        cfg.hover_gate_min = 0.05
        cfg.nav_progress_weight = 2.0
        cfg.nav_reach_bonus = 10.0
        cfg.nav_hold_step_weight = 8.0
        cfg.nav_hold_bonus = 800.0
        cfg.nav_braking_radius = max(0.10, args.goal_radius * 4.0)
        cfg.hover_reward_scale = 0.4
        cfg.hover_reward_constant = 0.8
        cfg.hover_height_weight = 22.0
        cfg.hover_orientation_weight = 28.0
        cfg.hover_xy_velocity_weight = 3.0
        cfg.hover_z_velocity_weight = 6.0
        cfg.hover_angular_velocity_weight = 3.5
        cfg.hover_action_rate_weight = 0.10
        cfg.nav_height_track_weight = 4.0
        cfg.nav_height_recovery_weight = 0.25
        cfg.nav_speed_penalty_weight = 0.5
        cfg.nav_speed_penalty_threshold = 1.0
    elif args.task_mode == "mixed_nav_hold":
        # Mixed navigation-and-hold task: most episodes require travel in XY/Z,
        # some start near the goal, and the policy is still rewarded for
        # settling and staying there once it arrives.
        cfg.goal_reach_threshold = max(args.goal_radius, 0.06)
        cfg.goal_min_distance = 0.15
        cfg.goal_max_distance = 0.80
        cfg.goal_height = args.goal_z_min + args.z_reference_offset
        cfg.goal_height_min = args.goal_z_min + args.z_reference_offset
        cfg.goal_height_max = args.goal_z_max + args.z_reference_offset
        cfg.init_guidance_probability = 0.15
        cfg.init_max_xy_offset = 0.20
        cfg.init_max_angle = 0.10
        cfg.init_max_linear_velocity = 0.10
        cfg.init_max_angular_velocity = 0.10
        cfg.init_height_offset_min = -0.10
        cfg.init_height_offset_max = 0.10

        cfg.hover_gate_radius = max(0.20, cfg.goal_reach_threshold * 4.0)
        cfg.hover_gate_min = 0.15
        cfg.nav_progress_weight = 5.0
        cfg.nav_reach_bonus = 30.0
        cfg.nav_hold_step_weight = 2.5
        cfg.nav_hold_bonus = 250.0
        cfg.nav_braking_radius = max(0.20, cfg.goal_reach_threshold * 5.0)
        cfg.hover_reward_scale = 0.25
        cfg.hover_reward_constant = 0.7
        cfg.hover_height_weight = 10.0
        cfg.hover_orientation_weight = 18.0
        cfg.hover_xy_velocity_weight = 1.0
        cfg.hover_z_velocity_weight = 3.0
        cfg.hover_angular_velocity_weight = 1.5
        cfg.hover_action_rate_weight = 0.06
        cfg.nav_height_track_weight = 2.5
        cfg.nav_height_recovery_weight = 0.8
        cfg.nav_speed_penalty_weight = 0.15
        cfg.nav_speed_penalty_threshold = 2.0
    elif args.task_mode == "fast_nav_hold":
        # Speed-focused navigation-and-hold task: bias the policy toward
        # covering distance quickly while still requiring brief stable capture.
        cfg.goal_reach_threshold = max(args.goal_radius, 0.05)
        cfg.goal_min_distance = 0.20
        cfg.goal_max_distance = 1.00
        cfg.goal_height = args.goal_z_min + args.z_reference_offset
        cfg.goal_height_min = args.goal_z_min + args.z_reference_offset
        cfg.goal_height_max = args.goal_z_max + args.z_reference_offset
        cfg.init_guidance_probability = 0.05
        cfg.init_max_xy_offset = 0.25
        cfg.init_max_angle = 0.12
        cfg.init_max_linear_velocity = 0.12
        cfg.init_max_angular_velocity = 0.12
        cfg.init_height_offset_min = -0.12
        cfg.init_height_offset_max = 0.12

        # Require a shorter stable capture during the speed phase so the agent
        # learns to transition aggressively without giving up control.
        cfg.goal_hold_steps = max(1, int(round(min(args.hold_time, 3.0) / cfg.sim.dt)))

        cfg.hover_gate_radius = max(0.30, cfg.goal_reach_threshold * 4.0)
        cfg.hover_gate_min = 0.10
        cfg.nav_progress_weight = 8.0
        cfg.nav_reach_bonus = 40.0
        cfg.nav_hold_step_weight = 1.5
        cfg.nav_hold_bonus = 120.0
        cfg.nav_braking_radius = max(0.18, cfg.goal_reach_threshold * 4.0)
        cfg.hover_reward_scale = 0.20
        cfg.hover_reward_constant = 0.6
        cfg.hover_height_weight = 8.0
        cfg.hover_orientation_weight = 14.0
        cfg.hover_xy_velocity_weight = 0.6
        cfg.hover_z_velocity_weight = 2.0
        cfg.hover_angular_velocity_weight = 1.2
        cfg.hover_action_rate_weight = 0.04
        cfg.nav_height_track_weight = 1.5
        cfg.nav_height_recovery_weight = 0.6
        cfg.nav_speed_penalty_weight = 0.05
        cfg.nav_speed_penalty_threshold = 3.0
    elif args.task_mode == "trajectory_track":
        # Timed trajectory tracking: follow a moving 3D reference on schedule.
        if not args.trajectory_csv:
            raise ValueError("--task_mode trajectory_track requires --trajectory_csv")

        cfg.goal_reach_threshold = max(args.goal_radius, 0.10)
        # The base env validates sampled-goal invariants at construction time,
        # even though trajectory_track overrides the goal from the CSV on reset
        # and before each policy step. Keep a small non-trivial placeholder
        # range so initialization passes without affecting the live target path.
        cfg.goal_min_distance = max(args.goal_radius + 0.02, 0.12)
        cfg.goal_max_distance = cfg.goal_min_distance + 0.01
        cfg.goal_height = world_target_z
        cfg.goal_height_min = world_target_z
        cfg.goal_height_max = world_target_z
        cfg.init_guidance_probability = 0.0
        cfg.init_max_xy_offset = 0.05
        cfg.init_max_angle = 0.06
        cfg.init_max_linear_velocity = 0.05
        cfg.init_max_angular_velocity = 0.05
        cfg.init_height_offset_min = -0.03
        cfg.init_height_offset_max = 0.03

        # Moving target: emphasize progress/tracking and smooth flight,
        # de-emphasize "stop and hold forever at one point".
        cfg.goal_hold_steps = max(1, int(round(min(args.hold_time, 0.3) / cfg.sim.dt)))
        cfg.hover_gate_radius = max(0.25, cfg.goal_reach_threshold * 3.0)
        cfg.hover_gate_min = 0.10
        cfg.nav_progress_weight = 11.0
        cfg.nav_reach_bonus = 6.0
        cfg.nav_hold_step_weight = 0.35
        cfg.nav_hold_bonus = 10.0
        cfg.nav_braking_radius = max(0.18, cfg.goal_reach_threshold * 2.5)
        cfg.hover_reward_scale = 0.28
        cfg.hover_reward_constant = 0.5
        cfg.hover_height_weight = 0.0
        cfg.hover_orientation_weight = 18.0
        cfg.hover_xy_velocity_weight = 0.7
        cfg.hover_z_velocity_weight = 4.0
        cfg.hover_angular_velocity_weight = 2.0
        cfg.hover_action_rate_weight = 0.12
        cfg.nav_height_track_weight = 0.0
        cfg.nav_height_recovery_weight = 0.0
        cfg.nav_speed_penalty_weight = 0.02
        cfg.nav_speed_penalty_threshold = 4.0
    else:
        # Hybrid phase: keep broad random 3D navigation while dedicating a
        # subset of envs to the timed trajectory schedule.
        if not args.trajectory_csv:
            raise ValueError("--task_mode trajectory_generalized requires --trajectory_csv")

        cfg.goal_reach_threshold = max(args.goal_radius, 0.05)
        cfg.goal_min_distance = 0.18
        cfg.goal_max_distance = 0.90
        cfg.goal_height = args.goal_z_min + args.z_reference_offset
        cfg.goal_height_min = args.goal_z_min + args.z_reference_offset
        cfg.goal_height_max = args.goal_z_max + args.z_reference_offset
        cfg.init_guidance_probability = 0.10
        cfg.init_max_xy_offset = 0.18
        cfg.init_max_angle = 0.08
        cfg.init_max_linear_velocity = 0.08
        cfg.init_max_angular_velocity = 0.08
        cfg.init_height_offset_min = -0.08
        cfg.init_height_offset_max = 0.08

        cfg.goal_hold_steps = max(1, int(round(min(args.hold_time, 1.0) / cfg.sim.dt)))
        cfg.hover_gate_radius = max(0.22, cfg.goal_reach_threshold * 3.5)
        cfg.hover_gate_min = 0.10
        cfg.nav_progress_weight = 7.5
        cfg.nav_reach_bonus = 18.0
        cfg.nav_hold_step_weight = 1.0
        cfg.nav_hold_bonus = 40.0
        cfg.nav_braking_weight = 1.5
        cfg.nav_braking_radius = max(0.24, cfg.goal_reach_threshold * 4.0)
        cfg.hover_reward_scale = 0.26
        cfg.hover_reward_constant = 0.55
        cfg.hover_height_weight = 6.0
        cfg.hover_orientation_weight = 18.0
        cfg.hover_xy_velocity_weight = 1.2
        cfg.hover_z_velocity_weight = 5.5
        cfg.hover_angular_velocity_weight = 2.2
        cfg.hover_action_rate_weight = 0.14
        cfg.nav_height_track_weight = 1.5
        cfg.nav_height_recovery_weight = 0.5
        cfg.nav_height_descent_weight = 1.6
        cfg.nav_speed_penalty_weight = 0.05
        cfg.nav_speed_penalty_threshold = 2.8

    cfg.init_target_height_min = spawn_height_min
    cfg.init_target_height_max = spawn_height_max

    # Keep training robust but slightly calmer than the original pointnav run.
    cfg.enable_disturbance = False

    if use_fixed_goal:
        # Fixed-goal specialization: keep the same target every episode.
        fixed_goal = (args.target_x, args.target_y, world_target_z)
    else:
        # Random-goal specialization: preserve pointnav-style random goal
        # sampling while also varying goal height.
        if args.task_mode == "precision_hold":
            cfg.goal_min_distance = 0.2
            cfg.goal_max_distance = 0.7
            cfg.goal_height = args.goal_z_min + args.z_reference_offset
            cfg.goal_height_min = args.goal_z_min + args.z_reference_offset
            cfg.goal_height_max = args.goal_z_max + args.z_reference_offset
        fixed_goal = None

    env = CrazyfliePointNavEnv(cfg)
    env._eval_fixed_goal = fixed_goal
    env._goal_height_min = args.goal_z_min + args.z_reference_offset
    env._goal_height_max = args.goal_z_max + args.z_reference_offset
    env._spawn_height_min = spawn_height_min
    env._spawn_height_max = spawn_height_max
    if trajectory_schedule:
        env._trajectory_schedule_t = torch.tensor([row[0] for row in trajectory_schedule], dtype=torch.float32, device=env.device)
        env._trajectory_schedule_xyz = torch.tensor([row[1:] for row in trajectory_schedule], dtype=torch.float32, device=env.device)
        env._trajectory_last_idx = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        if args.task_mode == "trajectory_generalized":
            frac = min(max(args.trajectory_fraction, 0.0), 1.0)
            timed_envs = int(round(env.num_envs * frac))
            env._trajectory_env_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            env._trajectory_env_mask[:timed_envs] = True
        else:
            env._trajectory_env_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    return env


def train(env: CrazyfliePointNavEnv, agent: L2FPPOAgent):
    checkpoint_dir = resolve_checkpoint_dir()
    os.makedirs(checkpoint_dir, exist_ok=True)
    spawn_height_min = getattr(env, "_spawn_height_min", env.cfg.init_target_height_min)
    spawn_height_max = getattr(env, "_spawn_height_max", env.cfg.init_target_height_max)

    warm_ckpt = resolve_warm_start_checkpoint()
    if os.path.exists(warm_ckpt):
        loaded_iter, loaded_best = agent.load(warm_ckpt)
        print(f"[Warm Start] Loaded: {warm_ckpt}")
        print(f"[Warm Start] Source iteration: {loaded_iter}, source best reward: {loaded_best:.3f}")
    else:
        print(f"[Warm Start] Checkpoint not found: {warm_ckpt}")
        print("[Warm Start] Training from scratch instead.")

    target_z = args.target_z if args.target_z is not None else args.goal_z_min
    world_target_z = target_z + args.z_reference_offset
    use_fixed_goal = args.target_x is not None and args.target_y is not None
    uses_trajectory_schedule = args.task_mode in ("trajectory_track", "trajectory_generalized")
    print(f"\n{'='*60}")
    print("HOLD FINE-TUNING")
    print(f"{'='*60}")
    print(f"  Task mode:          {args.task_mode}")
    if args.task_mode == "trajectory_track":
        print(f"  Goal mode:          timed trajectory")
        print(f"  Trajectory CSV:     {args.trajectory_csv}")
        print(f"  Timed targets:      {env._trajectory_schedule_t.numel()}")
        first_goal = env._trajectory_schedule_xyz[0].tolist()
        print(f"  Initial goal (virtual): ({first_goal[0]:.2f}, {first_goal[1]:.2f}, {first_goal[2]:.2f}) m")
        print(f"  Initial goal (world):   ({first_goal[0]:.2f}, {first_goal[1]:.2f}, {first_goal[2] + args.z_reference_offset:.2f}) m")
        print(f"  Spawn z (world):    [{spawn_height_min:.2f}, {spawn_height_max:.2f}] m")
    elif args.task_mode == "trajectory_generalized":
        print(f"  Goal mode:          hybrid timed + random")
        print(f"  Trajectory CSV:     {args.trajectory_csv}")
        print(f"  Timed targets:      {env._trajectory_schedule_t.numel()}")
        print(f"  Timed env fraction: {env._trajectory_env_mask.float().mean().item() * 100.0:.1f}%")
        first_goal = env._trajectory_schedule_xyz[0].tolist()
        print(f"  Initial timed goal (virtual): ({first_goal[0]:.2f}, {first_goal[1]:.2f}, {first_goal[2]:.2f}) m")
        print(f"  Goal distance:      [{env.cfg.goal_min_distance:.2f}, {env.cfg.goal_max_distance:.2f}] m")
        print(f"  Goal z (virtual):   [{args.goal_z_min:.2f}, {args.goal_z_max:.2f}] m")
        print(f"  Goal z (world):     [{args.goal_z_min + args.z_reference_offset:.2f}, {args.goal_z_max + args.z_reference_offset:.2f}] m")
        print(f"  Spawn z (world):    [{spawn_height_min:.2f}, {spawn_height_max:.2f}] m")
    elif use_fixed_goal:
        print(f"  Goal mode:          fixed")
        print(f"  Goal (virtual):     ({args.target_x:.2f}, {args.target_y:.2f}, {target_z:.2f}) m")
        print(f"  Goal (world):       ({args.target_x:.2f}, {args.target_y:.2f}, {world_target_z:.2f}) m")
        print(f"  Spawn z (world):    [{spawn_height_min:.2f}, {spawn_height_max:.2f}] m")
    else:
        print(f"  Goal mode:          random")
        print(f"  Goal distance:      [{env.cfg.goal_min_distance:.2f}, {env.cfg.goal_max_distance:.2f}] m")
        print(f"  Goal z (virtual):   [{args.goal_z_min:.2f}, {args.goal_z_max:.2f}] m")
        print(f"  Goal z (world):     [{args.goal_z_min + args.z_reference_offset:.2f}, {args.goal_z_max + args.z_reference_offset:.2f}] m")
        print(f"  Spawn z (world):    [{spawn_height_min:.2f}, {spawn_height_max:.2f}] m")
    print(f"  Goal metric:        {'3D' if env.cfg.use_3d_goal_distance else 'XY'}")
    print(f"  Goal radius:        {env.cfg.goal_reach_threshold:.3f} m")
    print(f"  Hold duration:      {args.hold_time:.1f} s")
    print(f"  Episode length:     {args.episode_length_s:.1f} s")
    print(f"  Environments:       {env.num_envs}")
    print(f"  Max iterations:     {args.max_iterations}")
    print(f"  Steps per rollout:  {args.steps_per_rollout}")
    print(f"  Resume best:        {args.resume_best}")
    print(f"  Checkpoints:        {checkpoint_dir}")
    print(f"{'='*60}\n")

    obs_dict, _ = env.reset()
    if uses_trajectory_schedule:
        apply_timed_trajectory_goals(env, env._trajectory_schedule_t, env._trajectory_schedule_xyz, env._trajectory_env_mask)
        obs_dict = env._get_observations()
    obs = obs_dict["policy"]

    best_reward = float("-inf")
    best_hold_event_rate = 0.0
    best_proxy_hold_s = float("-inf")
    curriculum_stages = build_hold_curriculum(args.hold_time) if args.curriculum else [args.hold_time]
    curriculum_stage_idx = 0
    current_hold_time = curriculum_stages[curriculum_stage_idx]
    current_hold_steps = set_hold_time(env, current_hold_time)
    current_hold_tag = format_hold_tag(current_hold_time)
    recent_hold_rates = deque(maxlen=max(1, args.curriculum_window))

    def save_stage_checkpoint(name: str, iteration_idx: int):
        path = os.path.join(checkpoint_dir, f"{name}_{current_hold_tag}.pt")
        agent.save(path, iteration_idx, best_reward)
        return path

    # Save the warm-started policy under the initial curriculum target so we can
    # recover the exact starting point for this stage if later stages regress.
    save_stage_checkpoint("stage_start", -1)

    for iteration in range(args.max_iterations):
        obs_buffer = []
        action_buffer = []
        log_prob_buffer = []
        value_buffer = []
        reward_buffer = []
        done_buffer = []

        episode_rewards = torch.zeros(env.num_envs, device=env.device)
        hold_events = 0.0
        touch_events = 0.0
        rollout_max_hold_steps = 0.0
        rollout_topk_hold_steps = 0.0

        for _ in range(args.steps_per_rollout):
            if uses_trajectory_schedule:
                apply_timed_trajectory_goals(env, env._trajectory_schedule_t, env._trajectory_schedule_xyz, env._trajectory_env_mask)
                obs = env._get_observations()["policy"]
            prev_goal_reached = env._goal_reached.clone()
            prev_goal_held = env._goal_held.clone()

            action, log_prob, value = agent.get_action_and_value(obs)

            obs_buffer.append(obs)
            action_buffer.append(action)
            log_prob_buffer.append(log_prob)
            value_buffer.append(value)

            obs_dict, reward, terminated, truncated, _ = env.step(action)
            obs = obs_dict["policy"]
            done = terminated | truncated

            touch_events += (env._goal_reached & (~prev_goal_reached)).sum().item()
            hold_events += (env._goal_held & (~prev_goal_held)).sum().item()
            rollout_max_hold_steps = max(rollout_max_hold_steps, float(env._goal_hold_counter.max().item()))
            topk = min(max(1, args.proxy_hold_topk), env.num_envs)
            topk_mean = torch.topk(env._goal_hold_counter.float(), k=topk).values.mean().item()
            rollout_topk_hold_steps = max(rollout_topk_hold_steps, float(topk_mean))

            reward_buffer.append(reward)
            done_buffer.append(done)
            episode_rewards += reward

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

        loss = agent.update(
            obs_t.reshape(-1, obs_t.shape[-1]),
            actions_t.reshape(-1, actions_t.shape[-1]),
            log_probs_t.reshape(-1),
            returns_t.reshape(-1),
            advantages_t.reshape(-1),
        )

        mean_reward = episode_rewards.mean().item() / args.steps_per_rollout
        hold_event_rate = hold_events / max(env.num_envs, 1)
        touch_event_rate = touch_events / max(env.num_envs, 1)
        proxy_hold_s = rollout_topk_hold_steps * env.cfg.sim.dt
        proxy_hold_max_s = rollout_max_hold_steps * env.cfg.sim.dt
        std = torch.exp(agent.actor.log_std).mean().item()
        recent_hold_rates.append(hold_event_rate)

        is_best_reward = mean_reward > best_reward
        if is_best_reward:
            best_reward = mean_reward
            agent.save(os.path.join(checkpoint_dir, "best_model.pt"), iteration, best_reward)
            save_stage_checkpoint("best_reward", iteration)

        if hold_event_rate > best_hold_event_rate:
            best_hold_event_rate = hold_event_rate
            agent.save(os.path.join(checkpoint_dir, "best_hold_model.pt"), iteration, best_reward)
            save_stage_checkpoint("best_hold", iteration)

        if proxy_hold_s > best_proxy_hold_s:
            best_proxy_hold_s = proxy_hold_s
            agent.save(os.path.join(checkpoint_dir, "best_proxy_hold_model.pt"), iteration, best_reward)
            save_stage_checkpoint("best_proxy_hold", iteration)

        if iteration % 10 == 0 or is_best_reward:
            star = " *BEST*" if is_best_reward else ""
            print(
                f"[Iter {iteration:4d}] Reward: {mean_reward:8.3f} | "
                f"HoldEvt/env: {hold_event_rate:5.3f} | TouchEvt/env: {touch_event_rate:5.3f} | "
                f"ProxyHold(top{min(max(1, args.proxy_hold_topk), env.num_envs)}): {proxy_hold_s:4.2f}s | "
                f"ProxyMax: {proxy_hold_max_s:4.2f}s | "
                f"HoldTarget: {current_hold_time:4.1f}s ({current_hold_steps:4d} steps) | "
                f"Std: {std:.3f} | Loss: {loss:.4f}{star}"
            )

        if args.curriculum and curriculum_stage_idx < len(curriculum_stages) - 1:
            mean_recent_hold = sum(recent_hold_rates) / len(recent_hold_rates)
            if len(recent_hold_rates) == recent_hold_rates.maxlen and mean_recent_hold >= args.curriculum_threshold:
                completed_hold_time = current_hold_time
                completed_hold_tag = current_hold_tag
                completed_steps = current_hold_steps
                promotion_checkpoint = os.path.join(
                    checkpoint_dir, f"stage_complete_{completed_hold_tag}.pt"
                )
                agent.save(promotion_checkpoint, iteration, best_reward)

                curriculum_stage_idx += 1
                current_hold_time = curriculum_stages[curriculum_stage_idx]
                current_hold_steps = set_hold_time(env, current_hold_time)
                current_hold_tag = format_hold_tag(current_hold_time)
                recent_hold_rates.clear()
                next_stage_start = save_stage_checkpoint("stage_start", iteration)
                print(
                    f"[Curriculum] Completed hold target {completed_hold_time:.1f}s "
                    f"({completed_steps} steps); saved {os.path.basename(promotion_checkpoint)}"
                )
                print(
                    f"[Curriculum] Advanced to hold target {current_hold_time:.1f}s "
                    f"({current_hold_steps} steps) at iteration {iteration}; "
                    f"saved {os.path.basename(next_stage_start)}"
                )

        if iteration > 0 and iteration % 50 == 0:
            ep_stats = env.get_episode_length_stats()
            print(
                f"  Episode Length: mean={ep_stats['mean']:.1f} p50={ep_stats['p50']:.1f} "
                f"p90={ep_stats['p90']:.1f} (n={ep_stats['count']})"
            )
            tc = env._term_counters
            total = max(tc["total"], 1)
            print(
                f"  Terms: xy:{tc['xy_exceeded']/total*100:.1f}% low:{tc['too_low']/total*100:.1f}% "
                f"high:{tc['too_high']/total*100:.1f}% tilt:{tc['too_tilted']/total*100:.1f}% "
                f"linvel:{tc['lin_vel_exceeded']/total*100:.1f}% angvel:{tc['ang_vel_exceeded']/total*100:.1f}% "
                f"goal:{tc['goal_reached']/total*100:.1f}% timeout:{tc['timeout']/total*100:.1f}%"
            )
            env.clear_episode_stats()

        if iteration > 0 and iteration % args.save_interval == 0:
            agent.save(os.path.join(checkpoint_dir, f"checkpoint_{iteration}.pt"), iteration, best_reward)

    agent.save(os.path.join(checkpoint_dir, "final_model.pt"), args.max_iterations, best_reward)
    print(
        f"\nFine-tuning complete. Best reward: {best_reward:.3f} | "
        f"Best hold events/env: {best_hold_event_rate:.3f} | "
        f"Best proxy hold: {best_proxy_hold_s:.2f}s | "
        f"Final hold target: {current_hold_time:.1f}s"
    )
    print(f"Checkpoints saved to: {checkpoint_dir}")


def main():
    if args.seed is not None:
        torch.manual_seed(args.seed)

    env = build_env()
    agent = L2FPPOAgent(
        obs_dim=env.cfg.observation_space,
        action_dim=env.cfg.action_space,
        device=env.device,
        lr=args.lr,
        gamma=args.gamma,
    )

    try:
        train(env, agent)
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
