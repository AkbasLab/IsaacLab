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
import atexit
import csv
import math
import os
import sys
import tempfile
import traceback
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


DEBUG_ENABLED = os.environ.get("ISAACLAB_HOLD_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")
DEBUG_LAST_STAGE = "script_import"


def debug_mark(stage: str, **kwargs):
    global DEBUG_LAST_STAGE
    DEBUG_LAST_STAGE = stage
    if not DEBUG_ENABLED:
        return
    if kwargs:
        details = " ".join(f"{key}={value}" for key, value in kwargs.items())
        print(f"[Debug] {stage} | {details}", flush=True)
    else:
        print(f"[Debug] {stage}", flush=True)


def _debug_exit_report():
    if DEBUG_ENABLED:
        print(f"[Debug] Process exiting. Last stage: {DEBUG_LAST_STAGE}", flush=True)


atexit.register(_debug_exit_report)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune pointnav checkpoint for fixed-position hold")
    parser.add_argument("--checkpoint", type=str, default=None, help="Warm-start checkpoint path")
    parser.add_argument("--task_mode", type=str, default="trajectory_generalized", choices=("precision_hold", "trajectory_generalized", "sequence_nav_precision", "unified_motion_curriculum"),
                        help="Training task mode. 'precision_hold' preserves tight hold specialization. 'trajectory_generalized' mixes timed trajectory envs with random 3D navigation envs. 'sequence_nav_precision' trains continuous multi-goal point-to-point travel with sharper turns and farther targets. 'unified_motion_curriculum' mixes single-goal, multi-goal, and structured snippet tasks under a staged curriculum.")
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
                        help="CSV with t,target.x,target.y,target.z columns for trajectory_generalized mode.")
    parser.add_argument("--trajectory_skip_initial", type=int, default=0,
                        help="Skip this many initial finite target rows from the trajectory CSV.")
    parser.add_argument("--trajectory_min_spacing", type=float, default=0.0,
                        help="Minimum 3D spacing in meters between kept trajectory targets.")
    parser.add_argument("--trajectory_stride", type=int, default=1,
                        help="Keep every Nth trajectory target after spacing.")
    parser.add_argument("--trajectory_fraction", type=float, default=0.5,
                        help="For trajectory_generalized, fraction of environments that use timed trajectory goals.")
    parser.add_argument("--sequence_min_distance", type=float, default=0.35,
                        help="For sequence_nav_precision, minimum distance from current position to the next goal.")
    parser.add_argument("--sequence_max_distance", type=float, default=0.90,
                        help="For sequence_nav_precision, maximum distance from current position to the next goal.")
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
    parser.add_argument("--mini_batch_size", type=int, default=4096, help="Mini-batch size for PPO updates")
    parser.add_argument("--ppo_epochs", type=int, default=8, help="Number of PPO optimization epochs per rollout")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy bonus coefficient for PPO")
    parser.add_argument("--target_kl", type=float, default=0.03, help="Approximate KL threshold for early-stopping PPO epochs")
    parser.add_argument("--preserve_optimizer_state", action="store_true",
                        help="Keep Adam moments from the warm-start checkpoint instead of resetting them for finetuning")
    parser.add_argument("--seed", type=int, default=None, help="Optional torch seed")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args = parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def _cli_flag_provided(flag_name: str) -> bool:
    return any(arg == flag_name or arg.startswith(f"{flag_name}=") for arg in sys.argv[1:])


USER_PROVIDED_GOAL_RADIUS = _cli_flag_provided("--goal_radius")

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


UNIFIED_SINGLE_FAMILIES = (
    "single_translation",
    "single_altitude_change",
    "single_diagonal",
    "single_far_retarget",
)
UNIFIED_SEQUENCE_FAMILIES = (
    "sharp_turn_chain",
    "switchback_chain",
    "zigzag_chain",
    "climb_turn_chain",
    "momentum_reversal_chain",
)
UNIFIED_SNIPPET_FAMILIES = ("arc_snippet", "figure8_snippet")
UNIFIED_ALL_FAMILIES = UNIFIED_SINGLE_FAMILIES + UNIFIED_SEQUENCE_FAMILIES + UNIFIED_SNIPPET_FAMILIES
UNIFIED_FAMILY_TO_ID = {name: idx for idx, name in enumerate(UNIFIED_ALL_FAMILIES)}
UNIFIED_ID_TO_FAMILY = {idx: name for name, idx in UNIFIED_FAMILY_TO_ID.items()}
UNIFIED_MAX_CHAIN_GOALS = 8


def clamp_local_goal(x: float, y: float, z: float, xy_limit: float, z_min: float, z_max: float) -> tuple[float, float, float]:
    return (
        max(min(x, xy_limit), -xy_limit),
        max(min(y, xy_limit), -xy_limit),
        max(min(z, z_max), z_min),
    )


def sample_distance_bucket(stage_cfg: dict[str, object], device: torch.device) -> tuple[float, float]:
    bucket_roll = torch.rand(1, device=device).item()
    cumulative = 0.0
    for name, prob in stage_cfg["distance_mix"]:
        cumulative += prob
        if bucket_roll <= cumulative:
            if name == "close":
                return 0.10, 0.25
            if name == "medium":
                return 0.25, 0.50
            return 0.50, 0.90
    return 0.25, 0.50


def sample_turn_bucket(stage_cfg: dict[str, object], device: torch.device) -> tuple[float, float]:
    turn_roll = torch.rand(1, device=device).item()
    cumulative = 0.0
    for name, prob in stage_cfg["turn_mix"]:
        cumulative += prob
        if turn_roll <= cumulative:
            if name == "gentle":
                return math.radians(30.0), math.radians(60.0)
            if name == "medium":
                return math.radians(60.0), math.radians(120.0)
            return math.radians(120.0), math.radians(170.0)
    return math.radians(60.0), math.radians(120.0)


def sample_weighted_name(weighted_names: tuple[tuple[str, float], ...], device: torch.device) -> str:
    roll = torch.rand(1, device=device).item()
    cumulative = 0.0
    for name, weight in weighted_names:
        cumulative += weight
        if roll <= cumulative:
            return name
    return weighted_names[-1][0]


def hold_time_to_steps(env: CrazyfliePointNavEnv, hold_time_s: float) -> int:
    return max(1, int(round(hold_time_s / env.cfg.sim.dt)))


def get_unified_stage(iteration: int, max_iterations: int) -> dict[str, object]:
    def resolve_stage_goal_radius(stage_default: float) -> float:
        # Respect an explicit CLI override. Otherwise keep the staged default
        # lower bound that makes the early curriculum more forgiving.
        if USER_PROVIDED_GOAL_RADIUS:
            return args.goal_radius
        return max(args.goal_radius, stage_default)

    progress = 0.0 if max_iterations <= 1 else iteration / max(max_iterations - 1, 1)
    if progress < 0.40:
        return {
            "name": "stable_translation",
            "single_weight": 0.70,
            "sequence_weight": 0.20,
            "snippet_weight": 0.10,
            "single_family_mix": (
                ("single_translation", 0.18),
                ("single_altitude_change", 0.32),
                ("single_diagonal", 0.32),
                ("single_far_retarget", 0.18),
            ),
            "sequence_family_mix": (
                ("sharp_turn_chain", 0.16),
                ("switchback_chain", 0.10),
                ("zigzag_chain", 0.12),
                ("climb_turn_chain", 0.40),
                ("momentum_reversal_chain", 0.22),
            ),
            "snippet_family_mix": (("arc_snippet", 0.35), ("figure8_snippet", 0.65)),
            "distance_mix": (("close", 0.65), ("medium", 0.30), ("far", 0.05)),
            "turn_mix": (("gentle", 0.80), ("medium", 0.17), ("sharp", 0.03)),
            "goal_radius": resolve_stage_goal_radius(0.08),
            "single_hold_time": 1.5,
            "sequence_hold_time": 0.35,
            "snippet_hold_time": 0.35,
        }
    if progress < 0.80:
        return {
            "name": "chained_motion",
            "single_weight": 0.50,
            "sequence_weight": 0.35,
            "snippet_weight": 0.15,
            "single_family_mix": (
                ("single_translation", 0.15),
                ("single_altitude_change", 0.34),
                ("single_diagonal", 0.31),
                ("single_far_retarget", 0.20),
            ),
            "sequence_family_mix": (
                ("sharp_turn_chain", 0.18),
                ("switchback_chain", 0.10),
                ("zigzag_chain", 0.14),
                ("climb_turn_chain", 0.38),
                ("momentum_reversal_chain", 0.20),
            ),
            "snippet_family_mix": (("arc_snippet", 0.30), ("figure8_snippet", 0.70)),
            "distance_mix": (("close", 0.35), ("medium", 0.45), ("far", 0.20)),
            "turn_mix": (("gentle", 0.35), ("medium", 0.45), ("sharp", 0.20)),
            "goal_radius": resolve_stage_goal_radius(0.06),
            "single_hold_time": 1.0,
            "sequence_hold_time": 0.25,
            "snippet_hold_time": 0.25,
        }
    return {
        "name": "sharp_curves",
        "single_weight": 0.30,
        "sequence_weight": 0.45,
        "snippet_weight": 0.25,
        "single_family_mix": (
            ("single_translation", 0.14),
            ("single_altitude_change", 0.34),
            ("single_diagonal", 0.32),
            ("single_far_retarget", 0.20),
        ),
        "sequence_family_mix": (
            ("sharp_turn_chain", 0.20),
            ("switchback_chain", 0.10),
            ("zigzag_chain", 0.16),
            ("climb_turn_chain", 0.34),
            ("momentum_reversal_chain", 0.20),
        ),
        "snippet_family_mix": (("arc_snippet", 0.25), ("figure8_snippet", 0.75)),
        "distance_mix": (("close", 0.20), ("medium", 0.35), ("far", 0.45)),
        "turn_mix": (("gentle", 0.12), ("medium", 0.38), ("sharp", 0.50)),
        "goal_radius": resolve_stage_goal_radius(0.05),
        "single_hold_time": 0.75,
        "sequence_hold_time": 0.20,
        "snippet_hold_time": 0.20,
    }


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
        if hasattr(env, "_prev_height_above_target"):
            env._prev_height_above_target[changed_env_ids] = torch.clamp(
                pos_w[changed_env_ids, 2] - env._goal_pos[changed_env_ids, 2],
                min=0.0,
            )
        env._goal_reached[changed_env_ids] = False
        env._goal_held[changed_env_ids] = False
        env._goal_hold_counter[changed_env_ids] = 0
        set_goal_hold_steps_batch(env, changed_env_ids, env.cfg.goal_hold_steps)


def sample_sequence_goals(env: CrazyfliePointNavEnv, env_ids: torch.Tensor):
    """Sample the next goal from the current position to encourage continuous travel and turning."""
    if env_ids.numel() == 0:
        return

    device = env.device
    pos_w = env._robot.data.root_pos_w[env_ids]
    origins = env._terrain.env_origins[env_ids]
    current_local = pos_w[:, :3] - origins[:, :3]
    min_dist = float(args.sequence_min_distance)
    max_dist = float(args.sequence_max_distance)
    xy_limit = max(0.75, env.cfg.obs_position_clip - 0.10)
    z_min = args.goal_z_min + args.z_reference_offset
    z_max = args.goal_z_max + args.z_reference_offset

    new_goals = pos_w.clone()
    prev_dir = getattr(env, "_sequence_prev_dir", None)

    for local_idx, env_id in enumerate(env_ids.tolist()):
        current_xy = current_local[local_idx, :2]
        current_z = pos_w[local_idx, 2]
        accepted = False
        for _ in range(12):
            dist = torch.empty(1, device=device).uniform_(min_dist, max_dist).item()
            if prev_dir is not None and torch.norm(prev_dir[env_id]).item() > 1e-4:
                base_angle = math.atan2(float(prev_dir[env_id, 1].item()), float(prev_dir[env_id, 0].item()))
                turn_sign = -1.0 if torch.rand(1, device=device).item() < 0.5 else 1.0
                turn_mag = torch.empty(1, device=device).uniform_(math.pi / 3.0, math.pi).item()
                angle = base_angle + turn_sign * turn_mag
            else:
                angle = torch.empty(1, device=device).uniform_(-math.pi, math.pi).item()

            delta_xy = torch.tensor([math.cos(angle), math.sin(angle)], device=device) * dist
            goal_xy = current_xy + delta_xy
            goal_z = torch.empty(1, device=device).uniform_(z_min, z_max).item()

            if abs(float(goal_xy[0].item())) > xy_limit or abs(float(goal_xy[1].item())) > xy_limit:
                continue
            if abs(goal_z - current_z) < 0.08:
                continue

            new_goals[local_idx, 0] = origins[local_idx, 0] + goal_xy[0]
            new_goals[local_idx, 1] = origins[local_idx, 1] + goal_xy[1]
            new_goals[local_idx, 2] = goal_z
            if prev_dir is not None:
                env._sequence_prev_dir[env_id] = delta_xy
            accepted = True
            break

        if not accepted:
            goal_xy = current_xy.clone()
            goal_xy[0] = torch.clamp(goal_xy[0] + 0.5, min=-xy_limit, max=xy_limit)
            goal_z = min(max(current_z + 0.12, z_min), z_max)
            new_goals[local_idx, 0] = origins[local_idx, 0] + goal_xy[0]
            new_goals[local_idx, 1] = origins[local_idx, 1] + goal_xy[1]
            new_goals[local_idx, 2] = goal_z
            if prev_dir is not None:
                env._sequence_prev_dir[env_id] = goal_xy - current_xy

    env._goal_pos[env_ids] = new_goals
    env._prev_dist_xy[env_ids] = env._goal_distance(env._robot.data.root_pos_w[env_ids], env_ids)
    env._prev_speed[env_ids] = 0.0
    env._prev_height_below_target[env_ids] = torch.clamp(env._goal_pos[env_ids, 2] - env._robot.data.root_pos_w[env_ids, 2], min=0.0)
    if hasattr(env, "_prev_height_above_target"):
        env._prev_height_above_target[env_ids] = torch.clamp(env._robot.data.root_pos_w[env_ids, 2] - env._goal_pos[env_ids, 2], min=0.0)
    env._goal_reached[env_ids] = False
    env._goal_held[env_ids] = False
    env._goal_hold_counter[env_ids] = 0
    set_goal_hold_steps_batch(env, env_ids, env.cfg.goal_hold_steps)


def initialize_unified_buffers(env: CrazyfliePointNavEnv):
    env._continue_on_goal_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    env._unified_family = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    env._unified_sequence_goals = torch.zeros((env.num_envs, UNIFIED_MAX_CHAIN_GOALS, 3), dtype=torch.float32, device=env.device)
    env._unified_sequence_len = torch.ones(env.num_envs, dtype=torch.long, device=env.device)
    env._unified_sequence_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._unified_stage_name = "stable_translation"
    env._unified_completion_bonus_count = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)


def set_goal_hold_steps_batch(
    env: CrazyfliePointNavEnv,
    env_ids: torch.Tensor,
    hold_steps: int | torch.Tensor,
):
    if env_ids.numel() == 0 or not hasattr(env, "_goal_hold_steps"):
        return
    if isinstance(hold_steps, torch.Tensor):
        env._goal_hold_steps[env_ids] = hold_steps.to(device=env.device, dtype=env._goal_hold_steps.dtype)
    else:
        env._goal_hold_steps[env_ids] = int(hold_steps)


def set_goal_batch(
    env: CrazyfliePointNavEnv,
    env_ids: torch.Tensor,
    goal_world: torch.Tensor,
    segment_start_world: torch.Tensor | None = None,
    hold_steps: int | torch.Tensor | None = None,
):
    if env_ids.numel() == 0:
        return
    env._goal_pos[env_ids] = goal_world
    pos_w = env._robot.data.root_pos_w
    if segment_start_world is None:
        segment_start_world = pos_w[env_ids].clone()
    env._path_start_pos[env_ids] = segment_start_world
    env._path_goal_pos[env_ids] = goal_world.clone()
    env._prev_path_progress[env_ids] = 0.0
    env._prev_dist_xy[env_ids] = env._goal_distance(pos_w[env_ids], env_ids)
    env._prev_speed[env_ids] = 0.0
    env._prev_height_below_target[env_ids] = torch.clamp(env._goal_pos[env_ids, 2] - pos_w[env_ids, 2], min=0.0)
    if hasattr(env, "_prev_height_above_target"):
        env._prev_height_above_target[env_ids] = torch.clamp(pos_w[env_ids, 2] - env._goal_pos[env_ids, 2], min=0.0)
    env._goal_reached[env_ids] = False
    env._goal_held[env_ids] = False
    env._goal_hold_counter[env_ids] = 0
    if hasattr(env, "_prev_goal_axis_sign"):
        segment_vec = env._path_goal_pos[env_ids] - env._path_start_pos[env_ids]
        safe_len = torch.clamp(torch.norm(segment_vec, dim=-1), min=1e-4)
        segment_dir = segment_vec / safe_len.unsqueeze(-1)
        axis_remaining = safe_len - ((env._goal_pos[env_ids] - pos_w[env_ids]) * segment_dir).sum(dim=-1)
        env._prev_goal_axis_sign[env_ids] = torch.sign(axis_remaining)
    if hasattr(env, "_latest_path_deviation"):
        env._latest_path_deviation[env_ids] = 0.0
    if hasattr(env, "_latest_overshoot"):
        env._latest_overshoot[env_ids] = 0.0
    if hasattr(env, "_latest_corner_speed"):
        env._latest_corner_speed[env_ids] = 0.0
    if hold_steps is not None:
        set_goal_hold_steps_batch(env, env_ids, hold_steps)


def sample_single_goal_world(
    env: CrazyfliePointNavEnv,
    env_id: int,
    family_name: str,
    stage_cfg: dict[str, object],
) -> torch.Tensor:
    device = env.device
    pos_w = env._robot.data.root_pos_w[env_id]
    origin = env._terrain.env_origins[env_id]
    current_local = pos_w[:3] - origin[:3]
    xy_limit = max(0.75, env.cfg.obs_position_clip - 0.10)
    z_min = args.goal_z_min + args.z_reference_offset
    z_max = args.goal_z_max + args.z_reference_offset
    dist_min, dist_max = sample_distance_bucket(stage_cfg, device)
    dist = torch.empty(1, device=device).uniform_(dist_min, dist_max).item()
    current_x = float(current_local[0].item())
    current_y = float(current_local[1].item())
    current_z = float(pos_w[2].item())

    if family_name == "single_translation":
        angle = torch.empty(1, device=device).uniform_(-math.pi, math.pi).item()
        dx = math.cos(angle) * dist
        dy = math.sin(angle) * dist
        goal_x, goal_y, goal_z = clamp_local_goal(current_x + dx, current_y + dy, current_z, xy_limit, z_min, z_max)
    elif family_name == "single_altitude_change":
        delta_z = dist if torch.rand(1, device=device).item() < 0.5 else -dist
        goal_x, goal_y, goal_z = clamp_local_goal(current_x, current_y, current_z + delta_z, xy_limit, z_min, z_max)
    elif family_name == "single_diagonal":
        angle = torch.empty(1, device=device).uniform_(-math.pi, math.pi).item()
        horiz = dist * 0.75
        vert = dist * 0.60 * (1.0 if torch.rand(1, device=device).item() < 0.5 else -1.0)
        goal_x, goal_y, goal_z = clamp_local_goal(
            current_x + math.cos(angle) * horiz,
            current_y + math.sin(angle) * horiz,
            current_z + vert,
            xy_limit,
            z_min,
            z_max,
        )
    else:  # single_far_retarget
        angle = torch.empty(1, device=device).uniform_(-math.pi, math.pi).item()
        horiz = dist * torch.empty(1, device=device).uniform_(0.70, 1.00).item()
        vert = dist * torch.empty(1, device=device).uniform_(0.05, 0.35).item()
        vert *= 1.0 if torch.rand(1, device=device).item() < 0.5 else -1.0
        goal_x, goal_y, goal_z = clamp_local_goal(
            current_x + math.cos(angle) * horiz,
            current_y + math.sin(angle) * horiz,
            current_z + vert,
            xy_limit,
            z_min,
            z_max,
        )

    goal_world = origin.clone()
    goal_world[0] += goal_x
    goal_world[1] += goal_y
    goal_world[2] = goal_z
    return goal_world


def build_motion_snippet_world(
    env: CrazyfliePointNavEnv,
    env_id: int,
    family_name: str,
    stage_cfg: dict[str, object],
) -> tuple[torch.Tensor, int]:
    device = env.device
    pos_w = env._robot.data.root_pos_w[env_id]
    origin = env._terrain.env_origins[env_id]
    xy_limit = max(0.75, env.cfg.obs_position_clip - 0.10)
    z_min = args.goal_z_min + args.z_reference_offset
    z_max = args.goal_z_max + args.z_reference_offset
    seq_len = 6 if family_name == "figure8_snippet" else 5
    goals = torch.zeros((UNIFIED_MAX_CHAIN_GOALS, 3), dtype=torch.float32, device=device)
    local = (pos_w[:3] - origin[:3]).clone()
    heading = torch.empty(1, device=device).uniform_(-math.pi, math.pi).item()
    radius = torch.empty(1, device=device).uniform_(0.12, 0.28).item()
    z_wave = torch.empty(1, device=device).uniform_(0.05, 0.16).item()

    for idx in range(seq_len):
        phase = (idx + 1) / seq_len
        if family_name == "figure8_snippet":
            x_offset = radius * math.sin(2.0 * math.pi * phase)
            y_offset = 0.55 * radius * math.sin(4.0 * math.pi * phase)
        else:
            x_offset = radius * math.sin(math.pi * phase)
            y_offset = radius * (1.0 - math.cos(math.pi * phase))

        rot_x = math.cos(heading) * x_offset - math.sin(heading) * y_offset
        rot_y = math.sin(heading) * x_offset + math.cos(heading) * y_offset
        local_x, local_y, local_z = clamp_local_goal(
            float(local[0].item()) + rot_x,
            float(local[1].item()) + rot_y,
            float(local[2].item()) + z_wave * math.sin(2.0 * math.pi * phase),
            xy_limit,
            z_min,
            z_max,
        )
        goals[idx, 0] = origin[0] + local_x
        goals[idx, 1] = origin[1] + local_y
        goals[idx, 2] = local_z

    return goals, seq_len


def build_sequence_chain_world(
    env: CrazyfliePointNavEnv,
    env_id: int,
    family_name: str,
    stage_cfg: dict[str, object],
) -> tuple[torch.Tensor, int]:
    device = env.device
    pos_w = env._robot.data.root_pos_w[env_id]
    origin = env._terrain.env_origins[env_id]
    xy_limit = max(0.75, env.cfg.obs_position_clip - 0.10)
    z_min = args.goal_z_min + args.z_reference_offset
    z_max = args.goal_z_max + args.z_reference_offset
    seq_len = int(torch.randint(3, 7, (1,), device=device).item())
    goals = torch.zeros((UNIFIED_MAX_CHAIN_GOALS, 3), dtype=torch.float32, device=device)
    local_x = float((pos_w[0] - origin[0]).item())
    local_y = float((pos_w[1] - origin[1]).item())
    local_z = float(pos_w[2].item())
    heading = torch.empty(1, device=device).uniform_(-math.pi, math.pi).item()
    prev_turn_sign = 1.0

    for idx in range(seq_len):
        if family_name == "switchback_chain":
            if idx % 2 == 0:
                dist_min, dist_max = 0.50, 0.90
            else:
                dist_min, dist_max = 0.10, 0.35
        else:
            dist_min, dist_max = sample_distance_bucket(stage_cfg, device)
        dist = torch.empty(1, device=device).uniform_(dist_min, dist_max).item()
        turn_min, turn_max = sample_turn_bucket(stage_cfg, device)
        if idx > 0:
            if family_name == "momentum_reversal_chain":
                turn = math.pi - torch.empty(1, device=device).uniform_(math.radians(10.0), math.radians(30.0)).item()
            elif family_name == "zigzag_chain":
                prev_turn_sign *= -1.0
                turn = prev_turn_sign * torch.empty(1, device=device).uniform_(math.radians(70.0), math.radians(120.0)).item()
            elif family_name == "sharp_turn_chain":
                turn = (1.0 if torch.rand(1, device=device).item() < 0.5 else -1.0) * torch.empty(1, device=device).uniform_(math.radians(120.0), math.radians(170.0)).item()
            else:
                turn = (1.0 if torch.rand(1, device=device).item() < 0.5 else -1.0) * torch.empty(1, device=device).uniform_(turn_min, turn_max).item()
            heading += turn

        horiz_scale = 1.0
        dz = 0.0
        if family_name == "climb_turn_chain":
            horiz_scale = 0.45 if idx % 2 == 0 else 0.65
            dz = dist * (0.90 if idx % 2 == 0 else 0.60) * (1.0 if idx % 2 == 0 else -1.0)
        else:
            if idx == 0 and torch.rand(1, device=device).item() < 0.35:
                dz = dist * 0.50 * (1.0 if torch.rand(1, device=device).item() < 0.5 else -1.0)
            elif idx > 0 and torch.rand(1, device=device).item() < 0.50:
                dz = dist * 0.45 * (1.0 if torch.rand(1, device=device).item() < 0.5 else -1.0)

        local_x, local_y, local_z = clamp_local_goal(
            local_x + math.cos(heading) * dist * horiz_scale,
            local_y + math.sin(heading) * dist * horiz_scale,
            local_z + dz,
            xy_limit,
            z_min,
            z_max,
        )
        goals[idx, 0] = origin[0] + local_x
        goals[idx, 1] = origin[1] + local_y
        goals[idx, 2] = local_z

    return goals, seq_len


def sample_unified_tasks(env: CrazyfliePointNavEnv, env_ids: torch.Tensor, stage_cfg: dict[str, object]):
    if env_ids.numel() == 0:
        return
    device = env.device
    env._unified_stage_name = str(stage_cfg["name"])
    for env_id in env_ids.tolist():
        roll = torch.rand(1, device=device).item()
        single_weight = float(stage_cfg["single_weight"])
        sequence_weight = float(stage_cfg["sequence_weight"])
        if roll < single_weight:
            family_name = sample_weighted_name(stage_cfg["single_family_mix"], device)
            hold_steps = hold_time_to_steps(env, float(stage_cfg["single_hold_time"]))
            env._unified_family[env_id] = UNIFIED_FAMILY_TO_ID[family_name]
            # Single-goal families complete after one clean arrival/hold.
            env._continue_on_goal_mask[env_id] = False
            env._unified_sequence_len[env_id] = 1
            env._unified_sequence_idx[env_id] = 0
            goal_world = sample_single_goal_world(env, env_id, family_name, stage_cfg)
            env._unified_sequence_goals[env_id, 0] = goal_world
            set_goal_batch(env, torch.tensor([env_id], device=device), goal_world.unsqueeze(0), hold_steps=hold_steps)
        elif roll < single_weight + sequence_weight:
            family_name = sample_weighted_name(stage_cfg["sequence_family_mix"], device)
            hold_steps = hold_time_to_steps(env, float(stage_cfg["sequence_hold_time"]))
            env._unified_family[env_id] = UNIFIED_FAMILY_TO_ID[family_name]
            env._continue_on_goal_mask[env_id] = True
            goals, seq_len = build_sequence_chain_world(env, env_id, family_name, stage_cfg)
            env._unified_sequence_goals[env_id] = 0.0
            env._unified_sequence_goals[env_id, :seq_len] = goals[:seq_len]
            env._unified_sequence_len[env_id] = seq_len
            env._unified_sequence_idx[env_id] = 0
            set_goal_batch(env, torch.tensor([env_id], device=device), goals[0].unsqueeze(0), hold_steps=hold_steps)
        else:
            family_name = sample_weighted_name(stage_cfg["snippet_family_mix"], device)
            hold_steps = hold_time_to_steps(env, float(stage_cfg["snippet_hold_time"]))
            env._unified_family[env_id] = UNIFIED_FAMILY_TO_ID[family_name]
            env._continue_on_goal_mask[env_id] = True
            goals, seq_len = build_motion_snippet_world(env, env_id, family_name, stage_cfg)
            env._unified_sequence_goals[env_id] = 0.0
            env._unified_sequence_goals[env_id, :seq_len] = goals[:seq_len]
            env._unified_sequence_len[env_id] = seq_len
            env._unified_sequence_idx[env_id] = 0
            set_goal_batch(env, torch.tensor([env_id], device=device), goals[0].unsqueeze(0), hold_steps=hold_steps)


def advance_unified_sequence_goals(env: CrazyfliePointNavEnv, env_ids: torch.Tensor, stage_cfg: dict[str, object]):
    if env_ids.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=env.device)
    device = env.device
    completed_ids: list[int] = []
    advance_ids: list[int] = []
    goals_to_set: list[torch.Tensor] = []
    segment_starts: list[torch.Tensor] = []
    hold_steps_to_set: list[int] = []
    for env_id in env_ids.tolist():
        next_idx = int(env._unified_sequence_idx[env_id].item()) + 1
        seq_len = int(env._unified_sequence_len[env_id].item())
        if next_idx >= seq_len:
            completed_ids.append(env_id)
        else:
            env._unified_sequence_idx[env_id] = next_idx
            advance_ids.append(env_id)
            goals_to_set.append(env._unified_sequence_goals[env_id, next_idx].clone())
            segment_starts.append(env._unified_sequence_goals[env_id, next_idx - 1].clone())
            family_name = UNIFIED_ID_TO_FAMILY[int(env._unified_family[env_id].item())]
            if family_name in UNIFIED_SNIPPET_FAMILIES:
                hold_steps_to_set.append(hold_time_to_steps(env, float(stage_cfg["snippet_hold_time"])))
            else:
                hold_steps_to_set.append(hold_time_to_steps(env, float(stage_cfg["sequence_hold_time"])))

    if advance_ids:
        set_goal_batch(
            env,
            torch.tensor(advance_ids, device=device),
            torch.stack(goals_to_set),
            torch.stack(segment_starts),
            hold_steps=torch.tensor(hold_steps_to_set, device=device, dtype=torch.int32),
        )
    return torch.tensor(completed_ids, device=device, dtype=torch.long)

def build_env() -> CrazyfliePointNavEnv:
    target_z = args.target_z if args.target_z is not None else args.goal_z_min
    world_target_z = target_z + args.z_reference_offset
    use_fixed_goal = args.target_x is not None and args.target_y is not None
    spawn_height_min = args.z_reference_offset + args.spawn_z_min
    spawn_height_max = args.z_reference_offset + args.spawn_z_max
    uses_trajectory_schedule = args.task_mode == "trajectory_generalized"
    is_sequence_nav = args.task_mode == "sequence_nav_precision"
    is_unified = args.task_mode == "unified_motion_curriculum"
    trajectory_schedule = load_timed_trajectory_from_csv(args.trajectory_csv) if uses_trajectory_schedule else None
    if trajectory_schedule:
        _, first_x, first_y, first_z = trajectory_schedule[0]
        target_z = first_z
        world_target_z = target_z + args.z_reference_offset
        use_fixed_goal = False

    cfg = CrazyfliePointNavEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.debug_vis = False
    cfg.seed = args.seed

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
    elif args.task_mode == "trajectory_generalized":
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
    elif args.task_mode == "sequence_nav_precision":
        # Sequential multi-goal precision navigation: keep episodes alive after
        # each captured goal and force the policy to chain farther moves and
        # sharper turns with stronger damping.
        cfg.goal_reach_threshold = max(args.goal_radius, 0.05)
        cfg.goal_min_distance = max(args.sequence_min_distance, 0.25)
        cfg.goal_max_distance = max(cfg.goal_min_distance + 0.05, args.sequence_max_distance)
        cfg.goal_height = args.goal_z_min + args.z_reference_offset
        cfg.goal_height_min = args.goal_z_min + args.z_reference_offset
        cfg.goal_height_max = args.goal_z_max + args.z_reference_offset
        cfg.init_guidance_probability = 0.0
        cfg.init_max_xy_offset = 0.12
        cfg.init_max_angle = 0.06
        cfg.init_max_linear_velocity = 0.06
        cfg.init_max_angular_velocity = 0.06
        cfg.init_height_offset_min = -0.06
        cfg.init_height_offset_max = 0.06

        cfg.goal_hold_steps = max(1, int(round(min(args.hold_time, 0.7) / cfg.sim.dt)))
        cfg.hover_gate_radius = max(0.34, cfg.goal_reach_threshold * 6.0)
        cfg.hover_gate_min = 0.08
        cfg.nav_progress_weight = 5.8
        cfg.nav_reach_bonus = 20.0
        cfg.nav_hold_step_weight = 1.0
        cfg.nav_hold_bonus = 32.0
        cfg.nav_braking_weight = 3.2
        cfg.nav_braking_radius = max(0.42, cfg.goal_reach_threshold * 7.0)
        cfg.hover_reward_scale = 0.30
        cfg.hover_reward_constant = 0.58
        cfg.hover_height_weight = 10.0
        cfg.hover_orientation_weight = 20.0
        cfg.hover_xy_velocity_weight = 2.4
        cfg.hover_z_velocity_weight = 8.5
        cfg.hover_angular_velocity_weight = 2.5
        cfg.hover_action_rate_weight = 0.24
        cfg.nav_height_track_weight = 3.0
        cfg.nav_height_recovery_weight = 1.1
        cfg.nav_height_descent_weight = 3.6
        cfg.nav_overshoot_penalty_weight = 1.1
        cfg.nav_corner_speed_penalty_weight = 0.24
        cfg.nav_corner_speed_threshold = 1.20
        cfg.nav_speed_penalty_weight = 0.18
        cfg.nav_speed_penalty_threshold = 1.55
        cfg.nav_goal_speed_penalty_radius = max(0.18, cfg.goal_reach_threshold * 3.5)
        cfg.nav_goal_closing_speed_penalty_weight = 1.0
        cfg.nav_goal_closing_speed_threshold = 0.55
        cfg.nav_goal_vertical_speed_penalty_weight = 0.85
        cfg.nav_goal_vertical_speed_threshold = 0.22
    else:
        # Unified motion curriculum: preserve the export-compatible 149-dim
        # observation interface while shifting supervision toward smooth chained
        # motion, cornering, and snippet completion.
        stage_cfg = get_unified_stage(0, max(args.max_iterations, 1))
        cfg.goal_reach_threshold = float(stage_cfg["goal_radius"])
        cfg.goal_min_distance = 0.10
        cfg.goal_max_distance = 0.90
        cfg.goal_height = args.goal_z_min + args.z_reference_offset
        cfg.goal_height_min = args.goal_z_min + args.z_reference_offset
        cfg.goal_height_max = args.goal_z_max + args.z_reference_offset
        cfg.init_guidance_probability = 0.12
        cfg.init_max_xy_offset = 0.10
        cfg.init_max_angle = 0.05
        cfg.init_max_linear_velocity = 0.035
        cfg.init_max_angular_velocity = 0.035
        cfg.init_height_offset_min = -0.04
        cfg.init_height_offset_max = 0.04

        cfg.goal_hold_steps = max(1, int(round(float(stage_cfg["sequence_hold_time"]) / cfg.sim.dt)))
        cfg.hover_gate_radius = max(0.32, cfg.goal_reach_threshold * 6.0)
        cfg.hover_gate_min = 0.08
        cfg.nav_progress_weight = 5.8
        cfg.nav_reach_bonus = 6.0
        cfg.nav_hold_step_weight = 0.45
        cfg.nav_hold_bonus = 12.0
        cfg.nav_segment_completion_bonus = 6.0
        cfg.nav_braking_weight = 3.4
        cfg.nav_braking_radius = max(0.42, cfg.goal_reach_threshold * 7.0)
        cfg.hover_reward_scale = 0.26
        cfg.hover_reward_constant = 0.55
        cfg.hover_height_weight = 8.0
        cfg.hover_orientation_weight = 18.0
        cfg.hover_xy_velocity_weight = 1.8
        cfg.hover_z_velocity_weight = 8.0
        cfg.hover_angular_velocity_weight = 2.2
        cfg.hover_action_rate_weight = 0.22
        cfg.nav_height_track_weight = 2.4
        cfg.nav_height_recovery_weight = 1.1
        cfg.nav_height_descent_weight = 3.2
        cfg.nav_path_progress_weight = 1.8
        cfg.nav_path_deviation_weight = 1.3
        cfg.nav_overshoot_penalty_weight = 1.5
        cfg.nav_oscillation_penalty_weight = 0.45
        cfg.nav_corner_speed_penalty_weight = 0.34
        cfg.nav_corner_speed_threshold = 1.15
        cfg.nav_speed_penalty_weight = 0.12
        cfg.nav_speed_penalty_threshold = 1.8
        cfg.nav_goal_speed_penalty_radius = max(0.18, cfg.goal_reach_threshold * 3.5)
        cfg.nav_goal_closing_speed_penalty_weight = 1.4
        cfg.nav_goal_closing_speed_threshold = 0.50
        cfg.nav_goal_vertical_speed_penalty_weight = 1.0
        cfg.nav_goal_vertical_speed_threshold = 0.20
        cfg.term_tilt_persistence_steps = max(cfg.term_tilt_persistence_steps, 60)
        cfg.term_linear_velocity_persistence_steps = max(cfg.term_linear_velocity_persistence_steps, 60)

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
    if args.task_mode == "sequence_nav_precision":
        env._continue_on_goal_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        env._sequence_prev_dir = torch.zeros(env.num_envs, 2, dtype=torch.float32, device=env.device)
    if is_unified:
        initialize_unified_buffers(env)
    return env


def train(env: CrazyfliePointNavEnv, agent: L2FPPOAgent):
    checkpoint_dir = resolve_checkpoint_dir()
    os.makedirs(checkpoint_dir, exist_ok=True)
    spawn_height_min = getattr(env, "_spawn_height_min", env.cfg.init_target_height_min)
    spawn_height_max = getattr(env, "_spawn_height_max", env.cfg.init_target_height_max)
    is_sequence_nav = args.task_mode == "sequence_nav_precision"
    is_unified = args.task_mode == "unified_motion_curriculum"

    warm_ckpt = resolve_warm_start_checkpoint()
    if os.path.exists(warm_ckpt):
        loaded_iter, loaded_best = agent.load(warm_ckpt, load_optimizer=args.preserve_optimizer_state)
        agent.set_learning_rate(args.lr)
        print(f"[Warm Start] Loaded: {warm_ckpt}")
        print(f"[Warm Start] Source iteration: {loaded_iter}, source best reward: {loaded_best:.3f}")
        if args.preserve_optimizer_state:
            print(f"[Warm Start] Preserved optimizer state; overriding LR to {agent.get_learning_rate():.2e}")
        else:
            print(f"[Warm Start] Reset optimizer state for finetuning; using LR {agent.get_learning_rate():.2e}")
        if is_unified and hasattr(agent, "loaded_log_std"):
            loaded_log_std = agent.loaded_log_std.detach().clone()
            agent.set_log_std_bounds(
                min_log_std=loaded_log_std - 0.50,
                max_log_std=loaded_log_std + 0.08,
            )
            print(
                f"[Warm Start] Log-std clamp enabled for unified curriculum "
                f"(mean std <= {torch.exp(agent.max_log_std).mean().item():.3f})"
            )
    else:
        print(f"[Warm Start] Checkpoint not found: {warm_ckpt}")
        print("[Warm Start] Training from scratch instead.")

    target_z = args.target_z if args.target_z is not None else args.goal_z_min
    world_target_z = target_z + args.z_reference_offset
    use_fixed_goal = args.target_x is not None and args.target_y is not None
    uses_trajectory_schedule = args.task_mode == "trajectory_generalized"
    initial_stage_cfg = get_unified_stage(0, args.max_iterations) if is_unified else None
    print(f"\n{'='*60}")
    print("HOLD FINE-TUNING")
    print(f"{'='*60}")
    print(f"  Task mode:          {args.task_mode}")
    if args.task_mode == "trajectory_generalized":
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
    elif is_sequence_nav:
        print(f"  Goal mode:          sequential random chain")
        print(f"  Sequence distance:  [{args.sequence_min_distance:.2f}, {args.sequence_max_distance:.2f}] m")
        print(f"  Goal z (virtual):   [{args.goal_z_min:.2f}, {args.goal_z_max:.2f}] m")
        print(f"  Goal z (world):     [{args.goal_z_min + args.z_reference_offset:.2f}, {args.goal_z_max + args.z_reference_offset:.2f}] m")
        print(f"  Spawn z (world):    [{spawn_height_min:.2f}, {spawn_height_max:.2f}] m")
    elif is_unified:
        print(f"  Goal mode:          unified curriculum")
        print(f"  Stage 1 mix:        70% single / 20% sequence / 10% snippet")
        print(f"  Stage 2 mix:        50% single / 35% sequence / 15% snippet")
        print(f"  Stage 3 mix:        30% single / 45% sequence / 25% snippet")
        print(f"  Distance buckets:   close 0.10-0.25 | medium 0.25-0.50 | far 0.50-0.90 m")
        print(f"  Goal z (virtual):   [{args.goal_z_min:.2f}, {args.goal_z_max:.2f}] m")
        print(f"  Goal z (world):     [{args.goal_z_min + args.z_reference_offset:.2f}, {args.goal_z_max + args.z_reference_offset:.2f}] m")
        print(f"  Spawn z (world):    [{spawn_height_min:.2f}, {spawn_height_max:.2f}] m")
        print(f"  Initial stage:      {initial_stage_cfg['name']}")
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
    if is_unified:
        print(
            "  Hold duration:      "
            f"single {float(initial_stage_cfg['single_hold_time']):.2f}s / "
            f"sequence {float(initial_stage_cfg['sequence_hold_time']):.2f}s / "
            f"snippet {float(initial_stage_cfg['snippet_hold_time']):.2f}s"
        )
    else:
        print(f"  Hold duration:      {args.hold_time:.1f} s")
    print(f"  Episode length:     {args.episode_length_s:.1f} s")
    print(f"  Environments:       {env.num_envs}")
    print(f"  Max iterations:     {args.max_iterations}")
    print(f"  Steps per rollout:  {args.steps_per_rollout}")
    print(f"  Mini-batch size:    {args.mini_batch_size}")
    print(f"  PPO epochs:         {args.ppo_epochs}")
    print(f"  Resume best:        {args.resume_best}")
    print(f"  Checkpoints:        {checkpoint_dir}")
    print(f"{'='*60}\n")

    debug_mark("startup_before_reset", task_mode=args.task_mode, num_envs=env.num_envs)
    print("[Startup] Resetting environments...", flush=True)
    obs_dict, _ = env.reset()
    debug_mark("startup_after_reset")
    if uses_trajectory_schedule:
        debug_mark("startup_apply_trajectory_schedule")
        apply_timed_trajectory_goals(env, env._trajectory_schedule_t, env._trajectory_schedule_xyz, env._trajectory_env_mask)
        obs_dict = env._get_observations()
    elif is_unified:
        all_env_ids = torch.arange(env.num_envs, device=env.device)
        debug_mark("startup_sample_unified_tasks_begin", env_count=int(all_env_ids.numel()), curriculum_stage=initial_stage_cfg["name"])
        sample_unified_tasks(env, all_env_ids, initial_stage_cfg)
        debug_mark("startup_sample_unified_tasks_done")
        set_hold_time(env, float(initial_stage_cfg["sequence_hold_time"]))
        env.cfg.goal_reach_threshold = float(initial_stage_cfg["goal_radius"])
        obs_dict = env._get_observations()
        debug_mark("startup_refresh_obs_done")
    obs = obs_dict["policy"]
    print("[Startup] Reset complete. Beginning rollout collection.", flush=True)

    best_reward = float("-inf")
    best_hold_event_rate = 0.0
    best_proxy_hold_s = float("-inf")
    best_precision_score = float("-inf")
    best_sequence_score = float("-inf")
    best_altitude_score = float("-inf")
    best_motion_score = float("-inf")
    curriculum_stages = build_hold_curriculum(args.hold_time) if args.curriculum else [args.hold_time]
    curriculum_stage_idx = 0
    if is_unified:
        current_hold_time = float(initial_stage_cfg["sequence_hold_time"])
        current_hold_steps = hold_time_to_steps(env, current_hold_time)
        current_hold_tag = format_hold_tag(current_hold_time)
        current_hold_desc = (
            f"S{float(initial_stage_cfg['single_hold_time']):.2f}/"
            f"Q{float(initial_stage_cfg['sequence_hold_time']):.2f}/"
            f"N{float(initial_stage_cfg['snippet_hold_time']):.2f}s"
        )
    else:
        current_hold_time = curriculum_stages[curriculum_stage_idx]
        current_hold_steps = set_hold_time(env, current_hold_time)
        current_hold_tag = format_hold_tag(current_hold_time)
        current_hold_desc = f"{current_hold_time:4.1f}s ({current_hold_steps:4d} steps)"
    recent_hold_rates = deque(maxlen=max(1, args.curriculum_window))

    def save_stage_checkpoint(name: str, iteration_idx: int):
        path = os.path.join(checkpoint_dir, f"{name}_{current_hold_tag}.pt")
        agent.save(path, iteration_idx, best_reward)
        return path

    # Save the warm-started policy under the initial curriculum target so we can
    # recover the exact starting point for this stage if later stages regress.
    save_stage_checkpoint("stage_start", -1)

    for iteration in range(args.max_iterations):
        stage_cfg = get_unified_stage(iteration, args.max_iterations) if is_unified else None
        if is_unified:
            current_hold_time = float(stage_cfg["sequence_hold_time"])
            current_hold_steps = set_hold_time(env, current_hold_time)
            env.cfg.goal_reach_threshold = float(stage_cfg["goal_radius"])
            current_hold_tag = format_hold_tag(current_hold_time)
            current_hold_desc = (
                f"S{float(stage_cfg['single_hold_time']):.2f}/"
                f"Q{float(stage_cfg['sequence_hold_time']):.2f}/"
                f"N{float(stage_cfg['snippet_hold_time']):.2f}s"
            )
        if iteration == 0:
            print("[Startup] Collecting first rollout...", flush=True)
            debug_mark("iter0_rollout_begin", steps=args.steps_per_rollout)
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
        family_touch_counts = {name: 0.0 for name in UNIFIED_ALL_FAMILIES} if is_unified else None
        family_hold_counts = {name: 0.0 for name in UNIFIED_ALL_FAMILIES} if is_unified else None
        family_z_error_sum = {name: 0.0 for name in UNIFIED_ALL_FAMILIES} if is_unified else None
        family_z_error_count = {name: 0.0 for name in UNIFIED_ALL_FAMILIES} if is_unified else None
        family_path_dev_sum = {name: 0.0 for name in UNIFIED_ALL_FAMILIES} if is_unified else None
        family_overshoot_sum = {name: 0.0 for name in UNIFIED_ALL_FAMILIES} if is_unified else None
        family_corner_speed_sum = {name: 0.0 for name in UNIFIED_ALL_FAMILIES} if is_unified else None

        for step_idx in range(args.steps_per_rollout):
            if uses_trajectory_schedule:
                apply_timed_trajectory_goals(env, env._trajectory_schedule_t, env._trajectory_schedule_xyz, env._trajectory_env_mask)
                obs = env._get_observations()["policy"]
            prev_goal_reached = env._goal_reached.clone()
            prev_goal_held = env._goal_held.clone()

            if iteration == 0 and step_idx < 3:
                debug_mark("iter0_pre_action", step=step_idx, obs_shape=tuple(obs.shape))
            action, log_prob, value = agent.get_action_and_value(obs)
            if iteration == 0 and step_idx < 3:
                debug_mark("iter0_post_action", step=step_idx, action_shape=tuple(action.shape))

            obs_buffer.append(obs.detach().cpu())
            action_buffer.append(action.detach().cpu())
            log_prob_buffer.append(log_prob.detach().cpu())
            value_buffer.append(value.detach().cpu())

            if iteration == 0 and step_idx < 3:
                debug_mark("iter0_pre_env_step", step=step_idx)
            obs_dict, reward, terminated, truncated, _ = env.step(action)
            if iteration == 0 and step_idx < 3:
                debug_mark(
                    "iter0_post_env_step",
                    step=step_idx,
                    reward_mean=f"{reward.mean().item():.4f}",
                    terminated=int(terminated.sum().item()),
                    truncated=int(truncated.sum().item()),
                )
            obs = obs_dict["policy"]
            done = terminated | truncated
            touch_transition = env._goal_reached & (~prev_goal_reached)
            hold_transition = env._goal_held & (~prev_goal_held)

            if is_sequence_nav:
                sequence_held = hold_transition.clone()
                if sequence_held.any():
                    seq_env_ids = sequence_held.nonzero(as_tuple=False).squeeze(-1)
                    sample_sequence_goals(env, seq_env_ids)
                    obs = env._get_observations()["policy"]
                    terminated[seq_env_ids] = False
                    done = terminated | truncated
            elif is_unified:
                if iteration == 0 and step_idx < 3:
                    debug_mark(
                        "iter0_unified_pre_transition",
                        step=step_idx,
                        hold_transitions=int(hold_transition.sum().item()),
                        touch_transitions=int(touch_transition.sum().item()),
                        continue_mask=int(env._continue_on_goal_mask.sum().item()),
                    )
                pos_w = env._robot.data.root_pos_w
                z_err = torch.abs(pos_w[:, 2] - env._goal_pos[:, 2])
                for family_id, family_name in UNIFIED_ID_TO_FAMILY.items():
                    family_mask = env._unified_family == family_id
                    if not torch.any(family_mask):
                        continue
                    family_touch_counts[family_name] += touch_transition[family_mask].sum().item()
                    family_hold_counts[family_name] += hold_transition[family_mask].sum().item()
                    family_z_error_sum[family_name] += z_err[family_mask].sum().item()
                    family_z_error_count[family_name] += float(family_mask.sum().item())
                    family_path_dev_sum[family_name] += env._latest_path_deviation[family_mask].sum().item()
                    family_overshoot_sum[family_name] += env._latest_overshoot[family_mask].sum().item()
                    family_corner_speed_sum[family_name] += env._latest_corner_speed[family_mask].sum().item()

                continue_ids = hold_transition & env._continue_on_goal_mask
                if continue_ids.any():
                    continuing_env_ids = continue_ids.nonzero(as_tuple=False).squeeze(-1)
                    single_mask = torch.zeros(continuing_env_ids.numel(), dtype=torch.bool, device=env.device)
                    sequence_mask = torch.zeros(continuing_env_ids.numel(), dtype=torch.bool, device=env.device)
                    for idx, env_id in enumerate(continuing_env_ids.tolist()):
                        family_name = UNIFIED_ID_TO_FAMILY[int(env._unified_family[env_id].item())]
                        if family_name in UNIFIED_SINGLE_FAMILIES:
                            single_mask[idx] = True
                        else:
                            sequence_mask[idx] = True

                    single_ids = continuing_env_ids[single_mask]
                    if single_ids.numel() > 0:
                        sample_unified_tasks(env, single_ids, stage_cfg)

                    seq_env_ids = continuing_env_ids[sequence_mask]
                    if seq_env_ids.numel() > 0:
                        completed_ids = advance_unified_sequence_goals(env, seq_env_ids, stage_cfg)
                        if completed_ids.numel() > 0:
                            sample_unified_tasks(env, completed_ids, stage_cfg)
                    obs = env._get_observations()["policy"]
                    if iteration == 0 and step_idx < 3:
                        debug_mark(
                            "iter0_unified_continue_processed",
                            step=step_idx,
                            continuing=int(continuing_env_ids.numel()),
                            single=int(single_ids.numel()),
                            sequence=int(seq_env_ids.numel()),
                        )

                done_ids = done.nonzero(as_tuple=False).squeeze(-1)
                if done_ids.numel() > 0:
                    sample_unified_tasks(env, done_ids, stage_cfg)
                    obs = env._get_observations()["policy"]
                if iteration == 0 and step_idx < 3:
                    debug_mark(
                        "iter0_unified_done_processed",
                        step=step_idx,
                        done_ids=int(done_ids.numel()),
                    )

            touch_events += touch_transition.sum().item()
            hold_events += hold_transition.sum().item()
            rollout_max_hold_steps = max(rollout_max_hold_steps, float(env._goal_hold_counter.max().item()))
            topk = min(max(1, args.proxy_hold_topk), env.num_envs)
            topk_mean = torch.topk(env._goal_hold_counter.float(), k=topk).values.mean().item()
            rollout_topk_hold_steps = max(rollout_topk_hold_steps, float(topk_mean))

            reward_buffer.append(reward.detach().cpu())
            done_buffer.append(done.detach().cpu())
            episode_rewards += reward
            if iteration == 0 and step_idx < 3:
                debug_mark("iter0_rollout_step_complete", step=step_idx)

        if iteration == 0:
            debug_mark("iter0_rollout_collection_complete", collected_steps=len(obs_buffer))
        obs_t = torch.stack(obs_buffer)
        actions_t = torch.stack(action_buffer)
        log_probs_t = torch.stack(log_prob_buffer)
        values_t = torch.stack(value_buffer)
        rewards_t = torch.stack(reward_buffer)
        dones_t = torch.stack(done_buffer)
        if iteration == 0:
            debug_mark(
                "iter0_post_stack",
                obs_shape=tuple(obs_t.shape),
                rewards_shape=tuple(rewards_t.shape),
                dones_shape=tuple(dones_t.shape),
            )

        with torch.no_grad():
            if iteration == 0:
                debug_mark("iter0_pre_next_value")
            next_value = agent.get_value(obs).detach().cpu()
            if iteration == 0:
                debug_mark("iter0_post_next_value", next_value_shape=tuple(next_value.shape))

        if iteration == 0:
            debug_mark("iter0_pre_gae")
        returns_t, advantages_t = compute_gae(
            rewards_t, values_t, dones_t, next_value,
            gamma=agent.gamma, gae_lambda=agent.gae_lambda
        )
        if iteration == 0:
            debug_mark(
                "iter0_post_gae",
                returns_shape=tuple(returns_t.shape),
                adv_shape=tuple(advantages_t.shape),
            )

        if iteration == 0:
            debug_mark("iter0_pre_update")
        update_stats = agent.update(
            obs_t.reshape(-1, obs_t.shape[-1]),
            actions_t.reshape(-1, actions_t.shape[-1]),
            log_probs_t.reshape(-1),
            returns_t.reshape(-1),
            advantages_t.reshape(-1),
        )
        if iteration == 0:
            debug_mark(
                "iter0_post_update",
                loss=f"{update_stats['loss']:.4f}",
                approx_kl=f"{update_stats.get('approx_kl', float('nan')):.6f}",
            )

        mean_reward = episode_rewards.mean().item() / args.steps_per_rollout
        hold_event_rate = hold_events / max(env.num_envs, 1)
        touch_event_rate = touch_events / max(env.num_envs, 1)
        proxy_hold_s = rollout_topk_hold_steps * env.cfg.sim.dt
        proxy_hold_max_s = rollout_max_hold_steps * env.cfg.sim.dt
        std = torch.exp(agent.actor.log_std).mean().item()
        recent_hold_rates.append(hold_event_rate)
        precision_score = float("-inf")
        sequence_score = float("-inf")
        altitude_score = float("-inf")
        motion_score = float("-inf")
        if is_unified:
            single_holds = sum(family_hold_counts[name] for name in UNIFIED_SINGLE_FAMILIES)
            sequence_holds = sum(family_hold_counts[name] for name in UNIFIED_SEQUENCE_FAMILIES + UNIFIED_SNIPPET_FAMILIES)
            precision_score = single_holds / max(env.num_envs, 1)
            sequence_score = touch_event_rate
            altitude_mean = sum(family_z_error_sum.values()) / max(sum(family_z_error_count.values()), 1.0)
            altitude_score = -altitude_mean
            path_dev_mean = sum(family_path_dev_sum.values()) / max(sum(family_z_error_count.values()), 1.0)
            overshoot_mean = sum(family_overshoot_sum.values()) / max(sum(family_z_error_count.values()), 1.0)
            corner_speed_mean = sum(family_corner_speed_sum.values()) / max(sum(family_z_error_count.values()), 1.0)
            motion_score = touch_event_rate - 0.35 * path_dev_mean - 0.55 * overshoot_mean - 0.05 * corner_speed_mean

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

        if is_unified and precision_score > best_precision_score:
            best_precision_score = precision_score
            agent.save(os.path.join(checkpoint_dir, "best_precision_model.pt"), iteration, best_reward)
        if is_unified and sequence_score > best_sequence_score:
            best_sequence_score = sequence_score
            agent.save(os.path.join(checkpoint_dir, "best_sequence_model.pt"), iteration, best_reward)
        if is_unified and altitude_score > best_altitude_score:
            best_altitude_score = altitude_score
            agent.save(os.path.join(checkpoint_dir, "best_altitude_control_model.pt"), iteration, best_reward)
        if is_unified and motion_score > best_motion_score:
            best_motion_score = motion_score
            agent.save(os.path.join(checkpoint_dir, "best_motion_model.pt"), iteration, best_reward)

        if iteration % 10 == 0 or is_best_reward:
            star = " *BEST*" if is_best_reward else ""
            print(
                f"[Iter {iteration:4d}] Reward: {mean_reward:8.3f} | "
                f"HoldEvt/env: {hold_event_rate:5.3f} | TouchEvt/env: {touch_event_rate:5.3f} | "
                f"ProxyHold(top{min(max(1, args.proxy_hold_topk), env.num_envs)}): {proxy_hold_s:4.2f}s | "
                f"ProxyMax: {proxy_hold_max_s:4.2f}s | "
                f"HoldTarget: {current_hold_desc} | "
                f"Std: {std:.3f} | Loss: {update_stats['loss']:.4f} | KL: {update_stats['approx_kl']:.4f}{star}"
            )
            if is_unified:
                family_summary = []
                for family_name in ("single_translation", "single_altitude_change", "single_diagonal", "single_far_retarget",
                                    "sharp_turn_chain", "climb_turn_chain", "zigzag_chain", "figure8_snippet"):
                    z_mean = family_z_error_sum[family_name] / max(family_z_error_count[family_name], 1.0)
                    path_dev_mean = family_path_dev_sum[family_name] / max(family_z_error_count[family_name], 1.0)
                    overshoot_mean = family_overshoot_sum[family_name] / max(family_z_error_count[family_name], 1.0)
                    family_summary.append(
                        f"{family_name}:T{family_touch_counts[family_name]:.1f}/H{family_hold_counts[family_name]:.1f}/Z{z_mean:.3f}/P{path_dev_mean:.3f}/O{overshoot_mean:.3f}"
                    )
                print(f"  Families: {' | '.join(family_summary)}")

        if (not is_unified) and args.curriculum and curriculum_stage_idx < len(curriculum_stages) - 1:
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
                current_hold_desc = f"{current_hold_time:4.1f}s ({current_hold_steps:4d} steps)"
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
        f"Final hold target: {current_hold_desc}"
    )
    print(f"Checkpoints saved to: {checkpoint_dir}")


def main():
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    debug_mark("main_before_build_env")
    env = build_env()
    debug_mark("main_after_build_env", device=env.device, obs_dim=env.cfg.observation_space, action_dim=env.cfg.action_space)
    agent = L2FPPOAgent(
        obs_dim=env.cfg.observation_space,
        action_dim=env.cfg.action_space,
        device=env.device,
        lr=args.lr,
        gamma=args.gamma,
        epochs=args.ppo_epochs,
        entropy_coef=args.entropy_coef,
        mini_batch_size=args.mini_batch_size,
        target_kl=args.target_kl,
    )
    debug_mark("main_after_agent_init", lr=args.lr, gamma=args.gamma)

    try:
        debug_mark("main_before_train")
        train(env, agent)
        debug_mark("main_after_train")
    except BaseException as exc:
        debug_mark("main_exception", exc_type=type(exc).__name__, exc=str(exc))
        traceback.print_exc()
        raise
    finally:
        debug_mark("main_finally_begin")
        env.close()
        debug_mark("main_env_closed")
        simulation_app.close()
        debug_mark("main_app_closed")


if __name__ == "__main__":
    main()
