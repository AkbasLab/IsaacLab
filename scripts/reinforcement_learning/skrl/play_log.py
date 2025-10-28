# play_log_io.py
# Copyright (c) 2022-2025, The Isaac Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play/evaluate a checkpoint of an RL agent from skrl, with optional logging of
observations and actions per step as newline-delimited JSON (JSONL).

Logging is robust to single-agent and multi-agent tasks and safely handles torch.Tensors and numpy arrays.
"""

import argparse
import sys
import os
import json
import time
import datetime
import traceback
from typing import Any, Dict, Union

from isaaclab.app import AppLauncher

# -------------------------------
# CLI
# -------------------------------
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl with optional IO logging.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default=None, help="Name of the RL agent configuration entry point.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint from Nucleus.")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"], help="ML framework.")
parser.add_argument("--algorithm", type=str, default="PPO", choices=["AMP", "PPO", "IPPO", "MAPPO"], help="RL algorithm.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# IO logging controls
parser.add_argument("--log-io", action="store_true", help="Enable logging of observations and actions to JSONL.")
parser.add_argument("--log-dir", type=str, default=None, help="Directory to write logs. Defaults to the run's log dir.")
parser.add_argument("--log-filename", type=str, default=None, help="Override log filename. Defaults to io_log_<timestamp>.jsonl")
parser.add_argument("--log-every", type=int, default=1, help="Log every Nth env step (>=1).")
parser.add_argument("--max-steps", type=int, default=None, help="Stop after this many env steps (for evaluation runs).")

# Append AppLauncher CLI args and parse
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Arg validation (fail fast on obvious mistakes)
if args_cli.log_every is not None and args_cli.log_every < 1:
    parser.error("--log-every must be >= 1")

if args_cli.video:
    args_cli.enable_cameras = True

# Clear out sys.argv for Hydra and launch app
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -------------------------------
# Imports that require the app
# -------------------------------
import gymnasium as gym  # noqa: E402
import random  # noqa: E402
import torch  # noqa: E402
import skrl  # noqa: E402
from packaging import version  # noqa: E402

from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

# -------------------------------
# Utilities
# -------------------------------
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    simulation_app.close()
    raise SystemExit(1)

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner  # noqa: E402
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner  # noqa: E402


def _to_serializable(x: Any) -> Any:
    """Convert tensors/arrays/typed values to JSON-serializable forms."""
    try:
        import numpy as np  # local import; numpy is a transitive dep via Isaac/torch

        if isinstance(x, torch.Tensor):
            if x.is_cuda:
                x = x.detach().cpu()
            else:
                x = x.detach()
            x = x.numpy()
        if isinstance(x, (np.generic,)):
            return x.item()
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
    except Exception:
        # numpy might not be available or conversion failed; fall through
        pass

    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="replace")
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    if isinstance(x, dict):
        return {str(k): _to_serializable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_serializable(v) for v in x]

    # Fallback: string representation (ensures logging never crashes)
    return str(x)


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Failed to create directory: {path}. Error: {e}") from e


def _now_iso() -> str:
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()


# -------------------------------
# Main
# -------------------------------
# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: Union[ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg], experiment_cfg: Dict[str, Any]):
    """Play with skrl agent and optionally log obs/actions."""

    # Task naming
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # CLI overrides
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Framework setup
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # Seeding
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # Checkpoint discovery
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print("[INFO] No pre-trained checkpoint available for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    if not resume_path or not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir

    # Optional override of IO log directory
    io_dir = args_cli.log_dir if args_cli.log_dir else os.path.join(log_dir, "io_logs")
    io_path = None
    io_fp = None

    # Environment creation
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Single-agent conversion if needed
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # Step dt for real-time
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # Video wrapper
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during evaluation.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # skrl wrapper + runner
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    runner.agent.set_running_mode("eval")

    # Prepare IO logging
    if args_cli.log_io:
        _ensure_dir(io_dir)
        fname = (
            args_cli.log_filename
            if args_cli.log_filename
            else f"io_log_{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.jsonl"
        )
        io_path = os.path.join(io_dir, fname)
        try:
            io_fp = open(io_path, "w", encoding="utf-8")
        except Exception as e:
            raise RuntimeError(f"Failed to open IO log file for writing: {io_path}. Error: {e}") from e
        meta = {
            "type": "meta",
            "created_utc": _now_iso(),
            "task": args_cli.task,
            "checkpoint": resume_path,
            "algorithm": args_cli.algorithm,
            "ml_framework": args_cli.ml_framework,
            "seed": experiment_cfg["seed"],
            "num_envs": env.unwrapped.num_envs if hasattr(env.unwrapped, "num_envs") else None,
            "log_every": args_cli.log_every,
        }
        io_fp.write(json.dumps(meta) + "\n")
        io_fp.flush()
        print(f"[INFO] IO logging enabled: {io_path}")

    # Reset and run
    obs, _ = env.reset()
    timestep = 0
    steps_written = 0
    max_steps = args_cli.max_steps if args_cli.max_steps is not None else float("inf")

    try:
        while simulation_app.is_running():
            start_time = time.time()

            with torch.inference_mode():
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)

                # Actions extraction (deterministic mean when available)
                if hasattr(env, "possible_agents"):
                    # Multi-agent dict
                    actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
                else:
                    # Single-agent
                    actions = outputs[-1].get("mean_actions", outputs[0])

                # Step env
                next_obs, _, _, _, _ = env.step(actions)

            # Logging control
            if args_cli.log_io and (timestep % args_cli.log_every == 0):
                try:
                    record = {
                        "type": "io_step",
                        "step": int(timestep),
                        "wall_time_utc": _now_iso(),
                        "obs": _to_serializable(obs),
                        "actions": _to_serializable(actions),
                    }
                    io_fp.write(json.dumps(record) + "\n")
                    steps_written += 1
                    # Flush occasionally to reduce data loss risk
                    if steps_written % 50 == 0:
                        io_fp.flush()
                except Exception:
                    # Do not crash evaluation because of logging issues; write an error record and continue
                    err_rec = {
                        "type": "io_error",
                        "step": int(timestep),
                        "wall_time_utc": _now_iso(),
                        "error": "exception during logging",
                        "traceback": traceback.format_exc(limit=1),
                    }
                    try:
                        if io_fp:
                            io_fp.write(json.dumps(err_rec) + "\n")
                            io_fp.flush()
                    except Exception:
                        pass  # give up on logging if even the error record fails

            obs = next_obs
            timestep += 1

            # Exit after one recorded clip if video-only evaluation
            if args_cli.video and timestep >= args_cli.video_length:
                break

            # User-defined max steps stop
            if timestep >= max_steps:
                break

            # Real-time pacing
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        try:
            env.close()
        except Exception:
            pass
        if io_fp:
            try:
                io_fp.flush()
                io_fp.close()
                print(f"[INFO] IO log closed: {io_path}")
            except Exception:
                pass


if __name__ == "__main__":
    main()
    simulation_app.close()
