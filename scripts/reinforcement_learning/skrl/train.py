# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--export_onnx",
    action="store_true",
    default=False,
    help="Export the trained policy to ONNX at the end of training (PyTorch only).",
)
parser.add_argument(
    "--onnx_opset",
    type=int,
    default=17,
    help="ONNX opset version to use when exporting (PyTorch only).",
)
parser.add_argument(
    "--onnx_filename",
    type=str,
    default="policy.onnx",
    help="Filename for the exported ONNX model (saved under the run's exported/ folder).",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
from datetime import datetime

import omni
import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    runner = Runner(env, agent_cfg)

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    # run training
    runner.run()

    # optional: export trained policy to ONNX (PyTorch only)
    if args_cli.export_onnx and args_cli.ml_framework.startswith("torch"):
        try:
            import torch

            # Build a dummy observation tensor from a real reset, which matches skrl expectations
            obs_sample = None
            try:
                obs_reset, _ = env.reset()
                obs_sample = obs_reset
            except Exception:
                obs_sample = None

            def _to_batched_tensor(x, device):
                if isinstance(x, torch.Tensor):
                    t = x
                else:
                    import numpy as np

                    if isinstance(x, (list, tuple)):
                        x = x[0]
                    t = torch.as_tensor(x, dtype=torch.float32)
                if t.ndim == 1:
                    t = t.unsqueeze(0)
                elif t.ndim > 1:
                    t = t[:1]
                return t.to(device)

            agent_device = getattr(runner.agent, "device", "cpu")

            if obs_sample is None:
                # last resort: try to infer from env attributes
                obs_space = getattr(env, "single_observation_space", None) or getattr(env, "observation_space", None)
                if (obs_space is not None) and hasattr(obs_space, "shape") and (getattr(obs_space, "shape", None) is not None):
                    obs_shape = (1,) + tuple(obs_space.shape)
                    dummy_obs = torch.zeros(obs_shape, dtype=torch.float32, device=agent_device)
                else:
                    dummy_obs = None
            else:
                # handle dict-like observations by taking the first tensor-like value
                if isinstance(obs_sample, dict):
                    # prefer common keys
                    preferred_keys = ["policy", "states", "obs", "observations"]
                    sel = None
                    for k in preferred_keys:
                        if k in obs_sample:
                            sel = obs_sample[k]
                            break
                    if sel is None and len(obs_sample):
                        sel = next(iter(obs_sample.values()))
                    dummy_obs = _to_batched_tensor(sel, agent_device) if sel is not None else None
                else:
                    dummy_obs = _to_batched_tensor(obs_sample, agent_device)

            export_dir = os.path.join(log_dir, "exported")
            os.makedirs(export_dir, exist_ok=True)
            onnx_path = os.path.join(export_dir, args_cli.onnx_filename)

            # Prefer exporting the actual policy model to keep parameters registered
            policy_model = None
            if hasattr(runner.agent, "models") and isinstance(getattr(runner.agent, "models"), dict):
                models = getattr(runner.agent, "models")
                policy_model = models.get("policy") or models.get("actor") or None
            # some agents expose .policy / .actor directly
            if policy_model is None:
                policy_model = getattr(runner.agent, "policy", None) or getattr(runner.agent, "actor", None)

            if dummy_obs is not None and isinstance(policy_model, torch.nn.Module):
                class _PolicyComputeWrapper(torch.nn.Module):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model

                    def forward(self, obs):
                        # SKRL models expect a dict with key 'states' for policy role
                        out = self.model.compute({"states": obs}, role="policy")
                        # out: (mean, log_std, info)
                        if isinstance(out, (list, tuple)) and len(out) >= 1:
                            return out[0]
                        return out

                model_cpu = policy_model.eval().cpu()
                wrapper = _PolicyComputeWrapper(model_cpu)
                dummy_cpu = dummy_obs.detach().cpu()
                dynamic_axes = {"obs": {0: "batch"}, "actions": {0: "batch"}}
                torch.onnx.export(
                    wrapper,
                    (dummy_cpu,),
                    onnx_path,
                    export_params=True,
                    opset_version=int(args_cli.onnx_opset),
                    do_constant_folding=True,
                    input_names=["obs"],
                    output_names=["actions"],
                    dynamic_axes=dynamic_axes,
                )
                print(f"[INFO] Exported policy (compute->mean) to ONNX: {onnx_path}")
            elif dummy_obs is not None:
                # Fallback: export via agent.act wrapper (may fail if parameters aren't registered)
                class _SkrlActWrapper(torch.nn.Module):
                    def __init__(self, agent):
                        super().__init__()
                        self._agent = agent

                    def forward(self, obs):  # obs: [B, obs_dim]
                        with torch.no_grad():
                            outputs = self._agent.act(obs, timestep=0, timesteps=0)
                            if isinstance(outputs, (list, tuple)):
                                info = outputs[-1] if len(outputs) >= 3 and isinstance(outputs[-1], dict) else {}
                                actions = info.get("mean_actions", outputs[0])
                            else:
                                actions = outputs
                        return actions

                wrapper = _SkrlActWrapper(runner.agent)
                dynamic_axes = {"obs": {0: "batch"}, "actions": {0: "batch"}}
                torch.onnx.export(
                    wrapper,
                    (dummy_obs,),
                    onnx_path,
                    export_params=True,
                    opset_version=int(args_cli.onnx_opset),
                    do_constant_folding=True,
                    input_names=["obs"],
                    output_names=["actions"],
                    dynamic_axes=dynamic_axes,
                )
                print(f"[INFO] Exported policy via act() wrapper to ONNX: {onnx_path}")
            else:
                print("[WARN] Could not obtain a sample observation for ONNX export; skipping.")
        except Exception as e:
            print(f"[WARN] ONNX export failed: {e}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
