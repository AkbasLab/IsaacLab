#!/usr/bin/env python3
"""Export trained policy to INT8 ONNX for STM32 deployment."""

import argparse
import os
import sys
import numpy as np
import torch
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Export policy to INT8 ONNX")
parser.add_argument("--task", type=str, required=True, help="Name of the task")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
parser.add_argument("--output_dir", type=str, default="exported", help="Output directory")
parser.add_argument("--num_calibration_samples", type=int, default=1000, help="Calibration samples for PTQ")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_onnx
import isaaclab_tasks
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg):
    """Export policy with INT8 quantization."""
    
    # Get checkpoint path
    checkpoint_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(checkpoint_path)
    output_dir = os.path.join(log_dir, args_cli.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")
    print(f"[INFO] Output directory: {output_dir}")
    
    # Create environment for calibration data collection
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 42
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)
    
    # Load trained policy
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(checkpoint_path)
    
    # Get policy network
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic
    
    # Get normalizer (should be None based on config)
    normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
    if normalizer is not None:
        print("[WARNING] Normalizer detected. Quantization may be affected.")
    
    # Step 1: Export FP32 ONNX
    fp32_path = os.path.join(output_dir, "policy_fp32.onnx")
    print(f"[INFO] Exporting FP32 ONNX to: {fp32_path}")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=output_dir, filename="policy_fp32.onnx")
    
    # Step 2: Collect calibration data
    print(f"[INFO] Collecting {args_cli.num_calibration_samples} calibration samples...")
    calibration_data = []
    obs = env.get_observations()
    
    for _ in range(args_cli.num_calibration_samples):
        with torch.inference_mode():
            actions = runner.get_inference_policy(device=env.unwrapped.device)(obs)
        obs, _, _, _ = env.step(actions)
        calibration_data.append(obs.cpu().numpy())
    
    calibration_data = np.vstack(calibration_data)
    calibration_path = os.path.join(output_dir, "calibration_data.npy")
    np.save(calibration_path, calibration_data)
    print(f"[INFO] Calibration data saved to: {calibration_path}")
    print(f"[INFO] Calibration data shape: {calibration_data.shape}")
    print(f"[INFO] Calibration data range: [{calibration_data.min():.3f}, {calibration_data.max():.3f}]")
    
    # Step 3: Quantize to INT8
    int8_path = os.path.join(output_dir, "policy_int8.onnx")
    print(f"[INFO] Quantizing to INT8: {int8_path}")
    
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,
        optimize_model=True,
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
        }
    )
    
    # Step 4: Validate quantization accuracy
    print("[INFO] Validating quantization accuracy...")
    session_fp32 = ort.InferenceSession(fp32_path)
    session_int8 = ort.InferenceSession(int8_path)
    
    # Test on calibration data
    test_samples = calibration_data[:100]
    actions_fp32 = session_fp32.run(None, {"obs": test_samples.astype(np.float32)})[0]
    actions_int8 = session_int8.run(None, {"obs": test_samples.astype(np.float32)})[0]
    
    max_error = np.abs(actions_fp32 - actions_int8).max()
    mean_error = np.abs(actions_fp32 - actions_int8).mean()
    correlation = np.corrcoef(actions_fp32.flatten(), actions_int8.flatten())[0, 1]
    
    print(f"[INFO] Quantization validation:")
    print(f"  Max error: {max_error:.6f}")
    print(f"  Mean error: {mean_error:.6f}")
    print(f"  Correlation: {correlation:.6f}")
    
    # Quality checks
    if max_error > 0.5:
        print("[WARNING] Max error exceeds 0.5. Quantization may degrade performance.")
    if correlation < 0.95:
        print("[WARNING] Correlation below 0.95. Consider QAT retraining.")
    
    # Step 5: Report file sizes
    fp32_size = os.path.getsize(fp32_path) / 1024
    int8_size = os.path.getsize(int8_path) / 1024
    print(f"[INFO] File sizes:")
    print(f"  FP32: {fp32_size:.1f} KB")
    print(f"  INT8: {int8_size:.1f} KB")
    print(f"  Compression: {100 * (1 - int8_size/fp32_size):.1f}%")
    
    print(f"[SUCCESS] Export complete. Files in: {output_dir}")
    
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
