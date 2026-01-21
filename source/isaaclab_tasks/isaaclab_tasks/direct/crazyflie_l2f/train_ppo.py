"""
PPO Training Script for Crazyflie Learning-to-Fly Pipeline

This script trains a PPO policy in Isaac Lab that is compatible with the
learning-to-fly sim-to-real pipeline. The trained policy can be directly
exported to Crazyflie firmware.

Usage:
    # From IsaacLab directory, train with default settings:
    ./isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/crazyflie_l2f/train_ppo.py --num_envs 4096
    
    # Train with visualization (fewer envs):
    ./isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/crazyflie_l2f/train_ppo.py --num_envs 16

    # Export trained policy:
    ./isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/crazyflie_l2f/train_ppo.py --checkpoint logs/crazyflie_l2f/model.pt --export_only

References:
- learning-to-fly/training/ppo_gpu/ppo.py
- IsaacLab/source/isaaclab_rl
"""

# ============================================================================
# IMPORTANT: Isaac Sim must be launched BEFORE importing isaaclab modules!
# ============================================================================

import argparse
import sys

# Step 1: Import AppLauncher FIRST (before any other isaaclab imports)
from isaaclab.app import AppLauncher

# Step 2: Parse arguments
parser = argparse.ArgumentParser(description="Train PPO for Crazyflie L2F")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments")
parser.add_argument("--max_iterations", type=int, default=500, help="Maximum training iterations")
parser.add_argument("--log_dir", type=str, default="logs/crazyflie_l2f", help="Log directory")
parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
parser.add_argument("--export_only", action="store_true", help="Only export checkpoint to firmware")
parser.add_argument("--export_path", type=str, default="actor.h", help="Firmware export path")
parser.add_argument("--skip_calibration", action="store_true", help="Skip calibration check (NOT recommended)")
# Add AppLauncher arguments (--headless, --enable_cameras, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Step 3: Launch Isaac Sim (this initializes Omniverse and makes omni.* available)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================================
# NOW we can import the rest of the modules that depend on omni/isaaclab
# ============================================================================

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Isaac Lab imports (safe to import now that simulation_app is initialized)
from isaaclab.envs import ManagerBasedRLEnvCfg

# Local imports  
from isaaclab_tasks.direct.crazyflie_l2f.crazyflie_l2f_env import CrazyflieL2FEnv, CrazyflieL2FEnvCfg
from isaaclab_tasks.direct.crazyflie_l2f.networks import L2FActorCritic, RunningMeanStd
from isaaclab_tasks.direct.crazyflie_l2f.export_to_firmware import export_policy_to_firmware


class PPOConfig:
    """PPO hyperparameters matching learning-to-fly defaults."""
    
    # Training
    num_envs: int = 4096
    max_iterations: int = 500
    steps_per_iteration: int = 24  # Steps per env per iteration
    
    # PPO specific
    clip_range: float = 0.2
    clip_range_vf: float = None  # If None, no clipping on value function
    entropy_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Learning rate
    learning_rate: float = 3e-4
    lr_schedule: str = "constant"  # "constant" or "linear"
    
    # GAE
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # Mini-batch
    num_minibatches: int = 4
    num_epochs: int = 4
    
    # Normalization
    normalize_observations: bool = True
    normalize_rewards: bool = False
    
    # Checkpointing
    save_interval: int = 50
    log_interval: int = 10
    
    # Network
    hidden_dim: int = 64
    init_std: float = 0.3


class RolloutBuffer:
    """Buffer for storing rollout data."""
    
    def __init__(
        self,
        num_envs: int,
        num_steps: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
    ):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        
        # Allocate buffers
        self.observations = torch.zeros(
            (num_steps, num_envs, obs_dim), device=device
        )
        self.actions = torch.zeros(
            (num_steps, num_envs, action_dim), device=device
        )
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        self.log_probs = torch.zeros((num_steps, num_envs), device=device)
        self.advantages = torch.zeros((num_steps, num_envs), device=device)
        self.returns = torch.zeros((num_steps, num_envs), device=device)
        
        self.step = 0
    
    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ):
        """Add a transition to the buffer."""
        self.observations[self.step] = obs
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value.squeeze(-1)
        self.log_probs[self.step] = log_prob.squeeze(-1)
        self.step += 1
    
    def compute_returns_and_advantages(
        self,
        last_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ):
        """Compute GAE advantages and returns."""
        last_value = last_value.squeeze(-1)
        last_gae_lam = 0
        
        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_values = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]
            
            delta = (
                self.rewards[step]
                + gamma * next_values * next_non_terminal
                - self.values[step]
            )
            self.advantages[step] = last_gae_lam = (
                delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            )
        
        self.returns = self.advantages + self.values
    
    def get_minibatch_generator(
        self, num_minibatches: int
    ):
        """Generate minibatches for PPO update."""
        batch_size = self.num_envs * self.num_steps
        minibatch_size = batch_size // num_minibatches
        
        # Flatten all data
        obs_flat = self.observations.view(-1, self.obs_dim)
        actions_flat = self.actions.view(-1, self.action_dim)
        log_probs_flat = self.log_probs.view(-1)
        advantages_flat = self.advantages.view(-1)
        returns_flat = self.returns.view(-1)
        
        # Random permutation
        indices = torch.randperm(batch_size, device=self.device)
        
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_indices = indices[start:end]
            
            yield (
                obs_flat[mb_indices],
                actions_flat[mb_indices],
                log_probs_flat[mb_indices],
                advantages_flat[mb_indices],
                returns_flat[mb_indices],
            )
    
    def reset(self):
        """Reset the buffer for new rollout."""
        self.step = 0


class PPOTrainer:
    """PPO trainer for Crazyflie L2F environment."""
    
    def __init__(
        self,
        env: CrazyflieL2FEnv,
        cfg: PPOConfig,
        log_dir: str = "logs/crazyflie_l2f",
    ):
        self.env = env
        self.cfg = cfg
        self.device = env.device
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Get dimensions
        self.obs_dim = env.cfg.observation.total_dim
        self.action_dim = env.cfg.action_space
        
        # Create actor-critic network
        self.actor_critic = L2FActorCritic(
            observation_dim=self.obs_dim,
            hidden_dim=cfg.hidden_dim,
            action_dim=self.action_dim,
            init_std=cfg.init_std,
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=cfg.learning_rate,
        )
        
        # Observation normalizer
        if cfg.normalize_observations:
            self.obs_normalizer = RunningMeanStd((self.obs_dim,)).to(self.device)
        else:
            self.obs_normalizer = None
        
        # Rollout buffer
        self.rollout_buffer = RolloutBuffer(
            num_envs=env.num_envs,
            num_steps=cfg.steps_per_iteration,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device,
        )
        
        # Training state
        self.iteration = 0
        self.total_timesteps = 0
        
        # Logging
        self.episode_rewards = []
        self.episode_lengths = []
    
    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize observations if enabled."""
        if self.obs_normalizer is not None:
            return self.obs_normalizer.normalize(obs)
        return obs
    
    def collect_rollouts(self) -> Tuple[float, float]:
        """Collect rollouts from the environment."""
        self.rollout_buffer.reset()
        
        obs_dict = self.env.reset()
        obs = obs_dict[0]["policy"]  # (num_envs, obs_dim)
        
        episode_rewards = []
        episode_lengths = []
        
        for step in range(self.cfg.steps_per_iteration):
            # Update observation normalizer
            if self.obs_normalizer is not None:
                self.obs_normalizer.update(obs)
            
            # Normalize observation
            obs_normalized = self.normalize_obs(obs)
            
            # Get action from policy
            with torch.no_grad():
                action, log_prob, value = self.actor_critic.get_action_and_value(
                    obs_normalized, deterministic=False
                )
            
            # Step environment
            next_obs_dict, rewards, terminated, truncated, infos = self.env.step(action)
            next_obs = next_obs_dict["policy"]
            dones = terminated | truncated
            
            # Store transition
            self.rollout_buffer.add(
                obs=obs,
                action=action,
                reward=rewards,
                done=dones.float(),
                value=value,
                log_prob=log_prob,
            )
            
            # Track episode stats from infos
            if "log" in infos:
                for env_id in range(self.env.num_envs):
                    if dones[env_id]:
                        if "Episode/total_reward" in infos["log"]:
                            episode_rewards.append(infos["log"]["Episode/total_reward"])
            
            obs = next_obs
            self.total_timesteps += self.env.num_envs
        
        # Compute returns and advantages
        with torch.no_grad():
            obs_normalized = self.normalize_obs(obs)
            _, _, last_value = self.actor_critic.get_action_and_value(
                obs_normalized, deterministic=False
            )
        
        self.rollout_buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
        )
        
        # Compute mean reward from rollout buffer (more reliable than episode completion)
        mean_reward = self.rollout_buffer.rewards.mean().item()
        mean_length = np.mean(episode_lengths) if episode_lengths else 0.0
        
        return mean_reward, mean_length
    
    def update(self) -> dict:
        """Perform PPO update."""
        # Normalize advantages
        advantages = self.rollout_buffer.advantages.view(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.rollout_buffer.advantages = advantages.view(
            self.cfg.steps_per_iteration, self.env.num_envs
        )
        
        # Track losses
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for epoch in range(self.cfg.num_epochs):
            for minibatch in self.rollout_buffer.get_minibatch_generator(
                self.cfg.num_minibatches
            ):
                obs, actions, old_log_probs, advantages_mb, returns = minibatch
                
                # Normalize observations
                obs_normalized = self.normalize_obs(obs)
                
                # Evaluate actions
                log_probs, values, entropy = self.actor_critic.evaluate_actions(
                    obs_normalized, actions
                )
                log_probs = log_probs.squeeze(-1)
                values = values.squeeze(-1)
                entropy = entropy.mean()
                
                # Ratio for PPO
                ratio = torch.exp(log_probs - old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * advantages_mb
                surr2 = torch.clamp(
                    ratio, 1 - self.cfg.clip_range, 1 + self.cfg.clip_range
                ) * advantages_mb
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                if self.cfg.clip_range_vf is not None:
                    # Clipped value loss
                    values_clipped = self.rollout_buffer.values.view(-1) + torch.clamp(
                        values - self.rollout_buffer.values.view(-1),
                        -self.cfg.clip_range_vf,
                        self.cfg.clip_range_vf,
                    )
                    value_loss1 = (values - returns) ** 2
                    value_loss2 = (values_clipped - returns) ** 2
                    value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                else:
                    value_loss = 0.5 * ((values - returns) ** 2).mean()
                
                # Entropy loss
                entropy_loss = -entropy
                
                # Total loss
                loss = (
                    policy_loss
                    + self.cfg.vf_coef * value_loss
                    + self.cfg.entropy_coef * entropy_loss
                )
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.cfg.max_grad_norm
                )
                self.optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(-entropy_loss.item())
        
        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropy_losses),
        }
    
    def train(self) -> None:
        """Main training loop."""
        print(f"[Training] Starting PPO training")
        print(f"[Training] Num envs: {self.env.num_envs}")
        print(f"[Training] Max iterations: {self.cfg.max_iterations}")
        print(f"[Training] Log directory: {self.log_dir}")
        
        start_time = time.time()
        
        for iteration in range(self.cfg.max_iterations):
            self.iteration = iteration
            
            # Collect rollouts
            mean_reward, mean_length = self.collect_rollouts()
            
            # Update policy
            update_stats = self.update()
            
            # Logging
            if iteration % self.cfg.log_interval == 0:
                elapsed = time.time() - start_time
                fps = self.total_timesteps / elapsed
                
                print(f"[Iter {iteration:4d}] "
                      f"Timesteps: {self.total_timesteps:8d} | "
                      f"FPS: {fps:6.0f} | "
                      f"Reward: {mean_reward:8.3f} | "
                      f"Policy Loss: {update_stats['policy_loss']:8.4f} | "
                      f"Value Loss: {update_stats['value_loss']:8.4f} | "
                      f"Entropy: {update_stats['entropy']:8.4f}")
            
            # Save checkpoint
            if iteration % self.cfg.save_interval == 0 and iteration > 0:
                self.save_checkpoint(f"checkpoint_{iteration}.pt")
        
        # Final save
        self.save_checkpoint("final_model.pt")
        print(f"[Training] Completed in {time.time() - start_time:.1f} seconds")
    
    def save_checkpoint(self, filename: str) -> str:
        """Save training checkpoint."""
        checkpoint_path = self.log_dir / filename
        
        checkpoint = {
            "iteration": self.iteration,
            "total_timesteps": self.total_timesteps,
            "actor_critic_state_dict": self.actor_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": vars(self.cfg),
        }
        
        if self.obs_normalizer is not None:
            checkpoint["obs_mean"] = self.obs_normalizer.mean
            checkpoint["obs_var"] = self.obs_normalizer.var
            checkpoint["obs_count"] = self.obs_normalizer.count
        
        torch.save(checkpoint, checkpoint_path)
        print(f"[Checkpoint] Saved to: {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.iteration = checkpoint["iteration"]
        self.total_timesteps = checkpoint["total_timesteps"]
        
        if self.obs_normalizer is not None and "obs_mean" in checkpoint:
            self.obs_normalizer.mean = checkpoint["obs_mean"]
            self.obs_normalizer.var = checkpoint["obs_var"]
            self.obs_normalizer.count = checkpoint["obs_count"]
        
        print(f"[Checkpoint] Loaded from: {checkpoint_path}")
        print(f"[Checkpoint] Resuming from iteration {self.iteration}")
    
    def export_to_firmware(self, output_path: str) -> str:
        """Export trained policy to firmware checkpoint format."""
        if self.obs_normalizer is not None:
            obs_mean = self.obs_normalizer.mean
            obs_std = torch.sqrt(self.obs_normalizer.var + 1e-8)
        else:
            obs_mean = torch.zeros(self.obs_dim, device=self.device)
            obs_std = torch.ones(self.obs_dim, device=self.device)
        
        return export_policy_to_firmware(
            policy=self.actor_critic.actor,
            obs_mean=obs_mean,
            obs_std=obs_std,
            output_path=output_path,
            model_name=f"isaac_lab_ppo_iter{self.iteration}",
        )


def run_calibration_check(env: CrazyflieL2FEnv, skip: bool = False) -> bool:
    """
    Run calibration check before training.
    
    Args:
        env: Environment instance
        skip: Skip calibration (for debugging only)
        
    Returns:
        True if calibration passed or skipped, False otherwise
    """
    from isaaclab_tasks.direct.crazyflie_l2f.calibrate import CalibrationSuite
    
    if skip:
        print("\n[WARNING] Calibration check skipped! Results may not match L2F.")
        return True
    
    print("\n" + "="*60)
    print("Running pre-training calibration check...")
    print("="*60)
    
    suite = CalibrationSuite(env, device=env.device)
    report = suite.run_all()
    report.print_report()
    
    if not report.all_passed:
        print("\n[ERROR] Calibration FAILED. Training aborted.")
        print("Fix physics parameters or use --skip-calibration to override.")
        return False
    
    print("[OK] Calibration passed. Proceeding to training...")
    return True


def main():
    """Main entry point for training."""
    # Args already parsed at module level for AppLauncher
    global args_cli
    
    # Create environment config
    env_cfg = CrazyflieL2FEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # Disable calibration requirement for training script (we handle it explicitly)
    env_cfg.calibration.require_calibration = False
    
    # Create environment
    env = CrazyflieL2FEnv(env_cfg)
    
    # Run calibration check before training
    skip_calibration = getattr(args_cli, 'skip_calibration', False)
    if not args_cli.export_only and not run_calibration_check(env, skip=skip_calibration):
        env.close()
        simulation_app.close()
        return
    
    # Create trainer
    cfg = PPOConfig()
    cfg.num_envs = args_cli.num_envs
    cfg.max_iterations = args_cli.max_iterations
    
    trainer = PPOTrainer(env, cfg, log_dir=args_cli.log_dir)
    
    # Load checkpoint if provided
    if args_cli.checkpoint:
        trainer.load_checkpoint(args_cli.checkpoint)
    
    # Export only or train
    if args_cli.export_only:
        if not args_cli.checkpoint:
            print("Error: --export_only requires --checkpoint")
            return
        trainer.export_to_firmware(args_cli.export_path)
    else:
        trainer.train()
        # Export final model
        trainer.export_to_firmware(
            str(Path(args_cli.log_dir) / "actor.h")
        )
    
    env.close()
    
    # Close simulation app
    simulation_app.close()


if __name__ == "__main__":
    main()
