# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""
Agent configurations for Crazyflie L2F environment.

These configurations are designed to produce policies compatible with
the learning-to-fly firmware export pipeline.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


# =============================================================================
# RSL-RL PPO Configuration
# =============================================================================

@configclass
class CrazyflieL2FPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration for Crazyflie L2F.
    
    Hyperparameters match learning-to-fly/src/config/ppo_config.h exactly:
    - 64-64 hidden layers with tanh activation (FAST_TANH)
    - PPO hyperparameters from PPOParameters struct
    """
    
    # Environment - Optimized for RTX 4090 (24GB VRAM)
    # RSL-RL: num_steps_per_env * num_envs = total steps per update
    # With 4096 envs: 32 * 4096 = 131k steps per update (good balance)
    num_steps_per_env = 32
    max_iterations = 500  # ~65M total samples
    
    # Logging
    save_interval = 50
    experiment_name = "crazyflie_l2f"
    run_name = "ppo"
    logger = "tensorboard"
    
    # Resume
    resume = False
    load_run = None
    load_checkpoint = None
    
    # Policy - CRITICAL: Must match L2F architecture exactly
    # Architecture: obs -> 64 -> 64 -> action_dim
    # Activation: FAST_TANH (same as TD3)
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.6,  # exp(-0.5) ≈ 0.6, from INITIAL_LOG_STD = -0.5
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        activation="tanh",  # Matches FAST_TANH in firmware
    )
    
    # Algorithm - PPO hyperparameters from ppo_config.h PPOParameters
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,  # VALUE_LOSS_COEFFICIENT
        use_clipped_value_loss=True,  # CLIP_VALUE_LOSS = true
        clip_param=0.2,  # CLIP_EPSILON
        entropy_coef=0.005,  # ENTROPY_COEFFICIENT (lower for more deterministic)
        num_learning_epochs=10,  # N_EPOCHS
        num_mini_batches=16,  # RTX 4090: 4096 envs * 32 steps / 16 = 8192 samples per batch
        learning_rate=3e-4,  # ACTOR_LEARNING_RATE / CRITIC_LEARNING_RATE
        schedule="fixed",
        gamma=0.99,  # GAMMA
        lam=0.95,  # GAE_LAMBDA
        desired_kl=0.01,
        max_grad_norm=0.5,  # MAX_GRAD_NORM
    )
