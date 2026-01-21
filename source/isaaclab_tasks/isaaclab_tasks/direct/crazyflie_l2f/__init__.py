# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""
Crazyflie Learning-to-Fly Integration Package

This package provides a complete, self-contained pipeline for training policies 
in Isaac Lab that can be deployed to real Crazyflie hardware.

Components:
    - CrazyflieL2FEnv: Isaac Lab environment matching L2F observation/dynamics
    - L2FActorCritic: Neural network architecture matching L2F firmware
    - export_to_firmware: Export trained policies to rl_tools checkpoint format
    - train_ppo: PPO training script
    - calibrate: Calibration suite for physics parity validation
    - firmware/: Complete firmware build toolchain (Docker-based)

Quick Start:
    1. Run calibration (validates physics parity):
        python -m isaaclab_tasks.direct.crazyflie_l2f.calibrate --num-envs 64
    
    2. Train a policy (calibration runs automatically):
        python -m isaaclab_tasks.direct.crazyflie_l2f.train_ppo --num_envs 4096
    
    3. Build firmware (actor.h is auto-exported):
        python -m isaaclab_tasks.direct.crazyflie_l2f.firmware.build_firmware \\
            --checkpoint logs/crazyflie_l2f/actor.h
    
    4. Flash to Crazyflie:
        cfloader flash build_firmware/cf2.bin stm32-fw

This package is self-contained and does not require the learning-to-fly repository.

References:
    - Original learning-to-fly: https://github.com/arplaboratory/learning-to-fly
    - rl_tools: https://github.com/rl-tools/rl-tools
"""

import gymnasium as gym

from . import agents
from . import firmware
from .crazyflie_l2f_env import (
    CrazyflieL2FEnv, 
    CrazyflieL2FEnvCfg,
    L2FPhysicsCfg,
    L2FObservationCfg,
    L2FRewardCfg,
    L2FInitializationCfg,
    L2FTerminationCfg,
    L2FDomainRandomizationCfg,
    L2FCalibrationCfg,
)
from .networks import L2FActorNetwork, L2FCriticNetwork, L2FActorCritic, RunningMeanStd
from .export_to_firmware import export_policy_to_firmware

__all__ = [
    # Environment
    "CrazyflieL2FEnv",
    "CrazyflieL2FEnvCfg",
    # Configuration classes
    "L2FPhysicsCfg",
    "L2FObservationCfg",
    "L2FRewardCfg",
    "L2FInitializationCfg",
    "L2FTerminationCfg",
    "L2FDomainRandomizationCfg",
    "L2FCalibrationCfg",
    # Networks
    "L2FActorNetwork",
    "L2FCriticNetwork", 
    "L2FActorCritic",
    "RunningMeanStd",
    # Export
    "export_policy_to_firmware",
    # Firmware
    "firmware",
]

# Register the environment with Gymnasium
gym.register(
    id="Isaac-Crazyflie-L2F-Direct-v0",
    entry_point="isaaclab_tasks.direct.crazyflie_l2f.crazyflie_l2f_env:CrazyflieL2FEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.crazyflie_l2f.crazyflie_l2f_env:CrazyflieL2FEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CrazyflieL2FPPORunnerCfg",
    },
)

# Also register a play/evaluation variant with fewer environments  
gym.register(
    id="Isaac-Crazyflie-L2F-Direct-Play-v0",
    entry_point="isaaclab_tasks.direct.crazyflie_l2f.crazyflie_l2f_env:CrazyflieL2FEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.direct.crazyflie_l2f.crazyflie_l2f_env:CrazyflieL2FEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CrazyflieL2FPPORunnerCfg",
    },
)
