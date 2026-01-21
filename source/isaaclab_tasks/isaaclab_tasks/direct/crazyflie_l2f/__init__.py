# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""
Crazyflie Learning-to-Fly Integration Package

This package provides a complete, self-contained pipeline for training hover 
policies in Isaac Lab that can be deployed to real Crazyflie 2.1 hardware.

Components:
    - train_hover.py: Self-contained PPO training script with environment, 
                      network, and export functionality built-in
    - export_to_firmware.py: Export trained policies to rl_tools C header format
    - firmware/: Complete firmware build toolchain (Docker-based)
    - crazyflie_21_cfg.py: Crazyflie 2.1 articulation configuration

Quick Start:
    1. Train a policy:
        python train_hover.py --num_envs 4096 --max_iterations 600 --headless
    
    2. Export to firmware header (done automatically, or manually):
        python export_to_firmware.py checkpoints/best_model.pt firmware/actor_isaac_lab.h
    
    3. Build firmware:
        python -m firmware.build_firmware firmware/actor_isaac_lab.h
    
    4. Flash to Crazyflie:
        cfloader flash build_firmware/cf2.bin stm32-fw -w radio://0/80/2M

This package is self-contained and does not require the learning-to-fly repository.

References:
    - Original learning-to-fly: https://github.com/arplaboratory/learning-to-fly
    - rl_tools: https://github.com/rl-tools/rl-tools
"""

from . import agents
from . import firmware

__all__ = [
    # Subpackages
    "agents",
    "firmware",
]
