# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""
Crazyflie Learning-to-Fly Integration Package

This package provides training for Crazyflie hover policies compatible with L2F firmware.

Quick Start:
    Train a policy:
        isaaclab.bat -p train_hover.py --num_envs 4096 --max_iterations 500
    
    Play/test a trained policy:
        isaaclab.bat -p train_hover.py --play --num_envs 4

Main scripts:
    - train_hover.py: Unified training and evaluation script
    - export_to_firmware.py: Export trained policies to firmware format
"""

# Don't auto-import modules - train_hover.py is standalone and handles its own imports
