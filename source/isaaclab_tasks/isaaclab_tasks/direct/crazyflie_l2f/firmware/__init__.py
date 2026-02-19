# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""
Crazyflie Firmware Deployment Package

This package provides tools for building and deploying trained
policies to real Crazyflie hardware.

Components:
    - build_firmware: Cross-platform firmware build script
    - rl_tools_adapter.cpp: Firmware adapter with PPO support
    - Dockerfile: Docker build environment

Usage:
    python -m isaaclab_tasks.direct.crazyflie_l2f.firmware.build_firmware \
        --checkpoint actor.h --output cf2.bin
"""

from pathlib import Path

# Package directory
FIRMWARE_DIR = Path(__file__).parent

# Key files
DOCKERFILE = FIRMWARE_DIR / "Dockerfile"
RL_TOOLS_ADAPTER_CPP = FIRMWARE_DIR / "rl_tools_adapter.cpp"
RL_TOOLS_ADAPTER_H = FIRMWARE_DIR / "rl_tools_adapter.h"


def get_firmware_dir() -> Path:
    """Get the firmware package directory."""
    return FIRMWARE_DIR


__all__ = [
    "FIRMWARE_DIR",
    "DOCKERFILE",
    "RL_TOOLS_ADAPTER_CPP",
    "RL_TOOLS_ADAPTER_H",
    "get_firmware_dir",
]
