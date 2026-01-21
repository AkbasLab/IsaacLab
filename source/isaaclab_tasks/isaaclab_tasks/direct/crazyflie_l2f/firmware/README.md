# Crazyflie RL Firmware Build System

This directory contains a **self-contained** firmware build system for deploying RL policies trained in Isaac Lab to the Crazyflie 2.x quadrotor. The implementation follows the **exact same format** as the [learning-to-fly](https://github.com/arplaboratory/learning-to-fly) project to ensure sim-to-real compatibility.

## Overview

The pipeline:
1. **Train** a policy in Isaac Lab using the `crazyflie_l2f` environment
2. **Export** the policy to rl_tools checkpoint format (`actor.h`)
3. **Build** the Crazyflie firmware with the embedded policy
4. **Flash** the firmware to the Crazyflie

## Directory Structure

```
firmware/
├── Dockerfile              # Docker build environment
├── build_firmware.py       # Build script
├── convert_checkpoint.py   # MLP → Sequential format converter
├── controller/             # Controller source files (from learning_to_fly_controller)
│   ├── rl_tools_adapter.cpp
│   ├── rl_tools_adapter.h
│   ├── rl_tools_controller.c
│   ├── rl_tools_controller.h
│   ├── Makefile
│   ├── Kbuild
│   ├── config
│   └── data/
│       └── actor.h         # Your trained policy goes here
└── README.md
```

## Requirements

- **Docker** - For reproducible cross-compilation environment
- **Python 3.8+** - For running build scripts
- **Trained policy** - From Isaac Lab `crazyflie_l2f` environment

## Quick Start

### 1. Train a Policy

```bash
cd IsaacLab
python source/isaaclab_tasks/isaaclab_tasks/direct/crazyflie_l2f/train_ppo.py
```

### 2. Export to Firmware Format

```python
from crazyflie_l2f.export_to_firmware import export_policy_to_firmware

# Load your trained policy
policy = ...  # Your L2FActorNetwork

# Export with observation normalization baked in
export_policy_to_firmware(
    policy=policy,
    output_path="./checkpoints/actor.h",
    model_name="my_isaac_lab_policy",
    obs_mean=running_mean_std.mean,  # From training
    obs_std=running_mean_std.std,
)
```

### 3. Build Firmware

```bash
cd source/isaaclab_tasks/isaaclab_tasks/direct/crazyflie_l2f/firmware

# Build firmware with your checkpoint
python build_firmware.py \
    --checkpoint ./checkpoints/actor.h \
    --output ./firmware_output/
```

### 4. Flash to Crazyflie

```bash
# Using cfloader (from crazyflie-clients-python)
cfloader flash ./firmware_output/cf2.bin stm32-fw
```

## Checkpoint Format

The checkpoint format follows the rl_tools Sequential module format exactly as used by learning-to-fly:

```cpp
namespace rl_tools::checkpoint::actor {
    namespace layer_0 { ... }  // 146 → 64, Input group
    namespace layer_1 { ... }  // 64 → 64, Normal group
    namespace layer_2 { ... }  // 64 → 4, Output group
    namespace model_definition { ... }
    const MODEL model = {...};
}

namespace rl_tools::checkpoint::observation { ... }  // Test observation
namespace rl_tools::checkpoint::action { ... }       // Expected action
namespace rl_tools::checkpoint::meta { name[], commit_hash[] }
```

### Observation Space (146 dimensions)

| Index | Dimension | Description |
|-------|-----------|-------------|
| 0-2   | 3 | Position error (relative to target) |
| 3-11  | 9 | Rotation matrix (flattened 3x3) |
| 12-14 | 3 | Linear velocity |
| 15-17 | 3 | Angular velocity |
| 18-145| 128 | Action history (32 steps × 4 actions) |

### Action Space (4 dimensions)

| Index | Description |
|-------|-------------|
| 0-3   | Motor commands [-1, 1], mapped to [0, MAX_RPM] |

## PPO Observation Normalization

PPO policies are trained with normalized observations. The export script **bakes the normalization into the first layer weights** so the firmware doesn't need modification:

```
normalized_obs = (obs - mean) / std
layer_0_output = W @ normalized_obs + b
              = W @ ((obs - mean) / std) + b
              = (W / std) @ obs + (b - W @ mean / std)
              = W_new @ obs + b_new
```

This approach:
- ✅ Uses the original firmware adapter (no modifications needed)
- ✅ Maintains exact compatibility with learning-to-fly
- ✅ Works with the existing Crazyflie control loop

## Docker Build Details

The Dockerfile sets up:
- Ubuntu 22.04 base image
- gcc-arm-none-eabi cross-compiler
- crazyflie-firmware (from Bitcraze)
- rl_tools headers (from rl-tools)

The build process:
1. Clones required submodules (crazyflie-firmware, rl_tools)
2. Configures for CF2 platform
3. Compiles with the out-of-tree controller
4. Produces `cf2.bin` firmware

## Troubleshooting

### Docker not found
Install Docker Desktop: https://docs.docker.com/get-docker/

### Build fails with "actor.h not found"
Ensure your checkpoint is exported and passed to build_firmware.py

### Firmware doesn't hover
1. Check observation normalization is correctly applied
2. Verify action scaling ([-1, 1] to RPM)
3. Check position error clipping limits match training

### Test fails on device
The firmware runs a self-test on boot that compares checkpoint output against expected values. Large errors (>0.2) indicate a problem with the checkpoint format.

## References

- [learning-to-fly](https://github.com/arplaboratory/learning-to-fly) - Original RL training and firmware
- [learning_to_fly_controller](https://github.com/arplaboratory/learning_to_fly_controller) - Crazyflie controller
- [rl_tools](https://github.com/rl-tools/rl-tools) - Neural network inference library
- [crazyflie-firmware](https://github.com/bitcraze/crazyflie-firmware) - Crazyflie base firmware
