# Crazyflie Firmware Deployment Guide

This guide covers deploying trained Isaac Lab hover policies to the physical Crazyflie 2.1 drone.

**All files are self-contained within the IsaacLab crazyflie_l2f module.**

## Directory Structure

```
crazyflie_l2f/
├── checkpoints/
│   └── best_model.pt          # Trained PPO policy
├── firmware/
│   └── actor_isaac_lab.h      # Exported C header (263 KB)
├── build_firmware/
│   └── cf2.bin                # Compiled firmware binary (419 KB)
├── export_to_c_header.py      # Export checkpoint → C header
├── train_hover.py             # Training script
└── DEPLOYMENT.md              # This guide
```

## Pipeline Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌────────────┐
│  Train Policy   │ --> │  Export Header   │ --> │  Build Firmware │ --> │  Flash     │
│  (train_hover)  │     │  (actor_*.h)     │     │  (cf2.bin)      │     │  (cfloader)│
└─────────────────┘     └──────────────────┘     └─────────────────┘     └────────────┘
```

## Step 1: Train Policy (Completed)

Your policy has been trained with:
- **Best reward**: 1.686 at iteration 597
- **Architecture**: 146 → 64 → 64 → 4 (L2F compatible)
- **Checkpoint**: `checkpoints/best_model.pt`

## Step 2: Export to C Header (Completed)

The checkpoint has been exported to:
- **Header file**: `firmware/actor_isaac_lab.h` (263 KB)
- **Format**: rl_tools Sequential module
- **Normalization**: Baked into layer_0 weights

## Step 3: Build Firmware

### Prerequisites
- Docker Desktop for Windows, OR
- WSL2 with Docker installed

### Using WSL + Docker (Windows)

All outputs stay within the IsaacLab repository:

```powershell
# From PowerShell - build firmware with output in crazyflie_l2f/build_firmware/
wsl -e bash -c 'docker run --rm \
    -v "/mnt/d/coding/Capstone/learning-to-fly/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/crazyflie_l2f/firmware/actor_isaac_lab.h:/controller/data/actor.h:ro" \
    -v "/mnt/d/coding/Capstone/learning-to-fly/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/crazyflie_l2f/build_firmware:/output" \
    arpllab/learning_to_fly_build_firmware'
```

### Using Docker Directly (Linux/Mac)

```bash
# From the crazyflie_l2f directory
docker run --rm \
    -v "${PWD}/firmware/actor_isaac_lab.h:/controller/data/actor.h:ro" \
    -v "${PWD}/build_firmware:/output" \
    arpllab/learning_to_fly_build_firmware
```

### Native Build (Linux/WSL)

```bash
# Clone the controller repository
git clone https://github.com/arplaboratory/learning_to_fly_controller.git
cd learning_to_fly_controller
git submodule update --init --recursive

# Copy your trained policy
cp /path/to/actor_isaac_lab.h data/actor.h

# Build
make
# Output: build/cf2.bin
```

## Step 4: Flash to Crazyflie

### Prerequisites
- Crazyradio PA USB dongle
- cfclient installed: `pip install cfclient`

### Flash Command

```bash
# Put Crazyflie in bootloader mode:
# 1. Turn off the Crazyflie
# 2. Hold power button for 3+ seconds
# 3. Blue LEDs should blink

# Flash the firmware (from crazyflie_l2f directory)
cfloader flash build_firmware/cf2.bin stm32-fw -w radio://0/80/2M
```

### Alternative Radio URIs
- `radio://0/80/2M` - Channel 80, 2Mbit (default)
- `radio://0/80/250K` - Channel 80, 250Kbit (longer range)
- `radio://0/60/2M` - Different channel

## Step 5: Fly the Crazyflie

### Using cfclient (with Gamepad)

1. Launch cfclient: `cfclient`
2. Connect to Crazyflie
3. Use gamepad "hover button" as dead man's switch
4. Default hover height: 0.3m above takeoff position

### Using trigger.py (Safer)

```bash
# Take off with original controller, switch to learned policy
python scripts/trigger.py --mode takeoff_and_switch --height 0.5

# Hover with learned policy from takeoff
python scripts/trigger.py --mode hover_learned --height 0.3

# Trajectory tracking (figure-8)
python scripts/trigger.py --mode trajectory_tracking --trajectory-scale 0.3
```

## Important Notes

### Position Estimation
The policy requires position estimation. Options:
- **Flow Deck v2** - Optical flow + height sensor (recommended for hover)
- **Lighthouse** - Sub-cm accuracy
- **Loco Positioning** - UWB-based

### Safety
⚠️ **Flying with learned controllers is at your own risk!**

- Always have a dead man's switch
- Start with low heights (0.2-0.3m)
- Use `takeoff_and_switch` mode first
- Keep yaw angle near 0 at takeoff
- Test in an open area away from obstacles

### Troubleshooting

| Issue | Solution |
|-------|----------|
| cfloader not found | `pip install cfclient` |
| Radio not detected | Reinstall Crazyradio drivers |
| Build fails | Check Docker is running |
| Drone unstable | Try `takeoff_and_switch` mode |
| Policy too aggressive | Reduce training iterations |

## File Locations

| File | Purpose |
|------|---------|
| `checkpoints/best_model.pt` | Trained PyTorch checkpoint |
| `firmware/actor_isaac_lab.h` | Exported C header |
| `build_firmware/cf2.bin` | Compiled firmware binary |
| `deploy_to_crazyflie.ps1` | Deployment helper script |

## Quick Deploy Script

```powershell
# All-in-one deployment (requires Docker)
.\deploy_to_crazyflie.ps1 -Build -Flash
```
