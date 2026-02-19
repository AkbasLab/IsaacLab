# Crazyflie PPO Hover Pipeline - Complete Guide

Complete pipeline for training a PPO hover policy in Isaac Lab and deploying it to a physical Crazyflie 2.1.

**Last Updated:** January 21, 2026  
**Status:** Fully Functional

---

## Prerequisites

- **Isaac Sim 4.5+** installed
- **Docker Desktop** with WSL2 backend
- **Crazyflie 2.1** with Crazyradio PA
- **cflib** Python package (`pip install cflib`)

---

## Pipeline Overview

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   TRAIN     │───▶│   EXPORT    │───▶│   BUILD     │───▶│   FLASH     │───▶│   TEST      │
│  (Isaac)    │    │  (Python)   │    │  (Docker)   │    │ (cfloader)  │    │  (cflib)    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
     ~30min             ~5sec             ~2min             ~30sec             Manual
```

---

## Step 1: Train the Policy

### Quick Test (verify setup works)
```powershell
cd <LEARNING_TO_FLY_ROOT>\IsaacLab

& "<ISAAC_SIM_PATH>\python.bat" source\isaaclab_tasks\isaaclab_tasks\direct\crazyflie_l2f\train_hover.py --headless --num_envs 16 --max_iterations 2
```

### Full Training
```powershell
cd <LEARNING_TO_FLY_ROOT>\IsaacLab

& "<ISAAC_SIM_PATH>\python.bat" source\isaaclab_tasks\isaaclab_tasks\direct\crazyflie_l2f\train_hover.py --headless --num_envs 4096 --max_iterations 500
```

### Expected Output
- Progress: `Iter 50/500 | Reward: 0.XXX | ...`
- Checkpoints saved to: `source/isaaclab_tasks/.../crazyflie_l2f/checkpoints/`
- Final reward should be **> 0.6** for good hover

### Training Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_envs` | 4096 | Parallel environments |
| `--max_iterations` | 500 | Training iterations |
| `--headless` | False | Run without GUI |

---

## Step 2: Export Policy to Firmware Header

```powershell
cd <LEARNING_TO_FLY_ROOT>\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\crazyflie_l2f

& "<ISAAC_SIM_PATH>\python.bat" export_policy_standalone.py
```

### What It Does
1. Loads `checkpoints/best_model.pt`
2. Converts PyTorch weights to C byte arrays
3. Writes to `<LEARNING_TO_FLY_ROOT>\controller\data\actor.h`

### Verify Export
```powershell
# Check file size (should be ~264,000 bytes)
(Get-Item "<LEARNING_TO_FLY_ROOT>\controller\data\actor.h").Length

# Should see layer statistics in console output:
# layer_0 weight: mean=X.XXXX, std=0.XXXX (should NOT be 0)
# layer_1 weight: mean=X.XXXX, std=0.XXXX  
# layer_2 weight: mean=X.XXXX, std=0.XXXX
```

---

## Step 3: Build Firmware with Docker

```powershell
cd <LEARNING_TO_FLY_ROOT>

# Create output directory (first time only)
New-Item -ItemType Directory -Path "output" -Force

# Build firmware (adjust WSL mount path to match your installation)
wsl docker run --rm -v <WSL_LEARNING_TO_FLY_PATH>:/data -v <WSL_LEARNING_TO_FLY_PATH>/output:/output arpllab/learning_to_fly_build_firmware
```

### Expected Output
```
...
Build for the cf2!
Build 0:XXXXXXXX (NA) CLEAN
Flash |  418128/1032192 (41%),  614064 free
RAM   |   83816/131072  (64%),   47256 free
CCM   |   62020/65536   (95%),    3516 free
```

### Verify Build
```powershell
# Check firmware was created (~418,000 bytes)
(Get-Item "<LEARNING_TO_FLY_ROOT>\output\cf2.bin").Length
```

---

## Step 4: Flash Firmware to Crazyflie

### Put Drone in Bootloader Mode
1. Turn off the Crazyflie
2. Hold the power button for **3+ seconds**
3. Blue LEDs should blink alternately

### Flash via Command Line
```powershell
cfloader flash <LEARNING_TO_FLY_ROOT>\output\cf2.bin stm32-fw -w radio://0/80/2M/<YOUR_DRONE_ADDRESS>
```

### Alternative: Crazyflie Client (GUI)
1. Open Crazyflie Client
2. Connect → Bootloader
3. Browse to `output/cf2.bin`
4. Click Flash

---

## Step 5: Test the Policy

### Scan for Drones
```powershell
cd <LEARNING_TO_FLY_ROOT>\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\crazyflie_l2f

python test_hover.py --scan
```

### Run Hover Test
```powershell
python test_hover.py --uri radio://0/80/2M/<YOUR_DRONE_ADDRESS> --height 0.3
```

### Controls
| Key | Action |
|-----|--------|
| **SPACE** or **ENTER** (hold) | Hover at target height |
| **Release** | Stop motors immediately |
| **Q** or **ESC** | Quit program |
| **Ctrl+C** | Emergency stop |

### Safety Features
- Dead-man's switch (motors stop when key released)
- 5-second connection timeout
- Automatic disconnect on packet loss
- Graceful radio cleanup

---

## Quick Reference (Copy-Paste)

Replace placeholders before running:
- `<LEARNING_TO_FLY_ROOT>`: Path to learning-to-fly repository
- `<ISAAC_SIM_PATH>`: Path to Isaac Sim installation
- `<WSL_LEARNING_TO_FLY_PATH>`: WSL-style path (e.g., /mnt/c/path/to/learning-to-fly)
- `<YOUR_DRONE_ADDRESS>`: Crazyflie address (e.g., E7E7E7E7E7)

```powershell
# === FULL PIPELINE ===

# 1. TRAIN
cd <LEARNING_TO_FLY_ROOT>\IsaacLab
& "<ISAAC_SIM_PATH>\python.bat" source\isaaclab_tasks\isaaclab_tasks\direct\crazyflie_l2f\train_hover.py --headless --num_envs 4096 --max_iterations 500

# 2. EXPORT
cd <LEARNING_TO_FLY_ROOT>\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\crazyflie_l2f
& "<ISAAC_SIM_PATH>\python.bat" export_policy_standalone.py

# 3. BUILD
cd <LEARNING_TO_FLY_ROOT>
wsl docker run --rm -v <WSL_LEARNING_TO_FLY_PATH>:/data -v <WSL_LEARNING_TO_FLY_PATH>/output:/output arpllab/learning_to_fly_build_firmware

# 4. FLASH (put drone in bootloader first!)
cfloader flash <LEARNING_TO_FLY_ROOT>\output\cf2.bin stm32-fw -w radio://0/80/2M/<YOUR_DRONE_ADDRESS>

# 5. TEST
cd <LEARNING_TO_FLY_ROOT>\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\crazyflie_l2f
python test_hover.py --height 0.3
```

---

## Architecture Details

### Network Architecture
```
Input (146) → Dense(64, tanh) → Dense(64, tanh) → Dense(4) → Output
```

### Observation Space (146 dimensions)
| Component | Dims | Description |
|-----------|------|-------------|
| Position | 3 | x, y, z in world frame |
| Rotation | 9 | Flattened 3×3 rotation matrix |
| Linear velocity | 3 | vx, vy, vz |
| Angular velocity | 3 | wx, wy, wz |
| Action history | 128 | 32 timesteps × 4 motors |

### Action Space (4 dimensions)
- Normalized motor commands in **[-1, 1]**
- Conversion: `rpm = (action + 1) * 0.5 * 21702`

### Key Physical Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| Mass | 0.027 kg | Crazyflie mass |
| Arm length | 0.046 m | Motor arm length |
| KF | 3.16e-10 | Thrust coefficient |
| Max RPM | 21702 | Maximum motor speed |
| Hover thrust | ~0.334 | Normalized action for hover |

---

## Troubleshooting

### Training

**Out of GPU memory:**
```powershell
--num_envs 2048  # Reduce parallel environments
```

**Reward not increasing:**
- Run more iterations: `--max_iterations 1000`
- Check environment is resetting properly

### Export

**"Layer weights are all zeros":**
- Verify checkpoint loaded: check console output for weight statistics
- Ensure using `export_policy_standalone.py` (has embedded network class)

### Build

**Docker command not found:**
```powershell
# Verify Docker is running
wsl docker --version
```

**"Cannot create /output/cf2.bin":**
```powershell
New-Item -ItemType Directory -Path "<LEARNING_TO_FLY_ROOT>\output" -Force
```

### Testing

**Connection timeout:**
- Check Crazyradio is plugged in
- Scan for drones: `python test_hover.py --scan`
- Verify URI matches your drone

**Drone flips/unstable:**
- Retrain with more iterations
- Check motor ordering in firmware
- Verify observation normalization

---

## Files Reference

| File | Purpose |
|------|---------|
| `train_hover.py` | PPO training script |
| `export_policy_standalone.py` | Export weights to C header |
| `test_hover.py` | Dead-man's switch flight test |
| `crazyflie_l2f_env.py` | Isaac Lab environment |
| `networks.py` | L2F-compatible actor network |
| `checkpoints/best_model.pt` | Trained model weights |
| `controller/data/actor.h` | Exported C header |
| `output/cf2.bin` | Compiled firmware |
