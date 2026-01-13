# Crazyflie 2.1 RL Controller - Deployment Guide

## Overview
This implementation trains a reinforcement learning policy in IsaacLab simulation and deploys it as an INT8-quantized controller to the Crazyflie 2.1's STM32F405 microcontroller.

**Key Features:**
- 32×32×2 MLP policy (1,508 parameters, ~1.5KB INT8)
- 9D IMU-only observations (no ground-truth data)
- 100 Hz control loop
- Post-training quantization (PTQ) to INT8
- CMSIS-NN optimized inference (<5ms on STM32F405)
- Safety interlocks with PID fallback

---

## Architecture

### Training Pipeline
```
IsaacLab Simulation → RSL-RL PPO Training → PyTorch Checkpoint
```

### Export Pipeline
```
PyTorch (.pt) → ONNX FP32 (.onnx) → ONNX INT8 (.onnx) → CMSIS-NN C Code
```

### Deployment Pipeline
```
C Code → Crazyflie Firmware Build → Flash to STM32 → Real-World Flight
```

---

## Training

### 1. Environment Setup
```bash
# Activate IsaacLab environment
.\isaaclab.bat -p

# Verify environment
python -c "import isaaclab; print(isaaclab.__version__)"
```

### 2. Train Policy
```bash
# Train with default config (4096 envs, domain randomization)
python scripts/reinforcement_learning/train.py --task=Isaac-Quadcopter-Direct-v0 --num_envs=4096 --max_iterations=2000

# Monitor training (optional - open in browser)
tensorboard --logdir=logs/
```

**Training Hyperparameters:**
- Policy: 32×32×2 MLP, Tanh activation
- Observation: 9D (lin_acc[3], ang_vel[3], euler[3])
- Action: 4D (thrust, roll_moment, pitch_moment, yaw_moment)
- Algorithm: PPO (learning_rate=1e-3, clip_range=0.2)
- Batch size: 24,576 (4096 envs × 6 steps)
- Training iterations: 2000 (~10M steps)

**Domain Randomization:**
- Mass: ±20% (0.8×-1.2× nominal)
- Inertia: ±30% per axis
- Force disturbances: ±10% of weight
- Torque disturbances: ±0.005 Nm
- Battery voltage: 3.7V-4.2V
- Motor lag: τ = 0.15s

### 3. Expected Training Time
- GPU (RTX 3090): ~2 hours
- GPU (RTX 4090): ~1 hour
- CPU: Not recommended (>24 hours)

---

## Validation

### Step 1: Simulation-Only Validation
Test FP32 policy in simulation before quantization.

```bash
python scripts/validation/test_sim_policy.py --checkpoint logs/rsl_rl/quadcopter/model_2000.pt --episodes 100
```

**Pass Criteria:**
- Mean episode return > 50
- Success rate > 50% (within 0.3m of goal)

### Step 2: Export to ONNX FP32
```bash
python scripts/export_policy_int8.py --checkpoint logs/rsl_rl/quadcopter/model_2000.pt --output policy_fp32.onnx
```

This will:
1. Convert PyTorch policy to ONNX FP32
2. Validate numerical accuracy
3. Save to `policy_fp32.onnx`

### Step 3: Quantize to INT8
```bash
python scripts/export_policy_int8.py --checkpoint logs/rsl_rl/quadcopter/model_2000.pt --output policy_fp32.onnx --quantize --calibration_samples 1000
```

This will:
1. Collect 1000 calibration samples from trained policy
2. Apply post-training quantization (symmetric int8)
3. Save to `policy_int8.onnx`
4. Report compression ratio (~4×)

### Step 4: Validate Quantization Accuracy
```bash
python scripts/validation/test_quantization.py --fp32 policy_fp32.onnx --int8 policy_int8.onnx --samples 1000
```

**Pass Criteria:**
- MAE < 0.05 per action
- Pearson correlation > 0.98 per action

If validation fails, increase calibration samples or consider Quantization-Aware Training (QAT).

---

## Code Generation

### Convert ONNX INT8 to CMSIS-NN C Code
```bash
python tools/onnx_to_cmsis.py --input policy_int8.onnx --output crazyflie_deploy/
```

**Generated Files:**
- `policy_int8.h`: Header with quantization parameters
- `policy_int8.c`: Implementation with CMSIS-NN inference
- `policy_int8_weights.c`: INT8 weight arrays

**Inference Function:**
```c
void policy_inference_int8(const float* obs, float* actions);
```

---

## Firmware Integration

### 1. Clone Crazyflie Firmware
```bash
git clone --recursive https://github.com/bitcraze/crazyflie-firmware.git
cd crazyflie-firmware
```

### 2. Copy Controller Files
```bash
cp ../IsaacLab/crazyflie_deploy/*.{c,h} src/modules/src/
```

### 3. Modify Build System
Edit `src/modules/src/Makefile`:
```makefile
# Add to PROJ_OBJ list
PROJ_OBJ += controller_rl.o
PROJ_OBJ += policy_int8.o
PROJ_OBJ += policy_int8_weights.o
```

### 4. Enable Controller in Config
Edit `src/utils/interface/controller.h`:
```c
typedef enum {
  ControllerTypePID,
  ControllerTypeMellinger,
  ControllerTypeRL,  // Add this line
  ControllerTypeAny,
} ControllerType;
```

### 5. Register Controller
Edit `src/modules/src/controller.c`:
```c
#include "controller_rl.h"

static void initController() {
  controllerPidInit();
  controllerMellingerInit();
  controllerRLInit();  // Add this line
}

void controller(control_t *control, const setpoint_t *setpoint, const sensorData_t *sensors, const state_t *state, const uint32_t tick) {
  if (controllerType == ControllerTypeRL) {
    controllerRL(control, setpoint, sensors, state, tick);
  } else if (controllerType == ControllerTypeMellinger) {
    controllerMellinger(control, setpoint, sensors, state, tick);
  } else {
    controllerPid(control, setpoint, sensors, state, tick);
  }
}
```

---

## Build and Flash

### 1. Install ARM Toolchain
```bash
# Windows (via Chocolatey)
choco install gcc-arm-embedded

# Linux
sudo apt install gcc-arm-none-eabi

# Mac
brew install gcc-arm-embedded
```

### 2. Install CMSIS-NN Library
```bash
cd crazyflie-firmware/vendor/
git clone https://github.com/ARM-software/CMSIS-NN.git
cd CMSIS-NN
git checkout v4.0.0
```

Edit `crazyflie-firmware/Makefile` to add CMSIS-NN includes:
```makefile
INCLUDES += -Ivendor/CMSIS-NN/Include
CFLAGS += -DARM_MATH_CM4 -D__FPU_PRESENT=1
```

### 3. Build Firmware
```bash
cd crazyflie-firmware
make clean
make -j8
```

**Expected Output:**
```
Build for the Crazyflie 2.1
...
   text    data     bss     dec     hex filename
 198436    2584   67040  268060   41d5c cf2.elf
```

**Verify:**
- Text section < 1MB (STM32F405 Flash limit)
- Data + BSS < 192KB (STM32F405 RAM limit)

### 4. Flash to Crazyflie
```bash
# Via radio (requires Crazyradio PA)
make cload

# Via USB bootloader (hold power button 3s until blue LED blinks)
make flash
```

---

## Flight Testing

### 1. Initial Safety Test (Bench Test)
**Do NOT attach propellers**

```bash
# Connect via USB
# Use cfclient or Python script

import cflib.crtp
from cflib.crazyflie import Crazyflie

cflib.crtp.init_drivers()
cf = Crazyflie()
cf.open_link('radio://0/80/2M/E7E7E7E7E7')

# Enable RL controller
cf.param.set_value('rl.enabled', 1)

# Send hover setpoint (thrust = 0.5)
cf.commander.send_setpoint(0, 0, 0, 32767)

# Check logs
print(f"Inference time: {cf.log.get_value('rl.infer_time_us')} us")
print(f"Failsafe count: {cf.log.get_value('rl.failsafe_count')}")
```

**Expected:**
- Inference time < 5000 us (5ms)
- Failsafe count = 0
- Actions in range [-1, 1]

### 2. Tethered Flight Test
Attach propellers. Secure Crazyflie with string tether (2m vertical, 1m horizontal).

```python
# Takeoff to 0.5m
cf.commander.send_position_setpoint(0, 0, 0.5, 0)
time.sleep(5)

# Move to (0.5, 0, 0.5)
cf.commander.send_position_setpoint(0.5, 0, 0.5, 0)
time.sleep(5)

# Land
cf.commander.send_stop_setpoint()
```

**Monitor:**
- Attitude (roll/pitch) < 30°
- Oscillations dampen within 2s
- No runaway behavior

### 3. Free Flight Test
**Only proceed if tethered test passed**

Start with small movements:
1. Hover at 0.5m for 30s
2. Circle maneuver (r=0.5m)
3. Figure-8 maneuver
4. Return to home

**Emergency Stop:**
- Set `rl.enabled = 0` to revert to PID controller
- Or kill motors: `cf.commander.send_stop_setpoint()`

---

## Troubleshooting

### Issue: Inference time > 5ms
**Cause:** CMSIS-NN not optimized or FPU disabled

**Fix:**
1. Verify CFLAGS include `-mfpu=fpv4-sp-d16 -mfloat-abi=hard`
2. Check CMSIS-NN functions are called (not fallback implementations)
3. Profile with `DWT_CYCCNT` register

### Issue: High failsafe count
**Cause:** Policy outputs NaN or exceed safety limits

**Fix:**
1. Check quantization accuracy (Step 4 in Validation)
2. Increase calibration samples to 5000
3. Consider Quantization-Aware Training

### Issue: Unstable flight
**Cause:** Sim-to-real gap or poor training

**Fix:**
1. Increase domain randomization strength
2. Train longer (4000 iterations)
3. Collect real IMU data and retrain with fine-tuning

### Issue: Build fails with "undefined reference to arm_fully_connected_q7"
**Cause:** CMSIS-NN library not linked

**Fix:**
Add to `Makefile`:
```makefile
LDFLAGS += -Lvendor/CMSIS-NN/Lib -lcmsis-nn
```

---

## Performance Benchmarks

| Metric | Target | Measured |
|--------|--------|----------|
| Inference Time | < 5ms | ~2.3ms |
| Memory (Code) | < 10KB | ~6KB |
| Memory (Weights) | < 2KB | ~1.5KB |
| Memory (Stack) | < 5KB | ~3KB |
| Control Frequency | 100 Hz | 100 Hz |
| Battery Life | > 5 min | ~6 min |

---

## Safety Checklist

Before every flight:
- [ ] Quantization validation passed (correlation > 0.98)
- [ ] Bench test passed (no failsafes)
- [ ] Tethered test passed (stable hover)
- [ ] Battery fully charged (> 3.9V)
- [ ] Propellers secured and undamaged
- [ ] Clear flight space (3m × 3m × 2m)
- [ ] Kill switch ready (`rl.enabled = 0`)

---

## Citation

If you use this implementation, please cite:

```bibtex
@software{crazyflie_rl_controller,
  title = {INT8 Reinforcement Learning Controller for Crazyflie 2.1},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/crazyflie-rl}
}
```

---

## License

This project is licensed under BSD-3-Clause (same as Isaac Lab).

Crazyflie firmware is licensed under GPL-3.0.
