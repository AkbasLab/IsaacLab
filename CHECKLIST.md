# Implementation Progress Checklist

Use this checklist to track your progress through the full pipeline.

---

## Phase 1: Setup ✓

- [x] IsaacLab installed and verified
- [x] Modified `quadcopter_env.py` (9D observations, motor dynamics, battery model)
- [x] Modified `rsl_rl_ppo_cfg.py` (32×32 architecture, Tanh activation)
- [x] Created export pipeline (`export_policy_int8.py`)
- [x] Created CMSIS-NN converter (`onnx_to_cmsis.py`)
- [x] Created validation scripts (`test_sim_policy.py`, `test_quantization.py`)
- [x] Created firmware controller (`controller_rl.c`, `controller_rl.h`)
- [x] Created documentation (DEPLOYMENT_GUIDE, QUICK_REFERENCE, IMPLEMENTATION_SUMMARY)

**Status:** COMPLETE - All code and documentation in place

---

## Phase 2: Training ⏳

- [ ] Start training run
  ```bash
  python scripts/reinforcement_learning/train.py --task=Isaac-Quadcopter-Direct-v0 --num_envs=4096 --max_iterations=2000
  ```

- [ ] Monitor training progress (TensorBoard)
  ```bash
  tensorboard --logdir=logs/
  ```

- [ ] Verify convergence
  - [ ] Episode return increasing (target: > 50)
  - [ ] Success rate improving (target: > 50%)
  - [ ] No NaN or exploding gradients

- [ ] Save final checkpoint
  - Location: `logs/rsl_rl/quadcopter/model_2000.pt`
  - Size: ~25KB (1,508 params × 4 bytes + metadata)

**Estimated Time:** 1-2 hours (RTX 3090/4090)

**Troubleshooting:**
- If stuck at low return: Adjust reward scales in `quadcopter_env.py`
- If training crashes: Reduce num_envs to 2048
- If GPU OOM: Reduce batch size or num_envs

---

## Phase 3: Validation ⏳

### 3.1 Simulation Validation
- [ ] Run FP32 policy test
  ```bash
  python scripts/validation/test_sim_policy.py --checkpoint logs/rsl_rl/quadcopter/model_2000.pt --episodes 100
  ```

- [ ] Check results
  - [ ] Mean return > 50 ✓ PASS / ✗ FAIL
  - [ ] Success rate > 50% ✓ PASS / ✗ FAIL

**If FAIL:** Retrain with adjusted hyperparameters or more iterations (4000)

### 3.2 Export to ONNX
- [ ] Export FP32 model
  ```bash
  python scripts/export_policy_int8.py --checkpoint logs/rsl_rl/quadcopter/model_2000.pt --output policy_fp32.onnx
  ```

- [ ] Verify FP32 ONNX
  - [ ] File created: `policy_fp32.onnx`
  - [ ] Size: ~6KB
  - [ ] No export errors

### 3.3 Quantize to INT8
- [ ] Run quantization
  ```bash
  python scripts/export_policy_int8.py --checkpoint logs/rsl_rl/quadcopter/model_2000.pt --output policy_fp32.onnx --quantize --calibration_samples 1000
  ```

- [ ] Verify INT8 ONNX
  - [ ] File created: `policy_int8.onnx`
  - [ ] Size: ~1.5KB (4× compression)
  - [ ] Compression ratio reported

### 3.4 Quantization Accuracy
- [ ] Run accuracy test
  ```bash
  python scripts/validation/test_quantization.py --fp32 policy_fp32.onnx --int8 policy_int8.onnx --samples 1000
  ```

- [ ] Check results per action
  - [ ] Thrust: MAE < 0.05, Corr > 0.98 ✓ PASS / ✗ FAIL
  - [ ] Roll: MAE < 0.05, Corr > 0.98 ✓ PASS / ✗ FAIL
  - [ ] Pitch: MAE < 0.05, Corr > 0.98 ✓ PASS / ✗ FAIL
  - [ ] Yaw: MAE < 0.05, Corr > 0.98 ✓ PASS / ✗ FAIL

**If FAIL:** Increase calibration samples to 5000, or consider Quantization-Aware Training (QAT)

---

## Phase 4: Code Generation ⏳

- [ ] Generate CMSIS-NN C code
  ```bash
  python tools/onnx_to_cmsis.py --input policy_int8.onnx --output crazyflie_deploy/
  ```

- [ ] Verify generated files
  - [ ] `crazyflie_deploy/policy_int8.h` (header with quantization params)
  - [ ] `crazyflie_deploy/policy_int8.c` (inference implementation)
  - [ ] `crazyflie_deploy/policy_int8_weights.c` (INT8 weight arrays)

- [ ] Inspect generated code
  - [ ] Weight arrays have 1,508 elements
  - [ ] Quantization scales and zero-points present
  - [ ] `policy_inference_int8()` function signature correct

---

## Phase 5: Firmware Integration ⏳

### 5.1 Setup
- [ ] Clone Crazyflie firmware
  ```bash
  git clone --recursive https://github.com/bitcraze/crazyflie-firmware.git
  ```

- [ ] Install ARM toolchain
  ```bash
  # Ubuntu
  sudo apt install gcc-arm-none-eabi
  
  # macOS
  brew install gcc-arm-embedded
  
  # Windows (Chocolatey)
  choco install gcc-arm-embedded
  ```

- [ ] Verify toolchain
  ```bash
  arm-none-eabi-gcc --version
  ```

### 5.2 Integration (Automated)
- [ ] Run integration script
  ```bash
  python tools/integrate_firmware.py --isaac-path . --firmware-path ../crazyflie-firmware
  ```

- [ ] Verify integration
  - [ ] Controller files copied to `src/modules/src/`
  - [ ] `Makefile` patched (controller_rl.o, policy_int8.o added)
  - [ ] `controller.h` patched (ControllerTypeRL added)
  - [ ] `controller.c` patched (init and dispatch added)
  - [ ] CMSIS-NN cloned to `vendor/CMSIS-NN/`
  - [ ] Root `Makefile` patched (CMSIS-NN includes added)

### 5.3 Manual Integration (If Automated Fails)
- [ ] Copy files manually
  ```bash
  cp crazyflie_deploy/*.{c,h} ../crazyflie-firmware/src/modules/src/
  ```

- [ ] Edit `src/modules/src/Makefile`
  - [ ] Add `PROJ_OBJ += controller_rl.o`
  - [ ] Add `PROJ_OBJ += policy_int8.o`
  - [ ] Add `PROJ_OBJ += policy_int8_weights.o`

- [ ] Edit `src/utils/interface/controller.h`
  - [ ] Add `ControllerTypeRL` to enum (before ControllerTypeAny)

- [ ] Edit `src/modules/src/controller.c`
  - [ ] Add `#include "controller_rl.h"`
  - [ ] Add `controllerRLInit();` in `initController()`
  - [ ] Add dispatch logic for ControllerTypeRL

- [ ] Clone CMSIS-NN
  ```bash
  cd ../crazyflie-firmware/vendor
  git clone https://github.com/ARM-software/CMSIS-NN.git
  cd CMSIS-NN
  git checkout v4.0.0
  ```

- [ ] Edit root `Makefile`
  - [ ] Add `INCLUDES += -Ivendor/CMSIS-NN/Include`
  - [ ] Add `CFLAGS += -DARM_MATH_CM4 -D__FPU_PRESENT=1`

---

## Phase 6: Build ⏳

- [ ] Build firmware
  ```bash
  cd ../crazyflie-firmware
  make clean
  make -j8
  ```

- [ ] Check build output
  - [ ] No errors
  - [ ] Text section < 1MB (1,048,576 bytes)
  - [ ] Data + BSS < 192KB (196,608 bytes)
  - [ ] `cf2.elf` and `cf2.bin` generated

**Expected Output:**
```
Build for the Crazyflie 2.1
...
   text    data     bss     dec     hex filename
 198436    2584   67040  268060   41d5c cf2.elf
```

**Troubleshooting:**
- Undefined reference to `arm_fully_connected_q7`: CMSIS-NN not linked
- Flash overflow: Model too large (should not happen with 1.5KB)
- RAM overflow: Reduce batch size in inference (should not happen)

---

## Phase 7: Flash ⏳

### Via Radio (Crazyradio PA)
- [ ] Connect Crazyradio to PC
- [ ] Power on Crazyflie
- [ ] Flash firmware
  ```bash
  make cload
  ```

### Via USB Bootloader
- [ ] Connect Crazyflie via USB
- [ ] Enter bootloader (hold power button 3s until blue LED blinks)
- [ ] Flash firmware
  ```bash
  make flash
  ```

- [ ] Verify flash success
  - [ ] No errors during flash
  - [ ] Crazyflie reboots automatically
  - [ ] LEDs show normal startup sequence

---

## Phase 8: Bench Testing ⏳

**⚠️ DO NOT ATTACH PROPELLERS YET**

### 8.1 Connection Test
- [ ] Connect via cfclient or Python
  ```python
  import cflib.crtp
  from cflib.crazyfie import Crazyflie
  
  cflib.crtp.init_drivers()
  cf = Crazyflie()
  cf.open_link('radio://0/80/2M/E7E7E7E7E7')
  ```

- [ ] Verify connection
  - [ ] Link established
  - [ ] Battery voltage > 3.5V
  - [ ] No firmware errors in console

### 8.2 Parameter Test
- [ ] Check RL parameters exist
  ```python
  cf.param.get_value('rl.enabled')  # Should be 0 (disabled by default)
  ```

- [ ] Enable RL controller
  ```python
  cf.param.set_value('rl.enabled', 1)
  ```

- [ ] Send test setpoint (hover)
  ```python
  cf.commander.send_setpoint(0, 0, 0, 32767)  # roll, pitch, yaw, thrust
  ```

### 8.3 Log Verification
- [ ] Check inference time
  ```python
  infer_time = cf.log.get_value('rl.infer_time_us')
  print(f"Inference time: {infer_time} us")
  ```
  - [ ] Inference time < 5000 us (5ms) ✓ PASS / ✗ FAIL

- [ ] Check failsafe count
  ```python
  failsafe_count = cf.log.get_value('rl.failsafe_count')
  print(f"Failsafes: {failsafe_count}")
  ```
  - [ ] Failsafe count = 0 ✓ PASS / ✗ FAIL

- [ ] Check action outputs
  ```python
  a0 = cf.log.get_value('rl.action0')  # thrust
  a1 = cf.log.get_value('rl.action1')  # roll
  a2 = cf.log.get_value('rl.action2')  # pitch
  a3 = cf.log.get_value('rl.action3')  # yaw
  print(f"Actions: [{a0:.3f}, {a1:.3f}, {a2:.3f}, {a3:.3f}]")
  ```
  - [ ] All actions in range [-1, 1] ✓ PASS / ✗ FAIL
  - [ ] No NaN values ✓ PASS / ✗ FAIL

**If ANY test fails, DO NOT proceed to flight testing. Debug first.**

---

## Phase 9: Tethered Flight ⏳

**⚠️ Now you can attach propellers, but SECURE with tether**

### Setup
- [ ] Attach propellers (check rotation direction)
- [ ] Secure tether (2m vertical, 1m horizontal slack)
- [ ] Clear 3m × 3m × 2m flight space
- [ ] Test emergency stop procedure

### 9.1 Hover Test
- [ ] Enable RL controller
  ```python
  cf.param.set_value('rl.enabled', 1)
  ```

- [ ] Takeoff to 0.5m
  ```python
  cf.commander.send_position_setpoint(0, 0, 0.5, 0)
  ```

- [ ] Observe for 1 minute
  - [ ] Stable hover (±0.2m drift acceptable)
  - [ ] Attitude < 30° ✓ PASS / ✗ FAIL
  - [ ] No oscillations or runaway ✓ PASS / ✗ FAIL

- [ ] Land
  ```python
  cf.commander.send_stop_setpoint()
  ```

### 9.2 Circle Maneuver
- [ ] Hover at 0.5m
- [ ] Execute circle (r=0.5m, 2 laps)
  ```python
  import time
  import math
  
  for t in range(0, 360*2, 10):
    rad = math.radians(t)
    x = 0.5 * math.cos(rad)
    y = 0.5 * math.sin(rad)
    cf.commander.send_position_setpoint(x, y, 0.5, 0)
    time.sleep(0.1)
  ```

- [ ] Check performance
  - [ ] Completes 2 laps ✓ PASS / ✗ FAIL
  - [ ] Trajectory tracking error < 0.3m ✓ PASS / ✗ FAIL

- [ ] Land

**If tethered tests PASS, proceed to free flight. If FAIL, retrain or adjust.**

---

## Phase 10: Free Flight ⚠️

**⚠️ ONLY proceed if tethered tests passed. Ensure safety measures in place.**

### Safety Checklist
- [ ] Kill switch ready (laptop keyboard or remote)
- [ ] Clear flight area (no people/obstacles within 5m)
- [ ] Soft landing surface (foam mat or grass)
- [ ] Battery fully charged (> 3.9V)
- [ ] Propellers secured and undamaged

### 10.1 Free Hover
- [ ] Enable RL controller
- [ ] Takeoff to 0.5m
- [ ] Hover for 30 seconds without intervention
  - [ ] Drift < 0.5m ✓ PASS / ✗ FAIL
  - [ ] No oscillations ✓ PASS / ✗ FAIL
  - [ ] Failsafe count = 0 ✓ PASS / ✗ FAIL

- [ ] Land

### 10.2 Waypoint Navigation
- [ ] Define 4-point square (1m × 1m)
- [ ] Execute waypoint mission
  ```python
  waypoints = [(0, 0, 0.5), (1, 0, 0.5), (1, 1, 0.5), (0, 1, 0.5), (0, 0, 0.5)]
  for (x, y, z) in waypoints:
      cf.commander.send_position_setpoint(x, y, z, 0)
      time.sleep(5)
  ```

- [ ] Check performance
  - [ ] Visits all waypoints ✓ PASS / ✗ FAIL
  - [ ] Tracking error < 0.5m ✓ PASS / ✗ FAIL

- [ ] Land

### 10.3 Battery Life Test
- [ ] Fully charge battery (4.2V)
- [ ] Hover at 0.5m until low battery warning (3.2V)
- [ ] Record flight time
  - [ ] Flight time > 5 minutes ✓ PASS / ✗ FAIL

**CONGRATULATIONS! If all tests pass, deployment is successful! 🎉**

---

## Phase 11: Data Collection (Optional) ⏳

- [ ] Log IMU data during real flights
- [ ] Compare to simulation observations
- [ ] Identify sim-to-real gaps (e.g., motor lag, noise levels)
- [ ] Fine-tune domain randomization parameters
- [ ] Retrain with updated parameters
- [ ] Re-deploy and test

---

## Troubleshooting Log

Use this section to track issues and solutions:

### Issue 1:
- **Problem:** 
- **Error message:** 
- **Root cause:** 
- **Solution:** 
- **Status:** Resolved / Ongoing

### Issue 2:
- **Problem:** 
- **Error message:** 
- **Root cause:** 
- **Solution:** 
- **Status:** Resolved / Ongoing

---

## Notes

Add any observations, parameter tunings, or lessons learned:

---

**Last Updated:** [Your Date]  
**Status:** [Phase X of 11 complete]  
**Next Action:** [What to do next]
