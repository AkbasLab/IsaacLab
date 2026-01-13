# Crazyflie RL Controller - Quick Reference

## 🚀 Quick Start (Full Pipeline)

```bash
# 1. Train policy
python scripts/reinforcement_learning/train.py --task=Isaac-Quadcopter-Direct-v0 --num_envs=4096 --max_iterations=2000

# 2. Validate in simulation
python scripts/validation/test_sim_policy.py --checkpoint logs/rsl_rl/quadcopter/model_2000.pt --episodes 100

# 3. Export and quantize
python scripts/export_policy_int8.py --checkpoint logs/rsl_rl/quadcopter/model_2000.pt --output policy_fp32.onnx --quantize --calibration_samples 1000

# 4. Validate quantization
python scripts/validation/test_quantization.py --fp32 policy_fp32.onnx --int8 policy_int8.onnx

# 5. Generate C code
python tools/onnx_to_cmsis.py --input policy_int8.onnx --output crazyflie_deploy/

# 6. Build firmware (in crazyflie-firmware directory)
make clean && make -j8

# 7. Flash to Crazyflie
make cload  # via radio, or
make flash  # via USB bootloader
```

---

## 📊 Key Specifications

| Parameter | Value |
|-----------|-------|
| **Policy Architecture** | 32×32×2 MLP (Tanh) |
| **Parameters** | 1,508 |
| **Observation Space** | 9D (lin_acc[3], ang_vel[3], euler[3]) |
| **Action Space** | 4D (thrust, roll, pitch, yaw moments) |
| **Control Frequency** | 100 Hz |
| **Inference Time** | ~2.3ms (STM32F405 @ 168MHz) |
| **Memory Footprint** | ~6KB code + ~1.5KB weights |
| **Quantization** | INT8 (Post-Training Quantization) |

---

## 🎛️ Runtime Parameters

Enable/disable RL controller at runtime (via cfclient or Python):

```python
# Enable RL controller
cf.param.set_value('rl.enabled', 1)

# Disable (revert to PID)
cf.param.set_value('rl.enabled', 0)
```

---

## 📈 Log Variables

Monitor performance in real-time:

```python
# Inference time (microseconds)
cf.log.get_value('rl.infer_time_us')

# Action outputs [-1, 1]
cf.log.get_value('rl.action0')  # thrust
cf.log.get_value('rl.action1')  # roll moment
cf.log.get_value('rl.action2')  # pitch moment
cf.log.get_value('rl.action3')  # yaw moment

# Failsafe count
cf.log.get_value('rl.failsafe_count')

# Total inferences
cf.log.get_value('rl.infer_count')
```

---

## 🔍 Validation Gates

| Stage | Pass Criteria |
|-------|---------------|
| **Sim Policy** | Mean return > 50, Success > 50% |
| **Quantization** | MAE < 0.05, Correlation > 0.98 |
| **Bench Test** | Inference < 5ms, Failsafes = 0 |
| **Tethered Test** | Stable hover, Attitude < 30° |
| **Free Flight** | 30s hover without intervention |

---

## ⚠️ Safety Limits (Hardcoded)

| Limit | Value |
|-------|-------|
| Max Roll/Pitch | 45° |
| Max Yaw Rate | 200°/s |
| Min Thrust PWM | 10,000 |
| Max Thrust PWM | 60,000 |
| Max Action Change | 0.3/step |
| Altitude Range | 0.05m - 2.0m |

Violations trigger automatic PID fallback.

---

## 🐛 Common Issues

### Inference time > 5ms
- ✅ Verify CMSIS-NN linked: `arm-none-eabi-nm cf2.elf | grep arm_fully_connected`
- ✅ Check FPU enabled: `CFLAGS` must include `-mfpu=fpv4-sp-d16 -mfloat-abi=hard`

### High failsafe count
- ✅ Re-run quantization validation
- ✅ Increase calibration samples to 5000
- ✅ Check for NaN in policy outputs

### Unstable flight
- ✅ Verify domain randomization was enabled during training
- ✅ Train longer (4000 iterations)
- ✅ Collect real IMU data and add to calibration set

### Build fails
- ✅ Ensure ARM toolchain installed: `arm-none-eabi-gcc --version`
- ✅ CMSIS-NN cloned: `ls crazyflie-firmware/vendor/CMSIS-NN`
- ✅ Makefile updated with controller files

---

## 📁 File Structure

```
IsaacLab/
├── scripts/
│   ├── reinforcement_learning/train.py      # Training script
│   ├── export_policy_int8.py                # Export + quantization
│   └── validation/
│       ├── test_sim_policy.py               # FP32 sim validation
│       └── test_quantization.py             # INT8 accuracy test
├── tools/
│   └── onnx_to_cmsis.py                     # ONNX → C converter
├── crazyflie_deploy/
│   ├── controller_rl.{c,h}                  # Controller logic
│   ├── policy_int8.{c,h}                    # Generated inference code
│   └── policy_int8_weights.c                # Generated weights
├── source/isaaclab_tasks/direct/quadcopter/
│   ├── quadcopter_env.py                    # Simulation environment
│   └── agents/rsl_rl_ppo_cfg.py             # Training config
└── DEPLOYMENT_GUIDE.md                      # Full documentation
```

---

## 🔗 External Dependencies

- **IsaacLab**: Training simulation (already installed)
- **Crazyflie Firmware**: https://github.com/bitcraze/crazyflie-firmware
- **CMSIS-NN**: https://github.com/ARM-software/CMSIS-NN
- **ARM GCC**: https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain

---

## 📞 Emergency Stop

**During flight, if controller becomes unstable:**

```python
# Method 1: Disable RL controller (PID takes over)
cf.param.set_value('rl.enabled', 0)

# Method 2: Kill motors immediately
cf.commander.send_stop_setpoint()

# Method 3: Physical kill switch (hold power button)
```

---

## 📊 Expected Training Curves

| Iteration | Mean Return | Success Rate | Notes |
|-----------|-------------|--------------|-------|
| 0 | -50 | 0% | Random policy |
| 500 | 0 | 10% | Learning to hover |
| 1000 | 30 | 40% | Basic navigation |
| 1500 | 50 | 60% | Reliable navigation |
| 2000 | 70 | 80% | Near-optimal |

If progress stalls, check:
- Reward scale tuning (try 2× or 0.5× current values)
- Domain randomization (may be too aggressive)
- Observation normalization (should be disabled for deployment)

---

## 🧪 Test Sequence

1. **Bench Test** (no props): 5 min, log inference time
2. **Tethered Hover**: 1 min @ 0.5m
3. **Tethered Circle**: r=0.5m, 2 laps
4. **Free Hover**: 30s @ 0.5m
5. **Free Navigation**: 4-point waypoints
6. **Aggressive Maneuvers**: Figure-8, flips (optional)

---

## 🎓 Learning Resources

- **IsaacLab Docs**: https://isaac-sim.github.io/IsaacLab
- **RSL-RL**: https://github.com/leggedrobotics/rsl_rl
- **Crazyflie Docs**: https://www.bitcraze.io/documentation
- **CMSIS-NN Guide**: https://arxiv.org/abs/1801.06601
- **INT8 Quantization**: https://onnxruntime.ai/docs/performance/quantization.html

---

**Last Updated:** 2025-01-XX  
**Tested On:** IsaacLab v1.2+, Crazyflie Firmware v2024.11+
