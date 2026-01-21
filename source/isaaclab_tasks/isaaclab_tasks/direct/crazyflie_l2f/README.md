# Crazyflie Learning-to-Fly (L2F) Isaac Lab Integration

This module provides a complete pipeline for training neural network policies in Isaac Lab
that can be deployed to real Crazyflie hardware using the learning-to-fly (L2F) firmware.

## Quick Start

```bash
# 1. Run calibration (validates physics parity with L2F)
./isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/crazyflie_l2f/calibrate.py --num-envs 64 --headless

# 2. Train a policy (calibration runs automatically)
./isaaclab.bat -p source/isaaclab_tasks/isaaclab_tasks/direct/crazyflie_l2f/train_ppo.py --num_envs 4096 --headless

# 3. Export to firmware (done automatically, or manually)
# The actor.h file is automatically created in logs/crazyflie_l2f/

# 4. Build firmware with Docker
cd source/isaaclab_tasks/isaaclab_tasks/direct/crazyflie_l2f/firmware
python build_firmware.py --checkpoint ../logs/crazyflie_l2f/actor.h
```

## Architecture

```
crazyflie_l2f/
├── crazyflie_l2f_env.py    # Main environment (single source of truth)
├── calibrate.py            # Calibration suite for physics validation
├── train_ppo.py            # PPO training with calibration gates
├── networks.py             # Actor-Critic matching L2F firmware
├── export_to_firmware.py   # Export policy to rl_tools format
├── agents/
│   └── rsl_rl_ppo_cfg.py   # RSL-RL PPO configuration
└── firmware/               # Firmware build toolchain
```

## Physics Contract (L2F Parity)

The environment implements the **exact** physics model from learning-to-fly.

### Physical Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Mass | 0.027 kg | crazy_flie.h |
| Arm length | 0.028 m | crazy_flie.h |
| Ixx = Iyy | 3.85e-6 kg·m² | crazy_flie.h |
| Izz | 5.9675e-6 kg·m² | crazy_flie.h |
| Thrust coef (kf) | 3.16e-10 N/RPM² | crazy_flie.h |
| Torque coef (kt/kf) | 0.005964552 | crazy_flie.h |
| Max RPM | 21702 | crazy_flie.h |
| Motor τ | 0.15 s | crazy_flie.h |

### Hover Parameters (Derived)

| Parameter | Value | Formula |
|-----------|-------|---------|
| Hover thrust | 0.265 N | m × g |
| Hover thrust/motor | 0.066 N | m × g / 4 |
| Hover RPM | 10851 | √(F / kf) |
| Hover action | -0.0002 ≈ 0 | 2 × rpm / max_rpm - 1 |

### Action Space

- **Dimension**: 4 (one per motor)
- **Range**: [-1, 1] (normalized)
- **Mapping**: `rpm = (action + 1) / 2 × max_rpm`
- **Motor ordering**: M1(front-right), M2(back-right), M3(back-left), M4(front-left)

### Observation Space (146 dimensions)

| Index | Content | Dim |
|-------|---------|-----|
| 0:3 | Position (x, y, z) in world frame | 3 |
| 3:12 | Rotation matrix (row-major) | 9 |
| 12:15 | Linear velocity in world frame | 3 |
| 15:18 | Angular velocity in BODY frame | 3 |
| 18:146 | Action history (32 × 4) | 128 |

### Motor Dynamics

The motor model implements first-order dynamics:

```
d(rpm)/dt = (target_rpm - rpm) / τ
```

Discrete update (Euler):
```
rpm_new = rpm + α × (target_rpm - rpm)
where α = dt / τ = 0.01 / 0.15 ≈ 0.067
```

### Force/Torque Computation

```
# Thrust per motor
F_i = kf × rpm_i²

# Total thrust (body Z-axis)
F_total = Σ F_i

# Roll torque (about body X)
τ_roll = Σ F_i × y_i

# Pitch torque (about body Y)
τ_pitch = -Σ F_i × x_i

# Yaw torque (reaction)
τ_yaw = (kt/kf) × Σ (dir_i × F_i)
```

## Calibration

The calibration suite validates physics parity before training.

### Tests

1. **Hover Thrust**: Verify total thrust at hover RPM equals body weight (±2%)
2. **Motor Dynamics**: Verify time constant matches L2F (±10%)
3. **Roll Response**: Verify differential thrust produces roll torque
4. **Pitch Response**: Verify differential thrust produces pitch torque
5. **Yaw Response**: Verify reaction torque produces yaw
6. **Altitude Hold**: Verify hover action maintains altitude (±0.1m over 5s)

### Running Calibration

```bash
# Full calibration
./isaaclab.bat -p .../calibrate.py --num-envs 64 --headless

# Specific test
./isaaclab.bat -p .../calibrate.py --test hover --num-envs 64

# View report
cat source/.../crazyflie_l2f/calibration_report.json
```

## Training

Training uses PPO with hyperparameters matching L2F.

### PPO Hyperparameters

| Parameter | Value | L2F Reference |
|-----------|-------|---------------|
| Hidden layers | 64-64 | actor_and_critic.h |
| Activation | tanh | FAST_TANH |
| Learning rate | 3e-4 | ppo_config.h |
| Clip range | 0.2 | CLIP_EPSILON |
| Entropy coef | 0.005 | ENTROPY_COEFFICIENT |
| GAE λ | 0.95 | GAE_LAMBDA |
| γ | 0.99 | GAMMA |
| Epochs per update | 10 | N_EPOCHS |
| Init log std | -0.5 | INITIAL_LOG_STD |

### RTX 4090 Optimization

```python
num_envs = 4096        # Fill VRAM
num_steps_per_env = 32  # Good balance
num_mini_batches = 16   # 8192 samples/batch
```

Expected throughput: ~18,000 steps/second

### Reward Function

Matching L2F `reward_squared_position_only_torque`:

```
weighted_cost = 5.0 × ||pos||² + 5.0 × (1 - qw²) + 0.01 × ||vel||² + 0.01 × ||action||²
reward = -0.5 × weighted_cost + 2.0
```

## Firmware Export

The `export_to_firmware.py` script converts PyTorch weights to rl_tools format.

### Output Format

```cpp
// actor.h
namespace rl_tools::checkpoint::actor {
    static constexpr unsigned int INPUT_DIM = 146;
    static constexpr unsigned int HIDDEN_DIM = 64;
    static constexpr unsigned int OUTPUT_DIM = 4;
    
    static constexpr float input_layer_weights[...] = {...};
    static constexpr float input_layer_biases[...] = {...};
    // ... etc
}
```

### Normalization

The export includes observation normalization statistics:
- `observation::mean[146]`
- `observation::std[146]`

## Troubleshooting

### Episode Length = 1

This indicates immediate termination, usually caused by:
1. Motor state initialized to 0 instead of hover RPM
2. Wrong action interpretation (thrust+moments vs motor RPM)
3. Too strict termination thresholds

**Fix**: Run calibration - it will catch these issues.

### Training Not Converging

1. Check reward scale - should be ~2.0 at hover
2. Verify observation normalization is enabled
3. Check action scale - hover action should be near 0

### Policy Not Hovering on Real Hardware

1. Verify firmware matches training architecture (64-64-tanh)
2. Check observation ordering matches firmware adapter
3. Verify normalization statistics are exported correctly

## References

- [learning-to-fly](https://github.com/arplaboratory/learning-to-fly)
- [rl_tools](https://github.com/rl-tools/rl-tools)
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)
- [Crazyflie](https://www.bitcraze.io/products/crazyflie-2-1/)

## License

BSD-3-Clause (matching Isaac Lab)
