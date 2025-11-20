# ONNX Model Export and STM32 Deployment Guide

This guide explains how to export trained policies from Isaac Lab and deploy them on STM32 microcontrollers using INT8 quantization.

## Workflow Overview

```
Train Policy → Export ONNX (FP32 & INT8) → Convert to C Code → Deploy on STM32
```

## Step 1: Train Your Policy

Train your policy using the standard Isaac Lab workflow:

```bash
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Flat-Anymal-D-v0
```

## Step 2: Export ONNX Models

When you run the play script, it will automatically export 3 model formats:

```bash
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Velocity-Flat-Anymal-D-v0 --checkpoint=path/to/checkpoint.pt
```

This generates in the `exported/` folder:
- `policy.pt` - PyTorch JIT model (FP32)
- `policy.onnx` - ONNX model (FP32, ~4 bytes per parameter)
- **`policy_int8.onnx`** - Quantized ONNX model (INT8, ~1 byte per parameter)

### File Size Comparison Example:
- FP32 model: 100 KB
- **INT8 model: ~25 KB** (75% reduction!)

## Step 3: Install Required Dependencies

```bash
pip install onnxruntime onnx
```

## Step 4: Convert ONNX to C Code

### Option A: INT8 Quantized (Recommended for STM32)

```bash
python tools/onnx_to_c_converter.py \
    --onnx logs/rsl_rl/your_task/exported/policy_int8.onnx \
    --output stm32_deployment/policy_int8.c \
    --dtype int8 \
    --namespace robot_policy
```

### Option B: FP32 (If you have memory to spare)

```bash
python tools/onnx_to_c_converter.py \
    --onnx logs/rsl_rl/your_task/exported/policy.onnx \
    --output stm32_deployment/policy_fp32.c \
    --dtype float32 \
    --namespace robot_policy
```

## Step 5: Generated Files

The converter creates:
- `robot_policy.h` - Header file with function declarations
- `robot_policy.c` - Implementation with weights and inference code

## Step 6: Using in Your STM32 Project

### Include in your STM32 code:

```c
#include "robot_policy.h"

// Initialize once
void setup() {
    robot_policy_init();
}

// Run inference
void control_loop() {
    float observations[ROBOT_POLICY_INPUT_DIM];
    float actions[ROBOT_POLICY_OUTPUT_DIM];
    
    // Fill observations from your sensors
    observations[0] = read_joint_position(0);
    observations[1] = read_joint_velocity(0);
    // ... etc
    
    // Run inference
    robot_policy_infer(observations, actions);
    
    // Apply actions to actuators
    set_motor_command(0, actions[0]);
    set_motor_command(1, actions[1]);
    // ... etc
}
```

## Memory Requirements

### INT8 Quantized Model:
- **RAM**: ~2-3x parameter count (for intermediate buffers)
- **Flash**: ~1 byte per parameter
- **Example**: 50K parameter model ≈ 50KB flash + 150KB RAM

### FP32 Model:
- **RAM**: ~8-12x parameter count
- **Flash**: ~4 bytes per parameter
- **Example**: 50K parameter model ≈ 200KB flash + 600KB RAM

## STM32 Board Recommendations

### Minimum (INT8 models):
- **STM32F4** series (168 MHz, 192 KB RAM, 1 MB Flash)
- Example: STM32F429 Discovery

### Recommended (Best performance):
- **STM32H7** series (480 MHz, 1 MB RAM, 2 MB Flash)
- Example: STM32H743 Nucleo
- Has hardware FPU and better memory

### For large models:
- **STM32MP1** (Cortex-A7 + M4, up to 8 GB RAM)
- Can run Linux + real-time control

## Optimization Tips

1. **Use INT8 quantization** - 75% smaller, faster inference
2. **Reduce observation dimensions** - Remove unnecessary inputs
3. **Smaller network architecture** - Train with fewer hidden units
4. **Fixed-point math** - Further optimize INT8 operations (advanced)
5. **Compile with -O2 or -O3** - Enable compiler optimizations

## Troubleshooting

### "INT8 export skipped: No module named 'onnxruntime'"
```bash
pip install onnxruntime
```

### "Model too large for STM32"
- Train a smaller network (reduce hidden layer sizes)
- Use INT8 quantization
- Reduce observation space dimensions

### "Inference too slow"
- Use INT8 model
- Increase STM32 clock speed
- Use STM32H7 series with hardware acceleration
- Consider ARM CMSIS-NN library for optimized operations

## Advanced: CMSIS-NN Integration

For maximum performance on ARM Cortex-M, consider using ARM's CMSIS-NN library:

```bash
# The converter output is compatible with CMSIS-NN
# You can replace basic operations with optimized CMSIS-NN functions:
# - arm_fully_connected_q7()
# - arm_relu_q7()
# - arm_sigmoid_q7()
```

## Example: Complete Workflow

```bash
# 1. Train
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Flat-Anymal-D-v0

# 2. Export (automatically creates policy_int8.onnx)
.\isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Velocity-Flat-Anymal-D-v0 --checkpoint=logs/rsl_rl/anymal_d_flat/model_5000.pt

# 3. Convert to C
python tools/onnx_to_c_converter.py \
    --onnx logs/rsl_rl/anymal_d_flat/exported/policy_int8.onnx \
    --output stm32_code/policy.c \
    --dtype int8

# 4. Copy to STM32 project
# - Copy policy.h and policy.c to your STM32 project
# - Include in your main control loop
# - Compile and flash to board
```

## Performance Expectations

### Inference Time (approximate):
- **STM32F4 (168 MHz)**: 
  - Small model (10K params): ~5-10 ms
  - Medium model (50K params): ~20-50 ms
  
- **STM32H7 (480 MHz)**:
  - Small model (10K params): ~1-2 ms
  - Medium model (50K params): ~5-15 ms

### Control Loop Frequencies:
- 100 Hz (10 ms): Achievable with medium models on H7
- 500 Hz (2 ms): Achievable with small models on H7
- 1 kHz (1 ms): Requires very small models and optimization

## Additional Resources

- [STM32 AI Documentation](https://www.st.com/en/embedded-software/x-cube-ai.html)
- [ARM CMSIS-NN](https://github.com/ARM-software/CMSIS-NN)
- [ONNX Runtime Quantization](https://onnxruntime.ai/docs/performance/quantization.html)

## Support

For issues with:
- Policy training: See Isaac Lab documentation
- ONNX export: Check `source/isaaclab_rl/isaaclab_rl/rsl_rl/exporter.py`
- C code generation: Check `tools/onnx_to_c_converter.py`
- STM32 integration: Refer to STM32CubeMX and your board documentation
