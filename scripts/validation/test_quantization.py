"""
Validate INT8 quantization accuracy
Compares FP32 vs INT8 policy outputs across representative states
"""

import argparse
import numpy as np
import torch
import onnxruntime as ort
from pathlib import Path


def generate_test_states(n_samples: int = 1000) -> np.ndarray:
    """Generate representative test states covering observation space
    
    Returns:
        Array of shape (n_samples, 9) with realistic IMU observations
    """
    states = []
    
    # Hovering states (±10° tilt, low velocities)
    for _ in range(n_samples // 3):
        lin_acc = np.random.randn(3) * 2.0  # ±2 m/s² noise
        lin_acc[2] += 9.81  # Gravity
        ang_vel = np.random.randn(3) * 0.5  # ±0.5 rad/s
        euler = np.random.randn(3) * 0.175  # ±10°
        states.append(np.concatenate([lin_acc, ang_vel, euler]))
    
    # Aggressive maneuvers (±45° tilt, high velocities)
    for _ in range(n_samples // 3):
        lin_acc = np.random.randn(3) * 10.0
        lin_acc[2] += 9.81
        ang_vel = np.random.randn(3) * 3.0  # ±3 rad/s
        euler = np.random.randn(3) * 0.785  # ±45°
        states.append(np.concatenate([lin_acc, ang_vel, euler]))
    
    # Edge cases (sensor limits)
    for _ in range(n_samples // 3):
        lin_acc = np.random.uniform(-20, 20, 3)
        lin_acc[2] += 9.81
        ang_vel = np.random.uniform(-5, 5, 3)
        euler = np.random.uniform(-np.pi, np.pi, 3)
        states.append(np.concatenate([lin_acc, ang_vel, euler]))
    
    return np.array(states, dtype=np.float32)


def load_onnx_model(path: str) -> ort.InferenceSession:
    """Load ONNX model"""
    return ort.InferenceSession(path, providers=['CPUExecutionProvider'])


def compute_accuracy_metrics(fp32_actions: np.ndarray, int8_actions: np.ndarray) -> dict:
    """Compute quantization accuracy metrics
    
    Returns:
        Dictionary with MAE, MSE, max error, correlation per action
    """
    mae = np.mean(np.abs(fp32_actions - int8_actions), axis=0)
    mse = np.mean((fp32_actions - int8_actions) ** 2, axis=0)
    max_err = np.max(np.abs(fp32_actions - int8_actions), axis=0)
    
    # Pearson correlation per action
    corr = []
    for i in range(fp32_actions.shape[1]):
        corr.append(np.corrcoef(fp32_actions[:, i], int8_actions[:, i])[0, 1])
    
    return {
        'mae': mae,
        'mse': mse,
        'max_error': max_err,
        'correlation': np.array(corr)
    }


def validate_quantization(fp32_path: str, int8_path: str, n_samples: int = 1000):
    """Main validation routine"""
    print("=" * 80)
    print("INT8 Quantization Validation")
    print("=" * 80)
    
    # Load models
    print(f"\nLoading FP32 model: {fp32_path}")
    fp32_sess = load_onnx_model(fp32_path)
    
    print(f"Loading INT8 model: {int8_path}")
    int8_sess = load_onnx_model(int8_path)
    
    # Generate test data
    print(f"\nGenerating {n_samples} test states...")
    test_states = generate_test_states(n_samples)
    
    # Run inference
    print("Running FP32 inference...")
    fp32_actions = fp32_sess.run(None, {'obs': test_states})[0]
    
    print("Running INT8 inference...")
    int8_actions = int8_sess.run(None, {'obs': test_states})[0]
    
    # Compute metrics
    metrics = compute_accuracy_metrics(fp32_actions, int8_actions)
    
    # Report
    action_names = ['thrust', 'roll_moment', 'pitch_moment', 'yaw_moment']
    print("\n" + "=" * 80)
    print("Quantization Accuracy Report")
    print("=" * 80)
    
    for i, name in enumerate(action_names):
        print(f"\n{name.upper()}:")
        print(f"  MAE:         {metrics['mae'][i]:.6f}")
        print(f"  MSE:         {metrics['mse'][i]:.6f}")
        print(f"  Max Error:   {metrics['max_error'][i]:.6f}")
        print(f"  Correlation: {metrics['correlation'][i]:.6f}")
    
    # Overall statistics
    print("\n" + "=" * 80)
    print("Overall Statistics")
    print("=" * 80)
    print(f"Mean MAE:         {np.mean(metrics['mae']):.6f}")
    print(f"Mean Correlation: {np.mean(metrics['correlation']):.6f}")
    
    # Pass/fail criteria
    print("\n" + "=" * 80)
    print("Validation Criteria")
    print("=" * 80)
    
    mae_threshold = 0.05  # 5% of action range [-1, 1]
    corr_threshold = 0.98
    
    mae_pass = np.all(metrics['mae'] < mae_threshold)
    corr_pass = np.all(metrics['correlation'] > corr_threshold)
    
    print(f"MAE < {mae_threshold}: {'PASS' if mae_pass else 'FAIL'}")
    print(f"Correlation > {corr_threshold}: {'PASS' if corr_pass else 'FAIL'}")
    
    if mae_pass and corr_pass:
        print("\n✓ INT8 quantization PASSED all criteria")
        return 0
    else:
        print("\n✗ INT8 quantization FAILED - consider more calibration data or QAT")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate INT8 quantization accuracy")
    parser.add_argument("--fp32", type=str, required=True, help="Path to FP32 ONNX model")
    parser.add_argument("--int8", type=str, required=True, help="Path to INT8 ONNX model")
    parser.add_argument("--samples", type=int, default=1000, help="Number of test samples")
    
    args = parser.parse_args()
    
    exit_code = validate_quantization(args.fp32, args.int8, args.samples)
    exit(exit_code)
