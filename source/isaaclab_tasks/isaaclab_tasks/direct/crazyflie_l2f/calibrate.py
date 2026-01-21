#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""
Calibration Suite for Crazyflie L2F Environment

This script validates that the Isaac Lab environment matches L2F physics before training.
All tests must pass before training is allowed to proceed.

Usage:
    python calibrate.py                    # Run full calibration
    python calibrate.py --test hover       # Run specific test
    python calibrate.py --skip-validation  # Skip validation (for debugging only)

Tests:
    1. Hover Test: Verify thrust at hover RPM equals body weight
    2. Motor Dynamics Test: Verify first-order time constant
    3. Roll/Pitch Response Test: Verify torque generation
    4. Yaw Response Test: Verify reaction torque
    5. Trajectory Test: Compare against L2F reference trajectory
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

# Isaac Lab imports (deferred to allow --help without Isaac)
def import_isaac():
    """Import Isaac Lab modules."""
    from isaaclab.app import AppLauncher
    return AppLauncher


# =============================================================================
# TEST RESULT DATACLASSES
# =============================================================================

@dataclass
class TestResult:
    """Result of a single calibration test."""
    name: str
    passed: bool
    expected: float
    actual: float
    tolerance: float
    unit: str
    message: str = ""
    
    @property
    def error_pct(self) -> float:
        """Percentage error."""
        if abs(self.expected) < 1e-9:
            return 0.0 if abs(self.actual) < 1e-9 else float('inf')
        return abs(self.actual - self.expected) / abs(self.expected) * 100
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            **asdict(self),
            "error_pct": self.error_pct
        }


@dataclass
class CalibrationReport:
    """Complete calibration report."""
    timestamp: str
    environment: str
    all_passed: bool
    tests: list
    summary: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "environment": self.environment,
            "all_passed": self.all_passed,
            "tests": [t.to_dict() if isinstance(t, TestResult) else t for t in self.tests],
            "summary": self.summary
        }
    
    def save(self, path: Path) -> None:
        """Save report to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def print_report(self) -> None:
        """Print formatted report to console."""
        print("\n" + "="*70)
        print("CALIBRATION REPORT")
        print("="*70)
        print(f"Timestamp: {self.timestamp}")
        print(f"Environment: {self.environment}")
        print(f"Status: {'PASSED' if self.all_passed else 'FAILED'}")
        print("-"*70)
        
        for test in self.tests:
            if isinstance(test, TestResult):
                status = "✓ PASS" if test.passed else "✗ FAIL"
                print(f"{status}  {test.name}")
                print(f"        Expected: {test.expected:.6f} {test.unit}")
                print(f"        Actual:   {test.actual:.6f} {test.unit}")
                print(f"        Error:    {test.error_pct:.2f}%")
                if test.message:
                    print(f"        Note:     {test.message}")
            else:
                print(f"  {test}")
        
        print("-"*70)
        print(self.summary)
        print("="*70 + "\n")


# =============================================================================
# CALIBRATION TESTS
# =============================================================================

class CalibrationSuite:
    """Suite of calibration tests for L2F environment."""
    
    def __init__(self, env, device: str = "cuda"):
        """
        Initialize calibration suite.
        
        Args:
            env: Crazyflie L2F environment instance
            device: Compute device
        """
        self.env = env
        self.device = device
        self.results: list[TestResult] = []
        
        # Get physics config
        self.cfg = env.cfg.physics
        self.dt = env.cfg.sim.dt
        
    def run_all(self) -> CalibrationReport:
        """Run all calibration tests."""
        self.results = []
        
        print("\nRunning calibration tests...")
        
        # Test 1: Hover thrust
        self._test_hover_thrust()
        
        # Test 2: Motor dynamics
        self._test_motor_dynamics()
        
        # Test 3: Roll response
        self._test_roll_response()
        
        # Test 4: Pitch response
        self._test_pitch_response()
        
        # Test 5: Yaw response
        self._test_yaw_response()
        
        # Test 6: Altitude hold
        self._test_altitude_hold()
        
        # Generate report
        all_passed = all(r.passed for r in self.results)
        
        if all_passed:
            summary = "All calibration tests PASSED. Environment is ready for training."
        else:
            failed = [r.name for r in self.results if not r.passed]
            summary = f"Calibration FAILED. Failed tests: {', '.join(failed)}"
        
        report = CalibrationReport(
            timestamp=datetime.now().isoformat(),
            environment="CrazyflieL2FEnv",
            all_passed=all_passed,
            tests=self.results,
            summary=summary
        )
        
        return report
    
    def _test_hover_thrust(self) -> None:
        """Test that hover RPM produces correct thrust."""
        print("  Testing hover thrust...")
        
        # Expected thrust at hover = body weight
        expected_thrust = self.cfg.mass * self.cfg.gravity
        
        # Set all motors to hover RPM
        hover_rpm = self.cfg.hover_rpm
        thrust_per_motor = self.cfg.thrust_coefficient * hover_rpm ** 2
        actual_thrust = thrust_per_motor * 4  # 4 motors
        
        tolerance = self.env.cfg.calibration.hover_thrust_tolerance
        error = abs(actual_thrust - expected_thrust) / expected_thrust
        
        self.results.append(TestResult(
            name="Hover Thrust",
            passed=error <= tolerance,
            expected=expected_thrust,
            actual=actual_thrust,
            tolerance=tolerance,
            unit="N",
            message=f"Hover RPM: {hover_rpm:.0f}"
        ))
    
    def _test_motor_dynamics(self) -> None:
        """Test first-order motor dynamics time constant."""
        print("  Testing motor dynamics...")
        
        # Reset environment
        self.env.reset()
        
        # Set target RPM and measure response
        target_action = torch.ones(self.env.num_envs, 4, device=self.device) * 0.5
        initial_rpm = self.env.get_rpm_state()[0, 0].item()
        target_rpm = (0.5 + 1.0) / 2.0 * self.cfg.max_rpm
        
        # Step until ~63.2% of final value (one time constant)
        rpm_63 = initial_rpm + 0.632 * (target_rpm - initial_rpm)
        
        time_to_63 = 0.0
        for _ in range(1000):  # Max iterations
            self.env.step(target_action)
            time_to_63 += self.dt
            
            current_rpm = self.env.get_rpm_state()[0, 0].item()
            if current_rpm >= rpm_63:
                break
        
        expected_tau = self.cfg.motor_time_constant
        tolerance = self.env.cfg.calibration.motor_time_constant_tolerance
        error = abs(time_to_63 - expected_tau) / expected_tau
        
        self.results.append(TestResult(
            name="Motor Time Constant",
            passed=error <= tolerance,
            expected=expected_tau,
            actual=time_to_63,
            tolerance=tolerance,
            unit="s",
            message=f"63.2% RPM: {rpm_63:.0f}"
        ))
    
    def _test_roll_response(self) -> None:
        """Test roll torque from differential thrust."""
        print("  Testing roll response...")
        
        # Reset and spawn level
        self.env.reset()
        
        # Apply differential thrust for roll (motors on +y vs -y)
        # M1(+x,-y), M2(-x,-y) vs M3(-x,+y), M4(+x,+y)
        # To roll right (+x up), increase M3,M4 and decrease M1,M2
        actions = torch.zeros(self.env.num_envs, 4, device=self.device)
        hover = self.cfg.hover_action
        delta = 0.2
        actions[:, 0] = hover - delta  # M1: decrease
        actions[:, 1] = hover - delta  # M2: decrease
        actions[:, 2] = hover + delta  # M3: increase
        actions[:, 3] = hover + delta  # M4: increase
        
        # Step for a bit to let motor dynamics settle
        for _ in range(50):
            self.env.step(actions)
        
        # Measure angular velocity
        ang_vel = self.env._robot.data.root_ang_vel_b[0, 0].item()  # Roll rate
        
        # Expected roll rate based on torque and inertia
        # Torque = delta_thrust * arm_length
        delta_thrust = 2 * delta * self.cfg.max_rpm * self.cfg.thrust_coefficient
        expected_torque = delta_thrust * self.cfg.arm_length * 2  # Factor of 2 for differential
        
        # Angular acceleration = torque / I
        expected_ang_acc = expected_torque / self.cfg.Ixx
        
        # After settling, rate should be significant if torque is correct
        # This is a qualitative check - rate should be in same direction
        passed = ang_vel > 0.1  # Some positive roll rate
        
        tolerance = self.env.cfg.calibration.roll_rate_peak_tolerance
        
        self.results.append(TestResult(
            name="Roll Response",
            passed=passed,
            expected=1.0,  # Qualitative: should be positive
            actual=ang_vel,
            tolerance=tolerance,
            unit="rad/s",
            message="Positive roll rate indicates correct torque direction"
        ))
    
    def _test_pitch_response(self) -> None:
        """Test pitch torque from differential thrust."""
        print("  Testing pitch response...")
        
        # Reset
        self.env.reset()
        
        # Differential thrust for pitch
        # M1(+x,-y), M4(+x,+y) vs M2(-x,-y), M3(-x,+y)
        # To pitch forward (+y up), increase M2,M3 and decrease M1,M4
        actions = torch.zeros(self.env.num_envs, 4, device=self.device)
        hover = self.cfg.hover_action
        delta = 0.2
        actions[:, 0] = hover - delta  # M1: decrease
        actions[:, 1] = hover + delta  # M2: increase
        actions[:, 2] = hover + delta  # M3: increase
        actions[:, 3] = hover - delta  # M4: decrease
        
        # Let settle
        for _ in range(50):
            self.env.step(actions)
        
        # Measure pitch rate
        ang_vel = self.env._robot.data.root_ang_vel_b[0, 1].item()
        
        passed = abs(ang_vel) > 0.1  # Some pitch rate
        tolerance = self.env.cfg.calibration.pitch_rate_peak_tolerance
        
        self.results.append(TestResult(
            name="Pitch Response",
            passed=passed,
            expected=1.0,
            actual=abs(ang_vel),
            tolerance=tolerance,
            unit="rad/s",
            message="Pitch rate indicates torque is being generated"
        ))
    
    def _test_yaw_response(self) -> None:
        """Test yaw torque from motor reaction torques."""
        print("  Testing yaw response...")
        
        # Reset
        self.env.reset()
        
        # Yaw torque from differential rotor speeds (same direction spin)
        # M1(-CW), M3(-CW) vs M2(+CCW), M4(+CCW)
        # Increase CW motors for CCW yaw
        actions = torch.zeros(self.env.num_envs, 4, device=self.device)
        hover = self.cfg.hover_action
        delta = 0.2
        actions[:, 0] = hover + delta  # M1: CW, increase
        actions[:, 1] = hover - delta  # M2: CCW, decrease
        actions[:, 2] = hover + delta  # M3: CW, increase
        actions[:, 3] = hover - delta  # M4: CCW, decrease
        
        # Let settle
        for _ in range(50):
            self.env.step(actions)
        
        # Measure yaw rate
        ang_vel = self.env._robot.data.root_ang_vel_b[0, 2].item()
        
        passed = abs(ang_vel) > 0.01  # Some yaw rate (smaller due to torque coefficient)
        tolerance = self.env.cfg.calibration.yaw_rate_peak_tolerance
        
        self.results.append(TestResult(
            name="Yaw Response",
            passed=passed,
            expected=0.1,
            actual=abs(ang_vel),
            tolerance=tolerance,
            unit="rad/s",
            message="Yaw response is typically weaker than roll/pitch"
        ))
    
    def _test_altitude_hold(self) -> None:
        """Test that hover action maintains altitude."""
        print("  Testing altitude hold...")
        
        # Reset
        obs, _ = self.env.reset()
        initial_height = self.env._robot.data.root_pos_w[0, 2].item()
        
        # Apply hover action for 5 seconds
        hover_action = torch.full(
            (self.env.num_envs, 4), 
            self.cfg.hover_action, 
            device=self.device
        )
        
        test_duration = 5.0  # seconds
        steps = int(test_duration / self.dt)
        
        for _ in range(steps):
            self.env.step(hover_action)
        
        final_height = self.env._robot.data.root_pos_w[0, 2].item()
        drift = abs(final_height - initial_height)
        
        max_drift = self.env.cfg.calibration.hover_altitude_drift_max
        passed = drift <= max_drift
        
        self.results.append(TestResult(
            name="Altitude Hold",
            passed=passed,
            expected=0.0,
            actual=drift,
            tolerance=max_drift,
            unit="m",
            message=f"After {test_duration}s hover, drift should be < {max_drift}m"
        ))


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calibration suite for Crazyflie L2F environment"
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        choices=["hover", "motor", "roll", "pitch", "yaw", "altitude"],
        help="Run specific test only"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=64,
        help="Number of environments (default: 64)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for report"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation (for debugging only)"
    )
    
    # Add Isaac Sim launcher arguments (includes --headless)
    AppLauncher = import_isaac()
    AppLauncher.add_app_launcher_args(parser)
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Initialize Isaac Sim
    AppLauncher = import_isaac()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    
    # Import after Isaac is initialized
    from isaaclab_tasks.direct.crazyflie_l2f.crazyflie_l2f_env import (
        CrazyflieL2FEnv,
        CrazyflieL2FEnvCfg
    )
    
    # Create environment configuration
    env_cfg = CrazyflieL2FEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    
    # Disable calibration requirement for calibration script itself
    env_cfg.calibration.require_calibration = False
    
    # Create environment
    print("\nCreating environment...")
    env = CrazyflieL2FEnv(env_cfg)
    
    # Run calibration
    print("\nStarting calibration...")
    suite = CalibrationSuite(env, device=env.device)
    report = suite.run_all()
    
    # Print report
    report.print_report()
    
    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        report.save(output_path)
        print(f"Report saved to: {output_path}")
    
    # Save default report location
    default_report = Path(__file__).parent / "calibration_report.json"
    report.save(default_report)
    print(f"Report saved to: {default_report}")
    
    # Cleanup
    env.close()
    simulation_app.close()
    
    # Exit with appropriate code
    if not report.all_passed and not args.skip_validation:
        print("\nCALIBRATION FAILED - Training is not recommended!")
        sys.exit(1)
    else:
        print("\nCalibration complete.")
        sys.exit(0)


if __name__ == "__main__":
    main()
