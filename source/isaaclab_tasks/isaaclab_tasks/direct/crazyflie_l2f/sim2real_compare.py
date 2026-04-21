#!/usr/bin/env python3
"""
Sim-to-Real Comparison Tool

Loads a sim CSV (from play_eval.py) and a real CSV (from WebUI flight_data_*.csv),
time-aligns them, and computes per-channel comparison metrics.

Both CSVs now share the same column schema (SHARED_CSV_FIELDS).

Metrics computed per signal:
  - RMSE          (root mean squared error)
  - MAE           (mean absolute error)
  - Max error     (worst-case absolute difference)
  - Pearson r     (correlation coefficient)
  - Mean (sim/real) & std (sim/real)

Outputs:
  - Console summary table
  - Per-signal overlay plots (sim vs real)
  - Summary CSV with all metrics

Usage:
  python sim2real_compare.py --sim hover_eval_data.csv --real flight_data_1234567890.csv
  python sim2real_compare.py --sim hover_eval_data.csv --real flight_data_1234567890.csv --output_dir compare_results
  python sim2real_compare.py --sim hover_eval_data.csv --real flight_data_1234567890.csv --align_method dtw
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── Shared column schema (must match play_eval.py and ui.html) ──────────────
# Signals suitable for quantitative comparison (excludes timestamps, NaN-heavy cols)
COMPARE_SIGNALS = [
    # Position
    ("stateEstimate.x", "Position X", "m"),
    ("stateEstimate.y", "Position Y", "m"),
    ("stateEstimate.z", "Position Z", "m"),
    # Attitude
    ("stabilizer.roll",  "Roll",  "deg"),
    ("stabilizer.pitch", "Pitch", "deg"),
    ("stabilizer.yaw",   "Yaw",   "deg"),
    # Accelerometer (Crazyflie firmware reports in g; sim now matches)
    ("acc.x", "Accel X", "g"),
    ("acc.y", "Accel Y", "g"),
    ("acc.z", "Accel Z", "g"),
    # Gyroscope
    ("gyro.x", "Gyro X", "deg/s"),
    ("gyro.y", "Gyro Y", "deg/s"),
    ("gyro.z", "Gyro Z", "deg/s"),
    # Velocity (may be NaN on real if firmware doesn't expose stateEstimate.vx/vy/vz)
    ("velocity.x", "Velocity X", "m/s"),
    ("velocity.y", "Velocity Y", "m/s"),
    ("velocity.z", "Velocity Z", "m/s"),
    # Motors (PWM 0-65535 on both sim and real)
    ("motor.m1", "Motor M1", "PWM"),
    ("motor.m2", "Motor M2", "PWM"),
    ("motor.m3", "Motor M3", "PWM"),
    ("motor.m4", "Motor M4", "PWM"),
    # Motor RPMs where available
    ("motor.rpm.m1", "Motor RPM M1", "rpm"),
    ("motor.rpm.m2", "Motor RPM M2", "rpm"),
    ("motor.rpm.m3", "Motor RPM M3", "rpm"),
    ("motor.rpm.m4", "Motor RPM M4", "rpm"),
    # IMU-style duplicate channels where available
    ("imu.acc_x", "IMU Accel X", "g"),
    ("imu.acc_y", "IMU Accel Y", "g"),
    ("imu.acc_z", "IMU Accel Z", "g"),
    ("imu.gyro_x", "IMU Gyro X", "deg/s"),
    ("imu.gyro_y", "IMU Gyro Y", "deg/s"),
    ("imu.gyro_z", "IMU Gyro Z", "deg/s"),
]

# Signals for tracking-error analysis (need target columns present)
TRACKING_SIGNALS = [
    ("stateEstimate.x", "target.x", "Tracking X", "m"),
    ("stateEstimate.y", "target.y", "Tracking Y", "m"),
    ("stateEstimate.z", "target.z", "Tracking Z", "m"),
]


# ── CSV loading ─────────────────────────────────────────────────────────────
def load_csv(path: str) -> Dict[str, np.ndarray]:
    """Load CSV into {column_name: np.ndarray}. Non-numeric values become NaN."""
    data: Dict[str, list] = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if key not in data:
                    data[key] = []
                try:
                    data[key].append(float(value))
                except (ValueError, TypeError):
                    data[key].append(float("nan"))
    return {k: np.array(v) for k, v in data.items()}


def _copy_if_present(dst: Dict[str, np.ndarray], src: Dict[str, np.ndarray], src_key: str, dst_key: str):
    if src_key in src and dst_key not in dst:
        dst[dst_key] = src[src_key].copy()


def normalize_sim_csv(raw: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Normalize sim CSVs from eval_pointnav_goal.py or flight_eval_utils.py."""
    data = dict(raw)

    if "t" not in data and "time_s" in data:
        data["t"] = data["time_s"].copy()

    _copy_if_present(data, raw, "pos_local_x", "stateEstimate.x")
    _copy_if_present(data, raw, "pos_local_y", "stateEstimate.y")
    _copy_if_present(data, raw, "pos_local_z_virtual", "stateEstimate.z")
    _copy_if_present(data, raw, "pos_x", "stateEstimate.x")
    _copy_if_present(data, raw, "pos_y", "stateEstimate.y")
    _copy_if_present(data, raw, "pos_z_virtual", "stateEstimate.z")
    if "stateEstimate.z" not in data:
        _copy_if_present(data, raw, "pos_z", "stateEstimate.z")

    _copy_if_present(data, raw, "vel_x", "velocity.x")
    _copy_if_present(data, raw, "vel_y", "velocity.y")
    _copy_if_present(data, raw, "vel_z", "velocity.z")

    _copy_if_present(data, raw, "roll_deg", "stabilizer.roll")
    _copy_if_present(data, raw, "pitch_deg", "stabilizer.pitch")
    _copy_if_present(data, raw, "yaw_deg", "stabilizer.yaw")
    _copy_if_present(data, raw, "ang_vel_x", "gyro.x")
    _copy_if_present(data, raw, "ang_vel_y", "gyro.y")
    _copy_if_present(data, raw, "ang_vel_z", "gyro.z")

    _copy_if_present(data, raw, "goal_local_x", "target.x")
    _copy_if_present(data, raw, "goal_local_y", "target.y")
    _copy_if_present(data, raw, "goal_local_z_virtual", "target.z")
    _copy_if_present(data, raw, "goal_x", "target.x")
    _copy_if_present(data, raw, "goal_y", "target.y")
    _copy_if_present(data, raw, "goal_z_virtual", "target.z")
    if "target.z" not in data:
        _copy_if_present(data, raw, "goal_z", "target.z")

    _copy_if_present(data, raw, "action_m1", "action.m1")
    _copy_if_present(data, raw, "action_m2", "action.m2")
    _copy_if_present(data, raw, "action_m3", "action.m3")
    _copy_if_present(data, raw, "action_m4", "action.m4")

    if all(k in data for k in ("t", "velocity.x", "velocity.y", "velocity.z")) and "acc.x" not in data:
        t = data["t"]
        vx, vy, vz = data["velocity.x"], data["velocity.y"], data["velocity.z"]
        ax = np.full_like(vx, np.nan, dtype=float)
        ay = np.full_like(vy, np.nan, dtype=float)
        az = np.full_like(vz, np.nan, dtype=float)
        if len(t) > 1:
            dt = np.diff(t)
            valid = np.abs(dt) > 1e-12
            ax[1:][valid] = np.diff(vx)[valid] / dt[valid] / 9.81
            ay[1:][valid] = np.diff(vy)[valid] / dt[valid] / 9.81
            az[1:][valid] = np.diff(vz)[valid] / dt[valid] / 9.81
        data["acc.x"] = ax
        data["acc.y"] = ay
        data["acc.z"] = az

    _copy_if_present(data, data, "acc.x", "imu.acc_x")
    _copy_if_present(data, data, "acc.y", "imu.acc_y")
    _copy_if_present(data, data, "acc.z", "imu.acc_z")
    _copy_if_present(data, data, "gyro.x", "imu.gyro_x")
    _copy_if_present(data, data, "gyro.y", "imu.gyro_y")
    _copy_if_present(data, data, "gyro.z", "imu.gyro_z")

    if all(k in data for k in ("action.m1", "action.m2", "action.m3", "action.m4")) and "motor.m1" not in data:
        for idx in range(1, 5):
            data[f"motor.m{idx}"] = np.clip((data[f"action.m{idx}"] + 1.0) * 0.5 * 65535.0, 0.0, 65535.0)

    if "target.yaw" not in data and "t" in data:
        data["target.yaw"] = np.full_like(data["t"], np.nan, dtype=float)
    if "pm.vbat" not in data and "t" in data:
        data["pm.vbat"] = np.full_like(data["t"], np.nan, dtype=float)

    return data


def normalize_real_csv(raw: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Normalize real-flight CSVs into the shared namespace."""
    data = dict(raw)
    _copy_if_present(data, data, "acc.x", "imu.acc_x")
    _copy_if_present(data, data, "acc.y", "imu.acc_y")
    _copy_if_present(data, data, "acc.z", "imu.acc_z")
    _copy_if_present(data, data, "gyro.x", "imu.gyro_x")
    _copy_if_present(data, data, "gyro.y", "imu.gyro_y")
    _copy_if_present(data, data, "gyro.z", "imu.gyro_z")
    return data


def resolve_output_dir(requested: Optional[str], sim_path: str) -> str:
    """Resolve output directory, defaulting to a timestamped folder."""
    if requested:
        return requested

    sim_dir = os.path.dirname(sim_path)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(sim_dir, f"sim2real_compare_{stamp}")


def _resolve_target_height(data: Dict[str, np.ndarray], fallback: Optional[float]) -> Optional[float]:
    """Resolve a target height from CSV columns or a fallback value."""
    for key in ("target.z", "goal_z_virtual", "goal_z", "target_z"):
        if key in data:
            values = data[key]
            finite = values[np.isfinite(values)]
            if finite.size > 0:
                return float(np.median(finite))
    return fallback


def trim_from_target_height_reach(
    data: Dict[str, np.ndarray],
    target_height: float,
    tolerance: float,
    label: str,
) -> Dict[str, np.ndarray]:
    """Trim a dataset so time starts at first upward reach of the target height."""
    if "t" not in data or "stateEstimate.z" not in data:
        print(f"WARNING: Cannot trim {label} on target height; missing t or stateEstimate.z.")
        return data

    t = data["t"]
    z = data["stateEstimate.z"]
    mask = np.isfinite(t) & np.isfinite(z)
    if not np.any(mask):
        print(f"WARNING: Cannot trim {label} on target height; no finite z samples.")
        return data

    valid_indices = np.nonzero(mask)[0]
    z_valid = z[mask]

    threshold = target_height - tolerance
    above = z_valid >= threshold
    if not np.any(above):
        print(
            f"WARNING: {label} never reaches target height {target_height:.3f}m "
            f"within tolerance ±{tolerance:.3f}m."
        )
        return data

    # Use the first upward crossing from below when possible so the start point
    # reflects the actual reach event rather than an already-near initial state.
    crossing_candidates = np.nonzero(above & np.concatenate(([True], ~above[:-1])))[0]
    if crossing_candidates.size > 0:
        reach_valid_idx = int(crossing_candidates[0])
    else:
        reach_valid_idx = int(np.nonzero(above)[0][0])

    start_idx = int(valid_indices[reach_valid_idx])
    t0 = float(t[start_idx])
    trimmed = {key: values[start_idx:].copy() for key, values in data.items()}
    trimmed["t"] = trimmed["t"] - t0
    print(
        f"Trimmed {label} to first target-height reach at t={t0:.3f}s "
        f"(target={target_height:.3f}m, tol=±{tolerance:.3f}m)."
    )
    return trimmed


# ── Time alignment ──────────────────────────────────────────────────────────
def align_by_resample(t_sim: np.ndarray, t_real: np.ndarray,
                       sim_data: Dict[str, np.ndarray],
                       real_data: Dict[str, np.ndarray]
                       ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Resample both datasets onto a common time grid using linear interpolation.
    
    The common grid spans [max(t_sim[0], t_real[0]), min(t_sim[-1], t_real[-1])]
    at the higher of the two sample rates.
    """
    t_start = max(t_sim[0], t_real[0])
    t_end = min(t_sim[-1], t_real[-1])
    
    if t_end <= t_start:
        raise ValueError(
            f"No time overlap between sim ({t_sim[0]:.2f}–{t_sim[-1]:.2f}s) "
            f"and real ({t_real[0]:.2f}–{t_real[-1]:.2f}s). "
            "Ensure both CSVs cover an overlapping time range, or use --t_offset."
        )
    
    # Use higher sample rate
    dt_sim = np.median(np.diff(t_sim))
    dt_real = np.median(np.diff(t_real))
    dt = min(dt_sim, dt_real)
    t_common = np.arange(t_start, t_end, dt)
    
    sim_resampled: Dict[str, np.ndarray] = {}
    real_resampled: Dict[str, np.ndarray] = {}
    
    for col in sim_data:
        if col == "t":
            continue
        sim_resampled[col] = np.interp(t_common, t_sim, sim_data[col])
    
    for col in real_data:
        if col == "t":
            continue
        real_resampled[col] = np.interp(t_common, t_real, real_data[col])
    
    return t_common, sim_resampled, real_resampled


# ── Metrics ─────────────────────────────────────────────────────────────────
def compute_metrics(sim: np.ndarray, real: np.ndarray) -> Dict[str, float]:
    """Compute comparison metrics between two same-length arrays.
    
    Ignores indices where either signal is NaN.
    """
    mask = ~(np.isnan(sim) | np.isnan(real))
    n = mask.sum()
    if n == 0:
        return {
            "rmse": float("nan"), "mae": float("nan"), "max_err": float("nan"),
            "pearson_r": float("nan"),
            "mean_sim": float("nan"), "mean_real": float("nan"),
            "std_sim": float("nan"), "std_real": float("nan"),
            "n_valid": 0,
        }
    
    s = sim[mask]
    r = real[mask]
    diff = s - r
    
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    max_err = float(np.max(np.abs(diff)))
    
    # Pearson correlation
    if np.std(s) < 1e-12 or np.std(r) < 1e-12:
        pearson_r = float("nan")
    else:
        pearson_r = float(np.corrcoef(s, r)[0, 1])
    
    return {
        "rmse": rmse,
        "mae": mae,
        "max_err": max_err,
        "pearson_r": pearson_r,
        "mean_sim": float(np.mean(s)),
        "mean_real": float(np.mean(r)),
        "std_sim": float(np.std(s)),
        "std_real": float(np.std(r)),
        "n_valid": int(n),
    }


def compute_tracking_error(actual: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Compute tracking error metrics (actual vs target from the same source)."""
    mask = ~(np.isnan(actual) | np.isnan(target))
    n = mask.sum()
    if n == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "max_err": float("nan"), "n_valid": 0}
    
    a = actual[mask]
    t = target[mask]
    diff = a - t
    
    return {
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "mae": float(np.mean(np.abs(diff))),
        "max_err": float(np.max(np.abs(diff))),
        "n_valid": int(n),
    }


# ── Plotting ────────────────────────────────────────────────────────────────
def plot_overlay(t: np.ndarray, sim: np.ndarray, real: np.ndarray,
                 label: str, unit: str, metrics: Dict[str, float],
                 output_path: Optional[str] = None):
    """Plot sim vs real overlay for a single signal."""
    if not HAS_MPL:
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), height_ratios=[3, 1],
                                    sharex=True, gridspec_kw={"hspace": 0.08})
    
    # Top: overlay
    ax1.plot(t, sim, label="Sim", linewidth=1, alpha=0.8)
    ax1.plot(t, real, label="Real", linewidth=1, alpha=0.8)
    ax1.set_ylabel(f"{label} [{unit}]")
    ax1.set_title(f"{label} — Sim vs Real  |  RMSE={metrics['rmse']:.4f} {unit}  |  r={metrics['pearson_r']:.3f}")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    
    # Bottom: error
    error = sim - real
    ax2.plot(t, error, color="red", linewidth=0.8, alpha=0.7)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel(f"Error [{unit}]")
    ax2.set_xlabel("Time [s]")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_tracking_comparison(t: np.ndarray,
                              sim_actual: np.ndarray, sim_target: np.ndarray,
                              real_actual: np.ndarray, real_target: np.ndarray,
                              label: str, unit: str,
                              output_path: Optional[str] = None):
    """Plot tracking error comparison: sim tracking-err vs real tracking-err."""
    if not HAS_MPL:
        return
    
    sim_err = sim_actual - sim_target
    real_err = real_actual - real_target
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, sim_err, label="Sim tracking error", linewidth=1, alpha=0.8)
    ax.plot(t, real_err, label="Real tracking error", linewidth=1, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel(f"Error [{unit}]")
    ax.set_xlabel("Time [s]")
    ax.set_title(f"{label} — Tracking Error Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_summary_bar(all_metrics: Dict[str, Dict[str, float]], output_path: Optional[str] = None):
    """Bar chart of RMSE for all compared signals."""
    if not HAS_MPL:
        return
    
    names = []
    rmses = []
    for (col, label, unit), m in all_metrics.items():
        if not math.isnan(m["rmse"]):
            names.append(label)
            rmses.append(m["rmse"])
    
    if not names:
        return
    
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.8), 5))
    bars = ax.bar(names, rmses, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_ylabel("RMSE")
    ax.set_title("Sim-to-Real RMSE per Signal")
    ax.tick_params(axis="x", rotation=45)
    
    # Value labels on bars
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Sim-to-Real Flight Data Comparison")
    parser.add_argument("--sim", required=True, help="Path to simulation CSV (from play_eval.py)")
    parser.add_argument("--real", required=True, help="Path to real-world CSV (from WebUI)")
    parser.add_argument("--output_dir", default=None,
                        help="Directory for output plots/CSV. If omitted, creates a timestamped folder next to the sim CSV.")
    parser.add_argument("--t_offset", type=float, default=0.0,
                        help="Time offset to add to real data (seconds). "
                             "Use this to manually align start times if needed.")
    parser.add_argument("--trim_idle", action="store_true", default=True,
                        help="Auto-trim idle rows from real data where motors are zero (default: True)")
    parser.add_argument("--no_trim_idle", action="store_false", dest="trim_idle",
                        help="Disable auto-trim of idle rows")
    parser.add_argument("--start_at_target_height", action="store_true", default=True,
                        help="Trim both datasets to start at first target-height reach (default: True)")
    parser.add_argument("--no_start_at_target_height", action="store_false", dest="start_at_target_height",
                        help="Disable target-height reach trimming")
    parser.add_argument("--target_height", type=float, default=None,
                        help="Optional override for target height in meters")
    parser.add_argument("--target_height_tolerance", type=float, default=0.0,
                        help="Tolerance in meters for detecting target-height reach")
    parser.add_argument("--no_plot", action="store_true", help="Skip plot generation")
    args = parser.parse_args()
    
    # Load data
    print(f"Loading sim data:  {args.sim}")
    sim_raw = normalize_sim_csv(load_csv(args.sim))
    print(f"  → {len(sim_raw.get('t', []))} rows, columns: {list(sim_raw.keys())[:10]}...")
    
    print(f"Loading real data: {args.real}")
    real_raw = normalize_real_csv(load_csv(args.real))
    print(f"  → {len(real_raw.get('t', []))} rows, columns: {list(real_raw.keys())[:10]}...")
    
    if "t" not in sim_raw or "t" not in real_raw:
        print("ERROR: Both CSVs must have a 't' (time) column.")
        sys.exit(1)
    
    # Auto-trim idle rows from real data (motors = 0 means drone wasn't flying)
    if args.trim_idle and "motor.m1" in real_raw:
        motor_sum = np.zeros(len(real_raw["t"]))
        for mc in ["motor.m1", "motor.m2", "motor.m3", "motor.m4"]:
            if mc in real_raw:
                motor_sum += np.nan_to_num(real_raw[mc], nan=0.0)
        flying_mask = motor_sum > 0
        n_before = len(real_raw["t"])
        if flying_mask.any():
            for col in real_raw:
                real_raw[col] = real_raw[col][flying_mask]
            n_after = len(real_raw["t"])
            print(f"\nAuto-trimmed idle rows: {n_before} → {n_after} "
                  f"(removed {n_before - n_after} rows where motors=0)")
        else:
            print("\nWARNING: All motor values are zero — drone may not have been flying!")

    if args.start_at_target_height:
        resolved_target_height = _resolve_target_height(real_raw, args.target_height)
        if resolved_target_height is None:
            resolved_target_height = _resolve_target_height(sim_raw, args.target_height)
        if resolved_target_height is None:
            print("\nWARNING: Could not resolve a target height; skipping target-height trimming.")
        else:
            print(
                f"\nTarget-height trim enabled: target={resolved_target_height:.3f}m, "
                f"tolerance=±{args.target_height_tolerance:.3f}m"
            )
            real_raw = trim_from_target_height_reach(
                real_raw, resolved_target_height, args.target_height_tolerance, "real"
            )
            sim_raw = trim_from_target_height_reach(
                sim_raw, resolved_target_height, args.target_height_tolerance, "sim"
            )

    # Reset both time series to start at 0
    sim_raw["t"] = sim_raw["t"] - sim_raw["t"][0]
    real_raw["t"] = real_raw["t"] - real_raw["t"][0]
    
    # Apply time offset to real data
    t_real = real_raw["t"] + args.t_offset
    t_sim = sim_raw["t"]
    
    print(f"\nSim time range:  {t_sim[0]:.3f} – {t_sim[-1]:.3f}s ({len(t_sim)} samples)")
    print(f"Real time range: {t_real[0]:.3f} – {t_real[-1]:.3f}s ({len(t_real)} samples)")
    
    # Time-align via resampling
    print("\nResampling onto common time grid...")
    t_common, sim_aligned, real_aligned = align_by_resample(t_sim, t_real, sim_raw, real_raw)
    print(f"  Common grid: {t_common[0]:.3f} – {t_common[-1]:.3f}s, {len(t_common)} samples")
    
    # Output directory
    args.output_dir = resolve_output_dir(args.output_dir, args.sim)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nOutput directory: {os.path.abspath(args.output_dir)}")
    
    # Compute metrics for each signal
    print(f"\n{'='*90}")
    print(f"{'Signal':<20} {'RMSE':>10} {'MAE':>10} {'MaxErr':>10} {'Pearson r':>10} "
          f"{'Mean(sim)':>10} {'Mean(real)':>11}")
    print(f"{'='*90}")
    
    all_metrics: Dict[Tuple, Dict[str, float]] = {}
    metrics_rows = []
    
    for col, label, unit in COMPARE_SIGNALS:
        if col not in sim_aligned or col not in real_aligned:
            continue
        
        m = compute_metrics(sim_aligned[col], real_aligned[col])
        all_metrics[(col, label, unit)] = m
        
        if m["n_valid"] == 0:
            print(f"{label:<20} {'(no valid data)':>10}")
            continue
        
        r_str = f"{m['pearson_r']:.4f}" if not math.isnan(m["pearson_r"]) else "N/A"
        print(f"{label:<20} {m['rmse']:>10.4f} {m['mae']:>10.4f} {m['max_err']:>10.4f} "
              f"{r_str:>10} {m['mean_sim']:>10.4f} {m['mean_real']:>11.4f}")
        
        metrics_rows.append({"signal": col, "label": label, "unit": unit, **m})
        
        # Per-signal overlay plot
        if not args.no_plot and args.output_dir:
            safe_name = col.replace(".", "_")
            plot_overlay(t_common, sim_aligned[col], real_aligned[col],
                         label, unit, m,
                         output_path=os.path.join(args.output_dir, f"overlay_{safe_name}.jpg"))
    
    print(f"{'='*90}")
    
    # Tracking error comparison
    print(f"\n{'='*70}")
    print(f"TRACKING ERROR (actual vs target, within each source)")
    print(f"{'='*70}")
    print(f"{'Signal':<20} {'RMSE(sim)':>10} {'RMSE(real)':>11} {'Δ RMSE':>10}")
    print(f"{'-'*70}")
    
    for actual_col, target_col, label, unit in TRACKING_SIGNALS:
        sim_has = actual_col in sim_aligned and target_col in sim_aligned
        real_has = actual_col in real_aligned and target_col in real_aligned
        
        if not sim_has or not real_has:
            continue
        
        sim_track = compute_tracking_error(sim_aligned[actual_col], sim_aligned[target_col])
        real_track = compute_tracking_error(real_aligned[actual_col], real_aligned[target_col])
        
        delta = real_track["rmse"] - sim_track["rmse"]
        print(f"{label:<20} {sim_track['rmse']:>10.4f} {real_track['rmse']:>11.4f} {delta:>+10.4f}")
        
        if not args.no_plot and args.output_dir:
            safe_name = actual_col.replace(".", "_")
            plot_tracking_comparison(
                t_common,
                sim_aligned[actual_col], sim_aligned[target_col],
                real_aligned[actual_col], real_aligned[target_col],
                label, unit,
                output_path=os.path.join(args.output_dir, f"tracking_{safe_name}.jpg")
            )
    
    print(f"{'='*70}")
    
    # Summary bar chart
    if not args.no_plot and args.output_dir and all_metrics:
        plot_summary_bar(all_metrics,
                         output_path=os.path.join(args.output_dir, "summary_rmse.jpg"))
    
    # Save metrics CSV
    if args.output_dir and metrics_rows:
        metrics_csv = os.path.join(args.output_dir, "sim2real_metrics.csv")
        fieldnames = list(metrics_rows[0].keys())
        with open(metrics_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(metrics_rows)
        print(f"\nMetrics saved to: {metrics_csv}")

    if args.output_dir:
        print(f"Plots saved to:   {args.output_dir}/")
    
    print("\nDone.")


if __name__ == "__main__":
    main()
