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
    # Accelerometer
    ("acc.x", "Accel X", "m/s²"),
    ("acc.y", "Accel Y", "m/s²"),
    ("acc.z", "Accel Z", "m/s²"),
    # Gyroscope
    ("gyro.x", "Gyro X", "deg/s"),
    ("gyro.y", "Gyro Y", "deg/s"),
    ("gyro.z", "Gyro Z", "deg/s"),
    # Motors (may be NaN on real if firmware doesn't expose them)
    ("motor.m1", "Motor M1", "RPM/PWM"),
    ("motor.m2", "Motor M2", "RPM/PWM"),
    ("motor.m3", "Motor M3", "RPM/PWM"),
    ("motor.m4", "Motor M4", "RPM/PWM"),
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
    parser.add_argument("--output_dir", default=None, help="Directory for output plots/CSV")
    parser.add_argument("--t_offset", type=float, default=0.0,
                        help="Time offset to add to real data (seconds). "
                             "Use this to manually align start times if needed.")
    parser.add_argument("--no_plot", action="store_true", help="Skip plot generation")
    args = parser.parse_args()
    
    # Load data
    print(f"Loading sim data:  {args.sim}")
    sim_raw = load_csv(args.sim)
    print(f"  → {len(sim_raw.get('t', []))} rows, columns: {list(sim_raw.keys())[:10]}...")
    
    print(f"Loading real data: {args.real}")
    real_raw = load_csv(args.real)
    print(f"  → {len(real_raw.get('t', []))} rows, columns: {list(real_raw.keys())[:10]}...")
    
    if "t" not in sim_raw or "t" not in real_raw:
        print("ERROR: Both CSVs must have a 't' (time) column.")
        sys.exit(1)
    
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
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
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
