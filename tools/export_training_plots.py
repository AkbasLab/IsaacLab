"""
Scan logs, detect each run, and export Seaborn PNGs + CSVs into dated folders.

Output structure:
  plots/<lib>/<exp>/<timestamp>/reward_vs_step.{png,csv}
  plots/<lib>/<exp>/<timestamp>/losses_vs_step.{png,csv}

Usage:
  python tools/export_training_plots.py --logs logs
  python tools/export_training_plots.py --logs logs/rl_games --lib rl_games
  python tools/export_training_plots.py --logs logs/rsl_rl --lib rsl_rl
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

from metrics_exporter.tb_reader import TensorBoardReader
from metrics_exporter.plotter import SeabornPlotter


_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[_ ]\d{2}-\d{2}-\d{2}$")


def _is_timestamp_dir(name: str) -> bool:
    return bool(_TS_RE.match(name))


def _infer_lib_exp_ts(event_file: Path, logs_root: Path, lib_override: Optional[str]) -> tuple[str, str, str]:
    """
    Infer (<lib>, <exp>, <timestamp>) from the *event file's* path.

    Strategy:
      - Walk up from the event file's parent.
      - Identify a timestamp-like directory (YYYY-MM-DD_HH-MM-SS).
      - The directory above that is treated as <exp> (e.g., 'ant').
      - The directory above that is treated as <lib> unless --lib overrides it.
      - If no timestamp-like directory is found, use the immediate parent as timestamp,
        its parent as exp, and the logs_root name (or override) as lib.

    This is resilient to being called with:
      --logs logs
      --logs logs/rsl_rl
      --logs logs/rsl_rl/ant
      --logs logs/rsl_rl/ant/<timestamp>
    """
    # Start at the directory containing the tfevents file
    d = event_file.parent

    # Try to find a timestamp directory at or above d, but not above logs_root
    ts_dir = None
    cur = d
    while True:
        try:
            rel = cur.relative_to(logs_root)
        except Exception:
            break
        if _is_timestamp_dir(cur.name):
            ts_dir = cur
            break
        if cur == logs_root:
            break
        cur = cur.parent

    if ts_dir is not None:
        # exp is the parent of timestamp dir if available
        exp_dir = ts_dir.parent if ts_dir.parent != logs_root.parent else ts_dir.parent
        exp = exp_dir.name if exp_dir is not None else "exp"
        # lib is parent of exp if available, otherwise logs_root name
        lib = lib_override or (exp_dir.parent.name if exp_dir and exp_dir.parent != exp_dir and exp_dir.parent.exists() else logs_root.name)
        ts = ts_dir.name
    else:
        # Fallback: use immediate parent as 'timestamp', its parent as 'exp'
        ts = d.name
        exp = d.parent.name if d.parent and d.parent.exists() else "exp"
        lib = lib_override or logs_root.name

    # Final sanitation to avoid weird characters in folder names
    def _clean(s: str) -> str:
        return re.sub(r"[^\w\-\.\+]+", "_", s)

    return _clean(lib or "unknown"), _clean(exp or "unknown"), _clean(ts or "run")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=str, default="logs", help="Root logs folder to scan for tfevents.")
    ap.add_argument("--lib", type=str, default=None, help="Override library folder name in output (rl_games, rsl_rl, sb3).")
    ap.add_argument("--style", type=str, default="whitegrid")
    ap.add_argument("--dpi", type=int, default=160)
    args = ap.parse_args()

    logs_root = Path(args.logs).resolve()
    if not logs_root.exists():
        print(f"[FATAL] Logs root not found: {logs_root}", file=sys.stderr)
        sys.exit(2)

    reader = TensorBoardReader(logs_root)
    any_exported = False

    for ev in reader.iter_event_files():
        try:
            lib, exp, ts = _infer_lib_exp_ts(ev.resolve(), logs_root, args.lib)
        except Exception as e:
            print(f"[WARN] Could not infer run folders for {ev}: {e}", file=sys.stderr)
            lib, exp, ts = (args.lib or "unknown", "unknown", "run")

        # Decide output directory
        out_dir = Path("plots") / lib / exp / ts
        plotter = SeabornPlotter(out_dir=out_dir, style=args.style, dpi=args.dpi)

        # Read all scalars from this file
        try:
            scalars = reader.read_scalars(ev)
        except Exception as e:
            print(f"[WARN] Skipping unreadable TB file {ev}: {e}", file=sys.stderr)
            continue

        tags = list(scalars.keys())

        # Reward
        reward_tag = reader.detect_reward_tag(tags)
        if reward_tag:
            try:
                plotter.plot_reward(scalars[reward_tag], reward_tag, run_name=f"{exp}/{ts}")
                any_exported = True
                print(f"[OK] Reward plot: {out_dir/'reward_vs_step.png'}")
            except Exception as e:
                print(f"[WARN] Skipping reward plot for {ev.name}: {e}", file=sys.stderr)
        else:
            print(f"[INFO] No reward-like tag found in: {ev}")

        # Losses
        loss_tags = reader.detect_loss_tags(tags)
        loss_series = {t: scalars[t] for t in loss_tags}
        if loss_series:
            try:
                plotter.plot_losses(loss_series, run_name=f"{exp}/{ts}")
                any_exported = True
                print(f"[OK] Losses plot: {out_dir/'losses_vs_step.png'}")
            except Exception as e:
                print(f"[WARN] Skipping losses plot for {ev.name}: {e}", file=sys.stderr)

    if not any_exported:
        print("[INFO] No plots exported. Ensure training wrote TensorBoard event files under --logs.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
