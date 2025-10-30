# tools/export_training_plots.py
"""
Export three graphs per run from TensorBoard logs:
1) Reward vs Environment Steps
2) Reward vs Wall-Clock Time
3) Loss Reduction vs Environment Steps   (loss_reduction = initial_loss - current_loss)

Output:
  plots/<lib>/<exp>/<timestamp>/reward_vs_step.{png,csv}
  plots/<lib>/<exp>/<timestamp>/reward_over_time.{png,csv}
  plots/<lib>/<exp>/<timestamp>/loss_reduction_vs_step.{png,csv}

Usage:
  python tools/export_training_plots.py --logs logs
  python tools/export_training_plots.py --logs logs/skrl --lib skrl
  python tools/export_training_plots.py --logs logs/skrl/quadcopter_direct --lib skrl
  python tools/export_training_plots.py --logs logs/skrl --lib skrl --reward-tag charts/episodic_return --loss-tag Loss/value_loss
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from metrics_exporter.tb_reader import TensorBoardReader

# Accept pure timestamp or timestamp with suffixes (e.g., "2025-10-17_18-22-01_ppo_torch")
_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[_ ]\d{2}-\d{2}-\d{2}($|[_\-].*)$")


def _is_timestamp_dir(name: str) -> bool:
    return bool(_TS_RE.match(name))


def _clean_segment(s: str) -> str:
    return re.sub(r"[^\w\-\.\+]+", "_", s or "")


def _infer_lib_exp_ts(event_file: Path, logs_root: Path, lib_override: Optional[str]) -> tuple[str, str, str]:
    """Infer (<lib>, <exp>, <timestamp>) from the event file path, resilient to different --logs roots."""
    d = event_file.parent
    ts_dir: Optional[Path] = None
    cur = d.resolve()
    logs_root = logs_root.resolve()

    while True:
        try:
            cur.relative_to(logs_root)
        except Exception:
            break
        if _is_timestamp_dir(cur.name):
            ts_dir = cur
            break
        if cur == logs_root:
            break
        cur = cur.parent

    if ts_dir is not None:
        exp_dir = ts_dir.parent
        exp = exp_dir.name if exp_dir and exp_dir.exists() else "exp"
        lib = lib_override or (exp_dir.parent.name if exp_dir and exp_dir.parent and exp_dir.parent.exists() else logs_root.name)
        ts = ts_dir.name
    else:
        ts = d.name
        exp = d.parent.name if d.parent and d.parent.exists() else "exp"
        lib = lib_override or logs_root.name

    return _clean_segment(lib or "unknown"), _clean_segment(exp or "unknown"), _clean_segment(ts or "run")


def _pick_reward_tag(all_tags: List[str], explicit: Optional[List[str]], reader: TensorBoardReader) -> Optional[str]:
    if explicit:
        for t in explicit:
            if t in all_tags:
                return t
    return reader.detect_reward_tag(all_tags)


def _pick_loss_tag(all_tags: List[str], explicit: Optional[str], reader: TensorBoardReader) -> Optional[str]:
    if explicit and explicit in all_tags:
        return explicit
    # Prefer common value loss keys if present, else first detected loss-like tag
    pref = [t for t in all_tags if "value_loss" in t.lower() or ("loss" in t.lower() and "value" in t.lower())]
    if pref:
        return pref[0]
    det = reader.detect_loss_tags(all_tags)
    return det[0] if det else None


@dataclass
class RewardSeries:
    tag: str
    steps: List[int]
    rewards: List[float]
    wall_times: List[float]
    elapsed_sec: List[float]


@dataclass
class LossSeries:
    tag: str
    steps: List[int]
    loss_values: List[float]
    loss_reduction: List[float]


def _read_tb_scalars(ev_path: Path, tag: str) -> Tuple[List[int], List[float], List[float]]:
    """Return (steps, values, wall_times) for a scalar tag from a tfevents file."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        print("[FATAL] Missing dependency 'tensorboard'. Install with: pip install tensorboard", file=sys.stderr)
        raise
    acc = EventAccumulator(str(ev_path))
    acc.Reload()
    scalars = acc.Scalars(tag)
    steps, values, wall_times = [], [], []
    for s in scalars:
        steps.append(int(getattr(s, "step", 0)))
        values.append(float(getattr(s, "value", 0.0)))
        wall_times.append(float(getattr(s, "wall_time", 0.0)))
    return steps, values, wall_times


def _read_reward_series(ev_path: Path, tag_candidates: Optional[List[str]], reader: TensorBoardReader) -> Optional[RewardSeries]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        print("[FATAL] Missing dependency 'tensorboard'. Install with: pip install tensorboard", file=sys.stderr)
        raise
    acc = EventAccumulator(str(ev_path))
    acc.Reload()
    all_tags = sorted(acc.Tags().get("scalars", []))
    picked = _pick_reward_tag(all_tags, tag_candidates, reader)
    if not picked:
        return None
    steps, values, wall_times = _read_tb_scalars(ev_path, picked)
    if not steps:
        return None
    t0 = wall_times[0] if wall_times else 0.0
    elapsed = [max(0.0, wt - t0) for wt in wall_times] if wall_times else [0.0] * len(steps)
    return RewardSeries(tag=picked, steps=steps, rewards=values, wall_times=wall_times, elapsed_sec=elapsed)


def _read_loss_reduction_series(ev_path: Path, explicit_loss_tag: Optional[str], reader: TensorBoardReader) -> Optional[LossSeries]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        print("[FATAL] Missing dependency 'tensorboard'. Install with: pip install tensorboard", file=sys.stderr)
        raise
    acc = EventAccumulator(str(ev_path))
    acc.Reload()
    all_tags = sorted(acc.Tags().get("scalars", []))
    picked = _pick_loss_tag(all_tags, explicit_loss_tag, reader)
    if not picked:
        return None
    steps, loss_vals, _ = _read_tb_scalars(ev_path, picked)
    if not steps:
        return None
    base = loss_vals[0]
    reduction = [base - v for v in loss_vals]
    return LossSeries(tag=picked, steps=steps, loss_values=loss_vals, loss_reduction=reduction)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    _ensure_dir(path.parent)
    pd.DataFrame(rows).to_csv(path, index=False)


def _plot_single_line(x, y, xlabel: str, ylabel: str, title: str, out_png: Path, style: str, dpi: int) -> None:
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
    except Exception:
        print("[FATAL] Missing plotting deps. Install with: pip install seaborn matplotlib", file=sys.stderr)
        raise
    sns.set_theme(style=style)
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.plot(x, y, linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    _ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=str, default="logs", help="Root logs folder to scan (or a subfolder under it).")
    ap.add_argument("--lib", type=str, default=None, help="Override library segment in output (e.g., rl_games, rsl_rl, skrl).")
    ap.add_argument("--out", type=str, default="plots", help="Output root folder for plots.")
    ap.add_argument("--style", type=str, default="whitegrid")
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--reward-tag", action="append", default=None, help="Explicit reward/return tag(s). Can repeat.")
    ap.add_argument("--loss-tag", type=str, default=None, help="Explicit loss tag to use for loss reduction plot.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logs_root = Path(args.logs).resolve()
    if not logs_root.exists():
        print(f"[FATAL] Logs root not found: {logs_root}", file=sys.stderr)
        sys.exit(2)

    reader = TensorBoardReader(logs_root)
    any_exported = False

    for ev in reader.iter_event_files():
        ev = ev.resolve()
        try:
            lib, exp, ts = _infer_lib_exp_ts(ev, logs_root, args.lib)
        except Exception as e:
            print(f"[WARN] Could not infer run folders for {ev}: {e}", file=sys.stderr)
            lib, exp, ts = (args.lib or "unknown", "unknown", "run")

        out_dir = Path(args.out) / lib / exp / ts
        run_name = f"{exp}/{ts}"

        # Reward series (for both reward plots)
        try:
            rs = _read_reward_series(ev, args.reward_tag, reader)
        except Exception as e:
            print(f"[WARN] Skipping TB read error {ev}: {e}", file=sys.stderr)
            continue

        if rs is not None:
            # CSV with all reward-related fields
            _save_csv(out_dir / "reward_series.csv", [
                {
                    "step": s,
                    "reward": r,
                    "wall_time_unix": wt,
                    "elapsed_seconds": es,
                    "elapsed_minutes": es / 60.0,
                }
                for s, r, wt, es in zip(rs.steps, rs.rewards, rs.wall_times, rs.elapsed_sec)
            ])
            # Reward vs Steps
            _plot_single_line(
                rs.steps, rs.rewards,
                xlabel="Environment Steps",
                ylabel="Episodic Reward",
                title=f"{run_name} — Reward vs Environment Steps",
                out_png=out_dir / "reward_vs_step.png",
                style=args.style, dpi=args.dpi
            )
            # Reward vs Time
            elapsed_minutes = [t / 60.0 for t in rs.elapsed_sec]
            _plot_single_line(
                elapsed_minutes, rs.rewards,
                xlabel="Wall-Clock Time (minutes)",
                ylabel="Episodic Reward",
                title=f"{run_name} — Reward vs Wall-Clock Time",
                out_png=out_dir / "reward_over_time.png",
                style=args.style, dpi=args.dpi
            )
            any_exported = True
        else:
            print(f"[INFO] No reward-like tag found in: {ev}")

        # Loss reduction series
        ls = _read_loss_reduction_series(ev, args.loss_tag, reader)
        if ls is not None:
            _save_csv(out_dir / "loss_reduction_series.csv", [
                {"step": s, "loss_value": v, "loss_reduction": red}
                for s, v, red in zip(ls.steps, ls.loss_values, ls.loss_reduction)
            ])
            _plot_single_line(
                ls.steps, ls.loss_reduction,
                xlabel="Environment Steps",
                ylabel=f"Loss Reduction",
                title=f"{run_name} — Loss Reduction vs Environment Steps",
                out_png=out_dir / "loss_reduction_vs_step.png",
                style=args.style, dpi=args.dpi
            )
            any_exported = True
        else:
            print(f"[INFO] No suitable loss tag found in: {ev}. Specify one with --loss-tag if needed.")

    if not any_exported:
        print("[INFO] No plots exported. Ensure training wrote TensorBoard event files and that reward/loss tags exist.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
