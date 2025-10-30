from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd


def _safe_rolling_mean(values: pd.Series, frac: float = 0.01, hard_min: int = 21) -> pd.Series:
    """
    Rolling mean with guards for short series:
      - window = max(1, min(len(values), max(hard_min, ceil(frac * len(values)))))
      - min_periods = max(1, min(5, window))
      - if len(values) < 5, return values unchanged (no smoothing)
    """
    n = int(values.shape[0])
    if n < 5:
        return values
    # dynamic window: at least `hard_min`, or 1% of the series, but not more than n
    est = max(hard_min, int((n * frac) + 0.5))
    window = max(1, min(n, est))
    min_periods = max(1, min(5, window))
    return values.rolling(window=window, min_periods=min_periods).mean()


@dataclass
class SeabornPlotter:
    out_dir: Path
    style: str = "whitegrid"
    dpi: int = 160

    def __post_init__(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _require_backends(self):
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
        except Exception as e:
            raise RuntimeError("Required packages missing. Install: pip install seaborn matplotlib") from e
        return sns, plt

    def plot_reward(self, series: List[Tuple[int, float]], tag: str, run_name: str) -> Path:
        sns, plt = self._require_backends()
        if not series:
            raise ValueError("Empty reward series")
        df = pd.DataFrame(series, columns=["step", "reward"]).sort_values("step", kind="mergesort")
        df["reward_smooth"] = _safe_rolling_mean(df["reward"])

        sns.set_theme(style=self.style)
        fig = plt.figure(figsize=(10, 6))
        ax = plt.gca()
        # Draw raw if very short; otherwise draw smoothed
        ycol = "reward_smooth" if df["reward_smooth"].notna().sum() >= 5 else "reward"
        ax.plot(df["step"], df[ycol], label="reward")
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Reward")
        ax.set_title(f"{run_name} — {tag}")
        ax.legend(loc="best")
        fig.tight_layout()

        out_png = self.out_dir / "reward_vs_step.png"
        fig.savefig(out_png, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        out_csv = self.out_dir / "reward_vs_step.csv"
        df.to_csv(out_csv, index=False)
        return out_png

    def plot_losses(self, loss_series: Dict[str, List[Tuple[int, float]]], run_name: str) -> Optional[Path]:
        if not loss_series:
            return None
        sns, plt = self._require_backends()

        frames = []
        for tag, seq in loss_series.items():
            if not seq:
                continue
            tmp = pd.DataFrame(seq, columns=["step", "value"]).sort_values("step", kind="mergesort")
            tmp["tag"] = tag
            frames.append(tmp)
        if not frames:
            return None

        df = pd.concat(frames, ignore_index=True)

        sns.set_theme(style=self.style)
        fig = plt.figure(figsize=(10, 6))
        ax = plt.gca()

        for tag, g in df.groupby("tag", sort=True):
            g = g.sort_values("step", kind="mergesort")
            smooth = _safe_rolling_mean(g["value"])
            ycol = "value"
            if smooth.notna().sum() >= 5:
                g = g.assign(value_smooth=smooth)
                ycol = "value_smooth"
            ax.plot(g["step"], g[ycol], label=tag)

        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Loss / Metric")
        ax.set_title(f"{run_name} — optimization metrics")
        ax.legend(loc="best")
        fig.tight_layout()

        out_png = self.out_dir / "losses_vs_step.png"
        fig.savefig(out_png, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        out_csv = self.out_dir / "losses_vs_step.csv"
        df.to_csv(out_csv, index=False)
        return out_png
