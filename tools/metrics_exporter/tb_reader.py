from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

class TensorBoardReader:
    """
    Robust reader for TensorBoard scalar data. Does not depend on the training codebase.
    """
    def __init__(self, log_root: str | Path):
        self.log_root = Path(log_root)
        if not self.log_root.exists():
            raise FileNotFoundError(f"Log root not found: {self.log_root}")

        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except Exception as e:
            raise RuntimeError("tensorboard package is required. Install with: pip install tensorboard") from e
        self._EA = EventAccumulator

    def iter_event_files(self) -> Iterable[Path]:
        # Typical layout: logs/<lib>/<exp>/<timestamp>/events.out.tfevents.*
        # Search two levels deep to avoid accidental recursion into very deep trees.
        patterns = [
            self.log_root.rglob("events.out.tfevents.*"),
            self.log_root.rglob("*tfevents*"),
        ]
        seen = set()
        for gen in patterns:
            for f in gen:
                if f.is_file():
                    if f.suffixes and any("tfevents" in s for s in f.suffixes) or "tfevents" in f.name:
                        if f not in seen:
                            seen.add(f)
                            yield f

    def read_scalars(self, event_file: Path) -> Dict[str, List[Tuple[int, float]]]:
        acc = self._EA(str(event_file))
        try:
            acc.Reload()
        except Exception as e:
            raise RuntimeError(f"Failed to load TensorBoard file: {event_file}") from e
        tags = acc.Tags().get("scalars", [])
        out: Dict[str, List[Tuple[int, float]]] = {}
        for tag in tags:
            seq = []
            for ev in acc.Scalars(tag):
                # Use global step; timestamp isn't stable across reruns
                seq.append((int(getattr(ev, 'step', 0)), float(getattr(ev, 'value', 0.0))))
            out[tag] = seq
        return out

    @staticmethod
    def detect_reward_tag(tags: List[str]) -> str | None:
        # Heuristic: reward/return and episode/train
        scored: List[tuple[int, str]] = []
        for t in tags:
            low = t.lower()
            score = 0
            if "reward" in low or "return" in low:
                score += 2
            if "episode" in low or "ep" in low:
                score += 1
            if "train" in low:
                score += 1
            if "eval" in low:
                score -= 1
            if score > 0:
                scored.append((score, t))
        if not scored:
            return None
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored[0][1]

    @staticmethod
    def detect_loss_tags(tags: List[str]) -> List[str]:
        # Common names across RSL-RL / RL-Games
        candidates = []
        keys = ["loss", "value_loss", "policy_loss", "entropy", "kl", "critic", "actor"]
        for t in tags:
            low = t.lower()
            if any(k in low for k in keys) and "eval" not in low:
                candidates.append(t)
        return sorted(set(candidates))
