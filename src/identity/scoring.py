from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np


@dataclass
class Thresholds:
    match_max: float = 0.40
    blocked_max: float = 0.60


@dataclass
class AutoCalConfig:
    min_samples: int = 25
    window: int = 120
    update_every: int = 10
    match_quantile: float = 0.95
    blocked_quantile: float = 0.995
    match_margin: float = 0.005
    blocked_margin: float = 0.02
    min_gap: float = 0.08
    max_blocked: float = 0.85


class AutoCalibrator:
    """Online threshold tuning from recent distance observations."""

    def __init__(self, cfg: AutoCalConfig) -> None:
        self.cfg = cfg
        self._samples: Deque[float] = deque(maxlen=max(1, cfg.window))
        self._since_update = 0

    def reset(self) -> None:
        self._samples.clear()
        self._since_update = 0

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def update(self, distance: float, thresholds: Thresholds) -> bool:
        self._samples.append(float(distance))
        self._since_update += 1

        if len(self._samples) < self.cfg.min_samples:
            return False
        if self._since_update < self.cfg.update_every:
            return False

        self._since_update = 0
        arr = np.array(self._samples, dtype=np.float32)
        q_match = float(np.quantile(arr, self.cfg.match_quantile))
        q_blocked = float(np.quantile(arr, self.cfg.blocked_quantile))

        target_match = q_match + self.cfg.match_margin
        target_blocked = q_blocked + self.cfg.blocked_margin
        target_blocked = max(target_blocked, target_match + self.cfg.min_gap)
        target_blocked = min(self.cfg.max_blocked, target_blocked)

        # Assign directly from observed data for calibration-true thresholds.
        thresholds.match_max = float(max(0.0, target_match))
        thresholds.blocked_max = float(max(thresholds.match_max + self.cfg.min_gap, target_blocked))
        return True


class DistanceSmoother:
    def __init__(self, alpha: float = 0.30, rolling: int = 5) -> None:
        self.alpha = alpha
        self.rolling = max(1, rolling)
        self._ema: Optional[float] = None
        self._window: Deque[float] = deque(maxlen=self.rolling)

    def reset(self) -> None:
        self._ema = None
        self._window.clear()

    def update(self, value: float) -> float:
        self._window.append(value)
        median = float(np.median(np.array(self._window, dtype=np.float32)))
        if self._ema is None:
            self._ema = median
        else:
            self._ema = self.alpha * median + (1.0 - self.alpha) * self._ema
        return self._ema


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def classify_distance(distance: float, thresholds: Thresholds) -> str:
    if distance <= thresholds.match_max:
        return "matched"
    if distance <= thresholds.blocked_max:
        return "blocked"
    return "blocked"
