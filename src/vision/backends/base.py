from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class FaceObservation:
    found: bool
    bbox: Optional[Tuple[int, int, int, int]] = None
    embedding: Optional[np.ndarray] = None
    confidence: float = 0.0
    reason: str = ""


class FaceBackend(ABC):
    name: str = "base"

    @abstractmethod
    def is_available(self) -> Tuple[bool, str]:
        """Return backend readiness and message."""

    @abstractmethod
    def observe(self, frame_bgr: np.ndarray) -> FaceObservation:
        """Detect face and produce an embedding for similarity."""
