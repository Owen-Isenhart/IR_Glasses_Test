from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class CameraConfig:
    index: int = 2
    width: int = 640
    height: int = 480
    fps: int = 30
    backend_prefer_v4l2: bool = True


class IRCamera:
    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self.backend_name = ""

    def open(self) -> None:
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY] if self.config.backend_prefer_v4l2 else [cv2.CAP_ANY]
        last_error = ""
        for backend in backends:
            cap = cv2.VideoCapture(self.config.index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
                cap.set(cv2.CAP_PROP_FPS, self.config.fps)
                self.cap = cap
                self.backend_name = "V4L2" if backend == cv2.CAP_V4L2 else "ANY"
                return
            last_error = f"failed backend {backend}"
            cap.release()
        raise RuntimeError(f"Could not open camera index {self.config.index}: {last_error}")

    def read(self) -> np.ndarray:
        if self.cap is None:
            raise RuntimeError("Camera is not open")
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read frame from camera")
        return frame

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def metadata(self) -> Dict[str, object]:
        if self.cap is None:
            return {}
        return {
            "index": self.config.index,
            "backend": self.backend_name,
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(self.cap.get(cv2.CAP_PROP_FPS)),
        }

    def preflight(self, sample_frames: int = 20) -> Dict[str, object]:
        if self.cap is None:
            raise RuntimeError("Camera is not open")

        start = time.perf_counter()
        gray_scores = []
        for _ in range(sample_frames):
            frame = self.read()
            if frame.ndim == 2 or frame.shape[2] == 1:
                gray_scores.append(1.0)
                continue
            b, g, r = cv2.split(frame)
            # IR-like streams often have near-equal channels when exposed through BGR.
            delta = np.mean(np.abs(b.astype(np.float32) - g.astype(np.float32)))
            delta += np.mean(np.abs(g.astype(np.float32) - r.astype(np.float32)))
            gray_scores.append(1.0 if delta < 8.0 else 0.0)

        elapsed = time.perf_counter() - start
        observed_fps = sample_frames / max(elapsed, 1e-6)
        return {
            "observed_fps": observed_fps,
            "gray_likelihood": float(np.mean(gray_scores)),
            "sample_frames": sample_frames,
        }
