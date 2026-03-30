from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .base import FaceBackend, FaceObservation


@dataclass
class _DetectorConfig:
    min_detection_confidence: float = 0.45
    model_selection: int = 0


class MediaPipeHogBackend(FaceBackend):
    name = "mediapipe"

    def __init__(self, cfg: Optional[_DetectorConfig] = None) -> None:
        self.cfg = cfg or _DetectorConfig()
        self._mp = None
        self._detector = None
        self._hog = cv2.HOGDescriptor(
            _winSize=(64, 64),
            _blockSize=(16, 16),
            _blockStride=(8, 8),
            _cellSize=(8, 8),
            _nbins=9,
        )

        try:
            import mediapipe as mp  # type: ignore

            self._mp = mp
            self._detector = mp.solutions.face_detection.FaceDetection(
                model_selection=self.cfg.model_selection,
                min_detection_confidence=self.cfg.min_detection_confidence,
            )
        except Exception:
            self._detector = None

    def is_available(self) -> Tuple[bool, str]:
        if self._detector is None:
            return False, "mediapipe is not installed or detector could not initialize"
        return True, "ok"

    def observe(self, frame_bgr: np.ndarray) -> FaceObservation:
        if self._detector is None:
            return FaceObservation(found=False, reason="backend unavailable")

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._detector.process(rgb)
        detections = result.detections if result and result.detections else []
        if not detections:
            return FaceObservation(found=False, reason="no face")

        # Keep the largest detected face to reduce multi-face ambiguity.
        best_bbox = None
        best_area = -1
        best_conf = 0.0
        for d in detections:
            rel = d.location_data.relative_bounding_box
            x = max(0, int(rel.xmin * w))
            y = max(0, int(rel.ymin * h))
            bw = max(1, int(rel.width * w))
            bh = max(1, int(rel.height * h))
            x2 = min(w, x + bw)
            y2 = min(h, y + bh)
            area = max(0, (x2 - x) * (y2 - y))
            if area > best_area:
                best_area = area
                best_bbox = (x, y, x2, y2)
                best_conf = float(d.score[0]) if d.score else 0.0

        if not best_bbox:
            return FaceObservation(found=False, reason="invalid bbox")

        emb = self._compute_hog_embedding(frame_bgr, best_bbox)
        if emb is None:
            return FaceObservation(found=False, reason="embedding failure")

        return FaceObservation(found=True, bbox=best_bbox, embedding=emb, confidence=best_conf)

    def _compute_hog_embedding(
        self, frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return None

        face = frame_bgr[y1:y2, x1:x2]
        if face.size == 0:
            return None

        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)

        desc = self._hog.compute(gray)
        if desc is None:
            return None

        vec = desc.flatten().astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-8:
            return None
        return vec / norm
