from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .base import FaceBackend, FaceObservation


class DlibFaceRecognitionBackend(FaceBackend):
    name = "dlib"

    def __init__(self) -> None:
        self._fr = None
        try:
            import face_recognition as fr  # type: ignore

            self._fr = fr
        except Exception:
            self._fr = None

    def is_available(self) -> Tuple[bool, str]:
        if self._fr is None:
            return False, "face_recognition/dlib is not installed"
        return True, "ok"

    def observe(self, frame_bgr: np.ndarray) -> FaceObservation:
        if self._fr is None:
            return FaceObservation(found=False, reason="backend unavailable")

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        locations = self._fr.face_locations(rgb, model="hog")
        if not locations:
            return FaceObservation(found=False, reason="no face")

        # Keep largest location by area.
        best = max(locations, key=lambda loc: (loc[1] - loc[3]) * (loc[2] - loc[0]))
        encodings = self._fr.face_encodings(rgb, [best])
        if not encodings:
            return FaceObservation(found=False, reason="encoding failure")

        top, right, bottom, left = best
        emb = np.array(encodings[0], dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        if norm > 1e-8:
            emb /= norm

        return FaceObservation(
            found=True,
            bbox=(left, top, right, bottom),
            embedding=emb,
            confidence=1.0,
        )
