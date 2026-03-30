from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from .base import FaceBackend, FaceObservation


@dataclass
class DlibPerfConfig:
    detect_every_n_frames: int = 3
    detection_scale: float = 0.5
    use_tracker: bool = True
    tracker_type: str = "MOSSE"


class DlibFaceRecognitionBackend(FaceBackend):
    name = "dlib"

    def __init__(
        self,
        detect_every_n_frames: int = 3,
        detection_scale: float = 0.5,
        use_tracker: bool = True,
        tracker_type: str = "MOSSE",
    ) -> None:
        self._fr = None
        self._cfg = DlibPerfConfig(
            detect_every_n_frames=max(1, int(detect_every_n_frames)),
            detection_scale=float(min(max(detection_scale, 0.25), 1.0)),
            use_tracker=bool(use_tracker),
            tracker_type=str(tracker_type).upper(),
        )
        self._frame_index = 0
        self._last_location = None
        self._last_embedding = None
        self._tracker = None
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

        self._frame_index += 1
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        should_detect = (
            self._last_location is None
            or self._frame_index % self._cfg.detect_every_n_frames == 0
        )

        if not should_detect and self._cfg.use_tracker and self._tracker is not None:
            tracked = self._track_location(frame_bgr)
            if tracked is not None:
                self._last_location = tracked
                if self._last_embedding is not None:
                    top, right, bottom, left = tracked
                    return FaceObservation(
                        found=True,
                        bbox=(left, top, right, bottom),
                        embedding=self._last_embedding,
                        confidence=0.85,
                    )
            should_detect = True

        if should_detect:
            best = self._detect_largest_location(rgb)
            if best is None:
                self._last_location = None
                self._last_embedding = None
                self._tracker = None
                return FaceObservation(found=False, reason="no face")
            self._last_location = best
            self._init_tracker(frame_bgr, best)
        else:
            best = self._last_location

        if best is None:
            return FaceObservation(found=False, reason="no face")

        if should_detect or self._last_embedding is None:
            encodings = self._fr.face_encodings(rgb, [best])
            if not encodings:
                self._last_embedding = None
                return FaceObservation(found=False, reason="encoding failure")

            emb = np.array(encodings[0], dtype=np.float32)
            norm = float(np.linalg.norm(emb))
            if norm > 1e-8:
                emb /= norm
            self._last_embedding = emb

        top, right, bottom, left = best

        return FaceObservation(
            found=True,
            bbox=(left, top, right, bottom),
            embedding=self._last_embedding,
            confidence=1.0,
        )

    def _detect_largest_location(self, rgb: np.ndarray):
        h, w = rgb.shape[:2]
        scale = self._cfg.detection_scale

        if scale < 0.999:
            small = cv2.resize(rgb, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            raw_locations = self._fr.face_locations(small, model="hog")
            locations = [self._scale_location_back(loc, 1.0 / scale, w, h) for loc in raw_locations]
        else:
            locations = self._fr.face_locations(rgb, model="hog")

        if not locations:
            return None
        return max(locations, key=lambda loc: (loc[1] - loc[3]) * (loc[2] - loc[0]))

    def _scale_location_back(self, loc, inv_scale: float, w: int, h: int):
        top, right, bottom, left = loc
        top = int(round(top * inv_scale))
        right = int(round(right * inv_scale))
        bottom = int(round(bottom * inv_scale))
        left = int(round(left * inv_scale))
        top = max(0, min(top, h - 1))
        left = max(0, min(left, w - 1))
        right = max(left + 1, min(right, w))
        bottom = max(top + 1, min(bottom, h))
        return (top, right, bottom, left)

    def _init_tracker(self, frame_bgr: np.ndarray, location) -> None:
        if not self._cfg.use_tracker:
            self._tracker = None
            return

        tracker = self._create_tracker(self._cfg.tracker_type)
        if tracker is None:
            self._tracker = None
            return

        top, right, bottom, left = location
        width = max(1, right - left)
        height = max(1, bottom - top)
        ok = tracker.init(frame_bgr, (left, top, width, height))
        self._tracker = tracker if ok else None

    def _track_location(self, frame_bgr: np.ndarray):
        if self._tracker is None:
            return None

        ok, rect = self._tracker.update(frame_bgr)
        if not ok:
            self._tracker = None
            return None

        x, y, w_box, h_box = rect
        h, w = frame_bgr.shape[:2]
        left = max(0, int(round(x)))
        top = max(0, int(round(y)))
        right = min(w, left + max(1, int(round(w_box))))
        bottom = min(h, top + max(1, int(round(h_box))))

        if right <= left or bottom <= top:
            self._tracker = None
            return None

        return (top, right, bottom, left)

    def _create_tracker(self, tracker_type: str):
        name = tracker_type.upper()
        create_names = {
            "MOSSE": ["TrackerMOSSE_create", "legacy.TrackerMOSSE_create"],
            "KCF": ["TrackerKCF_create", "legacy.TrackerKCF_create"],
            "CSRT": ["TrackerCSRT_create", "legacy.TrackerCSRT_create"],
        }
        candidates = create_names.get(name, create_names["MOSSE"])

        for candidate in candidates:
            if candidate.startswith("legacy."):
                legacy = getattr(cv2, "legacy", None)
                func = getattr(legacy, candidate.split(".", 1)[1], None) if legacy is not None else None
            else:
                func = getattr(cv2, candidate, None)
            if callable(func):
                try:
                    return func()
                except Exception:
                    continue
        return None
