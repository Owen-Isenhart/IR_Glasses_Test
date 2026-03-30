from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve
from typing import Optional, Tuple

import cv2
import numpy as np

from .base import FaceBackend, FaceObservation


@dataclass
class _DetectorConfig:
    min_detection_confidence: float = 0.35
    model_selection: int = 0
    lost_face_grace_frames: int = 8


@dataclass
class _EmbeddingConfig:
    model_path: str = "models/mobilefacenet.onnx"
    model_url: str = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-11-int8.onnx"
    input_size: int = 112
    normalize_mean: float = 127.5
    normalize_std: float = 128.0


class MediaPipeEmbeddingBackend(FaceBackend):
    name = "mediapipe"

    def __init__(
        self,
        detector_cfg: Optional[_DetectorConfig] = None,
        embedding_cfg: Optional[_EmbeddingConfig] = None,
    ) -> None:
        self.detector_cfg = detector_cfg or _DetectorConfig()
        self.embedding_cfg = embedding_cfg or _EmbeddingConfig()
        self._mp = None
        self._detector = None
        self._ort = None
        self._session = None
        self._input_name = ""
        self._input_layout = "nchw"
        self._last_observation: Optional[FaceObservation] = None
        self._missing_face_frames = 0

        try:
            import mediapipe as mp  # type: ignore

            self._mp = mp
            self._detector = mp.solutions.face_detection.FaceDetection(
                model_selection=self.detector_cfg.model_selection,
                min_detection_confidence=self.detector_cfg.min_detection_confidence,
            )
        except Exception:
            self._detector = None

        try:
            import onnxruntime as ort  # type: ignore

            self._ort = ort
            model_path = Path(self.embedding_cfg.model_path)
            self._ensure_model_file(model_path)
            if model_path.exists():
                providers = [
                    p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in ort.get_available_providers()
                ]
                if not providers:
                    providers = ["CPUExecutionProvider"]
                self._session = ort.InferenceSession(str(model_path), providers=providers)
                model_inputs = self._session.get_inputs()
                if model_inputs:
                    self._input_name = model_inputs[0].name
                    self._input_layout = self._infer_input_layout(model_inputs[0].shape)
        except Exception:
            self._session = None

    def _ensure_model_file(self, model_path: Path) -> None:
        if model_path.exists():
            return
        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            urlretrieve(self.embedding_cfg.model_url, model_path)
        except (URLError, OSError, ValueError):
            # Keep startup non-fatal: availability check will return a clear message.
            return

    def is_available(self) -> Tuple[bool, str]:
        if self._detector is None:
            return False, "mediapipe is not installed or detector could not initialize"
        if self._ort is None:
            return False, "onnxruntime is not installed"
        if self._session is None:
            return False, f"embedding model unavailable: {self.embedding_cfg.model_path}"
        return True, "ok"

    def observe(self, frame_bgr: np.ndarray) -> FaceObservation:
        if self._detector is None:
            return FaceObservation(found=False, reason="backend unavailable")

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._detector.process(rgb)
        detections = result.detections if result and result.detections else []
        if not detections:
            return self._handle_missed_face()

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
            return self._handle_missed_face()

        emb = self._compute_embedding(frame_bgr, best_bbox)
        if emb is None:
            return self._handle_missed_face()
        self._missing_face_frames = 0
        obs = FaceObservation(found=True, bbox=best_bbox, embedding=emb, confidence=best_conf)
        self._last_observation = obs
        return obs

    def _handle_missed_face(self) -> FaceObservation:
        if self._last_observation is None:
            return FaceObservation(found=False, reason="no face")

        self._missing_face_frames += 1
        if self._missing_face_frames <= max(0, int(self.detector_cfg.lost_face_grace_frames)):
            # Briefly reuse last observation to reduce UI/state flicker from detector jitter.
            return FaceObservation(
                found=True,
                bbox=self._last_observation.bbox,
                embedding=self._last_observation.embedding,
                confidence=float(self._last_observation.confidence * 0.8),
                reason="grace_hold",
            )

        self._last_observation = None
        return FaceObservation(found=False, reason="no face")

    def _compute_embedding(
        self, frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        if self._session is None or not self._input_name:
            return None

        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return None

        h, w = frame_bgr.shape[:2]
        pad_x = int((x2 - x1) * 0.2)
        pad_y = int((y2 - y1) * 0.2)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)

        face = frame_bgr[cy1:cy2, cx1:cx2]
        if face.size == 0 or face.shape[0] < 10 or face.shape[1] < 10:
            return None

        chip = cv2.resize(
            face,
            (self.embedding_cfg.input_size, self.embedding_cfg.input_size),
            interpolation=cv2.INTER_AREA,
        )
        chip_rgb = cv2.cvtColor(chip, cv2.COLOR_BGR2RGB)
        model_input = chip_rgb.astype(np.float32)
        model_input = (model_input - self.embedding_cfg.normalize_mean) / self.embedding_cfg.normalize_std

        if self._input_layout == "nchw":
            model_input = np.transpose(model_input, (2, 0, 1))[None, ...]
        else:
            model_input = model_input[None, ...]

        outputs = self._session.run(None, {self._input_name: model_input})
        if not outputs:
            return None

        vec = np.asarray(outputs[0]).reshape(-1).astype(np.float32)
        if vec.size == 0:
            return None

        norm = float(np.linalg.norm(vec))
        if norm < 1e-8:
            return None
        return vec / norm

    def _infer_input_layout(self, shape: object) -> str:
        if not isinstance(shape, (list, tuple)) or len(shape) != 4:
            return "nchw"
        c_dim = shape[1]
        last_dim = shape[3]
        if isinstance(last_dim, int) and last_dim == 3:
            return "nhwc"
        if isinstance(c_dim, int) and c_dim == 3:
            return "nchw"
        return "nchw"


class MediaPipeHogBackend(MediaPipeEmbeddingBackend):
    """Backward-compatible alias for legacy imports."""
