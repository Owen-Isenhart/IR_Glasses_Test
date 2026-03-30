from .base import FaceBackend, FaceObservation
from .mediapipe_backend import MediaPipeEmbeddingBackend, _DetectorConfig
from .dlib_backend import DlibFaceRecognitionBackend


def create_backend(name: str, options: dict | None = None) -> FaceBackend:
    """Factory for runtime backend selection."""
    normalized = (name or "").strip().lower()
    options = options or {}
    if normalized in {"mediapipe", "mp"}:
        mp_opts = options.get("mediapipe", {}) if isinstance(options, dict) else {}
        if not isinstance(mp_opts, dict):
            mp_opts = {}
        detector_cfg = _DetectorConfig(
            min_detection_confidence=float(mp_opts.get("min_detection_confidence", 0.35)),
            model_selection=int(mp_opts.get("model_selection", 0)),
            lost_face_grace_frames=int(mp_opts.get("lost_face_grace_frames", 8)),
        )
        return MediaPipeEmbeddingBackend(detector_cfg=detector_cfg)
    if normalized in {"dlib", "face_recognition", "fr"}:
        dlib_opts = options.get("dlib", {}) if isinstance(options, dict) else {}
        if not isinstance(dlib_opts, dict):
            dlib_opts = {}
        return DlibFaceRecognitionBackend(
            detect_every_n_frames=int(dlib_opts.get("detect_every_n_frames", 3)),
            detection_scale=float(dlib_opts.get("detection_scale", 0.5)),
            use_tracker=bool(dlib_opts.get("use_tracker", True)),
            tracker_type=str(dlib_opts.get("tracker_type", "MOSSE")),
            lost_face_grace_frames=int(dlib_opts.get("lost_face_grace_frames", 8)),
        )
    raise ValueError(f"Unsupported backend '{name}'.")
