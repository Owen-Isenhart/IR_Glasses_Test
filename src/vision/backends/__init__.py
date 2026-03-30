from .base import FaceBackend, FaceObservation
from .mediapipe_backend import MediaPipeEmbeddingBackend
from .dlib_backend import DlibFaceRecognitionBackend


def create_backend(name: str, options: dict | None = None) -> FaceBackend:
    """Factory for runtime backend selection."""
    normalized = (name or "").strip().lower()
    options = options or {}
    if normalized in {"mediapipe", "mp"}:
        return MediaPipeEmbeddingBackend()
    if normalized in {"dlib", "face_recognition", "fr"}:
        dlib_opts = options.get("dlib", {}) if isinstance(options, dict) else {}
        if not isinstance(dlib_opts, dict):
            dlib_opts = {}
        return DlibFaceRecognitionBackend(
            detect_every_n_frames=int(dlib_opts.get("detect_every_n_frames", 3)),
            detection_scale=float(dlib_opts.get("detection_scale", 0.5)),
            use_tracker=bool(dlib_opts.get("use_tracker", True)),
            tracker_type=str(dlib_opts.get("tracker_type", "MOSSE")),
        )
    raise ValueError(f"Unsupported backend '{name}'.")
