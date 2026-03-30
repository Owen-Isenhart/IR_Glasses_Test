from .base import FaceBackend, FaceObservation
from .mediapipe_backend import MediaPipeEmbeddingBackend
from .dlib_backend import DlibFaceRecognitionBackend


def create_backend(name: str) -> FaceBackend:
    """Factory for runtime backend selection."""
    normalized = (name or "").strip().lower()
    if normalized in {"mediapipe", "mp"}:
        return MediaPipeEmbeddingBackend()
    if normalized in {"dlib", "face_recognition", "fr"}:
        return DlibFaceRecognitionBackend()
    raise ValueError(f"Unsupported backend '{name}'.")
