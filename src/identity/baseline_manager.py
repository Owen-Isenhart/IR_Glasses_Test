from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.vision.backends.base import FaceBackend


@dataclass
class BaselineArtifacts:
    embedding_path: Path
    image_path: Path
    meta_path: Path


class BaselineManager:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts = BaselineArtifacts(
            embedding_path=self.base_dir / "baseline_embedding.npy",
            image_path=self.base_dir / "baseline_snapshot.jpg",
            meta_path=self.base_dir / "baseline_meta.json",
        )

    def artifacts_for_backend(
        self,
        backend_name: Optional[str],
        camera_index: Optional[int] = None,
    ) -> BaselineArtifacts:
        if not backend_name:
            return self.artifacts
        suffix = "".join(c if c.isalnum() else "_" for c in backend_name.strip().lower())
        if camera_index is not None:
            suffix = f"{suffix}_cam{int(camera_index)}"
        return BaselineArtifacts(
            embedding_path=self.base_dir / f"baseline_embedding_{suffix}.npy",
            image_path=self.base_dir / f"baseline_snapshot_{suffix}.jpg",
            meta_path=self.base_dir / f"baseline_meta_{suffix}.json",
        )

    def load_embedding(
        self,
        backend_name: Optional[str] = None,
        camera_index: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        embeddings = self.load_embeddings(backend_name, camera_index=camera_index)
        if embeddings is None:
            return None
        if embeddings.shape[0] == 1:
            return embeddings[0]

        # For legacy callers expecting a single vector, return normalized centroid.
        centroid = np.mean(embeddings, axis=0).astype(np.float32)
        norm = float(np.linalg.norm(centroid))
        if norm > 1e-8:
            centroid /= norm
        return centroid

    def load_embeddings(
        self,
        backend_name: Optional[str] = None,
        camera_index: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        artifacts = self.artifacts_for_backend(backend_name, camera_index=camera_index)
        backend_only_artifacts = self.artifacts_for_backend(backend_name)
        loaded: Optional[np.ndarray] = None
        if artifacts.embedding_path.exists():
            loaded = np.load(artifacts.embedding_path)
        elif camera_index is not None and backend_only_artifacts.embedding_path.exists():
            # Backward-compatible fallback for backend-only files from older versions.
            loaded = np.load(backend_only_artifacts.embedding_path)
        elif backend_name and self.artifacts.embedding_path.exists():
            # Backward-compatible fallback for legacy single-baseline files.
            loaded = np.load(self.artifacts.embedding_path)
        elif self.artifacts.embedding_path.exists():
            loaded = np.load(self.artifacts.embedding_path)

        if loaded is None:
            return None

        arr = np.asarray(loaded, dtype=np.float32)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        if arr.ndim == 2:
            return arr
        raise ValueError(f"Unexpected baseline embedding shape: {arr.shape}")

    def save(
        self,
        embedding: np.ndarray,
        snapshot_bgr: np.ndarray,
        meta: dict,
        backend_name: Optional[str] = None,
        camera_index: Optional[int] = None,
        append: bool = False,
    ) -> None:
        artifacts = self.artifacts_for_backend(backend_name, camera_index=camera_index)
        new_embedding = np.asarray(embedding, dtype=np.float32).reshape(1, -1)

        if append:
            existing = self.load_embeddings(backend_name, camera_index=camera_index)
            if existing is not None:
                if existing.shape[1] != new_embedding.shape[1]:
                    raise ValueError(
                        "Cannot append baseline embedding with different dimension "
                        f"({new_embedding.shape[1]} vs existing {existing.shape[1]})"
                    )
                to_save = np.concatenate([existing, new_embedding], axis=0)
            else:
                to_save = new_embedding
        else:
            to_save = new_embedding

        np.save(artifacts.embedding_path, to_save.astype(np.float32))
        cv2.imwrite(str(artifacts.image_path), snapshot_bgr)
        meta_to_write = dict(meta)
        meta_to_write["baseline_count"] = int(to_save.shape[0])
        with artifacts.meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta_to_write, f, indent=2)

    def capture_baseline(
        self,
        frames: List[np.ndarray],
        backend: FaceBackend,
        outlier_sigma: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        embeddings: List[np.ndarray] = []
        best_frame = None
        best_area = -1

        for frame in frames:
            obs = backend.observe(frame)
            if not obs.found or obs.embedding is None:
                continue
            embeddings.append(obs.embedding)
            if obs.bbox:
                x1, y1, x2, y2 = obs.bbox
                area = max(0, (x2 - x1) * (y2 - y1))
                if area > best_area:
                    best_area = area
                    best_frame = frame.copy()

        if not embeddings:
            raise RuntimeError("No usable face observations during baseline capture")

        stack = np.stack(embeddings, axis=0)
        mean = np.mean(stack, axis=0)
        dist = np.linalg.norm(stack - mean, axis=1)
        keep = dist <= (np.mean(dist) + outlier_sigma * np.std(dist))
        filtered = stack[keep] if np.any(keep) else stack

        baseline = np.mean(filtered, axis=0).astype(np.float32)
        norm = float(np.linalg.norm(baseline))
        if norm > 1e-8:
            baseline /= norm

        meta = {
            "samples": len(embeddings),
            "samples_kept": int(filtered.shape[0]),
            "outlier_sigma": outlier_sigma,
            "embedding_dim": int(baseline.shape[0]),
        }
        snapshot = best_frame if best_frame is not None else frames[-1]
        return baseline, snapshot, meta
