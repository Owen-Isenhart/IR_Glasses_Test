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

    def artifacts_for_backend(self, backend_name: Optional[str]) -> BaselineArtifacts:
        if not backend_name:
            return self.artifacts
        suffix = "".join(c if c.isalnum() else "_" for c in backend_name.strip().lower())
        return BaselineArtifacts(
            embedding_path=self.base_dir / f"baseline_embedding_{suffix}.npy",
            image_path=self.base_dir / f"baseline_snapshot_{suffix}.jpg",
            meta_path=self.base_dir / f"baseline_meta_{suffix}.json",
        )

    def load_embedding(self, backend_name: Optional[str] = None) -> Optional[np.ndarray]:
        artifacts = self.artifacts_for_backend(backend_name)
        if artifacts.embedding_path.exists():
            return np.load(artifacts.embedding_path)

        # Backward-compatible fallback for legacy single-baseline files.
        if backend_name and self.artifacts.embedding_path.exists():
            return np.load(self.artifacts.embedding_path)
        if not self.artifacts.embedding_path.exists():
            return None
        return np.load(self.artifacts.embedding_path)

    def save(
        self,
        embedding: np.ndarray,
        snapshot_bgr: np.ndarray,
        meta: dict,
        backend_name: Optional[str] = None,
    ) -> None:
        artifacts = self.artifacts_for_backend(backend_name)
        np.save(artifacts.embedding_path, embedding.astype(np.float32))
        cv2.imwrite(str(artifacts.image_path), snapshot_bgr)
        with artifacts.meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

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
