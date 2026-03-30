from pathlib import Path

import numpy as np
import pytest

from src.identity.baseline_manager import BaselineManager


class FakeBackend:
    def __init__(self):
        self.counter = 0

    def observe(self, frame):
        self.counter += 1
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return type(
            "Obs",
            (),
            {
                "found": True,
                "bbox": (0, 0, 10, 10),
                "embedding": emb,
            },
        )


def test_capture_and_save(tmp_path: Path):
    mgr = BaselineManager(tmp_path)
    backend = FakeBackend()
    frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(5)]

    baseline, snapshot, meta = mgr.capture_baseline(frames, backend)
    mgr.save(baseline, snapshot, meta)

    loaded = mgr.load_embedding()
    assert loaded is not None
    assert loaded.shape[0] == 3
    assert mgr.artifacts.image_path.exists()
    assert mgr.artifacts.meta_path.exists()


def test_backend_specific_baselines(tmp_path: Path):
    mgr = BaselineManager(tmp_path)
    emb_mp = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb_dlib = np.array([0.0, 1.0], dtype=np.float32)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)

    mgr.save(emb_mp, frame, {"backend": "mediapipe"}, backend_name="mediapipe")
    mgr.save(emb_dlib, frame, {"backend": "dlib"}, backend_name="dlib")

    loaded_mp = mgr.load_embedding("mediapipe")
    loaded_dlib = mgr.load_embedding("dlib")

    assert loaded_mp is not None and loaded_mp.shape[0] == 3
    assert loaded_dlib is not None and loaded_dlib.shape[0] == 2


def test_backend_load_falls_back_to_legacy(tmp_path: Path):
    mgr = BaselineManager(tmp_path)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    mgr.save(np.array([1.0, 2.0], dtype=np.float32), frame, {"legacy": True})

    loaded = mgr.load_embedding("mediapipe")
    assert loaded is not None
    assert loaded.shape[0] == 2


def test_append_multiple_baselines_and_load_set(tmp_path: Path):
    mgr = BaselineManager(tmp_path)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)

    emb_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    mgr.save(emb_a, frame, {"capture": 1}, backend_name="mediapipe", append=False)
    mgr.save(emb_b, frame, {"capture": 2}, backend_name="mediapipe", append=True)

    loaded = mgr.load_embeddings("mediapipe")
    assert loaded is not None
    assert loaded.shape == (2, 3)
    assert np.allclose(loaded[0], emb_a)
    assert np.allclose(loaded[1], emb_b)


def test_append_rejects_dimension_mismatch(tmp_path: Path):
    mgr = BaselineManager(tmp_path)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)

    mgr.save(np.array([1.0, 0.0, 0.0], dtype=np.float32), frame, {"capture": 1}, append=False)

    with pytest.raises(ValueError):
        mgr.save(np.array([1.0, 0.0], dtype=np.float32), frame, {"capture": 2}, append=True)


def test_camera_specific_baseline_sets_are_isolated(tmp_path: Path):
    mgr = BaselineManager(tmp_path)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)

    emb_cam0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb_cam2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    mgr.save(emb_cam0, frame, {"camera_index": 0}, backend_name="mediapipe", camera_index=0)
    mgr.save(emb_cam2, frame, {"camera_index": 2}, backend_name="mediapipe", camera_index=2)

    loaded_cam0 = mgr.load_embeddings("mediapipe", camera_index=0)
    loaded_cam2 = mgr.load_embeddings("mediapipe", camera_index=2)

    assert loaded_cam0 is not None and loaded_cam0.shape == (1, 3)
    assert loaded_cam2 is not None and loaded_cam2.shape == (1, 3)
    assert np.allclose(loaded_cam0[0], emb_cam0)
    assert np.allclose(loaded_cam2[0], emb_cam2)


def test_camera_specific_load_falls_back_to_backend_only(tmp_path: Path):
    mgr = BaselineManager(tmp_path)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    emb = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    mgr.save(emb, frame, {"backend": "mediapipe"}, backend_name="mediapipe")

    loaded = mgr.load_embeddings("mediapipe", camera_index=2)
    assert loaded is not None
    assert loaded.shape == (1, 3)
    assert np.allclose(loaded[0], emb)
