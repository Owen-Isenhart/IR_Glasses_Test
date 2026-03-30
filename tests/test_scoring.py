import numpy as np

from src.identity.scoring import (
    AutoCalConfig,
    AutoCalibrator,
    DistanceSmoother,
    Thresholds,
    classify_distance,
    cosine_distance,
)


def test_cosine_distance_identity():
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert cosine_distance(v, v) < 1e-6


def test_classification_boundaries():
    t = Thresholds(match_max=0.4, blocked_max=0.6)
    assert classify_distance(0.3, t) == "matched"
    assert classify_distance(0.5, t) == "blocked"
    assert classify_distance(0.8, t) == "blocked"


def test_smoother_monotonic_centering():
    s = DistanceSmoother(alpha=0.5, rolling=3)
    a = s.update(0.2)
    b = s.update(0.4)
    c = s.update(0.6)
    assert a <= b <= c


def test_autocalibrator_updates_thresholds():
    t = Thresholds(match_max=0.4, blocked_max=0.6)
    ac = AutoCalibrator(
        AutoCalConfig(min_samples=3, window=10, update_every=1, match_margin=0.04, blocked_margin=0.1)
    )
    assert not ac.update(0.33, t)
    assert not ac.update(0.34, t)
    assert ac.update(0.35, t)
    assert t.match_max != 0.4
    assert t.blocked_max != 0.6
    assert t.blocked_max > t.match_max
