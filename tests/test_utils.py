# tests/test_utils.py
import numpy as np
import pytest
from core.utils import extract_landmarks, normalize_landmarks, landmarks_to_feature
from config import FEATURE_DIM


class FakeLandmark:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


def make_fake_landmarks(n=21):
    """Create n fake landmarks spread linearly so they're not degenerate."""
    return [FakeLandmark(i * 0.05, i * 0.02, 0.0) for i in range(n)]


def test_extract_landmarks_shape():
    lms = make_fake_landmarks()
    raw = extract_landmarks(lms)
    assert raw.shape == (63,)
    assert raw.dtype == np.float32


def test_normalize_landmarks_output_dim():
    raw = extract_landmarks(make_fake_landmarks())
    result = normalize_landmarks(raw)
    assert result is not None
    assert result.shape == (FEATURE_DIM,)


def test_normalize_landmarks_translation_invariant():
    lms1 = make_fake_landmarks()
    lms2 = [FakeLandmark(lm.x + 0.3, lm.y + 0.3, lm.z) for lm in lms1]
    r1 = normalize_landmarks(extract_landmarks(lms1))
    r2 = normalize_landmarks(extract_landmarks(lms2))
    np.testing.assert_allclose(r1, r2, atol=1e-5)


def test_normalize_landmarks_returns_none_for_degenerate():
    # All landmarks at same point → max_dist = 0
    lms = [FakeLandmark(0.5, 0.5, 0.0) for _ in range(21)]
    raw = extract_landmarks(lms)
    assert normalize_landmarks(raw) is None


def test_landmarks_to_feature_shape():
    lms = make_fake_landmarks()
    feat = landmarks_to_feature(lms)
    assert feat is not None
    assert feat.shape == (FEATURE_DIM,)
