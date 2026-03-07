# tests/test_predictor.py
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from core.predictor import EWMAPredictor


def make_mock_model(label_idx=0, confidence=0.9):
    model = MagicMock()
    model.predict.return_value = np.array([label_idx])
    model.classes_ = np.arange(11)
    proba = np.zeros(11)
    proba[label_idx] = confidence
    model.predict_proba.return_value = proba.reshape(1, -1)
    return model


def make_mock_label_map():
    labels = ["A", "B", "C", "D", "L", "Y", "PEACE", "OK", "STOP", "HELP", "THANK_YOU"]
    return {i: l for i, l in enumerate(labels)}


def test_predict_returns_label_and_confidence():
    pred = EWMAPredictor(make_mock_model(0, 0.9), make_mock_label_map(), alpha=0.3)
    feature = np.zeros(66, dtype=np.float32)
    label, conf = pred.predict(feature)
    assert label == "A"
    assert 0.0 < conf <= 1.0


def test_ewma_smooths_confidence():
    pred = EWMAPredictor(make_mock_model(0, 0.9), make_mock_label_map(), alpha=0.3)
    feature = np.zeros(66, dtype=np.float32)
    # Run 10 frames — confidence should converge toward 0.9
    for _ in range(10):
        label, conf = pred.predict(feature)
    assert conf > 0.7  # converged upward


def test_reset_clears_state():
    pred = EWMAPredictor(make_mock_model(0, 0.9), make_mock_label_map(), alpha=0.3)
    feature = np.zeros(66, dtype=np.float32)
    pred.predict(feature)
    pred.reset()
    label, conf = pred.predict(feature)
    # After reset, first prediction has low smoothed confidence (alpha * raw)
    assert conf < 0.9


def test_below_threshold_returns_empty():
    pred = EWMAPredictor(
        make_mock_model(0, 0.1), make_mock_label_map(),
        alpha=1.0, threshold=0.6
    )
    feature = np.zeros(66, dtype=np.float32)
    label, conf = pred.predict(feature)
    assert label == ""
    assert conf == 0.0
