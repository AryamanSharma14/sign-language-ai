# core/predictor.py
"""
Model inference wrapper with Exponential Weighted Moving Average (EWMA) smoothing.

Usage:
    import joblib
    from core.predictor import EWMAPredictor
    from config import MODEL_PATH, EWMA_ALPHA, CONFIDENCE_THRESHOLD

    payload = joblib.load(MODEL_PATH)
    pred = EWMAPredictor(
        payload["model"], payload["label_map"],
        alpha=EWMA_ALPHA, threshold=CONFIDENCE_THRESHOLD,
    )
    label, conf = pred.predict(feature_array)   # feature_array shape: (66,)
"""

import numpy as np
from config import EWMA_ALPHA, CONFIDENCE_THRESHOLD


class EWMAPredictor:
    """
    Wraps an sklearn classifier with EWMA confidence smoothing.

    Each call to predict() returns the current gesture label and its smoothed
    confidence. When smoothed confidence is below threshold, returns ("", 0.0).

    Args:
        model:      Fitted sklearn classifier with predict() and predict_proba().
        label_map:  Dict[int, str] mapping class index -> gesture label string.
        alpha:      EWMA decay factor (0 < alpha <= 1). Higher = less smoothing.
                    Defaults to EWMA_ALPHA from config.
        threshold:  Minimum smoothed confidence to emit a gesture.
                    Defaults to 0.0 (no gating); pass CONFIDENCE_THRESHOLD for
                    production use so that noisy single-frame detections are
                    suppressed only after explicit opt-in.
    """

    def __init__(self, model, label_map: dict, alpha: float = EWMA_ALPHA,
                 threshold: float = 0.0):
        self.model = model
        self.label_map = label_map
        self.alpha = alpha
        self.threshold = threshold
        self._smoothed_conf: float = 0.0
        self._current_label: str = ""

    def predict(self, feature: np.ndarray) -> tuple[str, float]:
        """
        Run inference on a single feature vector.

        Args:
            feature: numpy array of shape (FEATURE_DIM,)

        Returns:
            (label, smoothed_confidence) -- label is "" if below threshold.
        """
        x = feature.reshape(1, -1)
        idx = int(self.model.predict(x)[0])
        label = self.label_map.get(idx, "?")
        raw_conf = float(self.model.predict_proba(x)[0][
            list(self.model.classes_).index(idx)
        ])

        # If label changed, reset smoothing to avoid blending two gestures
        if label != self._current_label:
            self._smoothed_conf = 0.0
            self._current_label = label

        self._smoothed_conf = (
            self.alpha * raw_conf + (1 - self.alpha) * self._smoothed_conf
        )

        if self._smoothed_conf < self.threshold:
            return "", 0.0

        return label, self._smoothed_conf

    def reset(self):
        """Clear smoothing state (call when hand disappears from frame)."""
        self._smoothed_conf = 0.0
        self._current_label = ""
