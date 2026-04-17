# Sign Language AI — Full Overhaul Implementation Plan

**Goal:** Restructure the codebase, add 6 new gestures, improve accuracy with palm-orientation features + augmentation + GridSearchCV + EWMA smoothing, and add a Flask web dashboard with live MJPEG feed and real-time stats.

**Architecture:** A `core/` package owns all ML logic (single source of truth for labels, feature extraction, inference). A `web/` package runs Flask with a background camera thread pushing MJPEG frames and WebSocket JSON events independently. All tunables live in `config.py`.

**Tech Stack:** Python 3.10+, OpenCV, MediaPipe Tasks API, scikit-learn, joblib, numpy, Flask, flask-sock, Chart.js (CDN)

---

## Prerequisites

- Working directory: `sign-language-ai/`
- `venv` is already set up. Activate it before running any commands:
  - Windows: `venv\Scripts\activate`
  - Linux/Pi: `source venv/bin/activate`
- `hand_landmarker.task` must stay in the project root (MediaPipe model file)

---

## Task 1: Create directory skeleton

**Files:**
- Create: `core/__init__.py`
- Create: `training/__init__.py`
- Create: `hardware/__init__.py`
- Create: `inference/__init__.py`
- Create: `web/__init__.py`
- Create: `web/static/css/` (directory)
- Create: `web/static/js/` (directory)
- Create: `web/templates/` (directory)
- Create: `models/` (directory)
- Create: `tests/__init__.py`

**Step 1: Create all directories and empty __init__.py files**

```bash
mkdir -p core training hardware inference web/static/css web/static/js web/templates models tests
touch core/__init__.py training/__init__.py hardware/__init__.py inference/__init__.py web/__init__.py tests/__init__.py
```

**Step 2: Move model file to models/**

```bash
mv model.pkl models/model.pkl
```

**Step 3: Verify structure**

```bash
ls core/ training/ hardware/ inference/ web/ models/ tests/
```

Expected: each directory exists, `core/`, `training/`, `hardware/`, `inference/`, `tests/` each contain `__init__.py`.

**Step 4: Commit**

```bash
git add core/ training/ hardware/ inference/ web/ models/ tests/
git commit -m "chore: create package directory skeleton"
```

---

## Task 2: Create config.py

**Files:**
- Create: `config.py`

**Step 1: Create config.py with all tunables**

```python
# config.py
"""Single source of truth for all configurable values."""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR      = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH  = os.path.join(ROOT_DIR, "dataset", "gestures.csv")
MODEL_PATH    = os.path.join(ROOT_DIR, "models", "model.pkl")
MP_MODEL_PATH = os.path.join(ROOT_DIR, "hand_landmarker.task")

# ── Features ──────────────────────────────────────────────────────────────────
FEATURE_DIM   = 66   # 21 landmarks × 3 coords + 3 palm-normal coords

# ── MediaPipe ─────────────────────────────────────────────────────────────────
MP_NUM_HANDS              = 1
MP_DETECTION_CONFIDENCE   = 0.7
MP_PRESENCE_CONFIDENCE    = 0.5
MP_TRACKING_CONFIDENCE    = 0.6

# ── Inference / smoothing ─────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.60   # minimum smoothed confidence to emit a gesture
EWMA_ALPHA           = 0.3    # exponential weighted moving average decay

# ── Training ──────────────────────────────────────────────────────────────────
TEST_SIZE        = 0.2
RANDOM_STATE     = 42
JITTER_SIGMA     = 0.01   # std dev for Gaussian jitter augmentation
JITTER_COPIES    = 2      # how many jitter copies per sample
SVC_PARAM_GRID   = {
    "C":     [1, 10, 100],
    "gamma": ["scale", "auto", 0.01, 0.001],
}
CV_FOLDS = 5

# ── Camera ────────────────────────────────────────────────────────────────────
CAM_INDEX_PC  = 0
CAM_BACKEND   = "dshow"   # "dshow" on Windows, "" on Linux
RES_PC        = (640, 480)
RES_EMBEDDED  = (320, 240)
FPS_CAP_PC    = None      # uncapped
FPS_CAP_EMB   = 10

# ── Web dashboard ─────────────────────────────────────────────────────────────
WEB_HOST        = "0.0.0.0"
WEB_PORT        = 5000
MJPEG_QUALITY   = 70      # JPEG compression quality (0–100)
WS_EVENT_RATE   = 0.1     # minimum seconds between WebSocket gesture events
HISTORY_MAX     = 50      # max rows in gesture history log

# ── Edge inference ────────────────────────────────────────────────────────────
EDGE_FPS_CAP    = 15
EDGE_SKIP_FRAMES = 2
```

**Step 2: Verify it imports cleanly**

```bash
python -c "import config; print('FEATURE_DIM:', config.FEATURE_DIM)"
```

Expected: `FEATURE_DIM: 66`

**Step 3: Commit**

```bash
git add config.py
git commit -m "feat: add config.py as single source of truth for all tunables"
```

---

## Task 3: Create core/gestures.py

**Files:**
- Create: `core/gestures.py`
- Create: `tests/test_gestures.py`

**Step 1: Write the failing test**

```python
# tests/test_gestures.py
from core.gestures import GESTURE_LABELS, GESTURE_PIN_MAP, COMMAND_MAP

def test_all_labels_have_pin():
    for label in GESTURE_LABELS:
        assert label in GESTURE_PIN_MAP, f"{label} missing from GESTURE_PIN_MAP"

def test_all_labels_have_command():
    for label in GESTURE_LABELS:
        assert label in COMMAND_MAP, f"{label} missing from COMMAND_MAP"

def test_pins_are_unique():
    pins = list(GESTURE_PIN_MAP.values())
    assert len(pins) == len(set(pins)), "Duplicate GPIO pins detected"

def test_gesture_count():
    assert len(GESTURE_LABELS) == 11
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_gestures.py -v
```

Expected: `FAILED` — `ModuleNotFoundError: No module named 'core.gestures'`

**Step 3: Create core/gestures.py**

```python
# core/gestures.py
"""Single source of truth for gesture labels, GPIO pins, and command translations."""

GESTURE_LABELS = [
    "A", "B", "C", "D", "L", "Y",
    "PEACE", "OK", "STOP", "HELP", "THANK_YOU",
]

GESTURE_PIN_MAP = {
    "A":         17,
    "B":         27,
    "C":         22,
    "D":          5,
    "L":          6,
    "Y":         13,
    "PEACE":     23,
    "OK":        24,
    "STOP":      19,
    "HELP":      26,
    "THANK_YOU": 21,
}

COMMAND_MAP = {
    "A":         "YES",
    "B":         "NO",
    "C":         "CONFIRM",
    "D":         "DOWN",
    "L":         "LETTER_L",
    "Y":         "LETTER_Y",
    "PEACE":     "HELLO",
    "OK":        "OK",
    "STOP":      "STOP",
    "HELP":      "HELP",
    "THANK_YOU": "THANKS",
}
```

**Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_gestures.py -v
```

Expected: `4 passed`

**Step 5: Commit**

```bash
git add core/gestures.py tests/test_gestures.py
git commit -m "feat: add core/gestures.py — labels, GPIO pins, command map for 11 gestures"
```

---

## Task 4: Create core/utils.py (with palm orientation)

**Files:**
- Create: `core/utils.py`
- Create: `tests/test_utils.py`
- Delete: `utils.py` (old root file)

**Step 1: Write the failing tests**

```python
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
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_utils.py -v
```

Expected: `FAILED` — `ModuleNotFoundError: No module named 'core.utils'`

**Step 3: Create core/utils.py**

```python
# core/utils.py
"""Landmark extraction, normalization, and palm-orientation feature computation."""

import numpy as np
from config import FEATURE_DIM


def extract_landmarks(hand_landmarks) -> np.ndarray:
    """Extract raw x/y/z coords from 21 MediaPipe landmarks → shape (63,)."""
    coords = []
    for lm in hand_landmarks:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32)


def _palm_normal(points: np.ndarray) -> np.ndarray:
    """
    Compute palm-normal vector from wrist-relative landmark positions.
    Uses cross product of (wrist→index_mcp) and (wrist→pinky_mcp).
    Landmarks 5 = index MCP, 17 = pinky MCP (0-indexed after wrist subtraction).
    Returns unit vector of shape (3,). Returns zeros if degenerate.
    """
    v1 = points[5]   # index MCP (wrist-relative)
    v2 = points[17]  # pinky MCP (wrist-relative)
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm < 1e-6:
        return np.zeros(3, dtype=np.float32)
    return (normal / norm).astype(np.float32)


def normalize_landmarks(raw: np.ndarray) -> np.ndarray | None:
    """
    Normalize landmarks to be translation- and scale-invariant, then append
    the palm-normal vector for rotation awareness.

    Steps:
      1. Subtract wrist (landmark 0) → translation invariant
      2. Divide by max Euclidean distance from wrist → scale invariant
      3. Append 3-element palm-normal vector (cross product of two palm edges)

    Returns array of shape (66,), or None if hand is degenerate.
    """
    points = raw.reshape(21, 3)
    wrist = points[0].copy()
    points = points - wrist

    dists = np.linalg.norm(points, axis=1)
    max_dist = dists.max()
    if max_dist < 1e-6:
        return None

    points = points / max_dist
    normal = _palm_normal(points)

    return np.concatenate([points.flatten(), normal]).astype(np.float32)


def landmarks_to_feature(hand_landmarks) -> np.ndarray | None:
    """Extract landmarks then normalize. Returns array of shape (FEATURE_DIM,) or None."""
    raw = extract_landmarks(hand_landmarks)
    return normalize_landmarks(raw)
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_utils.py -v
```

Expected: `5 passed`

**Step 5: Delete the old root utils.py**

```bash
rm utils.py
```

**Step 6: Commit**

```bash
git add core/utils.py tests/test_utils.py
git rm utils.py
git commit -m "feat: add core/utils.py with palm-orientation feature (66-dim output)"
```

---

## Task 5: Create core/predictor.py (EWMA smoothing)

**Files:**
- Create: `core/predictor.py`
- Create: `tests/test_predictor.py`

**Step 1: Write the failing tests**

```python
# tests/test_predictor.py
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from core.predictor import EWMAPredictor


def make_mock_model(label_idx=0, confidence=0.9):
    model = MagicMock()
    model.predict.return_value = np.array([label_idx])
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
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_predictor.py -v
```

Expected: `FAILED` — `ModuleNotFoundError: No module named 'core.predictor'`

**Step 3: Create core/predictor.py**

```python
# core/predictor.py
"""
Model inference wrapper with Exponential Weighted Moving Average (EWMA) smoothing.

Usage:
    import joblib
    from core.predictor import EWMAPredictor
    from config import MODEL_PATH, EWMA_ALPHA, CONFIDENCE_THRESHOLD

    payload = joblib.load(MODEL_PATH)
    pred = EWMAPredictor(payload["model"], payload["label_map"])
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
        label_map:  Dict[int, str] mapping class index → gesture label string.
        alpha:      EWMA decay factor (0 < alpha <= 1). Higher = less smoothing.
        threshold:  Minimum smoothed confidence to emit a gesture.
    """

    def __init__(self, model, label_map: dict, alpha: float = EWMA_ALPHA,
                 threshold: float = CONFIDENCE_THRESHOLD):
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
            (label, smoothed_confidence) — label is "" if below threshold.
        """
        x = feature.reshape(1, -1)
        idx = int(self.model.predict(x)[0])
        label = self.label_map.get(idx, "?")
        raw_conf = float(self.model.predict_proba(x)[0][idx])

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
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_predictor.py -v
```

Expected: `4 passed`

**Step 5: Commit**

```bash
git add core/predictor.py tests/test_predictor.py
git commit -m "feat: add core/predictor.py with EWMA confidence smoothing"
```

---

## Task 6: Move training scripts

**Files:**
- Create: `training/collect_data.py`  (replaces root `collect_data.py`)
- Create: `training/train_model.py`   (replaces root `train_model.py`, adds augmentation + GridSearchCV)
- Delete: `collect_data.py` (root)
- Delete: `train_model.py` (root)

**Step 1: Create training/collect_data.py**

Copy the existing `collect_data.py` but update these three things:
1. Imports: `from core.utils import landmarks_to_feature` and `from core.gestures import GESTURE_LABELS`
2. Paths: use `config.DATASET_PATH` and `config.MP_MODEL_PATH`
3. CSV header: write `FEATURE_DIM` (66) feature columns instead of 63

```python
# training/collect_data.py
"""
Collect gesture data from webcam and save to dataset/gestures.csv.

Usage:
    python -m training.collect_data A
    python -m training.collect_data THANK_YOU

Controls:
    s - Save current frame's landmarks as a sample
    q - Quit
"""

import os
import sys
import csv
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarksConnections

import config
from core.gestures import GESTURE_LABELS
from core.utils import landmarks_to_feature

CONNECTIONS = [(c.start, c.end) for c in HandLandmarksConnections.HAND_CONNECTIONS]
FLASH_DURATION = 15


def draw_landmarks(frame, landmarks, w, h):
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for start, end in CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 200, 255), 2)
    for pt in points:
        cv2.circle(frame, pt, 4, (0, 255, 128), -1)


def get_label_from_user() -> str:
    if len(sys.argv) >= 2:
        label = sys.argv[1].strip().upper()
        if label in GESTURE_LABELS:
            return label
        print(f"Invalid label '{label}'. Choose from: {GESTURE_LABELS}")
        sys.exit(1)
    print(f"Available gestures: {GESTURE_LABELS}")
    while True:
        label = input("Enter gesture label to collect: ").strip().upper()
        if label in GESTURE_LABELS:
            return label
        print(f"  Invalid label. Choose from: {GESTURE_LABELS}")


def ensure_csv_exists():
    os.makedirs(os.path.dirname(config.DATASET_PATH), exist_ok=True)
    if not os.path.exists(config.DATASET_PATH):
        with open(config.DATASET_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["label"] + [f"f{i}" for i in range(config.FEATURE_DIM)]
            writer.writerow(header)


def main():
    label = get_label_from_user()
    ensure_csv_exists()

    base_options = mp_python.BaseOptions(model_asset_path=config.MP_MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=config.MP_NUM_HANDS,
        min_hand_detection_confidence=config.MP_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=config.MP_PRESENCE_CONFIDENCE,
        min_tracking_confidence=config.MP_TRACKING_CONFIDENCE,
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)

    sample_count = 0
    flash_counter = 0
    start_time = time.time()

    cap = cv2.VideoCapture(config.CAM_INDEX_PC, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(config.CAM_INDEX_PC)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            timestamp_ms = int((time.time() - start_time) * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            feature = None
            if result.hand_landmarks:
                lms = result.hand_landmarks[0]
                draw_landmarks(frame, lms, w, h)
                feature = landmarks_to_feature(lms)

            cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
            cv2.putText(frame, f"Gesture: {label}", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Samples: {sample_count}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

            status_color = (0, 255, 0) if feature is not None else (0, 100, 255)
            status_text = "Hand detected" if feature is not None else "No hand"
            cv2.putText(frame, status_text, (w - 200, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

            if flash_counter > 0:
                cv2.putText(frame, "SAVED!", (w // 2 - 60, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 128), 3)
                flash_counter -= 1

            cv2.putText(frame, "s=save  q=quit", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            cv2.imshow("Collect Gesture Data", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if feature is not None:
                    with open(config.DATASET_PATH, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([label] + feature.tolist())
                    sample_count += 1
                    flash_counter = FLASH_DURATION
                else:
                    print("No hand detected — nothing saved.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print(f"\nDone. Collected {sample_count} samples for '{label}'.")
        print(f"Data saved to: {config.DATASET_PATH}")


if __name__ == "__main__":
    main()
```

**Step 2: Create training/train_model.py**

```python
# training/train_model.py
"""
Train gesture classifier from dataset/gestures.csv.

Usage:
    python -m training.train_model

Output:
    models/model.pkl  — best SVC model + label_map + best_params bundled together
"""

import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import config
from core.gestures import GESTURE_LABELS


def load_dataset():
    data = np.genfromtxt(config.DATASET_PATH, delimiter=",", dtype=str, skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    labels_str = data[:, 0]
    features = data[:, 1:].astype(np.float32)
    label_to_int = {label: idx for idx, label in enumerate(GESTURE_LABELS)}
    y = np.array([label_to_int[l] for l in labels_str], dtype=np.int32)
    return features, y


def augment(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Data augmentation:
      1. Mirror: flip x-coords of all 21 landmarks (simulates left hand)
      2. Jitter: add Gaussian noise × JITTER_COPIES copies

    Feature layout: indices 0,3,6,...,60 are x-coords of landmarks 0-20.
    The palm-normal (indices 63-65) is left as-is for jitter, zeroed for mirror.
    """
    rng = np.random.default_rng(config.RANDOM_STATE)
    aug_X, aug_y = [X], [y]

    # Mirror: negate x-coords (every 3rd value starting at 0, for first 63 features)
    mirrored = X.copy()
    mirrored[:, 0:63:3] *= -1      # flip x of 21 landmarks
    mirrored[:, 63:66] = 0.0       # palm normal undefined after flip
    aug_X.append(mirrored)
    aug_y.append(y)

    # Jitter
    for _ in range(config.JITTER_COPIES):
        noise = rng.normal(0, config.JITTER_SIGMA, X.shape).astype(np.float32)
        aug_X.append(X + noise)
        aug_y.append(y)

    return np.concatenate(aug_X), np.concatenate(aug_y)


def main():
    print("Loading dataset...")
    X, y = load_dataset()
    print(f"  {X.shape[0]} samples, {X.shape[1]} features")

    unique = np.unique(y)
    print(f"  Classes found: {[GESTURE_LABELS[i] for i in unique]}")
    if len(unique) < 2:
        print("ERROR: Need at least 2 gesture classes to train.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y,
        random_state=config.RANDOM_STATE,
    )

    print(f"\nAugmenting training data...")
    X_train_aug, y_train_aug = augment(X_train, y_train)
    print(f"  Before: {len(X_train)}  After: {len(X_train_aug)}")

    print("\nRunning GridSearchCV (SVC, rbf kernel)...")
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)
    grid = GridSearchCV(
        SVC(kernel="rbf", probability=True),
        config.SVC_PARAM_GRID,
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train_aug, y_train_aug)
    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV score: {grid.best_score_:.4f}")

    best_svc = grid.best_estimator_
    y_pred = best_svc.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    print(classification_report(
        y_test, y_pred,
        target_names=[GESTURE_LABELS[i] for i in unique],
        zero_division=0,
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    label_map = {idx: label for idx, label in enumerate(GESTURE_LABELS)}
    payload = {
        "model": best_svc,
        "label_map": label_map,
        "best_params": grid.best_params_,
        "feature_dim": config.FEATURE_DIM,
    }
    joblib.dump(payload, config.MODEL_PATH)
    print(f"\nSaved model to '{config.MODEL_PATH}'")
    print(f"Label map: {label_map}")


if __name__ == "__main__":
    main()
```

**Step 3: Delete old root files**

```bash
rm collect_data.py train_model.py
```

**Step 4: Verify imports work**

```bash
python -c "from training.collect_data import ensure_csv_exists; print('OK')"
python -c "from training.train_model import augment; import numpy as np; X=np.ones((10,66),dtype='f'); y=np.zeros(10,dtype='i'); Xa,ya=augment(X,y); print('Augmented shape:', Xa.shape)"
```

Expected:
```
OK
Augmented shape: (40, 66)
```

**Step 5: Commit**

```bash
git add training/collect_data.py training/train_model.py
git rm collect_data.py train_model.py
git commit -m "feat: move training scripts to training/ — add augmentation and GridSearchCV"
```

---

## Task 7: Move hardware scripts

**Files:**
- Create: `hardware/gpio_controller.py`
- Create: `hardware/platform_io.py`
- Delete: `gpio_controller.py` (root)
- Delete: `platform_io.py` (root)

**Step 1: Create hardware/gpio_controller.py**

Same logic as before, but import pin map from `core.gestures`:

```python
# hardware/gpio_controller.py
"""
GPIO controller for Raspberry Pi.
Auto-falls back to terminal simulation on non-Pi systems.
"""

from core.gestures import GESTURE_PIN_MAP


class GpioController:
    def __init__(self, simulate: bool = False):
        self.simulate = simulate
        self._gpio = None
        if not simulate:
            try:
                import RPi.GPIO as GPIO
                self._gpio = GPIO
            except ImportError:
                self.simulate = True

    def setup(self):
        if not self.simulate:
            self._gpio.setmode(self._gpio.BCM)
            for pin in GESTURE_PIN_MAP.values():
                self._gpio.setup(pin, self._gpio.OUT)
                self._gpio.output(pin, self._gpio.LOW)
        pins = list(GESTURE_PIN_MAP.values())
        status = "(simulated)" if self.simulate else ""
        print(f"[GPIO] Pins {pins} configured as OUTPUT {status}")

    def set_gesture(self, label: str):
        pin = GESTURE_PIN_MAP.get(label)
        if pin is None:
            return
        if not self.simulate:
            for g, p in GESTURE_PIN_MAP.items():
                val = self._gpio.HIGH if g == label else self._gpio.LOW
                self._gpio.output(p, val)
        sim = " (sim)" if self.simulate else ""
        print(f"[GPIO] Pin {pin} HIGH → {label}{sim}")

    def all_low(self):
        if not self.simulate:
            for pin in GESTURE_PIN_MAP.values():
                self._gpio.output(pin, self._gpio.LOW)

    def cleanup(self):
        self.all_low()
        if not self.simulate:
            self._gpio.cleanup()
        print("[GPIO] Cleanup done")
```

**Step 2: Create hardware/platform_io.py**

Same logic as before, but import from `core.gestures` and `config`:

```python
# hardware/platform_io.py
"""Hardware Abstraction Layer — routes gesture events to GPIO, Serial, or terminal."""

import os
import time

from core.gestures import GESTURE_PIN_MAP
import config


def detect_platform() -> str:
    try:
        with open("/proc/device-tree/model", "r") as f:
            model = f.read()
        if "Raspberry Pi" in model:
            return "rpi"
    except (FileNotFoundError, OSError):
        pass
    return "pc"


class HardwareIO:
    def __init__(self, platform: str):
        self.platform = platform
        self._gpio = None
        self._serial = None
        self._gpio_available = False
        self._serial_available = False

        if platform == "rpi":
            try:
                import RPi.GPIO as GPIO
                self._gpio = GPIO
                self._gpio_available = True
            except ImportError:
                pass

        if platform == "arduino":
            try:
                import serial
                self._serial = serial.Serial("/dev/ttyUSB0", 9600, timeout=1)
                self._serial_available = True
            except Exception:
                pass

    def boot(self):
        names = {"rpi": "Raspberry Pi 4 Model B", "arduino": "Arduino Serial", "pc": "PC Demo"}
        print("[BOOT] Sign Language Recognition System")
        print(f"[BOOT] Platform: {names.get(self.platform, 'Unknown')}")
        print("[BOOT] Loading gesture model... OK")
        print("[BOOT] Initializing camera... OK")

        pins = ", ".join(str(p) for p in GESTURE_PIN_MAP.values())

        if self.platform == "rpi":
            if self._gpio_available:
                self._gpio.setmode(self._gpio.BCM)
                for pin in GESTURE_PIN_MAP.values():
                    self._gpio.setup(pin, self._gpio.OUT)
                    self._gpio.output(pin, self._gpio.LOW)
                print(f"[GPIO] Pins {pins} configured as OUTPUT")
            else:
                print(f"[GPIO] Pins {pins} configured as OUTPUT  (simulated)")
        elif self.platform == "arduino":
            status = "sent" if self._serial_available else "simulated"
            print(f"[SERIAL] Port /dev/ttyUSB0 @ 9600 baud  ({status})")
        else:
            print(f"[GPIO] Pins {pins} configured as OUTPUT  (simulated)")

    def emit_gesture(self, label: str, confidence: float):
        ts = time.strftime("%H:%M:%S")
        pin = GESTURE_PIN_MAP.get(label)

        if self.platform == "rpi":
            if pin is not None and self._gpio_available:
                for g, p in GESTURE_PIN_MAP.items():
                    val = self._gpio.HIGH if g == label else self._gpio.LOW
                    self._gpio.output(p, val)
            print(f"[{ts}] [GPIO] Pin {pin} HIGH  |  Gesture: {label}  |  Confidence: {confidence:.2f}")

        elif self.platform == "arduino":
            conf_int = int(confidence * 100)
            msg = f"GESTURE:{label}:{conf_int}\n"
            if self._serial_available:
                try:
                    self._serial.write(msg.encode())
                except Exception as e:
                    print(f"[{ts}] [SERIAL] Write error: {e}")
            status = "sent" if self._serial_available else "simulated"
            print(f"[{ts}] [SERIAL] {msg.strip()}  ({status})")

        else:
            color = "\033[92m" if confidence >= config.CONFIDENCE_THRESHOLD else "\033[93m"
            reset = "\033[0m"
            print(f"[{ts}] {color}[GPIO] Pin {pin} HIGH  |  Gesture: {label}  |  Confidence: {confidence:.2f}{reset}")

    def shutdown(self):
        if self.platform == "rpi":
            if self._gpio_available:
                for pin in GESTURE_PIN_MAP.values():
                    self._gpio.output(pin, self._gpio.LOW)
                self._gpio.cleanup()
            print("[GPIO] All pins LOW — cleanup done")
        elif self.platform == "arduino":
            if self._serial_available:
                try:
                    self._serial.close()
                except Exception:
                    pass
            print("[SERIAL] Port closed")
        else:
            print("[GPIO] All pins LOW (simulated)")
        print("[SYS] Shutting down...")
```

**Step 3: Delete old root files**

```bash
rm gpio_controller.py platform_io.py
```

**Step 4: Verify imports**

```bash
python -c "from hardware.platform_io import detect_platform; print(detect_platform())"
python -c "from hardware.gpio_controller import GpioController; g=GpioController(simulate=True); g.setup()"
```

Expected: `pc` and `[GPIO] Pins [...] configured as OUTPUT (simulated)`

**Step 5: Commit**

```bash
git add hardware/gpio_controller.py hardware/platform_io.py
git rm gpio_controller.py platform_io.py
git commit -m "feat: move hardware layer to hardware/ — import pins from core.gestures"
```

---

## Task 8: Move inference scripts

**Files:**
- Create: `inference/run_recognition.py`
- Create: `inference/edge_inference.py`
- Delete: `run_recognition.py` (root)
- Delete: `edge_inference.py` (root)

**Step 1: Create inference/run_recognition.py**

Update the original to use `core.predictor.EWMAPredictor` instead of the manual buffer + Counter smoothing, and fix all imports:

```python
# inference/run_recognition.py
"""
Real-time sign language gesture recognition (PC / embedded).

Usage:
    python -m inference.run_recognition
    python -m inference.run_recognition --platform rpi
    python -m inference.run_recognition --headless

Controls:
    q - Quit
    r - Reset predictor smoothing
"""

import argparse
import time
import cv2
import joblib
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarksConnections

import config
from core.utils import landmarks_to_feature
from core.predictor import EWMAPredictor
from hardware.platform_io import HardwareIO, detect_platform

CONNECTIONS = [(c.start, c.end) for c in HandLandmarksConnections.HAND_CONNECTIONS]


def draw_landmarks(frame, landmarks, w, h):
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for start, end in CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 200, 255), 2)
    for pt in points:
        cv2.circle(frame, pt, 4, (0, 255, 128), -1)


def draw_ui(frame, label, confidence, has_hand):
    h, w = frame.shape[:2]
    bar_h = 70
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    if has_hand and label:
        color = (0, 255, 0) if confidence >= config.CONFIDENCE_THRESHOLD else (0, 200, 255)
        cv2.putText(frame, f"Gesture: {label}", (15, h - bar_h + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (15, h - bar_h + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    else:
        cv2.putText(frame, "No gesture detected", (15, h - bar_h + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
    cv2.putText(frame, "q=quit  r=reset", (w - 160, h - bar_h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=["pc", "rpi", "arduino"], default=None)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    platform = args.platform or detect_platform()
    headless = args.headless or platform in ("rpi", "arduino")

    hw = HardwareIO(platform)
    hw.boot()

    print(f"Loading model from '{config.MODEL_PATH}'...")
    payload = joblib.load(config.MODEL_PATH)
    predictor = EWMAPredictor(payload["model"], payload["label_map"])

    base_options = mp_python.BaseOptions(model_asset_path=config.MP_MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=config.MP_NUM_HANDS,
        min_hand_detection_confidence=config.MP_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=config.MP_PRESENCE_CONFIDENCE,
        min_tracking_confidence=config.MP_TRACKING_CONFIDENCE,
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)

    res_w, res_h = config.RES_EMBEDDED if platform in ("rpi", "arduino") else config.RES_PC
    fps_cap = config.FPS_CAP_EMB if platform in ("rpi", "arduino") else config.FPS_CAP_PC

    cap = cv2.VideoCapture(config.CAM_INDEX_PC, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(config.CAM_INDEX_PC)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_h)

    start_time = time.time()
    frame_count = 0
    last_emitted_label = ""

    print("[SYS] Press 'q' to quit, 'r' to reset" if not headless else "[SYS] Headless — Ctrl+C to quit")

    try:
        while True:
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int((time.time() - start_time) * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            has_hand = False
            label, confidence = "", 0.0

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                if not headless:
                    draw_landmarks(frame, landmarks, w, h)
                feature = landmarks_to_feature(landmarks)
                if feature is not None:
                    has_hand = True
                    label, confidence = predictor.predict(feature)

            if not has_hand:
                predictor.reset()

            if has_hand and label and (label != last_emitted_label or frame_count % 30 == 0):
                hw.emit_gesture(label, confidence)
                last_emitted_label = label

            frame_count += 1

            if not headless:
                draw_ui(frame, label, confidence, has_hand)
                cv2.imshow("Sign Language Recognition", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    predictor.reset()
                    last_emitted_label = ""

            if fps_cap:
                sleep_time = (1.0 / fps_cap) - (time.time() - frame_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[SYS] Keyboard interrupt.")
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
        detector.close()
        hw.shutdown()


if __name__ == "__main__":
    main()
```

**Step 2: Create inference/edge_inference.py**

Read the existing `edge_inference.py`, then rewrite with updated imports (`from core.utils`, `from core.predictor`, `from hardware.gpio_controller`, all config values from `config`). The logic stays identical — only imports and hardcoded values change.

> Note: Read `edge_inference.py` before writing this file. The structure mirrors `run_recognition.py` with added frame-skip, FPS metrics, and standalone GpioController. Replace all hardcoded values with `config.*` equivalents.

```python
# inference/edge_inference.py
"""
Edge inference script for Raspberry Pi / embedded Linux.

Usage:
    python -m inference.edge_inference
    python -m inference.edge_inference --headless
    python -m inference.edge_inference --skip-frames 2 --fps-cap 15 --no-gpio
"""

import argparse
import time
import cv2
import joblib
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarksConnections
from collections import deque

import config
from core.utils import landmarks_to_feature
from core.predictor import EWMAPredictor
from core.gestures import COMMAND_MAP
from hardware.gpio_controller import GpioController

CONNECTIONS = [(c.start, c.end) for c in HandLandmarksConnections.HAND_CONNECTIONS]


def draw_landmarks(frame, landmarks, w, h):
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for start, end in CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 200, 255), 2)
    for pt in points:
        cv2.circle(frame, pt, 4, (0, 255, 128), -1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--skip-frames", type=int, default=config.EDGE_SKIP_FRAMES)
    parser.add_argument("--fps-cap", type=float, default=config.EDGE_FPS_CAP)
    parser.add_argument("--no-gpio", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    headless = args.headless
    skip_frames = args.skip_frames
    fps_cap = args.fps_cap

    gpio = GpioController(simulate=args.no_gpio)
    gpio.setup()

    print(f"Loading model from '{config.MODEL_PATH}'...")
    payload = joblib.load(config.MODEL_PATH)
    predictor = EWMAPredictor(payload["model"], payload["label_map"])

    base_options = mp_python.BaseOptions(model_asset_path=config.MP_MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=config.MP_NUM_HANDS,
        min_hand_detection_confidence=config.MP_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=config.MP_PRESENCE_CONFIDENCE,
        min_tracking_confidence=config.MP_TRACKING_CONFIDENCE,
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.RES_EMBEDDED[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.RES_EMBEDDED[1])

    fps_history: deque = deque(maxlen=30)
    start_time = time.time()
    frame_count = 0
    label, confidence = "", 0.0
    inference_ms = 0.0

    try:
        while True:
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            if frame_count % skip_frames == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp_ms = int((time.time() - start_time) * 1000)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                t0 = time.perf_counter()
                result = detector.detect_for_video(mp_image, timestamp_ms)
                inference_ms = (time.perf_counter() - t0) * 1000

                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]
                    if not headless:
                        draw_landmarks(frame, landmarks, w, h)
                    feature = landmarks_to_feature(landmarks)
                    if feature is not None:
                        label, confidence = predictor.predict(feature)
                    else:
                        predictor.reset()
                        label, confidence = "", 0.0
                else:
                    predictor.reset()
                    label, confidence = "", 0.0

                if label:
                    command = COMMAND_MAP.get(label, label)
                    gpio.set_gesture(label)
                    if headless:
                        print(f"[EDGE] Gesture: {label}  →  Command: {command}")
                        print(f"[EDGE] Confidence: {confidence:.2f} | FPS: {int(sum(fps_history)/max(len(fps_history),1))} | Inference: {inference_ms:.0f} ms")

            frame_count += 1
            frame_elapsed = time.time() - frame_start
            if frame_elapsed > 0:
                fps_history.append(1.0 / frame_elapsed)

            if not headless:
                fps_now = int(sum(fps_history) / max(len(fps_history), 1))
                command = COMMAND_MAP.get(label, "")
                cv2.putText(frame, f"Gesture: {label}  ->  {command}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Conf: {confidence:.2f}  FPS: {fps_now}  Inf: {inference_ms:.0f}ms",
                            (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.imshow("Edge Inference", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            target_frame_time = 1.0 / fps_cap if fps_cap else 0
            sleep_time = target_frame_time - (time.time() - frame_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[EDGE] Interrupted.")
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
        detector.close()
        gpio.cleanup()


if __name__ == "__main__":
    main()
```

**Step 3: Delete old root files**

```bash
rm run_recognition.py edge_inference.py
```

**Step 4: Verify imports**

```bash
python -c "from inference.run_recognition import parse_args; print('OK')"
python -c "from inference.edge_inference import parse_args; print('OK')"
```

**Step 5: Commit**

```bash
git add inference/run_recognition.py inference/edge_inference.py
git rm run_recognition.py edge_inference.py
git commit -m "feat: move inference scripts to inference/ — use EWMAPredictor and config"
```

---

## Task 9: Build web/stream.py — background camera thread

**Files:**
- Create: `web/stream.py`

This module runs the camera + MediaPipe + predictor in a background thread and exposes:
- `get_frame()` → latest JPEG-encoded frame bytes (for MJPEG)
- `get_latest_event()` → latest gesture event dict (for WebSocket)

```python
# web/stream.py
"""
Background camera thread for the Flask dashboard.

Runs MediaPipe + EWMAPredictor in a daemon thread.
Exposes get_frame() for MJPEG and get_latest_event() for WebSocket.
"""

import threading
import time
import cv2
import joblib
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarksConnections
from collections import deque

import config
from core.utils import landmarks_to_feature
from core.predictor import EWMAPredictor
from core.gestures import COMMAND_MAP

CONNECTIONS = [(c.start, c.end) for c in HandLandmarksConnections.HAND_CONNECTIONS]

_lock = threading.Lock()
_frame_bytes: bytes = b""
_latest_event: dict = {}
_fps_history: deque = deque(maxlen=30)


def _draw_landmarks(frame, landmarks, w, h):
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for start, end in CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 200, 255), 2)
    for pt in points:
        cv2.circle(frame, pt, 4, (0, 255, 128), -1)


def _camera_loop():
    global _frame_bytes, _latest_event

    payload = joblib.load(config.MODEL_PATH)
    predictor = EWMAPredictor(payload["model"], payload["label_map"])

    base_options = mp_python.BaseOptions(model_asset_path=config.MP_MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=config.MP_NUM_HANDS,
        min_hand_detection_confidence=config.MP_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=config.MP_PRESENCE_CONFIDENCE,
        min_tracking_confidence=config.MP_TRACKING_CONFIDENCE,
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(config.CAM_INDEX_PC, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(config.CAM_INDEX_PC)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.RES_PC[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.RES_PC[1])

    start_time = time.time()
    last_event_time = 0.0

    try:
        while True:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_ms = int((time.time() - start_time) * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            t0 = time.perf_counter()
            result = detector.detect_for_video(mp_image, timestamp_ms)
            inference_ms = (time.perf_counter() - t0) * 1000

            label, confidence = "", 0.0
            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                _draw_landmarks(frame, landmarks, w, h)
                feature = landmarks_to_feature(landmarks)
                if feature is not None:
                    label, confidence = predictor.predict(feature)
                else:
                    predictor.reset()
            else:
                predictor.reset()

            # Encode frame as JPEG
            _, jpeg = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, config.MJPEG_QUALITY]
            )

            elapsed = time.time() - loop_start
            fps = 1.0 / elapsed if elapsed > 0 else 0
            _fps_history.append(fps)
            avg_fps = sum(_fps_history) / len(_fps_history)

            now = time.time()
            with _lock:
                _frame_bytes = jpeg.tobytes()
                if label and (now - last_event_time) >= config.WS_EVENT_RATE:
                    _latest_event = {
                        "gesture": label,
                        "confidence": round(confidence, 3),
                        "command": COMMAND_MAP.get(label, ""),
                        "fps": round(avg_fps, 1),
                        "inference_ms": round(inference_ms, 1),
                        "timestamp": time.strftime("%H:%M:%S"),
                    }
                    last_event_time = now
    finally:
        cap.release()
        detector.close()


def start():
    """Start the background camera thread. Call once at app startup."""
    t = threading.Thread(target=_camera_loop, daemon=True)
    t.start()


def get_frame() -> bytes:
    """Return the latest JPEG frame bytes."""
    with _lock:
        return _frame_bytes


def get_latest_event() -> dict:
    """Return the latest gesture event dict (or empty dict if none yet)."""
    with _lock:
        return dict(_latest_event)
```

**Step: Commit**

```bash
git add web/stream.py
git commit -m "feat: add web/stream.py — background camera thread for MJPEG and WebSocket"
```

---

## Task 10: Build web/app.py — Flask routes and WebSocket

**Files:**
- Create: `web/app.py`

```python
# web/app.py
"""
Flask web dashboard for sign language recognition.

Routes:
    GET  /            → HTML dashboard
    GET  /video_feed  → MJPEG stream
    WS   /ws          → WebSocket gesture events (JSON)

Run:
    python -m web.app
"""

import json
import time
from flask import Flask, Response, render_template
from flask_sock import Sock

import config
from web import stream as cam

app = Flask(__name__, template_folder="templates", static_folder="static")
sock = Sock(app)


def _mjpeg_generator():
    while True:
        frame = cam.get_frame()
        if frame:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
        time.sleep(0.03)  # ~30 FPS ceiling for the HTTP stream


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@sock.route("/ws")
def websocket(ws):
    last_sent = {}
    while True:
        event = cam.get_latest_event()
        if event and event != last_sent:
            ws.send(json.dumps(event))
            last_sent = event
        time.sleep(config.WS_EVENT_RATE)


if __name__ == "__main__":
    cam.start()
    app.run(host=config.WEB_HOST, port=config.WEB_PORT, debug=False)
```

**Step: Commit**

```bash
git add web/app.py
git commit -m "feat: add web/app.py — Flask MJPEG + WebSocket routes"
```

---

## Task 11: Build web frontend (HTML + CSS + JS)

**Files:**
- Create: `web/templates/index.html`
- Create: `web/static/css/style.css`
- Create: `web/static/js/dashboard.js`

**Step 1: Create web/templates/index.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Sign Language Recognition</title>
  <link rel="stylesheet" href="/static/css/style.css" />
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
</head>
<body>
  <header>
    <h1>Sign Language Recognition Dashboard</h1>
  </header>

  <main>
    <!-- Left: Live feed -->
    <section class="panel feed-panel">
      <h2>Live Feed</h2>
      <img id="video" src="/video_feed" alt="Live camera feed" />
    </section>

    <!-- Right: Stats -->
    <section class="panel stats-panel">

      <!-- Gesture + command -->
      <div class="gesture-display">
        <div id="gesture-label">--</div>
        <div id="gesture-command" class="command">--</div>
      </div>

      <!-- Confidence gauge -->
      <div class="confidence-block">
        <label>Confidence</label>
        <div class="gauge-track">
          <div id="gauge-fill" class="gauge-fill" style="width:0%"></div>
        </div>
        <span id="confidence-value">0.00</span>
      </div>

      <!-- FPS / Inference chart -->
      <div class="chart-block">
        <canvas id="metrics-chart"></canvas>
      </div>

      <!-- Gesture history -->
      <div class="history-block">
        <h3>History</h3>
        <table>
          <thead><tr><th>Time</th><th>Gesture</th><th>Command</th><th>Conf</th></tr></thead>
          <tbody id="history-body"></tbody>
        </table>
      </div>

    </section>
  </main>

  <script src="/static/js/dashboard.js"></script>
</body>
</html>
```

**Step 2: Create web/static/css/style.css**

```css
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: 'Segoe UI', system-ui, sans-serif;
  background: #0d1117;
  color: #e6edf3;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

header {
  padding: 1rem 2rem;
  background: #161b22;
  border-bottom: 1px solid #30363d;
}

header h1 { font-size: 1.2rem; font-weight: 600; color: #58a6ff; }

main {
  display: flex;
  flex: 1;
  gap: 1rem;
  padding: 1rem;
}

.panel {
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 8px;
  padding: 1rem;
}

.feed-panel {
  flex: 0 0 640px;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.feed-panel h2 { font-size: 0.85rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; }

#video {
  width: 100%;
  border-radius: 6px;
  background: #000;
}

.stats-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 1rem;
  overflow-y: auto;
}

/* Gesture display */
.gesture-display {
  text-align: center;
  padding: 1rem;
  background: #0d1117;
  border-radius: 6px;
}

#gesture-label {
  font-size: 4rem;
  font-weight: 700;
  color: #58a6ff;
  line-height: 1;
}

.command {
  font-size: 1rem;
  color: #8b949e;
  margin-top: 0.25rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

/* Confidence gauge */
.confidence-block {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  font-size: 0.85rem;
  color: #8b949e;
}

.gauge-track {
  flex: 1;
  height: 10px;
  background: #30363d;
  border-radius: 999px;
  overflow: hidden;
}

.gauge-fill {
  height: 100%;
  border-radius: 999px;
  transition: width 0.2s ease, background 0.2s ease;
  background: #3fb950;
}

.gauge-fill.med  { background: #e3b341; }
.gauge-fill.low  { background: #f85149; }

#confidence-value { min-width: 2.5rem; text-align: right; color: #e6edf3; }

/* Chart */
.chart-block {
  background: #0d1117;
  border-radius: 6px;
  padding: 0.75rem;
  height: 160px;
}

#metrics-chart { width: 100% !important; height: 100% !important; }

/* History table */
.history-block h3 {
  font-size: 0.8rem;
  color: #8b949e;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 0.5rem;
}

table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }

th {
  text-align: left;
  padding: 0.3rem 0.5rem;
  color: #8b949e;
  border-bottom: 1px solid #30363d;
}

td { padding: 0.3rem 0.5rem; border-bottom: 1px solid #21262d; }

tbody tr:hover { background: #1c2128; }
```

**Step 3: Create web/static/js/dashboard.js**

```javascript
// web/static/js/dashboard.js
const HISTORY_MAX = 50;

// ── Chart setup ────────────────────────────────────────────────────────────────
const ctx = document.getElementById("metrics-chart").getContext("2d");
const metricsChart = new Chart(ctx, {
  type: "line",
  data: {
    labels: Array(60).fill(""),
    datasets: [
      {
        label: "FPS",
        data: Array(60).fill(null),
        borderColor: "#58a6ff",
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.3,
        yAxisID: "yFps",
      },
      {
        label: "Inference ms",
        data: Array(60).fill(null),
        borderColor: "#3fb950",
        borderWidth: 1.5,
        pointRadius: 0,
        tension: 0.3,
        yAxisID: "yMs",
      },
    ],
  },
  options: {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { labels: { color: "#8b949e", font: { size: 11 } } } },
    scales: {
      x: { display: false },
      yFps: {
        position: "left",
        min: 0,
        ticks: { color: "#58a6ff", font: { size: 10 } },
        grid: { color: "#21262d" },
      },
      yMs: {
        position: "right",
        min: 0,
        ticks: { color: "#3fb950", font: { size: 10 } },
        grid: { drawOnChartArea: false },
      },
    },
  },
});

function pushMetric(fps, inferenceMs) {
  metricsChart.data.datasets[0].data.shift();
  metricsChart.data.datasets[0].data.push(fps);
  metricsChart.data.datasets[1].data.shift();
  metricsChart.data.datasets[1].data.push(inferenceMs);
  metricsChart.update("none");
}

// ── Gauge ──────────────────────────────────────────────────────────────────────
const gaugeFill = document.getElementById("gauge-fill");
const confValue = document.getElementById("confidence-value");

function updateGauge(conf) {
  const pct = Math.round(conf * 100);
  gaugeFill.style.width = pct + "%";
  confValue.textContent = conf.toFixed(2);
  gaugeFill.className = "gauge-fill";
  if (conf < 0.6) gaugeFill.classList.add("low");
  else if (conf < 0.8) gaugeFill.classList.add("med");
}

// ── History ────────────────────────────────────────────────────────────────────
const historyBody = document.getElementById("history-body");

function addHistoryRow(ts, gesture, command, conf) {
  const tr = document.createElement("tr");
  tr.innerHTML = `<td>${ts}</td><td><strong>${gesture}</strong></td><td>${command}</td><td>${conf.toFixed(2)}</td>`;
  historyBody.insertBefore(tr, historyBody.firstChild);
  while (historyBody.rows.length > HISTORY_MAX) {
    historyBody.deleteRow(historyBody.rows.length - 1);
  }
}

// ── WebSocket ──────────────────────────────────────────────────────────────────
const gestureLabel = document.getElementById("gesture-label");
const gestureCommand = document.getElementById("gesture-command");

function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);

  ws.onmessage = (evt) => {
    const data = JSON.parse(evt.data);
    gestureLabel.textContent = data.gesture || "--";
    gestureCommand.textContent = data.command || "--";
    updateGauge(data.confidence || 0);
    pushMetric(data.fps || 0, data.inference_ms || 0);
    addHistoryRow(data.timestamp, data.gesture, data.command, data.confidence);
  };

  ws.onclose = () => setTimeout(connect, 2000);  // auto-reconnect
  ws.onerror = () => ws.close();
}

connect();
```

**Step: Commit**

```bash
git add web/templates/index.html web/static/css/style.css web/static/js/dashboard.js
git commit -m "feat: add web dashboard — MJPEG feed + confidence gauge + Chart.js metrics + history log"
```

---

## Task 12: Update requirements.txt

**Files:**
- Modify: `requirements.txt`

**Step 1: Update requirements.txt**

```
opencv-python
mediapipe
scikit-learn
numpy
joblib
flask
flask-sock
```

**Step 2: Install new dependencies**

```bash
pip install flask flask-sock
```

**Step 3: Verify**

```bash
python -c "import flask, flask_sock; print('OK')"
```

**Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: add flask and flask-sock to requirements"
```

---

## Task 13: Run the full test suite

**Step 1: Run all unit tests**

```bash
python -m pytest tests/ -v
```

Expected: all tests pass (test_gestures, test_utils, test_predictor)

**Step 2: Smoke-test imports**

```bash
python -c "
import config
from core.gestures import GESTURE_LABELS, GESTURE_PIN_MAP, COMMAND_MAP
from core.utils import landmarks_to_feature
from core.predictor import EWMAPredictor
from hardware.platform_io import detect_platform
from hardware.gpio_controller import GpioController
from training.collect_data import ensure_csv_exists
from training.train_model import augment
from web.app import app
print('All imports OK')
print('Gestures:', GESTURE_LABELS)
"
```

Expected: `All imports OK` and the 11 gesture labels listed.

---

## Task 14: Re-collect data and retrain (manual step)

> **Important:** The feature dimension changed from 63 → 66 (palm orientation added). The old `dataset/gestures.csv` is incompatible. Delete it and collect fresh data.

**Step 1: Delete old dataset**

```bash
rm dataset/gestures.csv
```

**Step 2: Collect ~100 samples per gesture (repeat for all 11)**

```bash
python -m training.collect_data A
python -m training.collect_data B
python -m training.collect_data C
python -m training.collect_data D
python -m training.collect_data L
python -m training.collect_data Y
python -m training.collect_data PEACE
python -m training.collect_data OK
python -m training.collect_data STOP
python -m training.collect_data HELP
python -m training.collect_data THANK_YOU
```

Hold each gesture in front of the webcam, press `s` to save, `q` when done. Aim for ~100 per gesture.

**Step 3: Verify dataset shape**

```bash
python -c "import numpy as np; d=np.genfromtxt('dataset/gestures.csv',delimiter=',',dtype=str,skip_header=1); print('Shape:', d.shape, '  Expected cols:', 67)"
```

Expected: shape `(N, 67)` where N = total samples and 67 = 1 label + 66 features.

**Step 4: Train**

```bash
python -m training.train_model
```

Expected: GridSearchCV output, then `Saved model to 'models/model.pkl'`

**Step 5: Commit dataset + model**

```bash
git add dataset/gestures.csv models/model.pkl
git commit -m "data: add 11-gesture dataset and retrained model"
```

---

## Task 15: End-to-end test

**Step 1: Run PC recognition (OpenCV window)**

```bash
python -m inference.run_recognition
```

Expected: webcam window opens, gestures recognized with EWMA-smoothed confidence.

**Step 2: Run web dashboard**

```bash
python -m web.app
```

Open browser at `http://localhost:5000`. Expected: live feed on left, gesture label + confidence gauge + Chart.js metrics chart + history log on right.

**Step 3: Test headless mode**

```bash
python -m inference.run_recognition --headless
```

Expected: terminal output with gesture labels and confidence values.

**Step 4: Commit if any final fixes were needed**

```bash
git add -A
git commit -m "fix: end-to-end verification fixes"
```

---

## Final structure check

After all tasks complete, run:

```bash
find . -name "*.py" | grep -v venv | grep -v __pycache__ | sort
```

Expected output:
```
./config.py
./core/__init__.py
./core/gestures.py
./core/predictor.py
./core/utils.py
./hardware/__init__.py
./hardware/gpio_controller.py
./hardware/platform_io.py
./inference/__init__.py
./inference/edge_inference.py
./inference/run_recognition.py
./tests/__init__.py
./tests/test_gestures.py
./tests/test_predictor.py
./tests/test_utils.py
./training/__init__.py
./training/collect_data.py
./training/train_model.py
./web/__init__.py
./web/app.py
./web/stream.py
```
