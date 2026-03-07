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
CONFIDENCE_THRESHOLD = 0.60
EWMA_ALPHA           = 0.3

# ── Training ──────────────────────────────────────────────────────────────────
TEST_SIZE        = 0.2
RANDOM_STATE     = 42
JITTER_SIGMA     = 0.01
JITTER_COPIES    = 2
SVC_PARAM_GRID   = {
    "C":     [1, 10, 100],
    "gamma": ["scale", "auto", 0.01, 0.001],
}
CV_FOLDS = 5

# ── Camera ────────────────────────────────────────────────────────────────────
CAM_INDEX_PC  = 0
CAM_BACKEND   = "dshow"
RES_PC        = (640, 480)
RES_EMBEDDED  = (320, 240)
FPS_CAP_PC    = None
FPS_CAP_EMB   = 10

# ── Web dashboard ─────────────────────────────────────────────────────────────
WEB_HOST        = "0.0.0.0"
WEB_PORT        = 5000
MJPEG_QUALITY   = 70
WS_EVENT_RATE   = 0.1
HISTORY_MAX     = 50

# ── Edge inference ────────────────────────────────────────────────────────────
EDGE_FPS_CAP    = 15
EDGE_SKIP_FRAMES = 2
