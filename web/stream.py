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
    predictor = EWMAPredictor(
        payload["model"], payload["label_map"],
        threshold=config.CONFIDENCE_THRESHOLD
    )

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
