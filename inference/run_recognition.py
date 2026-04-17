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

    # Auto-download missing assets
    from setup_helper import ensure_assets
    if not ensure_assets():
        print("[Error] Required assets missing. Run 'python setup.py' first.")
        return

    hw = HardwareIO(platform)
    hw.boot()

    print(f"Loading model from '{config.MODEL_PATH}'...")
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
